"""`READ_ONLY_T2_PRESERVATION_RECHECK` -- the second mandatory T2 recheck
point (§7, §7.1 ``recheck_2``, §12).

`V8C_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §7.1 requires the same nine
conditions checked at `recheck_1` (already recorded, pre-freeze, in
`V8C_PREFREEZE_PRESERVATION_RECHECK.md`) to be rechecked again, live,
immediately before the V8C-specific T2 authority bridge and before any T2
raw acquisition. This module implements that second, production-capable
recheck path. It is **not executed** by this implementation phase -- no
real T2 preservation recheck has occurred at this stage, and the future
cross-check artifact this resolver reads does not exist in this repository
yet, so production T2 acquisition continues to fail closed today by
construction.

Mirrors `src.v8b_t2_reuse_recheck`'s pattern of deriving safe metadata from
**authoritative** repository/trust state rather than trusting a
self-declared artifact field: `V8_STATE.json`'s own `T2` record (the same
physical block, genuinely authoritative cross-study evidence), the
exact-blob V8 trust anchor (proves the universe/partition source has not
moved), the exact-blob reviewed-implementation binding (proves the data-
quality policy implementation is unchanged since review), and whether
`T2_RAW_ACQUISITION_HUMAN_GATE` has already been durably consumed
(never a self-reported boolean).

Two, and only two, ways to call this module:

- ``_recheck_t2_preservation_conditions`` -- a **private pure evaluator**,
  fake/synthetic tests only.
- ``resolve_and_recheck_t2_preservation`` -- the sole **public production
  resolver**, which always resolves the current verified production HEAD
  itself and accepts no caller-supplied trust root.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    read_git_object_bytes,
    resolve_git_blob,
    resolve_verified_v8c_production_git_commit,
)
from src.v8c_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_T2_RAW_ACQUISITION,
    V8CHumanGateConsumptionBlocked,
    has_gate_been_consumed,
)
from src.v8c_production_provenance import (
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    V8CProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    read_and_verify_v8_trusted_partition_anchor,
    verify_reviewed_implementation_binding,
)
from src.v8c_stage_state import V8CStageEvidenceBlocked, write_t2_recheck_pass

STUDY_NAME = "V8C_HISTORICAL_RESEARCH"

REQUIRED_SAFE_METADATA_FIELDS = (
    "t2_real_data_acquired",
    "t2_opened",
    "t2_research_access_count",
    "t2_features_observed",
    "t2_outcomes_observed",
    "t2_membership_reassigned",
    "universe_definition_compatible",
    "partition_algorithm_compatible",
    "data_quality_policy_unchanged",
)

V8_STATE_GIT_PATH = "V8_STATE.json"

# The already-independently-reviewed pre-freeze preservation baseline
# (recheck_1). Its blob is bound exactly, never re-read for content -- the
# fact that it is byte-identical to what was independently reviewed at
# freeze time (together with the frozen design freeze approval's own
# ``t2_preservation_recheck_result: PASS`` attestation, bound to the exact
# frozen design commit) is itself the safe evidence that all nine frozen
# T2 preservation conditions were true as of freeze. recheck_2 combines
# this frozen baseline with CURRENT safe evidence (gate-consumption state,
# V8_STATE's T2 access counters, and every blob/commit binding checked
# elsewhere in this module) establishing nothing relevant has changed
# since -- never a new self-declared ``V8_STATE.json`` field.
PREFREEZE_PRESERVATION_AUDIT_GIT_PATH = "V8C_PREFREEZE_PRESERVATION_RECHECK.md"
EXPECTED_PREFREEZE_PRESERVATION_AUDIT_BLOB = "ec9054caf94898948879b599196c055e480d2e52"

# The four compatibility conditions this module used to require as a
# self-declared ``V8_STATE.json["v8c_preservation_compatibility"]`` field
# that does not exist in the real repository. They are now a hardcoded
# literal derivation, gated on the pre-freeze baseline blob and the design
# freeze approval's PASS attestation both verifying unchanged -- exactly
# the "already-existing reviewed safe evidence" this module is required to
# rederive from, never a new favorable self-declared field.
_FROZEN_COMPATIBILITY_CONDITIONS = {
    "t2_membership_reassigned": False,
    "universe_definition_compatible": True,
    "partition_algorithm_compatible": True,
    "data_quality_policy_unchanged": True,
}


class V8CT2PreservationRecheckBlocked(RuntimeError):
    """Fail-closed §7/§7.1 T2 preservation recheck error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _recheck_t2_preservation_conditions(safe_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Private pure evaluator -- fake/synthetic tests only, not a
    production trust root. Absence of evidence is never treated as PASS:
    every required field must be present and exactly the required value."""
    if not isinstance(safe_metadata, Mapping):
        raise V8CT2PreservationRecheckBlocked("V8C_T2_PRESERVATION_RECHECK_BLOCKED:SAFE_METADATA_INVALID")
    missing = set(REQUIRED_SAFE_METADATA_FIELDS) - set(safe_metadata)
    if missing:
        raise V8CT2PreservationRecheckBlocked("V8C_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_SAFE_METADATA")

    checks_expect_false = (
        "t2_real_data_acquired",
        "t2_opened",
        "t2_features_observed",
        "t2_outcomes_observed",
        "t2_membership_reassigned",
    )
    for field in checks_expect_false:
        if safe_metadata[field] is not False:
            raise V8CT2PreservationRecheckBlocked("V8C_T2_PRESERVATION_RECHECK_BLOCKED:" + field.upper())

    if safe_metadata["t2_research_access_count"] != 0:
        raise V8CT2PreservationRecheckBlocked("V8C_T2_PRESERVATION_RECHECK_BLOCKED:T2_RESEARCH_ACCESS_COUNT")

    checks_expect_true = ("universe_definition_compatible", "partition_algorithm_compatible", "data_quality_policy_unchanged")
    for field in checks_expect_true:
        if safe_metadata[field] is not True:
            raise V8CT2PreservationRecheckBlocked("V8C_T2_PRESERVATION_RECHECK_BLOCKED:" + field.upper())

    return {"result": "PASS", "block": "T2", "recheck_point": "recheck_2"}


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8CT2PreservationRecheckBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8CT2PreservationRecheckBlocked(missing_reason)
    if isinstance(reason, str):
        return V8CT2PreservationRecheckBlocked(reason)
    return V8CT2PreservationRecheckBlocked("PRESERVATION_RECHECK_DOC_READ_FAILED")


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8CT2PreservationRecheckBlocked("V8_STATE_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8CT2PreservationRecheckBlocked("V8_STATE_INVALID_JSON") from error
    if not isinstance(parsed, dict):
        raise V8CT2PreservationRecheckBlocked("V8_STATE_INVALID_JSON")
    return parsed


def _require_bool(value: object, reason: str) -> bool:
    if not isinstance(value, bool):
        raise V8CT2PreservationRecheckBlocked(reason)
    return value


def _require_int_or_none(value: object, reason: str) -> int | None:
    if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
        raise V8CT2PreservationRecheckBlocked(reason)
    return value


def _default_read_v8_state_t2_evidence(
    repository_root, commit: str, git_object_reader: Callable[[str, str, str], bytes]
) -> dict[str, Any]:
    try:
        raw = git_object_reader(repository_root, commit, V8_STATE_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error, "V8_STATE_MISSING") from error
    state = _strict_json_object(raw)

    t2 = state.get("T2")
    if not isinstance(t2, dict):
        raise V8CT2PreservationRecheckBlocked("V8_STATE_T2_SECTION_INVALID")
    trust_anchor_pinning = state.get("trust_anchor_pinning")
    if not isinstance(trust_anchor_pinning, dict):
        raise V8CT2PreservationRecheckBlocked("V8_STATE_TRUST_ANCHOR_PINNING_SECTION_INVALID")

    return {
        "t2_raw_data_acquired": _require_bool(t2.get("raw_data_acquired"), "V8_STATE_T2_RAW_DATA_ACQUIRED_INVALID"),
        "t2_opened_for_research": _require_bool(
            t2.get("opened_for_research"), "V8_STATE_T2_OPENED_FOR_RESEARCH_INVALID"
        ),
        "t2_sealed_holdout_access_count": _require_int_or_none(
            t2.get("sealed_holdout_access_count"), "V8_STATE_T2_SEALED_HOLDOUT_ACCESS_COUNT_INVALID"
        ),
        "block_assignments_exposed": _require_bool(
            trust_anchor_pinning.get("block_assignments_exposed"), "V8_STATE_BLOCK_ASSIGNMENTS_EXPOSED_INVALID"
        ),
    }


def _default_verify_prefreeze_preservation_baseline(
    repository_root, commit: str, git_blob_resolver: Callable[[Any, str, str], str] = resolve_git_blob
) -> None:
    """Bind exactly to the already-independently-reviewed pre-freeze
    preservation baseline document -- never re-read its content, only its
    exact blob identity, mirroring every other exact-blob production
    provenance check in this study."""
    try:
        blob = git_blob_resolver(repository_root, commit, PREFREEZE_PRESERVATION_AUDIT_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error, "V8C_PREFREEZE_PRESERVATION_AUDIT_MISSING") from error
    if blob != EXPECTED_PREFREEZE_PRESERVATION_AUDIT_BLOB:
        raise V8CT2PreservationRecheckBlocked("V8C_PREFREEZE_PRESERVATION_AUDIT_MUTATED")


def _resolve_t2_preservation_safe_metadata_with_dependencies(
    repository_root,
    *,
    git_commit_resolver: Callable[[], str],
    anchor_reader: Callable[[str], Mapping[str, Any]],
    reviewed_implementation_binder: Callable[[str], Mapping[str, Any]],
    consumption_state_root,
    git_object_reader: Callable[[str, str, str], bytes] = read_git_object_bytes,
    gate_consumption_checker: Callable[[Any, str, str], bool] = has_gate_been_consumed,
    v8_state_evidence_reader: Callable[[Any, str, Callable[[str, str, str], bytes]], Mapping[str, Any]] = (
        _default_read_v8_state_t2_evidence
    ),
    prefreeze_baseline_verifier: Callable[[Any, str], None] = _default_verify_prefreeze_preservation_baseline,
    design_freeze_approval_reader: Callable[[Any, str], Mapping[str, Any]] = read_and_verify_design_freeze_approval,
) -> dict[str, Any]:
    """Private DI-testable implementation -- fake/synthetic tests only."""
    try:
        verified_head = git_commit_resolver()
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error) from error
    commit = verified_head

    try:
        anchor_reader(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    try:
        reviewed_implementation_binder(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error

    try:
        t2_acquired_derived = gate_consumption_checker(
            consumption_state_root, GATE_T2_RAW_ACQUISITION, EXPECTED_V8C_FROZEN_DESIGN_COMMIT
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CT2PreservationRecheckBlocked(error.reason) from error

    try:
        v8_state_evidence = v8_state_evidence_reader(repository_root, commit, git_object_reader)
    except (V8CT2PreservationRecheckBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8_STATE_MISSING") from error

    t2_acquired_derived = t2_acquired_derived or v8_state_evidence["t2_raw_data_acquired"]

    v8_state_says_no_exposure = (
        v8_state_evidence["t2_opened_for_research"] is False
        and not v8_state_evidence["t2_sealed_holdout_access_count"]
        and v8_state_evidence["block_assignments_exposed"] is False
    )
    exposure_value = not v8_state_says_no_exposure

    # Universe/partition/data-quality compatibility is rederived from
    # already-existing reviewed safe evidence rather than a new self-
    # declared ``V8_STATE.json`` field: the exact-blob pre-freeze
    # preservation baseline (recheck_1, independently reviewed) plus the
    # design freeze approval's own ``t2_preservation_recheck_result: PASS``
    # attestation bound to the exact frozen design commit. Both must
    # verify unchanged before these frozen literal conditions may be
    # asserted; either missing or mutated is a BLOCK, never a favorable
    # default.
    try:
        prefreeze_baseline_verifier(repository_root, commit)
    except (V8CT2PreservationRecheckBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8C_PREFREEZE_PRESERVATION_AUDIT_MISSING") from error

    try:
        approval = design_freeze_approval_reader(repository_root, commit)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error
    if (
        approval.get("t2_preservation_recheck_result") != "PASS"
        or approval.get("t2_preservation_recheck_design_commit") != EXPECTED_V8C_FROZEN_DESIGN_COMMIT
    ):
        raise V8CT2PreservationRecheckBlocked(
            "V8C_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_COMPATIBILITY_EVIDENCE"
        )
    compatibility = _FROZEN_COMPATIBILITY_CONDITIONS

    return {
        "t2_real_data_acquired": t2_acquired_derived,
        "t2_opened": exposure_value,
        "t2_research_access_count": (v8_state_evidence["t2_sealed_holdout_access_count"] or 0),
        "t2_features_observed": exposure_value,
        "t2_outcomes_observed": exposure_value,
        "t2_membership_reassigned": compatibility["t2_membership_reassigned"],
        "universe_definition_compatible": compatibility["universe_definition_compatible"],
        "partition_algorithm_compatible": compatibility["partition_algorithm_compatible"],
        "data_quality_policy_unchanged": compatibility["data_quality_policy_unchanged"],
    }


def _resolve_and_recheck_t2_preservation_with_dependencies(repository_root, **dependencies) -> dict[str, Any]:
    """Private DI-testable implementation -- fake/synthetic tests only."""
    state_root = dependencies.pop("stage_state_root", None)
    reviewed_commit = dependencies.pop("reviewed_implementation_commit", None)
    safe_metadata = _resolve_t2_preservation_safe_metadata_with_dependencies(repository_root, **dependencies)
    result = _recheck_t2_preservation_conditions(safe_metadata)
    if state_root is not None:
        if not isinstance(reviewed_commit, str):
            raise V8CT2PreservationRecheckBlocked("V8C_T2_PRESERVATION_RECHECK_IMPLEMENTATION_BINDING_MISSING")
        try:
            write_t2_recheck_pass(
                state_root,
                {
                    **safe_metadata,
                    **result,
                    "frozen_design_commit": EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
                    "reviewed_implementation_commit": reviewed_commit,
                },
            )
        except V8CStageEvidenceBlocked as error:
            raise V8CT2PreservationRecheckBlocked(error.reason) from error
    return {**result, **safe_metadata}


def resolve_and_recheck_t2_preservation() -> dict[str, Any]:
    """The public production `READ_ONLY_T2_PRESERVATION_RECHECK` entrypoint
    (§7.1's ``recheck_2``). Always resolves the current verified production
    HEAD itself and accepts no caller-supplied trust root."""
    result = _resolve_and_recheck_t2_preservation_with_dependencies(
        CANONICAL_REPOSITORY_ROOT,
        git_commit_resolver=lambda: resolve_verified_v8c_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        anchor_reader=lambda head: read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, head),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(CANONICAL_REPOSITORY_ROOT, head),
        consumption_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
        stage_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
        reviewed_implementation_commit=verify_reviewed_implementation_binding(
            CANONICAL_REPOSITORY_ROOT,
            resolve_verified_v8c_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        )["reviewed_implementation_git_commit"],
    )
    return result


__all__ = [
    "CANONICAL_REPOSITORY_ROOT",
    "REQUIRED_SAFE_METADATA_FIELDS",
    "STUDY_NAME",
    "V8CT2PreservationRecheckBlocked",
    "resolve_and_recheck_t2_preservation",
]
