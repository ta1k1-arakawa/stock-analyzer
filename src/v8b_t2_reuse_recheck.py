"""`READ_ONLY_T2_REUSE_CONDITIONS_RECHECK` (§12.4, §9, §3.3).

**Round-2 finding HIGH-2 correction.** This module previously (incorrectly)
derived its production evidence from `V8B_TSPARE_T2_T3_PRESERVATION_
RECHECK.md`. That document's own header records
`gate=READ_ONLY_TSPARE_T2_T3_PRESERVATION_RECHECK (...§12.2)` -- it is the
**pre-freeze** evidence for the *different* §12.2 gate that already ran
before `V8B_DESIGN_FINALIZED`/`HUMAN_DESIGN_FREEZE`. It cannot stand in for
§12.4's `READ_ONLY_T2_REUSE_CONDITIONS_RECHECK`, which the design draft's
own gate diagram positions **after** `Layer B` / `FROZEN FINAL CANDIDATE`
(§12's diagram; §12.4's rules), using **fresh, post-freeze** evidence. This
module no longer reads that §12.2 document for any production purpose.

**`FINAL_REPEAT_INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` finding
HIGH-3 correction.** Two further gaps in the production resolvers below
are now closed:

1. Neither public resolver accepts a caller-supplied ``verified_head``
   any more. Previously a caller could pass any string, including a stale
   or favorable head from an earlier check, rather than the currently
   verified production HEAD -- the resolver now always resolves the
   current verified production HEAD itself (via
   `src.v8b_git_provenance.resolve_verified_v8b_production_git_commit`)
   before deriving or checking anything.
2. Five of the seven §3.3/§9 conditions -- `t2_universe_definition_
   unchanged`, `t2_partition_algorithm_unchanged`, `t2_v8b_f1_c1_policy_
   fixed`, `t2_opened`, and the two research-exposure flags -- are no
   longer trusted purely because `V8B_T2_REUSE_CONDITIONS_RECHECK.json`
   *says* they are true. They are instead **derived** from authoritative
   safe repository/trust state this module can independently verify: the
   exact-frozen-blob immutable V8 trust anchor
   (`read_and_verify_v8_trusted_partition_anchor`, which proves the `T2`
   universe/partition source has not moved), the exact-blob reviewed-
   implementation binding (`verify_reviewed_implementation_binding`,
   which proves `src/v8_partition.py` and `src/v8b_historical_
   acquisition.py` -- where the F1_C1 policy is defined -- are unchanged
   since review), and a live check that no `open_for_*`/research-opening
   API exists anywhere in the bound `src.v8b_historical_acquisition`
   module. `t2_acquired` is derived from whether
   `T2_RAW_ACQUISITION_HUMAN_GATE`'s durable one-shot consumption receipt
   (`src.v8b_human_gate_consumption`) already exists -- not from a
   self-reported boolean. The evidence artifact's own claimed value for
   each of these fields is still required to *agree* with the derived
   value (a disagreement BLOCKs, `..._SELF_DECLARED_MISMATCH`), but the
   value actually used for the pass/BLOCK decision is always the derived
   one, never the artifact's bare claim. `layer_b_completed` and
   `frozen_final_candidate_established` remain read from the artifact --
   whether Layer B validation and the `FROZEN_FINAL_CANDIDATE` gate have
   actually occurred is a study-progress fact with no independently
   git-derivable proxy in this repository, so these two fields still
   require the future artifact's authored evidence.

Two distinct roles (first-round finding MEDIUM-2, tightened further in
round 3's repeat review):

- `_recheck_t2_reuse_conditions` -- a **private pure evaluator** (round-3
  repeat finding MEDIUM-2: not part of the production public surface --
  fake/synthetic tests import and call it directly as an internal
  helper). It checks §3.3/§9's preservation conditions against a plain
  ``safe_metadata`` mapping and performs no I/O of its own. It exists so
  fake/synthetic tests can exercise the pass/BLOCK logic directly without
  needing a real Git checkout -- it is not, by itself, a safe *production*
  trust root, because nothing stops a caller from fabricating a favorable
  mapping.
- `resolve_and_recheck_t2_reuse_conditions` -- the sole **public
  production resolver**. It resolves the current verified production HEAD
  itself, derives `safe_metadata` from authoritative repository/trust
  state as described above (cross-checked against the future
  `V8B_T2_REUSE_CONDITIONS_RECHECK.json` artifact, read from a **verified
  Git object**, read from the one fixed, non-overridable production
  repository root -- round-3 repeat finding HIGH-1: this public function
  accepts no ``repository_root`` or ``verified_head`` parameter; a private
  DI-testable variant carries injectable dependencies for fake/synthetic
  tests only) -- never a caller-supplied path, mapping, or head. That
  artifact does not exist in this repository yet -- the real post-Layer-B
  recheck has not been performed -- so this resolver, and therefore `T2`
  production acquisition, fails closed today by construction. This
  implementation does not create that artifact.

Neither path reads, accepts, or exposes a `T2` ticker identity, and
neither offers a `T_spare`/`T3` fallback -- this module defines no
alternate block-selection function of any kind. Any condition failing is
`V8B_T2_PRESERVATION_RECHECK_BLOCKED`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v8b_git_provenance import (
    V8BGitProvenanceBlocked,
    read_git_object_bytes,
    resolve_verified_v8b_production_git_commit,
)
from src.v8b_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_T2_RAW_ACQUISITION,
    V8BHumanGateConsumptionBlocked,
    has_gate_been_consumed,
)
from src.v8b_production_provenance import (
    EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
    V8BProductionProvenanceBlocked,
    read_and_verify_v8_trusted_partition_anchor,
    verify_reviewed_implementation_binding,
)

# The one fixed, non-overridable production repository root -- round-3
# repeat finding HIGH-1: no public function in this module accepts a
# caller-supplied repository_root as its trust root.
CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# The future §12.4 POST_FREEZE evidence artifact. Does not exist in this
# repository -- CREATE_V8B_TRUSTED_ALLOCATION_PIN-style future work, not
# performed by this implementation phase.
POST_FREEZE_RECHECK_GIT_PATH = "V8B_T2_REUSE_CONDITIONS_RECHECK.json"
POST_FREEZE_RECHECK_SCHEMA_VERSION = "V8B_T2_REUSE_CONDITIONS_RECHECK_V1"
POST_FREEZE_RECHECK_GATE = "READ_ONLY_T2_REUSE_CONDITIONS_RECHECK"
POST_FREEZE_RECHECK_STAGE = "POST_FREEZE"

POST_FREEZE_RECHECK_FIELDS = (
    "schema_version",
    "study",
    "gate",
    "frozen_design_git_commit",
    "stage",
    "result",
    "layer_b_completed",
    "frozen_final_candidate_established",
    "t2_acquired",
    "t2_opened",
    "t2_ticker_identities_exposed_to_human_public_research_loop",
    "t2_market_data_raw_ohlcv_feature_outcome_research_exposure",
    "t2_universe_definition_unchanged",
    "t2_partition_algorithm_unchanged",
    "t2_v8b_f1_c1_policy_fixed",
)


class V8BT2PreservationRecheckBlocked(RuntimeError):
    """Fail-closed §9/§12.4 `T2` reuse-condition recheck error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


REQUIRED_SAFE_METADATA_FIELDS = (
    "t2_acquired",
    "t2_opened",
    "t2_ticker_identities_exposed_to_human_public_research_loop",
    "t2_market_data_raw_ohlcv_feature_outcome_research_exposure",
    "t2_universe_definition_unchanged",
    "t2_partition_algorithm_unchanged",
    "t2_v8b_f1_c1_policy_fixed",
)

# Fields whose derived (authoritative) truth is always False once this
# point in the function is reached without raising -- i.e. "no research-
# opening capability exists in the bound production module".
_RESEARCH_EXPOSURE_FIELDS = (
    "t2_opened",
    "t2_ticker_identities_exposed_to_human_public_research_loop",
    "t2_market_data_raw_ohlcv_feature_outcome_research_exposure",
)


def _recheck_t2_reuse_conditions(safe_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Private pure evaluator -- fake/synthetic tests only, not a production
    trust root (see module docstring). Verify §3.3/§9's T2 preservation
    conditions from a plain ``safe_metadata`` mapping the caller already
    holds. ``safe_metadata`` must supply every field in
    ``REQUIRED_SAFE_METADATA_FIELDS`` -- absence of evidence is never
    treated as an implicit PASS (mirroring §12.2's identical rule for the
    T_spare/T2/T3 preservation recheck).
    """
    if not isinstance(safe_metadata, Mapping):
        raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_BLOCKED:SAFE_METADATA_INVALID")
    missing = set(REQUIRED_SAFE_METADATA_FIELDS) - set(safe_metadata)
    if missing:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_SAFE_METADATA")

    checks_expect_false = (
        "t2_acquired",
        "t2_opened",
        "t2_ticker_identities_exposed_to_human_public_research_loop",
        "t2_market_data_raw_ohlcv_feature_outcome_research_exposure",
    )
    for field in checks_expect_false:
        if safe_metadata[field] is not False:
            raise V8BT2PreservationRecheckBlocked(
                "V8B_T2_PRESERVATION_RECHECK_BLOCKED:" + field.upper()
            )

    checks_expect_true = (
        "t2_universe_definition_unchanged",
        "t2_partition_algorithm_unchanged",
        "t2_v8b_f1_c1_policy_fixed",
    )
    for field in checks_expect_true:
        if safe_metadata[field] is not True:
            raise V8BT2PreservationRecheckBlocked(
                "V8B_T2_PRESERVATION_RECHECK_BLOCKED:" + field.upper()
            )

    return {"result": "PASS", "block": "T2"}


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8BT2PreservationRecheckBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8BT2PreservationRecheckBlocked(missing_reason)
    if isinstance(reason, str):
        return V8BT2PreservationRecheckBlocked(reason)
    return V8BT2PreservationRecheckBlocked("PRESERVATION_RECHECK_DOC_READ_FAILED")


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_INVALID_JSON") from error
    if not isinstance(parsed, dict):
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_INVALID_JSON")
    return parsed


def _default_no_research_opening_api_exists() -> bool:
    """Live evidence, not a self-declared claim: no `open_for_*`/research-
    opening symbol exists anywhere in the bound `src.v8b_historical_
    acquisition` production module (mirrors the module's own MEDIUM-3
    guarantee -- see `test_no_open_for_functions_defined_at_all`).
    Imported lazily to avoid a module-level import cycle (`src.v8b_
    historical_acquisition` itself imports this module's public resolver
    at module scope).
    """
    from src import v8b_historical_acquisition as _acquisition_module

    return not any(
        name.startswith("open_for") or name.startswith("open_t2")
        for name in dir(_acquisition_module)
    )


def _resolve_t2_reuse_safe_metadata_with_dependencies(
    repository_root,
    *,
    git_commit_resolver: Callable[[], str],
    anchor_reader: Callable[[str], Mapping[str, Any]],
    reviewed_implementation_binder: Callable[[str], Mapping[str, Any]],
    consumption_state_root,
    no_research_opening_api_exists: Callable[[], bool] = _default_no_research_opening_api_exists,
    git_object_reader: Callable[[str, str, str], bytes] = read_git_object_bytes,
    gate_consumption_checker: Callable[[Any, str, str], bool] = has_gate_been_consumed,
) -> dict[str, Any]:
    """Private DI-testable implementation -- fake/synthetic tests only, not
    a production API. Derives ``safe_metadata`` from authoritative
    repository/trust state (HIGH-3), cross-checked against the future,
    fresh **POST_FREEZE** `V8B_T2_REUSE_CONDITIONS_RECHECK.json` artifact
    (§12.4), read from a **verified Git object** this function resolves
    itself -- never a caller-supplied head, mapping, or path, and never
    the §12.2 pre-freeze document. This artifact does not exist in this
    repository yet, so this fails closed today.
    """
    try:
        verified_head = git_commit_resolver()
    except V8BGitProvenanceBlocked as error:
        raise _wrap(error) from error
    # ``git_commit_resolver``'s production implementation is
    # `resolve_verified_v8b_production_git_commit`, which already
    # guarantees a validated 40-hex commit -- mirrors the equivalent
    # trust boundary in every sibling resolver module.
    commit = verified_head

    # Authoritative derivation, not a self-declared claim (HIGH-3): the
    # immutable V8 trust anchor's exact frozen blob proves the T2
    # universe/parent partition source has not moved.
    try:
        anchor_reader(verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    # Authoritative derivation: the exact-blob reviewed-implementation
    # binding proves src/v8_partition.py (partition algorithm) and
    # src/v8b_historical_acquisition.py (F1_C1 policy) are unchanged since
    # INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW.
    try:
        reviewed_implementation_binder(verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error, "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error

    # Authoritative derivation: whether T2_RAW_ACQUISITION_HUMAN_GATE has
    # already been durably consumed (src.v8b_human_gate_consumption),
    # never a self-reported "t2_acquired" boolean.
    try:
        t2_acquired_derived = gate_consumption_checker(
            consumption_state_root, GATE_T2_RAW_ACQUISITION, EXPECTED_V8B_FROZEN_DESIGN_COMMIT
        )
    except V8BHumanGateConsumptionBlocked as error:
        raise V8BT2PreservationRecheckBlocked(error.reason) from error

    no_opening_capability = no_research_opening_api_exists()
    research_exposure_value = not no_opening_capability

    derived: dict[str, Any] = {
        "t2_acquired": t2_acquired_derived,
        "t2_universe_definition_unchanged": True,
        "t2_partition_algorithm_unchanged": True,
        "t2_v8b_f1_c1_policy_fixed": True,
    }
    for field in _RESEARCH_EXPOSURE_FIELDS:
        derived[field] = research_exposure_value

    try:
        raw = git_object_reader(repository_root, commit, POST_FREEZE_RECHECK_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap(error, "V8B_T2_REUSE_CONDITIONS_RECHECK_MISSING") from error

    artifact = _strict_json_object(raw)
    if set(artifact) != set(POST_FREEZE_RECHECK_FIELDS):
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_SCHEMA_INVALID")
    if artifact["schema_version"] != POST_FREEZE_RECHECK_SCHEMA_VERSION:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_SCHEMA_VERSION_MISMATCH")
    if artifact["study"] != "V8B_HISTORICAL_RESEARCH":
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_STUDY_MISMATCH")
    if artifact["gate"] != POST_FREEZE_RECHECK_GATE:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_GATE_MISMATCH")
    if artifact["frozen_design_git_commit"] != EXPECTED_V8B_FROZEN_DESIGN_COMMIT:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_DESIGN_COMMIT_MISMATCH")
    if artifact["stage"] != POST_FREEZE_RECHECK_STAGE:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_NOT_POST_FREEZE")
    if artifact["result"] != "PASS":
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_NOT_PASS")
    if artifact["layer_b_completed"] is not True:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_LAYER_B_NOT_COMPLETE")
    if artifact["frozen_final_candidate_established"] is not True:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_NO_FROZEN_FINAL_CANDIDATE")

    # HIGH-3: the artifact's own self-declared value for each derivable
    # field must AGREE with the independently derived value -- a
    # disagreement BLOCKs rather than silently preferring either source.
    # The value actually used below is always the derived one, never the
    # artifact's bare claim.
    for field in REQUIRED_SAFE_METADATA_FIELDS:
        if artifact[field] != derived[field]:
            raise V8BT2PreservationRecheckBlocked(
                "V8B_T2_REUSE_CONDITIONS_RECHECK_SELF_DECLARED_MISMATCH:" + field
            )

    return derived


def _resolve_and_recheck_t2_reuse_conditions_with_dependencies(repository_root, **dependencies) -> dict[str, Any]:
    """Private DI-testable implementation -- fake/synthetic tests only, not
    a production API. Resolve safe metadata from authoritative repository/
    trust state, then apply the same pure evaluator fake tests use
    directly."""
    safe_metadata = _resolve_t2_reuse_safe_metadata_with_dependencies(repository_root, **dependencies)
    return _recheck_t2_reuse_conditions(safe_metadata)


def resolve_and_recheck_t2_reuse_conditions() -> dict[str, Any]:
    """The public production entrypoint (FINAL_REPEAT finding HIGH-3: no
    ``repository_root`` or ``verified_head`` parameter -- this function
    always resolves the current verified production HEAD itself from
    ``CANONICAL_REPOSITORY_ROOT``, so a caller can never supply a stale or
    favorable head). Derives safe metadata from authoritative repository/
    trust state, then applies the same pure evaluator fake tests use.
    Production code must call this, never construct or accept a
    ``safe_metadata`` mapping directly."""
    return _resolve_and_recheck_t2_reuse_conditions_with_dependencies(
        CANONICAL_REPOSITORY_ROOT,
        git_commit_resolver=lambda: resolve_verified_v8b_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        anchor_reader=lambda head: read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, head),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        consumption_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
    )


__all__ = [
    "CANONICAL_REPOSITORY_ROOT",
    "POST_FREEZE_RECHECK_FIELDS",
    "POST_FREEZE_RECHECK_GATE",
    "POST_FREEZE_RECHECK_GIT_PATH",
    "POST_FREEZE_RECHECK_SCHEMA_VERSION",
    "POST_FREEZE_RECHECK_STAGE",
    "REQUIRED_SAFE_METADATA_FIELDS",
    "V8BT2PreservationRecheckBlocked",
    "resolve_and_recheck_t2_reuse_conditions",
]
