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

Two distinct roles (first-round finding MEDIUM-2, preserved here):

- `recheck_t2_reuse_conditions` -- a **private pure evaluator**. It checks
  §3.3/§9's preservation conditions against a plain ``safe_metadata``
  mapping and performs no I/O of its own. It exists so fake/synthetic
  tests can exercise the pass/BLOCK logic directly without needing a real
  Git checkout -- it is not, by itself, a safe *production* trust root,
  because nothing stops a caller from fabricating a favorable mapping.
- `resolve_t2_reuse_safe_metadata_from_verified_head` /
  `resolve_and_recheck_t2_reuse_conditions` -- the **production
  resolver**. It derives the same safe-metadata fields by reading the
  future `V8B_T2_REUSE_CONDITIONS_RECHECK.json` artifact from a
  **verified Git object** (never a caller-supplied path or mapping).
  That artifact does not exist in this repository yet -- the real
  post-Layer-B recheck has not been performed -- so this resolver, and
  therefore `T2` production acquisition, fails closed today by
  construction. This implementation does not create that artifact.

Neither path reads, accepts, or exposes a `T2` ticker identity, and
neither offers a `T_spare`/`T3` fallback -- this module defines no
alternate block-selection function of any kind. Any condition failing is
`V8B_T2_PRESERVATION_RECHECK_BLOCKED`.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from src.v8b_git_provenance import (
    V8BGitProvenanceBlocked,
    read_git_object_bytes,
    require_git_commit,
)
from src.v8b_production_provenance import EXPECTED_V8B_FROZEN_DESIGN_COMMIT

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


def recheck_t2_reuse_conditions(safe_metadata: Mapping[str, Any]) -> dict[str, Any]:
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


def resolve_t2_reuse_safe_metadata_from_verified_head(repository_root, verified_head: str) -> dict[str, Any]:
    """Production resolver: derive ``safe_metadata`` from the future,
    fresh **POST_FREEZE** `V8B_T2_REUSE_CONDITIONS_RECHECK.json` artifact
    (§12.4), read from a **verified Git object** -- never a caller-supplied
    mapping or path, and never the §12.2 pre-freeze document. This artifact
    does not exist in this repository yet, so this fails closed today.
    """
    commit = require_git_commit(verified_head, "POST_FREEZE_RECHECK_HEAD_INVALID")
    try:
        raw = read_git_object_bytes(repository_root, commit, POST_FREEZE_RECHECK_GIT_PATH)
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

    return {field: artifact[field] for field in REQUIRED_SAFE_METADATA_FIELDS}


def resolve_and_recheck_t2_reuse_conditions(repository_root, verified_head: str) -> dict[str, Any]:
    """The production entrypoint: resolve safe metadata from a verified Git
    object, then apply the same pure evaluator fake tests use. Production
    code must call this, never construct or accept a ``safe_metadata``
    mapping directly."""
    safe_metadata = resolve_t2_reuse_safe_metadata_from_verified_head(repository_root, verified_head)
    return recheck_t2_reuse_conditions(safe_metadata)


__all__ = [
    "POST_FREEZE_RECHECK_FIELDS",
    "POST_FREEZE_RECHECK_GATE",
    "POST_FREEZE_RECHECK_GIT_PATH",
    "POST_FREEZE_RECHECK_SCHEMA_VERSION",
    "POST_FREEZE_RECHECK_STAGE",
    "REQUIRED_SAFE_METADATA_FIELDS",
    "V8BT2PreservationRecheckBlocked",
    "recheck_t2_reuse_conditions",
    "resolve_and_recheck_t2_reuse_conditions",
    "resolve_t2_reuse_safe_metadata_from_verified_head",
]
