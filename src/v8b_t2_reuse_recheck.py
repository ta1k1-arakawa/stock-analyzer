"""`READ_ONLY_T2_REUSE_CONDITIONS_RECHECK` (§12.4, §9, §3.3).

This module separates two distinct roles (first-round finding MEDIUM-2):

- `recheck_t2_reuse_conditions` -- a **private pure evaluator**. It checks
  §3.3/§9's preservation conditions against a plain ``safe_metadata``
  mapping and performs no I/O of its own. It exists so fake/synthetic
  tests can exercise the pass/BLOCK logic directly without needing a real
  Git checkout -- it is not, by itself, a safe *production* trust root,
  because nothing stops a caller from fabricating a favorable mapping.
- `resolve_t2_reuse_safe_metadata_from_verified_head` /
  `resolve_and_recheck_t2_reuse_conditions` -- the **production
  resolver**. It derives the same safe-metadata fields by reading
  `V8B_TSPARE_T2_T3_PRESERVATION_RECHECK.md` from a **verified Git
  object** (never a caller-supplied path or mapping), pinned to its exact
  expected blob, and only then calls the pure evaluator. Production code
  must call the resolver, never construct a `safe_metadata` mapping itself.

Neither path reads, accepts, or exposes a `T2` ticker identity, and
neither offers a `T_spare`/`T3` fallback -- this module defines no
alternate block-selection function of any kind. Any condition failing is
`V8B_T2_PRESERVATION_RECHECK_BLOCKED`.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.v8b_git_provenance import (
    V8BGitProvenanceBlocked,
    read_git_object_bytes,
    require_git_commit,
    resolve_git_blob,
)
from src.v8b_production_provenance import EXPECTED_V8B_FROZEN_DESIGN_COMMIT

PRESERVATION_RECHECK_GIT_PATH = "V8B_TSPARE_T2_T3_PRESERVATION_RECHECK.md"
EXPECTED_PRESERVATION_RECHECK_BLOB = "f46e9fd295fd2a2843e9e6edd9c833922e5aad44"

# Section B's field names in the committed doc do not all match this
# module's own safe_metadata schema (e.g. the doc's
# "v8b_f1_c1_production_policy_already_fixed_at_reviewed_design_sha" versus
# this schema's "t2_v8b_f1_c1_policy_fixed") -- this mapping is the single
# place that translation happens.
_DOC_FIELD_TO_SAFE_METADATA_FIELD = {
    "t2_acquired": "t2_acquired",
    "t2_opened": "t2_opened",
    "t2_ticker_identities_exposed_to_human_public_research_loop": (
        "t2_ticker_identities_exposed_to_human_public_research_loop"
    ),
    "t2_market_data_raw_ohlcv_feature_outcome_research_exposure": (
        "t2_market_data_raw_ohlcv_feature_outcome_research_exposure"
    ),
    "t2_universe_definition_unchanged": "t2_universe_definition_unchanged",
    "t2_partition_algorithm_unchanged": "t2_partition_algorithm_unchanged",
    "v8b_f1_c1_production_policy_already_fixed_at_reviewed_design_sha": "t2_v8b_f1_c1_policy_fixed",
}

_SECTION_B_START = "## B. `T2` recheck"
_SECTION_B_END = "## C. `T3` recheck"
_HEADER_LINE_PATTERN = re.compile(r"^(result|reviewed_design_commit)=(\S+)\s*$", re.MULTILINE)
_FIELD_LINE_PATTERN = re.compile(r"^([a-z0-9_]+)=(true|false)\s*--\s*PASS\s*$", re.MULTILINE)


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


def resolve_t2_reuse_safe_metadata_from_verified_head(repository_root, verified_head: str) -> dict[str, Any]:
    """Production resolver: derive ``safe_metadata`` from the committed,
    fixed `V8B_TSPARE_T2_T3_PRESERVATION_RECHECK.md` audit artifact, read
    from a **verified Git object** and pinned to its exact expected blob --
    never from a caller-supplied mapping or path.
    """
    commit = require_git_commit(verified_head, "PRESERVATION_RECHECK_HEAD_INVALID")
    try:
        blob = resolve_git_blob(repository_root, commit, PRESERVATION_RECHECK_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap(error, "V8B_T2_PRESERVATION_RECHECK_DOC_MISSING") from error
    if blob != EXPECTED_PRESERVATION_RECHECK_BLOB:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_DOC_MUTATED")
    try:
        raw = read_git_object_bytes(repository_root, commit, PRESERVATION_RECHECK_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap(error, "V8B_T2_PRESERVATION_RECHECK_DOC_MISSING") from error

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_DOC_INVALID_ENCODING") from error

    header = dict(_HEADER_LINE_PATTERN.findall(text))
    if header.get("result") != "PASS":
        raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_BLOCKED:DOC_RESULT_NOT_PASS")
    if header.get("reviewed_design_commit") != EXPECTED_V8B_FROZEN_DESIGN_COMMIT:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_BLOCKED:DOC_DESIGN_COMMIT_MISMATCH")

    start = text.find(_SECTION_B_START)
    end = text.find(_SECTION_B_END)
    if start == -1 or end == -1 or end <= start:
        raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_BLOCKED:DOC_SECTION_B_MISSING")
    section_b = text[start:end]

    doc_fields = dict(_FIELD_LINE_PATTERN.findall(section_b))
    safe_metadata: dict[str, Any] = {}
    for doc_field, safe_field in _DOC_FIELD_TO_SAFE_METADATA_FIELD.items():
        if doc_field not in doc_fields:
            raise V8BT2PreservationRecheckBlocked("V8B_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_SAFE_METADATA")
        safe_metadata[safe_field] = doc_fields[doc_field] == "true"
    return safe_metadata


def resolve_and_recheck_t2_reuse_conditions(repository_root, verified_head: str) -> dict[str, Any]:
    """The production entrypoint: resolve safe metadata from a verified Git
    object, then apply the same pure evaluator fake tests use. Production
    code must call this, never construct or accept a ``safe_metadata``
    mapping directly."""
    safe_metadata = resolve_t2_reuse_safe_metadata_from_verified_head(repository_root, verified_head)
    return recheck_t2_reuse_conditions(safe_metadata)


__all__ = [
    "EXPECTED_PRESERVATION_RECHECK_BLOB",
    "PRESERVATION_RECHECK_GIT_PATH",
    "REQUIRED_SAFE_METADATA_FIELDS",
    "V8BT2PreservationRecheckBlocked",
    "recheck_t2_reuse_conditions",
    "resolve_and_recheck_t2_reuse_conditions",
    "resolve_t2_reuse_safe_metadata_from_verified_head",
]
