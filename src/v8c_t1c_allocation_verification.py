"""`READ_ONLY_T1C_ALLOCATION_ARTIFACT_VERIFICATION` (§2, §9.2, §12).

Independently verifies every invariant a concrete `T1C` allocation artifact
(`src/v8c_t1c_allocation.py`'s output) must satisfy -- not merely the
implementation code that produced it. Any single invariant failing is
`BLOCK`: no trust pin may be created and no acquisition may proceed.

Two, and only two, ways to call this module, mirroring
`src.v8b_allocation_verification`'s proven pattern:

- ``_verify_t1c_allocation_artifact`` -- the **private/pure invariant
  evaluator**. Fake/synthetic tests import and call it directly. It
  performs no I/O and accepts every trusted comparison input directly from
  the caller.
- ``resolve_and_verify_t1c_allocation_artifact`` -- the sole **public
  production boundary**. Resolves verified V8C Git HEAD from the one fixed,
  non-overridable production repository root; verifies the frozen design
  object, freeze approval, and reviewed-implementation binding; verifies
  the exact immutable V8 anchor; reads the private V8 partition manifest
  and the private `T1C` allocation artifact from caller-supplied paths;
  derives every block's ticker assignment internally from that one
  verified manifest; and only then invokes the pure evaluator above.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.v8_partition import STUDY_NAME as V8_STUDY_NAME, V8PartitionBlocked, read_partition_manifest
from src.v8c_t1c_allocation import (
    ALLOCATION_ARTIFACT_FIELDS,
    ARTIFACT_ROLE,
    LOGICAL_BLOCK,
    PARENT_STUDY_NAME,
    SCHEMA_VERSION as ALLOCATION_SCHEMA_VERSION,
    SELECTION_RULE_ID,
    SELECTION_RULE_TEXT,
    STUDY_NAME as ALLOCATION_STUDY_NAME,
    T1C_SLICE_END_EXCLUSIVE,
    T1C_SLICE_START_INCLUSIVE,
    T1C_TICKER_COUNT,
    V8CAllocationBlocked,
    ticker_list_sha256,
    verify_allocation_artifact_self_hash,
)
from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    resolve_verified_v8c_production_git_commit,
)
from src.v8c_production_provenance import (
    EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
    EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    V8CProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    read_and_verify_v8_trusted_partition_anchor,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8c_t1c_allocator import V8_DESIGN_COMMIT

_REQUIRED_BLOCK_KEYS = ("T0", "T1", "T2", "T3", "T_spare")


class V8CAllocationVerificationBlocked(RuntimeError):
    """Fail-closed §9.2 invariant verification error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _verify_t1c_allocation_artifact(
    artifact: Mapping[str, Any],
    *,
    parent_t_spare_tickers: Sequence[str],
    t0_tickers: Sequence[str],
    old_t1_tickers: Sequence[str],
    t2_tickers: Sequence[str],
    t3_tickers: Sequence[str],
    expected_parent_t_spare_ticker_list_sha256: str,
    expected_v8c_frozen_design_commit: str,
) -> dict[str, Any]:
    """Verify every invariant; return a safe public PASS summary.

    T1B/T1C disjointness is established from the trusted unique parent
    ordering and the frozen coordinate slices [0:300] and [300:600]. The
    V8B T1B artifact is deliberately never accepted or read.
    """
    try:
        verified = verify_allocation_artifact_self_hash(artifact)
    except V8CAllocationBlocked as error:
        raise V8CAllocationVerificationBlocked("ARTIFACT_SELF_HASH_INVALID:" + error.reason) from error

    if verified["schema_version"] != ALLOCATION_SCHEMA_VERSION:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_SCHEMA_VERSION_MISMATCH")
    if verified["study_name"] != ALLOCATION_STUDY_NAME:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_STUDY_NAME_MISMATCH")
    if verified["artifact_role"] != ARTIFACT_ROLE:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_ROLE_MISMATCH")
    if verified["logical_block"] != LOGICAL_BLOCK:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_LOGICAL_BLOCK_MISMATCH")
    if verified["parent_study"] != PARENT_STUDY_NAME:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_PARENT_STUDY_MISMATCH")
    if verified["selection_rule_id"] != SELECTION_RULE_ID:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_SELECTION_RULE_ID_MISMATCH")
    if verified["t1c_slice_start_inclusive"] != T1C_SLICE_START_INCLUSIVE:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_T1C_SLICE_START_MISMATCH")
    if verified["t1c_slice_end_exclusive"] != T1C_SLICE_END_EXCLUSIVE:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_T1C_SLICE_END_MISMATCH")

    parent = list(parent_t_spare_tickers)
    if len(set(parent)) != len(parent):
        raise V8CAllocationVerificationBlocked("PARENT_T_SPARE_DUPLICATE_TICKER")
    if verified["parent_t_spare_ticker_count"] != len(parent):
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_PARENT_T_SPARE_COUNT_MISMATCH")

    t1c = verified["t1c_tickers"]
    remaining = verified["remaining_t_spare_tickers"]

    # T1C = original_parent_T_spare[300:600]
    if t1c != parent[300:600]:
        raise V8CAllocationVerificationBlocked("T1C_NOT_EXACT_300_600_SLICE")
    if remaining != parent[600:]:
        raise V8CAllocationVerificationBlocked("REMAINING_T_SPARE_NOT_EXACT_TAIL_SLICE")

    if len(t1c) != T1C_TICKER_COUNT:
        raise V8CAllocationVerificationBlocked("T1C_SIZE_INVALID")
    if len(t1c) + len(remaining) + T1C_SLICE_START_INCLUSIVE != len(parent):
        raise V8CAllocationVerificationBlocked("T1C_REMAINING_ACCOUNTING_INVALID")

    t1c_set = set(t1c)
    remaining_set = set(remaining)
    parent_set = set(parent)

    if t1c_set & remaining_set:
        raise V8CAllocationVerificationBlocked("T1C_REMAINING_NOT_DISJOINT")
    if (t1c_set | remaining_set) != set(parent[T1C_SLICE_START_INCLUSIVE:]):
        raise V8CAllocationVerificationBlocked("T1C_REMAINING_UNION_MISMATCH")

    if t1c_set & set(t0_tickers):
        raise V8CAllocationVerificationBlocked("T1C_NOT_DISJOINT_FROM_T0")
    if t1c_set & set(old_t1_tickers):
        raise V8CAllocationVerificationBlocked("T1C_NOT_DISJOINT_FROM_OLD_T1")
    if verified["predecessor_burned_count"] != T1C_SLICE_START_INCLUSIVE:
        raise V8CAllocationVerificationBlocked("PREDECESSOR_BURNED_COUNT_INVALID")
    if t1c_set & set(t2_tickers):
        raise V8CAllocationVerificationBlocked("T1C_NOT_DISJOINT_FROM_T2")
    if t1c_set & set(t3_tickers):
        raise V8CAllocationVerificationBlocked("T1C_NOT_DISJOINT_FROM_T3")

    computed_parent_hash = ticker_list_sha256(parent)
    if computed_parent_hash != expected_parent_t_spare_ticker_list_sha256:
        raise V8CAllocationVerificationBlocked("PARENT_T_SPARE_HASH_MISMATCH_TRUSTED_ANCHOR")
    if verified["parent_t_spare_ticker_list_sha256"] != computed_parent_hash:
        raise V8CAllocationVerificationBlocked("PARENT_T_SPARE_HASH_MISMATCH_ARTIFACT")

    if ticker_list_sha256(t1c) != verified["t1c_ticker_list_sha256"]:
        raise V8CAllocationVerificationBlocked("T1C_TICKER_LIST_SHA_MISMATCH")
    if ticker_list_sha256(remaining) != verified["remaining_t_spare_ticker_list_sha256"]:
        raise V8CAllocationVerificationBlocked("REMAINING_T_SPARE_TICKER_LIST_SHA_MISMATCH")

    if verified["selection_rule_canonical_text_or_hash"] != SELECTION_RULE_TEXT:
        raise V8CAllocationVerificationBlocked("SELECTION_RULE_TEXT_MISMATCH")

    if verified["v8c_frozen_design_commit"] != expected_v8c_frozen_design_commit:
        raise V8CAllocationVerificationBlocked("V8C_FROZEN_DESIGN_COMMIT_MISMATCH")

    if set(verified) != set(ALLOCATION_ARTIFACT_FIELDS):
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_SCHEMA_INVALID")

    return {
        "result": "PASS",
        "logical_block": verified["logical_block"],
        "study_name": verified["study_name"],
        "parent_t_spare_ticker_count": verified["parent_t_spare_ticker_count"],
        "parent_t_spare_ticker_list_sha256": verified["parent_t_spare_ticker_list_sha256"],
        "t1c_ticker_count": verified["t1c_ticker_count"],
        "t1c_ticker_list_sha256": verified["t1c_ticker_list_sha256"],
        "predecessor_burned_count": verified["predecessor_burned_count"],
        "remaining_t_spare_ticker_count": verified["remaining_t_spare_ticker_count"],
        "remaining_t_spare_ticker_list_sha256": verified["remaining_t_spare_ticker_list_sha256"],
        "artifact_self_hash": verified["artifact_self_hash"],
        "v8c_frozen_design_commit": verified["v8c_frozen_design_commit"],
        "v8c_allocation_implementation_commit": verified["v8c_allocation_implementation_commit"],
        "parent_v8_partition_manifest_sha256": verified["parent_v8_partition_manifest_sha256"],
        "parent_v8_partition_implementation_commit": verified["parent_v8_partition_implementation_commit"],
        "no_membership_choice_based_on_ohlcv_or_data_quality_outcomes": True,
    }


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8CAllocationVerificationBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8CAllocationVerificationBlocked(missing_reason)
    if isinstance(reason, str):
        return V8CAllocationVerificationBlocked(reason)
    return V8CAllocationVerificationBlocked("PROVENANCE_CHECK_FAILED")


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8CAllocationVerificationBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8CAllocationVerificationBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8CAllocationVerificationBlocked(invalid_reason)
    return parsed


def _resolve_and_verify_t1c_allocation_artifact_with_repository_root(
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    *,
    repository_root,
) -> dict[str, Any]:
    """Private DI-testable implementation -- fake/synthetic tests only.
    The predecessor T1B is burned by coordinate, so this resolver never
    reads a V8B T1B artifact or accepts its path.
    """
    root = repository_root

    try:
        verified_head = resolve_verified_v8c_production_git_commit(root)
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error) from error

    try:
        verify_frozen_design_object(root)
        read_and_verify_design_freeze_approval(root, verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    try:
        review_binding = verify_reviewed_implementation_binding(root, verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    reviewed_commit = review_binding["reviewed_implementation_git_commit"]

    try:
        anchor = read_and_verify_v8_trusted_partition_anchor(root, verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    try:
        partition_manifest = read_partition_manifest(partition_manifest_path)
    except V8PartitionBlocked as error:
        raise V8CAllocationVerificationBlocked(error.reason) from error

    manifest_sha = partition_manifest["manifest_sha256"]
    if manifest_sha != anchor["authorized_partition_manifest_sha256"]:
        raise V8CAllocationVerificationBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    partition_implementation_commit = partition_manifest["partition_implementation_git_commit"]
    if partition_implementation_commit != anchor["authorized_partition_implementation_git_commit"]:
        raise V8CAllocationVerificationBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")
    if partition_manifest["study_name"] != V8_STUDY_NAME:
        raise V8CAllocationVerificationBlocked("PARTITION_MANIFEST_STUDY_NAME_MISMATCH")
    if partition_manifest["design_commit"] != V8_DESIGN_COMMIT:
        raise V8CAllocationVerificationBlocked("PARTITION_MANIFEST_DESIGN_COMMIT_MISMATCH")

    assignments = partition_manifest["block_assignments"]
    if not isinstance(assignments, Mapping) or set(_REQUIRED_BLOCK_KEYS) - set(assignments):
        raise V8CAllocationVerificationBlocked("PARTITION_BLOCK_ASSIGNMENT_MISSING")
    blocks: dict[str, list[str]] = {}
    for key in _REQUIRED_BLOCK_KEYS:
        value = assignments[key]
        if not isinstance(value, list):
            raise V8CAllocationVerificationBlocked("PARTITION_BLOCK_ASSIGNMENT_INVALID:" + key)
        blocks[key] = list(value)

    parent_tickers = blocks["T_spare"]
    if len(parent_tickers) != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
        raise V8CAllocationVerificationBlocked("PARENT_T_SPARE_TICKER_COUNT_INVALID")
    computed_parent_hash = ticker_list_sha256(parent_tickers)
    if computed_parent_hash != EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256:
        raise V8CAllocationVerificationBlocked("PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH")
    if computed_parent_hash != partition_manifest["t_spare_ticker_list_sha256"]:
        raise V8CAllocationVerificationBlocked("PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH")

    try:
        artifact_raw = Path(allocation_artifact_path).read_bytes()
    except OSError as error:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_READ_FAILED") from error
    artifact = _strict_json_object(
        artifact_raw,
        invalid_reason="ALLOCATION_ARTIFACT_INVALID_JSON",
        duplicate_reason="ALLOCATION_ARTIFACT_DUPLICATE_KEY",
    )

    if artifact.get("parent_v8_partition_manifest_sha256") != manifest_sha:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_PARENT_MANIFEST_SHA_MISMATCH")
    if artifact.get("parent_v8_partition_implementation_commit") != partition_implementation_commit:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_PARENT_IMPLEMENTATION_COMMIT_MISMATCH")
    if artifact.get("v8c_allocation_implementation_commit") != reviewed_commit:
        raise V8CAllocationVerificationBlocked("ALLOCATION_ARTIFACT_IMPLEMENTATION_COMMIT_NOT_REVIEWED")

    return _verify_t1c_allocation_artifact(
        artifact,
        parent_t_spare_tickers=parent_tickers,
        t0_tickers=blocks["T0"],
        old_t1_tickers=blocks["T1"],
        t2_tickers=blocks["T2"],
        t3_tickers=blocks["T3"],
        expected_parent_t_spare_ticker_list_sha256=computed_parent_hash,
        expected_v8c_frozen_design_commit=EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    )


def resolve_and_verify_t1c_allocation_artifact(
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """The sole public production `READ_ONLY_T1C_ALLOCATION_ARTIFACT_
    VERIFICATION` boundary. Always resolves trust from
    ``CANONICAL_REPOSITORY_ROOT``. V8B's burned T1B artifact is never read;
    the fixed coordinate rule is the authority.
    """
    return _resolve_and_verify_t1c_allocation_artifact_with_repository_root(
        allocation_artifact_path,
        partition_manifest_path,
        repository_root=CANONICAL_REPOSITORY_ROOT,
    )


__all__ = [
    "CANONICAL_REPOSITORY_ROOT",
    "V8CAllocationVerificationBlocked",
    "resolve_and_verify_t1c_allocation_artifact",
]
