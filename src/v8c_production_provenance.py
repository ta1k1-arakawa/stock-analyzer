"""V8C production provenance: exact-blob-bound freeze/authority verification.

`V8C_HISTORICAL_RESEARCH_DESIGN_DRAFT.md`, `V8C_DESIGN_FREEZE_APPROVAL.json`,
`V8_TRUSTED_PARTITION.json`. Mirrors `src.v8b_production_provenance`'s exact
pattern: every function reads its input from a caller-supplied *verified*
Git commit (`src.v8c_git_provenance.resolve_verified_v8c_production_git_
commit`'s return value) via `read_git_object_bytes`/`resolve_git_blob` --
never from the working tree directly, and never from a caller-suppliable
path. Every check compares against a hardcoded literal constant, so a
self-consistent but mutated artifact still BLOCKs. Performs no network
access and no private-data access.

The original immutable V8 trust anchor is verified by directly reusing
`src.v8b_production_provenance.read_and_verify_v8_trusted_partition_anchor`
-- it is the same anchor artifact for every V8-descended study (V8, V8B,
V8C), already independently reviewed, and takes no V8B-specific assumption
(only ``repository_root``/``verified_head``).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from src.v8b_production_provenance import (
    read_and_verify_v8_trusted_partition_anchor as _read_and_verify_v8_trusted_partition_anchor,
)
from src.v8c_git_provenance import (
    V8CGitProvenanceBlocked,
    read_git_object_bytes,
    require_git_commit,
    resolve_git_blob,
)

STUDY_NAME = "V8C_HISTORICAL_RESEARCH"

EXPECTED_V8C_FROZEN_DESIGN_COMMIT = "c9c541ac7f7ba3bcca76db6250fe8273d9bb5756"
EXPECTED_V8C_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT = "9c2cd4081a6f5ec2e48daab4a30d5ca78aea64d0"
EXPECTED_V8C_DESIGN_FREEZE_APPROVAL_BLOB = "a43eed2274bdb433ac7314515b3b9c3492afbc57"
EXPECTED_HUMAN_FREEZE_GATE = "V8C_HUMAN_DESIGN_FREEZE_APPROVED_FOR_COMMIT_" + EXPECTED_V8C_FROZEN_DESIGN_COMMIT

DESIGN_DRAFT_GIT_PATH = "V8C_HISTORICAL_RESEARCH_DESIGN_DRAFT.md"
DESIGN_FREEZE_APPROVAL_GIT_PATH = "V8C_DESIGN_FREEZE_APPROVAL.json"

# --- Original immutable V8 authority (reused unchanged from V8/V8B) --------

EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA = "61faade0625139cec3fb61216ab2f97f572a7028"
EXPECTED_V8_PARTITION_MANIFEST_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT = "36cbed941050e728f7f96ce2af505e81175cc02c"
EXPECTED_T2_TICKER_COUNT = 300
EXPECTED_T2_TICKER_LIST_SHA256 = "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500"
EXPECTED_PARENT_T_SPARE_TICKER_COUNT = 1904
EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256 = "360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70"

TRUSTED_PARTITION_ANCHOR_GIT_PATH = "V8_TRUSTED_PARTITION.json"

# --- Exact classifier blob binding (§1.2 / §7.6) ----------------------------

CANONICAL_PARSER_CLASSIFIER_FILE = "src/v7_yahoo_collector.py"
CANONICAL_PARSER_CLASSIFIER_GIT_COMMIT = "28e281c3ee30d6b4c2f981c5da3ddc983c09724d"
CANONICAL_PARSER_CLASSIFIER_BLOB_SHA = "76b57b077f3214e666ff9dc06d9c224afc16df9f"
CLASSIFIER_VERSION_MISMATCH_ERROR = "V8C_PRODUCTION_CLASSIFIER_VERSION_MISMATCH"

# --- INDEPENDENT_TRUST_PIN_REVIEW (§12: between CREATE_V8C_TRUSTED_
# ALLOCATION_PIN and T1C_TRANSPORT_READINESS_HUMAN_GATE) -------------------

TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH = "V8C_TRUST_PIN_INDEPENDENT_REVIEW.json"
TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION = "V8C_TRUST_PIN_INDEPENDENT_REVIEW_V1"
TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE = "TRUST_PIN_INDEPENDENT_REVIEW"
TRUST_PIN_INDEPENDENT_REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "reviewed_v8c_frozen_design_commit",
    "reviewed_allocation_artifact_self_hash",
    "reviewed_trust_pin_human_gate",
    "reviewed_trust_pin_git_blob_sha",
    "reviewed_trust_pin_git_commit",
    "reviewed_allocation_implementation_commit",
    "reviewed_production_implementation_commit",
    "review_result",
    "approval_status",
)

TRUST_PIN_GIT_PATH = "V8C_TRUSTED_ALLOCATION.json"

# --- Future INDEPENDENT_V8C_PRODUCTION_IMPLEMENTATION_REVIEW binding -------

IMPLEMENTATION_REVIEW_GIT_PATH = "V8C_PRODUCTION_IMPLEMENTATION_REVIEW.json"
IMPLEMENTATION_REVIEW_SCHEMA_VERSION = "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_V1"
IMPLEMENTATION_REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "reviewed_implementation_git_commit",
    "review_result",
    "approval_status",
)

# Every production-relevant V8C source file that must be byte-for-byte
# identical between current verified HEAD and the exact commit
# INDEPENDENT_V8C_PRODUCTION_IMPLEMENTATION_REVIEW reviewed. A later
# docs/audit-only commit may move HEAD forward without invalidating review
# of these files, as long as none of their blobs actually changed.
BOUND_PRODUCTION_FILES: tuple[str, ...] = (
    "src/v8c_git_provenance.py",
    "src/v8c_production_provenance.py",
    "src/v8c_transport.py",
    "src/v8c_human_gate_consumption.py",
    "src/v8c_readiness.py",
    "src/v8c_t1c_allocation.py",
    "src/v8c_t1c_allocator.py",
    "src/v8c_t1c_allocation_verification.py",
    "src/v8c_trust_pin.py",
    "src/v8c_trust_pin_creation.py",
    "src/v8c_t2_bridge.py",
    "src/v8c_t2_preservation_recheck.py",
    "src/v8c_historical_acquisition.py",
    "src/v8c_acquisition_artifact_verification.py",
    "src/v8c_research_opening_guard.py",
    "src/v8c_stage_state.py",
)


class V8CProductionProvenanceBlocked(RuntimeError):
    """Fail-closed exact-blob production provenance error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8CProductionProvenanceBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8CProductionProvenanceBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8CProductionProvenanceBlocked(invalid_reason)
    return parsed


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap_git_provenance_error(error: V8CGitProvenanceBlocked, missing_reason: str | None = None) -> V8CProductionProvenanceBlocked:
    if missing_reason is not None and error.reason in _GIT_OBJECT_MISSING_REASONS:
        return V8CProductionProvenanceBlocked(missing_reason)
    return V8CProductionProvenanceBlocked(error.reason)


# ---------------------------------------------------------------------------
# Frozen design object (verified at the frozen commit itself)
# ---------------------------------------------------------------------------


def verify_frozen_design_object(repository_root) -> None:
    """Verify the frozen `V8C_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` object
    *at the frozen design commit itself* -- not at current HEAD -- exactly
    matches the blob independently reviewed."""
    try:
        blob = resolve_git_blob(repository_root, EXPECTED_V8C_FROZEN_DESIGN_COMMIT, DESIGN_DRAFT_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error) from error
    if blob != EXPECTED_V8C_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT:
        raise V8CProductionProvenanceBlocked("V8C_FROZEN_DESIGN_OBJECT_MUTATED")


# ---------------------------------------------------------------------------
# Design freeze approval (exact blob + exact field semantics)
# ---------------------------------------------------------------------------


def read_and_verify_design_freeze_approval(repository_root, verified_head: str) -> dict[str, Any]:
    commit = require_git_commit(verified_head, "DESIGN_FREEZE_APPROVAL_HEAD_INVALID")
    try:
        blob = resolve_git_blob(repository_root, commit, DESIGN_FREEZE_APPROVAL_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8C_DESIGN_FREEZE_APPROVAL_MISSING") from error
    if blob != EXPECTED_V8C_DESIGN_FREEZE_APPROVAL_BLOB:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_APPROVAL_BLOB_MUTATED")
    try:
        raw = read_git_object_bytes(repository_root, commit, DESIGN_FREEZE_APPROVAL_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8C_DESIGN_FREEZE_APPROVAL_MISSING") from error
    approval = _strict_json_object(
        raw,
        invalid_reason="V8C_DESIGN_FREEZE_APPROVAL_INVALID_JSON",
        duplicate_reason="V8C_DESIGN_FREEZE_APPROVAL_DUPLICATE_KEY",
    )

    if approval.get("schema_version") != "V8C_DESIGN_FREEZE_APPROVAL_V1":
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_APPROVAL_SCHEMA_VERSION_MISMATCH")
    if approval.get("study") != STUDY_NAME:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_APPROVAL_STUDY_MISMATCH")
    if approval.get("frozen_design_git_commit") != EXPECTED_V8C_FROZEN_DESIGN_COMMIT:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_APPROVAL_COMMIT_MISMATCH")
    if approval.get("frozen_design_git_blob_sha") != EXPECTED_V8C_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_APPROVAL_DESIGN_BLOB_MISMATCH")
    if approval.get("final_independent_review_result") != "PASS":
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_FINAL_REVIEW_NOT_PASS")
    if approval.get("final_independent_review_design_commit") != EXPECTED_V8C_FROZEN_DESIGN_COMMIT:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_FINAL_REVIEW_COMMIT_MISMATCH")
    if approval.get("t1c_t_spare_freshness_preservation_recheck_result") != "PASS":
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_T1C_PRESERVATION_RECHECK_NOT_PASS")
    if approval.get("t1c_t_spare_freshness_preservation_recheck_design_commit") != EXPECTED_V8C_FROZEN_DESIGN_COMMIT:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_T1C_PRESERVATION_RECHECK_COMMIT_MISMATCH")
    if approval.get("t2_preservation_recheck_result") != "PASS":
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_T2_PRESERVATION_RECHECK_NOT_PASS")
    if approval.get("t2_preservation_recheck_design_commit") != EXPECTED_V8C_FROZEN_DESIGN_COMMIT:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_T2_PRESERVATION_RECHECK_COMMIT_MISMATCH")
    if approval.get("approval_status") != "APPROVED":
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_APPROVAL_NOT_APPROVED")
    if approval.get("human_gate") != EXPECTED_HUMAN_FREEZE_GATE:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_HUMAN_GATE_MISMATCH")
    if approval.get("design_finalized") is not True:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_NOT_FINALIZED")
    if approval.get("human_design_freeze_complete") is not True:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_NOT_COMPLETE")
    if approval.get("t1c_allocation_authorized") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_T1C_ALLOCATION_AUTHORIZATION")
    if approval.get("t1c_allocation_executed") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_T1C_ALLOCATION_EXECUTION")
    if approval.get("t1c_raw_acquisition_authorized") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_T1C_ACQUISITION_AUTHORIZATION")
    if approval.get("t1c_raw_acquisition_executed") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_T1C_ACQUISITION_EXECUTION")
    if approval.get("t2_authority_bridge_authorized") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_T2_BRIDGE_AUTHORIZATION")
    if approval.get("t2_raw_acquisition_authorized") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_T2_ACQUISITION_AUTHORIZATION")
    if approval.get("t2_raw_acquisition_executed") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_T2_ACQUISITION_EXECUTION")
    if approval.get("real_network_authorized") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_NETWORK_AUTHORIZATION")
    if approval.get("private_partition_access_authorized") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_PRIVATE_PARTITION_ACCESS_AUTHORIZATION")
    if approval.get("research_opening_authorized") is not False:
        raise V8CProductionProvenanceBlocked("V8C_DESIGN_FREEZE_UNEXPECTED_RESEARCH_OPENING_AUTHORIZATION")
    return dict(approval)


# ---------------------------------------------------------------------------
# Reviewed-implementation binding (future INDEPENDENT_V8C_PRODUCTION_
# IMPLEMENTATION_REVIEW artifact -- does not exist in this repository yet,
# so this fails closed today by construction)
# ---------------------------------------------------------------------------


def verify_reviewed_implementation_binding(repository_root, verified_head: str) -> dict[str, Any]:
    """Bind production to the exact commit `INDEPENDENT_V8C_PRODUCTION_
    IMPLEMENTATION_REVIEW` actually reviewed, re-derived from the review
    artifact itself -- never assumed to equal current HEAD."""
    commit = require_git_commit(verified_head, "IMPLEMENTATION_REVIEW_HEAD_INVALID")
    try:
        raw = read_git_object_bytes(repository_root, commit, IMPLEMENTATION_REVIEW_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    review = _strict_json_object(
        raw,
        invalid_reason="V8C_PRODUCTION_IMPLEMENTATION_REVIEW_INVALID_JSON",
        duplicate_reason="V8C_PRODUCTION_IMPLEMENTATION_REVIEW_DUPLICATE_KEY",
    )
    if set(review) != set(IMPLEMENTATION_REVIEW_FIELDS):
        raise V8CProductionProvenanceBlocked("V8C_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_INVALID")
    if review["schema_version"] != IMPLEMENTATION_REVIEW_SCHEMA_VERSION:
        raise V8CProductionProvenanceBlocked("V8C_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_VERSION_MISMATCH")
    if review["study"] != STUDY_NAME:
        raise V8CProductionProvenanceBlocked("V8C_PRODUCTION_IMPLEMENTATION_REVIEW_STUDY_MISMATCH")
    if review["artifact_role"] != "PRODUCTION_IMPLEMENTATION_REVIEW":
        raise V8CProductionProvenanceBlocked("V8C_PRODUCTION_IMPLEMENTATION_REVIEW_ARTIFACT_ROLE_MISMATCH")
    if review["review_result"] != "PASS":
        raise V8CProductionProvenanceBlocked("V8C_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_PASS")
    if review["approval_status"] != "APPROVED":
        raise V8CProductionProvenanceBlocked("V8C_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_APPROVED")

    reviewed_commit = require_git_commit(
        review["reviewed_implementation_git_commit"], "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_COMMIT_INVALID"
    )

    for path in BOUND_PRODUCTION_FILES:
        try:
            blob_head = resolve_git_blob(repository_root, commit, path)
        except V8CGitProvenanceBlocked as error:
            raise _wrap_git_provenance_error(error, "V8C_BOUND_FILE_MISSING_AT_HEAD:" + path) from error
        try:
            blob_reviewed = resolve_git_blob(repository_root, reviewed_commit, path)
        except V8CGitProvenanceBlocked as error:
            raise _wrap_git_provenance_error(error, "V8C_BOUND_FILE_MISSING_AT_REVIEWED_COMMIT:" + path) from error
        if blob_head != blob_reviewed:
            raise V8CProductionProvenanceBlocked("V8C_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:" + path)

    return {
        "reviewed_implementation_git_commit": reviewed_commit,
        "verified_head": commit,
        "bound_files_verified": len(BOUND_PRODUCTION_FILES),
    }


# ---------------------------------------------------------------------------
# INDEPENDENT_TRUST_PIN_REVIEW
# ---------------------------------------------------------------------------


def read_and_verify_trust_pin_independent_review(
    repository_root,
    verified_head: str,
    *,
    expected_allocation_artifact_self_hash: str,
    expected_trust_pin_human_gate: str,
) -> dict[str, Any]:
    """Read and verify the future `V8C_TRUST_PIN_INDEPENDENT_REVIEW.json`
    artifact from a verified Git object -- never a caller-supplied path or
    mapping. Bound to ``expected_allocation_artifact_self_hash`` and
    ``expected_trust_pin_human_gate`` (exact-value), and independently
    re-resolves the trust-pin Git blob at both the review's claimed
    ``reviewed_trust_pin_git_commit`` and the current verified HEAD, so
    neither a fabricated review nor a trust pin swapped/mutated after
    review can pass. Does not exist in this repository yet, so this fails
    closed today.
    """
    commit = require_git_commit(verified_head, "TRUST_PIN_INDEPENDENT_REVIEW_HEAD_INVALID")
    if not isinstance(expected_allocation_artifact_self_hash, str) or len(expected_allocation_artifact_self_hash) != 64:
        raise V8CProductionProvenanceBlocked("TRUST_PIN_INDEPENDENT_REVIEW_EXPECTED_HASH_INVALID")
    if not isinstance(expected_trust_pin_human_gate, str) or not expected_trust_pin_human_gate:
        raise V8CProductionProvenanceBlocked("TRUST_PIN_INDEPENDENT_REVIEW_EXPECTED_HUMAN_GATE_INVALID")
    try:
        raw = read_git_object_bytes(repository_root, commit, TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8C_TRUST_PIN_INDEPENDENT_REVIEW_MISSING") from error
    review = _strict_json_object(
        raw,
        invalid_reason="V8C_TRUST_PIN_INDEPENDENT_REVIEW_INVALID_JSON",
        duplicate_reason="V8C_TRUST_PIN_INDEPENDENT_REVIEW_DUPLICATE_KEY",
    )
    if set(review) != set(TRUST_PIN_INDEPENDENT_REVIEW_FIELDS):
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_INVALID")
    if review["schema_version"] != TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION:
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION_MISMATCH")
    if review["study"] != STUDY_NAME:
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_STUDY_MISMATCH")
    if review["artifact_role"] != TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE:
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE_MISMATCH")
    if review["reviewed_v8c_frozen_design_commit"] != EXPECTED_V8C_FROZEN_DESIGN_COMMIT:
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_DESIGN_COMMIT_MISMATCH")
    if review["reviewed_allocation_artifact_self_hash"] != expected_allocation_artifact_self_hash:
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_HASH_MISMATCH")
    if review["reviewed_trust_pin_human_gate"] != expected_trust_pin_human_gate:
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_HUMAN_GATE_MISMATCH")
    try:
        current_pin = _strict_json_object(
            read_git_object_bytes(repository_root, commit, TRUST_PIN_GIT_PATH),
            invalid_reason="V8C_TRUST_PIN_INVALID_JSON",
            duplicate_reason="V8C_TRUST_PIN_DUPLICATE_KEY",
        )
    except V8CGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8C_TRUST_PIN_INDEPENDENT_REVIEW_CURRENT_TRUST_PIN_MISSING") from error
    if review["reviewed_allocation_implementation_commit"] != current_pin.get("v8c_allocation_implementation_commit"):
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_ALLOCATION_IMPLEMENTATION_MISMATCH")
    if review["reviewed_production_implementation_commit"] != current_pin.get("v8c_reviewed_production_implementation_commit"):
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_PRODUCTION_IMPLEMENTATION_MISMATCH")
    reviewed_commit = require_git_commit(
        review["reviewed_trust_pin_git_commit"], "V8C_TRUST_PIN_INDEPENDENT_REVIEW_REVIEWED_COMMIT_INVALID"
    )
    reviewed_blob_sha = require_git_commit(
        review["reviewed_trust_pin_git_blob_sha"], "V8C_TRUST_PIN_INDEPENDENT_REVIEW_REVIEWED_BLOB_SHA_INVALID"
    )
    try:
        blob_at_reviewed_commit = resolve_git_blob(repository_root, reviewed_commit, TRUST_PIN_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(
            error, "V8C_TRUST_PIN_INDEPENDENT_REVIEW_REVIEWED_COMMIT_TRUST_PIN_MISSING"
        ) from error
    if blob_at_reviewed_commit != reviewed_blob_sha:
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_BLOB_SELF_INCONSISTENT")
    try:
        blob_at_current_head = resolve_git_blob(repository_root, commit, TRUST_PIN_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8C_TRUST_PIN_INDEPENDENT_REVIEW_CURRENT_TRUST_PIN_MISSING") from error
    if blob_at_current_head != reviewed_blob_sha:
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_TRUST_PIN_BLOB_DRIFT")
    if review["review_result"] != "PASS":
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_NOT_PASS")
    if review["approval_status"] != "APPROVED":
        raise V8CProductionProvenanceBlocked("V8C_TRUST_PIN_INDEPENDENT_REVIEW_NOT_APPROVED")
    return dict(review)


# ---------------------------------------------------------------------------
# Exact classifier blob binding (§1.2 / §7.6)
# ---------------------------------------------------------------------------


def verify_classifier_blob(classifier_blob_sha: str) -> None:
    if classifier_blob_sha != CANONICAL_PARSER_CLASSIFIER_BLOB_SHA:
        raise V8CProductionProvenanceBlocked(CLASSIFIER_VERSION_MISMATCH_ERROR)


# ---------------------------------------------------------------------------
# Original immutable V8 authority (reused unchanged from V8/V8B)
# ---------------------------------------------------------------------------


def read_and_verify_v8_trusted_partition_anchor(repository_root, verified_head: str) -> dict[str, Any]:
    """Reuse `src.v8b_production_provenance`'s already-reviewed exact-blob
    V8 trust-anchor verification unchanged -- it is the same original,
    immutable V8 anchor for every V8-descended study and takes no
    V8B-specific assumption."""
    try:
        return _read_and_verify_v8_trusted_partition_anchor(repository_root, verified_head)
    except Exception as error:  # noqa: BLE001 - re-wrap into this module's exception type
        reason = getattr(error, "reason", None)
        if isinstance(reason, str):
            raise V8CProductionProvenanceBlocked(reason) from error
        raise


__all__ = [
    "BOUND_PRODUCTION_FILES",
    "CANONICAL_PARSER_CLASSIFIER_BLOB_SHA",
    "CANONICAL_PARSER_CLASSIFIER_FILE",
    "CANONICAL_PARSER_CLASSIFIER_GIT_COMMIT",
    "CLASSIFIER_VERSION_MISMATCH_ERROR",
    "DESIGN_DRAFT_GIT_PATH",
    "DESIGN_FREEZE_APPROVAL_GIT_PATH",
    "EXPECTED_HUMAN_FREEZE_GATE",
    "EXPECTED_PARENT_T_SPARE_TICKER_COUNT",
    "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256",
    "EXPECTED_T2_TICKER_COUNT",
    "EXPECTED_T2_TICKER_LIST_SHA256",
    "EXPECTED_V8C_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT",
    "EXPECTED_V8C_DESIGN_FREEZE_APPROVAL_BLOB",
    "EXPECTED_V8C_FROZEN_DESIGN_COMMIT",
    "EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT",
    "EXPECTED_V8_PARTITION_MANIFEST_SHA256",
    "EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA",
    "IMPLEMENTATION_REVIEW_FIELDS",
    "IMPLEMENTATION_REVIEW_GIT_PATH",
    "IMPLEMENTATION_REVIEW_SCHEMA_VERSION",
    "STUDY_NAME",
    "TRUSTED_PARTITION_ANCHOR_GIT_PATH",
    "TRUST_PIN_GIT_PATH",
    "TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE",
    "TRUST_PIN_INDEPENDENT_REVIEW_FIELDS",
    "TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH",
    "TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION",
    "V8CProductionProvenanceBlocked",
    "read_and_verify_design_freeze_approval",
    "read_and_verify_trust_pin_independent_review",
    "read_and_verify_v8_trusted_partition_anchor",
    "verify_classifier_blob",
    "verify_frozen_design_object",
    "verify_reviewed_implementation_binding",
]
