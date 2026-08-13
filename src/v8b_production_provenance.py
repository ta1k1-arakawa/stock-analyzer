"""V8B production provenance: exact-blob-bound freeze/authority verification.

`V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §11, §12.5, `V8B_DESIGN_FREEZE_
APPROVAL.json`, `V8_TRUSTED_PARTITION.json`, `V8B_T2_AUTHORITY_BRIDGE.json`.

Every function here reads its input from a caller-supplied *verified*
Git commit (`src.v8b_git_provenance.resolve_verified_v8b_production_git_
commit`'s return value) via `src.v8b_git_provenance.read_git_object_bytes`/
`resolve_git_blob` -- never from the working tree directly, and never from
a caller-suppliable path. Every check compares against a hardcoded literal
constant taken from this task's independent review, not merely "whatever
value the artifact currently states" -- so a self-consistent but mutated
artifact (an internally-consistent re-pin, a semantically similar modified
design/approval document) still BLOCKs. Performs no network access and no
private-data access.
"""

from __future__ import annotations

import json
from typing import Any

from src.v8b_git_provenance import (
    V8BGitProvenanceBlocked,
    read_git_object_bytes,
    require_git_commit,
    resolve_git_blob,
)

STUDY_NAME = "V8B_HISTORICAL_RESEARCH"

# --- Frozen V8B design / freeze approval (HIGH-5) --------------------------

EXPECTED_V8B_FROZEN_DESIGN_COMMIT = "eedf198b93185b963b825170ed0be97e93f923b7"
EXPECTED_V8B_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT = "33e6789e5dcca8ba9ea393460d14c3e9fd387508"
EXPECTED_V8B_DESIGN_FREEZE_APPROVAL_BLOB = "545ffaa360a48c3220100edf5d0f522e97a0a0f0"
EXPECTED_HUMAN_FREEZE_GATE = "V8B_HUMAN_DESIGN_FREEZE_APPROVED_FOR_COMMIT_" + EXPECTED_V8B_FROZEN_DESIGN_COMMIT

DESIGN_DRAFT_GIT_PATH = "V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md"
DESIGN_FREEZE_APPROVAL_GIT_PATH = "V8B_DESIGN_FREEZE_APPROVAL.json"

# --- Original immutable V8 authority (HIGH-4) -------------------------------

V8_DESIGN_COMMIT = "c414d3191cba356734d7ed08bdf1abc7d51fc384"
EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA = "61faade0625139cec3fb61216ab2f97f572a7028"
EXPECTED_V8_PARTITION_MANIFEST_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT = "36cbed941050e728f7f96ce2af505e81175cc02c"
EXPECTED_T2_TICKER_COUNT = 300
EXPECTED_T2_TICKER_LIST_SHA256 = "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500"
EXPECTED_PARENT_T_SPARE_TICKER_COUNT = 1904
EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256 = "360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70"
EXPECTED_T2_OPTION_2_HUMAN_GATE = "V8B_T2_AUTHORITY_INTEGRATION_OPTION_2_APPROVED"

TRUSTED_PARTITION_ANCHOR_GIT_PATH = "V8_TRUSTED_PARTITION.json"
T2_AUTHORITY_BRIDGE_GIT_PATH = "V8B_T2_AUTHORITY_BRIDGE.json"

TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION = "V8_TRUSTED_PARTITION_V1"
TRUSTED_PARTITION_ANCHOR_FIELDS = (
    "schema_version",
    "study_name",
    "design_commit",
    "authorization_status",
    "authorized_partition_manifest_sha256",
    "authorized_partition_implementation_git_commit",
    "authorization_note",
)

T2_AUTHORITY_BRIDGE_SCHEMA_VERSION = "V8B_T2_AUTHORITY_BRIDGE_V1"
T2_AUTHORITY_BRIDGE_FIELDS = (
    "schema_version",
    "study",
    "role",
    "source_authority",
    "v8_trust_anchor_git_path",
    "v8_trust_anchor_git_identity",
    "authorized_parent_v8_partition_manifest_sha256",
    "expected_t2_ticker_list_sha256",
    "t2_acquired_before_authorized_acquisition",
    "t2_research_open_count_before_official_opening",
    "v8b_frozen_design_commit",
    "t2_membership_reassignment",
    "v8_trusted_partition_json_mutated_or_repinned",
    "option",
    "human_gate",
    "authorization_note",
)

# --- Reviewed-implementation binding (HIGH-2) -------------------------------

IMPLEMENTATION_REVIEW_GIT_PATH = "V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json"
IMPLEMENTATION_REVIEW_SCHEMA_VERSION = "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_V1"
IMPLEMENTATION_REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "reviewed_implementation_git_commit",
    "review_result",
    "approval_status",
)

# --- INDEPENDENT_TRUST_PIN_REVIEW (FINAL_REPEAT finding HIGH-2) ------------
#
# §12's gate sequence requires INDEPENDENT_TRUST_PIN_REVIEW -- an
# independent review of the published §11.3.C trust-pin artifact, bound to
# the exact allocation-artifact self-hash it pins -- between
# CREATE_V8B_TRUSTED_ALLOCATION_PIN and T1B_RAW_ACQUISITION_HUMAN_GATE.
# This future artifact does not exist in this repository yet, so reading
# it fails closed today by construction, exactly like
# `V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json` and
# `V8B_TRUSTED_ALLOCATION.json`.

TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH = "V8B_TRUST_PIN_INDEPENDENT_REVIEW.json"
TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION = "V8B_TRUST_PIN_INDEPENDENT_REVIEW_V1"
TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE = "TRUST_PIN_INDEPENDENT_REVIEW"
TRUST_PIN_INDEPENDENT_REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "reviewed_allocation_artifact_self_hash",
    "reviewed_trust_pin_human_gate",
    "review_result",
    "approval_status",
)

# Every production-relevant fixed source/artifact file that must be
# byte-for-byte identical between current verified HEAD and the exact
# commit INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW reviewed. A
# later docs/audit-only commit may move HEAD forward without invalidating
# review of these files, as long as none of their blobs actually changed.
BOUND_PRODUCTION_FILES: tuple[str, ...] = (
    "src/v8b_git_provenance.py",
    "src/v8b_production_provenance.py",
    "src/v8b_allocation.py",
    "src/v8b_allocation_verification.py",
    "src/v8b_trust_pin.py",
    "src/v8b_historical_acquisition.py",
    "src/v8b_acquisition_artifact_verification.py",
    "src/v8b_t2_reuse_recheck.py",
    "src/v8b_t1b_allocator.py",
    "src/v8b_trust_pin_creation.py",
    "src/v8b_human_gate_consumption.py",
    "V8B_T2_AUTHORITY_BRIDGE.json",
    # V8B production executes src/v8_partition.py's read_partition_manifest,
    # require_absolute_output_path_outside_repository, ticker_list_sha256,
    # and its constants directly -- a later change to that file must BLOCK
    # even when every V8B-authored module blob is unchanged (round-2
    # finding HIGH-3). The independent exact classifier pin for
    # src/v7_yahoo_collector.py (§7.6) remains separate and unaffected.
    "src/v8_partition.py",
)


class V8BProductionProvenanceBlocked(RuntimeError):
    """Fail-closed exact-blob production provenance error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8BProductionProvenanceBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8BProductionProvenanceBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8BProductionProvenanceBlocked(invalid_reason)
    return parsed


def _require_sha256_hex(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V8BProductionProvenanceBlocked(reason)
    return value


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap_git_provenance_error(error: V8BGitProvenanceBlocked, missing_reason: str | None = None) -> V8BProductionProvenanceBlocked:
    if missing_reason is not None and error.reason in _GIT_OBJECT_MISSING_REASONS:
        return V8BProductionProvenanceBlocked(missing_reason)
    return V8BProductionProvenanceBlocked(error.reason)


# ---------------------------------------------------------------------------
# Frozen design object (verified at the frozen commit itself)
# ---------------------------------------------------------------------------


def verify_frozen_design_object(repository_root) -> None:
    """Verify the frozen `V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` object
    *at the frozen design commit itself* -- not at current HEAD -- exactly
    matches the blob independently reviewed."""
    try:
        blob = resolve_git_blob(repository_root, EXPECTED_V8B_FROZEN_DESIGN_COMMIT, DESIGN_DRAFT_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error) from error
    if blob != EXPECTED_V8B_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT:
        raise V8BProductionProvenanceBlocked("V8B_FROZEN_DESIGN_OBJECT_MUTATED")


# ---------------------------------------------------------------------------
# Design freeze approval (exact blob + exact field semantics)
# ---------------------------------------------------------------------------


def read_and_verify_design_freeze_approval(repository_root, verified_head: str) -> dict[str, Any]:
    commit = require_git_commit(verified_head, "DESIGN_FREEZE_APPROVAL_HEAD_INVALID")
    try:
        blob = resolve_git_blob(repository_root, commit, DESIGN_FREEZE_APPROVAL_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8B_DESIGN_FREEZE_APPROVAL_MISSING") from error
    if blob != EXPECTED_V8B_DESIGN_FREEZE_APPROVAL_BLOB:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_APPROVAL_BLOB_MUTATED")
    try:
        raw = read_git_object_bytes(repository_root, commit, DESIGN_FREEZE_APPROVAL_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8B_DESIGN_FREEZE_APPROVAL_MISSING") from error
    approval = _strict_json_object(
        raw,
        invalid_reason="V8B_DESIGN_FREEZE_APPROVAL_INVALID_JSON",
        duplicate_reason="V8B_DESIGN_FREEZE_APPROVAL_DUPLICATE_KEY",
    )

    if approval.get("schema_version") != "V8B_DESIGN_FREEZE_APPROVAL_V1":
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_APPROVAL_SCHEMA_VERSION_MISMATCH")
    if approval.get("study") != STUDY_NAME:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_APPROVAL_STUDY_MISMATCH")
    if approval.get("frozen_design_git_commit") != EXPECTED_V8B_FROZEN_DESIGN_COMMIT:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_APPROVAL_COMMIT_MISMATCH")
    if approval.get("final_independent_review_result") != "PASS":
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_FINAL_REVIEW_NOT_PASS")
    if approval.get("final_independent_review_design_commit") != EXPECTED_V8B_FROZEN_DESIGN_COMMIT:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_FINAL_REVIEW_COMMIT_MISMATCH")
    if approval.get("preservation_recheck_result") != "PASS":
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_PRESERVATION_RECHECK_NOT_PASS")
    if approval.get("preservation_recheck_design_commit") != EXPECTED_V8B_FROZEN_DESIGN_COMMIT:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_PRESERVATION_RECHECK_COMMIT_MISMATCH")
    if approval.get("approval_status") != "APPROVED":
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED")
    if approval.get("human_gate") != EXPECTED_HUMAN_FREEZE_GATE:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_HUMAN_GATE_MISMATCH")
    if approval.get("design_finalized") is not True:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_NOT_FINALIZED")
    if approval.get("human_design_freeze_complete") is not True:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_NOT_COMPLETE")
    if approval.get("t1b_allocation_authorized") is not False:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_UNEXPECTED_ALLOCATION_AUTHORIZATION")
    if approval.get("real_network_authorized") is not False:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_UNEXPECTED_NETWORK_AUTHORIZATION")
    if approval.get("t1b_acquisition_authorized") is not False:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_UNEXPECTED_T1B_ACQUISITION_AUTHORIZATION")
    if approval.get("t2_acquisition_authorized") is not False:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_UNEXPECTED_T2_ACQUISITION_AUTHORIZATION")
    if approval.get("research_opening_authorized") is not False:
        raise V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_UNEXPECTED_RESEARCH_OPENING_AUTHORIZATION")
    return dict(approval)


# ---------------------------------------------------------------------------
# Reviewed-implementation binding
# ---------------------------------------------------------------------------


def verify_reviewed_implementation_binding(repository_root, verified_head: str) -> dict[str, Any]:
    """Bind production to the exact commit `INDEPENDENT_V8B_PRODUCTION_
    IMPLEMENTATION_REVIEW` actually reviewed, re-derived from the review
    artifact itself -- never assumed to equal current HEAD. A later
    docs/audit-only commit may move HEAD forward without BLOCKing, as long
    as every bound production file's blob at HEAD still exactly matches its
    blob at the reviewed commit; a single changed bound blob BLOCKs.
    """
    commit = require_git_commit(verified_head, "IMPLEMENTATION_REVIEW_HEAD_INVALID")
    try:
        raw = read_git_object_bytes(repository_root, commit, IMPLEMENTATION_REVIEW_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    review = _strict_json_object(
        raw,
        invalid_reason="V8B_PRODUCTION_IMPLEMENTATION_REVIEW_INVALID_JSON",
        duplicate_reason="V8B_PRODUCTION_IMPLEMENTATION_REVIEW_DUPLICATE_KEY",
    )
    if set(review) != set(IMPLEMENTATION_REVIEW_FIELDS):
        raise V8BProductionProvenanceBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_INVALID")
    if review["schema_version"] != IMPLEMENTATION_REVIEW_SCHEMA_VERSION:
        raise V8BProductionProvenanceBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_VERSION_MISMATCH")
    if review["study"] != STUDY_NAME:
        raise V8BProductionProvenanceBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_STUDY_MISMATCH")
    if review["review_result"] != "PASS":
        raise V8BProductionProvenanceBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_PASS")
    if review["approval_status"] != "APPROVED":
        raise V8BProductionProvenanceBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_APPROVED")

    reviewed_commit = require_git_commit(
        review["reviewed_implementation_git_commit"], "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_COMMIT_INVALID"
    )

    for path in BOUND_PRODUCTION_FILES:
        try:
            blob_head = resolve_git_blob(repository_root, commit, path)
        except V8BGitProvenanceBlocked as error:
            raise _wrap_git_provenance_error(error, "V8B_BOUND_FILE_MISSING_AT_HEAD:" + path) from error
        try:
            blob_reviewed = resolve_git_blob(repository_root, reviewed_commit, path)
        except V8BGitProvenanceBlocked as error:
            raise _wrap_git_provenance_error(error, "V8B_BOUND_FILE_MISSING_AT_REVIEWED_COMMIT:" + path) from error
        if blob_head != blob_reviewed:
            raise V8BProductionProvenanceBlocked("V8B_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:" + path)

    return {
        "reviewed_implementation_git_commit": reviewed_commit,
        "verified_head": commit,
        "bound_files_verified": len(BOUND_PRODUCTION_FILES),
    }


# ---------------------------------------------------------------------------
# Original immutable V8 authority (exact blob, never mere internal consistency)
# ---------------------------------------------------------------------------


def read_and_verify_v8_trusted_partition_anchor(repository_root, verified_head: str) -> dict[str, Any]:
    commit = require_git_commit(verified_head, "TRUSTED_PARTITION_ANCHOR_HEAD_INVALID")
    try:
        blob = resolve_git_blob(repository_root, commit, TRUSTED_PARTITION_ANCHOR_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "TRUSTED_PARTITION_ANCHOR_MISSING") from error
    if blob != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        # A re-pin, mutation, or any other edit to the immutable V8 anchor
        # BLOCKs here, before its (possibly internally self-consistent)
        # field values are ever read -- an attacker who both mutates the
        # anchor and crafts a matching private manifest cannot pass this.
        raise V8BProductionProvenanceBlocked("V8_TRUSTED_PARTITION_BLOB_MUTATED")
    try:
        raw = read_git_object_bytes(repository_root, commit, TRUSTED_PARTITION_ANCHOR_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "TRUSTED_PARTITION_ANCHOR_MISSING") from error
    anchor = _strict_json_object(
        raw,
        invalid_reason="TRUSTED_PARTITION_ANCHOR_INVALID_JSON",
        duplicate_reason="TRUSTED_PARTITION_ANCHOR_DUPLICATE_KEY",
    )
    if set(anchor) != set(TRUSTED_PARTITION_ANCHOR_FIELDS):
        raise V8BProductionProvenanceBlocked("TRUSTED_PARTITION_ANCHOR_SCHEMA_INVALID")
    if anchor["schema_version"] != TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION:
        raise V8BProductionProvenanceBlocked("TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION_MISMATCH")
    if anchor["design_commit"] != V8_DESIGN_COMMIT:
        raise V8BProductionProvenanceBlocked("TRUSTED_PARTITION_ANCHOR_DESIGN_COMMIT_MISMATCH")
    if anchor["authorization_status"] != "AUTHORIZED":
        raise V8BProductionProvenanceBlocked("TRUSTED_PARTITION_NOT_AUTHORIZED")
    # Pinned to the exact frozen literal values, never merely "whatever the
    # anchor's own fields say" -- an internally self-consistent re-pin
    # still fails here.
    if anchor["authorized_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8BProductionProvenanceBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    if anchor["authorized_partition_implementation_git_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8BProductionProvenanceBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")
    return dict(anchor)


def read_and_verify_t2_authority_bridge(repository_root, verified_head: str) -> dict[str, Any]:
    commit = require_git_commit(verified_head, "T2_AUTHORITY_BRIDGE_HEAD_INVALID")
    try:
        raw = read_git_object_bytes(repository_root, commit, T2_AUTHORITY_BRIDGE_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8B_T2_AUTHORITY_BRIDGE_MISSING") from error
    bridge = _strict_json_object(
        raw,
        invalid_reason="V8B_T2_AUTHORITY_BRIDGE_INVALID_JSON",
        duplicate_reason="V8B_T2_AUTHORITY_BRIDGE_DUPLICATE_KEY",
    )
    if set(bridge) != set(T2_AUTHORITY_BRIDGE_FIELDS):
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_SCHEMA_INVALID")
    if bridge["schema_version"] != T2_AUTHORITY_BRIDGE_SCHEMA_VERSION:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_SCHEMA_VERSION_MISMATCH")
    if bridge["study"] != STUDY_NAME:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_STUDY_MISMATCH")
    if bridge["role"] != "SEALED_HOLDOUT":
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_ROLE_MISMATCH")
    if bridge["source_authority"] != "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY":
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_SOURCE_AUTHORITY_MISMATCH")
    if bridge["option"] != "OPTION_2":
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_OPTION_MISMATCH")
    if bridge["v8_trust_anchor_git_identity"] != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_ANCHOR_IDENTITY_MISMATCH")
    if bridge["authorized_parent_v8_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_MANIFEST_SHA_MISMATCH")
    if bridge["expected_t2_ticker_list_sha256"] != EXPECTED_T2_TICKER_LIST_SHA256:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_TICKER_LIST_SHA_MISMATCH")
    if bridge["v8b_frozen_design_commit"] != EXPECTED_V8B_FROZEN_DESIGN_COMMIT:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_DESIGN_COMMIT_MISMATCH")
    if bridge["t2_membership_reassignment"] != "PROHIBITED":
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_MEMBERSHIP_REASSIGNMENT_INVALID")
    if bridge["t2_acquired_before_authorized_acquisition"] is not False:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_ACQUIRED_BEFORE_INVALID")
    if bridge["t2_research_open_count_before_official_opening"] != 0:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_OPEN_COUNT_INVALID")
    if bridge["v8_trusted_partition_json_mutated_or_repinned"] is not False:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_ANCHOR_MUTATION_INVALID")
    if bridge["human_gate"] != EXPECTED_T2_OPTION_2_HUMAN_GATE:
        raise V8BProductionProvenanceBlocked("V8B_T2_AUTHORITY_BRIDGE_HUMAN_GATE_MISMATCH")
    return dict(bridge)


# ---------------------------------------------------------------------------
# INDEPENDENT_TRUST_PIN_REVIEW (HIGH-2)
# ---------------------------------------------------------------------------


def read_and_verify_trust_pin_independent_review(
    repository_root, verified_head: str, *, expected_allocation_artifact_self_hash: str
) -> dict[str, Any]:
    """Read and verify the future `V8B_TRUST_PIN_INDEPENDENT_REVIEW.json`
    artifact from a **verified Git object** -- never a caller-supplied path
    or mapping. Bound to ``expected_allocation_artifact_self_hash`` so a
    review of a *different* allocation artifact can never authorize this
    one (HIGH-2: "independent trust-pin review for that exact artifact").
    Does not exist in this repository yet, so this fails closed today.
    """
    commit = require_git_commit(verified_head, "TRUST_PIN_INDEPENDENT_REVIEW_HEAD_INVALID")
    hash_value = _require_sha256_hex(
        expected_allocation_artifact_self_hash, "TRUST_PIN_INDEPENDENT_REVIEW_EXPECTED_HASH_INVALID"
    )
    try:
        raw = read_git_object_bytes(repository_root, commit, TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH)
    except V8BGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8B_TRUST_PIN_INDEPENDENT_REVIEW_MISSING") from error
    review = _strict_json_object(
        raw,
        invalid_reason="V8B_TRUST_PIN_INDEPENDENT_REVIEW_INVALID_JSON",
        duplicate_reason="V8B_TRUST_PIN_INDEPENDENT_REVIEW_DUPLICATE_KEY",
    )
    if set(review) != set(TRUST_PIN_INDEPENDENT_REVIEW_FIELDS):
        raise V8BProductionProvenanceBlocked("V8B_TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_INVALID")
    if review["schema_version"] != TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION:
        raise V8BProductionProvenanceBlocked("V8B_TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION_MISMATCH")
    if review["study"] != STUDY_NAME:
        raise V8BProductionProvenanceBlocked("V8B_TRUST_PIN_INDEPENDENT_REVIEW_STUDY_MISMATCH")
    if review["artifact_role"] != TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE:
        raise V8BProductionProvenanceBlocked("V8B_TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE_MISMATCH")
    if review["reviewed_allocation_artifact_self_hash"] != hash_value:
        raise V8BProductionProvenanceBlocked("V8B_TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_HASH_MISMATCH")
    if review["review_result"] != "PASS":
        raise V8BProductionProvenanceBlocked("V8B_TRUST_PIN_INDEPENDENT_REVIEW_NOT_PASS")
    if review["approval_status"] != "APPROVED":
        raise V8BProductionProvenanceBlocked("V8B_TRUST_PIN_INDEPENDENT_REVIEW_NOT_APPROVED")
    return dict(review)


__all__ = [
    "BOUND_PRODUCTION_FILES",
    "DESIGN_DRAFT_GIT_PATH",
    "DESIGN_FREEZE_APPROVAL_GIT_PATH",
    "EXPECTED_HUMAN_FREEZE_GATE",
    "EXPECTED_PARENT_T_SPARE_TICKER_COUNT",
    "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256",
    "EXPECTED_T2_OPTION_2_HUMAN_GATE",
    "EXPECTED_T2_TICKER_COUNT",
    "EXPECTED_T2_TICKER_LIST_SHA256",
    "EXPECTED_V8B_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT",
    "EXPECTED_V8B_DESIGN_FREEZE_APPROVAL_BLOB",
    "EXPECTED_V8B_FROZEN_DESIGN_COMMIT",
    "EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT",
    "EXPECTED_V8_PARTITION_MANIFEST_SHA256",
    "EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA",
    "IMPLEMENTATION_REVIEW_FIELDS",
    "IMPLEMENTATION_REVIEW_GIT_PATH",
    "IMPLEMENTATION_REVIEW_SCHEMA_VERSION",
    "STUDY_NAME",
    "T2_AUTHORITY_BRIDGE_FIELDS",
    "T2_AUTHORITY_BRIDGE_GIT_PATH",
    "T2_AUTHORITY_BRIDGE_SCHEMA_VERSION",
    "TRUSTED_PARTITION_ANCHOR_FIELDS",
    "TRUSTED_PARTITION_ANCHOR_GIT_PATH",
    "TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION",
    "TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE",
    "TRUST_PIN_INDEPENDENT_REVIEW_FIELDS",
    "TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH",
    "TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION",
    "V8BProductionProvenanceBlocked",
    "V8_DESIGN_COMMIT",
    "read_and_verify_design_freeze_approval",
    "read_and_verify_t2_authority_bridge",
    "read_and_verify_trust_pin_independent_review",
    "read_and_verify_v8_trusted_partition_anchor",
    "verify_frozen_design_object",
    "verify_reviewed_implementation_binding",
]
