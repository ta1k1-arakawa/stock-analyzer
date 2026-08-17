"""Exact-SHA verification of the frozen V8E authority bridges.

This module only verifies already-committed bridge and independent-review
artifacts.  It never creates a bridge, consumes a gate, reads private
partition data, or performs network I/O.
"""

from __future__ import annotations

import json
from typing import Any

from src.v8e_git_provenance import (
    V8EGitProvenanceBlocked,
    read_git_object_bytes,
    resolve_git_blob,
)

T1C_STAGE = "T1C_TRANSPORT_READINESS"
T2_STAGE = "T2_TRANSPORT_READINESS"

STUDY = "V8E_HISTORICAL_RESEARCH"
FROZEN_DESIGN_COMMIT = "6f672404b93a1003253915196dd635ca76fd2be1"

T1C_BRIDGE_PATH = "V8E_T1C_ALLOCATION_AUTHORITY_BRIDGE.json"
T1C_REVIEW_PATH = "V8E_T1C_ALLOCATION_AUTHORITY_BRIDGE_REVIEW.json"
T2_BRIDGE_PATH = "V8E_T2_AUTHORITY_BRIDGE.json"
T2_REVIEW_PATH = "V8E_T2_AUTHORITY_BRIDGE_REVIEW.json"

T1C_BRIDGE_SCHEMA = "V8E_T1C_ALLOCATION_AUTHORITY_BRIDGE_V1"
T2_BRIDGE_SCHEMA = "V8E_T2_AUTHORITY_BRIDGE_V1"
REVIEW_SCHEMA = "V8E_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_V1"
REVIEW_ROLE = "AUTHORITY_BRIDGE_INDEPENDENT_REVIEW"

V8C_TRUST_PIN_COMMIT = "2a65674d8439f5964ff694494d5dad5ed19ad0f6"
V8C_TRUST_PIN_BLOB = "61082f9818efb68ca2a5ad29fa5918f887575c10"
V8_TRUST_ANCHOR_BLOB = "61faade0625139cec3fb61216ab2f97f572a7028"
V8_PARTITION_MANIFEST_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
V8_PARTITION_IMPLEMENTATION_COMMIT = "36cbed941050e728f7f96ce2af505e81175cc02c"

T1C_PRESERVATION_COMMIT = "12a05d59daca7986e4dacb27bce63e073d064240"
T1C_PRESERVATION_BLOB = "cd084dd6e49be724e876d01b27ac45fa11a2dc64"
T1C_ALLOCATION_SELF_HASH = "16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c"
T1C_TICKER_LIST_SHA256 = "85a06d4b88698915315f5cf72e0d3e04dfacafb5403786ac3bb613e14b0deb54"
T1C_PARENT_SPARE_LIST_SHA256 = "360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70"

T2_PRESERVATION_COMMIT = "22071e3fceaff56ac2043f79e2d79d617f3658a5"
T2_PRESERVATION_BLOB = "24248bf96877ffb47bdba8fac7924684b1cae5cb"
T2_TICKER_LIST_SHA256 = "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500"

REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "logical_block",
    "reviewed_bridge_git_commit",
    "reviewed_bridge_git_blob_sha",
    "review_result",
)

T1C_BRIDGE_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "logical_block",
    "v8e_frozen_design_commit",
    "source_v8c_terminal_commit",
    "source_v8c_trust_pin_git_commit",
    "source_v8c_trust_pin_git_blob_sha",
    "authorized_allocation_artifact_self_hash",
    "t1c_ticker_count",
    "t1c_ticker_list_sha256",
    "parent_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "parent_t_spare_ticker_list_sha256",
    "preservation_recheck_git_commit",
    "preservation_recheck_git_blob_sha",
    "preservation_recheck_result",
    "human_gate",
    "authorization_status",
    "authorization_note",
)

T2_BRIDGE_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "logical_block",
    "v8e_frozen_design_commit",
    "source_authority",
    "v8_trust_anchor_git_identity",
    "authorized_parent_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "expected_t2_ticker_count",
    "expected_t2_ticker_list_sha256",
    "preservation_recheck_git_commit",
    "preservation_recheck_git_blob_sha",
    "preservation_recheck_result",
    "human_gate",
    "authorization_status",
    "authorization_note",
)


class V8EAuthorityBridgeBlocked(RuntimeError):
    """Fail-closed bridge or independent-review verification error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8EAuthorityBridgeBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8EAuthorityBridgeBlocked(invalid_reason) from error
    if not isinstance(value, dict):
        raise V8EAuthorityBridgeBlocked(invalid_reason)
    return value


def _read_git_json(repository_root, commit: str, path: str, *, label: str) -> dict[str, Any]:
    try:
        raw = read_git_object_bytes(repository_root, commit, path)
    except V8EGitProvenanceBlocked as error:
        raise V8EAuthorityBridgeBlocked(f"V8E_{label}_MISSING") from error
    return _strict_json_object(
        raw,
        invalid_reason=f"V8E_{label}_INVALID_JSON",
        duplicate_reason=f"V8E_{label}_DUPLICATE_KEY",
    )


def _require_exact_fields(document: dict[str, Any], fields: tuple[str, ...], reason: str) -> None:
    if set(document) != set(fields):
        raise V8EAuthorityBridgeBlocked(reason)


def _verify_review(
    repository_root, verified_head: str, *, logical_block: str, review_path: str
) -> tuple[str, str]:
    review = _read_git_json(
        repository_root, verified_head, review_path,
        label=f"{logical_block}_AUTHORITY_BRIDGE_REVIEW",
    )
    _require_exact_fields(review, REVIEW_FIELDS, f"V8E_{logical_block}_AUTHORITY_BRIDGE_REVIEW_SCHEMA_INVALID")
    if review["schema_version"] != REVIEW_SCHEMA:
        raise V8EAuthorityBridgeBlocked(f"V8E_{logical_block}_AUTHORITY_BRIDGE_REVIEW_SCHEMA_VERSION_MISMATCH")
    if review["study"] != STUDY:
        raise V8EAuthorityBridgeBlocked(f"V8E_{logical_block}_AUTHORITY_BRIDGE_REVIEW_STUDY_MISMATCH")
    if review["artifact_role"] != REVIEW_ROLE:
        raise V8EAuthorityBridgeBlocked(f"V8E_{logical_block}_AUTHORITY_BRIDGE_REVIEW_ROLE_MISMATCH")
    if review["logical_block"] != logical_block:
        raise V8EAuthorityBridgeBlocked(f"V8E_{logical_block}_AUTHORITY_BRIDGE_REVIEW_BLOCK_MISMATCH")
    if review["review_result"] != "PASS":
        raise V8EAuthorityBridgeBlocked(f"V8E_{logical_block}_AUTHORITY_BRIDGE_REVIEW_NOT_PASS")

    reviewed_commit = review["reviewed_bridge_git_commit"]
    if not isinstance(reviewed_commit, str) or len(reviewed_commit) != 40 or any(
        char not in "0123456789abcdef" for char in reviewed_commit
    ):
        raise V8EAuthorityBridgeBlocked(f"V8E_{logical_block}_AUTHORITY_BRIDGE_REVIEW_COMMIT_INVALID")
    reviewed_blob = review["reviewed_bridge_git_blob_sha"]
    if not isinstance(reviewed_blob, str) or len(reviewed_blob) != 40 or any(
        char not in "0123456789abcdef" for char in reviewed_blob
    ):
        raise V8EAuthorityBridgeBlocked(f"V8E_{logical_block}_AUTHORITY_BRIDGE_REVIEW_BLOB_INVALID")
    try:
        resolved_blob = resolve_git_blob(repository_root, reviewed_commit, _bridge_path(logical_block))
    except V8EGitProvenanceBlocked as error:
        raise V8EAuthorityBridgeBlocked(f"V8E_{logical_block}_AUTHORITY_BRIDGE_MISSING") from error
    if resolved_blob != reviewed_blob:
        raise V8EAuthorityBridgeBlocked(f"V8E_{logical_block}_AUTHORITY_BRIDGE_REVIEW_BLOB_MISMATCH")
    return reviewed_commit, reviewed_blob


def _bridge_path(logical_block: str) -> str:
    if logical_block == "T1C":
        return T1C_BRIDGE_PATH
    if logical_block == "T2":
        return T2_BRIDGE_PATH
    raise V8EAuthorityBridgeBlocked("V8E_AUTHORITY_BRIDGE_LOGICAL_BLOCK_INVALID")


def _verify_t1c_bridge(bridge: dict[str, Any]) -> None:
    _require_exact_fields(bridge, T1C_BRIDGE_FIELDS, "V8E_T1C_AUTHORITY_BRIDGE_SCHEMA_INVALID")
    expected = {
        "schema_version": T1C_BRIDGE_SCHEMA,
        "study": STUDY,
        "artifact_role": "T1C_ALLOCATION_AUTHORITY_BRIDGE",
        "logical_block": "T1C",
        "v8e_frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "source_v8c_terminal_commit": "d18368c1ec1c26d752ea5862115ab9f4315d1780",
        "source_v8c_trust_pin_git_commit": V8C_TRUST_PIN_COMMIT,
        "source_v8c_trust_pin_git_blob_sha": V8C_TRUST_PIN_BLOB,
        "authorized_allocation_artifact_self_hash": T1C_ALLOCATION_SELF_HASH,
        "t1c_ticker_count": 300,
        "t1c_ticker_list_sha256": T1C_TICKER_LIST_SHA256,
        "parent_v8_partition_manifest_sha256": V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": V8_PARTITION_IMPLEMENTATION_COMMIT,
        "parent_t_spare_ticker_list_sha256": T1C_PARENT_SPARE_LIST_SHA256,
        "preservation_recheck_git_commit": T1C_PRESERVATION_COMMIT,
        "preservation_recheck_git_blob_sha": T1C_PRESERVATION_BLOB,
        "preservation_recheck_result": "PASS",
        "human_gate": f"V8E_HUMAN_AUTHORIZE_T1C_AUTHORITY_BRIDGE_AT_{FROZEN_DESIGN_COMMIT}_FOR_{T1C_ALLOCATION_SELF_HASH}",
        "authorization_status": "AUTHORIZED",
    }
    for key, value in expected.items():
        if bridge[key] != value:
            raise V8EAuthorityBridgeBlocked(f"V8E_T1C_AUTHORITY_BRIDGE_{key.upper()}_MISMATCH")
    if not isinstance(bridge["authorization_note"], str) or not bridge["authorization_note"]:
        raise V8EAuthorityBridgeBlocked("V8E_T1C_AUTHORITY_BRIDGE_AUTHORIZATION_NOTE_INVALID")


def _verify_t2_bridge(bridge: dict[str, Any]) -> None:
    _require_exact_fields(bridge, T2_BRIDGE_FIELDS, "V8E_T2_AUTHORITY_BRIDGE_SCHEMA_INVALID")
    expected = {
        "schema_version": T2_BRIDGE_SCHEMA,
        "study": STUDY,
        "artifact_role": "T2_AUTHORITY_BRIDGE",
        "logical_block": "T2",
        "v8e_frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "source_authority": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY",
        "v8_trust_anchor_git_identity": V8_TRUST_ANCHOR_BLOB,
        "authorized_parent_v8_partition_manifest_sha256": V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": V8_PARTITION_IMPLEMENTATION_COMMIT,
        "expected_t2_ticker_count": 300,
        "expected_t2_ticker_list_sha256": T2_TICKER_LIST_SHA256,
        "preservation_recheck_git_commit": T2_PRESERVATION_COMMIT,
        "preservation_recheck_git_blob_sha": T2_PRESERVATION_BLOB,
        "preservation_recheck_result": "PASS",
        "human_gate": f"V8E_HUMAN_AUTHORIZE_T2_AUTHORITY_BRIDGE_AT_{FROZEN_DESIGN_COMMIT}_FOR_{T2_TICKER_LIST_SHA256}",
        "authorization_status": "AUTHORIZED",
    }
    for key, value in expected.items():
        if bridge[key] != value:
            raise V8EAuthorityBridgeBlocked(f"V8E_T2_AUTHORITY_BRIDGE_{key.upper()}_MISMATCH")
    if not isinstance(bridge["authorization_note"], str) or not bridge["authorization_note"]:
        raise V8EAuthorityBridgeBlocked("V8E_T2_AUTHORITY_BRIDGE_AUTHORIZATION_NOTE_INVALID")


def verify_stage_authority_bridge(repository_root, verified_head: str, logical_stage: str) -> dict[str, str]:
    """Verify the exact stage-specific bridge and its independent review.

    The review is read from ``verified_head``.  Its reviewed bridge commit and
    blob are then resolved independently, and the bridge bytes are read from
    that reviewed commit before any semantic PASS is accepted.
    """
    if logical_stage == T1C_STAGE:
        logical_block, review_path = "T1C", T1C_REVIEW_PATH
    elif logical_stage == T2_STAGE:
        logical_block, review_path = "T2", T2_REVIEW_PATH
    else:
        raise V8EAuthorityBridgeBlocked("V8E_AUTHORITY_BRIDGE_STAGE_INVALID")

    reviewed_commit, reviewed_blob = _verify_review(
        repository_root, verified_head, logical_block=logical_block, review_path=review_path
    )
    bridge = _read_git_json(
        repository_root, reviewed_commit, _bridge_path(logical_block),
        label=f"{logical_block}_AUTHORITY_BRIDGE",
    )
    if logical_block == "T1C":
        _verify_t1c_bridge(bridge)
    else:
        _verify_t2_bridge(bridge)
    return {
        "logical_block": logical_block,
        "reviewed_bridge_git_commit": reviewed_commit,
        "reviewed_bridge_git_blob_sha": reviewed_blob,
        "review_result": "PASS",
    }


__all__ = [
    "FROZEN_DESIGN_COMMIT",
    "T1C_BRIDGE_PATH",
    "T1C_REVIEW_PATH",
    "T2_BRIDGE_PATH",
    "T2_REVIEW_PATH",
    "V8EAuthorityBridgeBlocked",
    "verify_stage_authority_bridge",
]
