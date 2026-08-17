"""V8E T1C pre-freeze preservation support.

This module is the isolated support boundary authorized by V8E §9.A.  It
binds every V8E value to the independently reviewed design candidate and
keeps the only future private read behind a dependency-injected bytes
boundary.  Tests may supply synthetic bytes and temporary state; the public
entry point is never called by this implementation task.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v8_partition import (
    MANIFEST_FIELDS,
    SCHEMA_VERSION as V8_PARTITION_SCHEMA_VERSION,
    SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
    SOURCE_SNAPSHOT_SEMANTICS,
    V8PartitionBlocked,
    canonical_sha256 as v8_canonical_sha256,
    require_git_commit as require_v8_git_commit,
)
from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    read_git_object_bytes,
    resolve_git_blob,
)
from src.v8c_human_gate_consumption import CANONICAL_CONSUMPTION_STATE_ROOT
from src.v8c_production_provenance import (
    V8CProductionProvenanceBlocked,
    read_and_verify_v8_trusted_partition_anchor,
)
from src.v8c_t1c_allocation import (
    V8CAllocationBlocked,
    read_t1c_allocation_artifact_bytes,
)
from src.v8c_t1c_allocation_verification import (
    V8CAllocationVerificationBlocked,
    _verify_t1c_allocation_artifact,
)


V8E_STUDY_NAME = "V8E_HISTORICAL_RESEARCH"
V8E_PRODUCTION_BRANCH = "v8e-dq-evidence-successor-design"
V8E_REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT = "6f672404b93a1003253915196dd635ca76fd2be1"
V8E_DESIGN_CANDIDATE_BLOB_SHA = "dac32f9e97d1ae2b90eb8b0820914e3845d0fa26"
V8E_V8D_PREDECESSOR_TERMINAL_COMMIT = "b8f8d0d500d349ccaa5d3e49294f351dc53ea7e8"
V8E_V8D_TERMINAL_RECORD_GIT_PATH = "V8D_DQ_EVIDENCE_CONTRACT_BLOCK_ADJUDICATION.md"
V8E_V8D_TERMINAL_RECORD_BLOB_SHA = "f81106f529c339e6762e60d3075e03e790335610"
V8E_V8D_HISTORICAL_T1C_GIT_PATH = "V8D_T1C_PRESERVATION_RECHECK.json"
V8E_V8D_HISTORICAL_T1C_BLOB_SHA = "049becb3d2743ef68dc278f424484919ba379cca"
V8E_V8_STATE_GIT_PATH = "V8_STATE.json"
V8E_V8_STATE_BLOB_SHA = "8e5fd2f39dc92a7983c0cdaab42f633d624b4956"

# This is deliberately an exact, narrow classification.  A future commit
# outside these paths is not silently treated as harmless chronology.
V8E_PREFREEZE_CHRONOLOGY_SAFE_PATHS = frozenset(
    {
        "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md",
        "src/v8e_t1c_preservation.py",
        "src/v8e_t2_prefreeze_preservation.py",
        "tests/test_v8e_t1c_preservation.py",
        "tests/test_v8e_t2_prefreeze_preservation.py",
    }
)

V8E_T1C_PRESERVATION_GATE = "HUMAN_V8E_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE"
V8E_AUTHORIZATION_PREFIX = "V8E_HUMAN_AUTHORIZE_T1C_PRESERVATION_VERIFY_AT_"
V8E_AUTHORIZATION_SEPARATOR = "_FOR_"
V8E_RECEIPT_SCHEMA_VERSION = "V8E_T1C_PRESERVATION_GATE_RECEIPT_V1"
V8E_RECEIPT_ARTIFACT_ROLE = "T1C_PRESERVATION_PRIVATE_GATE_RECEIPT"
V8E_RECEIPT_CONSUMPTION_BOUNDARY = "IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ"
V8E_RECEIPT_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "gate",
    "reviewed_v8e_design_candidate_commit",
    "authorization_identity_sha256",
    "authorized_allocation_artifact_self_hash",
    "consumed",
    "consumption_count",
    "consumption_boundary",
    "consumption_timestamp_utc",
)

V8E_PRESERVATION_ARTIFACT_FIELDS = (
    "schema_version",
    "artifact_role",
    "study",
    "reviewed_v8e_design_candidate_commit",
    "source_v8c_terminal_commit",
    "allocation_artifact_self_hash",
    "t1c_ticker_count",
    "t1c_ticker_list_sha256",
    "parent_t_spare_ticker_list_sha256",
    "remaining_t_spare_ticker_list_sha256",
    "t1c_raw_acquisition_performed",
    "t1c_research_opened",
    "t1c_ohlcv_research_access",
    "t1c_feature_access",
    "t1c_outcome_access",
    "t1c_identities_publicly_exposed",
    "t1c_membership_reassigned",
    "allocation_self_hash_unchanged",
    "parent_v8_provenance_unchanged",
    "v8c_terminal_adjudication_authoritative",
    "preservation_recheck_result",
)

V8E_FRESH_T1C_PUBLIC_EVIDENCE_FIELDS = (
    "schema_version",
    "evidence_role",
    "study",
    "reviewed_v8e_design_candidate_commit",
    "v8d_predecessor_terminal_commit",
    "v8d_terminal_status",
    "v8d_terminal_failure_class",
    "v8d_terminal_implementation_head",
    "v8d_historical_t1c_artifact_blob_sha",
    "allocation_artifact_self_hash",
    "t1c_ticker_count",
    "t1c_ticker_list_sha256",
    "parent_t_spare_ticker_list_sha256",
    "remaining_t_spare_ticker_list_sha256",
    "t1c_raw_acquisition_performed",
    "t1c_research_opened",
    "t1c_ohlcv_research_access",
    "t1c_feature_access",
    "t1c_outcome_access",
    "t1c_identities_publicly_exposed",
    "t1c_membership_reassigned",
    "allocation_self_hash_unchanged",
    "parent_v8_provenance_unchanged",
    "v8c_terminal_adjudication_authoritative",
    "fresh_public_preservation_evidence_result",
)

# Historical V8/V8C commitments are safe public provenance constants.  They
# are not V8E authority and are never substituted for the V8E candidate.
AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH = "16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c"
EXPECTED_V8E_T1C_TICKER_COUNT = 300
EXPECTED_V8E_T1C_TICKER_LIST_SHA256 = "85a06d4b88698915315f5cf72e0d3e04dfacafb5403786ac3bb613e14b0deb54"
EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256 = "360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70"
EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256 = "699e7bc29b2714128de99203bd6fedb38ee24c6f7bfee7c725b605669c178632"
EXPECTED_PARENT_T_SPARE_TICKER_COUNT = 1904
EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA = "61faade0625139cec3fb61216ab2f97f572a7028"
EXPECTED_V8_PARTITION_MANIFEST_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT = "36cbed941050e728f7f96ce2af505e81175cc02c"
EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT = "f9c4bfcc9dab1845a6252ce7e5dc30441fec16ba"
EXPECTED_V8C_FROZEN_DESIGN_COMMIT = "c9c541ac7f7ba3bcca76db6250fe8273d9bb5756"
V8C_TERMINAL_COMMIT = "d18368c1ec1c26d752ea5862115ab9f4315d1780"
V8C_TERMINAL_ADJUDICATION_GIT_PATH = "V8C_T1C_READINESS_BLOCK_ADJUDICATION.md"
V8C_TERMINAL_ADJUDICATION_BLOB_SHA = "d40b3ef6b071b150dab8269044398fd6fc8227c5"
V8C_PREFREEZE_AUDIT_GIT_PATH = "V8C_PREFREEZE_PRESERVATION_RECHECK.md"
V8C_PREFREEZE_AUDIT_BLOB_SHA = "ec9054caf94898948879b599196c055e480d2e52"
V8C_TRUSTED_ALLOCATION_GIT_PATH = "V8C_TRUSTED_ALLOCATION.json"
V8C_TRUSTED_ALLOCATION_BLOB_SHA = "61082f9818efb68ca2a5ad29fa5918f887575c10"

CANONICAL_V8E_T1C_PRESERVATION_STATE_ROOT = (
    CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8e-t1c-preservation-gate-state"
)

_HEX = re.compile(r"^[0-9a-f]+$")
_TIMESTAMP_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_TIMESTAMP_MICROS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_REQUIRED_BLOCK_KEYS = ("T0", "T1", "T2", "T3", "T_spare")


class V8ET1CPreservationBlocked(RuntimeError):
    """Fail-closed V8E T1C preservation support error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise V8ET1CPreservationBlocked(reason)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V8ET1CPreservationBlocked("V8E_PRESERVATION_NONFINITE_OR_UNSERIALIZABLE") from error


def authorization_identity_sha256(authorization_identity: str) -> str:
    """Return only the SHA-256; never persist or expose the raw identity."""
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8ET1CPreservationBlocked("V8E_AUTHORIZATION_IDENTITY_REQUIRED")
    return hashlib.sha256(authorization_identity.encode("utf-8")).hexdigest()


def validate_authorization_identity(
    authorization_identity: str,
    reviewed_v8e_design_candidate_commit: str = V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    authorized_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
) -> None:
    """Require the exact V8E §3A.4 identity grammar and candidate binding."""
    candidate = _require_hex(
        reviewed_v8e_design_candidate_commit, 40, "V8E_DESIGN_COMMIT_INVALID"
    )
    allocation_hash = _require_hex(
        authorized_allocation_artifact_self_hash, 64, "V8E_ALLOCATION_HASH_INVALID"
    )
    if candidate != V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT:
        raise V8ET1CPreservationBlocked("V8E_DESIGN_CANDIDATE_MISMATCH")
    if allocation_hash != AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH:
        raise V8ET1CPreservationBlocked("V8E_ALLOCATION_HASH_MISMATCH")
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8ET1CPreservationBlocked("V8E_AUTHORIZATION_GRAMMAR_MISMATCH")
    expected = V8E_AUTHORIZATION_PREFIX + candidate + V8E_AUTHORIZATION_SEPARATOR + allocation_hash
    if authorization_identity != expected:
        raise V8ET1CPreservationBlocked("V8E_AUTHORIZATION_GRAMMAR_MISMATCH")


def compute_receipt_key(
    authorization_identity: str,
    reviewed_v8e_design_candidate_commit: str = V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    authorized_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
) -> str:
    validate_authorization_identity(
        authorization_identity,
        reviewed_v8e_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    material = "|".join(
        (
            V8E_REPOSITORY_IDENTITY,
            V8E_T1C_PRESERVATION_GATE,
            reviewed_v8e_design_candidate_commit,
            authorization_identity_sha256(authorization_identity),
            authorized_allocation_artifact_self_hash,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _receipt_path(state_root: str | os.PathLike[str], receipt_key: str) -> Path:
    _require_hex(receipt_key, 64, "V8E_RECEIPT_KEY_INVALID")
    return Path(state_root) / (receipt_key + ".json")


def _timestamp_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_CLOCK_INVALID")
    utc = value.astimezone(timezone.utc)
    if utc.microsecond:
        return utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or not (_TIMESTAMP_SECONDS.fullmatch(value) or _TIMESTAMP_MICROS.fullmatch(value)):
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_TIMESTAMP_INVALID")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ" if "." not in value else "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as error:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_TIMESTAMP_INVALID") from error
    return value


def _strict_json_object(raw: bytes, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8ET1CPreservationBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8ET1CPreservationBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8ET1CPreservationBlocked(invalid_reason)
    return parsed


def _validate_receipt(receipt: Mapping[str, Any], receipt_key: str) -> dict[str, Any]:
    if set(receipt) != set(V8E_RECEIPT_FIELDS):
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_SCHEMA_INVALID")
    if receipt["schema_version"] != V8E_RECEIPT_SCHEMA_VERSION:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_SCHEMA_VERSION_MISMATCH")
    if receipt["study"] != V8E_STUDY_NAME or receipt["artifact_role"] != V8E_RECEIPT_ARTIFACT_ROLE:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_IDENTITY_INVALID")
    if receipt["gate"] != V8E_T1C_PRESERVATION_GATE:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_GATE_INVALID")
    if receipt["reviewed_v8e_design_candidate_commit"] != V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_DESIGN_COMMIT_INVALID")
    _require_hex(receipt["authorization_identity_sha256"], 64, "V8E_RECEIPT_IDENTITY_HASH_INVALID")
    if receipt["authorized_allocation_artifact_self_hash"] != AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_ALLOCATION_HASH_INVALID")
    if receipt["consumed"] is not True or type(receipt["consumption_count"]) is not int or receipt["consumption_count"] != 1:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_CONSUMPTION_INVALID")
    if receipt["consumption_boundary"] != V8E_RECEIPT_CONSUMPTION_BOUNDARY:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_CONSUMPTION_BOUNDARY_INVALID")
    _validate_timestamp(receipt["consumption_timestamp_utc"])
    material = "|".join(
        (
            V8E_REPOSITORY_IDENTITY,
            receipt["gate"],
            receipt["reviewed_v8e_design_candidate_commit"],
            receipt["authorization_identity_sha256"],
            receipt["authorized_allocation_artifact_self_hash"],
        )
    )
    if hashlib.sha256(material.encode("utf-8")).hexdigest() != receipt_key:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_KEY_CONTENT_MISMATCH")
    return dict(receipt)


def read_gate_receipt(state_root: str | os.PathLike[str], receipt_key: str) -> dict[str, Any]:
    path = _receipt_path(state_root, receipt_key)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_MISSING") from error
    return _validate_receipt(
        _strict_json_object(raw, "V8E_RECEIPT_INVALID_JSON", "V8E_RECEIPT_DUPLICATE_KEY"), receipt_key
    )


def gate_receipt_bytes_sha256(state_root: str | os.PathLike[str], receipt_key: str) -> str:
    """Validate and hash the exact durable receipt bytes externally."""
    path = _receipt_path(state_root, receipt_key)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_MISSING") from error
    _validate_receipt(
        _strict_json_object(raw, "V8E_RECEIPT_INVALID_JSON", "V8E_RECEIPT_DUPLICATE_KEY"), receipt_key
    )
    return hashlib.sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def consume_gate_once(
    state_root: str | os.PathLike[str],
    authorization_identity: str,
    *,
    clock: Callable[[], datetime],
    reviewed_v8e_design_candidate_commit: str = V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    authorized_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
) -> dict[str, Any]:
    """Durably publish exactly one V8E receipt; no reset/replay API exists."""
    validate_authorization_identity(
        authorization_identity,
        reviewed_v8e_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    receipt_key = compute_receipt_key(
        authorization_identity,
        reviewed_v8e_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    root = Path(state_root)
    path = _receipt_path(root, receipt_key)
    if path.exists():
        raise V8ET1CPreservationBlocked("V8E_GATE_ALREADY_CONSUMED")
    receipt = {
        "schema_version": V8E_RECEIPT_SCHEMA_VERSION,
        "study": V8E_STUDY_NAME,
        "artifact_role": V8E_RECEIPT_ARTIFACT_ROLE,
        "gate": V8E_T1C_PRESERVATION_GATE,
        "reviewed_v8e_design_candidate_commit": reviewed_v8e_design_candidate_commit,
        "authorization_identity_sha256": authorization_identity_sha256(authorization_identity),
        "authorized_allocation_artifact_self_hash": authorized_allocation_artifact_self_hash,
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": V8E_RECEIPT_CONSUMPTION_BOUNDARY,
        "consumption_timestamp_utc": _timestamp_text(clock()),
    }
    payload = _canonical_json_bytes(receipt)
    try:
        root.mkdir(parents=True, exist_ok=True)
        staging = root / (path.name + ".staging-" + os.urandom(8).hex())
        with open(staging, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, path)
        except FileExistsError as error:
            raise V8ET1CPreservationBlocked("V8E_GATE_ALREADY_CONSUMED") from error
        except OSError as error:
            raise V8ET1CPreservationBlocked("V8E_RECEIPT_STORAGE_WRITE_FAILED") from error
        _fsync_directory(root)
    except V8ET1CPreservationBlocked:
        raise
    except OSError as error:
        raise V8ET1CPreservationBlocked("V8E_RECEIPT_STORAGE_WRITE_FAILED") from error
    finally:
        staging_path = locals().get("staging")
        if isinstance(staging_path, Path):
            try:
                if staging_path.exists():
                    staging_path.unlink()
            except OSError:
                pass
    return dict(receipt)


def _require_safe_external_path(value: str | os.PathLike[str], repository_root: Path, reason: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise V8ET1CPreservationBlocked(reason)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError:
        return resolved
    raise V8ET1CPreservationBlocked(reason)


def _prepare_execution_paths(
    *,
    state_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    repository_root: Path,
    receipt_key: str,
) -> tuple[Path, Path, Path, Path]:
    state = _require_safe_external_path(state_root, repository_root, "V8E_STATE_PATH_INVALID")
    output = _require_safe_external_path(output_path, repository_root, "V8E_OUTPUT_PATH_INVALID")
    allocation = _require_safe_external_path(allocation_artifact_path, repository_root, "V8E_PRIVATE_PATH_INVALID")
    manifest = _require_safe_external_path(partition_manifest_path, repository_root, "V8E_PRIVATE_PATH_INVALID")
    if allocation == manifest or output in {allocation, manifest} or output == state / (receipt_key + ".json"):
        raise V8ET1CPreservationBlocked("V8E_OUTPUT_PATH_COLLISION")
    try:
        state.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8ET1CPreservationBlocked("V8E_OUTPUT_OR_STATE_PREPARATION_FAILED") from error
    if not state.is_dir() or not output.parent.is_dir() or output.exists():
        raise V8ET1CPreservationBlocked("V8E_OUTPUT_OR_STATE_PREPARATION_FAILED")
    if not allocation.is_file() or not manifest.is_file():
        raise V8ET1CPreservationBlocked("V8E_PRIVATE_ARTIFACT_UNAVAILABLE")
    if (state / (receipt_key + ".json")).exists():
        raise V8ET1CPreservationBlocked("V8E_GATE_ALREADY_CONSUMED")
    return state, output, allocation, manifest


_PUBLIC_PREFLIGHT_FIELDS = frozenset(
    {
        "repository_identity",
        "branch",
        "head",
        "origin_head",
        "worktree_clean",
        "reviewed_v8e_design_candidate_commit",
        "reviewed_v8e_design_blob_sha",
        "v8c_terminal_commit",
        "v8c_terminal_blob_sha",
        "v8c_prefreeze_blob_sha",
        "trusted_partition_blob_sha",
        "partition_manifest_sha256",
        "partition_implementation_commit",
        "v8c_allocation_implementation_commit",
        "parent_t_spare_ticker_count",
        "parent_t_spare_ticker_list_sha256",
    }
)


def _validate_public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(preflight, Mapping) or set(preflight) != _PUBLIC_PREFLIGHT_FIELDS:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_PREFLIGHT_SCHEMA_INVALID")
    if preflight["repository_identity"] != V8E_REPOSITORY_IDENTITY:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    if preflight["branch"] != V8E_PRODUCTION_BRANCH or preflight["worktree_clean"] is not True:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_GIT_BINDING_INVALID")
    head = _require_hex(preflight["head"], 40, "V8E_PUBLIC_HEAD_INVALID")
    if preflight["origin_head"] != head:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_HEAD_NOT_ORIGIN")
    if preflight["reviewed_v8e_design_candidate_commit"] != V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT:
        raise V8ET1CPreservationBlocked("V8E_DESIGN_CANDIDATE_MISMATCH")
    if preflight["reviewed_v8e_design_blob_sha"] != V8E_DESIGN_CANDIDATE_BLOB_SHA:
        raise V8ET1CPreservationBlocked("V8E_DESIGN_CANDIDATE_BLOB_MISMATCH")
    if preflight["v8c_terminal_commit"] != V8C_TERMINAL_COMMIT or preflight["v8c_terminal_blob_sha"] != V8C_TERMINAL_ADJUDICATION_BLOB_SHA:
        raise V8ET1CPreservationBlocked("V8E_V8C_TERMINAL_ADJUDICATION_INVALID")
    if preflight["v8c_prefreeze_blob_sha"] != V8C_PREFREEZE_AUDIT_BLOB_SHA:
        raise V8ET1CPreservationBlocked("V8E_V8C_PREFREEZE_AUDIT_INVALID")
    if preflight["trusted_partition_blob_sha"] != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8ET1CPreservationBlocked("V8E_TRUSTED_PARTITION_ANCHOR_INVALID")
    if preflight["partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8ET1CPreservationBlocked("V8E_PARTITION_MANIFEST_BINDING_INVALID")
    if preflight["partition_implementation_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8ET1CPreservationBlocked("V8E_PARTITION_IMPLEMENTATION_BINDING_INVALID")
    if preflight["v8c_allocation_implementation_commit"] != EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT:
        raise V8ET1CPreservationBlocked("V8E_ALLOCATION_IMPLEMENTATION_BINDING_INVALID")
    if preflight["parent_t_spare_ticker_count"] != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
        raise V8ET1CPreservationBlocked("V8E_PARENT_T_SPARE_COUNT_INVALID")
    if preflight["parent_t_spare_ticker_list_sha256"] != EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256:
        raise V8ET1CPreservationBlocked("V8E_PARENT_T_SPARE_HASH_INVALID")
    return dict(preflight)


def _git_text(repository_root: Path, args: list[str], reason: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_GIT_UNAVAILABLE") from error
    if result.returncode != 0:
        raise V8ET1CPreservationBlocked(reason)
    return result.stdout.strip()


def _strict_public_json(raw: bytes, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    return _strict_json_object(raw, invalid_reason, duplicate_reason)


def _require_exact_public_fields(value: Mapping[str, Any], fields: frozenset[str], reason: str) -> None:
    if set(value) != fields:
        raise V8ET1CPreservationBlocked(reason)


def _read_public_git_json(repository_root: Path, ref: str, git_path: str, *, invalid: str, duplicate: str) -> dict[str, Any]:
    try:
        raw = read_git_object_bytes(repository_root, ref, git_path)
    except V8CGitProvenanceBlocked as error:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_PROVENANCE_INVALID") from error
    return _strict_public_json(raw, invalid, duplicate)


def _parse_v8d_terminal_record(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V8ET1CPreservationBlocked("V8E_V8D_TERMINAL_RECORD_INVALID") from error
    values: dict[str, str] = {}
    for key in ("study", "terminal_status", "failure_class", "terminal_implementation_head"):
        matches = re.findall(r"(?m)^" + re.escape(key) + r"=([^\r\n]+)$", text)
        if len(matches) != 1:
            raise V8ET1CPreservationBlocked("V8E_V8D_TERMINAL_RECORD_INVALID")
        values[key] = matches[0]
    if values != {
        "study": "V8D_HISTORICAL_RESEARCH",
        "terminal_status": "BLOCK_CLOSED",
        "failure_class": "DESIGN_AUDITABILITY_FAILURE",
        "terminal_implementation_head": "a862efec34dcf4a89005c88b55b35c39be12b7bc",
    }:
        raise V8ET1CPreservationBlocked("V8E_V8D_TERMINAL_BINDING_INVALID")
    required_absence_evidence = (
        "No T1C/T2 outcomes or features were observed.",
        "accessed no ticker identities",
        "opened no research data",
    )
    if any(sentence not in text for sentence in required_absence_evidence):
        raise V8ET1CPreservationBlocked("V8E_V8D_TERMINAL_ABSENCE_EVIDENCE_INVALID")
    values.update(
        {
            "t1c_feature_access": False,
            "t1c_outcome_access": False,
            "t1c_identities_publicly_exposed": False,
        }
    )
    return values


_V8D_HISTORICAL_T1C_FIELDS = frozenset(
    {
        "allocation_artifact_self_hash",
        "allocation_self_hash_unchanged",
        "artifact_role",
        "parent_t_spare_ticker_list_sha256",
        "parent_v8_provenance_unchanged",
        "preservation_recheck_result",
        "remaining_t_spare_ticker_list_sha256",
        "reviewed_design_candidate_commit",
        "schema_version",
        "source_v8c_terminal_commit",
        "study",
        "t1c_feature_access",
        "t1c_identities_publicly_exposed",
        "t1c_membership_reassigned",
        "t1c_ohlcv_research_access",
        "t1c_outcome_access",
        "t1c_raw_acquisition_performed",
        "t1c_research_opened",
        "t1c_ticker_count",
        "t1c_ticker_list_sha256",
        "v8c_terminal_adjudication_authoritative",
    }
)


def _validate_historical_v8d_t1c_record(record: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_public_fields(record, _V8D_HISTORICAL_T1C_FIELDS, "V8E_V8D_HISTORICAL_T1C_SCHEMA_INVALID")
    exact = {
        "allocation_artifact_self_hash": AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "allocation_self_hash_unchanged": True,
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "parent_t_spare_ticker_list_sha256": EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "parent_v8_provenance_unchanged": True,
        "preservation_recheck_result": "PASS",
        "remaining_t_spare_ticker_list_sha256": EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "reviewed_design_candidate_commit": "eda657cde2383718d986c4c4bfaae794784fe04d",
        "schema_version": "V8D_T1C_PRESERVATION_RECHECK_V1",
        "source_v8c_terminal_commit": V8C_TERMINAL_COMMIT,
        "study": "V8D_HISTORICAL_RESEARCH",
        "t1c_feature_access": False,
        "t1c_identities_publicly_exposed": False,
        "t1c_membership_reassigned": False,
        "t1c_ohlcv_research_access": False,
        "t1c_outcome_access": False,
        "t1c_raw_acquisition_performed": False,
        "t1c_research_opened": False,
        "t1c_ticker_count": EXPECTED_V8E_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": EXPECTED_V8E_T1C_TICKER_LIST_SHA256,
        "v8c_terminal_adjudication_authoritative": True,
    }
    for key, expected in exact.items():
        if record[key] != expected or (isinstance(expected, bool) and type(record[key]) is not bool):
            raise V8ET1CPreservationBlocked("V8E_V8D_HISTORICAL_T1C_VALUE_INVALID:" + key)
    return dict(record)


_V8C_TRUSTED_ALLOCATION_FIELDS = frozenset(
    {
        "artifact_role",
        "authorization_note",
        "authorization_status",
        "authorized_allocation_artifact_self_hash",
        "human_gate",
        "logical_block",
        "parent_t_spare_ticker_count",
        "parent_t_spare_ticker_list_sha256",
        "parent_v8_partition_implementation_commit",
        "parent_v8_partition_manifest_sha256",
        "predecessor_burned_count",
        "remaining_t_spare_ticker_count",
        "remaining_t_spare_ticker_list_sha256",
        "schema_version",
        "study_name",
        "t1c_ticker_count",
        "t1c_ticker_list_sha256",
        "v8c_allocation_implementation_commit",
        "v8c_frozen_design_commit",
        "v8c_reviewed_production_implementation_commit",
        "verification_result",
    }
)


def _validate_current_trusted_t1c_allocation(record: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_public_fields(record, _V8C_TRUSTED_ALLOCATION_FIELDS, "V8E_TRUSTED_ALLOCATION_SCHEMA_INVALID")
    exact = {
        "artifact_role": "TRUSTED_T1C_ALLOCATION_PIN",
        "authorization_status": "AUTHORIZED",
        "authorized_allocation_artifact_self_hash": AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "logical_block": "T1C",
        "parent_t_spare_ticker_count": EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
        "parent_t_spare_ticker_list_sha256": EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "parent_v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "parent_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "predecessor_burned_count": 300,
        "remaining_t_spare_ticker_count": 1304,
        "remaining_t_spare_ticker_list_sha256": EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "schema_version": "V8C_TRUSTED_ALLOCATION_V1",
        "study_name": "V8C_HISTORICAL_RESEARCH",
        "t1c_ticker_count": EXPECTED_V8E_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": EXPECTED_V8E_T1C_TICKER_LIST_SHA256,
        "v8c_allocation_implementation_commit": EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT,
        "v8c_frozen_design_commit": EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        "v8c_reviewed_production_implementation_commit": EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT,
        "verification_result": "PASS",
    }
    for key, expected in exact.items():
        if record[key] != expected:
            raise V8ET1CPreservationBlocked("V8E_TRUSTED_ALLOCATION_VALUE_INVALID:" + key)
    return dict(record)


def _default_public_chronology(repository_root: Path, lower: str, upper: str) -> list[dict[str, Any]]:
    if _git_text(repository_root, ["merge-base", "--is-ancestor", lower, upper], "V8E_PUBLIC_CHRONOLOGY_INVALID") != "":
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_CHRONOLOGY_INVALID")
    commits_text = _git_text(repository_root, ["rev-list", "--reverse", f"{lower}..{upper}"], "V8E_PUBLIC_CHRONOLOGY_INVALID")
    records: list[dict[str, Any]] = []
    for commit in commits_text.splitlines():
        paths_text = _git_text(
            repository_root,
            ["diff-tree", "--no-commit-id", "--name-only", "-r", commit],
            "V8E_PUBLIC_CHRONOLOGY_INVALID",
        )
        paths = [path for path in paths_text.splitlines() if path]
        records.append({"commit": commit, "paths": paths})
    return records


def _validate_public_chronology(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_CHRONOLOGY_INVALID")
    validated: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"commit", "paths"}:
            raise V8ET1CPreservationBlocked("V8E_PUBLIC_CHRONOLOGY_UNVERIFIABLE")
        _require_hex(record["commit"], 40, "V8E_PUBLIC_CHRONOLOGY_UNVERIFIABLE")
        paths = record["paths"]
        if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
            raise V8ET1CPreservationBlocked("V8E_PUBLIC_CHRONOLOGY_UNVERIFIABLE")
        if len(set(paths)) != len(paths) or any(path not in V8E_PREFREEZE_CHRONOLOGY_SAFE_PATHS for path in paths):
            raise V8ET1CPreservationBlocked("V8E_PUBLIC_CHRONOLOGY_UNCLASSIFIED_CHANGE")
        validated.append({"commit": record["commit"], "paths": list(paths)})
    return validated


def _validate_current_v8_state(state: Mapping[str, Any]) -> None:
    if state.get("schema_version") != "V8_STATE_SNAPSHOT_V1" or state.get("study") != "V8_HISTORICAL_RESEARCH":
        raise V8ET1CPreservationBlocked("V8E_V8_STATE_BINDING_INVALID")
    partition = state.get("partition")
    t1 = state.get("T1")
    attempt = state.get("last_real_t1_acquisition_attempt")
    if not isinstance(partition, Mapping) or not isinstance(t1, Mapping) or not isinstance(attempt, Mapping):
        raise V8ET1CPreservationBlocked("V8E_V8_STATE_SCHEMA_INVALID")
    partition_exact = {
        "real_partition_manifest_exists": True,
        "real_partition_manifest_validated": True,
        "trusted_partition_authorized": True,
        "manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "partition_implementation_git_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "manifest_schema_version": "V8_PARTITION_MANIFEST_V3",
        "t1_ticker_list_sha256": "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d",
        "t2_ticker_list_sha256": "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500",
        "t_spare_ticker_list_sha256": EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "block_assignments_recorded": False,
        "block_size_frozen": EXPECTED_V8E_T1C_TICKER_COUNT,
    }
    t1_exact = {
        "raw_data_acquired": False,
        "real_acquisition_authorized": False,
        "raw_bundle_exists": False,
        "research_access_authorized": False,
        "ticker_count_frozen": EXPECTED_V8E_T1C_TICKER_COUNT,
        "validation_access_count": None,
        "layer_b_opened": False,
    }
    attempt_exact = {
        "result": "BLOCKED",
        "retry_performed": False,
        "t1_final_bundle_exists": False,
        "t1_opened_for_research": False,
        "t1_successfully_acquired": False,
        "validation_accessed": False,
        "attempt_3_authorized": False,
    }
    for key, expected in partition_exact.items():
        if partition.get(key) != expected or (isinstance(expected, bool) and type(partition.get(key)) is not bool):
            raise V8ET1CPreservationBlocked("V8E_V8_STATE_PARTITION_INVALID:" + key)
    for key, expected in t1_exact.items():
        if t1.get(key) != expected or (isinstance(expected, bool) and type(t1.get(key)) is not bool):
            raise V8ET1CPreservationBlocked("V8E_V8_STATE_T1_INVALID:" + key)
    for key, expected in attempt_exact.items():
        if attempt.get(key) != expected or (isinstance(expected, bool) and type(attempt.get(key)) is not bool):
            raise V8ET1CPreservationBlocked("V8E_V8_STATE_ATTEMPT_INVALID:" + key)


def _validate_fresh_t1c_public_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    _require_exact_public_fields(evidence, frozenset(V8E_FRESH_T1C_PUBLIC_EVIDENCE_FIELDS), "V8E_FRESH_EVIDENCE_SCHEMA_INVALID")
    exact = {
        "schema_version": "V8E_T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE_V1",
        "evidence_role": "T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE",
        "study": V8E_STUDY_NAME,
        "reviewed_v8e_design_candidate_commit": V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "v8d_predecessor_terminal_commit": V8E_V8D_PREDECESSOR_TERMINAL_COMMIT,
        "v8d_terminal_status": "BLOCK_CLOSED",
        "v8d_terminal_failure_class": "DESIGN_AUDITABILITY_FAILURE",
        "v8d_terminal_implementation_head": "a862efec34dcf4a89005c88b55b35c39be12b7bc",
        "v8d_historical_t1c_artifact_blob_sha": V8E_V8D_HISTORICAL_T1C_BLOB_SHA,
        "allocation_artifact_self_hash": AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": EXPECTED_V8E_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": EXPECTED_V8E_T1C_TICKER_LIST_SHA256,
        "parent_t_spare_ticker_list_sha256": EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "remaining_t_spare_ticker_list_sha256": EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "t1c_raw_acquisition_performed": False,
        "t1c_research_opened": False,
        "t1c_ohlcv_research_access": False,
        "t1c_feature_access": False,
        "t1c_outcome_access": False,
        "t1c_identities_publicly_exposed": False,
        "t1c_membership_reassigned": False,
        "allocation_self_hash_unchanged": True,
        "parent_v8_provenance_unchanged": True,
        "v8c_terminal_adjudication_authoritative": True,
        "fresh_public_preservation_evidence_result": "PASS",
    }
    for key, expected in exact.items():
        if evidence[key] != expected or (isinstance(expected, bool) and type(evidence[key]) is not bool):
            raise V8ET1CPreservationBlocked("V8E_FRESH_EVIDENCE_VALUE_INVALID:" + key)
    return dict(evidence)


def _default_fresh_t1c_public_evidence(
    repository_root: Path,
    preflight: Mapping[str, Any],
    *,
    chronology_reader: Callable[[Path, str, str], Any] | None = None,
) -> dict[str, Any]:
    head = _require_hex(preflight["head"], 40, "V8E_PUBLIC_HEAD_INVALID")
    if _git_text(repository_root, ["rev-parse", "HEAD"], "V8E_PUBLIC_HEAD_UNAVAILABLE") != head:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_HEAD_CHANGED")
    try:
        design_blob = resolve_git_blob(repository_root, V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT, "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md")
        current_design_blob = resolve_git_blob(repository_root, head, "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md")
        v8d_terminal_blob = resolve_git_blob(repository_root, V8E_V8D_PREDECESSOR_TERMINAL_COMMIT, V8E_V8D_TERMINAL_RECORD_GIT_PATH)
        historical_t1c_blob = resolve_git_blob(repository_root, V8E_V8D_PREDECESSOR_TERMINAL_COMMIT, V8E_V8D_HISTORICAL_T1C_GIT_PATH)
        v8c_terminal_blob = resolve_git_blob(repository_root, V8C_TERMINAL_COMMIT, V8C_TERMINAL_ADJUDICATION_GIT_PATH)
        v8c_prefreeze_blob = resolve_git_blob(repository_root, V8C_TERMINAL_COMMIT, V8C_PREFREEZE_AUDIT_GIT_PATH)
        trusted_allocation_blob = resolve_git_blob(repository_root, head, V8C_TRUSTED_ALLOCATION_GIT_PATH)
        state_blob = resolve_git_blob(repository_root, head, V8E_V8_STATE_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_PROVENANCE_INVALID") from error
    if design_blob != V8E_DESIGN_CANDIDATE_BLOB_SHA or current_design_blob != V8E_DESIGN_CANDIDATE_BLOB_SHA:
        raise V8ET1CPreservationBlocked("V8E_DESIGN_CANDIDATE_BLOB_MISMATCH")
    if v8d_terminal_blob != V8E_V8D_TERMINAL_RECORD_BLOB_SHA or historical_t1c_blob != V8E_V8D_HISTORICAL_T1C_BLOB_SHA:
        raise V8ET1CPreservationBlocked("V8E_V8D_HISTORICAL_BLOB_MISMATCH")
    if v8c_terminal_blob != V8C_TERMINAL_ADJUDICATION_BLOB_SHA or v8c_prefreeze_blob != V8C_PREFREEZE_AUDIT_BLOB_SHA:
        raise V8ET1CPreservationBlocked("V8E_V8C_PROVENANCE_BLOB_MISMATCH")
    if trusted_allocation_blob != V8C_TRUSTED_ALLOCATION_BLOB_SHA or state_blob != V8E_V8_STATE_BLOB_SHA:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_PROVENANCE_INVALID")
    try:
        anchor = read_and_verify_v8_trusted_partition_anchor(repository_root, head)
    except (V8CGitProvenanceBlocked, V8CProductionProvenanceBlocked, V8PartitionBlocked) as error:
        raise V8ET1CPreservationBlocked("V8E_TRUSTED_PARTITION_ANCHOR_INVALID") from error
    if (
        anchor.get("authorized_partition_manifest_sha256") != EXPECTED_V8_PARTITION_MANIFEST_SHA256
        or anchor.get("authorized_partition_implementation_git_commit") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT
    ):
        raise V8ET1CPreservationBlocked("V8E_TRUSTED_PARTITION_ANCHOR_INVALID")
    terminal = _parse_v8d_terminal_record(read_git_object_bytes(repository_root, V8E_V8D_PREDECESSOR_TERMINAL_COMMIT, V8E_V8D_TERMINAL_RECORD_GIT_PATH))
    historical = _validate_historical_v8d_t1c_record(
        _read_public_git_json(repository_root, V8E_V8D_PREDECESSOR_TERMINAL_COMMIT, V8E_V8D_HISTORICAL_T1C_GIT_PATH, invalid="V8E_V8D_HISTORICAL_T1C_INVALID_JSON", duplicate="V8E_V8D_HISTORICAL_T1C_DUPLICATE_KEY")
    )
    allocation = _validate_current_trusted_t1c_allocation(
        _read_public_git_json(repository_root, head, V8C_TRUSTED_ALLOCATION_GIT_PATH, invalid="V8E_TRUSTED_ALLOCATION_INVALID_JSON", duplicate="V8E_TRUSTED_ALLOCATION_DUPLICATE_KEY")
    )
    state = _read_public_git_json(repository_root, head, V8E_V8_STATE_GIT_PATH, invalid="V8E_V8_STATE_INVALID_JSON", duplicate="V8E_V8_STATE_DUPLICATE_KEY")
    _validate_current_v8_state(state)
    chronology = _validate_public_chronology(
        (chronology_reader or _default_public_chronology)(repository_root, V8E_V8D_PREDECESSOR_TERMINAL_COMMIT, head)
    )
    if not chronology:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_CHRONOLOGY_INVALID")
    if chronology_reader is None:
        for record in chronology:
            if "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md" in record["paths"]:
                if _git_text(
                    repository_root,
                    ["merge-base", "--is-ancestor", record["commit"], V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT],
                    "V8E_PUBLIC_CHRONOLOGY_UNVERIFIABLE",
                ) != "":
                    raise V8ET1CPreservationBlocked("V8E_PUBLIC_CHRONOLOGY_CONTRADICTION")
    # Every current absence is a conjunction of the historical boundary
    # evidence, the current safe state, and the verified absence of a
    # preservation-relevant committed change in the V8E interval.
    evidence = {
        "schema_version": "V8E_T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE_V1",
        "evidence_role": "T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE",
        "study": V8E_STUDY_NAME,
        "reviewed_v8e_design_candidate_commit": V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "v8d_predecessor_terminal_commit": V8E_V8D_PREDECESSOR_TERMINAL_COMMIT,
        "v8d_terminal_status": terminal["terminal_status"],
        "v8d_terminal_failure_class": terminal["failure_class"],
        "v8d_terminal_implementation_head": terminal["terminal_implementation_head"],
        "v8d_historical_t1c_artifact_blob_sha": historical_t1c_blob,
        "allocation_artifact_self_hash": allocation["authorized_allocation_artifact_self_hash"],
        "t1c_ticker_count": allocation["t1c_ticker_count"],
        "t1c_ticker_list_sha256": allocation["t1c_ticker_list_sha256"],
        "parent_t_spare_ticker_list_sha256": allocation["parent_t_spare_ticker_list_sha256"],
        "remaining_t_spare_ticker_list_sha256": allocation["remaining_t_spare_ticker_list_sha256"],
        "t1c_raw_acquisition_performed": state["T1"]["raw_data_acquired"] is not False or state["last_real_t1_acquisition_attempt"]["t1_successfully_acquired"] is not False,
        "t1c_research_opened": state["T1"]["layer_b_opened"] is not False or state["last_real_t1_acquisition_attempt"]["t1_opened_for_research"] is not False,
        "t1c_ohlcv_research_access": state["T1"]["validation_access_count"] is not None or state["last_real_t1_acquisition_attempt"]["validation_accessed"] is not False,
        "t1c_feature_access": terminal["t1c_feature_access"],
        "t1c_outcome_access": terminal["t1c_outcome_access"],
        "t1c_identities_publicly_exposed": terminal["t1c_identities_publicly_exposed"],
        "t1c_membership_reassigned": allocation["t1c_ticker_list_sha256"] != EXPECTED_V8E_T1C_TICKER_LIST_SHA256 or allocation["logical_block"] != "T1C",
        "allocation_self_hash_unchanged": allocation["authorized_allocation_artifact_self_hash"] == AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH and historical["allocation_self_hash_unchanged"] is True,
        "parent_v8_provenance_unchanged": allocation["parent_v8_partition_manifest_sha256"] == EXPECTED_V8_PARTITION_MANIFEST_SHA256 and allocation["parent_v8_partition_implementation_commit"] == EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT and historical["parent_v8_provenance_unchanged"] is True,
        "v8c_terminal_adjudication_authoritative": historical["v8c_terminal_adjudication_authoritative"] is True,
        "fresh_public_preservation_evidence_result": "PASS",
    }
    return _validate_fresh_t1c_public_evidence(evidence)


def derive_fresh_t1c_public_evidence(
    repository_root: Path = CANONICAL_REPOSITORY_ROOT,
    *,
    preflight: Mapping[str, Any] | None = None,
    chronology_reader: Callable[[Path, str, str], Any] | None = None,
) -> dict[str, Any]:
    """Resolve fresh public T1C evidence before any gate or private read."""
    verified_preflight = preflight or _default_public_preflight(repository_root)
    return _default_fresh_t1c_public_evidence(
        repository_root, _validate_public_preflight(verified_preflight), chronology_reader=chronology_reader
    )


def _default_public_preflight(repository_root: Path = CANONICAL_REPOSITORY_ROOT) -> dict[str, Any]:
    status = _git_text(repository_root, ["status", "--porcelain"], "V8E_PUBLIC_GIT_UNAVAILABLE")
    branch = _git_text(repository_root, ["branch", "--show-current"], "V8E_PUBLIC_BRANCH_UNAVAILABLE")
    head = _git_text(repository_root, ["rev-parse", "HEAD"], "V8E_PUBLIC_HEAD_UNAVAILABLE")
    origin_head = _git_text(
        repository_root,
        ["rev-parse", "origin/" + V8E_PRODUCTION_BRANCH],
        "V8E_PUBLIC_ORIGIN_UNAVAILABLE",
    )
    origin_url = _git_text(repository_root, ["config", "--get", "remote.origin.url"], "V8E_PUBLIC_ORIGIN_UNAVAILABLE")
    if origin_url not in {
        "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "git@github.com:ta1k1-arakawa/stock-analyzer.git",
        "ssh://git@github.com/ta1k1-arakawa/stock-analyzer.git",
    }:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    try:
        design_blob = resolve_git_blob(repository_root, V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT, "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md")
        terminal_blob = resolve_git_blob(repository_root, V8C_TERMINAL_COMMIT, V8C_TERMINAL_ADJUDICATION_GIT_PATH)
        prefreeze_blob = resolve_git_blob(repository_root, V8C_TERMINAL_COMMIT, V8C_PREFREEZE_AUDIT_GIT_PATH)
        trusted_allocation_blob = resolve_git_blob(repository_root, head, V8C_TRUSTED_ALLOCATION_GIT_PATH)
        anchor = read_and_verify_v8_trusted_partition_anchor(repository_root, head)
        trusted_allocation = _strict_public_json(
            read_git_object_bytes(repository_root, head, V8C_TRUSTED_ALLOCATION_GIT_PATH),
            "V8E_TRUSTED_ALLOCATION_INVALID_JSON",
            "V8E_TRUSTED_ALLOCATION_DUPLICATE_KEY",
        )
    except (V8CGitProvenanceBlocked, V8ET1CPreservationBlocked) as error:
        if isinstance(error, V8ET1CPreservationBlocked):
            raise
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_PROVENANCE_INVALID") from error
    if design_blob != V8E_DESIGN_CANDIDATE_BLOB_SHA or trusted_allocation_blob != V8C_TRUSTED_ALLOCATION_BLOB_SHA:
        raise V8ET1CPreservationBlocked("V8E_PUBLIC_PROVENANCE_INVALID")
    if set(trusted_allocation) < {
        "v8c_allocation_implementation_commit",
        "parent_v8_partition_manifest_sha256",
        "parent_v8_partition_implementation_commit",
        "parent_t_spare_ticker_count",
        "parent_t_spare_ticker_list_sha256",
    }:
        raise V8ET1CPreservationBlocked("V8E_TRUSTED_ALLOCATION_SCHEMA_INVALID")
    if trusted_allocation["v8c_allocation_implementation_commit"] != EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT:
        raise V8ET1CPreservationBlocked("V8E_ALLOCATION_IMPLEMENTATION_BINDING_INVALID")
    if trusted_allocation["parent_v8_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8ET1CPreservationBlocked("V8E_PARTITION_MANIFEST_BINDING_INVALID")
    if trusted_allocation["parent_v8_partition_implementation_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8ET1CPreservationBlocked("V8E_PARTITION_IMPLEMENTATION_BINDING_INVALID")
    return _validate_public_preflight(
        {
            "repository_identity": V8E_REPOSITORY_IDENTITY,
            "branch": branch,
            "head": head,
            "origin_head": origin_head,
            "worktree_clean": status == "",
            "reviewed_v8e_design_candidate_commit": V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
            "reviewed_v8e_design_blob_sha": design_blob,
            "v8c_terminal_commit": V8C_TERMINAL_COMMIT,
            "v8c_terminal_blob_sha": terminal_blob,
            "v8c_prefreeze_blob_sha": prefreeze_blob,
            "trusted_partition_blob_sha": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
            "partition_manifest_sha256": anchor["authorized_partition_manifest_sha256"],
            "partition_implementation_commit": anchor["authorized_partition_implementation_git_commit"],
            "v8c_allocation_implementation_commit": trusted_allocation["v8c_allocation_implementation_commit"],
            "parent_t_spare_ticker_count": trusted_allocation["parent_t_spare_ticker_count"],
            "parent_t_spare_ticker_list_sha256": trusted_allocation["parent_t_spare_ticker_list_sha256"],
        }
    )


def _read_partition_manifest_bytes(raw: bytes) -> dict[str, Any]:
    manifest = _strict_json_object(raw, "V8E_PARTITION_MANIFEST_INVALID_JSON", "V8E_PARTITION_MANIFEST_DUPLICATE_KEY")
    if set(manifest) != set(MANIFEST_FIELDS):
        raise V8ET1CPreservationBlocked("V8E_PARTITION_MANIFEST_SCHEMA_INVALID")
    if manifest["manifest_sha256"] != v8_canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise V8ET1CPreservationBlocked("V8E_PARTITION_MANIFEST_SHA_MISMATCH")
    try:
        require_v8_git_commit(manifest["partition_implementation_git_commit"])
    except V8PartitionBlocked as error:
        raise V8ET1CPreservationBlocked("V8E_PARTITION_IMPLEMENTATION_COMMIT_INVALID") from error
    if (
        manifest["schema_version"] != V8_PARTITION_SCHEMA_VERSION
        or manifest["source_snapshot_semantics"] != SOURCE_SNAPSHOT_SEMANTICS
        or manifest["source_snapshot_clarification_commit"] != SOURCE_SNAPSHOT_CLARIFICATION_COMMIT
        or manifest["v4_raw_sha_equality_required"] is not False
        or manifest["source_reproduction_status"] != "PASS"
        or manifest["t0_reproduction_status"] != "PASS"
    ):
        raise V8ET1CPreservationBlocked("V8E_PARTITION_MANIFEST_FROZEN_BINDING_INVALID")
    assignments = manifest["block_assignments"]
    if not isinstance(assignments, Mapping) or set(_REQUIRED_BLOCK_KEYS) - set(assignments):
        raise V8ET1CPreservationBlocked("V8E_PARTITION_BLOCK_ASSIGNMENT_INVALID")
    for key in _REQUIRED_BLOCK_KEYS:
        if not isinstance(assignments[key], list):
            raise V8ET1CPreservationBlocked("V8E_PARTITION_BLOCK_ASSIGNMENT_INVALID")
    return manifest


def _verify_private_artifacts(
    allocation_raw: bytes,
    partition_manifest_raw: bytes,
    *,
    expected_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
    expected_parent_t_spare_ticker_list_sha256: str = EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
    expected_t1c_ticker_list_sha256: str = EXPECTED_V8E_T1C_TICKER_LIST_SHA256,
    expected_remaining_t_spare_ticker_list_sha256: str = EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
    expected_partition_manifest_sha256: str = EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    expected_partition_implementation_commit: str = EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    expected_v8c_allocation_implementation_commit: str = EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT,
    expected_v8c_frozen_design_commit: str = EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
) -> dict[str, Any]:
    """Pure bytes-only evaluator; synthetic fixtures are the only test input."""
    try:
        artifact = read_t1c_allocation_artifact_bytes(allocation_raw)
        manifest = _read_partition_manifest_bytes(partition_manifest_raw)
        assignments = manifest["block_assignments"]
        if manifest["manifest_sha256"] != expected_partition_manifest_sha256:
            raise V8ET1CPreservationBlocked("V8E_PARTITION_MANIFEST_SHA_MISMATCH_TRUSTED")
        if manifest["partition_implementation_git_commit"] != expected_partition_implementation_commit:
            raise V8ET1CPreservationBlocked("V8E_PARTITION_IMPLEMENTATION_COMMIT_MISMATCH_TRUSTED")
        if manifest["study_name"] != "V8_HISTORICAL_RESEARCH":
            raise V8ET1CPreservationBlocked("V8E_PARTITION_STUDY_MISMATCH")
        if manifest["design_commit"] != "c414d3191cba356734d7ed08bdf1abc7d51fc384":
            raise V8ET1CPreservationBlocked("V8E_PARTITION_DESIGN_COMMIT_MISMATCH")
        parent = assignments["T_spare"]
        if len(parent) != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
            raise V8ET1CPreservationBlocked("V8E_PARENT_T_SPARE_COUNT_MISMATCH")
        if manifest["t_spare_ticker_list_sha256"] != expected_parent_t_spare_ticker_list_sha256:
            raise V8ET1CPreservationBlocked("V8E_PARENT_T_SPARE_HASH_MISMATCH")
        if artifact.get("parent_v8_partition_manifest_sha256") != expected_partition_manifest_sha256:
            raise V8ET1CPreservationBlocked("V8E_ALLOCATION_PARENT_MANIFEST_MISMATCH")
        if artifact.get("parent_v8_partition_implementation_commit") != expected_partition_implementation_commit:
            raise V8ET1CPreservationBlocked("V8E_ALLOCATION_PARENT_IMPLEMENTATION_MISMATCH")
        if artifact.get("v8c_allocation_implementation_commit") != expected_v8c_allocation_implementation_commit:
            raise V8ET1CPreservationBlocked("V8E_ALLOCATION_IMPLEMENTATION_NOT_REVIEWED")
        if artifact.get("artifact_self_hash") != expected_allocation_artifact_self_hash:
            raise V8ET1CPreservationBlocked("V8E_ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH_TRUSTED")
        safe = _verify_t1c_allocation_artifact(
            artifact,
            parent_t_spare_tickers=parent,
            t0_tickers=assignments["T0"],
            old_t1_tickers=assignments["T1"],
            t2_tickers=assignments["T2"],
            t3_tickers=assignments["T3"],
            expected_parent_t_spare_ticker_list_sha256=expected_parent_t_spare_ticker_list_sha256,
            expected_v8c_frozen_design_commit=expected_v8c_frozen_design_commit,
        )
    except (V8ET1CPreservationBlocked, V8CAllocationBlocked, V8CAllocationVerificationBlocked) as error:
        if isinstance(error, V8ET1CPreservationBlocked):
            raise
        raise V8ET1CPreservationBlocked("V8E_PRIVATE_ALLOCATION_VERIFICATION_BLOCKED") from error
    if safe["artifact_self_hash"] != expected_allocation_artifact_self_hash:
        raise V8ET1CPreservationBlocked("V8E_ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH_TRUSTED")
    if safe["t1c_ticker_count"] != EXPECTED_V8E_T1C_TICKER_COUNT:
        raise V8ET1CPreservationBlocked("V8E_T1C_COUNT_MISMATCH")
    if safe["t1c_ticker_list_sha256"] != expected_t1c_ticker_list_sha256:
        raise V8ET1CPreservationBlocked("V8E_T1C_HASH_MISMATCH")
    if safe["parent_t_spare_ticker_list_sha256"] != expected_parent_t_spare_ticker_list_sha256:
        raise V8ET1CPreservationBlocked("V8E_PARENT_T_SPARE_HASH_MISMATCH")
    if safe["remaining_t_spare_ticker_list_sha256"] != expected_remaining_t_spare_ticker_list_sha256:
        raise V8ET1CPreservationBlocked("V8E_REMAINING_T_SPARE_HASH_MISMATCH")
    return {
        "allocation_artifact_self_hash": safe["artifact_self_hash"],
        "t1c_ticker_count": safe["t1c_ticker_count"],
        "t1c_ticker_list_sha256": safe["t1c_ticker_list_sha256"],
        "parent_t_spare_ticker_list_sha256": safe["parent_t_spare_ticker_list_sha256"],
        "remaining_t_spare_ticker_list_sha256": safe["remaining_t_spare_ticker_list_sha256"],
        "t1c_membership_reassigned": False,
        "allocation_self_hash_unchanged": True,
        "parent_v8_provenance_unchanged": True,
    }


def _build_public_artifact(
    private_summary: Mapping[str, Any], fresh_public_evidence: Mapping[str, Any]
) -> dict[str, Any]:
    fresh = _validate_fresh_t1c_public_evidence(fresh_public_evidence)
    for key in (
        "allocation_artifact_self_hash",
        "t1c_ticker_count",
        "t1c_ticker_list_sha256",
        "parent_t_spare_ticker_list_sha256",
        "remaining_t_spare_ticker_list_sha256",
        "t1c_membership_reassigned",
        "allocation_self_hash_unchanged",
        "parent_v8_provenance_unchanged",
    ):
        if private_summary[key] != fresh[key]:
            raise V8ET1CPreservationBlocked("V8E_PRIVATE_PUBLIC_EVIDENCE_MISMATCH:" + key)
    artifact = {
        "schema_version": "V8E_T1C_PRESERVATION_RECHECK_V1",
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "study": V8E_STUDY_NAME,
        "reviewed_v8e_design_candidate_commit": V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "source_v8c_terminal_commit": V8C_TERMINAL_COMMIT,
        "allocation_artifact_self_hash": fresh["allocation_artifact_self_hash"],
        "t1c_ticker_count": fresh["t1c_ticker_count"],
        "t1c_ticker_list_sha256": fresh["t1c_ticker_list_sha256"],
        "parent_t_spare_ticker_list_sha256": fresh["parent_t_spare_ticker_list_sha256"],
        "remaining_t_spare_ticker_list_sha256": fresh["remaining_t_spare_ticker_list_sha256"],
        "t1c_raw_acquisition_performed": fresh["t1c_raw_acquisition_performed"],
        "t1c_research_opened": fresh["t1c_research_opened"],
        "t1c_ohlcv_research_access": fresh["t1c_ohlcv_research_access"],
        "t1c_feature_access": fresh["t1c_feature_access"],
        "t1c_outcome_access": fresh["t1c_outcome_access"],
        "t1c_identities_publicly_exposed": fresh["t1c_identities_publicly_exposed"],
        "t1c_membership_reassigned": fresh["t1c_membership_reassigned"],
        "allocation_self_hash_unchanged": fresh["allocation_self_hash_unchanged"],
        "parent_v8_provenance_unchanged": fresh["parent_v8_provenance_unchanged"],
        "v8c_terminal_adjudication_authoritative": fresh["v8c_terminal_adjudication_authoritative"],
        "preservation_recheck_result": "PASS",
    }
    if set(artifact) != set(V8E_PRESERVATION_ARTIFACT_FIELDS):
        raise V8ET1CPreservationBlocked("V8E_PRESERVATION_ARTIFACT_SCHEMA_INVALID")
    return artifact


def _validate_public_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping) or set(artifact) != set(V8E_PRESERVATION_ARTIFACT_FIELDS):
        raise V8ET1CPreservationBlocked("V8E_PRESERVATION_ARTIFACT_SCHEMA_INVALID")
    exact = {
        "schema_version": "V8E_T1C_PRESERVATION_RECHECK_V1",
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "study": V8E_STUDY_NAME,
        "reviewed_v8e_design_candidate_commit": V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "source_v8c_terminal_commit": V8C_TERMINAL_COMMIT,
        "allocation_artifact_self_hash": AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": EXPECTED_V8E_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": EXPECTED_V8E_T1C_TICKER_LIST_SHA256,
        "parent_t_spare_ticker_list_sha256": EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "remaining_t_spare_ticker_list_sha256": EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "t1c_raw_acquisition_performed": False,
        "t1c_research_opened": False,
        "t1c_ohlcv_research_access": False,
        "t1c_feature_access": False,
        "t1c_outcome_access": False,
        "t1c_identities_publicly_exposed": False,
        "t1c_membership_reassigned": False,
        "allocation_self_hash_unchanged": True,
        "parent_v8_provenance_unchanged": True,
        "v8c_terminal_adjudication_authoritative": True,
        "preservation_recheck_result": "PASS",
    }
    for key, expected in exact.items():
        if artifact[key] != expected or (isinstance(expected, bool) and type(artifact[key]) is not bool):
            raise V8ET1CPreservationBlocked("V8E_PRESERVATION_ARTIFACT_VALUE_MISMATCH:" + key)
    return dict(artifact)


def verify_t1c_preservation_artifact_and_receipt(
    artifact: Mapping[str, Any] | bytes,
    receipt: Mapping[str, Any] | bytes,
    *,
    receipt_key: str,
) -> dict[str, Any]:
    """Independently verify exact artifact, receipt key, and receipt bytes."""
    if isinstance(artifact, bytes):
        artifact_value = _strict_json_object(
            artifact, "V8E_PRESERVATION_ARTIFACT_INVALID_JSON", "V8E_PRESERVATION_ARTIFACT_DUPLICATE_KEY"
        )
        artifact_bytes = artifact
    else:
        artifact_value = dict(artifact)
        artifact_bytes = None
    if isinstance(receipt, bytes):
        receipt_value = _strict_json_object(receipt, "V8E_RECEIPT_INVALID_JSON", "V8E_RECEIPT_DUPLICATE_KEY")
        receipt_bytes = receipt
    else:
        receipt_value = dict(receipt)
        receipt_bytes = None
    artifact_value = _validate_public_artifact(artifact_value)
    receipt_value = _validate_receipt(receipt_value, receipt_key)
    if artifact_value["reviewed_v8e_design_candidate_commit"] != receipt_value["reviewed_v8e_design_candidate_commit"]:
        raise V8ET1CPreservationBlocked("V8E_ARTIFACT_RECEIPT_CANDIDATE_MISMATCH")
    if artifact_value["allocation_artifact_self_hash"] != receipt_value["authorized_allocation_artifact_self_hash"]:
        raise V8ET1CPreservationBlocked("V8E_ARTIFACT_RECEIPT_ALLOCATION_MISMATCH")
    result = {
        "result": "PASS",
        "artifact_schema_verified": True,
        "receipt_validation_result": "PASS",
        "gate_receipt_key_sha256": receipt_key,
    }
    if receipt_bytes is not None:
        result["gate_receipt_bytes_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    if artifact_bytes is not None:
        result["artifact_bytes_sha256"] = hashlib.sha256(artifact_bytes).hexdigest()
    return result


def verify_t1c_preservation_artifact_bytes(
    artifact_raw: bytes, receipt_raw: bytes, *, receipt_key: str
) -> dict[str, Any]:
    return verify_t1c_preservation_artifact_and_receipt(artifact_raw, receipt_raw, receipt_key=receipt_key)


def _execute_with_dependencies(
    *,
    authorization_identity: str,
    state_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    repository_root: Path,
    public_preflight: Callable[[], Mapping[str, Any]],
    private_reader: Callable[[Path], bytes],
    gate_consumer: Callable[..., Mapping[str, Any]],
    clock: Callable[[], datetime],
    public_evidence_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    reviewed_v8e_design_candidate_commit: str = V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    authorized_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
) -> dict[str, Any]:
    """DI-only future execution boundary; never called with real private paths here."""
    preflight = _validate_public_preflight(public_preflight())
    validate_authorization_identity(
        authorization_identity,
        reviewed_v8e_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    receipt_key = compute_receipt_key(
        authorization_identity,
        reviewed_v8e_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    fresh_public_evidence = _validate_fresh_t1c_public_evidence(
        (public_evidence_resolver or (lambda value: _default_fresh_t1c_public_evidence(repository_root, value)))(preflight)
    )
    state, output, allocation_path, manifest_path = _prepare_execution_paths(
        state_root=state_root,
        output_path=output_path,
        allocation_artifact_path=allocation_artifact_path,
        partition_manifest_path=partition_manifest_path,
        repository_root=repository_root,
        receipt_key=receipt_key,
    )
    # Exact frozen boundary: no private reader is called before durable receipt.
    gate_consumer(
        state,
        authorization_identity,
        clock=clock,
        reviewed_v8e_design_candidate_commit=reviewed_v8e_design_candidate_commit,
        authorized_allocation_artifact_self_hash=authorized_allocation_artifact_self_hash,
    )
    try:
        allocation_raw = private_reader(allocation_path)
        manifest_raw = private_reader(manifest_path)
    except OSError as error:
        raise V8ET1CPreservationBlocked("V8E_PRIVATE_ARTIFACT_READ_FAILED") from error
    private_summary = _verify_private_artifacts(
        allocation_raw,
        manifest_raw,
        expected_allocation_artifact_self_hash=preflight.get(
            "authorized_allocation_artifact_self_hash", AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH
        ),
        expected_parent_t_spare_ticker_list_sha256=preflight["parent_t_spare_ticker_list_sha256"],
        expected_partition_manifest_sha256=preflight["partition_manifest_sha256"],
        expected_partition_implementation_commit=preflight["partition_implementation_commit"],
        expected_v8c_allocation_implementation_commit=preflight["v8c_allocation_implementation_commit"],
    )
    artifact = _build_public_artifact(private_summary, fresh_public_evidence)
    if output.exists():
        raise V8ET1CPreservationBlocked("V8E_PRESERVATION_ARTIFACT_ALREADY_EXISTS")
    payload = _canonical_json_bytes(artifact)
    staging = output.parent / (output.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(staging, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output)
        except FileExistsError as error:
            raise V8ET1CPreservationBlocked("V8E_PRESERVATION_ARTIFACT_ALREADY_EXISTS") from error
        except OSError as error:
            raise V8ET1CPreservationBlocked("V8E_PRESERVATION_ARTIFACT_ATOMIC_PUBLISH_FAILED") from error
        _fsync_directory(output.parent)
    except V8ET1CPreservationBlocked:
        raise
    except OSError as error:
        raise V8ET1CPreservationBlocked("V8E_PRESERVATION_ARTIFACT_WRITE_FAILED") from error
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass
    return dict(artifact)


def resolve_and_verify_t1c_preservation(
    authorization_identity: str,
    *,
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Prepared future entry point; not executed by this support task."""
    return _execute_with_dependencies(
        authorization_identity=authorization_identity,
        state_root=CANONICAL_V8E_T1C_PRESERVATION_STATE_ROOT,
        output_path=output_path,
        allocation_artifact_path=allocation_artifact_path,
        partition_manifest_path=partition_manifest_path,
        repository_root=CANONICAL_REPOSITORY_ROOT,
        public_preflight=lambda: _default_public_preflight(CANONICAL_REPOSITORY_ROOT),
        private_reader=lambda path: path.read_bytes(),
        gate_consumer=consume_gate_once,
        clock=clock or (lambda: datetime.now(timezone.utc)),
    )


__all__ = [
    "AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH",
    "CANONICAL_V8E_T1C_PRESERVATION_STATE_ROOT",
    "EXPECTED_PARENT_T_SPARE_TICKER_COUNT",
    "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256",
    "EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256",
    "EXPECTED_V8E_T1C_TICKER_COUNT",
    "EXPECTED_V8E_T1C_TICKER_LIST_SHA256",
    "V8E_DESIGN_CANDIDATE_BLOB_SHA",
    "V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT",
    "V8E_PRESERVATION_ARTIFACT_FIELDS",
    "V8E_FRESH_T1C_PUBLIC_EVIDENCE_FIELDS",
    "V8E_RECEIPT_FIELDS",
    "V8E_STUDY_NAME",
    "V8E_T1C_PRESERVATION_GATE",
    "V8ET1CPreservationBlocked",
    "authorization_identity_sha256",
    "compute_receipt_key",
    "consume_gate_once",
    "derive_fresh_t1c_public_evidence",
    "gate_receipt_bytes_sha256",
    "read_gate_receipt",
    "resolve_and_verify_t1c_preservation",
    "validate_authorization_identity",
    "verify_t1c_preservation_artifact_and_receipt",
    "verify_t1c_preservation_artifact_bytes",
]
