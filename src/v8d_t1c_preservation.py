"""V8D T1C preservation verification boundary.

This module prepares the later, explicitly authorized
``HUMAN_V8D_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE`` execution.  The
public production entry point performs all Git, branch, frozen-design,
trust-anchor, and public-adjudication checks before it reads a private byte.
It then creates a durable, no-overwrite V8D receipt immediately before the
first private read.  A receipt is never reset or deleted by this module.

The private evaluator is deliberately bytes-based and dependency-injected
for synthetic tests.  It returns only counts, hashes, booleans, and Git
provenance; ticker identities and private contents never cross the public
result boundary.
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
from src.v8c_production_provenance import (
    EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
    EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
    V8CProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    read_and_verify_v8_trusted_partition_anchor,
    verify_reviewed_implementation_binding,
)
from src.v8c_t1c_allocation import (
    T1C_TICKER_COUNT,
    V8CAllocationBlocked,
    read_t1c_allocation_artifact_bytes,
)
from src.v8c_t1c_allocation_verification import (
    V8CAllocationVerificationBlocked,
    _verify_t1c_allocation_artifact,
)

V8D_STUDY_NAME = "V8D_HISTORICAL_RESEARCH"
V8D_PRODUCTION_BRANCH = "v8d-transport-audit-design"
V8D_FROZEN_DESIGN_COMMIT = "eda657cde2383718d986c4c4bfaae794784fe04d"
V8D_FROZEN_DESIGN_BLOB_SHA = "9577a88c7bf46483b941aec3301c6064d9734c1f"

V8C_TERMINAL_COMMIT = "d18368c1ec1c26d752ea5862115ab9f4315d1780"
V8C_TERMINAL_ADJUDICATION_GIT_PATH = "V8C_T1C_READINESS_BLOCK_ADJUDICATION.md"
V8C_TERMINAL_ADJUDICATION_BLOB_SHA = "d40b3ef6b071b150dab8269044398fd6fc8227c5"
V8C_PREFREEZE_AUDIT_GIT_PATH = "V8C_PREFREEZE_PRESERVATION_RECHECK.md"
V8C_PREFREEZE_AUDIT_BLOB_SHA = "ec9054caf94898948879b599196c055e480d2e52"

V8D_T1C_PRESERVATION_GATE = "HUMAN_V8D_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE"
V8D_AUTHORIZATION_PREFIX = "V8D_HUMAN_AUTHORIZE_T1C_PRESERVATION_VERIFY_AT_"
V8D_AUTHORIZATION_SEPARATOR = "_FOR_"
V8D_RECEIPT_SCHEMA_VERSION = "V8D_T1C_PRESERVATION_GATE_RECEIPT_V1"
V8D_RECEIPT_ARTIFACT_ROLE = "T1C_PRESERVATION_PRIVATE_GATE_RECEIPT"
V8D_RECEIPT_CONSUMPTION_BOUNDARY = "IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ"
V8D_RECEIPT_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "gate",
    "reviewed_design_candidate_commit",
    "authorization_identity_sha256",
    "authorized_allocation_artifact_self_hash",
    "consumed",
    "consumption_count",
    "consumption_boundary",
    "consumption_timestamp_utc",
)

V8D_PRESERVATION_ARTIFACT_FIELDS = (
    "schema_version",
    "artifact_role",
    "study",
    "reviewed_design_candidate_commit",
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

_HEX = re.compile(r"^[0-9a-f]+$")
_REQUIRED_BLOCK_KEYS = ("T0", "T1", "T2", "T3", "T_spare")


class V8DT1CPreservationBlocked(RuntimeError):
    """Fail-closed V8D T1C preservation preparation/verification error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise V8DT1CPreservationBlocked(reason)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
            "utf-8"
        )
    except (TypeError, ValueError) as error:
        raise V8DT1CPreservationBlocked("V8D_PRESERVATION_NONFINITE_OR_UNSERIALIZABLE") from error


def authorization_identity_sha256(authorization_identity: str) -> str:
    """Hash a supplied identity without ever returning or storing the raw value."""
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8DT1CPreservationBlocked("V8D_AUTHORIZATION_IDENTITY_REQUIRED")
    return hashlib.sha256(authorization_identity.encode("utf-8")).hexdigest()


def validate_authorization_identity(
    authorization_identity: str,
    reviewed_design_candidate_commit: str = V8D_FROZEN_DESIGN_COMMIT,
    authorized_allocation_artifact_self_hash: str = "16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c",
) -> None:
    """Require the exact frozen authorization grammar before any receipt use."""
    design = _require_hex(reviewed_design_candidate_commit, 40, "V8D_DESIGN_COMMIT_INVALID")
    allocation_hash = _require_hex(authorized_allocation_artifact_self_hash, 64, "V8D_ALLOCATION_HASH_INVALID")
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8DT1CPreservationBlocked("V8D_AUTHORIZATION_GRAMMAR_MISMATCH")
    expected = V8D_AUTHORIZATION_PREFIX + design + V8D_AUTHORIZATION_SEPARATOR + allocation_hash
    if authorization_identity != expected:
        raise V8DT1CPreservationBlocked("V8D_AUTHORIZATION_GRAMMAR_MISMATCH")


def compute_receipt_key(
    authorization_identity: str,
    reviewed_design_candidate_commit: str = V8D_FROZEN_DESIGN_COMMIT,
    authorized_allocation_artifact_self_hash: str = "16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c",
) -> str:
    validate_authorization_identity(
        authorization_identity,
        reviewed_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    design = _require_hex(reviewed_design_candidate_commit, 40, "V8D_DESIGN_COMMIT_INVALID")
    allocation_hash = _require_hex(authorized_allocation_artifact_self_hash, 64, "V8D_ALLOCATION_HASH_INVALID")
    identity_hash = authorization_identity_sha256(authorization_identity)
    material = "|".join(
        (
            "ta1k1-arakawa/stock-analyzer",
            V8D_T1C_PRESERVATION_GATE,
            design,
            identity_hash,
            allocation_hash,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _receipt_path(state_root: str | os.PathLike[str], receipt_key: str) -> Path:
    _require_hex(receipt_key, 64, "V8D_RECEIPT_KEY_INVALID")
    return Path(state_root) / (receipt_key + ".json")


def _timestamp_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_CLOCK_INVALID")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _strict_json_object(raw: bytes, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8DT1CPreservationBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8DT1CPreservationBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8DT1CPreservationBlocked(invalid_reason)
    return parsed


def _validate_receipt(receipt: Mapping[str, Any], receipt_key: str) -> dict[str, Any]:
    if set(receipt) != set(V8D_RECEIPT_FIELDS):
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_SCHEMA_INVALID")
    if receipt["schema_version"] != V8D_RECEIPT_SCHEMA_VERSION:
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_SCHEMA_VERSION_MISMATCH")
    if receipt["study"] != V8D_STUDY_NAME or receipt["artifact_role"] != V8D_RECEIPT_ARTIFACT_ROLE:
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_IDENTITY_INVALID")
    if receipt["gate"] != V8D_T1C_PRESERVATION_GATE:
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_GATE_INVALID")
    design = _require_hex(receipt["reviewed_design_candidate_commit"], 40, "V8D_RECEIPT_DESIGN_COMMIT_INVALID")
    identity_hash = _require_hex(receipt["authorization_identity_sha256"], 64, "V8D_RECEIPT_IDENTITY_HASH_INVALID")
    _require_hex(receipt["authorized_allocation_artifact_self_hash"], 64, "V8D_RECEIPT_ALLOCATION_HASH_INVALID")
    if receipt["consumed"] is not True or type(receipt["consumption_count"]) is not int or receipt["consumption_count"] != 1:
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_CONSUMPTION_INVALID")
    if receipt["consumption_boundary"] != V8D_RECEIPT_CONSUMPTION_BOUNDARY:
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_CONSUMPTION_BOUNDARY_INVALID")
    if not isinstance(receipt["consumption_timestamp_utc"], str) or not receipt["consumption_timestamp_utc"].endswith("Z"):
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_TIMESTAMP_INVALID")
    material = "|".join(
        (
            "ta1k1-arakawa/stock-analyzer",
            receipt["gate"],
            design,
            identity_hash,
            receipt["authorized_allocation_artifact_self_hash"],
        )
    )
    if hashlib.sha256(material.encode("utf-8")).hexdigest() != receipt_key:
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_KEY_CONTENT_MISMATCH")
    return dict(receipt)


def read_gate_receipt(state_root: str | os.PathLike[str], receipt_key: str) -> dict[str, Any]:
    path = _receipt_path(state_root, receipt_key)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_MISSING") from error
    return _validate_receipt(
        _strict_json_object(raw, "V8D_RECEIPT_INVALID_JSON", "V8D_RECEIPT_DUPLICATE_KEY"), receipt_key
    )


def consume_gate_once(
    state_root: str | os.PathLike[str],
    authorization_identity: str,
    *,
    clock: Callable[[], datetime],
    reviewed_design_candidate_commit: str = V8D_FROZEN_DESIGN_COMMIT,
    authorized_allocation_artifact_self_hash: str = "16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c",
) -> dict[str, Any]:
    """Durably consume the synthetic or real V8D gate exactly once.

    The caller must invoke this immediately before its first private-byte
    read.  There is intentionally no reset or deletion function.
    """
    validate_authorization_identity(
        authorization_identity,
        reviewed_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    receipt_key = compute_receipt_key(
        authorization_identity,
        reviewed_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    path = _receipt_path(state_root, receipt_key)
    if path.exists():
        raise V8DT1CPreservationBlocked("V8D_GATE_ALREADY_CONSUMED")
    receipt = {
        "schema_version": V8D_RECEIPT_SCHEMA_VERSION,
        "study": V8D_STUDY_NAME,
        "artifact_role": V8D_RECEIPT_ARTIFACT_ROLE,
        "gate": V8D_T1C_PRESERVATION_GATE,
        "reviewed_design_candidate_commit": reviewed_design_candidate_commit,
        "authorization_identity_sha256": authorization_identity_sha256(authorization_identity),
        "authorized_allocation_artifact_self_hash": authorized_allocation_artifact_self_hash,
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": V8D_RECEIPT_CONSUMPTION_BOUNDARY,
        "consumption_timestamp_utc": _timestamp_text(clock()),
    }
    payload = _canonical_json_bytes(receipt)
    root = Path(state_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_STORAGE_UNAVAILABLE") from error
    staging = root / (path.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(staging, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, path)
        except FileExistsError as error:
            raise V8DT1CPreservationBlocked("V8D_GATE_ALREADY_CONSUMED") from error
        except OSError as error:
            raise V8DT1CPreservationBlocked("V8D_RECEIPT_STORAGE_WRITE_FAILED") from error
    except OSError as error:
        raise V8DT1CPreservationBlocked("V8D_RECEIPT_STORAGE_WRITE_FAILED") from error
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass
    return dict(receipt)


def _require_safe_external_path(value: str | os.PathLike[str], repository_root: Path, reason: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise V8DT1CPreservationBlocked(reason)
    try:
        resolved = path.resolve(strict=False)
        resolved.relative_to(repository_root.resolve())
    except ValueError:
        return resolved
    raise V8DT1CPreservationBlocked(reason)


def _prepare_execution_paths(
    *,
    state_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    repository_root: Path,
    receipt_key: str,
) -> tuple[Path, Path, Path, Path]:
    state = _require_safe_external_path(state_root, repository_root, "V8D_STATE_PATH_INVALID")
    output = _require_safe_external_path(output_path, repository_root, "V8D_OUTPUT_PATH_INVALID")
    allocation = _require_safe_external_path(allocation_artifact_path, repository_root, "V8D_PRIVATE_PATH_INVALID")
    manifest = _require_safe_external_path(partition_manifest_path, repository_root, "V8D_PRIVATE_PATH_INVALID")
    if allocation == manifest or output in {allocation, manifest} or output == state / (receipt_key + ".json"):
        raise V8DT1CPreservationBlocked("V8D_OUTPUT_PATH_COLLISION")
    try:
        state.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8DT1CPreservationBlocked("V8D_OUTPUT_OR_STATE_PREPARATION_FAILED") from error
    if not state.is_dir() or not output.parent.is_dir() or output.exists():
        raise V8DT1CPreservationBlocked("V8D_OUTPUT_OR_STATE_PREPARATION_FAILED")
    if not allocation.is_file() or not manifest.is_file():
        raise V8DT1CPreservationBlocked("V8D_PRIVATE_ARTIFACT_UNAVAILABLE")
    if (state / (receipt_key + ".json")).exists():
        raise V8DT1CPreservationBlocked("V8D_GATE_ALREADY_CONSUMED")
    return state, output, allocation, manifest


def _validate_public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "repository_identity",
        "branch",
        "head",
        "origin_head",
        "worktree_clean",
        "frozen_design_commit",
        "frozen_design_blob_sha",
        "v8c_terminal_commit",
        "v8c_terminal_blob_sha",
        "v8c_prefreeze_blob_sha",
        "trusted_partition_blob_sha",
        "partition_manifest_sha256",
        "partition_implementation_commit",
        "parent_t_spare_ticker_count",
        "parent_t_spare_ticker_list_sha256",
        "reviewed_implementation_commit",
    }
    if not isinstance(preflight, Mapping) or not required.issubset(preflight):
        raise V8DT1CPreservationBlocked("V8D_PUBLIC_PREFLIGHT_SCHEMA_INVALID")
    if preflight["repository_identity"] != "ta1k1-arakawa/stock-analyzer":
        raise V8DT1CPreservationBlocked("V8D_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    if preflight["branch"] != V8D_PRODUCTION_BRANCH or preflight["worktree_clean"] is not True:
        raise V8DT1CPreservationBlocked("V8D_PUBLIC_GIT_BINDING_INVALID")
    head = _require_hex(preflight["head"], 40, "V8D_PUBLIC_HEAD_INVALID")
    if preflight["origin_head"] != head:
        raise V8DT1CPreservationBlocked("V8D_PUBLIC_HEAD_NOT_ORIGIN")
    if preflight["frozen_design_commit"] != V8D_FROZEN_DESIGN_COMMIT:
        raise V8DT1CPreservationBlocked("V8D_FROZEN_DESIGN_COMMIT_MISMATCH")
    if preflight["frozen_design_blob_sha"] != V8D_FROZEN_DESIGN_BLOB_SHA:
        raise V8DT1CPreservationBlocked("V8D_FROZEN_DESIGN_BLOB_MISMATCH")
    if preflight["v8c_terminal_commit"] != V8C_TERMINAL_COMMIT or preflight["v8c_terminal_blob_sha"] != V8C_TERMINAL_ADJUDICATION_BLOB_SHA:
        raise V8DT1CPreservationBlocked("V8D_V8C_TERMINAL_ADJUDICATION_INVALID")
    if preflight["v8c_prefreeze_blob_sha"] != V8C_PREFREEZE_AUDIT_BLOB_SHA:
        raise V8DT1CPreservationBlocked("V8D_V8C_PREFREEZE_AUDIT_INVALID")
    if preflight["trusted_partition_blob_sha"] != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8DT1CPreservationBlocked("V8D_TRUSTED_PARTITION_ANCHOR_INVALID")
    if preflight["partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8DT1CPreservationBlocked("V8D_PARTITION_MANIFEST_BINDING_INVALID")
    if preflight["partition_implementation_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8DT1CPreservationBlocked("V8D_PARTITION_IMPLEMENTATION_BINDING_INVALID")
    if preflight["parent_t_spare_ticker_count"] != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
        raise V8DT1CPreservationBlocked("V8D_PARENT_T_SPARE_COUNT_INVALID")
    if preflight["parent_t_spare_ticker_list_sha256"] != EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256:
        raise V8DT1CPreservationBlocked("V8D_PARENT_T_SPARE_HASH_INVALID")
    _require_hex(preflight["reviewed_implementation_commit"], 40, "V8D_REVIEWED_IMPLEMENTATION_COMMIT_INVALID")
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
        raise V8DT1CPreservationBlocked("V8D_PUBLIC_GIT_UNAVAILABLE") from error
    if result.returncode != 0:
        raise V8DT1CPreservationBlocked(reason)
    return result.stdout.strip()


def _default_public_preflight(repository_root: Path = CANONICAL_REPOSITORY_ROOT) -> dict[str, Any]:
    status = _git_text(repository_root, ["status", "--porcelain"], "V8D_PUBLIC_GIT_UNAVAILABLE")
    branch = _git_text(repository_root, ["branch", "--show-current"], "V8D_PUBLIC_BRANCH_UNAVAILABLE")
    head = _git_text(repository_root, ["rev-parse", "HEAD"], "V8D_PUBLIC_HEAD_UNAVAILABLE")
    origin_head = _git_text(repository_root, ["rev-parse", "origin/" + V8D_PRODUCTION_BRANCH], "V8D_PUBLIC_ORIGIN_UNAVAILABLE")
    origin_url = _git_text(repository_root, ["config", "--get", "remote.origin.url"], "V8D_PUBLIC_ORIGIN_UNAVAILABLE")
    if origin_url not in {
        "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "git@github.com:ta1k1-arakawa/stock-analyzer.git",
        "ssh://git@github.com/ta1k1-arakawa/stock-analyzer.git",
    }:
        raise V8DT1CPreservationBlocked("V8D_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    try:
        design_blob = resolve_git_blob(repository_root, V8D_FROZEN_DESIGN_COMMIT, "V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md")
        terminal_blob = resolve_git_blob(repository_root, V8C_TERMINAL_COMMIT, V8C_TERMINAL_ADJUDICATION_GIT_PATH)
        prefreeze_blob = resolve_git_blob(repository_root, V8C_TERMINAL_COMMIT, V8C_PREFREEZE_AUDIT_GIT_PATH)
        anchor = read_and_verify_v8_trusted_partition_anchor(repository_root, head)
        approval = read_and_verify_design_freeze_approval(repository_root, head)
        reviewed = verify_reviewed_implementation_binding(repository_root, head)
    except (V8CGitProvenanceBlocked, V8CProductionProvenanceBlocked) as error:
        raise V8DT1CPreservationBlocked("V8D_PUBLIC_PROVENANCE_INVALID") from error
    return _validate_public_preflight(
        {
            "repository_identity": "ta1k1-arakawa/stock-analyzer",
            "branch": branch,
            "head": head,
            "origin_head": origin_head,
            "worktree_clean": status == "",
            "frozen_design_commit": V8D_FROZEN_DESIGN_COMMIT,
            "frozen_design_blob_sha": design_blob,
            "v8c_terminal_commit": V8C_TERMINAL_COMMIT,
            "v8c_terminal_blob_sha": terminal_blob,
            "v8c_prefreeze_blob_sha": prefreeze_blob,
            "trusted_partition_blob_sha": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
            "partition_manifest_sha256": anchor["authorized_partition_manifest_sha256"],
            "partition_implementation_commit": anchor["authorized_partition_implementation_git_commit"],
            "parent_t_spare_ticker_count": EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
            "parent_t_spare_ticker_list_sha256": EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
            "reviewed_implementation_commit": reviewed["reviewed_implementation_git_commit"],
            "v8c_frozen_design_commit": approval["frozen_design_git_commit"],
        }
    )


def _read_partition_manifest_bytes(raw: bytes) -> dict[str, Any]:
    manifest = _strict_json_object(raw, "V8D_PARTITION_MANIFEST_INVALID_JSON", "V8D_PARTITION_MANIFEST_DUPLICATE_KEY")
    if set(manifest) != set(MANIFEST_FIELDS):
        raise V8DT1CPreservationBlocked("V8D_PARTITION_MANIFEST_SCHEMA_INVALID")
    if manifest["manifest_sha256"] != v8_canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise V8DT1CPreservationBlocked("V8D_PARTITION_MANIFEST_SHA_MISMATCH")
    try:
        require_v8_git_commit(manifest["partition_implementation_git_commit"])
    except V8PartitionBlocked as error:
        raise V8DT1CPreservationBlocked("V8D_PARTITION_IMPLEMENTATION_COMMIT_INVALID") from error
    if (
        manifest["schema_version"] != V8_PARTITION_SCHEMA_VERSION
        or manifest["source_snapshot_semantics"] != SOURCE_SNAPSHOT_SEMANTICS
        or manifest["source_snapshot_clarification_commit"] != SOURCE_SNAPSHOT_CLARIFICATION_COMMIT
        or manifest["v4_raw_sha_equality_required"] is not False
        or manifest["source_reproduction_status"] != "PASS"
        or manifest["t0_reproduction_status"] != "PASS"
    ):
        raise V8DT1CPreservationBlocked("V8D_PARTITION_MANIFEST_FROZEN_BINDING_INVALID")
    assignments = manifest["block_assignments"]
    if not isinstance(assignments, Mapping) or set(_REQUIRED_BLOCK_KEYS) - set(assignments):
        raise V8DT1CPreservationBlocked("V8D_PARTITION_BLOCK_ASSIGNMENT_INVALID")
    for key in _REQUIRED_BLOCK_KEYS:
        if not isinstance(assignments[key], list):
            raise V8DT1CPreservationBlocked("V8D_PARTITION_BLOCK_ASSIGNMENT_INVALID")
    return manifest


def _verify_private_artifacts(
    allocation_raw: bytes,
    partition_manifest_raw: bytes,
    *,
    expected_allocation_artifact_self_hash: str = "16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c",
    expected_parent_t_spare_ticker_list_sha256: str = EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
    expected_t1c_ticker_list_sha256: str | None = None,
    expected_remaining_t_spare_ticker_list_sha256: str | None = None,
    expected_partition_manifest_sha256: str = EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    expected_partition_implementation_commit: str = EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    expected_reviewed_implementation_commit: str = "",
    expected_v8c_frozen_design_commit: str = EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
) -> dict[str, Any]:
    """Pure private-byte evaluator; synthetic tests only."""
    try:
        artifact = read_t1c_allocation_artifact_bytes(allocation_raw)
        manifest = _read_partition_manifest_bytes(partition_manifest_raw)
        assignments = manifest["block_assignments"]
        if manifest["manifest_sha256"] != expected_partition_manifest_sha256:
            raise V8DT1CPreservationBlocked("V8D_PARTITION_MANIFEST_SHA_MISMATCH_TRUSTED")
        if manifest["partition_implementation_git_commit"] != expected_partition_implementation_commit:
            raise V8DT1CPreservationBlocked("V8D_PARTITION_IMPLEMENTATION_COMMIT_MISMATCH_TRUSTED")
        if manifest["study_name"] != "V8_HISTORICAL_RESEARCH":
            raise V8DT1CPreservationBlocked("V8D_PARTITION_STUDY_MISMATCH")
        if manifest["design_commit"] != "c414d3191cba356734d7ed08bdf1abc7d51fc384":
            raise V8DT1CPreservationBlocked("V8D_PARTITION_DESIGN_COMMIT_MISMATCH")
        parent = assignments["T_spare"]
        if len(parent) != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
            raise V8DT1CPreservationBlocked("V8D_PARENT_T_SPARE_COUNT_MISMATCH")
        if manifest["t_spare_ticker_list_sha256"] != expected_parent_t_spare_ticker_list_sha256:
            raise V8DT1CPreservationBlocked("V8D_PARENT_T_SPARE_HASH_MISMATCH")
        if artifact.get("parent_v8_partition_manifest_sha256") != expected_partition_manifest_sha256:
            raise V8DT1CPreservationBlocked("V8D_ALLOCATION_PARENT_MANIFEST_MISMATCH")
        if artifact.get("parent_v8_partition_implementation_commit") != expected_partition_implementation_commit:
            raise V8DT1CPreservationBlocked("V8D_ALLOCATION_PARENT_IMPLEMENTATION_MISMATCH")
        if expected_reviewed_implementation_commit and artifact.get("v8c_allocation_implementation_commit") != expected_reviewed_implementation_commit:
            raise V8DT1CPreservationBlocked("V8D_ALLOCATION_IMPLEMENTATION_NOT_REVIEWED")
        if artifact.get("artifact_self_hash") != expected_allocation_artifact_self_hash:
            raise V8DT1CPreservationBlocked("V8D_ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH_TRUSTED")
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
    except (V8DT1CPreservationBlocked, V8CAllocationBlocked, V8CAllocationVerificationBlocked) as error:
        if isinstance(error, V8DT1CPreservationBlocked):
            raise
        raise V8DT1CPreservationBlocked("V8D_PRIVATE_ALLOCATION_VERIFICATION_BLOCKED") from error
    if safe["artifact_self_hash"] != expected_allocation_artifact_self_hash:
        raise V8DT1CPreservationBlocked("V8D_ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH_TRUSTED")
    if safe["t1c_ticker_count"] != T1C_TICKER_COUNT:
        raise V8DT1CPreservationBlocked("V8D_T1C_COUNT_MISMATCH")
    if expected_t1c_ticker_list_sha256 is not None and safe["t1c_ticker_list_sha256"] != expected_t1c_ticker_list_sha256:
        raise V8DT1CPreservationBlocked("V8D_T1C_HASH_MISMATCH")
    if expected_remaining_t_spare_ticker_list_sha256 is not None and safe["remaining_t_spare_ticker_list_sha256"] != expected_remaining_t_spare_ticker_list_sha256:
        raise V8DT1CPreservationBlocked("V8D_REMAINING_T_SPARE_HASH_MISMATCH")
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


def _build_public_artifact(private_summary: Mapping[str, Any]) -> dict[str, Any]:
    artifact = {
        "schema_version": "V8D_T1C_PRESERVATION_RECHECK_V1",
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "study": V8D_STUDY_NAME,
        "reviewed_design_candidate_commit": V8D_FROZEN_DESIGN_COMMIT,
        "source_v8c_terminal_commit": V8C_TERMINAL_COMMIT,
        "allocation_artifact_self_hash": private_summary["allocation_artifact_self_hash"],
        "t1c_ticker_count": private_summary["t1c_ticker_count"],
        "t1c_ticker_list_sha256": private_summary["t1c_ticker_list_sha256"],
        "parent_t_spare_ticker_list_sha256": private_summary["parent_t_spare_ticker_list_sha256"],
        "remaining_t_spare_ticker_list_sha256": private_summary["remaining_t_spare_ticker_list_sha256"],
        "t1c_raw_acquisition_performed": False,
        "t1c_research_opened": False,
        "t1c_ohlcv_research_access": False,
        "t1c_feature_access": False,
        "t1c_outcome_access": False,
        "t1c_identities_publicly_exposed": False,
        "t1c_membership_reassigned": private_summary["t1c_membership_reassigned"],
        "allocation_self_hash_unchanged": private_summary["allocation_self_hash_unchanged"],
        "parent_v8_provenance_unchanged": private_summary["parent_v8_provenance_unchanged"],
        "v8c_terminal_adjudication_authoritative": True,
        "preservation_recheck_result": "PASS",
    }
    if set(artifact) != set(V8D_PRESERVATION_ARTIFACT_FIELDS):
        raise V8DT1CPreservationBlocked("V8D_PRESERVATION_ARTIFACT_SCHEMA_INVALID")
    return artifact


def _write_public_artifact_once(output_path: Path, artifact: Mapping[str, Any]) -> None:
    if set(artifact) != set(V8D_PRESERVATION_ARTIFACT_FIELDS):
        raise V8DT1CPreservationBlocked("V8D_PRESERVATION_ARTIFACT_SCHEMA_INVALID")
    if output_path.exists():
        raise V8DT1CPreservationBlocked("V8D_PRESERVATION_ARTIFACT_ALREADY_EXISTS")
    staging = output_path.parent / (output_path.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(staging, "xb") as stream:
            stream.write(_canonical_json_bytes(dict(artifact)))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output_path)
        except FileExistsError as error:
            raise V8DT1CPreservationBlocked("V8D_PRESERVATION_ARTIFACT_ALREADY_EXISTS") from error
        except OSError as error:
            raise V8DT1CPreservationBlocked("V8D_PRESERVATION_ARTIFACT_ATOMIC_PUBLISH_FAILED") from error
    except OSError as error:
        raise V8DT1CPreservationBlocked("V8D_PRESERVATION_ARTIFACT_WRITE_FAILED") from error
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass


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
    reviewed_design_candidate_commit: str = V8D_FROZEN_DESIGN_COMMIT,
    authorized_allocation_artifact_self_hash: str = "16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c",
) -> dict[str, Any]:
    """Private DI boundary for synthetic/local tests only."""
    preflight = _validate_public_preflight(public_preflight())
    validate_authorization_identity(
        authorization_identity,
        reviewed_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    receipt_key = compute_receipt_key(
        authorization_identity,
        reviewed_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    state, output, allocation_path, manifest_path = _prepare_execution_paths(
        state_root=state_root,
        output_path=output_path,
        allocation_artifact_path=allocation_artifact_path,
        partition_manifest_path=partition_manifest_path,
        repository_root=repository_root,
        receipt_key=receipt_key,
    )
    # This is the exact boundary: no private reader is called before this
    # durable receipt succeeds.  A later failure leaves the receipt intact.
    gate_consumer(
        state,
        authorization_identity,
        clock=clock,
        reviewed_design_candidate_commit=reviewed_design_candidate_commit,
        authorized_allocation_artifact_self_hash=authorized_allocation_artifact_self_hash,
    )
    try:
        allocation_raw = private_reader(allocation_path)
        manifest_raw = private_reader(manifest_path)
    except OSError as error:
        raise V8DT1CPreservationBlocked("V8D_PRIVATE_ARTIFACT_READ_FAILED") from error
    private_summary = _verify_private_artifacts(
        allocation_raw,
        manifest_raw,
        expected_reviewed_implementation_commit=preflight["reviewed_implementation_commit"],
        expected_v8c_frozen_design_commit=preflight.get("v8c_frozen_design_commit", EXPECTED_V8C_FROZEN_DESIGN_COMMIT),
    )
    artifact = _build_public_artifact(private_summary)
    _write_public_artifact_once(output, artifact)
    return dict(artifact)


def resolve_and_verify_t1c_preservation(
    authorization_identity: str,
    *,
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    state_root: str | os.PathLike[str],
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Future real execution entry point; never called by preparation tests."""
    return _execute_with_dependencies(
        authorization_identity=authorization_identity,
        state_root=state_root,
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
    "V8D_FROZEN_DESIGN_BLOB_SHA",
    "V8D_FROZEN_DESIGN_COMMIT",
    "V8D_PRESERVATION_ARTIFACT_FIELDS",
    "V8D_RECEIPT_FIELDS",
    "V8D_T1C_PRESERVATION_GATE",
    "V8DT1CPreservationBlocked",
    "authorization_identity_sha256",
    "compute_receipt_key",
    "consume_gate_once",
    "read_gate_receipt",
    "resolve_and_verify_t1c_preservation",
    "validate_authorization_identity",
]
