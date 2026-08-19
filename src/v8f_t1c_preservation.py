"""V8F T1C pre-freeze preservation support.

This module is the isolated support boundary authorized by V8F design §8/§10
("Design task scope boundary") and inherited V8E §9.A.  It binds every V8F
value to the independently reviewed V8F design candidate and keeps the only
future private read behind a dependency-injected bytes boundary.  Tests may
supply synthetic bytes and temporary state; the public entry point is never
called by this implementation task.

Scope note: this is the *minimum* prefreeze preservation support authorized
for V8F.  Unlike the V8E precedent this module mechanically rebinds, it does
not attempt to independently re-derive every historical V8/V8B/V8C/V8D
committed-evidence chain, because those exact blob bindings were not supplied
to this task.  The default (non-injected) resolvers below only bind to the
exact V8F/V8E facts explicitly supplied for this task; every other producer
of "fresh public evidence" must be supplied by dependency injection, exactly
as the V8E precedent already supports.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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


V8F_STUDY_NAME = "V8F_HISTORICAL_RESEARCH"
V8F_PRODUCTION_BRANCH = "v8f-transport-window-semantics-successor-design"
V8F_REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT = "cd67a9d61172da74364504cd4f93caec521a2bfc"
V8F_DESIGN_CANDIDATE_BLOB_SHA = "b7eec2b84217ad53d2e2f7dfe917396f13e51428"

# Historical V8E predecessor terminal evidence.  This is the exact commit and
# blob of the V8E T1C readiness terminal adjudication supplied by the current
# task.  It is historical evidence only and is never renamed to V8F.
V8F_V8E_PREDECESSOR_TERMINAL_COMMIT = "1dd7f838fd16996a8d2c9a9501e2d45440422cc7"
V8F_V8E_TERMINAL_RECORD_GIT_PATH = "V8E_T1C_READINESS_TERMINAL_ADJUDICATION.json"
V8F_V8E_TERMINAL_RECORD_BLOB_SHA = "4a9f4153bae40ca43533850eaf4953ac13ce5562"
V8F_V8E_TERMINAL_DISPOSITION = "BLOCK_CLOSED"
V8F_V8E_TERMINAL_FAILURE_CLASS = "TRANSPORT_PARSER_FAILURE"

# Historical V8E's own T1C preservation recheck artifact: the direct
# predecessor-study analogue of the V8D_T1C_PRESERVATION_RECHECK.json record
# the V8E precedent itself bound to.  Real Git introduction commit found via
# `git log --follow -- V8E_T1C_PRESERVATION_RECHECK.json`, not invented.
V8E_T1C_PRESERVATION_RECHECK_COMMIT = "12a05d59daca7986e4dacb27bce63e073d064240"
V8E_T1C_PRESERVATION_RECHECK_GIT_PATH = "V8E_T1C_PRESERVATION_RECHECK.json"
V8E_T1C_PRESERVATION_RECHECK_BLOB_SHA = "cd084dd6e49be724e876d01b27ac45fa11a2dc64"

V8F_T1C_PRESERVATION_GATE = "HUMAN_V8F_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE"
V8F_AUTHORIZATION_PREFIX = "V8F_HUMAN_AUTHORIZE_T1C_PRESERVATION_VERIFY_AT_"
V8F_AUTHORIZATION_SEPARATOR = "_FOR_"
V8F_RECEIPT_SCHEMA_VERSION = "V8F_T1C_PRESERVATION_GATE_RECEIPT_V1"
V8F_RECEIPT_ARTIFACT_ROLE = "T1C_PRESERVATION_PRIVATE_GATE_RECEIPT"
V8F_RECEIPT_CONSUMPTION_BOUNDARY = "IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ"
V8F_RECEIPT_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "gate",
    "reviewed_v8f_design_candidate_commit",
    "authorization_identity_sha256",
    "authorized_allocation_artifact_self_hash",
    "consumed",
    "consumption_count",
    "consumption_boundary",
    "consumption_timestamp_utc",
)

V8F_PRESERVATION_ARTIFACT_FIELDS = (
    "schema_version",
    "artifact_role",
    "study",
    "reviewed_v8f_design_candidate_commit",
    "source_v8e_terminal_commit",
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
    "v8e_terminal_adjudication_authoritative",
    "preservation_recheck_result",
)

V8F_FRESH_T1C_PUBLIC_EVIDENCE_FIELDS = (
    "schema_version",
    "evidence_role",
    "study",
    "reviewed_v8f_design_candidate_commit",
    "v8e_predecessor_terminal_commit",
    "v8e_terminal_disposition",
    "v8e_terminal_failure_class",
    "v8e_terminal_artifact_blob_sha",
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
    "v8e_terminal_adjudication_authoritative",
    "fresh_public_preservation_evidence_result",
)

# Historical V8/V8C commitments are safe public provenance constants.  They
# are not V8F authority and are never substituted for the V8F candidate.
AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH = "16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c"
EXPECTED_V8F_T1C_TICKER_COUNT = 300
EXPECTED_V8F_T1C_TICKER_LIST_SHA256 = "85a06d4b88698915315f5cf72e0d3e04dfacafb5403786ac3bb613e14b0deb54"
EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256 = "360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70"
EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256 = "699e7bc29b2714128de99203bd6fedb38ee24c6f7bfee7c725b605669c178632"
EXPECTED_PARENT_T_SPARE_TICKER_COUNT = 1904
EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA = "61faade0625139cec3fb61216ab2f97f572a7028"
EXPECTED_V8_PARTITION_MANIFEST_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT = "36cbed941050e728f7f96ce2af505e81175cc02c"
EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT = "f9c4bfcc9dab1845a6252ce7e5dc30441fec16ba"
EXPECTED_V8C_FROZEN_DESIGN_COMMIT = "c9c541ac7f7ba3bcca76db6250fe8273d9bb5756"

# Real, currently-committed public evidence source for the parent T_spare /
# allocation-implementation provenance fields.  This file's Git blob is read
# and independently derived at every preflight; the EXPECTED_* constants
# above are comparison targets only, never a substitute for that read.
V8C_TRUSTED_ALLOCATION_GIT_PATH = "V8C_TRUSTED_ALLOCATION.json"
V8C_TRUSTED_ALLOCATION_BLOB_SHA = "61082f9818efb68ca2a5ad29fa5918f887575c10"

# Real, currently-committed public evidence source for T1/access-flag
# provenance.  Read at the current head, not asserted.
V8_STATE_GIT_PATH = "V8_STATE.json"
V8_STATE_BLOB_SHA = "8e5fd2f39dc92a7983c0cdaab42f633d624b4956"

CANONICAL_V8F_T1C_PRESERVATION_STATE_ROOT = (
    CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8f-t1c-preservation-gate-state"
)

_HEX = re.compile(r"^[0-9a-f]+$")
_TIMESTAMP_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_TIMESTAMP_MICROS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_REQUIRED_BLOCK_KEYS = ("T0", "T1", "T2", "T3", "T_spare")


class V8FT1CPreservationBlocked(RuntimeError):
    """Fail-closed V8F T1C preservation support error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise V8FT1CPreservationBlocked(reason)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V8FT1CPreservationBlocked("V8F_PRESERVATION_NONFINITE_OR_UNSERIALIZABLE") from error


def authorization_identity_sha256(authorization_identity: str) -> str:
    """Return only the SHA-256; never persist or expose the raw identity."""
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8FT1CPreservationBlocked("V8F_AUTHORIZATION_IDENTITY_REQUIRED")
    return hashlib.sha256(authorization_identity.encode("utf-8")).hexdigest()


def validate_authorization_identity(
    authorization_identity: str,
    reviewed_v8f_design_candidate_commit: str = V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    authorized_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
) -> None:
    """Require the exact V8F T1C preservation identity grammar and binding."""
    candidate = _require_hex(
        reviewed_v8f_design_candidate_commit, 40, "V8F_DESIGN_COMMIT_INVALID"
    )
    allocation_hash = _require_hex(
        authorized_allocation_artifact_self_hash, 64, "V8F_ALLOCATION_HASH_INVALID"
    )
    if candidate != V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT:
        raise V8FT1CPreservationBlocked("V8F_DESIGN_CANDIDATE_MISMATCH")
    if allocation_hash != AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH:
        raise V8FT1CPreservationBlocked("V8F_ALLOCATION_HASH_MISMATCH")
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8FT1CPreservationBlocked("V8F_AUTHORIZATION_GRAMMAR_MISMATCH")
    expected = V8F_AUTHORIZATION_PREFIX + candidate + V8F_AUTHORIZATION_SEPARATOR + allocation_hash
    if authorization_identity != expected:
        raise V8FT1CPreservationBlocked("V8F_AUTHORIZATION_GRAMMAR_MISMATCH")


def compute_receipt_key(
    authorization_identity: str,
    reviewed_v8f_design_candidate_commit: str = V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    authorized_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
) -> str:
    validate_authorization_identity(
        authorization_identity,
        reviewed_v8f_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    material = "|".join(
        (
            V8F_REPOSITORY_IDENTITY,
            V8F_T1C_PRESERVATION_GATE,
            reviewed_v8f_design_candidate_commit,
            authorization_identity_sha256(authorization_identity),
            authorized_allocation_artifact_self_hash,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _receipt_path(state_root: str | os.PathLike[str], receipt_key: str) -> Path:
    _require_hex(receipt_key, 64, "V8F_RECEIPT_KEY_INVALID")
    return Path(state_root) / (receipt_key + ".json")


def _timestamp_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_CLOCK_INVALID")
    utc = value.astimezone(timezone.utc)
    if utc.microsecond:
        return utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or not (_TIMESTAMP_SECONDS.fullmatch(value) or _TIMESTAMP_MICROS.fullmatch(value)):
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_TIMESTAMP_INVALID")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ" if "." not in value else "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as error:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_TIMESTAMP_INVALID") from error
    return value


def _strict_json_object(raw: bytes, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8FT1CPreservationBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8FT1CPreservationBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8FT1CPreservationBlocked(invalid_reason)
    return parsed


def _validate_receipt(receipt: Mapping[str, Any], receipt_key: str) -> dict[str, Any]:
    if set(receipt) != set(V8F_RECEIPT_FIELDS):
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_SCHEMA_INVALID")
    if receipt["schema_version"] != V8F_RECEIPT_SCHEMA_VERSION:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_SCHEMA_VERSION_MISMATCH")
    if receipt["study"] != V8F_STUDY_NAME or receipt["artifact_role"] != V8F_RECEIPT_ARTIFACT_ROLE:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_IDENTITY_INVALID")
    if receipt["gate"] != V8F_T1C_PRESERVATION_GATE:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_GATE_INVALID")
    if receipt["reviewed_v8f_design_candidate_commit"] != V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_DESIGN_COMMIT_INVALID")
    _require_hex(receipt["authorization_identity_sha256"], 64, "V8F_RECEIPT_IDENTITY_HASH_INVALID")
    if receipt["authorized_allocation_artifact_self_hash"] != AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_ALLOCATION_HASH_INVALID")
    if receipt["consumed"] is not True or type(receipt["consumption_count"]) is not int or receipt["consumption_count"] != 1:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_CONSUMPTION_INVALID")
    if receipt["consumption_boundary"] != V8F_RECEIPT_CONSUMPTION_BOUNDARY:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_CONSUMPTION_BOUNDARY_INVALID")
    _validate_timestamp(receipt["consumption_timestamp_utc"])
    material = "|".join(
        (
            V8F_REPOSITORY_IDENTITY,
            receipt["gate"],
            receipt["reviewed_v8f_design_candidate_commit"],
            receipt["authorization_identity_sha256"],
            receipt["authorized_allocation_artifact_self_hash"],
        )
    )
    if hashlib.sha256(material.encode("utf-8")).hexdigest() != receipt_key:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_KEY_CONTENT_MISMATCH")
    return dict(receipt)


def read_gate_receipt(state_root: str | os.PathLike[str], receipt_key: str) -> dict[str, Any]:
    path = _receipt_path(state_root, receipt_key)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_MISSING") from error
    return _validate_receipt(
        _strict_json_object(raw, "V8F_RECEIPT_INVALID_JSON", "V8F_RECEIPT_DUPLICATE_KEY"), receipt_key
    )


def gate_receipt_bytes_sha256(state_root: str | os.PathLike[str], receipt_key: str) -> str:
    """Validate and hash the exact durable receipt bytes externally."""
    path = _receipt_path(state_root, receipt_key)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_MISSING") from error
    _validate_receipt(
        _strict_json_object(raw, "V8F_RECEIPT_INVALID_JSON", "V8F_RECEIPT_DUPLICATE_KEY"), receipt_key
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
    reviewed_v8f_design_candidate_commit: str = V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    authorized_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
) -> dict[str, Any]:
    """Durably publish exactly one V8F receipt; no reset/replay API exists."""
    validate_authorization_identity(
        authorization_identity,
        reviewed_v8f_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    receipt_key = compute_receipt_key(
        authorization_identity,
        reviewed_v8f_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    root = Path(state_root)
    path = _receipt_path(root, receipt_key)
    if path.exists():
        raise V8FT1CPreservationBlocked("V8F_GATE_ALREADY_CONSUMED")
    receipt = {
        "schema_version": V8F_RECEIPT_SCHEMA_VERSION,
        "study": V8F_STUDY_NAME,
        "artifact_role": V8F_RECEIPT_ARTIFACT_ROLE,
        "gate": V8F_T1C_PRESERVATION_GATE,
        "reviewed_v8f_design_candidate_commit": reviewed_v8f_design_candidate_commit,
        "authorization_identity_sha256": authorization_identity_sha256(authorization_identity),
        "authorized_allocation_artifact_self_hash": authorized_allocation_artifact_self_hash,
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": V8F_RECEIPT_CONSUMPTION_BOUNDARY,
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
            raise V8FT1CPreservationBlocked("V8F_GATE_ALREADY_CONSUMED") from error
        except OSError as error:
            raise V8FT1CPreservationBlocked("V8F_RECEIPT_STORAGE_WRITE_FAILED") from error
        _fsync_directory(root)
    except V8FT1CPreservationBlocked:
        raise
    except OSError as error:
        raise V8FT1CPreservationBlocked("V8F_RECEIPT_STORAGE_WRITE_FAILED") from error
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
        raise V8FT1CPreservationBlocked(reason)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError:
        return resolved
    raise V8FT1CPreservationBlocked(reason)


def _prepare_execution_paths(
    *,
    state_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    repository_root: Path,
    receipt_key: str,
) -> tuple[Path, Path, Path, Path]:
    state = _require_safe_external_path(state_root, repository_root, "V8F_STATE_PATH_INVALID")
    output = _require_safe_external_path(output_path, repository_root, "V8F_OUTPUT_PATH_INVALID")
    allocation = _require_safe_external_path(allocation_artifact_path, repository_root, "V8F_PRIVATE_PATH_INVALID")
    manifest = _require_safe_external_path(partition_manifest_path, repository_root, "V8F_PRIVATE_PATH_INVALID")
    if allocation == manifest or output in {allocation, manifest} or output == state / (receipt_key + ".json"):
        raise V8FT1CPreservationBlocked("V8F_OUTPUT_PATH_COLLISION")
    try:
        state.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8FT1CPreservationBlocked("V8F_OUTPUT_OR_STATE_PREPARATION_FAILED") from error
    if not state.is_dir() or not output.parent.is_dir() or output.exists():
        raise V8FT1CPreservationBlocked("V8F_OUTPUT_OR_STATE_PREPARATION_FAILED")
    if not allocation.is_file() or not manifest.is_file():
        raise V8FT1CPreservationBlocked("V8F_PRIVATE_ARTIFACT_UNAVAILABLE")
    if (state / (receipt_key + ".json")).exists():
        raise V8FT1CPreservationBlocked("V8F_GATE_ALREADY_CONSUMED")
    return state, output, allocation, manifest


_PUBLIC_PREFLIGHT_FIELDS = frozenset(
    {
        "repository_identity",
        "branch",
        "head",
        "origin_head",
        "worktree_clean",
        "reviewed_v8f_design_candidate_commit",
        "reviewed_v8f_design_blob_sha",
        "v8e_terminal_commit",
        "v8e_terminal_blob_sha",
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
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_PREFLIGHT_SCHEMA_INVALID")
    if preflight["repository_identity"] != V8F_REPOSITORY_IDENTITY:
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    if preflight["branch"] != V8F_PRODUCTION_BRANCH or preflight["worktree_clean"] is not True:
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_GIT_BINDING_INVALID")
    head = _require_hex(preflight["head"], 40, "V8F_PUBLIC_HEAD_INVALID")
    if preflight["origin_head"] != head:
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_HEAD_NOT_ORIGIN")
    if preflight["reviewed_v8f_design_candidate_commit"] != V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT:
        raise V8FT1CPreservationBlocked("V8F_DESIGN_CANDIDATE_MISMATCH")
    if preflight["reviewed_v8f_design_blob_sha"] != V8F_DESIGN_CANDIDATE_BLOB_SHA:
        raise V8FT1CPreservationBlocked("V8F_DESIGN_CANDIDATE_BLOB_MISMATCH")
    if (
        preflight["v8e_terminal_commit"] != V8F_V8E_PREDECESSOR_TERMINAL_COMMIT
        or preflight["v8e_terminal_blob_sha"] != V8F_V8E_TERMINAL_RECORD_BLOB_SHA
    ):
        raise V8FT1CPreservationBlocked("V8F_V8E_TERMINAL_ADJUDICATION_INVALID")
    if preflight["trusted_partition_blob_sha"] != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8FT1CPreservationBlocked("V8F_TRUSTED_PARTITION_ANCHOR_INVALID")
    if preflight["partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8FT1CPreservationBlocked("V8F_PARTITION_MANIFEST_BINDING_INVALID")
    if preflight["partition_implementation_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8FT1CPreservationBlocked("V8F_PARTITION_IMPLEMENTATION_BINDING_INVALID")
    if preflight["v8c_allocation_implementation_commit"] != EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT:
        raise V8FT1CPreservationBlocked("V8F_ALLOCATION_IMPLEMENTATION_BINDING_INVALID")
    if preflight["parent_t_spare_ticker_count"] != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
        raise V8FT1CPreservationBlocked("V8F_PARENT_T_SPARE_COUNT_INVALID")
    if preflight["parent_t_spare_ticker_list_sha256"] != EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256:
        raise V8FT1CPreservationBlocked("V8F_PARENT_T_SPARE_HASH_INVALID")
    return dict(preflight)


def _strict_public_json(raw: bytes, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    return _strict_json_object(raw, invalid_reason, duplicate_reason)


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
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_GIT_UNAVAILABLE") from error
    if result.returncode != 0:
        raise V8FT1CPreservationBlocked(reason)
    return result.stdout.strip()


_REVIEWED_SUPPORT_RUNTIME_FIELDS = frozenset(
    {"branch", "head", "origin_head", "worktree_clean", "commits_after_reviewed_support_sha"}
)


def _default_reviewed_support_runtime_state(
    repository_root: Path, reviewed_support_implementation_sha: str
) -> dict[str, Any]:
    reviewed_sha = _require_hex(
        reviewed_support_implementation_sha, 40, "V8F_REVIEWED_SUPPORT_SHA_MALFORMED"
    )
    resolved_sha = _git_text(
        repository_root,
        ["rev-parse", "--verify", f"{reviewed_sha}^{{commit}}"],
        "V8F_REVIEWED_SUPPORT_SHA_UNRESOLVABLE",
    )
    if resolved_sha != reviewed_sha:
        raise V8FT1CPreservationBlocked("V8F_REVIEWED_SUPPORT_SHA_UNRESOLVABLE")
    count_text = _git_text(
        repository_root,
        ["rev-list", "--count", f"{reviewed_sha}..HEAD"],
        "V8F_REVIEWED_SUPPORT_CHRONOLOGY_INVALID",
    )
    if not count_text.isdecimal():
        raise V8FT1CPreservationBlocked("V8F_REVIEWED_SUPPORT_CHRONOLOGY_INVALID")
    return {
        "branch": _git_text(repository_root, ["branch", "--show-current"], "V8F_PUBLIC_BRANCH_UNAVAILABLE"),
        "head": _git_text(repository_root, ["rev-parse", "HEAD"], "V8F_PUBLIC_HEAD_UNAVAILABLE"),
        "origin_head": _git_text(
            repository_root,
            ["rev-parse", "origin/" + V8F_PRODUCTION_BRANCH],
            "V8F_PUBLIC_ORIGIN_UNAVAILABLE",
        ),
        "worktree_clean": _git_text(repository_root, ["status", "--porcelain"], "V8F_PUBLIC_GIT_UNAVAILABLE") == "",
        "commits_after_reviewed_support_sha": int(count_text),
    }


def _validate_reviewed_support_implementation_binding(
    repository_root: Path,
    preflight: Mapping[str, Any],
    reviewed_support_implementation_sha: str,
    *,
    runtime_state_reader: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> str:
    reviewed_sha = _require_hex(
        reviewed_support_implementation_sha, 40, "V8F_REVIEWED_SUPPORT_SHA_MALFORMED"
    )
    runtime = (runtime_state_reader or _default_reviewed_support_runtime_state)(repository_root, reviewed_sha)
    if not isinstance(runtime, Mapping) or set(runtime) != _REVIEWED_SUPPORT_RUNTIME_FIELDS:
        raise V8FT1CPreservationBlocked("V8F_REVIEWED_SUPPORT_RUNTIME_SCHEMA_INVALID")
    if (
        runtime["branch"] != V8F_PRODUCTION_BRANCH
        or runtime["head"] != reviewed_sha
        or runtime["origin_head"] != reviewed_sha
        or runtime["worktree_clean"] is not True
        or type(runtime["commits_after_reviewed_support_sha"]) is not int
        or runtime["commits_after_reviewed_support_sha"] != 0
    ):
        raise V8FT1CPreservationBlocked("V8F_REVIEWED_SUPPORT_RUNTIME_BINDING_INVALID")
    if (
        preflight["branch"] != V8F_PRODUCTION_BRANCH
        or preflight["head"] != reviewed_sha
        or preflight["origin_head"] != reviewed_sha
        or preflight["worktree_clean"] is not True
    ):
        raise V8FT1CPreservationBlocked("V8F_REVIEWED_SUPPORT_PREFLIGHT_BINDING_INVALID")
    return reviewed_sha


def _parse_v8e_terminal_record(raw: bytes) -> dict[str, Any]:
    parsed = _strict_json_object(raw, "V8F_V8E_TERMINAL_RECORD_INVALID_JSON", "V8F_V8E_TERMINAL_RECORD_DUPLICATE_KEY")
    exact = {
        "schema_version": "V8E_T1C_READINESS_TERMINAL_ADJUDICATION_V1",
        "study": "V8E_HISTORICAL_RESEARCH",
        "readiness_result": "BLOCK",
        "failure_class": V8F_V8E_TERMINAL_FAILURE_CLASS,
        "t1c_raw_acquisition_allowed": False,
        "t1c_research_opening_allowed": False,
        "disposition": V8F_V8E_TERMINAL_DISPOSITION,
    }
    for key, expected in exact.items():
        if parsed.get(key) != expected:
            raise V8FT1CPreservationBlocked("V8F_V8E_TERMINAL_RECORD_VALUE_INVALID:" + key)
    return parsed


def _validate_fresh_t1c_public_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or set(evidence) != set(V8F_FRESH_T1C_PUBLIC_EVIDENCE_FIELDS):
        raise V8FT1CPreservationBlocked("V8F_FRESH_EVIDENCE_SCHEMA_INVALID")
    exact = {
        "schema_version": "V8F_T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE_V1",
        "evidence_role": "T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE",
        "study": V8F_STUDY_NAME,
        "reviewed_v8f_design_candidate_commit": V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "v8e_predecessor_terminal_commit": V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
        "v8e_terminal_disposition": V8F_V8E_TERMINAL_DISPOSITION,
        "v8e_terminal_failure_class": V8F_V8E_TERMINAL_FAILURE_CLASS,
        "v8e_terminal_artifact_blob_sha": V8F_V8E_TERMINAL_RECORD_BLOB_SHA,
        "allocation_artifact_self_hash": AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
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
        "v8e_terminal_adjudication_authoritative": True,
        "fresh_public_preservation_evidence_result": "PASS",
    }
    for key, expected in exact.items():
        if evidence[key] != expected or (isinstance(expected, bool) and type(evidence[key]) is not bool):
            raise V8FT1CPreservationBlocked("V8F_FRESH_EVIDENCE_VALUE_INVALID:" + key)
    return dict(evidence)


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
    """Independently validate the real, currently-committed T1C allocation
    pin.  Every EXPECTED_* constant here is a comparison target only; the
    observed value comes from ``record``, itself read from the real Git
    object by the caller."""
    if not isinstance(record, Mapping) or set(record) != _V8C_TRUSTED_ALLOCATION_FIELDS:
        raise V8FT1CPreservationBlocked("V8F_TRUSTED_ALLOCATION_SCHEMA_INVALID")
    exact = {
        "artifact_role": "TRUSTED_T1C_ALLOCATION_PIN",
        "authorization_status": "AUTHORIZED",
        "authorized_allocation_artifact_self_hash": AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "logical_block": "T1C",
        "parent_t_spare_ticker_count": EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
        "parent_t_spare_ticker_list_sha256": EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "parent_v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "parent_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "schema_version": "V8C_TRUSTED_ALLOCATION_V1",
        "t1c_ticker_count": EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
        "v8c_allocation_implementation_commit": EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT,
        "v8c_frozen_design_commit": EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        "verification_result": "PASS",
    }
    for key, expected in exact.items():
        if record[key] != expected:
            raise V8FT1CPreservationBlocked("V8F_TRUSTED_ALLOCATION_VALUE_INVALID:" + key)
    return dict(record)


_V8E_T1C_PRESERVATION_RECHECK_FIELDS = frozenset(
    {
        "allocation_artifact_self_hash",
        "allocation_self_hash_unchanged",
        "artifact_role",
        "parent_t_spare_ticker_list_sha256",
        "parent_v8_provenance_unchanged",
        "preservation_recheck_result",
        "remaining_t_spare_ticker_list_sha256",
        "reviewed_v8e_design_candidate_commit",
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


def _validate_historical_v8e_t1c_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Independently validate V8E's own real, historical T1C preservation
    recheck artifact -- the direct predecessor-study evidence source for the
    no-access/no-reassignment/unchanged-provenance facts, exactly as the V8E
    precedent used V8D's own historical T1C record for the same purpose."""
    if not isinstance(record, Mapping) or set(record) != _V8E_T1C_PRESERVATION_RECHECK_FIELDS:
        raise V8FT1CPreservationBlocked("V8F_V8E_HISTORICAL_T1C_SCHEMA_INVALID")
    exact = {
        "allocation_artifact_self_hash": AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "allocation_self_hash_unchanged": True,
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "parent_t_spare_ticker_list_sha256": EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "parent_v8_provenance_unchanged": True,
        "preservation_recheck_result": "PASS",
        "remaining_t_spare_ticker_list_sha256": EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "schema_version": "V8E_T1C_PRESERVATION_RECHECK_V1",
        "study": "V8E_HISTORICAL_RESEARCH",
        "t1c_feature_access": False,
        "t1c_identities_publicly_exposed": False,
        "t1c_membership_reassigned": False,
        "t1c_ohlcv_research_access": False,
        "t1c_outcome_access": False,
        "t1c_raw_acquisition_performed": False,
        "t1c_research_opened": False,
        "t1c_ticker_count": EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
        "v8c_terminal_adjudication_authoritative": True,
    }
    for key, expected in exact.items():
        if record[key] != expected or (isinstance(expected, bool) and type(record[key]) is not bool):
            raise V8FT1CPreservationBlocked("V8F_V8E_HISTORICAL_T1C_VALUE_INVALID:" + key)
    return dict(record)


def _validate_current_v8_state_t1(state: Mapping[str, Any]) -> dict[str, Any]:
    """Independently derive the observed T1C access-flag facts from the
    real, currently-committed V8_STATE.json T1 and last-attempt sections."""
    t1 = state.get("T1")
    attempt = state.get("last_real_t1_acquisition_attempt")
    if not isinstance(t1, Mapping) or not isinstance(attempt, Mapping):
        raise V8FT1CPreservationBlocked("V8F_V8_STATE_SCHEMA_INVALID")
    required_t1 = ("raw_data_acquired", "layer_b_opened", "validation_access_count", "ticker_count_frozen")
    required_attempt = ("t1_successfully_acquired", "t1_opened_for_research", "validation_accessed")
    for key in required_t1:
        if key not in t1:
            raise V8FT1CPreservationBlocked("V8F_V8_STATE_T1_FIELD_MISSING:" + key)
    for key in required_attempt:
        if key not in attempt:
            raise V8FT1CPreservationBlocked("V8F_V8_STATE_ATTEMPT_FIELD_MISSING:" + key)
    if t1["ticker_count_frozen"] != EXPECTED_V8F_T1C_TICKER_COUNT:
        raise V8FT1CPreservationBlocked("V8F_V8_STATE_T1_TICKER_COUNT_MISMATCH")
    return {
        "raw_acquisition_performed": t1["raw_data_acquired"] is not False or attempt["t1_successfully_acquired"] is not False,
        "research_opened": t1["layer_b_opened"] is not False or attempt["t1_opened_for_research"] is not False,
        "ohlcv_research_access": t1["validation_access_count"] is not None or attempt["validation_accessed"] is not False,
    }


def _default_public_chronology(repository_root: Path, lower: str, upper: str) -> list[dict[str, Any]]:
    if _git_text(repository_root, ["merge-base", "--is-ancestor", lower, upper], "V8F_PUBLIC_CHRONOLOGY_INVALID") != "":
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_CHRONOLOGY_INVALID")
    commits_text = _git_text(repository_root, ["rev-list", "--reverse", f"{lower}..{upper}"], "V8F_PUBLIC_CHRONOLOGY_INVALID")
    records: list[dict[str, Any]] = []
    for commit in commits_text.splitlines():
        paths_text = _git_text(
            repository_root,
            ["diff-tree", "--no-commit-id", "--name-only", "-r", commit],
            "V8F_PUBLIC_CHRONOLOGY_INVALID",
        )
        paths = [path for path in paths_text.splitlines() if path]
        records.append({"commit": commit, "paths": paths})
    return records


def _validate_public_chronology(records: Any) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_CHRONOLOGY_INVALID")
    validated: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"commit", "paths"}:
            raise V8FT1CPreservationBlocked("V8F_PUBLIC_CHRONOLOGY_UNVERIFIABLE")
        _require_hex(record["commit"], 40, "V8F_PUBLIC_CHRONOLOGY_UNVERIFIABLE")
        paths = record["paths"]
        if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
            raise V8FT1CPreservationBlocked("V8F_PUBLIC_CHRONOLOGY_UNVERIFIABLE")
        if len(set(paths)) != len(paths):
            raise V8FT1CPreservationBlocked("V8F_PUBLIC_CHRONOLOGY_UNVERIFIABLE")
        validated.append({"commit": record["commit"], "paths": list(paths)})
    return validated


def _default_fresh_t1c_public_evidence(
    repository_root: Path,
    preflight: Mapping[str, Any],
    reviewed_support_implementation_sha: str,
    *,
    chronology_reader: Callable[[Path, str, str], Any] | None = None,
    runtime_state_reader: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Independently derive fresh public T1C evidence from committed Git
    evidence: the real current V8_STATE.json, the real current
    V8C_TRUSTED_ALLOCATION.json, V8E's own real historical T1C preservation
    recheck artifact, the V8E T1C readiness terminal adjudication, and a
    non-empty committed chronology between the two.  No fact here is a bare
    caller assertion or a hardcoded constant standing in for an observation;
    every EXPECTED_* constant is compared against, never substituted for, an
    independently read value.
    """
    reviewed_sha = _validate_reviewed_support_implementation_binding(
        repository_root,
        preflight,
        reviewed_support_implementation_sha,
        runtime_state_reader=runtime_state_reader,
    )
    try:
        terminal_blob = resolve_git_blob(
            repository_root, V8F_V8E_PREDECESSOR_TERMINAL_COMMIT, V8F_V8E_TERMINAL_RECORD_GIT_PATH
        )
        historical_blob = resolve_git_blob(
            repository_root, V8E_T1C_PRESERVATION_RECHECK_COMMIT, V8E_T1C_PRESERVATION_RECHECK_GIT_PATH
        )
        trusted_allocation_blob = resolve_git_blob(repository_root, reviewed_sha, V8C_TRUSTED_ALLOCATION_GIT_PATH)
        state_blob = resolve_git_blob(repository_root, reviewed_sha, V8_STATE_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_PROVENANCE_INVALID") from error
    if terminal_blob != V8F_V8E_TERMINAL_RECORD_BLOB_SHA:
        raise V8FT1CPreservationBlocked("V8F_V8E_TERMINAL_BLOB_MISMATCH")
    if historical_blob != V8E_T1C_PRESERVATION_RECHECK_BLOB_SHA:
        raise V8FT1CPreservationBlocked("V8F_V8E_HISTORICAL_T1C_BLOB_MISMATCH")
    if trusted_allocation_blob != V8C_TRUSTED_ALLOCATION_BLOB_SHA or state_blob != V8_STATE_BLOB_SHA:
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_PROVENANCE_INVALID")

    terminal = _parse_v8e_terminal_record(
        read_git_object_bytes(repository_root, V8F_V8E_PREDECESSOR_TERMINAL_COMMIT, V8F_V8E_TERMINAL_RECORD_GIT_PATH)
    )
    historical = _validate_historical_v8e_t1c_record(
        _strict_public_json(
            read_git_object_bytes(repository_root, V8E_T1C_PRESERVATION_RECHECK_COMMIT, V8E_T1C_PRESERVATION_RECHECK_GIT_PATH),
            "V8F_V8E_HISTORICAL_T1C_INVALID_JSON",
            "V8F_V8E_HISTORICAL_T1C_DUPLICATE_KEY",
        )
    )
    allocation = _validate_current_trusted_t1c_allocation(
        _strict_public_json(
            read_git_object_bytes(repository_root, reviewed_sha, V8C_TRUSTED_ALLOCATION_GIT_PATH),
            "V8F_TRUSTED_ALLOCATION_INVALID_JSON",
            "V8F_TRUSTED_ALLOCATION_DUPLICATE_KEY",
        )
    )
    state = _strict_public_json(
        read_git_object_bytes(repository_root, reviewed_sha, V8_STATE_GIT_PATH),
        "V8F_V8_STATE_INVALID_JSON",
        "V8F_V8_STATE_DUPLICATE_KEY",
    )
    observed_t1 = _validate_current_v8_state_t1(state)
    chronology = _validate_public_chronology(
        (chronology_reader or _default_public_chronology)(repository_root, V8F_V8E_PREDECESSOR_TERMINAL_COMMIT, reviewed_sha)
    )
    if not chronology:
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_CHRONOLOGY_INVALID")

    # Every current absence below is a conjunction of the current safe state,
    # the current trusted allocation, and the historical V8E preservation
    # evidence -- never a bare assertion.
    evidence = {
        "schema_version": "V8F_T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE_V1",
        "evidence_role": "T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE",
        "study": V8F_STUDY_NAME,
        "reviewed_v8f_design_candidate_commit": V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "v8e_predecessor_terminal_commit": V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
        "v8e_terminal_disposition": terminal["disposition"],
        "v8e_terminal_failure_class": terminal["failure_class"],
        "v8e_terminal_artifact_blob_sha": terminal_blob,
        "allocation_artifact_self_hash": allocation["authorized_allocation_artifact_self_hash"],
        "t1c_ticker_count": allocation["t1c_ticker_count"],
        "t1c_ticker_list_sha256": allocation["t1c_ticker_list_sha256"],
        "parent_t_spare_ticker_list_sha256": allocation["parent_t_spare_ticker_list_sha256"],
        "remaining_t_spare_ticker_list_sha256": allocation["remaining_t_spare_ticker_list_sha256"],
        "t1c_raw_acquisition_performed": observed_t1["raw_acquisition_performed"] or historical["t1c_raw_acquisition_performed"],
        "t1c_research_opened": observed_t1["research_opened"] or historical["t1c_research_opened"],
        "t1c_ohlcv_research_access": observed_t1["ohlcv_research_access"] or historical["t1c_ohlcv_research_access"],
        "t1c_feature_access": historical["t1c_feature_access"],
        "t1c_outcome_access": historical["t1c_outcome_access"],
        "t1c_identities_publicly_exposed": historical["t1c_identities_publicly_exposed"],
        "t1c_membership_reassigned": (
            allocation["t1c_ticker_list_sha256"] != EXPECTED_V8F_T1C_TICKER_LIST_SHA256
            or allocation["logical_block"] != "T1C"
            or historical["t1c_membership_reassigned"] is not False
        ),
        "allocation_self_hash_unchanged": (
            allocation["authorized_allocation_artifact_self_hash"] == AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH
            and historical["allocation_self_hash_unchanged"] is True
        ),
        "parent_v8_provenance_unchanged": (
            allocation["parent_v8_partition_manifest_sha256"] == EXPECTED_V8_PARTITION_MANIFEST_SHA256
            and allocation["parent_v8_partition_implementation_commit"] == EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT
            and historical["parent_v8_provenance_unchanged"] is True
        ),
        "v8e_terminal_adjudication_authoritative": (
            terminal["disposition"] == V8F_V8E_TERMINAL_DISPOSITION
            and historical["v8c_terminal_adjudication_authoritative"] is True
        ),
        "fresh_public_preservation_evidence_result": "PASS",
    }
    return _validate_fresh_t1c_public_evidence(evidence)


def derive_fresh_t1c_public_evidence(
    repository_root: Path = CANONICAL_REPOSITORY_ROOT,
    *,
    reviewed_support_implementation_sha: str,
    preflight: Mapping[str, Any] | None = None,
    chronology_reader: Callable[[Path, str, str], Any] | None = None,
    runtime_state_reader: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve fresh public T1C evidence before any gate or private read."""
    verified_preflight = preflight or _default_public_preflight(repository_root)
    return _default_fresh_t1c_public_evidence(
        repository_root,
        _validate_public_preflight(verified_preflight),
        reviewed_support_implementation_sha,
        chronology_reader=chronology_reader,
        runtime_state_reader=runtime_state_reader,
    )


def _default_public_preflight(repository_root: Path = CANONICAL_REPOSITORY_ROOT) -> dict[str, Any]:
    status = _git_text(repository_root, ["status", "--porcelain"], "V8F_PUBLIC_GIT_UNAVAILABLE")
    branch = _git_text(repository_root, ["branch", "--show-current"], "V8F_PUBLIC_BRANCH_UNAVAILABLE")
    head = _git_text(repository_root, ["rev-parse", "HEAD"], "V8F_PUBLIC_HEAD_UNAVAILABLE")
    origin_head = _git_text(
        repository_root,
        ["rev-parse", "origin/" + V8F_PRODUCTION_BRANCH],
        "V8F_PUBLIC_ORIGIN_UNAVAILABLE",
    )
    origin_url = _git_text(repository_root, ["config", "--get", "remote.origin.url"], "V8F_PUBLIC_ORIGIN_UNAVAILABLE")
    if origin_url not in {
        "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "git@github.com:ta1k1-arakawa/stock-analyzer.git",
        "ssh://git@github.com/ta1k1-arakawa/stock-analyzer.git",
    }:
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    try:
        design_blob = resolve_git_blob(
            repository_root, V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT, "V8F_TRANSPORT_WINDOW_SEMANTICS_SUCCESSOR_DESIGN_DRAFT.md"
        )
        terminal_blob = resolve_git_blob(repository_root, V8F_V8E_PREDECESSOR_TERMINAL_COMMIT, V8F_V8E_TERMINAL_RECORD_GIT_PATH)
        trusted_allocation_blob = resolve_git_blob(repository_root, head, V8C_TRUSTED_ALLOCATION_GIT_PATH)
        trusted_partition_blob = resolve_git_blob(repository_root, head, "V8_TRUSTED_PARTITION.json")
        # Independently read-only verify the real V8 trust anchor rather than
        # asserting its provenance as a constant.
        anchor = read_and_verify_v8_trusted_partition_anchor(repository_root, head)
        trusted_allocation = _strict_public_json(
            read_git_object_bytes(repository_root, head, V8C_TRUSTED_ALLOCATION_GIT_PATH),
            "V8F_TRUSTED_ALLOCATION_INVALID_JSON",
            "V8F_TRUSTED_ALLOCATION_DUPLICATE_KEY",
        )
    except (V8CGitProvenanceBlocked, V8CProductionProvenanceBlocked, V8PartitionBlocked) as error:
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_PROVENANCE_INVALID") from error
    if (
        design_blob != V8F_DESIGN_CANDIDATE_BLOB_SHA
        or terminal_blob != V8F_V8E_TERMINAL_RECORD_BLOB_SHA
        or trusted_allocation_blob != V8C_TRUSTED_ALLOCATION_BLOB_SHA
    ):
        raise V8FT1CPreservationBlocked("V8F_PUBLIC_PROVENANCE_INVALID")
    if set(trusted_allocation) < {
        "v8c_allocation_implementation_commit",
        "parent_v8_partition_manifest_sha256",
        "parent_v8_partition_implementation_commit",
        "parent_t_spare_ticker_count",
        "parent_t_spare_ticker_list_sha256",
    }:
        raise V8FT1CPreservationBlocked("V8F_TRUSTED_ALLOCATION_SCHEMA_INVALID")
    return _validate_public_preflight(
        {
            "repository_identity": V8F_REPOSITORY_IDENTITY,
            "branch": branch,
            "head": head,
            "origin_head": origin_head,
            "worktree_clean": status == "",
            "reviewed_v8f_design_candidate_commit": V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
            "reviewed_v8f_design_blob_sha": design_blob,
            "v8e_terminal_commit": V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
            "v8e_terminal_blob_sha": terminal_blob,
            # Every value below is the observed value independently derived
            # from the real V8 trust anchor / V8C_TRUSTED_ALLOCATION.json Git
            # objects read above; `_validate_public_preflight` compares each
            # against its EXPECTED_* constant and fails closed on mismatch.
            "trusted_partition_blob_sha": trusted_partition_blob,
            "partition_manifest_sha256": anchor["authorized_partition_manifest_sha256"],
            "partition_implementation_commit": anchor["authorized_partition_implementation_git_commit"],
            "v8c_allocation_implementation_commit": trusted_allocation["v8c_allocation_implementation_commit"],
            "parent_t_spare_ticker_count": trusted_allocation["parent_t_spare_ticker_count"],
            "parent_t_spare_ticker_list_sha256": trusted_allocation["parent_t_spare_ticker_list_sha256"],
        }
    )


def _read_partition_manifest_bytes(raw: bytes) -> dict[str, Any]:
    manifest = _strict_json_object(raw, "V8F_PARTITION_MANIFEST_INVALID_JSON", "V8F_PARTITION_MANIFEST_DUPLICATE_KEY")
    if set(manifest) != set(MANIFEST_FIELDS):
        raise V8FT1CPreservationBlocked("V8F_PARTITION_MANIFEST_SCHEMA_INVALID")
    if manifest["manifest_sha256"] != v8_canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise V8FT1CPreservationBlocked("V8F_PARTITION_MANIFEST_SHA_MISMATCH")
    try:
        require_v8_git_commit(manifest["partition_implementation_git_commit"])
    except V8PartitionBlocked as error:
        raise V8FT1CPreservationBlocked("V8F_PARTITION_IMPLEMENTATION_COMMIT_INVALID") from error
    if (
        manifest["schema_version"] != V8_PARTITION_SCHEMA_VERSION
        or manifest["source_snapshot_semantics"] != SOURCE_SNAPSHOT_SEMANTICS
        or manifest["source_snapshot_clarification_commit"] != SOURCE_SNAPSHOT_CLARIFICATION_COMMIT
        or manifest["v4_raw_sha_equality_required"] is not False
        or manifest["source_reproduction_status"] != "PASS"
        or manifest["t0_reproduction_status"] != "PASS"
    ):
        raise V8FT1CPreservationBlocked("V8F_PARTITION_MANIFEST_FROZEN_BINDING_INVALID")
    assignments = manifest["block_assignments"]
    if not isinstance(assignments, Mapping) or set(_REQUIRED_BLOCK_KEYS) - set(assignments):
        raise V8FT1CPreservationBlocked("V8F_PARTITION_BLOCK_ASSIGNMENT_INVALID")
    for key in _REQUIRED_BLOCK_KEYS:
        if not isinstance(assignments[key], list):
            raise V8FT1CPreservationBlocked("V8F_PARTITION_BLOCK_ASSIGNMENT_INVALID")
    return manifest


def _verify_private_artifacts(
    allocation_raw: bytes,
    partition_manifest_raw: bytes,
    *,
    expected_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
    expected_parent_t_spare_ticker_list_sha256: str = EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
    expected_t1c_ticker_list_sha256: str = EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
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
            raise V8FT1CPreservationBlocked("V8F_PARTITION_MANIFEST_SHA_MISMATCH_TRUSTED")
        if manifest["partition_implementation_git_commit"] != expected_partition_implementation_commit:
            raise V8FT1CPreservationBlocked("V8F_PARTITION_IMPLEMENTATION_COMMIT_MISMATCH_TRUSTED")
        if manifest["study_name"] != "V8_HISTORICAL_RESEARCH":
            raise V8FT1CPreservationBlocked("V8F_PARTITION_STUDY_MISMATCH")
        if manifest["design_commit"] != "c414d3191cba356734d7ed08bdf1abc7d51fc384":
            raise V8FT1CPreservationBlocked("V8F_PARTITION_DESIGN_COMMIT_MISMATCH")
        parent = assignments["T_spare"]
        if len(parent) != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
            raise V8FT1CPreservationBlocked("V8F_PARENT_T_SPARE_COUNT_MISMATCH")
        if manifest["t_spare_ticker_list_sha256"] != expected_parent_t_spare_ticker_list_sha256:
            raise V8FT1CPreservationBlocked("V8F_PARENT_T_SPARE_HASH_MISMATCH")
        if artifact.get("parent_v8_partition_manifest_sha256") != expected_partition_manifest_sha256:
            raise V8FT1CPreservationBlocked("V8F_ALLOCATION_PARENT_MANIFEST_MISMATCH")
        if artifact.get("parent_v8_partition_implementation_commit") != expected_partition_implementation_commit:
            raise V8FT1CPreservationBlocked("V8F_ALLOCATION_PARENT_IMPLEMENTATION_MISMATCH")
        if artifact.get("v8c_allocation_implementation_commit") != expected_v8c_allocation_implementation_commit:
            raise V8FT1CPreservationBlocked("V8F_ALLOCATION_IMPLEMENTATION_NOT_REVIEWED")
        if artifact.get("artifact_self_hash") != expected_allocation_artifact_self_hash:
            raise V8FT1CPreservationBlocked("V8F_ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH_TRUSTED")
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
    except (V8FT1CPreservationBlocked, V8CAllocationBlocked, V8CAllocationVerificationBlocked) as error:
        if isinstance(error, V8FT1CPreservationBlocked):
            raise
        raise V8FT1CPreservationBlocked("V8F_PRIVATE_ALLOCATION_VERIFICATION_BLOCKED") from error
    if safe["artifact_self_hash"] != expected_allocation_artifact_self_hash:
        raise V8FT1CPreservationBlocked("V8F_ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH_TRUSTED")
    if safe["t1c_ticker_count"] != EXPECTED_V8F_T1C_TICKER_COUNT:
        raise V8FT1CPreservationBlocked("V8F_T1C_COUNT_MISMATCH")
    if safe["t1c_ticker_list_sha256"] != expected_t1c_ticker_list_sha256:
        raise V8FT1CPreservationBlocked("V8F_T1C_HASH_MISMATCH")
    if safe["parent_t_spare_ticker_list_sha256"] != expected_parent_t_spare_ticker_list_sha256:
        raise V8FT1CPreservationBlocked("V8F_PARENT_T_SPARE_HASH_MISMATCH")
    if safe["remaining_t_spare_ticker_list_sha256"] != expected_remaining_t_spare_ticker_list_sha256:
        raise V8FT1CPreservationBlocked("V8F_REMAINING_T_SPARE_HASH_MISMATCH")
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


PARTITION_MANIFEST_BASENAME = "partition_manifest.json"


def validate_candidate_partition_manifest_paths(
    candidate_partition_manifest_paths: Sequence[str | os.PathLike[str]],
    repository_root: Path = CANONICAL_REPOSITORY_ROOT,
) -> tuple[Path, ...]:
    """Pre-gate, metadata-only candidate validation.  Reads no bytes.

    Every candidate must be an absolute path outside the repository whose
    basename is exactly ``partition_manifest.json``; the normalized list
    must be non-empty and free of duplicates.  This only normalizes and
    compares path strings (and, via ``Path.resolve``, filesystem metadata
    such as symlinks); it never opens or reads a candidate file.
    """
    if isinstance(candidate_partition_manifest_paths, (str, bytes)):
        raise V8FT1CPreservationBlocked("V8F_LOCATOR_CANDIDATE_LIST_INVALID")
    try:
        candidate_list = list(candidate_partition_manifest_paths)
    except TypeError as error:
        raise V8FT1CPreservationBlocked("V8F_LOCATOR_CANDIDATE_LIST_INVALID") from error
    if len(candidate_list) == 0:
        raise V8FT1CPreservationBlocked("V8F_LOCATOR_CANDIDATE_LIST_EMPTY")
    normalized: list[Path] = []
    seen: set[Path] = set()
    for value in candidate_list:
        resolved = _require_safe_external_path(value, repository_root, "V8F_LOCATOR_CANDIDATE_PATH_INVALID")
        if resolved.name != PARTITION_MANIFEST_BASENAME:
            raise V8FT1CPreservationBlocked("V8F_LOCATOR_CANDIDATE_BASENAME_INVALID")
        if resolved in seen:
            raise V8FT1CPreservationBlocked("V8F_LOCATOR_CANDIDATE_DUPLICATE_PATH")
        seen.add(resolved)
        normalized.append(resolved)
    return tuple(normalized)


def _locate_authorized_partition_manifest(
    private_reader: Callable[[Path], bytes],
    candidate_paths: Sequence[Path],
    *,
    expected_partition_manifest_sha256: str,
    expected_partition_implementation_commit: str,
) -> tuple[bytes, dict[str, int]]:
    """Single post-gate content-addressed scan over pre-validated candidates.

    A candidate can only ever fail to become a match; an unreadable,
    malformed, or non-matching candidate never aborts the scan of the
    remaining candidates, and this never trusts a candidate's self-declared
    ``manifest_sha256`` -- ``_read_partition_manifest_bytes`` always
    recomputes it first.  Exactly one exact match is required; zero or more
    than one is fail-closed.  Never returns or logs a candidate path.
    """
    candidates_read_count = 0
    matches: list[bytes] = []
    for candidate_path in candidate_paths:
        try:
            candidate_raw = private_reader(candidate_path)
        except OSError:
            continue
        candidates_read_count += 1
        try:
            manifest = _read_partition_manifest_bytes(candidate_raw)
        except V8FT1CPreservationBlocked:
            continue
        if (
            manifest["manifest_sha256"] == expected_partition_manifest_sha256
            and manifest["partition_implementation_git_commit"] == expected_partition_implementation_commit
        ):
            matches.append(candidate_raw)
    stats = {
        "candidate_count": len(candidate_paths),
        "candidates_read_count": candidates_read_count,
        "exact_match_count": len(matches),
    }
    if len(matches) == 0:
        raise V8FT1CPreservationBlocked("V8F_LOCATOR_ZERO_MATCHING_CANDIDATES")
    if len(matches) > 1:
        raise V8FT1CPreservationBlocked("V8F_LOCATOR_MULTIPLE_MATCHING_CANDIDATES")
    return matches[0], stats


def _prepare_locator_execution_paths(
    *,
    state_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    allocation_artifact_path: str | os.PathLike[str],
    candidates: Sequence[Path],
    repository_root: Path,
    receipt_key: str,
) -> tuple[Path, Path, Path]:
    """Locator analogue of `_prepare_execution_paths`: the same pre-gate,
    metadata-only output/allocation/state safety properties (absolute,
    outside the repository, no existing destination, no collision), checked
    against every already-normalized candidate path instead of a single
    exact manifest path.  Never reads a candidate or allocation byte.
    """
    state = _require_safe_external_path(state_root, repository_root, "V8F_STATE_PATH_INVALID")
    output = _require_safe_external_path(output_path, repository_root, "V8F_OUTPUT_PATH_INVALID")
    allocation = _require_safe_external_path(allocation_artifact_path, repository_root, "V8F_PRIVATE_PATH_INVALID")
    if (
        allocation in candidates
        or output == allocation
        or output in candidates
        or output == state / (receipt_key + ".json")
    ):
        raise V8FT1CPreservationBlocked("V8F_OUTPUT_PATH_COLLISION")
    try:
        state.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8FT1CPreservationBlocked("V8F_OUTPUT_OR_STATE_PREPARATION_FAILED") from error
    if not state.is_dir() or not output.parent.is_dir() or output.exists():
        raise V8FT1CPreservationBlocked("V8F_OUTPUT_OR_STATE_PREPARATION_FAILED")
    if not allocation.is_file():
        raise V8FT1CPreservationBlocked("V8F_PRIVATE_ARTIFACT_UNAVAILABLE")
    if (state / (receipt_key + ".json")).exists():
        raise V8FT1CPreservationBlocked("V8F_GATE_ALREADY_CONSUMED")
    return state, output, allocation


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
            raise V8FT1CPreservationBlocked("V8F_PRIVATE_PUBLIC_EVIDENCE_MISMATCH:" + key)
    artifact = {
        "schema_version": "V8F_T1C_PRESERVATION_RECHECK_V1",
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "study": V8F_STUDY_NAME,
        "reviewed_v8f_design_candidate_commit": V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "source_v8e_terminal_commit": V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
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
        "v8e_terminal_adjudication_authoritative": fresh["v8e_terminal_adjudication_authoritative"],
        "preservation_recheck_result": "PASS",
    }
    if set(artifact) != set(V8F_PRESERVATION_ARTIFACT_FIELDS):
        raise V8FT1CPreservationBlocked("V8F_PRESERVATION_ARTIFACT_SCHEMA_INVALID")
    return artifact


def _validate_public_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping) or set(artifact) != set(V8F_PRESERVATION_ARTIFACT_FIELDS):
        raise V8FT1CPreservationBlocked("V8F_PRESERVATION_ARTIFACT_SCHEMA_INVALID")
    exact = {
        "schema_version": "V8F_T1C_PRESERVATION_RECHECK_V1",
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "study": V8F_STUDY_NAME,
        "reviewed_v8f_design_candidate_commit": V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "source_v8e_terminal_commit": V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
        "allocation_artifact_self_hash": AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
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
        "v8e_terminal_adjudication_authoritative": True,
        "preservation_recheck_result": "PASS",
    }
    for key, expected in exact.items():
        if artifact[key] != expected or (isinstance(expected, bool) and type(artifact[key]) is not bool):
            raise V8FT1CPreservationBlocked("V8F_PRESERVATION_ARTIFACT_VALUE_MISMATCH:" + key)
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
            artifact, "V8F_PRESERVATION_ARTIFACT_INVALID_JSON", "V8F_PRESERVATION_ARTIFACT_DUPLICATE_KEY"
        )
        artifact_bytes = artifact
    else:
        artifact_value = dict(artifact)
        artifact_bytes = None
    if isinstance(receipt, bytes):
        receipt_value = _strict_json_object(receipt, "V8F_RECEIPT_INVALID_JSON", "V8F_RECEIPT_DUPLICATE_KEY")
        receipt_bytes = receipt
    else:
        receipt_value = dict(receipt)
        receipt_bytes = None
    artifact_value = _validate_public_artifact(artifact_value)
    receipt_value = _validate_receipt(receipt_value, receipt_key)
    if artifact_value["reviewed_v8f_design_candidate_commit"] != receipt_value["reviewed_v8f_design_candidate_commit"]:
        raise V8FT1CPreservationBlocked("V8F_ARTIFACT_RECEIPT_CANDIDATE_MISMATCH")
    if artifact_value["allocation_artifact_self_hash"] != receipt_value["authorized_allocation_artifact_self_hash"]:
        raise V8FT1CPreservationBlocked("V8F_ARTIFACT_RECEIPT_ALLOCATION_MISMATCH")
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


def _publish_preservation_artifact(artifact: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Write-once atomic publication of the canonical T1C preservation
    artifact: canonical JSON bytes, staging write, fsync file, atomic
    no-overwrite link publication, fsync directory, staging cleanup.  Never
    replaces an existing destination.  Shared by every V8F T1C preservation
    execution seam so publication semantics cannot drift between them.
    """
    if output.exists():
        raise V8FT1CPreservationBlocked("V8F_PRESERVATION_ARTIFACT_ALREADY_EXISTS")
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
            raise V8FT1CPreservationBlocked("V8F_PRESERVATION_ARTIFACT_ALREADY_EXISTS") from error
        except OSError as error:
            raise V8FT1CPreservationBlocked("V8F_PRESERVATION_ARTIFACT_ATOMIC_PUBLISH_FAILED") from error
        _fsync_directory(output.parent)
    except V8FT1CPreservationBlocked:
        raise
    except OSError as error:
        raise V8FT1CPreservationBlocked("V8F_PRESERVATION_ARTIFACT_WRITE_FAILED") from error
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass
    return dict(artifact)


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
    reviewed_support_implementation_sha: str,
    public_evidence_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    runtime_state_reader: Callable[[Path, str], Mapping[str, Any]] | None = None,
    reviewed_v8f_design_candidate_commit: str = V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    authorized_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
) -> dict[str, Any]:
    """DI-only future execution boundary; never called with real private paths here."""
    preflight = _validate_public_preflight(public_preflight())
    reviewed_support_sha = _validate_reviewed_support_implementation_binding(
        repository_root,
        preflight,
        reviewed_support_implementation_sha,
        runtime_state_reader=runtime_state_reader,
    )
    validate_authorization_identity(
        authorization_identity,
        reviewed_v8f_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    receipt_key = compute_receipt_key(
        authorization_identity,
        reviewed_v8f_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    if public_evidence_resolver is None:
        fresh_public_evidence = _default_fresh_t1c_public_evidence(
            repository_root,
            preflight,
            reviewed_support_sha,
            runtime_state_reader=runtime_state_reader,
        )
    else:
        fresh_public_evidence = _validate_fresh_t1c_public_evidence(public_evidence_resolver(preflight))
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
        reviewed_v8f_design_candidate_commit=reviewed_v8f_design_candidate_commit,
        authorized_allocation_artifact_self_hash=authorized_allocation_artifact_self_hash,
    )
    try:
        allocation_raw = private_reader(allocation_path)
        manifest_raw = private_reader(manifest_path)
    except OSError as error:
        raise V8FT1CPreservationBlocked("V8F_PRIVATE_ARTIFACT_READ_FAILED") from error
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
    return _publish_preservation_artifact(artifact, output)


def resolve_and_verify_t1c_preservation(
    authorization_identity: str,
    *,
    reviewed_support_implementation_sha: str,
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Prepared future entry point; not executed by this support task."""
    return _execute_with_dependencies(
        authorization_identity=authorization_identity,
        reviewed_support_implementation_sha=reviewed_support_implementation_sha,
        state_root=CANONICAL_V8F_T1C_PRESERVATION_STATE_ROOT,
        output_path=output_path,
        allocation_artifact_path=allocation_artifact_path,
        partition_manifest_path=partition_manifest_path,
        repository_root=CANONICAL_REPOSITORY_ROOT,
        public_preflight=lambda: _default_public_preflight(CANONICAL_REPOSITORY_ROOT),
        private_reader=lambda path: path.read_bytes(),
        gate_consumer=consume_gate_once,
        clock=clock or (lambda: datetime.now(timezone.utc)),
    )


def _execute_locator_with_dependencies(
    *,
    authorization_identity: str,
    state_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    allocation_artifact_path: str | os.PathLike[str],
    candidate_partition_manifest_paths: Sequence[str | os.PathLike[str]],
    repository_root: Path,
    public_preflight: Callable[[], Mapping[str, Any]],
    private_reader: Callable[[Path], bytes],
    gate_consumer: Callable[..., Mapping[str, Any]],
    clock: Callable[[], datetime],
    reviewed_support_implementation_sha: str,
    public_evidence_resolver: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    runtime_state_reader: Callable[[Path, str], Mapping[str, Any]] | None = None,
    reviewed_v8f_design_candidate_commit: str = V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
    authorized_allocation_artifact_self_hash: str = AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
) -> dict[str, Any]:
    """DI-only future content-addressed locator boundary; never called with
    real private paths here.  Mirrors `_execute_with_dependencies`'s exact
    pre-gate ordering and completes the *same* preservation transaction --
    durably publishing the same canonical `V8F_T1C_PRESERVATION_RECHECK`
    artifact -- replacing only its single exact `partition_manifest_path`
    with a pre-validated candidate list resolved by content address after
    the same one-shot gate is durably consumed.
    """
    preflight = _validate_public_preflight(public_preflight())
    reviewed_support_sha = _validate_reviewed_support_implementation_binding(
        repository_root,
        preflight,
        reviewed_support_implementation_sha,
        runtime_state_reader=runtime_state_reader,
    )
    validate_authorization_identity(
        authorization_identity,
        reviewed_v8f_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    receipt_key = compute_receipt_key(
        authorization_identity,
        reviewed_v8f_design_candidate_commit,
        authorized_allocation_artifact_self_hash,
    )
    if public_evidence_resolver is None:
        fresh_public_evidence = _default_fresh_t1c_public_evidence(
            repository_root,
            preflight,
            reviewed_support_sha,
            runtime_state_reader=runtime_state_reader,
        )
    else:
        fresh_public_evidence = _validate_fresh_t1c_public_evidence(public_evidence_resolver(preflight))

    # Pre-gate, metadata-only: normalize/dedupe/basename/outside-repo only.
    candidates = validate_candidate_partition_manifest_paths(
        candidate_partition_manifest_paths, repository_root
    )
    state, output, allocation_path = _prepare_locator_execution_paths(
        state_root=state_root,
        output_path=output_path,
        allocation_artifact_path=allocation_artifact_path,
        candidates=candidates,
        repository_root=repository_root,
        receipt_key=receipt_key,
    )

    # Exact frozen boundary: no private reader is called before durable receipt.
    gate_consumer(
        state,
        authorization_identity,
        clock=clock,
        reviewed_v8f_design_candidate_commit=reviewed_v8f_design_candidate_commit,
        authorized_allocation_artifact_self_hash=authorized_allocation_artifact_self_hash,
    )

    expected_partition_manifest_sha256 = preflight["partition_manifest_sha256"]
    expected_partition_implementation_commit = preflight["partition_implementation_commit"]
    expected_allocation_artifact_self_hash = preflight.get(
        "authorized_allocation_artifact_self_hash", authorized_allocation_artifact_self_hash
    )

    try:
        allocation_raw = private_reader(allocation_path)
    except OSError as error:
        raise V8FT1CPreservationBlocked("V8F_PRIVATE_ARTIFACT_READ_FAILED") from error
    allocation_artifact = read_t1c_allocation_artifact_bytes(allocation_raw)
    if allocation_artifact.get("artifact_self_hash") != expected_allocation_artifact_self_hash:
        raise V8FT1CPreservationBlocked("V8F_ALLOCATION_ARTIFACT_SELF_HASH_MISMATCH_TRUSTED")
    if allocation_artifact.get("parent_v8_partition_manifest_sha256") != expected_partition_manifest_sha256:
        raise V8FT1CPreservationBlocked("V8F_ALLOCATION_PARENT_MANIFEST_MISMATCH")
    if (
        allocation_artifact.get("parent_v8_partition_implementation_commit")
        != expected_partition_implementation_commit
    ):
        raise V8FT1CPreservationBlocked("V8F_ALLOCATION_PARENT_IMPLEMENTATION_MISMATCH")

    matched_manifest_raw, stats = _locate_authorized_partition_manifest(
        private_reader,
        candidates,
        expected_partition_manifest_sha256=expected_partition_manifest_sha256,
        expected_partition_implementation_commit=expected_partition_implementation_commit,
    )

    private_summary = _verify_private_artifacts(
        allocation_raw,
        matched_manifest_raw,
        expected_allocation_artifact_self_hash=expected_allocation_artifact_self_hash,
        expected_parent_t_spare_ticker_list_sha256=preflight["parent_t_spare_ticker_list_sha256"],
        expected_partition_manifest_sha256=expected_partition_manifest_sha256,
        expected_partition_implementation_commit=expected_partition_implementation_commit,
        expected_v8c_allocation_implementation_commit=preflight["v8c_allocation_implementation_commit"],
    )
    artifact = _build_public_artifact(private_summary, fresh_public_evidence)
    _publish_preservation_artifact(artifact, output)
    return {
        "result": "PASS",
        "candidate_count": stats["candidate_count"],
        "candidates_read_count": stats["candidates_read_count"],
        "exact_match_count": stats["exact_match_count"],
        "private_read_count": 1 + stats["candidates_read_count"],
        "expected_partition_manifest_sha256": expected_partition_manifest_sha256,
        "artifact_written": True,
        "gate_receipt_key_sha256": receipt_key,
    }


def resolve_and_verify_t1c_preservation_by_content_address(
    authorization_identity: str,
    *,
    reviewed_support_implementation_sha: str,
    allocation_artifact_path: str | os.PathLike[str],
    candidate_partition_manifest_paths: Sequence[str | os.PathLike[str]],
    output_path: str | os.PathLike[str],
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Prepared future entry point: locate and verify the already-authorized
    V8 partition manifest among externally-supplied candidate paths by
    content address, then durably publish the same canonical
    `V8F_T1C_PRESERVATION_RECHECK` artifact the exact-path seam publishes --
    only after the existing one-shot V8F T1C preservation gate is durably
    consumed.  Not executed by this support task.

    ``candidate_partition_manifest_paths`` must already be a metadata-only
    resolved candidate list (a future PowerShell/runbook step); this module
    performs no filesystem-wide discovery of its own.  The exact-path
    production entry point, `resolve_and_verify_t1c_preservation`, is
    unchanged and remains available for compatibility.
    """
    return _execute_locator_with_dependencies(
        authorization_identity=authorization_identity,
        reviewed_support_implementation_sha=reviewed_support_implementation_sha,
        state_root=CANONICAL_V8F_T1C_PRESERVATION_STATE_ROOT,
        output_path=output_path,
        allocation_artifact_path=allocation_artifact_path,
        candidate_partition_manifest_paths=candidate_partition_manifest_paths,
        repository_root=CANONICAL_REPOSITORY_ROOT,
        public_preflight=lambda: _default_public_preflight(CANONICAL_REPOSITORY_ROOT),
        private_reader=lambda path: path.read_bytes(),
        gate_consumer=consume_gate_once,
        clock=clock or (lambda: datetime.now(timezone.utc)),
    )


__all__ = [
    "AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH",
    "CANONICAL_V8F_T1C_PRESERVATION_STATE_ROOT",
    "EXPECTED_PARENT_T_SPARE_TICKER_COUNT",
    "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256",
    "EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256",
    "EXPECTED_V8F_T1C_TICKER_COUNT",
    "EXPECTED_V8F_T1C_TICKER_LIST_SHA256",
    "PARTITION_MANIFEST_BASENAME",
    "V8F_DESIGN_CANDIDATE_BLOB_SHA",
    "V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT",
    "V8F_PRESERVATION_ARTIFACT_FIELDS",
    "V8F_FRESH_T1C_PUBLIC_EVIDENCE_FIELDS",
    "V8F_RECEIPT_FIELDS",
    "V8F_STUDY_NAME",
    "V8F_T1C_PRESERVATION_GATE",
    "V8FT1CPreservationBlocked",
    "authorization_identity_sha256",
    "compute_receipt_key",
    "consume_gate_once",
    "derive_fresh_t1c_public_evidence",
    "gate_receipt_bytes_sha256",
    "read_gate_receipt",
    "resolve_and_verify_t1c_preservation",
    "resolve_and_verify_t1c_preservation_by_content_address",
    "validate_authorization_identity",
    "validate_candidate_partition_manifest_paths",
    "verify_t1c_preservation_artifact_and_receipt",
    "verify_t1c_preservation_artifact_bytes",
]
