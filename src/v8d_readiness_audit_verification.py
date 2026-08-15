"""Canonical machine-local bindings for verified V8D readiness audits.

The production readers in this module accept no paths or verification seams.
They re-derive the audit paths from a safe, stage-specific receipt and then
run the existing independent production audit verifier again.  The writers
accept only locators from the just-completed run; those locators never become
authority by themselves.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.v8d_audit import (
    V8DAuditVerificationBlocked,
    verify_aggregate_production,
    verify_dossier_production,
)
from src.v8d_human_gate_consumption import CANONICAL_CONSUMPTION_STATE_ROOT
from src.v8d_readiness import CANONICAL_PRODUCTION_AUDIT_ROOT
from src.v8d_transport import (
    CANONICAL_PARSER_CLASSIFIER_BLOB,
    CANONICAL_PARSER_CLASSIFIER_COMMIT,
    FROZEN_DESIGN_COMMIT,
    SENTINEL_END_EXCLUSIVE,
    SENTINEL_INDICES,
    SENTINEL_START,
    canonical_sha256,
)

STUDY = "V8D_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8D_READINESS_AUDIT_VERIFICATION_RECEIPT_V1"
ARTIFACT_ROLE = "READINESS_TRANSPORT_AUDIT_VERIFICATION"
VERIFICATION_RESULT = "PASS"

T1C_VERIFICATION_STAGE = "READ_ONLY_T1C_READINESS_TRANSPORT_AUDIT_VERIFICATION"
T2_VERIFICATION_STAGE = "READ_ONLY_T2_READINESS_TRANSPORT_AUDIT_VERIFICATION"
T1C_LOGICAL_STAGE = "T1C_TRANSPORT_READINESS"
T2_LOGICAL_STAGE = "T2_TRANSPORT_READINESS"

CANONICAL_READINESS_AUDIT_VERIFICATION_ROOT = (
    CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8d-readiness-audit-verification-state"
)
T1C_RECEIPT_PATH = CANONICAL_READINESS_AUDIT_VERIFICATION_ROOT / "t1c-readiness-audit-verification.json"
T2_RECEIPT_PATH = CANONICAL_READINESS_AUDIT_VERIFICATION_ROOT / "t2-readiness-audit-verification.json"

RECEIPT_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "verification_stage",
    "logical_stage",
    "verification_result",
    "frozen_design_commit",
    "reviewed_production_implementation_commit",
    "aggregate_filename",
    "aggregate_artifact_self_hash",
    "dossier_bindings",
    "gate_receipt_key_sha256",
    "gate_receipt_bytes_sha256",
    "authorization_identity_sha256",
    "receipt_self_hash",
)
DOSSIER_BINDING_FIELDS = ("filename", "audit_artifact_self_hash")


class V8DReadinessAuditVerificationBlocked(RuntimeError):
    """Fail-closed canonical readiness-audit receipt error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise V8DReadinessAuditVerificationBlocked(duplicate_reason)
            value[key] = item
        return value

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8DReadinessAuditVerificationBlocked(invalid_reason) from error
    if not isinstance(value, dict):
        raise V8DReadinessAuditVerificationBlocked(invalid_reason)
    return value


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise V8DReadinessAuditVerificationBlocked(reason)
    return value


def _stage_constants(logical_stage: str) -> tuple[str, str, Path]:
    if logical_stage == T1C_LOGICAL_STAGE:
        return T1C_VERIFICATION_STAGE, T1C_LOGICAL_STAGE, T1C_RECEIPT_PATH
    if logical_stage == T2_LOGICAL_STAGE:
        return T2_VERIFICATION_STAGE, T2_LOGICAL_STAGE, T2_RECEIPT_PATH
    raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_VERIFICATION_STAGE_INVALID")


def _receipt_fields_for_stage(verification_stage: str) -> tuple[str, str, Path]:
    if verification_stage == T1C_VERIFICATION_STAGE:
        return T1C_VERIFICATION_STAGE, T1C_LOGICAL_STAGE, T1C_RECEIPT_PATH
    if verification_stage == T2_VERIFICATION_STAGE:
        return T2_VERIFICATION_STAGE, T2_LOGICAL_STAGE, T2_RECEIPT_PATH
    raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_VERIFICATION_STAGE_INVALID")


def _canonical_root(root: Path) -> Path:
    try:
        return root.resolve(strict=True)
    except OSError as error:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_CANONICAL_ROOT_UNAVAILABLE") from error


def _safe_audit_file(path_value: str | Path, audit_root: Path, *, reason: str) -> tuple[Path, str]:
    try:
        supplied = Path(path_value)
    except (TypeError, ValueError) as error:
        raise V8DReadinessAuditVerificationBlocked(reason) from error
    if supplied.is_symlink():
        raise V8DReadinessAuditVerificationBlocked(reason)
    try:
        resolved = supplied.resolve(strict=True)
        root = _canonical_root(audit_root)
    except OSError as error:
        raise V8DReadinessAuditVerificationBlocked(reason) from error
    if not resolved.is_file() or resolved.parent != root:
        raise V8DReadinessAuditVerificationBlocked(reason)
    if resolved.name in {"", ".", ".."} or resolved.name != supplied.name:
        raise V8DReadinessAuditVerificationBlocked(reason)
    return resolved, resolved.name


def _read_json_file(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8DReadinessAuditVerificationBlocked(f"V8D_READINESS_AUDIT_{label}_READ_FAILED") from error
    return _strict_json_object(
        raw,
        invalid_reason=f"V8D_READINESS_AUDIT_{label}_INVALID_JSON",
        duplicate_reason=f"V8D_READINESS_AUDIT_{label}_DUPLICATE_KEY",
    )


def _require_verified_dossier_bindings(
    dossiers: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any], *, logical_stage: str
) -> tuple[list[dict[str, str]], dict[str, str]]:
    if len(dossiers) != 3:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_DOSSIER_COUNT_INVALID")
    bindings: list[dict[str, str]] = []
    reference: dict[str, str] | None = None
    for dossier in dossiers:
        if dossier.get("logical_stage") != logical_stage:
            raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_DOSSIER_STAGE_INVALID")
        binding = {
            "gate_receipt_key_sha256": _require_hex(
                dossier.get("gate_receipt_key_sha256"), 64, "V8D_READINESS_AUDIT_GATE_KEY_INVALID"
            ),
            "gate_receipt_bytes_sha256": _require_hex(
                dossier.get("gate_receipt_bytes_sha256"), 64, "V8D_READINESS_AUDIT_GATE_BYTES_HASH_INVALID"
            ),
            "authorization_identity_sha256": _require_hex(
                dossier.get("authorization_identity_sha256"), 64, "V8D_READINESS_AUDIT_AUTHORIZATION_HASH_INVALID"
            ),
            "audit_artifact_self_hash": _require_hex(
                dossier.get("audit_artifact_self_hash"), 64, "V8D_READINESS_AUDIT_DOSSIER_HASH_INVALID"
            ),
        }
        if reference is None:
            reference = binding
        elif any(
            binding[key] != reference[key]
            for key in ("gate_receipt_key_sha256", "gate_receipt_bytes_sha256", "authorization_identity_sha256")
        ):
            raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_GATE_BINDING_MISMATCH")
        bindings.append({"filename": "", "audit_artifact_self_hash": binding["audit_artifact_self_hash"]})
    if aggregate.get("result") != VERIFICATION_RESULT:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RESULT_NOT_PASS")
    if reference is None:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_DOSSIER_COUNT_INVALID")
    return bindings, reference


def _require_readiness_pass(aggregate: Mapping[str, Any], logical_stage: str) -> None:
    if aggregate.get("logical_stage") != logical_stage:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_STAGE_MISMATCH")
    if aggregate.get("result") != VERIFICATION_RESULT:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RESULT_NOT_PASS")
    if aggregate.get("sentinel_indices") != list(SENTINEL_INDICES):
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_SENTINEL_BINDING_INVALID")
    if aggregate.get("sentinel_count") != 3 or aggregate.get("sentinel_pass_count") != 3:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_SENTINEL_PASS_COUNT_INVALID")
    if aggregate.get("window_start") != SENTINEL_START or aggregate.get("window_end_exclusive") != SENTINEL_END_EXCLUSIVE:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_WINDOW_BINDING_INVALID")
    if aggregate.get("frozen_design_commit") != FROZEN_DESIGN_COMMIT:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_DESIGN_BINDING_INVALID")
    if aggregate.get("canonical_parser_classifier_commit") != CANONICAL_PARSER_CLASSIFIER_COMMIT:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_CLASSIFIER_COMMIT_INVALID")
    if aggregate.get("canonical_parser_classifier_blob") != CANONICAL_PARSER_CLASSIFIER_BLOB:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_CLASSIFIER_BLOB_INVALID")


def _build_receipt(
    *,
    verification_stage: str,
    logical_stage: str,
    aggregate_filename: str,
    aggregate: Mapping[str, Any],
    dossier_bindings: Sequence[Mapping[str, str]],
    gate_binding: Mapping[str, str],
) -> dict[str, Any]:
    if len(dossier_bindings) != 3 or any(set(binding) != set(DOSSIER_BINDING_FIELDS) for binding in dossier_bindings):
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_DOSSIER_BINDINGS_INVALID")
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY,
        "artifact_role": ARTIFACT_ROLE,
        "verification_stage": verification_stage,
        "logical_stage": logical_stage,
        "verification_result": VERIFICATION_RESULT,
        "frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "reviewed_production_implementation_commit": _require_hex(
            aggregate.get("reviewed_production_implementation_commit"), 40,
            "V8D_READINESS_AUDIT_IMPLEMENTATION_SHA_INVALID",
        ),
        "aggregate_filename": aggregate_filename,
        "aggregate_artifact_self_hash": _require_hex(
            aggregate.get("aggregate_self_hash"), 64, "V8D_READINESS_AUDIT_AGGREGATE_HASH_INVALID"
        ),
        "dossier_bindings": [dict(binding) for binding in dossier_bindings],
        "gate_receipt_key_sha256": _require_hex(
            gate_binding.get("gate_receipt_key_sha256"), 64, "V8D_READINESS_AUDIT_GATE_KEY_INVALID"
        ),
        "gate_receipt_bytes_sha256": _require_hex(
            gate_binding.get("gate_receipt_bytes_sha256"), 64, "V8D_READINESS_AUDIT_GATE_BYTES_HASH_INVALID"
        ),
        "authorization_identity_sha256": _require_hex(
            gate_binding.get("authorization_identity_sha256"), 64,
            "V8D_READINESS_AUDIT_AUTHORIZATION_HASH_INVALID",
        ),
    }
    body["receipt_self_hash"] = canonical_sha256(body)
    return body


def _validate_receipt(receipt: Mapping[str, Any], *, expected_verification_stage: str) -> dict[str, Any]:
    if set(receipt) != set(RECEIPT_FIELDS):
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_SCHEMA_INVALID")
    verification_stage, logical_stage, _ = _receipt_fields_for_stage(expected_verification_stage)
    if receipt["schema_version"] != SCHEMA_VERSION or receipt["study"] != STUDY:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_PROVENANCE_INVALID")
    if receipt["artifact_role"] != ARTIFACT_ROLE or receipt["verification_result"] != VERIFICATION_RESULT:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_RESULT_INVALID")
    if receipt["verification_stage"] != verification_stage or receipt["logical_stage"] != logical_stage:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_STAGE_MISMATCH")
    if receipt["frozen_design_commit"] != FROZEN_DESIGN_COMMIT:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_DESIGN_MISMATCH")
    _require_hex(receipt["reviewed_production_implementation_commit"], 40, "V8D_READINESS_AUDIT_RECEIPT_IMPLEMENTATION_INVALID")
    _require_hex(receipt["aggregate_artifact_self_hash"], 64, "V8D_READINESS_AUDIT_RECEIPT_AGGREGATE_HASH_INVALID")
    _require_hex(receipt["gate_receipt_key_sha256"], 64, "V8D_READINESS_AUDIT_RECEIPT_GATE_KEY_INVALID")
    _require_hex(receipt["gate_receipt_bytes_sha256"], 64, "V8D_READINESS_AUDIT_RECEIPT_GATE_BYTES_INVALID")
    _require_hex(receipt["authorization_identity_sha256"], 64, "V8D_READINESS_AUDIT_RECEIPT_AUTHORIZATION_INVALID")
    if receipt["receipt_self_hash"] != canonical_sha256({key: value for key, value in receipt.items() if key != "receipt_self_hash"}):
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_SELF_HASH_INVALID")
    dossier_bindings = receipt["dossier_bindings"]
    if not isinstance(dossier_bindings, list) or len(dossier_bindings) != 3:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_DOSSIER_BINDINGS_INVALID")
    filenames: list[str] = []
    for binding in dossier_bindings:
        if not isinstance(binding, dict) or set(binding) != set(DOSSIER_BINDING_FIELDS):
            raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_DOSSIER_BINDING_SCHEMA_INVALID")
        if (
            not isinstance(binding["filename"], str)
            or not binding["filename"]
            or Path(binding["filename"]).name != binding["filename"]
        ):
            raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_FILENAME_INVALID")
        filenames.append(binding["filename"])
        _require_hex(binding["audit_artifact_self_hash"], 64, "V8D_READINESS_AUDIT_RECEIPT_DOSSIER_HASH_INVALID")
    if len(set(filenames)) != 3:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_DOSSIER_DUPLICATE")
    if (
        not isinstance(receipt["aggregate_filename"], str)
        or not receipt["aggregate_filename"]
        or Path(receipt["aggregate_filename"]).name != receipt["aggregate_filename"]
    ):
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_FILENAME_INVALID")
    return dict(receipt)


def _persist_receipt(receipt: dict[str, Any], destination: Path) -> dict[str, Any]:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink():
            raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_PATH_INVALID")
        if destination.exists():
            existing = _validate_receipt(
                _read_json_file(destination, label="RECEIPT"),
                expected_verification_stage=receipt["verification_stage"],
            )
            if existing != receipt:
                raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_CONFLICT")
            return existing
        fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".staging-", dir=str(destination.parent))
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write((json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            if destination.exists() or destination.is_symlink():
                raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_CONFLICT")
            os.replace(str(temporary), str(destination))
        finally:
            if temporary.exists():
                temporary.unlink()
        return receipt
    except V8DReadinessAuditVerificationBlocked:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_RECEIPT_WRITE_FAILED") from error


def _write_verified_receipt(
    *,
    logical_stage: str,
    aggregate_path: str | Path,
    dossier_paths: Sequence[str | Path],
    audit_root: Path,
    receipt_root: Path,
    aggregate_verifier: Callable[..., Mapping[str, Any]],
    dossier_verifier: Callable[..., Mapping[str, Any]],
    gate_root: Path,
) -> dict[str, Any]:
    verification_stage, logical_stage, receipt_path = _stage_constants(logical_stage)
    audit_root = _canonical_root(audit_root)
    receipt_root = _canonical_root(receipt_root) if receipt_root.exists() else receipt_root
    aggregate_file, aggregate_filename = _safe_audit_file(
        aggregate_path, audit_root, reason="V8D_READINESS_AUDIT_AGGREGATE_PATH_INVALID"
    )
    if len(dossier_paths) != 3:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_DOSSIER_COUNT_INVALID")
    dossier_files: list[Path] = []
    dossier_filenames: list[str] = []
    for path in dossier_paths:
        file_path, filename = _safe_audit_file(
            path, audit_root, reason="V8D_READINESS_AUDIT_DOSSIER_PATH_INVALID"
        )
        dossier_files.append(file_path)
        dossier_filenames.append(filename)
    if len(set(dossier_filenames)) != 3:
        raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_DOSSIER_DUPLICATE")
    try:
        aggregate = dict(aggregate_verifier(
            aggregate_file, dossier_files, gate_receipt_state_root=gate_root, expected_stage=logical_stage
        ))
        dossiers = [dict(dossier_verifier(
            path, gate_receipt_state_root=gate_root, expected_stage=logical_stage
        )) for path in dossier_files]
    except (V8DAuditVerificationBlocked, V8DReadinessAuditVerificationBlocked) as error:
        if isinstance(error, V8DReadinessAuditVerificationBlocked):
            raise
        raise V8DReadinessAuditVerificationBlocked(error.reason) from error
    _require_readiness_pass(aggregate, logical_stage)
    dossier_bindings, gate_binding = _require_verified_dossier_bindings(dossiers, aggregate, logical_stage=logical_stage)
    for binding, filename in zip(dossier_bindings, dossier_filenames):
        binding["filename"] = filename
    receipt = _build_receipt(
        verification_stage=verification_stage,
        logical_stage=logical_stage,
        aggregate_filename=aggregate_filename,
        aggregate=aggregate,
        dossier_bindings=dossier_bindings,
        gate_binding=gate_binding,
    )
    destination = receipt_root / receipt_path.name
    return _persist_receipt(receipt, destination)


def _write_t1c_production(aggregate_path: str | Path, dossier_paths: Sequence[str | Path]) -> dict[str, Any]:
    return _write_verified_receipt(
        logical_stage=T1C_LOGICAL_STAGE, aggregate_path=aggregate_path, dossier_paths=dossier_paths,
        audit_root=CANONICAL_PRODUCTION_AUDIT_ROOT,
        receipt_root=CANONICAL_READINESS_AUDIT_VERIFICATION_ROOT,
        aggregate_verifier=verify_aggregate_production, dossier_verifier=verify_dossier_production,
        gate_root=CANONICAL_CONSUMPTION_STATE_ROOT,
    )


def _write_t2_production(aggregate_path: str | Path, dossier_paths: Sequence[str | Path]) -> dict[str, Any]:
    return _write_verified_receipt(
        logical_stage=T2_LOGICAL_STAGE, aggregate_path=aggregate_path, dossier_paths=dossier_paths,
        audit_root=CANONICAL_PRODUCTION_AUDIT_ROOT,
        receipt_root=CANONICAL_READINESS_AUDIT_VERIFICATION_ROOT,
        aggregate_verifier=verify_aggregate_production, dossier_verifier=verify_dossier_production,
        gate_root=CANONICAL_CONSUMPTION_STATE_ROOT,
    )


def record_t1c_readiness_audit_verification(aggregate_path: str | Path, dossier_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Record a verified T1C readiness PASS using canonical production state."""
    return _write_t1c_production(aggregate_path, dossier_paths)


def record_t2_readiness_audit_verification(aggregate_path: str | Path, dossier_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Record a verified T2 readiness PASS using canonical production state."""
    return _write_t2_production(aggregate_path, dossier_paths)


def _read_canonical_pass(verification_stage: str) -> dict[str, Any]:
    expected_stage, logical_stage, receipt_path = _receipt_fields_for_stage(verification_stage)
    try:
        receipt = _validate_receipt(
            _read_json_file(receipt_path, label="RECEIPT"), expected_verification_stage=expected_stage
        )
        aggregate_path, _ = _safe_audit_file(
            CANONICAL_PRODUCTION_AUDIT_ROOT / receipt["aggregate_filename"],
            _canonical_root(CANONICAL_PRODUCTION_AUDIT_ROOT),
            reason="V8D_READINESS_AUDIT_AGGREGATE_PATH_INVALID",
        )
        dossier_paths: list[Path] = []
        for binding in receipt["dossier_bindings"]:
            dossier_path, _ = _safe_audit_file(
                CANONICAL_PRODUCTION_AUDIT_ROOT / binding["filename"],
                _canonical_root(CANONICAL_PRODUCTION_AUDIT_ROOT),
                reason="V8D_READINESS_AUDIT_DOSSIER_PATH_INVALID",
            )
            dossier_paths.append(dossier_path)
        aggregate = dict(verify_aggregate_production(
            aggregate_path, dossier_paths, gate_receipt_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
            expected_stage=logical_stage,
        ))
        dossiers = [dict(verify_dossier_production(
            path, gate_receipt_state_root=CANONICAL_CONSUMPTION_STATE_ROOT, expected_stage=logical_stage
        )) for path in dossier_paths]
        _require_readiness_pass(aggregate, logical_stage)
        actual_bindings, gate_binding = _require_verified_dossier_bindings(dossiers, aggregate, logical_stage=logical_stage)
        for actual, declared in zip(actual_bindings, receipt["dossier_bindings"]):
            if actual["audit_artifact_self_hash"] != declared["audit_artifact_self_hash"]:
                raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_DOSSIER_BINDING_MISMATCH")
        if receipt["aggregate_artifact_self_hash"] != aggregate.get("aggregate_self_hash"):
            raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_AGGREGATE_BINDING_MISMATCH")
        if receipt["reviewed_production_implementation_commit"] != aggregate.get("reviewed_production_implementation_commit"):
            raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_IMPLEMENTATION_BINDING_MISMATCH")
        for key in ("gate_receipt_key_sha256", "gate_receipt_bytes_sha256", "authorization_identity_sha256"):
            if receipt[key] != gate_binding[key]:
                raise V8DReadinessAuditVerificationBlocked("V8D_READINESS_AUDIT_GATE_BINDING_MISMATCH")
        return {
            "schema_version": receipt["schema_version"],
            "study": receipt["study"],
            "artifact_role": receipt["artifact_role"],
            "verification_stage": receipt["verification_stage"],
            "logical_stage": receipt["logical_stage"],
            "verification_result": receipt["verification_result"],
            "frozen_design_commit": receipt["frozen_design_commit"],
            "reviewed_production_implementation_commit": receipt["reviewed_production_implementation_commit"],
            "aggregate_filename": receipt["aggregate_filename"],
            "aggregate_artifact_self_hash": receipt["aggregate_artifact_self_hash"],
            "dossier_bindings": receipt["dossier_bindings"],
            "gate_receipt_key_sha256": receipt["gate_receipt_key_sha256"],
            "gate_receipt_bytes_sha256": receipt["gate_receipt_bytes_sha256"],
            "authorization_identity_sha256": receipt["authorization_identity_sha256"],
            "receipt_self_hash": receipt["receipt_self_hash"],
        }
    except V8DReadinessAuditVerificationBlocked:
        raise
    except (V8DAuditVerificationBlocked, OSError, TypeError, ValueError) as error:
        raise V8DReadinessAuditVerificationBlocked(
            getattr(error, "reason", "V8D_READINESS_AUDIT_VERIFICATION_BLOCKED")
        ) from error


def require_t1c_readiness_audit_verification_pass() -> dict[str, Any]:
    """Re-verify and return the canonical T1C readiness PASS metadata."""
    return _read_canonical_pass(T1C_VERIFICATION_STAGE)


def require_t2_readiness_audit_verification_pass() -> dict[str, Any]:
    """Re-verify and return the canonical T2 readiness PASS metadata."""
    return _read_canonical_pass(T2_VERIFICATION_STAGE)


def _write_synthetic_receipt_for_tests(
    *, logical_stage: str, aggregate_path: str | Path, dossier_paths: Sequence[str | Path],
    audit_root: str | Path, receipt_root: str | Path,
    aggregate_verifier: Callable[..., Mapping[str, Any]],
    dossier_verifier: Callable[..., Mapping[str, Any]], gate_root: str | Path,
) -> dict[str, Any]:
    """TEST-ONLY.  This helper cannot consume a gate or establish authority."""
    return _write_verified_receipt(
        logical_stage=logical_stage, aggregate_path=aggregate_path, dossier_paths=dossier_paths,
        audit_root=Path(audit_root), receipt_root=Path(receipt_root),
        aggregate_verifier=aggregate_verifier, dossier_verifier=dossier_verifier,
        gate_root=Path(gate_root),
    )


__all__ = [
    "CANONICAL_READINESS_AUDIT_VERIFICATION_ROOT",
    "T1C_RECEIPT_PATH",
    "T2_RECEIPT_PATH",
    "record_t1c_readiness_audit_verification",
    "record_t2_readiness_audit_verification",
    "require_t1c_readiness_audit_verification_pass",
    "require_t2_readiness_audit_verification_pass",
]
