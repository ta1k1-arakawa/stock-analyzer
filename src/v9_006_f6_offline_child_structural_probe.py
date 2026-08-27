"""Synthetic-testable, read-only F6 production CHILD structural probe.

This module has no network capability and never creates or mutates durable
state.  Its public production binding is frozen below; tests may inject a
synthetic binding and structural inspector without weakening that entrypoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable


SOURCE_FAMILY = "SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE"
APPLICABLE_PERIOD = "TOPIX_GLOBAL_2017_2025"
EXPECTED_OUTPUT_ROOT_ID_SHA256 = "5705fa3dae30c17a57208a1a03edbb5f4fac8a0986603ba39d21229262abbeee"
EXPECTED_CHILD_SHA256 = "060d74a7f5a3b413d351de05ed07f412d093a3ebf41f6ea3d4e0de3f313b4b0c"
EXPECTED_CHILD_BYTE_LENGTH = 36352
EXPECTED_ONE_SHOT_RECEIPT_COUNT = 1
EXPECTED_PRODUCTION_ROOT_COUNT = 1
RECEIPT_FILENAME = "V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT.json"
RECEIPT_SCHEMA = "V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_V1"
RECEIPT_TASK = "V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION"
RECEIPT_CONTRACT = "V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_ONE_SHOT"
RAW_FIELDS = frozenset({"schema_version", "source_family", "applicable_period", "requested_url", "resolved_url", "http_status", "retrieval_timestamp_utc", "byte_length", "sha256"})
OUTCOMES = frozenset({"STRUCTURAL_FORMAT_CAPTURED", "STRUCTURAL_FORMAT_UNSUPPORTED", "STRUCTURAL_FORMAT_AMBIGUOUS", "CHATGPT_DECISION_REQUIRED", "IMPLEMENTATION_FAILURE"})


@dataclass(frozen=True)
class ProbeBindings:
    output_root_id_sha256: str = EXPECTED_OUTPUT_ROOT_ID_SHA256
    child_sha256: str = EXPECTED_CHILD_SHA256
    child_byte_length: int = EXPECTED_CHILD_BYTE_LENGTH
    one_shot_receipt_count: int = EXPECTED_ONE_SHOT_RECEIPT_COUNT
    production_root_count: int = EXPECTED_PRODUCTION_ROOT_COUNT
    source_family: str = SOURCE_FAMILY
    applicable_period: str = APPLICABLE_PERIOD


FROZEN_BINDINGS = ProbeBindings()


class ProbeBlocked(Exception):
    def __init__(self, outcome: str):
        super().__init__(outcome)
        self.outcome = outcome


def normalize_full_path(path: str | Path) -> str:
    """The frozen UTF-8 full-path representation for the runtime binding."""
    return str(Path(path).resolve())


def output_root_id_sha256(path: str | Path) -> str:
    return sha256(normalize_full_path(path).encode("utf-8")).hexdigest()


def _blocked(outcome: str) -> None:
    raise ProbeBlocked(outcome)


def _receipt_is_exact(value: object) -> bool:
    return isinstance(value, dict) and set(value) == {"schema_version", "task", "confirmation_contract", "gate_consumed", "consumption_timestamp_utc"} and value.get("schema_version") == RECEIPT_SCHEMA and value.get("task") == RECEIPT_TASK and value.get("confirmation_contract") == RECEIPT_CONTRACT and value.get("gate_consumed") is True and isinstance(value.get("consumption_timestamp_utc"), str)


def _metadata_is_schema_valid(value: object, bindings: ProbeBindings) -> bool:
    if not isinstance(value, dict) or set(value) != RAW_FIELDS:
        return False
    return (
        value.get("schema_version") == "V9_005_STAGE_A_RAW_LOCK_V1"
        and value.get("source_family") == bindings.source_family
        and value.get("applicable_period") == bindings.applicable_period
        and isinstance(value.get("requested_url"), str) and bool(value["requested_url"])
        and isinstance(value.get("resolved_url"), str) and bool(value["resolved_url"])
        and isinstance(value.get("http_status"), int) and not isinstance(value["http_status"], bool)
        and isinstance(value.get("byte_length"), int) and not isinstance(value["byte_length"], bool)
        and isinstance(value.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is not None
        and isinstance(value.get("retrieval_timestamp_utc"), str)
    )


def locate_metadata_only(*, production_state_parent: str | Path, output_root: str | Path, bindings: ProbeBindings = FROZEN_BINDINGS) -> tuple[Path, dict[str, Any], Path]:
    """Phase A: reads only the exact bound root's receipt and JSON metadata."""
    parent = Path(production_state_parent).resolve()
    root = Path(output_root).resolve()
    try:
        root.relative_to(parent)
    except ValueError:
        _blocked("CHATGPT_DECISION_REQUIRED")
    if output_root_id_sha256(root) != bindings.output_root_id_sha256 or not root.is_dir() or bindings.production_root_count != 1:
        _blocked("CHATGPT_DECISION_REQUIRED")
    receipts = list(root.glob(RECEIPT_FILENAME))
    if len(receipts) != bindings.one_shot_receipt_count:
        _blocked("IMPLEMENTATION_FAILURE")
    try:
        receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    except Exception:
        _blocked("IMPLEMENTATION_FAILURE")
    if not _receipt_is_exact(receipt):
        _blocked("IMPLEMENTATION_FAILURE")
    raw_dir = root / "raw"
    if not raw_dir.is_dir():
        _blocked("CHATGPT_DECISION_REQUIRED")
    candidates: list[tuple[Path, dict[str, Any], Path]] = []
    for meta_path in raw_dir.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            _blocked("IMPLEMENTATION_FAILURE")
        if isinstance(meta, dict) and meta.get("source_family") == bindings.source_family and meta.get("applicable_period") == bindings.applicable_period:
            if not _metadata_is_schema_valid(meta, bindings):
                _blocked("IMPLEMENTATION_FAILURE")
            raw_path = meta_path.with_suffix(".bin")
            if not raw_path.is_file():
                _blocked("IMPLEMENTATION_FAILURE")
            candidates.append((meta_path, meta, raw_path))
    if len(candidates) != 1:
        _blocked("CHATGPT_DECISION_REQUIRED")
    return candidates[0]


def content_blind_integrity_read(raw_path: Path, meta: dict[str, Any], *, bindings: ProbeBindings = FROZEN_BINDINGS) -> bytes:
    """Phase B: opaque bytes only; no format/container inspection occurs here."""
    try:
        raw = raw_path.read_bytes()
    except Exception:
        _blocked("IMPLEMENTATION_FAILURE")
    digest = sha256(raw).hexdigest()
    if len(raw) != bindings.child_byte_length or digest != bindings.child_sha256 or meta.get("byte_length") != len(raw) or meta.get("sha256") != digest:
        _blocked("IMPLEMENTATION_FAILURE")
    return raw


def _default_structural_inspector(raw: bytes) -> dict[str, Any]:
    # No package/container parser is introduced by this task.  Magic-byte
    # classification is a safe format enum, never decompression or inspection.
    if raw.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        container = "OLE_COMPOUND_FILE"
    elif raw.startswith(b"PK\x03\x04"):
        container = "ZIP_CONTAINER"
    else:
        container = "UNKNOWN_CONTAINER"
    return {"status": "STRUCTURAL_FORMAT_UNSUPPORTED", "container_format": container, "open_parse_status": "PARSER_NOT_IMPLEMENTED", "sheet_table_count": 0, "structural_dimensions": []}


def _safe_structural_evidence(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("status") not in OUTCOMES:
        _blocked("IMPLEMENTATION_FAILURE")
    allowed = {"status", "container_format", "open_parse_status", "sheet_table_count", "structural_dimensions", "candidate_header_column_count", "candidate_date_column_count", "candidate_value_column_count"}
    if set(value) - allowed:
        _blocked("IMPLEMENTATION_FAILURE")
    safe = {key: value[key] for key in value if key in allowed}
    if not isinstance(safe.get("status"), str):
        _blocked("IMPLEMENTATION_FAILURE")
    return safe


def run_offline_child_structural_probe(*, production_state_parent: str | Path, output_root: str | Path, bindings: ProbeBindings = FROZEN_BINDINGS, structural_inspector: Callable[[bytes], dict[str, Any]] = _default_structural_inspector) -> dict[str, Any]:
    """Run A -> B -> C; no filesystem mutation and no network path exists."""
    _meta_path, meta, raw_path = locate_metadata_only(production_state_parent=production_state_parent, output_root=output_root, bindings=bindings)
    raw = content_blind_integrity_read(raw_path, meta, bindings=bindings)
    evidence = _safe_structural_evidence(structural_inspector(raw))
    return {"execution_result": "COMPLETE", "status": evidence["status"], "network_request_count": 0, "raw_bytes_read_for_integrity": True, "child_content_inspected": True, "coverage_evaluated": False, "structural_evidence": evidence}
