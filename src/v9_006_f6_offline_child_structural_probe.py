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

from src.v9_005_stage_a_jpx_probe import (
    SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
    V9005StageABlocked,
    _is_canonical_raw_lock_timestamp,
    source_object_slot_id,
    validate_jpx_url,
)


# Reused verbatim from src/v9_005_stage_a_jpx_probe.py: this is the exact
# source_family value the real F6 production raw acquisition passes when
# locking both ROOT and CHILD. A locally redefined literal previously
# diverged from this canonical value (V9_006_F6_STRUCTURAL_PROBE_IMPL_
# MEDIUM_4_SOURCE_FAMILY_BINDING_LITERAL_MISMATCH).
SOURCE_FAMILY = SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
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

# Closed-set enums for every allowed structural-evidence field. No field may
# ever carry an arbitrary string: an allowed top-level key must never become
# a channel for payload-derived text/dates/years/values/URLs/paths/names.
_SAFE_EVIDENCE_ALLOWED_KEYS = frozenset({"status", "container_format", "open_parse_status", "sheet_table_count", "structural_dimensions", "candidate_header_column_count", "candidate_date_column_count", "candidate_value_column_count"})
_CONTAINER_FORMATS = frozenset({"OLE_COMPOUND_FILE", "ZIP_CONTAINER", "UNKNOWN_CONTAINER"})
_OPEN_PARSE_STATUSES = frozenset({"PARSER_NOT_IMPLEMENTED", "OPEN_PARSE_OK", "OPEN_PARSE_UNSUPPORTED", "OPEN_PARSE_AMBIGUOUS"})
_NONNEGATIVE_COUNT_KEYS = frozenset({"sheet_table_count", "candidate_header_column_count", "candidate_date_column_count", "candidate_value_column_count"})
_DIMENSION_ALLOWED_KEYS = frozenset({"ordinal", "row_count", "column_count", "visibility", "object_type"})
_DIMENSION_VISIBILITY_VALUES = frozenset({"VISIBLE", "HIDDEN", "VERY_HIDDEN", "UNKNOWN"})
_DIMENSION_OBJECT_TYPE_VALUES = frozenset({"WORKSHEET", "TABLE", "UNKNOWN"})


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
    """Carries exactly which information boundary was reached at failure time.

    ``raw_bytes_read_for_integrity`` is ``False`` only before any CHILD byte
    read is attempted, ``True`` once the exact CHILD bytes were successfully
    read (regardless of what fails afterward), and ``"unknown"`` only when a
    byte-read attempt itself failed and whether bytes were exposed cannot be
    proven -- it must never be fabricated ``False`` in that case.
    """

    def __init__(self, outcome: str, *, raw_bytes_read_for_integrity: bool | str = False, child_content_inspected: bool = False):
        super().__init__(outcome)
        self.outcome = outcome
        self.raw_bytes_read_for_integrity = raw_bytes_read_for_integrity
        self.child_content_inspected = child_content_inspected


def normalize_full_path(path: str | Path) -> str:
    """The frozen UTF-8 full-path representation for the runtime binding."""
    return str(Path(path).resolve())


def output_root_id_sha256(path: str | Path) -> str:
    return sha256(normalize_full_path(path).encode("utf-8")).hexdigest()


def _blocked(outcome: str, *, raw_bytes_read_for_integrity: bool | str = False, child_content_inspected: bool = False) -> None:
    raise ProbeBlocked(outcome, raw_bytes_read_for_integrity=raw_bytes_read_for_integrity, child_content_inspected=child_content_inspected)


def _receipt_is_exact(value: object) -> bool:
    return isinstance(value, dict) and set(value) == {"schema_version", "task", "confirmation_contract", "gate_consumed", "consumption_timestamp_utc"} and value.get("schema_version") == RECEIPT_SCHEMA and value.get("task") == RECEIPT_TASK and value.get("confirmation_contract") == RECEIPT_CONTRACT and value.get("gate_consumed") is True and isinstance(value.get("consumption_timestamp_utc"), str)


def _metadata_is_schema_valid(value: object, meta_path: Path, bindings: ProbeBindings) -> bool:
    """Canonical V9 raw-lock provenance validation, metadata-only (no .bin
    read). Reuses the existing repository raw-lock URL/timestamp/key
    semantics from src/v9_005_stage_a_jpx_probe.py rather than inventing
    divergent rules."""
    if not isinstance(value, dict) or set(value) != RAW_FIELDS:
        return False
    if (
        value.get("schema_version") != "V9_005_STAGE_A_RAW_LOCK_V1"
        or value.get("source_family") != bindings.source_family
        or value.get("applicable_period") != bindings.applicable_period
        or isinstance(value.get("http_status"), bool)
        or not isinstance(value.get("http_status"), int)
        or not 100 <= value["http_status"] <= 599
        or isinstance(value.get("byte_length"), bool)
        or not isinstance(value.get("byte_length"), int)
        or not isinstance(value.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
        or not _is_canonical_raw_lock_timestamp(value.get("retrieval_timestamp_utc"))
    ):
        return False
    requested_url = value.get("requested_url")
    resolved_url = value.get("resolved_url")
    try:
        validate_jpx_url(requested_url)
        validate_jpx_url(resolved_url)
    except V9005StageABlocked:
        return False
    expected_key = source_object_slot_id(bindings.source_family, bindings.applicable_period, requested_url)
    return meta_path.stem == expected_key


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
            if not _metadata_is_schema_valid(meta, meta_path, bindings):
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
        # A failed read attempt does not prove bytes were never exposed
        # (e.g. a partial OS-level read before the failure); never fabricate
        # False here.
        _blocked("IMPLEMENTATION_FAILURE", raw_bytes_read_for_integrity="unknown", child_content_inspected=False)
    digest = sha256(raw).hexdigest()
    if len(raw) != bindings.child_byte_length or digest != bindings.child_sha256 or meta.get("byte_length") != len(raw) or meta.get("sha256") != digest:
        # The exact CHILD bytes were successfully read before this mismatch
        # was detected.
        _blocked("IMPLEMENTATION_FAILURE", raw_bytes_read_for_integrity=True, child_content_inspected=False)
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


def _is_nonneg_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_allowed_enum_str(value: object, allowed: frozenset[str]) -> bool:
    # A closed-set membership check must be total for arbitrary Python
    # objects: `x in a_frozenset` raises TypeError for an unhashable value
    # (e.g. a list or dict), which would otherwise escape as an ordinary
    # exception rather than a fail-closed ProbeBlocked. Guarding with
    # isinstance(value, str) first makes every enum check total.
    return isinstance(value, str) and value in allowed


def _is_valid_dimension_item(item: object) -> bool:
    if not isinstance(item, dict) or set(item) - _DIMENSION_ALLOWED_KEYS:
        return False
    if "ordinal" in item and not (isinstance(item["ordinal"], int) and not isinstance(item["ordinal"], bool) and item["ordinal"] >= 1):
        return False
    if "row_count" in item and not _is_nonneg_int(item["row_count"]):
        return False
    if "column_count" in item and not _is_nonneg_int(item["column_count"]):
        return False
    if "visibility" in item and not _is_allowed_enum_str(item["visibility"], _DIMENSION_VISIBILITY_VALUES):
        return False
    if "object_type" in item and not _is_allowed_enum_str(item["object_type"], _DIMENSION_OBJECT_TYPE_VALUES):
        return False
    return True


def _is_valid_structural_dimensions(value: object) -> bool:
    if not isinstance(value, list):
        return False
    seen_ordinals: set[object] = set()
    for item in value:
        if not _is_valid_dimension_item(item):
            return False
        if "ordinal" in item:
            if item["ordinal"] in seen_ordinals:
                return False
            seen_ordinals.add(item["ordinal"])
    return True


def _safe_structural_evidence(value: object) -> dict[str, Any]:
    # Only ever called after the structural inspection boundary (section 4)
    # has been reached, so a rejection here always proves the CHILD bytes
    # were read and structural inspection was invoked.
    #
    # Every allowed field is validated against a closed enum/type/range set
    # -- never a free-form string -- so an allowed top-level key can never
    # become a channel for arbitrary payload-derived text/dates/years/
    # values/URLs/paths/names, however deeply nested.
    if (
        not isinstance(value, dict)
        or not _is_allowed_enum_str(value.get("status"), OUTCOMES)
        or set(value) - _SAFE_EVIDENCE_ALLOWED_KEYS
        or ("container_format" in value and not _is_allowed_enum_str(value["container_format"], _CONTAINER_FORMATS))
        or ("open_parse_status" in value and not _is_allowed_enum_str(value["open_parse_status"], _OPEN_PARSE_STATUSES))
        or any(key in value and not _is_nonneg_int(value[key]) for key in _NONNEGATIVE_COUNT_KEYS)
        or ("structural_dimensions" in value and not _is_valid_structural_dimensions(value["structural_dimensions"]))
    ):
        _blocked("IMPLEMENTATION_FAILURE", raw_bytes_read_for_integrity=True, child_content_inspected=True)
    return {key: value[key] for key in value if key in _SAFE_EVIDENCE_ALLOWED_KEYS}


def run_offline_child_structural_probe(*, production_state_parent: str | Path, output_root: str | Path, bindings: ProbeBindings = FROZEN_BINDINGS, structural_inspector: Callable[[bytes], dict[str, Any]] = _default_structural_inspector) -> dict[str, Any]:
    """Run A -> B -> C; no filesystem mutation and no network path exists."""
    _meta_path, meta, raw_path = locate_metadata_only(production_state_parent=production_state_parent, output_root=output_root, bindings=bindings)
    raw = content_blind_integrity_read(raw_path, meta, bindings=bindings)
    # Structural inspection boundary reached: the exact CHILD bytes are
    # verified and about to be handed to the structural inspector. Any
    # failure from this point on -- inspector exception, ProbeBlocked from an
    # injected inspector, safe-evidence rejection, or any other unexpected
    # exception while validating safe-evidence -- proves both that raw bytes
    # were read and that structural inspection was reached, so it must never
    # be allowed to escape as an ordinary exception and reach the CLI's
    # unproven-phase fallback (which would wrongly report unknown/false).
    try:
        raw_structural_result = structural_inspector(raw)
        evidence = _safe_structural_evidence(raw_structural_result)
    except ProbeBlocked as exc:
        _blocked(exc.outcome, raw_bytes_read_for_integrity=True, child_content_inspected=True)
    except Exception:
        _blocked("IMPLEMENTATION_FAILURE", raw_bytes_read_for_integrity=True, child_content_inspected=True)
    return {"execution_result": "COMPLETE", "status": evidence["status"], "network_request_count": 0, "raw_bytes_read_for_integrity": True, "child_content_inspected": True, "coverage_evaluated": False, "structural_evidence": evidence}
