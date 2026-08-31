"""Offline, fail-closed structural evidence for the reviewed F1 TERMINAL lock."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable

from src import v9_006_stage_a_schema_discovery as schema

TASK = "V9_006_F1_TERMINAL_MONTH_STRUCTURE_EVIDENCE_DIAGNOSTIC"
ACQUISITION_DESIGN_GIT_SHA = "0ee4b338110c626fb92267343586fa6936699805"
ACQUISITION_IMPLEMENTATION_GIT_SHA = "4efd6ab8ca951a9bbc67bc0146ecb86a20533a0e"
DIAGNOSTIC_DESIGN_GIT_SHA = "870f4f7a6e26a661c60f33e9d4aa0c4f4af6611c"
TERMINAL_PAYLOAD_SHA256 = "3119fb5c0854544b0f17b2abda1db836201fac60027695ff95d10bea103df187"
TERMINAL_BYTE_LENGTH = 851456
RAW_LOCK_SET_SHA256 = "f7d641052f3cb1e1ab33936303e2e504bc480ff9d89cde85ccade5d214f193cf"
STATE_ROOT_BASENAME = "v9-006-f1-successor-public-acquisition-state"

RESULTS = frozenset({"EVIDENCE_CAPTURED", "INPUT_BINDING_FAILURE", "FORMAT_OR_STRUCTURE_UNSUPPORTED", "SAFE_OUTPUT_VALIDATION_FAILURE", "IMPLEMENTATION_FAILURE"})
STAGES = frozenset({"NONE", "PRE_READ_BINDING", "TERMINAL_LOCK_READ", "STRUCTURE_PROFILE", "SAFE_PROJECTION", "IMPLEMENTATION"})
_FAILURE_STAGES = {
    "INPUT_BINDING_FAILURE": frozenset({"PRE_READ_BINDING", "TERMINAL_LOCK_READ"}),
    "FORMAT_OR_STRUCTURE_UNSUPPORTED": frozenset({"STRUCTURE_PROFILE"}),
    "SAFE_OUTPUT_VALIDATION_FAILURE": frozenset({"SAFE_PROJECTION"}),
    "IMPLEMENTATION_FAILURE": frozenset({"IMPLEMENTATION"}),
}
SAFE_KEYS = frozenset({
    "task", "acquisition_design_git_sha", "acquisition_implementation_git_sha",
    "diagnostic_design_git_sha", "diagnostic_implementation_git_sha",
    "terminal_payload_sha256", "terminal_byte_length", "raw_lock_set_sha256",
    "diagnostic_result", "failure_stage", "container_format", "sheet_count",
    "sheets", "text_neighborhood", "neighborhood_truncated", "structural_evidence_sha256",
    "network_request_count", "safe_provenance_verified",
})

_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DAY = r"(?:0?[1-9]|[12][0-9]|3[01])"
_YEAR = r"[0-9]{4}"
_NUMERIC = rf"{_YEAR}[-/.](?:0?[1-9]|1[0-2])(?:[-/.]{_DAY})?"
_JAPANESE = rf"{_YEAR}年(?:0?[1-9]|1[0-2])月(?:{_DAY}日)?"
_ENGLISH = rf"(?:{_MONTH}[ ]+{_YEAR}|{_MONTH}[ ]+{_DAY}[,]?[ ]+{_YEAR})"
_DATE = rf"(?:{_ENGLISH}|{_NUMERIC}|{_JAPANESE})[.]?"
_DATE_RE = re.compile(rf"(?:{_DATE}|As[ ]+of[ ]+{_DATE}|List[ ]+of[ ]+TSE-listed[ ]+Issues[ ]\([ ]*{_DATE}[ ]*\))\Z", re.IGNORECASE)


class DiagnosticContractError(ValueError):
    pass


class UnsupportedStructure(Exception):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hex(value: Any, length: int) -> bool:
    return type(value) is str and len(value) == length and bool(re.fullmatch(r"[0-9a-f]+", value))


def is_allowlisted_date_text(value: Any) -> bool:
    if type(value) is not str:
        return False
    normalized = value.strip()[:schema.MAX_TEXT_CODEPOINTS]
    return bool(normalized and _DATE_RE.fullmatch(normalized))


def _allowlisted_text(value: Any) -> str | None:
    if not is_allowlisted_date_text(value):
        return None
    return value.strip()[:schema.MAX_TEXT_CODEPOINTS]


def _empty_structure() -> dict[str, Any]:
    return {"container_format": None, "sheet_count": None, "sheets": [], "text_neighborhood": [], "neighborhood_truncated": False}


def _base(implementation_sha: str, result: str, stage: str) -> dict[str, Any]:
    return {
        "task": TASK,
        "acquisition_design_git_sha": ACQUISITION_DESIGN_GIT_SHA,
        "acquisition_implementation_git_sha": ACQUISITION_IMPLEMENTATION_GIT_SHA,
        "diagnostic_design_git_sha": DIAGNOSTIC_DESIGN_GIT_SHA,
        "diagnostic_implementation_git_sha": implementation_sha,
        "terminal_payload_sha256": TERMINAL_PAYLOAD_SHA256,
        "terminal_byte_length": TERMINAL_BYTE_LENGTH,
        "raw_lock_set_sha256": RAW_LOCK_SET_SHA256,
        "diagnostic_result": result,
        "failure_stage": stage,
        **_empty_structure(),
        "structural_evidence_sha256": None,
        "network_request_count": 0,
        "safe_provenance_verified": False,
    }


def structural_evidence_sha256(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "structural_evidence_sha256"}
    return sha256(canonical_json(body).encode("utf-8")).hexdigest()


def _validate_sheet(sheet: Any) -> None:
    expected = {"sheet_ordinal", "visibility", "row_count", "column_count", "column_cell_type_counts", "sheet_name_date_text", "sheet_name_was_redacted"}
    if type(sheet) is not dict or set(sheet) != expected:
        raise DiagnosticContractError("sheet")
    if type(sheet["sheet_ordinal"]) is not int or sheet["sheet_ordinal"] < 1 or sheet["visibility"] not in {"VISIBLE", "HIDDEN", "VERY_HIDDEN"}:
        raise DiagnosticContractError("sheet metadata")
    if type(sheet["row_count"]) is not int or sheet["row_count"] < 0 or type(sheet["column_count"]) is not int or sheet["column_count"] < 0:
        raise DiagnosticContractError("dimensions")
    counts = sheet["column_cell_type_counts"]
    if type(counts) is not list or len(counts) != sheet["column_count"]:
        raise DiagnosticContractError("counts")
    for item in counts:
        if type(item) is not dict or set(item) != set(schema._CELL_TYPES) or any(type(n) is not int or n < 0 for n in item.values()) or sum(item.values()) != sheet["row_count"]:
            raise DiagnosticContractError("taxonomy")
    if sheet["sheet_name_date_text"] is not None and _allowlisted_text(sheet["sheet_name_date_text"]) != sheet["sheet_name_date_text"]:
        raise DiagnosticContractError("sheet text")
    if type(sheet["sheet_name_was_redacted"]) is not bool or sheet["sheet_name_was_redacted"] != (sheet["sheet_name_date_text"] is None):
        raise DiagnosticContractError("sheet redaction")


def _validate_neighborhood(rows: Any, sheets: list[dict[str, Any]], sheet_count: int) -> None:
    if type(rows) is not list:
        raise DiagnosticContractError("neighborhood")
    previous = None
    rows_by_sheet: dict[int, int] = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"sheet_ordinal", "row_ordinal", "cells"}:
            raise DiagnosticContractError("row")
        key = (row["sheet_ordinal"], row["row_ordinal"])
        if type(row["sheet_ordinal"]) is not int or type(row["row_ordinal"]) is not int or row["sheet_ordinal"] < 1 or row["row_ordinal"] < 1 or (previous is not None and key <= previous):
            raise DiagnosticContractError("row order")
        if row["sheet_ordinal"] > sheet_count:
            raise DiagnosticContractError("row sheet range")
        referenced_sheet = sheets[row["sheet_ordinal"] - 1]
        if referenced_sheet["sheet_ordinal"] != row["sheet_ordinal"] or referenced_sheet["visibility"] != "VISIBLE":
            raise DiagnosticContractError("row sheet visibility")
        if row["row_ordinal"] > referenced_sheet["row_count"]:
            raise DiagnosticContractError("row range")
        rows_by_sheet[row["sheet_ordinal"]] = rows_by_sheet.get(row["sheet_ordinal"], 0) + 1
        if rows_by_sheet[row["sheet_ordinal"]] > schema.MAX_SAMPLE_ROWS_PER_TABLE:
            raise DiagnosticContractError("row sample bound")
        previous = key
        if type(row["cells"]) is not list or len(row["cells"]) > schema.MAX_SAMPLE_CELLS_PER_ROW:
            raise DiagnosticContractError("cells")
        columns = []
        for cell in row["cells"]:
            if type(cell) is not dict or set(cell) not in ({"column_ordinal", "cell_type"}, {"column_ordinal", "cell_type", "text"}):
                raise DiagnosticContractError("cell")
            if type(cell["column_ordinal"]) is not int or cell["column_ordinal"] < 1 or cell["column_ordinal"] > referenced_sheet["column_count"] or cell["column_ordinal"] in columns or cell["cell_type"] not in schema._CELL_TYPES:
                raise DiagnosticContractError("cell metadata")
            columns.append(cell["column_ordinal"])
            if cell["cell_type"] == "TEXT":
                if set(cell) == {"column_ordinal", "cell_type", "text"} and _allowlisted_text(cell["text"]) != cell["text"]:
                    raise DiagnosticContractError("text allowlist")
            elif "text" in cell:
                raise DiagnosticContractError("nontext")
        if columns != sorted(columns):
            raise DiagnosticContractError("cell order")


def validate_safe_result(value: Any) -> None:
    if type(value) is not dict or set(value) != SAFE_KEYS:
        raise DiagnosticContractError("keys")
    fixed = {
        "acquisition_design_git_sha": ACQUISITION_DESIGN_GIT_SHA,
        "acquisition_implementation_git_sha": ACQUISITION_IMPLEMENTATION_GIT_SHA,
        "diagnostic_design_git_sha": DIAGNOSTIC_DESIGN_GIT_SHA,
        "terminal_payload_sha256": TERMINAL_PAYLOAD_SHA256,
        "terminal_byte_length": TERMINAL_BYTE_LENGTH,
        "raw_lock_set_sha256": RAW_LOCK_SET_SHA256,
    }
    if value["task"] != TASK or any(not _hex(value[key], 40) for key in ("acquisition_design_git_sha", "acquisition_implementation_git_sha", "diagnostic_design_git_sha", "diagnostic_implementation_git_sha")) or any(value[key] != expected for key, expected in fixed.items()):
        raise DiagnosticContractError("identity")
    if type(value["terminal_byte_length"]) is not int or type(value["diagnostic_result"]) is not str or value["diagnostic_result"] not in RESULTS or type(value["failure_stage"]) is not str or value["failure_stage"] not in STAGES or type(value["network_request_count"]) is not int or value["network_request_count"] != 0 or type(value["safe_provenance_verified"]) is not bool:
        raise DiagnosticContractError("status")
    if (value["diagnostic_result"], value["failure_stage"]) == ("EVIDENCE_CAPTURED", "NONE"):
        if not value["safe_provenance_verified"] or value["container_format"] != schema.FORMAT_OLE_BIFF or type(value["sheet_count"]) is not int or value["sheet_count"] < 0 or type(value["sheets"]) is not list or len(value["sheets"]) != value["sheet_count"] or type(value["neighborhood_truncated"]) is not bool or type(value["structural_evidence_sha256"]) is not str or not _hex(value["structural_evidence_sha256"], 64):
            raise DiagnosticContractError("success structure")
        for ordinal, sheet in enumerate(value["sheets"], 1):
            _validate_sheet(sheet)
            if sheet["sheet_ordinal"] != ordinal:
                raise DiagnosticContractError("sheet order")
        _validate_neighborhood(value["text_neighborhood"], value["sheets"], value["sheet_count"])
        if value["structural_evidence_sha256"] != structural_evidence_sha256(value):
            raise DiagnosticContractError("digest")
    else:
        if value["failure_stage"] not in _FAILURE_STAGES.get(value["diagnostic_result"], set()) or value["container_format"] is not None or value["sheet_count"] is not None or value["sheets"] != [] or value["text_neighborhood"] != [] or value["neighborhood_truncated"] is not False or value["structural_evidence_sha256"] is not None or value["safe_provenance_verified"]:
            raise DiagnosticContractError("failure structure")


def finalize_safe_result(value: dict[str, Any]) -> dict[str, Any]:
    if value["diagnostic_result"] == "EVIDENCE_CAPTURED":
        value["safe_provenance_verified"] = True
        value["structural_evidence_sha256"] = structural_evidence_sha256(value)
    validate_safe_result(value)
    return value


def _project_profile(profile: dict[str, Any]) -> dict[str, Any]:
    evidence = profile["structural_evidence"]
    sheets = []
    for sheet in evidence["sheets"]:
        date_name = _allowlisted_text(sheet.get("sheet_name"))
        sheets.append({"sheet_ordinal": sheet["sheet_ordinal"], "visibility": sheet["visibility"], "row_count": sheet["row_count"], "column_count": sheet["column_count"], "column_cell_type_counts": sheet["column_cell_type_counts"], "sheet_name_date_text": date_name, "sheet_name_was_redacted": date_name is None})
    rows = []
    for row in sorted(evidence.get("schema_neighborhood", []), key=lambda item: (item["sheet_ordinal"], item["row_ordinal"])):
        cells = []
        for cell in sorted(row["cells"], key=lambda item: item["column_ordinal"]):
            item = {"column_ordinal": cell["column_ordinal"], "cell_type": cell["cell_type"]}
            if cell["cell_type"] == "TEXT":
                date_text = _allowlisted_text(cell.get("text"))
                if date_text is not None:
                    item["text"] = date_text
            cells.append(item)
        rows.append({"sheet_ordinal": row["sheet_ordinal"], "row_ordinal": row["row_ordinal"], "cells": cells})
    return {"container_format": evidence["format"], "sheet_count": evidence["sheet_count"], "sheets": sheets, "text_neighborhood": rows, "neighborhood_truncated": bool(evidence.get("SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE", False))}


def profile_terminal_bytes(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes:
        raise DiagnosticContractError("raw bytes")
    if schema.detect_container_format(raw) != schema.FORMAT_OLE_BIFF:
        raise UnsupportedStructure("container")
    structure, evidence = schema._ole_structure(raw)
    return _project_profile({"structural_profile_sha256": sha256(canonical_json(structure).encode("utf-8")).hexdigest(), "structural_evidence": evidence})


def run_terminal_structure_diagnostic(raw: bytes, diagnostic_implementation_git_sha: str, *, profiler: Callable[[bytes], dict[str, Any]] = profile_terminal_bytes) -> dict[str, Any]:
    if not _hex(diagnostic_implementation_git_sha, 40):
        raise DiagnosticContractError("diagnostic implementation")
    if type(raw) is not bytes or sha256(raw).hexdigest() != TERMINAL_PAYLOAD_SHA256 or len(raw) != TERMINAL_BYTE_LENGTH:
        value = _base(diagnostic_implementation_git_sha, "INPUT_BINDING_FAILURE", "TERMINAL_LOCK_READ")
        validate_safe_result(value)
        return value
    try:
        projection = profiler(raw)
    except UnsupportedStructure:
        value = _base(diagnostic_implementation_git_sha, "FORMAT_OR_STRUCTURE_UNSUPPORTED", "STRUCTURE_PROFILE")
        validate_safe_result(value)
        return value
    except DiagnosticContractError:
        raise
    except Exception as exc:
        raise RuntimeError("diagnostic profiler failure") from exc
    value = _base(diagnostic_implementation_git_sha, "EVIDENCE_CAPTURED", "NONE")
    value.update(projection)
    return finalize_safe_result(value)


run_diagnostic = run_terminal_structure_diagnostic
validate_safe_acquisition_result = validate_safe_result
