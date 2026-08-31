from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from src import v9_006_f1_terminal_month_structure_evidence as diagnostic
from src import v9_006_stage_a_schema_discovery as schema


IMPL = "a" * 40


def _raw(monkeypatch):
    raw = b"synthetic-terminal"
    monkeypatch.setattr(diagnostic, "TERMINAL_PAYLOAD_SHA256", sha256(raw).hexdigest())
    monkeypatch.setattr(diagnostic, "TERMINAL_BYTE_LENGTH", len(raw))
    return raw


def _profile(*, arbitrary_sheet="Acme Security Holdings", arbitrary_text="1320"):
    counts = [{key: 0 for key in schema._CELL_TYPES}]
    counts[0]["TEXT"] = 2
    return {
        "container_format": schema.FORMAT_OLE_BIFF,
        "sheet_count": 1,
        "sheets": [{
            "sheet_ordinal": 1,
            "visibility": "VISIBLE",
            "row_count": 2,
            "column_count": 1,
            "column_cell_type_counts": counts,
            "sheet_name_date_text": None if arbitrary_sheet else "January 2026",
            "sheet_name_was_redacted": bool(arbitrary_sheet),
        }],
        "text_neighborhood": [{
            "sheet_ordinal": 1,
            "row_ordinal": 1,
            "cells": [{"column_ordinal": 1, "cell_type": "TEXT"}],
        }, {
            "sheet_ordinal": 1,
            "row_ordinal": 2,
            "cells": [{"column_ordinal": 1, "cell_type": "TEXT", "text": "January 2026"}],
        }],
        "neighborhood_truncated": False,
    }


def test_allowlist_and_sheet_redaction_are_applied_before_digest(monkeypatch):
    raw = _raw(monkeypatch)
    result = diagnostic.run_terminal_structure_diagnostic(raw, IMPL, profiler=lambda _raw: _profile())
    diagnostic.validate_safe_result(result)
    sheet = result["sheets"][0]
    assert sheet["sheet_name_date_text"] is None and sheet["sheet_name_was_redacted"] is True
    assert all("1320" not in json.dumps(item) for item in result["text_neighborhood"])
    assert "Acme Security Holdings" not in diagnostic.canonical_json(result)
    assert result["structural_evidence_sha256"] == diagnostic.structural_evidence_sha256(result)


def test_allowlisted_sheet_name_and_text_are_retained(monkeypatch):
    raw = _raw(monkeypatch)
    profile = _profile(arbitrary_sheet="")
    profile["sheets"][0]["sheet_name_date_text"] = "January 2026"
    profile["sheets"][0]["sheet_name_was_redacted"] = False
    result = diagnostic.run_terminal_structure_diagnostic(raw, IMPL, profiler=lambda _raw: profile)
    assert result["sheets"][0]["sheet_name_date_text"] == "January 2026"
    assert result["text_neighborhood"][1]["cells"][0]["text"] == "January 2026"


def _valid_result(monkeypatch):
    raw = _raw(monkeypatch)
    return diagnostic.run_terminal_structure_diagnostic(raw, IMPL, profiler=lambda _raw: _profile())


def _assert_invalid_neighborhood(result):
    result["structural_evidence_sha256"] = diagnostic.structural_evidence_sha256(result)
    with pytest.raises(diagnostic.DiagnosticContractError):
        diagnostic.validate_safe_result(result)


def test_neighborhood_rows_and_cells_bind_to_visible_sheet_geometry(monkeypatch):
    base = _valid_result(monkeypatch)

    for sheet_ordinal in (0, 2):
        candidate = copy.deepcopy(base)
        candidate["text_neighborhood"][0]["sheet_ordinal"] = sheet_ordinal
        _assert_invalid_neighborhood(candidate)

    candidate = copy.deepcopy(base)
    candidate["text_neighborhood"][0]["row_ordinal"] = 3
    _assert_invalid_neighborhood(candidate)

    candidate = copy.deepcopy(base)
    candidate["text_neighborhood"] = list(reversed(candidate["text_neighborhood"]))
    _assert_invalid_neighborhood(candidate)

    candidate = copy.deepcopy(base)
    candidate["text_neighborhood"].append(copy.deepcopy(candidate["text_neighborhood"][0]))
    _assert_invalid_neighborhood(candidate)

    candidate = copy.deepcopy(base)
    candidate["sheets"][0]["visibility"] = "HIDDEN"
    _assert_invalid_neighborhood(candidate)

    candidate = copy.deepcopy(base)
    candidate["text_neighborhood"][0]["cells"][0]["column_ordinal"] = 2
    _assert_invalid_neighborhood(candidate)

    candidate = copy.deepcopy(base)
    candidate["sheets"][0]["column_count"] = 2
    counts = candidate["sheets"][0]["column_cell_type_counts"][0]
    candidate["sheets"][0]["column_cell_type_counts"] = [copy.deepcopy(counts), copy.deepcopy(counts)]
    candidate["text_neighborhood"][0]["cells"] = [
        {"column_ordinal": 2, "cell_type": "TEXT"},
        {"column_ordinal": 1, "cell_type": "TEXT"},
    ]
    _assert_invalid_neighborhood(candidate)

    candidate = copy.deepcopy(base)
    candidate["sheets"][0]["column_count"] = 2
    counts = candidate["sheets"][0]["column_cell_type_counts"][0]
    candidate["sheets"][0]["column_cell_type_counts"] = [copy.deepcopy(counts), copy.deepcopy(counts)]
    candidate["text_neighborhood"][0]["cells"] = [
        {"column_ordinal": 1, "cell_type": "TEXT"},
        {"column_ordinal": 1, "cell_type": "TEXT"},
    ]
    _assert_invalid_neighborhood(candidate)


@pytest.mark.parametrize("text", ["Acme Security Holdings", "1320", "2026", "Price", "ticker ABC"])
def test_non_allowlisted_text_is_not_emittable(monkeypatch, text):
    raw = _raw(monkeypatch)
    profile = _profile(arbitrary_sheet="")
    profile["sheets"][0]["sheet_name_date_text"] = None
    profile["sheets"][0]["sheet_name_was_redacted"] = True
    profile["text_neighborhood"][1]["cells"][0] = {"column_ordinal": 1, "cell_type": "TEXT"}
    result = diagnostic.run_terminal_structure_diagnostic(raw, IMPL, profiler=lambda _raw: profile)
    assert text not in diagnostic.canonical_json(result)


@pytest.mark.parametrize("text", ["January 2026", "January 31, 2026", "2026-01", "2026/01/31", "2026年1月", "As of January 2026", "List of TSE-listed Issues (January 2026)"])
def test_frozen_date_allowlist(text):
    assert diagnostic.is_allowlisted_date_text(text)
for text in ("2026", "1320", "January", "Acme January 2026", "2026-13", "2026-01-99"):
    assert not diagnostic.is_allowlisted_date_text(text)


def test_invalid_input_binding_does_not_call_profiler(monkeypatch):
    raw = _raw(monkeypatch)
    calls = []
    result = diagnostic.run_terminal_structure_diagnostic(raw + b"x", IMPL, profiler=lambda value: calls.append(value))
    assert result["diagnostic_result"] == "INPUT_BINDING_FAILURE" and result["failure_stage"] == "TERMINAL_LOCK_READ"
    assert calls == []


def test_unexpected_profiler_exception_propagates_without_safe_evidence(monkeypatch):
    raw = _raw(monkeypatch)
    with pytest.raises(RuntimeError):
        diagnostic.run_terminal_structure_diagnostic(raw, IMPL, profiler=lambda _raw: (_ for _ in ()).throw(RuntimeError("private")))


def test_unsupported_format_maps_to_closed_failure(monkeypatch):
    raw = _raw(monkeypatch)
    monkeypatch.setattr(schema, "detect_container_format", lambda _raw: schema.FORMAT_UNKNOWN)
    result = diagnostic.run_terminal_structure_diagnostic(raw, IMPL)
    diagnostic.validate_safe_result(result)
    assert (result["diagnostic_result"], result["failure_stage"]) == ("FORMAT_OR_STRUCTURE_UNSUPPORTED", "STRUCTURE_PROFILE")


def test_every_failure_row_and_provenance_mismatch_are_rejected_or_accepted():
    for result_name, stage in (("INPUT_BINDING_FAILURE", "PRE_READ_BINDING"), ("INPUT_BINDING_FAILURE", "TERMINAL_LOCK_READ"), ("FORMAT_OR_STRUCTURE_UNSUPPORTED", "STRUCTURE_PROFILE"), ("SAFE_OUTPUT_VALIDATION_FAILURE", "SAFE_PROJECTION"), ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION")):
        value = diagnostic._base(IMPL, result_name, stage)
        diagnostic.validate_safe_result(value)
        for key in ("acquisition_design_git_sha", "acquisition_implementation_git_sha", "diagnostic_design_git_sha", "diagnostic_implementation_git_sha"):
            altered = dict(value); altered[key] = "b" * 40
            if key == "diagnostic_implementation_git_sha":
                diagnostic.validate_safe_result(altered)
            else:
                with pytest.raises(ValueError): diagnostic.validate_safe_result(altered)


def test_deterministic_repeated_synthetic_result(monkeypatch):
    raw = _raw(monkeypatch)
    first = diagnostic.run_terminal_structure_diagnostic(raw, IMPL, profiler=lambda _raw: _profile())
    second = diagnostic.run_terminal_structure_diagnostic(raw, IMPL, profiler=lambda _raw: _profile())
    assert diagnostic.canonical_json(first) == diagnostic.canonical_json(second)


def test_no_filesystem_or_network_seam_is_used(monkeypatch, tmp_path: Path):
    raw = _raw(monkeypatch)
    result = diagnostic.run_terminal_structure_diagnostic(raw, IMPL, profiler=lambda _raw: _profile())
    assert result["network_request_count"] == 0 and list(tmp_path.iterdir()) == []
