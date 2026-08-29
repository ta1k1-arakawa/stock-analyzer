from __future__ import annotations

from hashlib import sha256
import json

import pytest

from src import v9_006_f6_required_period_coverage_successor_parser as successor


class Sheet:
    def __init__(self, years4: list[int], years6: list[int]):
        self.nrows = max(len(years4), len(years6))
        self.types = [[0] * 6 for _ in range(self.nrows)]
        self.values = [[None] * 6 for _ in range(self.nrows)]
        self.calls: list[tuple[int, int]] = []
        for row, year in enumerate(years4):
            self.types[row][3] = successor.xlrd.XL_CELL_DATE
            self.values[row][3] = year
        for row, year in enumerate(years6):
            self.types[row][5] = successor.xlrd.XL_CELL_DATE
            self.values[row][5] = year

    def cell_type(self, row: int, column: int) -> int:
        return self.types[row][column]

    def cell_value(self, row: int, column: int) -> int:
        self.calls.append((row, column))
        return self.values[row][column]


class Book:
    datemode = 0

    def __init__(self, sheet: Sheet):
        self.sheet = sheet

    def sheet_by_index(self, index: int) -> Sheet:
        assert index == 0
        return self.sheet


def structural(c4: int, c6: int) -> dict[str, object]:
    profiles = []
    for ordinal, count in ((4, c4), (6, c6)):
        profiles.append({
            "sheet_ordinal": 1, "column_ordinal": ordinal,
            "cell_type_counts": {"EMPTY": 0, "BLANK": 0, "TEXT": 0,
                                 "NUMBER": 0, "DATE": count, "BOOLEAN": 0,
                                 "ERROR": 0},
        })
    return {
        "status": "STRUCTURAL_FORMAT_CAPTURED",
        "container_format": "OLE_COMPOUND_FILE",
        "open_parse_status": "OPEN_PARSE_OK", "sheet_table_count": 1,
        "structural_dimensions": [{"ordinal": 1, "row_count": 1,
                                      "column_count": 6, "visibility": "VISIBLE",
                                      "object_type": "WORKSHEET"}],
        "cell_type_profiles": profiles,
    }


def digest(evidence: object) -> str:
    return sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def candidate(evidence: dict[str, object], h4: list[dict[str, int]], h6: list[dict[str, int]]) -> dict[str, object]:
    covered = sorted(set(successor.REQUIRED_YEARS) & {x["year"] for x in h4} & {x["year"] for x in h6})
    missing = sorted(set(successor.REQUIRED_YEARS) - set(covered))
    out4 = [x for x in h4 if x["year"] not in successor.REQUIRED_YEARS]
    out6 = [x for x in h6 if x["year"] not in successor.REQUIRED_YEARS]
    return {
        "status": successor.CAPTURED if not missing else successor.PARTIAL,
        "structural_profile_sha256": digest(evidence),
        "structural_profile_hash_verified": True,
        "date_column_ordinals": [4, 6], "raw_bytes_read_for_integrity": True,
        "child_content_inspected": True, "date_year_value_read": True,
        "coverage_evaluated": True, "coverage_result_accepted": True,
        "network_request_count": 0, "year_histograms": {"4": h4, "6": h6},
        "covered_required_years": covered, "missing_required_years": missing,
        "all_required_years_covered": not missing,
        "out_of_scope_histogram_col4": out4,
        "out_of_scope_histogram_col6": out6,
        "out_of_scope_disagreement": out4 != out6,
    }


def run_synthetic(monkeypatch: pytest.MonkeyPatch, years4: list[int], years6: list[int]):
    evidence = structural(len(years4), len(years6))
    monkeypatch.setattr(successor, "EXPECTED_STRUCTURAL_PROFILE_SHA256", digest(evidence))
    monkeypatch.setattr(successor, "_structural_gate", lambda *_: (evidence, digest(evidence)))
    sheet = Sheet(years4, years6)
    opened: dict[str, object] = {}
    monkeypatch.setattr(successor.xlrd, "open_workbook", lambda **kwargs: (opened.update(kwargs) or Book(sheet)))
    monkeypatch.setattr(successor.xlrd, "xldate_as_tuple", lambda serial, datemode: (serial, 1, 1, 0, 0, 0))
    return successor._run_verified_bytes(b"synthetic"), sheet, opened


def test_constants_are_frozen():
    assert successor.DATE_COLUMN_ORDINALS == (4, 6)
    assert successor.REQUIRED_YEARS == tuple(range(2017, 2026))


def test_all_required_years_capture_and_exact_open(monkeypatch):
    years = list(successor.REQUIRED_YEARS)
    result, sheet, opened = run_synthetic(monkeypatch, years, years)
    assert result["status"] == successor.CAPTURED
    assert result["covered_required_years"] == years
    assert result["missing_required_years"] == []
    assert result["all_required_years_covered"] is True
    assert sheet.calls == [(row, 3) for row in range(9)] + [(row, 5) for row in range(9)]
    assert opened == {"file_contents": b"synthetic", "formatting_info": True, "on_demand": False, "ragged_rows": False}


def test_one_column_required_year_missing_is_partial_not_ambiguous_or_terminal(monkeypatch):
    years = list(successor.REQUIRED_YEARS)
    result, _, _ = run_synthetic(monkeypatch, years, years[:-1])
    assert result["status"] == successor.PARTIAL
    assert result["missing_required_years"] == [2025]
    assert result["coverage_result_accepted"] is True


@pytest.mark.parametrize("left,right", [([1999] + list(successor.REQUIRED_YEARS), list(successor.REQUIRED_YEARS)), ([1999, 1999] + list(successor.REQUIRED_YEARS), [1999] + list(successor.REQUIRED_YEARS))])
def test_outside_scope_differences_are_nonfatal_diagnostics(monkeypatch, left, right):
    result, _, _ = run_synthetic(monkeypatch, left, right)
    assert result["status"] == successor.CAPTURED
    assert result["out_of_scope_disagreement"] is True


def test_identical_outside_scope_histograms_have_false_diagnostic(monkeypatch):
    years = [1999] + list(successor.REQUIRED_YEARS)
    result, _, _ = run_synthetic(monkeypatch, years, years)
    assert result["status"] == successor.CAPTURED
    assert result["out_of_scope_disagreement"] is False


def test_full_history_inequality_and_one_column_only_year_do_not_restore_equality_gate(monkeypatch):
    years = list(successor.REQUIRED_YEARS)
    result, _, _ = run_synthetic(monkeypatch, [2001] + years, years)
    assert result["status"] == successor.CAPTURED
    result, _, _ = run_synthetic(monkeypatch, [2017], [])
    assert result["status"] == successor.PARTIAL
    assert 2017 in result["missing_required_years"]


def test_validator_rejects_forged_status_lists_and_diagnostic(monkeypatch):
    evidence = structural(1, 1)
    monkeypatch.setattr(successor, "EXPECTED_STRUCTURAL_PROFILE_SHA256", digest(evidence))
    value = candidate(evidence, [{"year": 2017, "count": 1}], [{"year": 2017, "count": 1}])
    assert successor.safe_successor_coverage_evidence(value, evidence) == value
    for key, bad in (("status", successor.CAPTURED), ("covered_required_years", []), ("missing_required_years", []), ("out_of_scope_disagreement", True)):
        forged = json.loads(json.dumps(value))
        forged[key] = bad
        with pytest.raises(successor.CoverageBlocked):
            successor.safe_successor_coverage_evidence(forged, evidence)


def test_validator_rejects_count_mismatch_types_and_unhashable_payload(monkeypatch):
    evidence = structural(1, 1)
    monkeypatch.setattr(successor, "EXPECTED_STRUCTURAL_PROFILE_SHA256", digest(evidence))
    value = candidate(evidence, [{"year": 2017, "count": 1}], [{"year": 2017, "count": 1}])
    for key, bad in (("raw_bytes_read_for_integrity", 1), ("network_request_count", False), ("date_column_ordinals", [4.0, 6.0])):
        forged = dict(value); forged[key] = bad
        with pytest.raises(successor.CoverageBlocked): successor.safe_successor_coverage_evidence(forged, evidence)
    forged = json.loads(json.dumps(value)); forged["year_histograms"]["4"][0]["count"] = 2
    with pytest.raises(successor.CoverageBlocked): successor.safe_successor_coverage_evidence(forged, evidence)
    with pytest.raises(successor.CoverageBlocked): successor.safe_successor_coverage_evidence({"status": []}, {})


def test_validator_rejects_forged_failure_provenance_and_boolean_year_lookalike(monkeypatch):
    evidence = structural(1, 1)
    monkeypatch.setattr(successor, "EXPECTED_STRUCTURAL_PROFILE_SHA256", digest(evidence))
    failure = {"status": successor.FAILURE, "structural_profile_sha256": digest(evidence), "structural_profile_hash_verified": False, "date_column_ordinals": [4, 6], "raw_bytes_read_for_integrity": True, "child_content_inspected": True, "date_year_value_read": True, "coverage_evaluated": True, "coverage_result_accepted": False, "network_request_count": 0}
    with pytest.raises(successor.CoverageBlocked): successor.safe_successor_coverage_evidence(failure, evidence)
    value = candidate(evidence, [{"year": 2017, "count": 1}], [{"year": 2017, "count": 1}])
    value["covered_required_years"] = [True]
    with pytest.raises(successor.CoverageBlocked): successor.safe_successor_coverage_evidence(value, evidence)


def test_hash_mismatch_prevents_date_reads(monkeypatch):
    evidence = structural(1, 1)
    monkeypatch.setattr(successor, "_structural_gate", lambda *_: (evidence, digest(evidence)))
    monkeypatch.setattr(successor.xlrd, "open_workbook", lambda **_: pytest.fail("must not open"))
    with pytest.raises(successor.CoverageBlocked) as exc:
        successor._run_verified_bytes(b"x", inspector=lambda _: evidence)
    assert exc.value.evidence["structural_profile_sha256"] == digest(evidence)
    assert exc.value.evidence["date_year_value_read"] is False


def test_non_date_values_are_never_read(monkeypatch):
    result, sheet, _ = run_synthetic(monkeypatch, [2017], [2017])
    assert result["date_year_value_read"] is True
    assert all(column in (3, 5) for _, column in sheet.calls)


@pytest.mark.parametrize("outcome,raw,inspected", [("CHATGPT_DECISION_REQUIRED", False, False), ("IMPLEMENTATION_FAILURE", True, False)])
def test_inherited_phase_outcome_and_provenance_are_preserved(monkeypatch, outcome, raw, inspected):
    def blocked(**_):
        raise successor.ProbeBlocked(outcome, raw_bytes_read_for_integrity=raw, child_content_inspected=inspected)
    monkeypatch.setattr(successor, "locate_metadata_only", blocked)
    with pytest.raises(successor.CoverageBlocked) as exc:
        successor.run_required_period_coverage_successor_parser(production_state_parent="x", output_root="y")
    assert exc.value.evidence["status"] == outcome
    assert exc.value.evidence["raw_bytes_read_for_integrity"] == raw
    assert exc.value.evidence["child_content_inspected"] is inspected


def test_late_failure_preserves_reached_phase_booleans(monkeypatch):
    years = list(successor.REQUIRED_YEARS)
    evidence = structural(len(years), len(years))
    monkeypatch.setattr(successor, "EXPECTED_STRUCTURAL_PROFILE_SHA256", digest(evidence))
    monkeypatch.setattr(successor, "_structural_gate", lambda *_: (evidence, digest(evidence)))
    monkeypatch.setattr(successor.xlrd, "open_workbook", lambda **_: Book(Sheet(years, years)))
    monkeypatch.setattr(successor.xlrd, "xldate_as_tuple", lambda serial, _: (serial, 1, 1, 0, 0, 0))
    monkeypatch.setattr(successor, "safe_successor_coverage_evidence", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(successor.CoverageBlocked) as exc:
        successor._run_verified_bytes(b"x", inspector=lambda _: evidence)
    result = exc.value.evidence
    assert result["structural_profile_hash_verified"] is True
    assert result["date_year_value_read"] is True and result["coverage_evaluated"] is True
    assert result["coverage_result_accepted"] is False
