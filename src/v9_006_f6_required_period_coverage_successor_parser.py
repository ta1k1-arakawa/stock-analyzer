"""Offline successor F6 required-period coverage parser.

This module deliberately leaves the historical full-history-equality parser
unchanged.  It shares only the reviewed locked-child Phase A/B and structural
identity gate, then applies the successor's required-year intersection rule.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Callable

import xlrd

from src.v9_006_f6_offline_child_structural_probe import (
    FROZEN_BINDINGS,
    ProbeBlocked,
    _default_structural_inspector,
    _safe_structural_evidence,
    content_blind_integrity_read,
    locate_metadata_only,
)

EXPECTED_STRUCTURAL_PROFILE_SHA256 = "4332d0b27a1e35256abef4c0e240b2c576c20122a264374ea0c5da3729beacce"
DATE_COLUMN_ORDINALS = (4, 6)
REQUIRED_YEARS = tuple(range(2017, 2026))
CAPTURED = "SUCCESSOR_REQUIRED_PERIOD_COVERAGE_CAPTURED"
PARTIAL = "SUCCESSOR_REQUIRED_PERIOD_COVERAGE_PARTIAL"
FAILURE = "IMPLEMENTATION_FAILURE"
_SUCCESS = frozenset({CAPTURED, PARTIAL})
_STATUSES = _SUCCESS | {FAILURE}
_COMMON = frozenset({
    "status", "structural_profile_sha256", "structural_profile_hash_verified",
    "date_column_ordinals", "raw_bytes_read_for_integrity",
    "child_content_inspected", "date_year_value_read", "coverage_evaluated",
    "coverage_result_accepted", "network_request_count",
})
_DERIVED = frozenset({
    "covered_required_years", "missing_required_years", "all_required_years_covered",
})
_DIAGNOSTIC = frozenset({
    "out_of_scope_histogram_col4", "out_of_scope_histogram_col6",
    "out_of_scope_disagreement",
})


class CoverageBlocked(Exception):
    def __init__(self, evidence: dict[str, Any]):
        super().__init__(evidence.get("status", FAILURE))
        self.evidence = evidence


def _is_int(value: object, *, positive: bool = False) -> bool:
    return type(value) is int and (value >= 1 if positive else True)


def _failure(*, digest: str | None, verified: bool, raw: bool | str,
             inspected: bool, date_read: bool, evaluated: bool = False,
             histograms: dict[str, Any] | None = None,
             status: str = FAILURE) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "structural_profile_sha256": digest,
        "structural_profile_hash_verified": verified,
        "date_column_ordinals": [4, 6],
        "raw_bytes_read_for_integrity": raw,
        "child_content_inspected": inspected,
        "date_year_value_read": date_read,
        "coverage_evaluated": evaluated,
        "coverage_result_accepted": False,
        "network_request_count": 0,
    }
    if histograms is not None:
        result["year_histograms"] = histograms
    return result


def _histogram_valid(value: object) -> bool:
    if type(value) is not list:
        return False
    prior: int | None = None
    for entry in value:
        if type(entry) is not dict or set(entry) != {"year", "count"}:
            return False
        year, count = entry.get("year"), entry.get("count")
        if not _is_int(year) or not _is_int(count, positive=True):
            return False
        if prior is not None and year <= prior:
            return False
        prior = year
    return True


def _date_count(structural: object, ordinal: int) -> int | None:
    if type(structural) is not dict or structural.get("status") != "STRUCTURAL_FORMAT_CAPTURED":
        return None
    profiles = structural.get("cell_type_profiles")
    if type(profiles) is not list:
        return None
    matches = [item for item in profiles if type(item) is dict and item.get("sheet_ordinal") == 1 and item.get("column_ordinal") == ordinal]
    if len(matches) != 1 or type(matches[0].get("cell_type_counts")) is not dict:
        return None
    count = matches[0]["cell_type_counts"].get("DATE")
    return count if _is_int(count) else None


def _histograms_valid(value: object, structural: object) -> bool:
    return (
        type(value) is dict and set(value) == {"4", "6"}
        and _histogram_valid(value.get("4")) and _histogram_valid(value.get("6"))
        and sum(item["count"] for item in value["4"]) == _date_count(structural, 4)
        and sum(item["count"] for item in value["6"]) == _date_count(structural, 6)
    )


def _trusted_failure(provenance: object, structural: object) -> dict[str, Any]:
    """Whitelist actual phase facts; never copy a rejected candidate."""
    if type(provenance) is not dict:
        return _failure(digest=None, verified=False, raw="unknown", inspected=False, date_read=False)
    digest = provenance.get("structural_profile_sha256")
    verified = provenance.get("structural_profile_hash_verified")
    raw = provenance.get("raw_bytes_read_for_integrity")
    inspected = provenance.get("child_content_inspected")
    date_read = provenance.get("date_year_value_read")
    evaluated = provenance.get("coverage_evaluated")
    if ((digest is not None and (type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None))
            or type(verified) is not bool
            or (type(raw) is not bool and raw != "unknown")
            or type(inspected) is not bool or type(date_read) is not bool
            or type(evaluated) is not bool):
        return _failure(digest=None, verified=False, raw="unknown", inspected=False, date_read=False)
    if verified != (digest == EXPECTED_STRUCTURAL_PROFILE_SHA256):
        return _failure(digest=None, verified=False, raw=raw, inspected=inspected, date_read=False)
    histograms = provenance.get("year_histograms")
    safe_histograms = histograms if evaluated and _histograms_valid(histograms, structural) else None
    safe_evaluated = evaluated and safe_histograms is not None
    return _failure(
        digest=digest, verified=verified, raw=raw, inspected=inspected,
        date_read=date_read, evaluated=safe_evaluated, histograms=safe_histograms,
    )


def _required_lists(histograms: dict[str, list[dict[str, int]]]) -> tuple[list[int], list[int]]:
    years4 = {entry["year"] for entry in histograms["4"]}
    years6 = {entry["year"] for entry in histograms["6"]}
    covered = sorted(set(REQUIRED_YEARS) & years4 & years6)
    return covered, sorted(set(REQUIRED_YEARS) - set(covered))


def _filtered(histogram: list[dict[str, int]]) -> list[dict[str, int]]:
    return [entry for entry in histogram if entry["year"] not in REQUIRED_YEARS]


def _exact_year_list(value: object, expected: list[int]) -> bool:
    return type(value) is list and all(_is_int(year) for year in value) and value == expected


def safe_successor_coverage_evidence(value: object, structural_evidence: object,
                                     *, failure_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate closed successor evidence; invalid input becomes safe failure."""
    try:
        if type(value) is not dict or not _COMMON <= set(value):
            raise ValueError
        status = value.get("status")
        if status not in _STATUSES:
            raise ValueError
        if type(value.get("date_column_ordinals")) is not list or value["date_column_ordinals"] != [4, 6] or any(not _is_int(item) for item in value["date_column_ordinals"]):
            raise ValueError
        if type(value.get("network_request_count")) is not int or value["network_request_count"] != 0:
            raise ValueError
        if (type(value.get("raw_bytes_read_for_integrity")) is not bool and value.get("raw_bytes_read_for_integrity") != "unknown") or any(type(value.get(key)) is not bool for key in ("child_content_inspected", "date_year_value_read", "coverage_evaluated", "coverage_result_accepted", "structural_profile_hash_verified")):
            raise ValueError
        digest = value.get("structural_profile_sha256")
        if digest is not None and (type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError
        verified = value["structural_profile_hash_verified"]
        if verified != (digest == EXPECTED_STRUCTURAL_PROFILE_SHA256):
            raise ValueError
        if status == FAILURE:
            if set(value) - (_COMMON | {"year_histograms"}) or value["coverage_result_accepted"]:
                raise ValueError
            if not verified and (value["date_year_value_read"] or value["coverage_evaluated"]):
                raise ValueError
            if "year_histograms" in value:
                if not (verified and value["date_year_value_read"] and value["coverage_evaluated"] and _histograms_valid(value["year_histograms"], structural_evidence)):
                    raise ValueError
            return dict(value)
        if set(value) != (_COMMON | _DERIVED | _DIAGNOSTIC | {"year_histograms"}):
            raise ValueError
        if not (verified and value["date_year_value_read"] and value["coverage_evaluated"] and value["coverage_result_accepted"]):
            raise ValueError
        histograms = value["year_histograms"]
        if not _histograms_valid(histograms, structural_evidence):
            raise ValueError
        expected4, expected6 = _filtered(histograms["4"]), _filtered(histograms["6"])
        if type(value["out_of_scope_histogram_col4"]) is not list or type(value["out_of_scope_histogram_col6"]) is not list or value["out_of_scope_histogram_col4"] != expected4 or value["out_of_scope_histogram_col6"] != expected6 or type(value["out_of_scope_disagreement"]) is not bool or value["out_of_scope_disagreement"] != (expected4 != expected6):
            raise ValueError
        covered, missing = _required_lists(histograms)
        if not _exact_year_list(value["covered_required_years"], covered) or not _exact_year_list(value["missing_required_years"], missing) or type(value["all_required_years_covered"]) is not bool or value["all_required_years_covered"] != (missing == []):
            raise ValueError
        if (status == CAPTURED) != (missing == []):
            raise ValueError
        return dict(value)
    except CoverageBlocked:
        raise
    except Exception:
        raise CoverageBlocked(_trusted_failure(failure_provenance, structural_evidence)) from None


def _structural_gate(raw: bytes, inspector: Callable[[bytes], dict[str, Any]]) -> tuple[dict[str, Any], str]:
    evidence = _safe_structural_evidence(inspector(raw))
    digest = sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return evidence, digest


def _run_verified_bytes(raw: bytes, *, inspector: Callable[[bytes], dict[str, Any]] = _default_structural_inspector) -> dict[str, Any]:
    date_read = False
    evaluated = False
    histograms: dict[str, list[dict[str, int]]] | None = None
    try:
        structural, digest = _structural_gate(raw, inspector)
    except Exception:
        raise CoverageBlocked(_failure(digest=None, verified=False, raw=True, inspected=True, date_read=False)) from None
    if digest != EXPECTED_STRUCTURAL_PROFILE_SHA256:
        raise CoverageBlocked(_failure(digest=digest, verified=False, raw=True, inspected=True, date_read=False))
    try:
        book = xlrd.open_workbook(file_contents=raw, formatting_info=True, on_demand=False, ragged_rows=False)
        sheet = book.sheet_by_index(0)
        histograms = {}
        for ordinal in DATE_COLUMN_ORDINALS:
            years: Counter[int] = Counter()
            for rowx in range(sheet.nrows):
                if sheet.cell_type(rowx, ordinal - 1) == xlrd.XL_CELL_DATE:
                    date_read = True
                    year = xlrd.xldate_as_tuple(sheet.cell_value(rowx, ordinal - 1), book.datemode)[0]
                    years[year] += 1
            if _date_count(structural, ordinal) is None or sum(years.values()) != _date_count(structural, ordinal):
                raise CoverageBlocked(_failure(digest=digest, verified=True, raw=True, inspected=True, date_read=date_read))
            histograms[str(ordinal)] = [{"year": year, "count": years[year]} for year in sorted(years)]
        covered, missing = _required_lists(histograms)
        evaluated = True
        out4, out6 = _filtered(histograms["4"]), _filtered(histograms["6"])
        candidate: dict[str, Any] = {
            "status": CAPTURED if not missing else PARTIAL,
            "structural_profile_sha256": digest,
            "structural_profile_hash_verified": True,
            "date_column_ordinals": [4, 6],
            "raw_bytes_read_for_integrity": True,
            "child_content_inspected": True,
            "date_year_value_read": date_read,
            "coverage_evaluated": True,
            "coverage_result_accepted": True,
            "network_request_count": 0,
            "year_histograms": histograms,
            "covered_required_years": covered,
            "missing_required_years": missing,
            "all_required_years_covered": not missing,
            "out_of_scope_histogram_col4": out4,
            "out_of_scope_histogram_col6": out6,
            "out_of_scope_disagreement": out4 != out6,
        }
        return safe_successor_coverage_evidence(candidate, structural, failure_provenance=candidate)
    except CoverageBlocked:
        raise
    except Exception:
        raise CoverageBlocked(_failure(digest=digest, verified=True, raw=True, inspected=True, date_read=date_read, evaluated=evaluated, histograms=histograms if evaluated else None)) from None


def run_required_period_coverage_successor_parser(*, production_state_parent: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Production entry point with frozen inherited binding and no overrides."""
    try:
        _meta_path, meta, raw_path = locate_metadata_only(production_state_parent=production_state_parent, output_root=output_root, bindings=FROZEN_BINDINGS)
        raw = content_blind_integrity_read(raw_path, meta, bindings=FROZEN_BINDINGS)
    except ProbeBlocked as exc:
        raise CoverageBlocked(_failure(digest=None, verified=False, raw=exc.raw_bytes_read_for_integrity, inspected=exc.child_content_inspected, date_read=False, status=exc.outcome)) from None
    return _run_verified_bytes(raw)
