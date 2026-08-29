"""Offline, fail-closed F6 date/year coverage parser.

The public entry point deliberately has no acquisition, retry, or methodology
parameters.  It first reuses the reviewed Phase A/B and structural evidence
gate, then reads only DATE-typed values from the two frozen columns.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
from pathlib import Path
from typing import Any, Callable

import xlrd

from src.v9_006_f6_offline_child_structural_probe import (
    FROZEN_BINDINGS, ProbeBlocked, _default_structural_inspector,
    _safe_structural_evidence, content_blind_integrity_read,
    locate_metadata_only,
)

EXPECTED_STRUCTURAL_PROFILE_SHA256 = "4332d0b27a1e35256abef4c0e240b2c576c20122a264374ea0c5da3729beacce"
DATE_COLUMN_ORDINALS = (4, 6)
REQUIRED_YEARS = (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
_STATUSES = frozenset({"F6_YEAR_COVERAGE_CAPTURED", "F6_YEAR_COVERAGE_AMBIGUOUS", "IMPLEMENTATION_FAILURE"})
_COMMON = frozenset({"status", "structural_profile_sha256", "structural_profile_hash_verified", "date_column_ordinals", "raw_bytes_read_for_integrity", "child_content_inspected", "date_year_value_read", "coverage_evaluated", "coverage_result_accepted", "network_request_count"})
_DERIVED = frozenset({"covered_years", "covered_required_years", "missing_required_years", "all_required_years_covered"})


class CoverageBlocked(Exception):
    def __init__(self, evidence: dict[str, Any]):
        super().__init__(evidence.get("status", "IMPLEMENTATION_FAILURE"))
        self.evidence = evidence


def _is_int(value: object, *, positive: bool = False) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and (value >= 1 if positive else True)


def _failure(*, sha: str | None, verified: bool, raw: bool | str, inspected: bool, date_read: bool, evaluated: bool = False, histograms: dict[str, Any] | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"status": "IMPLEMENTATION_FAILURE", "structural_profile_sha256": sha, "structural_profile_hash_verified": verified, "date_column_ordinals": [4, 6], "raw_bytes_read_for_integrity": raw, "child_content_inspected": inspected, "date_year_value_read": date_read, "coverage_evaluated": evaluated, "coverage_result_accepted": False, "network_request_count": 0}
    if histograms is not None:
        value["year_histograms"] = histograms
    return value


def _histogram_is_valid(value: object) -> bool:
    if not isinstance(value, list):
        return False
    last: int | None = None
    for item in value:
        if not isinstance(item, dict) or set(item) != {"year", "count"}:
            return False
        year, count = item.get("year"), item.get("count")
        if not _is_int(year) or not _is_int(count, positive=True) or (last is not None and year <= last):
            return False
        last = year
    return True


def _date_count(structural: object, ordinal: int) -> int | None:
    if not isinstance(structural, dict) or structural.get("status") != "STRUCTURAL_FORMAT_CAPTURED":
        return None
    profiles = structural.get("cell_type_profiles")
    if not isinstance(profiles, list):
        return None
    matches = [item for item in profiles if isinstance(item, dict) and item.get("sheet_ordinal") == 1 and item.get("column_ordinal") == ordinal]
    if len(matches) != 1 or not isinstance(matches[0].get("cell_type_counts"), dict):
        return None
    count = matches[0]["cell_type_counts"].get("DATE")
    return count if _is_int(count) else None


def safe_coverage_evidence(value: object, structural_evidence: object) -> dict[str, Any]:
    """Closed-schema validator. Raises CoverageBlocked, never TypeError."""
    try:
        if not isinstance(value, dict) or set(value) - (_COMMON | _DERIVED | {"year_histograms"}):
            raise ValueError
        if set(_COMMON) - set(value) or value.get("status") not in _STATUSES or value.get("date_column_ordinals") != [4, 6] or value.get("network_request_count") != 0:
            raise ValueError
        if value.get("raw_bytes_read_for_integrity") not in (True, False, "unknown") or not isinstance(value.get("child_content_inspected"), bool) or not isinstance(value.get("date_year_value_read"), bool) or not isinstance(value.get("coverage_evaluated"), bool) or not isinstance(value.get("coverage_result_accepted"), bool) or not isinstance(value.get("structural_profile_hash_verified"), bool):
            raise ValueError
        digest = value.get("structural_profile_sha256")
        if digest is not None and (not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None):
            raise ValueError
        status = value["status"]
        if status == "IMPLEMENTATION_FAILURE":
            if value["coverage_result_accepted"] or _DERIVED & set(value):
                raise ValueError
            if value["structural_profile_hash_verified"]:
                if digest != EXPECTED_STRUCTURAL_PROFILE_SHA256:
                    raise ValueError
            elif value["date_year_value_read"] or value["coverage_evaluated"]:
                raise ValueError
            if "year_histograms" in value:
                histograms = value["year_histograms"]
                if (not value["structural_profile_hash_verified"] or not value["date_year_value_read"] or not value["coverage_evaluated"] or not isinstance(histograms, dict) or set(histograms) != {"4", "6"} or not _histogram_is_valid(histograms.get("4")) or not _histogram_is_valid(histograms.get("6")) or sum(item["count"] for item in histograms["4"]) != _date_count(structural_evidence, 4) or sum(item["count"] for item in histograms["6"]) != _date_count(structural_evidence, 6)):
                    raise ValueError
            elif value["coverage_evaluated"]:
                raise ValueError
            return dict(value)
        if not value["structural_profile_hash_verified"] or not value["date_year_value_read"] or not value["coverage_evaluated"] or digest != EXPECTED_STRUCTURAL_PROFILE_SHA256 or "year_histograms" not in value:
            raise ValueError
        histograms = value["year_histograms"]
        if not isinstance(histograms, dict) or set(histograms) != {"4", "6"} or not _histogram_is_valid(histograms.get("4")) or not _histogram_is_valid(histograms.get("6")):
            raise ValueError
        if sum(item["count"] for item in histograms["4"]) != _date_count(structural_evidence, 4) or sum(item["count"] for item in histograms["6"]) != _date_count(structural_evidence, 6):
            raise ValueError
        if status == "F6_YEAR_COVERAGE_AMBIGUOUS":
            if histograms["4"] == histograms["6"] or value["coverage_result_accepted"] or _DERIVED & set(value):
                raise ValueError
            return dict(value)
        if histograms["4"] != histograms["6"] or not value["coverage_result_accepted"] or not _DERIVED <= set(value):
            raise ValueError
        years = [item["year"] for item in histograms["4"]]
        required = [year for year in years if year in REQUIRED_YEARS]
        missing = [year for year in REQUIRED_YEARS if year not in years]
        if value.get("covered_years") != years or value.get("covered_required_years") != required or value.get("missing_required_years") != missing or value.get("all_required_years_covered") != (missing == []):
            raise ValueError
        return dict(value)
    except CoverageBlocked:
        raise
    except Exception:
        raise CoverageBlocked(_failure(sha=None, verified=False, raw=True, inspected=True, date_read=False)) from None


def _structural_gate(raw: bytes, inspector: Callable[[bytes], dict[str, Any]]) -> tuple[dict[str, Any], str]:
    evidence = _safe_structural_evidence(inspector(raw))
    digest = sha256(json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return evidence, digest


def _run_verified_bytes(raw: bytes, *, inspector: Callable[[bytes], dict[str, Any]] = _default_structural_inspector) -> dict[str, Any]:
    date_read = False
    try:
        structural, digest = _structural_gate(raw, inspector)
    except Exception:
        raise CoverageBlocked(_failure(sha=None, verified=False, raw=True, inspected=True, date_read=False)) from None
    if digest != EXPECTED_STRUCTURAL_PROFILE_SHA256:
        raise CoverageBlocked(_failure(sha=digest, verified=False, raw=True, inspected=True, date_read=False))
    try:
        book = xlrd.open_workbook(file_contents=raw, formatting_info=True, on_demand=False, ragged_rows=False)
        sheet = book.sheet_by_index(0)
        histograms: dict[str, list[dict[str, int]]] = {}
        for ordinal in DATE_COLUMN_ORDINALS:
            years: Counter[int] = Counter()
            for rowx in range(sheet.nrows):
                if sheet.cell_type(rowx, ordinal - 1) == xlrd.XL_CELL_DATE:
                    date_read = True
                    year = xlrd.xldate_as_tuple(sheet.cell_value(rowx, ordinal - 1), book.datemode)[0]
                    years[year] += 1
            expected = _date_count(structural, ordinal)
            if expected is None or sum(years.values()) != expected:
                raise CoverageBlocked(_failure(sha=digest, verified=True, raw=True, inspected=True, date_read=date_read))
            histograms[str(ordinal)] = [{"year": year, "count": years[year]} for year in sorted(years)]
        equal = histograms["4"] == histograms["6"]
        base: dict[str, Any] = {"status": "F6_YEAR_COVERAGE_CAPTURED" if equal else "F6_YEAR_COVERAGE_AMBIGUOUS", "structural_profile_sha256": digest, "structural_profile_hash_verified": True, "date_column_ordinals": [4, 6], "raw_bytes_read_for_integrity": True, "child_content_inspected": True, "date_year_value_read": date_read, "coverage_evaluated": True, "coverage_result_accepted": equal, "network_request_count": 0, "year_histograms": histograms}
        if equal:
            years = [item["year"] for item in histograms["4"]]
            missing = [year for year in REQUIRED_YEARS if year not in years]
            base.update({"covered_years": years, "covered_required_years": [year for year in years if year in REQUIRED_YEARS], "missing_required_years": missing, "all_required_years_covered": missing == []})
        return safe_coverage_evidence(base, structural)
    except CoverageBlocked:
        raise
    except Exception:
        raise CoverageBlocked(_failure(sha=digest, verified=True, raw=True, inspected=True, date_read=date_read)) from None


def run_date_year_coverage_parser(*, production_state_parent: str | Path, output_root: str | Path) -> dict[str, Any]:
    """Production entry point: frozen inherited binding, no overrides."""
    try:
        _meta_path, meta, raw_path = locate_metadata_only(production_state_parent=production_state_parent, output_root=output_root, bindings=FROZEN_BINDINGS)
        raw = content_blind_integrity_read(raw_path, meta, bindings=FROZEN_BINDINGS)
    except ProbeBlocked as exc:
        raise CoverageBlocked(_failure(sha=None, verified=False, raw=exc.raw_bytes_read_for_integrity, inspected=exc.child_content_inspected, date_read=False)) from None
    return _run_verified_bytes(raw)
