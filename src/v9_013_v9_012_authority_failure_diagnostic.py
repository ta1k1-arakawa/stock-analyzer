"""Read-only synthetic and future protected-input V9_013 diagnostics.

This module intentionally has no acquisition, HTTP, credential, or network
dependency.  The protected-state entry point is explicit and read-only; all
semantic development is performed through the same payload/result functions
against caller-supplied synthetic bytes.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


STUDY_ID = "V9_013_V9_012_AUTHORITY_FAILURE_DIAGNOSTIC"
RESULT_SCHEMA_VERSION = "V9_013_AUTHORITY_FAILURE_DIAGNOSTIC_RESULT_V1"
SOURCE_A = "SOURCE_A"
SOURCE_B = "SOURCE_B"
SOURCE_A_ROLE = "SCHEDULED_TSE_BUSINESS_DAY_SUPERSET"
SOURCE_B_ROLE = "ACTUAL_TSE_MARKET_ACTIVITY_DATE_EVIDENCE"
SOURCE_A_API_IDENTITY = "https://api.jquants.com/v2/markets/calendar"
SOURCE_B_API_IDENTITY = "https://api.jquants.com/v2/indices/bars/daily/topix"
COVERED_START = "2017-01-01"
COVERED_END = "2026-01-31"
EXPECTED_EXCEPTION_SET = frozenset({"2020-10-01"})
FROZEN_SOURCE_A_CHAIN_SHA256 = (
    "aee49fac48358be373ac4efbcf0568b796c68fa31177e0f34c5031352297fe45"
)
FROZEN_SOURCE_B_CHAIN_SHA256 = (
    "7b4c8624b78d51a30625672c411a76fcd85ab692765e99ee9cf6cc2239a3e33e"
)
FROZEN_SOURCE_A_PAGE_COUNT = 1
FROZEN_SOURCE_B_PAGE_COUNT = 1
FROZEN_DESIGN_GIT_SHA = "16b29e1ce0fbce4e73f4fb99aa7c7e38e3d78506"
FROZEN_DESIGN_BLOB_SHA = "5862648998ff571e4801f862bb78a0e66269ba44"
DATE_RE = re.compile(r"^20[0-9]{2}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

SOURCE_A_CATEGORIES = (
    "A_PAYLOAD_JSON_DECODE_FAILURE",
    "A_PAYLOAD_ROOT_SCHEMA_FAILURE",
    "A_DATA_FIELD_SCHEMA_FAILURE",
    "A_ROW_SCHEMA_FAILURE",
    "A_REQUIRED_FIELD_MISSING",
    "A_DATE_TYPE_OR_FORMAT_INVALID",
    "A_DATE_VALUE_INVALID",
    "A_DATE_OUT_OF_COVERAGE",
    "A_HOLDIV_TYPE_OR_DOMAIN_INVALID",
    "A_DUPLICATE_DATE",
    "A_COVERAGE_DATE_SET_MISMATCH",
    "A_VALID",
)
SOURCE_B_CATEGORIES = (
    "B_PAYLOAD_JSON_DECODE_FAILURE",
    "B_PAYLOAD_ROOT_SCHEMA_FAILURE",
    "B_DATA_FIELD_SCHEMA_FAILURE",
    "B_ROW_SCHEMA_FAILURE",
    "B_REQUIRED_FIELD_MISSING",
    "B_DATE_TYPE_OR_FORMAT_INVALID",
    "B_DATE_VALUE_INVALID",
    "B_DATE_OUT_OF_COVERAGE",
    "B_DUPLICATE_DATE",
    "B_OHLC_MIXED_NULL_FAILURE",
    "B_OHLC_NONFINITE_OR_TYPE_FAILURE",
    "B_VALID",
)
DIAGNOSTIC_CLASSES = (
    "SOURCE_A_SEMANTIC_FAILURE",
    "SOURCE_B_SEMANTIC_FAILURE",
    "RELATION_OR_SENTINEL_FAILURE",
    "NO_V9_012_FAILURE_REPRODUCED",
)
OBSERVED_JSON_TYPES = frozenset({"null", "bool", "int", "float", "string", "list", "object"})
LOCATION_KEYS = frozenset({
    "source_role", "page_index", "row_index", "field_name", "observed_json_type",
})
PUBLIC_KEYS = frozenset({
    "schema_version", "study_id", "status", "diagnostic_class",
    "source_a_category", "source_a_failure_location",
    "source_b_category", "source_b_failure_location",
    "source_a_row_count", "source_b_row_count", "scheduled_open_count",
    "topix_active_count", "relation_evaluated", "left_diff_count",
    "right_diff_count", "unexpected_left_diff_count",
    "missing_expected_exception_count", "left_diff_sha256",
    "right_diff_sha256", "left_exact_expected", "right_empty",
    "neighbor_2020_09_30_active", "sentinel_2020_10_01_inactive",
    "neighbor_2020_10_02_active", "source_a_chain_sha256",
    "source_b_chain_sha256", "diagnostic_design_git_sha",
    "diagnostic_implementation_git_sha",
})
RELATION_KEYS = (
    "left_diff_count", "right_diff_count", "unexpected_left_diff_count",
    "missing_expected_exception_count", "left_diff_sha256", "right_diff_sha256",
    "left_exact_expected", "right_empty", "neighbor_2020_09_30_active",
    "sentinel_2020_10_01_inactive", "neighbor_2020_10_02_active",
)
SOURCE_A_FIELDS = frozenset({"root", "data", "Date", "HolDiv"})
SOURCE_B_FIELDS = frozenset({"root", "data", "Date", "O", "H", "L", "C"})


def canonical_json_no_lf(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class DiagnosticError(RuntimeError):
    """A safe, public-reason-free diagnostic failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PayloadPage:
    page_index: int
    payload: bytes


@dataclass(frozen=True)
class _ParsedPage:
    page_index: int
    rows: list[dict[str, object]]


def _observed_json_type(value: object) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "bool"
    if type(value) is int:
        return "int"
    if type(value) is float:
        return "float"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "list"
    if type(value) is dict:
        return "object"
    raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")


def observed_json_type(value: object) -> str:
    """Expose the frozen mapping for synthetic tests and future callers."""
    return _observed_json_type(value)


def _location(
    role: str,
    page_index: int | None,
    row_index: int | None,
    field_name: str | None,
    observed_type: str | None,
) -> dict[str, object]:
    return {
        "source_role": role,
        "page_index": page_index,
        "row_index": row_index,
        "field_name": field_name,
        "observed_json_type": observed_type,
    }


def _coverage_dates() -> list[str]:
    start = _dt.date.fromisoformat(COVERED_START)
    end = _dt.date.fromisoformat(COVERED_END)
    return [
        (start + _dt.timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    ]


def _failure(
    category: str,
    role: str,
    page_index: int | None,
    row_index: int | None,
    field_name: str | None,
    observed_type: str | None,
    rows: int | None,
) -> tuple[str, dict[str, object], int | None]:
    return category, _location(role, page_index, row_index, field_name, observed_type), rows


def _decode_page(
    page: PayloadPage,
    role: str,
    source: str,
) -> tuple[_ParsedPage | None, tuple[str, dict[str, object], int | None] | None]:
    try:
        value = json.loads(page.payload.decode("utf-8"))
    except Exception:
        return None, _failure(
            f"{source}_PAYLOAD_JSON_DECODE_FAILURE", role, page.page_index,
            None, None, None, None,
        )
    if type(value) is not dict:
        return None, _failure(
            f"{source}_PAYLOAD_ROOT_SCHEMA_FAILURE", role, page.page_index,
            None, "root", _observed_json_type(value), None,
        )
    data = value.get("data")
    if type(data) is not list:
        return None, _failure(
            f"{source}_DATA_FIELD_SCHEMA_FAILURE", role, page.page_index,
            None, "data", _observed_json_type(data), None,
        )
    for row_number, row in enumerate(data, 1):
        if type(row) is not dict:
            return None, _failure(
                f"{source}_ROW_SCHEMA_FAILURE", role, page.page_index,
                row_number, None, _observed_json_type(row), len(data),
            )
    return _ParsedPage(page.page_index, data), None


def _strict_date(value: object) -> tuple[str | None, str | None]:
    if type(value) is not str or DATE_RE.fullmatch(value) is None:
        return None, "TYPE_OR_FORMAT"
    try:
        _dt.date.fromisoformat(value)
    except ValueError:
        return None, "VALUE"
    if value < COVERED_START or value > COVERED_END:
        return None, "OUT_OF_COVERAGE"
    return value, None


def _diagnose_source_a(
    pages: Sequence[PayloadPage],
) -> tuple[str, dict[str, object] | None, int | None, int | None, set[str] | None]:
    parsed: list[_ParsedPage] = []
    for page in sorted(pages, key=lambda item: item.page_index):
        decoded, failure = _decode_page(page, SOURCE_A_ROLE, "A")
        if failure is not None:
            category, location, row_count = failure
            if category in {
                "A_PAYLOAD_JSON_DECODE_FAILURE",
                "A_PAYLOAD_ROOT_SCHEMA_FAILURE",
                "A_DATA_FIELD_SCHEMA_FAILURE",
            }:
                total_rows = None
            else:
                total_rows = sum(len(item.rows) for item in parsed) + (row_count or 0)
            return category, location, total_rows, None, None
        assert decoded is not None
        parsed.append(decoded)
    seen: set[str] = set()
    rows_total = sum(len(item.rows) for item in parsed)
    for parsed_page in parsed:
        for row_index, row in enumerate(parsed_page.rows, 1):
            if "Date" not in row:
                return _failure(
                    "A_REQUIRED_FIELD_MISSING", SOURCE_A_ROLE, parsed_page.page_index,
                    row_index, "Date", None, rows_total,
                ) + (None, None)
            if "HolDiv" not in row:
                return _failure(
                    "A_REQUIRED_FIELD_MISSING", SOURCE_A_ROLE, parsed_page.page_index,
                    row_index, "HolDiv", None, rows_total,
                ) + (None, None)
            date_value, date_error = _strict_date(row["Date"])
            if date_error == "TYPE_OR_FORMAT":
                return _failure(
                    "A_DATE_TYPE_OR_FORMAT_INVALID", SOURCE_A_ROLE, parsed_page.page_index,
                    row_index, "Date", _observed_json_type(row["Date"]), rows_total,
                ) + (None, None)
            if date_error == "VALUE":
                return _failure(
                    "A_DATE_VALUE_INVALID", SOURCE_A_ROLE, parsed_page.page_index,
                    row_index, "Date", "string", rows_total,
                ) + (None, None)
            if date_error == "OUT_OF_COVERAGE":
                return _failure(
                    "A_DATE_OUT_OF_COVERAGE", SOURCE_A_ROLE, parsed_page.page_index,
                    row_index, "Date", "string", rows_total,
                ) + (None, None)
            assert date_value is not None
            hol_div = row["HolDiv"]
            if type(hol_div) is not str or hol_div not in {"0", "1", "2", "3"}:
                return _failure(
                    "A_HOLDIV_TYPE_OR_DOMAIN_INVALID", SOURCE_A_ROLE, parsed_page.page_index,
                    row_index, "HolDiv", _observed_json_type(hol_div), rows_total,
                ) + (None, None)
            if date_value in seen:
                return _failure(
                    "A_DUPLICATE_DATE", SOURCE_A_ROLE, parsed_page.page_index,
                    row_index, "Date", "string", rows_total,
                ) + (None, None)
            seen.add(date_value)
    if sorted(seen) != _coverage_dates():
        return _failure(
            "A_COVERAGE_DATE_SET_MISMATCH", SOURCE_A_ROLE, None, None,
            "Date", None, rows_total,
        ) + (None, None)
    scheduled = {
        row["Date"]
        for item in parsed
        for row in item.rows
        if row["HolDiv"] in {"1", "2"}
    }
    return "A_VALID", None, rows_total, len(scheduled), scheduled


def _finite_real(value: object) -> bool:
    if type(value) not in {int, float} or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _diagnose_source_b(
    pages: Sequence[PayloadPage],
) -> tuple[str, dict[str, object] | None, int | None, int | None, set[str] | None]:
    parsed: list[_ParsedPage] = []
    for page in sorted(pages, key=lambda item: item.page_index):
        decoded, failure = _decode_page(page, SOURCE_B_ROLE, "B")
        if failure is not None:
            category, location, row_count = failure
            if category in {
                "B_PAYLOAD_JSON_DECODE_FAILURE",
                "B_PAYLOAD_ROOT_SCHEMA_FAILURE",
                "B_DATA_FIELD_SCHEMA_FAILURE",
            }:
                total_rows = None
            else:
                total_rows = sum(len(item.rows) for item in parsed) + (row_count or 0)
            return category, location, total_rows, None, None
        assert decoded is not None
        parsed.append(decoded)
    seen: set[str] = set()
    active: set[str] = set()
    rows_total = sum(len(item.rows) for item in parsed)
    for parsed_page in parsed:
        for row_index, row in enumerate(parsed_page.rows, 1):
            for field in ("Date", "O", "H", "L", "C"):
                if field not in row:
                    return _failure(
                        "B_REQUIRED_FIELD_MISSING", SOURCE_B_ROLE, parsed_page.page_index,
                        row_index, field, None, rows_total,
                    ) + (None, None)
            date_value, date_error = _strict_date(row["Date"])
            if date_error == "TYPE_OR_FORMAT":
                return _failure(
                    "B_DATE_TYPE_OR_FORMAT_INVALID", SOURCE_B_ROLE, parsed_page.page_index,
                    row_index, "Date", _observed_json_type(row["Date"]), rows_total,
                ) + (None, None)
            if date_error == "VALUE":
                return _failure(
                    "B_DATE_VALUE_INVALID", SOURCE_B_ROLE, parsed_page.page_index,
                    row_index, "Date", "string", rows_total,
                ) + (None, None)
            if date_error == "OUT_OF_COVERAGE":
                return _failure(
                    "B_DATE_OUT_OF_COVERAGE", SOURCE_B_ROLE, parsed_page.page_index,
                    row_index, "Date", "string", rows_total,
                ) + (None, None)
            assert date_value is not None
            if date_value in seen:
                return _failure(
                    "B_DUPLICATE_DATE", SOURCE_B_ROLE, parsed_page.page_index,
                    row_index, "Date", "string", rows_total,
                ) + (None, None)
            seen.add(date_value)
            values = [row[field] for field in ("O", "H", "L", "C")]
            null_count = sum(value is None for value in values)
            if null_count == 4:
                continue
            if null_count != 0:
                return _failure(
                    "B_OHLC_MIXED_NULL_FAILURE", SOURCE_B_ROLE, parsed_page.page_index,
                    row_index, None, None, rows_total,
                ) + (None, None)
            for field, value in zip(("O", "H", "L", "C"), values):
                if not _finite_real(value):
                    return _failure(
                        "B_OHLC_NONFINITE_OR_TYPE_FAILURE", SOURCE_B_ROLE,
                        parsed_page.page_index, row_index, field,
                        _observed_json_type(value), rows_total,
                    ) + (None, None)
            active.add(date_value)
    return "B_VALID", None, rows_total, len(active), active


def _relation_result(
    scheduled_open_dates: set[str],
    topix_active_dates: set[str],
) -> dict[str, object]:
    left = scheduled_open_dates - topix_active_dates
    right = topix_active_dates - scheduled_open_dates
    return {
        "relation_evaluated": True,
        "left_diff_count": len(left),
        "right_diff_count": len(right),
        "unexpected_left_diff_count": len(left - EXPECTED_EXCEPTION_SET),
        "missing_expected_exception_count": len(EXPECTED_EXCEPTION_SET - left),
        "left_diff_sha256": sha256_bytes(canonical_json_no_lf(sorted(left))),
        "right_diff_sha256": sha256_bytes(canonical_json_no_lf(sorted(right))),
        "left_exact_expected": left == EXPECTED_EXCEPTION_SET,
        "right_empty": not right,
        "neighbor_2020_09_30_active": "2020-09-30" in topix_active_dates,
        "sentinel_2020_10_01_inactive": "2020-10-01" not in topix_active_dates,
        "neighbor_2020_10_02_active": "2020-10-02" in topix_active_dates,
    }


def _null_relation() -> dict[str, object]:
    return {"relation_evaluated": False, **{key: None for key in RELATION_KEYS}}


def _empty_result(
    diagnostic_class: str,
    source_a_category: str,
    source_a_location: dict[str, object] | None,
    source_a_row_count: int | None,
    scheduled_open_count: int | None,
    source_b_category: str | None,
    source_b_location: dict[str, object] | None,
    source_b_row_count: int | None,
    topix_active_count: int | None,
    relation: dict[str, object],
    diagnostic_design_git_sha: str,
    diagnostic_implementation_git_sha: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "status": "COMPLETE",
        "diagnostic_class": diagnostic_class,
        "source_a_category": source_a_category,
        "source_a_failure_location": source_a_location,
        "source_b_category": source_b_category,
        "source_b_failure_location": source_b_location,
        "source_a_row_count": source_a_row_count,
        "source_b_row_count": source_b_row_count,
        "scheduled_open_count": scheduled_open_count,
        "topix_active_count": topix_active_count,
        "source_a_chain_sha256": FROZEN_SOURCE_A_CHAIN_SHA256,
        "source_b_chain_sha256": FROZEN_SOURCE_B_CHAIN_SHA256,
        "diagnostic_design_git_sha": diagnostic_design_git_sha,
        "diagnostic_implementation_git_sha": diagnostic_implementation_git_sha,
    }
    result.update(relation)
    return result


def _validate_sha(value: object, pattern: re.Pattern[str]) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _validate_location(value: object, source: str) -> None:
    if type(value) is not dict or set(value) != LOCATION_KEYS:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    role = SOURCE_A_ROLE if source == SOURCE_A else SOURCE_B_ROLE
    if value["source_role"] != role:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if value["page_index"] is not None and (
        type(value["page_index"]) is not int or value["page_index"] < 1
    ):
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if value["row_index"] is not None and (
        type(value["row_index"]) is not int or value["row_index"] < 1
    ):
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    field_names = SOURCE_A_FIELDS if source == SOURCE_A else SOURCE_B_FIELDS
    if value["field_name"] is not None and value["field_name"] not in field_names:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if value["observed_json_type"] is not None and value["observed_json_type"] not in OBSERVED_JSON_TYPES:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")


def _reject_public_date_values(value: object) -> None:
    if type(value) is str and DATE_RE.fullmatch(value):
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if type(value) is dict:
        for child in value.values():
            _reject_public_date_values(child)
    elif type(value) is list:
        for child in value:
            _reject_public_date_values(child)


def validate_public_result(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != PUBLIC_KEYS:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if value["schema_version"] != RESULT_SCHEMA_VERSION or value["study_id"] != STUDY_ID:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if value["status"] != "COMPLETE" or value["diagnostic_class"] not in DIAGNOSTIC_CLASSES:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if value["source_a_category"] not in SOURCE_A_CATEGORIES:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    a_valid = value["source_a_category"] == "A_VALID"
    if (value["source_a_failure_location"] is None) != a_valid:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if not a_valid:
        _validate_location(value["source_a_failure_location"], SOURCE_A)
    b_category = value["source_b_category"]
    if not a_valid:
        if b_category is not None or value["source_b_failure_location"] is not None:
            raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    else:
        if b_category not in SOURCE_B_CATEGORIES:
            raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
        b_valid = b_category == "B_VALID"
        if (value["source_b_failure_location"] is None) != b_valid:
            raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
        if not b_valid:
            _validate_location(value["source_b_failure_location"], SOURCE_B)
    for field in ("source_a_row_count", "source_b_row_count", "scheduled_open_count", "topix_active_count"):
        if value[field] is not None and (type(value[field]) is not int or value[field] < 0):
            raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    a_row_count_is_null = value["source_a_row_count"] is None
    a_row_count_must_be_null = value["source_a_category"] in {
        "A_PAYLOAD_JSON_DECODE_FAILURE", "A_PAYLOAD_ROOT_SCHEMA_FAILURE", "A_DATA_FIELD_SCHEMA_FAILURE",
    }
    if a_row_count_is_null != a_row_count_must_be_null:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if not a_valid:
        if value["source_b_row_count"] is not None:
            raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    else:
        b_row_count_is_null = value["source_b_row_count"] is None
        b_row_count_must_be_null = b_category in {
            "B_PAYLOAD_JSON_DECODE_FAILURE", "B_PAYLOAD_ROOT_SCHEMA_FAILURE", "B_DATA_FIELD_SCHEMA_FAILURE",
        }
        if b_row_count_is_null != b_row_count_must_be_null:
            raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if (value["scheduled_open_count"] is None) != (not a_valid):
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    b_valid = a_valid and b_category == "B_VALID"
    if (value["topix_active_count"] is None) != (not b_valid):
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if type(value["relation_evaluated"]) is not bool:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if value["relation_evaluated"] != b_valid:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    relation_values = [value[field] for field in RELATION_KEYS]
    if value["relation_evaluated"] and any(item is None for item in relation_values):
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if not value["relation_evaluated"] and any(item is not None for item in relation_values):
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    for field in RELATION_KEYS[:4]:
        if value[field] is not None and (type(value[field]) is not int or value[field] < 0):
            raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    for field in ("left_diff_sha256", "right_diff_sha256"):
        if value[field] is not None and not _validate_sha(value[field], HEX64_RE):
            raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    for field in RELATION_KEYS[6:]:
        if value[field] is not None and type(value[field]) is not bool:
            raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if not a_valid:
        expected_class = "SOURCE_A_SEMANTIC_FAILURE"
    elif not b_valid:
        expected_class = "SOURCE_B_SEMANTIC_FAILURE"
    elif not (
        value["left_exact_expected"]
        and value["right_empty"]
        and value["neighbor_2020_09_30_active"]
        and value["sentinel_2020_10_01_inactive"]
        and value["neighbor_2020_10_02_active"]
    ):
        expected_class = "RELATION_OR_SENTINEL_FAILURE"
    else:
        expected_class = "NO_V9_012_FAILURE_REPRODUCED"
    if value["diagnostic_class"] != expected_class:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if value["source_a_chain_sha256"] != FROZEN_SOURCE_A_CHAIN_SHA256 or value["source_b_chain_sha256"] != FROZEN_SOURCE_B_CHAIN_SHA256:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    if not _validate_sha(value["diagnostic_design_git_sha"], HEX40_RE) or not _validate_sha(value["diagnostic_implementation_git_sha"], HEX40_RE):
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    _reject_public_date_values(value)
    return dict(value)


def serialize_public_result(value: Mapping[str, object]) -> bytes:
    checked = validate_public_result(dict(value))
    try:
        return canonical_json_no_lf(checked)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE") from exc


def diagnose_payloads(
    source_a_pages: Sequence[PayloadPage | bytes],
    source_b_pages: Sequence[PayloadPage | bytes],
    *,
    diagnostic_design_git_sha: str = FROZEN_DESIGN_GIT_SHA,
    diagnostic_implementation_git_sha: str = "0000000000000000000000000000000000000000",
) -> dict[str, object]:
    """Run the production semantic/result computation on synthetic payloads."""
    def normalize(items: Sequence[PayloadPage | bytes]) -> list[PayloadPage]:
        normalized: list[PayloadPage] = []
        for index, item in enumerate(items, 1):
            if isinstance(item, PayloadPage):
                normalized.append(item)
            elif type(item) is bytes:
                normalized.append(PayloadPage(index, item))
            else:
                raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
        return normalized

    a_pages = normalize(source_a_pages)
    b_pages = normalize(source_b_pages)
    if not _validate_sha(diagnostic_design_git_sha, HEX40_RE) or not _validate_sha(diagnostic_implementation_git_sha, HEX40_RE):
        raise DiagnosticError("DIAGNOSTIC_RESULT_VALIDATION_FAILURE")
    a_category, a_location, a_rows, a_open_count, scheduled = _diagnose_source_a(a_pages)
    if a_category != "A_VALID":
        result = _empty_result(
            "SOURCE_A_SEMANTIC_FAILURE", a_category, a_location, a_rows, None,
            None, None, None, None, _null_relation(),
            diagnostic_design_git_sha, diagnostic_implementation_git_sha,
        )
        return validate_public_result(result)
    assert scheduled is not None
    b_category, b_location, b_rows, b_active_count, active = _diagnose_source_b(b_pages)
    if b_category != "B_VALID":
        result = _empty_result(
            "SOURCE_B_SEMANTIC_FAILURE", a_category, None, a_rows, a_open_count,
            b_category, b_location, b_rows, None, _null_relation(),
            diagnostic_design_git_sha, diagnostic_implementation_git_sha,
        )
        return validate_public_result(result)
    assert active is not None
    relation = _relation_result(scheduled, active)
    relation_class = (
        "NO_V9_012_FAILURE_REPRODUCED"
        if relation["left_exact_expected"] and relation["right_empty"]
        and relation["neighbor_2020_09_30_active"]
        and relation["sentinel_2020_10_01_inactive"]
        and relation["neighbor_2020_10_02_active"]
        else "RELATION_OR_SENTINEL_FAILURE"
    )
    result = _empty_result(
        relation_class, a_category, None, a_rows, a_open_count,
        b_category, None, b_rows, b_active_count, relation,
        diagnostic_design_git_sha, diagnostic_implementation_git_sha,
    )
    return validate_public_result(result)


LOCK_KEYS = frozenset({
    "byte_count", "http_status", "page_index", "page_request_identity_sha256",
    "payload_sha256", "source_api_identity", "source_role",
})
SOURCE_CHAIN_KEYS = frozenset({
    "base_query_sha256", "page_count", "pages", "source_api_identity",
    "source_role", "terminal_page_index",
})
SOURCE_CHAIN_PAGE_KEYS = frozenset({
    "byte_count", "continuation_issued", "continuation_key_sha256",
    "page_index", "page_request_identity_sha256", "payload_sha256",
})


def _base_query_sha256(source: str) -> str:
    return sha256_bytes(canonical_json_no_lf({"from": COVERED_START, "to": COVERED_END}))


def _source_identity(source: str) -> tuple[str, str]:
    if source == SOURCE_A:
        return SOURCE_A_API_IDENTITY, SOURCE_A_ROLE
    if source == SOURCE_B:
        return SOURCE_B_API_IDENTITY, SOURCE_B_ROLE
    raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")


def _pagination_key_sha256(key: str) -> str:
    if type(key) is not str or key == "":
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
    return sha256_bytes(key.encode("utf-8"))


def _request_identity_sha256(source: str, page_index: int, previous_key: str | None) -> str:
    api_identity, role = _source_identity(source)
    value = {
        "base_query_sha256": _base_query_sha256(source),
        "continuation_key_sha256": None if previous_key is None else _pagination_key_sha256(previous_key),
        "page_index": page_index,
        "source_api_identity": api_identity,
        "source_role": role,
    }
    return sha256_bytes(canonical_json_no_lf(value))


def _read_locked_chain(state_root: str | Path, source: str) -> list[PayloadPage]:
    root = Path(state_root).resolve()
    source_dir = root / ("source_a" if source == SOURCE_A else "source_b")
    raw_dir = source_dir / "raw_pages"
    lock_dir = source_dir / "page_locks"
    try:
        if not source_dir.is_dir() or not raw_dir.is_dir() or not lock_dir.is_dir():
            raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
        raw_paths = list(raw_dir.iterdir())
        lock_paths = list(lock_dir.iterdir())
        if not raw_paths or len(raw_paths) != len(lock_paths):
            raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
        indices = sorted(int(path.stem) for path in lock_paths if path.suffix == ".json" and path.stem.isdigit())
        if len(indices) != len(lock_paths) or indices != list(range(1, len(indices) + 1)):
            raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
        if {int(path.stem) for path in raw_paths if path.suffix == ".bin" and path.stem.isdigit()} != set(indices):
            raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
        pages: list[PayloadPage] = []
        previous_key: str | None = None
        seen_keys: set[str] = set()
        for index in indices:
            lock_path = lock_dir / f"{index:06d}.json"
            raw_path = raw_dir / f"{index:06d}.bin"
            lock_bytes = lock_path.read_bytes()
            record = json.loads(lock_bytes.decode("utf-8"))
            payload = raw_path.read_bytes()
            if type(record) is not dict or set(record) != LOCK_KEYS:
                raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
            api_identity, role = _source_identity(source)
            if (
                type(record["http_status"]) is not int or record["http_status"] != 200
                or record["page_index"] != index
                or record["source_api_identity"] != api_identity
                or record["source_role"] != role
                or record["page_request_identity_sha256"] != _request_identity_sha256(source, index, previous_key)
                or type(record["byte_count"]) is not int or record["byte_count"] < 0
                or record["byte_count"] != len(payload)
                or type(record["payload_sha256"]) is not str
                or not HEX64_RE.fullmatch(record["payload_sha256"])
                or record["payload_sha256"] != sha256_bytes(payload)
                or lock_bytes != canonical_json_no_lf(record) + b"\n"
            ):
                raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
            try:
                envelope = json.loads(payload.decode("utf-8"))
            except Exception as exc:
                raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE") from exc
            if type(envelope) is not dict:
                raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
            if "pagination_key" not in envelope:
                continued = False
                next_key = None
            else:
                next_key = envelope["pagination_key"]
                if type(next_key) is not str or next_key == "" or next_key in seen_keys:
                    raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
                seen_keys.add(next_key)
                continued = True
            if index < indices[-1] and not continued:
                raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
            if index == indices[-1] and continued:
                raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
            pages.append(PayloadPage(index, payload))
            previous_key = next_key
        return pages
    except DiagnosticError:
        raise
    except Exception as exc:
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE") from exc


def _build_chain_manifest(source: str, pages: Sequence[PayloadPage]) -> dict[str, object]:
    api_identity, role = _source_identity(source)
    if not pages:
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
    entries: list[dict[str, object]] = []
    previous_key: str | None = None
    for page in pages:
        try:
            envelope = json.loads(page.payload.decode("utf-8"))
        except Exception as exc:
            raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE") from exc
        continued = "pagination_key" in envelope
        next_key = envelope.get("pagination_key") if continued else None
        entries.append({
            "byte_count": len(page.payload),
            "continuation_issued": continued,
            "continuation_key_sha256": None if next_key is None else _pagination_key_sha256(next_key),
            "page_index": page.page_index,
            "page_request_identity_sha256": _request_identity_sha256(source, page.page_index, previous_key),
            "payload_sha256": sha256_bytes(page.payload),
        })
        previous_key = next_key
    manifest = {
        "base_query_sha256": _base_query_sha256(source),
        "page_count": len(entries),
        "pages": entries,
        "source_api_identity": api_identity,
        "source_role": role,
        "terminal_page_index": len(entries),
    }
    return manifest


def _verify_frozen_chains(state_root: str | Path) -> tuple[list[PayloadPage], list[PayloadPage]]:
    root = Path(state_root).resolve()
    try:
        children = {child.name for child in root.iterdir()}
    except Exception as exc:
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE") from exc
    if children != {"source_a", "source_b"}:
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
    a_pages = _read_locked_chain(root, SOURCE_A)
    if len(a_pages) != FROZEN_SOURCE_A_PAGE_COUNT:
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
    a_manifest = _build_chain_manifest(SOURCE_A, a_pages)
    if sha256_bytes(canonical_json_no_lf(a_manifest)) != FROZEN_SOURCE_A_CHAIN_SHA256:
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
    b_pages = _read_locked_chain(root, SOURCE_B)
    if len(b_pages) != FROZEN_SOURCE_B_PAGE_COUNT:
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
    b_manifest = _build_chain_manifest(SOURCE_B, b_pages)
    if sha256_bytes(canonical_json_no_lf(b_manifest)) != FROZEN_SOURCE_B_CHAIN_SHA256:
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
    return a_pages, b_pages


def diagnose_preserved_state(
    state_root: str | Path,
    *,
    diagnostic_design_git_sha: str,
    diagnostic_implementation_git_sha: str,
) -> dict[str, object]:
    """Explicit future protected-state path; no acquisition or mutation."""
    if diagnostic_design_git_sha != FROZEN_DESIGN_GIT_SHA:
        raise DiagnosticError("PRESERVED_V9_012_INPUT_BINDING_FAILURE")
    if not _validate_sha(diagnostic_implementation_git_sha, HEX40_RE):
        raise DiagnosticError("DIAGNOSTIC_PROVENANCE_INVALID")
    source_a_pages, source_b_pages = _verify_frozen_chains(state_root)
    return diagnose_payloads(
        source_a_pages,
        source_b_pages,
        diagnostic_design_git_sha=diagnostic_design_git_sha,
        diagnostic_implementation_git_sha=diagnostic_implementation_git_sha,
    )


def safe_error_bytes(reason: str) -> bytes:
    if reason not in {
        "PRESERVED_V9_012_INPUT_BINDING_FAILURE",
        "DIAGNOSTIC_PROVENANCE_INVALID",
        "DIAGNOSTIC_RESULT_VALIDATION_FAILURE",
        "IMPLEMENTATION_FAILURE",
    }:
        reason = "IMPLEMENTATION_FAILURE"
    return canonical_json_no_lf({"reason": reason, "status": "BLOCKED"})
