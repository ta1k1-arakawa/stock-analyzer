"""Synthetic-only implementation of the frozen V9_017 NORMAL locator."""

from __future__ import annotations

from typing import Any

from src.v9_005_stage_a_jpx_probe import _parse_monthly_statistics_html


SCHEMA_VERSION = "V9_017_FROZEN_NORMAL_LOCATOR_V1"
REPORT_ROW_ANCHOR = "2 Stock Trading Volume & Value"
MONTH_TOKENS = {
    1: "Jan.",
    3: "Mar.",
    4: "Apr.",
    5: "May",
    12: "Dec.",
}


def _normalize(raw_text: str) -> str:
    return " ".join(raw_text.split())


def _href_count_value(count: int) -> int | str:
    if count <= 1:
        return count
    return "MANY"


class NormalLocatorOutcome:
    """Internal outcome whose public representation never contains hrefs."""

    def __init__(
        self,
        *,
        status: str,
        reason: str,
        month: str | None,
        report_row_count: int,
        header_row_count: int,
        href_count: int | str,
        candidate_href: str | None,
    ) -> None:
        self.schema_version = SCHEMA_VERSION
        self.status = status
        self.reason = reason
        self.month = month
        self.report_row_count = report_row_count
        self.header_row_count = header_row_count
        self.href_count = href_count
        self._candidate_href = candidate_href

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "reason": self.reason,
            "month": self.month,
            "report_row_count": self.report_row_count,
            "header_row_count": self.header_row_count,
            "href_count": self.href_count,
        }

    def __repr__(self) -> str:
        return repr(self.to_dict())


def _outcome(
    *,
    reason: str,
    month: str | None,
    report_row_count: int = 0,
    header_row_count: int = 0,
    href_count: int | str = 0,
    candidate_href: str | None = None,
) -> NormalLocatorOutcome:
    return NormalLocatorOutcome(
        status="PASS" if reason == "OK" else "FAIL",
        reason=reason,
        month=month,
        report_row_count=report_row_count,
        header_row_count=header_row_count,
        href_count=href_count,
        candidate_href=candidate_href,
    )


def locate_frozen_normal_month(
    page_bytes: bytes, month: int
) -> NormalLocatorOutcome:
    """Locate one frozen NORMAL month using inherited parser mechanics only."""

    if not isinstance(page_bytes, bytes):
        return _outcome(reason="INPUT_NOT_BYTES", month=None)
    if isinstance(month, bool) or not isinstance(month, int) or month not in MONTH_TOKENS:
        return _outcome(reason="UNSUPPORTED_MONTH", month=None)

    month_text = f"{month:02d}"
    header_column = month - 1
    report_data_column = month
    try:
        parser = _parse_monthly_statistics_html(page_bytes)
    except Exception:
        return _outcome(reason="PARSER_FAILURE", month=month_text)

    report_rows: list[tuple[Any, list[Any]]] = []
    for table in parser.tables:
        for row in table.rows:
            if row and _normalize(row[0].text) == REPORT_ROW_ANCHOR:
                report_rows.append((table, row))

    report_row_count = len(report_rows)
    if report_row_count == 0:
        return _outcome(
            reason="REPORT_ROW_ZERO",
            month=month_text,
            report_row_count=report_row_count,
        )
    if report_row_count > 1:
        return _outcome(
            reason="REPORT_ROW_MANY",
            month=month_text,
            report_row_count=report_row_count,
        )

    containing_table, report_row = report_rows[0]
    header_rows: list[list[Any]] = []
    for row in containing_table.rows:
        if (
            len(row) > header_column
            and row[header_column].tag == "th"
            and _normalize(row[header_column].text) == MONTH_TOKENS[month]
        ):
            header_rows.append(row)

    header_row_count = len(header_rows)
    if header_row_count == 0:
        return _outcome(
            reason="HEADER_ROW_ZERO",
            month=month_text,
            report_row_count=report_row_count,
            header_row_count=header_row_count,
        )
    if header_row_count > 1:
        return _outcome(
            reason="HEADER_ROW_MANY",
            month=month_text,
            report_row_count=report_row_count,
            header_row_count=header_row_count,
        )

    if len(report_row) <= report_data_column:
        return _outcome(
            reason="REPORT_DATA_COLUMN_MISSING",
            month=month_text,
            report_row_count=report_row_count,
            header_row_count=header_row_count,
        )

    hrefs = report_row[report_data_column].hrefs
    href_count = len(hrefs)
    if href_count == 0:
        return _outcome(
            reason="HREF_ZERO",
            month=month_text,
            report_row_count=report_row_count,
            header_row_count=header_row_count,
            href_count=0,
        )
    if href_count > 1:
        return _outcome(
            reason="HREF_MANY",
            month=month_text,
            report_row_count=report_row_count,
            header_row_count=header_row_count,
            href_count=_href_count_value(href_count),
        )

    return _outcome(
        reason="OK",
        month=month_text,
        report_row_count=report_row_count,
        header_row_count=header_row_count,
        href_count=1,
        candidate_href=hrefs[0],
    )


def _sole_candidate_href(outcome: NormalLocatorOutcome) -> str:
    """Return the sole href for later internal composition after PASS."""

    if outcome.status != "PASS" or outcome.reason != "OK" or outcome._candidate_href is None:
        raise ValueError("candidate unavailable")
    return outcome._candidate_href


__all__ = [
    "MONTH_TOKENS",
    "NormalLocatorOutcome",
    "REPORT_ROW_ANCHOR",
    "SCHEMA_VERSION",
    "_sole_candidate_href",
    "locate_frozen_normal_month",
]
