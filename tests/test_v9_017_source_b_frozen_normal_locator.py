import json
from pathlib import Path

import pytest

from src import v9_017_source_b_frozen_normal_locator as locator


def _cell(text: str = "", tag: str = "td") -> str:
    return f"<{tag}>{text}</{tag}>"


def _row(cells: list[str]) -> str:
    return "<tr>" + "".join(cells) + "</tr>"


def _table(rows: list[str]) -> str:
    return "<table>" + "".join(rows) + "</table>"


def _page(
    month: int,
    *,
    header_token: str | None = None,
    report_anchor: str = locator.REPORT_ROW_ANCHOR,
    report_cell_count: int | None = None,
    report_href: str | None = "synthetic-target.pdf",
    extra_report_href: str | None = None,
    header_tag: str = "th",
    header_table: bool = True,
    report_table: bool = True,
    header_column_override: int | None = None,
    header_href: str | None = None,
) -> bytes:
    header_column = month - 1 if header_column_override is None else header_column_override
    token = locator.MONTH_TOKENS[month] if header_token is None else header_token
    header_cells = [_cell("header filler") for _ in range(header_column + 1)]
    header_cells[header_column] = _cell(
        f'<a href="{header_href}">{token}</a>' if header_href is not None else token,
        header_tag,
    )
    header_html = _row(header_cells)

    count = month + 1 if report_cell_count is None else report_cell_count
    report_cells = [_cell(report_anchor)] + [_cell("report filler") for _ in range(max(0, count - 1))]
    if count > month:
        report_value = ""
        if report_href is not None:
            report_value += f'<a href="{report_href}">candidate</a>'
        if extra_report_href is not None:
            report_value += f'<a href="{extra_report_href}">other</a>'
        report_cells[month] = _cell(report_value)
    report_html = _row(report_cells)

    tables: list[str] = []
    if header_table and report_table:
        tables.append(_table([header_html, report_html]))
    elif header_table:
        tables.append(_table([header_html]))
    elif report_table:
        tables.append(_table([report_html]))
    return ("<html><body>" + "".join(tables) + "</body></html>").encode("utf-8")


@pytest.mark.parametrize(
    ("month", "href"),
    [
        (1, "jan.pdf"),
        (3, "mar.pdf"),
        (4, "apr.pdf"),
        (5, "may.pdf"),
        (12, "dec.pdf"),
    ],
)
def test_each_supported_month_uses_header_m_minus_one_and_data_m(month: int, href: str) -> None:
    outcome = locator.locate_frozen_normal_month(_page(month, report_href=href), month)
    assert outcome.to_dict() == {
        "schema_version": locator.SCHEMA_VERSION,
        "status": "PASS",
        "reason": "OK",
        "month": f"{month:02d}",
        "report_row_count": 1,
        "header_row_count": 1,
        "href_count": 1,
    }
    assert locator._sole_candidate_href(outcome) == href
    assert href not in repr(outcome)


def test_may_period_is_not_accepted() -> None:
    outcome = locator.locate_frozen_normal_month(_page(5, header_token="May."), 5)
    assert outcome.to_dict()["reason"] == "HEADER_ROW_ZERO"


@pytest.mark.parametrize("token", ["Jan", "January", "01"])
def test_jan_alternate_tokens_are_not_accepted(token: str) -> None:
    outcome = locator.locate_frozen_normal_month(_page(1, header_token=token), 1)
    assert outcome.to_dict()["reason"] == "HEADER_ROW_ZERO"


def test_report_row_zero_and_stripped_anchor_fail() -> None:
    no_report = locator.locate_frozen_normal_month(_page(1, report_table=False), 1)
    stripped = locator.locate_frozen_normal_month(
        _page(1, report_anchor="Stock Trading Volume & Value"), 1
    )
    assert no_report.to_dict()["reason"] == "REPORT_ROW_ZERO"
    assert stripped.to_dict()["reason"] == "REPORT_ROW_ZERO"


def test_report_row_many_same_table_and_across_tables_fail() -> None:
    header_row = _row([_cell("Jan.", "th")])
    report_row = _row([_cell(locator.REPORT_ROW_ANCHOR), _cell("data")])
    duplicate_row = _row([_cell(locator.REPORT_ROW_ANCHOR), _cell("filler")])
    same_table = ("<html><body>" + _table([header_row, report_row, duplicate_row]) + "</body></html>").encode()
    assert locator.locate_frozen_normal_month(same_table, 1).to_dict()["reason"] == "REPORT_ROW_MANY"

    across = ("<html><body>" + _table([header_row, report_row]) + _table([duplicate_row]) + "</body></html>").encode()
    assert locator.locate_frozen_normal_month(across, 1).to_dict()["reason"] == "REPORT_ROW_MANY"


def test_header_zero_wrong_column_other_table_and_wrong_tag_fail() -> None:
    assert locator.locate_frozen_normal_month(_page(1, header_table=False), 1).to_dict()["reason"] == "HEADER_ROW_ZERO"
    assert locator.locate_frozen_normal_month(_page(3, header_column_override=1), 3).to_dict()["reason"] == "HEADER_ROW_ZERO"
    other_table = ("<html><body>" + _table([_row([_cell(locator.REPORT_ROW_ANCHOR), _cell("data")])]) + _table([_row([_cell("Jan.", "th")])]) + "</body></html>").encode()
    assert locator.locate_frozen_normal_month(other_table, 1).to_dict()["reason"] == "HEADER_ROW_ZERO"
    assert locator.locate_frozen_normal_month(_page(1, header_tag="td"), 1).to_dict()["reason"] == "HEADER_ROW_ZERO"


def test_matching_header_row_many_fails() -> None:
    base = _page(1).decode()
    duplicate_header = _row([_cell("Jan.", "th")])
    page = base.replace("</table>", duplicate_header + "</table>", 1).encode()
    outcome = locator.locate_frozen_normal_month(page, 1)
    assert outcome.to_dict()["reason"] == "HEADER_ROW_MANY"
    assert outcome.to_dict()["header_row_count"] == 2


def test_report_data_column_missing_fails() -> None:
    outcome = locator.locate_frozen_normal_month(_page(1, report_cell_count=1), 1)
    assert outcome.to_dict()["reason"] == "REPORT_DATA_COLUMN_MISSING"


def test_href_zero_and_many_fail() -> None:
    zero = locator.locate_frozen_normal_month(_page(1, report_href=None), 1)
    many = locator.locate_frozen_normal_month(
        _page(1, report_href="one.pdf", extra_report_href="two.pdf"), 1
    )
    assert zero.to_dict()["reason"] == "HREF_ZERO"
    assert zero.to_dict()["href_count"] == 0
    assert many.to_dict()["reason"] == "HREF_MANY"
    assert many.to_dict()["href_count"] == "MANY"


def test_plus_one_offset_rejects_href_at_header_column() -> None:
    outcome = locator.locate_frozen_normal_month(
        _page(3, report_href=None, header_href="wrong-column.pdf"), 3
    )
    assert outcome.to_dict()["reason"] == "HREF_ZERO"


def test_exact_whitespace_normalization() -> None:
    page = _page(1, header_token=" \tJan.\n ", report_anchor=" 2\n Stock Trading Volume & Value ")
    outcome = locator.locate_frozen_normal_month(page, 1)
    assert outcome.status == "PASS"


@pytest.mark.parametrize("month", [2, 6, "01", True, None])
def test_unsupported_month_fails_without_fallback(month) -> None:
    outcome = locator.locate_frozen_normal_month(b"<not-used>", month)
    assert outcome.to_dict()["reason"] == "UNSUPPORTED_MONTH"


def test_non_bytes_and_malformed_parser_fail_safely() -> None:
    non_bytes = locator.locate_frozen_normal_month("not bytes", 1)
    malformed = locator.locate_frozen_normal_month(b"<table><tr><td>bad", 1)
    assert non_bytes.to_dict()["reason"] == "INPUT_NOT_BYTES"
    assert malformed.to_dict()["reason"] == "PARSER_FAILURE"
    assert "bad" not in json.dumps(malformed.to_dict())


def test_safe_serialization_has_no_href_url_html_path_or_exception() -> None:
    page = _page(1, report_href="https://example.invalid/private.pdf")
    rendered = json.dumps(locator.locate_frozen_normal_month(page, 1).to_dict())
    assert "https://example.invalid/private.pdf" not in rendered
    assert "private.pdf" not in rendered
    assert "<a" not in rendered
    assert "\"href\":\"" not in rendered
    assert "Path(" not in repr(locator.locate_frozen_normal_month(page, 1))


def test_repeated_synthetic_input_is_deterministic_and_internal_href_is_not_reparsed() -> None:
    page = _page(12, report_href="dec.pdf")
    first = locator.locate_frozen_normal_month(page, 12)
    second = locator.locate_frozen_normal_month(page, 12)
    assert first.to_dict() == second.to_dict()
    assert locator._sole_candidate_href(first) == "dec.pdf"
    assert locator._sole_candidate_href(second) == "dec.pdf"


def test_source_has_no_network_or_selection_logic() -> None:
    source = Path(locator.__file__).read_text(encoding="utf-8")
    assert "requests" not in source
    assert "urllib" not in source
    assert "urljoin" not in source
    assert "category" not in source.lower()
    assert "report-label" not in source.lower()
    assert "month-grammar" not in source.lower()
