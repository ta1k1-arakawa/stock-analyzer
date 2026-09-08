from __future__ import annotations

import importlib
import re

import pytest

import src.v9_017_source_b_fixed_eight_locator_application as application
from src.v9_014_jpx_monthly_auction_activity_source_b_pdf_calibration_probe import (
    REQUIRED_CALIBRATION_IDENTITIES,
)


ANCHOR = "2 Stock Trading Volume & Value"
PRE_LABEL = "(Reference) Status on April 1, 2022"
YEAR_PAGE_URL_2022 = (
    "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/2022/index.html"
)
MONTH_TOKENS = {1: "Jan.", 3: "Mar.", 4: "Apr.", 5: "May", 12: "Dec."}


def _cell(text: str = "", tag: str = "td", href: str | None = None) -> str:
    content = f'<a href="{href}">{text}</a>' if href is not None else text
    return f"<{tag}>{content}</{tag}>"


def _row(cells: list[str]) -> str:
    return "<tr>" + "".join(cells) + "</tr>"


def _normal_page(months: list[int], *, pre_count: int = 0) -> bytes:
    headers = [_cell(f"column-{index}") for index in range(13)]
    for month in months:
        headers[month - 1] = _cell(MONTH_TOKENS[month], "th")
    report = [_cell(ANCHOR)]
    for column in range(1, 13):
        href = f"/english/markets/statistics-equities/monthly/object-{column}.pdf"
        report.append(_cell(f"object-{column}", href=href))
    rows = [_row(headers), _row(report)]
    for count in range(pre_count):
        href = f"/english/markets/statistics-equities/monthly/pre-{count}.pdf"
        rows.append(_row([_cell(PRE_LABEL, href=href)]))
    return ("<html><body><table>" + "".join(rows) + "</table></body></html>").encode()


def _all_pages() -> tuple[dict[int, bytes], dict[int, str]]:
    return (
        {
            2017: _normal_page([1]),
            2019: _normal_page([12]),
            2020: _normal_page([1]),
            2022: _normal_page([3, 4, 5], pre_count=1),
            2026: _normal_page([1]),
        },
        {2022: YEAR_PAGE_URL_2022},
    )


def test_fixed_eight_order_and_all_eight_success() -> None:
    pages, urls = _all_pages()
    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "PASS"
    assert result.resolved_identity_count == 8
    assert result.to_dict()["identity_results"] == [
        {
            "logical_month": identity.logical_month,
            "object_part": identity.object_part,
            "status": "PASS",
        }
        for identity in REQUIRED_CALIBRATION_IDENTITIES
    ]
    assert len(application._selected_urls(result)) == 8


def test_all_normal_identities_use_frozen_normal_locator(monkeypatch: pytest.MonkeyPatch) -> None:
    pages, urls = _all_pages()
    original = application.locate_frozen_normal_month
    observed_months: list[int] = []

    def wrapped(page_bytes: bytes, month: int):
        observed_months.append(month)
        return original(page_bytes, month)

    monkeypatch.setattr(application, "locate_frozen_normal_month", wrapped)
    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "PASS"
    assert observed_months == [1, 12, 1, 3, 4, 5, 1]


def test_pre_uses_only_inherited_resolver_and_selected_year_2022(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages, urls = _all_pages()
    extract_original = application.extract_april_pre_candidates
    resolve_original = application.resolve_source_b_april_pre_object
    observed_years: list[int] = []
    resolver_calls = 0

    def wrapped_extract(page_bytes: bytes, page_url: str, *, selected_year: int):
        observed_years.append(selected_year)
        return extract_original(page_bytes, page_url, selected_year=selected_year)

    def wrapped_resolve(candidates, *, selected_year_page_url: str):
        nonlocal resolver_calls
        resolver_calls += 1
        return resolve_original(candidates, selected_year_page_url=selected_year_page_url)

    monkeypatch.setattr(application, "extract_april_pre_candidates", wrapped_extract)
    monkeypatch.setattr(application, "resolve_source_b_april_pre_object", wrapped_resolve)
    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "PASS"
    assert observed_years == [2022]
    assert resolver_calls == 1


def test_old_v9014_normal_resolver_is_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    pages, urls = _all_pages()
    old_locator = importlib.import_module(
        "src.v9_014_jpx_monthly_auction_activity_source_b_locator"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("old normal resolver used")

    monkeypatch.setattr(old_locator, "resolve_source_b_normal_month_object", forbidden)
    assert application.apply_fixed_eight_locators(pages, urls).status == "PASS"


def test_normal_zero_candidate_fails_closed_without_fallback() -> None:
    pages, urls = _all_pages()
    pages[2017] = _normal_page([3])
    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "FAIL"
    assert result.resolved_identity_count == 0
    assert result.failure_reason == "HEADER_ROW_ZERO"
    assert result.to_dict()["identity_results"][1]["status"] == "NOT_REACHED"


def test_normal_multiple_candidate_fails_closed() -> None:
    pages, urls = _all_pages()
    first = _normal_page([1]).decode()
    duplicate_header = _row([_cell("Jan.", "th")])
    pages[2017] = first.replace("</table>", duplicate_header + "</table>").encode()
    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "FAIL"
    assert result.failure_reason == "HEADER_ROW_MANY"


@pytest.mark.parametrize(
    ("variant", "expected_reason"),
    [
        ("report_many", "REPORT_ROW_MANY"),
        ("data_missing", "REPORT_DATA_COLUMN_MISSING"),
        ("href_zero", "HREF_ZERO"),
        ("href_many", "HREF_MANY"),
    ],
)
def test_normal_data_and_report_multiplicity_fail_closed(
    variant: str, expected_reason: str
) -> None:
    page = _normal_page([1]).decode()
    if variant == "report_many":
        page = page.replace(
            "</table>",
            _row([_cell(ANCHOR), _cell("duplicate")]) + "</table>",
        )
    elif variant == "data_missing":
        page = ("<table>" + _row([_cell("Jan.", "th")]) + _row([_cell(ANCHOR)]) + "</table>")
    elif variant == "href_zero":
        page = page.replace(
            '<a href="/english/markets/statistics-equities/monthly/object-1.pdf">object-1</a>',
            "object-1",
        )
    elif variant == "href_many":
        page = page.replace(
            '<a href="/english/markets/statistics-equities/monthly/object-1.pdf">object-1</a>',
            '<a href="/english/markets/statistics-equities/monthly/object-1.pdf">object-1</a>'
            '<a href="/english/markets/statistics-equities/monthly/object-1b.pdf">second</a>',
        )
    pages, urls = _all_pages()
    pages[2017] = page.encode()

    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "FAIL"
    assert result.failure_reason == expected_reason


def test_header_token_must_be_at_frozen_header_column() -> None:
    headers = [_cell("column-0"), _cell("Jan.", "th")]
    report = _row([_cell(ANCHOR), _cell("value", href="/report.pdf")])
    page = ("<table>" + _row(headers) + report + "</table>").encode()
    pages, urls = _all_pages()
    pages[2017] = page

    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "FAIL"
    assert result.failure_reason == "HEADER_ROW_ZERO"


@pytest.mark.parametrize("pre_count", [0, 2])
def test_pre_zero_or_multiple_candidates_fails_closed(pre_count: int) -> None:
    pages, urls = _all_pages()
    pages[2022] = _normal_page([3, 4, 5], pre_count=pre_count)
    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "FAIL"
    assert result.failure_identity == {
        "logical_month": "2022-04",
        "object_part": "PRE_APRIL_1_REFERENCE_OBJECT",
    }
    assert result.failure_reason == "PRE_RESOLVER_FAILURE"
    assert result.resolved_identity_count == 4


def test_wrong_month_token_and_no_alternate_fallback() -> None:
    pages, urls = _all_pages()
    pages[2017] = _normal_page([1]).replace(b"Jan.", b"January")
    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "FAIL"
    assert result.failure_reason == "HEADER_ROW_ZERO"


def test_wrong_table_and_wrong_column_remain_rejected() -> None:
    report = _row([_cell(ANCHOR), _cell("value", href="/report.pdf")])
    wrong_table = "<table>" + _row([_cell("Jan.", "th")]) + "</table>"
    page = ("<table>" + report + "</table>" + wrong_table).encode()
    result = application.apply_fixed_eight_locators(
        {2017: page, 2019: _normal_page([12]), 2020: _normal_page([1]),
         2022: _normal_page([3, 4, 5], pre_count=1), 2026: _normal_page([1])},
        {2022: YEAR_PAGE_URL_2022},
    )

    assert result.status == "FAIL"
    assert result.failure_reason == "HEADER_ROW_ZERO"


def test_wrong_tag_and_header_data_column_do_not_match() -> None:
    report = _row([_cell(ANCHOR), _cell("value", href="/report.pdf")])
    wrong_tag = _row([_cell("Jan.")])
    page = ("<table>" + wrong_tag + report + "</table>").encode()
    result = application.apply_fixed_eight_locators(
        {2017: page, 2019: _normal_page([12]), 2020: _normal_page([1]),
         2022: _normal_page([3, 4, 5], pre_count=1), 2026: _normal_page([1])},
        {2022: YEAR_PAGE_URL_2022},
    )
    assert result.status == "FAIL"
    assert result.failure_reason == "HEADER_ROW_ZERO"


def test_normal_failure_does_not_try_an_alternate_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages, urls = _all_pages()
    pages[2017] = _normal_page([1]).replace(b"Jan.", b"Jan")
    called = False

    def alternate(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("alternate resolver used")

    monkeypatch.setattr(application, "_sole_candidate_href", alternate)
    result = application.apply_fixed_eight_locators(pages, urls)

    assert result.status == "FAIL"
    assert result.failure_reason == "HEADER_ROW_ZERO"
    assert called is False


def test_safe_serialization_has_no_url_href_html_path_or_exception() -> None:
    pages, urls = _all_pages()
    result = application.apply_fixed_eight_locators(pages, urls)
    serialized = repr(result) + repr(result.to_dict())

    assert "jpx.co.jp" not in serialized
    assert "/english/" not in serialized
    assert ".pdf" not in serialized
    assert "<table>" not in serialized
    assert "Jan." not in serialized
    assert "C:\\" not in serialized
    assert "failure_identity" in serialized


def test_safe_schema_has_no_methodology_or_outcome_fields() -> None:
    pages, urls = _all_pages()
    result = application.apply_fixed_eight_locators(pages, urls).to_dict()

    assert set(result) == {
        "schema_version",
        "status",
        "resolved_identity_count",
        "identity_results",
        "failure_identity",
        "failure_reason",
    }
    assert "category" not in repr(result)
    assert "trading_dates" not in repr(result)
    assert "profitability" not in repr(result)


def test_deterministic_first_failure_and_no_filesystem_or_network_surface() -> None:
    pages, urls = _all_pages()
    pages[2017] = _normal_page([3])
    first = application.apply_fixed_eight_locators(pages, urls).to_dict()
    second = application.apply_fixed_eight_locators(pages, urls).to_dict()

    assert first == second
    source = open(application.__file__, encoding="utf-8").read()
    assert not re.search(r"\b(requests|urllib|subprocess)\b", source)
    assert "probe_calibration_bundle" not in source
    assert "resolve_source_b_normal_month_object" not in source
    assert "pdfplumber" not in source


def test_missing_page_or_pre_parent_url_is_safe_input_failure() -> None:
    pages, urls = _all_pages()
    del pages[2017]
    result = application.apply_fixed_eight_locators(pages, urls)
    assert result.status == "FAIL"
    assert result.failure_reason == "INPUT_FAILURE"

    pages, _urls = _all_pages()
    result = application.apply_fixed_eight_locators(pages, {})
    assert result.status == "FAIL"
    assert result.failure_reason == "INPUT_FAILURE"
