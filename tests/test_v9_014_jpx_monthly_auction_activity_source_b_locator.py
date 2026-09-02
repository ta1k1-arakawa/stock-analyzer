"""Synthetic tests for the V9_014 deterministic OFFLINE SOURCE_B locator.

All fixtures here are synthetic already-extracted link/semantic records.
No real JPX HTML/PDF, no network access, and no filesystem access are used
anywhere in this file.
"""

import inspect

from src import v9_014_jpx_monthly_auction_activity_source_b_locator as loc
from src.v9_014_jpx_monthly_auction_activity_authority import (
    APRIL_2022_LOGICAL_MONTH,
    NORMAL_MONTHLY_REPORT2_OBJECT,
    PRE_APRIL_1_REFERENCE_OBJECT,
    REQUIRED_LOGICAL_MONTHS,
)


VALID_YEAR_PAGE_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/2019.html"
OTHER_YEAR_PAGE_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/2020.html"
VALID_APRIL_YEAR_PAGE_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/2022.html"
VALID_MONTH_PDF_URL = "https://www.jpx.co.jp/data/reports/2019/06/trading-volume.pdf"
VALID_APRIL_PRE_PDF_URL = "https://www.jpx.co.jp/data/reports/2022/04/reference-april-1.pdf"


# ---------------------------------------------------------------------------
# Deterministic planner: 109 logical months / 110 physical requests
# ---------------------------------------------------------------------------

def test_planner_yields_exactly_109_months_110_requests():
    plan = loc.plan_source_b_requests()
    assert len({e.logical_month for e in plan}) == 109
    assert len(plan) == 110


def test_planner_first_and_last_month():
    plan = loc.plan_source_b_requests()
    first_month_entries = [e for e in plan if e.logical_month == REQUIRED_LOGICAL_MONTHS[0]]
    last_month_entries = [e for e in plan if e.logical_month == REQUIRED_LOGICAL_MONTHS[-1]]
    assert REQUIRED_LOGICAL_MONTHS[0] == "2017-01"
    assert REQUIRED_LOGICAL_MONTHS[-1] == "2026-01"
    assert [e.kind for e in first_month_entries] == [NORMAL_MONTHLY_REPORT2_OBJECT]
    assert [e.kind for e in last_month_entries] == [NORMAL_MONTHLY_REPORT2_OBJECT]


def test_planner_april_2022_split_into_two_entries():
    plan = loc.plan_source_b_requests()
    april_entries = [e for e in plan if e.logical_month == APRIL_2022_LOGICAL_MONTH]
    assert len(april_entries) == 2
    assert {e.kind for e in april_entries} == {
        PRE_APRIL_1_REFERENCE_OBJECT,
        NORMAL_MONTHLY_REPORT2_OBJECT,
    }


def test_planner_every_other_month_has_exactly_one_normal_entry():
    plan = loc.plan_source_b_requests()
    for month in REQUIRED_LOGICAL_MONTHS:
        if month == APRIL_2022_LOGICAL_MONTH:
            continue
        entries = [e for e in plan if e.logical_month == month]
        assert len(entries) == 1
        assert entries[0].kind == NORMAL_MONTHLY_REPORT2_OBJECT


# ---------------------------------------------------------------------------
# Year-page resolution
# ---------------------------------------------------------------------------

def test_year_page_normal_happy_path():
    candidates = [
        loc.RootYearCandidate(label="2018", href="https://www.jpx.co.jp/archives/2018.html"),
        loc.RootYearCandidate(label="2019", href=VALID_YEAR_PAGE_URL),
        loc.RootYearCandidate(label="2020", href="https://www.jpx.co.jp/archives/2020.html"),
    ]
    result = loc.resolve_source_b_year_page(candidates, 2019)
    assert result.status == loc.LOCATOR_OK
    assert result.url == VALID_YEAR_PAGE_URL


def test_year_page_missing_year_fails():
    candidates = [loc.RootYearCandidate(label="2018", href="https://www.jpx.co.jp/archives/2018.html")]
    result = loc.resolve_source_b_year_page(candidates, 2019)
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_year_page_duplicate_year_fails():
    candidates = [
        loc.RootYearCandidate(label="2019", href="https://www.jpx.co.jp/archives/2019-a.html"),
        loc.RootYearCandidate(label="2019", href="https://www.jpx.co.jp/archives/2019-b.html"),
    ]
    result = loc.resolve_source_b_year_page(candidates, 2019)
    assert result.status == loc.LOCATOR_MULTIPLE_CANDIDATES_FAILURE


def test_year_page_off_domain_fails():
    candidates = [loc.RootYearCandidate(label="2019", href="https://evil.example.com/2019.html")]
    result = loc.resolve_source_b_year_page(candidates, 2019)
    assert result.status == loc.LOCATOR_OFF_DOMAIN_OR_INVALID_URL_FAILURE


def test_year_page_non_https_fails():
    candidates = [loc.RootYearCandidate(label="2019", href="http://www.jpx.co.jp/archives/2019.html")]
    result = loc.resolve_source_b_year_page(candidates, 2019)
    assert result.status == loc.LOCATOR_OFF_DOMAIN_OR_INVALID_URL_FAILURE


def test_year_page_no_constructed_url_from_pattern():
    # Even though a plausible pattern-derived URL would "work", the label
    # must come only from a supplied candidate -- an empty candidate list
    # can never synthesize one.
    result = loc.resolve_source_b_year_page([], 2019)
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


# ---------------------------------------------------------------------------
# Normal monthly Report-2 object resolution, bound to the selected parent
# year-page URL (V9_014_SOURCE_B_LOCATOR_HIGH_1 remediation)
# ---------------------------------------------------------------------------

def test_normal_month_object_happy_path():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label="Report 1 \"Something Else\"", month="2019-06",
            href="https://www.jpx.co.jp/data/other.pdf",
        ),
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-05",
            href="https://www.jpx.co.jp/data/reports/2019/05/trading-volume.pdf",
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_OK
    assert result.url == VALID_MONTH_PDF_URL


def test_normal_month_object_missing_report_row_fails():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label="Report 1 \"Something Else\"", month="2019-06",
            href="https://www.jpx.co.jp/data/other.pdf",
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_normal_month_object_missing_month_column_fails():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-05", href=VALID_MONTH_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_normal_month_object_duplicate_fails():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06",
            href="https://www.jpx.co.jp/data/reports/2019/06/a.pdf",
        ),
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06",
            href="https://www.jpx.co.jp/data/reports/2019/06/b.pdf",
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_MULTIPLE_CANDIDATES_FAILURE


def test_normal_month_object_off_domain_fails():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06", href="https://evil.example.com/report.pdf",
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_OFF_DOMAIN_OR_INVALID_URL_FAILURE


def test_normal_month_object_non_https_fails():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06",
            href="http://www.jpx.co.jp/data/reports/2019/06/trading-volume.pdf",
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_OFF_DOMAIN_OR_INVALID_URL_FAILURE


def test_normal_month_object_non_pdf_fails():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06",
            href="https://www.jpx.co.jp/data/reports/2019/06/trading-volume.html",
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_NON_PDF_FAILURE


def test_normal_month_object_rejects_case_changed_report_label():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT.lower(), month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_normal_month_object_rejects_substring_report_label():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT + " (Domestic)", month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label="Stock Trading Volume & Value", month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_normal_month_object_rejects_fuzzy_report_label():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label='Report2 "Stock Trading Volume and Value"', month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_normal_month_object_rejects_month_outside_frozen_set():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2016-12", href=VALID_MONTH_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2016-12", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_INVALID_INPUT_FAILURE


# ---------------------------------------------------------------------------
# Parent-snapshot provenance binding (V9_014_SOURCE_B_LOCATOR_HIGH_1)
# ---------------------------------------------------------------------------

def test_normal_month_object_root_selects_x_child_parent_x_passes():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
    ]
    year_result = loc.resolve_source_b_year_page(
        [loc.RootYearCandidate(label="2019", href=VALID_YEAR_PAGE_URL)], 2019
    )
    assert year_result.status == loc.LOCATOR_OK
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=year_result.url
    )
    assert result.status == loc.LOCATOR_OK
    assert result.url == VALID_MONTH_PDF_URL


def test_normal_month_object_root_selects_x_child_parent_y_fails():
    # The child candidate is otherwise perfect (correct label, correct
    # month, JPX domain, PDF) but its parent snapshot is a DIFFERENT year
    # page than the one resolved from root -- it must never match.
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=OTHER_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_normal_month_object_matching_labels_cannot_bypass_parent_mismatch():
    # A mismatched-parent candidate and a correctly-parented candidate for a
    # DIFFERENT month both present; only an exact parent+label+month match
    # may resolve, and here none does.
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=OTHER_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-05",
            href="https://www.jpx.co.jp/data/reports/2019/05/trading-volume.pdf",
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_normal_month_object_valid_pdf_from_unrelated_parent_cannot_bypass_binding():
    unrelated_href = "https://www.jpx.co.jp/data/reports/2019/06/from-unrelated-parent.pdf"
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=OTHER_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06", href=unrelated_href,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status != loc.LOCATOR_OK
    assert result.url != unrelated_href


def test_normal_month_object_missing_selected_parent_url_fails():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(candidates, "2019-06", selected_year_page_url="")
    assert result.status == loc.LOCATOR_INVALID_INPUT_FAILURE


def test_normal_month_object_off_domain_selected_parent_url_fails():
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url="https://evil.example.com/2019.html",
            report_label=loc.SOURCE_B_REPORT, month="2019-06", href=VALID_MONTH_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url="https://evil.example.com/2019.html"
    )
    assert result.status == loc.LOCATOR_OFF_DOMAIN_OR_INVALID_URL_FAILURE


def test_normal_month_object_parenthood_not_inferred_from_href_year_substring():
    # The href itself contains "2019" and looks entirely plausible for the
    # requested year, but its declared parent_year_page_url points at a
    # different page -- the href content must never substitute for the
    # explicit provenance field.
    misleading_href = "https://www.jpx.co.jp/data/reports/2019/06/trading-volume.pdf"
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=OTHER_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2019-06", href=misleading_href,
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


# ---------------------------------------------------------------------------
# April-2022 two-part split: PRE special reference branch
# ---------------------------------------------------------------------------

def test_april_2022_two_distinct_objects():
    pre_candidates = [
        loc.AprilPreReferenceCandidate(
            parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
            label=loc.APRIL_1_2022_REFERENCE_LABEL, href=VALID_APRIL_PRE_PDF_URL,
        ),
    ]
    normal_candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
            report_label=loc.SOURCE_B_REPORT, month="2022-04",
            href="https://www.jpx.co.jp/data/reports/2022/04/trading-volume.pdf",
        ),
    ]
    pre_result = loc.resolve_source_b_april_pre_object(
        pre_candidates, selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL
    )
    normal_result = loc.resolve_source_b_normal_month_object(
        normal_candidates, "2022-04", selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL
    )
    assert pre_result.status == loc.LOCATOR_OK
    assert normal_result.status == loc.LOCATOR_OK
    assert pre_result.url != normal_result.url
    assert pre_result.url == VALID_APRIL_PRE_PDF_URL


def test_april_pre_missing_fails():
    result = loc.resolve_source_b_april_pre_object([], selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL)
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_april_pre_duplicate_fails():
    candidates = [
        loc.AprilPreReferenceCandidate(
            parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
            label=loc.APRIL_1_2022_REFERENCE_LABEL, href="https://www.jpx.co.jp/a.pdf",
        ),
        loc.AprilPreReferenceCandidate(
            parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
            label=loc.APRIL_1_2022_REFERENCE_LABEL, href="https://www.jpx.co.jp/b.pdf",
        ),
    ]
    result = loc.resolve_source_b_april_pre_object(
        candidates, selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_MULTIPLE_CANDIDATES_FAILURE


def test_april_pre_rejects_case_changed_label():
    candidates = [
        loc.AprilPreReferenceCandidate(
            parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
            label=loc.APRIL_1_2022_REFERENCE_LABEL.lower(), href=VALID_APRIL_PRE_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_april_pre_object(
        candidates, selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_april_pre_rejects_substring_label():
    candidates = [
        loc.AprilPreReferenceCandidate(
            parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
            label="Status on April 1, 2022", href=VALID_APRIL_PRE_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_april_pre_object(
        candidates, selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_april_pre_off_domain_fails():
    candidates = [
        loc.AprilPreReferenceCandidate(
            parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
            label=loc.APRIL_1_2022_REFERENCE_LABEL, href="https://evil.example.com/x.pdf",
        ),
    ]
    result = loc.resolve_source_b_april_pre_object(
        candidates, selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_OFF_DOMAIN_OR_INVALID_URL_FAILURE


def test_april_pre_non_pdf_fails():
    candidates = [
        loc.AprilPreReferenceCandidate(
            parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
            label=loc.APRIL_1_2022_REFERENCE_LABEL, href="https://www.jpx.co.jp/data/reference-april-1.html",
        ),
    ]
    result = loc.resolve_source_b_april_pre_object(
        candidates, selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_NON_PDF_FAILURE


def test_april_pre_root_selects_x_snapshot_parent_y_fails():
    candidates = [
        loc.AprilPreReferenceCandidate(
            parent_year_page_url=OTHER_YEAR_PAGE_URL,
            label=loc.APRIL_1_2022_REFERENCE_LABEL, href=VALID_APRIL_PRE_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_april_pre_object(
        candidates, selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_april_pre_missing_selected_parent_url_fails():
    candidates = [
        loc.AprilPreReferenceCandidate(
            parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
            label=loc.APRIL_1_2022_REFERENCE_LABEL, href=VALID_APRIL_PRE_PDF_URL,
        ),
    ]
    result = loc.resolve_source_b_april_pre_object(candidates, selected_year_page_url=None)
    assert result.status == loc.LOCATOR_INVALID_INPUT_FAILURE


# ---------------------------------------------------------------------------
# The returned URL originates only from supplied links; no guessed URL
# ---------------------------------------------------------------------------

def test_resolved_url_is_exactly_one_of_the_supplied_hrefs():
    supplied_hrefs = {
        "https://www.jpx.co.jp/data/reports/2019/06/a.pdf",
    }
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL, report_label=loc.SOURCE_B_REPORT, month="2019-06", href=href
        )
        for href in supplied_hrefs
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.status == loc.LOCATOR_OK
    assert result.url in supplied_hrefs


def test_unrelated_candidate_href_never_leaks_into_result():
    target_href = "https://www.jpx.co.jp/data/reports/2019/06/target.pdf"
    unrelated_href = "https://www.jpx.co.jp/data/reports/2019/06/UNRELATED-DO-NOT-RETURN.pdf"
    candidates = [
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL, report_label=loc.SOURCE_B_REPORT, month="2019-06", href=target_href
        ),
        loc.MonthlyReportCandidate(
            parent_year_page_url=VALID_YEAR_PAGE_URL, report_label="Report 1 other", month="2019-06", href=unrelated_href
        ),
    ]
    result = loc.resolve_source_b_normal_month_object(
        candidates, "2019-06", selected_year_page_url=VALID_YEAR_PAGE_URL
    )
    assert result.url == target_href
    assert result.url != unrelated_href


# ---------------------------------------------------------------------------
# No network/filesystem/credential access from import or any resolver call
# ---------------------------------------------------------------------------

def test_module_source_contains_no_io_capable_imports():
    source = inspect.getsource(loc)
    for forbidden in (
        "urllib.request", "import socket", "import subprocess", "import os",
        "import requests", "http.client", "open(",
    ):
        assert forbidden not in source, forbidden


def test_full_resolver_pass_performs_no_io_side_effects():
    # A complete, deterministic exercise of every public resolver plus the
    # planner; if any of these touched the network or filesystem in this
    # sandboxed offline test environment they would raise or hang rather
    # than return promptly.
    loc.plan_source_b_requests()
    loc.resolve_source_b_year_page(
        [loc.RootYearCandidate(label="2019", href=VALID_YEAR_PAGE_URL)], 2019
    )
    loc.resolve_source_b_normal_month_object(
        [
            loc.MonthlyReportCandidate(
                parent_year_page_url=VALID_YEAR_PAGE_URL,
                report_label=loc.SOURCE_B_REPORT, month="2019-06", href=VALID_MONTH_PDF_URL,
            )
        ],
        "2019-06",
        selected_year_page_url=VALID_YEAR_PAGE_URL,
    )
    loc.resolve_source_b_april_pre_object(
        [
            loc.AprilPreReferenceCandidate(
                parent_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
                label=loc.APRIL_1_2022_REFERENCE_LABEL, href=VALID_APRIL_PRE_PDF_URL,
            )
        ],
        selected_year_page_url=VALID_APRIL_YEAR_PAGE_URL,
    )


# ---------------------------------------------------------------------------
# Reuse of frozen V9_014 SOURCE_B constants; no invented methodology
# ---------------------------------------------------------------------------

def test_module_reuses_frozen_core_constants():
    assert loc.SOURCE_B_ARCHIVE_ROOT == (
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html"
    )
    assert loc.SOURCE_B_REPORT == 'Report 2 "Stock Trading Volume & Value"'
    assert loc.SOURCE_B_OBJECT_FORMAT == "PDF"
    assert loc.APRIL_1_2022_REFERENCE_LABEL == "(Reference) Status on April 1, 2022"


def test_module_never_materializes_trading_dates_or_relation_output():
    assert not hasattr(loc, "DateClassification")
    assert not hasattr(loc, "evaluate_cross_source_relation")
    assert not hasattr(loc, "trading_dates")
    assert not hasattr(loc, "materialize_trading_dates")
