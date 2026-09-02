"""Synthetic tests for the V9_014 deterministic OFFLINE SOURCE_B archive
HTML candidate extraction.

All fixtures here are synthetic HTML byte strings. No real JPX HTML/PDF, no
network access, and no filesystem access are used anywhere in this file.
"""

import pytest

from src import v9_014_jpx_monthly_auction_activity_source_b_archive_parser as parser
from src import v9_014_jpx_monthly_auction_activity_source_b_locator as loc
from src.v9_005_stage_a_jpx_probe import V9005StageABlocked


ROOT_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html"
YEAR_PAGE_URL_2019 = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/2019.html"
YEAR_PAGE_URL_2022 = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/2022.html"

REPORT2_LABEL_HTML = "Report 2 &quot;Stock Trading Volume &amp; Value&quot;"
APRIL_LABEL_HTML = "(Reference) Status on April 1, 2022"


def _month_table(rows_html: str) -> bytes:
    return f"<html><body><table>{rows_html}</table></body></html>".encode("utf-8")


# ---------------------------------------------------------------------------
# ROOT -> YEAR CANDIDATES
# ---------------------------------------------------------------------------

def test_root_valid_relative_year_href_yields_exact_absolute_candidate():
    html = b'<a href="archives/2019.html">2019</a>'
    candidates = parser.extract_root_year_candidates(html, ROOT_URL, 2019)
    assert candidates == (loc.RootYearCandidate(label="2019", href=YEAR_PAGE_URL_2019),)


def test_root_zero_year_matches_yields_empty_tuple():
    html = b'<a href="archives/2018.html">2018</a>'
    candidates = parser.extract_root_year_candidates(html, ROOT_URL, 2019)
    assert candidates == ()
    result = loc.resolve_source_b_year_page(list(candidates), 2019)
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_root_duplicate_exact_year_anchors_preserved_and_fail_downstream():
    html = (
        b'<a href="archives/2019-a.html">2019</a>'
        b'<a href="archives/2019-b.html">2019</a>'
    )
    candidates = parser.extract_root_year_candidates(html, ROOT_URL, 2019)
    assert len(candidates) == 2
    result = loc.resolve_source_b_year_page(list(candidates), 2019)
    assert result.status == loc.LOCATOR_MULTIPLE_CANDIDATES_FAILURE


def test_root_case_fuzzy_substring_year_text_rejected():
    html = (
        b'<a href="a.html"> 2019 </a>'  # extra whitespace normalizes to "2019" -- should MATCH
        b'<a href="b.html">Year 2019</a>'  # substring -- must NOT match
        b'<a href="c.html">2019a</a>'  # fuzzy -- must NOT match
    )
    candidates = parser.extract_root_year_candidates(html, ROOT_URL, 2019)
    # Only the whitespace-normalized exact "2019" anchor matches.
    assert len(candidates) == 1
    assert candidates[0].href == (
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/a.html"
    )


def test_root_matching_target_missing_href_fails():
    html = b"<a>2019</a>"
    with pytest.raises(V9005StageABlocked):
        parser.extract_root_year_candidates(html, ROOT_URL, 2019)


def test_root_off_domain_resolved_target_fails():
    html = b'<a href="https://evil.example.com/2019.html">2019</a>'
    with pytest.raises(V9005StageABlocked):
        parser.extract_root_year_candidates(html, ROOT_URL, 2019)


def test_root_non_https_resolved_target_fails():
    html = b'<a href="http://www.jpx.co.jp/archives/2019.html">2019</a>'
    with pytest.raises(V9005StageABlocked):
        parser.extract_root_year_candidates(html, ROOT_URL, 2019)


def test_root_malformed_html_fails():
    html = b"<table><tr><td>2019</td>"  # unclosed table/tr/td
    with pytest.raises(V9005StageABlocked):
        parser.extract_root_year_candidates(html, ROOT_URL, 2019)


def test_root_no_guessed_url_when_candidate_absent():
    # An empty page never synthesizes a plausible archive URL.
    candidates = parser.extract_root_year_candidates(b"<html></html>", ROOT_URL, 2019)
    assert candidates == ()


def test_root_requires_bytes_input():
    with pytest.raises(V9005StageABlocked):
        parser.extract_root_year_candidates("<a href='x'>2019</a>", ROOT_URL, 2019)


# ---------------------------------------------------------------------------
# NORMAL REPORT-2 CANDIDATES
# ---------------------------------------------------------------------------

def test_normal_month_valid_table_row_column_yields_candidate_and_locator_pass():
    html = _month_table(
        f"<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="r2.pdf">x</a></td></tr>'
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert len(candidates) == 1
    result = loc.resolve_source_b_normal_month_object(
        list(candidates), "2019-06", selected_year_page_url=YEAR_PAGE_URL_2019
    )
    assert result.status == loc.LOCATOR_OK
    assert result.url == "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/r2.pdf"


def test_normal_month_wrong_report_label_yields_no_candidate():
    html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        '<tr><td>Report 1 Something Else</td><td><a href="r1.pdf">x</a></td></tr>'
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert candidates == ()


def test_normal_month_wrong_month_yields_no_candidate():
    html = _month_table(
        "<tr><th></th><th>2019-05</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="r2.pdf">x</a></td></tr>'
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert candidates == ()


def test_normal_month_report_row_and_month_header_in_different_tables_do_not_cross_bind():
    html = (
        b"<html><body>"
        b"<table><tr><th></th><th>2019-06</th></tr></table>"
        + f'<table><tr><td>{REPORT2_LABEL_HTML}</td><td><a href="r2.pdf">x</a></td></tr></table>'.encode("utf-8")
        + b"</body></html>"
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert candidates == ()


def test_normal_month_duplicate_report_rows_preserved_no_silent_choice():
    html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="a.pdf">x</a></td></tr>'
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="b.pdf">y</a></td></tr>'
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert len(candidates) == 2
    result = loc.resolve_source_b_normal_month_object(
        list(candidates), "2019-06", selected_year_page_url=YEAR_PAGE_URL_2019
    )
    assert result.status == loc.LOCATOR_MULTIPLE_CANDIDATES_FAILURE


def test_normal_month_duplicate_month_headers_preserved_no_silent_choice():
    html = _month_table(
        "<tr><th></th><th>2019-06</th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="a.pdf">x</a></td><td><a href="b.pdf">y</a></td></tr>'
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert len(candidates) == 2
    result = loc.resolve_source_b_normal_month_object(
        list(candidates), "2019-06", selected_year_page_url=YEAR_PAGE_URL_2019
    )
    assert result.status == loc.LOCATOR_MULTIPLE_CANDIDATES_FAILURE


def test_normal_month_multiple_hrefs_in_one_cell_preserved_no_silent_choice():
    html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="a.pdf">x</a><a href="b.pdf">y</a></td></tr>'
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert len(candidates) == 2
    result = loc.resolve_source_b_normal_month_object(
        list(candidates), "2019-06", selected_year_page_url=YEAR_PAGE_URL_2019
    )
    assert result.status == loc.LOCATOR_MULTIPLE_CANDIDATES_FAILURE


def test_normal_month_zero_hrefs_in_matched_cell_yields_no_candidate():
    html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f"<tr><td>{REPORT2_LABEL_HTML}</td><td></td></tr>"
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert candidates == ()
    result = loc.resolve_source_b_normal_month_object(
        list(candidates), "2019-06", selected_year_page_url=YEAR_PAGE_URL_2019
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_normal_month_parent_year_page_url_exactly_preserved():
    html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="r2.pdf">x</a></td></tr>'
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert candidates[0].parent_year_page_url == YEAR_PAGE_URL_2019


def test_normal_month_requested_year_context_mismatch_fails():
    html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="r2.pdf">x</a></td></tr>'
    )
    with pytest.raises(V9005StageABlocked):
        parser.extract_normal_month_candidates(
            html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2020
        )


def test_normal_month_requested_month_outside_frozen_set_fails():
    html = _month_table("<tr><th></th></tr>")
    with pytest.raises(V9005StageABlocked):
        parser.extract_normal_month_candidates(
            html, YEAR_PAGE_URL_2019, "2016-12", selected_year=2016
        )


def test_normal_month_off_domain_child_fails():
    html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="https://evil.example.com/r2.pdf">x</a></td></tr>'
    )
    with pytest.raises(V9005StageABlocked):
        parser.extract_normal_month_candidates(
            html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
        )


def test_normal_month_non_https_child_fails():
    html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="http://www.jpx.co.jp/r2.pdf">x</a></td></tr>'
    )
    with pytest.raises(V9005StageABlocked):
        parser.extract_normal_month_candidates(
            html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
        )


def test_normal_month_child_url_never_inferred_from_pattern():
    # Two structurally distinct hrefs, neither containing an obvious
    # "2019-06"-shaped filename -- the resolver must still return exactly
    # the actual supplied href, never a constructed guess.
    html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="opaque-object-id-4471.pdf">x</a></td></tr>'
    )
    candidates = parser.extract_normal_month_candidates(
        html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
    )
    assert candidates[0].href == (
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/opaque-object-id-4471.pdf"
    )


def test_normal_month_malformed_html_fails():
    html = b"<table><tr><td>x</td>"
    with pytest.raises(V9005StageABlocked):
        parser.extract_normal_month_candidates(
            html, YEAR_PAGE_URL_2019, "2019-06", selected_year=2019
        )


# ---------------------------------------------------------------------------
# APRIL PRE CANDIDATES
# ---------------------------------------------------------------------------

def test_april_pre_exact_reference_anchor_yields_candidate_and_locator_pass():
    html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022)
    assert len(candidates) == 1
    result = loc.resolve_source_b_april_pre_object(
        list(candidates), selected_year_page_url=YEAR_PAGE_URL_2022
    )
    assert result.status == loc.LOCATOR_OK
    assert result.url == (
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/reference-april-1.pdf"
    )


def test_april_pre_missing_reference_fails_downstream():
    html = b"<html></html>"
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022)
    assert candidates == ()
    result = loc.resolve_source_b_april_pre_object(
        list(candidates), selected_year_page_url=YEAR_PAGE_URL_2022
    )
    assert result.status == loc.LOCATOR_ZERO_CANDIDATES_FAILURE


def test_april_pre_duplicate_exact_reference_fails_downstream():
    html = (
        f'<a href="a.pdf">{APRIL_LABEL_HTML}</a>'
        f'<a href="b.pdf">{APRIL_LABEL_HTML}</a>'
    ).encode("utf-8")
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022)
    assert len(candidates) == 2
    result = loc.resolve_source_b_april_pre_object(
        list(candidates), selected_year_page_url=YEAR_PAGE_URL_2022
    )
    assert result.status == loc.LOCATOR_MULTIPLE_CANDIDATES_FAILURE


def test_april_pre_case_substring_fuzzy_rejected():
    html = (
        b'<a href="a.pdf">(reference) status on april 1, 2022</a>'
        b'<a href="b.pdf">Status on April 1, 2022</a>'
        b'<a href="c.pdf">(Reference) Status on April 1 2022</a>'
    )
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022)
    assert candidates == ()


def test_april_pre_relative_url_resolution():
    html = f'<a href="ref/april1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022)
    assert candidates[0].href == (
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/ref/april1.pdf"
    )


def test_april_pre_off_domain_parent_fails():
    html = f'<a href="a.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(html, "https://evil.example.com/2022.html")


def test_april_pre_explicit_parent_provenance_preserved():
    html = f'<a href="a.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022)
    assert candidates[0].parent_year_page_url == YEAR_PAGE_URL_2022


def test_april_pre_malformed_html_fails():
    html = b"<table><tr><td>x</td>"
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022)


# ---------------------------------------------------------------------------
# INTEGRATION: synthetic HTML -> extraction -> existing locator resolvers
# ---------------------------------------------------------------------------

def test_end_to_end_root_to_normal_month_locator_ok():
    root_html = b'<a href="archives/2019.html">2019</a>'
    root_candidates = parser.extract_root_year_candidates(root_html, ROOT_URL, 2019)
    year_result = loc.resolve_source_b_year_page(list(root_candidates), 2019)
    assert year_result.status == loc.LOCATOR_OK

    year_html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="r2.pdf">x</a></td></tr>'
    )
    month_candidates = parser.extract_normal_month_candidates(
        year_html, year_result.url, "2019-06", selected_year=2019
    )
    month_result = loc.resolve_source_b_normal_month_object(
        list(month_candidates), "2019-06", selected_year_page_url=year_result.url
    )
    assert month_result.status == loc.LOCATOR_OK
    assert month_result.url == (
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/r2.pdf"
    )


def test_end_to_end_root_to_april_pre_locator_ok():
    root_html = b'<a href="archives/2022.html">2022</a>'
    root_candidates = parser.extract_root_year_candidates(root_html, ROOT_URL, 2022)
    year_result = loc.resolve_source_b_year_page(list(root_candidates), 2022)
    assert year_result.status == loc.LOCATOR_OK

    april_html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    april_candidates = parser.extract_april_pre_candidates(april_html, year_result.url)
    april_result = loc.resolve_source_b_april_pre_object(
        list(april_candidates), selected_year_page_url=year_result.url
    )
    assert april_result.status == loc.LOCATOR_OK


def test_end_to_end_invalid_traversal_never_yields_locator_ok():
    # A normal-month candidate extracted from an UNRELATED year page (2020)
    # can never resolve against a DIFFERENT selected year page (2019),
    # even though it is otherwise a perfectly well-formed candidate.
    other_year_page_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/2020.html"
    other_year_html = _month_table(
        "<tr><th></th><th>2019-06</th></tr>"
        f'<tr><td>{REPORT2_LABEL_HTML}</td><td><a href="r2.pdf">x</a></td></tr>'
    )
    stray_candidates = parser.extract_normal_month_candidates(
        other_year_html, other_year_page_url, "2019-06", selected_year=2019
    )
    result = loc.resolve_source_b_normal_month_object(
        list(stray_candidates), "2019-06", selected_year_page_url=YEAR_PAGE_URL_2019
    )
    assert result.status != loc.LOCATOR_OK
