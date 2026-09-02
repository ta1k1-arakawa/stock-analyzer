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
# A different, but genuinely valid HTTPS same-domain JPX page -- must never
# be accepted as a substitute for the exact frozen root.
FAKE_SAME_DOMAIN_ROOT = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/other.html"
# Differs from ROOT_URL only by a trailing empty query string.
ROOT_URL_DIFFERING_ONLY_BY_TRAILING_QUERY = ROOT_URL + "?"
YEAR_PAGE_URL_2019 = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/2019.html"
YEAR_PAGE_URL_2022 = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/2022.html"

REPORT2_LABEL_HTML = "Report 2 &quot;Stock Trading Volume &amp; Value&quot;"
APRIL_LABEL_HTML = "(Reference) Status on April 1, 2022"


def _month_table(rows_html: str) -> bytes:
    return f"<html><body><table>{rows_html}</table></body></html>".encode("utf-8")


# ---------------------------------------------------------------------------
# ROOT -> YEAR CANDIDATES, bound to the exact frozen SOURCE_B_ARCHIVE_ROOT
# ---------------------------------------------------------------------------

def test_root_exact_frozen_root_valid_year_anchor_passes():
    html = b'<a href="archives/2019.html">2019</a>'
    candidates = parser.extract_root_year_candidates(html, parser.SOURCE_B_ARCHIVE_ROOT, 2019)
    assert candidates == (loc.RootYearCandidate(label="2019", href=YEAR_PAGE_URL_2019),)


def test_root_valid_relative_year_href_yields_exact_absolute_candidate():
    html = b'<a href="archives/2019.html">2019</a>'
    candidates = parser.extract_root_year_candidates(html, ROOT_URL, 2019)
    assert candidates == (loc.RootYearCandidate(label="2019", href=YEAR_PAGE_URL_2019),)


def test_root_another_valid_same_domain_jpx_page_fails():
    # FAKE_SAME_DOMAIN_ROOT is genuinely https and genuinely jpx.co.jp --
    # it would pass validate_jpx_url's domain policy -- but it is not the
    # exact frozen SOURCE_B_ARCHIVE_ROOT, so it must still fail closed.
    html = b'<a href="archives/2019.html">2019</a>'
    with pytest.raises(V9005StageABlocked):
        parser.extract_root_year_candidates(html, FAKE_SAME_DOMAIN_ROOT, 2019)


def test_root_differing_only_by_trailing_query_fails():
    html = b'<a href="archives/2019.html">2019</a>'
    with pytest.raises(V9005StageABlocked):
        parser.extract_root_year_candidates(html, ROOT_URL_DIFFERING_ONLY_BY_TRAILING_QUERY, 2019)


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
# APRIL PRE CANDIDATES, bound to the exact 2022 selected-year context
# ---------------------------------------------------------------------------

def test_april_pre_selected_year_2022_exact_reference_yields_candidate_and_locator_pass():
    html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    candidates = parser.extract_april_pre_candidates(
        html, YEAR_PAGE_URL_2022, selected_year=2022
    )
    assert len(candidates) == 1
    result = loc.resolve_source_b_april_pre_object(
        list(candidates), selected_year_page_url=YEAR_PAGE_URL_2022
    )
    assert result.status == loc.LOCATOR_OK
    assert result.url == (
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/reference-april-1.pdf"
    )


def test_april_pre_selected_year_2019_with_same_anchor_and_valid_parent_fails():
    # Otherwise perfect: exact reference anchor, valid JPX parent URL --
    # only the declared year context is wrong, and that alone must fail.
    html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year=2019)


@pytest.mark.parametrize("wrong_year", [2020, 2023])
def test_april_pre_representative_wrong_years_fail(wrong_year):
    html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year=wrong_year)


def test_april_pre_selected_year_bool_true_fails():
    html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year=True)


def test_april_pre_selected_year_non_int_fails():
    html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year="2022")


def test_april_pre_missing_selected_year_keyword_fails():
    html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    with pytest.raises(TypeError):
        parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022)


def test_april_pre_year_never_inferred_from_url_path_containing_2022():
    # The URL literally contains "2022" in its path, but the declared
    # selected_year is the only authority -- a wrong declared year must
    # still fail even though the URL "looks like" 2022.
    misleading_but_wrong_declared_year_url = YEAR_PAGE_URL_2022
    html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(
            html, misleading_but_wrong_declared_year_url, selected_year=2019
        )


def test_april_pre_missing_reference_fails_downstream():
    html = b"<html></html>"
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year=2022)
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
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year=2022)
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
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year=2022)
    assert candidates == ()


def test_april_pre_relative_url_resolution():
    html = f'<a href="ref/april1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year=2022)
    assert candidates[0].href == (
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archives/ref/april1.pdf"
    )


def test_april_pre_off_domain_parent_fails():
    html = f'<a href="a.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(
            html, "https://evil.example.com/2022.html", selected_year=2022
        )


def test_april_pre_explicit_parent_provenance_preserved():
    html = f'<a href="a.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    candidates = parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year=2022)
    assert candidates[0].parent_year_page_url == YEAR_PAGE_URL_2022


def test_april_pre_malformed_html_fails():
    html = b"<table><tr><td>x</td>"
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(html, YEAR_PAGE_URL_2022, selected_year=2022)


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


def test_end_to_end_exact_root_to_2022_to_april_pre_locator_ok():
    root_html = b'<a href="archives/2022.html">2022</a>'
    root_candidates = parser.extract_root_year_candidates(root_html, parser.SOURCE_B_ARCHIVE_ROOT, 2022)
    year_result = loc.resolve_source_b_year_page(list(root_candidates), 2022)
    assert year_result.status == loc.LOCATOR_OK

    april_html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    april_candidates = parser.extract_april_pre_candidates(
        april_html, year_result.url, selected_year=2022
    )
    april_result = loc.resolve_source_b_april_pre_object(
        list(april_candidates), selected_year_page_url=year_result.url
    )
    assert april_result.status == loc.LOCATOR_OK


def test_end_to_end_same_domain_fake_root_cannot_reach_locator_ok():
    root_html = b'<a href="archives/2022.html">2022</a>'
    with pytest.raises(V9005StageABlocked):
        parser.extract_root_year_candidates(root_html, FAKE_SAME_DOMAIN_ROOT, 2022)


def test_end_to_end_exact_root_wrong_selected_year_cannot_reach_locator_ok():
    root_html = b'<a href="archives/2019.html">2019</a>'
    root_candidates = parser.extract_root_year_candidates(root_html, ROOT_URL, 2019)
    year_result = loc.resolve_source_b_year_page(list(root_candidates), 2019)
    assert year_result.status == loc.LOCATOR_OK

    # A synthetic, otherwise exact April reference label found on the 2019
    # year page -- declaring selected_year=2019 (correctly matching the
    # actual traversal context) must still be rejected, since only 2022 is
    # ever a valid April-PRE year context.
    april_html = f'<a href="reference-april-1.pdf">{APRIL_LABEL_HTML}</a>'.encode("utf-8")
    with pytest.raises(V9005StageABlocked):
        parser.extract_april_pre_candidates(april_html, year_result.url, selected_year=2019)


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
