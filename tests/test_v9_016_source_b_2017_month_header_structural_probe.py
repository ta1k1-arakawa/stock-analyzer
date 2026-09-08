import json

import pytest

from src.v9_016_source_b_2017_month_header_structural_probe import (
    CATEGORY_NAMES,
    SCHEMA_VERSION,
    SOURCE_B_REPORT,
    probe_2017_month_header_structure,
)


def _html_table(header_tokens, report_rows=None, *, extra_tables=()) -> bytes:
    rows = [
        "<tr>"
        + "".join(f"<th>{token}</th>" for token in header_tokens)
        + "</tr>"
    ]
    for values in report_rows or []:
        rows.append(
            "<tr><td>"
            + SOURCE_B_REPORT
            + "</td>"
            + "".join(
                (
                    f'<td><a href="{href}">payload</a></td>'
                    if hrefs
                    else "<td></td>"
                )
                for hrefs in values
                for href in [hrefs[0]]
            )
            + "</tr>"
        )
    table = "<table>" + "".join(rows) + "</table>"
    return (table + "".join(extra_tables)).encode("utf-8")


def _report_row(*hrefs):
    return [[href] for href in hrefs]


def _single_token_fixture(token: str) -> bytes:
    return (
        "<table><tr><th>Other</th><th>"
        + token
        + "</th></tr><tr><td>"
        + SOURCE_B_REPORT
        + '</td><td><a href="synthetic-payload">payload</a></td></tr></table>'
    ).encode("utf-8")


@pytest.mark.parametrize(
    ("category", "token"),
    [
        ("LOGICAL_YYYY_MM", "2017-01"),
        ("EN_MONTH_ABBR_DOT", "Jan."),
        ("EN_MONTH_ABBR", "Jan"),
        ("EN_MONTH_FULL", "January"),
        ("NUMERIC_MONTH", "1"),
        ("NUMERIC_MONTH_ZERO_PADDED", "01"),
    ],
)
def test_each_frozen_category_is_reported_without_selection(category, token):
    result = probe_2017_month_header_structure(_single_token_fixture(token))

    assert result.status == "PASS"
    assert result.parser_success is True
    assert result.categories[category].th_count == 1
    assert result.categories[category].same_table_th_count == 1
    assert result.categories[category].intersection_pair_count == 1
    assert result.categories[category].in_bounds_intersection_count == 1
    assert result.categories[category].out_of_bounds_intersection_count == 0
    assert result.categories[category].intersection_href_count == 1
    assert result.categories[category].intersection_href_multiplicity == "ONE"


def test_logical_category_is_legacy_structural_count():
    result = probe_2017_month_header_structure(_single_token_fixture("2017-01"))

    assert result.legacy_candidate_count == 1


def test_exact_schema_and_multiple_categories_are_reported_not_selected():
    result = probe_2017_month_header_structure(
        b'<table><tr><th>Other</th><th>2017-01</th><th>Jan.</th><th>Jan</th>'
        b"</tr><tr><td>Stock Trading Volume &amp; Value</td>"
        b'<td><a href="one">one</a></td><td><a href="two">two</a></td>'
        b'<td><a href="three">three</a></td></tr></table>'
    )
    safe = result.to_dict()

    assert set(safe) == {
        "schema_version",
        "status",
        "failure_class",
        "parser_success",
        "table_count",
        "report_cell_count",
        "report_row_count",
        "categories",
        "legacy_candidate_count",
    }
    assert safe["schema_version"] == SCHEMA_VERSION
    assert set(safe["categories"]) == set(CATEGORY_NAMES)
    assert all(
        set(category) == {
            "th_count",
            "same_table_th_count",
            "intersection_pair_count",
            "in_bounds_intersection_count",
            "out_of_bounds_intersection_count",
            "intersection_href_count",
            "intersection_href_multiplicity",
        }
        for category in safe["categories"].values()
    )
    assert safe["categories"]["LOGICAL_YYYY_MM"]["intersection_href_count"] == 1
    assert safe["categories"]["EN_MONTH_ABBR_DOT"]["intersection_href_count"] == 1
    assert safe["categories"]["EN_MONTH_ABBR"]["intersection_href_count"] == 1


def test_report_label_absent():
    result = probe_2017_month_header_structure(
        b'<table><tr><th>2017-01</th></tr><tr><td>Other label</td></tr></table>'
    )

    assert result.status == "PASS"
    assert result.report_cell_count == 0
    assert result.report_row_count == 0
    assert result.categories["LOGICAL_YYYY_MM"].intersection_pair_count == 0


def test_duplicate_report_rows_are_counted_independently():
    result = probe_2017_month_header_structure(
        _html_table(["Unused", "2017-01"], [_report_row("a"), _report_row("b")])
    )

    logical = result.categories["LOGICAL_YYYY_MM"]
    assert result.report_cell_count == 2
    assert result.report_row_count == 2
    assert logical.intersection_pair_count == 2
    assert logical.intersection_href_count == 2
    assert logical.intersection_href_multiplicity == "MANY"


def test_duplicate_matching_th_columns_are_counted_as_pairs():
    result = probe_2017_month_header_structure(
        _html_table(["Unused", "2017-01", "2017-01"], [_report_row("a")])
    )

    logical = result.categories["LOGICAL_YYYY_MM"]
    assert logical.th_count == 2
    assert logical.same_table_th_count == 2
    assert logical.intersection_pair_count == 2
    assert logical.in_bounds_intersection_count == 1
    assert logical.out_of_bounds_intersection_count == 1
    assert logical.intersection_href_count == 1


def test_th_in_different_table_does_not_intersect():
    result = probe_2017_month_header_structure(
        (
            b"<table><tr><td>"
            + SOURCE_B_REPORT.encode()
            + b"</td></tr></table>"
            b"<table><tr><th>2017-01</th></tr></table>"
        )
    )

    logical = result.categories["LOGICAL_YYYY_MM"]
    assert result.report_row_count == 1
    assert logical.th_count == 1
    assert logical.same_table_th_count == 0
    assert logical.intersection_pair_count == 0


def test_out_of_bounds_intersection_is_reported_without_href_access():
    result = probe_2017_month_header_structure(
        b'<table><tr><th>Unused</th><th>Unused</th><th>2017-01</th></tr>'
        b'<tr><td>Stock Trading Volume &amp; Value</td></tr></table>'
    )

    logical = result.categories["LOGICAL_YYYY_MM"]
    assert logical.intersection_pair_count == 1
    assert logical.in_bounds_intersection_count == 0
    assert logical.out_of_bounds_intersection_count == 1
    assert logical.intersection_href_count == 0
    assert logical.intersection_href_multiplicity == "ZERO"


def test_zero_href_at_matching_intersection():
    result = probe_2017_month_header_structure(
        b'<table><tr><th>2017-01</th></tr><tr><td>'
        b"Stock Trading Volume &amp; Value</td></tr></table>"
    )

    logical = result.categories["LOGICAL_YYYY_MM"]
    assert logical.in_bounds_intersection_count == 1
    assert logical.intersection_href_count == 0
    assert logical.intersection_href_multiplicity == "ZERO"


def test_multiple_hrefs_at_matching_intersection_are_many():
    result = probe_2017_month_header_structure(
        b'<table><tr><th>Unused</th><th>2017-01</th></tr><tr><td>'
        b'Stock Trading Volume &amp; Value</td><td><a href="a">a</a>'
        b'<a href="b">b</a></td></tr></table>'
    )

    logical = result.categories["LOGICAL_YYYY_MM"]
    assert logical.intersection_href_count == 2
    assert logical.intersection_href_multiplicity == "MANY"


def test_inherited_whitespace_normalization_is_used_for_exact_matching():
    result = probe_2017_month_header_structure(
        b'<table><tr><th>  2017-01\n</th></tr><tr><td>'
        b" Stock   Trading Volume &amp; Value </td></tr></table>"
    )

    logical = result.categories["LOGICAL_YYYY_MM"]
    assert result.report_cell_count == 1
    assert logical.th_count == 1
    assert logical.intersection_pair_count == 1


def test_near_month_and_report_strings_do_not_match():
    result = probe_2017_month_header_structure(
        b'<table><tr><th>2017/01</th><th>2017-1</th><th>Jan,</th></tr>'
        b'<tr><td>Stock Trading Volume &amp; Value Extra</td></tr></table>'
    )

    assert result.report_cell_count == 0
    assert result.report_row_count == 0
    assert all(category.th_count == 0 for category in result.categories.values())


def test_non_bytes_input_fails_closed():
    result = probe_2017_month_header_structure("<table></table>")

    assert result.status == "FAIL"
    assert result.failure_class == "IMPLEMENTATION_FAILURE"
    assert result.parser_success is False
    assert set(result.categories) == set(CATEGORY_NAMES)


def test_malformed_relevant_html_fails_closed_through_inherited_parser():
    result = probe_2017_month_header_structure(
        b"<table><tr><th>2017-01</th></tr><tr><td>"
        b"Stock Trading Volume &amp; Value"
    )

    assert result.status == "FAIL"
    assert result.failure_class == "IMPLEMENTATION_FAILURE"
    assert result.parser_success is False


def test_safe_serialization_contains_no_fixture_payload_or_raw_observation():
    html = (
        b'<table><tr><th>2017-01</th></tr><tr><td>'
        b'Stock Trading Volume &amp; Value<a href="https://private.example/raw.pdf">'
        b'RAW_HTML_TEXT</a></td></tr></table>'
    )
    result = probe_2017_month_header_structure(html)
    serialized = json.dumps(result.to_dict(), sort_keys=True)

    assert "private.example" not in serialized
    assert "raw.pdf" not in serialized
    assert "RAW_HTML_TEXT" not in serialized
    assert "<table>" not in serialized
    assert "Stock Trading Volume & Value" not in serialized
