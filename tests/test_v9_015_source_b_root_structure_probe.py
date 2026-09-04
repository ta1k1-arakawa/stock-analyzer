import hashlib

import pytest

from src.v9_015_source_b_root_structure_probe import probe_root_structure


YEARS = ("2017", "2019", "2020", "2022", "2026")


def _anchor_document(years=YEARS, *, href="href-{year}-SYNTHETIC-SECRET"):
    body = "".join(f'<a href="{href.format(year=year)}">{year}</a>' for year in years)
    return f"<html><body>{body}</body></html>".encode()


def _option_document(years=YEARS, *, value="value-{year}-SYNTHETIC-SECRET"):
    body = "".join(f'<option value="{value.format(year=year)}">{year}</option>' for year in years)
    return f"<html><body>{body}</body></html>".encode()


def _assert_common_result(result, raw):
    assert result["root_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["root_byte_count"] == len(raw)
    assert result["schema_version"] == "V9_015_ROOT_STRUCTURE_CALIBRATION_V1"
    assert tuple(result["required_year_anchor_token_counts"]) == YEARS
    assert tuple(result["required_year_option_token_counts"]) == YEARS
    assert tuple(result["required_year_visible_token_counts"]) == YEARS


def test_all_years_unique_via_anchor_are_deterministically_bindable():
    raw = _anchor_document()
    result = probe_root_structure(raw)
    _assert_common_result(result, raw)
    assert result["safe_calibration_status"] == "PASS"
    assert result["deterministic_candidate_category"] == "ANCHOR_HREF"
    assert result["all_required_years_deterministically_bindable"] is True
    assert result["anchor_category_complete_unique"] is True
    assert result["option_category_complete_unique"] is False
    assert result["other_candidate_category_all_zero"] is True
    assert all(value == 1 for value in result["required_year_anchor_token_counts"].values())
    assert all(value == 1 for value in result["required_year_anchor_nonempty_href_counts"].values())
    assert all(value == "ONE" for value in result["required_year_anchor_multiplicity"].values())


def test_all_years_unique_via_option_are_deterministically_bindable():
    result = probe_root_structure(_option_document())
    assert result["deterministic_candidate_category"] == "OPTION_VALUE"
    assert result["option_category_complete_unique"] is True
    assert result["anchor_category_complete_unique"] is False
    assert result["other_candidate_category_all_zero"] is True
    assert all(value == "ONE" for value in result["required_year_option_multiplicity"].values())


def test_missing_year_is_valid_html_but_not_bindable():
    result = probe_root_structure(_anchor_document(YEARS[:-1]))
    assert result["html_parser_success"] is True
    assert result["safe_calibration_status"] == "PASS"
    assert result["required_year_anchor_multiplicity"]["2026"] == "ZERO"
    assert result["deterministic_candidate_category"] is None
    assert result["all_required_years_deterministically_bindable"] is False


def test_duplicate_candidate_is_many_and_not_bindable():
    raw = b'<a href="one">2017</a><a href="two">2017</a>' + _anchor_document(YEARS[1:])
    result = probe_root_structure(raw)
    assert result["html_parser_success"] is True
    assert result["required_year_anchor_multiplicity"]["2017"] == "MANY"
    assert result["deterministic_candidate_category"] is None


def test_mixed_categories_are_not_bindable():
    anchors = "".join(f'<a href="a-{year}">{year}</a>' for year in YEARS[:-1])
    raw = f'{anchors}<option value="option-2026">2026</option>'.encode()
    result = probe_root_structure(raw)
    assert result["safe_calibration_status"] == "PASS"
    assert result["deterministic_candidate_category"] is None
    assert result["other_candidate_category_all_zero"] is False


def test_both_categories_complete_are_not_bindable():
    raw = _anchor_document() + _option_document()
    result = probe_root_structure(raw)
    assert result["anchor_category_complete_unique"] is True
    assert result["option_category_complete_unique"] is True
    assert result["deterministic_candidate_category"] is None


def test_visible_text_only_has_zero_candidate_authority():
    result = probe_root_structure(b"<div>2026</div>")
    assert result["required_year_visible_token_counts"]["2026"] == 1
    assert result["required_year_anchor_token_counts"]["2026"] == 0
    assert result["required_year_option_token_counts"]["2026"] == 0
    assert result["deterministic_candidate_category"] is None


@pytest.mark.parametrize(
    "raw, field",
    [
        (b'<a href="">2017</a>', "required_year_anchor_nonempty_href_counts"),
        (b"<a>2017</a>", "required_year_anchor_nonempty_href_counts"),
        (b'<option value="">2017</option>', "required_year_option_nonempty_value_counts"),
        (b"<option>2017</option>", "required_year_option_nonempty_value_counts"),
    ],
)
def test_empty_or_missing_attribute_is_not_eligible(raw, field):
    result = probe_root_structure(raw)
    assert result["html_parser_success"] is True
    assert result[field]["2017"] == 0
    assert result["deterministic_candidate_category"] is None


def test_near_or_nonexact_labels_are_rejected():
    raw = b'<a href="x">2017x</a><a href="y">2020-01</a><a href="z">2019 00</a>'
    result = probe_root_structure(raw)
    assert result["required_year_anchor_token_counts"] == {year: 0 for year in YEARS}
    assert result["deterministic_candidate_category"] is None


def test_label_whitespace_normalization_and_nested_ordinary_tag_text_are_exact():
    raw = b'<a href="synthetic-secret"><span> 20</span>17 </a>'
    result = probe_root_structure(raw)
    assert result["required_year_anchor_token_counts"]["2017"] == 1
    assert result["required_year_anchor_nonempty_href_counts"]["2017"] == 1


@pytest.mark.parametrize(
    "raw",
    [
        b'<a href="outer">2017<a href="inner">2020</a></a>',
        b"<div></a>",
        b'<a href="unclosed">2017',
        b'<a href="one" href="two">2017</a>',
        b'<option value="one" value="two">2017</option>',
    ],
)
def test_malformed_relevant_candidate_structure_fails_closed(raw):
    result = probe_root_structure(raw)
    assert result["html_parser_success"] is False
    assert result["safe_calibration_status"] == "FAIL_TERMINAL"
    assert result["structure_failure_class"] == "DATA_QUALITY_FAILURE"


def test_invalid_utf8_is_data_quality_failure_without_leakage():
    result = probe_root_structure(b"prefix-\xff-secret")
    assert result["structure_failure_class"] == "DATA_QUALITY_FAILURE"
    assert result["safe_calibration_status"] == "FAIL_TERMINAL"
    assert "prefix" not in str(result)
    assert "secret" not in str(result)


def test_non_bytes_is_implementation_failure_without_leakage():
    result = probe_root_structure("<a href='secret'>2017</a>")
    assert result["structure_failure_class"] == "IMPLEMENTATION_FAILURE"
    assert result["root_sha256"] is None
    assert "secret" not in str(result)


def test_script_and_style_data_are_excluded_from_visible_text():
    raw = b"<script>2017</script><style>2020</style><div>2022</div>"
    result = probe_root_structure(raw)
    assert result["required_year_visible_token_counts"]["2017"] == 0
    assert result["required_year_visible_token_counts"]["2020"] == 0
    assert result["required_year_visible_token_counts"]["2022"] == 1


def test_script_text_inside_anchor_is_not_candidate_label_text():
    result = probe_root_structure(b'<a href="secret"><script>2017</script></a>')
    assert result["required_year_anchor_token_counts"]["2017"] == 0
    assert result["required_year_anchor_multiplicity"]["2017"] == "ZERO"
    assert result["all_required_years_deterministically_bindable"] is False


def test_style_text_inside_anchor_is_not_candidate_label_text():
    result = probe_root_structure(b'<a href="secret"><style>2020</style></a>')
    assert result["required_year_anchor_token_counts"]["2020"] == 0
    assert result["required_year_anchor_multiplicity"]["2020"] == "ZERO"


def test_visible_text_after_script_still_contributes_to_candidate_label():
    result = probe_root_structure(b'<a href="secret"><script>noise</script>2019</a>')
    assert result["required_year_anchor_token_counts"]["2019"] == 1
    assert result["required_year_anchor_nonempty_href_counts"]["2019"] == 1
    assert result["required_year_anchor_multiplicity"]["2019"] == "ONE"


def test_option_candidate_also_excludes_script_and_style_text():
    result = probe_root_structure(
        b'<option value="secret"><script>2017</script><style>2020</style>2022</option>'
    )
    assert result["required_year_option_token_counts"]["2017"] == 0
    assert result["required_year_option_token_counts"]["2020"] == 0
    assert result["required_year_option_token_counts"]["2022"] == 1


def test_safe_output_contains_no_raw_candidate_attributes_html_or_arbitrary_text():
    raw = b'<a href="https://synthetic.invalid/SECRET-HREF">2017 SECRET-LABEL</a>'
    result = probe_root_structure(raw)
    rendered = repr(result)
    assert "SECRET-HREF" not in rendered
    assert "synthetic.invalid" not in rendered
    assert "SECRET-LABEL" not in rendered
    assert "<a" not in rendered


def test_repeated_calls_are_deterministically_equal():
    raw = _anchor_document()
    assert probe_root_structure(raw) == probe_root_structure(raw)
