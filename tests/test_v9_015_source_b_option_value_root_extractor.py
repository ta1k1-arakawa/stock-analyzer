from __future__ import annotations

import inspect
import urllib.parse

import pytest

from src.v9_005_stage_a_jpx_probe import V9005StageABlocked
from src.v9_014_jpx_monthly_auction_activity_source_b_locator import (
    LOCATOR_OK,
    SOURCE_B_ARCHIVE_ROOT,
    resolve_source_b_year_page,
)
from src.v9_015_source_b_option_value_root_extractor import (
    REQUIRED_YEAR_LABELS,
    extract_option_value_root_year_candidates,
)


YEARS = REQUIRED_YEAR_LABELS


def _option(year: str, value: str | None = None, label: str | None = None) -> str:
    actual_value = f"synthetic/{year}.html" if value is None else value
    actual_label = year if label is None else label
    return f'<option value="{actual_value}">{actual_label}</option>'


def _document(*parts: str) -> bytes:
    return ("<html><body><select>" + "".join(parts) + "</select></body></html>").encode(
        "utf-8"
    )


def test_all_five_options_bind_exact_values_and_downstream_years():
    raw = _document(*(_option(year, f"synthetic/value-{year}.html") for year in YEARS))
    candidates = extract_option_value_root_year_candidates(raw, SOURCE_B_ARCHIVE_ROOT)

    assert tuple(candidate.label for candidate in candidates) == YEARS
    for candidate, year in zip(candidates, YEARS):
        expected = urllib.parse.urljoin(SOURCE_B_ARCHIVE_ROOT, f"synthetic/value-{year}.html")
        assert candidate.label == year
        assert candidate.href == expected
        result = resolve_source_b_year_page([candidate], int(year))
        assert result.status == LOCATOR_OK
        assert result.url == expected


def test_whitespace_normalization_is_exact_and_nested_visible_text_is_kept():
    raw = _document(
        '<option value="synthetic/2017.html"> 20<span>17</span> </option>',
        *(_option(year) for year in YEARS[1:]),
    )
    candidates = extract_option_value_root_year_candidates(raw)
    assert candidates[0].label == "2017"


def test_irrelevant_options_do_not_affect_binding_or_value_selection():
    raw = _document(
        '<option value="https://evil.example/not-selected">not a year</option>',
        '<option value="year-looking-value">2017</option>',
        *(_option(year) for year in YEARS[1:]),
    )
    candidates = extract_option_value_root_year_candidates(raw)
    assert candidates[0].href == urllib.parse.urljoin(SOURCE_B_ARCHIVE_ROOT, "year-looking-value")


def test_script_and_style_fake_year_text_is_excluded():
    raw = (
        b"<html><body><script>2017 2019</script><style>2020</style>"
        + _document(*(_option(year) for year in YEARS))[len(b"<html><body><select>") : -len(b"</select></body></html>")]
        + b"</body></html>"
    )
    candidates = extract_option_value_root_year_candidates(raw)
    assert tuple(candidate.label for candidate in candidates) == YEARS


def test_missing_required_year_fails_closed_without_visible_text_fallback():
    raw = _document(*(_option(year) for year in YEARS[:-1]), "<div>2026</div>")
    with pytest.raises(V9005StageABlocked):
        extract_option_value_root_year_candidates(raw)


def test_duplicate_eligible_option_fails_closed_without_first_last_selection():
    raw = _document(
        _option("2017", "synthetic/first.html"),
        _option("2017", "synthetic/last.html"),
        *(_option(year) for year in YEARS[1:]),
    )
    with pytest.raises(V9005StageABlocked):
        extract_option_value_root_year_candidates(raw)


@pytest.mark.parametrize(
    "fragment",
    [
        '<option>2017</option>',
        '<option value="">2017</option>',
    ],
)
def test_missing_or_empty_value_fails_closed(fragment: str):
    raw = _document(fragment, *(_option(year) for year in YEARS[1:]))
    with pytest.raises(V9005StageABlocked):
        extract_option_value_root_year_candidates(raw)


def test_duplicate_value_attribute_fails_closed():
    raw = _document(
        '<option value="one" value="two">2017</option>',
        *(_option(year) for year in YEARS[1:]),
    )
    with pytest.raises(V9005StageABlocked):
        extract_option_value_root_year_candidates(raw)


def test_required_year_anchor_candidate_is_never_a_fallback():
    raw = (
        _document(*(_option(year) for year in YEARS[1:]))
        .replace(b"</select>", b'<a href="synthetic/2017.html">2017</a></select>')
    )
    with pytest.raises(V9005StageABlocked):
        extract_option_value_root_year_candidates(raw)


def test_year_looking_value_or_url_cannot_create_a_match():
    raw = _document(
        '<option value="2017">not-2017</option>',
        '<option value="https://www.jpx.co.jp/english/2020.html">2020x</option>',
        *(_option(year) for year in ("2019", "2022", "2026")),
    )
    with pytest.raises(V9005StageABlocked):
        extract_option_value_root_year_candidates(raw)


def test_invalid_resolved_url_fails_through_inherited_reviewed_validator():
    raw = _document(
        *(_option(year, "synthetic/valid.html") for year in YEARS[:-1]),
        _option("2026", "https://evil.example/2026.html"),
    )
    with pytest.raises(V9005StageABlocked) as excinfo:
        extract_option_value_root_year_candidates(raw)
    assert excinfo.value.reason == "OFF_DOMAIN_REQUEST_REJECTED"


def test_wrong_root_provenance_fails_closed():
    raw = _document(*(_option(year) for year in YEARS))
    with pytest.raises(V9005StageABlocked):
        extract_option_value_root_year_candidates(raw, SOURCE_B_ARCHIVE_ROOT + "?wrong")


def test_extractor_does_not_import_or_call_v9014_anchor_parser():
    source = inspect.getsource(__import__(
        "src.v9_015_source_b_option_value_root_extractor", fromlist=["*"]
    ))
    assert "v9_014_jpx_monthly_auction_activity_source_b_archive_parser" not in source
    assert "extract_root_year_candidates as" not in source
    assert "casefold" not in source
    assert ".lower(" not in source
