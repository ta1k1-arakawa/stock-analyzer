from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import v11_core30_selector as selector


def _codes() -> list[str]:
    values = [str(1300 + index) for index in range(30)]
    values[0] = "6857"
    values[1] = "7203"
    values[2] = "8035"
    values[3] = "9432"
    values[4] = "9984"
    return sorted(set(values), key=int)


def _table(codes: list[str] | None = None, *, classification: str = "TOPIX Core30") -> list[list[str]]:
    codes = _codes() if codes is None else codes
    rows = [["旧（2025年10月7日時点）", "", "", "新（2025年10月31日適用）", "", ""]]
    rows.append(["旧コード", "旧区分", "", "新コード", "新区分", ""])
    rows.extend([["0000", "TOPIX Core30", "", code, classification, ""] for code in codes])
    return rows


def _valid_result() -> dict:
    selection = selector.select_core30_code(_codes(), ["7203"])
    result = {
        "schema_version": selector.SAFE_SCHEMA,
        "study": selector.STUDY,
        "implementation_sha": "a" * 40,
        "source_url": selector.JPX_SOURCE_URL,
        "source_sha256": "b" * 64,
        "source_byte_count": 10,
        "publication_date": selector.JPX_PUBLICATION_DATE,
        "snapshot_effective_date": selector.JPX_SNAPSHOT_EFFECTIVE_DATE,
        "core30_count": 30,
        "core30_codes_sha256": selection["core30_codes_sha256"],
        "exclusion_parent_sha": selector.EXCLUSION_PARENT_SHA,
        "exclusion_count": 1,
        "exclusion_codes_sha256": selection["exclusion_codes_sha256"],
        "eligible_count": len(selection["eligible_codes"]),
        "eligible_codes_sha256": selection["eligible_codes_sha256"],
        "selection_hash": selection["selection_hash"],
        "selected_index": selection["selected_index"],
        "selected_ticker": selection["selected_ticker"],
        "validation": {key: True for key in selector.VALIDATION_KEYS},
    }
    return result


def test_valid_new_core30_extraction_and_orientation() -> None:
    assert selector.extract_new_core30_codes_from_tables([_table()]) == _codes()


@pytest.mark.parametrize(
    "codes, expected",
    [
        (_codes()[:-1], "CORE30_COUNT_OR_DUPLICATE_INVALID"),
        (_codes()[:-1] + [_codes()[0]], "CORE30_COUNT_OR_DUPLICATE_INVALID"),
        ([code for code in _codes() if code != "6857"] + ["6999"], "CORE30_COLUMN_ORIENTATION_INVALID"),
        (_codes()[:-1] + ["6981"], "CORE30_COLUMN_ORIENTATION_INVALID"),
    ],
)
def test_core30_shape_and_orientation_fail_closed(codes: list[str], expected: str) -> None:
    with pytest.raises(selector.SelectorError, match=expected):
        selector.extract_new_core30_codes_from_tables([_table(codes)])


def test_old_new_orientation_reversal_fails_closed() -> None:
    rows = [["新（2025年10月31日適用）", "", "", "旧（2025年10月7日時点）", "", ""]]
    rows.append(["新コード", "新区分", "", "旧コード", "旧区分", ""])
    rows.extend([[code, "TOPIX Core30", "", "0000", "TOPIX Core30", ""] for code in _codes()])
    with pytest.raises(selector.SelectorError):
        selector.extract_new_core30_codes_from_tables([rows])


def test_malformed_non_numeric_code_fails_closed() -> None:
    codes = _codes()
    table = _table(codes)
    table[2][3] = "68A7"
    with pytest.raises(selector.SelectorError, match="SECURITY_CODE_INVALID"):
        selector.extract_new_core30_codes_from_tables([table])


def test_deterministic_hashes_and_selected_index() -> None:
    selected = selector.select_core30_code(_codes(), ["7203", "8035"])
    expected_hash = hashlib.sha256(selector.SEED_TEXT.encode("utf-8")).hexdigest()
    assert selected["selection_hash"] == expected_hash
    assert selected["selected_index"] == int(expected_hash, 16) % len(selected["eligible_codes"])
    assert selected["core30_codes_sha256"] == selector._canonical_codes_sha256(_codes())


def test_exclusion_set_can_be_derived_from_frozen_style_paths() -> None:
    paths = [
        "data/benchmark/ohlcv/7203.csv",
        "data/benchmark/ohlcv/1301.csv",
        "data/benchmark/ohlcv/notes.csv",
        "data/benchmark/ohlcv/nested/8035.csv",
    ]
    codes = {Path(path).stem for path in paths if path.startswith("data/benchmark/ohlcv/") and "/" not in path[len("data/benchmark/ohlcv/") :] and path.lower().endswith(".csv") and Path(path).stem.isdigit()}
    assert sorted(codes, key=int) == ["1301", "7203"]


def test_no_eligible_codes_fails() -> None:
    with pytest.raises(selector.SelectorError, match="NO_ELIGIBLE_CODES"):
        selector.select_core30_code(_codes(), _codes())


def test_safe_result_exact_schema_and_types() -> None:
    result = _valid_result()
    selector.validate_safe_result(result)
    extra = dict(result)
    extra["unexpected"] = True
    with pytest.raises(selector.SelectorError, match="SAFE_RESULT_KEYS_INVALID"):
        selector.validate_safe_result(extra)
    wrong = dict(result)
    wrong["selected_index"] = True
    with pytest.raises(selector.SelectorError):
        selector.validate_safe_result(wrong)


def test_malformed_implementation_sha_fails() -> None:
    result = _valid_result()
    result["implementation_sha"] = "not-a-sha"
    with pytest.raises(selector.SelectorError, match="SAFE_RESULT_IMPLEMENTATION_SHA_INVALID"):
        selector.validate_safe_result(result)


def test_cli_has_no_network_or_real_pdf_selection() -> None:
    source = Path(selector.__file__).read_text(encoding="utf-8")
    assert "urlopen" not in source
    assert "requests" not in source
    assert "--jpx-pdf" in source
    assert "--implementation-sha" in source
