from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import v12_core30_selector as selector


@pytest.fixture(autouse=True)
def _synthetic_candidate_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use synthetic identities without exposing the frozen real candidate set."""

    monkeypatch.setattr(selector, "CORE30_CODES_SHA256", selector._canonical_codes_sha256(_codes()))


def _codes() -> list[str]:
    values = [str(1300 + index) for index in range(31)]
    values[0] = "6857"
    values[1] = "7203"
    values[2] = "8035"
    values[3] = "9432"
    values[4] = "9984"
    return sorted(set(values), key=int)


def _table(
    codes: list[str] | None = None,
    *,
    old_classification: str = "TOPIX Core30",
    classification: str = "TOPIX Core30",
) -> list[list[str]]:
    codes = _codes() if codes is None else codes
    rows = [["No.", "コード", "銘柄名", "TOPIXニューインデックスシリーズ区分", ""]]
    rows.append(["", "", "", "旧（2025年10月7日時点）", "新（2025年10月31日適用）"])
    rows.extend([[str(index), code, "Synthetic", old_classification, classification] for index, code in enumerate(codes, 1)])
    return rows


def _valid_result() -> dict:
    selection = selector.select_core30_code(_codes(), ["7203"])
    return {
        "schema_version": selector.SAFE_SCHEMA,
        "study": selector.STUDY,
        "implementation_sha": "a" * 40,
        "source_url": selector.JPX_SOURCE_URL,
        "source_sha256": selector.SOURCE_SHA256,
        "source_byte_count": selector.SOURCE_BYTE_COUNT,
        "publication_date": selector.JPX_PUBLICATION_DATE,
        "snapshot_effective_date": selector.JPX_SNAPSHOT_EFFECTIVE_DATE,
        "core30_count": selector.CORE30_COUNT,
        "core30_codes_sha256": selector.CORE30_CODES_SHA256,
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


def test_valid_new_core30_extraction_and_orientation() -> None:
    assert selector.extract_new_core30_codes_from_tables([_table()]) == _codes()


@pytest.mark.parametrize(
    "codes, expected",
    [
        (_codes()[:-1], "CORE30_COUNT_OR_DUPLICATE_INVALID"),
        (_codes() + ["9999"], "CORE30_COUNT_OR_DUPLICATE_INVALID"),
        (_codes()[:-1] + [_codes()[0]], "CORE30_COUNT_OR_DUPLICATE_INVALID"),
        ([code for code in _codes() if code != "6857"] + ["6999"], "CORE30_COLUMN_ORIENTATION_INVALID"),
        (_codes()[:-1] + ["6981"], "CORE30_COLUMN_ORIENTATION_INVALID"),
    ],
)
def test_candidate_shape_and_orientation_fail_closed(codes: list[str], expected: str) -> None:
    with pytest.raises(selector.SelectorError, match=expected):
        selector.extract_new_core30_codes_from_tables([_table(codes)])


def test_old_new_orientation_reversal_fails_closed() -> None:
    rows = [["No.", "コード", "銘柄名", "", ""]]
    rows.append(["", "", "", "新（2025年10月31日適用）", "旧（2025年10月7日時点）"])
    rows.extend([[str(index), code, "Synthetic", "TOPIX Core30", "TOPIX Core30"] for index, code in enumerate(_codes(), 1)])
    with pytest.raises(selector.SelectorError):
        selector.extract_new_core30_codes_from_tables([rows])


def test_exact_candidate_hash_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(selector, "CORE30_CODES_SHA256", "0" * 64)
    with pytest.raises(selector.SelectorError, match="CORE30_CODES_SHA256_MISMATCH"):
        selector.extract_new_core30_codes_from_tables([_table()])


def test_malformed_non_numeric_code_fails_closed() -> None:
    table = _table()
    table[2][1] = "68A7"
    with pytest.raises(selector.SelectorError, match="SECURITY_CODE_INVALID"):
        selector.extract_new_core30_codes_from_tables([table])


def test_missing_shared_code_header_fails_closed() -> None:
    table = _table()
    table[0][1] = "銘柄名"
    with pytest.raises(selector.SelectorError, match="CONSTITUENT_SHARED_CODE_HEADER_MISSING"):
        selector.extract_new_core30_codes_from_tables([table])


def test_duplicate_shared_code_header_fails_closed() -> None:
    table = _table()
    table[0].insert(2, "コード")
    with pytest.raises(selector.SelectorError, match="CONSTITUENT_SHARED_CODE_HEADER_AMBIGUOUS"):
        selector.extract_new_core30_codes_from_tables([table])


def test_shared_code_after_classification_fails_closed() -> None:
    table = [
        ["No.", "", "銘柄名", "コード", ""],
        ["", "旧（2025年10月7日時点）", "新（2025年10月31日適用）", "", ""],
    ]
    table.extend([[str(index), "TOPIX Core30", "TOPIX Core30", code, ""] for index, code in enumerate(_codes(), 1)])
    with pytest.raises(selector.SelectorError, match="CONSTITUENT_SHARED_CODE_COLUMN_AFTER_CLASSIFICATION"):
        selector.extract_new_core30_codes_from_tables([table])


def test_short_row_fails_closed() -> None:
    table = _table()
    table.append(["32", "6857"])
    with pytest.raises(selector.SelectorError, match="CONSTITUENT_ROW_SHORT"):
        selector.extract_new_core30_codes_from_tables([table])


def test_old_core30_but_new_not_core30_is_not_selected() -> None:
    table = _table(old_classification="TOPIX Core30", classification="TOPIX Mid400")
    with pytest.raises(selector.SelectorError, match="CORE30_COUNT_OR_DUPLICATE_INVALID"):
        selector.extract_new_core30_codes_from_tables([table])


def test_deterministic_hashes_and_selected_index() -> None:
    selected = selector.select_core30_code(_codes(), ["7203", "8035"])
    expected_hash = hashlib.sha256(selector.SEED_TEXT.encode("utf-8")).hexdigest()
    assert expected_hash == selector.SELECTION_HASH
    assert selected["selection_hash"] == expected_hash
    assert selected["selected_index"] == int(expected_hash, 16) % len(selected["eligible_codes"])
    assert selected["core30_codes_sha256"] == selector.CORE30_CODES_SHA256


def test_exclusion_set_can_be_derived_from_frozen_style_paths() -> None:
    paths = [
        "data/benchmark/ohlcv/7203.csv",
        "data/benchmark/ohlcv/1301.csv",
        "data/benchmark/ohlcv/notes.csv",
        "data/benchmark/ohlcv/nested/8035.csv",
    ]
    codes = {Path(path).stem for path in paths if path.startswith("data/benchmark/ohlcv/") and "/" not in path[len("data/benchmark/ohlcv/") :] and path.lower().endswith(".csv") and Path(path).stem.isdigit()}
    assert sorted(codes, key=int) == ["1301", "7203"]


def test_real_frozen_bindings_pass_against_repository() -> None:
    repository_root = Path(__file__).parents[1].resolve()
    selector._verify_frozen_bindings(repository_root)


def test_real_exclusion_tree_matches_frozen_parent() -> None:
    repository_root = Path(__file__).parents[1].resolve()
    assert selector._exclusion_codes_from_git_tree(repository_root) == [
        "1570",
        "4188",
        "4689",
        "5020",
        "7211",
        "7267",
        "8306",
        "9432",
    ]


def test_source_byte_count_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    repository_root = Path(__file__).parents[1].resolve()
    raw = b"synthetic source bytes"
    monkeypatch.setattr(selector, "_verify_repository_binding", lambda *_: None)
    monkeypatch.setattr(selector, "_read_pdf", lambda *_: (raw, selector.SOURCE_SHA256, len(raw)))
    with pytest.raises(selector.SelectorError, match="SOURCE_BYTE_COUNT_MISMATCH"):
        selector.build_safe_result(repository_root, Path("synthetic_locked_jpx.pdf"), selector._git_text(repository_root, "rev-parse", "HEAD"))


def test_source_sha256_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    repository_root = Path(__file__).parents[1].resolve()
    raw = b"synthetic source bytes"
    monkeypatch.setattr(selector, "_verify_repository_binding", lambda *_: None)
    monkeypatch.setattr(selector, "_read_pdf", lambda *_: (raw, "0" * 64, selector.SOURCE_BYTE_COUNT))
    with pytest.raises(selector.SelectorError, match="SOURCE_SHA256_MISMATCH"):
        selector.build_safe_result(repository_root, Path("synthetic_locked_jpx.pdf"), selector._git_text(repository_root, "rev-parse", "HEAD"))


def test_real_build_safe_result_and_validator_closure(monkeypatch: pytest.MonkeyPatch) -> None:
    repository_root = Path(__file__).parents[1].resolve()
    synthetic_bytes = b"synthetic locked PDF boundary bytes"
    monkeypatch.setattr(selector, "_verify_repository_binding", lambda *_: None)
    monkeypatch.setattr(
        selector,
        "_read_pdf",
        lambda *_: (synthetic_bytes, selector.SOURCE_SHA256, selector.SOURCE_BYTE_COUNT),
    )
    monkeypatch.setattr(selector, "extract_new_core30_codes_from_pdf_bytes", lambda _: _codes())

    implementation_sha = selector._git_text(repository_root, "rev-parse", "HEAD")
    result = selector.build_safe_result(repository_root, Path("synthetic_locked_jpx.pdf"), implementation_sha)
    expected = selector.select_core30_code(
        _codes(),
        selector._exclusion_codes_from_git_tree(repository_root),
    )

    assert set(result) == set(selector.SAFE_RESULT_KEYS)
    assert set(result["validation"]) == set(selector.VALIDATION_KEYS)
    assert all(value is True for value in result["validation"].values())
    assert result["source_byte_count"] == selector.SOURCE_BYTE_COUNT
    assert result["source_sha256"] == selector.SOURCE_SHA256
    assert result["core30_count"] == selector.CORE30_COUNT
    assert result["core30_codes_sha256"] == selector.CORE30_CODES_SHA256
    assert result["eligible_count"] > 0
    assert result["selected_index"] == int(result["selection_hash"], 16) % result["eligible_count"]
    assert result["selected_index"] == expected["selected_index"]
    assert result["selected_ticker"] == expected["selected_ticker"]


@pytest.mark.parametrize("mutation", ["validation", "source_url", "source_sha256", "source_byte_count", "candidate_hash", "selected_index"])
def test_real_safe_result_validator_negative_closure(mutation: str) -> None:
    result = _valid_result()
    if mutation == "validation":
        result["validation"] = dict(result["validation"])
        result["validation"]["no_backtest"] = False
    elif mutation == "source_url":
        result["source_url"] = "https://invalid.example/"
    elif mutation == "source_sha256":
        result["source_sha256"] = "0" * 64
    elif mutation == "source_byte_count":
        result["source_byte_count"] = selector.SOURCE_BYTE_COUNT - 1
    elif mutation == "candidate_hash":
        result["core30_codes_sha256"] = "0" * 64
    else:
        result["selected_index"] = (result["selected_index"] + 1) % result["eligible_count"]
    with pytest.raises(selector.SelectorError):
        selector.validate_safe_result(result)


def test_cli_selector_error_is_safe_and_stdout_empty(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fail(*_: object, **__: object) -> dict:
        raise selector.SelectorError("JPX_PDF_UNAVAILABLE")

    monkeypatch.setattr(selector, "build_safe_result", fail)
    assert selector.main(["--repository-root", ".", "--jpx-pdf", "locked.pdf", "--implementation-sha", "a" * 40]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "SELECTOR_ERROR=JPX_PDF_UNAVAILABLE\n"


def test_cli_unexpected_exception_is_safe_and_stdout_empty(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def fail(*_: object, **__: object) -> dict:
        raise RuntimeError("private path and PDF contents must not escape")

    monkeypatch.setattr(selector, "build_safe_result", fail)
    assert selector.main(["--repository-root", ".", "--jpx-pdf", "locked.pdf", "--implementation-sha", "a" * 40]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "SELECTOR_ERROR=UNEXPECTED_IMPLEMENTATION_FAILURE\n"


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
    assert "urllib" not in source
    assert "--jpx-pdf" in source
    assert "--implementation-sha" in source
