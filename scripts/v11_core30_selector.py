"""Offline, deterministic TOPIX Core30 selector for V11.

This module consumes an already locked JPX PDF.  It has no source-network
transport and never reads research results or market-price data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

STUDY = "V11_MINIMAL_FIXED_STRATEGY_SINGLE_SECURITY"
JPX_SOURCE_URL = "https://www.jpx.co.jp/news/6030/um3qrc0000023smr-att/mei2_12_size.pdf"
JPX_PUBLICATION_DATE = "2025-10-07"
JPX_SNAPSHOT_EFFECTIVE_DATE = "2025-10-31"
FROZEN_DESIGN_COMMIT = "ff7b3e9c8ed44862367f341cd8ad4e9b9a899684"
FROZEN_DESIGN_BLOB = "35661977397cf007e298e224a6fbaa4f170eeffb"
FROZEN_DESIGN_DOCUMENT = "V11_MINIMAL_FIXED_STRATEGY_SINGLE_SECURITY_DESIGN_DRAFT.md"
FROZEN_DESIGN_SHA256 = "b229b8941aa8a4a4f5dfef5ff11cb759efe9223209b895eb025fd7a0ce892e40"
FREEZE_APPROVAL_COMMIT = "f767ef64949265b927824a2c5e151c8d572091fb"
FREEZE_APPROVAL_BLOB = "47b98097743f031c3b5c964078c1b94568d888ed"
FREEZE_APPROVAL_DOCUMENT = "V11_DESIGN_FREEZE_APPROVAL.json"
FREEZE_APPROVAL_SHA256 = "5d979566f9872866b7668481c05c393ec102579e60cab3d49ec401fb058431d1"
EXCLUSION_PARENT_SHA = "4fc91c19c5534d59e041876f5310b97e373eb1fc"
AUTHORITATIVE_BRANCH = "v11-minimal-single-security-study"
SEED_TEXT = "V11_MINIMAL_SINGLE_SECURITY|4fc91c19c5534d59e041876f5310b97e373eb1fc"
SAFE_SCHEMA = "V11_CORE30_DETERMINISTIC_SELECTION_SAFE_RESULT_V1"

OLD_HEADER = "旧（2025年10月7日時点）"
NEW_HEADER = "新（2025年10月31日適用）"
CORE30_CLASSIFICATION = "TOPIX Core30"
CODE_PATTERN = re.compile(r"^[0-9]{4}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SAFE_ERROR_PATTERN = re.compile(r"^[A-Z0-9_]+$")

VALIDATION_KEYS = (
    "frozen_design_binding",
    "freeze_approval_binding",
    "official_source_identity",
    "snapshot_header_identity",
    "exact_core30_count",
    "new_column_orientation",
    "exclusion_from_frozen_git_tree_only",
    "no_prior_profitability_output_read",
    "deterministic_selection",
    "no_network_in_selector",
    "no_historical_price_read",
    "no_model_fit",
    "no_backtest",
)
SAFE_RESULT_KEYS = (
    "schema_version",
    "study",
    "implementation_sha",
    "source_url",
    "source_sha256",
    "source_byte_count",
    "publication_date",
    "snapshot_effective_date",
    "core30_count",
    "core30_codes_sha256",
    "exclusion_parent_sha",
    "exclusion_count",
    "exclusion_codes_sha256",
    "eligible_count",
    "eligible_codes_sha256",
    "selection_hash",
    "selected_index",
    "selected_ticker",
    "validation",
)


class SelectorError(ValueError):
    """Fail-closed selector error without exposing source contents."""


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _compact_cell(value: object) -> str:
    return "".join(_normalize_cell(value).split())


def _canonical_codes_sha256(codes: Iterable[str]) -> str:
    ordered = sorted(codes, key=lambda code: (int(code), code))
    return hashlib.sha256(("".join(code + "\n" for code in ordered)).encode("utf-8")).hexdigest()


def _validate_code(value: object) -> str:
    code = _normalize_cell(value)
    if CODE_PATTERN.fullmatch(code) is None:
        raise SelectorError("SECURITY_CODE_INVALID")
    return code


def _header_positions(rows: Sequence[Sequence[object]], marker: str) -> list[tuple[int, int]]:
    marker_compact = _compact_cell(marker)
    return [
        (row_index, column_index)
        for row_index, row in enumerate(rows[:6])
        for column_index, cell in enumerate(row)
        if _compact_cell(cell) == marker_compact
    ]


def _table_columns(table: object) -> tuple[int, int] | None:
    if not isinstance(table, list) or not table or any(not isinstance(row, list) for row in table):
        raise SelectorError("CONSTITUENT_TABLE_INVALID")
    rows = [["" if cell is None else cell for cell in row] for row in table]
    old_positions = _header_positions(rows, OLD_HEADER)
    new_positions = _header_positions(rows, NEW_HEADER)
    if not old_positions and not new_positions:
        return None
    if len(old_positions) != 1 or len(new_positions) != 1:
        raise SelectorError("CONSTITUENT_HEADER_ORIENTATION_INVALID")
    old_classification_index = old_positions[0][1]
    new_classification_index = new_positions[0][1]
    if old_classification_index >= new_classification_index:
        raise SelectorError("CONSTITUENT_HEADER_ORIENTATION_INVALID")
    code_positions = [
        (row_index, column_index)
        for row_index, row in enumerate(rows[:6])
        for column_index, cell in enumerate(row)
        if _compact_cell(cell) == "コード"
    ]
    if not code_positions:
        raise SelectorError("CONSTITUENT_SHARED_CODE_HEADER_MISSING")
    if len(code_positions) != 1:
        raise SelectorError("CONSTITUENT_SHARED_CODE_HEADER_AMBIGUOUS")
    shared_code_index = code_positions[0][1]
    if shared_code_index >= old_classification_index or shared_code_index >= new_classification_index:
        raise SelectorError("CONSTITUENT_SHARED_CODE_COLUMN_AFTER_CLASSIFICATION")
    return shared_code_index, new_classification_index


def extract_new_core30_codes_from_tables(tables: Sequence[object]) -> list[str]:
    """Extract exactly the NEW TOPIX Core30 codes from pdfplumber tables."""

    matches: list[str] = []
    found_table = False
    for table in tables:
        columns = _table_columns(table)
        if columns is None:
            continue
        found_table = True
        code_index, class_index = columns
        for row in table:
            if not isinstance(row, list):
                raise SelectorError("CONSTITUENT_ROW_INVALID")
            if max(code_index, class_index) >= len(row):
                raise SelectorError("CONSTITUENT_ROW_SHORT")
            classification = _normalize_cell(row[class_index])
            if classification != CORE30_CLASSIFICATION:
                continue
            matches.append(_validate_code(row[code_index]))
    if not found_table:
        raise SelectorError("CONSTITUENT_HEADER_MISSING")
    if len(matches) != 30 or len(set(matches)) != 30:
        raise SelectorError("CORE30_COUNT_OR_DUPLICATE_INVALID")
    if "6857" not in matches or "6981" in matches:
        raise SelectorError("CORE30_COLUMN_ORIENTATION_INVALID")
    return sorted(matches, key=lambda code: int(code))


def extract_new_core30_codes_from_pdf_bytes(pdf_bytes: bytes) -> list[str]:
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise SelectorError("PDF_BYTES_INVALID")
    tables: list[object] = []
    try:
        import io
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
    except SelectorError:
        raise
    except Exception as error:
        raise SelectorError("PDF_TABLE_EXTRACTION_FAILED") from error
    return extract_new_core30_codes_from_tables(tables)


def select_core30_code(core30_codes: Sequence[str], exclusion_codes: Sequence[str]) -> dict[str, Any]:
    """Apply the frozen exclusion and hash selection rules."""

    normalized_core = [_validate_code(code) for code in core30_codes]
    if len(normalized_core) != 30 or len(set(normalized_core)) != 30:
        raise SelectorError("CORE30_COUNT_OR_DUPLICATE_INVALID")
    if "6857" not in normalized_core or "6981" in normalized_core:
        raise SelectorError("CORE30_COLUMN_ORIENTATION_INVALID")
    normalized_exclusion: list[str] = []
    for code in exclusion_codes:
        text = _normalize_cell(code)
        if not text.isdigit():
            raise SelectorError("EXCLUSION_CODE_INVALID")
        normalized_exclusion.append(text)
    exclusion_set = set(normalized_exclusion)
    eligible = sorted(set(normalized_core) - exclusion_set, key=lambda code: int(code))
    if not eligible:
        raise SelectorError("NO_ELIGIBLE_CODES")
    selection_hash = hashlib.sha256(SEED_TEXT.encode("utf-8")).hexdigest()
    selected_index = int(selection_hash, 16) % len(eligible)
    return {
        "core30_codes": sorted(normalized_core, key=lambda code: int(code)),
        "exclusion_codes": sorted(exclusion_set, key=lambda code: (int(code), code)),
        "eligible_codes": eligible,
        "core30_codes_sha256": _canonical_codes_sha256(normalized_core),
        "exclusion_codes_sha256": _canonical_codes_sha256(exclusion_set),
        "eligible_codes_sha256": _canonical_codes_sha256(eligible),
        "selection_hash": selection_hash,
        "selected_index": selected_index,
        "selected_ticker": eligible[selected_index],
    }


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
    )
    if completed.returncode != 0 or not isinstance(completed.stdout, bytes):
        raise SelectorError("GIT_OBSERVATION_FAILED")
    return completed.stdout


def _git_text(repository_root: Path, *arguments: str) -> str:
    raw = _git_bytes(repository_root, *arguments)
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise SelectorError("GIT_OUTPUT_UTF8_INVALID") from error


def _verify_repository_binding(repository_root: Path, implementation_sha: str) -> None:
    if SHA1_PATTERN.fullmatch(implementation_sha) is None:
        raise SelectorError("IMPLEMENTATION_SHA_INVALID")
    if _git_text(repository_root, "rev-parse", "--abbrev-ref", "HEAD") != AUTHORITATIVE_BRANCH:
        raise SelectorError("BRANCH_MISMATCH")
    if _git_text(repository_root, "rev-parse", "HEAD") != implementation_sha:
        raise SelectorError("IMPLEMENTATION_SHA_HEAD_MISMATCH")
    if _git_text(repository_root, "rev-parse", "refs/remotes/origin/" + AUTHORITATIVE_BRANCH) != implementation_sha:
        raise SelectorError("IMPLEMENTATION_SHA_TRACKING_MISMATCH")
    if _git_text(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise SelectorError("WORKTREE_NOT_CLEAN")


def _verify_frozen_blob(repository_root: Path, commit: str, path: str, expected_blob: str, expected_sha256: str) -> None:
    if _git_text(repository_root, "rev-parse", f"{commit}:{path}") != expected_blob:
        raise SelectorError("FROZEN_BLOB_MISMATCH")
    raw = _git_bytes(repository_root, "show", f"{commit}:{path}")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise SelectorError("FROZEN_SHA256_MISMATCH")


def _verify_frozen_bindings(repository_root: Path) -> None:
    _verify_frozen_blob(
        repository_root,
        FROZEN_DESIGN_COMMIT,
        FROZEN_DESIGN_DOCUMENT,
        FROZEN_DESIGN_BLOB,
        FROZEN_DESIGN_SHA256,
    )
    _verify_frozen_blob(
        repository_root,
        FREEZE_APPROVAL_COMMIT,
        FREEZE_APPROVAL_DOCUMENT,
        FREEZE_APPROVAL_BLOB,
        FREEZE_APPROVAL_SHA256,
    )
    if _git_text(repository_root, "rev-parse", EXCLUSION_PARENT_SHA) != EXCLUSION_PARENT_SHA:
        raise SelectorError("EXCLUSION_PARENT_MISSING")


def _exclusion_codes_from_git_tree(repository_root: Path) -> list[str]:
    listing = _git_text(repository_root, "ls-tree", "-r", "--name-only", EXCLUSION_PARENT_SHA, "--", "data/benchmark/ohlcv")
    codes: set[str] = set()
    prefix = "data/benchmark/ohlcv/"
    for relative_path in listing.splitlines():
        if not relative_path.startswith(prefix) or "/" in relative_path[len(prefix) :]:
            continue
        if not relative_path.lower().endswith(".csv"):
            continue
        stem = relative_path[len(prefix) : -4]
        if stem.isdigit():
            codes.add(stem)
    return sorted(codes, key=lambda code: (int(code), code))


def _read_pdf(repository_root: Path, pdf_path: Path) -> tuple[bytes, str, int]:
    if not pdf_path.is_absolute():
        pdf_path = repository_root / pdf_path
    if pdf_path.is_symlink() or not pdf_path.is_file():
        raise SelectorError("JPX_PDF_UNAVAILABLE")
    try:
        raw = pdf_path.read_bytes()
    except OSError as error:
        raise SelectorError("JPX_PDF_READ_FAILED") from error
    if not raw:
        raise SelectorError("JPX_PDF_EMPTY")
    return raw, hashlib.sha256(raw).hexdigest(), len(raw)


def build_safe_result(repository_root: Path, pdf_path: Path, implementation_sha: str) -> dict[str, Any]:
    _verify_repository_binding(repository_root, implementation_sha)
    _verify_frozen_bindings(repository_root)
    pdf_bytes, source_sha256, source_byte_count = _read_pdf(repository_root, pdf_path)
    core30_codes = extract_new_core30_codes_from_pdf_bytes(pdf_bytes)
    exclusion_codes = _exclusion_codes_from_git_tree(repository_root)
    selection = select_core30_code(core30_codes, exclusion_codes)
    result = {
        "schema_version": SAFE_SCHEMA,
        "study": STUDY,
        "implementation_sha": implementation_sha,
        "source_url": JPX_SOURCE_URL,
        "source_sha256": source_sha256,
        "source_byte_count": source_byte_count,
        "publication_date": JPX_PUBLICATION_DATE,
        "snapshot_effective_date": JPX_SNAPSHOT_EFFECTIVE_DATE,
        "core30_count": len(selection["core30_codes"]),
        "core30_codes_sha256": selection["core30_codes_sha256"],
        "exclusion_parent_sha": EXCLUSION_PARENT_SHA,
        "exclusion_count": len(selection["exclusion_codes"]),
        "exclusion_codes_sha256": selection["exclusion_codes_sha256"],
        "eligible_count": len(selection["eligible_codes"]),
        "eligible_codes_sha256": selection["eligible_codes_sha256"],
        "selection_hash": selection["selection_hash"],
        "selected_index": selection["selected_index"],
        "selected_ticker": selection["selected_ticker"],
        "validation": {key: True for key in VALIDATION_KEYS},
    }
    validate_safe_result(result)
    return result


def validate_safe_result(result: object) -> None:
    if not isinstance(result, dict) or set(result.keys()) != set(SAFE_RESULT_KEYS):
        raise SelectorError("SAFE_RESULT_KEYS_INVALID")
    validation = result["validation"]
    if not isinstance(validation, dict) or set(validation.keys()) != set(VALIDATION_KEYS):
        raise SelectorError("SAFE_RESULT_VALIDATION_KEYS_INVALID")
    if result["schema_version"] != SAFE_SCHEMA or result["study"] != STUDY:
        raise SelectorError("SAFE_RESULT_IDENTITY_INVALID")
    if not isinstance(result["implementation_sha"], str) or SHA1_PATTERN.fullmatch(result["implementation_sha"]) is None:
        raise SelectorError("SAFE_RESULT_IMPLEMENTATION_SHA_INVALID")
    if result["source_url"] != JPX_SOURCE_URL or result["publication_date"] != JPX_PUBLICATION_DATE or result["snapshot_effective_date"] != JPX_SNAPSHOT_EFFECTIVE_DATE:
        raise SelectorError("SAFE_RESULT_SOURCE_IDENTITY_INVALID")
    for key in ("source_sha256", "core30_codes_sha256", "exclusion_codes_sha256", "eligible_codes_sha256", "selection_hash"):
        if not isinstance(result[key], str) or SHA256_PATTERN.fullmatch(result[key]) is None:
            raise SelectorError("SAFE_RESULT_HASH_INVALID")
    if result["exclusion_parent_sha"] != EXCLUSION_PARENT_SHA:
        raise SelectorError("SAFE_RESULT_EXCLUSION_PARENT_INVALID")
    int_keys = ("source_byte_count", "core30_count", "exclusion_count", "eligible_count", "selected_index")
    for key in int_keys:
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < 0:
            raise SelectorError("SAFE_RESULT_INTEGER_INVALID")
    if result["source_byte_count"] <= 0 or result["core30_count"] != 30 or result["eligible_count"] <= 0:
        raise SelectorError("SAFE_RESULT_COUNT_INVALID")
    if result["selected_index"] >= result["eligible_count"] or not isinstance(result["selected_ticker"], str) or CODE_PATTERN.fullmatch(result["selected_ticker"]) is None:
        raise SelectorError("SAFE_RESULT_SELECTION_INVALID")
    if result["selection_hash"] != hashlib.sha256(SEED_TEXT.encode("utf-8")).hexdigest():
        raise SelectorError("SAFE_RESULT_SELECTION_HASH_INVALID")
    if result["selected_index"] != int(result["selection_hash"], 16) % result["eligible_count"]:
        raise SelectorError("SAFE_RESULT_SELECTION_INDEX_INVALID")
    if any(type(validation[key]) is not bool or validation[key] is not True for key in VALIDATION_KEYS):
        raise SelectorError("SAFE_RESULT_VALIDATION_INVALID")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline V11 TOPIX Core30 selector")
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--jpx-pdf", required=True, type=Path)
    parser.add_argument("--implementation-sha", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parse_args(argv)
        result = build_safe_result(arguments.repository_root.resolve(), arguments.jpx_pdf, arguments.implementation_sha)
    except SelectorError as error:
        safe_enum = str(error)
        if SAFE_ERROR_PATTERN.fullmatch(safe_enum) is None:
            safe_enum = "UNEXPECTED_IMPLEMENTATION_FAILURE"
        sys.stderr.write(f"SELECTOR_ERROR={safe_enum}\n")
        return 1
    except Exception:
        sys.stderr.write("SELECTOR_ERROR=UNEXPECTED_IMPLEMENTATION_FAILURE\n")
        return 1
    sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
