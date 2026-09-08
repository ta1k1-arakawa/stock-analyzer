"""V9_016 C2 one-shot execution boundary.

The production CLI is intentionally narrow: it reads one caller-supplied
year-page file, binds its bytes to the frozen 2017 evidence identity, and
invokes the reviewed C1 probe once.  It never selects a category or resolves
links.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Mapping

if __package__ in {None, ""}:
    _script_directory = os.path.normcase(str(Path(__file__).resolve().parent))
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.normcase(os.path.abspath(entry or os.curdir)) != _script_directory
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v9_016_source_b_2017_month_header_structural_probe import (
    C1StructureResult,
    probe_2017_month_header_structure,
)


CONFIRMATION_CONTRACT = "V9_016_C2_2017_MONTH_HEADER_STRUCTURE"
SCHEMA_VERSION = "V9_016_C2_EXECUTION_V1"
TARGET_YEAR = 2017
EXPECTED_YEAR_BYTE_COUNT = 98936
EXPECTED_YEAR_SHA256 = (
    "1dc982e97b1d4ce7d52bc25631ddc46a219d22f82b393881f40e5d2478177821"
)

_HEX_40 = re.compile(r"[0-9a-fA-F]{40}\Z")
Probe = Callable[[bytes], C1StructureResult]


def _safe_failure(
    failure_class: str,
    reason: str,
    expected_git_sha: str | None,
    expected_year_byte_count: int,
    expected_year_sha256: str,
    probe_invocations: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL",
        "failure_class": failure_class,
        "reason": reason,
        "expected_git_sha": expected_git_sha,
        "year": TARGET_YEAR,
        "year_byte_count": expected_year_byte_count,
        "year_sha256": expected_year_sha256,
        "probe_invocations": probe_invocations,
    }


def _is_valid_git_sha(value: object) -> bool:
    return isinstance(value, str) and _HEX_40.fullmatch(value) is not None


def _write_exclusive(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _write_failure(
    output_root: Path,
    receipt: dict[str, object],
) -> None:
    try:
        _write_exclusive(output_root / "failure.json", receipt)
    except Exception:
        pass


def _run_c2_with_binding(
    year_page: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    expected_git_sha: str,
    confirmation: str,
    *,
    execute_c2: bool,
    expected_year_byte_count: int,
    expected_year_sha256: str,
    probe: Probe,
) -> dict[str, object]:
    """Run the boundary with explicit binding values for synthetic tests."""

    output_path = Path(output_root)
    input_path = Path(year_page)
    valid_sha = _is_valid_git_sha(expected_git_sha)

    if output_path.exists():
        return _safe_failure(
            "GOVERNANCE_FAILURE",
            "OUTPUT_ROOT_COLLISION",
            expected_git_sha if valid_sha else None,
            expected_year_byte_count,
            expected_year_sha256,
            0,
        )
    if not execute_c2:
        return _safe_failure(
            "GOVERNANCE_FAILURE",
            "EXECUTE_C2_FLAG_REQUIRED",
            expected_git_sha if valid_sha else None,
            expected_year_byte_count,
            expected_year_sha256,
            0,
        )
    if not valid_sha:
        return _safe_failure(
            "GOVERNANCE_FAILURE",
            "EXPECTED_GIT_SHA_INVALID",
            None,
            expected_year_byte_count,
            expected_year_sha256,
            0,
        )
    if confirmation != CONFIRMATION_CONTRACT:
        return _safe_failure(
            "GOVERNANCE_FAILURE",
            "CONFIRMATION_CONTRACT_MISMATCH",
            expected_git_sha,
            expected_year_byte_count,
            expected_year_sha256,
            0,
        )

    try:
        output_path.mkdir(exist_ok=False)
    except Exception:
        return _safe_failure(
            "GOVERNANCE_FAILURE",
            "OUTPUT_ROOT_CREATE_FAILURE",
            expected_git_sha,
            expected_year_byte_count,
            expected_year_sha256,
            0,
        )

    attempt = {
        "schema_version": SCHEMA_VERSION,
        "status": "IN_PROGRESS",
        "expected_git_sha": expected_git_sha,
        "confirmation_contract": CONFIRMATION_CONTRACT,
        "target_year": TARGET_YEAR,
        "expected_year_byte_count": expected_year_byte_count,
        "expected_year_sha256": expected_year_sha256,
        "target_probe_invocations": 1,
    }
    try:
        _write_exclusive(output_path / "attempt.json", attempt)
    except Exception:
        failure = _safe_failure(
            "IMPLEMENTATION_FAILURE",
            "ATTEMPT_METADATA_WRITE_FAILURE",
            expected_git_sha,
            expected_year_byte_count,
            expected_year_sha256,
            0,
        )
        _write_failure(output_path, failure)
        return failure

    try:
        page_bytes = input_path.read_bytes()
    except Exception:
        failure = _safe_failure(
            "GOVERNANCE_FAILURE",
            "YEAR_PAGE_READ_FAILURE",
            expected_git_sha,
            expected_year_byte_count,
            expected_year_sha256,
            0,
        )
        _write_failure(output_path, failure)
        return failure

    if len(page_bytes) != expected_year_byte_count:
        failure = _safe_failure(
            "GOVERNANCE_FAILURE",
            "YEAR_PAGE_BYTE_COUNT_MISMATCH",
            expected_git_sha,
            expected_year_byte_count,
            expected_year_sha256,
            0,
        )
        _write_failure(output_path, failure)
        return failure
    if hashlib.sha256(page_bytes).hexdigest() != expected_year_sha256:
        failure = _safe_failure(
            "GOVERNANCE_FAILURE",
            "YEAR_PAGE_SHA256_MISMATCH",
            expected_git_sha,
            expected_year_byte_count,
            expected_year_sha256,
            0,
        )
        _write_failure(output_path, failure)
        return failure

    try:
        probe_result = probe(page_bytes)
        probe_safe = probe_result.to_dict()
        probe_invocations = 1
        if probe_safe.get("status") != "PASS" or probe_safe.get("parser_success") is not True:
            raise ValueError
    except Exception:
        failure = _safe_failure(
            "IMPLEMENTATION_FAILURE",
            "C1_PROBE_FAILURE",
            expected_git_sha,
            expected_year_byte_count,
            expected_year_sha256,
            1,
        )
        _write_failure(output_path, failure)
        return failure

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "C2_PROBE_EXECUTION_COMPLETE",
        "failure_class": None,
        "reason": None,
        "expected_git_sha": expected_git_sha,
        "year": TARGET_YEAR,
        "year_byte_count": expected_year_byte_count,
        "year_sha256": expected_year_sha256,
        "probe_invocations": 1,
        "probe_result": probe_safe,
    }
    try:
        _write_exclusive(output_path / "result.json", result)
    except Exception:
        failure = _safe_failure(
            "IMPLEMENTATION_FAILURE",
            "SAFE_RESULT_WRITE_FAILURE",
            expected_git_sha,
            expected_year_byte_count,
            expected_year_sha256,
            1,
        )
        _write_failure(output_path, failure)
        return failure

    complete = {
        "schema_version": SCHEMA_VERSION,
        "status": "C2_PROBE_EXECUTION_COMPLETE",
        "probe_invocations": 1,
    }
    try:
        _write_exclusive(output_path / "complete.json", complete)
    except Exception:
        failure = _safe_failure(
            "IMPLEMENTATION_FAILURE",
            "COMPLETE_MARKER_WRITE_FAILURE",
            expected_git_sha,
            expected_year_byte_count,
            expected_year_sha256,
            1,
        )
        _write_failure(output_path, failure)
        return failure

    return result


def run_c2(
    year_page: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    expected_git_sha: str,
    confirmation: str,
    *,
    execute_c2: bool,
) -> dict[str, object]:
    """Run C2 using the immutable production byte-count and SHA binding."""

    return _run_c2_with_binding(
        year_page,
        output_root,
        expected_git_sha,
        confirmation,
        execute_c2=execute_c2,
        expected_year_byte_count=EXPECTED_YEAR_BYTE_COUNT,
        expected_year_sha256=EXPECTED_YEAR_SHA256,
        probe=probe_2017_month_header_structure,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V9_016 C2 probe boundary.")
    parser.add_argument("--year-page", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--execute-c2", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    receipt = run_c2(
        args.year_page,
        args.output_root,
        args.expected_git_sha,
        args.confirmation,
        execute_c2=args.execute_c2,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["status"] == "C2_PROBE_EXECUTION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIRMATION_CONTRACT",
    "EXPECTED_YEAR_BYTE_COUNT",
    "EXPECTED_YEAR_SHA256",
    "SCHEMA_VERSION",
    "TARGET_YEAR",
    "main",
    "run_c2",
]
