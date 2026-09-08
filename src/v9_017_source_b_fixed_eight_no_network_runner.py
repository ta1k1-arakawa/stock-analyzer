"""V9_017 fixed-eight no-network production boundary.

The CLI reads only caller-supplied preserved root/year-page files after
governance checks, verifies each immutable byte identity once, derives parent
year-page URLs from the verified root with the reviewed V9_015/V9_014
mechanics, and composes the pure fixed-eight locator application.  It never
performs acquisition, PDF work, or research calculations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__ in {None, ""}:
    _script_directory = os.path.normcase(str(Path(__file__).resolve().parent))
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.normcase(os.path.abspath(entry or os.curdir)) != _script_directory
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v9_014_jpx_monthly_auction_activity_source_b_locator import (
    LOCATOR_OK,
    SOURCE_B_ARCHIVE_ROOT,
    resolve_source_b_year_page,
)
from src.v9_015_source_b_option_value_root_extractor import (
    extract_option_value_root_year_candidates,
)
from src.v9_017_source_b_fixed_eight_locator_application import (
    FixedEightLocatorApplicationResult,
    apply_fixed_eight_locators,
)


SCHEMA_VERSION = "V9_017_FIXED_EIGHT_NO_NETWORK_RUNNER_V1"
ROOT_ID = "V9_014_SOURCE_B_ARCHIVE_ROOT"
ROOT_BYTE_COUNT = 75185
ROOT_SHA256 = "2e839c60bfb9d6edb59a903a590a505130124b8380b096ae84b06e4b0972098c"
REQUIRED_YEARS = (2017, 2019, 2020, 2022, 2026)
YEAR_PAGE_TARGET_COUNT = 5
FIXED_IDENTITY_COUNT = 8
ROOT_NETWORK_REQUESTS = 0
CONFIRMATION_CONTRACT = "V9_017_FIXED_EIGHT_NO_NETWORK"

YEAR_PAGE_BINDINGS: dict[int, tuple[int, str]] = {
    2017: (
        98936,
        "1dc982e97b1d4ce7d52bc25631ddc46a219d22f82b393881f40e5d2478177821",
    ),
    2019: (
        98954,
        "039814a2a90c47825ecc132b87c6e013943c02e21d607ac4725c58e66971471f",
    ),
    2020: (
        98990,
        "4f47da1c8be1a58cbd8998aca0261ec958ccf6212dc450863c47ab930a74f40a",
    ),
    2022: (
        98976,
        "89f715d5e63fd2486cdddde69ea481065c9b97ec3daef6db615dda508471ee4f",
    ),
    2026: (ROOT_BYTE_COUNT, ROOT_SHA256),
}

GOVERNANCE_FAILURE = "GOVERNANCE_FAILURE"
DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"
_HEX_40 = re.compile(r"[0-9a-fA-F]{40}\Z")


def _safe_payload(
    *,
    status: str,
    failure_class: str | None,
    reason: str | None,
    expected_git_sha: str | None,
    root_byte_count: int | None = None,
    root_sha256: str | None = None,
    root_binding_verified: bool = False,
    year_page_binding_count: int = 0,
    parent_year_url_binding_count: int = 0,
    fixed_eight_invocations: int = 0,
    application_status: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "failure_class": failure_class,
        "reason": reason,
        "expected_git_sha": expected_git_sha,
        "root_id": ROOT_ID,
        "root_byte_count": root_byte_count,
        "root_sha256": root_sha256,
        "root_binding_verified": root_binding_verified,
        "root_network_requests": ROOT_NETWORK_REQUESTS,
        "year_page_target_count": YEAR_PAGE_TARGET_COUNT,
        "year_page_binding_count": year_page_binding_count,
        "parent_year_url_binding_count": parent_year_url_binding_count,
        "fixed_identity_count": FIXED_IDENTITY_COUNT,
        "fixed_eight_invocations": fixed_eight_invocations,
        "application_status": application_status,
    }


def _valid_git_sha(value: object) -> bool:
    return isinstance(value, str) and _HEX_40.fullmatch(value) is not None


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _write_failure(output_root: Path | None, payload: Mapping[str, Any]) -> None:
    if output_root is None:
        return
    try:
        _write_exclusive(output_root / "failure.json", payload)
    except Exception:
        return


def _fail(
    output_root: Path | None,
    *,
    failure_class: str,
    reason: str,
    expected_git_sha: str | None,
    **kwargs: Any,
) -> dict[str, Any]:
    payload = _safe_payload(
        status="FAIL",
        failure_class=failure_class,
        reason=reason,
        expected_git_sha=expected_git_sha,
        **kwargs,
    )
    _write_failure(output_root, payload)
    return payload


def _read_once(input_path: str) -> bytes | None:
    try:
        return Path(input_path).read_bytes()
    except Exception:
        return None


def _attempt_metadata(expected_git_sha: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "IN_PROGRESS",
        "expected_git_sha": expected_git_sha,
        "confirmation_contract": CONFIRMATION_CONTRACT,
        "root_id": ROOT_ID,
        "root_byte_count": ROOT_BYTE_COUNT,
        "root_sha256": ROOT_SHA256,
        "root_network_requests": ROOT_NETWORK_REQUESTS,
        "target_years": list(REQUIRED_YEARS),
        "target_year_page_count": YEAR_PAGE_TARGET_COUNT,
        "fixed_identity_count": FIXED_IDENTITY_COUNT,
        "target_fixed_eight_invocations": 1,
    }


def _run_with_expected_bindings(
    preserved_root: str,
    year_page_paths: Mapping[int, str],
    output_root: str,
    *,
    expected_git_sha: str,
    confirmation: str,
    expected_root_byte_count: int,
    expected_root_sha256: str,
    expected_year_bindings: Mapping[int, tuple[int, str]],
    application: Callable[..., FixedEightLocatorApplicationResult] | None = None,
) -> dict[str, Any]:
    """Internal runner seam with immutable production defaults supplied by caller."""

    if not _valid_git_sha(expected_git_sha):
        return _fail(
            None,
            failure_class=GOVERNANCE_FAILURE,
            reason="EXPECTED_GIT_SHA_INVALID",
            expected_git_sha=expected_git_sha,
        )
    if confirmation != CONFIRMATION_CONTRACT:
        return _fail(
            None,
            failure_class=GOVERNANCE_FAILURE,
            reason="CONFIRMATION_CONTRACT_MISMATCH",
            expected_git_sha=expected_git_sha,
        )
    if not isinstance(preserved_root, str) or not preserved_root:
        return _fail(
            None,
            failure_class=GOVERNANCE_FAILURE,
            reason="INPUT_ARGUMENTS_REQUIRED",
            expected_git_sha=expected_git_sha,
        )
    if not isinstance(output_root, str) or not output_root:
        return _fail(
            None,
            failure_class=GOVERNANCE_FAILURE,
            reason="OUTPUT_ROOT_INVALID",
            expected_git_sha=expected_git_sha,
        )
    if not isinstance(year_page_paths, Mapping) or any(
        not isinstance(year_page_paths.get(year), str) or not year_page_paths.get(year)
        for year in REQUIRED_YEARS
    ):
        return _fail(
            None,
            failure_class=GOVERNANCE_FAILURE,
            reason="INPUT_ARGUMENTS_REQUIRED",
            expected_git_sha=expected_git_sha,
        )

    try:
        output_path = Path(output_root)
    except Exception:
        return _fail(
            None,
            failure_class=GOVERNANCE_FAILURE,
            reason="OUTPUT_ROOT_INVALID",
            expected_git_sha=expected_git_sha,
        )
    if output_path.exists():
        return _fail(
            None,
            failure_class=GOVERNANCE_FAILURE,
            reason="OUTPUT_ROOT_COLLISION",
            expected_git_sha=expected_git_sha,
        )

    try:
        output_path.mkdir(parents=True, exist_ok=False)
    except Exception:
        return _fail(
            None,
            failure_class=IMPLEMENTATION_FAILURE,
            reason="OUTPUT_ROOT_CREATE_FAILURE",
            expected_git_sha=expected_git_sha,
        )

    try:
        _write_exclusive(output_path / "attempt.json", _attempt_metadata(expected_git_sha))
    except Exception:
        return _fail(
            output_path,
            failure_class=IMPLEMENTATION_FAILURE,
            reason="ATTEMPT_METADATA_WRITE_FAILURE",
            expected_git_sha=expected_git_sha,
        )

    root_bytes = _read_once(preserved_root)
    if root_bytes is None:
        return _fail(
            output_path,
            failure_class=GOVERNANCE_FAILURE,
            reason="PRESERVED_ROOT_READ_FAILURE",
            expected_git_sha=expected_git_sha,
        )
    observed_root_count = len(root_bytes)
    observed_root_sha = hashlib.sha256(root_bytes).hexdigest()
    if observed_root_count != expected_root_byte_count:
        return _fail(
            output_path,
            failure_class=GOVERNANCE_FAILURE,
            reason="PRESERVED_ROOT_BYTE_COUNT_MISMATCH",
            expected_git_sha=expected_git_sha,
            root_byte_count=observed_root_count,
            root_sha256=observed_root_sha,
        )
    if observed_root_sha != expected_root_sha256:
        return _fail(
            output_path,
            failure_class=GOVERNANCE_FAILURE,
            reason="PRESERVED_ROOT_SHA256_MISMATCH",
            expected_git_sha=expected_git_sha,
            root_byte_count=observed_root_count,
            root_sha256=observed_root_sha,
        )

    try:
        root_candidates = extract_option_value_root_year_candidates(
            root_bytes, SOURCE_B_ARCHIVE_ROOT
        )
    except Exception:
        return _fail(
            output_path,
            failure_class=DATA_QUALITY_FAILURE,
            reason="ROOT_YEAR_BINDING_FAILURE",
            expected_git_sha=expected_git_sha,
            root_byte_count=observed_root_count,
            root_sha256=observed_root_sha,
            root_binding_verified=True,
        )

    parent_year_urls: dict[int, str] = {}
    for year in REQUIRED_YEARS:
        try:
            resolved = resolve_source_b_year_page(root_candidates, year)
        except Exception:
            resolved = None
        if resolved is None or resolved.status != LOCATOR_OK or not isinstance(resolved.url, str):
            return _fail(
                output_path,
                failure_class=DATA_QUALITY_FAILURE,
                reason="ROOT_YEAR_BINDING_FAILURE",
                expected_git_sha=expected_git_sha,
                root_byte_count=observed_root_count,
                root_sha256=observed_root_sha,
                root_binding_verified=True,
                parent_year_url_binding_count=len(parent_year_urls),
            )
        parent_year_urls[year] = resolved.url

    year_page_bytes: dict[int, bytes] = {}
    for year in REQUIRED_YEARS:
        page_bytes = _read_once(year_page_paths[year])
        if page_bytes is None:
            return _fail(
                output_path,
                failure_class=GOVERNANCE_FAILURE,
                reason="YEAR_PAGE_READ_FAILURE",
                expected_git_sha=expected_git_sha,
                root_byte_count=observed_root_count,
                root_sha256=observed_root_sha,
                root_binding_verified=True,
                year_page_binding_count=len(year_page_bytes),
                parent_year_url_binding_count=len(parent_year_urls),
            )
        expected_count, expected_sha = expected_year_bindings[year]
        observed_count = len(page_bytes)
        observed_sha = hashlib.sha256(page_bytes).hexdigest()
        if observed_count != expected_count:
            return _fail(
                output_path,
                failure_class=GOVERNANCE_FAILURE,
                reason="YEAR_PAGE_BYTE_COUNT_MISMATCH",
                expected_git_sha=expected_git_sha,
                root_byte_count=observed_root_count,
                root_sha256=observed_root_sha,
                root_binding_verified=True,
                year_page_binding_count=len(year_page_bytes),
                parent_year_url_binding_count=len(parent_year_urls),
            )
        if observed_sha != expected_sha:
            return _fail(
                output_path,
                failure_class=GOVERNANCE_FAILURE,
                reason="YEAR_PAGE_SHA256_MISMATCH",
                expected_git_sha=expected_git_sha,
                root_byte_count=observed_root_count,
                root_sha256=observed_root_sha,
                root_binding_verified=True,
                year_page_binding_count=len(year_page_bytes),
                parent_year_url_binding_count=len(parent_year_urls),
            )
        year_page_bytes[year] = page_bytes

    try:
        application_result = (application or apply_fixed_eight_locators)(
            year_page_bytes,
            parent_year_urls,
        )
    except Exception:
        return _fail(
            output_path,
            failure_class=IMPLEMENTATION_FAILURE,
            reason="FIXED_EIGHT_APPLICATION_FAILURE",
            expected_git_sha=expected_git_sha,
            root_byte_count=observed_root_count,
            root_sha256=observed_root_sha,
            root_binding_verified=True,
            year_page_binding_count=len(year_page_bytes),
            parent_year_url_binding_count=len(parent_year_urls),
            fixed_eight_invocations=1,
        )

    if not isinstance(application_result, FixedEightLocatorApplicationResult):
        return _fail(
            output_path,
            failure_class=IMPLEMENTATION_FAILURE,
            reason="FIXED_EIGHT_APPLICATION_FAILURE",
            expected_git_sha=expected_git_sha,
            root_byte_count=observed_root_count,
            root_sha256=observed_root_sha,
            root_binding_verified=True,
            year_page_binding_count=len(year_page_bytes),
            parent_year_url_binding_count=len(parent_year_urls),
            fixed_eight_invocations=1,
        )

    if application_result.status != "PASS":
        return _fail(
            output_path,
            failure_class=DATA_QUALITY_FAILURE,
            reason="FIXED_EIGHT_LOCATOR_FAILURE",
            expected_git_sha=expected_git_sha,
            root_byte_count=observed_root_count,
            root_sha256=observed_root_sha,
            root_binding_verified=True,
            year_page_binding_count=len(year_page_bytes),
            parent_year_url_binding_count=len(parent_year_urls),
            fixed_eight_invocations=1,
            application_status=application_result.status,
        )

    result = _safe_payload(
        status="PASS",
        failure_class=None,
        reason=None,
        expected_git_sha=expected_git_sha,
        root_byte_count=observed_root_count,
        root_sha256=observed_root_sha,
        root_binding_verified=True,
        year_page_binding_count=len(year_page_bytes),
        parent_year_url_binding_count=len(parent_year_urls),
        fixed_eight_invocations=1,
        application_status=application_result.status,
    )
    result["fixed_eight_result"] = application_result.to_dict()
    try:
        _write_exclusive(output_path / "result.json", result)
    except Exception:
        return _fail(
            output_path,
            failure_class=IMPLEMENTATION_FAILURE,
            reason="RESULT_WRITE_FAILURE",
            expected_git_sha=expected_git_sha,
            root_byte_count=observed_root_count,
            root_sha256=observed_root_sha,
            root_binding_verified=True,
            year_page_binding_count=len(year_page_bytes),
            parent_year_url_binding_count=len(parent_year_urls),
            fixed_eight_invocations=1,
            application_status=application_result.status,
        )
    try:
        _write_exclusive(
            output_path / "complete.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "FIXED_EIGHT_COMPLETE",
                "fixed_eight_invocations": 1,
            },
        )
    except Exception:
        return _fail(
            output_path,
            failure_class=IMPLEMENTATION_FAILURE,
            reason="COMPLETE_WRITE_FAILURE",
            expected_git_sha=expected_git_sha,
            root_byte_count=observed_root_count,
            root_sha256=observed_root_sha,
            root_binding_verified=True,
            year_page_binding_count=len(year_page_bytes),
            parent_year_url_binding_count=len(parent_year_urls),
            fixed_eight_invocations=1,
            application_status=application_result.status,
        )
    return result


def run_fixed_eight_no_network(
    preserved_root: str,
    year_page_paths: Mapping[int, str],
    output_root: str,
    *,
    expected_git_sha: str,
    confirmation: str,
) -> dict[str, Any]:
    """Run using the immutable production root/year-page bindings."""

    return _run_with_expected_bindings(
        preserved_root,
        year_page_paths,
        output_root,
        expected_git_sha=expected_git_sha,
        confirmation=confirmation,
        expected_root_byte_count=ROOT_BYTE_COUNT,
        expected_root_sha256=ROOT_SHA256,
        expected_year_bindings=YEAR_PAGE_BINDINGS,
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V9_017 fixed-eight binding without network.")
    parser.add_argument("--preserved-root")
    parser.add_argument("--year-page-2017")
    parser.add_argument("--year-page-2019")
    parser.add_argument("--year-page-2020")
    parser.add_argument("--year-page-2022")
    parser.add_argument("--year-page-2026")
    parser.add_argument("--output-root")
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--confirmation")
    parser.add_argument("--execute-fixed-eight", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _argument_parser().parse_args(argv)
    if not arguments.execute_fixed_eight:
        print(
            json.dumps(
                _safe_payload(
                    status="FAIL",
                    failure_class=GOVERNANCE_FAILURE,
                    reason="EXECUTE_FIXED_EIGHT_REQUIRED",
                    expected_git_sha=arguments.expected_git_sha,
                ),
                sort_keys=True,
            )
        )
        return 1

    year_page_paths = {
        2017: arguments.year_page_2017,
        2019: arguments.year_page_2019,
        2020: arguments.year_page_2020,
        2022: arguments.year_page_2022,
        2026: arguments.year_page_2026,
    }
    try:
        result = run_fixed_eight_no_network(
            arguments.preserved_root,
            year_page_paths,
            arguments.output_root,
            expected_git_sha=arguments.expected_git_sha,
            confirmation=arguments.confirmation,
        )
    except Exception:
        result = _safe_payload(
            status="FAIL",
            failure_class=IMPLEMENTATION_FAILURE,
            reason="RUNNER_FAILURE",
            expected_git_sha=arguments.expected_git_sha,
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


__all__ = [
    "CONFIRMATION_CONTRACT",
    "FIXED_IDENTITY_COUNT",
    "ROOT_BYTE_COUNT",
    "ROOT_ID",
    "ROOT_NETWORK_REQUESTS",
    "ROOT_SHA256",
    "REQUIRED_YEARS",
    "SCHEMA_VERSION",
    "YEAR_PAGE_BINDINGS",
    "main",
    "run_fixed_eight_no_network",
]


if __name__ == "__main__":
    raise SystemExit(main())
