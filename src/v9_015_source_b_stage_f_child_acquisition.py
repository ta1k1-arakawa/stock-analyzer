"""V9_015 Stage-F child-acquisition boundary.

The public runner in this module starts from caller-supplied preserved root
bytes or a caller-supplied preserved-root path.  It verifies the frozen root
identity, binds the five year pages with the reviewed V9_015 Stage-E
extractor, and acquires only the five year pages plus the fixed eight
calibration objects.  V9_014's reviewed transport, retry, redirect, URL,
locking, archive-parser, and locator mechanics are composed directly.

This module does not fetch or write a root payload, does not request the root,
does not invoke a PDF structural probe, and does not perform any semantic,
calendar, T0, model, or profitability work.  The default transport is only
reachable from a future explicitly flagged/confirmed real invocation; the
tests for this task inject a synthetic transport.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

from src import v9_014_jpx_monthly_auction_activity_source_b_archive_parser as _archive_parser
from src import v9_014_jpx_monthly_auction_activity_source_b_calibration_acquisition as _acquisition
from src import v9_014_jpx_monthly_auction_activity_source_b_locator as _locator
from src.v9_005_stage_a_jpx_probe import V9005StageABlocked
from src.v9_015_source_b_option_value_root_extractor import (
    extract_option_value_root_year_candidates,
)

ROOT_SHA256 = "2e839c60bfb9d6edb59a903a590a505130124b8380b096ae84b06e4b0972098c"
ROOT_BYTE_COUNT = 75185
REQUIRED_YEAR_LABELS = ("2017", "2019", "2020", "2022", "2026")
YEAR_PAGE_TARGET_COUNT = 5
CALIBRATION_PDF_TARGET_COUNT = 8
CHILD_LOCK_TARGET_COUNT = 13
ROOT_NETWORK_REQUESTS = 0
PROBE_INVOCATIONS = 0
STAGE_F_CONFIRMATION_CONTRACT = "V9_015_STAGE_F_CHILD_ACQUISITION"

STAGE_F_PASS = "STAGE_F_CHILD_ACQUISITION_PASS"
STAGE_F_FAILURE = "STAGE_F_CHILD_ACQUISITION_FAILURE"
GOVERNANCE_FAILURE = "GOVERNANCE_FAILURE"
DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"

Transport = Callable[[str, int], _acquisition.TransportResponse]


@dataclass(frozen=True)
class StageFResult:
    status: str
    failure_class: Optional[str] = None
    reason: Optional[str] = None
    preserved_root_sha256: Optional[str] = None
    preserved_root_byte_count: Optional[int] = None
    preserved_root_binding_verified: bool = False
    root_network_requests: int = ROOT_NETWORK_REQUESTS
    child_network_requests: int = 0
    year_page_lock_count: int = 0
    calibration_pdf_lock_count: int = 0
    child_lock_count: int = 0
    probe_invocations: int = PROBE_INVOCATIONS
    locked_payloads: tuple[_acquisition.LockedPayload, ...] = ()

    def to_safe_dict(self) -> dict[str, object]:
        """Project only approved hashes, counts, roles, and identities."""

        return {
            "status": self.status,
            "failure_class": self.failure_class,
            "reason": self.reason,
            "preserved_root_sha256": self.preserved_root_sha256,
            "preserved_root_byte_count": self.preserved_root_byte_count,
            "preserved_root_binding_verified": self.preserved_root_binding_verified,
            "root_network_requests": self.root_network_requests,
            "child_network_requests": self.child_network_requests,
            "year_page_lock_count": self.year_page_lock_count,
            "calibration_pdf_lock_count": self.calibration_pdf_lock_count,
            "child_lock_count": self.child_lock_count,
            "probe_invocations": self.probe_invocations,
            "locked_payloads": [asdict(item) for item in self.locked_payloads],
        }


def _result(
    context: Optional[_acquisition._RunContext],
    *,
    status: str,
    failure_class: Optional[str] = None,
    reason: Optional[str] = None,
    root_sha256: Optional[str] = None,
    root_byte_count: Optional[int] = None,
    root_binding_verified: bool = False,
    child_network_requests: int = 0,
    year_page_lock_count: int = 0,
    calibration_pdf_count: int = 0,
) -> StageFResult:
    locked_payloads = () if context is None else tuple(context.locked)
    return StageFResult(
        status=status,
        failure_class=failure_class,
        reason=reason,
        preserved_root_sha256=root_sha256,
        preserved_root_byte_count=root_byte_count,
        preserved_root_binding_verified=root_binding_verified,
        child_network_requests=child_network_requests,
        year_page_lock_count=year_page_lock_count,
        calibration_pdf_lock_count=calibration_pdf_count,
        child_lock_count=len(locked_payloads),
        locked_payloads=locked_payloads,
    )


def _persist_result(context: _acquisition._RunContext, filename: str, value: dict[str, object]) -> bool:
    try:
        _acquisition._write_json_exclusive(context.output_root / filename, value)
    except Exception:
        return False
    return True


def _failure(
    context: Optional[_acquisition._RunContext],
    failure_class: str,
    reason: str,
    *,
    root_sha256: Optional[str] = None,
    root_byte_count: Optional[int] = None,
    root_binding_verified: bool = False,
    child_network_requests: int = 0,
    year_page_lock_count: int = 0,
    calibration_pdf_count: int = 0,
) -> StageFResult:
    result = _result(
        context,
        status=STAGE_F_FAILURE,
        failure_class=failure_class,
        reason=reason,
        root_sha256=root_sha256,
        root_byte_count=root_byte_count,
        root_binding_verified=root_binding_verified,
        child_network_requests=child_network_requests,
        year_page_lock_count=year_page_lock_count,
        calibration_pdf_count=calibration_pdf_count,
    )
    if context is not None and not _persist_result(context, "failure.json", result.to_safe_dict()):
        return _result(
            context,
            status=STAGE_F_FAILURE,
            failure_class=IMPLEMENTATION_FAILURE,
            reason="SAFE_RECEIPT_WRITE_FAILURE",
            root_sha256=root_sha256,
            root_byte_count=root_byte_count,
            root_binding_verified=root_binding_verified,
            child_network_requests=child_network_requests,
            year_page_lock_count=year_page_lock_count,
            calibration_pdf_count=calibration_pdf_count,
        )
    return result


def _valid_git_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _read_preserved_root(preserved_root: object) -> tuple[Optional[bytes], Optional[StageFResult]]:
    if isinstance(preserved_root, bytes):
        return preserved_root, None
    try:
        root_path = Path(preserved_root)
        if not root_path.is_file():
            return None, _failure(None, GOVERNANCE_FAILURE, "PRESERVED_ROOT_READ_FAILURE")
        return root_path.read_bytes(), None
    except Exception:
        return None, _failure(None, GOVERNANCE_FAILURE, "PRESERVED_ROOT_READ_FAILURE")


def _locator_url(value: object) -> Optional[str]:
    return value.url if getattr(value, "status", None) == _locator.LOCATOR_OK else None


def _parse_failure(exc: BaseException, fallback_reason: str) -> tuple[str, str]:
    if isinstance(exc, V9005StageABlocked):
        return DATA_QUALITY_FAILURE, fallback_reason
    return IMPLEMENTATION_FAILURE, fallback_reason


def _write_attempt_metadata(context: _acquisition._RunContext, expected_git_sha: str) -> bool:
    return _persist_result(
        context,
        "attempt.json",
        {
            "schema_version": 1,
            "status": "IN_PROGRESS",
            "expected_git_sha": expected_git_sha,
            "confirmation_contract": STAGE_F_CONFIRMATION_CONTRACT,
            "preserved_root_sha256": ROOT_SHA256,
            "preserved_root_byte_count": ROOT_BYTE_COUNT,
            "root_network_requests": ROOT_NETWORK_REQUESTS,
            "target_year_page_count": YEAR_PAGE_TARGET_COUNT,
            "target_calibration_pdf_count": CALIBRATION_PDF_TARGET_COUNT,
            "target_child_lock_count": CHILD_LOCK_TARGET_COUNT,
            "probe_invocations": PROBE_INVOCATIONS,
        },
    )


def run_stage_f_child_acquisition(
    preserved_root: bytes | Path | str,
    output_root: Path | str,
    *,
    expected_git_sha: str,
    confirmation: str,
    transport: Optional[Transport] = None,
) -> StageFResult:
    """Run the Stage-F child boundary with injected or future real transport.

    ``preserved_root`` may be synthetic bytes for offline tests or a
    caller-supplied preserved-root path for a future authorized run.  The
    path is never written to metadata or returned in safe output.
    """

    try:
        output_path = Path(output_root)
    except Exception:
        return _failure(None, GOVERNANCE_FAILURE, "OUTPUT_ROOT_INVALID")
    if output_path.exists():
        return _failure(None, GOVERNANCE_FAILURE, "OUTPUT_ROOT_COLLISION")
    if not _valid_git_sha(expected_git_sha):
        return _failure(None, GOVERNANCE_FAILURE, "EXPECTED_GIT_SHA_INVALID")
    if confirmation != STAGE_F_CONFIRMATION_CONTRACT:
        return _failure(None, GOVERNANCE_FAILURE, "CONFIRMATION_CONTRACT_MISMATCH")

    try:
        output_path.mkdir(parents=True, exist_ok=False)
        context = _acquisition._RunContext(output_path)
    except Exception:
        return _failure(None, IMPLEMENTATION_FAILURE, "OUTPUT_ROOT_CREATE_FAILURE")
    if not _write_attempt_metadata(context, expected_git_sha):
        return _failure(context, IMPLEMENTATION_FAILURE, "ATTEMPT_METADATA_WRITE_FAILURE")

    root_bytes, root_error = _read_preserved_root(preserved_root)
    if root_error is not None or root_bytes is None:
        return _failure(context, GOVERNANCE_FAILURE, "PRESERVED_ROOT_READ_FAILURE")
    observed_root_sha256 = hashlib.sha256(root_bytes).hexdigest()
    observed_root_byte_count = len(root_bytes)
    if observed_root_sha256 != ROOT_SHA256:
        return _failure(
            context,
            GOVERNANCE_FAILURE,
            "PRESERVED_ROOT_SHA256_MISMATCH",
            root_sha256=observed_root_sha256,
            root_byte_count=observed_root_byte_count,
        )
    if observed_root_byte_count != ROOT_BYTE_COUNT:
        return _failure(
            context,
            GOVERNANCE_FAILURE,
            "PRESERVED_ROOT_BYTE_COUNT_MISMATCH",
            root_sha256=observed_root_sha256,
            root_byte_count=observed_root_byte_count,
        )

    try:
        root_candidates = extract_option_value_root_year_candidates(
            root_bytes, _locator.SOURCE_B_ARCHIVE_ROOT
        )
    except Exception as exc:
        failure_class, failure_reason = _parse_failure(exc, "ROOT_LOCATOR_FAILURE")
        return _failure(
            context,
            failure_class,
            failure_reason,
            root_sha256=observed_root_sha256,
            root_byte_count=observed_root_byte_count,
            root_binding_verified=True,
        )

    child_request_count = 0

    def counted_transport(url: str, timeout: int) -> _acquisition.TransportResponse:
        nonlocal child_request_count
        child_request_count += 1
        request = transport or _acquisition._default_transport
        return request(url, timeout)

    year_urls: dict[int, str] = {}
    year_bodies: dict[int, bytes] = {}
    selected_child_urls: set[str] = set()
    for year_label in REQUIRED_YEAR_LABELS:
        year = int(year_label)
        try:
            selected = _locator_url(_locator.resolve_source_b_year_page(root_candidates, year))
        except Exception as exc:
            failure_class, failure_reason = _parse_failure(exc, "ROOT_LOCATOR_FAILURE")
            return _failure(
                context,
                failure_class,
                failure_reason,
                root_sha256=observed_root_sha256,
                root_byte_count=observed_root_byte_count,
                root_binding_verified=True,
                child_network_requests=child_request_count,
                year_page_lock_count=len(year_urls),
            )
        if selected is None:
            return _failure(
                context,
                DATA_QUALITY_FAILURE,
                "ROOT_LOCATOR_FAILURE",
                root_sha256=observed_root_sha256,
                root_byte_count=observed_root_byte_count,
                root_binding_verified=True,
                child_network_requests=child_request_count,
                year_page_lock_count=len(year_urls),
            )
        if selected in selected_child_urls:
            return _failure(
                context,
                DATA_QUALITY_FAILURE,
                "DUPLICATE_YEAR_PAYLOAD_URL",
                root_sha256=observed_root_sha256,
                root_byte_count=observed_root_byte_count,
                root_binding_verified=True,
                child_network_requests=child_request_count,
                year_page_lock_count=len(year_urls),
            )
        year_urls[year] = selected
        locked, failure = _acquisition.fetch_and_lock_payload(
            context,
            selected,
            role="year_page",
            relative_path=f"{_acquisition.YEAR_LOCK_DIRECTORY}/{year}.html",
            transport=counted_transport,
            identity=str(year),
        )
        if failure is not None or locked is None:
            failure_class, failure_reason = failure or (
                IMPLEMENTATION_FAILURE,
                "YEAR_PAGE_LOCK_FAILURE",
            )
            return _failure(
                context,
                failure_class,
                failure_reason,
                root_sha256=observed_root_sha256,
                root_byte_count=observed_root_byte_count,
                root_binding_verified=True,
                child_network_requests=child_request_count,
                year_page_lock_count=len(year_urls) - 1,
            )
        year_bodies[year] = context.payload_bytes[selected]
        selected_child_urls.add(selected)

    seen_pdf_urls: set[str] = set()
    calibration_pdf_count = 0
    for index, identity in enumerate(_acquisition.REQUIRED_CALIBRATION_IDENTITIES, start=1):
        year = int(identity.logical_month[:4])
        try:
            if identity.object_part == _acquisition.PRE_APRIL_1_REFERENCE_OBJECT:
                candidates = _archive_parser.extract_april_pre_candidates(
                    year_bodies[year], year_urls[year], selected_year=year
                )
                selected = _locator_url(
                    _locator.resolve_source_b_april_pre_object(
                        candidates, selected_year_page_url=year_urls[year]
                    )
                )
            else:
                candidates = _archive_parser.extract_normal_month_candidates(
                    year_bodies[year], year_urls[year], identity.logical_month,
                    selected_year=year,
                )
                selected = _locator_url(
                    _locator.resolve_source_b_normal_month_object(
                        candidates,
                        identity.logical_month,
                        selected_year_page_url=year_urls[year],
                    )
                )
        except Exception as exc:
            failure_class, failure_reason = _parse_failure(exc, "PDF_LOCATOR_FAILURE")
            return _failure(
                context,
                failure_class,
                failure_reason,
                root_sha256=observed_root_sha256,
                root_byte_count=observed_root_byte_count,
                root_binding_verified=True,
                child_network_requests=child_request_count,
                year_page_lock_count=len(year_urls),
                calibration_pdf_count=calibration_pdf_count,
            )
        if selected is None:
            return _failure(
                context,
                DATA_QUALITY_FAILURE,
                "PDF_LOCATOR_FAILURE",
                root_sha256=observed_root_sha256,
                root_byte_count=observed_root_byte_count,
                root_binding_verified=True,
                child_network_requests=child_request_count,
                year_page_lock_count=len(year_urls),
                calibration_pdf_count=calibration_pdf_count,
            )
        if selected in seen_pdf_urls or selected in selected_child_urls:
            return _failure(
                context,
                DATA_QUALITY_FAILURE,
                "DUPLICATE_CALIBRATION_PDF_URL",
                root_sha256=observed_root_sha256,
                root_byte_count=observed_root_byte_count,
                root_binding_verified=True,
                child_network_requests=child_request_count,
                year_page_lock_count=len(year_urls),
                calibration_pdf_count=calibration_pdf_count,
            )
        seen_pdf_urls.add(selected)
        locked, failure = _acquisition.fetch_and_lock_payload(
            context,
            selected,
            role="calibration_pdf",
            relative_path=(
                f"{_acquisition.PDF_LOCK_DIRECTORY}/{index:02d}_"
                f"{identity.logical_month}_{identity.object_part}.pdf"
            ),
            transport=counted_transport,
            identity=_acquisition._identity_text(identity),
        )
        if failure is not None or locked is None:
            failure_class, failure_reason = failure or (
                IMPLEMENTATION_FAILURE,
                "CALIBRATION_PDF_LOCK_FAILURE",
            )
            return _failure(
                context,
                failure_class,
                failure_reason,
                root_sha256=observed_root_sha256,
                root_byte_count=observed_root_byte_count,
                root_binding_verified=True,
                child_network_requests=child_request_count,
                year_page_lock_count=len(year_urls),
                calibration_pdf_count=calibration_pdf_count,
            )
        calibration_pdf_count += 1
        selected_child_urls.add(selected)

    result = _result(
        context,
        status=STAGE_F_PASS,
        root_sha256=observed_root_sha256,
        root_byte_count=observed_root_byte_count,
        root_binding_verified=True,
        child_network_requests=child_request_count,
        year_page_lock_count=len(year_urls),
        calibration_pdf_count=calibration_pdf_count,
    )
    if not _persist_result(context, "receipt.json", result.to_safe_dict()) or not _persist_result(
        context,
        "complete.json",
        {
            "schema_version": 1,
            "status": result.status,
            "root_network_requests": result.root_network_requests,
            "child_network_requests": result.child_network_requests,
            "year_page_lock_count": result.year_page_lock_count,
            "calibration_pdf_lock_count": result.calibration_pdf_lock_count,
            "child_lock_count": result.child_lock_count,
            "probe_invocations": result.probe_invocations,
        },
    ):
        return _failure(
            context,
            IMPLEMENTATION_FAILURE,
            "SAFE_RECEIPT_WRITE_FAILURE",
            root_sha256=observed_root_sha256,
            root_byte_count=observed_root_byte_count,
            root_binding_verified=True,
            child_network_requests=child_request_count,
            year_page_lock_count=len(year_urls),
            calibration_pdf_count=calibration_pdf_count,
        )
    return result


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V9_015 Stage-F SOURCE_B child acquisition runner")
    parser.add_argument("--preserved-root-path")
    parser.add_argument("--output-root")
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--confirmation")
    parser.add_argument("--production-acquire", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    arguments = _cli_parser().parse_args(argv)
    if not arguments.production_acquire:
        print(json.dumps({
            "status": STAGE_F_FAILURE,
            "failure_class": GOVERNANCE_FAILURE,
            "reason": "PRODUCTION_CONFIRMATION_FLAG_REQUIRED",
        }, sort_keys=True))
        return 1
    if not all((
        arguments.preserved_root_path,
        arguments.output_root,
        arguments.expected_git_sha,
        arguments.confirmation,
    )):
        print(json.dumps({
            "status": STAGE_F_FAILURE,
            "failure_class": GOVERNANCE_FAILURE,
            "reason": "PRODUCTION_ARGUMENTS_REQUIRED",
        }, sort_keys=True))
        return 1
    try:
        result = run_stage_f_child_acquisition(
            arguments.preserved_root_path,
            arguments.output_root,
            expected_git_sha=arguments.expected_git_sha,
            confirmation=arguments.confirmation,
        )
    except Exception:
        print(json.dumps({
            "status": STAGE_F_FAILURE,
            "failure_class": IMPLEMENTATION_FAILURE,
            "reason": "RUNNER_FAILURE",
        }, sort_keys=True))
        return 1
    print(json.dumps(result.to_safe_dict(), sort_keys=True))
    return 0 if result.status == STAGE_F_PASS else 1


__all__ = [
    "ROOT_SHA256",
    "ROOT_BYTE_COUNT",
    "REQUIRED_YEAR_LABELS",
    "YEAR_PAGE_TARGET_COUNT",
    "CALIBRATION_PDF_TARGET_COUNT",
    "CHILD_LOCK_TARGET_COUNT",
    "ROOT_NETWORK_REQUESTS",
    "PROBE_INVOCATIONS",
    "STAGE_F_CONFIRMATION_CONTRACT",
    "STAGE_F_PASS",
    "STAGE_F_FAILURE",
    "GOVERNANCE_FAILURE",
    "DATA_QUALITY_FAILURE",
    "IMPLEMENTATION_FAILURE",
    "StageFResult",
    "run_stage_f_child_acquisition",
    "main",
]
