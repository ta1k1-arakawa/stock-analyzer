"""Future Stage-D acquisition runner for the fixed C1 calibration bundle.

The orchestration is deliberately dependency-injected so that this task can
test the complete order offline.  The default transport is present for the
later one-shot runner, but it is never selected by the tests in this module.
Successful bodies are locked byte-for-byte before any reviewed archive parser
or C1 probe is called.  Only hashes, counts, identities, and masked C1
structural evidence are emitted as safe receipt data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from src.v9_014_jpx_monthly_auction_activity_source_b_archive_parser import (
    extract_april_pre_candidates,
    extract_normal_month_candidates,
    extract_root_year_candidates,
)
from src.v9_014_jpx_monthly_auction_activity_source_b_locator import (
    LOCATOR_OK,
    MonthlyReportCandidate,
    RootYearCandidate,
    SOURCE_B_ARCHIVE_ROOT,
    resolve_source_b_april_pre_object,
    resolve_source_b_normal_month_object,
    resolve_source_b_year_page,
)
from src.v9_014_jpx_monthly_auction_activity_source_b_pdf_calibration_probe import (
    CALIBRATION_BUNDLE_PASS,
    CALIBRATION_OBJECT_COUNT,
    CalibrationBundleResult,
    CalibrationIdentity,
    CalibrationObjectInput,
    PRE_APRIL_1_REFERENCE_OBJECT,
    REQUIRED_CALIBRATION_IDENTITIES,
    probe_calibration_bundle,
)
from src.v9_005_stage_a_jpx_probe import V9005StageABlocked, validate_jpx_url


REQUEST_TIMEOUT_SECONDS = 30
MAX_PRE_COMPLETE_ATTEMPTS_PER_URL = 3
RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
V9_014_STAGE_D_REDIRECT_POLICY = "REJECT_ALL_HTTP_REDIRECTS"
ALLOWED_REDIRECT_COUNT = 0
REQUESTED_URL_MUST_EQUAL_RESOLVED_URL = True
REDIRECT_IS_RETRYABLE = False
CALIBRATION_YEARS = (2017, 2019, 2020, 2022, 2026)
EXPECTED_PAYLOAD_COUNT = 1 + len(CALIBRATION_YEARS) + CALIBRATION_OBJECT_COUNT
CONFIRMATION_CONTRACT = "V9_014_STAGE_D_FIXED_8_CALIBRATION_ACQUISITION"

ACQUISITION_PASS = "CALIBRATION_ACQUISITION_PASS"
ACQUISITION_FAILURE = "CALIBRATION_ACQUISITION_FAILURE"
PLUMBING_FAILURE_RETRIABLE = "PLUMBING_FAILURE_RETRIABLE"
DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
GOVERNANCE_FAILURE = "GOVERNANCE_FAILURE"
IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"
CHATGPT_DECISION_REQUIRED = "CHATGPT_DECISION_REQUIRED"

ROOT_LOCK_RELATIVE_PATH = "raw/archive_root.html"
YEAR_LOCK_DIRECTORY = "raw/year_pages"
PDF_LOCK_DIRECTORY = "raw/calibration_pdfs"


class RedirectRejectedError(Exception):
    """Internal safe signal for a redirect rejected before target access."""


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject every redirect status handled by urllib's redirect machinery."""

    @staticmethod
    def _reject(*_args: object, **_kwargs: object) -> None:
        raise RedirectRejectedError()

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return self._reject(*args, **kwargs)

    def http_error_301(self, *args: object, **kwargs: object) -> None:
        return self._reject(*args, **kwargs)

    def http_error_302(self, *args: object, **kwargs: object) -> None:
        return self._reject(*args, **kwargs)

    def http_error_303(self, *args: object, **kwargs: object) -> None:
        return self._reject(*args, **kwargs)

    def http_error_307(self, *args: object, **kwargs: object) -> None:
        return self._reject(*args, **kwargs)

    def http_error_308(self, *args: object, **kwargs: object) -> None:
        return self._reject(*args, **kwargs)

    def http_error_default(self, request: object, response: object, code: int, message: object, headers: object) -> None:
        if isinstance(code, int) and 300 <= code < 400:
            return self._reject(request, response, code, message, headers)
        return super().http_error_default(request, response, code, message, headers)


NO_REDIRECT_OPENER = urllib.request.build_opener(NoRedirectHandler())


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    body: bytes = b""
    resolved_url: Optional[str] = None


Transport = Callable[[str, int], TransportResponse]


@dataclass(frozen=True)
class LockedPayload:
    role: str
    relative_path: str
    sha256: str
    byte_count: int
    identity: Optional[str] = None


@dataclass(frozen=True)
class AcquisitionResult:
    status: str
    failure_class: Optional[str] = None
    reason: Optional[str] = None
    unique_complete_payload_count: int = 0
    year_page_count: int = 0
    calibration_pdf_count: int = 0
    probe_invocations: int = 0
    locked_payloads: tuple[LockedPayload, ...] = ()
    calibration_evidence: tuple[Mapping[str, Any], ...] = ()

    def to_safe_dict(self) -> dict[str, object]:
        """Return receipt data without bytes, paths, URLs, or exceptions."""

        return {
            "status": self.status,
            "failure_class": self.failure_class,
            "reason": self.reason,
            "unique_complete_payload_count": self.unique_complete_payload_count,
            "year_page_count": self.year_page_count,
            "calibration_pdf_count": self.calibration_pdf_count,
            "probe_invocations": self.probe_invocations,
            "locked_payloads": [asdict(item) for item in self.locked_payloads],
            "calibration_evidence": list(self.calibration_evidence),
        }


class _RunContext:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.locked_by_url: dict[str, LockedPayload] = {}
        self.payload_bytes: dict[str, bytes] = {}
        self.locked: list[LockedPayload] = []
        self.probe_invocations = 0


def _identity_text(identity: CalibrationIdentity) -> str:
    return f"{identity.logical_month}:{identity.object_part}"


def _write_exclusive(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    _write_exclusive(path, (json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8"))


def _failure(
    context: _RunContext,
    failure_class: str,
    reason: str,
    *,
    year_page_count: int = 0,
    calibration_pdf_count: int = 0,
    probe_invocations: Optional[int] = None,
) -> AcquisitionResult:
    result = AcquisitionResult(
        ACQUISITION_FAILURE,
        failure_class,
        reason,
        len(context.locked),
        year_page_count,
        calibration_pdf_count,
        context.probe_invocations if probe_invocations is None else probe_invocations,
        tuple(context.locked),
    )
    try:
        _write_json_exclusive(
            context.output_root / "failure.json",
            {
                "schema_version": 1,
                "status": result.status,
                "failure_class": failure_class,
                "reason": reason,
                "unique_complete_payload_count": len(context.locked),
            },
        )
    except Exception:
        # The original locks are still preserved; receipt failure is itself
        # represented only by the already safe in-memory result.
        return AcquisitionResult(
            ACQUISITION_FAILURE,
            IMPLEMENTATION_FAILURE,
            "SAFE_RECEIPT_WRITE_FAILURE",
            len(context.locked),
            year_page_count,
            calibration_pdf_count,
            context.probe_invocations,
            tuple(context.locked),
        )
    return result


def _response_status(response: object) -> Optional[int]:
    if isinstance(response, TransportResponse):
        return response.status_code
    value = getattr(response, "status_code", getattr(response, "status", None))
    return value if isinstance(value, int) else None


def _response_body(response: object) -> object:
    if isinstance(response, TransportResponse):
        return response.body
    return getattr(response, "body", None)


def _response_resolved_url(response: object) -> Optional[str]:
    value = getattr(response, "resolved_url", None)
    return value if isinstance(value, str) else None


def _lock_path(context: _RunContext, role: str, relative_path: str) -> Path:
    if role == "root":
        return context.output_root / ROOT_LOCK_RELATIVE_PATH
    return context.output_root / relative_path


def fetch_and_lock_payload(
    context: _RunContext,
    url: str,
    *,
    role: str,
    relative_path: str,
    transport: Transport,
    identity: Optional[str] = None,
) -> tuple[Optional[LockedPayload], Optional[tuple[str, str]]]:
    """Fetch one URL with bounded pre-lock retries, then lock its first 200."""

    try:
        validate_jpx_url(url)
    except V9005StageABlocked:
        return None, (DATA_QUALITY_FAILURE, "OFFICIAL_URL_VALIDATION_FAILURE")

    existing = context.locked_by_url.get(url)
    if existing is not None:
        return existing, None

    for _attempt in range(MAX_PRE_COMPLETE_ATTEMPTS_PER_URL):
        try:
            response = transport(url, REQUEST_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as error:
            if error.code in REDIRECT_STATUS_CODES:
                return None, (DATA_QUALITY_FAILURE, "HTTP_REDIRECT_REJECTED")
            if error.code in RETRYABLE_HTTP_STATUSES:
                continue
            return None, (DATA_QUALITY_FAILURE, "NONRETRYABLE_HTTP_STATUS")
        except RedirectRejectedError:
            if REDIRECT_IS_RETRYABLE:
                continue
            return None, (DATA_QUALITY_FAILURE, "HTTP_REDIRECT_REJECTED")
        except V9005StageABlocked:
            return None, (DATA_QUALITY_FAILURE, "OFFICIAL_TRANSPORT_VALIDATION_FAILURE")
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
        except Exception:
            return None, (IMPLEMENTATION_FAILURE, "TRANSPORT_IMPLEMENTATION_FAILURE")

        status = _response_status(response)
        if status == 200:
            resolved_url = _response_resolved_url(response)
            if resolved_url is not None:
                try:
                    validate_jpx_url(resolved_url, reason="OFF_DOMAIN_REDIRECT_REJECTED")
                except V9005StageABlocked:
                    return None, (DATA_QUALITY_FAILURE, "OFF_DOMAIN_REDIRECT_REJECTED")
                if REQUESTED_URL_MUST_EQUAL_RESOLVED_URL and resolved_url != url:
                    return None, (DATA_QUALITY_FAILURE, "RESOLVED_URL_MISMATCH")
            body = _response_body(response)
            if not isinstance(body, bytes):
                return None, (IMPLEMENTATION_FAILURE, "TRANSPORT_BODY_TYPE_FAILURE")
            digest = hashlib.sha256(body).hexdigest()
            path = _lock_path(context, role, relative_path)
            try:
                _write_exclusive(path, body)
            except Exception:
                return None, (IMPLEMENTATION_FAILURE, "RAW_LOCK_WRITE_FAILURE")
            locked = LockedPayload(role, relative_path, digest, len(body), identity)
            context.locked_by_url[url] = locked
            context.payload_bytes[url] = body
            context.locked.append(locked)
            return locked, None
        if status in REDIRECT_STATUS_CODES:
            return None, (DATA_QUALITY_FAILURE, "HTTP_REDIRECT_REJECTED")
        if status in RETRYABLE_HTTP_STATUSES:
            continue
        return None, (DATA_QUALITY_FAILURE, "NONRETRYABLE_HTTP_STATUS")
    return None, (PLUMBING_FAILURE_RETRIABLE, "PRE_COMPLETE_ATTEMPT_LIMIT")


def _default_transport(url: str, timeout: int) -> TransportResponse:
    """Later real-mode transport with all HTTP redirects rejected."""

    validate_jpx_url(url)
    request = urllib.request.Request(url, method="GET")
    with NO_REDIRECT_OPENER.open(request, timeout=timeout) as response:
        resolved_url = response.geturl()
        validate_jpx_url(resolved_url, reason="OFF_DOMAIN_REDIRECT_REJECTED")
        if REQUESTED_URL_MUST_EQUAL_RESOLVED_URL and resolved_url != url:
            raise RedirectRejectedError()
        status = response.status if isinstance(response.status, int) else response.getcode()
        body = response.read() if status == 200 else b""
    return TransportResponse(status, body, resolved_url)


def _locator_url(value: object) -> Optional[str]:
    return value.url if getattr(value, "status", None) == LOCATOR_OK else None


def _parse_failure(exc: BaseException, reason: str) -> tuple[str, str]:
    if isinstance(exc, V9005StageABlocked):
        return DATA_QUALITY_FAILURE, reason
    return IMPLEMENTATION_FAILURE, reason


def _persist_success(context: _RunContext, result: AcquisitionResult) -> None:
    _write_json_exclusive(context.output_root / "calibration_evidence.json", {
        "schema_version": 1,
        "evidence": list(result.calibration_evidence),
    })
    _write_json_exclusive(context.output_root / "receipt.json", result.to_safe_dict())
    _write_json_exclusive(context.output_root / "complete.json", {
        "schema_version": 1,
        "status": result.status,
        "unique_complete_payload_count": result.unique_complete_payload_count,
        "year_page_count": result.year_page_count,
        "calibration_pdf_count": result.calibration_pdf_count,
    })


def run_fixed_eight_calibration_acquisition(
    output_root: Path,
    *,
    expected_git_sha: str,
    confirmation: str,
    transport: Optional[Transport] = None,
) -> AcquisitionResult:
    """Run the future fixed-eight traversal against an injected transport."""

    output_root = Path(output_root)
    if output_root.exists():
        return AcquisitionResult(ACQUISITION_FAILURE, GOVERNANCE_FAILURE, "OUTPUT_ROOT_COLLISION")
    if not isinstance(expected_git_sha, str) or len(expected_git_sha) != 40 or any(
        character not in "0123456789abcdef" for character in expected_git_sha.lower()
    ):
        return AcquisitionResult(ACQUISITION_FAILURE, GOVERNANCE_FAILURE, "EXPECTED_GIT_SHA_INVALID")
    if confirmation != CONFIRMATION_CONTRACT:
        return AcquisitionResult(ACQUISITION_FAILURE, GOVERNANCE_FAILURE, "CONFIRMATION_CONTRACT_MISMATCH")

    context = _RunContext(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(output_root / "attempt.json", {
        "schema_version": 1,
        "status": "IN_PROGRESS",
        "expected_git_sha": expected_git_sha,
        "confirmation_contract": CONFIRMATION_CONTRACT,
        "target_unique_complete_payload_count": EXPECTED_PAYLOAD_COUNT,
    })
    request = transport or _default_transport

    root, failure = fetch_and_lock_payload(
        context, SOURCE_B_ARCHIVE_ROOT, role="root", relative_path=ROOT_LOCK_RELATIVE_PATH,
        transport=request,
    )
    if failure:
        return _failure(context, *failure)

    year_urls: dict[int, str] = {}
    year_bodies: dict[int, bytes] = {}
    for year in CALIBRATION_YEARS:
        try:
            candidates = extract_root_year_candidates(
                context.payload_bytes[SOURCE_B_ARCHIVE_ROOT], SOURCE_B_ARCHIVE_ROOT, year
            )
            selected = _locator_url(resolve_source_b_year_page(candidates, year))
        except Exception as exc:
            failure_class, reason = _parse_failure(exc, "ROOT_LOCATOR_FAILURE")
            return _failure(context, failure_class, reason)
        if selected is None:
            return _failure(context, DATA_QUALITY_FAILURE, "ROOT_LOCATOR_FAILURE")
        if selected in year_urls.values():
            return _failure(context, DATA_QUALITY_FAILURE, "DUPLICATE_YEAR_PAYLOAD_URL", year_page_count=len(year_urls))
        year_urls[year] = selected
        locked, failure = fetch_and_lock_payload(
            context, selected, role="year_page", relative_path=f"{YEAR_LOCK_DIRECTORY}/{year}.html",
            transport=request, identity=str(year),
        )
        if failure:
            return _failure(context, *failure, year_page_count=len(year_urls) - 1)
        year_bodies[year] = context.payload_bytes[selected]

    pdf_inputs: list[CalibrationObjectInput] = []
    seen_pdf_urls: set[str] = set()
    for index, identity in enumerate(REQUIRED_CALIBRATION_IDENTITIES, start=1):
        year = int(identity.logical_month[:4])
        try:
            if identity.object_part == PRE_APRIL_1_REFERENCE_OBJECT:
                candidates = extract_april_pre_candidates(
                    year_bodies[year], year_urls[year], selected_year=year
                )
                selected = _locator_url(resolve_source_b_april_pre_object(
                    candidates, selected_year_page_url=year_urls[year]
                ))
            else:
                candidates = extract_normal_month_candidates(
                    year_bodies[year], year_urls[year], identity.logical_month,
                    selected_year=year,
                )
                selected = _locator_url(resolve_source_b_normal_month_object(
                    candidates, identity.logical_month, selected_year_page_url=year_urls[year]
                ))
        except Exception as exc:
            failure_class, reason = _parse_failure(exc, "PDF_LOCATOR_FAILURE")
            return _failure(context, failure_class, reason, year_page_count=len(year_urls), calibration_pdf_count=len(pdf_inputs))
        if selected is None:
            return _failure(context, DATA_QUALITY_FAILURE, "PDF_LOCATOR_FAILURE", year_page_count=len(year_urls), calibration_pdf_count=len(pdf_inputs))
        if selected in seen_pdf_urls:
            return _failure(context, DATA_QUALITY_FAILURE, "DUPLICATE_CALIBRATION_PDF_URL", year_page_count=len(year_urls), calibration_pdf_count=len(pdf_inputs))
        seen_pdf_urls.add(selected)
        locked, failure = fetch_and_lock_payload(
            context, selected, role="calibration_pdf",
            relative_path=f"{PDF_LOCK_DIRECTORY}/{index:02d}_{identity.logical_month}_{identity.object_part}.pdf",
            transport=request, identity=_identity_text(identity),
        )
        if failure:
            return _failure(context, *failure, year_page_count=len(year_urls), calibration_pdf_count=len(pdf_inputs))
        pdf_inputs.append(CalibrationObjectInput(
            context.payload_bytes[selected], identity.logical_month, identity.object_part, locked.sha256
        ))

    context.probe_invocations += 1
    try:
        bundle: CalibrationBundleResult = probe_calibration_bundle(pdf_inputs)
    except Exception:
        return _failure(context, IMPLEMENTATION_FAILURE, "CALIBRATION_PROBE_FAILURE", year_page_count=len(year_urls), calibration_pdf_count=len(pdf_inputs))
    if bundle.status != CALIBRATION_BUNDLE_PASS:
        return _failure(context, DATA_QUALITY_FAILURE, "CALIBRATION_PROBE_FAILURE", year_page_count=len(year_urls), calibration_pdf_count=len(pdf_inputs))
    evidence = tuple(
        result.evidence for result in bundle.results
        if result.evidence is not None
    )
    result = AcquisitionResult(
        ACQUISITION_PASS, None, None, len(context.locked), len(year_urls),
        len(pdf_inputs), context.probe_invocations, tuple(context.locked), evidence,
    )
    try:
        _persist_success(context, result)
    except Exception:
        return _failure(context, IMPLEMENTATION_FAILURE, "SAFE_RECEIPT_WRITE_FAILURE", year_page_count=len(year_urls), calibration_pdf_count=len(pdf_inputs))
    return result


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Future fixed-eight SOURCE_B calibration acquisition runner")
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--production-acquire", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _cli_parser().parse_args(argv)
    if not args.production_acquire:
        print(json.dumps({
            "status": ACQUISITION_FAILURE,
            "failure_class": GOVERNANCE_FAILURE,
            "reason": "PRODUCTION_CONFIRMATION_FLAG_REQUIRED",
        }, sort_keys=True))
        return 1
    try:
        result = run_fixed_eight_calibration_acquisition(
            Path(args.output_root), expected_git_sha=args.expected_git_sha,
            confirmation=args.confirmation,
        )
    except Exception:
        print(json.dumps({
            "status": ACQUISITION_FAILURE,
            "failure_class": IMPLEMENTATION_FAILURE,
            "reason": "RUNNER_FAILURE",
        }, sort_keys=True))
        return 1
    print(json.dumps(result.to_safe_dict(), sort_keys=True))
    return 0 if result.status == ACQUISITION_PASS else 1


__all__ = [
    "ACQUISITION_FAILURE", "ACQUISITION_PASS", "CALIBRATION_YEARS",
    "CHATGPT_DECISION_REQUIRED", "CONFIRMATION_CONTRACT", "DATA_QUALITY_FAILURE",
    "EXPECTED_PAYLOAD_COUNT", "GOVERNANCE_FAILURE", "IMPLEMENTATION_FAILURE",
    "MAX_PRE_COMPLETE_ATTEMPTS_PER_URL", "PLUMBING_FAILURE_RETRIABLE",
    "REQUIRED_CALIBRATION_IDENTITIES", "RETRYABLE_HTTP_STATUSES",
    "ROOT_LOCK_RELATIVE_PATH", "TransportResponse", "AcquisitionResult",
    "NoRedirectHandler", "RedirectRejectedError", "REDIRECT_STATUS_CODES",
    "V9_014_STAGE_D_REDIRECT_POLICY", "ALLOWED_REDIRECT_COUNT",
    "REQUESTED_URL_MUST_EQUAL_RESOLVED_URL", "REDIRECT_IS_RETRYABLE",
    "fetch_and_lock_payload", "run_fixed_eight_calibration_acquisition", "main",
]
