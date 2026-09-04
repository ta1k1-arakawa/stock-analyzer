from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from src import v9_014_jpx_monthly_auction_activity_source_b_calibration_acquisition as acquisition
from src.v9_014_jpx_monthly_auction_activity_source_b_locator import (
    APRIL_1_2022_REFERENCE_LABEL,
    LOCATOR_MULTIPLE_CANDIDATES_FAILURE,
    LOCATOR_ZERO_CANDIDATES_FAILURE,
    AprilPreReferenceCandidate,
    LocatorResult,
    MonthlyReportCandidate,
    RootYearCandidate,
    SOURCE_B_REPORT,
)
from src.v9_014_jpx_monthly_auction_activity_source_b_pdf_calibration_probe import (
    CALIBRATION_BUNDLE_PASS,
    CalibrationBundleResult,
    CalibrationProbeResult,
    NORMAL_MONTHLY_REPORT2_OBJECT,
    PRE_APRIL_1_REFERENCE_OBJECT,
    REQUIRED_CALIBRATION_IDENTITIES,
)


SHA = "a" * 40
ROOT = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html"


class FakeTransport:
    def __init__(self, bodies: dict[str, bytes], responses: dict[str, list[object]] | None = None):
        self.bodies = bodies
        self.responses = responses or {}
        self.calls: list[tuple[str, int]] = []

    def __call__(self, url: str, timeout: int):
        self.calls.append((url, timeout))
        queue = self.responses.get(url)
        if queue:
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return acquisition.TransportResponse(200, self.bodies[url])


def _urls():
    year_urls = {year: f"https://www.jpx.co.jp/synthetic/year-{year}.html" for year in acquisition.CALIBRATION_YEARS}
    pdf_urls = {identity: f"https://www.jpx.co.jp/synthetic/pdf-{index:02d}.pdf" for index, identity in enumerate(REQUIRED_CALIBRATION_IDENTITIES, 1)}
    return year_urls, pdf_urls


def _install_synthetic(monkeypatch, tmp_path, probe_status=CALIBRATION_BUNDLE_PASS):
    year_urls, pdf_urls = _urls()
    bodies = {ROOT: b"synthetic-root-4821"}
    for year, url in year_urls.items():
        bodies[url] = f"synthetic-year-{year}-731".encode()
    for index, url in enumerate(pdf_urls.values(), 1):
        bodies[url] = f"SYNTHETIC PDF BODY 4821-{index}".encode()

    def root_candidates(root_bytes, root_url, requested_year):
        assert (tmp_path / acquisition.ROOT_LOCK_RELATIVE_PATH).read_bytes() == root_bytes
        return (RootYearCandidate(str(requested_year), year_urls[requested_year]),)

    def normal_candidates(year_bytes, selected_url, requested_month, *, selected_year):
        year = int(requested_month[:4])
        assert (tmp_path / acquisition.YEAR_LOCK_DIRECTORY / f"{year}.html").read_bytes() == year_bytes
        identity = next(item for item in REQUIRED_CALIBRATION_IDENTITIES if item.logical_month == requested_month and item.object_part == NORMAL_MONTHLY_REPORT2_OBJECT)
        return (MonthlyReportCandidate(selected_url, SOURCE_B_REPORT, requested_month, pdf_urls[identity]),)

    def pre_candidates(year_bytes, selected_url, *, selected_year):
        assert (tmp_path / acquisition.YEAR_LOCK_DIRECTORY / "2022.html").read_bytes() == year_bytes
        identity = next(item for item in REQUIRED_CALIBRATION_IDENTITIES if item.object_part == PRE_APRIL_1_REFERENCE_OBJECT)
        return (AprilPreReferenceCandidate(selected_url, APRIL_1_2022_REFERENCE_LABEL, pdf_urls[identity]),)

    seen_probe = []

    def fake_probe(objects):
        seen_probe.append(objects)
        assert len(objects) == 8
        assert len({item.expected_sha256 for item in objects}) == 8
        if probe_status != CALIBRATION_BUNDLE_PASS:
            return CalibrationBundleResult(probe_status)
        return CalibrationBundleResult(
            CALIBRATION_BUNDLE_PASS,
            tuple(CalibrationProbeResult("CALIBRATION_PROBE_PASS", item.logical_month, item.object_part, item.expected_sha256, {"masked_character_text": "SYNTHETIC ###"}) for item in objects),
        )

    monkeypatch.setattr(acquisition, "extract_root_year_candidates", root_candidates)
    monkeypatch.setattr(acquisition, "extract_normal_month_candidates", normal_candidates)
    monkeypatch.setattr(acquisition, "extract_april_pre_candidates", pre_candidates)
    monkeypatch.setattr(acquisition, "probe_calibration_bundle", fake_probe)
    return bodies, year_urls, pdf_urls, seen_probe


def _run(tmp_path, monkeypatch, responses=None, probe_status=CALIBRATION_BUNDLE_PASS):
    output_root = tmp_path / "attempt"
    bodies, year_urls, pdf_urls, seen_probe = _install_synthetic(monkeypatch, output_root, probe_status)
    transport = FakeTransport(bodies, responses)
    result = acquisition.run_fixed_eight_calibration_acquisition(
        output_root, expected_git_sha=SHA, confirmation=acquisition.CONFIRMATION_CONTRACT, transport=transport
    )
    return result, transport, bodies, year_urls, pdf_urls, seen_probe


def test_exact_fixed_eight_success_and_lock_order(tmp_path, monkeypatch):
    result, transport, bodies, year_urls, pdf_urls, seen_probe = _run(tmp_path, monkeypatch)
    assert result.status == acquisition.ACQUISITION_PASS
    assert result.unique_complete_payload_count == 14
    assert result.year_page_count == 5
    assert result.calibration_pdf_count == 8
    assert len(seen_probe) == 1
    assert [url for url, _ in transport.calls[:6]] == [ROOT] + [year_urls[y] for y in acquisition.CALIBRATION_YEARS]
    assert all(timeout == 30 for _, timeout in transport.calls)
    assert len(transport.calls) == 14
    for item in result.locked_payloads:
        path = tmp_path / "attempt" / item.relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item.sha256
        if item.role == "calibration_pdf":
            assert path.read_bytes() == bodies[pdf_urls[next(identity for identity in pdf_urls if f"{identity.logical_month}_{identity.object_part}" in item.relative_path)]]


def test_exact_identities_and_no_extra_ninth_pdf(tmp_path, monkeypatch):
    result, transport, _, _, pdf_urls, seen_probe = _run(tmp_path, monkeypatch)
    assert result.status == acquisition.ACQUISITION_PASS
    assert [item.identity for item in result.locked_payloads if item.role == "calibration_pdf"] == [
        f"{identity.logical_month}:{identity.object_part}" for identity in REQUIRED_CALIBRATION_IDENTITIES
    ]
    assert len(set(pdf_urls.values())) == 8
    assert len(transport.calls) == 14
    assert len(seen_probe) == 1


def test_collision_fails_before_any_fetch(tmp_path, monkeypatch):
    calls = []
    result = acquisition.run_fixed_eight_calibration_acquisition(
        tmp_path, expected_git_sha=SHA, confirmation=acquisition.CONFIRMATION_CONTRACT,
        transport=lambda url, timeout: calls.append(url),
    )
    assert result.failure_class == acquisition.GOVERNANCE_FAILURE
    assert calls == []


def test_wrong_confirmation_and_sha_fail_before_fetch(tmp_path):
    calls = []
    transport = lambda url, timeout: calls.append(url)
    result = acquisition.run_fixed_eight_calibration_acquisition(tmp_path / "a", expected_git_sha="bad", confirmation=acquisition.CONFIRMATION_CONTRACT, transport=transport)
    assert result.failure_class == acquisition.GOVERNANCE_FAILURE and calls == []
    result = acquisition.run_fixed_eight_calibration_acquisition(tmp_path / "b", expected_git_sha=SHA, confirmation="wrong", transport=transport)
    assert result.failure_class == acquisition.GOVERNANCE_FAILURE and calls == []


def test_locator_missing_duplicate_and_substitute_fail_closed(tmp_path, monkeypatch):
    output_root = tmp_path / "missing"
    bodies, year_urls, _, _ = _install_synthetic(monkeypatch, output_root)
    transport = FakeTransport(bodies)
    monkeypatch.setattr(acquisition, "resolve_source_b_year_page", lambda candidates, year: LocatorResult(LOCATOR_ZERO_CANDIDATES_FAILURE))
    result = acquisition.run_fixed_eight_calibration_acquisition(output_root, expected_git_sha=SHA, confirmation=acquisition.CONFIRMATION_CONTRACT, transport=transport)
    assert result.status == acquisition.ACQUISITION_FAILURE and len(transport.calls) == 1

    tmp2 = tmp_path / "duplicate"
    output_root = tmp2 / "run"
    bodies, year_urls, _, _ = _install_synthetic(monkeypatch, output_root)
    transport = FakeTransport(bodies)
    monkeypatch.setattr(acquisition, "resolve_source_b_year_page", lambda candidates, year: LocatorResult(LOCATOR_MULTIPLE_CANDIDATES_FAILURE))
    result = acquisition.run_fixed_eight_calibration_acquisition(output_root, expected_git_sha=SHA, confirmation=acquisition.CONFIRMATION_CONTRACT, transport=transport)
    assert result.status == acquisition.ACQUISITION_FAILURE and len(transport.calls) == 1


def test_retryable_transport_and_http_codes_are_bounded(tmp_path):
    context = acquisition._RunContext(tmp_path)
    url = "https://www.jpx.co.jp/synthetic/one.pdf"
    calls = []

    def transport(value, timeout):
        calls.append(value)
        if len(calls) < 3:
            raise TimeoutError()
        return acquisition.TransportResponse(200, b"exact-pdf-4821")

    locked, failure = acquisition.fetch_and_lock_payload(context, url, role="calibration_pdf", relative_path="raw/calibration_pdfs/01.pdf", transport=transport)
    assert locked is not None and failure is None and len(calls) == 3
    second, failure = acquisition.fetch_and_lock_payload(context, url, role="calibration_pdf", relative_path="raw/calibration_pdfs/01.pdf", transport=lambda *_: (_ for _ in ()).throw(AssertionError("refetch")))
    assert second == locked and failure is None

    for status in sorted(acquisition.RETRYABLE_HTTP_STATUSES):
        root = tmp_path / str(status)
        root.mkdir()
        count = []
        def retry_transport(value, timeout, status=status, count=count):
            count.append(value)
            if len(count) < 2:
                return acquisition.TransportResponse(status, b"ignored")
            return acquisition.TransportResponse(200, b"body")
        locked, failure = acquisition.fetch_and_lock_payload(acquisition._RunContext(root), url, role="root", relative_path=acquisition.ROOT_LOCK_RELATIVE_PATH, transport=retry_transport)
        assert locked is not None and failure is None and len(count) == 2

    root = tmp_path / "nonretry"
    root.mkdir()
    count = []
    locked, failure = acquisition.fetch_and_lock_payload(acquisition._RunContext(root), url, role="root", relative_path=acquisition.ROOT_LOCK_RELATIVE_PATH, transport=lambda value, timeout: (count.append(value), acquisition.TransportResponse(403, b"no"))[1])
    assert locked is None and failure == (acquisition.DATA_QUALITY_FAILURE, "NONRETRYABLE_HTTP_STATUS") and len(count) == 1


def test_parse_or_probe_failure_preserves_locks_and_never_refetches(tmp_path, monkeypatch):
    output_root = tmp_path / "probe"
    bodies, _, _, _ = _install_synthetic(monkeypatch, output_root, probe_status="FAIL")
    transport = FakeTransport(bodies)
    result = acquisition.run_fixed_eight_calibration_acquisition(output_root, expected_git_sha=SHA, confirmation=acquisition.CONFIRMATION_CONTRACT, transport=transport)
    assert result.status == acquisition.ACQUISITION_FAILURE
    assert result.probe_invocations == 1
    assert len(transport.calls) == 14
    assert len(list((output_root / "raw").rglob("*"))) >= 14
    assert not (output_root / "receipt.json").exists()

    tmp2 = tmp_path / "parse"
    bodies, _, _, _ = _install_synthetic(monkeypatch, tmp2)
    transport = FakeTransport(bodies)
    monkeypatch.setattr(acquisition, "extract_root_year_candidates", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError()))
    parse_root = tmp2 / "attempt"
    # The synthetic callbacks bind to the output-root path, so the run starts
    # in a fresh child while retaining the same fake payload map.
    result = acquisition.run_fixed_eight_calibration_acquisition(parse_root, expected_git_sha=SHA, confirmation=acquisition.CONFIRMATION_CONTRACT, transport=transport)
    assert result.status == acquisition.ACQUISITION_FAILURE
    assert len(transport.calls) == 1
    assert (parse_root / acquisition.ROOT_LOCK_RELATIVE_PATH).read_bytes() == bodies[ROOT]


def test_safe_receipt_has_no_raw_url_or_pdf_text(tmp_path, monkeypatch):
    result, _, _, _, _, _ = _run(tmp_path, monkeypatch)
    safe = json.dumps(result.to_safe_dict(), ensure_ascii=True)
    assert "https://" not in safe
    assert "SYNTHETIC PDF BODY 4821" not in safe
    assert "4821" not in safe
    assert all("url" not in item for item in result.to_safe_dict()["locked_payloads"])


def test_static_surface_has_no_semantic_or_alternate_parser_path():
    source = Path(acquisition.__file__).read_text(encoding="utf-8")
    forbidden = ("extract_text", "extract_table", "extract_tables", "find_tables", "classify_date", "trading_dates", "evaluate_relation", "pypdf", "PyMuPDF", "fitz", "ocr")
    assert not any(token in source for token in forbidden)
    assert "validate_jpx_url" in source
    assert "NO_REDIRECT_OPENER" in source


@pytest.mark.parametrize("status", sorted(acquisition.REDIRECT_STATUS_CODES))
def test_no_redirect_handler_rejects_all_frozen_redirect_statuses(status):
    handler = acquisition.NoRedirectHandler()
    request = urllib.request.Request("https://www.jpx.co.jp/synthetic/request")
    with pytest.raises(acquisition.RedirectRejectedError):
        getattr(handler, f"http_error_{status}")(request, object(), status, "redirect", {})
    with pytest.raises(acquisition.RedirectRejectedError):
        handler.redirect_request(request, object(), status, "redirect", {"Location": "https://www.jpx.co.jp/synthetic/target"}, "https://www.jpx.co.jp/synthetic/target")


def test_redirect_is_nonretryable_and_target_is_never_requested(tmp_path):
    requested = "https://www.jpx.co.jp/synthetic/request.pdf"
    target = "https://www.jpx.co.jp/synthetic/target.pdf"
    calls = []

    def transport(url, timeout):
        calls.append(url)
        raise acquisition.RedirectRejectedError()

    locked, failure = acquisition.fetch_and_lock_payload(
        acquisition._RunContext(tmp_path), requested, role="calibration_pdf",
        relative_path="raw/calibration_pdfs/01.pdf", transport=transport,
    )
    assert locked is None
    assert failure == (acquisition.DATA_QUALITY_FAILURE, "HTTP_REDIRECT_REJECTED")
    assert calls == [requested]
    assert target not in calls


def test_mismatched_resolved_url_fails_closed_without_raw_lock(tmp_path):
    requested = "https://www.jpx.co.jp/synthetic/request.pdf"
    mismatched = "https://www.jpx.co.jp/synthetic/other.pdf"
    calls = []

    def transport(url, timeout):
        calls.append(url)
        return acquisition.TransportResponse(200, b"unlocked-body-4821", mismatched)

    locked, failure = acquisition.fetch_and_lock_payload(
        acquisition._RunContext(tmp_path), requested, role="calibration_pdf",
        relative_path="raw/calibration_pdfs/01.pdf", transport=transport,
    )
    assert locked is None
    assert failure == (acquisition.DATA_QUALITY_FAILURE, "RESOLVED_URL_MISMATCH")
    assert calls == [requested]
    assert not (tmp_path / "raw").exists()


class _FakeResponse:
    def __init__(self, resolved_url, events):
        self._resolved_url = resolved_url
        self._events = events
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        self._events.append("geturl")
        return self._resolved_url

    def read(self):
        self._events.append("read")
        return b"validated-body"


class _FakeOpener:
    def __init__(self, response, events):
        self.response = response
        self.events = events

    def open(self, request, timeout):
        self.events.append(("open", request.full_url, timeout))
        return self.response


def test_default_transport_validates_url_and_equality_before_body_read(monkeypatch):
    requested = "https://www.jpx.co.jp/synthetic/request.pdf"
    events = []
    monkeypatch.setattr(acquisition, "NO_REDIRECT_OPENER", _FakeOpener(_FakeResponse(requested, events), events))
    result = acquisition._default_transport(requested, 30)
    assert result.body == b"validated-body"
    assert events == [("open", requested, 30), "geturl", "read"]

    events = []
    monkeypatch.setattr(acquisition, "NO_REDIRECT_OPENER", _FakeOpener(_FakeResponse("https://www.jpx.co.jp/synthetic/target.pdf", events), events))
    with pytest.raises(acquisition.RedirectRejectedError):
        acquisition._default_transport(requested, 30)
    assert events == [("open", requested, 30), "geturl"]


def _run_wrapper_without_pythonpath(*arguments):
    environment = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-B", "scripts/run_v9_014_source_b_calibration_acquisition.py", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_wrapper_file_path_help_imports_without_pythonpath():
    completed = _run_wrapper_without_pythonpath("--help")
    assert completed.returncode == 0
    assert "ModuleNotFoundError" not in completed.stderr
    assert "usage:" in completed.stdout


def test_wrapper_file_path_reaches_governance_guard_without_network_or_output_root(tmp_path):
    output_root = tmp_path / "stage-d-not-created"
    completed = _run_wrapper_without_pythonpath(
        "--expected-git-sha", "a" * 40,
        "--output-root", str(output_root),
        "--confirmation", acquisition.CONFIRMATION_CONTRACT,
    )
    assert completed.returncode == 1
    assert "ModuleNotFoundError" not in completed.stderr
    receipt = json.loads(completed.stdout)
    assert receipt == {
        "failure_class": acquisition.GOVERNANCE_FAILURE,
        "reason": "PRODUCTION_CONFIRMATION_FLAG_REQUIRED",
        "status": acquisition.ACQUISITION_FAILURE,
    }
    assert not output_root.exists()
