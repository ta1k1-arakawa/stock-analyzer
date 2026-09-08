from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src import v9_015_source_b_stage_f_child_acquisition as stage_f
from src.v9_014_jpx_monthly_auction_activity_source_b_calibration_acquisition import (
    REQUIRED_CALIBRATION_IDENTITIES,
    TransportResponse,
)
from src.v9_014_jpx_monthly_auction_activity_source_b_locator import (
    SOURCE_B_ARCHIVE_ROOT,
)


SHA = "a" * 40
YEARS = stage_f.REQUIRED_YEAR_LABELS


class FakeTransport:
    def __init__(self, bodies: dict[str, bytes], responses: dict[str, list[object]] | None = None):
        self.bodies = bodies
        self.responses = responses or {}
        self.calls: list[tuple[str, int]] = []

    def __call__(self, url: str, timeout: int) -> TransportResponse:
        self.calls.append((url, timeout))
        queue = self.responses.get(url)
        if queue:
            item = queue.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item
        return TransportResponse(200, self.bodies[url])


def _root(year_values: dict[str, str] | None = None, missing: str | None = None, duplicate: str | None = None) -> bytes:
    values = year_values or {
        year: f"https://www.jpx.co.jp/synthetic/year-{year}.html"
        for year in YEARS
    }
    parts = [
        f'<option value="{values[year]}">{year}</option>'
        for year in YEARS
        if year != missing
    ]
    if duplicate is not None:
        parts.append(f'<option value="synthetic/duplicate-{duplicate}.html">{duplicate}</option>')
    return ("<html><body><select>" + "".join(parts) + "</select></body></html>").encode()


def _normal_table(month_to_href: dict[str, str], *, missing_months: set[str] | None = None) -> bytes:
    missing = missing_months or set()
    headers = "".join(f"<th>{month}</th>" for month in month_to_href)
    cells = "".join(
        f'<td><a href="{href}">synthetic-pdf</a></td>'
        for month, href in month_to_href.items()
        if month not in missing
    )
    return (
        "<html><body><table><tr><th>Report</th>"
        + headers
        + "</tr><tr><td>Report 2 &quot;Stock Trading Volume &amp; Value&quot;</td>"
        + cells
        + "</tr></table></body></html>"
    ).encode()


def _pre_table(href: str) -> bytes:
    return f'<html><body><a href="{href}">(Reference) Status on April 1, 2022</a></body></html>'.encode()


def _fixture(*, duplicate_pdf: bool = False, missing_2019_month: bool = False):
    year_urls = {year: f"https://www.jpx.co.jp/synthetic/year-{year}.html" for year in YEARS}
    pdf_urls = {
        identity: f"https://www.jpx.co.jp/synthetic/pdf-{index:02d}.pdf"
        for index, identity in enumerate(REQUIRED_CALIBRATION_IDENTITIES, 1)
    }
    if duplicate_pdf:
        first_identity = REQUIRED_CALIBRATION_IDENTITIES[0]
        second_identity = REQUIRED_CALIBRATION_IDENTITIES[1]
        pdf_urls[second_identity] = pdf_urls[first_identity]

    bodies = {
        url: b"synthetic-year-page"
        for url in year_urls.values()
    }
    identity_2017 = REQUIRED_CALIBRATION_IDENTITIES[0]
    identity_2019 = REQUIRED_CALIBRATION_IDENTITIES[1]
    identity_2020 = REQUIRED_CALIBRATION_IDENTITIES[2]
    identity_2022_mar = REQUIRED_CALIBRATION_IDENTITIES[3]
    identity_2022_apr_pre = REQUIRED_CALIBRATION_IDENTITIES[4]
    identity_2022_apr_post = REQUIRED_CALIBRATION_IDENTITIES[5]
    identity_2022_may = REQUIRED_CALIBRATION_IDENTITIES[6]
    identity_2026 = REQUIRED_CALIBRATION_IDENTITIES[7]
    bodies[year_urls["2017"]] = _normal_table({"2017-01": pdf_urls[identity_2017]})
    bodies[year_urls["2019"]] = _normal_table(
        {"2019-12": pdf_urls[identity_2019]},
        missing_months={"2019-12"} if missing_2019_month else None,
    )
    bodies[year_urls["2020"]] = _normal_table({"2020-01": pdf_urls[identity_2020]})
    bodies[year_urls["2022"]] = (
        _normal_table({
            "2022-03": pdf_urls[identity_2022_mar],
            "2022-04": pdf_urls[identity_2022_apr_post],
            "2022-05": pdf_urls[identity_2022_may],
        })
        + _pre_table(pdf_urls[identity_2022_apr_pre])
    )
    bodies[year_urls["2026"]] = _normal_table({"2026-01": pdf_urls[identity_2026]})
    for index, url in enumerate(dict.fromkeys(pdf_urls.values()), 1):
        bodies[url] = f"SYNTHETIC-PDF-{index}".encode()
    root = _root()
    return root, bodies, year_urls, pdf_urls


def _run(tmp_path, monkeypatch, *, root=None, bodies=None, year_values=None, responses=None, duplicate_pdf=False, missing_2019_month=False):
    fixture_root, fixture_bodies, year_urls, pdf_urls = _fixture(
        duplicate_pdf=duplicate_pdf, missing_2019_month=missing_2019_month
    )
    raw_root = fixture_root if root is None else root
    body_map = fixture_bodies if bodies is None else bodies
    if year_values is not None:
        raw_root = _root(year_values)
    monkeypatch.setattr(stage_f, "ROOT_SHA256", hashlib.sha256(raw_root).hexdigest())
    monkeypatch.setattr(stage_f, "ROOT_BYTE_COUNT", len(raw_root))
    transport = FakeTransport(body_map, responses)
    result = stage_f.run_stage_f_child_acquisition(
        raw_root,
        tmp_path / "attempt",
        expected_git_sha=SHA,
        confirmation=stage_f.STAGE_F_CONFIRMATION_CONTRACT,
        transport=transport,
    )
    return result, transport, raw_root, body_map, year_urls, pdf_urls


def test_synthetic_happy_path_uses_stage_e_and_real_archive_parser(tmp_path, monkeypatch):
    result, transport, _, _, _, _ = _run(tmp_path, monkeypatch)
    assert result.status == stage_f.STAGE_F_PASS
    assert result.preserved_root_binding_verified is True
    assert result.root_network_requests == 0
    assert result.child_network_requests == 13
    assert result.year_page_lock_count == 5
    assert result.calibration_pdf_lock_count == 8
    assert result.child_lock_count == 13
    assert result.probe_invocations == 0
    assert len(transport.calls) == 13
    assert all(timeout == 30 for _, timeout in transport.calls)
    assert (tmp_path / "attempt" / "attempt.json").exists()
    assert (tmp_path / "attempt" / "receipt.json").exists()
    assert (tmp_path / "attempt" / "complete.json").exists()


def test_root_sha_mismatch_fails_before_any_child_transport(tmp_path, monkeypatch):
    root, bodies, *_ = _fixture()
    monkeypatch.setattr(stage_f, "ROOT_SHA256", "0" * 64)
    monkeypatch.setattr(stage_f, "ROOT_BYTE_COUNT", len(root))
    transport = FakeTransport(bodies)
    result = stage_f.run_stage_f_child_acquisition(
        root, tmp_path / "attempt", expected_git_sha=SHA,
        confirmation=stage_f.STAGE_F_CONFIRMATION_CONTRACT, transport=transport,
    )
    assert result.reason == "PRESERVED_ROOT_SHA256_MISMATCH"
    assert result.root_network_requests == 0 and transport.calls == []


def test_root_byte_count_mismatch_fails_before_any_child_transport(tmp_path, monkeypatch):
    root, bodies, *_ = _fixture()
    monkeypatch.setattr(stage_f, "ROOT_SHA256", hashlib.sha256(root).hexdigest())
    monkeypatch.setattr(stage_f, "ROOT_BYTE_COUNT", len(root) + 1)
    transport = FakeTransport(bodies)
    result = stage_f.run_stage_f_child_acquisition(
        root, tmp_path / "attempt", expected_git_sha=SHA,
        confirmation=stage_f.STAGE_F_CONFIRMATION_CONTRACT, transport=transport,
    )
    assert result.reason == "PRESERVED_ROOT_BYTE_COUNT_MISMATCH"
    assert transport.calls == []


@pytest.mark.parametrize("root_kind", ["missing", "duplicate"])
def test_required_option_value_failure_precedes_child_acquisition(tmp_path, monkeypatch, root_kind):
    fixture_root, bodies, *_ = _fixture()
    raw = _root(missing="2026") if root_kind == "missing" else _root(duplicate="2017")
    monkeypatch.setattr(stage_f, "ROOT_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(stage_f, "ROOT_BYTE_COUNT", len(raw))
    transport = FakeTransport(bodies)
    result = stage_f.run_stage_f_child_acquisition(
        raw, tmp_path / "attempt", expected_git_sha=SHA,
        confirmation=stage_f.STAGE_F_CONFIRMATION_CONTRACT, transport=transport,
    )
    assert result.reason == "ROOT_LOCATOR_FAILURE"
    assert transport.calls == []


@pytest.mark.parametrize(
    "root_mutation",
    [
        lambda raw: raw.replace(
            b'value="https://www.jpx.co.jp/synthetic/year-2017.html"', b""
        ),
        lambda raw: raw.replace(
            b'value="https://www.jpx.co.jp/synthetic/year-2017.html"',
            b'value=""',
        ),
        lambda raw: raw.replace(
            b'<option value="https://www.jpx.co.jp/synthetic/year-2017.html">2017</option>',
            b'<option value="one" value="two">2017</option>',
        ),
        lambda raw: raw.replace(
            b'<select>', b'<select><a href="https://www.jpx.co.jp/synthetic/fake.html">2017</a>'
        ),
        lambda raw: raw.replace(b">2017</option>", b">Year 2017</option>"),
        lambda raw: raw.replace(
            b'value="https://www.jpx.co.jp/synthetic/year-2017.html"',
            b'value="https://evil.example/year.html"',
        ),
    ],
)
def test_option_value_root_edge_cases_fail_closed_before_child_transport(
    tmp_path, monkeypatch, root_mutation
):
    raw = root_mutation(_fixture()[0])
    monkeypatch.setattr(stage_f, "ROOT_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(stage_f, "ROOT_BYTE_COUNT", len(raw))
    transport = FakeTransport(_fixture()[1])
    result = stage_f.run_stage_f_child_acquisition(
        raw,
        tmp_path / "attempt",
        expected_git_sha=SHA,
        confirmation=stage_f.STAGE_F_CONFIRMATION_CONTRACT,
        transport=transport,
    )
    assert result.reason == "ROOT_LOCATOR_FAILURE"
    assert transport.calls == []


def test_wrong_confirmation_invalid_sha_and_collision_fail_before_network(tmp_path, monkeypatch):
    root, bodies, *_ = _fixture()
    transport = FakeTransport(bodies)
    wrong = stage_f.run_stage_f_child_acquisition(
        root, tmp_path / "wrong", expected_git_sha=SHA,
        confirmation="WRONG", transport=transport,
    )
    invalid = stage_f.run_stage_f_child_acquisition(
        root, tmp_path / "invalid", expected_git_sha="bad",
        confirmation=stage_f.STAGE_F_CONFIRMATION_CONTRACT, transport=transport,
    )
    collision_root = tmp_path / "collision"
    collision_root.mkdir()
    collision = stage_f.run_stage_f_child_acquisition(
        tmp_path / "missing-preserved-root", collision_root,
        expected_git_sha=SHA, confirmation=stage_f.STAGE_F_CONFIRMATION_CONTRACT,
        transport=transport,
    )
    assert wrong.reason == "CONFIRMATION_CONTRACT_MISMATCH"
    assert invalid.reason == "EXPECTED_GIT_SHA_INVALID"
    assert collision.reason == "OUTPUT_ROOT_COLLISION"
    assert transport.calls == []


def test_duplicate_year_url_fails_closed_before_duplicate_fetch(tmp_path, monkeypatch):
    root, bodies, year_urls, _ = _fixture()
    values = {
        year: f"https://www.jpx.co.jp/synthetic/year-{year}.html"
        for year in YEARS
    }
    values["2019"] = values["2017"]
    raw = _root(values)
    monkeypatch.setattr(stage_f, "ROOT_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(stage_f, "ROOT_BYTE_COUNT", len(raw))
    transport = FakeTransport(bodies)
    result = stage_f.run_stage_f_child_acquisition(
        raw, tmp_path / "attempt", expected_git_sha=SHA,
        confirmation=stage_f.STAGE_F_CONFIRMATION_CONTRACT, transport=transport,
    )
    assert result.reason == "DUPLICATE_YEAR_PAYLOAD_URL"
    assert result.year_page_lock_count == 1
    assert len(transport.calls) == 1


def test_duplicate_calibration_pdf_url_preserves_prior_locks(tmp_path, monkeypatch):
    result, transport, _, _, _, _ = _run(tmp_path, monkeypatch, duplicate_pdf=True)
    assert result.reason == "DUPLICATE_CALIBRATION_PDF_URL"
    assert result.year_page_lock_count == 5
    assert result.calibration_pdf_lock_count == 1
    assert result.child_lock_count == 6
    assert len(transport.calls) == 6
    locked_files = list((tmp_path / "attempt" / "raw").rglob("*"))
    assert len([path for path in locked_files if path.is_file()]) == 6


def test_child_locator_failure_preserves_prior_locks_without_refetch(tmp_path, monkeypatch):
    result, transport, _, _, _, _ = _run(tmp_path, monkeypatch, missing_2019_month=True)
    assert result.reason == "PDF_LOCATOR_FAILURE"
    assert result.year_page_lock_count == 5
    assert result.calibration_pdf_lock_count == 1
    assert result.child_lock_count == 6
    requested_urls = [url for url, _ in transport.calls]
    assert len(requested_urls) == len(set(requested_urls)) == 6
    assert len(list((tmp_path / "attempt" / "raw").rglob("*"))) >= 6


def test_inherited_retry_behavior_is_bounded_for_child_transport(tmp_path, monkeypatch):
    _, bodies, _, pdf_urls = _fixture()
    first_pdf = pdf_urls[REQUIRED_CALIBRATION_IDENTITIES[0]]
    responses = {first_pdf: [TimeoutError(), TimeoutError(), TransportResponse(200, bodies[first_pdf])]}
    result, transport, *_ = _run(tmp_path, monkeypatch, responses=responses)
    assert result.status == stage_f.STAGE_F_PASS
    assert len(transport.calls) == 15
    assert [url for url, _ in transport.calls].count(first_pdf) == 3


def test_inherited_resolved_url_mismatch_fails_before_lock(tmp_path, monkeypatch):
    root, bodies, year_urls, _ = _fixture()
    responses = {
        year_urls["2017"]: [TransportResponse(
            200, bodies[year_urls["2017"]], "https://www.jpx.co.jp/synthetic/other.html"
        )]
    }
    result, transport, *_ = _run(tmp_path, monkeypatch, responses=responses)
    assert result.reason == "RESOLVED_URL_MISMATCH"
    assert result.year_page_lock_count == 0
    assert result.child_lock_count == 0
    assert len(transport.calls) == 1
    assert not (tmp_path / "attempt" / "raw").exists()


def test_safe_result_contains_no_urls_payload_text_or_private_path(tmp_path, monkeypatch):
    result, _, *_ = _run(tmp_path, monkeypatch)
    safe = json.dumps(result.to_safe_dict(), sort_keys=True)
    assert "https://" not in safe
    assert "synthetic-pdf" not in safe
    assert str(tmp_path) not in safe
    assert "raw/calibration_pdfs" in safe
    assert all("url" not in item for item in result.to_safe_dict()["locked_payloads"])


def test_stage_f_source_omits_stage_g_and_root_transport_paths():
    source = Path(stage_f.__file__).read_text(encoding="utf-8")
    assert "probe_calibration_bundle" not in source
    assert "run_fixed_eight_calibration_acquisition" not in source
    assert 'role="root"' not in source
    assert "fetch_and_lock_payload" in source
    assert "ROOT_NETWORK_REQUESTS = 0" in source


def test_cli_without_production_flag_is_safe_failure_from_external_cwd(tmp_path):
    script = Path("scripts/run_v9_015_source_b_stage_f_child_acquisition.py").resolve()
    environment = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "-B", str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 1
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload == {
        "failure_class": stage_f.GOVERNANCE_FAILURE,
        "reason": "PRODUCTION_CONFIRMATION_FLAG_REQUIRED",
        "status": stage_f.STAGE_F_FAILURE,
    }


def test_fixed_production_baseline_and_targets_are_closed():
    assert stage_f.ROOT_SHA256 == "2e839c60bfb9d6edb59a903a590a505130124b8380b096ae84b06e4b0972098c"
    assert stage_f.ROOT_BYTE_COUNT == 75185
    assert stage_f.YEAR_PAGE_TARGET_COUNT == 5
    assert stage_f.CALIBRATION_PDF_TARGET_COUNT == 8
    assert stage_f.CHILD_LOCK_TARGET_COUNT == 13
