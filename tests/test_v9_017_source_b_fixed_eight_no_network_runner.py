from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import src.v9_017_source_b_fixed_eight_no_network_runner as runner
from src.v9_014_jpx_monthly_auction_activity_source_b_pdf_calibration_probe import (
    REQUIRED_CALIBRATION_IDENTITIES,
)
from src.v9_017_source_b_fixed_eight_locator_application import (
    FixedEightLocatorApplicationResult,
)


VALID_GIT_SHA = "a" * 40
CONFIRMATION = runner.CONFIRMATION_CONTRACT
ROOT_URL = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/index.html"


def _root_page(years: tuple[int, ...] = runner.REQUIRED_YEARS) -> bytes:
    options = "".join(
        f'<option value="/english/markets/statistics-equities/monthly/{year}.html">{year}</option>'
        for year in years
    )
    return f"<html><body><select>{options}</select></body></html>".encode()


def _year_pages() -> dict[int, bytes]:
    return {year: f"synthetic-year-{year}".encode() for year in runner.REQUIRED_YEARS}


def _expected_bindings(pages: dict[int, bytes]) -> dict[int, tuple[int, str]]:
    return {
        year: (len(payload), hashlib.sha256(payload).hexdigest())
        for year, payload in pages.items()
    }


def _fake_success_application(calls: list[tuple[dict[int, bytes], dict[int, str]]]):
    def apply(pages: dict[int, bytes], parent_urls: dict[int, str]):
        calls.append((dict(pages), dict(parent_urls)))
        identity_results = tuple(
            {
                "logical_month": identity.logical_month,
                "object_part": identity.object_part,
                "status": "PASS",
            }
            for identity in REQUIRED_CALIBRATION_IDENTITIES
        )
        return FixedEightLocatorApplicationResult(
            status="PASS",
            resolved_identity_count=8,
            identity_results=identity_results,
            failure_identity=None,
            failure_reason=None,
        )

    return apply


def _run_synthetic(
    tmp_path: Path,
    *,
    root_bytes: bytes | None = None,
    pages: dict[int, bytes] | None = None,
    expected_root_count: int | None = None,
    expected_root_sha: str | None = None,
    expected_bindings: dict[int, tuple[int, str]] | None = None,
    application=None,
):
    root_bytes = _root_page() if root_bytes is None else root_bytes
    pages = _year_pages() if pages is None else pages
    root_path = tmp_path / "root.synthetic"
    root_path.write_bytes(root_bytes)
    page_paths: dict[int, str] = {}
    for year, payload in pages.items():
        path = tmp_path / f"year-{year}.synthetic"
        path.write_bytes(payload)
        page_paths[year] = str(path)
    output_root = tmp_path / "output"
    return runner._run_with_expected_bindings(
        str(root_path),
        page_paths,
        str(output_root),
        expected_git_sha=VALID_GIT_SHA,
        confirmation=CONFIRMATION,
        expected_root_byte_count=(
            len(root_bytes) if expected_root_count is None else expected_root_count
        ),
        expected_root_sha256=(
            hashlib.sha256(root_bytes).hexdigest()
            if expected_root_sha is None
            else expected_root_sha
        ),
        expected_year_bindings=(
            _expected_bindings(pages)
            if expected_bindings is None
            else expected_bindings
        ),
        application=application,
    )


def test_synthetic_root_binds_exact_five_parent_urls_and_keeps_roles_separate(
    tmp_path: Path,
) -> None:
    root_bytes = _root_page()
    pages = _year_pages()
    pages[2026] = root_bytes
    calls: list[tuple[dict[int, bytes], dict[int, str]]] = []
    result = _run_synthetic(
        tmp_path,
        root_bytes=root_bytes,
        pages=pages,
        expected_bindings=_expected_bindings(pages),
        application=_fake_success_application(calls),
    )

    assert result["status"] == "PASS"
    assert result["root_binding_verified"] is True
    assert result["parent_year_url_binding_count"] == 5
    assert result["fixed_identity_count"] == 8
    assert len(calls) == 1
    received_pages, received_urls = calls[0]
    assert received_pages[2026] == root_bytes
    assert set(received_urls) == set(runner.REQUIRED_YEARS)
    assert received_urls[2017] == "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/2017.html"


def test_verified_inputs_reach_application_with_exact_eight_order(tmp_path: Path) -> None:
    calls: list[tuple[dict[int, bytes], dict[int, str]]] = []
    result = _run_synthetic(
        tmp_path,
        application=_fake_success_application(calls),
    )

    assert result["status"] == "PASS"
    assert len(calls) == 1
    assert result["fixed_eight_invocations"] == 1
    assert result["fixed_eight_result"]["resolved_identity_count"] == 8
    assert result["fixed_eight_result"]["identity_results"] == [
        {
            "logical_month": identity.logical_month,
            "object_part": identity.object_part,
            "status": "PASS",
        }
        for identity in REQUIRED_CALIBRATION_IDENTITIES
    ]


def test_attempt_metadata_uses_target_before_application_boundary(tmp_path: Path) -> None:
    pages = _year_pages()
    expected = _expected_bindings(pages)
    expected[2017] = (expected[2017][0], "e" * 64)
    calls: list[tuple[dict[int, bytes], dict[int, str]]] = []
    result = _run_synthetic(
        tmp_path,
        pages=pages,
        expected_bindings=expected,
        application=_fake_success_application(calls),
    )

    attempt = json.loads((tmp_path / "output" / "attempt.json").read_text(encoding="utf-8"))
    assert attempt["target_fixed_eight_invocations"] == 1
    assert "fixed_eight_invocations" not in attempt
    assert result["reason"] == "YEAR_PAGE_SHA256_MISMATCH"
    assert result["fixed_eight_invocations"] == 0
    assert calls == []


@pytest.mark.parametrize("years", [(2017, 2019, 2020, 2022), (2017, 2019, 2020, 2022, 2022, 2026)])
def test_missing_or_ambiguous_required_root_year_fails_closed(
    tmp_path: Path, years: tuple[int, ...]
) -> None:
    calls: list[tuple[dict[int, bytes], dict[int, str]]] = []
    result = _run_synthetic(
        tmp_path,
        root_bytes=_root_page(years),
        application=_fake_success_application(calls),
    )

    assert result["status"] == "FAIL"
    assert result["reason"] == "ROOT_YEAR_BINDING_FAILURE"
    assert result["fixed_eight_invocations"] == 0
    assert calls == []


def test_root_byte_count_mismatch_precedes_application(tmp_path: Path) -> None:
    calls: list[tuple[dict[int, bytes], dict[int, str]]] = []
    root_bytes = _root_page()
    result = _run_synthetic(
        tmp_path,
        root_bytes=root_bytes,
        expected_root_count=len(root_bytes) + 1,
        application=_fake_success_application(calls),
    )

    assert result["reason"] == "PRESERVED_ROOT_BYTE_COUNT_MISMATCH"
    assert result["fixed_eight_invocations"] == 0
    assert calls == []


def test_root_sha_mismatch_precedes_application(tmp_path: Path) -> None:
    calls: list[tuple[dict[int, bytes], dict[int, str]]] = []
    root_bytes = _root_page()
    result = _run_synthetic(
        tmp_path,
        root_bytes=root_bytes,
        expected_root_sha="b" * 64,
        application=_fake_success_application(calls),
    )

    assert result["reason"] == "PRESERVED_ROOT_SHA256_MISMATCH"
    assert result["fixed_eight_invocations"] == 0
    assert calls == []


@pytest.mark.parametrize("year", runner.REQUIRED_YEARS)
def test_each_year_page_binding_mismatch_fails_closed(tmp_path: Path, year: int) -> None:
    pages = _year_pages()
    expected = _expected_bindings(pages)
    expected[year] = (expected[year][0], "c" * 64)
    calls: list[tuple[dict[int, bytes], dict[int, str]]] = []
    result = _run_synthetic(
        tmp_path,
        pages=pages,
        expected_bindings=expected,
        application=_fake_success_application(calls),
    )

    assert result["status"] == "FAIL"
    assert result["reason"] == "YEAR_PAGE_SHA256_MISMATCH"
    assert result["fixed_eight_invocations"] == 0
    assert calls == []


def test_collision_is_checked_before_any_input_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir()
    calls = 0

    def forbidden(_path: str):
        nonlocal calls
        calls += 1
        raise AssertionError("input read occurred")

    monkeypatch.setattr(runner, "_read_once", forbidden)
    root_path = tmp_path / "missing-root"
    page_paths = {year: str(tmp_path / f"missing-{year}") for year in runner.REQUIRED_YEARS}
    result = runner._run_with_expected_bindings(
        str(root_path),
        page_paths,
        str(output_root),
        expected_git_sha=VALID_GIT_SHA,
        confirmation=CONFIRMATION,
        expected_root_byte_count=1,
        expected_root_sha256="d" * 64,
        expected_year_bindings={},
    )

    assert result["reason"] == "OUTPUT_ROOT_COLLISION"
    assert calls == 0


def test_exclusive_durable_outputs_and_safe_serialization(tmp_path: Path) -> None:
    result = _run_synthetic(tmp_path, application=_fake_success_application([]))
    output_root = tmp_path / "output"
    assert result["status"] == "PASS"
    assert sorted(path.name for path in output_root.iterdir()) == [
        "attempt.json",
        "complete.json",
        "result.json",
    ]
    serialized = json.dumps(result, sort_keys=True) + json.dumps(
        [json.loads(path.read_text(encoding="utf-8")) for path in output_root.iterdir()],
        sort_keys=True,
    )
    assert ROOT_URL not in serialized
    assert ".html" not in serialized
    assert ".pdf" not in serialized
    assert str(tmp_path) not in serialized
    assert "href" not in serialized

    with pytest.raises(FileExistsError):
        runner._write_exclusive(output_root / "attempt.json", {"overwrite": True})


def test_input_files_are_read_once_each(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root_bytes = _root_page()
    pages = _year_pages()
    root_path = tmp_path / "root"
    root_path.write_bytes(root_bytes)
    page_paths = {}
    for year, payload in pages.items():
        path = tmp_path / str(year)
        path.write_bytes(payload)
        page_paths[year] = str(path)

    original = runner._read_once
    read_paths: list[str] = []

    def wrapped(path: str):
        read_paths.append(path)
        return original(path)

    monkeypatch.setattr(runner, "_read_once", wrapped)
    result = runner._run_with_expected_bindings(
        str(root_path),
        page_paths,
        str(tmp_path / "output"),
        expected_git_sha=VALID_GIT_SHA,
        confirmation=CONFIRMATION,
        expected_root_byte_count=len(root_bytes),
        expected_root_sha256=hashlib.sha256(root_bytes).hexdigest(),
        expected_year_bindings=_expected_bindings(pages),
        application=_fake_success_application([]),
    )

    assert result["status"] == "PASS"
    assert len(read_paths) == 6
    assert len(set(read_paths)) == 6


def _script_command(runner_path: Path, external_cwd: Path, output_root: Path, *, execute: bool, sha: str):
    command = [
        sys.executable,
        str(runner_path),
        "--preserved-root", str(external_cwd / "private-placeholder-root"),
        "--year-page-2017", str(external_cwd / "private-placeholder-2017"),
        "--year-page-2019", str(external_cwd / "private-placeholder-2019"),
        "--year-page-2020", str(external_cwd / "private-placeholder-2020"),
        "--year-page-2022", str(external_cwd / "private-placeholder-2022"),
        "--year-page-2026", str(external_cwd / "private-placeholder-2026"),
        "--output-root", str(output_root),
        "--expected-git-sha", sha,
        "--confirmation", CONFIRMATION,
    ]
    if execute:
        command.append("--execute-fixed-eight")
    return command


def test_direct_script_external_cwd_without_execute_is_safe(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    output = tmp_path / "output-no-execute"
    completed = subprocess.run(
        _script_command(Path(runner.__file__), external, output, execute=False, sha=VALID_GIT_SHA),
        cwd=external,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    receipt = json.loads(completed.stdout)
    assert receipt["reason"] == "EXECUTE_FIXED_EIGHT_REQUIRED"
    assert completed.stderr == ""
    assert not output.exists()


def test_direct_script_invalid_sha_external_cwd_fails_before_input_read(tmp_path: Path) -> None:
    external = tmp_path / "external"
    external.mkdir()
    output = tmp_path / "output-invalid-sha"
    completed = subprocess.run(
        _script_command(Path(runner.__file__), external, output, execute=True, sha="bad"),
        cwd=external,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    receipt = json.loads(completed.stdout)
    assert receipt["reason"] == "EXPECTED_GIT_SHA_INVALID"
    assert receipt["fixed_eight_invocations"] == 0
    assert completed.stderr == ""
    assert not output.exists()


def test_no_network_or_pdf_execution_surface_in_runner_source() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert not re.search(
        r"(?m)^\s*(?:from|import)\s+(?:requests|httpx|urllib|subprocess)\b",
        source,
    )
    assert "probe_calibration_bundle" not in source
    assert "T0" not in source
    assert "profitability" not in source
    assert "resolve_source_b_normal_month_object" not in source
