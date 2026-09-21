"""No-network tests for the current V10C protected-environment authority."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import scripts.check_current_protected_environment as checker
import scripts.check_real_execution_env as historical_checker


def _real_lock_text() -> str:
    return checker.CURRENT_LOCK_PATH.read_text(encoding="utf-8")


def _distribution_records(package_map: dict[str, str]) -> list[SimpleNamespace]:
    return [SimpleNamespace(metadata={"Name": name}, version=version) for name, version in package_map.items()]


def test_current_authority_resolves_to_reviewed_v10c_27_package_chain() -> None:
    result = checker.resolve_current_authority()
    assert result["status"] == "PASS"
    assert result["study"] == "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR"
    assert result["reviewed_sha"] == checker.CURRENT_AUTHORITY_REVIEWED_SHA
    assert result["package_count"] == 27
    assert result["historical_predecessor_package_count"] == 20


def test_historical_fifteen_package_checker_remains_independently_bound() -> None:
    assert historical_checker.REVIEWED_PACKAGE_COUNT == 15
    blob = historical_checker._git_blob_bytes(
        historical_checker.REPO_ROOT, historical_checker.REVIEWED_LOCK_GIT_BLOB_SHA1
    )
    assert blob is not None
    assert len(historical_checker._parse_pinned_lock_lines(blob.decode("utf-8"))) == 15
    assert historical_checker.REVIEWED_LOCK_SHA256 != checker.CURRENT_AUTHORITY_LOCK_SHA256


def test_current_checker_does_not_compare_against_historical_lock(monkeypatch) -> None:
    calls: list[str] = []
    real_reader = checker._git_blob_bytes

    def recording_reader(repo_root: Path, git_ref: str):
        calls.append(git_ref)
        return real_reader(repo_root, git_ref)

    result = checker.resolve_current_authority(git_blob_reader=recording_reader)
    assert result["status"] == "PASS"
    assert any(checker.CURRENT_AUTHORITY_REVIEWED_SHA in item for item in calls)
    assert not any(historical_checker.REVIEWED_LOCK_GIT_BLOB_SHA1 in item for item in calls)


def test_current_package_set_matches_the_resolved_27_package_lock() -> None:
    authority = checker.resolve_current_authority()
    result = checker.check_live_package_set(
        authority["package_map"], _distribution_records(authority["package_map"])
    )
    assert result == {"status": "PASS", "package_count": 27}


def test_current_package_observer_uses_exact_installed_metadata() -> None:
    authority = checker.resolve_current_authority()
    result = checker.check_live_package_set(
        authority["package_map"], _distribution_records(authority["package_map"])
    )
    assert result == {"status": "PASS", "package_count": 27}


def test_direct_reference_pip_freeze_presentation_does_not_control_acceptance(monkeypatch) -> None:
    authority = checker.resolve_current_authority()
    metadata_distributions = _distribution_records(authority["package_map"])
    monkeypatch.setattr(checker.importlib.metadata, "distributions", lambda: metadata_distributions)
    with mock.patch.object(
        checker.subprocess,
        "run",
        side_effect=AssertionError("pip freeze must not be consulted"),
    ):
        result = checker.check_live_package_set(authority["package_map"])
    assert result == {"status": "PASS", "package_count": 27}


def test_missing_extra_and_version_drift_metadata_fail_closed() -> None:
    authority = checker.resolve_current_authority()
    package_map = dict(authority["package_map"])
    package_map.pop("scipy")
    package_map["unexpected-package"] = "1.0.0"
    package_map["lightgbm"] = "4.5.0"
    result = checker.check_live_package_set(
        authority["package_map"], _distribution_records(package_map)
    )
    assert result["status"] == "FAIL"
    assert result["reason"] == "CURRENT_AUTHORITY_PACKAGE_SET_MISMATCH"
    assert result["missing_packages"] == ["scipy"]
    assert result["extra_packages"] == ["unexpected-package"]
    assert result["version_mismatches"] == ["lightgbm"]


def test_duplicate_normalized_metadata_fails_closed() -> None:
    authority = checker.resolve_current_authority()
    records = _distribution_records(authority["package_map"])
    records.append(SimpleNamespace(metadata={"Name": "pandas.market_calendars"}, version="2.1.4"))
    result = checker.check_live_package_set(authority["package_map"], records)
    assert result == {
        "status": "FAIL",
        "reason": "CURRENT_AUTHORITY_INSTALLED_METADATA_DUPLICATE",
    }


def test_malformed_metadata_fails_closed() -> None:
    authority = checker.resolve_current_authority()
    records = _distribution_records(authority["package_map"])
    records[0] = SimpleNamespace(metadata={}, version="1.0.0")
    result = checker.check_live_package_set(authority["package_map"], records)
    assert result == {
        "status": "FAIL",
        "reason": "CURRENT_AUTHORITY_INSTALLED_METADATA_MALFORMED",
    }


def test_missing_current_authority_fails_closed(tmp_path: Path) -> None:
    state = checker.PROJECT_STATE_PATH.read_text(encoding="utf-8").replace(
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_STATE=CANONICAL_FROZEN",
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_STATE=",
        1,
    )
    state_path = tmp_path / "PROJECT_STATE.md"
    state_path.write_text(state, encoding="utf-8")
    result = checker.resolve_current_authority(state_path=state_path)
    assert result["status"] == "FAIL"
    assert result["reason"] == "CURRENT_AUTHORITY_STATE_VALUE_MISMATCH"


def test_ambiguous_current_authority_fails_closed(tmp_path: Path) -> None:
    state_path = tmp_path / "PROJECT_STATE.md"
    state_path.write_text(
        checker.PROJECT_STATE_PATH.read_text(encoding="utf-8")
        + "\nV10C_T0_CANONICAL_ML_ENVIRONMENT_STATE=CANONICAL_FROZEN\n",
        encoding="utf-8",
    )
    result = checker.resolve_current_authority(state_path=state_path)
    assert result["status"] == "FAIL"
    assert result["reason"] == "CURRENT_AUTHORITY_STATE_KEY_AMBIGUOUS"


def test_current_lock_drift_fails_closed(tmp_path: Path) -> None:
    lock_path = tmp_path / checker.CURRENT_LOCK_PATH.name
    lock_path.write_text(_real_lock_text().replace("scipy==1.18.1", "scipy==1.18.0"), encoding="utf-8")
    result = checker.resolve_current_authority(lock_path=lock_path)
    assert result["status"] == "FAIL"
    assert result["reason"] == "CURRENT_AUTHORITY_WORKING_ARTIFACT_MISMATCH"


def test_wrong_interpreter_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(checker.sys, "executable", str(checker.REPO_ROOT / ".venv" / "Scripts" / "python.exe"))
    result = checker.check_interpreter_identity()
    assert result["status"] == "FAIL"
    assert result["failure_class"] == "PRE_GATE_WRONG_PYTHON_ENVIRONMENT"


def test_wrong_python_patch_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(checker.sys, "version_info", (3, 13, 0))
    result = checker.check_interpreter_identity()
    assert result["status"] == "FAIL"
    assert result["python_patch_match"] is False
    assert result["failure_class"] == "PRE_GATE_WRONG_PYTHON_ENVIRONMENT"


def test_wrong_platform_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(checker.platform, "system", lambda: "Linux")
    result = checker.check_interpreter_identity()
    assert result["status"] == "FAIL"
    assert result["platform_match"] is False
    assert result["failure_class"] == "PRE_GATE_WRONG_PYTHON_ENVIRONMENT"


def test_package_drift_fails_closed() -> None:
    packages = checker.resolve_current_authority()["package_map"]
    drifted = dict(packages)
    drifted["lightgbm"] = "4.5.0"
    result = checker.check_live_package_set(packages, _distribution_records(drifted))
    assert result["status"] == "FAIL"
    assert result["reason"] == "CURRENT_AUTHORITY_PACKAGE_SET_MISMATCH"
    assert "lightgbm" in result["version_mismatches"]


def test_current_readiness_does_not_run_pip_when_authority_or_interpreter_fails(monkeypatch) -> None:
    monkeypatch.setattr(checker, "resolve_current_authority", lambda: {"status": "FAIL", "reason": "ambiguous"})
    monkeypatch.setattr(checker, "check_interpreter_identity", lambda: {"status": "FAIL"})
    with mock.patch.object(checker.subprocess, "run") as run:
        result = checker.run_current_readiness()
    assert result["CURRENT_ENVIRONMENT_READY"] is False
    run.assert_not_called()


def test_pdf_probe_pass_is_required_for_current_readiness(monkeypatch) -> None:
    authority = checker.resolve_current_authority()
    monkeypatch.setattr(checker, "resolve_current_authority", lambda: authority)
    monkeypatch.setattr(checker, "check_interpreter_identity", lambda: {"status": "PASS"})
    monkeypatch.setattr(
        checker,
        "check_live_package_set",
        lambda expected: {"status": "PASS", "package_count": 27},
    )
    monkeypatch.setattr(
        checker,
        "check_pdf_parser_synthetic_probe",
        lambda: {"status": "FAIL", "reason": "probe failure"},
    )
    result = checker.run_current_readiness()
    assert result["PDF_OPERATIONAL_PROBE"]["status"] == "FAIL"
    assert result["CURRENT_ENVIRONMENT_READY"] is False


def test_pdf_probe_pass_allows_current_readiness(monkeypatch) -> None:
    authority = checker.resolve_current_authority()
    monkeypatch.setattr(checker, "resolve_current_authority", lambda: authority)
    monkeypatch.setattr(checker, "check_interpreter_identity", lambda: {"status": "PASS"})
    monkeypatch.setattr(
        checker,
        "check_live_package_set",
        lambda expected: {"status": "PASS", "package_count": 27},
    )
    monkeypatch.setattr(
        checker,
        "check_pdf_parser_synthetic_probe",
        lambda: {"status": "PASS"},
    )
    result = checker.run_current_readiness()
    assert result["CURRENT_ENVIRONMENT_READY"] is True


def test_pdf_probe_failure_and_exception_are_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        checker,
        "_run_reviewed_pdf_probe",
        lambda fixture_path: SimpleNamespace(
            status="WRONG", observed_fixture_sha256=checker.PDF_PROBE_EXPECTED_FIXTURE_SHA256
        ),
    )
    assert checker.check_pdf_parser_synthetic_probe()["status"] == "FAIL"
    monkeypatch.setattr(checker, "_run_reviewed_pdf_probe", mock.Mock(side_effect=RuntimeError("boom")))
    assert checker.check_pdf_parser_synthetic_probe() == {
        "status": "FAIL",
        "reason": "CURRENT_AUTHORITY_PDF_PROBE_EXCEPTION",
    }


def test_pdf_probe_wrong_fixture_identity_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        checker,
        "_run_reviewed_pdf_probe",
        lambda fixture_path: SimpleNamespace(
            status=checker.PDF_PROBE_PASS,
            observed_fixture_sha256="0" * 64,
            observed_pdfplumber_version=checker.PDF_PROBE_REQUIRED_PDFPLUMBER_VERSION,
            observed_page_count=checker.PDF_PROBE_EXPECTED_PAGE_COUNT,
        ),
    )
    result = checker.check_pdf_parser_synthetic_probe()
    assert result == {
        "status": "FAIL",
        "reason": "CURRENT_AUTHORITY_PDF_PROBE_FIXTURE_IDENTITY_MISMATCH",
    }


def test_pdf_probe_success_validates_reviewed_runtime_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        checker,
        "_run_reviewed_pdf_probe",
        lambda fixture_path: SimpleNamespace(
            status=checker.PDF_PROBE_PASS,
            observed_fixture_sha256=checker.PDF_PROBE_EXPECTED_FIXTURE_SHA256,
            observed_pdfplumber_version=checker.PDF_PROBE_REQUIRED_PDFPLUMBER_VERSION,
            observed_page_count=checker.PDF_PROBE_EXPECTED_PAGE_COUNT,
        ),
    )
    assert checker.check_pdf_parser_synthetic_probe() == {
        "status": "PASS",
        "fixture_sha256": checker.PDF_PROBE_EXPECTED_FIXTURE_SHA256,
        "pdfplumber_version": checker.PDF_PROBE_REQUIRED_PDFPLUMBER_VERSION,
        "page_count": checker.PDF_PROBE_EXPECTED_PAGE_COUNT,
    }


def test_pdf_probe_wrong_runtime_version_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        checker,
        "_run_reviewed_pdf_probe",
        lambda fixture_path: SimpleNamespace(
            status=checker.PDF_PROBE_PASS,
            observed_fixture_sha256=checker.PDF_PROBE_EXPECTED_FIXTURE_SHA256,
            observed_pdfplumber_version="0.0.0",
            observed_page_count=checker.PDF_PROBE_EXPECTED_PAGE_COUNT,
        ),
    )
    assert checker.check_pdf_parser_synthetic_probe() == {
        "status": "FAIL",
        "reason": "CURRENT_AUTHORITY_PDF_PROBE_VERSION_MISMATCH",
    }
