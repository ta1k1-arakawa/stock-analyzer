"""Offline E8 tests for the future V9_014 canonical-promotion tooling."""

from __future__ import annotations

import scripts.check_real_execution_env as checker


def _platform_evidence() -> dict[str, object]:
    return {
        "implementation": "CPython",
        "version": [3, 12, 10],
        "os_name": "nt",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "interpreter_match": True,
    }


def _pdf_probe(*, fixture_sha256: str | None = None) -> dict[str, object]:
    return {
        "status": "SYNTHETIC_PDF_PROBE_PASS",
        "observed_fixture_sha256": fixture_sha256 or "5eecb758a50e829af16bd42833f89a8329bfaaaa561aee209fbd2249b507b413",
        "observed_pdfplumber_version": "0.11.10",
        "observed_page_count": 1,
    }


def test_e8_e7_bundle_binding_passes():
    result = checker.check_v9_014_e7_bundle()
    assert result["status"] == "PASS"
    assert result["detail"]["resolved_package_count"] == 15
    assert result["detail"]["predecessor_package_count"] == 7


def test_e8_predecessor_state_is_distinguished_and_never_authorizes_mutation():
    result = checker.check_v9_014_successor_promotion(checker.V9_014_PREDECESSOR_CANONICAL_FROZEN)
    assert result["status"] == "PASS"
    assert result["canonical_environment_mutation_authorized"] is False
    assert result["successor_live_validation"] == "NOT_RUN_PREDECESSOR_STATE"


def test_e8_migration_validation_accepts_exact_reviewed_closure():
    bundle = checker.check_v9_014_e7_bundle()
    result = checker.check_v9_014_successor_promotion(
        checker.V9_014_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED,
        live_packages=dict(bundle["lock_candidate"]["resolved_packages"]),
        platform_evidence=_platform_evidence(),
        xls_probe={"status": "PASS"},
        pdf_probe=_pdf_probe(),
    )
    assert result["status"] == "PASS"
    assert result["canonical_environment_mutation_authorized"] is False
    assert result["successor_package_count"] == 15


def test_e8_migration_validation_rejects_wrong_package_set_and_predecessor_drift():
    bundle = checker.check_v9_014_e7_bundle()
    packages = dict(bundle["lock_candidate"]["resolved_packages"])
    packages["pandas"] = "0.0.0"
    result = checker.check_v9_014_successor_promotion(
        checker.V9_014_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED,
        live_packages=packages,
        platform_evidence=_platform_evidence(),
        xls_probe={"status": "PASS"},
        pdf_probe=_pdf_probe(),
    )
    assert result["status"] == "FAIL"
    assert result["reason"] == "V9_014_SUCCESSOR_PACKAGE_SET_MISMATCH"


def test_e8_migration_validation_rejects_wrong_pdf_probe_identity():
    bundle = checker.check_v9_014_e7_bundle()
    result = checker.check_v9_014_successor_promotion(
        checker.V9_014_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED,
        live_packages=dict(bundle["lock_candidate"]["resolved_packages"]),
        platform_evidence=_platform_evidence(),
        xls_probe={"status": "PASS"},
        pdf_probe=_pdf_probe(fixture_sha256="0" * 64),
    )
    assert result["status"] == "FAIL"
    assert result["reason"] == "V9_014_SUCCESSOR_PDF_PROBE_FAILED"


def test_e8_bundle_rejects_wrong_reviewed_lock_identity(monkeypatch):
    monkeypatch.setattr(checker, "V9_014_E7_LOCK_CANDIDATE_SHA256", "0" * 64)
    result = checker.check_v9_014_e7_bundle()
    assert result["status"] == "FAIL"
    assert result["reason"] == "V9_014_E7_LOCK_CANDIDATE_HASH_MISMATCH"


def test_e8_malformed_or_future_frozen_promotion_state_fails_closed():
    malformed = checker.check_v9_014_successor_promotion("NOT_A_STATE")
    frozen = checker.check_v9_014_successor_promotion(checker.V9_014_SUCCESSOR_CANONICAL_FROZEN)
    assert malformed["status"] == "FAIL"
    assert malformed["reason"] == "V9_014_PROMOTION_STATE_INVALID"
    assert frozen["status"] == "FAIL"
    assert frozen["reason"] == "V9_014_SUCCESSOR_FROZEN_REQUIRES_E15_REVIEW"


def test_e8_checker_and_bootstrap_expose_the_same_explicit_state_machine():
    bootstrap = (checker.REPO_ROOT / "scripts" / "bootstrap_real_execution_env.ps1").read_text(encoding="utf-8")
    checker_source = (checker.REPO_ROOT / "scripts" / "check_real_execution_env.py").read_text(encoding="utf-8")
    for state in checker.V9_014_PROMOTION_STATES:
        assert state in bootstrap
        assert state in checker_source
    assert "--v9-014-promotion-state" in checker_source
    assert "V9_014 E7 candidate/evidence closure verified" in bootstrap
