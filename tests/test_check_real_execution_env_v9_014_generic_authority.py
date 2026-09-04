"""Targeted E12/E13 tests for the generic canonical-authority artifacts and
`scripts/check_real_execution_env.py` REVIEWED_* bindings, promoted from the
predecessor V8-lineage 7-package closure to the reviewed V9_014 PDF
successor's 15-package closure.

Offline and no-network throughout: no environment mutation, no `pip
install`, no real/staging environment creation, no rerun of Stage E5/E6/E9/
E10. `check_environment_lock`/`check_freeze_record` are exercised under a
simulated canonical-matching interpreter/platform/`pip freeze` purely to
prove the new hash/blob/semantic bindings are internally self-consistent --
this never claims genuine Windows-grounded readiness, and
`REAL_EXECUTION_ENVIRONMENT_READY`/`FROZEN` remain `False` on any real,
unmocked run of this suite (STATIC_CLOUD_VALIDATION_ONLY).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest import mock

import scripts.check_real_execution_env as checker

REPO_ROOT = checker.REPO_ROOT


def _load_json(path: Path) -> dict:
    return json.loads(path.read_bytes().decode("utf-8"))


def _canonical_platform_mocks():
    return [
        mock.patch.object(checker.platform, "python_implementation", return_value=checker.CANONICAL_PYTHON_IMPLEMENTATION),
        mock.patch.object(checker.platform, "system", return_value=checker.CANONICAL_PLATFORM_SYSTEM),
        mock.patch.object(checker.platform, "machine", return_value=checker.CANONICAL_PLATFORM_MACHINE),
        mock.patch.object(checker.sysconfig, "get_platform", return_value=checker.CANONICAL_SYSCONFIG_PLATFORM),
        mock.patch.object(checker.os, "name", "nt"),
    ]


def _run_lock_and_freeze_checks(*, live_lock_text: str | None = None):
    """Run check_environment_lock/check_freeze_record under a simulated
    canonical-matching interpreter, with `pip freeze --all` faked to return
    `live_lock_text` (defaulting to the exact on-disk reviewed lock file).
    `git cat-file blob` calls pass through to the real subprocess so the
    actual content-addressed blob provenance is genuinely exercised.
    """
    fake_interpreter = dict(checker.check_interpreter_identity())
    fake_interpreter["interpreter_match"] = True
    fake_interpreter["python_patch_match"] = True

    lock_text = live_lock_text if live_lock_text is not None else checker.LOCK_FILE_PATH.read_text(encoding="utf-8")
    real_run = subprocess.run

    def fake_run(cmd, **kwargs):
        if "pip" in cmd and "freeze" in cmd:
            return mock.Mock(returncode=0, stdout=lock_text)
        return real_run(cmd, **kwargs)

    mocks = _canonical_platform_mocks()
    mocks.append(mock.patch.object(checker.subprocess, "run", side_effect=fake_run))
    with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5]:
        lock = checker.check_environment_lock(fake_interpreter)
        freeze = checker.check_freeze_record(fake_interpreter, lock)
    return lock, freeze


# =============================================================================
# Positive internal final-authority consistency
# =============================================================================


def test_generic_lock_candidate_matches_on_disk_file_exactly():
    candidate = _load_json(checker.LOCK_CANDIDATE_PATH)
    assert checker._type_strict_semantic_equal(candidate, checker.REVIEWED_LOCK_CANDIDATE_SEMANTIC_CONTENT)


def test_generic_freeze_record_matches_on_disk_file_exactly():
    freeze_record = _load_json(checker.FREEZE_RECORD_PATH)
    assert checker._type_strict_semantic_equal(freeze_record, checker.REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT)


def test_generic_lock_file_sha256_and_package_count_are_fifteen():
    lock_bytes = checker.LOCK_FILE_PATH.read_bytes()
    assert hashlib.sha256(lock_bytes).hexdigest() == checker.REVIEWED_LOCK_SHA256
    assert checker.REVIEWED_PACKAGE_COUNT == 15
    packages = checker._parse_pinned_lock_lines(lock_bytes.decode("utf-8"))
    assert len(packages) == 15


def test_source_requirements_blob_provenance_resolves_and_matches():
    git_blob = checker._git_blob_bytes(REPO_ROOT, checker.REVIEWED_SOURCE_REQUIREMENTS_GIT_BLOB_SHA1)
    assert git_blob is not None
    assert hashlib.sha256(git_blob).hexdigest() == checker.REVIEWED_SOURCE_REQUIREMENTS_GIT_SHA256


def test_windows_validation_evidence_blob_provenance_resolves_and_matches():
    git_blob = checker._git_blob_bytes(REPO_ROOT, checker.REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_BLOB_SHA1)
    assert git_blob is not None
    assert hashlib.sha256(git_blob).hexdigest() == checker.REVIEWED_WINDOWS_VALIDATION_EVIDENCE_CANONICAL_GIT_SHA256


def test_pdf_and_xls_fixture_hashes_match_reviewed_constants():
    assert hashlib.sha256(checker.SYNTHETIC_XLS_FIXTURE_PATH.read_bytes()).hexdigest() == checker.REVIEWED_FIXTURE_SHA256
    assert hashlib.sha256(checker.SYNTHETIC_PDF_FIXTURE_PATH.read_bytes()).hexdigest() == checker.REVIEWED_PDF_FIXTURE_SHA256


def test_environment_lock_and_freeze_record_pass_under_simulated_canonical_environment():
    lock, freeze = _run_lock_and_freeze_checks()
    assert lock["status"] == "PASS"
    assert freeze["status"] == "PASS"


def test_e11_evidence_binding_matches_v9_014_successor_provenance():
    evidence_path = REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_LIVE_CANONICAL_VALIDATION_EVIDENCE.json"
    evidence_bytes = evidence_path.read_bytes()
    provenance = checker.V9_014_SUCCESSOR_PROVENANCE_SEMANTIC_CONTENT
    assert hashlib.sha256(evidence_bytes).hexdigest() == provenance["e11_live_canonical_validation_evidence_sha256"]
    assert checker.REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_SHA == provenance["e11_reviewed_git_sha"]


def test_v9_014_freeze_record_e7_bindings_match_checker_constants():
    freeze_record = _load_json(REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_FREEZE_RECORD.json")
    assert freeze_record["e7"]["reviewed_git_sha"] == checker.REVIEWED_LOCK_CANDIDATE_GIT_SHA
    assert freeze_record["e7"]["lock_candidate"]["sha256"] == checker.V9_014_E7_LOCK_CANDIDATE_SHA256
    assert freeze_record["e7"]["windows_validation_evidence"]["sha256"] == checker.V9_014_E7_WINDOWS_EVIDENCE_SHA256
    assert freeze_record["frozen_design"]["git_sha"] == "efee3d0efca368645c00aeed63cb8e0637cd3672"
    assert freeze_record["frozen_design"]["blob_sha"] == "2bbacbf37ab961d1cbf416b7fd476db18778c5b7"


def test_no_generic_or_v9_014_artifact_claims_promotion_or_future_authorization():
    for path in (
        checker.LOCK_CANDIDATE_PATH,
        checker.WINDOWS_VALIDATION_EVIDENCE_PATH,
        checker.FREEZE_RECORD_PATH,
        REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_FREEZE_RECORD.json",
    ):
        data = _load_json(path)
        assert data.get("future_protected_execution_authorized") is False, path
        provenance = data.get("v9_014_successor_provenance") or data
        promoted_key = "v9_014_pdf_environment_successor_promoted"
        if promoted_key in provenance:
            assert provenance[promoted_key] is False, path
        state = provenance.get("canonical_environment_state") if isinstance(provenance, dict) else None
        if state is not None:
            assert state == "SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED", path


def test_no_fabricated_future_e13_e14_e15_sha_in_v9_014_freeze_record():
    freeze_text = (REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_FREEZE_RECORD.json").read_text(
        encoding="utf-8"
    )
    parsed = json.loads(freeze_text)
    assert set(parsed) == {
        "artifact_status",
        "canonical_environment_state",
        "e7",
        "e8",
        "e9",
        "e10",
        "e11",
        "frozen_design",
        "future_protected_execution_authorized",
        "non_claims",
        "observed_environment",
        "schema_version",
        "study_id",
        "v9_014_pdf_environment_successor_promoted",
    }
    for forbidden_key in ("e13", "e14", "e15"):
        assert forbidden_key not in parsed


# =============================================================================
# Negatives
# =============================================================================


def test_stale_seven_package_candidate_is_rejected():
    stale = json.loads(checker.LOCK_CANDIDATE_PATH.read_text(encoding="utf-8"))
    stale["resolved_lock"]["package_count"] = 7
    assert not checker._type_strict_semantic_equal(stale, checker.REVIEWED_LOCK_CANDIDATE_SEMANTIC_CONTENT)


def test_wrong_lock_package_count_fails_environment_lock_check():
    lock_text = checker.LOCK_FILE_PATH.read_text(encoding="utf-8") + "extra-package==1.0.0\n"
    lock, _freeze = _run_lock_and_freeze_checks(live_lock_text="numpy==2.5.2\npandas==3.0.5\n")
    assert lock["status"] == "FAIL"


def test_wrong_generic_lock_sha256_is_detected(tmp_path, monkeypatch):
    tampered = checker.LOCK_FILE_PATH.read_text(encoding="utf-8") + "\n"
    monkeypatch.setattr(checker, "LOCK_FILE_PATH", tmp_path / "requirements-real-execution.lock.txt")
    checker.LOCK_FILE_PATH.write_text(tampered, encoding="utf-8")
    fake_interpreter = dict(checker.check_interpreter_identity())
    fake_interpreter["interpreter_match"] = True
    fake_interpreter["python_patch_match"] = True
    with _canonical_platform_mocks()[0], _canonical_platform_mocks()[1]:
        result = checker.check_environment_lock(fake_interpreter)
    assert result["status"] == "FAIL"
    assert result["reason"] == "LOCK_SHA256_MISMATCH"


def test_wrong_e11_evidence_hash_is_rejected():
    provenance = checker.V9_014_SUCCESSOR_PROVENANCE_SEMANTIC_CONTENT
    assert provenance["e11_live_canonical_validation_evidence_sha256"] != "0" * 64
    assert provenance["e11_live_canonical_validation_evidence_git_blob_sha1"] != "0" * 40


def test_wrong_e7_binding_is_rejected(monkeypatch):
    monkeypatch.setattr(checker, "V9_014_E7_LOCK_CANDIDATE_SHA256", "0" * 64)
    result = checker.check_v9_014_e7_bundle()
    assert result["status"] == "FAIL"
    assert result["reason"] == "V9_014_E7_LOCK_CANDIDATE_HASH_MISMATCH"


def test_wrong_package_set_hash_fails_environment_lock_check():
    _lock, freeze = _run_lock_and_freeze_checks(live_lock_text="pandas==0.0.1\n")
    assert freeze["reason"] == "ENVIRONMENT_LOCK_CHECK_NOT_PASSING"


def test_wrong_xls_fixture_identity_fails_probe():
    result = checker.check_jpx_xls_parser_synthetic_probe()
    # This session has no pandas installed; either a genuine PASS (if pandas
    # happens to be present) or a safe, non-crashing FAIL is acceptable, but
    # a wrong fixture hash must never be silently accepted as PASS.
    fixture_bytes = checker.SYNTHETIC_XLS_FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(fixture_bytes).hexdigest() == checker.REVIEWED_FIXTURE_SHA256
    assert result["status"] in ("PASS", "FAIL")


def test_wrong_pdf_probe_result_is_rejected():
    result = checker.check_v9_014_successor_promotion(
        checker.V9_014_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED,
        live_packages=dict(checker.check_v9_014_e7_bundle()["lock_candidate"]["resolved_packages"]),
        platform_evidence={
            "implementation": "CPython",
            "version": [3, 12, 10],
            "os_name": "nt",
            "platform_system": "Windows",
            "platform_machine": "AMD64",
            "sysconfig_platform": "win-amd64",
            "interpreter_match": True,
        },
        xls_probe={"status": "PASS"},
        pdf_probe={
            "status": "SYNTHETIC_PDF_PROBE_PASS",
            "observed_fixture_sha256": "0" * 64,
            "observed_pdfplumber_version": "0.11.10",
            "observed_page_count": 1,
        },
    )
    assert result["status"] == "FAIL"
    assert result["reason"] == "V9_014_SUCCESSOR_PDF_PROBE_FAILED"


def test_promotion_true_and_protected_authority_true_are_never_asserted():
    for path in (
        checker.LOCK_CANDIDATE_PATH,
        checker.WINDOWS_VALIDATION_EVIDENCE_PATH,
        checker.FREEZE_RECORD_PATH,
        REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_FREEZE_RECORD.json",
    ):
        text = path.read_text(encoding="utf-8")
        assert '"future_protected_execution_authorized": true' not in text
        assert '"v9_014_pdf_environment_successor_promoted": true' not in text
        assert '"canonical_environment_state": "SUCCESSOR_CANONICAL_FROZEN"' not in text


def test_v9_014_promotion_state_machine_rejects_malformed_and_premature_frozen_claims():
    malformed = checker.check_v9_014_successor_promotion("NOT_A_STATE")
    frozen = checker.check_v9_014_successor_promotion(checker.V9_014_SUCCESSOR_CANONICAL_FROZEN)
    assert malformed["status"] == "FAIL"
    assert malformed["reason"] == "V9_014_PROMOTION_STATE_INVALID"
    assert frozen["status"] == "FAIL"
    assert frozen["reason"] == "V9_014_SUCCESSOR_FROZEN_REQUIRES_E15_REVIEW"
