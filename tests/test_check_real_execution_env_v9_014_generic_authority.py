"""Targeted E12/E13 tests for the generic canonical-authority artifacts and
`scripts/check_real_execution_env.py` REVIEWED_* bindings, promoted from the
predecessor V8-lineage 7-package closure to the reviewed V9_014 PDF
successor's 15-package closure.

E13 MEDIUM_2 remediation (E12_TARGETED_TEST_EVIDENCE_NOT_FULLY_EFFECTIVE_OR_
SCOPE_COMPLIANT): every negative test below genuinely perturbs an input or
identity and calls the real validator/checker function, asserting a specific
FAIL status/reason -- never a bare "differs from an all-zero placeholder"
assertion, and never a `status in ("PASS", "FAIL")` non-assertion.

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

import pytest

import scripts.check_real_execution_env as checker

REPO_ROOT = checker.REPO_ROOT
V9_014_LIVE_EVIDENCE_PATH = (
    REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_LIVE_CANONICAL_VALIDATION_EVIDENCE.json"
)
V9_014_FREEZE_RECORD_PATH = REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_FREEZE_RECORD.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_bytes().decode("utf-8"))


def _real_e11_evidence() -> dict:
    """The genuine, unmodified E11 live evidence, fetched via its real
    reviewed Git blob SHA-1 -- the exact same path `check_e11_live_evidence_
    binding()` itself uses when `evidence` is not injected. Callers mutate a
    *copy* of this dict to build a real negative-test perturbation.
    """
    git_blob = checker._git_blob_bytes(REPO_ROOT, checker.V9_014_E11_LIVE_EVIDENCE_GIT_BLOB_SHA1)
    assert git_blob is not None
    return json.loads(git_blob.decode("utf-8"))


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


def test_canonical_lock_git_blob_sha256_and_package_closure_are_fifteen():
    lock_bytes = checker._git_blob_bytes(REPO_ROOT, checker.REVIEWED_LOCK_GIT_BLOB_SHA1)
    assert lock_bytes is not None
    assert hashlib.sha256(lock_bytes).hexdigest() == checker.REVIEWED_LOCK_SHA256
    assert checker.REVIEWED_PACKAGE_COUNT == 15
    packages = checker._parse_pinned_lock_lines(lock_bytes.decode("utf-8"))
    assert packages == checker.REVIEWED_LOCK_PACKAGE_MAP


def test_source_requirements_blob_provenance_resolves_and_matches():
    git_blob = checker._git_blob_bytes(REPO_ROOT, checker.REVIEWED_SOURCE_REQUIREMENTS_GIT_BLOB_SHA1)
    assert git_blob is not None
    assert hashlib.sha256(git_blob).hexdigest() == checker.REVIEWED_SOURCE_REQUIREMENTS_GIT_SHA256


def test_windows_validation_evidence_blob_provenance_resolves_and_matches():
    git_blob = checker._git_blob_bytes(REPO_ROOT, checker.REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_BLOB_SHA1)
    assert git_blob is not None
    assert hashlib.sha256(git_blob).hexdigest() == checker.REVIEWED_WINDOWS_VALIDATION_EVIDENCE_CANONICAL_GIT_SHA256


def test_generic_candidate_blob_provenance_resolves_and_matches():
    """E13 MEDIUM_1: the FINAL generic candidate is now bound by its own
    exact blob SHA-1, independently of Stage E7's reviewed commit."""
    git_blob = checker._git_blob_bytes(REPO_ROOT, checker.REVIEWED_GENERIC_LOCK_CANDIDATE_GIT_BLOB_SHA1)
    assert git_blob is not None
    assert hashlib.sha256(git_blob).hexdigest() == checker.REVIEWED_GENERIC_LOCK_CANDIDATE_CANONICAL_GIT_SHA256
    candidate = json.loads(git_blob.decode("utf-8"))
    assert checker._type_strict_semantic_equal(candidate, checker.REVIEWED_LOCK_CANDIDATE_SEMANTIC_CONTENT)


def test_freeze_record_reviewed_lock_candidate_block_matches_working_tree_candidate():
    freeze_record = _load_json(checker.FREEZE_RECORD_PATH)
    ref = freeze_record["reviewed_lock_candidate"]
    assert ref["git_blob_sha1"] == checker.REVIEWED_GENERIC_LOCK_CANDIDATE_GIT_BLOB_SHA1
    assert ref["canonical_git_sha256"] == checker.REVIEWED_GENERIC_LOCK_CANDIDATE_CANONICAL_GIT_SHA256
    working_tree_bytes = checker.LOCK_CANDIDATE_PATH.read_bytes()
    result = subprocess.run(
        ["git", "hash-object", "--stdin"],
        input=working_tree_bytes,
        capture_output=True,
        check=True,
    )
    assert result.stdout.decode("ascii").strip() == ref["git_blob_sha1"]


def test_pdf_and_xls_fixture_hashes_match_reviewed_constants():
    assert hashlib.sha256(checker.SYNTHETIC_XLS_FIXTURE_PATH.read_bytes()).hexdigest() == checker.REVIEWED_FIXTURE_SHA256
    assert hashlib.sha256(checker.SYNTHETIC_PDF_FIXTURE_PATH.read_bytes()).hexdigest() == checker.REVIEWED_PDF_FIXTURE_SHA256


def test_environment_lock_and_freeze_record_pass_under_simulated_canonical_environment():
    lock, freeze = _run_lock_and_freeze_checks()
    assert lock["status"] == "PASS"
    assert freeze["status"] == "PASS"
    assert freeze["detail"]["e11_live_evidence_binding"]["status"] == "PASS"
    assert freeze["detail"]["candidate_git_semantic_match"] is True


def test_crlf_working_tree_lock_is_semantically_equivalent_to_canonical_git_blob(tmp_path, monkeypatch):
    crlf_lock = tmp_path / "requirements-real-execution.lock.txt"
    crlf_lock.write_bytes(checker.LOCK_FILE_PATH.read_bytes().replace(b"\n", b"\r\n"))
    monkeypatch.setattr(checker, "LOCK_FILE_PATH", crlf_lock)

    lock, _freeze = _run_lock_and_freeze_checks()

    assert lock["status"] == "PASS"
    assert lock["detail"]["canonical_lock_git_blob_sha1"] == checker.REVIEWED_LOCK_GIT_BLOB_SHA1
    assert lock["detail"]["canonical_lock_package_authority_match"] is True
    assert lock["detail"]["working_lock_semantic_match"] is True


def test_e11_live_evidence_binding_passes_on_the_real_committed_artifact():
    """Real, non-injected exercise of check_e11_live_evidence_binding():
    fetches the actual reviewed Git blob and validates every mechanical
    sub-check against it."""
    result = checker.check_e11_live_evidence_binding()
    assert result["status"] == "PASS"
    assert all(result["detail"].values())


def test_v9_014_freeze_record_e7_bindings_match_checker_constants():
    freeze_record = _load_json(V9_014_FREEZE_RECORD_PATH)
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
        V9_014_FREEZE_RECORD_PATH,
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
    parsed = _load_json(V9_014_FREEZE_RECORD_PATH)
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
# Negatives -- every case below genuinely perturbs an input/identity and
# calls the real validator/checker, asserting a specific FAIL status/reason.
# =============================================================================


def test_stale_seven_package_generic_candidate_is_rejected():
    stale = json.loads(checker.LOCK_CANDIDATE_PATH.read_text(encoding="utf-8"))
    stale["resolved_lock"]["package_count"] = 7
    assert not checker._type_strict_semantic_equal(stale, checker.REVIEWED_LOCK_CANDIDATE_SEMANTIC_CONTENT)


def test_stale_seven_package_e11_evidence_is_rejected():
    tampered = dict(_real_e11_evidence())
    tampered["observed_environment"] = dict(tampered["observed_environment"])
    tampered["observed_environment"]["package_set"] = dict(tampered["observed_environment"]["package_set"])
    tampered["observed_environment"]["package_set"]["count"] = 7
    result = checker.check_e11_live_evidence_binding(evidence=tampered)
    assert result["status"] == "FAIL"
    assert result["reason"] == "E11_LIVE_EVIDENCE_SEMANTIC_MISMATCH"
    assert "package_set_count" in result["detail"]["failing_checks"]


def test_wrong_generic_lock_missing_package_fails_environment_lock_check():
    lock, _freeze = _run_lock_and_freeze_checks(live_lock_text="numpy==2.5.2\npandas==3.0.5\n")
    assert lock["status"] == "FAIL"
    assert lock["reason"] == "PIP_FREEZE_PACKAGE_SET_MISMATCH"
    assert set(lock["detail"]["missing_packages"]) >= {"pdfplumber", "xlrd"}


def test_wrong_generic_lock_package_version_fails_environment_lock_check():
    live_lock_text = checker.LOCK_FILE_PATH.read_text(encoding="utf-8").replace("pandas==3.0.5", "pandas==2.0.0")
    lock, _freeze = _run_lock_and_freeze_checks(live_lock_text=live_lock_text)
    assert lock["status"] == "FAIL"
    assert lock["reason"] == "PIP_FREEZE_PACKAGE_SET_MISMATCH"
    assert "pandas" in lock["detail"]["version_mismatched_packages"]


@pytest.mark.parametrize(
    ("replacement", "expected_reason"),
    [
        ("pandas==2.0.0", "LOCK_WORKING_TREE_SEMANTIC_MISMATCH"),
        ("", "LOCK_WORKING_TREE_SEMANTIC_MISMATCH"),
        ("extra-package==1.0.0\n", "LOCK_WORKING_TREE_SEMANTIC_MISMATCH"),
        ("not-a-pinned-package", "LOCK_WORKING_TREE_SEMANTIC_MISMATCH"),
    ],
)
def test_working_tree_lock_semantic_drift_is_rejected(tmp_path, monkeypatch, replacement, expected_reason):
    working_lock = tmp_path / "requirements-real-execution.lock.txt"
    source = checker.LOCK_FILE_PATH.read_text(encoding="utf-8")
    if replacement:
        tampered = source.replace("pandas==3.0.5", replacement)
    else:
        tampered = source.replace("pdfplumber==0.11.10\n", "")
    if replacement.startswith("extra-package"):
        tampered = source + replacement
    working_lock.write_text(tampered, encoding="utf-8")
    monkeypatch.setattr(checker, "LOCK_FILE_PATH", working_lock)

    lock, _freeze = _run_lock_and_freeze_checks()

    assert lock["status"] == "FAIL"
    assert lock["reason"] == expected_reason


def test_wrong_canonical_lock_sha256_is_detected(monkeypatch):
    monkeypatch.setattr(checker, "REVIEWED_LOCK_SHA256", "0" * 64)
    fake_interpreter = dict(checker.check_interpreter_identity())
    result = checker.check_environment_lock(fake_interpreter)
    assert result["status"] == "FAIL"
    assert result["reason"] == "LOCK_GIT_PROVENANCE_MISMATCH"


def test_wrong_canonical_lock_git_blob_is_detected(monkeypatch):
    monkeypatch.setattr(checker, "REVIEWED_LOCK_GIT_BLOB_SHA1", "0" * 40)
    fake_interpreter = dict(checker.check_interpreter_identity())
    result = checker.check_environment_lock(fake_interpreter)
    assert result["status"] == "FAIL"
    assert result["reason"] == "LOCK_GIT_PROVENANCE_UNAVAILABLE"


def test_malformed_canonical_lock_package_authority_is_detected(monkeypatch):
    malformed = checker._git_blob_bytes(REPO_ROOT, checker.REVIEWED_LOCK_GIT_BLOB_SHA1)
    assert malformed is not None
    malformed += b"not-a-pinned-package\n"
    real_git_blob_bytes = checker._git_blob_bytes

    def fake_git_blob_bytes(repo_root, git_ref):
        if git_ref == checker.REVIEWED_LOCK_GIT_BLOB_SHA1:
            return malformed
        return real_git_blob_bytes(repo_root, git_ref)

    monkeypatch.setattr(checker, "_git_blob_bytes", fake_git_blob_bytes)
    monkeypatch.setattr(checker, "REVIEWED_LOCK_SHA256", hashlib.sha256(malformed).hexdigest())
    result = checker.check_environment_lock(dict(checker.check_interpreter_identity()))
    assert result["status"] == "FAIL"
    assert result["reason"] == "LOCK_GIT_CANONICAL_PACKAGE_AUTHORITY_INVALID"


def test_wrong_e11_evidence_sha256_constant_is_rejected(monkeypatch):
    monkeypatch.setattr(checker, "V9_014_E11_LIVE_EVIDENCE_SHA256", "0" * 64)
    result = checker.check_e11_live_evidence_binding()
    assert result["status"] == "FAIL"
    assert result["reason"] == "E11_LIVE_EVIDENCE_SHA256_MISMATCH"


def test_wrong_e11_evidence_git_blob_constant_is_rejected(monkeypatch):
    monkeypatch.setattr(checker, "V9_014_E11_LIVE_EVIDENCE_GIT_BLOB_SHA1", "0" * 40)
    result = checker.check_e11_live_evidence_binding()
    assert result["status"] == "FAIL"
    assert result["reason"] == "E11_LIVE_EVIDENCE_GIT_PROVENANCE_UNAVAILABLE"


def test_wrong_e11_package_set_sha256_is_rejected():
    tampered = dict(_real_e11_evidence())
    tampered["observed_environment"] = dict(tampered["observed_environment"])
    tampered["observed_environment"]["package_set"] = dict(tampered["observed_environment"]["package_set"])
    tampered["observed_environment"]["package_set"]["sha256"] = "0" * 64
    result = checker.check_e11_live_evidence_binding(evidence=tampered)
    assert result["status"] == "FAIL"
    assert "package_set_sha256" in result["detail"]["failing_checks"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fixture_sha256", "0" * 64),
        ("status", "FAIL"),
    ],
)
def test_wrong_e11_xls_fixture_hash_or_status_is_rejected(field, value):
    tampered = dict(_real_e11_evidence())
    tampered["synthetic_probes"] = dict(tampered["synthetic_probes"])
    tampered["synthetic_probes"]["xls"] = dict(tampered["synthetic_probes"]["xls"])
    tampered["synthetic_probes"]["xls"][field] = value
    result = checker.check_e11_live_evidence_binding(evidence=tampered)
    assert result["status"] == "FAIL"
    expected_check = "xls_probe_fixture_sha256" if field == "fixture_sha256" else "xls_probe_status"
    assert expected_check in result["detail"]["failing_checks"]


@pytest.mark.parametrize(
    ("field", "value", "expected_check"),
    [
        ("fixture_sha256", "0" * 64, "pdf_probe_fixture_sha256"),
        ("pdfplumber_version", "0.11.9", "pdf_probe_pdfplumber_version"),
        ("status", "SYNTHETIC_PDF_PROBE_FIXTURE_HASH_MISMATCH_FAILURE", "pdf_probe_status"),
        ("page_count", 2, "pdf_probe_page_count"),
    ],
)
def test_wrong_e11_pdf_fixture_version_status_or_page_count_is_rejected(field, value, expected_check):
    tampered = dict(_real_e11_evidence())
    tampered["synthetic_probes"] = dict(tampered["synthetic_probes"])
    tampered["synthetic_probes"]["pdf"] = dict(tampered["synthetic_probes"]["pdf"])
    tampered["synthetic_probes"]["pdf"][field] = value
    result = checker.check_e11_live_evidence_binding(evidence=tampered)
    assert result["status"] == "FAIL"
    assert expected_check in result["detail"]["failing_checks"]


def test_wrong_e7_binding_in_e11_evidence_is_rejected():
    tampered = dict(_real_e11_evidence())
    tampered["reviewed_successor"] = dict(tampered["reviewed_successor"])
    tampered["reviewed_successor"]["lock_candidate"] = dict(tampered["reviewed_successor"]["lock_candidate"])
    tampered["reviewed_successor"]["lock_candidate"]["sha256"] = "0" * 64
    result = checker.check_e11_live_evidence_binding(evidence=tampered)
    assert result["status"] == "FAIL"
    assert "e7_lock_candidate_sha256" in result["detail"]["failing_checks"]


def test_wrong_e7_binding_in_v9_014_e7_bundle_is_rejected(monkeypatch):
    monkeypatch.setattr(checker, "V9_014_E7_LOCK_CANDIDATE_SHA256", "0" * 64)
    result = checker.check_v9_014_e7_bundle()
    assert result["status"] == "FAIL"
    assert result["reason"] == "V9_014_E7_LOCK_CANDIDATE_HASH_MISMATCH"


def test_wrong_generic_candidate_blob_in_freeze_record_is_rejected(monkeypatch):
    """E13 MEDIUM_1: a tampered/stale generic-candidate blob binding
    (whether in the REVIEWED_* constant or the freeze record's own claimed
    field) must fail check_freeze_record's independent blob re-derivation,
    not merely be trusted."""
    monkeypatch.setattr(checker, "REVIEWED_GENERIC_LOCK_CANDIDATE_GIT_BLOB_SHA1", "0" * 40)
    _lock, freeze = _run_lock_and_freeze_checks()
    assert freeze["status"] == "FAIL"
    assert freeze["reason"] == "FREEZE_RECORD_CROSS_CHECK_MISMATCH"


def test_wrong_generic_candidate_hash_in_freeze_record_is_rejected(monkeypatch):
    monkeypatch.setattr(checker, "REVIEWED_GENERIC_LOCK_CANDIDATE_CANONICAL_GIT_SHA256", "0" * 64)
    _lock, freeze = _run_lock_and_freeze_checks()
    assert freeze["status"] == "FAIL"
    assert freeze["reason"] == "FREEZE_RECORD_CROSS_CHECK_MISMATCH"


def test_xls_fixture_identity_tamper_fails_probe_with_exact_reason(tmp_path, monkeypatch):
    """Real perturbation: point the probe at a fixture with the WRONG bytes
    and require the specific FIXTURE_SHA256_MISMATCH FAIL -- not a bare
    `status in ("PASS", "FAIL")` non-assertion."""
    wrong_fixture = tmp_path / "wrong_synthetic_jpx_source_snapshot.xls"
    wrong_fixture.write_bytes(b"not the reviewed synthetic fixture bytes")
    monkeypatch.setattr(checker, "SYNTHETIC_XLS_FIXTURE_PATH", wrong_fixture)
    result = checker.check_jpx_xls_parser_synthetic_probe()
    assert result["status"] == "FAIL"
    assert result["reason"] == "FIXTURE_SHA256_MISMATCH"


def test_pdf_fixture_identity_tamper_fails_environment_lock_check(tmp_path, monkeypatch):
    # PDF_FIXTURE_SHA256_MISMATCH is raised before the interpreter/platform
    # checks are reached (see check_environment_lock's documented ordering),
    # so no interpreter/platform simulation is needed for this negative.
    wrong_fixture = tmp_path / "wrong_v9_014_synthetic_pdf_env_probe.pdf"
    wrong_fixture.write_bytes(b"not the reviewed synthetic pdf fixture bytes")
    monkeypatch.setattr(checker, "SYNTHETIC_PDF_FIXTURE_PATH", wrong_fixture)
    fake_interpreter = dict(checker.check_interpreter_identity())
    result = checker.check_environment_lock(fake_interpreter)
    assert result["status"] == "FAIL"
    assert result["reason"] == "PDF_FIXTURE_SHA256_MISMATCH"


def test_wrong_pdf_probe_result_rejected_by_v9_014_successor_promotion_validator():
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


def test_e11_evidence_promotion_true_before_e15_is_rejected():
    tampered = dict(_real_e11_evidence())
    tampered["successor_promoted"] = True
    result = checker.check_e11_live_evidence_binding(evidence=tampered)
    assert result["status"] == "FAIL"
    assert "successor_promoted_false" in result["detail"]["failing_checks"]


def test_e11_evidence_canonical_state_frozen_before_e15_is_rejected():
    tampered = dict(_real_e11_evidence())
    tampered["canonical_environment_state"] = "SUCCESSOR_CANONICAL_FROZEN"
    result = checker.check_e11_live_evidence_binding(evidence=tampered)
    assert result["status"] == "FAIL"
    assert "canonical_environment_state" in result["detail"]["failing_checks"]


def test_e11_evidence_future_protected_execution_authorized_true_is_rejected():
    tampered = dict(_real_e11_evidence())
    tampered["future_protected_execution_authorized"] = True
    result = checker.check_e11_live_evidence_binding(evidence=tampered)
    assert result["status"] == "FAIL"
    assert "future_protected_execution_authorized_false" in result["detail"]["failing_checks"]


def test_freeze_record_promotion_true_is_rejected_by_semantic_equality():
    tampered = _load_json(checker.FREEZE_RECORD_PATH)
    tampered["v9_014_successor_provenance"] = dict(tampered["v9_014_successor_provenance"])
    tampered["v9_014_successor_provenance"]["v9_014_pdf_environment_successor_promoted"] = True
    assert not checker._type_strict_semantic_equal(tampered, checker.REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT)


def test_freeze_record_future_protected_execution_authorized_true_is_rejected_by_semantic_equality():
    tampered = _load_json(checker.FREEZE_RECORD_PATH)
    tampered["future_protected_execution_authorized"] = True
    assert not checker._type_strict_semantic_equal(tampered, checker.REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT)


def test_promotion_true_and_protected_authority_true_never_appear_literally_in_any_artifact():
    for path in (
        checker.LOCK_CANDIDATE_PATH,
        checker.WINDOWS_VALIDATION_EVIDENCE_PATH,
        checker.FREEZE_RECORD_PATH,
        V9_014_FREEZE_RECORD_PATH,
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
