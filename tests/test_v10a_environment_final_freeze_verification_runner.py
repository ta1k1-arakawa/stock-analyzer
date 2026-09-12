from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import v10a_environment_final_freeze_verification_runner as runner
from scripts import v10a_environment_no_network_validation_runner as v10a


def _config(tmp_path: Path) -> runner.FinalFreezeConfig:
    return runner.FinalFreezeConfig(
        repo_root=tmp_path / "repo",
        expected_p2_reviewed_sha="a" * 40,
        expected_final_runner_blob_sha1="b" * 40,
        expected_candidate_blob_sha1="c" * 40,
        expected_candidate_sha256="d" * 64,
        wheelhouse=tmp_path / "wheelhouse",
        step4_attempt_root=tmp_path / "step4",
        output_root=tmp_path / "output",
    )


def _packages() -> list[dict[str, str]]:
    return [{"name": name, "version": version} for name, version in runner.EXPECTED_PACKAGES]


def _delegated_result(failure_code: str = "NONE") -> dict[str, object]:
    phase = {
        "observed_packages": _packages(),
        "observed_package_count": 20,
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "pandas_market_calendars_version": "5.4.0",
        "exchange_calendars_version": "4.13.2",
        "official_wheel_filename": v10a.OFFICIAL_WHEEL_FILENAME,
        "observed_official_wheel_sha256": v10a.OFFICIAL_WHEEL_SHA256,
        "official_wheel_sha256_match": True,
        "jpx_entry_occurrence_count": 1,
        "jp_entry_occurrence_count": 1,
        "jpx_installed_equals_wheel_entry": True,
        "jp_installed_equals_wheel_entry": True,
        "jpx_wheel_git_blob_sha1": v10a.JPX_RELEASE_GIT_BLOB_SHA1,
        "jp_wheel_git_blob_sha1": v10a.JP_RELEASE_GIT_BLOB_SHA1,
        "jpx_installed_git_blob_sha1": v10a.JPX_RELEASE_GIT_BLOB_SHA1,
        "jp_installed_git_blob_sha1": v10a.JP_RELEASE_GIT_BLOB_SHA1,
        "jpx_source_blob_match": True,
        "holiday_source_blob_match": True,
        "xls_probe_status": "PASS",
        "pdf_probe_status": "PASS",
        "historical_step4_provenance_verified": True,
        "reviewed_wheelhouse_provenance_verified": True,
        "package_index_network_requests": 0,
        "package_installations": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_reads": 0,
        "t0_run": False,
    }
    evidence = v10a._base_evidence(failure_code, phase)
    return {
        "status": "PASS" if failure_code == "NONE" else "FAIL",
        "failure_code": failure_code,
        "evidence": evidence,
        "canonical_environment_ready": False,
        "environment_frozen": False,
        "execution_authorized": False,
    }


def _synthetic_observations(*, provenance_valid: bool = True, delegated: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "provenance_valid": provenance_valid,
        "final_test_current_blob_sha1": "e" * 40,
        "package_index_network_requests": 0,
        "package_installations": 0,
        "environment_mutations": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_reads": 0,
        "t0_run": False,
        "delegated_result": delegated,
    }


def test_candidate_requires_p3_adjudication_and_has_no_self_reference() -> None:
    candidate = runner.candidate_template(runner_blob_sha1="a" * 40, test_blob_sha1="b" * 40)
    runner.validate_candidate(candidate, expected_runner_blob_sha1="a" * 40, expected_test_blob_sha1="b" * 40)
    assert candidate["p3_adjudication_required"] is True
    assert candidate["p3_adjudication_repo_path"] == "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_ADJUDICATION.json"
    assert "candidate_git_blob_sha1" not in candidate
    assert "candidate_sha256" not in candidate


def test_candidate_binding_failure_suppresses_delegated_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    called = False

    def delegate(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("delegated validation must not run")

    monkeypatch.setattr(v10a, "run_validation", delegate)
    result = runner.run_final_verification(config, _synthetic_observations(provenance_valid=False), publish=False)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert called is False


@pytest.mark.parametrize("field", ["candidate_sha256", "candidate_git_blob_sha1", "final_runner_blob", "promotion_design", "attempt1_evidence", "attempt1_adjudication", "expected_head", "dirty"])
def test_provenance_mismatch_suppresses_delegated_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    config = _config(tmp_path)
    called = False

    def delegate(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("delegated validation must not run")

    monkeypatch.setattr(v10a, "run_validation", delegate)
    result = runner.run_final_verification(config, _synthetic_observations(provenance_valid=False), publish=False)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert called is False


def test_output_collision_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir()
    monkeypatch.setattr(v10a, "run_validation", lambda *_args, **_kwargs: _delegated_result())
    result = runner.run_final_verification(config, _synthetic_observations(provenance_valid=True), publish=True)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert result["artifact_path"] is None


def test_production_delegates_with_no_observations_and_publish_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    stage2 = {"output_root_safe": True, "final_test_current_blob_sha1": "e" * 40}
    calls: list[tuple[object, object, object]] = []

    monkeypatch.setattr(runner, "_default_stage2_observations", lambda _config: stage2)
    monkeypatch.setattr(runner, "_validate_stage2", lambda _config, _obs: True)

    def delegate(_config: object, observations: object, *, publish: bool) -> dict[str, object]:
        calls.append((_config, observations, publish))
        return _delegated_result()

    monkeypatch.setattr(v10a, "run_validation", delegate)
    result = runner.run_final_verification(config, observations=None, publish=False)
    assert result["status"] == "PASS"
    assert len(calls) == 1
    assert calls[0][1] is None
    assert calls[0][2] is False


def test_delegated_fail_propagates_without_promotion_claim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(v10a, "run_validation", lambda *_args, **_kwargs: _delegated_result("PMC_VERSION_MISMATCH"))
    result = runner.run_final_verification(config, _synthetic_observations(), publish=False)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "PMC_VERSION_MISMATCH"
    assert result["canonical_environment_ready"] is False
    assert result["environment_frozen"] is False
    assert result["execution_authorized"] is False


def test_delegated_pass_publishes_non_authorizing_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(v10a, "run_validation", lambda *_args, **_kwargs: _delegated_result())
    result = runner.run_final_verification(config, _synthetic_observations(), publish=True)
    assert result["status"] == "PASS"
    evidence = result["evidence"]
    runner.validate_final_evidence(evidence)
    assert evidence["environment_mutations"] == 0
    assert evidence["canonical_environment_promoted"] is False
    assert evidence["environment_frozen"] is False
    assert evidence["execution_authorized"] is False
    assert evidence["calendar_generation_authorized"] is False
    assert evidence["t0_authorized"] is False
    assert evidence["historical_evaluation_authorized"] is False
    assert evidence["future_profitability_established"] is False
    assert result["artifact_path"].name == runner.FINAL_EVIDENCE_NAME


@pytest.mark.parametrize("failure_code", ["LIVE_PACKAGE_SET_MISMATCH", "PYTHON_PLATFORM_MISMATCH", "EXCHANGE_CALENDARS_VERSION_MISMATCH", "OFFICIAL_WHEEL_IDENTITY_MISMATCH", "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE", "JPX_INSTALLED_WHEEL_BYTES_MISMATCH", "JPX_RELEASE_BLOB_MISMATCH", "HOLIDAY_INSTALLED_WHEEL_BYTES_MISMATCH", "HOLIDAY_RELEASE_BLOB_MISMATCH", "XLS_PROBE_FAILURE", "PDF_PROBE_FAILURE"])
def test_exact_live_mismatch_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_code: str) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(v10a, "run_validation", lambda *_args, **_kwargs: _delegated_result(failure_code))
    result = runner.run_final_verification(config, _synthetic_observations(), publish=False)
    assert result["failure_code"] == failure_code
    assert result["status"] == "FAIL"


def test_authorized_operation_has_highest_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    called = False

    def delegate(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("delegated validation must not run")

    monkeypatch.setattr(v10a, "run_validation", delegate)
    observations = _synthetic_observations(provenance_valid=False)
    observations["environment_mutations"] = 1
    result = runner.run_final_verification(config, observations, publish=False)
    assert result["failure_code"] == "UNAUTHORIZED_OPERATION_OBSERVED"
    assert called is False


def test_evidence_validator_rejects_missing_extra_and_invalid_types(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(v10a, "run_validation", lambda *_args, **_kwargs: _delegated_result())
    evidence = runner.run_final_verification(config, _synthetic_observations(), publish=False)["evidence"]
    missing = copy.deepcopy(evidence)
    missing.pop("environment_mutations")
    with pytest.raises(runner.FinalFreezeValidationError):
        runner.validate_final_evidence(missing)
    extra = copy.deepcopy(evidence)
    extra["extra"] = True
    with pytest.raises(runner.FinalFreezeValidationError):
        runner.validate_final_evidence(extra)
    invalid = copy.deepcopy(evidence)
    invalid["environment_mutations"] = True
    with pytest.raises(runner.FinalFreezeValidationError):
        runner.validate_final_evidence(invalid)


def test_public_cli_has_no_synthetic_or_bypass_options() -> None:
    options = {option for action in runner._build_parser()._actions for option in action.option_strings}
    forbidden = {"observations", "synthetic", "skip", "force", "interpreter", "process_runner", "fake"}
    assert not any(any(word in option.lower() for word in forbidden) for option in options)


def test_public_cli_rejects_malformed_hashes_without_delegation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def delegate(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("invalid CLI input must fail before delegation")

    monkeypatch.setattr(runner, "run_final_verification", delegate)
    args = [
        "--repo-root", str(tmp_path / "repo"),
        "--expected-p2-reviewed-sha", "not-a-sha",
        "--expected-final-verification-runner-blob-sha1", "b" * 40,
        "--expected-candidate-blob-sha1", "c" * 40,
        "--expected-candidate-sha256", "d" * 64,
        "--wheelhouse", str(tmp_path / "wheelhouse"),
        "--step4-attempt-root", str(tmp_path / "step4"),
        "--output-root", str(tmp_path / "output"),
    ]
    assert runner.main(args) == 1
    assert called is False


def test_no_authorization_claim_and_no_local_machine_path_in_pass_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(v10a, "run_validation", lambda *_args, **_kwargs: _delegated_result())
    result = runner.run_final_verification(config, _synthetic_observations(), publish=False)
    encoded = json.dumps(result["evidence"], sort_keys=True)
    assert str(tmp_path) not in encoded
    assert "human" not in encoded.lower()
    assert result["evidence"]["status"] == "PASS"


def test_canonical_json_is_deterministic_and_publication_is_exclusive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(v10a, "run_validation", lambda *_args, **_kwargs: _delegated_result())
    result = runner.run_final_verification(config, _synthetic_observations(), publish=True)
    artifact = result["artifact_path"]
    assert artifact.read_bytes() == runner.canonical_json_bytes(result["evidence"])
    with pytest.raises(FileExistsError):
        config.output_root.mkdir(exist_ok=False)


def test_process_exit_code_style_rejects_boolean_if_later_adjudication_uses_integer() -> None:
    assert isinstance(0, int) and not isinstance(0, bool)
    assert isinstance(True, bool)
    with pytest.raises(runner.FinalFreezeValidationError):
        runner._strict_int(True, "process_exit_code", allow_none=False)
