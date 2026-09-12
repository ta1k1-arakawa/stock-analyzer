from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import v10_environment_successor_live_validation_runner as runner
from scripts import v10_environment_mutation_preflight_runner as preflight
from scripts import check_real_execution_env
from scripts.v10_environment_extension_contract import PREDECESSOR_PACKAGE_SET


REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_COMMIT = "a" * 40
LIVE_BLOB = "b" * 40
STEP5_CURRENT_HEAD = "c" * 40
STEP4_EXECUTION_HEAD = runner.STEP4_EXECUTION_HEAD_SHA


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_pdf_probe_uses_reviewed_public_status_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_real_execution_env, "check_pdf_parser_synthetic_probe", lambda: {"status": "PASS"})
    public_success = check_real_execution_env.check_pdf_parser_synthetic_probe()

    assert public_success["status"] == "PASS"
    assert runner._normalize_synthetic_probe_status(public_success) == "PASS"
    assert runner._normalize_synthetic_probe_status({"status": "FAIL"}) == "FAIL"
    assert runner._normalize_synthetic_probe_status({"status": "SYNTHETIC_PDF_PROBE_PASS"}) == "FAIL"


def _successor_packages() -> list[dict[str, str]]:
    packages = [{"name": name, "version": version} for name, version in PREDECESSOR_PACKAGE_SET]
    packages.extend({"name": name, "version": version} for name, version in runner.EXPECTED_DELTA)
    return packages


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[runner.LiveValidationConfig, dict[str, object], dict[str, object]]:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    attempt = tmp_path / "step4-attempt"
    attempt.mkdir()
    output = tmp_path / "live-output"
    candidate = json.loads((REPO_ROOT / runner.CANDIDATE_RELATIVE).read_text(encoding="utf-8"))
    evidence = (REPO_ROOT / runner.EVIDENCE_RELATIVE).read_bytes()
    authority = (REPO_ROOT / runner.MIGRATION_AUTHORITY_RELATIVE).read_bytes()
    lock = (REPO_ROOT / runner.LOCK_RELATIVE).read_bytes()
    candidate_bytes = (REPO_ROOT / runner.CANDIDATE_RELATIVE).read_bytes()
    wheel_paths = {item["filename"]: str(wheelhouse / item["filename"]) for item in candidate["resolved_wheels"]}
    delta_paths = tuple(Path(wheel_paths[item["filename"]]) for item in candidate["resolved_wheels"] if (item["name"], item["version"]) in runner.EXPECTED_DELTA)
    stdout = b"synthetic step4 stdout"
    stderr = b"synthetic step4 stderr"
    monkeypatch.setattr(runner, "STEP4_STDOUT_SHA256", _hash(stdout))
    monkeypatch.setattr(runner, "STEP4_STDERR_SHA256", _hash(stderr))
    state = {
        "schema_version": "V10_CANONICAL_ENVIRONMENT_EXACT_DELTA_MUTATION_ATTEMPT_V1",
        "expected_current_head": STEP4_EXECUTION_HEAD,
        "mutation_runner_blob_sha1": runner.STEP4_MUTATION_RUNNER_BLOB_SHA1,
        "step3_receipt_sha256": runner.STEP3_RECEIPT_SHA256,
        "reviewed_successor_lock_candidate_sha256": runner.CANDIDATE_SHA256,
        "wheel_manifest": [
            {**item, "path": wheel_paths[item["filename"]], "is_delta": (item["name"], item["version"]) in set(runner.EXPECTED_DELTA)}
            for item in candidate["resolved_wheels"]
        ],
        "delta_wheel_paths": [str(path) for path in delta_paths],
        "process_start_attempted": True,
        "process_started": True,
        "mutation_authority_consumed": True,
        "mutation_started": True,
        "retry_authorized": False,
        "process_exit_code": 0,
        "stdout_capture_path": str(attempt / "stdout.bin"),
        "stdout_sha256": _hash(stdout),
        "stderr_capture_path": str(attempt / "stderr.bin"),
        "stderr_sha256": _hash(stderr),
        "failure_code": "NONE",
    }
    config = runner.LiveValidationConfig(
        repo_root=REPO_ROOT,
        expected_current_head=STEP5_CURRENT_HEAD,
        expected_live_validation_runner_commit_sha=LIVE_COMMIT,
        expected_live_validation_runner_blob_sha1=LIVE_BLOB,
        step4_attempt_root=attempt,
        output_root=output,
        wheelhouse=wheelhouse,
    )
    candidate_sha = _hash(candidate_bytes)
    observations: dict[str, object] = {
        "repository_identity": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "branch": runner.AUTHORITATIVE_BRANCH,
        "head": config.expected_current_head,
        "clean": True,
        "frozen_design_commit_exists": True,
        "extension_design_commit_exists": True,
        "generic_authority_transition_reviewed_sha": runner.GENERIC_AUTHORITY_TRANSITION_SHA,
        "generic_authority_transition_commit_exists": True,
        "reviewed_candidate_commit_exists": True,
        "reviewed_evidence_commit_exists": True,
        "reviewed_runner_commit_exists": True,
        "reviewed_runner_blob_sha1": runner.PREMUTATION_RUNNER_BLOB_SHA1,
        "current_runner_blob_sha1": runner.PREMUTATION_RUNNER_BLOB_SHA1,
        "candidate_git_blob_sha1": runner.git_blob_sha1(candidate_bytes),
        "evidence_git_blob_sha1": runner.EVIDENCE_BLOB_SHA1,
        "current_head_candidate_git_blob_sha1": runner.CANDIDATE_BLOB_SHA1,
        "current_head_evidence_git_blob_sha1": runner.EVIDENCE_BLOB_SHA1,
        "migration_authority_git_blob_sha1": runner.MIGRATION_AUTHORITY_BLOB_SHA1,
        "candidate_bytes": candidate_bytes,
        "evidence_bytes": evidence,
        "migration_authority_bytes": authority,
        "candidate_bytes_git_blob_sha1": runner.git_blob_sha1(candidate_bytes),
        "evidence_bytes_git_blob_sha1": runner.git_blob_sha1(evidence),
        "migration_authority_bytes_git_blob_sha1": runner.git_blob_sha1(authority),
        "generic_lock_git_blob_sha1": runner.GENERIC_LOCK_BLOB_SHA1,
        "generic_lock_committed_bytes": lock,
        "generic_lock_worktree_bytes": lock,
        "live_validation_runner_commit_exists": True,
        "reviewed_live_validation_runner_blob_sha1": LIVE_BLOB,
        "current_live_validation_runner_blob_sha1": LIVE_BLOB,
        "step4_state": state,
        "stdout.bin": stdout,
        "stderr.bin": stderr,
        "observed_packages": _successor_packages(),
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "pandas_market_calendars_version": "5.4.0",
        "exchange_calendars_version": "4.13.2",
        "jpx_source_blob_sha1": runner.EXPECTED_PMC_SOURCE_BLOB_SHA1,
        "holiday_source_blob_sha1": runner.EXPECTED_HOLIDAY_SOURCE_BLOB_SHA1,
        "xls_probe_status": "PASS",
        "pdf_probe_status": "PASS",
        "package_index_network_requests": 0,
        "package_installations": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_reads": 0,
        "t0_run": False,
    }
    assert candidate_sha == runner.CANDIDATE_SHA256
    return config, observations, candidate


def _patch_wheel_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    def verified(wheelhouse: object, resolved_wheels: object, *_args: object, **_kwargs: object) -> dict[str, object]:
        root = Path(wheelhouse)
        paths = tuple(
            root / item["filename"]
            for item in resolved_wheels
            if (item["name"], item["version"]) in set(runner.EXPECTED_DELTA)
        )
        return {
            "ok": True,
            "failure_code": "NONE",
            "wheelhouse_integrity_verified": True,
            "delta_wheel_count": 5,
            "delta_packages": runner.EXPECTED_DELTA,
            "delta_wheel_paths": paths,
        }

    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", verified)


def test_exact_pass_schema_and_non_claims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    # The injected wheel paths must agree with the durable Step-4 state.
    paths = tuple(Path(path) for path in observations["step4_state"]["delta_wheel_paths"])
    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", lambda *_a, **_k: {
        "ok": True, "failure_code": "NONE", "delta_packages": runner.EXPECTED_DELTA,
        "delta_wheel_count": 5, "delta_wheel_paths": paths,
    })
    result = runner.run_live_validation(config, observations, publish=True)
    assert result["status"] == "PASS"
    assert result["failure_code"] == "NONE"
    evidence = result["evidence"]
    runner.validate_live_validation_evidence(evidence)
    assert set(evidence) == runner.EVIDENCE_KEYS
    assert evidence["installed_delta_package_count"] == 5
    assert evidence["installed_delta_wheel_count"] == 5
    assert evidence["observed_package_count"] == 20
    assert result["artifact_path"].read_bytes() == runner.canonical_json_bytes(evidence)
    assert result["canonical_environment_ready"] is False
    assert result["environment_frozen"] is False
    assert result["execution_authorized"] is False


def test_production_path_collects_live_observation_only_after_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    paths = tuple(Path(path) for path in observations["step4_state"]["delta_wheel_paths"])
    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", lambda *_a, **_k: {"ok": True, "delta_packages": runner.EXPECTED_DELTA, "delta_wheel_count": 5, "delta_wheel_paths": paths})
    provenance_calls = 0
    live_calls = 0
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _c: observations)

    def live(_c: runner.LiveValidationConfig) -> dict[str, object]:
        nonlocal live_calls
        live_calls += 1
        result = {key: observations[key] for key in ("observed_packages", "python_version", "platform_system", "platform_machine", "sysconfig_platform", "pandas_market_calendars_version", "exchange_calendars_version", "jpx_source_blob_sha1", "holiday_source_blob_sha1", "xls_probe_status", "pdf_probe_status")}
        result["interpreter_executable"] = str(config.canonical_interpreter.resolve())
        return result

    original = runner._validate_provenance

    def provenance(c: runner.LiveValidationConfig, o: object) -> object:
        nonlocal provenance_calls
        provenance_calls += 1
        return original(c, o)

    monkeypatch.setattr(runner, "_validate_provenance", provenance)
    monkeypatch.setattr(runner, "_default_live_observations", live)
    result = runner.run_live_validation(config, observations=None, publish=False)
    assert result["status"] == "PASS"
    assert provenance_calls == 1
    assert live_calls == 1


@pytest.mark.parametrize("field", ["head", "clean", "live_validation_runner_commit_exists", "current_live_validation_runner_blob_sha1", "migration_authority_git_blob_sha1"])
def test_provenance_failures_do_not_observe_live_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    bad = copy.deepcopy(observations)
    bad[field] = False if field in {"clean", "live_validation_runner_commit_exists"} else "0" * 40
    called = 0

    def live(_c: runner.LiveValidationConfig) -> dict[str, object]:
        nonlocal called
        called += 1
        raise AssertionError("live observation must not run")

    monkeypatch.setattr(runner, "_default_live_observations", live)
    result = runner.run_live_validation(config, bad, publish=False)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert called == 0


@pytest.mark.parametrize("field", ["expected_current_head", "expected_live_validation_runner_blob_sha1"])
def test_step4_provenance_binding_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    if field == "expected_current_head":
        observations["step4_state"][field] = "0" * 40
    else:
        observations["step4_state"]["mutation_runner_blob_sha1"] = "0" * 40
    result = runner.run_live_validation(config, observations, publish=False)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"


def test_step4_execution_head_is_historical_and_distinct_from_step5_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)

    assert config.expected_current_head == STEP5_CURRENT_HEAD
    assert observations["step4_state"]["expected_current_head"] == STEP4_EXECUTION_HEAD
    assert STEP5_CURRENT_HEAD != STEP4_EXECUTION_HEAD

    result = runner.run_live_validation(config, observations, publish=False)
    assert result["status"] == "PASS"


@pytest.mark.parametrize("bound_head", [STEP5_CURRENT_HEAD, "0" * 40])
def test_wrong_step4_execution_head_fails_before_live_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound_head: str
) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    observations["step4_state"]["expected_current_head"] = bound_head
    called = False

    def live(_config: runner.LiveValidationConfig) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("live observation must not run")

    monkeypatch.setattr(runner, "_default_live_observations", live)
    result = runner.run_live_validation(config, observations, publish=False)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert called is False


def test_step5_current_head_remains_independently_required(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    observations["head"] = STEP4_EXECUTION_HEAD
    called = False

    def live(_config: runner.LiveValidationConfig) -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("live observation must not run")

    monkeypatch.setattr(runner, "_default_live_observations", live)
    result = runner.run_live_validation(config, observations, publish=False)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert called is False


@pytest.mark.parametrize("field", ["candidate_sha256", "delta_wheel_count", "step3_receipt_sha256", "process_exit_code", "retry_authorized", "stdout_sha256"])
def test_invalid_step4_evidence_is_provenance_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    state = observations["step4_state"]
    if field == "candidate_sha256":
        state["reviewed_successor_lock_candidate_sha256"] = "0" * 64
    elif field == "delta_wheel_count":
        state["delta_wheel_paths"] = state["delta_wheel_paths"][:-1]
    elif field == "step3_receipt_sha256":
        state[field] = "0" * 64
    elif field == "process_exit_code":
        state[field] = 1
    elif field == "retry_authorized":
        state[field] = True
    else:
        state[field] = "0" * 64
    result = runner.run_live_validation(config, observations, publish=False)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"


@pytest.mark.parametrize("field", ["observed_packages", "python_version", "platform_system", "platform_machine", "sysconfig_platform", "pandas_market_calendars_version", "exchange_calendars_version", "jpx_source_blob_sha1", "holiday_source_blob_sha1", "xls_probe_status", "pdf_probe_status"])
def test_frozen_failure_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    paths = tuple(Path(path) for path in observations["step4_state"]["delta_wheel_paths"])
    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", lambda *_a, **_k: {"ok": True, "delta_packages": runner.EXPECTED_DELTA, "delta_wheel_count": 5, "delta_wheel_paths": paths})
    if field == "observed_packages":
        observations[field] = observations[field][:-1]
    elif field in {"xls_probe_status", "pdf_probe_status"}:
        observations[field] = "FAIL"
    elif field.endswith("source_blob_sha1"):
        observations[field] = "0" * 40
    else:
        observations[field] = "wrong"
    result = runner.run_live_validation(config, observations, publish=False)
    expected = {
        "observed_packages": "LIVE_PACKAGE_SET_MISMATCH",
        "python_version": "PYTHON_PLATFORM_MISMATCH",
        "platform_system": "PYTHON_PLATFORM_MISMATCH",
        "platform_machine": "PYTHON_PLATFORM_MISMATCH",
        "sysconfig_platform": "PYTHON_PLATFORM_MISMATCH",
        "pandas_market_calendars_version": "PMC_VERSION_MISMATCH",
        "exchange_calendars_version": "EXCHANGE_CALENDARS_VERSION_MISMATCH",
        "jpx_source_blob_sha1": "JPX_SOURCE_BLOB_MISMATCH",
        "holiday_source_blob_sha1": "HOLIDAY_SOURCE_BLOB_MISMATCH",
        "xls_probe_status": "XLS_PROBE_FAILURE",
        "pdf_probe_status": "PDF_PROBE_FAILURE",
    }
    assert result["failure_code"] == expected[field]


def test_unauthorized_precedes_provenance_and_live_observation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    observations["package_installations"] = 1
    observations["head"] = "0" * 40
    called = False

    def gate(*_a: object, **_k: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("provenance must not run")

    monkeypatch.setattr(runner, "_validate_provenance", gate)
    result = runner.run_live_validation(config, observations, publish=False)
    assert result["failure_code"] == "UNAUTHORIZED_OPERATION_OBSERVED"
    assert called is False


@pytest.mark.parametrize("mutation", ["missing", "extra", "collision", "malformed"])
def test_package_metadata_failures_are_live_set_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    paths = tuple(Path(path) for path in observations["step4_state"]["delta_wheel_paths"])
    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", lambda *_a, **_k: {"ok": True, "delta_packages": runner.EXPECTED_DELTA, "delta_wheel_count": 5, "delta_wheel_paths": paths})
    if mutation == "missing":
        observations["observed_packages"] = observations["observed_packages"][:-1]
    elif mutation == "extra":
        observations["observed_packages"].append({"name": "extra", "version": "1"})
    elif mutation == "collision":
        observations["observed_packages"][1]["name"] = observations["observed_packages"][0]["name"].replace("-", "_")
    else:
        observations["observed_packages"] = [{"name": "", "version": "1"}]
    result = runner.run_live_validation(config, observations, publish=False)
    assert result["failure_code"] == "LIVE_PACKAGE_SET_MISMATCH"
    if mutation in {"missing", "extra"}:
        assert result["evidence"]["observed_package_count"] in {19, 21}
    else:
        assert result["evidence"]["observed_package_count"] is None


@pytest.mark.parametrize("field", ["jpx_source_blob_sha1", "holiday_source_blob_sha1"])
def test_source_blob_hashes_use_git_blob_algorithm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    paths = tuple(Path(path) for path in observations["step4_state"]["delta_wheel_paths"])
    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", lambda *_a, **_k: {"ok": True, "delta_packages": runner.EXPECTED_DELTA, "delta_wheel_count": 5, "delta_wheel_paths": paths})
    observations[field] = "0" * 40
    result = runner.run_live_validation(config, observations, publish=False)
    assert result["failure_code"] == ("JPX_SOURCE_BLOB_MISMATCH" if field.startswith("jpx") else "HOLIDAY_SOURCE_BLOB_MISMATCH")


def test_no_overwrite_output_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    config.output_root.mkdir()
    with pytest.raises(runner.LiveValidationError, match="PROVENANCE_BINDING_FAILURE"):
        runner.run_live_validation(config, observations, publish=True)


def test_validator_rejects_boolean_integer_and_extra_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _ = _fixture(tmp_path, monkeypatch)
    _patch_wheel_gate(monkeypatch)
    paths = tuple(Path(path) for path in observations["step4_state"]["delta_wheel_paths"])
    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", lambda *_a, **_k: {"ok": True, "delta_packages": runner.EXPECTED_DELTA, "delta_wheel_count": 5, "delta_wheel_paths": paths})
    evidence = runner.run_live_validation(config, observations, publish=False)["evidence"]
    bad = dict(evidence)
    bad["generic_lock_package_count"] = True
    with pytest.raises(runner.LiveValidationError):
        runner.validate_live_validation_evidence(bad)
    bad = dict(evidence)
    bad["extra"] = True
    with pytest.raises(runner.LiveValidationError):
        runner.validate_live_validation_evidence(bad)


def test_cli_has_only_real_bindings_and_does_not_expose_injection(tmp_path: Path) -> None:
    options = {option for action in runner._build_parser()._actions for option in action.option_strings}
    assert "--observations" not in options
    assert "--fake-packages" not in options
    assert "--probe-result" not in options
    assert "--alternate-interpreter" not in options
    assert "--process-runner" not in options
    assert "--wheelhouse" in options


def test_cli_constructs_real_config_and_calls_production_observation_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    calls: dict[str, object] = {}

    def production(config: runner.LiveValidationConfig, observations: object = None, *, publish: bool = True) -> dict[str, object]:
        calls.update(config=config, observations=observations, publish=publish)
        return {"status": "PASS", "failure_code": "NONE", "canonical_environment_ready": False, "environment_frozen": False, "execution_authorized": False}

    monkeypatch.setattr(runner, "run_live_validation", production)
    args = [
        "--repo-root", str(REPO_ROOT),
        "--expected-current-head", "c" * 40,
        "--expected-live-validation-runner-commit-sha", LIVE_COMMIT,
        "--expected-live-validation-runner-blob-sha1", LIVE_BLOB,
        "--step4-attempt-root", str(tmp_path / "attempt"),
        "--output-root", str(tmp_path / "output"),
        "--wheelhouse", str(tmp_path / "wheelhouse"),
    ]
    assert runner.main(args) == 0
    assert isinstance(calls["config"], runner.LiveValidationConfig)
    assert calls["observations"] is None
    assert calls["publish"] is True
    assert json.loads(capsys.readouterr().out)["canonical_environment_ready"] is False
