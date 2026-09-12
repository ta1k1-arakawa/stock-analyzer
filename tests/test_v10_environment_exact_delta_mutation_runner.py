from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pytest

from scripts import v10_environment_exact_delta_mutation_runner as runner
from scripts import v10_environment_mutation_preflight_runner as preflight
from tests.test_v10_environment_mutation_preflight_runner import _fixture


MUTATION_COMMIT = "e" * 40
MUTATION_BLOB = "f" * 40


def _mutation_fixture(tmp_path: Path) -> tuple[runner.MutationConfig, dict[str, object]]:
    preflight_config, observations, _ = _fixture(tmp_path / "fixture")
    receipt = preflight._receipt(
        preflight_config,
        status="PASS",
        failure_code="NONE",
        integrity=True,
        delta_count=5,
    )
    receipt_bytes = preflight.canonical_json_bytes(receipt)
    observations = copy.deepcopy(observations)
    observations.update(
        mutation_runner_commit_exists=True,
        reviewed_mutation_runner_blob_sha1=MUTATION_BLOB,
        current_mutation_runner_blob_sha1=MUTATION_BLOB,
        step3_receipt_bytes=receipt_bytes,
    )
    config = runner.MutationConfig(
        preflight_config=preflight_config,
        expected_mutation_runner_commit_sha=MUTATION_COMMIT,
        expected_mutation_runner_blob_sha1=MUTATION_BLOB,
        expected_step3_receipt_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
        attempt_root=tmp_path / "attempt",
    )
    return config, observations


class FakeProcess:
    def __init__(self, returncode: object = 0, stdout: bytes = b"out", stderr: bytes = b"err") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.argv: list[str] | None = None
        self.calls = 0
        self.communicate_calls = 0

    def __call__(self, argv: list[str] | tuple[str, ...]) -> "FakeProcess":
        self.calls += 1
        self.argv = list(argv)
        return self

    def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        return self.stdout, self.stderr


def _complete_state(config: runner.MutationConfig) -> dict[str, object]:
    return json.loads((config.attempt_root / "attempt_state_complete.json").read_text(encoding="utf-8"))


def _cli_arguments(config: runner.MutationConfig) -> list[str]:
    return [
        "--repo-root", str(config.repo_root),
        "--expected-current-head", config.preflight_config.expected_current_head,
        "--expected-mutation-runner-commit-sha", config.expected_mutation_runner_commit_sha,
        "--expected-mutation-runner-blob-sha1", config.expected_mutation_runner_blob_sha1,
        "--wheelhouse", str(config.wheelhouse),
        "--step3-receipt", str(config.default_step3_receipt_path),
        "--attempt-root", str(config.attempt_root),
        "--fresh-mutation-authority-token", runner.FRESH_MUTATION_AUTHORITY_TOKEN,
    ]


def test_production_cli_constructs_real_config_and_closes_injection_seams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config, _ = _mutation_fixture(tmp_path)
    captured: dict[str, object] = {}

    def production_call(
        actual_config: runner.MutationConfig,
        observations: object = None,
        *,
        mutation_authority_token: str | None = None,
        process_runner: object = None,
    ) -> dict[str, object]:
        captured.update(
            config=actual_config,
            observations=observations,
            mutation_authority_token=mutation_authority_token,
            process_runner=process_runner,
        )
        return {
            "status": "FAIL",
            "failure_code": runner.MUTATION_AUTHORITY_MISSING,
            "mutation_authority_consumed": False,
            "mutation_started": False,
            "canonical_environment_ready": False,
            "environment_frozen": False,
        }

    monkeypatch.setattr(runner, "run_mutation", production_call)
    assert runner.main(_cli_arguments(config)) == 1
    actual = captured["config"]
    assert isinstance(actual, runner.MutationConfig)
    assert actual.preflight_config.expected_extension_design_sha == runner.EXPECTED_EXTENSION_DESIGN_SHA
    assert actual.preflight_config.expected_reviewed_runner_commit_sha == runner.EXPECTED_PREMUTATION_RUNNER_COMMIT_SHA
    assert actual.preflight_config.expected_reviewed_runner_blob_sha1 == runner.EXPECTED_PREMUTATION_RUNNER_BLOB_SHA1
    assert actual.wheelhouse == config.wheelhouse
    assert actual.attempt_root == config.attempt_root
    assert captured["observations"] is None
    assert captured["process_runner"] is None
    assert captured["mutation_authority_token"] == runner.FRESH_MUTATION_AUTHORITY_TOKEN
    output_raw = capsys.readouterr().out
    output = json.loads(output_raw)
    assert output["canonical_environment_ready"] is False
    assert output["environment_frozen"] is False
    assert "pip-out" not in output_raw


def test_cli_exposes_no_synthetic_or_alternate_authority_options() -> None:
    options = {
        option
        for action in runner._build_parser()._actions
        for option in action.option_strings
    }
    assert "--observations" not in options
    assert "--process-runner" not in options
    assert "--alternate-interpreter" not in options
    assert "--install-argv" not in options
    assert "--candidate" not in options
    assert "--evidence" not in options
    assert "--lock" not in options


def test_cli_rejects_invalid_authority_input() -> None:
    parser = runner._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--repo-root", str(Path.cwd()),
            "--expected-current-head", "0" * 40,
            "--expected-mutation-runner-commit-sha", MUTATION_COMMIT,
            "--expected-mutation-runner-blob-sha1", MUTATION_BLOB,
            "--wheelhouse", str(Path.cwd() / "synthetic-wheelhouse"),
            "--step3-receipt", str(Path.cwd() / "synthetic-receipt.json"),
            "--attempt-root", str(Path.cwd() / "synthetic-attempt"),
            "--fresh-mutation-authority-token", "NOT_A_FRESH_AUTHORITY",
        ])


def test_cli_wrong_head_fails_closed_before_canonical_probe(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "--repo-root", str(Path.cwd()),
        "--expected-current-head", "0" * 40,
        "--expected-mutation-runner-commit-sha", MUTATION_COMMIT,
        "--expected-mutation-runner-blob-sha1", MUTATION_BLOB,
        "--wheelhouse", str(Path.cwd() / "synthetic-wheelhouse"),
        "--step3-receipt", str(Path.cwd() / "synthetic-receipt.json"),
        "--attempt-root", str(Path.cwd() / "synthetic-attempt"),
        "--fresh-mutation-authority-token", runner.FRESH_MUTATION_AUTHORITY_TOKEN,
    ]
    assert runner.main(args) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["failure_code"] == runner.PROVENANCE_BINDING_FAILURE
    assert output["canonical_environment_ready"] is False
    assert output["environment_frozen"] is False


def test_exact_pass_rechecks_wheels_and_persists_exact_argv_and_state(tmp_path: Path) -> None:
    config, observations = _mutation_fixture(tmp_path)
    process = FakeProcess(returncode=0, stdout=b"pip-out", stderr=b"pip-err")
    result = runner.run_mutation(
        config,
        observations,
        mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN,
        process_runner=process,
    )
    assert result["status"] == "PASS"
    assert result["failure_code"] == runner.NONE
    assert process.calls == 1
    assert process.communicate_calls == 1
    assert result["canonical_environment_ready"] is False
    assert result["environment_frozen"] is False
    expected_prefix = [
        str(config.canonical_interpreter),
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
    ]
    assert process.argv[:6] == expected_prefix
    assert [Path(path).name for path in process.argv[6:]] == [
        "exchange_calendars-4.13.2-py3-none-any.whl",
        "korean_lunar_calendar-0.4.0-py3-none-any.whl",
        "pandas_market_calendars-5.4.0-py3-none-any.whl",
        "pyluach-2.3.0-py3-none-any.whl",
        "toolz-1.1.0-py3-none-any.whl",
    ]
    assert not any("--find-links" in item or "==" in item or "://" in item for item in process.argv)
    assert len(process.argv[6:]) == 5
    state = _complete_state(config)
    runner.validate_attempt_state(state)
    assert state["process_exit_code"] == 0
    assert state["mutation_authority_consumed"] is True
    assert state["mutation_started"] is True
    assert state["retry_authorized"] is False
    assert (config.attempt_root / "stdout.bin").read_bytes() == b"pip-out"
    assert (config.attempt_root / "stderr.bin").read_bytes() == b"pip-err"
    assert state["stdout_sha256"] == hashlib.sha256(b"pip-out").hexdigest()
    assert state["stderr_sha256"] == hashlib.sha256(b"pip-err").hexdigest()
    assert json.loads((config.attempt_root / "attempt_state_prelaunch.json").read_text())[
        "mutation_authority_consumed"
    ] is False
    prelaunch = json.loads((config.attempt_root / "attempt_state_prelaunch.json").read_text())
    assert prelaunch["process_start_attempted"] is False
    assert prelaunch["process_started"] is False
    attempted = json.loads((config.attempt_root / "attempt_state_attempted.json").read_text())
    assert attempted["process_start_attempted"] is True
    assert attempted["process_started"] is False
    assert attempted["mutation_authority_consumed"] is True
    assert attempted["mutation_started"] is True


@pytest.mark.parametrize("field, value", [
    ("status", "FAIL"),
    ("failure_code", "PROVENANCE_BINDING_FAILURE"),
    ("wheelhouse_integrity_verified", False),
    ("wheelhouse_integrity_verified", None),
    ("delta_wheel_count", 4),
    ("mutation_authority_consumed", True),
    ("mutation_started", True),
])
def test_invalid_step3_receipt_is_rejected_before_pip(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config, observations = _mutation_fixture(tmp_path)
    receipt = json.loads(observations["step3_receipt_bytes"].decode("utf-8"))
    receipt[field] = value
    raw = preflight.canonical_json_bytes(receipt)
    observations["step3_receipt_bytes"] = raw
    config = replace(config, expected_step3_receipt_sha256=hashlib.sha256(raw).hexdigest())
    process = FakeProcess()
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.PREMUTATION_RECEIPT_INVALID
    assert process.calls == 0


def test_wrong_step3_receipt_sha_is_rejected(tmp_path: Path) -> None:
    config, observations = _mutation_fixture(tmp_path)
    config = replace(config, expected_step3_receipt_sha256="0" * 64)
    process = FakeProcess()
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.PREMUTATION_RECEIPT_INVALID
    assert process.calls == 0


@pytest.mark.parametrize("field", [
    "head",
    "clean",
    "current_runner_blob_sha1",
    "reviewed_mutation_runner_blob_sha1",
    "current_mutation_runner_blob_sha1",
    "current_head_candidate_git_blob_sha1",
    "current_head_evidence_git_blob_sha1",
])
def test_provenance_failure_precedes_environment_process_and_wheelhouse(
    tmp_path: Path,
    field: str,
) -> None:
    config, observations = _mutation_fixture(tmp_path)
    observations[field] = False if field == "clean" else "0" * 40
    process = FakeProcess()
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.PROVENANCE_BINDING_FAILURE
    assert process.calls == 0


def test_predecessor_drift_is_rejected_before_wheelhouse_and_pip(tmp_path: Path) -> None:
    config, observations = _mutation_fixture(tmp_path)
    observations["live_packages"] = []
    process = FakeProcess()
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.PREDECESSOR_LIVE_BASELINE_MISMATCH
    assert process.calls == 0


@pytest.mark.parametrize("mutation", ["missing", "extra", "tamper"])
def test_phase_a_wheel_failures_never_launch_pip(tmp_path: Path, mutation: str) -> None:
    config, observations = _mutation_fixture(tmp_path)
    target = config.wheelhouse / "exchange_calendars-4.13.2-py3-none-any.whl"
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        target.with_name("extra-package-1.0-py3-none-any.whl").write_bytes(b"extra")
    else:
        target.write_bytes(b"tampered")
    process = FakeProcess()
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE
    assert process.calls == 0


def test_immediate_preinstall_recheck_detects_path_change_before_pip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations = _mutation_fixture(tmp_path)
    original = preflight._derive_wheelhouse
    calls = 0
    target = config.wheelhouse / "toolz-1.1.0-py3-none-any.whl"

    def recheck_then_remove(preflight_config: preflight.PreflightConfig, candidate: dict[str, object]) -> tuple[dict[str, object], list[str]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            target.unlink()
        return original(preflight_config, candidate)

    monkeypatch.setattr(preflight, "_derive_wheelhouse", recheck_then_remove)
    process = FakeProcess()
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE
    assert calls == 2
    assert process.calls == 0


def test_missing_fresh_authority_does_not_consume_step3_authority_or_launch(tmp_path: Path) -> None:
    config, observations = _mutation_fixture(tmp_path)
    process = FakeProcess()
    result = runner.run_mutation(config, observations, process_runner=process)
    assert result["failure_code"] == runner.MUTATION_AUTHORITY_MISSING
    assert result["mutation_authority_consumed"] is False
    assert result["mutation_started"] is False
    assert process.calls == 0
    assert not config.attempt_root.exists()


def test_nonzero_process_exit_is_exactly_persisted_without_retry(tmp_path: Path) -> None:
    config, observations = _mutation_fixture(tmp_path)
    process = FakeProcess(returncode=7, stdout=b"partial-out", stderr=b"failure-err")
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.MUTATION_PROCESS_FAILURE
    assert process.calls == 1
    assert process.communicate_calls == 1
    state = _complete_state(config)
    assert state["process_exit_code"] == 7
    assert state["stdout_sha256"] == hashlib.sha256(b"partial-out").hexdigest()
    assert state["stderr_sha256"] == hashlib.sha256(b"failure-err").hexdigest()
    assert (config.attempt_root / "stdout.bin").read_bytes() == b"partial-out"
    assert (config.attempt_root / "stderr.bin").read_bytes() == b"failure-err"


def test_boolean_process_exit_code_is_rejected_and_not_coerced(tmp_path: Path) -> None:
    config, observations = _mutation_fixture(tmp_path)
    process = FakeProcess(returncode=True)
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.MUTATION_ATTEMPT_STATE_FAILURE
    assert process.calls == 1
    assert process.communicate_calls == 1
    state = _complete_state(config)
    assert state["process_exit_code"] is None
    assert isinstance(state["process_exit_code"], (int, type(None)))


def test_process_start_failure_consumes_authority_once_and_preserves_state(tmp_path: Path) -> None:
    config, observations = _mutation_fixture(tmp_path)
    calls = 0

    def fail_to_start(_argv: Sequence[str]) -> object:
        nonlocal calls
        calls += 1
        raise OSError("synthetic process start failure")

    result = runner.run_mutation(
        config,
        observations,
        mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN,
        process_runner=fail_to_start,
    )
    assert result["failure_code"] == runner.MUTATION_PROCESS_START_FAILURE
    assert calls == 1
    state = _complete_state(config)
    assert state["process_start_attempted"] is True
    assert state["process_started"] is False
    assert state["mutation_authority_consumed"] is True
    assert state["mutation_started"] is True
    assert state["process_exit_code"] is None
    assert state["retry_authorized"] is False
    runner.validate_attempt_state(state)


def test_existing_attempt_root_is_no_overwrite_failure(tmp_path: Path) -> None:
    config, observations = _mutation_fixture(tmp_path)
    config.attempt_root.mkdir()
    process = FakeProcess()
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.MUTATION_ATTEMPT_STATE_FAILURE
    assert process.calls == 0


def test_started_state_persistence_failure_waits_for_started_child_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, observations = _mutation_fixture(tmp_path)
    original = runner._write_state

    def fail_started_state(root: Path, filename: str, state: dict[str, object]) -> None:
        if filename == "attempt_state_started.json":
            raise runner.MutationValidationError("synthetic state persistence failure")
        original(root, filename, state)

    monkeypatch.setattr(runner, "_write_state", fail_started_state)
    process = FakeProcess()
    result = runner.run_mutation(
        config,
        observations,
        mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN,
        process_runner=process,
    )
    assert result["failure_code"] == runner.MUTATION_ATTEMPT_STATE_FAILURE
    assert result["mutation_authority_consumed"] is True
    assert result["mutation_started"] is True
    assert process.calls == 1
    assert process.communicate_calls == 1
    state = _complete_state(config)
    assert state["process_started"] is True
    runner.validate_attempt_state(state)
