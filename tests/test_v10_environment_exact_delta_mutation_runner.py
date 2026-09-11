from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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

    def __call__(self, argv: list[str] | tuple[str, ...]) -> SimpleNamespace:
        self.calls += 1
        self.argv = list(argv)
        return SimpleNamespace(returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)


def _complete_state(config: runner.MutationConfig) -> dict[str, object]:
    return json.loads((config.attempt_root / "attempt_state_complete.json").read_text(encoding="utf-8"))


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
    assert state["process_started"] is True
    assert state["mutation_authority_consumed"] is True
    assert state["mutation_started"] is True
    assert state["process_exit_code"] is None
    assert state["retry_authorized"] is False


def test_existing_attempt_root_is_no_overwrite_failure(tmp_path: Path) -> None:
    config, observations = _mutation_fixture(tmp_path)
    config.attempt_root.mkdir()
    process = FakeProcess()
    result = runner.run_mutation(config, observations, mutation_authority_token=runner.FRESH_MUTATION_AUTHORITY_TOKEN, process_runner=process)
    assert result["failure_code"] == runner.MUTATION_ATTEMPT_STATE_FAILURE
    assert process.calls == 0


def test_started_state_persistence_failure_does_not_consume_authority_or_launch(
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
    assert result["mutation_authority_consumed"] is False
    assert result["mutation_started"] is False
    assert process.calls == 0
