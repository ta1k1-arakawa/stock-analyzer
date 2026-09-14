from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts import v10c_t0_ml_environment_contract as contract
from scripts import v10c_t0_ml_environment_resolution_runner as runner


REPO = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> runner.PhaseAConfig:
    repo = tmp_path / "repo"
    repo.mkdir()
    return runner.PhaseAConfig(
        repo_root=repo,
        expected_current_head="a" * 40,
        expected_reviewed_runner_sha="b" * 40,
        expected_direct_spec_git_blob_sha1=contract.git_blob_sha1(contract.DIRECT_SPEC_BYTES),
        expected_direct_spec_sha256=contract.DIRECT_SPEC_SHA256,
        durable_root=tmp_path / "attempt",
    )


def _observations(config: runner.PhaseAConfig) -> dict[str, object]:
    approval = json.loads((REPO / runner.APPROVAL_RELATIVE).read_text(encoding="utf-8"))
    lock = (REPO / runner.LOCK_RELATIVE).read_bytes()
    return {
        "repository_identity": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "branch": runner.AUTHORITATIVE_BRANCH,
        "origin_head": config.expected_current_head,
        "head": config.expected_current_head,
        "clean": True,
        "frozen_design_commit": contract.FROZEN_DESIGN_SHA,
        "frozen_design_blob": contract.FROZEN_DESIGN_BLOB,
        "approval_record_blob": contract.APPROVAL_RECORD_BLOB,
        "approval_record": approval,
        "reviewed_runner_blob": "runner-blob",
        "current_runner_blob": "runner-blob",
        "direct_spec_bytes": contract.DIRECT_SPEC_BYTES,
        "direct_spec_committed_bytes": contract.DIRECT_SPEC_BYTES,
        "direct_spec_blob": config.expected_direct_spec_git_blob_sha1,
        "direct_spec_sha256": config.expected_direct_spec_sha256,
        "predecessor_lock_bytes": lock,
        "predecessor_lock_blob": contract.PREDECESSOR_LOCK_BLOB,
        "predecessor_lock_sha256": contract.PREDECESSOR_LOCK_SHA256,
        "live_packages": [{"name": name, "version": version} for name, version in contract.PREDECESSOR_PACKAGE_SET],
        "interpreter_executable": str(config.canonical_interpreter),
        "python_implementation": "CPython",
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "pip_version": "25.0.1",
    }


def _write_wheel(root: Path, name: str, version: str) -> None:
    filename_name = name.replace("-", "_")
    filename = f"{filename_name}-{version}-py3-none-any.whl"
    path = root / filename
    dist_info = f"{filename_name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{dist_info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n")
        archive.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\nTag: py3-none-any\n")


def _populate_wheelhouse(root: Path) -> None:
    wheelhouse = root / runner.WHEELHOUSE_NAME
    for name, version in contract.PREDECESSOR_PACKAGE_SET + (("lightgbm", "4.6.0"), ("scikit-learn", "1.9.0")):
        _write_wheel(wheelhouse, name, version)


class _FakeProcess:
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code

    def wait(self) -> int:
        return self.exit_code


def test_phase_a_is_metadata_only_and_passes_synthetic_observations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    observations = _observations(config)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess in Phase A")))
    result = runner.run_phase_a(config, observations)
    assert result["status"] == "PASS"
    assert result["network_requests"] == 0
    assert result["writes"] == 0
    assert not config.durable_root.exists()


@pytest.mark.parametrize("field", ["branch", "origin_head", "head", "clean"])
def test_phase_a_rejects_governance_mismatch(tmp_path: Path, field: str) -> None:
    config = _config(tmp_path)
    observations = _observations(config)
    observations[field] = "wrong" if field != "clean" else False
    result = runner.run_phase_a(config, observations)
    assert result["status"] == "FAIL"


@pytest.mark.parametrize("field", ["frozen_design_blob", "approval_record_blob", "direct_spec_blob", "predecessor_lock_blob"])
def test_phase_a_rejects_frozen_provenance_mismatch(tmp_path: Path, field: str) -> None:
    config = _config(tmp_path)
    observations = _observations(config)
    observations[field] = "0" * 40
    result = runner.run_phase_a(config, observations)
    assert result["status"] == "FAIL"


def test_phase_a_rejects_existing_root_and_unsafe_ancestor(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.durable_root.mkdir()
    assert runner.run_phase_a(config, _observations(config))["status"] == "FAIL"
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir()
    link = tmp_path / "reparse"
    try:
        link.symlink_to(unsafe_parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    unsafe_config = runner.PhaseAConfig(**{**config.__dict__, "durable_root": link / "future"})
    result = runner.run_phase_a(unsafe_config, _observations(unsafe_config))
    assert result["status"] == "FAIL"


def test_exact_resolution_argv_and_environment_sanitization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    argv = runner.build_resolution_argv(config.repo_root, config.durable_root / runner.WHEELHOUSE_NAME)
    assert argv == [
        str(config.canonical_interpreter), "-m", "pip", "download", "--dest", str(config.durable_root / runner.WHEELHOUSE_NAME),
        "--only-binary=:all:", "--no-cache-dir", "--disable-pip-version-check", "--no-input", "--progress-bar", "off",
        "--retries", "0", "--timeout", "15", "--index-url", "https://pypi.org/simple", "--requirement",
        str(config.repo_root / runner.LOCK_RELATIVE), "--requirement", str(config.repo_root / runner.DIRECT_SPEC_RELATIVE),
        "--constraint", str(config.repo_root / runner.LOCK_RELATIVE),
    ]
    assert "install" not in argv
    assert runner.sanitize_environment({"PIP_INDEX_URL": "bad", "pip_extra": "bad", "HTTPS_PROXY": "keep"}) == {"HTTPS_PROXY": "keep", "PIP_CONFIG_FILE": "NUL"}


def test_phase_b_launches_at_most_once_and_preserves_nonzero_exit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def popen(argv: list[str], **kwargs: object) -> _FakeProcess:
        calls.append((argv, kwargs))
        return _FakeProcess(7)

    result = runner.run_phase_b(config, execute_resolution=True, fresh_human_authority_confirmed=True, observations=_observations(config), popen_factory=popen)
    assert result["process_started"] is True
    assert result["process_exit_code"] == 7
    assert result["package_resolution_process_invocations"] == 1
    assert len(calls) == 1
    state = json.loads((config.durable_root / runner.STATE_NAME).read_text(encoding="utf-8"))
    assert state["process_exit_code"] == 7
    assert state["human_authority_consumed"] is True


def test_phase_b_launch_failure_has_null_exit_and_no_retry(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def popen(*args: object, **kwargs: object) -> _FakeProcess:
        raise OSError("synthetic launch failure")

    result = runner.run_phase_b(config, execute_resolution=True, fresh_human_authority_confirmed=True, observations=_observations(config), popen_factory=popen)
    assert result["process_started"] is False
    assert result["process_exit_code"] is None
    assert result["package_resolution_process_invocations"] == 0
    state = json.loads((config.durable_root / runner.STATE_NAME).read_text(encoding="utf-8"))
    assert state["process_exit_code"] is None


def test_phase_b_requires_fresh_authority_and_execute_flag(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_b(config, execute_resolution=False, fresh_human_authority_confirmed=True, observations=_observations(config))
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_b(config, execute_resolution=True, fresh_human_authority_confirmed=False, observations=_observations(config))


def test_phase_c_accepts_first_synthetic_wheel_resolution(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner.run_phase_b(config, execute_resolution=True, fresh_human_authority_confirmed=True, observations=_observations(config), popen_factory=lambda *args, **kwargs: _FakeProcess(0))
    _populate_wheelhouse(config.durable_root)
    result = runner.run_phase_c(config)
    assert result["status"] == "PASS"
    assert result["candidate_artifact_created"] is True
    candidate = json.loads((config.durable_root / runner.CANDIDATE_NAME).read_text(encoding="utf-8"))
    assert candidate["resolved_package_count"] == 22
    assert candidate["lightgbm_version"] == "4.6.0"
    evidence = json.loads((config.durable_root / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    assert evidence["package_installations"] == 0
    assert evidence["t0_runs"] == 0
    assert evidence["payload_reads"] == 0


def test_phase_c_rejects_source_distribution_without_t0_or_network(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner.run_phase_b(config, execute_resolution=True, fresh_human_authority_confirmed=True, observations=_observations(config), popen_factory=lambda *args, **kwargs: _FakeProcess(0))
    (config.durable_root / runner.WHEELHOUSE_NAME / "lightgbm-4.6.0.tar.gz").write_bytes(b"source")
    result = runner.run_phase_c(config)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "SOURCE_DISTRIBUTION_REQUIRED"


def test_runner_has_no_t0_or_ml_import_path() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "import lightgbm" not in source
    assert "import sklearn" not in source
    assert "pip install" not in source
    assert "payload" not in source.lower() or "payload_reads" in source
