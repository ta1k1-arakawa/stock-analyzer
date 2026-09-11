from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import scripts.v10_environment_resolution_runner as runner
from scripts.v10_environment_extension_contract import (
    PREDECESSOR_LOCK_BLOB_SHA1,
    PREDECESSOR_LOCK_SHA256,
    PREDECESSOR_PACKAGE_SET,
    ContractValidationError,
)


EXTENSION_SHA = "a" * 40
RUNNER_SHA = "b" * 40


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()


def _config(tmp_path: Path, *, durable_root: Path | None = None) -> runner.PhaseAConfig:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    direct = runner.DIRECT_SPEC_BYTES
    lock = (Path(__file__).parents[1] / runner.LOCK_RELATIVE).read_bytes().replace(b"\r\n", b"\n")
    return runner.PhaseAConfig(
        repo_root=repo,
        expected_current_head="c" * 40,
        expected_extension_design_sha=EXTENSION_SHA,
        expected_reviewed_runner_sha=RUNNER_SHA,
        expected_direct_spec_git_blob_sha1=_git_blob_sha1(direct),
        expected_direct_spec_sha256=hashlib.sha256(direct).hexdigest(),
        durable_root=durable_root or (tmp_path / "durable-attempt"),
        governed_roots=(tmp_path / "governed-root",),
    )


def _valid_observations(config: runner.PhaseAConfig) -> dict[str, object]:
    lock = (Path(__file__).parents[1] / runner.LOCK_RELATIVE).read_bytes().replace(b"\r\n", b"\n")
    return {
        "repository_identity": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "branch": runner.AUTHORITATIVE_BRANCH,
        "head": config.expected_current_head,
        "clean": True,
        "frozen_design_ok": True,
        "extension_design_ok": True,
        "runner_binding_ok": True,
        "direct_spec_bytes": runner.DIRECT_SPEC_BYTES,
        "direct_spec_committed_bytes": runner.DIRECT_SPEC_BYTES,
        "direct_spec_git_blob_sha1": config.expected_direct_spec_git_blob_sha1,
        "direct_spec_sha256": config.expected_direct_spec_sha256,
        "predecessor_lock_committed_bytes": lock,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "live_packages": [{"name": name, "version": version} for name, version in PREDECESSOR_PACKAGE_SET],
        "interpreter_executable": str(config.canonical_interpreter),
        "python_implementation": "CPython",
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "pip_version": "25.0.1",
    }


def test_direct_spec_is_exact_utf8_lf_bytes() -> None:
    raw = (Path(__file__).parents[1] / runner.DIRECT_SPEC_RELATIVE).read_bytes()
    assert raw == runner.DIRECT_SPEC_BYTES
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 4
    assert runner.validate_direct_spec_bytes(raw) == runner.DIRECT_SPEC_SHA256


def test_phase_a_valid_synthetic_preflight_is_read_only(tmp_path: Path) -> None:
    config = _config(tmp_path)
    result = runner.run_phase_a(config, _valid_observations(config))
    assert result["status"] == "PASS"
    assert result["network_requests"] == 0
    assert result["writes"] == 0
    assert result["human_authority_consumed"] is False
    assert not config.durable_root.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("branch", "wrong-branch"),
        ("head", "d" * 40),
        ("clean", False),
        ("direct_spec_bytes", b"pandas\r\n"),
        ("predecessor_lock_sha256", "0" * 64),
        ("python_version", "3.13.0"),
        ("platform_machine", "ARM64"),
        ("pip_version", "25.0.0"),
    ],
)
def test_phase_a_rejects_provenance_environment_or_repo_mismatch(tmp_path: Path, field: str, value: object) -> None:
    config = _config(tmp_path)
    observations = _valid_observations(config)
    observations[field] = value
    result = runner.run_phase_a(config, observations)
    assert result["status"] == "FAIL"
    assert not config.durable_root.exists()


def test_phase_a_rejects_predecessor_package_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    observations = _valid_observations(config)
    observations["live_packages"] = copy.deepcopy(observations["live_packages"])
    observations["live_packages"][0]["version"] = "99.0.0"  # type: ignore[index]
    result = runner.run_phase_a(config, observations)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "PREDECESSOR_BASELINE_MISMATCH"


def test_phase_a_rejects_durable_root_collision_and_governed_overlap(tmp_path: Path) -> None:
    collision = tmp_path / "collision"
    collision.mkdir()
    config = _config(tmp_path, durable_root=collision)
    result = runner.run_phase_a(config, _valid_observations(config))
    assert result["failure_code"] == "DURABLE_ROOT_SAFETY_FAILURE"
    assert result["durable_root_status"] == "DURABLE_ROOT_ALREADY_EXISTS"

    overlap = _config(tmp_path, durable_root=config.repo_root / "attempt")
    result = runner.run_phase_a(overlap, _valid_observations(overlap))
    assert result["failure_code"] == "DURABLE_ROOT_SAFETY_FAILURE"
    assert result["durable_root_status"] == "DURABLE_ROOT_GOVERNED_PATH_OVERLAP"


def test_phase_a_rejects_symlink_or_reparse_parent_without_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(runner, "_is_reparse_or_symlink", lambda _path: True)
    result = runner.run_phase_a(config, _valid_observations(config))
    assert result["failure_code"] == "DURABLE_ROOT_SAFETY_FAILURE"
    assert result["durable_root_status"] == "DURABLE_ROOT_REPARSE_OR_SYMLINK"
    assert not config.durable_root.exists()


class _FakeProcess:
    def __init__(self, exit_code: int):
        self.exit_code = exit_code
        self.wait_calls = 0

    def wait(self) -> int:
        self.wait_calls += 1
        return self.exit_code


def test_phase_b_requires_flag_and_uses_exact_single_resolver(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ContractValidationError, match="EXPLICIT_EXECUTE_RESOLUTION_REQUIRED"):
        runner.run_phase_b(config, execute_resolution=False, observations=_valid_observations(config))
    assert not config.durable_root.exists()

    calls: list[tuple[list[str], dict[str, object]]] = []
    process = _FakeProcess(0)

    def fake_popen(argv: list[str], **kwargs: object) -> _FakeProcess:
        calls.append((argv, kwargs))
        return process

    parent_env = {"PIP_INDEX_URL": "bad", "pip_extra": "bad", "HTTPS_PROXY": "proxy"}
    result = runner.run_phase_b(
        config,
        execute_resolution=True,
        authority_consumer=lambda: True,
        observations=_valid_observations(config),
        popen_factory=fake_popen,
        parent_environment=parent_env,
    )
    assert result["status"] == "PASS"
    assert process.wait_calls == 1
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv == runner.build_resolution_argv(config.repo_root, config.durable_root / runner.WHEELHOUSE_NAME)
    assert kwargs["shell"] is False
    child_env = kwargs["env"]
    assert child_env["PIP_CONFIG_FILE"] == "NUL"
    assert all(not key.casefold().startswith("pip_") or key == "PIP_CONFIG_FILE" for key in child_env)
    assert child_env["HTTPS_PROXY"] == "proxy"
    assert kwargs["stdout"] is not kwargs["stderr"]
    assert "--find-links" not in argv
    assert "--extra-index-url" not in argv
    assert argv.count("download") == 1
    state = json.loads((config.durable_root / runner.STATE_NAME).read_text(encoding="utf-8"))
    assert state["human_authority_consumed"] is True
    assert state["package_resolution_process_invocations"] == 1


def test_phase_b_collision_prevents_launch(tmp_path: Path) -> None:
    root = tmp_path / "collision"
    root.mkdir()
    config = _config(tmp_path, durable_root=root)
    launched = False

    def fake_popen(*_args: object, **_kwargs: object) -> None:
        nonlocal launched
        launched = True

    with pytest.raises(ContractValidationError, match="PHASE_A_DURABLE_ROOT_SAFETY_FAILURE"):
        runner.run_phase_b(
            config,
            execute_resolution=True,
            authority_consumer=lambda: True,
            observations=_valid_observations(config),
            popen_factory=fake_popen,
        )
    assert launched is False


@pytest.mark.parametrize("exit_code", [None, 7])
def test_phase_b_failure_process_semantics(tmp_path: Path, exit_code: int | None) -> None:
    config = _config(tmp_path)

    def fake_popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        if exit_code is None:
            raise OSError("synthetic launch failure")
        return _FakeProcess(exit_code)

    result = runner.run_phase_b(
        config,
        execute_resolution=True,
        authority_consumer=lambda: True,
        observations=_valid_observations(config),
        popen_factory=fake_popen,
    )
    state = json.loads((config.durable_root / runner.STATE_NAME).read_text(encoding="utf-8"))
    if exit_code is None:
        assert result["process_started"] is False
        assert result["process_exit_code"] is None
        assert result["package_resolution_process_invocations"] == 0
        assert state["process_exit_code"] is None
    else:
        assert result["process_started"] is True
        assert result["process_exit_code"] == exit_code
        assert result["package_resolution_process_invocations"] == 1
        assert state["process_exit_code"] == exit_code


def _write_wheel(root: Path, name: str, version: str) -> Path:
    filename = f"{name.replace('-', '_')}-{version}-py3-none-any.whl"
    path = root / filename
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n".encode()
    wheel = b"Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel)
    return path


def _populate_valid_wheelhouse(wheelhouse: Path) -> None:
    packages = [{"name": name, "version": version} for name, version in PREDECESSOR_PACKAGE_SET]
    packages.extend(
        [
            {"name": "exchange-calendars", "version": "5.0.0"},
            {"name": "pandas-market-calendars", "version": "5.4.0"},
        ]
    )
    for package in sorted(packages, key=lambda item: item["name"]):
        _write_wheel(wheelhouse, package["name"], package["version"])


def _run_successful_phase_b(tmp_path: Path) -> tuple[runner.PhaseAConfig, dict[str, object]]:
    config = _config(tmp_path)

    def fake_popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        return _FakeProcess(0)

    runner.run_phase_b(
        config,
        execute_resolution=True,
        authority_consumer=lambda: True,
        observations=_valid_observations(config),
        popen_factory=fake_popen,
    )
    _populate_valid_wheelhouse(config.durable_root / runner.WHEELHOUSE_NAME)
    return config, _valid_observations(config)


def test_phase_c_success_builds_and_validates_candidate_and_evidence(tmp_path: Path) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    result = runner.run_phase_c(config)
    assert result["status"] == "PASS"
    assert result["candidate_artifact_created"] is True
    assert result["resolved_package_count"] == 17
    candidate_path = config.durable_root / runner.CANDIDATE_NAME
    evidence_path = config.durable_root / runner.EVIDENCE_NAME
    candidate_bytes = candidate_path.read_bytes()
    candidate = json.loads(candidate_bytes)
    evidence = json.loads(evidence_path.read_bytes())
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    assert result["candidate_sha256"] == candidate_sha
    assert evidence["successor_lock_candidate_sha256"] == candidate_sha
    assert evidence["candidate_artifact_created"] is True
    assert "exchange-calendars" in [item["name"] for item in candidate["resolved_packages"]]
    assert "pandas-market-calendars" in [item["name"] for item in candidate["resolved_packages"]]
    assert str(config.durable_root) not in candidate_path.read_text(encoding="utf-8")
    assert "https://pypi.org/simple" not in evidence_path.read_text(encoding="utf-8")


def test_phase_c_launch_failure_writes_fail_evidence_without_candidate(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def fake_popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        raise OSError("synthetic launch failure")

    runner.run_phase_b(
        config,
        execute_resolution=True,
        authority_consumer=lambda: True,
        observations=_valid_observations(config),
        popen_factory=fake_popen,
    )
    result = runner.run_phase_c(config)
    assert result["failure_code"] == "RESOLUTION_PROCESS_FAILURE"
    assert not (config.durable_root / runner.CANDIDATE_NAME).exists()
    evidence = json.loads((config.durable_root / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    assert evidence["process_started"] is False
    assert evidence["process_exit_code"] is None
    assert evidence["candidate_artifact_created"] is False
    assert str(config.durable_root) not in json.dumps(evidence)


@pytest.mark.parametrize("mutation", ["missing", "extra", "tamper", "source"])
def test_phase_c_invalid_wheelhouse_fails_without_candidate(tmp_path: Path, mutation: str) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    wheelhouse = config.durable_root / runner.WHEELHOUSE_NAME
    if mutation == "missing":
        (wheelhouse / "exchange_calendars-5.0.0-py3-none-any.whl").unlink()
    elif mutation == "extra":
        _write_wheel(wheelhouse, "cffi", "99.0.0")
    elif mutation == "tamper":
        (wheelhouse / "exchange_calendars-5.0.0-py3-none-any.whl").write_bytes(b"tampered")
    else:
        (wheelhouse / "source-package-1.0.0.tar.gz").write_bytes(b"source")
    result = runner.run_phase_c(config)
    assert result["status"] == "FAIL"
    assert result["candidate_artifact_created"] is False
    assert not (config.durable_root / runner.CANDIDATE_NAME).exists()
    evidence = json.loads((config.durable_root / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    assert evidence["candidate_artifact_created"] is False


def test_phase_c_predecessor_drift_is_not_success(tmp_path: Path) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    _write_wheel(config.durable_root / runner.WHEELHOUSE_NAME, "cffi", "99.0.0")
    (config.durable_root / runner.WHEELHOUSE_NAME / "cffi-2.1.1-py3-none-any.whl").unlink()
    result = runner.run_phase_c(config)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "PREDECESSOR_PIN_DRIFT"
    assert not (config.durable_root / runner.CANDIDATE_NAME).exists()
