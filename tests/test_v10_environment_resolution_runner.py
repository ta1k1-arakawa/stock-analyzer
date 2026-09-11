from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

import scripts.v10_environment_resolution_runner as runner
from scripts.v10_environment_extension_contract import (
    PREDECESSOR_LOCK_BLOB_SHA1,
    PREDECESSOR_LOCK_SHA256,
    PREDECESSOR_PACKAGE_SET,
    ContractValidationError,
    validate_resolution_evidence,
    validate_successor_lock_candidate,
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
    repo_root = Path(__file__).parents[1]
    raw = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"HEAD:{runner.DIRECT_SPEC_RELATIVE.as_posix()}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        shell=False,
    ).stdout
    assert raw == runner.DIRECT_SPEC_BYTES
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 4
    assert runner.validate_direct_spec_bytes(raw) == runner.DIRECT_SPEC_SHA256


def test_direct_spec_git_attributes_bind_lf_and_worktree_bytes() -> None:
    repo_root = Path(__file__).parents[1]
    relative = runner.DIRECT_SPEC_RELATIVE.as_posix()
    attributes = subprocess.run(
        ["git", "-C", str(repo_root), "check-attr", "text", "eol", "--", relative],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        shell=False,
    ).stdout.decode("utf-8").strip().splitlines()
    assert attributes == [
        f"{relative}: text: set",
        f"{relative}: eol: lf",
    ]

    worktree = (repo_root / runner.DIRECT_SPEC_RELATIVE).read_bytes()
    committed = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"HEAD:{relative}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        shell=False,
    ).stdout
    assert worktree == committed == runner.DIRECT_SPEC_BYTES
    assert hashlib.sha256(worktree).hexdigest() == runner.DIRECT_SPEC_SHA256


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

    with pytest.raises(ContractValidationError, match="FRESH_HUMAN_AUTHORITY_REQUIRED"):
        runner.run_phase_b(config, execute_resolution=True, observations=_valid_observations(config))
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
        fresh_human_authority_confirmed=True,
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


def test_phase_b_cli_without_human_flag_creates_no_root_or_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(runner, "run_phase_a", lambda *_args, **_kwargs: {"status": "PASS"})
    arguments = [
        "phase-b",
        "--repo-root",
        str(config.repo_root),
        "--expected-head",
        config.expected_current_head,
        "--expected-extension-design-sha",
        config.expected_extension_design_sha,
        "--expected-runner-sha",
        config.expected_reviewed_runner_sha,
        "--expected-direct-spec-git-blob-sha1",
        config.expected_direct_spec_git_blob_sha1,
        "--expected-direct-spec-sha256",
        config.expected_direct_spec_sha256,
        "--durable-root",
        str(config.durable_root),
        "--execute-resolution",
    ]
    assert runner.main(arguments) == 1
    assert not config.durable_root.exists()


def test_phase_b_cli_forwards_both_explicit_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    seen: dict[str, object] = {}

    def fake_phase_b(received: runner.PhaseAConfig, **kwargs: object) -> dict[str, object]:
        seen["config"] = received
        seen.update(kwargs)
        return {"status": "PASS", "failure_code": "NONE"}

    monkeypatch.setattr(runner, "run_phase_b", fake_phase_b)
    arguments = [
        "phase-b",
        "--repo-root", str(config.repo_root),
        "--expected-head", config.expected_current_head,
        "--expected-extension-design-sha", config.expected_extension_design_sha,
        "--expected-runner-sha", config.expected_reviewed_runner_sha,
        "--expected-direct-spec-git-blob-sha1", config.expected_direct_spec_git_blob_sha1,
        "--expected-direct-spec-sha256", config.expected_direct_spec_sha256,
        "--durable-root", str(config.durable_root),
        "--execute-resolution",
        "--fresh-human-authority-confirmed",
    ]
    assert runner.main(arguments) == 0
    received = seen["config"]
    assert isinstance(received, runner.PhaseAConfig)
    assert received.repo_root == config.repo_root
    assert received.durable_root == config.durable_root
    assert seen["execute_resolution"] is True
    assert seen["fresh_human_authority_confirmed"] is True


def test_phase_b_persists_boundary_before_popen_and_exact_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    seen: dict[str, object] = {}

    def fake_popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        state = json.loads((config.durable_root / runner.STATE_NAME).read_text(encoding="utf-8"))
        seen.update(state)
        return _FakeProcess(0)

    runner.run_phase_b(
        config,
        execute_resolution=True,
        fresh_human_authority_confirmed=True,
        observations=_valid_observations(config),
        popen_factory=fake_popen,
    )
    assert seen["attempt_boundary_crossed"] is True
    assert seen["human_authority_consumed"] is True
    assert seen["process_started"] is None
    assert seen["process_exit_code"] is None
    assert seen["package_resolution_process_invocations"] == 0
    assert seen["expected_current_head"] == config.expected_current_head
    assert seen["frozen_v10_design_git_sha"] == runner.FROZEN_V10_DESIGN_SHA
    assert seen["extension_design_git_sha"] == config.expected_extension_design_sha
    assert seen["reviewed_resolution_implementation_git_sha"] == config.expected_reviewed_runner_sha
    assert seen["direct_spec_git_blob_sha1"] == config.expected_direct_spec_git_blob_sha1
    assert seen["direct_spec_sha256"] == config.expected_direct_spec_sha256
    assert seen["predecessor_lock_git_blob_sha1"] == PREDECESSOR_LOCK_BLOB_SHA1
    assert seen["predecessor_lock_sha256"] == PREDECESSOR_LOCK_SHA256
    assert seen["resolution_policy_id"] == runner.RESOLUTION_POLICY_ID
    assert seen["package_index_id"] == runner.PACKAGE_INDEX_ID
    assert set(seen) == runner.ATTEMPT_STATE_KEYS


def test_phase_b_root_creation_failure_does_not_consume_authority(tmp_path: Path) -> None:
    blocking_parent = tmp_path / "not-a-directory"
    blocking_parent.write_text("block", encoding="utf-8")
    config = _config(tmp_path, durable_root=blocking_parent / "attempt")
    with pytest.raises(ContractValidationError, match="DURABLE_ATTEMPT_ROOT_CREATION_FAILURE"):
        runner.run_phase_b(
            config,
            execute_resolution=True,
            fresh_human_authority_confirmed=True,
            observations=_valid_observations(config),
        )
    assert not (config.durable_root / runner.STATE_NAME).exists()


class _InspectingWaitProcess:
    def __init__(self, state_path: Path, *, fail: bool = False):
        self.state_path = state_path
        self.fail = fail

    def wait(self) -> int:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        assert state["process_started"] is True
        assert state["process_exit_code"] is None
        assert state["package_resolution_process_invocations"] == 1
        assert state["resolution_completed"] is False
        if self.fail:
            raise RuntimeError("synthetic wait failure")
        return 0


def test_phase_b_marks_started_before_wait_and_wait_failure_is_ambiguous_to_phase_c(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def fake_popen(*_args: object, **_kwargs: object) -> _InspectingWaitProcess:
        return _InspectingWaitProcess(config.durable_root / runner.STATE_NAME, fail=True)

    with pytest.raises(RuntimeError, match="synthetic wait failure"):
        runner.run_phase_b(
            config,
            execute_resolution=True,
            fresh_human_authority_confirmed=True,
            observations=_valid_observations(config),
            popen_factory=fake_popen,
        )
    state = json.loads((config.durable_root / runner.STATE_NAME).read_text(encoding="utf-8"))
    assert state["process_started"] is True
    assert state["process_exit_code"] is None
    assert state["package_resolution_process_invocations"] == 1
    runner._validate_attempt_state(config, state)
    with pytest.raises(ContractValidationError, match="ATTEMPT_STATE_AMBIGUOUS"):
        runner.run_phase_c(config)
    assert not runner.published_candidate_path(config.durable_root).exists()
    assert not runner.published_evidence_path(config.durable_root).exists()


@pytest.mark.parametrize("invocation_count", [True, False, -1, 2, 1.0, "1", None])
def test_phase_c_rejects_non_strict_attempt_invocation_count_before_inspection(
    tmp_path: Path, invocation_count: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    state_path = config.durable_root / runner.STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["package_resolution_process_invocations"] = invocation_count
    state_path.write_bytes(runner.canonical_json_bytes(state))
    monkeypatch.setattr(
        runner,
        "_inspect_phase_c_wheelhouse",
        lambda *_args: pytest.fail("wheelhouse inspection must not occur"),
    )
    with pytest.raises(ContractValidationError, match="ATTEMPT_STATE_SCHEMA_INVALID"):
        runner.run_phase_c(config)
    assert not runner.published_candidate_path(config.durable_root).exists()
    assert not runner.published_evidence_path(config.durable_root).exists()


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
            fresh_human_authority_confirmed=True,
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
        fresh_human_authority_confirmed=True,
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
        fresh_human_authority_confirmed=True,
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
    candidate_path = runner.published_candidate_path(config.durable_root)
    evidence_path = runner.published_evidence_path(config.durable_root)
    assert candidate_path.parent == runner.published_artifact_directory(config.durable_root)
    assert candidate_path.parent.exists()
    assert not runner.staging_artifact_directory(config.durable_root).exists()
    assert candidate_path.exists() and evidence_path.exists()
    candidate_bytes = candidate_path.read_bytes()
    candidate = json.loads(candidate_bytes)
    evidence = json.loads(evidence_path.read_bytes())
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    assert result["candidate_sha256"] == candidate_sha
    assert evidence["successor_lock_candidate_sha256"] == candidate_sha
    assert evidence["candidate_artifact_created"] is True
    validate_successor_lock_candidate(
        candidate,
        expected_extension_design_sha=config.expected_extension_design_sha,
        expected_reviewed_resolution_implementation_sha=config.expected_reviewed_runner_sha,
    )
    validate_resolution_evidence(
        evidence,
        expected_extension_design_sha=config.expected_extension_design_sha,
        expected_reviewed_resolution_implementation_sha=config.expected_reviewed_runner_sha,
        expected_direct_spec_git_blob_sha1=config.expected_direct_spec_git_blob_sha1,
        expected_direct_spec_sha256=config.expected_direct_spec_sha256,
        expected_successor_lock_candidate_sha256=candidate_sha,
    )
    assert "exchange-calendars" in [item["name"] for item in candidate["resolved_packages"]]
    assert "pandas-market-calendars" in [item["name"] for item in candidate["resolved_packages"]]
    assert str(config.durable_root) not in candidate_path.read_text(encoding="utf-8")
    assert "https://pypi.org/simple" not in evidence_path.read_text(encoding="utf-8")


def test_phase_c_pass_exposes_both_files_only_after_one_directory_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    original_rename = runner.os.rename
    rename_calls = 0

    def observing_rename(source: str | bytes, destination: str | bytes, *args: object, **kwargs: object) -> None:
        nonlocal rename_calls
        rename_calls += 1
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path == runner.staging_artifact_directory(config.durable_root)
        assert destination_path == runner.published_artifact_directory(config.durable_root)
        assert runner.published_candidate_path(config.durable_root).exists() is False
        assert runner.published_evidence_path(config.durable_root).exists() is False
        assert (source_path / runner.CANDIDATE_NAME).exists()
        assert (source_path / runner.EVIDENCE_NAME).exists()
        original_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(runner.os, "rename", observing_rename)
    result = runner.run_phase_c(config)
    assert result["status"] == "PASS"
    assert rename_calls == 1
    assert runner.published_candidate_path(config.durable_root).exists()
    assert runner.published_evidence_path(config.durable_root).exists()
    assert not runner.staging_artifact_directory(config.durable_root).exists()


def test_phase_c_candidate_staging_write_failure_never_exposes_final_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    original_create = runner._atomic_create_no_overwrite

    def fail_candidate(path: Path, raw: bytes) -> None:
        if path.name == runner.CANDIDATE_NAME:
            raise OSError("synthetic candidate staging failure")
        original_create(path, raw)

    monkeypatch.setattr(runner, "_atomic_create_no_overwrite", fail_candidate)
    with pytest.raises(OSError, match="synthetic candidate staging failure"):
        runner.run_phase_c(config)
    assert not runner.published_artifact_directory(config.durable_root).exists()


def test_phase_c_evidence_staging_write_failure_never_exposes_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    original_create = runner._atomic_create_no_overwrite

    def fail_evidence(path: Path, raw: bytes) -> None:
        if path.name == runner.EVIDENCE_NAME:
            raise OSError("synthetic evidence staging failure")
        original_create(path, raw)

    monkeypatch.setattr(runner, "_atomic_create_no_overwrite", fail_evidence)
    with pytest.raises(OSError, match="synthetic evidence staging failure"):
        runner.run_phase_c(config)
    assert not runner.published_artifact_directory(config.durable_root).exists()
    assert not runner.published_candidate_path(config.durable_root).exists()
    assert (runner.staging_artifact_directory(config.durable_root) / runner.CANDIDATE_NAME).exists()


def test_phase_c_rename_failure_leaves_staging_without_final_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _run_successful_phase_b(tmp_path)

    def fail_rename(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic publication rename failure")

    monkeypatch.setattr(runner.os, "rename", fail_rename)
    with pytest.raises(OSError, match="synthetic publication rename failure"):
        runner.run_phase_c(config)
    assert runner.staging_artifact_directory(config.durable_root).exists()
    assert not runner.published_artifact_directory(config.durable_root).exists()


@pytest.mark.parametrize("existing_namespace", ["staging", "final"])
def test_phase_c_existing_publication_namespace_stops_without_cleanup_or_overwrite(
    tmp_path: Path, existing_namespace: str
) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    namespace = (
        runner.staging_artifact_directory(config.durable_root)
        if existing_namespace == "staging"
        else runner.published_artifact_directory(config.durable_root)
    )
    namespace.mkdir()
    marker = namespace / "marker"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(ContractValidationError, match="CHATGPT_DECISION_REQUIRED"):
        runner.run_phase_c(config)
    assert marker.read_text(encoding="utf-8") == "preserve"
    assert namespace.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("expected_current_head", "d" * 40),
        ("expected_reviewed_runner_sha", "e" * 40),
        ("expected_extension_design_sha", "f" * 40),
        ("expected_direct_spec_git_blob_sha1", "1" * 40),
        ("expected_direct_spec_sha256", "2" * 64),
    ],
)
def test_phase_c_rejects_changed_phase_b_provenance_before_artifacts(
    tmp_path: Path, field: str, replacement: str
) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    changed = replace(config, **{field: replacement})
    with pytest.raises(ContractValidationError, match="ATTEMPT_STATE_PROVENANCE_MISMATCH"):
        runner.run_phase_c(changed)
    assert not runner.published_candidate_path(config.durable_root).exists()
    assert not runner.published_evidence_path(config.durable_root).exists()


def test_phase_c_rejects_changed_predecessor_binding_before_artifacts(tmp_path: Path) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    state_path = config.durable_root / runner.STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["predecessor_lock_sha256"] = "3" * 64
    state_path.write_bytes(runner.canonical_json_bytes(state))
    with pytest.raises(ContractValidationError, match="ATTEMPT_STATE_PROVENANCE_MISMATCH"):
        runner.run_phase_c(config)
    assert not runner.published_candidate_path(config.durable_root).exists()
    assert not runner.published_evidence_path(config.durable_root).exists()


def test_phase_c_rejects_extra_or_missing_attempt_state_key_before_artifacts(tmp_path: Path) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    state_path = config.durable_root / runner.STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["unexpected"] = True
    state_path.write_bytes(runner.canonical_json_bytes(state))
    with pytest.raises(ContractValidationError, match="ATTEMPT_STATE_SCHEMA_INVALID"):
        runner.run_phase_c(config)
    assert not runner.published_candidate_path(config.durable_root).exists()
    assert not runner.published_evidence_path(config.durable_root).exists()


def test_phase_c_launch_failure_writes_fail_evidence_without_candidate(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def fake_popen(*_args: object, **_kwargs: object) -> _FakeProcess:
        raise OSError("synthetic launch failure")

    runner.run_phase_b(
        config,
        execute_resolution=True,
        fresh_human_authority_confirmed=True,
        observations=_valid_observations(config),
        popen_factory=fake_popen,
    )
    result = runner.run_phase_c(config)
    assert result["failure_code"] == "RESOLUTION_PROCESS_FAILURE"
    assert not runner.published_candidate_path(config.durable_root).exists()
    evidence = json.loads(runner.published_evidence_path(config.durable_root).read_text(encoding="utf-8"))
    assert evidence["process_started"] is False
    assert evidence["process_exit_code"] is None
    assert evidence["candidate_artifact_created"] is False
    assert str(config.durable_root) not in json.dumps(evidence)
    assert set(runner.published_artifact_directory(config.durable_root).iterdir()) == {runner.published_evidence_path(config.durable_root)}


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
    assert not runner.published_candidate_path(config.durable_root).exists()
    evidence = json.loads(runner.published_evidence_path(config.durable_root).read_text(encoding="utf-8"))
    assert evidence["candidate_artifact_created"] is False


def _apply_compound_failure(wheelhouse: Path, case: str) -> None:
    source = wheelhouse / "source-package-1.0.0.tar.gz"
    if case in {"malformed_plus_source", "malformed_missing_source", "wrong_pmc_plus_source", "wrong_pmc_exchange_missing_source", "wrong_pmc_exchange_missing"}:
        if case.startswith("wrong_pmc"):
            (wheelhouse / "pandas_market_calendars-5.4.0-py3-none-any.whl").unlink()
            _write_wheel(wheelhouse, "pandas-market-calendars", "9.9.9")
            if "exchange_missing" in case:
                (wheelhouse / "exchange_calendars-5.0.0-py3-none-any.whl").unlink()
        else:
            (wheelhouse / "exchange_calendars-5.0.0-py3-none-any.whl").write_bytes(b"malformed")
        if case == "malformed_missing_source":
            (wheelhouse / "pandas_market_calendars-5.4.0-py3-none-any.whl").unlink()
        if case.endswith("_source"):
            source.write_bytes(b"source")
    elif case in {"drift_plus_source", "drift_missing_required_plus_source"}:
        (wheelhouse / "cffi-2.1.1-py3-none-any.whl").unlink()
        if case == "drift_missing_required_plus_source":
            (wheelhouse / "pandas_market_calendars-5.4.0-py3-none-any.whl").unlink()
        source.write_bytes(b"source")
    elif case == "missing_pmc_plus_source":
        (wheelhouse / "pandas_market_calendars-5.4.0-py3-none-any.whl").unlink()
        source.write_bytes(b"source")
    elif case == "missing_exchange_plus_source":
        (wheelhouse / "exchange_calendars-5.0.0-py3-none-any.whl").unlink()
        source.write_bytes(b"source")
    elif case == "valid_plus_source":
        source.write_bytes(b"source")
    else:
        raise AssertionError(f"unknown compound case: {case}")


@pytest.mark.parametrize(
    ("case", "expected_failure"),
    [
        ("malformed_plus_source", "RESOLUTION_REPORT_INVALID"),
        ("malformed_missing_source", "RESOLUTION_REPORT_INVALID"),
        ("drift_plus_source", "PREDECESSOR_PIN_DRIFT"),
        ("drift_missing_required_plus_source", "PREDECESSOR_PIN_DRIFT"),
        ("missing_pmc_plus_source", "REQUIRED_DISTRIBUTION_MISSING"),
        ("missing_exchange_plus_source", "REQUIRED_DISTRIBUTION_MISSING"),
        ("valid_plus_source", "SOURCE_DISTRIBUTION_REQUIRED"),
        ("wrong_pmc_plus_source", "RESOLUTION_REPORT_INVALID"),
        ("wrong_pmc_exchange_missing_source", "RESOLUTION_REPORT_INVALID"),
        ("wrong_pmc_exchange_missing", "RESOLUTION_REPORT_INVALID"),
    ],
)
def test_phase_c_compound_wheelhouse_failures_use_frozen_precedence(
    tmp_path: Path, case: str, expected_failure: str
) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    _apply_compound_failure(config.durable_root / runner.WHEELHOUSE_NAME, case)
    result = runner.run_phase_c(config)
    assert result["failure_code"] == expected_failure
    assert result["candidate_artifact_created"] is False
    assert not runner.published_candidate_path(config.durable_root).exists()
    evidence = json.loads(runner.published_evidence_path(config.durable_root).read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL"
    assert evidence["failure_code"] == expected_failure
    assert evidence["candidate_artifact_created"] is False


def test_phase_c_predecessor_drift_is_not_success(tmp_path: Path) -> None:
    config, _ = _run_successful_phase_b(tmp_path)
    _write_wheel(config.durable_root / runner.WHEELHOUSE_NAME, "cffi", "99.0.0")
    (config.durable_root / runner.WHEELHOUSE_NAME / "cffi-2.1.1-py3-none-any.whl").unlink()
    result = runner.run_phase_c(config)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "PREDECESSOR_PIN_DRIFT"
    assert not runner.published_candidate_path(config.durable_root).exists()
