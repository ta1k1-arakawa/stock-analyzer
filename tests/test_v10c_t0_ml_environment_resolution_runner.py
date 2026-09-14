from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

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


def _write_wheel(root: Path, name: str, version: str, requires_dist: tuple[str, ...] = ()) -> None:
    filename_name = name.replace("-", "_")
    filename = f"{filename_name}-{version}-py3-none-any.whl"
    path = root / filename
    dist_info = f"{filename_name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        metadata += "".join(f"Requires-Dist: {requirement}\n" for requirement in requires_dist)
        archive.writestr(f"{dist_info}/METADATA", metadata + "\n")
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


def _metadata_probe_payload(
    packages: list[dict[str, str | None]], *, pip_version: str = "25.0.1"
) -> bytes:
    return json.dumps(
        {
            "interpreter_executable": "C:/canonical/.venv-real-execution/Scripts/python.exe",
            "python_implementation": "CPython",
            "python_version": "3.12.10",
            "platform_system": "Windows",
            "platform_machine": "AMD64",
            "sysconfig_platform": "win-amd64",
            "pip_version": pip_version,
            "live_packages": packages,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def test_phase_a_is_metadata_only_and_passes_synthetic_observations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    observations = _observations(config)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess in Phase A")))
    result = runner.run_phase_a(config, observations)
    assert result["status"] == "PASS"
    assert result["network_requests"] == 0
    assert result["writes"] == 0
    assert result["human_authority_consumed"] is False
    assert not config.durable_root.exists()


def test_canonical_child_observation_returns_exact_predecessor_20(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    packages = [{"name": name, "version": version} for name, version in contract.PREDECESSOR_PACKAGE_SET]

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(stdout=_metadata_probe_payload(packages))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner._probe_environment(tmp_path / "python.exe")
    assert result["live_packages"] == packages
    assert result["pip_version"] == "25.0.1"
    assert len(calls) == 1
    assert calls[0][0] == str(tmp_path / "python.exe")
    assert calls[0][1:3] == ["-c", runner._CANONICAL_METADATA_PROBE]


@pytest.mark.parametrize(
    ("raw_name", "expected"),
    [
        ("pdfminer.six", "pdfminer-six"),
        ("pandas_market_calendars", "pandas-market-calendars"),
    ],
)
def test_installed_metadata_name_uses_reviewed_normalization(raw_name: str, expected: str) -> None:
    assert runner._parse_installed_metadata_packages([{"name": raw_name, "version": "1.0"}]) == [
        {"name": expected, "version": "1.0"}
    ]


@pytest.mark.parametrize(
    "packages",
    [
        [{"name": "foo.bar", "version": "1.0"}, {"name": "foo-bar", "version": "1.0"}],
        [{"name": None, "version": "1.0"}],
        [{"name": "foo", "version": ""}],
    ],
)
def test_installed_metadata_malformed_identity_fails_closed(packages: list[dict[str, str | None]]) -> None:
    with pytest.raises(runner.RunnerValidationError, match="LIVE_PACKAGE_SET_UNPARSEABLE"):
        runner._parse_installed_metadata_packages(packages)


def test_phase_a_exact_predecessor_metadata_passes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert runner.run_phase_a(config, _observations(config))["status"] == "PASS"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda packages: packages[:-1],
        lambda packages: packages + [{"name": "unrelated", "version": "1.0"}],
        lambda packages: [{"name": item["name"], "version": "9.9.9"} if item["name"] == "cffi" else item for item in packages],
    ],
)
def test_phase_a_installed_metadata_set_drift_is_predecessor_pin_drift(
    tmp_path: Path, mutation: object
) -> None:
    config = _config(tmp_path)
    observations = _observations(config)
    observations["live_packages"] = mutation(observations["live_packages"])  # type: ignore[operator]
    assert runner.run_phase_a(config, observations)["failure_code"] == "PREDECESSOR_PIN_DRIFT"


def test_probe_pip_version_is_installed_metadata_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packages = [{"name": name, "version": version} for name, version in contract.PREDECESSOR_PACKAGE_SET]
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=_metadata_probe_payload(packages, pip_version="25.0.1")),
    )
    assert runner._probe_environment(tmp_path / "python.exe")["pip_version"] == "25.0.1"


def test_phase_a_rejects_installed_metadata_pip_version_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    observations = _observations(config)
    observations["pip_version"] = "24.0.0"
    assert runner.run_phase_a(config, observations)["failure_code"] == "CANONICAL_INTERPRETER_BINDING_FAILURE"


@pytest.mark.parametrize("raw", [b"not-json", _metadata_probe_payload([]).replace(b"\"live_packages\":[]", b"\"live_packages\":null")])
def test_malformed_child_json_or_probe_shape_is_invalid(raw: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=raw))
    with pytest.raises(runner.RunnerValidationError, match="CANONICAL_INTERPRETER_PROBE_INVALID"):
        runner._probe_environment(tmp_path / "python.exe")


def test_production_probe_has_no_pip_freeze_path() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "pip freeze" not in source
    assert "freeze --all" not in source
    assert "importlib.metadata" in runner._CANONICAL_METADATA_PROBE
    assert "lightgbm" not in runner._CANONICAL_METADATA_PROBE
    assert "sklearn" not in runner._CANONICAL_METADATA_PROBE


def test_direct_reference_presentation_is_irrelevant_to_production_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages = [{"name": name, "version": version} for name, version in contract.PREDECESSOR_PACKAGE_SET]
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(argv)
        return SimpleNamespace(stdout=_metadata_probe_payload(packages))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner._probe_environment(tmp_path / "python.exe")
    assert result["live_packages"] == packages
    assert len(calls) == 1
    assert "freeze" not in " ".join(calls[0])


@pytest.mark.parametrize(
    "failure_code",
    [
        "DURABLE_ROOT_NOT_ABSOLUTE",
        "DURABLE_ROOT_ALREADY_EXISTS",
        "DURABLE_ROOT_REPARSE_OR_SYMLINK",
        "DURABLE_ROOT_SAFETY_UNDETERMINED",
        "DURABLE_ROOT_GOVERNED_PATH_OVERLAP",
        "DURABLE_ROOT_PARENT_MISSING",
    ],
)
def test_phase_a_preserves_closed_durable_root_failure_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_code: str
) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(runner, "validate_durable_root", lambda *args, **kwargs: failure_code)
    result = runner.run_phase_a(config, _observations(config))
    assert result["failure_code"] == failure_code


def test_phase_a_preserves_canonical_interpreter_probe_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "_default_phase_a_observations",
        lambda config: (_ for _ in ()).throw(runner.RunnerValidationError("CANONICAL_INTERPRETER_PROBE_INVALID")),
    )
    result = runner.run_phase_a(_config(tmp_path))
    assert result["failure_code"] == "CANONICAL_INTERPRETER_PROBE_INVALID"


def test_phase_a_unknown_failure_remains_generic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner,
        "_default_phase_a_observations",
        lambda config: (_ for _ in ()).throw(runner.RunnerValidationError("UNEXPECTED_INTERNAL_FAILURE")),
    )
    result = runner.run_phase_a(_config(tmp_path))
    assert result["failure_code"] == "PHASE_A_PRECHECK_FAILURE"


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
    assert runner.run_phase_a(config, _observations(config))["failure_code"] == "DURABLE_ROOT_ALREADY_EXISTS"
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir()
    link = tmp_path / "reparse"
    try:
        link.symlink_to(unsafe_parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    unsafe_config = runner.PhaseAConfig(**{**config.__dict__, "durable_root": link / "future"})
    result = runner.run_phase_a(unsafe_config, _observations(unsafe_config))
    assert result["failure_code"] == "DURABLE_ROOT_REPARSE_OR_SYMLINK"


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
    candidate = json.loads(runner.published_candidate_path(config.durable_root).read_text(encoding="utf-8"))
    assert candidate["resolved_package_count"] == 22
    assert candidate["lightgbm_version"] == "4.6.0"
    evidence = json.loads(runner.published_evidence_path(config.durable_root).read_text(encoding="utf-8"))
    assert evidence["package_installations"] == 0
    assert evidence["t0_runs"] == 0
    assert evidence["payload_reads"] == 0


def test_phase_c_rejects_source_distribution_without_t0_or_network(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner.run_phase_b(config, execute_resolution=True, fresh_human_authority_confirmed=True, observations=_observations(config), popen_factory=lambda *args, **kwargs: _FakeProcess(0))
    wheelhouse = config.durable_root / runner.WHEELHOUSE_NAME
    _populate_wheelhouse(config.durable_root)
    (wheelhouse / "lightgbm-4.6.0.tar.gz").write_bytes(b"source")
    result = runner.run_phase_c(config)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "SOURCE_DISTRIBUTION_REQUIRED"
    assert runner.published_evidence_path(config.durable_root).is_file()
    assert {entry.name for entry in runner.published_artifact_directory(config.durable_root).iterdir()} == {runner.EVIDENCE_NAME}
    assert not (config.durable_root / runner.CANDIDATE_NAME).exists()
    assert not (config.durable_root / runner.EVIDENCE_NAME).exists()


def _prepare_phase_c_attempt(tmp_path: Path, mutate: object | None = None) -> runner.PhaseAConfig:
    config = _config(tmp_path)
    runner.run_phase_b(config, execute_resolution=True, fresh_human_authority_confirmed=True, observations=_observations(config), popen_factory=lambda *args, **kwargs: _FakeProcess(0))
    _populate_wheelhouse(config.durable_root)
    if mutate is not None:
        mutate(config.durable_root / runner.WHEELHOUSE_NAME)
    return config


def _phase_c_with_populated_wheelhouse(tmp_path: Path, mutate: object) -> dict[str, object]:
    config = _prepare_phase_c_attempt(tmp_path, mutate)
    return runner.run_phase_c(config)


def _wheel_path(wheelhouse: Path, name: str, version: str) -> Path:
    return wheelhouse / f"{name.replace('-', '_')}-{version}-py3-none-any.whl"


def test_phase_c_accepts_resolver_derived_dependency_closure(tmp_path: Path) -> None:
    def mutate(wheelhouse: Path) -> None:
        _wheel_path(wheelhouse, "lightgbm", "4.6.0").unlink()
        _write_wheel(wheelhouse, "lightgbm", "4.6.0", ("scipy>=1.8.0; platform_system == 'Windows'",))
        _write_wheel(wheelhouse, "scipy", "1.14.1")

    result = _phase_c_with_populated_wheelhouse(tmp_path, mutate)
    assert result["status"] == "PASS"
    assert result["resolved_package_count"] == 23


def test_phase_c_rejects_unrelated_extra_wheel_as_report_invalid(tmp_path: Path) -> None:
    result = _phase_c_with_populated_wheelhouse(tmp_path, lambda wheelhouse: _write_wheel(wheelhouse, "unrelated", "1.0.0"))
    assert result["failure_code"] == "RESOLUTION_REPORT_INVALID"


def test_candidate_rejects_unrelated_extra_package(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner.run_phase_b(config, execute_resolution=True, fresh_human_authority_confirmed=True, observations=_observations(config), popen_factory=lambda *args, **kwargs: _FakeProcess(0))
    _populate_wheelhouse(config.durable_root)
    assert runner.run_phase_c(config)["status"] == "PASS"
    candidate = json.loads(runner.published_candidate_path(config.durable_root).read_text(encoding="utf-8"))
    extra_wheelhouse = config.durable_root / "extra"
    extra_wheelhouse.mkdir()
    _write_wheel(extra_wheelhouse, "unrelated", "1.0.0")
    extra = contract.inspect_wheel_file(extra_wheelhouse / "unrelated-1.0.0-py3-none-any.whl")
    candidate["resolved_packages"].append({"name": "unrelated", "version": "1.0.0"})
    candidate["resolved_wheels"].append(extra)
    candidate["resolved_dependency_metadata"].append({"name": "unrelated", "version": "1.0.0", "requires_dist": [], "requires_python": None})
    candidate["resolved_packages"] = sorted(candidate["resolved_packages"], key=lambda item: (item["name"], item["version"]))
    candidate["resolved_wheels"] = sorted(candidate["resolved_wheels"], key=lambda item: (item["name"], item["version"]))
    candidate["resolved_dependency_metadata"] = sorted(candidate["resolved_dependency_metadata"], key=lambda item: (item["name"], item["version"]))
    candidate["resolved_package_count"] = 23
    with pytest.raises(contract.ContractValidationError):
        contract.validate_lock_candidate(candidate, expected_reviewed_sha=config.expected_reviewed_runner_sha, expected_direct_blob=config.expected_direct_spec_git_blob_sha1, expected_direct_sha=config.expected_direct_spec_sha256)


def test_phase_c_predecessor_missing_precedes_direct_and_source(tmp_path: Path) -> None:
    def mutate(wheelhouse: Path) -> None:
        _wheel_path(wheelhouse, "cffi", "2.1.1").unlink()
        (wheelhouse / "extra.tar.gz").write_bytes(b"source")

    result = _phase_c_with_populated_wheelhouse(tmp_path, mutate)
    assert result["failure_code"] == "PREDECESSOR_PIN_DRIFT"


def test_phase_c_predecessor_wrong_version_precedes_direct_and_source(tmp_path: Path) -> None:
    def mutate(wheelhouse: Path) -> None:
        _wheel_path(wheelhouse, "cffi", "2.1.1").unlink()
        _write_wheel(wheelhouse, "cffi", "9.9.9")
        (wheelhouse / "extra.tar.gz").write_bytes(b"source")

    result = _phase_c_with_populated_wheelhouse(tmp_path, mutate)
    assert result["failure_code"] == "PREDECESSOR_PIN_DRIFT"


def test_phase_c_missing_lightgbm_is_required_direct_distribution_missing(tmp_path: Path) -> None:
    result = _phase_c_with_populated_wheelhouse(tmp_path, lambda wheelhouse: _wheel_path(wheelhouse, "lightgbm", "4.6.0").unlink())
    assert result["failure_code"] == "REQUIRED_DIRECT_DISTRIBUTION_MISSING"


def test_phase_c_wrong_lightgbm_is_required_direct_distribution_missing(tmp_path: Path) -> None:
    def mutate(wheelhouse: Path) -> None:
        _wheel_path(wheelhouse, "lightgbm", "4.6.0").unlink()
        _write_wheel(wheelhouse, "lightgbm", "4.6.1")

    result = _phase_c_with_populated_wheelhouse(tmp_path, mutate)
    assert result["failure_code"] == "REQUIRED_DIRECT_DISTRIBUTION_MISSING"


def test_phase_c_missing_scikit_learn_is_required_direct_distribution_missing(tmp_path: Path) -> None:
    result = _phase_c_with_populated_wheelhouse(tmp_path, lambda wheelhouse: _wheel_path(wheelhouse, "scikit-learn", "1.9.0").unlink())
    assert result["failure_code"] == "REQUIRED_DIRECT_DISTRIBUTION_MISSING"


def test_phase_c_wrong_scikit_learn_is_required_direct_distribution_missing(tmp_path: Path) -> None:
    def mutate(wheelhouse: Path) -> None:
        _wheel_path(wheelhouse, "scikit-learn", "1.9.0").unlink()
        _write_wheel(wheelhouse, "scikit-learn", "1.9.1")

    result = _phase_c_with_populated_wheelhouse(tmp_path, mutate)
    assert result["failure_code"] == "REQUIRED_DIRECT_DISTRIBUTION_MISSING"


def test_phase_c_missing_direct_with_source_stays_required_direct_missing(tmp_path: Path) -> None:
    def mutate(wheelhouse: Path) -> None:
        _wheel_path(wheelhouse, "lightgbm", "4.6.0").unlink()
        (wheelhouse / "lightgbm-4.6.0.tar.gz").write_bytes(b"source")

    result = _phase_c_with_populated_wheelhouse(tmp_path, mutate)
    assert result["failure_code"] == "REQUIRED_DIRECT_DISTRIBUTION_MISSING"


def test_phase_c_predecessor_drift_with_source_stays_predecessor_drift(tmp_path: Path) -> None:
    def mutate(wheelhouse: Path) -> None:
        _wheel_path(wheelhouse, "cffi", "2.1.1").unlink()
        (wheelhouse / "cffi-2.1.1.tar.gz").write_bytes(b"source")

    result = _phase_c_with_populated_wheelhouse(tmp_path, mutate)
    assert result["failure_code"] == "PREDECESSOR_PIN_DRIFT"


def test_phase_c_malformed_wheel_with_source_is_report_invalid(tmp_path: Path) -> None:
    def mutate(wheelhouse: Path) -> None:
        (wheelhouse / "malformed-1.0.0-py3-none-any.whl").write_bytes(b"not a zip")
        (wheelhouse / "extra.tar.gz").write_bytes(b"source")

    result = _phase_c_with_populated_wheelhouse(tmp_path, mutate)
    assert result["failure_code"] == "RESOLUTION_REPORT_INVALID"


def test_phase_c_source_only_failure_after_valid_closure(tmp_path: Path) -> None:
    result = _phase_c_with_populated_wheelhouse(tmp_path, lambda wheelhouse: (wheelhouse / "extra.tar.gz").write_bytes(b"source"))
    assert result["failure_code"] == "SOURCE_DISTRIBUTION_REQUIRED"


def test_phase_c_closure_never_uses_general_venv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_general_environment_is_consulted(*args: object, **kwargs: object) -> None:
        raise AssertionError("general .venv consulted")

    monkeypatch.setattr(runner, "_probe_environment", fail_if_general_environment_is_consulted)
    result = _phase_c_with_populated_wheelhouse(tmp_path, lambda wheelhouse: None)
    assert result["status"] == "PASS"


def test_phase_c_failure_precedence_installation_precedes_alternate_environment(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner.run_phase_b(config, execute_resolution=True, fresh_human_authority_confirmed=True, observations=_observations(config), popen_factory=lambda *args, **kwargs: _FakeProcess(0))
    state_path = config.durable_root / runner.STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["package_installations"] = 1
    state["alternate_venv_created"] = True
    state_path.write_text(json.dumps(state), encoding="utf-8")
    result = runner.run_phase_c(config)
    assert result["failure_code"] == "UNAUTHORIZED_INSTALLATION"


def test_phase_c_candidate_staging_failure_leaves_no_final_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _prepare_phase_c_attempt(tmp_path)
    original_write = runner._write_bytes

    def fail_candidate(path: Path, raw: bytes, *, exclusive: bool = False) -> None:
        if path == runner.staging_artifact_directory(config.durable_root) / runner.CANDIDATE_NAME:
            raise OSError("synthetic candidate staging failure")
        original_write(path, raw, exclusive=exclusive)

    monkeypatch.setattr(runner, "_write_bytes", fail_candidate)
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_c(config)
    assert not runner.published_artifact_directory(config.durable_root).exists()
    assert runner.staging_artifact_directory(config.durable_root).is_dir()
    assert not runner.published_candidate_path(config.durable_root).exists()


def test_phase_c_evidence_staging_failure_leaves_candidate_staged_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _prepare_phase_c_attempt(tmp_path)
    original_write = runner._write_bytes

    def fail_evidence(path: Path, raw: bytes, *, exclusive: bool = False) -> None:
        if path == runner.staging_artifact_directory(config.durable_root) / runner.EVIDENCE_NAME:
            raise OSError("synthetic evidence staging failure")
        original_write(path, raw, exclusive=exclusive)

    monkeypatch.setattr(runner, "_write_bytes", fail_evidence)
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_c(config)
    assert not runner.published_artifact_directory(config.durable_root).exists()
    assert runner.staging_artifact_directory(config.durable_root).is_dir()
    assert (runner.staging_artifact_directory(config.durable_root) / runner.CANDIDATE_NAME).is_file()
    assert not runner.published_candidate_path(config.durable_root).exists()


def test_phase_c_rename_failure_preserves_staging_and_publishes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _prepare_phase_c_attempt(tmp_path)
    original_rename = runner.os.rename

    def fail_rename(source: str | bytes | Path, destination: str | bytes | Path) -> None:
        if Path(source) == runner.staging_artifact_directory(config.durable_root):
            raise OSError("synthetic rename failure")
        original_rename(source, destination)

    monkeypatch.setattr(runner.os, "rename", fail_rename)
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_c(config)
    assert not runner.published_artifact_directory(config.durable_root).exists()
    assert runner.staging_artifact_directory(config.durable_root).is_dir()


def test_phase_c_existing_final_directory_fails_without_overwrite(tmp_path: Path) -> None:
    config = _prepare_phase_c_attempt(tmp_path)
    final = runner.published_artifact_directory(config.durable_root)
    final.mkdir()
    sentinel = final / "sentinel"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_c(config)
    assert sentinel.read_bytes() == b"preserve"


def test_phase_c_existing_staging_directory_fails_without_resume(tmp_path: Path) -> None:
    config = _prepare_phase_c_attempt(tmp_path)
    staging = runner.staging_artifact_directory(config.durable_root)
    staging.mkdir()
    sentinel = staging / "sentinel"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_c(config)
    assert sentinel.read_bytes() == b"preserve"


@pytest.mark.parametrize("target_name", [runner.ARTIFACT_DIRECTORY_NAME, runner.ARTIFACT_STAGING_DIRECTORY_NAME])
def test_phase_c_rejects_artifact_namespace_symlink(tmp_path: Path, target_name: str) -> None:
    config = _prepare_phase_c_attempt(tmp_path)
    target = config.durable_root / target_name
    link_target = tmp_path / "link-target"
    link_target.mkdir()
    try:
        target.symlink_to(link_target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_c(config)
    assert target.is_symlink()


def test_phase_c_pass_final_file_set_is_exact(tmp_path: Path) -> None:
    config = _prepare_phase_c_attempt(tmp_path)
    assert runner.run_phase_c(config)["status"] == "PASS"
    assert {entry.name for entry in runner.published_artifact_directory(config.durable_root).iterdir()} == {runner.CANDIDATE_NAME, runner.EVIDENCE_NAME}
    assert not (config.durable_root / runner.CANDIDATE_NAME).exists()
    assert not (config.durable_root / runner.EVIDENCE_NAME).exists()


def test_phase_c_fail_final_file_set_is_evidence_only(tmp_path: Path) -> None:
    config = _prepare_phase_c_attempt(tmp_path, lambda wheelhouse: (wheelhouse / "extra.tar.gz").write_bytes(b"source"))
    assert runner.run_phase_c(config)["status"] == "FAIL"
    assert {entry.name for entry in runner.published_artifact_directory(config.durable_root).iterdir()} == {runner.EVIDENCE_NAME}


def test_phase_c_staged_byte_reread_mismatch_fails_before_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _prepare_phase_c_attempt(tmp_path)
    original_reread = runner._reread_staged_bytes

    def mismatch(path: Path) -> bytes:
        if path.name == runner.EVIDENCE_NAME:
            return b"mismatch"
        return original_reread(path)

    monkeypatch.setattr(runner, "_reread_staged_bytes", mismatch)
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_c(config)
    assert not runner.published_artifact_directory(config.durable_root).exists()
    assert runner.staging_artifact_directory(config.durable_root).is_dir()


def test_phase_c_second_invocation_cannot_overwrite_published_bundle(tmp_path: Path) -> None:
    config = _prepare_phase_c_attempt(tmp_path)
    assert runner.run_phase_c(config)["status"] == "PASS"
    candidate_before = runner.published_candidate_path(config.durable_root).read_bytes()
    evidence_before = runner.published_evidence_path(config.durable_root).read_bytes()
    with pytest.raises(runner.RunnerValidationError):
        runner.run_phase_c(config)
    assert runner.published_candidate_path(config.durable_root).read_bytes() == candidate_before
    assert runner.published_evidence_path(config.durable_root).read_bytes() == evidence_before


def test_runner_has_no_t0_or_ml_import_path() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "import lightgbm" not in source
    assert "import sklearn" not in source
    assert "pip install" not in source
    assert "payload" not in source.lower() or "payload_reads" in source
