from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import v10_environment_mutation_preflight_runner as runner
from scripts.v10_environment_extension_contract import (
    FROZEN_V10_DESIGN_SHA,
    PREDECESSOR_PACKAGE_SET,
    ContractValidationError,
    inspect_wheel_file,
    validate_mutation_preflight_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTENSION_SHA = "a" * 40
REVIEWED_RUNNER_COMMIT = "b" * 40
REVIEWED_RUNNER_BLOB = "c" * 40
EXPECTED_HEAD = "d" * 40


def _wheel_filename(name: str, version: str) -> str:
    return f"{name.replace('-', '_')}-{version}-py3-none-any.whl"


def _write_wheel(root: Path, name: str, version: str, *, metadata_name: str | None = None) -> Path:
    filename = _wheel_filename(name, version)
    path = root / filename
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    metadata_name = metadata_name or name
    metadata = f"Metadata-Version: 2.1\nName: {metadata_name}\nVersion: {version}\n\n".encode()
    wheel = b"Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel)
    return path


def _successor_packages() -> list[dict[str, str]]:
    packages = [{"name": name, "version": version} for name, version in PREDECESSOR_PACKAGE_SET]
    packages.extend({"name": name, "version": version} for name, version in runner.EXPECTED_DELTA_PACKAGES)
    packages.sort(key=lambda item: item["name"])
    return packages


def _fixture(tmp_path: Path) -> tuple[runner.PreflightConfig, dict[str, object], dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    packages = _successor_packages()
    manifest = []
    for package in packages:
        manifest.append(inspect_wheel_file(_write_wheel(wheelhouse, package["name"], package["version"])))
    candidate = {
        "schema_version": "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE_V2",
        "artifact_status": "WINDOWS_RESOLUTION_CANDIDATE_NOT_INSTALL_AUTHORITY",
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": EXTENSION_SHA,
        "reviewed_resolution_implementation_git_sha": runner.EXPECTED_REVIEWED_RESOLUTION_IMPLEMENTATION_SHA,
        "direct_spec_git_blob_sha1": runner.EXPECTED_DIRECT_SPEC_BLOB_SHA1,
        "direct_spec_sha256": runner.EXPECTED_DIRECT_SPEC_SHA256,
        "predecessor_lock_git_blob_sha1": runner.PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": runner.PREDECESSOR_LOCK_SHA256,
        "predecessor_package_count": 15,
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "resolution_policy_id": "PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1",
        "resolved_packages": packages,
        "resolved_package_count": 20,
        "resolved_wheels": manifest,
        "predecessor_pin_drift_count": 0,
        "pandas_market_calendars_version": "5.4.0",
        "exchange_calendars_version": "4.13.2",
    }
    candidate_bytes = runner.canonical_json_bytes(candidate)
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    evidence = {
        "schema_version": "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE_V1",
        "artifact_status": "WINDOWS_RESOLUTION_EVIDENCE",
        "status": "PASS",
        "failure_code": "NONE",
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": EXTENSION_SHA,
        "reviewed_resolution_implementation_git_sha": runner.EXPECTED_REVIEWED_RESOLUTION_IMPLEMENTATION_SHA,
        "direct_spec_git_blob_sha1": runner.EXPECTED_DIRECT_SPEC_BLOB_SHA1,
        "direct_spec_sha256": runner.EXPECTED_DIRECT_SPEC_SHA256,
        "predecessor_lock_git_blob_sha1": runner.PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": runner.PREDECESSOR_LOCK_SHA256,
        "resolution_policy_id": "PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1",
        "process_started": True,
        "process_exit_code": 0,
        "resolution_completed": True,
        "candidate_artifact_created": True,
        "successor_lock_candidate_sha256": candidate_sha,
        "resolved_package_count": 20,
        "package_index_id": "PYPI_OFFICIAL_SIMPLE",
        "package_resolution_process_invocations": 1,
        "human_authority_consumed": True,
        "package_installations": 0,
        "alternate_venv_created": False,
        "calendar_imports": 0,
        "calendar_dates_inspected": 0,
    }
    evidence_bytes = runner.canonical_json_bytes(evidence)
    authority_bytes = (REPO_ROOT / runner.MIGRATION_AUTHORITY_RELATIVE).read_bytes()
    lock_bytes = (REPO_ROOT / runner.LOCK_RELATIVE).read_bytes()
    config = runner.PreflightConfig(
        repo_root=REPO_ROOT,
        expected_current_head=EXPECTED_HEAD,
        expected_extension_design_sha=EXTENSION_SHA,
        expected_reviewed_runner_commit_sha=REVIEWED_RUNNER_COMMIT,
        expected_reviewed_runner_blob_sha1=REVIEWED_RUNNER_BLOB,
        wheelhouse=wheelhouse,
        output_root=tmp_path / "receipt-output",
        expected_candidate_sha256=candidate_sha,
        expected_candidate_blob_sha1=runner._git_blob_sha1(candidate_bytes),
        expected_evidence_blob_sha1=runner._git_blob_sha1(evidence_bytes),
    )
    observations: dict[str, object] = {
        "repository_identity": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "branch": runner.AUTHORITATIVE_BRANCH,
        "head": EXPECTED_HEAD,
        "clean": True,
        "frozen_design_commit_exists": True,
        "extension_design_commit_exists": True,
        "generic_authority_transition_reviewed_sha": runner.EXPECTED_GENERIC_AUTHORITY_TRANSITION_REVIEW_SHA,
        "generic_authority_transition_commit_exists": True,
        "reviewed_candidate_commit_exists": True,
        "reviewed_evidence_commit_exists": True,
        "reviewed_runner_commit_exists": True,
        "reviewed_runner_blob_sha1": REVIEWED_RUNNER_BLOB,
        "current_runner_blob_sha1": REVIEWED_RUNNER_BLOB,
        "candidate_git_blob_sha1": config.expected_candidate_blob_sha1,
        "evidence_git_blob_sha1": config.expected_evidence_blob_sha1,
        "current_head_candidate_git_blob_sha1": runner.CURRENT_HEAD_CANDIDATE_GIT_BLOB_SHA1,
        "current_head_evidence_git_blob_sha1": runner.CURRENT_HEAD_EVIDENCE_GIT_BLOB_SHA1,
        "migration_authority_git_blob_sha1": config.expected_migration_authority_blob_sha1,
        "candidate_bytes": candidate_bytes,
        "evidence_bytes": evidence_bytes,
        "migration_authority_bytes": authority_bytes,
        "candidate_bytes_git_blob_sha1": config.expected_candidate_blob_sha1,
        "evidence_bytes_git_blob_sha1": config.expected_evidence_blob_sha1,
        "migration_authority_bytes_git_blob_sha1": config.expected_migration_authority_blob_sha1,
        "generic_lock_git_blob_sha1": config.expected_generic_lock_blob_sha1,
        "generic_lock_committed_bytes": lock_bytes,
        "generic_lock_worktree_bytes": lock_bytes,
        "interpreter_executable": str(config.canonical_interpreter),
        "python_implementation": "CPython",
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "pip_version": "25.0.1",
        "live_packages": [{"name": name, "version": version} for name, version in PREDECESSOR_PACKAGE_SET],
        "predecessor_lock_git_blob_sha1": runner.PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": runner.PREDECESSOR_LOCK_SHA256,
        "predecessor_lock_package_count": 15,
    }
    return config, observations, candidate


def _run(config: runner.PreflightConfig, observations: dict[str, object]) -> dict[str, object]:
    return runner.run_preflight(config, observations)


def test_exact_canonical_pass_derives_five_delta_paths_and_argv_without_launch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _ = _fixture(tmp_path)

    def fail_if_launched(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("preflight must not launch a process")

    monkeypatch.setattr(runner.subprocess, "run", fail_if_launched)
    result = _run(config, observations)
    assert result["receipt"]["status"] == "PASS"
    assert result["receipt"]["failure_code"] == "NONE"
    assert result["receipt"]["delta_wheel_count"] == 5
    assert result["receipt"]["mutation_authority_consumed"] is False
    assert result["receipt"]["mutation_started"] is False
    assert [path.name for path in result["wheel_result"]["delta_wheel_paths"]] == [
        "exchange_calendars-4.13.2-py3-none-any.whl",
        "korean_lunar_calendar-0.4.0-py3-none-any.whl",
        "pandas_market_calendars-5.4.0-py3-none-any.whl",
        "pyluach-2.3.0-py3-none-any.whl",
        "toolz-1.1.0-py3-none-any.whl",
    ]
    assert result["install_argv"] == [
        str(config.canonical_interpreter), "-m", "pip", "install", "--no-deps", "--no-index",
        *[str(path) for path in result["wheel_result"]["delta_wheel_paths"]],
    ]


def test_pass_receipt_publishes_and_validates_through_production_validator(tmp_path: Path) -> None:
    config, observations, _ = _fixture(tmp_path)
    result = runner.run_preflight(config, observations, publish=True)
    receipt_path = config.output_root / runner.RECEIPT_NAME
    assert receipt_path.exists()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == result["receipt"]
    validate_mutation_preflight_receipt(
        result["receipt"],
        expected_extension_design_sha=config.expected_extension_design_sha,
        expected_successor_lock_candidate_sha256=config.expected_candidate_sha256,
        expected_migration_authority_git_blob_sha1=config.expected_migration_authority_blob_sha1,
        expected_generic_lock_git_blob_sha1=config.expected_generic_lock_blob_sha1,
    )


@pytest.mark.parametrize("field", ["head", "clean", "candidate_git_blob_sha1", "evidence_git_blob_sha1", "migration_authority_git_blob_sha1", "generic_authority_transition_reviewed_sha"])
def test_provenance_failures_precede_all_later_checks(tmp_path: Path, field: str) -> None:
    config, observations, _ = _fixture(tmp_path)
    if field == "clean":
        observations[field] = False
    elif field == "head":
        observations[field] = "0" * 40
    elif field == "generic_authority_transition_reviewed_sha":
        observations[field] = "0" * 40
    else:
        observations[field] = "0" * (64 if field == "candidate_git_blob_sha1" and False else 40)
    observations["live_packages"] = []
    observations["generic_lock_worktree_bytes"] = b"broken"
    (config.wheelhouse / "exchange_calendars-4.13.2-py3-none-any.whl").unlink()
    result = _run(config, observations)
    assert result["receipt"]["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert result["receipt"]["wheelhouse_integrity_verified"] is None
    assert result["receipt"]["delta_wheel_count"] is None


@pytest.mark.parametrize("mutation", ["wrong", "missing", "extra"])
def test_provenance_rejects_migration_authority_content_drift(tmp_path: Path, mutation: str) -> None:
    config, observations, _ = _fixture(tmp_path)
    authority = json.loads(observations["migration_authority_bytes"].decode("utf-8"))
    if mutation == "wrong":
        authority["canonical_environment_state"] = "V10_SUCCESSOR_MIGRATION_IN_PROGRESS_AUTHORIZED"
    elif mutation == "missing":
        del authority["new_generic_lock_sha256"]
    else:
        authority["unexpected"] = True
    observations["migration_authority_bytes"] = runner.canonical_json_bytes(authority)
    observations["migration_authority_bytes_git_blob_sha1"] = runner._git_blob_sha1(observations["migration_authority_bytes"])
    result = _run(config, observations)
    assert result["receipt"]["failure_code"] == "PROVENANCE_BINDING_FAILURE"


@pytest.mark.parametrize("mutation", ["missing", "extra", "version", "equivalent", "collision", "python", "platform", "pip"])
def test_predecessor_baseline_is_exact_but_accepts_normalized_names(tmp_path: Path, mutation: str) -> None:
    config, observations, _ = _fixture(tmp_path)
    packages = copy.deepcopy(observations["live_packages"])
    if mutation == "missing":
        packages.pop()
    elif mutation == "extra":
        packages.append({"name": "extra-package", "version": "1.0"})
    elif mutation == "version":
        packages[0]["version"] = "99.0"
    elif mutation == "equivalent":
        packages[1]["name"] = "CHARSET_normalizer"
    elif mutation == "collision":
        packages[1]["name"] = packages[0]["name"].replace("-", "_")
    elif mutation == "python":
        observations["python_version"] = "3.12.9"
    elif mutation == "platform":
        observations["platform_machine"] = "x86_64"
    else:
        observations["pip_version"] = "25.0.0"
    observations["live_packages"] = packages
    result = _run(config, observations)
    if mutation == "equivalent":
        assert result["receipt"]["status"] == "PASS"
    else:
        assert result["receipt"]["failure_code"] == "PREDECESSOR_LIVE_BASELINE_MISMATCH"
        assert result["receipt"]["wheelhouse_integrity_verified"] is None


@pytest.mark.parametrize("mutation", ["blob", "hash", "package", "missing", "extra", "eol"])
def test_successor_lock_mismatch_precedes_wheelhouse(tmp_path: Path, mutation: str) -> None:
    config, observations, candidate = _fixture(tmp_path)
    if mutation == "blob":
        observations["generic_lock_git_blob_sha1"] = "0" * 40
    elif mutation == "hash":
        observations["generic_lock_committed_bytes"] = b"cffi==2.1.1\n"
        observations["generic_lock_worktree_bytes"] = b"cffi==2.1.1\n"
    elif mutation == "package":
        raw = observations["generic_lock_committed_bytes"].replace(b"toolz==1.1.0", b"toolz==9.9.9")
        observations["generic_lock_committed_bytes"] = raw
        observations["generic_lock_worktree_bytes"] = raw
    elif mutation == "missing":
        raw = observations["generic_lock_committed_bytes"].replace(b"toolz==1.1.0\n", b"")
        observations["generic_lock_committed_bytes"] = raw
        observations["generic_lock_worktree_bytes"] = raw
    elif mutation == "extra":
        raw = observations["generic_lock_committed_bytes"] + b"extra-package==1.0\n"
        observations["generic_lock_committed_bytes"] = raw
        observations["generic_lock_worktree_bytes"] = raw
    else:
        observations["generic_lock_worktree_bytes"] = observations["generic_lock_committed_bytes"].replace(b"\n", b"\r\n")
    result = _run(config, observations)
    expected = "PROVENANCE_BINDING_FAILURE" if mutation == "blob" else "GENERIC_SUCCESSOR_LOCK_MISMATCH"
    assert result["receipt"]["failure_code"] == expected
    assert result["receipt"]["wheelhouse_integrity_verified"] is None
    assert result["receipt"]["delta_wheel_count"] is None
    assert candidate["resolved_package_count"] == 20


@pytest.mark.parametrize("mutation", ["missing", "extra", "tamper", "malformed", "metadata", "manifest", "wrong_delta"])
def test_wheelhouse_integrity_failures_are_fourth_precedence(tmp_path: Path, mutation: str) -> None:
    config, observations, candidate = _fixture(tmp_path)
    target = config.wheelhouse / "exchange_calendars-4.13.2-py3-none-any.whl"
    if mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        _write_wheel(config.wheelhouse, "extra-package", "1.0")
    elif mutation == "tamper":
        target.write_bytes(b"tampered")
    elif mutation == "malformed":
        target.write_bytes(b"not-a-zip")
    elif mutation == "metadata":
        target.unlink()
        _write_wheel(config.wheelhouse, "exchange-calendars", "4.13.2", metadata_name="other-name")
    elif mutation == "manifest":
        target.unlink()
        _write_wheel(config.wheelhouse, "exchange-calendars", "4.13.2", metadata_name="other-name")
    else:
        target.unlink()
        _write_wheel(config.wheelhouse, "exchange-calendars", "9.9.9")
    result = _run(config, observations)
    assert result["receipt"]["failure_code"] == "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE"
    assert result["receipt"]["wheelhouse_integrity_verified"] is False
    assert result["receipt"]["delta_wheel_count"] == 5


@pytest.mark.parametrize("mutations,expected", [
    (("provenance", "predecessor", "lock", "wheel"), "PROVENANCE_BINDING_FAILURE"),
    (("predecessor", "lock", "wheel"), "PREDECESSOR_LIVE_BASELINE_MISMATCH"),
    (("lock", "wheel"), "GENERIC_SUCCESSOR_LOCK_MISMATCH"),
])
def test_failure_precedence_is_not_populated_by_later_checks(tmp_path: Path, mutations: tuple[str, ...], expected: str) -> None:
    config, observations, _ = _fixture(tmp_path)
    if "provenance" in mutations:
        observations["head"] = "0" * 40
    if "predecessor" in mutations:
        observations["live_packages"] = []
    if "lock" in mutations:
        observations["generic_lock_worktree_bytes"] = b"changed"
    if "wheel" in mutations:
        (config.wheelhouse / "exchange_calendars-4.13.2-py3-none-any.whl").unlink()
    result = _run(config, observations)
    assert result["receipt"]["failure_code"] == expected
    assert result["receipt"]["wheelhouse_integrity_verified"] is None
    assert result["receipt"]["delta_wheel_count"] is None


def test_output_collision_and_failed_publication_never_overwrite_or_publish_false_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, observations, _ = _fixture(tmp_path)
    config.output_root.mkdir()
    with pytest.raises(ContractValidationError, match="DURABLE_OUTPUT_ROOT_SAFETY_FAILURE"):
        runner.run_preflight(config, observations, publish=True)
    assert not (config.output_root / runner.RECEIPT_NAME).exists()

    config2, observations2, _ = _fixture(tmp_path / "second")
    original = runner._atomic_create_no_overwrite

    def fail_publication(path: Path, raw: bytes) -> None:
        raise runner.PreflightValidationError("injected publication failure")

    monkeypatch.setattr(runner, "_atomic_create_no_overwrite", fail_publication)
    with pytest.raises(ContractValidationError, match="injected publication failure"):
        runner.run_preflight(config2, observations2, publish=True)
    assert not (config2.output_root / runner.RECEIPT_NAME).exists()
    monkeypatch.setattr(runner, "_atomic_create_no_overwrite", original)


def test_symlink_wheelhouse_is_rejected_as_filesystem_safety_when_supported(tmp_path: Path) -> None:
    config, observations, _ = _fixture(tmp_path)
    target = config.wheelhouse
    replacement = tmp_path / "real-wheelhouse"
    target.rename(replacement)
    try:
        target.symlink_to(replacement, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ContractValidationError, match="WHEELHOUSE_FILESYSTEM_SAFETY_FAILURE"):
        _run(config, observations)


def _cli_config(config: runner.PreflightConfig, wheelhouse: Path, output_root: Path) -> runner.PreflightConfig:
    parser = runner._build_parser()
    args = parser.parse_args([
        "--repo-root", str(config.repo_root),
        "--expected-current-head", config.expected_current_head,
        "--expected-extension-design-sha", config.expected_extension_design_sha,
        "--expected-reviewed-runner-commit-sha", config.expected_reviewed_runner_commit_sha,
        "--expected-reviewed-runner-blob-sha1", config.expected_reviewed_runner_blob_sha1,
        "--wheelhouse", str(wheelhouse),
        "--output-root", str(output_root),
    ])
    return runner._config_from_args(args)


def _make_directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlink/reparse creation unavailable")


@pytest.mark.parametrize("wheelhouse_kind", ["ancestor", "itself"])
def test_cli_wheelhouse_lexical_reparse_paths_stop_and_match_direct_config(
    tmp_path: Path,
    wheelhouse_kind: str,
) -> None:
    config, observations, _ = _fixture(tmp_path / "fixture")
    real_root = tmp_path / "real-root"
    real_root.mkdir()
    link_root = tmp_path / "link-root"
    _make_directory_symlink(link_root, real_root)
    if wheelhouse_kind == "ancestor":
        supplied = link_root / "nested" / "wheelhouse"
    else:
        supplied = link_root
    cli = _cli_config(config, supplied, tmp_path / "safe-output")
    direct = replace(config, wheelhouse=supplied, output_root=tmp_path / "safe-output")
    assert cli.wheelhouse == supplied
    assert cli.wheelhouse == direct.wheelhouse
    assert runner._wheelhouse_path_is_safe(cli.wheelhouse) is False
    assert runner._wheelhouse_path_is_safe(direct.wheelhouse) is False
    with pytest.raises(ContractValidationError, match="WHEELHOUSE_FILESYSTEM_SAFETY_FAILURE"):
        _run(cli, observations)


def test_cli_output_root_lexical_reparse_ancestor_stops_before_publication(tmp_path: Path) -> None:
    config, observations, _ = _fixture(tmp_path / "fixture")
    real_root = tmp_path / "real-output-root"
    real_root.mkdir()
    link_root = tmp_path / "link-output-root"
    _make_directory_symlink(link_root, real_root)
    supplied = link_root / "receipt"
    cli = _cli_config(config, config.wheelhouse, supplied)
    direct = replace(config, output_root=supplied)
    result = _run(config, observations)
    for candidate in (cli, direct):
        with pytest.raises(ContractValidationError, match="DURABLE_OUTPUT_ROOT_SAFETY_FAILURE"):
            runner.publish_receipt(candidate, result["receipt"])


def test_cli_safe_ordinary_absolute_paths_remain_usable(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path / "fixture")
    wheelhouse = tmp_path / "ordinary-wheelhouse"
    wheelhouse.mkdir()
    output_root = tmp_path / "ordinary-output"
    cli = _cli_config(config, wheelhouse, output_root)
    assert cli.wheelhouse == wheelhouse
    assert cli.output_root == output_root
    assert runner._wheelhouse_path_is_safe(cli.wheelhouse)
    assert runner.validate_durable_root(
        cli.output_root,
        repo_root=config.repo_root,
        protected_environment=config.canonical_environment,
        governed_roots=(cli.wheelhouse,),
    ) == "DURABLE_ROOT_OK"


@pytest.mark.parametrize("field", [
    "head",
    "clean",
    "current_runner_blob_sha1",
    "migration_authority_git_blob_sha1",
])
def test_default_stage_p_failure_launches_no_canonical_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    config, observations, _ = _fixture(tmp_path)
    bad = copy.deepcopy(observations)
    bad[field] = False if field == "clean" else "0" * 40
    launch_count = 0

    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _config: bad)

    def probe(_config: runner.PreflightConfig) -> dict[str, object]:
        nonlocal launch_count
        launch_count += 1
        raise AssertionError("canonical interpreter must not launch after provenance failure")

    monkeypatch.setattr(runner, "_probe_canonical_environment", probe)
    result = runner.run_preflight(config)
    assert result["receipt"]["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert launch_count == 0


def test_default_valid_provenance_reaches_only_expected_predecessor_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, observations, _ = _fixture(tmp_path)
    launch_count = 0
    predecessor_keys = {
        "interpreter_executable",
        "python_implementation",
        "python_version",
        "platform_system",
        "platform_machine",
        "sysconfig_platform",
        "pip_version",
        "live_packages",
        "predecessor_lock_git_blob_sha1",
        "predecessor_lock_sha256",
        "predecessor_lock_package_count",
    }
    predecessor = {key: observations[key] for key in predecessor_keys}
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _config: copy.deepcopy(observations))

    def probe(_config: runner.PreflightConfig) -> dict[str, object]:
        nonlocal launch_count
        launch_count += 1
        return predecessor

    monkeypatch.setattr(runner, "_probe_canonical_environment", probe)
    result = runner.run_preflight(config)
    assert result["receipt"]["status"] == "PASS"
    assert launch_count == 1


@pytest.mark.parametrize("current_field", [
    "current_head_candidate_git_blob_sha1",
    "current_head_evidence_git_blob_sha1",
])
def test_exact_current_head_artifact_blob_binding_allows_provenance_to_proceed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_field: str,
) -> None:
    config, observations, _ = _fixture(tmp_path)
    predecessor_keys = {
        "interpreter_executable",
        "python_implementation",
        "python_version",
        "platform_system",
        "platform_machine",
        "sysconfig_platform",
        "pip_version",
        "live_packages",
        "predecessor_lock_git_blob_sha1",
        "predecessor_lock_sha256",
        "predecessor_lock_package_count",
    }
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _config: copy.deepcopy(observations))
    monkeypatch.setattr(
        runner,
        "_default_predecessor_observations",
        lambda _config: {key: observations[key] for key in predecessor_keys},
    )
    result = runner.run_preflight(config)
    assert result["receipt"]["failure_code"] != "PROVENANCE_BINDING_FAILURE"
    assert observations[current_field] == (
        runner.CURRENT_HEAD_CANDIDATE_GIT_BLOB_SHA1
        if "candidate" in current_field
        else runner.CURRENT_HEAD_EVIDENCE_GIT_BLOB_SHA1
    )


@pytest.mark.parametrize("current_field", [
    "current_head_candidate_git_blob_sha1",
    "current_head_evidence_git_blob_sha1",
])
@pytest.mark.parametrize("mutation", ["wrong", "missing"])
def test_current_head_artifact_drift_fails_before_canonical_process_or_wheelhouse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_field: str,
    mutation: str,
) -> None:
    config, observations, _ = _fixture(tmp_path)
    bad = copy.deepcopy(observations)
    if mutation == "wrong":
        bad[current_field] = "0" * 40
    else:
        del bad[current_field]
    process_count = 0
    wheelhouse_count = 0
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _config: bad)

    def probe(_config: runner.PreflightConfig) -> dict[str, object]:
        nonlocal process_count
        process_count += 1
        raise AssertionError("canonical process must not launch after current-artifact drift")

    def inspect_wheelhouse(*_args: object, **_kwargs: object) -> object:
        nonlocal wheelhouse_count
        wheelhouse_count += 1
        raise AssertionError("wheelhouse must not be inspected after current-artifact drift")

    monkeypatch.setattr(runner, "_probe_canonical_environment", probe)
    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", inspect_wheelhouse)
    result = runner.run_preflight(config)
    assert result["receipt"]["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert process_count == 0
    assert wheelhouse_count == 0


def test_current_head_artifact_drift_fails_even_when_historical_bindings_remain_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, observations, _ = _fixture(tmp_path)
    bad = copy.deepcopy(observations)
    bad["current_head_candidate_git_blob_sha1"] = "0" * 40
    assert bad["candidate_git_blob_sha1"] == config.expected_candidate_blob_sha1
    assert bad["evidence_git_blob_sha1"] == config.expected_evidence_blob_sha1
    process_count = 0
    wheelhouse_count = 0
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _config: bad)

    def probe(_config: runner.PreflightConfig) -> dict[str, object]:
        nonlocal process_count
        process_count += 1
        raise AssertionError("canonical process must not launch after current-artifact drift")

    def inspect_wheelhouse(*_args: object, **_kwargs: object) -> object:
        nonlocal wheelhouse_count
        wheelhouse_count += 1
        raise AssertionError("wheelhouse must not be inspected after current-artifact drift")

    monkeypatch.setattr(runner, "_probe_canonical_environment", probe)
    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", inspect_wheelhouse)
    result = runner.run_preflight(config)
    assert result["receipt"]["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert process_count == 0
    assert wheelhouse_count == 0


@pytest.mark.parametrize("stage", ["provenance", "predecessor", "successor"])
def test_wheelhouse_verification_is_not_called_before_prior_stage_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    config, observations, _ = _fixture(tmp_path)
    if stage == "provenance":
        observations["head"] = "0" * 40
    elif stage == "predecessor":
        observations["live_packages"] = []
    else:
        observations["generic_lock_worktree_bytes"] = b"changed"
    calls = 0

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("wheelhouse verification called before its stage")

    monkeypatch.setattr(runner, "verify_reviewed_wheelhouse", fail_if_called)
    result = _run(config, observations)
    assert result["receipt"]["failure_code"] != "NONE"
    assert calls == 0
