from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import v10c_t0_ml_environment_offline_readjudication_runner as runner
from scripts import v10c_t0_ml_environment_resolution_runner as source_runner
from scripts.v10c_t0_ml_environment_contract import PREDECESSOR_PACKAGE_SET, canonical_json_bytes, git_blob_sha1


def _config(tmp_path: Path) -> runner.OfflineReadjudicationConfig:
    return runner.OfflineReadjudicationConfig(
        repo_root=tmp_path / "repo",
        expected_current_head="a" * 40,
        expected_reviewed_readjudication_runner_sha="b" * 40,
        expected_readjudication_runner_blob_sha1="c" * 40,
        expected_contract_blob_sha1="d" * 40,
        expected_source_provenance_blob_sha1="e" * 40,
        source_attempt_root=tmp_path / "source",
        output_root=tmp_path / "output",
    )


def _repository_probe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[runner.OfflineReadjudicationConfig, dict[str, str]]:
    current_head = "a" * 40
    runner_blob = git_blob_sha1(Path(runner.__file__).read_bytes())
    contract_path = Path(runner.__file__).resolve().parents[1] / "scripts" / "v10c_t0_ml_environment_contract.py"
    contract_blob = git_blob_sha1(contract_path.read_bytes())
    provenance_path = Path(runner.__file__).resolve().parents[1] / runner.SOURCE_PROVENANCE_NAME
    provenance_blob = git_blob_sha1(provenance_path.read_bytes())
    config = runner.OfflineReadjudicationConfig(
        repo_root=tmp_path / "repo",
        expected_current_head=current_head,
        expected_reviewed_readjudication_runner_sha=current_head,
        expected_readjudication_runner_blob_sha1=runner_blob,
        expected_contract_blob_sha1=contract_blob,
        expected_source_provenance_blob_sha1=provenance_blob,
        source_attempt_root=tmp_path / "source",
        output_root=tmp_path / "output",
    )
    facts = {
        "current_runner": runner_blob,
        "current_contract": contract_blob,
        "current_provenance": provenance_blob,
        "reviewed_runner": runner_blob,
        "reviewed_contract": contract_blob,
        "reviewed_provenance": provenance_blob,
    }

    def fake_git_output(repo_root: Path, args: tuple[str, ...] | list[str]) -> bytes:
        if args[:3] == ["config", "--get", "remote.origin.url"]:
            return b"https://github.com/ta1k1-arakawa/stock-analyzer.git\n"
        if args[:2] == ["branch", "--show-current"]:
            return (runner.AUTHORITATIVE_BRANCH + "\n").encode()
        if args[0:2] == ["rev-parse", f"refs/remotes/origin/{runner.AUTHORITATIVE_BRANCH}"]:
            return (current_head + "\n").encode()
        if args[0:2] == ["rev-parse", "HEAD"]:
            return (current_head + "\n").encode()
        if args[:2] == ["status", "--porcelain"]:
            return b""
        if args[:2] == ["rev-parse", "HEAD:"]:
            raise AssertionError(args)
        if args[0] == "rev-parse" and ":" in args[1]:
            revision, path = args[1].split(":", 1)
            prefix = "current" if revision == "HEAD" else "reviewed"
            if path == runner.READJUDICATION_RUNNER_RELATIVE.as_posix():
                return (facts[f"{prefix}_runner"] + "\n").encode()
            if path == runner.CONTRACT_RELATIVE.as_posix():
                return (facts[f"{prefix}_contract"] + "\n").encode()
            if path == runner.SOURCE_PROVENANCE_NAME:
                return (facts[f"{prefix}_provenance"] + "\n").encode()
            raise AssertionError((revision, path))
        raise AssertionError(args)

    monkeypatch.setattr(runner, "_git_output", fake_git_output)
    return config, facts


def test_validate_repository_production_path_checks_self_bytes_and_reviewed_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config, _ = _repository_probe(monkeypatch, tmp_path)
    runner._validate_repository(config)


def test_reviewed_sha_must_equal_current_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config, _ = _repository_probe(monkeypatch, tmp_path)
    config = runner.OfflineReadjudicationConfig(**{**config.__dict__, "expected_reviewed_readjudication_runner_sha": "b" * 40})
    with pytest.raises(runner.OfflineReadjudicationError, match="SOURCE_PROVENANCE_MISMATCH"):
        runner._validate_repository(config)


@pytest.mark.parametrize("mismatch", ["runner", "contract", "provenance"])
def test_reviewed_sha_objects_must_match_expected_blobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mismatch: str) -> None:
    config, facts = _repository_probe(monkeypatch, tmp_path)
    facts[f"reviewed_{mismatch}"] = "0" * 40
    with pytest.raises(runner.OfflineReadjudicationError, match="SOURCE_PROVENANCE_MISMATCH"):
        runner._validate_repository(config)


@pytest.mark.parametrize("mismatch", ["runner", "contract", "provenance"])
def test_current_head_objects_must_match_expected_blobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mismatch: str) -> None:
    config, facts = _repository_probe(monkeypatch, tmp_path)
    facts[f"current_{mismatch}"] = "0" * 40
    with pytest.raises(runner.OfflineReadjudicationError, match="SOURCE_PROVENANCE_MISMATCH"):
        runner._validate_repository(config)


def test_working_runner_bytes_must_match_expected_blob(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config, _ = _repository_probe(monkeypatch, tmp_path)
    working_copy = tmp_path / "working-runner.py"
    working_copy.write_bytes(b"different runner bytes")
    monkeypatch.setattr(runner, "__file__", str(working_copy))
    with pytest.raises(runner.OfflineReadjudicationError, match="SOURCE_PROVENANCE_MISMATCH"):
        runner._validate_repository(config)


def _wheels() -> tuple[dict[str, object], ...]:
    packages = sorted(
        [*PREDECESSOR_PACKAGE_SET, ("lightgbm", "4.6.0"), ("scikit-learn", "1.9.0")]
    )
    return tuple(
        {
            "name": name,
            "version": version,
            "filename": f"{name.replace('-', '_')}-{version}-py3-none-any.whl",
            "sha256": "a" * 64,
            "requires_dist": [],
            "requires_python": None,
        }
        for name, version in packages
    )


def _patch_validated_inputs(monkeypatch: pytest.MonkeyPatch, wheels: tuple[dict[str, object], ...]) -> None:
    monkeypatch.setattr(runner, "_validate_repository", lambda config: None)
    monkeypatch.setattr(runner, "_validate_provenance", lambda config: runner.SOURCE_PROVENANCE_EXPECTED)
    monkeypatch.setattr(runner, "_validate_source_inputs", lambda config, provenance: (wheels, {}))


def _published_names(root: Path) -> set[str]:
    return {entry.name for entry in runner.published_artifact_directory(root).iterdir()}


def test_source_provenance_exact_content_passes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    raw = canonical_json_bytes(runner.SOURCE_PROVENANCE_EXPECTED)
    config.repo_root.mkdir()
    (config.repo_root / runner.SOURCE_PROVENANCE_NAME).write_bytes(raw)
    checked = runner._validate_provenance(
        runner.OfflineReadjudicationConfig(
            **{**config.__dict__, "expected_source_provenance_blob_sha1": git_blob_sha1(raw)}
        )
    )
    assert checked == runner.SOURCE_PROVENANCE_EXPECTED


def test_source_provenance_blob_mismatch_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.repo_root.mkdir()
    (config.repo_root / runner.SOURCE_PROVENANCE_NAME).write_bytes(canonical_json_bytes(runner.SOURCE_PROVENANCE_EXPECTED))
    with pytest.raises(runner.OfflineReadjudicationError, match="SOURCE_PROVENANCE_MISMATCH"):
        runner._validate_provenance(config)


def _small_manifest_fixture(tmp_path: Path) -> tuple[Path, tuple[dict[str, object], ...], str]:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir(parents=True)
    wheels = []
    for name, version, content in (("alpha", "1.0", b"one"), ("beta", "2.0", b"two")):
        filename = f"{name}-{version}-py3-none-any.whl"
        (wheelhouse / filename).write_bytes(content)
        wheels.append({"name": name, "version": version, "filename": filename, "sha256": "a" * 64})
    manifest = [
        {**wheel, "size_bytes": (wheelhouse / wheel["filename"]).stat().st_size}
        for wheel in sorted(wheels, key=lambda item: (item["name"], item["version"], item["filename"].casefold()))
    ]
    return wheelhouse, tuple(wheels), hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


def test_wheel_manifest_count_total_and_hash_are_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wheelhouse, wheels, manifest_sha = _small_manifest_fixture(tmp_path)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_COUNT", 2)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_TOTAL_BYTES", 6)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_MANIFEST_SHA256", manifest_sha)
    runner._validate_wheelhouse_manifest(wheelhouse, wheels)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_COUNT", 3)
    with pytest.raises(runner.OfflineReadjudicationError, match="OFFLINE_WHEEL_VALIDATION_FAILURE"):
        runner._validate_wheelhouse_manifest(wheelhouse, wheels)


def test_wheel_manifest_sha_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wheelhouse, wheels, _ = _small_manifest_fixture(tmp_path)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_COUNT", 2)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_TOTAL_BYTES", 6)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(runner.OfflineReadjudicationError, match="OFFLINE_WHEEL_VALIDATION_FAILURE"):
        runner._validate_wheelhouse_manifest(wheelhouse, wheels)


def _source_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[runner.OfflineReadjudicationConfig, dict[str, object], tuple[dict[str, object], ...]]:
    config = _config(tmp_path)
    source = config.source_attempt_root
    source.mkdir()
    (source / source_runner.STATE_NAME).write_text("{}", encoding="utf-8")
    (source / source_runner.STDOUT_NAME).write_bytes(b"stdout")
    (source / source_runner.STDERR_NAME).write_bytes(b"stderr")
    artifact_directory = source_runner.published_artifact_directory(source)
    artifact_directory.mkdir()
    failure_evidence_path = artifact_directory / source_runner.EVIDENCE_NAME
    failure_evidence_path.write_text(json.dumps({"status": "FAIL", "failure_code": "RESOLUTION_REPORT_INVALID"}), encoding="utf-8")
    wheelhouse, wheels, manifest_sha = _small_manifest_fixture(tmp_path / "wheel-data")
    source_wheelhouse = source / source_runner.WHEELHOUSE_NAME
    source_wheelhouse.mkdir()
    for wheel in wheels:
        (source_wheelhouse / wheel["filename"]).write_bytes((wheelhouse / wheel["filename"]).read_bytes())
    monkeypatch.setattr(runner, "SOURCE_WHEEL_COUNT", 2)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_TOTAL_BYTES", 6)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(source_runner, "validate_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(source_runner, "_validate_attempt_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(source_runner, "inspect_wheelhouse", lambda root: ("NONE", wheels))
    provenance = dict(runner.SOURCE_PROVENANCE_EXPECTED)
    provenance.update(
        {
            "source_attempt_state_sha256": runner._sha256_file(source / source_runner.STATE_NAME),
            "source_stdout_sha256": runner._sha256_file(source / source_runner.STDOUT_NAME),
            "source_stderr_sha256": runner._sha256_file(source / source_runner.STDERR_NAME),
            "source_failure_evidence_sha256": runner._sha256_file(failure_evidence_path),
            "source_wheel_count": 2,
            "source_wheel_total_bytes": 6,
            "source_wheel_manifest_sha256": manifest_sha,
        }
    )
    return config, provenance, wheels


def test_source_attempt_and_failure_evidence_hashes_are_checked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config, provenance, wheels = _source_fixture(monkeypatch, tmp_path)
    runner._validate_source_inputs(config, provenance)
    for key in ("source_attempt_state_sha256", "source_failure_evidence_sha256"):
        broken = dict(provenance, **{key: "0" * 64})
        with pytest.raises(runner.OfflineReadjudicationError, match="SOURCE_PROVENANCE_MISMATCH"):
            runner._validate_source_inputs(config, broken)


def test_source_wheel_count_mismatch_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config, provenance, _ = _source_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "SOURCE_WHEEL_COUNT", 3)
    with pytest.raises(runner.OfflineReadjudicationError, match="OFFLINE_WHEEL_VALIDATION_FAILURE"):
        runner._validate_source_inputs(config, provenance)


def test_source_wheel_hash_inspection_failure_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config, provenance, _ = _source_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(source_runner, "inspect_wheelhouse", lambda root: ("WHEEL_MANIFEST_INVALID", None))
    with pytest.raises(runner.OfflineReadjudicationError, match="OFFLINE_WHEEL_VALIDATION_FAILURE"):
        runner._validate_source_inputs(config, provenance)


def test_pass_publishes_exact_candidate_and_evidence_and_binds_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    wheels = _wheels()
    _patch_validated_inputs(monkeypatch, wheels)
    result = runner.run_offline_readjudication(config)
    assert result["status"] == "PASS"
    assert _published_names(config.output_root) == {runner.READJUDICATION_CANDIDATE_NAME, runner.READJUDICATION_EVIDENCE_NAME}
    candidate = json.loads(runner.published_candidate_path(config.output_root).read_text(encoding="utf-8"))
    evidence = json.loads(runner.published_evidence_path(config.output_root).read_text(encoding="utf-8"))
    assert candidate["reviewed_resolution_implementation_git_sha"] == runner.SOURCE_RESOLUTION_HEAD
    assert evidence["readjudication_validator_git_sha"] == config.expected_reviewed_readjudication_runner_sha
    assert not (config.output_root / source_runner.CANDIDATE_NAME).exists()
    assert not (config.output_root / source_runner.EVIDENCE_NAME).exists()


def test_fail_publishes_evidence_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _patch_validated_inputs(monkeypatch, _wheels())
    monkeypatch.setattr(source_runner, "_build_candidate", lambda config, wheels: (_ for _ in ()).throw(source_runner.RunnerValidationError("RESOLUTION_REPORT_INVALID")))
    result = runner.run_offline_readjudication(config)
    assert result == {"status": "FAIL", "failure_code": "OFFLINE_WHEEL_VALIDATION_FAILURE", "candidate_artifact_created": False}
    assert _published_names(config.output_root) == {runner.READJUDICATION_EVIDENCE_NAME}


@pytest.mark.parametrize("failure", ["candidate", "evidence"])
def test_staging_failure_never_publishes_final_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str) -> None:
    output = tmp_path / "output"
    original = runner._write_bytes
    calls = 0

    def fail_write(path: Path, raw: bytes) -> None:
        nonlocal calls
        calls += 1
        if (failure == "candidate" and calls == 1) or (failure == "evidence" and calls == 2):
            raise OSError("injected staging failure")
        original(path, raw)

    monkeypatch.setattr(runner, "_write_bytes", fail_write)
    with pytest.raises(runner.OfflineReadjudicationError, match="ARTIFACT_PUBLICATION_FAILURE"):
        runner._publish_bundle(output, candidate_bytes=b"candidate", evidence_bytes=b"evidence")
    assert not runner.published_artifact_directory(output).exists()
    assert runner.staging_artifact_directory(output).is_dir()
    assert (runner.staging_artifact_directory(output) / runner.READJUDICATION_CANDIDATE_NAME).exists() is (failure == "evidence")


def test_rename_failure_preserves_staging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(runner.os, "rename", lambda source, target: (_ for _ in ()).throw(OSError("injected rename failure")))
    with pytest.raises(runner.OfflineReadjudicationError, match="ARTIFACT_PUBLICATION_FAILURE"):
        runner._publish_bundle(output, candidate_bytes=None, evidence_bytes=b"evidence")
    assert not runner.published_artifact_directory(output).exists()
    assert runner.staging_artifact_directory(output).is_dir()


def test_existing_output_or_staging_cannot_be_overwritten_or_resumed(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(runner.OfflineReadjudicationError, match="ARTIFACT_PUBLICATION_FAILURE"):
        runner._publish_bundle(output, candidate_bytes=None, evidence_bytes=b"evidence")
    second = tmp_path / "second"
    second.mkdir()
    runner.staging_artifact_directory(second).mkdir()
    with pytest.raises(runner.OfflineReadjudicationError, match="ARTIFACT_PUBLICATION_FAILURE"):
        runner._publish_bundle(second, candidate_bytes=None, evidence_bytes=b"evidence")


def test_staged_byte_reread_mismatch_fails_before_publish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(runner, "_reread_staged_bytes", lambda path: b"different")
    with pytest.raises(runner.OfflineReadjudicationError, match="ARTIFACT_PUBLICATION_FAILURE"):
        runner._publish_bundle(output, candidate_bytes=None, evidence_bytes=b"evidence")
    assert not runner.published_artifact_directory(output).exists()
    assert runner.staging_artifact_directory(output).is_dir()


def test_source_and_output_overlap_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = _config(tmp_path)
    config = runner.OfflineReadjudicationConfig(**{**config.__dict__, "source_attempt_root": source, "output_root": source / "child"})
    with pytest.raises(runner.OfflineReadjudicationError, match="SOURCE_PROVENANCE_MISMATCH"):
        runner._validate_source_inputs(config, runner.SOURCE_PROVENANCE_EXPECTED)


def test_symlink_source_or_output_is_rejected_where_supported(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    source_link = tmp_path / "source-link"
    output_link = tmp_path / "output-link"
    try:
        source_link.symlink_to(target, target_is_directory=True)
        output_link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")
    assert runner._is_reparse_or_symlink(source_link)
    assert runner._is_reparse_or_symlink(output_link)


def test_zero_resolution_and_install_activity_is_encoded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _patch_validated_inputs(monkeypatch, _wheels())
    result = runner.run_offline_readjudication(config)
    evidence = json.loads(runner.published_evidence_path(config.output_root).read_text(encoding="utf-8"))
    assert result["status"] == "PASS"
    assert all(evidence[key] == 0 for key in ("network_requests", "package_resolution_reruns", "package_installations", "environment_mutations", "model_fits", "t0_runs", "payload_reads"))
    assert evidence["human_authority_consumed"] is False


def test_second_invocation_cannot_overwrite_first_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _patch_validated_inputs(monkeypatch, _wheels())
    runner.run_offline_readjudication(config)
    with pytest.raises(runner.OfflineReadjudicationError, match="ARTIFACT_PUBLICATION_FAILURE"):
        runner.run_offline_readjudication(config)
