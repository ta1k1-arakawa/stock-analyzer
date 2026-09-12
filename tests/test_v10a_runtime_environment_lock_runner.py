from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from scripts import v10a_runtime_environment_lock_runner as runner


def _config(tmp_path: Path, *, output: Path | None = None) -> runner.RuntimeLockConfig:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    return runner.RuntimeLockConfig(
        repo_root=repo,
        reviewed_baseline_sha="1" * 40,
        expected_runner_blob_sha1="2" * 40,
        expected_test_blob_sha1="3" * 40,
        expected_design_blob_sha1="4" * 40,
        output_root=output or (tmp_path / "output"),
    )


def _provenance(config: runner.RuntimeLockConfig) -> dict[str, object]:
    return {
        "repository_identity": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "branch": runner.AUTHORITATIVE_BRANCH,
        "head": config.reviewed_baseline_sha,
        "clean": True,
        "design_blob": config.expected_design_blob_sha1,
        "current_runner_blob": config.expected_runner_blob_sha1,
        "current_test_blob": config.expected_test_blob_sha1,
        "approved_design_exists": True,
        "approved_design_blob": runner.APPROVED_DESIGN_BLOB_SHA1,
        "freeze_record_exists": True,
        "freeze_record_blob": runner.FREEZE_RECORD_BLOB_SHA1,
        "p5_p4_exists": True,
        "p5_bookkeeping_exists": True,
        "final_evidence_blob": runner.FINAL_EVIDENCE_BLOB_SHA1,
        "final_evidence_sha256": runner.FINAL_EVIDENCE_SHA256,
        "current_final_evidence_blob": runner.FINAL_EVIDENCE_BLOB_SHA1,
        "final_adjudication_blob": runner.P3_ADJUDICATION_BLOB_SHA1,
        "current_final_adjudication_blob": runner.P3_ADJUDICATION_BLOB_SHA1,
        "state": {
            "V10A_ENVIRONMENT_STATE": "CANONICAL_FROZEN",
            "V10A_CANONICAL_ENVIRONMENT_PROMOTED": "true",
            "V10A_ENVIRONMENT_FROZEN": "true",
            "V10A_EXECUTION_AUTHORIZED": "false",
            "V10A_CALENDAR_GENERATION_AUTHORIZED": "false",
            "V10A_T0_AUTHORIZED": "false",
            "V10A_HISTORICAL_EVALUATION_AUTHORIZED": "false",
        },
    }


def _packages() -> dict[str, object]:
    return {
        "runtime_distributions": list(runner.EXPECTED_PACKAGES),
        "runtime_distribution_count": 20,
    }


def _install_success_collectors(monkeypatch: pytest.MonkeyPatch, config: runner.RuntimeLockConfig) -> None:
    monkeypatch.setattr(runner, "_default_operation_observations", runner._default_operation_observations)
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _: _provenance(config))
    monkeypatch.setattr(
        runner,
        "_default_interpreter_observations",
        lambda _: {"executable": str(config.canonical_interpreter.resolve()), "expected_executable": str(config.canonical_interpreter.resolve()), "python_version": runner.PYTHON_VERSION},
    )
    monkeypatch.setattr(runner, "_default_package_observations", _packages)
    monkeypatch.setattr(
        runner,
        "_default_source_observations",
        lambda: {"calendar_source_blob": runner.CALENDAR_SOURCE_BLOB, "holiday_source_blob": runner.HOLIDAY_SOURCE_BLOB},
    )


def _assert_evidence_only_failure(config: runner.RuntimeLockConfig, expected_code: str) -> dict[str, object]:
    assert config.output_root.exists()
    assert sorted(path.name for path in config.output_root.iterdir()) == [runner.EVIDENCE_NAME]
    evidence = json.loads((config.output_root / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL"
    assert evidence["failure_code"] == expected_code
    runner.validate_evidence(evidence)
    return evidence


def _successful_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[runner.RuntimeLockConfig, dict[str, object]]:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    result = runner.run_lock(config)
    assert result["status"] == "PASS"
    return config, result


def test_normalize_distribution_name_exactly() -> None:
    assert runner.normalize_distribution_name("Pandas_Market.Calendars") == "pandas-market-calendars"
    assert runner.normalize_distribution_name("x--__..y") == "x-y"


@pytest.mark.parametrize(
    "packages",
    [
        [("foo", "1"), ("foo-bar", "1")],
        [("foo_bar", "1"), ("foo-bar", "2")],
    ],
)
def test_duplicate_normalized_names_reject(packages: list[tuple[str, str]]) -> None:
    assert runner.validate_package_observations({"runtime_distributions": packages, "runtime_distribution_count": len(packages)}) is False


def test_package_count_mismatch_rejects() -> None:
    observed = _packages()
    observed["runtime_distribution_count"] = 19
    assert runner.validate_package_observations(observed) is False


def test_exact_package_mapping_mismatch_rejects() -> None:
    observed = _packages()
    observed["runtime_distributions"] = list(runner.EXPECTED_PACKAGES[:-1]) + [("wrong", "1")]
    assert runner.validate_package_observations(observed) is False


def test_exact_package_mapping_is_twenty_entries() -> None:
    assert len(runner.EXPECTED_PACKAGES) == 20
    assert runner.validate_package_observations(_packages()) is True


def test_calendar_scalar_uses_inherited_underscore_and_runtime_entry_hyphen() -> None:
    lock = runner.build_runtime_lock(_packages())
    assert lock["calendar_distribution_name"] == "pandas_market_calendars"
    assert dict(runner.EXPECTED_PACKAGES)["pandas-market-calendars"] == "5.4.0"
    assert any(item["name"] == "pandas-market-calendars" for item in lock["runtime_distributions"])


def test_old_v10_jpx_blob_is_rejected() -> None:
    jpx_ok, holiday_ok = runner.validate_source_observations(
        {"calendar_source_blob": runner.OLD_V10_JPX_BLOB, "holiday_source_blob": runner.HOLIDAY_SOURCE_BLOB}
    )
    assert jpx_ok is False
    assert holiday_ok is True


def test_corrected_jpx_and_jp_blobs_are_accepted() -> None:
    assert runner.validate_source_observations({"calendar_source_blob": runner.CALENDAR_SOURCE_BLOB, "holiday_source_blob": runner.HOLIDAY_SOURCE_BLOB}) == (True, True)


def test_jp_blob_mismatch_rejects() -> None:
    assert runner.validate_source_observations({"calendar_source_blob": runner.CALENDAR_SOURCE_BLOB, "holiday_source_blob": "0" * 40}) == (True, False)


def test_git_blob_algorithm_uses_raw_bytes() -> None:
    assert runner.git_blob_sha1(b"abc") == "f2ba8f84ab5c1bce84a7b441cb1959cfc7093b7f"


def test_canonical_lock_has_exact_keys_and_one_final_lf() -> None:
    lock = runner.build_runtime_lock(_packages())
    raw = runner.canonical_json_bytes(lock)
    assert set(lock) == {
        "schema_version", "python_version", "calendar_distribution_name",
        "calendar_distribution_version", "calendar_name", "calendar_source_blob",
        "holiday_source_blob", "runtime_distributions", "runtime_distribution_count",
    }
    assert raw.endswith(b"\n")
    assert raw.count(b"\n") == 1
    assert raw == runner.canonical_json_bytes(json.loads(raw))
    assert "sha256" not in lock


def test_validate_runtime_lock_rejects_extra_key() -> None:
    lock = runner.build_runtime_lock(_packages())
    lock["self_hash"] = "forbidden"
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_runtime_lock(lock)


def test_validate_evidence_rejects_boolean_as_integer(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evidence = runner._base_evidence(config, "PROVENANCE_BINDING_FAILURE")
    evidence["network_requests"] = True
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(evidence)


def test_execution_evidence_v2_baseline_key_is_exact() -> None:
    assert runner.EVIDENCE_SCHEMA == "V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE_V2"
    assert "reviewed_baseline_sha" in runner.EVIDENCE_KEYS
    assert "expected_r2_reviewed_sha" not in runner.EVIDENCE_KEYS


def test_v2_evidence_requires_reviewed_baseline_sha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config, result = _successful_evidence(monkeypatch, tmp_path)
    assert result["reviewed_baseline_sha"] == config.reviewed_baseline_sha
    missing = dict(result)
    del missing["reviewed_baseline_sha"]
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(missing, config)
    legacy = dict(result)
    legacy["expected_r2_reviewed_sha"] = "f" * 40
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(legacy, config)


def test_reviewed_baseline_sha_is_validated_for_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert runner.validate_provenance(config, _provenance(config)) is True
    config_with_wrong_head = runner.RuntimeLockConfig(
        repo_root=config.repo_root,
        reviewed_baseline_sha="f" * 40,
        expected_runner_blob_sha1=config.expected_runner_blob_sha1,
        expected_test_blob_sha1=config.expected_test_blob_sha1,
        expected_design_blob_sha1=config.expected_design_blob_sha1,
        output_root=config.output_root,
    )
    assert runner.validate_provenance(config_with_wrong_head, _provenance(config)) is False


def test_provenance_failure_suppresses_all_live_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _: {})
    monkeypatch.setattr(runner, "_default_interpreter_observations", lambda _: calls.append("interpreter") or {})
    monkeypatch.setattr(runner, "_default_package_observations", lambda: calls.append("packages") or {})
    monkeypatch.setattr(runner, "_default_source_observations", lambda: calls.append("source") or {})
    result = runner.run_lock(config)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert calls == []


def test_design_blob_failure_suppresses_live_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    observed = _provenance(config)
    observed["design_blob"] = "f" * 40
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _: observed)
    monkeypatch.setattr(runner, "_default_interpreter_observations", lambda _: pytest.fail("live collection called"))
    assert runner.run_lock(config)["failure_code"] == "PROVENANCE_BINDING_FAILURE"


def test_runner_blob_failure_suppresses_live_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    observed = _provenance(config)
    observed["current_runner_blob"] = "f" * 40
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _: observed)
    monkeypatch.setattr(runner, "_default_interpreter_observations", lambda _: pytest.fail("live collection called"))
    assert runner.run_lock(config)["failure_code"] == "PROVENANCE_BINDING_FAILURE"


def test_test_blob_failure_suppresses_live_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    observed = _provenance(config)
    observed["current_test_blob"] = "f" * 40
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _: observed)
    monkeypatch.setattr(runner, "_default_interpreter_observations", lambda _: pytest.fail("live collection called"))
    assert runner.run_lock(config)["failure_code"] == "PROVENANCE_BINDING_FAILURE"


def test_final_evidence_or_adjudication_failure_suppresses_live_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    observed = _provenance(config)
    observed["final_evidence_blob"] = "f" * 40
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _: observed)
    monkeypatch.setattr(runner, "_default_interpreter_observations", lambda _: pytest.fail("live collection called"))
    assert runner.run_lock(config)["failure_code"] == "PROVENANCE_BINDING_FAILURE"


def test_wrong_canonical_interpreter_suppresses_package_and_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    calls: list[str] = []
    monkeypatch.setattr(runner, "_default_interpreter_observations", lambda _: {"executable": "wrong", "expected_executable": "right", "python_version": runner.PYTHON_VERSION})
    monkeypatch.setattr(runner, "_default_package_observations", lambda: calls.append("packages") or _packages())
    monkeypatch.setattr(runner, "_default_source_observations", lambda: calls.append("source") or {})
    result = runner.run_lock(config)
    assert result["failure_code"] == "WRONG_CANONICAL_INTERPRETER"
    assert calls == []


def test_python_mismatch_suppresses_package_and_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    calls: list[str] = []
    monkeypatch.setattr(runner, "_default_interpreter_observations", lambda _: {"executable": str(config.canonical_interpreter.resolve()), "expected_executable": str(config.canonical_interpreter.resolve()), "python_version": "3.11.0"})
    monkeypatch.setattr(runner, "_default_package_observations", lambda: calls.append("packages") or _packages())
    monkeypatch.setattr(runner, "_default_source_observations", lambda: calls.append("source") or {})
    result = runner.run_lock(config)
    assert result["failure_code"] == "PYTHON_VERSION_MISMATCH"
    assert calls == []


def test_package_failure_suppresses_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    calls: list[str] = []
    monkeypatch.setattr(runner, "_default_package_observations", lambda: {"runtime_distributions": [], "runtime_distribution_count": 0})
    monkeypatch.setattr(runner, "_default_source_observations", lambda: calls.append("source") or {})
    result = runner.run_lock(config)
    assert result["failure_code"] == "PACKAGE_SET_MISMATCH"
    assert calls == []


def test_source_failure_suppresses_pass_publication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    monkeypatch.setattr(runner, "_default_source_observations", lambda: {"calendar_source_blob": "f" * 40, "holiday_source_blob": runner.HOLIDAY_SOURCE_BLOB})
    result = runner.run_lock(config)
    assert result["failure_code"] == "CALENDAR_SOURCE_BLOB_MISMATCH"
    assert sorted(path.name for path in config.output_root.iterdir()) == [runner.EVIDENCE_NAME]
    evidence = json.loads((config.output_root / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL"
    assert evidence["failure_code"] == "CALENDAR_SOURCE_BLOB_MISMATCH"
    assert evidence["durable_lock_created"] is False
    assert evidence["runtime_lock_sha256"] is None
    assert evidence["runtime_lock_size"] is None
    runner.validate_evidence(evidence)


def test_wrong_interpreter_publishes_evidence_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    monkeypatch.setattr(runner, "_default_interpreter_observations", lambda _: {"executable": "wrong", "expected_executable": "right", "python_version": runner.PYTHON_VERSION})
    result = runner.run_lock(config)
    assert result["failure_code"] == "WRONG_CANONICAL_INTERPRETER"
    _assert_evidence_only_failure(config, "WRONG_CANONICAL_INTERPRETER")


def test_python_mismatch_publishes_evidence_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    expected = str(config.canonical_interpreter.resolve())
    monkeypatch.setattr(runner, "_default_interpreter_observations", lambda _: {"executable": expected, "expected_executable": expected, "python_version": "3.11.0"})
    result = runner.run_lock(config)
    assert result["failure_code"] == "PYTHON_VERSION_MISMATCH"
    _assert_evidence_only_failure(config, "PYTHON_VERSION_MISMATCH")


def test_package_mismatch_publishes_evidence_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    monkeypatch.setattr(runner, "_default_package_observations", lambda: {"runtime_distributions": [], "runtime_distribution_count": 0})
    result = runner.run_lock(config)
    assert result["failure_code"] == "PACKAGE_SET_MISMATCH"
    _assert_evidence_only_failure(config, "PACKAGE_SET_MISMATCH")


def test_holiday_source_mismatch_publishes_evidence_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    monkeypatch.setattr(runner, "_default_source_observations", lambda: {"calendar_source_blob": runner.CALENDAR_SOURCE_BLOB, "holiday_source_blob": "0" * 40})
    result = runner.run_lock(config)
    assert result["failure_code"] == "HOLIDAY_SOURCE_BLOB_MISMATCH"
    _assert_evidence_only_failure(config, "HOLIDAY_SOURCE_BLOB_MISMATCH")


def test_canonicalization_failure_publishes_evidence_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    monkeypatch.setattr(runner, "build_runtime_lock", lambda _: (_ for _ in ()).throw(runner.RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")))
    result = runner.run_lock(config)
    assert result["failure_code"] == "RUNTIME_LOCK_CANONICALIZATION_FAILURE"
    _assert_evidence_only_failure(config, "RUNTIME_LOCK_CANONICALIZATION_FAILURE")


def test_provenance_failure_writes_no_output_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _: {})
    result = runner.run_lock(config)
    assert result["failure_code"] == "PROVENANCE_BINDING_FAILURE"
    assert not config.output_root.exists()


def test_lock_write_failure_preserves_prepared_hash_and_writes_evidence_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    original = runner._exclusive_write
    calls: list[str] = []

    def fail_lock(path: Path, raw: bytes) -> None:
        calls.append(path.name)
        if path.name == runner.LOCK_NAME:
            raise OSError("synthetic lock write failure")
        original(path, raw)

    monkeypatch.setattr(runner, "_exclusive_write", fail_lock)
    result = runner.run_lock(config)
    assert result["failure_code"] == "DURABLE_WRITE_FAILURE"
    assert result["durable_lock_created"] is False
    assert isinstance(result["runtime_lock_sha256"], str)
    assert isinstance(result["runtime_lock_size"], int)
    assert calls == [runner.LOCK_NAME, runner.EVIDENCE_NAME]
    _assert_evidence_only_failure(config, "DURABLE_WRITE_FAILURE")
    evidence = json.loads((config.output_root / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    assert evidence["durable_lock_created"] is False
    assert evidence["runtime_lock_sha256"] == result["runtime_lock_sha256"]
    assert evidence["runtime_lock_size"] == result["runtime_lock_size"]


def test_evidence_write_failure_is_not_retried_and_preserves_partial_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    original = runner._exclusive_write
    calls: list[str] = []

    def fail_evidence(path: Path, raw: bytes) -> None:
        calls.append(path.name)
        if path.name == runner.EVIDENCE_NAME:
            raise OSError("synthetic evidence write failure")
        original(path, raw)

    monkeypatch.setattr(runner, "_exclusive_write", fail_evidence)
    result = runner.run_lock(config)
    assert result["failure_code"] == "DURABLE_WRITE_FAILURE"
    assert result["status"] == "FAIL"
    assert result["durable_lock_created"] is True
    assert calls == [runner.LOCK_NAME, runner.EVIDENCE_NAME]
    assert (config.output_root / runner.LOCK_NAME).exists()
    assert not (config.output_root / runner.EVIDENCE_NAME).exists()


def test_pass_evidence_is_validated_before_evidence_publication(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    original_validate = runner.validate_evidence
    original_write = runner._exclusive_write
    events: list[str] = []

    def record_validate(evidence: object, config_arg: runner.RuntimeLockConfig | None = None) -> None:
        events.append("validate")
        original_validate(evidence, config_arg)

    def record_write(path: Path, raw: bytes) -> None:
        if path.name == runner.EVIDENCE_NAME:
            assert events and events[-1] == "validate"
            events.append("evidence-write")
        original_write(path, raw)

    monkeypatch.setattr(runner, "validate_evidence", record_validate)
    monkeypatch.setattr(runner, "_exclusive_write", record_write)
    result = runner.run_lock(config)
    assert result["status"] == "PASS"
    assert events[-2:] == ["validate", "evidence-write"]


def test_output_collision_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    config = _config(tmp_path, output=output)
    _install_success_collectors(monkeypatch, config)
    result = runner.run_lock(config)
    assert result["failure_code"] == "DURABLE_OUTPUT_COLLISION"
    assert list(output.iterdir()) == []


def test_success_writes_exactly_lock_and_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    result = runner.run_lock(config)
    assert result["status"] == "PASS"
    assert result["failure_code"] == "NONE"
    assert sorted(path.name for path in config.output_root.iterdir()) == sorted([runner.LOCK_NAME, runner.EVIDENCE_NAME])
    lock = json.loads((config.output_root / runner.LOCK_NAME).read_text(encoding="utf-8"))
    evidence = json.loads((config.output_root / runner.EVIDENCE_NAME).read_text(encoding="utf-8"))
    runner.validate_runtime_lock(lock)
    runner.validate_evidence(evidence)
    assert evidence == result


def test_success_evidence_has_zero_counters_and_false_authority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    result = runner.run_lock(config)
    for key in ("network_requests", "package_installations", "environment_mutations", "calendar_imports", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_research_reads"):
        assert result[key] == 0
    for key in ("execution_authorized", "calendar_generation_authorized", "t0_authorized", "historical_evaluation_authorized", "future_profitability_established"):
        assert result[key] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("python_version", "3.11.0"),
        ("canonical_interpreter_verified", False),
        ("calendar_distribution_name", "pandas-market-calendars"),
        ("calendar_distribution_version", "5.3.0"),
        ("calendar_name", "NYSE"),
        ("runtime_distribution_count", 19),
        ("exact_package_mapping", False),
        ("calendar_source_blob", "0" * 40),
        ("holiday_source_blob", "0" * 40),
        ("t0_run", True),
        ("network_requests", 1),
        ("package_installations", 1),
        ("environment_mutations", 1),
        ("calendar_imports", 1),
        ("calendar_object_creations", 1),
        ("calendar_dates_inspected", 1),
        ("protected_or_private_research_reads", 1),
        ("execution_authorized", True),
        ("calendar_generation_authorized", True),
        ("t0_authorized", True),
        ("historical_evaluation_authorized", True),
        ("future_profitability_established", True),
    ],
)
def test_pass_evidence_rejects_each_frozen_semantic_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: object
) -> None:
    _, result = _successful_evidence(monkeypatch, tmp_path)
    mutated = dict(result)
    mutated[field] = value
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(mutated)


def test_pass_evidence_must_match_dynamic_config_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config, result = _successful_evidence(monkeypatch, tmp_path)
    mutated = dict(result)
    mutated["reviewed_baseline_sha"] = "f" * 40
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(mutated, config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("frozen_v10a_design_sha", "f" * 40),
        ("v10a_freeze_record_sha", "e" * 40),
        ("p5_reviewed_p4_sha", "d" * 40),
        ("p5_bookkeeping_sha", "c" * 40),
        ("final_freeze_evidence_git_blob_sha1", "b" * 40),
        ("final_freeze_evidence_sha256", "a" * 64),
        ("p3_adjudication_git_blob_sha1", "9" * 40),
    ],
)
def test_pass_evidence_rejects_each_frozen_provenance_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, value: str
) -> None:
    config, result = _successful_evidence(monkeypatch, tmp_path)
    mutated = dict(result)
    mutated[field] = value
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(mutated, config)


@pytest.mark.parametrize(
    "field",
    [
        "reviewed_baseline_sha",
        "runtime_lock_runner_git_blob_sha1",
        "runtime_lock_test_git_blob_sha1",
        "runtime_lock_design_git_blob_sha1",
    ],
)
def test_pass_evidence_rejects_each_dynamic_config_provenance_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str
) -> None:
    config, result = _successful_evidence(monkeypatch, tmp_path)
    mutated = dict(result)
    mutated[field] = "f" * 40
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(mutated, config)


@pytest.mark.parametrize("bad_sha", ["F" * 40, "f" * 39, "f" * 41, "not-a-sha"])
def test_p5_bookkeeping_sha_requires_lowercase_40_hex(tmp_path: Path, bad_sha: str) -> None:
    config = _config(tmp_path)
    evidence = runner._base_evidence(config, "PACKAGE_SET_MISMATCH")
    evidence["p5_bookkeeping_sha"] = bad_sha
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(evidence)


def test_post_provenance_fail_binding_rejects_before_any_evidence_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir()
    evidence = runner._base_evidence(config, "PACKAGE_SET_MISMATCH")
    evidence["p5_bookkeeping_sha"] = "f" * 40
    writes: list[Path] = []
    monkeypatch.setattr(runner, "_exclusive_write", lambda path, raw: writes.append(path))
    with pytest.raises(runner.RuntimeLockError):
        runner._publish_evidence_once(config, evidence)
    assert writes == []
    assert not (config.output_root / runner.EVIDENCE_NAME).exists()


def test_fail_evidence_rejects_positive_counter_for_non_unauthorized_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evidence = runner._base_evidence(config, "PACKAGE_SET_MISMATCH")
    evidence["network_requests"] = 1
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(evidence)


def test_fail_evidence_allows_unauthorized_counter_only_for_unauthorized_failure(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evidence = runner._base_evidence(config, "UNAUTHORIZED_OPERATION_OBSERVED")
    evidence["network_requests"] = 1
    runner.validate_evidence(evidence)


def test_durable_write_failure_requires_consistent_lock_hash_state(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evidence = runner._base_evidence(config, "DURABLE_WRITE_FAILURE")
    evidence["durable_lock_created"] = True
    with pytest.raises(runner.RuntimeLockError):
        runner.validate_evidence(evidence)


def test_failure_never_promotes_authority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    monkeypatch.setattr(runner, "_default_provenance_observations", lambda _: {})
    result = runner.run_lock(config)
    assert result["status"] == "FAIL"
    for key in (
        "execution_authorized", "calendar_generation_authorized", "t0_authorized",
        "historical_evaluation_authorized", "future_profitability_established",
    ):
        assert result[key] is False
    assert result["t0_run"] is False


def test_evidence_contains_no_machine_path_or_exception_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    _install_success_collectors(monkeypatch, config)
    evidence = runner.run_lock(config)
    encoded = json.dumps(evidence)
    assert str(config.output_root) not in encoded
    assert "exception" not in encoded.lower()
    assert "traceback" not in encoded.lower()


def test_cli_has_only_provenance_and_path_inputs() -> None:
    parser = runner._build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {
        "-h", "--help", "--repo-root", "--reviewed-baseline-sha", "--expected-runner-blob-sha1",
        "--expected-test-blob-sha1", "--expected-design-blob-sha1", "--output-root",
    }
    help_text = parser.format_help()
    assert "--reviewed-baseline-sha" in help_text
    assert "--expected-r2-reviewed-sha" not in help_text


def test_legacy_baseline_cli_argument_is_rejected() -> None:
    with pytest.raises(SystemExit):
        runner._build_parser().parse_args(["--expected-r2-reviewed-sha", "1" * 40])


def test_run_lock_has_no_observation_injection_parameter() -> None:
    assert "observations" not in inspect.signature(runner.run_lock).parameters
    assert "interpreter" not in inspect.signature(runner.RuntimeLockConfig).parameters


def test_production_source_has_no_third_party_imports() -> None:
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert "pandas" not in imported
    assert "pandas_market_calendars" not in imported
    assert "exchange_calendars" not in imported


def test_source_identity_is_file_and_raw_byte_based() -> None:
    assert runner.CALENDAR_SOURCE_FILE.endswith("pandas_market_calendars/calendars/jpx.py")
    assert runner.HOLIDAY_SOURCE_FILE.endswith("pandas_market_calendars/holidays/jp.py")
    assert runner.OLD_V10_JPX_BLOB != runner.CALENDAR_SOURCE_BLOB


def test_canonical_interpreter_is_not_configurable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.canonical_interpreter == config.repo_root / ".venv-real-execution" / "Scripts" / "python.exe"


def test_main_rejects_relative_paths_and_malformed_sha() -> None:
    with pytest.raises(runner.RuntimeLockError):
        runner._config_from_args(type("Args", (), {"repo_root": "repo", "reviewed_baseline_sha": "x", "expected_runner_blob_sha1": "2" * 40, "expected_test_blob_sha1": "3" * 40, "expected_design_blob_sha1": "4" * 40, "output_root": "out"})())


def test_public_failure_codes_are_closed() -> None:
    assert set(runner.FAILURE_CODES) == {
        "NONE", "UNAUTHORIZED_OPERATION_OBSERVED", "PROVENANCE_BINDING_FAILURE",
        "WRONG_CANONICAL_INTERPRETER", "PYTHON_VERSION_MISMATCH", "PACKAGE_SET_MISMATCH",
        "CALENDAR_DISTRIBUTION_VERSION_MISMATCH", "CALENDAR_SOURCE_BLOB_MISMATCH",
        "HOLIDAY_SOURCE_BLOB_MISMATCH", "RUNTIME_LOCK_CANONICALIZATION_FAILURE",
        "DURABLE_OUTPUT_COLLISION", "DURABLE_WRITE_FAILURE",
    }
