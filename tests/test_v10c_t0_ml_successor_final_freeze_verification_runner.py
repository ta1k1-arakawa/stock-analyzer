import inspect
import json
from pathlib import Path

import pytest

from scripts import v10c_t0_ml_successor_final_freeze_verification_runner as r


def make_config(tmp_path: Path) -> r.Config:
    repo = tmp_path / "repo"
    (repo / ".venv-real-execution" / "Scripts").mkdir(parents=True)
    (repo / ".venv-real-execution" / "Scripts" / "python.exe").write_bytes(b"synthetic")
    mutation = tmp_path / r.MUTATION_ATTEMPT_NAME
    mutation.mkdir()
    attempt = tmp_path / r.ATTEMPT_NAME
    return r.Config(repo, "a" * 40, "b" * 40, "c" * 64, "d" * 40, "e" * 40, mutation, attempt)


def valid_observations(config: r.Config) -> dict:
    return {
        "branch": r.AUTHORITATIVE_BRANCH,
        "head": config.reviewed_tooling_sha,
        "origin_head": config.reviewed_tooling_sha,
        "clean": True,
        "reviewed_tooling_sha": config.reviewed_tooling_sha,
        "reviewed_runner_blob_sha1": config.expected_runner_blob_sha1,
        "current_runner_blob_sha1": config.expected_runner_blob_sha1,
        "reviewed_test_blob_sha1": config.expected_test_blob_sha1,
        "current_test_blob_sha1": config.expected_test_blob_sha1,
        "candidate_blob": config.expected_candidate_blob_sha1,
        "candidate_sha256": config.expected_candidate_sha256,
        "mutation_attempt_safe": True,
        "canonical_interpreter_configured": True,
        "canonical_interpreter_existing": True,
        "final_freeze_attempt_name": r.ATTEMPT_NAME,
        "final_freeze_attempt_absent": True,
        "reserved_children_absent": True,
        "ancestors_safe": True,
        "writes": 0,
        "network_requests": 0,
    }


def live_payload(config: r.Config, *, packages=None, python="3.12.10", probe="PASS", lgbm=True, ridge=True, interpreter="PASS") -> dict:
    return {
        "canonical_interpreter_status": interpreter,
        "executable": str(config.canonical_python),
        "live_package_observation_status": "PASS",
        "packages": packages if packages is not None else [{"name": n.split("==")[0], "version": n.split("==", 1)[1]} for n in r.SUCCESSOR],
        "probe_status": probe,
        "python_version": python,
        "lightgbm_probe": lgbm,
        "ridge_probe": ridge,
    }


def fake_collect(config: r.Config):
    return valid_observations(config)


def test_constants_and_candidate_are_frozen(tmp_path):
    config = make_config(tmp_path)
    raw = Path("V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_RECORD_CANDIDATE.json").read_bytes()
    candidate = r._validate_candidate(config, raw) if False else r._strict_json(raw)
    assert candidate["successor_package_count"] == 27
    assert candidate["predecessor_package_count"] == 20
    assert candidate["successor_delta"] == list(r.DELTA)
    assert r.MUTATION_EVIDENCE_SCHEMA.endswith("EVIDENCE_V1")
    assert r.ATTEMPT_NAME.endswith("FINAL_FREEZE_ATTEMPT_1")


def test_candidate_validation_binds_actual_candidate_hash(tmp_path):
    config = make_config(tmp_path)
    raw = Path("V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_RECORD_CANDIDATE.json").read_bytes()
    candidate = r._strict_json(raw)
    assert candidate["study"] == "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR"
    actual = r._sha256(raw)
    bound = r.Config(config.repo_root, config.reviewed_tooling_sha, r._blob_sha1(raw), actual, config.expected_runner_blob_sha1, config.expected_test_blob_sha1, config.mutation_attempt_root, config.attempt_root)
    assert r._validate_candidate(bound, raw)["global_t0_readiness"] == "NO"


def test_candidate_tamper_fails(tmp_path):
    config = make_config(tmp_path)
    raw = Path("V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_RECORD_CANDIDATE.json").read_bytes()
    value = r._strict_json(raw)
    value["t0_authorized"] = True
    tampered = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    bound = r.Config(config.repo_root, config.reviewed_tooling_sha, r._blob_sha1(tampered), r._sha256(tampered), config.expected_runner_blob_sha1, config.expected_test_blob_sha1, config.mutation_attempt_root, config.attempt_root)
    with pytest.raises(r.FinalFreezeError):
        r._validate_candidate(bound, tampered)


def test_package_map_requires_exact_unique_normalized_names():
    assert len(r.EXPECTED_PREDECESSOR) == 20
    assert len(r.EXPECTED_SUCCESSOR) == 27
    with pytest.raises(r.FinalFreezeError):
        r.package_map(["scikit-learn==1.9.0", "scikit_learn==1.9.0"])


def test_phase_a_uses_collector_and_never_writes(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)
    before = sorted(tmp_path.rglob("*"))
    result = r.phase_a(config)
    after = sorted(tmp_path.rglob("*"))
    assert result["status"] == "PASS"
    assert result["writes"] == 0
    assert before == after


def test_phase_a_missing_observation_fails_closed(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", lambda _config: {})
    assert r.phase_a(config)["failure_code"] == "PRE_GATE_ENVIRONMENT_BLOCK"


def test_phase_a_path_safety_rejects_existing_attempt(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.attempt_root.mkdir()
    monkeypatch.setattr(r, "collect_production", fake_collect)
    assert r.phase_a(config)["failure_code"] == "PRE_GATE_ENVIRONMENT_BLOCK"


def test_public_phase_b_has_no_observation_or_wheel_override():
    signature = inspect.signature(r.phase_b)
    assert "observations" not in signature.parameters
    assert "wheel_paths" not in signature.parameters
    assert "canonical_python" not in signature.parameters
    assert "--observations" not in r._build_parser().format_help()
    assert "--wheel-root" not in r._build_parser().format_help()


def test_phase_b_requires_authority_before_boundary(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)
    result = r.phase_b(config, final_freeze_authorized=False)
    assert result["failure_code"] == "PRE_GATE_ENVIRONMENT_BLOCK"
    assert not config.attempt_root.exists()


def test_mkdir_without_initial_receipt_publication_is_preboundary(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)
    phase_c_calls = []
    monkeypatch.setattr(r, "phase_c", lambda _config: phase_c_calls.append(True) or {"status": "FAIL"})
    monkeypatch.setattr(r, "_atomic_json", lambda _path, _value: (_ for _ in ()).throw(RuntimeError("before publication")))

    result = r.phase_b(config, final_freeze_authorized=True)

    assert result["failure_code"] == "PRE_GATE_ENVIRONMENT_BLOCK"
    assert result["authority_consumed"] is False
    assert result["retry_authorized"] is False
    assert result["phase_c_required"] is False
    assert phase_c_calls == []
    assert config.attempt_root.exists()
    assert not (config.attempt_root / r.RESERVED[0]).exists()


def test_initial_receipt_publication_crosses_boundary_before_later_failure(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)
    phase_c_calls = []
    original_phase_c = r.phase_c
    monkeypatch.setattr(r, "phase_c", lambda supplied: phase_c_calls.append(True) or original_phase_c(supplied))

    def launch(_canonical, stdout, stderr):
        stdout.write_bytes(b"")
        stderr.write_bytes(b"failure")
        raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(r, "_launch", launch)
    result = r.phase_b(config, final_freeze_authorized=True)

    assert result["authority_consumed"] is True
    assert phase_c_calls == [True]
    assert result["phase_c_result"]["status"] == "FAIL"


def test_initial_receipt_raise_after_durable_state_is_consumed(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)
    phase_c_calls = []
    original_phase_c = r.phase_c
    monkeypatch.setattr(r, "phase_c", lambda supplied: phase_c_calls.append(True) or original_phase_c(supplied))
    original_atomic = r._atomic_json
    calls = {"count": 0}

    def publish_then_raise(path, value):
        calls["count"] += 1
        original_atomic(path, value)
        if calls["count"] == 1:
            raise RuntimeError("publication acknowledgement lost")

    monkeypatch.setattr(r, "_atomic_json", publish_then_raise)
    monkeypatch.setattr(r, "_launch", lambda _canonical, stdout, stderr: 7)

    result = r.phase_b(config, final_freeze_authorized=True)

    assert result["authority_consumed"] is True
    assert result["retry_authorized"] is False
    assert phase_c_calls == [True]


def test_uncertain_initial_receipt_publication_fails_closed_and_inspects(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)
    phase_c_calls = []
    original_phase_c = r.phase_c
    monkeypatch.setattr(r, "phase_c", lambda supplied: phase_c_calls.append(True) or original_phase_c(supplied))

    def leave_unpublished_temp(path, _value):
        path.with_name(path.name + ".tmp").write_bytes(b"uncertain")
        raise RuntimeError("publication status unknown")

    monkeypatch.setattr(r, "_atomic_json", leave_unpublished_temp)
    result = r.phase_b(config, final_freeze_authorized=True)

    assert result["authority_consumed"] is True
    assert result["retry_authorized"] is False
    assert phase_c_calls == [True]
    assert result["phase_b_result"]["boundary_uncertain"] is True
    assert (config.attempt_root / (r.RESERVED[0] + ".tmp")).exists()


def test_phase_b_success_routes_to_phase_c_and_publishes_pass(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)

    def launch(canonical, stdout, stderr):
        stdout.write_text(json.dumps(live_payload(config), sort_keys=True), encoding="utf-8")
        stderr.write_bytes(b"")
        return 0

    monkeypatch.setattr(r, "_launch", launch)
    result = r.phase_b(config, final_freeze_authorized=True)
    assert result["status"] == "PASS"
    assert result["authority_consumed"] is True
    assert result["retry_authorized"] is False
    assert result["phase_c_result"]["status"] == "PASS"
    assert (config.attempt_root / "final_freeze_evidence.json").exists()
    evidence = r._strict_json((config.attempt_root / "final_freeze_evidence.json").read_bytes())
    assert evidence["canonical_environment_promoted"] is False
    assert evidence["environment_frozen"] is False
    assert evidence["global_t0_readiness"] == "NO"


@pytest.mark.parametrize("launch_mode", ["nonzero", "exception"])
def test_post_boundary_failure_always_routes_to_phase_c(tmp_path, monkeypatch, launch_mode):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)

    def launch(_canonical, stdout, stderr):
        stdout.write_bytes(b"")
        stderr.write_bytes(b"failure")
        if launch_mode == "exception":
            raise RuntimeError("synthetic launch failure")
        return 7

    monkeypatch.setattr(r, "_launch", launch)
    result = r.phase_b(config, final_freeze_authorized=True)
    assert result["authority_consumed"] is True
    assert result["retry_authorized"] is False
    assert result["phase_c_result"]["failure_code"] == "IMPLEMENTATION_FAILURE"
    assert (config.attempt_root / "final_freeze_evidence.json").exists()


def test_exit_zero_package_drift_is_live_validation_failure(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)
    drift = [{"name": n.split("==")[0], "version": n.split("==", 1)[1]} for n in r.SUCCESSOR[:-1]]

    def launch(_canonical, stdout, stderr):
        stdout.write_text(json.dumps(live_payload(config, packages=drift), sort_keys=True), encoding="utf-8")
        stderr.write_bytes(b"")
        return 0

    monkeypatch.setattr(r, "_launch", launch)
    result = r.phase_b(config, final_freeze_authorized=True)
    assert result["phase_c_result"]["failure_code"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"


def test_exit_zero_interpreter_or_probe_drift_is_live_validation_failure(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)

    def launch(_canonical, stdout, stderr):
        stdout.write_text(json.dumps(live_payload(config, interpreter="FAIL", probe="FAIL", lgbm=False, ridge=False), sort_keys=True), encoding="utf-8")
        stderr.write_bytes(b"")
        return 0

    monkeypatch.setattr(r, "_launch", launch)
    result = r.phase_b(config, final_freeze_authorized=True)
    assert result["phase_c_result"]["failure_code"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"


def test_exit_zero_alternate_executable_is_live_validation_failure(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)

    def launch(_canonical, stdout, stderr):
        payload = live_payload(config)
        payload["executable"] = str(config.repo_root / ".venv" / "Scripts" / "python.exe")
        stdout.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        stderr.write_bytes(b"")
        return 0

    monkeypatch.setattr(r, "_launch", launch)
    result = r.phase_b(config, final_freeze_authorized=True)
    assert result["phase_c_result"]["failure_code"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"


def test_existing_pass_evidence_is_inspect_only(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(r, "collect_production", fake_collect)

    def launch(_canonical, stdout, stderr):
        stdout.write_text(json.dumps(live_payload(config), sort_keys=True), encoding="utf-8")
        stderr.write_bytes(b"")
        return 0

    monkeypatch.setattr(r, "_launch", launch)
    first = r.phase_b(config, final_freeze_authorized=True)
    evidence_path = config.attempt_root / "final_freeze_evidence.json"
    before = evidence_path.read_bytes()
    monkeypatch.setattr(r, "_live_fields_from_stdout", lambda _path: (_ for _ in ()).throw(AssertionError("reprobe")))
    second = r.phase_c(config)
    assert first["phase_c_result"]["status"] == "PASS"
    assert second["status"] == "PASS"
    assert second["existing_evidence_inspected"] is True
    assert evidence_path.read_bytes() == before


def test_malformed_existing_evidence_fails_without_rewrite(tmp_path):
    config = make_config(tmp_path)
    config.attempt_root.mkdir()
    state = {"schema_version": r.STATE_SCHEMA, "attempt_name": r.ATTEMPT_NAME, "reviewed_implementation_sha": config.reviewed_tooling_sha, "authority_consumed": True, "retry_authorized": False, "phase_c_required": True, "launch_attempted": True, "process_started": True, "process_exit_code": 0}
    (config.attempt_root / r.RESERVED[0]).write_text(json.dumps(state), encoding="utf-8")
    (config.attempt_root / r.RESERVED[1]).write_bytes(b"")
    (config.attempt_root / r.RESERVED[2]).write_bytes(b"")
    evidence = config.attempt_root / r.RESERVED[3]
    evidence.write_text('{"schema_version":"wrong"}', encoding="utf-8")
    before = evidence.read_bytes()
    result = r.phase_c(config)
    assert result["failure_code"] == "IMPLEMENTATION_FAILURE"
    assert evidence.read_bytes() == before


def test_phase_c_isolated_from_collector_and_wheel_root(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.attempt_root.mkdir()
    monkeypatch.setattr(r, "collect_production", lambda _config: (_ for _ in ()).throw(AssertionError("collector called")))
    result = r.phase_c(config)
    assert result["failure_code"] == "IMPLEMENTATION_FAILURE"
    assert "wheel_root" not in result


def test_main_dispatches_phase_a_without_injection_option(tmp_path, monkeypatch, capsys):
    config = make_config(tmp_path)
    called = []
    monkeypatch.setattr(r, "collect_production", lambda supplied: called.append(supplied) or valid_observations(supplied))
    args = ["phase-a", "--repo-root", str(config.repo_root), "--reviewed-tooling-sha", config.reviewed_tooling_sha, "--expected-candidate-blob-sha1", config.expected_candidate_blob_sha1, "--expected-candidate-sha256", config.expected_candidate_sha256, "--expected-runner-blob-sha1", config.expected_runner_blob_sha1, "--expected-test-blob-sha1", config.expected_test_blob_sha1, "--mutation-attempt-root", str(config.mutation_attempt_root), "--attempt-root", str(config.attempt_root)]
    assert r.main(args) == 0
    assert called
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
