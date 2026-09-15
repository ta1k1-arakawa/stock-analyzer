import hashlib
import inspect
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import v10c_t0_ml_canonical_mutation_runner as r
from scripts import v10c_t0_ml_environment_contract as contract
from scripts import v10c_t0_ml_environment_offline_readjudication_runner as offline


REVIEWED_SHA = "a" * 40
PUBLIC_ARTIFACTS = (
    "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_DRAFT.md",
    "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_FREEZE_APPROVAL.json",
    "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_RESOLUTION_PROMOTION.json",
    "V10C_T0_ML_RESOLUTION_SOURCE_PROVENANCE.json",
    "V10A_RUNTIME_ENVIRONMENT_LOCK.json",
    "V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json",
    "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json",
    "requirements-real-execution.lock.txt",
    "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK.txt",
)


def _config(root: Path, tmp_path: Path) -> r.Config:
    audit = tmp_path / "audit"
    audit.mkdir(exist_ok=True)
    return r.Config(
        root,
        root / ".venv-real-execution" / "Scripts" / "python.exe",
        audit / r.ATTEMPT_NAME,
        root / "wheels",
        REVIEWED_SHA,
    )


def _wheel_rows() -> tuple[dict[str, str], ...]:
    rows = []
    for pin in r.SUCCESSOR:
        name, version = pin.split("==")
        rows.append({
            "name": name,
            "version": version,
            "filename": f"{name}-{version}-py3-none-any.whl",
            "sha256": hashlib.sha256(pin.encode()).hexdigest(),
        })
    return tuple(rows)


@pytest.fixture
def synthetic(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "repo"
    root.mkdir()
    for name in PUBLIC_ARTIFACTS:
        shutil.copyfile(source / name, root / name)

    interpreter = root / ".venv-real-execution" / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"synthetic interpreter")
    wheel_root = root / "wheels"
    wheel_root.mkdir()
    rows = _wheel_rows()
    for index, row in enumerate(rows):
        path = wheel_root / row["filename"]
        with path.open("wb") as handle:
            if index == 0:
                handle.truncate(r.SOURCE_WHEEL_TOTAL_BYTES)

    current_blob = contract.git_blob_sha1(
        (source / "scripts" / "v10c_t0_ml_canonical_mutation_runner.py").read_bytes()
    )
    artifact_blobs = {
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_DRAFT.md": r.DESIGN_BLOB,
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_FREEZE_APPROVAL.json": r.APPROVAL_BLOB,
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_RESOLUTION_PROMOTION.json": r.PROMOTION_BLOB,
        "V10C_T0_ML_RESOLUTION_SOURCE_PROVENANCE.json": r.SOURCE_PROVENANCE_BLOB,
        "V10A_RUNTIME_ENVIRONMENT_LOCK.json": "9dfe03cf807b3580d432146839e8eb013bfa3c63",
        "V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json": "e07040f75a92ef0669215f4a7e2e98b71ea29d36",
        "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json": r.V10A_FREEZE_BLOB,
        "requirements-real-execution.lock.txt": r.PREDECESSOR_BLOB,
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK.txt": r.SUCCESSOR_BLOB,
    }

    def fake_run(command, **kwargs):
        if command[0] == "git":
            args = command[1:]
            if args[:2] == ["config", "--get"]:
                output = "git@github.com:ta1k1-arakawa/stock-analyzer.git"
            elif args == ["branch", "--show-current"]:
                output = r.AUTHORITATIVE_BRANCH
            elif args == ["rev-parse", "HEAD"] or args == [
                "rev-parse", f"refs/remotes/origin/{r.AUTHORITATIVE_BRANCH}"
            ]:
                output = REVIEWED_SHA
            elif args[:2] == ["rev-parse", "--verify"]:
                output = args[2].split("^{", 1)[0]
            elif args[:2] == ["status", "--porcelain=v1"]:
                output = ""
            elif args[:2] == ["hash-object", "scripts/v10c_t0_ml_canonical_mutation_runner.py"]:
                output = current_blob
            elif args[0] == "rev-parse" and ":" in args[-1]:
                ref, name = args[-1].split(":", 1)
                output = current_blob if ref == REVIEWED_SHA else artifact_blobs[name]
            else:
                raise AssertionError(f"unexpected git call: {args}")
            return subprocess.CompletedProcess(command, 0, output, "")

        if "-c" in command:
            payload = {
                "python": "3.12.10",
                "executable": str(interpreter),
                "packages": [
                    {"name": item.split("==", 1)[0], "version": item.split("==", 1)[1]}
                    for item in r.PREDECESSOR
                ],
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(command, 0, "pip 25.0.1", "")

    monkeypatch.setattr(r.subprocess, "run", fake_run)
    monkeypatch.setattr(contract, "inspect_wheelhouse", lambda _: ("NONE", rows))
    monkeypatch.setattr(offline, "_validate_wheelhouse_manifest", lambda *_: None)
    return root, _config(root, tmp_path), rows


def _main_phase_a(config: r.Config) -> int:
    return r.main([
        "phase-a", "--reviewed-implementation-sha", REVIEWED_SHA,
        "--repo-root", str(config.repo_root),
        "--canonical-python", str(config.canonical_python),
        "--attempt-root", str(config.attempt_root),
        "--wheel-root", str(config.wheel_root),
    ])


def test_production_cli_phase_a_reaches_collector_and_phase_a(synthetic, monkeypatch, capsys):
    root, config, _ = synthetic
    called = {"count": 0}
    original = r.collect_production

    def wrapped(value):
        called["count"] += 1
        return original(value)

    monkeypatch.setattr(r, "collect_production", wrapped)
    assert _main_phase_a(config) == 0
    assert called["count"] == 1
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert not config.attempt_root.exists()


@pytest.mark.parametrize("kind", ["successor", "promotion", "source", "v10a"])
def test_production_cli_fail_closed_on_public_artifact_drift(synthetic, kind):
    root, config, _ = synthetic
    names = {
        "successor": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK.txt",
        "promotion": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_RESOLUTION_PROMOTION.json",
        "source": "V10C_T0_ML_RESOLUTION_SOURCE_PROVENANCE.json",
        "v10a": "V10A_RUNTIME_ENVIRONMENT_LOCK.json",
    }
    path = root / names[kind]
    path.write_bytes(path.read_bytes() + b"\n")
    assert _main_phase_a(config) == 1


def test_production_cli_fail_closed_on_unsafe_interpreter_and_manifest(synthetic, monkeypatch):
    _, config, _ = synthetic
    monkeypatch.setattr(
        r, "_canonical_identity",
        lambda _: (_ for _ in ()).throw(r.MutationError("unsafe")),
    )
    assert _main_phase_a(config) == 1


def test_production_cli_fail_closed_when_reviewed_manifest_validator_fails(synthetic, monkeypatch):
    _, config, _ = synthetic
    monkeypatch.setattr(
        offline, "_validate_wheelhouse_manifest",
        lambda *_: (_ for _ in ()).throw(r.MutationError("manifest drift")),
    )
    assert _main_phase_a(config) == 1


def test_actual_lock_and_provenance_helpers_reject_mismatches(synthetic):
    root, config, _ = synthetic
    with pytest.raises(r.MutationError):
        r._artifact(
            root, lambda *args: r.SUCCESSOR_BLOB,
            "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK.txt",
            r.SUCCESSOR_BLOB, "0" * 64,
        )
    assert r._approval_semantics(
        (root / "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_FREEZE_APPROVAL.json").read_bytes()
    )
    assert config.attempt_root.name == r.ATTEMPT_NAME


def test_verified_result_and_public_phase_b_surface(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    wheel_root = root / "wheels"
    wheel_root.mkdir()
    paths = []
    for index, pin in enumerate(r.DELTA):
        path = wheel_root / f"{index}.whl"
        path.write_bytes(pin.encode())
        paths.append(path)
    config = r.Config(
        root, root / ".venv-real-execution" / "Scripts" / "python.exe",
        tmp_path / r.ATTEMPT_NAME, wheel_root, REVIEWED_SHA,
    )
    observed = {
        "branch": r.AUTHORITATIVE_BRANCH, "head": REVIEWED_SHA, "origin_head": REVIEWED_SHA,
        "dirty": False, "reviewed_runner_blob": "b" * 40, "current_runner_blob": "b" * 40,
        "design_sha": r.DESIGN_SHA, "design_blob": r.DESIGN_BLOB,
        "approval_commit": r.APPROVAL_COMMIT, "approval_blob": r.APPROVAL_BLOB,
        "predecessor_blob": r.PREDECESSOR_BLOB, "predecessor_sha256": r.PREDECESSOR_SHA256,
        "successor_blob": r.SUCCESSOR_BLOB, "successor_sha256": r.SUCCESSOR_SHA256,
        "promotion_blob": r.PROMOTION_BLOB, "source_resolution_head": r.SOURCE_RESOLUTION_HEAD,
        "wheel_count": 27, "wheel_total_bytes": r.SOURCE_WHEEL_TOTAL_BYTES,
        "wheel_manifest_verified": True, "candidate_sha256": r.OFFLINE_CANDIDATE_SHA256,
        "evidence_sha256": r.OFFLINE_EVIDENCE_SHA256, "python_version": "3.12.10",
        "pip_reachable": True, "attempt_root_absent": True, "reserved_absent": True,
        "ancestors_safe": True, "governed_root_safe": True, "approval_semantics": True,
        "v10a_predecessor_authority": True, "packages": r.PREDECESSOR,
        "delta_wheels": dict(zip(r.DELTA, paths)), "wheel_root_realpath": wheel_root.resolve(),
        "network_requests": 0, "writes": 0,
        "wheel_sha256": {
            pin: hashlib.sha256(path.read_bytes()).hexdigest()
            for pin, path in zip(r.DELTA, paths)
        },
    }
    monkeypatch.setattr(r, "collect_production", lambda _: observed)
    verified = r._verified_result(config, observed)
    assert len(verified.verified_delta_wheels) == 7
    assert "observed" not in inspect.signature(r.phase_b).parameters
    assert "wheel_paths" not in inspect.signature(r.phase_b).parameters


def test_phase_a_zero_writes_and_global_t0_unchanged():
    assert not r.T0_AUTHORIZED and r.GLOBAL_T0_READINESS == "NO"
    assert len(r.PREDECESSOR) == 20 and len(r.SUCCESSOR) == 27 and len(r.DELTA) == 7


def _attempt_config(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    interpreter = repo / ".venv-real-execution" / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_bytes(b"synthetic interpreter")
    attempt = tmp_path / "audit" / r.ATTEMPT_NAME
    attempt.parent.mkdir()
    return r.Config(repo, interpreter, attempt, tmp_path / "missing-wheel-root", REVIEWED_SHA)


def _write_state(config, exit_code=0, *, launch_attempted=True, process_started=True):
    config.attempt_root.mkdir(parents=True, exist_ok=True)
    (config.attempt_root / r.RESERVED[0]).write_text(json.dumps({
        "authority_consumed": True,
        "retry_authorized": False,
        "phase_c_required": True,
        "launch_attempted": launch_attempted,
        "process_started": process_started,
        "exit_code": exit_code,
    }), encoding="utf-8")
    (config.attempt_root / r.RESERVED[1]).write_bytes(b"stdout")
    (config.attempt_root / r.RESERVED[2]).write_bytes(b"stderr")


def _runtime_payload(config, packages=None, *, probe_status="PASS", executable=None):
    return {
        "python_version": "3.12.10",
        "executable": str(executable or config.canonical_python),
        "packages": [
            {"name": pin.split("==", 1)[0], "version": pin.split("==", 1)[1]}
            for pin in (packages or r.SUCCESSOR)
        ],
        "probe_status": probe_status,
        "lightgbm_probe": probe_status == "PASS",
        "ridge_probe": probe_status == "PASS",
    }


def _runtime_run(payload):
    def fake_run(command, **kwargs):
        assert "-m" not in command
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")
    return fake_run


def test_phase_c_success_is_dedicated_and_publishes_safe_evidence(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    _write_state(config)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    calls = []
    payload = _runtime_payload(config)

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(r.subprocess, "run", fake_run)
    monkeypatch.setattr(r, "collect_production", lambda _: (_ for _ in ()).throw(AssertionError("phase A called")))
    result = r.phase_c(config)
    assert result["status"] == "PASS"
    assert result["authority_consumed"] is True
    assert result["retry_authorized"] is False
    assert result["full_validation_run"] is True
    assert result["schema_version"] == r.PHASE_C_EVIDENCE_SCHEMA
    assert result["reviewed_implementation_sha"] == REVIEWED_SHA
    assert result["inspection"]["stdout"]["sha256"] == hashlib.sha256(b"stdout").hexdigest()
    assert len(calls) == 1 and "-m" not in calls[0]
    evidence = (config.attempt_root / r.RESERVED[3]).read_text(encoding="utf-8")
    assert str(config.repo_root) not in evidence


@pytest.mark.parametrize("field,value", [
    ("canonical_interpreter_status", "FAIL"),
    ("live_package_observation_status", "FAIL"),
    ("probe_status", "FAIL"),
    ("lightgbm_probe", False),
    ("ridge_probe", False),
    ("package_count", 0),
    ("python_version", "3.11.0"),
])
def test_phase_c_existing_pass_semantic_tamper_fails_closed_without_reprobe(
    tmp_path, monkeypatch, field, value
):
    config = _attempt_config(tmp_path)
    _write_state(config)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    monkeypatch.setattr(r.subprocess, "run", _runtime_run(_runtime_payload(config)))
    assert r.phase_c(config)["status"] == "PASS"
    evidence_path = config.attempt_root / r.RESERVED[3]
    tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
    tampered[field] = value
    evidence_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    tampered_bytes = evidence_path.read_bytes()
    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(AssertionError("re-probed")))
    result = r.phase_c(config)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "CANONICAL_MUTATION_FAILURE"
    assert result["failure_class"] == "CANONICAL_MUTATION_FAILURE"
    assert evidence_path.read_bytes() == tampered_bytes


@pytest.mark.parametrize("field,value", [
    ("reviewed_implementation_sha", "b" * 40),
    ("schema_version", "WRONG_SCHEMA"),
])
def test_phase_c_existing_provenance_tamper_fails_closed_without_reprobe(
    tmp_path, monkeypatch, field, value
):
    config = _attempt_config(tmp_path)
    _write_state(config)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    monkeypatch.setattr(r.subprocess, "run", _runtime_run(_runtime_payload(config)))
    assert r.phase_c(config)["status"] == "PASS"
    evidence_path = config.attempt_root / r.RESERVED[3]
    tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
    tampered[field] = value
    evidence_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    tampered_bytes = evidence_path.read_bytes()
    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(AssertionError("re-probed")))
    result = r.phase_c(config)
    assert result["failure_code"] == "CANONICAL_MUTATION_FAILURE"
    assert evidence_path.read_bytes() == tampered_bytes


@pytest.mark.parametrize("field,value", [
    ("status", "PASS"),
    ("failure_class", "CANONICAL_MUTATION_FAILURE"),
])
def test_phase_c_existing_live_failure_contradiction_fails_closed(
    tmp_path, monkeypatch, field, value
):
    config = _attempt_config(tmp_path)
    _write_state(config)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    monkeypatch.setattr(
        r.subprocess,
        "run",
        _runtime_run(_runtime_payload(config, r.PREDECESSOR, probe_status="NOT_RUN")),
    )
    assert r.phase_c(config)["failure_class"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"
    evidence_path = config.attempt_root / r.RESERVED[3]
    tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
    tampered[field] = value
    evidence_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    tampered_bytes = evidence_path.read_bytes()
    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(AssertionError("re-probed")))
    result = r.phase_c(config)
    assert result["failure_code"] == "CANONICAL_MUTATION_FAILURE"
    assert evidence_path.read_bytes() == tampered_bytes


@pytest.mark.parametrize("field,value", [
    ("authority_consumed", False),
    ("retry_authorized", True),
])
def test_phase_c_existing_canonical_failure_contradiction_fails_closed(
    tmp_path, monkeypatch, field, value
):
    config = _attempt_config(tmp_path)
    _write_state(config, exit_code=7)
    first = r.phase_c(config)
    assert first["failure_class"] == "CANONICAL_MUTATION_FAILURE"
    evidence_path = config.attempt_root / r.RESERVED[3]
    tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
    tampered[field] = value
    evidence_path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    tampered_bytes = evidence_path.read_bytes()
    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(AssertionError("re-probed")))
    result = r.phase_c(config)
    assert result["failure_code"] == "CANONICAL_MUTATION_FAILURE"
    assert evidence_path.read_bytes() == tampered_bytes


@pytest.mark.parametrize("exit_code", [7, "UNKNOWN"])
def test_phase_c_post_boundary_failure_is_safe_and_no_runtime_probe(tmp_path, monkeypatch, exit_code):
    config = _attempt_config(tmp_path)
    _write_state(config, exit_code=exit_code, process_started="UNKNOWN" if exit_code == "UNKNOWN" else True)
    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(AssertionError("probe ran")))
    result = r.phase_c(config)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "CANONICAL_MUTATION_FAILURE"
    assert result["full_validation_run"] is False
    assert (config.attempt_root / r.RESERVED[3]).exists()


def test_phase_c_package_drift_and_probe_failure_are_live_validation_failures(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    _write_state(config)
    monkeypatch.setattr(r.subprocess, "run", _runtime_run(
        _runtime_payload(config, r.PREDECESSOR)
    ))
    drift = r.phase_c(config)
    assert drift["failure_code"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"
    assert drift["full_validation_run"] is False

    config = _attempt_config(tmp_path / "probe")
    _write_state(config)
    monkeypatch.setattr(r.subprocess, "run", _runtime_run(
        _runtime_payload(config, probe_status="FAIL")
    ))
    probe = r.phase_c(config)
    assert probe["failure_code"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"
    assert probe["full_validation_run"] is False


def test_phase_c_existing_evidence_is_never_overwritten(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    _write_state(config)
    evidence_path = config.attempt_root / r.RESERVED[3]
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    calls = {"runtime": 0}

    def runtime_once(command, **kwargs):
        calls["runtime"] += 1
        return subprocess.CompletedProcess(command, 0, json.dumps(_runtime_payload(config)), "")

    monkeypatch.setattr(r.subprocess, "run", runtime_once)
    first = r.phase_c(config)
    assert first["status"] == "PASS"
    evidence_bytes = evidence_path.read_bytes()

    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(AssertionError("re-probed")))
    second = r.phase_c(config)
    assert second["status"] == "PASS"
    assert second["existing_evidence_inspected"] is True
    assert calls["runtime"] == 1
    assert evidence_path.read_bytes() == evidence_bytes


def test_phase_c_existing_failure_class_is_preserved_without_reprobe(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    _write_state(config, exit_code=7)
    evidence_path = config.attempt_root / r.RESERVED[3]
    first = r.phase_c(config)
    assert first["failure_class"] == "CANONICAL_MUTATION_FAILURE"
    evidence_bytes = evidence_path.read_bytes()
    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(AssertionError("re-probed")))
    result = r.phase_c(config)
    assert result["status"] == "FAIL"
    assert result["failure_class"] == "CANONICAL_MUTATION_FAILURE"
    assert result["existing_evidence_inspected"] is True
    assert evidence_path.read_bytes() == evidence_bytes


def test_phase_c_existing_live_failure_class_is_preserved_without_reprobe(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    _write_state(config)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    monkeypatch.setattr(
        r.subprocess,
        "run",
        _runtime_run(_runtime_payload(config, r.PREDECESSOR, probe_status="NOT_RUN")),
    )
    first = r.phase_c(config)
    assert first["failure_class"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"
    evidence_path = config.attempt_root / r.RESERVED[3]
    evidence_bytes = evidence_path.read_bytes()
    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(AssertionError("re-probed")))
    result = r.phase_c(config)
    assert result["failure_class"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"
    assert result["existing_evidence_inspected"] is True
    assert evidence_path.read_bytes() == evidence_bytes


def test_phase_c_tampered_existing_evidence_fails_closed_without_rewrite(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    _write_state(config)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    monkeypatch.setattr(r.subprocess, "run", _runtime_run(_runtime_payload(config)))
    assert r.phase_c(config)["status"] == "PASS"
    evidence_path = config.attempt_root / r.RESERVED[3]
    evidence_path.write_bytes(b"{\"status\":\"PASS\"}")
    tampered = evidence_path.read_bytes()
    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(AssertionError("re-probed")))
    result = r.phase_c(config)
    assert result["failure_class"] == "CANONICAL_MUTATION_FAILURE"
    assert result["existing_evidence_inspected"] is True
    assert evidence_path.read_bytes() == tampered


def test_phase_c_publication_failure_always_becomes_canonical_failure(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    _write_state(config)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    monkeypatch.setattr(r.subprocess, "run", _runtime_run(_runtime_payload(config, r.PREDECESSOR)))
    monkeypatch.setattr(r, "_publish_phase_c_evidence", lambda *_: False)
    result = r.phase_c(config)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "CANONICAL_MUTATION_FAILURE"
    assert result["failure_class"] == "CANONICAL_MUTATION_FAILURE"
    assert result["evidence_published"] is False


def test_phase_c_exit_zero_interpreter_failure_is_live_validation_failure(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    _write_state(config)
    monkeypatch.setattr(r, "_canonical_identity", lambda _: (_ for _ in ()).throw(r.MutationError("identity")))
    result = r.phase_c(config)
    assert result["failure_code"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"
    assert result["failure_class"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"


def test_phase_c_exit_zero_runtime_observer_failure_is_live_validation_failure(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    _write_state(config)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    monkeypatch.setattr(
        r.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "observer failed"),
    )
    result = r.phase_c(config)
    assert result["failure_code"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"
    assert result["failure_class"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"


def test_phase_c_duplicate_normalized_package_blocks_probes(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    _write_state(config)
    duplicate = list(r.SUCCESSOR)
    duplicate[-1] = "cloud_pickle==3.1.2"
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    monkeypatch.setattr(r.subprocess, "run", _runtime_run(_runtime_payload(config, duplicate)))
    result = r.phase_c(config)
    assert result["failure_code"] == "LIVE_ENVIRONMENT_VALIDATION_FAILURE"
    assert result["full_validation_run"] is False
    script = r._phase_c_probe_script()
    assert script.index("if key in names") < script.index("from lightgbm import")


def test_phase_b_pre_boundary_failure_does_not_call_phase_c(tmp_path, monkeypatch):
    config = _attempt_config(tmp_path)
    called = {"phase_c": 0}
    monkeypatch.setattr(r, "collect_production", lambda _: (_ for _ in ()).throw(r.MutationError("blocked")))
    original = r.phase_c

    def phase_c_spy(value):
        called["phase_c"] += 1
        return original(value)

    monkeypatch.setattr(r, "phase_c", phase_c_spy)
    result = r.phase_b(config, mutation_authorized=True, launcher=lambda *_: 0)
    assert result["failure_code"] == "PRE_GATE_ENVIRONMENT_BLOCK"
    assert called["phase_c"] == 0
    assert not config.attempt_root.exists()


def test_phase_b_success_runs_phase_c_once_and_returns_nested_results(synthetic, monkeypatch):
    _, config, _ = synthetic
    observed = r.collect_production(config)
    for pin, path in observed["delta_wheels"].items():
        observed["wheel_sha256"][pin] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(r, "collect_production", lambda _: observed)
    monkeypatch.setattr(r, "_canonical_identity", lambda value: value.canonical_python)
    monkeypatch.setattr(r.subprocess, "run", _runtime_run(_runtime_payload(config)))
    calls = {"phase_c": 0}
    original = r.phase_c

    def phase_c_spy(value):
        calls["phase_c"] += 1
        return original(value)

    monkeypatch.setattr(r, "phase_c", phase_c_spy)

    def launcher(argv, stdout, stderr):
        assert argv[1:6] == ["-m", "pip", "install", "--no-deps", "--no-index"]
        stdout.write_bytes(b"synthetic stdout")
        stderr.write_bytes(b"")
        return 0

    result = r.phase_b(config, mutation_authorized=True, launcher=launcher)
    assert result["status"] == "PASS"
    assert result["phase_c_result"]["status"] == "PASS"
    assert result["authority_consumed"] is True
    assert result["retry_authorized"] is False
    assert calls["phase_c"] == 1


@pytest.mark.parametrize("launcher", [
    lambda *_: 9,
    lambda *_: (_ for _ in ()).throw(RuntimeError("launch")),
])
def test_phase_b_nonzero_or_launch_exception_runs_phase_c_once(synthetic, monkeypatch, launcher):
    _, config, _ = synthetic
    observed = r.collect_production(config)
    for pin, path in observed["delta_wheels"].items():
        observed["wheel_sha256"][pin] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(r, "collect_production", lambda _: observed)
    calls = {"phase_c": 0}
    original = r.phase_c

    def phase_c_spy(value):
        calls["phase_c"] += 1
        return original(value)

    monkeypatch.setattr(r, "phase_c", phase_c_spy)
    result = r.phase_b(config, mutation_authorized=True, launcher=launcher)
    assert result["status"] == "FAIL"
    assert result["failure_code"] == "CANONICAL_MUTATION_FAILURE"
    assert result["phase_c_result"]["full_validation_run"] is False
    assert calls["phase_c"] == 1


def test_phase_b_state_update_failure_after_boundary_is_not_retried(synthetic, monkeypatch):
    _, config, _ = synthetic
    observed = r.collect_production(config)
    for pin, path in observed["delta_wheels"].items():
        observed["wheel_sha256"][pin] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(r, "collect_production", lambda _: observed)
    original_atomic = r._atomic_json
    calls = {"atomic": 0, "launch": 0}

    def failing_atomic(path, value):
        calls["atomic"] += 1
        if calls["atomic"] == 2:
            raise OSError("synthetic state publication failure")
        return original_atomic(path, value)

    monkeypatch.setattr(r, "_atomic_json", failing_atomic)

    def launcher(*_):
        calls["launch"] += 1
        return 0

    result = r.phase_b(config, mutation_authorized=True, launcher=launcher)
    assert result["failure_code"] == "CANONICAL_MUTATION_FAILURE"
    assert result["authority_consumed"] is True
    assert result["retry_authorized"] is False
    assert calls["launch"] == 0
    assert calls["atomic"] == 3
    assert (config.attempt_root / r.RESERVED[3]).exists()


def test_main_phase_c_does_not_collect_phase_a_or_require_wheel_root(tmp_path, monkeypatch, capsys):
    config = _attempt_config(tmp_path)
    _write_state(config, exit_code=7)
    monkeypatch.setattr(r, "collect_production", lambda _: (_ for _ in ()).throw(AssertionError("phase A called")))
    rc = r.main([
        "phase-c", "--reviewed-implementation-sha", REVIEWED_SHA,
        "--repo-root", str(config.repo_root),
        "--canonical-python", str(config.canonical_python),
        "--attempt-root", str(config.attempt_root),
        "--wheel-root", str(config.wheel_root),
    ])
    assert rc == 1
    output = json.loads(capsys.readouterr().out)
    assert output["failure_code"] == "CANONICAL_MUTATION_FAILURE"
    assert not config.wheel_root.exists()
