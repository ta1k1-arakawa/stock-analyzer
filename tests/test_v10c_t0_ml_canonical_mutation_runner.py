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


def test_phase_c_safe_failure_remains_unchanged(tmp_path):
    config = r.Config(
        tmp_path, tmp_path / ".venv-real-execution" / "Scripts" / "python.exe",
        tmp_path / r.ATTEMPT_NAME, tmp_path / "wheels", REVIEWED_SHA,
    )
    config.attempt_root.mkdir()
    (config.attempt_root / "mutation_state.json").write_text("{}", encoding="utf-8")
    assert r.phase_c(config, {})["failure_code"] == "CANONICAL_MUTATION_FAILURE"
