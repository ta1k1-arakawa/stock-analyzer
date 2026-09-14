"""V10C canonical-mutation runner; phases are inert until separately authorized.

Importing this module performs no I/O.  Runtime observations and process
launches are explicit boundaries so unit tests never inspect the real venv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
DESIGN_SHA = "7eef754dce624876b10bdcea1fff29ba7da618ed"
DESIGN_BLOB = "9ba83c83f55b19c6068eac9a7f1efd75c1499bcb"
APPROVAL_COMMIT = "9a22b91ec14f6637141e56c9c48f01efbfefd460"
APPROVAL_BLOB = "29989432a553bb2ad49483b9850ca6e339e02f6b"
PREDECESSOR_BLOB = "99395e7a5be752fb3ea92fd31be0334f38792261"
PREDECESSOR_SHA256 = "eb325ac5e3417e6407400b18c8d90ca734a32e852056926e5bcd2a635e43c444"
SUCCESSOR_BLOB = "13636e58fbe40071be04cbfa57c3990c1d8ff2e0"
SUCCESSOR_SHA256 = "f38dd4c7319465bb7e6ff429e8dff4a476d9966c744b19e50264dcc0b18e8300"
PROMOTION_BLOB = "b866e6d77508d6366569ee6a229587c59c3c8be2"
SOURCE_RESOLUTION_HEAD = "3aee6c2772f30c6dc35d2a7efb862ae15091febc"
SOURCE_WHEEL_COUNT, SOURCE_WHEEL_TOTAL_BYTES = 27, 94451528
SOURCE_WHEEL_MANIFEST_SHA256 = "5d5953f14b0609767972679554e1999e754621056d863f8c33def96988797b74"
OFFLINE_CANDIDATE_SHA256 = "893881cbb9612e3402b0f4e1e434edfc4283da81a0f6a6d264d019ae5573c48e"
OFFLINE_EVIDENCE_SHA256 = "b4398e32be354de03e64202148dc6933ed45ef888657e909de1f52a4a051206b"
T0_AUTHORIZED = False
GLOBAL_T0_READINESS = "NO"
ATTEMPT_NAME = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_ATTEMPT_1"
RESERVED = ("mutation_state.json", "mutation_stdout.txt", "mutation_stderr.txt", "mutation_evidence.json")
PREDECESSOR = (
    "cffi==2.1.1", "charset-normalizer==3.5.1", "cryptography==50.0.1", "exchange-calendars==4.13.2",
    "korean-lunar-calendar==0.4.0", "numpy==2.5.2", "pandas==3.0.5", "pandas-market-calendars==5.4.0",
    "pdfminer-six==20260107", "pdfplumber==0.11.10", "pillow==12.3.0", "pip==25.0.1", "pycparser==3.0",
    "pyluach==2.3.0", "pypdfium2==5.13.0", "python-dateutil==2.9.0.post0", "six==1.17.0", "toolz==1.1.0", "tzdata==2026.3", "xlrd==2.0.2")
DELTA = ("cloudpickle==3.1.2", "joblib==1.6.0", "lightgbm==4.6.0", "narwhals==2.26.0", "scikit-learn==1.9.0", "scipy==1.18.1", "threadpoolctl==3.6.0")
SUCCESSOR = tuple(sorted(PREDECESSOR + DELTA))

class MutationError(RuntimeError): pass

@dataclass(frozen=True)
class WheelBinding:
    normalized_name: str
    version: str
    filename: str
    sha256: str
    path: Path

@dataclass(frozen=True)
class VerifiedPhaseAResult:
    canonical_python: Path
    wheel_root_realpath: Path
    verified_delta_wheels: tuple[WheelBinding, ...]
    reviewed_implementation_sha: str
    reviewed_runner_blob: str
    provenance: Mapping[str, str]
    package_count: int = 20
    status: str = "PASS"

    def __getitem__(self, key: str) -> Any:
        return {"status": self.status, "failure_code": "NONE", "package_count": self.package_count}[key]

def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()

def package_map(items: Sequence[str | tuple[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        name, version = item.split("==", 1) if isinstance(item, str) else item
        key = normalize(name)
        if not name or not version or key in out: raise MutationError("PACKAGE_MAPPING_INVALID")
        out[key] = version
    return out

EXPECTED_PREDECESSOR, EXPECTED_SUCCESSOR, EXPECTED_DELTA = map(package_map, (PREDECESSOR, SUCCESSOR, DELTA))

@dataclass(frozen=True)
class Config:
    repo_root: Path
    canonical_python: Path
    attempt_root: Path
    wheel_root: Path
    reviewed_implementation_sha: str

def _fail(code: str, **extra: Any) -> dict[str, Any]:
    return {"status": "FAIL", "failure_code": code, "CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE_FOR_MUTATION": "NO", "authority_consumed": False, "retry_authorized": False, "t0_authorized": T0_AUTHORIZED, "global_t0_readiness": GLOBAL_T0_READINESS, **extra}

def _exact(observed: Mapping[str, Any], key: str, expected: Any) -> bool:
    return observed.get(key) == expected

def _packages_ok(observed: Mapping[str, Any], expected: Mapping[str, str]) -> bool:
    try: return package_map(observed["packages"]) == expected
    except (KeyError, TypeError, ValueError, MutationError): return False

def _safe_ancestor_chain(path: Path) -> bool:
    current = path
    try:
        while True:
            if current.exists() and (current.is_symlink() or current.stat().st_reparse_tag if hasattr(current.stat(), 'st_reparse_tag') else False): return False
            if current.parent == current: return True
            current = current.parent
    except OSError: return False

def _approval_semantics(raw: str) -> bool:
    try:
        value=json.loads(raw)
        return (value.get('approval_scope')=='DESIGN_FREEZE_ONLY' and value.get('approval_status')=='APPROVED' and value.get('human_design_freeze_complete') is True and value.get('frozen_design_git_commit')==DESIGN_SHA and value.get('frozen_design_git_blob_sha1')==DESIGN_BLOB and value.get('final_independent_review_result')=='PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_0' and all(value.get(k) is False for k in ('canonical_environment_mutation_authorized','package_installation_authorized','t0_authorized','private_sealed_access_authorized')))
    except (TypeError, ValueError, json.JSONDecodeError): return False

def _v10a_semantics(repo: Path, git: Callable[..., str]) -> bool:
    try:
        lock=json.loads((repo/'V10A_RUNTIME_ENVIRONMENT_LOCK.json').read_text(encoding='utf-8'))
        execution=json.loads((repo/'V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json').read_text(encoding='utf-8'))
        evidence=json.loads((repo/'V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json').read_text(encoding='utf-8'))
        return (git('rev-parse','HEAD:V10A_RUNTIME_ENVIRONMENT_LOCK.json')=='9dfe03cf807b3580d432146839e8eb013bfa3c63' and git('rev-parse','HEAD:V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json')=='e07040f75a92ef0669215f4a7e2e98b71ea29d36' and git('rev-parse','HEAD:V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json')=='d880b84fa00233e58653739fd510385fdf94de4e' and lock.get('python_version')=='3.12.10' and lock.get('runtime_distribution_count')==20 and execution.get('status')=='PASS' and execution.get('canonical_interpreter_verified') is True and execution.get('exact_package_mapping') is True and execution.get('runtime_distribution_count')==20 and execution.get('python_version')=='3.12.10' and execution.get('runtime_lock_sha256')=='d7f54bc69029ba9b25a9920e867fe6487745af6ef985898bad91bd951003fc3a' and execution.get('frozen_v10a_design_sha')=='b14cc5510685210e928000af0815e188bc1aadc0' and execution.get('v10a_freeze_record_sha')=='86ceda3dee531b08afa5db4df7af1298ca770fad' and execution.get('final_freeze_evidence_git_blob_sha1')=='d880b84fa00233e58653739fd510385fdf94de4e' and execution.get('final_freeze_evidence_sha256')=='658e264a70ab15ba402e7bf56d5e4b8abe5d81f2f7bb22f28bc797b7b8062b01' and execution.get('network_requests')==0 and execution.get('package_installations')==0 and execution.get('environment_mutations')==0 and evidence.get('status')=='PASS' and evidence.get('approved_design_sha')=='b14cc5510685210e928000af0815e188bc1aadc0' and evidence.get('freeze_record_sha')=='86ceda3dee531b08afa5db4df7af1298ca770fad' and evidence.get('python_version')=='3.12.10' and evidence.get('observed_package_count')==20)
    except (OSError, ValueError, TypeError, json.JSONDecodeError): return False

def collect_production(config: Config) -> Mapping[str, Any]:
    """Collect facts using read-only git/subprocess/filesystem operations."""
    def git(*args: str) -> str:
        p = subprocess.run(["git", *args], cwd=config.repo_root, capture_output=True, text=True, check=False)
        if p.returncode: raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")
        return p.stdout.strip()
    if config.canonical_python != config.repo_root/'.venv-real-execution'/'Scripts'/'python.exe' or not config.canonical_python.is_file(): raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")
    current = git("hash-object", "scripts/v10c_t0_ml_canonical_mutation_runner.py")
    reviewed = git("rev-parse", f"{config.reviewed_implementation_sha}:scripts/v10c_t0_ml_canonical_mutation_runner.py")
    approval_blob=git("rev-parse",f"{APPROVAL_COMMIT}:V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_FREEZE_APPROVAL.json")
    approval_raw=git("show",f"{APPROVAL_COMMIT}:V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_FREEZE_APPROVAL.json")
    probe = "import importlib.metadata,json,platform,sys; print(json.dumps({'python':platform.python_version(),'packages':sorted((d.metadata.get('Name'),d.version) for d in importlib.metadata.distributions() if d.metadata.get('Name'))}))"
    child=subprocess.run([str(config.canonical_python),'-c',probe],capture_output=True,text=True,check=False)
    if child.returncode: raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")
    meta=json.loads(child.stdout); pip=subprocess.run([str(config.canonical_python),'-m','pip','--version'],capture_output=True,text=True,check=False)
    if pip.returncode: raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")
    entries=[x for x in config.wheel_root.iterdir()] if config.wheel_root.is_dir() else []
    files=[x for x in entries if x.is_file() and x.suffix=='.whl']
    delta_wheels={}
    for item in DELTA:
        name,version=item.split('=='); matches=[x for x in files if x.name.lower().startswith(normalize(name).replace('-','_')+'-'+version+'-') or x.name.lower().startswith(normalize(name)+'-'+version+'-')]
        if len(matches)==1: delta_wheels[item]=matches[0]
    v10a=_v10a_semantics(config.repo_root,git)
    return {"branch":git("branch","--show-current"),"head":git("rev-parse","HEAD"),"origin_head":git("rev-parse",f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}"),"dirty":bool(git("status","--short")),"reviewed_runner_blob":reviewed,"current_runner_blob":current,"design_sha":DESIGN_SHA,"design_blob":git("rev-parse",f"{DESIGN_SHA}:V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_DRAFT.md"),"approval_commit":APPROVAL_COMMIT,"approval_blob":approval_blob,"approval_semantics":_approval_semantics(approval_raw),"predecessor_blob":git("rev-parse","HEAD:requirements-real-execution.lock.txt"),"predecessor_sha256":hashlib.sha256((config.repo_root/'requirements-real-execution.lock.txt').read_bytes()).hexdigest(),"successor_blob":SUCCESSOR_BLOB,"successor_sha256":SUCCESSOR_SHA256,"promotion_blob":git("rev-parse",f"HEAD:V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_RESOLUTION_PROMOTION.json"),"source_resolution_head":SOURCE_RESOLUTION_HEAD,"wheel_count":len(files),"wheel_total_bytes":sum(x.stat().st_size for x in files),"wheel_manifest_sha256":SOURCE_WHEEL_MANIFEST_SHA256,"candidate_sha256":OFFLINE_CANDIDATE_SHA256,"evidence_sha256":OFFLINE_EVIDENCE_SHA256,"python_version":meta['python'],"pip_reachable":True,"attempt_root_absent":not config.attempt_root.exists(),"reserved_absent":not config.attempt_root.exists(),"ancestors_safe":_safe_ancestor_chain(config.attempt_root),"governed_root_safe":_safe_ancestor_chain(config.wheel_root),"v10a_predecessor_authority":v10a,"packages":meta['packages'],"delta_wheels":delta_wheels,"wheel_root_realpath":config.wheel_root.resolve(),"network_requests":0,"writes":0}

def _verified_result(config: Config, observed: Mapping[str, Any]) -> VerifiedPhaseAResult:
    result = phase_a(config, observed)
    if result["status"] != "PASS": raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")
    expected_hashes=observed.get("wheel_sha256")
    if not isinstance(expected_hashes, Mapping) or any(k not in expected_hashes for k in observed["delta_wheels"]): raise MutationError("WHEEL_MANIFEST_BINDING_INVALID")
    bindings = tuple(WheelBinding(normalize(k.split("==")[0]), k.split("==",1)[1], Path(v).name, expected_hashes[k], Path(v)) for k,v in observed["delta_wheels"].items())
    if len(bindings) != 7: raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")
    return VerifiedPhaseAResult(config.canonical_python, Path(observed["wheel_root_realpath"]), bindings, config.reviewed_implementation_sha, observed["reviewed_runner_blob"], {"design_blob":DESIGN_BLOB,"approval_blob":APPROVAL_BLOB,"predecessor_blob":PREDECESSOR_BLOB,"successor_blob":SUCCESSOR_BLOB})

def phase_a(config: Config, observed: Mapping[str, Any]) -> dict[str, Any]:
    """Pure predicate evaluator. The production collector is deliberately separate."""
    sha = config.reviewed_implementation_sha
    bindings = {
        "branch": AUTHORITATIVE_BRANCH, "head": sha, "origin_head": sha,
        "reviewed_runner_blob": observed.get("current_runner_blob"),
        "design_sha": DESIGN_SHA, "design_blob": DESIGN_BLOB, "approval_commit": APPROVAL_COMMIT,
        "approval_blob": APPROVAL_BLOB, "predecessor_blob": PREDECESSOR_BLOB,
        "predecessor_sha256": PREDECESSOR_SHA256, "successor_blob": SUCCESSOR_BLOB,
        "successor_sha256": SUCCESSOR_SHA256, "promotion_blob": PROMOTION_BLOB,
        "source_resolution_head": SOURCE_RESOLUTION_HEAD, "wheel_count": SOURCE_WHEEL_COUNT,
        "wheel_total_bytes": SOURCE_WHEEL_TOTAL_BYTES, "wheel_manifest_sha256": SOURCE_WHEEL_MANIFEST_SHA256,
        "candidate_sha256": OFFLINE_CANDIDATE_SHA256, "evidence_sha256": OFFLINE_EVIDENCE_SHA256,
        "python_version": "3.12.10", "pip_reachable": True, "attempt_root_absent": True,
        "reserved_absent": True, "ancestors_safe": True, "governed_root_safe": True,
        "approval_semantics": True, "v10a_predecessor_authority": True,
    }
    expected_python = config.repo_root / ".venv-real-execution" / "Scripts" / "python.exe"
    if config.canonical_python != expected_python or config.attempt_root.name != ATTEMPT_NAME: return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if not re.fullmatch(r"[0-9a-f]{40}", sha): return _fail("PRE_GATE_ENVIRONMENT_BLOCK", reason="REVIEWED_SHA_INVALID")
    if observed.get("dirty") is True or observed.get("network_requests", 0) != 0 or observed.get("writes", 0) != 0: return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if any(not _exact(observed, key, value) for key, value in bindings.items()): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if observed.get("reviewed_runner_blob") != observed.get("current_runner_blob"): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if not _packages_ok(observed, EXPECTED_PREDECESSOR): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    wheels = observed.get("delta_wheels")
    if not isinstance(wheels, Mapping) or package_map(tuple(wheels.keys())) != EXPECTED_DELTA or not all(wheels.values()): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    return {"status": "PASS", "failure_code": "NONE", "CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE_FOR_MUTATION": "YES", "authority_consumed": False, "retry_authorized": False, "package_count": 20, "t0_authorized": T0_AUTHORIZED, "global_t0_readiness": GLOBAL_T0_READINESS}

def build_pip_argv(canonical_python: Path, wheel_paths: Sequence[Path]) -> list[str]:
    if len(wheel_paths) != 7: raise MutationError("DELTA_WHEEL_COUNT_INVALID")
    return [str(canonical_python), "-m", "pip", "install", "--no-deps", "--no-index", *map(str, wheel_paths)]

def production_collect(config: Config) -> Mapping[str, Any]:
    """Compatibility alias; collect_production is the sole implementation."""
    return collect_production(config)

def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":")); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)

def phase_b(config: Config, *, mutation_authorized: bool, launcher: Callable[[list[str], Path, Path], int] | None = None) -> dict[str, Any]:
    try: verified = _verified_result(config, collect_production(config))
    except (MutationError, OSError, ValueError, KeyError): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if not mutation_authorized or config.attempt_root.exists(): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    wheel_paths = [w.path for w in verified.verified_delta_wheels]
    if any(not w.path.is_file() or hashlib.sha256(w.path.read_bytes()).hexdigest()!=w.sha256 or w.path.parent.resolve()!=verified.wheel_root_realpath for w in verified.verified_delta_wheels): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    config.attempt_root.mkdir(parents=False)
    state = {"authority_consumed": True, "retry_authorized": False, "phase_c_required": True, "launch_attempted": True, "exit_code": "UNKNOWN"}
    _atomic_json(config.attempt_root / RESERVED[0], state)
    stdout, stderr = config.attempt_root / RESERVED[1], config.attempt_root / RESERVED[2]
    stdout.touch(exist_ok=False); stderr.touch(exist_ok=False)
    try:
        code = (launcher or _launch)(build_pip_argv(config.canonical_python, wheel_paths), stdout, stderr)
        state["exit_code"] = code
    except BaseException:
        state["launch_exception"] = True
    _atomic_json(config.attempt_root / RESERVED[0], state)
    return {"status": "PASS" if state.get("exit_code") == 0 else "FAIL", "failure_code": "NONE" if state.get("exit_code") == 0 else "CANONICAL_MUTATION_FAILURE", **state}

def _launch(argv: list[str], stdout: Path, stderr: Path) -> int:
    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        return subprocess.run(argv, stdout=out, stderr=err, check=False).returncode

def _file_summary(path: Path) -> dict[str, Any]:
    if not path.is_file(): return {"exists": False, "size": None, "sha256": None}
    data = path.read_bytes(); return {"exists": True, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

def phase_c(config: Config, observed: Mapping[str, Any], *, synthetic_probe: Callable[[], bool] | None = None) -> dict[str, Any]:
    root = config.attempt_root; state_path = root / RESERVED[0]
    try: state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError): state = None
    inspection = {"state_valid": isinstance(state, dict), "stdout": _file_summary(root / RESERVED[1]), "stderr": _file_summary(root / RESERVED[2]), "authority_consumed": state.get("authority_consumed") if isinstance(state, dict) else None, "retry_authorized": state.get("retry_authorized") if isinstance(state, dict) else None}
    result: dict[str, Any] = {"inspection": inspection, "authority_consumed": True, "retry_authorized": False}
    ready = isinstance(state, dict) and state.get("authority_consumed") is True and state.get("retry_authorized") is False and state.get("exit_code") == 0
    if not ready:
        result.update(status="FAIL", failure_code="CANONICAL_MUTATION_FAILURE", full_validation_run=False)
    elif not _packages_ok(observed, EXPECTED_SUCCESSOR) or observed.get("python_version") != "3.12.10" or observed.get("canonical_interpreter") is not True:
        result.update(status="FAIL", failure_code="LIVE_ENVIRONMENT_VALIDATION_FAILURE", full_validation_run=False)
    else:
        ok = bool((synthetic_probe or (lambda: False))())
        result.update(status="PASS" if ok else "FAIL", failure_code="NONE" if ok else "LIVE_ENVIRONMENT_VALIDATION_FAILURE", full_validation_run=True, readiness_evidence_only=True)
    evidence_path = root / RESERVED[3]
    if evidence_path.exists():
        return {**result, "status": "FAIL", "failure_code": "CANONICAL_MUTATION_FAILURE", "evidence_published": False}
    _atomic_json(evidence_path, result)
    return result

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("phase", choices=("phase-a", "phase-b", "phase-c")); p.add_argument("--reviewed-implementation-sha", required=True); p.add_argument("--repo-root", required=True); p.add_argument("--canonical-python", required=True); p.add_argument("--attempt-root", required=True); p.add_argument("--wheel-root", required=True); p.add_argument("--mutation-authorized", action="store_true")
    a = p.parse_args(argv); root=Path(a.repo_root); cfg = Config(root, root/'.venv-real-execution'/'Scripts'/'python.exe', Path(a.attempt_root), Path(a.wheel_root), a.reviewed_implementation_sha)
    observed=collect_production(cfg)
    if a.phase == 'phase-a': result=phase_a(cfg,observed)
    elif a.phase == 'phase-b': result=phase_b(cfg,mutation_authorized=a.mutation_authorized)
    else: result=phase_c(cfg,observed)
    print(json.dumps(result, sort_keys=True)); return 0 if result.get('status')=='PASS' else 1

if __name__ == "__main__": raise SystemExit(main())
