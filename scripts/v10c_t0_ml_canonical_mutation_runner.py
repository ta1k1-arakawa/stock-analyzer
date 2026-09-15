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
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
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
        if isinstance(item, Mapping):
            _require(set(item) == {"name", "version"})
            name, version = item["name"], item["version"]
        else:
            name, version = item.split("==", 1) if isinstance(item, str) else item
        if not isinstance(name, str) or not isinstance(version, str):
            raise MutationError("PACKAGE_MAPPING_INVALID")
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
    """Inspect every existing component with lstat, including dangling links."""
    if not path.is_absolute() or ".." in path.parts:
        return False
    try:
        for node in (path, *path.parents):
            try:
                info = node.lstat()
            except FileNotFoundError:
                if os.path.lexists(node):
                    return False
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                return False
            if node != path and not stat.S_ISDIR(info.st_mode):
                return False
        return True
    except OSError:
        return False


def _require(condition: bool) -> None:
    if not condition:
        raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")


def _json_object(raw: bytes | str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result)
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    _require(isinstance(value, dict))
    return value


def _fields(value: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    # bool and int are deliberately not interchangeable.
    return all(type(value.get(key)) is type(want) and value[key] == want
               for key, want in expected.items())


def _artifact(repo: Path, git: Callable[..., str], name: str,
              blob: str, sha256: str | None = None) -> tuple[bytes, str, str]:
    path = repo / name
    _require(_safe_ancestor_chain(path) and stat.S_ISREG(path.lstat().st_mode))
    raw = path.read_bytes()
    actual_blob = git("rev-parse", f"HEAD:{name}")
    # Bind BOTH current Git identity and actual working bytes, even if status lies.
    working_blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    actual_sha = hashlib.sha256(raw).hexdigest()
    _require(actual_blob == blob == working_blob)
    _require(sha256 is None or actual_sha == sha256)
    return raw, actual_blob, actual_sha


def _approval_semantics(raw: bytes | str) -> bool:
    try:
        value = _json_object(raw)
        expected = {
            "schema_version": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_FREEZE_APPROVAL_V1",
            "study": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR",
            "artifact_role": "MUTATION_DESIGN_FREEZE_APPROVAL",
            "approval_scope": "DESIGN_FREEZE_ONLY", "approval_status": "APPROVED",
            "human_design_freeze_complete": True,
            "frozen_design_git_commit": DESIGN_SHA,
            "frozen_design_git_blob_sha1": DESIGN_BLOB,
            "final_independent_review_result": "PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_0",
            "final_independent_review_design_commit": DESIGN_SHA,
            "approval_record_gpt_exact_sha_pass_required_for_mutation_implementation_preparation": True,
        }
        expected.update({key: False for key in (
            "canonical_environment_mutation_authorized", "package_installation_authorized",
            "t0_authorized", "private_sealed_access_authorized", "future_profitability_established",
            "training_payload_read_authorized", "evaluation_payload_read_authorized",
            "model_fit_authorized", "historical_evaluation_authorized",
            "package_index_network_access_authorized")})
        return _fields(value, expected)
    except (TypeError, ValueError, MutationError):
        return False


V10A_LOCK_SHA256 = "d7f54bc69029ba9b25a9920e867fe6487745af6ef985898bad91bd951003fc3a"
V10A_FREEZE_SHA256 = "658e264a70ab15ba402e7bf56d5e4b8abe5d81f2f7bb22f28bc797b7b8062b01"
V10A_FREEZE_BLOB = "d880b84fa00233e58653739fd510385fdf94de4e"
V10A_DESIGN_SHA = "b14cc5510685210e928000af0815e188bc1aadc0"
V10A_FREEZE_SHA = "86ceda3dee531b08afa5db4df7af1298ca770fad"
SOURCE_PROVENANCE_BLOB = "55d8705d33d316fc0aef2103850db1f025307870"


def _v10a_semantics(repo: Path, git: Callable[..., str]) -> bool:
    try:
        lock = _json_object(_artifact(repo, git, "V10A_RUNTIME_ENVIRONMENT_LOCK.json",
            "9dfe03cf807b3580d432146839e8eb013bfa3c63", V10A_LOCK_SHA256)[0])
        execution = _json_object(_artifact(repo, git, "V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json",
            "e07040f75a92ef0669215f4a7e2e98b71ea29d36")[0])
        evidence = _json_object(_artifact(repo, git,
            "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json",
            V10A_FREEZE_BLOB, V10A_FREEZE_SHA256)[0])
        return (
            _fields(lock, {"python_version": "3.12.10", "runtime_distribution_count": 20})
            and package_map(lock["runtime_distributions"]) == EXPECTED_PREDECESSOR
            and _fields(execution, {
                "status": "PASS", "failure_code": "NONE",
                "canonical_interpreter_verified": True, "exact_package_mapping": True,
                "runtime_distribution_count": 20, "python_version": "3.12.10",
                "runtime_lock_sha256": V10A_LOCK_SHA256,
                "frozen_v10a_design_sha": V10A_DESIGN_SHA,
                "v10a_freeze_record_sha": V10A_FREEZE_SHA,
                "final_freeze_evidence_git_blob_sha1": V10A_FREEZE_BLOB,
                "final_freeze_evidence_sha256": V10A_FREEZE_SHA256,
                "network_requests": 0, "package_installations": 0, "environment_mutations": 0,
            })
            and _fields(evidence, {
                "status": "PASS", "failure_code": "NONE",
                "approved_design_sha": V10A_DESIGN_SHA, "freeze_record_sha": V10A_FREEZE_SHA,
                "python_version": "3.12.10", "observed_package_count": 20,
                "package_index_network_requests": 0, "package_installations": 0,
                "environment_mutations": 0,
            })
            and package_map(evidence["observed_packages"]) == EXPECTED_PREDECESSOR
        )
    except (OSError, ValueError, TypeError, KeyError, MutationError):
        return False


def _promotion_semantics(value: Mapping[str, Any]) -> bool:
    expected = {
        "schema_version": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_RESOLUTION_PROMOTION_V1",
        "study": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR",
        "artifact_role": "RESOLUTION_PROMOTION_RECORD",
        "promotion_status_at_creation": "AWAITING_GPT_EXACT_SHA_REVIEW",
        "promotion_effective_only_after_gpt_exact_sha_pass": True,
        "source_resolution_head": SOURCE_RESOLUTION_HEAD,
        "source_resolution_runner_blob_sha1": "4a941e4631174c3dc56a1623f2eb270045832905",
        "source_wheel_count": SOURCE_WHEEL_COUNT,
        "source_wheel_total_bytes": SOURCE_WHEEL_TOTAL_BYTES,
        "source_wheel_manifest_sha256": SOURCE_WHEEL_MANIFEST_SHA256,
        "offline_readjudication_evidence_sha256": OFFLINE_EVIDENCE_SHA256,
        "offline_readjudication_result": "PASS",
        "successor_lock_candidate_sha256": OFFLINE_CANDIDATE_SHA256,
        "successor_lock_sha256": SUCCESSOR_SHA256,
        "successor_lock_package_count": 27, "predecessor_package_count": 20,
        "successor_delta_package_count": 7, "successor_delta_packages": list(DELTA),
        "successor_lock_file": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK.txt",
        "resolution_authority_consumed": True, "resolution_retry_authorized": False,
        "source_phase_b_process_exit_code": 0, "source_phase_b_resolution_invocations": 1,
        "source_phase_b_authority_consumed": True,
        "original_phase_c_result": "FAIL", "original_phase_c_failure_code": "RESOLUTION_REPORT_INVALID",
        "offline_readjudication_validator_git_sha": "a70525c54e5c294f1f0052565a09127580ee4ee0",
        "offline_readjudication_runner_git_blob_sha1": "0fcec134c8392fc89b31d6c673d140c177d2185d",
        "offline_readjudication_contract_git_blob_sha1": "2b9722774ef77a66137f7396ea91075eb1fc97e0",
        "candidate_reviewed_resolution_implementation_git_sha": SOURCE_RESOLUTION_HEAD,
        "next_required_action": "GPT_EXACT_SHA_V10C_T0_ML_RESOLUTION_PROMOTION_REVIEW",
    }
    expected.update({key: False for key in (
        "canonical_environment_mutation_authorized", "package_installation_authorized",
        "t0_authorized", "private_sealed_access_authorized", "future_profitability_established",
        "historical_evaluation_authorized", "successor_lock_install_authority_established")})
    return _fields(value, expected)


def _canonical_identity(config: Config) -> Path:
    expected = config.repo_root / ".venv-real-execution" / "Scripts" / "python.exe"
    _require(config.canonical_python == expected and _safe_ancestor_chain(expected))
    _require(os.path.lexists(expected) and stat.S_ISREG(expected.lstat().st_mode))
    resolved = expected.resolve(strict=True)
    _require(resolved == expected and not resolved.is_relative_to(config.repo_root / ".venv"))
    return resolved


def _namespace_safety(config: Config) -> tuple[Path, Path]:
    _require(config.attempt_root.name == ATTEMPT_NAME)
    for path in (config.repo_root, config.wheel_root, config.attempt_root):
        _require(_safe_ancestor_chain(path))
    repo = config.repo_root.resolve(strict=True)
    wheel = config.wheel_root.resolve(strict=True)
    _require(repo.is_dir() and wheel.is_dir())
    _require(not os.path.lexists(config.attempt_root))
    _require(all(not os.path.lexists(config.attempt_root / name) for name in RESERVED))
    # The immediate parent must exist; Phase B may not create a new parent chain.
    config.attempt_root.parent.resolve(strict=True)
    attempt = config.attempt_root.resolve(strict=False)
    _require(attempt == config.attempt_root)
    _require(not attempt.is_relative_to(repo) and not repo.is_relative_to(attempt))
    _require(not attempt.is_relative_to(wheel) and not wheel.is_relative_to(attempt))
    _require(not config.canonical_python.is_relative_to(attempt))
    # The reviewed source wheelhouse is a child of a governed resolution attempt.
    # Exclude that entire source namespace, without opening its state/payloads.
    if wheel.name == "wheelhouse":
        source = wheel.parent
        _require(not attempt.is_relative_to(source) and not source.is_relative_to(attempt))
    return wheel, attempt


def collect_production(config: Config) -> Mapping[str, Any]:
    """Observe actual public bytes and read-only machine facts; never mint GPT authority."""
    def git(*args: str) -> str:
        p = subprocess.run(["git", *args], cwd=config.repo_root,
                           capture_output=True, text=True, check=False,
                           env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
        _require(p.returncode == 0)
        return p.stdout.strip()

    from scripts.v10c_t0_ml_environment_contract import inspect_wheelhouse
    from scripts.v10c_t0_ml_environment_offline_readjudication_runner import (
        SOURCE_PROVENANCE_EXPECTED, _repo_identity_matches, _validate_wheelhouse_manifest,
    )
    sha = config.reviewed_implementation_sha
    _require(isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha) is not None)
    _require(_safe_ancestor_chain(config.repo_root))
    _require(_repo_identity_matches(git("config", "--get", "remote.origin.url")))
    branch = git("branch", "--show-current")
    head = git("rev-parse", "HEAD")
    origin = git("rev-parse", f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}")
    commit = git("rev-parse", "--verify", f"{sha}^{{commit}}")
    dirty = bool(git("status", "--porcelain=v1", "--untracked-files=all"))
    current = git("hash-object", "scripts/v10c_t0_ml_canonical_mutation_runner.py")
    reviewed = git("rev-parse", f"{sha}:scripts/v10c_t0_ml_canonical_mutation_runner.py")
    _require(branch == AUTHORITATIVE_BRANCH and head == origin == commit == sha and not dirty)
    _require(re.fullmatch(r"[0-9a-f]{40}", current) is not None and current == reviewed)
    design_name = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_DRAFT.md"
    approval_name = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_DESIGN_FREEZE_APPROVAL.json"
    design_commit = git("rev-parse", "--verify", f"{DESIGN_SHA}^{{commit}}")
    design_blob = git("rev-parse", f"{DESIGN_SHA}:{design_name}")
    approval_commit = git("rev-parse", "--verify", f"{APPROVAL_COMMIT}^{{commit}}")
    approval_blob = git("rev-parse", f"{APPROVAL_COMMIT}:{approval_name}")
    _require(design_commit == DESIGN_SHA and design_blob == DESIGN_BLOB)
    _require(approval_commit == APPROVAL_COMMIT and approval_blob == APPROVAL_BLOB)
    _artifact(config.repo_root, git, design_name, DESIGN_BLOB)
    approval_raw = _artifact(config.repo_root, git, approval_name, APPROVAL_BLOB)[0]
    _require(_approval_semantics(approval_raw))
    pred_raw, pred_blob, pred_sha = _artifact(config.repo_root, git,
        "requirements-real-execution.lock.txt", PREDECESSOR_BLOB, PREDECESSOR_SHA256)
    succ_raw, succ_blob, succ_sha = _artifact(config.repo_root, git,
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK.txt", SUCCESSOR_BLOB, SUCCESSOR_SHA256)
    predecessor = package_map(pred_raw.decode("utf-8").splitlines())
    successor = package_map(succ_raw.decode("utf-8").splitlines())
    _require(predecessor == EXPECTED_PREDECESSOR and successor == EXPECTED_SUCCESSOR)
    _require({k: v for k, v in successor.items() if k not in predecessor} == EXPECTED_DELTA)
    promotion_raw, promotion_blob, _ = _artifact(config.repo_root, git,
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_RESOLUTION_PROMOTION.json", PROMOTION_BLOB)
    promotion = _json_object(promotion_raw)
    _require(_promotion_semantics(promotion))
    source_raw, source_blob, _ = _artifact(config.repo_root, git,
        "V10C_T0_ML_RESOLUTION_SOURCE_PROVENANCE.json", SOURCE_PROVENANCE_BLOB)
    source = _json_object(source_raw)
    _require(set(source) == set(SOURCE_PROVENANCE_EXPECTED) and _fields(source, SOURCE_PROVENANCE_EXPECTED))
    v10a_verified = _v10a_semantics(config.repo_root, git)
    _require(v10a_verified)
    canonical = _canonical_identity(config)
    wheel_root, checked_attempt = _namespace_safety(config)
    probe = (
        "import importlib.metadata,json,platform,sys; "
        "print(json.dumps({'python':platform.python_version(),'executable':sys.executable,"
        "'packages':[(d.metadata.get('Name'),d.version) for d in importlib.metadata.distributions()]}))"
    )
    child_env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                 "PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    child = subprocess.run([str(canonical), "-B", "-I", "-c", probe],
                           capture_output=True, text=True, check=False, env=child_env)
    _require(child.returncode == 0)
    meta = _json_object(child.stdout)
    _require(meta.get("python") == "3.12.10" and Path(meta["executable"]) == canonical)
    _require(package_map(meta["packages"]) == predecessor)
    pip = subprocess.run([str(canonical), "-B", "-I", "-m", "pip", "--version"],
                         capture_output=True, text=True, check=False, env=child_env)
    _require(pip.returncode == 0 and bool(pip.stdout.strip()))
    entries = list(wheel_root.iterdir())
    _require(len(entries) == SOURCE_WHEEL_COUNT)
    for entry in entries:
        _require(_safe_ancestor_chain(entry) and stat.S_ISREG(entry.lstat().st_mode)
                 and entry.suffix.lower() == ".whl")
    wheel_code, wheels = inspect_wheelhouse(wheel_root)
    _require(wheel_code == "NONE" and wheels is not None)
    # This reviewed validator computes CANONICAL_WHEEL_MANIFEST_V1 and compares
    # its hash to the frozen source hash. It returns None, not an observed hash.
    _validate_wheelhouse_manifest(wheel_root, wheels)
    _require(package_map([(w["name"], w["version"]) for w in wheels]) == successor)
    _require({w["filename"] for w in wheels} == {p.name for p in entries})
    by_identity = {(w["name"], w["version"]): w for w in wheels}
    delta_wheels, hashes = {}, {}
    for pin in DELTA:
        name, version = pin.split("==")
        item = by_identity[(name, version)]
        _require(Path(item["filename"]).name == item["filename"]
                 and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None)
        delta_wheels[pin] = wheel_root / item["filename"]
        hashes[pin] = item["sha256"]
    return {
        "branch": branch, "head": head, "origin_head": origin, "dirty": dirty,
        "reviewed_runner_blob": reviewed, "current_runner_blob": current,
        "design_sha": design_commit, "design_blob": design_blob,
        "approval_commit": approval_commit, "approval_blob": approval_blob,
        "approval_semantics": _approval_semantics(approval_raw),
        "predecessor_blob": pred_blob, "predecessor_sha256": pred_sha,
        "successor_blob": succ_blob, "successor_sha256": succ_sha,
        "promotion_blob": promotion_blob, "source_provenance_blob": source_blob,
        "source_resolution_head": source["source_resolution_head"],
        "wheel_count": len(entries), "wheel_total_bytes": sum(p.stat().st_size for p in entries),
        "wheel_manifest_verified": True,
        "candidate_sha256": promotion["successor_lock_candidate_sha256"],
        "evidence_sha256": promotion["offline_readjudication_evidence_sha256"],
        "python_version": meta["python"], "pip_reachable": pip.returncode == 0,
        "attempt_root_absent": not os.path.lexists(config.attempt_root),
        "reserved_absent": all(not os.path.lexists(config.attempt_root / n) for n in RESERVED),
        "ancestors_safe": _safe_ancestor_chain(config.attempt_root),
        "governed_root_safe": checked_attempt == config.attempt_root,
        "v10a_predecessor_authority": v10a_verified,
        "packages": meta["packages"], "delta_wheels": delta_wheels, "wheel_sha256": hashes,
        "wheel_root_realpath": wheel_root, "network_requests": 0, "writes": 0,
    }

def _verified_result(config: Config, observed: Mapping[str, Any]) -> VerifiedPhaseAResult:
    result = phase_a(config, observed)
    if result["status"] != "PASS": raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")
    expected_hashes=observed.get("wheel_sha256")
    if not isinstance(expected_hashes, Mapping) or any(k not in expected_hashes for k in observed["delta_wheels"]): raise MutationError("WHEEL_MANIFEST_BINDING_INVALID")
    bindings = tuple(WheelBinding(normalize(k.split("==")[0]), k.split("==",1)[1], Path(v).name, expected_hashes[k], Path(v)) for k,v in observed["delta_wheels"].items())
    if len(bindings) != 7: raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")
    return VerifiedPhaseAResult(config.canonical_python, Path(observed["wheel_root_realpath"]), bindings, config.reviewed_implementation_sha, observed["reviewed_runner_blob"], MappingProxyType({"design_blob":DESIGN_BLOB,"approval_blob":APPROVAL_BLOB,"predecessor_blob":PREDECESSOR_BLOB,"successor_blob":SUCCESSOR_BLOB}))

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
        "wheel_total_bytes": SOURCE_WHEEL_TOTAL_BYTES, "wheel_manifest_verified": True,
        "candidate_sha256": OFFLINE_CANDIDATE_SHA256, "evidence_sha256": OFFLINE_EVIDENCE_SHA256,
        "python_version": "3.12.10", "pip_reachable": True, "attempt_root_absent": True,
        "reserved_absent": True, "ancestors_safe": True, "governed_root_safe": True,
        "approval_semantics": True, "v10a_predecessor_authority": True,
    }
    expected_python = config.repo_root / ".venv-real-execution" / "Scripts" / "python.exe"
    if config.canonical_python != expected_python or config.attempt_root.name != ATTEMPT_NAME: return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if not re.fullmatch(r"[0-9a-f]{40}", sha): return _fail("PRE_GATE_ENVIRONMENT_BLOCK", reason="REVIEWED_SHA_INVALID")
    if observed.get("dirty") is not False or observed.get("network_requests", 0) != 0 or observed.get("writes", 0) != 0: return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if any(not _exact(observed, key, value) for key, value in bindings.items()): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if not re.fullmatch(r"[0-9a-f]{40}", str(observed.get("current_runner_blob"))) or observed.get("reviewed_runner_blob") != observed.get("current_runner_blob"): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if not _packages_ok(observed, EXPECTED_PREDECESSOR): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    wheels = observed.get("delta_wheels")
    if not isinstance(wheels, Mapping) or package_map(tuple(wheels.keys())) != EXPECTED_DELTA or not all(wheels.values()): return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    return {"status": "PASS", "failure_code": "NONE", "CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE_FOR_MUTATION": "YES", "authority_consumed": False, "retry_authorized": False, "package_count": 20, "t0_authorized": T0_AUTHORIZED, "global_t0_readiness": GLOBAL_T0_READINESS}

def build_pip_argv(verified: VerifiedPhaseAResult) -> list[str]:
    if len(verified.verified_delta_wheels) != 7: raise MutationError("DELTA_WHEEL_COUNT_INVALID")
    return [str(verified.canonical_python), "-m", "pip", "install", "--no-deps", "--no-index", *map(lambda item: str(item.path), verified.verified_delta_wheels)]

def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":")); handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)

def _prelaunch_wheels(verified: VerifiedPhaseAResult) -> None:
    for wheel in verified.verified_delta_wheels:
        try:
            info = wheel.path.lstat()
            resolved = wheel.path.resolve(strict=True)
            digest = hashlib.sha256(wheel.path.read_bytes()).hexdigest()
        except OSError as error:
            raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK") from error
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & 0x400
            or not _safe_ancestor_chain(wheel.path)
            or resolved.parent != verified.wheel_root_realpath
            or wheel.path.name != wheel.filename
            or EXPECTED_DELTA.get(wheel.normalized_name) != wheel.version
            or digest != wheel.sha256
        ):
            raise MutationError("PRE_GATE_ENVIRONMENT_BLOCK")


def _post_boundary_result(config: Config, phase_b_result: Mapping[str, Any]) -> dict[str, Any]:
    """Run the dedicated safe observer exactly once after the boundary."""
    phase_c_result = phase_c(config)
    return {
        "status": phase_c_result.get("status", "FAIL"),
        "failure_code": phase_c_result.get("failure_code", "CANONICAL_MUTATION_FAILURE"),
        "authority_consumed": True,
        "retry_authorized": False,
        "phase_b_result": dict(phase_b_result),
        "phase_c_result": phase_c_result,
    }


def phase_b(
    config: Config,
    *,
    mutation_authorized: bool,
    launcher: Callable[[list[str], Path, Path], int] | None = None,
) -> dict[str, Any]:
    try:
        verified = _verified_result(config, collect_production(config))
        _prelaunch_wheels(verified)
    except (MutationError, OSError, ValueError, KeyError):
        return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if not mutation_authorized or os.path.lexists(config.attempt_root):
        return _fail("PRE_GATE_ENVIRONMENT_BLOCK")

    try:
        config.attempt_root.mkdir(parents=False, exist_ok=False)
    except (FileExistsError, OSError):
        return _fail("PRE_GATE_ENVIRONMENT_BLOCK")

    state = {
        "authority_consumed": True,
        "retry_authorized": False,
        "phase_c_required": True,
        "launch_attempted": False,
        "process_started": False,
        "exit_code": "UNKNOWN",
    }
    # The publication attempt is the sticky boundary. Any later exception
    # must go through Phase C and may never enter a retry path.
    try:
        _atomic_json(config.attempt_root / RESERVED[0], state)
    except BaseException:
        return _post_boundary_result(
            config,
            {"status": "FAIL", "failure_code": "CANONICAL_MUTATION_FAILURE", **state},
        )

    stdout = config.attempt_root / RESERVED[1]
    stderr = config.attempt_root / RESERVED[2]
    phase_b_failure: str | None = None
    try:
        stdout.touch(exist_ok=False)
        stderr.touch(exist_ok=False)
        state["launch_attempted"] = True
        state["process_started"] = "UNKNOWN"
        _atomic_json(config.attempt_root / RESERVED[0], state)
        try:
            code = (launcher or _launch)(build_pip_argv(verified), stdout, stderr)
            state["process_started"] = True
            state["exit_code"] = code if isinstance(code, int) else "UNKNOWN"
            if state["exit_code"] != 0:
                phase_b_failure = "CANONICAL_MUTATION_FAILURE"
        except BaseException:
            state["launch_exception"] = True
            state["process_started"] = "UNKNOWN"
            phase_b_failure = "CANONICAL_MUTATION_FAILURE"
        try:
            _atomic_json(config.attempt_root / RESERVED[0], state)
        except BaseException:
            phase_b_failure = "CANONICAL_MUTATION_FAILURE"
    except BaseException:
        phase_b_failure = "CANONICAL_MUTATION_FAILURE"

    phase_b_result = {
        "status": "PASS" if phase_b_failure is None and state.get("exit_code") == 0 else "FAIL",
        "failure_code": phase_b_failure or "NONE",
        **state,
    }
    return _post_boundary_result(config, phase_b_result)

def _launch(argv: list[str], stdout: Path, stderr: Path) -> int:
    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        return subprocess.run(argv, stdout=out, stderr=err, check=False).returncode

def _safe_file_summary(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            return {"exists": True, "size": None, "sha256": None, "integrity": "FAIL"}
        data = path.read_bytes()
        return {"exists": True, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "integrity": "PASS"}
    except FileNotFoundError:
        return {"exists": False, "size": None, "sha256": None, "integrity": "MISSING"}
    except OSError:
        return {"exists": "UNKNOWN", "size": None, "sha256": None, "integrity": "UNKNOWN"}


def _read_mutation_state(config: Config) -> tuple[dict[str, Any] | None, bool]:
    try:
        state = _json_object((config.attempt_root / RESERVED[0]).read_bytes())
    except (OSError, TypeError, ValueError, MutationError):
        return None, False
    exit_code = state.get("exit_code")
    valid = (
        state.get("authority_consumed") is True
        and state.get("retry_authorized") is False
        and state.get("phase_c_required") is True
        and isinstance(state.get("launch_attempted"), bool)
        and state.get("process_started") in {True, False, "UNKNOWN"}
        and ((isinstance(exit_code, int) and not isinstance(exit_code, bool)) or exit_code == "UNKNOWN")
    )
    return state, valid


def _phase_c_probe_script() -> str:
    expected = repr(EXPECTED_SUCCESSOR)
    return f"""
import importlib.metadata, json, math, platform, sys
packages = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get('Name')
    version = distribution.version
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError('MALFORMED_METADATA')
    packages.append({{'name': name, 'version': version}})
result = {{
    'python_version': platform.python_version(),
    'executable': sys.executable,
    'packages': packages,
    'probe_status': 'NOT_RUN',
    'lightgbm_probe': False,
    'ridge_probe': False,
}}
try:
    names = {{item['name'].lower().replace('_', '-').replace('.', '-'): item['version'] for item in packages}}
    if names == {expected}:
        X = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
        y = [0.0, 1.0, 2.0, 3.0]
        from lightgbm import LGBMRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        lgbm = LGBMRegressor(n_estimators=1, random_state=0, n_jobs=1, verbosity=-1)
        lgbm.fit(X, y)
        lgbm_prediction = lgbm.predict([[1.5, 1.5]])
        scaler = StandardScaler()
        transformed = scaler.fit_transform(X)
        ridge = Ridge()
        ridge.fit(transformed, y)
        ridge_prediction = ridge.predict(scaler.transform([[1.5, 1.5]]))
        result['lightgbm_probe'] = len(lgbm_prediction) == 1 and math.isfinite(float(lgbm_prediction[0]))
        result['ridge_probe'] = len(ridge_prediction) == 1 and math.isfinite(float(ridge_prediction[0]))
        result['probe_status'] = 'PASS' if result['lightgbm_probe'] and result['ridge_probe'] else 'FAIL'
except Exception:
    result['probe_status'] = 'FAIL'
print(json.dumps(result, sort_keys=True))
"""


def _observe_phase_c_runtime(canonical: Path) -> tuple[dict[str, Any] | None, str]:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull}
    try:
        child = subprocess.run(
            [str(canonical), "-B", "-I", "-c", _phase_c_probe_script()],
            capture_output=True, text=True, check=False, env=env,
        )
        if child.returncode != 0:
            return None, "CANONICAL_MUTATION_FAILURE"
        return _json_object(child.stdout), "NONE"
    except (OSError, TypeError, ValueError, MutationError):
        return None, "CANONICAL_MUTATION_FAILURE"


def _publish_phase_c_evidence(config: Config, evidence: Mapping[str, Any]) -> bool:
    path = config.attempt_root / RESERVED[3]
    try:
        if os.path.lexists(path):
            return False
        _atomic_json(path, evidence)
        return True
    except (FileExistsError, OSError, ValueError, TypeError):
        return False


def phase_c(config: Config) -> dict[str, Any]:
    """Inspect one existing attempt without Phase-A, pip, or wheel-root access."""
    state, state_valid = _read_mutation_state(config)
    launch_attempted = state.get("launch_attempted") if state and isinstance(state.get("launch_attempted"), bool) else "UNKNOWN"
    process_started = state.get("process_started") if state and state.get("process_started") in {True, False, "UNKNOWN"} else "UNKNOWN"
    raw_exit_code = state.get("exit_code") if state else "UNKNOWN"
    process_exit_code = raw_exit_code if isinstance(raw_exit_code, int) and not isinstance(raw_exit_code, bool) else "UNKNOWN"
    inspection: dict[str, Any] = {
        "state_valid": state_valid,
        "authority_consumed": True,
        "retry_authorized": False,
        "launch_attempted": launch_attempted,
        "process_started": process_started,
        "process_exit_code": process_exit_code,
        "stdout": _safe_file_summary(config.attempt_root / RESERVED[1]),
        "stderr": _safe_file_summary(config.attempt_root / RESERVED[2]),
    }
    result: dict[str, Any] = {
        "inspection": inspection,
        "authority_consumed": True,
        "retry_authorized": False,
        "full_validation_run": False,
        "canonical_interpreter_status": "UNKNOWN",
        "live_package_observation_status": "NOT_RUN",
        "failure_class": "CANONICAL_MUTATION_FAILURE",
    }
    if not state_valid or state.get("exit_code") != 0:
        result.update(status="FAIL", failure_code="CANONICAL_MUTATION_FAILURE")
    else:
        try:
            canonical = _canonical_identity(config)
        except (MutationError, OSError, ValueError):
            result.update(status="FAIL", failure_code="CANONICAL_MUTATION_FAILURE")
        else:
            runtime, runtime_failure = _observe_phase_c_runtime(canonical)
            if runtime_failure != "NONE" or runtime is None:
                result.update(status="FAIL", failure_code="CANONICAL_MUTATION_FAILURE")
            else:
                try:
                    package_status = "PASS" if _packages_ok(runtime, EXPECTED_SUCCESSOR) else "FAIL"
                except (TypeError, ValueError, MutationError):
                    package_status = "FAIL"
                interpreter_ok = (
                    runtime.get("python_version") == "3.12.10"
                    and Path(runtime.get("executable", "")) == canonical
                )
                result["canonical_interpreter_status"] = "PASS" if interpreter_ok else "FAIL"
                result["live_package_observation_status"] = package_status
                result["python_version"] = runtime.get("python_version")
                result["package_count"] = len(runtime.get("packages", ())) if isinstance(runtime.get("packages"), list) else None
                result["probe_status"] = runtime.get("probe_status", "NOT_RUN")
                result["lightgbm_probe"] = runtime.get("lightgbm_probe") is True
                result["ridge_probe"] = runtime.get("ridge_probe") is True
                if not interpreter_ok or package_status != "PASS":
                    result.update(status="FAIL", failure_code="LIVE_ENVIRONMENT_VALIDATION_FAILURE", failure_class="LIVE_ENVIRONMENT_VALIDATION_FAILURE")
                elif result["probe_status"] != "PASS" or not result["lightgbm_probe"] or not result["ridge_probe"]:
                    result.update(status="FAIL", failure_code="LIVE_ENVIRONMENT_VALIDATION_FAILURE", failure_class="LIVE_ENVIRONMENT_VALIDATION_FAILURE")
                else:
                    result.update(status="PASS", failure_code="NONE", failure_class="PASS", full_validation_run=True, readiness_evidence_only=True)
    published = _publish_phase_c_evidence(config, result)
    result["evidence_published"] = published
    if not published and result.get("status") == "PASS":
        result.update(status="FAIL", failure_code="CANONICAL_MUTATION_FAILURE", failure_class="CANONICAL_MUTATION_FAILURE")
    return result

def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("phase", choices=("phase-a", "phase-b", "phase-c")); p.add_argument("--reviewed-implementation-sha", required=True); p.add_argument("--repo-root", required=True); p.add_argument("--canonical-python", required=True); p.add_argument("--attempt-root", required=True); p.add_argument("--wheel-root", required=True); p.add_argument("--mutation-authorized", action="store_true")
    a = p.parse_args(argv); root=Path(a.repo_root); cfg = Config(root, Path(a.canonical_python), Path(a.attempt_root), Path(a.wheel_root), a.reviewed_implementation_sha)
    try:
        if a.phase == 'phase-a': result=phase_a(cfg,collect_production(cfg))
        elif a.phase == 'phase-b': result=phase_b(cfg,mutation_authorized=a.mutation_authorized)
        else: result=phase_c(cfg)
    except (MutationError, OSError, ValueError, TypeError, KeyError, RuntimeError):
        result=_fail("PRE_GATE_ENVIRONMENT_BLOCK")
    print(json.dumps(result, sort_keys=True)); return 0 if result.get('status')=='PASS' else 1

if __name__ == "__main__": raise SystemExit(main())
