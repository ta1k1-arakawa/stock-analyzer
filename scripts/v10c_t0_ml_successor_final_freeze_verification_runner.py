"""V10C successor final-freeze verification tooling.

Importing this module performs no I/O.  The production CLI has only the
frozen phase-a/phase-b/phase-c entrypoints; synthetic tests replace private
observation and launch seams without exposing those seams through the CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
DESIGN_SHA = "3654dd9838f89dbac03bdd6f2232b5a0a2dd89bf"
DESIGN_BLOB = "fcfba59e4cf5e9fd4376d525d85f9cf0c02d3fc1"
APPROVAL_RECORD_SHA = "b99bebab75aad4800b088ae15e0a2383f8e4cf73"
APPROVAL_RECORD_BLOB = "286f2f6f7750769f626eb7da69ed55455955694e"
MUTATION_IMPLEMENTATION_SHA = "06d6c5b6498baed0591a522832bd1c804966d5ea"
MUTATION_RUNNER_BLOB = "24ac0d1cc6dfb68dffa2ff75cf73bfe682f76886"
PREDECESSOR_BLOB = "99395e7a5be752fb3ea92fd31be0334f38792261"
PREDECESSOR_SHA256 = "eb325ac5e3417e6407400b18c8d90ca734a32e852056926e5bcd2a635e43c444"
SUCCESSOR_BLOB = "13636e58fbe40071be04cbfa57c3990c1d8ff2e0"
SUCCESSOR_SHA256 = "f38dd4c7319465bb7e6ff429e8dff4a476d9966c744b19e50264dcc0b18e8300"
PROMOTION_BLOB = "b866e6d77508d6366569ee6a229587c59c3c8be2"
SOURCE_PROVENANCE_BLOB = "55d8705d33d316fc0aef2103850db1f025307870"
SOURCE_RESOLUTION_HEAD = "3aee6c2772f30c6dc35d2a7efb862ae15091febc"
SOURCE_WHEEL_COUNT = 27
SOURCE_WHEEL_TOTAL_BYTES = 94451528
SOURCE_WHEEL_MANIFEST_SHA256 = "5d5953f14b0609767972679554e1999e754621056d863f8c33def96988797b74"
OFFLINE_CANDIDATE_SHA256 = "893881cbb9612e3402b0f4e1e434edfc4283da81a0f6a6d264d019ae5573c48e"
OFFLINE_EVIDENCE_SHA256 = "b4398e32be354de03e64202148dc6933ed45ef888657e909de1f52a4a051206b"
MUTATION_EVIDENCE_SCHEMA = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_PHASE_C_EVIDENCE_V1"
FINAL_EVIDENCE_SCHEMA = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_VERIFICATION_EVIDENCE_V1"
STATE_SCHEMA = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_STATE_V1"
ATTEMPT_NAME = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_ATTEMPT_1"
MUTATION_ATTEMPT_NAME = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_MUTATION_ATTEMPT_1"
RESERVED = ("final_freeze_state.json", "final_freeze_stdout.txt", "final_freeze_stderr.txt", "final_freeze_evidence.json")

PREDECESSOR = (
    "cffi==2.1.1", "charset-normalizer==3.5.1", "cryptography==50.0.1", "exchange-calendars==4.13.2",
    "korean-lunar-calendar==0.4.0", "numpy==2.5.2", "pandas==3.0.5", "pandas-market-calendars==5.4.0",
    "pdfminer-six==20260107", "pdfplumber==0.11.10", "pillow==12.3.0", "pip==25.0.1", "pycparser==3.0",
    "pyluach==2.3.0", "pypdfium2==5.13.0", "python-dateutil==2.9.0.post0", "six==1.17.0", "toolz==1.1.0", "tzdata==2026.3", "xlrd==2.0.2",
)
DELTA = ("cloudpickle==3.1.2", "joblib==1.6.0", "lightgbm==4.6.0", "narwhals==2.26.0", "scikit-learn==1.9.0", "scipy==1.18.1", "threadpoolctl==3.6.0")
SUCCESSOR = tuple(sorted(PREDECESSOR + DELTA))

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FinalFreezeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    repo_root: Path
    reviewed_tooling_sha: str
    expected_candidate_blob_sha1: str
    expected_candidate_sha256: str
    expected_runner_blob_sha1: str
    expected_test_blob_sha1: str
    mutation_attempt_root: Path
    attempt_root: Path

    @property
    def candidate_path(self) -> Path:
        return self.repo_root / "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_RECORD_CANDIDATE.json"

    @property
    def canonical_python(self) -> Path:
        return self.repo_root / ".venv-real-execution" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class VerifiedPhaseAResult:
    canonical_python: Path
    mutation_attempt_root: Path
    attempt_root: Path
    reviewed_tooling_sha: str
    candidate_blob_sha1: str
    candidate_sha256: str
    runner_blob_sha1: str
    test_blob_sha1: str
    provenance: Mapping[str, str]


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def package_map(items: Sequence[Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if isinstance(item, Mapping):
            if set(item) != {"name", "version"}:
                raise FinalFreezeError("PACKAGE_MAPPING_INVALID")
            name, version = item["name"], item["version"]
        elif isinstance(item, str) and "==" in item:
            name, version = item.split("==", 1)
        else:
            raise FinalFreezeError("PACKAGE_MAPPING_INVALID")
        if not isinstance(name, str) or not isinstance(version, str) or not name or not version:
            raise FinalFreezeError("PACKAGE_MAPPING_INVALID")
        key = normalize(name)
        if key in result:
            raise FinalFreezeError("PACKAGE_MAPPING_DUPLICATE")
        result[key] = version
    return result


EXPECTED_PREDECESSOR = package_map(PREDECESSOR)
EXPECTED_SUCCESSOR = package_map(SUCCESSOR)


def _fail(code: str, *, authority_consumed: bool = False, **extra: Any) -> dict[str, Any]:
    return {
        "status": "FAIL",
        "failure_code": code,
        "failure_class": code,
        "authority_consumed": authority_consumed,
        "retry_authorized": False,
        "global_t0_readiness": "NO",
        "t0_authorized": False,
        **extra,
    }


def _strict_json(raw: bytes | str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise FinalFreezeError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise FinalFreezeError("JSON_OBJECT_REQUIRED")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _repo_identity(value: str) -> bool:
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        normalized = normalized[len("git@github.com:") :]
    if normalized.startswith("https://github.com/"):
        normalized = normalized[len("https://github.com/") :]
    return normalized == REPOSITORY_IDENTITY


def _run_git(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
        shell=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if completed.returncode != 0:
        raise FinalFreezeError("GIT_OBSERVATION_FAILED")
    return completed.stdout.strip()


def _git_blob_at(repo_root: Path, revision: str, relative: str) -> str:
    return _run_git(repo_root, ["rev-parse", f"{revision}:{relative}"])


def _safe_ancestor_chain(path: Path) -> bool:
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
            if stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
                return False
            if node != path and not stat.S_ISDIR(info.st_mode):
                return False
        return True
    except OSError:
        return False


def _regular_nonreparse(path: Path) -> bool:
    try:
        info = path.lstat()
        return stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and not bool(getattr(info, "st_file_attributes", 0) & 0x400)
    except OSError:
        return False


def _artifact(repo_root: Path, relative: str, expected_blob: str | None = None, expected_sha256: str | None = None) -> tuple[bytes, str, str]:
    path = repo_root / relative
    if not _safe_ancestor_chain(path) or not _regular_nonreparse(path):
        raise FinalFreezeError("ARTIFACT_UNSAFE")
    raw = path.read_bytes()
    actual_blob = _run_git(repo_root, ["rev-parse", f"HEAD:{relative}"])
    actual_sha256 = _sha256(raw)
    if actual_blob != _blob_sha1(raw):
        raise FinalFreezeError("ARTIFACT_WORKTREE_DRIFT")
    if expected_blob is not None and actual_blob != expected_blob:
        raise FinalFreezeError("ARTIFACT_BLOB_MISMATCH")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise FinalFreezeError("ARTIFACT_SHA256_MISMATCH")
    return raw, actual_blob, actual_sha256


def _path_separation(config: Config) -> bool:
    if config.attempt_root.name != ATTEMPT_NAME or config.mutation_attempt_root.name != MUTATION_ATTEMPT_NAME:
        return False
    for path in (config.repo_root, config.mutation_attempt_root, config.attempt_root, config.canonical_python):
        if not _safe_ancestor_chain(path):
            return False
    if os.path.lexists(config.attempt_root):
        return False
    if any(os.path.lexists(config.attempt_root / name) for name in RESERVED):
        return False
    try:
        repo = config.repo_root.resolve(strict=True)
        mutation = config.mutation_attempt_root.resolve(strict=True)
        attempt = config.attempt_root.resolve(strict=False)
        canonical = config.canonical_python.resolve(strict=False)
    except OSError:
        return False

    def inside(child: Path, parent: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    if inside(attempt, repo) or inside(repo, attempt):
        return False
    if inside(attempt, mutation) or inside(mutation, attempt):
        return False
    if inside(canonical, attempt):
        return False
    return config.attempt_root.parent.is_dir()


def _canonical_identity(config: Config, *, require_existing: bool) -> Path:
    expected = config.canonical_python
    if expected != config.repo_root / ".venv-real-execution" / "Scripts" / "python.exe":
        raise FinalFreezeError("CANONICAL_INTERPRETER_PATH_INVALID")
    if not _safe_ancestor_chain(expected):
        raise FinalFreezeError("CANONICAL_INTERPRETER_UNSAFE")
    if require_existing and not _regular_nonreparse(expected):
        raise FinalFreezeError("CANONICAL_INTERPRETER_MISSING_OR_UNSAFE")
    try:
        resolved = expected.resolve(strict=require_existing)
    except OSError as error:
        raise FinalFreezeError("CANONICAL_INTERPRETER_RESOLUTION_FAILED") from error
    if resolved != expected or resolved.is_relative_to(config.repo_root / ".venv"):
        raise FinalFreezeError("CANONICAL_INTERPRETER_IDENTITY_INVALID")
    return resolved


def _validate_mutation_attempt(config: Config) -> dict[str, Any]:
    root = config.mutation_attempt_root
    if root.name != MUTATION_ATTEMPT_NAME or not root.is_dir() or not _safe_ancestor_chain(root):
        raise FinalFreezeError("MUTATION_ATTEMPT_UNSAFE")
    state_path = root / "mutation_state.json"
    evidence_path = root / "mutation_evidence.json"
    if not _regular_nonreparse(state_path) or not _regular_nonreparse(evidence_path):
        raise FinalFreezeError("MUTATION_EVIDENCE_MISSING_OR_UNSAFE")
    state = _strict_json(state_path.read_bytes())
    evidence = _strict_json(evidence_path.read_bytes())
    if state.get("authority_consumed") is not True or state.get("retry_authorized") is not False:
        raise FinalFreezeError("MUTATION_AUTHORITY_INVALID")
    if state.get("launch_attempted") is not True or state.get("process_started") is not True or state.get("exit_code") != 0:
        raise FinalFreezeError("MUTATION_RESULT_INVALID")
    required = {
        "schema_version": MUTATION_EVIDENCE_SCHEMA,
        "reviewed_implementation_sha": MUTATION_IMPLEMENTATION_SHA,
        "status": "PASS",
        "failure_code": "NONE",
        "failure_class": "PASS",
        "authority_consumed": True,
        "retry_authorized": False,
        "full_validation_run": True,
        "canonical_interpreter_status": "PASS",
        "live_package_observation_status": "PASS",
        "python_version": "3.12.10",
        "package_count": 27,
        "probe_status": "PASS",
        "lightgbm_probe": True,
        "ridge_probe": True,
        "evidence_published": True,
    }
    if any(type(evidence.get(key)) is not type(value) or evidence.get(key) != value for key, value in required.items()):
        raise FinalFreezeError("MUTATION_EVIDENCE_SEMANTICS_INVALID")
    return {"state_valid": True, "evidence_valid": True}


def _validate_candidate(config: Config, raw: bytes) -> dict[str, Any]:
    candidate = _strict_json(raw)
    expected = {
        "schema_version": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_RECORD_CANDIDATE_V1",
        "artifact_role": "SUCCESSOR_FINAL_FREEZE_RECORD_CANDIDATE",
        "study": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR",
        "frozen_design_git_commit": DESIGN_SHA,
        "frozen_design_git_blob_sha1": DESIGN_BLOB,
        "approval_record_reviewed_sha": APPROVAL_RECORD_SHA,
        "approval_record_git_blob_sha1": APPROVAL_RECORD_BLOB,
        "mutation_implementation_reviewed_sha": MUTATION_IMPLEMENTATION_SHA,
        "mutation_phase_c_evidence_schema": MUTATION_EVIDENCE_SCHEMA,
        "mutation_result": "GPT_ADJUDICATED_PASS",
        "successor_lock_git_blob_sha1": SUCCESSOR_BLOB,
        "successor_lock_sha256": SUCCESSOR_SHA256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_BLOB,
        "predecessor_lock_sha256": PREDECESSOR_SHA256,
        "predecessor_package_count": 20,
        "successor_package_count": 27,
        "successor_delta": list(DELTA),
        "successor_delta_package_count": 7,
        "python_version": "3.12.10",
        "canonical_environment_promoted": False,
        "environment_frozen": False,
        "environment_state": "MUTATED_VALIDATED_NOT_FROZEN",
        "global_t0_readiness": "NO",
        "t0_authorized": False,
        "future_profitability_established": False,
    }
    if set(candidate) != set(expected) or any(type(candidate.get(key)) is not type(value) or candidate.get(key) != value for key, value in expected.items()):
        raise FinalFreezeError("CANDIDATE_SEMANTICS_INVALID")
    if candidate.get("successor_delta") != list(DELTA):
        raise FinalFreezeError("CANDIDATE_DELTA_INVALID")
    if _sha256(raw) != config.expected_candidate_sha256:
        raise FinalFreezeError("CANDIDATE_SHA256_MISMATCH")
    return candidate


def collect_production(config: Config) -> Mapping[str, Any]:
    """Collect Phase-A facts mechanically without launching canonical Python."""
    sha = config.reviewed_tooling_sha
    if not isinstance(sha, str) or SHA1_RE.fullmatch(sha) is None:
        raise FinalFreezeError("REVIEWED_TOOLING_SHA_INVALID")
    if not _repo_identity(_run_git(config.repo_root, ["config", "--get", "remote.origin.url"])):
        raise FinalFreezeError("REPOSITORY_IDENTITY_INVALID")
    branch = _run_git(config.repo_root, ["branch", "--show-current"])
    head = _run_git(config.repo_root, ["rev-parse", "HEAD"])
    origin = _run_git(config.repo_root, ["rev-parse", f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}"])
    reviewed_commit = _run_git(config.repo_root, ["rev-parse", "--verify", f"{sha}^{{commit}}"])
    dirty = _run_git(config.repo_root, ["status", "--porcelain=v1", "--untracked-files=all"])
    current_runner = _run_git(config.repo_root, ["hash-object", "--", "scripts/v10c_t0_ml_successor_final_freeze_verification_runner.py"])
    reviewed_runner = _git_blob_at(config.repo_root, sha, "scripts/v10c_t0_ml_successor_final_freeze_verification_runner.py")
    current_test = _run_git(config.repo_root, ["hash-object", "--", "tests/test_v10c_t0_ml_successor_final_freeze_verification_runner.py"])
    reviewed_test = _git_blob_at(config.repo_root, sha, "tests/test_v10c_t0_ml_successor_final_freeze_verification_runner.py")
    if not (branch == AUTHORITATIVE_BRANCH and head == origin == reviewed_commit == sha and dirty == ""):
        raise FinalFreezeError("PROVENANCE_BINDING_FAILURE")
    if current_runner != reviewed_runner or current_runner != config.expected_runner_blob_sha1:
        raise FinalFreezeError("RUNNER_BLOB_MISMATCH")
    if current_test != reviewed_test or current_test != config.expected_test_blob_sha1:
        raise FinalFreezeError("TEST_BLOB_MISMATCH")
    if _git_blob_at(config.repo_root, DESIGN_SHA, "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_PROMOTION_AND_FINAL_FREEZE_DESIGN.md") != DESIGN_BLOB or _git_blob_at(config.repo_root, "HEAD", "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_PROMOTION_AND_FINAL_FREEZE_DESIGN.md") != DESIGN_BLOB:
        raise FinalFreezeError("DESIGN_BLOB_MISMATCH")
    if _git_blob_at(config.repo_root, APPROVAL_RECORD_SHA, "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_PROMOTION_AND_FINAL_FREEZE_DESIGN_FREEZE_APPROVAL.json") != APPROVAL_RECORD_BLOB or _git_blob_at(config.repo_root, "HEAD", "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_PROMOTION_AND_FINAL_FREEZE_DESIGN_FREEZE_APPROVAL.json") != APPROVAL_RECORD_BLOB:
        raise FinalFreezeError("APPROVAL_BLOB_MISMATCH")
    if _git_blob_at(config.repo_root, MUTATION_IMPLEMENTATION_SHA, "scripts/v10c_t0_ml_canonical_mutation_runner.py") != MUTATION_RUNNER_BLOB:
        raise FinalFreezeError("MUTATION_RUNNER_BLOB_MISMATCH")
    design_raw = _run_git(config.repo_root, ["show", f"{DESIGN_SHA}:V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_PROMOTION_AND_FINAL_FREEZE_DESIGN.md"])
    approval_raw = bytes(_run_git(config.repo_root, ["show", f"{APPROVAL_RECORD_SHA}:V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_PROMOTION_AND_FINAL_FREEZE_DESIGN_FREEZE_APPROVAL.json"],), "utf-8")
    _ = design_raw
    approval = _strict_json(approval_raw)
    if approval.get("approval_scope") != "DESIGN_FREEZE_ONLY" or approval.get("approval_status") != "APPROVED" or approval.get("human_design_freeze_complete") is not True:
        raise FinalFreezeError("APPROVAL_SEMANTICS_INVALID")
    pred_raw, pred_blob, pred_sha = _artifact(config.repo_root, "requirements-real-execution.lock.txt", PREDECESSOR_BLOB, PREDECESSOR_SHA256)
    succ_raw, succ_blob, succ_sha = _artifact(config.repo_root, "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK.txt", SUCCESSOR_BLOB, SUCCESSOR_SHA256)
    if package_map(pred_raw.decode("utf-8").splitlines()) != EXPECTED_PREDECESSOR:
        raise FinalFreezeError("PREDECESSOR_LOCK_INVALID")
    successor = package_map(succ_raw.decode("utf-8").splitlines())
    if successor != EXPECTED_SUCCESSOR or {k: v for k, v in successor.items() if k not in EXPECTED_PREDECESSOR} != package_map(DELTA):
        raise FinalFreezeError("SUCCESSOR_LOCK_INVALID")
    promotion_raw, promotion_blob, _ = _artifact(config.repo_root, "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_RESOLUTION_PROMOTION.json", PROMOTION_BLOB)
    source_raw, source_blob, _ = _artifact(config.repo_root, "V10C_T0_ML_RESOLUTION_SOURCE_PROVENANCE.json", SOURCE_PROVENANCE_BLOB)
    promotion = _strict_json(promotion_raw)
    source = _strict_json(source_raw)
    if promotion.get("source_resolution_head") != SOURCE_RESOLUTION_HEAD or promotion.get("source_wheel_count") != SOURCE_WHEEL_COUNT or promotion.get("source_wheel_total_bytes") != SOURCE_WHEEL_TOTAL_BYTES or promotion.get("source_wheel_manifest_sha256") != SOURCE_WHEEL_MANIFEST_SHA256 or promotion.get("offline_readjudication_evidence_sha256") != OFFLINE_EVIDENCE_SHA256 or promotion.get("successor_lock_sha256") != SUCCESSOR_SHA256 or promotion.get("successor_lock_package_count") != 27 or promotion.get("predecessor_package_count") != 20 or promotion.get("successor_delta_package_count") != 7 or promotion.get("successor_delta_packages") != list(DELTA) or promotion.get("resolution_authority_consumed") is not True or promotion.get("resolution_retry_authorized") is not False:
        raise FinalFreezeError("PROMOTION_SEMANTICS_INVALID")
    if source.get("source_resolution_head") != SOURCE_RESOLUTION_HEAD or source.get("source_wheel_count") != SOURCE_WHEEL_COUNT or source.get("source_wheel_total_bytes") != SOURCE_WHEEL_TOTAL_BYTES or source.get("source_wheel_manifest_sha256") != SOURCE_WHEEL_MANIFEST_SHA256 or source.get("wheel_manifest_algorithm") != "CANONICAL_WHEEL_MANIFEST_V1" or source.get("source_phase_b_authority_consumed") is not True or source.get("source_phase_c_result") != "FAIL" or source.get("source_phase_c_failure_code") != "RESOLUTION_REPORT_INVALID" or source.get("resolution_retry_authorized") is not False:
        raise FinalFreezeError("SOURCE_PROVENANCE_INVALID")
    mutation = _validate_mutation_attempt(config)
    canonical = _canonical_identity(config, require_existing=True)
    if not _path_separation(config):
        raise FinalFreezeError("ATTEMPT_NAMESPACE_UNSAFE")
    candidate_raw, candidate_blob, candidate_sha = _artifact(config.repo_root, "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_RECORD_CANDIDATE.json", config.expected_candidate_blob_sha1, config.expected_candidate_sha256)
    _validate_candidate(config, candidate_raw)
    return {
        "branch": branch,
        "head": head,
        "origin_head": origin,
        "clean": True,
        "reviewed_tooling_sha": sha,
        "reviewed_runner_blob_sha1": reviewed_runner,
        "current_runner_blob_sha1": current_runner,
        "reviewed_test_blob_sha1": reviewed_test,
        "current_test_blob_sha1": current_test,
        "design_commit": DESIGN_SHA,
        "design_blob": DESIGN_BLOB,
        "approval_commit": APPROVAL_RECORD_SHA,
        "approval_blob": APPROVAL_RECORD_BLOB,
        "predecessor_blob": pred_blob,
        "predecessor_sha256": pred_sha,
        "successor_blob": succ_blob,
        "successor_sha256": succ_sha,
        "promotion_blob": promotion_blob,
        "source_provenance_blob": source_blob,
        "candidate_blob": candidate_blob,
        "candidate_sha256": candidate_sha,
        "mutation_attempt_identity": MUTATION_ATTEMPT_NAME,
        "mutation_attempt_safe": mutation["state_valid"] and mutation["evidence_valid"],
        "canonical_interpreter_configured": str(canonical).endswith(os.path.join(".venv-real-execution", "Scripts", "python.exe")),
        "canonical_interpreter_existing": True,
        "final_freeze_attempt_name": ATTEMPT_NAME,
        "final_freeze_attempt_absent": True,
        "reserved_children_absent": True,
        "ancestors_safe": True,
        "writes": 0,
        "network_requests": 0,
    }


def _phase_a_from_observations(config: Config, observed: Mapping[str, Any]) -> VerifiedPhaseAResult:
    required = ("branch", "head", "origin_head", "clean", "reviewed_tooling_sha", "reviewed_runner_blob_sha1", "current_runner_blob_sha1", "reviewed_test_blob_sha1", "current_test_blob_sha1", "candidate_blob", "candidate_sha256", "mutation_attempt_safe", "canonical_interpreter_configured", "canonical_interpreter_existing", "final_freeze_attempt_name", "final_freeze_attempt_absent", "reserved_children_absent", "ancestors_safe", "writes", "network_requests")
    if any(key not in observed for key in required):
        raise FinalFreezeError("PRE_GATE_ENVIRONMENT_BLOCK")
    if observed.get("branch") != AUTHORITATIVE_BRANCH or observed.get("head") != config.reviewed_tooling_sha or observed.get("origin_head") != config.reviewed_tooling_sha or observed.get("clean") is not True:
        raise FinalFreezeError("PRE_GATE_ENVIRONMENT_BLOCK")
    if observed.get("reviewed_tooling_sha") != config.reviewed_tooling_sha or observed.get("reviewed_runner_blob_sha1") != config.expected_runner_blob_sha1 or observed.get("current_runner_blob_sha1") != config.expected_runner_blob_sha1 or observed.get("reviewed_test_blob_sha1") != config.expected_test_blob_sha1 or observed.get("current_test_blob_sha1") != config.expected_test_blob_sha1:
        raise FinalFreezeError("PRE_GATE_ENVIRONMENT_BLOCK")
    if observed.get("candidate_blob") != config.expected_candidate_blob_sha1 or observed.get("candidate_sha256") != config.expected_candidate_sha256 or observed.get("mutation_attempt_safe") is not True or observed.get("canonical_interpreter_configured") is not True or observed.get("canonical_interpreter_existing") is not True or observed.get("final_freeze_attempt_name") != ATTEMPT_NAME or observed.get("final_freeze_attempt_absent") is not True or observed.get("reserved_children_absent") is not True or observed.get("ancestors_safe") is not True or observed.get("writes") != 0 or observed.get("network_requests") != 0:
        raise FinalFreezeError("PRE_GATE_ENVIRONMENT_BLOCK")
    if not _path_separation(config):
        raise FinalFreezeError("PRE_GATE_ENVIRONMENT_BLOCK")
    return VerifiedPhaseAResult(config.canonical_python, config.mutation_attempt_root, config.attempt_root, config.reviewed_tooling_sha, config.expected_candidate_blob_sha1, config.expected_candidate_sha256, config.expected_runner_blob_sha1, config.expected_test_blob_sha1, MappingProxyType({"design_blob": DESIGN_BLOB, "approval_blob": APPROVAL_RECORD_BLOB, "predecessor_blob": PREDECESSOR_BLOB, "successor_blob": SUCCESSOR_BLOB, "promotion_blob": PROMOTION_BLOB, "source_provenance_blob": SOURCE_PROVENANCE_BLOB}))


def phase_a(config: Config) -> dict[str, Any]:
    """Production Phase-A entrypoint; it always collects its own observations."""
    try:
        observed = collect_production(config)
        _phase_a_from_observations(config, observed)
        return {"status": "PASS", "failure_code": "NONE", "failure_class": "PASS", "preflight_ready": True, "CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE": "NO", "global_t0_readiness": "NO", "t0_authorized": False, "authority_consumed": False, "retry_authorized": False, "writes": 0, "network_requests": 0}
    except (FinalFreezeError, OSError, UnicodeError, ValueError, TypeError, KeyError):
        return _fail("PRE_GATE_ENVIRONMENT_BLOCK")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_file_summary(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
            return {"exists": True, "size": None, "sha256": None, "integrity": "FAIL"}
        raw = path.read_bytes()
        return {"exists": True, "size": len(raw), "sha256": _sha256(raw), "integrity": "PASS"}
    except FileNotFoundError:
        return {"exists": False, "size": None, "sha256": None, "integrity": "MISSING"}
    except OSError:
        return {"exists": "UNKNOWN", "size": None, "sha256": None, "integrity": "UNKNOWN"}


def _read_state(config: Config) -> tuple[dict[str, Any] | None, bool]:
    try:
        path = config.attempt_root / RESERVED[0]
        if not _regular_nonreparse(path):
            return None, False
        state = _strict_json(path.read_bytes())
        exit_code = state.get("process_exit_code", state.get("exit_code"))
        valid = (
            state.get("schema_version") == STATE_SCHEMA
            and state.get("attempt_name") == ATTEMPT_NAME
            and state.get("reviewed_implementation_sha") == config.reviewed_tooling_sha
            and state.get("authority_consumed") is True
            and state.get("retry_authorized") is False
            and state.get("phase_c_required") is True
            and isinstance(state.get("launch_attempted"), bool)
            and state.get("process_started") in {True, False, "UNKNOWN"}
            and ((isinstance(exit_code, int) and not isinstance(exit_code, bool)) or exit_code == "UNKNOWN")
        )
        return state, valid
    except (OSError, UnicodeError, ValueError, TypeError, FinalFreezeError):
        return None, False


def _inspection(config: Config, state: Mapping[str, Any] | None, valid: bool) -> dict[str, Any]:
    exit_code = state.get("process_exit_code", state.get("exit_code")) if state else "UNKNOWN"
    if not (isinstance(exit_code, int) and not isinstance(exit_code, bool)):
        exit_code = "UNKNOWN"
    return {
        "state_valid": valid,
        "authority_consumed": True,
        "retry_authorized": False,
        "launch_attempted": state.get("launch_attempted") if state and isinstance(state.get("launch_attempted"), bool) else "UNKNOWN",
        "process_started": state.get("process_started") if state and state.get("process_started") in {True, False, "UNKNOWN"} else "UNKNOWN",
        "process_exit_code": exit_code,
        "stdout": _safe_file_summary(config.attempt_root / RESERVED[1]),
        "stderr": _safe_file_summary(config.attempt_root / RESERVED[2]),
    }


def _base_evidence(config: Config, inspection: Mapping[str, Any], *, status: str, code: str, failure_class: str, full_validation_run: bool) -> dict[str, Any]:
    return {
        "schema_version": FINAL_EVIDENCE_SCHEMA,
        "attempt_name": ATTEMPT_NAME,
        "reviewed_implementation_sha": config.reviewed_tooling_sha,
        "status": status,
        "failure_code": code,
        "failure_class": failure_class,
        "authority_consumed": True,
        "retry_authorized": False,
        "evidence_published": False,
        "inspection": dict(inspection),
        "full_validation_run": full_validation_run,
        "canonical_interpreter_status": "UNKNOWN",
        "live_package_observation_status": "NOT_RUN",
        "readiness_evidence_only": status == "PASS",
        "canonical_environment_promoted": False,
        "environment_frozen": False,
        "global_t0_readiness": "NO",
        "t0_authorized": False,
        "future_profitability_established": False,
    }


def _live_observer_script() -> str:
    expected = repr(EXPECTED_SUCCESSOR)
    return f"""
import importlib.metadata, json, math, platform, re, sys
packages=[]
for distribution in importlib.metadata.distributions():
    name=distribution.metadata.get('Name')
    version=distribution.version
    if not isinstance(name,str) or not isinstance(version,str):
        raise ValueError('MALFORMED_METADATA')
    packages.append({{'name':name,'version':version}})
result={{'python_version':platform.python_version(),'executable':sys.executable,'packages':packages,'canonical_interpreter_status':'FAIL','live_package_observation_status':'FAIL','probe_status':'NOT_RUN','lightgbm_probe':False,'ridge_probe':False}}
names={{}}
for item in packages:
    key=re.sub(r'[-_.]+','-',item['name']).lower()
    if key in names:
        print(json.dumps(result,sort_keys=True)); raise SystemExit(0)
    names[key]=item['version']
if len(packages)!=27 or names!={expected}:
    print(json.dumps(result,sort_keys=True)); raise SystemExit(0)
result['live_package_observation_status']='PASS'
result['canonical_interpreter_status']='PASS' if result['python_version']=='3.12.10' else 'FAIL'
X=[[0.0,0.0],[1.0,1.0],[2.0,2.0],[3.0,3.0]]
y=[0.0,1.0,2.0,3.0]
try:
    from lightgbm import LGBMRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    lgbm=LGBMRegressor(n_estimators=1,random_state=0,n_jobs=1,verbosity=-1)
    lgbm.fit(X,y)
    lgbm_prediction=lgbm.predict([[1.5,1.5]])
    scaler=StandardScaler()
    transformed=scaler.fit_transform(X)
    ridge=Ridge()
    ridge.fit(transformed,y)
    ridge_prediction=ridge.predict(scaler.transform([[1.5,1.5]]))
    result['lightgbm_probe']=len(lgbm_prediction)==1 and math.isfinite(float(lgbm_prediction[0]))
    result['ridge_probe']=len(ridge_prediction)==1 and math.isfinite(float(ridge_prediction[0]))
    result['probe_status']='PASS' if result['lightgbm_probe'] and result['ridge_probe'] else 'FAIL'
except Exception:
    result['probe_status']='FAIL'
print(json.dumps(result,sort_keys=True))
"""


def _launch(canonical: Path, stdout: Path, stderr: Path) -> int:
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull, "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        return subprocess.run([str(canonical), "-B", "-I", "-c", _live_observer_script()], stdout=out, stderr=err, check=False, shell=False, env=env).returncode


def _live_fields_from_stdout(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = _strict_json(raw)
    if set(value) != {"canonical_interpreter_status", "executable", "live_package_observation_status", "packages", "probe_status", "python_version", "lightgbm_probe", "ridge_probe"}:
        raise FinalFreezeError("LIVE_OBSERVER_SCHEMA_INVALID")
    if not isinstance(value["packages"], list) or type(value["lightgbm_probe"]) is not bool or type(value["ridge_probe"]) is not bool:
        raise FinalFreezeError("LIVE_OBSERVER_TYPES_INVALID")
    return value


def _validate_existing_evidence(stored: Mapping[str, Any], config: Config, inspection: Mapping[str, Any]) -> bool:
    if stored.get("schema_version") != FINAL_EVIDENCE_SCHEMA or stored.get("attempt_name") != ATTEMPT_NAME or stored.get("reviewed_implementation_sha") != config.reviewed_tooling_sha or stored.get("authority_consumed") is not True or stored.get("retry_authorized") is not False or stored.get("evidence_published") is not True or stored.get("inspection") != dict(inspection):
        return False
    if stored.get("status") == "PASS":
        return stored.get("failure_code") == "NONE" and stored.get("failure_class") == "PASS" and stored.get("full_validation_run") is True and stored.get("readiness_evidence_only") is True and stored.get("canonical_interpreter_status") == "PASS" and stored.get("live_package_observation_status") == "PASS" and stored.get("python_version") == "3.12.10" and stored.get("package_count") == 27 and stored.get("probe_status") == "PASS" and stored.get("lightgbm_probe") is True and stored.get("ridge_probe") is True
    if stored.get("status") != "FAIL" or stored.get("full_validation_run") is not False or stored.get("readiness_evidence_only") is True:
        return False
    return stored.get("failure_code") == stored.get("failure_class") and stored.get("failure_class") in {"IMPLEMENTATION_FAILURE", "LIVE_ENVIRONMENT_VALIDATION_FAILURE"}


def phase_c(config: Config) -> dict[str, Any]:
    """Dedicated no-network inspection; it never calls Phase A or the wheel root."""
    state, state_valid = _read_state(config)
    inspection = _inspection(config, state, state_valid)
    evidence_path = config.attempt_root / RESERVED[3]
    if os.path.lexists(evidence_path):
        try:
            if not _regular_nonreparse(evidence_path):
                raise FinalFreezeError("EVIDENCE_UNSAFE")
            stored = _strict_json(evidence_path.read_bytes())
            if not _validate_existing_evidence(stored, config, inspection):
                raise FinalFreezeError("EVIDENCE_INCONSISTENT")
            result = dict(stored)
            result["existing_evidence_inspected"] = True
            return result
        except (OSError, UnicodeError, ValueError, TypeError, FinalFreezeError):
            return {"status": "FAIL", "failure_code": "IMPLEMENTATION_FAILURE", "failure_class": "IMPLEMENTATION_FAILURE", "authority_consumed": True, "retry_authorized": False, "evidence_published": True, "existing_evidence_inspected": True, "inspection": inspection, "global_t0_readiness": "NO", "t0_authorized": False}
    if not state_valid:
        result = _base_evidence(config, inspection, status="FAIL", code="IMPLEMENTATION_FAILURE", failure_class="IMPLEMENTATION_FAILURE", full_validation_run=False)
    elif state.get("process_exit_code", state.get("exit_code")) != 0 or state.get("launch_attempted") is not True or state.get("process_started") is not True:
        result = _base_evidence(config, inspection, status="FAIL", code="IMPLEMENTATION_FAILURE", failure_class="IMPLEMENTATION_FAILURE", full_validation_run=False)
    else:
        try:
            runtime = _live_fields_from_stdout(config.attempt_root / RESERVED[1])
            package_status = "PASS" if package_map(runtime["packages"]) == EXPECTED_SUCCESSOR else "FAIL"
            result = _base_evidence(config, inspection, status="FAIL", code="LIVE_ENVIRONMENT_VALIDATION_FAILURE", failure_class="LIVE_ENVIRONMENT_VALIDATION_FAILURE", full_validation_run=False)
            result.update({"canonical_interpreter_status": runtime["canonical_interpreter_status"], "live_package_observation_status": package_status, "python_version": runtime["python_version"], "package_count": len(runtime["packages"]), "probe_status": runtime["probe_status"], "lightgbm_probe": runtime["lightgbm_probe"], "ridge_probe": runtime["ridge_probe"]})
            interpreter_pass = runtime["canonical_interpreter_status"] == "PASS" and runtime["python_version"] == "3.12.10" and Path(runtime["executable"]) == config.canonical_python
            probes_pass = runtime["probe_status"] == "PASS" and runtime["lightgbm_probe"] is True and runtime["ridge_probe"] is True
            if interpreter_pass and package_status == "PASS" and probes_pass:
                result.update(status="PASS", failure_code="NONE", failure_class="PASS", full_validation_run=True, readiness_evidence_only=True)
        except (OSError, UnicodeError, ValueError, TypeError, FinalFreezeError):
            result = _base_evidence(config, inspection, status="FAIL", code="IMPLEMENTATION_FAILURE", failure_class="IMPLEMENTATION_FAILURE", full_validation_run=False)
    try:
        _atomic_json(evidence_path, {**result, "evidence_published": True})
        result["evidence_published"] = True
    except BaseException:
        result = {**result, "status": "FAIL", "failure_code": "IMPLEMENTATION_FAILURE", "failure_class": "IMPLEMENTATION_FAILURE", "evidence_published": False, "authority_consumed": True, "retry_authorized": False}
    return result


def phase_b(config: Config, *, final_freeze_authorized: bool) -> dict[str, Any]:
    """Run one authorized attempt and route every crossed boundary to Phase C."""
    try:
        verified = _phase_a_from_observations(config, collect_production(config))
    except (FinalFreezeError, OSError, ValueError, TypeError, KeyError):
        return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if not final_freeze_authorized:
        return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    if os.path.lexists(verified.attempt_root):
        return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    boundary = False
    state = {
        "schema_version": STATE_SCHEMA,
        "attempt_name": ATTEMPT_NAME,
        "reviewed_implementation_sha": verified.reviewed_tooling_sha,
        "authority_consumed": True,
        "retry_authorized": False,
        "phase_c_required": True,
        "launch_attempted": False,
        "process_started": False,
        "process_exit_code": "UNKNOWN",
    }
    try:
        verified.attempt_root.mkdir(parents=False, exist_ok=False)
        boundary = True
        _atomic_json(verified.attempt_root / RESERVED[0], state)
        (verified.attempt_root / RESERVED[1]).touch(exist_ok=False)
        (verified.attempt_root / RESERVED[2]).touch(exist_ok=False)
        state.update(launch_attempted=True, process_started="UNKNOWN")
        _atomic_json(verified.attempt_root / RESERVED[0], state)
        try:
            code = _launch(verified.canonical_python, verified.attempt_root / RESERVED[1], verified.attempt_root / RESERVED[2])
            state.update(process_started=True, process_exit_code=code if isinstance(code, int) and not isinstance(code, bool) else "UNKNOWN")
        except BaseException:
            state.update(process_started="UNKNOWN", process_exit_code="UNKNOWN", launch_exception=True)
        try:
            _atomic_json(verified.attempt_root / RESERVED[0], state)
        except BaseException:
            state.update(state_publication_failure=True)
    except BaseException:
        state.update(process_started="UNKNOWN", process_exit_code="UNKNOWN", boundary_uncertain=True)
    phase_b_result = {"status": "PASS" if state.get("process_exit_code") == 0 and state.get("process_started") is True else "FAIL", "failure_code": "NONE" if state.get("process_exit_code") == 0 and state.get("process_started") is True else "IMPLEMENTATION_FAILURE", **state}
    if not boundary:
        return _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    phase_c_result = phase_c(Config(config.repo_root, config.reviewed_tooling_sha, config.expected_candidate_blob_sha1, config.expected_candidate_sha256, config.expected_runner_blob_sha1, config.expected_test_blob_sha1, config.mutation_attempt_root, verified.attempt_root))
    return {"status": phase_c_result.get("status", "FAIL"), "failure_code": phase_c_result.get("failure_code", "IMPLEMENTATION_FAILURE"), "authority_consumed": True, "retry_authorized": False, "phase_b_result": phase_b_result, "phase_c_result": phase_c_result}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("phase-a", "phase-b", "phase-c"))
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--reviewed-tooling-sha", required=True)
    parser.add_argument("--expected-candidate-blob-sha1", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-runner-blob-sha1", required=True)
    parser.add_argument("--expected-test-blob-sha1", required=True)
    parser.add_argument("--mutation-attempt-root", required=True)
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--final-freeze-authorized", action="store_true")
    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    for value, pattern, label in ((args.reviewed_tooling_sha, SHA1_RE, "reviewed tooling sha"), (args.expected_candidate_blob_sha1, SHA1_RE, "candidate blob"), (args.expected_candidate_sha256, SHA256_RE, "candidate sha256"), (args.expected_runner_blob_sha1, SHA1_RE, "runner blob"), (args.expected_test_blob_sha1, SHA1_RE, "test blob")):
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise FinalFreezeError(f"{label.upper().replace(' ', '_')}_INVALID")
    paths = {name: Path(getattr(args, name)) for name in ("repo_root", "mutation_attempt_root", "attempt_root")}
    if any(not path.is_absolute() for path in paths.values()):
        raise FinalFreezeError("PATH_MUST_BE_ABSOLUTE")
    return Config(paths["repo_root"], args.reviewed_tooling_sha, args.expected_candidate_blob_sha1, args.expected_candidate_sha256, args.expected_runner_blob_sha1, args.expected_test_blob_sha1, paths["mutation_attempt_root"], paths["attempt_root"])


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
        config = _config_from_args(args)
        if args.phase == "phase-a":
            result = phase_a(config)
        elif args.phase == "phase-b":
            result = phase_b(config, final_freeze_authorized=args.final_freeze_authorized)
        else:
            result = phase_c(config)
    except (FinalFreezeError, OSError, UnicodeError, ValueError, TypeError, KeyError):
        result = _fail("PRE_GATE_ENVIRONMENT_BLOCK")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
