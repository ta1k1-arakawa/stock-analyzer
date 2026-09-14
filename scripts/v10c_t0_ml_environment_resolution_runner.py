"""Synthetic-testable V10C Windows wheel-resolution Phase A/B/C runner.

The runner contains no import-time I/O and never imports the T0 ML stack.
The real Phase-B subprocess is injectable and is not invoked by this
implementation task; tests use temporary synthetic wheelhouses and fakes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from scripts.v10c_t0_ml_environment_contract import (
    APPROVAL_RECORD_BLOB,
    CANDIDATE_SCHEMA,
    DIRECT_SPEC_SHA256,
    DIRECT_SPEC_BYTES,
    EVIDENCE_SCHEMA,
    FROZEN_DESIGN_BLOB,
    FROZEN_DESIGN_SHA,
    PACKAGE_INDEX_ID,
    PREDECESSOR_LOCK_BLOB,
    PREDECESSOR_LOCK_SHA256,
    PREDECESSOR_PACKAGE_SET,
    RESOLUTION_POLICY_ID,
    STUDY,
    ContractValidationError,
    SHA1_RE,
    SHA256_RE,
    canonical_json_bytes,
    git_blob_sha1,
    inspect_wheel_file,
    inspect_wheelhouse,
    validate_approval_record,
    validate_direct_spec_bytes,
    validate_evidence,
    validate_lock_candidate,
    validate_predecessor_packages,
    validate_resolved_packages,
    validate_wheel_manifest,
)


REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
CANONICAL_INTERPRETER_RELATIVE = Path(".venv-real-execution") / "Scripts" / "python.exe"
CANONICAL_ENVIRONMENT_RELATIVE = Path(".venv-real-execution")
DESIGN_RELATIVE = Path("V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DESIGN_DRAFT.md")
APPROVAL_RELATIVE = Path("V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DESIGN_FREEZE_APPROVAL.json")
LOCK_RELATIVE = Path("requirements-real-execution.lock.txt")
DIRECT_SPEC_RELATIVE = Path("V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt")
RUNNER_RELATIVE = Path("scripts/v10c_t0_ml_environment_resolution_runner.py")
WHEELHOUSE_NAME = "wheelhouse"
STATE_NAME = "attempt_state.json"
STDOUT_NAME = "stdout.txt"
STDERR_NAME = "stderr.txt"
CANDIDATE_NAME = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json"
EVIDENCE_NAME = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE.json"
ATTEMPT_STATE_SCHEMA = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_ATTEMPT_STATE_V1"
ATTEMPT_STATE_KEYS = frozenset(
    {
        "schema_version", "expected_current_head", "frozen_design_git_sha",
        "frozen_design_git_blob_sha1", "approval_record_git_blob_sha1",
        "reviewed_resolution_implementation_git_sha", "direct_spec_git_blob_sha1",
        "direct_spec_sha256", "predecessor_lock_git_blob_sha1", "predecessor_lock_sha256",
        "resolution_policy_id", "package_index_id", "attempt_boundary_crossed",
        "human_authority_consumed", "process_started", "process_exit_code",
        "package_resolution_process_invocations", "resolution_completed",
    }
)


class RunnerValidationError(ContractValidationError):
    """A bounded, safe, fail-closed runner error."""


@dataclass(frozen=True)
class PhaseAConfig:
    repo_root: Path
    expected_current_head: str
    expected_reviewed_runner_sha: str
    expected_direct_spec_git_blob_sha1: str
    expected_direct_spec_sha256: str = DIRECT_SPEC_SHA256
    durable_root: Path = Path(".")
    protected_environment: Path | None = None
    governed_roots: tuple[Path, ...] = ()

    @property
    def canonical_interpreter(self) -> Path:
        return self.repo_root / CANONICAL_INTERPRETER_RELATIVE

    @property
    def direct_spec(self) -> Path:
        return self.repo_root / DIRECT_SPEC_RELATIVE

    @property
    def approval_file(self) -> Path:
        return self.repo_root / APPROVAL_RELATIVE

    @property
    def lock_file(self) -> Path:
        return self.repo_root / LOCK_RELATIVE


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RunnerValidationError(message)


def _safe_sha(value: Any, pattern: re.Pattern[str], label: str) -> None:
    _require(isinstance(value, str) and pattern.fullmatch(value) is not None, f"{label}_INVALID")


def _repo_identity_matches(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().replace("\\", "/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        normalized = normalized[len("git@github.com:") :]
    if normalized.startswith("https://github.com/"):
        normalized = normalized[len("https://github.com/") :]
    return normalized.rstrip("/") == REPOSITORY_IDENTITY


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & 0x400)


def _nearest_existing(path: Path) -> Path:
    node = path
    while not os.path.lexists(node) and node.parent != node:
        node = node.parent
    return node


def _unsafe_existing_ancestor(path: Path) -> bool:
    node = _nearest_existing(path)
    while True:
        if _is_reparse_or_symlink(node):
            return True
        if node.parent == node:
            return False
        node = node.parent


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_durable_root(durable_root: Path, *, repo_root: Path, protected_environment: Path, governed_roots: Sequence[Path] = ()) -> str:
    root = Path(durable_root)
    if not root.is_absolute():
        return "DURABLE_ROOT_NOT_ABSOLUTE"
    if os.path.lexists(root):
        return "DURABLE_ROOT_ALREADY_EXISTS"
    if _unsafe_existing_ancestor(root):
        return "DURABLE_ROOT_REPARSE_OR_SYMLINK"
    try:
        resolved_root = Path(os.path.realpath(root))
        forbidden = [Path(os.path.realpath(repo_root)), Path(os.path.realpath(protected_environment))]
        forbidden.extend(Path(os.path.realpath(item)) for item in governed_roots)
    except OSError:
        return "DURABLE_ROOT_SAFETY_UNDETERMINED"
    if any(_path_inside(resolved_root, item) or _path_inside(item, resolved_root) for item in forbidden):
        return "DURABLE_ROOT_GOVERNED_PATH_OVERLAP"
    if not _nearest_existing(root).is_dir():
        return "DURABLE_ROOT_PARENT_MISSING"
    return "DURABLE_ROOT_OK"


def _parse_lock_packages(raw: bytes) -> list[dict[str, str]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RunnerValidationError("PREDECESSOR_LOCK_UNREADABLE") from error
    result: list[dict[str, str]] = []
    for line in lines:
        if not line or line.count("==") != 1:
            raise RunnerValidationError("PREDECESSOR_LOCK_INVALID")
        name, version = line.split("==", 1)
        result.append({"name": name, "version": version})
    return result


def _phase_a_failure(code: str, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "FAIL", "failure_code": code,
        "repository_identity_verified": False, "authoritative_branch_verified": False,
        "origin_authoritative_ref_verified": False, "expected_head_verified": False,
        "working_tree_clean": False, "frozen_design_verified": False,
        "approval_record_verified": False, "implementation_verified": False,
        "direct_spec_verified": False, "predecessor_lock_verified": False,
        "canonical_interpreter_verified": False,
        "live_predecessor_package_set_verified": False,
        "durable_root_status": "NOT_CHECKED", "network_requests": 0, "writes": 0,
    }
    result.update(extra)
    return result


def _validate_observed_bindings(config: PhaseAConfig, obs: Mapping[str, Any]) -> None:
    for value, pattern, label in (
        (config.expected_current_head, SHA1_RE, "expected head"),
        (config.expected_reviewed_runner_sha, SHA1_RE, "reviewed runner"),
        (config.expected_direct_spec_git_blob_sha1, SHA1_RE, "direct spec blob"),
        (config.expected_direct_spec_sha256, SHA256_RE, "direct spec SHA"),
    ):
        _safe_sha(value, pattern, label)
    if not _repo_identity_matches(obs.get("repository_identity")):
        raise RunnerValidationError("REPOSITORY_IDENTITY_MISMATCH")
    if obs.get("branch") != AUTHORITATIVE_BRANCH:
        raise RunnerValidationError("AUTHORITATIVE_BRANCH_MISMATCH")
    if obs.get("origin_head") != config.expected_current_head:
        raise RunnerValidationError("ORIGIN_AUTHORITATIVE_REF_MISMATCH")
    if obs.get("head") != config.expected_current_head:
        raise RunnerValidationError("EXPECTED_HEAD_MISMATCH")
    if obs.get("clean") is not True:
        raise RunnerValidationError("WORKING_TREE_DIRTY")
    if obs.get("frozen_design_commit") != FROZEN_DESIGN_SHA or obs.get("frozen_design_blob") != FROZEN_DESIGN_BLOB:
        raise RunnerValidationError("FROZEN_DESIGN_BINDING_FAILURE")
    if obs.get("approval_record_blob") != APPROVAL_RECORD_BLOB:
        raise RunnerValidationError("APPROVAL_RECORD_BLOB_BINDING_FAILURE")
    record = obs.get("approval_record")
    if not isinstance(record, Mapping):
        raise RunnerValidationError("APPROVAL_RECORD_UNREADABLE")
    try:
        validate_approval_record(record)
    except ContractValidationError as error:
        raise RunnerValidationError("APPROVAL_RECORD_SEMANTICS_FAILURE") from error
    if obs.get("reviewed_runner_blob") != obs.get("current_runner_blob"):
        raise RunnerValidationError("IMPLEMENTATION_PROVENANCE_FAILURE")
    direct_bytes = obs.get("direct_spec_bytes")
    committed_direct = obs.get("direct_spec_committed_bytes")
    if not isinstance(direct_bytes, bytes) or direct_bytes != committed_direct:
        raise RunnerValidationError("DIRECT_SPEC_BYTES_MISMATCH")
    if validate_direct_spec_bytes(direct_bytes) != config.expected_direct_spec_sha256:
        raise RunnerValidationError("DIRECT_SPEC_SHA256_MISMATCH")
    if obs.get("direct_spec_sha256") != config.expected_direct_spec_sha256 or obs.get("direct_spec_blob") != config.expected_direct_spec_git_blob_sha1:
        raise RunnerValidationError("DIRECT_SPEC_PROVENANCE_FAILURE")
    lock_bytes = obs.get("predecessor_lock_bytes")
    if not isinstance(lock_bytes, bytes) or hashlib.sha256(lock_bytes).hexdigest() != PREDECESSOR_LOCK_SHA256 or git_blob_sha1(lock_bytes) != PREDECESSOR_LOCK_BLOB:
        raise RunnerValidationError("PREDECESSOR_LOCK_PROVENANCE_FAILURE")
    if obs.get("predecessor_lock_blob") != PREDECESSOR_LOCK_BLOB or obs.get("predecessor_lock_sha256") != PREDECESSOR_LOCK_SHA256:
        raise RunnerValidationError("PREDECESSOR_LOCK_PROVENANCE_FAILURE")
    validate_predecessor_packages(_parse_lock_packages(lock_bytes))
    validate_predecessor_packages(obs.get("live_packages"))
    expected = str(config.canonical_interpreter.resolve())
    if str(Path(str(obs.get("interpreter_executable"))).resolve()) != expected:
        raise RunnerValidationError("CANONICAL_INTERPRETER_BINDING_FAILURE")
    for key, expected_value in (
        ("python_implementation", "CPython"), ("python_version", "3.12.10"),
        ("platform_system", "Windows"), ("platform_machine", "AMD64"),
        ("sysconfig_platform", "win-amd64"), ("pip_version", "25.0.1"),
    ):
        if obs.get(key) != expected_value:
            raise RunnerValidationError("CANONICAL_INTERPRETER_BINDING_FAILURE")


def run_phase_a(config: PhaseAConfig, observations: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Perform only local provenance/interpreter/root checks; never resolve."""

    try:
        obs = dict(_default_phase_a_observations(config) if observations is None else observations)
        _validate_observed_bindings(config, obs)
        protected = config.protected_environment or config.repo_root / CANONICAL_ENVIRONMENT_RELATIVE
        root_status = validate_durable_root(config.durable_root, repo_root=config.repo_root, protected_environment=protected, governed_roots=config.governed_roots)
        if root_status != "DURABLE_ROOT_OK":
            raise RunnerValidationError(f"DURABLE_ROOT_{root_status.replace('DURABLE_ROOT_', '')}")
        return {
            "status": "PASS", "failure_code": "NONE", "repository_identity_verified": True,
            "authoritative_branch_verified": True, "origin_authoritative_ref_verified": True,
            "expected_head_verified": True, "working_tree_clean": True,
            "frozen_design_verified": True, "approval_record_verified": True,
            "implementation_verified": True, "direct_spec_verified": True,
            "direct_spec_sha256": config.expected_direct_spec_sha256,
            "predecessor_lock_verified": True, "predecessor_package_count": len(PREDECESSOR_PACKAGE_SET),
            "canonical_interpreter_verified": True, "live_predecessor_package_set_verified": True,
            "durable_root_status": root_status, "network_requests": 0, "writes": 0,
            "human_authority_consumed": False,
        }
    except (KeyError, TypeError, ValueError, RunnerValidationError, OSError) as error:
        error_code = str(error)
        allowed_codes = {
            "REPOSITORY_IDENTITY_MISMATCH", "AUTHORITATIVE_BRANCH_MISMATCH", "ORIGIN_AUTHORITATIVE_REF_MISMATCH",
            "EXPECTED_HEAD_MISMATCH", "WORKING_TREE_DIRTY", "FROZEN_DESIGN_BINDING_FAILURE",
            "APPROVAL_RECORD_BLOB_BINDING_FAILURE", "APPROVAL_RECORD_UNREADABLE",
            "APPROVAL_RECORD_SEMANTICS_FAILURE", "IMPLEMENTATION_PROVENANCE_FAILURE",
            "DIRECT_SPEC_BYTES_MISMATCH", "DIRECT_SPEC_SHA256_MISMATCH", "DIRECT_SPEC_PROVENANCE_FAILURE",
            "PREDECESSOR_LOCK_PROVENANCE_FAILURE", "PREDECESSOR_LOCK_UNREADABLE", "PREDECESSOR_LOCK_INVALID",
            "PREDECESSOR_PIN_DRIFT", "CANONICAL_INTERPRETER_BINDING_FAILURE",
        }
        code = error_code if error_code in allowed_codes else "PHASE_A_PRECHECK_FAILURE"
        return _phase_a_failure(code)


def _git_output(repo_root: Path, args: Sequence[str]) -> bytes:
    return subprocess.run(["git", "-C", str(repo_root), *args], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, shell=False).stdout


def _probe_environment(interpreter: Path) -> dict[str, Any]:
    probe = subprocess.run(
        [str(interpreter), "-c", "import platform,sys,sysconfig; print('|'.join((sys.executable,platform.python_implementation(),platform.python_version(),platform.system(),platform.machine(),sysconfig.get_platform())))"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, shell=False,
    ).stdout.decode("utf-8").strip().split("|")
    pip_output = subprocess.run([str(interpreter), "-m", "pip", "--version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, shell=False).stdout.decode("utf-8")
    freeze = subprocess.run([str(interpreter), "-m", "pip", "freeze", "--all"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, shell=False).stdout
    match = re.search(r"\bpip\s+([0-9][^\s]*)", pip_output)
    packages: list[dict[str, str]] = []
    for line in freeze.decode("utf-8").splitlines():
        if line and line.count("==") == 1:
            name, version = line.split("==", 1)
            packages.append({"name": name.lower().replace("_", "-"), "version": version})
    if len(probe) != 6 or match is None:
        raise RunnerValidationError("CANONICAL_INTERPRETER_PROBE_INVALID")
    return {
        "interpreter_executable": probe[0], "python_implementation": probe[1],
        "python_version": probe[2], "platform_system": probe[3],
        "platform_machine": probe[4], "sysconfig_platform": probe[5],
        "pip_version": match.group(1), "live_packages": sorted(packages, key=lambda item: item["name"]),
    }


def _default_phase_a_observations(config: PhaseAConfig) -> dict[str, Any]:
    obs: dict[str, Any] = {"repository_identity": "", "branch": "", "origin_head": "", "head": "", "clean": False}
    try:
        obs["repository_identity"] = _git_output(config.repo_root, ["config", "--get", "remote.origin.url"]).decode().strip()
        obs["branch"] = _git_output(config.repo_root, ["branch", "--show-current"]).decode().strip()
        obs["origin_head"] = _git_output(config.repo_root, ["rev-parse", f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}"]).decode().strip()
        obs["head"] = _git_output(config.repo_root, ["rev-parse", "HEAD"]).decode().strip()
        obs["clean"] = _git_output(config.repo_root, ["status", "--porcelain", "--untracked-files=all"]) == b""
        obs["frozen_design_commit"] = FROZEN_DESIGN_SHA
        obs["frozen_design_blob"] = _git_output(config.repo_root, ["rev-parse", f"{FROZEN_DESIGN_SHA}:{DESIGN_RELATIVE.as_posix()}"]).decode().strip()
        approval_bytes = _git_output(config.repo_root, ["show", f"HEAD:{APPROVAL_RELATIVE.as_posix()}"])
        obs["approval_record_blob"] = git_blob_sha1(approval_bytes)
        obs["approval_record"] = json.loads(approval_bytes.decode("utf-8"))
        obs["reviewed_runner_blob"] = _git_output(config.repo_root, ["rev-parse", f"{config.expected_reviewed_runner_sha}:{RUNNER_RELATIVE.as_posix()}"]).decode().strip()
        obs["current_runner_blob"] = git_blob_sha1(Path(__file__).read_bytes())
        direct_bytes = config.direct_spec.read_bytes()
        committed_direct = _git_output(config.repo_root, ["show", f"HEAD:{DIRECT_SPEC_RELATIVE.as_posix()}"])
        obs.update(direct_spec_bytes=direct_bytes, direct_spec_committed_bytes=committed_direct, direct_spec_blob=git_blob_sha1(committed_direct), direct_spec_sha256=hashlib.sha256(committed_direct).hexdigest())
        lock_bytes = _git_output(config.repo_root, ["show", f"HEAD:{LOCK_RELATIVE.as_posix()}"])
        obs.update(predecessor_lock_bytes=lock_bytes, predecessor_lock_blob=git_blob_sha1(lock_bytes), predecessor_lock_sha256=hashlib.sha256(lock_bytes).hexdigest())
        obs.update(_probe_environment(config.canonical_interpreter))
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError):
        pass
    return obs


def build_resolution_argv(repo_root: Path, wheelhouse: Path) -> list[str]:
    interpreter = repo_root / CANONICAL_INTERPRETER_RELATIVE
    lock = repo_root / LOCK_RELATIVE
    direct = repo_root / DIRECT_SPEC_RELATIVE
    return [
        str(interpreter), "-m", "pip", "download", "--dest", str(wheelhouse),
        "--only-binary=:all:", "--no-cache-dir", "--disable-pip-version-check",
        "--no-input", "--progress-bar", "off", "--retries", "0", "--timeout", "15",
        "--index-url", "https://pypi.org/simple", "--requirement", str(lock),
        "--requirement", str(direct), "--constraint", str(lock),
    ]


def sanitize_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    source = dict(os.environ if parent is None else parent)
    child = {key: value for key, value in source.items() if not key.casefold().startswith("pip_")}
    child["PIP_CONFIG_FILE"] = "NUL"
    return child


def _write_bytes(path: Path, raw: bytes, *, exclusive: bool = False) -> None:
    mode = "xb" if exclusive else "wb"
    with path.open(mode) as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _write_state(root: Path, state: Mapping[str, Any]) -> None:
    temporary = root / f".{STATE_NAME}.tmp"
    _write_bytes(temporary, canonical_json_bytes(state))
    os.replace(temporary, root / STATE_NAME)


def _initial_state(config: PhaseAConfig) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_STATE_SCHEMA, "expected_current_head": config.expected_current_head,
        "frozen_design_git_sha": FROZEN_DESIGN_SHA, "frozen_design_git_blob_sha1": FROZEN_DESIGN_BLOB,
        "approval_record_git_blob_sha1": APPROVAL_RECORD_BLOB,
        "reviewed_resolution_implementation_git_sha": config.expected_reviewed_runner_sha,
        "direct_spec_git_blob_sha1": config.expected_direct_spec_git_blob_sha1,
        "direct_spec_sha256": config.expected_direct_spec_sha256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "resolution_policy_id": RESOLUTION_POLICY_ID, "package_index_id": PACKAGE_INDEX_ID,
        "attempt_boundary_crossed": False, "human_authority_consumed": False,
        "process_started": None, "process_exit_code": None,
        "package_resolution_process_invocations": 0, "resolution_completed": False,
    }


def _validate_attempt_state(config: PhaseAConfig, state: Mapping[str, Any]) -> None:
    _require(set(state) == ATTEMPT_STATE_KEYS, "ATTEMPT_STATE_SCHEMA_INVALID")
    _require(state["schema_version"] == ATTEMPT_STATE_SCHEMA, "ATTEMPT_STATE_SCHEMA_INVALID")
    expected = _initial_state(config)
    for key in ("expected_current_head", "frozen_design_git_sha", "frozen_design_git_blob_sha1", "approval_record_git_blob_sha1", "reviewed_resolution_implementation_git_sha", "direct_spec_git_blob_sha1", "direct_spec_sha256", "predecessor_lock_git_blob_sha1", "predecessor_lock_sha256", "resolution_policy_id", "package_index_id"):
        _require(state[key] == expected[key], "ATTEMPT_STATE_PROVENANCE_MISMATCH")
    _require(state["attempt_boundary_crossed"] is True and state["human_authority_consumed"] is True, "ATTEMPT_STATE_AUTHORITY_MISSING")
    invocations = state["package_resolution_process_invocations"]
    _require(isinstance(invocations, int) and not isinstance(invocations, bool) and invocations in (0, 1), "ATTEMPT_STATE_INVOCATION_INVALID")
    started = state["process_started"]
    exit_code = state["process_exit_code"]
    if started is False:
        _require(exit_code is None and invocations == 0 and state["resolution_completed"] is False, "ATTEMPT_STATE_AMBIGUOUS")
    elif started is True:
        _require(invocations == 1 and isinstance(state["resolution_completed"], bool), "ATTEMPT_STATE_AMBIGUOUS")
        _require(exit_code is None or (isinstance(exit_code, int) and not isinstance(exit_code, bool)), "ATTEMPT_STATE_EXIT_INVALID")
    else:
        raise RunnerValidationError("ATTEMPT_STATE_AMBIGUOUS")


def run_phase_b(config: PhaseAConfig, *, execute_resolution: bool, fresh_human_authority_confirmed: bool = False, observations: Mapping[str, Any] | None = None, popen_factory: Callable[..., Any] = subprocess.Popen, parent_environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run the one-shot resolver only when explicitly requested and authorized."""

    if execute_resolution is not True:
        raise RunnerValidationError("EXPLICIT_EXECUTE_RESOLUTION_REQUIRED")
    preflight = run_phase_a(config, observations)
    if preflight["status"] != "PASS":
        raise RunnerValidationError(f"PHASE_A_{preflight['failure_code']}")
    if fresh_human_authority_confirmed is not True:
        raise RunnerValidationError("FRESH_HUMAN_AUTHORITY_REQUIRED")
    root = config.durable_root
    try:
        root.mkdir()
        (root / WHEELHOUSE_NAME).mkdir()
        _write_bytes(root / STDOUT_NAME, b"")
        _write_bytes(root / STDERR_NAME, b"")
        state = _initial_state(config)
        _write_state(root, state)
        state.update(attempt_boundary_crossed=True, human_authority_consumed=True)
        _write_state(root, state)
        argv = build_resolution_argv(config.repo_root, root / WHEELHOUSE_NAME)
        child_env = sanitize_environment(parent_environment)
        with (root / STDOUT_NAME).open("ab") as stdout_handle, (root / STDERR_NAME).open("ab") as stderr_handle:
            try:
                process = popen_factory(argv, shell=False, env=child_env, stdout=stdout_handle, stderr=stderr_handle)
            except OSError:
                state.update(process_started=False, process_exit_code=None, package_resolution_process_invocations=0, resolution_completed=False)
                _write_state(root, state)
                return {"status": "FAIL", "failure_code": "RESOLUTION_PROCESS_FAILURE", "process_started": False, "process_exit_code": None, "package_resolution_process_invocations": 0, "human_authority_consumed": True}
            state.update(process_started=True, process_exit_code=None, package_resolution_process_invocations=1)
            _write_state(root, state)
            exit_code = process.wait()
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise RunnerValidationError("PROCESS_EXIT_CODE_INVALID")
        state.update(process_started=True, process_exit_code=exit_code, package_resolution_process_invocations=1, resolution_completed=(exit_code == 0))
        _write_state(root, state)
        return {"status": "PASS" if exit_code == 0 else "FAIL", "failure_code": "NONE" if exit_code == 0 else "RESOLUTION_PROCESS_FAILURE", "process_started": True, "process_exit_code": exit_code, "package_resolution_process_invocations": 1, "human_authority_consumed": True}
    except (OSError, RunnerValidationError):
        raise


def _base_evidence(config: PhaseAConfig, *, status: str, failure_code: str, process_started: bool, process_exit_code: int | None, resolution_completed: bool, candidate_created: bool, candidate_sha: str | None, package_count: int | None, invocations: int) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA, "artifact_status": "WINDOWS_RESOLUTION_EVIDENCE", "status": status,
        "failure_code": failure_code, "study": STUDY, "frozen_design_git_sha": FROZEN_DESIGN_SHA,
        "frozen_design_git_blob_sha1": FROZEN_DESIGN_BLOB,
        "approval_record_git_blob_sha1": APPROVAL_RECORD_BLOB,
        "reviewed_resolution_implementation_git_sha": config.expected_reviewed_runner_sha,
        "direct_spec_git_blob_sha1": config.expected_direct_spec_git_blob_sha1,
        "direct_spec_sha256": config.expected_direct_spec_sha256, "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256, "resolution_policy_id": RESOLUTION_POLICY_ID,
        "package_index_id": PACKAGE_INDEX_ID, "process_started": process_started,
        "process_exit_code": process_exit_code, "resolution_completed": resolution_completed,
        "candidate_artifact_created": candidate_created, "successor_lock_candidate_sha256": candidate_sha,
        "resolved_package_count": package_count, "package_resolution_process_invocations": invocations,
        "human_authority_consumed": True, "package_installations": 0,
        "alternate_venv_created": False, "t0_runs": 0, "payload_reads": 0,
    }


def _build_candidate(config: PhaseAConfig, wheels: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    resolved_wheels = [dict(item) for item in wheels]
    packages = [{"name": item["name"], "version": item["version"]} for item in resolved_wheels]
    package_sets = validate_resolved_packages(packages)
    candidate: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA, "artifact_status": "WINDOWS_RESOLUTION_CANDIDATE_NOT_INSTALL_AUTHORITY", "study": STUDY,
        "frozen_design_git_sha": FROZEN_DESIGN_SHA, "frozen_design_git_blob_sha1": FROZEN_DESIGN_BLOB,
        "approval_record_git_blob_sha1": APPROVAL_RECORD_BLOB, "reviewed_resolution_implementation_git_sha": config.expected_reviewed_runner_sha,
        "direct_spec_git_blob_sha1": config.expected_direct_spec_git_blob_sha1, "direct_spec_sha256": config.expected_direct_spec_sha256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB, "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "predecessor_package_count": len(PREDECESSOR_PACKAGE_SET), "python_version": "3.12.10", "platform_system": "Windows",
        "platform_machine": "AMD64", "sysconfig_platform": "win-amd64", "resolution_policy_id": RESOLUTION_POLICY_ID,
        "resolved_packages": packages, "resolved_package_count": len(packages), "resolved_wheels": resolved_wheels,
        "predecessor_pin_drift_count": 0, "lightgbm_version": dict(package_sets["successor"])["lightgbm"],
        "scikit_learn_version": dict(package_sets["successor"])["scikit-learn"],
    }
    validate_lock_candidate(candidate, expected_reviewed_sha=config.expected_reviewed_runner_sha, expected_direct_blob=config.expected_direct_spec_git_blob_sha1, expected_direct_sha=config.expected_direct_spec_sha256)
    return candidate


def run_phase_c(config: PhaseAConfig, *, expected_candidate_sha256: str | None = None) -> dict[str, Any]:
    """Inspect one completed synthetic/future attempt without network or imports."""

    root = config.durable_root
    try:
        state = json.loads((root / STATE_NAME).read_text(encoding="utf-8"))
        _validate_attempt_state(config, state)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, RunnerValidationError) as error:
        raise RunnerValidationError("ATTEMPT_STATE_INVALID") from error
    started = state["process_started"]
    exit_code = state["process_exit_code"]
    invocations = state["package_resolution_process_invocations"]
    candidate: dict[str, Any] | None = None
    wheels: tuple[dict[str, str], ...] | None = None
    if started is False:
        failure_code = "RESOLUTION_PROCESS_FAILURE"
    elif exit_code != 0:
        failure_code = "RESOLUTION_PROCESS_FAILURE"
    else:
        failure_code, wheels = inspect_wheelhouse(root / WHEELHOUSE_NAME)
        if failure_code == "NONE" and wheels is not None:
            try:
                candidate = _build_candidate(config, wheels)
            except ContractValidationError as error:
                failure_code = str(error) if str(error) in {"PREDECESSOR_PIN_DRIFT", "REQUIRED_DIRECT_DISTRIBUTION_MISSING"} else "WHEEL_PROVENANCE_FAILURE"
                candidate = None
    if failure_code == "NONE" and candidate is not None:
        candidate_bytes = canonical_json_bytes(candidate)
        candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
        evidence = _base_evidence(config, status="PASS", failure_code="NONE", process_started=True, process_exit_code=0, resolution_completed=True, candidate_created=True, candidate_sha=candidate_sha, package_count=candidate["resolved_package_count"], invocations=invocations)
        validate_lock_candidate(candidate, expected_reviewed_sha=config.expected_reviewed_runner_sha, expected_direct_blob=config.expected_direct_spec_git_blob_sha1, expected_direct_sha=config.expected_direct_spec_sha256)
        validate_evidence(evidence, expected_reviewed_sha=config.expected_reviewed_runner_sha, expected_direct_blob=config.expected_direct_spec_git_blob_sha1, expected_direct_sha=config.expected_direct_spec_sha256, expected_candidate_sha=candidate_sha)
        _write_bytes(root / CANDIDATE_NAME, candidate_bytes, exclusive=True)
        _write_bytes(root / EVIDENCE_NAME, canonical_json_bytes(evidence), exclusive=True)
        return {"status": "PASS", "failure_code": "NONE", "candidate_artifact_created": True, "candidate_sha256": candidate_sha, "resolved_package_count": candidate["resolved_package_count"]}
    evidence = _base_evidence(config, status="FAIL", failure_code=failure_code, process_started=bool(started), process_exit_code=exit_code, resolution_completed=bool(state["resolution_completed"]), candidate_created=False, candidate_sha=None, package_count=None, invocations=invocations)
    validate_evidence(evidence, expected_reviewed_sha=config.expected_reviewed_runner_sha, expected_direct_blob=config.expected_direct_spec_git_blob_sha1, expected_direct_sha=config.expected_direct_spec_sha256)
    _write_bytes(root / EVIDENCE_NAME, canonical_json_bytes(evidence), exclusive=True)
    return {"status": "FAIL", "failure_code": failure_code, "candidate_artifact_created": False}
