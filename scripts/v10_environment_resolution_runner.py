"""Purely testable V10 Windows resolution Phase A/B/C runner.

The default probes and Phase-B subprocess path are intentionally explicit,
but this module has no import-time I/O.  Tests inject all observations and
the process launcher; no test needs a package index, a real resolver, or the
canonical environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from scripts.v10_environment_extension_contract import (
        FROZEN_V10_DESIGN_SHA,
        PREDECESSOR_LOCK_BLOB_SHA1,
        PREDECESSOR_LOCK_SHA256,
        PREDECESSOR_PACKAGE_SET,
        ContractValidationError,
        SHA1_RE,
        SHA256_RE,
        _validate_wheel_manifest,
        derive_package_sets,
        inspect_wheel_file,
        validate_resolution_evidence,
        validate_successor_lock_candidate,
    )
except ModuleNotFoundError:  # direct ``python scripts/<runner>.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.v10_environment_extension_contract import (
        FROZEN_V10_DESIGN_SHA,
        PREDECESSOR_LOCK_BLOB_SHA1,
        PREDECESSOR_LOCK_SHA256,
        PREDECESSOR_PACKAGE_SET,
        ContractValidationError,
        SHA1_RE,
        SHA256_RE,
        _validate_wheel_manifest,
        derive_package_sets,
        inspect_wheel_file,
        validate_resolution_evidence,
        validate_successor_lock_candidate,
    )


REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
CANONICAL_INTERPRETER_RELATIVE = Path(".venv-real-execution") / "Scripts" / "python.exe"
CANONICAL_ENVIRONMENT_RELATIVE = Path(".venv-real-execution")
EXTENSION_DESIGN_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_EXTENSION_DESIGN_DRAFT.md")
RUNNER_RELATIVE = Path("scripts/v10_environment_resolution_runner.py")
DIRECT_SPEC_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt")
LOCK_RELATIVE = Path("requirements-real-execution.lock.txt")
WHEELHOUSE_NAME = "wheelhouse"
STDOUT_NAME = "stdout.txt"
STDERR_NAME = "stderr.txt"
STATE_NAME = "attempt_state.json"
CANDIDATE_NAME = "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json"
EVIDENCE_NAME = "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE.json"
ATTEMPT_STATE_SCHEMA = "V10_CANONICAL_ENVIRONMENT_RESOLUTION_ATTEMPT_STATE_V1"
RESOLUTION_POLICY_ID = "PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1"
PACKAGE_INDEX_ID = "PYPI_OFFICIAL_SIMPLE"
ATTEMPT_STATE_KEYS = frozenset(
    {
        "schema_version",
        "expected_current_head",
        "frozen_v10_design_git_sha",
        "extension_design_git_sha",
        "reviewed_resolution_implementation_git_sha",
        "direct_spec_git_blob_sha1",
        "direct_spec_sha256",
        "predecessor_lock_git_blob_sha1",
        "predecessor_lock_sha256",
        "resolution_policy_id",
        "package_index_id",
        "attempt_boundary_crossed",
        "human_authority_consumed",
        "process_started",
        "process_exit_code",
        "package_resolution_process_invocations",
        "resolution_completed",
    }
)
DIRECT_SPEC_BYTES = (
    b"pandas\n"
    b"xlrd==2.0.2\n"
    b"pdfplumber==0.11.10\n"
    b"pandas-market-calendars==5.4.0\n"
)
DIRECT_SPEC_SHA256 = hashlib.sha256(DIRECT_SPEC_BYTES).hexdigest()


class RunnerValidationError(ContractValidationError):
    """A safe, fail-closed runner contract error."""


@dataclass(frozen=True)
class PhaseAConfig:
    repo_root: Path
    expected_current_head: str
    expected_extension_design_sha: str
    expected_reviewed_runner_sha: str
    expected_direct_spec_git_blob_sha1: str
    expected_direct_spec_sha256: str
    durable_root: Path
    protected_environment: Path | None = None
    governed_roots: tuple[Path, ...] = ()

    @property
    def canonical_interpreter(self) -> Path:
        return self.repo_root / CANONICAL_INTERPRETER_RELATIVE

    @property
    def direct_spec(self) -> Path:
        return self.repo_root / DIRECT_SPEC_RELATIVE

    @property
    def lock_file(self) -> Path:
        return self.repo_root / LOCK_RELATIVE


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the reviewed UTF-8, sorted, compact JSON representation."""

    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )


def validate_direct_spec_bytes(raw: bytes) -> str:
    """Validate exact direct-spec bytes and return their SHA-256."""

    if raw != DIRECT_SPEC_BYTES:
        raise RunnerValidationError("DIRECT_SPEC_BYTES_MISMATCH")
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & 0x400)


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _nearest_existing_ancestor(path: Path) -> Path:
    node = path
    while not os.path.lexists(node):
        if node.parent == node:
            break
        node = node.parent
    return node


def _path_has_unsafe_existing_component(path: Path) -> bool:
    node = _nearest_existing_ancestor(path)
    while True:
        if _is_reparse_or_symlink(node):
            return True
        if node.parent == node:
            return False
        node = node.parent


def validate_durable_root(
    durable_root: Path,
    *,
    repo_root: Path,
    protected_environment: Path,
    governed_roots: Sequence[Path] = (),
) -> str:
    """Read-only, fail-closed safety validation for a future attempt root."""

    root = Path(durable_root)
    if not root.is_absolute():
        return "DURABLE_ROOT_NOT_ABSOLUTE"
    if root.exists() or os.path.lexists(root):
        return "DURABLE_ROOT_ALREADY_EXISTS"
    if _path_has_unsafe_existing_component(root):
        return "DURABLE_ROOT_REPARSE_OR_SYMLINK"
    try:
        resolved_root = Path(os.path.realpath(root))
        resolved_repo = Path(os.path.realpath(repo_root))
        resolved_protected = Path(os.path.realpath(protected_environment))
        resolved_governed = [Path(os.path.realpath(item)) for item in governed_roots]
    except OSError:
        return "DURABLE_ROOT_SAFETY_UNDETERMINED"
    forbidden = [resolved_repo, resolved_protected, *resolved_governed]
    if any(_path_inside(resolved_root, item) or _path_inside(item, resolved_root) for item in forbidden):
        return "DURABLE_ROOT_GOVERNED_PATH_OVERLAP"
    if not _nearest_existing_ancestor(root).exists():
        return "DURABLE_ROOT_PARENT_MISSING"
    return "DURABLE_ROOT_OK"


def _safe_sha(value: Any, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RunnerValidationError(f"{label}_INVALID")


def _safe_package_set(packages: Any) -> tuple[tuple[str, str], ...]:
    try:
        result = derive_package_sets(list(packages))
    except ContractValidationError as error:
        raise RunnerValidationError(str(error)) from error
    successor = result["successor"]
    if successor != PREDECESSOR_PACKAGE_SET:
        raise RunnerValidationError("PREDECESSOR_PACKAGE_SET_MISMATCH")
    return successor


def _parse_lock_packages(raw: bytes) -> list[dict[str, str]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RunnerValidationError("PREDECESSOR_LOCK_UNREADABLE") from error
    packages: list[dict[str, str]] = []
    for line in lines:
        if not line or line.count("==") != 1:
            raise RunnerValidationError("PREDECESSOR_LOCK_INVALID")
        name, version = line.split("==")
        packages.append({"name": name, "version": version})
    return packages


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


def _phase_a_failure(config: PhaseAConfig, failure_code: str, **extra: Any) -> dict[str, Any]:
    result = {
        "status": "FAIL",
        "failure_code": failure_code,
        "repository_identity_verified": False,
        "authoritative_branch_verified": False,
        "expected_head_verified": False,
        "working_tree_clean": False,
        "direct_spec_verified": False,
        "predecessor_lock_verified": False,
        "canonical_interpreter_verified": False,
        "live_predecessor_package_set_verified": False,
        "durable_root_status": "NOT_CHECKED",
    }
    result.update(extra)
    return result


def run_phase_a(config: PhaseAConfig, observations: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run no-network/read-only preflight; injected observations are test-only."""

    obs = dict(_default_phase_a_observations(config) if observations is None else observations)
    for value, pattern, label in (
        (config.expected_current_head, SHA1_RE, "expected head"),
        (config.expected_extension_design_sha, SHA1_RE, "expected extension design SHA"),
        (config.expected_reviewed_runner_sha, SHA1_RE, "expected runner SHA"),
        (config.expected_direct_spec_git_blob_sha1, SHA1_RE, "expected direct spec blob SHA"),
        (config.expected_direct_spec_sha256, SHA256_RE, "expected direct spec SHA"),
    ):
        _safe_sha(value, pattern, label)

    if not _repo_identity_matches(obs.get("repository_identity")):
        return _phase_a_failure(config, "REPOSITORY_IDENTITY_MISMATCH")
    if obs.get("branch") != AUTHORITATIVE_BRANCH:
        return _phase_a_failure(config, "AUTHORITATIVE_BRANCH_MISMATCH", repository_identity_verified=True)
    if obs.get("head") != config.expected_current_head:
        return _phase_a_failure(config, "EXPECTED_HEAD_MISMATCH", repository_identity_verified=True, authoritative_branch_verified=True)
    if obs.get("clean") is not True:
        return _phase_a_failure(
            config,
            "WORKING_TREE_DIRTY",
            repository_identity_verified=True,
            authoritative_branch_verified=True,
            expected_head_verified=True,
        )
    if obs.get("frozen_design_ok") is not True:
        return _phase_a_failure(config, "FROZEN_DESIGN_BINDING_FAILURE", repository_identity_verified=True, authoritative_branch_verified=True, expected_head_verified=True, working_tree_clean=True)
    if obs.get("extension_design_ok") is not True or obs.get("runner_binding_ok") is not True:
        return _phase_a_failure(config, "IMPLEMENTATION_PROVENANCE_FAILURE", repository_identity_verified=True, authoritative_branch_verified=True, expected_head_verified=True, working_tree_clean=True)

    try:
        direct_bytes = obs["direct_spec_bytes"]
        committed_direct_bytes = obs["direct_spec_committed_bytes"]
        if direct_bytes != committed_direct_bytes or direct_bytes != DIRECT_SPEC_BYTES:
            raise RunnerValidationError("DIRECT_SPEC_BYTES_MISMATCH")
        direct_sha = hashlib.sha256(committed_direct_bytes).hexdigest()
        if direct_sha != config.expected_direct_spec_sha256 or obs.get("direct_spec_sha256") != direct_sha:
            raise RunnerValidationError("DIRECT_SPEC_SHA256_MISMATCH")
        if _git_blob_sha1(committed_direct_bytes) != config.expected_direct_spec_git_blob_sha1 or obs.get("direct_spec_git_blob_sha1") != config.expected_direct_spec_git_blob_sha1:
            raise RunnerValidationError("DIRECT_SPEC_GIT_BLOB_MISMATCH")
    except (KeyError, TypeError, RunnerValidationError):
        return _phase_a_failure(config, "DIRECT_SPEC_BINDING_FAILURE", repository_identity_verified=True, authoritative_branch_verified=True, expected_head_verified=True, working_tree_clean=True)

    try:
        lock_bytes = obs["predecessor_lock_committed_bytes"]
        if hashlib.sha256(lock_bytes).hexdigest() != PREDECESSOR_LOCK_SHA256:
            raise RunnerValidationError("PREDECESSOR_LOCK_SHA256_MISMATCH")
        if _git_blob_sha1(lock_bytes) != PREDECESSOR_LOCK_BLOB_SHA1:
            raise RunnerValidationError("PREDECESSOR_LOCK_GIT_BLOB_MISMATCH")
        if obs.get("predecessor_lock_git_blob_sha1") != PREDECESSOR_LOCK_BLOB_SHA1 or obs.get("predecessor_lock_sha256") != PREDECESSOR_LOCK_SHA256:
            raise RunnerValidationError("PREDECESSOR_LOCK_BINDING_MISMATCH")
        _safe_package_set(_parse_lock_packages(lock_bytes))
        _safe_package_set(obs["live_packages"])
    except (KeyError, TypeError, RunnerValidationError):
        return _phase_a_failure(config, "PREDECESSOR_BASELINE_MISMATCH", repository_identity_verified=True, authoritative_branch_verified=True, expected_head_verified=True, working_tree_clean=True, direct_spec_verified=True)

    expected_interpreter = str(config.canonical_interpreter.resolve())
    actual_interpreter = obs.get("interpreter_executable")
    interpreter_ok = (
        isinstance(actual_interpreter, str)
        and str(Path(actual_interpreter).resolve()) == expected_interpreter
        and obs.get("python_implementation") == "CPython"
        and obs.get("python_version") == "3.12.10"
        and obs.get("platform_system") == "Windows"
        and obs.get("platform_machine") == "AMD64"
        and obs.get("sysconfig_platform") == "win-amd64"
        and obs.get("pip_version") == "25.0.1"
    )
    if not interpreter_ok:
        return _phase_a_failure(config, "CANONICAL_INTERPRETER_BINDING_FAILURE", repository_identity_verified=True, authoritative_branch_verified=True, expected_head_verified=True, working_tree_clean=True, direct_spec_verified=True, predecessor_lock_verified=True)

    protected = config.protected_environment or (config.repo_root / CANONICAL_ENVIRONMENT_RELATIVE)
    root_status = validate_durable_root(
        config.durable_root,
        repo_root=config.repo_root,
        protected_environment=protected,
        governed_roots=config.governed_roots,
    )
    if root_status != "DURABLE_ROOT_OK":
        return _phase_a_failure(
            config,
            "DURABLE_ROOT_SAFETY_FAILURE",
            repository_identity_verified=True,
            authoritative_branch_verified=True,
            expected_head_verified=True,
            working_tree_clean=True,
            direct_spec_verified=True,
            predecessor_lock_verified=True,
            canonical_interpreter_verified=True,
            live_predecessor_package_set_verified=True,
            durable_root_status=root_status,
        )
    return {
        "status": "PASS",
        "failure_code": "NONE",
        "repository_identity_verified": True,
        "authoritative_branch_verified": True,
        "expected_head_verified": True,
        "working_tree_clean": True,
        "frozen_design_verified": True,
        "extension_design_verified": True,
        "runner_binding_verified": True,
        "direct_spec_verified": True,
        "direct_spec_sha256": config.expected_direct_spec_sha256,
        "predecessor_lock_verified": True,
        "predecessor_package_count": len(PREDECESSOR_PACKAGE_SET),
        "canonical_interpreter_verified": True,
        "live_predecessor_package_set_verified": True,
        "durable_root_status": root_status,
        "network_requests": 0,
        "writes": 0,
        "human_authority_consumed": False,
    }


def _run_local(repo_root: Path, args: Sequence[str], *, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    )
    return completed.stdout


def _default_phase_a_observations(config: PhaseAConfig) -> dict[str, Any]:
    """Gather only local, no-network facts for a real future Phase-A call."""

    obs: dict[str, Any] = {
        "repository_identity": "",
        "branch": "",
        "head": "",
        "clean": False,
        "frozen_design_ok": False,
        "extension_design_ok": False,
        "runner_binding_ok": False,
    }
    try:
        remote = _run_local(config.repo_root, ["config", "--get", "remote.origin.url"]).decode().strip()
        obs["repository_identity"] = remote
        obs["branch"] = _run_local(config.repo_root, ["branch", "--show-current"]).decode().strip()
        obs["head"] = _run_local(config.repo_root, ["rev-parse", "HEAD"]).decode().strip()
        obs["clean"] = _run_local(config.repo_root, ["status", "--porcelain", "--untracked-files=all"]) == b""
        obs["frozen_design_ok"] = _run_local(config.repo_root, ["cat-file", "-e", f"{FROZEN_V10_DESIGN_SHA}^{{commit}}"]) == b""
        obs["extension_design_ok"] = _run_local(config.repo_root, ["cat-file", "-e", f"{config.expected_extension_design_sha}^{{commit}}"]) == b""
        expected_runner_blob = _run_local(config.repo_root, ["rev-parse", f"{config.expected_reviewed_runner_sha}:{RUNNER_RELATIVE.as_posix()}"]).decode().strip()
        current_runner_blob = _run_local(config.repo_root, ["hash-object", "--", str(repo_root_relative(config.repo_root, RUNNER_RELATIVE))]).decode().strip()
        obs["runner_binding_ok"] = bool(expected_runner_blob == current_runner_blob)
        direct_committed = _run_local(config.repo_root, ["show", f"HEAD:{DIRECT_SPEC_RELATIVE.as_posix()}"])
        lock_committed = _run_local(config.repo_root, ["show", f"HEAD:{LOCK_RELATIVE.as_posix()}"])
        obs.update(
            direct_spec_bytes=config.direct_spec.read_bytes(),
            direct_spec_committed_bytes=direct_committed,
            direct_spec_git_blob_sha1=_run_local(config.repo_root, ["rev-parse", f"HEAD:{DIRECT_SPEC_RELATIVE.as_posix()}"]).decode().strip(),
            direct_spec_sha256=hashlib.sha256(direct_committed).hexdigest(),
            predecessor_lock_committed_bytes=lock_committed,
            predecessor_lock_git_blob_sha1=_run_local(config.repo_root, ["rev-parse", f"HEAD:{LOCK_RELATIVE.as_posix()}"]).decode().strip(),
            predecessor_lock_sha256=hashlib.sha256(lock_committed).hexdigest(),
        )
        obs.update(_probe_canonical_environment(config.canonical_interpreter))
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError):
        return obs
    return obs


def repo_root_relative(repo_root: Path, relative: Path) -> Path:
    return repo_root / relative


def _probe_canonical_environment(interpreter: Path) -> dict[str, Any]:
    probe = subprocess.run(
        [str(interpreter), "-c", "import platform,sys,sysconfig; print('|'.join((sys.executable,platform.python_implementation(),platform.python_version(),platform.system(),platform.machine(),sysconfig.get_platform())))"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    ).stdout.decode().strip().split("|")
    pip_output = subprocess.run(
        [str(interpreter), "-m", "pip", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    ).stdout.decode().strip()
    freeze = subprocess.run(
        [str(interpreter), "-m", "pip", "freeze", "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    ).stdout.decode()
    packages = []
    for line in freeze.splitlines():
        if "==" not in line or line.startswith("-"):
            raise RunnerValidationError("LIVE_PACKAGE_SET_UNPARSEABLE")
        name, version = line.split("==", 1)
        packages.append({"name": name, "version": version})
    pip_match = re.search(r"\bpip\s+([0-9][^\s]*)", pip_output)
    if len(probe) != 6 or pip_match is None:
        raise RunnerValidationError("CANONICAL_INTERPRETER_PROBE_INVALID")
    return {
        "interpreter_executable": probe[0],
        "python_implementation": probe[1],
        "python_version": probe[2],
        "platform_system": probe[3],
        "platform_machine": probe[4],
        "sysconfig_platform": probe[5],
        "pip_version": pip_match.group(1),
        "live_packages": packages,
    }


def _atomic_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _atomic_create_no_overwrite(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise RunnerValidationError("DURABLE_FILE_ALREADY_EXISTS") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def build_resolution_argv(repo_root: Path, wheelhouse: Path) -> list[str]:
    interpreter = repo_root / CANONICAL_INTERPRETER_RELATIVE
    lock = repo_root / LOCK_RELATIVE
    direct_spec = repo_root / DIRECT_SPEC_RELATIVE
    return [
        str(interpreter),
        "-m",
        "pip",
        "download",
        "--dest",
        str(wheelhouse),
        "--only-binary=:all:",
        "--no-cache-dir",
        "--disable-pip-version-check",
        "--no-input",
        "--progress-bar",
        "off",
        "--retries",
        "0",
        "--timeout",
        "15",
        "--index-url",
        "https://pypi.org/simple",
        "--requirement",
        str(lock),
        "--requirement",
        str(direct_spec),
        "--constraint",
        str(lock),
    ]


def _sanitize_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    source = dict(os.environ if parent is None else parent)
    child = {key: value for key, value in source.items() if not key.casefold().startswith("pip_")}
    child["PIP_CONFIG_FILE"] = "NUL"
    return child


def _write_attempt_state(root: Path, state: Mapping[str, Any]) -> None:
    _atomic_replace(root / STATE_NAME, canonical_json_bytes(state))


def _initial_attempt_state(config: PhaseAConfig) -> dict[str, Any]:
    """Return the closed internal state before the human-gated boundary."""

    return {
        "schema_version": ATTEMPT_STATE_SCHEMA,
        "expected_current_head": config.expected_current_head,
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": config.expected_extension_design_sha,
        "reviewed_resolution_implementation_git_sha": config.expected_reviewed_runner_sha,
        "direct_spec_git_blob_sha1": config.expected_direct_spec_git_blob_sha1,
        "direct_spec_sha256": config.expected_direct_spec_sha256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "resolution_policy_id": RESOLUTION_POLICY_ID,
        "package_index_id": PACKAGE_INDEX_ID,
        "attempt_boundary_crossed": False,
        "human_authority_consumed": False,
        "process_started": None,
        "process_exit_code": None,
        "package_resolution_process_invocations": 0,
        "resolution_completed": False,
    }


def _validate_attempt_state(config: PhaseAConfig, state: Mapping[str, Any]) -> None:
    """Fail closed unless Phase-C input is the exact Phase-B durable state."""

    if set(state) != ATTEMPT_STATE_KEYS:
        raise RunnerValidationError("ATTEMPT_STATE_SCHEMA_INVALID")
    if state["schema_version"] != ATTEMPT_STATE_SCHEMA:
        raise RunnerValidationError("ATTEMPT_STATE_SCHEMA_INVALID")
    for value, pattern, label in (
        (state["expected_current_head"], SHA1_RE, "attempt expected head"),
        (state["frozen_v10_design_git_sha"], SHA1_RE, "attempt frozen design SHA"),
        (state["extension_design_git_sha"], SHA1_RE, "attempt extension design SHA"),
        (state["reviewed_resolution_implementation_git_sha"], SHA1_RE, "attempt runner SHA"),
        (state["direct_spec_git_blob_sha1"], SHA1_RE, "attempt direct spec blob SHA"),
        (state["direct_spec_sha256"], SHA256_RE, "attempt direct spec SHA"),
        (state["predecessor_lock_git_blob_sha1"], SHA1_RE, "attempt predecessor blob SHA"),
        (state["predecessor_lock_sha256"], SHA256_RE, "attempt predecessor SHA"),
    ):
        _safe_sha(value, pattern, label)
    exact_bindings = {
        "expected_current_head": config.expected_current_head,
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": config.expected_extension_design_sha,
        "reviewed_resolution_implementation_git_sha": config.expected_reviewed_runner_sha,
        "direct_spec_git_blob_sha1": config.expected_direct_spec_git_blob_sha1,
        "direct_spec_sha256": config.expected_direct_spec_sha256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "resolution_policy_id": RESOLUTION_POLICY_ID,
        "package_index_id": PACKAGE_INDEX_ID,
    }
    if any(state[key] != value for key, value in exact_bindings.items()):
        raise RunnerValidationError("ATTEMPT_STATE_PROVENANCE_MISMATCH")
    if state["attempt_boundary_crossed"] is not True or state["human_authority_consumed"] is not True:
        raise RunnerValidationError("ATTEMPT_STATE_AMBIGUOUS")
    if not isinstance(state["resolution_completed"], bool):
        raise RunnerValidationError("ATTEMPT_STATE_SCHEMA_INVALID")
    started = state["process_started"]
    exit_code = state["process_exit_code"]
    invocations = state["package_resolution_process_invocations"]
    if not isinstance(invocations, int) or isinstance(invocations, bool) or invocations < 0:
        raise RunnerValidationError("ATTEMPT_STATE_SCHEMA_INVALID")
    if invocations not in (0, 1):
        raise RunnerValidationError("ATTEMPT_STATE_SCHEMA_INVALID")
    if started is False:
        if exit_code is not None or invocations != 0 or state["resolution_completed"] is not False:
            raise RunnerValidationError("ATTEMPT_STATE_AMBIGUOUS")
    elif started is True:
        if invocations != 1:
            raise RunnerValidationError("ATTEMPT_STATE_AMBIGUOUS")
        if exit_code is None:
            if state["resolution_completed"] is not False:
                raise RunnerValidationError("ATTEMPT_STATE_AMBIGUOUS")
        elif not isinstance(exit_code, int) or isinstance(exit_code, bool) or state["resolution_completed"] is not True:
            raise RunnerValidationError("ATTEMPT_STATE_AMBIGUOUS")
    else:
        raise RunnerValidationError("ATTEMPT_STATE_AMBIGUOUS")


def run_phase_b(
    config: PhaseAConfig,
    *,
    execute_resolution: bool,
    fresh_human_authority_confirmed: bool = False,
    observations: Mapping[str, Any] | None = None,
    popen_factory: Callable[..., Any] = subprocess.Popen,
    parent_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute the future one-shot resolver only when explicitly authorized."""

    if not execute_resolution:
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
    except (FileExistsError, OSError) as error:
        raise RunnerValidationError("DURABLE_ATTEMPT_ROOT_CREATION_FAILURE") from error
    wheelhouse = root / WHEELHOUSE_NAME
    if any(wheelhouse.iterdir()):
        raise RunnerValidationError("WHEELHOUSE_NOT_EMPTY_BEFORE_LAUNCH")

    stdout_path = root / STDOUT_NAME
    stderr_path = root / STDERR_NAME
    state = _initial_attempt_state(config)
    _write_attempt_state(root, state)
    argv = build_resolution_argv(config.repo_root, wheelhouse)
    child_environment = _sanitize_environment(parent_environment)
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        state.update(attempt_boundary_crossed=True, human_authority_consumed=True)
        _write_attempt_state(root, state)
        try:
            process = popen_factory(
                argv,
                shell=False,
                env=child_environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
            )
        except OSError:
            state.update(process_started=False, process_exit_code=None, package_resolution_process_invocations=0)
            _write_attempt_state(root, state)
            return {
                "status": "FAIL",
                "failure_code": "RESOLUTION_PROCESS_FAILURE",
                "process_started": False,
                "process_exit_code": None,
                "package_resolution_process_invocations": 0,
                "human_authority_consumed": True,
            }
        state.update(process_started=True, process_exit_code=None, package_resolution_process_invocations=1)
        _write_attempt_state(root, state)
        exit_code = process.wait()
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise RunnerValidationError("PROCESS_EXIT_CODE_INVALID")
    state.update(process_started=True, process_exit_code=exit_code, package_resolution_process_invocations=1, resolution_completed=True)
    _write_attempt_state(root, state)
    return {
        "status": "PASS" if exit_code == 0 else "FAIL",
        "failure_code": "NONE" if exit_code == 0 else "RESOLUTION_PROCESS_FAILURE",
        "process_started": True,
        "process_exit_code": exit_code,
        "package_resolution_process_invocations": 1,
        "human_authority_consumed": True,
    }


def _base_resolution_evidence(
    *,
    config: PhaseAConfig,
    status: str,
    failure_code: str,
    process_started: bool,
    process_exit_code: int | None,
    resolution_completed: bool,
    candidate_artifact_created: bool,
    candidate_sha: str | None,
    package_count: int | None,
    invocations: int,
) -> dict[str, Any]:
    return {
        "schema_version": "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE_V1",
        "artifact_status": "WINDOWS_RESOLUTION_EVIDENCE",
        "status": status,
        "failure_code": failure_code,
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": config.expected_extension_design_sha,
        "reviewed_resolution_implementation_git_sha": config.expected_reviewed_runner_sha,
        "direct_spec_git_blob_sha1": config.expected_direct_spec_git_blob_sha1,
        "direct_spec_sha256": config.expected_direct_spec_sha256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "resolution_policy_id": RESOLUTION_POLICY_ID,
        "process_started": process_started,
        "process_exit_code": process_exit_code,
        "resolution_completed": resolution_completed,
        "candidate_artifact_created": candidate_artifact_created,
        "successor_lock_candidate_sha256": candidate_sha,
        "resolved_package_count": package_count,
        "package_index_id": PACKAGE_INDEX_ID,
        "package_resolution_process_invocations": invocations,
        "human_authority_consumed": True,
        "package_installations": 0,
        "alternate_venv_created": False,
        "calendar_imports": 0,
        "calendar_dates_inspected": 0,
    }


def _validate_and_write_evidence(config: PhaseAConfig, evidence: Mapping[str, Any], destination: Path) -> None:
    validate_resolution_evidence(
        evidence,
        expected_extension_design_sha=config.expected_extension_design_sha,
        expected_reviewed_resolution_implementation_sha=config.expected_reviewed_runner_sha,
        expected_direct_spec_git_blob_sha1=config.expected_direct_spec_git_blob_sha1,
        expected_direct_spec_sha256=config.expected_direct_spec_sha256,
        expected_successor_lock_candidate_sha256=evidence["successor_lock_candidate_sha256"],
    )
    _atomic_create_no_overwrite(destination, canonical_json_bytes(evidence))


def _inspect_phase_c_wheelhouse(wheelhouse: Path) -> tuple[str | None, list[dict[str, str]] | None]:
    """Classify one wheelhouse using the frozen Phase-C precedence order."""

    try:
        entries = list(wheelhouse.iterdir())
    except OSError:
        return "REQUIRED_DISTRIBUTION_MISSING", None
    if not entries:
        return "REQUIRED_DISTRIBUTION_MISSING", None
    if any(not entry.is_file() for entry in entries):
        return "RESOLUTION_REPORT_INVALID", None
    source_present = any(not entry.name.lower().endswith(".whl") for entry in entries)
    wheel_entries = [entry for entry in entries if entry.name.lower().endswith(".whl")]
    try:
        wheels = [inspect_wheel_file(entry) for entry in wheel_entries]
        wheels.sort(key=lambda item: item["name"])
        manifest = list(_validate_wheel_manifest(wheels))
    except (OSError, ContractValidationError, ValueError):
        return "RESOLUTION_REPORT_INVALID", None
    pairs = tuple((item["name"], item["version"]) for item in manifest)
    try:
        derive_package_sets([{"name": name, "version": version} for name, version in pairs])
    except ContractValidationError as error:
        if str(error) == "PREDECESSOR_PIN_DRIFT":
            return "PREDECESSOR_PIN_DRIFT", manifest
        return "RESOLUTION_REPORT_INVALID", manifest
    package_map = dict(pairs)
    if "pandas-market-calendars" in package_map and package_map["pandas-market-calendars"] != "5.4.0":
        return "RESOLUTION_REPORT_INVALID", manifest
    if "pandas-market-calendars" not in package_map or "exchange-calendars" not in package_map:
        return "REQUIRED_DISTRIBUTION_MISSING", manifest
    if source_present:
        return "SOURCE_DISTRIBUTION_REQUIRED", manifest
    return None, manifest


def run_phase_c(
    config: PhaseAConfig,
    *,
    expected_successor_lock_candidate_sha256: str | None = None,
) -> dict[str, Any]:
    """Inspect one preserved attempt offline and publish only safe artifacts."""

    root = config.durable_root
    state_path = root / STATE_NAME
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise RunnerValidationError("ATTEMPT_STATE_UNREADABLE") from error
    if not isinstance(state, dict):
        raise RunnerValidationError("ATTEMPT_STATE_SCHEMA_INVALID")
    _validate_attempt_state(config, state)
    started = state.get("process_started")
    exit_code = state.get("process_exit_code")
    invocations = state.get("package_resolution_process_invocations")
    if started is False:
        evidence = _base_resolution_evidence(config=config, status="FAIL", failure_code="RESOLUTION_PROCESS_FAILURE", process_started=False, process_exit_code=None, resolution_completed=False, candidate_artifact_created=False, candidate_sha=None, package_count=None, invocations=0)
        _validate_and_write_evidence(config, evidence, root / EVIDENCE_NAME)
        return {"status": "FAIL", "failure_code": "RESOLUTION_PROCESS_FAILURE", "candidate_artifact_created": False}
    if started is not True or not isinstance(exit_code, int) or isinstance(exit_code, bool) or invocations != 1:
        raise RunnerValidationError("ATTEMPT_STATE_AMBIGUOUS")
    if exit_code != 0:
        evidence = _base_resolution_evidence(config=config, status="FAIL", failure_code="RESOLUTION_PROCESS_FAILURE", process_started=True, process_exit_code=exit_code, resolution_completed=False, candidate_artifact_created=False, candidate_sha=None, package_count=None, invocations=1)
        _validate_and_write_evidence(config, evidence, root / EVIDENCE_NAME)
        return {"status": "FAIL", "failure_code": "RESOLUTION_PROCESS_FAILURE", "candidate_artifact_created": False}

    failure_code, manifest = _inspect_phase_c_wheelhouse(root / WHEELHOUSE_NAME)
    if failure_code is not None or manifest is None:
        evidence = _base_resolution_evidence(config=config, status="FAIL", failure_code=failure_code or "RESOLUTION_REPORT_INVALID", process_started=True, process_exit_code=0, resolution_completed=False, candidate_artifact_created=False, candidate_sha=None, package_count=None, invocations=1)
        _validate_and_write_evidence(config, evidence, root / EVIDENCE_NAME)
        return {"status": "FAIL", "failure_code": evidence["failure_code"], "candidate_artifact_created": False}

    packages = [{"name": item["name"], "version": item["version"]} for item in manifest]
    package_map = dict((item["name"], item["version"]) for item in manifest)
    candidate = {
        "schema_version": "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE_V2",
        "artifact_status": "WINDOWS_RESOLUTION_CANDIDATE_NOT_INSTALL_AUTHORITY",
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": config.expected_extension_design_sha,
        "reviewed_resolution_implementation_git_sha": config.expected_reviewed_runner_sha,
        "direct_spec_git_blob_sha1": config.expected_direct_spec_git_blob_sha1,
        "direct_spec_sha256": config.expected_direct_spec_sha256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "predecessor_package_count": 15,
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "resolution_policy_id": "PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1",
        "resolved_packages": packages,
        "resolved_package_count": len(packages),
        "resolved_wheels": manifest,
        "predecessor_pin_drift_count": 0,
        "pandas_market_calendars_version": "5.4.0",
        "exchange_calendars_version": package_map["exchange-calendars"],
    }
    try:
        validate_successor_lock_candidate(
            candidate,
            expected_extension_design_sha=config.expected_extension_design_sha,
            expected_reviewed_resolution_implementation_sha=config.expected_reviewed_runner_sha,
        )
    except ContractValidationError:
        evidence = _base_resolution_evidence(config=config, status="FAIL", failure_code="RESOLUTION_REPORT_INVALID", process_started=True, process_exit_code=0, resolution_completed=False, candidate_artifact_created=False, candidate_sha=None, package_count=None, invocations=1)
        _validate_and_write_evidence(config, evidence, root / EVIDENCE_NAME)
        return {"status": "FAIL", "failure_code": "RESOLUTION_REPORT_INVALID", "candidate_artifact_created": False}
    candidate_bytes = canonical_json_bytes(candidate)
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    if expected_successor_lock_candidate_sha256 is not None and candidate_sha != expected_successor_lock_candidate_sha256:
        raise RunnerValidationError("CANDIDATE_SHA_EXPECTATION_MISMATCH")
    _atomic_create_no_overwrite(root / CANDIDATE_NAME, candidate_bytes)
    evidence = _base_resolution_evidence(config=config, status="PASS", failure_code="NONE", process_started=True, process_exit_code=0, resolution_completed=True, candidate_artifact_created=True, candidate_sha=candidate_sha, package_count=len(packages), invocations=1)
    _validate_and_write_evidence(config, evidence, root / EVIDENCE_NAME)
    return {"status": "PASS", "failure_code": "NONE", "candidate_artifact_created": True, "candidate_sha256": candidate_sha, "resolved_package_count": len(packages)}


def _config_from_args(args: argparse.Namespace) -> PhaseAConfig:
    return PhaseAConfig(
        repo_root=Path(args.repo_root).resolve(),
        expected_current_head=args.expected_head,
        expected_extension_design_sha=args.expected_extension_design_sha,
        expected_reviewed_runner_sha=args.expected_runner_sha,
        expected_direct_spec_git_blob_sha1=args.expected_direct_spec_git_blob_sha1,
        expected_direct_spec_sha256=args.expected_direct_spec_sha256,
        durable_root=Path(args.durable_root).resolve(),
        governed_roots=tuple(Path(item).resolve() for item in args.governed_root),
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-extension-design-sha", required=True)
    parser.add_argument("--expected-runner-sha", required=True)
    parser.add_argument("--expected-direct-spec-git-blob-sha1", required=True)
    parser.add_argument("--expected-direct-spec-sha256", required=True)
    parser.add_argument("--durable-root", required=True)
    parser.add_argument("--governed-root", action="append", default=[])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="v10_environment_resolution_runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    phase_a = subparsers.add_parser("phase-a")
    _add_common_arguments(phase_a)
    phase_b = subparsers.add_parser("phase-b")
    _add_common_arguments(phase_b)
    phase_b.add_argument("--execute-resolution", action="store_true")
    phase_b.add_argument("--fresh-human-authority-confirmed", action="store_true")
    phase_c = subparsers.add_parser("phase-c")
    _add_common_arguments(phase_c)
    phase_c.add_argument("--expected-successor-lock-candidate-sha256")
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    try:
        if args.command == "phase-a":
            result = run_phase_a(config)
        elif args.command == "phase-b":
            result = run_phase_b(
                config,
                execute_resolution=args.execute_resolution,
                fresh_human_authority_confirmed=args.fresh_human_authority_confirmed,
            )
        else:
            result = run_phase_c(config, expected_successor_lock_candidate_sha256=args.expected_successor_lock_candidate_sha256)
    except RunnerValidationError as error:
        result = {"status": "FAIL", "failure_code": str(error)}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
