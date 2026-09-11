"""No-network V10 pre-mutation exact-wheel integrity preflight.

The real entry point is intentionally read-only.  Synthetic tests inject all
observations, candidate artifacts, and temporary wheelhouses; importing this
module performs no filesystem, subprocess, network, or environment action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.v10_environment_extension_contract import (
        FROZEN_V10_DESIGN_SHA,
        GENERIC_SUCCESSOR_LOCK_GIT_BLOB_SHA1,
        GENERIC_SUCCESSOR_LOCK_PACKAGE_COUNT,
        GENERIC_SUCCESSOR_LOCK_SHA256,
        PREDECESSOR_FREEZE_RECORD_GIT_BLOB_SHA1,
        PREDECESSOR_LOCK_BLOB_SHA1,
        PREDECESSOR_LOCK_CANDIDATE_GIT_BLOB_SHA1,
        PREDECESSOR_LOCK_SHA256,
        PREDECESSOR_PACKAGE_SET,
        REVIEWED_RESOLUTION_EVIDENCE_GIT_BLOB_SHA1,
        REVIEWED_RESOLUTION_EVIDENCE_GIT_SHA,
        REVIEWED_SUCCESSOR_CANDIDATE_GIT_BLOB_SHA1,
        REVIEWED_SUCCESSOR_CANDIDATE_GIT_SHA,
        ContractValidationError,
        build_exact_delta_install_argv,
        inspect_wheel_file,
        normalize_distribution_name,
        validate_generic_migration_authority,
        validate_mutation_preflight_receipt,
        validate_resolution_evidence,
        validate_successor_lock_candidate,
        verify_reviewed_wheelhouse,
    )
    from scripts.v10_environment_resolution_runner import (
        _is_reparse_or_symlink,
        _path_has_unsafe_existing_component,
        validate_durable_root,
    )
except ModuleNotFoundError:  # direct ``python scripts/<runner>.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.v10_environment_extension_contract import (
        FROZEN_V10_DESIGN_SHA,
        GENERIC_SUCCESSOR_LOCK_GIT_BLOB_SHA1,
        GENERIC_SUCCESSOR_LOCK_PACKAGE_COUNT,
        GENERIC_SUCCESSOR_LOCK_SHA256,
        PREDECESSOR_FREEZE_RECORD_GIT_BLOB_SHA1,
        PREDECESSOR_LOCK_BLOB_SHA1,
        PREDECESSOR_LOCK_CANDIDATE_GIT_BLOB_SHA1,
        PREDECESSOR_LOCK_SHA256,
        PREDECESSOR_PACKAGE_SET,
        REVIEWED_RESOLUTION_EVIDENCE_GIT_BLOB_SHA1,
        REVIEWED_RESOLUTION_EVIDENCE_GIT_SHA,
        REVIEWED_SUCCESSOR_CANDIDATE_GIT_BLOB_SHA1,
        REVIEWED_SUCCESSOR_CANDIDATE_GIT_SHA,
        ContractValidationError,
        build_exact_delta_install_argv,
        inspect_wheel_file,
        normalize_distribution_name,
        validate_generic_migration_authority,
        validate_mutation_preflight_receipt,
        validate_resolution_evidence,
        validate_successor_lock_candidate,
        verify_reviewed_wheelhouse,
    )
    from v10_environment_resolution_runner import (
        _is_reparse_or_symlink,
        _path_has_unsafe_existing_component,
        validate_durable_root,
    )


REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
CANONICAL_INTERPRETER_RELATIVE = Path(".venv-real-execution") / "Scripts" / "python.exe"
CANONICAL_ENVIRONMENT_RELATIVE = Path(".venv-real-execution")
RUNNER_RELATIVE = Path("scripts/v10_environment_mutation_preflight_runner.py")
FROZEN_DESIGN_RELATIVE = Path("V10_CALENDAR_AUTHORITY_SUCCESSOR_DESIGN_DRAFT.md")
EXTENSION_DESIGN_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_EXTENSION_DESIGN_DRAFT.md")
CANDIDATE_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json")
EVIDENCE_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE.json")
MIGRATION_AUTHORITY_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_GENERIC_MIGRATION_AUTHORITY.json")
LOCK_RELATIVE = Path("requirements-real-execution.lock.txt")
WHEELHOUSE_NAME = "wheelhouse"
RECEIPT_NAME = "V10_CANONICAL_ENVIRONMENT_MUTATION_PREFLIGHT_RECEIPT.json"
RECEIPT_SCHEMA = "V10_CANONICAL_ENVIRONMENT_MUTATION_PREFLIGHT_RECEIPT_V1"
RECEIPT_STATUS = "V10_CANONICAL_ENVIRONMENT_MUTATION_PREFLIGHT_RECEIPT"
EXPECTED_GENERIC_AUTHORITY_TRANSITION_REVIEW_SHA = "d0e0ee33bd18580e2932f405c288c3e659aeacdd"
EXPECTED_REVIEWED_RESOLUTION_IMPLEMENTATION_SHA = "1b9de998a4ef1757aca2f73b8993fb22b4d697c8"
EXPECTED_DIRECT_SPEC_BLOB_SHA1 = "8dbd6f599c61032d5fb6235bd06db3307420ab24"
EXPECTED_DIRECT_SPEC_SHA256 = "80c553ad35db95c41b2c8809b946b6f4095411ce761436cd082af5a74a1a3f08"
EXPECTED_DELTA_PACKAGES = (
    ("exchange-calendars", "4.13.2"),
    ("korean-lunar-calendar", "0.4.0"),
    ("pandas-market-calendars", "5.4.0"),
    ("pyluach", "2.3.0"),
    ("toolz", "1.1.0"),
)
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PreflightValidationError(ContractValidationError):
    """A fail-closed preflight or durable-publication error."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )


@dataclass(frozen=True)
class PreflightConfig:
    repo_root: Path
    expected_current_head: str
    expected_extension_design_sha: str
    expected_reviewed_runner_commit_sha: str
    expected_reviewed_runner_blob_sha1: str
    wheelhouse: Path
    output_root: Path | None = None
    protected_environment: Path | None = None
    governed_roots: tuple[Path, ...] = ()
    expected_generic_authority_transition_review_sha: str = EXPECTED_GENERIC_AUTHORITY_TRANSITION_REVIEW_SHA
    expected_candidate_commit_sha: str = REVIEWED_SUCCESSOR_CANDIDATE_GIT_SHA
    expected_candidate_blob_sha1: str = REVIEWED_SUCCESSOR_CANDIDATE_GIT_BLOB_SHA1
    expected_candidate_sha256: str = "aeff939030c80041d252fadf792bb9e7469b122af5a4320fe3a6f621afa98848"
    expected_evidence_commit_sha: str = REVIEWED_RESOLUTION_EVIDENCE_GIT_SHA
    expected_evidence_blob_sha1: str = REVIEWED_RESOLUTION_EVIDENCE_GIT_BLOB_SHA1
    expected_migration_authority_blob_sha1: str = "3e2d1061e0df7aa7c789e8ff6fa577d4fabc7320"
    expected_generic_lock_blob_sha1: str = GENERIC_SUCCESSOR_LOCK_GIT_BLOB_SHA1
    expected_generic_lock_sha256: str = GENERIC_SUCCESSOR_LOCK_SHA256
    expected_generic_lock_package_count: int = GENERIC_SUCCESSOR_LOCK_PACKAGE_COUNT

    @property
    def canonical_interpreter(self) -> Path:
        return self.repo_root / CANONICAL_INTERPRETER_RELATIVE

    @property
    def canonical_environment(self) -> Path:
        return self.repo_root / CANONICAL_ENVIRONMENT_RELATIVE

    @property
    def lock_file(self) -> Path:
        return self.repo_root / LOCK_RELATIVE


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _safe_sha(value: Any, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PreflightValidationError(f"{label}_INVALID")


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


def _normalize_package_observation(value: Any) -> tuple[tuple[str, str], ...]:
    if isinstance(value, bytes):
        try:
            lines = value.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise PreflightValidationError("PREDECESSOR_LIVE_PACKAGE_SET_UNPARSEABLE") from error
        value = []
        for line in lines:
            if not line or line.startswith("-") or line.count("==") != 1:
                raise PreflightValidationError("PREDECESSOR_LIVE_PACKAGE_SET_UNPARSEABLE")
            name, version = line.split("==", 1)
            value.append({"name": name, "version": version})
    if not isinstance(value, list):
        raise PreflightValidationError("PREDECESSOR_LIVE_PACKAGE_SET_UNPARSEABLE")
    result: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "version"}:
            raise PreflightValidationError("PREDECESSOR_LIVE_PACKAGE_SET_UNPARSEABLE")
        name = item["name"]
        version = item["version"]
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            raise PreflightValidationError("PREDECESSOR_LIVE_PACKAGE_SET_UNPARSEABLE")
        try:
            normalized = normalize_distribution_name(name)
        except ContractValidationError as error:
            raise PreflightValidationError("PREDECESSOR_LIVE_PACKAGE_SET_UNPARSEABLE") from error
        if normalized in names:
            raise PreflightValidationError("PREDECESSOR_LIVE_PACKAGE_NAME_COLLISION")
        names.add(normalized)
        result.append((normalized, version))
    return tuple(sorted(result))


def _parse_lock_packages(raw: bytes) -> tuple[tuple[str, str], ...]:
    if b"\r" in raw:
        raise PreflightValidationError("GENERIC_SUCCESSOR_LOCK_MISMATCH")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise PreflightValidationError("GENERIC_SUCCESSOR_LOCK_MISMATCH") from error
    if not lines:
        raise PreflightValidationError("GENERIC_SUCCESSOR_LOCK_MISMATCH")
    result: list[tuple[str, str]] = []
    names: set[str] = set()
    for line in lines:
        if not line or line.count("==") != 1:
            raise PreflightValidationError("GENERIC_SUCCESSOR_LOCK_MISMATCH")
        name, version = line.split("==", 1)
        if not version:
            raise PreflightValidationError("GENERIC_SUCCESSOR_LOCK_MISMATCH")
        try:
            normalized = normalize_distribution_name(name)
        except ContractValidationError as error:
            raise PreflightValidationError("GENERIC_SUCCESSOR_LOCK_MISMATCH") from error
        if normalized in names:
            raise PreflightValidationError("GENERIC_SUCCESSOR_LOCK_MISMATCH")
        names.add(normalized)
        result.append((normalized, version))
    if tuple(result) != tuple(sorted(result)):
        raise PreflightValidationError("GENERIC_SUCCESSOR_LOCK_MISMATCH")
    return tuple(result)


def _json_object(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, (bytes, bytearray)):
        raise PreflightValidationError(f"{label}_BYTES_MISSING")
    try:
        value = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PreflightValidationError(f"{label}_JSON_INVALID") from error
    if not isinstance(value, dict):
        raise PreflightValidationError(f"{label}_JSON_INVALID")
    return value


def _receipt(config: PreflightConfig, *, status: str, failure_code: str, integrity: bool | None, delta_count: int | None) -> dict[str, Any]:
    value = {
        "schema_version": RECEIPT_SCHEMA,
        "artifact_status": RECEIPT_STATUS,
        "status": status,
        "failure_code": failure_code,
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": config.expected_extension_design_sha,
        "reviewed_successor_lock_candidate_sha256": config.expected_candidate_sha256,
        "migration_authority_git_blob_sha1": config.expected_migration_authority_blob_sha1,
        "generic_lock_git_blob_sha1": config.expected_generic_lock_blob_sha1,
        "wheelhouse_integrity_verified": integrity,
        "delta_wheel_count": delta_count,
        "mutation_authority_consumed": False,
        "mutation_started": False,
    }
    validate_mutation_preflight_receipt(
        value,
        expected_extension_design_sha=config.expected_extension_design_sha,
        expected_successor_lock_candidate_sha256=config.expected_candidate_sha256,
        expected_migration_authority_git_blob_sha1=config.expected_migration_authority_blob_sha1,
        expected_generic_lock_git_blob_sha1=config.expected_generic_lock_blob_sha1,
    )
    return value


def _failure(config: PreflightConfig, failure_code: str, *, integrity: bool | None = None, delta_count: int | None = None) -> dict[str, Any]:
    receipt = _receipt(config, status="FAIL", failure_code=failure_code, integrity=integrity, delta_count=delta_count)
    return {"receipt": receipt, "failure_code": failure_code, "install_argv": None}


def _validate_provenance(config: PreflightConfig, obs: Mapping[str, Any]) -> dict[str, Any] | None:
    for value, pattern, label in (
        (config.expected_current_head, SHA1_RE, "expected current head"),
        (config.expected_extension_design_sha, SHA1_RE, "expected extension design SHA"),
        (config.expected_reviewed_runner_commit_sha, SHA1_RE, "expected runner commit"),
        (config.expected_reviewed_runner_blob_sha1, SHA1_RE, "expected runner blob"),
        (config.expected_generic_authority_transition_review_sha, SHA1_RE, "expected transition review"),
        (config.expected_candidate_commit_sha, SHA1_RE, "expected candidate commit"),
        (config.expected_candidate_blob_sha1, SHA1_RE, "expected candidate blob"),
        (config.expected_candidate_sha256, SHA256_RE, "expected candidate SHA"),
        (config.expected_evidence_commit_sha, SHA1_RE, "expected evidence commit"),
        (config.expected_evidence_blob_sha1, SHA1_RE, "expected evidence blob"),
        (config.expected_migration_authority_blob_sha1, SHA1_RE, "expected migration authority blob"),
        (config.expected_generic_lock_blob_sha1, SHA1_RE, "expected generic lock blob"),
        (config.expected_generic_lock_sha256, SHA256_RE, "expected generic lock SHA"),
    ):
        try:
            _safe_sha(value, pattern, label)
        except PreflightValidationError:
            return None
    if not _repo_identity_matches(obs.get("repository_identity")):
        return None
    if obs.get("branch") != AUTHORITATIVE_BRANCH or obs.get("head") != config.expected_current_head:
        return None
    if obs.get("clean") is not True:
        return None
    if obs.get("frozen_design_commit_exists") is not True or obs.get("extension_design_commit_exists") is not True:
        return None
    if obs.get("generic_authority_transition_reviewed_sha") != config.expected_generic_authority_transition_review_sha:
        return None
    if obs.get("generic_authority_transition_commit_exists") is not True:
        return None
    if obs.get("reviewed_candidate_commit_exists") is not True or obs.get("reviewed_evidence_commit_exists") is not True:
        return None
    if obs.get("reviewed_runner_commit_exists") is not True:
        return None
    if obs.get("reviewed_runner_blob_sha1") != config.expected_reviewed_runner_blob_sha1:
        return None
    if obs.get("current_runner_blob_sha1") != config.expected_reviewed_runner_blob_sha1:
        return None
    if obs.get("candidate_git_blob_sha1") != config.expected_candidate_blob_sha1:
        return None
    if obs.get("evidence_git_blob_sha1") != config.expected_evidence_blob_sha1:
        return None
    if obs.get("migration_authority_git_blob_sha1") != config.expected_migration_authority_blob_sha1:
        return None
    if obs.get("generic_lock_git_blob_sha1") != config.expected_generic_lock_blob_sha1:
        return None
    candidate_bytes = obs.get("candidate_bytes")
    evidence_bytes = obs.get("evidence_bytes")
    authority_bytes = obs.get("migration_authority_bytes")
    if not isinstance(candidate_bytes, bytes) or not isinstance(evidence_bytes, bytes) or not isinstance(authority_bytes, bytes):
        return None
    if hashlib.sha256(candidate_bytes).hexdigest() != config.expected_candidate_sha256:
        return None
    if obs.get("candidate_bytes_git_blob_sha1") != _git_blob_sha1(candidate_bytes):
        return None
    if obs.get("evidence_bytes_git_blob_sha1") != _git_blob_sha1(evidence_bytes):
        return None
    if obs.get("migration_authority_bytes_git_blob_sha1") != _git_blob_sha1(authority_bytes):
        return None
    try:
        candidate = _json_object(candidate_bytes, "candidate")
        evidence = _json_object(evidence_bytes, "evidence")
        authority = _json_object(authority_bytes, "migration authority")
        package_sets = validate_successor_lock_candidate(
            candidate,
            expected_extension_design_sha=config.expected_extension_design_sha,
            expected_reviewed_resolution_implementation_sha=EXPECTED_REVIEWED_RESOLUTION_IMPLEMENTATION_SHA,
        )
        validate_resolution_evidence(
            evidence,
            expected_extension_design_sha=config.expected_extension_design_sha,
            expected_reviewed_resolution_implementation_sha=EXPECTED_REVIEWED_RESOLUTION_IMPLEMENTATION_SHA,
            expected_direct_spec_git_blob_sha1=EXPECTED_DIRECT_SPEC_BLOB_SHA1,
            expected_direct_spec_sha256=EXPECTED_DIRECT_SPEC_SHA256,
            expected_successor_lock_candidate_sha256=config.expected_candidate_sha256,
        )
        validate_generic_migration_authority(
            authority,
            expected_reviewed_successor_lock_candidate_git_sha=REVIEWED_SUCCESSOR_CANDIDATE_GIT_SHA,
            expected_reviewed_successor_lock_candidate_git_blob_sha1=REVIEWED_SUCCESSOR_CANDIDATE_GIT_BLOB_SHA1,
            expected_reviewed_resolution_evidence_git_sha=REVIEWED_RESOLUTION_EVIDENCE_GIT_SHA,
            expected_reviewed_resolution_evidence_git_blob_sha1=REVIEWED_RESOLUTION_EVIDENCE_GIT_BLOB_SHA1,
            expected_new_generic_lock_git_blob_sha1=config.expected_generic_lock_blob_sha1,
            expected_new_generic_lock_sha256=config.expected_generic_lock_sha256,
            expected_new_generic_lock_package_count=config.expected_generic_lock_package_count,
        )
    except (ContractValidationError, TypeError, ValueError, KeyError):
        return None
    return {"candidate": candidate, "evidence": evidence, "authority": authority, "package_sets": package_sets}


def _validate_predecessor(config: PreflightConfig, obs: Mapping[str, Any]) -> bool:
    try:
        interpreter = Path(str(obs["interpreter_executable"])).resolve()
        expected = config.canonical_interpreter.resolve()
        live_packages = _normalize_package_observation(obs["live_packages"])
        predecessor_lock_count = obs["predecessor_lock_package_count"]
        if not isinstance(predecessor_lock_count, int) or isinstance(predecessor_lock_count, bool):
            return False
        return (
            interpreter == expected
            and obs["python_implementation"] == "CPython"
            and obs["python_version"] == "3.12.10"
            and obs["platform_system"] == "Windows"
            and obs["platform_machine"] == "AMD64"
            and obs["sysconfig_platform"] == "win-amd64"
            and obs["pip_version"] == "25.0.1"
            and obs["predecessor_lock_git_blob_sha1"] == PREDECESSOR_LOCK_BLOB_SHA1
            and obs["predecessor_lock_sha256"] == PREDECESSOR_LOCK_SHA256
            and predecessor_lock_count == 15
            and live_packages == PREDECESSOR_PACKAGE_SET
        )
    except (KeyError, TypeError, ValueError, OSError, PreflightValidationError):
        return False


def _validate_successor_lock(config: PreflightConfig, obs: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    try:
        committed = obs["generic_lock_committed_bytes"]
        worktree = obs["generic_lock_worktree_bytes"]
        if not isinstance(committed, bytes) or not isinstance(worktree, bytes):
            return False
        if worktree != committed or b"\r" in worktree:
            return False
        if obs.get("generic_lock_git_blob_sha1") != config.expected_generic_lock_blob_sha1:
            return False
        if _git_blob_sha1(committed) != config.expected_generic_lock_blob_sha1:
            return False
        if hashlib.sha256(committed).hexdigest() != config.expected_generic_lock_sha256:
            return False
        packages = _parse_lock_packages(committed)
        expected = tuple((item["name"], item["version"]) for item in candidate["resolved_packages"])
        return packages == expected and len(packages) == config.expected_generic_lock_package_count
    except (KeyError, TypeError, ValueError, PreflightValidationError):
        return False


def _wheelhouse_path_is_safe(path: Path) -> bool:
    return path.is_absolute() and not _path_has_unsafe_existing_component(path) and not _is_reparse_or_symlink(path)


def _derive_wheelhouse(config: PreflightConfig, candidate: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    if not _wheelhouse_path_is_safe(config.wheelhouse):
        raise PreflightValidationError("WHEELHOUSE_FILESYSTEM_SAFETY_FAILURE")
    try:
        result = verify_reviewed_wheelhouse(
            config.wheelhouse,
            candidate["resolved_wheels"],
            candidate["resolved_packages"],
        )
    except (ContractValidationError, OSError, TypeError, ValueError) as error:
        result = {
            "ok": False,
            "failure_code": "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE",
            "wheelhouse_integrity_verified": False,
            "delta_wheel_count": len(EXPECTED_DELTA_PACKAGES),
            "delta_packages": EXPECTED_DELTA_PACKAGES,
            "delta_wheel_paths": (),
            "reason": str(error),
        }
    if not result.get("ok"):
        return result, []
    if tuple(result["delta_packages"]) != EXPECTED_DELTA_PACKAGES or result["delta_wheel_count"] != 5:
        return {
            "ok": False,
            "failure_code": "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE",
            "wheelhouse_integrity_verified": False,
            "delta_wheel_count": result.get("delta_wheel_count", 5),
            "delta_packages": result.get("delta_packages", ()),
            "delta_wheel_paths": (),
        }, []
    paths = list(result["delta_wheel_paths"])
    argv = build_exact_delta_install_argv(config.canonical_interpreter, paths)
    if argv[:6] != [str(config.canonical_interpreter), "-m", "pip", "install", "--no-deps", "--no-index"]:
        raise PreflightValidationError("INSTALL_ARGV_BINDING_FAILURE")
    if [Path(path).name for path in paths] != [Path(path).name for path in result["delta_wheel_paths"]]:
        raise PreflightValidationError("INSTALL_ARGV_BINDING_FAILURE")
    return result, argv


def run_preflight(
    config: PreflightConfig,
    observations: Mapping[str, Any] | None = None,
    *,
    publish: bool = False,
) -> dict[str, Any]:
    """Run the frozen receipt precedence without mutation or package install."""

    if observations is None:
        # Stage P is isolated: this collector performs repository/file
        # binding only and cannot launch the canonical interpreter.
        obs = _default_provenance_observations(config)
        provenance = _validate_provenance(config, obs)
        if provenance is None:
            result = _failure(config, "PROVENANCE_BINDING_FAILURE")
            if publish:
                publish_receipt(config, result["receipt"])
            return result
        # Stage L is reached only after Stage P has passed.
        try:
            obs.update(_default_predecessor_observations(config))
        except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError, PreflightValidationError):
            pass
        if not _validate_predecessor(config, obs):
            result = _failure(config, "PREDECESSOR_LIVE_BASELINE_MISMATCH")
            if publish:
                publish_receipt(config, result["receipt"])
            return result
        # Stage S reads the committed/worktree lock only after the live
        # predecessor baseline has passed.
        try:
            obs.update(_default_successor_lock_observations(config))
        except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError, PreflightValidationError):
            pass
    else:
        # Synthetic callers inject all observations and therefore perform no
        # real process or environment probe.
        obs = dict(observations)
        provenance = _validate_provenance(config, obs)
    if provenance is None:
        result = _failure(config, "PROVENANCE_BINDING_FAILURE")
    elif not _validate_predecessor(config, obs):
        result = _failure(config, "PREDECESSOR_LIVE_BASELINE_MISMATCH")
    elif not _validate_successor_lock(config, obs, provenance["candidate"]):
        result = _failure(config, "GENERIC_SUCCESSOR_LOCK_MISMATCH")
    else:
        wheel_result, install_argv = _derive_wheelhouse(config, provenance["candidate"])
        if not wheel_result.get("ok"):
            result = _failure(
                config,
                "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE",
                integrity=False,
                delta_count=wheel_result.get("delta_wheel_count"),
            )
            result["wheel_result"] = wheel_result
            result["install_argv"] = None
        else:
            result = {
                "receipt": _receipt(
                    config,
                    status="PASS",
                    failure_code="NONE",
                    integrity=True,
                    delta_count=wheel_result["delta_wheel_count"],
                ),
                "failure_code": "NONE",
                "wheel_result": wheel_result,
                "install_argv": install_argv,
            }
    if publish:
        publish_receipt(config, result["receipt"])
    return result


def _run_git(repo_root: Path, args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    ).stdout


def _git_exists(repo_root: Path, revision: str, kind: str = "commit") -> bool:
    try:
        _run_git(repo_root, ["cat-file", "-e", f"{revision}^{{{kind}}}"])
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _git_show(repo_root: Path, revision: str, relative: Path) -> bytes:
    return _run_git(repo_root, ["show", f"{revision}:{relative.as_posix()}"])


def _parse_pip_freeze(raw: bytes) -> list[dict[str, str]]:
    value = _normalize_package_observation(raw)
    return [{"name": name, "version": version} for name, version in value]


def _probe_canonical_environment(config: PreflightConfig) -> dict[str, Any]:
    interpreter = config.canonical_interpreter
    probe = subprocess.run(
        [str(interpreter), "-c", "import platform,sys,sysconfig; print('|'.join((sys.executable,platform.python_implementation(),platform.python_version(),platform.system(),platform.machine(),sysconfig.get_platform())))"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    ).stdout.decode("utf-8").strip().split("|")
    pip_output = subprocess.run(
        [str(interpreter), "-m", "pip", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    ).stdout.decode("utf-8").strip()
    freeze = subprocess.run(
        [str(interpreter), "-m", "pip", "freeze", "--all"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    ).stdout
    match = re.search(r"\bpip\s+([0-9][^\s]*)", pip_output)
    if len(probe) != 6 or match is None:
        raise PreflightValidationError("CANONICAL_INTERPRETER_PROBE_INVALID")
    return {
        "interpreter_executable": probe[0],
        "python_implementation": probe[1],
        "python_version": probe[2],
        "platform_system": probe[3],
        "platform_machine": probe[4],
        "sysconfig_platform": probe[5],
        "pip_version": match.group(1),
        "live_packages": _parse_pip_freeze(freeze),
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "predecessor_lock_package_count": 15,
    }


def _default_provenance_observations(config: PreflightConfig) -> dict[str, Any]:
    """Gather only Stage-P local/no-network observations."""

    obs: dict[str, Any] = {}
    try:
        obs.update(
            repository_identity=_run_git(config.repo_root, ["config", "--get", "remote.origin.url"]).decode().strip(),
            branch=_run_git(config.repo_root, ["branch", "--show-current"]).decode().strip(),
            head=_run_git(config.repo_root, ["rev-parse", "HEAD"]).decode().strip(),
            clean=_run_git(config.repo_root, ["status", "--porcelain", "--untracked-files=all"]) == b"",
            frozen_design_commit_exists=_git_exists(config.repo_root, FROZEN_V10_DESIGN_SHA),
            extension_design_commit_exists=_git_exists(config.repo_root, config.expected_extension_design_sha),
            generic_authority_transition_reviewed_sha=config.expected_generic_authority_transition_review_sha,
            generic_authority_transition_commit_exists=_git_exists(config.repo_root, config.expected_generic_authority_transition_review_sha),
            reviewed_candidate_commit_exists=_git_exists(config.repo_root, config.expected_candidate_commit_sha),
            reviewed_evidence_commit_exists=_git_exists(config.repo_root, config.expected_evidence_commit_sha),
            reviewed_runner_commit_exists=_git_exists(config.repo_root, config.expected_reviewed_runner_commit_sha),
            reviewed_runner_blob_sha1=_run_git(config.repo_root, ["rev-parse", f"{config.expected_reviewed_runner_commit_sha}:{RUNNER_RELATIVE.as_posix()}"]).decode().strip(),
            current_runner_blob_sha1=_run_git(config.repo_root, ["hash-object", "--", str(config.repo_root / RUNNER_RELATIVE)]).decode().strip(),
            candidate_git_blob_sha1=_run_git(config.repo_root, ["rev-parse", f"{config.expected_candidate_commit_sha}:{CANDIDATE_RELATIVE.as_posix()}"]).decode().strip(),
            evidence_git_blob_sha1=_run_git(config.repo_root, ["rev-parse", f"{config.expected_evidence_commit_sha}:{EVIDENCE_RELATIVE.as_posix()}"]).decode().strip(),
            migration_authority_git_blob_sha1=_run_git(config.repo_root, ["rev-parse", f"HEAD:{MIGRATION_AUTHORITY_RELATIVE.as_posix()}"]).decode().strip(),
            candidate_bytes=_git_show(config.repo_root, config.expected_candidate_commit_sha, CANDIDATE_RELATIVE),
            evidence_bytes=_git_show(config.repo_root, config.expected_evidence_commit_sha, EVIDENCE_RELATIVE),
            migration_authority_bytes=_git_show(config.repo_root, "HEAD", MIGRATION_AUTHORITY_RELATIVE),
            generic_lock_git_blob_sha1=_run_git(config.repo_root, ["rev-parse", f"HEAD:{LOCK_RELATIVE.as_posix()}"]).decode().strip(),
        )
        obs.update(
            candidate_bytes_git_blob_sha1=_git_blob_sha1(obs["candidate_bytes"]),
            evidence_bytes_git_blob_sha1=_git_blob_sha1(obs["evidence_bytes"]),
            migration_authority_bytes_git_blob_sha1=_git_blob_sha1(obs["migration_authority_bytes"]),
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError, PreflightValidationError):
        return obs
    return obs


def _default_predecessor_observations(config: PreflightConfig) -> dict[str, Any]:
    """Gather Stage-L observations only after provenance has passed."""

    return _probe_canonical_environment(config)


def _default_successor_lock_observations(config: PreflightConfig) -> dict[str, Any]:
    """Gather Stage-S lock bytes only after the predecessor gate has passed."""

    return {
        "generic_lock_committed_bytes": _git_show(config.repo_root, "HEAD", LOCK_RELATIVE),
        "generic_lock_worktree_bytes": (config.repo_root / LOCK_RELATIVE).read_bytes(),
    }


def _default_observations(config: PreflightConfig) -> dict[str, Any]:
    """Backward-compatible alias for the Stage-P-only collector."""

    return _default_provenance_observations(config)


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
        raise PreflightValidationError("DURABLE_FILE_ALREADY_EXISTS") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def publish_receipt(config: PreflightConfig, receipt: Mapping[str, Any]) -> Path:
    """Publish one receipt only into a new safe durable root, never overwrite."""

    if config.output_root is None:
        raise PreflightValidationError("OUTPUT_ROOT_REQUIRED")
    output_root = Path(config.output_root)
    protected = config.protected_environment or config.canonical_environment
    safety = validate_durable_root(
        output_root,
        repo_root=config.repo_root,
        protected_environment=protected,
        governed_roots=(*config.governed_roots, config.wheelhouse),
    )
    if safety != "DURABLE_ROOT_OK":
        raise PreflightValidationError(f"DURABLE_OUTPUT_ROOT_SAFETY_FAILURE:{safety}")
    if _is_reparse_or_symlink(output_root):
        raise PreflightValidationError("DURABLE_OUTPUT_ROOT_SAFETY_FAILURE")
    try:
        output_root.mkdir()
    except (FileExistsError, OSError) as error:
        raise PreflightValidationError("DURABLE_OUTPUT_ROOT_CREATION_FAILURE") from error
    path = output_root / RECEIPT_NAME
    _atomic_create_no_overwrite(path, canonical_json_bytes(dict(receipt)))
    return path


def _config_from_args(args: argparse.Namespace) -> PreflightConfig:
    return PreflightConfig(
        repo_root=Path(args.repo_root).resolve(),
        expected_current_head=args.expected_current_head,
        expected_extension_design_sha=args.expected_extension_design_sha,
        expected_reviewed_runner_commit_sha=args.expected_reviewed_runner_commit_sha,
        expected_reviewed_runner_blob_sha1=args.expected_reviewed_runner_blob_sha1,
        # Keep user-supplied lexical paths intact until the safety gate has
        # inspected every existing ancestor for symlinks/reparse points.
        wheelhouse=Path(args.wheelhouse),
        output_root=Path(args.output_root),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-current-head", required=True)
    parser.add_argument("--expected-extension-design-sha", required=True)
    parser.add_argument("--expected-reviewed-runner-commit-sha", required=True)
    parser.add_argument("--expected-reviewed-runner-blob-sha1", required=True)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    result = run_preflight(config, publish=True)
    sys.stdout.write(canonical_json_bytes(result["receipt"]).decode("utf-8"))
    return 0 if result["receipt"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
