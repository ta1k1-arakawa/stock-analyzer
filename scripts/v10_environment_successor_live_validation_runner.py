"""No-network V10 successor live-validation runner.

This module has a deliberately injectable observation boundary for synthetic
tests.  The production CLI never supplies observations: it binds the real
repository, Step-4 attempt, and output root and collects live metadata only
after all provenance gates pass.  It never installs packages, opens a network
connection, creates a calendar, or inspects dates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts import v10_environment_mutation_preflight_runner as preflight
    from scripts import v10_environment_exact_delta_mutation_runner as mutation
    from scripts.v10_environment_extension_contract import (
        ContractValidationError,
        PREDECESSOR_PACKAGE_SET,
        PREDECESSOR_LOCK_BLOB_SHA1,
        build_exact_delta_install_argv,
        normalize_distribution_name,
        validate_resolution_evidence,
        validate_successor_lock_candidate,
        verify_reviewed_wheelhouse,
    )
except ModuleNotFoundError:  # direct ``python scripts/<runner>.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import v10_environment_exact_delta_mutation_runner as mutation
    import v10_environment_mutation_preflight_runner as preflight
    from v10_environment_extension_contract import (
        ContractValidationError,
        PREDECESSOR_PACKAGE_SET,
        PREDECESSOR_LOCK_BLOB_SHA1,
        build_exact_delta_install_argv,
        normalize_distribution_name,
        validate_resolution_evidence,
        validate_successor_lock_candidate,
        verify_reviewed_wheelhouse,
    )


REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
CANONICAL_ENVIRONMENT_RELATIVE = Path(".venv-real-execution")
CANONICAL_INTERPRETER_RELATIVE = CANONICAL_ENVIRONMENT_RELATIVE / "Scripts" / "python.exe"
WHEELHOUSE_RELATIVE = Path("wheelhouse")
RUNNER_RELATIVE = Path("scripts/v10_environment_successor_live_validation_runner.py")
PREMUTATION_RUNNER_RELATIVE = Path("scripts/v10_environment_mutation_preflight_runner.py")
CANDIDATE_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json")
EVIDENCE_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE.json")
MIGRATION_AUTHORITY_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_GENERIC_MIGRATION_AUTHORITY.json")
LOCK_RELATIVE = Path("requirements-real-execution.lock.txt")
EVIDENCE_NAME = "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LIVE_VALIDATION_EVIDENCE.json"
EVIDENCE_SCHEMA = "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LIVE_VALIDATION_EVIDENCE_V1"
EVIDENCE_STATUS = "V10_SUCCESSOR_LIVE_VALIDATION_EVIDENCE"

FROZEN_V10_DESIGN_SHA = "8c923ed1734c6bdfe95a743cd9e15a5156d62c03"
EXTENSION_DESIGN_SHA = "efe2e9d8cfab696c74b94cfd1cfaa2a2a4706c58"
GENERIC_AUTHORITY_TRANSITION_SHA = "d0e0ee33bd18580e2932f405c288c3e659aeacdd"
MIGRATION_AUTHORITY_BLOB_SHA1 = "3e2d1061e0df7aa7c789e8ff6fa577d4fabc7320"
GENERIC_LOCK_BLOB_SHA1 = "99395e7a5be752fb3ea92fd31be0334f38792261"
GENERIC_LOCK_SHA256 = "eb325ac5e3417e6407400b18c8d90ca734a32e852056926e5bcd2a635e43c444"
GENERIC_LOCK_PACKAGE_COUNT = 20
CANDIDATE_SHA256 = "aeff939030c80041d252fadf792bb9e7469b122af5a4320fe3a6f621afa98848"
CANDIDATE_BLOB_SHA1 = "eb5c95d9b5cac096fadce870e34ca138cee395c2"
EVIDENCE_BLOB_SHA1 = "c0213acda4e9713dbed9911b25db7748d131ccf7"
REVIEWED_CANDIDATE_COMMIT_SHA = "1f045ee8e962827f2cf6e8218c5dd28d364d2a18"
REVIEWED_EVIDENCE_COMMIT_SHA = REVIEWED_CANDIDATE_COMMIT_SHA
PREMUTATION_RUNNER_COMMIT_SHA = "9f37cc5c0c10db11a0164ab7a3d4d2dc9d311adb"
PREMUTATION_RUNNER_BLOB_SHA1 = "21e9f94928f6e6ce315fa473bf52d65ff7537ffb"
STEP3_RECEIPT_SHA256 = "0c15278d79f88110766a581aa3c14bdc03434b88806b8a7975327bbd6bee07ae"
STEP4_MUTATION_RUNNER_COMMIT_SHA = "810ccf4bcb89ed7fdc2aea2cda287cfd3f278cb0"
STEP4_MUTATION_RUNNER_BLOB_SHA1 = "60bd6b7cd527f89ee4ae20b45490a1acac2bdd59"
STEP4_STDOUT_SHA256 = "af5b9f275a874c529d5db5648c5062298bccd79e386c9c62a3fd3485fe860553"
STEP4_STDERR_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
EXPECTED_DELTA = (
    ("exchange-calendars", "4.13.2"),
    ("korean-lunar-calendar", "0.4.0"),
    ("pandas-market-calendars", "5.4.0"),
    ("pyluach", "2.3.0"),
    ("toolz", "1.1.0"),
)
EXPECTED_SUCCESSOR_PACKAGES = tuple(sorted((*PREDECESSOR_PACKAGE_SET, *EXPECTED_DELTA)))
EXPECTED_PMC_SOURCE_BLOB_SHA1 = "0c2041b1300d1dbbd505202b00ac0ada38c712e1"
EXPECTED_HOLIDAY_SOURCE_BLOB_SHA1 = "4c34214d06862e02ac22e946757463f748074fde"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

FAILURE_CODES = (
    "NONE",
    "UNAUTHORIZED_OPERATION_OBSERVED",
    "PROVENANCE_BINDING_FAILURE",
    "LIVE_PACKAGE_SET_MISMATCH",
    "PYTHON_PLATFORM_MISMATCH",
    "PMC_VERSION_MISMATCH",
    "EXCHANGE_CALENDARS_VERSION_MISMATCH",
    "JPX_SOURCE_BLOB_MISMATCH",
    "HOLIDAY_SOURCE_BLOB_MISMATCH",
    "XLS_PROBE_FAILURE",
    "PDF_PROBE_FAILURE",
)
EVIDENCE_KEYS = frozenset(
    {
        "schema_version", "artifact_status", "status", "failure_code",
        "frozen_v10_design_git_sha", "extension_design_git_sha",
        "reviewed_generic_authority_transition_git_sha", "migration_authority_git_blob_sha1",
        "generic_lock_git_blob_sha1", "generic_lock_sha256", "generic_lock_package_count",
        "reviewed_successor_lock_candidate_sha256", "installed_delta_packages",
        "installed_delta_package_count", "installed_delta_wheel_count", "observed_packages",
        "observed_package_count", "python_version", "platform_system", "platform_machine",
        "sysconfig_platform", "pandas_market_calendars_version", "exchange_calendars_version",
        "jpx_source_blob_match", "holiday_source_blob_match", "xls_probe_status", "pdf_probe_status",
        "package_index_network_requests", "package_installations", "calendar_object_creations",
        "calendar_dates_inspected", "protected_or_private_reads", "t0_run",
    }
)


class LiveValidationError(ContractValidationError):
    """Fail-closed live-validation or durable-publication error."""


@dataclass(frozen=True)
class LiveValidationConfig:
    repo_root: Path
    expected_current_head: str
    expected_live_validation_runner_commit_sha: str
    expected_live_validation_runner_blob_sha1: str
    step4_attempt_root: Path
    output_root: Path
    wheelhouse: Path | None = None

    @property
    def canonical_environment(self) -> Path:
        return self.repo_root / CANONICAL_ENVIRONMENT_RELATIVE

    @property
    def canonical_interpreter(self) -> Path:
        return self.repo_root / CANONICAL_INTERPRETER_RELATIVE

    @property
    def effective_wheelhouse(self) -> Path:
        return self.wheelhouse or self.repo_root / WHEELHOUSE_RELATIVE


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _strict_sha(value: Any, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise LiveValidationError(f"{label}_INVALID")


def _strict_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LiveValidationError(f"{label}_INVALID")
    return value


def _strict_bool_or_none(value: Any, label: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise LiveValidationError(f"{label}_INVALID")


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _nearest_existing(path: Path) -> Path:
    node = path
    while not os.path.lexists(node) and node.parent != node:
        node = node.parent
    return node


def _unsafe_existing_component(path: Path) -> bool:
    node = _nearest_existing(path)
    while True:
        try:
            info = os.lstat(node)
        except OSError:
            return True
        if stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400):
            return True
        if node.parent == node:
            return False
        node = node.parent


def _output_root_safe(config: LiveValidationConfig) -> bool:
    root = config.output_root
    if not root.is_absolute() or os.path.lexists(root) or _unsafe_existing_component(root):
        return False
    try:
        resolved = Path(os.path.realpath(root))
        forbidden = [
            Path(os.path.realpath(config.repo_root)),
            Path(os.path.realpath(config.canonical_environment)),
            Path(os.path.realpath(config.effective_wheelhouse)),
            Path(os.path.realpath(config.step4_attempt_root)),
        ]
    except OSError:
        return False
    if any(_path_inside(resolved, item) or _path_inside(item, resolved) for item in forbidden):
        return False
    return _nearest_existing(root).exists()


def _existing_attempt_root_safe(config: LiveValidationConfig) -> bool:
    root = config.step4_attempt_root
    if not root.is_absolute() or not root.is_dir() or _unsafe_existing_component(root):
        return False
    try:
        resolved = Path(os.path.realpath(root))
        forbidden = [Path(os.path.realpath(config.repo_root)), Path(os.path.realpath(config.canonical_environment))]
    except OSError:
        return False
    return not any(_path_inside(resolved, item) or _path_inside(item, resolved) for item in forbidden)


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


def _preflight_config(config: LiveValidationConfig) -> preflight.PreflightConfig:
    return preflight.PreflightConfig(
        repo_root=config.repo_root,
        expected_current_head=config.expected_current_head,
        expected_extension_design_sha=EXTENSION_DESIGN_SHA,
        expected_reviewed_runner_commit_sha=PREMUTATION_RUNNER_COMMIT_SHA,
        expected_reviewed_runner_blob_sha1=PREMUTATION_RUNNER_BLOB_SHA1,
        wheelhouse=config.effective_wheelhouse,
        protected_environment=config.canonical_environment,
        governed_roots=(config.step4_attempt_root,),
        expected_generic_authority_transition_review_sha=GENERIC_AUTHORITY_TRANSITION_SHA,
        expected_candidate_commit_sha=REVIEWED_CANDIDATE_COMMIT_SHA,
        expected_candidate_blob_sha1=CANDIDATE_BLOB_SHA1,
        expected_candidate_sha256=CANDIDATE_SHA256,
        expected_evidence_commit_sha=REVIEWED_EVIDENCE_COMMIT_SHA,
        expected_evidence_blob_sha1=EVIDENCE_BLOB_SHA1,
        expected_migration_authority_blob_sha1=MIGRATION_AUTHORITY_BLOB_SHA1,
        expected_generic_lock_blob_sha1=GENERIC_LOCK_BLOB_SHA1,
        expected_generic_lock_sha256=GENERIC_LOCK_SHA256,
        expected_generic_lock_package_count=GENERIC_LOCK_PACKAGE_COUNT,
    )


def _default_provenance_observations(config: LiveValidationConfig) -> dict[str, Any]:
    pc = _preflight_config(config)
    obs = preflight._default_provenance_observations(pc)
    try:
        obs.update(
            live_validation_runner_commit_exists=_git_exists(config.repo_root, config.expected_live_validation_runner_commit_sha),
            reviewed_live_validation_runner_blob_sha1=_run_git(
                config.repo_root,
                ["rev-parse", f"{config.expected_live_validation_runner_commit_sha}:{RUNNER_RELATIVE.as_posix()}"],
            ).decode().strip(),
            current_live_validation_runner_blob_sha1=_run_git(
                config.repo_root, ["hash-object", "--", str(config.repo_root / RUNNER_RELATIVE)]
            ).decode().strip(),
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError):
        return obs
    return obs


def _unauthorized_observed(obs: Mapping[str, Any]) -> bool:
    if obs.get("unauthorized_operation_observed") is True:
        return True
    for key in (
        "package_index_network_requests", "package_installations", "calendar_object_creations",
        "calendar_dates_inspected", "protected_or_private_reads",
    ):
        value = obs.get(key, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    return obs.get("t0_run") is True


def _validate_live_runner_provenance(config: LiveValidationConfig, obs: Mapping[str, Any]) -> bool:
    try:
        _strict_sha(config.expected_current_head, SHA1_RE, "expected head")
        _strict_sha(config.expected_live_validation_runner_commit_sha, SHA1_RE, "live runner commit")
        _strict_sha(config.expected_live_validation_runner_blob_sha1, SHA1_RE, "live runner blob")
    except LiveValidationError:
        return False
    return (
        obs.get("live_validation_runner_commit_exists") is True
        and obs.get("reviewed_live_validation_runner_blob_sha1") == config.expected_live_validation_runner_blob_sha1
        and obs.get("current_live_validation_runner_blob_sha1") == config.expected_live_validation_runner_blob_sha1
    )


def _read_step4_state(config: LiveValidationConfig, obs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    value = obs.get("step4_state")
    if value is not None:
        return value if isinstance(value, dict) else None
    try:
        raw = (config.step4_attempt_root / "attempt_state_complete.json").read_bytes()
        value = json.loads(raw.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_attempt_file(config: LiveValidationConfig, obs: Mapping[str, Any], name: str) -> bytes | None:
    injected = obs.get(name)
    if injected is not None:
        return injected if isinstance(injected, bytes) else None
    try:
        return (config.step4_attempt_root / name).read_bytes()
    except OSError:
        return None


def _expected_manifest(candidate: Mapping[str, Any], wheel_paths: Mapping[str, str]) -> list[dict[str, Any]]:
    delta = set(EXPECTED_DELTA)
    return [
        {
            **dict(item),
            "path": wheel_paths[item["filename"]],
            "is_delta": (item["name"], item["version"]) in delta,
        }
        for item in candidate["resolved_wheels"]
    ]


def _validate_step4_evidence(config: LiveValidationConfig, obs: Mapping[str, Any], candidate: Mapping[str, Any], delta_paths: Sequence[Path]) -> bool:
    state = _read_step4_state(config, obs)
    if state is None:
        return False
    try:
        mutation.validate_attempt_state(state)
        required = {
            "expected_current_head": config.expected_current_head,
            "mutation_runner_blob_sha1": STEP4_MUTATION_RUNNER_BLOB_SHA1,
            "step3_receipt_sha256": STEP3_RECEIPT_SHA256,
            "reviewed_successor_lock_candidate_sha256": CANDIDATE_SHA256,
            "process_start_attempted": True,
            "process_started": True,
            "mutation_authority_consumed": True,
            "mutation_started": True,
            "retry_authorized": False,
            "process_exit_code": 0,
            "failure_code": "NONE",
        }
        if any(state.get(key) != value for key, value in required.items()):
            return False
        stdout = _read_attempt_file(config, obs, "stdout.bin")
        stderr = _read_attempt_file(config, obs, "stderr.bin")
        if stdout is None or stderr is None:
            return False
        if hashlib.sha256(stdout).hexdigest() != STEP4_STDOUT_SHA256 or hashlib.sha256(stderr).hexdigest() != STEP4_STDERR_SHA256:
            return False
        expected_stdout_path = str(config.step4_attempt_root / "stdout.bin")
        expected_stderr_path = str(config.step4_attempt_root / "stderr.bin")
        if state.get("stdout_capture_path") != expected_stdout_path or state.get("stderr_capture_path") != expected_stderr_path:
            return False
        if state.get("stdout_sha256") != STEP4_STDOUT_SHA256 or state.get("stderr_sha256") != STEP4_STDERR_SHA256:
            return False
        actual_paths = tuple(str(path) for path in delta_paths)
        if tuple(state.get("delta_wheel_paths", ())) != actual_paths:
            return False
        wheel_paths = {item["filename"]: str(config.effective_wheelhouse / item["filename"]) for item in candidate["resolved_wheels"]}
        if state.get("wheel_manifest") != _expected_manifest(candidate, wheel_paths):
            return False
    except (ContractValidationError, KeyError, TypeError, ValueError, OSError):
        return False
    return True


def _validate_provenance(config: LiveValidationConfig, obs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not _validate_live_runner_provenance(config, obs):
        return None
    pc = _preflight_config(config)
    base = preflight._validate_provenance(pc, obs)
    if base is None:
        return None
    lock_obs = obs
    if "generic_lock_committed_bytes" not in lock_obs:
        try:
            lock_obs = {**obs, **preflight._default_successor_lock_observations(pc)}
        except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError, preflight.PreflightValidationError):
            return None
    if not preflight._validate_successor_lock(pc, lock_obs, base["candidate"]):
        return None
    if not _existing_attempt_root_safe(config) or not _output_root_safe(config):
        return None
    try:
        wheel_result = verify_reviewed_wheelhouse(
            config.effective_wheelhouse,
            base["candidate"]["resolved_wheels"],
            base["candidate"]["resolved_packages"],
        )
        if not wheel_result.get("ok") or tuple(wheel_result.get("delta_packages", ())) != EXPECTED_DELTA:
            return None
        delta_paths = tuple(Path(path) for path in wheel_result["delta_wheel_paths"])
        if len(delta_paths) != len(EXPECTED_DELTA) or not _validate_step4_evidence(config, obs, base["candidate"], delta_paths):
            return None
    except (ContractValidationError, OSError, TypeError, ValueError):
        return None
    return base, {"wheel_result": wheel_result, "delta_paths": delta_paths}


def _normalize_live_packages(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise LiveValidationError("LIVE_PACKAGE_SET_MISMATCH")
    result: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "version"}:
            raise LiveValidationError("LIVE_PACKAGE_SET_MISMATCH")
        name, version = item["name"], item["version"]
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            raise LiveValidationError("LIVE_PACKAGE_SET_MISMATCH")
        try:
            normalized = normalize_distribution_name(name)
        except ContractValidationError as error:
            raise LiveValidationError("LIVE_PACKAGE_SET_MISMATCH") from error
        if not normalized or normalized in names:
            raise LiveValidationError("LIVE_PACKAGE_SET_MISMATCH")
        names.add(normalized)
        result.append((normalized, version))
    result.sort()
    return tuple(result)


def _packages_as_json(packages: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"name": name, "version": version} for name, version in packages]


def _normalize_synthetic_probe_status(probe_result: Any) -> str:
    """Normalize a reviewed synthetic-probe helper's public status contract."""
    return "PASS" if isinstance(probe_result, Mapping) and probe_result.get("status") == "PASS" else "FAIL"


def _default_live_observations(config: LiveValidationConfig) -> dict[str, Any]:
    observations: dict[str, Any] = {
        "package_index_network_requests": 0,
        "package_installations": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_reads": 0,
        "t0_run": False,
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "sysconfig_platform": sysconfig.get_platform(),
    }
    try:
        observations["interpreter_executable"] = str(Path(sys.executable).resolve())
        distributions = []
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name")
            version = dist.version
            distributions.append({"name": name, "version": version})
        observations["observed_packages"] = _packages_as_json(_normalize_live_packages(distributions))
        for dist_name, key in (("pandas-market-calendars", "pandas_market_calendars_version"), ("exchange-calendars", "exchange_calendars_version")):
            observations[key] = importlib.metadata.version(dist_name)
        pmc = importlib.metadata.distribution("pandas-market-calendars")
        for relative, key in (
            ("pandas_market_calendars/calendars/jpx.py", "jpx_source_blob_sha1"),
            ("pandas_market_calendars/holidays/jp.py", "holiday_source_blob_sha1"),
        ):
            observations[key] = git_blob_sha1(Path(pmc.locate_file(relative)).read_bytes())
        from scripts.check_real_execution_env import check_jpx_xls_parser_synthetic_probe, check_pdf_parser_synthetic_probe

        observations["xls_probe_status"] = "PASS" if check_jpx_xls_parser_synthetic_probe().get("status") == "PASS" else "FAIL"
        observations["pdf_probe_status"] = _normalize_synthetic_probe_status(check_pdf_parser_synthetic_probe())
    except (OSError, ImportError, KeyError, TypeError, ValueError):
        observations.setdefault("observed_packages", None)
    return observations


def _base_evidence(config: LiveValidationConfig, *, failure_code: str, phase: Mapping[str, Any] | None = None) -> dict[str, Any]:
    phase = phase or {}
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "artifact_status": EVIDENCE_STATUS,
        "status": "PASS" if failure_code == "NONE" else "FAIL",
        "failure_code": failure_code,
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": EXTENSION_DESIGN_SHA,
        "reviewed_generic_authority_transition_git_sha": GENERIC_AUTHORITY_TRANSITION_SHA,
        "migration_authority_git_blob_sha1": MIGRATION_AUTHORITY_BLOB_SHA1,
        "generic_lock_git_blob_sha1": GENERIC_LOCK_BLOB_SHA1,
        "generic_lock_sha256": GENERIC_LOCK_SHA256,
        "generic_lock_package_count": GENERIC_LOCK_PACKAGE_COUNT,
        "reviewed_successor_lock_candidate_sha256": phase.get("candidate_sha256"),
        "installed_delta_packages": phase.get("installed_delta_packages"),
        "installed_delta_package_count": phase.get("installed_delta_package_count"),
        "installed_delta_wheel_count": phase.get("installed_delta_wheel_count"),
        "observed_packages": phase.get("observed_packages"),
        "observed_package_count": phase.get("observed_package_count"),
        "python_version": phase.get("python_version"),
        "platform_system": phase.get("platform_system"),
        "platform_machine": phase.get("platform_machine"),
        "sysconfig_platform": phase.get("sysconfig_platform"),
        "pandas_market_calendars_version": phase.get("pandas_market_calendars_version"),
        "exchange_calendars_version": phase.get("exchange_calendars_version"),
        "jpx_source_blob_match": phase.get("jpx_source_blob_match"),
        "holiday_source_blob_match": phase.get("holiday_source_blob_match"),
        "xls_probe_status": phase.get("xls_probe_status", "NOT_CHECKED"),
        "pdf_probe_status": phase.get("pdf_probe_status", "NOT_CHECKED"),
        "package_index_network_requests": phase.get("package_index_network_requests", 0),
        "package_installations": phase.get("package_installations", 0),
        "calendar_object_creations": phase.get("calendar_object_creations", 0),
        "calendar_dates_inspected": phase.get("calendar_dates_inspected", 0),
        "protected_or_private_reads": phase.get("protected_or_private_reads", 0),
        "t0_run": phase.get("t0_run", False),
    }


def validate_live_validation_evidence(evidence: Mapping[str, Any]) -> None:
    if not isinstance(evidence, dict) or set(evidence) != set(EVIDENCE_KEYS):
        raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_SCHEMA_INVALID")
    if evidence["schema_version"] != EVIDENCE_SCHEMA or evidence["artifact_status"] != EVIDENCE_STATUS:
        raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_SCHEMA_INVALID")
    if evidence["status"] not in {"PASS", "FAIL"} or evidence["failure_code"] not in FAILURE_CODES:
        raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_SCHEMA_INVALID")
    if (evidence["status"] == "PASS") != (evidence["failure_code"] == "NONE"):
        raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_STATUS_INVALID")
    for key, value in (
        ("frozen_v10_design_git_sha", evidence["frozen_v10_design_git_sha"]),
        ("extension_design_git_sha", evidence["extension_design_git_sha"]),
        ("reviewed_generic_authority_transition_git_sha", evidence["reviewed_generic_authority_transition_git_sha"]),
        ("migration_authority_git_blob_sha1", evidence["migration_authority_git_blob_sha1"]),
        ("generic_lock_git_blob_sha1", evidence["generic_lock_git_blob_sha1"]),
        ("reviewed_successor_lock_candidate_sha256", evidence["reviewed_successor_lock_candidate_sha256"]),
    ):
        if value is not None:
            _strict_sha(value, SHA256_RE if key.endswith("sha256") else SHA1_RE, key)
    if evidence["frozen_v10_design_git_sha"] != FROZEN_V10_DESIGN_SHA or evidence["extension_design_git_sha"] != EXTENSION_DESIGN_SHA or evidence["reviewed_generic_authority_transition_git_sha"] != GENERIC_AUTHORITY_TRANSITION_SHA or evidence["migration_authority_git_blob_sha1"] != MIGRATION_AUTHORITY_BLOB_SHA1 or evidence["generic_lock_git_blob_sha1"] != GENERIC_LOCK_BLOB_SHA1 or evidence["generic_lock_sha256"] != GENERIC_LOCK_SHA256 or evidence["generic_lock_package_count"] != GENERIC_LOCK_PACKAGE_COUNT:
        raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_PROVENANCE_INVALID")
    for key in ("generic_lock_package_count", "installed_delta_package_count", "installed_delta_wheel_count", "observed_package_count"):
        if evidence[key] is not None and (_strict_int(evidence[key], key) < 0):
            raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_INTEGER_INVALID")
    for key in ("jpx_source_blob_match", "holiday_source_blob_match"):
        _strict_bool_or_none(evidence[key], key)
    for key in (
        "python_version", "platform_system", "platform_machine", "sysconfig_platform",
        "pandas_market_calendars_version", "exchange_calendars_version",
    ):
        if evidence[key] is not None and not isinstance(evidence[key], str):
            raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_STRING_INVALID")
    if evidence["xls_probe_status"] not in {"PASS", "FAIL", "NOT_CHECKED"} or evidence["pdf_probe_status"] not in {"PASS", "FAIL", "NOT_CHECKED"}:
        raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_PROBE_INVALID")
    for key in ("package_index_network_requests", "package_installations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads"):
        if evidence[key] is not None and _strict_int(evidence[key], key) < 0:
            raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_COUNTER_INVALID")
    if evidence["t0_run"] is not None and not isinstance(evidence["t0_run"], bool):
        raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_BOOL_INVALID")
    observed = evidence["observed_packages"]
    if observed is not None:
        normalized = _normalize_live_packages(observed)
        if _packages_as_json(normalized) != observed or evidence["observed_package_count"] != len(normalized):
            raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_OBSERVED_PACKAGES_INVALID")
    elif evidence["observed_package_count"] is not None:
        raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_OBSERVED_COUNT_INVALID")
    installed = evidence["installed_delta_packages"]
    if installed is not None:
        normalized = _normalize_live_packages(installed)
        if normalized != EXPECTED_DELTA or evidence["installed_delta_package_count"] != len(normalized) or evidence["installed_delta_wheel_count"] != len(normalized):
            raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_DELTA_INVALID")
    elif any(evidence[key] is not None for key in ("installed_delta_package_count", "installed_delta_wheel_count")):
        raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_DELTA_INVALID")
    if evidence["status"] == "PASS":
        if evidence["failure_code"] != "NONE" or evidence["reviewed_successor_lock_candidate_sha256"] != CANDIDATE_SHA256 or evidence["installed_delta_packages"] is None or evidence["observed_packages"] is None:
            raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_PASS_INVALID")
        if (_normalize_live_packages(evidence["observed_packages"]) != EXPECTED_SUCCESSOR_PACKAGES or evidence["observed_package_count"] != GENERIC_LOCK_PACKAGE_COUNT or evidence["python_version"] != "3.12.10" or evidence["platform_system"] != "Windows" or evidence["platform_machine"] != "AMD64" or evidence["sysconfig_platform"] != "win-amd64" or evidence["pandas_market_calendars_version"] != "5.4.0" or evidence["exchange_calendars_version"] != "4.13.2" or evidence["jpx_source_blob_match"] is not True or evidence["holiday_source_blob_match"] is not True or evidence["xls_probe_status"] != "PASS" or evidence["pdf_probe_status"] != "PASS"):
            raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_PASS_INVALID")
        for key in ("package_index_network_requests", "package_installations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads"):
            if evidence[key] != 0:
                raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_PASS_INVALID")
        if evidence["t0_run"] is not False:
            raise LiveValidationError("LIVE_VALIDATION_EVIDENCE_PASS_INVALID")


def _publish(config: LiveValidationConfig, evidence: Mapping[str, Any]) -> Path:
    if not _output_root_safe(config):
        raise LiveValidationError("PROVENANCE_BINDING_FAILURE")
    try:
        config.output_root.mkdir(parents=True, exist_ok=False)
        target = config.output_root / EVIDENCE_NAME
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        fd = os.open(str(target), flags)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(canonical_json_bytes(evidence))
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                target.unlink()
            except OSError:
                pass
            raise
        return target
    except (FileExistsError, OSError) as error:
        raise LiveValidationError("PROVENANCE_BINDING_FAILURE") from error


def _result(config: LiveValidationConfig, evidence: dict[str, Any], *, publish: bool) -> dict[str, Any]:
    validate_live_validation_evidence(evidence)
    artifact_path = None
    if publish:
        artifact_path = _publish(config, evidence)
    return {"evidence": evidence, "artifact_path": artifact_path, "status": evidence["status"], "failure_code": evidence["failure_code"], "canonical_environment_ready": False, "environment_frozen": False, "execution_authorized": False}


def run_live_validation(config: LiveValidationConfig, observations: Mapping[str, Any] | None = None, *, publish: bool = True) -> dict[str, Any]:
    """Run the frozen Step-5 order; injected observations are test-only."""
    obs = dict(_default_provenance_observations(config) if observations is None else observations)
    if _unauthorized_observed(obs):
        unauthorized_phase = {
            key: obs.get(key, 0)
            for key in (
                "package_index_network_requests", "package_installations", "calendar_object_creations",
                "calendar_dates_inspected", "protected_or_private_reads", "t0_run",
            )
        }
        return _result(config, _base_evidence(config, failure_code="UNAUTHORIZED_OPERATION_OBSERVED", phase=unauthorized_phase), publish=publish)
    validated = _validate_provenance(config, obs)
    if validated is None:
        return _result(config, _base_evidence(config, failure_code="PROVENANCE_BINDING_FAILURE"), publish=publish)
    base, wheel = validated
    if observations is None:
        # This is the first live observation and is reachable only after the
        # complete provenance/Step-4/wheelhouse gate above has passed.
        obs.update(_default_live_observations(config))
    candidate = base["candidate"]
    delta_packages = tuple(wheel["wheel_result"].get("delta_packages", ()))
    if delta_packages != EXPECTED_DELTA or len(wheel["delta_paths"]) != len(delta_packages):
        return _result(config, _base_evidence(config, failure_code="PROVENANCE_BINDING_FAILURE"), publish=publish)
    installed = _packages_as_json(delta_packages)
    phase: dict[str, Any] = {
        "candidate_sha256": CANDIDATE_SHA256,
        "installed_delta_packages": installed,
        "installed_delta_package_count": len(delta_packages),
        "installed_delta_wheel_count": len(wheel["delta_paths"]),
        "package_index_network_requests": obs.get("package_index_network_requests", 0),
        "package_installations": obs.get("package_installations", 0),
        "calendar_object_creations": obs.get("calendar_object_creations", 0),
        "calendar_dates_inspected": obs.get("calendar_dates_inspected", 0),
        "protected_or_private_reads": obs.get("protected_or_private_reads", 0),
        "t0_run": obs.get("t0_run", False),
    }
    try:
        normalized = _normalize_live_packages(obs.get("observed_packages"))
    except (LiveValidationError, TypeError, ValueError):
        phase.update(observed_packages=None, observed_package_count=None)
        return _result(config, _base_evidence(config, failure_code="LIVE_PACKAGE_SET_MISMATCH", phase=phase), publish=publish)
    phase.update(observed_packages=_packages_as_json(normalized), observed_package_count=len(normalized))
    expected_packages = tuple((item["name"], item["version"]) for item in candidate["resolved_packages"])
    if normalized != expected_packages:
        return _result(config, _base_evidence(config, failure_code="LIVE_PACKAGE_SET_MISMATCH", phase=phase), publish=publish)
    for key in ("python_version", "platform_system", "platform_machine", "sysconfig_platform"):
        phase[key] = obs.get(key)
    interpreter_match = observations is not None or obs.get("interpreter_executable") == str(config.canonical_interpreter.resolve())
    if not interpreter_match or (phase["python_version"], phase["platform_system"], phase["platform_machine"], phase["sysconfig_platform"]) != ("3.12.10", "Windows", "AMD64", "win-amd64"):
        return _result(config, _base_evidence(config, failure_code="PYTHON_PLATFORM_MISMATCH", phase=phase), publish=publish)
    phase["pandas_market_calendars_version"] = obs.get("pandas_market_calendars_version")
    if phase["pandas_market_calendars_version"] != "5.4.0":
        return _result(config, _base_evidence(config, failure_code="PMC_VERSION_MISMATCH", phase=phase), publish=publish)
    phase["exchange_calendars_version"] = obs.get("exchange_calendars_version")
    if phase["exchange_calendars_version"] != "4.13.2":
        return _result(config, _base_evidence(config, failure_code="EXCHANGE_CALENDARS_VERSION_MISMATCH", phase=phase), publish=publish)
    phase["jpx_source_blob_match"] = obs.get("jpx_source_blob_sha1") == EXPECTED_PMC_SOURCE_BLOB_SHA1
    if not phase["jpx_source_blob_match"]:
        return _result(config, _base_evidence(config, failure_code="JPX_SOURCE_BLOB_MISMATCH", phase=phase), publish=publish)
    phase["holiday_source_blob_match"] = obs.get("holiday_source_blob_sha1") == EXPECTED_HOLIDAY_SOURCE_BLOB_SHA1
    if not phase["holiday_source_blob_match"]:
        return _result(config, _base_evidence(config, failure_code="HOLIDAY_SOURCE_BLOB_MISMATCH", phase=phase), publish=publish)
    phase["xls_probe_status"] = obs.get("xls_probe_status", "FAIL")
    if phase["xls_probe_status"] != "PASS":
        return _result(config, _base_evidence(config, failure_code="XLS_PROBE_FAILURE", phase=phase), publish=publish)
    phase["pdf_probe_status"] = obs.get("pdf_probe_status", "FAIL")
    if phase["pdf_probe_status"] != "PASS":
        return _result(config, _base_evidence(config, failure_code="PDF_PROBE_FAILURE", phase=phase), publish=publish)
    evidence = _base_evidence(config, failure_code="NONE", phase=phase)
    return _result(config, evidence, publish=publish)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-current-head", required=True)
    parser.add_argument("--expected-live-validation-runner-commit-sha", required=True)
    parser.add_argument("--expected-live-validation-runner-blob-sha1", required=True)
    parser.add_argument("--step4-attempt-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wheelhouse", required=True)
    return parser


def _config_from_args(args: argparse.Namespace) -> LiveValidationConfig:
    for value, pattern, label in (
        (args.expected_current_head, SHA1_RE, "expected current head"),
        (args.expected_live_validation_runner_commit_sha, SHA1_RE, "live runner commit"),
        (args.expected_live_validation_runner_blob_sha1, SHA1_RE, "live runner blob"),
    ):
        _strict_sha(value, pattern, label)
    paths = {name: Path(getattr(args, name)) for name in ("repo_root", "step4_attempt_root", "output_root", "wheelhouse")}
    if not paths["repo_root"].is_absolute() or not paths["step4_attempt_root"].is_absolute() or not paths["output_root"].is_absolute() or not paths["wheelhouse"].is_absolute():
        raise LiveValidationError("PATH_MUST_BE_ABSOLUTE")
    return LiveValidationConfig(
        repo_root=paths["repo_root"],
        expected_current_head=args.expected_current_head,
        expected_live_validation_runner_commit_sha=args.expected_live_validation_runner_commit_sha,
        expected_live_validation_runner_blob_sha1=args.expected_live_validation_runner_blob_sha1,
        step4_attempt_root=paths["step4_attempt_root"],
        output_root=paths["output_root"],
        wheelhouse=paths["wheelhouse"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _config_from_args(_build_parser().parse_args(argv))
        result = run_live_validation(config, observations=None, publish=True)
    except (ContractValidationError, OSError, TypeError, ValueError):
        result = {"status": "FAIL", "failure_code": "PROVENANCE_BINDING_FAILURE", "canonical_environment_ready": False, "environment_frozen": False, "execution_authorized": False}
    print(json.dumps({key: result.get(key) for key in ("status", "failure_code", "canonical_environment_ready", "environment_frozen", "execution_authorized")}, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
