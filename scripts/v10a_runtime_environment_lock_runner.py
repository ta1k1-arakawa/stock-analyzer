"""Create the V10A runtime-environment lock from the existing environment.

The production path is deliberately no-network and read-only with respect to
the canonical environment.  Synthetic tests patch narrow collectors; the
production CLI has no observation or process-injection input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
RUNNER_RELATIVE = Path("scripts/v10a_runtime_environment_lock_runner.py")
TEST_RELATIVE = Path("tests/test_v10a_runtime_environment_lock_runner.py")
DESIGN_RELATIVE = Path("V10A_RUNTIME_ENVIRONMENT_LOCK_OPERATIONAL_DESIGN.md")
FROZEN_DESIGN_RELATIVE = Path("V10A_CALENDAR_AUTHORITY_RELEASE_ARTIFACT_SUCCESSOR_DESIGN_DRAFT.md")
FINAL_EVIDENCE_RELATIVE = Path("V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json")
FINAL_ADJUDICATION_RELATIVE = Path("V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_ADJUDICATION.json")
STATE_RELATIVE = Path("PROJECT_STATE.md")
LOCK_NAME = "V10A_RUNTIME_ENVIRONMENT_LOCK.json"
EVIDENCE_NAME = "V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE.json"

APPROVED_DESIGN_SHA = "b14cc5510685210e928000af0815e188bc1aadc0"
APPROVED_DESIGN_BLOB_SHA1 = "3217b155c7d226f8f6edbcba2162c74e8d9d4e0e"
FREEZE_RECORD_SHA = "86ceda3dee531b08afa5db4df7af1298ca770fad"
FREEZE_RECORD_BLOB_SHA1 = "a3f913857966cb0593f3218d882c4f91b2bc1f2f"
P5_REVIEWED_P4_SHA = "1280c222f6114aa0363684a22d9b24e47cb1d5e0"
P5_BOOKKEEPING_SHA = "4090c9693ee7afa0fb6439a187d2cef4399f945c"
FINAL_EVIDENCE_BLOB_SHA1 = "d880b84fa00233e58653739fd510385fdf94de4e"
FINAL_EVIDENCE_SHA256 = "658e264a70ab15ba402e7bf56d5e4b8abe5d81f2f7bb22f28bc797b7b8062b01"
P3_ADJUDICATION_BLOB_SHA1 = "6f5571961bd1fa533ea7aea30730dc3383ddc2e8"

SCHEMA_VERSION = "V10A_RUNTIME_ENVIRONMENT_LOCK_V1"
EVIDENCE_SCHEMA = "V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE_V2"
EVIDENCE_STATUS = "V10A_RUNTIME_ENVIRONMENT_LOCK_EXECUTION_EVIDENCE"
PYTHON_VERSION = "3.12.10"
CALENDAR_DISTRIBUTION_NAME = "pandas_market_calendars"
CALENDAR_DISTRIBUTION_VERSION = "5.4.0"
CALENDAR_NAME = "JPX"
CALENDAR_SOURCE_FILE = "pandas_market_calendars/calendars/jpx.py"
CALENDAR_SOURCE_BLOB = "a7a59b6cf910e325c85fc042459ff57ca8f70613"
HOLIDAY_SOURCE_FILE = "pandas_market_calendars/holidays/jp.py"
HOLIDAY_SOURCE_BLOB = "4c34214d06862e02ac22e946757463f748074fde"
OLD_V10_JPX_BLOB = "0c2041b1300d1dbbd505202b00ac0ada38c712e1"

EXPECTED_PACKAGES = (
    ("cffi", "2.1.1"),
    ("charset-normalizer", "3.5.1"),
    ("cryptography", "50.0.1"),
    ("exchange-calendars", "4.13.2"),
    ("korean-lunar-calendar", "0.4.0"),
    ("numpy", "2.5.2"),
    ("pandas", "3.0.5"),
    ("pandas-market-calendars", "5.4.0"),
    ("pdfminer-six", "20260107"),
    ("pdfplumber", "0.11.10"),
    ("pillow", "12.3.0"),
    ("pip", "25.0.1"),
    ("pycparser", "3.0"),
    ("pyluach", "2.3.0"),
    ("pypdfium2", "5.13.0"),
    ("python-dateutil", "2.9.0.post0"),
    ("six", "1.17.0"),
    ("toolz", "1.1.0"),
    ("tzdata", "2026.3"),
    ("xlrd", "2.0.2"),
)

FAILURE_CODES = (
    "NONE",
    "UNAUTHORIZED_OPERATION_OBSERVED",
    "PROVENANCE_BINDING_FAILURE",
    "WRONG_CANONICAL_INTERPRETER",
    "PYTHON_VERSION_MISMATCH",
    "PACKAGE_SET_MISMATCH",
    "CALENDAR_DISTRIBUTION_VERSION_MISMATCH",
    "CALENDAR_SOURCE_BLOB_MISMATCH",
    "HOLIDAY_SOURCE_BLOB_MISMATCH",
    "RUNTIME_LOCK_CANONICALIZATION_FAILURE",
    "DURABLE_OUTPUT_COLLISION",
    "DURABLE_WRITE_FAILURE",
)

EVIDENCE_KEYS = frozenset(
    {
        "schema_version", "artifact_status", "status", "failure_code",
        "reviewed_baseline_sha", "runtime_lock_design_git_blob_sha1",
        "runtime_lock_runner_git_blob_sha1", "runtime_lock_test_git_blob_sha1",
        "frozen_v10a_design_sha", "v10a_freeze_record_sha", "p5_reviewed_p4_sha",
        "p5_bookkeeping_sha", "final_freeze_evidence_git_blob_sha1",
        "final_freeze_evidence_sha256", "p3_adjudication_git_blob_sha1",
        "python_version", "canonical_interpreter_verified",
        "calendar_distribution_name", "calendar_distribution_version", "calendar_name",
        "runtime_distribution_count", "exact_package_mapping", "calendar_source_blob",
        "holiday_source_blob", "durable_lock_created", "runtime_lock_sha256",
        "runtime_lock_size", "network_requests", "package_installations",
        "environment_mutations", "calendar_imports", "calendar_object_creations",
        "calendar_dates_inspected", "protected_or_private_research_reads", "t0_run",
        "execution_authorized", "calendar_generation_authorized", "t0_authorized",
        "historical_evaluation_authorized", "future_profitability_established",
    }
)
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeLockError(ValueError):
    """Internal fail-closed validation error."""


@dataclass(frozen=True)
class RuntimeLockConfig:
    repo_root: Path
    reviewed_baseline_sha: str
    expected_runner_blob_sha1: str
    expected_test_blob_sha1: str
    expected_design_blob_sha1: str
    output_root: Path

    @property
    def canonical_interpreter(self) -> Path:
        return self.repo_root / ".venv-real-execution" / "Scripts" / "python.exe"


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
        + b"\n"
    )


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def normalize_distribution_name(value: Any) -> str:
    if not isinstance(value, str):
        raise RuntimeLockError("PACKAGE_SET_MISMATCH")
    normalized = re.sub(r"[-_.]+", "-", value.lower())
    if not normalized:
        raise RuntimeLockError("PACKAGE_SET_MISMATCH")
    return normalized


def _strict_sha(value: Any, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RuntimeLockError(f"{label}_INVALID")


def _strict_int(value: Any, label: str, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeLockError(f"{label}_INVALID")


def _strict_bool(value: Any, label: str, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, bool):
        raise RuntimeLockError(f"{label}_INVALID")


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


def _unsafe_existing_ancestry(path: Path) -> bool:
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


def output_root_safe(config: RuntimeLockConfig) -> bool:
    root = config.output_root
    if not root.is_absolute() or os.path.lexists(root) or _unsafe_existing_ancestry(root):
        return False
    try:
        resolved_root = Path(os.path.realpath(root))
        repo = Path(os.path.realpath(config.repo_root))
    except OSError:
        return False
    if _path_inside(resolved_root, repo) or _path_inside(repo, resolved_root):
        return False
    return _nearest_existing(root).is_dir()


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


def _git_blob_at(repo_root: Path, revision: str, relative: Path) -> str:
    return _run_git(repo_root, ["rev-parse", f"{revision}:{relative.as_posix()}"]).decode("ascii").strip()


def _repository_identity(repo_root: Path) -> str:
    return _run_git(repo_root, ["config", "--get", "remote.origin.url"]).decode("utf-8").strip()


def _repository_identity_matches(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().rstrip("/")
    for prefix in ("git@github.com:", "https://github.com/", "http://github.com/", "ssh://git@github.com/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    return normalized == REPOSITORY_IDENTITY


def _state_values(repo_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (repo_root / STATE_RELATIVE).read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _default_provenance_observations(config: RuntimeLockConfig) -> dict[str, Any]:
    """Collect repository-only facts; never inspect the environment."""
    try:
        return {
            "repository_identity": _repository_identity(config.repo_root),
            "branch": _run_git(config.repo_root, ["branch", "--show-current"]).decode().strip(),
            "head": _run_git(config.repo_root, ["rev-parse", "HEAD"]).decode().strip(),
            "clean": _run_git(config.repo_root, ["status", "--porcelain", "--untracked-files=all"]) == b"",
            "design_blob": _git_blob_at(config.repo_root, "HEAD", DESIGN_RELATIVE),
            "current_runner_blob": _git_blob_at(config.repo_root, "HEAD", RUNNER_RELATIVE),
            "current_test_blob": _git_blob_at(config.repo_root, "HEAD", TEST_RELATIVE),
            "approved_design_exists": _git_exists(config.repo_root, APPROVED_DESIGN_SHA),
            "approved_design_blob": _git_blob_at(config.repo_root, APPROVED_DESIGN_SHA, FROZEN_DESIGN_RELATIVE),
            "freeze_record_exists": _git_exists(config.repo_root, FREEZE_RECORD_SHA),
            "freeze_record_blob": _git_blob_at(config.repo_root, FREEZE_RECORD_SHA, FROZEN_DESIGN_RELATIVE),
            "p5_p4_exists": _git_exists(config.repo_root, P5_REVIEWED_P4_SHA),
            "p5_bookkeeping_exists": _git_exists(config.repo_root, P5_BOOKKEEPING_SHA),
            "final_evidence_blob": _git_blob_at(config.repo_root, P5_BOOKKEEPING_SHA, FINAL_EVIDENCE_RELATIVE),
            "final_evidence_sha256": hashlib.sha256(_run_git(config.repo_root, ["show", f"{P5_BOOKKEEPING_SHA}:{FINAL_EVIDENCE_RELATIVE.as_posix()}"])).hexdigest(),
            "current_final_evidence_blob": _git_blob_at(config.repo_root, "HEAD", FINAL_EVIDENCE_RELATIVE),
            "final_adjudication_blob": _git_blob_at(config.repo_root, P5_BOOKKEEPING_SHA, FINAL_ADJUDICATION_RELATIVE),
            "current_final_adjudication_blob": _git_blob_at(config.repo_root, "HEAD", FINAL_ADJUDICATION_RELATIVE),
            "state": _state_values(config.repo_root),
        }
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError):
        return {}


def validate_provenance(config: RuntimeLockConfig, obs: Mapping[str, Any]) -> bool:
    try:
        _strict_sha(config.reviewed_baseline_sha, SHA1_RE, "reviewed_baseline_sha")
        _strict_sha(config.expected_runner_blob_sha1, SHA1_RE, "expected_runner_blob_sha1")
        _strict_sha(config.expected_test_blob_sha1, SHA1_RE, "expected_test_blob_sha1")
        _strict_sha(config.expected_design_blob_sha1, SHA1_RE, "expected_design_blob_sha1")
    except RuntimeLockError:
        return False
    state = obs.get("state")
    if not isinstance(state, Mapping):
        return False
    return all(
        (
            _repository_identity_matches(obs.get("repository_identity")),
            obs.get("branch") == AUTHORITATIVE_BRANCH,
            obs.get("head") == config.reviewed_baseline_sha,
            obs.get("clean") is True,
            obs.get("design_blob") == config.expected_design_blob_sha1,
            obs.get("current_runner_blob") == config.expected_runner_blob_sha1,
            obs.get("current_test_blob") == config.expected_test_blob_sha1,
            obs.get("approved_design_exists") is True,
            obs.get("approved_design_blob") == APPROVED_DESIGN_BLOB_SHA1,
            obs.get("freeze_record_exists") is True,
            obs.get("freeze_record_blob") == FREEZE_RECORD_BLOB_SHA1,
            obs.get("p5_p4_exists") is True,
            obs.get("p5_bookkeeping_exists") is True,
            obs.get("final_evidence_blob") == FINAL_EVIDENCE_BLOB_SHA1,
            obs.get("final_evidence_sha256") == FINAL_EVIDENCE_SHA256,
            obs.get("current_final_evidence_blob") == FINAL_EVIDENCE_BLOB_SHA1,
            obs.get("final_adjudication_blob") == P3_ADJUDICATION_BLOB_SHA1,
            obs.get("current_final_adjudication_blob") == P3_ADJUDICATION_BLOB_SHA1,
            state.get("V10A_ENVIRONMENT_STATE") == "CANONICAL_FROZEN",
            state.get("V10A_CANONICAL_ENVIRONMENT_PROMOTED") == "true",
            state.get("V10A_ENVIRONMENT_FROZEN") == "true",
            state.get("V10A_EXECUTION_AUTHORIZED") == "false",
            state.get("V10A_CALENDAR_GENERATION_AUTHORIZED") == "false",
            state.get("V10A_T0_AUTHORIZED") == "false",
            state.get("V10A_HISTORICAL_EVALUATION_AUTHORIZED") == "false",
        )
    )


def _default_operation_observations() -> dict[str, Any]:
    return {
        "network_requests": 0,
        "package_installations": 0,
        "environment_mutations": 0,
        "calendar_imports": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_research_reads": 0,
        "t0_run": False,
    }


def _unauthorized_operation(obs: Mapping[str, Any]) -> bool:
    if obs.get("t0_run") is True or obs.get("unauthorized_operation_observed") is True:
        return True
    for key in (
        "network_requests", "package_installations", "environment_mutations", "calendar_imports",
        "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_research_reads",
    ):
        value = obs.get(key, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    return False


def _default_interpreter_observations(config: RuntimeLockConfig) -> dict[str, Any]:
    return {
        "executable": str(Path(sys.executable).resolve()),
        "expected_executable": str(config.canonical_interpreter.resolve()),
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
    }


def _default_package_observations() -> dict[str, Any]:
    packages: list[tuple[str, str]] = []
    names: set[str] = set()
    try:
        for distribution in importlib.metadata.distributions():
            name = normalize_distribution_name(distribution.metadata.get("Name"))
            if name in names:
                raise RuntimeLockError("PACKAGE_SET_MISMATCH")
            names.add(name)
            version = distribution.version
            if not isinstance(version, str) or not version:
                raise RuntimeLockError("PACKAGE_SET_MISMATCH")
            packages.append((name, version))
    except RuntimeLockError:
        raise
    except Exception as exc:
        raise RuntimeLockError("PACKAGE_SET_MISMATCH") from exc
    return {"runtime_distributions": sorted(packages), "runtime_distribution_count": len(packages)}


def validate_package_observations(obs: Mapping[str, Any]) -> bool:
    packages = obs.get("runtime_distributions")
    if not isinstance(packages, list):
        return False
    normalized: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in packages:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        try:
            name = normalize_distribution_name(item[0])
        except RuntimeLockError:
            return False
        version = item[1]
        if not isinstance(version, str) or not version or name in names:
            return False
        names.add(name)
        normalized.append((name, version))
    return (
        obs.get("runtime_distribution_count") == len(normalized)
        and normalized == sorted(normalized)
        and tuple(normalized) == EXPECTED_PACKAGES
    )


def _default_source_observations() -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("pandas-market-calendars")
        jpx_path = Path(distribution.locate_file(CALENDAR_SOURCE_FILE))
        jp_path = Path(distribution.locate_file(HOLIDAY_SOURCE_FILE))
        jpx_bytes = jpx_path.read_bytes()
        jp_bytes = jp_path.read_bytes()
    except Exception as exc:
        raise RuntimeLockError("CALENDAR_SOURCE_BLOB_MISMATCH") from exc
    return {
        "calendar_source_blob": git_blob_sha1(jpx_bytes),
        "holiday_source_blob": git_blob_sha1(jp_bytes),
    }


def validate_source_observations(obs: Mapping[str, Any]) -> tuple[bool, bool]:
    return (
        obs.get("calendar_source_blob") == CALENDAR_SOURCE_BLOB and obs.get("calendar_source_blob") != OLD_V10_JPX_BLOB,
        obs.get("holiday_source_blob") == HOLIDAY_SOURCE_BLOB,
    )


def build_runtime_lock(package_obs: Mapping[str, Any]) -> dict[str, Any]:
    packages = package_obs.get("runtime_distributions")
    if not validate_package_observations(package_obs):
        raise RuntimeLockError("PACKAGE_SET_MISMATCH")
    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": PYTHON_VERSION,
        "calendar_distribution_name": CALENDAR_DISTRIBUTION_NAME,
        "calendar_distribution_version": CALENDAR_DISTRIBUTION_VERSION,
        "calendar_name": CALENDAR_NAME,
        "calendar_source_blob": CALENDAR_SOURCE_BLOB,
        "holiday_source_blob": HOLIDAY_SOURCE_BLOB,
        "runtime_distributions": [
            {"name": name, "version": version} for name, version in packages
        ],
        "runtime_distribution_count": 20,
    }


def validate_runtime_lock(lock: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version", "python_version", "calendar_distribution_name",
        "calendar_distribution_version", "calendar_name", "calendar_source_blob",
        "holiday_source_blob", "runtime_distributions", "runtime_distribution_count",
    }
    if set(lock) != expected_keys:
        raise RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")
    if lock["schema_version"] != SCHEMA_VERSION or lock["python_version"] != PYTHON_VERSION:
        raise RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")
    if lock["calendar_distribution_name"] != CALENDAR_DISTRIBUTION_NAME:
        raise RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")
    if lock["calendar_distribution_version"] != CALENDAR_DISTRIBUTION_VERSION or lock["calendar_name"] != CALENDAR_NAME:
        raise RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")
    if lock["calendar_source_blob"] != CALENDAR_SOURCE_BLOB or lock["holiday_source_blob"] != HOLIDAY_SOURCE_BLOB:
        raise RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")
    packages = lock["runtime_distributions"]
    if not isinstance(packages, list) or len(packages) != 20:
        raise RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")
    pairs = []
    for item in packages:
        if not isinstance(item, Mapping) or set(item) != {"name", "version"}:
            raise RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")
        pairs.append((item["name"], item["version"]))
    if tuple(pairs) != EXPECTED_PACKAGES or lock["runtime_distribution_count"] != 20:
        raise RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")


def _base_evidence(config: RuntimeLockConfig, failure_code: str) -> dict[str, Any]:
    if failure_code not in FAILURE_CODES:
        raise RuntimeLockError("UNKNOWN_FAILURE_CODE")
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "artifact_status": EVIDENCE_STATUS,
        "status": "PASS" if failure_code == "NONE" else "FAIL",
        "failure_code": failure_code,
        "reviewed_baseline_sha": config.reviewed_baseline_sha,
        "runtime_lock_design_git_blob_sha1": config.expected_design_blob_sha1,
        "runtime_lock_runner_git_blob_sha1": config.expected_runner_blob_sha1,
        "runtime_lock_test_git_blob_sha1": config.expected_test_blob_sha1,
        "frozen_v10a_design_sha": APPROVED_DESIGN_SHA,
        "v10a_freeze_record_sha": FREEZE_RECORD_SHA,
        "p5_reviewed_p4_sha": P5_REVIEWED_P4_SHA,
        "p5_bookkeeping_sha": P5_BOOKKEEPING_SHA,
        "final_freeze_evidence_git_blob_sha1": FINAL_EVIDENCE_BLOB_SHA1,
        "final_freeze_evidence_sha256": FINAL_EVIDENCE_SHA256,
        "p3_adjudication_git_blob_sha1": P3_ADJUDICATION_BLOB_SHA1,
        "python_version": None,
        "canonical_interpreter_verified": None,
        "calendar_distribution_name": CALENDAR_DISTRIBUTION_NAME,
        "calendar_distribution_version": None,
        "calendar_name": CALENDAR_NAME,
        "runtime_distribution_count": None,
        "exact_package_mapping": None,
        "calendar_source_blob": None,
        "holiday_source_blob": None,
        "durable_lock_created": False,
        "runtime_lock_sha256": None,
        "runtime_lock_size": None,
        "network_requests": 0,
        "package_installations": 0,
        "environment_mutations": 0,
        "calendar_imports": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_research_reads": 0,
        "t0_run": False,
        "execution_authorized": False,
        "calendar_generation_authorized": False,
        "t0_authorized": False,
        "historical_evaluation_authorized": False,
        "future_profitability_established": False,
    }


def _complete_evidence(evidence: dict[str, Any], lock_bytes: bytes | None = None) -> dict[str, Any]:
    if lock_bytes is not None:
        evidence["runtime_lock_sha256"] = hashlib.sha256(lock_bytes).hexdigest()
        evidence["runtime_lock_size"] = len(lock_bytes)
    return evidence


def validate_evidence(evidence: Mapping[str, Any], config: RuntimeLockConfig | None = None) -> None:
    if set(evidence) != EVIDENCE_KEYS:
        raise RuntimeLockError("EVIDENCE_KEYSET_INVALID")
    if evidence["schema_version"] != EVIDENCE_SCHEMA or evidence["artifact_status"] != EVIDENCE_STATUS:
        raise RuntimeLockError("EVIDENCE_SCHEMA_INVALID")
    if evidence["status"] not in {"PASS", "FAIL"} or evidence["failure_code"] not in FAILURE_CODES:
        raise RuntimeLockError("EVIDENCE_STATUS_INVALID")
    if (evidence["status"] == "PASS") != (evidence["failure_code"] == "NONE"):
        raise RuntimeLockError("EVIDENCE_STATUS_INVALID")
    for key in ("reviewed_baseline_sha", "runtime_lock_runner_git_blob_sha1", "runtime_lock_test_git_blob_sha1", "runtime_lock_design_git_blob_sha1", "frozen_v10a_design_sha", "v10a_freeze_record_sha", "p5_reviewed_p4_sha", "p5_bookkeeping_sha", "final_freeze_evidence_git_blob_sha1", "p3_adjudication_git_blob_sha1"):
        _strict_sha(evidence[key], SHA1_RE, key)
    _strict_sha(evidence["final_freeze_evidence_sha256"], SHA256_RE, "final_freeze_evidence_sha256")
    for key in ("canonical_interpreter_verified", "exact_package_mapping", "durable_lock_created", "t0_run", "execution_authorized", "calendar_generation_authorized", "t0_authorized", "historical_evaluation_authorized", "future_profitability_established"):
        _strict_bool(evidence[key], key)
    for key in ("network_requests", "package_installations", "environment_mutations", "calendar_imports", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_research_reads", "runtime_distribution_count", "runtime_lock_size"):
        _strict_int(evidence[key], key)
    if evidence["runtime_lock_sha256"] is not None:
        _strict_sha(evidence["runtime_lock_sha256"], SHA256_RE, "runtime_lock_sha256")
    if config is not None:
        for key, expected in (
            ("frozen_v10a_design_sha", APPROVED_DESIGN_SHA),
            ("v10a_freeze_record_sha", FREEZE_RECORD_SHA),
            ("p5_reviewed_p4_sha", P5_REVIEWED_P4_SHA),
            ("p5_bookkeeping_sha", P5_BOOKKEEPING_SHA),
            ("final_freeze_evidence_git_blob_sha1", FINAL_EVIDENCE_BLOB_SHA1),
            ("final_freeze_evidence_sha256", FINAL_EVIDENCE_SHA256),
            ("p3_adjudication_git_blob_sha1", P3_ADJUDICATION_BLOB_SHA1),
            ("reviewed_baseline_sha", config.reviewed_baseline_sha),
            ("runtime_lock_runner_git_blob_sha1", config.expected_runner_blob_sha1),
            ("runtime_lock_test_git_blob_sha1", config.expected_test_blob_sha1),
            ("runtime_lock_design_git_blob_sha1", config.expected_design_blob_sha1),
        ):
            if evidence[key] != expected:
                raise RuntimeLockError("EVIDENCE_PROVENANCE_INVALID")
    if evidence["status"] == "PASS":
        if evidence["python_version"] != PYTHON_VERSION or evidence["canonical_interpreter_verified"] is not True:
            raise RuntimeLockError("EVIDENCE_INTERPRETER_INVALID")
        if (
            evidence["calendar_distribution_name"] != CALENDAR_DISTRIBUTION_NAME
            or evidence["calendar_distribution_version"] != CALENDAR_DISTRIBUTION_VERSION
            or evidence["calendar_name"] != CALENDAR_NAME
        ):
            raise RuntimeLockError("EVIDENCE_CALENDAR_INVALID")
        if evidence["runtime_distribution_count"] != 20 or evidence["exact_package_mapping"] is not True:
            raise RuntimeLockError("EVIDENCE_PACKAGE_INVALID")
        if evidence["calendar_source_blob"] != CALENDAR_SOURCE_BLOB or evidence["holiday_source_blob"] != HOLIDAY_SOURCE_BLOB:
            raise RuntimeLockError("EVIDENCE_SOURCE_INVALID")
        if any(evidence[key] != 0 for key in ("network_requests", "package_installations", "environment_mutations", "calendar_imports", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_research_reads")):
            raise RuntimeLockError("EVIDENCE_COUNTER_INVALID")
        if any(evidence[key] is not False for key in ("execution_authorized", "calendar_generation_authorized", "t0_authorized", "historical_evaluation_authorized", "future_profitability_established")):
            raise RuntimeLockError("EVIDENCE_AUTHORITY_INVALID")
        if evidence["runtime_lock_sha256"] is None or evidence["runtime_lock_size"] is None or not isinstance(evidence["runtime_lock_size"], int) or isinstance(evidence["runtime_lock_size"], bool) or evidence["runtime_lock_size"] <= 0 or evidence["durable_lock_created"] is not True:
            raise RuntimeLockError("EVIDENCE_LOCK_INVALID")
        if evidence["t0_run"] is not False:
            raise RuntimeLockError("EVIDENCE_T0_INVALID")
    else:
        if evidence["failure_code"] == "NONE":
            raise RuntimeLockError("EVIDENCE_STATUS_INVALID")
        if evidence["t0_run"] is not False:
            raise RuntimeLockError("EVIDENCE_T0_INVALID")
        if any(evidence[key] is not False for key in ("execution_authorized", "calendar_generation_authorized", "t0_authorized", "historical_evaluation_authorized", "future_profitability_established")):
            raise RuntimeLockError("EVIDENCE_AUTHORITY_INVALID")
        if evidence["failure_code"] != "UNAUTHORIZED_OPERATION_OBSERVED" and any(
            evidence[key] != 0 for key in ("network_requests", "package_installations", "environment_mutations", "calendar_imports", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_research_reads")
        ):
            raise RuntimeLockError("EVIDENCE_COUNTER_INVALID")
        if evidence["failure_code"] != "DURABLE_WRITE_FAILURE":
            if evidence["durable_lock_created"] is not False or evidence["runtime_lock_sha256"] is not None or evidence["runtime_lock_size"] is not None:
                raise RuntimeLockError("EVIDENCE_LOCK_INVALID")
        elif evidence["durable_lock_created"] is True:
            if evidence["runtime_lock_sha256"] is None or evidence["runtime_lock_size"] is None or evidence["runtime_lock_size"] <= 0:
                raise RuntimeLockError("EVIDENCE_LOCK_INVALID")
        elif evidence["durable_lock_created"] is False:
            hashes_absent = evidence["runtime_lock_sha256"] is None and evidence["runtime_lock_size"] is None
            hashes_present = evidence["runtime_lock_sha256"] is not None and evidence["runtime_lock_size"] is not None and evidence["runtime_lock_size"] > 0
            if not (hashes_absent or hashes_present):
                raise RuntimeLockError("EVIDENCE_LOCK_INVALID")
        else:
            raise RuntimeLockError("EVIDENCE_LOCK_INVALID")


def _exclusive_write(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _publish_evidence_once(config: RuntimeLockConfig, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Write one already-validated evidence artifact, without retrying."""
    validate_evidence(evidence, config)
    try:
        _exclusive_write(config.output_root / EVIDENCE_NAME, canonical_json_bytes(evidence))
        return dict(evidence)
    except OSError:
        summary = dict(evidence)
        summary["status"] = "FAIL"
        summary["failure_code"] = "DURABLE_WRITE_FAILURE"
        validate_evidence(summary, config)
        return summary


def _post_provenance_failure(config: RuntimeLockConfig, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Publish a safe FAIL result after the trusted output root exists."""
    return _publish_evidence_once(config, evidence)


def publish_artifacts(config: RuntimeLockConfig, lock_bytes: bytes, evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Create the two artifacts with one lock write and one evidence write."""
    try:
        _exclusive_write(config.output_root / LOCK_NAME, lock_bytes)
    except OSError:
        failure = dict(evidence)
        failure["status"] = "FAIL"
        failure["failure_code"] = "DURABLE_WRITE_FAILURE"
        failure["durable_lock_created"] = False
        _complete_evidence(failure, lock_bytes)
        return _publish_evidence_once(config, failure)
    return _publish_evidence_once(config, evidence)


def run_lock(config: RuntimeLockConfig) -> dict[str, Any]:
    """Run the production sequence; no caller-supplied observations exist."""
    operation = _default_operation_observations()
    if _unauthorized_operation(operation):
        evidence = _base_evidence(config, "UNAUTHORIZED_OPERATION_OBSERVED")
        validate_evidence(evidence)
        return evidence

    provenance = _default_provenance_observations(config)
    if not validate_provenance(config, provenance):
        evidence = _base_evidence(config, "PROVENANCE_BINDING_FAILURE")
        validate_evidence(evidence)
        return evidence
    if not output_root_safe(config):
        evidence = _base_evidence(config, "DURABLE_OUTPUT_COLLISION")
        validate_evidence(evidence, config)
        return evidence
    try:
        config.output_root.mkdir()
    except OSError:
        evidence = _base_evidence(config, "DURABLE_WRITE_FAILURE")
        validate_evidence(evidence, config)
        return evidence

    evidence = _base_evidence(config, "NONE")
    interpreter = _default_interpreter_observations(config)
    evidence["python_version"] = interpreter.get("python_version")
    evidence["canonical_interpreter_verified"] = interpreter.get("executable") == interpreter.get("expected_executable")
    if evidence["canonical_interpreter_verified"] is not True:
        evidence["status"] = "FAIL"
        evidence["failure_code"] = "WRONG_CANONICAL_INTERPRETER"
        return _post_provenance_failure(config, evidence)
    if evidence["python_version"] != PYTHON_VERSION:
        evidence["status"] = "FAIL"
        evidence["failure_code"] = "PYTHON_VERSION_MISMATCH"
        return _post_provenance_failure(config, evidence)

    try:
        package_obs = _default_package_observations()
    except RuntimeLockError:
        package_obs = {}
    evidence["runtime_distribution_count"] = package_obs.get("runtime_distribution_count")
    evidence["exact_package_mapping"] = validate_package_observations(package_obs)
    if evidence["exact_package_mapping"] is not True:
        evidence["status"] = "FAIL"
        evidence["failure_code"] = "PACKAGE_SET_MISMATCH"
        return _post_provenance_failure(config, evidence)
    packages = package_obs["runtime_distributions"]
    if dict(packages).get("pandas-market-calendars") != CALENDAR_DISTRIBUTION_VERSION:
        evidence["status"] = "FAIL"
        evidence["failure_code"] = "CALENDAR_DISTRIBUTION_VERSION_MISMATCH"
        return _post_provenance_failure(config, evidence)
    evidence["calendar_distribution_version"] = CALENDAR_DISTRIBUTION_VERSION

    try:
        source_obs = _default_source_observations()
    except RuntimeLockError:
        source_obs = {}
    evidence["calendar_source_blob"] = source_obs.get("calendar_source_blob")
    evidence["holiday_source_blob"] = source_obs.get("holiday_source_blob")
    jpx_ok, holiday_ok = validate_source_observations(source_obs)
    if not jpx_ok:
        evidence["status"] = "FAIL"
        evidence["failure_code"] = "CALENDAR_SOURCE_BLOB_MISMATCH"
        return _post_provenance_failure(config, evidence)
    if not holiday_ok:
        evidence["status"] = "FAIL"
        evidence["failure_code"] = "HOLIDAY_SOURCE_BLOB_MISMATCH"
        return _post_provenance_failure(config, evidence)

    try:
        lock = build_runtime_lock(package_obs)
        validate_runtime_lock(lock)
        lock_bytes = canonical_json_bytes(lock)
        if lock_bytes.count(b"\n") != 1 or not lock_bytes.endswith(b"\n"):
            raise RuntimeLockError("RUNTIME_LOCK_CANONICALIZATION_FAILURE")
    except (RuntimeLockError, TypeError, ValueError, UnicodeError):
        evidence["status"] = "FAIL"
        evidence["failure_code"] = "RUNTIME_LOCK_CANONICALIZATION_FAILURE"
        return _post_provenance_failure(config, evidence)

    evidence["durable_lock_created"] = True
    _complete_evidence(evidence, lock_bytes)
    try:
        return publish_artifacts(config, lock_bytes, evidence)
    except (OSError, ValueError):
        evidence["status"] = "FAIL"
        evidence["failure_code"] = "DURABLE_WRITE_FAILURE"
        evidence["durable_lock_created"] = False
        _complete_evidence(evidence, lock_bytes)
        return _publish_evidence_once(config, evidence)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--reviewed-baseline-sha", required=True)
    parser.add_argument("--expected-runner-blob-sha1", required=True)
    parser.add_argument("--expected-test-blob-sha1", required=True)
    parser.add_argument("--expected-design-blob-sha1", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def _config_from_args(args: argparse.Namespace) -> RuntimeLockConfig:
    config = RuntimeLockConfig(
        repo_root=Path(args.repo_root),
        reviewed_baseline_sha=args.reviewed_baseline_sha,
        expected_runner_blob_sha1=args.expected_runner_blob_sha1,
        expected_test_blob_sha1=args.expected_test_blob_sha1,
        expected_design_blob_sha1=args.expected_design_blob_sha1,
        output_root=Path(args.output_root),
    )
    if not config.repo_root.is_absolute() or not config.output_root.is_absolute():
        raise RuntimeLockError("PATH_INVALID")
    for value, pattern, label in (
        (config.reviewed_baseline_sha, SHA1_RE, "reviewed_baseline_sha"),
        (config.expected_runner_blob_sha1, SHA1_RE, "expected_runner_blob_sha1"),
        (config.expected_test_blob_sha1, SHA1_RE, "expected_test_blob_sha1"),
        (config.expected_design_blob_sha1, SHA1_RE, "expected_design_blob_sha1"),
    ):
        _strict_sha(value, pattern, label)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _config_from_args(_build_parser().parse_args(argv))
        evidence = run_lock(config)
    except (RuntimeLockError, OSError, ValueError):
        print(json.dumps({"status": "FAIL", "failure_code": "PROVENANCE_BINDING_FAILURE"}, separators=(",", ":")))
        return 2
    print(json.dumps({key: evidence[key] for key in ("status", "failure_code", "durable_lock_created", "execution_authorized", "calendar_generation_authorized", "t0_authorized", "historical_evaluation_authorized")}, separators=(",", ":")))
    return 0 if evidence["status"] == "PASS" and evidence["failure_code"] == "NONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
