"""Synthetic-testable, no-network V10A environment validation runner.

The production entry point observes only the canonical interpreter and local
reviewed artifacts.  The observation mapping is an internal test seam; the
CLI never accepts or constructs one.  No package installation, network
request, calendar construction, or date inspection is present in this
module.
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
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts import v10_environment_mutation_preflight_runner as preflight
    from scripts import v10_environment_successor_live_validation_runner as v10_live
    from scripts.v10_environment_extension_contract import (
        ContractValidationError,
        PREDECESSOR_PACKAGE_SET,
        normalize_distribution_name,
        verify_reviewed_wheelhouse,
    )
except ModuleNotFoundError:  # direct ``python scripts/<runner>.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import v10_environment_mutation_preflight_runner as preflight
    import v10_environment_successor_live_validation_runner as v10_live
    from v10_environment_extension_contract import (
        ContractValidationError,
        PREDECESSOR_PACKAGE_SET,
        normalize_distribution_name,
        verify_reviewed_wheelhouse,
    )


REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
CANONICAL_ENVIRONMENT_RELATIVE = Path(".venv-real-execution")
CANONICAL_INTERPRETER_RELATIVE = CANONICAL_ENVIRONMENT_RELATIVE / "Scripts" / "python.exe"
WHEELHOUSE_RELATIVE = Path("wheelhouse")
RUNNER_RELATIVE = Path("scripts/v10a_environment_no_network_validation_runner.py")
FROZEN_DESIGN_RELATIVE = Path("V10A_CALENDAR_AUTHORITY_RELEASE_ARTIFACT_SUCCESSOR_DESIGN_DRAFT.md")
STEP3_RECEIPT_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_MUTATION_PREFLIGHT_RECEIPT.json")
OFFICIAL_WHEEL_FILENAME = "pandas_market_calendars-5.4.0-py3-none-any.whl"
EVIDENCE_NAME = "V10A_CANONICAL_ENVIRONMENT_NO_NETWORK_VALIDATION_EVIDENCE.json"
EVIDENCE_SCHEMA = "V10A_CANONICAL_ENVIRONMENT_NO_NETWORK_VALIDATION_EVIDENCE_V1"
EVIDENCE_STATUS = "V10A_CANONICAL_ENVIRONMENT_NO_NETWORK_VALIDATION_EVIDENCE"

APPROVED_DESIGN_SHA = "b14cc5510685210e928000af0815e188bc1aadc0"
APPROVED_DESIGN_BLOB_SHA1 = "3217b155c7d226f8f6edbcba2162c74e8d9d4e0e"
FREEZE_RECORD_SHA = "86ceda3dee531b08afa5db4df7af1298ca770fad"
FREEZE_RECORD_BLOB_SHA1 = "a3f913857966cb0593f3218d882c4f91b2bc1f2f"
PMC_VERSION = "5.4.0"
EXCHANGE_CALENDARS_VERSION = "4.13.2"
OFFICIAL_WHEEL_SHA256 = "bb2b93b28d496cab173b41c7d120fd5cd9d506b31f3bb0ad3d1d9f2b60d9d9e3"
JPX_ENTRY = "pandas_market_calendars/calendars/jpx.py"
JPX_RELEASE_GIT_BLOB_SHA1 = "a7a59b6cf910e325c85fc042459ff57ca8f70613"
JP_ENTRY = "pandas_market_calendars/holidays/jp.py"
JP_RELEASE_GIT_BLOB_SHA1 = "4c34214d06862e02ac22e946757463f748074fde"
EXPECTED_DELTA = (
    ("exchange-calendars", "4.13.2"),
    ("korean-lunar-calendar", "0.4.0"),
    ("pandas-market-calendars", "5.4.0"),
    ("pyluach", "2.3.0"),
    ("toolz", "1.1.0"),
)
EXPECTED_SUCCESSOR_PACKAGES = tuple(sorted((*PREDECESSOR_PACKAGE_SET, *EXPECTED_DELTA)))
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
    "OFFICIAL_WHEEL_IDENTITY_MISMATCH",
    "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE",
    "JPX_INSTALLED_WHEEL_BYTES_MISMATCH",
    "JPX_RELEASE_BLOB_MISMATCH",
    "HOLIDAY_INSTALLED_WHEEL_BYTES_MISMATCH",
    "HOLIDAY_RELEASE_BLOB_MISMATCH",
    "XLS_PROBE_FAILURE",
    "PDF_PROBE_FAILURE",
)
EVIDENCE_KEYS = frozenset(
    {
        "schema_version", "artifact_status", "status", "failure_code",
        "approved_design_sha", "freeze_record_sha", "official_wheel_filename",
        "observed_official_wheel_sha256", "official_wheel_sha256_match",
        "jpx_entry_occurrence_count", "jp_entry_occurrence_count",
        "jpx_installed_equals_wheel_entry", "jp_installed_equals_wheel_entry",
        "jpx_wheel_git_blob_sha1", "jp_wheel_git_blob_sha1",
        "jpx_installed_git_blob_sha1", "jp_installed_git_blob_sha1",
        "jpx_source_blob_match", "holiday_source_blob_match",
        "observed_packages", "observed_package_count", "python_version",
        "platform_system", "platform_machine", "sysconfig_platform",
        "pandas_market_calendars_version", "exchange_calendars_version",
        "xls_probe_status", "pdf_probe_status",
        "historical_step4_provenance_verified", "reviewed_wheelhouse_provenance_verified",
        "package_index_network_requests", "package_installations",
        "calendar_object_creations", "calendar_dates_inspected",
        "protected_or_private_reads", "t0_run",
    }
)


class V10AValidationError(ContractValidationError):
    """Fail-closed V10A validation/publication error."""


@dataclass(frozen=True)
class V10AValidationConfig:
    repo_root: Path
    expected_current_head: str
    expected_live_validation_runner_commit_sha: str
    expected_live_validation_runner_blob_sha1: str
    wheelhouse: Path
    step4_attempt_root: Path
    output_root: Path

    @property
    def canonical_environment(self) -> Path:
        return self.repo_root / CANONICAL_ENVIRONMENT_RELATIVE

    @property
    def canonical_interpreter(self) -> Path:
        return self.repo_root / CANONICAL_INTERPRETER_RELATIVE

    @property
    def effective_wheelhouse(self) -> Path:
        return self.wheelhouse


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _strict_sha(value: Any, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise V10AValidationError(f"{label}_INVALID")


def _strict_int(value: Any, label: str, *, allow_none: bool = True) -> int | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise V10AValidationError(f"{label}_INVALID")
    return value


def _strict_bool(value: Any, label: str, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, bool):
        raise V10AValidationError(f"{label}_INVALID")


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


def _output_root_safe(config: V10AValidationConfig) -> bool:
    root = config.output_root
    if not root.is_absolute() or os.path.lexists(root) or _unsafe_existing_component(root):
        return False
    try:
        resolved = Path(os.path.realpath(root))
        forbidden = [
            Path(os.path.realpath(config.repo_root)),
            Path(os.path.realpath(config.canonical_environment)),
            Path(os.path.realpath(config.wheelhouse)),
            Path(os.path.realpath(config.step4_attempt_root)),
        ]
    except OSError:
        return False
    if any(_path_inside(resolved, item) or _path_inside(item, resolved) for item in forbidden):
        return False
    return _nearest_existing(root).is_dir()


def _repo_identity(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if normalized.startswith("git@github.com:"):
        normalized = normalized[len("git@github.com:") :]
    if normalized.startswith("https://github.com/"):
        normalized = normalized[len("https://github.com/") :]
    return normalized == REPOSITORY_IDENTITY


def _run_git(repo_root: Path, args: Sequence[str]) -> bytes:
    import subprocess

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
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False


def _git_blob_at(repo_root: Path, revision: str, relative: Path) -> str:
    return _run_git(repo_root, ["rev-parse", f"{revision}:{relative.as_posix()}"]).decode("ascii").strip()


def _preflight_config(config: V10AValidationConfig) -> preflight.PreflightConfig:
    return v10_live._preflight_config(config)  # type: ignore[arg-type]


def _read_step4_inputs(config: V10AValidationConfig, obs: dict[str, Any]) -> None:
    state_path = config.step4_attempt_root / "attempt_state_complete.json"
    try:
        obs["step4_state"] = json.loads(state_path.read_text(encoding="utf-8"))
        obs["stdout.bin"] = (config.step4_attempt_root / "stdout.bin").read_bytes()
        obs["stderr.bin"] = (config.step4_attempt_root / "stderr.bin").read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return


def _default_provenance_observations(config: V10AValidationConfig) -> dict[str, Any]:
    """Collect only Stage-2 repository/design/runner provenance."""
    obs: dict[str, Any] = {}
    try:
        obs.update(
            repository_identity=_run_git(config.repo_root, ["config", "--get", "remote.origin.url"]).decode().strip(),
            branch=_run_git(config.repo_root, ["branch", "--show-current"]).decode().strip(),
            head=_run_git(config.repo_root, ["rev-parse", "HEAD"]).decode().strip(),
            clean=_run_git(config.repo_root, ["status", "--porcelain", "--untracked-files=all"]) == b"",
            approved_design_commit_exists=_git_exists(config.repo_root, APPROVED_DESIGN_SHA),
            approved_design_blob_sha1=_git_blob_at(config.repo_root, APPROVED_DESIGN_SHA, FROZEN_DESIGN_RELATIVE),
            freeze_record_commit_exists=_git_exists(config.repo_root, FREEZE_RECORD_SHA),
            freeze_record_blob_sha1=_git_blob_at(config.repo_root, FREEZE_RECORD_SHA, FROZEN_DESIGN_RELATIVE),
            current_frozen_design_blob_sha1=_git_blob_at(config.repo_root, "HEAD", FROZEN_DESIGN_RELATIVE),
            current_design_matches_freeze_record=True,
            v10a_runner_commit_exists=_git_exists(config.repo_root, config.expected_live_validation_runner_commit_sha),
            reviewed_v10a_runner_blob_sha1=_git_blob_at(config.repo_root, config.expected_live_validation_runner_commit_sha, RUNNER_RELATIVE),
            current_v10a_runner_blob_sha1=_run_git(config.repo_root, ["hash-object", "--", str(config.repo_root / RUNNER_RELATIVE)]).decode("ascii").strip(),
        )
    except (ContractValidationError, OSError, UnicodeError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        return obs
    return obs


def _validate_repository_provenance(config: V10AValidationConfig, obs: Mapping[str, Any]) -> bool:
    """Validate only Stage 2; no Step-4 or wheelhouse facts are required."""
    try:
        _strict_sha(config.expected_current_head, SHA1_RE, "expected current head")
        _strict_sha(config.expected_live_validation_runner_commit_sha, SHA1_RE, "runner commit")
        _strict_sha(config.expected_live_validation_runner_blob_sha1, SHA1_RE, "runner blob")
    except V10AValidationError:
        return False
    return all(
        (
            _repo_identity(obs.get("repository_identity")),
            obs.get("branch") == AUTHORITATIVE_BRANCH,
            obs.get("head") == config.expected_current_head,
            obs.get("clean") is True,
            obs.get("approved_design_commit_exists") is True,
            obs.get("approved_design_blob_sha1") == APPROVED_DESIGN_BLOB_SHA1,
            obs.get("freeze_record_commit_exists") is True,
            obs.get("freeze_record_blob_sha1") == FREEZE_RECORD_BLOB_SHA1,
            obs.get("current_frozen_design_blob_sha1") == FREEZE_RECORD_BLOB_SHA1,
            obs.get("current_design_matches_freeze_record") is True,
            obs.get("v10a_runner_commit_exists") is True,
            obs.get("reviewed_v10a_runner_blob_sha1") == config.expected_live_validation_runner_blob_sha1,
            obs.get("current_v10a_runner_blob_sha1") == config.expected_live_validation_runner_blob_sha1,
            _output_root_safe(config),
        )
    )


def _default_historical_provenance_observations(config: V10AValidationConfig, obs: Mapping[str, Any]) -> dict[str, Any]:
    """Collect Stage-3 Step-4, generic-lock, and reviewed-wheelhouse facts."""
    result = dict(obs)
    try:
        _read_step4_inputs(config, result)
        pc = _preflight_config(config)
        historical = preflight._default_provenance_observations(pc)
        result.update(historical)
        validated = preflight._validate_provenance(pc, result)
        if validated is None:
            return result
        lock_obs = {**result, **preflight._default_successor_lock_observations(pc)}
        if not preflight._validate_successor_lock(pc, lock_obs, validated["candidate"]):
            return result
        result.update(lock_obs)
        wheel, _ = preflight._derive_wheelhouse(pc, validated["candidate"])
        delta_paths = tuple(Path(path) for path in wheel["delta_wheel_paths"])
        result["historical_step4_provenance_valid"] = v10_live._validate_step4_evidence(  # type: ignore[arg-type]
            config, result, validated["candidate"], delta_paths
        )
        result["reviewed_wheelhouse_provenance_valid"] = bool(wheel.get("wheelhouse_integrity_verified") and wheel.get("ok"))
    except (ContractValidationError, OSError, UnicodeError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
        return result
    return result


def _validate_historical_provenance(obs: Mapping[str, Any]) -> bool:
    return obs.get("historical_step4_provenance_valid") is True and obs.get("reviewed_wheelhouse_provenance_valid") is True


def _unauthorized_observed(obs: Mapping[str, Any]) -> bool:
    if obs.get("unauthorized_operation_observed") is True or obs.get("t0_run") is True:
        return True
    for key in ("package_index_network_requests", "package_installations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads"):
        value = obs.get(key, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    return False


def _default_operation_observations() -> dict[str, Any]:
    """Initialize Stage 1 counters without reading any later-stage input."""
    return {
        "package_index_network_requests": 0,
        "package_installations": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_reads": 0,
        "t0_run": False,
    }


def _validate_provenance(config: V10AValidationConfig, obs: Mapping[str, Any]) -> bool:
    return _validate_repository_provenance(config, obs) and _validate_historical_provenance(obs)


def _normalize_packages(value: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise V10AValidationError("LIVE_PACKAGE_SET_MISMATCH")
    result: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "version"}:
            raise V10AValidationError("LIVE_PACKAGE_SET_MISMATCH")
        name, version = item["name"], item["version"]
        if not isinstance(name, str) or not isinstance(version, str) or not version:
            raise V10AValidationError("LIVE_PACKAGE_SET_MISMATCH")
        try:
            normalized = normalize_distribution_name(name)
        except ContractValidationError as error:
            raise V10AValidationError("LIVE_PACKAGE_SET_MISMATCH") from error
        if not normalized or normalized in names:
            raise V10AValidationError("LIVE_PACKAGE_SET_MISMATCH")
        names.add(normalized)
        result.append((normalized, version))
    return tuple(sorted(result))


def _packages_json(value: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"name": name, "version": version} for name, version in value]


def _default_package_observations(config: V10AValidationConfig) -> dict[str, Any]:
    """Collect only Stage-4 installed distribution metadata."""
    observations: dict[str, Any] = {
        "package_index_network_requests": 0,
        "package_installations": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_reads": 0,
        "t0_run": False,
    }
    try:
        distributions = []
        for dist in importlib.metadata.distributions():
            distributions.append({"name": dist.metadata.get("Name"), "version": dist.version})
        observations["observed_packages"] = _packages_json(_normalize_packages(distributions))
    except (OSError, ImportError, KeyError, TypeError, ValueError):
        observations.setdefault("observed_packages", None)
    return observations


def _default_platform_observations(config: V10AValidationConfig) -> dict[str, Any]:
    """Collect only Stage-5 interpreter/platform observations."""
    return {
        "interpreter_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "sysconfig_platform": sysconfig.get_platform(),
    }


def _default_package_version_observations(config: V10AValidationConfig) -> dict[str, Any]:
    """Collect only Stage-6 package-version observations."""
    result: dict[str, Any] = {}
    try:
        result["pandas_market_calendars_version"] = importlib.metadata.version("pandas-market-calendars")
        result["exchange_calendars_version"] = importlib.metadata.version("exchange-calendars")
    except (OSError, ImportError, KeyError, TypeError, ValueError):
        pass
    return result


def _default_installed_source_paths(config: V10AValidationConfig) -> dict[str, Any]:
    """Resolve Stage-9 installed source paths after ZIP uniqueness passes."""
    try:
        pmc = importlib.metadata.distribution("pandas-market-calendars")
        return {
            "installed_jpx_path": str(pmc.locate_file(JPX_ENTRY)),
            "installed_jp_path": str(pmc.locate_file(JP_ENTRY)),
        }
    except (OSError, ImportError, KeyError, TypeError, ValueError):
        return {}


def _default_live_observations(config: V10AValidationConfig) -> dict[str, Any]:
    """Backward-compatible Stage-4-only alias; production uses granular gates."""
    return _default_package_observations(config)


def _default_xls_probe() -> str:
    from scripts.check_real_execution_env import check_jpx_xls_parser_synthetic_probe
    return "PASS" if check_jpx_xls_parser_synthetic_probe().get("status") == "PASS" else "FAIL"


def _default_pdf_probe() -> str:
    from scripts.check_real_execution_env import check_pdf_parser_synthetic_probe
    return "PASS" if check_pdf_parser_synthetic_probe().get("status") == "PASS" else "FAIL"



def _official_wheel_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode)


def _wheel_path(config: V10AValidationConfig, obs: Mapping[str, Any]) -> Path:
    wheel_value = obs.get("official_wheel_path") or str(config.effective_wheelhouse / OFFICIAL_WHEEL_FILENAME)
    return Path(wheel_value)


def _validate_official_wheel_identity(config: V10AValidationConfig, obs: Mapping[str, Any], phase: dict[str, Any]) -> str:
    """Run Stage 7; do not open the ZIP on an archive-hash mismatch."""
    wheel_path = _wheel_path(config, obs)
    if wheel_path.name != OFFICIAL_WHEEL_FILENAME or not _regular_file(wheel_path):
        return "OFFICIAL_WHEEL_IDENTITY_MISMATCH"
    try:
        observed_sha = _official_wheel_sha256(wheel_path)
    except OSError:
        return "OFFICIAL_WHEEL_IDENTITY_MISMATCH"
    phase["official_wheel_filename"] = OFFICIAL_WHEEL_FILENAME
    phase["observed_official_wheel_sha256"] = observed_sha
    phase["official_wheel_sha256_match"] = observed_sha == OFFICIAL_WHEEL_SHA256
    if not phase["official_wheel_sha256_match"]:
        return "OFFICIAL_WHEEL_IDENTITY_MISMATCH"
    phase["_wheel_path"] = wheel_path
    return "NONE"


def _zip_entry_name(info: zipfile.ZipInfo) -> str:
    # ZipInfo.filename can present a backslash-bearing central-directory name
    # with a normalized slash view.  orig_filename retains the exact decoded
    # spelling that must be compared without any normalization here.
    return info.orig_filename


def _enumerate_unique_source_entries(phase: dict[str, Any]) -> str:
    """Run Stage 8 and retain the unique ZipInfo objects for Stage 9."""
    wheel_path = phase.get("_wheel_path")
    if not isinstance(wheel_path, Path):
        return "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE"
    try:
        archive = zipfile.ZipFile(wheel_path, "r")
        try:
            infos = archive.infolist()
            # ``ZipInfo.filename`` may expose a platform-normalized view of
            # backslash-bearing names.  ``orig_filename`` preserves the
            # decoded central-directory spelling, which is the identity that
            # must be compared without slash/backslash normalization.
            jpx_infos = [info for info in infos if _zip_entry_name(info) == JPX_ENTRY]
            jp_infos = [info for info in infos if _zip_entry_name(info) == JP_ENTRY]
            phase["jpx_entry_occurrence_count"] = len(jpx_infos)
            phase["jp_entry_occurrence_count"] = len(jp_infos)
            if len(jpx_infos) != 1 or len(jp_infos) != 1:
                archive.close()
                return "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE"
            phase["_wheel_archive"] = archive
            phase["_source_infos"] = (jpx_infos[0], jp_infos[0])
            return "NONE"
        except Exception:
            archive.close()
            raise
    except (OSError, RuntimeError, zipfile.BadZipFile, KeyError, ValueError, AttributeError, UnicodeError):
        return "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE"


def _close_wheel_archive(phase: dict[str, Any]) -> None:
    archive = phase.pop("_wheel_archive", None)
    phase.pop("_source_infos", None)
    phase.pop("_wheel_path", None)
    if isinstance(archive, zipfile.ZipFile):
        archive.close()


def _validate_source_entries(config: V10AValidationConfig, obs: Mapping[str, Any], phase: dict[str, Any]) -> str:
    """Run Stage 9; callers must complete Stages 7-8 first."""
    archive = phase.get("_wheel_archive")
    source_infos = phase.get("_source_infos")
    if not isinstance(archive, zipfile.ZipFile) or not isinstance(source_infos, tuple) or len(source_infos) != 2:
        return "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE"
    try:
        jpx_wheel_bytes = archive.read(source_infos[0])
        jp_wheel_bytes = archive.read(source_infos[1])
    except (OSError, RuntimeError, KeyError, ValueError, zipfile.BadZipFile):
        return "WHEEL_SOURCE_ENTRY_UNIQUENESS_FAILURE"
    jpx_path = Path(str(obs.get("installed_jpx_path", "")))
    jp_path = Path(str(obs.get("installed_jp_path", "")))
    try:
        jpx_installed = jpx_path.read_bytes()
    except OSError:
        phase["jpx_installed_equals_wheel_entry"] = False
        return "JPX_INSTALLED_WHEEL_BYTES_MISMATCH"
    phase["jpx_installed_equals_wheel_entry"] = jpx_installed == jpx_wheel_bytes
    if not phase["jpx_installed_equals_wheel_entry"]:
        return "JPX_INSTALLED_WHEEL_BYTES_MISMATCH"
    jpx_wheel_blob = git_blob_sha1(jpx_wheel_bytes)
    jpx_installed_blob = git_blob_sha1(jpx_installed)
    phase["jpx_wheel_git_blob_sha1"] = jpx_wheel_blob
    phase["jpx_installed_git_blob_sha1"] = jpx_installed_blob
    phase["jpx_source_blob_match"] = jpx_wheel_blob == JPX_RELEASE_GIT_BLOB_SHA1 and jpx_installed_blob == JPX_RELEASE_GIT_BLOB_SHA1
    if not phase["jpx_source_blob_match"]:
        return "JPX_RELEASE_BLOB_MISMATCH"
    try:
        jp_installed = jp_path.read_bytes()
    except OSError:
        phase["jp_installed_equals_wheel_entry"] = False
        return "HOLIDAY_INSTALLED_WHEEL_BYTES_MISMATCH"
    phase["jp_installed_equals_wheel_entry"] = jp_installed == jp_wheel_bytes
    if not phase["jp_installed_equals_wheel_entry"]:
        return "HOLIDAY_INSTALLED_WHEEL_BYTES_MISMATCH"
    jp_wheel_blob = git_blob_sha1(jp_wheel_bytes)
    jp_installed_blob = git_blob_sha1(jp_installed)
    phase["jp_wheel_git_blob_sha1"] = jp_wheel_blob
    phase["jp_installed_git_blob_sha1"] = jp_installed_blob
    phase["holiday_source_blob_match"] = jp_wheel_blob == JP_RELEASE_GIT_BLOB_SHA1 and jp_installed_blob == JP_RELEASE_GIT_BLOB_SHA1
    if not phase["holiday_source_blob_match"]:
        return "HOLIDAY_RELEASE_BLOB_MISMATCH"
    return "NONE"


def _base_evidence(failure_code: str, phase: Mapping[str, Any] | None = None) -> dict[str, Any]:
    phase = phase or {}
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "artifact_status": EVIDENCE_STATUS,
        "status": "PASS" if failure_code == "NONE" else "FAIL",
        "failure_code": failure_code,
        "approved_design_sha": APPROVED_DESIGN_SHA,
        "freeze_record_sha": FREEZE_RECORD_SHA,
        "official_wheel_filename": phase.get("official_wheel_filename"),
        "observed_official_wheel_sha256": phase.get("observed_official_wheel_sha256"),
        "official_wheel_sha256_match": phase.get("official_wheel_sha256_match"),
        "jpx_entry_occurrence_count": phase.get("jpx_entry_occurrence_count"),
        "jp_entry_occurrence_count": phase.get("jp_entry_occurrence_count"),
        "jpx_installed_equals_wheel_entry": phase.get("jpx_installed_equals_wheel_entry"),
        "jp_installed_equals_wheel_entry": phase.get("jp_installed_equals_wheel_entry"),
        "jpx_wheel_git_blob_sha1": phase.get("jpx_wheel_git_blob_sha1"),
        "jp_wheel_git_blob_sha1": phase.get("jp_wheel_git_blob_sha1"),
        "jpx_installed_git_blob_sha1": phase.get("jpx_installed_git_blob_sha1"),
        "jp_installed_git_blob_sha1": phase.get("jp_installed_git_blob_sha1"),
        "jpx_source_blob_match": phase.get("jpx_source_blob_match"),
        "holiday_source_blob_match": phase.get("holiday_source_blob_match"),
        "observed_packages": phase.get("observed_packages"),
        "observed_package_count": phase.get("observed_package_count"),
        "python_version": phase.get("python_version"),
        "platform_system": phase.get("platform_system"),
        "platform_machine": phase.get("platform_machine"),
        "sysconfig_platform": phase.get("sysconfig_platform"),
        "pandas_market_calendars_version": phase.get("pandas_market_calendars_version"),
        "exchange_calendars_version": phase.get("exchange_calendars_version"),
        "xls_probe_status": phase.get("xls_probe_status", "NOT_CHECKED"),
        "pdf_probe_status": phase.get("pdf_probe_status", "NOT_CHECKED"),
        "historical_step4_provenance_verified": phase.get("historical_step4_provenance_verified"),
        "reviewed_wheelhouse_provenance_verified": phase.get("reviewed_wheelhouse_provenance_verified"),
        "package_index_network_requests": phase.get("package_index_network_requests", 0),
        "package_installations": phase.get("package_installations", 0),
        "calendar_object_creations": phase.get("calendar_object_creations", 0),
        "calendar_dates_inspected": phase.get("calendar_dates_inspected", 0),
        "protected_or_private_reads": phase.get("protected_or_private_reads", 0),
        "t0_run": phase.get("t0_run", False),
    }


def validate_evidence(evidence: Mapping[str, Any]) -> None:
    if not isinstance(evidence, dict) or set(evidence) != set(EVIDENCE_KEYS):
        raise V10AValidationError("EVIDENCE_SCHEMA_INVALID")
    if evidence["schema_version"] != EVIDENCE_SCHEMA or evidence["artifact_status"] != EVIDENCE_STATUS:
        raise V10AValidationError("EVIDENCE_SCHEMA_INVALID")
    if evidence["status"] not in {"PASS", "FAIL"} or evidence["failure_code"] not in FAILURE_CODES:
        raise V10AValidationError("EVIDENCE_DOMAIN_INVALID")
    if (evidence["status"] == "PASS") != (evidence["failure_code"] == "NONE"):
        raise V10AValidationError("EVIDENCE_STATUS_INVALID")
    _strict_sha(evidence["approved_design_sha"], SHA1_RE, "approved design")
    _strict_sha(evidence["freeze_record_sha"], SHA1_RE, "freeze record")
    if evidence["approved_design_sha"] != APPROVED_DESIGN_SHA or evidence["freeze_record_sha"] != FREEZE_RECORD_SHA:
        raise V10AValidationError("EVIDENCE_PROVENANCE_INVALID")
    if evidence["official_wheel_filename"] is not None and evidence["official_wheel_filename"] != OFFICIAL_WHEEL_FILENAME:
        raise V10AValidationError("EVIDENCE_WHEEL_INVALID")
    if evidence["observed_official_wheel_sha256"] is not None:
        _strict_sha(evidence["observed_official_wheel_sha256"], SHA256_RE, "official wheel")
    _strict_bool(evidence["official_wheel_sha256_match"], "official wheel match")
    for key in ("jpx_entry_occurrence_count", "jp_entry_occurrence_count", "observed_package_count"):
        _strict_int(evidence[key], key)
    for key in ("jpx_installed_equals_wheel_entry", "jp_installed_equals_wheel_entry", "jpx_source_blob_match", "holiday_source_blob_match", "historical_step4_provenance_verified", "reviewed_wheelhouse_provenance_verified"):
        _strict_bool(evidence[key], key)
    for key in ("jpx_wheel_git_blob_sha1", "jp_wheel_git_blob_sha1", "jpx_installed_git_blob_sha1", "jp_installed_git_blob_sha1"):
        if evidence[key] is not None:
            _strict_sha(evidence[key], SHA1_RE, key)
    for key in ("python_version", "platform_system", "platform_machine", "sysconfig_platform", "pandas_market_calendars_version", "exchange_calendars_version"):
        if evidence[key] is not None and not isinstance(evidence[key], str):
            raise V10AValidationError("EVIDENCE_STRING_INVALID")
    if evidence["xls_probe_status"] not in {"PASS", "FAIL", "NOT_CHECKED"} or evidence["pdf_probe_status"] not in {"PASS", "FAIL", "NOT_CHECKED"}:
        raise V10AValidationError("EVIDENCE_PROBE_INVALID")
    for key in ("package_index_network_requests", "package_installations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads"):
        _strict_int(evidence[key], key, allow_none=False)
    _strict_bool(evidence["t0_run"], "t0", allow_none=False)
    if evidence["observed_packages"] is not None:
        normalized = _normalize_packages(evidence["observed_packages"])
        if _packages_json(normalized) != evidence["observed_packages"] or evidence["observed_package_count"] != len(normalized):
            raise V10AValidationError("EVIDENCE_PACKAGES_INVALID")
    elif evidence["observed_package_count"] is not None:
        raise V10AValidationError("EVIDENCE_PACKAGE_COUNT_INVALID")
    if evidence["status"] == "PASS":
        if (
            evidence["official_wheel_filename"] != OFFICIAL_WHEEL_FILENAME
            or evidence["observed_official_wheel_sha256"] != OFFICIAL_WHEEL_SHA256
            or evidence["official_wheel_sha256_match"] is not True
            or evidence["jpx_entry_occurrence_count"] != 1
            or evidence["jp_entry_occurrence_count"] != 1
            or evidence["jpx_installed_equals_wheel_entry"] is not True
            or evidence["jp_installed_equals_wheel_entry"] is not True
            or evidence["jpx_wheel_git_blob_sha1"] != JPX_RELEASE_GIT_BLOB_SHA1
            or evidence["jp_wheel_git_blob_sha1"] != JP_RELEASE_GIT_BLOB_SHA1
            or evidence["jpx_installed_git_blob_sha1"] != JPX_RELEASE_GIT_BLOB_SHA1
            or evidence["jp_installed_git_blob_sha1"] != JP_RELEASE_GIT_BLOB_SHA1
            or evidence["jpx_source_blob_match"] is not True
            or evidence["holiday_source_blob_match"] is not True
            or evidence["historical_step4_provenance_verified"] is not True
            or evidence["reviewed_wheelhouse_provenance_verified"] is not True
            or evidence["observed_packages"] is None
            or _normalize_packages(evidence["observed_packages"]) != EXPECTED_SUCCESSOR_PACKAGES
            or evidence["observed_package_count"] != len(EXPECTED_SUCCESSOR_PACKAGES)
            or evidence["python_version"] != "3.12.10"
            or evidence["platform_system"] != "Windows"
            or evidence["platform_machine"] != "AMD64"
            or evidence["sysconfig_platform"] != "win-amd64"
            or evidence["pandas_market_calendars_version"] != PMC_VERSION
            or evidence["exchange_calendars_version"] != EXCHANGE_CALENDARS_VERSION
            or evidence["xls_probe_status"] != "PASS"
            or evidence["pdf_probe_status"] != "PASS"
        ):
            raise V10AValidationError("EVIDENCE_PASS_INVALID")
        if any(evidence[key] != 0 for key in ("package_index_network_requests", "package_installations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads")) or evidence["t0_run"] is not False:
            raise V10AValidationError("EVIDENCE_PASS_INVALID")


def _publish(config: V10AValidationConfig, evidence: Mapping[str, Any]) -> Path:
    if not _output_root_safe(config):
        raise V10AValidationError("PROVENANCE_BINDING_FAILURE")
    try:
        config.output_root.mkdir(parents=True, exist_ok=False)
        target = config.output_root / EVIDENCE_NAME
        fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
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
        raise V10AValidationError("PROVENANCE_BINDING_FAILURE") from error


def _result(config: V10AValidationConfig, evidence: dict[str, Any], *, publish: bool) -> dict[str, Any]:
    validate_evidence(evidence)
    artifact_path = _publish(config, evidence) if publish else None
    return {
        "evidence": evidence,
        "artifact_path": artifact_path,
        "status": evidence["status"],
        "failure_code": evidence["failure_code"],
        "canonical_environment_ready": False,
        "environment_frozen": False,
        "execution_authorized": False,
    }


def _phase_from_observations(obs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: obs[key]
        for key in (
            "package_index_network_requests", "package_installations", "calendar_object_creations",
            "calendar_dates_inspected", "protected_or_private_reads", "t0_run",
        )
        if key in obs
    }


def run_validation(config: V10AValidationConfig, observations: Mapping[str, Any] | None = None, *, publish: bool = True) -> dict[str, Any]:
    """Run the frozen V10A order; injected observations are test-only."""
    injected = observations is not None
    if injected:
        obs = dict(observations)
    else:
        obs = _default_operation_observations()
    if _unauthorized_observed(obs):
        return _result(config, _base_evidence("UNAUTHORIZED_OPERATION_OBSERVED", _phase_from_observations(obs)), publish=publish)

    # Stage 2 is repository/design/runner provenance only.  Stage 3 is not
    # entered until this validation succeeds, so it cannot read Step-4 or the
    # wheelhouse after a base provenance failure.
    if injected:
        if not _validate_provenance(config, obs):
            return _result(config, _base_evidence("PROVENANCE_BINDING_FAILURE"), publish=publish)
    else:
        obs.update(_default_provenance_observations(config))
        if not _validate_repository_provenance(config, obs):
            return _result(config, _base_evidence("PROVENANCE_BINDING_FAILURE"), publish=publish)
        obs = _default_historical_provenance_observations(config, obs)
        if not _validate_historical_provenance(obs):
            return _result(config, _base_evidence("PROVENANCE_BINDING_FAILURE"), publish=publish)

    phase: dict[str, Any] = {
        **_phase_from_observations(obs),
        "historical_step4_provenance_verified": obs.get("historical_step4_provenance_valid") if not injected else True,
        "reviewed_wheelhouse_provenance_verified": obs.get("reviewed_wheelhouse_provenance_valid") if not injected else True,
    }

    # Stage 4: package set only.
    if not injected:
        obs.update(_default_package_observations(config))
    try:
        normalized = _normalize_packages(obs.get("observed_packages"))
    except (V10AValidationError, TypeError, ValueError):
        return _result(config, _base_evidence("LIVE_PACKAGE_SET_MISMATCH", phase), publish=publish)
    phase["observed_packages"] = _packages_json(normalized)
    phase["observed_package_count"] = len(normalized)
    if normalized != EXPECTED_SUCCESSOR_PACKAGES:
        return _result(config, _base_evidence("LIVE_PACKAGE_SET_MISMATCH", phase), publish=publish)

    # Stage 5: interpreter/platform only.
    if not injected:
        obs.update(_default_platform_observations(config))
    phase.update({key: obs.get(key) for key in ("python_version", "platform_system", "platform_machine", "sysconfig_platform")})
    if (
        obs.get("interpreter_executable") != str(config.canonical_interpreter.resolve())
        or (phase["python_version"], phase["platform_system"], phase["platform_machine"], phase["sysconfig_platform"])
        != ("3.12.10", "Windows", "AMD64", "win-amd64")
    ):
        return _result(config, _base_evidence("PYTHON_PLATFORM_MISMATCH", phase), publish=publish)

    # Stage 6: package-specific versions only.
    if not injected:
        obs.update(_default_package_version_observations(config))
    phase["pandas_market_calendars_version"] = obs.get("pandas_market_calendars_version")
    if phase["pandas_market_calendars_version"] != PMC_VERSION:
        return _result(config, _base_evidence("PMC_VERSION_MISMATCH", phase), publish=publish)
    phase["exchange_calendars_version"] = obs.get("exchange_calendars_version")
    if phase["exchange_calendars_version"] != EXCHANGE_CALENDARS_VERSION:
        return _result(config, _base_evidence("EXCHANGE_CALENDARS_VERSION_MISMATCH", phase), publish=publish)

    # Stage 7: verify the archive bytes before opening the ZIP.
    source_failure = _validate_official_wheel_identity(config, obs, phase)
    if source_failure != "NONE":
        return _result(config, _base_evidence(source_failure, phase), publish=publish)

    # Stage 8: enumerate exact central-directory names.  No source bytes or
    # installed paths are read until both exact names occur once.
    source_failure = _enumerate_unique_source_entries(phase)
    if source_failure != "NONE":
        return _result(config, _base_evidence(source_failure, phase), publish=publish)

    try:
        # Stage 9: source paths are resolved only after Stage 8 succeeds.
        if not injected:
            obs.update(_default_installed_source_paths(config))
        source_failure = _validate_source_entries(config, obs, phase)
    finally:
        _close_wheel_archive(phase)
    if source_failure != "NONE":
        return _result(config, _base_evidence(source_failure, phase), publish=publish)

    # Stage 10: invoke and check XLS before making the PDF probe callable.
    if not injected:
        obs["xls_probe_status"] = _default_xls_probe()
    phase["xls_probe_status"] = obs.get("xls_probe_status", "FAIL")
    if phase["xls_probe_status"] != "PASS":
        return _result(config, _base_evidence("XLS_PROBE_FAILURE", phase), publish=publish)

    # Stage 11: PDF is reachable only after XLS passes.
    if not injected:
        obs["pdf_probe_status"] = _default_pdf_probe()
    phase["pdf_probe_status"] = obs.get("pdf_probe_status", "FAIL")
    if phase["pdf_probe_status"] != "PASS":
        return _result(config, _base_evidence("PDF_PROBE_FAILURE", phase), publish=publish)
    return _result(config, _base_evidence("NONE", phase), publish=publish)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-current-head", required=True)
    parser.add_argument("--expected-live-validation-runner-commit-sha", required=True)
    parser.add_argument("--expected-live-validation-runner-blob-sha1", required=True)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--step4-attempt-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def _config_from_args(args: argparse.Namespace) -> V10AValidationConfig:
    for value, pattern, label in (
        (args.expected_current_head, SHA1_RE, "expected current head"),
        (args.expected_live_validation_runner_commit_sha, SHA1_RE, "runner commit"),
        (args.expected_live_validation_runner_blob_sha1, SHA1_RE, "runner blob"),
    ):
        _strict_sha(value, pattern, label)
    paths = {name: Path(getattr(args, name)) for name in ("repo_root", "wheelhouse", "step4_attempt_root", "output_root")}
    if any(not path.is_absolute() for path in paths.values()):
        raise V10AValidationError("PATH_MUST_BE_ABSOLUTE")
    return V10AValidationConfig(
        repo_root=paths["repo_root"],
        expected_current_head=args.expected_current_head,
        expected_live_validation_runner_commit_sha=args.expected_live_validation_runner_commit_sha,
        expected_live_validation_runner_blob_sha1=args.expected_live_validation_runner_blob_sha1,
        wheelhouse=paths["wheelhouse"],
        step4_attempt_root=paths["step4_attempt_root"],
        output_root=paths["output_root"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _config_from_args(_build_parser().parse_args(argv))
        result = run_validation(config, observations=None, publish=True)
    except (ContractValidationError, OSError, TypeError, ValueError):
        result = {
            "status": "FAIL",
            "failure_code": "PROVENANCE_BINDING_FAILURE",
            "canonical_environment_ready": False,
            "environment_frozen": False,
            "execution_authorized": False,
        }
    print(json.dumps({key: result.get(key) for key in ("status", "failure_code", "canonical_environment_ready", "environment_frozen", "execution_authorized")}, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
