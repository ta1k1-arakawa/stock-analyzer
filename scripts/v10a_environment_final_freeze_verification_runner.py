"""Future P3 final V10A environment verification runner.

This module is intentionally a thin, fail-closed wrapper around the reviewed
V10A no-network validator.  Its production CLI has no synthetic observation,
process, interpreter, authority, or bypass options.  The optional
``observations`` argument is an internal synthetic-test seam only.

The runner performs repository/candidate provenance first, then delegates the
read-only live validation with ``observations=None`` and ``publish=False``.
It never promotes or freezes the environment.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts import v10a_environment_no_network_validation_runner as v10a
except ModuleNotFoundError:  # direct ``python scripts/<runner>.py`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import v10a_environment_no_network_validation_runner as v10a


REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
FINAL_RUNNER_RELATIVE = Path("scripts/v10a_environment_final_freeze_verification_runner.py")
FINAL_TEST_RELATIVE = Path("tests/test_v10a_environment_final_freeze_verification_runner.py")
CANDIDATE_RELATIVE = Path("V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_RECORD_CANDIDATE.json")
PROMOTION_DESIGN_RELATIVE = Path("V10A_CANONICAL_ENVIRONMENT_PROMOTION_AND_FINAL_FREEZE_DESIGN.md")
ATTEMPT1_EVIDENCE_RELATIVE = Path("V10A_CANONICAL_ENVIRONMENT_NO_NETWORK_VALIDATION_EVIDENCE.json")
ATTEMPT1_ADJUDICATION_RELATIVE = Path("V10A_NO_NETWORK_VALIDATION_ATTEMPT_1_ADJUDICATION.json")

APPROVED_DESIGN_SHA = "b14cc5510685210e928000af0815e188bc1aadc0"
APPROVED_DESIGN_BLOB_SHA1 = "3217b155c7d226f8f6edbcba2162c74e8d9d4e0e"
FREEZE_RECORD_SHA = "86ceda3dee531b08afa5db4df7af1298ca770fad"
FREEZE_RECORD_BLOB_SHA1 = "a3f913857966cb0593f3218d882c4f91b2bc1f2f"
PROMOTION_DESIGN_SHA = "f312fdab7f98ab1a6ce9be7b9f9a494f2548c502"
PROMOTION_DESIGN_BLOB_SHA1 = "25962ef5d4b0feab102668614dbca22d39cb34b5"
REVIEWED_ATTEMPT1_RUNNER_SHA = "4b6c89915fe5016301ac7635258dfe8d2f452bcd"
REVIEWED_ATTEMPT1_RUNNER_BLOB_SHA1 = "9bf27e9c22d0108a82304f7e8f43eb1f55c45f05"
ATTEMPT1_EVIDENCE_SHA256 = "bcef3587f8a86c889306a4808e3cf033b12e16fef45009788ded444983c77d1e"
ATTEMPT1_EVIDENCE_BLOB_SHA1 = "0a8e3afd914a905887f58d1f23a224ce5138c3ac"
ATTEMPT1_ADJUDICATION_BLOB_SHA1 = "fbc0357d211bfa6f3450f909d6044c2e6f25ac19"
FINAL_EVIDENCE_NAME = "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE.json"
FINAL_EVIDENCE_SCHEMA = "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE_V1"
FINAL_EVIDENCE_STATUS = "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_EVIDENCE"

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_DELTA = v10a.EXPECTED_DELTA
EXPECTED_PACKAGES = v10a.EXPECTED_SUCCESSOR_PACKAGES

FAILURE_CODES = frozenset((*v10a.FAILURE_CODES, "NONE"))

FINAL_EVIDENCE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_status",
        "status",
        "failure_code",
        "expected_p2_reviewed_sha",
        "promotion_design_sha",
        "promotion_design_blob_sha1",
        "candidate_git_blob_sha1",
        "candidate_sha256",
        "final_verification_runner_blob_sha1",
        "final_verification_test_blob_sha1",
        "approved_design_sha",
        "freeze_record_sha",
        "reviewed_attempt1_runner_sha",
        "reviewed_attempt1_runner_blob_sha1",
        "attempt1_evidence_repo_path",
        "attempt1_evidence_sha256",
        "attempt1_evidence_git_blob_sha1",
        "attempt1_adjudication_git_blob_sha1",
        "observed_packages",
        "observed_package_count",
        "python_version",
        "platform_system",
        "platform_machine",
        "sysconfig_platform",
        "pandas_market_calendars_version",
        "exchange_calendars_version",
        "official_wheel_filename",
        "observed_official_wheel_sha256",
        "official_wheel_sha256_match",
        "jpx_entry_occurrence_count",
        "jp_entry_occurrence_count",
        "jpx_installed_equals_wheel_entry",
        "jp_installed_equals_wheel_entry",
        "jpx_wheel_git_blob_sha1",
        "jp_wheel_git_blob_sha1",
        "jpx_installed_git_blob_sha1",
        "jp_installed_git_blob_sha1",
        "jpx_source_blob_match",
        "holiday_source_blob_match",
        "xls_probe_status",
        "pdf_probe_status",
        "historical_step4_provenance_verified",
        "reviewed_wheelhouse_provenance_verified",
        "package_index_network_requests",
        "package_installations",
        "environment_mutations",
        "calendar_object_creations",
        "calendar_dates_inspected",
        "protected_or_private_reads",
        "t0_run",
        "canonical_environment_promoted",
        "environment_frozen",
        "execution_authorized",
        "calendar_generation_authorized",
        "t0_authorized",
        "historical_evaluation_authorized",
        "future_profitability_established",
    }
)


class FinalFreezeValidationError(ValueError):
    """Fail-closed final-verification error."""


@dataclass(frozen=True)
class FinalFreezeConfig:
    repo_root: Path
    expected_p2_reviewed_sha: str
    expected_final_runner_blob_sha1: str
    expected_candidate_blob_sha1: str
    expected_candidate_sha256: str
    wheelhouse: Path
    step4_attempt_root: Path
    output_root: Path

    @property
    def candidate_path(self) -> Path:
        return self.repo_root / CANDIDATE_RELATIVE

    @property
    def attempt1_evidence_path(self) -> Path:
        return self.repo_root / ATTEMPT1_EVIDENCE_RELATIVE

    @property
    def attempt1_adjudication_path(self) -> Path:
        return self.repo_root / ATTEMPT1_ADJUDICATION_RELATIVE


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def _strict_sha(value: Any, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise FinalFreezeValidationError(f"{label}_INVALID")


def _strict_int(value: Any, label: str, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise FinalFreezeValidationError(f"{label}_INVALID")


def _strict_bool(value: Any, label: str, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if not isinstance(value, bool):
        raise FinalFreezeValidationError(f"{label}_INVALID")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


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
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    ).stdout


def _git_blob_at(repo_root: Path, revision: str, relative: Path) -> str:
    return _run_git(repo_root, ["rev-parse", f"{revision}:{relative.as_posix()}"]).decode("ascii").strip()


def _git_exists(repo_root: Path, revision: str, kind: str = "commit") -> bool:
    try:
        _run_git(repo_root, ["cat-file", "-e", f"{revision}^{{{kind}}}"])
        return True
    except (OSError, UnicodeError, ValueError, subprocess.CalledProcessError):
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


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _output_root_safe(config: FinalFreezeConfig) -> bool:
    root = config.output_root
    if not root.is_absolute() or os.path.lexists(root) or _unsafe_existing_component(root):
        return False
    try:
        resolved = Path(os.path.realpath(root))
        forbidden = [
            Path(os.path.realpath(config.repo_root)),
            Path(os.path.realpath(config.repo_root / v10a.CANONICAL_ENVIRONMENT_RELATIVE)),
            Path(os.path.realpath(config.wheelhouse)),
            Path(os.path.realpath(config.step4_attempt_root)),
        ]
    except OSError:
        return False
    if any(_path_inside(resolved, item) or _path_inside(item, resolved) for item in forbidden):
        return False
    return _nearest_existing(root).is_dir()


def _candidate_expected_packages() -> list[dict[str, str]]:
    return [{"name": name, "version": version} for name, version in EXPECTED_PACKAGES]


def candidate_template(*, runner_blob_sha1: str, test_blob_sha1: str) -> dict[str, Any]:
    """Return the fixed P1 candidate shape used by tests and P1 tooling."""
    return {
        "schema_version": "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_RECORD_CANDIDATE_V1",
        "artifact_status": "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_RECORD_CANDIDATE",
        "study_identity": "V10A_CALENDAR_AUTHORITY_RELEASE_ARTIFACT_SUCCESSOR",
        "candidate_status": "CANDIDATE_ONLY",
        "approved_v10a_design_sha": APPROVED_DESIGN_SHA,
        "freeze_record_sha": FREEZE_RECORD_SHA,
        "promotion_design_reviewed_sha": PROMOTION_DESIGN_SHA,
        "promotion_design_blob_sha1": PROMOTION_DESIGN_BLOB_SHA1,
        "attempt1_evidence_repo_path": str(ATTEMPT1_EVIDENCE_RELATIVE).replace("\\", "/"),
        "attempt1_evidence_sha256": ATTEMPT1_EVIDENCE_SHA256,
        "attempt1_evidence_git_blob_sha1": ATTEMPT1_EVIDENCE_BLOB_SHA1,
        "attempt1_adjudication_git_blob_sha1": ATTEMPT1_ADJUDICATION_BLOB_SHA1,
        "reviewed_attempt1_runner_sha": REVIEWED_ATTEMPT1_RUNNER_SHA,
        "reviewed_attempt1_runner_blob_sha1": REVIEWED_ATTEMPT1_RUNNER_BLOB_SHA1,
        "final_verification_runner_repo_path": str(FINAL_RUNNER_RELATIVE).replace("\\", "/"),
        "final_verification_runner_git_blob_sha1": runner_blob_sha1,
        "final_verification_test_repo_path": str(FINAL_TEST_RELATIVE).replace("\\", "/"),
        "final_verification_test_git_blob_sha1": test_blob_sha1,
        "expected_packages": _candidate_expected_packages(),
        "expected_package_count": 20,
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "pandas_market_calendars_version": "5.4.0",
        "exchange_calendars_version": "4.13.2",
        "official_wheel_filename": v10a.OFFICIAL_WHEEL_FILENAME,
        "official_wheel_sha256": v10a.OFFICIAL_WHEEL_SHA256,
        "jpx_entry": v10a.JPX_ENTRY,
        "jpx_entry_occurrence_count": 1,
        "jp_entry": v10a.JP_ENTRY,
        "jp_entry_occurrence_count": 1,
        "installed_to_wheel_raw_byte_equality_required": True,
        "jpx_release_blob_sha1": v10a.JPX_RELEASE_GIT_BLOB_SHA1,
        "jp_release_blob_sha1": v10a.JP_RELEASE_GIT_BLOB_SHA1,
        "xls_probe_required_status": "PASS",
        "pdf_probe_required_status": "PASS",
        "package_index_network_requests": 0,
        "package_installations": 0,
        "environment_mutations": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_reads": 0,
        "t0_run": False,
        "p3_adjudication_required": True,
        "p3_adjudication_repo_path": "V10A_CANONICAL_ENVIRONMENT_FINAL_FREEZE_VERIFICATION_ADJUDICATION.json",
        "canonical_environment_promoted": False,
        "environment_frozen": False,
        "future_protected_execution_authorized": False,
        "execution_authorized": False,
        "calendar_generation_authorized": False,
        "t0_authorized": False,
        "historical_evaluation_authorized": False,
        "future_profitability_established": False,
    }


def validate_candidate(candidate: Mapping[str, Any], *, expected_runner_blob_sha1: str, expected_test_blob_sha1: str) -> None:
    """Validate the exact safe P1 candidate semantics."""
    if not isinstance(candidate, dict):
        raise FinalFreezeValidationError("CANDIDATE_SCHEMA_INVALID")
    expected = candidate_template(runner_blob_sha1=expected_runner_blob_sha1, test_blob_sha1=expected_test_blob_sha1)
    if set(candidate) != set(expected):
        raise FinalFreezeValidationError("CANDIDATE_SCHEMA_INVALID")
    if candidate != expected:
        raise FinalFreezeValidationError("CANDIDATE_BINDING_INVALID")
    if "candidate_git_blob_sha1" in candidate or "candidate_sha256" in candidate:
        raise FinalFreezeValidationError("CANDIDATE_SELF_REFERENCE_INVALID")
    for key in ("approved_v10a_design_sha", "freeze_record_sha", "promotion_design_reviewed_sha", "reviewed_attempt1_runner_sha"):
        _strict_sha(candidate[key], SHA1_RE, key)
    for key in ("promotion_design_blob_sha1", "attempt1_evidence_git_blob_sha1", "attempt1_adjudication_git_blob_sha1", "reviewed_attempt1_runner_blob_sha1", "final_verification_runner_git_blob_sha1", "final_verification_test_git_blob_sha1", "jpx_release_blob_sha1", "jp_release_blob_sha1"):
        _strict_sha(candidate[key], SHA1_RE, key)
    _strict_sha(candidate["attempt1_evidence_sha256"], SHA256_RE, "attempt1 evidence")
    _strict_sha(candidate["official_wheel_sha256"], SHA256_RE, "official wheel")
    _strict_int(candidate["expected_package_count"], "expected package count", allow_none=False)
    if candidate["expected_package_count"] != len(EXPECTED_PACKAGES):
        raise FinalFreezeValidationError("CANDIDATE_PACKAGE_COUNT_INVALID")
    if candidate["expected_packages"] != _candidate_expected_packages():
        raise FinalFreezeValidationError("CANDIDATE_PACKAGE_SET_INVALID")
    for key in ("jpx_entry_occurrence_count", "jp_entry_occurrence_count", "package_index_network_requests", "package_installations", "environment_mutations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads"):
        _strict_int(candidate[key], key, allow_none=False)
    for key in ("installed_to_wheel_raw_byte_equality_required", "p3_adjudication_required", "canonical_environment_promoted", "environment_frozen", "future_protected_execution_authorized", "execution_authorized", "calendar_generation_authorized", "t0_authorized", "historical_evaluation_authorized", "future_profitability_established", "t0_run"):
        _strict_bool(candidate[key], key, allow_none=False)
    if candidate["jpx_entry_occurrence_count"] != 1 or candidate["jp_entry_occurrence_count"] != 1 or candidate["p3_adjudication_required"] is not True:
        raise FinalFreezeValidationError("CANDIDATE_REQUIREMENT_INVALID")
    if any(candidate[key] != 0 for key in ("package_index_network_requests", "package_installations", "environment_mutations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads")) or candidate["t0_run"] is not False:
        raise FinalFreezeValidationError("CANDIDATE_OPERATION_BINDING_INVALID")


def _default_stage2_observations(config: FinalFreezeConfig) -> dict[str, Any]:
    """Collect only P1/P2 repository and safe-path facts."""
    obs: dict[str, Any] = {}
    try:
        candidate_raw = config.candidate_path.read_bytes()
        evidence_raw = config.attempt1_evidence_path.read_bytes()
        adjudication_raw = config.attempt1_adjudication_path.read_bytes()
        obs.update(
            repository_identity=_run_git(config.repo_root, ["config", "--get", "remote.origin.url"]).decode().strip(),
            branch=_run_git(config.repo_root, ["branch", "--show-current"]).decode().strip(),
            head=_run_git(config.repo_root, ["rev-parse", "HEAD"]).decode().strip(),
            clean=_run_git(config.repo_root, ["status", "--porcelain", "--untracked-files=all"]) == b"",
            promotion_design_commit_exists=_git_exists(config.repo_root, PROMOTION_DESIGN_SHA),
            promotion_design_blob_sha1=_git_blob_at(config.repo_root, PROMOTION_DESIGN_SHA, PROMOTION_DESIGN_RELATIVE),
            current_promotion_design_blob_sha1=_git_blob_at(config.repo_root, "HEAD", PROMOTION_DESIGN_RELATIVE),
            approved_design_commit_exists=_git_exists(config.repo_root, APPROVED_DESIGN_SHA),
            approved_design_blob_sha1=_git_blob_at(config.repo_root, APPROVED_DESIGN_SHA, v10a.FROZEN_DESIGN_RELATIVE),
            freeze_record_commit_exists=_git_exists(config.repo_root, FREEZE_RECORD_SHA),
            freeze_record_blob_sha1=_git_blob_at(config.repo_root, FREEZE_RECORD_SHA, v10a.FROZEN_DESIGN_RELATIVE),
            current_frozen_design_blob_sha1=_git_blob_at(config.repo_root, "HEAD", v10a.FROZEN_DESIGN_RELATIVE),
            reviewed_attempt1_runner_commit_exists=_git_exists(config.repo_root, REVIEWED_ATTEMPT1_RUNNER_SHA),
            reviewed_attempt1_runner_blob_sha1=_git_blob_at(config.repo_root, REVIEWED_ATTEMPT1_RUNNER_SHA, v10a.RUNNER_RELATIVE),
            current_attempt1_runner_blob_sha1=_git_blob_at(config.repo_root, "HEAD", v10a.RUNNER_RELATIVE),
            attempt1_evidence_sha256=_sha256_bytes(evidence_raw),
            attempt1_evidence_git_blob_sha1=_git_blob_at(config.repo_root, "HEAD", ATTEMPT1_EVIDENCE_RELATIVE),
            attempt1_adjudication_git_blob_sha1=_git_blob_at(config.repo_root, "HEAD", ATTEMPT1_ADJUDICATION_RELATIVE),
            final_runner_current_blob_sha1=_run_git(config.repo_root, ["hash-object", "--", str(config.repo_root / FINAL_RUNNER_RELATIVE)]).decode("ascii").strip(),
            final_test_current_blob_sha1=_run_git(config.repo_root, ["hash-object", "--", str(config.repo_root / FINAL_TEST_RELATIVE)]).decode("ascii").strip(),
            candidate_raw_sha256=_sha256_bytes(candidate_raw),
            candidate_raw_json=json.loads(candidate_raw.decode("utf-8")),
            candidate_current_blob_sha1=_git_blob_at(config.repo_root, "HEAD", CANDIDATE_RELATIVE),
            output_root_safe=_output_root_safe(config),
        )
        evidence_json = json.loads(evidence_raw.decode("utf-8"))
        obs["attempt1_evidence_package_set"] = evidence_json.get("observed_packages")
        obs["attempt1_evidence_package_count"] = evidence_json.get("observed_package_count")
        obs["attempt1_adjudication_json"] = json.loads(adjudication_raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, subprocess.CalledProcessError):
        return obs
    return obs


def _validate_stage2(config: FinalFreezeConfig, obs: Mapping[str, Any]) -> bool:
    try:
        for value, pattern, label in (
            (config.expected_p2_reviewed_sha, SHA1_RE, "expected p2 sha"),
            (config.expected_final_runner_blob_sha1, SHA1_RE, "final runner blob"),
            (config.expected_candidate_blob_sha1, SHA1_RE, "candidate blob"),
            (config.expected_candidate_sha256, SHA256_RE, "candidate sha256"),
        ):
            _strict_sha(value, pattern, label)
        candidate = obs.get("candidate_raw_json")
        validate_candidate(candidate, expected_runner_blob_sha1=config.expected_final_runner_blob_sha1, expected_test_blob_sha1=str(obs.get("final_test_current_blob_sha1", "")))
        return all(
            (
                _repo_identity(obs.get("repository_identity")),
                obs.get("branch") == AUTHORITATIVE_BRANCH,
                obs.get("head") == config.expected_p2_reviewed_sha,
                obs.get("clean") is True,
                obs.get("promotion_design_commit_exists") is True,
                obs.get("promotion_design_blob_sha1") == PROMOTION_DESIGN_BLOB_SHA1,
                obs.get("current_promotion_design_blob_sha1") == PROMOTION_DESIGN_BLOB_SHA1,
                obs.get("approved_design_commit_exists") is True,
                obs.get("approved_design_blob_sha1") == APPROVED_DESIGN_BLOB_SHA1,
                obs.get("freeze_record_commit_exists") is True,
                obs.get("freeze_record_blob_sha1") == FREEZE_RECORD_BLOB_SHA1,
                obs.get("current_frozen_design_blob_sha1") == FREEZE_RECORD_BLOB_SHA1,
                obs.get("reviewed_attempt1_runner_commit_exists") is True,
                obs.get("reviewed_attempt1_runner_blob_sha1") == REVIEWED_ATTEMPT1_RUNNER_BLOB_SHA1,
                obs.get("current_attempt1_runner_blob_sha1") == REVIEWED_ATTEMPT1_RUNNER_BLOB_SHA1,
                obs.get("attempt1_evidence_sha256") == ATTEMPT1_EVIDENCE_SHA256,
                obs.get("attempt1_evidence_git_blob_sha1") == ATTEMPT1_EVIDENCE_BLOB_SHA1,
                obs.get("attempt1_adjudication_git_blob_sha1") == ATTEMPT1_ADJUDICATION_BLOB_SHA1,
                obs.get("final_runner_current_blob_sha1") == config.expected_final_runner_blob_sha1,
                obs.get("candidate_current_blob_sha1") == config.expected_candidate_blob_sha1,
                obs.get("candidate_raw_sha256") == config.expected_candidate_sha256,
                obs.get("output_root_safe") is True,
                obs.get("attempt1_evidence_package_set") == _candidate_expected_packages(),
                obs.get("attempt1_evidence_package_count") == len(EXPECTED_PACKAGES),
            )
        )
    except (FinalFreezeValidationError, TypeError, ValueError):
        return False


def _default_operation_observations() -> dict[str, Any]:
    return {
        "package_index_network_requests": 0,
        "package_installations": 0,
        "environment_mutations": 0,
        "calendar_object_creations": 0,
        "calendar_dates_inspected": 0,
        "protected_or_private_reads": 0,
        "t0_run": False,
        "unauthorized_operation_observed": False,
    }


def _unauthorized_observed(obs: Mapping[str, Any]) -> bool:
    if obs.get("unauthorized_operation_observed") is True or obs.get("t0_run") is True:
        return True
    for key in ("package_index_network_requests", "package_installations", "environment_mutations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads"):
        value = obs.get(key, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return True
    return False


def _provenance_fields(config: FinalFreezeConfig, obs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "expected_p2_reviewed_sha": config.expected_p2_reviewed_sha,
        "promotion_design_sha": PROMOTION_DESIGN_SHA,
        "promotion_design_blob_sha1": PROMOTION_DESIGN_BLOB_SHA1,
        "candidate_git_blob_sha1": config.expected_candidate_blob_sha1,
        "candidate_sha256": config.expected_candidate_sha256,
        "final_verification_runner_blob_sha1": config.expected_final_runner_blob_sha1,
        "final_verification_test_blob_sha1": obs.get("final_test_current_blob_sha1"),
        "approved_design_sha": APPROVED_DESIGN_SHA,
        "freeze_record_sha": FREEZE_RECORD_SHA,
        "reviewed_attempt1_runner_sha": REVIEWED_ATTEMPT1_RUNNER_SHA,
        "reviewed_attempt1_runner_blob_sha1": REVIEWED_ATTEMPT1_RUNNER_BLOB_SHA1,
        "attempt1_evidence_repo_path": str(ATTEMPT1_EVIDENCE_RELATIVE).replace("\\", "/"),
        "attempt1_evidence_sha256": ATTEMPT1_EVIDENCE_SHA256,
        "attempt1_evidence_git_blob_sha1": ATTEMPT1_EVIDENCE_BLOB_SHA1,
        "attempt1_adjudication_git_blob_sha1": ATTEMPT1_ADJUDICATION_BLOB_SHA1,
    }


def _base_final_evidence(config: FinalFreezeConfig, failure_code: str, obs: Mapping[str, Any], delegated: Mapping[str, Any] | None = None) -> dict[str, Any]:
    delegated_evidence = dict((delegated or {}).get("evidence", {}))
    evidence: dict[str, Any] = {
        "schema_version": FINAL_EVIDENCE_SCHEMA,
        "artifact_status": FINAL_EVIDENCE_STATUS,
        "status": "PASS" if failure_code == "NONE" else "FAIL",
        "failure_code": failure_code,
        **_provenance_fields(config, obs),
    }
    for key in (
        "observed_packages", "observed_package_count", "python_version", "platform_system", "platform_machine", "sysconfig_platform",
        "pandas_market_calendars_version", "exchange_calendars_version", "official_wheel_filename", "observed_official_wheel_sha256",
        "official_wheel_sha256_match", "jpx_entry_occurrence_count", "jp_entry_occurrence_count", "jpx_installed_equals_wheel_entry",
        "jp_installed_equals_wheel_entry", "jpx_wheel_git_blob_sha1", "jp_wheel_git_blob_sha1", "jpx_installed_git_blob_sha1",
        "jp_installed_git_blob_sha1", "jpx_source_blob_match", "holiday_source_blob_match", "xls_probe_status", "pdf_probe_status",
        "historical_step4_provenance_verified", "reviewed_wheelhouse_provenance_verified",
    ):
        evidence[key] = delegated_evidence.get(key)
    evidence.update(
        package_index_network_requests=delegated_evidence.get("package_index_network_requests", obs.get("package_index_network_requests", 0)),
        package_installations=delegated_evidence.get("package_installations", obs.get("package_installations", 0)),
        environment_mutations=0,
        calendar_object_creations=delegated_evidence.get("calendar_object_creations", obs.get("calendar_object_creations", 0)),
        calendar_dates_inspected=delegated_evidence.get("calendar_dates_inspected", obs.get("calendar_dates_inspected", 0)),
        protected_or_private_reads=delegated_evidence.get("protected_or_private_reads", obs.get("protected_or_private_reads", 0)),
        t0_run=delegated_evidence.get("t0_run", obs.get("t0_run", False)),
        canonical_environment_promoted=False,
        environment_frozen=False,
        execution_authorized=False,
        calendar_generation_authorized=False,
        t0_authorized=False,
        historical_evaluation_authorized=False,
        future_profitability_established=False,
    )
    return evidence


def validate_final_evidence(evidence: Mapping[str, Any]) -> None:
    if not isinstance(evidence, dict) or set(evidence) != set(FINAL_EVIDENCE_KEYS):
        raise FinalFreezeValidationError("FINAL_EVIDENCE_SCHEMA_INVALID")
    if evidence["schema_version"] != FINAL_EVIDENCE_SCHEMA or evidence["artifact_status"] != FINAL_EVIDENCE_STATUS:
        raise FinalFreezeValidationError("FINAL_EVIDENCE_SCHEMA_INVALID")
    if evidence["status"] not in {"PASS", "FAIL"} or evidence["failure_code"] not in FAILURE_CODES:
        raise FinalFreezeValidationError("FINAL_EVIDENCE_DOMAIN_INVALID")
    if (evidence["status"] == "PASS") != (evidence["failure_code"] == "NONE"):
        raise FinalFreezeValidationError("FINAL_EVIDENCE_STATUS_INVALID")
    for key in ("expected_p2_reviewed_sha", "promotion_design_sha", "promotion_design_blob_sha1", "candidate_git_blob_sha1", "final_verification_runner_blob_sha1", "approved_design_sha", "freeze_record_sha", "reviewed_attempt1_runner_sha", "reviewed_attempt1_runner_blob_sha1", "attempt1_evidence_git_blob_sha1", "attempt1_adjudication_git_blob_sha1"):
        _strict_sha(evidence[key], SHA1_RE, key)
    for key in ("candidate_sha256", "attempt1_evidence_sha256"):
        _strict_sha(evidence[key], SHA256_RE, key)
    if evidence["final_verification_test_blob_sha1"] is not None:
        _strict_sha(evidence["final_verification_test_blob_sha1"], SHA1_RE, "final test blob")
    for key in ("observed_package_count", "jpx_entry_occurrence_count", "jp_entry_occurrence_count"):
        _strict_int(evidence[key], key)
    for key in ("package_index_network_requests", "package_installations", "environment_mutations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads"):
        _strict_int(evidence[key], key, allow_none=False)
    for key in ("official_wheel_sha256_match", "jpx_installed_equals_wheel_entry", "jp_installed_equals_wheel_entry", "jpx_source_blob_match", "holiday_source_blob_match", "historical_step4_provenance_verified", "reviewed_wheelhouse_provenance_verified", "canonical_environment_promoted", "environment_frozen", "execution_authorized", "calendar_generation_authorized", "t0_authorized", "historical_evaluation_authorized", "future_profitability_established", "t0_run"):
        _strict_bool(evidence[key], key)
    if evidence["t0_run"] is None:
        raise FinalFreezeValidationError("FINAL_EVIDENCE_T0_INVALID")
    for key in ("python_version", "platform_system", "platform_machine", "sysconfig_platform", "pandas_market_calendars_version", "exchange_calendars_version", "official_wheel_filename", "observed_official_wheel_sha256", "attempt1_evidence_repo_path"):
        if evidence[key] is not None and not isinstance(evidence[key], str):
            raise FinalFreezeValidationError("FINAL_EVIDENCE_STRING_INVALID")
    if evidence["xls_probe_status"] not in {"PASS", "FAIL", "NOT_CHECKED", None} or evidence["pdf_probe_status"] not in {"PASS", "FAIL", "NOT_CHECKED", None}:
        raise FinalFreezeValidationError("FINAL_EVIDENCE_PROBE_INVALID")
    if evidence["status"] == "PASS":
        required = (
            evidence["observed_packages"] is not None,
            v10a._normalize_packages(evidence["observed_packages"]) == EXPECTED_PACKAGES,
            evidence["observed_package_count"] == len(EXPECTED_PACKAGES),
            evidence["python_version"] == "3.12.10",
            evidence["platform_system"] == "Windows",
            evidence["platform_machine"] == "AMD64",
            evidence["sysconfig_platform"] == "win-amd64",
            evidence["pandas_market_calendars_version"] == "5.4.0",
            evidence["exchange_calendars_version"] == "4.13.2",
            evidence["official_wheel_filename"] == v10a.OFFICIAL_WHEEL_FILENAME,
            evidence["official_wheel_sha256_match"] is True,
            evidence["jpx_entry_occurrence_count"] == 1,
            evidence["jp_entry_occurrence_count"] == 1,
            evidence["jpx_installed_equals_wheel_entry"] is True,
            evidence["jp_installed_equals_wheel_entry"] is True,
            evidence["jpx_source_blob_match"] is True,
            evidence["holiday_source_blob_match"] is True,
            evidence["xls_probe_status"] == "PASS",
            evidence["pdf_probe_status"] == "PASS",
            evidence["historical_step4_provenance_verified"] is True,
            evidence["reviewed_wheelhouse_provenance_verified"] is True,
            evidence["canonical_environment_promoted"] is False,
            evidence["environment_frozen"] is False,
            evidence["execution_authorized"] is False,
            evidence["calendar_generation_authorized"] is False,
            evidence["t0_authorized"] is False,
            evidence["historical_evaluation_authorized"] is False,
            evidence["future_profitability_established"] is False,
            evidence["t0_run"] is False,
        )
        if not all(required) or any(evidence[key] != 0 for key in ("package_index_network_requests", "package_installations", "environment_mutations", "calendar_object_creations", "calendar_dates_inspected", "protected_or_private_reads")):
            raise FinalFreezeValidationError("FINAL_EVIDENCE_PASS_INVALID")


def _publish(config: FinalFreezeConfig, evidence: Mapping[str, Any]) -> Path:
    if not _output_root_safe(config):
        raise FinalFreezeValidationError("PROVENANCE_BINDING_FAILURE")
    try:
        config.output_root.mkdir(parents=True, exist_ok=False)
        target = config.output_root / FINAL_EVIDENCE_NAME
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
        raise FinalFreezeValidationError("PROVENANCE_BINDING_FAILURE") from error


def _finish(config: FinalFreezeConfig, evidence: dict[str, Any], *, publish: bool) -> dict[str, Any]:
    validate_final_evidence(evidence)
    artifact_path: Path | None = None
    if publish:
        try:
            artifact_path = _publish(config, evidence)
        except FinalFreezeValidationError:
            evidence = _base_final_evidence(config, "PROVENANCE_BINDING_FAILURE", {}, None)
    return {
        "evidence": evidence,
        "artifact_path": artifact_path,
        "status": evidence["status"],
        "failure_code": evidence["failure_code"],
        "canonical_environment_ready": False,
        "environment_frozen": False,
        "execution_authorized": False,
    }


def run_final_verification(config: FinalFreezeConfig, observations: Mapping[str, Any] | None = None, *, publish: bool = True) -> dict[str, Any]:
    """Run the future P3 wrapper; injected observations are test-only."""
    injected = observations is not None
    obs = dict(observations) if injected else _default_operation_observations()
    if _unauthorized_observed(obs):
        return _finish(config, _base_final_evidence(config, "UNAUTHORIZED_OPERATION_OBSERVED", obs), publish=publish)

    stage2 = obs if injected else _default_stage2_observations(config)
    if injected:
        provenance_pass = stage2.get("provenance_valid") is True
    else:
        provenance_pass = _validate_stage2(config, stage2)
    if not provenance_pass:
        return _finish(config, _base_final_evidence(config, "PROVENANCE_BINDING_FAILURE", stage2), publish=publish)

    delegated_config = v10a.V10AValidationConfig(
        repo_root=config.repo_root,
        expected_current_head=config.expected_p2_reviewed_sha,
        expected_live_validation_runner_commit_sha=REVIEWED_ATTEMPT1_RUNNER_SHA,
        expected_live_validation_runner_blob_sha1=REVIEWED_ATTEMPT1_RUNNER_BLOB_SHA1,
        wheelhouse=config.wheelhouse,
        step4_attempt_root=config.step4_attempt_root,
        output_root=config.output_root,
    )
    try:
        delegated = v10a.run_validation(delegated_config, observations=None, publish=False)
    except Exception:
        return _finish(config, _base_final_evidence(config, "PROVENANCE_BINDING_FAILURE", stage2), publish=publish)
    delegated_failure = delegated.get("failure_code")
    if delegated_failure not in FAILURE_CODES:
        delegated_failure = "PROVENANCE_BINDING_FAILURE"
    evidence = _base_final_evidence(config, delegated_failure, stage2, delegated)
    return _finish(config, evidence, publish=publish)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-p2-reviewed-sha", required=True)
    parser.add_argument("--expected-final-verification-runner-blob-sha1", required=True)
    parser.add_argument("--expected-candidate-blob-sha1", required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--step4-attempt-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser


def _config_from_args(args: argparse.Namespace) -> FinalFreezeConfig:
    for value, pattern, label in (
        (args.expected_p2_reviewed_sha, SHA1_RE, "expected p2 sha"),
        (args.expected_final_verification_runner_blob_sha1, SHA1_RE, "final runner blob"),
        (args.expected_candidate_blob_sha1, SHA1_RE, "candidate blob"),
        (args.expected_candidate_sha256, SHA256_RE, "candidate sha256"),
    ):
        _strict_sha(value, pattern, label)
    paths = {name: Path(getattr(args, name)) for name in ("repo_root", "wheelhouse", "step4_attempt_root", "output_root")}
    if any(not path.is_absolute() for path in paths.values()):
        raise FinalFreezeValidationError("PATH_MUST_BE_ABSOLUTE")
    return FinalFreezeConfig(
        repo_root=paths["repo_root"],
        expected_p2_reviewed_sha=args.expected_p2_reviewed_sha,
        expected_final_runner_blob_sha1=args.expected_final_verification_runner_blob_sha1,
        expected_candidate_blob_sha1=args.expected_candidate_blob_sha1,
        expected_candidate_sha256=args.expected_candidate_sha256,
        wheelhouse=paths["wheelhouse"],
        step4_attempt_root=paths["step4_attempt_root"],
        output_root=paths["output_root"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _config_from_args(_build_parser().parse_args(argv))
        result = run_final_verification(config, observations=None, publish=True)
    except (FinalFreezeValidationError, OSError, TypeError, ValueError):
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
