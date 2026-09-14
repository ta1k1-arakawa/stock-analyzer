"""Offline, fail-closed readjudication of the frozen V10C wheel resolution."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import v10c_t0_ml_environment_resolution_runner as resolution_runner
from scripts.v10c_t0_ml_environment_contract import (
    APPROVAL_RECORD_BLOB,
    DIRECT_SPEC_SHA256,
    FROZEN_DESIGN_BLOB,
    FROZEN_DESIGN_SHA,
    PREDECESSOR_LOCK_BLOB,
    PREDECESSOR_LOCK_SHA256,
    STUDY,
    ContractValidationError,
    SHA1_RE,
    SHA256_RE,
    canonical_json_bytes,
    git_blob_sha1,
)


REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
SOURCE_PROVENANCE_NAME = "V10C_T0_ML_RESOLUTION_SOURCE_PROVENANCE.json"
READJUDICATION_ARTIFACT_DIRECTORY_NAME = "offline_readjudication_artifacts"
READJUDICATION_ARTIFACT_STAGING_DIRECTORY_NAME = "offline_readjudication_artifacts.staging"
READJUDICATION_CANDIDATE_NAME = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json"
READJUDICATION_EVIDENCE_NAME = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_OFFLINE_READJUDICATION_EVIDENCE.json"
READJUDICATION_EVIDENCE_SCHEMA = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_OFFLINE_READJUDICATION_EVIDENCE_V1"
READJUDICATION_RUNNER_RELATIVE = Path("scripts/v10c_t0_ml_environment_offline_readjudication_runner.py")
CONTRACT_RELATIVE = Path("scripts/v10c_t0_ml_environment_contract.py")
SOURCE_RESOLUTION_HEAD = "3aee6c2772f30c6dc35d2a7efb862ae15091febc"
SOURCE_RESOLUTION_RUNNER_BLOB = "4a941e4631174c3dc56a1623f2eb270045832905"
SOURCE_WHEEL_MANIFEST_ALGORITHM = "CANONICAL_WHEEL_MANIFEST_V1"
SOURCE_WHEEL_COUNT = 27
SOURCE_WHEEL_TOTAL_BYTES = 94451528
SOURCE_WHEEL_MANIFEST_SHA256 = "5d5953f14b0609767972679554e1999e754621056d863f8c33def96988797b74"
SOURCE_ATTEMPT_STATE_SHA256 = "a4143c2722adcd86223b44910b05c98ed119d6be33c579ff8903582ac7d11439"
SOURCE_STDOUT_SHA256 = "9844249c636532933181bd3e3af4126b6b46f39308086101392d5d00104eff14"
SOURCE_STDERR_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
SOURCE_FAILURE_EVIDENCE_SHA256 = "c183559bb49fc76f6f8fb3567f8354f2e9db7a0c647a6b14cb48b489d00c1a87"
SOURCE_DIRECT_SPEC_BLOB = "f6d3f47da87bb2ac39489dc3a1d7896a604667ab"

SOURCE_PROVENANCE_EXPECTED: dict[str, Any] = {
    "artifact_role": "SOURCE_RESOLUTION_PROVENANCE",
    "future_profitability_established": False,
    "mutation_authorized": False,
    "resolution_retry_authorized": False,
    "schema_version": "V10C_T0_ML_RESOLUTION_SOURCE_PROVENANCE_V1",
    "source_attempt_state_sha256": SOURCE_ATTEMPT_STATE_SHA256,
    "source_direct_spec_git_blob_sha1": SOURCE_DIRECT_SPEC_BLOB,
    "source_direct_spec_sha256": "3f1169e66952148f072d2fcd3b92ab2de840a5cd02bdb46b62833b8ea2c062a9",
    "source_failure_evidence_sha256": SOURCE_FAILURE_EVIDENCE_SHA256,
    "source_phase_b_authority_consumed": True,
    "source_phase_b_process_exit_code": 0,
    "source_phase_b_resolution_invocations": 1,
    "source_phase_c_candidate_artifact_created": False,
    "source_phase_c_failure_code": "RESOLUTION_REPORT_INVALID",
    "source_phase_c_result": "FAIL",
    "source_predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB,
    "source_predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
    "source_resolution_head": SOURCE_RESOLUTION_HEAD,
    "source_resolution_runner_blob_sha1": SOURCE_RESOLUTION_RUNNER_BLOB,
    "source_stderr_sha256": SOURCE_STDERR_SHA256,
    "source_stdout_sha256": SOURCE_STDOUT_SHA256,
    "source_wheel_count": SOURCE_WHEEL_COUNT,
    "source_wheel_manifest_sha256": SOURCE_WHEEL_MANIFEST_SHA256,
    "source_wheel_total_bytes": SOURCE_WHEEL_TOTAL_BYTES,
    "study": STUDY,
    "t0_authorized": False,
    "wheel_manifest_algorithm": SOURCE_WHEEL_MANIFEST_ALGORITHM,
}
SOURCE_PROVENANCE_KEYS = frozenset(SOURCE_PROVENANCE_EXPECTED)
READJUDICATION_EVIDENCE_KEYS = frozenset(
    {
        "schema_version", "study", "status", "failure_code", "source_resolution_head",
        "source_resolution_runner_blob_sha1", "source_provenance_git_blob_sha1",
        "source_wheel_count", "source_wheel_total_bytes", "source_wheel_manifest_sha256",
        "source_attempt_state_sha256", "source_stdout_sha256", "source_stderr_sha256",
        "source_failure_evidence_sha256", "source_phase_b_process_exit_code",
        "source_phase_b_resolution_invocations", "source_phase_b_authority_consumed",
        "source_phase_c_failure_code", "source_resolution_retry_authorized",
        "readjudication_validator_git_sha", "readjudication_runner_git_blob_sha1",
        "readjudication_contract_git_blob_sha1", "direct_spec_git_blob_sha1",
        "direct_spec_sha256", "predecessor_lock_git_blob_sha1", "predecessor_lock_sha256",
        "candidate_artifact_created", "successor_lock_candidate_sha256",
        "resolved_package_count", "network_requests", "package_resolution_reruns",
        "package_installations", "environment_mutations", "model_fits", "t0_runs",
        "payload_reads", "mutation_authorized", "t0_authorized", "human_authority_consumed",
    }
)


class OfflineReadjudicationError(ContractValidationError):
    """Safe, closed offline readjudication failure."""


@dataclass(frozen=True)
class OfflineReadjudicationConfig:
    repo_root: Path
    expected_current_head: str
    expected_reviewed_readjudication_runner_sha: str
    expected_readjudication_runner_blob_sha1: str
    expected_contract_blob_sha1: str
    expected_source_provenance_blob_sha1: str
    source_attempt_root: Path
    output_root: Path


def published_artifact_directory(root: Path) -> Path:
    return Path(root) / READJUDICATION_ARTIFACT_DIRECTORY_NAME


def staging_artifact_directory(root: Path) -> Path:
    return Path(root) / READJUDICATION_ARTIFACT_STAGING_DIRECTORY_NAME


def published_candidate_path(root: Path) -> Path:
    return published_artifact_directory(root) / READJUDICATION_CANDIDATE_NAME


def published_evidence_path(root: Path) -> Path:
    return published_artifact_directory(root) / READJUDICATION_EVIDENCE_NAME


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(repo_root: Path, args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
        shell=False,
    ).stdout


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


def _require_sha(value: Any, pattern: Any, label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise OfflineReadjudicationError(label)


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH") from error
    if not isinstance(value, dict):
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    return value, raw


def _validate_repository(config: OfflineReadjudicationConfig) -> None:
    _require_sha(config.expected_current_head, SHA1_RE, "SOURCE_PROVENANCE_MISMATCH")
    _require_sha(config.expected_reviewed_readjudication_runner_sha, SHA1_RE, "SOURCE_PROVENANCE_MISMATCH")
    _require_sha(config.expected_readjudication_runner_blob_sha1, SHA1_RE, "SOURCE_PROVENANCE_MISMATCH")
    _require_sha(config.expected_contract_blob_sha1, SHA1_RE, "SOURCE_PROVENANCE_MISMATCH")
    _require_sha(config.expected_source_provenance_blob_sha1, SHA1_RE, "SOURCE_PROVENANCE_MISMATCH")
    if config.expected_reviewed_readjudication_runner_sha != config.expected_current_head:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    try:
        remote = _git_output(config.repo_root, ["config", "--get", "remote.origin.url"]).decode().strip()
        branch = _git_output(config.repo_root, ["branch", "--show-current"]).decode().strip()
        origin = _git_output(config.repo_root, ["rev-parse", f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}"]).decode().strip()
        head = _git_output(config.repo_root, ["rev-parse", "HEAD"]).decode().strip()
        clean = _git_output(config.repo_root, ["status", "--porcelain", "--untracked-files=all"]) == b""
        runner_blob = _git_output(config.repo_root, ["rev-parse", f"HEAD:{READJUDICATION_RUNNER_RELATIVE.as_posix()}"]).decode().strip()
        contract_blob = _git_output(config.repo_root, ["rev-parse", f"HEAD:{CONTRACT_RELATIVE.as_posix()}"]).decode().strip()
        provenance_blob = _git_output(config.repo_root, ["rev-parse", f"HEAD:{SOURCE_PROVENANCE_NAME}"]).decode().strip()
        reviewed_runner_blob = _git_output(config.repo_root, ["rev-parse", f"{config.expected_reviewed_readjudication_runner_sha}:{READJUDICATION_RUNNER_RELATIVE.as_posix()}"]).decode().strip()
        reviewed_contract_blob = _git_output(config.repo_root, ["rev-parse", f"{config.expected_reviewed_readjudication_runner_sha}:{CONTRACT_RELATIVE.as_posix()}"]).decode().strip()
        reviewed_provenance_blob = _git_output(config.repo_root, ["rev-parse", f"{config.expected_reviewed_readjudication_runner_sha}:{SOURCE_PROVENANCE_NAME}"]).decode().strip()
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError) as error:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH") from error
    if (
        not _repo_identity_matches(remote)
        or branch != AUTHORITATIVE_BRANCH
        or origin != config.expected_current_head
        or head != config.expected_current_head
        or not clean
        or runner_blob != config.expected_readjudication_runner_blob_sha1
        or contract_blob != config.expected_contract_blob_sha1
        or provenance_blob != config.expected_source_provenance_blob_sha1
        or reviewed_runner_blob != config.expected_readjudication_runner_blob_sha1
        or reviewed_contract_blob != config.expected_contract_blob_sha1
        or reviewed_provenance_blob != config.expected_source_provenance_blob_sha1
        or git_blob_sha1(Path(__file__).read_bytes()) != config.expected_readjudication_runner_blob_sha1
    ):
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")


def _validate_provenance(config: OfflineReadjudicationConfig) -> dict[str, Any]:
    value, raw = _read_json(config.repo_root / SOURCE_PROVENANCE_NAME)
    if (
        git_blob_sha1(raw) != config.expected_source_provenance_blob_sha1
        or set(value) != SOURCE_PROVENANCE_KEYS
        or value != SOURCE_PROVENANCE_EXPECTED
    ):
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    return value


def _source_config(config: OfflineReadjudicationConfig) -> resolution_runner.PhaseAConfig:
    return resolution_runner.PhaseAConfig(
        repo_root=config.repo_root,
        expected_current_head=SOURCE_RESOLUTION_HEAD,
        expected_reviewed_runner_sha=SOURCE_RESOLUTION_HEAD,
        expected_direct_spec_git_blob_sha1=SOURCE_DIRECT_SPEC_BLOB,
        expected_direct_spec_sha256=DIRECT_SPEC_SHA256,
        durable_root=config.source_attempt_root,
    )


def _validate_wheelhouse_manifest(
    wheelhouse: Path,
    wheels: Sequence[Mapping[str, Any]],
) -> None:
    try:
        entries = list(wheelhouse.iterdir())
        if (
            len(entries) != SOURCE_WHEEL_COUNT
            or any(
                _is_reparse_or_symlink(entry)
                or not entry.is_file()
                or not entry.name.lower().endswith(".whl")
                for entry in entries
            )
        ):
            raise OfflineReadjudicationError("OFFLINE_WHEEL_VALIDATION_FAILURE")
        total_bytes = sum(entry.stat().st_size for entry in entries)
        if total_bytes != SOURCE_WHEEL_TOTAL_BYTES or len(wheels) != SOURCE_WHEEL_COUNT:
            raise OfflineReadjudicationError("OFFLINE_WHEEL_VALIDATION_FAILURE")
        manifest = sorted(
            [
                {
                    "name": item["name"],
                    "version": item["version"],
                    "filename": item["filename"],
                    "sha256": item["sha256"],
                    "size_bytes": (wheelhouse / item["filename"]).stat().st_size,
                }
                for item in wheels
            ],
            key=lambda item: (item["name"], item["version"], item["filename"].casefold()),
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise OfflineReadjudicationError("OFFLINE_WHEEL_VALIDATION_FAILURE") from error
    if hashlib.sha256(canonical_json_bytes(manifest)).hexdigest() != SOURCE_WHEEL_MANIFEST_SHA256:
        raise OfflineReadjudicationError("OFFLINE_WHEEL_VALIDATION_FAILURE")


def _validate_source_inputs(config: OfflineReadjudicationConfig, provenance: Mapping[str, Any]) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    source = config.source_attempt_root
    if not source.is_dir() or _unsafe_existing_ancestor(source):
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    if os.path.lexists(config.output_root) or _unsafe_existing_ancestor(config.output_root):
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
    try:
        source_resolved = Path(os.path.realpath(source))
        output_resolved = Path(os.path.realpath(config.output_root))
    except OSError as error:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH") from error
    if _path_inside(source_resolved, output_resolved) or _path_inside(output_resolved, source_resolved):
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    if _sha256_file(source / resolution_runner.STATE_NAME) != provenance["source_attempt_state_sha256"]:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    if _sha256_file(source / resolution_runner.STDOUT_NAME) != provenance["source_stdout_sha256"]:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    if _sha256_file(source / resolution_runner.STDERR_NAME) != provenance["source_stderr_sha256"]:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    artifact_directory = resolution_runner.published_artifact_directory(source)
    if not artifact_directory.is_dir() or _is_reparse_or_symlink(artifact_directory):
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    entries = list(artifact_directory.iterdir())
    if {entry.name for entry in entries} != {resolution_runner.EVIDENCE_NAME} or any(_is_reparse_or_symlink(entry) or not entry.is_file() for entry in entries):
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    if (artifact_directory / resolution_runner.CANDIDATE_NAME).exists() or (source / resolution_runner.CANDIDATE_NAME).exists() or (source / resolution_runner.EVIDENCE_NAME).exists():
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    failure_evidence_path = artifact_directory / resolution_runner.EVIDENCE_NAME
    if _sha256_file(failure_evidence_path) != provenance["source_failure_evidence_sha256"]:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    failure_evidence, _ = _read_json(failure_evidence_path)
    if failure_evidence.get("status") != "FAIL" or failure_evidence.get("failure_code") != "RESOLUTION_REPORT_INVALID":
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    try:
        resolution_runner.validate_evidence(
            failure_evidence,
            expected_reviewed_sha=SOURCE_RESOLUTION_HEAD,
            expected_direct_blob=SOURCE_DIRECT_SPEC_BLOB,
            expected_direct_sha=DIRECT_SPEC_SHA256,
            expected_candidate_sha=None,
        )
        state, _ = _read_json(source / resolution_runner.STATE_NAME)
        resolution_runner._validate_attempt_state(_source_config(config), state)
    except (ContractValidationError, KeyError, TypeError, ValueError) as error:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH") from error
    wheelhouse = source / resolution_runner.WHEELHOUSE_NAME
    if not wheelhouse.is_dir() or _is_reparse_or_symlink(wheelhouse):
        raise OfflineReadjudicationError("OFFLINE_WHEEL_VALIDATION_FAILURE")
    code, wheels = resolution_runner.inspect_wheelhouse(wheelhouse)
    if code != "NONE" or wheels is None or len(wheels) != SOURCE_WHEEL_COUNT:
        raise OfflineReadjudicationError("OFFLINE_WHEEL_VALIDATION_FAILURE")
    _validate_wheelhouse_manifest(wheelhouse, wheels)
    if provenance["source_wheel_manifest_sha256"] != SOURCE_WHEEL_MANIFEST_SHA256:
        raise OfflineReadjudicationError("SOURCE_PROVENANCE_MISMATCH")
    return wheels, failure_evidence


def _build_evidence(config: OfflineReadjudicationConfig, provenance: Mapping[str, Any], *, status: str, failure_code: str, candidate_sha: str | None, package_count: int | None) -> dict[str, Any]:
    evidence = {
        "schema_version": READJUDICATION_EVIDENCE_SCHEMA,
        "study": STUDY,
        "status": status,
        "failure_code": failure_code,
        "source_resolution_head": SOURCE_RESOLUTION_HEAD,
        "source_resolution_runner_blob_sha1": SOURCE_RESOLUTION_RUNNER_BLOB,
        "source_provenance_git_blob_sha1": config.expected_source_provenance_blob_sha1,
        "source_wheel_count": SOURCE_WHEEL_COUNT,
        "source_wheel_total_bytes": SOURCE_WHEEL_TOTAL_BYTES,
        "source_wheel_manifest_sha256": SOURCE_WHEEL_MANIFEST_SHA256,
        "source_attempt_state_sha256": SOURCE_ATTEMPT_STATE_SHA256,
        "source_stdout_sha256": SOURCE_STDOUT_SHA256,
        "source_stderr_sha256": SOURCE_STDERR_SHA256,
        "source_failure_evidence_sha256": SOURCE_FAILURE_EVIDENCE_SHA256,
        "source_phase_b_process_exit_code": 0,
        "source_phase_b_resolution_invocations": 1,
        "source_phase_b_authority_consumed": True,
        "source_phase_c_failure_code": "RESOLUTION_REPORT_INVALID",
        "source_resolution_retry_authorized": False,
        "readjudication_validator_git_sha": config.expected_reviewed_readjudication_runner_sha,
        "readjudication_runner_git_blob_sha1": config.expected_readjudication_runner_blob_sha1,
        "readjudication_contract_git_blob_sha1": config.expected_contract_blob_sha1,
        "direct_spec_git_blob_sha1": SOURCE_DIRECT_SPEC_BLOB,
        "direct_spec_sha256": DIRECT_SPEC_SHA256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "candidate_artifact_created": status == "PASS",
        "successor_lock_candidate_sha256": candidate_sha,
        "resolved_package_count": package_count,
        "network_requests": 0,
        "package_resolution_reruns": 0,
        "package_installations": 0,
        "environment_mutations": 0,
        "model_fits": 0,
        "t0_runs": 0,
        "payload_reads": 0,
        "mutation_authorized": False,
        "t0_authorized": False,
        "human_authority_consumed": False,
    }
    if set(evidence) != READJUDICATION_EVIDENCE_KEYS:
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
    return evidence


def _validate_evidence(evidence: Mapping[str, Any], *, expected_candidate_sha: str | None) -> None:
    if set(evidence) != READJUDICATION_EVIDENCE_KEYS or evidence["schema_version"] != READJUDICATION_EVIDENCE_SCHEMA:
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
    if evidence["status"] not in {"PASS", "FAIL"} or evidence["failure_code"] not in {"NONE", "SOURCE_PROVENANCE_MISMATCH", "OFFLINE_WHEEL_VALIDATION_FAILURE", "ARTIFACT_PUBLICATION_FAILURE"}:
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
    for key, pattern in (
        ("source_resolution_head", SHA1_RE), ("source_resolution_runner_blob_sha1", SHA1_RE),
        ("source_provenance_git_blob_sha1", SHA1_RE), ("source_wheel_manifest_sha256", SHA256_RE),
        ("source_attempt_state_sha256", SHA256_RE), ("source_stdout_sha256", SHA256_RE),
        ("source_stderr_sha256", SHA256_RE), ("source_failure_evidence_sha256", SHA256_RE),
        ("readjudication_validator_git_sha", SHA1_RE), ("readjudication_runner_git_blob_sha1", SHA1_RE),
        ("readjudication_contract_git_blob_sha1", SHA1_RE), ("direct_spec_git_blob_sha1", SHA1_RE),
        ("direct_spec_sha256", SHA256_RE), ("predecessor_lock_git_blob_sha1", SHA1_RE),
        ("predecessor_lock_sha256", SHA256_RE),
    ):
        _require_sha(evidence[key], pattern, "ARTIFACT_PUBLICATION_FAILURE")
    if evidence["candidate_artifact_created"] != (evidence["status"] == "PASS") or evidence["successor_lock_candidate_sha256"] != expected_candidate_sha:
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
    if evidence["status"] == "PASS" and evidence["failure_code"] != "NONE":
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
    if evidence["status"] == "FAIL" and evidence["failure_code"] == "NONE":
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
    if any(evidence[key] != 0 for key in ("network_requests", "package_resolution_reruns", "package_installations", "environment_mutations", "model_fits", "t0_runs", "payload_reads")):
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
    if evidence["mutation_authorized"] is not False or evidence["t0_authorized"] is not False or evidence["human_authority_consumed"] is not False:
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")


def _write_bytes(path: Path, raw: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _reread_staged_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _publish_bundle(output_root: Path, *, candidate_bytes: bytes | None, evidence_bytes: bytes) -> None:
    if os.path.lexists(output_root) or _unsafe_existing_ancestor(output_root) or not output_root.parent.is_dir():
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
    final = published_artifact_directory(output_root)
    staging = staging_artifact_directory(output_root)
    try:
        output_root.mkdir()
        staging.mkdir()
        if candidate_bytes is not None:
            _write_bytes(staging / READJUDICATION_CANDIDATE_NAME, candidate_bytes)
        _write_bytes(staging / READJUDICATION_EVIDENCE_NAME, evidence_bytes)
        expected = {READJUDICATION_EVIDENCE_NAME} | ({READJUDICATION_CANDIDATE_NAME} if candidate_bytes is not None else set())
        entries = list(staging.iterdir())
        if {entry.name for entry in entries} != expected or any(_is_reparse_or_symlink(entry) or not entry.is_file() for entry in entries):
            raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
        if candidate_bytes is not None and _reread_staged_bytes(staging / READJUDICATION_CANDIDATE_NAME) != candidate_bytes:
            raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
        if _reread_staged_bytes(staging / READJUDICATION_EVIDENCE_NAME) != evidence_bytes:
            raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE")
        os.rename(staging, final)
    except OfflineReadjudicationError:
        raise
    except OSError as error:
        raise OfflineReadjudicationError("ARTIFACT_PUBLICATION_FAILURE") from error


def run_offline_readjudication(config: OfflineReadjudicationConfig) -> dict[str, Any]:
    """Readjudicate the frozen source wheelhouse without resolution or installation."""

    try:
        _validate_repository(config)
        provenance = _validate_provenance(config)
        wheels, _ = _validate_source_inputs(config, provenance)
    except OfflineReadjudicationError as error:
        if str(error) == "ARTIFACT_PUBLICATION_FAILURE":
            raise
        evidence = _build_evidence(
            config,
            SOURCE_PROVENANCE_EXPECTED,
            status="FAIL",
            failure_code=str(error),
            candidate_sha=None,
            package_count=None,
        )
        _validate_evidence(evidence, expected_candidate_sha=None)
        _publish_bundle(config.output_root, candidate_bytes=None, evidence_bytes=canonical_json_bytes(evidence))
        return {"status": "FAIL", "failure_code": str(error), "candidate_artifact_created": False}

    source_config = _source_config(config)
    try:
        candidate = resolution_runner._build_candidate(source_config, wheels)
        candidate_bytes = canonical_json_bytes(candidate)
        candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
        evidence = _build_evidence(config, provenance, status="PASS", failure_code="NONE", candidate_sha=candidate_sha, package_count=candidate["resolved_package_count"])
        _validate_evidence(evidence, expected_candidate_sha=candidate_sha)
        _publish_bundle(config.output_root, candidate_bytes=candidate_bytes, evidence_bytes=canonical_json_bytes(evidence))
        return {"status": "PASS", "failure_code": "NONE", "candidate_artifact_created": True, "resolved_package_count": candidate["resolved_package_count"], "candidate_sha256": candidate_sha}
    except ContractValidationError as error:
        if isinstance(error, OfflineReadjudicationError) and str(error) == "ARTIFACT_PUBLICATION_FAILURE":
            raise
        evidence = _build_evidence(config, provenance, status="FAIL", failure_code="OFFLINE_WHEEL_VALIDATION_FAILURE", candidate_sha=None, package_count=None)
        _validate_evidence(evidence, expected_candidate_sha=None)
        _publish_bundle(config.output_root, candidate_bytes=None, evidence_bytes=canonical_json_bytes(evidence))
        return {"status": "FAIL", "failure_code": "OFFLINE_WHEEL_VALIDATION_FAILURE", "candidate_artifact_created": False}
