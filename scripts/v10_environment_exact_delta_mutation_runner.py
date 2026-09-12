"""Synthetic-testable, exact-local-wheel V10 canonical mutation runner.

The default entry point is for a future separately authorized operation.  This
module performs no work on import.  Tests inject observations and use only
temporary wheelhouses, receipt bytes, attempt roots, and process launchers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from scripts import v10_environment_mutation_preflight_runner as preflight
    from scripts.v10_environment_extension_contract import (
        ContractValidationError,
        build_exact_delta_install_argv,
        inspect_wheel_file,
        validate_mutation_preflight_receipt,
    )
except ModuleNotFoundError:  # direct ``python scripts/<runner>.py`` invocation
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import v10_environment_mutation_preflight_runner as preflight
    from v10_environment_extension_contract import (
        ContractValidationError,
        build_exact_delta_install_argv,
        inspect_wheel_file,
        validate_mutation_preflight_receipt,
    )


MUTATION_RUNNER_RELATIVE = Path("scripts/v10_environment_exact_delta_mutation_runner.py")
RECEIPT_RELATIVE = Path("V10_CANONICAL_ENVIRONMENT_MUTATION_PREFLIGHT_RECEIPT.json")
ATTEMPT_STATE_SCHEMA = "V10_CANONICAL_ENVIRONMENT_EXACT_DELTA_MUTATION_ATTEMPT_V1"
FRESH_MUTATION_AUTHORITY_TOKEN = "FRESH_CANONICAL_ENVIRONMENT_MUTATION_AUTHORITY_V1"
EXPECTED_STEP3_RECEIPT_SHA256 = "0c15278d79f88110766a581aa3c14bdc03434b88806b8a7975327bbd6bee07ae"
EXPECTED_EXTENSION_DESIGN_SHA = "efe2e9d8cfab696c74b94cfd1cfaa2a2a4706c58"
EXPECTED_PREMUTATION_RUNNER_COMMIT_SHA = "9f37cc5c0c10db11a0164ab7a3d4d2dc9d311adb"
EXPECTED_PREMUTATION_RUNNER_BLOB_SHA1 = "21e9f94928f6e6ce315fa473bf52d65ff7537ffb"
EXPECTED_DELTA_COUNT = 5
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

PROVENANCE_BINDING_FAILURE = "PROVENANCE_BINDING_FAILURE"
PREMUTATION_RECEIPT_INVALID = "PREMUTATION_RECEIPT_INVALID"
PREDECESSOR_LIVE_BASELINE_MISMATCH = "PREDECESSOR_LIVE_BASELINE_MISMATCH"
REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE = "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE"
MUTATION_AUTHORITY_MISSING = "MUTATION_AUTHORITY_MISSING"
MUTATION_ATTEMPT_STATE_FAILURE = "MUTATION_ATTEMPT_STATE_FAILURE"
MUTATION_PROCESS_START_FAILURE = "MUTATION_PROCESS_START_FAILURE"
MUTATION_PROCESS_FAILURE = "MUTATION_PROCESS_FAILURE"
NONE = "NONE"


class MutationValidationError(ContractValidationError):
    """Fail-closed mutation-stage or durable-state error."""


@dataclass(frozen=True)
class MutationConfig:
    preflight_config: preflight.PreflightConfig
    expected_mutation_runner_commit_sha: str
    expected_mutation_runner_blob_sha1: str
    attempt_root: Path
    step3_receipt_path: Path | None = None
    expected_step3_receipt_sha256: str = EXPECTED_STEP3_RECEIPT_SHA256

    @property
    def repo_root(self) -> Path:
        return self.preflight_config.repo_root

    @property
    def wheelhouse(self) -> Path:
        return self.preflight_config.wheelhouse

    @property
    def canonical_interpreter(self) -> Path:
        return self.preflight_config.canonical_interpreter

    @property
    def default_step3_receipt_path(self) -> Path:
        return self.repo_root / RECEIPT_RELATIVE


@dataclass(frozen=True)
class PhaseAResult:
    candidate: Mapping[str, Any]
    receipt: Mapping[str, Any]
    receipt_sha256: str
    wheel_result: Mapping[str, Any]
    install_argv: tuple[str, ...]
    wheel_manifest: tuple[Mapping[str, str], ...]
    delta_wheel_paths: tuple[str, ...]


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )


def _strict_sha(value: Any, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise MutationValidationError(f"{label}_INVALID")


def _failure(
    code: str,
    *,
    phase_a: PhaseAResult | None = None,
    authority_consumed: bool = False,
    mutation_started: bool = False,
    attempt_root: Path | None = None,
    attempt_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": "FAIL",
        "failure_code": code,
        "phase_a": phase_a,
        "mutation_authority_consumed": authority_consumed,
        "mutation_started": mutation_started,
        "attempt_root": str(attempt_root) if attempt_root is not None else None,
        "attempt_state": dict(attempt_state) if attempt_state is not None else None,
        "canonical_environment_ready": False,
        "environment_frozen": False,
    }


def _success(phase_a: PhaseAResult, *, attempt_root: Path, state: Mapping[str, Any], process_result: Any) -> dict[str, Any]:
    return {
        "status": "PASS",
        "failure_code": NONE,
        "phase_a": phase_a,
        "mutation_authority_consumed": True,
        "mutation_started": True,
        "attempt_root": str(attempt_root),
        "attempt_state": dict(state),
        "process_result": process_result,
        "canonical_environment_ready": False,
        "environment_frozen": False,
    }


def _read_step3_receipt(config: MutationConfig, observations: Mapping[str, Any]) -> bytes | None:
    supplied = observations.get("step3_receipt_bytes")
    if supplied is not None:
        return supplied if isinstance(supplied, bytes) else None
    path = config.step3_receipt_path or config.default_step3_receipt_path
    try:
        return path.read_bytes()
    except (OSError, ValueError):
        return None


def _validate_step3_receipt(config: MutationConfig, raw: bytes | None) -> tuple[dict[str, Any], str] | None:
    if raw is None:
        return None
    if hashlib.sha256(raw).hexdigest() != config.expected_step3_receipt_sha256:
        return None
    try:
        receipt = json.loads(raw.decode("utf-8"))
        if not isinstance(receipt, dict):
            return None
        validate_mutation_preflight_receipt(
            receipt,
            expected_extension_design_sha=config.preflight_config.expected_extension_design_sha,
            expected_successor_lock_candidate_sha256=config.preflight_config.expected_candidate_sha256,
            expected_migration_authority_git_blob_sha1=config.preflight_config.expected_migration_authority_blob_sha1,
            expected_generic_lock_git_blob_sha1=config.preflight_config.expected_generic_lock_blob_sha1,
        )
        if (
            receipt["status"] != "PASS"
            or receipt["failure_code"] != NONE
            or receipt["wheelhouse_integrity_verified"] is not True
            or receipt["delta_wheel_count"] != EXPECTED_DELTA_COUNT
            or receipt["mutation_authority_consumed"] is not False
            or receipt["mutation_started"] is not False
        ):
            return None
    except (ContractValidationError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return receipt, hashlib.sha256(raw).hexdigest()


def _mutation_runner_provenance_ok(config: MutationConfig, observations: Mapping[str, Any]) -> bool:
    for value, pattern, label in (
        (config.expected_mutation_runner_commit_sha, SHA1_RE, "expected mutation runner commit"),
        (config.expected_mutation_runner_blob_sha1, SHA1_RE, "expected mutation runner blob"),
        (config.expected_step3_receipt_sha256, SHA256_RE, "expected Step-3 receipt SHA"),
    ):
        try:
            _strict_sha(value, pattern, label)
        except MutationValidationError:
            return False
    return (
        observations.get("mutation_runner_commit_exists") is True
        and observations.get("reviewed_mutation_runner_blob_sha1") == config.expected_mutation_runner_blob_sha1
        and observations.get("current_mutation_runner_blob_sha1") == config.expected_mutation_runner_blob_sha1
    )


def _validate_manifest_binding(config: MutationConfig, candidate: Mapping[str, Any], wheel_result: Mapping[str, Any]) -> tuple[Mapping[str, str], ...]:
    bindings: list[Mapping[str, str]] = []
    delta_paths = {str(path) for path in wheel_result["delta_wheel_paths"]}
    for item in candidate["resolved_wheels"]:
        filename = item["filename"]
        path = config.wheelhouse / filename
        inspected = inspect_wheel_file(path)
        if inspected != item:
            raise MutationValidationError("REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE")
        bindings.append(
            {
                "name": item["name"],
                "version": item["version"],
                "filename": filename,
                "sha256": item["sha256"],
                "path": str(path),
                "is_delta": str(path) in delta_paths,
            }
        )
    return tuple(bindings)


def _wheel_signature(wheel_result: Mapping[str, Any], manifest: Sequence[Mapping[str, str]]) -> tuple[Any, ...]:
    return (
        wheel_result.get("ok"),
        wheel_result.get("failure_code"),
        wheel_result.get("wheelhouse_integrity_verified"),
        wheel_result.get("delta_wheel_count"),
        tuple(wheel_result.get("delta_packages", ())),
        tuple(str(path) for path in wheel_result.get("delta_wheel_paths", ())),
        tuple(tuple(sorted(item.items())) for item in manifest),
    )


def _phase_a(config: MutationConfig, observations: Mapping[str, Any] | None) -> tuple[PhaseAResult | None, str | None]:
    injected = observations is not None
    if observations is None:
        observations = _default_provenance_observations(config)
    # Reject malformed/mismatched mutation-runner CLI bindings before any
    # shared predecessor/environment observation can be reached.
    if not _mutation_runner_provenance_ok(config, observations):
        return None, PROVENANCE_BINDING_FAILURE
    base = preflight._validate_provenance(config.preflight_config, observations)
    if base is None:
        return None, PROVENANCE_BINDING_FAILURE
    receipt_pair = _validate_step3_receipt(config, _read_step3_receipt(config, observations))
    if receipt_pair is None:
        return None, PREMUTATION_RECEIPT_INVALID
    receipt, receipt_sha = receipt_pair

    if not preflight._wheelhouse_path_is_safe(config.wheelhouse):
        return None, REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE

    if not injected:
        try:
            live = _default_predecessor_observations(config)
        except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError, MutationValidationError):
            return None, PREDECESSOR_LIVE_BASELINE_MISMATCH
    else:
        live = observations
    if not preflight._validate_predecessor(config.preflight_config, live):
        return None, PREDECESSOR_LIVE_BASELINE_MISMATCH
    try:
        wheel_result, install_argv = preflight._derive_wheelhouse(config.preflight_config, base["candidate"])
    except (ContractValidationError, OSError, TypeError, ValueError) as error:
        if str(error).startswith("WHEELHOUSE_FILESYSTEM_SAFETY_FAILURE"):
            return None, str(error)
        return None, REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE
    if (
        not wheel_result.get("ok")
        or wheel_result.get("delta_wheel_count") != EXPECTED_DELTA_COUNT
        or tuple(wheel_result.get("delta_packages", ())) != preflight.EXPECTED_DELTA_PACKAGES
    ):
        return None, REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE
    try:
        manifest = _validate_manifest_binding(config, base["candidate"], wheel_result)
    except (ContractValidationError, OSError, TypeError, ValueError):
        return None, REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE
    return (
        PhaseAResult(
            candidate=base["candidate"],
            receipt=receipt,
            receipt_sha256=receipt_sha,
            wheel_result=wheel_result,
            install_argv=tuple(install_argv),
            wheel_manifest=manifest,
            delta_wheel_paths=tuple(str(path) for path in wheel_result["delta_wheel_paths"]),
        ),
        None,
    )


def _default_provenance_observations(config: MutationConfig) -> dict[str, Any]:
    observations = preflight._default_provenance_observations(config.preflight_config)
    try:
        observations.update(
            mutation_runner_commit_exists=preflight._git_exists(
                config.repo_root, config.expected_mutation_runner_commit_sha
            ),
            reviewed_mutation_runner_blob_sha1=preflight._run_git(
                config.repo_root,
                [
                    "rev-parse",
                    f"{config.expected_mutation_runner_commit_sha}:{MUTATION_RUNNER_RELATIVE.as_posix()}",
                ],
            ).decode().strip(),
            current_mutation_runner_blob_sha1=preflight._run_git(
                config.repo_root,
                ["hash-object", "--", str(config.repo_root / MUTATION_RUNNER_RELATIVE)],
            ).decode().strip(),
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError):
        return observations
    return observations


def _default_predecessor_observations(config: MutationConfig) -> dict[str, Any]:
    return preflight._default_predecessor_observations(config.preflight_config)


def _atomic_create(path: Path, raw: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE") from error
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _prepare_attempt_root(config: MutationConfig) -> None:
    root = config.attempt_root
    safety = preflight.validate_durable_root(
        root,
        repo_root=config.repo_root,
        protected_environment=config.preflight_config.canonical_environment,
        governed_roots=(config.wheelhouse,),
    )
    if safety != "DURABLE_ROOT_OK" or root.exists():
        raise MutationValidationError(f"MUTATION_ATTEMPT_STATE_FAILURE:{safety}")
    root.mkdir(parents=True, exist_ok=False)


def _state_base(config: MutationConfig, phase_a: PhaseAResult) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_STATE_SCHEMA,
        "expected_current_head": config.preflight_config.expected_current_head,
        "mutation_runner_blob_sha1": config.expected_mutation_runner_blob_sha1,
        "step3_receipt_sha256": phase_a.receipt_sha256,
        "reviewed_successor_lock_candidate_sha256": config.preflight_config.expected_candidate_sha256,
        "wheel_manifest": [dict(item) for item in phase_a.wheel_manifest],
        "delta_wheel_paths": list(phase_a.delta_wheel_paths),
        "process_start_attempted": False,
        "process_started": False,
        "mutation_authority_consumed": False,
        "mutation_started": False,
        "retry_authorized": False,
        "process_exit_code": None,
        "stdout_capture_path": None,
        "stdout_sha256": None,
        "stderr_capture_path": None,
        "stderr_sha256": None,
        "failure_code": None,
    }


def _write_state(root: Path, filename: str, state: Mapping[str, Any]) -> None:
    _atomic_create(root / filename, canonical_json_bytes(state))


def _launch_child(argv: Sequence[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )


def _strict_process_result(child: Any) -> tuple[int, bytes, bytes]:
    communicate = getattr(child, "communicate", None)
    if not callable(communicate):
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    stdout, stderr = communicate()
    returncode = getattr(child, "returncode", None)
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    return returncode, stdout, stderr


def _capture_output(root: Path, filename: str, raw: bytes) -> tuple[str, str]:
    path = root / filename
    _atomic_create(path, raw)
    return str(path), hashlib.sha256(raw).hexdigest()


def _launch_phase_b(
    config: MutationConfig,
    phase_a: PhaseAResult,
    process_runner: Callable[[Sequence[str]], Any],
) -> dict[str, Any]:
    try:
        _prepare_attempt_root(config)
        root = config.attempt_root
        initial = _state_base(config, phase_a)
        _write_state(root, "attempt_state_prelaunch.json", initial)
    except (OSError, MutationValidationError) as error:
        return _failure(MUTATION_ATTEMPT_STATE_FAILURE, phase_a=phase_a)

    attempted = dict(initial)
    attempted.update(
        process_start_attempted=True,
        mutation_authority_consumed=True,
        mutation_started=True,
    )
    try:
        _write_state(root, "attempt_state_attempted.json", attempted)
    except (OSError, MutationValidationError):
        return _failure(
            MUTATION_ATTEMPT_STATE_FAILURE,
            phase_a=phase_a,
            authority_consumed=False,
            mutation_started=False,
            attempt_root=root,
            attempt_state=initial,
        )

    start_failure = False
    state_failure = False
    child_started = False
    started = dict(attempted)
    try:
        child = process_runner(phase_a.install_argv)
        child_started = True
        started["process_started"] = True
        try:
            _write_state(root, "attempt_state_started.json", started)
        except (OSError, MutationValidationError):
            # The child exists.  Do not abandon it or start another one;
            # communicate() below remains the sole wait/capture operation.
            state_failure = True
        returncode, stdout, stderr = _strict_process_result(child)
    except OSError:
        stdout = b""
        stderr = b""
        returncode = None
        start_failure = not child_started
        state_failure = child_started
    except MutationValidationError:
        stdout = b""
        stderr = b""
        returncode = None
        state_failure = True

    try:
        stdout_path, stdout_sha = _capture_output(root, "stdout.bin", stdout)
        stderr_path, stderr_sha = _capture_output(root, "stderr.bin", stderr)
        final = dict(started if child_started else attempted)
        final.update(
            process_start_attempted=True,
            process_started=child_started,
            process_exit_code=returncode,
            stdout_capture_path=stdout_path,
            stdout_sha256=stdout_sha,
            stderr_capture_path=stderr_path,
            stderr_sha256=stderr_sha,
            failure_code=(
                MUTATION_PROCESS_START_FAILURE
                if start_failure
                else MUTATION_ATTEMPT_STATE_FAILURE
                if state_failure
                else NONE
                if returncode == 0
                else MUTATION_PROCESS_FAILURE
            ),
        )
        _write_state(root, "attempt_state_complete.json", final)
    except (OSError, MutationValidationError):
        return _failure(
            MUTATION_ATTEMPT_STATE_FAILURE,
            phase_a=phase_a,
            authority_consumed=True,
            mutation_started=True,
            attempt_root=root,
            attempt_state=started,
        )

    failure = final["failure_code"]
    if failure == NONE:
        return _success(phase_a, attempt_root=root, state=final, process_result={"returncode": returncode})
    return _failure(
        failure,
        phase_a=phase_a,
        authority_consumed=True,
        mutation_started=True,
        attempt_root=root,
        attempt_state=final,
    )


def run_mutation(
    config: MutationConfig,
    observations: Mapping[str, Any] | None = None,
    *,
    mutation_authority_token: str | None = None,
    process_runner: Callable[[Sequence[str]], Any] | None = None,
) -> dict[str, Any]:
    """Run the future exact-delta mutation workflow without retry or rollback."""

    phase_a, failure = _phase_a(config, observations)
    if failure is not None or phase_a is None:
        return _failure(failure or PROVENANCE_BINDING_FAILURE)
    if mutation_authority_token != FRESH_MUTATION_AUTHORITY_TOKEN:
        return _failure(MUTATION_AUTHORITY_MISSING, phase_a=phase_a)

    # The wheel gate is intentionally repeated after authority presentation
    # and immediately before the launch boundary.  It has not consumed
    # authority yet and creates no durable attempt root on failure.
    try:
        second_wheel_result, second_argv = preflight._derive_wheelhouse(
            config.preflight_config, phase_a.candidate
        )
        second_manifest = _validate_manifest_binding(config, phase_a.candidate, second_wheel_result)
    except (ContractValidationError, OSError, TypeError, ValueError):
        return _failure(REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE, phase_a=phase_a)
    if (
        not second_wheel_result.get("ok")
        or tuple(second_argv) != phase_a.install_argv
        or _wheel_signature(second_wheel_result, second_manifest)
        != _wheel_signature(phase_a.wheel_result, phase_a.wheel_manifest)
    ):
        return _failure(REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE, phase_a=phase_a)

    launcher = process_runner or _launch_child
    return _launch_phase_b(config, phase_a, launcher)


def validate_attempt_state(state: Mapping[str, Any]) -> None:
    """Validate the strict durable state shape used by synthetic tests/audits."""

    required = {
        "schema_version",
        "expected_current_head",
        "mutation_runner_blob_sha1",
        "step3_receipt_sha256",
        "reviewed_successor_lock_candidate_sha256",
        "wheel_manifest",
        "delta_wheel_paths",
        "process_start_attempted",
        "process_started",
        "mutation_authority_consumed",
        "mutation_started",
        "retry_authorized",
        "process_exit_code",
        "stdout_capture_path",
        "stdout_sha256",
        "stderr_capture_path",
        "stderr_sha256",
        "failure_code",
    }
    if set(state) != required or state["schema_version"] != ATTEMPT_STATE_SCHEMA:
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    for key in (
        "process_start_attempted",
        "process_started",
        "mutation_authority_consumed",
        "mutation_started",
        "retry_authorized",
    ):
        if not isinstance(state[key], bool):
            raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    if state["retry_authorized"] is not False:
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    if not state["process_start_attempted"] and (
        state["process_started"] or state["mutation_authority_consumed"] or state["mutation_started"]
    ):
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    if state["process_start_attempted"] and (
        state["mutation_authority_consumed"] is not True or state["mutation_started"] is not True
    ):
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    if state["process_started"] and (state["mutation_authority_consumed"] is not True or state["mutation_started"] is not True):
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    exit_code = state["process_exit_code"]
    if exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool)):
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    if state["failure_code"] not in {
        None,
        NONE,
        MUTATION_PROCESS_START_FAILURE,
        MUTATION_PROCESS_FAILURE,
        MUTATION_ATTEMPT_STATE_FAILURE,
    }:
        raise MutationValidationError("MUTATION_ATTEMPT_STATE_FAILURE")
    _strict_sha(state["expected_current_head"], SHA1_RE, "state head")
    _strict_sha(state["mutation_runner_blob_sha1"], SHA1_RE, "state runner blob")
    _strict_sha(state["step3_receipt_sha256"], SHA256_RE, "state receipt SHA")
    _strict_sha(state["reviewed_successor_lock_candidate_sha256"], SHA256_RE, "state candidate SHA")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-current-head", required=True)
    parser.add_argument("--expected-mutation-runner-commit-sha", required=True)
    parser.add_argument("--expected-mutation-runner-blob-sha1", required=True)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--step3-receipt", required=True)
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument(
        "--fresh-mutation-authority-token",
        required=True,
        choices=(FRESH_MUTATION_AUTHORITY_TOKEN,),
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> MutationConfig:
    preflight_config = preflight.PreflightConfig(
        repo_root=Path(args.repo_root),
        expected_current_head=args.expected_current_head,
        expected_extension_design_sha=EXPECTED_EXTENSION_DESIGN_SHA,
        expected_reviewed_runner_commit_sha=EXPECTED_PREMUTATION_RUNNER_COMMIT_SHA,
        expected_reviewed_runner_blob_sha1=EXPECTED_PREMUTATION_RUNNER_BLOB_SHA1,
        wheelhouse=Path(args.wheelhouse),
    )
    return MutationConfig(
        preflight_config=preflight_config,
        expected_mutation_runner_commit_sha=args.expected_mutation_runner_commit_sha,
        expected_mutation_runner_blob_sha1=args.expected_mutation_runner_blob_sha1,
        attempt_root=Path(args.attempt_root),
        step3_receipt_path=Path(args.step3_receipt),
    )


def _safe_cli_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status", "FAIL"),
        "failure_code": result.get("failure_code", MUTATION_ATTEMPT_STATE_FAILURE),
        "mutation_authority_consumed": result.get("mutation_authority_consumed", False),
        "mutation_started": result.get("mutation_started", False),
        "canonical_environment_ready": False,
        "environment_frozen": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        result = run_mutation(
            config,
            observations=None,
            mutation_authority_token=args.fresh_mutation_authority_token,
            process_runner=None,
        )
    except (ContractValidationError, OSError, TypeError, ValueError):
        result = _failure(MUTATION_ATTEMPT_STATE_FAILURE)
    print(json.dumps(_safe_cli_summary(result), sort_keys=True, separators=(",", ":")))
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
