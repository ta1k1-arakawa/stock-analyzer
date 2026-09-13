"""Read-only V10A calendar-boundary wrapper for the inherited V9_009 T0."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.v9_009_t0_top1_kill_screen import (  # noqa: E402
    T0DataIncompatible,
    T0ImplementationFailure,
    make_safe_result,
    run_from_cache,
    signal_grid,
    synthetic_provenance,
    validate_safe_result,
)


AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
EXPECTED_REPOSITORY_URL = "https://github.com/ta1ki-arakawa/stock-analyzer.git"
BRIDGE_DESIGN_PATH = "V10A_T0_CALENDAR_INPUT_BINDING_BRIDGE_DESIGN_DRAFT.md"
BRIDGE_DESIGN_GIT_BLOB_SHA1 = "6df95aa8354c3d335a51747ee98ed9f2741c2410"

CALENDAR_ARTIFACT_NAME = "V10A_CANONICAL_CALENDAR.json"
CALENDAR_ARTIFACT_GIT_BLOB_SHA1 = "b3d9dee8fb20abfd966400873a7f1ff18df2880b"
CALENDAR_ARTIFACT_FILE_SHA256 = "b24cab4b322a216e6cf24e55524c5b360ac4de6404f4b161014a1a2d0d64814d"
SAFE_RECEIPT_NAME = "V10A_CALENDAR_FEASIBILITY_SAFE_RECEIPT.json"
SAFE_RECEIPT_GIT_BLOB_SHA1 = "da76889db285062a8f9ac902263ed7c2e63dc43a"
SAFE_RECEIPT_FILE_SHA256 = "e7266539c8a11c59d3be775623fde6023e835940e981f17c03f6e0bdf65b005c"

CANONICAL_CALENDAR_SHA256 = "2e9fbfbf64777d448e5a98dd85d5bb4c679cd22b19b80a07b78deac9aad507e0"
RUNTIME_ENVIRONMENT_LOCK_SHA256 = "d7f54bc69029ba9b25a9920e867fe6487745af6ef985898bad91bd951003fc3a"
GENERATOR_IMPLEMENTATION_SHA = "0830d86675f447231e77b1687c7a23cf0b135d7f"
DESIGN_GIT_SHA = "b14cc5510685210e928000af0815e188bc1aadc0"
EXPECTED_COVERAGE_START = "2017-01-01"
EXPECTED_COVERAGE_END = "2026-01-31"
EXPECTED_TRADING_DATE_COUNT = 2217

IMPLEMENTATION_FAILURE = "V9_009_T0_TOP1_KILL_SCREEN_IMPLEMENTATION_FAILURE"
PREFLIGHT_FAILURE = "V10A_T0_CALENDAR_INPUT_BINDING_PREFLIGHT_FAILURE"


class GovernanceFailure(RuntimeError):
    """A repository/provenance failure that cannot become a T0 result."""

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")

_ARTIFACT_KEYS = {
    "schema_version",
    "calendar_method",
    "calendar_package",
    "calendar_package_version",
    "upstream_commit",
    "calendar_source_blob",
    "holiday_source_blob",
    "calendar_name",
    "runtime_environment_lock_sha256",
    "coverage_start",
    "coverage_end",
    "trading_dates",
    "trading_date_count",
    "python_version",
    "pandas_version",
    "canonical_calendar_sha256",
    "generator_implementation_git_sha",
}
_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "failure_code",
    "design_git_sha",
    "generator_implementation_git_sha",
    "runtime_environment_lock_sha256",
    "calendar_name",
    "coverage_start",
    "coverage_end",
    "calendar_artifact_created",
    "canonical_calendar_sha256",
    "trading_date_count",
    "anchor_2020_10_01",
    "anchor_2020_10_02",
    "research_data_network_requests",
    "historical_calendar_data_acquisition",
    "private_or_sealed_reads",
    "human_gate_consumed",
    "t0_run",
}


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha40(value: object) -> bool:
    return isinstance(value, str) and bool(_HEX40.fullmatch(value))


def _sha64(value: object) -> bool:
    return isinstance(value, str) and bool(_HEX64.fullmatch(value))


def _contract_failure(reason: str) -> T0DataIncompatible:
    return T0DataIncompatible(reason)


def _fixed_repo_file(repository_root: Path, name: str) -> Path:
    try:
        root = repository_root.resolve(strict=True)
        candidate = root / name
        file_stat = candidate.lstat()
        if candidate.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
            raise _contract_failure("FIXED_INPUT_PATH_UNSAFE")
        if os.name == "nt" and getattr(file_stat, "st_file_attributes", 0) & 0x400:
            raise _contract_failure("FIXED_INPUT_PATH_REPARSE_POINT")
        resolved = candidate.resolve(strict=True)
        if resolved != candidate or root not in resolved.parents:
            raise _contract_failure("FIXED_INPUT_PATH_ESCAPE")
        return candidate
    except T0DataIncompatible:
        raise
    except (OSError, RuntimeError) as error:
        raise _contract_failure("FIXED_INPUT_MISSING") from error


def _git_value(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("GIT_PROVENANCE_UNAVAILABLE") from error
    return completed.stdout.strip()


def validate_repository_preflight(repository_root: Path, implementation_sha: str) -> None:
    if not _sha40(implementation_sha):
        raise GovernanceFailure("IMPLEMENTATION_SHA_INVALID")
    try:
        if _git_value(repository_root, "remote", "get-url", "origin") != EXPECTED_REPOSITORY_URL:
            raise GovernanceFailure("REPOSITORY_IDENTITY_MISMATCH")
        if _git_value(repository_root, "rev-parse", "--abbrev-ref", "HEAD") != AUTHORITATIVE_BRANCH:
            raise GovernanceFailure("BRANCH_MISMATCH")
        if _git_value(repository_root, "rev-parse", "HEAD") != implementation_sha:
            raise GovernanceFailure("IMPLEMENTATION_HEAD_MISMATCH")
        if _git_value(repository_root, "status", "--porcelain", "--untracked-files=all"):
            raise GovernanceFailure("WORKTREE_DIRTY")
        if _git_value(repository_root, "rev-parse", f"HEAD:{BRIDGE_DESIGN_PATH}") != BRIDGE_DESIGN_GIT_BLOB_SHA1:
            raise GovernanceFailure("BRIDGE_DESIGN_PROVENANCE_MISMATCH")
    except RuntimeError as error:
        if isinstance(error, GovernanceFailure):
            raise
        raise GovernanceFailure("REPOSITORY_PROVENANCE_UNAVAILABLE") from error


def _parse_json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _contract_failure("FIXED_INPUT_JSON_INVALID") from error
    if not isinstance(value, dict):
        raise _contract_failure("FIXED_INPUT_SCHEMA_INVALID")
    return value


def _validate_date(value: object) -> bool:
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        return False
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def validate_canonical_artifact(value: Mapping[str, Any]) -> list[str]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_KEYS:
        raise _contract_failure("ARTIFACT_SCHEMA_INVALID")
    expected_scalars = {
        "schema_version": "V10A_CANONICAL_CALENDAR_V1",
        "calendar_method": "PANDAS_MARKET_CALENDARS_JPX_5_4_0",
        "calendar_package": "pandas_market_calendars",
        "calendar_package_version": "5.4.0",
        "upstream_commit": "275890784073a3a3a347e4f05f4dc986456e6a75",
        "calendar_source_blob": "a7a59b6cf910e325c85fc042459ff57ca8f70613",
        "holiday_source_blob": "4c34214d06862e02ac22e946757463f748074fde",
        "calendar_name": "JPX",
        "runtime_environment_lock_sha256": RUNTIME_ENVIRONMENT_LOCK_SHA256,
        "coverage_start": EXPECTED_COVERAGE_START,
        "coverage_end": EXPECTED_COVERAGE_END,
        "python_version": "3.12.10",
        "pandas_version": "3.0.5",
    }
    for key, expected in expected_scalars.items():
        if value[key] != expected:
            raise _contract_failure("ARTIFACT_PROVENANCE_MISMATCH")
    if not _sha40(value["generator_implementation_git_sha"]):
        raise _contract_failure("ARTIFACT_IMPLEMENTATION_SHA_INVALID")
    if value["generator_implementation_git_sha"] != GENERATOR_IMPLEMENTATION_SHA:
        raise _contract_failure("ARTIFACT_IMPLEMENTATION_SHA_MISMATCH")
    if not _sha64(value["canonical_calendar_sha256"]):
        raise _contract_failure("ARTIFACT_DIGEST_INVALID")
    count = value["trading_date_count"]
    dates = value["trading_dates"]
    if isinstance(count, bool) or not isinstance(count, int) or count != EXPECTED_TRADING_DATE_COUNT:
        raise _contract_failure("ARTIFACT_DATE_COUNT_INVALID")
    if not isinstance(dates, list) or len(dates) != count:
        raise _contract_failure("ARTIFACT_DATE_COUNT_INVALID")
    if any(not _validate_date(item) for item in dates):
        raise _contract_failure("ARTIFACT_DATE_INVALID")
    if dates != sorted(set(dates)) or len(set(dates)) != len(dates):
        raise _contract_failure("ARTIFACT_DATE_ORDER_INVALID")
    if dates[0] < EXPECTED_COVERAGE_START or dates[-1] > EXPECTED_COVERAGE_END:
        raise _contract_failure("ARTIFACT_COVERAGE_INVALID")
    without_digest = dict(value)
    without_digest.pop("canonical_calendar_sha256")
    if _sha256(_canonical_json_bytes(without_digest)) != value["canonical_calendar_sha256"]:
        raise _contract_failure("ARTIFACT_SELF_DIGEST_MISMATCH")
    return list(dates)


def validate_safe_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RECEIPT_KEYS:
        raise _contract_failure("RECEIPT_SCHEMA_INVALID")
    expected = {
        "schema_version": "V10A_CALENDAR_FEASIBILITY_SAFE_RECEIPT_V1",
        "status": "PASS",
        "failure_code": "NONE",
        "design_git_sha": DESIGN_GIT_SHA,
        "generator_implementation_git_sha": GENERATOR_IMPLEMENTATION_SHA,
        "runtime_environment_lock_sha256": RUNTIME_ENVIRONMENT_LOCK_SHA256,
        "calendar_name": "JPX",
        "coverage_start": EXPECTED_COVERAGE_START,
        "coverage_end": EXPECTED_COVERAGE_END,
        "calendar_artifact_created": True,
        "canonical_calendar_sha256": CANONICAL_CALENDAR_SHA256,
        "trading_date_count": EXPECTED_TRADING_DATE_COUNT,
        "anchor_2020_10_01": "INELIGIBLE",
        "anchor_2020_10_02": "ELIGIBLE",
        "research_data_network_requests": 0,
        "historical_calendar_data_acquisition": 0,
        "private_or_sealed_reads": 0,
        "human_gate_consumed": 0,
        "t0_run": "NOT_RUN",
    }
    for key, required in expected.items():
        if value[key] != required:
            raise _contract_failure("RECEIPT_SEMANTICS_INVALID")
    if not _sha40(value["generator_implementation_git_sha"]):
        raise _contract_failure("RECEIPT_IMPLEMENTATION_SHA_INVALID")
    if not _sha64(value["canonical_calendar_sha256"]):
        raise _contract_failure("RECEIPT_DIGEST_INVALID")
    count = value["trading_date_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != EXPECTED_TRADING_DATE_COUNT:
        raise _contract_failure("RECEIPT_DATE_COUNT_INVALID")
    return dict(value)


def validate_calendar_cross_binding(artifact: Mapping[str, Any], receipt: Mapping[str, Any]) -> list[str]:
    dates = validate_canonical_artifact(artifact)
    validated_receipt = validate_safe_receipt(receipt)
    if validated_receipt["canonical_calendar_sha256"] != artifact["canonical_calendar_sha256"]:
        raise _contract_failure("ARTIFACT_RECEIPT_DIGEST_MISMATCH")
    if validated_receipt["trading_date_count"] != artifact["trading_date_count"] != len(dates):
        raise _contract_failure("ARTIFACT_RECEIPT_COUNT_MISMATCH")
    if validated_receipt["runtime_environment_lock_sha256"] != artifact["runtime_environment_lock_sha256"]:
        raise _contract_failure("ARTIFACT_RECEIPT_RUNTIME_LOCK_MISMATCH")
    if validated_receipt["generator_implementation_git_sha"] != artifact["generator_implementation_git_sha"]:
        raise _contract_failure("ARTIFACT_RECEIPT_GENERATOR_MISMATCH")
    if dates.count("2020-10-01") != 0 or dates.count("2020-10-02") != 1:
        raise _contract_failure("ANCHOR_BINDING_MISMATCH")
    return dates


def load_fixed_calendar_binding(repository_root: Path, implementation_sha: str) -> list[str]:
    artifact_path = _fixed_repo_file(repository_root, CALENDAR_ARTIFACT_NAME)
    receipt_path = _fixed_repo_file(repository_root, SAFE_RECEIPT_NAME)
    artifact_raw = artifact_path.read_bytes()
    receipt_raw = receipt_path.read_bytes()
    if _sha256(artifact_raw) != CALENDAR_ARTIFACT_FILE_SHA256:
        raise _contract_failure("ARTIFACT_FILE_SHA_MISMATCH")
    if _sha256(receipt_raw) != SAFE_RECEIPT_FILE_SHA256:
        raise _contract_failure("RECEIPT_FILE_SHA_MISMATCH")
    if _git_value(repository_root, "rev-parse", f"HEAD:{CALENDAR_ARTIFACT_NAME}") != CALENDAR_ARTIFACT_GIT_BLOB_SHA1:
        raise _contract_failure("ARTIFACT_GIT_BLOB_MISMATCH")
    if _git_value(repository_root, "rev-parse", f"HEAD:{SAFE_RECEIPT_NAME}") != SAFE_RECEIPT_GIT_BLOB_SHA1:
        raise _contract_failure("RECEIPT_GIT_BLOB_MISMATCH")
    artifact = _parse_json(artifact_raw)
    receipt = _parse_json(receipt_raw)
    dates = validate_calendar_cross_binding(artifact, receipt)
    try:
        signal_grid(dates)
    except T0DataIncompatible:
        raise
    return dates


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the read-only V9 TOP1 kill screen")
    parser.add_argument("--training-cache", required=True)
    parser.add_argument("--evaluation-cache", required=True)
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--implementation-sha", required=True)
    return parser.parse_args(argv)


def _write_safe_result(result: Mapping[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    try:
        args = _arguments(argv)
        validate_repository_preflight(REPOSITORY_ROOT, args.implementation_sha)
        calendar_dates = load_fixed_calendar_binding(REPOSITORY_ROOT, args.implementation_sha)
        try:
            result = run_from_cache(
                args.training_cache,
                args.evaluation_cache,
                args.universe_csv,
                calendar_dates,
                args.implementation_sha,
            )
            _write_safe_result(validate_safe_result(result))
            return 0
        except T0DataIncompatible:
            _write_safe_result(
                make_safe_result(
                    "NO_VERDICT_DATA_INCOMPATIBLE",
                    args.implementation_sha,
                    synthetic_provenance(),
                    cache_identity=False,
                    exact_semantics=False,
                )
            )
            return 0
    except GovernanceFailure:
        sys.stderr.write(PREFLIGHT_FAILURE + "\n")
        return 4
    except T0DataIncompatible:
        _write_safe_result(
            make_safe_result(
                "NO_VERDICT_DATA_INCOMPATIBLE",
                args.implementation_sha,  # type: ignore[name-defined]
                synthetic_provenance(),
                cache_identity=False,
                exact_semantics=False,
            )
        )
        return 0
    except RuntimeError:
        sys.stderr.write(IMPLEMENTATION_FAILURE + "\n")
        return 3
    except Exception:
        sys.stderr.write(IMPLEMENTATION_FAILURE + "\n")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
