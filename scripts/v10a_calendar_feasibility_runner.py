"""V10A fixed JPX calendar-feasibility runner.

The production entrypoint is intentionally dormant until a separately
authorized execution.  Pure schedule validation is exposed for synthetic
tests; the pinned calendar package is imported only inside the production
generator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


CANONICAL_ARTIFACT_NAME = "V10A_CANONICAL_CALENDAR.json"
SAFE_RECEIPT_NAME = "V10A_CALENDAR_FEASIBILITY_SAFE_RECEIPT.json"
CANONICAL_SCHEMA = "V10A_CANONICAL_CALENDAR_V1"
SAFE_RECEIPT_SCHEMA = "V10A_CALENDAR_FEASIBILITY_SAFE_RECEIPT_V1"

COVERAGE_START = "2017-01-01"
COVERAGE_END = "2026-01-31"
CALENDAR_NAME = "JPX"
CALENDAR_METHOD = "PANDAS_MARKET_CALENDARS_JPX_5_4_0"
CALENDAR_PACKAGE = "pandas_market_calendars"
CALENDAR_PACKAGE_VERSION = "5.4.0"
UPSTREAM_COMMIT = "275890784073a3a3a347e4f05f4dc986456e6a75"
CALENDAR_SOURCE_BLOB = "a7a59b6cf910e325c85fc042459ff57ca8f70613"
HOLIDAY_SOURCE_BLOB = "4c34214d06862e02ac22e946757463f748074fde"
LEGACY_V10_CALENDAR_SOURCE_BLOB = "0c2041b1300d1dbbd505202b00ac0ada38c712e1"
RUNTIME_LOCK_SHA256 = "d7f54bc69029ba9b25a9920e867fe6487745af6ef985898bad91bd951003fc3a"
DESIGN_GIT_SHA = "b14cc5510685210e928000af0815e188bc1aadc0"
PYTHON_VERSION = "3.12.10"
PANDAS_VERSION = "3.0.5"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")

FAILURE_CODES = (
    "NONE",
    "RUNTIME_CALENDAR_PROVENANCE_MISMATCH",
    "CALENDAR_GENERATOR_FAILURE",
    "DUPLICATE_SESSION_LABEL",
    "MALFORMED_SESSION_LABEL",
    "OUT_OF_COVERAGE_SESSION_LABEL",
    "INVALID_MARKET_CLOSE",
    "ANCHOR_2020_10_01_FAILURE",
    "ANCHOR_2020_10_02_FAILURE",
    "CANONICALIZATION_FAILURE",
    "DURABLE_ARTIFACT_WRITE_FAILURE",
)

ARTIFACT_KEYS = (
    "schema_version", "calendar_method", "calendar_package",
    "calendar_package_version", "upstream_commit", "calendar_source_blob",
    "holiday_source_blob", "calendar_name", "runtime_environment_lock_sha256",
    "coverage_start", "coverage_end", "trading_dates", "trading_date_count",
    "python_version", "pandas_version", "canonical_calendar_sha256",
    "generator_implementation_git_sha",
)
RECEIPT_KEYS = (
    "schema_version", "status", "failure_code", "design_git_sha",
    "generator_implementation_git_sha", "runtime_environment_lock_sha256",
    "calendar_name", "coverage_start", "coverage_end", "calendar_artifact_created",
    "canonical_calendar_sha256", "trading_date_count", "anchor_2020_10_01",
    "anchor_2020_10_02", "research_data_network_requests",
    "historical_calendar_data_acquisition", "private_or_sealed_reads",
    "human_gate_consumed", "t0_run",
)


class CalendarFeasibilityError(ValueError):
    def __init__(self, code: str, anchor_2020_10_01: str = "NOT_CHECKED", anchor_2020_10_02: str = "NOT_CHECKED") -> None:
        if code not in FAILURE_CODES or code == "NONE":
            raise ValueError("invalid feasibility failure code")
        self.code = code
        self.anchor_2020_10_01 = anchor_2020_10_01
        self.anchor_2020_10_02 = anchor_2020_10_02
        super().__init__(code)


@dataclass(frozen=True)
class ScheduleResult:
    trading_dates: tuple[str, ...]
    anchor_2020_10_01: str
    anchor_2020_10_02: str


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha1(value: str) -> None:
    if not isinstance(value, str) or not SHA1_RE.fullmatch(value):
        raise CalendarFeasibilityError("RUNTIME_CALENDAR_PROVENANCE_MISMATCH")


def verify_runtime_lock_bytes(raw: bytes) -> None:
    """Validate only the fixed provenance needed before calendar generation."""
    try:
        lock = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalendarFeasibilityError("RUNTIME_CALENDAR_PROVENANCE_MISMATCH") from exc
    required = {
        "schema_version": "V10A_RUNTIME_ENVIRONMENT_LOCK_V1",
        "python_version": PYTHON_VERSION,
        "calendar_distribution_name": CALENDAR_PACKAGE,
        "calendar_distribution_version": CALENDAR_PACKAGE_VERSION,
        "calendar_name": CALENDAR_NAME,
        "calendar_source_blob": CALENDAR_SOURCE_BLOB,
        "holiday_source_blob": HOLIDAY_SOURCE_BLOB,
        "runtime_distribution_count": 20,
    }
    if _sha256(raw) != RUNTIME_LOCK_SHA256 or any(lock.get(key) != value for key, value in required.items()):
        raise CalendarFeasibilityError("RUNTIME_CALENDAR_PROVENANCE_MISMATCH")
    if lock.get("calendar_source_blob") == LEGACY_V10_CALENDAR_SOURCE_BLOB:
        raise CalendarFeasibilityError("RUNTIME_CALENDAR_PROVENANCE_MISMATCH")


def _canonical_label(label: Any) -> str | None:
    try:
        timestamp = pd.Timestamp(label)
    except (TypeError, ValueError, OverflowError):
        return None
    if pd.isna(timestamp) or timestamp.tzinfo is not None:
        return None
    if any((timestamp.hour, timestamp.minute, timestamp.second, timestamp.microsecond, timestamp.nanosecond)):
        return None
    rendered = timestamp.date().isoformat()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", rendered):
        return None
    return rendered


def validate_schedule(schedule: pd.DataFrame) -> ScheduleResult:
    """Apply the frozen V10/V10A session and anchor semantics to a schedule."""
    if not isinstance(schedule, pd.DataFrame):
        raise CalendarFeasibilityError("CALENDAR_GENERATOR_FAILURE")

    labels = [_canonical_label(label) for label in schedule.index]
    valid_labels = [label for label in labels if label is not None]
    if len(set(valid_labels)) != len(valid_labels):
        raise CalendarFeasibilityError("DUPLICATE_SESSION_LABEL")
    if any(label is None for label in labels):
        raise CalendarFeasibilityError("MALFORMED_SESSION_LABEL")
    if any(label < COVERAGE_START or label > COVERAGE_END for label in valid_labels):
        raise CalendarFeasibilityError("OUT_OF_COVERAGE_SESSION_LABEL")

    if "market_close" not in schedule.columns:
        raise CalendarFeasibilityError("INVALID_MARKET_CLOSE")

    invalid_close = False
    for value in schedule["market_close"]:
        try:
            close = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            invalid_close = True
            continue
        if pd.isna(close) or close.tzinfo is None:
            invalid_close = True
    if invalid_close:
        raise CalendarFeasibilityError("INVALID_MARKET_CLOSE")

    dates = tuple(sorted(valid_labels))
    first = "ELIGIBLE" if "2020-10-01" in dates else "INELIGIBLE"
    if first != "INELIGIBLE":
        raise CalendarFeasibilityError("ANCHOR_2020_10_01_FAILURE", first, "NOT_CHECKED")
    second = "ELIGIBLE" if "2020-10-02" in dates else "INELIGIBLE"
    if second != "ELIGIBLE":
        raise CalendarFeasibilityError("ANCHOR_2020_10_02_FAILURE", first, second)
    return ScheduleResult(dates, first, second)


def build_canonical_artifact(trading_dates: tuple[str, ...], generator_implementation_git_sha: str) -> dict[str, Any]:
    _require_sha1(generator_implementation_git_sha)
    if not trading_dates or tuple(sorted(set(trading_dates))) != trading_dates:
        raise CalendarFeasibilityError("CANONICALIZATION_FAILURE")
    artifact: dict[str, Any] = {
        "schema_version": CANONICAL_SCHEMA,
        "calendar_method": CALENDAR_METHOD,
        "calendar_package": CALENDAR_PACKAGE,
        "calendar_package_version": CALENDAR_PACKAGE_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "calendar_source_blob": CALENDAR_SOURCE_BLOB,
        "holiday_source_blob": HOLIDAY_SOURCE_BLOB,
        "calendar_name": CALENDAR_NAME,
        "runtime_environment_lock_sha256": RUNTIME_LOCK_SHA256,
        "coverage_start": COVERAGE_START,
        "coverage_end": COVERAGE_END,
        "trading_dates": list(trading_dates),
        "trading_date_count": len(trading_dates),
        "python_version": PYTHON_VERSION,
        "pandas_version": PANDAS_VERSION,
        "generator_implementation_git_sha": generator_implementation_git_sha,
    }
    digest = _sha256(canonical_json_bytes(artifact))
    artifact["canonical_calendar_sha256"] = digest
    validate_canonical_artifact(artifact)
    return artifact


def validate_canonical_artifact(artifact: Mapping[str, Any]) -> None:
    if set(artifact) != set(ARTIFACT_KEYS):
        raise CalendarFeasibilityError("CANONICALIZATION_FAILURE")
    digest = artifact.get("canonical_calendar_sha256")
    without_digest = dict(artifact)
    without_digest.pop("canonical_calendar_sha256", None)
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CalendarFeasibilityError("CANONICALIZATION_FAILURE")
    if _sha256(canonical_json_bytes(without_digest)) != digest:
        raise CalendarFeasibilityError("CANONICALIZATION_FAILURE")
    dates = artifact["trading_dates"]
    count = artifact["trading_date_count"]
    if (not isinstance(dates, list) or type(count) is not int or count <= 0
            or count != len(dates) or dates != sorted(set(dates))):
        raise CalendarFeasibilityError("CANONICALIZATION_FAILURE")
    fixed = {
        "schema_version": CANONICAL_SCHEMA,
        "calendar_method": CALENDAR_METHOD,
        "calendar_package": CALENDAR_PACKAGE,
        "calendar_package_version": CALENDAR_PACKAGE_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "calendar_source_blob": CALENDAR_SOURCE_BLOB,
        "holiday_source_blob": HOLIDAY_SOURCE_BLOB,
        "calendar_name": CALENDAR_NAME,
        "runtime_environment_lock_sha256": RUNTIME_LOCK_SHA256,
        "coverage_start": COVERAGE_START,
        "coverage_end": COVERAGE_END,
        "python_version": PYTHON_VERSION,
        "pandas_version": PANDAS_VERSION,
    }
    if any(artifact[key] != value for key, value in fixed.items()):
        raise CalendarFeasibilityError("CANONICALIZATION_FAILURE")
    if (not isinstance(artifact["generator_implementation_git_sha"], str)
            or not SHA1_RE.fullmatch(artifact["generator_implementation_git_sha"])):
        raise CalendarFeasibilityError("CANONICALIZATION_FAILURE")
    if any(
        not isinstance(date, str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date)
        or _canonical_label(date) != date
        or date < COVERAGE_START
        or date > COVERAGE_END
        for date in dates
    ):
        raise CalendarFeasibilityError("CANONICALIZATION_FAILURE")


def build_safe_receipt(
    generator_implementation_git_sha: str,
    *,
    status: str,
    failure_code: str,
    calendar_artifact_created: bool,
    canonical_calendar_sha256: str | None,
    trading_date_count: int | None,
    anchor_2020_10_01: str,
    anchor_2020_10_02: str,
) -> dict[str, Any]:
    _require_sha1(generator_implementation_git_sha)
    if status not in {"PASS", "FAIL"} or failure_code not in FAILURE_CODES:
        raise ValueError("invalid receipt status")
    receipt = {
        "schema_version": SAFE_RECEIPT_SCHEMA,
        "status": status,
        "failure_code": failure_code,
        "design_git_sha": DESIGN_GIT_SHA,
        "generator_implementation_git_sha": generator_implementation_git_sha,
        "runtime_environment_lock_sha256": RUNTIME_LOCK_SHA256,
        "calendar_name": CALENDAR_NAME,
        "coverage_start": COVERAGE_START,
        "coverage_end": COVERAGE_END,
        "calendar_artifact_created": calendar_artifact_created,
        "canonical_calendar_sha256": canonical_calendar_sha256,
        "trading_date_count": trading_date_count,
        "anchor_2020_10_01": anchor_2020_10_01,
        "anchor_2020_10_02": anchor_2020_10_02,
        "research_data_network_requests": 0,
        "historical_calendar_data_acquisition": 0,
        "private_or_sealed_reads": 0,
        "human_gate_consumed": 0,
        "t0_run": "NOT_RUN",
    }
    validate_safe_receipt(receipt)
    return receipt


def validate_safe_receipt(receipt: Mapping[str, Any]) -> None:
    if set(receipt) != set(RECEIPT_KEYS):
        raise ValueError("receipt field set mismatch")
    fixed = {
        "schema_version": SAFE_RECEIPT_SCHEMA,
        "design_git_sha": DESIGN_GIT_SHA,
        "runtime_environment_lock_sha256": RUNTIME_LOCK_SHA256,
        "calendar_name": CALENDAR_NAME,
        "coverage_start": COVERAGE_START,
        "coverage_end": COVERAGE_END,
        "research_data_network_requests": 0,
        "historical_calendar_data_acquisition": 0,
        "private_or_sealed_reads": 0,
        "human_gate_consumed": 0,
        "t0_run": "NOT_RUN",
    }
    if any(receipt[key] != value for key, value in fixed.items()):
        raise ValueError("receipt fixed value mismatch")
    if receipt["status"] not in {"PASS", "FAIL"} or receipt["failure_code"] not in FAILURE_CODES:
        raise ValueError("receipt status mismatch")
    if (not isinstance(receipt["generator_implementation_git_sha"], str)
            or not SHA1_RE.fullmatch(receipt["generator_implementation_git_sha"])):
        raise ValueError("receipt implementation SHA mismatch")
    digest = receipt["canonical_calendar_sha256"]
    count = receipt["trading_date_count"]
    if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
        raise ValueError("receipt calendar hash mismatch")
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError("receipt trading-date count mismatch")
    if receipt["status"] == "PASS":
        if (receipt["failure_code"] != "NONE" or not receipt["calendar_artifact_created"]
                or digest is None or count is None
                or receipt["anchor_2020_10_01"] != "INELIGIBLE"
                or receipt["anchor_2020_10_02"] != "ELIGIBLE"):
            raise ValueError("receipt pass semantics mismatch")
    elif receipt["failure_code"] == "NONE" or receipt["calendar_artifact_created"]:
        raise ValueError("receipt failure semantics mismatch")
    elif receipt["failure_code"] == "DURABLE_ARTIFACT_WRITE_FAILURE":
        if digest is None or count is None:
            raise ValueError("receipt durable-write failure semantics mismatch")
    elif digest is not None or count is not None:
        raise ValueError("receipt pre-artifact failure semantics mismatch")
    if receipt["anchor_2020_10_01"] not in {"INELIGIBLE", "ELIGIBLE", "NOT_CHECKED"}:
        raise ValueError("invalid first anchor")
    if receipt["anchor_2020_10_02"] not in {"INELIGIBLE", "ELIGIBLE", "NOT_CHECKED"}:
        raise ValueError("invalid second anchor")
    if (receipt["anchor_2020_10_01"] == "NOT_CHECKED"
            and receipt["anchor_2020_10_02"] != "NOT_CHECKED"):
        raise ValueError("anchor ordering mismatch")
    if (receipt["anchor_2020_10_01"] == "ELIGIBLE"
            and receipt["anchor_2020_10_02"] != "NOT_CHECKED"):
        raise ValueError("anchor ordering mismatch")


def validate_persisted_pass_artifacts(
    artifact: Mapping[str, Any], receipt: Mapping[str, Any], expected_generator_implementation_git_sha: str,
) -> None:
    """Pure Phase-C-style validation of a persisted PASS artifact pair."""
    if (not isinstance(expected_generator_implementation_git_sha, str)
            or not SHA1_RE.fullmatch(expected_generator_implementation_git_sha)):
        raise ValueError("expected implementation SHA mismatch")
    validate_canonical_artifact(artifact)
    validate_safe_receipt(receipt)
    if receipt["status"] != "PASS" or receipt["failure_code"] != "NONE":
        raise ValueError("persisted receipt is not PASS")
    if artifact["canonical_calendar_sha256"] != receipt["canonical_calendar_sha256"]:
        raise ValueError("artifact/receipt calendar hash mismatch")
    if (artifact["trading_date_count"] != receipt["trading_date_count"]
            or artifact["trading_date_count"] != len(artifact["trading_dates"])):
        raise ValueError("artifact/receipt trading-date count mismatch")
    if artifact["runtime_environment_lock_sha256"] != receipt["runtime_environment_lock_sha256"]:
        raise ValueError("artifact/receipt runtime-lock mismatch")
    if artifact["generator_implementation_git_sha"] != receipt["generator_implementation_git_sha"]:
        raise ValueError("artifact/receipt implementation SHA mismatch")
    if artifact["generator_implementation_git_sha"] != expected_generator_implementation_git_sha:
        raise ValueError("unexpected implementation SHA")


def _generate_fixed_jpx_schedule() -> pd.DataFrame:
    # Delayed import: synthetic tests and ordinary module import create no calendar object.
    import pandas_market_calendars as mcal

    return mcal.get_calendar(CALENDAR_NAME).schedule(start_date=COVERAGE_START, end_date=COVERAGE_END)


def _write_new(path: Path, data: bytes) -> None:
    if path.exists():
        raise FileExistsError(path.name)
    path.write_bytes(data)


def run_feasibility(repo_root: Path, output_root: Path, generator_implementation_git_sha: str) -> dict[str, Any]:
    """Future production entrypoint; no schedule injection or alternate provider exists."""
    _require_sha1(generator_implementation_git_sha)
    if output_root.exists():
        raise FileExistsError("exclusive output root already exists")
    lock_bytes = (repo_root / "V10A_RUNTIME_ENVIRONMENT_LOCK.json").read_bytes()
    try:
        verify_runtime_lock_bytes(lock_bytes)
        schedule = _generate_fixed_jpx_schedule()
        result = validate_schedule(schedule)
        artifact = build_canonical_artifact(result.trading_dates, generator_implementation_git_sha)
    except CalendarFeasibilityError as exc:
        output_root.mkdir(parents=True)
        receipt = build_safe_receipt(generator_implementation_git_sha, status="FAIL", failure_code=exc.code,
                                     calendar_artifact_created=False, canonical_calendar_sha256=None,
                                     trading_date_count=None, anchor_2020_10_01=exc.anchor_2020_10_01,
                                     anchor_2020_10_02=exc.anchor_2020_10_02)
        _write_new(output_root / SAFE_RECEIPT_NAME, canonical_json_bytes(receipt))
        return receipt
    except Exception:
        output_root.mkdir(parents=True)
        receipt = build_safe_receipt(generator_implementation_git_sha, status="FAIL",
                                     failure_code="CALENDAR_GENERATOR_FAILURE", calendar_artifact_created=False,
                                     canonical_calendar_sha256=None, trading_date_count=None,
                                     anchor_2020_10_01="NOT_CHECKED", anchor_2020_10_02="NOT_CHECKED")
        _write_new(output_root / SAFE_RECEIPT_NAME, canonical_json_bytes(receipt))
        return receipt

    output_root.mkdir(parents=True)
    artifact_bytes = canonical_json_bytes(artifact)
    try:
        _write_new(output_root / CANONICAL_ARTIFACT_NAME, artifact_bytes)
    except OSError:
        receipt = build_safe_receipt(generator_implementation_git_sha, status="FAIL",
                                     failure_code="DURABLE_ARTIFACT_WRITE_FAILURE", calendar_artifact_created=False,
                                     canonical_calendar_sha256=artifact["canonical_calendar_sha256"],
                                     trading_date_count=artifact["trading_date_count"],
                                     anchor_2020_10_01=result.anchor_2020_10_01,
                                     anchor_2020_10_02=result.anchor_2020_10_02)
        _write_new(output_root / SAFE_RECEIPT_NAME, canonical_json_bytes(receipt))
        return receipt
    receipt = build_safe_receipt(generator_implementation_git_sha, status="PASS", failure_code="NONE",
                                 calendar_artifact_created=True,
                                 canonical_calendar_sha256=artifact["canonical_calendar_sha256"],
                                 trading_date_count=artifact["trading_date_count"],
                                 anchor_2020_10_01=result.anchor_2020_10_01,
                                 anchor_2020_10_02=result.anchor_2020_10_02)
    _write_new(output_root / SAFE_RECEIPT_NAME, canonical_json_bytes(receipt))
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V10A fixed JPX calendar feasibility runner")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generator-implementation-git-sha", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_feasibility(args.repo_root, args.output_root, args.generator_implementation_git_sha)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
