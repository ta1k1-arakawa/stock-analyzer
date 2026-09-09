"""Read-only Gate 4 provenance preflight.

This module deliberately has no network, activation, persistence, or study
execution path.  It verifies the already published seed and calendar inputs
and keeps source-payload provenance separate from selected-seed provenance.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .v7_capacity_engine import (
        V7EngineParameters,
        canonical_sha256,
        validate_single_parameter_difference,
    )
    from .v7_forward_protocol import validate_seed_rows
    from .v7_jpx_calendar import (
        CALENDAR_DEFINITION_VERSION,
        CALENDAR_SOURCE,
        CALENDAR_TIMEZONE,
        SOURCE_HOST,
        is_jpx_trading_day,
        load_calendar_snapshot,
        next_jpx_trading_day,
    )
    from .v7_seed_acquisition import validate_universe_file
except ImportError:
    from v7_capacity_engine import V7EngineParameters, canonical_sha256, validate_single_parameter_difference
    from v7_forward_protocol import validate_seed_rows
    from v7_jpx_calendar import CALENDAR_DEFINITION_VERSION, CALENDAR_SOURCE, CALENDAR_TIMEZONE, SOURCE_HOST, is_jpx_trading_day, load_calendar_snapshot, next_jpx_trading_day
    from v7_seed_acquisition import validate_universe_file


DESIGN_COMMIT = "e3e1367efd913b601a70328a815d88c20af6d147"
PREREGISTRATION_UTC = "2026-08-07T02:48:27Z"
SEED_GENERATION_COMMIT = "0facf819c14e681036d2a081db0a5208c14b7cf9"
COLLECTOR_COMMIT = "4ca41c53895e75910ae65809fea6018868929afa"
CALENDAR_COMMIT = "03ce048b0eedca632f79ad925a627cb9e967d78d"
PROSPECTIVE_BOUNDARY = "2026-08-10"
SEED_CUTOFF = "2026-08-07"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SEED_COLUMNS = (
    "ticker", "trading_date", "raw_open", "raw_high", "raw_low",
    "raw_close", "adj_close", "raw_volume",
)


class V7Gate4PreflightBlocked(ValueError):
    """Fail-closed reason for a provenance mismatch."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> tuple[str, int]:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise V7Gate4PreflightBlocked("FILE_READ_FAILED:" + str(path)) from error
    return sha256_bytes(raw), len(raw)


def _read_json(path: str | Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V7Gate4PreflightBlocked("JSON_READ_FAILED:" + str(path)) from error
    if not isinstance(value, dict):
        raise V7Gate4PreflightBlocked("JSON_NOT_OBJECT:" + str(path))
    return value, raw


def _require(value: Any, expected: Any, reason: str) -> None:
    if value != expected:
        raise V7Gate4PreflightBlocked(reason)


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value.lower()) is not None


def _valid_commit(value: Any) -> bool:
    return isinstance(value, str) and COMMIT_RE.fullmatch(value.lower()) is not None


def _utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise V7Gate4PreflightBlocked("UTC_TIMESTAMP_INVALID:" + field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise V7Gate4PreflightBlocked("UTC_TIMESTAMP_INVALID:" + field) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise V7Gate4PreflightBlocked("UTC_TIMESTAMP_MUST_BE_UTC:" + field)
    return parsed


def hash_payload_manifest(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the acquisition-side raw Yahoo provenance records (A)."""
    return sha256_bytes(canonical_json_bytes(list(records)))


def hash_ticker_manifest(ticker_manifest: Sequence[Mapping[str, Any]]) -> str:
    """Hash the selected-seed ticker CSV provenance records (B)."""
    reduced = [
        {
            "ticker": item["ticker"],
            "ticker_payload_sha256": item["ticker_payload_sha256"],
        }
        for item in ticker_manifest
    ]
    return sha256_bytes(canonical_json_bytes(reduced))


def _load_seed_rows(seed_csv: str | Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = Path(seed_csv).read_bytes()
        text = raw.decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V7Gate4PreflightBlocked("SEED_CSV_READ_FAILED") from error
    if not rows or tuple(rows[0].keys()) != SEED_COLUMNS:
        raise V7Gate4PreflightBlocked("SEED_SCHEMA_INVALID")
    converted: list[dict[str, Any]] = []
    for row in rows:
        if tuple(row.keys()) != SEED_COLUMNS:
            raise V7Gate4PreflightBlocked("SEED_SCHEMA_INVALID")
        converted_row: dict[str, Any] = dict(row)
        for field in SEED_COLUMNS[2:]:
            try:
                converted_row[field] = float(row[field])
            except (TypeError, ValueError) as error:
                raise V7Gate4PreflightBlocked("SEED_NUMERIC_INVALID:" + field) from error
        converted.append(converted_row)
    return converted, raw


def validate_seed_semantics(
    rows: Sequence[Mapping[str, Any]],
    fixed_universe: Sequence[str],
    boundary: str,
    *,
    expected_seed_csv_sha256: str | None = None,
    seed_csv_bytes: bytes | None = None,
) -> dict[str, Any]:
    if expected_seed_csv_sha256 is not None and seed_csv_bytes is not None:
        if sha256_bytes(seed_csv_bytes) != expected_seed_csv_sha256:
            raise V7Gate4PreflightBlocked("SEED_CSV_HASH_MISMATCH")
    try:
        result = validate_seed_rows(rows, fixed_universe, boundary, expected_seed_canonical_sha256=expected_seed_csv_sha256)
    except Exception as error:
        reason = getattr(error, "args", ["SEED_VALIDATION_FAILED"])[0]
        raise V7Gate4PreflightBlocked("SEED_VALIDATION_FAILED:" + str(reason)) from error
    if result.get("seed_validation_result") != "PASS":
        raise V7Gate4PreflightBlocked("SEED_VALIDATION_FAILED")
    if result["seed_payload_manifest_sha256"] != hash_ticker_manifest(result["ticker_manifest"]):
        raise V7Gate4PreflightBlocked("SEED_TICKER_MANIFEST_HASH_INTERNAL_MISMATCH")
    return result


def validate_payload_manifest_records(
    records: Sequence[Mapping[str, Any]],
    raw_dir: str | Path,
    universe_tickers: Sequence[str],
) -> dict[str, Any]:
    expected = list(universe_tickers)
    if len(records) != len(expected):
        raise V7Gate4PreflightBlocked("PAYLOAD_MANIFEST_COUNT_MISMATCH")
    observed = [str(item.get("ticker")) for item in records]
    if len(set(observed)) != len(observed):
        raise V7Gate4PreflightBlocked("PAYLOAD_MANIFEST_DUPLICATE_TICKER")
    if observed != expected:
        raise V7Gate4PreflightBlocked("PAYLOAD_MANIFEST_TICKER_ORDER_MISMATCH")
    directory = Path(raw_dir)
    try:
        files = sorted(directory.glob("*.json"), key=lambda path: path.name)
    except OSError as error:
        raise V7Gate4PreflightBlocked("RAW_DIRECTORY_READ_FAILED") from error
    if len(files) != len(expected):
        raise V7Gate4PreflightBlocked("RAW_FILE_COUNT_MISMATCH")
    if {path.stem for path in files} != set(expected):
        raise V7Gate4PreflightBlocked("RAW_TICKER_SET_MISMATCH")
    for item in records:
        ticker = str(item["ticker"])
        path = directory / (ticker + ".json")
        actual_sha, actual_count = file_sha256(path)
        if item.get("payload_sha256") != actual_sha:
            raise V7Gate4PreflightBlocked("RAW_PAYLOAD_SHA_MISMATCH:" + ticker)
        if item.get("byte_count") != actual_count:
            raise V7Gate4PreflightBlocked("RAW_PAYLOAD_BYTE_COUNT_MISMATCH:" + ticker)
    return {
        "raw_file_count": len(files),
        "payload_manifest_record_count": len(records),
        "payload_ticker_order_parity": True,
        "duplicate_payload_tickers": 0,
        "source_seed_payload_manifest_sha256": hash_payload_manifest(records),
    }


def validate_calendar_provenance(calendar: Mapping[str, Any], raw_bytes: bytes) -> dict[str, Any]:
    _require(calendar.get("calendar_source"), CALENDAR_SOURCE, "CALENDAR_SOURCE_MISMATCH")
    _require(calendar.get("calendar_source_host"), SOURCE_HOST, "CALENDAR_HOST_MISMATCH")
    _require(calendar.get("calendar_timezone"), CALENDAR_TIMEZONE, "CALENDAR_TIMEZONE_MISMATCH")
    _require(calendar.get("calendar_definition_version"), CALENDAR_DEFINITION_VERSION, "CALENDAR_VERSION_MISMATCH")
    _require(calendar.get("covered_years"), [2026, 2027], "CALENDAR_COVERAGE_YEARS_MISMATCH")
    _require(calendar.get("coverage_start"), "2026-01-01", "CALENDAR_COVERAGE_START_MISMATCH")
    _require(calendar.get("coverage_end"), "2027-12-31", "CALENDAR_COVERAGE_END_MISMATCH")
    _require(calendar.get("study_calendar_generated"), False, "CALENDAR_STUDY_FLAG_INVALID")
    actual_sha = sha256_bytes(raw_bytes)
    if calendar.get("source_payload_sha256") != actual_sha:
        raise V7Gate4PreflightBlocked("CALENDAR_RAW_SHA_MISMATCH")
    if calendar.get("source_byte_count") != len(raw_bytes):
        raise V7Gate4PreflightBlocked("CALENDAR_RAW_BYTE_COUNT_MISMATCH")
    holidays = calendar.get("market_holidays")
    if not isinstance(holidays, list):
        raise V7Gate4PreflightBlocked("CALENDAR_HOLIDAYS_INVALID")
    counts = {year: sum(item.get("year") == year for item in holidays) for year in (2026, 2027)}
    if counts != {2026: 21, 2027: 20}:
        raise V7Gate4PreflightBlocked("CALENDAR_HOLIDAY_COUNT_MISMATCH")
    return {
        "calendar_source": CALENDAR_SOURCE,
        "calendar_source_host": SOURCE_HOST,
        "calendar_timezone": CALENDAR_TIMEZONE,
        "calendar_definition_version": CALENDAR_DEFINITION_VERSION,
        "calendar_snapshot_sha256": sha256_bytes(canonical_json_bytes(dict(calendar))),
        "calendar_source_payload_sha256": actual_sha,
        "calendar_source_byte_count": len(raw_bytes),
        "calendar_holiday_counts": counts,
    }


def validate_arm_provenance() -> dict[str, Any]:
    control = V7EngineParameters.control()
    variant = V7EngineParameters.capacity_3()
    try:
        validate_single_parameter_difference(control, variant)
    except Exception as error:
        raise V7Gate4PreflightBlocked("ARM_PARAMETER_DIFFERENCE_INVALID") from error
    shared_control = control.to_dict()
    shared_variant = variant.to_dict()
    shared_control.pop("max_open_positions")
    shared_variant.pop("max_open_positions")
    if shared_control != shared_variant:
        raise V7Gate4PreflightBlocked("SHARED_RULES_MISMATCH")
    difference = {"max_open_positions": [control.max_open_positions, variant.max_open_positions]}
    return {
        "arm_a_parameters_sha256": control.sha256(),
        "arm_b_parameters_sha256": variant.sha256(),
        "shared_rules_sha256": canonical_sha256(shared_control),
        "single_changed_parameter": "max_open_positions",
        "single_parameter_difference": difference,
    }


def validate_seed_manifest_identity(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("mode"), "PRE_ACTIVATION_SEED_ACQUISITION", "SEED_MODE_MISMATCH")
    _require(manifest.get("design_commit"), DESIGN_COMMIT, "SEED_DESIGN_COMMIT_MISMATCH")
    _require(manifest.get("collector_commit"), COLLECTOR_COMMIT, "SEED_COLLECTOR_COMMIT_MISMATCH")
    for key, expected in {
        "ticker_count": 300, "request_count": 300, "success_count": 300,
        "failed_count": 0, "retry_count": 0, "http_429_count": 0,
        "eligible_seed_ticker_count": 300, "ineligible_seed_ticker_count": 0,
        "seed_row_count": 75600, "activation_boundary_status": "NOT_SET",
        "activation_status": "NOT_ACTIVATED", "study_calendar_generated": False,
    }.items():
        _require(manifest.get(key), expected, "SEED_MANIFEST_MISMATCH:" + key)
    for key in ("seed_payload_manifest_sha256", "seed_canonical_csv_sha256", "canonical_price_rows_csv_sha256", "canonical_split_events_sha256"):
        if not _valid_sha(manifest.get(key)):
            raise V7Gate4PreflightBlocked("SEED_MANIFEST_SHA_INVALID:" + key)
    _utc(manifest.get("acquisition_started_utc"), "acquisition_started_utc")
    _utc(manifest.get("acquisition_completed_utc"), "acquisition_completed_utc")
    started = _utc(manifest["acquisition_started_utc"], "acquisition_started_utc")
    completed = _utc(manifest["acquisition_completed_utc"], "acquisition_completed_utc")
    if not _utc(PREREGISTRATION_UTC, "preregistration") < started:
        raise V7Gate4PreflightBlocked("ACQUISITION_PREREGISTRATION_ORDER_INVALID")
    if not started <= completed:
        raise V7Gate4PreflightBlocked("ACQUISITION_TIME_ORDER_INVALID")


def validate_artifact_hashes(bundle: str | Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    root = Path(bundle)
    values = {
        "canonical_price_rows_csv_sha256": file_sha256(root / "canonical_price_rows.csv")[0],
        "canonical_split_events_sha256": file_sha256(root / "canonical_split_events.json")[0],
        "seed_canonical_csv_sha256": file_sha256(root / "seed.csv")[0],
    }
    for key, actual in values.items():
        if manifest.get(key) != actual:
            raise V7Gate4PreflightBlocked("ARTIFACT_HASH_MISMATCH:" + key)
    return values


def run_gate4_preflight(
    *,
    seed_bundle: str | Path,
    calendar_json: str | Path,
    calendar_raw: str | Path,
    universe_csv: str | Path,
    prospective_boundary: str,
) -> dict[str, Any]:
    if prospective_boundary != PROSPECTIVE_BOUNDARY:
        raise V7Gate4PreflightBlocked("PROSPECTIVE_BOUNDARY_MISMATCH")
    bundle = Path(seed_bundle)
    manifest, _ = _read_json(bundle / "seed_manifest.json")
    validate_seed_manifest_identity(manifest)
    try:
        universe = validate_universe_file(universe_csv)
    except Exception as error:
        raise V7Gate4PreflightBlocked("UNIVERSE_VALIDATION_FAILED") from error
    _require(manifest.get("universe_csv_sha256"), universe["universe_csv_sha256"], "UNIVERSE_CSV_HASH_MISMATCH")
    _require(manifest.get("ticker_list_sha256"), universe["ticker_list_sha256"], "TICKER_LIST_HASH_MISMATCH")
    _require(manifest.get("ticker_count"), universe["ticker_count"], "UNIVERSE_COUNT_MISMATCH")

    payload_manifest = manifest.get("payload_manifest")
    if not isinstance(payload_manifest, list):
        raise V7Gate4PreflightBlocked("PAYLOAD_MANIFEST_MISSING")
    payload_check = validate_payload_manifest_records(payload_manifest, bundle / "raw", universe["tickers"])
    _require(payload_check["source_seed_payload_manifest_sha256"], manifest["seed_payload_manifest_sha256"], "SOURCE_PAYLOAD_MANIFEST_SHA_MISMATCH")

    artifact_hashes = validate_artifact_hashes(bundle, manifest)
    price_sha, split_sha, seed_sha = (artifact_hashes["canonical_price_rows_csv_sha256"], artifact_hashes["canonical_split_events_sha256"], artifact_hashes["seed_canonical_csv_sha256"])
    seed_bytes = (bundle / "seed.csv").read_bytes()
    seed_rows, _ = _load_seed_rows(bundle / "seed.csv")
    seed_validation = validate_seed_semantics(
        seed_rows, universe["tickers"], prospective_boundary,
        expected_seed_csv_sha256=manifest["seed_canonical_csv_sha256"], seed_csv_bytes=seed_bytes,
    )
    _require(seed_validation["ticker_count"], 300, "SEED_VALIDATION_TICKER_COUNT_MISMATCH")
    _require(seed_validation["row_count"], 75600, "SEED_VALIDATION_ROW_COUNT_MISMATCH")
    _require(seed_validation["eligible_ticker_count"], 300, "SEED_VALIDATION_ELIGIBLE_COUNT_MISMATCH")
    _require(seed_validation["ineligible_ticker_count"], 0, "SEED_VALIDATION_INELIGIBLE_COUNT_MISMATCH")
    _require(seed_validation["seed_cutoff_trading_date"], SEED_CUTOFF, "SEED_CUTOFF_MISMATCH")
    ticker_manifest_sha = seed_validation["seed_payload_manifest_sha256"]
    dates = [str(row["trading_date"]) for row in seed_rows]
    if any(value >= prospective_boundary for value in dates):
        raise V7Gate4PreflightBlocked("SEED_ROW_ON_OR_AFTER_PROSPECTIVE_BOUNDARY")
    if max(dates) != SEED_CUTOFF:
        raise V7Gate4PreflightBlocked("SEED_MAX_DATE_MISMATCH")
    rows_by_ticker: dict[str, int] = {ticker: 0 for ticker in universe["tickers"]}
    for row in seed_rows:
        rows_by_ticker[str(row["ticker"])] += 1
        if not isinstance(row["adj_close"], (int, float)) or not math.isfinite(float(row["adj_close"])) or float(row["adj_close"]) <= 0:
            raise V7Gate4PreflightBlocked("SEED_ADJ_CLOSE_INVALID")
    if set(rows_by_ticker.values()) != {252}:
        raise V7Gate4PreflightBlocked("SEED_PER_TICKER_COUNT_MISMATCH")
    started = _utc(manifest["acquisition_started_utc"], "acquisition_started_utc")
    completed = _utc(manifest["acquisition_completed_utc"], "acquisition_completed_utc")
    if not _utc(PREREGISTRATION_UTC, "preregistration") < started <= completed:
        raise V7Gate4PreflightBlocked("ACQUISITION_PREREGISTRATION_ORDER_INVALID")

    calendar, calendar_bytes = _read_json(calendar_json)
    calendar_raw_bytes = Path(calendar_raw).read_bytes()
    calendar_check = validate_calendar_provenance(calendar, calendar_raw_bytes)
    snapshot = load_calendar_snapshot(calendar)
    if next_jpx_trading_day(snapshot, SEED_CUTOFF) != prospective_boundary:
        raise V7Gate4PreflightBlocked("PROSPECTIVE_BOUNDARY_CALENDAR_MISMATCH")
    if not is_jpx_trading_day(snapshot, prospective_boundary) or is_jpx_trading_day(snapshot, "2026-08-11"):
        raise V7Gate4PreflightBlocked("PROSPECTIVE_BOUNDARY_CALENDAR_INVALID")

    arms = validate_arm_provenance()
    return {
        "status": "PASS",
        "mode": "GATE4_PROVENANCE_PREFLIGHT",
        "design_commit": DESIGN_COMMIT,
        "preregistration_utc": PREREGISTRATION_UTC,
        "seed_generation_commit": SEED_GENERATION_COMMIT,
        "collector_commit": COLLECTOR_COMMIT,
        "calendar_commit": CALENDAR_COMMIT,
        "prospective_activation_boundary": prospective_boundary,
        "activation_authorization_utc": "NOT_SET",
        "acquisition_window_jst": "UNRESOLVED_HUMAN_GATE",
        "output_root": "UNRESOLVED_HUMAN_GATE",
        "source_seed_payload_manifest_sha256": payload_check["source_seed_payload_manifest_sha256"],
        "seed_ticker_manifest_sha256": ticker_manifest_sha,
        "seed_hash_semantics_separated": True,
        "actual_seed_csv_sha256": seed_sha,
        "manifest_seed_csv_sha256": manifest["seed_canonical_csv_sha256"],
        "validate_seed_rows_canonical_sha256": seed_validation["seed_canonical_sha256"],
        "three_way_seed_canonical_sha_parity": len({seed_sha, manifest["seed_canonical_csv_sha256"], seed_validation["seed_canonical_sha256"]}) == 1,
        "raw_file_count": payload_check["raw_file_count"],
        "payload_manifest_record_count": payload_check["payload_manifest_record_count"],
        "payload_ticker_order_parity": payload_check["payload_ticker_order_parity"],
        "duplicate_payload_tickers": payload_check["duplicate_payload_tickers"],
        "ticker_count": seed_validation["ticker_count"],
        "seed_row_count": seed_validation["row_count"],
        "eligible_seed_ticker_count": seed_validation["eligible_ticker_count"],
        "ineligible_seed_ticker_count": seed_validation["ineligible_ticker_count"],
        "seed_cutoff_trading_date": SEED_CUTOFF,
        "max_seed_trading_date": max(dates),
        "rows_on_or_after_prospective_boundary": sum(value >= prospective_boundary for value in dates),
        "acquisition_started_utc": manifest["acquisition_started_utc"],
        "acquisition_completed_utc": manifest["acquisition_completed_utc"],
        "calendar_definition_version": calendar_check["calendar_definition_version"],
        "calendar_snapshot_sha256": sha256_bytes(calendar_bytes),
        "calendar_source_payload_sha256": calendar_check["calendar_source_payload_sha256"],
        "calendar_source_byte_count": calendar_check["calendar_source_byte_count"],
        "calendar_holiday_counts": calendar_check["calendar_holiday_counts"],
        "prospective_boundary_calendar_validation": "PASS",
        **arms,
        "gate4_machine_preflight_pass": True,
        "gate4_activation_ready": False,
        "activation_manifest_created": False,
        "activation_boundary_status": "NOT_SET",
        "activation_status": "NOT_ACTIVATED",
        "network_request_count": 0,
        "seed_acquisition": 0,
        "seed_modification": 0,
        "calendar_modification": 0,
        "candidate_generation": 0,
        "portfolio_simulation": 0,
        "profit_calculation": 0,
        "formal_evaluation": 0,
        "activation": 0,
        "real_order": 0,
    }


__all__ = [
    "CALENDAR_COMMIT", "COLLECTOR_COMMIT", "DESIGN_COMMIT", "PREREGISTRATION_UTC",
    "PROSPECTIVE_BOUNDARY", "SEED_CUTOFF", "SEED_GENERATION_COMMIT",
    "V7Gate4PreflightBlocked", "canonical_json_bytes", "file_sha256",
    "hash_payload_manifest", "hash_ticker_manifest", "run_gate4_preflight",
    "validate_arm_provenance", "validate_calendar_provenance",
    "validate_payload_manifest_records", "validate_seed_manifest_identity",
    "validate_artifact_hashes", "validate_seed_semantics", "sha256_bytes",
]
