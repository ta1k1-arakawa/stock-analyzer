"""V7 Gate 4 production activation manifest contract.

This module defines, builds, validates, and (once) writes the production
activation manifest for the V7 forward capacity study.  It is a static
contract layer only:

* it performs no network request, no collector run, no daily acquisition, no
  candidate generation, no portfolio processing, and no forward-store creation;
* it never chooses an activation boundary, an authorization timestamp, an
  acquisition window, or an output root -- every one of those is supplied by
  the human Gate 4 decision and is only *validated* here;
* ``write_activation_manifest_once`` is the single function that can create a
  manifest file, it is append-only (write-once), and it requires an explicit
  human confirmation token.

The already accepted ``v7_forward_protocol.validate_activation_manifest`` is a
DRY_RUN_ONLY validator and is deliberately left untouched; this module is an
independent production contract with its own schema.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

from src.v7_capacity_engine import V7EngineParameters, validate_single_parameter_difference
from src.v7_forward_protocol import ProtocolBlocked, validate_seed_rows
from src.v7_jpx_calendar import (
    V7JpxCalendarBlocked,
    is_jpx_trading_day,
    load_calendar_snapshot,
    next_jpx_trading_day,
)
from src.v7_seed_acquisition import V7SeedAcquisitionBlocked, validate_universe_file


# ---------------------------------------------------------------------------
# Frozen identity
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "V7_FORWARD_ACTIVATION_V1"
MODE = "FORWARD_ONLY_EXPLORATORY_PAPER_STUDY"
STUDY_NAME = "V7_FORWARD_CAPACITY"
ACTIVATION_STATUS = "ACTIVATED"

DESIGN_COMMIT = "e3e1367efd913b601a70328a815d88c20af6d147"
PREREGISTRATION_UTC = "2026-08-07T02:48:27Z"
IMPLEMENTATION_COMMIT = "98b328ec905837fc1c7dfca91950529c573bc5db"
COLLECTOR_COMMIT = "4ca41c53895e75910ae65809fea6018868929afa"
CALENDAR_COMMIT = "03ce048b0eedca632f79ad925a627cb9e967d78d"
SEED_GENERATION_COMMIT = "0facf819c14e681036d2a081db0a5208c14b7cf9"

CALENDAR_SOURCE = "JPX_OFFICIAL_MARKET_HOLIDAYS"
CALENDAR_TIMEZONE = "Asia/Tokyo"
CALENDAR_DEFINITION_VERSION = "V7_JPX_OFFICIAL_MARKET_HOLIDAYS_V1"
CALENDAR_SNAPSHOT_SHA256 = "6114094de84f9f9833ceddaa9fb4a46290662423f425b3e24be1b60eb00968a0"

DATA_SOURCE = "Yahoo Chart"
DATA_SOURCE_HOST = "query1.finance.yahoo.com"
DATA_SOURCE_SCHEMA = "V7_YAHOO_CHART_DAILY_RAW_OHLCV_V1"
SEED_DATA_SOURCE = DATA_SOURCE
SEED_DATA_SCHEMA = DATA_SOURCE_SCHEMA

UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
TICKER_COUNT = 300

ARM_A_PARAMETERS_SHA256 = "0ace638e6c40a222cd5b4ca107ddf6012c1f4e40e45dd45d49f37a1673b71b41"
ARM_B_PARAMETERS_SHA256 = "d505d325d1c573595b9af26e141564f69a6ac8efdb8e6388d7eb61d50440a779"
SINGLE_CHANGED_PARAMETER = "max_open_positions"

PROHIBITION_FIELDS = {
    "historical_backtest_allowed": False,
    "historical_replay_allowed": False,
    "same_data_tuning_allowed": False,
    "real_orders_allowed": False,
    "deployment_allowed": False,
    "survivorship_bias": True,
}

ARM_PARITY_FIELDS = (
    "arm_seed_hash_equal",
    "arm_candidate_input_hash_equal",
    "arm_market_gate_input_hash_equal",
)

SEED_VALIDATION_RESULT = "PASS"

HUMAN_ACTIVATION_CONFIRMATION = "V7_GATE4_HUMAN_ACTIVATION_APPROVED"

PLACEHOLDER_VALUES = frozenset({
    "NOT_SET",
    "UNRESOLVED_HUMAN_GATE",
    "TBD",
    "TODO",
    "UNKNOWN",
    "",
})

HUMAN_DECISION_FIELDS = (
    "activation_authorization_utc",
    "activation_boundary_first_jpx_trading_date",
    "acquisition_window_jst",
    "output_root",
)

# The shared (non-differing) study rules, hashed as an explicit canonical object.
SHARED_RULES = {
    "starting_cash": 400000,
    "quantity": 100,
    "cash_reserve": 40000,
    "capital_limit_per_position": 220000,
    "same_industry_concurrent": False,
    "duplicate_ticker_concurrent": False,
    "same_day_proceeds_reuse": False,
    "entry_source": "D1_RAW_OPEN",
    "entry_gap_multiplier": 1.02,
    "entry_slippage": 0.0003,
    "exit_source": "D10_RAW_OPEN",
    "exit_slippage": 0.0003,
    "exit_reason": "TIME",
    "stop_loss": "NONE",
    "candidate_rules": "FROZEN_V6_A_R2",
    "market_gate": "FROZEN_V6_A_R2",
    "ranking_rules": "FROZEN_V6_A_R2",
    "top_candidates_per_signal_day": 20,
}

MANIFEST_FIELDS = (
    "schema_version",
    "mode",
    "study_name",
    "activation_status",
    "design_commit",
    "preregistration_utc",
    "implementation_commit",
    "collector_commit",
    "calendar_commit",
    "seed_generation_commit",
    "activation_authorization_utc",
    "activation_boundary_first_jpx_trading_date",
    "calendar_source",
    "calendar_timezone",
    "calendar_definition_version",
    "calendar_snapshot_sha256",
    "data_source",
    "data_source_host",
    "data_source_schema",
    "acquisition_window_jst",
    "universe_csv_sha256",
    "ticker_list_sha256",
    "ticker_count",
    "arm_a_parameters_sha256",
    "arm_b_parameters_sha256",
    "single_changed_parameter",
    "shared_rules_sha256",
    "output_root",
    "seed_data_source",
    "seed_data_schema",
    "seed_acquisition_utc",
    "seed_cutoff_trading_date",
    "seed_ticker_count",
    "seed_row_count",
    "seed_source_payload_manifest_sha256",
    "seed_ticker_manifest_sha256",
    "seed_canonical_csv_sha256",
    "seed_validation_result",
    "arm_seed_hash_equal",
    "arm_candidate_input_hash_equal",
    "arm_market_gate_input_hash_equal",
    "historical_backtest_allowed",
    "historical_replay_allowed",
    "same_data_tuning_allowed",
    "real_orders_allowed",
    "deployment_allowed",
    "survivorship_bias",
    "manifest_sha256",
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
WINDOW_RE = re.compile(r"^(\d{2}):(\d{2})-(\d{2}):(\d{2}) Asia/Tokyo$")
URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://")
WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:[\\/]")

WINDOW_EARLIEST_START_MINUTES = 15 * 60 + 30
WINDOW_LATEST_END_MINUTES = 23 * 60 + 59

JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc

SEED_CSV_COLUMNS = (
    "ticker",
    "trading_date",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "adj_close",
    "raw_volume",
)


class V7ActivationManifestBlocked(RuntimeError):
    """Fail-closed activation manifest contract violation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _blocked(reason: str) -> V7ActivationManifestBlocked:
    return V7ActivationManifestBlocked(reason)


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except ValueError as error:
        raise _blocked("NONFINITE_VALUE") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def shared_rules_sha256() -> str:
    return canonical_sha256(SHARED_RULES)


SHARED_RULES_SHA256 = shared_rules_sha256()


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _valid_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(COMMIT_RE.fullmatch(value))


def _parse_iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise _blocked("INVALID_DATE:" + field)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise _blocked("INVALID_DATE:" + field) from error
    if parsed.isoformat() != value:
        raise _blocked("INVALID_DATE:" + field)
    return parsed


def _parse_utc(value: Any, field: str) -> datetime:
    """Accept only an aware UTC ISO-8601 timestamp in trailing-Z form."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise _blocked("UTC_TIMESTAMP_INVALID:" + field)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _blocked("UTC_TIMESTAMP_INVALID:" + field) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise _blocked("UTC_TIMESTAMP_MUST_BE_UTC:" + field)
    return parsed


# ---------------------------------------------------------------------------
# Human-decision field guards
# ---------------------------------------------------------------------------


def require_human_decisions_resolved(values: Mapping[str, Any]) -> None:
    for field in HUMAN_DECISION_FIELDS:
        value = values.get(field)
        if value is None:
            raise _blocked("HUMAN_DECISION_UNRESOLVED:" + field)
        if not isinstance(value, str):
            raise _blocked("HUMAN_DECISION_INVALID:" + field)
        if value.strip() in PLACEHOLDER_VALUES or value.strip().upper() in PLACEHOLDER_VALUES:
            raise _blocked("HUMAN_DECISION_UNRESOLVED:" + field)


# ---------------------------------------------------------------------------
# Acquisition window
# ---------------------------------------------------------------------------


def validate_acquisition_window(value: Any) -> dict[str, int]:
    if not isinstance(value, str):
        raise _blocked("ACQUISITION_WINDOW_INVALID")
    match = WINDOW_RE.fullmatch(value)
    if match is None:
        raise _blocked("ACQUISITION_WINDOW_SYNTAX_INVALID")
    start_hour, start_minute, end_hour, end_minute = (int(part) for part in match.groups())
    if start_hour > 23 or end_hour > 23 or start_minute > 59 or end_minute > 59:
        raise _blocked("ACQUISITION_WINDOW_SYNTAX_INVALID")
    start = start_hour * 60 + start_minute
    end = end_hour * 60 + end_minute
    if start >= end:
        raise _blocked("ACQUISITION_WINDOW_ORDER_INVALID")
    if start < WINDOW_EARLIEST_START_MINUTES:
        raise _blocked("ACQUISITION_WINDOW_BEFORE_MARKET_CLOSE")
    if end > WINDOW_LATEST_END_MINUTES:
        raise _blocked("ACQUISITION_WINDOW_AFTER_DAY_END")
    return {"start_minutes": start, "end_minutes": end}


# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------


def _output_root_filesystem_path(value: str) -> tuple[Any, Any] | None:
    """Return (PurePath class, pure path) for filesystem-style roots, else None."""
    if URI_RE.match(value):
        if value.lower().startswith("file://"):
            remainder = value[len("file://"):]
            if remainder.startswith("/"):
                return PurePosixPath, PurePosixPath(remainder)
        return None
    if WINDOWS_ABSOLUTE_RE.match(value) or value.startswith("\\\\"):
        return PureWindowsPath, PureWindowsPath(value)
    if value.startswith("/"):
        return PurePosixPath, PurePosixPath(value)
    return None


def validate_output_root(value: Any, repository_root: str | os.PathLike[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _blocked("OUTPUT_ROOT_INVALID")
    if value != value.strip():
        raise _blocked("OUTPUT_ROOT_INVALID")
    if URI_RE.match(value) is None and not (
        value.startswith("/") or value.startswith("\\\\") or WINDOWS_ABSOLUTE_RE.match(value)
    ):
        raise _blocked("OUTPUT_ROOT_NOT_ABSOLUTE")

    resolved = _output_root_filesystem_path(value)
    if resolved is None:
        return value
    path_class, output_path = resolved
    try:
        repository = Path(repository_root).resolve()
    except OSError as error:
        raise _blocked("REPOSITORY_ROOT_INVALID") from error
    if path_class is PurePosixPath:
        repository_pure = PurePosixPath(repository.as_posix())
    else:
        repository_pure = PureWindowsPath(str(repository))
        if not repository_pure.is_absolute():
            return value
    if output_path == repository_pure or output_path.is_relative_to(repository_pure):
        raise _blocked("OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY")
    return value


# ---------------------------------------------------------------------------
# Calendar binding
# ---------------------------------------------------------------------------


def validate_calendar_binding(
    calendar_path: str | os.PathLike[str], expected_snapshot_sha256: str = CALENDAR_SNAPSHOT_SHA256
) -> dict[str, Any]:
    path = Path(calendar_path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise _blocked("CALENDAR_FILE_READ_FAILED") from error
    actual = sha256_bytes(raw)
    if actual != expected_snapshot_sha256:
        raise _blocked("CALENDAR_SNAPSHOT_SHA_MISMATCH")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _blocked("CALENDAR_FILE_INVALID") from error
    if payload.get("calendar_source") != CALENDAR_SOURCE:
        raise _blocked("CALENDAR_SOURCE_MISMATCH")
    if payload.get("calendar_timezone") != CALENDAR_TIMEZONE:
        raise _blocked("CALENDAR_TIMEZONE_MISMATCH")
    if payload.get("calendar_definition_version") != CALENDAR_DEFINITION_VERSION:
        raise _blocked("CALENDAR_DEFINITION_VERSION_MISMATCH")
    try:
        snapshot = load_calendar_snapshot(payload)
    except V7JpxCalendarBlocked as error:
        raise _blocked("CALENDAR_SNAPSHOT_INVALID:" + error.reason) from error
    return {"calendar_snapshot_sha256": actual, "snapshot": snapshot}


# ---------------------------------------------------------------------------
# Authorization and activation boundary
# ---------------------------------------------------------------------------


def expected_activation_boundary(snapshot: Any, activation_authorization_utc: str) -> str:
    """The first JPX trading day strictly after the JST date of human approval."""
    authorized = _parse_utc(activation_authorization_utc, "activation_authorization_utc")
    approval_jst_date = authorized.astimezone(JST).date()
    try:
        return next_jpx_trading_day(snapshot, approval_jst_date)
    except V7JpxCalendarBlocked as error:
        raise _blocked("ACTIVATION_BOUNDARY_OUTSIDE_CALENDAR_COVERAGE") from error


def validate_authorization_and_boundary(
    *,
    snapshot: Any,
    activation_authorization_utc: str,
    activation_boundary_first_jpx_trading_date: str,
    seed_acquisition_utc: str,
) -> dict[str, Any]:
    authorized = _parse_utc(activation_authorization_utc, "activation_authorization_utc")
    preregistered = _parse_utc(PREREGISTRATION_UTC, "preregistration_utc")
    seeded = _parse_utc(seed_acquisition_utc, "seed_acquisition_utc")
    if not preregistered < authorized:
        raise _blocked("AUTHORIZATION_NOT_AFTER_PREREGISTRATION")
    if not seeded < authorized:
        raise _blocked("AUTHORIZATION_NOT_AFTER_SEED_ACQUISITION")

    boundary = _parse_iso_date(
        activation_boundary_first_jpx_trading_date, "activation_boundary_first_jpx_trading_date"
    )
    try:
        trading = is_jpx_trading_day(snapshot, boundary)
    except V7JpxCalendarBlocked as error:
        raise _blocked("ACTIVATION_BOUNDARY_OUTSIDE_CALENDAR_COVERAGE") from error
    if not trading:
        raise _blocked("ACTIVATION_BOUNDARY_NOT_JPX_TRADING_DAY")

    approval_jst_date = authorized.astimezone(JST).date()
    if boundary <= approval_jst_date:
        raise _blocked("ACTIVATION_BOUNDARY_NOT_AFTER_AUTHORIZATION_JST_DATE")
    expected = expected_activation_boundary(snapshot, activation_authorization_utc)
    if activation_boundary_first_jpx_trading_date != expected:
        raise _blocked("ACTIVATION_BOUNDARY_NOT_FIRST_JPX_TRADING_DAY_AFTER_AUTHORIZATION")
    return {
        "authorization_jst_date": approval_jst_date.isoformat(),
        "expected_activation_boundary": expected,
    }


# ---------------------------------------------------------------------------
# Seed provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedProvenanceExpectation:
    """Expected seed provenance values; the default pins the actual V7 study seed."""

    seed_source_payload_manifest_sha256: str
    seed_ticker_manifest_sha256: str
    seed_canonical_csv_sha256: str
    seed_ticker_count: int
    seed_row_count: int
    seed_cutoff_trading_date: str


PRODUCTION_SEED_PROVENANCE = SeedProvenanceExpectation(
    seed_source_payload_manifest_sha256="f71446043ad88e1688069ce1f438b11fa0e5172ca5ab21e96fe679ff1b74043f",
    seed_ticker_manifest_sha256="edd06a02103f36b22552124d73f81f9826f609ea10a327d817ccd2c4281d0eff",
    seed_canonical_csv_sha256="8ac3adde3be58ea62072bb6fd7af242ba8c7c5701df1cc67ca2f3b411cde84d3",
    seed_ticker_count=300,
    seed_row_count=75600,
    seed_cutoff_trading_date="2026-08-07",
)


def hash_source_payload_manifest(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash (A): the raw Yahoo acquisition-side provenance records."""
    return canonical_sha256(list(records))


def read_seed_csv_rows(seed_csv: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], bytes]:
    path = Path(seed_csv)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise _blocked("SEED_CSV_READ_FAILED") from error
    try:
        text = raw.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        raw_rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as error:
        raise _blocked("SEED_CSV_INVALID") from error
    if not raw_rows or tuple(raw_rows[0]) != SEED_CSV_COLUMNS:
        raise _blocked("SEED_CSV_SCHEMA_INVALID")
    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        row: dict[str, Any] = {"ticker": raw_row["ticker"], "trading_date": raw_row["trading_date"]}
        for column in SEED_CSV_COLUMNS[2:]:
            try:
                row[column] = float(raw_row[column])
            except (TypeError, ValueError) as error:
                raise _blocked("SEED_CSV_NONNUMERIC:" + column) from error
        rows.append(row)
    return rows, raw


def validate_seed_provenance(
    *,
    universe_csv: str | os.PathLike[str],
    seed_csv: str | os.PathLike[str],
    seed_acquisition_manifest: Mapping[str, Any],
    activation_boundary_first_jpx_trading_date: str,
    expected: SeedProvenanceExpectation | None = PRODUCTION_SEED_PROVENANCE,
) -> dict[str, Any]:
    """Read-only re-derivation of every seed provenance value in the manifest."""
    try:
        universe = validate_universe_file(universe_csv)
    except V7SeedAcquisitionBlocked as error:
        raise _blocked("UNIVERSE_VALIDATION_FAILED:" + error.reason) from error
    if universe["universe_csv_sha256"] != UNIVERSE_CSV_SHA256:
        raise _blocked("UNIVERSE_CSV_SHA_MISMATCH")
    if universe["ticker_list_sha256"] != TICKER_LIST_SHA256:
        raise _blocked("TICKER_LIST_SHA_MISMATCH")

    rows, raw = read_seed_csv_rows(seed_csv)
    seed_canonical_csv_sha256 = sha256_bytes(raw)

    if not isinstance(seed_acquisition_manifest, Mapping):
        raise _blocked("SEED_ACQUISITION_MANIFEST_INVALID")
    payload_manifest = seed_acquisition_manifest.get("payload_manifest")
    if not isinstance(payload_manifest, list) or not payload_manifest:
        raise _blocked("SEED_SOURCE_PAYLOAD_MANIFEST_INVALID")
    seed_source_payload_manifest_sha256 = hash_source_payload_manifest(payload_manifest)

    try:
        validation = validate_seed_rows(
            rows,
            universe["tickers"],
            activation_boundary_first_jpx_trading_date,
            expected_seed_canonical_sha256=seed_canonical_csv_sha256,
        )
    except ProtocolBlocked as error:
        raise _blocked("SEED_VALIDATION_FAILED:" + str(error)) from error
    if validation["seed_validation_result"] != SEED_VALIDATION_RESULT:
        raise _blocked("SEED_VALIDATION_RESULT_NOT_PASS")

    derived = {
        "seed_source_payload_manifest_sha256": seed_source_payload_manifest_sha256,
        "seed_ticker_manifest_sha256": validation["seed_payload_manifest_sha256"],
        "seed_canonical_csv_sha256": seed_canonical_csv_sha256,
        "seed_ticker_count": validation["ticker_count"],
        "seed_row_count": validation["row_count"],
        "seed_cutoff_trading_date": validation["seed_cutoff_trading_date"],
        "seed_validation_result": SEED_VALIDATION_RESULT,
    }
    if derived["seed_source_payload_manifest_sha256"] == derived["seed_ticker_manifest_sha256"]:
        raise _blocked("SEED_HASH_SEMANTICS_COLLISION")

    if expected is not None:
        for field in (
            "seed_source_payload_manifest_sha256",
            "seed_ticker_manifest_sha256",
            "seed_canonical_csv_sha256",
            "seed_ticker_count",
            "seed_row_count",
            "seed_cutoff_trading_date",
        ):
            if derived[field] != getattr(expected, field):
                raise _blocked("SEED_PROVENANCE_MISMATCH:" + field)
    return derived


# ---------------------------------------------------------------------------
# Candidate builder (pure; no write, no network, no activation)
# ---------------------------------------------------------------------------


def build_activation_manifest_candidate(
    *,
    activation_authorization_utc: str,
    activation_boundary_first_jpx_trading_date: str,
    acquisition_window_jst: str,
    output_root: str,
    seed_acquisition_utc: str,
    seed_provenance: Mapping[str, Any],
    calendar_snapshot_sha256: str = CALENDAR_SNAPSHOT_SHA256,
) -> dict[str, Any]:
    """Return a candidate manifest dict.  Writes nothing and activates nothing."""
    require_human_decisions_resolved({
        "activation_authorization_utc": activation_authorization_utc,
        "activation_boundary_first_jpx_trading_date": activation_boundary_first_jpx_trading_date,
        "acquisition_window_jst": acquisition_window_jst,
        "output_root": output_root,
    })
    if not isinstance(seed_provenance, Mapping):
        raise _blocked("SEED_PROVENANCE_INVALID")
    missing = [
        field for field in (
            "seed_source_payload_manifest_sha256",
            "seed_ticker_manifest_sha256",
            "seed_canonical_csv_sha256",
            "seed_ticker_count",
            "seed_row_count",
            "seed_cutoff_trading_date",
        )
        if field not in seed_provenance
    ]
    if missing:
        raise _blocked("SEED_PROVENANCE_MISSING_FIELD:" + ",".join(missing))

    body = {
        "schema_version": SCHEMA_VERSION,
        "mode": MODE,
        "study_name": STUDY_NAME,
        "activation_status": ACTIVATION_STATUS,
        "design_commit": DESIGN_COMMIT,
        "preregistration_utc": PREREGISTRATION_UTC,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "collector_commit": COLLECTOR_COMMIT,
        "calendar_commit": CALENDAR_COMMIT,
        "seed_generation_commit": SEED_GENERATION_COMMIT,
        "activation_authorization_utc": activation_authorization_utc,
        "activation_boundary_first_jpx_trading_date": activation_boundary_first_jpx_trading_date,
        "calendar_source": CALENDAR_SOURCE,
        "calendar_timezone": CALENDAR_TIMEZONE,
        "calendar_definition_version": CALENDAR_DEFINITION_VERSION,
        "calendar_snapshot_sha256": calendar_snapshot_sha256,
        "data_source": DATA_SOURCE,
        "data_source_host": DATA_SOURCE_HOST,
        "data_source_schema": DATA_SOURCE_SCHEMA,
        "acquisition_window_jst": acquisition_window_jst,
        "universe_csv_sha256": UNIVERSE_CSV_SHA256,
        "ticker_list_sha256": TICKER_LIST_SHA256,
        "ticker_count": TICKER_COUNT,
        "arm_a_parameters_sha256": ARM_A_PARAMETERS_SHA256,
        "arm_b_parameters_sha256": ARM_B_PARAMETERS_SHA256,
        "single_changed_parameter": SINGLE_CHANGED_PARAMETER,
        "shared_rules_sha256": SHARED_RULES_SHA256,
        "output_root": output_root,
        "seed_data_source": SEED_DATA_SOURCE,
        "seed_data_schema": SEED_DATA_SCHEMA,
        "seed_acquisition_utc": seed_acquisition_utc,
        "seed_cutoff_trading_date": seed_provenance["seed_cutoff_trading_date"],
        "seed_ticker_count": seed_provenance["seed_ticker_count"],
        "seed_row_count": seed_provenance["seed_row_count"],
        "seed_source_payload_manifest_sha256": seed_provenance["seed_source_payload_manifest_sha256"],
        "seed_ticker_manifest_sha256": seed_provenance["seed_ticker_manifest_sha256"],
        "seed_canonical_csv_sha256": seed_provenance["seed_canonical_csv_sha256"],
        "seed_validation_result": SEED_VALIDATION_RESULT,
        "arm_seed_hash_equal": True,
        "arm_candidate_input_hash_equal": True,
        "arm_market_gate_input_hash_equal": True,
        **PROHIBITION_FIELDS,
    }
    manifest = dict(body)
    manifest["manifest_sha256"] = canonical_sha256(body)
    return manifest


def manifest_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: manifest[key] for key in MANIFEST_FIELDS if key != "manifest_sha256"}


def compute_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_sha256(manifest_body(manifest))


# ---------------------------------------------------------------------------
# Full candidate validation
# ---------------------------------------------------------------------------


def validate_activation_manifest_candidate(
    manifest: Mapping[str, Any],
    *,
    repository_root: str | os.PathLike[str],
    calendar_path: str | os.PathLike[str],
    universe_csv: str | os.PathLike[str],
    seed_csv: str | os.PathLike[str],
    seed_acquisition_manifest: Mapping[str, Any],
    expected_seed_provenance: SeedProvenanceExpectation | None = PRODUCTION_SEED_PROVENANCE,
) -> dict[str, Any]:
    """Validate a production activation manifest candidate.  Creates nothing."""
    if not isinstance(manifest, Mapping):
        raise _blocked("MANIFEST_INVALID")
    unknown = sorted(set(manifest) - set(MANIFEST_FIELDS))
    missing = sorted(set(MANIFEST_FIELDS) - set(manifest))
    if unknown:
        raise _blocked("MANIFEST_UNKNOWN_FIELD:" + ",".join(unknown))
    if missing:
        raise _blocked("MANIFEST_MISSING_FIELD:" + ",".join(missing))

    for field, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("mode", MODE),
        ("study_name", STUDY_NAME),
        ("activation_status", ACTIVATION_STATUS),
        ("design_commit", DESIGN_COMMIT),
        ("preregistration_utc", PREREGISTRATION_UTC),
        ("implementation_commit", IMPLEMENTATION_COMMIT),
        ("collector_commit", COLLECTOR_COMMIT),
        ("calendar_commit", CALENDAR_COMMIT),
        ("seed_generation_commit", SEED_GENERATION_COMMIT),
        ("calendar_source", CALENDAR_SOURCE),
        ("calendar_timezone", CALENDAR_TIMEZONE),
        ("calendar_definition_version", CALENDAR_DEFINITION_VERSION),
        ("data_source", DATA_SOURCE),
        ("data_source_host", DATA_SOURCE_HOST),
        ("data_source_schema", DATA_SOURCE_SCHEMA),
        ("seed_data_source", SEED_DATA_SOURCE),
        ("seed_data_schema", SEED_DATA_SCHEMA),
        ("universe_csv_sha256", UNIVERSE_CSV_SHA256),
        ("ticker_list_sha256", TICKER_LIST_SHA256),
        ("ticker_count", TICKER_COUNT),
        ("arm_a_parameters_sha256", ARM_A_PARAMETERS_SHA256),
        ("arm_b_parameters_sha256", ARM_B_PARAMETERS_SHA256),
        ("single_changed_parameter", SINGLE_CHANGED_PARAMETER),
        ("shared_rules_sha256", SHARED_RULES_SHA256),
        ("seed_validation_result", SEED_VALIDATION_RESULT),
    ):
        if manifest[field] != expected:
            raise _blocked("FROZEN_FIELD_MISMATCH:" + field)

    for field, expected in PROHIBITION_FIELDS.items():
        if manifest[field] is not expected:
            raise _blocked("PROHIBITION_FIELD_INVALID:" + field)
    for field in ARM_PARITY_FIELDS:
        if manifest[field] is not True:
            raise _blocked("ARM_PARITY_FIELD_INVALID:" + field)

    control = V7EngineParameters.control()
    variant = V7EngineParameters.capacity_3()
    try:
        validate_single_parameter_difference(control, variant)
    except ValueError as error:
        raise _blocked("ARM_PARAMETER_DIFFERENCE_INVALID") from error
    if control.sha256() != manifest["arm_a_parameters_sha256"]:
        raise _blocked("ARM_A_PARAMETERS_SHA_MISMATCH")
    if variant.sha256() != manifest["arm_b_parameters_sha256"]:
        raise _blocked("ARM_B_PARAMETERS_SHA_MISMATCH")
    if shared_rules_sha256() != manifest["shared_rules_sha256"]:
        raise _blocked("SHARED_RULES_SHA_MISMATCH")

    require_human_decisions_resolved(manifest)
    window = validate_acquisition_window(manifest["acquisition_window_jst"])
    validate_output_root(manifest["output_root"], repository_root)

    calendar = validate_calendar_binding(calendar_path)
    if manifest["calendar_snapshot_sha256"] != calendar["calendar_snapshot_sha256"]:
        raise _blocked("CALENDAR_SNAPSHOT_SHA_MISMATCH")

    authorization = validate_authorization_and_boundary(
        snapshot=calendar["snapshot"],
        activation_authorization_utc=manifest["activation_authorization_utc"],
        activation_boundary_first_jpx_trading_date=manifest["activation_boundary_first_jpx_trading_date"],
        seed_acquisition_utc=manifest["seed_acquisition_utc"],
    )

    seed = validate_seed_provenance(
        universe_csv=universe_csv,
        seed_csv=seed_csv,
        seed_acquisition_manifest=seed_acquisition_manifest,
        activation_boundary_first_jpx_trading_date=manifest["activation_boundary_first_jpx_trading_date"],
        expected=expected_seed_provenance,
    )
    for field in (
        "seed_source_payload_manifest_sha256",
        "seed_ticker_manifest_sha256",
        "seed_canonical_csv_sha256",
        "seed_ticker_count",
        "seed_row_count",
        "seed_cutoff_trading_date",
    ):
        if manifest[field] != seed[field]:
            raise _blocked("SEED_FIELD_MISMATCH:" + field)

    if not _valid_sha256(manifest["manifest_sha256"]):
        raise _blocked("MANIFEST_SHA_INVALID")
    if compute_manifest_sha256(manifest) != manifest["manifest_sha256"]:
        raise _blocked("MANIFEST_SHA_MISMATCH")

    return {
        "status": "PASS",
        "manifest_sha256": manifest["manifest_sha256"],
        "activation_boundary_first_jpx_trading_date": manifest["activation_boundary_first_jpx_trading_date"],
        "authorization_jst_date": authorization["authorization_jst_date"],
        "acquisition_window_minutes": window,
        "seed_ticker_count": seed["seed_ticker_count"],
        "seed_row_count": seed["seed_row_count"],
    }


# ---------------------------------------------------------------------------
# Write-once
# ---------------------------------------------------------------------------


def write_activation_manifest_once(
    *,
    output_path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    repository_root: str | os.PathLike[str],
    confirmation: str,
    calendar_path: str | os.PathLike[str],
    universe_csv: str | os.PathLike[str],
    seed_csv: str | os.PathLike[str],
    seed_acquisition_manifest: Mapping[str, Any],
    expected_seed_provenance: SeedProvenanceExpectation | None = PRODUCTION_SEED_PROVENANCE,
) -> dict[str, Any]:
    """Atomically write the manifest exactly once.  There is no overwrite path."""
    if confirmation != HUMAN_ACTIVATION_CONFIRMATION:
        raise _blocked("HUMAN_ACTIVATION_CONFIRMATION_REQUIRED")
    validation = validate_activation_manifest_candidate(
        manifest,
        repository_root=repository_root,
        calendar_path=calendar_path,
        universe_csv=universe_csv,
        seed_csv=seed_csv,
        seed_acquisition_manifest=seed_acquisition_manifest,
        expected_seed_provenance=expected_seed_provenance,
    )
    destination = Path(output_path)
    if destination.exists():
        raise _blocked("ACTIVATION_MANIFEST_ALREADY_EXISTS")
    parent = destination.parent
    if not parent.is_dir():
        raise _blocked("ACTIVATION_MANIFEST_PARENT_MISSING")

    payload = canonical_json_bytes(dict(manifest))
    staging_path: Path | None = None
    try:
        handle, staging_name = tempfile.mkstemp(
            prefix="." + destination.name + ".staging-", dir=str(parent)
        )
        staging_path = Path(staging_name)
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.exists():
            raise _blocked("ACTIVATION_MANIFEST_ALREADY_EXISTS")
        os.replace(str(staging_path), str(destination))
        staging_path = None
    finally:
        if staging_path is not None and staging_path.exists():
            staging_path.unlink()

    return {
        "status": "WRITTEN",
        "output_path": str(destination),
        "manifest_sha256": validation["manifest_sha256"],
        "byte_count": len(payload),
    }


def read_activation_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read-only manifest load; performs no validation side effects."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _blocked("ACTIVATION_MANIFEST_READ_FAILED") from error


__all__ = [
    "ACTIVATION_STATUS",
    "ARM_A_PARAMETERS_SHA256",
    "ARM_B_PARAMETERS_SHA256",
    "ARM_PARITY_FIELDS",
    "CALENDAR_COMMIT",
    "CALENDAR_DEFINITION_VERSION",
    "CALENDAR_SNAPSHOT_SHA256",
    "CALENDAR_SOURCE",
    "CALENDAR_TIMEZONE",
    "COLLECTOR_COMMIT",
    "DATA_SOURCE",
    "DATA_SOURCE_HOST",
    "DATA_SOURCE_SCHEMA",
    "DESIGN_COMMIT",
    "HUMAN_ACTIVATION_CONFIRMATION",
    "HUMAN_DECISION_FIELDS",
    "IMPLEMENTATION_COMMIT",
    "MANIFEST_FIELDS",
    "MODE",
    "PLACEHOLDER_VALUES",
    "PREREGISTRATION_UTC",
    "PRODUCTION_SEED_PROVENANCE",
    "PROHIBITION_FIELDS",
    "SCHEMA_VERSION",
    "SEED_GENERATION_COMMIT",
    "SHARED_RULES",
    "SHARED_RULES_SHA256",
    "SINGLE_CHANGED_PARAMETER",
    "STUDY_NAME",
    "TICKER_COUNT",
    "TICKER_LIST_SHA256",
    "UNIVERSE_CSV_SHA256",
    "SeedProvenanceExpectation",
    "V7ActivationManifestBlocked",
    "build_activation_manifest_candidate",
    "canonical_json_bytes",
    "canonical_sha256",
    "compute_manifest_sha256",
    "expected_activation_boundary",
    "hash_source_payload_manifest",
    "manifest_body",
    "read_activation_manifest",
    "read_seed_csv_rows",
    "require_human_decisions_resolved",
    "sha256_bytes",
    "shared_rules_sha256",
    "validate_acquisition_window",
    "validate_activation_manifest_candidate",
    "validate_authorization_and_boundary",
    "validate_calendar_binding",
    "validate_output_root",
    "validate_seed_provenance",
    "write_activation_manifest_once",
]
