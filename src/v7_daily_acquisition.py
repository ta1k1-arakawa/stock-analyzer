"""Append-only V7 daily 300-ticker acquisition bundle.

This module acquires exactly one JPX engine day of Yahoo Chart D0 price
observations for the fixed 300-ticker V4 universe and atomically publishes an
append-only ``acquisitions/YYYY-MM-DD/`` bundle.  It performs no candidate
generation, market-gate evaluation, portfolio processing, profit calculation,
formal evaluation, or activation.  Tests inject an opener, clock, monotonic
clock, and sleeper; importing this module performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.v7_jpx_calendar import CalendarSnapshot, V7JpxCalendarBlocked, is_jpx_trading_day, load_calendar_snapshot
from src.v7_seed_acquisition import V7SeedAcquisitionBlocked, validate_universe_file
from src.v7_yahoo_collector import FRAME_FIELDS, HOST, V7YahooCollectorBlocked, canonical_ticker, fetch_chart_once


SCHEMA_VERSION = "V7_DAILY_ACQUISITION_V1"
MODE = "FORWARD_DAILY_ACQUISITION"
CALENDAR_COMMIT = "03ce048b0eedca632f79ad925a627cb9e967d78d"
CALENDAR_DEFINITION_VERSION = "V7_JPX_OFFICIAL_MARKET_HOLIDAYS_V1"
COLLECTOR_COMMIT = "4ca41c53895e75910ae65809fea6018868929afa"
DATA_SOURCE = "Yahoo Chart"
DATA_SOURCE_HOST = HOST
EXPECTED_TICKER_COUNT = 300
MIN_REQUEST_INTERVAL_SECONDS = 2.0
UTC = timezone.utc

ACQUISITIONS_DIRNAME = "acquisitions"
RAW_DIRNAME = "raw"
PRICE_SNAPSHOT_FILENAME = "price_snapshot.json"
MISSING_SNAPSHOT_FILENAME = "missing_snapshot.json"
SPLIT_SNAPSHOT_FILENAME = "split_snapshot.json"
MANIFEST_FILENAME = "acquisition_manifest.json"
DAY_FILES = (
    PRICE_SNAPSHOT_FILENAME,
    MISSING_SNAPSHOT_FILENAME,
    SPLIT_SNAPSHOT_FILENAME,
    MANIFEST_FILENAME,
)

MANIFEST_FIELDS = (
    "schema_version",
    "mode",
    "engine_day",
    "calendar_commit",
    "calendar_definition_version",
    "collector_commit",
    "data_source",
    "data_source_host",
    "universe_csv_sha256",
    "ticker_list_sha256",
    "ticker_count",
    "request_start",
    "request_end_exclusive",
    "request_count",
    "retry_count",
    "http_429_count",
    "success_transport_count",
    "valid_d0_count",
    "missing_d0_count",
    "split_event_count",
    "payload_manifest",
    "payload_manifest_sha256",
    "price_snapshot_sha256",
    "missing_snapshot_sha256",
    "split_snapshot_sha256",
    "acquisition_started_utc",
    "acquisition_completed_utc",
    "candidate_generation_started",
    "portfolio_processing_started",
    "profit_calculation_started",
    "formal_evaluation_started",
    "activation_created",
)

STATUS_VALID_D0 = "VALID_D0"
STATUS_AUDITED_MISSING = "AUDITED_MISSING"
REASON_D0_DATA_UNAVAILABLE = "D0_DATA_UNAVAILABLE"


class V7DailyAcquisitionBlocked(RuntimeError):
    """Fail-closed daily acquisition transport, schema, or provenance error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except ValueError as error:
        raise V7DailyAcquisitionBlocked("NONFINITE_VALUE") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _parse_date(value: object, field: str) -> date:
    if not isinstance(value, str):
        raise V7DailyAcquisitionBlocked("INVALID_DATE:" + field)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise V7DailyAcquisitionBlocked("INVALID_DATE:" + field) from error
    if parsed.isoformat() != value:
        raise V7DailyAcquisitionBlocked("INVALID_DATE:" + field)
    return parsed


def _utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V7DailyAcquisitionBlocked("UTC_TIMESTAMP_INVALID:" + field)
    return value.astimezone(UTC)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V7DailyAcquisitionBlocked("BUNDLE_FILE_READ_FAILED:" + path.name) from error


# ---------------------------------------------------------------------------
# Pre-network fail-closed validation
# ---------------------------------------------------------------------------


def require_engine_day_trading(calendar: CalendarSnapshot, engine_day: str) -> None:
    try:
        trading = is_jpx_trading_day(calendar, engine_day)
    except V7JpxCalendarBlocked as error:
        raise V7DailyAcquisitionBlocked("ENGINE_DAY_OUTSIDE_CALENDAR_COVERAGE") from error
    if not trading:
        raise V7DailyAcquisitionBlocked("ENGINE_DAY_NOT_JPX_TRADING_DAY")


# ---------------------------------------------------------------------------
# Minimal fail-closed raw-payload classifier for the narrow D0_DATA_UNAVAILABLE case
# ---------------------------------------------------------------------------


def classify_missing_timestamp_payload(payload_bytes: bytes, expected_ticker: str) -> bool:
    """Return True only for a transport-successful response with a null/empty timestamp series."""
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    chart = payload.get("chart")
    if not isinstance(chart, Mapping) or "error" not in chart or chart.get("error") is not None:
        return False
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], Mapping):
        return False
    result = results[0]
    meta = result.get("meta")
    if not isinstance(meta, Mapping):
        return False
    try:
        symbol = canonical_ticker(meta.get("symbol"))
    except Exception:
        return False
    if symbol != expected_ticker:
        return False
    timestamps = result.get("timestamp")
    if timestamps is None:
        return True
    if isinstance(timestamps, list) and len(timestamps) == 0:
        return True
    return False


# ---------------------------------------------------------------------------
# Split classification (defense-in-depth, independent of transport window)
# ---------------------------------------------------------------------------


def classify_engine_day_splits(
    events: Sequence[Mapping[str, Any]], ticker: str, engine_day: str
) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for event in events:
        effective_date = str(event["effective_date"])
        _parse_date(effective_date, "effective_date")
        if effective_date > engine_day:
            raise V7DailyAcquisitionBlocked("FUTURE_SPLIT_ACCESS")
        if effective_date < engine_day:
            raise V7DailyAcquisitionBlocked("SPLIT_EFFECTIVE_DATE_BEFORE_ENGINE_DAY")
        if effective_date in seen:
            raise V7DailyAcquisitionBlocked("DUPLICATE_SPLIT_EVENT")
        seen.add(effective_date)
        accepted.append({
            "ticker": ticker,
            "effective_date": effective_date,
            "numerator": event["numerator"],
            "denominator": event["denominator"],
            "split_ratio": event["split_ratio"],
        })
    return accepted


def _classify_transport_error(error: BaseException) -> tuple[str, bool]:
    reason = getattr(error, "reason", None)
    if isinstance(reason, str) and reason:
        text = reason
    else:
        code = getattr(error, "code", None)
        text = ("HTTP_STATUS_" + str(code)) if code is not None else (str(error) or error.__class__.__name__)
    return text, text == "HTTP_STATUS_429"


class _RecordingResponse:
    def __init__(self, response: Any, capture: bytearray) -> None:
        self._response = response
        self._capture = capture

    @property
    def status(self) -> Any:
        return getattr(self._response, "status", None)

    @property
    def url(self) -> Any:
        return getattr(self._response, "url", None)

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        value = self._response.read(*args, **kwargs)
        if isinstance(value, bytes):
            self._capture.extend(value)
        return value

    def close(self) -> None:
        close = getattr(self._response, "close", None)
        if callable(close):
            close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


def _wait_for_next_request_start(
    index: int,
    previous_start: float | None,
    monotonic_clock: Callable[[], float],
    sleep_fn: Callable[[float], None],
) -> float:
    if index == 0 or previous_start is None:
        return monotonic_clock()
    elapsed = monotonic_clock() - previous_start
    remaining = MIN_REQUEST_INTERVAL_SECONDS - elapsed
    if remaining > 0:
        sleep_fn(remaining)
    return monotonic_clock()


# ---------------------------------------------------------------------------
# Bundle acquisition
# ---------------------------------------------------------------------------


def acquire_daily_bundle(
    *,
    output_root: str | os.PathLike[str],
    universe_csv: str | os.PathLike[str],
    calendar_snapshot: Mapping[str, Any] | str | os.PathLike[str],
    engine_day: str,
    opener: Callable[[Any], Any],
    clock: Callable[[], datetime],
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    _parse_date(engine_day, "engine_day")
    try:
        calendar = load_calendar_snapshot(calendar_snapshot)
    except V7JpxCalendarBlocked as error:
        raise V7DailyAcquisitionBlocked("CALENDAR_SNAPSHOT_INVALID") from error
    require_engine_day_trading(calendar, engine_day)

    try:
        universe = validate_universe_file(universe_csv)
    except V7SeedAcquisitionBlocked as error:
        raise V7DailyAcquisitionBlocked("UNIVERSE_VALIDATION_FAILED:" + error.reason) from error
    if universe["ticker_count"] != EXPECTED_TICKER_COUNT:
        raise V7DailyAcquisitionBlocked("UNIVERSE_TICKER_COUNT_MISMATCH")

    acquisitions_root = Path(output_root) / ACQUISITIONS_DIRNAME
    acquisitions_root.mkdir(parents=True, exist_ok=True)
    final_dir = acquisitions_root / engine_day
    if final_dir.exists():
        raise V7DailyAcquisitionBlocked("DUPLICATE_ACQUISITION_DAY")
    if any(".staging-" in entry.name for entry in acquisitions_root.iterdir()):
        raise V7DailyAcquisitionBlocked("PARTIAL_ACQUISITION_COMMIT")

    request_start = engine_day
    request_end_exclusive = (date.fromisoformat(engine_day) + timedelta(days=1)).isoformat()

    started_dt = _utc_timestamp(clock(), "acquisition_started_utc")

    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f"{engine_day}.staging-", dir=str(acquisitions_root)))
        (staging / RAW_DIRNAME).mkdir()

        price_rows: list[dict[str, Any]] = []
        missing_rows: list[dict[str, Any]] = []
        split_rows: list[dict[str, Any]] = []
        payload_manifest: list[dict[str, Any]] = []
        request_count = 0
        http_429_count = 0
        success_transport_count = 0
        previous_start: float | None = None

        for index, ticker in enumerate(universe["tickers"]):
            previous_start = _wait_for_next_request_start(index, previous_start, monotonic_clock, sleep_fn)
            capture = bytearray()

            def recording_opener(request_obj: Any, *, _capture: bytearray = capture) -> Any:
                return _RecordingResponse(opener(request_obj), _capture)

            request_count += 1
            parsed: dict[str, Any] | None = None
            try:
                parsed = fetch_chart_once(
                    ticker, request_start, request_end_exclusive, opener=recording_opener
                )
            except V7YahooCollectorBlocked as error:
                raw_bytes = bytes(capture)
                if error.reason == "TIMESTAMP_MISSING" and classify_missing_timestamp_payload(raw_bytes, ticker):
                    parsed = None
                else:
                    reason, is_429 = _classify_transport_error(error)
                    if is_429:
                        http_429_count += 1
                    raise V7DailyAcquisitionBlocked("TICKER_" + ticker + ":" + reason) from error
            except BaseException as error:
                reason, is_429 = _classify_transport_error(error)
                if is_429:
                    http_429_count += 1
                raise V7DailyAcquisitionBlocked("TICKER_" + ticker + ":" + reason) from error

            raw_bytes = bytes(capture)
            payload_sha256 = sha256_bytes(raw_bytes)
            byte_count = len(raw_bytes)

            if parsed is not None:
                if payload_sha256 != parsed.get("payload_sha256") or byte_count != parsed.get("byte_count"):
                    raise V7DailyAcquisitionBlocked("RAW_PAYLOAD_MISMATCH:" + ticker)
                valid_rows = parsed["valid_price_rows"]
                invalid_rows = parsed["invalid_price_rows"]
                split_events_raw = parsed["canonical_split_events"]
                if len(valid_rows) == 1 and len(invalid_rows) == 0:
                    status = STATUS_VALID_D0
                    missing_reason: str | None = None
                elif len(valid_rows) == 0 and len(invalid_rows) == 1:
                    status = STATUS_AUDITED_MISSING
                    missing_reason = str(invalid_rows[0]["reason"])
                else:
                    raise V7DailyAcquisitionBlocked("UNEXPECTED_PARSER_RESULT_SHAPE:" + ticker)
            else:
                status = STATUS_AUDITED_MISSING
                missing_reason = REASON_D0_DATA_UNAVAILABLE
                split_events_raw = []

            _atomic_write(staging / RAW_DIRNAME / (ticker + ".json"), raw_bytes)

            split_events = classify_engine_day_splits(split_events_raw, ticker, engine_day)
            split_rows.extend(split_events)

            if status == STATUS_VALID_D0:
                row = dict(valid_rows[0])
                if row["trading_date"] != engine_day:
                    raise V7DailyAcquisitionBlocked("PRICE_ROW_ENGINE_DAY_MISMATCH:" + ticker)
                canonical_row = {field: row[field] for field in FRAME_FIELDS}
                price_rows.append({**canonical_row, "payload_sha256": payload_sha256})
                canonical_d0_row_sha256: str | None = canonical_sha256(canonical_row)
                valid_row_count = 1
                invalid_row_count = 0
            else:
                missing_rows.append({
                    "ticker": ticker,
                    "engine_day": engine_day,
                    "reason": missing_reason,
                    "payload_sha256": payload_sha256,
                    "byte_count": byte_count,
                })
                canonical_d0_row_sha256 = None
                valid_row_count = 0
                invalid_row_count = 1 if parsed is not None else 0

            payload_manifest.append({
                "ticker": ticker,
                "status": status,
                "payload_sha256": payload_sha256,
                "byte_count": byte_count,
                "valid_price_row_count": valid_row_count,
                "invalid_price_row_count": invalid_row_count,
                "split_event_count": len(split_events),
                "canonical_d0_row_sha256": canonical_d0_row_sha256,
                "canonical_engine_day_split_sha256": canonical_sha256(split_events),
                "missing_reason": missing_reason,
            })
            success_transport_count += 1

        if len(price_rows) + len(missing_rows) != EXPECTED_TICKER_COUNT:
            raise V7DailyAcquisitionBlocked("ACCOUNTING_MISMATCH")
        price_tickers = {row["ticker"] for row in price_rows}
        missing_tickers = {row["ticker"] for row in missing_rows}
        if price_tickers & missing_tickers:
            raise V7DailyAcquisitionBlocked("TICKER_STATUS_OVERLAP")
        if len(price_tickers) + len(missing_tickers) != EXPECTED_TICKER_COUNT:
            raise V7DailyAcquisitionBlocked("ACCOUNTING_MISMATCH")

        split_keys: set[tuple[str, str]] = set()
        for row in split_rows:
            key = (row["ticker"], row["effective_date"])
            if key in split_keys:
                raise V7DailyAcquisitionBlocked("DUPLICATE_SPLIT_EVENT")
            split_keys.add(key)

        price_snapshot = sorted(price_rows, key=lambda row: row["ticker"])
        missing_snapshot = sorted(missing_rows, key=lambda row: row["ticker"])
        split_snapshot = sorted(split_rows, key=lambda row: (row["effective_date"], row["ticker"]))

        price_bytes = canonical_json_bytes(price_snapshot)
        missing_bytes = canonical_json_bytes(missing_snapshot)
        split_bytes = canonical_json_bytes(split_snapshot)
        payload_manifest_bytes = canonical_json_bytes(payload_manifest)

        _atomic_write(staging / PRICE_SNAPSHOT_FILENAME, price_bytes)
        _atomic_write(staging / MISSING_SNAPSHOT_FILENAME, missing_bytes)
        _atomic_write(staging / SPLIT_SNAPSHOT_FILENAME, split_bytes)

        completed_dt = _utc_timestamp(clock(), "acquisition_completed_utc")
        if completed_dt < started_dt:
            raise V7DailyAcquisitionBlocked("ACQUISITION_CLOCK_NONMONOTONIC")

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "mode": MODE,
            "engine_day": engine_day,
            "calendar_commit": CALENDAR_COMMIT,
            "calendar_definition_version": CALENDAR_DEFINITION_VERSION,
            "collector_commit": COLLECTOR_COMMIT,
            "data_source": DATA_SOURCE,
            "data_source_host": DATA_SOURCE_HOST,
            "universe_csv_sha256": universe["universe_csv_sha256"],
            "ticker_list_sha256": universe["ticker_list_sha256"],
            "ticker_count": universe["ticker_count"],
            "request_start": request_start,
            "request_end_exclusive": request_end_exclusive,
            "request_count": request_count,
            "retry_count": 0,
            "http_429_count": http_429_count,
            "success_transport_count": success_transport_count,
            "valid_d0_count": len(price_snapshot),
            "missing_d0_count": len(missing_snapshot),
            "split_event_count": len(split_snapshot),
            "payload_manifest": payload_manifest,
            "payload_manifest_sha256": sha256_bytes(payload_manifest_bytes),
            "price_snapshot_sha256": sha256_bytes(price_bytes),
            "missing_snapshot_sha256": sha256_bytes(missing_bytes),
            "split_snapshot_sha256": sha256_bytes(split_bytes),
            "acquisition_started_utc": _timestamp_text(started_dt),
            "acquisition_completed_utc": _timestamp_text(completed_dt),
            "candidate_generation_started": 0,
            "portfolio_processing_started": 0,
            "profit_calculation_started": 0,
            "formal_evaluation_started": 0,
            "activation_created": False,
        }
        _atomic_write(staging / MANIFEST_FILENAME, canonical_json_bytes(manifest))
        os.replace(str(staging), str(final_dir))
        staging = None
        return manifest
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Read-only verifier
# ---------------------------------------------------------------------------


def verify_daily_acquisition_bundle(
    root: str | os.PathLike[str],
    engine_day: str,
    expected_calendar_commit: str,
    expected_collector_commit: str,
) -> dict[str, Any]:
    _parse_date(engine_day, "engine_day")
    acquisitions_root = Path(root) / ACQUISITIONS_DIRNAME
    if not acquisitions_root.is_dir():
        raise V7DailyAcquisitionBlocked("ACQUISITION_DAY_NOT_FOUND")
    if any(".staging-" in entry.name for entry in acquisitions_root.iterdir()):
        raise V7DailyAcquisitionBlocked("PARTIAL_ACQUISITION_COMMIT")
    day_dir = acquisitions_root / engine_day
    if not day_dir.is_dir():
        raise V7DailyAcquisitionBlocked("ACQUISITION_DAY_NOT_FOUND")

    actual_files = {entry.name for entry in day_dir.iterdir()}
    if actual_files != set(DAY_FILES) | {RAW_DIRNAME}:
        raise V7DailyAcquisitionBlocked("BUNDLE_SCHEMA_INVALID")

    manifest = _read_json(day_dir / MANIFEST_FILENAME)
    if set(manifest) != set(MANIFEST_FIELDS):
        raise V7DailyAcquisitionBlocked("MANIFEST_SCHEMA_INVALID")
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["mode"] != MODE:
        raise V7DailyAcquisitionBlocked("MANIFEST_IDENTITY_INVALID")
    if manifest["engine_day"] != engine_day:
        raise V7DailyAcquisitionBlocked("MANIFEST_ENGINE_DAY_MISMATCH")
    if manifest["calendar_commit"] != expected_calendar_commit:
        raise V7DailyAcquisitionBlocked("CALENDAR_COMMIT_MISMATCH")
    if manifest["collector_commit"] != expected_collector_commit:
        raise V7DailyAcquisitionBlocked("COLLECTOR_COMMIT_MISMATCH")
    if manifest["ticker_count"] != EXPECTED_TICKER_COUNT:
        raise V7DailyAcquisitionBlocked("TICKER_COUNT_MISMATCH")
    if manifest["request_count"] != EXPECTED_TICKER_COUNT:
        raise V7DailyAcquisitionBlocked("REQUEST_COUNT_MISMATCH")
    if manifest["retry_count"] != 0:
        raise V7DailyAcquisitionBlocked("RETRY_COUNT_INVALID")

    payload_manifest = manifest["payload_manifest"]
    if not isinstance(payload_manifest, list) or len(payload_manifest) != EXPECTED_TICKER_COUNT:
        raise V7DailyAcquisitionBlocked("PAYLOAD_MANIFEST_COUNT_MISMATCH")

    manifest_tickers = [record["ticker"] for record in payload_manifest]
    if len(set(manifest_tickers)) != EXPECTED_TICKER_COUNT:
        raise V7DailyAcquisitionBlocked("PAYLOAD_MANIFEST_TICKER_DUPLICATE")
    raw_files = {entry.name for entry in (day_dir / RAW_DIRNAME).iterdir()}
    expected_raw_files = {ticker + ".json" for ticker in manifest_tickers}
    if raw_files != expected_raw_files:
        raise V7DailyAcquisitionBlocked("RAW_TICKER_ORDER_PARITY_MISMATCH")

    valid_tickers: set[str] = set()
    missing_tickers: set[str] = set()
    for record in payload_manifest:
        ticker = record["ticker"]
        raw_bytes = (day_dir / RAW_DIRNAME / (ticker + ".json")).read_bytes()
        if sha256_bytes(raw_bytes) != record["payload_sha256"]:
            raise V7DailyAcquisitionBlocked("RAW_SHA_MISMATCH:" + ticker)
        if len(raw_bytes) != record["byte_count"]:
            raise V7DailyAcquisitionBlocked("RAW_BYTE_COUNT_MISMATCH:" + ticker)
        if record["status"] == STATUS_VALID_D0:
            if record["missing_reason"] is not None or record["canonical_d0_row_sha256"] is None:
                raise V7DailyAcquisitionBlocked("PAYLOAD_MANIFEST_STATUS_SCHEMA_INVALID:" + ticker)
            valid_tickers.add(ticker)
        elif record["status"] == STATUS_AUDITED_MISSING:
            if record["missing_reason"] is None or record["canonical_d0_row_sha256"] is not None:
                raise V7DailyAcquisitionBlocked("PAYLOAD_MANIFEST_STATUS_SCHEMA_INVALID:" + ticker)
            missing_tickers.add(ticker)
        else:
            raise V7DailyAcquisitionBlocked("PAYLOAD_MANIFEST_STATUS_INVALID:" + ticker)

    if valid_tickers & missing_tickers:
        raise V7DailyAcquisitionBlocked("TICKER_STATUS_OVERLAP")
    if len(valid_tickers) + len(missing_tickers) != EXPECTED_TICKER_COUNT:
        raise V7DailyAcquisitionBlocked("ACCOUNTING_MISMATCH")
    if manifest["valid_d0_count"] != len(valid_tickers) or manifest["missing_d0_count"] != len(missing_tickers):
        raise V7DailyAcquisitionBlocked("ACCOUNTING_MISMATCH")

    price_snapshot = _read_json(day_dir / PRICE_SNAPSHOT_FILENAME)
    if sha256_bytes(canonical_json_bytes(price_snapshot)) != manifest["price_snapshot_sha256"]:
        raise V7DailyAcquisitionBlocked("PRICE_SNAPSHOT_HASH_MISMATCH")
    if {row["ticker"] for row in price_snapshot} != valid_tickers:
        raise V7DailyAcquisitionBlocked("PRICE_SNAPSHOT_TICKER_MISMATCH")
    for row in price_snapshot:
        if row["trading_date"] != engine_day:
            raise V7DailyAcquisitionBlocked("PRICE_SNAPSHOT_DATE_MISMATCH:" + row["ticker"])

    missing_snapshot = _read_json(day_dir / MISSING_SNAPSHOT_FILENAME)
    if sha256_bytes(canonical_json_bytes(missing_snapshot)) != manifest["missing_snapshot_sha256"]:
        raise V7DailyAcquisitionBlocked("MISSING_SNAPSHOT_HASH_MISMATCH")
    if {row["ticker"] for row in missing_snapshot} != missing_tickers:
        raise V7DailyAcquisitionBlocked("MISSING_SNAPSHOT_TICKER_MISMATCH")
    for row in missing_snapshot:
        if row["engine_day"] != engine_day:
            raise V7DailyAcquisitionBlocked("MISSING_SNAPSHOT_DATE_MISMATCH:" + row["ticker"])

    split_snapshot = _read_json(day_dir / SPLIT_SNAPSHOT_FILENAME)
    if sha256_bytes(canonical_json_bytes(split_snapshot)) != manifest["split_snapshot_sha256"]:
        raise V7DailyAcquisitionBlocked("SPLIT_SNAPSHOT_HASH_MISMATCH")
    for row in split_snapshot:
        if row["effective_date"] != engine_day:
            raise V7DailyAcquisitionBlocked("SPLIT_SNAPSHOT_DATE_MISMATCH:" + row["ticker"])
    if len(split_snapshot) != manifest["split_event_count"]:
        raise V7DailyAcquisitionBlocked("SPLIT_EVENT_COUNT_MISMATCH")

    payload_manifest_bytes = canonical_json_bytes(payload_manifest)
    if sha256_bytes(payload_manifest_bytes) != manifest["payload_manifest_sha256"]:
        raise V7DailyAcquisitionBlocked("PAYLOAD_MANIFEST_HASH_MISMATCH")

    return {
        "status": "PASS",
        "engine_day": engine_day,
        "valid_d0_count": len(valid_tickers),
        "missing_d0_count": len(missing_tickers),
        "split_event_count": len(split_snapshot),
    }


__all__ = [
    "ACQUISITIONS_DIRNAME",
    "CALENDAR_COMMIT",
    "CALENDAR_DEFINITION_VERSION",
    "COLLECTOR_COMMIT",
    "DATA_SOURCE",
    "DATA_SOURCE_HOST",
    "EXPECTED_TICKER_COUNT",
    "MANIFEST_FIELDS",
    "MIN_REQUEST_INTERVAL_SECONDS",
    "MODE",
    "REASON_D0_DATA_UNAVAILABLE",
    "SCHEMA_VERSION",
    "STATUS_AUDITED_MISSING",
    "STATUS_VALID_D0",
    "V7DailyAcquisitionBlocked",
    "acquire_daily_bundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "classify_engine_day_splits",
    "classify_missing_timestamp_payload",
    "require_engine_day_trading",
    "sha256_bytes",
    "verify_daily_acquisition_bundle",
]
