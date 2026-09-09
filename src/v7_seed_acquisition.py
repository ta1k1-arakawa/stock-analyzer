"""Pre-activation V7 feature-seed acquisition boundary.

The module is deliberately a thin, fail-closed orchestration layer around
``v7_yahoo_collector.fetch_chart_once``.  It does not create a study calendar,
generate candidates, simulate a portfolio, or activate a study.  Tests inject
an opener, clock, and sleeper; importing this module performs no I/O.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from .v7_yahoo_collector import (
        FRAME_FIELDS,
        HOST,
        V7YahooCollectorBlocked,
        canonical_ticker,
        fetch_chart_once,
    )
except ImportError:
    from v7_yahoo_collector import (
        FRAME_FIELDS,
        HOST,
        V7YahooCollectorBlocked,
        canonical_ticker,
        fetch_chart_once,
    )


DESIGN_COMMIT = "e3e1367efd913b601a70328a815d88c20af6d147"
LATEST_PREREGISTRATION_UTC = "2026-08-07T02:48:27Z"
COLLECTOR_COMMIT = "4ca41c53895e75910ae65809fea6018868929afa"
EXPECTED_UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
EXPECTED_TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
EXPECTED_TICKER_COUNT = 300
CONFIRMATION = "V7_FORWARD_SEED_ACQUISITION"
DEFAULT_REQUEST_START = "2025-07-01"
DEFAULT_REQUEST_END_EXCLUSIVE = "2026-08-08"
DEFAULT_SEED_CUTOFF = "2026-08-07"
MODE = "PRE_ACTIVATION_SEED_ACQUISITION"
ACTIVATION_BOUNDARY_STATUS = "NOT_SET"
ACTIVATION_BOUNDARY_VALIDATION = "DEFERRED_TO_ACTIVATION_GATE"
ACTIVATION_STATUS = "NOT_ACTIVATED"
STUDY_CALENDAR_GENERATED = False
MIN_REQUEST_INTERVAL_SECONDS = 2.0
SEED_COLUMNS = FRAME_FIELDS
UTC = timezone.utc
PREREGISTRATION = datetime.fromisoformat(LATEST_PREREGISTRATION_UTC.replace("Z", "+00:00"))


class V7SeedAcquisitionBlocked(RuntimeError):
    """Raised when acquisition would violate the preregistered boundary."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_date(value: object, field: str) -> date:
    if not isinstance(value, str):
        raise V7SeedAcquisitionBlocked("INVALID_DATE:" + field)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise V7SeedAcquisitionBlocked("INVALID_DATE:" + field) from error
    if parsed.isoformat() != value:
        raise V7SeedAcquisitionBlocked("INVALID_DATE:" + field)
    return parsed


def _utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V7SeedAcquisitionBlocked("UTC_TIMESTAMP_INVALID:" + field)
    return value.astimezone(UTC)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [{field: row[field] for field in SEED_COLUMNS} for row in rows],
        key=lambda row: (str(row["ticker"]), str(row["trading_date"])),
    )


def canonical_seed_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SEED_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in _canonical_rows(rows):
        writer.writerow({field: row[field] for field in SEED_COLUMNS})
    return stream.getvalue().encode("utf-8")


def canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(_canonical_rows(rows)))


def _canonical_splits(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(event) for event in events],
        key=lambda event: (
            str(event["effective_date"]),
            str(event["ticker"]),
            float(event["numerator"]),
            float(event["denominator"]),
        ),
    )


def validate_canonical_split_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reject duplicate or conflicting ticker/effective-date provenance."""
    seen: dict[tuple[str, str], tuple[float, float]] = {}
    for event in events:
        try:
            key = (str(event["ticker"]), str(event["effective_date"]))
            ratio = (float(event["numerator"]), float(event["denominator"]))
        except (KeyError, TypeError, ValueError) as error:
            raise V7SeedAcquisitionBlocked("SPLIT_SCHEMA_INVALID") from error
        if not all(math.isfinite(value) and value > 0 for value in ratio):
            raise V7SeedAcquisitionBlocked("SPLIT_SCHEMA_INVALID")
        if key in seen:
            if seen[key] != ratio:
                raise V7SeedAcquisitionBlocked("CONFLICTING_SPLIT_EVENT")
            raise V7SeedAcquisitionBlocked("DUPLICATE_SPLIT_EVENT")
        seen[key] = ratio
    return _canonical_splits(events)


def _ticker_list_sha(tickers: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(tickers) + "\n").encode("utf-8"))


def validate_universe_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate the immutable V4 universe without changing its row order."""
    universe_path = Path(path)
    try:
        raw = universe_path.read_bytes()
    except OSError as error:
        raise V7SeedAcquisitionBlocked("UNIVERSE_READ_FAILED") from error
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if sha256_bytes(normalized) != EXPECTED_UNIVERSE_CSV_SHA256:
        raise V7SeedAcquisitionBlocked("UNIVERSE_CSV_SHA_MISMATCH")
    try:
        text = normalized.decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise V7SeedAcquisitionBlocked("UNIVERSE_CSV_INVALID") from error
    if not rows or set(rows[0]) < {"ticker"}:
        raise V7SeedAcquisitionBlocked("UNIVERSE_SCHEMA_INVALID")
    tickers: list[str] = []
    industries: dict[str, str | None] = {}
    for row in rows:
        try:
            ticker = canonical_ticker(row.get("ticker"))
        except Exception as error:
            raise V7SeedAcquisitionBlocked("UNIVERSE_TICKER_INVALID") from error
        if ticker in industries:
            raise V7SeedAcquisitionBlocked("UNIVERSE_DUPLICATE_TICKER")
        tickers.append(ticker)
        industries[ticker] = row.get("industry")
    if len(tickers) != EXPECTED_TICKER_COUNT:
        raise V7SeedAcquisitionBlocked("UNIVERSE_TICKER_COUNT_MISMATCH")
    if _ticker_list_sha(tickers) != EXPECTED_TICKER_LIST_SHA256:
        raise V7SeedAcquisitionBlocked("TICKER_LIST_SHA_MISMATCH")
    return {
        "path": str(universe_path),
        "universe_csv_sha256": sha256_bytes(normalized),
        "ticker_list_sha256": _ticker_list_sha(tickers),
        "tickers": tickers,
        "industries": industries,
        "ticker_count": len(tickers),
    }


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


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> bytes:
    value = canonical_seed_csv_bytes(rows)
    _write_bytes(path, value)
    return value


def _validate_row(row: Mapping[str, Any]) -> None:
    if tuple(row) != tuple(SEED_COLUMNS) and set(row) != set(SEED_COLUMNS):
        raise V7SeedAcquisitionBlocked("CANONICAL_ROW_SCHEMA_INVALID")
    for field in ("raw_open", "raw_high", "raw_low", "raw_close", "adj_close"):
        value = row[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise V7SeedAcquisitionBlocked("CANONICAL_ROW_NONFINITE")
        if float(value) <= 0:
            raise V7SeedAcquisitionBlocked("CANONICAL_ROW_NONPOSITIVE_PRICE")
    volume = row["raw_volume"]
    if isinstance(volume, bool) or not isinstance(volume, (int, float)) or not math.isfinite(float(volume)):
        raise V7SeedAcquisitionBlocked("CANONICAL_ROW_NONFINITE_VOLUME")
    if float(volume) < 0:
        raise V7SeedAcquisitionBlocked("CANONICAL_ROW_NEGATIVE_VOLUME")


def _classify_error(error: BaseException) -> tuple[str, bool]:
    code = getattr(error, "code", None)
    if code == 429:
        return "HTTP_STATUS_429", True
    reason = getattr(error, "reason", str(error))
    text = str(reason)
    return text, text == "HTTP_STATUS_429"


def acquire_seed_bundle(
    *,
    output_dir: str | os.PathLike[str],
    universe_csv: str | os.PathLike[str],
    request_start: str,
    request_end_exclusive: str,
    seed_cutoff: str,
    confirmation: str,
    opener: Callable[[Any], Any],
    clock: Callable[[], datetime],
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Acquire and atomically publish a complete local synthetic-testable bundle."""
    start = _parse_date(request_start, "request_start")
    end = _parse_date(request_end_exclusive, "request_end_exclusive")
    cutoff = _parse_date(seed_cutoff, "seed_cutoff")
    if not start < end or not start <= cutoff < end:
        raise V7SeedAcquisitionBlocked("REQUEST_DATE_BOUNDS_INVALID")
    if confirmation != CONFIRMATION:
        raise V7SeedAcquisitionBlocked("CONFIRMATION_MISMATCH")
    destination = Path(output_dir)
    if destination.exists():
        raise V7SeedAcquisitionBlocked("OUTPUT_EXISTS")
    universe = validate_universe_file(universe_csv)
    parent = destination.parent
    if not parent.exists():
        raise V7SeedAcquisitionBlocked("OUTPUT_PARENT_MISSING")
    started_dt = _utc_timestamp(clock(), "acquisition_started_utc")
    if not PREREGISTRATION < started_dt:
        raise V7SeedAcquisitionBlocked("ACQUISITION_NOT_AFTER_PREREGISTRATION")

    staging: Path | None = None
    request_count = 0
    http_429_count = 0
    records: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    all_splits: list[dict[str, Any]] = []
    invalid_reason_counts: Counter[str] = Counter()
    try:
        staging = Path(tempfile.mkdtemp(prefix=destination.name + ".staging-", dir=str(parent)))
        (staging / "raw").mkdir()
        for index, ticker in enumerate(universe["tickers"]):
            if index:
                sleep_fn(MIN_REQUEST_INTERVAL_SECONDS)
            capture = bytearray()

            def recording_opener(request_obj: Any, *, _capture: bytearray = capture) -> Any:
                return _RecordingResponse(opener(request_obj), _capture)

            request_count += 1
            try:
                parsed = fetch_chart_once(
                    ticker,
                    request_start,
                    request_end_exclusive,
                    opener=recording_opener,
                )
            except BaseException as error:
                reason, is_429 = _classify_error(error)
                if is_429:
                    http_429_count += 1
                raise V7SeedAcquisitionBlocked("TICKER_" + ticker + ":" + reason) from error
            payload_bytes = bytes(capture)
            if sha256_bytes(payload_bytes) != parsed.get("payload_sha256"):
                raise V7SeedAcquisitionBlocked("RAW_PAYLOAD_SHA_MISMATCH:" + ticker)
            if len(payload_bytes) != parsed.get("byte_count"):
                raise V7SeedAcquisitionBlocked("RAW_PAYLOAD_BYTE_COUNT_MISMATCH:" + ticker)
            raw_path = staging / "raw" / (ticker + ".json")
            _write_bytes(raw_path, payload_bytes)
            valid_rows = [dict(row) for row in parsed["valid_price_rows"]]
            for row in valid_rows:
                _validate_row(row)
            invalid_rows = [dict(row) for row in parsed["invalid_price_rows"]]
            for row in invalid_rows:
                invalid_reason_counts[str(row["reason"])] += 1
            split_rows = [dict(row) for row in parsed["canonical_split_events"]]
            all_rows.extend(valid_rows)
            all_splits.extend(split_rows)
            records.append({
                "ticker": ticker,
                "payload_sha256": parsed["payload_sha256"],
                "byte_count": parsed["byte_count"],
                "canonical_price_rows_sha256": parsed["canonical_price_rows_sha256"],
                "canonical_split_events_sha256": parsed["canonical_split_events_sha256"],
                "valid_price_row_count": len(valid_rows),
                "invalid_price_row_count": len(invalid_rows),
                "split_event_count": len(split_rows),
            })

        keyed_rows: set[tuple[str, str]] = set()
        for row in all_rows:
            key = (str(row["ticker"]), str(row["trading_date"]))
            if key in keyed_rows:
                raise V7SeedAcquisitionBlocked("DUPLICATE_TICKER_DATE")
            keyed_rows.add(key)
        canonical_rows = _canonical_rows(all_rows)

        canonical_split_events = validate_canonical_split_events(all_splits)

        price_csv = _write_rows_csv(staging / "canonical_price_rows.csv", canonical_rows)
        split_bytes = canonical_json_bytes(canonical_split_events)
        _write_bytes(staging / "canonical_split_events.json", split_bytes)

        selected_rows: list[dict[str, Any]] = []
        seed_manifest: list[dict[str, Any]] = []
        for ticker in universe["tickers"]:
            ticker_rows = [row for row in canonical_rows if row["ticker"] == ticker and _parse_date(row["trading_date"], "trading_date") <= cutoff]
            selected = ticker_rows[-252:]
            selected_rows.extend(selected)
            seed_manifest.append({
                "ticker": ticker,
                "first_seed_trading_date": selected[0]["trading_date"] if selected else None,
                "last_seed_trading_date": selected[-1]["trading_date"] if selected else None,
                "valid_observation_count": len(selected),
                "eligibility_at_seed": len(selected) == 252,
            })
        canonical_seed_rows = _canonical_rows(selected_rows)
        seed_csv = _write_rows_csv(staging / "seed.csv", canonical_seed_rows)
        payload_manifest_bytes = canonical_json_bytes(records)
        payload_manifest_sha = sha256_bytes(payload_manifest_bytes)
        completed_dt = _utc_timestamp(clock(), "acquisition_completed_utc")
        if completed_dt < started_dt:
            raise V7SeedAcquisitionBlocked("ACQUISITION_CLOCK_NONMONOTONIC")
        seed_manifest_value: dict[str, Any] = {
            "schema_version": "V7_SEED_ACQUISITION_V1",
            "mode": MODE,
            "design_commit": DESIGN_COMMIT,
            "latest_preregistration_utc": LATEST_PREREGISTRATION_UTC,
            "collector_commit": COLLECTOR_COMMIT,
            "data_source": "Yahoo Chart",
            "data_source_host": HOST,
            "data_source_schema": "Yahoo Chart v8/finance/chart interval=1d events=div,splits includeAdjustedClose=true",
            "request_start": request_start,
            "request_end_exclusive": request_end_exclusive,
            "seed_cutoff_trading_date": seed_cutoff,
            "universe_csv_sha256": universe["universe_csv_sha256"],
            "ticker_list_sha256": universe["ticker_list_sha256"],
            "ticker_count": universe["ticker_count"],
            "request_count": request_count,
            "retry_count": 0,
            "http_429_count": http_429_count,
            "success_count": len(records),
            "failed_count": 0,
            "valid_price_row_count": len(canonical_rows),
            "invalid_price_row_count": sum(invalid_reason_counts.values()),
            "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
            "split_event_count": len(canonical_split_events),
            "eligible_seed_ticker_count": sum(1 for item in seed_manifest if item["eligibility_at_seed"]),
            "ineligible_seed_ticker_count": sum(1 for item in seed_manifest if not item["eligibility_at_seed"]),
            "seed_row_count": len(canonical_seed_rows),
            "seed_ticker_manifest": seed_manifest,
            "payload_manifest": records,
            "seed_payload_manifest_sha256": payload_manifest_sha,
            "canonical_price_rows_csv_sha256": sha256_bytes(price_csv),
            "canonical_split_events_sha256": sha256_bytes(split_bytes),
            "seed_canonical_csv_sha256": sha256_bytes(seed_csv),
            "seed_canonical_rows_sha256": canonical_rows_sha256(canonical_seed_rows),
            "acquisition_started_utc": _timestamp_text(started_dt),
            "acquisition_completed_utc": _timestamp_text(completed_dt),
            "activation_boundary_status": ACTIVATION_BOUNDARY_STATUS,
            "activation_boundary_validation": ACTIVATION_BOUNDARY_VALIDATION,
            "activation_status": ACTIVATION_STATUS,
            "study_calendar_generated": STUDY_CALENDAR_GENERATED,
            "portfolio_simulation_started": 0,
            "candidate_generation_started": 0,
            "profit_calculation_started": 0,
        }
        _write_bytes(staging / "seed_manifest.json", canonical_json_bytes(seed_manifest_value))
        os.replace(str(staging), str(destination))
        staging = None
        return seed_manifest_value
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "ACTIVATION_BOUNDARY_STATUS",
    "ACTIVATION_BOUNDARY_VALIDATION",
    "ACTIVATION_STATUS",
    "COLLECTOR_COMMIT",
    "CONFIRMATION",
    "DEFAULT_REQUEST_END_EXCLUSIVE",
    "DEFAULT_REQUEST_START",
    "DEFAULT_SEED_CUTOFF",
    "DESIGN_COMMIT",
    "EXPECTED_TICKER_COUNT",
    "EXPECTED_TICKER_LIST_SHA256",
    "EXPECTED_UNIVERSE_CSV_SHA256",
    "LATEST_PREREGISTRATION_UTC",
    "MODE",
    "STUDY_CALENDAR_GENERATED",
    "V7SeedAcquisitionBlocked",
    "acquire_seed_bundle",
    "canonical_json_bytes",
    "canonical_rows_sha256",
    "canonical_seed_csv_bytes",
    "sha256_bytes",
    "validate_canonical_split_events",
    "validate_universe_file",
]
