"""V8 raw-only historical OHLCV acquisition for the T1/T2 fresh ticker blocks.

This module never imports, reads, writes, or otherwise touches any V7
activation manifest or V7 durable study root. It does reuse
``src.v7_yahoo_collector`` -- the already-accepted, generic, single-ticker
Yahoo Chart request builder and canonical parser that V7's own daily and
seed acquisition modules are themselves built on. Reusing it is read-only
(a plain import); this module makes no edit to ``v7_yahoo_collector.py`` or
any other V7 file.

Only ``T1`` (Layer B validation) and ``T2`` (Layer C sealed holdout) may be
acquired here; ``T3`` is a reserve and acquiring it is unconditionally
prohibited (`V8_HISTORICAL_RESEARCH_DESIGN.md` Decision 6). There is no
"acquire everything" option -- each call acquires exactly one block.

This module computes and stores nothing beyond raw payload bytes, integrity
counts, and hashes: no return, moving average, volatility, signal,
candidate, ranking, trade, portfolio, or profit value is ever computed or
printed here (`V8_HISTORICAL_RESEARCH_DESIGN.md` Sec 6, raw-only acquisition
constraint). A block acquired here is RAW and, for T2, PROCEDURALLY SEALED;
opening it for research is a separate, later, gated action this module does
not perform.

Tests inject an opener, clock, monotonic clock, and sleeper; importing this
module performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.v7_yahoo_collector import (
    FRAME_FIELDS,
    HOST,
    V7YahooCollectorBlocked,
    canonical_ticker,
    fetch_chart_once,
)

from src.v8_partition import (
    DESIGN_COMMIT,
    STUDY_NAME,
    V8PartitionBlocked,
    require_absolute_output_path_outside_repository,
)

SCHEMA_VERSION = "V8_HISTORICAL_ACQUISITION_V1"
MODE = "V8_RAW_HISTORICAL_ACQUISITION"
DATA_SOURCE = "Yahoo Chart"
DATA_SOURCE_HOST = HOST
DATA_SOURCE_SCHEMA = "Yahoo Chart v8/finance/chart interval=1d events=div,splits includeAdjustedClose=true"

REQUEST_START = "2016-04-01"
REQUEST_END_EXCLUSIVE = "2026-01-01"

MIN_REQUEST_INTERVAL_SECONDS = 2.0
RETRY_COUNT = 0

ALLOWED_ACQUISITION_BLOCKS = ("T1", "T2")
PROHIBITED_ACQUISITION_BLOCKS = ("T0", "T3", "T_spare")

BLOCK_ROLE = {"T1": "VALIDATION", "T2": "SEALED_HOLDOUT"}
BLOCK_STATUS = {"T1": "RAW_ACQUIRED_NOT_OPENED", "T2": "RAW_ACQUIRED_SEALED"}
BLOCK_SEALED = {"T1": False, "T2": True}

ACQUISITIONS_DIRNAME = "acquisitions"
RAW_DIRNAME = "raw"
MANIFEST_FILENAME = "acquisition_manifest.json"
SEALED_FILENAME = "SEALED.json"

ACQUISITION_MANIFEST_FIELDS = (
    "schema_version",
    "study_name",
    "design_commit",
    "partition_manifest_sha256",
    "block",
    "role",
    "status",
    "sealed",
    "research_access_authorized",
    "data_source",
    "data_source_host",
    "data_source_schema",
    "request_start",
    "request_end_exclusive",
    "ticker_count",
    "ticker_list_sha256",
    "request_count",
    "retry_count",
    "http_429_count",
    "success_transport_count",
    "valid_price_row_count",
    "invalid_price_row_count",
    "invalid_reason_counts",
    "split_event_count",
    "payload_manifest",
    "payload_manifest_sha256",
    "canonical_price_rows_sha256",
    "canonical_split_events_sha256",
    "acquisition_started_utc",
    "acquisition_completed_utc",
    "validation_access_count",
    "feature_computation_count",
    "outcome_access_count",
    "sealed_holdout_access_count",
)

PAYLOAD_RECORD_FIELDS = (
    "ticker",
    "payload_sha256",
    "byte_count",
    "canonical_price_rows_sha256",
    "canonical_split_events_sha256",
    "valid_price_row_count",
    "invalid_price_row_count",
    "split_event_count",
)


class V8HistoricalAcquisitionBlocked(RuntimeError):
    """Fail-closed historical acquisition transport, schema, or seal error."""

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
        raise V8HistoricalAcquisitionBlocked("NONFINITE_VALUE") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _ticker_list_sha(tickers: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(tickers) + "\n").encode("utf-8"))


def _parse_date(value: object, field: str) -> date:
    if not isinstance(value, str):
        raise V8HistoricalAcquisitionBlocked("INVALID_DATE:" + field)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise V8HistoricalAcquisitionBlocked("INVALID_DATE:" + field) from error
    if parsed.isoformat() != value:
        raise V8HistoricalAcquisitionBlocked("INVALID_DATE:" + field)
    return parsed


def _utc_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8HistoricalAcquisitionBlocked("UTC_TIMESTAMP_INVALID:" + field)
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [{field: row[field] for field in FRAME_FIELDS} for row in rows],
        key=lambda row: (str(row["ticker"]), str(row["trading_date"])),
    )


def _classify_error(error: BaseException) -> tuple[str, bool]:
    code = getattr(error, "code", None)
    if code == 429:
        return "HTTP_STATUS_429", True
    reason = getattr(error, "reason", str(error))
    text = str(reason)
    return text, text == "HTTP_STATUS_429"


class _RecordingResponse:
    """Captures raw response bytes so the exact wire payload can be hashed
    and stored, independent of how the collector parsed it."""

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


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def acquire_historical_block_bundle(
    *,
    output_root: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    block: str,
    tickers: Sequence[str],
    partition_manifest_sha256: str,
    opener: Callable[[Any], Any],
    clock: Callable[[], datetime],
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    request_start: str = REQUEST_START,
    request_end_exclusive: str = REQUEST_END_EXCLUSIVE,
) -> dict[str, Any]:
    """Acquire and atomically publish one block's raw historical OHLCV bundle.

    ``block`` must be exactly ``"T1"`` or ``"T2"``. Any other value --
    including ``"T3"`` -- is rejected before any network access. Every row
    is fetched via the already-accepted ``v7_yahoo_collector.fetch_chart_once``
    transport (sequential, one HTTP request per ticker); this function adds
    no additional network path.

    Any invalid, non-finite, or otherwise malformed OHLCV row anywhere in a
    ticker's response blocks the whole acquisition -- unlike V7's daily
    acquisition, which tolerates a single missing day, a multi-year bulk
    historical fetch treats any row anomaly as fail-closed rather than as a
    per-row soft classification.
    """
    if block not in ALLOWED_ACQUISITION_BLOCKS:
        raise V8HistoricalAcquisitionBlocked("V8_BLOCK_ACQUISITION_PROHIBITED:" + str(block))

    start = _parse_date(request_start, "request_start")
    end = _parse_date(request_end_exclusive, "request_end_exclusive")
    if not start < end:
        raise V8HistoricalAcquisitionBlocked("REQUEST_DATE_BOUNDS_INVALID")

    tickers_list = list(tickers)
    if not tickers_list or len(set(tickers_list)) != len(tickers_list):
        raise V8HistoricalAcquisitionBlocked("V8_TICKER_LIST_INVALID")
    for ticker in tickers_list:
        try:
            if canonical_ticker(ticker) != ticker:
                raise V8HistoricalAcquisitionBlocked("V8_TICKER_NOT_CANONICAL:" + str(ticker))
        except V7YahooCollectorBlocked as error:
            raise V8HistoricalAcquisitionBlocked("V8_TICKER_NOT_CANONICAL:" + str(ticker)) from error

    try:
        output_path = require_absolute_output_path_outside_repository(output_root, repository_root)
    except V8PartitionBlocked as error:
        raise V8HistoricalAcquisitionBlocked(error.reason) from error

    acquisitions_root = output_path / ACQUISITIONS_DIRNAME
    final_dir = acquisitions_root / block
    if final_dir.exists():
        raise V8HistoricalAcquisitionBlocked("V8_ACQUISITION_ALREADY_EXISTS:" + block)
    acquisitions_root.mkdir(parents=True, exist_ok=True)
    if any(entry.name.startswith(block + ".staging-") for entry in acquisitions_root.iterdir()):
        raise V8HistoricalAcquisitionBlocked("V8_PARTIAL_ACQUISITION_COMMIT:" + block)

    started_dt = _utc_timestamp(clock(), "acquisition_started_utc")

    staging: Path | None = None
    try:
        staging = Path(tempfile.mkdtemp(prefix=f"{block}.staging-", dir=str(acquisitions_root)))
        (staging / RAW_DIRNAME).mkdir()

        payload_manifest: list[dict[str, Any]] = []
        all_price_rows: list[dict[str, Any]] = []
        all_split_rows: list[dict[str, Any]] = []
        invalid_reason_counts: Counter[str] = Counter()
        request_count = 0
        http_429_count = 0
        success_transport_count = 0
        previous_start: float | None = None

        for index, ticker in enumerate(tickers_list):
            previous_start = _wait_for_next_request_start(index, previous_start, monotonic_clock, sleep_fn)
            capture = bytearray()

            def recording_opener(request_obj: Any, *, _capture: bytearray = capture) -> Any:
                return _RecordingResponse(opener(request_obj), _capture)

            request_count += 1
            try:
                parsed = fetch_chart_once(
                    ticker, request_start, request_end_exclusive, opener=recording_opener
                )
            except V7YahooCollectorBlocked as error:
                reason = error.reason
                if reason == "HTTP_STATUS_429":
                    http_429_count += 1
                raise V8HistoricalAcquisitionBlocked("TICKER_" + str(ticker) + ":" + reason) from error
            except BaseException as error:
                reason, is_429 = _classify_error(error)
                if is_429:
                    http_429_count += 1
                raise V8HistoricalAcquisitionBlocked("TICKER_" + str(ticker) + ":" + reason) from error

            payload_bytes = bytes(capture)
            if sha256_bytes(payload_bytes) != parsed.get("payload_sha256"):
                raise V8HistoricalAcquisitionBlocked("RAW_PAYLOAD_SHA_MISMATCH:" + ticker)
            if len(payload_bytes) != parsed.get("byte_count"):
                raise V8HistoricalAcquisitionBlocked("RAW_PAYLOAD_BYTE_COUNT_MISMATCH:" + ticker)

            invalid_rows = parsed["invalid_price_rows"]
            if invalid_rows:
                for row in invalid_rows:
                    invalid_reason_counts[str(row["reason"])] += 1
                raise V8HistoricalAcquisitionBlocked("MALFORMED_OHLCV:" + ticker)

            _write_bytes(staging / RAW_DIRNAME / (ticker + ".json"), payload_bytes)

            valid_rows = [dict(row) for row in parsed["valid_price_rows"]]
            split_rows = [dict(row) for row in parsed["canonical_split_events"]]
            all_price_rows.extend(valid_rows)
            all_split_rows.extend(split_rows)

            payload_manifest.append({
                "ticker": ticker,
                "payload_sha256": parsed["payload_sha256"],
                "byte_count": parsed["byte_count"],
                "canonical_price_rows_sha256": parsed["canonical_price_rows_sha256"],
                "canonical_split_events_sha256": parsed["canonical_split_events_sha256"],
                "valid_price_row_count": len(valid_rows),
                "invalid_price_row_count": 0,
                "split_event_count": len(split_rows),
            })
            success_transport_count += 1

        keyed_rows: set[tuple[str, str]] = set()
        for row in all_price_rows:
            key = (str(row["ticker"]), str(row["trading_date"]))
            if key in keyed_rows:
                raise V8HistoricalAcquisitionBlocked("DUPLICATE_TICKER_DATE")
            keyed_rows.add(key)
        canonical_rows = _canonical_rows(all_price_rows)
        canonical_splits = sorted(all_split_rows, key=lambda row: (row["effective_date"], row["ticker"]))

        completed_dt = _utc_timestamp(clock(), "acquisition_completed_utc")
        if completed_dt < started_dt:
            raise V8HistoricalAcquisitionBlocked("ACQUISITION_CLOCK_NONMONOTONIC")

        payload_manifest_bytes = canonical_json_bytes(payload_manifest)

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "study_name": STUDY_NAME,
            "design_commit": DESIGN_COMMIT,
            "partition_manifest_sha256": partition_manifest_sha256,
            "block": block,
            "role": BLOCK_ROLE[block],
            "status": BLOCK_STATUS[block],
            "sealed": BLOCK_SEALED[block],
            "research_access_authorized": False,
            "data_source": DATA_SOURCE,
            "data_source_host": DATA_SOURCE_HOST,
            "data_source_schema": DATA_SOURCE_SCHEMA,
            "request_start": request_start,
            "request_end_exclusive": request_end_exclusive,
            "ticker_count": len(tickers_list),
            "ticker_list_sha256": _ticker_list_sha(tickers_list),
            "request_count": request_count,
            "retry_count": RETRY_COUNT,
            "http_429_count": http_429_count,
            "success_transport_count": success_transport_count,
            "valid_price_row_count": len(canonical_rows),
            "invalid_price_row_count": 0,
            "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
            "split_event_count": len(canonical_splits),
            "payload_manifest": payload_manifest,
            "payload_manifest_sha256": sha256_bytes(payload_manifest_bytes),
            "canonical_price_rows_sha256": canonical_sha256(canonical_rows),
            "canonical_split_events_sha256": canonical_sha256(canonical_splits),
            "acquisition_started_utc": _timestamp_text(started_dt),
            "acquisition_completed_utc": _timestamp_text(completed_dt),
            "validation_access_count": 0,
            "feature_computation_count": 0,
            "outcome_access_count": 0,
            "sealed_holdout_access_count": 0,
        }
        if set(manifest) != set(ACQUISITION_MANIFEST_FIELDS):
            raise V8HistoricalAcquisitionBlocked("MANIFEST_SCHEMA_INVALID")

        _write_bytes(staging / MANIFEST_FILENAME, canonical_json_bytes(manifest))
        if block == "T2":
            sealed_record = {
                "sealed": True,
                "research_access_authorized": False,
                "note": (
                    "Procedural seal, not cryptographic. Opening this block for "
                    "feature generation, candidate generation, validation, "
                    "backtest, or profit evaluation requires the FROZEN_FINAL_"
                    "CANDIDATE gate; the official V8 access-guard API in this "
                    "module BLOCKs every such call while sealed=true."
                ),
            }
            _write_bytes(staging / SEALED_FILENAME, canonical_json_bytes(sealed_record))

        os.replace(str(staging), str(final_dir))
        staging = None
        return manifest
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def read_acquisition_manifest(output_root: str | os.PathLike[str], block: str) -> dict[str, Any]:
    """Read-only load of a previously published block manifest."""
    manifest_path = Path(output_root) / ACQUISITIONS_DIRNAME / block / MANIFEST_FILENAME
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise V8HistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_READ_FAILED") from error
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8HistoricalAcquisitionBlocked("ACQUISITION_MANIFEST_INVALID_JSON") from error
    if not isinstance(manifest, Mapping) or set(manifest) != set(ACQUISITION_MANIFEST_FIELDS):
        raise V8HistoricalAcquisitionBlocked("MANIFEST_SCHEMA_INVALID")
    return dict(manifest)


# ---------------------------------------------------------------------------
# T2 sealed-holdout access guard (procedural, not cryptographic)
# ---------------------------------------------------------------------------


class V8SealedHoldoutBlocked(RuntimeError):
    """Raised by the official V8 access-guard API when a caller attempts to
    open a sealed block for research use.

    This is a PROCEDURAL seal: it works by refusing to proceed when the
    acquisition manifest it is handed says ``sealed=true``. It does not
    encrypt, checksum-lock, or otherwise make the underlying raw files
    physically inaccessible -- a caller that reads
    ``acquisitions/T2/raw/*.json`` directly, bypassing this API, is not
    stopped by this guard. The guarantee this module provides is that the
    *official* V8 research code path cannot silently open T2 before
    ``FROZEN_FINAL_CANDIDATE`` (`V8_HISTORICAL_RESEARCH_DESIGN.md` Sec 5.4).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_research_access_authorized(acquisition_manifest: Mapping[str, Any], operation: str) -> None:
    """Fail closed unless the manifest explicitly authorizes research access.

    Distinguishes two BLOCK reasons for the same underlying refusal, so
    callers and tests can tell a permanently-sealed block (T2, blocked by
    procedural seal) apart from a block that simply has not yet been
    authorized for its layer (e.g. T1 before Layer B validation formally
    opens it) -- neither path implemented by this module grants access on
    its own; both require ``research_access_authorized: true`` to already be
    recorded in the manifest handed to this guard.
    """
    if not isinstance(acquisition_manifest, Mapping) or "sealed" not in acquisition_manifest:
        raise V8SealedHoldoutBlocked("ACQUISITION_MANIFEST_INVALID:" + operation)
    if acquisition_manifest["sealed"] is True:
        raise V8SealedHoldoutBlocked("SEALED_HOLDOUT_ACCESS_DENIED:" + operation)
    if acquisition_manifest.get("research_access_authorized") is not True:
        raise V8SealedHoldoutBlocked("RESEARCH_ACCESS_NOT_AUTHORIZED:" + operation)


def open_for_feature_generation(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "feature_generation")


def open_for_candidate_generation(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "candidate_generation")


def open_for_validation(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "validation")


def open_for_backtest(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "backtest")


def open_for_profit_evaluation(acquisition_manifest: Mapping[str, Any]) -> None:
    _require_research_access_authorized(acquisition_manifest, "profit_evaluation")


__all__ = [
    "ACQUISITIONS_DIRNAME",
    "ACQUISITION_MANIFEST_FIELDS",
    "ALLOWED_ACQUISITION_BLOCKS",
    "BLOCK_ROLE",
    "BLOCK_SEALED",
    "BLOCK_STATUS",
    "DATA_SOURCE",
    "DATA_SOURCE_HOST",
    "DATA_SOURCE_SCHEMA",
    "MANIFEST_FILENAME",
    "MIN_REQUEST_INTERVAL_SECONDS",
    "MODE",
    "PAYLOAD_RECORD_FIELDS",
    "PROHIBITED_ACQUISITION_BLOCKS",
    "RAW_DIRNAME",
    "REQUEST_END_EXCLUSIVE",
    "REQUEST_START",
    "RETRY_COUNT",
    "SCHEMA_VERSION",
    "SEALED_FILENAME",
    "V8HistoricalAcquisitionBlocked",
    "V8SealedHoldoutBlocked",
    "acquire_historical_block_bundle",
    "canonical_json_bytes",
    "canonical_sha256",
    "open_for_backtest",
    "open_for_candidate_generation",
    "open_for_feature_generation",
    "open_for_profit_evaluation",
    "open_for_validation",
    "read_acquisition_manifest",
    "sha256_bytes",
]
