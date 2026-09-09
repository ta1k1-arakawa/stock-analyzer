"""Static Yahoo Chart transport and canonical parser for V7.

This module does not acquire seed data or define a study calendar.  Its
network-capable boundary is ``fetch_chart_once``; tests inject a fake opener.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timezone
from typing import Any, Callable, Mapping, Sequence
from urllib import parse, request
from zoneinfo import ZoneInfo


HOST = "query1.finance.yahoo.com"
ENDPOINT = "/v8/finance/chart/{ticker}.T"
INTERVAL = "1d"
EVENTS = "div,splits"
INCLUDE_ADJUSTED_CLOSE = "true"
HEADERS = {
    "User-Agent": "V7-Forward-Collector/1.0",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",
    "Connection": "close",
}
FRAME_FIELDS = (
    "ticker",
    "trading_date",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "adj_close",
    "raw_volume",
)
JST = ZoneInfo("Asia/Tokyo")


class V7YahooCollectorBlocked(ValueError):
    """Fail closed for transport, schema, provenance, or canonical errors."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def canonical_ticker(value: object) -> str:
    symbol = str(value).strip().upper()
    if symbol.endswith(".T"):
        symbol = symbol[:-2]
    if symbol.endswith(".0") and symbol[:-2].isdigit():
        symbol = symbol[:-2]
    if not symbol:
        raise V7YahooCollectorBlocked("EMPTY_TICKER")
    return symbol


def _parse_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise V7YahooCollectorBlocked("INVALID_DATE:" + field)
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise V7YahooCollectorBlocked("INVALID_DATE:" + field) from error
    if parsed.isoformat() != value:
        raise V7YahooCollectorBlocked("INVALID_DATE:" + field)
    return parsed


def _epoch(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp())


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def build_chart_request(ticker: object, start_date: str, end_exclusive_date: str) -> request.Request:
    canonical = canonical_ticker(ticker)
    start = _parse_date(start_date, "start")
    end = _parse_date(end_exclusive_date, "end_exclusive")
    if not start < end:
        raise V7YahooCollectorBlocked("INVALID_REQUEST_DATE_ORDER")
    query = parse.urlencode(
        (
            ("period1", str(_epoch(start))),
            ("period2", str(_epoch(end))),
            ("interval", INTERVAL),
            ("events", EVENTS),
            ("includeAdjustedClose", INCLUDE_ADJUSTED_CLOSE),
        )
    )
    url = "https://" + HOST + ENDPOINT.format(ticker=parse.quote(canonical, safe="")) + "?" + query
    return request.Request(url, headers=dict(HEADERS), method="GET")


def validate_response_host(response_url: str) -> str:
    try:
        parsed = parse.urlparse(response_url)
    except ValueError as error:
        raise V7YahooCollectorBlocked("RESPONSE_HOST_MISMATCH") from error
    if parsed.scheme != "https" or parsed.hostname != HOST:
        raise V7YahooCollectorBlocked("RESPONSE_HOST_MISMATCH")
    return parsed.hostname


def _jst_date(epoch_value: Any) -> date:
    numeric = _number(epoch_value)
    if numeric is None and isinstance(epoch_value, str):
        try:
            parsed = float(epoch_value)
        except ValueError:
            parsed = float("nan")
        numeric = parsed if math.isfinite(parsed) else None
    if numeric is None:
        raise V7YahooCollectorBlocked("TIMESTAMP_INVALID")
    return datetime.fromtimestamp(numeric, tz=timezone.utc).astimezone(JST).date()


def _payload_root(payload_bytes: bytes) -> tuple[dict[str, Any], str, int]:
    payload_sha256 = hashlib.sha256(payload_bytes).hexdigest()
    try:
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V7YahooCollectorBlocked("PAYLOAD_JSON_INVALID") from error
    if not isinstance(payload, Mapping):
        raise V7YahooCollectorBlocked("PAYLOAD_ROOT_INVALID")
    chart = payload.get("chart")
    if not isinstance(chart, Mapping) or "error" not in chart or chart.get("error") is not None:
        raise V7YahooCollectorBlocked("CHART_ERROR")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], Mapping):
        raise V7YahooCollectorBlocked("CHART_RESULT_INVALID")
    return dict(results[0]), payload_sha256, len(payload_bytes)


def _row_invalid_reason(values: Mapping[str, Any]) -> str | None:
    for field in ("open", "high", "low", "close", "adjclose"):
        value = _number(values.get(field))
        if value is None:
            return "NONFINITE_" + field.upper()
        if value <= 0:
            return "NONPOSITIVE_" + field.upper()
    volume = _number(values.get("volume"))
    if volume is None:
        return "NONFINITE_VOLUME"
    if volume < 0:
        return "NEGATIVE_VOLUME"
    return None


def _series(result: Mapping[str, Any], section: str, name: str, length: int) -> list[Any]:
    indicators = result.get("indicators")
    if not isinstance(indicators, Mapping):
        raise V7YahooCollectorBlocked("INDICATORS_MISSING")
    section_value = indicators.get(section)
    if not isinstance(section_value, list) or len(section_value) != 1 or not isinstance(section_value[0], Mapping):
        raise V7YahooCollectorBlocked("INDICATOR_SECTION_INVALID:" + section)
    values = section_value[0].get(name)
    if not isinstance(values, list) or len(values) != length:
        raise V7YahooCollectorBlocked("ARRAY_LENGTH_MISMATCH:" + name)
    return list(values)


def _parse_split_ratio(value: Any) -> tuple[float, float] | None:
    if isinstance(value, str):
        pieces = value.split(":")
        if len(pieces) != 2:
            raise V7YahooCollectorBlocked("SPLIT_RATIO_INVALID")
        try:
            numerator = float(pieces[0])
            denominator = float(pieces[1])
        except ValueError:
            numerator = None
            denominator = None
    else:
        return None
    if numerator is None or denominator is None or numerator <= 0 or denominator <= 0:
        raise V7YahooCollectorBlocked("SPLIT_RATIO_INVALID")
    return numerator, denominator


def _parse_splits(result: Mapping[str, Any], ticker: str, start: date, end: date) -> list[dict[str, Any]]:
    events = result.get("events")
    if events is None:
        return []
    if not isinstance(events, Mapping):
        raise V7YahooCollectorBlocked("EVENTS_INVALID")
    splits = events.get("splits", {})
    if splits is None:
        return []
    if not isinstance(splits, Mapping):
        raise V7YahooCollectorBlocked("SPLITS_INVALID")
    parsed: list[dict[str, Any]] = []
    seen: set[date] = set()
    for key, event in sorted(splits.items(), key=lambda item: str(item[0])):
        if not isinstance(event, Mapping):
            raise V7YahooCollectorBlocked("SPLIT_EVENT_INVALID")
        epoch_value = event.get("date", key)
        effective_date = _jst_date(epoch_value)
        if not start <= effective_date < end:
            raise V7YahooCollectorBlocked("SPLIT_OUT_OF_REQUEST_WINDOW")
        if effective_date in seen:
            raise V7YahooCollectorBlocked("DUPLICATE_SPLIT_EVENT")
        seen.add(effective_date)
        ratio = _parse_split_ratio(event.get("splitRatio"))
        numerator = _number(event.get("numerator"))
        denominator = _number(event.get("denominator"))
        if numerator is None or denominator is None:
            if ratio is None:
                raise V7YahooCollectorBlocked("SPLIT_NUMERATOR_DENOMINATOR_MISSING")
            numerator, denominator = ratio
        if numerator <= 0 or denominator <= 0:
            raise V7YahooCollectorBlocked("SPLIT_NUMERATOR_DENOMINATOR_INVALID")
        if ratio is not None and not math.isclose(numerator / denominator, ratio[0] / ratio[1], rel_tol=0, abs_tol=1e-12):
            raise V7YahooCollectorBlocked("SPLIT_RATIO_MISMATCH")
        parsed.append({
            "ticker": ticker,
            "effective_date": effective_date.isoformat(),
            "numerator": numerator,
            "denominator": denominator,
            "split_ratio": numerator / denominator,
        })
    return sorted(parsed, key=lambda row: (row["effective_date"], row["ticker"]))


def parse_chart_payload(
    payload_bytes: bytes,
    expected_ticker: object,
    request_start: str,
    request_end_exclusive: str,
    *,
    response_host: str = HOST,
) -> dict[str, Any]:
    ticker = canonical_ticker(expected_ticker)
    start = _parse_date(request_start, "start")
    end = _parse_date(request_end_exclusive, "end_exclusive")
    if not start < end:
        raise V7YahooCollectorBlocked("INVALID_REQUEST_DATE_ORDER")
    if response_host != HOST:
        raise V7YahooCollectorBlocked("RESPONSE_HOST_MISMATCH")
    result, payload_sha256, byte_count = _payload_root(payload_bytes)
    meta = result.get("meta")
    if not isinstance(meta, Mapping):
        raise V7YahooCollectorBlocked("METADATA_MISSING")
    yahoo_symbol = meta.get("symbol")
    if canonical_ticker(yahoo_symbol) != ticker:
        raise V7YahooCollectorBlocked("SYMBOL_MISMATCH")
    timestamps = result.get("timestamp")
    if not isinstance(timestamps, list) or not timestamps:
        raise V7YahooCollectorBlocked("TIMESTAMP_MISSING")
    length = len(timestamps)
    opens = _series(result, "quote", "open", length)
    highs = _series(result, "quote", "high", length)
    lows = _series(result, "quote", "low", length)
    closes = _series(result, "quote", "close", length)
    volumes = _series(result, "quote", "volume", length)
    adjcloses = _series(result, "adjclose", "adjclose", length)
    dates = [_jst_date(value) for value in timestamps]
    if any(not start <= observed < end for observed in dates):
        raise V7YahooCollectorBlocked("OUT_OF_REQUEST_WINDOW")
    if len(set(dates)) != len(dates):
        raise V7YahooCollectorBlocked("DUPLICATE_TRADING_DATE")
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for index, trading_date in enumerate(dates):
        values = {
            "open": opens[index],
            "high": highs[index],
            "low": lows[index],
            "close": closes[index],
            "adjclose": adjcloses[index],
            "volume": volumes[index],
        }
        reason = _row_invalid_reason(values)
        if reason is not None:
            invalid_rows.append({"ticker": ticker, "trading_date": trading_date.isoformat(), "reason": reason})
            continue
        valid_rows.append({
            "ticker": ticker,
            "trading_date": trading_date.isoformat(),
            "raw_open": float(values["open"]),
            "raw_high": float(values["high"]),
            "raw_low": float(values["low"]),
            "raw_close": float(values["close"]),
            "adj_close": float(values["adjclose"]),
            "raw_volume": float(values["volume"]),
        })
    split_events = _parse_splits(result, ticker, start, end)
    valid_rows.sort(key=lambda row: (row["trading_date"], row["ticker"]))
    invalid_rows.sort(key=lambda row: (row["trading_date"], row["ticker"], row["reason"]))
    return {
        "ticker": ticker,
        "yahoo_symbol": str(yahoo_symbol),
        "request_start": request_start,
        "request_end_exclusive": request_end_exclusive,
        "request_host": HOST,
        "response_host": response_host,
        "payload_sha256": payload_sha256,
        "byte_count": byte_count,
        "valid_price_rows": valid_rows,
        "invalid_price_rows": invalid_rows,
        "canonical_split_events": split_events,
        "canonical_price_rows_sha256": _canonical_sha256(valid_rows),
        "canonical_split_events_sha256": _canonical_sha256(split_events),
        "study_calendar_generated": False,
    }


def fetch_chart_once(
    ticker: object,
    start_date: str,
    end_exclusive_date: str,
    opener: Callable[..., Any] = request.urlopen,
) -> dict[str, Any]:
    chart_request = build_chart_request(ticker, start_date, end_exclusive_date)
    response = opener(chart_request)
    try:
        status = getattr(response, "status", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            status = getcode() if callable(getcode) else None
        if status != 200:
            raise V7YahooCollectorBlocked("HTTP_STATUS_" + str(status))
        response_url = getattr(response, "url", None)
        if not isinstance(response_url, str):
            raise V7YahooCollectorBlocked("RESPONSE_HOST_MISMATCH")
        response_host = validate_response_host(response_url)
        payload_bytes = response.read()
        if not isinstance(payload_bytes, bytes):
            raise V7YahooCollectorBlocked("RESPONSE_BYTES_INVALID")
        return parse_chart_payload(
            payload_bytes,
            ticker,
            start_date,
            end_exclusive_date,
            response_host=response_host,
        )
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


__all__ = [
    "ENDPOINT",
    "EVENTS",
    "FRAME_FIELDS",
    "HEADERS",
    "HOST",
    "V7YahooCollectorBlocked",
    "build_chart_request",
    "canonical_ticker",
    "fetch_chart_once",
    "parse_chart_payload",
    "validate_response_host",
]
