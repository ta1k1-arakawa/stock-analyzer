from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import pytest

from src.v7_forward_protocol import validate_seed_rows
from src.v7_yahoo_collector import (
    HEADERS,
    HOST,
    V7YahooCollectorBlocked,
    build_chart_request,
    canonical_ticker,
    fetch_chart_once,
    parse_chart_payload,
    validate_response_host,
)


def _epoch(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp())


def _payload(
    symbol: str = "7203.T",
    days: tuple[date, ...] = (date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)),
    *,
    error=None,
    events=None,
    quote_overrides=None,
    include_adjclose=True,
    timestamp_values=None,
) -> bytes:
    quote = {
        "open": [100.0 + index for index in range(len(days))],
        "high": [101.0 + index for index in range(len(days))],
        "low": [99.0 + index for index in range(len(days))],
        "close": [100.5 + index for index in range(len(days))],
        "volume": [1000.0 + index for index in range(len(days))],
    }
    if quote_overrides:
        for field, values in quote_overrides.items():
            quote[field] = values
    indicators = {"quote": [quote]}
    if include_adjclose:
        indicators["adjclose"] = [{"adjclose": [100.25 + index for index in range(len(days))]}]
    timestamps = timestamp_values if timestamp_values is not None else [_epoch(day) for day in days]
    body = {
        "chart": {
            "error": error,
            "result": [{
                "meta": {"symbol": symbol},
                "timestamp": timestamps,
                "indicators": indicators,
                "events": events if events is not None else {},
            }],
        }
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _parse(payload: bytes, start="2020-01-01", end="2020-01-10"):
    return parse_chart_payload(payload, "7203", start, end, response_host=HOST)


class FakeResponse:
    def __init__(self, payload: bytes, status=200, url=None):
        self.payload = payload
        self.status = status
        self.url = url or f"https://{HOST}/v8/finance/chart/7203.T"
        self.closed = False

    def read(self):
        return self.payload

    def close(self):
        self.closed = True


def test_ticker_canonicalization():
    assert canonical_ticker("7203") == "7203"
    assert canonical_ticker("7203.T") == "7203"
    assert canonical_ticker("7203.0") == "7203"


def test_request_url_is_deterministic_and_uses_fixed_host():
    first = build_chart_request("7203.0", "2020-01-01", "2020-01-10")
    second = build_chart_request("7203", "2020-01-01", "2020-01-10")
    assert first.full_url == second.full_url
    parsed = urlparse(first.full_url)
    assert parsed.scheme == "https"
    assert parsed.hostname == HOST
    assert parsed.path == "/v8/finance/chart/7203.T"
    assert list(parse_qs(parsed.query).keys()) == ["period1", "period2", "interval", "events", "includeAdjustedClose"]


def test_request_uses_explicit_utc_epoch_bounds():
    req = build_chart_request("7203", "2020-01-01", "2020-01-10")
    query = parse_qs(urlparse(req.full_url).query)
    assert query["period1"] == [str(_epoch(date(2020, 1, 1)))]
    assert query["period2"] == [str(_epoch(date(2020, 1, 10)))]


@pytest.mark.parametrize(
    "start,end",
    [("2020-01-10", "2020-01-01"), ("2020-01-01", "2020-01-01"), ("bad", "2020-01-01")],
)
def test_invalid_request_bounds_fail_closed(start, end):
    with pytest.raises(V7YahooCollectorBlocked):
        build_chart_request("7203", start, end)


def test_response_host_validation_is_pinned_and_https_only():
    assert validate_response_host(f"https://{HOST}/x") == HOST
    with pytest.raises(V7YahooCollectorBlocked, match="RESPONSE_HOST_MISMATCH"):
        validate_response_host("https://evil.example/x")
    with pytest.raises(V7YahooCollectorBlocked, match="RESPONSE_HOST_MISMATCH"):
        validate_response_host(f"http://{HOST}/x")


def test_fixed_headers_are_present():
    req = build_chart_request("7203", "2020-01-01", "2020-01-10")
    headers = {key.lower(): value for key, value in req.header_items()}
    for key, value in HEADERS.items():
        assert headers[key.lower()] == value


def test_fetch_rejects_response_host_mismatch():
    with pytest.raises(V7YahooCollectorBlocked, match="RESPONSE_HOST_MISMATCH"):
        fetch_chart_once("7203", "2020-01-01", "2020-01-10", lambda req: FakeResponse(_payload(), url="https://evil.example/x"))


def test_symbol_mismatch_fails_closed():
    with pytest.raises(V7YahooCollectorBlocked, match="SYMBOL_MISMATCH"):
        _parse(_payload(symbol="9984.T"))


def test_chart_error_fails_closed():
    with pytest.raises(V7YahooCollectorBlocked, match="CHART_ERROR"):
        _parse(_payload(error={"code": "Bad"}))


def test_missing_quote_field_fails_closed():
    payload = json.loads(_payload().decode())
    del payload["chart"]["result"][0]["indicators"]["quote"][0]["high"]
    with pytest.raises(V7YahooCollectorBlocked, match="ARRAY_LENGTH_MISMATCH:high"):
        _parse(json.dumps(payload).encode())


def test_missing_adjclose_fails_closed():
    with pytest.raises(V7YahooCollectorBlocked, match="INDICATOR_SECTION_INVALID:adjclose"):
        _parse(_payload(include_adjclose=False))


def test_array_length_mismatch_fails_closed():
    payload = json.loads(_payload().decode())
    payload["chart"]["result"][0]["indicators"]["quote"][0]["open"] = [100.0]
    with pytest.raises(V7YahooCollectorBlocked, match="ARRAY_LENGTH_MISMATCH:open"):
        _parse(json.dumps(payload).encode())


def test_valid_payload_returns_seed_schema_rows():
    result = _parse(_payload())
    assert result["ticker"] == "7203"
    assert result["yahoo_symbol"] == "7203.T"
    assert list(result["valid_price_rows"][0]) == [
        "ticker", "trading_date", "raw_open", "raw_high", "raw_low", "raw_close", "adj_close", "raw_volume"
    ]
    assert len(result["valid_price_rows"]) == 3


def test_timestamp_is_converted_from_utc_to_jst_date():
    payload = _payload(days=(date(2020, 1, 1),), timestamp_values=[_epoch(date(2020, 1, 1), 15)])
    assert _parse(payload)["valid_price_rows"][0]["trading_date"] == "2020-01-02"


def test_invalid_numeric_rows_are_audited_and_excluded():
    payload = _payload(quote_overrides={"close": [100.5, float("nan"), 102.5]})
    result = _parse(payload)
    assert len(result["valid_price_rows"]) == 2
    assert result["invalid_price_rows"] == [{"ticker": "7203", "trading_date": "2020-01-03", "reason": "NONFINITE_CLOSE"}]


def test_duplicate_jst_date_rejects_payload():
    payload = _payload(timestamp_values=[_epoch(date(2020, 1, 2), 0), _epoch(date(2020, 1, 2), 12), _epoch(date(2020, 1, 4), 0)])
    with pytest.raises(V7YahooCollectorBlocked, match="DUPLICATE_TRADING_DATE"):
        _parse(payload)


def test_out_of_request_window_rejects_payload():
    with pytest.raises(V7YahooCollectorBlocked, match="OUT_OF_REQUEST_WINDOW"):
        _parse(_payload(days=(date(2019, 12, 31),)))


def test_split_events_are_canonicalized():
    events = {"splits": {str(_epoch(date(2020, 1, 3))): {"date": _epoch(date(2020, 1, 3)), "numerator": 2, "denominator": 1}}}
    result = _parse(_payload(events=events))
    assert result["canonical_split_events"] == [{
        "ticker": "7203", "effective_date": "2020-01-03", "numerator": 2.0,
        "denominator": 1.0, "split_ratio": 2.0,
    }]


def test_split_ratio_only_is_supported():
    events = {"splits": {str(_epoch(date(2020, 1, 3))): {"splitRatio": "3:2"}}}
    assert _parse(_payload(events=events))["canonical_split_events"][0]["split_ratio"] == 1.5


def test_duplicate_split_events_reject_payload():
    timestamp = str(_epoch(date(2020, 1, 3)))
    events = {"splits": {timestamp: {"numerator": 2, "denominator": 1}, "x": {"date": _epoch(date(2020, 1, 3)), "numerator": 2, "denominator": 1}}}
    with pytest.raises(V7YahooCollectorBlocked, match="DUPLICATE_SPLIT_EVENT"):
        _parse(_payload(events=events))


def test_price_hash_is_deterministic():
    first = _parse(_payload())
    second = _parse(_payload())
    assert first["canonical_price_rows_sha256"] == second["canonical_price_rows_sha256"]


def test_split_hash_is_deterministic():
    events = {"splits": {str(_epoch(date(2020, 1, 3))): {"numerator": 2, "denominator": 1}}}
    first = _parse(_payload(events=events))
    second = _parse(_payload(events=events))
    assert first["canonical_split_events_sha256"] == second["canonical_split_events_sha256"]


def test_payload_dict_and_event_order_do_not_change_canonical_hashes():
    events = {"splits": {"b": {"date": _epoch(date(2020, 1, 4)), "numerator": 3, "denominator": 2}, "a": {"date": _epoch(date(2020, 1, 3)), "numerator": 2, "denominator": 1}}}
    first_payload = _payload(events=events)
    second_payload = json.dumps(json.loads(first_payload.decode()), separators=(",", ":")).encode()
    first = _parse(first_payload)
    second = _parse(second_payload)
    assert first["canonical_price_rows_sha256"] == second["canonical_price_rows_sha256"]
    assert first["canonical_split_events_sha256"] == second["canonical_split_events_sha256"]


def test_raw_payload_hash_and_byte_count_are_provenance_values():
    payload = _payload()
    result = _parse(payload)
    assert result["payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["byte_count"] == len(payload)
    assert result["payload_sha256"] != result["canonical_price_rows_sha256"]


def test_valid_rows_are_compatible_with_seed_validator():
    payload = _payload(days=tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(252)))
    result = parse_chart_payload(payload, "7203", "2020-01-01", "2021-01-01", response_host=HOST)
    seed_result = validate_seed_rows(result["valid_price_rows"], ["7203"], "2021-01-01")
    assert seed_result["eligible_ticker_count"] == 1
    assert seed_result["seed_canonical_sha256"]


def test_parser_never_generates_study_calendar():
    assert _parse(_payload())["study_calendar_generated"] is False


def test_fetch_with_fake_opener_returns_parsed_result_and_closes_response():
    response = FakeResponse(_payload())
    calls = []
    result = fetch_chart_once("7203", "2020-01-01", "2020-01-10", lambda req: (calls.append(req) or response))
    assert result["ticker"] == "7203"
    assert len(calls) == 1
    assert response.closed is True


def test_fetch_non_200_fails_closed():
    with pytest.raises(V7YahooCollectorBlocked, match="HTTP_STATUS_429"):
        fetch_chart_once("7203", "2020-01-01", "2020-01-10", lambda req: FakeResponse(b"{}", status=429))


def test_default_global_urlopen_is_never_used_by_local_parser(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("REAL_NETWORK_CALL")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    result = _parse(_payload())
    assert result["valid_price_rows"]
    assert calls == []
