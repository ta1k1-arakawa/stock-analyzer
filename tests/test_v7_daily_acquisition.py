from __future__ import annotations

import csv
import json
import socket
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v7_daily_acquisition as daily
from src.v7_jpx_calendar import build_calendar_snapshot, parse_jpx_holiday_html

ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "V4_UNIVERSE.csv"
ENGINE_DAY = "2026-08-10"
NEXT_DAY = "2026-08-11"
WEEKEND_DAY = "2026-08-08"
HOLIDAY_DAY = "2026-01-01"
OUTSIDE_COVERAGE_DAY = "2025-12-31"
FIXED_START = datetime(2026, 8, 10, 7, 0, tzinfo=timezone.utc)
FIXED_END = datetime(2026, 8, 10, 7, 20, tzinfo=timezone.utc)


def _tickers() -> list[str]:
    with UNIVERSE.open(encoding="utf-8", newline="") as handle:
        return [row["ticker"] for row in csv.DictReader(handle)]


TICKERS = _tickers()


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def _calendar_html() -> bytes:
    return (
        "<html><body><nav><a href='/calendar/2026'>2026</a><a href='/calendar/2027'>2027</a></nav>"
        "<h2>Market Holidays</h2><table class='calendar-table'>"
        "<tr><th>2026</th></tr><tr><td> Jan. 1 (Thu.) </td><td> New Year Day </td></tr>"
        "<tr><th>2027</th></tr><tr><td> Jan. 1 (Fri.) </td><td> New Year Day </td></tr>"
        "</table></body></html>"
    ).encode("utf-8")


def calendar_snapshot():
    html = _calendar_html()
    holidays = parse_jpx_holiday_html(html)
    return build_calendar_snapshot(html, holidays, "2026-08-07T03:00:00Z")


def valid_payload(ticker: str, *, day: str = ENGINE_DAY, open_: float = 100.0) -> bytes:
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": [_epoch(day)],
        "indicators": {
            "quote": [{"open": [open_], "high": [open_ + 1], "low": [open_ - 1], "close": [open_], "volume": [1000]}],
            "adjclose": [{"adjclose": [open_]}],
        },
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


def valid_payload_with_split(ticker: str, *, split_day: str = ENGINE_DAY) -> bytes:
    body = json.loads(valid_payload(ticker).decode("utf-8"))
    body["chart"]["result"][0]["events"] = {"splits": {
        str(_epoch(split_day)): {"date": _epoch(split_day), "numerator": 2, "denominator": 1, "splitRatio": "2:1"},
    }}
    return json.dumps(body).encode("utf-8")


def invalid_price_payload(ticker: str, *, day: str = ENGINE_DAY) -> bytes:
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": [_epoch(day)],
        "indicators": {
            "quote": [{"open": [None], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1000]}],
            "adjclose": [{"adjclose": [100.0]}],
        },
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


def empty_timestamp_payload(ticker: str) -> bytes:
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": [],
        "indicators": {"quote": [{"open": [], "high": [], "low": [], "close": [], "volume": []}], "adjclose": [{"adjclose": []}]},
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


def null_timestamp_payload(ticker: str) -> bytes:
    result = {"meta": {"symbol": ticker + ".T"}, "timestamp": None, "indicators": {"quote": [{}], "adjclose": [{}]}}
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


def chart_error_payload() -> bytes:
    return json.dumps({"chart": {"error": {"code": "Not Found"}, "result": None}}).encode("utf-8")


def symbol_mismatch_payload() -> bytes:
    return valid_payload("WRONGSYM")


def unexpected_schema_payload(ticker: str) -> bytes:
    result = {"meta": {"symbol": ticker + ".T"}, "timestamp": [_epoch(ENGINE_DAY)]}
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


class FakeResponse:
    def __init__(self, payload: bytes, *, status: int = 200, url: str | None = None):
        self.payload = payload
        self.status = status
        self.url = url
        self.closed = False

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    def __init__(self, payloads: dict[str, bytes] | None = None, *, overrides: dict[str, object] | None = None):
        self.payloads = payloads if payloads is not None else {ticker: valid_payload(ticker) for ticker in TICKERS}
        self.overrides = overrides or {}
        self.calls: list[str] = []

    def __call__(self, request_obj):
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        if ticker in self.overrides:
            override = self.overrides[ticker]
            if isinstance(override, BaseException):
                raise override
            return override
        return FakeResponse(
            self.payloads[ticker], url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T"
        )


def _run(tmp_path, *, opener=None, engine_day=ENGINE_DAY, calendar=None, clock_values=None, universe_csv=UNIVERSE):
    opener = opener or FakeOpener()
    calendar = calendar if calendar is not None else calendar_snapshot()
    clock_values = clock_values if clock_values is not None else [FIXED_START, FIXED_END]
    clock_iter = iter(clock_values)
    mono_state = {"now": 0.0}
    sleep_calls: list[float] = []

    def monotonic_clock() -> float:
        return mono_state["now"]

    def sleep_fn(seconds: float) -> None:
        sleep_calls.append(seconds)
        mono_state["now"] += seconds

    manifest = daily.acquire_daily_bundle(
        output_root=tmp_path,
        universe_csv=universe_csv,
        calendar_snapshot=calendar,
        engine_day=engine_day,
        opener=opener,
        clock=lambda: next(clock_iter),
        monotonic_clock=monotonic_clock,
        sleep_fn=sleep_fn,
    )
    return manifest, opener, sleep_calls


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


# ---------------------------------------------------------------------------
# Universe validation
# ---------------------------------------------------------------------------


def test_universe_validation_matches_expected_hashes():
    from src.v7_seed_acquisition import EXPECTED_TICKER_LIST_SHA256, EXPECTED_UNIVERSE_CSV_SHA256, validate_universe_file

    value = validate_universe_file(UNIVERSE)
    assert value["universe_csv_sha256"] == EXPECTED_UNIVERSE_CSV_SHA256
    assert value["ticker_list_sha256"] == EXPECTED_TICKER_LIST_SHA256
    assert value["ticker_count"] == 300


def test_universe_sha_mismatch_blocks_before_network(tmp_path):
    altered = tmp_path / "V4_UNIVERSE.csv"
    altered.write_bytes(UNIVERSE.read_bytes() + b"\n")
    opener = FakeOpener()
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="UNIVERSE_VALIDATION_FAILED"):
        _run(tmp_path, opener=opener, universe_csv=altered)
    assert opener.calls == []


def test_duplicate_ticker_in_universe_blocks_before_network(tmp_path):
    rows = list(csv.DictReader(UNIVERSE.open(encoding="utf-8", newline="")))
    duplicated = rows + [rows[0]]
    altered = tmp_path / "dup.csv"
    with altered.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(duplicated)
    opener = FakeOpener()
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="UNIVERSE_VALIDATION_FAILED"):
        _run(tmp_path, opener=opener, universe_csv=altered)
    assert opener.calls == []


# ---------------------------------------------------------------------------
# Calendar validation, network=0
# ---------------------------------------------------------------------------


def test_weekend_engine_day_blocks_before_network(tmp_path):
    opener = FakeOpener()
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="ENGINE_DAY_NOT_JPX_TRADING_DAY"):
        _run(tmp_path, opener=opener, engine_day=WEEKEND_DAY)
    assert opener.calls == []


def test_holiday_engine_day_blocks_before_network(tmp_path):
    opener = FakeOpener()
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="ENGINE_DAY_NOT_JPX_TRADING_DAY"):
        _run(tmp_path, opener=opener, engine_day=HOLIDAY_DAY)
    assert opener.calls == []


def test_outside_coverage_engine_day_blocks_before_network(tmp_path):
    opener = FakeOpener()
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="ENGINE_DAY_OUTSIDE_CALENDAR_COVERAGE"):
        _run(tmp_path, opener=opener, engine_day=OUTSIDE_COVERAGE_DAY)
    assert opener.calls == []


# ---------------------------------------------------------------------------
# Request window
# ---------------------------------------------------------------------------


def test_request_window_is_engine_day_to_next_calendar_day(tmp_path):
    manifest, opener, _ = _run(tmp_path)
    assert manifest["request_start"] == ENGINE_DAY
    assert manifest["request_end_exclusive"] == NEXT_DAY


def test_request_window_does_not_extend_past_next_calendar_day(tmp_path):
    manifest, _, _ = _run(tmp_path)
    assert manifest["request_end_exclusive"] != "2026-08-12"


# ---------------------------------------------------------------------------
# Universe order / request count / retry
# ---------------------------------------------------------------------------


def test_request_order_matches_universe_canonical_order(tmp_path):
    _, opener, _ = _run(tmp_path)
    assert opener.calls == TICKERS


def test_request_count_equals_300(tmp_path):
    manifest, opener, _ = _run(tmp_path)
    assert manifest["request_count"] == 300
    assert len(opener.calls) == 300


def test_retry_count_always_zero(tmp_path):
    manifest, _, _ = _run(tmp_path)
    assert manifest["retry_count"] == 0


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def test_rate_limiter_enforces_minimum_interval():
    state = {"now": 0.0}
    sleeps: list[float] = []

    def monotonic_clock() -> float:
        return state["now"]

    def sleep_fn(seconds: float) -> None:
        sleeps.append(seconds)
        state["now"] += seconds

    start0 = daily._wait_for_next_request_start(0, None, monotonic_clock, sleep_fn)
    assert start0 == 0.0
    assert sleeps == []
    state["now"] = 0.5
    start1 = daily._wait_for_next_request_start(1, start0, monotonic_clock, sleep_fn)
    assert start1 - start0 >= daily.MIN_REQUEST_INTERVAL_SECONDS
    assert sleeps == [1.5]


def test_rate_limiter_skips_sleep_when_already_elapsed():
    state = {"now": 5.0}
    sleeps: list[float] = []

    def monotonic_clock() -> float:
        return state["now"]

    def sleep_fn(seconds: float) -> None:
        sleeps.append(seconds)

    start = daily._wait_for_next_request_start(1, 0.0, monotonic_clock, sleep_fn)
    assert start == 5.0
    assert sleeps == []


def test_full_run_sleeps_299_times_not_300(tmp_path):
    _, _, sleep_calls = _run(tmp_path)
    assert len(sleep_calls) == 299
    assert all(value == pytest.approx(daily.MIN_REQUEST_INTERVAL_SECONDS) for value in sleep_calls)


# ---------------------------------------------------------------------------
# D0 observation classification
# ---------------------------------------------------------------------------


def test_valid_d0_status_and_price_snapshot_fields(tmp_path):
    manifest, _, _ = _run(tmp_path)
    assert manifest["valid_d0_count"] == 300
    assert manifest["missing_d0_count"] == 0
    record = next(r for r in manifest["payload_manifest"] if r["ticker"] == TICKERS[0])
    assert record["status"] == "VALID_D0"
    assert record["missing_reason"] is None
    assert record["canonical_d0_row_sha256"] is not None


def test_invalid_price_row_becomes_audited_missing(tmp_path):
    ticker = TICKERS[10]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = invalid_price_payload(ticker)
    manifest, _, _ = _run(tmp_path, opener=FakeOpener(payloads))
    assert manifest["missing_d0_count"] == 1
    record = next(r for r in manifest["payload_manifest"] if r["ticker"] == ticker)
    assert record["status"] == "AUDITED_MISSING"
    assert record["missing_reason"] == "NONFINITE_OPEN"
    assert record["canonical_d0_row_sha256"] is None


def test_empty_timestamp_list_becomes_d0_data_unavailable(tmp_path):
    ticker = TICKERS[20]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = empty_timestamp_payload(ticker)
    manifest, _, _ = _run(tmp_path, opener=FakeOpener(payloads))
    record = next(r for r in manifest["payload_manifest"] if r["ticker"] == ticker)
    assert record["status"] == "AUDITED_MISSING"
    assert record["missing_reason"] == "D0_DATA_UNAVAILABLE"


def test_null_timestamp_becomes_d0_data_unavailable(tmp_path):
    ticker = TICKERS[21]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = null_timestamp_payload(ticker)
    manifest, _, _ = _run(tmp_path, opener=FakeOpener(payloads))
    record = next(r for r in manifest["payload_manifest"] if r["ticker"] == ticker)
    assert record["status"] == "AUDITED_MISSING"
    assert record["missing_reason"] == "D0_DATA_UNAVAILABLE"


def test_classify_missing_timestamp_payload_rejects_wrong_symbol():
    payload = null_timestamp_payload("AAAA")
    assert daily.classify_missing_timestamp_payload(payload, "BBBB") is False


def test_classify_missing_timestamp_payload_rejects_chart_error():
    assert daily.classify_missing_timestamp_payload(chart_error_payload(), "AAAA") is False


def test_classify_missing_timestamp_payload_rejects_non_empty_timestamp():
    payload = valid_payload("AAAA")
    assert daily.classify_missing_timestamp_payload(payload, "AAAA") is False


# ---------------------------------------------------------------------------
# Transport failure classification -> hard BLOCK, no publish
# ---------------------------------------------------------------------------


def _final_dir(tmp_path, engine_day=ENGINE_DAY) -> Path:
    return Path(tmp_path) / daily.ACQUISITIONS_DIRNAME / engine_day


def _assert_no_publish_and_no_remnant(tmp_path, engine_day=ENGINE_DAY):
    assert not _final_dir(tmp_path, engine_day).exists()
    acquisitions_root = Path(tmp_path) / daily.ACQUISITIONS_DIRNAME
    if acquisitions_root.exists():
        assert not any(".staging-" in entry.name for entry in acquisitions_root.iterdir())


def test_http_429_blocks_and_no_publish(tmp_path):
    ticker = TICKERS[5]
    opener = FakeOpener(overrides={ticker: FakeResponse(b"", status=429, url="https://query1.finance.yahoo.com/x")})
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="HTTP_STATUS_429"):
        _run(tmp_path, opener=opener)
    assert opener.calls[-1] == ticker
    _assert_no_publish_and_no_remnant(tmp_path)


def test_http_500_blocks_and_no_publish(tmp_path):
    ticker = TICKERS[6]
    opener = FakeOpener(overrides={ticker: FakeResponse(b"", status=500, url="https://query1.finance.yahoo.com/x")})
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="HTTP_STATUS_500"):
        _run(tmp_path, opener=opener)
    _assert_no_publish_and_no_remnant(tmp_path)


def test_timeout_blocks_and_no_publish(tmp_path):
    ticker = TICKERS[7]
    opener = FakeOpener(overrides={ticker: TimeoutError("timed out")})
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="TICKER_" + ticker):
        _run(tmp_path, opener=opener)
    _assert_no_publish_and_no_remnant(tmp_path)


def test_dns_failure_blocks_and_no_publish(tmp_path):
    ticker = TICKERS[8]
    opener = FakeOpener(overrides={ticker: urllib.error.URLError(socket.gaierror("Name or service not known"))})
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="TICKER_" + ticker):
        _run(tmp_path, opener=opener)
    _assert_no_publish_and_no_remnant(tmp_path)


def test_response_host_mismatch_blocks_and_no_publish(tmp_path):
    ticker = TICKERS[9]
    opener = FakeOpener(overrides={ticker: FakeResponse(valid_payload(ticker), url="https://evil.example.com/x")})
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="RESPONSE_HOST_MISMATCH"):
        _run(tmp_path, opener=opener)
    _assert_no_publish_and_no_remnant(tmp_path)


def test_chart_error_blocks_and_no_publish(tmp_path):
    ticker = TICKERS[11]
    opener = FakeOpener(overrides={
        ticker: FakeResponse(chart_error_payload(), url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")
    })
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="CHART_ERROR"):
        _run(tmp_path, opener=opener)
    _assert_no_publish_and_no_remnant(tmp_path)


def test_symbol_mismatch_blocks_and_no_publish(tmp_path):
    ticker = TICKERS[12]
    opener = FakeOpener(overrides={
        ticker: FakeResponse(symbol_mismatch_payload(), url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")
    })
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="SYMBOL_MISMATCH"):
        _run(tmp_path, opener=opener)
    _assert_no_publish_and_no_remnant(tmp_path)


def test_unexpected_schema_blocks_and_no_publish(tmp_path):
    ticker = TICKERS[13]
    opener = FakeOpener(overrides={
        ticker: FakeResponse(unexpected_schema_payload(ticker), url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")
    })
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="TICKER_STATUS_OVERLAP|TIMESTAMP_MISSING|INDICATORS_MISSING"):
        _run(tmp_path, opener=opener)
    _assert_no_publish_and_no_remnant(tmp_path)


# ---------------------------------------------------------------------------
# Raw provenance
# ---------------------------------------------------------------------------


def test_raw_bytes_are_exact_response_body(tmp_path):
    manifest, _, _ = _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    ticker = TICKERS[0]
    raw_bytes = (day_dir / "raw" / (ticker + ".json")).read_bytes()
    assert raw_bytes == valid_payload(ticker)


def test_raw_sha256_matches_manifest_record(tmp_path):
    manifest, _, _ = _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    for record in manifest["payload_manifest"][:5]:
        raw_bytes = (day_dir / "raw" / (record["ticker"] + ".json")).read_bytes()
        assert daily.sha256_bytes(raw_bytes) == record["payload_sha256"]


def test_raw_byte_count_matches_manifest_record(tmp_path):
    manifest, _, _ = _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    for record in manifest["payload_manifest"][:5]:
        raw_bytes = (day_dir / "raw" / (record["ticker"] + ".json")).read_bytes()
        assert len(raw_bytes) == record["byte_count"]


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------


def test_valid_plus_missing_equals_300(tmp_path):
    ticker = TICKERS[30]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = invalid_price_payload(ticker)
    manifest, _, _ = _run(tmp_path, opener=FakeOpener(payloads))
    assert manifest["valid_d0_count"] + manifest["missing_d0_count"] == 300


def test_valid_and_missing_mutually_exclusive_ticker_sets(tmp_path):
    day_dir_manifest, _, _ = _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    price = json.loads((day_dir / "price_snapshot.json").read_text(encoding="utf-8"))
    missing = json.loads((day_dir / "missing_snapshot.json").read_text(encoding="utf-8"))
    price_tickers = {row["ticker"] for row in price}
    missing_tickers = {row["ticker"] for row in missing}
    assert price_tickers.isdisjoint(missing_tickers)
    assert len(price_tickers) + len(missing_tickers) == 300


# ---------------------------------------------------------------------------
# Payload manifest ordering
# ---------------------------------------------------------------------------


def test_payload_manifest_preserves_universe_order(tmp_path):
    manifest, _, _ = _run(tmp_path)
    assert [record["ticker"] for record in manifest["payload_manifest"]] == TICKERS


# ---------------------------------------------------------------------------
# Deterministic canonical serialization
# ---------------------------------------------------------------------------


def test_price_snapshot_is_ticker_sorted(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    price = json.loads((day_dir / "price_snapshot.json").read_text(encoding="utf-8"))
    assert [row["ticker"] for row in price] == sorted(row["ticker"] for row in price)


def test_missing_snapshot_is_ticker_sorted(tmp_path):
    ticker = TICKERS[40]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = invalid_price_payload(ticker)
    _run(tmp_path, opener=FakeOpener(payloads))
    day_dir = _final_dir(tmp_path)
    missing = json.loads((day_dir / "missing_snapshot.json").read_text(encoding="utf-8"))
    assert [row["ticker"] for row in missing] == sorted(row["ticker"] for row in missing)


def test_split_snapshot_is_date_then_ticker_sorted(tmp_path):
    tickers = [TICKERS[0], TICKERS[1]]
    payloads = {t: valid_payload(t) for t in TICKERS}
    for ticker in tickers:
        payloads[ticker] = valid_payload_with_split(ticker)
    _run(tmp_path, opener=FakeOpener(payloads))
    day_dir = _final_dir(tmp_path)
    splits = json.loads((day_dir / "split_snapshot.json").read_text(encoding="utf-8"))
    keys = [(row["effective_date"], row["ticker"]) for row in splits]
    assert keys == sorted(keys)
    assert len(splits) == 2


def test_payload_manifest_is_byte_deterministic_across_identical_runs(tmp_path_factory):
    dir1 = tmp_path_factory.mktemp("run1")
    dir2 = tmp_path_factory.mktemp("run2")
    manifest1, _, _ = _run(dir1)
    manifest2, _, _ = _run(dir2)
    assert daily.canonical_json_bytes(manifest1) == daily.canonical_json_bytes(manifest2)


# ---------------------------------------------------------------------------
# Split handling
# ---------------------------------------------------------------------------


def test_engine_day_split_accepted(tmp_path):
    ticker = TICKERS[0]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = valid_payload_with_split(ticker)
    manifest, _, _ = _run(tmp_path, opener=FakeOpener(payloads))
    assert manifest["split_event_count"] == 1
    day_dir = _final_dir(tmp_path)
    splits = json.loads((day_dir / "split_snapshot.json").read_text(encoding="utf-8"))
    assert splits[0]["ticker"] == ticker
    assert splits[0]["effective_date"] == ENGINE_DAY


def test_future_split_blocked():
    events = [{"effective_date": NEXT_DAY, "numerator": 2.0, "denominator": 1.0, "split_ratio": 2.0}]
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="FUTURE_SPLIT_ACCESS"):
        daily.classify_engine_day_splits(events, "AAAA", ENGINE_DAY)


def test_prior_split_blocked():
    events = [{"effective_date": "2026-08-09", "numerator": 2.0, "denominator": 1.0, "split_ratio": 2.0}]
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="SPLIT_EFFECTIVE_DATE_BEFORE_ENGINE_DAY"):
        daily.classify_engine_day_splits(events, "AAAA", ENGINE_DAY)


def test_duplicate_split_blocked():
    events = [
        {"effective_date": ENGINE_DAY, "numerator": 2.0, "denominator": 1.0, "split_ratio": 2.0},
        {"effective_date": ENGINE_DAY, "numerator": 3.0, "denominator": 1.0, "split_ratio": 3.0},
    ]
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="DUPLICATE_SPLIT_EVENT"):
        daily.classify_engine_day_splits(events, "AAAA", ENGINE_DAY)


# ---------------------------------------------------------------------------
# Existing final day / staging remnant, network=0
# ---------------------------------------------------------------------------


def test_duplicate_acquisition_day_blocks_before_network(tmp_path):
    _run(tmp_path)
    opener = FakeOpener()
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="DUPLICATE_ACQUISITION_DAY"):
        _run(tmp_path, opener=opener)
    assert opener.calls == []


def test_staging_remnant_blocks_before_network(tmp_path):
    acquisitions_root = Path(tmp_path) / daily.ACQUISITIONS_DIRNAME
    acquisitions_root.mkdir(parents=True)
    (acquisitions_root / f"{ENGINE_DAY}.staging-remnant").mkdir()
    opener = FakeOpener()
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="PARTIAL_ACQUISITION_COMMIT"):
        _run(tmp_path, opener=opener)
    assert opener.calls == []


# ---------------------------------------------------------------------------
# Failure position and cleanup
# ---------------------------------------------------------------------------


def test_failure_at_first_ticker_no_final_publish(tmp_path):
    ticker = TICKERS[0]
    opener = FakeOpener(overrides={ticker: FakeResponse(b"", status=500, url="https://query1.finance.yahoo.com/x")})
    with pytest.raises(daily.V7DailyAcquisitionBlocked):
        _run(tmp_path, opener=opener)
    assert len(opener.calls) == 1
    _assert_no_publish_and_no_remnant(tmp_path)


def test_failure_at_middle_ticker_no_final_publish(tmp_path):
    ticker = TICKERS[149]
    opener = FakeOpener(overrides={ticker: FakeResponse(b"", status=500, url="https://query1.finance.yahoo.com/x")})
    with pytest.raises(daily.V7DailyAcquisitionBlocked):
        _run(tmp_path, opener=opener)
    assert len(opener.calls) == 150
    _assert_no_publish_and_no_remnant(tmp_path)


def test_failure_at_last_ticker_no_final_publish(tmp_path):
    ticker = TICKERS[299]
    opener = FakeOpener(overrides={ticker: FakeResponse(b"", status=500, url="https://query1.finance.yahoo.com/x")})
    with pytest.raises(daily.V7DailyAcquisitionBlocked):
        _run(tmp_path, opener=opener)
    assert len(opener.calls) == 300
    _assert_no_publish_and_no_remnant(tmp_path)


def test_previous_complete_day_unchanged_after_later_failure(tmp_path):
    manifest1, _, _ = _run(tmp_path, engine_day=ENGINE_DAY)
    day1_manifest_path = _final_dir(tmp_path, ENGINE_DAY) / daily.MANIFEST_FILENAME
    before = day1_manifest_path.read_bytes()

    ticker = TICKERS[15]
    opener = FakeOpener(overrides={ticker: FakeResponse(b"", status=500, url="https://query1.finance.yahoo.com/x")})
    with pytest.raises(daily.V7DailyAcquisitionBlocked):
        _run(
            tmp_path,
            opener=opener,
            engine_day=NEXT_DAY,
            clock_values=[datetime(2026, 8, 11, 7, 0, tzinfo=timezone.utc), datetime(2026, 8, 11, 7, 20, tzinfo=timezone.utc)],
        )
    after = day1_manifest_path.read_bytes()
    assert before == after
    assert not _final_dir(tmp_path, NEXT_DAY).exists()


# ---------------------------------------------------------------------------
# verify_daily_acquisition_bundle
# ---------------------------------------------------------------------------


def _verify(tmp_path, *, engine_day=ENGINE_DAY, calendar_commit=None, collector_commit=None, universe_csv=UNIVERSE):
    return daily.verify_daily_acquisition_bundle(
        tmp_path,
        engine_day,
        calendar_commit if calendar_commit is not None else daily.CALENDAR_COMMIT,
        collector_commit if collector_commit is not None else daily.COLLECTOR_COMMIT,
        universe_csv,
    )


def test_verify_passes_on_clean_bundle(tmp_path):
    _run(tmp_path)
    result = _verify(tmp_path)
    assert result["status"] == "PASS"
    assert result["valid_d0_count"] == 300


def test_verify_detects_price_snapshot_tamper(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    (day_dir / "price_snapshot.json").write_text("[]", encoding="utf-8")
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="PRICE_SNAPSHOT_HASH_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_missing_snapshot_tamper(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    (day_dir / "missing_snapshot.json").write_text(
        json.dumps([{"ticker": "ZZZZ", "engine_day": ENGINE_DAY, "reason": "X", "payload_sha256": "a" * 64, "byte_count": 1}]),
        encoding="utf-8",
    )
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="MISSING_SNAPSHOT_HASH_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_split_snapshot_tamper(tmp_path):
    ticker = TICKERS[0]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = valid_payload_with_split(ticker)
    _run(tmp_path, opener=FakeOpener(payloads))
    day_dir = _final_dir(tmp_path)
    (day_dir / "split_snapshot.json").write_text("[]", encoding="utf-8")
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="SPLIT_SNAPSHOT_HASH_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_raw_hash_tamper(tmp_path):
    manifest, _, _ = _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    ticker = manifest["payload_manifest"][0]["ticker"]
    (day_dir / "raw" / (ticker + ".json")).write_bytes(valid_payload(ticker, open_=999.0))
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="RAW_SHA_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_raw_byte_tamper(tmp_path):
    manifest, _, _ = _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    ticker = manifest["payload_manifest"][0]["ticker"]
    raw_path = day_dir / "raw" / (ticker + ".json")
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="RAW_SHA_MISMATCH|RAW_BYTE_COUNT_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_payload_manifest_tamper(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest_path = day_dir / daily.MANIFEST_FILENAME
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    record["payload_manifest"][0]["split_event_count"] = record["payload_manifest"][0]["split_event_count"] + 1
    manifest_path.write_text(daily.canonical_json_bytes(record).decode("utf-8"), encoding="utf-8")
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="PAYLOAD_MANIFEST_HASH_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_calendar_commit_mismatch(tmp_path):
    _run(tmp_path)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="CALENDAR_COMMIT_MISMATCH"):
        _verify(tmp_path, calendar_commit="f" * 40)


def test_verify_detects_collector_commit_mismatch(tmp_path):
    _run(tmp_path)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="COLLECTOR_COMMIT_MISMATCH"):
        _verify(tmp_path, collector_commit="f" * 40)


def test_verify_detects_staging_remnant(tmp_path):
    _run(tmp_path)
    acquisitions_root = Path(tmp_path) / daily.ACQUISITIONS_DIRNAME
    (acquisitions_root / f"{NEXT_DAY}.staging-remnant").mkdir()
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="PARTIAL_ACQUISITION_COMMIT"):
        _verify(tmp_path)


# ---------------------------------------------------------------------------
# Clock validation
# ---------------------------------------------------------------------------


def test_clock_must_be_aware_utc(tmp_path):
    naive = datetime(2026, 8, 10, 7, 0)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="UTC_TIMESTAMP_INVALID"):
        _run(tmp_path, clock_values=[naive, naive])


def test_completed_before_started_blocks(tmp_path):
    later = datetime(2026, 8, 10, 8, 0, tzinfo=timezone.utc)
    earlier = datetime(2026, 8, 10, 7, 0, tzinfo=timezone.utc)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="ACQUISITION_CLOCK_NONMONOTONIC"):
        _run(tmp_path, clock_values=[later, earlier])


# ---------------------------------------------------------------------------
# Zero-impact markers
# ---------------------------------------------------------------------------


def test_manifest_records_zero_downstream_processing(tmp_path):
    manifest, _, _ = _run(tmp_path)
    assert manifest["candidate_generation_started"] == 0
    assert manifest["portfolio_processing_started"] == 0
    assert manifest["profit_calculation_started"] == 0
    assert manifest["formal_evaluation_started"] == 0
    assert manifest["activation_created"] is False


def test_manifest_identity_constants(tmp_path):
    manifest, _, _ = _run(tmp_path)
    assert manifest["schema_version"] == "V7_DAILY_ACQUISITION_V1"
    assert manifest["mode"] == "FORWARD_DAILY_ACQUISITION"
    assert manifest["calendar_commit"] == "03ce048b0eedca632f79ad925a627cb9e967d78d"
    assert manifest["calendar_definition_version"] == "V7_JPX_OFFICIAL_MARKET_HOLIDAYS_V1"
    assert manifest["collector_commit"] == "4ca41c53895e75910ae65809fea6018868929afa"
    assert manifest["data_source"] == "Yahoo Chart"
    assert manifest["data_source_host"] == "query1.finance.yahoo.com"


def test_real_urlopen_not_invoked_directly(tmp_path):
    with pytest.raises(AssertionError, match="real urlopen executed"):
        urllib.request.urlopen("https://example.com")


# ---------------------------------------------------------------------------
# FIX A: verifier universe binding (order, sha, ticker-list) and manifest invariants
# ---------------------------------------------------------------------------


def _write_manifest(day_dir: Path, manifest: dict) -> None:
    (day_dir / daily.MANIFEST_FILENAME).write_bytes(daily.canonical_json_bytes(manifest))


def _load_manifest(day_dir: Path) -> dict:
    return json.loads((day_dir / daily.MANIFEST_FILENAME).read_text(encoding="utf-8"))


def test_verify_detects_payload_manifest_reordered_same_ticker_set(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["payload_manifest"][0], manifest["payload_manifest"][1] = (
        manifest["payload_manifest"][1],
        manifest["payload_manifest"][0],
    )
    manifest["payload_manifest_sha256"] = daily.sha256_bytes(daily.canonical_json_bytes(manifest["payload_manifest"]))
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="PAYLOAD_MANIFEST_TICKER_ORDER_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_wrong_universe_csv_sha(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["universe_csv_sha256"] = "f" * 64
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="UNIVERSE_CSV_SHA_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_wrong_ticker_list_sha(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["ticker_list_sha256"] = "f" * 64
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="TICKER_LIST_SHA_MISMATCH"):
        _verify(tmp_path)


# ---------------------------------------------------------------------------
# FIX B: strict empty-timestamp classifier tests
# ---------------------------------------------------------------------------


def _payload_with_open(ticker: str, *, timestamp) -> bytes:
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": timestamp,
        "indicators": {
            "quote": [{"open": [100.0], "high": [], "low": [], "close": [], "volume": []}],
            "adjclose": [{"adjclose": []}],
        },
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


def _payload_with_adjclose(ticker: str, *, timestamp) -> bytes:
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": timestamp,
        "indicators": {
            "quote": [{"open": [], "high": [], "low": [], "close": [], "volume": []}],
            "adjclose": [{"adjclose": [100.0]}],
        },
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


def test_timestamp_empty_with_nonempty_open_is_hard_blocked():
    payload = _payload_with_open("AAAA", timestamp=[])
    assert daily.classify_missing_timestamp_payload(payload, "AAAA") is False


def test_timestamp_null_with_nonempty_adjclose_is_hard_blocked():
    payload = _payload_with_adjclose("AAAA", timestamp=None)
    assert daily.classify_missing_timestamp_payload(payload, "AAAA") is False


def test_timestamp_empty_with_all_indicator_arrays_empty_is_audited_missing():
    payload = empty_timestamp_payload("AAAA")
    assert daily.classify_missing_timestamp_payload(payload, "AAAA") is True


def test_timestamp_empty_with_nonempty_open_blocks_full_acquisition(tmp_path):
    ticker = TICKERS[50]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = _payload_with_open(ticker, timestamp=[])
    opener = FakeOpener(payloads)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="TICKER_" + ticker):
        _run(tmp_path, opener=opener)
    _assert_no_publish_and_no_remnant(tmp_path)


def test_timestamp_null_with_nonempty_adjclose_blocks_full_acquisition(tmp_path):
    ticker = TICKERS[51]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = _payload_with_adjclose(ticker, timestamp=None)
    opener = FakeOpener(payloads)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="TICKER_" + ticker):
        _run(tmp_path, opener=opener)
    _assert_no_publish_and_no_remnant(tmp_path)


# ---------------------------------------------------------------------------
# FIX C: payload -> snapshot provenance cross-check (top-level snapshot hash
# is recomputed to match the tamper, so only the provenance chain catches it)
# ---------------------------------------------------------------------------


def test_verify_detects_canonical_d0_row_mutation_with_recomputed_snapshot_hash(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    price_snapshot = json.loads((day_dir / daily.PRICE_SNAPSHOT_FILENAME).read_text(encoding="utf-8"))
    price_snapshot[0]["raw_open"] = price_snapshot[0]["raw_open"] + 1.0
    price_bytes = daily.canonical_json_bytes(price_snapshot)
    (day_dir / daily.PRICE_SNAPSHOT_FILENAME).write_bytes(price_bytes)
    manifest = _load_manifest(day_dir)
    manifest["price_snapshot_sha256"] = daily.sha256_bytes(price_bytes)
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="CANONICAL_D0_ROW_HASH_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_price_payload_sha_mutation_with_recomputed_snapshot_hash(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    price_snapshot = json.loads((day_dir / daily.PRICE_SNAPSHOT_FILENAME).read_text(encoding="utf-8"))
    price_snapshot[0]["payload_sha256"] = "e" * 64
    price_bytes = daily.canonical_json_bytes(price_snapshot)
    (day_dir / daily.PRICE_SNAPSHOT_FILENAME).write_bytes(price_bytes)
    manifest = _load_manifest(day_dir)
    manifest["price_snapshot_sha256"] = daily.sha256_bytes(price_bytes)
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="PRICE_SNAPSHOT_PAYLOAD_SHA_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_split_row_mutation_with_recomputed_snapshot_hash(tmp_path):
    ticker = TICKERS[0]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = valid_payload_with_split(ticker)
    _run(tmp_path, opener=FakeOpener(payloads))
    day_dir = _final_dir(tmp_path)
    split_snapshot = json.loads((day_dir / daily.SPLIT_SNAPSHOT_FILENAME).read_text(encoding="utf-8"))
    split_snapshot[0]["numerator"] = split_snapshot[0]["numerator"] + 1
    split_bytes = daily.canonical_json_bytes(split_snapshot)
    (day_dir / daily.SPLIT_SNAPSHOT_FILENAME).write_bytes(split_bytes)
    manifest = _load_manifest(day_dir)
    manifest["split_snapshot_sha256"] = daily.sha256_bytes(split_bytes)
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="SPLIT_PROVENANCE_HASH_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_manifest_split_hash_mismatch_without_touching_split_file(tmp_path):
    ticker = TICKERS[0]
    payloads = {t: valid_payload(t) for t in TICKERS}
    payloads[ticker] = valid_payload_with_split(ticker)
    _run(tmp_path, opener=FakeOpener(payloads))
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    for record in manifest["payload_manifest"]:
        if record["ticker"] == ticker:
            record["canonical_engine_day_split_sha256"] = daily.canonical_sha256([])
            break
    manifest["payload_manifest_sha256"] = daily.sha256_bytes(daily.canonical_json_bytes(manifest["payload_manifest"]))
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="SPLIT_PROVENANCE_HASH_MISMATCH"):
        _verify(tmp_path)


# ---------------------------------------------------------------------------
# Additional manifest invariants
# ---------------------------------------------------------------------------


def test_verify_detects_request_window_tamper(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["request_end_exclusive"] = "2026-08-12"
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="REQUEST_END_EXCLUSIVE_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_request_start_tamper(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["request_start"] = "2026-08-09"
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="REQUEST_START_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_success_transport_count_tamper(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["success_transport_count"] = 299
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="SUCCESS_TRANSPORT_COUNT_INVALID"):
        _verify(tmp_path)


def test_verify_detects_nonzero_http_429_count(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["http_429_count"] = 1
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="HTTP_429_COUNT_INVALID"):
        _verify(tmp_path)


@pytest.mark.parametrize("field", [
    "candidate_generation_started",
    "portfolio_processing_started",
    "profit_calculation_started",
    "formal_evaluation_started",
])
def test_verify_detects_nonzero_downstream_activity_flag(tmp_path, field):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest[field] = 1
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="DOWNSTREAM_PROCESSING_FLAG_INVALID"):
        _verify(tmp_path)


def test_verify_detects_activation_created_flag_true(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["activation_created"] = True
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="ACTIVATION_CREATED_FLAG_INVALID"):
        _verify(tmp_path)


def test_verify_detects_calendar_definition_version_tamper(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["calendar_definition_version"] = "WRONG_VERSION"
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="CALENDAR_DEFINITION_VERSION_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_data_source_tamper(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["data_source"] = "Other Source"
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="DATA_SOURCE_MISMATCH"):
        _verify(tmp_path)


def test_verify_detects_data_source_host_tamper(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["data_source_host"] = "evil.example.com"
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="DATA_SOURCE_HOST_MISMATCH"):
        _verify(tmp_path)


# ---------------------------------------------------------------------------
# Payload manifest record schema strictness
# ---------------------------------------------------------------------------


def test_verify_detects_payload_record_unknown_field(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["payload_manifest"][0]["extra_field"] = "x"
    manifest["payload_manifest_sha256"] = daily.sha256_bytes(daily.canonical_json_bytes(manifest["payload_manifest"]))
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="PAYLOAD_MANIFEST_RECORD_SCHEMA_INVALID"):
        _verify(tmp_path)


def test_verify_detects_payload_record_missing_field(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    del manifest["payload_manifest"][0]["split_event_count"]
    manifest["payload_manifest_sha256"] = daily.sha256_bytes(daily.canonical_json_bytes(manifest["payload_manifest"]))
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="PAYLOAD_MANIFEST_RECORD_SCHEMA_INVALID"):
        _verify(tmp_path)


def test_verify_detects_payload_record_invalid_sha_format(tmp_path):
    _run(tmp_path)
    day_dir = _final_dir(tmp_path)
    manifest = _load_manifest(day_dir)
    manifest["payload_manifest"][0]["payload_sha256"] = "not-a-sha"
    manifest["payload_manifest_sha256"] = daily.sha256_bytes(daily.canonical_json_bytes(manifest["payload_manifest"]))
    _write_manifest(day_dir, manifest)
    with pytest.raises(daily.V7DailyAcquisitionBlocked, match="PAYLOAD_MANIFEST_SHA_INVALID"):
        _verify(tmp_path)
