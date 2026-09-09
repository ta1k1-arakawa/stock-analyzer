from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import v7_seed_acquisition as acquisition
from src.v7_forward_protocol import validate_seed_rows


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE = ROOT / "V4_UNIVERSE.csv"
START = "2025-07-01"
END = "2026-08-08"
CUTOFF = "2026-08-07"
FIXED_NOW = datetime(2026, 8, 7, 3, 0, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(self, payload: bytes, *, status: int = 200, url: str | None = None):
        self.payload = payload
        self.status = status
        self.url = url or "https://query1.finance.yahoo.com/v8/finance/chart/X.T"
        self.closed = False

    def read(self):
        return self.payload

    def close(self):
        self.closed = True


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def payload_for(ticker: str, count: int = 1, *, start: str = START, adj_delta: float = 0.0, split=None) -> bytes:
    start_date = datetime.fromisoformat(start).date()
    dates = [(start_date + timedelta(days=index)).isoformat() for index in range(count)]
    values = [100.0 + index for index in range(count)]
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": [_epoch(day) for day in dates],
        "indicators": {
            "quote": [{
                "open": values,
                "high": [value + 1 for value in values],
                "low": [value - 1 for value in values],
                "close": values,
                "volume": [1000000 for _ in values],
            }],
            "adjclose": [{"adjclose": [value + adj_delta for value in values]}],
        },
    }
    if split is not None:
        split_date, numerator, denominator = split
        result["events"] = {"splits": {
            str(_epoch(split_date)): {
                "date": _epoch(split_date),
                "numerator": numerator,
                "denominator": denominator,
                "splitRatio": f"{numerator}:{denominator}",
            }
        }}
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


def _tickers() -> list[str]:
    with UNIVERSE.open(encoding="utf-8", newline="") as handle:
        return [row["ticker"] for row in csv.DictReader(handle)]


class FakeOpener:
    def __init__(self, payloads: dict[str, bytes], *, failures: dict[str, FakeResponse] | None = None):
        self.payloads = payloads
        self.failures = failures or {}
        self.calls: list[str] = []

    def __call__(self, request):
        ticker = request.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        if ticker in self.failures:
            response = self.failures[ticker]
            response.url = response.url.replace("X.T", ticker + ".T")
            return response
        return FakeResponse(self.payloads[ticker], url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")


def _payloads(*, first_count: int = 1, first_adj_delta: float = 0.0, first_split=None) -> dict[str, bytes]:
    tickers = _tickers()
    return {
        ticker: payload_for(
            ticker,
            first_count if index == 0 else 1,
            adj_delta=first_adj_delta if index == 0 else 0.0,
            split=first_split if index == 0 else None,
        )
        for index, ticker in enumerate(tickers)
    }


def _run(tmp_path: Path, *, payloads=None, opener=None, clock=FIXED_NOW, start=START, end=END, cutoff=CUTOFF, sleeps=None, universe_csv=UNIVERSE):
    payloads = payloads or _payloads()
    fake = opener or FakeOpener(payloads)
    sleep_calls = sleeps if sleeps is not None else []
    manifest = acquisition.acquire_seed_bundle(
        output_dir=tmp_path / "bundle",
        universe_csv=universe_csv,
        request_start=start,
        request_end_exclusive=end,
        seed_cutoff=cutoff,
        confirmation=acquisition.CONFIRMATION,
        opener=fake,
        clock=lambda: clock,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    return manifest, fake, tmp_path / "bundle", sleep_calls


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield calls


def test_exact_universe_sha_passes_and_preserves_order():
    value = acquisition.validate_universe_file(UNIVERSE)
    assert value["universe_csv_sha256"] == acquisition.EXPECTED_UNIVERSE_CSV_SHA256
    assert value["ticker_list_sha256"] == acquisition.EXPECTED_TICKER_LIST_SHA256
    assert value["ticker_count"] == 300
    assert value["tickers"][:3] == ["3633", "2984", "6150"]


def test_wrong_universe_sha_blocks_before_request(tmp_path, monkeypatch):
    altered = tmp_path / "V4_UNIVERSE.csv"
    altered.write_bytes(UNIVERSE.read_bytes() + b"\n")
    opener = FakeOpener(_payloads())
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="UNIVERSE_CSV_SHA_MISMATCH"):
        _run(tmp_path, opener=opener, universe_csv=altered)
    assert opener.calls == []


def test_ticker_list_hash_mismatch_blocks_before_request(monkeypatch, tmp_path):
    monkeypatch.setattr(acquisition, "EXPECTED_TICKER_LIST_SHA256", "0" * 64)
    opener = FakeOpener(_payloads())
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="TICKER_LIST_SHA_MISMATCH"):
        _run(tmp_path, opener=opener)
    assert opener.calls == []


def test_universe_count_mismatch_blocks_before_request(monkeypatch, tmp_path):
    monkeypatch.setattr(acquisition, "EXPECTED_TICKER_COUNT", 299)
    opener = FakeOpener(_payloads())
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="UNIVERSE_TICKER_COUNT_MISMATCH"):
        _run(tmp_path, opener=opener)
    assert opener.calls == []


def test_duplicate_ticker_blocks_before_request(monkeypatch, tmp_path):
    rows = list(csv.DictReader(UNIVERSE.read_text(encoding="utf-8").splitlines()))
    rows[-1]["ticker"] = rows[0]["ticker"]
    stream = __import__("io").StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["ticker", "market", "industry"], lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    altered = tmp_path / "V4_UNIVERSE.csv"
    raw = stream.getvalue().encode()
    altered.write_bytes(raw)
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    monkeypatch.setattr(acquisition, "EXPECTED_UNIVERSE_CSV_SHA256", hashlib.sha256(normalized).hexdigest())
    monkeypatch.setattr(acquisition, "EXPECTED_TICKER_LIST_SHA256", hashlib.sha256(("\n".join(row["ticker"] for row in rows) + "\n").encode()).hexdigest())
    opener = FakeOpener(_payloads())
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="UNIVERSE_DUPLICATE_TICKER"):
        _run(tmp_path, opener=opener, universe_csv=altered)
    assert opener.calls == []


def test_confirmation_mismatch_blocks_before_request(tmp_path):
    opener = FakeOpener(_payloads())
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="CONFIRMATION_MISMATCH"):
        acquisition.acquire_seed_bundle(
            output_dir=tmp_path / "bundle", universe_csv=UNIVERSE,
            request_start=START, request_end_exclusive=END, seed_cutoff=CUTOFF,
            confirmation="WRONG", opener=opener, clock=lambda: FIXED_NOW,
        )
    assert opener.calls == []


def test_existing_output_blocks_before_request(tmp_path):
    output = tmp_path / "bundle"; output.mkdir()
    opener = FakeOpener(_payloads())
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="OUTPUT_EXISTS"):
        acquisition.acquire_seed_bundle(
            output_dir=output, universe_csv=UNIVERSE, request_start=START,
            request_end_exclusive=END, seed_cutoff=CUTOFF,
            confirmation=acquisition.CONFIRMATION, opener=opener, clock=lambda: FIXED_NOW,
        )
    assert opener.calls == []


def test_acquisition_before_preregistration_blocks_before_request(tmp_path):
    opener = FakeOpener(_payloads())
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="ACQUISITION_NOT_AFTER_PREREGISTRATION"):
        _run(tmp_path, opener=opener, clock=datetime(2026, 8, 7, 2, 48, 27, tzinfo=timezone.utc))
    assert opener.calls == []


def test_invalid_request_bounds_block_before_request(tmp_path):
    opener = FakeOpener(_payloads())
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="REQUEST_DATE_BOUNDS_INVALID"):
        acquisition.acquire_seed_bundle(
            output_dir=tmp_path / "bundle", universe_csv=UNIVERSE,
            request_start="2026-08-08", request_end_exclusive="2026-08-08", seed_cutoff=CUTOFF,
            confirmation=acquisition.CONFIRMATION, opener=opener, clock=lambda: FIXED_NOW,
        )
    assert opener.calls == []


def test_sequential_fixed_order_and_one_request_per_ticker(tmp_path):
    _, opener, _, sleeps = _run(tmp_path)
    assert opener.calls == _tickers()
    assert len(opener.calls) == len(set(opener.calls)) == 300
    assert sleeps == [2.0] * 299


def test_retry_count_is_zero_and_first_429_stops_future_requests(tmp_path):
    tickers = _tickers()
    failures = {tickers[1]: FakeResponse(b"{}", status=429)}
    opener = FakeOpener(_payloads(), failures=failures)
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="HTTP_STATUS_429"):
        _run(tmp_path, opener=opener)
    assert opener.calls == tickers[:2]
    assert not (tmp_path / "bundle").exists()


def test_parser_failure_stops_future_requests(tmp_path):
    tickers = _tickers()
    bad = payload_for("WRONG")
    opener = FakeOpener(_payloads())
    opener.payloads[tickers[1]] = bad
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="SYMBOL_MISMATCH"):
        _run(tmp_path, opener=opener)
    assert opener.calls == tickers[:2]


def test_staging_only_during_acquisition_and_failure_cleans_staging(tmp_path):
    tickers = _tickers()
    output = tmp_path / "bundle"
    seen = []
    payloads = _payloads()

    class InspectingOpener(FakeOpener):
        def __call__(self, request):
            seen.append((output.exists(), list(tmp_path.glob("bundle.staging-*"))))
            return super().__call__(request)

    opener = InspectingOpener(payloads, failures={tickers[2]: FakeResponse(b"{}", status=500)})
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked):
        _run(tmp_path, opener=opener)
    assert seen[0][0] is False and seen[0][1]
    assert not output.exists()
    assert list(tmp_path.glob("bundle.staging-*")) == []


def test_success_publishes_atomically_and_exact_artifact_classes(tmp_path):
    manifest, _, output, _ = _run(tmp_path)
    assert set(item.name for item in output.iterdir()) == {
        "raw", "canonical_price_rows.csv", "canonical_split_events.json", "seed.csv", "seed_manifest.json"
    }
    assert len(list((output / "raw").glob("*.json"))) == 300
    assert manifest["success_count"] == manifest["ticker_count"] == 300


def test_raw_payload_sha_matches_collector_result(tmp_path):
    payloads = _payloads()
    manifest, _, output, _ = _run(tmp_path, payloads=payloads)
    raw = (output / "raw" / "3633.json").read_bytes()
    item = next(item for item in manifest["payload_manifest"] if item["ticker"] == "3633")
    assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(payloads["3633"]).hexdigest() == item["payload_sha256"]
    assert len(item["payload_sha256"]) == 64 and item["byte_count"] == len(raw)
    assert manifest["request_count"] == 300


def test_canonical_price_csv_schema_and_order(tmp_path):
    _, _, output, _ = _run(tmp_path)
    rows = list(csv.DictReader((output / "canonical_price_rows.csv").open(encoding="utf-8", newline="")))
    assert tuple(rows[0]) == acquisition.SEED_COLUMNS
    assert [(row["ticker"], row["trading_date"]) for row in rows] == sorted((row["ticker"], row["trading_date"]) for row in rows)


def test_seed_cutoff_excludes_later_rows(tmp_path):
    payloads = {ticker: payload_for(ticker, 1, start="2026-08-06") for ticker in _tickers()}
    payloads["3633"] = payload_for("3633", 2, start="2026-08-06")
    _, _, output, _ = _run(tmp_path, payloads=payloads, start="2026-08-06", end="2026-08-10", cutoff="2026-08-07")
    rows = list(csv.DictReader((output / "seed.csv").open(encoding="utf-8", newline="")))
    first = [row for row in rows if row["ticker"] == "3633"]
    assert all(row["trading_date"] <= "2026-08-07" for row in first)
    assert len(first) == 2


def test_latest_252_selection_and_ineligible_semantics(tmp_path):
    manifest, _, output, _ = _run(tmp_path, payloads=_payloads(first_count=253))
    first = next(item for item in manifest["seed_ticker_manifest"] if item["ticker"] == "3633")
    assert first["valid_observation_count"] == 252
    assert first["eligibility_at_seed"] is True
    assert first["first_seed_trading_date"] == "2025-07-02"
    assert manifest["ineligible_seed_ticker_count"] == 299
    assert len(list(csv.DictReader((output / "seed.csv").open(encoding="utf-8", newline="")))) == 252 + 299


def test_adj_close_is_preserved_in_seed(tmp_path):
    _, _, output, _ = _run(tmp_path, payloads=_payloads(first_adj_delta=-1.0))
    row = next(row for row in csv.DictReader((output / "seed.csv").open(encoding="utf-8", newline="")) if row["ticker"] == "3633")
    assert row["raw_close"] == "100.0"
    assert row["adj_close"] == "99.0"


def test_seed_hashes_are_deterministic_under_input_order():
    rows = [
        {"ticker": "B", "trading_date": "2026-01-02", "raw_open": 2.0, "raw_high": 3.0, "raw_low": 1.0, "raw_close": 2.5, "adj_close": 2.25, "raw_volume": 10.0},
        {"ticker": "A", "trading_date": "2026-01-01", "raw_open": 1.0, "raw_high": 2.0, "raw_low": 0.5, "raw_close": 1.5, "adj_close": 1.4, "raw_volume": 20.0},
    ]
    assert acquisition.canonical_seed_csv_bytes(rows) == acquisition.canonical_seed_csv_bytes(list(reversed(rows)))
    assert acquisition.canonical_rows_sha256(rows) == acquisition.canonical_rows_sha256(list(reversed(rows)))


def test_adj_close_change_changes_seed_hash():
    row = {"ticker": "A", "trading_date": "2026-01-01", "raw_open": 1.0, "raw_high": 2.0, "raw_low": 0.5, "raw_close": 1.5, "adj_close": 1.4, "raw_volume": 20.0}
    changed = dict(row, adj_close=1.3)
    assert acquisition.canonical_rows_sha256([row]) != acquisition.canonical_rows_sha256([changed])


def test_split_aggregation_is_canonical(tmp_path):
    payloads = _payloads(first_split=("2026-01-05", 2, 1))
    manifest, _, output, _ = _run(tmp_path, payloads=payloads)
    events = json.loads((output / "canonical_split_events.json").read_text(encoding="utf-8"))
    assert events == [{"ticker": "3633", "effective_date": "2026-01-05", "numerator": 2.0, "denominator": 1.0, "split_ratio": 2.0}]
    assert manifest["split_event_count"] == 1


def test_conflicting_and_duplicate_split_provenance_are_blocked():
    event = {"ticker": "3633", "effective_date": "2026-01-05", "numerator": 2.0, "denominator": 1.0}
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="DUPLICATE_SPLIT_EVENT"):
        acquisition.validate_canonical_split_events([event, dict(event)])
    with pytest.raises(acquisition.V7SeedAcquisitionBlocked, match="CONFLICTING_SPLIT_EVENT"):
        acquisition.validate_canonical_split_events([event, dict(event, numerator=3.0)])


def test_invalid_price_rows_are_excluded_and_audited(tmp_path):
    payloads = _payloads()
    value = json.loads(payloads["3633"])
    value["chart"]["result"][0]["indicators"]["quote"][0]["open"][0] = None
    payloads["3633"] = json.dumps(value).encode("utf-8")
    manifest, _, _, _ = _run(tmp_path, payloads=payloads)
    assert manifest["invalid_price_row_count"] == 1
    assert manifest["invalid_reason_counts"] == {"NONFINITE_OPEN": 1}
    assert manifest["valid_price_row_count"] == 299


def test_two_successful_fake_runs_are_byte_identical(tmp_path):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    _, _, first, _ = _run(tmp_path / "first")
    _, _, second, _ = _run(tmp_path / "second")
    first_files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    second_files = sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    assert first_files == second_files
    assert all((first / relative).read_bytes() == (second / relative).read_bytes() for relative in first_files)


def test_manifest_activation_and_calendar_boundaries_are_deferred(tmp_path):
    manifest, _, _, _ = _run(tmp_path)
    assert manifest["activation_boundary_status"] == "NOT_SET"
    assert manifest["activation_boundary_validation"] == "DEFERRED_TO_ACTIVATION_GATE"
    assert manifest["activation_status"] == "NOT_ACTIVATED"
    assert manifest["study_calendar_generated"] is False


def test_no_candidate_portfolio_profit_or_calendar_side_effects(tmp_path):
    manifest, _, output, _ = _run(tmp_path)
    assert manifest["candidate_generation_started"] == 0
    assert manifest["portfolio_simulation_started"] == 0
    assert manifest["profit_calculation_started"] == 0
    assert not (output / "calendar.json").exists()


def test_validate_seed_rows_hash_parity_on_known_boundary(tmp_path):
    manifest, _, output, _ = _run(tmp_path, payloads=_payloads(first_count=253))
    rows = list(csv.DictReader((output / "seed.csv").open(encoding="utf-8", newline="")))
    numeric_rows = []
    for row in rows:
        numeric_rows.append({
            **row,
            **{field: float(row[field]) for field in ("raw_open", "raw_high", "raw_low", "raw_close", "adj_close", "raw_volume")},
        })
    validation = validate_seed_rows(numeric_rows, _tickers(), "2026-08-08")
    assert validation["seed_canonical_sha256"] == manifest["seed_canonical_csv_sha256"]
    assert validation["row_count"] == manifest["seed_row_count"]


def test_fake_opener_is_the_only_network_boundary(tmp_path, no_real_urlopen):
    _run(tmp_path)
    assert no_real_urlopen == []
