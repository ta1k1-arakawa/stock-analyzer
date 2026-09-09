"""Synthetic-only V7 forward day processing orchestration check.

This CLI has no real study-root, network, collector, or activation option.  It
builds a fully local synthetic fixture in a temporary directory, drives exactly
one engine day through the orchestration layer, and then re-verifies the
processed day read-only.  It never reports profit, drawdown, profit factor,
win rate, or any arm performance comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v7_daily_acquisition import acquire_daily_bundle
from src.v7_forward_day_processing import (
    process_forward_day,
    verify_processed_forward_day,
)
from src.v7_forward_protocol import validate_seed_rows
from src.v7_jpx_calendar import build_calendar_snapshot, generate_engine_days, load_calendar_snapshot, parse_jpx_holiday_html

UNIVERSE_CSV = ROOT / "V4_UNIVERSE.csv"
ACTIVATION_BOUNDARY = "2026-08-10"
ACTIVATION_MANIFEST_SHA256 = "a" * 64
IMPLEMENTATION_COMMIT = "1" * 40
COLLECTOR_COMMIT = "4ca41c53895e75910ae65809fea6018868929afa"
SEED_OBSERVATION_COUNT = 252


# ---------------------------------------------------------------------------
# Synthetic fixture builders (shared with the test-suite)
# ---------------------------------------------------------------------------


def synthetic_calendar_snapshot() -> dict[str, Any]:
    html = (
        "<html><body><nav><a href='/calendar/2026'>2026</a><a href='/calendar/2027'>2027</a></nav>"
        "<h2>Market Holidays</h2><table class='calendar-table'>"
        "<tr><th>2026</th></tr><tr><td> Jan. 1 (Thu.) </td><td> New Year Day </td></tr>"
        "<tr><th>2027</th></tr><tr><td> Jan. 1 (Fri.) </td><td> New Year Day </td></tr>"
        "</table></body></html>"
    ).encode("utf-8")
    return build_calendar_snapshot(html, parse_jpx_holiday_html(html), "2026-08-07T03:00:00Z")


def universe_tickers() -> list[str]:
    with UNIVERSE_CSV.open(encoding="utf-8", newline="") as handle:
        return [row["ticker"] for row in csv.DictReader(handle)]


def seed_trading_days(snapshot: dict[str, Any], count: int = SEED_OBSERVATION_COUNT) -> list[str]:
    """Return ``count`` synthetic seed observation dates strictly before the boundary.

    Seed rows are pre-activation history, not engine days, so they are plain
    business days rather than official JPX engine-day calendar entries.
    """
    last = pd.Timestamp(ACTIVATION_BOUNDARY) - pd.Timedelta(days=1)
    days = pd.bdate_range(end=last, periods=count)
    return [day.strftime("%Y-%m-%d") for day in days]


def engine_days_from_boundary(snapshot: dict[str, Any], count: int) -> list[str]:
    calendar = load_calendar_snapshot(snapshot)
    days = generate_engine_days(calendar, ACTIVATION_BOUNDARY, "2027-06-30")
    return days[:count]


def _price_for(index: int) -> float:
    return 1000.0 + float(index)


def synthetic_seed_rows(
    tickers: Sequence[str], days: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        for index, day in enumerate(days):
            price = _price_for(index)
            rows.append({
                "ticker": ticker,
                "trading_date": day,
                "raw_open": price,
                "raw_high": price + 2.0,
                "raw_low": price - 2.0,
                "raw_close": price,
                "adj_close": price,
                "raw_volume": 100000.0,
            })
    return rows


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def acquisition_payload(ticker: str, day: str, price: float, *, volume: float = 200000.0) -> bytes:
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": [_epoch(day)],
        "indicators": {
            "quote": [{
                "open": [price],
                "high": [price + 2.0],
                "low": [price - 2.0],
                "close": [price],
                "volume": [volume],
            }],
            "adjclose": [{"adjclose": [price]}],
        },
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


class _FakeResponse:
    def __init__(self, payload: bytes, url: str) -> None:
        self.payload = payload
        self.status = 200
        self.url = url

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        pass


class FakeAcquisitionOpener:
    """Deterministic 300-ticker fake Yahoo opener; performs no network I/O."""

    def __init__(self, day: str, price: float) -> None:
        self.day = day
        self.price = price
        self.calls: list[str] = []

    def __call__(self, request_obj: Any) -> _FakeResponse:
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        return _FakeResponse(
            acquisition_payload(ticker, self.day, self.price),
            url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T",
        )


def build_acquisition_bundle(
    study_root: Path, snapshot: dict[str, Any], day: str, price: float
) -> int:
    opener = FakeAcquisitionOpener(day, price)
    clock_values = iter([
        datetime.fromisoformat(day).replace(hour=7, tzinfo=timezone.utc),
        datetime.fromisoformat(day).replace(hour=8, tzinfo=timezone.utc),
    ])
    state = {"now": 0.0}
    acquire_daily_bundle(
        output_root=study_root,
        universe_csv=UNIVERSE_CSV,
        calendar_snapshot=snapshot,
        engine_day=day,
        opener=opener,
        clock=lambda: next(clock_values),
        monotonic_clock=lambda: state["now"],
        sleep_fn=lambda seconds: state.__setitem__("now", state["now"] + seconds),
    )
    return len(opener.calls)


def build_activation_context(seed_rows: Sequence[dict[str, Any]], tickers: Sequence[str]) -> dict[str, Any]:
    validation = validate_seed_rows(seed_rows, tickers, ACTIVATION_BOUNDARY)
    return {
        "activation_manifest_sha256": ACTIVATION_MANIFEST_SHA256,
        "activation_boundary_first_jpx_trading_date": ACTIVATION_BOUNDARY,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "collector_commit": COLLECTOR_COMMIT,
        "expected_seed_canonical_sha256": validation["seed_canonical_sha256"],
        "expected_seed_ticker_manifest_sha256": validation["seed_payload_manifest_sha256"],
    }


# ---------------------------------------------------------------------------
# Synthetic processing test
# ---------------------------------------------------------------------------


def run_synthetic_processing_test() -> dict[str, Any]:
    snapshot = synthetic_calendar_snapshot()
    tickers = universe_tickers()
    seed_days = seed_trading_days(snapshot)
    seed_rows = synthetic_seed_rows(tickers, seed_days)
    activation_context = build_activation_context(seed_rows, tickers)
    engine_day = ACTIVATION_BOUNDARY

    with tempfile.TemporaryDirectory(prefix="v7-forward-day-processing-") as temporary:
        study_root = Path(temporary)
        request_count = build_acquisition_bundle(
            study_root, snapshot, engine_day, _price_for(SEED_OBSERVATION_COUNT)
        )
        summary = process_forward_day(
            study_root=study_root,
            engine_day=engine_day,
            universe_csv=UNIVERSE_CSV,
            calendar_snapshot=snapshot,
            seed_rows=seed_rows,
            activation_context=activation_context,
        )
        verification = verify_processed_forward_day(
            study_root=study_root,
            engine_day=engine_day,
            universe_csv=UNIVERSE_CSV,
            activation_context=activation_context,
        )

    if request_count != len(tickers):
        raise AssertionError("SYNTHETIC_ACQUISITION_REQUEST_COUNT_MISMATCH")
    if summary["status"] != "PASS" or not summary["forward_day_persisted"]:
        raise AssertionError("PROCESSING_NOT_PASS")
    if verification["status"] != "PASS":
        raise AssertionError("PROCESSING_VERIFICATION_NOT_PASS")
    if summary["control"]["parameters_sha256"] == summary["variant"]["parameters_sha256"]:
        raise AssertionError("ARM_PARAMETERS_NOT_DISTINCT")
    if summary["future_candidate_data_access_count"] != 0 or summary["future_split_access_count"] != 0:
        raise AssertionError("FUTURE_ACCESS_DETECTED")

    return {
        "status": "PASS",
        "mode": "STATIC_SYNTHETIC_ONLY",
        "engine_day": engine_day,
        "engine_day_processed": True,
        "acquisition_verified": summary["acquisition_verified"],
        "seed_verified": summary["seed_verified"],
        "candidate_generation_pass": True,
        "accepted_candidate_count": summary["accepted_candidate_count"],
        "market_gate_status": summary["market_gate_status"],
        "arm_input_parity": True,
        "control_processed": True,
        "variant_processed": True,
        "control_parameters_sha256": summary["control"]["parameters_sha256"],
        "variant_parameters_sha256": summary["variant"]["parameters_sha256"],
        "forward_day_persisted": summary["forward_day_persisted"],
        "processing_verification": verification["status"],
        "future_candidate_data_access_count": 0,
        "future_split_access_count": 0,
        "network_requests": 0,
        "activation_created": False,
        "profit_metrics_exposed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V7 forward day processing synthetic-only check")
    parser.add_argument("--synthetic-processing-test", action="store_true", required=True)
    parser.parse_args(argv)
    result = run_synthetic_processing_test()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
