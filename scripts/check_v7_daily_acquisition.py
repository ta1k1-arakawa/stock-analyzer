"""Synthetic-only V7 daily acquisition bundle verification CLI.

This CLI intentionally has no real network, real Yahoo, or activation
option. It exercises the append-only daily acquisition and verification
contract against a fully local, fake-opener-driven 300-ticker fixture.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v7_daily_acquisition import (
    CALENDAR_COMMIT,
    COLLECTOR_COMMIT,
    acquire_daily_bundle,
    verify_daily_acquisition_bundle,
)
from src.v7_jpx_calendar import build_calendar_snapshot, parse_jpx_holiday_html

UNIVERSE_CSV = ROOT / "V4_UNIVERSE.csv"
ENGINE_DAY = "2026-08-10"


def _synthetic_calendar_snapshot() -> dict[str, Any]:
    html = (
        "<html><body><nav><a href='/calendar/2026'>2026</a><a href='/calendar/2027'>2027</a></nav>"
        "<h2>Market Holidays</h2><table class='calendar-table'>"
        "<tr><th>2026</th></tr><tr><td> Jan. 1 (Thu.) </td><td> New Year Day </td></tr>"
        "<tr><th>2027</th></tr><tr><td> Jan. 1 (Fri.) </td><td> New Year Day </td></tr>"
        "</table></body></html>"
    ).encode("utf-8")
    holidays = parse_jpx_holiday_html(html)
    return build_calendar_snapshot(html, holidays, "2026-08-07T03:00:00Z")


def _universe_tickers() -> list[str]:
    with UNIVERSE_CSV.open(encoding="utf-8", newline="") as handle:
        return [row["ticker"] for row in csv.DictReader(handle)]


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def _payload_for(ticker: str) -> bytes:
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": [_epoch(ENGINE_DAY)],
        "indicators": {
            "quote": [{"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1000]}],
            "adjclose": [{"adjclose": [100.0]}],
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


class _FakeOpener:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, request_obj: Any) -> _FakeResponse:
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        return _FakeResponse(
            _payload_for(ticker),
            url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T",
        )


def run_synthetic_acquisition_test() -> dict[str, Any]:
    opener = _FakeOpener()
    calendar_snapshot = _synthetic_calendar_snapshot()
    clock_values = iter([
        datetime(2026, 8, 10, 7, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 10, 7, 12, tzinfo=timezone.utc),
    ])
    mono_state = [0.0]

    def monotonic_clock() -> float:
        return mono_state[0]

    def sleep_fn(seconds: float) -> None:
        mono_state[0] += seconds

    with tempfile.TemporaryDirectory(prefix="v7-daily-acquisition-") as temporary:
        manifest = acquire_daily_bundle(
            output_root=temporary,
            universe_csv=UNIVERSE_CSV,
            calendar_snapshot=calendar_snapshot,
            engine_day=ENGINE_DAY,
            opener=opener,
            clock=lambda: next(clock_values),
            monotonic_clock=monotonic_clock,
            sleep_fn=sleep_fn,
        )
        verification = verify_daily_acquisition_bundle(
            temporary, ENGINE_DAY, CALENDAR_COMMIT, COLLECTOR_COMMIT
        )

    tickers = _universe_tickers()
    if len(opener.calls) != len(tickers):
        raise AssertionError("REQUEST_COUNT_MISMATCH")
    if manifest["retry_count"] != 0:
        raise AssertionError("RETRY_COUNT_NOT_ZERO")
    if manifest["valid_d0_count"] != len(tickers):
        raise AssertionError("VALID_D0_COUNT_MISMATCH")
    if verification["status"] != "PASS":
        raise AssertionError("BUNDLE_VERIFICATION_NOT_PASS")

    return {
        "status": "PASS",
        "mode": "STATIC_SYNTHETIC_ONLY",
        "ticker_count": manifest["ticker_count"],
        "request_count": manifest["request_count"],
        "retry_count": manifest["retry_count"],
        "bundle_verification": verification["status"],
        "atomic_publish": True,
        "network_requests": 0,
        "candidate_generation": 0,
        "portfolio_processing": 0,
        "activation_created": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V7 daily acquisition synthetic-only checks")
    parser.add_argument("--synthetic-acquisition-test", action="store_true", required=True)
    parser.parse_args(argv)
    result = run_synthetic_acquisition_test()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
