"""Synthetic-only V8 raw historical acquisition check.

This CLI has no real-network, real-output-root, or real-partition option,
and no bypass flag of any kind (no ``--skip-source-hash``, ``--force``,
``--ignore-parity``, or similar). It builds a fully local synthetic ticker
fixture and OHLCV payload generator in a temporary workspace, drives the
production T1 and T2 acquisition paths with a fake opener (zero real HTTP
requests), proves T3 acquisition is unconditionally rejected, and proves the
T2 sealed-holdout access guard blocks every official research entry point.
It never touches the real Yahoo host or any real private V8 storage.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v8_historical_acquisition import (
    REQUEST_END_EXCLUSIVE,
    REQUEST_START,
    V8HistoricalAcquisitionBlocked,
    V8SealedHoldoutBlocked,
    _acquire_historical_block_bundle_with_validated_inputs,
    open_for_backtest,
    open_for_candidate_generation,
    open_for_feature_generation,
    open_for_profit_evaluation,
    open_for_validation,
    read_acquisition_manifest,
)

T1_TICKERS = ("9101", "9102", "9103")
T2_TICKERS = ("9201", "9202", "9203")
SYNTHETIC_PARTITION_MANIFEST_SHA256 = "s" * 64  # not a real partition manifest
SYNTHETIC_IMPLEMENTATION_GIT_COMMIT = "a" * 40


def _epoch(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def synthetic_payload(ticker: str, price: float) -> bytes:
    dates = [(2016, 4, 1), (2016, 4, 4), (2025, 12, 30)]
    timestamps = [_epoch(*d) for d in dates]
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": timestamps,
        "indicators": {
            "quote": [{
                "open": [price] * len(timestamps),
                "high": [price + 2.0] * len(timestamps),
                "low": [price - 2.0] * len(timestamps),
                "close": [price] * len(timestamps),
                "volume": [10000.0] * len(timestamps),
            }],
            "adjclose": [{"adjclose": [price] * len(timestamps)}],
        },
        "events": {},
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


class FakeYahooOpener:
    """Deterministic fake Yahoo Chart opener; performs no network I/O."""

    def __init__(self, base_price: float) -> None:
        self.base_price = base_price
        self.calls: list[str] = []

    def __call__(self, request_obj: Any) -> _FakeResponse:
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        price = self.base_price + float(len(self.calls))
        return _FakeResponse(
            synthetic_payload(ticker, price),
            url="https://query1.finance.yahoo.com/v8/finance/chart/" + ticker + ".T",
        )


def run_synthetic_acquisition_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v8-historical-acquisition-") as temporary:
        output_root = Path(temporary) / "private-v8-storage"

        fake_clock_state = {"now": 0.0}
        sleep_calls: list[float] = []

        def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)
            fake_clock_state["now"] += seconds

        opener_t1 = FakeYahooOpener(base_price=1000.0)
        manifest_t1 = _acquire_historical_block_bundle_with_validated_inputs(
            output_root=output_root,
            repository_root=ROOT,
            block="T1",
            tickers=T1_TICKERS,
            partition_manifest_sha256=SYNTHETIC_PARTITION_MANIFEST_SHA256,
            implementation_git_commit=SYNTHETIC_IMPLEMENTATION_GIT_COMMIT,
            opener=opener_t1,
            clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
            monotonic_clock=lambda: fake_clock_state["now"],
            sleep_fn=fake_sleep,
        )
        if manifest_t1["status"] != "RAW_ACQUIRED_NOT_OPENED":
            raise AssertionError("T1_STATUS_UNEXPECTED")
        if manifest_t1["validation_access_count"] != 0:
            raise AssertionError("T1_VALIDATION_ACCESS_COUNT_NOT_ZERO")

        opener_t2 = FakeYahooOpener(base_price=2000.0)
        manifest_t2 = _acquire_historical_block_bundle_with_validated_inputs(
            output_root=output_root,
            repository_root=ROOT,
            block="T2",
            tickers=T2_TICKERS,
            partition_manifest_sha256=SYNTHETIC_PARTITION_MANIFEST_SHA256,
            implementation_git_commit=SYNTHETIC_IMPLEMENTATION_GIT_COMMIT,
            opener=opener_t2,
            clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
            monotonic_clock=lambda: fake_clock_state["now"],
            sleep_fn=fake_sleep,
        )
        if manifest_t2["status"] != "RAW_ACQUIRED_SEALED":
            raise AssertionError("T2_STATUS_UNEXPECTED")
        if manifest_t2["sealed"] is not True:
            raise AssertionError("T2_NOT_SEALED")

        # T3 acquisition must always BLOCK, with no bypass.
        t3_blocked = False
        try:
            _acquire_historical_block_bundle_with_validated_inputs(
                output_root=output_root,
                repository_root=ROOT,
                block="T3",
                tickers=("9301",),
                partition_manifest_sha256=SYNTHETIC_PARTITION_MANIFEST_SHA256,
                implementation_git_commit=SYNTHETIC_IMPLEMENTATION_GIT_COMMIT,
                opener=FakeYahooOpener(base_price=3000.0),
                clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
            )
        except V8HistoricalAcquisitionBlocked as error:
            t3_blocked = error.reason.startswith("V8_BLOCK_ACQUISITION_PROHIBITED")
        if not t3_blocked:
            raise AssertionError("T3_ACQUISITION_NOT_BLOCKED")

        # The T2 sealed-holdout guard must block every official research
        # entry point using the just-published manifest.
        sealed_manifest = read_acquisition_manifest(output_root, "T2")
        guard_functions = (
            open_for_feature_generation,
            open_for_candidate_generation,
            open_for_validation,
            open_for_backtest,
            open_for_profit_evaluation,
        )
        guard_block_count = 0
        for guard in guard_functions:
            try:
                guard(sealed_manifest)
            except V8SealedHoldoutBlocked:
                guard_block_count += 1
        if guard_block_count != len(guard_functions):
            raise AssertionError("T2_GUARD_DID_NOT_BLOCK_ALL_OPERATIONS")

        if len(sleep_calls) == 0 or any(seconds < 2.0 - 1e-9 for seconds in sleep_calls):
            raise AssertionError("RATE_LIMIT_SPACING_VIOLATED")

    return {
        "status": "PASS",
        "mode": "STATIC_SYNTHETIC_ONLY",
        "request_start": REQUEST_START,
        "request_end_exclusive": REQUEST_END_EXCLUSIVE,
        "t1_status": manifest_t1["status"],
        "t1_ticker_count": manifest_t1["ticker_count"],
        "t1_validation_access_count": manifest_t1["validation_access_count"],
        "t2_status": manifest_t2["status"],
        "t2_sealed": manifest_t2["sealed"],
        "t2_research_access_authorized": manifest_t2["research_access_authorized"],
        "t2_opened": False,
        "t3_acquisition_blocked": True,
        "guard_blocks_all_research_operations": True,
        "retry_count": 0,
        "http_429_count": manifest_t1["http_429_count"] + manifest_t2["http_429_count"],
        "network_requests": 0,
        "data_acquired": 0,
        "real_acquisition_created": False,
        "backtests": 0,
        "profit_calculated": 0,
        "models_fitted": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V8 historical acquisition synthetic-only check")
    parser.add_argument("--synthetic-test", action="store_true", required=True)
    parser.parse_args(argv)
    result = run_synthetic_acquisition_test()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
