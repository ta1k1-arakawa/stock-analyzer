"""V8 raw historical acquisition CLI: synthetic check and production runner.

``--synthetic-test`` is unchanged and uses only local fixtures. The mutually
exclusive ``--production-acquire`` path requires block, persisted partition
manifest, private output root, and a block-specific confirmation. It delegates
all integrity, provenance, storage, and T1/T2 enforcement to the public
manifest-bound acquisition API; it exposes no ticker, hash, date, retry, host,
T3, or seal-bypass override. Tests inject a fake opener and never contact Yahoo.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import urllib.request
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
    acquire_historical_block_bundle,
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
PRODUCTION_CONFIRMATIONS = {
    "T1": "V8_PRODUCTION_ACQUIRE_T1",
    "T2": "V8_PRODUCTION_ACQUIRE_T2",
}


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


def _utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def run_production_acquisition(
    *,
    block: str,
    partition_manifest_path: Path,
    output_root: Path,
    opener: Any = urllib.request.urlopen,
    clock: Any = _utc_clock,
    implementation_git_commit_resolver: Any = None,
    monotonic_clock: Any = None,
    sleep_fn: Any = None,
) -> dict[str, Any]:
    """Run one manifest-bound production acquisition without CLI overrides.

    The public acquisition API owns all manifest, identity, ticker, hash,
    provenance, storage, and block validation. Optional injected callables are
    for fake-only tests; the CLI never exposes them as user inputs.
    """
    kwargs: dict[str, Any] = {
        "output_root": output_root,
        "repository_root": ROOT,
        "block": block,
        "partition_manifest_path": partition_manifest_path,
        "opener": opener,
        "clock": clock,
    }
    if implementation_git_commit_resolver is not None:
        kwargs["implementation_git_commit_resolver"] = implementation_git_commit_resolver
    if monotonic_clock is not None:
        kwargs["monotonic_clock"] = monotonic_clock
    if sleep_fn is not None:
        kwargs["sleep_fn"] = sleep_fn
    manifest = acquire_historical_block_bundle(**kwargs)
    return {
        "status": "PASS",
        "mode": "PRODUCTION",
        "block": manifest["block"],
        "role": manifest["role"],
        "acquisition_manifest_path": str(
            Path(output_root) / "acquisitions" / manifest["block"] / "acquisition_manifest.json"
        ),
        "partition_manifest_sha256": manifest["partition_manifest_sha256"],
        "implementation_git_commit": manifest["implementation_git_commit"],
        "sealed": manifest["sealed"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V8 historical acquisition CLI")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-test", action="store_true")
    mode.add_argument("--production-acquire", action="store_true")
    parser.add_argument("--block", default=None)
    parser.add_argument("--partition-manifest", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--confirmation", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.synthetic_test:
        result = run_synthetic_acquisition_test()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    if not args.block or not args.partition_manifest or not args.output_root or not args.confirmation:
        parser.error("--production-acquire requires --block, --partition-manifest, --output-root, and --confirmation")
    expected_confirmation = PRODUCTION_CONFIRMATIONS.get(args.block)
    if expected_confirmation is None:
        print(json.dumps({"status": "BLOCKED", "reason": "V8_BLOCK_ACQUISITION_PROHIBITED:" + args.block}, sort_keys=True))
        return 2
    if args.confirmation != expected_confirmation:
        print(json.dumps({"status": "BLOCKED", "reason": "CONFIRMATION_MISMATCH"}, sort_keys=True))
        return 2
    try:
        result = run_production_acquisition(
            block=args.block,
            partition_manifest_path=Path(args.partition_manifest),
            output_root=Path(args.output_root),
        )
    except V8HistoricalAcquisitionBlocked as error:
        print(json.dumps({"status": "BLOCKED", "reason": error.reason}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
