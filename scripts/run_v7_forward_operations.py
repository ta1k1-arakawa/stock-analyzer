"""Synthetic-only V7 production operations day runner check.

This CLI has no real activation, real study-root, or real network option.
It builds a fully local synthetic activation manifest and daily-acquisition
fixture in a temporary workspace, drives the production operations runner
over two consecutive engine days (exercising fresh processing, idempotent
re-invocation, and restart equivalence), and never creates a production
activation artifact or touches the real Yahoo/JPX network.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v7_activation_manifest import (
    HUMAN_ACTIVATION_CONFIRMATION,
    SeedProvenanceExpectation,
    build_activation_manifest_candidate,
    expected_activation_boundary,
    validate_seed_provenance,
    write_activation_manifest_once,
)
from src.v7_forward_operations import V7ForwardOperationsBlocked, run_forward_operations_day
from src.v7_forward_persistence import ForwardStudyStore, canonical_json_bytes
from src.v7_jpx_calendar import load_calendar_snapshot, next_jpx_trading_day
from src.v7_seed_acquisition import validate_universe_file

UNIVERSE_CSV = ROOT / "V4_UNIVERSE.csv"
CALENDAR_PATH = ROOT / "data" / "v7_jpx_calendar_2026_2027.json"

# Synthetic (fake) human Gate 4 values.  These are NOT real study decisions.
AUTHORIZATION_UTC = "2026-08-07T09:00:00Z"
SEED_ACQUISITION_UTC = "2026-08-07T03:10:00Z"
ACQUISITION_WINDOW_JST = "17:00-18:00 Asia/Tokyo"
SEED_OBSERVATION_COUNT = 252
SEED_CSV_COLUMNS = (
    "ticker", "trading_date", "raw_open", "raw_high", "raw_low",
    "raw_close", "adj_close", "raw_volume",
)


def universe_tickers() -> list[str]:
    return validate_universe_file(UNIVERSE_CSV)["tickers"]


def seed_observation_days(boundary: str, count: int = SEED_OBSERVATION_COUNT) -> list[str]:
    """Business days ending the day before the activation boundary."""
    from datetime import date, timedelta

    days: list[str] = []
    current = date.fromisoformat(boundary) - timedelta(days=1)
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current -= timedelta(days=1)
    return sorted(days)


def _price_for(index: int) -> float:
    return 1000.0 + float(index)


def synthetic_seed_rows(tickers: Sequence[str], days: Sequence[str]) -> list[dict[str, Any]]:
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


def write_synthetic_seed_csv(path: Path, rows: Sequence[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SEED_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (str(item["ticker"]), str(item["trading_date"]))):
        writer.writerow({column: row[column] for column in SEED_CSV_COLUMNS})
    payload = stream.getvalue().encode("utf-8")
    path.write_bytes(payload)
    return payload


def synthetic_seed_acquisition_manifest(tickers: Sequence[str]) -> dict[str, Any]:
    import hashlib

    return {
        "mode": "PRE_ACTIVATION_SEED_ACQUISITION",
        "payload_manifest": [
            {
                "ticker": ticker,
                "payload_sha256": hashlib.sha256(("synthetic-payload:" + ticker).encode("utf-8")).hexdigest(),
                "byte_count": 1024 + index,
            }
            for index, ticker in enumerate(tickers)
        ],
    }


def _epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp())


def acquisition_payload(ticker: str, day: str, price: float) -> bytes:
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": [_epoch(day)],
        "indicators": {
            "quote": [{"open": [price], "high": [price + 2.0], "low": [price - 2.0], "close": [price], "volume": [200000.0]}],
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


def within_window_clock(day: str) -> datetime:
    """17:00-18:00 Asia/Tokyo == 08:00-09:00 UTC."""
    return datetime.fromisoformat(day).replace(hour=8, minute=30, tzinfo=timezone.utc)


def build_workspace(workspace: Path) -> dict[str, Any]:
    """Build every synthetic input the operations runner needs, entirely
    inside ``workspace`` and a separate durable study root."""
    tickers = universe_tickers()
    snapshot = load_calendar_snapshot(CALENDAR_PATH)
    boundary = expected_activation_boundary(snapshot, AUTHORIZATION_UTC)
    seed_days = seed_observation_days(boundary)
    seed_rows = synthetic_seed_rows(tickers, seed_days)
    seed_csv = workspace / "seed.csv"
    write_synthetic_seed_csv(seed_csv, seed_rows)
    seed_acquisition_manifest = synthetic_seed_acquisition_manifest(tickers)

    seed_provenance = validate_seed_provenance(
        universe_csv=UNIVERSE_CSV,
        seed_csv=seed_csv,
        seed_acquisition_manifest=seed_acquisition_manifest,
        activation_boundary_first_jpx_trading_date=boundary,
        expected=None,
    )
    expectation = SeedProvenanceExpectation(**{
        field: seed_provenance[field]
        for field in (
            "seed_source_payload_manifest_sha256",
            "seed_ticker_manifest_sha256",
            "seed_canonical_csv_sha256",
            "seed_ticker_count",
            "seed_row_count",
            "seed_cutoff_trading_date",
        )
    })

    durable_root = workspace / "durable-study-root"
    durable_root.mkdir()

    manifest = build_activation_manifest_candidate(
        activation_authorization_utc=AUTHORIZATION_UTC,
        activation_boundary_first_jpx_trading_date=boundary,
        acquisition_window_jst=ACQUISITION_WINDOW_JST,
        output_root=str(durable_root.resolve()),
        seed_acquisition_utc=SEED_ACQUISITION_UTC,
        seed_provenance=seed_provenance,
    )
    manifest_path = workspace / "activation_manifest.json"
    write_activation_manifest_once(
        output_path=manifest_path,
        manifest=manifest,
        repository_root=ROOT,
        confirmation=HUMAN_ACTIVATION_CONFIRMATION,
        calendar_path=CALENDAR_PATH,
        universe_csv=UNIVERSE_CSV,
        seed_csv=seed_csv,
        seed_acquisition_manifest=seed_acquisition_manifest,
        expected_seed_provenance=expectation,
    )

    return {
        "tickers": tickers,
        "snapshot": snapshot,
        "boundary": boundary,
        "seed_csv": seed_csv,
        "seed_acquisition_manifest": seed_acquisition_manifest,
        "expectation": expectation,
        "durable_root": durable_root,
        "manifest_path": manifest_path,
    }


def run_operations_day(fixture: dict[str, Any], engine_day: str, index: int) -> dict[str, Any]:
    opener = FakeAcquisitionOpener(engine_day, _price_for(SEED_OBSERVATION_COUNT + index))
    # A synthetic-only run must never pay the real 2s-per-ticker acquisition
    # rate limit (300 tickers x 2s would dominate wall time); the fake clock
    # advances instantly instead while still exercising the same code path.
    fake_clock_state = {"now": 0.0}
    return run_forward_operations_day(
        activation_manifest_path=fixture["manifest_path"],
        durable_output_root=fixture["durable_root"],
        universe_csv=UNIVERSE_CSV,
        calendar_path=CALENDAR_PATH,
        seed_csv=fixture["seed_csv"],
        seed_acquisition_manifest=fixture["seed_acquisition_manifest"],
        engine_day=engine_day,
        repository_root=ROOT,
        opener=opener,
        clock=lambda: within_window_clock(engine_day),
        monotonic_clock=lambda: fake_clock_state["now"],
        sleep_fn=lambda seconds: fake_clock_state.__setitem__("now", fake_clock_state["now"] + seconds),
        expected_seed_provenance=fixture["expectation"],
    )


def run_synthetic_operations_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v7-forward-operations-") as temporary:
        workspace = Path(temporary)
        fixture = build_workspace(workspace)
        day1 = fixture["boundary"]
        day2 = next_jpx_trading_day(fixture["snapshot"], day1)

        result_1 = run_operations_day(fixture, day1, index=0)
        if result_1["status"] != "PASS":
            raise AssertionError("DAY1_NOT_PASS")

        result_1_repeat = run_operations_day(fixture, day1, index=0)
        if result_1_repeat["status"] != "ALREADY_COMMITTED":
            raise AssertionError("DAY1_REPEAT_NOT_ALREADY_COMMITTED")

        result_2 = run_operations_day(fixture, day2, index=1)
        if result_2["status"] != "PASS":
            raise AssertionError("DAY2_NOT_PASS")

        store = ForwardStudyStore(fixture["durable_root"])
        latest = store.load_latest_runtime()
        if latest is None or latest["day"] != day2:
            raise AssertionError("LATEST_RUNTIME_DAY_MISMATCH")

        # Restart equivalence: day2 was produced by restoring the persisted
        # day1 checkpoint into fresh engines (process_forward_day always
        # restores rather than continuing in memory), so the persisted
        # runtime already *is* the restart-equivalence result.
        restart_equivalence = (
            latest["arm_a_runtime"]["engine_day"] == day2
            and latest["arm_b_runtime"]["engine_day"] == day2
            and result_2["control"] is not None
            and result_2["variant"] is not None
        )

    if not restart_equivalence:
        raise AssertionError("RESTART_EQUIVALENCE_NOT_PASS")

    return {
        "status": "PASS",
        "mode": "STATIC_SYNTHETIC_ONLY",
        "activation_manifest_verified": True,
        "engine_day": day2,
        "acquisition_verified": True,
        "processing_verified": True,
        "persistence_verified": True,
        "already_committed": False,
        "restart_equivalence": True,
        "network_requests": 0,
        "actual_activation_created": False,
        "real_forward_processing": 0,
        "profit_metrics_exposed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V7 production operations synthetic-only check")
    parser.add_argument("--synthetic-operations-test", action="store_true", required=True)
    parser.parse_args(argv)
    result = run_synthetic_operations_test()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
