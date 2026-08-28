from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.v7_forward_candidate import generate_forward_candidates_for_day
from src.v7_forward_protocol import ProtocolBlocked, validate_seed_rows
from src.v7_seed_bridge import FRAME_COLUMNS, V7SeedBridgeBlocked, build_forward_frames_from_seed_and_d0


COLLECTOR_COMMIT = "b" * 40


@pytest.fixture(scope="module")
def bridge_fixture():
    calendar = pd.bdate_range("2019-01-02", periods=263)
    engine_day = calendar[252]
    tickers = [f"T{index:03d}" for index in range(120)]
    universe = pd.DataFrame({
        "ticker": tickers,
        "market": ["JP"] * len(tickers),
        "industry": [f"IND{index:03d}" for index in range(len(tickers))],
    })
    frames = {}
    seed_rows = []
    d0_rows = []
    for ticker in tickers:
        close = 1000.0 + np.arange(253, dtype=float)
        volume = np.full(253, 100000.0)
        volume[-1] = 200000.0
        frames[ticker] = pd.DataFrame({
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        }, index=calendar[:253])
        for index, day in enumerate(calendar[:252]):
            price = 1000.0 + index
            seed_rows.append({
                "ticker": ticker,
                "trading_date": day.strftime("%Y-%m-%d"),
                "raw_open": price,
                "raw_high": price + 2.0,
                "raw_low": price - 2.0,
                "raw_close": price,
                "adj_close": price,
                "raw_volume": 100000.0,
            })
        price = 1252.0
        d0_rows.append({
            "ticker": ticker,
            "trading_date": engine_day.strftime("%Y-%m-%d"),
            "raw_open": price,
            "raw_high": price + 2.0,
            "raw_low": price - 2.0,
            "raw_close": price,
            "adj_close": price,
            "raw_volume": 200000.0,
        })
    validated = validate_seed_rows(seed_rows, tickers, engine_day.strftime("%Y-%m-%d"))
    return {
        "calendar": calendar,
        "engine_day": engine_day,
        "tickers": tickers,
        "universe": universe,
        "frames": frames,
        "seed_rows": seed_rows,
        "d0_rows": d0_rows,
        "validated": validated,
    }


def _candidate(frames, fixture):
    return generate_forward_candidates_for_day(
        frames,
        fixture["universe"],
        {ticker: set() for ticker in fixture["tickers"]},
        fixture["calendar"],
        fixture["engine_day"],
        COLLECTOR_COMMIT,
    )


def test_adj_close_is_required(bridge_fixture):
    rows = [dict(row) for row in bridge_fixture["seed_rows"]]
    rows[0].pop("adj_close")
    with pytest.raises(ProtocolBlocked, match="SEED_SCHEMA_MISSING:adj_close"):
        validate_seed_rows(rows, bridge_fixture["tickers"], bridge_fixture["engine_day"].strftime("%Y-%m-%d"))


@pytest.mark.parametrize("value,reason", [(float("nan"), "SEED_NONFINITE_ADJ_CLOSE"), (float("inf"), "SEED_NONFINITE_ADJ_CLOSE")])
def test_adj_close_nonfinite_is_rejected(bridge_fixture, value, reason):
    rows = [dict(row) for row in bridge_fixture["seed_rows"]]
    rows[0]["adj_close"] = value
    with pytest.raises(ProtocolBlocked, match=reason):
        validate_seed_rows(rows, bridge_fixture["tickers"], bridge_fixture["engine_day"].strftime("%Y-%m-%d"))


def test_adj_close_nonpositive_is_rejected(bridge_fixture):
    rows = [dict(row) for row in bridge_fixture["seed_rows"]]
    rows[0]["adj_close"] = 0.0
    with pytest.raises(ProtocolBlocked, match="SEED_NONPOSITIVE_ADJ_CLOSE"):
        validate_seed_rows(rows, bridge_fixture["tickers"], bridge_fixture["engine_day"].strftime("%Y-%m-%d"))


def test_adj_close_is_bound_into_seed_hash(bridge_fixture):
    first = bridge_fixture["validated"]
    changed = [dict(row) for row in bridge_fixture["seed_rows"]]
    changed[0]["adj_close"] += 1.0
    second = validate_seed_rows(changed, bridge_fixture["tickers"], bridge_fixture["engine_day"].strftime("%Y-%m-%d"))
    assert first["seed_canonical_sha256"] != second["seed_canonical_sha256"]


def test_seed_row_order_does_not_change_hash(bridge_fixture):
    first = bridge_fixture["validated"]
    second = validate_seed_rows(
        list(reversed(bridge_fixture["seed_rows"])),
        list(reversed(bridge_fixture["tickers"])),
        bridge_fixture["engine_day"].strftime("%Y-%m-%d"),
    )
    assert first["seed_canonical_sha256"] == second["seed_canonical_sha256"]
    assert first["seed_payload_manifest_sha256"] == second["seed_payload_manifest_sha256"]


def test_bridge_reconstructs_252_prior_plus_d0(bridge_fixture):
    frames = build_forward_frames_from_seed_and_d0(
        bridge_fixture["validated"], bridge_fixture["d0_rows"], bridge_fixture["engine_day"]
    )
    assert set(frames) == set(bridge_fixture["tickers"])
    assert all(len(frame) == 253 for frame in frames.values())
    assert all(frame.index[-1] == bridge_fixture["engine_day"] for frame in frames.values())


def test_251_prior_plus_d0_is_ineligible(bridge_fixture):
    last_seed_date = bridge_fixture["seed_rows"][251]["trading_date"]
    short_seed = [
        row for row in bridge_fixture["seed_rows"]
        if row["trading_date"] != last_seed_date
    ]
    validated = validate_seed_rows(short_seed, bridge_fixture["tickers"], bridge_fixture["engine_day"].strftime("%Y-%m-%d"))
    frames = build_forward_frames_from_seed_and_d0(validated, bridge_fixture["d0_rows"], bridge_fixture["engine_day"])
    result = _candidate(frames, bridge_fixture)
    assert result["accepted_top20"] == []
    ticker_rows = [row for row in result["full_candidate_audit"] if row.get("ticker")]
    assert all(row.get("candidate_rejection_reason") == "D0_DATA_UNAVAILABLE" for row in ticker_rows)


def test_future_bridge_row_is_rejected(bridge_fixture):
    future = dict(bridge_fixture["d0_rows"][0])
    future["trading_date"] = (bridge_fixture["engine_day"] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with pytest.raises(V7SeedBridgeBlocked, match="FUTURE_BRIDGE_ROW"):
        build_forward_frames_from_seed_and_d0(bridge_fixture["validated"], [future], bridge_fixture["engine_day"])


def test_duplicate_ticker_date_is_rejected(bridge_fixture):
    duplicate = [bridge_fixture["d0_rows"][0], dict(bridge_fixture["d0_rows"][0])]
    with pytest.raises(V7SeedBridgeBlocked, match="DUPLICATE_TICKER_DATE"):
        build_forward_frames_from_seed_and_d0(bridge_fixture["validated"], duplicate, bridge_fixture["engine_day"])


def test_bridge_output_has_exact_six_v6_columns(bridge_fixture):
    frames = build_forward_frames_from_seed_and_d0(
        bridge_fixture["validated"], bridge_fixture["d0_rows"], bridge_fixture["engine_day"]
    )
    assert all(tuple(frame.columns) == FRAME_COLUMNS for frame in frames.values())
    assert all(frame.index.is_monotonic_increasing for frame in frames.values())


def test_bridge_is_deterministic_under_row_and_ticker_order_changes(bridge_fixture):
    first = build_forward_frames_from_seed_and_d0(
        bridge_fixture["validated"], bridge_fixture["d0_rows"], bridge_fixture["engine_day"]
    )
    reversed_seed = dict(bridge_fixture["validated"])
    reversed_seed["canonical_rows"] = list(reversed(bridge_fixture["validated"]["canonical_rows"]))
    second = build_forward_frames_from_seed_and_d0(
        reversed_seed, list(reversed(bridge_fixture["d0_rows"])), bridge_fixture["engine_day"]
    )
    assert list(first) == list(second)
    for ticker in first:
        pd.testing.assert_frame_equal(first[ticker], second[ticker])


def test_missing_d0_ticker_returns_seed_only_frame(bridge_fixture):
    d0_without_one = bridge_fixture["d0_rows"][1:]
    frames = build_forward_frames_from_seed_and_d0(
        bridge_fixture["validated"], d0_without_one, bridge_fixture["engine_day"]
    )
    missing_ticker = bridge_fixture["d0_rows"][0]["ticker"]
    assert len(frames[missing_ticker]) == 252
    result = _candidate(frames, bridge_fixture)
    rejected = next(row for row in result["full_candidate_audit"] if row["ticker"] == missing_ticker)
    assert rejected["candidate_rejection_reason"] == "D0_DATA_UNAVAILABLE"


def test_direct_253_frame_and_seed_bridge_are_candidate_identical(bridge_fixture):
    bridged = build_forward_frames_from_seed_and_d0(
        bridge_fixture["validated"], bridge_fixture["d0_rows"], bridge_fixture["engine_day"]
    )
    direct = _candidate(bridge_fixture["frames"], bridge_fixture)
    reconstructed = _candidate(bridged, bridge_fixture)
    assert direct["market_gate"] == reconstructed["market_gate"]
    assert direct["accepted_top20"] == reconstructed["accepted_top20"]
    assert direct["candidate_snapshot_sha256"] == reconstructed["candidate_snapshot_sha256"]
    assert direct["price_snapshot_sha256"] == reconstructed["price_snapshot_sha256"]
    assert direct["market_gate_snapshot_sha256"] == reconstructed["market_gate_snapshot_sha256"]
    for left, right in zip(direct["accepted_top20"], reconstructed["accepted_top20"]):
        for field in (
            "raw_close", "adj_close", "ma20", "ma60", "return1", "return20", "return60",
            "volatility10", "volatility60", "median_turnover60", "median_volume60",
            "prior_high20", "volume_surprise", "atr14", "atr14_percent", "breakout_strength_atr",
        ):
            assert math.isclose(float(left[field]), float(right[field]), rel_tol=0, abs_tol=0)
