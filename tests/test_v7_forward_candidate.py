from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.v6_a_confirmed_breakout import generate_candidates, normalize_universe
from src.v7_forward_candidate import V7CandidateBlocked, generate_forward_candidates_for_day


COLLECTOR_COMMIT = "a" * 40
FEATURE_FIELDS = (
    "raw_close", "adj_close", "ma20", "ma60", "return1", "return20", "return60",
    "volatility10", "volatility60", "median_turnover60", "median_volume60",
    "prior_high20", "volume_surprise", "atr14", "atr14_percent",
    "breakout_strength_atr",
)


@pytest.fixture(scope="module")
def fixture():
    calendar = pd.bdate_range("2019-01-02", periods=263)
    engine_day = calendar[252]
    tickers = [f"T{index:03d}" for index in range(300)]
    universe = pd.DataFrame({
        "ticker": tickers,
        "market": ["JP"] * len(tickers),
        "industry": [f"IND{index % 10:02d}" for index in range(len(tickers))],
    })
    frames = {}
    for ticker in tickers:
        close = 1000.0 + np.arange(len(calendar), dtype=float)
        volume = np.full(len(calendar), 100000.0)
        volume[252] = 200000.0
        frames[ticker] = pd.DataFrame({
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        }, index=calendar)
    splits = {ticker: set() for ticker in tickers}
    reference_universe = normalize_universe(universe)
    reference_candidates, reference_gates, reference_audit = generate_candidates(
        frames, reference_universe, splits, calendar,
        signal_from=engine_day, signal_to=engine_day,
    )
    forward_frames = {ticker: frame.loc[:engine_day].copy() for ticker, frame in frames.items()}
    forward = generate_forward_candidates_for_day(
        forward_frames, universe, splits, calendar, engine_day, COLLECTOR_COMMIT,
    )
    return {
        "calendar": calendar, "engine_day": engine_day, "universe": universe,
        "frames": frames, "forward_frames": forward_frames, "splits": splits,
        "reference_candidates": reference_candidates,
        "reference_gate": reference_gates[engine_day],
        "reference_audit": reference_audit, "forward": forward,
    }


def _forward(data, frames=None, splits=None, calendar=None):
    return generate_forward_candidates_for_day(
        data["forward_frames"] if frames is None else frames,
        data["universe"], data["splits"] if splits is None else splits,
        data["calendar"] if calendar is None else calendar,
        data["engine_day"], COLLECTOR_COMMIT,
    )


def test_no_split_candidate_key_parity(fixture):
    reference = fixture["reference_candidates"].head(20)
    forward = fixture["forward"]["accepted_top20"]
    assert [(str(row.signal_date.date()), row.ticker) for _, row in reference.iterrows()] == [
        (row["signal_date"], row["ticker"]) for row in forward
    ]


def test_rank_parity(fixture):
    reference = fixture["reference_candidates"].head(20)
    assert list(reference["rank"]) == [int(row["rank"]) for row in fixture["forward"]["accepted_top20"]]


def test_feature_parity(fixture):
    reference = fixture["reference_candidates"].head(20).sort_values("ticker")
    forward = sorted(fixture["forward"]["accepted_top20"], key=lambda row: row["ticker"])
    for (_, expected), actual in zip(reference.iterrows(), forward):
        for field in FEATURE_FIELDS:
            assert np.isclose(float(expected[field]), float(actual[field]), rtol=1e-12, atol=1e-12)


def test_market_gate_parity(fixture):
    expected = fixture["reference_gate"]
    actual = fixture["forward"]["market_gate"]
    assert actual["market_gate_status"] == expected["market_gate_status"]
    assert actual["market_denominator_count"] == expected["market_denominator_count"]
    assert np.isclose(actual["breadth_above_ma60"], expected["breadth_above_ma60"])
    assert np.isclose(actual["cross_sectional_median_return20"], expected["cross_sectional_median_return20"])


def test_d1_row_absent_at_d0_passes(fixture):
    result = _forward(fixture)
    assert result["accepted_top20"]
    assert result["entry_attempt_date"] not in {
        date for frame in fixture["forward_frames"].values() for date in frame.index.astype(str)
    }


def test_d10_row_absent_at_d0_passes(fixture):
    result = _forward(fixture)
    assert result["planned_exit_date"] > result["engine_day"]
    assert result["accepted_top20"]


def test_future_d1_d10_value_changes_do_not_change_forward_hash(fixture):
    first = _forward(fixture)
    changed_reference = {ticker: frame.copy() for ticker, frame in fixture["frames"].items()}
    for frame in changed_reference.values():
        frame.loc[fixture["calendar"][253], "Open"] = 999999.0
        frame.loc[fixture["calendar"][262], "Open"] = 888888.0
    changed = {ticker: frame.loc[:fixture["engine_day"]].copy() for ticker, frame in changed_reference.items()}
    second = _forward(fixture, frames=changed)
    assert first["candidate_snapshot_sha256"] == second["candidate_snapshot_sha256"]
    assert first["market_gate_snapshot_sha256"] == second["market_gate_snapshot_sha256"]
    assert first["price_snapshot_sha256"] == second["price_snapshot_sha256"]


def test_future_price_row_is_rejected_with_audited_counter(fixture):
    with pytest.raises(V7CandidateBlocked, match="FUTURE_CANDIDATE_DATA_ACCESS") as error:
        _forward(fixture, frames=fixture["frames"])
    assert error.value.future_candidate_data_access_count == 1
    assert error.value.future_split_access_count == 0


def test_future_split_is_rejected_with_audited_counter(fixture):
    splits = deepcopy(fixture["splits"])
    splits["T000"] = {fixture["calendar"][253]}
    with pytest.raises(V7CandidateBlocked, match="FUTURE_SPLIT_ACCESS") as error:
        _forward(fixture, splits=splits)
    assert error.value.future_candidate_data_access_count == 0
    assert error.value.future_split_access_count == 1


def test_calendar_without_d10_fails_closed(fixture):
    with pytest.raises(V7CandidateBlocked, match="D10_UNAVAILABLE"):
        _forward(fixture, calendar=fixture["calendar"][:260])


def test_candidate_and_market_price_hashes_are_deterministic(fixture):
    first = _forward(fixture)
    second = _forward(fixture)
    assert first["candidate_snapshot_sha256"] == second["candidate_snapshot_sha256"]
    assert first["market_gate_snapshot_sha256"] == second["market_gate_snapshot_sha256"]
    assert first["price_snapshot_sha256"] == second["price_snapshot_sha256"]


def test_d0_data_unavailable_is_audited_not_study_blocked(fixture):
    frames = dict(fixture["forward_frames"])
    frames.pop("T000")
    result = _forward(fixture, frames=frames)
    rows = [row for row in result["full_candidate_audit"] if row.get("ticker") == "T000"]
    assert rows[0]["candidate_rejection_reason"] == "D0_DATA_UNAVAILABLE"
    assert result["market_gate"]["market_gate_status"] == "MARKET_GATE_PASS"


def test_ticker_tie_break_and_top20_are_fixed(fixture):
    accepted = fixture["forward"]["accepted_top20"]
    assert len(accepted) == 20
    assert [row["rank"] for row in accepted] == list(range(1, 21))
    assert [row["ticker"] for row in accepted] == sorted(row["ticker"] for row in accepted)
    outside = [row for row in fixture["forward"]["full_candidate_audit"] if row.get("candidate_status") == "RANK_OUTSIDE_TOP20"]
    assert len(outside) == 280


def test_251_observations_are_ineligible_and_252_prior_observations_are_eligible(fixture):
    short = dict(fixture["forward_frames"])
    short["T000"] = short["T000"].iloc[1:].copy()
    result = _forward(fixture, frames=short)
    row = next(row for row in result["full_candidate_audit"] if row.get("ticker") == "T000")
    assert row["candidate_rejection_reason"] == "D0_DATA_UNAVAILABLE"
    assert any(row["ticker"] == "T001" for row in result["accepted_top20"])


def test_result_contains_forward_read_counters_and_calendar_dates(fixture):
    result = fixture["forward"]
    assert result["future_candidate_data_access_count"] == 0
    assert result["future_split_access_count"] == 0
    assert result["entry_attempt_date"] == fixture["calendar"][253].strftime("%Y-%m-%d")
    assert result["planned_exit_date"] == fixture["calendar"][262].strftime("%Y-%m-%d")
