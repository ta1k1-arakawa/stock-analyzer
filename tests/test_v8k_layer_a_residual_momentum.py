from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v5_b_candidate_ranker import BASELINE_ARM, generate_candidates, simulate_portfolio
from src.v8k_layer_a_residual_momentum import (
    MAX_CANDIDATES,
    arm_metrics,
    attach_residual_scores,
    build_ranked_arms,
    build_scorecard,
    canonical_scorecard_bytes,
    execute_arms,
    generate_eligible_candidates,
    rank_baseline,
    rank_residual,
    write_scorecard,
)


def _frame(slope: float, *, periods: int = 290) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=periods)
    close = 100.0 + slope * np.arange(periods)
    signal_index = 270
    close[signal_index] = close[signal_index - 5] * 0.99
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Adj Close": close, "Volume": np.full(periods, 1_000_000)},
        index=dates,
    )


def _inputs(count: int = 25):
    prices = {str(1000 + index): _frame(.12 + index * .012) for index in range(count)}
    universe = pd.DataFrame({"ticker": list(prices), "industry": ["A" if index < count // 2 else "B" for index in range(count)]})
    day = next(iter(prices.values())).index[270]
    return prices, universe, day


def _manual_eligible(count: int = 21) -> pd.DataFrame:
    day = pd.Timestamp("2020-01-15")
    return pd.DataFrame([
        {"evaluation_year": 2020, "signal_date": day, "entry_date": day + pd.offsets.BDay(1), "exit_date": day + pd.offsets.BDay(5),
         "ticker": f"T{index:02d}", "industry": "A", "return_5d": -.01, "return_20d": .02,
         "return_60d": float(count - index), "close_to_ma20": .01, "close_to_ma60": .02, "atr14": 1.0,
         "candidate_status": "CANDIDATE"}
        for index in range(count)
    ])


def test_baseline_top20_matches_existing_v5b_and_preserves_pre_cutoff_eligibility():
    prices, universe, day = _inputs()
    eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day)
    expected = generate_candidates(prices, universe, signal_from=day, signal_to=day)
    actual = rank_baseline(eligible)
    assert len(eligible) > MAX_CANDIDATES
    assert set(eligible.candidate_status) == {"CANDIDATE"}
    assert list(actual[["ticker", "baseline_rank"]].itertuples(index=False, name=None)) == list(expected[["ticker", "rank"]].itertuples(index=False, name=None))


def test_peer_population_uses_universe_peers_and_excludes_candidate_itself():
    prices, universe, day = _inputs(4)
    universe.loc[2:, "industry"] = "B"
    eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day)
    candidate = eligible[eligible.ticker.eq("1000")].copy()
    scored = attach_residual_scores(candidate, prices, universe)
    peer_value = (prices["1001"].loc[day, "Adj Close"] / prices["1001"].loc[prices["1001"].index[210], "Adj Close"]) - 1
    assert int(scored.iloc[0].industry_peer_count) == 1
    assert scored.iloc[0].industry_peer_median_60d == pytest.approx(peer_value)
    assert scored.iloc[0].residual_momentum == pytest.approx(candidate.iloc[0].return_60d - peer_value)


def test_missing_or_empty_peer_is_unavailable_without_fallback():
    prices, universe, day = _inputs(3)
    universe.loc[0, "industry"] = ""
    universe.loc[1:, "industry"] = "B"
    eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day)
    empty_industry = attach_residual_scores(eligible[eligible.ticker.eq("1000")], prices, universe)
    lone_industry = attach_residual_scores(eligible[eligible.ticker.eq("1001")], prices, universe.iloc[[0, 1]])
    assert empty_industry.iloc[0].residual_status == "RESIDUAL_SCORE_UNAVAILABLE"
    assert lone_industry.iloc[0].residual_status == "RESIDUAL_SCORE_UNAVAILABLE"
    assert rank_residual(empty_industry).empty


def test_future_price_and_outcome_mutations_cannot_change_residual_score_or_ranking():
    prices, universe, day = _inputs(4)
    eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day)
    original = attach_residual_scores(eligible, prices, universe)
    changed = {ticker: frame.copy() for ticker, frame in prices.items()}
    for frame in changed.values():
        frame.loc[frame.index > day, ["Open", "High", "Low", "Close", "Adj Close"]] *= 9.0
    altered = attach_residual_scores(eligible, changed, universe)
    columns = ["ticker", "residual_momentum", "industry_peer_median_60d", "residual_status"]
    pd.testing.assert_frame_equal(original[columns].reset_index(drop=True), altered[columns].reset_index(drop=True))
    pd.testing.assert_frame_equal(rank_residual(original)[["ticker", "ai_rank"]], rank_residual(altered)[["ticker", "ai_rank"]])


def test_residual_order_ties_and_top20_difference_are_exact():
    eligible = _manual_eligible()
    baseline = rank_baseline(eligible)
    scored = eligible.copy()
    scored["residual_status"] = "RESIDUAL_SCORE_AVAILABLE"
    scored["residual_momentum"] = list(reversed(range(len(scored))))
    # Equal residual and return values must use ticker ascending as the final tie-break.
    scored.loc[0:1, ["residual_momentum", "return_60d", "return_20d"]] = 100.0
    scored.loc[20, "residual_momentum"] = 99.0
    residual = rank_residual(scored)
    assert list(residual.iloc[:2].ticker) == ["T00", "T01"]
    assert set(baseline.ticker) != set(residual.ticker)


def test_new_baseline_execution_is_identical_to_direct_v5b_execution():
    prices, universe, day = _inputs()
    _, baseline, residual = build_ranked_arms(prices, universe, signal_from=day, signal_to=day)
    direct_trades, direct_equity = simulate_portfolio(baseline, prices, BASELINE_ARM)
    core_trades, core_equity, _, _ = execute_arms(baseline, residual, prices)
    pd.testing.assert_frame_equal(direct_trades, core_trades)
    pd.testing.assert_frame_equal(direct_equity, core_equity)
    scorecard = build_scorecard(prices, universe, provenance={"synthetic": True}, repository_commit="SYNTHETIC")
    assert scorecard["baseline"]["filled_trade_count"] >= 0


def test_mtm_and_book_drawdowns_are_separate():
    trades = pd.DataFrame(columns=["status", "realized_net_profit_yen", "evaluation_year", "exit_date", "entry_cost", "industry"])
    equity = pd.DataFrame({"evaluation_year": [2020, 2020, 2020], "open_positions": [0, 1, 0],
                           "mark_to_market_equity": [400000.0, 300000.0, 400000.0],
                           "book_equity": [400000.0, 390000.0, 400000.0]})
    metrics = arm_metrics(trades, equity)
    assert metrics["mtm_maximum_drawdown"] == pytest.approx(25.0)
    assert metrics["book_cost_maximum_drawdown"] == pytest.approx(2.5)


def test_concentration_turnover_and_exposure_metrics():
    trades = pd.DataFrame({"status": ["FILLED", "FILLED", "FILLED"], "realized_net_profit_yen": [100.0, 50.0, -25.0],
                           "evaluation_year": [2020, 2020, 2021], "exit_date": pd.to_datetime(["2020-01-03", "2020-02-03", "2021-01-03"]),
                           "entry_cost": [1000.0, 2000.0, 3000.0], "industry": ["A", "A", "B"]})
    equity = pd.DataFrame({"evaluation_year": [2020, 2020], "open_positions": [1, 2],
                           "mark_to_market_equity": [400000.0, 400000.0], "book_equity": [400000.0, 400000.0]})
    metrics = arm_metrics(trades, equity)
    assert metrics["gross_entry_notional_yen"] == 6000.0
    assert metrics["entry_notional_turnover_multiple"] == pytest.approx(6000.0 / 2_400_000.0)
    assert metrics["average_open_positions"] == 1.5
    assert metrics["slot_utilization_fraction"] == .75
    assert metrics["top5_positive_trade_profit_share"] == 1.0
    assert metrics["maximum_industry_positive_profit_share"] == 1.0


def test_scorecard_is_deterministic_and_writes_exactly_one_artifact(tmp_path: Path):
    prices, universe, _ = _inputs()
    first = canonical_scorecard_bytes(build_scorecard(prices, universe, provenance={"safe": 1}))
    second = canonical_scorecard_bytes(build_scorecard(prices, universe, provenance={"safe": 1}))
    assert first == second
    output = tmp_path / "outside-repo-output"
    write_scorecard(output, first, Path.cwd())
    assert [path.name for path in output.iterdir()] == ["scorecard.json"]
    assert json.loads((output / "scorecard.json").read_text())["measurement_status"] == "COMPLETE"


def test_output_guards_and_no_2026_signals(tmp_path: Path):
    prices, universe, day = _inputs()
    with pytest.raises(ValueError, match="OUTPUT_INSIDE_REPOSITORY_PROHIBITED"):
        write_scorecard(Path.cwd() / "v8k-test-output", b"{}", Path.cwd())
    nonempty = tmp_path / "nonempty"; nonempty.mkdir(); (nonempty / "old").write_text("x")
    with pytest.raises(ValueError, match="OUTPUT_DIRECTORY_NONEMPTY_OR_FILE"):
        write_scorecard(nonempty, b"{}", Path.cwd())
    with pytest.raises(ValueError, match="EVALUATION_SIGNAL_OUTSIDE_2020_2025"):
        generate_eligible_candidates(prices, universe, signal_from="2026-01-01", signal_to="2026-01-01")
    assert day.year == 2020


def test_module_has_no_network_client():
    source = inspect.getsource(__import__("src.v8k_layer_a_residual_momentum", fromlist=["*"]))
    assert "requests" not in source
    assert "urllib" not in source
