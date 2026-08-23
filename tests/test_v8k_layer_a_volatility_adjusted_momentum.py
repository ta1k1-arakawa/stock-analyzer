from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.v5_b_candidate_ranker as v5
import src.v8k_layer_a_volatility_adjusted_momentum as variant
from src.v5_b_candidate_ranker import BASELINE_ARM, generate_candidates, simulate_portfolio
from src.v8k_layer_a_volatility_adjusted_momentum import (
    MAX_CANDIDATES, attach_volatility_adjusted_scores, build_ranked_arms,
    build_scorecard, canonical_scorecard_bytes, execute_arms,
    fill_mechanism, generate_eligible_candidates, rank_baseline,
    rank_volatility_adjusted, top20_mechanism, write_scorecard,
)


def _frame(slope: float, *, periods: int = 290) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=periods); close = 100 + slope * np.arange(periods); close[270] = close[265] * .99
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * .99, "Close": close, "Adj Close": close, "Volume": np.full(periods, 1_000_000)}, index=dates)


def _inputs(count: int = 25):
    prices = {str(1000 + number): _frame(.12 + .012 * number) for number in range(count)}; universe = pd.DataFrame({"ticker": list(prices), "industry": ["A" if number < count // 2 else "B" for number in range(count)]})
    return prices, universe, next(iter(prices.values())).index[270]


def test_baseline_parity_and_exact_score_formula():
    prices, universe, day = _inputs(); eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day); expected = generate_candidates(prices, universe, signal_from=day, signal_to=day)
    assert list(rank_baseline(eligible)[["ticker", "baseline_rank"]].itertuples(index=False, name=None)) == list(expected[["ticker", "rank"]].itertuples(index=False, name=None))
    scored = attach_volatility_adjusted_scores(eligible, prices); row = scored.iloc[0]; close = prices[row.ticker].loc[row.signal_date, "Adj Close"]; normalized = row.atr14 / close
    assert row.normalized_atr14 == pytest.approx(normalized); assert row.risk_adjusted_momentum_score == pytest.approx(row.return_60d / normalized)


def test_nonfinite_or_zero_denominators_are_unavailable():
    prices, universe, day = _inputs(3); eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day).iloc[:3].copy()
    prices[eligible.iloc[0].ticker].loc[day, "Adj Close"] = 0; eligible.loc[eligible.index[1], "atr14"] = 0; eligible.loc[eligible.index[2], "atr14"] = np.nan
    scored = attach_volatility_adjusted_scores(eligible, prices)
    assert set(scored.volatility_adjusted_status) == {"SCORE_UNAVAILABLE"}


def test_future_and_outcome_mutations_cannot_change_score_or_rank():
    prices, universe, day = _inputs(); eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day); original = attach_volatility_adjusted_scores(eligible, prices); changed = {ticker: frame.copy() for ticker, frame in prices.items()}
    for frame in changed.values(): frame.loc[frame.index > day, ["Open", "High", "Low", "Close", "Adj Close"]] *= 8
    altered = attach_volatility_adjusted_scores(eligible.assign(realized_d5_return=999), changed)
    columns = ["ticker", "normalized_atr14", "risk_adjusted_momentum_score", "volatility_adjusted_status"]
    pd.testing.assert_frame_equal(original[columns].reset_index(drop=True), altered[columns].reset_index(drop=True)); pd.testing.assert_frame_equal(rank_volatility_adjusted(original)[["ticker", "ai_rank"]], rank_volatility_adjusted(altered)[["ticker", "ai_rank"]])


def test_ties_and_variant_top20_difference():
    day = pd.Timestamp("2020-01-15"); eligible = pd.DataFrame({"signal_date": day, "ticker": [f"T{number:02d}" for number in range(25)], "return_20d": .1, "return_60d": list(range(25))})
    scored = eligible.copy(); scored["volatility_adjusted_status"] = "SCORE_AVAILABLE"; scored["risk_adjusted_momentum_score"] = list(reversed(range(25))); scored.loc[:1, ["risk_adjusted_momentum_score", "return_60d", "return_20d"]] = 100
    ranked = rank_volatility_adjusted(scored); assert list(ranked.ticker[:2]) == ["T00", "T01"]; assert set(ranked.ticker) != set(rank_baseline(eligible).ticker)


def test_execution_and_fast_d5_outcomes_match_existing_reference():
    prices, universe, day = _inputs(); _, baseline, rows = build_ranked_arms(prices, universe, signal_from=day, signal_to=day); direct_trades, direct_equity = simulate_portfolio(baseline, prices, BASELINE_ARM); trades, equity, _, _ = execute_arms(baseline, rows, prices)
    pd.testing.assert_frame_equal(direct_trades, trades); pd.testing.assert_frame_equal(direct_equity, equity)
    frame = prices["1000"].copy(); gap = frame.copy(); position = frame.index.get_loc(day); gap.iloc[position + 1, gap.columns.get_loc("Open")] = gap.iloc[position].Close * 1.0101; short = frame.iloc[:position + 3]
    raw = {"VALID": frame, "GAP": gap, "SHORT": short}; frames = variant._normalized_price_frames(raw); inputs = pd.DataFrame({"ticker": ["VALID", "GAP", "SHORT"], "signal_date": [day, day, short.index[-3]]}); actual = variant._realized_d5_state(inputs, frames)
    for ticker, signal in zip(inputs.ticker, inputs.signal_date):
        expected = v5.d5_target(raw[ticker], signal); value = actual[(pd.Timestamp(signal), ticker)]; assert value is None if expected is None else value == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_outcome_reuse_and_normalization_structure(monkeypatch):
    prices, universe, day = _inputs(); eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day); unique = len({(pd.Timestamp(signal), ticker) for ticker, signal in zip(eligible.ticker, eligible.signal_date)}); normalizations, outcomes = [], []
    original_frame, original_target = variant._as_frame, variant._normalized_d5_target
    def counted_frame(frame): normalizations.append(id(frame)); return original_frame(frame)
    def counted_target(frame, signal): outcomes.append((id(frame), pd.Timestamp(signal))); return original_target(frame, signal)
    monkeypatch.setattr(variant, "_as_frame", counted_frame); monkeypatch.setattr(variant, "_normalized_d5_target", counted_target)
    build_scorecard(prices, universe, provenance={"safe": 1})
    assert len(normalizations) == len(prices) == len(set(normalizations)); assert len(outcomes) == unique == len(set(outcomes))


def test_discrimination_mechanism_and_deterministic_scorecard():
    prices, universe, day = _inputs(); scored = attach_volatility_adjusted_scores(generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day), prices); outcomes = variant._realized_d5_state(scored, variant._normalized_price_frames(prices)); diagnostics = variant._all_eligible_discrimination(scored, outcomes)
    assert sum(item["count"] for item in diagnostics["pooled_score_quintiles"].values()) == diagnostics["valid_row_count"]
    base_rows = pd.DataFrame({"signal_date": [day, day], "ticker": ["A", "B"]}); variant_rows = pd.DataFrame({"signal_date": [day, day], "ticker": ["B", "C"]}); assert top20_mechanism(base_rows, variant_rows)["overall_jaccard"] == pytest.approx(1 / 3)
    columns = {"evaluation_year": [2020, 2020], "signal_date": [day, day], "ticker": ["A", "B"], "status": ["FILLED", "FILLED"], "realized_net_profit_yen": [10., -5.]}; filled = fill_mechanism(pd.DataFrame(columns), pd.DataFrame({**columns, "ticker": ["B", "C"], "realized_net_profit_yen": [-5., 20.]})); assert filled["common_fills"] == 1 and filled["baseline_only_pnl"] == 10
    assert canonical_scorecard_bytes(build_scorecard(prices, universe, provenance={"safe": 1})) == canonical_scorecard_bytes(build_scorecard(prices, universe, provenance={"safe": 1}))


def test_output_guards_no_2026_and_no_network_client(tmp_path: Path):
    prices, universe, _ = _inputs()
    with pytest.raises(ValueError, match="OUTPUT_INSIDE_REPOSITORY_PROHIBITED"): write_scorecard(Path.cwd() / "volatility-output", b"{}", Path.cwd())
    output = tmp_path / "output"; write_scorecard(output, b"{}", Path.cwd()); assert [path.name for path in output.iterdir()] == ["scorecard.json"]
    with pytest.raises(ValueError, match="OUTPUT_DIRECTORY_NONEMPTY_OR_FILE"): write_scorecard(output, b"{}", Path.cwd())
    with pytest.raises(ValueError, match="EVALUATION_SIGNAL_OUTSIDE_2020_2025"): generate_eligible_candidates(prices, universe, signal_from="2026-01-01", signal_to="2026-01-01")
    source = inspect.getsource(variant); assert "requests" not in source and "urllib" not in source
