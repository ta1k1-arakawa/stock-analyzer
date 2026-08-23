from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.v8k_layer_a_short_horizon_reversal as reversal
from src.v5_b_candidate_ranker import BASELINE_ARM, generate_candidates, simulate_portfolio
from src.v8k_layer_a_short_horizon_reversal import (
    MAX_CANDIDATES, attach_reversal_scores, build_ranked_arms, build_scorecard,
    canonical_scorecard_bytes, execute_arms, fill_mechanism,
    generate_eligible_candidates, rank_baseline, rank_reversal, top20_mechanism,
    write_scorecard,
)


def _frame(slope: float, *, periods: int = 290) -> pd.DataFrame:
    dates = pd.bdate_range("2019-01-01", periods=periods)
    close = 100.0 + slope * np.arange(periods)
    close[270] = close[265] * .99
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * .99,
                         "Close": close, "Adj Close": close, "Volume": np.full(periods, 1_000_000)}, index=dates)


def _inputs(count: int = 25):
    prices = {str(1000 + index): _frame(.12 + index * .012) for index in range(count)}
    universe = pd.DataFrame({"ticker": list(prices), "industry": ["A" if index < count // 2 else "B" for index in range(count)]})
    return prices, universe, next(iter(prices.values())).index[270]


def _slow_scores(eligible, prices, universe, _normalized_frames=None):
    normalized = reversal.normalize_universe(universe)
    industries = normalized.set_index("ticker").industry.to_dict(); members = set(normalized.ticker)
    frames = {reversal.canonical_ticker(ticker): frame for ticker, frame in prices.items()}
    output = eligible.copy(); medians = []; scores = []; statuses = []; counts = []
    for _, row in output.iterrows():
        ticker, day = reversal.canonical_ticker(row.ticker), pd.Timestamp(row.signal_date); industry = industries.get(ticker, "")
        values = []
        if industry:
            for peer in members:
                if peer == ticker or industries.get(peer, "") != industry or peer not in frames: continue
                frame = reversal._as_frame(frames[peer])
                try: position = frame.index.get_loc(day)
                except KeyError: continue
                if isinstance(position, (int, np.integer)) and position >= 5:
                    value = float(frame.iloc[position].AdjClose) / float(frame.iloc[position - 5].AdjClose) - 1.0
                    if np.isfinite(value): values.append(value)
        if values:
            median = float(np.median(values)); medians.append(median); scores.append(median - float(row.return_5d)); statuses.append("REVERSAL_SCORE_AVAILABLE"); counts.append(len(values))
        else:
            medians.append(np.nan); scores.append(np.nan); statuses.append("REVERSAL_SCORE_UNAVAILABLE"); counts.append(0)
    output["industry_peer_median_5d"] = medians; output["relative_selloff_score"] = scores; output["reversal_status"] = statuses; output["industry_peer_count"] = counts
    return output


def _edge_inputs():
    prices, _, day = _inputs(1)
    prices = {"A0": prices["1000"], "A1": _frame(.14), "A2": _frame(.16), "AMISSING": _frame(.18), "ASHORT": _frame(.2), "ANAN": _frame(.22), "B0": _frame(.24), "B1": _frame(.26), "C0": _frame(.28), "C1": _frame(.30), "EMPTY": _frame(.32)}
    prices["AMISSING"] = prices["AMISSING"].drop(day); prices["ASHORT"] = prices["ASHORT"].iloc[:4].copy(); prices["ANAN"].loc[day, "Adj Close"] = np.nan
    industries = {"A0": "A", "A1": "A", "A2": "A", "AMISSING": "A", "ASHORT": "A", "ANAN": "A", "B0": "B", "B1": "B", "C0": "C", "C1": "C", "EMPTY": ""}
    universe = pd.DataFrame({"ticker": list(prices), "industry": [industries[key] for key in prices]})
    eligible = pd.DataFrame({"ticker": list(prices), "signal_date": day, "return_5d": np.linspace(-.04, -.01, len(prices)), "return_20d": .1, "return_60d": .2})
    return prices, universe, eligible, day


def test_baseline_eligibility_and_top20_match_v5b():
    prices, universe, day = _inputs()
    eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day)
    expected = generate_candidates(prices, universe, signal_from=day, signal_to=day)
    actual = rank_baseline(eligible)
    assert list(actual[["ticker", "baseline_rank"]].itertuples(index=False, name=None)) == list(expected[["ticker", "rank"]].itertuples(index=False, name=None))


def test_reference_equivalence_all_peer_edge_cases_and_exact_five_observations():
    prices, universe, eligible, day = _edge_inputs()
    actual, expected = attach_reversal_scores(eligible, prices, universe), _slow_scores(eligible, prices, universe)
    assert list(actual.industry_peer_count) == list(expected.industry_peer_count)
    assert list(actual.reversal_status) == list(expected.reversal_status)
    np.testing.assert_allclose(actual.industry_peer_median_5d, expected.industry_peer_median_5d, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(actual.relative_selloff_score, expected.relative_selloff_score, rtol=1e-12, atol=1e-12, equal_nan=True)
    peer = prices["C1"]; position = peer.index.get_loc(day)
    assert actual.loc[actual.ticker.eq("C0"), "industry_peer_median_5d"].iloc[0] == pytest.approx(float(peer.iloc[position]["Adj Close"] / peer.iloc[position - 5]["Adj Close"] - 1.0))
    assert actual.loc[actual.ticker.eq("EMPTY"), "reversal_status"].iloc[0] == "REVERSAL_SCORE_UNAVAILABLE"


def test_universe_peer_self_exclusion_and_missing_peer_fallback():
    prices, universe, eligible, _ = _edge_inputs()
    candidate = eligible[eligible.ticker.eq("C0")]
    scored = attach_reversal_scores(candidate, prices, universe)
    assert scored.iloc[0].industry_peer_count == 1
    assert scored.iloc[0].reversal_status == "REVERSAL_SCORE_AVAILABLE"
    peer, day = prices["C1"], pd.Timestamp(candidate.signal_date.iloc[0])
    position = peer.index.get_loc(day)
    assert scored.iloc[0].industry_peer_median_5d == pytest.approx(float(peer.iloc[position]["Adj Close"] / peer.iloc[position - 5]["Adj Close"] - 1.0))
    only = attach_reversal_scores(candidate, prices, universe[universe.ticker.eq("C0")])
    assert only.iloc[0].reversal_status == "REVERSAL_SCORE_UNAVAILABLE"


def test_future_and_outcome_mutations_cannot_change_score_or_rank():
    prices, universe, day = _inputs()
    eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day)
    original = attach_reversal_scores(eligible, prices, universe); changed = {ticker: frame.copy() for ticker, frame in prices.items()}
    for frame in changed.values(): frame.loc[frame.index > day, ["Open", "High", "Low", "Close", "Adj Close"]] *= 7
    altered = attach_reversal_scores(eligible.assign(realized_d5_return=999), changed, universe)
    columns = ["ticker", "industry_peer_median_5d", "relative_selloff_score", "reversal_status"]
    pd.testing.assert_frame_equal(original[columns].reset_index(drop=True), altered[columns].reset_index(drop=True))
    pd.testing.assert_frame_equal(rank_reversal(original)[["ticker", "ai_rank"]], rank_reversal(altered)[["ticker", "ai_rank"]])


def test_reversal_ties_and_top20_difference():
    day = pd.Timestamp("2020-01-15")
    eligible = pd.DataFrame({"signal_date": day, "ticker": [f"T{i:02d}" for i in range(25)], "return_5d": -.01, "return_20d": .1, "return_60d": list(range(25))})
    scored = eligible.copy(); scored["reversal_status"] = "REVERSAL_SCORE_AVAILABLE"; scored["relative_selloff_score"] = list(reversed(range(25)))
    scored.loc[:1, ["relative_selloff_score", "return_60d", "return_20d"]] = 100
    ranked = rank_reversal(scored)
    assert list(ranked.ticker[:2]) == ["T00", "T01"]
    assert set(ranked.ticker) != set(rank_baseline(eligible).ticker)


def test_execution_parity_and_scorecard_is_two_pass_deterministic():
    prices, universe, day = _inputs()
    _, baseline, reversal_rows = build_ranked_arms(prices, universe, signal_from=day, signal_to=day)
    direct_trades, direct_equity = simulate_portfolio(baseline, prices, BASELINE_ARM)
    trades, equity, _, _ = execute_arms(baseline, reversal_rows, prices)
    pd.testing.assert_frame_equal(direct_trades, trades); pd.testing.assert_frame_equal(direct_equity, equity)
    assert canonical_scorecard_bytes(build_scorecard(prices, universe, provenance={"safe": 1})) == canonical_scorecard_bytes(build_scorecard(prices, universe, provenance={"safe": 1}))


def test_mechanism_diagnostics_formulas():
    base_rows = pd.DataFrame({"signal_date": [pd.Timestamp("2020-01-01")] * 2, "ticker": ["A", "B"]})
    rev_rows = pd.DataFrame({"signal_date": [pd.Timestamp("2020-01-01")] * 2, "ticker": ["B", "C"]})
    top = top20_mechanism(base_rows, rev_rows)
    assert top["intersection_count"] == 1 and top["overall_jaccard"] == pytest.approx(1 / 3)
    columns = {"evaluation_year": [2020, 2020], "signal_date": [pd.Timestamp("2020-01-01")] * 2, "ticker": ["A", "B"], "status": ["FILLED", "FILLED"], "realized_net_profit_yen": [10., -5.]}
    filled = fill_mechanism(pd.DataFrame(columns), pd.DataFrame({**columns, "ticker": ["B", "C"], "realized_net_profit_yen": [-5., 20.]}))
    assert filled["common_fills"] == 1 and filled["baseline_only_pnl"] == 10 and filled["reversal_only_pnl"] == 20


def test_precompute_normalizes_once_per_ticker(monkeypatch):
    prices, universe, day = _inputs(); eligible = generate_eligible_candidates(prices, universe, signal_from=day, signal_to=day)
    calls = []; original = reversal._as_frame
    def counted(frame): calls.append(id(frame)); return original(frame)
    monkeypatch.setattr(reversal, "_as_frame", counted)
    attach_reversal_scores(eligible, prices, universe)
    assert len(calls) == len(prices) == len(set(calls))


def test_output_guards_no_2026_and_no_network_client(tmp_path: Path):
    prices, universe, _ = _inputs()
    with pytest.raises(ValueError, match="OUTPUT_INSIDE_REPOSITORY_PROHIBITED"): write_scorecard(Path.cwd() / "reversal-output", b"{}", Path.cwd())
    output = tmp_path / "output"; write_scorecard(output, b"{}", Path.cwd()); assert [path.name for path in output.iterdir()] == ["scorecard.json"]
    with pytest.raises(ValueError, match="OUTPUT_DIRECTORY_NONEMPTY_OR_FILE"): write_scorecard(output, b"{}", Path.cwd())
    with pytest.raises(ValueError, match="EVALUATION_SIGNAL_OUTSIDE_2020_2025"): generate_eligible_candidates(prices, universe, signal_from="2026-01-01", signal_to="2026-01-01")
    source = inspect.getsource(reversal); assert "requests" not in source and "urllib" not in source
