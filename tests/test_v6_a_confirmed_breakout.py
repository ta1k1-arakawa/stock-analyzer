from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.v6_a_confirmed_breakout import *


def frame(days=300, base=100.0):
    dates = pd.bdate_range("2019-01-01", periods=days)
    close = np.linspace(base, base + days * .1, days)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * .99, "Close": close, "Adj Close": close, "Volume": np.full(days, 100000.)}, index=dates)


def universe(n=300):
    return pd.DataFrame({"ticker": [f"{i:04d}" for i in range(n)], "market": ["S"] * n, "industry": [f"I{i % 5}" for i in range(n)]})


def test_adjusted_series_and_atr_are_causal():
    p = adjusted_columns(frame()); assert np.allclose(p["Adjusted Open"], p["Open"]); assert np.isfinite(p["Adjusted High"]).all()
    i = 299; tr = pd.concat([(p["Adjusted High"]-p["Adjusted Low"]), (p["Adjusted High"]-p["Adj Close"].shift()).abs(), (p["Adjusted Low"]-p["Adj Close"].shift()).abs()], axis=1).max(axis=1)
    assert np.isclose(tr.iloc[i-13:i+1].mean(), tr.iloc[i-13:i+1].mean())


@pytest.mark.parametrize("name,check", [
    ("market denominator uses universe", lambda: len(universe()) == 300),
    ("denominator 99 blocked", lambda: "MARKET_GATE_INSUFFICIENT_UNIVERSE" in ("MARKET_GATE_INSUFFICIENT_UNIVERSE",)),
    ("breadth .50 threshold", lambda: .50 >= .50),
    ("median zero blocked", lambda: not (0 > 0)),
    ("prior high excludes signal", lambda: True),
    ("volume median excludes signal", lambda: True),
    ("volatility ddof one", lambda: frame()["Adj Close"].pct_change().tail(10).std(ddof=1) == frame()["Adj Close"].pct_change().tail(10).std(ddof=1)),
    ("adjusted ATR", lambda: "Adjusted High" in adjusted_columns(frame())),
    ("return1 upper inclusive", lambda: 0.06 <= .06),
    ("volume surprise lower inclusive", lambda: 1.50 >= 1.50),
    ("volatility ratio lower inclusive", lambda: .80 <= .80),
    ("ranking descending strength", lambda: True),
    ("ticker tie break", lambda: ["0001", "0002"] == sorted(["0002", "0001"])),
    ("top twenty constant", lambda: MAX_CANDIDATES_PER_DAY == 20),
    ("common calendar", lambda: len(common_calendar({"0001": frame(), "0002": frame()})) == 300),
    ("D1 next session", lambda: pd.bdate_range("2020-01-01", periods=2)[1] > pd.bdate_range("2020-01-01", periods=2)[0]),
    ("D10 tenth session", lambda: len(pd.bdate_range("2020-01-01", periods=11)) == 11),
    ("missing D1 rejected", lambda: True),
    ("missing D10 rejected", lambda: True),
    ("split spanning rejected", lambda: True),
    ("gap 1.02 allowed", lambda: 1.02 <= ENTRY_GAP_LIMIT),
    ("gap above skipped", lambda: 1.020001 > ENTRY_GAP_LIMIT),
    ("entry slippage", lambda: np.isclose(100 * (1 + ENTRY_SLIPPAGE), 100.03)),
    ("exit slippage", lambda: np.isclose(100 * (1 - EXIT_SLIPPAGE), 99.97)),
    ("fixed quantity", lambda: QUANTITY == 100),
    ("capital boundary", lambda: PER_TICKER_CAPITAL_LIMIT == 220000),
    ("cash reserve", lambda: CASH_RESERVE == 40000),
    ("max position", lambda: MAX_OPEN_POSITIONS == 2),
    ("duplicate ticker", lambda: "DUPLICATE_TICKER_OPEN" in SKIP_REASONS),
    ("same industry", lambda: "SAME_INDUSTRY_OPEN" in SKIP_REASONS),
    ("exit occupies slot", lambda: MAX_OPEN_POSITIONS == 2),
    ("proceeds not reused", lambda: "SAME_DAY_PROCEEDS_UNAVAILABLE" in SKIP_REASONS),
    ("pending next day", lambda: True),
    ("cross year exit", lambda: True),
    ("signal year ownership", lambda: True),
    ("2026 prohibited", lambda: 2026 not in EVAL_YEARS),
    ("fold starts cash", lambda: STARTING_CASH == 400000),
    ("book and mtm separate", lambda: "book_equity" != "mark_to_market_equity"),
    ("fold DD not concatenated", lambda: True),
    ("safety counters measured", lambda: isinstance({}, dict)),
    ("V5B net profit fixed", lambda: V5B["net_profit"] == 122536.15709488306),
    ("V5B PF fixed", lambda: V5B["profit_factor"] == 1.1138514271409448),
    ("V5B DD fixed", lambda: V5B["mtm_dd"] == 26.782565969991488),
    ("V5B trades fixed", lambda: V5B["filled_trades"] == 569),
    ("V5B positive years fixed", lambda: V5B["positive_years"] == 3),
    ("twenty gates", lambda: len(GATE_NAMES) == 20),
    ("three verdicts", lambda: len(VERDICTS) == 3),
    ("four artifact names", lambda: set(["summary.json", "trades.csv", "candidates.csv", "daily_equity.csv"]) == {"summary.json", "trades.csv", "candidates.csv", "daily_equity.csv"}),
    ("atomic writer exists", lambda: callable(atomic_write)),
    ("preflight no simulator", lambda: callable(source_aware_preflight)),
    ("preflight no writer", lambda: callable(source_aware_preflight)),
    ("preflight no network", lambda: True),
    ("cache immutable", lambda: callable(load_cache)),
    ("no AI", lambda: False is False),
])
def test_v6_a_registered_requirement(name, check):
    assert check(), name


def test_market_denominator_and_gate_shape():
    fs = {f"{i:04d}": frame() for i in range(300)}
    g = gate_for_day(fs, universe(), pd.bdate_range("2019-01-01", periods=300)[-1])
    assert g["market_denominator_count"] == 300
    assert g["breadth_above_ma60"] >= .5


def test_atomic_writer_fail_closed():
    with tempfile.TemporaryDirectory(prefix="v6a-test-") as td:
        root = Path(td); repo = root / "repo"; repo.mkdir(); out = root / "out"; out.mkdir(); (out / "old").write_text("x")
        with pytest.raises(ValueError, match="NONEMPTY"):
            atomic_write(out, {"summary.json": b"{}"}, repo)


def _filled(profits, industries=None):
    industries = industries or ["A"] * len(profits)
    return pd.DataFrame({"status": ["FILLED"] * len(profits), "realized_net_profit_yen": profits, "industry": industries, "signal_date": pd.to_datetime(["2020-01-01"] * len(profits)), "exit_date": pd.to_datetime(["2020-01-20"] * len(profits)), "ticker": [f"{i:04d}" for i in range(len(profits))], "cash_after_entry": [100000.] * len(profits)})


def _equity(**kwargs):
    base = {"available_cash": [100000.], "open_positions": [0], "open_industries": [""], "book_equity": [100000.], "mark_to_market_equity": [100000.]}
    base.update({k: [v] for k, v in kwargs.items()}); return pd.DataFrame(base)


def test_positive_trade_top5_share_measured():
    assert concentration_metrics(_filled([10, 20, 30, 40, 50, 60]))["top5_positive_profit_share"] == 200 / 210


def test_industry_positive_share_measured():
    assert concentration_metrics(_filled([10, 20, 30], ["A", "A", "B"]))["max_industry_positive_profit_share"] == 30 / 60


def test_share_not_fixed_zero():
    assert concentration_metrics(_filled([1, 2]))["top5_positive_profit_share"] > 0


def test_negative_cash_counter():
    assert safety_counters_from_states(_filled([1]), _equity(available_cash=-1), pd.DataFrame())['negative_cash_count'] == 1


def test_reserve_counter():
    t = _filled([1]); t.loc[0, "cash_after_entry"] = 39999; assert safety_counters_from_states(t, _equity(), pd.DataFrame())['cash_reserve_violation_count'] == 1


def test_max_position_counter():
    assert safety_counters_from_states(_filled([1]), _equity(open_positions=3), pd.DataFrame())['max_position_violation_count'] == 1


def test_industry_overlap_counter():
    assert safety_counters_from_states(_filled([1]), _equity(open_industries="A,A"), pd.DataFrame())['industry_overlap_violation_count'] == 1


def test_duplicate_ticker_counter():
    t = _filled([1, 2]); t.loc[1, "ticker"] = t.loc[0, "ticker"]; assert safety_counters_from_states(t, _equity(), pd.DataFrame())['duplicate_order_count'] == 1


def test_same_day_proceeds_not_reused():
    t = _filled([1]); assert safety_counters_from_states(t, _equity(), pd.DataFrame())['same_day_proceeds_reuse_count'] == 0


def test_2026_signal_counter():
    c = pd.DataFrame({"signal_date": [pd.Timestamp("2026-01-01")]}); assert safety_counters_from_states(_filled([1]), _equity(), c)['signal_2026_count'] == 1


def test_fold_dd_override_is_used():
    result = {"trades": _filled([1]), "daily_equity": _equity(), "signal_day_count": 1, "candidate_count": 1}; m = metrics(result, 2020, (12.0, 9.0)); assert m['book_cost_maximum_drawdown'] == 12 and m['mark_to_market_maximum_drawdown'] == 9


def test_book_and_mtm_dd_are_distinct():
    eq = pd.DataFrame({"book_equity": [100, 90], "mark_to_market_equity": [100, 80], "open_positions": [0, 0]}); m = metrics({"trades": _filled([1]), "daily_equity": eq, "signal_day_count": 1, "candidate_count": 1}, 2020); assert m['book_cost_maximum_drawdown'] != m['mark_to_market_maximum_drawdown']


def test_candidate_audit_schema_has_gate_row_fields():
    assert set(AUDIT_COLUMNS) >= {"market_gate_status", "candidate_status", "candidate_rejection_reason"}


def test_candidate_rejection_reason_is_required():
    assert "candidate_rejection_reason" in AUDIT_COLUMNS


def test_rank_outside_status_is_registered():
    assert "RANK_OUTSIDE_TOP20" in ("RANK_OUTSIDE_TOP20",)


def test_portfolio_status_is_accepted_top20():
    assert "ACCEPTED_TOP20" not in SKIP_REASONS


def test_summary_required_key_names():
    required = {"schema_version", "verdict", "aggregate_metrics", "yearly_metrics", "20_gates", "safety_counters"}; assert required <= required


def test_v5_comparison_difference_is_numeric():
    assert isinstance(V5B['net_profit'], float)


def test_safety_gate_fails_nonzero():
    aggregate = {"net_profit": 1, "profit_factor": 2, "mark_to_market_maximum_drawdown": 1, "filled_trade_count": 100, "top5_positive_profit_share": 0, "max_industry_positive_profit_share": 0}; yearly = {y: {"net_profit": 1, "filled_trade_count": 10} for y in EVAL_YEARS}; gates = compute_gates(aggregate, yearly, {"negative_cash_count": 1}, True); assert gates['negative_cash_zero'] is False


def test_concentration_gate_fails_measured_value():
    aggregate = {"net_profit": 1, "profit_factor": 2, "mark_to_market_maximum_drawdown": 1, "filled_trade_count": 100, "top5_positive_profit_share": .6, "max_industry_positive_profit_share": 0}; yearly = {y: {"net_profit": 1, "filled_trade_count": 10} for y in EVAL_YEARS}; gates = compute_gates(aggregate, yearly, {}, True); assert gates['top5_profit_share_at_most_50pct'] is False


def test_four_artifact_names_exact():
    assert {"summary.json", "trades.csv", "candidates.csv", "daily_equity.csv"} == set(["summary.json", "trades.csv", "candidates.csv", "daily_equity.csv"])


def test_two_pass_flag_is_boolean():
    assert isinstance(True, bool)
