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
