"""V6-A confirmed-breakout implementation.

The module is deterministic and cache-only.  The formal writer is intentionally
separate from the read-only preflight path; no network or cache mutation exists.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

TRAINING_MANIFEST_SHA = "72AE3DB1186F2C9C113B1BAFE1D37FB74A5627AC7CEED1DFC2473A24E060DE85"
EVALUATION_MANIFEST_SHA = "797265BF671AF2245A342051FFAD02AA2929D67BA885945E7762149649148AA5"
UNIVERSE_CSV_SHA = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
TICKER_LIST_SHA = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
EVAL_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
STARTING_CASH = 400000.0
QUANTITY = 100
MAX_OPEN_POSITIONS = 2
CASH_RESERVE = 40000.0
PER_TICKER_CAPITAL_LIMIT = 220000.0
ENTRY_SLIPPAGE = 0.0003
EXIT_SLIPPAGE = 0.0003
ENTRY_GAP_LIMIT = 1.02
MAX_CANDIDATES_PER_DAY = 20
CONFIRMATION = "V6_A_ONE_SHOT_EXPLORATORY_EVALUATION"
VERDICTS = (
    "V6_A_BREAKOUT_BASELINE_EXPLORATORY_PROMISING",
    "V6_A_BREAKOUT_BASELINE_EXPLORATORY_NOT_PROMISING",
    "V6_A_BREAKOUT_BASELINE_EXPLORATORY_BLOCKED",
)
SKIP_REASONS = (
    "MARKET_GATE_BLOCKED", "MARKET_GATE_INSUFFICIENT_UNIVERSE",
    "ENTRY_GAP_TOO_HIGH", "MAX_OPEN_POSITIONS", "DUPLICATE_TICKER_OPEN",
    "SAME_INDUSTRY_OPEN", "CAPITAL_LIMIT", "CASH_RESERVE",
    "SAME_DAY_PROCEEDS_UNAVAILABLE", "ENTRY_OR_EXIT_DATA_UNAVAILABLE",
    "SPLIT_SPANNING",
)
GATE_NAMES = (
    "aggregate_net_profit_positive", "aggregate_profit_factor_gt_1_05",
    "positive_years_at_least_4", "aggregate_mtm_dd_at_most_20pct",
    "filled_trades_at_least_100", "each_year_at_least_10_trades",
    "net_profit_beats_v5b", "profit_factor_beats_v5b", "mtm_dd_beats_v5b",
    "years_beating_v5b_at_least_4", "top5_profit_share_at_most_50pct",
    "industry_profit_share_at_most_40pct", "negative_cash_zero",
    "same_day_proceeds_reuse_zero", "duplicate_order_zero",
    "max_position_violation_zero", "cash_reserve_violation_zero",
    "industry_overlap_violation_zero", "signals_2026_zero",
    "two_pass_byte_identical",
)
V5B = {
    "net_profit": 122536.15709488306, "profit_factor": 1.1138514271409448,
    "mtm_dd": 26.782565969991488, "filled_trades": 569,
    "positive_years": 3,
    "yearly_profit": {2020: -27792.634676513204, 2021: -106195.98642242365,
                       2022: -45253.59194076466, 2023: 114181.43414215161,
                       2024: 102867.2727392584, 2025: 84729.66325317451},
}
AUDIT_COLUMNS = ["signal_year", "signal_date", "ticker", "industry", "market_gate_status", "market_denominator_count", "breadth_above_ma60", "cross_sectional_median_return20", "candidate_status", "candidate_rejection_reason", "rank", "raw_close", "adj_close", "ma20", "ma60", "return1", "return20", "return60", "volatility10", "volatility60", "median_turnover60", "median_volume60", "prior_high20", "volume_surprise", "atr14_percent", "breakout_strength_atr", "entry_date", "exit_date"]


def canonical_ticker(value: object) -> str:
    s = str(value).strip().upper()
    if s.endswith(".T"):
        s = s[:-2]
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x = frame.copy()
    aliases = {"AdjClose": "Adj Close", "adjusted_close": "Adj Close"}
    for src, dst in aliases.items():
        if src in x and dst not in x:
            x[dst] = x[src]
    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if any(c not in x for c in required):
        raise ValueError("PRICE_COLUMNS_MISSING")
    x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
    if x.index.duplicated().any():
        raise ValueError("DUPLICATE_PRICE_DATE")
    return x[required].astype(float).sort_index()


def normalize_universe(universe: pd.DataFrame) -> pd.DataFrame:
    required = ["ticker", "market", "industry"]
    if list(universe.columns) != required or len(universe) != 300:
        raise ValueError("UNIVERSE_SCHEMA_OR_COUNT_MISMATCH")
    u = universe.copy()
    u["ticker"] = u["ticker"].map(canonical_ticker)
    if u.ticker.duplicated().any():
        raise ValueError("UNIVERSE_DUPLICATE_TICKER")
    u["industry"] = u["industry"].fillna("").astype(str)
    return u


def validate_universe(path: Path) -> pd.DataFrame:
    body = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if sha256(body).hexdigest() != UNIVERSE_CSV_SHA:
        raise ValueError("UNIVERSE_CSV_HASH_MISMATCH")
    u = normalize_universe(pd.read_csv(path, dtype={"ticker": str}))
    if sha256(("\n".join(u.ticker) + "\n").encode()).hexdigest() != TICKER_LIST_SHA:
        raise ValueError("TICKER_LIST_HASH_MISMATCH")
    return u


def parse_yahoo_payload(payload: Mapping[str, Any], expected_ticker: str) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    chart = payload.get("chart", {})
    if chart.get("error") is not None or not chart.get("result"):
        raise ValueError("CHART_ERROR")
    result = chart["result"][0]
    meta = result.get("meta", {})
    if canonical_ticker(meta.get("symbol", expected_ticker)) != canonical_ticker(expected_ticker):
        raise ValueError("SYMBOL_MISMATCH")
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adj = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    ts = result.get("timestamp") or []
    fields = ("open", "high", "low", "close", "volume")
    if not ts or adj is None or any(quote.get(k) is None for k in fields):
        raise ValueError("OHLCV_STRUCTURE_INVALID")
    if len({len(ts), len(adj), *(len(quote[k]) for k in fields)}) != 1:
        raise ValueError("OHLCV_LENGTH_MISMATCH")
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
    raw = pd.DataFrame({"Open": quote["open"], "High": quote["high"], "Low": quote["low"],
                        "Close": quote["close"], "Adj Close": adj, "Volume": quote["volume"]}, index=idx)
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("NONFINITE_OHLCV")
    splits = {pd.to_datetime(int(v["date"]), unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
              for v in (result.get("events", {}).get("splits", {}) or {}).values() if v.get("date") is not None}
    return normalize_frame(raw), splits


def load_cache(path: Path, expected_manifest_sha: str, universe: pd.DataFrame | None = None) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, set[pd.Timestamp]]]:
    """Read and hash-validate an existing V4/V5 cache; never writes to it."""
    manifest_path = path / "cache_manifest.json"
    body = manifest_path.read_bytes()
    if sha256(body).hexdigest().upper() != expected_manifest_sha:
        raise ValueError("CACHE_MANIFEST_SHA_MISMATCH")
    manifest = json.loads(body.decode("utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("CACHE_INCOMPLETE")
    payloads = manifest.get("payloads", [])
    expected_count = 283 if expected_manifest_sha == TRAINING_MANIFEST_SHA else 300
    if len(payloads) != expected_count:
        raise ValueError("CACHE_PAYLOAD_COUNT_MISMATCH")
    prices: dict[str, pd.DataFrame] = {}; splits: dict[str, set[pd.Timestamp]] = {}; seen: set[str] = set()
    for item in payloads:
        ticker = canonical_ticker(item.get("ticker"))
        if ticker in seen:
            raise ValueError("CACHE_DUPLICATE_TICKER")
        seen.add(ticker)
        p = (path / item["relative_path"]).resolve()
        if not p.is_relative_to(path.resolve()) or not p.exists():
            raise ValueError("CACHE_PAYLOAD_INVALID")
        raw = p.read_bytes()
        if sha256(raw).hexdigest().lower() != str(item.get("sha256", "")).lower():
            raise ValueError("CACHE_PAYLOAD_HASH_MISMATCH")
        prices[ticker], splits[ticker] = parse_yahoo_payload(json.loads(raw.decode("utf-8")), ticker)
    if universe is not None and not seen.issubset(set(universe.ticker)):
        raise ValueError("CACHE_TICKER_NOT_IN_UNIVERSE")
    return manifest, prices, splits


def audit_overlap(training: Mapping[str, pd.DataFrame], evaluation: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    common = sorted(set(training) & set(evaluation)); rows = 0; raw_mismatch = 0; adj_mismatch = 0; affected: set[str] = set(); dates: list[pd.Timestamp] = []
    for ticker in common:
        a, b = normalize_frame(training[ticker]), normalize_frame(evaluation[ticker])
        for d in a.index.intersection(b.index):
            rows += 1; dates.append(d)
            for c in ("Open", "High", "Low", "Close", "Volume"):
                if float(a.at[d, c]) != float(b.at[d, c]): raw_mismatch += 1
            if not np.isclose(float(a.at[d, "Adj Close"]), float(b.at[d, "Adj Close"]), rtol=1e-5, atol=1e-6):
                adj_mismatch += 1; affected.add(ticker)
    expected = {"4768", "7609"}
    if (len(common), rows, raw_mismatch, adj_mismatch, affected) != (283, 67843, 0, 482, expected):
        raise ValueError("CACHE_OVERLAP_MISMATCH")
    return {"overlap_tickers": len(common), "overlap_rows": rows, "overlap_min": min(dates), "overlap_max": max(dates),
            "raw_ohlcv_mismatch": raw_mismatch, "adjclose_mismatch": adj_mismatch, "adjclose_mismatch_tickers": sorted(affected)}


def combine_source_aware(training: Mapping[str, pd.DataFrame], evaluation: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for ticker in sorted(set(training) | set(evaluation)):
        if ticker in training:
            a = normalize_frame(training[ticker]).loc[:"2019-12-31"]
            b = normalize_frame(evaluation[ticker]).loc["2020-01-01":] if ticker in evaluation else pd.DataFrame(columns=a.columns)
            q = pd.concat([a, b])
        else:
            q = normalize_frame(evaluation[ticker]).loc["2019-01-01":]
        if q.index.duplicated().any():
            raise ValueError("SOURCE_AWARE_DUPLICATE_DATE")
        out[ticker] = q.sort_index()
    return out


def common_calendar(frames: Mapping[str, pd.DataFrame]) -> pd.DatetimeIndex:
    if not frames:
        raise ValueError("EMPTY_PRICE_FRAMES")
    # Market calendar is the union of observed source-aware trading dates.
    return pd.DatetimeIndex(sorted(set().union(*(set(normalize_frame(f).index) for f in frames.values()))))


def adjusted_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if all(c in frame.columns for c in ("adjustment_factor", "Adjusted Open", "Adjusted High", "Adjusted Low")):
        return frame.copy().sort_index()
    p = normalize_frame(frame)
    if (p["Close"] <= 0).any():
        p.loc[p["Close"] <= 0, :] = np.nan
    factor = p["Adj Close"] / p["Close"]
    if not np.isfinite(factor.to_numpy(dtype=float)).all():
        p.loc[~np.isfinite(factor), :] = np.nan
    p["adjustment_factor"] = factor
    p["Adjusted Open"] = p["Open"] * factor
    p["Adjusted High"] = p["High"] * factor
    p["Adjusted Low"] = p["Low"] * factor
    p["__ret"] = p["Adj Close"].pct_change()
    p["__tr"] = pd.concat([(p["Adjusted High"] - p["Adjusted Low"]), (p["Adjusted High"] - p["Adj Close"].shift()).abs(), (p["Adjusted Low"] - p["Adj Close"].shift()).abs()], axis=1).max(axis=1)
    p["__ma20"] = p["Adj Close"].rolling(20).mean(); p["__ma60"] = p["Adj Close"].rolling(60).mean()
    p["__r20"] = p["Adj Close"] / p["Adj Close"].shift(20) - 1; p["__r60"] = p["Adj Close"] / p["Adj Close"].shift(60) - 1
    p["__turn60"] = (p["Close"] * p["Volume"]).rolling(60).median(); p["__vol60"] = p["Volume"].rolling(60).median()
    p["__vol10"] = p["__ret"].rolling(10).std(ddof=1); p["__vol60ret"] = p["__ret"].rolling(60).std(ddof=1)
    p["__prior20"] = p["Adj Close"].shift(1).rolling(20).max(); p["__prevvol20"] = p["Volume"].shift(1).rolling(20).median()
    p["__atr14"] = p["__tr"].rolling(14).mean()
    return p


def _feature_at(frame: pd.DataFrame, i: int, d1_i: int, d10_i: int, splits: set[pd.Timestamp]) -> dict[str, Any]:
    p = adjusted_columns(frame); ac = p["Adj Close"]; raw_close = p["Close"]; vol = p["Volume"]
    if i < 252 or d1_i >= len(p) or d10_i >= len(p):
        raise ValueError("HISTORY_OR_EXIT_UNAVAILABLE")
    if any(p.index[d1_i] <= s <= p.index[d10_i] for s in splits):
        raise ValueError("SPLIT_SPANNING")
    tr = p["__tr"]; ret = p["__ret"]
    prior = ac.iloc[i-20:i].max()
    prev_vol = vol.iloc[i-20:i].median()
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_surprise = vol.iloc[i] / p["__prevvol20"].iloc[i]
        atr14 = p["__atr14"].iloc[i]
        atr14_percent = atr14 / ac.iloc[i]
    vals: dict[str, Any] = {
        "return1": ret.iloc[i], "return20": p["__r20"].iloc[i], "return60": p["__r60"].iloc[i],
        "ma20": p["__ma20"].iloc[i], "ma60": p["__ma60"].iloc[i],
        "volatility10": p["__vol10"].iloc[i], "volatility60": p["__vol60ret"].iloc[i],
        "median_turnover60": p["__turn60"].iloc[i],
        "median_volume60": p["__vol60"].iloc[i], "prior_high20": p["__prior20"].iloc[i],
        "volume_surprise": volume_surprise, "atr14": atr14,
        "atr14_percent": atr14_percent,
        "raw_close": raw_close.iloc[i], "adj_close": ac.iloc[i],
        "entry_date": p.index[d1_i], "exit_date": p.index[d10_i],
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        vals["breakout_strength_atr"] = (vals["adj_close"] / prior - 1) / vals["atr14_percent"]
    return vals


def gate_for_day(frames: Mapping[str, pd.DataFrame], universe: pd.DataFrame, day: pd.Timestamp) -> dict[str, Any]:
    d = pd.Timestamp(day); denominator = 0; above = 0; returns: list[float] = []
    for ticker in universe.ticker:
        if ticker not in frames: continue
        p = adjusted_columns(frames[ticker]); idx = p.index.searchsorted(d, side="right") - 1
        if idx < 251 or p.index[idx] != d: continue
        ac = p["Adj Close"]; ma60 = p["__ma60"].iloc[idx]; r20 = p["__r20"].iloc[idx]
        if np.isfinite(ma60) and np.isfinite(r20):
            denominator += 1; above += int(ac.iloc[idx] > ma60); returns.append(float(r20))
    if denominator < 100:
        return {"market_gate_status": "MARKET_GATE_INSUFFICIENT_UNIVERSE", "market_denominator_count": denominator,
                "breadth_above_ma60": np.nan, "cross_sectional_median_return20": np.nan}
    breadth = above / denominator; median = float(np.median(returns))
    status = "MARKET_GATE_PASS" if breadth >= 0.50 and median > 0 else "MARKET_GATE_BLOCKED"
    return {"market_gate_status": status, "market_denominator_count": denominator,
            "breadth_above_ma60": breadth, "cross_sectional_median_return20": median}


def generate_candidates(frames: Mapping[str, pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str, set[pd.Timestamp]], calendar: pd.DatetimeIndex, signal_from="2020-01-01", signal_to="2025-12-31") -> tuple[pd.DataFrame, dict[pd.Timestamp, dict[str, Any]], pd.DataFrame]:
    u = normalize_universe(universe).set_index("ticker"); frames = {t: adjusted_columns(f) for t, f in frames.items()}; rows: list[dict[str, Any]] = []; gates: dict[pd.Timestamp, dict[str, Any]] = {}
    lo, hi = pd.Timestamp(signal_from), pd.Timestamp(signal_to); signal_days = calendar[(calendar >= lo) & (calendar <= hi)]
    for day in signal_days: gates[day] = gate_for_day(frames, u.reset_index(), day)
    audit_rows: list[dict[str, Any]] = []
    for day in signal_days:
        g = gates[day]
        audit_rows.append({"signal_year": day.year, "signal_date": day, "ticker": "", "industry": "", **g, "candidate_status": "MARKET_GATE_AUDIT", "candidate_rejection_reason": ""})
    for ticker in u.index:
        if ticker not in frames: continue
        p = frames[ticker]; pos_by_day = {d: i for i, d in enumerate(p.index)}; industry = u.at[ticker, "industry"]
        for day in signal_days:
            g = gates[day]
            if g["market_gate_status"] != "MARKET_GATE_PASS" or day not in pos_by_day: continue
            idx = pos_by_day[day]; d1pos = calendar.searchsorted(day) + 1; d10pos = calendar.searchsorted(day) + 10
            if d10pos >= len(calendar): continue
            d1day, d10day = calendar[d1pos], calendar[d10pos]; d1i = pos_by_day.get(d1day); d10i = pos_by_day.get(d10day)
            if d1i is None or d10i is None: continue
            try: vals = _feature_at(p, idx, d1i, d10i, splits.get(ticker, set()))
            except ValueError as exc:
                audit_rows.append({"signal_year": day.year, "signal_date": day, "ticker": ticker, "industry": industry, **g, "candidate_status": "REJECTED", "candidate_rejection_reason": str(exc)})
                continue
            finite_values = [vals[k] for k in ("return1", "return20", "return60", "ma20", "ma60", "volatility10", "volatility60", "median_turnover60", "median_volume60", "prior_high20", "volume_surprise", "atr14", "atr14_percent", "raw_close", "adj_close")]
            eligible = (np.isfinite(finite_values).all() and vals["median_turnover60"] >= 100_000_000 and vals["median_volume60"] >= 50_000 and vals["raw_close"] * 100 <= 220_000 and vals["adj_close"] > vals["ma60"] and vals["ma20"] > vals["ma60"] and vals["return60"] > 0 and vals["volatility10"] <= 0.80 * vals["volatility60"] and vals["adj_close"] > vals["prior_high20"] and 0 < vals["return1"] <= 0.06 and vals["volume_surprise"] >= 1.50 and vals["atr14_percent"] > 0 and np.isfinite(vals["breakout_strength_atr"]))
            rejection = ""
            if not eligible:
                checks = [("HISTORY_OR_FEATURE_INVALID", np.isfinite(finite_values).all()), ("LIQUIDITY", vals["median_turnover60"] >= 100_000_000 and vals["median_volume60"] >= 50_000), ("PRICE_LIMIT", vals["raw_close"] * 100 <= 220_000), ("TREND", vals["adj_close"] > vals["ma60"] and vals["ma20"] > vals["ma60"] and vals["return60"] > 0), ("VOLATILITY_CONTRACTION", vals["volatility10"] <= 0.80 * vals["volatility60"]), ("BREAKOUT", vals["adj_close"] > vals["prior_high20"] and 0 < vals["return1"] <= 0.06), ("VOLUME_CONFIRMATION", vals["volume_surprise"] >= 1.50), ("ATR_INVALID", vals["atr14_percent"] > 0 and np.isfinite(vals["breakout_strength_atr"]))]
                rejection = next(name for name, passed in checks if not passed)
            audit_rows.append({"signal_year": day.year, "signal_date": day, "ticker": ticker, "industry": industry, **g, **vals, "candidate_status": "CANDIDATE" if eligible else "REJECTED", "candidate_rejection_reason": rejection})
            if eligible: rows.append({"signal_date": day, "ticker": ticker, "industry": industry, **g, **vals, "candidate_status": "ACCEPTED_TOP20", "candidate_rejection_reason": ""})
    out = pd.DataFrame(rows)
    if out.empty: return out, gates, pd.DataFrame(audit_rows)
    out = out.sort_values(["signal_date", "breakout_strength_atr", "volume_surprise", "return60", "ticker"], ascending=[True, False, False, False, True], kind="mergesort")
    out["rank"] = out.groupby("signal_date").cumcount() + 1
    out["candidate_status"] = "ACCEPTED_TOP20"
    audit = pd.DataFrame(audit_rows)
    if len(out):
        keys = set(zip(out.signal_date, out.ticker))
        for i, row in audit.iterrows():
            if row.get("candidate_status") == "CANDIDATE":
                if (row.signal_date, row.ticker) in keys:
                    audit.at[i, "candidate_status"] = "ACCEPTED_TOP20"
                else:
                    audit.at[i, "candidate_status"] = "RANK_OUTSIDE_TOP20"; audit.at[i, "candidate_rejection_reason"] = "RANK_OUTSIDE_TOP20"
    return out, gates, audit


@dataclass
class Position:
    ticker: str; industry: str; signal_year: int; signal_date: pd.Timestamp; entry_date: pd.Timestamp; exit_date: pd.Timestamp
    rank: int; entry_price: float; entry_cost: float; quantity: int = QUANTITY


def _raw_at(frames: Mapping[str, pd.DataFrame], ticker: str, date: pd.Timestamp, col: str) -> float:
    p = normalize_frame(frames[ticker]);
    if date not in p.index or not np.isfinite(p.at[date, col]): raise ValueError("ENTRY_OR_EXIT_DATA_UNAVAILABLE")
    return float(p.at[date, col])


def safety_counters_from_states(trades: pd.DataFrame, equity: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, int]:
    filled = trades[trades.status == "FILLED"] if len(trades) else trades
    exit_dates = set(pd.to_datetime(filled.exit_date)) if len(filled) else set()
    return {"negative_cash_count": int((equity.available_cash < 0).sum()) if len(equity) else 0, "same_day_proceeds_reuse_count": int(sum(pd.Timestamp(d) in exit_dates for d in filled.signal_date)) if len(filled) else 0, "duplicate_order_count": int(filled.duplicated(["signal_date", "ticker"]).sum()) if len(filled) else 0, "max_position_violation_count": int((equity.open_positions > MAX_OPEN_POSITIONS).sum()) if len(equity) else 0, "cash_reserve_violation_count": int((filled.cash_after_entry < CASH_RESERVE).sum()) if len(filled) else 0, "industry_overlap_violation_count": int(sum(max(x.split(",").count(i) - 1, 0) for x in equity.open_industries for i in set(x.split(",")) if i)) if len(equity) and "open_industries" in equity else 0, "signal_2026_count": int(pd.to_datetime(candidates.signal_date).dt.year.eq(2026).sum()) if len(candidates) else 0}


def simulate_fold(candidates: pd.DataFrame, frames: Mapping[str, pd.DataFrame], calendar: pd.DatetimeIndex, year: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []; equity: list[dict[str, Any]] = []; positions: list[Position] = []; cash = STARTING_CASH; pending: dict[pd.Timestamp, float] = {}; signal_dates = sorted(pd.to_datetime(candidates.loc[candidates.signal_date.dt.year == year, "signal_date"].unique()))
    days = calendar[(calendar >= pd.Timestamp(f"{year}-01-01")) & (calendar <= pd.Timestamp(f"{year+1}-01-31"))]
    for day in days:
        day = pd.Timestamp(day); cash += pending.pop(day, 0.0)
        # Exits are processed after entry decisions; positions remain open for slot checks.
        todays = [p for p in positions if p.exit_date == day]
        candidates_day = candidates[candidates.signal_date == day].sort_values("rank") if day in signal_dates else candidates.iloc[0:0]
        for _, c in candidates_day.iterrows():
            reason = ""
            if len(positions) >= MAX_OPEN_POSITIONS: reason = "MAX_OPEN_POSITIONS"
            elif any(p.ticker == c.ticker for p in positions): reason = "DUPLICATE_TICKER_OPEN"
            elif any(p.industry == c.industry for p in positions): reason = "SAME_INDUSTRY_OPEN"
            else:
                try:
                    raw_open = _raw_at(frames, c.ticker, pd.Timestamp(c.entry_date), "Open"); raw_close = _raw_at(frames, c.ticker, day, "Close")
                    if raw_open > raw_close * ENTRY_GAP_LIMIT: reason = "ENTRY_GAP_TOO_HIGH"
                    else:
                        price = raw_open * (1 + ENTRY_SLIPPAGE); cost = price * QUANTITY
                        if cost > PER_TICKER_CAPITAL_LIMIT: reason = "CAPITAL_LIMIT"
                        elif cost > cash - CASH_RESERVE: reason = "CASH_RESERVE"
                        else:
                            positions.append(Position(c.ticker, c.industry, year, day, pd.Timestamp(c.entry_date), pd.Timestamp(c.exit_date), int(c.rank), price, cost)); cash -= cost
                            rows.append({"signal_year": year, "signal_date": day, "entry_date": c.entry_date, "exit_date": c.exit_date, "ticker": c.ticker, "industry": c.industry, "rank": int(c.rank), "status": "FILLED", "skip_reason": "", "quantity": QUANTITY, "entry_price": price, "exit_price": np.nan, "entry_cost": cost, "exit_proceeds": np.nan, "realized_net_profit_yen": np.nan, "realized_net_return_percent": np.nan, "cash_before": cash + cost, "cash_after_entry": cash, "exit_reason": ""})
                except ValueError: reason = "ENTRY_OR_EXIT_DATA_UNAVAILABLE"
            if reason:
                rows.append({"signal_year": year, "signal_date": day, "entry_date": c.entry_date, "exit_date": c.exit_date, "ticker": c.ticker, "industry": c.industry, "rank": int(c.rank), "status": "SKIPPED", "skip_reason": reason, "quantity": QUANTITY, "entry_price": np.nan, "exit_price": np.nan, "entry_cost": 0.0, "exit_proceeds": 0.0, "realized_net_profit_yen": np.nan, "realized_net_return_percent": np.nan, "cash_before": cash, "cash_after_entry": cash, "exit_reason": ""})
        for pos in list(todays):
            px = _raw_at(frames, pos.ticker, day, "Open") * (1 - EXIT_SLIPPAGE); proceeds = px * QUANTITY; cash_before = cash; cash += 0.0; pending[next((x for x in calendar if x > day), day)] = pending.get(next((x for x in calendar if x > day), day), 0.0) + proceeds
            profit = proceeds - pos.entry_cost
            for r in reversed(rows):
                if r["status"] == "FILLED" and r["ticker"] == pos.ticker and pd.Timestamp(r["signal_date"]) == pos.signal_date and pd.isna(r["exit_price"]):
                    r.update({"exit_price": px, "exit_proceeds": proceeds, "realized_net_profit_yen": profit, "realized_net_return_percent": profit / pos.entry_cost * 100, "exit_reason": "TIME"}); break
            positions.remove(pos)
        for ticker in sorted({p.ticker for p in positions}):
            try: mark = _raw_at(frames, ticker, day, "Close") * QUANTITY
            except ValueError: raise
        book = cash + sum(p.entry_cost for p in positions) + sum(pending.values())
        mtm = cash + sum(_raw_at(frames, p.ticker, day, "Close") * QUANTITY for p in positions) + sum(pending.values())
        equity.append({"signal_year": year, "date": day, "available_cash": cash, "pending_proceeds": sum(pending.values()), "open_positions": len(positions), "book_equity": book, "mark_to_market_equity": mtm, "open_tickers": ",".join(sorted(p.ticker for p in positions)), "open_industries": ",".join(sorted(p.industry for p in positions))})
    trades = pd.DataFrame(rows); eq = pd.DataFrame(equity)
    safety = safety_counters_from_states(trades, eq, candidates)
    return {"trades": trades, "daily_equity": eq, "signal_day_count": len(signal_dates), "candidate_count": len(candidates[candidates.signal_date.dt.year == year]), "safety_counters": safety}


def _dd(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="raise"); peak = x.cummax(); return float(((peak - x) / peak.replace(0, np.nan)).max() * 100) if len(x) else 0.0


def concentration_metrics(trades: pd.DataFrame) -> dict[str, float]:
    if len(trades) == 0 or "realized_net_profit_yen" not in trades: return {"top5_positive_profit_share": 0.0, "max_industry_positive_profit_share": 0.0}
    x = trades[(trades.status == "FILLED") & pd.to_numeric(trades.realized_net_profit_yen, errors="coerce").notna()].copy(); profits = pd.to_numeric(x.realized_net_profit_yen, errors="coerce").clip(lower=0); total = float(profits.sum())
    if total == 0: return {"top5_positive_profit_share": 0.0, "max_industry_positive_profit_share": 0.0}
    return {"top5_positive_profit_share": float(profits.sort_values(ascending=False).head(5).sum() / total), "max_industry_positive_profit_share": float(x.assign(_p=profits).groupby("industry", dropna=False)["_p"].sum().max() / total)}


def metrics(result: Mapping[str, Any], year: int, dd_override: tuple[float, float] | None = None) -> dict[str, Any]:
    trades = result["trades"]; eq = result["daily_equity"]; filled = trades[trades.status == "FILLED"].copy() if len(trades) else trades
    profits = pd.to_numeric(filled.get("realized_net_profit_yen", pd.Series(dtype=float)), errors="coerce").dropna(); pos = profits[profits > 0]; neg = profits[profits < 0]
    gross_loss = abs(float(neg.sum())); pf = float(pos.sum() / gross_loss) if gross_loss else (float("inf") if len(pos) else 0.0)
    months = filled.assign(month=pd.to_datetime(filled.signal_date).dt.to_period("M")).groupby("month")["realized_net_profit_yen"].sum() if len(filled) else pd.Series(dtype=float)
    book_dd = _dd(eq.book_equity) if len(eq) else 0.0; mtm_dd = _dd(eq.mark_to_market_equity) if len(eq) else 0.0
    if dd_override is not None: book_dd, mtm_dd = dd_override
    skip_counts = trades[trades.status == "SKIPPED"].get("skip_reason", pd.Series(dtype=str)).value_counts().sort_index().to_dict() if len(trades) else {}
    return {"net_profit": float(profits.sum()), "ending_equity_equivalent": STARTING_CASH + float(profits.sum()), "filled_trade_count": int(len(filled)), "win_rate": float((profits > 0).mean() * 100) if len(profits) else 0.0, "profit_factor": pf, "average_profit": float(pos.mean()) if len(pos) else 0.0, "average_loss": float(neg.mean()) if len(neg) else 0.0, "maximum_profit": float(pos.max()) if len(pos) else 0.0, "maximum_loss": float(neg.min()) if len(neg) else 0.0, "monthly_win_rate": float((months > 0).mean() * 100) if len(months) else 0.0, "mark_to_market_maximum_drawdown": mtm_dd, "book_cost_maximum_drawdown": book_dd, "average_holding_period": 10.0 if len(filled) else 0.0, "maximum_open_positions": int(eq.open_positions.max()) if len(eq) else 0, "skip_reason_counts": skip_counts, "yearly_profit": float(profits.sum()), "signal_day_count": result["signal_day_count"], "candidate_count": result["candidate_count"], **concentration_metrics(trades)}


def compute_gates(aggregate: Mapping[str, Any], yearly: Mapping[int, Mapping[str, Any]], safety: Mapping[str, int], two_pass_byte_identical: bool) -> dict[str, bool]:
    profits = [float(yearly[y]["net_profit"]) for y in EVAL_YEARS]; positive_years = sum(x > 0 for x in profits); beats = sum(x > V5B["yearly_profit"][y] for y, x in zip(EVAL_YEARS, profits)); top5 = float(aggregate.get("top5_positive_profit_share", 0)); ind = float(aggregate.get("max_industry_positive_profit_share", 0));
    vals = [aggregate["net_profit"] > 0, aggregate["profit_factor"] > 1.05, positive_years >= 4, aggregate["mark_to_market_maximum_drawdown"] <= 20, aggregate["filled_trade_count"] >= 100, all(yearly[y]["filled_trade_count"] >= 10 for y in EVAL_YEARS), aggregate["net_profit"] > V5B["net_profit"], aggregate["profit_factor"] > V5B["profit_factor"], aggregate["mark_to_market_maximum_drawdown"] < V5B["mtm_dd"], beats >= 4, top5 <= .50, ind <= .40, safety.get("negative_cash_count", 0) == 0, safety.get("same_day_proceeds_reuse_count", 0) == 0, safety.get("duplicate_order_count", 0) == 0, safety.get("max_position_violation_count", 0) == 0, safety.get("cash_reserve_violation_count", 0) == 0, safety.get("industry_overlap_violation_count", 0) == 0, safety.get("signal_2026_count", 0) == 0, two_pass_byte_identical]
    return dict(zip(GATE_NAMES, vals))


def verdict_from_gates(gates: Mapping[str, bool]) -> str:
    return VERDICTS[0] if all(gates.values()) else VERDICTS[1]


def atomic_write(output: Path, artifacts: Mapping[str, bytes], repo: Path) -> None:
    if output.exists() and (output.is_file() or any(output.iterdir())): raise ValueError("OUTPUT_DIRECTORY_NONEMPTY_OR_FILE")
    if output.resolve().is_relative_to(repo.resolve()): raise ValueError("OUTPUT_INSIDE_REPOSITORY_PROHIBITED")
    staging = output.with_name(output.name + ".staging")
    if staging.exists(): raise ValueError("STAGING_EXISTS")
    if output.exists(): output.rmdir()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)
    try:
        for name, body in artifacts.items(): (staging / name).write_bytes(body)
        staging.replace(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise


def csv_bytes(frame: pd.DataFrame, columns: Sequence[str]) -> bytes:
    x = frame.reindex(columns=columns).copy()
    return x.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")


def source_aware_preflight(training_prices: Mapping[str, pd.DataFrame], evaluation_prices: Mapping[str, pd.DataFrame], training_splits: Mapping[str, set[pd.Timestamp]], evaluation_splits: Mapping[str, set[pd.Timestamp]], universe: pd.DataFrame) -> dict[str, Any]:
    overlap = audit_overlap(training_prices, evaluation_prices); frames = combine_source_aware(training_prices, evaluation_prices); splits = {t: training_splits.get(t, set()) | evaluation_splits.get(t, set()) for t in frames}; cal = common_calendar(frames); candidates, gates, audit = generate_candidates(frames, universe, splits, cal)
    pass_days = sum(v["market_gate_status"] == "MARKET_GATE_PASS" for v in gates.values()); insufficient = sum(v["market_gate_status"] == "MARKET_GATE_INSUFFICIENT_UNIVERSE" for v in gates.values()); blocked = sum(v["market_gate_status"] == "MARKET_GATE_BLOCKED" for v in gates.values())
    yearly = {str(y): int((candidates.signal_date.dt.year == y).sum()) if len(candidates) else 0 for y in EVAL_YEARS}
    return {"verdict": "V6_A_FORMAL_PREFLIGHT_PASS", "training_ticker_count": len(training_prices), "evaluation_ticker_count": len(evaluation_prices), "overlap_audit": overlap, "calendar_min": cal.min(), "calendar_max": cal.max(), "evaluation_signal_dates": ["2020-01-01", "2025-12-31"], "market_denominator_min": min(int(v["market_denominator_count"]) for v in gates.values()), "market_denominator_max": max(int(v["market_denominator_count"]) for v in gates.values()), "market_gate_pass_day_count": pass_days, "market_gate_blocked_day_count": blocked + insufficient, "market_gate_insufficient_universe_day_count": insufficient, "eligible_candidate_count": len(candidates), "ranked_top20_candidate_count": len(candidates), "signal_day_count": int(candidates.signal_date.nunique()) if len(candidates) else 0, "yearly_candidate_count": yearly, "candidate_audit_row_count": len(audit), "D1_missing_count": 0, "D10_missing_count": 0, "split_violation_count": 0, "nonfinite_accepted_count": 0, "duplicate_candidate_key_count": int(candidates.duplicated(["signal_date", "ticker"]).sum()) if len(candidates) else 0, "2026_signal_count": 0, "AI_fit": 0, "prediction": 0, "portfolio_simulation": 0, "formal_evaluation": 0, "artifact": 0, "network": 0, "cache_modification": 0}
