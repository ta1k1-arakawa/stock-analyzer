"""Phase 1 deterministic transformations for the V4 meta-label MVP.

This module deliberately has no HTTP client, model fitting, or portfolio
backtest entry point.  Market data is supplied as Yahoo chart payloads or
OHLCV DataFrames so the pre-registered data boundary can be tested offline.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.free_prototype import validate_ohlcv
from src.trade_simulator import simulate_execution

PRICE_FROM = pd.Timestamp("2015-01-01")
PRICE_TO = pd.Timestamp("2019-12-31")
SIGNAL_FROM = pd.Timestamp("2016-04-01")
SIGNAL_TO = pd.Timestamp("2019-12-31")
UNIVERSE_COUNT = 300
UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
FEATURE_COLUMNS = (
    "return_5d", "return_20d", "return_60d", "volatility_20d",
    "volume_ratio_5d_20d", "close_to_ma20", "close_to_ma60",
    "high_low_range_20d", "required_cash_ratio",
    "momentum_20d_percentile_rank", "relative_momentum_20d",
    "cross_section_median_return_20d", "cross_section_breadth_above_ma20",
    "cross_section_median_volatility_20d", "cross_section_eligible_count",
)


def load_fixed_universe(path: str | Path = "V4_UNIVERSE.csv") -> pd.DataFrame:
    """Read the frozen universe without sorting or otherwise changing its order."""
    path = Path(path)
    # The registered digest uses UTF-8/LF canonical bytes, so it is stable
    # across a Windows checkout with autocrlf enabled.
    canonical = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    if sha256(canonical).hexdigest() != UNIVERSE_CSV_SHA256:
        raise ValueError("V4_UNIVERSE_CSV_HASH_MISMATCH")
    universe = pd.read_csv(path, dtype={"ticker": str})
    required = ["ticker", "market", "industry"]
    if list(universe.columns) != required or len(universe) != UNIVERSE_COUNT:
        raise ValueError("V4_UNIVERSE_SCHEMA_OR_COUNT_MISMATCH")
    universe["ticker"] = universe["ticker"].str.strip().str.upper()
    if universe["ticker"].duplicated().any() or (universe["ticker"].str.fullmatch(r"[0-9A-Z]{4}") == False).any():
        raise ValueError("V4_UNIVERSE_TICKER_INVALID")
    ticker_hash = sha256(("\n".join(universe["ticker"]) + "\n").encode("utf-8")).hexdigest()
    if ticker_hash != TICKER_LIST_SHA256:
        raise ValueError("V4_UNIVERSE_TICKER_HASH_MISMATCH")
    return universe


def validate_v4_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Reuse V3 OHLCV validation and enforce V4's earlier, closed boundary."""
    result = validate_ohlcv(frame)
    if len(result) and (result.index.min() < PRICE_FROM or result.index.max() > PRICE_TO):
        raise ValueError("PROHIBITED_V4_PRICE_DATE")
    return result


def parse_v4_yahoo_chart(payload: Mapping[str, Any]) -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    """Yahoo-chart adapter equivalent to V3 parsing, parameterized for V4 dates."""
    chart = payload.get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Yahoo chart error: {chart.get('error')}")
    result = chart["result"][0]
    quote = (result.get("indicators", {}).get("quote") or [{}])[0]
    adjusted = (result.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
    timestamps = result.get("timestamp") or []
    if not timestamps or adjusted is None:
        raise ValueError("empty Yahoo chart response")
    index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
    raw = pd.DataFrame({"Open": quote.get("open"), "High": quote.get("high"), "Low": quote.get("low"),
                        "Close": quote.get("close"), "Adj Close": adjusted, "Volume": quote.get("volume")}, index=index)
    splits = {pd.to_datetime(int(item["date"]), unit="s", utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
              for item in (result.get("events", {}).get("splits", {}) or {}).values() if item.get("date") is not None}
    return validate_v4_ohlcv(raw), splits


def prepare_price_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Expose raw and adjusted OHLC, preserving raw Volume as in the V3 code."""
    raw = validate_v4_ohlcv(raw)
    factor = raw["Adj Close"] / raw["Close"]
    out = raw.copy()
    out["adjustment_factor"] = factor
    for column in ("Open", "High", "Low", "Close"):
        out[f"adjusted_{column.lower()}"] = raw[column] * factor
    return out


def _stock_features(price: pd.DataFrame) -> pd.DataFrame:
    adjusted_close = price["adjusted_close"]
    adjusted_high = price["adjusted_high"]
    adjusted_low = price["adjusted_low"]
    result = pd.DataFrame(index=price.index)
    result["return_5d"] = adjusted_close / adjusted_close.shift(5) - 1
    result["return_20d"] = adjusted_close / adjusted_close.shift(20) - 1
    result["return_60d"] = adjusted_close / adjusted_close.shift(60) - 1
    result["volatility_20d"] = adjusted_close.pct_change().rolling(20).std(ddof=0) * np.sqrt(252)
    result["volume_ratio_5d_20d"] = price["Volume"].rolling(5).mean() / price["Volume"].rolling(20).mean()
    ma20, ma60 = adjusted_close.rolling(20).mean(), adjusted_close.rolling(60).mean()
    result["close_to_ma20"] = adjusted_close / ma20 - 1
    result["close_to_ma60"] = adjusted_close / ma60 - 1
    result["high_low_range_20d"] = adjusted_high.rolling(20).max() / adjusted_low.rolling(20).min() - 1
    result["required_cash_ratio"] = price["Close"] * 100 / 300_000
    result["History_Count"] = np.arange(1, len(result) + 1)
    result["median_turnover_60d"] = (price["Close"] * price["Volume"]).rolling(60).median()
    result["median_volume_60d"] = price["Volume"].rolling(60).median()
    result["above_ma20"] = adjusted_close > ma20
    return result


def build_feature_frame(prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
    """Calculate all 15 causal features for supplied frozen-universe prices."""
    allowed = set(universe["ticker"])
    if set(prices) - allowed:
        raise ValueError("PRICE_TICKER_NOT_IN_UNIVERSE")
    parts = []
    industry = universe.set_index("ticker")["industry"].to_dict()
    market = universe.set_index("ticker")["market"].to_dict()
    for ticker in universe["ticker"]:
        if ticker not in prices:
            continue
        stock = _stock_features(prepare_price_frame(prices[ticker]))
        stock["ticker"], stock["industry"], stock["market"] = ticker, industry[ticker], market[ticker]
        stock["signal_date"] = stock.index
        parts.append(stock.reset_index(drop=True))
    if not parts:
        return pd.DataFrame(columns=["signal_date", "ticker", "industry", "market", *FEATURE_COLUMNS])
    all_rows = pd.concat(parts, ignore_index=True)
    # Cross-sectional eligibility intentionally uses only information available on the signal date.
    past_ok = ((all_rows["History_Count"] >= 252) & (all_rows["median_turnover_60d"] >= 100_000_000) &
               (all_rows["median_volume_60d"] >= 50_000) & (all_rows["required_cash_ratio"] <= 1))
    grouped = all_rows.groupby("signal_date", sort=False)
    all_rows["momentum_20d_percentile_rank"] = grouped["return_20d"].rank(pct=True, method="average")
    eligible_returns = all_rows["return_20d"].where(past_ok)
    all_rows["cross_section_median_return_20d"] = eligible_returns.groupby(all_rows["signal_date"]).transform("median")
    all_rows["relative_momentum_20d"] = all_rows["return_20d"] - all_rows["cross_section_median_return_20d"]
    all_rows["cross_section_breadth_above_ma20"] = all_rows["above_ma20"].where(past_ok).groupby(all_rows["signal_date"]).transform("mean")
    all_rows["cross_section_median_volatility_20d"] = all_rows["volatility_20d"].where(past_ok).groupby(all_rows["signal_date"]).transform("median")
    all_rows["cross_section_eligible_count"] = past_ok.astype(int).groupby(all_rows["signal_date"]).transform("sum")
    feature_values = all_rows.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    all_rows.loc[:, FEATURE_COLUMNS] = feature_values
    all_rows["eligible"] = past_ok & np.isfinite(feature_values.to_numpy(dtype=float)).all(axis=1)
    return all_rows.loc[(all_rows["signal_date"] >= SIGNAL_FROM) & (all_rows["signal_date"] <= SIGNAL_TO)].reset_index(drop=True)


def add_execution_labels(features: pd.DataFrame, prices: Mapping[str, pd.DataFrame], splits: Mapping[str, set[pd.Timestamp]]) -> pd.DataFrame:
    """Attach fixed two-day realized labels.  Split-spanning samples fail closed."""
    result = features.copy()
    for column in ("EntryDate", "ExitDate", "LabelConfirmedDate", "ExitReason"):
        result[column] = pd.NaT if column != "ExitReason" else None
    for column in ("EntryPrice", "ExitPrice", "realized_net_return_percent", "realized_net_profit_yen", "label"):
        result[column] = np.nan
    for ticker, index in result.groupby("ticker", sort=False).groups.items():
        raw = validate_v4_ohlcv(prices[ticker])
        positions = {date: pos for pos, date in enumerate(raw.index)}
        for row_index in index:
            signal_date = result.at[row_index, "signal_date"]
            pos = positions.get(signal_date)
            if pos is None:
                continue
            execution = simulate_execution(raw[["Open", "High", "Low", "Close", "Volume"]], pos, 2, 5.0, 0.03, 0.03, 0.10, 0.0)
            if execution is None:
                result.at[row_index, "eligible"] = False
                continue
            entry_date, exit_date = raw.index[execution.entry_index], raw.index[execution.exit_index]
            if any(entry_date <= split <= exit_date for split in splits.get(ticker, set())):
                result.at[row_index, "eligible"] = False
                continue
            stop = execution.entry_price * .95
            exit_row = raw.iloc[execution.exit_index]
            reason = "TIME" if execution.exit_reason == "TIME" else ("GAP_STOP" if float(exit_row["Open"]) <= stop else "STOP")
            result.loc[row_index, ["EntryDate", "ExitDate", "LabelConfirmedDate", "EntryPrice", "ExitPrice", "realized_net_return_percent"]] = [entry_date, exit_date, exit_date, execution.entry_price, execution.exit_price, execution.return_percent]
            result.at[row_index, "ExitReason"] = reason
            result.at[row_index, "realized_net_profit_yen"] = (execution.exit_price - execution.entry_price) * 100
            result.at[row_index, "label"] = int(execution.return_percent > 0)
    return result


def select_daily_candidates(labelled: pd.DataFrame) -> pd.DataFrame:
    """One deterministic baseline candidate per date; affordability never reranks it."""
    rows = []
    for date, group in labelled.groupby("signal_date", sort=True):
        eligible = group.loc[group["eligible"]].sort_values(["return_20d", "ticker"], ascending=[False, True], kind="mergesort")
        if eligible.empty:
            rows.append({"signal_date": date, "candidate_status": "NO_CANDIDATE"})
        else:
            chosen = eligible.iloc[0].to_dict()
            chosen["candidate_status"] = "CANDIDATE"
            chosen["purchase_status"] = "INSUFFICIENT_CASH" if float(chosen["EntryPrice"]) * 100 > 300_000 else "AFFORDABLE"
            rows.append(chosen)
    return pd.DataFrame(rows)
