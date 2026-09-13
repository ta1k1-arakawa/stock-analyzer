"""Dependency-minimal copy of the reviewed V4 Yahoo-chart parser path."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


DATE_TO = "2025-03-31"
PRICE_FROM = pd.Timestamp("2015-01-01")
PRICE_TO = pd.Timestamp("2019-12-31")


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing OHLCV columns: {missing}")
    result = frame.copy()
    result.index = pd.to_datetime(result.index).tz_localize(None).normalize()
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("duplicate or unordered OHLCV dates")
    if len(result) and result.index.max() > pd.Timestamp(DATE_TO):
        raise ValueError("PROHIBITED_POST_CUTOFF_DATA")
    for column in required:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    finite = result[["Open", "High", "Low", "Close", "Volume"]].replace([np.inf, -np.inf], np.nan)
    if finite[["Open", "High", "Low", "Close"]].isna().any().any():
        raise ValueError("non-finite OHLC price")
    invalid = (result["Low"] > result[["Open", "Close", "High"]].min(axis=1)) | (result["High"] < result[["Open", "Close", "Low"]].max(axis=1))
    if invalid.any() or (result[["Open", "High", "Low", "Close"]] <= 0).any().any() or (result["Volume"] < 0).any():
        raise ValueError("invalid OHLCV relationship")
    return result


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
