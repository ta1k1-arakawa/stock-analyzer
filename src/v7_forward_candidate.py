"""Causal, forward-only D0 candidate generation for V7.

The production path in this module never calls the historical V6 candidate
generator or its future-dependent feature helper.  It consumes only rows and
split events whose effective dates are at or before ``engine_day``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.v6_a_confirmed_breakout import adjusted_columns, gate_for_day


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
FEATURE_FIELDS = (
    "raw_close",
    "adj_close",
    "ma20",
    "ma60",
    "return1",
    "return20",
    "return60",
    "volatility10",
    "volatility60",
    "median_turnover60",
    "median_volume60",
    "prior_high20",
    "volume_surprise",
    "atr14",
    "atr14_percent",
    "breakout_strength_atr",
)


class V7CandidateBlocked(RuntimeError):
    """Fail-closed causal or calendar violation."""

    def __init__(
        self,
        reason: str,
        *,
        future_candidate_data_access_count: int = 0,
        future_split_access_count: int = 0,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.future_candidate_data_access_count = future_candidate_data_access_count
        self.future_split_access_count = future_split_access_count
        self.audit = {
            "reason": reason,
            "future_candidate_data_access_count": future_candidate_data_access_count,
            "future_split_access_count": future_split_access_count,
        }


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise V7CandidateBlocked(f"INVALID_{field.upper()}") from error
    if pd.isna(parsed):
        raise V7CandidateBlocked(f"INVALID_{field.upper()}")
    if parsed.tz is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()


def _iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return str(value)


def _json_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Mapping):
        return {str(key): _json_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _validate_commit(value: str) -> None:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise ValueError("COLLECTOR_COMMIT_INVALID")


def _universe_frame(universe: Any) -> pd.DataFrame:
    if isinstance(universe, pd.DataFrame):
        frame = universe.copy()
    else:
        values = list(universe)
        if values and isinstance(values[0], Mapping):
            frame = pd.DataFrame(values)
        else:
            frame = pd.DataFrame({"ticker": [str(value) for value in values]})
    if "ticker" not in frame.columns:
        raise ValueError("UNIVERSE_TICKER_COLUMN_MISSING")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    if frame["ticker"].duplicated().any() or frame["ticker"].eq("").any():
        raise ValueError("UNIVERSE_DUPLICATE_OR_EMPTY_TICKER")
    if "industry" not in frame.columns:
        frame["industry"] = ""
    frame["industry"] = frame["industry"].fillna("").astype(str)
    if "market" not in frame.columns:
        frame["market"] = ""
    return frame[["ticker", "market", "industry"]].sort_values("ticker").reset_index(drop=True)


def _canonical_frame(frame: pd.DataFrame, engine_day: pd.Timestamp) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("PRICE_FRAME_NOT_DATAFRAME")
    if frame.index.has_duplicates:
        raise ValueError("DUPLICATE_PRICE_DATE")
    dates = pd.to_datetime(frame.index)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_convert(None)
    dates = dates.normalize()
    if any(date > engine_day for date in dates):
        raise V7CandidateBlocked(
            "FUTURE_CANDIDATE_DATA_ACCESS",
            future_candidate_data_access_count=1,
        )
    copy = frame.copy()
    copy.index = dates
    return copy.sort_index()


def _canonical_splits(
    split_history: Mapping[str, Sequence[Any]] | None,
    engine_day: pd.Timestamp,
) -> dict[str, tuple[pd.Timestamp, ...]]:
    result: dict[str, tuple[pd.Timestamp, ...]] = {}
    for ticker, values in (split_history or {}).items():
        parsed = tuple(sorted({_timestamp(value, "split_date") for value in values}))
        if any(value > engine_day for value in parsed):
            raise V7CandidateBlocked("FUTURE_SPLIT_ACCESS", future_split_access_count=1)
        result[str(ticker).strip().upper()] = parsed
    return result


def _calendar_days(study_calendar: Sequence[Any], engine_day: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    days = tuple(_timestamp(value, "calendar_date") for value in study_calendar)
    if not days or tuple(sorted(set(days))) != days:
        raise V7CandidateBlocked("INVALID_STUDY_CALENDAR")
    try:
        position = days.index(engine_day)
    except ValueError as error:
        raise V7CandidateBlocked("ENGINE_DAY_NOT_IN_STUDY_CALENDAR") from error
    if position + 10 >= len(days):
        raise V7CandidateBlocked("STUDY_CALENDAR_D10_UNAVAILABLE")
    return days[position + 1 : position + 11]


def _feature_at_d0(frame: pd.DataFrame, engine_day: pd.Timestamp) -> dict[str, Any] | None:
    processed = adjusted_columns(frame)
    if engine_day not in processed.index:
        return None
    index = processed.index.get_loc(engine_day)
    if not isinstance(index, (int, np.integer)) or index < 252:
        return None
    ac = processed["Adj Close"]
    raw_close = processed["Close"]
    volume = processed["Volume"]
    prior = ac.iloc[index - 20 : index].max()
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_surprise = volume.iloc[index] / processed["__prevvol20"].iloc[index]
        atr14 = processed["__atr14"].iloc[index]
        atr14_percent = atr14 / ac.iloc[index]
        breakout_strength = (ac.iloc[index] / prior - 1) / atr14_percent
    values = {
        "return1": processed["__ret"].iloc[index],
        "return20": processed["__r20"].iloc[index],
        "return60": processed["__r60"].iloc[index],
        "ma20": processed["__ma20"].iloc[index],
        "ma60": processed["__ma60"].iloc[index],
        "volatility10": processed["__vol10"].iloc[index],
        "volatility60": processed["__vol60ret"].iloc[index],
        "median_turnover60": processed["__turn60"].iloc[index],
        "median_volume60": processed["__vol60"].iloc[index],
        "prior_high20": processed["__prior20"].iloc[index],
        "volume_surprise": volume_surprise,
        "atr14": atr14,
        "atr14_percent": atr14_percent,
        "raw_close": raw_close.iloc[index],
        "adj_close": ac.iloc[index],
        "breakout_strength_atr": breakout_strength,
    }
    return {key: float(value) if isinstance(value, (np.floating, np.integer)) else value for key, value in values.items()}


def _eligible(values: Mapping[str, Any]) -> tuple[bool, str]:
    finite = all(
        isinstance(values.get(key), (int, float, np.integer, np.floating))
        and math.isfinite(float(values[key]))
        for key in FEATURE_FIELDS
    )
    if not finite:
        return False, "HISTORY_OR_FEATURE_INVALID"
    checks = (
        ("LIQUIDITY", values["median_turnover60"] >= 100_000_000 and values["median_volume60"] >= 50_000),
        ("PRICE_LIMIT", values["raw_close"] * 100 <= 220_000),
        ("TREND", values["adj_close"] > values["ma60"] and values["ma20"] > values["ma60"] and values["return60"] > 0),
        ("VOLATILITY_CONTRACTION", values["volatility10"] <= 0.80 * values["volatility60"]),
        ("BREAKOUT", values["adj_close"] > values["prior_high20"] and 0 < values["return1"] <= 0.06),
        ("VOLUME_CONFIRMATION", values["volume_surprise"] >= 1.50),
        ("ATR_INVALID", values["atr14_percent"] > 0 and math.isfinite(values["breakout_strength_atr"])),
    )
    for name, passed in checks:
        if not passed:
            return False, name
    return True, ""


def _price_snapshot(frames: Mapping[str, pd.DataFrame], engine_day: pd.Timestamp) -> str:
    rows: list[dict[str, Any]] = []
    for ticker in sorted(frames):
        frame = frames[ticker]
        for day, row in frame.iterrows():
            parsed = _timestamp(day, "price_date")
            rows.append({
                "ticker": ticker,
                "trading_date": parsed.strftime("%Y-%m-%d"),
                **{field: row[field] if field in row else None for field in ("Open", "High", "Low", "Close", "Adj Close", "Volume")},
            })
    return _sha(rows)


def _audit_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(row[key]) for key in sorted(row)}


def generate_forward_candidates_for_day(
    frames: Mapping[str, pd.DataFrame],
    universe: Any,
    split_history: Mapping[str, Sequence[Any]] | None,
    study_calendar: Sequence[Any],
    engine_day: Any,
    collector_commit: str,
) -> dict[str, Any]:
    """Generate the immutable D0 candidate snapshot without future reads."""
    _validate_commit(collector_commit)
    day = _timestamp(engine_day, "engine_day")
    next_days = _calendar_days(study_calendar, day)
    d1, d10 = next_days[0], next_days[9]
    universe_frame = _universe_frame(universe)
    canonical_frames = {
        str(ticker).strip().upper(): _canonical_frame(frame, day)
        for ticker, frame in frames.items()
    }
    splits = _canonical_splits(split_history, day)
    price_hash = _price_snapshot(canonical_frames, day)
    gate = gate_for_day(canonical_frames, universe_frame, day)
    market_gate = {
        "engine_day": day,
        **gate,
    }
    audit: list[dict[str, Any]] = [{
        "arm": None,
        "engine_day": day,
        "ticker": None,
        "industry": None,
        "signal_year": day.year,
        "signal_date": day,
        **gate,
        "candidate_status": "MARKET_GATE_AUDIT",
        "candidate_rejection_reason": None,
        "rank": None,
        **{field: None for field in FEATURE_FIELDS},
        "entry_date": d1,
        "exit_date": d10,
        "collector_commit": collector_commit,
    }]
    candidate_rows: list[dict[str, Any]] = []
    if gate["market_gate_status"] == "MARKET_GATE_PASS":
        industry_by_ticker = dict(zip(universe_frame["ticker"], universe_frame["industry"]))
        for ticker in sorted(industry_by_ticker):
            values = _feature_at_d0(canonical_frames[ticker], day) if ticker in canonical_frames else None
            base = {
                "arm": None,
                "engine_day": day,
                "ticker": ticker,
                "industry": industry_by_ticker[ticker],
                "signal_year": day.year,
                "signal_date": day,
                **gate,
                "rank": None,
                "entry_date": d1,
                "exit_date": d10,
                "collector_commit": collector_commit,
            }
            if values is None:
                audit.append(_audit_row({**base, "candidate_status": "REJECTED", "candidate_rejection_reason": "D0_DATA_UNAVAILABLE", **{field: None for field in FEATURE_FIELDS}}))
                continue
            eligible, reason = _eligible(values)
            row = {**base, **values, "candidate_status": "CANDIDATE" if eligible else "REJECTED", "candidate_rejection_reason": reason or None}
            audit.append(_audit_row(row))
            if eligible:
                candidate_rows.append(row)
    candidate_rows.sort(key=lambda row: (
        -float(row["breakout_strength_atr"]),
        -float(row["volume_surprise"]),
        -float(row["return60"]),
        str(row["ticker"]),
    ))
    for rank, row in enumerate(candidate_rows, start=1):
        row["rank"] = rank
    accepted = []
    accepted_tickers = {row["ticker"] for row in candidate_rows[:20]}
    for row in candidate_rows:
        if row["ticker"] in accepted_tickers:
            row["candidate_status"] = "ACCEPTED_TOP20"
            accepted.append(_audit_row(row))
        else:
            for item in audit:
                if item.get("ticker") == row["ticker"]:
                    item["candidate_status"] = "RANK_OUTSIDE_TOP20"
                    item["candidate_rejection_reason"] = "RANK_OUTSIDE_TOP20"
                    item["rank"] = row["rank"]
                    break
    accepted.sort(key=lambda row: (int(row["rank"]), str(row["ticker"])))
    audit.sort(key=lambda row: (
        0 if row.get("ticker") is None else 1,
        str(row.get("ticker") or ""),
    ))
    accepted_hash = _sha(accepted)
    gate_hash = _sha(_audit_row(market_gate))
    return {
        "engine_day": day.strftime("%Y-%m-%d"),
        "market_gate": _audit_row(market_gate),
        "accepted_top20": accepted,
        "full_candidate_audit": audit,
        "candidate_snapshot_sha256": accepted_hash,
        "market_gate_snapshot_sha256": gate_hash,
        "price_snapshot_sha256": price_hash,
        "future_candidate_data_access_count": 0,
        "future_split_access_count": 0,
        "entry_attempt_date": d1.strftime("%Y-%m-%d"),
        "planned_exit_date": d10.strftime("%Y-%m-%d"),
    }


__all__ = [
    "V7CandidateBlocked",
    "generate_forward_candidates_for_day",
]
