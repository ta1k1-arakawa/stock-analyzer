"""Pure bridge from validated feature seed rows to V6-compatible D0 frames."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


FRAME_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
SEED_FIELDS = (
    "ticker",
    "trading_date",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "adj_close",
    "raw_volume",
)


class V7SeedBridgeBlocked(ValueError):
    """Raised when seed/D0 rows violate the causal bridge contract."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _date(value: Any) -> pd.Timestamp:
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise V7SeedBridgeBlocked("INVALID_TRADING_DATE") from error
    if pd.isna(parsed):
        raise V7SeedBridgeBlocked("INVALID_TRADING_DATE")
    if parsed.tz is not None:
        parsed = parsed.tz_convert(None)
    return parsed.normalize()


def _rows_from_validated_seed(value: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        rows = value.get("canonical_rows")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise V7SeedBridgeBlocked("VALIDATED_SEED_ROWS_MISSING")
        return [dict(row) for row in rows]
    return [dict(row) for row in value]


def _validate_row(row: Mapping[str, Any], engine_day: pd.Timestamp) -> dict[str, Any]:
    missing = [field for field in SEED_FIELDS if field not in row]
    if missing:
        raise V7SeedBridgeBlocked("ROW_SCHEMA_MISSING:" + ",".join(missing))
    result = dict(row)
    ticker = str(result["ticker"]).strip().upper()
    if not ticker:
        raise V7SeedBridgeBlocked("EMPTY_TICKER")
    trading_date = _date(result["trading_date"])
    if trading_date > engine_day:
        raise V7SeedBridgeBlocked("FUTURE_BRIDGE_ROW")
    for field in ("raw_open", "raw_high", "raw_low", "raw_close", "adj_close"):
        value = result[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise V7SeedBridgeBlocked("NONFINITE_" + field.upper())
        if float(value) <= 0:
            raise V7SeedBridgeBlocked("NONPOSITIVE_" + field.upper())
    volume = result["raw_volume"]
    if not isinstance(volume, (int, float)) or isinstance(volume, bool) or not math.isfinite(float(volume)):
        raise V7SeedBridgeBlocked("NONFINITE_RAW_VOLUME")
    if float(volume) < 0:
        raise V7SeedBridgeBlocked("NEGATIVE_RAW_VOLUME")
    result["ticker"] = ticker
    result["trading_date"] = trading_date
    return result


def build_forward_frames_from_seed_and_d0(
    validated_seed_rows: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    d0_rows: Sequence[Mapping[str, Any]],
    engine_day: Any,
) -> dict[str, pd.DataFrame]:
    """Build deterministic six-column frames using only seed rows and D0."""
    day = _date(engine_day)
    seed_rows = [_validate_row(row, day) for row in _rows_from_validated_seed(validated_seed_rows)]
    if not isinstance(d0_rows, Sequence) or isinstance(d0_rows, (str, bytes)):
        raise V7SeedBridgeBlocked("D0_ROWS_NOT_SEQUENCE")
    forward_rows = [_validate_row(row, day) for row in d0_rows]

    combined = seed_rows + forward_rows
    seen: set[tuple[str, pd.Timestamp]] = set()
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in combined:
        key = (row["ticker"], row["trading_date"])
        if key in seen:
            raise V7SeedBridgeBlocked("DUPLICATE_TICKER_DATE")
        seen.add(key)
        if row["trading_date"] == day and row in seed_rows:
            raise V7SeedBridgeBlocked("SEED_ROW_ON_ENGINE_DAY")
        by_ticker.setdefault(row["ticker"], []).append(row)

    frames: dict[str, pd.DataFrame] = {}
    for ticker in sorted(by_ticker):
        rows = sorted(by_ticker[ticker], key=lambda row: row["trading_date"])
        prior = [row for row in rows if row["trading_date"] < day]
        d0 = [row for row in rows if row["trading_date"] == day]
        if len(d0) > 1:
            raise V7SeedBridgeBlocked("DUPLICATE_D0_ROW")
        selected = prior[-252:] + d0
        frame = pd.DataFrame(
            [
                {
                    "Open": row["raw_open"],
                    "High": row["raw_high"],
                    "Low": row["raw_low"],
                    "Close": row["raw_close"],
                    "Adj Close": row["adj_close"],
                    "Volume": row["raw_volume"],
                }
                for row in selected
            ],
            index=[row["trading_date"] for row in selected],
            columns=FRAME_COLUMNS,
        )
        frame.index = pd.DatetimeIndex(frame.index)
        frames[ticker] = frame.sort_index()
    return frames


__all__ = [
    "FRAME_COLUMNS",
    "V7SeedBridgeBlocked",
    "build_forward_frames_from_seed_and_d0",
]
