"""Deterministic, visibly artificial raw OHLCV for the offline V13 proof."""
from __future__ import annotations

from datetime import date, timedelta
import math

from .v13_feasibility import SessionCalendar, labeled_rows_from_raw, select_universe, sha256_text


def synthetic_calendar() -> SessionCalendar:
    days, cursor = [], date(2015, 1, 1)
    while cursor <= date(2025, 12, 31):
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return SessionCalendar(tuple(days))


def synthetic_manifest() -> dict:
    eligible = [f"{n:04d}" for n in range(1000, 1705)]
    excluded = ["1001", "1011", "1021"]
    seed = "V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON|f9c38ad771710ffd157ac4ad0da15185db82707"
    selected = select_universe(eligible, excluded, seed)
    return {"seed": seed, "eligible": eligible, "excluded": excluded, "selected": selected, "selected_sha256": sha256_text("|".join(selected))}


def synthetic_metadata() -> dict[str, str]:
    return {f"{1000 + i:04d}": ("SYNTHETIC_ALPHA" if i < 6 else "SYNTHETIC_BETA") for i in range(12)}


def synthetic_signal_dates() -> tuple[date, ...]:
    calendar = synthetic_calendar()
    return tuple(day for day in calendar.sessions if day.day <= 7 and day.weekday() < 5 and (day == calendar.first_session_of_month(day.year, day.month)) and day.year >= 2016)


def raw_ohlcv(variant: str = "success") -> dict[tuple[str, date], dict[str, float]]:
    out: dict[tuple[str, date], dict[str, float]] = {}
    calendar = synthetic_calendar()
    for i, day in enumerate(calendar.sessions):
        for j, code in enumerate(synthetic_metadata()):
            close = (850 + j * 31) * (1 + .00028 * i + .025 * math.sin(i * .071 + j * .43))
            opening = close * (1 + .003 * math.sin(i * .13 + j))
            out[code, day] = {"open": opening, "high": max(opening, close) * 1.01, "low": min(opening, close) * .99, "close": close, "volume": 220000 + j * 8000., "adj_open": opening, "adj_close": close}
    first_code, first_day = next(iter(out))
    if variant == "affordability":
        out[first_code, first_day]["close"] = 3001
    elif variant == "missing_open":
        out[first_code, first_day]["adj_open"] = float("nan")
    elif variant == "missing_exit":
        out[first_code, first_day]["close"] = float("nan")
    elif variant == "no_rank":
        for row in out.values():
            row["close"] = row["adj_close"] = 1000.
    return out


def stage_a_rows():
    from .v13_feasibility import stage_a_from_raw
    calendar, prices = synthetic_calendar(), raw_ohlcv()
    return stage_a_from_raw(calendar, prices, synthetic_metadata(), date(2020, 1, 2))


def active_rows():
    rows, _ = labeled_rows_from_raw(synthetic_calendar(), raw_ohlcv(), synthetic_metadata(), synthetic_signal_dates())
    return rows
