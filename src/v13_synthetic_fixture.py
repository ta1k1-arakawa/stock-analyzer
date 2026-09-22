"""Deterministic, visibly artificial inputs for V13 mechanics tests."""
from __future__ import annotations

from datetime import date, timedelta
import math
from .v13_feasibility import FEATURES, SessionCalendar, select_universe, sha256_text


def synthetic_calendar() -> SessionCalendar:
    days = []; cursor = date(2018, 1, 1)
    while len(days) < 800:
        if cursor.weekday() < 5: days.append(cursor)
        cursor += timedelta(days=1)
    return SessionCalendar(tuple(days))


def synthetic_manifest() -> dict:
    eligible = [f"{n:04d}" for n in range(1000, 1700)]
    excluded = ["1001", "1011", "1021"]
    seed = "V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON|f9c38ad771710ffd157ac4fad0da15185db82707"
    selected = select_universe(eligible, excluded, seed)
    return {"seed": seed, "eligible": eligible, "excluded": excluded, "selected": selected,
            "selected_sha256": sha256_text("|".join(selected))}


def active_rows() -> list[dict]:
    """Synthetic labels/features spanning Jan/Feb 2020 and no market identity."""
    calendar = synthetic_calendar(); codes = synthetic_manifest()["selected"][:12]
    sectors = ["SYNTHETIC_ALPHA"] * 6 + ["SYNTHETIC_BETA"] * 6
    rows = []
    for idx, signal in enumerate(calendar.sessions):
        if signal.year not in (2019, 2020): continue
        exit_day = calendar.plus(signal, 3)
        if exit_day is None: continue
        for j, (code, sector) in enumerate(zip(codes, sectors)):
            row = {"code": code, "sector": sector, "signal": signal, "exit": exit_day}
            for k, feature in enumerate(FEATURES):
                # deliberately non-empirical periodic values with cross-sectional variance
                row[feature] = math.sin((idx + 1) * (k + 3) * .017 + j * .31) + j * .012
            row["target"] = math.sin(idx * .09 + j * .53) * 1.2 + (j - 5) * .03
            rows.append(row)
    return rows


def stage_a_rows() -> list[dict]:
    rows = []
    for i in range(12):
        row = {"code": f"{3000+i:04d}", "sector": "SYNTHETIC_ALPHA" if i < 6 else "SYNTHETIC_BETA"}
        for k, feature in enumerate(FEATURES[:19]): row[feature] = math.sin(i * .7 + k * .11) + i * .04
        rows.append(row)
    return rows
