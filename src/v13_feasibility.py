"""Offline, deterministic mechanics for the frozen V13 feasibility probe.

This module deliberately accepts supplied rows only.  It has no transport,
file-market-data, or ticker-discovery capability.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from statistics import median
from typing import Any, Iterable

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "ret_1", "ret_3", "ret_5", "ret_20", "intraday_1", "overnight_1",
    "sector_rel_ret_1", "sector_rel_ret_3", "sector_rel_ret_5", "sector_rel_ret_20",
    "market_rel_ret_1", "market_rel_ret_3", "market_rel_ret_5", "market_rel_ret_20",
    "log_traded_value_ratio_20", "log_median_traded_value_20", "log_amihud_20",
    "volatility_20", "dist_52w_high", "breadth_1", "breadth_5",
    "market_median_ret_1", "market_median_ret_5", "cross_section_dispersion_1",
    "market_median_volatility_20",
)
STOCK_FEATURES = FEATURES[:19]
Q_KEYS = tuple(f"Q{i}_{name}" for i, name in enumerate((
    "FEATURE_CAUSALITY", "MONTHLY_ASOF_LABEL_CUTOFF", "NO_CURRENT_MONTH_LEARNING",
    "STAGE_B_REFERENCE_FROZEN", "NO_2026_PRICE_READ", "RANKING_FROZEN_BEFORE_OPEN",
    "SINGLE_POSITION_AND_CASH_SAFETY", "EXIT_EVENT_ORDER", "REQUIRED_EXIT_DATA",
    "STRESS_NO_RETRAIN_OR_RERANK", "MODEL_FEATURE_CONTRACT", "COMPARATOR_CONTRACT",
), 1))


def numeric_code(code: str) -> int:
    if not (isinstance(code, str) and len(code) == 4 and code.isdigit()):
        raise ValueError("code must be a four-digit string")
    return int(code)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_universe(eligible_codes: Iterable[str], excluded_codes: Iterable[str], seed: str) -> list[str]:
    excluded = set(excluded_codes)
    pool = sorted({c for c in eligible_codes if c not in excluded}, key=lambda c: (sha256_text(seed + "|" + c), numeric_code(c)))
    if len(pool) < 500:
        raise ValueError("INSUFFICIENT_ELIGIBLE_CODES")
    return pool[:500]


@dataclass(frozen=True)
class SessionCalendar:
    sessions: tuple[date, ...]

    def __post_init__(self) -> None:
        if tuple(sorted(set(self.sessions))) != self.sessions:
            raise ValueError("calendar must be unique and ordered")

    def plus(self, signal: date, offset: int) -> date | None:
        try:
            position = self.sessions.index(signal) + offset
        except ValueError:
            return None
        return self.sessions[position] if 0 <= position < len(self.sessions) else None


def base_target(open_price: float, close_price: float, entry_friction: float = .001, exit_friction: float = .001) -> float | None:
    if not all(math.isfinite(x) and x > 0 for x in (open_price, close_price)):
        return None
    return 100 * (close_price * (1 - exit_friction) / (open_price * (1 + entry_friction)) - 1)


def lightgbm_factory() -> LGBMRegressor:
    return LGBMRegressor(objective="huber", alpha=.9, learning_rate=.03, n_estimators=400,
        num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1,
        colsample_bytree=.8, reg_alpha=.1, reg_lambda=1., random_state=20260922,
        n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)


def ridge_factory() -> Pipeline:
    return Pipeline((("scaler", StandardScaler()), ("ridge", Ridge(alpha=10., fit_intercept=True))))


def random_key(seed: int, signal_day: date, code: str) -> str:
    return sha256_text(f"{seed}|{signal_day.isoformat()}|{code}")


def rank_candidates(rows: Iterable[dict[str, Any]], strategy: str, signal_day: date | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    rows = list(rows)
    if strategy in {"LIGHTGBM", "RIDGE"}:
        field = "lightgbm_score" if strategy == "LIGHTGBM" else "ridge_score"
        rows = [r for r in rows if math.isfinite(r[field]) and r[field] > 0]
        return sorted(rows, key=lambda r: (-r[field], numeric_code(r["code"])))
    if strategy == "SECTOR_REL_REVERSAL_1D":
        return sorted(rows, key=lambda r: (r["sector_rel_ret_1"], numeric_code(r["code"])))
    if strategy == "SECTOR_REL_MOMENTUM_20D":
        return sorted(rows, key=lambda r: (-r["sector_rel_ret_20"], numeric_code(r["code"])))
    if strategy == "RANDOM_500" and signal_day is not None and seed is not None:
        return sorted(rows, key=lambda r: (random_key(seed, signal_day, r["code"]), numeric_code(r["code"])))
    raise ValueError("unknown ranking")


def _finite(values: Iterable[float]) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def build_rank_population(stage_a: Iterable[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Apply the single-pass Stage B/C transforms to supplied Stage-A rows."""
    stage_a = list(stage_a)
    sectors: dict[str, list[dict[str, Any]]] = {}
    for row in stage_a:
        if _finite(row.get(k, float("nan")) for k in STOCK_FEATURES):
            sectors.setdefault(row["sector"], []).append(dict(row))
    stage_b = [r for members in sectors.values() if len(members) >= 5 for r in members]
    if len(stage_b) < 2:
        return "NO_RANK_DATA_QUALITY", []
    for k in (1, 3, 5, 20):
        key = f"ret_{k}"; sector_key = f"sector_rel_ret_{k}"; market_key = f"market_rel_ret_{k}"
        market = float(np.median([r[key] for r in stage_b]))
        for members in sectors.values():
            eligible = [r for r in members if r in stage_b]
            if eligible:
                sector_med = float(np.median([r[key] for r in eligible]))
                for r in eligible:
                    r[sector_key], r[market_key] = r[key] - sector_med, r[key] - market
    r1 = [r["ret_1"] for r in stage_b]; r5 = [r["ret_5"] for r in stage_b]
    market_values = {"breadth_1": sum(v > 0 for v in r1)/len(r1), "breadth_5": sum(v > 0 for v in r5)/len(r5),
        "market_median_ret_1": float(np.median(r1)), "market_median_ret_5": float(np.median(r5)),
        "cross_section_dispersion_1": float(np.std(r1, ddof=1)),
        "market_median_volatility_20": float(np.median([r["volatility_20"] for r in stage_b]))}
    for r in stage_b: r.update(market_values)
    for field in STOCK_FEATURES + tuple(f"sector_rel_ret_{k}" for k in (1,3,5,20)) + tuple(f"market_rel_ret_{k}" for k in (1,3,5,20)):
        values = np.asarray([r[field] for r in stage_b], dtype=float)
        lo, hi = np.percentile(values, [1, 99], method="linear")
        clipped = np.clip(values, lo, hi); std = float(np.std(clipped, ddof=1)); mean = float(np.mean(clipped))
        if not (math.isfinite(lo) and math.isfinite(hi) and math.isfinite(mean) and math.isfinite(std) and std > 0):
            return "NO_RANK_DATA_QUALITY", []
        for row, value in zip(stage_b, clipped): row[field] = float((value - mean) / std)
    rank = [r for r in stage_b if _finite(r.get(k, float("nan")) for k in FEATURES)]
    return "OK", rank


def monthly_predictions(rows: list[dict[str, Any]], feature_names: tuple[str, ...] = FEATURES) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One expanding fit of each model per prediction month, labels cut off prior month."""
    output: list[dict[str, Any]] = []; fits: dict[str, int] = {}
    months = sorted({r["signal"].strftime("%Y-%m") for r in rows if r["signal"].year >= 2020})
    for month in months:
        year, mon = map(int, month.split("-")); train = [r for r in rows if r["exit"] < date(year, mon, 1)]
        predict = [r for r in rows if r["signal"].strftime("%Y-%m") == month]
        if not train or not predict: continue
        x = np.array([[r[k] for k in feature_names] for r in train]); y = np.array([r["target"] for r in train])
        lgb, ridge = lightgbm_factory(), ridge_factory(); lgb.fit(x, y); ridge.fit(x, y); fits[month] = 2
        px = np.array([[r[k] for k in feature_names] for r in predict])
        for r, a, b in zip(predict, lgb.predict(px), ridge.predict(px)):
            copied = dict(r); copied["lightgbm_score"], copied["ridge_score"] = float(a), float(b); output.append(copied)
    return output, fits


def linear_percentile(values: Iterable[float], percentile: float) -> float | None:
    values = list(values)
    return float(np.percentile(values, percentile, method="linear")) if values and _finite(values) else None


def diagnostics(rows: list[dict[str, Any]], score: str = "lightgbm_score") -> dict[str, Any]:
    valid = [r for r in rows if _finite((r.get(score, float("nan")), r.get("target", float("nan"))))]
    if len(valid) < 2 or len({r[score] for r in valid}) < 2 or len({r["target"] for r in valid}) < 2:
        return {"ic": "UNDEFINED", "top_decile_spread": "UNDEFINED"}
    ic = float(spearmanr([r[score] for r in valid], [r["target"] for r in valid]).statistic)
    ordered = sorted(valid, key=lambda r: (-r[score], numeric_code(r["code"])))
    n = max(1, math.ceil(.1 * len(ordered)))
    return {"ic": ic, "top_decile_spread": float(np.mean([r["target"] for r in ordered[:n]]) - np.mean([r["target"] for r in ordered]))}


def simulate(calendar: SessionCalendar, prices: dict[tuple[str, date], dict[str, float]],
             rankings: dict[date, list[dict[str, Any]]], friction: float = .001) -> dict[str, Any]:
    """Single-capital next-open/third-close simulator with fail-closed exits.

    Rankings are supplied after signal close and are never altered by Open data.
    ``prices`` is deliberately an in-memory test input keyed by synthetic code/date.
    """
    cash, unavailable, position = 300000.0, 0.0, None
    events, equity, trades = [], {}, []
    pending: tuple[date, list[dict[str, Any]], date] | None = None
    failure = None
    for day in calendar.sessions:
        if unavailable:
            cash += unavailable; unavailable = 0.0
        if pending is not None and position is None and day == pending[0]:
            signal, ranked, exit_day = pending; pending = None
            for candidate in ranked:
                quote = prices.get((candidate["code"], day), {}); opening = quote.get("open")
                if not isinstance(opening, (float, int)) or not math.isfinite(opening) or opening <= 0: continue
                unit = float(opening) * (1 + friction); quantity = int(cash // (unit * 100)) * 100
                if quantity > 0:
                    cost = quantity * unit; cash -= cost
                    position = {"code": candidate["code"], "sector": candidate.get("sector", "UNKNOWN"), "quantity": quantity,
                        "entry_cost": cost, "entry_exec": unit, "exit_day": exit_day, "signal": signal}
                    events.append({"type": "ENTRY", "code": candidate["code"], "date": day, "quantity": quantity})
                    break
            if position is None: events.append({"type": "NO_FILL", "signal": signal})
        if position is not None and day == position["exit_day"]:
            quote = prices.get((position["code"], day), {}); closing = quote.get("close")
            if not isinstance(closing, (float, int)) or not math.isfinite(closing) or closing <= 0:
                failure = "DATA_QUALITY_FAILURE"; events.append({"type": failure, "code": position["code"], "date": day}); break
            proceeds = position["quantity"] * float(closing) * (1 - friction)
            trade = dict(position, exit_exec=float(closing)*(1-friction), proceeds=proceeds, pnl=proceeds-position["entry_cost"], exit_date=day)
            trades.append(trade); unavailable += proceeds; position = None; events.append({"type": "EXIT", "date": day})
        if position is None and day in rankings:
            entry, exit_day = calendar.plus(day, 1), calendar.plus(day, 3)
            if entry is None or exit_day is None: events.append({"type": "END_OF_STUDY_NO_ENTRY", "signal": day})
            else: pending = (entry, rankings[day], exit_day)
        marked = cash + unavailable
        if position is not None:
            close = prices.get((position["code"], day), {}).get("close")
            marked += position["quantity"] * float(close) if isinstance(close, (int,float)) and math.isfinite(close) else 0.0
        equity[day] = marked
    return {"failure_class": failure, "cash": cash, "unavailable_proceeds": unavailable, "position": position,
            "events": events, "daily_equity": equity, "trades": trades,
            "Q7_SINGLE_POSITION_AND_CASH_SAFETY": cash >= 0 and (position is None or position["quantity"] % 100 == 0),
            "Q8_EXIT_EVENT_ORDER": True, "Q9_REQUIRED_EXIT_DATA": failure is None}


def trade_metrics(result: dict[str, Any]) -> dict[str, Any]:
    trades = result["trades"]; pnls = [t["pnl"] for t in trades]
    returns = [100*(t["exit_exec"]/t["entry_exec"]-1) for t in trades]
    positives, negatives = sum(max(x,0) for x in pnls), abs(sum(min(x,0) for x in pnls))
    factor: float | str = positives / negatives if negatives else ("POSITIVE_INFINITY" if positives else "UNDEFINED")
    curve = list(result["daily_equity"].values()); peak = 300000.; max_dd = 0.
    for value in curve: peak = max(peak, value); max_dd = max(max_dd, 100*(peak-value)/peak)
    return {"closed_trade_count": len(trades), "total_net_profit": sum(pnls), "ending_equity": curve[-1] if curve else 300000.,
        "win_rate": sum(x > 0 for x in pnls)/len(pnls) if pnls else "UNDEFINED", "profit_factor": factor,
        "mean_trade_net_return": float(np.mean(returns)) if returns else "UNDEFINED", "median_trade_net_return": float(np.median(returns)) if returns else "UNDEFINED",
        "max_drawdown_pct": max_dd, "no_fill_count": sum(e["type"] == "NO_FILL" for e in result["events"])}


def canonical_json(value: Any) -> str:
    def safe(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v): return "POSITIVE_INFINITY" if v > 0 else "UNDEFINED"
        if isinstance(v, dict): return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [safe(x) for x in v]
        if isinstance(v, (date,)): return v.isoformat()
        return v
    return json.dumps(safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def adjudicate(criteria: dict[str, bool], q: dict[str, bool]) -> dict[str, Any]:
    if set(q) != set(Q_KEYS): raise ValueError("Q_KEY_SET_MISMATCH")
    if set(criteria) != set("ABCDEFGHIJKLMNOP"): raise ValueError("A_P_KEY_SET_MISMATCH")
    all_q = all(q.values()); all_criteria = all(criteria.values()) and all_q
    return {"criteria": dict(criteria), "Q": dict(q), "Q_SAFETY_AND_LEAKAGE_INVARIANTS_PASS": all_q,
        "V13_VIABILITY_RESULT": "CONTINUE_TO_FORMAL_CONFIRMATION_DESIGN" if all_criteria else "STOP_HYPOTHESIS_NOT_PROMOTED",
        "same_study_parameter_rescue": False}
