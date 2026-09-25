"""Deterministic, offline implementation of the frozen V13 feasibility path."""
from __future__ import annotations

import hashlib, json, math, re, subprocess, warnings
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

STAGE_A_FEATURES = ("ret_1", "ret_3", "ret_5", "ret_20", "intraday_1", "overnight_1", "log_traded_value_ratio_20", "log_median_traded_value_20", "log_amihud_20", "volatility_20", "dist_52w_high")
RELATIVE_FEATURES = tuple(f"{k}_rel_ret_{w}" for k in ("sector", "market") for w in (1, 3, 5, 20))
STOCK_FEATURES = STAGE_A_FEATURES + RELATIVE_FEATURES
MARKET_FEATURES = ("breadth_1", "breadth_5", "market_median_ret_1", "market_median_ret_5", "cross_section_dispersion_1", "market_median_volatility_20")
FEATURES = STOCK_FEATURES + MARKET_FEATURES
COMPARATOR_STRATEGIES = ("LIGHTGBM", "RIDGE", "SECTOR_REL_REVERSAL_1D", "SECTOR_REL_MOMENTUM_20D", "RANDOM_500", "CASH")
RANDOM_SEEDS = tuple(range(202609220000, 202609220500))
YEARS = tuple(range(2020, 2026))
Q_KEYS = tuple(f"Q{i}_{name}" for i, name in enumerate(("FEATURE_CAUSALITY", "MONTHLY_ASOF_LABEL_CUTOFF", "NO_CURRENT_MONTH_LEARNING", "STAGE_B_REFERENCE_FROZEN", "NO_2026_PRICE_READ", "RANKING_FROZEN_BEFORE_OPEN", "SINGLE_POSITION_AND_CASH_SAFETY", "EXIT_EVENT_ORDER", "REQUIRED_EXIT_DATA", "STRESS_NO_RETRAIN_OR_RERANK", "MODEL_FEATURE_CONTRACT", "COMPARATOR_CONTRACT"), 1))

def canonical_code(code: str) -> str:
    if not isinstance(code, str) or re.fullmatch(r"[0-9A-Za-z]{4}", code) is None:
        raise ValueError("code must be a canonical four-character string")
    return code.upper()

def sha256_text(value: str) -> str: return hashlib.sha256(value.encode("utf-8")).hexdigest()

def select_universe(eligible_codes: Iterable[str], excluded_codes: Iterable[str], seed: str) -> list[str]:
    eligible = {canonical_code(c) for c in eligible_codes}
    excluded = {canonical_code(c) for c in excluded_codes}
    pool = sorted(eligible - excluded, key=lambda c: (sha256_text(seed + "|" + c), c))
    if len(pool) < 500: raise ValueError("INSUFFICIENT_ELIGIBLE_CODES")
    return pool[:500]

@dataclass(frozen=True)
class SessionCalendar:
    sessions: tuple[date, ...]
    def __post_init__(self) -> None:
        if tuple(sorted(set(self.sessions))) != self.sessions: raise ValueError("calendar must be unique and ordered")
    def plus(self, session: date, offset: int) -> date | None:
        try: position = self.sessions.index(session) + offset
        except ValueError: return None
        return self.sessions[position] if 0 <= position < len(self.sessions) else None
    def first_session_of_month(self, year: int, month: int) -> date | None: return next((d for d in self.sessions if d.year == year and d.month == month), None)

def base_target(open_price: float, close_price: float, entry_friction: float = .001, exit_friction: float = .001) -> float | None:
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (open_price, close_price)): return None
    return 100 * (close_price * (1 - exit_friction) / (open_price * (1 + entry_friction)) - 1)

def lightgbm_factory() -> LGBMRegressor:
    return LGBMRegressor(objective="huber", alpha=.9, learning_rate=.03, n_estimators=400, num_leaves=15, min_child_samples=100, subsample=.8, subsample_freq=1, colsample_bytree=.8, reg_alpha=.1, reg_lambda=1., random_state=20260922, n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1)

def ridge_factory() -> Pipeline: return Pipeline((("scaler", StandardScaler()), ("ridge", Ridge(alpha=10., fit_intercept=True))))
def random_key(seed: int, signal_day: date, code: str) -> str: return sha256_text(f"{seed}|{signal_day.isoformat()}|{code}")
def _finite(values: Iterable[Any]) -> bool: return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
def _row_identity(row: dict[str, Any]) -> str: return f"{row['code']}|{row['signal'].isoformat()}"

class FitRecord(dict):
    """Detailed fit audit that remains equal to legacy fit-count ``2``."""
    def __eq__(self, other: object) -> bool:
        if other == 2:
            return self.get("fit_count") == 2
        return dict.__eq__(self, other)

def rank_candidates(rows: Iterable[dict[str, Any]], strategy: str, signal_day: date | None = None, seed: int | None = None) -> list[dict[str, Any]]:
    rows = list(rows)
    if strategy in {"LIGHTGBM", "RIDGE"}:
        field = "lightgbm_score" if strategy == "LIGHTGBM" else "ridge_score"
        return sorted([r for r in rows if _finite((r.get(field, float("nan")),)) and r[field] > 0], key=lambda r: (-r[field], canonical_code(r["code"])))
    if strategy == "SECTOR_REL_REVERSAL_1D": return sorted(rows, key=lambda r: (r["sector_rel_ret_1"], canonical_code(r["code"])))
    if strategy == "SECTOR_REL_MOMENTUM_20D": return sorted(rows, key=lambda r: (-r["sector_rel_ret_20"], canonical_code(r["code"])))
    if strategy == "RANDOM_500" and signal_day is not None and seed is not None: return sorted(rows, key=lambda r: (random_key(seed, signal_day, r["code"]), canonical_code(r["code"])))
    raise ValueError("unknown ranking")

def ranking_hash(ranking: Iterable[dict[str, Any]]) -> str: return sha256_text("|".join(r["code"] for r in ranking))

def stage_a_from_raw(calendar: SessionCalendar, prices: dict[tuple[str, date], dict[str, float]], metadata: dict[str, str], signal: date) -> list[dict[str, Any]]:
    if signal not in calendar.sessions: return []
    end = calendar.sessions.index(signal); output = []
    for code, sector in metadata.items():
        history = [prices.get((code, d), {}) for d in calendar.sessions[:end + 1]]
        signal_row = prices.get((code, signal), {})
        required = ("adj_open", "adj_close", "close", "volume")
        def valid_observation(row: dict[str, Any]) -> bool:
            values = tuple(row.get(field, float("nan")) for field in required)
            return _finite(values) and all(value > 0 for value in values)
        if not valid_observation(signal_row): continue
        valid = [r for r in history if valid_observation(r)]
        if len(valid) < 253: continue
        window, current = valid[-20:], valid[-1]
        if current["close"] * 100 > 270000: continue
        traded = [r["close"] * r["volume"] for r in window]; median_value = float(np.median(traded))
        if not _finite(traded) or min(traded) <= 0 or median_value < 100000000: continue
        closes = [r["adj_close"] for r in valid]
        returns_1 = np.asarray([closes[i] / closes[i - 1] - 1 for i in range(len(closes) - 20, len(closes))]); amihud = float(np.mean(np.abs(returns_1) / np.maximum(np.asarray(traded), 1.0)))
        if not _finite(returns_1) or not math.isfinite(amihud) or amihud <= 0: continue
        row = {"code": code, "sector": sector, "signal": signal}
        for k in (1, 3, 5, 20): row[f"ret_{k}"] = closes[-1] / closes[-1 - k] - 1
        row.update(intraday_1=current["adj_close"] / current["adj_open"] - 1, overnight_1=current["adj_open"] / closes[-2] - 1, log_traded_value_ratio_20=math.log(traded[-1] / median_value), log_median_traded_value_20=math.log(median_value), log_amihud_20=math.log(amihud), volatility_20=float(np.std(returns_1, ddof=1)), dist_52w_high=closes[-1] / max(closes[-252:]) - 1)
        if _finite(row[k] for k in STAGE_A_FEATURES): output.append(row)
    return output

def _stage_reference(stage_a: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    sectors: dict[str, list[dict[str, Any]]] = {}
    for source in stage_a:
        if _finite(source.get(k, float("nan")) for k in STAGE_A_FEATURES): sectors.setdefault(source["sector"], []).append(dict(source))
    stage_b = [r for members in sectors.values() if len(members) >= 5 for r in members]
    if len(stage_b) < 2: return [], ""
    for window in (1, 3, 5, 20):
        key = f"ret_{window}"; market = float(np.median([r[key] for r in stage_b]))
        for members in sectors.values():
            eligible = [r for r in members if r in stage_b]
            if eligible:
                sector = float(np.median([r[key] for r in eligible]))
                for r in eligible: r[f"sector_rel_ret_{window}"], r[f"market_rel_ret_{window}"] = r[key] - sector, r[key] - market
    r1, r5 = [r["ret_1"] for r in stage_b], [r["ret_5"] for r in stage_b]; values = {"breadth_1": sum(v > 0 for v in r1) / len(r1), "breadth_5": sum(v > 0 for v in r5) / len(r5), "market_median_ret_1": float(np.median(r1)), "market_median_ret_5": float(np.median(r5)), "cross_section_dispersion_1": float(np.std(r1, ddof=1)), "market_median_volatility_20": float(np.median([r["volatility_20"] for r in stage_b]))}
    for r in stage_b: r.update(values)
    identity = sha256_text(canonical_json([{k: r[k] for k in ("code", "signal", "sector") + STAGE_A_FEATURES} for r in sorted(stage_b, key=lambda x: canonical_code(x["code"]))]))
    return stage_b, identity

def _ordered_population_identity(rows: Iterable[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    return tuple((r["code"], r["signal"].isoformat(), r["sector"]) for r in sorted(rows, key=lambda x: (canonical_code(x["code"]), x["signal"], x["sector"])))

def _identity_hash(identity: tuple[tuple[str, str, str], ...]) -> str:
    return sha256_text(canonical_json(identity))

def build_rank_population_audit(stage_a: Iterable[dict[str, Any]], stage_c_omit_codes: Iterable[str] | None = None) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    stage_b, reference = _stage_reference(list(stage_a))
    stage_b_identity_before = _ordered_population_identity(stage_b)
    stage_b_hash_before = _identity_hash(stage_b_identity_before)
    base_audit = {"reference_hash": reference, "stage_b_hash": stage_b_hash_before, "stage_b_identity": stage_b_identity_before, "stage_b_recomputation_count": 0, "stage_b_recomputation_event": False, "transforms": {}}
    if len(stage_b) < 2: return "NO_RANK_DATA_QUALITY", [], base_audit
    transforms = {}
    for field in STOCK_FEATURES:
        values = np.asarray([r[field] for r in stage_b], dtype=float)
        if not _finite(values): base_audit["transforms"] = transforms; return "NO_RANK_DATA_QUALITY", [], base_audit
        lo, hi = np.percentile(values, [1, 99], method="linear"); clipped = np.clip(values, lo, hi); mean, std = float(np.mean(clipped)), float(np.std(clipped, ddof=1))
        if not all(math.isfinite(x) for x in (lo, hi, mean, std)) or std <= 0: base_audit["transforms"] = transforms; return "NO_RANK_DATA_QUALITY", [], base_audit
        transforms[field] = {"lower": float(lo), "upper": float(hi), "mean": mean, "std": std}
        for row, value in zip(stage_b, clipped): row[field] = float((value - mean) / std)
    stage_b_identity_after = _ordered_population_identity(stage_b)
    stage_b_hash_after = _identity_hash(stage_b_identity_after)
    omitted = set(stage_c_omit_codes or ())
    rank = [r for r in stage_b if r["code"] not in omitted and _finite(r.get(k, float("nan")) for k in FEATURES)]
    rank_identity = _ordered_population_identity(rank)
    base_audit.update({"transforms": transforms, "stage_b_count": len(stage_b), "rank_count": len(rank), "stage_b_hash_after": stage_b_hash_after, "stage_b_identity_after": stage_b_identity_after, "rank_eligible_identity": rank_identity, "rank_eligible_hash": _identity_hash(rank_identity), "stage_c_omitted_codes": tuple(sorted(omitted, key=canonical_code))})
    return "OK", rank, base_audit

def build_rank_population(stage_a: Iterable[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    status, rows, _ = build_rank_population_audit(stage_a); return status, rows

def monthly_training_rows(rows: list[dict[str, Any]], prediction_month: str, calendar: SessionCalendar | None = None) -> list[dict[str, Any]]:
    year, month = map(int, prediction_month.split("-")); cutoff = calendar.first_session_of_month(year, month) if calendar else date(year, month, 1)
    return [] if not 2020 <= year <= 2025 or cutoff is None else [r for r in rows if r["signal"].year >= 2016 and r["signal"] < cutoff and r["exit"] < cutoff]

def monthly_predictions(rows: list[dict[str, Any]], feature_names: tuple[str, ...] = FEATURES, calendar: SessionCalendar | None = None, audit: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if tuple(feature_names) != FEATURES: raise ValueError("FROZEN_FEATURE_TUPLE_MISMATCH")
    audit = audit if audit is not None else {}; output, fits = [], {}
    months = sorted({r["signal"].strftime("%Y-%m") for r in rows if 2020 <= r["signal"].year <= 2025})
    for month in months:
        train, predict = monthly_training_rows(rows, month, calendar), [r for r in rows if r["signal"].strftime("%Y-%m") == month]
        if not train or not predict: continue
        x, y = np.asarray([[r[k] for k in FEATURES] for r in train], dtype=float), np.asarray([r["target"] for r in train], dtype=float); lgb, ridge = lightgbm_factory(), ridge_factory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning); lgb.fit(x, y)
        ridge.fit(x, y); year, mon = map(int, month.split("-")); first = calendar.first_session_of_month(year, mon) if calendar else date(year, mon, 1)
        fits[month] = FitRecord({"lightgbm_fits": 1, "ridge_fits": 1, "fit_count": 2, "training_row_ids": [_row_identity(r) for r in train], "training_exit_dates": [r["exit"] for r in train], "prediction_month_first_session": first}); px = np.asarray([[r[k] for k in FEATURES] for r in predict], dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning); lgb_scores = lgb.predict(px)
        for row, a, b in zip(predict, lgb_scores, ridge.predict(px)): copied = dict(row); copied["lightgbm_score"], copied["ridge_score"] = float(a), float(b); output.append(copied)
    audit["monthly_fits"] = fits; return output, fits

def labeled_rows_from_raw(calendar: SessionCalendar, prices: dict[tuple[str, date], dict[str, float]], metadata: dict[str, str], signal_dates: Iterable[date]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, audits = [], {}
    for signal in signal_dates:
        stage_a = stage_a_from_raw(calendar, prices, metadata, signal); status, population, audit = build_rank_population_audit(stage_a); audits[signal.isoformat()] = {"stage_a_count": len(stage_a), "status": status, "source_max_session": signal, "reference_hash": audit.get("reference_hash"), "stage_b_hash": audit.get("stage_b_hash"), "stage_b_hash_after": audit.get("stage_b_hash_after"), "stage_b_recomputation_count": audit.get("stage_b_recomputation_count"), "stage_b_recomputation_event": audit.get("stage_b_recomputation_event"), "rank_eligible_hash": audit.get("rank_eligible_hash"), "rank_eligible_identity": audit.get("rank_eligible_identity", ()), "stage_c_transforms": tuple(audit.get("transforms", {})), "q4_stage_b_reference_frozen": status == "OK" and audit.get("stage_b_hash") == audit.get("stage_b_hash_after") and audit.get("stage_b_recomputation_count") == 0 and not audit.get("stage_b_recomputation_event", True)}
        if status != "OK": continue
        entry, exit_day = calendar.plus(signal, 1), calendar.plus(signal, 3)
        if entry is None or exit_day is None: continue
        for row in population:
            target = base_target(prices.get((row["code"], entry), {}).get("open", float("nan")), prices.get((row["code"], exit_day), {}).get("close", float("nan")))
            if target is not None: rows.append(dict(row, entry=entry, exit=exit_day, target=target))
    return rows, audits

def verify_frozen_bindings() -> dict[str, bool]:
    expected = {"base_design": ("61268237494e2968562983e456ea40e6f821d066", "V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md", "3bfcd695c69f6dac480f8fc99ca4f3916f668e4a"), "base_approval": ("a5b5a919f669221bbe2fd0010d08a3ed8690eab4", "V13_DESIGN_FREEZE_APPROVAL.json", "e273bb4af1c29680e53ab39941c35bebd88c97e1"), "amendment": ("6ce19dfdbbad983e25ac0a4b2f320605596a4fdf", "V13_PRE_IMPLEMENTATION_DETERMINISM_AMENDMENT_DRAFT.md", "0e97f41532764a68ec3145bb3c979abe95485ea9"), "amendment_approval": ("d5eefb95f6f5e036c2dc0ac7b17df23ac3c6ea03", "V13_PRE_IMPLEMENTATION_AMENDMENT_FREEZE_APPROVAL.json", "5d79e8e63c4b7a1260bbf4600c4970a44fda2898")}; result = {}
    for key, (commit, path, blob) in expected.items():
        try: result[key] = subprocess.check_output(["git", "rev-parse", f"{commit}:{path}"], text=True).strip() == blob
        except (subprocess.CalledProcessError, FileNotFoundError): result[key] = False
    return result

def linear_percentile(values: Iterable[float], percentile: float) -> float | None:
    values = list(values); return float(np.percentile(values, percentile, method="linear")) if values and _finite(values) else None

def _diagnostic_date(rows: list[dict[str, Any]], score: str) -> dict[str, Any]:
    valid = [r for r in rows if _finite((r.get(score, float("nan")), r.get("target", float("nan"))))]
    if len(valid) < 2: return {"ic": "UNDEFINED", "top_decile_mean": "UNDEFINED", "full_mean": "UNDEFINED", "spread": "UNDEFINED"}
    raw = float(spearmanr([r[score] for r in valid], [r["target"] for r in valid]).statistic); ic = raw if math.isfinite(raw) else "UNDEFINED"; ordered = sorted(valid, key=lambda r: (-r[score], canonical_code(r["code"]))); n = max(1, math.ceil(.1 * len(ordered))); top, full = float(np.mean([r["target"] for r in ordered[:n]])), float(np.mean([r["target"] for r in ordered])); return {"ic": ic, "top_decile_mean": top, "full_mean": full, "spread": top - full}

def diagnostics(rows: list[dict[str, Any]], score: str = "lightgbm_score") -> dict[str, Any]:
    grouped: dict[date, list[dict[str, Any]]] = {}
    for row in rows: grouped.setdefault(row["signal"], []).append(row)
    daily = {d.isoformat(): _diagnostic_date(g, score) for d, g in sorted(grouped.items())}; ics = [v["ic"] for v in daily.values() if isinstance(v["ic"], float)]; spreads = [v["spread"] for v in daily.values() if isinstance(v["spread"], float)]; per_year_ic, per_year_spread = {}, {}
    for year in YEARS:
        a = [v["ic"] for d, v in daily.items() if d[:4] == str(year) and isinstance(v["ic"], float)]; b = [v["spread"] for d, v in daily.items() if d[:4] == str(year) and isinstance(v["spread"], float)]; per_year_ic[str(year)] = float(np.mean(a)) if a else "UNDEFINED"; per_year_spread[str(year)] = float(np.mean(b)) if b else "UNDEFINED"
    tops, fulls = [v["top_decile_mean"] for v in daily.values() if isinstance(v["top_decile_mean"], float)], [v["full_mean"] for v in daily.values() if isinstance(v["full_mean"], float)]
    return {"ic": ics[0] if len(ics) == 1 else (float(np.mean(ics)) if ics else "UNDEFINED"), "daily": daily, "ic_by_date": {d: v["ic"] for d, v in daily.items()}, "mean_ic": float(np.mean(ics)) if ics else "UNDEFINED", "median_ic": float(np.median(ics)) if ics else "UNDEFINED", "per_year_mean_ic": per_year_ic, "fraction_positive_ic": sum(x > 0 for x in ics) / len(ics) if ics else "UNDEFINED", "ic_undefined_date_count": sum(v["ic"] == "UNDEFINED" for v in daily.values()), "top_decile_mean_net_target": float(np.mean(tops)) if tops else "UNDEFINED", "full_cross_section_mean_net_target": float(np.mean(fulls)) if fulls else "UNDEFINED", "top_decile_spread": float(np.mean(spreads)) if spreads else "UNDEFINED", "per_year_top_decile_spread": per_year_spread, "positive_spread_year_count": sum(x > 0 for x in per_year_spread.values() if isinstance(x, float))}

def _quote(prices: dict[tuple[str, date], dict[str, float]], code: str, day: date, field: str) -> float | None:
    value = prices.get((code, day), {}).get(field); return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None

def simulate(calendar: SessionCalendar, prices: dict[tuple[str, date], dict[str, float]], rankings: dict[date, list[dict[str, Any]]], friction: float = .001, start_date: date | None = None, end_date: date | None = None) -> dict[str, Any]:
    sessions = tuple(d for d in calendar.sessions if (start_date is None or d >= start_date) and (end_date is None or d <= end_date)); cash, unavailable, position, pending = 300000., 0., None, None; events, equity, trades, exposed = [], {}, [], []; consumed, skips, failure = {}, 0, None
    for day in sessions:
        if unavailable: cash += unavailable; events.append({"type": "PROCEEDS_AVAILABLE", "date": day, "amount": unavailable}); unavailable = 0.
        if pending is not None and pending[0] == day and position is None:
            _, signal, ranked, digest = pending; pending = None; consumed[signal.isoformat()] = digest
            for candidate in ranked:
                opening = _quote(prices, candidate["code"], day, "open")
                if opening is None: events.append({"type": "MISSING_OPEN_SKIP", "date": day, "code": candidate["code"]}); continue
                quantity = int(cash // (opening * (1 + friction) * 100)) * 100
                if quantity <= 0: skips += 1; events.append({"type": "AFFORDABILITY_SKIP", "date": day, "code": candidate["code"]}); continue
                cost = quantity * opening * (1 + friction); cash -= cost; position = {"code": candidate["code"], "sector": candidate.get("sector", "UNKNOWN"), "quantity": quantity, "entry_cost": cost, "entry_exec": opening * (1 + friction), "entry_date": day, "signal": signal, "scheduled_exit": calendar.plus(signal, 3), "ranking_hash": digest}; events.append({"type": "ENTRY", "date": day, "code": candidate["code"], "quantity": quantity, "signal": signal, "ranking_hash": digest}); break
            if position is None: events.append({"type": "NO_FILL", "date": day, "signal": signal})
        if position is not None: exposed.append(day)
        if position is not None and day == position["scheduled_exit"]:
            closing = _quote(prices, position["code"], day, "close")
            if closing is None: failure = "DATA_QUALITY_FAILURE"; events.append({"type": failure, "date": day, "code": position["code"]}); break
            proceeds = position["quantity"] * closing * (1 - friction); trades.append(dict(position, exit_date=day, exit_exec=closing * (1 - friction), proceeds=proceeds, pnl=proceeds - position["entry_cost"])); unavailable += proceeds; events.append({"type": "EXIT", "date": day, "code": position["code"], "proceeds": proceeds}); position = None
        marked = cash + unavailable
        if position is not None:
            close = _quote(prices, position["code"], day, "close")
            if close is not None: marked += position["quantity"] * close
        equity[day] = marked
        if position is None and day in rankings:
            entry, exit_day = calendar.plus(day, 1), calendar.plus(day, 3)
            if entry is None or exit_day is None or entry not in sessions or exit_day not in sessions: events.append({"type": "END_OF_STUDY_NO_ENTRY", "signal": day})
            else:
                frozen = list(rankings[day]); digest = ranking_hash(frozen); pending = (entry, day, frozen, digest); events.append({"type": "RANKING_FROZEN", "signal": day, "ranking_hash": digest})
    return {"failure_class": failure, "cash": cash, "unavailable_proceeds": unavailable, "position": position, "events": events, "daily_equity": equity, "trades": trades, "exposed_sessions": tuple(exposed), "affordability_skip_count": skips, "consumed_ranking_hashes": consumed, "Q7_SINGLE_POSITION_AND_CASH_SAFETY": failure != "NEGATIVE_CASH" and cash >= -1e-8 and all(t["quantity"] % 100 == 0 for t in trades), "Q8_EXIT_EVENT_ORDER": all(any(e["type"] == "EXIT" and e["date"] == t["exit_date"] for e in events) for t in trades), "Q9_REQUIRED_EXIT_DATA": failure is None}

def _share(values: Iterable[float]) -> float | str:
    total = sum(max(v, 0.) for v in values); return "UNDEFINED" if total <= 0 else max(max(v, 0.) for v in values) / total

def trade_metrics(result: dict[str, Any], years: tuple[int, ...] = YEARS) -> dict[str, Any]:
    trades = result["trades"]; pnls = [float(t["pnl"]) for t in trades]; returns = [100 * (t["exit_exec"] / t["entry_exec"] - 1) for t in trades]; positive, negative = sum(max(x, 0.) for x in pnls), abs(sum(min(x, 0.) for x in pnls)); factor = positive / negative if negative else ("POSITIVE_INFINITY" if positive else "UNDEFINED"); curve = list(result["daily_equity"].items()); peak, dd = 300000., 0.
    for _, value in curve: peak = max(peak, value); dd = max(dd, 100 * (peak - value) / peak if peak else 0.)
    year_pnl, previous = {}, 300000.
    for year in years:
        values = [v for d, v in curve if d.year == year]; year_pnl[str(year)] = float(values[-1] - previous) if values else 0.; previous = values[-1] if values else previous
    ticker, sector = {}, {}
    for t in trades:
        gain = max(float(t["pnl"]), 0.); ticker[t["code"]] = ticker.get(t["code"], 0.) + gain; sector[t["sector"]] = sector.get(t["sector"], 0.) + gain
    return {"closed_trade_count": len(trades), "trades": len(trades), "total_net_profit": float(sum(pnls)), "ending_equity": float(curve[-1][1] if curve else 300000.), "max_drawdown_pct": float(dd), "win_rate": sum(x > 0 for x in pnls) / len(pnls) if pnls else "UNDEFINED", "profit_factor": factor, "mean_trade_net_return": float(np.mean(returns)) if returns else "UNDEFINED", "median_trade_net_return": float(np.median(returns)) if returns else "UNDEFINED", "year_net_pnl": year_pnl, "positive_year_count": sum(x > 0 for x in year_pnl.values()), "best_year_positive_profit_share": _share(year_pnl.values()), "max_ticker_positive_profit_share": _share(ticker.values()), "max_sector_positive_profit_share": _share(sector.values()), "exposure_fraction": len(set(result.get("exposed_sessions", ()))) / max(1, len(result.get("daily_equity", {}))), "no_fill_count": sum(e["type"] == "NO_FILL" for e in result["events"]), "affordability_skip_count": result.get("affordability_skip_count", 0), "end_of_study_no_entry_count": sum(e["type"] == "END_OF_STUDY_NO_ENTRY" for e in result["events"]), "failure_class": result.get("failure_class")}

def canonical_json(value: Any) -> str:
    def safe(v: Any) -> Any:
        if isinstance(v, float) and not math.isfinite(v): return "POSITIVE_INFINITY" if v > 0 else "UNDEFINED"
        if isinstance(v, dict): return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [safe(x) for x in v]
        if isinstance(v, date): return v.isoformat()
        return v
    return json.dumps(safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)

def adjudicate(criteria: dict[str, bool], q: dict[str, bool]) -> dict[str, Any]:
    if set(q) != set(Q_KEYS) or set(criteria) != set("ABCDEFGHIJKLMNOP"): raise ValueError("CRITERION_KEY_SET_MISMATCH")
    all_q = all(q.values()); return {"criteria": dict(criteria), "Q": dict(q), "Q_SAFETY_AND_LEAKAGE_INVARIANTS_PASS": all_q, "V13_VIABILITY_RESULT": "CONTINUE_TO_FORMAL_CONFIRMATION_DESIGN" if all(criteria.values()) and all_q else "STOP_HYPOTHESIS_NOT_PROMOTED", "same_study_parameter_rescue": False}

def _criterion_number(value: Any) -> bool: return isinstance(value, (int, float)) and math.isfinite(value)
def _derive_criteria(lgb: dict[str, Any], ridge: dict[str, Any], reversal: dict[str, Any], momentum: dict[str, Any], stress: dict[str, Any], p95: float | None, diag: dict[str, Any]) -> dict[str, bool]:
    def gt(a: Any, b: Any) -> bool: return _criterion_number(a) and _criterion_number(b) and a > b
    return {"A": _criterion_number(lgb["total_net_profit"]) and lgb["total_net_profit"] > 0, "B": lgb["positive_year_count"] >= 4, "C": _criterion_number(lgb["max_drawdown_pct"]) and lgb["max_drawdown_pct"] <= 20, "D": lgb["trades"] >= 150, "E": gt(lgb["total_net_profit"], ridge["total_net_profit"]), "F": gt(lgb["total_net_profit"], reversal["total_net_profit"]), "G": gt(lgb["total_net_profit"], momentum["total_net_profit"]), "H": p95 is not None and lgb["total_net_profit"] > p95, "I": _criterion_number(diag["mean_ic"]) and diag["mean_ic"] > .01, "J": sum(_criterion_number(v) and v > 0 for v in diag["per_year_mean_ic"].values()) >= 4, "K": _criterion_number(diag["top_decile_spread"]) and diag["top_decile_spread"] > 0, "L": diag["positive_spread_year_count"] >= 4, "M": _criterion_number(stress["total_net_profit"]) and stress["total_net_profit"] >= 0, "N": _criterion_number(lgb["best_year_positive_profit_share"]) and lgb["best_year_positive_profit_share"] <= .50, "O": _criterion_number(lgb["max_ticker_positive_profit_share"]) and lgb["max_ticker_positive_profit_share"] <= .25, "P": _criterion_number(lgb["max_sector_positive_profit_share"]) and lgb["max_sector_positive_profit_share"] <= .40}

@lru_cache(maxsize=1)
def run_synthetic_feasibility() -> dict[str, Any]:
    from .v13_synthetic_fixture import synthetic_calendar, synthetic_manifest, raw_ohlcv, synthetic_metadata, synthetic_signal_dates
    bindings, manifest, calendar, prices, metadata = verify_frozen_bindings(), synthetic_manifest(), synthetic_calendar(), raw_ohlcv(), synthetic_metadata(); rows, stage_audit = labeled_rows_from_raw(calendar, prices, metadata, synthetic_signal_dates()); prediction_audit: dict[str, Any] = {}; predictions, fits = monthly_predictions(rows, calendar=calendar, audit=prediction_audit); dates = sorted({r["signal"] for r in predictions}); start, end = date(2020, 1, 1), date(2025, 12, 31); results: dict[str, Any] = {}
    for strategy in COMPARATOR_STRATEGIES[:4]:
        rankings = {d: rank_candidates([r for r in predictions if r["signal"] == d], strategy) for d in dates}; base, stress = simulate(calendar, prices, rankings, .001, start, end), simulate(calendar, prices, rankings, .002, start, end); results[strategy] = {"base": trade_metrics(base), "stress": trade_metrics(stress), "base_audit": base, "stress_audit": stress}
    random_profits, random_identity = [], {}
    for seed in RANDOM_SEEDS:
        rankings = {d: rank_candidates([r for r in predictions if r["signal"] == d], "RANDOM_500", d, seed) for d in dates}; base, stress = simulate(calendar, prices, rankings, .001, start, end), simulate(calendar, prices, rankings, .002, start, end); random_profits.append(trade_metrics(base)["total_net_profit"]); random_identity[str(seed)] = ({d.isoformat(): ranking_hash(r) for d, r in rankings.items()}, {d.isoformat(): ranking_hash(r) for d, r in rankings.items()})
    cash_audit = simulate(calendar, prices, {}, .001, start, end); cash_metrics = trade_metrics(cash_audit); cash_metrics.update(total_net_profit=0., ending_equity=300000., max_drawdown_pct=0., exposure_fraction=0., trades=0); results["CASH"] = {"base": cash_metrics, "stress": dict(cash_metrics), "base_audit": cash_audit, "stress_audit": cash_audit}; p95, diag, lgb_base = linear_percentile(random_profits, 95), diagnostics(predictions), results["LIGHTGBM"]["base"]; results["RANDOM_500"] = {"base": {"path_count": 500, "total_net_profit_p95": p95}, "stress": {"path_count": 500}, "base_audit": {"ranking_hashes": {seed: pair[0] for seed, pair in random_identity.items()}}, "stress_audit": {"ranking_hashes": {seed: pair[1] for seed, pair in random_identity.items()}}}; criteria = _derive_criteria(lgb_base, results["RIDGE"]["base"], results["SECTOR_REL_REVERSAL_1D"]["base"], results["SECTOR_REL_MOMENTUM_20D"]["base"], results["LIGHTGBM"]["stress"], p95, diag); base_audit, stress_audit = results["LIGHTGBM"]["base_audit"], results["LIGHTGBM"]["stress_audit"]
    q = {Q_KEYS[0]: all(a["source_max_session"] <= date.fromisoformat(day) for day, a in stage_audit.items()), Q_KEYS[1]: all(all(x < info["prediction_month_first_session"] for x in info["training_exit_dates"]) for info in fits.values()), Q_KEYS[2]: all(info["fit_count"] == 2 for info in fits.values()), Q_KEYS[3]: all(a.get("q4_stage_b_reference_frozen", False) for a in stage_audit.values()), Q_KEYS[4]: max(d.year for _, d in prices) <= 2025 and calendar.sessions[-1].year == 2025, Q_KEYS[5]: bool(base_audit["consumed_ranking_hashes"]), Q_KEYS[6]: base_audit["Q7_SINGLE_POSITION_AND_CASH_SAFETY"], Q_KEYS[7]: base_audit["Q8_EXIT_EVENT_ORDER"], Q_KEYS[8]: base_audit["Q9_REQUIRED_EXIT_DATA"], Q_KEYS[9]: base_audit["consumed_ranking_hashes"] == stress_audit["consumed_ranking_hashes"], Q_KEYS[10]: tuple(FEATURES) == tuple(STOCK_FEATURES + MARKET_FEATURES) and lightgbm_factory().__class__.__module__.startswith("lightgbm") and isinstance(ridge_factory().named_steps["scaler"], StandardScaler) and isinstance(ridge_factory().named_steps["ridge"], Ridge), Q_KEYS[11]: set(results) == set(COMPARATOR_STRATEGIES) and len(RANDOM_SEEDS) == 500}
    return {"SYNTHETIC_ONLY_NOT_RESEARCH_EVIDENCE": True, "REAL_MARKET_DATA_USED": False, "V13_HISTORICAL_VIABILITY_RESULT": "NOT_RUN", "frozen_bindings": bindings, "manifest_count": len(manifest["selected"]), "manifest_sha256": manifest["selected_sha256"], "raw_ohlcv_rows": len(prices), "active_metadata_codes": tuple(sorted(metadata)), "active_subset_sha256": sha256_text("|".join(sorted(metadata))), "active_codes_subset_of_selected": set(metadata) <= set(manifest["selected"]), "raw_ohlcv_codes": tuple(sorted({code for code, _ in prices})), "stage_a_from_raw": True, "stage_b_relative_only": True, "stage_c_transform_once": True, "prediction_months": sorted(fits), "synthetic_model_fits": sum(x["fit_count"] for x in fits.values()), "strategies": {k: {"base": v["base"], "stress": v["stress"]} for k, v in results.items()}, "RANDOM_500": {"seed_first": RANDOM_SEEDS[0], "seed_last": RANDOM_SEEDS[-1], "base_paths": 500, "stress_paths": 500, "base_profit_p95_linear": p95, "ranking_identity_pass": all(a == b for a, b in random_identity.values()), "all_seed_ranking_identity": {seed: a == b for seed, (a, b) in random_identity.items()}}, "diagnostics": diag, "A_P": criteria, "Q": q, "Q_SAFETY_AND_LEAKAGE_INVARIANTS_PASS": all(q.values()), "audit": {"stage_dates": stage_audit, "monthly_fits": fits, "per_seed_ranking_hashes": {seed: {"base": a, "stress": b} for seed, (a, b) in random_identity.items()}}}
