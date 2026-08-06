"""Formal evaluator for V6-A-R2; portfolio decisions stay in the causal engine."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from v6_a_r2_causal_breakout import CausalEventEngine, LEDGER_FIELDS, concentration_metrics, fold_max_drawdown
from v6_a_r2_preflight import EXPECTED_PREFLIGHT, R2_CANDIDATE_COLUMNS, candidate_key_sha256

CONFIRMATION = "V6_A_R2_ONE_SHOT_EXPLORATORY_EVALUATION"
YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
ARTIFACTS = ("summary.json", "trades.csv", "candidates.csv", "daily_equity.csv")
EXPECTED_HASH = "4c550c8635a192fc4d60a753d8ac77ca9f992dc62bad3f36f19ef7512c29e818"
V5B = {"net_profit": 122536.15709488306, "profit_factor": 1.1138514271409448,
       "mtm_dd": 26.782565969991488, "filled_trades": 569, "positive_years": 3,
       "yearly_profit": {"2020": -27792.634676513204, "2021": -106195.98642242365,
                         "2022": -45253.59194076466, "2023": 114181.43414215161,
                         "2024": 102867.2727392584, "2025": 84729.66325317451}}
SAFETY_KEYS = ("negative_cash_count", "same_day_proceeds_reuse_count", "duplicate_order_count",
               "max_position_violation_count", "cash_reserve_violation_count", "industry_overlap_violation_count",
               "signal_2026_count", "future_price_access_violation_count", "d0_state_mutation_violation_count")

class FormalBlocked(RuntimeError):
    pass

def _iso(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")

def build_engine_price_frames(raw_frames: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    """Adapt source-aware raw frames; this function never makes trade decisions."""
    output: dict[str, dict[str, dict[str, float]]] = {}
    for ticker, frame in raw_frames.items():
        if isinstance(frame, pd.DataFrame):
            rows = ((_iso(index), row) for index, row in frame.iterrows())
        else:
            rows = ((str(day), values) for day, values in frame.items())
        output[str(ticker)] = {day: {"Open": float(values["Open"]), "Close": float(values["Close"])}
                               for day, values in rows}
    return output

def build_formal_bundle(preflight: Mapping[str, Any], raw_frames: Mapping[str, Any], common_calendar: Sequence[Any],
                        accepted_candidates: Sequence[Mapping[str, Any]], full_candidate_audit: Any,
                        market_gate_audit: Any = None, diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    rows = [dict(row) for row in accepted_candidates]
    bundle = {"raw_price_frames": raw_frames, "price_frames": build_engine_price_frames(raw_frames),
              "common_calendar": [_iso(day) for day in common_calendar], "accepted_candidates": rows,
              "candidate_audit": full_candidate_audit, "market_gate_audit": market_gate_audit,
              "preflight_diagnostics": dict(diagnostics or preflight),
              "accepted_candidate_key_sha256": candidate_key_sha256(
                  [f"{r['signal_date']}|{r['ticker']}|{int(r['rank'])}" for r in rows])}
    return bundle

def build_fold_calendar(bundle: Mapping[str, Any], year: int) -> list[str]:
    candidates = [row for row in bundle["accepted_candidates"] if int(row["signal_year"]) == year]
    if not candidates: return []
    calendar = list(bundle["common_calendar"]); end = max(str(row["planned_exit_date"]) for row in candidates)
    index = calendar.index(end)
    if index + 1 >= len(calendar): raise FormalBlocked("FOLD_PROCEEDS_RELEASE_DATE_MISSING")
    start = next((i for i, day in enumerate(calendar) if str(day).startswith(f"{year}-")), None)
    if start is None: raise FormalBlocked("FOLD_START_DATE_MISSING")
    return calendar[start:index + 2]

def _validate_bundle(bundle: Mapping[str, Any]) -> None:
    diag = bundle["preflight_diagnostics"]
    if bundle["accepted_candidate_key_sha256"] != EXPECTED_HASH: raise FormalBlocked("ACCEPTED_CANDIDATE_HASH_MISMATCH")
    if any(set(row) != set(R2_CANDIDATE_COLUMNS) for row in bundle["accepted_candidates"]):
        raise FormalBlocked("CANDIDATE_FUTURE_VALUE_COLUMN_PROHIBITED")
    checks = {"accepted_top20_candidates": len(bundle["accepted_candidates"]),
              "signal_days": len({r["signal_date"] for r in bundle["accepted_candidates"]}),
              "yearly_candidate_counts": {str(y): sum(int(r["signal_year"]) == y for r in bundle["accepted_candidates"]) for y in YEARS},
              "market_gate_pass_days": diag.get("market_gate_counts", {}).get("pass_days"),
              "market_gate_blocked_days": diag.get("market_gate_counts", {}).get("blocked_days"),
              "split_violations": diag.get("split_violations"), "duplicate_accepted_key": diag.get("duplicate_accepted_key"),
              "2026_signals": diag.get("2026_signals")}
    for name in ("accepted_top20_candidates", "signal_days", "yearly_candidate_counts", "market_gate_pass_days", "market_gate_blocked_days", "split_violations", "duplicate_accepted_key", "2026_signals"):
        if checks[name] != EXPECTED_PREFLIGHT[name]: raise FormalBlocked(f"FORMAL_PREFLIGHT_MISMATCH:{name}")

def run_one_fold(bundle: Mapping[str, Any], year: int, engine_factory: Callable[..., Any] = CausalEventEngine) -> dict[str, Any]:
    calendar = build_fold_calendar(bundle, year); candidates = [r for r in bundle["accepted_candidates"] if int(r["signal_year"]) == year]
    engine = engine_factory(bundle["price_frames"], calendar, candidates, starting_cash=400000.0).run()
    state = engine.state
    if state.open_positions or state.pending_orders_by_entry_date or state.pending_proceeds_by_available_date: raise FormalBlocked("FOLD_TERMINAL_STATE_NOT_EMPTY")
    ids = [r.get("order_id") for r in state.completed_trades]
    expected = {f"{r['signal_date']}|{int(r['rank']):02d}|{r['ticker']}" for r in candidates}
    if len(ids) != len(candidates) or len(ids) != len(set(ids)) or set(ids) != expected: raise FormalBlocked("FOLD_COMPLETED_ORDER_MISMATCH")
    if any(row["status"] not in {"CLOSED", "SKIPPED"} for row in state.completed_trades): raise FormalBlocked("FOLD_NONTERMINAL_ORDER")
    if any(r["status"] == "CLOSED" and (not r.get("exit_execution_date") or not r.get("proceeds_available_date")) for r in state.completed_trades): raise FormalBlocked("FOLD_CLOSED_FIELDS_MISSING")
    if any(e.get("event") == "EXIT_EXECUTED" for e in state.event_audit) and not any(e.get("event") == "PROCEEDS_RELEASED" and e.get("date") in calendar for e in state.event_audit): raise FormalBlocked("FOLD_PROCEEDS_RELEASE_MISSING")
    return {"year": year, "engine": engine, "trades": [dict(x) for x in state.completed_trades], "equity": [dict(x) for x in state.daily_equity], "candidates": candidates}

def _pf(profits: Sequence[float]) -> tuple[float | None, bool]:
    gross_profit = sum(x for x in profits if x > 0); gross_loss = -sum(x for x in profits if x < 0)
    if gross_loss: return gross_profit / gross_loss, False
    return (None, True) if gross_profit else (0.0, False)

def compute_fold_metrics(fold: Mapping[str, Any]) -> dict[str, Any]:
    closed = [r for r in fold["trades"] if r["status"] == "CLOSED"]; profits = [float(r["realized_net_profit_yen"]) for r in closed]
    wins, losses = [x for x in profits if x > 0], [x for x in profits if x < 0]; pf, infinite = _pf(profits)
    months = {str(r.get("signal_date", r.get("exit_execution_date")))[:7]: [] for r in closed}
    for r, p in zip(closed, profits): months[str(r.get("signal_date", r.get("exit_execution_date")))[:7]].append(p)
    return {"net_profit": sum(profits), "ending_equity_equivalent": 400000.0 + sum(profits), "filled_trade_count": len(closed),
            "win_rate": 100.0 * len(wins) / len(closed) if closed else 0.0, "profit_factor": pf, "profit_factor_infinite": infinite,
            "average_profit": sum(wins) / len(wins) if wins else 0.0, "average_loss": sum(losses) / len(losses) if losses else 0.0,
            "maximum_profit": max(wins, default=0.0), "maximum_loss": min(losses, default=0.0),
            "monthly_win_rate": 100.0 * sum(sum(p) > 0 for p in months.values()) / len(months) if months else 0.0,
            "mark_to_market_maximum_drawdown": fold_max_drawdown({str(fold["year"]): [x["mtm_equity"] for x in fold["equity"]]}),
            "book_cost_maximum_drawdown": fold_max_drawdown({str(fold["year"]): [x["book_equity"] for x in fold["equity"]]}),
            "average_holding_period": 10.0, "maximum_open_positions": max((x["open_position_count"] for x in fold["equity"]), default=0),
            "skip_reason_counts": {reason: sum(r["skip_reason"] == reason for r in fold["trades"]) for reason in sorted({r["skip_reason"] for r in fold["trades"] if r["status"] == "SKIPPED"})},
            "yearly_profit": sum(profits), "signal_day_count": len({r["signal_date"] for r in fold["candidates"]}), "candidate_count": len(fold["candidates"])}

def compute_aggregate_metrics(folds: Sequence[Mapping[str, Any]], metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    trades = [r for f in folds for r in f["trades"] if r["status"] == "CLOSED"]; profits = [float(r["realized_net_profit_yen"]) for r in trades]; pf, infinite = _pf(profits)
    wins, losses = [x for x in profits if x > 0], [x for x in profits if x < 0]
    months: dict[str, list[float]] = {}
    for r in trades: months.setdefault(str(r["signal_date"])[:7], []).append(float(r["realized_net_profit_yen"]))
    skipped = [r for f in folds for r in f["trades"] if r["status"] == "SKIPPED"]
    return {"net_profit": sum(profits), "ending_equity_equivalent": 400000.0 + sum(profits), "filled_trade_count": len(trades), "win_rate": 100.0 * len(wins) / len(trades) if trades else 0.0,
            "profit_factor": pf, "profit_factor_infinite": infinite, "average_profit": sum(wins)/len(wins) if wins else 0.0, "average_loss": sum(losses)/len(losses) if losses else 0.0, "maximum_profit": max(wins, default=0.0), "maximum_loss": min(losses, default=0.0), "monthly_win_rate": 100.0 * sum(sum(x)>0 for x in months.values()) / len(months) if months else 0.0,
            "mark_to_market_maximum_drawdown": max((m["mark_to_market_maximum_drawdown"] for m in metrics.values()), default=0.0),
            "book_cost_maximum_drawdown": max((m["book_cost_maximum_drawdown"] for m in metrics.values()), default=0.0), "average_holding_period": 10.0, "maximum_open_positions": max((m.get("maximum_open_positions", 0) for m in metrics.values()), default=0), "skip_reason_counts": {x: sum(r.get("skip_reason")==x for r in skipped) for x in sorted({r.get("skip_reason") for r in skipped})}, "yearly_profit": {y: m["yearly_profit"] for y,m in metrics.items()}, "signal_day_count": len({r["signal_date"] for f in folds for r in f.get("candidates", [])}), "candidate_count": sum(len(f.get("candidates", [])) for f in folds), **concentration_metrics(trades)}

def validate_output_target(output_dir: str | Path, repository_root: str | Path) -> None:
    out, repo = Path(output_dir).resolve(), Path(repository_root).resolve(); staging = out.with_name(out.name + ".staging")
    if repo == out or repo in out.parents: raise FormalBlocked("OUTPUT_DIRECTORY_INSIDE_REPOSITORY")
    if out.exists() and out.is_file(): raise FormalBlocked("OUTPUT_PATH_IS_FILE")
    if out.exists() and any(out.iterdir()): raise FormalBlocked("OUTPUT_DIRECTORY_NOT_EMPTY")
    if staging.exists(): raise FormalBlocked("STAGING_DIRECTORY_EXISTS")
    if set(ARTIFACTS) != {"summary.json", "trades.csv", "candidates.csv", "daily_equity.csv"}: raise FormalBlocked("ARTIFACT_SET_INVALID")
    try: out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error: raise FormalBlocked("OUTPUT_PARENT_UNWRITABLE") from error

def compute_v5b_comparison(aggregate: Mapping[str, Any], yearly: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    pf = math.inf if aggregate["profit_factor_infinite"] else float(aggregate["profit_factor"])
    return {"net_profit_difference": aggregate["net_profit"] - V5B["net_profit"], "profit_factor_difference": None if math.isinf(pf) else pf - V5B["profit_factor"],
            "profit_factor_infinite": math.isinf(pf), "mtm_dd_difference": aggregate["mark_to_market_maximum_drawdown"] - V5B["mtm_dd"],
            "filled_trade_difference": aggregate["filled_trade_count"] - V5B["filled_trades"],
            "positive_year_count_difference": sum(m["yearly_profit"] > 0 for m in yearly.values()) - V5B["positive_years"],
            "yearly_profit_difference": {y: yearly[y]["yearly_profit"] - V5B["yearly_profit"][y] for y in V5B["yearly_profit"]}}

def compute_twenty_gates(aggregate: Mapping[str, Any], yearly: Mapping[str, Mapping[str, Any]], comparison: Mapping[str, Any], safety: Mapping[str, int], two_pass_byte_identical: bool) -> dict[str, bool]:
    pf = math.inf if aggregate["profit_factor_infinite"] else float(aggregate["profit_factor"])
    names = (("aggregate_net_profit_positive", aggregate["net_profit"] > 0), ("aggregate_profit_factor_gt_1_05", pf > 1.05),
             ("positive_years_at_least_4", sum(x["yearly_profit"] > 0 for x in yearly.values()) >= 4), ("aggregate_mtm_dd_at_most_20pct", aggregate["mark_to_market_maximum_drawdown"] <= 20),
             ("filled_trades_at_least_100", aggregate["filled_trade_count"] >= 100), ("each_year_at_least_10_trades", all(x["filled_trade_count"] >= 10 for x in yearly.values())),
             ("net_profit_beats_v5b", comparison["net_profit_difference"] > 0), ("profit_factor_beats_v5b", comparison["profit_factor_infinite"] or comparison["profit_factor_difference"] > 0),
             ("mtm_dd_beats_v5b", comparison["mtm_dd_difference"] < 0), ("years_beating_v5b_at_least_4", sum(x > 0 for x in comparison["yearly_profit_difference"].values()) >= 4),
             ("top5_profit_share_at_most_50pct", aggregate["top5_positive_profit_share"] <= .5), ("industry_profit_share_at_most_40pct", aggregate["max_industry_positive_profit_share"] <= .4))
    safety_names = (("negative_cash_zero", "negative_cash_count"), ("same_day_proceeds_reuse_zero", "same_day_proceeds_reuse_count"), ("duplicate_order_zero", "duplicate_order_count"), ("max_position_violation_zero", "max_position_violation_count"), ("cash_reserve_violation_zero", "cash_reserve_violation_count"), ("industry_overlap_violation_zero", "industry_overlap_violation_count"), ("signals_2026_zero", "signal_2026_count"))
    return dict(names + tuple((n, safety[k] == 0) for n, k in safety_names) + (("two_pass_byte_identical", two_pass_byte_identical),))

def _csv_bytes(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
    out = io.StringIO(newline=""); writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows([{k: "" if r.get(k) is None else r.get(k) for k in fields} for r in rows]); return out.getvalue().encode("utf-8")

def build_formal_artifacts(summary: Mapping[str, Any], folds: Sequence[Mapping[str, Any]], candidate_audit: Any) -> dict[str, bytes]:
    trades = [{**r, "order_id": r["order_id"], "fold_year": f["year"], "exit_reason": "TIME" if r["status"] == "CLOSED" else ""} for f in folds for r in f["trades"]]
    trades.sort(key=lambda r: (r["fold_year"], r["signal_date"], r["rank"], r["ticker"]))
    equity = [{"fold_year": f["year"], **r} for f in folds for r in f["equity"]]; equity.sort(key=lambda r: (r["fold_year"], r["date"]))
    audit_rows = candidate_audit.to_dict("records") if isinstance(candidate_audit, pd.DataFrame) else list(candidate_audit)
    candidate_fields = sorted({k for r in audit_rows for k in r}) or ["candidate_status"]
    return {"summary.json": (json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode("utf-8"),
            "trades.csv": _csv_bytes([*LEDGER_FIELDS, "order_id", "fold_year", "exit_reason"], trades), "candidates.csv": _csv_bytes(candidate_fields, audit_rows),
            "daily_equity.csv": _csv_bytes(["fold_year", "date", "pending_proceeds", "book_equity", "mtm_equity", "available_cash", "open_position_count"], equity)}

def atomic_write_formal_artifacts(output_dir: str | Path, artifacts: Mapping[str, bytes], repository_root: str | Path) -> None:
    validate_output_target(output_dir, repository_root)
    out = Path(output_dir).resolve()
    staging = out.with_name(out.name + ".staging")
    if staging.exists(): raise FormalBlocked("STAGING_DIRECTORY_EXISTS")
    try:
        if out.exists(): out.rmdir()
        staging.mkdir(parents=True); [ (staging / name).write_bytes(artifacts[name]) for name in ARTIFACTS ]
        if set(p.name for p in staging.iterdir()) != set(ARTIFACTS): raise FormalBlocked("ARTIFACT_SET_INVALID")
        os.replace(staging, out)
    except Exception:
        if staging.exists(): shutil.rmtree(staging)
        raise

def run_formal_two_pass(bundle: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    _validate_bundle(bundle); passes = []
    for _ in range(2):
        folds = [run_one_fold(bundle, y) for y in YEARS]; yearly = {str(f["year"]): compute_fold_metrics(f) for f in folds}
        safety = {k: sum(f["engine"].safety_counters()[k] for f in folds) for k in SAFETY_KEYS}
        if safety["future_price_access_violation_count"] or safety["d0_state_mutation_violation_count"]: raise FormalBlocked("ENGINE_INTEGRITY_VIOLATION")
        aggregate = compute_aggregate_metrics(folds, yearly); comparison = compute_v5b_comparison(aggregate, yearly)
        gates = compute_twenty_gates(aggregate, yearly, comparison, safety, True)
        summary = {"schema_version":"V6-A-R2-1", **metadata, "experiment":"V6-A-R2", "exploratory_only":True,"unused_holdout":False,"deployment_allowed":False,"ai_used":False,"survivorship_bias":True,"accepted_candidate_key_sha256":EXPECTED_HASH,"evaluation_years":list(YEARS),"candidate_rules":{"history":252,"turnover60":100000000,"volume60":50000,"price_cap_for_100_shares":220000,"breadth":.5,"volatility_ratio":.8,"volume_surprise":1.5,"max_candidates":20},"ranking_rules":{"breakout_strength_atr":"descending","volume_surprise":"descending","return60":"descending","ticker":"ascending"},"portfolio_rules":{"starting_cash":400000,"quantity":100,"max_open_positions":2,"cash_reserve":40000,"capital_limit":220000,"entry_gap_limit":1.02,"entry_slippage":.0003,"exit_slippage":.0003,"exit":"D10_TIME","same_day_proceeds_reuse":False,"same_industry_concurrent":False},"market_gate_pass_day_count":691,"market_gate_blocked_day_count":774,"signal_day_count":346,"accepted_candidate_count":608,"yearly_candidate_counts":{y:yearly[y]["candidate_count"] for y in yearly},"candidate_audit_row_count":len(bundle["candidate_audit"]),"aggregate_metrics":aggregate,"yearly_metrics":yearly,"comparison_to_v5b":comparison,"20_gates":gates,"safety_counters":safety,"two_pass_byte_identical":True,"formal_confirmation":CONFIRMATION,"verdict":"V6_A_BREAKOUT_BASELINE_EXPLORATORY_PROMISING" if all(gates.values()) else "V6_A_BREAKOUT_BASELINE_EXPLORATORY_NOT_PROMISING"}
        passes.append((summary, build_formal_artifacts(summary, folds, bundle["candidate_audit"])))
    if passes[0][1] != passes[1][1]: raise FormalBlocked("TWO_PASS_ARTIFACT_MISMATCH")
    return {"summary": passes[0][0], "artifacts": passes[0][1]}
