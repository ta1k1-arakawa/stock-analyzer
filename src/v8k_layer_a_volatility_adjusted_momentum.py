"""Offline V8K Layer A volatility-adjusted momentum measurement."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.v5_b_candidate_ranker import AI_ARM, BASELINE_ARM, EVAL_YEARS, _atr14, _frame, canonical_ticker, normalize_universe, simulate_portfolio

SCHEMA_VERSION = "V8K_LAYER_A_VOLATILITY_ADJUSTED_MOMENTUM_SCORECARD_V1"
MAX_CANDIDATES = 20
STARTING_CASH = 400_000.0
MAX_OPEN_POSITIONS = 2
REQUIRED_COLUMNS = ("evaluation_year", "signal_date", "entry_date", "exit_date", "ticker", "industry", "return_5d", "return_20d", "return_60d", "close_to_ma20", "close_to_ma60", "atr14", "candidate_status")


def _as_frame(raw: pd.DataFrame) -> pd.DataFrame: return _frame(raw)


def _normalized_price_frames(prices: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {canonical_ticker(ticker): _as_frame(frame) for ticker, frame in prices.items()}


def generate_eligible_candidates(prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str, set[pd.Timestamp]] | None = None, signal_from: str | pd.Timestamp = "2020-01-01", signal_to: str | pd.Timestamp = "2025-12-31", _normalized_frames: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    normalized = normalize_universe(universe); allowed = set(normalized.ticker); industries = normalized.set_index("ticker").industry.to_dict()
    lo, hi = pd.Timestamp(signal_from), pd.Timestamp(signal_to)
    if lo < pd.Timestamp("2020-01-01") or hi > pd.Timestamp("2025-12-31"): raise ValueError("EVALUATION_SIGNAL_OUTSIDE_2020_2025")
    rows = []
    for raw_ticker, raw in prices.items():
        ticker = canonical_ticker(raw_ticker)
        if ticker not in allowed: continue
        frame = _normalized_frames[ticker] if _normalized_frames is not None else _as_frame(raw)
        adjusted = frame.AdjClose.astype(float); close = frame.Close.astype(float); factor = adjusted / close
        atr14 = _atr14(frame.High * factor, frame.Low * factor, adjusted); r5 = adjusted / adjusted.shift(5) - 1; r20 = adjusted / adjusted.shift(20) - 1; r60 = adjusted / adjusted.shift(60) - 1
        ma20, ma60 = adjusted.rolling(20).mean(), adjusted.rolling(60).mean(); turnover = (frame.Close * frame.Volume).rolling(60).median(); volume = frame.Volume.rolling(60).median()
        for index, day in enumerate(frame.index):
            if day < lo or day > hi or index + 5 >= len(frame) or index < 252: continue
            entry, exit_ = frame.index[index + 1], frame.index[index + 5]
            if any(entry <= pd.Timestamp(split_day) <= exit_ for split_day in (splits or {}).get(ticker, set())): continue
            row = {"evaluation_year": int(day.year), "signal_date": day, "entry_date": entry, "exit_date": exit_, "ticker": ticker, "industry": industries.get(ticker, ""), "return_5d": r5.iloc[index], "return_20d": r20.iloc[index], "return_60d": r60.iloc[index], "close_to_ma20": adjusted.iloc[index] / ma20.iloc[index] - 1, "close_to_ma60": adjusted.iloc[index] / ma60.iloc[index] - 1, "atr14": atr14.iloc[index], "candidate_status": "CANDIDATE"}
            names = ("return_5d", "return_20d", "return_60d", "close_to_ma20", "close_to_ma60", "atr14")
            if np.isfinite([row[name] for name in names]).all() and np.isfinite(turnover.iloc[index]) and np.isfinite(volume.iloc[index]) and turnover.iloc[index] >= 100_000_000 and volume.iloc[index] >= 50_000 and adjusted.iloc[index] > ma60.iloc[index] and r60.iloc[index] > 0 and -.05 <= r5.iloc[index] <= 0 and row["close_to_ma20"] >= -.03: rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["signal_date", "ticker"], kind="mergesort").reset_index(drop=True) if len(result) else pd.DataFrame(columns=REQUIRED_COLUMNS)


def rank_baseline(eligible: pd.DataFrame) -> pd.DataFrame:
    ordered = eligible.sort_values(["signal_date", "return_60d", "return_20d", "ticker"], ascending=[True, False, False, True], kind="mergesort").copy(); ordered["baseline_rank"] = ordered.groupby("signal_date").cumcount() + 1
    selected = ordered[ordered.baseline_rank <= MAX_CANDIDATES].copy(); selected["ai_rank"] = selected.baseline_rank; selected["volatility_adjusted_status"] = "NOT_APPLICABLE_BASELINE"
    return selected.reset_index(drop=True)


def attach_volatility_adjusted_scores(eligible: pd.DataFrame, prices: Mapping[str, pd.DataFrame], _normalized_frames: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Use exact D0 adjusted close and the already-fixed candidate ATR14."""
    frames = dict(_normalized_frames) if _normalized_frames is not None else _normalized_price_frames(prices)
    output = eligible.copy(); normalized_atr, scores, statuses = [], [], []
    for _, row in output.iterrows():
        ticker, day = canonical_ticker(row.ticker), pd.Timestamp(row.signal_date)
        try: close = float(frames[ticker].at[day, "AdjClose"])
        except KeyError: close = np.nan
        atr = float(row.atr14)
        if not (np.isfinite(close) and close > 0 and np.isfinite(atr)):
            normalized_atr.append(np.nan); scores.append(np.nan); statuses.append("SCORE_UNAVAILABLE")
            continue
        value = atr / close
        if not (np.isfinite(value) and value > 0): normalized_atr.append(np.nan); scores.append(np.nan); statuses.append("SCORE_UNAVAILABLE")
        else: normalized_atr.append(value); scores.append(float(row.return_60d) / value); statuses.append("SCORE_AVAILABLE")
    output["normalized_atr14"] = normalized_atr; output["risk_adjusted_momentum_score"] = scores; output["volatility_adjusted_status"] = statuses
    return output


def rank_volatility_adjusted(scored: pd.DataFrame) -> pd.DataFrame:
    available = scored[scored.volatility_adjusted_status.eq("SCORE_AVAILABLE")].copy()
    ordered = available.sort_values(["signal_date", "risk_adjusted_momentum_score", "return_60d", "return_20d", "ticker"], ascending=[True, False, False, False, True], kind="mergesort"); ordered["ai_rank"] = ordered.groupby("signal_date").cumcount() + 1; ordered["baseline_rank"] = ordered.ai_rank
    return ordered[ordered.ai_rank <= MAX_CANDIDATES].reset_index(drop=True)


def build_ranked_arms(prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str, set[pd.Timestamp]] | None = None, signal_from: str | pd.Timestamp = "2020-01-01", signal_to: str | pd.Timestamp = "2025-12-31") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = _normalized_price_frames(prices); eligible = generate_eligible_candidates(prices, universe, splits, signal_from, signal_to, frames)
    return eligible, rank_baseline(eligible), rank_volatility_adjusted(attach_volatility_adjusted_scores(eligible, prices, frames))


def execute_arms(baseline_rows: pd.DataFrame, variant_rows: pd.DataFrame, prices: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_trades, baseline_equity = simulate_portfolio(baseline_rows, prices, BASELINE_ARM); variant_trades, variant_equity = simulate_portfolio(variant_rows, prices, AI_ARM)
    return baseline_trades, baseline_equity, variant_trades, variant_equity


def _drawdown(series: pd.Series) -> float: return float(((series.cummax() - series) / series.cummax() * 100).max()) if len(series) else 0.0


def arm_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict[str, Any]:
    filled = trades[trades.status.eq("FILLED")].copy() if len(trades) else pd.DataFrame(); profits = filled.realized_net_profit_yen.astype(float) if len(filled) else pd.Series(dtype=float); gains, losses = float(profits[profits > 0].sum()), float(-profits[profits < 0].sum())
    yearly = {str(year): 0.0 for year in EVAL_YEARS}; yearly.update({str(year): float(group.realized_net_profit_yen.sum()) for year, group in filled.groupby("evaluation_year")}) if len(filled) else None
    mtm = {str(year): _drawdown(group.mark_to_market_equity.astype(float)) for year, group in equity.groupby("evaluation_year")} if len(equity) else {}; book = {str(year): _drawdown(group.book_equity.astype(float)) for year, group in equity.groupby("evaluation_year")} if len(equity) else {}
    positive = filled[filled.realized_net_profit_yen > 0] if len(filled) else filled; total = float(positive.realized_net_profit_yen.sum()) if len(positive) else 0.; monthly = filled.assign(_month=pd.to_datetime(filled.exit_date).dt.to_period("M")).groupby("_month").realized_net_profit_yen.sum() if len(filled) else pd.Series(dtype=float); shares = positive.groupby("industry").realized_net_profit_yen.sum() / total if total else pd.Series(dtype=float); gross = float(filled.entry_cost.sum()) if len(filled) else 0.; average_open = float(equity.open_positions.mean()) if len(equity) else 0.
    return {"net_profit": float(profits.sum()), "profit_factor": gains / losses if losses else 0., "filled_trade_count": int(len(filled)), "win_rate": float((profits > 0).mean()) if len(profits) else 0., "average_profit": float(profits[profits > 0].mean()) if (profits > 0).any() else 0., "average_loss": float(profits[profits < 0].mean()) if (profits < 0).any() else 0., "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else 0., "mtm_maximum_drawdown": max(mtm.values()) if mtm else 0., "book_cost_maximum_drawdown": max(book.values()) if book else 0., "yearly_net_profit": yearly, "positive_year_count": int(sum(value > 0 for value in yearly.values())), "gross_entry_notional_yen": gross, "entry_notional_turnover_multiple": gross / (STARTING_CASH * len(EVAL_YEARS)), "average_open_positions": average_open, "slot_utilization_fraction": average_open / MAX_OPEN_POSITIONS, "top5_positive_trade_profit_share": float(positive.realized_net_profit_yen.nlargest(5).sum() / total) if total else 0., "maximum_industry_positive_profit_share": float(shares.max()) if len(shares) else 0., "mtm_drawdown_by_year": mtm, "book_cost_drawdown_by_year": book}


def _normalized_d5_target(frame: pd.DataFrame, signal_date: pd.Timestamp) -> float | None:
    try:
        index = frame.index.get_loc(pd.Timestamp(signal_date)); entry = index + 1; exit_ = index + 5
        if exit_ >= len(frame) or float(frame.iloc[entry].Open) > float(frame.iloc[index].Close) * 1.01: return None
        return float(frame.iloc[exit_].Open * .9997 / (frame.iloc[entry].Open * 1.0003) - 1)
    except (KeyError, IndexError, TypeError): return None


def _realized_d5_state(rows: pd.DataFrame, frames: Mapping[str, pd.DataFrame]) -> dict[tuple[pd.Timestamp, str], float | None]:
    keys = sorted({(pd.Timestamp(day), canonical_ticker(ticker)) for ticker, day in zip(rows.ticker, rows.signal_date)}, key=lambda item: (item[0], item[1])); return {(day, ticker): _normalized_d5_target(frames[ticker], day) for day, ticker in keys}


def _with_outcomes(rows: pd.DataFrame, outcomes: Mapping[tuple[pd.Timestamp, str], float | None]) -> pd.DataFrame:
    result = rows.copy(); result["realized_d5_net_return"] = [outcomes[(pd.Timestamp(day), canonical_ticker(ticker))] for ticker, day in zip(result.ticker, result.signal_date)]; return result


def _discrimination(rows: pd.DataFrame, outcomes: Mapping[tuple[pd.Timestamp, str], float | None]) -> dict[str, Any]:
    valid = _with_outcomes(rows, outcomes).dropna(subset=["risk_adjusted_momentum_score", "realized_d5_net_return"])
    def corr(frame: pd.DataFrame) -> float | None:
        value = frame.risk_adjusted_momentum_score.corr(frame.realized_d5_net_return, method="spearman") if len(frame) >= 2 else np.nan; return None if pd.isna(value) else float(value)
    return {"available_row_count": int(rows.volatility_adjusted_status.eq("SCORE_AVAILABLE").sum()), "valid_row_count": int(len(valid)), "pooled_spearman": corr(valid), "yearly": {str(year): {"spearman": corr(valid[valid.evaluation_year.eq(year)]), "observation_count": int(valid.evaluation_year.eq(year).sum())} for year in EVAL_YEARS}}


def _all_eligible_discrimination(scored: pd.DataFrame, outcomes: Mapping[tuple[pd.Timestamp, str], float | None]) -> dict[str, Any]:
    result = _discrimination(scored, outcomes); valid = _with_outcomes(scored, outcomes).dropna(subset=["risk_adjusted_momentum_score", "realized_d5_net_return"]).copy(); quintiles = {str(number): {"count": 0, "mean_realized_d5_net_return": None, "positive_rate": None} for number in range(1, 6)}
    if len(valid):
        valid["quintile"] = pd.qcut(valid.risk_adjusted_momentum_score.rank(method="first"), 5, labels=False) + 1
        for number in range(1, 6):
            group = valid[valid.quintile.eq(number)]
            if len(group): quintiles[str(number)] = {"count": int(len(group)), "mean_realized_d5_net_return": float(group.realized_d5_net_return.mean()), "positive_rate": float((group.realized_d5_net_return > 0).mean())}
    result["pooled_score_quintiles"] = quintiles; return result


def top20_mechanism(baseline_rows: pd.DataFrame, variant_rows: pd.DataFrame) -> dict[str, Any]:
    base = set(map(tuple, baseline_rows[["signal_date", "ticker"]].itertuples(index=False, name=None))); variant = set(map(tuple, variant_rows[["signal_date", "ticker"]].itertuples(index=False, name=None))); dates = sorted(set(baseline_rows.signal_date) | set(variant_rows.signal_date)); per_date = []
    for day in dates:
        left = {ticker for signal, ticker in base if signal == day}; right = {ticker for signal, ticker in variant if signal == day}; per_date.append(len(left & right) / len(left | right) if left | right else 1.)
    return {"baseline_selected_count": len(base), "variant_selected_count": len(variant), "intersection_count": len(base & variant), "baseline_only_count": len(base - variant), "variant_only_count": len(variant - base), "overall_jaccard": len(base & variant) / len(base | variant) if base | variant else 1., "signal_date_count": len(dates), "changed_top20_set_dates": int(sum(value < 1 for value in per_date)), "mean_per_date_jaccard": float(np.mean(per_date)) if per_date else 1.}


def fill_mechanism(baseline_trades: pd.DataFrame, variant_trades: pd.DataFrame) -> dict[str, Any]:
    key = ["evaluation_year", "signal_date", "ticker"]; left = baseline_trades[baseline_trades.status.eq("FILLED")] if len(baseline_trades) else pd.DataFrame(); right = variant_trades[variant_trades.status.eq("FILLED")] if len(variant_trades) else pd.DataFrame(); base = left.set_index(key).realized_net_profit_yen.astype(float) if len(left) else pd.Series(dtype=float); variant = right.set_index(key).realized_net_profit_yen.astype(float) if len(right) else pd.Series(dtype=float); common = base.index.intersection(variant.index); contribution = variant.reindex(base.index.union(variant.index), fill_value=0.) - base.reindex(base.index.union(variant.index), fill_value=0.); total = float(contribution.abs().sum())
    return {"baseline_fills": int(len(base)), "variant_fills": int(len(variant)), "common_fills": int(len(common)), "baseline_only_fills": int(len(base.index.difference(variant.index))), "variant_only_fills": int(len(variant.index.difference(base.index))), "baseline_only_pnl": float(base[base.index.difference(variant.index)].sum()), "variant_only_pnl": float(variant[variant.index.difference(base.index)].sum()), "common_pnl_difference": float((variant.reindex(common) - base.reindex(common)).sum()), "net_profit_difference": float(variant.sum() - base.sum()), "top1_absolute_contribution_share": float(contribution.abs().nlargest(1).sum() / total) if total else 0., "top5_absolute_contribution_share": float(contribution.abs().nlargest(5).sum() / total) if total else 0., "top10_absolute_contribution_share": float(contribution.abs().nlargest(10).sum() / total) if total else 0.}


def _metric_differences(baseline: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, float]:
    names = ("net_profit", "profit_factor", "filled_trade_count", "win_rate", "average_profit", "average_loss", "monthly_win_rate", "mtm_maximum_drawdown", "book_cost_maximum_drawdown", "positive_year_count", "gross_entry_notional_yen", "entry_notional_turnover_multiple", "average_open_positions", "slot_utilization_fraction", "top5_positive_trade_profit_share", "maximum_industry_positive_profit_share"); return {name: float(variant[name] - baseline[name]) for name in names}


def build_scorecard(prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str, set[pd.Timestamp]] | None = None, provenance: Mapping[str, Any] | None = None, repository_commit: str = "SYNTHETIC") -> dict[str, Any]:
    frames = _normalized_price_frames(prices); eligible = generate_eligible_candidates(prices, universe, splits, _normalized_frames=frames); baseline_rows = rank_baseline(eligible); scored = attach_volatility_adjusted_scores(eligible, prices, frames); variant_rows = rank_volatility_adjusted(scored); outcomes = _realized_d5_state(scored, frames); baseline_trades, baseline_equity, variant_trades, variant_equity = execute_arms(baseline_rows, variant_rows, prices); baseline, variant = arm_metrics(baseline_trades, baseline_equity), arm_metrics(variant_trades, variant_equity)
    return {"schema_version": SCHEMA_VERSION, "study": "V8K_HISTORICAL_RESEARCH", "layer_a_role": "HYPOTHESIS_GENERATION_AND_VIABILITY_SCREEN", "evidence_capacity": "ZERO", "exploratory_only": True, "measurement_status": "COMPLETE", "interpretation": "GPT_DECISION_REQUIRED", "promotion_thresholds_defined": False, "deployment_allowed": False, "future_profitability_established": False, "parameter_neighbor_robustness_status": "NOT_RUN_NO_FREE_PARAMETER_SEARCH", "repository_commit": repository_commit, "provenance": dict(provenance or {}), "safe_row_counts": {"eligible_pre_top20": int(len(eligible)), "baseline_selected": int(len(baseline_rows)), "variant_selected": int(len(variant_rows))}, "baseline": baseline, "variant": variant, "baseline_vs_variant_difference": _metric_differences(baseline, variant), "all_eligible_discrimination": _all_eligible_discrimination(scored, outcomes), "selected_discrimination": _discrimination(variant_rows, outcomes), "top20_mechanism": top20_mechanism(baseline_rows, variant_rows), "fill_mechanism": fill_mechanism(baseline_trades, variant_trades)}


def canonical_scorecard_bytes(scorecard: Mapping[str, Any]) -> bytes: return (json.dumps(scorecard, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_scorecard(output_dir: Path, body: bytes, repository_root: Path) -> None:
    output, root = output_dir.resolve(), repository_root.resolve()
    try: output.relative_to(root)
    except ValueError: pass
    else: raise ValueError("OUTPUT_INSIDE_REPOSITORY_PROHIBITED")
    if output.exists() and (output.is_file() or any(output.iterdir())): raise ValueError("OUTPUT_DIRECTORY_NONEMPTY_OR_FILE")
    staging = output.with_name(output.name + ".staging")
    if staging.exists(): shutil.rmtree(staging)
    try:
        staging.mkdir(parents=True); target = staging / "scorecard.json"
        with target.open("wb") as handle: handle.write(body); handle.flush(); os.fsync(handle.fileno())
        if target.read_bytes() != body: raise ValueError("SCORECARD_WRITE_VERIFY_FAILED")
        os.replace(staging, output)
    finally:
        if staging.exists(): shutil.rmtree(staging, ignore_errors=True)


def git_provenance(repository_root: Path, universe_path: Path, cache_validation: Mapping[str, Any]) -> dict[str, Any]:
    def git(*args: str) -> str: return subprocess.run(["git", *args], cwd=repository_root, text=True, capture_output=True, check=True).stdout.strip()
    return {"repository_exact_sha": git("rev-parse", "HEAD"), "universe_git_blob": git("rev-parse", ":V4_UNIVERSE.csv"), "universe_sha256": sha256(universe_path.read_bytes()).hexdigest(), "cache_manifest_sha256": cache_validation["manifest_sha256"], "payload_count": cache_validation["payload_count"], "usable_date_range": {"min_date": cache_validation["min_date"], "max_date": cache_validation["max_date"]}}


def run_cache_measurement(evaluation_cache: Path, output_dir: Path, repository_root: Path) -> None:
    from scripts.run_v5_b_candidate_ranker import _raw_cache, validate_evaluation_cache
    universe_path = repository_root / "V4_UNIVERSE.csv"; validation = validate_evaluation_cache(evaluation_cache); manifest, prices, splits = _raw_cache(evaluation_cache); universe = normalize_universe(pd.read_csv(universe_path)); provenance = git_provenance(repository_root, universe_path, validation); provenance["payload_hash_list_sha256"] = manifest["payload_hash_list_sha256"]
    first = canonical_scorecard_bytes(build_scorecard(prices, universe, splits, provenance, provenance["repository_exact_sha"])); second = canonical_scorecard_bytes(build_scorecard(prices, universe, splits, provenance, provenance["repository_exact_sha"]))
    if first != second: raise ValueError("TWO_PASS_SCORECARD_MISMATCH")
    write_scorecard(output_dir, first, repository_root)
