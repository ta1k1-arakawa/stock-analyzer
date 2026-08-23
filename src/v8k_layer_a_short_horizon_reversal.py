"""Offline V8K Layer A short-horizon reversal measurement."""
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

from src.v5_b_candidate_ranker import (
    AI_ARM, BASELINE_ARM, EVAL_YEARS, _atr14, _frame, canonical_ticker,
    d5_target, normalize_universe, simulate_portfolio,
)

SCHEMA_VERSION = "V8K_LAYER_A_SHORT_HORIZON_REVERSAL_SCORECARD_V1"
MAX_CANDIDATES = 20
STARTING_CASH = 400_000.0
MAX_OPEN_POSITIONS = 2
REQUIRED_COLUMNS = (
    "evaluation_year", "signal_date", "entry_date", "exit_date", "ticker",
    "industry", "return_5d", "return_20d", "return_60d", "close_to_ma20",
    "close_to_ma60", "atr14", "candidate_status",
)


def _as_frame(raw: pd.DataFrame) -> pd.DataFrame:
    return _frame(raw)


def _normalized_price_frames(prices: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {canonical_ticker(ticker): _as_frame(frame) for ticker, frame in prices.items()}


def generate_eligible_candidates(
    prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame,
    splits: Mapping[str, set[pd.Timestamp]] | None = None,
    signal_from: str | pd.Timestamp = "2020-01-01",
    signal_to: str | pd.Timestamp = "2025-12-31",
    _normalized_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Mechanical V5-B admission copy, before its top-20 cutoff."""
    normalized = normalize_universe(universe)
    allowed = set(normalized.ticker)
    industries = normalized.set_index("ticker")["industry"].to_dict()
    lo, hi = pd.Timestamp(signal_from), pd.Timestamp(signal_to)
    if lo < pd.Timestamp("2020-01-01") or hi > pd.Timestamp("2025-12-31"):
        raise ValueError("EVALUATION_SIGNAL_OUTSIDE_2020_2025")
    rows: list[dict[str, Any]] = []
    for raw_ticker, raw in prices.items():
        ticker = canonical_ticker(raw_ticker)
        if ticker not in allowed:
            continue
        frame = _normalized_frames[ticker] if _normalized_frames is not None else _as_frame(raw)
        adjusted_close = frame["AdjClose"].astype(float)
        raw_close = frame["Close"].astype(float)
        factor = adjusted_close / raw_close
        atr14 = _atr14(frame["High"] * factor, frame["Low"] * factor, adjusted_close)
        return_5d = adjusted_close / adjusted_close.shift(5) - 1.0
        return_20d = adjusted_close / adjusted_close.shift(20) - 1.0
        return_60d = adjusted_close / adjusted_close.shift(60) - 1.0
        ma20, ma60 = adjusted_close.rolling(20).mean(), adjusted_close.rolling(60).mean()
        turnover = (frame["Close"] * frame["Volume"]).rolling(60).median()
        volume = frame["Volume"].rolling(60).median()
        for index, day in enumerate(frame.index):
            if day < lo or day > hi or index + 5 >= len(frame) or index < 252:
                continue
            entry_date, exit_date = frame.index[index + 1], frame.index[index + 5]
            if any(entry_date <= pd.Timestamp(split_day) <= exit_date for split_day in (splits or {}).get(ticker, set())):
                continue
            row = {"evaluation_year": int(day.year), "signal_date": day, "entry_date": entry_date,
                   "exit_date": exit_date, "ticker": ticker, "industry": industries.get(ticker, ""),
                   "return_5d": return_5d.iloc[index], "return_20d": return_20d.iloc[index],
                   "return_60d": return_60d.iloc[index], "close_to_ma20": adjusted_close.iloc[index] / ma20.iloc[index] - 1.0,
                   "close_to_ma60": adjusted_close.iloc[index] / ma60.iloc[index] - 1.0,
                   "atr14": atr14.iloc[index], "candidate_status": "CANDIDATE"}
            names = ("return_5d", "return_20d", "return_60d", "close_to_ma20", "close_to_ma60", "atr14")
            finite = np.isfinite([row[name] for name in names]).all()
            liquid = np.isfinite(turnover.iloc[index]) and np.isfinite(volume.iloc[index]) and turnover.iloc[index] >= 100_000_000 and volume.iloc[index] >= 50_000
            signal = adjusted_close.iloc[index] > ma60.iloc[index] and return_60d.iloc[index] > 0 and -.05 <= return_5d.iloc[index] <= 0 and row["close_to_ma20"] >= -.03
            if finite and liquid and signal:
                rows.append(row)
    result = pd.DataFrame(rows)
    return (result.sort_values(["signal_date", "ticker"], kind="mergesort").reset_index(drop=True)
            if len(result) else pd.DataFrame(columns=REQUIRED_COLUMNS))


def rank_baseline(eligible: pd.DataFrame) -> pd.DataFrame:
    ordered = eligible.sort_values(["signal_date", "return_60d", "return_20d", "ticker"], ascending=[True, False, False, True], kind="mergesort").copy()
    ordered["baseline_rank"] = ordered.groupby("signal_date").cumcount() + 1
    selected = ordered[ordered["baseline_rank"] <= MAX_CANDIDATES].copy()
    selected["ai_rank"] = selected["baseline_rank"]
    selected["reversal_status"] = "NOT_APPLICABLE_BASELINE"
    return selected.reset_index(drop=True)


def _peer_return_state(frames: Mapping[str, pd.DataFrame], universe_tickers: set[str]) -> dict[str, pd.Series]:
    state = {}
    for ticker, frame in frames.items():
        if ticker in universe_tickers:
            close = frame["AdjClose"].astype(float)
            state[ticker] = close / close.shift(5) - 1.0
    return state


def attach_reversal_scores(
    eligible: pd.DataFrame, prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame,
    _normalized_frames: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Attach exact causal five-observation same-industry selloff scores."""
    normalized = normalize_universe(universe)
    industry_by_ticker = normalized.set_index("ticker")["industry"].to_dict()
    universe_tickers = set(normalized.ticker)
    frames = dict(_normalized_frames) if _normalized_frames is not None else _normalized_price_frames(prices)
    members = {industry: tuple(sorted(ticker for ticker in universe_tickers if industry_by_ticker.get(ticker, "") == industry))
               for industry in sorted({industry for industry in industry_by_ticker.values() if industry})}
    returns_5 = _peer_return_state(frames, universe_tickers)
    output = eligible.copy()
    needed = {(industry_by_ticker.get(canonical_ticker(row["ticker"]), ""), pd.Timestamp(row["signal_date"]))
              for _, row in output.iterrows() if industry_by_ticker.get(canonical_ticker(row["ticker"]), "")}
    peer_values: dict[tuple[str, pd.Timestamp], dict[str, float]] = {}
    for industry, day in sorted(needed, key=lambda item: (item[0], item[1])):
        values = {}
        for ticker in members[industry]:
            series = returns_5.get(ticker)
            if series is not None and day in series.index:
                value = float(series.at[day])
                if np.isfinite(value):
                    values[ticker] = value
        peer_values[(industry, day)] = values
    medians, scores, statuses, counts = [], [], [], []
    for _, row in output.iterrows():
        ticker, day = canonical_ticker(row["ticker"]), pd.Timestamp(row["signal_date"])
        industry = industry_by_ticker.get(ticker, "")
        peers = [] if not industry else [value for peer, value in peer_values[(industry, day)].items() if peer != ticker]
        if not peers:
            medians.append(np.nan); scores.append(np.nan); statuses.append("REVERSAL_SCORE_UNAVAILABLE"); counts.append(0)
            continue
        median = float(np.median(peers))
        medians.append(median); scores.append(median - float(row["return_5d"])); statuses.append("REVERSAL_SCORE_AVAILABLE"); counts.append(len(peers))
    output["industry_peer_median_5d"] = medians
    output["relative_selloff_score"] = scores
    output["reversal_status"] = statuses
    output["industry_peer_count"] = counts
    return output


def rank_reversal(scored: pd.DataFrame) -> pd.DataFrame:
    available = scored[scored["reversal_status"].eq("REVERSAL_SCORE_AVAILABLE")].copy()
    ordered = available.sort_values(["signal_date", "relative_selloff_score", "return_60d", "return_20d", "ticker"], ascending=[True, False, False, False, True], kind="mergesort")
    ordered["ai_rank"] = ordered.groupby("signal_date").cumcount() + 1
    ordered["baseline_rank"] = ordered["ai_rank"]
    return ordered[ordered["ai_rank"] <= MAX_CANDIDATES].reset_index(drop=True)


def build_ranked_arms(prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str, set[pd.Timestamp]] | None = None,
                      signal_from: str | pd.Timestamp = "2020-01-01", signal_to: str | pd.Timestamp = "2025-12-31") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = _normalized_price_frames(prices)
    eligible = generate_eligible_candidates(prices, universe, splits, signal_from, signal_to, frames)
    return eligible, rank_baseline(eligible), rank_reversal(attach_reversal_scores(eligible, prices, universe, frames))


def execute_arms(baseline_rows: pd.DataFrame, reversal_rows: pd.DataFrame, prices: Mapping[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_trades, baseline_equity = simulate_portfolio(baseline_rows, prices, BASELINE_ARM)
    reversal_trades, reversal_equity = simulate_portfolio(reversal_rows, prices, AI_ARM)
    return baseline_trades, baseline_equity, reversal_trades, reversal_equity


def _drawdown(series: pd.Series) -> float:
    return float(((series.cummax() - series) / series.cummax() * 100.0).max()) if len(series) else 0.0


def arm_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict[str, Any]:
    filled = trades[trades.status.eq("FILLED")].copy() if len(trades) else pd.DataFrame()
    profits = filled.realized_net_profit_yen.astype(float) if len(filled) else pd.Series(dtype=float)
    gains, losses = float(profits[profits > 0].sum()), float(-profits[profits < 0].sum())
    yearly = {str(year): 0.0 for year in EVAL_YEARS}
    if len(filled): yearly.update({str(year): float(group.realized_net_profit_yen.sum()) for year, group in filled.groupby("evaluation_year")})
    mtm = {str(year): _drawdown(group.mark_to_market_equity.astype(float)) for year, group in equity.groupby("evaluation_year")} if len(equity) else {}
    book = {str(year): _drawdown(group.book_equity.astype(float)) for year, group in equity.groupby("evaluation_year")} if len(equity) else {}
    positive = filled[filled.realized_net_profit_yen > 0] if len(filled) else filled
    total_positive = float(positive.realized_net_profit_yen.sum()) if len(positive) else 0.0
    monthly = filled.assign(_month=pd.to_datetime(filled.exit_date).dt.to_period("M")).groupby("_month").realized_net_profit_yen.sum() if len(filled) else pd.Series(dtype=float)
    shares = positive.groupby("industry").realized_net_profit_yen.sum() / total_positive if total_positive else pd.Series(dtype=float)
    average_open = float(equity.open_positions.mean()) if len(equity) else 0.0
    return {"net_profit": float(profits.sum()), "profit_factor": gains / losses if losses else 0.0, "filled_trade_count": int(len(filled)), "win_rate": float((profits > 0).mean()) if len(profits) else 0.0,
            "average_profit": float(profits[profits > 0].mean()) if (profits > 0).any() else 0.0, "average_loss": float(profits[profits < 0].mean()) if (profits < 0).any() else 0.0,
            "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else 0.0, "mtm_maximum_drawdown": max(mtm.values()) if mtm else 0.0,
            "book_cost_maximum_drawdown": max(book.values()) if book else 0.0, "yearly_net_profit": yearly, "positive_year_count": int(sum(value > 0 for value in yearly.values())),
            "gross_entry_notional_yen": float(filled.entry_cost.sum()) if len(filled) else 0.0, "entry_notional_turnover_multiple": (float(filled.entry_cost.sum()) if len(filled) else 0.0) / (STARTING_CASH * len(EVAL_YEARS)),
            "average_open_positions": average_open, "slot_utilization_fraction": average_open / MAX_OPEN_POSITIONS,
            "top5_positive_trade_profit_share": float(positive.realized_net_profit_yen.nlargest(5).sum() / total_positive) if total_positive else 0.0,
            "maximum_industry_positive_profit_share": float(shares.max()) if len(shares) else 0.0, "mtm_drawdown_by_year": mtm, "book_cost_drawdown_by_year": book}


def _with_realized_return(rows: pd.DataFrame, prices: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames = {canonical_ticker(ticker): frame for ticker, frame in prices.items()}
    result = rows.copy()
    result["realized_d5_net_return"] = [d5_target(frames[canonical_ticker(ticker)], pd.Timestamp(day)) for ticker, day in zip(result.ticker, result.signal_date)]
    return result


def _discrimination(rows: pd.DataFrame, prices: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    valid = _with_realized_return(rows, prices).dropna(subset=["relative_selloff_score", "realized_d5_net_return"])
    def corr(frame: pd.DataFrame) -> float | None:
        value = frame.relative_selloff_score.corr(frame.realized_d5_net_return, method="spearman") if len(frame) >= 2 else np.nan
        return None if pd.isna(value) else float(value)
    yearly = {str(year): {"spearman": corr(valid[valid.evaluation_year.eq(year)]), "observation_count": int(valid.evaluation_year.eq(year).sum())} for year in EVAL_YEARS}
    return {"available_row_count": int(rows.reversal_status.eq("REVERSAL_SCORE_AVAILABLE").sum()), "valid_row_count": int(len(valid)), "pooled_spearman": corr(valid), "yearly": yearly}


def _all_eligible_discrimination(scored: pd.DataFrame, prices: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    result = _discrimination(scored, prices)
    valid = _with_realized_return(scored, prices).dropna(subset=["relative_selloff_score", "realized_d5_net_return"]).copy()
    quintiles = {str(number): {"count": 0, "mean_realized_d5_net_return": None, "positive_rate": None} for number in range(1, 6)}
    if len(valid):
        valid["quintile"] = pd.qcut(valid.relative_selloff_score.rank(method="first"), 5, labels=False) + 1
        for number in range(1, 6):
            group = valid[valid.quintile.eq(number)]
            if len(group): quintiles[str(number)] = {"count": int(len(group)), "mean_realized_d5_net_return": float(group.realized_d5_net_return.mean()), "positive_rate": float((group.realized_d5_net_return > 0).mean())}
    result["pooled_score_quintiles"] = quintiles
    return result


def top20_mechanism(baseline_rows: pd.DataFrame, reversal_rows: pd.DataFrame) -> dict[str, Any]:
    base = set(map(tuple, baseline_rows[["signal_date", "ticker"]].itertuples(index=False, name=None)))
    rev = set(map(tuple, reversal_rows[["signal_date", "ticker"]].itertuples(index=False, name=None)))
    dates = sorted(set(baseline_rows.signal_date) | set(reversal_rows.signal_date))
    date_jaccard = []
    for day in dates:
        a = {ticker for signal, ticker in base if signal == day}; b = {ticker for signal, ticker in rev if signal == day}
        date_jaccard.append(len(a & b) / len(a | b) if a | b else 1.0)
    return {"baseline_selected_count": len(base), "reversal_selected_count": len(rev), "intersection_count": len(base & rev), "baseline_only_count": len(base - rev), "reversal_only_count": len(rev - base),
            "overall_jaccard": len(base & rev) / len(base | rev) if base | rev else 1.0, "signal_date_count": len(dates), "changed_top20_set_dates": int(sum(value < 1.0 for value in date_jaccard)), "mean_per_date_jaccard": float(np.mean(date_jaccard)) if date_jaccard else 1.0}


def fill_mechanism(baseline_trades: pd.DataFrame, reversal_trades: pd.DataFrame) -> dict[str, Any]:
    key = ["evaluation_year", "signal_date", "ticker"]
    base_filled = baseline_trades[baseline_trades.status.eq("FILLED")] if len(baseline_trades) else pd.DataFrame()
    reversal_filled = reversal_trades[reversal_trades.status.eq("FILLED")] if len(reversal_trades) else pd.DataFrame()
    base = base_filled.set_index(key).realized_net_profit_yen.astype(float) if len(base_filled) else pd.Series(dtype=float)
    rev = reversal_filled.set_index(key).realized_net_profit_yen.astype(float) if len(reversal_filled) else pd.Series(dtype=float)
    common = base.index.intersection(rev.index)
    contribution = rev.reindex(base.index.union(rev.index), fill_value=0.0) - base.reindex(base.index.union(rev.index), fill_value=0.0)
    absolute = float(contribution.abs().sum())
    return {"baseline_fills": int(len(base)), "reversal_fills": int(len(rev)), "common_fills": int(len(common)), "baseline_only_fills": int(len(base.index.difference(rev.index))), "reversal_only_fills": int(len(rev.index.difference(base.index))),
            "baseline_only_pnl": float(base[base.index.difference(rev.index)].sum()), "reversal_only_pnl": float(rev[rev.index.difference(base.index)].sum()), "common_pnl_difference": float((rev.reindex(common) - base.reindex(common)).sum()), "net_profit_difference": float(rev.sum() - base.sum()),
            "top1_absolute_contribution_share": float(contribution.abs().nlargest(1).sum() / absolute) if absolute else 0.0, "top5_absolute_contribution_share": float(contribution.abs().nlargest(5).sum() / absolute) if absolute else 0.0, "top10_absolute_contribution_share": float(contribution.abs().nlargest(10).sum() / absolute) if absolute else 0.0}


def _metric_differences(baseline: Mapping[str, Any], reversal: Mapping[str, Any]) -> dict[str, float]:
    names = ("net_profit", "profit_factor", "filled_trade_count", "win_rate", "average_profit", "average_loss", "monthly_win_rate", "mtm_maximum_drawdown", "book_cost_maximum_drawdown", "positive_year_count", "gross_entry_notional_yen", "entry_notional_turnover_multiple", "average_open_positions", "slot_utilization_fraction", "top5_positive_trade_profit_share", "maximum_industry_positive_profit_share")
    return {name: float(reversal[name] - baseline[name]) for name in names}


def build_scorecard(prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str, set[pd.Timestamp]] | None = None, provenance: Mapping[str, Any] | None = None, repository_commit: str = "SYNTHETIC") -> dict[str, Any]:
    eligible, baseline_rows, reversal_rows = build_ranked_arms(prices, universe, splits)
    baseline_trades, baseline_equity, reversal_trades, reversal_equity = execute_arms(baseline_rows, reversal_rows, prices)
    baseline, reversal = arm_metrics(baseline_trades, baseline_equity), arm_metrics(reversal_trades, reversal_equity)
    return {"schema_version": SCHEMA_VERSION, "study": "V8K_HISTORICAL_RESEARCH", "layer_a_role": "HYPOTHESIS_GENERATION_AND_VIABILITY_SCREEN", "evidence_capacity": "ZERO", "exploratory_only": True, "measurement_status": "COMPLETE", "interpretation": "GPT_DECISION_REQUIRED", "promotion_thresholds_defined": False, "deployment_allowed": False, "future_profitability_established": False, "parameter_neighbor_robustness_status": "NOT_RUN_NO_FREE_PARAMETER_SEARCH", "repository_commit": repository_commit, "provenance": dict(provenance or {}), "safe_row_counts": {"eligible_pre_top20": int(len(eligible)), "baseline_selected": int(len(baseline_rows)), "reversal_selected": int(len(reversal_rows))}, "baseline": baseline, "reversal": reversal, "baseline_vs_reversal_difference": _metric_differences(baseline, reversal), "all_eligible_discrimination": _all_eligible_discrimination(attach_reversal_scores(eligible, prices, universe), prices), "selected_discrimination": _discrimination(reversal_rows, prices), "top20_mechanism": top20_mechanism(baseline_rows, reversal_rows), "fill_mechanism": fill_mechanism(baseline_trades, reversal_trades)}


def canonical_scorecard_bytes(scorecard: Mapping[str, Any]) -> bytes:
    return (json.dumps(scorecard, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


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
    universe_path = repository_root / "V4_UNIVERSE.csv"; validation = validate_evaluation_cache(evaluation_cache)
    manifest, prices, splits = _raw_cache(evaluation_cache); universe = normalize_universe(pd.read_csv(universe_path))
    provenance = git_provenance(repository_root, universe_path, validation); provenance["payload_hash_list_sha256"] = manifest["payload_hash_list_sha256"]
    first = canonical_scorecard_bytes(build_scorecard(prices, universe, splits, provenance, provenance["repository_exact_sha"]))
    second = canonical_scorecard_bytes(build_scorecard(prices, universe, splits, provenance, provenance["repository_exact_sha"]))
    if first != second: raise ValueError("TWO_PASS_SCORECARD_MISMATCH")
    write_scorecard(output_dir, first, repository_root)
