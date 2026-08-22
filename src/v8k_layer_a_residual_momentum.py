"""Offline V8K Layer A residual-momentum measurement.

This module consumes already-exposed cache frames only.  It has no transport,
model-fitting, sealed-data, or V7 dependency.  The sole comparison changes the
ordering of the pre-existing V5-A2/V5-B eligible rows.
"""
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
    AI_ARM,
    BASELINE_ARM,
    EVAL_YEARS,
    _atr14,
    _frame,
    canonical_ticker,
    d5_target,
    normalize_universe,
    simulate_portfolio,
)

SCHEMA_VERSION = "V8K_LAYER_A_RESIDUAL_MOMENTUM_SCORECARD_V1"
MAX_CANDIDATES = 20
STARTING_CASH = 400_000.0
MAX_OPEN_POSITIONS = 2
REQUIRED_COLUMNS = (
    "evaluation_year", "signal_date", "entry_date", "exit_date", "ticker",
    "industry", "return_5d", "return_20d", "return_60d", "close_to_ma20",
    "close_to_ma60", "atr14", "candidate_status",
)


def _as_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Use V5-B's canonical, offline input normalization."""
    return _frame(raw)


def generate_eligible_candidates(
    prices: Mapping[str, pd.DataFrame],
    universe: pd.DataFrame,
    splits: Mapping[str, set[pd.Timestamp]] | None = None,
    signal_from: str | pd.Timestamp = "2020-01-01",
    signal_to: str | pd.Timestamp = "2025-12-31",
) -> pd.DataFrame:
    """Return every V5-A2/V5-B eligible row before its absolute top-20 cut.

    The admission predicates and D1/D5 availability/split handling are copied
    mechanically from V5-B's post-2019 generator.  Only its final absolute
    ranking/cutoff is deliberately deferred to the two comparison arms.
    """
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
        frame = _as_frame(raw)
        adjusted_close = frame["AdjClose"].astype(float)
        raw_close = frame["Close"].astype(float)
        factor = adjusted_close / raw_close
        adjusted_high, adjusted_low = frame["High"] * factor, frame["Low"] * factor
        atr14 = _atr14(adjusted_high, adjusted_low, adjusted_close)
        return_5d = adjusted_close / adjusted_close.shift(5) - 1.0
        return_20d = adjusted_close / adjusted_close.shift(20) - 1.0
        return_60d = adjusted_close / adjusted_close.shift(60) - 1.0
        ma20, ma60 = adjusted_close.rolling(20).mean(), adjusted_close.rolling(60).mean()
        turnover = (frame["Close"] * frame["Volume"]).rolling(60).median()
        volume = frame["Volume"].rolling(60).median()
        split_days = (splits or {}).get(ticker, set())
        for index, day in enumerate(frame.index):
            if day < lo or day > hi or index + 5 >= len(frame) or index < 252:
                continue
            entry_date, exit_date = frame.index[index + 1], frame.index[index + 5]
            if any(entry_date <= pd.Timestamp(split_day) <= exit_date for split_day in split_days):
                continue
            row = {
                "evaluation_year": int(day.year), "signal_date": day,
                "entry_date": entry_date, "exit_date": exit_date, "ticker": ticker,
                "industry": industries.get(ticker, ""), "return_5d": return_5d.iloc[index],
                "return_20d": return_20d.iloc[index], "return_60d": return_60d.iloc[index],
                "close_to_ma20": adjusted_close.iloc[index] / ma20.iloc[index] - 1.0,
                "close_to_ma60": adjusted_close.iloc[index] / ma60.iloc[index] - 1.0,
                "atr14": atr14.iloc[index], "candidate_status": "CANDIDATE",
            }
            required = ("return_5d", "return_20d", "return_60d", "close_to_ma20", "close_to_ma60", "atr14")
            finite = np.isfinite([row[name] for name in required]).all()
            liquid = np.isfinite(turnover.iloc[index]) and np.isfinite(volume.iloc[index]) and turnover.iloc[index] >= 100_000_000 and volume.iloc[index] >= 50_000
            signal = adjusted_close.iloc[index] > ma60.iloc[index] and return_60d.iloc[index] > 0 and -.05 <= return_5d.iloc[index] <= 0 and row["close_to_ma20"] >= -.03
            if finite and liquid and signal:
                rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    return result.sort_values(["signal_date", "ticker"], kind="mergesort").reset_index(drop=True)


def rank_baseline(eligible: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen absolute-momentum V5-B ordering to all eligible rows."""
    ordered = eligible.sort_values(
        ["signal_date", "return_60d", "return_20d", "ticker"],
        ascending=[True, False, False, True], kind="mergesort",
    ).copy()
    ordered["baseline_rank"] = ordered.groupby("signal_date").cumcount() + 1
    selected = ordered[ordered["baseline_rank"] <= MAX_CANDIDATES].copy()
    selected["ai_rank"] = selected["baseline_rank"]
    selected["residual_status"] = "NOT_APPLICABLE_BASELINE"
    return selected.reset_index(drop=True)


def _peer_return_60(frame: pd.DataFrame, day: pd.Timestamp) -> float | None:
    normalized = _as_frame(frame)
    try:
        position = normalized.index.get_loc(pd.Timestamp(day))
    except KeyError:
        return None
    if not isinstance(position, (int, np.integer)) or position < 60:
        return None
    now, prior = float(normalized.iloc[position]["AdjClose"]), float(normalized.iloc[position - 60]["AdjClose"])
    value = now / prior - 1.0
    return value if np.isfinite(value) else None


def attach_residual_scores(
    eligible: pd.DataFrame,
    prices: Mapping[str, pd.DataFrame],
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """Score every eligible row against all same-industry universe peers.

    The candidate itself is excluded, and peer eligibility never affects the
    peer population.  No fallback is permitted for an unavailable peer median.
    """
    normalized = normalize_universe(universe)
    industry_by_ticker = normalized.set_index("ticker")["industry"].to_dict()
    universe_tickers = set(normalized.ticker)
    frames = {canonical_ticker(ticker): frame for ticker, frame in prices.items()}
    output = eligible.copy()
    medians: list[float] = []
    scores: list[float] = []
    statuses: list[str] = []
    peer_counts: list[int] = []
    for _, row in output.iterrows():
        ticker, day = canonical_ticker(row["ticker"]), pd.Timestamp(row["signal_date"])
        industry = industry_by_ticker.get(ticker, "")
        if not industry:
            medians.append(np.nan); scores.append(np.nan); statuses.append("RESIDUAL_SCORE_UNAVAILABLE"); peer_counts.append(0)
            continue
        peers = []
        for peer in universe_tickers:
            if peer == ticker or industry_by_ticker.get(peer, "") != industry or peer not in frames:
                continue
            value = _peer_return_60(frames[peer], day)
            if value is not None:
                peers.append(value)
        if not peers:
            medians.append(np.nan); scores.append(np.nan); statuses.append("RESIDUAL_SCORE_UNAVAILABLE"); peer_counts.append(0)
            continue
        median = float(np.median(peers))
        medians.append(median); scores.append(float(row["return_60d"]) - median)
        statuses.append("RESIDUAL_SCORE_AVAILABLE"); peer_counts.append(len(peers))
    output["industry_peer_median_60d"] = medians
    output["residual_momentum"] = scores
    output["residual_status"] = statuses
    output["industry_peer_count"] = peer_counts
    return output


def rank_residual(scored: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen residual ordering, without a market/industry fallback."""
    available = scored[scored["residual_status"].eq("RESIDUAL_SCORE_AVAILABLE")].copy()
    ordered = available.sort_values(
        ["signal_date", "residual_momentum", "return_60d", "return_20d", "ticker"],
        ascending=[True, False, False, False, True], kind="mergesort",
    )
    ordered["ai_rank"] = ordered.groupby("signal_date").cumcount() + 1
    ordered["baseline_rank"] = ordered["ai_rank"]
    return ordered[ordered["ai_rank"] <= MAX_CANDIDATES].reset_index(drop=True)


def build_ranked_arms(
    prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame,
    splits: Mapping[str, set[pd.Timestamp]] | None = None,
    signal_from: str | pd.Timestamp = "2020-01-01",
    signal_to: str | pd.Timestamp = "2025-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    eligible = generate_eligible_candidates(prices, universe, splits, signal_from, signal_to)
    baseline = rank_baseline(eligible)
    residual = rank_residual(attach_residual_scores(eligible, prices, universe))
    return eligible, baseline, residual


def execute_arms(
    baseline_rows: pd.DataFrame, residual_rows: pd.DataFrame, prices: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Delegate both arms to the unchanged V5-B fixed-100 D5-only engine."""
    baseline_trades, baseline_equity = simulate_portfolio(baseline_rows, prices, BASELINE_ARM)
    residual_trades, residual_equity = simulate_portfolio(residual_rows, prices, AI_ARM)
    return baseline_trades, baseline_equity, residual_trades, residual_equity


def _drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(((series.cummax() - series) / series.cummax() * 100.0).max())


def arm_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict[str, Any]:
    """Compute the Layer A scorecard metrics, including independent DD series."""
    filled = trades[trades.status.eq("FILLED")].copy() if len(trades) else pd.DataFrame()
    profits = filled.realized_net_profit_yen.astype(float) if len(filled) else pd.Series(dtype=float)
    gains, losses = float(profits[profits > 0].sum()), float(-profits[profits < 0].sum())
    yearly = {str(year): 0.0 for year in EVAL_YEARS}
    if len(filled):
        yearly.update({str(year): float(group.realized_net_profit_yen.sum()) for year, group in filled.groupby("evaluation_year")})
    mtm_by_year = {str(year): _drawdown(group.mark_to_market_equity.astype(float)) for year, group in equity.groupby("evaluation_year")} if len(equity) else {}
    book_by_year = {str(year): _drawdown(group.book_equity.astype(float)) for year, group in equity.groupby("evaluation_year")} if len(equity) else {}
    positive = filled[filled.realized_net_profit_yen > 0] if len(filled) else filled
    total_positive = float(positive.realized_net_profit_yen.sum()) if len(positive) else 0.0
    monthly = filled.assign(_month=pd.to_datetime(filled.exit_date).dt.to_period("M")).groupby("_month").realized_net_profit_yen.sum() if len(filled) else pd.Series(dtype=float)
    gross_entry = float(filled.entry_cost.sum()) if len(filled) else 0.0
    industry_share = positive.groupby("industry").realized_net_profit_yen.sum() / total_positive if total_positive else pd.Series(dtype=float)
    top5_share = float(positive.realized_net_profit_yen.nlargest(5).sum() / total_positive) if total_positive else 0.0
    average_open = float(equity.open_positions.mean()) if len(equity) else 0.0
    return {
        "net_profit": float(profits.sum()),
        "profit_factor": gains / losses if losses else 0.0,
        "filled_trade_count": int(len(filled)),
        "win_rate": float((profits > 0).mean()) if len(profits) else 0.0,
        "average_profit": float(profits[profits > 0].mean()) if (profits > 0).any() else 0.0,
        "average_loss": float(profits[profits < 0].mean()) if (profits < 0).any() else 0.0,
        "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "mtm_maximum_drawdown": max(mtm_by_year.values()) if mtm_by_year else 0.0,
        "book_cost_maximum_drawdown": max(book_by_year.values()) if book_by_year else 0.0,
        "yearly_net_profit": yearly,
        "positive_year_count": int(sum(value > 0 for value in yearly.values())),
        "gross_entry_notional_yen": gross_entry,
        "entry_notional_turnover_multiple": gross_entry / (STARTING_CASH * len(EVAL_YEARS)),
        "average_open_positions": average_open,
        "slot_utilization_fraction": average_open / MAX_OPEN_POSITIONS,
        "top5_positive_trade_profit_share": top5_share,
        "maximum_industry_positive_profit_share": float(industry_share.max()) if len(industry_share) else 0.0,
        "mtm_drawdown_by_year": mtm_by_year,
        "book_cost_drawdown_by_year": book_by_year,
    }


def _discrimination(residual_rows: pd.DataFrame, prices: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    frames = {canonical_ticker(ticker): frame for ticker, frame in prices.items()}
    rows = residual_rows.copy()
    rows["realized_d5_net_return"] = [d5_target(frames[canonical_ticker(ticker)], pd.Timestamp(day)) for ticker, day in zip(rows.ticker, rows.signal_date)]
    valid = rows.dropna(subset=["residual_momentum", "realized_d5_net_return"])
    def correlation(frame: pd.DataFrame) -> float | None:
        if len(frame) < 2:
            return None
        value = frame["residual_momentum"].corr(frame["realized_d5_net_return"], method="spearman")
        return None if pd.isna(value) else float(value)
    yearly = {}
    for year in EVAL_YEARS:
        group = valid[valid.evaluation_year.eq(year)]
        yearly[str(year)] = {"spearman": correlation(group), "observation_count": int(len(group))}
    return {"pooled_spearman": correlation(valid), "observation_count": int(len(valid)), "yearly": yearly}


def _metric_differences(baseline: Mapping[str, Any], residual: Mapping[str, Any]) -> dict[str, float]:
    names = ("net_profit", "profit_factor", "filled_trade_count", "win_rate", "average_profit", "average_loss", "monthly_win_rate", "mtm_maximum_drawdown", "book_cost_maximum_drawdown", "positive_year_count", "gross_entry_notional_yen", "entry_notional_turnover_multiple", "average_open_positions", "slot_utilization_fraction", "top5_positive_trade_profit_share", "maximum_industry_positive_profit_share")
    return {name: float(residual[name] - baseline[name]) for name in names}


def build_scorecard(
    prices: Mapping[str, pd.DataFrame], universe: pd.DataFrame,
    splits: Mapping[str, set[pd.Timestamp]] | None = None,
    provenance: Mapping[str, Any] | None = None,
    repository_commit: str = "SYNTHETIC",
) -> dict[str, Any]:
    eligible, baseline_rows, residual_rows = build_ranked_arms(prices, universe, splits)
    baseline_trades, baseline_equity, residual_trades, residual_equity = execute_arms(baseline_rows, residual_rows, prices)
    baseline = arm_metrics(baseline_trades, baseline_equity)
    residual = arm_metrics(residual_trades, residual_equity)
    return {
        "schema_version": SCHEMA_VERSION,
        "study": "V8K_HISTORICAL_RESEARCH",
        "layer_a_role": "HYPOTHESIS_GENERATION_AND_VIABILITY_SCREEN",
        "evidence_capacity": "ZERO",
        "exploratory_only": True,
        "measurement_status": "COMPLETE",
        "interpretation": "GPT_DECISION_REQUIRED",
        "promotion_thresholds_defined": False,
        "deployment_allowed": False,
        "future_profitability_established": False,
        "parameter_neighbor_robustness_status": "NOT_RUN_NO_FREE_PARAMETER_SEARCH",
        "repository_commit": repository_commit,
        "provenance": dict(provenance or {}),
        "safe_row_counts": {"eligible_pre_top20": int(len(eligible)), "baseline_selected": int(len(baseline_rows)), "residual_selected": int(len(residual_rows))},
        "baseline": baseline,
        "residual": residual,
        "baseline_vs_residual_difference": _metric_differences(baseline, residual),
        "residual_discrimination": _discrimination(residual_rows, prices),
    }


def canonical_scorecard_bytes(scorecard: Mapping[str, Any]) -> bytes:
    return (json.dumps(scorecard, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def write_scorecard(output_dir: Path, body: bytes, repository_root: Path) -> None:
    """Atomically write exactly one public Layer A artifact outside the repo."""
    output = output_dir.resolve()
    root = repository_root.resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError("OUTPUT_INSIDE_REPOSITORY_PROHIBITED")
    if output.exists() and (output.is_file() or any(output.iterdir())):
        raise ValueError("OUTPUT_DIRECTORY_NONEMPTY_OR_FILE")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    try:
        staging.mkdir(parents=True)
        target = staging / "scorecard.json"
        with target.open("wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        if target.read_bytes() != body:
            raise ValueError("SCORECARD_WRITE_VERIFY_FAILED")
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def git_provenance(repository_root: Path, universe_path: Path, cache_validation: Mapping[str, Any]) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repository_root, text=True, capture_output=True, check=True).stdout.strip()
    return {
        "repository_exact_sha": git("rev-parse", "HEAD"),
        "universe_git_blob": git("rev-parse", ":V4_UNIVERSE.csv"),
        "universe_sha256": sha256(universe_path.read_bytes()).hexdigest(),
        "cache_manifest_sha256": cache_validation["manifest_sha256"],
        "payload_count": cache_validation["payload_count"],
        "usable_date_range": {"min_date": cache_validation["min_date"], "max_date": cache_validation["max_date"]},
    }


def run_cache_measurement(evaluation_cache: Path, output_dir: Path, repository_root: Path) -> None:
    """Validate the frozen public cache, then perform two byte-identical cores."""
    from scripts.run_v5_b_candidate_ranker import _raw_cache, validate_evaluation_cache
    universe_path = repository_root / "V4_UNIVERSE.csv"
    validation = validate_evaluation_cache(evaluation_cache)
    manifest, prices, splits = _raw_cache(evaluation_cache)
    universe = normalize_universe(pd.read_csv(universe_path))
    provenance = git_provenance(repository_root, universe_path, validation)
    provenance["payload_hash_list_sha256"] = manifest["payload_hash_list_sha256"]
    first = canonical_scorecard_bytes(build_scorecard(prices, universe, splits, provenance, provenance["repository_exact_sha"]))
    second = canonical_scorecard_bytes(build_scorecard(prices, universe, splits, provenance, provenance["repository_exact_sha"]))
    if first != second:
        raise ValueError("TWO_PASS_SCORECARD_MISMATCH")
    write_scorecard(output_dir, first, repository_root)
