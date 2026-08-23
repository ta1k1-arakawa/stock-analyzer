"""Offline V8K Layer A pullback volume-dry-up measurement."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

import src.v8k_layer_a_volatility_adjusted_momentum as common
from src.v5_b_candidate_ranker import canonical_ticker, normalize_universe

SCHEMA_VERSION = "V8K_LAYER_A_PULLBACK_VOLUME_DRYUP_SCORECARD_V1"
MAX_CANDIDATES = common.MAX_CANDIDATES


def generate_eligible_candidates(*args, **kwargs):
    """Exact V5-B admission implementation reused without changing it."""
    return common.generate_eligible_candidates(*args, **kwargs)


def rank_baseline(eligible: pd.DataFrame) -> pd.DataFrame:
    return common.rank_baseline(eligible)


def attach_volume_dryup_scores(eligible: pd.DataFrame, prices: Mapping[str, pd.DataFrame], _normalized_frames: Mapping[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    frames = dict(_normalized_frames) if _normalized_frames is not None else common._normalized_price_frames(prices)
    output = eligible.copy(); ratios = []; scores = []; statuses = []
    for _, row in output.iterrows():
        try:
            frame = frames[canonical_ticker(row.ticker)]; position = frame.index.get_loc(pd.Timestamp(row.signal_date))
            if not isinstance(position, (int, np.integer)) or position < 19: raise KeyError
            volume = frame.Volume.astype(float); denominator = float(volume.iloc[position - 19:position + 1].mean()); numerator = float(volume.iloc[position - 4:position + 1].mean()); ratio = numerator / denominator
            if not (np.isfinite(denominator) and denominator > 0 and np.isfinite(ratio)): raise ValueError
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            ratios.append(np.nan); scores.append(np.nan); statuses.append("SCORE_UNAVAILABLE")
        else:
            ratios.append(ratio); scores.append(1.0 - ratio); statuses.append("SCORE_AVAILABLE")
    output["volume_ratio_5_20"] = ratios; output["volume_dryup_score"] = scores; output["volume_dryup_status"] = statuses
    # Generic diagnostic helpers retain their internal score/status names only.
    output["risk_adjusted_momentum_score"] = output["volume_dryup_score"]
    output["volatility_adjusted_status"] = output["volume_dryup_status"]
    return output


def rank_volume_dryup(scored: pd.DataFrame) -> pd.DataFrame:
    available = scored[scored.volume_dryup_status.eq("SCORE_AVAILABLE")].copy()
    ordered = available.sort_values(["signal_date", "volume_dryup_score", "return_60d", "return_20d", "ticker"], ascending=[True, False, False, False, True], kind="mergesort")
    ordered["ai_rank"] = ordered.groupby("signal_date").cumcount() + 1; ordered["baseline_rank"] = ordered.ai_rank
    return ordered[ordered.ai_rank <= MAX_CANDIDATES].reset_index(drop=True)


def build_ranked_arms(prices, universe, splits=None, signal_from="2020-01-01", signal_to="2025-12-31"):
    frames = common._normalized_price_frames(prices); eligible = generate_eligible_candidates(prices, universe, splits, signal_from, signal_to, frames); scored = attach_volume_dryup_scores(eligible, prices, frames)
    return eligible, rank_baseline(eligible), rank_volume_dryup(scored)


def execute_arms(baseline_rows, variant_rows, prices): return common.execute_arms(baseline_rows, variant_rows, prices)
def arm_metrics(trades, equity): return common.arm_metrics(trades, equity)
def top20_mechanism(baseline_rows, variant_rows):
    result = common.top20_mechanism(baseline_rows, variant_rows); result["variant_selected_count"] = result.pop("variant_selected_count"); return result
def fill_mechanism(baseline_trades, variant_trades): return common.fill_mechanism(baseline_trades, variant_trades)
def _normalized_d5_target(frame, day): return common._normalized_d5_target(frame, day)
def _realized_d5_state(rows, frames): return common._realized_d5_state(rows, frames)


def build_scorecard(prices, universe, splits=None, provenance=None, repository_commit="SYNTHETIC") -> dict[str, Any]:
    frames = common._normalized_price_frames(prices); eligible = generate_eligible_candidates(prices, universe, splits, _normalized_frames=frames); baseline_rows = rank_baseline(eligible); scored = attach_volume_dryup_scores(eligible, prices, frames); variant_rows = rank_volume_dryup(scored); outcomes = _realized_d5_state(scored, frames)
    baseline_trades, baseline_equity, variant_trades, variant_equity = execute_arms(baseline_rows, variant_rows, prices); baseline, variant = arm_metrics(baseline_trades, baseline_equity), arm_metrics(variant_trades, variant_equity)
    return {"schema_version": SCHEMA_VERSION, "study": "V8K_HISTORICAL_RESEARCH", "layer_a_role": "HYPOTHESIS_GENERATION_AND_VIABILITY_SCREEN", "evidence_capacity": "ZERO", "exploratory_only": True, "measurement_status": "COMPLETE", "interpretation": "GPT_DECISION_REQUIRED", "promotion_thresholds_defined": False, "deployment_allowed": False, "future_profitability_established": False, "parameter_neighbor_robustness_status": "NOT_RUN_NO_FREE_PARAMETER_SEARCH", "repository_commit": repository_commit, "provenance": dict(provenance or {}), "safe_row_counts": {"eligible_pre_top20": int(len(eligible)), "baseline_selected": int(len(baseline_rows)), "variant_selected": int(len(variant_rows))}, "baseline": baseline, "variant": variant, "baseline_vs_variant_difference": common._metric_differences(baseline, variant), "all_eligible_discrimination": common._all_eligible_discrimination(scored, outcomes), "selected_discrimination": common._discrimination(variant_rows, outcomes), "top20_mechanism": top20_mechanism(baseline_rows, variant_rows), "fill_mechanism": fill_mechanism(baseline_trades, variant_trades)}


def canonical_scorecard_bytes(scorecard): return (json.dumps(scorecard, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
def write_scorecard(output_dir: Path, body: bytes, repository_root: Path): return common.write_scorecard(output_dir, body, repository_root)


def run_cache_measurement(evaluation_cache: Path, output_dir: Path, repository_root: Path) -> None:
    from scripts.run_v5_b_candidate_ranker import _raw_cache, validate_evaluation_cache
    universe_path = repository_root / "V4_UNIVERSE.csv"; validation = validate_evaluation_cache(evaluation_cache); manifest, prices, splits = _raw_cache(evaluation_cache); universe = normalize_universe(pd.read_csv(universe_path)); provenance = common.git_provenance(repository_root, universe_path, validation); provenance["payload_hash_list_sha256"] = manifest["payload_hash_list_sha256"]
    first = canonical_scorecard_bytes(build_scorecard(prices, universe, splits, provenance, provenance["repository_exact_sha"])); second = canonical_scorecard_bytes(build_scorecard(prices, universe, splits, provenance, provenance["repository_exact_sha"]))
    if first != second: raise ValueError("TWO_PASS_SCORECARD_MISMATCH")
    write_scorecard(output_dir, first, repository_root)
