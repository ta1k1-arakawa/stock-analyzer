"""Offline V8K Layer A ten-observation trend-persistence measurement."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
import src.v8k_layer_a_volatility_adjusted_momentum as common
from src.v5_b_candidate_ranker import canonical_ticker, normalize_universe

SCHEMA_VERSION="V8K_LAYER_A_TREND_PERSISTENCE_10D_SCORECARD_V1"
MAX_CANDIDATES=common.MAX_CANDIDATES
def generate_eligible_candidates(*args,**kwargs): return common.generate_eligible_candidates(*args,**kwargs)
def rank_baseline(eligible): return common.rank_baseline(eligible)

def attach_trend_persistence_scores(eligible:pd.DataFrame,prices:Mapping[str,pd.DataFrame],_normalized_frames=None)->pd.DataFrame:
    frames=dict(_normalized_frames) if _normalized_frames is not None else common._normalized_price_frames(prices); out=eligible.copy(); scores=[]; statuses=[]
    for _,row in out.iterrows():
        try:
            frame=frames[canonical_ticker(row.ticker)]; position=frame.index.get_loc(pd.Timestamp(row.signal_date))
            if not isinstance(position,(int,np.integer)) or position<10: raise KeyError
            returns=frame.AdjClose.astype(float).pct_change().iloc[position-9:position+1]
            if len(returns)!=10 or not np.isfinite(returns).all(): raise ValueError
            score=float((returns>0).mean())
        except (KeyError,ValueError,TypeError): scores.append(np.nan); statuses.append("SCORE_UNAVAILABLE")
        else: scores.append(score); statuses.append("SCORE_AVAILABLE")
    out["trend_persistence_score"]=scores; out["trend_persistence_status"]=statuses; out["risk_adjusted_momentum_score"]=out["trend_persistence_score"]; out["volatility_adjusted_status"]=out["trend_persistence_status"]; return out

def rank_trend_persistence(scored):
    a=scored[scored.trend_persistence_status.eq("SCORE_AVAILABLE")].copy(); a=a.sort_values(["signal_date","trend_persistence_score","return_60d","return_20d","ticker"],ascending=[True,False,False,False,True],kind="mergesort"); a["ai_rank"]=a.groupby("signal_date").cumcount()+1; a["baseline_rank"]=a.ai_rank; return a[a.ai_rank<=MAX_CANDIDATES].reset_index(drop=True)
def build_ranked_arms(prices,universe,splits=None,signal_from="2020-01-01",signal_to="2025-12-31"):
    frames=common._normalized_price_frames(prices); e=generate_eligible_candidates(prices,universe,splits,signal_from,signal_to,frames); return e,rank_baseline(e),rank_trend_persistence(attach_trend_persistence_scores(e,prices,frames))
def execute_arms(base,var,prices): return common.execute_arms(base,var,prices)
def arm_metrics(t,e): return common.arm_metrics(t,e)
def top20_mechanism(a,b): return common.top20_mechanism(a,b)
def fill_mechanism(a,b): return common.fill_mechanism(a,b)
def _normalized_d5_target(f,d): return common._normalized_d5_target(f,d)
def _realized_d5_state(r,f): return common._realized_d5_state(r,f)
def write_scorecard(o,b,r): return common.write_scorecard(o,b,r)

def build_scorecard(prices,universe,splits=None,provenance=None,repository_commit="SYNTHETIC")->dict[str,Any]:
    frames=common._normalized_price_frames(prices); e=generate_eligible_candidates(prices,universe,splits,_normalized_frames=frames); base=rank_baseline(e); scored=attach_trend_persistence_scores(e,prices,frames); var=rank_trend_persistence(scored); outcomes=_realized_d5_state(scored,frames); bt,be,vt,ve=execute_arms(base,var,prices); bm,vm=arm_metrics(bt,be),arm_metrics(vt,ve)
    return {"schema_version":SCHEMA_VERSION,"study":"V8K_HISTORICAL_RESEARCH","layer_a_role":"HYPOTHESIS_GENERATION_AND_VIABILITY_SCREEN","evidence_capacity":"ZERO","exploratory_only":True,"measurement_status":"COMPLETE","interpretation":"GPT_DECISION_REQUIRED","promotion_thresholds_defined":False,"deployment_allowed":False,"future_profitability_established":False,"parameter_neighbor_robustness_status":"NOT_RUN_NO_FREE_PARAMETER_SEARCH","repository_commit":repository_commit,"provenance":dict(provenance or {}),"safe_row_counts":{"eligible_pre_top20":int(len(e)),"baseline_selected":int(len(base)),"variant_selected":int(len(var))},"baseline":bm,"variant":vm,"baseline_vs_variant_difference":common._metric_differences(bm,vm),"all_eligible_discrimination":common._all_eligible_discrimination(scored,outcomes),"selected_discrimination":common._discrimination(var,outcomes),"top20_mechanism":top20_mechanism(base,var),"fill_mechanism":fill_mechanism(bt,vt)}
def canonical_scorecard_bytes(s): return (json.dumps(s,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def run_cache_measurement(evaluation_cache:Path,output_dir:Path,repository_root:Path):
    from scripts.run_v5_b_candidate_ranker import _raw_cache,validate_evaluation_cache
    u=repository_root/"V4_UNIVERSE.csv"; validation=validate_evaluation_cache(evaluation_cache); manifest,prices,splits=_raw_cache(evaluation_cache); universe=normalize_universe(pd.read_csv(u)); prov=common.git_provenance(repository_root,u,validation); prov["payload_hash_list_sha256"]=manifest["payload_hash_list_sha256"]; first=canonical_scorecard_bytes(build_scorecard(prices,universe,splits,prov,prov["repository_exact_sha"])); second=canonical_scorecard_bytes(build_scorecard(prices,universe,splits,prov,prov["repository_exact_sha"]));
    if first!=second: raise ValueError("TWO_PASS_SCORECARD_MISMATCH")
    write_scorecard(output_dir,first,repository_root)
