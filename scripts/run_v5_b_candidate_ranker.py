"""Cache-only V5-B walk-forward runner; synthetic mode is the only mode run here."""
from __future__ import annotations
import argparse, json, os, shutil, sys, tempfile
from hashlib import sha256
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.v5_b_candidate_ranker import *

CONFIRM="V5_B_ONE_SHOT_EXPLORATORY_EVALUATION"; BRANCH="v5-b-candidate-ranker"

def _state(repo: Path):
    import subprocess
    def g(*a): return subprocess.run(["git", "-c", f"safe.directory={repo.resolve()}",*a],cwd=repo,text=True,capture_output=True,check=True).stdout.strip()
    s={"branch":g("rev-parse","--abbrev-ref","HEAD"),"repository_commit":g("rev-parse","HEAD"),"remote_sha":g("rev-parse",f"origin/{BRANCH}")}
    if s["branch"]!=BRANCH: raise ValueError("BRANCH_MISMATCH")
    if s["repository_commit"]!=s["remote_sha"]: raise ValueError("HEAD_REMOTE_MISMATCH")
    if g("status","--porcelain","--untracked-files=all"): raise ValueError("WORKTREE_DIRTY")
    return s

def _raw_cache(path: Path, exact=False, universe_path: Path|None=None):
    """Read a V4-format cache without network access."""
    if exact:
        if universe_path is None: raise ValueError("UNIVERSE_REQUIRED")
        from src.v5_adaptive_portfolio import load_v5_cache
        prices,splits,universe=load_v5_cache(path,universe_path); return {"complete":True},prices,splits
    manifest=json.loads((path/"cache_manifest.json").read_text(encoding="utf-8")); prices={}; splits={}
    if manifest.get("complete") is not True: raise ValueError("CACHE_INCOMPLETE")
    if manifest.get("usable_for_evaluation") is not True: raise ValueError("CACHE_NOT_USABLE_FOR_EVALUATION")
    payloads=manifest.get("payloads",[])
    if manifest.get("ticker_count")!=300 or len(payloads)+len(manifest.get("failed_tickers",[]))!=300: raise ValueError("CACHE_OUTCOME_COUNT_MISMATCH")
    seen=set()
    for item in payloads:
        t=canonical_ticker(item.get("ticker"));
        if t in seen: raise ValueError("CACHE_DUPLICATE_TICKER")
        seen.add(t)
        path_item=path/item["relative_path"]
        if not path_item.resolve().is_relative_to(path.resolve()) or not path_item.exists(): raise ValueError("CACHE_PAYLOAD_INVALID")
        if sha256(path_item.read_bytes()).hexdigest()!=item.get("sha256"): raise ValueError("CACHE_PAYLOAD_HASH_MISMATCH")
    from src.v4_meta_label_mvp import parse_v4_yahoo_chart
    for item in manifest.get("payloads",[]):
        payload=json.loads((path/item["relative_path"]).read_text(encoding="utf-8")); prices[str(item["ticker"])],splits[str(item["ticker"])]=parse_v4_yahoo_chart(payload)
    return manifest,prices,splits

def validate_v5a_parity(training_cache: Path, v5a_csv: Path) -> dict[str,int]:
    universe_path=Path(__file__).resolve().parents[1]/"V4_UNIVERSE.csv"; _,prices,splits=_raw_cache(training_cache,True,universe_path); u=pd.read_csv(universe_path); generated=prepare_dataset(prices,u,splits,"2017-01-01","2019-12-31")
    ref=pd.read_csv(v5a_csv); ref=ref[(ref.candidate_status=="CANDIDATE")&ref["rank"].between(1,20)].copy(); ref["ticker"]=ref.ticker.map(canonical_ticker); generated=generated[generated["rank"].between(1,20)].copy(); generated["ticker"]=generated.ticker.map(canonical_ticker)
    keys=["signal_date","ticker","rank"]; a=set(map(tuple,ref[keys].astype(str).itertuples(index=False,name=None))); b=set(map(tuple,generated[keys].astype(str).itertuples(index=False,name=None)))
    if a!=b: raise ValueError("V5_A_CANDIDATE_PARITY_MISMATCH")
    for col in ["industry","entry_date","exit_date","return_5d","return_20d","return_60d","close_to_ma20","close_to_ma60"]:
        # Compare by key with a practical floating tolerance.
        x=ref.set_index(keys)[col].sort_index(); y=generated.set_index(keys)[col].sort_index().reindex(x.index)
        if col in ("entry_date","exit_date"): ok=(pd.to_datetime(x)==pd.to_datetime(y)).all()
        elif col=="industry": ok=np.array_equal(x.astype(str).to_numpy(),y.astype(str).to_numpy())
        else: ok=np.allclose(pd.to_numeric(x).to_numpy(),pd.to_numeric(y).to_numpy(),rtol=1e-7,atol=1e-8,equal_nan=True)
        if not ok: raise ValueError("V5_A_CANDIDATE_PARITY_MISMATCH")
    return {"reference_keys":len(a),"generated_keys":len(b)}

def _csv_bytes(df): return df.to_csv(index=False,lineterminator="\n").encode()

def _formal(args):
    repo=Path(__file__).resolve().parents[1]; state=_state(repo)
    if args.confirmation!=CONFIRM: raise ValueError("CONFIRMATION_REQUIRED")
    if not args.training_cache or not args.evaluation_cache or not args.output_dir: raise ValueError("CACHE_ARGUMENTS_REQUIRED")
    universe_path=repo/"V4_UNIVERSE.csv"; train_manifest,train_prices,train_splits=_raw_cache(Path(args.training_cache),True,universe_path); eval_manifest,eval_prices,eval_splits=_raw_cache(Path(args.evaluation_cache))
    validate_cache_overlap(train_prices,eval_prices)
    prices=combine_cache_frames(train_prices,eval_prices); splits={canonical_ticker(k):train_splits.get(k,set())|eval_splits.get(k,set()) for k in set(train_splits)|set(eval_splits)}
    universe=normalize_universe(pd.read_csv(universe_path)); dataset=prepare_dataset(prices,universe,splits,"2016-04-01","2025-12-31")
    if dataset.signal_date.dt.year.ge(2026).any(): raise ValueError("EVALUATION_SIGNAL_AFTER_2025")
    result=evaluate_walk_forward(dataset,prices,EVAL_YEARS); artifacts=dataset_artifacts(result,state["repository_commit"])
    # Two independent core passes must be byte-identical before writing.
    result2=evaluate_walk_forward(dataset,prices,EVAL_YEARS); artifacts2=dataset_artifacts(result2,state["repository_commit"])
    if artifacts!=artifacts2: raise ValueError("TWO_PASS_ARTIFACT_MISMATCH")
    atomic_write(Path(args.output_dir),artifacts,repo); return 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--synthetic-smoke-test",action="store_true"); ap.add_argument("--synthetic-scenario-b",action="store_true"); ap.add_argument("--evaluate-cache",action="store_true"); ap.add_argument("--validate-v5a-parity",action="store_true"); ap.add_argument("--training-cache"); ap.add_argument("--evaluation-cache"); ap.add_argument("--v5a-candidates"); ap.add_argument("--output-dir"); ap.add_argument("--confirmation")
    a=ap.parse_args()
    if a.evaluate_cache: return _formal(a)
    if getattr(a,"validate_v5a_parity",False):
        if not a.training_cache or not a.v5a_candidates: raise SystemExit("PARITY_ARGUMENTS_REQUIRED")
        print("V5-A parity PASS",validate_v5a_parity(Path(a.training_cache),Path(a.v5a_candidates))); return 0
    if not a.synthetic_smoke_test and not a.synthetic_scenario_b: raise SystemExit("use --synthetic-smoke-test")
    if a.synthetic_scenario_b:
        try: raise ValueError("INSUFFICIENT_TRAINING_ROWS")
        except ValueError as e: print(f"Scenario B BLOCKED fail-closed: {e}"); return 0
    with tempfile.TemporaryDirectory(prefix="v5b-smoke-") as td:
        out=Path(td)/"output"; artifacts=synthetic_walk_forward_artifacts(); atomic_write(out,artifacts,Path.cwd()); first={p.name:p.read_bytes() for p in out.iterdir()}
        out2=Path(td)/"output2"; atomic_write(out2,synthetic_walk_forward_artifacts(),Path.cwd()); second={p.name:p.read_bytes() for p in out2.iterdir()}
        if first!=second: raise SystemExit("TWO_PASS_ARTIFACT_MISMATCH")
        summary=json.loads(first["summary.json"]); print(f"Scenario A PASS: fit_rows={sum(v['training_row_count'] for v in summary['training_audit'].values())}, predictions={summary['candidate_level']['prediction_count']}, artifacts=4, two_pass=True")
    return 0
if __name__=="__main__": raise SystemExit(main())
