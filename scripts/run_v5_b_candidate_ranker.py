"""Cache-only V5-B walk-forward runner; synthetic mode is the only mode run here."""
from __future__ import annotations
import argparse, json, os, shutil, sys, tempfile
from hashlib import sha256
from pathlib import Path
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

def _raw_cache(path: Path):
    """Read a V4-format cache without network access."""
    manifest=json.loads((path/"cache_manifest.json").read_text(encoding="utf-8")); prices={}; splits={}
    from src.v4_meta_label_mvp import parse_v4_yahoo_chart
    for item in manifest.get("payloads",[]):
        payload=json.loads((path/item["relative_path"]).read_text(encoding="utf-8")); prices[str(item["ticker"])],splits[str(item["ticker"])]=parse_v4_yahoo_chart(payload)
    return manifest,prices,splits

def _csv_bytes(df): return df.to_csv(index=False,lineterminator="\n").encode()

def _formal(args):
    repo=Path(__file__).resolve().parents[1]; state=_state(repo)
    if args.confirmation!=CONFIRM: raise ValueError("CONFIRMATION_REQUIRED")
    train_manifest,train_prices,train_splits=_raw_cache(Path(args.training_cache)); eval_manifest,eval_prices,eval_splits=_raw_cache(Path(args.evaluation_cache))
    validate_cache_overlap(train_prices,eval_prices)
    prices={**train_prices,**eval_prices}; splits={k:train_splits.get(k,set())|eval_splits.get(k,set()) for k in set(train_splits)|set(eval_splits)}
    universe_path=repo/"V4_UNIVERSE.csv"; universe=pd.read_csv(universe_path); dataset=prepare_dataset(prices,universe,splits,"2016-04-01","2025-12-31")
    result=evaluate_walk_forward(dataset,prices,EVAL_YEARS); artifacts=dataset_artifacts(result,state["repository_commit"])
    # Two independent core passes must be byte-identical before writing.
    result2=evaluate_walk_forward(dataset,prices,EVAL_YEARS); artifacts2=dataset_artifacts(result2,state["repository_commit"])
    if artifacts!=artifacts2: raise ValueError("TWO_PASS_ARTIFACT_MISMATCH")
    atomic_write(Path(args.output_dir),artifacts,repo); return 0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--synthetic-smoke-test",action="store_true"); ap.add_argument("--synthetic-scenario-b",action="store_true"); ap.add_argument("--evaluate-cache",action="store_true"); ap.add_argument("--training-cache"); ap.add_argument("--evaluation-cache"); ap.add_argument("--output-dir"); ap.add_argument("--confirmation")
    a=ap.parse_args()
    if a.evaluate_cache: return _formal(a)
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
