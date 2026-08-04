from __future__ import annotations

import argparse, json, shutil, tempfile
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).parents[1]))
from src.v5_adaptive_portfolio import build_artifacts, atomic_write_artifacts, repository_state, run_two_pass_formal_evaluation

def _prices(dates: pd.DatetimeIndex, base: float, *, stop: bool=False, gap: bool=False) -> pd.DataFrame:
    trend=base+np.arange(len(dates))*1.0
    # A modest final pullback creates the V5 signal while remaining above MA60.
    close=trend.copy(); close[-6:]-=np.array([0,8,16,24,32,40],dtype=float)
    open_=close.copy(); high=close+6; low=close-6
    if stop: low[-4]=close[-4]*.90
    if gap: open_[-5]=close[-6]*1.02
    high=np.maximum(high,open_+1); low=np.minimum(low,open_-1)
    return pd.DataFrame({"Open":open_,"High":high,"Low":low,"Close":close,"Adj Close":close,"Volume":np.repeat(200_000.,len(dates))},index=dates)

def smoke() -> None:
    root=Path(tempfile.mkdtemp(prefix="v5-adaptive-smoke-"))
    try:
        dates=pd.date_range("2016-01-01",periods=330,freq="B")
        universe=pd.DataFrame({"ticker":["1001","1002","1003","1004"],"market":["M"]*4,"industry":["A","B","C","D"]})
        prices={"1001":_prices(dates,1800,stop=True),"1002":_prices(dates,800),"1003":_prices(dates,1000,gap=True),"1004":_prices(dates,1500)}
        artifacts=build_artifacts(prices,universe,{ticker:set() for ticker in prices})
        again=build_artifacts(prices,universe,{ticker:set() for ticker in prices})
        assert artifacts==again and set(artifacts)=={"summary.json","trades.csv","candidates.csv","daily_equity.csv"}
        out=root/"a"; atomic_write_artifacts(out,artifacts,Path.cwd())
        assert {p.name for p in out.iterdir()}==set(artifacts)
        trades=pd.read_csv(out/"trades.csv"); assert (trades["quantity"].dropna()%100==0).all()
        # Scenario B: all source rows are too short; writer still produces headers.
        short_dates=pd.date_range("2019-01-01",periods=30,freq="B"); short_universe=universe.iloc[:1].copy(); short={"1001":_prices(short_dates,1000)}
        b=build_artifacts(short,short_universe,{"1001":set()}); outb=root/"b"; atomic_write_artifacts(outb,b,Path.cwd())
        summary=json.loads(b["summary.json"]); assert summary["verdict"] in {"V5_ADAPTIVE_BASELINE_BLOCKED","V5_ADAPTIVE_BASELINE_NOT_PROMISING"}
        assert b["trades.csv"].count(b"\n")==1 and b["candidates.csv"].count(b"\n")>=1 and len(list(outb.iterdir()))==4
        print("V5 synthetic smoke passed: Scenario A artifacts=4 deterministic=true; Scenario B header-only trades=1 verdict="+summary["verdict"])
    finally:
        shutil.rmtree(root,ignore_errors=True)

def main(argv=None) -> int:
    p=argparse.ArgumentParser(); mode=p.add_mutually_exclusive_group(required=True); mode.add_argument("--synthetic-smoke-test",action="store_true"); mode.add_argument("--evaluate-cache",action="store_true")
    p.add_argument("--cache-dir"); p.add_argument("--output-dir"); p.add_argument("--confirmation"); a=p.parse_args(argv)
    if a.synthetic_smoke_test: smoke(); return 0
    if a.confirmation!="V5_A_ONE_SHOT_CACHE_EVALUATION" or not a.cache_dir or not a.output_dir: p.error("--evaluate-cache requires exact confirmation, --cache-dir, and --output-dir")
    repo=Path(__file__).parents[1]; state=repository_state(repo)
    artifacts=run_two_pass_formal_evaluation(Path(a.cache_dir),Path(a.output_dir),repo/"V4_UNIVERSE.csv",repo,state)
    print("V5_FORMAL_CACHE_ONLY_EVALUATION_COMPLETE artifacts="+str(len(artifacts))); return 0
if __name__=="__main__": raise SystemExit(main())
