from __future__ import annotations
import argparse, json, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from src.v5_a2_fixed100_stop_study import run_study, atomic_write_artifacts, run_two_pass
from src.v5_adaptive_portfolio import repository_state

def _prices(dates,base,stop=False,gap=False):
    i=np.arange(len(dates),dtype=float); phase=np.arange(len(dates))%20
    close=base+i*.35-np.where(phase>=15,(phase-14)*base*.008,0)
    op=close.copy(); hi=close+6; lo=close-6
    if stop: lo[phase==16]=close[phase==16]*.90
    if gap: op[phase==16]=np.roll(close,1)[phase==16]*1.02
    hi=np.maximum(hi,op+1); lo=np.minimum(lo,op-1)
    return pd.DataFrame({'Open':op,'High':hi,'Low':lo,'Close':close,'Adj Close':close,'Volume':200000.},index=dates)

def smoke():
    root=Path(tempfile.mkdtemp(prefix='v5-a2-smoke-'))
    try:
        d=pd.date_range('2015-01-01','2019-12-31',freq='B'); u=pd.DataFrame({'ticker':['1001','1002','1003'],'market':['M']*3,'industry':['A','B','C']}); p={'1001':_prices(d,1800,stop=True),'1002':_prices(d,800),'1003':_prices(d,1000,gap=True)}; splits={k:set() for k in p}
        a=run_study(p,u,splits); b=run_study(p,u,splits); assert a==b
        tr=pd.read_csv(pd.io.common.BytesIO(a['trades.csv'])); filled=tr[tr.status=='FILLED']; assert set(filled.quantity)=={100}; assert (filled[filled.arm=='FIXED100_CURRENT_STOP'].exit_reason.isin(['STOP','GAP_STOP','TIME'])).all(); assert set(filled[filled.arm=='FIXED100_D5_ONLY'].exit_reason)=={'TIME'}
        out=root/'a'; atomic_write_artifacts(out,a,Path.cwd()); assert len(list(out.iterdir()))==4
        sd=pd.DataFrame({'ticker':['1001'],'market':['M'],'industry':['A']}); short={'1001':_prices(pd.date_range('2019-01-01',periods=30,freq='B'),1000)}; sb=run_study(short,sd,{'1001':set()}); ob=root/'b'; atomic_write_artifacts(ob,sb,Path.cwd()); ss=json.loads(sb['summary.json']); assert ss['verdict'] in {'V5_A2_NEITHER_EXPLORATORY_SUPPORTED','V5_A2_EXPLORATORY_BLOCKED'}; assert len(list(ob.iterdir()))==4
        print('V5-A2 synthetic smoke passed: Scenario A deterministic=true artifacts=4; Scenario B header-only/zero-trade path passed')
    finally: shutil.rmtree(root,ignore_errors=True)

def main(argv=None):
    p=argparse.ArgumentParser(); g=p.add_mutually_exclusive_group(required=True); g.add_argument('--synthetic-smoke-test',action='store_true'); g.add_argument('--evaluate-cache',action='store_true'); p.add_argument('--cache-dir'); p.add_argument('--output-dir'); p.add_argument('--confirmation'); a=p.parse_args(argv)
    if a.synthetic_smoke_test: smoke(); return 0
    if a.confirmation!='V5_A2_ONE_SHOT_EXPLORATORY_EVALUATION' or not a.cache_dir or not a.output_dir: p.error('exact confirmation, --cache-dir, and --output-dir required')
    repo=Path(__file__).parents[1]; state=repository_state(repo,branch='v5-a2-fixed100-stop-study'); run_two_pass(Path(a.cache_dir),Path(a.output_dir),repo/'V4_UNIVERSE.csv',repo,state); print('V5-A2 formal cache-only artifacts written'); return 0
if __name__=='__main__': raise SystemExit(main())
