from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.v5_adaptive_portfolio import *
from src.v5_adaptive_portfolio import _execution, _frame

def raw(n=330, base=1000., *, low_at=None, gap_at=None):
    d=pd.date_range("2016-01-01",periods=n,freq="B"); c=base+np.arange(n,dtype=float); c[-8:]-=np.arange(8)*7
    o=c.copy(); h=c+5; l=c-5
    if low_at is not None: l[low_at]=c[low_at]*.90
    if gap_at is not None: o[gap_at]=c[gap_at-1]*1.02
    h=np.maximum(h,o+1); l=np.minimum(l,o-1)
    return pd.DataFrame({"Open":o,"High":h,"Low":l,"Close":c,"Adj Close":c,"Volume":200_000.},index=d)

def data():
    u=pd.DataFrame({"ticker":["1001","1002","1003"],"market":["M"]*3,"industry":["A","B","A"]})
    p={"1001":raw(base=1800),"1002":raw(base=800),"1003":raw(base=1200)}
    return p,u,{x:set() for x in p}

def test_fixed_constants_and_lots():
    assert (STARTING_CASH,MAX_OPEN_POSITIONS,CASH_RESERVE,MAX_POSITION_YEN,RISK_BUDGET_YEN,LOT_SIZE)==(400000.,2,40000.,220000.,8000.,100)

def test_atr_and_stop_clamp():
    p=_frame(raw()); assert p.atr14.notna().sum()>0
    assert stop_percent(1,1000)==.04 and stop_percent(100,1000)==.08 and .04<=stop_percent(30,1000)<=.08

def test_candidate_ranking_and_top20():
    p,u,s=data(); out=build_candidates(p,u,s); eligible=out.query("candidate_status == 'CANDIDATE'")
    assert eligible.groupby("signal_date").size().max()<=20
    for _,g in eligible.groupby("signal_date"): assert list(g.sort_values("rank").ticker)==list(g.sort_values(["return_60d","return_20d","ticker"],ascending=[False,False,True],kind="mergesort").ticker)

def test_candidate_conditions_and_2020_rejected():
    p,u,s=data(); assert build_candidates(p,u,s).signal_date.max()<=pd.Timestamp("2019-12-31")
    bad=raw(); bad.index=pd.date_range("2019-01-01",periods=len(bad),freq="B")
    with pytest.raises(ValueError,match="PROHIBITED"): _frame(bad)

def test_execution_gap_stop_and_time():
    p=raw(); signal=p.index[-8]; e=_execution(_frame(p),signal,.04); assert e and e["exit_reason"]=="TIME" and e["holding_days"]==5
    q=raw(low_at=-6); e=_execution(_frame(q),q.index[-8],.04); assert e and e["exit_reason"] in {"STOP","GAP_STOP"}
    g=raw(gap_at=-7); assert _execution(_frame(g),g.index[-8],.04)["skip_reason"]=="ENTRY_GAP_TOO_HIGH"

def test_split_spanning_excluded():
    p,u,s=data(); d=raw().index[-6]; out=build_candidates({"1001":p["1001"]},u.iloc[:1],{"1001":{d}}); assert "SPLIT_SPANNING" in set(out.skip_reason.dropna())

def test_portfolio_constraints_and_quantities():
    p,u,s=data(); c=build_candidates(p,u,s); o,e=run_portfolio(c,p); f=o[o.status=="FILLED"]
    assert (f.quantity%100==0).all() and (f.quantity>=100).all() and (f.entry_cost<=220000+1e-8).all()
    assert e.open_positions.max()<=2 and (e.available_cash>=40000-1e-8).all()
    assert not f.duplicated(["fold","ticker","signal_date"]).any()

def test_artifacts_headers_deterministic_and_atomic(tmp_path):
    p,u,s=data(); first=build_artifacts(p,u,s); assert first==build_artifacts(p,u,s)
    atomic_write_artifacts(tmp_path/"output",first,Path.cwd())
    assert {x.name for x in (tmp_path/"output").iterdir()}=={"summary.json","trades.csv","candidates.csv","daily_equity.csv"}
    assert json.loads(first["summary.json"])["ai_used"] is False

def test_empty_scenario_header_only(tmp_path):
    u=pd.DataFrame({"ticker":["1001"],"market":["M"],"industry":["A"]}); p={"1001":raw(30)}; a=build_artifacts(p,u,{"1001":set()})
    assert a["trades.csv"].count(b"\n")==1 and json.loads(a["summary.json"])["verdict"].endswith("NOT_PROMISING")

def test_writer_rejects_repo_and_schema(tmp_path):
    with pytest.raises(ValueError): atomic_write_artifacts(Path.cwd()/"bad",{},Path.cwd())
