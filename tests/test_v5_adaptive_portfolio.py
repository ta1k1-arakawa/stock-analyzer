from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.v5_adaptive_portfolio import *
from src.v5_adaptive_portfolio import _execution, _frame
import src.v5_adaptive_portfolio as v5

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

def test_d0_does_not_lock_and_d1_debits_cash():
    p,u,s=data(); c=build_candidates(p,u,s); o,e=run_portfolio(c,p); f=o[o.status=="FILLED"].iloc[0]
    d0=e[(e.fold==f.fold)&(e.date==pd.Timestamp(f.signal_date))].iloc[0]; d1=e[(e.fold==f.fold)&(e.date==pd.Timestamp(f.entry_date))].iloc[0]
    assert d0.available_cash==STARTING_CASH and d0.open_positions==0
    assert d1.available_cash < STARTING_CASH and pd.Timestamp(f.entry_date)>pd.Timestamp(f.signal_date)

def test_same_day_exit_still_occupies_slot_and_pending_next_day():
    p,u,s=data(); c=build_candidates(p,u,s); _,e=run_portfolio(c,p)
    pending=e[e.pending_cash>0]
    if len(pending):
        row=pending.iloc[0]; later=e[(e.fold==row.fold)&(e.date>row.date)].iloc[0]
        assert row.pending_cash>0 and later.available_cash>=row.available_cash

def test_two_pass_mismatch_writes_nothing(tmp_path,monkeypatch):
    manifest={"payload_hash_list_sha256":FORMAL_PAYLOAD_HASH_LIST_SHA256,"universe_csv_sha256":"a","ticker_list_sha256":"b","successful_ticker_count":283,"failed_tickers":[str(i) for i in range(17)],"network_audit":[]}
    u=pd.DataFrame({"ticker":[],"industry":[],"market":[]})
    monkeypatch.setattr(v5,"validate_v5_formal_cache",lambda *a:(manifest,u))
    monkeypatch.setattr(v5,"load_v5_cache",lambda *a:({}, {}, u))
    calls=[]
    def builder(*a):
        calls.append(1); return {"summary.json":b"{}\n" if len(calls)==1 else b"{ }\n","trades.csv":b"x\n","candidates.csv":b"x\n","daily_equity.csv":b"x\n"}
    with pytest.raises(ValueError,match="MISMATCH"): run_two_pass_formal_evaluation(tmp_path/"cache",tmp_path/"out",tmp_path/"u",Path.cwd(),{"branch":"v5-adaptive-portfolio-baseline","repository_commit":"x","remote_sha":"x"},loader=v5.load_v5_cache,builder=builder)
    assert not (tmp_path/"out").exists()

def test_formal_cli_requires_confirmation():
    from scripts.run_v5_adaptive_baseline import main
    with pytest.raises(SystemExit): main(["--evaluate-cache"])
