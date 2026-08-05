from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.v5_a2_fixed100_stop_study import *
from src.v5_a2_fixed100_stop_study import _d5_execution, _run_arm
from src.v5_adaptive_portfolio import _frame
from scripts.run_v5_a2_fixed100_stop_study import _prices

def sample():
    d=pd.date_range('2015-01-01','2019-12-31',freq='B'); u=pd.DataFrame({'ticker':['1001','1002','1003'],'market':['M']*3,'industry':['A','B','C']}); p={'1001':_prices(d,1800,stop=True),'1002':_prices(d,800),'1003':_prices(d,1000,gap=True)}; return p,u,{k:set() for k in p}

def test_both_arms_have_same_candidates_and_quantity_100():
    p,u,s=sample(); a=run_study(p,u,s); tr=pd.read_csv(pd.io.common.BytesIO(a['trades.csv'])); f=tr[tr.status.eq('FILLED')]; assert set(f.quantity)=={100}; assert '200' not in set(f.quantity.astype(str)); assert json.loads(a['summary.json'])['comparison']['candidate_byte_identical']

def test_arm_s_has_stop_and_arm_d_time_only():
    p,u,s=sample(); tr=pd.read_csv(pd.io.common.BytesIO(run_study(p,u,s)['trades.csv'])); assert set(tr[(tr.arm==ARM_D5)&tr.status.eq('FILLED')].exit_reason)=={'TIME'}; assert set(tr[(tr.arm==ARM_STOP)&tr.status.eq('FILLED')].exit_reason)<= {'STOP','GAP_STOP','TIME'}; assert (tr.quantity.dropna()%100==0).all()

def test_d5_holds_through_stop_level():
    d=pd.date_range('2017-01-02',periods=270,freq='B'); p=_prices(d,1000,stop=True); signal=d[-8]; ex=_d5_execution(_frame(p),signal); assert ex['exit_reason']=='TIME' and ex['exit_date']==d[-3]

def test_portfolio_limits_and_audits():
    p,u,s=sample(); a=run_study(p,u,s); tr=pd.read_csv(pd.io.common.BytesIO(a['trades.csv'])); eq=pd.read_csv(pd.io.common.BytesIO(a['daily_equity.csv'])); sm=json.loads(a['summary.json']);
    assert eq.open_positions.max()<=2 and (eq.available_cash>=40000-1e-8).all(); assert all(v==0 for arm in sm['arms'].values() for v in arm['aggregate']['safety_audit'].values())

def test_mtm_and_book_equity_present_and_distinct():
    p,u,s=sample(); eq=pd.read_csv(pd.io.common.BytesIO(run_study(p,u,s)['daily_equity.csv'])); assert {'raw_close_market_value','book_equity','mark_to_market_equity'}<=set(eq.columns); assert (eq.book_equity!=eq.mark_to_market_equity).any()

def test_gate_and_comparison_schema():
    p,u,s=sample(); a=run_study(p,u,s); sm=json.loads(a['summary.json']); assert sm['exploratory_only'] and not sm['unused_holdout'] and not sm['deployment_allowed'] and not sm['ai_used']; assert {'common_filled_count','stop_only_filled_count','d5_only_filled_count'}<=set(sm['comparison']); assert set(a)=={'summary.json','trades.csv','daily_equity.csv','comparison.csv'}

def test_two_pass_byte_identical_and_writer(tmp_path):
    p,u,s=sample(); a=run_study(p,u,s); assert a==run_study(p,u,s); atomic_write_artifacts(tmp_path/'out',a,Path.cwd()); assert len(list((tmp_path/'out').iterdir()))==4

def test_writer_mismatch_leaves_zero_artifacts(tmp_path):
    with pytest.raises(ValueError): atomic_write_artifacts(tmp_path/'out',{'summary.json':b'x'},Path.cwd()); assert not (tmp_path/'out').exists()

def test_scenario_b_header_only():
    d=pd.date_range('2019-01-01',periods=30,freq='B'); u=pd.DataFrame({'ticker':['1001'],'market':['M'],'industry':['A']}); a=run_study({'1001':_prices(d,1000)},u,{'1001':set()}); tr=pd.read_csv(pd.io.common.BytesIO(a['trades.csv'])); assert tr.empty; assert json.loads(a['summary.json'])['verdict'].startswith('V5_A2_')

def test_entry_gap_rejection_is_shared():
    p,u,s=sample(); tr=pd.read_csv(pd.io.common.BytesIO(run_study(p,u,s)['trades.csv'])); assert tr[tr.skip_reason.eq('ENTRY_GAP_TOO_HIGH')].arm.nunique()==2

def test_cross_year_calendar_closes_position():
    p,u,s=sample(); c=build_candidates(p,u,s); c=c[(c.fold==1)&c.candidate_status.eq('CANDIDATE')&(pd.to_datetime(c.signal_date).dt.month==12)].sort_values('signal_date').tail(1).copy(); assert len(c)==1; idx=p['1001'].index; signal=pd.Timestamp('2017-12-29'); c.loc[:,['signal_date','entry_date','exit_date']]=[signal,idx[idx.get_loc(signal)+1],idx[idx.get_loc(signal)+5]]; o,e=_run_arm(ARM_D5,c,p); f=o[o.status.eq('FILLED')]; assert len(f)==1 and pd.Timestamp(f.iloc[0].exit_date).year==2018 and e.open_positions.iloc[-1]==0

def test_arm_d_never_has_stop_rows():
    p,u,s=sample(); tr=pd.read_csv(pd.io.common.BytesIO(run_study(p,u,s)['trades.csv'])); assert not tr[(tr.arm==ARM_D5)&tr.exit_reason.isin(['STOP','GAP_STOP'])].shape[0]
