from __future__ import annotations
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import src.v5_b_candidate_ranker as v5
import src.v8k_layer_a_trend_persistence_10d as trend
from src.v5_b_candidate_ranker import BASELINE_ARM,generate_candidates,simulate_portfolio
def _frame(s,periods=290):
 d=pd.bdate_range("2019-01-01",periods=periods);c=100+s*np.arange(periods);c[270]=c[265]*.99;return pd.DataFrame({"Open":c,"High":c*1.01,"Low":c*.99,"Close":c,"Adj Close":c,"Volume":np.full(periods,1000000)},index=d)
def _inputs(n=25):
 p={str(1000+i):_frame(.12+.012*i) for i in range(n)};u=pd.DataFrame({"ticker":list(p),"industry":["A" if i<n//2 else "B" for i in range(n)]});return p,u,next(iter(p.values())).index[270]
def test_baseline_and_score_reference_and_zero_semantics():
 p,u,d=_inputs();e=trend.generate_eligible_candidates(p,u,signal_from=d,signal_to=d);x=generate_candidates(p,u,signal_from=d,signal_to=d);assert list(trend.rank_baseline(e)[["ticker","baseline_rank"]].itertuples(index=False,name=None))==list(x[["ticker","rank"]].itertuples(index=False,name=None));s=trend.attach_trend_persistence_scores(e,p);r=s.iloc[0];assert r.trend_persistence_score==pytest.approx(v5._one_features(p[r.ticker],r.signal_date)["up_day_fraction_10"])
 f=p[r.ticker].copy();i=f.index.get_loc(d);f.iloc[i,f.columns.get_loc("Adj Close")]=f.iloc[i-1].loc["Adj Close"];m=trend.attach_trend_persistence_scores(pd.DataFrame([r]),{r.ticker:f});assert m.iloc[0].trend_persistence_score==pytest.approx(0.9)
def test_unavailable_causality_ties_and_difference():
 p,u,d=_inputs(3);e=trend.generate_eligible_candidates(p,u,signal_from=d,signal_to=d).iloc[:1].copy();short=p[e.iloc[0].ticker].iloc[:10];assert trend.attach_trend_persistence_scores(e,{e.iloc[0].ticker:short}).iloc[0].trend_persistence_status=="SCORE_UNAVAILABLE"
 p,u,d=_inputs();e=trend.generate_eligible_candidates(p,u,signal_from=d,signal_to=d);a=trend.attach_trend_persistence_scores(e,p);q={k:v.copy() for k,v in p.items()}
 for f in q.values():f.loc[f.index>d,"Adj Close"]*=9
 b=trend.attach_trend_persistence_scores(e.assign(outcome=1),q);pd.testing.assert_frame_equal(a[["trend_persistence_score"]],b[["trend_persistence_score"]])
 m=pd.DataFrame({"signal_date":pd.Timestamp("2020-01-15"),"ticker":[f"T{i:02d}"for i in range(25)],"return_60d":list(range(25)),"return_20d":.1,"trend_persistence_score":list(reversed(range(25))),"trend_persistence_status":"SCORE_AVAILABLE"});m.loc[:1,["trend_persistence_score","return_60d","return_20d"]]=100;z=trend.rank_trend_persistence(m);assert list(z.ticker[:2])==["T00","T01"] and set(z.ticker)!=set(trend.rank_baseline(m).ticker)
def test_execution_outcomes_diagnostics_and_output(tmp_path:Path,monkeypatch):
 p,u,d=_inputs();e,base,var=trend.build_ranked_arms(p,u,signal_from=d,signal_to=d);direct=simulate_portfolio(base,p,BASELINE_ARM);actual=trend.execute_arms(base,var,p);pd.testing.assert_frame_equal(direct[0],actual[0]);frames=trend.common._normalized_price_frames(p);state=trend._realized_d5_state(e,frames);key=(pd.Timestamp(e.iloc[0].signal_date),e.iloc[0].ticker);assert state[key]==pytest.approx(v5.d5_target(p[key[1]],key[0]))
 calls=[];old=trend.common._as_frame
 def counted(f):calls.append(id(f));return old(f)
 monkeypatch.setattr(trend.common,"_as_frame",counted);first=trend.canonical_scorecard_bytes(trend.build_scorecard(p,u,provenance={"x":1}));second=trend.canonical_scorecard_bytes(trend.build_scorecard(p,u,provenance={"x":1}));assert first==second and len(calls)==2*len(p)
 card=trend.build_scorecard(p,u,provenance={"x":1});assert sum(x["count"]for x in card["all_eligible_discrimination"]["pooled_score_quintiles"].values())==card["all_eligible_discrimination"]["valid_row_count"]
 with pytest.raises(ValueError):trend.write_scorecard(Path.cwd()/"trend",b"{}",Path.cwd())
 out=tmp_path/"o";trend.write_scorecard(out,b"{}",Path.cwd());assert [x.name for x in out.iterdir()]==["scorecard.json"]
 with pytest.raises(ValueError):trend.generate_eligible_candidates(p,u,signal_from="2026-01-01",signal_to="2026-01-01")
 assert "requests"not in inspect.getsource(trend) and "urllib"not in inspect.getsource(trend)
