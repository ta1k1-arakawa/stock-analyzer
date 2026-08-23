from __future__ import annotations
import inspect
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import src.v5_b_candidate_ranker as v5
import src.v8k_layer_a_pullback_volume_dryup as dryup
from src.v5_b_candidate_ranker import BASELINE_ARM, generate_candidates, simulate_portfolio

def _frame(slope, periods=290):
    dates=pd.bdate_range("2019-01-01",periods=periods); close=100+slope*np.arange(periods); close[270]=close[265]*.99
    return pd.DataFrame({"Open":close,"High":close*1.01,"Low":close*.99,"Close":close,"Adj Close":close,"Volume":1000000+np.arange(periods)*100},index=dates)
def _inputs(count=25):
    prices={str(1000+i):_frame(.12+.012*i) for i in range(count)}; universe=pd.DataFrame({"ticker":list(prices),"industry":["A" if i<count//2 else "B" for i in range(count)]}); return prices,universe,next(iter(prices.values())).index[270]

def test_baseline_parity_and_volume_ratio_reference_and_score():
    prices,u,d=_inputs(); eligible=dryup.generate_eligible_candidates(prices,u,signal_from=d,signal_to=d); expected=generate_candidates(prices,u,signal_from=d,signal_to=d); assert list(dryup.rank_baseline(eligible)[["ticker","baseline_rank"]].itertuples(index=False,name=None))==list(expected[["ticker","rank"]].itertuples(index=False,name=None))
    scored=dryup.attach_volume_dryup_scores(eligible,prices); row=scored.iloc[0]; reference=v5._one_features(prices[row.ticker],row.signal_date)["volume_ratio_5_20"]
    assert row.volume_ratio_5_20==pytest.approx(reference); assert row.volume_dryup_score==pytest.approx(1-reference)

def test_unavailable_future_outcome_and_tie_semantics():
    prices,u,d=_inputs(3); eligible=dryup.generate_eligible_candidates(prices,u,signal_from=d,signal_to=d).iloc[:3].copy(); prices[eligible.iloc[0].ticker].loc[d,"Volume"]=0; prices[eligible.iloc[1].ticker].loc[:d,"Volume"]=np.nan; scored=dryup.attach_volume_dryup_scores(eligible,prices); assert "SCORE_UNAVAILABLE" in set(scored.volume_dryup_status)
    prices,u,d=_inputs(); eligible=dryup.generate_eligible_candidates(prices,u,signal_from=d,signal_to=d); original=dryup.attach_volume_dryup_scores(eligible,prices); changed={k:v.copy() for k,v in prices.items()}
    for f in changed.values(): f.loc[f.index>d,"Volume"]*=9
    altered=dryup.attach_volume_dryup_scores(eligible.assign(outcome=1),changed); pd.testing.assert_frame_equal(original[["volume_dryup_score","volume_dryup_status"]],altered[["volume_dryup_score","volume_dryup_status"]])
    manual=pd.DataFrame({"signal_date":pd.Timestamp("2020-01-15"),"ticker":[f"T{i:02d}" for i in range(25)],"return_60d":list(range(25)),"return_20d":.1,"volume_dryup_score":list(reversed(range(25))),"volume_dryup_status":"SCORE_AVAILABLE"}); manual.loc[:1,["volume_dryup_score","return_60d","return_20d"]]=100; ranked=dryup.rank_volume_dryup(manual); assert list(ranked.ticker[:2])==["T00","T01"]; assert set(ranked.ticker)!=set(dryup.rank_baseline(manual).ticker)

def test_execution_d5_reuse_diagnostics_and_determinism(monkeypatch):
    prices,u,d=_inputs(); eligible,base,rows=dryup.build_ranked_arms(prices,u,signal_from=d,signal_to=d); direct=simulate_portfolio(base,prices,BASELINE_ARM); actual=dryup.execute_arms(base,rows,prices); pd.testing.assert_frame_equal(direct[0],actual[0]); pd.testing.assert_frame_equal(direct[1],actual[1])
    frames=dryup.common._normalized_price_frames(prices); state=dryup._realized_d5_state(eligible,frames); key=(pd.Timestamp(eligible.iloc[0].signal_date),eligible.iloc[0].ticker); assert state[key]==pytest.approx(v5.d5_target(prices[key[1]],key[0]))
    calls=[]; original=dryup.common._as_frame
    def counted(frame): calls.append(id(frame)); return original(frame)
    monkeypatch.setattr(dryup.common,"_as_frame",counted); first=dryup.canonical_scorecard_bytes(dryup.build_scorecard(prices,u,provenance={"safe":1})); second=dryup.canonical_scorecard_bytes(dryup.build_scorecard(prices,u,provenance={"safe":1})); assert first==second; assert len(calls)==2*len(prices)
    card=dryup.build_scorecard(prices,u,provenance={"safe":1}); assert sum(x["count"] for x in card["all_eligible_discrimination"]["pooled_score_quintiles"].values())==card["all_eligible_discrimination"]["valid_row_count"]

def test_mechanisms_output_guards_and_no_network(tmp_path: Path):
    d=pd.Timestamp("2020-01-01"); base=pd.DataFrame({"signal_date":[d,d],"ticker":["A","B"]}); var=pd.DataFrame({"signal_date":[d,d],"ticker":["B","C"]}); assert dryup.top20_mechanism(base,var)["overall_jaccard"]==pytest.approx(1/3)
    trades=pd.DataFrame({"evaluation_year":[2020],"signal_date":[d],"ticker":["A"],"status":["FILLED"],"realized_net_profit_yen":[1.]}); assert dryup.fill_mechanism(trades,trades)["common_fills"]==1
    with pytest.raises(ValueError): dryup.write_scorecard(Path.cwd()/"dryup",b"{}",Path.cwd())
    output=tmp_path/"out"; dryup.write_scorecard(output,b"{}",Path.cwd()); assert [x.name for x in output.iterdir()]==["scorecard.json"]
    prices,u,_=_inputs()
    with pytest.raises(ValueError): dryup.generate_eligible_candidates(prices,u,signal_from="2026-01-01",signal_to="2026-01-01")
    assert "requests" not in inspect.getsource(dryup) and "urllib" not in inspect.getsource(dryup)
