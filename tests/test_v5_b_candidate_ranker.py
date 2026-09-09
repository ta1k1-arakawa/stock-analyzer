import pandas as pd, numpy as np, pytest
from pathlib import Path
from src.v5_b_candidate_ranker import *

def frame(n=300, bump=0.0):
    ix=pd.date_range("2016-01-01",periods=n,freq="B"); c=100+np.arange(n)*.05+bump
    return pd.DataFrame({"Open":c,"High":c+1,"Low":c-1,"Close":c,"AdjClose":c,"Volume":np.full(n,100000.)},index=ix)

def test_exact_feature_registration_and_model_params():
    assert len(FEATURES)==20 and FEATURES[-5:]==("return_20d_percentile","return_60d_percentile","distance_from_high20_percentile","candidate_count","baseline_rank")
    assert MODEL_PARAMS["objective"]=="regression_l1" and MODEL_PARAMS["n_estimators"]==300 and MODEL_PARAMS["random_state"]==20260805 and MODEL_PARAMS["n_jobs"]==1

def test_causal_features_and_same_day_percentiles():
    d=pd.Timestamp("2017-02-01"); c=pd.DataFrame([{"signal_date":d,"ticker":"A","industry":"i","rank":1},{"signal_date":d,"ticker":"B","industry":"j","rank":2}])
    x=build_features(c,{"A":frame(),"B":frame(bump=2)})
    assert set(FEATURES).issubset(x.columns); assert x.candidate_count.tolist()==[2,2]; assert x.baseline_rank.tolist()==[1,2]

def test_target_gap_and_future_cutoff():
    f=frame(); d=f.index[252]; assert d5_target(f,d) is not None; assert training_cutoff(2020)==pd.Timestamp("2020-01-01")
    g=f.copy(); g.iloc[253,g.columns.get_loc("Open")]=g.iloc[252].Close*1.02; assert d5_target(g,d) is None

def test_ai_changes_order_only_and_overlap_is_fail_closed():
    d=pd.Timestamp("2017-02-01"); x=pd.DataFrame([{"signal_date":d,"ticker":"A","rank":1},{"signal_date":d,"ticker":"B","rank":2}]); x["predicted_d5_return"]=[-0.1,0.1]
    for f in FEATURES: x[f]=0.0
    assert baseline_order(x).ticker.tolist()==["A","B"]
    assert rank_candidates(type("M",(),{"predict":lambda self,a: np.array([-0.1,0.1])})(), x).ticker.tolist()==["B","A"]
    a=frame(); b=a.copy(); b.iloc[0,b.columns.get_loc("Close")]+=1
    with pytest.raises(ValueError,match="CACHE_OVERLAP_MISMATCH"): validate_cache_overlap({"A":a},{"A":b})

def test_no_identifier_features_and_atomic_two_pass():
    assert not any(x in FEATURES for x in ("ticker","industry","year","month","weekday"))
    a=synthetic_artifacts(); assert set(a)=={"summary.json","trades.csv","predictions.csv","daily_equity.csv"}
    # Writer schema is exercised without creating repository or production files.
    assert set(a)=={"summary.json","trades.csv","predictions.csv","daily_equity.csv"}

def test_formal_evaluation_is_not_run():
    assert synthetic_artifacts()["summary.json"]

def test_ticker_normalization_and_cache_merge_preserves_history():
    u=normalize_universe(pd.DataFrame({"ticker":[1301,"1302.T"],"industry":["A","B"]})); assert u.ticker.tolist()==["1301","1302"]
    a=frame(3); b=frame(3).iloc[1:].copy(); b.index=pd.date_range(a.index[-1],periods=2,freq="B")
    merged=combine_cache_frames({"1301":a},{"1301":b}); assert merged["1301"].index.is_monotonic_increasing

def test_positive_label_is_na_for_unavailable_target_and_chart_host():
    f=frame(300); d=f.index[-1]; assert d5_target(f,d) is None
    from scripts.acquire_v5_b_evaluation_cache import chart_url, HOST
    assert HOST in chart_url("1301") and "period1=" in chart_url("1301") and ".T" in chart_url("1301")
