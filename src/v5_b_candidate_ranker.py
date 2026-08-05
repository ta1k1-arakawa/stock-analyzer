"""V5-B candidate-ranker registration and offline synthetic harness.

The module deliberately separates feature/label construction from execution.  It
does not download data and never changes the V5-A2 execution parameters.
"""
from __future__ import annotations

import json, os, shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

FEATURES = (
    "return_1d", "return_5d", "return_10d", "return_20d", "return_60d",
    "volatility_20", "downside_volatility_20", "atr14_percent",
    "close_to_ma20", "close_to_ma60", "distance_from_high20",
    "distance_from_high60", "volume_ratio_5_20", "turnover_ratio_5_20",
    "up_day_fraction_10", "return_20d_percentile", "return_60d_percentile",
    "distance_from_high20_percentile", "candidate_count", "baseline_rank",
)
BASELINE_ARM = "BASELINE_RANK"
AI_ARM = "AI_RANK"
QUANTITY = 100
FUTURE_DAYS = 5
MODEL_PARAMS = {
    "objective": "regression_l1", "n_estimators": 300, "learning_rate": 0.03,
    "num_leaves": 15, "max_depth": -1, "min_child_samples": 40,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_alpha": 0.0, "reg_lambda": 1.0, "random_state": 20260805,
    "n_jobs": 1, "deterministic": True, "force_col_wise": True,
    "verbosity": -1,
}
EVAL_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)


def feature_hash() -> str:
    return sha256(json.dumps(FEATURES, separators=(",", ":")).encode()).hexdigest()


def model_hash() -> str:
    return sha256(json.dumps(MODEL_PARAMS, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _frame(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    y.index = pd.to_datetime(y.index).tz_localize(None)
    return y.sort_index()


def _atr14(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev = c.shift(1)
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def _one_features(frame: pd.DataFrame, day: pd.Timestamp) -> dict[str, float]:
    p = _frame(frame).loc[:pd.Timestamp(day)]
    if len(p) < 252: raise ValueError("FEATURE_HISTORY_UNAVAILABLE")
    ac = p["AdjClose"].astype(float); close = p["Close"].astype(float); vol = p["Volume"].astype(float)
    dr = ac.pct_change()
    atr = _atr14(p["High"], p["Low"], ac).iloc[-1]
    def ret(n): return ac.iloc[-1] / ac.iloc[-1-n] - 1.0
    return {
        "return_1d": float(ret(1)), "return_5d": float(ret(5)), "return_10d": float(ret(10)),
        "return_20d": float(ret(20)), "return_60d": float(ret(60)),
        "volatility_20": float(dr.tail(20).std()),
        "downside_volatility_20": float(np.sqrt(np.mean(np.minimum(dr.tail(20).dropna(), 0.0)**2))),
        "atr14_percent": float(atr/ac.iloc[-1]),
        "close_to_ma20": float(ac.iloc[-1]/ac.tail(20).mean()-1),
        "close_to_ma60": float(ac.iloc[-1]/ac.tail(60).mean()-1),
        "distance_from_high20": float(ac.iloc[-1]/ac.tail(20).max()-1),
        "distance_from_high60": float(ac.iloc[-1]/ac.tail(60).max()-1),
        "volume_ratio_5_20": float(vol.tail(5).mean()/vol.tail(20).mean()),
        "turnover_ratio_5_20": float((close*vol).tail(5).mean()/(close*vol).tail(20).mean()),
        "up_day_fraction_10": float((dr.tail(10)>0).mean()),
    }


def build_features(candidates: pd.DataFrame, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build exactly the pre-registered 20 causal features."""
    x = candidates.copy(); x["signal_date"] = pd.to_datetime(x["signal_date"])
    base=[]
    for _, r in x.iterrows():
        f = frames[str(r["ticker"])]
        try: vals = _one_features(f, r.signal_date)
        except (KeyError, ValueError): vals = {k: np.nan for k in FEATURES[:15]}
        vals.update({"signal_date": r.signal_date, "entry_date": r.get("entry_date"), "exit_date": r.get("exit_date"),
                     "ticker": r["ticker"], "industry": r.get("industry", ""), "rank": r["rank"], "baseline_rank": np.nan})
        base.append(vals)
    out = pd.DataFrame(base)
    valid = out.assign(_valid=out["return_20d"].notna()).groupby("signal_date")
    out["candidate_count"] = out.groupby("signal_date")["ticker"].transform("count")
    # percentile ranks are within the same candidate date, ties averaged and normalized.
    for col, name in (("return_20d", "return_20d_percentile"), ("return_60d", "return_60d_percentile"), ("distance_from_high20", "distance_from_high20_percentile")):
        out[name] = out.groupby("signal_date")[col].rank(method="average", pct=True)
    out["baseline_rank"] = out.groupby("signal_date")["rank"].rank(method="average", pct=True, ascending=False)
    return out


def d5_target(frame: pd.DataFrame, signal_date: pd.Timestamp) -> float | None:
    p=_frame(frame); d=pd.Timestamp(signal_date)
    try:
        i=p.index.get_loc(d); e=i+1; z=i+5
        if z>=len(p) or float(p.iloc[e].Open)>float(p.iloc[i].Close)*1.01: return None
        return float(p.iloc[z].Open*.9997/(p.iloc[e].Open*1.0003)-1)
    except (KeyError, IndexError, TypeError): return None


def attach_targets(features: pd.DataFrame, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    out=features.copy(); out["realized_d5_return"]=[d5_target(frames[str(t)], d) for t,d in zip(out.ticker,out.signal_date)]
    out["positive_label"]=(out.realized_d5_return>0).astype("Int64")
    return out


def training_cutoff(evaluation_year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{evaluation_year}-01-01")


def fit_year(train: pd.DataFrame, evaluation_year: int):
    cutoff=training_cutoff(evaluation_year)
    use=train[(pd.to_datetime(train.exit_date)<cutoff) & train.realized_d5_return.notna()].dropna(subset=list(FEATURES))
    if len(use)<1000: raise ValueError("INSUFFICIENT_TRAINING_ROWS")
    from lightgbm import LGBMRegressor
    model=LGBMRegressor(**MODEL_PARAMS)
    model.fit(use.loc[:, FEATURES], use.realized_d5_return)
    return model, len(use), cutoff


def rank_candidates(model, frame: pd.DataFrame) -> pd.DataFrame:
    x=frame.copy(); x["predicted_d5_return"]=model.predict(x.loc[:, FEATURES])
    return x.sort_values(["signal_date", "predicted_d5_return", "rank", "ticker"], ascending=[True,False,True,True], kind="mergesort").assign(ai_rank=lambda z:z.groupby("signal_date").cumcount()+1)


def baseline_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen baseline order without changing admission or execution."""
    return frame.sort_values(["signal_date", "rank", "ticker"], kind="mergesort").assign(ai_rank=lambda z:z.groupby("signal_date").cumcount()+1)


def validate_cache_overlap(training: Mapping[str, pd.DataFrame], evaluation: Mapping[str, pd.DataFrame]) -> None:
    """Fail closed when overlapping ticker/date OHLCV rows differ."""
    for ticker in sorted(set(training) & set(evaluation)):
        a, b = _frame(training[ticker]), _frame(evaluation[ticker])
        common=a.index.intersection(b.index)
        for day in common:
            for col in ("Open", "High", "Low", "Close", "Volume", "AdjClose"):
                if col in a and col in b and not np.isclose(float(a.at[day,col]), float(b.at[day,col]), equal_nan=True):
                    raise ValueError("CACHE_OVERLAP_MISMATCH")


def atomic_write(output: Path, artifacts: Mapping[str, bytes], repo: Path) -> None:
    names={"summary.json","trades.csv","predictions.csv","daily_equity.csv"}
    if set(artifacts)!=names: raise ValueError("ARTIFACT_SCHEMA_INVALID")
    if output.resolve().is_relative_to(repo.resolve()): raise ValueError("OUTPUT_INSIDE_REPOSITORY")
    if output.exists() and any(output.iterdir()): raise ValueError("OUTPUT_NONEMPTY")
    stage=output.with_name(output.name+".staging");
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir(parents=True)
    try:
        for n,b in artifacts.items():
            q=stage/n; q.write_bytes(b)
            if q.read_bytes()!=b: raise ValueError("ARTIFACT_VERIFY_FAILED")
        os.replace(stage, output)
    finally:
        if stage.exists(): shutil.rmtree(stage, ignore_errors=True)


def synthetic_artifacts() -> dict[str, bytes]:
    """Small deterministic smoke output with two arms and changed order."""
    pred=pd.DataFrame([{"evaluation_year":2020,"signal_date":"2019-12-30","ticker":"B","industry":"J","baseline_rank":2,"ai_rank":1,"predicted_d5_return":.02,"realized_d5_return":.01,"positive_label":1,"training_cutoff":"2020-01-01","training_row_count":1000,**{f:0.0 for f in FEATURES}}, {"evaluation_year":2020,"signal_date":"2019-12-30","ticker":"A","industry":"I","baseline_rank":1,"ai_rank":2,"predicted_d5_return":-.01,"realized_d5_return":-.02,"positive_label":0,"training_cutoff":"2020-01-01","training_row_count":1000,**{f:0.0 for f in FEATURES}}])
    trades=pd.DataFrame([{"arm":a,"evaluation_year":2020,"signal_date":"2019-12-30","ticker":t,"status":"FILLED","quantity":100,"exit_reason":"TIME","realized_net_profit_yen":p} for a,t,p in [("BASELINE_RANK","A",-200),("BASELINE_RANK","B",100),("AI_RANK","B",100),("AI_RANK","A",-200)]])
    equity=pd.DataFrame([{"arm":a,"evaluation_year":2020,"date":"2019-12-30","available_cash":400000,"pending_cash":0,"book_equity":400000,"mark_to_market_equity":400000,"open_positions":0} for a in ("BASELINE_RANK","AI_RANK")])
    summary={"schema_version":1,"exploratory_only":True,"unused_holdout":False,"deployment_allowed":False,"ai_used":True,"feature_list":list(FEATURES),"feature_hash":feature_hash(),"model_parameters":MODEL_PARAMS,"model_hash":model_hash(),"evaluation_years":list(EVAL_YEARS),"candidate_order_changed":True,"verdict":"V5_B_CANDIDATE_RANKER_EXPLORATORY_NOT_PROMISING","synthetic":True}
    enc=lambda x:x.to_csv(index=False,lineterminator="\n").encode()
    return {"summary.json":(json.dumps(summary,sort_keys=True,separators=(",",":"))+"\n").encode(),"trades.csv":enc(trades),"predictions.csv":enc(pred),"daily_equity.csv":enc(equity)}
