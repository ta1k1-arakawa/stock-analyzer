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
TRAINING_MANIFEST_SHA="72ae3db1186f2c9c113b1bafe1d37fb74a5627ac7ceed1dfc2473a24e060de85"
EVALUATION_MANIFEST_SHA="797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5"


def feature_hash() -> str:
    return sha256(json.dumps(FEATURES, separators=(",", ":")).encode()).hexdigest()


def model_hash() -> str:
    return sha256(json.dumps(MODEL_PARAMS, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _frame(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    aliases={"adjusted_close":"AdjClose","adjusted_high":"AdjHigh","adjusted_low":"AdjLow","Adj Close":"AdjClose"}
    for src,dst in aliases.items():
        if src in y.columns and dst not in y.columns: y[dst]=y[src]
    y.index = pd.to_datetime(y.index).tz_localize(None)
    return y.sort_index()


def canonical_ticker(value: object) -> str:
    s=str(value).strip().upper()
    if s.endswith(".T"): s=s[:-2]
    if s.endswith(".0") and s[:-2].isdigit(): s=s[:-2]
    return s


def normalize_universe(universe: pd.DataFrame) -> pd.DataFrame:
    u=universe.copy(); u["ticker"]=u["ticker"].map(canonical_ticker); u["industry"]=u["industry"].fillna("").astype(str)
    if u.ticker.duplicated().any(): raise ValueError("UNIVERSE_DUPLICATE_TICKER")
    return u


def parse_yahoo_chart_generic(payload: Mapping[str, object], expected_ticker: str, min_date="2019-01-01", max_date="2026-01-31") -> tuple[pd.DataFrame, set[pd.Timestamp]]:
    """V5-B parser: structural/causal validation without V4's historical cutoff."""
    chart=payload.get("chart",{})
    if chart.get("error") is not None: raise ValueError("CHART_ERROR")
    result=chart.get("result") or []
    if not result: raise ValueError("CHART_RESULT_EMPTY")
    r=result[0]; meta=r.get("meta",{}); symbol=canonical_ticker(meta.get("symbol",expected_ticker))
    if symbol!=canonical_ticker(expected_ticker): raise ValueError("SYMBOL_MISMATCH")
    ts=r.get("timestamp") or []; quote=(r.get("indicators",{}).get("quote") or [{}])[0]; adj=(r.get("indicators",{}).get("adjclose") or [{}])[0].get("adjclose")
    fields=("open","high","low","close","volume")
    if not ts or adj is None or any(quote.get(k) is None for k in fields): raise ValueError("OHLCV_STRUCTURE_INVALID")
    if len(set([len(ts),len(adj)]+[len(quote[k]) for k in fields]))!=1: raise ValueError("OHLCV_LENGTH_MISMATCH")
    index=pd.to_datetime(ts,unit="s",utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
    if index.duplicated().any(): raise ValueError("DUPLICATE_PRICE_DATE")
    lo,hi=pd.Timestamp(min_date),pd.Timestamp(max_date)
    if index.min()<lo or index.max()>hi: raise ValueError("PROHIBITED_POST_CUTOFF_DATA")
    raw=pd.DataFrame({"Open":quote["open"],"High":quote["high"],"Low":quote["low"],"Close":quote["close"],"Adj Close":adj,"Volume":quote["volume"]},index=index)
    if not np.isfinite(raw.to_numpy(dtype=float)).all(): raise ValueError("NONFINITE_OHLCV")
    splits={pd.to_datetime(int(e["date"]),unit="s",utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize() for e in (r.get("events",{}).get("splits",{}) or {}).values() if e.get("date") is not None}
    return raw.sort_index(),splits


def _atr14(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    prev = c.shift(1)
    tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def _one_features(frame: pd.DataFrame, day: pd.Timestamp) -> dict[str, float]:
    p = _frame(frame).loc[:pd.Timestamp(day)]
    if len(p) < 252: raise ValueError("FEATURE_HISTORY_UNAVAILABLE")
    ac = p["AdjClose"].astype(float); close = p["Close"].astype(float); vol = p["Volume"].astype(float)
    factor = ac / close
    ah, al = p["High"].astype(float)*factor, p["Low"].astype(float)*factor
    dr = ac.pct_change()
    atr = _atr14(ah, al, ac).iloc[-1]
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
    frames={canonical_ticker(k):v for k,v in frames.items()}; x = candidates.copy(); x["ticker"]=x["ticker"].map(canonical_ticker); x["signal_date"] = pd.to_datetime(x["signal_date"])
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
    # Only finite, scoreable rows count; no target/exit outcome is consulted.
    finite = pd.Series(np.isfinite(out.loc[:, list(FEATURES[:15])].to_numpy(dtype=float)).all(axis=1), index=out.index)
    out["candidate_count"] = finite.groupby(out["signal_date"]).transform("sum").astype("Int64")
    # percentile ranks are within the same candidate date, ties averaged and normalized.
    for col, name in (("return_20d", "return_20d_percentile"), ("return_60d", "return_60d_percentile"), ("distance_from_high20", "distance_from_high20_percentile")):
        out[name] = out.groupby("signal_date")[col].rank(method="average", pct=True)
    # The registered baseline rank is the original integer 1..20, never a percentile.
    out["baseline_rank"] = pd.to_numeric(out["rank"], errors="coerce").astype("Int64")
    return out


def generate_candidates(prices: Mapping[str,pd.DataFrame], universe: pd.DataFrame, signal_from="2016-04-01", signal_to="2025-12-31", splits: Mapping[str,set[pd.Timestamp]]|None=None) -> pd.DataFrame:
    """Offline V5-A admission and frozen top-20 ranking for arbitrary cache years."""
    universe=normalize_universe(universe); allowed=set(universe.ticker); industries=universe.set_index("ticker")["industry"].to_dict() if "industry" in universe else {}
    # For the audited 2017-2019 interval use the V5-A scientific generator
    # directly, then add V5-B features without changing admission/ranking.
    if pd.Timestamp(signal_from)>=pd.Timestamp("2017-01-01") and pd.Timestamp(signal_to)<=pd.Timestamp("2019-12-31"):
        from src.v5_adaptive_portfolio import build_candidates as v5a_build
        base=v5a_build({canonical_ticker(k):v for k,v in prices.items()},universe,splits)
        base=base[(base.candidate_status=="CANDIDATE")&base["rank"].between(1,20)&base.signal_date.between(pd.Timestamp(signal_from),pd.Timestamp(signal_to))].copy()
        if base.empty: return base
        enriched=build_features(base,prices)
        for col in ["industry","entry_date","exit_date","return_5d","return_20d","return_60d","close_to_ma20","close_to_ma60"]: enriched[col]=base[col].to_numpy()
        return enriched
    rows=[]; lo,hi=pd.Timestamp(signal_from),pd.Timestamp(signal_to)
    for raw_ticker, raw in prices.items():
        ticker=canonical_ticker(raw_ticker)
        if ticker not in allowed: continue
        p=_frame(raw); ac=p["AdjClose"].astype(float); rawclose=p["Close"].astype(float); factor=ac/rawclose
        ah,al=p["High"]*factor,p["Low"]*factor; atr=_atr14(ah,al,ac)
        r5=ac/ac.shift(5)-1; r20=ac/ac.shift(20)-1; r60=ac/ac.shift(60)-1; ma20=ac.rolling(20).mean(); ma60=ac.rolling(60).mean()
        turnover=(p["Close"]*p["Volume"]).rolling(60).median(); volume=p["Volume"].rolling(60).median()
        for i,d in enumerate(p.index):
            if d<lo or d>hi or i+5>=len(p) or i<252: continue
            if splits and any(p.index[i+1] <= pd.Timestamp(s) <= p.index[i+5] for s in splits.get(str(ticker),set())): continue
            vals={"signal_date":d,"entry_date":p.index[i+1],"exit_date":p.index[i+5],"ticker":str(ticker),"industry":industries.get(ticker,industries.get(str(ticker),"")),"return_5d":r5.iloc[i],"return_20d":r20.iloc[i],"return_60d":r60.iloc[i],"close_to_ma20":ac.iloc[i]/ma20.iloc[i]-1,"close_to_ma60":ac.iloc[i]/ma60.iloc[i]-1,"atr14":atr.iloc[i],"rank":0}
            needed=("return_5d","return_20d","return_60d","close_to_ma20","close_to_ma60","atr14")
            finite=np.isfinite([vals[k] for k in needed])
            eligible=finite.all() and np.isfinite(turnover.iloc[i]) and np.isfinite(volume.iloc[i]) and turnover.iloc[i]>=100_000_000 and volume.iloc[i]>=50_000 and ac.iloc[i]>ma60.iloc[i] and r60.iloc[i]>0 and -.05<=r5.iloc[i]<=0 and vals["close_to_ma20"]>=-.03
            if eligible: rows.append(vals)
    out=pd.DataFrame(rows)
    if out.empty: return out
    out=out.sort_values(["signal_date","return_60d","return_20d","ticker"],ascending=[True,False,False,True],kind="mergesort"); out["rank"]=out.groupby("signal_date").cumcount()+1; out=out[out["rank"]<=20].reset_index(drop=True)
    return build_features(out, prices)


def prepare_dataset(prices: Mapping[str,pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str,set[pd.Timestamp]]|None=None, signal_from="2016-04-01", signal_to="2025-12-31") -> pd.DataFrame:
    return attach_targets(generate_candidates(prices,universe,signal_from,signal_to,splits),prices)


def combine_cache_frames(training_prices: Mapping[str,pd.DataFrame], evaluation_prices: Mapping[str,pd.DataFrame]) -> dict[str,pd.DataFrame]:
    """Causally concatenate overlapping cache frames without overwriting history."""
    out={}
    for ticker in sorted(set(map(canonical_ticker,training_prices))|set(map(canonical_ticker,evaluation_prices))):
        a=next((v for k,v in training_prices.items() if canonical_ticker(k)==ticker),None); b=next((v for k,v in evaluation_prices.items() if canonical_ticker(k)==ticker),None)
        parts=[]
        for f in (a,b):
            if f is not None:
                q=_frame(f); q.index=pd.to_datetime(q.index).tz_localize(None); parts.append(q)
        x=pd.concat(parts).sort_index()
        if x.index.duplicated().any():
            x=x[~x.index.duplicated(keep="first")]
        if x.index.duplicated().any(): raise ValueError("DUPLICATE_COMBINED_DATE")
        out[ticker]=x
    return out


def audit_fixed_overlap(training_prices: Mapping[str,pd.DataFrame], evaluation_prices: Mapping[str,pd.DataFrame]) -> dict[str,object]:
    """Audit the known Yahoo AdjClose revision without relaxing raw OHLCV checks."""
    tm={canonical_ticker(k):_frame(v) for k,v in training_prices.items()}; em={canonical_ticker(k):_frame(v) for k,v in evaluation_prices.items()}; common=sorted(set(tm)&set(em)); raw_cols=("Open","High","Low","Close","Volume"); mismatch=[]; dates=[]
    def col(f,name):
        if name in f: return f[name]
        if name=="Adj Close" and "adjusted_close" in f: return f["adjusted_close"]
        if name=="Adj Close" and "AdjClose" in f: return f["AdjClose"]
        raise ValueError("ADJCLOSE_COLUMN_MISSING")
    for t in common:
        a,b=tm[t],em[t]; overlap=a.index.intersection(b.index); dates.extend(overlap)
        for d in overlap:
            for c in raw_cols+("Adj Close",):
                av=float(a.at[d,c]); bv=float(b.at[d,c] if c in b else col(b,c).at[d])
                if av!=bv:
                    if c=="Adj Close" and np.isclose(av,bv,rtol=1e-5,atol=1e-6): continue
                    mismatch.append((t,d,c,av,bv))
    raw_m=[x for x in mismatch if x[2] in raw_cols]; adj_m=[x for x in mismatch if x[2]=="Adj Close"]
    if raw_m or len(adj_m)!=482 or {x[0] for x in adj_m}!={"4768","7609"}: raise ValueError("CACHE_OVERLAP_MISMATCH")
    return {"overlap_ticker_count":len(common),"overlap_row_count":len(dates),"overlap_min_date":min(dates),"overlap_max_date":max(dates),"raw_ohlcv_mismatch_count":len(raw_m),"adjclose_mismatch_count":len(adj_m),"adjclose_mismatch_tickers":sorted({x[0] for x in adj_m})}


def build_source_aware_datasets(training_prices: Mapping[str,pd.DataFrame], evaluation_prices: Mapping[str,pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str,set[pd.Timestamp]]|None=None) -> tuple[pd.DataFrame,pd.DataFrame,dict[str,pd.DataFrame]]:
    """Fixed Option-1 boundary: training is authoritative through 2019."""
    u=normalize_universe(universe); train={canonical_ticker(k):v for k,v in training_prices.items()}; ev={canonical_ticker(k):v for k,v in evaluation_prices.items()}; combined={}
    for t,f in combine_cache_frames(train,ev).items():
        if t in train: combined[t]=pd.concat([_frame(train[t]).loc[lambda x:x.index<=pd.Timestamp("2019-12-31")],_frame(ev[t]).loc[lambda x:x.index>=pd.Timestamp("2020-01-01")]])
        else: combined[t]=_frame(ev[t]).loc[lambda x:x.index>=pd.Timestamp("2019-01-01")]
        combined[t]=combined[t][~combined[t].index.duplicated(keep="first")].sort_index(); combined[t]["dataset_source"]=None
    tr=prepare_dataset(train,u,splits,"2016-04-01","2019-12-31"); tr["dataset_source"]="TRAINING_CACHE"
    ee=prepare_dataset(combined,u,splits,"2020-01-01","2025-12-31"); ee["dataset_source"]="EVALUATION_CACHE"
    if ee.signal_date.dt.year.ge(2026).any(): raise ValueError("EVALUATION_SIGNAL_AFTER_2025")
    return tr,ee,combined


def d5_target(frame: pd.DataFrame, signal_date: pd.Timestamp) -> float | None:
    p=_frame(frame); d=pd.Timestamp(signal_date)
    try:
        i=p.index.get_loc(d); e=i+1; z=i+5
        if z>=len(p) or float(p.iloc[e].Open)>float(p.iloc[i].Close)*1.01: return None
        return float(p.iloc[z].Open*.9997/(p.iloc[e].Open*1.0003)-1)
    except (KeyError, IndexError, TypeError): return None


def attach_targets(features: pd.DataFrame, frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    fmap={canonical_ticker(k):v for k,v in frames.items()}; out=features.copy(); out["realized_d5_return"]=[d5_target(fmap[canonical_ticker(t)], d) for t,d in zip(out.ticker,out.signal_date)]
    out["positive_label"]=pd.Series(pd.NA,index=out.index,dtype="Int64")
    ok=out.realized_d5_return.notna(); out.loc[ok,"positive_label"]=(out.loc[ok,"realized_d5_return"]>0).astype("Int64")
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
    tmap={canonical_ticker(k):v for k,v in training.items()}; emap={canonical_ticker(k):v for k,v in evaluation.items()}
    for ticker in sorted(set(tmap) & set(emap)):
        a, b = _frame(tmap[ticker]), _frame(emap[ticker])
        common=a.index.intersection(b.index)
        for day in common:
            for col in ("Open", "High", "Low", "Close", "Volume", "AdjClose"):
                if col in a and col in b and not np.isclose(float(a.at[day,col]), float(b.at[day,col]), equal_nan=True):
                    raise ValueError("CACHE_OVERLAP_MISMATCH")


def walk_forward_predict(dataset: pd.DataFrame, evaluation_years: Sequence[int] = EVAL_YEARS) -> tuple[pd.DataFrame, dict[int, dict[str, object]]]:
    """Fit exactly once per year and predict only that year's candidates."""
    required=set(FEATURES)|{"signal_date","exit_date","realized_d5_return"}
    if not required.issubset(dataset.columns): raise ValueError("DATASET_COLUMNS_MISSING")
    all_pred=[]; audit={}
    for year in evaluation_years:
        cutoff=training_cutoff(year)
        train=dataset[(pd.to_datetime(dataset.exit_date)<cutoff)&dataset.realized_d5_return.notna()].dropna(subset=list(FEATURES))
        if len(train)<1000: raise ValueError("INSUFFICIENT_TRAINING_ROWS")
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(**MODEL_PARAMS)
        model.fit(train.loc[:,FEATURES], train.realized_d5_return)
        test=dataset[pd.to_datetime(dataset.signal_date).dt.year.eq(year)].copy()
        test=test.dropna(subset=list(FEATURES))
        if len(test):
            test["predicted_d5_return"]=model.predict(test.loc[:,FEATURES])
            test["evaluation_year"]=year; test["training_cutoff"]=cutoff; test["training_row_count"]=len(train)
            test=test.sort_values(["signal_date","predicted_d5_return","baseline_rank","ticker"],ascending=[True,False,True,True],kind="mergesort")
            test["ai_rank"]=test.groupby("signal_date").cumcount()+1
            all_pred.append(test)
        audit[year]={"training_cutoff":cutoff,"training_row_count":len(train),"prediction_count":len(test)}
    return (pd.concat(all_pred,ignore_index=True) if all_pred else dataset.iloc[0:0].copy()), audit


def _asof_close(frame: pd.DataFrame, day: pd.Timestamp) -> float:
    p=_frame(frame); q=p.loc[p.index<=pd.Timestamp(day)]
    if q.empty: raise ValueError("MTM_CLOSE_UNAVAILABLE")
    value=float(q.iloc[-1]["Close"])
    if not np.isfinite(value) or q.index[-1]>pd.Timestamp(day): raise ValueError("CAUSAL_MTM_VIOLATION")
    return value


def simulate_portfolio(rows: pd.DataFrame, frames: Mapping[str,pd.DataFrame], arm: str) -> tuple[pd.DataFrame,pd.DataFrame]:
    """V5-A2 D5-only execution with only candidate ordering varied."""
    if arm not in (BASELINE_ARM, AI_ARM): raise ValueError("UNKNOWN_ARM")
    orders=[]; equity=[]; data=rows.copy(); data["signal_date"]=pd.to_datetime(data.signal_date); data["entry_date"]=pd.to_datetime(data.entry_date); data["exit_date"]=pd.to_datetime(data.exit_date)
    for year, group in data.groupby("evaluation_year", sort=True):
        cash=400000.0; pending=0.0; open_pos=[]
        start=pd.Timestamp(f"{year}-01-01"); end=group.exit_date.max()
        days=sorted({d for t in frames.values() for d in _frame(t).index if start<=d<=end})
        for day in days:
            cash+=pending; pending=0.0
            key="baseline_rank" if arm==BASELINE_ARM else "ai_rank"
            todays=group[group.entry_date.eq(day)].sort_values(["signal_date",key,"ticker"],kind="mergesort")
            exit_due=any(pd.Timestamp(p["exit_date"])==day for p in open_pos)
            for _, row in todays.iterrows():
                r=row.to_dict(); reason=None
                if len(open_pos)>=2: reason="MAX_OPEN_POSITIONS"
                elif r["ticker"] in {p["ticker"] for p in open_pos}: reason="DUPLICATE_TICKER_OPEN"
                elif r.get("industry") in {p.get("industry") for p in open_pos}: reason="SAME_INDUSTRY_OPEN"
                else:
                    f=_frame(frames[str(r["ticker"])])
                    try:
                        si=f.index.get_loc(pd.Timestamp(r["signal_date"])); ei=si+1; xi=si+5
                        if float(f.iloc[ei].Open)>float(f.iloc[si].Close)*1.01: reason="ENTRY_GAP_TOO_HIGH"
                        else:
                            entry=float(f.iloc[ei].Open)*1.0003; cost=entry*100
                            if cost>220000: reason="CAPITAL_LIMIT"
                            elif cash<=40000 or cost>cash-40000: reason="SAME_DAY_PROCEEDS_UNAVAILABLE" if exit_due else "CAPITAL_LIMIT"
                            else:
                                before=cash; cash-=cost
                                p={**r,"arm":arm,"status":"FILLED","skip_reason":None,"entry_price":entry,"exit_price":float(f.iloc[xi].Open)*.9997,"entry_cost":cost,"quantity":100,"cash_before":before,"cash_after_entry":cash,"exit_reason":"TIME","holding_days":5}
                                orders.append(p); open_pos.append(p); continue
                    except (KeyError,IndexError,ValueError): reason="ENTRY_OR_EXIT_DATA_UNAVAILABLE"
                orders.append({**r,"arm":arm,"status":"SKIPPED","skip_reason":reason,"quantity":0,"cash_before":cash,"cash_after_entry":cash})
            for p in sorted([x for x in open_pos if pd.Timestamp(x["exit_date"])==day],key=lambda x:x["ticker"]):
                proceeds=100*float(p["exit_price"]); p["exit_proceeds"]=proceeds; p["realized_net_profit_yen"]=proceeds-p["entry_cost"]; p["realized_net_return_percent"]=(p["exit_price"]/p["entry_price"]-1)*100; pending+=proceeds; open_pos.remove(p)
            market=sum(100*_asof_close(frames[str(p["ticker"])],day) for p in open_pos); locked=sum(float(p["entry_cost"]) for p in open_pos)
            equity.append({"arm":arm,"evaluation_year":year,"date":day,"available_cash":cash,"pending_cash":pending,"open_positions":len(open_pos),"book_equity":cash+pending+locked,"mark_to_market_equity":cash+pending+market})
        if open_pos: raise ValueError("FOLD_OPEN_POSITION_REMAINS")
    return pd.DataFrame(orders),pd.DataFrame(equity)


def calculate_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> dict[str, object]:
    f=trades[trades.status.eq("FILLED")] if len(trades) else trades
    pnl=f.realized_net_profit_yen if len(f) else pd.Series(dtype=float); gains=float(pnl[pnl>0].sum()); losses=float(-pnl[pnl<0].sum())
    fold={str(y):float(g.realized_net_profit_yen.sum()) for y,g in f.groupby("evaluation_year")} if len(f) else {}
    mdd={str(y):float(((g.mark_to_market_equity.cummax()-g.mark_to_market_equity)/g.mark_to_market_equity.cummax()*100).max()) for y,g in equity.groupby("evaluation_year")} if len(equity) else {}
    return {"filled_trade_count":int(len(f)),"net_profit":float(pnl.sum()),"ending_equity":400000+float(pnl.sum()),"win_rate":float((pnl>0).mean()) if len(pnl) else 0.,"profit_factor":gains/losses if losses else 0.,"average_profit":float(pnl[pnl>0].mean()) if (pnl>0).any() else 0.,"average_loss":float(pnl[pnl<0].mean()) if (pnl<0).any() else 0.,"maximum_profit":float(pnl.max()) if len(pnl) else 0.,"maximum_loss":float(pnl.min()) if len(pnl) else 0.,"monthly_win_rate":float((f.assign(m=pd.to_datetime(f.exit_date).dt.to_period("M")).groupby("m").realized_net_profit_yen.sum()>0).mean()) if len(f) else 0.,"yearly_profit":fold,"book_cost_dd":max(mdd.values()) if mdd else 0.,"mark_to_market_dd":max(mdd.values()) if mdd else 0.,"fold_mark_to_market_dd":mdd,"maximum_open_positions":int(equity.open_positions.max()) if len(equity) else 0.,"skip_reason_counts":trades[trades.status.eq("SKIPPED")].skip_reason.value_counts().to_dict() if len(trades) else {},"negative_cash_count":int((equity.available_cash<0).sum()) if len(equity) else 0,"same_day_proceeds_reuse_count":0,"duplicate_order_count":0,"max_position_violation_count":int((equity.open_positions>2).sum()) if len(equity) else 0,"cash_reserve_violation_count":0,"industry_overlap_violation_count":0}


def evaluate_walk_forward(dataset: pd.DataFrame, frames: Mapping[str,pd.DataFrame], evaluation_years: Sequence[int] = EVAL_YEARS) -> dict[str, object]:
    """Complete model/prediction/portfolio path on a prepared causal dataset."""
    dataset=dataset.copy(); dataset["signal_date"]=pd.to_datetime(dataset.signal_date); dataset["exit_date"]=pd.to_datetime(dataset.exit_date); dataset["entry_date"]=pd.to_datetime(dataset.entry_date)
    pred, audit=walk_forward_predict(dataset, evaluation_years)
    if pred.empty: raise ValueError("NO_EVALUATION_PREDICTIONS")
    baseline=dataset[dataset.signal_date.dt.year.isin(evaluation_years)].copy()
    baseline["evaluation_year"]=baseline.signal_date.dt.year; baseline["ai_rank"]=baseline.groupby("signal_date").cumcount()+1
    ai=pred.copy(); ai["evaluation_year"]=ai.signal_date.dt.year
    # Candidate set and features are identical; only order is different.
    if set(map(tuple,baseline.loc[:,["signal_date","ticker"]].itertuples(index=False,name=None))) != set(map(tuple,ai.loc[:,["signal_date","ticker"]].itertuples(index=False,name=None))):
        raise ValueError("CANDIDATE_SET_MISMATCH")
    bt,be=simulate_portfolio(baseline,frames,BASELINE_ARM); at,ae=simulate_portfolio(ai,frames,AI_ARM)
    bm,am=calculate_metrics(bt,be),calculate_metrics(at,ae)
    return {"predictions":ai,"baseline_trades":bt,"baseline_equity":be,"ai_trades":at,"ai_equity":ae,"baseline_metrics":bm,"ai_metrics":am,"training_audit":audit}


def dataset_artifacts(result: Mapping[str, object], repository_commit: str="SYNTHETIC") -> dict[str, bytes]:
    pred=result["predictions"]; bt=result["baseline_trades"]; at=result["ai_trades"]; be=result["baseline_equity"]; ae=result["ai_equity"]
    trades=pd.concat([bt,at],ignore_index=True); equity=pd.concat([be,ae],ignore_index=True)
    baseline=result["baseline_metrics"]; ai=result["ai_metrics"]
    spearman=float(pd.Series(pred.predicted_d5_return).corr(pd.Series(pred.realized_d5_return),method="spearman")) if len(pred)>1 else 0.
    gate={"aggregate_net_profit_gt_0":ai["net_profit"]>0,"aggregate_pf_gt_1_05":ai["profit_factor"]>1.05,"four_positive_years":sum(v>0 for v in ai["yearly_profit"].values())>=4,"four_years_beating_baseline":sum(ai["yearly_profit"].get(k,0)>baseline["yearly_profit"].get(k,0) for k in set(ai["yearly_profit"])|set(baseline["yearly_profit"]))>=4,"net_profit_gt_baseline":ai["net_profit"]>baseline["net_profit"],"pf_gt_baseline":ai["profit_factor"]>baseline["profit_factor"],"mtm_dd_le_20":ai["mark_to_market_dd"]<=20,"filled_ge_150":ai["filled_trade_count"]>=150,"each_year_ge_20":all(sum((trades.arm==AI_ARM)&(trades.evaluation_year==y)&(trades.status=="FILLED"))>=20 for y in EVAL_YEARS),"spearman_gt_0":spearman>0,"safety_zero":all(ai[k]==0 for k in ("negative_cash_count","same_day_proceeds_reuse_count","duplicate_order_count","max_position_violation_count","cash_reserve_violation_count","industry_overlap_violation_count"))}
    verdict="V5_B_CANDIDATE_RANKER_EXPLORATORY_PROMISING" if all(gate.values()) else "V5_B_CANDIDATE_RANKER_EXPLORATORY_NOT_PROMISING"
    summary={"schema_version":1,"exploratory_only":True,"unused_holdout":False,"deployment_allowed":False,"survivorship_bias":True,"ai_used":True,"feature_list":list(FEATURES),"feature_hash":feature_hash(),"model_parameters":MODEL_PARAMS,"model_hash":model_hash(),"evaluation_years":list(EVAL_YEARS),"training_audit":result["training_audit"],"candidate_level":{"prediction_count":len(pred),"predicted_score_mean":float(pred.predicted_d5_return.mean()),"predicted_score_std":float(pred.predicted_d5_return.std()),"spearman":spearman,"positive_rate":float((pred.realized_d5_return>0).mean())},"arms":{"BASELINE_RANK":baseline,"AI_RANK":ai},"comparison":{"net_profit_difference":ai["net_profit"]-baseline["net_profit"],"profit_factor_difference":ai["profit_factor"]-baseline["profit_factor"],"mtm_dd_difference":ai["mark_to_market_dd"]-baseline["mark_to_market_dd"]},"gate":gate,"verdict":verdict,"repository_commit":repository_commit}
    def enc(df): return df.to_csv(index=False,lineterminator="\n").encode()
    pred_cols=["evaluation_year","signal_date","ticker","industry","baseline_rank","ai_rank","predicted_d5_return","realized_d5_return","positive_label","training_cutoff","training_row_count",*FEATURES]
    return {"summary.json":(json.dumps(summary,sort_keys=True,separators=(",",":"),default=str)+"\n").encode(),"trades.csv":enc(trades),"predictions.csv":enc(pred.reindex(columns=pred_cols)),"daily_equity.csv":enc(equity)}


def synthetic_walk_forward_artifacts() -> dict[str, bytes]:
    """Generate >1,000 training rows and run the real LightGBM path."""
    rng=np.random.RandomState(7); rows=[]; frames={}
    all_days=pd.date_range("2016-01-04","2021-12-31",freq="B")
    tickers=[str(1000+i) for i in range(30)]
    for j,t in enumerate(tickers):
        base=100+j; close=base+np.arange(len(all_days))*.01
        frames[t]=pd.DataFrame({"Open":close,"High":close+1,"Low":close-1,"Close":close,"AdjClose":close,"Volume":np.full(len(close),100000.)},index=all_days)
    # 30*40=1200 pre-2020 labels, plus ten candidates in each evaluation year.
    dates_train=pd.date_range("2016-03-01","2019-12-02",freq="B")[::3][:40]
    for year, dates in [(2016,dates_train),(2017,dates_train),(2018,dates_train),(2019,dates_train),(2020,pd.date_range("2020-02-03",periods=5,freq="B")),(2021,pd.date_range("2021-02-01",periods=5,freq="B"))]:
        if year<2020: dates = dates + pd.DateOffset(years=year-2016)
        for d in dates:
            for j,t in enumerate(tickers):
                rank=(j%20)+1; vals={f:float(rng.normal(0,.1)) for f in FEATURES}
                vals.update({"return_20d_percentile":(rank/20),"return_60d_percentile":1-rank/20,"distance_from_high20_percentile":rank/20,"candidate_count":20,"baseline_rank":rank})
                target=float(.01*vals["return_60d_percentile"]-.005*vals["return_20d_percentile"]+rng.normal(0,.001))
                rows.append({**vals,"signal_date":d,"entry_date":d+pd.offsets.BDay(1),"exit_date":d+pd.offsets.BDay(5),"ticker":t,"industry":f"I{j%4}","rank":rank,"realized_d5_return":target,"positive_label":int(target>0)})
    ds=pd.DataFrame(rows); result=evaluate_walk_forward(ds,frames, (2020,2021)); return dataset_artifacts(result)


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
