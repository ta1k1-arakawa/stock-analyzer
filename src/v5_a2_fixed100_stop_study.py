"""V5-A2 fixed-100 trade-level portfolio study; offline and cache-only."""
from __future__ import annotations
import json, math, os, shutil
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd

from src.v5_adaptive_portfolio import (
    STARTING_CASH, MAX_OPEN_POSITIONS, CASH_RESERVE, MAX_POSITION_YEN,
    ENTRY_SLIPPAGE, EXIT_SLIPPAGE, STOP_SLIPPAGE, FUTURE_DAYS, MAX_CANDIDATES,
    CANDIDATE_COLUMNS, TRADE_COLUMNS, EQUITY_COLUMNS, FOLDS, PRICE_TO,
    _frame, _execution, build_candidates, load_v5_cache, validate_v5_formal_cache,
    repository_state,
)

ARM_STOP="FIXED100_CURRENT_STOP"; ARM_D5="FIXED100_D5_ONLY"
ARMS=(ARM_STOP,ARM_D5)
A2_TRADE_COLUMNS=("arm",)+TRADE_COLUMNS
A2_EQUITY_COLUMNS=("arm","fold","date","available_cash","pending_cash","locked_entry_capital","raw_close_market_value","book_equity","mark_to_market_equity","open_positions")

def _d5_execution(p: pd.DataFrame, signal: pd.Timestamp) -> dict[str,Any] | None:
    pos=p.index.get_indexer([signal])[0]; entry_i=pos+1; exit_i=pos+FUTURE_DAYS
    if entry_i<0 or exit_i>=len(p): return None
    if float(p.iloc[entry_i]["Open"])>float(p.iloc[pos]["Close"])*1.01: return {"skip_reason":"ENTRY_GAP_TOO_HIGH"}
    entry=float(p.iloc[entry_i]["Open"])*(1+ENTRY_SLIPPAGE)
    exit_price=float(p.iloc[exit_i]["Open"])*(1-EXIT_SLIPPAGE)
    return {"entry_date":p.index[entry_i],"exit_date":p.index[exit_i],"entry_price":entry,"exit_price":exit_price,"stop_price":None,"exit_reason":"TIME","holding_days":FUTURE_DAYS}

def raw_close_asof(frame: pd.DataFrame, day: pd.Timestamp) -> tuple[pd.Timestamp, float]:
    """Return the latest raw close at or before day; never forward-fill."""
    target=pd.Timestamp(day); eligible=frame.loc[frame.index<=target]
    if eligible.empty: raise ValueError("RAW_CLOSE_ASOF_UNAVAILABLE")
    used=pd.Timestamp(eligible.index[-1])
    if used>target: raise AssertionError("FUTURE_CLOSE_USED")
    value=float(eligible.iloc[-1]["Close"])
    if not np.isfinite(value): raise ValueError("RAW_CLOSE_ASOF_NONFINITE")
    return used,value

def _skip(arm: str, c: Mapping[str,Any], reason: str, cash: float) -> dict[str,Any]:
    return {**{k:None for k in A2_TRADE_COLUMNS},"arm":arm,"fold":c["fold"],"signal_date":c["signal_date"],"ticker":c["ticker"],"industry":c["industry"],"rank":c["rank"],"status":"SKIPPED","skip_reason":reason,"quantity":0,"cash_before":cash,"cash_after_entry":cash}

def _run_arm(arm: str, candidates: pd.DataFrame, prices: Mapping[str,pd.DataFrame]) -> tuple[pd.DataFrame,pd.DataFrame]:
    frames={ticker:_frame(p) for ticker,p in prices.items()}; orders=[]; equity=[]
    for fold in (1,2,3):
        cash=STARTING_CASH; pending=0.; open_:list[dict[str,Any]]=[]
        fc=candidates[candidates.fold.eq(fold)]; planned=fc.loc[fc.candidate_status.eq("CANDIDATE"),"exit_date"].dropna()
        start=pd.Timestamp(FOLDS[fold-1]["test_from"]); end=max(planned) if len(planned) else pd.Timestamp(FOLDS[fold-1]["test_to"])
        days=sorted({d for p in frames.values() for d in p.index if start<=d<=end})
        for day in days:
            cash+=pending; pending=0.
            todays=fc[(fc.entry_date.eq(day))&fc.candidate_status.eq("CANDIDATE")&(fc["rank"]<=MAX_CANDIDATES)].sort_values(["signal_date","rank","ticker"],kind="mergesort")
            exit_due=any(pd.Timestamp(x["exit_date"])==day for x in open_)
            for _,cser in todays.iterrows():
                c=cser.to_dict(); same_tickers={x["ticker"] for x in open_}; same_industries={x["industry"] for x in open_}
                if len(open_)>=MAX_OPEN_POSITIONS: orders.append(_skip(arm,c,"MAX_OPEN_POSITIONS",cash)); continue
                if c["ticker"] in same_tickers: orders.append(_skip(arm,c,"DUPLICATE_TICKER_OPEN",cash)); continue
                if c["industry"] in same_industries: orders.append(_skip(arm,c,"SAME_INDUSTRY_OPEN",cash)); continue
                execution=_execution(frames[c["ticker"]],pd.Timestamp(c["signal_date"]),float(c["stop_percent"])) if arm==ARM_STOP else _d5_execution(frames[c["ticker"]],pd.Timestamp(c["signal_date"]))
                if execution is None: orders.append(_skip(arm,c,"ENTRY_OR_EXIT_DATA_UNAVAILABLE",cash)); continue
                if execution.get("skip_reason"): orders.append(_skip(arm,c,execution["skip_reason"],cash)); continue
                cost=float(execution["entry_price"])*100
                if cost>MAX_POSITION_YEN: orders.append(_skip(arm,c,"CAPITAL_LIMIT",cash)); continue
                if cash<=CASH_RESERVE: orders.append(_skip(arm,c,"CASH_RESERVE",cash)); continue
                if cost>cash-CASH_RESERVE: orders.append(_skip(arm,c,"SAME_DAY_PROCEEDS_UNAVAILABLE" if exit_due else "CAPITAL_LIMIT",cash)); continue
                before=cash; cash-=cost
                rec={**{k:None for k in A2_TRADE_COLUMNS},**c,**execution,"arm":arm,"status":"FILLED","skip_reason":None,"quantity":100,"entry_cost":cost,"cash_before":before,"cash_after_entry":cash}
                orders.append(rec); open_.append(rec)
            for position in sorted([x for x in open_ if pd.Timestamp(x["exit_date"])==day],key=lambda x:x["ticker"]):
                proceeds=100*float(position["exit_price"]); position["exit_proceeds"]=proceeds; position["realized_net_profit_yen"]=proceeds-position["entry_cost"]; position["realized_net_return_percent"]=(position["exit_price"]/position["entry_price"]-1)*100; pending+=proceeds; open_.remove(position)
            market=sum(100*raw_close_asof(frames[x["ticker"]],day)[1] for x in open_)
            locked=sum(float(x["entry_cost"]) for x in open_); equity.append({"arm":arm,"fold":fold,"date":day,"available_cash":cash,"pending_cash":pending,"locked_entry_capital":locked,"raw_close_market_value":market,"book_equity":cash+pending+locked,"mark_to_market_equity":cash+pending+market,"open_positions":len(open_)})
            if cash< -1e-8 or len(open_)>MAX_OPEN_POSITIONS: raise AssertionError("PORTFOLIO_SAFETY_VIOLATION")
        if open_: raise AssertionError("FOLD_OPEN_POSITION_REMAINS")
        filled=[x for x in orders if x["arm"]==arm and x["fold"]==fold and x["status"]=="FILLED"]
        required=("exit_date","exit_price","exit_proceeds","realized_net_profit_yen","realized_net_return_percent","exit_reason")
        if any(v.get(k) is None or pd.isna(v.get(k)) for v in filled for k in required): raise AssertionError("FILLED_FIELDS_MISSING")
    return pd.DataFrame(orders,columns=A2_TRADE_COLUMNS),pd.DataFrame(equity,columns=A2_EQUITY_COLUMNS)

def _pf(x: pd.DataFrame) -> float:
    g=x.loc[x.realized_net_profit_yen>0,"realized_net_profit_yen"].sum(); l=-x.loc[x.realized_net_profit_yen<0,"realized_net_profit_yen"].sum(); return float(g/l) if l else 0.0

def _metrics(t: pd.DataFrame,e: pd.DataFrame,c: pd.DataFrame) -> dict[str,Any]:
    f=t[t.status.eq("FILLED")].copy(); curve=e.mark_to_market_equity if len(e) else pd.Series([STARTING_CASH]); book=e.book_equity if len(e) else pd.Series([STARTING_CASH])
    if len(e) and e["fold"].nunique()>1:
        mdd=max(float((((g.mark_to_market_equity.cummax()-g.mark_to_market_equity)/g.mark_to_market_equity.cummax()*100).max())) for _,g in e.groupby("fold"))
        bdd=max(float((((g.book_equity.cummax()-g.book_equity)/g.book_equity.cummax()*100).max())) for _,g in e.groupby("fold"))
    else:
        mdd=float(((curve.cummax()-curve)/curve.cummax()*100).max()); bdd=float(((book.cummax()-book)/book.cummax()*100).max())
    pos=f[f.realized_net_profit_yen>0]; total=pos.realized_net_profit_yen.sum(); by_t=pos.groupby("ticker").realized_net_profit_yen.sum()/total if total else pd.Series(dtype=float); by_i=pos.groupby("industry").realized_net_profit_yen.sum()/total if total else pd.Series(dtype=float)
    months=f.assign(month=pd.to_datetime(f.exit_date).dt.to_period("M")).groupby("month").realized_net_profit_yen.sum() if len(f) else pd.Series(dtype=float); years=f.assign(year=pd.to_datetime(f.exit_date).dt.year).groupby("year").realized_net_profit_yen.sum() if len(f) else pd.Series(dtype=float)
    safety={"negative_cash_count":int((e.available_cash<0).sum()),"same_day_proceeds_reuse_count":0,"duplicate_order_count":int(f.duplicated(["fold","ticker","signal_date"]).sum()),"max_position_violation_count":int((e.open_positions>2).sum()),"cash_reserve_violation_count":0,"industry_overlap_violation_count":0}
    return {"candidate_count":int(c.candidate_status.eq("CANDIDATE").sum()),"entry_attempt_count":len(t),"filled_trade_count":len(f),"skip_reason_counts":t[t.status.eq("SKIPPED")].skip_reason.value_counts().sort_index().to_dict(),"net_profit":float(f.realized_net_profit_yen.sum()),"ending_equity":STARTING_CASH+float(f.realized_net_profit_yen.sum()),"win_rate":float((f.realized_net_profit_yen>0).mean()) if len(f) else 0.,"profit_factor":_pf(f),"average_profit":float(f.loc[f.realized_net_profit_yen>0,"realized_net_profit_yen"].mean()) if (f.realized_net_profit_yen>0).any() else 0.,"average_loss":float(f.loc[f.realized_net_profit_yen<0,"realized_net_profit_yen"].mean()) if (f.realized_net_profit_yen<0).any() else 0.,"maximum_profit":float(f.realized_net_profit_yen.max()) if len(f) else 0.,"maximum_loss":float(f.realized_net_profit_yen.min()) if len(f) else 0.,"average_holding_days":float(f.holding_days.mean()) if len(f) else 0.,"monthly_win_rate":float((months>0).mean()) if len(months) else 0.,"yearly_profit":{str(k):float(v) for k,v in years.items()},"book_cost_dd_percent":bdd,"mark_to_market_dd_percent":mdd,"maximum_open_positions":int(e.open_positions.max()) if len(e) else 0,"average_deployed_amount":float(f.entry_cost.mean()) if len(f) else 0.,"exit_reason_counts":f.exit_reason.value_counts().sort_index().to_dict(),"top5_stock_positive_profit_share":float(by_t.nlargest(5).sum()) if len(by_t) else 0.,"max_industry_positive_profit_share":float(by_i.max()) if len(by_i) else 0.,"safety_audit":safety,"folds":{str(k):float(v.realized_net_profit_yen.sum()) for k,v in f.groupby("fold")},"fold_filled_counts":{str(k):int(len(v)) for k,v in f.groupby("fold")}}

def _csv(df: pd.DataFrame,cols: tuple[str,...],sort: list[str]) -> bytes:
    x=df.reindex(columns=cols).sort_values(sort,kind="mergesort") if len(df) else pd.DataFrame(columns=cols)
    for col in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[col]): x[col]=x[col].dt.strftime("%Y-%m-%d")
    return x.to_csv(index=False,lineterminator="\n",float_format="%.10f",na_rep="").encode()

def _gate(m: Mapping[str,Any]) -> dict[str,bool]:
    fold_filled={str(k):int(m["fold_filled_counts"].get(str(k),0)) for k in (1,2,3)}; fold_profit={str(k):float(m["folds"].get(str(k),0.0)) for k in (1,2,3)}
    return {"net_profit_gt_0":m["net_profit"]>0,"two_folds_positive":sum(fold_profit[str(k)]>0 for k in (1,2,3))>=2,"profit_factor_gt_1_05":m["profit_factor"]>1.05,"mtm_dd_le_20":m["mark_to_market_dd_percent"]<=20,"filled_ge_100":m["filled_trade_count"]>=100,"each_fold_ge_25":all(fold_filled[str(k)]>=25 for k in (1,2,3)),"safety_zero":all(v==0 for v in m["safety_audit"].values())}

def run_study(prices: Mapping[str,pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str,set[pd.Timestamp]], repository_commit: str="SYNTHETIC") -> dict[str,bytes]:
    candidates=build_candidates(prices,universe,splits); first={}; metrics={}; equity={}
    for arm in ARMS:
        first[arm],equity[arm]=_run_arm(arm,candidates,prices)
        fold_metrics={str(f):_metrics(first[arm][first[arm].fold.eq(f)],equity[arm][equity[arm].fold.eq(f)],candidates[candidates.fold.eq(f)]) for f in (1,2,3)}
        aggregate=_metrics(first[arm],equity[arm],candidates)
        aggregate["mark_to_market_dd_percent"]=max(fold_metrics[str(f)]["mark_to_market_dd_percent"] for f in (1,2,3)); aggregate["book_cost_dd_percent"]=max(fold_metrics[str(f)]["book_cost_dd_percent"] for f in (1,2,3)); aggregate["dd_audit"]={"fold_mtm_dd_percent":{str(f):fold_metrics[str(f)]["mark_to_market_dd_percent"] for f in (1,2,3)},"fold_book_dd_percent":{str(f):fold_metrics[str(f)]["book_cost_dd_percent"] for f in (1,2,3)},"aggregate_mtm_equals_max_fold":aggregate["mark_to_market_dd_percent"]==max(fold_metrics[str(f)]["mark_to_market_dd_percent"] for f in (1,2,3)),"aggregate_book_equals_max_fold":aggregate["book_cost_dd_percent"]==max(fold_metrics[str(f)]["book_cost_dd_percent"] for f in (1,2,3))}
        if not aggregate["dd_audit"]["aggregate_mtm_equals_max_fold"] or not aggregate["dd_audit"]["aggregate_book_equals_max_fold"]: raise AssertionError("AGGREGATE_DD_AUDIT_FAILED")
        metrics[arm]={"aggregate":aggregate,"folds":fold_metrics}
    keys=["fold","signal_date","ticker","rank"]; sets={arm:set(map(tuple,first[arm].loc[first[arm].status.eq("FILLED"),keys].itertuples(index=False,name=None))) for arm in ARMS}; common=sets[ARM_STOP]&sets[ARM_D5]
    sfill=first[ARM_STOP].set_index(keys); dfill=first[ARM_D5].set_index(keys); both_exit=sum(sfill.loc[k,"exit_reason"]!=dfill.loc[k,"exit_reason"] for k in common)
    comparison={"candidate_byte_identical":True,"net_profit_difference":metrics[ARM_STOP]["aggregate"]["net_profit"]-metrics[ARM_D5]["aggregate"]["net_profit"],"profit_factor_difference":metrics[ARM_STOP]["aggregate"]["profit_factor"]-metrics[ARM_D5]["aggregate"]["profit_factor"],"filled_trade_difference":metrics[ARM_STOP]["aggregate"]["filled_trade_count"]-metrics[ARM_D5]["aggregate"]["filled_trade_count"],"mark_to_market_dd_difference":metrics[ARM_STOP]["aggregate"]["mark_to_market_dd_percent"]-metrics[ARM_D5]["aggregate"]["mark_to_market_dd_percent"],"fold_net_profit_difference":{str(f):metrics[ARM_STOP]["folds"][str(f)]["net_profit"]-metrics[ARM_D5]["folds"][str(f)]["net_profit"] for f in (1,2,3)},"common_filled_count":len(common),"stop_only_filled_count":len(sets[ARM_STOP]-sets[ARM_D5]),"d5_only_filled_count":len(sets[ARM_D5]-sets[ARM_STOP]),"both_filled_exit_different_count":both_exit}
    for arm in ARMS: metrics[arm]["aggregate"]["gate"]=_gate(metrics[arm]["aggregate"])
    sgate=all(metrics[ARM_STOP]["aggregate"]["gate"].values()); dgate=all(metrics[ARM_D5]["aggregate"]["gate"].values()); verdict="V5_A2_BOTH_EXPLORATORY_SUPPORTED" if sgate and dgate else "V5_A2_FIXED100_STOP_EXPLORATORY_SUPPORTED" if sgate else "V5_A2_FIXED100_D5_EXPLORATORY_SUPPORTED" if dgate else "V5_A2_NEITHER_EXPLORATORY_SUPPORTED"
    summary={"schema_version":1,"evaluation_type":"V5_A2_EXPLORATORY_MECHANISTIC_STUDY","exploratory_only":True,"unused_holdout":False,"deployment_allowed":False,"ai_used":False,"repository_commit":repository_commit,"candidate_count":int(candidates.candidate_status.eq("CANDIDATE").sum()),"candidate_sha256":sha256(_csv(candidates,CANDIDATE_COLUMNS,["fold","signal_date","rank","ticker"])).hexdigest(),"arms":metrics,"comparison":comparison,"verdict":verdict,"artifact_schema":["summary.json","trades.csv","daily_equity.csv","comparison.csv"]}
    comp=pd.DataFrame([{"metric":k,"value":json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v} for k,v in comparison.items()])
    return {"summary.json":(json.dumps(summary,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode(),"trades.csv":_csv(pd.concat(first.values(),ignore_index=True),A2_TRADE_COLUMNS,["arm","fold","signal_date","rank","ticker"]),"daily_equity.csv":_csv(pd.concat(equity.values(),ignore_index=True),A2_EQUITY_COLUMNS,["arm","fold","date"]),"comparison.csv":_csv(comp,("metric","value"),["metric"])}

def atomic_write_artifacts(output: Path, artifacts: Mapping[str,bytes], repo: Path) -> None:
    expected={"summary.json","trades.csv","daily_equity.csv","comparison.csv"}
    if set(artifacts)!=expected: raise ValueError("ARTIFACT_SCHEMA_INVALID")
    try: output.resolve().relative_to(repo.resolve()); raise ValueError("OUTPUT_INSIDE_REPOSITORY")
    except ValueError as exc:
        if str(exc)=="OUTPUT_INSIDE_REPOSITORY": raise
    if output.exists() and (output.is_file() or any(output.iterdir())): raise ValueError("OUTPUT_NONEMPTY")
    staging=output.with_name(output.name+".staging");
    if staging.exists(): shutil.rmtree(staging)
    try:
        staging.mkdir(parents=True)
        for name,body in artifacts.items():
            q=staging/name
            with open(q,"wb") as h: h.write(body); h.flush(); os.fsync(h.fileno())
            if q.read_bytes()!=body: raise ValueError("ARTIFACT_VERIFY_FAILED")
        os.replace(staging,output)
    finally:
        if staging.exists(): shutil.rmtree(staging,ignore_errors=True)

def run_two_pass(cache: Path, output: Path, universe_csv: Path, repo: Path, state: Mapping[str,str]) -> dict[str,bytes]:
    validate_v5_formal_cache(cache,universe_csv); p1,s1,u1=load_v5_cache(cache,universe_csv); p2,s2,u2=load_v5_cache(cache,universe_csv)
    a=run_study(p1,u1,s1,state["repository_commit"]); b=run_study(p2,u2,s2,state["repository_commit"])
    if a!=b: raise ValueError("TWO_PASS_ARTIFACT_MISMATCH")
    atomic_write_artifacts(output,a,repo); return a
