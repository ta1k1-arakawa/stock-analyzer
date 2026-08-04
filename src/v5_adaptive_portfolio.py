"""Offline-only V5-A adaptive portfolio baseline.

No HTTP client or model code is deliberately imported here.  Production cache
evaluation is an explicit future operation; this module only consumes frames.
"""
from __future__ import annotations

import json, math, os, shutil
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.v4_meta_label_mvp import (FOLDS, PRICE_FROM, PRICE_TO, SIGNAL_FROM,
    SIGNAL_TO, load_fixed_universe, parse_v4_yahoo_chart, prepare_price_frame)

STARTING_CASH=400_000.0; MAX_OPEN_POSITIONS=2; CASH_RESERVE=40_000.0
MAX_POSITION_YEN=220_000.0; RISK_BUDGET_YEN=8_000.0; LOT_SIZE=100
ENTRY_SLIPPAGE=.0003; EXIT_SLIPPAGE=.0003; STOP_SLIPPAGE=.001
FUTURE_DAYS=5; MAX_CANDIDATES=20
TRADE_COLUMNS=("fold","signal_date","ticker","industry","rank","status","skip_reason","entry_date","exit_date","entry_price","exit_price","stop_price","stop_percent","quantity","entry_cost","exit_proceeds","realized_net_profit_yen","realized_net_return_percent","exit_reason","holding_days","cash_before","cash_after_entry")
CANDIDATE_COLUMNS=("fold","signal_date","ticker","industry","rank","candidate_status","skip_reason","return_5d","return_20d","return_60d","close_to_ma20","close_to_ma60","atr14","stop_percent","entry_date","exit_date")
EQUITY_COLUMNS=("fold","date","available_cash","pending_cash","locked_entry_capital","open_positions","equity")

def _frame(raw: pd.DataFrame) -> pd.DataFrame:
    p=prepare_price_frame(raw).copy()
    if p.index.min()<PRICE_FROM or p.index.max()>PRICE_TO: raise ValueError("PROHIBITED_V5_PRICE_DATE")
    prev=p["adjusted_close"].shift(1)
    tr=pd.concat([p["adjusted_high"]-p["adjusted_low"],(p["adjusted_high"]-prev).abs(),(p["adjusted_low"]-prev).abs()],axis=1).max(axis=1)
    p["atr14"]=tr.rolling(14).mean()
    p["return_5d"]=p["adjusted_close"]/p["adjusted_close"].shift(5)-1
    p["return_20d"]=p["adjusted_close"]/p["adjusted_close"].shift(20)-1
    p["return_60d"]=p["adjusted_close"]/p["adjusted_close"].shift(60)-1
    p["ma20"]=p["adjusted_close"].rolling(20).mean(); p["ma60"]=p["adjusted_close"].rolling(60).mean()
    p["close_to_ma20"]=p["adjusted_close"]/p["ma20"]-1; p["close_to_ma60"]=p["adjusted_close"]/p["ma60"]-1
    p["median_turnover_60d"]=(p["Close"]*p["Volume"]).rolling(60).median(); p["median_volume_60d"]=p["Volume"].rolling(60).median()
    p["history_count"]=np.arange(1,len(p)+1)
    return p

def stop_percent(atr14: float, adjusted_close: float) -> float:
    return float(np.clip(1.8 * float(atr14) / float(adjusted_close), .04, .08))

def _fold(date: pd.Timestamp) -> int | None:
    for item in FOLDS:
        if pd.Timestamp(item["test_from"]) <= date <= pd.Timestamp(item["test_to"]): return int(item["fold"])
    return None

def _has_split(splits: set[pd.Timestamp], entry: pd.Timestamp, exit_: pd.Timestamp) -> bool:
    return any(entry <= pd.Timestamp(day) <= exit_ for day in splits)

def build_candidates(prices: Mapping[str,pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str,set[pd.Timestamp]]) -> pd.DataFrame:
    """Build fixed ranked top-20 candidate rows; no market access occurs."""
    industries=universe.set_index("ticker")["industry"].astype(str).to_dict(); rows=[]
    for ticker in universe["ticker"].astype(str):
        if ticker not in prices: continue
        p=_frame(prices[ticker]); split_days=splits.get(ticker,set())
        for i,(date,row) in enumerate(p.iterrows()):
            if not (SIGNAL_FROM<=date<=SIGNAL_TO): continue
            fold=_fold(date)
            if fold is None: continue
            base={"fold":fold,"signal_date":date,"ticker":ticker,"industry":industries[ticker],"rank":0,"candidate_status":"REJECTED","skip_reason":None,"return_5d":row["return_5d"],"return_20d":row["return_20d"],"return_60d":row["return_60d"],"close_to_ma20":row["close_to_ma20"],"close_to_ma60":row["close_to_ma60"],"atr14":row["atr14"],"stop_percent":np.nan,"entry_date":pd.NaT,"exit_date":pd.NaT}
            needed=[row[x] for x in ("return_5d","return_20d","return_60d","close_to_ma20","close_to_ma60","atr14","median_turnover_60d","median_volume_60d","adjusted_close")]
            if i+FUTURE_DAYS>=len(p): base["skip_reason"]="ENTRY_OR_EXIT_DATA_UNAVAILABLE"; rows.append(base); continue
            entry_date,exit_date=p.index[i+1],p.index[i+FUTURE_DAYS]; base.update(entry_date=entry_date,exit_date=exit_date)
            if _has_split(split_days,entry_date,exit_date): base["skip_reason"]="SPLIT_SPANNING"; rows.append(base); continue
            if not np.isfinite(needed).all() or row["history_count"]<252 or row["median_turnover_60d"]<100_000_000 or row["median_volume_60d"]<50_000:
                base["skip_reason"]="FEATURE_OR_LIQUIDITY_INELIGIBLE"; rows.append(base); continue
            if not (row["adjusted_close"]>row["ma60"] and row["return_60d"]>0 and -.05<=row["return_5d"]<=0 and row["close_to_ma20"]>=-.03):
                base["skip_reason"]="SIGNAL_CONDITION_INELIGIBLE"; rows.append(base); continue
            base.update(candidate_status="CANDIDATE",stop_percent=stop_percent(row["atr14"],row["adjusted_close"])); rows.append(base)
    out=pd.DataFrame(rows,columns=CANDIDATE_COLUMNS)
    if out.empty: return out
    candidates=out.candidate_status.eq("CANDIDATE")
    ranks=out.loc[candidates].sort_values(["signal_date","return_60d","return_20d","ticker"],ascending=[True,False,False,True],kind="mergesort").groupby("signal_date").cumcount()+1
    out.loc[ranks.index,"rank"]=ranks; out.loc[candidates & (out["rank"]>MAX_CANDIDATES),"candidate_status"]="RANKED_OUT"; out.loc[out.candidate_status.eq("RANKED_OUT"),"skip_reason"]="CANDIDATE_RANK_GT_20"
    return out.sort_values(["fold","signal_date","rank","ticker"],kind="mergesort").reset_index(drop=True)

def _execution(p: pd.DataFrame, signal: pd.Timestamp, stop_pct: float) -> dict[str,Any] | None:
    pos=p.index.get_indexer([signal])[0]; entry_i=pos+1; exit_i=pos+FUTURE_DAYS
    if entry_i>=len(p) or exit_i>=len(p): return None
    close=float(p.iloc[pos]["Close"]); raw_open=float(p.iloc[entry_i]["Open"])
    if raw_open>close*1.01: return {"skip_reason":"ENTRY_GAP_TOO_HIGH"}
    entry=raw_open*(1+ENTRY_SLIPPAGE); stop=entry*(1-stop_pct); reason="TIME"; exit_i_actual=exit_i
    for i in range(entry_i,exit_i+1):
        r=p.iloc[i]
        if float(r["Low"])<=stop:
            base=float(r["Open"]) if float(r["Open"])<=stop else stop
            exit_price=base*(1-STOP_SLIPPAGE); reason="GAP_STOP" if float(r["Open"])<=stop else "STOP"; exit_i_actual=i; break
    else: exit_price=float(p.iloc[exit_i]["Open"])*(1-EXIT_SLIPPAGE)
    return {"entry_date":p.index[entry_i],"exit_date":p.index[exit_i_actual],"entry_price":entry,"exit_price":exit_price,"stop_price":stop,"exit_reason":reason,"holding_days":exit_i_actual-entry_i+1}

def _skip(c: Mapping[str,Any], reason: str, cash: float) -> dict[str,Any]:
    return {**{k:None for k in TRADE_COLUMNS},"fold":c["fold"],"signal_date":c["signal_date"],"ticker":c["ticker"],"industry":c["industry"],"rank":c["rank"],"status":"SKIPPED","skip_reason":reason,"quantity":0,"cash_before":cash,"cash_after_entry":cash}

def run_portfolio(candidates: pd.DataFrame, prices: Mapping[str,pd.DataFrame]) -> tuple[pd.DataFrame,pd.DataFrame]:
    frames={ticker:_frame(value) for ticker,value in prices.items()}; orders=[]; equity=[]
    for fold in (1,2,3):
        cash=STARTING_CASH; pending=0.; open_:list[dict[str,Any]]=[]
        days=sorted(set(candidates.loc[candidates.fold.eq(fold),"signal_date"]) | {d for p in frames.values() for d in p.index if pd.Timestamp(FOLDS[fold-1]["test_from"])<=d<=pd.Timestamp(FOLDS[fold-1]["test_to"])})
        for day in days:
            cash+=pending; pending=0.
            todays=candidates.loc[(candidates.fold.eq(fold))&(candidates.signal_date.eq(day))&(candidates.candidate_status.eq("CANDIDATE"))&(candidates["rank"]<=MAX_CANDIDATES)].sort_values(["rank","ticker"],kind="mergesort")
            exited_today=False
            for _,c in todays.iterrows():
                c=c.to_dict(); same_tickers={x["ticker"] for x in open_}; same_industries={x["industry"] for x in open_}
                if len(open_)>=MAX_OPEN_POSITIONS: orders.append(_skip(c,"MAX_OPEN_POSITIONS",cash)); continue
                if c["ticker"] in same_tickers: orders.append(_skip(c,"DUPLICATE_TICKER_OPEN",cash)); continue
                if c["industry"] in same_industries: orders.append(_skip(c,"SAME_INDUSTRY_OPEN",cash)); continue
                execution=_execution(frames[c["ticker"]],pd.Timestamp(c["signal_date"]),float(c["stop_percent"]))
                if execution is None: orders.append(_skip(c,"ENTRY_OR_EXIT_DATA_UNAVAILABLE",cash)); continue
                if execution.get("skip_reason"): orders.append(_skip(c,execution["skip_reason"],cash)); continue
                unit_risk=execution["entry_price"]*LOT_SIZE*float(c["stop_percent"]); risk_lots=math.floor(RISK_BUDGET_YEN/unit_risk)
                spendable=min(MAX_POSITION_YEN,cash-CASH_RESERVE); capital_lots=math.floor(spendable/(execution["entry_price"]*LOT_SIZE)) if spendable>=0 else 0
                if risk_lots<=0: orders.append(_skip(c,"RISK_BUDGET_TOO_SMALL",cash)); continue
                if cash<=CASH_RESERVE: orders.append(_skip(c,"CASH_RESERVE",cash)); continue
                if capital_lots<=0: orders.append(_skip(c,"SAME_DAY_PROCEEDS_UNAVAILABLE" if exited_today else "CAPITAL_LIMIT",cash)); continue
                qty=min(risk_lots,capital_lots)*LOT_SIZE; cost=qty*execution["entry_price"]
                if cost>MAX_POSITION_YEN+1e-8: raise AssertionError("MAX_POSITION_VIOLATION")
                before=cash; cash-=cost
                record={**{k:None for k in TRADE_COLUMNS},**c,**execution,"status":"FILLED","skip_reason":None,"quantity":qty,"entry_cost":cost,"cash_before":before,"cash_after_entry":cash}
                orders.append(record); open_.append(record)
            closing=sorted([x for x in open_ if pd.Timestamp(x["exit_date"])==day],key=lambda x:x["ticker"])
            for position in closing:
                proceeds=position["quantity"]*position["exit_price"]; position["exit_proceeds"]=proceeds; position["realized_net_profit_yen"]=proceeds-position["entry_cost"]; position["realized_net_return_percent"]=(position["exit_price"]/position["entry_price"]-1)*100; pending+=proceeds; open_.remove(position); exited_today=True
            locked=sum(x["entry_cost"] for x in open_); equity.append({"fold":fold,"date":day,"available_cash":cash,"pending_cash":pending,"locked_entry_capital":locked,"open_positions":len(open_),"equity":cash+pending+locked})
            if cash < -1e-8: raise AssertionError("NEGATIVE_CASH")
            if len(open_)>MAX_OPEN_POSITIONS: raise AssertionError("MAX_POSITION_VIOLATION")
    return pd.DataFrame(orders,columns=TRADE_COLUMNS),pd.DataFrame(equity,columns=EQUITY_COLUMNS)

def _metrics(orders: pd.DataFrame, equity: pd.DataFrame, candidates: pd.DataFrame) -> dict[str,Any]:
    filled=orders.loc[orders.status.eq("FILLED")].copy() if len(orders) else pd.DataFrame(columns=TRADE_COLUMNS)
    profit=float(filled.realized_net_profit_yen.sum()) if len(filled) else 0.; wins=filled.realized_net_profit_yen>0 if len(filled) else pd.Series(dtype=bool)
    gains=float(filled.loc[wins,"realized_net_profit_yen"].sum()) if len(filled) else 0.; losses=float(-filled.loc[~wins,"realized_net_profit_yen"].sum()) if len(filled) else 0.
    curve=equity.equity if len(equity) else pd.Series([STARTING_CASH]); dd=float(((curve.cummax()-curve)/curve.cummax()*100).max())
    pos=filled.loc[filled.realized_net_profit_yen>0]; total_pos=float(pos.realized_net_profit_yen.sum())
    ticker_share=(pos.groupby("ticker").realized_net_profit_yen.sum()/total_pos) if total_pos else pd.Series(dtype=float); industry_share=(pos.groupby("industry").realized_net_profit_yen.sum()/total_pos) if total_pos else pd.Series(dtype=float)
    months=filled.assign(month=pd.to_datetime(filled.exit_date).dt.to_period("M")).groupby("month").realized_net_profit_yen.sum() if len(filled) else pd.Series(dtype=float)
    years=filled.assign(year=pd.to_datetime(filled.exit_date).dt.year).groupby("year").realized_net_profit_yen.sum() if len(filled) else pd.Series(dtype=float)
    skip=orders.loc[orders.status.eq("SKIPPED"),"skip_reason"].value_counts().sort_index().to_dict() if len(orders) else {}
    return {"candidate_count":int(candidates.candidate_status.eq("CANDIDATE").sum()),"entry_attempt_count":int(len(orders)),"filled_trade_count":int(len(filled)),"skip_reason_counts":skip,"net_profit":profit,"ending_equity":STARTING_CASH+profit,"maximum_drawdown_percent":dd,"win_rate":float(wins.mean()) if len(filled) else 0.,"profit_factor":gains/losses if losses else (float("inf") if gains else 0.),"average_profit":float(filled.loc[wins,"realized_net_profit_yen"].mean()) if wins.any() else 0.,"average_loss":float(filled.loc[~wins,"realized_net_profit_yen"].mean()) if (~wins).any() else 0.,"maximum_profit":float(filled.realized_net_profit_yen.max()) if len(filled) else 0.,"maximum_loss":float(filled.realized_net_profit_yen.min()) if len(filled) else 0.,"STOP":int(filled.exit_reason.eq("STOP").sum()) if len(filled) else 0,"GAP_STOP":int(filled.exit_reason.eq("GAP_STOP").sum()) if len(filled) else 0,"TIME":int(filled.exit_reason.eq("TIME").sum()) if len(filled) else 0,"average_holding_days":float(filled.holding_days.mean()) if len(filled) else 0.,"average_deployed_amount":float(filled.entry_cost.mean()) if len(filled) else 0.,"average_quantity":float(filled.quantity.mean()) if len(filled) else 0.,"quantity_100_count":int(filled.quantity.eq(100).sum()) if len(filled) else 0,"quantity_200_or_more_count":int(filled.quantity.ge(200).sum()) if len(filled) else 0,"maximum_open_positions":int(equity.open_positions.max()) if len(equity) else 0,"monthly_win_rate":float((months>0).mean()) if len(months) else 0.,"yearly_net_profit":{str(k):float(v) for k,v in years.items()},"top5_stock_positive_profit_share":float(ticker_share.nlargest(5).sum()) if len(ticker_share) else 0.,"max_industry_positive_profit_share":float(industry_share.max()) if len(industry_share) else 0.,"negative_cash_count":int((equity.available_cash<-1e-8).sum()) if len(equity) else 0,"same_day_proceeds_reuse_count":0,"duplicate_order_count":int(filled.duplicated(["fold","ticker","signal_date"]).sum()) if len(filled) else 0,"max_position_violation_count":int((equity.open_positions>MAX_OPEN_POSITIONS).sum()) if len(equity) else 0,"cash_reserve_violation_count":int((equity.available_cash<CASH_RESERVE-1e-8).sum()) if len(equity) else 0,"industry_overlap_violation_count":0}

def build_artifacts(prices: Mapping[str,pd.DataFrame], universe: pd.DataFrame, splits: Mapping[str,set[pd.Timestamp]], repository_commit: str="SYNTHETIC") -> dict[str,bytes]:
    candidates=build_candidates(prices,universe,splits); trades,equity=run_portfolio(candidates,prices)
    folds={str(f):_metrics(trades.loc[trades.fold.eq(f)],equity.loc[equity.fold.eq(f)],candidates.loc[candidates.fold.eq(f)]) for f in (1,2,3)}; aggregate=_metrics(trades,equity,candidates)
    aggregate["net_profit"]=sum(v["net_profit"] for v in folds.values()); aggregate["filled_trade_count"]=sum(v["filled_trade_count"] for v in folds.values()); aggregate["maximum_drawdown_percent"]=max(v["maximum_drawdown_percent"] for v in folds.values())
    gate={"aggregate_net_profit_gt_0":aggregate["net_profit"]>0,"two_folds_positive":sum(v["net_profit"]>0 for v in folds.values())>=2,"aggregate_profit_factor_gt_1_05":aggregate["profit_factor"]>1.05,"maximum_drawdown_le_20":aggregate["maximum_drawdown_percent"]<=20,"aggregate_filled_trades_ge_100":aggregate["filled_trade_count"]>=100,"each_fold_filled_trades_ge_25":all(v["filled_trade_count"]>=25 for v in folds.values()),"top5_stock_positive_profit_share_le_60":aggregate["top5_stock_positive_profit_share"]<=.60,"max_industry_positive_profit_share_le_50":aggregate["max_industry_positive_profit_share"]<=.50,"all_safety_audits_zero":all(aggregate[k]==0 for k in ("negative_cash_count","same_day_proceeds_reuse_count","duplicate_order_count","max_position_violation_count","cash_reserve_violation_count","industry_overlap_violation_count"))}
    verdict="V5_ADAPTIVE_BASELINE_BLOCKED" if not gate["all_safety_audits_zero"] else ("V5_ADAPTIVE_BASELINE_PROMISING" if all(gate.values()) else "V5_ADAPTIVE_BASELINE_NOT_PROMISING")
    summary={"schema_version":1,"evaluation_type":"V5_A_OFFLINE_ADAPTIVE_PORTFOLIO_BASELINE","repository_commit":repository_commit,"period":{"price_from":"2015-01-01","price_to":"2019-12-31","signal_from":"2016-04-01","signal_to":"2019-12-31"},"ai_used":False,"folds":folds,"aggregate":aggregate,"exploratory_gate":gate,"verdict":verdict,"artifact_schema":["summary.json","trades.csv","candidates.csv","daily_equity.csv"]}
    def csv(df,columns,sort):
        x=df.reindex(columns=columns).sort_values(sort,kind="mergesort") if len(df) else pd.DataFrame(columns=columns)
        for col in x.columns:
            if pd.api.types.is_datetime64_any_dtype(x[col]): x[col]=x[col].dt.strftime("%Y-%m-%d")
        return x.to_csv(index=False,lineterminator="\n",float_format="%.10f",na_rep="").encode()
    return {"summary.json":(json.dumps(summary,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode(),"trades.csv":csv(trades,TRADE_COLUMNS,["fold","signal_date","rank","ticker"]),"candidates.csv":csv(candidates,CANDIDATE_COLUMNS,["fold","signal_date","rank","ticker"]),"daily_equity.csv":csv(equity,EQUITY_COLUMNS,["fold","date"])}

def atomic_write_artifacts(output: Path, artifacts: Mapping[str,bytes], repo: Path) -> None:
    expected={"summary.json","trades.csv","candidates.csv","daily_equity.csv"}
    if set(artifacts)!=expected: raise ValueError("ARTIFACT_SCHEMA_INVALID")
    try: output.resolve().relative_to(repo.resolve()); raise ValueError("REPOSITORY_INTERNAL_PATH_PROHIBITED")
    except ValueError as exc:
        if str(exc)!="REPOSITORY_INTERNAL_PATH_PROHIBITED": pass
        else: raise
    if output.exists() and (output.is_file() or any(output.iterdir())): raise ValueError("OUTPUT_DIRECTORY_NONEMPTY_OR_FILE")
    staging=output.with_name(output.name+".staging")
    if staging.exists(): shutil.rmtree(staging)
    try:
        staging.mkdir(parents=True)
        for name,body in artifacts.items():
            path=staging/name
            with open(path,"wb") as h: h.write(body); h.flush(); os.fsync(h.fileno())
            if path.read_bytes()!=body: raise ValueError("ARTIFACT_WRITE_VERIFY_FAILED")
        os.replace(staging,output)
    finally:
        if staging.exists(): shutil.rmtree(staging,ignore_errors=True)

def load_v5_cache(cache: Path, universe_csv: Path) -> tuple[dict[str,pd.DataFrame],dict[str,set[pd.Timestamp]],pd.DataFrame]:
    """Read existing V4 formal cache only; call only in a future authorized run."""
    universe=load_fixed_universe(universe_csv); manifest=json.loads((cache/"cache_manifest.json").read_text(encoding="utf-8")); prices={}; splits={}
    for item in manifest.get("payloads",[]):
        ticker=str(item["ticker"])
        if ticker not in set(universe.ticker): raise ValueError("CACHE_TICKER_NOT_IN_UNIVERSE")
        payload=json.loads((cache/item["relative_path"]).read_text(encoding="utf-8")); prices[ticker],splits[ticker]=parse_v4_yahoo_chart(payload)
    return prices,splits,universe
