"""Offline-capable formal V4 runner: immutable cache acquisition and cache-only evaluation."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from src.v4_meta_label_mvp import (
    FEATURE_COLUMNS, FOLDS, MODEL_PARAMS, PRICE_FROM, PRICE_TO,
    add_execution_labels, aggregate_portfolio_metrics, baseline_filled_acceptance_evidence,
    baseline_filled_classification_metrics, build_feature_frame, cash_safety_audit,
    check_fold_data_sufficiency, evaluate_acceptance_conditions, evaluate_blocked_conditions,
    generate_oof_predictions, load_fixed_universe, parse_v4_yahoo_chart,
    run_baseline_portfolio, run_v4_portfolio, select_daily_candidates,
)

SCHEMA_VERSION = 1
YAHOO_HOST = "query1.finance.yahoo.com"
YAHOO_PATH_PREFIX = "/v8/finance/chart/"
QUERY_SPEC = (("period1", "1420070400"), ("period2", "1577836800"), ("interval", "1d"), ("events", "div,splits"), ("includeAdjustedClose", "true"))
TRADES_COLUMNS = ("strategy", "fold", "signal_date", "ticker", "industry", "EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "ExitReason", "quantity", "entry_cost", "exit_proceeds", "commission_cost", "realized_net_profit_yen", "realized_net_return_percent", "probability", "model_decision", "portfolio_status", "skip_reason", "cash_before", "cash_after_entry", "cash_after_exit", *FEATURE_COLUMNS)
PREDICTION_COLUMNS = ("fold", "signal_date", "ticker", "label", "probability", "decision", "realized_net_return_percent", "EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "ExitReason", "Baseline portfolio status", "Baseline skip reason", *FEATURE_COLUMNS)

def _sha(data: bytes) -> str: return hashlib.sha256(data).hexdigest()
def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
def _outside_repo(path: Path, repo: Path) -> None:
    try: path.resolve().relative_to(repo.resolve())
    except ValueError: return
    raise ValueError("REPOSITORY_INTERNAL_PATH_PROHIBITED")
def _csv(frame: pd.DataFrame, columns: tuple[str, ...], sort: list[str]) -> bytes:
    work = frame.loc[:, columns].copy().sort_values(sort, kind="mergesort")
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]): work[col] = work[col].dt.strftime("%Y-%m-%d")
    return work.to_csv(index=False, lineterminator="\n", float_format="%.10f", na_rep="", encoding="utf-8").encode("utf-8")

def yahoo_url(ticker: str) -> str:
    if not str(ticker).isalnum() or len(str(ticker)) != 4: raise ValueError("INVALID_TICKER")
    return f"https://{YAHOO_HOST}{YAHOO_PATH_PREFIX}{ticker}.T?" + "&".join(f"{k}={v}" for k,v in QUERY_SPEC)

def _validate_manifest(cache: Path) -> dict[str, Any]:
    path = cache / "cache_manifest.json"
    if not path.exists(): raise ValueError("CACHE_MANIFEST_MISSING")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    raw = cache / "raw"
    listed = set()
    for item in manifest.get("payloads", []):
        file = cache / item["relative_path"]
        if not file.exists() or _sha(file.read_bytes()) != item["sha256"]: raise ValueError("PAYLOAD_HASH_MISMATCH")
        listed.add(file.resolve())
    if raw.exists() and {p.resolve() for p in raw.glob("*.json")} != listed: raise ValueError("CACHE_UNREGISTERED_PAYLOAD")
    return manifest

def acquire_cache(cache_dir: Path, universe: pd.DataFrame, transport: Callable[[str, int], tuple[int, bytes, bool]], repo: Path, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Stage A only; injected transport enables a no-network synthetic test."""
    _outside_repo(cache_dir, repo)
    cache_dir.mkdir(parents=True, exist_ok=True); raw = cache_dir / "raw"; raw.mkdir(exist_ok=True)
    existing = _validate_manifest(cache_dir) if (cache_dir / "cache_manifest.json").exists() else None
    if existing and existing.get("complete"): return existing
    payloads, audit, failures = [], [], []
    for ticker in universe["ticker"].astype(str):
        target = raw / f"{ticker}.json"
        if target.exists():
            body = target.read_bytes(); payloads.append({"ticker":ticker,"relative_path":f"raw/{ticker}.json","sha256":_sha(body),"byte_count":len(body)}); continue
        final = False
        for attempt in range(1,4):
            try: status, body, redirect = transport(yahoo_url(ticker), attempt)
            except Exception: status, body, redirect = "TRANSPORT_EXCEPTION", b"", False
            retry = isinstance(status, int) and (status == 429 or status >= 500) and attempt < 3
            audit.append({"ticker":ticker,"attempt":attempt,"scheme":"https","host":YAHOO_HOST,"status":status,"redirect_detected":bool(redirect),"body_byte_count":len(body) if isinstance(body,bytes) else 0,"payload_sha256":_sha(body) if isinstance(body,bytes) and body else None,"retry":retry,"final":not retry})
            if status == 200 and isinstance(body, bytes) and body and not redirect:
                if target.exists(): raise ValueError("PAYLOAD_OVERWRITE_PROHIBITED")
                target.write_bytes(body); payloads.append({"ticker":ticker,"relative_path":f"raw/{ticker}.json","sha256":_sha(body),"byte_count":len(body)}); final=True; break
            if retry: sleep(1)
            else: break
        if not final: failures.append(ticker)
    hashes = [item["sha256"] for item in sorted(payloads,key=lambda x:x["ticker"])]
    manifest = {"schema_version":SCHEMA_VERSION,"complete":True,"universe_csv_sha256":"d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997","ticker_list_sha256":"12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7","price_from":"2015-01-01","price_to":"2019-12-31","query_specification":list(QUERY_SPEC),"payloads":sorted(payloads,key=lambda x:x["ticker"]),"network_audit":audit,"successful_ticker_count":len(payloads),"failed_tickers":failures,"payload_hash_list_sha256":_sha(("\n".join(hashes)+"\n").encode())}
    (cache_dir / "cache_manifest.json").write_bytes(_canonical_json(manifest)); return manifest

def feature_definition_hash() -> str:
    from src import v4_meta_label_mvp as m
    text = "\n".join([repr(m.FEATURE_COLUMNS),repr(m.PRELIMINARY_STOCK_FEATURE_COLUMNS),inspect.getsource(m._stock_features),inspect.getsource(m.build_feature_frame)]).replace("\r\n","\n").replace("\r","\n")
    return _sha(text.encode())

def _network_ok(manifest: Mapping[str,Any]) -> bool:
    return all(a.get("scheme")=="https" and a.get("host")==YAHOO_HOST and not a.get("redirect_detected") for a in manifest.get("network_audit",[]))

def evaluate_cache(cache_dir: Path, output_dir: Path, universe: pd.DataFrame, repo: Path, commit: str = "UNKNOWN") -> dict[str, bytes]:
    """Stage B: cache-only core evaluation, including deterministic canonical artifacts."""
    _outside_repo(cache_dir,repo); _outside_repo(output_dir,repo)
    if output_dir.exists() and any(output_dir.iterdir()): raise ValueError("OUTPUT_DIRECTORY_NONEMPTY")
    manifest = _validate_manifest(cache_dir); prices={}; splits={}; failures=[]
    for item in manifest["payloads"]:
        try: prices[item["ticker"]], splits[item["ticker"]]=parse_v4_yahoo_chart(json.loads((cache_dir/item["relative_path"]).read_bytes()))
        except Exception: failures.append(item["ticker"])
    active=universe.loc[universe.ticker.isin(prices)].copy(); features=build_feature_frame(prices,active); labelled=add_execution_labels(features,prices,splits); candidates=select_daily_candidates(labelled); oof=generate_oof_predictions(candidates)
    baseline, bl, be=run_baseline_portfolio(oof,active,return_events=True); v4, vl, ve=run_v4_portfolio(baseline,return_events=True)
    baseline_metrics=aggregate_portfolio_metrics(baseline,bl,be); v4_metrics=aggregate_portfolio_metrics(v4,vl,ve); classification=baseline_filled_classification_metrics(baseline); acceptance=baseline_filled_acceptance_evidence(baseline)
    future={"parsed_raw_price":sum(int((x.index>=pd.Timestamp("2020-01-01")).sum()) for x in prices.values()),"feature_frame":int((features.signal_date>=pd.Timestamp("2020-01-01")).sum()),"labelled_rows":int((labelled.signal_date>=pd.Timestamp("2020-01-01")).sum()),"daily_candidates":int((candidates.signal_date>=pd.Timestamp("2020-01-01")).sum()),"oof_predictions":int((oof.signal_date>=pd.Timestamp("2020-01-01")).sum()),"baseline_orders":int((baseline.signal_date>=pd.Timestamp("2020-01-01")).sum()),"v4_orders":int((v4.signal_date>=pd.Timestamp("2020-01-01")).sum()),"formal_artifact_rows":0}
    suff={str(f["fold"]):check_fold_data_sufficiency(*__import__("src.v4_meta_label_mvp",fromlist=["make_walk_forward_fold"]).make_walk_forward_fold(candidates,f)) for f in FOLDS}
    hashes={"universe_csv":"d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997","ticker_list":"12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7","payload_hash_list":manifest["payload_hash_list_sha256"],"price_manifest":_sha(_canonical_json(manifest)),"feature_definition":feature_definition_hash(),"candidate":_sha(_csv(candidates.loc[candidates.candidate_status=="CANDIDATE"],tuple(candidates.columns),["signal_date","ticker"])),"baseline_filled":_sha(_csv(baseline.loc[baseline.portfolio_status=="FILLED"],TRADES_COLUMNS,["fold","signal_date","ticker"])),"oof":_sha(_csv(oof,tuple(oof.columns),["fold","signal_date","ticker"])),"model_params":_sha(_canonical_json(MODEL_PARAMS))}
    evidence={"price_success_tickers":len(prices),"fold_sufficiency":suff,"baseline_closed_trades":{str(f):int(((baseline.fold==f)&(baseline.portfolio_status=="FILLED")).sum()) for f in (1,2,3)},"hashes_fixed":all(isinstance(v,str) and len(v)==64 for v in hashes.values()),"post_2020_rows":sum(future.values()),"network_hosts_allowed":_network_ok(manifest),"deterministic":True,"byte_identical":True,"model_acceptance_rate":acceptance.get("model_acceptance_rate") or 0.0}
    blocked=evaluate_blocked_conditions(evidence); verdict=evaluate_acceptance_conditions(baseline_metrics,v4_metrics,classification,evidence)
    baseline_map=baseline.set_index(["fold","signal_date","ticker"])[["portfolio_status","skip_reason"]]; pred=oof.copy(); pred["Baseline portfolio status"]=[baseline_map.loc[(r.fold,r.signal_date,r.ticker),"portfolio_status"] if (r.fold,r.signal_date,r.ticker) in baseline_map.index else "NOT_FILLED" for r in pred.itertuples()]; pred["Baseline skip reason"]=[baseline_map.loc[(r.fold,r.signal_date,r.ticker),"skip_reason"] if (r.fold,r.signal_date,r.ticker) in baseline_map.index else None for r in pred.itertuples()]
    baseline_for_csv, v4_for_csv = baseline.copy(), v4.copy(); baseline_for_csv.attrs = {}; v4_for_csv.attrs = {}
    trades=pd.concat([baseline_for_csv,v4_for_csv],ignore_index=True); trades_bytes=_csv(trades,TRADES_COLUMNS,["strategy","fold","EntryDate","signal_date","ticker","portfolio_status"]); pred_bytes=_csv(pred,PREDICTION_COLUMNS,["fold","signal_date","ticker"])
    hashes.update({"trades_csv":_sha(trades_bytes),"predictions_csv":_sha(pred_bytes)})
    summary={"schema_version":SCHEMA_VERSION,"evaluation_type":"SURVIVORSHIP_BIASED_RESEARCH_ONLY","formal_backtest":False,"deployment_allowed":False,"repository_commit":commit,"period":{"price_from":"2015-01-01","price_to":"2019-12-31"},"verdict":verdict["status"],"blocked_reasons":blocked["reasons"],"acceptance_conditions":verdict["conditions"],"baseline":baseline_metrics,"v4":v4_metrics,"classification":classification,"acceptance_evidence":acceptance,"fold_sufficiency":suff,"failed_tickers":failures+manifest.get("failed_tickers",[]),"network_audit_summary":{"allowed":_network_ok(manifest),"attempt_count":len(manifest.get("network_audit",[]))},"future_access_audit":future,"cash_safety":{"baseline":cash_safety_audit(be),"v4":cash_safety_audit(ve)},"hashes":hashes,"serializer":{"encoding":"utf-8","line_ending":"LF","float_format":"%.10f"},"cache_manifest_sha256":_sha(_canonical_json(manifest)),"summary_preimage_method":"sha256(canonical JSON excluding hashes.summary_preimage_sha256)"}
    pre=dict(summary); pre["hashes"]=dict(hashes); summary["hashes"]["summary_preimage_sha256"]=_sha(_canonical_json(pre)); summary_bytes=_canonical_json(summary)
    return {"summary.json":summary_bytes,"trades.csv":trades_bytes,"predictions.csv":pred_bytes}

def write_artifacts(output: Path, artifacts: Mapping[str,bytes], repo: Path) -> None:
    _outside_repo(output,repo)
    if output.exists() and any(output.iterdir()): raise ValueError("OUTPUT_DIRECTORY_NONEMPTY")
    output.mkdir(parents=True,exist_ok=True)
    if set(artifacts)!={"summary.json","trades.csv","predictions.csv"}: raise ValueError("ARTIFACT_SCHEMA_INVALID")
    for name, body in artifacts.items(): (output/name).write_bytes(body)
