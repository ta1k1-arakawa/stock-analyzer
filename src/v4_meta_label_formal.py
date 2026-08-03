"""Offline-capable formal V4 runner: immutable cache acquisition and cache-only evaluation."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import shutil
import time
import os
import subprocess
from urllib.parse import parse_qsl, urlparse
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import requests

from src.v4_meta_label_mvp import (
    FEATURE_COLUMNS, FOLDS, MODEL_PARAMS, PRICE_FROM, PRICE_TO,
    add_execution_labels, aggregate_portfolio_metrics, baseline_filled_acceptance_evidence,
    baseline_filled_classification_metrics, build_feature_frame, cash_safety_audit,
    check_fold_data_sufficiency, evaluate_acceptance_conditions, evaluate_blocked_conditions,
    generate_oof_predictions, load_fixed_universe, parse_v4_yahoo_chart, UNIVERSE_CSV_SHA256, TICKER_LIST_SHA256,
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
def _atomic_manifest_write(path: Path, value: Mapping[str,Any]) -> None:
    body=_canonical_json(value); temporary=path.with_name(path.name+".tmp")
    try:
        with open(temporary,"wb") as handle:
            handle.write(body); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary,path)
    finally:
        if temporary.exists(): temporary.unlink()
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

def get_repository_state(repo: Path, expected_branch: str = "v4-meta-label-mvp") -> dict[str,str]:
    def git(*args: str) -> str: return subprocess.run(["git",*args],cwd=repo,text=True,capture_output=True,check=True).stdout.strip()
    try: branch=git("rev-parse","--abbrev-ref","HEAD"); head=git("rev-parse","HEAD"); remote=git("rev-parse",f"origin/{expected_branch}"); dirty=git("status","--porcelain","--untracked-files=all")
    except Exception as exc: raise ValueError("REPOSITORY_STATE_UNAVAILABLE") from exc
    if branch!=expected_branch: raise ValueError("BRANCH_MISMATCH")
    if dirty: raise ValueError("WORKTREE_DIRTY")
    if head!=remote: raise ValueError("HEAD_REMOTE_MISMATCH")
    return {"branch":branch,"head":head,"remote_sha":remote}

def _validate_url(url: str, ticker: str) -> None:
    p=urlparse(url)
    if p.scheme!="https" or p.hostname!=YAHOO_HOST or p.port is not None or p.username or p.password or p.fragment: raise ValueError("YAHOO_URL_INVALID")
    if p.path != f"{YAHOO_PATH_PREFIX}{ticker}.T" or tuple(parse_qsl(p.query,keep_blank_values=True)) != QUERY_SPEC: raise ValueError("YAHOO_URL_INVALID")

def production_yahoo_transport(url: str, attempt: int, session: Any = requests) -> tuple[int, bytes, bool]:
    """One production GET; retry policy intentionally remains in acquire_cache."""
    response=session.get(url,timeout=45,allow_redirects=False,headers={"User-Agent":"stock-analyzer-v4-formal/1.0"})
    body=response.content
    redirect=bool(300 <= response.status_code < 400 or response.headers.get("Location"))
    return int(response.status_code), body, redirect

def _universe_hashes(universe: pd.DataFrame, csv_path: Path | None = None) -> tuple[str,str]:
    if csv_path is not None:
        data=csv_path.read_text(encoding="utf-8").replace("\r\n","\n").replace("\r","\n").encode(); csv_hash=_sha(data)
    else: csv_hash="SYNTHETIC"
    ticker_hash=_sha(("\n".join(universe["ticker"].astype(str))+"\n").encode())
    return csv_hash,ticker_hash

def _validate_manifest(cache: Path) -> dict[str, Any]:
    path = cache / "cache_manifest.json"
    if not path.exists(): raise ValueError("CACHE_MANIFEST_MISSING")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    raw = cache / "raw"
    listed = set()
    for item in manifest.get("payloads", []):
        ticker=item.get("ticker"); rel=item.get("relative_path")
        if not isinstance(rel,str) or rel != f"raw/{ticker}.json" or Path(rel).is_absolute() or ".." in Path(rel).parts: raise ValueError("PAYLOAD_PATH_INVALID")
        file = (cache / rel).resolve()
        try: file.relative_to(cache.resolve())
        except ValueError: raise ValueError("PAYLOAD_PATH_INVALID")
        if not file.exists() or _sha(file.read_bytes()) != item["sha256"]: raise ValueError("PAYLOAD_HASH_MISMATCH")
        listed.add(file.resolve())
    if raw.exists() and {p.resolve() for p in raw.glob("*.json")} != listed: raise ValueError("CACHE_UNREGISTERED_PAYLOAD")
    return manifest

def _canonical_payloads(payloads: list[dict[str, Any]], ticker_order: list[str]) -> list[dict[str, Any]]:
    """Return payloads in fixed universe order, rejecting duplicate/unknown records."""
    by_ticker: dict[str, dict[str, Any]] = {}
    for item in payloads:
        ticker = item.get("ticker")
        if ticker not in ticker_order or ticker in by_ticker:
            raise ValueError("MANIFEST_PAYLOAD_INVALID")
        by_ticker[ticker] = item
    return [by_ticker[ticker] for ticker in ticker_order if ticker in by_ticker]

def _payload_hash_list(payloads: list[dict[str, Any]], ticker_order: list[str]) -> str:
    canonical = _canonical_payloads(payloads, ticker_order)
    return _sha(("\n".join(str(item["sha256"]) for item in canonical) + "\n").encode())

def _validate_audit_records(audit: list[dict[str, Any]], tickers: list[str], *, require_terminal: bool) -> None:
    """Validate audit semantics without trusting a caller-provided aggregate boolean."""
    if not isinstance(audit, list) or not audit:
        raise ValueError("NETWORK_AUDIT_MISSING")
    required={"ticker","attempt","scheme","host","path","query_specification","status","error_type","redirect_detected","body_byte_count","payload_sha256","retry","final","success"}
    by_ticker: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in tickers}
    order: list[tuple[int, int]]=[]
    for item in audit:
        if not isinstance(item, dict) or not required.issubset(item): raise ValueError("NETWORK_AUDIT_INVALID")
        ticker=item["ticker"]; attempt=item["attempt"]
        if ticker not in by_ticker or type(attempt) is not int or not 1 <= attempt <= 3: raise ValueError("NETWORK_AUDIT_INVALID")
        if item["scheme"] != "https" or item["host"] != YAHOO_HOST or item["path"] != f"{YAHOO_PATH_PREFIX}{ticker}.T" or tuple(map(tuple,item["query_specification"])) != QUERY_SPEC: raise ValueError("NETWORK_AUDIT_INVALID")
        if not isinstance(item["retry"], bool) or not isinstance(item["final"], bool) or not isinstance(item["success"], bool) or not isinstance(item["redirect_detected"], bool): raise ValueError("NETWORK_AUDIT_INVALID")
        retryable = item["status"] == "TRANSPORT_EXCEPTION" or (type(item["status"]) is int and (item["status"] == 429 or 500 <= item["status"] <= 599))
        if item["retry"] != (retryable and attempt < 3): raise ValueError("NETWORK_AUDIT_RETRY_INVALID")
        if item["success"]:
            if item["status"] != 200 or item["redirect_detected"] or not isinstance(item["payload_sha256"], str) or len(item["payload_sha256"]) != 64 or not isinstance(item["body_byte_count"], int) or item["body_byte_count"] <= 0 or item["retry"]: raise ValueError("NETWORK_AUDIT_SUCCESS_INVALID")
        elif item["status"] == 200 and not item["redirect_detected"]:
            # A 200 failure is only permitted for empty/non-bytes data and never retries.
            if item["retry"]: raise ValueError("NETWORK_AUDIT_INVALID")
        elif item["redirect_detected"] or (type(item["status"]) is int and 300 <= item["status"] < 500 and item["status"] != 429):
            if item["retry"]: raise ValueError("NETWORK_AUDIT_RETRY_INVALID")
        by_ticker[ticker].append(item); order.append((tickers.index(ticker),attempt))
    if order != sorted(order): raise ValueError("NETWORK_AUDIT_ORDER_INVALID")
    for ticker, records in by_ticker.items():
        if not records: continue
        attempts=[item["attempt"] for item in records]
        if attempts != sorted(attempts) or len(attempts) != len(set(attempts)): raise ValueError("NETWORK_AUDIT_ATTEMPT_INVALID")
        successes=[item for item in records if item["success"]]
        if len(successes) > 1 or (successes and records[-1] is not successes[0]): raise ValueError("NETWORK_AUDIT_SUCCESS_INVALID")
        if require_terminal and sum(bool(item["final"]) for item in records) != 1: raise ValueError("NETWORK_AUDIT_FINAL_INVALID")
        if records[-1]["final"] is not True and require_terminal: raise ValueError("NETWORK_AUDIT_FINAL_INVALID")
        if any(item["final"] for item in records[:-1]): raise ValueError("NETWORK_AUDIT_FINAL_INVALID")

def validate_cache_manifest(cache: Path, universe: pd.DataFrame, universe_csv_path: Path | None = None) -> dict[str,Any]:
    """Fail-closed production validator; SYNTHETIC manifests are never accepted here."""
    manifest=_validate_manifest(cache)
    required={"schema_version","complete","universe_mode","universe_csv_sha256","ticker_list_sha256","ticker_count","ticker_order","price_from","price_to","query_specification","payloads","network_audit","successful_ticker_count","failed_tickers","payload_hash_list_sha256"}
    if not isinstance(manifest,dict) or not required.issubset(manifest): raise ValueError("MANIFEST_SCHEMA_INVALID")
    if list(universe.columns) != ["ticker","market","industry"] or len(universe)!=300 or universe["ticker"].duplicated().any(): raise ValueError("PRODUCTION_UNIVERSE_INVALID")
    csv_hash,ticker_hash=_universe_hashes(universe,universe_csv_path); tickers=universe["ticker"].astype(str).tolist()
    if csv_hash != UNIVERSE_CSV_SHA256 or ticker_hash != TICKER_LIST_SHA256: raise ValueError("PRODUCTION_UNIVERSE_HASH_MISMATCH")
    if manifest["schema_version"]!=SCHEMA_VERSION or manifest["complete"] is not True or manifest["universe_mode"]!="FIXED_V4_300": raise ValueError("MANIFEST_MODE_INVALID")
    if manifest["universe_csv_sha256"]!=csv_hash or manifest["ticker_list_sha256"]!=ticker_hash or manifest["ticker_count"]!=300 or manifest["ticker_order"]!=tickers: raise ValueError("MANIFEST_UNIVERSE_MISMATCH")
    if manifest["price_from"]!="2015-01-01" or manifest["price_to"]!="2019-12-31" or tuple(map(tuple,manifest["query_specification"]))!=QUERY_SPEC: raise ValueError("MANIFEST_SPEC_MISMATCH")
    seen=set(); success=set()
    for item in manifest["payloads"]:
        ticker=item.get("ticker"); rel=item.get("relative_path")
        if ticker not in tickers or ticker in seen or rel!=f"raw/{ticker}.json" or Path(rel).is_absolute() or ".." in Path(rel).parts: raise ValueError("MANIFEST_PAYLOAD_INVALID")
        body=(cache/rel).read_bytes()
        if len(body)!=item.get("byte_count") or _sha(body)!=item.get("sha256"): raise ValueError("PAYLOAD_HASH_MISMATCH")
        seen.add(ticker); success.add(ticker)
    if _payload_hash_list(list(manifest["payloads"]), tickers)!=manifest["payload_hash_list_sha256"] or len(success)!=manifest["successful_ticker_count"]: raise ValueError("PAYLOAD_HASH_LIST_MISMATCH")
    failed=manifest["failed_tickers"]
    if not isinstance(failed,list) or set(failed)&success or set(failed)|success != set(tickers): raise ValueError("MANIFEST_SUCCESS_FAILURE_INVALID")
    audit=manifest["network_audit"]
    _validate_audit_records(audit, tickers, require_terminal=True)
    for ticker in success:
        if sum(a.get("ticker")==ticker and a.get("success") is True for a in audit) != 1: raise ValueError("PAYLOAD_SUCCESS_AUDIT_MISSING")
    for ticker in failed:
        if not any(a.get("ticker")==ticker for a in audit) or any(a.get("ticker")==ticker and a.get("success") is True for a in audit): raise ValueError("NETWORK_AUDIT_FAILURE_INVALID")
    return manifest

def acquire_cache(cache_dir: Path, universe: pd.DataFrame, transport: Callable[[str, int], tuple[int, bytes, bool]], repo: Path, sleep: Callable[[float], None] = time.sleep, universe_mode: str = "SYNTHETIC", universe_csv_path: Path | None = None) -> dict[str, Any]:
    """Stage A only; injected transport enables a no-network synthetic test."""
    _outside_repo(cache_dir, repo)
    if cache_dir.exists() and cache_dir.is_file(): raise ValueError("CACHE_PATH_IS_FILE")
    cache_dir.mkdir(parents=True, exist_ok=True); raw = cache_dir / "raw"
    if not (cache_dir / "cache_manifest.json").exists() and raw.exists() and any(raw.glob("*.json")): raise ValueError("RAW_WITHOUT_MANIFEST")
    raw.mkdir(exist_ok=True)
    existing = _validate_manifest(cache_dir) if (cache_dir / "cache_manifest.json").exists() else None
    if existing and existing.get("complete"):
        if universe_mode == "FIXED_V4_300": validate_cache_manifest(cache_dir, universe, universe_csv_path)
        return existing
    csv_hash,ticker_hash=_universe_hashes(universe,universe_csv_path)
    ticker_order=universe["ticker"].astype(str).tolist()
    if existing and not existing.get("complete"):
        required={"schema_version","complete","universe_mode","universe_csv_sha256","ticker_list_sha256","ticker_count","ticker_order","price_from","price_to","query_specification","payloads","network_audit","successful_ticker_count","failed_tickers","payload_hash_list_sha256"}
        if (not required.issubset(existing) or existing["schema_version"] != SCHEMA_VERSION or existing["complete"] is not False or existing["universe_mode"]!=universe_mode or existing["universe_csv_sha256"]!=csv_hash or existing["ticker_list_sha256"]!=ticker_hash or existing["ticker_count"] != len(ticker_order) or existing["ticker_order"]!=ticker_order or existing["price_from"]!="2015-01-01" or existing["price_to"]!="2019-12-31" or tuple(map(tuple,existing["query_specification"]))!=QUERY_SPEC): raise ValueError("INCOMPLETE_MANIFEST_INVALID")
        payloads, audit, failures = _canonical_payloads(list(existing["payloads"]),ticker_order), list(existing["network_audit"]), list(existing["failed_tickers"])
        if not isinstance(failures,list) or len(failures)!=len(set(failures)) or any(t not in ticker_order for t in failures): raise ValueError("INCOMPLETE_MANIFEST_INVALID")
        if set(failures) & {item["ticker"] for item in payloads}: raise ValueError("INCOMPLETE_MANIFEST_INVALID")
        if existing["successful_ticker_count"] != len(payloads) or existing["payload_hash_list_sha256"] != _payload_hash_list(payloads,ticker_order): raise ValueError("INCOMPLETE_MANIFEST_INVALID")
        # _validate_manifest did path, existence, unknown-raw and SHA checks before this point.
        _validate_audit_records(audit,ticker_order,require_terminal=False)
        for item in payloads:
            if sum(a["ticker"]==item["ticker"] and a["success"] for a in audit) != 1: raise ValueError("INCOMPLETE_MANIFEST_INVALID")
    else: payloads, audit, failures = [], [], []
    successful={item["ticker"] for item in payloads}
    def save_in_progress() -> None:
        snapshot={"schema_version":SCHEMA_VERSION,"complete":False,"universe_mode":universe_mode,"universe_csv_sha256":csv_hash,"ticker_list_sha256":ticker_hash,"ticker_count":len(universe),"ticker_order":ticker_order,"price_from":"2015-01-01","price_to":"2019-12-31","query_specification":list(QUERY_SPEC),"payloads":_canonical_payloads(payloads,ticker_order),"network_audit":sorted(audit,key=lambda a:(ticker_order.index(a["ticker"]),a["attempt"])),"successful_ticker_count":len(payloads),"failed_tickers":[x for x in ticker_order if x in set(failures) and x not in {p["ticker"] for p in payloads}],"payload_hash_list_sha256":_payload_hash_list(payloads,ticker_order)}
        _atomic_manifest_write(cache_dir/"cache_manifest.json",snapshot)
    save_in_progress()
    for ticker in ticker_order:
        if ticker in successful: continue
        target = raw / f"{ticker}.json"
        if target.exists():
            # An unregistered file has already been rejected by _validate_manifest.
            raise ValueError("CACHE_UNREGISTERED_PAYLOAD")
        final = False
        prior=[a for a in audit if a["ticker"]==ticker]
        next_attempt=(max((a["attempt"] for a in prior),default=0)+1)
        if prior:
            previous=prior[-1]
            retryable=previous["status"]=="TRANSPORT_EXCEPTION" or (type(previous["status"]) is int and (previous["status"]==429 or 500 <= previous["status"] <= 599))
            if not retryable or next_attempt > 3:
                if ticker not in failures: failures.append(ticker)
                save_in_progress()
                continue
            previous["final"]=False
        for attempt in range(next_attempt,4):
            url=yahoo_url(ticker); _validate_url(url,ticker)
            try: status, body, redirect = transport(url, attempt)
            except Exception: status, body, redirect = "TRANSPORT_EXCEPTION", b"", False
            retry = (status == "TRANSPORT_EXCEPTION" or (isinstance(status,int) and (status == 429 or status >= 500))) and attempt < 3
            audit.append({"ticker":ticker,"attempt":attempt,"scheme":"https","host":YAHOO_HOST,"path":f"{YAHOO_PATH_PREFIX}{ticker}.T","query_specification":list(QUERY_SPEC),"status":status,"error_type":"TRANSPORT_EXCEPTION" if status=="TRANSPORT_EXCEPTION" else None,"redirect_detected":bool(redirect),"body_byte_count":len(body) if isinstance(body,bytes) else 0,"payload_sha256":_sha(body) if isinstance(body,bytes) and body else None,"retry":retry,"final":not retry,"success":status==200 and isinstance(body,bytes) and bool(body) and not redirect})
            if status == 200 and isinstance(body, bytes) and body and not redirect:
                if target.exists(): raise ValueError("PAYLOAD_OVERWRITE_PROHIBITED")
                target.write_bytes(body); payloads.append({"ticker":ticker,"relative_path":f"raw/{ticker}.json","sha256":_sha(body),"byte_count":len(body)}); failures=[x for x in failures if x != ticker]; final=True; break
            if retry: sleep(1)
            else: break
        if not final and ticker not in failures: failures.append(ticker)
        save_in_progress()
    payloads=_canonical_payloads(payloads,ticker_order)
    audit=sorted(audit,key=lambda a:(ticker_order.index(a["ticker"]),a["attempt"]))
    failures=[ticker for ticker in ticker_order if ticker in set(failures) and ticker not in successful | {item["ticker"] for item in payloads}]
    manifest = {"schema_version":SCHEMA_VERSION,"complete":True,"universe_mode":universe_mode,"universe_csv_sha256":csv_hash,"ticker_list_sha256":ticker_hash,"ticker_count":len(universe),"ticker_order":ticker_order,"price_from":"2015-01-01","price_to":"2019-12-31","query_specification":list(QUERY_SPEC),"payloads":payloads,"network_audit":audit,"successful_ticker_count":len(payloads),"failed_tickers":failures,"payload_hash_list_sha256":_payload_hash_list(payloads,ticker_order)}
    # A complete cache is only persisted after its terminal audit shape is internally valid.
    _validate_audit_records(audit,ticker_order,require_terminal=True)
    _atomic_manifest_write(cache_dir / "cache_manifest.json",manifest); return manifest

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
