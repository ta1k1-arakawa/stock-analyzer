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

def validate_formal_inputs(cache_dir: Path, output_dir: Path, universe: pd.DataFrame, repo: Path) -> dict[str,Any]:
    _outside_repo(cache_dir, repo); _outside_repo(output_dir, repo)
    if output_dir.exists() and (output_dir.is_file() or any(output_dir.iterdir())): raise ValueError("OUTPUT_DIRECTORY_NONEMPTY_OR_FILE")
    return _validate_manifest(cache_dir)

def parse_cache_prices(cache: Path, manifest: Mapping[str,Any]) -> tuple[dict[str,pd.DataFrame],dict[str,set[pd.Timestamp]],list[dict[str,Any]]]:
    prices: dict[str,pd.DataFrame]={}; splits={}; status=[]
    for item in manifest["payloads"]:
        ticker=item["ticker"]; body=(cache/item["relative_path"]).read_bytes()
        try: payload=json.loads(body)
        except Exception: status.append({"ticker":ticker,"status":"JSON_PARSE_FAILURE","reason":"JSON_PARSE_FAILURE","row_count":0}); continue
        try:
            if payload.get("chart",{}).get("error") is not None: raise RuntimeError("YAHOO_CHART_ERROR")
            frame, split=parse_v4_yahoo_chart(payload)
            if frame.empty: status.append({"ticker":ticker,"status":"PRICE_ROWS_EMPTY","reason":"PRICE_ROWS_EMPTY","row_count":0}); continue
            prices[ticker]=frame; splits[ticker]=split
            status.append({"ticker":ticker,"status":"SUCCESS","reason":None,"row_count":len(frame),"first_date":str(frame.index.min().date()),"last_date":str(frame.index.max().date()),"canonical_price_sha256":_sha(_csv(frame.reset_index().rename(columns={"index":"date"}),tuple(frame.reset_index().rename(columns={"index":"date"}).columns),["date"]))})
        except RuntimeError as exc: status.append({"ticker":ticker,"status":"YAHOO_CHART_ERROR","reason":str(exc),"row_count":0})
        except Exception: status.append({"ticker":ticker,"status":"OHLCV_VALIDATION_FAILURE","reason":"OHLCV_VALIDATION_FAILURE","row_count":0})
    return prices,splits,status

def build_parsed_price_manifest(prices: Mapping[str,pd.DataFrame], splits: Mapping[str,Any], ticker_order: list[str]) -> tuple[list[dict[str,Any]],bytes]:
    rows=[]
    for ticker in ticker_order:
        if ticker not in prices: continue
        frame=prices[ticker].sort_index(); raw=frame.reset_index().rename(columns={"index":"date"})
        raw_hash=_sha(_csv(raw,tuple(raw.columns),["date"])); adjusted_cols=[c for c in raw.columns if c.startswith("adjusted_") or c=="date"]
        rows.append({"ticker":ticker,"row_count":len(frame),"first_date":str(frame.index.min().date()),"last_date":str(frame.index.max().date()),"raw_ohlcv_sha256":raw_hash,"adjusted_ohlc_sha256":_sha(_csv(raw,tuple(adjusted_cols),["date"])),"split_event_count":len(splits.get(ticker,[]))})
    return rows,_canonical_json(rows)

def _empty(columns: tuple[str,...]) -> pd.DataFrame: return pd.DataFrame(columns=columns)
def _future(frame: pd.DataFrame, dates: list[str]) -> int:
    return int(sum((pd.to_datetime(frame[col],errors="coerce") >= pd.Timestamp("2020-01-01")).sum() for col in dates if col in frame))

def run_formal_core_once(cache: Path, universe: pd.DataFrame, manifest: Mapping[str,Any]) -> dict[str,Any]:
    """Pure cache-only core; this function never imports or invokes a transport."""
    prices,splits,parse_status=parse_cache_prices(cache,manifest); active=universe.loc[universe.ticker.isin(prices)].copy()
    features=build_feature_frame(prices,active) if prices else pd.DataFrame(); labelled=add_execution_labels(features,prices,splits) if not features.empty else features
    candidates=select_daily_candidates(labelled) if not labelled.empty else labelled
    from src.v4_meta_label_mvp import make_walk_forward_fold
    candidate_ready=not candidates.empty and {"candidate_status","eligible","label","signal_date","ticker","LabelConfirmedDate"}.issubset(candidates.columns)
    suff={str(f["fold"]):check_fold_data_sufficiency(*make_walk_forward_fold(candidates,f)) for f in FOLDS} if candidate_ready else {str(f["fold"]):{"blocked":True,"reasons":["TRAIN_CANDIDATES_LT_100","TRAIN_LABEL_NOT_TWO_CLASSES","TEST_LABEL_NOT_TWO_CLASSES"],"train_count":0,"test_count":0,"train_positive":0,"train_negative":0} for f in FOLDS}
    blocked_fit=any(v["blocked"] for v in suff.values()); oof=_empty(("fold","signal_date","ticker","label","probability","decision","realized_net_return_percent","EntryDate","ExitDate","EntryPrice","ExitPrice","ExitReason",*FEATURE_COLUMNS))
    baseline=v4=_empty(TRADES_COLUMNS); bl=vl=be=ve=pd.DataFrame(); not_computed="NOT_COMPUTED_DUE_TO_UPSTREAM_BLOCKER" if blocked_fit else None
    if not blocked_fit:
        oof=generate_oof_predictions(candidates); baseline,bl,be=run_baseline_portfolio(oof,active,return_events=True); v4,vl,ve=run_v4_portfolio(baseline,return_events=True)
    parsed,parsed_bytes=build_parsed_price_manifest(prices,splits,universe.ticker.astype(str).tolist())
    future={"parsed_raw_price":sum(int((x.index>=pd.Timestamp("2020-01-01")).sum()) for x in prices.values()),"feature_frame":_future(features,["signal_date"]),"labelled_rows":_future(labelled,["signal_date"]),"daily_candidates":_future(candidates,["signal_date"]),"oof_predictions":_future(oof,["signal_date"]),"baseline_orders":_future(baseline,["signal_date","EntryDate","ExitDate"]),"v4_orders":_future(v4,["signal_date","EntryDate","ExitDate"])}
    return {"prices":prices,"parse_status":parse_status,"parsed_price_manifest":parsed,"parsed_price_manifest_bytes":parsed_bytes,"features":features,"labelled":labelled,"candidates":candidates,"fold_sufficiency":suff,"oof":oof,"baseline":baseline,"v4":v4,"baseline_ledger":bl,"v4_ledger":vl,"baseline_events":be,"v4_events":ve,"not_computed_reason":not_computed,"future":future}

def _strategy_csv(frame: pd.DataFrame) -> bytes:
    if frame.empty: return _csv(_empty(TRADES_COLUMNS),TRADES_COLUMNS,["fold"])
    work=frame.copy(); work["_strategy_rank"]=work["strategy"].map({"BASELINE":0,"V4":1}).fillna(9)
    work=work.sort_values(["_strategy_rank","fold","EntryDate","signal_date","ticker","portfolio_status"],kind="mergesort").loc[:,TRADES_COLUMNS]
    for col in work.columns:
        if pd.api.types.is_datetime64_any_dtype(work[col]): work[col]=work[col].dt.strftime("%Y-%m-%d")
    return work.to_csv(index=False,lineterminator="\n",float_format="%.10f",na_rep="",encoding="utf-8").encode()

def build_formal_artifacts(core: Mapping[str,Any], manifest: Mapping[str,Any], universe: pd.DataFrame, repository_state: Mapping[str,str], determinism: Mapping[str,Any]) -> dict[str,bytes]:
    baseline,v4,oof=core["baseline"],core["v4"],core["oof"]
    if core["not_computed_reason"]: trades=_empty(TRADES_COLUMNS); predictions=_empty(PREDICTION_COLUMNS)
    else:
        baseline,v4=baseline.copy(),v4.copy(); baseline.attrs={}; v4.attrs={}; trades=pd.concat([baseline,v4],ignore_index=True); baseline_map=baseline.set_index(["fold","signal_date","ticker"])[["portfolio_status","skip_reason"]]
        predictions=oof.copy(); predictions["Baseline portfolio status"]=[baseline_map.loc[(r.fold,r.signal_date,r.ticker),"portfolio_status"] for r in predictions.itertuples()]; predictions["Baseline skip reason"]=[baseline_map.loc[(r.fold,r.signal_date,r.ticker),"skip_reason"] for r in predictions.itertuples()]
    trades_bytes=_strategy_csv(trades); prediction_bytes=_csv(predictions,PREDICTION_COLUMNS,["fold","signal_date","ticker"])
    future=dict(core["future"]); future["trades_artifact_rows"]=_future(trades,["signal_date","EntryDate","ExitDate"]); future["predictions_artifact_rows"]=_future(predictions,["signal_date","EntryDate","ExitDate"]); future["total"]=sum(future.values())
    if core["not_computed_reason"]:
        baseline_metrics=v4_metrics=classification={"status":"NOT_COMPUTED_DUE_TO_UPSTREAM_BLOCKER"}; acceptance={"status":"NOT_COMPUTED_DUE_TO_UPSTREAM_BLOCKER"}; cash={"baseline":"NOT_COMPUTED","v4":"NOT_COMPUTED"}; closed={}
    else:
        baseline_metrics=aggregate_portfolio_metrics(baseline,core["baseline_ledger"],core["baseline_events"]); v4_metrics=aggregate_portfolio_metrics(v4,core["v4_ledger"],core["v4_events"]); classification=baseline_filled_classification_metrics(baseline); acceptance=baseline_filled_acceptance_evidence(baseline); cash={"baseline":cash_safety_audit(core["baseline_events"]),"v4":cash_safety_audit(core["v4_events"])}; closed={str(f):int(((baseline.fold==f)&(baseline.portfolio_status=="FILLED")).sum()) for f in (1,2,3)}
    candidate_columns=("signal_date","ticker","candidate_status","eligible","label","LabelConfirmedDate","EntryDate","ExitDate","EntryPrice","ExitPrice","ExitReason","realized_net_return_percent",*FEATURE_COLUMNS)
    cand=(core["candidates"].loc[core["candidates"].candidate_status.eq("CANDIDATE")] if "candidate_status" in core["candidates"] else _empty(candidate_columns)).reindex(columns=candidate_columns)
    hashes={"universe_csv_sha256":_universe_hashes(universe)[0],"ticker_list_sha256":_universe_hashes(universe)[1],"payload_hash_list_sha256":manifest["payload_hash_list_sha256"],"cache_manifest_sha256":_sha(_canonical_json(manifest)),"parsed_price_manifest_sha256":_sha(core["parsed_price_manifest_bytes"]),"feature_definition_sha256":feature_definition_hash(),"candidate_sha256":_sha(_csv(cand,candidate_columns,["signal_date","ticker"])),"baseline_filled_sha256":_sha(_csv(baseline.loc[baseline.portfolio_status.eq("FILLED")] if not baseline.empty else _empty(TRADES_COLUMNS),TRADES_COLUMNS,["fold","signal_date","ticker"])),"oof_predictions_sha256":_sha(_csv(oof,tuple(oof.columns),["fold","signal_date","ticker"])),"model_params_sha256":_sha(_canonical_json(MODEL_PARAMS)),"trades_csv_sha256":_sha(trades_bytes),"predictions_csv_sha256":_sha(prediction_bytes)}
    network={"attempt_count":len(manifest.get("network_audit",[])),"success_attempt_count":sum(a.get("success") is True for a in manifest.get("network_audit",[])),"transport_exception_count":sum(a.get("status")=="TRANSPORT_EXCEPTION" for a in manifest.get("network_audit",[])),"http_429_count":sum(a.get("status")==429 for a in manifest.get("network_audit",[])),"http_5xx_count":sum(type(a.get("status")) is int and a["status"]>=500 for a in manifest.get("network_audit",[])),"redirect_count":sum(bool(a.get("redirect_detected")) for a in manifest.get("network_audit",[])),"schemes":sorted({a.get("scheme") for a in manifest.get("network_audit",[])}),"hosts":sorted({a.get("host") for a in manifest.get("network_audit",[])}),"query_specification_matches":bool(manifest.get("network_audit")) and all(tuple(map(tuple,a.get("query_specification",[])))==QUERY_SPEC for a in manifest.get("network_audit",[])),"validation_status":"PASS" if bool(manifest.get("network_audit")) and _network_ok(manifest) else "FAIL"}
    evidence={"price_success_tickers":len(core["prices"]),"fold_sufficiency":core["fold_sufficiency"],"baseline_closed_trades":closed,"hashes_fixed":False,"post_2020_rows":future["total"],"network_hosts_allowed":network["validation_status"]=="PASS","deterministic":determinism["deterministic"],"byte_identical":determinism["byte_identical"],"model_acceptance_rate":acceptance.get("model_acceptance_rate",0) if isinstance(acceptance,dict) else 0}
    if core["not_computed_reason"]: evidence["baseline_closed_trades"]={str(f):0 for f in (1,2,3)}
    hashes["summary_preimage_sha256"]=""
    hashes["hashes_fixed"] = all(isinstance(value,str) and len(value)==64 for key,value in hashes.items() if key not in {"summary_preimage_sha256","hashes_fixed"})
    evidence["hashes_fixed"]=hashes["hashes_fixed"]
    blocked=evaluate_blocked_conditions(evidence)
    if core["not_computed_reason"]: blocked["reasons"].append("BASELINE_NOT_COMPUTED_DUE_TO_UPSTREAM_BLOCKER")
    verdict={"status":"FREE_META_LABEL_PROTOTYPE_BLOCKED","conditions":[]} if blocked["reasons"] else evaluate_acceptance_conditions(baseline_metrics,v4_metrics,classification,evidence)
    stock={"history_lt_252":0,"turnover_below_100m":0,"volume_below_50k":0,"required_cash_over_300k":0,"nonfinite_stock_feature":0,"nonfinite_final_feature":0,"entry_unavailable":0,"exit_unavailable":0,"split_span":0,"no_candidate_days":int((core["candidates"].get("candidate_status",pd.Series(dtype=object))=="NO_CANDIDATE").sum())}
    if not core["labelled"].empty:
        l=core["labelled"]; numeric=l.reindex(columns=FEATURE_COLUMNS).apply(pd.to_numeric,errors="coerce"); stock.update({"history_lt_252":int((l.get("History_Count",pd.Series(dtype=float))<252).sum()),"turnover_below_100m":int((l.get("Median_Turnover_60",pd.Series(dtype=float))<100000000).sum()),"volume_below_50k":int((l.get("Median_Volume_60",pd.Series(dtype=float))<50000).sum()),"required_cash_over_300k":int((l.get("required_cash_ratio",pd.Series(dtype=float))>1).sum()),"nonfinite_final_feature":int((~np.isfinite(numeric)).any(axis=1).sum())})
    exclusions={"counting_units":{"ticker":"ticker; overlapping reasons count independently","stock_day":"stock-day; overlapping reasons count independently","portfolio":"order"},"ticker":{"acquisition_failure":len(manifest.get("failed_tickers",[])),"json_parse_failure":sum(x["status"]=="JSON_PARSE_FAILURE" for x in core["parse_status"]),"yahoo_chart_error":sum(x["status"]=="YAHOO_CHART_ERROR" for x in core["parse_status"]),"ohlcv_validation_failure":sum(x["status"]=="OHLCV_VALIDATION_FAILURE" for x in core["parse_status"]),"price_rows_empty":sum(x["status"]=="PRICE_ROWS_EMPTY" for x in core["parse_status"])},"stock_day":stock,"portfolio":{"baseline_skip_reasons":baseline.get("skip_reason",pd.Series(dtype=object)).value_counts(dropna=True).to_dict(),"v4_model_abstain":int((v4.get("portfolio_status",pd.Series(dtype=object))=="ABSTAIN").sum()),"v4_cash_skip_reasons":v4.get("skip_reason",pd.Series(dtype=object)).value_counts(dropna=True).to_dict()}}
    summary={"schema_version":SCHEMA_VERSION,"evaluation_type":"FORMAL_CACHE_ONLY_SURVIVORSHIP_BIASED_RESEARCH_ONLY","formal_evaluation":True,"deployment_allowed":False,"repository_commit":repository_state["head"],"branch":repository_state["branch"],"period":{"price_from":"2015-01-01","price_to":"2019-12-31"},"universe":{"ticker_count":len(universe),"ticker_order":universe.ticker.astype(str).tolist()},"cache_manifest":{"sha256":hashes["cache_manifest_sha256"]},"verdict":verdict["status"],"blocked_reasons":sorted(set(blocked["reasons"])),"acceptance_conditions":verdict["conditions"],"baseline_metrics":baseline_metrics,"v4_metrics":v4_metrics,"classification_metrics":classification,"acceptance_evidence":acceptance,"fold_sufficiency":core["fold_sufficiency"],"parse_status":core["parse_status"],"parsed_price_manifest":core["parsed_price_manifest"],"exclusion_reason_counts":exclusions,"network_audit":network,"future_access_audit":future,"cash_safety_audit":cash,"determinism_evidence":determinism,"not_computed_reason":core["not_computed_reason"],"hashes":hashes,"serializer":{"encoding":"utf-8","line_ending":"LF","float_format":"%.10f","nan":""},"summary_preimage_method":"sha256(canonical JSON excluding hashes.summary_preimage_sha256)"}
    pre={**summary,"hashes":{k:v for k,v in hashes.items() if k!="summary_preimage_sha256"}}; hashes["summary_preimage_sha256"]=_sha(_canonical_json(pre)); assert _sha(_canonical_json({**summary,"hashes":{k:v for k,v in hashes.items() if k!="summary_preimage_sha256"}}))==hashes["summary_preimage_sha256"]
    return {"summary.json":_canonical_json(summary),"trades.csv":trades_bytes,"predictions.csv":prediction_bytes}

def compare_core_runs(first: Mapping[str,Any], second: Mapping[str,Any]) -> dict[str,Any]:
    keys=("parsed_price_manifest_bytes","fold_sufficiency","parse_status","future")
    comparison={key:(_sha(first[key]) == _sha(second[key]) if isinstance(first[key],bytes) else _sha(_canonical_json(first[key])) == _sha(_canonical_json(second[key]))) for key in keys}
    run_hashes={key:{"run_1":_sha(first[key]) if isinstance(first[key],bytes) else _sha(_canonical_json(first[key])),"run_2":_sha(second[key]) if isinstance(second[key],bytes) else _sha(_canonical_json(second[key]))} for key in keys}
    for key in ("oof","baseline","v4","baseline_ledger","v4_ledger","baseline_events","v4_events"):
        left,right=first[key],second[key]
        one,two=_csv(left,tuple(left.columns),list(left.columns[:1])),_csv(right,tuple(right.columns),list(right.columns[:1])); comparison[key]=one == two; run_hashes[key]={"run_1":_sha(one),"run_2":_sha(two)}
    return {"deterministic":all(comparison.values()),"byte_identical":all(comparison.values()),"comparisons":comparison,"comparison_hashes":run_hashes,"mismatched_targets":[key for key,value in comparison.items() if not value]}

def atomic_write_formal_artifacts(output: Path, artifacts: Mapping[str,bytes], repo: Path, file_writer: Callable[[Path,bytes],None] | None = None) -> None:
    _outside_repo(output,repo)
    if set(artifacts)!={"summary.json","trades.csv","predictions.csv"}: raise ValueError("ARTIFACT_SCHEMA_INVALID")
    if output.exists() and (output.is_file() or any(output.iterdir())): raise ValueError("OUTPUT_DIRECTORY_NONEMPTY_OR_FILE")
    staging=output.with_name(output.name+".staging")
    if staging.exists(): shutil.rmtree(staging)
    try:
        staging.mkdir(parents=True)
        for name,body in artifacts.items():
            path=staging/name
            if file_writer: file_writer(path,body)
            else:
                with open(path,"wb") as handle: handle.write(body); handle.flush(); os.fsync(handle.fileno())
            if path.read_bytes()!=body: raise ValueError("ARTIFACT_WRITE_VERIFY_FAILED")
        if {p.name for p in staging.iterdir()} != set(artifacts): raise ValueError("ARTIFACT_SCHEMA_INVALID")
        if output.exists(): output.rmdir()
        os.replace(staging,output)
    except Exception:
        if output.exists() and output.is_dir() and not any(output.iterdir()): output.rmdir()
        raise
    finally:
        if staging.exists(): shutil.rmtree(staging,ignore_errors=True)

def run_two_pass_formal_evaluation(cache: Path, output: Path, universe: pd.DataFrame, repo: Path, repository_state: Mapping[str,str], core_runner: Callable[[Path,pd.DataFrame,Mapping[str,Any]],dict[str,Any]] = run_formal_core_once) -> dict[str,bytes]:
    manifest=validate_formal_inputs(cache,output,universe,repo)
    first=core_runner(cache,universe,manifest); second=core_runner(cache,universe,manifest); determinism=compare_core_runs(first,second)
    if not determinism["deterministic"]:
        first=dict(first); first["not_computed_reason"]="DETERMINISM_NOT_CONFIRMED"; first["baseline"]=first["v4"]=_empty(TRADES_COLUMNS); first["oof"]=_empty(("fold","signal_date","ticker","label","probability","decision","realized_net_return_percent","EntryDate","ExitDate","EntryPrice","ExitPrice","ExitReason",*FEATURE_COLUMNS))
    artifacts=build_formal_artifacts(first,manifest,universe,repository_state,determinism)
    # Final artifact bytes are explicitly compared on the same core, before writing.
    other=build_formal_artifacts(second,manifest,universe,repository_state,determinism)
    if artifacts != other:
        determinism={**determinism,"deterministic":False,"byte_identical":False,"final_artifacts_identical":False}
        artifacts=build_formal_artifacts(first,manifest,universe,repository_state,determinism)
    else: determinism["final_artifacts_identical"]=True
    atomic_write_formal_artifacts(output,artifacts,repo)
    return artifacts

def evaluate_cache(cache_dir: Path, output_dir: Path, universe: pd.DataFrame, repo: Path, commit: str = "UNKNOWN") -> dict[str, bytes]:
    """Backward-compatible cache-only evaluation entrypoint used by synthetic tests."""
    state={"head":commit if commit != "UNKNOWN" else "SYNTHETIC","branch":"SYNTHETIC"}
    manifest=validate_formal_inputs(cache_dir,output_dir,universe,repo)
    core=run_formal_core_once(cache_dir,universe,manifest)
    return build_formal_artifacts(core,manifest,universe,state,{"deterministic":False,"byte_identical":False,"comparisons":{}})

def write_artifacts(output: Path, artifacts: Mapping[str,bytes], repo: Path) -> None:
    atomic_write_formal_artifacts(output,artifacts,repo)
