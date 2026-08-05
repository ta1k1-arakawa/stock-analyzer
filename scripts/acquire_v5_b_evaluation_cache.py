"""Safe, sequential Yahoo evaluation-cache acquisition.

This module is transport-only.  It never fits a model or runs a portfolio.
All network calls are injected in tests; this turn does not invoke ``main``.
"""
from __future__ import annotations
import argparse, datetime as dt, json, os, shutil, sys, tempfile, time, urllib.error, urllib.parse, urllib.request
from hashlib import sha256
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.v5_b_candidate_ranker import canonical_ticker, normalize_universe

HOST="query1.finance.yahoo.com"; CONFIRM="V5_B_EVALUATION_CACHE_ACQUISITION"
PREFLIGHT_CONFIRM="V5_B_YAHOO_PREFLIGHT"; START="2019-01-01"; END="2026-01-31"; END_EXCLUSIVE="2026-02-01"
INTERVAL_SECONDS=2.0; USER_AGENT="V5-B-Cache-Acquirer/1.0 (research; contact=local)"
HEADERS={"User-Agent":USER_AGENT,"Accept":"application/json,text/plain,*/*","Accept-Language":"ja,en-US;q=0.9,en;q=0.8","Accept-Encoding":"identity","Connection":"close"}

def _epoch(value: str) -> int:
    return int(dt.datetime.fromisoformat(value).replace(tzinfo=dt.timezone.utc).timestamp())

def chart_url(ticker: str) -> str:
    return f"https://{HOST}/v8/finance/chart/{urllib.parse.quote(canonical_ticker(ticker)+'.T')}?period1={_epoch(START)}&period2={_epoch(END_EXCLUSIVE)}&interval=1d&events=div%2Csplits&includeAdjustedClose=true"

def make_request(ticker: str) -> urllib.request.Request:
    return urllib.request.Request(chart_url(ticker), headers=dict(HEADERS), method="GET")

def _payload_info(body: bytes, ticker: str, response_host: str) -> dict:
    if response_host!=HOST: raise ValueError("RESPONSE_HOST_MISMATCH")
    obj=json.loads(body.decode("utf-8")); chart=obj.get("chart",{})
    if chart.get("error") is not None: raise ValueError("CHART_ERROR")
    result=chart.get("result") or []
    if not result: raise ValueError("CHART_RESULT_EMPTY")
    r=result[0]; meta=r.get("meta",{}); symbol=canonical_ticker(meta.get("symbol",ticker))
    if symbol!=canonical_ticker(ticker): raise ValueError("SYMBOL_MISMATCH")
    ts=r.get("timestamp") or []; q=(r.get("indicators",{}).get("quote") or [{}])[0]; adj=(r.get("indicators",{}).get("adjclose") or [{}])[0].get("adjclose")
    if not ts or adj is None: raise ValueError("PRICE_ARRAY_EMPTY")
    required=("open","high","low","close","volume")
    if any(q.get(k) is None for k in required): raise ValueError("QUOTE_FIELD_MISSING")
    lengths=[len(ts),len(adj)]+[len(q[k]) for k in required]
    if len(set(lengths))!=1: raise ValueError("PRICE_ARRAY_LENGTH_MISMATCH")
    dates=pd.to_datetime(ts,unit="s",utc=True).tz_convert("Asia/Tokyo").tz_localize(None).normalize()
    return {"row_count":len(ts),"min_date":dates.min().strftime("%Y-%m-%d"),"max_date":dates.max().strftime("%Y-%m-%d")}

def _retry_after(exc) -> float:
    value=getattr(exc,"headers",{}).get("Retry-After") if getattr(exc,"headers",None) else None
    try: return min(120.0,float(value)) if value is not None else 60.0
    except (TypeError,ValueError): return 60.0

def preflight(ticker: str, opener=urllib.request.urlopen) -> dict:
    req=make_request(ticker)
    try:
        with opener(req,timeout=30) as response:
            host=urllib.parse.urlparse(getattr(response,"url",req.full_url)).hostname; status=getattr(response,"status",200); body=response.read()
        if status==429:
            return {"preflight":"BLOCKED","status":429,"retry_after":getattr(response,"headers",{}).get("Retry-After"),"request_count":1}
        if status!=200: return {"preflight":"BLOCKED","status":status,"request_count":1}
        info=_payload_info(body,ticker,host); return {"preflight":"PASS","ticker":canonical_ticker(ticker),"status":200,"host":host,**info}
    except urllib.error.HTTPError as exc:
        return {"preflight":"BLOCKED","status":exc.code,"retry_after":exc.headers.get("Retry-After"),"request_count":1}
    except Exception as exc:
        return {"preflight":"BLOCKED","error":type(exc).__name__,"request_count":1}

def acquire(output_dir: Path, universe_csv: Path, confirmation: str, opener=urllib.request.urlopen, sleep_fn=time.sleep, monotonic=time.monotonic, interval: float=INTERVAL_SECONDS) -> dict:
    if confirmation!=CONFIRM: raise ValueError("CONFIRMATION_REQUIRED")
    if output_dir.exists(): raise ValueError("OUTPUT_EXISTS")
    u=normalize_universe(pd.read_csv(universe_csv));
    if len(u)!=300: raise ValueError("UNIVERSE_COUNT_MISMATCH")
    stage=Path(tempfile.mkdtemp(prefix=output_dir.name+".staging.",dir=str(output_dir.parent))); (stage/"raw").mkdir(); outcomes=[]; request_count=retry_count=http429_count=server_count=0; consecutive429=0; circuit_reason=None; last_start=None
    try:
        for ticker in u.ticker:
            ticker=canonical_ticker(ticker); rec={"ticker":ticker,"host":HOST,"request_start":START,"request_end":END,"url":chart_url(ticker),"retry_count":0}; attempts=0; done=False
            while attempts<2 and not done:
                if last_start is not None:
                    wait=max(0.0,interval-(monotonic()-last_start));
                    if wait: sleep_fn(wait)
                last_start=monotonic(); attempts+=1; request_count+=1
                try:
                    req=make_request(ticker)
                    with opener(req,timeout=30) as response:
                        response_host=urllib.parse.urlparse(getattr(response,"url",req.full_url)).hostname; status=getattr(response,"status",200); body=response.read()
                    rec["http_status"]=status
                    if status==429: raise urllib.error.HTTPError(req.full_url,429,"Too Many Requests",getattr(response,"headers",{}),None)
                    if status in (500,502,503,504): raise urllib.error.HTTPError(req.full_url,status,"Server error",getattr(response,"headers",{}),None)
                    if status!=200: raise ValueError(f"HTTP_STATUS_{status}")
                    info=_payload_info(body,ticker,response_host); rel=Path("raw")/(ticker+".json"); (stage/rel).write_bytes(body); rec.update(info,success=True,relative_path=str(rel).replace("\\","/"),sha256=sha256(body).hexdigest(),byte_count=len(body),attempts=attempts); done=True; consecutive429=0
                except urllib.error.HTTPError as exc:
                    code=exc.code; rec.update(http_status=code,exception_class=type(exc).__name__,exception=str(exc));
                    if code==429:
                        http429_count+=1; consecutive429+=1; rec["retry_after_seconds"]=_retry_after(exc)
                        if request_count==1 or consecutive429>=2 or http429_count>=3: circuit_reason=f"HTTP_429_CIRCUIT_BREAKER_{code}"; break
                        if attempts<2: sleep_fn(rec["retry_after_seconds"]); retry_count+=1; rec["retry_count"]+=1; continue
                    if code in (500,502,503,504): server_count+=1
                    if code in (500,502,503,504) and attempts<2: sleep_fn(15); retry_count+=1; rec["retry_count"]+=1; continue
                    done=True
                except Exception as exc:
                    rec.update(exception_class=type(exc).__name__,exception=str(exc));
                    if attempts<2: sleep_fn(15); retry_count+=1; rec["retry_count"]+=1; continue
                    done=True
            outcomes.append(rec)
            if circuit_reason: break
        payloads=[x for x in outcomes if x.get("success")]; failed=[x["ticker"] for x in outcomes if not x.get("success")]; complete=len(outcomes)==300 and circuit_reason is None; usable=complete and len(payloads)>0
        manifest={"schema_version":2,"complete":complete,"usable_for_evaluation":usable,"attempted_ticker_count":len(outcomes),"request_count":request_count,"retry_count":retry_count,"success_count":len(payloads),"failed_count":len(failed),"http_429_count":http429_count,"http_5xx_count":server_count,"circuit_breaker_triggered":circuit_reason is not None,"circuit_breaker_reason":circuit_reason,"fixed_request_interval_seconds":interval,"fixed_user_agent":USER_AGENT,"host":HOST,"request_start":START,"request_end":END,"ticker_count":300,"outcomes":outcomes,"payloads":payloads,"failed_tickers":failed,"payload_hash_list_sha256":sha256(json.dumps([x["sha256"] for x in payloads],separators=(",",":")).encode()).hexdigest()}
        (stage/"cache_manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n",encoding="utf-8")
        if usable: os.replace(stage,output_dir)
        else: os.replace(stage,Path(str(output_dir)+".failed"))
        return manifest
    except Exception:
        shutil.rmtree(stage,ignore_errors=True); raise

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir"); ap.add_argument("--universe-csv",default=str(Path(__file__).resolve().parents[1]/"V4_UNIVERSE.csv")); ap.add_argument("--confirmation",required=True); ap.add_argument("--preflight-only",action="store_true"); ap.add_argument("--ticker"); a=ap.parse_args()
    if a.preflight_only:
        if not a.ticker or a.confirmation!=PREFLIGHT_CONFIRM: raise SystemExit("PREFLIGHT_ARGUMENTS_REQUIRED")
        print(json.dumps(preflight(a.ticker),ensure_ascii=False,sort_keys=True)); return 0
    if not a.output_dir: raise SystemExit("OUTPUT_DIR_REQUIRED")
    m=acquire(Path(a.output_dir),Path(a.universe_csv),a.confirmation); print(json.dumps({"complete":m["complete"],"usable_for_evaluation":m["usable_for_evaluation"],"request_count":m["request_count"],"circuit_breaker_triggered":m["circuit_breaker_triggered"]},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
