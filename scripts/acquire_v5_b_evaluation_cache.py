"""Acquisition-only runner for the future V5-B evaluation cache.

Never imported by the evaluator and never run during this implementation task.
"""
from __future__ import annotations
import argparse, json, os, tempfile, urllib.parse, urllib.request, sys
from hashlib import sha256
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.v5_b_candidate_ranker import canonical_ticker, normalize_universe

HOST="query1.finance.yahoo.com"; CONFIRM="V5_B_EVALUATION_CACHE_ACQUISITION"
START="2019-01-01"; END="2026-01-31"

def chart_url(ticker: str) -> str:
    p=int(pd.Timestamp(START).timestamp()); q=int(pd.Timestamp(END).timestamp())+86400
    return f"https://{HOST}/v8/finance/chart/{urllib.parse.quote(canonical_ticker(ticker)+'.T')}?period1={p}&period2={q}&interval=1d&events=div%2Csplits&includeAdjustedClose=true"

def acquire(output_dir: Path, universe_csv: Path, confirmation: str, opener=urllib.request.urlopen) -> dict:
    if confirmation!=CONFIRM: raise ValueError("CONFIRMATION_REQUIRED")
    if output_dir.exists() and any(output_dir.iterdir()): raise ValueError("OUTPUT_NONEMPTY")
    u=normalize_universe(pd.read_csv(universe_csv));
    if len(u)!=300: raise ValueError("UNIVERSE_COUNT_MISMATCH")
    stage=Path(tempfile.mkdtemp(prefix=output_dir.name+".staging.",dir=str(output_dir.parent))); raw=stage/"raw"; raw.mkdir()
    outcomes=[]
    try:
        for ticker in u.ticker:
            url=chart_url(ticker); rec={"ticker":ticker,"url":url,"host":HOST,"request_start":START,"request_end":END}
            try:
                with opener(url,timeout=30) as response:
                    host=urllib.parse.urlparse(getattr(response,"url",url)).hostname
                    if host!=HOST: raise ValueError("RESPONSE_HOST_MISMATCH")
                    body=response.read(); status=getattr(response,"status",200)
                if status!=200: raise ValueError(f"HTTP_STATUS_{status}")
                rel=Path("raw")/(ticker+".json"); path=stage/rel; path.write_bytes(body); digest=sha256(body).hexdigest(); rec.update({"success":True,"http_status":status,"relative_path":str(rel).replace("\\","/"),"sha256":digest,"byte_count":len(body)})
            except Exception as exc:
                rec.update({"success":False,"exception_class":type(exc).__name__,"exception":str(exc)})
            outcomes.append(rec)
        payloads=[x for x in outcomes if x.get("success")]; failed=[x["ticker"] for x in outcomes if not x.get("success")]
        manifest={"schema_version":1,"complete":len(outcomes)==300,"universe_csv_sha256":sha256(universe_csv.read_bytes()).hexdigest(),"ticker_list_sha256":sha256(json.dumps(list(u.ticker),separators=(",",":" )).encode()).hexdigest(),"ticker_count":300,"request_start":START,"request_end":END,"host":HOST,"outcomes":outcomes,"payloads":payloads,"failed_tickers":failed,"success_count":len(payloads),"failed_count":len(failed)}
        manifest["payload_hash_list_sha256"]=sha256(json.dumps([x["sha256"] for x in payloads],separators=(",",":")).encode()).hexdigest()
        (stage/"cache_manifest.json").write_text(json.dumps(manifest,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n",encoding="utf-8")
        os.replace(stage,output_dir); return manifest
    except Exception:
        # An interrupted/failed acquisition is never published as a complete cache.
        import shutil; shutil.rmtree(stage,ignore_errors=True); raise

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",required=True); ap.add_argument("--universe-csv",default=str(Path(__file__).resolve().parents[1]/"V4_UNIVERSE.csv")); ap.add_argument("--confirmation",required=True); a=ap.parse_args(); m=acquire(Path(a.output_dir),Path(a.universe_csv),a.confirmation); print(json.dumps({"complete":m["complete"],"manifest_sha256":sha256((Path(a.output_dir)/"cache_manifest.json").read_bytes()).hexdigest()},sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())
