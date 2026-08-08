"""Generate the only network-enabled fixed OHLCV snapshot."""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmark import REQUIRED_COLUMNS, sha256_file, snapshot_hash, validate_snapshot
from src.fetchers.yfinance import YFinanceFetcher

DATE_FROM, DATE_TO, FETCH_TO = "2020-01-01", "2026-05-20", "2026-05-21"

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument("--config",default="config.yaml"); p.add_argument("--output",default="data/benchmark"); a=p.parse_args()
 raw=yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}; codes=[str(x["code"]) for x in raw.get("stocks",[])]
 if not codes or len(codes)!=len(set(codes)): raise SystemExit("stocks must be non-empty and unique")
 root=Path(a.output); out=root/"ohlcv"; out.mkdir(parents=True,exist_ok=True); files={}; fetcher=YFinanceFetcher()
 for code in codes:
  df=fetcher.get_daily_stock_prices(code,DATE_FROM,FETCH_TO)
  if df is None or df.empty: raise SystemExit(f"fetch failed: {code}")
  df=df.copy(); df.index=pd.to_datetime(df.index).tz_localize(None).normalize(); df=df[(df.index>=DATE_FROM)&(df.index<=DATE_TO)][["Open","High","Low","Close","Volume"]].sort_index()
  if df.empty or df.index.duplicated().any() or not df.index.is_monotonic_increasing: raise SystemExit(f"invalid dates: {code}")
  if df.index[-1]!=pd.Timestamp(DATE_TO): raise SystemExit(f"period incomplete: {code} ends {df.index[-1].date()}")
  export=df.reset_index(names="Date"); export["Date"]=export["Date"].dt.strftime("%Y-%m-%d"); export=export[REQUIRED_COLUMNS]
  path=out/f"{code}.csv"; export.to_csv(path,index=False,encoding="utf-8",lineterminator="\n")
  files[code]={"code":code,"first_date":export.iloc[0]["Date"],"last_date":export.iloc[-1]["Date"],"rows":len(export),"sha256":sha256_file(path)}
 combined=snapshot_hash(files); manifest={"snapshot_id":f"yahoo-jp-adjusted-{DATE_FROM}-{DATE_TO}-{combined[:12]}","generated_at":datetime.now().astimezone().isoformat(timespec="seconds"),"source":"Yahoo Finance chart API via src.fetchers.yfinance.YFinanceFetcher","price_adjustment_method":"adjclose/close factor applied to Open, High, Low; Close replaced by adjclose; Volume unchanged","timezone":"Asia/Tokyo source timestamps normalized to timezone-naive trading dates","date_from":DATE_FROM,"date_to":DATE_TO,"columns":REQUIRED_COLUMNS,"stock_codes":codes,"files":files,"snapshot_hash":combined}
 (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); validate_snapshot(root); print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
