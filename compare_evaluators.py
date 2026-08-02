"""Run immutable baseline with a fixed CSV loader; baseline files are never written."""
from __future__ import annotations
import argparse, hashlib, json, runpy, sys
from pathlib import Path
import pandas as pd

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument("--baseline-worktree", required=True); p.add_argument("--prices", required=True); p.add_argument("--output", default="data/backtest_results/comparison.json"); a=p.parse_args()
 prices=Path(a.prices); digest=hashlib.sha256(prices.read_bytes()).hexdigest()
 # Import baseline only through sys.path and monkeypatch its network fetch method.
 sys.path.insert(0, str(Path(a.baseline_worktree).resolve()))
 from src.fetchers.yfinance import YFinanceFetcher
 def fixed(self, code, start, end):
  d=pd.read_csv(prices / f"{code}.csv", index_col=0, parse_dates=True)
  return d[(d.index >= pd.Timestamp(start)) & (d.index <= pd.Timestamp(end))]
 YFinanceFetcher.get_daily_stock_prices=fixed
 result={"baseline_commit": __import__("subprocess").check_output(["git","-C",a.baseline_worktree,"rev-parse","HEAD"], text=True).strip(), "price_snapshot_sha256":digest, "note":"baseline loader monkeypatched; baseline worktree unchanged"}
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
if __name__ == "__main__": main()
