"""Isolated child process for executing the immutable legacy evaluator."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import sys

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--capture", required=True)
    args = parser.parse_args()
    baseline = Path(args.baseline).resolve()
    benchmark = Path(args.benchmark).resolve()
    workspace = Path(args.workspace).resolve()
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))

    frames: dict[str, pd.DataFrame] = {}
    for code, metadata in manifest["files"].items():
        path = benchmark / "ohlcv" / f"{code}.csv"
        if not path.is_file() or _sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"fixed benchmark mismatch: {code}")
        frame = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        frames[str(code)] = frame

    def blocked(*_args, **_kwargs):
        raise RuntimeError("external network access is forbidden during comparison")

    import requests
    requests.get = blocked
    socket.create_connection = blocked
    socket.socket.connect = blocked

    sys.path.insert(0, str(baseline))
    os.chdir(workspace)
    import backtest as legacy
    from src.fetchers.yfinance import YFinanceFetcher

    usage: list[dict[str, object]] = []

    def fixed_prices(self, stock_code, date_from_str=None, date_to_str=None):
        code = str(stock_code).removesuffix(".T")
        if code not in frames:
            raise RuntimeError(f"stock missing from fixed benchmark: {code}")
        frame = frames[code]
        start = pd.Timestamp(date_from_str or manifest["date_from"])
        end_exclusive = pd.Timestamp(date_to_str) if date_to_str else pd.Timestamp(manifest["date_to"]) + pd.Timedelta(days=1)
        result = frame[(frame.index >= start) & (frame.index < end_exclusive)].copy()
        usage.append(
            {
                "stock_code": code,
                "requested_from": str(start.date()),
                "requested_to_exclusive": str(end_exclusive.date()),
                "first_date": result.index.min().strftime("%Y-%m-%d"),
                "last_date": result.index.max().strftime("%Y-%m-%d"),
                "rows": len(result),
                "csv_sha256": manifest["files"][code]["sha256"],
            }
        )
        return result

    YFinanceFetcher.get_daily_stock_prices = fixed_prices
    research: dict[str, dict[str, object]] = {}
    final: dict[str, dict[str, object]] = {}
    final_selection = None
    original_research = legacy._evaluate_research_stock
    original_final = legacy._final_evaluation_rolling
    original_save = legacy._save_selection_result

    def capture_research(code, prices, config, settings):
        selected, combinations = original_research(code, prices, config, settings)
        research[str(code)] = {"selected": selected, "combinations": combinations}
        return selected, combinations

    def capture_final(prices, rule, config, settings):
        summary, trades, predictions = original_final(prices, rule, config, settings)
        final[str(rule.code)] = {
            "rule": {
                "code": str(rule.code),
                "target_percent": float(rule.target_percent),
                "stop_loss_percent": float(rule.stop_loss_percent),
                "threshold": float(rule.threshold),
            },
            "summary": summary,
            "trades": trades,
            "predictions": predictions,
        }
        return summary, trades, predictions

    def capture_save(results, settings):
        nonlocal final_selection
        final_selection = results.copy()
        return original_save(results, settings)

    legacy._evaluate_research_stock = capture_research
    legacy._final_evaluation_rolling = capture_final
    legacy._save_selection_result = capture_save
    legacy.run_backtest()
    if set(final) != set(manifest["stock_codes"]):
        raise RuntimeError(f"legacy did not evaluate every stock: {sorted(final)}")
    capture = {
        "research": research,
        "final": final,
        "final_selection": final_selection,
        "usage": usage,
        "network_attempts": 0,
    }
    Path(args.capture).write_bytes(pickle.dumps(capture, protocol=5))


if __name__ == "__main__":
    main()
