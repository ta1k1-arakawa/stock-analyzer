"""Read-only diagnostics shared by the immutable baseline comparison runner."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import socket
import subprocess
from typing import Any, Iterator

import pandas as pd

from src.trade_simulator import PortfolioSettings, simulate_execution, simulate_portfolio


class ComparisonError(RuntimeError):
    """Raised when comparison preconditions or invariants are violated."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_file_unchanged(path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ComparisonError(f"protected file changed: {path}")


def worktree_files(path: str | Path) -> dict[str, str]:
    """Hash every non-.git file, including ignored files and caches."""
    root = Path(path)
    return {
        file.relative_to(root).as_posix(): sha256_file(file)
        for file in sorted(root.rglob("*"))
        if file.is_file() and ".git" not in file.relative_to(root).parts
    }


def verify_baseline(path: str | Path, expected_commit: str) -> dict[str, str]:
    root = Path(path).resolve()
    if not root.is_dir():
        raise ComparisonError(f"baseline worktree missing: {root}")
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != expected_commit:
        raise ComparisonError(f"baseline HEAD mismatch: {head} != {expected_commit}")
    symbolic = subprocess.run(
        ["git", "-C", str(root), "symbolic-ref", "-q", "HEAD"],
        text=True, capture_output=True, check=False,
    )
    if symbolic.returncode == 0:
        raise ComparisonError("baseline must be at detached HEAD")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if status.strip():
        raise ComparisonError(f"baseline worktree is dirty:\n{status}")
    return worktree_files(root)


def assert_baseline_unchanged(
    path: str | Path, expected_commit: str, before: dict[str, str]
) -> None:
    after = verify_baseline(path, expected_commit)
    if after != before:
        added = sorted(set(after) - set(before))
        removed = sorted(set(before) - set(after))
        changed = sorted(key for key in set(before) & set(after) if before[key] != after[key])
        raise ComparisonError(
            f"baseline files changed; added={added}, removed={removed}, changed={changed}"
        )


def _blocked_network(*_args: Any, **_kwargs: Any) -> None:
    raise ComparisonError("external network access is forbidden during comparison")


@contextmanager
def forbid_network() -> Iterator[None]:
    """Testable process-local guard. The legacy subprocess installs the same guard."""
    import requests

    old_get = requests.get
    old_create = socket.create_connection
    old_connect = socket.socket.connect
    requests.get = _blocked_network
    socket.create_connection = _blocked_network
    socket.socket.connect = _blocked_network
    try:
        yield
    finally:
        requests.get = old_get
        socket.create_connection = old_create
        socket.socket.connect = old_connect


def build_execution_orders(
    predictions: pd.DataFrame,
    rules: dict[str, dict[str, float]],
    prices: dict[str, pd.DataFrame],
    future_days: int,
    commission_percent: float,
    entry_slippage_percent: float,
    exit_slippage_percent: float,
    stop_slippage_percent: float,
    signal_column: str = "is_signal",
) -> list[dict[str, Any]]:
    """Apply v2 execution only; supplied predictions and rules remain immutable."""
    orders: list[dict[str, Any]] = []
    frame = predictions.sort_values(["signal_date", "code"], kind="mergesort")
    for row in frame.to_dict("records"):
        if not bool(row[signal_column]):
            continue
        code = str(row["code"])
        signal_date = pd.Timestamp(row["signal_date"])
        stock_prices = prices[code]
        if signal_date not in stock_prices.index:
            orders.append({
                "code": code, "signal_date": signal_date.strftime("%Y-%m-%d"),
                "order_date": signal_date.strftime("%Y-%m-%d"),
                "planned_entry_date": "", "prob": float(row["prob"]),
                "skip_reason": "SKIPPED_NO_FUTURE_DATA",
            })
            continue
        pos = int(stock_prices.index.get_loc(signal_date))
        execution = simulate_execution(
            stock_prices, pos, future_days, float(rules[code]["stop_loss_percent"]),
            entry_slippage_percent, exit_slippage_percent, stop_slippage_percent,
            commission_percent,
        )
        if execution is None:
            orders.append({
                "code": code, "signal_date": signal_date.strftime("%Y-%m-%d"),
                "order_date": signal_date.strftime("%Y-%m-%d"),
                "planned_entry_date": "", "prob": float(row["prob"]),
                "skip_reason": "SKIPPED_NO_FUTURE_DATA",
            })
            continue
        entry_date = stock_prices.index[execution.entry_index].strftime("%Y-%m-%d")
        orders.append({
            "code": code,
            "signal_date": signal_date.strftime("%Y-%m-%d"),
            "order_date": entry_date,
            "planned_entry_date": entry_date,
            "entry_date": entry_date,
            "exit_date": stock_prices.index[execution.exit_index].strftime("%Y-%m-%d"),
            "prob": float(row["prob"]),
            "entry_price": float(execution.entry_price),
            "exit_price": float(execution.exit_price),
            "exit_reason": execution.exit_reason,
            "return_percent": float(execution.return_percent),
            "commission_percent": float(commission_percent),
            "stop_loss_percent": float(rules[code]["stop_loss_percent"]),
        })
    return orders


def run_v2_portfolio(
    orders: list[dict[str, Any]], budget: float, settings: PortfolioSettings,
    calendar_dates: list[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, ledger = simulate_portfolio(orders, budget, settings, calendar_dates)
    return pd.DataFrame(results), pd.DataFrame(ledger)


def run_independent_budget(
    orders: list[dict[str, Any]], budget: float, lot_size: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnostic legacy-style evaluator: every signal independently gets budget."""
    rows: list[dict[str, Any]] = []
    for order in orders:
        result = dict(order)
        if order.get("skip_reason"):
            result.update(status=order["skip_reason"], qty=0, available_cash=budget)
            rows.append(result)
            continue
        entry = float(order.get("entry_price", 0))
        commission = float(order.get("commission_percent", 0))
        unit_cost = entry * (1 + commission / 100)
        qty = int(budget / unit_cost / lot_size) * lot_size if unit_cost > 0 else 0
        if qty <= 0:
            result.update(status="SKIPPED_INSUFFICIENT_CASH", qty=0, available_cash=budget)
        else:
            entry_value = entry * qty
            exit_value = float(order["exit_price"]) * qty
            result.update(
                status="FILLED", qty=qty, available_cash=budget,
                entry_commission=entry_value * commission / 100,
                exit_commission=exit_value * commission / 100,
                profit=(exit_value - entry_value)
                - (entry_value + exit_value) * commission / 100,
            )
        rows.append(result)
    results = pd.DataFrame(rows)
    overlap = capital_overlap(results, budget)
    ledger = pd.DataFrame({
        "date": overlap.get("date", pd.Series(dtype=str)),
        "equity": budget + overlap.get("cumulative_realized_profit", pd.Series(dtype=float)),
        "locked_capital": overlap.get("locked_capital", pd.Series(dtype=float)),
        "open_positions": overlap.get("open_positions", pd.Series(dtype=int)),
    })
    return results, ledger


def capital_overlap(results: pd.DataFrame, budget: float) -> pd.DataFrame:
    filled = results[results.get("status", pd.Series(dtype=str)) == "FILLED"].copy()
    if filled.empty:
        return pd.DataFrame(columns=[
            "date", "locked_capital", "open_positions", "overlap_amount",
            "cumulative_realized_profit",
        ])
    filled["entry_date"] = pd.to_datetime(filled["entry_date"])
    filled["exit_date"] = pd.to_datetime(filled["exit_date"])
    dates = pd.date_range(filled["entry_date"].min(), filled["exit_date"].max(), freq="D")
    rows = []
    for date in dates:
        active = filled[(filled["entry_date"] <= date) & (filled["exit_date"] >= date)]
        realized = filled[filled["exit_date"] <= date]
        locked = float((active["entry_price"] * active["qty"]).sum())
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "locked_capital": locked,
            "open_positions": int(len(active)),
            "overlap_amount": max(0.0, locked - budget),
            "cumulative_realized_profit": float(realized["profit"].sum()),
        })
    return pd.DataFrame(rows)


def scenario_metrics(
    results: pd.DataFrame, ledger: pd.DataFrame, budget: float,
    prices: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    filled = results[results.get("status", pd.Series(dtype=str)) == "FILLED"].copy()
    skipped = results[results.get("status", pd.Series(dtype=str)) != "FILLED"].copy()
    profits = pd.to_numeric(filled.get("profit", pd.Series(dtype=float)), errors="coerce").fillna(0)
    exit_dates = pd.to_datetime(filled.get("exit_date", pd.Series(dtype=str)), errors="coerce")
    monthly = profits.groupby(exit_dates.dt.to_period("M")).sum() if len(profits) else pd.Series(dtype=float)
    if not ledger.empty and "equity" in ledger:
        equity = pd.to_numeric(ledger["equity"], errors="coerce").fillna(budget)
    else:
        ordered = filled.assign(_exit=exit_dates).sort_values("_exit")
        equity = budget + pd.to_numeric(ordered.get("profit", pd.Series(dtype=float))).cumsum()
    peak = equity.cummax() if len(equity) else equity
    drawdown = equity - peak if len(equity) else equity
    overlap = capital_overlap(filled, budget)
    normal_stops = gap_stops = 0
    for trade in filled.to_dict("records"):
        if trade.get("exit_reason") != "STOP":
            continue
        code = str(trade["code"])
        date = pd.Timestamp(trade["exit_date"])
        stop = float(trade["entry_price"]) * (1 - float(trade.get("stop_loss_percent", 0)) / 100)
        if date in prices[code].index and float(prices[code].loc[date, "Open"]) <= stop:
            gap_stops += 1
        else:
            normal_stops += 1
    duplicate_ids: set[tuple[str, str]] = set()
    if not filled.empty:
        for _, point in overlap[overlap["locked_capital"] > budget + 1e-8].iterrows():
            date = pd.Timestamp(point["date"])
            active = filled[
                (pd.to_datetime(filled["entry_date"]) <= date)
                & (pd.to_datetime(filled["exit_date"]) >= date)
            ]
            duplicate_ids.update((str(row.code), str(row.signal_date)) for row in active.itertuples())
    return {
        "profit": round(float(profits.sum()), 8),
        "trades": int(len(filled)),
        "win_rate": round(float((profits > 0).mean() * 100), 8) if len(profits) else 0.0,
        "max_drawdown": round(float(drawdown.min()), 8) if len(drawdown) else 0.0,
        "max_drawdown_percent": round(float((equity / peak - 1).min() * 100), 8) if len(equity) else 0.0,
        "monthly_win_rate": round(float((monthly > 0).mean() * 100), 8) if len(monthly) else 0.0,
        "profit_by_stock": {
            str(code): round(float(group["profit"].sum()), 8)
            for code, group in filled.groupby("code", sort=True)
        },
        "normal_stop_count": normal_stops,
        "gap_down_stop_count": gap_stops,
        "skip_counts": {
            str(reason): int(count)
            for reason, count in skipped.get("status", pd.Series(dtype=str)).value_counts().sort_index().items()
        },
        "max_simultaneous_locked_capital": round(float(overlap["locked_capital"].max()), 8) if len(overlap) else 0.0,
        "max_simultaneous_positions": int(overlap["open_positions"].max()) if len(overlap) else 0,
        "max_capital_overlap": round(float(overlap["overlap_amount"].max()), 8) if len(overlap) else 0.0,
        "duplicate_capital_trade_count": len(duplicate_ids),
    }


def deterministic_hashes(directory: str | Path) -> dict[str, str]:
    root = Path(directory)
    return {
        path.name: sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "run_metadata.json"
    }
