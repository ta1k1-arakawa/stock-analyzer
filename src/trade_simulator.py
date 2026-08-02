"""Deterministic execution primitives shared by labels and evaluator-v2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd


@dataclass(frozen=True)
class Execution:
    entry_index: int; exit_index: int; entry_price: float; exit_price: float
    exit_reason: str; return_percent: float


def simulate_execution(df: pd.DataFrame, signal_pos: int, future_days: int,
                       stop_loss_percent: float, entry_slippage_percent: float = 0.0,
                       exit_slippage_percent: float = 0.0, stop_slippage_percent: float = 0.0,
                       commission_percent: float = 0.0) -> Execution | None:
    """One-share execution.  A gap through stop fills from Open, never stop price."""
    entry = signal_pos + 1
    exit_at = entry + future_days - 1
    if entry >= len(df) or exit_at >= len(df): return None
    entry_price = float(df.iloc[entry]["Open"]) * (1 + entry_slippage_percent / 100)
    if entry_price <= 0: return None
    stop = entry_price * (1 - stop_loss_percent / 100)
    for i in range(entry, exit_at + 1):
        row = df.iloc[i]
        if float(row["Low"]) <= stop:
            base = float(row["Open"]) if float(row["Open"]) <= stop else stop
            exit_price = base * (1 - stop_slippage_percent / 100)
            break
    else:
        i = exit_at; exit_price = float(df.iloc[i]["Close"]) * (1 - exit_slippage_percent / 100)
    cost = (entry_price + exit_price) * commission_percent / 100
    return Execution(entry, i, entry_price, exit_price, "STOP" if i != exit_at or float(df.iloc[i]["Low"]) <= stop else "TIME", ((exit_price-entry_price-cost)/entry_price)*100)


@dataclass(frozen=True)
class PortfolioSettings:
    lot_size: int = 1; max_position_percent: float = 100.0; max_open_positions: int = 1


def simulate_portfolio(signals: list[dict[str, Any]], starting_cash: float, settings: PortfolioSettings) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Signals must contain entry_date, exit_date, entry_price, exit_price and probability.
    Cash from exits becomes available only on the following event date, preventing time reversal.
    """
    cash = float(starting_cash); positions: list[dict[str, Any]]=[]; trades=[]; ledger=[]
    dates = sorted({pd.Timestamp(x[k]) for x in signals for k in ("entry_date", "exit_date")})
    for date in dates:
        # Releases were scheduled at prior close, so only exits before today are usable now.
        for p in [x for x in positions if x["release_date"] < date]:
            cash += p["proceeds"]; positions.remove(p)
        todays = sorted((x for x in signals if pd.Timestamp(x["entry_date"]) == date), key=lambda x: (-float(x["prob"]), str(x["code"])))
        for s in todays:
            cap = min(cash, starting_cash * settings.max_position_percent / 100)
            qty = int(cap / float(s["entry_price"]) / settings.lot_size) * settings.lot_size
            result = dict(s, qty=qty, status="FILLED" if qty and len(positions)<settings.max_open_positions else "SKIPPED_INSUFFICIENT_CASH")
            if qty and len(positions) < settings.max_open_positions:
                cost=qty*float(s["entry_price"]); cash -= cost
                positions.append(dict(release_date=pd.Timestamp(s["exit_date"]), proceeds=qty*float(s["exit_price"])))
            trades.append(result)
        ledger.append({"date": date.strftime("%Y-%m-%d"), "cash": cash, "open_positions": len(positions)})
        if cash < -1e-8: raise AssertionError("negative cash")
    return trades, ledger
