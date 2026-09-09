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


def simulate_portfolio(
    signals: list[dict[str, Any]],
    starting_cash: float,
    settings: PortfolioSettings,
    calendar_dates: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute ranked orders using one cash account.

    Proceeds from an exit are pending until the next calendar date.  Orders are
    therefore always processed before same-day stops/time exits can fund cash.
    The returned first list contains both fills and every skipped order.
    """
    if settings.lot_size <= 0 or settings.max_open_positions <= 0:
        raise ValueError("portfolio limits must be positive")
    cash = float(starting_cash)
    pending_cash = 0.0
    positions: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    signal_dates = {
        pd.Timestamp(value)
        for signal in signals
        for key in ("entry_date", "exit_date", "signal_date")
        if (value := signal.get(key)) not in (None, "")
    }
    calendar_values = [] if calendar_dates is None else calendar_dates
    dates = sorted({pd.Timestamp(value) for value in calendar_values} | signal_dates)

    for date in dates:
        cash += pending_cash
        pending_cash = 0.0
        todays = sorted(
            (
                signal for signal in signals
                if pd.Timestamp(signal.get("order_date") or signal.get("entry_date") or signal["signal_date"]) == date
            ),
            key=lambda signal: (-float(signal["prob"]), str(signal["code"])),
        )
        for signal in todays:
            available_before = cash
            result = dict(signal)
            result["available_cash"] = round(available_before, 8)
            preset_reason = signal.get("skip_reason")
            if preset_reason:
                result.update(status=preset_reason, qty=0)
                results.append(result)
                continue
            entry_price = float(signal.get("entry_price", 0.0))
            exit_price = float(signal.get("exit_price", 0.0))
            if not pd.notna(entry_price) or entry_price <= 0 or not pd.notna(exit_price) or exit_price < 0:
                result.update(status="SKIPPED_INVALID_PRICE", qty=0)
                results.append(result)
                continue
            if len(positions) >= settings.max_open_positions:
                result.update(status="SKIPPED_MAX_OPEN_POSITIONS", qty=0)
                results.append(result)
                continue
            commission_pct = float(signal.get("commission_percent", 0.0))
            unit_cost = entry_price * (1 + commission_pct / 100)
            position_cap = starting_cash * settings.max_position_percent / 100
            if position_cap < unit_cost * settings.lot_size:
                result.update(status="SKIPPED_POSITION_LIMIT", qty=0)
                results.append(result)
                continue
            if cash < unit_cost * settings.lot_size:
                result.update(status="SKIPPED_INSUFFICIENT_CASH", qty=0)
                results.append(result)
                continue
            allocation = min(cash, position_cap)
            qty = int(allocation / unit_cost / settings.lot_size) * settings.lot_size
            if qty <= 0:
                result.update(status="SKIPPED_POSITION_LIMIT", qty=0)
                results.append(result)
                continue
            entry_value = qty * entry_price
            entry_commission = entry_value * commission_pct / 100
            cash -= entry_value + entry_commission
            position = dict(
                signal,
                qty=qty,
                entry_value=entry_value,
                entry_commission=entry_commission,
                exit_date=pd.Timestamp(signal["exit_date"]),
            )
            positions.append(position)
            result.update(status="FILLED", qty=qty)
            results.append(result)

        closing = sorted(
            (position for position in positions if position["exit_date"] == date),
            key=lambda position: str(position["code"]),
        )
        for position in closing:
            exit_value = position["qty"] * float(position["exit_price"])
            exit_commission = exit_value * float(position.get("commission_percent", 0.0)) / 100
            proceeds = exit_value - exit_commission
            pending_cash += proceeds
            profit = proceeds - position["entry_value"] - position["entry_commission"]
            for result in reversed(results):
                if result.get("status") == "FILLED" and result.get("code") == position["code"] and result.get("signal_date") == position.get("signal_date"):
                    result.update(
                        entry_commission=round(position["entry_commission"], 8),
                        exit_commission=round(exit_commission, 8),
                        profit=round(profit, 8),
                    )
                    break
            positions.remove(position)

        locked = sum(position["entry_value"] for position in positions)
        ledger.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "cash": round(cash, 8),
                "available_cash": round(cash, 8),
                "pending_cash": round(pending_cash, 8),
                "locked_capital": round(locked, 8),
                "open_positions": len(positions),
                "equity": round(cash + pending_cash + locked, 8),
            }
        )
        if cash < -1e-8:
            raise AssertionError("negative cash")
        if len(positions) > settings.max_open_positions:
            raise AssertionError("max_open_positions exceeded")
    return results, ledger
