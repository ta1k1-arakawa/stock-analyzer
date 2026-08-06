"""Synthetic-only causal portfolio event engine for V6-A-R2.

This module deliberately has no cache, data-acquisition, network, model, or
formal-evaluation path.  Frames are plain mappings so the portfolio engine
has exactly one price-reading boundary: :func:`read_price`.
"""

from __future__ import annotations

import csv
import copy
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def _parse_iso_date(value: str) -> date:
    if not isinstance(value, str):
        raise ValueError("INVALID_DATE_FORMAT")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError) as error:
        raise ValueError("INVALID_DATE_FORMAT") from error
    if parsed.isoformat() != value:
        raise ValueError("INVALID_DATE_FORMAT")
    return parsed


def read_price(frames: Mapping[str, Mapping[str, Mapping[str, float]]], ticker: str,
               requested_date: str, field: str, engine_day: str) -> float:
    """Read one finite price, failing closed on invalid or future access."""
    if _parse_iso_date(requested_date) > _parse_iso_date(engine_day):
        raise ValueError("FUTURE_PRICE_ACCESS_PROHIBITED")
    if ticker not in frames:
        raise ValueError("TICKER_NOT_FOUND")
    if requested_date not in frames[ticker]:
        raise ValueError("DATE_NOT_FOUND")
    if field not in frames[ticker][requested_date]:
        raise ValueError("FIELD_NOT_FOUND")
    value = frames[ticker][requested_date][field]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("NONFINITE_PRICE")
    return float(value)


@dataclass(frozen=True)
class PendingOrder:
    order_id: str
    signal_year: int
    signal_date: str
    order_created_date: str
    entry_attempt_date: str
    planned_exit_date: str
    ticker: str
    industry: str
    rank: int
    signal_raw_close: float
    candidate_status: str


@dataclass(frozen=True)
class OpenPosition:
    order_id: str
    signal_year: int
    signal_date: str
    ticker: str
    industry: str
    rank: int
    entry_date: str
    planned_exit_date: str
    quantity: int
    entry_price: float
    entry_cost: float


@dataclass(frozen=True)
class PendingProceeds:
    order_id: str
    exit_date: str
    availability_date: str
    proceeds: float


@dataclass
class EngineState:
    engine_day: str | None = None
    available_cash: float = 400000.0
    open_positions: list[OpenPosition] = field(default_factory=list)
    pending_orders_by_entry_date: dict[str, list[PendingOrder]] = field(default_factory=dict)
    pending_proceeds_by_available_date: dict[str, list[PendingProceeds]] = field(default_factory=dict)
    completed_trades: list[dict[str, Any]] = field(default_factory=list)
    daily_equity: list[dict[str, Any]] = field(default_factory=list)
    event_audit: list[dict[str, Any]] = field(default_factory=list)


LEDGER_FIELDS = [
    "signal_year", "signal_date", "order_created_date", "entry_attempt_date",
    "entry_state_transition_date", "entry_price_source_date", "planned_exit_date",
    "exit_execution_date", "exit_price_source_date", "proceeds_available_date",
    "ticker", "industry", "rank", "status", "skip_reason", "quantity",
    "entry_price", "exit_price", "entry_cost", "exit_proceeds",
    "realized_net_profit_yen", "realized_net_return_percent", "cash_before_entry",
    "cash_after_entry", "position_count_before_entry", "position_count_after_entry",
]


def _order_id(row: Mapping[str, Any]) -> str:
    return f"{row['signal_date']}|{int(row['rank']):02d}|{row['ticker']}"


def _require_candidate(row: Mapping[str, Any], calendar: Sequence[str]) -> None:
    required = {"signal_year", "signal_date", "ticker", "industry", "rank",
                "signal_raw_close", "entry_attempt_date", "planned_exit_date",
                "candidate_status"}
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"CANDIDATE_SCHEMA_MISSING:{','.join(missing)}")
    if row["candidate_status"] != "ACCEPTED_TOP20":
        raise ValueError("CANDIDATE_STATUS_NOT_ACCEPTED_TOP20")
    rank = row["rank"]
    if not isinstance(rank, int) or isinstance(rank, bool):
        raise ValueError("INVALID_CANDIDATE_RANK")
    if rank > 20:
        raise ValueError("OUTSIDE_TOP20_CANDIDATE_PROHIBITED")
    if rank < 1:
        raise ValueError("INVALID_CANDIDATE_RANK")
    signal = str(row["signal_date"])
    entry = str(row["entry_attempt_date"])
    planned = str(row["planned_exit_date"])
    _parse_iso_date(signal)
    _parse_iso_date(entry)
    _parse_iso_date(planned)
    if signal not in calendar or entry not in calendar or planned not in calendar:
        raise ValueError("CANDIDATE_DATE_NOT_IN_COMMON_CALENDAR")
    index = {day: i for i, day in enumerate(calendar)}
    if not index[signal] < index[entry] <= index[planned]:
        raise ValueError("INVALID_CANDIDATE_DATE_ORDER")
    if index[entry] != index[signal] + 1:
        raise ValueError("ENTRY_DATE_NOT_NEXT_COMMON_CALENDAR_DAY")
    if index[planned] != index[signal] + 10:
        raise ValueError("EXIT_DATE_NOT_TENTH_COMMON_CALENDAR_DAY")
    if int(row["signal_year"]) == 2026:
        raise ValueError("SIGNAL_2026_PROHIBITED")
    if int(row["signal_year"]) != int(signal[:4]):
        raise ValueError("SIGNAL_YEAR_MISMATCH")
    if not isinstance(row["signal_raw_close"], (int, float)) or not math.isfinite(float(row["signal_raw_close"])):
        raise ValueError("NONFINITE_CANDIDATE_CLOSE")


def validate_candidate_schema(calendar: Sequence[str], candidates: Sequence[Mapping[str, Any]]) -> None:
    """Validate R2 candidate rows without constructing or running the engine."""
    calendar = tuple(calendar)
    parsed_calendar = tuple(_parse_iso_date(day) for day in calendar)
    if tuple(sorted(parsed_calendar)) != parsed_calendar or len(set(calendar)) != len(calendar):
        raise ValueError("INVALID_COMMON_CALENDAR")
    ids: set[str] = set()
    signal_tickers: set[tuple[str, str]] = set()
    signal_ranks: set[tuple[str, int]] = set()
    for row in candidates:
        _require_candidate(row, calendar)
        order_id = _order_id(row)
        ticker_key = (str(row["signal_date"]), str(row["ticker"]))
        rank_key = (str(row["signal_date"]), int(row["rank"]))
        if order_id in ids or ticker_key in signal_tickers or rank_key in signal_ranks:
            raise ValueError("DUPLICATE_CANDIDATE_KEY")
        ids.add(order_id)
        signal_tickers.add(ticker_key)
        signal_ranks.add(rank_key)


class CausalEventEngine:
    """A new, explicit five-phase state-transition engine."""

    def __init__(self, frames: Mapping[str, Mapping[str, Mapping[str, float]]],
                 calendar: Sequence[str], candidates: Sequence[Mapping[str, Any]],
                 starting_cash: float = 400000.0) -> None:
        self.frames = frames
        self.calendar = tuple(calendar)
        self._calendar_dates = tuple(_parse_iso_date(day) for day in self.calendar)
        if len(self.calendar) != len(set(self.calendar)):
            raise ValueError("DUPLICATE_COMMON_CALENDAR_DATE")
        if tuple(sorted(self._calendar_dates)) != self._calendar_dates:
            raise ValueError("COMMON_CALENDAR_NOT_SORTED")
        self._calendar_index = {day: index for index, day in enumerate(self.calendar)}
        self.state = EngineState(available_cash=float(starting_cash))
        self.candidates = [dict(row) for row in candidates]
        self._candidate_ids: set[str] = set()
        self._candidate_keys: set[tuple[str, str]] = set()
        self._candidate_ranks: set[tuple[str, int]] = set()
        for row in self.candidates:
            _require_candidate(row, self.calendar)
            oid = _order_id(row)
            key = (str(row["signal_date"]), str(row["ticker"]))
            rank_key = (str(row["signal_date"]), int(row["rank"]))
            if oid in self._candidate_ids or key in self._candidate_keys or rank_key in self._candidate_ranks:
                raise ValueError("DUPLICATE_CANDIDATE_KEY")
            self._candidate_ids.add(oid)
            self._candidate_keys.add(key)
            self._candidate_ranks.add(rank_key)
        self._safety = {name: 0 for name in (
            "negative_cash_count", "same_day_proceeds_reuse_count",
            "duplicate_order_count", "max_position_violation_count",
            "cash_reserve_violation_count", "industry_overlap_violation_count",
            "signal_2026_count", "future_price_access_violation_count",
            "d0_state_mutation_violation_count")}

    def _ledger(self, order: PendingOrder) -> dict[str, Any]:
        return {
            "signal_year": order.signal_year, "signal_date": order.signal_date,
            "order_created_date": order.order_created_date,
            "entry_attempt_date": order.entry_attempt_date,
            "entry_state_transition_date": None, "entry_price_source_date": None,
            "planned_exit_date": order.planned_exit_date,
            "exit_execution_date": None, "exit_price_source_date": None,
            "proceeds_available_date": None, "ticker": order.ticker,
            "industry": order.industry, "rank": order.rank, "status": "QUEUED",
            "skip_reason": None, "quantity": 100, "entry_price": None,
            "exit_price": None, "entry_cost": None, "exit_proceeds": None,
            "realized_net_profit_yen": None, "realized_net_return_percent": None,
            "cash_before_entry": None, "cash_after_entry": None,
            "position_count_before_entry": None, "position_count_after_entry": None,
        }

    def _ledger_for(self, order_id: str) -> dict[str, Any]:
        for row in self.state.completed_trades:
            if row["order_id"] == order_id:
                return row
        raise ValueError("LEDGER_ORDER_NOT_FOUND")

    def read_engine_price(self, ticker: str, requested_date: str, field: str,
                          engine_day: str) -> float:
        try:
            return read_price(self.frames, ticker, requested_date, field, engine_day)
        except ValueError as error:
            if str(error) == "FUTURE_PRICE_ACCESS_PROHIBITED":
                self._safety["future_price_access_violation_count"] += 1
                self.state.event_audit.append({"event": "FUTURE_PRICE_ACCESS_PROHIBITED",
                                               "date": engine_day,
                                               "requested_date": requested_date,
                                               "ticker": ticker, "field": field})
            raise

    def _ensure_ledger(self, order: PendingOrder) -> dict[str, Any]:
        for row in self.state.completed_trades:
            if row["order_id"] == order.order_id:
                return row
        row = self._ledger(order) | {"order_id": order.order_id}
        self.state.completed_trades.append(row)
        return row

    def _next_day(self, day: str) -> str:
        index = self.calendar.index(day)
        if index + 1 >= len(self.calendar):
            raise ValueError("NEXT_PROCEEDS_DATE_UNAVAILABLE")
        return self.calendar[index + 1]

    def phase1_release_proceeds(self, day: str) -> None:
        self.state.engine_day = day
        proceeds = self.state.pending_proceeds_by_available_date.pop(day, [])
        for item in proceeds:
            self.state.available_cash += item.proceeds
            self.state.event_audit.append({"event": "PROCEEDS_RELEASED", "date": day,
                                           "order_id": item.order_id, "amount": item.proceeds,
                                           "exit_date": item.exit_date})
        if self.state.available_cash < 0:
            self._safety["negative_cash_count"] += 1

    def phase2_attempt_entries(self, day: str) -> None:
        orders = list(self.state.pending_orders_by_entry_date.pop(day, []))
        orders.sort(key=lambda order: (order.rank, order.ticker))
        for order in orders:
            ledger = self._ensure_ledger(order)
            cash_before = self.state.available_cash
            position_count_before = len(self.state.open_positions)
            signal_close = self.read_engine_price(order.ticker, order.signal_date, "Close", day)
            if not math.isclose(signal_close, order.signal_raw_close, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("CANDIDATE_SIGNAL_CLOSE_MISMATCH")
            raw_open = self.read_engine_price(order.ticker, day, "Open", day)
            tickers = {position.ticker for position in self.state.open_positions}
            industries = {position.industry for position in self.state.open_positions}
            reason = None
            if order.ticker in tickers:
                reason = "DUPLICATE_TICKER_OPEN"
            elif order.industry in industries:
                reason = "SAME_INDUSTRY_OPEN"
            elif len(self.state.open_positions) >= 2:
                reason = "MAX_OPEN_POSITIONS"
            elif raw_open > signal_close * 1.02:
                reason = "ENTRY_GAP_TOO_HIGH"
            else:
                entry_price = raw_open * 1.0003
                entry_cost = entry_price * 100
                if entry_cost > 220000:
                    reason = "CAPITAL_LIMIT"
                elif self.state.available_cash - entry_cost < 40000:
                    reason = "CASH_RESERVE"
            if reason is not None:
                ledger.update({"status": "SKIPPED", "skip_reason": reason,
                               "entry_state_transition_date": day,
                               "cash_before_entry": cash_before,
                               "cash_after_entry": self.state.available_cash,
                               "position_count_before_entry": position_count_before,
                               "position_count_after_entry": len(self.state.open_positions)})
                self.state.event_audit.append({"event": "ENTRY_SKIPPED", "date": day,
                                               "order_id": order.order_id, "reason": reason})
                continue
            entry_price = raw_open * 1.0003
            entry_cost = entry_price * 100
            self.state.available_cash -= entry_cost
            position = OpenPosition(order_id=order.order_id, signal_year=order.signal_year,
                                    signal_date=order.signal_date, ticker=order.ticker,
                                    industry=order.industry, rank=order.rank, entry_date=day,
                                    planned_exit_date=order.planned_exit_date, quantity=100,
                                    entry_price=entry_price, entry_cost=entry_cost)
            self.state.open_positions.append(position)
            ledger.update({"status": "FILLED", "entry_state_transition_date": day,
                           "entry_price_source_date": day, "entry_price": entry_price,
                           "entry_cost": entry_cost, "cash_before_entry": cash_before,
                           "cash_after_entry": self.state.available_cash,
                           "position_count_before_entry": position_count_before,
                           "position_count_after_entry": len(self.state.open_positions)})
            self.state.event_audit.extend([
                {"event": "CASH_DEDUCTED", "date": day, "order_id": order.order_id,
                 "amount": entry_cost, "cash_after": self.state.available_cash},
                {"event": "ENTRY_FILLED", "date": day, "order_id": order.order_id,
                 "position_count_after": len(self.state.open_positions)},
            ])
            if self.state.available_cash < 0:
                self._safety["negative_cash_count"] += 1

    def phase3_execute_exits(self, day: str) -> None:
        exits = [position for position in self.state.open_positions if position.planned_exit_date == day]
        for position in exits:
            next_day = self._next_day(day)
            cash_before_exit = self.state.available_cash
            raw_open = self.read_engine_price(position.ticker, day, "Open", day)
            exit_price = raw_open * 0.9997
            proceeds = exit_price * position.quantity
            self.state.open_positions.remove(position)
            self.state.pending_proceeds_by_available_date.setdefault(next_day, []).append(
                PendingProceeds(position.order_id, day, next_day, proceeds))
            ledger = self._ledger_for(position.order_id)
            profit = proceeds - float(ledger["entry_cost"])
            ledger.update({"status": "CLOSED", "exit_execution_date": day,
                           "exit_price_source_date": day, "exit_price": exit_price,
                           "exit_proceeds": proceeds, "realized_net_profit_yen": profit,
                           "realized_net_return_percent": profit / float(ledger["entry_cost"]) * 100.0,
                           "proceeds_available_date": next_day})
            self.state.event_audit.append({"event": "EXIT_EXECUTED", "date": day,
                                           "order_id": position.order_id, "exit_proceeds": proceeds,
                                           "cash_before_exit": cash_before_exit,
                                           "cash_after_exit": self.state.available_cash,
                                           "proceeds_available_date": next_day,
                                           "exit_date": day})

    def phase4_record_equity(self, day: str) -> None:
        pending_total = sum(item.proceeds for items in self.state.pending_proceeds_by_available_date.values()
                             for item in items)
        book = self.state.available_cash + sum(p.entry_cost for p in self.state.open_positions) + pending_total
        mtm = self.state.available_cash + pending_total
        for position in self.state.open_positions:
            close = self.read_engine_price(position.ticker, day, "Close", day)
            mtm += close * position.quantity
        self.state.daily_equity.append({"date": day, "book_equity": book,
                                        "mtm_equity": mtm, "available_cash": self.state.available_cash,
                                        "pending_proceeds": pending_total,
                                        "open_position_count": len(self.state.open_positions)})

    def phase5_queue_signals(self, day: str) -> None:
        before = self._phase5_snapshot()
        next_day = self._next_day(day) if any(row["signal_date"] == day for row in self.candidates) else None
        for row in sorted((candidate for candidate in self.candidates if candidate["signal_date"] == day),
                          key=lambda candidate: (int(candidate["rank"]), str(candidate["ticker"]))):
            order = PendingOrder(order_id=_order_id(row), signal_year=int(row["signal_year"]),
                                 signal_date=str(row["signal_date"]), order_created_date=day,
                                 entry_attempt_date=str(row["entry_attempt_date"]),
                                 planned_exit_date=str(row["planned_exit_date"]), ticker=str(row["ticker"]),
                                 industry=str(row["industry"]), rank=int(row["rank"]),
                                 signal_raw_close=float(row["signal_raw_close"]),
                                 candidate_status=str(row["candidate_status"]))
            if order.entry_attempt_date != next_day:
                raise ValueError("ENTRY_QUEUE_DATE_MISMATCH")
            self.state.pending_orders_by_entry_date.setdefault(next_day, []).append(order)
            self.state.event_audit.append({"event": "ORDER_QUEUED", "date": day,
                                           "order_id": order.order_id, "entry_attempt_date": next_day,
                                           "ledger": self._ledger(order) | {"order_id": order.order_id}})
        after = self._phase5_snapshot()
        if before != after:
            self._safety["d0_state_mutation_violation_count"] += 1
            raise AssertionError("D0_PHASE5_STATE_MUTATION")

    def _phase5_snapshot(self) -> tuple[Any, ...]:
        return (copy.deepcopy(self.state.available_cash), copy.deepcopy(self.state.open_positions),
                copy.deepcopy(self.state.pending_proceeds_by_available_date),
                copy.deepcopy(self.state.completed_trades), copy.deepcopy(self.state.daily_equity))

    def process_day(self, day: str) -> None:
        self.state.engine_day = day
        self.phase1_release_proceeds(day)
        self.phase2_attempt_entries(day)
        self.phase3_execute_exits(day)
        self.phase4_record_equity(day)
        self.phase5_queue_signals(day)

    def run(self) -> "CausalEventEngine":
        for day in self.calendar:
            self.process_day(day)
        validate_event_invariants(self)
        return self

    def safety_counters(self) -> dict[str, int]:
        counters = dict(self._safety)
        ids = [row["order_id"] for row in self.state.completed_trades]
        counters["duplicate_order_count"] = len(ids) - len(set(ids))
        counters["negative_cash_count"] = sum(
            1 for row in self.state.daily_equity if float(row["available_cash"]) < 0
        ) + sum(1 for event in self.state.event_audit
                if event["event"] == "CASH_DEDUCTED" and float(event["cash_after"]) < 0)
        reuse_count = 0
        for event in self.state.event_audit:
            if event["event"] == "EXIT_EXECUTED":
                if float(event["cash_after_exit"]) > float(event["cash_before_exit"]):
                    reuse_count += 1
                if event["proceeds_available_date"] <= event["exit_date"]:
                    reuse_count += 1
        indexed_events = list(enumerate(self.state.event_audit))
        for index, event in indexed_events:
            if event["event"] == "ENTRY_FILLED":
                if any(other["event"] == "EXIT_EXECUTED" and other["date"] == event["date"]
                       and other_index < index for other_index, other in indexed_events):
                    reuse_count += 1
        for event in self.state.event_audit:
            if event["event"] == "PROCEEDS_RELEASED" and _parse_iso_date(event["date"]) <= _parse_iso_date(event["exit_date"]):
                reuse_count += 1
        counters["same_day_proceeds_reuse_count"] = reuse_count
        counters["max_position_violation_count"] = sum(
            1 for event in self.state.event_audit
            if event["event"] == "ENTRY_FILLED" and event.get("position_count_after", 0) > 2)
        counters["cash_reserve_violation_count"] = sum(
            1 for row in self.state.completed_trades
            if row["status"] in {"FILLED", "CLOSED"} and float(row["cash_after_entry"]) < 40000)
        counters["signal_2026_count"] = sum(1 for row in self.candidates if int(row["signal_year"]) == 2026)
        filled = [row for row in self.state.completed_trades if row["status"] in {"FILLED", "CLOSED"}]
        counters["industry_overlap_violation_count"] = sum(
            1 for left in filled for right in filled
            if left["order_id"] < right["order_id"] and left["industry"] == right["industry"]
            and _parse_iso_date(left["entry_state_transition_date"]) <= _parse_iso_date(right["exit_execution_date"] or right["planned_exit_date"])
            and _parse_iso_date(right["entry_state_transition_date"]) <= _parse_iso_date(left["exit_execution_date"] or left["planned_exit_date"]))
        counters["future_price_access_violation_count"] = sum(
            1 for event in self.state.event_audit if event["event"] == "FUTURE_PRICE_ACCESS_PROHIBITED")
        return counters


def validate_event_invariants(engine: CausalEventEngine) -> None:
    for row in engine.state.completed_trades:
        if row["order_created_date"] != row["signal_date"]:
            raise ValueError("INVARIANT_ORDER_CREATED_DATE")
        if row["status"] in {"FILLED", "CLOSED"}:
            if row["entry_state_transition_date"] != row["entry_attempt_date"]:
                raise ValueError("INVARIANT_ENTRY_STATE_DATE")
            if row["entry_price_source_date"] != row["entry_attempt_date"]:
                raise ValueError("INVARIANT_ENTRY_PRICE_DATE")
        if row["status"] == "CLOSED":
            if row["exit_execution_date"] != row["planned_exit_date"]:
                raise ValueError("INVARIANT_EXIT_EXECUTION_DATE")
            if row["exit_price_source_date"] != row["planned_exit_date"]:
                raise ValueError("INVARIANT_EXIT_PRICE_DATE")
            if not row["proceeds_available_date"] > row["exit_execution_date"]:
                raise ValueError("INVARIANT_PROCEEDS_DATE")
    for event in engine.state.event_audit:
        if event["event"] in {"ENTRY_FILLED", "CASH_DEDUCTED"}:
            order = next(row for row in engine.state.completed_trades if row["order_id"] == event["order_id"])
            if event["date"] == order["signal_date"]:
                raise ValueError("D0_FILL_OR_CASH_DEDUCTION")


def fold_max_drawdown(equity_by_fold: Mapping[str, Sequence[float]]) -> float:
    """Return the maximum percentage drawdown across independent folds."""
    maxima = []
    for values in equity_by_fold.values():
        peak = None
        worst = 0.0
        for value in values:
            value = float(value)
            peak = value if peak is None else max(peak, value)
            if peak:
                worst = max(worst, (peak - value) / peak * 100.0)
        maxima.append(worst)
    return max(maxima, default=0.0)


def concentration_metrics(trades: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    positive = [row for row in trades if row.get("status") == "CLOSED" and float(row.get("realized_net_profit_yen") or 0) > 0]
    total = sum(float(row["realized_net_profit_yen"]) for row in positive)
    top5 = sum(sorted((float(row["realized_net_profit_yen"]) for row in positive), reverse=True)[:5])
    industry_totals: dict[str, float] = {}
    for row in positive:
        industry_totals[str(row["industry"])] = industry_totals.get(str(row["industry"]), 0.0) + float(row["realized_net_profit_yen"])
    return {"top5_positive_profit_share": top5 / total if total else 0.0,
            "max_industry_positive_profit_share": max(industry_totals.values(), default=0.0) / total if total else 0.0}


def _dates(count: int = 21) -> list[str]:
    from datetime import date, timedelta
    start = date(2020, 1, 1)
    return [(start + timedelta(days=i)).isoformat() for i in range(count)]


def synthetic_fixture() -> tuple[list[str], dict[str, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    calendar = _dates()
    frames: dict[str, dict[str, dict[str, float]]] = {}
    specs = {"AAA": (100.0, 100.0, 110.0), "BBB": (100.0, 101.0, 102.0),
             "GAP": (100.0, 103.0, 102.0), "CCC": (50.0, 51.0, 55.0)}
    for ticker, (close0, open1, open10) in specs.items():
        frames[ticker] = {}
        for day in calendar:
            frames[ticker][day] = {"Open": open1, "Close": close0}
        frames[ticker][calendar[0]]["Close"] = close0
        frames[ticker][calendar[1]]["Open"] = open1
        frames[ticker][calendar[10]]["Open"] = open10
    candidates = [
        {"signal_year": 2020, "signal_date": calendar[0], "ticker": "AAA", "industry": "TECH",
         "rank": 1, "signal_raw_close": 100.0, "entry_attempt_date": calendar[1],
         "planned_exit_date": calendar[10], "candidate_status": "ACCEPTED_TOP20"},
        {"signal_year": 2020, "signal_date": calendar[0], "ticker": "BBB", "industry": "FINANCE",
         "rank": 2, "signal_raw_close": 100.0, "entry_attempt_date": calendar[1],
         "planned_exit_date": calendar[10], "candidate_status": "ACCEPTED_TOP20"},
        {"signal_year": 2020, "signal_date": calendar[9], "ticker": "CCC", "industry": "TECH",
         "rank": 1, "signal_raw_close": 50.0, "entry_attempt_date": calendar[10],
         "planned_exit_date": calendar[19], "candidate_status": "ACCEPTED_TOP20"},
    ]
    return calendar, frames, candidates


def synthetic_scenario_b() -> CausalEventEngine:
    calendar, frames, _ = synthetic_fixture()
    rows = [
        {"signal_year": 2020, "signal_date": calendar[0], "ticker": "GAP", "industry": "FINANCE",
         "rank": 1, "signal_raw_close": 100.0, "entry_attempt_date": calendar[1],
         "planned_exit_date": calendar[10], "candidate_status": "ACCEPTED_TOP20"},
        {"signal_year": 2020, "signal_date": calendar[0], "ticker": "BBB", "industry": "ENERGY",
         "rank": 2, "signal_raw_close": 100.0, "entry_attempt_date": calendar[1],
         "planned_exit_date": calendar[10], "candidate_status": "ACCEPTED_TOP20"},
    ]
    return CausalEventEngine(frames, calendar, rows).run()


@dataclass(frozen=True)
class SyntheticGoldenResult:
    engine: CausalEventEngine
    scenario_b: CausalEventEngine
    future_read_error: str
    artifacts: dict[str, bytes]


def _artifact_bytes(engine: CausalEventEngine, candidates: Sequence[Mapping[str, Any]]) -> dict[str, bytes]:
    trades = []
    for row in engine.state.completed_trades:
        trades.append({field_name: row.get(field_name) for field_name in LEDGER_FIELDS})
    candidate_fields = ["signal_year", "signal_date", "ticker", "industry", "rank", "signal_raw_close",
                        "entry_attempt_date", "planned_exit_date", "candidate_status"]
    candidate_rows = [{name: row[name] for name in candidate_fields} for row in candidates]
    closed = [row for row in trades if row["status"] == "CLOSED"]
    profits = sum(float(row["realized_net_profit_yen"] or 0.0) for row in closed)
    summary = {"schema_version": "v6-a-r2-synthetic-1", "engine": "V6-A-R2", "synthetic_only": True,
               "formal_result": "NOT_RUN", "future_read_guard": True,
               "daily_phase_order": ["phase1_release_proceeds", "phase2_attempt_entries", "phase3_execute_exits",
                                      "phase4_record_equity", "phase5_queue_signals"],
               "aggregate_metrics": {"closed_trade_count": len(closed), "net_profit_yen": profits},
               "safety_counters": engine.safety_counters(),
               "concentration_metrics": concentration_metrics(trades),
               "fold_drawdown_max_percent": fold_max_drawdown({"2020": [row["mtm_equity"] for row in engine.state.daily_equity]}),
               "event_invariants": "validated", "two_pass_byte_identical": True}
    def csv_bytes(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> bytes:
        import io
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{name: "" if row.get(name) is None else row.get(name) for name in fieldnames} for row in rows])
        return stream.getvalue().encode("utf-8")
    return {
        "summary.json": (json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"),
        "trades.csv": csv_bytes(LEDGER_FIELDS, trades),
        "candidates.csv": csv_bytes(candidate_fields, candidate_rows),
        "daily_equity.csv": csv_bytes(["date", "pending_proceeds", "book_equity", "mtm_equity", "available_cash", "open_position_count"], engine.state.daily_equity),
    }


def write_synthetic_artifacts(output_dir: str | Path, result: SyntheticGoldenResult) -> dict[str, bytes]:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name, payload in result.artifacts.items():
        (path / name).write_bytes(payload)
    return dict(result.artifacts)


def run_synthetic_golden() -> SyntheticGoldenResult:
    calendar, frames, candidates = synthetic_fixture()
    engine = CausalEventEngine(frames, calendar, candidates).run()
    scenario_b = synthetic_scenario_b()
    guard_engine = CausalEventEngine(frames, calendar, [])
    try:
        guard_engine.read_engine_price("AAA", calendar[1], "Open", calendar[0])
    except ValueError as error:
        future_error = str(error)
    else:
        raise AssertionError("future-read negative fixture did not fail")
    first = _artifact_bytes(engine, candidates)
    second_engine = CausalEventEngine(frames, calendar, candidates).run()
    second = _artifact_bytes(second_engine, candidates)
    if first != second:
        raise AssertionError("SYNTHETIC_TWO_PASS_BYTE_MISMATCH")
    return SyntheticGoldenResult(engine, scenario_b, future_error, first)
