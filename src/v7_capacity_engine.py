"""Offline-only V7 capacity engine.

This module has no network, collector, cache, activation, or evaluation path.
It is a parameterized copy of the accepted V6-A-R2 causal event semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Sequence


PARAMETER_FIELDS = (
    "starting_cash",
    "quantity",
    "max_open_positions",
    "cash_reserve",
    "capital_limit_per_position",
    "entry_gap_multiplier",
    "entry_slippage",
    "exit_slippage",
    "same_industry_concurrent",
    "duplicate_ticker_concurrent",
    "same_day_proceeds_reuse",
)

SAFETY_COUNTER_NAMES = (
    "future_price_access",
    "negative_cash",
    "same_day_proceeds_reuse",
    "duplicate_order",
    "duplicate_ticker_open",
    "same_industry_overlap",
    "max_position_violation",
    "cash_reserve_violation",
    "capital_limit_violation",
    "D0_state_mutation",
    "historical_backfill",
    "snapshot_rewrite",
    "cross_arm_state_leakage",
    "future_candidate_data_access",
    "future_split_access",
    "open_position_split_spanning",
    "planned_exit_price_unavailable",
    "open_position_mtm_price_unavailable",
    "candidate_snapshot_rerank",
    "outside_top20_replacement",
)

DERIVED_SAFETY_COUNTER_NAMES = (
    "future_price_access",
    "negative_cash",
    "same_day_proceeds_reuse",
    "duplicate_order",
    "duplicate_ticker_open",
    "same_industry_overlap",
    "max_position_violation",
    "cash_reserve_violation",
    "capital_limit_violation",
)

STICKY_SAFETY_COUNTER_NAMES = (
    "historical_backfill",
    "snapshot_rewrite",
    "cross_arm_state_leakage",
    "D0_state_mutation",
    "future_candidate_data_access",
    "future_split_access",
    "open_position_split_spanning",
    "planned_exit_price_unavailable",
    "open_position_mtm_price_unavailable",
    "candidate_snapshot_rerank",
    "outside_top20_replacement",
)

SKIP_REASONS = (
    "MAX_OPEN_POSITIONS",
    "DUPLICATE_TICKER_OPEN",
    "SAME_INDUSTRY_OPEN",
    "CASH_RESERVE",
    "CAPITAL_LIMIT",
    "ENTRY_GAP_TOO_HIGH",
    "ENTRY_DATA_UNAVAILABLE",
    "SPLIT_EFFECTIVE_BEFORE_ENTRY",
)


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


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class V7EngineParameters:
    starting_cash: float = 400000
    quantity: int = 100
    max_open_positions: int = 2
    cash_reserve: float = 40000
    capital_limit_per_position: float = 220000
    entry_gap_multiplier: float = 1.02
    entry_slippage: float = 0.0003
    exit_slippage: float = 0.0003
    same_industry_concurrent: bool = False
    duplicate_ticker_concurrent: bool = False
    same_day_proceeds_reuse: bool = False

    @classmethod
    def control(cls) -> "V7EngineParameters":
        return cls(max_open_positions=2)

    @classmethod
    def capacity_3(cls) -> "V7EngineParameters":
        return cls(max_open_positions=3)

    def __post_init__(self) -> None:
        if self.max_open_positions not in (2, 3):
            raise ValueError("MAX_OPEN_POSITIONS_MUST_BE_2_OR_3")
        if self.quantity <= 0 or self.starting_cash <= 0:
            raise ValueError("INVALID_POSITIVE_PARAMETER")
        if self.cash_reserve < 0 or self.capital_limit_per_position <= 0:
            raise ValueError("INVALID_CASH_PARAMETER")
        if self.entry_gap_multiplier <= 0 or self.entry_slippage < 0 or self.exit_slippage < 0:
            raise ValueError("INVALID_EXECUTION_PARAMETER")
        if self.same_industry_concurrent or self.duplicate_ticker_concurrent or self.same_day_proceeds_reuse:
            raise ValueError("V7_FIXED_BOOLEAN_RULE_MUST_BE_FALSE")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in PARAMETER_FIELDS}

    def canonical_json(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()


def canonical_parameter_json(parameters: V7EngineParameters) -> bytes:
    return parameters.canonical_json()


def parameters_sha256(parameters: V7EngineParameters) -> str:
    return parameters.sha256()


def validate_single_parameter_difference(
    control: V7EngineParameters, variant: V7EngineParameters
) -> bool:
    if not isinstance(control, V7EngineParameters) or not isinstance(variant, V7EngineParameters):
        raise ValueError("PARAMETER_CONTRACT_INVALID")
    if control.max_open_positions != 2 or variant.max_open_positions != 3:
        raise ValueError("SINGLE_PARAMETER_DIFFERENCE_MAX_POSITION_VALUES_INVALID")
    control_values = control.to_dict()
    variant_values = variant.to_dict()
    differences = {
        key: (control_values[key], variant_values[key])
        for key in PARAMETER_FIELDS
        if control_values[key] != variant_values[key]
    }
    if differences != {"max_open_positions": (2, 3)}:
        raise ValueError("SINGLE_PARAMETER_DIFFERENCE_REQUIRED")
    return True


def read_price(
    frames: Mapping[str, Mapping[str, Mapping[str, float]]],
    ticker: str,
    requested_date: str,
    field: str,
    engine_day: str,
) -> float:
    """Read one finite value and fail closed on any future date."""
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
    if field in {"Open", "Close", "High", "Low"} and float(value) <= 0:
        raise ValueError("NONPOSITIVE_PRICE")
    return float(value)


class V7StudyBlocked(RuntimeError):
    """Fail-closed forward-study boundary violation."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


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
    required = {
        "signal_year", "signal_date", "ticker", "industry", "rank",
        "signal_raw_close", "entry_attempt_date", "planned_exit_date",
        "candidate_status",
    }
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"CANDIDATE_SCHEMA_MISSING:{','.join(missing)}")
    if row["candidate_status"] != "ACCEPTED_TOP20":
        raise ValueError("CANDIDATE_STATUS_NOT_ACCEPTED_TOP20")
    if not isinstance(row["rank"], int) or isinstance(row["rank"], bool):
        raise ValueError("INVALID_CANDIDATE_RANK")
    if not 1 <= row["rank"] <= 20:
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
    if int(row["signal_year"]) != int(signal[:4]):
        raise ValueError("SIGNAL_YEAR_MISMATCH")
    if not isinstance(row["signal_raw_close"], (int, float)) or not math.isfinite(float(row["signal_raw_close"])):
        raise ValueError("NONFINITE_CANDIDATE_CLOSE")


def validate_candidate_schema(
    calendar: Sequence[str], candidates: Sequence[Mapping[str, Any]]
) -> None:
    parsed = tuple(_parse_iso_date(day) for day in calendar)
    if tuple(sorted(parsed)) != parsed or len(set(calendar)) != len(calendar):
        raise ValueError("INVALID_COMMON_CALENDAR")
    ids: set[str] = set()
    ticker_keys: set[tuple[str, str]] = set()
    rank_keys: set[tuple[str, int]] = set()
    for row in candidates:
        _require_candidate(row, calendar)
        order_id = _order_id(row)
        ticker_key = (str(row["signal_date"]), str(row["ticker"]))
        rank_key = (str(row["signal_date"]), int(row["rank"]))
        if order_id in ids or ticker_key in ticker_keys or rank_key in rank_keys:
            raise ValueError("DUPLICATE_CANDIDATE_KEY")
        ids.add(order_id)
        ticker_keys.add(ticker_key)
        rank_keys.add(rank_key)


class CausalEventEngine:
    """V7 causal engine with only max-open-positions parameterized."""

    def __init__(
        self,
        frames: Mapping[str, Mapping[str, Mapping[str, float]]],
        calendar: Sequence[str],
        candidates: Sequence[Mapping[str, Any]],
        parameters: V7EngineParameters | None = None,
        split_events_by_day: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.frames = copy.deepcopy(frames)
        self.calendar = tuple(calendar)
        self._calendar_dates = tuple(_parse_iso_date(day) for day in self.calendar)
        if len(self.calendar) != len(set(self.calendar)):
            raise ValueError("DUPLICATE_COMMON_CALENDAR_DATE")
        if tuple(sorted(self._calendar_dates)) != self._calendar_dates:
            raise ValueError("COMMON_CALENDAR_NOT_SORTED")
        self._calendar_index = {day: index for index, day in enumerate(self.calendar)}
        self.parameters = parameters or V7EngineParameters.control()
        self.split_events_by_day = split_events_by_day or {}
        self.state = EngineState(available_cash=float(self.parameters.starting_cash))
        self.candidates = [dict(row) for row in candidates]
        validate_candidate_schema(self.calendar, self.candidates)
        self._candidate_ids = {_order_id(row) for row in self.candidates}
        self._candidate_keys = {(str(row["signal_date"]), str(row["ticker"])) for row in self.candidates}
        self._candidate_ranks = {(str(row["signal_date"]), int(row["rank"])) for row in self.candidates}
        self._safety = {name: 0 for name in SAFETY_COUNTER_NAMES}
        self._skip_reason_counts = {name: 0 for name in SKIP_REASONS}

    def _split_tickers_for_day(self, day: str) -> set[str]:
        return {
            str(ticker).strip().upper()
            for ticker in self.split_events_by_day.get(day, ())
        }

    def record_safety_violation(self, name: str, count: int = 1) -> None:
        """Record an immutable-study safety violation through the narrow API."""
        if name not in STICKY_SAFETY_COUNTER_NAMES:
            raise ValueError("STICKY_SAFETY_COUNTER_NAME_INVALID")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("SAFETY_VIOLATION_COUNT_INVALID")
        self._safety[name] += count

    def _ledger(self, order: PendingOrder) -> dict[str, Any]:
        return {
            "signal_year": order.signal_year,
            "signal_date": order.signal_date,
            "order_created_date": order.order_created_date,
            "entry_attempt_date": order.entry_attempt_date,
            "entry_state_transition_date": None,
            "entry_price_source_date": None,
            "planned_exit_date": order.planned_exit_date,
            "exit_execution_date": None,
            "exit_price_source_date": None,
            "proceeds_available_date": None,
            "ticker": order.ticker,
            "industry": order.industry,
            "rank": order.rank,
            "status": "QUEUED",
            "skip_reason": None,
            "quantity": self.parameters.quantity,
            "entry_price": None,
            "exit_price": None,
            "entry_cost": None,
            "exit_proceeds": None,
            "realized_net_profit_yen": None,
            "realized_net_return_percent": None,
            "cash_before_entry": None,
            "cash_after_entry": None,
            "position_count_before_entry": None,
            "position_count_after_entry": None,
            "order_id": order.order_id,
        }

    def _ledger_for(self, order_id: str) -> dict[str, Any]:
        for row in self.state.completed_trades:
            if row["order_id"] == order_id:
                return row
        raise ValueError("LEDGER_ORDER_NOT_FOUND")

    def _ensure_ledger(self, order: PendingOrder) -> dict[str, Any]:
        for row in self.state.completed_trades:
            if row["order_id"] == order.order_id:
                return row
        row = self._ledger(order)
        self.state.completed_trades.append(row)
        return row

    def read_engine_price(self, ticker: str, requested_date: str, field: str, engine_day: str) -> float:
        try:
            return read_price(self.frames, ticker, requested_date, field, engine_day)
        except ValueError as error:
            if str(error) == "FUTURE_PRICE_ACCESS_PROHIBITED":
                self._safety["future_price_access"] += 1
                self.state.event_audit.append({
                    "event": "FUTURE_PRICE_ACCESS_PROHIBITED",
                    "date": engine_day,
                    "requested_date": requested_date,
                    "ticker": ticker,
                    "field": field,
                })
            raise

    def _next_day(self, day: str) -> str:
        index = self._calendar_index[day]
        if index + 1 >= len(self.calendar):
            raise ValueError("NEXT_PROCEEDS_DATE_UNAVAILABLE")
        return self.calendar[index + 1]

    def phase1_release_proceeds(self, day: str) -> None:
        self.state.engine_day = day
        proceeds = self.state.pending_proceeds_by_available_date.pop(day, [])
        for item in proceeds:
            self.state.available_cash += item.proceeds
            self.state.event_audit.append({
                "event": "PROCEEDS_RELEASED",
                "date": day,
                "order_id": item.order_id,
                "amount": item.proceeds,
                "exit_date": item.exit_date,
            })
        if self.state.available_cash < 0:
            self._safety["negative_cash"] += 1

    def phase2_attempt_entries(self, day: str) -> None:
        orders = list(self.state.pending_orders_by_entry_date.pop(day, []))
        orders.sort(key=lambda order: (order.rank, order.ticker))
        split_tickers = self._split_tickers_for_day(day)
        for order in orders:
            ledger = self._ensure_ledger(order)
            cash_before = self.state.available_cash
            position_count_before = len(self.state.open_positions)
            if order.ticker.strip().upper() in split_tickers:
                self._skip_reason_counts["SPLIT_EFFECTIVE_BEFORE_ENTRY"] += 1
                ledger.update({
                    "status": "SKIPPED",
                    "skip_reason": "SPLIT_EFFECTIVE_BEFORE_ENTRY",
                    "entry_state_transition_date": day,
                    "cash_before_entry": cash_before,
                    "cash_after_entry": self.state.available_cash,
                    "position_count_before_entry": position_count_before,
                    "position_count_after_entry": len(self.state.open_positions),
                })
                self.state.event_audit.append({
                    "event": "ENTRY_SKIPPED",
                    "date": day,
                    "order_id": order.order_id,
                    "reason": "SPLIT_EFFECTIVE_BEFORE_ENTRY",
                })
                continue
            signal_close = self.read_engine_price(order.ticker, order.signal_date, "Close", day)
            if not math.isclose(signal_close, order.signal_raw_close, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("CANDIDATE_SIGNAL_CLOSE_MISMATCH")
            try:
                raw_open = self.read_engine_price(order.ticker, day, "Open", day)
            except ValueError as error:
                if str(error) not in {
                    "DATE_NOT_FOUND",
                    "FIELD_NOT_FOUND",
                    "NONFINITE_PRICE",
                    "NONPOSITIVE_PRICE",
                }:
                    raise
                self._skip_reason_counts["ENTRY_DATA_UNAVAILABLE"] += 1
                ledger.update({
                    "status": "SKIPPED",
                    "skip_reason": "ENTRY_DATA_UNAVAILABLE",
                    "entry_state_transition_date": day,
                    "cash_before_entry": cash_before,
                    "cash_after_entry": self.state.available_cash,
                    "position_count_before_entry": position_count_before,
                    "position_count_after_entry": len(self.state.open_positions),
                })
                self.state.event_audit.append({
                    "event": "ENTRY_SKIPPED",
                    "date": day,
                    "order_id": order.order_id,
                    "reason": "ENTRY_DATA_UNAVAILABLE",
                })
                continue
            tickers = {position.ticker for position in self.state.open_positions}
            industries = {position.industry for position in self.state.open_positions}
            reason: str | None = None
            if not self.parameters.duplicate_ticker_concurrent and order.ticker in tickers:
                reason = "DUPLICATE_TICKER_OPEN"
            elif not self.parameters.same_industry_concurrent and order.industry in industries:
                reason = "SAME_INDUSTRY_OPEN"
            elif len(self.state.open_positions) >= self.parameters.max_open_positions:
                reason = "MAX_OPEN_POSITIONS"
            elif raw_open > signal_close * self.parameters.entry_gap_multiplier:
                reason = "ENTRY_GAP_TOO_HIGH"
            else:
                entry_price = raw_open * (1.0 + self.parameters.entry_slippage)
                entry_cost = entry_price * self.parameters.quantity
                if entry_cost > self.parameters.capital_limit_per_position:
                    reason = "CAPITAL_LIMIT"
                elif (
                    self.state.available_cash - entry_cost
                    < self.parameters.cash_reserve
                ):
                    reason = "CASH_RESERVE"
            if reason is not None:
                self._skip_reason_counts[reason] += 1
                ledger.update({
                    "status": "SKIPPED",
                    "skip_reason": reason,
                    "entry_state_transition_date": day,
                    "cash_before_entry": cash_before,
                    "cash_after_entry": self.state.available_cash,
                    "position_count_before_entry": position_count_before,
                    "position_count_after_entry": len(self.state.open_positions),
                })
                self.state.event_audit.append({
                    "event": "ENTRY_SKIPPED",
                    "date": day,
                    "order_id": order.order_id,
                    "reason": reason,
                })
                continue
            entry_price = raw_open * (1.0 + self.parameters.entry_slippage)
            entry_cost = entry_price * self.parameters.quantity
            self.state.available_cash -= entry_cost
            position = OpenPosition(
                order_id=order.order_id,
                signal_year=order.signal_year,
                signal_date=order.signal_date,
                ticker=order.ticker,
                industry=order.industry,
                rank=order.rank,
                entry_date=day,
                planned_exit_date=order.planned_exit_date,
                quantity=self.parameters.quantity,
                entry_price=entry_price,
                entry_cost=entry_cost,
            )
            self.state.open_positions.append(position)
            ledger.update({
                "status": "FILLED",
                "entry_state_transition_date": day,
                "entry_price_source_date": day,
                "entry_price": entry_price,
                "entry_cost": entry_cost,
                "cash_before_entry": cash_before,
                "cash_after_entry": self.state.available_cash,
                "position_count_before_entry": position_count_before,
                "position_count_after_entry": len(self.state.open_positions),
            })
            self.state.event_audit.extend([
                {
                    "event": "CASH_DEDUCTED",
                    "date": day,
                    "order_id": order.order_id,
                    "amount": entry_cost,
                    "cash_after": self.state.available_cash,
                },
                {
                    "event": "ENTRY_FILLED",
                    "date": day,
                    "order_id": order.order_id,
                    "position_count_after": len(self.state.open_positions),
                },
            ])
            if self.state.available_cash < 0:
                self._safety["negative_cash"] += 1

    def phase2b_check_open_position_splits(self, day: str) -> None:
        split_tickers = self._split_tickers_for_day(day)
        for position in list(self.state.open_positions):
            if position.ticker.strip().upper() not in split_tickers:
                continue
            self.record_safety_violation("open_position_split_spanning")
            self.state.event_audit.append({
                "event": "OPEN_POSITION_SPLIT_DETECTED",
                "date": day,
                "order_id": position.order_id,
                "ticker": position.ticker,
                "planned_exit_date": position.planned_exit_date,
            })
            raise V7StudyBlocked("OPEN_POSITION_SPLIT_SPANNING")

    def phase3_execute_exits(self, day: str) -> None:
        exits = [
            position for position in self.state.open_positions
            if position.planned_exit_date == day
        ]
        for position in exits:
            next_day = self._next_day(day)
            cash_before_exit = self.state.available_cash
            try:
                raw_open = self.read_engine_price(position.ticker, day, "Open", day)
            except ValueError as error:
                if str(error) not in {
                    "DATE_NOT_FOUND",
                    "FIELD_NOT_FOUND",
                    "NONFINITE_PRICE",
                    "NONPOSITIVE_PRICE",
                }:
                    raise
                self.record_safety_violation("planned_exit_price_unavailable")
                self.state.event_audit.append({
                    "event": "D10_EXIT_BLOCKED_MISSING_PRICE",
                    "date": day,
                    "order_id": position.order_id,
                    "ticker": position.ticker,
                })
                raise V7StudyBlocked("PLANNED_EXIT_PRICE_UNAVAILABLE")
            exit_price = raw_open * (1.0 - self.parameters.exit_slippage)
            proceeds = exit_price * position.quantity
            self.state.open_positions.remove(position)
            self.state.pending_proceeds_by_available_date.setdefault(next_day, []).append(
                PendingProceeds(position.order_id, day, next_day, proceeds)
            )
            ledger = self._ledger_for(position.order_id)
            profit = proceeds - float(ledger["entry_cost"])
            ledger.update({
                "status": "CLOSED",
                "exit_execution_date": day,
                "exit_price_source_date": day,
                "exit_price": exit_price,
                "exit_proceeds": proceeds,
                "realized_net_profit_yen": profit,
                "realized_net_return_percent": profit / float(ledger["entry_cost"]) * 100.0,
                "proceeds_available_date": next_day,
            })
            self.state.event_audit.append({
                "event": "EXIT_EXECUTED",
                "date": day,
                "order_id": position.order_id,
                "exit_proceeds": proceeds,
                "cash_before_exit": cash_before_exit,
                "cash_after_exit": self.state.available_cash,
                "proceeds_available_date": next_day,
                "exit_date": day,
            })

    def phase4_record_equity(self, day: str) -> None:
        pending_total = sum(
            item.proceeds
            for items in self.state.pending_proceeds_by_available_date.values()
            for item in items
        )
        book = (
            self.state.available_cash
            + sum(position.entry_cost for position in self.state.open_positions)
            + pending_total
        )
        mtm = self.state.available_cash + pending_total
        for position in self.state.open_positions:
            try:
                close = self.read_engine_price(position.ticker, day, "Close", day)
            except ValueError as error:
                if str(error) not in {
                    "DATE_NOT_FOUND",
                    "FIELD_NOT_FOUND",
                    "NONFINITE_PRICE",
                    "NONPOSITIVE_PRICE",
                }:
                    raise
                self.record_safety_violation("open_position_mtm_price_unavailable")
                self.state.event_audit.append({
                    "event": "MTM_BLOCKED_MISSING_PRICE",
                    "date": day,
                    "ticker": position.ticker,
                })
                raise V7StudyBlocked("OPEN_POSITION_MTM_PRICE_UNAVAILABLE")
            mtm += close * position.quantity
        self.state.daily_equity.append({
            "date": day,
            "book_equity": book,
            "mtm_equity": mtm,
            "available_cash": self.state.available_cash,
            "pending_proceeds": pending_total,
            "open_position_count": len(self.state.open_positions),
        })

    def _phase5_snapshot(self) -> tuple[Any, ...]:
        return (
            copy.deepcopy(self.state.available_cash),
            copy.deepcopy(self.state.open_positions),
            copy.deepcopy(self.state.pending_proceeds_by_available_date),
            copy.deepcopy(self.state.completed_trades),
            copy.deepcopy(self.state.daily_equity),
        )

    def phase5_queue_signals(self, day: str) -> None:
        before = self._phase5_snapshot()
        candidates = [
            row for row in self.candidates if str(row["signal_date"]) == day
        ]
        next_day = self._next_day(day) if candidates else None
        for row in sorted(candidates, key=lambda candidate: (int(candidate["rank"]), str(candidate["ticker"]))):
            order = PendingOrder(
                order_id=_order_id(row),
                signal_year=int(row["signal_year"]),
                signal_date=str(row["signal_date"]),
                order_created_date=day,
                entry_attempt_date=str(row["entry_attempt_date"]),
                planned_exit_date=str(row["planned_exit_date"]),
                ticker=str(row["ticker"]),
                industry=str(row["industry"]),
                rank=int(row["rank"]),
                signal_raw_close=float(row["signal_raw_close"]),
                candidate_status=str(row["candidate_status"]),
            )
            if order.entry_attempt_date != next_day:
                raise ValueError("ENTRY_QUEUE_DATE_MISMATCH")
            self.state.pending_orders_by_entry_date.setdefault(next_day, []).append(order)
            self.state.event_audit.append({
                "event": "ORDER_QUEUED",
                "date": day,
                "order_id": order.order_id,
                "entry_attempt_date": next_day,
                "ledger": self._ledger(order),
            })
        after = self._phase5_snapshot()
        if before != after:
            self.record_safety_violation("D0_state_mutation")
            raise AssertionError("D0_PHASE5_STATE_MUTATION")

    def process_day(self, day: str) -> None:
        self.state.engine_day = day
        self.phase1_release_proceeds(day)
        self.phase2_attempt_entries(day)
        self.phase2b_check_open_position_splits(day)
        self.phase3_execute_exits(day)
        self.phase4_record_equity(day)
        self.phase5_queue_signals(day)

    def run(self) -> "CausalEventEngine":
        for day in self.calendar:
            self.process_day(day)
        validate_event_invariants(self)
        self._refresh_safety_counters()
        return self

    def _refresh_safety_counters(self) -> None:
        for name in DERIVED_SAFETY_COUNTER_NAMES:
            self._safety[name] = 0
        order_ids = [row["order_id"] for row in self.state.completed_trades]
        self._safety["duplicate_order"] = len(order_ids) - len(set(order_ids))
        self._safety["negative_cash"] = (
            sum(1 for row in self.state.daily_equity if float(row["available_cash"]) < 0)
            + sum(
                1 for event in self.state.event_audit
                if event["event"] == "CASH_DEDUCTED"
                and float(event["cash_after"]) < 0
            )
        )
        reuse_count = 0
        indexed_events = list(enumerate(self.state.event_audit))
        for index, event in indexed_events:
            if event["event"] == "EXIT_EXECUTED":
                if float(event["cash_after_exit"]) > float(event["cash_before_exit"]):
                    reuse_count += 1
                if event["proceeds_available_date"] <= event["exit_date"]:
                    reuse_count += 1
            if event["event"] == "ENTRY_FILLED" and any(
                other["event"] == "EXIT_EXECUTED"
                and other["date"] == event["date"]
                and other_index < index
                for other_index, other in indexed_events
            ):
                reuse_count += 1
            if event["event"] == "PROCEEDS_RELEASED" and _parse_iso_date(event["date"]) <= _parse_iso_date(event["exit_date"]):
                reuse_count += 1
        self._safety["same_day_proceeds_reuse"] = reuse_count

        filled = [
            row for row in self.state.completed_trades
            if row["status"] in {"FILLED", "CLOSED"}
        ]
        self._safety["max_position_violation"] = sum(
            1 for row in filled
            if int(row.get("position_count_after_entry") or 0) > self.parameters.max_open_positions
        )
        self._safety["cash_reserve_violation"] = sum(
            1 for row in filled
            if float(row.get("cash_after_entry") or 0) < self.parameters.cash_reserve
        )
        self._safety["capital_limit_violation"] = sum(
            1 for row in filled
            if float(row.get("entry_cost") or 0) > self.parameters.capital_limit_per_position
        )
        self._safety["duplicate_ticker_open"] = _overlap_violation_count(filled, "ticker")
        self._safety["same_industry_overlap"] = _overlap_violation_count(filled, "industry")
        self._safety["future_price_access"] = sum(
            1 for event in self.state.event_audit
            if event["event"] == "FUTURE_PRICE_ACCESS_PROHIBITED"
        )

    def safety_counters(self) -> dict[str, int]:
        self._refresh_safety_counters()
        return dict(self._safety)

    def skip_reason_counts(self) -> dict[str, int]:
        return dict(self._skip_reason_counts)

    def legacy_safety_counters(self) -> dict[str, int]:
        current = self.safety_counters()
        return {
            "negative_cash_count": current["negative_cash"],
            "same_day_proceeds_reuse_count": current["same_day_proceeds_reuse"],
            "duplicate_order_count": current["duplicate_order"],
            "max_position_violation_count": current["max_position_violation"],
            "cash_reserve_violation_count": current["cash_reserve_violation"],
            "industry_overlap_violation_count": current["same_industry_overlap"],
            "signal_2026_count": sum(
                1 for row in self.candidates if int(row["signal_year"]) == 2026
            ),
            "future_price_access_violation_count": current["future_price_access"],
            "d0_state_mutation_violation_count": current["D0_state_mutation"],
        }

    def state_snapshot(self) -> dict[str, Any]:
        return {
            "available_cash": self.state.available_cash,
            "open_positions": [asdict(item) for item in self.state.open_positions],
            "pending_orders": {
                day: [asdict(item) for item in items]
                for day, items in sorted(self.state.pending_orders_by_entry_date.items())
            },
            "pending_proceeds": {
                day: [asdict(item) for item in items]
                for day, items in sorted(self.state.pending_proceeds_by_available_date.items())
            },
            "completed_trades": copy.deepcopy(self.state.completed_trades),
            "daily_equity": copy.deepcopy(self.state.daily_equity),
            "event_audit": copy.deepcopy(self.state.event_audit),
            "safety_counters": self.safety_counters(),
        }


def _overlap_violation_count(rows: Sequence[Mapping[str, Any]], field_name: str) -> int:
    violations = 0
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if left[field_name] != right[field_name]:
                continue
            left_start = _parse_iso_date(str(left["entry_state_transition_date"]))
            right_start = _parse_iso_date(str(right["entry_state_transition_date"]))
            left_end = _parse_iso_date(str(left.get("exit_execution_date") or left["planned_exit_date"]))
            right_end = _parse_iso_date(str(right.get("exit_execution_date") or right["planned_exit_date"]))
            if left_start <= right_end and right_start <= left_end:
                violations += 1
    return violations


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
