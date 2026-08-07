from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, timedelta

import pytest

from src.v6_a_r2_causal_breakout import CausalEventEngine as V6Engine
from src.v7_capacity_engine import (
    CausalEventEngine as V7Engine,
    V7EngineParameters,
    validate_single_parameter_difference,
)


def calendar(count: int = 21) -> list[str]:
    start = date(2020, 1, 2)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def frames_for(days: list[str], tickers: tuple[str, ...]) -> dict:
    return {
        ticker: {day: {"Open": 100.0, "Close": 100.0} for day in days}
        for ticker in tickers
    }


def candidate(days, ticker, industry, rank, signal_index=0, exit_index=10):
    return {
        "signal_year": 2020,
        "signal_date": days[signal_index],
        "ticker": ticker,
        "industry": industry,
        "rank": rank,
        "signal_raw_close": 100.0,
        "entry_attempt_date": days[signal_index + 1],
        "planned_exit_date": days[exit_index],
        "candidate_status": "ACCEPTED_TOP20",
    }


def normalized_state(engine):
    state = engine.state
    return {
        "available_cash": state.available_cash,
        "open_positions": [asdict(item) for item in state.open_positions],
        "pending_orders": {
            day: [asdict(item) for item in rows]
            for day, rows in sorted(state.pending_orders_by_entry_date.items())
        },
        "pending_proceeds": {
            day: [asdict(item) for item in rows]
            for day, rows in sorted(state.pending_proceeds_by_available_date.items())
        },
        "completed_trades": state.completed_trades,
        "daily_equity": state.daily_equity,
        "event_audit": state.event_audit,
    }


def parity_fixture():
    days = calendar()
    frames = frames_for(days, ("AAA", "BBB"))
    rows = [
        candidate(days, "AAA", "TECH", 1, 0, 10),
        candidate(days, "BBB", "FINANCE", 1, 9, 19),
    ]
    return days, frames, rows


def test_parameter_contract_and_canonical_hash_are_frozen():
    control = V7EngineParameters.control()
    variant = V7EngineParameters.capacity_3()
    assert control.max_open_positions == 2
    assert variant.max_open_positions == 3
    assert control.canonical_json() == V7EngineParameters.control().canonical_json()
    assert len(control.sha256()) == 64
    assert control.to_dict()["cash_reserve"] == 40000


def test_single_parameter_difference_rejects_any_other_difference():
    control = V7EngineParameters.control()
    variant = V7EngineParameters.capacity_3()
    assert validate_single_parameter_difference(control, variant)
    with pytest.raises(ValueError, match="SINGLE_PARAMETER"):
        validate_single_parameter_difference(control, replace(variant, entry_slippage=0.001))


def test_control_matches_v6_multiple_event_scenarios():
    days, frames, rows = parity_fixture()
    v6 = V6Engine(frames, days, rows).run()
    v7 = V7Engine(frames, days, rows, V7EngineParameters.control()).run()
    assert normalized_state(v7) == normalized_state(v6)
    assert v7.legacy_safety_counters() == v6.safety_counters()
    assert v7.state_snapshot()["safety_counters"]["max_position_violation"] == 0


@pytest.mark.parametrize("scenario", ["gap", "duplicate_ticker", "same_industry"])
def test_control_matches_v6_skip_scenarios(scenario):
    days = calendar(12)
    if scenario == "gap":
        frames = frames_for(days, ("GAP",))
        frames["GAP"][days[1]]["Open"] = 103.0
        rows = [candidate(days, "GAP", "TECH", 1)]
    elif scenario == "duplicate_ticker":
        frames = frames_for(days, ("AAA",))
        rows = [
            candidate(days, "AAA", "TECH", 1, 0, 10),
            candidate(days, "AAA", "FINANCE", 1, 1, 11),
        ]
    else:
        frames = frames_for(days, ("AAA", "BBB"))
        rows = [
            candidate(days, "AAA", "TECH", 1, 0, 10),
            candidate(days, "BBB", "TECH", 2, 0, 10),
        ]
    v6 = V6Engine(frames, days, rows).run()
    v7 = V7Engine(frames, days, rows, V7EngineParameters.control()).run()
    assert normalized_state(v7) == normalized_state(v6)
    assert v7.legacy_safety_counters() == v6.safety_counters()


def test_capacity_difference_is_only_max_position_and_three_fill():
    days = calendar()
    frames = frames_for(days, ("AAA", "BBB", "CCC"))
    rows = [
        candidate(days, "AAA", "TECH", 1),
        candidate(days, "BBB", "FINANCE", 2),
        candidate(days, "CCC", "ENERGY", 3),
    ]
    control = V7Engine(frames, days, rows, V7EngineParameters.control()).run()
    variant = V7Engine(frames, days, rows, V7EngineParameters.capacity_3()).run()
    assert [row["status"] for row in control.state.completed_trades] == ["CLOSED", "CLOSED", "SKIPPED"]
    assert control.skip_reason_counts()["MAX_OPEN_POSITIONS"] == 1
    assert sum(row["status"] in {"FILLED", "CLOSED"} for row in variant.state.completed_trades) == 3
    assert replace(control.parameters, max_open_positions=3) == variant.parameters
    assert control.safety_counters()["max_position_violation"] == 0
    assert variant.safety_counters()["max_position_violation"] == 0


def test_event_order_entry_before_same_day_exit_and_no_proceeds_reuse():
    days, frames, rows = parity_fixture()
    engine = V7Engine(frames, days, rows, V7EngineParameters.control())
    for day in days[:11]:
        engine.process_day(day)
    events = engine.state.event_audit
    entry_index = next(
        i for i, event in enumerate(events)
        if event["event"] == "ENTRY_FILLED" and event["date"] == days[10]
    )
    exit_index = next(
        i for i, event in enumerate(events)
        if event["event"] == "EXIT_EXECUTED" and event["date"] == days[10]
    )
    assert entry_index < exit_index
    assert engine.safety_counters()["same_day_proceeds_reuse"] == 0
    assert engine.state.pending_proceeds_by_available_date[days[11]][0].availability_date == days[11]


@pytest.mark.parametrize(
    ("reason", "ticker", "industry", "open_value", "parameters"),
    [
        ("ENTRY_GAP_TOO_HIGH", "GAP", "TECH", 103.0, V7EngineParameters.control()),
        ("CAPITAL_LIMIT", "CAP", "TECH", 3000.0, V7EngineParameters(
            entry_gap_multiplier=100.0
        )),
        ("CASH_RESERVE", "CASH", "TECH", 100.0, V7EngineParameters(
            starting_cash=45000, cash_reserve=40000
        )),
    ],
)
def test_ledger_skip_reasons_are_not_safety_violations(
    reason, ticker, industry, open_value, parameters
):
    days = calendar(12)
    frames = frames_for(days, (ticker,))
    frames[ticker][days[1]]["Open"] = open_value
    rows = [candidate(days, ticker, industry, 1)]
    engine = V7Engine(frames, days, rows, parameters).run()
    assert engine.skip_reason_counts()[reason] == 1
    counter_name = {
        "ENTRY_GAP_TOO_HIGH": "max_position_violation",
        "CAPITAL_LIMIT": "capital_limit_violation",
        "CASH_RESERVE": "cash_reserve_violation",
    }[reason]
    assert engine.safety_counters()[counter_name] == 0


def test_duplicate_ticker_and_same_industry_skip_without_violation():
    days = calendar(12)
    frames = frames_for(days, ("AAA", "BBB"))
    duplicate_rows = [
        candidate(days, "AAA", "TECH", 1, 0, 10),
        candidate(days, "AAA", "FINANCE", 1, 1, 11),
    ]
    duplicate = V7Engine(frames, days, duplicate_rows, V7EngineParameters.control()).run()
    assert duplicate.skip_reason_counts()["DUPLICATE_TICKER_OPEN"] == 1
    assert duplicate.safety_counters()["duplicate_ticker_open"] == 0

    industry_rows = [
        candidate(days, "AAA", "TECH", 1, 0, 10),
        candidate(days, "BBB", "TECH", 2, 0, 10),
    ]
    industry = V7Engine(frames, days, industry_rows, V7EngineParameters.control()).run()
    assert industry.skip_reason_counts()["SAME_INDUSTRY_OPEN"] == 1
    assert industry.safety_counters()["same_industry_overlap"] == 0


def test_future_price_read_fails_closed_and_counts_safety():
    days = calendar(12)
    engine = V7Engine(frames_for(days, ("AAA",)), days, [], V7EngineParameters.control())
    with pytest.raises(ValueError, match="FUTURE_PRICE_ACCESS"):
        engine.read_engine_price("AAA", days[1], "Open", days[0])
    assert engine.safety_counters()["future_price_access"] == 1


def test_d0_only_queues_order_and_does_not_mutate_portfolio_state():
    days = calendar(12)
    frames = frames_for(days, ("AAA",))
    engine = V7Engine(
        frames, days, [candidate(days, "AAA", "TECH", 1)], V7EngineParameters.control()
    )
    before = {
        "cash": engine.state.available_cash,
        "positions": list(engine.state.open_positions),
        "proceeds": dict(engine.state.pending_proceeds_by_available_date),
        "trades": list(engine.state.completed_trades),
        "equity": list(engine.state.daily_equity),
    }
    engine.phase5_queue_signals(days[0])
    after = {
        "cash": engine.state.available_cash,
        "positions": list(engine.state.open_positions),
        "proceeds": dict(engine.state.pending_proceeds_by_available_date),
        "trades": list(engine.state.completed_trades),
        "equity": list(engine.state.daily_equity),
    }
    assert before == after
    assert days[1] in engine.state.pending_orders_by_entry_date


def test_terminal_state_has_no_open_positions_or_pending_proceeds():
    days, frames, rows = parity_fixture()
    engine = V7Engine(frames, days, rows, V7EngineParameters.control()).run()
    assert engine.state.open_positions == []
    assert engine.state.pending_proceeds_by_available_date == {}
    assert engine.state.daily_equity[-1]["date"] == days[-1]


def test_v7_state_snapshot_contains_all_arm_state_components():
    days = calendar(12)
    engine = V7Engine(frames_for(days, ("AAA",)), days, [], V7EngineParameters.control()).run()
    snapshot = engine.state_snapshot()
    assert {
        "available_cash",
        "open_positions",
        "pending_orders",
        "pending_proceeds",
        "completed_trades",
        "daily_equity",
        "event_audit",
        "safety_counters",
    } <= set(snapshot)
