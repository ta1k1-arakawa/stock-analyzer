from __future__ import annotations

from dataclasses import asdict, replace
from datetime import date, timedelta

import pytest

from src.v6_a_r2_causal_breakout import CausalEventEngine as V6Engine
from src.v7_capacity_engine import (
    CausalEventEngine as V7Engine,
    V7StudyBlocked,
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


def candidate(days, ticker, industry, rank, signal_index=0, exit_index=10, signal_year=None):
    return {
        "signal_year": int(signal_year if signal_year is not None else days[signal_index][:4]),
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


@pytest.mark.parametrize("year", [2026, 2027])
def test_forward_year_candidate_queues_fills_and_d10_exits(year):
    start = date(year, 1, 2)
    days = [(start + timedelta(days=index)).isoformat() for index in range(12)]
    frames = frames_for(days, ("AAA",))
    engine = V7Engine(
        frames,
        days,
        [candidate(days, "AAA", "TECH", 1, signal_year=year)],
        V7EngineParameters.control(),
    ).run()
    assert any(event["event"] == "ORDER_QUEUED" for event in engine.state.event_audit)
    assert engine.state.completed_trades[0]["status"] == "CLOSED"
    assert engine.state.completed_trades[0]["exit_execution_date"] == days[10]


def test_signal_year_must_match_signal_date_year():
    days = [
        (date(2026, 1, 2) + timedelta(days=index)).isoformat()
        for index in range(12)
    ]
    row = candidate(days, "AAA", "TECH", 1, signal_year=2027)
    with pytest.raises(ValueError, match="SIGNAL_YEAR_MISMATCH"):
        V7Engine(frames_for(days, ("AAA",)), days, [row], V7EngineParameters.control())


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


def test_sticky_safety_counters_persist_across_refresh_and_snapshot():
    days = calendar(12)
    engine = V7Engine(frames_for(days, ("AAA",)), days, [], V7EngineParameters.control())
    for name in ("historical_backfill", "snapshot_rewrite", "cross_arm_state_leakage"):
        engine.record_safety_violation(name)
    assert engine.safety_counters()["historical_backfill"] == 1
    assert engine.state_snapshot()["safety_counters"]["snapshot_rewrite"] == 1
    engine._refresh_safety_counters()
    assert engine.safety_counters()["cross_arm_state_leakage"] == 1


@pytest.mark.parametrize(
    "name,count",
    [("unknown", 1), ("historical_backfill", 0), ("snapshot_rewrite", -1), ("D0_state_mutation", True), ("future_price_access", 1)],
)
def test_sticky_safety_record_api_rejects_invalid_requests(name, count):
    days = calendar(12)
    engine = V7Engine(frames_for(days, ("AAA",)), days, [], V7EngineParameters.control())
    with pytest.raises(ValueError):
        engine.record_safety_violation(name, count)


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


@pytest.mark.parametrize("open_value", ["MISSING", float("nan"), 0.0])
def test_missing_or_invalid_d1_open_is_entry_data_unavailable(open_value):
    days = calendar(12)
    frames = frames_for(days, ("AAA",))
    if open_value == "MISSING":
        del frames["AAA"][days[1]]["Open"]
    else:
        frames["AAA"][days[1]]["Open"] = open_value
    engine = V7Engine(frames, days, [candidate(days, "AAA", "TECH", 1)])
    engine.run()
    row = engine.state.completed_trades[0]
    assert row["status"] == "SKIPPED"
    assert row["skip_reason"] == "ENTRY_DATA_UNAVAILABLE"
    assert row["entry_price"] is None
    assert row["entry_cost"] is None
    assert engine.safety_counters()["planned_exit_price_unavailable"] == 0


def test_d1_missing_skip_preserves_cash_positions_and_proceeds():
    days = calendar(12)
    frames = frames_for(days, ("AAA",))
    del frames["AAA"][days[1]]["Open"]
    engine = V7Engine(frames, days, [candidate(days, "AAA", "TECH", 1)])
    engine.process_day(days[0])
    before = (
        engine.state.available_cash,
        list(engine.state.open_positions),
        dict(engine.state.pending_proceeds_by_available_date),
    )
    engine.process_day(days[1])
    after = (
        engine.state.available_cash,
        list(engine.state.open_positions),
        dict(engine.state.pending_proceeds_by_available_date),
    )
    assert before == after
    assert engine.skip_reason_counts()["ENTRY_DATA_UNAVAILABLE"] == 1
    assert engine.safety_counters()["open_position_split_spanning"] == 0


def test_d0_signal_close_integrity_error_is_not_entry_data_skip():
    days = calendar(12)
    frames = frames_for(days, ("AAA",))
    del frames["AAA"][days[0]]["Close"]
    engine = V7Engine(frames, days, [candidate(days, "AAA", "TECH", 1)])
    engine.process_day(days[0])
    with pytest.raises(ValueError, match="FIELD_NOT_FOUND"):
        engine.process_day(days[1])
    assert engine.skip_reason_counts()["ENTRY_DATA_UNAVAILABLE"] == 0


def test_split_effective_on_d1_skips_without_safety_violation():
    days = calendar(12)
    frames = frames_for(days, ("AAA",))
    engine = V7Engine(
        frames, days, [candidate(days, "AAA", "TECH", 1)],
        split_events_by_day={days[1]: ["AAA"]},
    )
    engine.process_day(days[0])
    before = engine.state.available_cash
    engine.process_day(days[1])
    row = engine.state.completed_trades[0]
    assert row["status"] == "SKIPPED"
    assert row["skip_reason"] == "SPLIT_EFFECTIVE_BEFORE_ENTRY"
    assert row["cash_before_entry"] == row["cash_after_entry"] == before
    assert row["position_count_before_entry"] == row["position_count_after_entry"] == 0
    assert engine.state.open_positions == []
    assert engine.state.pending_proceeds_by_available_date == {}
    assert engine.safety_counters()["open_position_split_spanning"] == 0


def test_future_split_date_does_not_affect_current_entry():
    days = calendar(12)
    engine = V7Engine(
        frames_for(days, ("AAA",)), days, [candidate(days, "AAA", "TECH", 1)],
        split_events_by_day={days[5]: ["AAA"]},
    )
    engine.process_day(days[0])
    engine.process_day(days[1])
    assert engine.state.completed_trades[0]["status"] == "FILLED"
    assert engine.state.open_positions[0].ticker == "AAA"
    assert engine.safety_counters()["future_split_access"] == 0


def test_split_after_fill_blocks_before_later_phases():
    days = calendar(12)
    engine = V7Engine(
        frames_for(days, ("AAA",)), days, [candidate(days, "AAA", "TECH", 1)],
        split_events_by_day={days[2]: ["AAA"]},
    )
    engine.process_day(days[0])
    engine.process_day(days[1])
    with pytest.raises(V7StudyBlocked, match="OPEN_POSITION_SPLIT_SPANNING") as error:
        engine.process_day(days[2])
    assert error.value.reason == "OPEN_POSITION_SPLIT_SPANNING"
    assert any(event["event"] == "OPEN_POSITION_SPLIT_DETECTED" for event in engine.state.event_audit)
    assert not any(event["event"] == "EXIT_EXECUTED" and event["date"] == days[2] for event in engine.state.event_audit)
    assert not any(row["date"] == days[2] for row in engine.state.daily_equity)
    assert not any(event["event"] == "ORDER_QUEUED" and event["date"] == days[2] for event in engine.state.event_audit)
    assert engine.safety_counters()["open_position_split_spanning"] == 1


def test_split_effective_exactly_d10_blocks_before_exit():
    days = calendar(12)
    engine = V7Engine(
        frames_for(days, ("AAA",)), days, [candidate(days, "AAA", "TECH", 1)],
        split_events_by_day={days[10]: ["AAA"]},
    )
    for day in days[:10]:
        engine.process_day(day)
    with pytest.raises(V7StudyBlocked, match="OPEN_POSITION_SPLIT_SPANNING"):
        engine.process_day(days[10])
    assert engine.state.open_positions
    assert engine.state.completed_trades[0]["status"] == "FILLED"
    assert not any(event["event"] == "EXIT_EXECUTED" for event in engine.state.event_audit)


def test_missing_d10_open_blocks_without_position_or_ledger_mutation():
    days = calendar(12)
    frames = frames_for(days, ("AAA",))
    del frames["AAA"][days[10]]["Open"]
    engine = V7Engine(frames, days, [candidate(days, "AAA", "TECH", 1)])
    for day in days[:10]:
        engine.process_day(day)
    before_position = list(engine.state.open_positions)
    before_cash = engine.state.available_cash
    with pytest.raises(V7StudyBlocked, match="PLANNED_EXIT_PRICE_UNAVAILABLE") as error:
        engine.process_day(days[10])
    assert error.value.reason == "PLANNED_EXIT_PRICE_UNAVAILABLE"
    assert engine.state.open_positions == before_position
    assert engine.state.available_cash == before_cash
    assert engine.state.pending_proceeds_by_available_date == {}
    assert engine.state.completed_trades[0]["status"] == "FILLED"
    assert any(event["event"] == "D10_EXIT_BLOCKED_MISSING_PRICE" for event in engine.state.event_audit)
    assert not any(row["date"] == days[10] for row in engine.state.daily_equity)
    assert engine.safety_counters()["planned_exit_price_unavailable"] == 1


def test_missing_open_position_mtm_close_blocks_without_equity_append():
    days = calendar(12)
    frames = frames_for(days, ("AAA",))
    del frames["AAA"][days[2]]["Close"]
    engine = V7Engine(frames, days, [candidate(days, "AAA", "TECH", 1)])
    engine.process_day(days[0])
    engine.process_day(days[1])
    before_position = list(engine.state.open_positions)
    before_cash = engine.state.available_cash
    before_proceeds = dict(engine.state.pending_proceeds_by_available_date)
    with pytest.raises(V7StudyBlocked, match="OPEN_POSITION_MTM_PRICE_UNAVAILABLE") as error:
        engine.process_day(days[2])
    assert error.value.reason == "OPEN_POSITION_MTM_PRICE_UNAVAILABLE"
    assert engine.state.open_positions == before_position
    assert engine.state.available_cash == before_cash
    assert engine.state.pending_proceeds_by_available_date == before_proceeds
    assert not any(row["date"] == days[2] for row in engine.state.daily_equity)
    assert not any(event["event"] == "ORDER_QUEUED" and event["date"] == days[2] for event in engine.state.event_audit)
    assert any(event["event"] == "MTM_BLOCKED_MISSING_PRICE" for event in engine.state.event_audit)
    assert engine.safety_counters()["open_position_mtm_price_unavailable"] == 1


def test_new_sticky_safety_counters_survive_refresh_and_snapshot():
    days = calendar(12)
    engine = V7Engine(frames_for(days, ("AAA",)), days, [])
    names = (
        "future_candidate_data_access", "future_split_access",
        "open_position_split_spanning", "planned_exit_price_unavailable",
        "open_position_mtm_price_unavailable", "candidate_snapshot_rerank",
        "outside_top20_replacement",
    )
    for name in names:
        engine.record_safety_violation(name)
    engine.safety_counters()
    engine.state_snapshot()
    engine._refresh_safety_counters()
    counters = engine.safety_counters()
    assert all(counters[name] == 1 for name in names)
