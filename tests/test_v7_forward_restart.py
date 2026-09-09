from __future__ import annotations

import copy
import math
from datetime import date, timedelta

import pytest

from src.v7_capacity_engine import CausalEventEngine, V7EngineParameters
from src.v7_forward_persistence import (
    ForwardStudyStore,
    V7ForwardPersistenceBlocked,
    canonical_json_bytes,
    export_engine_runtime,
    restore_engine_runtime,
    verify_forward_store,
)

ACTIVATION_SHA = "a" * 64
COLLECTOR_COMMIT = "b" * 40
SPLIT_INDEX = 11
CALENDAR_LENGTH = 24


def calendar(count: int = CALENDAR_LENGTH) -> list[str]:
    start = date(2020, 1, 2)
    return [(start + timedelta(days=index)).isoformat() for index in range(count)]


def _candidate(days, ticker, industry, rank, signal_index, exit_offset=10):
    return {
        "signal_year": int(days[signal_index][:4]),
        "signal_date": days[signal_index],
        "ticker": ticker,
        "industry": industry,
        "rank": rank,
        "signal_raw_close": 100.0 + signal_index,
        "entry_attempt_date": days[signal_index + 1],
        "planned_exit_date": days[signal_index + exit_offset],
        "candidate_status": "ACCEPTED_TOP20",
    }


def restart_fixture():
    days = calendar()
    tickers = ("OPEN", "PROCEEDS", "ORDER", "SKIP")
    frames = {
        ticker: {day: {"Open": 100.0 + index, "Close": 100.0 + index} for index, day in enumerate(days)}
        for ticker in tickers
    }
    candidates = [
        _candidate(days, "PROCEEDS", "IND_PROCEEDS", 1, signal_index=0),
        _candidate(days, "OPEN", "IND_OPEN", 1, signal_index=2),
        _candidate(days, "SKIP", "IND_SKIP", 1, signal_index=3),
        _candidate(days, "ORDER", "IND_ORDER", 1, signal_index=10),
    ]
    return days, frames, candidates


def run_continuous(parameters: V7EngineParameters) -> CausalEventEngine:
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, parameters)
    for day in days:
        engine.process_day(day)
    return engine


def run_with_restart(parameters: V7EngineParameters, split_index: int = SPLIT_INDEX) -> CausalEventEngine:
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, parameters)
    for day in days[:split_index]:
        engine.process_day(day)
    exported = export_engine_runtime(engine)
    resumed = CausalEventEngine(frames, days, candidates, parameters)
    restore_engine_runtime(resumed, exported)
    for day in days[split_index:]:
        resumed.process_day(day)
    return resumed


# ---------------------------------------------------------------------------
# export/restore roundtrip basics
# ---------------------------------------------------------------------------


def test_export_engine_runtime_has_expected_schema():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    for day in days[:SPLIT_INDEX]:
        engine.process_day(day)
    payload = export_engine_runtime(engine)
    assert set(payload) == {
        "schema_version", "parameters_sha256", "engine_day", "available_cash",
        "open_positions", "pending_orders_by_entry_date", "pending_proceeds_by_available_date",
        "completed_trades", "daily_equity", "event_audit", "safety_counters", "skip_reason_counts",
    }
    assert payload["schema_version"] == 1
    assert payload["parameters_sha256"] == engine.parameters.sha256()


def test_control_export_restore_roundtrip_available_cash():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    for day in days[:SPLIT_INDEX]:
        engine.process_day(day)
    payload = export_engine_runtime(engine)
    resumed = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    restore_engine_runtime(resumed, payload)
    assert resumed.state.available_cash == engine.state.available_cash
    assert resumed.state.engine_day == engine.state.engine_day


def test_capacity3_export_restore_roundtrip_available_cash():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.capacity_3())
    for day in days[:SPLIT_INDEX]:
        engine.process_day(day)
    payload = export_engine_runtime(engine)
    resumed = CausalEventEngine(frames, days, candidates, V7EngineParameters.capacity_3())
    restore_engine_runtime(resumed, payload)
    assert resumed.state.available_cash == engine.state.available_cash


def test_restore_control_runtime_into_capacity3_engine_blocked():
    days, frames, candidates = restart_fixture()
    control = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    for day in days[:SPLIT_INDEX]:
        control.process_day(day)
    payload = export_engine_runtime(control)
    variant = CausalEventEngine(frames, days, candidates, V7EngineParameters.capacity_3())
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        restore_engine_runtime(variant, payload)
    assert excinfo.value.reason == "RUNTIME_PARAMETERS_MISMATCH"


def test_restore_capacity3_runtime_into_control_engine_blocked():
    days, frames, candidates = restart_fixture()
    variant = CausalEventEngine(frames, days, candidates, V7EngineParameters.capacity_3())
    for day in days[:SPLIT_INDEX]:
        variant.process_day(day)
    payload = export_engine_runtime(variant)
    control = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        restore_engine_runtime(control, payload)
    assert excinfo.value.reason == "RUNTIME_PARAMETERS_MISMATCH"


def test_restore_missing_field_blocked():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    payload = export_engine_runtime(engine)
    del payload["daily_equity"]
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        restore_engine_runtime(engine, payload)
    assert excinfo.value.reason.startswith("RUNTIME_FIELD_MISSING:")


def test_restore_unknown_field_blocked():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    payload = export_engine_runtime(engine)
    payload["unexpected_field"] = 1
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        restore_engine_runtime(engine, payload)
    assert excinfo.value.reason.startswith("RUNTIME_FIELD_UNKNOWN:")


def test_restore_schema_version_mismatch_blocked():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    payload = export_engine_runtime(engine)
    payload["schema_version"] = 2
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        restore_engine_runtime(engine, payload)
    assert excinfo.value.reason == "RUNTIME_SCHEMA_VERSION_MISMATCH"


def test_restore_nonfinite_value_blocked():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    payload = export_engine_runtime(engine)
    payload["available_cash"] = math.nan
    with pytest.raises(V7ForwardPersistenceBlocked):
        restore_engine_runtime(engine, payload)


# ---------------------------------------------------------------------------
# State-shape preservation across restart
# ---------------------------------------------------------------------------


def test_pending_order_preserved_across_restart():
    days, frames, candidates = restart_fixture()
    continuous = run_continuous(V7EngineParameters.control())
    restarted = run_with_restart(V7EngineParameters.control())
    continuous_orders = export_engine_runtime(continuous)["completed_trades"]
    restarted_orders = export_engine_runtime(restarted)["completed_trades"]
    order_row = next(row for row in continuous_orders if row["ticker"] == "ORDER")
    assert order_row["status"] in {"FILLED", "CLOSED"}
    assert order_row == next(row for row in restarted_orders if row["ticker"] == "ORDER")


def test_open_position_preserved_at_split_boundary():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    for day in days[:SPLIT_INDEX]:
        engine.process_day(day)
    payload = export_engine_runtime(engine)
    assert any(item["ticker"] == "OPEN" for item in payload["open_positions"])


def test_pending_order_present_at_split_boundary():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    for day in days[:SPLIT_INDEX]:
        engine.process_day(day)
    payload = export_engine_runtime(engine)
    queued = payload["pending_orders_by_entry_date"].get(days[SPLIT_INDEX], [])
    assert any(item["ticker"] == "ORDER" for item in queued)


def test_pending_proceeds_present_at_split_boundary():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    for day in days[:SPLIT_INDEX]:
        engine.process_day(day)
    payload = export_engine_runtime(engine)
    pending = payload["pending_proceeds_by_available_date"].get(days[SPLIT_INDEX], [])
    assert any(item["order_id"].startswith(days[0]) for item in pending)


def test_completed_trade_preserved_across_restart():
    continuous = run_continuous(V7EngineParameters.control())
    restarted = run_with_restart(V7EngineParameters.control())
    assert export_engine_runtime(continuous)["completed_trades"] == export_engine_runtime(restarted)["completed_trades"]


def test_daily_equity_preserved_across_restart():
    continuous = run_continuous(V7EngineParameters.control())
    restarted = run_with_restart(V7EngineParameters.control())
    assert export_engine_runtime(continuous)["daily_equity"] == export_engine_runtime(restarted)["daily_equity"]


def test_event_audit_preserved_across_restart():
    continuous = run_continuous(V7EngineParameters.control())
    restarted = run_with_restart(V7EngineParameters.control())
    assert export_engine_runtime(continuous)["event_audit"] == export_engine_runtime(restarted)["event_audit"]


def test_sticky_safety_counter_preserved_across_restart():
    days, frames, candidates = restart_fixture()
    engine = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    for day in days[:SPLIT_INDEX]:
        engine.process_day(day)
    engine.record_safety_violation("snapshot_rewrite", 3)
    payload = export_engine_runtime(engine)
    assert payload["safety_counters"]["snapshot_rewrite"] == 3
    resumed = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    restore_engine_runtime(resumed, payload)
    assert resumed.safety_counters()["snapshot_rewrite"] == 3
    for day in days[SPLIT_INDEX:]:
        resumed.process_day(day)
    assert resumed.safety_counters()["snapshot_rewrite"] == 3


def test_derived_safety_counter_matches_continuous_after_restart():
    continuous = run_continuous(V7EngineParameters.control())
    restarted = run_with_restart(V7EngineParameters.control())
    assert continuous.safety_counters() == restarted.safety_counters()


def test_skip_reason_counts_preserved_across_restart():
    continuous = run_continuous(V7EngineParameters.control())
    restarted = run_with_restart(V7EngineParameters.control())
    assert continuous.skip_reason_counts()["MAX_OPEN_POSITIONS"] >= 1
    assert continuous.skip_reason_counts() == restarted.skip_reason_counts()


def test_arm_independence_restore_does_not_leak_across_engines():
    days, frames, candidates = restart_fixture()
    control = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    variant = CausalEventEngine(frames, days, candidates, V7EngineParameters.capacity_3())
    for day in days[:SPLIT_INDEX]:
        control.process_day(day)
        variant.process_day(day)
    control_payload = export_engine_runtime(control)
    variant_payload = export_engine_runtime(variant)
    assert control_payload["skip_reason_counts"]["MAX_OPEN_POSITIONS"] >= 1
    assert variant_payload["skip_reason_counts"]["MAX_OPEN_POSITIONS"] == 0
    assert len(control_payload["open_positions"]) != len(variant_payload["open_positions"]) or (
        control_payload["completed_trades"] != variant_payload["completed_trades"]
    )


# ---------------------------------------------------------------------------
# Restart equivalence (byte-identical final runtime)
# ---------------------------------------------------------------------------


def test_control_restart_final_runtime_byte_identical():
    continuous = export_engine_runtime(run_continuous(V7EngineParameters.control()))
    restarted = export_engine_runtime(run_with_restart(V7EngineParameters.control()))
    assert canonical_json_bytes(continuous) == canonical_json_bytes(restarted)


def test_capacity3_restart_final_runtime_byte_identical():
    continuous = export_engine_runtime(run_continuous(V7EngineParameters.capacity_3()))
    restarted = export_engine_runtime(run_with_restart(V7EngineParameters.capacity_3()))
    assert canonical_json_bytes(continuous) == canonical_json_bytes(restarted)


@pytest.mark.parametrize("split_index", [1, 5, 11, 19, 23])
def test_control_restart_byte_identical_various_split_points(split_index):
    continuous = export_engine_runtime(run_continuous(V7EngineParameters.control()))
    restarted = export_engine_runtime(run_with_restart(V7EngineParameters.control(), split_index))
    assert canonical_json_bytes(continuous) == canonical_json_bytes(restarted)


def test_double_restart_still_byte_identical():
    days, frames, candidates = restart_fixture()
    parameters = V7EngineParameters.control()
    engine = CausalEventEngine(frames, days, candidates, parameters)
    for day in days[:5]:
        engine.process_day(day)
    payload_one = export_engine_runtime(engine)
    resumed_one = CausalEventEngine(frames, days, candidates, parameters)
    restore_engine_runtime(resumed_one, payload_one)
    for day in days[5:15]:
        resumed_one.process_day(day)
    payload_two = export_engine_runtime(resumed_one)
    resumed_two = CausalEventEngine(frames, days, candidates, parameters)
    restore_engine_runtime(resumed_two, payload_two)
    for day in days[15:]:
        resumed_two.process_day(day)

    continuous = export_engine_runtime(run_continuous(parameters))
    assert canonical_json_bytes(continuous) == canonical_json_bytes(export_engine_runtime(resumed_two))


# ---------------------------------------------------------------------------
# Store-integrated checkpoint chain and resume
# ---------------------------------------------------------------------------


def _store_write(store, day, control, variant):
    return store.write_day(
        day,
        price_snapshot={"date": day},
        candidate_snapshot={"date": day},
        market_gate_snapshot={"date": day},
        arm_a_runtime=export_engine_runtime(control),
        arm_b_runtime=export_engine_runtime(variant),
        activation_manifest_sha256=ACTIVATION_SHA,
        collector_commit=COLLECTOR_COMMIT,
    )


def test_one_day_checkpoint_chain(tmp_path):
    days, frames, candidates = restart_fixture()
    control = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    variant = CausalEventEngine(frames, days, candidates, V7EngineParameters.capacity_3())
    store = ForwardStudyStore(tmp_path)
    control.process_day(days[0])
    variant.process_day(days[0])
    record = _store_write(store, days[0], control, variant)
    assert record["previous_checkpoint_sha256"] is None
    result = verify_forward_store(tmp_path, ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert result["day_count"] == 1


def test_two_day_checkpoint_chain(tmp_path):
    days, frames, candidates = restart_fixture()
    control = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    variant = CausalEventEngine(frames, days, candidates, V7EngineParameters.capacity_3())
    store = ForwardStudyStore(tmp_path)
    for day in days[:2]:
        control.process_day(day)
        variant.process_day(day)
        _store_write(store, day, control, variant)
    result = verify_forward_store(tmp_path, ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert result["day_count"] == 2
    assert result["verified_days"] == days[:2]


def test_three_day_checkpoint_chain(tmp_path):
    days, frames, candidates = restart_fixture()
    control = CausalEventEngine(frames, days, candidates, V7EngineParameters.control())
    variant = CausalEventEngine(frames, days, candidates, V7EngineParameters.capacity_3())
    store = ForwardStudyStore(tmp_path)
    for day in days[:3]:
        control.process_day(day)
        variant.process_day(day)
        _store_write(store, day, control, variant)
    result = verify_forward_store(tmp_path, ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert result["day_count"] == 3
    assert result["verified_days"] == days[:3]


def test_resume_from_store_matches_continuous_run(tmp_path):
    days, frames, candidates = restart_fixture()
    parameters = V7EngineParameters.control()
    other_parameters = V7EngineParameters.capacity_3()
    control = CausalEventEngine(frames, days, candidates, parameters)
    variant = CausalEventEngine(frames, days, candidates, other_parameters)
    store = ForwardStudyStore(tmp_path)
    for day in days[:SPLIT_INDEX]:
        control.process_day(day)
        variant.process_day(day)
        _store_write(store, day, control, variant)

    latest = store.load_latest_runtime()
    assert latest["day"] == days[SPLIT_INDEX - 1]
    resumed_control = CausalEventEngine(frames, days, candidates, parameters)
    restore_engine_runtime(resumed_control, latest["arm_a_runtime"])
    resumed_variant = CausalEventEngine(frames, days, candidates, other_parameters)
    restore_engine_runtime(resumed_variant, latest["arm_b_runtime"])
    for day in days[SPLIT_INDEX:]:
        resumed_control.process_day(day)
        resumed_variant.process_day(day)

    continuous_control = run_continuous(parameters)
    continuous_variant = run_continuous(other_parameters)
    assert canonical_json_bytes(export_engine_runtime(resumed_control)) == canonical_json_bytes(
        export_engine_runtime(continuous_control)
    )
    assert canonical_json_bytes(export_engine_runtime(resumed_variant)) == canonical_json_bytes(
        export_engine_runtime(continuous_variant)
    )
