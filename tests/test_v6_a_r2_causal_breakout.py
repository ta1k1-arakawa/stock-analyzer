from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from v6_a_r2_causal_breakout import (  # noqa: E402
    CausalEventEngine,
    fold_max_drawdown,
    read_price,
    run_synthetic_golden,
    synthetic_fixture,
    write_synthetic_artifacts,
)


def test_phase5_snapshot_only_queue_changes():
    result = run_synthetic_golden()
    engine = result.engine
    assert engine.safety_counters()["d0_state_mutation_violation_count"] == 0
    assert engine.state.completed_trades


def test_future_read_guard():
    calendar, frames, _ = synthetic_fixture()
    with pytest.raises(ValueError, match="FUTURE_PRICE_ACCESS_PROHIBITED"):
        read_price(frames, "AAA", calendar[1], "Open", calendar[0])


def test_fill_and_cash_only_on_d1():
    result = run_synthetic_golden()
    fills = [event for event in result.engine.state.event_audit if event["event"] == "ENTRY_FILLED"]
    assert fills and all(event["date"] == "2020-01-02" for event in fills)
    assert not any(event["event"] == "CASH_DEDUCTED" and event["date"] == "2020-01-01"
                   for event in result.engine.state.event_audit)


def test_d1_open_change_does_not_change_d0_snapshot_or_d0_equity():
    calendar, frames, candidates = synthetic_fixture()
    first = CausalEventEngine(frames, calendar, candidates)
    first.process_day(calendar[0])
    changed = copy.deepcopy(frames)
    changed["AAA"][calendar[1]]["Open"] = 999.0
    second = CausalEventEngine(changed, calendar, candidates)
    second.process_day(calendar[0])
    assert first.state.daily_equity == second.state.daily_equity
    assert first.state.available_cash == second.state.available_cash
    assert first.state.open_positions == second.state.open_positions


def test_later_prices_do_not_change_d1_decision():
    calendar, frames, candidates = synthetic_fixture()
    first = CausalEventEngine(frames, calendar, candidates).run()
    changed = copy.deepcopy(frames)
    changed["AAA"][calendar[5]]["Close"] = 777.0
    changed["AAA"][calendar[10]]["Open"] = 1.0
    second = CausalEventEngine(changed, calendar, candidates).run()
    first_d1 = [(r["ticker"], r["status"]) for r in first.state.completed_trades]
    second_d1 = [(r["ticker"], r["status"]) for r in second.state.completed_trades]
    assert first_d1 == second_d1


def test_exit_only_on_d10_and_proceeds_pending_then_released_d11():
    result = run_synthetic_golden()
    exits = [event for event in result.engine.state.event_audit if event["event"] == "EXIT_EXECUTED"]
    assert exits and all(event["date"] == "2020-01-11" for event in exits)
    releases = [event for event in result.engine.state.event_audit if event["event"] == "PROCEEDS_RELEASED"]
    assert releases and all(event["date"] == "2020-01-12" for event in releases)


def test_same_day_exit_proceeds_are_not_reused():
    result = run_synthetic_golden()
    assert not any(event["event"] == "CASH_DEDUCTED" and event["date"] == "2020-01-11"
                   for event in result.engine.state.event_audit)


def test_same_day_exit_occupies_slot_and_industry():
    result = run_synthetic_golden()
    skipped = [row for row in result.engine.state.completed_trades if row["ticker"] == "CCC"]
    assert skipped[0]["skip_reason"] == "SAME_INDUSTRY_OPEN"


def test_rank1_gap_skip_continues_to_rank2():
    result = run_synthetic_golden()
    rows = {row["ticker"]: row for row in result.scenario_b.state.completed_trades}
    assert rows["GAP"]["skip_reason"] == "ENTRY_GAP_TOO_HIGH"
    assert rows["BBB"]["status"] == "CLOSED"


def test_rank21_fails_closed():
    calendar, frames, _ = synthetic_fixture()
    row = {"signal_year": 2020, "signal_date": calendar[0], "ticker": "AAA", "industry": "TECH",
           "rank": 21, "signal_raw_close": 100.0, "entry_attempt_date": calendar[1],
           "planned_exit_date": calendar[10], "candidate_status": "ACCEPTED_TOP20"}
    with pytest.raises(ValueError, match="OUTSIDE_TOP20_CANDIDATE_PROHIBITED"):
        CausalEventEngine(frames, calendar, [row])


def test_duplicate_ticker_and_max_two_and_cash_reserve():
    result = run_synthetic_golden()
    assert len([row for row in result.engine.state.completed_trades if row["status"] in {"FILLED", "CLOSED"}]) == 2
    assert result.engine.safety_counters()["max_position_violation_count"] == 0
    assert result.engine.safety_counters()["cash_reserve_violation_count"] == 0


def test_ledger_dates_and_invariants():
    result = run_synthetic_golden()
    for row in result.engine.state.completed_trades:
        assert row["order_created_date"] == row["signal_date"]
        if row["status"] == "CLOSED":
            assert row["exit_execution_date"] == row["planned_exit_date"]
            assert row["proceeds_available_date"] > row["exit_execution_date"]


def test_book_and_mtm_are_separate():
    result = run_synthetic_golden()
    assert any(row["book_equity"] != row["mtm_equity"] for row in result.engine.state.daily_equity)


def test_fold_drawdown_uses_maximum_fold_dd():
    assert fold_max_drawdown({"a": [100, 90], "b": [100, 50]}) == pytest.approx(50.0)


def test_safety_and_concentration_metrics_are_measured():
    result = run_synthetic_golden()
    counters = result.engine.safety_counters()
    assert set(counters) == {"negative_cash_count", "same_day_proceeds_reuse_count", "duplicate_order_count",
                             "max_position_violation_count", "cash_reserve_violation_count",
                             "industry_overlap_violation_count", "signal_2026_count",
                             "future_price_access_violation_count", "d0_state_mutation_violation_count"}
    summary = json.loads(result.artifacts["summary.json"])
    assert "concentration_metrics" in summary
    assert summary["safety_counters"] == counters


def test_four_artifacts_and_two_pass_byte_equality():
    result = run_synthetic_golden()
    tmp_path = Path(__file__).resolve().parents[1] / ".v6_a_r2_test_artifacts"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    try:
        write_synthetic_artifacts(tmp_path, result)
        assert {path.name for path in tmp_path.iterdir()} == {"summary.json", "trades.csv", "candidates.csv", "daily_equity.csv"}
        first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
        other = tmp_path / "other"
        write_synthetic_artifacts(other, run_synthetic_golden())
        second = {path.name: path.read_bytes() for path in other.iterdir()}
        assert first == second
        assert json.loads(first["summary.json"])["two_pass_byte_identical"] is True
    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)


def test_2026_signal_fails_closed():
    calendar, frames, _ = synthetic_fixture()
    row = {"signal_year": 2026, "signal_date": calendar[0], "ticker": "AAA", "industry": "TECH",
           "rank": 1, "signal_raw_close": 100.0, "entry_attempt_date": calendar[1],
           "planned_exit_date": calendar[10], "candidate_status": "ACCEPTED_TOP20"}
    with pytest.raises(ValueError, match="SIGNAL_YEAR_MISMATCH|SIGNAL_2026_PROHIBITED"):
        CausalEventEngine(frames, calendar, [row])


def test_cli_and_new_files_have_no_data_acquisition_path():
    source = Path(__file__).resolve().parents[1]
    for path in [source / "src" / "v6_a_r2_causal_breakout.py", source / "scripts" / "run_v6_a_r2_causal_breakout.py"]:
        text = path.read_text(encoding="utf-8").lower()
        assert "requests" not in text and "urllib" not in text and "yfinance" not in text
