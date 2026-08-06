from __future__ import annotations
import json, sys, shutil
from datetime import date, timedelta
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import v6_a_r2_formal as formal  # noqa: E402
from v6_a_r2_formal import ARTIFACTS, FormalBlocked, atomic_write_formal_artifacts, build_engine_price_frames, build_fold_calendar, build_formal_artifacts, compute_aggregate_metrics, compute_fold_metrics, compute_twenty_gates, compute_v5b_comparison, run_one_fold

def _fixture():
    calendar = [(date(2020, 12, 22) + timedelta(days=i)).isoformat() for i in range(15)]
    frames = {"AAA": {day: {"Open": 100.0, "Close": 100.0} for day in calendar}}
    frames["AAA"][calendar[10]]["Open"] = 110.0
    candidate = {"signal_year": 2020, "signal_date": calendar[0], "ticker": "AAA", "industry": "TECH", "rank": 1, "signal_raw_close": 100.0, "entry_attempt_date": calendar[1], "planned_exit_date": calendar[10], "candidate_status": "ACCEPTED_TOP20"}
    return {"price_frames": frames, "common_calendar": calendar, "accepted_candidates": [candidate]}, calendar

def test_fold_isolation_cross_year_exit_and_d11_release():
    bundle, calendar = _fixture(); assert build_fold_calendar(bundle, 2020) == calendar[:12]
    fold = run_one_fold(bundle, 2020); state = fold["engine"].state
    assert state.available_cash == pytest.approx(400000 + (110 * .9997 - 100 * 1.0003) * 100)
    assert not state.open_positions and not state.pending_orders_by_entry_date and not state.pending_proceeds_by_available_date
    assert fold["trades"][0]["exit_execution_date"].startswith("2021-") and fold["equity"][-1]["date"] == calendar[11]

def test_price_adapter_and_metrics_exclude_skips():
    bundle, _ = _fixture(); assert build_engine_price_frames(bundle["price_frames"])["AAA"]
    fold = run_one_fold(bundle, 2020); fold["trades"].append({"status": "SKIPPED", "skip_reason": "CASH_RESERVE", "realized_net_profit_yen": 99999})
    assert compute_fold_metrics(fold)["filled_trade_count"] == 1

def test_hash_and_future_candidate_column_block_before_engine(monkeypatch):
    bundle, _ = _fixture(); bundle["preflight_diagnostics"] = {"market_gate_counts": {"pass_days": 691, "blocked_days": 774}, "split_violations": 0, "duplicate_accepted_key": 0, "2026_signals": 0}
    bundle["accepted_candidate_key_sha256"] = "bad"
    with pytest.raises(FormalBlocked, match="HASH"): formal._validate_bundle(bundle)
    bundle["accepted_candidate_key_sha256"] = formal.EXPECTED_HASH
    bundle["accepted_candidates"][0]["d10_open"] = 1.0
    with pytest.raises(FormalBlocked, match="FUTURE_VALUE"): formal._validate_bundle(bundle)

@pytest.mark.parametrize("profits,pf,infinite", [([10, -5], 2.0, False), ([10], None, True), ([], 0.0, False)])
def test_profit_factor_json_safe(profits, pf, infinite):
    fold = {"year": 2020, "candidates": [], "equity": [], "trades": [{"status": "CLOSED", "realized_net_profit_yen": x, "exit_execution_date": "2020-01-01"} for x in profits]}
    metrics = compute_fold_metrics(fold); assert metrics["profit_factor"] == pf and metrics["profit_factor_infinite"] is infinite; json.dumps(metrics, allow_nan=False)

def test_aggregate_dd_and_gate_order():
    yearly = {str(y): {"yearly_profit": 0, "filled_trade_count": 0, "mark_to_market_maximum_drawdown": dd, "book_cost_maximum_drawdown": dd} for y, dd in ((2020, 10), (2021, 30))}
    aggregate = compute_aggregate_metrics([{"trades": []}, {"trades": []}], yearly); assert aggregate["mark_to_market_maximum_drawdown"] == 30
    comparison = compute_v5b_comparison(aggregate, {str(y): {"yearly_profit": 0} for y in range(2020, 2026)})
    safety = {k: 0 for k in ("negative_cash_count", "same_day_proceeds_reuse_count", "duplicate_order_count", "max_position_violation_count", "cash_reserve_violation_count", "industry_overlap_violation_count", "signal_2026_count")}
    gates = compute_twenty_gates(aggregate, {str(y): {"yearly_profit": 0, "filled_trade_count": 0} for y in range(2020, 2026)}, comparison, safety, True)
    assert len(gates) == 20 and list(gates)[-1] == "two_pass_byte_identical"

def test_artifacts_atomic_and_full_audit():
    bundle, _ = _fixture(); fold = run_one_fold(bundle, 2020)
    artifacts = build_formal_artifacts({"schema_version": "V6-A-R2-1"}, [fold], [{"candidate_status": "REJECTED", "ticker": "ZZZ"}, {"candidate_status": "ACCEPTED_TOP20", "ticker": "AAA"}])
    assert set(artifacts) == set(ARTIFACTS) and b"REJECTED" in artifacts["candidates.csv"] and json.loads(artifacts["summary.json"])
    root = Path.cwd() / ".v6_a_r2_formal_test_tmp"
    if root.exists(): shutil.rmtree(root)
    try:
        target = root / "output"; atomic_write_formal_artifacts(target, artifacts, root / "repo")
        assert {p.name for p in target.iterdir()} == set(ARTIFACTS)
        with pytest.raises(FormalBlocked): atomic_write_formal_artifacts(target, artifacts, root / "repo")
        with pytest.raises(FormalBlocked): atomic_write_formal_artifacts(root / "repo" / "out", artifacts, root / "repo")
    finally:
        if root.exists(): shutil.rmtree(root)
