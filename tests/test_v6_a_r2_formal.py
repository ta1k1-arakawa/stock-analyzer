from __future__ import annotations
import json, sys, shutil
from datetime import date, timedelta
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import v6_a_r2_formal as formal  # noqa: E402
from v6_a_r2_formal import ARTIFACTS, CONFIRMATION, FormalBlocked, atomic_write_formal_artifacts, build_engine_price_frames, build_fold_calendar, build_formal_artifacts, compute_aggregate_metrics, compute_fold_metrics, compute_twenty_gates, compute_v5b_comparison, run_formal_two_pass, run_one_fold

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
    fold = {"year": 2020, "candidates": [], "equity": [], "trades": [{"status": "CLOSED", "signal_date": "2020-01-01", "realized_net_profit_yen": x, "exit_execution_date": "2020-01-01"} for x in profits]}
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


def test_monthly_win_rate_uses_signal_month_not_exit_month():
    fold = {"year": 2020, "candidates": [], "equity": [], "trades": [{
        "status": "CLOSED", "signal_date": "2020-01-31", "exit_execution_date": "2020-02-10",
        "realized_net_profit_yen": 10.0}]}
    assert compute_fold_metrics(fold)["monthly_win_rate"] == 100.0


def test_missing_signal_date_is_formally_blocked():
    fold = {"year": 2020, "candidates": [], "equity": [], "trades": [{
        "status": "CLOSED", "exit_execution_date": "2020-02-10", "realized_net_profit_yen": 10.0}]}
    with pytest.raises(FormalBlocked, match="TRADE_SIGNAL_DATE_MISSING"):
        compute_fold_metrics(fold)


def _patch_two_pass_dependencies(monkeypatch, mutate_second=False):
    calls = {"artifact": 0}
    class Engine:
        def safety_counters(self):
            return {key: 0 for key in formal.SAFETY_KEYS}
    fold = {"year": 2020, "engine": Engine(), "trades": [], "equity": [], "candidates": []}
    monkeypatch.setattr(formal, "_validate_bundle", lambda bundle: None)
    monkeypatch.setattr(formal, "run_one_fold", lambda bundle, year: fold)
    monkeypatch.setattr(formal, "compute_fold_metrics", lambda value: {"yearly_profit": 0, "candidate_count": 0})
    monkeypatch.setattr(formal, "compute_aggregate_metrics", lambda folds, metrics: {
        "profit_factor_infinite": False, "profit_factor": 0.0, "net_profit": 0.0,
        "mark_to_market_maximum_drawdown": 0.0, "filled_trade_count": 0,
        "yearly_profit": {}, "top5_positive_profit_share": 0.0,
        "max_industry_positive_profit_share": 0.0})
    monkeypatch.setattr(formal, "compute_v5b_comparison", lambda aggregate, yearly: {
        "profit_factor_infinite": False, "profit_factor_difference": 0.0,
        "net_profit_difference": 0.0, "mtm_dd_difference": 0.0,
        "filled_trade_difference": 0, "positive_year_count_difference": 0,
        "yearly_profit_difference": {}})
    monkeypatch.setattr(formal, "compute_twenty_gates", lambda *args: {"two_pass_byte_identical": True})
    def artifacts(summary, folds, audit):
        calls["artifact"] += 1
        suffix = b"-changed" if mutate_second and calls["artifact"] == 2 else b""
        return {name: name.encode() + suffix for name in ARTIFACTS}
    monkeypatch.setattr(formal, "build_formal_artifacts", artifacts)
    return calls


def test_run_formal_two_pass_final_artifacts_match(monkeypatch):
    calls = _patch_two_pass_dependencies(monkeypatch)
    result = run_formal_two_pass({"candidate_audit": []}, {})
    assert calls["artifact"] == 2
    assert set(result["artifacts"]) == set(ARTIFACTS)


def test_run_formal_two_pass_detects_pass_two_artifact_change(monkeypatch):
    _patch_two_pass_dependencies(monkeypatch, mutate_second=True)
    with pytest.raises(FormalBlocked, match="TWO_PASS_ARTIFACT_MISMATCH"):
        run_formal_two_pass({"candidate_audit": []}, {})


def _patch_runner_repository(monkeypatch, runner):
    class Completed:
        def __init__(self, stdout): self.stdout = stdout
    def fake_run(command, **kwargs):
        if command[1] == "branch": return Completed("v6-a-r2-causal-breakout-baseline\n")
        if command[1] == "rev-parse" and command[2] == "HEAD": return Completed("sha\n")
        if command[1] == "rev-parse": return Completed("sha\n")
        return Completed("")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)


def _runner_args():
    return ["--evaluate-cache", "--confirmation", CONFIRMATION,
            "--training-cache", "training", "--evaluation-cache", "evaluation",
            "--output-dir", "C:/outside/formal"]


def _runner_preflight():
    return {"training_manifest_sha": "training-sha", "evaluation_manifest_sha": "evaluation-sha",
            "universe_csv_sha": "universe-sha", "ticker_list_sha": "ticker-sha"}


def _runner_json(capsys):
    return json.loads(capsys.readouterr().out.strip())


def test_runner_preparation_failure_reports_zero_counters(monkeypatch, capsys):
    import run_v6_a_r2_causal_breakout as runner
    _patch_runner_repository(monkeypatch, runner)
    monkeypatch.setattr(runner, "validate_output_target", lambda *args: None)
    monkeypatch.setattr(runner, "prepare_read_only_formal_bundle", lambda *args: (_ for _ in ()).throw(FormalBlocked("PREPARATION_FAILED")))
    assert runner.main(_runner_args()) == 1
    payload = _runner_json(capsys)
    assert payload["portfolio_simulation_started"] == 0 and payload["formal_artifacts_written"] == 0
    assert payload["error_code"] == "PREPARATION_FAILED"


def test_runner_formal_failure_reports_started_only(monkeypatch, capsys):
    import run_v6_a_r2_causal_breakout as runner
    _patch_runner_repository(monkeypatch, runner)
    monkeypatch.setattr(runner, "validate_output_target", lambda *args: None)
    preparation = type("Preparation", (), {"preflight_result": _runner_preflight(), "raw_price_frames": {}, "common_calendar": [], "accepted_candidates": [], "full_candidate_audit": [], "market_gate_audit": {}})()
    monkeypatch.setattr(runner, "prepare_read_only_formal_bundle", lambda *args: preparation)
    monkeypatch.setattr(runner, "build_formal_bundle", lambda *args: {})
    monkeypatch.setattr(runner, "run_formal_two_pass", lambda *args: (_ for _ in ()).throw(FormalBlocked("FORMAL_FAILED")))
    assert runner.main(_runner_args()) == 1
    payload = _runner_json(capsys)
    assert payload["portfolio_simulation_started"] == 1 and payload["formal_artifacts_written"] == 0
    assert payload["error_code"] == "FORMAL_FAILED"


def test_runner_atomic_writer_failure_reports_started_only(monkeypatch, capsys):
    import run_v6_a_r2_causal_breakout as runner
    _patch_runner_repository(monkeypatch, runner)
    monkeypatch.setattr(runner, "validate_output_target", lambda *args: None)
    preparation = type("Preparation", (), {"preflight_result": _runner_preflight(), "raw_price_frames": {}, "common_calendar": [], "accepted_candidates": [], "full_candidate_audit": [], "market_gate_audit": {}})()
    monkeypatch.setattr(runner, "prepare_read_only_formal_bundle", lambda *args: preparation)
    monkeypatch.setattr(runner, "build_formal_bundle", lambda *args: {})
    monkeypatch.setattr(runner, "run_formal_two_pass", lambda *args: {"summary": {"verdict": "PASS"}, "artifacts": {}})
    monkeypatch.setattr(runner, "atomic_write_formal_artifacts", lambda *args: (_ for _ in ()).throw(FormalBlocked("WRITE_FAILED")))
    assert runner.main(_runner_args()) == 1
    payload = _runner_json(capsys)
    assert payload["portfolio_simulation_started"] == 1 and payload["formal_artifacts_written"] == 0
    assert payload["error_code"] == "WRITE_FAILED"


def test_runner_success_reports_four_artifacts_and_single_preparation(monkeypatch, capsys):
    import run_v6_a_r2_causal_breakout as runner
    _patch_runner_repository(monkeypatch, runner)
    calls = {"prepare": 0, "formal": 0, "atomic": 0}
    monkeypatch.setattr(runner, "validate_output_target", lambda *args: None)
    preparation = type("Preparation", (), {"preflight_result": _runner_preflight(), "raw_price_frames": {}, "common_calendar": [], "accepted_candidates": [], "full_candidate_audit": [], "market_gate_audit": {}})()
    def prepare(*args): calls["prepare"] += 1; return preparation
    monkeypatch.setattr(runner, "prepare_read_only_formal_bundle", prepare)
    monkeypatch.setattr(runner, "build_formal_bundle", lambda *args: {})
    def formal_run(*args): calls["formal"] += 1; return {"summary": {"verdict": "PASS"}, "artifacts": {}}
    monkeypatch.setattr(runner, "run_formal_two_pass", formal_run)
    monkeypatch.setattr(runner, "atomic_write_formal_artifacts", lambda *args: calls.__setitem__("atomic", calls["atomic"] + 1))
    assert runner.main(_runner_args()) == 0
    payload = _runner_json(capsys)
    assert calls == {"prepare": 1, "formal": 1, "atomic": 1}
    assert payload["portfolio_simulation_started"] == 1 and payload["formal_artifacts_written"] == 4


def test_runner_confirmation_mismatch_does_not_prepare(capsys):
    import run_v6_a_r2_causal_breakout as runner
    assert runner.main(["--evaluate-cache", "--confirmation", "WRONG"]) == 2
    payload = _runner_json(capsys)
    assert payload["portfolio_simulation_started"] == 0 and payload["formal_artifacts_written"] == 0


def test_no_placeholder_tests():
    root = Path(__file__).resolve().parents[1] / "tests"
    text = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("test_v6_a2_*.py"))
    assert "pass\n" not in text
