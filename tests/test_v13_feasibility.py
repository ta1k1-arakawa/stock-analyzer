from datetime import date
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.v13_feasibility import (FEATURES, Q_KEYS, SessionCalendar, adjudicate, base_target,
    build_rank_population, canonical_json, diagnostics, lightgbm_factory, monthly_predictions,
    random_key, rank_candidates, ridge_factory, select_universe, simulate, trade_metrics)
from src.v13_synthetic_fixture import active_rows, stage_a_rows, synthetic_manifest

ROOT = Path(__file__).resolve().parents[1]

def test_frozen_bindings_and_manifest_are_deterministic():
    assert subprocess.check_output(["git", "rev-parse", "61268237494e2968562983e456ea40e6f821d066:V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md"], cwd=ROOT, text=True).strip() == "3bfcd695c69f6dac480f8fc99ca4f3916f668e4a"
    assert synthetic_manifest() == synthetic_manifest()
    assert len(synthetic_manifest()["selected"]) == 500

def test_selection_and_calendar_target_semantics():
    with pytest.raises(ValueError): select_universe(["1000"], [], "x")
    cal = SessionCalendar((date(2020,1,2), date(2020,1,3), date(2020,1,6), date(2020,1,7)))
    assert cal.plus(date(2020,1,2), 3) == date(2020,1,7)
    assert cal.plus(date(2020,1,3), 3) is None
    assert base_target(100, 110) == pytest.approx(9.7802197802)
    assert base_target(0, 110) is None

def test_stage_b_c_and_no_rank_failure():
    status, ranked = build_rank_population(stage_a_rows())
    assert status == "OK" and len(ranked) == 12 and all(set(FEATURES) <= set(r) for r in ranked)
    broken = stage_a_rows()
    for row in broken: row["ret_1"] = 1.0
    assert build_rank_population(broken)[0] == "NO_RANK_DATA_QUALITY"

def test_factories_actual_fit_and_monthly_asof():
    x, y = np.arange(40).reshape(20,2), np.arange(20, dtype=float)
    assert lightgbm_factory().get_params()["n_estimators"] == 400
    assert lightgbm_factory().fit(x,y).predict(x[:1]).shape == (1,)
    assert ridge_factory().fit(x,y).predict(x[:1]).shape == (1,)
    predicted, fits = monthly_predictions(active_rows())
    assert len(fits) >= 2 and all(n == 2 for n in fits.values())
    # The production implementation filters every fit to labels known before
    # the prediction-month boundary; predictions themselves naturally exit later.
    import inspect
    assert 'r["exit"] < date(year, mon, 1)' in inspect.getsource(monthly_predictions)

def test_rankings_random_and_diagnostics():
    rows = [{"code":"1001","lightgbm_score":1.,"ridge_score":1.,"sector_rel_ret_1":.1,"sector_rel_ret_20":.2}, {"code":"1000","lightgbm_score":1.,"ridge_score":1.,"sector_rel_ret_1":-.1,"sector_rel_ret_20":.1}]
    assert [r["code"] for r in rank_candidates(rows,"LIGHTGBM")] == ["1000","1001"]
    assert random_key(202609220000, date(2020,1,2), "1000") == hashlib.sha256(b"202609220000|2020-01-02|1000").hexdigest()
    assert rank_candidates(rows,"RANDOM_500",date(2020,1,2),202609220000) == rank_candidates(rows,"RANDOM_500",date(2020,1,2),202609220000)
    assert diagnostics([])["ic"] == "UNDEFINED"

def test_aq_key_sets_and_finite_serialization():
    criteria = {key: True for key in "ABCDEFGHIJKLMNOP"}; q = {key: True for key in Q_KEYS}
    assert adjudicate(criteria,q)["V13_VIABILITY_RESULT"] == "CONTINUE_TO_FORMAL_CONFIRMATION_DESIGN"
    q[Q_KEYS[0]] = False
    assert adjudicate(criteria,q)["V13_VIABILITY_RESULT"] == "STOP_HYPOTHESIS_NOT_PROMOTED"
    assert "NaN" not in canonical_json({"x": float("nan")})

def test_single_position_fallback_missing_exit_and_end_boundary():
    sessions = (date(2020,1,2),date(2020,1,3),date(2020,1,6),date(2020,1,7))
    cal = SessionCalendar(sessions)
    ranking = [{"code":"1000","sector":"A"},{"code":"1001","sector":"B"}]
    prices = {("1000",sessions[1]): {"open": 0.}, ("1001",sessions[1]): {"open":100.}, ("1001",sessions[3]): {"close":110.}}
    result = simulate(cal, prices, {sessions[0]:ranking, sessions[3]:ranking})
    assert result["trades"][0]["quantity"] % 100 == 0 and any(e["type"] == "END_OF_STUDY_NO_ENTRY" for e in result["events"])
    assert trade_metrics(result)["closed_trade_count"] == 1
    prices.pop(("1001",sessions[3]))
    assert simulate(cal, prices, {sessions[0]:ranking})["failure_class"] == "DATA_QUALITY_FAILURE"

def test_cli_is_offline_safe_and_byte_deterministic():
    command = [sys.executable, str(ROOT / "scripts" / "run_v13_synthetic_feasibility.py")]
    one, two = subprocess.check_output(command, cwd=ROOT), subprocess.check_output(command, cwd=ROOT)
    assert one == two
    result = json.loads(one)
    assert result["SYNTHETIC_ONLY_NOT_RESEARCH_EVIDENCE"] is True
    assert result["REAL_MARKET_DATA_USED"] is False and result["V13_HISTORICAL_VIABILITY_RESULT"] == "NOT_RUN"
