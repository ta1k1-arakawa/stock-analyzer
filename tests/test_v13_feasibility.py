from datetime import date
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.v13_feasibility import (FEATURES, Q_KEYS, SessionCalendar, adjudicate, base_target,
    build_rank_population, build_rank_population_audit, canonical_json, diagnostics, lightgbm_factory, monthly_predictions,
    random_key, rank_candidates, ridge_factory, select_universe, simulate, trade_metrics,
    linear_percentile,
    stage_a_from_raw, monthly_training_rows, verify_frozen_bindings, _derive_criteria)
from src.v13_synthetic_fixture import (active_rows, raw_ohlcv, stage_a_rows, synthetic_calendar,
    synthetic_manifest, synthetic_metadata)

ROOT = Path(__file__).resolve().parents[1]

def test_frozen_bindings_and_manifest_are_deterministic():
    assert subprocess.check_output(["git", "rev-parse", "61268237494e2968562983e456ea40e6f821d066:V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md"], cwd=ROOT, text=True).strip() == "3bfcd695c69f6dac480f8fc99ca4f3916f668e4a"
    assert synthetic_manifest() == synthetic_manifest()
    assert len(synthetic_manifest()["selected"]) == 500

def test_synthetic_manifest_uses_exact_frozen_seed_and_active_subset():
    manifest = synthetic_manifest()
    metadata = synthetic_metadata()
    assert manifest["seed"] == "V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON|f9c38ad771710ffd157ac4fad0da15185db82707"
    assert len(manifest["selected"]) == 500
    assert set(metadata) <= set(manifest["selected"])
    assert set(metadata).isdisjoint(manifest["excluded"])
    assert len(metadata) == 12 and list(metadata.values()).count("SYNTHETIC_ALPHA") == 6
    assert list(metadata.values()).count("SYNTHETIC_BETA") == 6

def test_synthetic_selected_and_active_subset_hashes_are_deterministic():
    first_manifest, second_manifest = synthetic_manifest(), synthetic_manifest()
    first_active = hashlib.sha256("|".join(sorted(synthetic_metadata())).encode()).hexdigest()
    second_active = hashlib.sha256("|".join(sorted(synthetic_metadata())).encode()).hexdigest()
    assert first_manifest["selected_sha256"] == second_manifest["selected_sha256"]
    assert first_active == second_active

def test_synthetic_raw_codes_equal_active_metadata_codes():
    assert {code for code, _ in raw_ohlcv()} == set(synthetic_metadata())

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
    # The production implementation exposes the actual training identities and
    # cutoff facts; the assertion is behavioral rather than source inspection.
    audit = {}
    _, audited = monthly_predictions(active_rows(), audit=audit)
    assert all(all(exit_day < record["prediction_month_first_session"]
                   for exit_day in record["training_exit_dates"])
               for record in audited.values())

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

@pytest.mark.parametrize("field", ["ret_1","ret_3","ret_5","ret_20","intraday_1","overnight_1","log_traded_value_ratio_20","log_median_traded_value_20","log_amihud_20","volatility_20","dist_52w_high"])
def test_stage_a_has_only_native_raw_derived_fields(field):
    from src.v13_synthetic_fixture import raw_ohlcv, synthetic_calendar, synthetic_metadata
    rows=stage_a_from_raw(synthetic_calendar(),raw_ohlcv(),synthetic_metadata(),date(2020,1,2))
    assert rows and field in rows[0]
    assert not any(k.startswith("sector_rel_") or k.startswith("market_") for k in rows[0])

@pytest.mark.parametrize("strategy", ["LIGHTGBM","RIDGE","SECTOR_REL_REVERSAL_1D","SECTOR_REL_MOMENTUM_20D","RANDOM_500"])
def test_comparator_ranking_is_deterministic(strategy):
    rows=[{"code":"1000","lightgbm_score":1.,"ridge_score":2.,"sector_rel_ret_1":-.1,"sector_rel_ret_20":.1},{"code":"1001","lightgbm_score":2.,"ridge_score":1.,"sector_rel_ret_1":.1,"sector_rel_ret_20":-.1}]
    args=(date(2020,1,2),202609220000) if strategy=="RANDOM_500" else (None,None)
    assert rank_candidates(rows,strategy,*args)==rank_candidates(rows,strategy,*args)

def test_real_training_identities_are_before_month_cutoff():
    rows=active_rows(); selected=monthly_training_rows(rows,"2020-01")
    assert selected and all(r["exit"]<date(2020,1,1) and r["signal"].year>=2016 for r in selected)

def test_frozen_binding_contract_shape():
    result=verify_frozen_bindings()
    assert set(result)=={"base_design","base_approval","amendment","amendment_approval"}

@pytest.mark.parametrize("variant", ["affordability","missing_open","missing_exit","no_rank"])
def test_isolated_fixture_variants_are_deterministic(variant):
    from src.v13_synthetic_fixture import raw_ohlcv
    a,b=raw_ohlcv(variant),raw_ohlcv(variant)
    assert set(a)==set(b)
    first=next(iter(a)); assert a[first].keys()==b[first].keys()
    for key in a[first]:
        if isinstance(a[first][key],float) and np.isnan(a[first][key]): assert np.isnan(b[first][key])
        else: assert a[first][key]==b[first][key]

def test_real_dependency_identity_and_exact_feature_contract():
    import lightgbm, scipy, sklearn
    from src.v13_feasibility import MARKET_FEATURES, RELATIVE_FEATURES, STAGE_A_FEATURES
    assert lightgbm_factory().__class__.__module__.startswith("lightgbm")
    assert scipy.__version__ and sklearn.__version__
    assert len(STAGE_A_FEATURES) == 11 and len(RELATIVE_FEATURES) == 8 and len(FEATURES) == 25
    assert FEATURES == STAGE_A_FEATURES + RELATIVE_FEATURES + MARKET_FEATURES

def test_stage_c_transform_audit_has_one_pass_for_each_stock_field():
    from src.v13_feasibility import build_rank_population_audit
    status, rows, audit = build_rank_population_audit(stage_a_rows())
    assert status == "OK" and len(rows) == 12
    assert tuple(audit["transforms"]) == tuple(STOCK for STOCK in FEATURES[:19])
    assert len(audit["transforms"]) == 19

def test_stage_b_audit_has_independent_matching_pre_post_identity_hashes():
    status, rows, audit = build_rank_population_audit(stage_a_rows())
    assert status == "OK" and len(rows) == 12
    assert audit["reference_hash"] != audit["stage_b_hash"]
    assert audit["stage_b_hash"] == audit["stage_b_hash_after"]
    assert audit["stage_b_identity"] == audit["stage_b_identity_after"]
    assert audit["stage_b_recomputation_count"] == 0
    assert audit["stage_b_recomputation_event"] is False
    assert audit["rank_eligible_hash"] == hashlib.sha256(canonical_json(audit["rank_eligible_identity"]).encode()).hexdigest()

def test_stage_c_omission_does_not_recompute_stage_b_reference():
    source = stage_a_rows()
    omitted = source[0]["code"]
    status, rows, audit = build_rank_population_audit(source, stage_c_omit_codes={omitted})
    assert status == "OK" and len(rows) == len(source) - 1
    assert omitted not in {row[0] for row in audit["rank_eligible_identity"]}
    assert audit["stage_b_hash"] == audit["stage_b_hash_after"]
    assert audit["stage_b_recomputation_count"] == 0

def test_stage_a_rejects_invalid_signal_divisor_without_imputation():
    from src.v13_synthetic_fixture import raw_ohlcv, synthetic_calendar, synthetic_metadata
    prices = raw_ohlcv(); signal = date(2020, 1, 2); code = next(iter(synthetic_metadata()))
    prices[code, signal]["adj_open"] = 0.
    assert not any(row["code"] == code for row in stage_a_from_raw(synthetic_calendar(), prices, synthetic_metadata(), signal))

def test_stage_a_uses_valid_observations_through_recent_calendar_gap():
    calendar, prices, metadata = synthetic_calendar(), raw_ohlcv(), synthetic_metadata()
    signal, code = date(2020, 1, 2), next(iter(metadata))
    signal_index = calendar.sessions.index(signal)
    prices.pop((code, calendar.sessions[signal_index - 10]))
    row = next(row for row in stage_a_from_raw(calendar, prices, metadata, signal) if row["code"] == code)
    valid = [prices[(code, day)] for day in calendar.sessions[:signal_index + 1] if (code, day) in prices]
    traded = [item["close"] * item["volume"] for item in valid[-20:]]
    closes = [item["adj_close"] for item in valid]
    returns_1 = np.asarray([closes[i] / closes[i - 1] - 1 for i in range(len(closes) - 20, len(closes))])
    assert row["ret_3"] == pytest.approx(closes[-1] / closes[-4] - 1)
    assert row["log_median_traded_value_20"] == pytest.approx(np.log(np.median(traded)))
    assert row["volatility_20"] == pytest.approx(np.std(returns_1, ddof=1))
    assert row["log_amihud_20"] == pytest.approx(np.log(np.mean(np.abs(returns_1) / np.maximum(traded, 1.0))))

def test_stage_a_invalid_signal_observation_excludes_ticker():
    calendar, prices, metadata = synthetic_calendar(), raw_ohlcv(), synthetic_metadata()
    signal, code = date(2020, 1, 2), next(iter(metadata))
    prices[code, signal]["adj_close"] = float("nan")
    assert code not in {row["code"] for row in stage_a_from_raw(calendar, prices, metadata, signal)}

def test_stage_a_fewer_than_253_valid_observations_excludes_ticker():
    calendar, prices, metadata = synthetic_calendar(), raw_ohlcv(), synthetic_metadata()
    signal, code = date(2020, 1, 2), next(iter(metadata))
    prior_days = [day for day in calendar.sessions if day < signal]
    for day in prior_days[:-251]:
        prices.pop((code, day))
    assert code not in {row["code"] for row in stage_a_from_raw(calendar, prices, metadata, signal)}

def test_scipy_spearman_tie_semantics_are_used():
    rows = [{"code": f"{1000+i:04d}", "signal": date(2020,1,2), "lightgbm_score": score, "target": target} for i, (score, target) in enumerate([(1., 1.), (1., 2.), (2., 3.)])]
    result = diagnostics(rows)
    from scipy.stats import spearmanr
    assert result["ic"] == pytest.approx(float(spearmanr([1., 1., 2.], [1., 2., 3.]).statistic))

def test_training_audit_contains_actual_row_identities_and_month_cutoff():
    audit = {}
    _, fits = monthly_predictions(active_rows(), audit=audit)
    assert fits and audit["monthly_fits"] is fits
    for month, record in fits.items():
        assert record["fit_count"] == 2
        assert all(exit_day < record["prediction_month_first_session"] for exit_day in record["training_exit_dates"])
        assert all("|" in row_id for row_id in record["training_row_ids"])

def test_fixture_has_no_2026_session_or_price():
    from src.v13_synthetic_fixture import raw_ohlcv, synthetic_calendar
    assert synthetic_calendar().sessions[-1] == date(2025, 12, 31)
    assert all(day.year <= 2025 for _, day in raw_ohlcv())

def test_all_comparator_sign_and_tie_rules():
    rows = [{"code":"1000", "lightgbm_score":1., "ridge_score":-1., "sector_rel_ret_1":0., "sector_rel_ret_20":0.}, {"code":"1001", "lightgbm_score":2., "ridge_score":1., "sector_rel_ret_1":-.1, "sector_rel_ret_20":.2}]
    assert [r["code"] for r in rank_candidates(rows, "RIDGE")] == ["1001"]
    assert [r["code"] for r in rank_candidates(rows, "SECTOR_REL_REVERSAL_1D")] == ["1001", "1000"]
    assert [r["code"] for r in rank_candidates(rows, "SECTOR_REL_MOMENTUM_20D")] == ["1001", "1000"]

def test_linear_p95_matches_numpy_interpolation():
    assert linear_percentile([0., 1., 2., 3., 4.], 95) == pytest.approx(3.8)

def test_cash_comparator_contract_from_production_run():
    from src.v13_feasibility import run_synthetic_feasibility
    result = run_synthetic_feasibility()
    cash = result["strategies"]["CASH"]["base"]
    assert cash["trades"] == 0 and cash["total_net_profit"] == 0 and cash["ending_equity"] == 300000
    assert set(result["Q"]) == set(Q_KEYS)

def test_production_q4_is_derived_from_stage_b_audit_facts():
    from src.v13_feasibility import run_synthetic_feasibility
    result = run_synthetic_feasibility()
    audits = result["audit"]["stage_dates"].values()
    assert all(a["stage_b_hash"] == a["stage_b_hash_after"] for a in audits)
    assert all(a["stage_b_recomputation_count"] == 0 for a in audits)
    assert all(a["rank_eligible_hash"] for a in audits)
    assert result["Q"][Q_KEYS[3]] is True

def test_production_result_has_actual_a_to_p_and_q_without_research_verdict():
    from src.v13_feasibility import run_synthetic_feasibility
    result = run_synthetic_feasibility()
    assert set(result["A_P"]) == set("ABCDEFGHIJKLMNOP")
    assert set(result["Q"]) == set(Q_KEYS)
    assert result["V13_HISTORICAL_VIABILITY_RESULT"] == "NOT_RUN"

def test_random_500_production_identity_is_proven_for_all_seeds():
    from src.v13_feasibility import run_synthetic_feasibility
    result = run_synthetic_feasibility()
    identity = result["RANDOM_500"]["all_seed_ranking_identity"]
    assert len(identity) == 500 and all(identity.values())

def test_criterion_j_counts_positive_annual_mean_ic_at_zero_point_one_threshold():
    metrics = {"total_net_profit": 1., "positive_year_count": 4, "max_drawdown_pct": 0., "trades": 150, "best_year_positive_profit_share": 0., "max_ticker_positive_profit_share": 0., "max_sector_positive_profit_share": 0.}
    diagnostics_with_small_positive_years = {"mean_ic": .02, "per_year_mean_ic": {str(year): value for year, value in zip(range(2020, 2026), (.001, .002, .003, .004, 0., -.001))}, "top_decile_spread": 1., "positive_spread_year_count": 4}
    criteria = _derive_criteria(metrics, {"total_net_profit": 0.}, {"total_net_profit": 0.}, {"total_net_profit": 0.}, metrics, 0., diagnostics_with_small_positive_years)
    assert criteria["I"] is True and criteria["J"] is True

def test_criterion_i_retains_global_strictly_greater_than_point_zero_one():
    metrics = {"total_net_profit": 1., "positive_year_count": 4, "max_drawdown_pct": 0., "trades": 150, "best_year_positive_profit_share": 0., "max_ticker_positive_profit_share": 0., "max_sector_positive_profit_share": 0.}
    diagnostics_with_small_global_ic = {"mean_ic": .01, "per_year_mean_ic": {str(year): .001 for year in range(2020, 2026)}, "top_decile_spread": 1., "positive_spread_year_count": 4}
    criteria = _derive_criteria(metrics, {"total_net_profit": 0.}, {"total_net_profit": 0.}, {"total_net_profit": 0.}, metrics, 0., diagnostics_with_small_global_ic)
    assert criteria["I"] is False and criteria["J"] is True
