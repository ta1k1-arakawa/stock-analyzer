from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src.v7_forward_candidate import generate_forward_candidates_for_day
from src.v7_forward_protocol import validate_seed_rows
from src.v7_gate3_dry_run import canonical_json_bytes, run_gate3_dry_run


COLLECTOR_COMMIT = "b" * 40


@pytest.fixture(scope="module")
def synthetic_inputs():
    calendar = pd.bdate_range("2019-01-02", periods=264)
    engine_day = calendar[252]
    tickers = [f"T{index:03d}" for index in range(300)]
    universe = pd.DataFrame({
        "ticker": tickers,
        "market": ["JP"] * len(tickers),
        "industry": [f"IND{index:03d}" for index in range(len(tickers))],
    })
    frames = {}
    for ticker in tickers:
        close = 1000.0 + np.arange(len(calendar), dtype=float)
        volume = np.full(len(calendar), 100000.0)
        volume[252] = 200000.0
        frames[ticker] = pd.DataFrame({
            "Open": close,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        }, index=calendar)
    seed_rows = []
    for ticker in tickers:
        for index, day in enumerate(calendar[:252]):
            price = 1000.0 + index
            seed_rows.append({
                "ticker": ticker,
                "trading_date": day.strftime("%Y-%m-%d"),
                "raw_open": price,
                "raw_high": price + 2.0,
                "raw_low": price - 2.0,
                "raw_close": price,
                "raw_volume": 100000.0,
            })
    return {
        "calendar": calendar,
        "engine_day": engine_day,
        "universe": universe,
        "frames": frames,
        "split_history": {ticker: set() for ticker in tickers},
        "seed_rows": seed_rows,
    }


@pytest.fixture(scope="module")
def result(synthetic_inputs):
    return run_gate3_dry_run(
        synthetic_inputs["frames"],
        synthetic_inputs["universe"],
        synthetic_inputs["split_history"],
        synthetic_inputs["calendar"],
        synthetic_inputs["engine_day"],
        synthetic_inputs["seed_rows"],
        COLLECTOR_COMMIT,
    )


def test_all_12_cases_pass_with_fixed_ids(result):
    cases = result["case_results"]
    assert [case["case_id"] for case in cases] == list(range(1, 13))
    assert all(case["status"] == "PASS" for case in cases)
    assert result["case_pass_count"] == 12
    assert result["case_fail_count"] == 0


@pytest.mark.parametrize("case_id", range(1, 13))
def test_each_preregistered_case_has_deterministic_schema(result, case_id):
    case = result["case_results"][case_id - 1]
    assert set(case) == {"case_id", "name", "status", "details"}
    assert isinstance(case["details"], dict)


def test_actual_seed_hash_is_bound_from_seed_rows(result, synthetic_inputs):
    expected = validate_seed_rows(
        synthetic_inputs["seed_rows"],
        synthetic_inputs["universe"]["ticker"].tolist(),
        synthetic_inputs["engine_day"].strftime("%Y-%m-%d"),
    )["seed_canonical_sha256"]
    assert result["seed_canonical_sha256"] == expected
    assert result["seed_canonical_sha256"] != "a" * 64


def test_actual_candidate_price_and_market_hashes_are_bound(result, synthetic_inputs):
    d0_frames = {
        ticker: frame.loc[:synthetic_inputs["engine_day"]].copy()
        for ticker, frame in synthetic_inputs["frames"].items()
    }
    candidate = generate_forward_candidates_for_day(
        d0_frames,
        synthetic_inputs["universe"],
        synthetic_inputs["split_history"],
        synthetic_inputs["calendar"],
        synthetic_inputs["engine_day"],
        COLLECTOR_COMMIT,
    )
    assert result["candidate_snapshot_sha256"] == candidate["candidate_snapshot_sha256"]
    assert result["price_snapshot_sha256"] == candidate["price_snapshot_sha256"]
    assert result["market_gate_snapshot_sha256"] == candidate["market_gate_snapshot_sha256"]


def test_actual_hashes_are_equal_between_arms_and_not_caller_injected(result):
    assert result["control_input_hashes"] == result["variant_input_hashes"]
    assert result["arm_input_hash_equal"] is True
    assert set(result["control_input_hashes"]) == {"seed_hash", "price_snapshot_hash", "candidate_snapshot_hash", "market_gate_snapshot_hash"}
    assert all(len(value) == 64 for value in result["control_input_hashes"].values())


def test_single_parameter_and_arm_capacity_are_preserved(result):
    assert result["control_max_open_positions"] == 2
    assert result["variant_max_open_positions"] == 3
    assert result["single_changed_parameter"] == "max_open_positions"


def test_state_objects_are_independent(result):
    assert result["state_objects_independent"] is True
    assert result["control_state_sha256"] != result["variant_state_sha256"]


def test_rank21_is_not_promoted(result):
    case = result["case_results"][8]
    assert case["details"]["rank21_promoted"] == 0


def test_case_block_reasons_are_exact(result):
    assert result["case_results"][5]["details"] == {
        "expected_block_reason": "OPEN_POSITION_SPLIT_SPANNING",
        "observed_block_reason": "OPEN_POSITION_SPLIT_SPANNING",
    }
    assert result["case_results"][6]["details"] == {
        "expected_block_reason": "PLANNED_EXIT_PRICE_UNAVAILABLE",
        "observed_block_reason": "PLANNED_EXIT_PRICE_UNAVAILABLE",
    }
    assert result["case_results"][7]["details"] == {
        "expected_block_reason": "OPEN_POSITION_MTM_PRICE_UNAVAILABLE",
        "observed_block_reason": "OPEN_POSITION_MTM_PRICE_UNAVAILABLE",
    }


def test_activation_and_preactivation_boundaries_are_not_created(result):
    assert result["mode"] == "DRY_RUN_ONLY"
    assert result["activation_status"] == "NOT_ACTIVATED"
    assert result["activation_boundary"] == "NOT_SET"
    assert result["persistent_study_root_created"] is False
    assert result["pre_activation_persisted_study_events"] == 0


def test_enriched_audit_has_required_fields_and_all_boundary_event_types(result):
    required = {"arm", "engine_day", "ticker", "candidate_snapshot_sha256", "price_snapshot_sha256", "reason", "collector_commit"}
    assert result["enriched_event_audit"]
    assert all(required <= set(event) for event in result["enriched_event_audit"])
    event_types = {event["event"] for event in result["enriched_event_audit"]}
    assert {
        "D0_MARKET_GATE_COMPUTED", "D0_TOP20_FROZEN", "ORDER_QUEUED", "ENTRY_FILLED",
        "ENTRY_SKIPPED_DATA_UNAVAILABLE", "ENTRY_SKIPPED_SPLIT",
        "OPEN_POSITION_SPLIT_DETECTED", "D10_EXIT_BLOCKED_MISSING_PRICE",
        "MTM_BLOCKED_MISSING_PRICE",
    } <= event_types


def test_two_pass_dry_run_result_is_canonical_bytes_identical(synthetic_inputs):
    first = run_gate3_dry_run(
        synthetic_inputs["frames"], synthetic_inputs["universe"], synthetic_inputs["split_history"],
        synthetic_inputs["calendar"], synthetic_inputs["engine_day"], synthetic_inputs["seed_rows"], COLLECTOR_COMMIT,
    )
    second = run_gate3_dry_run(
        synthetic_inputs["frames"], synthetic_inputs["universe"], synthetic_inputs["split_history"],
        synthetic_inputs["calendar"], synthetic_inputs["engine_day"], synthetic_inputs["seed_rows"], COLLECTOR_COMMIT,
    )
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_no_persistent_or_real_data_activity_flags(result):
    assert result["network_requests"] == 0
    assert result["seed_acquisition"] == 0
    assert result["real_data_read"] == 0
    assert result["historical_replay"] == 0
    assert result["real_portfolio_simulation"] == 0
