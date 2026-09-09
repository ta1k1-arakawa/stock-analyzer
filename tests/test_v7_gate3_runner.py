from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_v7_forward_capacity import (
    ROOT,
    run_gate3_dry_run_cli,
    run_static_check,
)


SCRIPT = ROOT / "scripts" / "run_v7_forward_capacity.py"


@pytest.fixture(scope="module")
def gate3_summary():
    return run_gate3_dry_run_cli()


def _subprocess_json(*arguments: str) -> dict:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_gate3_dry_run_verdict_is_pass(gate3_summary):
    assert gate3_summary["verdict"] == "V7_FORWARD_CAPACITY_GATE3_DRY_RUN_PASS"
    assert gate3_summary["mode"] == "DRY_RUN_ONLY"
    assert gate3_summary["activation_status"] == "NOT_ACTIVATED"
    assert gate3_summary["activation_boundary"] == "NOT_SET"


def test_gate3_cases_are_all_passed(gate3_summary):
    assert gate3_summary["case_pass_count"] == 12
    assert gate3_summary["case_fail_count"] == 0
    assert gate3_summary["candidate_generation_count"] == 1
    assert gate3_summary["candidate_parity"] == "PASS"


def test_gate3_hashes_are_actual_four_way_bindings(gate3_summary):
    fields = {
        "seed_canonical_sha256": "seed_hash",
        "price_snapshot_sha256": "price_snapshot_hash",
        "candidate_snapshot_sha256": "candidate_snapshot_hash",
        "market_gate_snapshot_sha256": "market_gate_snapshot_hash",
    }
    for result_field, input_field in fields.items():
        value = gate3_summary[result_field]
        assert len(value) == 64
        assert value == gate3_summary["control_input_hashes"][input_field]
        assert value != "a" * 64
        assert value != "b" * 64


def test_gate3_arm_hashes_are_equal_and_state_hashes_are_real(gate3_summary):
    assert gate3_summary["control_input_hashes"] == gate3_summary["variant_input_hashes"]
    assert gate3_summary["arm_input_hash_equal"] is True
    assert len(gate3_summary["control_state_sha256"]) == 64
    assert len(gate3_summary["variant_state_sha256"]) == 64
    assert gate3_summary["control_state_sha256"] != gate3_summary["variant_state_sha256"]


def test_gate3_parameter_contract_and_state_isolation(gate3_summary):
    assert gate3_summary["control_max_open_positions"] == 2
    assert gate3_summary["variant_max_open_positions"] == 3
    assert gate3_summary["single_changed_parameter"] == "max_open_positions"
    assert gate3_summary["state_objects_independent"] is True


def test_gate3_block_reasons_are_exact(gate3_summary):
    assert gate3_summary["case6_block_reason"] == "OPEN_POSITION_SPLIT_SPANNING"
    assert gate3_summary["case7_block_reason"] == "PLANNED_EXIT_PRICE_UNAVAILABLE"
    assert gate3_summary["case8_block_reason"] == "OPEN_POSITION_MTM_PRICE_UNAVAILABLE"
    assert gate3_summary["rank21_promotion"] == 0


def test_gate3_two_pass_and_boundary_flags(gate3_summary):
    assert gate3_summary["two_pass_byte_identical"] is True
    assert gate3_summary["pre_activation_persisted_study_events"] == 0
    assert gate3_summary["candidate_future_reads"] == 0
    assert gate3_summary["future_split_reads"] == 0


def test_gate3_temporary_summary_is_removed(gate3_summary):
    assert gate3_summary["temporary_output_removed"] is True


def test_gate3_static_check_has_frozen_lineage_and_boundaries():
    result = run_static_check()
    assert result["static_check"] == "PASS"
    assert result["gate3_static_check"] == "PASS"
    assert result["design_commit"] == "e3e1367efd913b601a70328a815d88c20af6d147"
    assert result["latest_preregistration_utc"] == "2026-08-07T02:48:27Z"
    assert result["mode"] == "DRY_RUN_ONLY"
    inspected = set(result["inspected"])
    assert "src/v7_gate3_dry_run.py" in inspected
    assert result["network"] is False
    assert result["activation_created"] is False
    assert result["persistent_study_root_created"] is False
    assert result["real_order_path"] is False


def test_forbidden_activation_option_is_not_accepted():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--gate3-static-check", "--activate"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "unrecognized arguments" in completed.stderr


def test_gate3_dry_run_subprocess_returns_valid_json():
    result = _subprocess_json("--gate3-dry-run")
    assert result["verdict"] == "V7_FORWARD_CAPACITY_GATE3_DRY_RUN_PASS"
    assert result["case_pass_count"] == 12
    assert result["two_pass_byte_identical"] is True


def test_gate3_static_subprocess_returns_valid_json():
    result = _subprocess_json("--gate3-static-check")
    assert result["gate3_static_check"] == "PASS"
    assert result["design_commit"] == "e3e1367efd913b601a70328a815d88c20af6d147"
    assert result["latest_preregistration_utc"] == "2026-08-07T02:48:27Z"
    assert result["mode"] == "DRY_RUN_ONLY"


def test_existing_gate2_synthetic_cli_regression():
    result = _subprocess_json("--synthetic-golden-test")
    assert result["mode"] == "DRY_RUN_ONLY"
    assert result["two_pass_byte_identical"] is True
    assert result["control_filled"] == 2
    assert result["variant_filled"] == 3


def test_existing_gate2_static_cli_regression():
    result = _subprocess_json("--gate2-static-check")
    assert result["static_check"] == "PASS"
    assert result["gate3_static_check"] == "PASS"
    assert result["network"] is False
