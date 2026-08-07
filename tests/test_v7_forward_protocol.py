from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from scripts.run_v7_forward_capacity import synthetic_forward_fixture
from src.v7_capacity_engine import V7EngineParameters
from src.v7_forward_protocol import (
    ACTIVATION_MANIFEST_FIELDS,
    ArmInputHashes,
    CheckpointWriter,
    ProtocolBlocked,
    create_dual_arm_study,
    validate_activation_manifest,
    validate_seed_rows,
)
from scripts.run_v7_forward_capacity import run_static_check, run_synthetic_golden


def seed_rows(ticker: str = "AAA", count: int = 252) -> list[dict]:
    start = date(2018, 1, 2)
    return [
        {
            "ticker": ticker,
            "trading_date": (start + timedelta(days=index)).isoformat(),
            "raw_open": 100.0,
            "raw_high": 101.0,
            "raw_low": 99.0,
            "raw_close": 100.0,
            "raw_volume": 1000,
        }
        for index in range(count)
    ]


def full_seed_result():
    rows = seed_rows("AAA") + seed_rows("BBB")
    return validate_seed_rows(rows, ("AAA", "BBB"), "2019-01-01")


def manifest_for(seed_result):
    control = V7EngineParameters.control()
    variant = V7EngineParameters.capacity_3()
    return {
        "mode": "DRY_RUN_ONLY",
        "design_commit": "e3e1367efd913b601a70328a815d88c20af6d147",
        "implementation_commit": "1" * 40,
        "collector_commit": "2" * 40,
        "activation_authorization_utc": "2026-08-07T04:00:00Z",
        "activation_boundary_first_jpx_trading_date": "2026-08-10",
        "calendar_source": "synthetic",
        "calendar_version": "synthetic-v1",
        "calendar_timezone": "Asia/Tokyo",
        "data_source": "synthetic",
        "data_source_schema": "ohlcv-v1",
        "acquisition_window_jst": "15:30-16:00",
        "universe_csv_sha": "a" * 64,
        "ticker_list_sha": "b" * 64,
        "arm_a_parameters_sha256": control.sha256(),
        "arm_b_parameters_sha256": variant.sha256(),
        "shared_rules_sha256": "c" * 64,
        "output_root": "synthetic-only",
        "seed_data_source": "synthetic",
        "seed_data_schema": "seed-v1",
        "seed_acquisition_utc": "2026-08-07T03:00:00Z",
        "seed_cutoff_trading_date": seed_result["seed_cutoff_trading_date"],
        "seed_ticker_count": seed_result["ticker_count"],
        "seed_row_count": seed_result["row_count"],
        "seed_payload_manifest_sha256": seed_result["seed_payload_manifest_sha256"],
        "seed_canonical_csv_sha256": seed_result["seed_canonical_sha256"],
        "seed_generation_commit": "3" * 40,
        "seed_validation_result": "PASS",
        "arm_seed_hash_equal": True,
        "arm_candidate_input_hash_equal": True,
        "arm_market_gate_input_hash_equal": True,
    }


def checkpoint_values(previous=None, day="2020-01-02"):
    return {
        "previous_checkpoint_sha256": previous,
        "last_completed_engine_day": day,
        "arm_a_state_sha256": "a" * 64,
        "arm_b_state_sha256": "b" * 64,
        "candidate_snapshot_sha256": "c" * 64,
        "price_snapshot_sha256": "d" * 64,
        "collector_commit": "e" * 40,
        "status": "COMPLETE",
    }


def test_seed_252_eligible_and_251_ineligible():
    eligible = validate_seed_rows(seed_rows(), ("AAA",), "2019-01-01")
    assert eligible["row_count"] == 252
    assert eligible["eligible_ticker_count"] == 1
    assert eligible["ineligible_ticker_count"] == 0
    ineligible = validate_seed_rows(seed_rows(count=251), ("AAA",), "2019-01-01")
    assert ineligible["eligible_ticker_count"] == 0
    assert ineligible["ineligible_ticker_count"] == 1
    assert ineligible["ticker_manifest"][0]["eligibility_at_activation"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows + [{**rows[0], "trading_date": "2019-01-01"}],
        lambda rows: rows + [{**rows[0], "trading_date": "2019-01-02"}],
        lambda rows: rows + [{**rows[0]}],
        lambda rows: [{**rows[0], "raw_close": float("nan")}] + rows[1:],
        lambda rows: [{**rows[0], "raw_close": 0.0}] + rows[1:],
    ],
)
def test_seed_rejects_activation_duplicate_nonfinite_and_nonpositive_rows(mutate):
    rows = seed_rows()
    with pytest.raises(ProtocolBlocked):
        validate_seed_rows(mutate(rows), ("AAA",), "2019-01-01")


def test_seed_rejects_ticker_outside_fixed_universe():
    rows = seed_rows()
    rows[0]["ticker"] = "OUTSIDE"
    with pytest.raises(ProtocolBlocked, match="OUTSIDE_FIXED_UNIVERSE"):
        validate_seed_rows(rows, ("AAA",), "2019-01-01")


def test_seed_hash_is_deterministic_under_ticker_order_change():
    rows = seed_rows("AAA") + seed_rows("BBB")
    first = validate_seed_rows(rows, ("AAA", "BBB"), "2019-01-01")
    second = validate_seed_rows(list(reversed(rows)), ("BBB", "AAA"), "2019-01-01")
    assert first["seed_canonical_sha256"] == second["seed_canonical_sha256"]
    assert first["seed_payload_manifest_sha256"] == second["seed_payload_manifest_sha256"]


def test_seed_validator_has_zero_study_events_and_required_result_fields():
    result = full_seed_result()
    assert result.get("pre_activation_study_events", 0) == 0
    assert {
        "ticker_count",
        "row_count",
        "eligible_ticker_count",
        "ineligible_ticker_count",
        "seed_cutoff_trading_date",
        "ticker_manifest",
        "seed_canonical_sha256",
        "seed_payload_manifest_sha256",
    } <= set(result)


def test_manifest_passes_only_dry_run_and_all_required_fields():
    seed_result = full_seed_result()
    manifest = manifest_for(seed_result)
    assert set(manifest) == ACTIVATION_MANIFEST_FIELDS
    assert validate_activation_manifest(
        manifest,
        control=V7EngineParameters.control(),
        variant=V7EngineParameters.capacity_3(),
        seed_validation=seed_result,
    )["mode"] == "DRY_RUN_ONLY"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: {**m, "unknown": 1},
        lambda m: {key: value for key, value in m.items() if key != "data_source"},
        lambda m: {**m, "mode": "ACTIVE"},
        lambda m: {**m, "universe_csv_sha": "bad"},
        lambda m: {**m, "calendar_timezone": "UTC"},
        lambda m: {**m, "design_commit": "f" * 40},
        lambda m: {**m, "arm_seed_hash_equal": False},
        lambda m: {**m, "activation_boundary_first_jpx_trading_date": "2018-01-01"},
    ],
)
def test_manifest_fail_closed(mutate):
    seed_result = full_seed_result()
    with pytest.raises(ProtocolBlocked):
        validate_activation_manifest(
            mutate(manifest_for(seed_result)),
            control=V7EngineParameters.control(),
            variant=V7EngineParameters.capacity_3(),
            seed_validation=seed_result,
        )


def test_manifest_rejects_parameter_difference_and_seed_hash_mismatch():
    seed_result = full_seed_result()
    manifest = manifest_for(seed_result)
    with pytest.raises(ValueError, match="SINGLE_PARAMETER"):
        validate_activation_manifest(
            manifest,
            control=V7EngineParameters.control(),
            variant=V7EngineParameters( max_open_positions=3, entry_slippage=0.001),
            seed_validation=seed_result,
        )
    bad = {**manifest, "seed_canonical_csv_sha256": "d" * 64}
    with pytest.raises(ProtocolBlocked, match="SEED_HASH"):
        validate_activation_manifest(bad, seed_validation=seed_result)


def test_manifest_requires_strict_utc_preregistration_seed_activation_order():
    seed_result = full_seed_result()
    base = manifest_for(seed_result)
    assert validate_activation_manifest(base, seed_validation=seed_result)["status"] == "PASS"
    with pytest.raises(ProtocolBlocked, match="PREREGISTRATION"):
        validate_activation_manifest(
            {**base, "seed_acquisition_utc": "2026-08-07T01:43:27Z"},
            seed_validation=seed_result,
        )
    with pytest.raises(ProtocolBlocked, match="ACTIVATION_TIME_ORDER"):
        validate_activation_manifest(
            {**base, "activation_authorization_utc": "2026-08-07T02:00:00Z"},
            seed_validation=seed_result,
        )
    with pytest.raises(ProtocolBlocked, match="AWARE_UTC"):
        validate_activation_manifest(
            {**base, "seed_acquisition_utc": "2026-08-07T02:00:00"},
            seed_validation=seed_result,
        )
    with pytest.raises(ProtocolBlocked, match="AWARE_UTC"):
        validate_activation_manifest(
            {**base, "seed_acquisition_utc": "2026-08-07T11:00:00+09:00"},
            seed_validation=seed_result,
        )


def test_manifest_requires_validated_seed_cutoff_equality():
    seed_result = full_seed_result()
    manifest = {**manifest_for(seed_result), "seed_cutoff_trading_date": "2018-09-01"}
    with pytest.raises(ProtocolBlocked, match="SEED_CUTOFF_MISMATCH"):
        validate_activation_manifest(manifest, seed_validation=seed_result)


def test_dual_arm_state_identity_mutation_isolation_and_hash_guards():
    calendar, frames, candidates = synthetic_forward_fixture()
    hashes = ArmInputHashes("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    study = create_dual_arm_study(frames, calendar, candidates, hashes, hashes).run()
    assert study.state_objects_are_independent()
    original_variant_cash = study.variant.state.available_cash
    study.control.state.available_cash -= 1
    assert study.variant.state.available_cash == original_variant_cash
    with pytest.raises(ProtocolBlocked, match="ARM_INPUT_HASH"):
        create_dual_arm_study(frames, calendar, candidates, hashes, ArmInputHashes("e" * 64, "b" * 64, "c" * 64, "d" * 64))
    from src.v7_capacity_engine import CausalEventEngine
    from src.v7_forward_protocol import DualArmStudy
    shared = CausalEventEngine(frames, calendar, candidates, V7EngineParameters.control())
    with pytest.raises(ProtocolBlocked, match="CROSS_ARM"):
        DualArmStudy(shared, shared, hashes, hashes)


def test_dual_arm_split_mapping_is_deepcopied_and_default_is_compatible():
    calendar, frames, candidates = synthetic_forward_fixture()
    hashes = ArmInputHashes("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    split_events = {calendar[1]: ["AAA"]}
    study = create_dual_arm_study(
        frames, calendar, candidates, hashes, hashes,
        split_events_by_day=split_events,
    )
    assert study.control.split_events_by_day == split_events
    assert study.variant.split_events_by_day == split_events
    assert study.control.split_events_by_day is not study.variant.split_events_by_day
    study.control.split_events_by_day[calendar[1]].append("BBB")
    assert study.variant.split_events_by_day[calendar[1]] == ["AAA"]

    default_study = create_dual_arm_study(frames, calendar, candidates, hashes, hashes)
    assert default_study.control.split_events_by_day == {}
    assert default_study.variant.split_events_by_day == {}


def test_dual_arm_split_mapping_is_independent_during_execution():
    calendar, frames, candidates = synthetic_forward_fixture()
    hashes = ArmInputHashes("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    split_events = {calendar[1]: ["AAA"]}
    study = create_dual_arm_study(
        frames, calendar, candidates, hashes, hashes,
        split_events_by_day=split_events,
    )
    assert study.control.split_events_by_day is not split_events
    assert study.variant.split_events_by_day is not split_events
    study.control.split_events_by_day[calendar[1]][0] = "MUTATED"
    assert study.variant.split_events_by_day[calendar[1]] == ["AAA"]


def test_capacity_scenario_uses_equal_inputs_and_only_parameter_difference_in_audit():
    calendar, frames, candidates = synthetic_forward_fixture()
    hashes = ArmInputHashes("a" * 64, "b" * 64, "c" * 64, "d" * 64)
    study = create_dual_arm_study(frames, calendar, candidates, hashes, hashes).run()
    assert study.control_input_hashes == study.variant_input_hashes
    assert study.control.parameters.to_dict() | {"max_open_positions": 3} == study.variant.parameters.to_dict()
    control_queued = [event for event in study.control.state.event_audit if event["event"] == "ORDER_QUEUED"]
    variant_queued = [event for event in study.variant.state.event_audit if event["event"] == "ORDER_QUEUED"]
    assert control_queued == variant_queued
    assert study.control.skip_reason_counts()["MAX_OPEN_POSITIONS"] == 1
    assert study.variant.skip_reason_counts().get("MAX_OPEN_POSITIONS", 0) == 0


def test_checkpoint_first_second_chain_restart_and_duplicate_guard(tmp_path):
    writer = CheckpointWriter(tmp_path)
    first = writer.write_complete(**checkpoint_values())
    second = writer.write_complete(**checkpoint_values(first["current_checkpoint_sha256"], "2020-01-03"))
    assert second["previous_checkpoint_sha256"] == first["current_checkpoint_sha256"]
    assert writer.load_last_complete()["current_checkpoint_sha256"] == second["current_checkpoint_sha256"]
    assert writer.restart_from_last_checkpoint()["last_completed_engine_day"] == "2020-01-03"
    with pytest.raises(ProtocolBlocked, match="DUPLICATE_ENGINE_DAY"):
        writer.write_complete(**checkpoint_values(second["current_checkpoint_sha256"], "2020-01-03"))


def test_checkpoint_requires_strictly_increasing_day_and_no_new_file_on_failure(tmp_path):
    writer = CheckpointWriter(tmp_path)
    first = writer.write_complete(**checkpoint_values())
    before = sorted(Path(tmp_path).glob("checkpoint-*.json"))
    with pytest.raises(ProtocolBlocked, match="ENGINE_DAY_NOT_INCREASING"):
        writer.write_complete(**checkpoint_values(first["current_checkpoint_sha256"], "2020-01-01"))
    assert sorted(Path(tmp_path).glob("checkpoint-*.json")) == before


@pytest.mark.parametrize(
    "field,value,pattern",
    [
        ("arm_a_state_sha256", "bad", "SHA_INVALID"),
        ("candidate_snapshot_sha256", "bad", "SHA_INVALID"),
        ("collector_commit", "bad", "COMMIT_INVALID"),
    ],
)
def test_checkpoint_field_validation_fails_before_final_write(tmp_path, field, value, pattern):
    writer = CheckpointWriter(tmp_path)
    values = checkpoint_values()
    values[field] = value
    with pytest.raises(ProtocolBlocked, match=pattern):
        writer.write_complete(**values)
    assert not list(Path(tmp_path).glob("checkpoint-*.json"))


def test_checkpoint_filename_date_mismatch_is_rejected_on_read(tmp_path):
    writer = CheckpointWriter(tmp_path)
    first = writer.write_complete(**checkpoint_values())
    path = Path(tmp_path) / "checkpoint-2020-01-02.json"
    renamed = Path(tmp_path) / "checkpoint-2020-01-03.json"
    path.rename(renamed)
    with pytest.raises(ProtocolBlocked, match="FILENAME_DATE_MISMATCH"):
        writer.load_last_complete()


def test_checkpoint_three_day_chain_and_tampered_previous_hash_are_blocked(tmp_path):
    writer = CheckpointWriter(tmp_path)
    first = writer.write_complete(**checkpoint_values())
    second = writer.write_complete(**checkpoint_values(first["current_checkpoint_sha256"], "2020-01-03"))
    third = writer.write_complete(**checkpoint_values(second["current_checkpoint_sha256"], "2020-01-04"))
    assert writer.restart_from_last_checkpoint()["current_checkpoint_sha256"] == third["current_checkpoint_sha256"]
    path = Path(tmp_path) / "checkpoint-2020-01-04.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["previous_checkpoint_sha256"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolBlocked, match="CHECKPOINT_HASH_MISMATCH"):
        writer.load_last_complete()


def test_checkpoint_previous_hash_mismatch_and_partial_staging_fail_closed(tmp_path):
    writer = CheckpointWriter(tmp_path)
    first = writer.write_complete(**checkpoint_values())
    with pytest.raises(ProtocolBlocked, match="PREVIOUS_HASH"):
        writer.write_complete(**checkpoint_values("f" * 64, "2020-01-03"))
    (Path(tmp_path) / "orphan.staging-2020-01-04").write_text("partial", encoding="utf-8")
    with pytest.raises(ProtocolBlocked, match="PARTIAL_CHECKPOINT"):
        writer.write_complete(**checkpoint_values(first["current_checkpoint_sha256"], "2020-01-04"))


def test_checkpoint_tamper_and_partial_final_are_not_read_as_complete(tmp_path):
    writer = CheckpointWriter(tmp_path)
    first = writer.write_complete(**checkpoint_values())
    path = Path(tmp_path) / "checkpoint-2020-01-02.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "PARTIAL"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProtocolBlocked):
        writer.load_last_complete()


def test_synthetic_golden_and_static_check_are_deterministic_and_offline():
    first = run_synthetic_golden()
    second = run_synthetic_golden()
    assert first == second
    assert first["seed_study_event_count"] == 0
    assert run_static_check()["static_check"] == "PASS"
