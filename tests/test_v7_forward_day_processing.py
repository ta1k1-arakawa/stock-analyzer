from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import pytest

from scripts.check_v7_forward_day_processing import (
    ACTIVATION_BOUNDARY,
    ACTIVATION_MANIFEST_SHA256,
    COLLECTOR_COMMIT,
    IMPLEMENTATION_COMMIT,
    UNIVERSE_CSV,
    build_acquisition_bundle,
    build_activation_context,
    engine_days_from_boundary,
    seed_trading_days,
    synthetic_calendar_snapshot,
    synthetic_seed_rows,
    universe_tickers,
)
from src import v7_forward_day_processing as processing
from src.v7_forward_persistence import ForwardStudyStore, canonical_json_bytes, canonical_sha256
from src.v7_jpx_calendar import load_calendar_snapshot

TICKERS = universe_tickers()
SNAPSHOT = synthetic_calendar_snapshot()
SEED_DAYS = seed_trading_days(SNAPSHOT, 3)
ENGINE_DAYS = engine_days_from_boundary(SNAPSHOT, 40)
BASE_PRICE = 1000.0
CANDIDATE_INDUSTRIES = ("SYN_TECH", "SYN_FINANCE", "SYN_ENERGY", "SYN_RETAIL")


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


# ---------------------------------------------------------------------------
# Deterministic fake candidate generation (real generator covered in runner tests)
# ---------------------------------------------------------------------------


def day_price(day: str) -> float:
    return BASE_PRICE + float(len(SEED_DAYS) + ENGINE_DAYS.index(day))


def fake_candidate_result(engine_day: str, *, candidate_count: int = 3) -> dict[str, Any]:
    index = ENGINE_DAYS.index(engine_day)
    entry_date = ENGINE_DAYS[index + 1]
    exit_date = ENGINE_DAYS[index + 10]
    price = day_price(engine_day)
    accepted = [
        {
            "candidate_status": "ACCEPTED_TOP20",
            "engine_day": engine_day,
            "signal_date": engine_day,
            "signal_year": int(engine_day[:4]),
            "ticker": TICKERS[position],
            "industry": CANDIDATE_INDUSTRIES[position],
            "rank": position + 1,
            "raw_close": price,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "collector_commit": COLLECTOR_COMMIT,
        }
        for position in range(candidate_count)
    ]
    market_gate = {
        "engine_day": engine_day,
        "market_gate_status": "MARKET_GATE_PASS",
        "market_denominator_count": len(TICKERS),
        "breadth_above_ma60": 1.0,
        "cross_sectional_median_return20": 0.01,
    }
    return {
        "engine_day": engine_day,
        "market_gate": market_gate,
        "accepted_top20": accepted,
        "full_candidate_audit": list(accepted),
        "candidate_snapshot_sha256": canonical_sha256(accepted),
        "market_gate_snapshot_sha256": canonical_sha256(market_gate),
        "price_snapshot_sha256": canonical_sha256({"engine_day": engine_day}),
        "future_candidate_data_access_count": 0,
        "future_split_access_count": 0,
        "entry_attempt_date": entry_date,
        "planned_exit_date": exit_date,
    }


def patch_candidates(monkeypatch, *, candidate_count: int = 3, override=None) -> None:
    def fake(frames, universe, split_history, study_calendar, engine_day, collector_commit):
        if override is not None:
            return override(engine_day)
        return fake_candidate_result(engine_day, candidate_count=candidate_count)

    monkeypatch.setattr(processing, "generate_forward_candidates_for_day", fake)


# ---------------------------------------------------------------------------
# Study fixture helpers
# ---------------------------------------------------------------------------


def seed_rows_for(days: Sequence[str] = SEED_DAYS) -> list[dict[str, Any]]:
    return synthetic_seed_rows(TICKERS, days)


def activation_context(seed_rows=None, **overrides) -> dict[str, Any]:
    context = build_activation_context(seed_rows if seed_rows is not None else seed_rows_for(), TICKERS)
    context.update(overrides)
    return context


def make_acquisition(study_root: Path, day: str) -> None:
    build_acquisition_bundle(study_root, SNAPSHOT, day, day_price(day))


def run_day(study_root: Path, day: str, *, seed_rows=None, context=None, calendar=None) -> dict[str, Any]:
    rows = seed_rows if seed_rows is not None else seed_rows_for()
    return processing.process_forward_day(
        study_root=study_root,
        engine_day=day,
        universe_csv=UNIVERSE_CSV,
        calendar_snapshot=calendar if calendar is not None else SNAPSHOT,
        seed_rows=rows,
        activation_context=context if context is not None else activation_context(rows),
    )


def process_first_day(tmp_path: Path, monkeypatch, **kwargs) -> dict[str, Any]:
    patch_candidates(monkeypatch, **kwargs)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    return run_day(tmp_path, ACTIVATION_BOUNDARY)


def days_dir(study_root: Path, day: str) -> Path:
    return ForwardStudyStore(study_root).days_root / day


# ---------------------------------------------------------------------------
# Engine-day sequencing
# ---------------------------------------------------------------------------


def test_first_day_must_equal_activation_boundary(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    second_day = ENGINE_DAYS[1]
    make_acquisition(tmp_path, second_day)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, second_day)
    assert excinfo.value.reason == "FIRST_ENGINE_DAY_NOT_ACTIVATION_BOUNDARY"


def test_first_day_at_activation_boundary_succeeds(tmp_path, monkeypatch):
    summary = process_first_day(tmp_path, monkeypatch)
    assert summary["status"] == "PASS"
    assert summary["previous_complete_engine_day"] is None
    assert summary["forward_day_persisted"] is True


def test_second_day_must_be_next_jpx_trading_day(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    second_day = ENGINE_DAYS[1]
    make_acquisition(tmp_path, second_day)
    summary = run_day(tmp_path, second_day)
    assert summary["previous_complete_engine_day"] == ACTIVATION_BOUNDARY


def test_gap_day_blocked(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    gap_day = ENGINE_DAYS[2]
    make_acquisition(tmp_path, gap_day)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, gap_day)
    assert excinfo.value.reason == "ENGINE_DAY_NOT_NEXT_JPX_TRADING_DAY"


def test_backfill_day_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ENGINE_DAYS[0])
    make_acquisition(tmp_path, ENGINE_DAYS[1])
    run_day(tmp_path, ENGINE_DAYS[0])
    run_day(tmp_path, ENGINE_DAYS[1])
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ENGINE_DAYS[0])
    assert excinfo.value.reason == "ENGINE_DAY_NOT_FORWARD_OF_PERSISTED_STORE"


def test_duplicate_day_blocked(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason == "ENGINE_DAY_NOT_FORWARD_OF_PERSISTED_STORE"


def test_weekend_engine_day_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, "2026-08-15")
    assert excinfo.value.reason == "ENGINE_DAY_NOT_JPX_TRADING_DAY"


def test_holiday_engine_day_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, "2026-01-01")
    assert excinfo.value.reason == "ENGINE_DAY_NOT_JPX_TRADING_DAY"


def test_engine_day_outside_calendar_coverage_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, "2025-12-31")
    assert excinfo.value.reason == "ENGINE_DAY_OUTSIDE_CALENDAR_COVERAGE"


def test_engine_day_before_activation_boundary_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, "2026-08-07")
    assert excinfo.value.reason == "ENGINE_DAY_BEFORE_ACTIVATION_BOUNDARY"


# ---------------------------------------------------------------------------
# Activation context validation
# ---------------------------------------------------------------------------


def test_activation_context_unknown_field_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    context = activation_context()
    context["unexpected"] = 1
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, context=context)
    assert excinfo.value.reason == "ACTIVATION_CONTEXT_SCHEMA_INVALID"


def test_activation_context_missing_field_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    context = activation_context()
    del context["implementation_commit"]
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, context=context)
    assert excinfo.value.reason == "ACTIVATION_CONTEXT_SCHEMA_INVALID"


def test_activation_context_wrong_collector_commit_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    context = activation_context(collector_commit="a" * 40)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, context=context)
    assert excinfo.value.reason == "COLLECTOR_COMMIT_MISMATCH"


def test_activation_context_invalid_manifest_sha_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    context = activation_context(activation_manifest_sha256="short")
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, context=context)
    assert excinfo.value.reason == "ACTIVATION_MANIFEST_SHA_INVALID"


def test_activation_context_invalid_implementation_commit_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    context = activation_context(implementation_commit="zz")
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, context=context)
    assert excinfo.value.reason == "IMPLEMENTATION_COMMIT_INVALID"


def test_activation_boundary_must_be_jpx_trading_day(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    context = activation_context(activation_boundary_first_jpx_trading_date="2026-08-15")
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, context=context)
    assert excinfo.value.reason == "ACTIVATION_BOUNDARY_NOT_JPX_TRADING_DAY"


def test_processing_never_creates_activation_manifest(tmp_path, monkeypatch):
    summary = process_first_day(tmp_path, monkeypatch)
    assert summary["activation_created"] is False
    assert not list(Path(tmp_path).glob("**/activation_manifest.json"))


# ---------------------------------------------------------------------------
# Seed validation binding
# ---------------------------------------------------------------------------


def test_seed_canonical_hash_mismatch_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    context = activation_context(expected_seed_canonical_sha256="b" * 64)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, context=context)
    assert excinfo.value.reason.startswith("SEED_VALIDATION_FAILED:")


def test_seed_ticker_manifest_hash_mismatch_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    context = activation_context(expected_seed_ticker_manifest_sha256="c" * 64)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, context=context)
    assert excinfo.value.reason == "SEED_TICKER_MANIFEST_HASH_MISMATCH"


def test_seed_row_on_activation_boundary_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    rows = seed_rows_for()
    rows.append({**rows[0], "trading_date": ACTIVATION_BOUNDARY})
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, seed_rows=rows, context=activation_context())
    assert excinfo.value.reason.startswith("SEED_VALIDATION_FAILED:")


def test_seed_row_after_activation_boundary_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    rows = seed_rows_for()
    rows.append({**rows[0], "trading_date": ENGINE_DAYS[3]})
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, seed_rows=rows, context=activation_context())
    assert excinfo.value.reason.startswith("SEED_VALIDATION_FAILED:")


def test_seed_is_revalidated_not_trusted(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    rows = seed_rows_for()
    tampered = [dict(row) for row in rows]
    tampered[0]["raw_close"] = tampered[0]["raw_close"] + 1.0
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY, seed_rows=tampered, context=activation_context(rows))
    assert excinfo.value.reason.startswith("SEED_VALIDATION_FAILED:")


def test_universe_sha_mismatch_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    altered = tmp_path / "V4_UNIVERSE.csv"
    altered.write_bytes(UNIVERSE_CSV.read_bytes() + b"\n")
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        processing.process_forward_day(
            study_root=tmp_path,
            engine_day=ACTIVATION_BOUNDARY,
            universe_csv=altered,
            calendar_snapshot=SNAPSHOT,
            seed_rows=seed_rows_for(),
            activation_context=activation_context(),
        )
    assert excinfo.value.reason.startswith("UNIVERSE_VALIDATION_FAILED:")


# ---------------------------------------------------------------------------
# Acquisition verification binding
# ---------------------------------------------------------------------------


def test_missing_acquisition_bundle_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason.startswith("ACQUISITION_VERIFICATION_FAILED:")


def test_tampered_acquisition_bundle_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    price_path = tmp_path / "acquisitions" / ACTIVATION_BOUNDARY / "price_snapshot.json"
    price_path.write_text("[]", encoding="utf-8")
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason.startswith("ACQUISITION_VERIFICATION_FAILED:")


def test_acquisition_staging_remnant_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    (tmp_path / "acquisitions" / f"{ENGINE_DAYS[1]}.staging-remnant").mkdir()
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert "PARTIAL_ACQUISITION_COMMIT" in excinfo.value.reason


def test_acquisition_for_other_engine_day_blocked(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ENGINE_DAYS[1])
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason.startswith("ACQUISITION_VERIFICATION_FAILED:")


def test_verification_failure_leaves_no_forward_day(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked):
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert not (tmp_path / "days" / ACTIVATION_BOUNDARY).exists()


# ---------------------------------------------------------------------------
# Processing price snapshot schema and provenance
# ---------------------------------------------------------------------------


def read_snapshot(study_root: Path, day: str, name: str) -> Any:
    return json.loads((days_dir(study_root, day) / name).read_text(encoding="utf-8"))


def test_processing_price_snapshot_schema(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    payload = read_snapshot(tmp_path, ACTIVATION_BOUNDARY, "price_snapshot.json")
    assert set(payload) == set(processing.PRICE_SNAPSHOT_FIELDS)
    assert payload["schema_version"] == "V7_FORWARD_PROCESSING_PRICE_V1"
    assert payload["engine_day"] == ACTIVATION_BOUNDARY
    assert payload["previous_complete_engine_day"] is None
    assert payload["implementation_commit"] == IMPLEMENTATION_COMMIT


def test_processing_price_snapshot_binds_acquisition_manifest_sha(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    payload = read_snapshot(tmp_path, ACTIVATION_BOUNDARY, "price_snapshot.json")
    manifest_path = tmp_path / "acquisitions" / ACTIVATION_BOUNDARY / "acquisition_manifest.json"
    assert payload["acquisition_manifest_sha256"] == processing.sha256_bytes(manifest_path.read_bytes())


@pytest.mark.parametrize("filename,field", [
    ("price_snapshot.json", "acquisition_price_snapshot_sha256"),
    ("missing_snapshot.json", "acquisition_missing_snapshot_sha256"),
    ("split_snapshot.json", "acquisition_split_snapshot_sha256"),
])
def test_processing_price_snapshot_binds_acquisition_hashes(tmp_path, monkeypatch, filename, field):
    process_first_day(tmp_path, monkeypatch)
    payload = read_snapshot(tmp_path, ACTIVATION_BOUNDARY, "price_snapshot.json")
    path = tmp_path / "acquisitions" / ACTIVATION_BOUNDARY / filename
    assert payload[field] == processing.sha256_bytes(path.read_bytes())


def test_processing_price_snapshot_carries_d0_content(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    payload = read_snapshot(tmp_path, ACTIVATION_BOUNDARY, "price_snapshot.json")
    acquisition_price = json.loads(
        (tmp_path / "acquisitions" / ACTIVATION_BOUNDARY / "price_snapshot.json").read_text(encoding="utf-8")
    )
    assert canonical_json_bytes(payload["d0_price_rows"]) == canonical_json_bytes(acquisition_price)
    assert len(payload["d0_price_rows"]) == len(TICKERS)


def test_second_day_records_previous_complete_engine_day(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    make_acquisition(tmp_path, ENGINE_DAYS[1])
    run_day(tmp_path, ENGINE_DAYS[1])
    payload = read_snapshot(tmp_path, ENGINE_DAYS[1], "price_snapshot.json")
    assert payload["previous_complete_engine_day"] == ACTIVATION_BOUNDARY


# ---------------------------------------------------------------------------
# Candidate snapshot / market gate snapshot
# ---------------------------------------------------------------------------


def test_candidate_snapshot_schema_and_hash(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    payload = read_snapshot(tmp_path, ACTIVATION_BOUNDARY, "candidate_snapshot.json")
    assert set(payload) == set(processing.CANDIDATE_SNAPSHOT_FIELDS)
    assert payload["schema_version"] == "V7_FORWARD_PROCESSING_CANDIDATE_V1"
    assert canonical_sha256(payload["accepted_top20"]) == payload["candidate_snapshot_sha256"]
    assert payload["future_candidate_data_access_count"] == 0
    assert payload["future_split_access_count"] == 0


def test_candidate_snapshot_binds_processing_price_snapshot(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    price = read_snapshot(tmp_path, ACTIVATION_BOUNDARY, "price_snapshot.json")
    candidate = read_snapshot(tmp_path, ACTIVATION_BOUNDARY, "candidate_snapshot.json")
    assert candidate["source_processing_price_snapshot_sha256"] == canonical_sha256(price)


def test_market_gate_snapshot_schema_and_hash(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    payload = read_snapshot(tmp_path, ACTIVATION_BOUNDARY, "market_gate_snapshot.json")
    assert set(payload) == set(processing.MARKET_GATE_SNAPSHOT_FIELDS)
    assert payload["schema_version"] == "V7_FORWARD_PROCESSING_MARKET_GATE_V1"
    assert canonical_sha256(payload["market_gate"]) == payload["market_gate_snapshot_sha256"]


def test_candidate_generated_once_and_shared_by_both_arms(tmp_path, monkeypatch):
    calls: list[str] = []

    def counting(engine_day: str) -> dict[str, Any]:
        calls.append(engine_day)
        return fake_candidate_result(engine_day)

    patch_candidates(monkeypatch, override=counting)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    summary = run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert calls == [ACTIVATION_BOUNDARY]
    assert summary["accepted_candidate_count"] == 3


def test_candidate_future_access_counter_blocks(tmp_path, monkeypatch):
    def leaking(engine_day: str) -> dict[str, Any]:
        return {**fake_candidate_result(engine_day), "future_candidate_data_access_count": 1}

    patch_candidates(monkeypatch, override=leaking)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason == "FUTURE_CANDIDATE_DATA_ACCESS"


def test_candidate_future_split_counter_blocks(tmp_path, monkeypatch):
    def leaking(engine_day: str) -> dict[str, Any]:
        return {**fake_candidate_result(engine_day), "future_split_access_count": 1}

    patch_candidates(monkeypatch, override=leaking)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason == "FUTURE_SPLIT_ACCESS"


def test_candidate_hash_inconsistency_blocks_before_persist(tmp_path, monkeypatch):
    def inconsistent(engine_day: str) -> dict[str, Any]:
        return {**fake_candidate_result(engine_day), "candidate_snapshot_sha256": "d" * 64}

    patch_candidates(monkeypatch, override=inconsistent)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason == "CANDIDATE_SNAPSHOT_HASH_MISMATCH"
    assert not (tmp_path / "days" / ACTIVATION_BOUNDARY).exists()


def test_market_gate_hash_inconsistency_blocks_before_persist(tmp_path, monkeypatch):
    def inconsistent(engine_day: str) -> dict[str, Any]:
        return {**fake_candidate_result(engine_day), "market_gate_snapshot_sha256": "e" * 64}

    patch_candidates(monkeypatch, override=inconsistent)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason == "MARKET_GATE_SNAPSHOT_HASH_MISMATCH"
    assert not (tmp_path / "days" / ACTIVATION_BOUNDARY).exists()


def test_candidate_failure_leaves_no_final_day(tmp_path, monkeypatch):
    def exploding(engine_day: str):
        from src.v7_forward_candidate import V7CandidateBlocked

        raise V7CandidateBlocked("SYNTHETIC_CANDIDATE_FAILURE")

    patch_candidates(monkeypatch, override=exploding)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason.startswith("CANDIDATE_GENERATION_FAILED:")
    assert not (tmp_path / "days" / ACTIVATION_BOUNDARY).exists()


# ---------------------------------------------------------------------------
# Dual arm construction
# ---------------------------------------------------------------------------


def test_control_parameter_hash(tmp_path, monkeypatch):
    summary = process_first_day(tmp_path, monkeypatch)
    assert summary["control"]["parameters_sha256"] == processing.CONTROL_PARAMETERS_SHA256


def test_variant_parameter_hash(tmp_path, monkeypatch):
    summary = process_first_day(tmp_path, monkeypatch)
    assert summary["variant"]["parameters_sha256"] == processing.CAPACITY_3_PARAMETERS_SHA256


def test_single_parameter_difference_enforced():
    from src.v7_capacity_engine import V7EngineParameters, validate_single_parameter_difference

    assert validate_single_parameter_difference(
        V7EngineParameters.control(), V7EngineParameters.capacity_3()
    ) is True


def test_arm_state_objects_independent(tmp_path, monkeypatch):
    control, variant = processing.build_dual_arms(
        {"AAA": {ACTIVATION_BOUNDARY: {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Adj Close": 100.0, "Volume": 1000.0}}},
        ENGINE_DAYS,
        [],
        {},
    )
    assert control.state is not variant.state
    control.state.available_cash -= 1.0
    assert control.state.available_cash != variant.state.available_cash


def test_capacity_arm_fills_more_positions_than_control(tmp_path, monkeypatch):
    patch_candidates(monkeypatch, candidate_count=3)
    for day in ENGINE_DAYS[:2]:
        make_acquisition(tmp_path, day)
    run_day(tmp_path, ENGINE_DAYS[0])
    summary = run_day(tmp_path, ENGINE_DAYS[1])
    assert summary["control"]["open_position_count"] == 2
    assert summary["variant"]["open_position_count"] == 3
    assert summary["control"]["skip_reason_counts"]["MAX_OPEN_POSITIONS"] == 1
    assert summary["variant"]["skip_reason_counts"]["MAX_OPEN_POSITIONS"] == 0


# ---------------------------------------------------------------------------
# Restart / previous runtime
# ---------------------------------------------------------------------------


def test_previous_runtime_restored_into_matching_arms(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    for day in ENGINE_DAYS[:2]:
        make_acquisition(tmp_path, day)
    run_day(tmp_path, ENGINE_DAYS[0])
    run_day(tmp_path, ENGINE_DAYS[1])
    store = ForwardStudyStore(tmp_path)
    latest = store.load_latest_runtime()
    assert latest["arm_a_runtime"]["parameters_sha256"] == processing.CONTROL_PARAMETERS_SHA256
    assert latest["arm_b_runtime"]["parameters_sha256"] == processing.CAPACITY_3_PARAMETERS_SHA256
    assert latest["arm_a_runtime"]["engine_day"] == ENGINE_DAYS[1]


def test_arm_swapped_runtime_restore_blocked(tmp_path, monkeypatch):
    from src.v7_capacity_engine import V7EngineParameters
    from src.v7_forward_persistence import V7ForwardPersistenceBlocked, restore_engine_runtime

    process_first_day(tmp_path, monkeypatch)
    store = ForwardStudyStore(tmp_path)
    latest = store.load_latest_runtime()
    control, variant = processing.build_dual_arms(
        {"AAA": {ACTIVATION_BOUNDARY: {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Adj Close": 100.0, "Volume": 1000.0}}},
        ENGINE_DAYS,
        [],
        {},
    )
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        restore_engine_runtime(control, latest["arm_b_runtime"])
    assert excinfo.value.reason == "RUNTIME_PARAMETERS_MISMATCH"
    with pytest.raises(V7ForwardPersistenceBlocked):
        restore_engine_runtime(variant, latest["arm_a_runtime"])


def test_pending_order_survives_next_day_orchestration(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ENGINE_DAYS[0])
    summary = run_day(tmp_path, ENGINE_DAYS[0])
    assert summary["control"]["pending_order_count"] == 3
    make_acquisition(tmp_path, ENGINE_DAYS[1])
    summary_two = run_day(tmp_path, ENGINE_DAYS[1])
    assert summary_two["control"]["pending_order_count"] == 3
    assert summary_two["control"]["open_position_count"] == 2


def test_open_position_survives_across_days(tmp_path, monkeypatch):
    patch_candidates(monkeypatch, candidate_count=0)
    make_acquisition(tmp_path, ENGINE_DAYS[0])
    monkeypatch.setattr(
        processing, "generate_forward_candidates_for_day",
        lambda *args, **kwargs: fake_candidate_result(ENGINE_DAYS[0], candidate_count=2),
    )
    run_day(tmp_path, ENGINE_DAYS[0])
    patch_candidates(monkeypatch, candidate_count=0)
    for day in ENGINE_DAYS[1:4]:
        make_acquisition(tmp_path, day)
        summary = run_day(tmp_path, day)
    assert summary["control"]["open_position_count"] == 2
    assert summary["variant"]["open_position_count"] == 2


def test_pending_proceeds_survive_after_exit(tmp_path, monkeypatch):
    def candidates_first_day_only(engine_day: str) -> dict[str, Any]:
        count = 2 if engine_day == ENGINE_DAYS[0] else 0
        return fake_candidate_result(engine_day, candidate_count=count)

    patch_candidates(monkeypatch, override=candidates_first_day_only)
    summary = None
    for day in ENGINE_DAYS[:11]:
        make_acquisition(tmp_path, day)
        summary = run_day(tmp_path, day)
    assert summary["control"]["closed_trade_count"] == 2
    assert summary["control"]["pending_proceeds_count"] == 2
    assert summary["control"]["open_position_count"] == 0


# ---------------------------------------------------------------------------
# Missing / split behavior
# ---------------------------------------------------------------------------


def _acquisition_with_missing(study_root: Path, day: str, missing_ticker: str) -> None:
    """Publish a bundle where one ticker has an audited-missing D0 observation."""
    from scripts.check_v7_forward_day_processing import FakeAcquisitionOpener, _FakeResponse
    from src.v7_daily_acquisition import acquire_daily_bundle
    from datetime import datetime, timezone

    price = day_price(day)

    class Opener(FakeAcquisitionOpener):
        def __call__(self, request_obj):
            ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
            self.calls.append(ticker)
            if ticker == missing_ticker:
                body = {"chart": {"error": None, "result": [{
                    "meta": {"symbol": ticker + ".T"},
                    "timestamp": [],
                    "indicators": {"quote": [{"open": [], "high": [], "low": [], "close": [], "volume": []}],
                                   "adjclose": [{"adjclose": []}]},
                }]}}
                return _FakeResponse(json.dumps(body).encode("utf-8"),
                                     url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")
            return super().__call__(request_obj)

    opener = Opener(day, price)
    clock_values = iter([
        datetime.fromisoformat(day).replace(hour=7, tzinfo=timezone.utc),
        datetime.fromisoformat(day).replace(hour=8, tzinfo=timezone.utc),
    ])
    state = {"now": 0.0}
    acquire_daily_bundle(
        output_root=study_root,
        universe_csv=UNIVERSE_CSV,
        calendar_snapshot=SNAPSHOT,
        engine_day=day,
        opener=opener,
        clock=lambda: next(clock_values),
        monotonic_clock=lambda: state["now"],
        sleep_fn=lambda seconds: state.__setitem__("now", state["now"] + seconds),
    )


def test_missing_d1_entry_price_yields_entry_data_unavailable(tmp_path, monkeypatch):
    patch_candidates(monkeypatch, candidate_count=1)
    make_acquisition(tmp_path, ENGINE_DAYS[0])
    run_day(tmp_path, ENGINE_DAYS[0])
    _acquisition_with_missing(tmp_path, ENGINE_DAYS[1], TICKERS[0])
    summary = run_day(tmp_path, ENGINE_DAYS[1])
    assert summary["control"]["skip_reason_counts"]["ENTRY_DATA_UNAVAILABLE"] == 1
    assert summary["control"]["open_position_count"] == 0


def test_missing_open_position_close_blocks_whole_day(tmp_path, monkeypatch):
    def candidates_first_day_only(engine_day: str) -> dict[str, Any]:
        count = 1 if engine_day == ENGINE_DAYS[0] else 0
        return fake_candidate_result(engine_day, candidate_count=count)

    patch_candidates(monkeypatch, override=candidates_first_day_only)
    for day in ENGINE_DAYS[:2]:
        make_acquisition(tmp_path, day)
        run_day(tmp_path, day)
    _acquisition_with_missing(tmp_path, ENGINE_DAYS[2], TICKERS[0])
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ENGINE_DAYS[2])
    assert "OPEN_POSITION_MTM_PRICE_UNAVAILABLE" in excinfo.value.reason
    assert not (tmp_path / "days" / ENGINE_DAYS[2]).exists()


def test_missing_planned_exit_open_blocks_whole_day(tmp_path, monkeypatch):
    def candidates_first_day_only(engine_day: str) -> dict[str, Any]:
        count = 1 if engine_day == ENGINE_DAYS[0] else 0
        return fake_candidate_result(engine_day, candidate_count=count)

    patch_candidates(monkeypatch, override=candidates_first_day_only)
    for day in ENGINE_DAYS[:10]:
        make_acquisition(tmp_path, day)
        run_day(tmp_path, day)
    _acquisition_with_missing(tmp_path, ENGINE_DAYS[10], TICKERS[0])
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ENGINE_DAYS[10])
    assert "PLANNED_EXIT_PRICE_UNAVAILABLE" in excinfo.value.reason
    assert not (tmp_path / "days" / ENGINE_DAYS[10]).exists()


def _acquisition_with_split(study_root: Path, day: str, split_ticker: str) -> None:
    from scripts.check_v7_forward_day_processing import FakeAcquisitionOpener, _FakeResponse, _epoch
    from src.v7_daily_acquisition import acquire_daily_bundle
    from datetime import datetime, timezone

    price = day_price(day)

    class Opener(FakeAcquisitionOpener):
        def __call__(self, request_obj):
            ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
            response = super().__call__(request_obj)
            if ticker != split_ticker:
                return response
            body = json.loads(response.payload.decode("utf-8"))
            body["chart"]["result"][0]["events"] = {"splits": {
                str(_epoch(day)): {"date": _epoch(day), "numerator": 2, "denominator": 1, "splitRatio": "2:1"},
            }}
            return _FakeResponse(json.dumps(body).encode("utf-8"), url=response.url)

    opener = Opener(day, price)
    clock_values = iter([
        datetime.fromisoformat(day).replace(hour=7, tzinfo=timezone.utc),
        datetime.fromisoformat(day).replace(hour=8, tzinfo=timezone.utc),
    ])
    state = {"now": 0.0}
    acquire_daily_bundle(
        output_root=study_root,
        universe_csv=UNIVERSE_CSV,
        calendar_snapshot=SNAPSHOT,
        engine_day=day,
        opener=opener,
        clock=lambda: next(clock_values),
        monotonic_clock=lambda: state["now"],
        sleep_fn=lambda seconds: state.__setitem__("now", state["now"] + seconds),
    )


def test_split_before_entry_yields_skip(tmp_path, monkeypatch):
    patch_candidates(monkeypatch, candidate_count=1)
    make_acquisition(tmp_path, ENGINE_DAYS[0])
    run_day(tmp_path, ENGINE_DAYS[0])
    _acquisition_with_split(tmp_path, ENGINE_DAYS[1], TICKERS[0])
    summary = run_day(tmp_path, ENGINE_DAYS[1])
    assert summary["control"]["skip_reason_counts"]["SPLIT_EFFECTIVE_BEFORE_ENTRY"] == 1
    assert summary["control"]["open_position_count"] == 0


def test_split_on_open_position_blocks_whole_day(tmp_path, monkeypatch):
    def candidates_first_day_only(engine_day: str) -> dict[str, Any]:
        count = 1 if engine_day == ENGINE_DAYS[0] else 0
        return fake_candidate_result(engine_day, candidate_count=count)

    patch_candidates(monkeypatch, override=candidates_first_day_only)
    for day in ENGINE_DAYS[:2]:
        make_acquisition(tmp_path, day)
        run_day(tmp_path, day)
    _acquisition_with_split(tmp_path, ENGINE_DAYS[2], TICKERS[0])
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ENGINE_DAYS[2])
    assert "OPEN_POSITION_SPLIT_SPANNING" in excinfo.value.reason
    assert not (tmp_path / "days" / ENGINE_DAYS[2]).exists()


def test_split_history_rejects_future_effective_date():
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        processing.build_split_history(
            [], [{"ticker": "AAA", "effective_date": ENGINE_DAYS[3]}], ENGINE_DAYS[0]
        )
    assert excinfo.value.reason == "FUTURE_SPLIT_ACCESS"


def test_split_history_merges_past_and_current(tmp_path):
    history, by_day = processing.build_split_history(
        [{"ticker": "AAA", "effective_date": ENGINE_DAYS[0]}],
        [{"ticker": "BBB", "effective_date": ENGINE_DAYS[1]}],
        ENGINE_DAYS[1],
    )
    assert history == {"AAA": [ENGINE_DAYS[0]], "BBB": [ENGINE_DAYS[1]]}
    assert by_day == {ENGINE_DAYS[0]: ["AAA"], ENGINE_DAYS[1]: ["BBB"]}


# ---------------------------------------------------------------------------
# History reconstruction / causality
# ---------------------------------------------------------------------------


def test_past_d0_rows_added_to_history(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    for day in ENGINE_DAYS[:2]:
        make_acquisition(tmp_path, day)
        run_day(tmp_path, day)
    store = ForwardStudyStore(tmp_path)
    history = processing.read_past_forward_history(store, ENGINE_DAYS[2])
    assert len(history["history_rows"]) == 2 * len(TICKERS)
    assert {row["trading_date"] for row in history["history_rows"]} == set(ENGINE_DAYS[:2])


def test_history_excludes_current_and_future_days(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ENGINE_DAYS[0])
    run_day(tmp_path, ENGINE_DAYS[0])
    store = ForwardStudyStore(tmp_path)
    history = processing.read_past_forward_history(store, ENGINE_DAYS[1])
    assert {row["trading_date"] for row in history["history_rows"]} == {ENGINE_DAYS[0]}
    assert all(row["trading_date"] < ENGINE_DAYS[1] for row in history["history_rows"])


def test_future_persisted_day_is_never_read(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    for day in ENGINE_DAYS[:2]:
        make_acquisition(tmp_path, day)
        run_day(tmp_path, day)
    store = ForwardStudyStore(tmp_path)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        processing.read_past_forward_history(store, ENGINE_DAYS[0])
    assert excinfo.value.reason == "ENGINE_DAY_NOT_FORWARD_OF_PERSISTED_STORE"


def test_future_acquisition_bundle_is_not_read(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    make_acquisition(tmp_path, ENGINE_DAYS[1])
    summary = run_day(tmp_path, ACTIVATION_BOUNDARY)
    payload = read_snapshot(tmp_path, ACTIVATION_BOUNDARY, "price_snapshot.json")
    assert {row["trading_date"] for row in payload["d0_price_rows"]} == {ACTIVATION_BOUNDARY}
    assert summary["engine_day"] == ACTIVATION_BOUNDARY


def test_engine_frames_never_contain_future_dates(tmp_path):
    import pandas as pd

    frame = pd.DataFrame(
        [{"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Adj Close": 1.0, "Volume": 1.0}],
        index=pd.DatetimeIndex([pd.Timestamp(ENGINE_DAYS[2])]),
    )
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        processing.build_engine_frames({"AAA": frame}, ENGINE_DAYS[0])
    assert excinfo.value.reason.startswith("ENGINE_FRAME_FUTURE_DATE:")


def test_latest252_bridge_selection_is_delegated(tmp_path, monkeypatch):
    """The processing layer must not re-implement the 252-observation window."""
    source = Path(processing.__file__).read_text(encoding="utf-8")
    assert "252" not in source
    assert "build_forward_frames_from_seed_and_d0" in source


# ---------------------------------------------------------------------------
# Processed-day verifier
# ---------------------------------------------------------------------------


def verify(tmp_path: Path, day: str = ACTIVATION_BOUNDARY, context=None) -> dict[str, Any]:
    return processing.verify_processed_forward_day(
        study_root=tmp_path,
        engine_day=day,
        universe_csv=UNIVERSE_CSV,
        activation_context=context if context is not None else activation_context(),
    )


def test_verifier_passes_on_clean_processed_day(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    result = verify(tmp_path)
    assert result["status"] == "PASS"
    assert result["engine_day"] == ACTIVATION_BOUNDARY
    assert result["valid_d0_count"] == len(TICKERS)


def test_verifier_detects_implementation_commit_mismatch(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        verify(tmp_path, context=activation_context(implementation_commit="9" * 40))
    assert excinfo.value.reason.startswith("IMPLEMENTATION_COMMIT_MISMATCH:")


def test_verifier_detects_activation_manifest_hash_mismatch(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        verify(tmp_path, context=activation_context(activation_manifest_sha256="9" * 64))
    assert excinfo.value.reason.startswith("FORWARD_STORE_VERIFICATION_FAILED:")


def test_verifier_detects_collector_commit_mismatch(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        verify(tmp_path, context=activation_context(collector_commit="9" * 40))
    assert excinfo.value.reason.startswith("FORWARD_STORE_VERIFICATION_FAILED:")


def test_verifier_detects_latest_day_mismatch(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    for day in ENGINE_DAYS[:2]:
        make_acquisition(tmp_path, day)
        run_day(tmp_path, day)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        verify(tmp_path, day=ENGINE_DAYS[0])
    assert excinfo.value.reason == "LATEST_COMPLETE_DAY_MISMATCH"


def test_verifier_detects_candidate_snapshot_tamper(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    path = days_dir(tmp_path, ACTIVATION_BOUNDARY) / "candidate_snapshot.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["accepted_top20"] = payload["accepted_top20"][:1]
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(processing.V7ForwardDayProcessingBlocked):
        verify(tmp_path)


def test_verifier_detects_market_gate_hash_tamper(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    path = days_dir(tmp_path, ACTIVATION_BOUNDARY) / "market_gate_snapshot.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["market_gate_snapshot_sha256"] = "7" * 64
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(processing.V7ForwardDayProcessingBlocked):
        verify(tmp_path)


def test_verifier_detects_processing_price_provenance_tamper(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    path = days_dir(tmp_path, ACTIVATION_BOUNDARY) / "candidate_snapshot.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["source_processing_price_snapshot_sha256"] = "6" * 64
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(processing.V7ForwardDayProcessingBlocked):
        verify(tmp_path)


def test_verifier_detects_acquisition_snapshot_hash_tamper(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    path = days_dir(tmp_path, ACTIVATION_BOUNDARY) / "price_snapshot.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["acquisition_manifest_sha256"] = "5" * 64
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(processing.V7ForwardDayProcessingBlocked):
        verify(tmp_path)


def test_verifier_detects_processing_price_schema_tamper(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    path = days_dir(tmp_path, ACTIVATION_BOUNDARY) / "price_snapshot.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = "WRONG"
    path.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(processing.V7ForwardDayProcessingBlocked):
        verify(tmp_path)


def test_verifier_detects_missing_acquisition_bundle(tmp_path, monkeypatch):
    import shutil

    process_first_day(tmp_path, monkeypatch)
    shutil.rmtree(tmp_path / "acquisitions" / ACTIVATION_BOUNDARY)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        verify(tmp_path)
    assert excinfo.value.reason.startswith("ACQUISITION_VERIFICATION_FAILED:")


# ---------------------------------------------------------------------------
# No interim profit peeking
# ---------------------------------------------------------------------------


FORBIDDEN_SUMMARY_TOKENS = (
    "profit", "realized", "equity_value", "drawdown", "profit_factor",
    "win_rate", "pnl", "net_return", "performance",
)


def _summary_keys(value: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            keys.append(prefix + str(key))
            keys.extend(_summary_keys(item, prefix + str(key) + "."))
    return keys


def test_processing_summary_exposes_no_profit_metrics(tmp_path, monkeypatch):
    summary = process_first_day(tmp_path, monkeypatch)
    keys = [key.lower() for key in _summary_keys(summary)]
    offending = [
        key for key in keys
        for token in FORBIDDEN_SUMMARY_TOKENS
        if token in key and not key.endswith("profit_metrics_exposed")
    ]
    assert offending == []
    assert summary["profit_metrics_exposed"] is False


def test_processing_summary_exposes_only_structural_counters(tmp_path, monkeypatch):
    summary = process_first_day(tmp_path, monkeypatch)
    assert set(summary["control"]) == {
        "open_position_count", "pending_order_count", "pending_proceeds_count",
        "closed_trade_count", "ledger_row_count", "daily_equity_row_count",
        "event_audit_row_count", "safety_counters", "skip_reason_counts",
        "parameters_sha256",
    }


def test_verifier_result_exposes_no_profit_metrics(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    result = verify(tmp_path)
    keys = [key.lower() for key in _summary_keys(result)]
    assert not any(token in key for key in keys for token in FORBIDDEN_SUMMARY_TOKENS)


def test_no_arm_performance_comparison_in_summary(tmp_path, monkeypatch):
    summary = process_first_day(tmp_path, monkeypatch)
    assert "arm_comparison" not in summary
    assert "control_vs_variant" not in summary


# ---------------------------------------------------------------------------
# Persistence integration
# ---------------------------------------------------------------------------


def test_forward_day_directory_created_only_by_write_day(tmp_path, monkeypatch):
    process_first_day(tmp_path, monkeypatch)
    day_dir = days_dir(tmp_path, ACTIVATION_BOUNDARY)
    assert sorted(entry.name for entry in day_dir.iterdir()) == [
        "arm_a_runtime.json", "arm_b_runtime.json", "candidate_snapshot.json",
        "checkpoint.json", "market_gate_snapshot.json", "price_snapshot.json",
    ]


def test_persistence_failure_leaves_no_complete_day(tmp_path, monkeypatch):
    from src.v7_forward_persistence import V7ForwardPersistenceBlocked

    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)

    def exploding_write(self, day, **kwargs):
        raise V7ForwardPersistenceBlocked("SYNTHETIC_PERSISTENCE_FAILURE")

    monkeypatch.setattr(ForwardStudyStore, "write_day", exploding_write)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason.startswith("FORWARD_DAY_PERSISTENCE_FAILED:")
    assert not (tmp_path / "days" / ACTIVATION_BOUNDARY).exists()


def test_arm_processing_failure_persists_neither_arm(tmp_path, monkeypatch):
    from src.v7_capacity_engine import CausalEventEngine, V7StudyBlocked

    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    original = CausalEventEngine.process_day
    calls = {"count": 0}

    def failing(self, day):
        calls["count"] += 1
        if calls["count"] == 2:
            raise V7StudyBlocked("SYNTHETIC_ARM_FAILURE")
        return original(self, day)

    monkeypatch.setattr(CausalEventEngine, "process_day", failing)
    with pytest.raises(processing.V7ForwardDayProcessingBlocked) as excinfo:
        run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert excinfo.value.reason.startswith("ARM_PROCESSING_FAILED:CAPACITY_3:")
    assert not (tmp_path / "days" / ACTIVATION_BOUNDARY).exists()


def test_acquisition_bundle_unchanged_after_processing(tmp_path, monkeypatch):
    patch_candidates(monkeypatch)
    make_acquisition(tmp_path, ACTIVATION_BOUNDARY)
    manifest_path = tmp_path / "acquisitions" / ACTIVATION_BOUNDARY / "acquisition_manifest.json"
    before = manifest_path.read_bytes()
    run_day(tmp_path, ACTIVATION_BOUNDARY)
    assert manifest_path.read_bytes() == before


def test_network_requests_reported_zero(tmp_path, monkeypatch):
    summary = process_first_day(tmp_path, monkeypatch)
    assert summary["network_requests"] == 0
