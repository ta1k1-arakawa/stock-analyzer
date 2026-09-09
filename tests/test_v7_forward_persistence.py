from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.v7_forward_persistence import (
    CHECKPOINT_FIELDS,
    DAY_FILE_ARM_A,
    DAY_FILE_ARM_B,
    DAY_FILE_CANDIDATE,
    DAY_FILE_CHECKPOINT,
    DAY_FILE_MARKET_GATE,
    DAY_FILE_PRICE,
    ForwardStudyStore,
    V7ForwardPersistenceBlocked,
    canonical_json_bytes,
    canonical_sha256,
    load_latest_runtime,
    sha256_bytes,
    verify_forward_store,
)

ACTIVATION_SHA = "a" * 64
OTHER_ACTIVATION_SHA = "c" * 64
COLLECTOR_COMMIT = "b" * 40
OTHER_COLLECTOR_COMMIT = "d" * 40


def runtime_payload(day: str, tag: str = "x") -> dict:
    return {"engine_day": day, "tag": tag}


def write(store: ForwardStudyStore, day: str, **overrides) -> dict:
    kwargs = dict(
        price_snapshot={"date": day},
        candidate_snapshot={"date": day},
        market_gate_snapshot={"date": day},
        arm_a_runtime=runtime_payload(day, "a"),
        arm_b_runtime=runtime_payload(day, "b"),
        activation_manifest_sha256=ACTIVATION_SHA,
        collector_commit=COLLECTOR_COMMIT,
    )
    kwargs.update(overrides)
    return store.write_day(day, **kwargs)


# ---------------------------------------------------------------------------
# Determinism primitives
# ---------------------------------------------------------------------------


def test_canonical_json_bytes_dict_insertion_order_invariant():
    first = {"a": 1, "b": {"y": 2, "x": 1}}
    second = {"b": {"x": 1, "y": 2}, "a": 1}
    assert canonical_json_bytes(first) == canonical_json_bytes(second)


def test_canonical_sha256_matches_manual_digest():
    value = {"k": 1}
    assert canonical_sha256(value) == sha256_bytes(canonical_json_bytes(value))


def test_canonical_json_bytes_blocks_nan():
    with pytest.raises(V7ForwardPersistenceBlocked):
        canonical_json_bytes({"v": math.nan})


def test_canonical_json_bytes_blocks_infinity():
    with pytest.raises(V7ForwardPersistenceBlocked):
        canonical_json_bytes({"v": math.inf})


# ---------------------------------------------------------------------------
# Store construction / append-only write mechanics
# ---------------------------------------------------------------------------


def test_store_creates_days_root(tmp_path):
    store = ForwardStudyStore(tmp_path / "study")
    assert store.days_root.is_dir()


def test_write_day_first_day_previous_checkpoint_null(tmp_path):
    store = ForwardStudyStore(tmp_path)
    record = write(store, "2020-01-01")
    assert record["previous_checkpoint_sha256"] is None
    assert record["status"] == "COMPLETE"
    for name in DAY_FILE_PRICE, DAY_FILE_CANDIDATE, DAY_FILE_MARKET_GATE, DAY_FILE_ARM_A, DAY_FILE_ARM_B, DAY_FILE_CHECKPOINT:
        assert (store.days_root / "2020-01-01" / name).exists()


def test_write_day_second_day_chains_to_previous(tmp_path):
    store = ForwardStudyStore(tmp_path)
    first = write(store, "2020-01-01")
    second = write(store, "2020-01-02")
    assert second["previous_checkpoint_sha256"] == first["current_checkpoint_sha256"]


def test_checkpoint_hash_excludes_itself(tmp_path):
    store = ForwardStudyStore(tmp_path)
    record = write(store, "2020-01-01")
    body = {key: record[key] for key in CHECKPOINT_FIELDS if key != "current_checkpoint_sha256"}
    assert sha256_bytes(canonical_json_bytes(body)) == record["current_checkpoint_sha256"]


def test_write_day_duplicate_engine_day_processing_blocked(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-01")
    assert excinfo.value.reason == "DUPLICATE_ENGINE_DAY_PROCESSING"


def test_write_day_rewrite_with_different_content_still_blocked(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-01", price_snapshot={"date": "2020-01-01", "changed": True})
    assert excinfo.value.reason == "DUPLICATE_ENGINE_DAY_PROCESSING"


def test_write_day_past_day_not_increasing_blocked(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-05")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-03")
    assert excinfo.value.reason == "ENGINE_DAY_NOT_INCREASING"


def test_write_day_gap_between_days_allowed(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    record = write(store, "2020-01-10")
    assert record["status"] == "COMPLETE"
    assert store._final_days() == ["2020-01-01", "2020-01-10"]


def test_write_day_activation_manifest_mismatch_blocked(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-02", activation_manifest_sha256=OTHER_ACTIVATION_SHA)
    assert excinfo.value.reason == "ACTIVATION_MANIFEST_MISMATCH"


def test_write_day_collector_commit_mismatch_blocked(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-02", collector_commit=OTHER_COLLECTOR_COMMIT)
    assert excinfo.value.reason == "COLLECTOR_COMMIT_MISMATCH"


def test_write_day_invalid_activation_sha_format_blocked(tmp_path):
    store = ForwardStudyStore(tmp_path)
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-01", activation_manifest_sha256="not-a-sha")
    assert excinfo.value.reason == "ACTIVATION_MANIFEST_SHA_INVALID"


def test_write_day_invalid_collector_commit_format_blocked(tmp_path):
    store = ForwardStudyStore(tmp_path)
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-01", collector_commit="short")
    assert excinfo.value.reason == "COLLECTOR_COMMIT_INVALID"


def test_write_day_arm_a_runtime_day_mismatch_blocked(tmp_path):
    store = ForwardStudyStore(tmp_path)
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-01", arm_a_runtime=runtime_payload("2020-01-02"))
    assert excinfo.value.reason == "ARM_A_RUNTIME_DAY_MISMATCH"


def test_write_day_arm_b_runtime_day_mismatch_blocked(tmp_path):
    store = ForwardStudyStore(tmp_path)
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-01", arm_b_runtime=runtime_payload("2020-01-02"))
    assert excinfo.value.reason == "ARM_B_RUNTIME_DAY_MISMATCH"


# ---------------------------------------------------------------------------
# verify_forward_store
# ---------------------------------------------------------------------------


def test_verify_forward_store_pass_multi_day(tmp_path):
    store = ForwardStudyStore(tmp_path)
    for day in ("2020-01-01", "2020-01-02", "2020-01-03"):
        write(store, day)
    result = verify_forward_store(tmp_path, ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert result["status"] == "PASS"
    assert result["day_count"] == 3
    assert result["verified_days"] == ["2020-01-01", "2020-01-02", "2020-01-03"]


@pytest.mark.parametrize(
    "filename",
    [DAY_FILE_PRICE, DAY_FILE_CANDIDATE, DAY_FILE_MARKET_GATE, DAY_FILE_ARM_A, DAY_FILE_ARM_B],
)
def test_verify_forward_store_snapshot_tamper_detected(tmp_path, filename):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    path = store.days_root / "2020-01-01" / filename
    path.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        verify_forward_store(tmp_path, ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert excinfo.value.reason.startswith("SNAPSHOT_HASH_MISMATCH:")


def test_verify_forward_store_checkpoint_tamper_detected(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    path = store.days_root / "2020-01-01" / DAY_FILE_CHECKPOINT
    record = json.loads(path.read_text(encoding="utf-8"))
    record["current_checkpoint_sha256"] = "f" * 64
    path.write_text(canonical_json_bytes(record).decode("utf-8"), encoding="utf-8")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        verify_forward_store(tmp_path, ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert excinfo.value.reason.startswith("CHECKPOINT_HASH_MISMATCH:")


def test_verify_forward_store_previous_chain_tamper_detected(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    write(store, "2020-01-02")
    path = store.days_root / "2020-01-02" / DAY_FILE_CHECKPOINT
    record = json.loads(path.read_text(encoding="utf-8"))
    record["previous_checkpoint_sha256"] = "e" * 64
    record["current_checkpoint_sha256"] = sha256_bytes(
        canonical_json_bytes({k: v for k, v in record.items() if k != "current_checkpoint_sha256"})
    )
    path.write_text(canonical_json_bytes(record).decode("utf-8"), encoding="utf-8")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        verify_forward_store(tmp_path, ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert excinfo.value.reason.startswith("CHECKPOINT_CHAIN_MISMATCH:")


def test_verify_forward_store_activation_manifest_expected_mismatch(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        verify_forward_store(tmp_path, OTHER_ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert excinfo.value.reason.startswith("ACTIVATION_MANIFEST_MISMATCH:")


def test_verify_forward_store_collector_commit_expected_mismatch(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        verify_forward_store(tmp_path, ACTIVATION_SHA, OTHER_COLLECTOR_COMMIT)
    assert excinfo.value.reason.startswith("COLLECTOR_COMMIT_MISMATCH:")


def test_verify_forward_store_empty_root_pass(tmp_path):
    result = verify_forward_store(tmp_path, ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert result == {
        "status": "PASS",
        "verified_days": [],
        "day_count": 0,
        "latest_checkpoint_sha256": None,
    }


def test_verify_forward_store_rejects_invalid_expected_sha(tmp_path):
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        verify_forward_store(tmp_path, "not-a-sha", COLLECTOR_COMMIT)
    assert excinfo.value.reason == "ACTIVATION_MANIFEST_SHA_INVALID"


def test_verify_forward_store_rejects_invalid_expected_commit(tmp_path):
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        verify_forward_store(tmp_path, ACTIVATION_SHA, "short")
    assert excinfo.value.reason == "COLLECTOR_COMMIT_INVALID"


# ---------------------------------------------------------------------------
# Partial / staging remnants
# ---------------------------------------------------------------------------


def _leave_staging_remnant(store: ForwardStudyStore, day: str) -> Path:
    staging = store.days_root / f"{day}.staging-remnant"
    staging.mkdir()
    (staging / DAY_FILE_PRICE).write_text("{}", encoding="utf-8")
    return staging


def test_staging_remnant_blocks_write(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    _leave_staging_remnant(store, "2020-01-02")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-02")
    assert excinfo.value.reason == "PARTIAL_DAY_COMMIT"


def test_staging_remnant_blocks_verify(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    _leave_staging_remnant(store, "2020-01-02")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        verify_forward_store(tmp_path, ACTIVATION_SHA, COLLECTOR_COMMIT)
    assert excinfo.value.reason == "PARTIAL_DAY_COMMIT"


def test_staging_remnant_blocks_load_latest_runtime(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01")
    _leave_staging_remnant(store, "2020-01-02")
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        load_latest_runtime(tmp_path)
    assert excinfo.value.reason == "PARTIAL_DAY_COMMIT"


# ---------------------------------------------------------------------------
# Atomic fault injection
# ---------------------------------------------------------------------------


class _InjectedFailure(RuntimeError):
    pass


def _failing_at(stage: str):
    def _inject(current_stage: str) -> None:
        if current_stage == stage:
            raise _InjectedFailure(stage)
    return _inject


@pytest.mark.parametrize(
    "stage",
    ["after_price_write", "after_candidate_write", "after_arm_a_write", "before_checkpoint_write"],
)
def test_atomic_failure_leaves_final_day_unpublished(tmp_path, stage):
    store = ForwardStudyStore(tmp_path)
    with pytest.raises(_InjectedFailure):
        write(store, "2020-01-01", fault_injector=_failing_at(stage))
    assert not (store.days_root / "2020-01-01").exists()


def test_atomic_failure_leaves_staging_remnant_detectable(tmp_path):
    store = ForwardStudyStore(tmp_path)
    with pytest.raises(_InjectedFailure):
        write(store, "2020-01-01", fault_injector=_failing_at("after_price_write"))
    with pytest.raises(V7ForwardPersistenceBlocked) as excinfo:
        write(store, "2020-01-02")
    assert excinfo.value.reason == "PARTIAL_DAY_COMMIT"


def test_atomic_failure_does_not_mutate_past_complete_day(tmp_path):
    store = ForwardStudyStore(tmp_path)
    first = write(store, "2020-01-01")
    checkpoint_path = store.days_root / "2020-01-01" / DAY_FILE_CHECKPOINT
    before = checkpoint_path.read_bytes()
    with pytest.raises(_InjectedFailure):
        write(store, "2020-01-02", fault_injector=_failing_at("after_candidate_write"))
    after = checkpoint_path.read_bytes()
    assert before == after
    assert first["status"] == "COMPLETE"


# ---------------------------------------------------------------------------
# load_latest_runtime
# ---------------------------------------------------------------------------


def test_load_latest_runtime_none_when_empty(tmp_path):
    assert load_latest_runtime(tmp_path) is None


def test_load_latest_runtime_returns_latest_complete_day_only(tmp_path):
    store = ForwardStudyStore(tmp_path)
    write(store, "2020-01-01", arm_a_runtime=runtime_payload("2020-01-01", "old"))
    write(store, "2020-01-02", arm_a_runtime=runtime_payload("2020-01-02", "new"))
    latest = load_latest_runtime(tmp_path)
    assert latest["day"] == "2020-01-02"
    assert latest["arm_a_runtime"]["tag"] == "new"
