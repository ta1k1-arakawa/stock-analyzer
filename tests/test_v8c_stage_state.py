from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src import v8c_human_gate_consumption as gate_consumption
from src import v8c_stage_state as state

FROZEN_DESIGN_COMMIT = "a" * 40
REVIEWED_IMPLEMENTATION_COMMIT = "b" * 40
CLASSIFIER_BLOB_SHA = "c" * 40


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def _authority():
    return {"binding": "synthetic"}


def _consume(state_root, stage, identity):
    gate = state.STAGE_READINESS_GATE[stage]
    gate_consumption.consume_gate_once(
        state_root, gate, FROZEN_DESIGN_COMMIT, clock=clock_stub, authorization_identity=identity,
    )


def _write(state_root, *, stage="T1C", result="PASS", sentinel_pass_count=3, identity="AUTH-1", **overrides):
    kwargs = dict(
        stage=stage, result=result, frozen_design_commit=FROZEN_DESIGN_COMMIT,
        reviewed_implementation_commit=REVIEWED_IMPLEMENTATION_COMMIT,
        sentinel_indices=[0, 149, 299], probe_start="2025-12-01", probe_end_exclusive="2025-12-08",
        classifier_blob_sha=CLASSIFIER_BLOB_SHA, authority_prerequisites=_authority(),
        sentinel_count=3, sentinel_pass_count=sentinel_pass_count,
        human_authorization_identity=identity, consumption_state_root=state_root,
        clock_text="2026-08-14T00:00:00Z",
    )
    kwargs.update(overrides)
    return state.write_readiness_pass(state_root, **kwargs)


def test_readiness_pass_is_restart_safe_and_privacy_safe(tmp_path):
    _consume(tmp_path, "T1C", "AUTH-1")
    evidence = _write(tmp_path)
    assert state.read_valid_readiness_pass(
        tmp_path, stage="T1C", frozen_design_commit=FROZEN_DESIGN_COMMIT,
        reviewed_implementation_commit=REVIEWED_IMPLEMENTATION_COMMIT,
        classifier_blob_sha=CLASSIFIER_BLOB_SHA, authority_prerequisites=_authority(),
    ) == evidence
    assert "ticker" not in (tmp_path / "v8c_readiness_pass_T1C.json").read_text().lower()


def test_readiness_pass_tampering_blocks(tmp_path):
    _consume(tmp_path, "T2", "AUTH-1")
    _write(tmp_path, stage="T2")
    path = tmp_path / "v8c_readiness_pass_T2.json"
    raw = path.read_text().replace('"result":"PASS"', '"result":"BLOCK"')
    path.write_text(raw)
    with pytest.raises(state.V8CStageEvidenceBlocked):
        state.read_valid_readiness_pass(
            tmp_path, stage="T2", frozen_design_commit=FROZEN_DESIGN_COMMIT,
            reviewed_implementation_commit=REVIEWED_IMPLEMENTATION_COMMIT,
            classifier_blob_sha=CLASSIFIER_BLOB_SHA, authority_prerequisites=_authority(),
        )


# ---------------------------------------------------------------------------
# HIGH-2: readiness durable evidence must be authority-bound, never
# manufacturable from known public values alone.
# ---------------------------------------------------------------------------


def test_write_requires_a_real_consumed_gate_receipt(tmp_path):
    """No gate consumption ever occurred for this identity: even though
    every other value is a real, publicly-derivable constant, the writer
    must independently verify the gate was actually consumed."""
    with pytest.raises(state.V8CStageEvidenceBlocked) as excinfo:
        _write(tmp_path, identity="NEVER-CONSUMED")
    assert excinfo.value.reason == "STAGE_EVIDENCE_NO_MATCHING_CONSUMED_GATE_RECEIPT"


def test_write_with_wrong_receipt_identity_blocks(tmp_path):
    _consume(tmp_path, "T1C", "AUTH-REAL")
    with pytest.raises(state.V8CStageEvidenceBlocked) as excinfo:
        _write(tmp_path, identity="AUTH-CLAIMED")
    assert excinfo.value.reason == "STAGE_EVIDENCE_NO_MATCHING_CONSUMED_GATE_RECEIPT"


def test_block_result_is_durably_recorded(tmp_path):
    _consume(tmp_path, "T1C", "AUTH-1")
    evidence = _write(tmp_path, result="BLOCK", sentinel_pass_count=2)
    assert evidence["result"] == "BLOCK"
    with pytest.raises(state.V8CStageEvidenceBlocked) as excinfo:
        state.read_valid_readiness_pass(
            tmp_path, stage="T1C", frozen_design_commit=FROZEN_DESIGN_COMMIT,
            reviewed_implementation_commit=REVIEWED_IMPLEMENTATION_COMMIT,
            classifier_blob_sha=CLASSIFIER_BLOB_SHA, authority_prerequisites=_authority(),
        )
    assert excinfo.value.reason == "V8C_READINESS_LATEST_RESULT_NOT_PASS"


def test_pass_requires_all_three_sentinels(tmp_path):
    _consume(tmp_path, "T1C", "AUTH-1")
    with pytest.raises(state.V8CStageEvidenceBlocked) as excinfo:
        _write(tmp_path, result="PASS", sentinel_pass_count=2)
    assert excinfo.value.reason == "STAGE_EVIDENCE_PASS_REQUIRES_ALL_SENTINELS"


def test_pass_after_block_at_same_identity_replay_still_blocked_by_gate_replay(tmp_path):
    """A later authorized BLOCK must not leave an older PASS usable: once
    BLOCK is recorded, only a fresh authorized execution (a distinct
    authorization identity, since readiness gates are per-authorization)
    can produce a new usable PASS at this destination."""
    _consume(tmp_path, "T1C", "AUTH-1")
    _write(tmp_path, result="PASS", sentinel_pass_count=3, identity="AUTH-1")
    assert state.read_valid_readiness_pass(
        tmp_path, stage="T1C", frozen_design_commit=FROZEN_DESIGN_COMMIT,
        reviewed_implementation_commit=REVIEWED_IMPLEMENTATION_COMMIT,
        classifier_blob_sha=CLASSIFIER_BLOB_SHA, authority_prerequisites=_authority(),
    )["result"] == "PASS"

    _consume(tmp_path, "T1C", "AUTH-2")
    _write(tmp_path, result="BLOCK", sentinel_pass_count=1, identity="AUTH-2")
    with pytest.raises(state.V8CStageEvidenceBlocked) as excinfo:
        state.read_valid_readiness_pass(
            tmp_path, stage="T1C", frozen_design_commit=FROZEN_DESIGN_COMMIT,
            reviewed_implementation_commit=REVIEWED_IMPLEMENTATION_COMMIT,
            classifier_blob_sha=CLASSIFIER_BLOB_SHA, authority_prerequisites=_authority(),
        )
    assert excinfo.value.reason == "V8C_READINESS_LATEST_RESULT_NOT_PASS"

    _consume(tmp_path, "T1C", "AUTH-3")
    _write(tmp_path, result="PASS", sentinel_pass_count=3, identity="AUTH-3")
    assert state.read_valid_readiness_pass(
        tmp_path, stage="T1C", frozen_design_commit=FROZEN_DESIGN_COMMIT,
        reviewed_implementation_commit=REVIEWED_IMPLEMENTATION_COMMIT,
        classifier_blob_sha=CLASSIFIER_BLOB_SHA, authority_prerequisites=_authority(),
    )["result"] == "PASS"


def test_missing_readiness_execution_blocks_read(tmp_path):
    with pytest.raises(state.V8CStageEvidenceBlocked) as excinfo:
        state.read_valid_readiness_pass(
            tmp_path, stage="T1C", frozen_design_commit=FROZEN_DESIGN_COMMIT,
            reviewed_implementation_commit=REVIEWED_IMPLEMENTATION_COMMIT,
            classifier_blob_sha=CLASSIFIER_BLOB_SHA, authority_prerequisites=_authority(),
        )
    assert excinfo.value.reason == "V8C_READINESS_PASS_MISSING"


def test_t2_recheck_pass_requires_all_derived_conditions(tmp_path):
    safe = {
        "result": "PASS", "block": "T2", "recheck_point": "recheck_2",
        "frozen_design_commit": "a" * 40,
        "reviewed_implementation_commit": "b" * 40,
        "t2_real_data_acquired": False, "t2_opened": False,
        "t2_research_access_count": 0, "t2_features_observed": False,
        "t2_outcomes_observed": False, "t2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
    }
    state.write_t2_recheck_pass(tmp_path, safe)
    assert state.read_valid_t2_recheck_pass(
        tmp_path, frozen_design_commit="a" * 40, reviewed_implementation_commit="b" * 40
    )["result"] == "PASS"
    path = tmp_path / state.T2_RECHECK_PASS_FILENAME
    path.write_text(path.read_text().replace('"t2_opened":false', '"t2_opened":true'))
    with pytest.raises(state.V8CStageEvidenceBlocked):
        state.read_valid_t2_recheck_pass(
            tmp_path, frozen_design_commit="a" * 40, reviewed_implementation_commit="b" * 40
        )


def test_t2_recheck_pass_writer_rejects_wrong_condition_value(tmp_path):
    """The writer is not a generic pass-through for an arbitrary caller-
    supplied mapping: a condition set to the wrong value must BLOCK before
    ever being durably recorded, even if ``result``/``recheck_point`` are
    correct."""
    safe = {
        "result": "PASS", "block": "T2", "recheck_point": "recheck_2",
        "frozen_design_commit": "a" * 40,
        "reviewed_implementation_commit": "b" * 40,
        "t2_real_data_acquired": False, "t2_opened": True,  # tampered
        "t2_research_access_count": 0, "t2_features_observed": False,
        "t2_outcomes_observed": False, "t2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
    }
    with pytest.raises(state.V8CStageEvidenceBlocked) as excinfo:
        state.write_t2_recheck_pass(tmp_path, safe)
    assert excinfo.value.reason == "V8C_T2_RECHECK_PASS_CONDITION_INVALID:t2_opened"
