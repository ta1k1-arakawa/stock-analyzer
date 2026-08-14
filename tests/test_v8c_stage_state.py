from __future__ import annotations

import pytest

from src import v8c_stage_state as state


def _authority():
    return {"binding": "synthetic"}


def test_readiness_pass_is_restart_safe_and_privacy_safe(tmp_path):
    evidence = state.write_readiness_pass(
        tmp_path,
        stage="T1C",
        frozen_design_commit="a" * 40,
        reviewed_implementation_commit="b" * 40,
        sentinel_indices=[0, 149, 299],
        probe_start="2025-12-01",
        probe_end_exclusive="2025-12-08",
        classifier_blob_sha="c" * 40,
        authority_prerequisites=_authority(),
        clock_text="2026-08-14T00:00:00Z",
    )
    assert state.read_valid_readiness_pass(
        tmp_path,
        stage="T1C",
        frozen_design_commit="a" * 40,
        reviewed_implementation_commit="b" * 40,
        classifier_blob_sha="c" * 40,
        authority_prerequisites=_authority(),
    ) == evidence
    assert "ticker" not in (tmp_path / "v8c_readiness_pass_T1C.json").read_text().lower()


def test_readiness_pass_tampering_blocks(tmp_path):
    state.write_readiness_pass(
        tmp_path, stage="T2", frozen_design_commit="a" * 40,
        reviewed_implementation_commit="b" * 40, sentinel_indices=[0, 149, 299],
        probe_start="2025-12-01", probe_end_exclusive="2025-12-08",
        classifier_blob_sha="c" * 40, authority_prerequisites=_authority(),
        clock_text="2026-08-14T00:00:00Z",
    )
    path = tmp_path / "v8c_readiness_pass_T2.json"
    raw = path.read_text().replace('"result":"PASS"', '"result":"BLOCK"')
    path.write_text(raw)
    with pytest.raises(state.V8CStageEvidenceBlocked):
        state.read_valid_readiness_pass(
            tmp_path, stage="T2", frozen_design_commit="a" * 40,
            reviewed_implementation_commit="b" * 40, classifier_blob_sha="c" * 40,
            authority_prerequisites=_authority(),
        )


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
