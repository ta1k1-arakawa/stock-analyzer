from __future__ import annotations

import inspect
import json

import pytest

from src import v8e_t2_prefreeze_preservation as recheck


def _safe(**overrides):
    value = {
        "T2_real_data_acquired": False,
        "T2_opened": False,
        "T2_research_access_count": 0,
        "T2_features_observed": False,
        "T2_outcomes_observed": False,
        "T2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
        "v8_trusted_partition_git_blob": recheck.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "original_v8_partition_manifest_sha256": recheck.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": recheck.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "t2_count": recheck.EXPECTED_T2_COUNT,
        "t2_ticker_list_sha256": recheck.EXPECTED_T2_TICKER_LIST_SHA256,
    }
    value.update(overrides)
    return value


def _state():
    return {
        "T2": {
            "raw_data_acquired": False,
            "opened_for_research": False,
            "sealed_holdout_access_count": None,
        },
        "trust_anchor_pinning": {"block_assignments_exposed": False},
        "partition": {
            "manifest_sha256": recheck.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
            "partition_implementation_git_commit": recheck.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
            "block_size_frozen": recheck.EXPECTED_T2_COUNT,
            "t2_ticker_list_sha256": recheck.EXPECTED_T2_TICKER_LIST_SHA256,
        },
        "real_partition_build_history": {
            "partition_implementation_git_commit": recheck.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
            "manifest_sha256": recheck.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
            "t2_ticker_list_sha256": recheck.EXPECTED_T2_TICKER_LIST_SHA256,
            "block_assignments_exposed": False,
            "retry_performed": False,
        },
        "real_data_acquired": False,
        "backtests": 0,
        "models_fitted": 0,
        "profit_calculated": 0,
    }


def _bridge():
    return {
        "t2_acquired_before_authorized_acquisition": False,
        "t2_research_open_count_before_official_opening": 0,
        "t2_membership_reassignment": "PROHIBITED",
        "v8_trusted_partition_json_mutated_or_repinned": False,
    }


def _anchor():
    return {
        "authorized_partition_manifest_sha256": recheck.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "authorized_partition_implementation_git_commit": recheck.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    }


def test_all_nine_conditions_pass_only_with_exact_safe_evidence():
    record = recheck.build_t2_prefreeze_record(_safe())
    result = recheck.verify_t2_prefreeze_record(record, safe_evidence=_safe())
    assert result["result"] == "PASS"
    assert result["nine_conditions_independently_verified"] is True
    assert record["reviewed_v8e_design_candidate_commit"] == recheck.V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT


@pytest.mark.parametrize("field", recheck.T2_SAFE_CONDITION_FIELDS)
def test_each_nine_condition_failure_blocks(field):
    bad = _safe()
    if field == "T2_research_access_count":
        bad[field] = 1
    elif field.endswith("compatible") or field == "data_quality_policy_unchanged":
        bad[field] = False
    else:
        bad[field] = True
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        recheck.build_t2_prefreeze_record(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("v8_trusted_partition_git_blob", "0" * 40),
        ("original_v8_partition_manifest_sha256", "0" * 64),
        ("parent_v8_partition_implementation_commit", "0" * 40),
        ("t2_count", 299),
        ("t2_ticker_list_sha256", "0" * 64),
    ],
)
def test_t2_provenance_count_and_hash_mismatch_blocks(field, value):
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        recheck.build_t2_prefreeze_record(_safe(**{field: value}))


def test_record_missing_extra_and_duplicate_fields_block():
    record = recheck.build_t2_prefreeze_record(_safe())
    missing = dict(record)
    del missing["OVERALL_RESULT"]
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        recheck.verify_t2_prefreeze_record(missing, safe_evidence=_safe())
    extra = dict(record)
    extra["extra"] = True
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        recheck.verify_t2_prefreeze_record(extra, safe_evidence=_safe())
    raw = json.dumps(record, separators=(",", ":")).replace(
        '"OVERALL_RESULT":"PASS"', '"OVERALL_RESULT":"PASS","OVERALL_RESULT":"PASS"'
    ).encode("utf-8")
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked) as excinfo:
        recheck.verify_t2_prefreeze_record_bytes(raw, safe_evidence=_safe())
    assert excinfo.value.reason == "V8E_T2_RECORD_DUPLICATE_KEY"


def test_candidate_mismatch_blocks_even_when_record_otherwise_passes():
    record = recheck.build_t2_prefreeze_record(_safe())
    record["reviewed_v8e_design_candidate_commit"] = "0" * 40
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        recheck.verify_t2_prefreeze_record(record, safe_evidence=_safe())


def test_producer_declared_pass_is_insufficient_without_matching_independent_evidence():
    record = recheck.build_t2_prefreeze_record(_safe())
    safe = _safe(T2_opened=True)
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        recheck.verify_t2_prefreeze_record(record, safe_evidence=safe)


def test_safe_git_evidence_resolver_passes_synthetic_public_evidence():
    blobs = {
        "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md": recheck.V8E_DESIGN_CANDIDATE_BLOB_SHA,
        recheck.V8_STATE_GIT_PATH: recheck.V8_STATE_BLOB_SHA,
        recheck.V8B_T2_AUTHORITY_BRIDGE_GIT_PATH: recheck.V8B_T2_AUTHORITY_BRIDGE_BLOB_SHA,
        recheck.V8C_READINESS_ADJUDICATION_GIT_PATH: recheck.V8C_READINESS_ADJUDICATION_BLOB_SHA,
        recheck.V8_TRUSTED_PARTITION_GIT_PATH: recheck.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
    }
    safe = recheck._resolve_t2_prefreeze_safe_evidence_with_dependencies(
        "synthetic-repo",
        verified_head="1" * 40,
        git_blob_resolver=lambda root, commit, path: blobs[path],
        safe_state_reader=lambda root, commit, reader: _state(),
        safe_bridge_reader=lambda root, commit, reader: _bridge(),
        trusted_anchor_reader=lambda root, commit: _anchor(),
    )
    assert safe["T2_real_data_acquired"] is False
    assert safe["t2_count"] == 300


@pytest.mark.parametrize(
    "mutation",
    [
        ("T2", "raw_data_acquired", True),
        ("bridge", "t2_research_open_count_before_official_opening", 1),
        ("partition", "t2_ticker_list_sha256", "0" * 64),
        ("history", "retry_performed", True),
    ],
)
def test_safe_git_evidence_mismatch_blocks(mutation):
    state = _state()
    bridge = _bridge()
    if mutation[0] == "T2":
        state["T2"][mutation[1]] = mutation[2]
    elif mutation[0] == "bridge":
        bridge[mutation[1]] = mutation[2]
    else:
        key = "partition" if mutation[0] == "partition" else "real_partition_build_history"
        state[key][mutation[1]] = mutation[2]
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        recheck._derive_safe_evidence(state, bridge, _anchor())


def test_no_private_t2_read_path_or_artifact_writer_exists():
    source = inspect.getsource(recheck)
    assert "partition_manifest_path" not in source
    assert "private_reader" not in source
    assert "read_bytes" not in source
    assert "write_bytes" not in source
    assert "open(" not in source
