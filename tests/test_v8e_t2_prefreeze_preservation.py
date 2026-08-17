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
        "schema_version": "V8_STATE_SNAPSHOT_V1",
        "study": "V8_HISTORICAL_RESEARCH",
        "T2": {
            "raw_data_acquired": False,
            "opened_for_research": False,
            "real_acquisition_authorized": False,
            "sealed_holdout_access_count": None,
            "research_access_authorized": None,
            "ticker_count_frozen": recheck.EXPECTED_T2_COUNT,
        },
        "trust_anchor_pinning": {"block_assignments_exposed": False},
        "partition": {
            "manifest_sha256": recheck.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
            "partition_implementation_git_commit": recheck.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
            "manifest_schema_version": "V8_PARTITION_MANIFEST_V3",
            "block_size_frozen": recheck.EXPECTED_T2_COUNT,
            "t2_ticker_list_sha256": recheck.EXPECTED_T2_TICKER_LIST_SHA256,
            "block_assignments_recorded": False,
        },
        "real_partition_build_history": [
            {
                "authorized_implementation_head": recheck.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
                "mode": "PRODUCTION",
                "process_result": "PASS",
                "exit_code": 0,
                "source_reproduction_status": "PASS",
                "t0_reproduction_status": "PASS",
                "partition_manifest_written": True,
                "real_block_assignments_created": True,
                "manifest_sha256": recheck.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
                "partition_implementation_git_commit": recheck.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
                "manifest_schema_version": "V8_PARTITION_MANIFEST_V3",
                "block_sizes": {"T0": 300, "T1": 300, "T2": 300, "T3": 300, "T_spare": 1904},
                "t1_ticker_list_sha256": "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d",
                "t2_ticker_list_sha256": recheck.EXPECTED_T2_TICKER_LIST_SHA256,
                "one_time_authorization_consumed": True,
                "retry_performed": False,
                "raw_jpx_bytes_persisted": False,
                "block_assignments_exposed": False,
            }
        ],
        "real_data_acquired": False,
        "backtests": 0,
        "models_fitted": 0,
        "profit_calculated": 0,
    }


def _bridge():
    return {
        "schema_version": "V8B_T2_AUTHORITY_BRIDGE_V1",
        "study": "V8B_HISTORICAL_RESEARCH",
        "role": "SEALED_HOLDOUT",
        "source_authority": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY",
        "v8_trust_anchor_git_path": "V8_TRUSTED_PARTITION.json",
        "v8_trust_anchor_git_identity": recheck.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "authorized_parent_v8_partition_manifest_sha256": recheck.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "expected_t2_ticker_list_sha256": recheck.EXPECTED_T2_TICKER_LIST_SHA256,
        "t2_acquired_before_authorized_acquisition": False,
        "t2_research_open_count_before_official_opening": 0,
        "v8b_frozen_design_commit": "eedf198b93185b963b825170ed0be97e93f923b7",
        "t2_membership_reassignment": "PROHIBITED",
        "v8_trusted_partition_json_mutated_or_repinned": False,
        "option": "OPTION_2",
        "human_gate": "V8B_T2_AUTHORITY_INTEGRATION_OPTION_2_APPROVED",
        "authorization_note": "This bridge never mutates, reinterprets, or re-pins the V8 trust anchor.",
    }


def _anchor():
    return {
        "schema_version": "V8_TRUSTED_PARTITION_V1",
        "study_name": "V8_HISTORICAL_RESEARCH",
        "design_commit": "c414d3191cba356734d7ed08bdf1abc7d51fc384",
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": recheck.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "authorized_partition_implementation_git_commit": recheck.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "authorization_note": "This trust-anchor authorization does NOT authorize T2 acquisition.",
    }


def _historical_t2():
    return {
        "study": "V8D_HISTORICAL_RESEARCH",
        "document_type": "T2_PREFREEZE_PRESERVATION_RECHECK_AUDIT_RECORD",
        "reviewed_design_candidate_commit": "eda657cde2383718d986c4c4bfaae794784fe04d",
        "checkpoint": "V8D_T2_PREFREEZE_PRESERVATION_RECHECK",
        "recheck_1": "before_V8D_design_freeze",
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
        "T2_PREFREEZE_PRESERVATION_RECHECK": "PASS",
        "OVERALL_RESULT": "PASS",
    }


def _terminal():
    return {
        "study": "V8D_HISTORICAL_RESEARCH",
        "terminal_status": "BLOCK_CLOSED",
        "failure_class": "DESIGN_AUDITABILITY_FAILURE",
        "terminal_implementation_head": "a862efec34dcf4a89005c88b55b35c39be12b7bc",
        "t2_features_observed": False,
        "t2_outcomes_observed": False,
    }


def _text_block(values):
    return ("```text\n" + "\n".join(f"{key}={value}" for key, value in values.items()) + "\n```\n").encode()


def _historical_t2_bytes():
    values = _historical_t2()
    return _text_block({key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in values.items()})


def _terminal_bytes():
    values = {key: value for key, value in _terminal().items() if key not in {"t2_features_observed", "t2_outcomes_observed"}}
    return _text_block(values) + b"No T1C/T2 outcomes or features were observed.\n"


def _design_bytes():
    return (
        b"policy=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE\n"
        b"invalid_fraction_threshold=1/252\n"
        b"max_consecutive_invalid_returned_rows=1\n"
        b"full_P_hist_check=true\n"
        b"threshold_failure_action=BLOCK_WHOLE_ACQUISITION\n"
    )


def _readiness_bytes():
    return b"T2_ACCESS=PROHIBITED\nT1C_RAW_ACQUISITION=PROHIBITED\nT1C_RESEARCH_OPENING=PROHIBITED\nsuccessor_study_required=true\nreadiness_result=BLOCK\n"


def _runtime(sha="1" * 40, **overrides):
    value = {
        "resolved_support_sha": sha,
        "branch": recheck.V8E_PRODUCTION_BRANCH,
        "head": sha,
        "origin_head": sha,
        "worktree_clean": True,
        "origin_url": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "commits_after_reviewed_support_sha": 0,
    }
    value.update(overrides)
    return value


def _chronology():
    paths = sorted(recheck.V8E_EXPECTED_PREFREEZE_CHRONOLOGY_PATHS)
    return [{"commit": f"{index + 1:040x}", "paths": [path]} for index, path in enumerate(paths)]


def _resolver_kwargs(**overrides):
    reviewed_sha = "1" * 40
    blobs = {
        "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md": recheck.V8E_DESIGN_CANDIDATE_BLOB_SHA,
        recheck.V8_STATE_GIT_PATH: recheck.V8_STATE_BLOB_SHA,
        recheck.V8B_T2_AUTHORITY_BRIDGE_GIT_PATH: recheck.V8B_T2_AUTHORITY_BRIDGE_BLOB_SHA,
        recheck.V8C_READINESS_ADJUDICATION_GIT_PATH: recheck.V8C_READINESS_ADJUDICATION_BLOB_SHA,
        recheck.V8_TRUSTED_PARTITION_GIT_PATH: recheck.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        recheck.V8D_TERMINAL_GIT_PATH: recheck.V8D_TERMINAL_BLOB_SHA,
        recheck.V8D_T2_PREFREEZE_GIT_PATH: recheck.V8D_T2_PREFREEZE_BLOB_SHA,
    }

    def object_reader(root, commit, path):
        if path == recheck.V8D_T2_PREFREEZE_GIT_PATH:
            return _historical_t2_bytes()
        if path == recheck.V8D_TERMINAL_GIT_PATH:
            return _terminal_bytes()
        if path == recheck.V8C_READINESS_ADJUDICATION_GIT_PATH:
            return _readiness_bytes()
        if path == "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md":
            return _design_bytes()
        raise AssertionError(path)

    defaults = {
        "verified_head": reviewed_sha,
        "reviewed_support_implementation_sha": reviewed_sha,
        "git_blob_resolver": lambda root, commit, path: blobs[path],
        "git_object_reader": object_reader,
        "safe_state_reader": lambda root, commit, reader: _state(),
        "safe_bridge_reader": lambda root, commit, reader: _bridge(),
        "trusted_anchor_reader": lambda root, commit: _anchor(),
        "runtime_state_reader": lambda root, sha: _runtime(sha),
        "chronology_reader": lambda root, lower, upper: _chronology(),
        "commit_ancestor_reader": lambda root, commit, candidate: True,
    }
    defaults.update(overrides)
    return defaults


def _resolve(**overrides):
    return recheck._resolve_t2_prefreeze_safe_evidence_with_dependencies(
        "synthetic-repo", **_resolver_kwargs(**overrides)
    )


def _derived(**overrides):
    values = {
        "state": _state(),
        "bridge": _bridge(),
        "anchor": _anchor(),
        "historical_t2": _historical_t2(),
        "terminal": _terminal(),
        "readiness": {"t2_access_prohibited": True},
        "design": {"policy_unchanged": True},
    }
    values.update(overrides)
    return recheck._derive_safe_evidence(**values)


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


def test_real_history_shape_is_exactly_one_mapping():
    assert _derived()["t2_count"] == 300
    for bad in ([], [_state()["real_partition_build_history"][0], _state()["real_partition_build_history"][0]], ["not-a-mapping"]):
        state = _state()
        state["real_partition_build_history"] = bad
        with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
            _derived(state=state)
    state = _state()
    state["real_partition_build_history"] = dict(state["real_partition_build_history"][0])
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _derived(state=state)


@pytest.mark.parametrize(
    "key,value",
    [
        ("partition_implementation_git_commit", "0" * 40),
        ("manifest_sha256", "0" * 64),
        ("t2_ticker_list_sha256", "0" * 64),
        ("retry_performed", True),
        ("block_assignments_exposed", True),
        ("block_sizes", {"T0": 300, "T1": 300, "T2": 299, "T3": 300, "T_spare": 1904}),
    ],
)
def test_history_frozen_fact_mismatch_blocks(key, value):
    state = _state()
    state["real_partition_build_history"][0][key] = value
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _derived(state=state)


def test_safe_git_evidence_resolver_passes_exact_reviewed_runtime_and_history():
    safe = _resolve()
    assert safe["T2_real_data_acquired"] is False
    assert safe["t2_count"] == 300


@pytest.mark.parametrize(
    "runtime,reviewed,head",
    [
        (_runtime("1" * 40), "z" * 40, "1" * 40),
        (_runtime("2" * 40), "1" * 40, "1" * 40),
        (_runtime("1" * 40, origin_head="2" * 40), "1" * 40, "1" * 40),
        (_runtime("1" * 40, worktree_clean=False), "1" * 40, "1" * 40),
        (_runtime("1" * 40, commits_after_reviewed_support_sha=1), "1" * 40, "1" * 40),
    ],
)
def test_reviewed_support_runtime_binding_fail_closed(runtime, reviewed, head):
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _resolve(
            reviewed_support_implementation_sha=reviewed,
            verified_head=head,
            runtime_state_reader=lambda root, sha, value=runtime: value,
        )


def test_chronology_rejects_unexpected_path_anywhere_in_history():
    records = _chronology()
    records[0]["paths"] = ["V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md", "unexpected.txt"]
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _resolve(chronology_reader=lambda root, lower, upper: records)


def test_chronology_rejects_add_then_delete_unexpected_path():
    records = _chronology() + [
        {"commit": "a" * 40, "paths": ["unexpected.txt"]},
        {"commit": "b" * 40, "paths": ["unexpected.txt"]},
    ]
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _resolve(chronology_reader=lambda root, lower, upper: records)


def test_chronology_rejects_design_change_after_reviewed_candidate():
    records = _chronology()
    design_record = next(record for record in records if "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md" in record["paths"])
    design_record["commit"] = "5" * 40
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _resolve(
            chronology_reader=lambda root, lower, upper: records,
            commit_ancestor_reader=lambda root, commit, candidate: commit != "5" * 40,
        )


@pytest.mark.parametrize("path", [recheck.V8D_TERMINAL_GIT_PATH, recheck.V8D_T2_PREFREEZE_GIT_PATH])
def test_v8d_historical_blob_binding_is_exact(path):
    defaults = _resolver_kwargs()
    original = defaults["git_blob_resolver"]
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _resolve(git_blob_resolver=lambda root, commit, candidate: "0" * 40 if candidate == path else original(root, commit, candidate))


@pytest.mark.parametrize("kind", ["state", "bridge", "readiness", "anchor"])
def test_current_safe_provenance_mismatch_blocks(kind):
    kwargs = {}
    if kind == "state":
        bad = _state()
        bad["partition"]["manifest_sha256"] = "0" * 64
        kwargs["safe_state_reader"] = lambda root, commit, reader, bad=bad: bad
    elif kind == "bridge":
        bad = _bridge()
        bad["v8_trusted_partition_json_mutated_or_repinned"] = True
        kwargs["safe_bridge_reader"] = lambda root, commit, reader, bad=bad: bad
    elif kind == "readiness":
        defaults = _resolver_kwargs()
        original = defaults["git_object_reader"]
        kwargs["git_object_reader"] = lambda root, commit, path: b"readiness_result=PASS\n" if path == recheck.V8C_READINESS_ADJUDICATION_GIT_PATH else original(root, commit, path)
    else:
        bad = _anchor()
        bad["authorized_partition_manifest_sha256"] = "0" * 64
        kwargs["trusted_anchor_reader"] = lambda root, commit, bad=bad: bad
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _resolve(**kwargs)


@pytest.mark.parametrize("field,value", [("T2", ("raw_data_acquired", True)), ("bridge", ("t2_research_open_count_before_official_opening", 1)), ("partition", ("t2_ticker_list_sha256", "0" * 64)), ("history", ("retry_performed", True))])
def test_public_safe_evidence_contradiction_blocks(field, value):
    state, bridge = _state(), _bridge()
    if field == "T2":
        state["T2"][value[0]] = value[1]
    elif field == "bridge":
        bridge[value[0]] = value[1]
    elif field == "history":
        state["real_partition_build_history"][0][value[0]] = value[1]
    else:
        state["partition"][value[0]] = value[1]
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _derived(state=state, bridge=bridge)


def test_design_candidate_blob_mismatch_blocks():
    defaults = _resolver_kwargs()
    original = defaults["git_blob_resolver"]
    with pytest.raises(recheck.V8ET2PrefreezePreservationBlocked):
        _resolve(git_blob_resolver=lambda root, commit, path: "0" * 40 if path == "V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md" else original(root, commit, path))


def test_no_private_t2_read_path_or_artifact_writer_exists():
    source = inspect.getsource(recheck)
    assert "partition_manifest_path" not in source
    assert "private_reader" not in source
    assert "read_bytes" not in source
    assert "write_bytes" not in source
    assert "open(" not in source
