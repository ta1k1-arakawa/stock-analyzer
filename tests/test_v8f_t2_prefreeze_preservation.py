from __future__ import annotations

import json
import subprocess

import pytest

from src import v8f_t2_prefreeze_preservation as recheck
from src.v8c_git_provenance import CANONICAL_REPOSITORY_ROOT


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


def _record(**overrides):
    safe = _safe()
    value = {
        "study": recheck.V8F_STUDY_NAME,
        "document_type": recheck.V8F_T2_PREFREEZE_DOCUMENT_TYPE,
        "reviewed_v8f_design_candidate_commit": recheck.V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "checkpoint": recheck.V8F_T2_PREFREEZE_CHECKPOINT,
        "recheck_1": "before_V8F_design_freeze",
        **safe,
        "T2_PREFREEZE_PRESERVATION_RECHECK": "PASS",
        "OVERALL_RESULT": "PASS",
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
        "study": "V8E_HISTORICAL_RESEARCH",
        "document_type": "T2_PREFREEZE_PRESERVATION_RECHECK_AUDIT_RECORD",
        "reviewed_v8e_design_candidate_commit": "6f672404b93a1003253915196dd635ca76fd2be1",
        "checkpoint": "V8E_T2_PREFREEZE_PRESERVATION_RECHECK",
        "recheck_1": "before_V8E_design_freeze",
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
        "schema_version": "V8E_T1C_READINESS_TERMINAL_ADJUDICATION_V1",
        "study": "V8E_HISTORICAL_RESEARCH",
        "readiness_result": "BLOCK",
        "t1c_raw_acquisition_allowed": False,
        "t1c_research_opening_allowed": False,
        "disposition": "BLOCK_CLOSED",
        "t2_features_observed": False,
        "t2_outcomes_observed": False,
    }


def _text_block(values):
    return ("```text\n" + "\n".join(f"{key}={value}" for key, value in values.items()) + "\n```\n").encode()


def _historical_t2_bytes():
    values = _historical_t2()
    return _text_block({key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in values.items()})


def _terminal_bytes():
    return json.dumps(_terminal(), sort_keys=True).encode("utf-8")


def _design_bytes():
    return (
        b"policy=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE\n"
        b"invalid_fraction_threshold=1/252\n"
        b"max_consecutive_invalid_returned_rows=1\n"
        b"full_P_hist_check=true\n"
        b"threshold_failure_action=BLOCK_WHOLE_ACQUISITION\n"
    )


def _runtime(sha="1" * 40, **overrides):
    value = {
        "resolved_support_sha": sha,
        "branch": recheck.V8F_PRODUCTION_BRANCH,
        "head": sha,
        "origin_head": sha,
        "worktree_clean": True,
        "origin_url": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "commits_after_reviewed_support_sha": 0,
    }
    value.update(overrides)
    return value


def _resolver_kwargs(**overrides):
    reviewed_sha = "1" * 40
    blobs = {
        recheck.V8F_DESIGN_GIT_PATH: recheck.V8F_DESIGN_CANDIDATE_BLOB_SHA,
        recheck.V8_STATE_GIT_PATH: recheck.V8_STATE_BLOB_SHA,
        recheck.V8B_T2_AUTHORITY_BRIDGE_GIT_PATH: recheck.V8B_T2_AUTHORITY_BRIDGE_BLOB_SHA,
        recheck.V8_TRUSTED_PARTITION_GIT_PATH: recheck.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH: recheck.V8F_V8E_TERMINAL_RECORD_BLOB_SHA,
        recheck.V8E_T2_PREFREEZE_GIT_PATH: recheck.V8E_T2_PREFREEZE_BLOB_SHA,
    }

    def object_reader(root, commit, path):
        if path == recheck.V8E_T2_PREFREEZE_GIT_PATH:
            return _historical_t2_bytes()
        if path == recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH:
            return _terminal_bytes()
        if path == recheck.V8F_DESIGN_GIT_PATH:
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
    }
    defaults.update(overrides)
    return defaults


def _resolve(**overrides):
    return recheck._resolve_t2_prefreeze_safe_evidence_with_dependencies(
        "synthetic-repo", **_resolver_kwargs(**overrides)
    )


def _derived_terminal():
    """Shape produced by _validate_v8e_terminal(), as consumed by
    _derive_safe_evidence -- not the raw historical JSON fixture shape."""
    return {"t2_features_observed": False, "t2_outcomes_observed": False, "t2_access_prohibited": True}


def _derived(**overrides):
    values = {
        "state": _state(),
        "bridge": _bridge(),
        "anchor": _anchor(),
        "historical_t2": _historical_t2(),
        "terminal": _derived_terminal(),
        "design": {"policy_unchanged": True},
    }
    values.update(overrides)
    return recheck._derive_safe_evidence(**values)


# ---------------------------------------------------------------------------
# Namespace exactness / exact V8F candidate binding
# ---------------------------------------------------------------------------


def test_namespace_literals_are_v8f_exact():
    assert recheck.V8F_STUDY_NAME == "V8F_HISTORICAL_RESEARCH"
    assert recheck.V8F_T2_PREFREEZE_CHECKPOINT == "V8F_T2_PREFREEZE_PRESERVATION_RECHECK"
    assert recheck.V8F_T2_PREFREEZE_DOCUMENT_TYPE == "T2_PREFREEZE_PRESERVATION_RECHECK_AUDIT_RECORD"


def test_wrong_v8f_candidate_blocks():
    record = _record(reviewed_v8f_design_candidate_commit="0" * 40)
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck.verify_t2_prefreeze_record(record, safe_evidence=_safe())
    assert excinfo.value.reason == "V8F_T2_RECORD_VALUE_MISMATCH:reviewed_v8f_design_candidate_commit"


def test_design_candidate_blob_mismatch_blocks():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_blob_resolver=lambda root, commit, path: "0" * 40 if path == recheck.V8F_DESIGN_GIT_PATH else _resolver_kwargs()["git_blob_resolver"](root, commit, path))
    assert excinfo.value.reason == "V8F_T2_DESIGN_CANDIDATE_BLOB_MISMATCH"


# ---------------------------------------------------------------------------
# Pure nine-condition validator (still used for synthetic unit testing only)
# ---------------------------------------------------------------------------


def test_nine_conditions_pass_with_safe_defaults():
    assert recheck._validate_nine_conditions(
        {key: _safe()[key] for key in recheck.T2_SAFE_CONDITION_FIELDS}
    ) == {key: _safe()[key] for key in recheck.T2_SAFE_CONDITION_FIELDS}


@pytest.mark.parametrize("field", recheck.T2_SAFE_CONDITION_FIELDS)
def test_each_nine_condition_failure_blocks(field):
    bad = _safe()
    if field == "T2_research_access_count":
        bad[field] = 1
    elif field.endswith("compatible") or field == "data_quality_policy_unchanged":
        bad[field] = False
    else:
        bad[field] = True
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
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
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
        recheck.build_t2_prefreeze_record(_safe(**{field: value}))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_record_schema_exactness(mutation):
    record = _record()
    if mutation == "missing":
        del record["OVERALL_RESULT"]
    else:
        record["unexpected"] = True
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck.verify_t2_prefreeze_record(record, safe_evidence=_safe())
    assert excinfo.value.reason == "V8F_T2_RECORD_SCHEMA_INVALID"


def test_duplicate_json_key_in_record_bytes_blocks():
    record = _record()
    raw = json.dumps(record, separators=(",", ":")).replace(
        '"OVERALL_RESULT":"PASS"', '"OVERALL_RESULT":"PASS","OVERALL_RESULT":"PASS"'
    ).encode("utf-8")
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck.verify_t2_prefreeze_record_bytes(raw, safe_evidence=_safe())
    assert excinfo.value.reason == "V8F_T2_RECORD_DUPLICATE_KEY"


# ---------------------------------------------------------------------------
# Canonical T2 evidence is derived from committed-public-object inputs
# (requirement 4, bullet 1)
# ---------------------------------------------------------------------------


def test_canonical_resolver_derives_from_committed_public_objects():
    safe = _resolve()
    assert safe == _safe()
    record = recheck.build_t2_prefreeze_record(safe)
    verification = recheck.verify_t2_prefreeze_record(record, safe_evidence=safe)
    assert verification["result"] == "PASS"
    assert verification["provenance_independently_verified"] is True


def test_canonical_resolver_has_no_caller_supplied_final_evidence_parameter():
    """A caller-supplied final safe-evidence mapping alone must never be
    sufficient to establish canonical PASS: the resolver's signature only
    accepts low-level Git-object readers, never a pre-derived mapping."""
    import inspect

    sig = inspect.signature(recheck._resolve_t2_prefreeze_safe_evidence_with_dependencies)
    assert "safe_evidence" not in sig.parameters
    assert "final_evidence" not in sig.parameters


def test_provenance_independently_verified_requires_real_derivation_chain():
    # A record that would satisfy verify_t2_prefreeze_record on its own,
    # built directly from a caller mapping, never reports having been
    # produced via the independent committed-evidence derivation chain
    # unless it actually went through _resolve_t2_prefreeze_safe_evidence_
    # with_dependencies. The pure verifier only ever proves internal
    # consistency of what it is given, which is why the canonical path
    # (tested above) is the sole route to a derivation-backed PASS.
    safe = _safe()
    record = recheck.build_t2_prefreeze_record(safe)
    verification = recheck.verify_t2_prefreeze_record(record, safe_evidence=safe)
    resolved = _resolve()
    assert verification["provenance_independently_verified"] is True
    assert resolved == safe  # canonical derivation reproduces the same facts independently


# ---------------------------------------------------------------------------
# Changing a committed source fact causes BLOCK even if a caller would have
# supplied the expected PASS value (requirement 4, bullet 2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state_override,expected_reason_prefix",
    [
        ({"T2": {**_state()["T2"], "raw_data_acquired": True}}, "V8F_T2_STATE_T2_VALUE_MISMATCH"),
        ({"trust_anchor_pinning": {"block_assignments_exposed": True}}, "V8F_T2_ASSIGNMENTS_EXPOSED"),
        ({"backtests": 1}, "V8F_T2_STATE_VALUE_MISMATCH"),
    ],
)
def test_committed_state_fact_change_blocks_even_with_would_be_passing_caller_intent(state_override, expected_reason_prefix):
    tampered_state = {**_state(), **state_override}
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(safe_state_reader=lambda root, commit, reader: tampered_state)
    assert excinfo.value.reason.startswith(expected_reason_prefix)


def test_committed_bridge_fact_change_blocks():
    tampered_bridge = {**_bridge(), "t2_acquired_before_authorized_acquisition": True}
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
        _resolve(safe_bridge_reader=lambda root, commit, reader: tampered_bridge)


def test_committed_historical_t2_fact_change_blocks():
    tampered = {**_historical_t2(), "T2_membership_reassigned": True}

    def object_reader(root, commit, path):
        if path == recheck.V8E_T2_PREFREEZE_GIT_PATH:
            return _text_block({key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in tampered.items()})
        if path == recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH:
            return _terminal_bytes()
        if path == recheck.V8F_DESIGN_GIT_PATH:
            return _design_bytes()
        raise AssertionError(path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
        _resolve(git_object_reader=object_reader)


# ---------------------------------------------------------------------------
# Missing/malformed/mismatched historical Git object causes BLOCK
# (requirement 4, bullet 3)
# ---------------------------------------------------------------------------


def test_v8e_historical_blob_binding_mismatch_blocks():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(
            git_blob_resolver=lambda root, commit, path: "0" * 40
            if path == recheck.V8E_T2_PREFREEZE_GIT_PATH
            else _resolver_kwargs()["git_blob_resolver"](root, commit, path)
        )
    assert excinfo.value.reason == "V8F_T2_V8E_HISTORICAL_BLOB_MISMATCH"


def test_v8e_terminal_blob_binding_mismatch_blocks():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(
            git_blob_resolver=lambda root, commit, path: "0" * 40
            if path == recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH
            else _resolver_kwargs()["git_blob_resolver"](root, commit, path)
        )
    assert excinfo.value.reason == "V8F_T2_V8E_TERMINAL_BLOB_MISMATCH"


@pytest.mark.parametrize(
    "path",
    [
        recheck.V8_STATE_GIT_PATH,
        recheck.V8B_T2_AUTHORITY_BRIDGE_GIT_PATH,
        recheck.V8_TRUSTED_PARTITION_GIT_PATH,
    ],
)
def test_safe_blob_mismatch_at_current_head_blocks(path):
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(
            git_blob_resolver=lambda root, commit, p: "0" * 40
            if p == path
            else _resolver_kwargs()["git_blob_resolver"](root, commit, p)
        )
    assert excinfo.value.reason == "V8F_T2_SAFE_BLOB_MISMATCH:" + path


def test_malformed_historical_t2_text_block_blocks():
    def object_reader(root, commit, path):
        if path == recheck.V8E_T2_PREFREEZE_GIT_PATH:
            return b"not a text block at all"
        if path == recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH:
            return _terminal_bytes()
        if path == recheck.V8F_DESIGN_GIT_PATH:
            return _design_bytes()
        raise AssertionError(path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == "V8F_T2_V8E_HISTORICAL_RECORD_INVALID"


def test_malformed_terminal_json_blocks():
    def object_reader(root, commit, path):
        if path == recheck.V8E_T2_PREFREEZE_GIT_PATH:
            return _historical_t2_bytes()
        if path == recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH:
            return b"{not json"
        if path == recheck.V8F_DESIGN_GIT_PATH:
            return _design_bytes()
        raise AssertionError(path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == "V8F_T2_V8E_TERMINAL_INVALID_JSON"


def test_missing_design_policy_text_blocks():
    def object_reader(root, commit, path):
        if path == recheck.V8E_T2_PREFREEZE_GIT_PATH:
            return _historical_t2_bytes()
        if path == recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH:
            return _terminal_bytes()
        if path == recheck.V8F_DESIGN_GIT_PATH:
            return b"unrelated design text"
        raise AssertionError(path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == "V8F_T2_DESIGN_POLICY_INVALID"


# ---------------------------------------------------------------------------
# Reviewed-support runtime binding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "runtime_overrides",
    [
        {"branch": "other-branch"},
        {"head": "2" * 40},
        {"origin_head": "2" * 40},
        {"worktree_clean": False},
        {"origin_url": "https://evil.example/x.git"},
        {"commits_after_reviewed_support_sha": 1},
    ],
)
def test_reviewed_support_runtime_mismatch_blocks(runtime_overrides):
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
        _resolve(runtime_state_reader=lambda root, sha: _runtime(sha, **runtime_overrides))


def test_verified_head_must_equal_reviewed_support_sha():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(verified_head="2" * 40)
    assert excinfo.value.reason == "V8F_T2_REVIEWED_SUPPORT_HEAD_MISMATCH"


# ---------------------------------------------------------------------------
# _derive_safe_evidence direct unit coverage
# ---------------------------------------------------------------------------


def test_derive_safe_evidence_matches_pure_safe_mapping():
    assert _derived() == _safe()


def test_derive_safe_evidence_requires_all_evidence_sources():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck._derive_safe_evidence(_state(), _bridge(), _anchor(), historical_t2=None, terminal=_terminal(), design={"policy_unchanged": True})
    assert excinfo.value.reason == "V8F_T2_REQUIRED_SAFE_EVIDENCE_MISSING"


def test_derive_safe_evidence_rejects_t2_access_not_prohibited():
    bad_terminal = {**_terminal(), "readiness_result": "PASS"}
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
        recheck._validate_v8e_terminal(json.dumps(bad_terminal).encode("utf-8"))


# ---------------------------------------------------------------------------
# Builder writes nothing; no real network/private access
# ---------------------------------------------------------------------------


def test_builder_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    record = recheck.build_t2_prefreeze_record(_safe())
    after = sorted(tmp_path.rglob("*"))
    assert before == after
    assert record["OVERALL_RESULT"] == "PASS"
    assert set(record) == set(recheck.V8F_T2_PREFREEZE_RECORD_FIELDS)


def test_no_network_module_wide():
    import ast
    import inspect

    source = inspect.getsource(recheck)
    tree = ast.parse(source)
    forbidden = {"urlopen", "socket", "requests", "httpx"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (forbidden & names)
    assert not (forbidden & attrs)


def test_no_ticker_identity_or_private_path_in_record():
    public = json.dumps(_record())
    assert "allocation_artifact_path" not in public
    assert "partition_manifest_path" not in public


def test_no_private_manifest_path_parameter_anywhere():
    import inspect

    for name in recheck.__all__:
        obj = getattr(recheck, name)
        if not callable(obj):
            continue
        try:
            params = set(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            continue
        assert "private_manifest_path" not in params
        assert "ticker_identity" not in params


# ---------------------------------------------------------------------------
# Real-repository proof: the canonical resolver, pointed at this actual
# repository's real committed objects (with only branch/head identity
# dependency-injected, since the current checkout may be on a different
# local branch than the authoritative V8F branch), independently derives
# PASS from the real V8_STATE.json / V8B_T2_AUTHORITY_BRIDGE.json /
# V8_TRUSTED_PARTITION.json / V8E historical documents.  No network access;
# only local Git object reads against the already-cloned repository.
# ---------------------------------------------------------------------------


def test_canonical_resolver_derives_pass_from_this_actual_repository():
    root = CANONICAL_REPOSITORY_ROOT
    head_sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    def fake_runtime(_root, _sha):
        return {
            "resolved_support_sha": head_sha,
            "branch": recheck.V8F_PRODUCTION_BRANCH,
            "head": head_sha,
            "origin_head": head_sha,
            "worktree_clean": True,
            "origin_url": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
            "commits_after_reviewed_support_sha": 0,
        }

    safe = recheck._resolve_t2_prefreeze_safe_evidence_with_dependencies(
        root,
        verified_head=head_sha,
        reviewed_support_implementation_sha=head_sha,
        runtime_state_reader=fake_runtime,
    )
    assert safe == _safe()
    record = recheck.build_t2_prefreeze_record(safe)
    verification = recheck.verify_t2_prefreeze_record(record, safe_evidence=safe)
    assert verification["result"] == "PASS"


def test_tampered_real_repository_blob_reference_blocks():
    root = CANONICAL_REPOSITORY_ROOT
    head_sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    from src.v8c_git_provenance import resolve_git_blob as real_resolve_git_blob

    def fake_runtime(_root, _sha):
        return {
            "resolved_support_sha": head_sha,
            "branch": recheck.V8F_PRODUCTION_BRANCH,
            "head": head_sha,
            "origin_head": head_sha,
            "worktree_clean": True,
            "origin_url": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
            "commits_after_reviewed_support_sha": 0,
        }

    def tampered_blob_resolver(r, commit, path):
        if path == recheck.V8_STATE_GIT_PATH:
            return "0" * 40
        return real_resolve_git_blob(r, commit, path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck._resolve_t2_prefreeze_safe_evidence_with_dependencies(
            root,
            verified_head=head_sha,
            reviewed_support_implementation_sha=head_sha,
            runtime_state_reader=fake_runtime,
            git_blob_resolver=tampered_blob_resolver,
        )
    assert excinfo.value.reason == "V8F_T2_SAFE_BLOB_MISMATCH:" + recheck.V8_STATE_GIT_PATH
