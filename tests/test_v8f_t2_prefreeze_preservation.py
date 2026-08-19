from __future__ import annotations

import json
import subprocess

import pytest

from src import v8f_t2_prefreeze_preservation as recheck
from src import v8f_t1c_preservation as t1c_preservation
from src.v8c_git_provenance import CANONICAL_REPOSITORY_ROOT


SUPPORT_SHA = "1" * 40
T1C_COMMIT = "2" * 40
HEAD = T1C_COMMIT
T1C_ARTIFACT_BLOB = "3" * 40


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


def _t1c_artifact(**overrides):
    """Synthetic V8F T1C preservation recheck artifact matching the exact
    contract already defined by src/v8f_t1c_preservation.py.  This is a
    synthetic fixture used only to prove future stage-order enforcement; it
    does NOT constitute or claim a real V8F T1C preservation PASS."""
    value = {
        "schema_version": "V8F_T1C_PRESERVATION_RECHECK_V1",
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "study": t1c_preservation.V8F_STUDY_NAME,
        "reviewed_v8f_design_candidate_commit": t1c_preservation.V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "source_v8e_terminal_commit": t1c_preservation.V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
        "allocation_artifact_self_hash": t1c_preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": t1c_preservation.EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": t1c_preservation.EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
        "parent_t_spare_ticker_list_sha256": t1c_preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "remaining_t_spare_ticker_list_sha256": t1c_preservation.EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "t1c_raw_acquisition_performed": False,
        "t1c_research_opened": False,
        "t1c_ohlcv_research_access": False,
        "t1c_feature_access": False,
        "t1c_outcome_access": False,
        "t1c_identities_publicly_exposed": False,
        "t1c_membership_reassigned": False,
        "allocation_self_hash_unchanged": True,
        "parent_v8_provenance_unchanged": True,
        "v8e_terminal_adjudication_authoritative": True,
        "preservation_recheck_result": "PASS",
    }
    value.update(overrides)
    return value


def _t1c_artifact_bytes(**overrides):
    return json.dumps(_t1c_artifact(**overrides), sort_keys=True, separators=(",", ":")).encode("utf-8")


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


def _runtime(support_sha=SUPPORT_SHA, t1c_commit=T1C_COMMIT, head=HEAD, **overrides):
    value = {
        "branch": recheck.V8F_PRODUCTION_BRANCH,
        "head": head,
        "origin_head": head,
        "worktree_clean": True,
        "origin_url": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "resolved_support_sha": support_sha,
        "resolved_t1c_commit": t1c_commit,
        "support_sha_ancestor_of_head": True,
        "t1c_commit_ancestor_of_head": True,
        "support_sha_ancestor_of_t1c_commit": True,
        "t2_source_blob_at_head": "same-blob",
        "t2_source_blob_at_reviewed_support_sha": "same-blob",
    }
    value.update(overrides)
    return value


def _default_blobs():
    return {
        recheck.V8F_DESIGN_GIT_PATH: recheck.V8F_DESIGN_CANDIDATE_BLOB_SHA,
        recheck.V8_STATE_GIT_PATH: recheck.V8_STATE_BLOB_SHA,
        recheck.V8B_T2_AUTHORITY_BRIDGE_GIT_PATH: recheck.V8B_T2_AUTHORITY_BRIDGE_BLOB_SHA,
        recheck.V8_TRUSTED_PARTITION_GIT_PATH: recheck.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH: recheck.V8F_V8E_TERMINAL_RECORD_BLOB_SHA,
        recheck.V8E_T2_PREFREEZE_GIT_PATH: recheck.V8E_T2_PREFREEZE_BLOB_SHA,
        recheck.V8F_T1C_PRESERVATION_ARTIFACT_GIT_PATH: T1C_ARTIFACT_BLOB,
    }


def _default_object_reader(root, commit, path):
    if path == recheck.V8E_T2_PREFREEZE_GIT_PATH:
        return _historical_t2_bytes()
    if path == recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH:
        return _terminal_bytes()
    if path == recheck.V8F_DESIGN_GIT_PATH:
        return _design_bytes()
    if path == recheck.V8F_T1C_PRESERVATION_ARTIFACT_GIT_PATH:
        return _t1c_artifact_bytes()
    raise AssertionError(path)


def _resolver_kwargs(**overrides):
    blobs = _default_blobs()

    defaults = {
        "verified_head": HEAD,
        "reviewed_support_implementation_sha": SUPPORT_SHA,
        "reviewed_v8f_t1c_preservation_recheck_commit": T1C_COMMIT,
        "git_blob_resolver": lambda root, commit, path: blobs[path],
        "git_object_reader": _default_object_reader,
        "safe_state_reader": lambda root, commit, reader: _state(),
        "safe_bridge_reader": lambda root, commit, reader: _bridge(),
        "trusted_anchor_reader": lambda root, commit: _anchor(),
        "runtime_state_reader": lambda root, s_sha, t_commit: _runtime(s_sha, t_commit),
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
    blobs = _default_blobs()
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(
            git_blob_resolver=lambda root, commit, path: "0" * 40
            if path == recheck.V8F_DESIGN_GIT_PATH
            else blobs[path]
        )
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
# Canonical T2 evidence is derived from committed-public-object inputs,
# GIVEN a valid synthetic T1C preservation prerequisite chain
# ---------------------------------------------------------------------------


def test_correct_synthetic_chronology_and_t1c_artifact_permits_resolver_to_continue():
    """With a valid synthetic T1C preservation prerequisite chain, the
    resolver continues to the existing independently-derived T2 evidence.
    This is a synthetic proof of stage-order plumbing only -- it does NOT
    constitute or claim a real V8F T1C preservation PASS."""
    safe = _resolve()
    assert safe == _safe()
    record = recheck.build_t2_prefreeze_record(safe)
    verification = recheck.verify_t2_prefreeze_record(record, safe_evidence=safe)
    assert verification["result"] == "PASS"
    # The pure verifier call above only proves record-vs-supplied-evidence
    # consistency; it does not itself read a Git object, so it must not
    # claim independent derivation even though `safe` happened to originate
    # from the resolver in this test.
    assert verification["provenance_values_verified_against_supplied_safe_evidence"] is True
    assert verification["provenance_independently_derived"] is False


def test_canonical_resolver_has_no_caller_supplied_final_evidence_parameter():
    """A caller-supplied final safe-evidence mapping alone must never be
    sufficient to establish canonical PASS: the resolver's signature only
    accepts low-level Git-object readers and ancestor-derived runtime facts,
    never a pre-derived mapping."""
    import inspect

    sig = inspect.signature(recheck._resolve_t2_prefreeze_safe_evidence_with_dependencies)
    assert "safe_evidence" not in sig.parameters
    assert "final_evidence" not in sig.parameters
    sig2 = inspect.signature(recheck.resolve_and_verify_t2_prefreeze_preservation)
    assert "safe_evidence" not in sig2.parameters
    assert "final_evidence" not in sig2.parameters
    assert "reviewed_v8f_t1c_preservation_recheck_commit" in sig2.parameters


# ---------------------------------------------------------------------------
# V8F-PREFREEZE-MEDIUM-001: pure verifier must not claim independent
# provenance derivation; only the canonical wrapper may, and only after a
# real derivation succeeded.
# ---------------------------------------------------------------------------


def test_pure_verify_record_does_not_claim_independent_provenance_derivation():
    safe = _safe()
    record = recheck.build_t2_prefreeze_record(safe)
    verification = recheck.verify_t2_prefreeze_record(record, safe_evidence=safe)
    assert set(verification) == {
        "result",
        "checkpoint",
        "reviewed_v8f_design_candidate_commit",
        "nine_conditions_independently_verified",
        "provenance_values_verified_against_supplied_safe_evidence",
        "provenance_independently_derived",
    }
    assert verification["provenance_independently_derived"] is False
    assert verification["provenance_values_verified_against_supplied_safe_evidence"] is True
    assert "provenance_independently_verified" not in verification


def test_pure_bytes_verifier_also_does_not_claim_independent_provenance_derivation():
    safe = _safe()
    record = recheck.build_t2_prefreeze_record(safe)
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    verification = recheck.verify_t2_prefreeze_record_bytes(raw, safe_evidence=safe)
    assert verification["provenance_independently_derived"] is False
    assert verification["provenance_values_verified_against_supplied_safe_evidence"] is True


def test_caller_supplied_safe_mapping_cannot_produce_independently_derived_true():
    """No matter how a caller constructs a syntactically-valid safe_evidence
    mapping and record, the pure verifier path can never itself set
    provenance_independently_derived=True -- there is no parameter or code
    path in verify_t2_prefreeze_record/verify_t2_prefreeze_record_bytes that
    accepts or is influenced by such a claim."""
    import inspect

    sig = inspect.signature(recheck.verify_t2_prefreeze_record)
    assert "provenance_independently_derived" not in sig.parameters
    assert "independently_derived" not in sig.parameters
    sig_bytes = inspect.signature(recheck.verify_t2_prefreeze_record_bytes)
    assert "provenance_independently_derived" not in sig_bytes.parameters

    safe = _safe()
    record = recheck.build_t2_prefreeze_record(safe)
    # Even an attempted extra field on the caller's record is rejected by
    # exact schema validation before any evidence claim could be forged.
    forged = dict(record)
    forged["provenance_independently_derived"] = True
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
        recheck.verify_t2_prefreeze_record(forged, safe_evidence=safe)
    # The legitimate call is unaffected and still reports False.
    verification = recheck.verify_t2_prefreeze_record(record, safe_evidence=safe)
    assert verification["provenance_independently_derived"] is False


def test_canonical_wrapper_adds_independently_derived_true_only_after_real_derivation(monkeypatch):
    """Staged synthetic fixture proving the wrapper's own composition logic:
    monkeypatch _resolve_t2_prefreeze_safe_evidence_with_dependencies itself
    (that function's own correctness -- real Git-object derivation, HIGH-002
    stage-order enforcement -- is already fully covered elsewhere via
    _resolve()/the HIGH-002 test suite) together with the module's Git
    branch/head lookups, so resolve_and_verify_t2_prefreeze_preservation
    runs its real internal control flow end-to-end and demonstrably adds
    provenance_independently_derived=True only after that (mocked-successful)
    derivation call returns."""
    calls = []

    def fake_resolver(root, *, verified_head, reviewed_support_implementation_sha, reviewed_v8f_t1c_preservation_recheck_commit):
        calls.append(
            (verified_head, reviewed_support_implementation_sha, reviewed_v8f_t1c_preservation_recheck_commit)
        )
        return _safe()

    monkeypatch.setattr(recheck, "_resolve_t2_prefreeze_safe_evidence_with_dependencies", fake_resolver)
    monkeypatch.setattr(
        recheck,
        "_git_text",
        lambda root, args, reason: {
            ("branch", "--show-current"): recheck.V8F_PRODUCTION_BRANCH,
            ("rev-parse", "HEAD"): HEAD,
        }[tuple(args)],
    )

    result = recheck.resolve_and_verify_t2_prefreeze_preservation(
        reviewed_support_implementation_sha=SUPPORT_SHA,
        reviewed_v8f_t1c_preservation_recheck_commit=T1C_COMMIT,
    )
    # Confirms the derivation call actually happened (and with the exact
    # caller-supplied identifiers) before the claim was added.
    assert calls == [(HEAD, SUPPORT_SHA, T1C_COMMIT)]
    assert result["provenance_independently_derived"] is True
    assert result["nine_conditions_independently_verified"] is True

    # The exact same record/evidence, verified through the pure path alone,
    # still reports False -- the wrapper's claim is not smuggled back into
    # the pure verifier's own semantics.
    pure = recheck.verify_t2_prefreeze_record(result["record"], safe_evidence=result["safe_evidence"])
    assert pure["provenance_independently_derived"] is False


def test_canonical_wrapper_never_reaches_claim_when_derivation_raises(monkeypatch):
    """If the real derivation call fails, provenance_independently_derived
    must never appear as True anywhere -- the wrapper raises before it can
    be added."""

    def failing_resolver(*args, **kwargs):
        raise recheck.V8FT2PrefreezePreservationBlocked("V8F_T2_SAFE_GIT_EVIDENCE_UNAVAILABLE")

    monkeypatch.setattr(recheck, "_resolve_t2_prefreeze_safe_evidence_with_dependencies", failing_resolver)
    monkeypatch.setattr(
        recheck,
        "_git_text",
        lambda root, args, reason: {
            ("branch", "--show-current"): recheck.V8F_PRODUCTION_BRANCH,
            ("rev-parse", "HEAD"): HEAD,
        }[tuple(args)],
    )

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
        recheck.resolve_and_verify_t2_prefreeze_preservation(
            reviewed_support_implementation_sha=SUPPORT_SHA,
            reviewed_v8f_t1c_preservation_recheck_commit=T1C_COMMIT,
        )


# ---------------------------------------------------------------------------
# V8F-PREFREEZE-HIGH-002: current V8F T1C preservation is a hard prerequisite
# ---------------------------------------------------------------------------


def test_current_real_repository_cannot_reach_t2_prefreeze_pass_before_t1c_preservation():
    """At the current repository stage, the canonical resolver MUST BLOCK,
    because no committed current-study V8F_T1C_PRESERVATION_RECHECK.json
    exists yet -- the human-gated V8F T1C preservation stage has not
    happened.  This uses the REAL repository's Git objects (no network, no
    private read), with only the runtime ancestor facts dependency-injected
    since this checkout's local branch differs from the authoritative V8F
    branch name."""
    root = CANONICAL_REPOSITORY_ROOT
    head_sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    def fake_runtime(_root, support_sha, t1c_commit):
        return {
            "branch": recheck.V8F_PRODUCTION_BRANCH,
            "head": head_sha,
            "origin_head": head_sha,
            "worktree_clean": True,
            "origin_url": "https://github.com/ta1k1-arakawa/stock-analyzer.git",
            "resolved_support_sha": support_sha,
            "resolved_t1c_commit": t1c_commit,
            "support_sha_ancestor_of_head": True,
            "t1c_commit_ancestor_of_head": True,
            "support_sha_ancestor_of_t1c_commit": True,
            "t2_source_blob_at_head": "same-blob",
            "t2_source_blob_at_reviewed_support_sha": "same-blob",
        }

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck._resolve_t2_prefreeze_safe_evidence_with_dependencies(
            root,
            verified_head=head_sha,
            reviewed_support_implementation_sha=head_sha,
            reviewed_v8f_t1c_preservation_recheck_commit=head_sha,
            runtime_state_reader=fake_runtime,
        )
    # The real V8F_T1C_PRESERVATION_RECHECK.json does not exist anywhere in
    # this repository yet, so the real Git blob resolver cannot find it.
    assert excinfo.value.reason == "V8F_T2_SAFE_GIT_EVIDENCE_UNAVAILABLE"


def test_missing_t1c_artifact_blocks():
    def missing_blob_resolver(root, commit, path):
        if path == recheck.V8F_T1C_PRESERVATION_ARTIFACT_GIT_PATH:
            from src.v8c_git_provenance import V8CGitProvenanceBlocked
            raise V8CGitProvenanceBlocked("NOT_FOUND")
        return _default_blobs()[path]

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_blob_resolver=missing_blob_resolver)
    assert excinfo.value.reason == "V8F_T2_SAFE_GIT_EVIDENCE_UNAVAILABLE"


@pytest.mark.parametrize(
    "mutation,expected_reason",
    [
        ("schema_version", "V8F_T2_T1C_PRESERVATION_ARTIFACT_INVALID:V8F_PRESERVATION_ARTIFACT_VALUE_MISMATCH:schema_version"),
        ("study", "V8F_T2_T1C_PRESERVATION_ARTIFACT_INVALID:V8F_PRESERVATION_ARTIFACT_VALUE_MISMATCH:study"),
        (
            "reviewed_v8f_design_candidate_commit",
            "V8F_T2_T1C_PRESERVATION_ARTIFACT_INVALID:V8F_PRESERVATION_ARTIFACT_VALUE_MISMATCH:reviewed_v8f_design_candidate_commit",
        ),
        (
            "preservation_recheck_result",
            "V8F_T2_T1C_PRESERVATION_ARTIFACT_INVALID:V8F_PRESERVATION_ARTIFACT_VALUE_MISMATCH:preservation_recheck_result",
        ),
    ],
)
def test_wrong_t1c_artifact_schema_or_binding_blocks(mutation, expected_reason):
    overrides = {
        "schema_version": "WRONG_SCHEMA_V1",
        "study": "V8E_HISTORICAL_RESEARCH",
        "reviewed_v8f_design_candidate_commit": "0" * 40,
        "preservation_recheck_result": "BLOCK",
    }
    tampered_bytes = _t1c_artifact_bytes(**{mutation: overrides[mutation]})

    def object_reader(root, commit, path):
        if path == recheck.V8F_T1C_PRESERVATION_ARTIFACT_GIT_PATH:
            return tampered_bytes
        return _default_object_reader(root, commit, path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == expected_reason


@pytest.mark.parametrize(
    "field",
    [
        "t1c_raw_acquisition_performed",
        "t1c_research_opened",
        "t1c_ohlcv_research_access",
        "t1c_feature_access",
        "t1c_outcome_access",
        "t1c_identities_publicly_exposed",
        "t1c_membership_reassigned",
    ],
)
def test_tampered_t1c_no_access_flag_blocks(field):
    tampered_bytes = _t1c_artifact_bytes(**{field: True})

    def object_reader(root, commit, path):
        if path == recheck.V8F_T1C_PRESERVATION_ARTIFACT_GIT_PATH:
            return tampered_bytes
        return _default_object_reader(root, commit, path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == "V8F_T2_T1C_PRESERVATION_ARTIFACT_INVALID:V8F_PRESERVATION_ARTIFACT_VALUE_MISMATCH:" + field


@pytest.mark.parametrize(
    "field,value",
    [
        ("t1c_ticker_count", 299),
        ("t1c_ticker_list_sha256", "0" * 64),
        ("allocation_artifact_self_hash", "0" * 64),
        ("parent_t_spare_ticker_list_sha256", "0" * 64),
        ("remaining_t_spare_ticker_list_sha256", "0" * 64),
        ("allocation_self_hash_unchanged", False),
        ("parent_v8_provenance_unchanged", False),
    ],
)
def test_tampered_t1c_count_hash_or_provenance_blocks(field, value):
    tampered_bytes = _t1c_artifact_bytes(**{field: value})

    def object_reader(root, commit, path):
        if path == recheck.V8F_T1C_PRESERVATION_ARTIFACT_GIT_PATH:
            return tampered_bytes
        return _default_object_reader(root, commit, path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == "V8F_T2_T1C_PRESERVATION_ARTIFACT_INVALID:V8F_PRESERVATION_ARTIFACT_VALUE_MISMATCH:" + field


def test_t1c_artifact_blob_changed_after_reviewed_commit_blocks():
    # verified_head must differ from reviewed_t1c_commit for this mutation
    # to be observable at all -- otherwise "blob at commit" and "blob at
    # head" trivially read the identical commit.
    later_head = "4" * 40

    def mutated_blob_resolver(root, commit, path):
        blobs = _default_blobs()
        if path == recheck.V8F_T1C_PRESERVATION_ARTIFACT_GIT_PATH:
            return T1C_ARTIFACT_BLOB if commit == T1C_COMMIT else "9" * 40
        return blobs[path]

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(
            verified_head=later_head,
            git_blob_resolver=mutated_blob_resolver,
            runtime_state_reader=lambda root, s, t: _runtime(s, t, head=later_head, origin_head=later_head),
        )
    assert excinfo.value.reason == "V8F_T2_T1C_PRESERVATION_ARTIFACT_MUTATED_AFTER_REVIEW"


def test_t1c_artifact_duplicate_json_key_blocks():
    raw = b'{"schema_version":"V8F_T1C_PRESERVATION_RECHECK_V1","schema_version":"X"}'

    def object_reader(root, commit, path):
        if path == recheck.V8F_T1C_PRESERVATION_ARTIFACT_GIT_PATH:
            return raw
        return _default_object_reader(root, commit, path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == "V8F_T2_T1C_PRESERVATION_ARTIFACT_INVALID:V8F_PRESERVATION_ARTIFACT_DUPLICATE_KEY"


# ---------------------------------------------------------------------------
# Fix runtime binding for the frozen stage order
# ---------------------------------------------------------------------------


def test_support_sha_not_ancestor_of_t1c_commit_blocks():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(
            runtime_state_reader=lambda root, s, t: _runtime(s, t, support_sha_ancestor_of_t1c_commit=False)
        )
    assert excinfo.value.reason == "V8F_T2_SUPPORT_SHA_NOT_ANCESTOR_OF_T1C_COMMIT"


def test_t1c_commit_not_ancestor_of_head_blocks():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(runtime_state_reader=lambda root, s, t: _runtime(s, t, t1c_commit_ancestor_of_head=False))
    assert excinfo.value.reason == "V8F_T2_T1C_COMMIT_NOT_ANCESTOR_OF_HEAD"


def test_support_sha_not_ancestor_of_head_blocks():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(runtime_state_reader=lambda root, s, t: _runtime(s, t, support_sha_ancestor_of_head=False))
    assert excinfo.value.reason == "V8F_T2_SUPPORT_SHA_NOT_ANCESTOR_OF_HEAD"


def test_t2_support_source_blob_changed_since_review_blocks():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(
            runtime_state_reader=lambda root, s, t: _runtime(
                s, t, t2_source_blob_at_head="new-blob", t2_source_blob_at_reviewed_support_sha="old-blob"
            )
        )
    assert excinfo.value.reason == "V8F_T2_SUPPORT_SOURCE_BLOB_CHANGED_SINCE_REVIEW"


def test_resolved_support_sha_mismatch_blocks():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(runtime_state_reader=lambda root, s, t: _runtime(s, t, resolved_support_sha="9" * 40))
    assert excinfo.value.reason == "V8F_T2_REVIEWED_SUPPORT_SHA_UNRESOLVABLE"


def test_resolved_t1c_commit_mismatch_blocks():
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(runtime_state_reader=lambda root, s, t: _runtime(s, t, resolved_t1c_commit="9" * 40))
    assert excinfo.value.reason == "V8F_T2_REVIEWED_T1C_COMMIT_UNRESOLVABLE"


@pytest.mark.parametrize(
    "runtime_overrides,expected_reason",
    [
        ({"branch": "other-branch"}, "V8F_T2_BRANCH_MISMATCH"),
        ({"head": "9" * 40}, "V8F_T2_HEAD_NOT_ORIGIN"),
        ({"origin_head": "9" * 40}, "V8F_T2_HEAD_NOT_ORIGIN"),
        ({"worktree_clean": False}, "V8F_T2_WORKTREE_DIRTY"),
        ({"origin_url": "https://evil.example/x.git"}, "V8F_T2_ORIGIN_UNTRUSTED"),
    ],
)
def test_reviewed_support_runtime_mismatch_blocks(runtime_overrides, expected_reason):
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(runtime_state_reader=lambda root, s, t: _runtime(s, t, **runtime_overrides))
    assert excinfo.value.reason == expected_reason


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_runtime_schema_exactness(mutation):
    def bad_runtime(root, s, t):
        value = _runtime(s, t)
        if mutation == "missing":
            del value["worktree_clean"]
        else:
            value["unexpected"] = True
        return value

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(runtime_state_reader=bad_runtime)
    assert excinfo.value.reason == "V8F_T2_REVIEWED_SUPPORT_RUNTIME_SCHEMA_INVALID"


def test_later_stage_artifact_committed_on_top_of_support_sha_no_longer_blocks_by_itself():
    """The whole point of the fix: HEAD may legitimately be a strict
    descendant of reviewed_support_implementation_sha (because the T1C
    preservation artifact was committed afterward) without that alone
    causing BLOCK -- only an actual change to this T2 support module's own
    source blob does."""
    safe = _resolve(
        verified_head=T1C_COMMIT,
        runtime_state_reader=lambda root, s, t: _runtime(s, t, head=T1C_COMMIT, origin_head=T1C_COMMIT),
    )
    assert safe == _safe()


# ---------------------------------------------------------------------------
# Changing a committed source fact causes BLOCK even if a caller would have
# supplied the expected PASS value
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
        return _default_object_reader(root, commit, path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
        _resolve(git_object_reader=object_reader)


# ---------------------------------------------------------------------------
# Missing/malformed/mismatched historical Git object causes BLOCK
# ---------------------------------------------------------------------------


def test_v8e_historical_blob_binding_mismatch_blocks():
    blobs = _default_blobs()
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(
            git_blob_resolver=lambda root, commit, path: "0" * 40
            if path == recheck.V8E_T2_PREFREEZE_GIT_PATH
            else blobs[path]
        )
    assert excinfo.value.reason == "V8F_T2_V8E_HISTORICAL_BLOB_MISMATCH"


def test_v8e_terminal_blob_binding_mismatch_blocks():
    blobs = _default_blobs()
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(
            git_blob_resolver=lambda root, commit, path: "0" * 40
            if path == recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH
            else blobs[path]
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
    blobs = _default_blobs()
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_blob_resolver=lambda root, commit, p: "0" * 40 if p == path else blobs[p])
    assert excinfo.value.reason == "V8F_T2_SAFE_BLOB_MISMATCH:" + path


def test_malformed_historical_t2_text_block_blocks():
    def object_reader(root, commit, path):
        if path == recheck.V8E_T2_PREFREEZE_GIT_PATH:
            return b"not a text block at all"
        return _default_object_reader(root, commit, path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == "V8F_T2_V8E_HISTORICAL_RECORD_INVALID"


def test_malformed_terminal_json_blocks():
    def object_reader(root, commit, path):
        if path == recheck.V8F_V8E_TERMINAL_RECORD_GIT_PATH:
            return b"{not json"
        return _default_object_reader(root, commit, path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == "V8F_T2_V8E_TERMINAL_INVALID_JSON"


def test_missing_design_policy_text_blocks():
    def object_reader(root, commit, path):
        if path == recheck.V8F_DESIGN_GIT_PATH:
            return b"unrelated design text"
        return _default_object_reader(root, commit, path)

    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        _resolve(git_object_reader=object_reader)
    assert excinfo.value.reason == "V8F_T2_DESIGN_POLICY_INVALID"


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
# Builder writes nothing; no real network/private access/gate consumption
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


def test_no_gate_consumer_or_private_reader_anywhere():
    import inspect

    source = inspect.getsource(recheck)
    assert "gate_consumer" not in source
    assert "consume_gate" not in source
    assert "private_reader" not in source


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


def test_resolve_uses_real_default_readers_when_uninjected():
    """Confirms git_blob_resolver / git_object_reader / trusted_anchor_reader
    default to the real Git-object functions (proving the DI parameters are
    genuine overrides of a real path, not the only path that exists)."""
    import inspect

    sig = inspect.signature(recheck._resolve_t2_prefreeze_safe_evidence_with_dependencies)
    from src.v8c_git_provenance import read_git_object_bytes, resolve_git_blob
    from src.v8c_production_provenance import read_and_verify_v8_trusted_partition_anchor

    assert sig.parameters["git_blob_resolver"].default is resolve_git_blob
    assert sig.parameters["git_object_reader"].default is read_git_object_bytes
    assert sig.parameters["trusted_anchor_reader"].default is read_and_verify_v8_trusted_partition_anchor
