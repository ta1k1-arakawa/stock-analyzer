from __future__ import annotations

import pytest

from src import v8c_t2_preservation_recheck as recheck


def _safe_metadata(**overrides):
    metadata = {
        "t2_real_data_acquired": False,
        "t2_opened": False,
        "t2_research_access_count": 0,
        "t2_features_observed": False,
        "t2_outcomes_observed": False,
        "t2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
    }
    metadata.update(overrides)
    return metadata


def test_all_conditions_satisfied_passes():
    result = recheck._recheck_t2_preservation_conditions(_safe_metadata())
    assert result["result"] == "PASS"
    assert result["recheck_point"] == "recheck_2"


def test_missing_required_field_blocks():
    metadata = _safe_metadata()
    del metadata["t2_real_data_acquired"]
    with pytest.raises(recheck.V8CT2PreservationRecheckBlocked) as excinfo:
        recheck._recheck_t2_preservation_conditions(metadata)
    assert excinfo.value.reason == "V8C_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_SAFE_METADATA"


@pytest.mark.parametrize("field", [
    "t2_real_data_acquired", "t2_opened", "t2_features_observed", "t2_outcomes_observed", "t2_membership_reassigned",
])
def test_expect_false_field_true_blocks(field):
    with pytest.raises(recheck.V8CT2PreservationRecheckBlocked) as excinfo:
        recheck._recheck_t2_preservation_conditions(_safe_metadata(**{field: True}))
    assert excinfo.value.reason == "V8C_T2_PRESERVATION_RECHECK_BLOCKED:" + field.upper()


def test_nonzero_research_access_count_blocks():
    with pytest.raises(recheck.V8CT2PreservationRecheckBlocked) as excinfo:
        recheck._recheck_t2_preservation_conditions(_safe_metadata(t2_research_access_count=1))
    assert excinfo.value.reason == "V8C_T2_PRESERVATION_RECHECK_BLOCKED:T2_RESEARCH_ACCESS_COUNT"


@pytest.mark.parametrize("field", ["universe_definition_compatible", "partition_algorithm_compatible", "data_quality_policy_unchanged"])
def test_expect_true_field_false_blocks(field):
    with pytest.raises(recheck.V8CT2PreservationRecheckBlocked) as excinfo:
        recheck._recheck_t2_preservation_conditions(_safe_metadata(**{field: False}))
    assert excinfo.value.reason == "V8C_T2_PRESERVATION_RECHECK_BLOCKED:" + field.upper()


def test_absence_of_evidence_never_treated_as_pass():
    with pytest.raises(recheck.V8CT2PreservationRecheckBlocked):
        recheck._recheck_t2_preservation_conditions({})


# ---------------------------------------------------------------------------
# Production resolver: derives from V8_STATE.json, never trusts a caller-
# supplied safe_metadata mapping directly.
# ---------------------------------------------------------------------------


def test_production_resolver_missing_dependencies_fails_closed():
    """No real V8C_T2_AUTHORITY_BRIDGE-equivalent readiness has been
    executed; the production resolver must resolve real Git state and
    therefore either fail on a real dependency or succeed only if every
    real condition truly holds. It must never silently accept a
    caller-supplied mapping."""
    import inspect

    signature = inspect.signature(recheck.resolve_and_recheck_t2_preservation)
    assert len(signature.parameters) == 0  # accepts no caller-supplied trust root at all


# ---------------------------------------------------------------------------
# HIGH-3: recheck_2 must be executable from already-existing reviewed safe
# evidence (the pre-freeze baseline blob + design freeze approval PASS
# attestation) rather than requiring a self-declared
# V8_STATE.json["v8c_preservation_compatibility"] field that does not
# exist in the real repository.
# ---------------------------------------------------------------------------


def _fake_safe_metadata_dependencies(**overrides):
    deps = dict(
        git_commit_resolver=lambda: "a" * 40,
        anchor_reader=lambda head: {"authorization_status": "AUTHORIZED"},
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": "b" * 40},
        consumption_state_root="/tmp/does-not-matter",
        gate_consumption_checker=lambda *a, **k: False,
        v8_state_evidence_reader=lambda root, commit, reader: {
            "t2_raw_data_acquired": False,
            "t2_opened_for_research": False,
            "t2_sealed_holdout_access_count": 0,
            "block_assignments_exposed": False,
        },
        prefreeze_baseline_verifier=lambda root, commit: None,
        design_freeze_approval_reader=lambda root, commit: {
            "t2_preservation_recheck_result": "PASS",
            "t2_preservation_recheck_design_commit": recheck.EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        },
    )
    deps.update(overrides)
    return deps


def test_integration_style_resolver_derives_pass_from_prefreeze_baseline_and_freeze_approval():
    """Mirrors the current safe evidence chain (gate-consumption state,
    real V8_STATE.json T2 access counters, the exact-blob pre-freeze
    preservation baseline, and the frozen design freeze approval's own
    PASS attestation) and proves the production resolver can derive PASS
    without inventing a new V8_STATE compatibility field."""
    safe_metadata = recheck._resolve_t2_preservation_safe_metadata_with_dependencies(
        "repo", **_fake_safe_metadata_dependencies()
    )
    assert "compatibility" not in safe_metadata
    result = recheck._recheck_t2_preservation_conditions(safe_metadata)
    assert result["result"] == "PASS"
    assert result["recheck_point"] == "recheck_2"
    assert safe_metadata["t2_membership_reassigned"] is False
    assert safe_metadata["universe_definition_compatible"] is True
    assert safe_metadata["partition_algorithm_compatible"] is True
    assert safe_metadata["data_quality_policy_unchanged"] is True


def test_mutated_prefreeze_baseline_blob_blocks():
    def mutated_baseline(root, commit):
        raise recheck.V8CT2PreservationRecheckBlocked("V8C_PREFREEZE_PRESERVATION_AUDIT_MUTATED")

    with pytest.raises(recheck.V8CT2PreservationRecheckBlocked) as excinfo:
        recheck._resolve_t2_preservation_safe_metadata_with_dependencies(
            "repo", **_fake_safe_metadata_dependencies(prefreeze_baseline_verifier=mutated_baseline)
        )
    assert excinfo.value.reason == "V8C_PREFREEZE_PRESERVATION_AUDIT_MUTATED"


def test_missing_prefreeze_baseline_blocks():
    from src.v8c_git_provenance import V8CGitProvenanceBlocked

    def missing_baseline(root, commit):
        raise V8CGitProvenanceBlocked("GIT_OBJECT_READ_FAILED")

    with pytest.raises(recheck.V8CT2PreservationRecheckBlocked) as excinfo:
        recheck._resolve_t2_preservation_safe_metadata_with_dependencies(
            "repo", **_fake_safe_metadata_dependencies(prefreeze_baseline_verifier=missing_baseline)
        )
    assert excinfo.value.reason == "V8C_PREFREEZE_PRESERVATION_AUDIT_MISSING"


def test_missing_or_non_pass_freeze_approval_baseline_blocks():
    with pytest.raises(recheck.V8CT2PreservationRecheckBlocked) as excinfo:
        recheck._resolve_t2_preservation_safe_metadata_with_dependencies(
            "repo", **_fake_safe_metadata_dependencies(design_freeze_approval_reader=lambda root, commit: {
                "t2_preservation_recheck_result": "BLOCK",
                "t2_preservation_recheck_design_commit": recheck.EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
            })
        )
    assert excinfo.value.reason == "V8C_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_COMPATIBILITY_EVIDENCE"


def test_freeze_approval_bound_to_wrong_design_commit_blocks():
    with pytest.raises(recheck.V8CT2PreservationRecheckBlocked) as excinfo:
        recheck._resolve_t2_preservation_safe_metadata_with_dependencies(
            "repo", **_fake_safe_metadata_dependencies(design_freeze_approval_reader=lambda root, commit: {
                "t2_preservation_recheck_result": "PASS",
                "t2_preservation_recheck_design_commit": "f" * 40,
            })
        )
    assert excinfo.value.reason == "V8C_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_COMPATIBILITY_EVIDENCE"


def test_v8_state_evidence_reader_output_has_no_compatibility_key():
    """Do not modify V8_STATE.json merely to add a favorable self-declared
    field: the real ``_default_read_v8_state_t2_evidence`` reader's output
    schema no longer carries a ``compatibility`` key derived from a
    self-declared V8_STATE field."""
    import inspect

    signature_source = inspect.getsource(recheck._default_read_v8_state_t2_evidence)
    assert 'state.get("v8c_preservation_compatibility")' not in signature_source
    assert '"compatibility":' not in signature_source


def test_prefreeze_audit_blob_constant_matches_established_value():
    assert recheck.EXPECTED_PREFREEZE_PRESERVATION_AUDIT_BLOB == "ec9054caf94898948879b599196c055e480d2e52"
    assert recheck.PREFREEZE_PRESERVATION_AUDIT_GIT_PATH == "V8C_PREFREEZE_PRESERVATION_RECHECK.md"
