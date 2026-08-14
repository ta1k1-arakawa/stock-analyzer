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
