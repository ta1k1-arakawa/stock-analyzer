from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src import v8b_human_gate_consumption as gate_consumption
from src import v8b_t2_reuse_recheck as recheck

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_DESIGN_COMMIT = "eedf198b93185b963b825170ed0be97e93f923b7"


def _real_head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Private pure evaluator -- fake/synthetic tests
# ---------------------------------------------------------------------------


def _pass_metadata(**overrides) -> dict:
    metadata = {
        "t2_acquired": False,
        "t2_opened": False,
        "t2_ticker_identities_exposed_to_human_public_research_loop": False,
        "t2_market_data_raw_ohlcv_feature_outcome_research_exposure": False,
        "t2_universe_definition_unchanged": True,
        "t2_partition_algorithm_unchanged": True,
        "t2_v8b_f1_c1_policy_fixed": True,
    }
    metadata.update(overrides)
    return metadata


def test_pure_evaluator_pass():
    result = recheck._recheck_t2_reuse_conditions(_pass_metadata())
    assert result == {"result": "PASS", "block": "T2"}


def test_pure_evaluator_blocks_on_missing_field():
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck._recheck_t2_reuse_conditions({"t2_acquired": False})
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_SAFE_METADATA"


def test_pure_evaluator_blocks_on_already_acquired():
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck._recheck_t2_reuse_conditions(_pass_metadata(t2_acquired=True))
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_BLOCKED:T2_ACQUIRED"


def test_pure_evaluator_blocks_on_universe_definition_changed():
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck._recheck_t2_reuse_conditions(_pass_metadata(t2_universe_definition_unchanged=False))
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_BLOCKED:T2_UNIVERSE_DEFINITION_UNCHANGED"


def test_pure_evaluator_module_defines_no_fallback_substitution():
    assert not any("spare" in name.lower() or "t3" in name.lower() for name in recheck.__all__)


# ---------------------------------------------------------------------------
# FINAL_REPEAT finding HIGH-3: the public production resolver takes no
# caller-supplied ``verified_head`` (or ``repository_root``) -- it resolves
# the current verified production HEAD itself.
# ---------------------------------------------------------------------------


def test_public_resolver_accepts_no_arguments_at_all():
    import inspect

    assert dict(inspect.signature(recheck.resolve_and_recheck_t2_reuse_conditions).parameters) == {}


def test_public_resolver_blocks_on_real_repo_today():
    """The real V8B_T2_REUSE_CONDITIONS_RECHECK.json does not exist yet --
    the real post-Layer-B recheck has not been performed -- so production
    must fail closed today. The real repo's provenance chain may also
    legitimately block earlier (dirty worktree in this working session,
    missing freeze approval, etc.) -- any of those is an acceptable
    fail-closed outcome; what matters is that the call requires zero
    arguments and never trusts a caller-supplied head."""
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked):
        recheck.resolve_and_recheck_t2_reuse_conditions()


def test_module_no_longer_reads_the_pre_freeze_section_12_2_document():
    """Confirms the fix: this module's own field set does not name the old
    §12.2 markdown path, and no function accepts it."""
    assert "V8B_TSPARE_T2_T3_PRESERVATION_RECHECK.md" not in recheck.POST_FREEZE_RECHECK_GIT_PATH
    assert recheck.POST_FREEZE_RECHECK_GIT_PATH == "V8B_T2_REUSE_CONDITIONS_RECHECK.json"
    assert recheck.POST_FREEZE_RECHECK_STAGE == "POST_FREEZE"


# ---------------------------------------------------------------------------
# DI-testable private implementation: fake Git/provenance/consumption
# dependencies (no real git checkout needed).
# ---------------------------------------------------------------------------


def _post_freeze_artifact(**overrides) -> dict:
    artifact = {
        "schema_version": recheck.POST_FREEZE_RECHECK_SCHEMA_VERSION,
        "study": "V8B_HISTORICAL_RESEARCH",
        "gate": recheck.POST_FREEZE_RECHECK_GATE,
        "frozen_design_git_commit": SYNTHETIC_DESIGN_COMMIT,
        "stage": "POST_FREEZE",
        "result": "PASS",
        "layer_b_completed": True,
        "frozen_final_candidate_established": True,
        "t2_acquired": False,
        "t2_opened": False,
        "t2_ticker_identities_exposed_to_human_public_research_loop": False,
        "t2_market_data_raw_ohlcv_feature_outcome_research_exposure": False,
        "t2_universe_definition_unchanged": True,
        "t2_partition_algorithm_unchanged": True,
        "t2_v8b_f1_c1_policy_fixed": True,
    }
    artifact.update(overrides)
    return artifact


def _safe_v8_state_evidence(**overrides) -> dict:
    evidence = {
        "t2_raw_data_acquired": False,
        "t2_opened_for_research": False,
        "t2_sealed_holdout_access_count": None,
        "block_assignments_exposed": False,
    }
    evidence.update(overrides)
    return evidence


def _default_dependencies(**overrides) -> dict:
    deps = dict(
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        anchor_reader=lambda head: {"authorization_status": "AUTHORIZED"},
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": "b" * 40},
        consumption_state_root="/nonexistent/never-created",
        layer_b_completion_reader=lambda head: {"ok": True},
        frozen_final_candidate_reader=lambda head: {"ok": True},
        no_research_opening_api_exists=lambda: True,
        git_object_reader=lambda root, commit, path: json.dumps(_post_freeze_artifact()).encode("utf-8"),
        gate_consumption_checker=lambda state_root, gate, design_commit: False,
        v8_state_evidence_reader=lambda root, commit, git_object_reader: _safe_v8_state_evidence(),
    )
    deps.update(overrides)
    return deps


def run(**overrides):
    return recheck._resolve_t2_reuse_safe_metadata_with_dependencies(ROOT, **_default_dependencies(**overrides))


def run_full(**overrides):
    return recheck._resolve_and_recheck_t2_reuse_conditions_with_dependencies(ROOT, **_default_dependencies(**overrides))


def test_di_seam_passes_on_well_formed_synthetic_artifact_and_dependencies():
    result = run_full()
    assert result == {"result": "PASS", "block": "T2"}


def test_resolve_safe_metadata_matches_pure_evaluator_schema():
    safe_metadata = run()
    assert set(safe_metadata) == set(recheck.REQUIRED_SAFE_METADATA_FIELDS)
    for value in safe_metadata.values():
        assert isinstance(value, bool)


def test_anchor_reader_failure_blocks_before_reading_recheck_artifact():
    def unreachable_reader(root, commit, path):
        raise AssertionError("recheck artifact must not be read if the anchor check already failed")

    def failing_anchor(head):
        raise recheck.V8BProductionProvenanceBlocked("V8_TRUSTED_PARTITION_BLOB_MUTATED")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run(anchor_reader=failing_anchor, git_object_reader=unreachable_reader)
    assert excinfo.value.reason == "V8_TRUSTED_PARTITION_BLOB_MUTATED"


def test_reviewed_implementation_binder_failure_blocks():
    def failing_binder(head):
        raise recheck.V8BProductionProvenanceBlocked("V8B_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:src/v8_partition.py")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run(reviewed_implementation_binder=failing_binder)
    assert excinfo.value.reason == "V8B_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:src/v8_partition.py"


def test_missing_reviewed_implementation_review_maps_to_fixed_missing_reason():
    def missing_review(head):
        raise recheck.V8BGitProvenanceBlocked("GIT_OBJECT_READ_FAILED")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run(reviewed_implementation_binder=missing_review)
    assert excinfo.value.reason == "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING"


def test_production_resolver_missing_artifact_blocks():
    def missing_artifact(root, commit, path):
        raise recheck.V8BGitProvenanceBlocked("GIT_OBJECT_READ_FAILED")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(git_object_reader=missing_artifact)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_MISSING"


def test_duplicate_key_artifact_blocks():
    def dup_key_reader(root, commit, path):
        return b'{"schema_version": "a", "schema_version": "b"}'

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(git_object_reader=dup_key_reader)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_DUPLICATE_KEY"


@pytest.mark.parametrize(
    "field,value,expected_reason",
    [
        ("stage", "PRE_FREEZE", "V8B_T2_REUSE_CONDITIONS_RECHECK_NOT_POST_FREEZE"),
        ("result", "BLOCK", "V8B_T2_REUSE_CONDITIONS_RECHECK_NOT_PASS"),
        ("layer_b_completed", False, "V8B_T2_REUSE_CONDITIONS_RECHECK_LAYER_B_NOT_COMPLETE"),
        ("frozen_final_candidate_established", False, "V8B_T2_REUSE_CONDITIONS_RECHECK_NO_FROZEN_FINAL_CANDIDATE"),
        ("frozen_design_git_commit", "0" * 40, "V8B_T2_REUSE_CONDITIONS_RECHECK_DESIGN_COMMIT_MISMATCH"),
        ("gate", "SOME_OTHER_GATE", "V8B_T2_REUSE_CONDITIONS_RECHECK_GATE_MISMATCH"),
        ("study", "V8_HISTORICAL_RESEARCH", "V8B_T2_REUSE_CONDITIONS_RECHECK_STUDY_MISMATCH"),
    ],
)
def test_production_resolver_field_semantics_enforced(field, value, expected_reason):
    def reader(root, commit, path):
        return json.dumps(_post_freeze_artifact(**{field: value})).encode("utf-8")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(git_object_reader=reader)
    assert excinfo.value.reason == expected_reason


def test_stale_pre_freeze_evidence_cannot_satisfy_the_post_freeze_gate():
    """A forged artifact that claims the OLD §12.2 pre-freeze evidence is
    good enough (stage=PRE_FREEZE) must not satisfy the §12.4 gate."""
    def reader(root, commit, path):
        return json.dumps(_post_freeze_artifact(stage="PRE_FREEZE")).encode("utf-8")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(git_object_reader=reader)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_NOT_POST_FREEZE"


# ---------------------------------------------------------------------------
# FINAL_REPEAT finding HIGH-3 core behavior: facts are DERIVED from
# authoritative state, not merely trusted because the artifact claims them.
# ---------------------------------------------------------------------------


def test_artifact_falsely_claiming_universe_unchanged_is_irrelevant_if_anchor_check_itself_fails():
    """Even an artifact honestly claiming t2_universe_definition_unchanged
    cannot compensate for the anchor check itself failing -- the anchor
    check must actually pass for the recheck to proceed at all."""
    def failing_anchor(head):
        raise recheck.V8BProductionProvenanceBlocked("V8_TRUSTED_PARTITION_BLOB_MUTATED")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(anchor_reader=failing_anchor)
    assert excinfo.value.reason == "V8_TRUSTED_PARTITION_BLOB_MUTATED"


def test_artifact_falsely_claiming_universe_changed_is_rejected_as_self_declared_mismatch():
    """The inverse: the anchor/binder checks PASS (authoritative truth is
    "unchanged"), but the artifact dishonestly (or stale-ly) claims
    t2_universe_definition_unchanged=False -- this must BLOCK as a
    self-declared mismatch, never silently trust the artifact's claim over
    the derived truth."""
    def reader(root, commit, path):
        return json.dumps(_post_freeze_artifact(t2_universe_definition_unchanged=False)).encode("utf-8")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(git_object_reader=reader)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_SELF_DECLARED_MISMATCH:t2_universe_definition_unchanged"


def test_artifact_falsely_claiming_policy_fixed_is_rejected_as_self_declared_mismatch():
    def reader(root, commit, path):
        return json.dumps(_post_freeze_artifact(t2_v8b_f1_c1_policy_fixed=False)).encode("utf-8")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(git_object_reader=reader)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_SELF_DECLARED_MISMATCH:t2_v8b_f1_c1_policy_fixed"


def test_t2_acquired_is_derived_from_durable_gate_receipt_not_artifact_claim():
    """The artifact claims t2_acquired=False (as an honest artifact would
    before any real acquisition), but the durable T2_RAW_ACQUISITION_
    HUMAN_GATE receipt already exists -- proving a real acquisition attempt
    already happened. The recheck must derive t2_acquired=True from that
    authoritative state and BLOCK on the mismatch, never trust the stale
    "False" claim."""
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(gate_consumption_checker=lambda state_root, gate, design_commit: True)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_SELF_DECLARED_MISMATCH:t2_acquired"


def test_t2_acquired_derivation_calls_gate_checker_with_t2_gate_and_frozen_design_commit():
    seen: list[tuple] = []

    def recording_checker(state_root, gate, design_commit):
        seen.append((state_root, gate, design_commit))
        return False

    run(gate_consumption_checker=recording_checker)
    assert len(seen) == 1
    state_root, gate, design_commit = seen[0]
    assert gate == gate_consumption.GATE_T2_RAW_ACQUISITION
    assert design_commit == recheck.EXPECTED_V8B_FROZEN_DESIGN_COMMIT


def test_gate_consumption_checker_error_propagates_as_blocked():
    def broken_checker(state_root, gate, design_commit):
        raise gate_consumption.V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_STATE_UNAVAILABLE")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run(gate_consumption_checker=broken_checker)
    assert excinfo.value.reason == "V8B_HUMAN_GATE_STATE_UNAVAILABLE"


def test_research_opening_capability_derivation_drives_exposure_fields():
    """If the live check somehow found a research-opening capability
    (hypothetical -- none exists in this repository today), the derived
    exposure fields must flip to True and BLOCK against an honest
    "no exposure" artifact claim, rather than silently trusting the
    artifact."""
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(no_research_opening_api_exists=lambda: False)
    assert excinfo.value.reason.startswith("V8B_T2_REUSE_CONDITIONS_RECHECK_SELF_DECLARED_MISMATCH:t2_")


def test_default_no_research_opening_api_exists_reflects_real_repository_state():
    """The real, non-injected default check against the actual bound
    src.v8b_historical_acquisition module: no open_for_*/research-opening
    API exists today."""
    assert recheck._default_no_research_opening_api_exists() is True


# ---------------------------------------------------------------------------
# Repeat-round finding HIGH-2: V8_STATE.json evidence + dedicated
# LAYER_B/FROZEN_FINAL_CANDIDATE stage-completion approval artifacts.
# ---------------------------------------------------------------------------


def test_layer_b_completion_reader_failure_blocks():
    def failing_reader(head):
        raise recheck.V8BProductionProvenanceBlocked("V8B_LAYER_B_COMPLETION_APPROVAL_MISSING")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(layer_b_completion_reader=failing_reader)
    assert excinfo.value.reason == "V8B_LAYER_B_COMPLETION_APPROVAL_MISSING"


def test_frozen_final_candidate_reader_failure_blocks():
    def failing_reader(head):
        raise recheck.V8BProductionProvenanceBlocked("V8B_FROZEN_FINAL_CANDIDATE_APPROVAL_MISSING")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(frozen_final_candidate_reader=failing_reader)
    assert excinfo.value.reason == "V8B_FROZEN_FINAL_CANDIDATE_APPROVAL_MISSING"


def test_layer_b_completion_checked_before_reading_recheck_artifact():
    def unreachable_reader(root, commit, path):
        raise AssertionError("recheck artifact must not be read if LAYER_B completion already failed")

    def failing_layer_b(head):
        raise recheck.V8BProductionProvenanceBlocked("V8B_LAYER_B_COMPLETION_APPROVAL_MISSING")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked):
        run(layer_b_completion_reader=failing_layer_b, git_object_reader=unreachable_reader)


def test_frozen_final_candidate_checked_before_reading_recheck_artifact():
    def unreachable_reader(root, commit, path):
        raise AssertionError("recheck artifact must not be read if FROZEN_FINAL_CANDIDATE already failed")

    def failing_candidate(head):
        raise recheck.V8BProductionProvenanceBlocked("V8B_FROZEN_FINAL_CANDIDATE_APPROVAL_MISSING")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked):
        run(frozen_final_candidate_reader=failing_candidate, git_object_reader=unreachable_reader)


def test_self_declared_layer_b_and_frozen_final_candidate_still_checked_in_addition():
    """Repeat-round HIGH-2: the two dedicated stage-completion artifacts are
    required IN ADDITION TO, never instead of, the existing self-declared
    fields on the recheck artifact itself -- a well-formed recheck artifact
    honestly declaring layer_b_completed=False must still BLOCK even when
    both dedicated stage-completion readers report PASS."""
    def reader(root, commit, path):
        return json.dumps(_post_freeze_artifact(layer_b_completed=False)).encode("utf-8")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(git_object_reader=reader)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_LAYER_B_NOT_COMPLETE"


def test_v8_state_evidence_reader_receives_repository_root_commit_and_git_object_reader():
    seen: list[tuple] = []

    def recording_reader(root, commit, git_object_reader):
        seen.append((root, commit, git_object_reader))
        return _safe_v8_state_evidence()

    run(v8_state_evidence_reader=recording_reader)
    assert len(seen) == 1
    root, commit, git_object_reader = seen[0]
    assert root == ROOT
    assert commit == SYNTHETIC_COMMIT


def test_v8_state_raw_data_acquired_true_ors_into_t2_acquired_derivation():
    """Even though the durable gate receipt says t2_acquired=False,
    V8_STATE.json's own T2.raw_data_acquired=True is independent
    authoritative evidence that acquisition happened -- the derived value
    must flip to True and BLOCK the artifact's honest "False" claim as a
    mismatch, never silently trust either single source alone."""
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(v8_state_evidence_reader=lambda root, commit, r: _safe_v8_state_evidence(t2_raw_data_acquired=True))
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_SELF_DECLARED_MISMATCH:t2_acquired"


@pytest.mark.parametrize(
    "evidence_overrides",
    [
        {"t2_opened_for_research": True},
        {"t2_sealed_holdout_access_count": 1},
        {"block_assignments_exposed": True},
    ],
)
def test_v8_state_evidence_alone_flips_exposure_fields_even_when_api_absence_says_safe(evidence_overrides):
    """HIGH-2's core fix: "no open_for_* API exists" alone is not sufficient
    -- even with no_research_opening_api_exists()=True (API absent), any
    single V8_STATE.json signal of exposure must still flip the derived
    exposure fields to True (AND-for-safety across independent sources)."""
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run_full(
            no_research_opening_api_exists=lambda: True,
            v8_state_evidence_reader=lambda root, commit, r: _safe_v8_state_evidence(**evidence_overrides),
        )
    assert excinfo.value.reason.startswith("V8B_T2_REUSE_CONDITIONS_RECHECK_SELF_DECLARED_MISMATCH:t2_")


def test_v8_state_evidence_missing_blocks():
    def missing_v8_state(root, commit, git_object_reader):
        raise recheck.V8BGitProvenanceBlocked("GIT_OBJECT_READ_FAILED")

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        run(v8_state_evidence_reader=missing_v8_state)
    assert excinfo.value.reason == "V8_STATE_MISSING"


def test_default_v8_state_evidence_reader_parses_duplicate_key_safe():
    def dup_key_object_reader(root, commit, path):
        return b'{"T2": {}, "T2": {}}'

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck._default_read_v8_state_t2_evidence(ROOT, SYNTHETIC_COMMIT, dup_key_object_reader)
    assert excinfo.value.reason == "V8_STATE_DUPLICATE_KEY"


def test_default_v8_state_evidence_reader_rejects_missing_t2_section():
    def object_reader(root, commit, path):
        return b'{"trust_anchor_pinning": {"block_assignments_exposed": false}}'

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck._default_read_v8_state_t2_evidence(ROOT, SYNTHETIC_COMMIT, object_reader)
    assert excinfo.value.reason == "V8_STATE_T2_SECTION_INVALID"


def test_default_v8_state_evidence_reader_rejects_wrong_typed_field():
    def object_reader(root, commit, path):
        return json.dumps({
            "T2": {"raw_data_acquired": "false", "opened_for_research": False, "sealed_holdout_access_count": None},
            "trust_anchor_pinning": {"block_assignments_exposed": False},
        }).encode()

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck._default_read_v8_state_t2_evidence(ROOT, SYNTHETIC_COMMIT, object_reader)
    assert excinfo.value.reason == "V8_STATE_T2_RAW_DATA_ACQUIRED_INVALID"


def test_default_v8_state_evidence_reader_rejects_boolean_access_count():
    """`sealed_holdout_access_count` must be an int or None -- a bool would
    silently pass a naive ``isinstance(x, int)`` check (bool subclasses
    int), so this must be explicitly rejected."""
    def object_reader(root, commit, path):
        return json.dumps({
            "T2": {"raw_data_acquired": False, "opened_for_research": False, "sealed_holdout_access_count": True},
            "trust_anchor_pinning": {"block_assignments_exposed": False},
        }).encode()

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck._default_read_v8_state_t2_evidence(ROOT, SYNTHETIC_COMMIT, object_reader)
    assert excinfo.value.reason == "V8_STATE_T2_SEALED_HOLDOUT_ACCESS_COUNT_INVALID"


def test_default_v8_state_evidence_reader_against_real_v8_state_json():
    """The real, non-injected default reader against this repository's
    actual current V8_STATE.json at the real HEAD -- proves the production
    default parses the real file's schema successfully."""
    from src.v8b_git_provenance import read_git_object_bytes

    evidence = recheck._default_read_v8_state_t2_evidence(ROOT, _real_head(), read_git_object_bytes)
    assert evidence == {
        "t2_raw_data_acquired": False,
        "t2_opened_for_research": False,
        "t2_sealed_holdout_access_count": None,
        "block_assignments_exposed": False,
    }


def test_public_resolver_wires_real_layer_b_and_frozen_final_candidate_readers():
    """The public zero-arg entrypoint must fail closed on the real repo
    specifically because the two new dedicated stage-completion artifacts
    do not exist yet, once earlier provenance steps are satisfied -- proven
    indirectly: the real production functions are reachable (imported, not
    stubbed) from the public resolver."""
    import inspect

    source = inspect.getsource(recheck.resolve_and_recheck_t2_reuse_conditions)
    assert "read_and_verify_layer_b_completion" in source
    assert "read_and_verify_frozen_final_candidate" in source


# ---------------------------------------------------------------------------
# Round-3 repeat MEDIUM-2: pure/private helper is not a public production API
# ---------------------------------------------------------------------------


def test_private_pure_evaluator_is_not_publicly_exported():
    assert "recheck_t2_reuse_conditions" not in recheck.__all__
    assert not hasattr(recheck, "recheck_t2_reuse_conditions")
    assert hasattr(recheck, "_recheck_t2_reuse_conditions")


def test_only_the_production_resolver_and_safe_constants_are_public():
    assert set(recheck.__all__) == {
        "CANONICAL_REPOSITORY_ROOT",
        "POST_FREEZE_RECHECK_FIELDS",
        "POST_FREEZE_RECHECK_GATE",
        "POST_FREEZE_RECHECK_GIT_PATH",
        "POST_FREEZE_RECHECK_SCHEMA_VERSION",
        "POST_FREEZE_RECHECK_STAGE",
        "REQUIRED_SAFE_METADATA_FIELDS",
        "V8BT2PreservationRecheckBlocked",
        "resolve_and_recheck_t2_reuse_conditions",
    }
