from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8b_allocation_verification as verification
from src import v8b_trust_pin as trust_pin
from src import v8b_trust_pin_creation as creation

SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_REVIEWED_COMMIT = "b" * 40


def clock_stub():
    return datetime(2026, 8, 13, tzinfo=timezone.utc)


def _pass_verification_result(**overrides) -> dict:
    result = {
        "result": "PASS",
        "logical_block": "T1B",
        "study_name": "V8B_HISTORICAL_RESEARCH",
        "parent_t_spare_ticker_count": 1904,
        "parent_t_spare_ticker_list_sha256": "1" * 64,
        "t1b_ticker_count": 300,
        "t1b_ticker_list_sha256": "2" * 64,
        "remaining_t_spare_ticker_count": 1604,
        "remaining_t_spare_ticker_list_sha256": "3" * 64,
        "artifact_self_hash": "4" * 64,
        "v8b_frozen_design_commit": SYNTHETIC_COMMIT,
        "v8b_allocation_implementation_commit": SYNTHETIC_REVIEWED_COMMIT,
        "parent_v8_partition_manifest_sha256": "0" * 64,
        "parent_v8_partition_implementation_commit": SYNTHETIC_COMMIT,
        "no_membership_choice_based_on_ohlcv_or_data_quality_outcomes": True,
    }
    result.update(overrides)
    return result


def _fresh_state_root() -> Path:
    return Path(tempfile.gettempdir()) / ("v8b_pin_gate_state-" + uuid.uuid4().hex)


def default_deps(**overrides):
    verification_result = overrides.pop("verification_result", None) or _pass_verification_result()
    artifact_hash = verification_result["artifact_self_hash"]
    deps = dict(
        confirmation=creation.PIN_CREATION_CONFIRMATION,
        human_pin_authorization=trust_pin.expected_human_gate(artifact_hash),
        allocation_artifact_path="/dev/null/unreachable/allocation_artifact.json",
        partition_manifest_path="/dev/null/unreachable/partition.json",
        # Deliberately unreachable/unwritable (a path segment under the
        # /dev/null *file*, not a directory) -- every test that expects a
        # successful write overrides this with an explicit tmp_path, so a
        # test that forgets to do so fails loudly with a filesystem error
        # instead of silently writing to (and colliding across runs on) a
        # real, persistent filesystem path.
        output_path="/dev/null/unreachable/V8B_TRUSTED_ALLOCATION.json",
        authorization_note="test-only pin",
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        allocation_verification_resolver=lambda: verification_result,
        clock=clock_stub,
        consumption_state_root=_fresh_state_root(),
    )
    deps.update(overrides)
    return deps


def run(**overrides):
    return creation._create_v8b_trusted_allocation_pin_production_with_dependencies(**default_deps(**overrides))


# ---------------------------------------------------------------------------
# Confirmation token
# ---------------------------------------------------------------------------


def test_confirmation_token_is_frozen_literal():
    assert creation.PIN_CREATION_CONFIRMATION == "V8B_PRODUCTION_CREATE_TRUSTED_ALLOCATION_PIN"


def test_wrong_confirmation_blocks_before_any_dependency_call():
    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called")

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(
            confirmation="wrong",
            git_commit_resolver=forbidden,
            design_freeze_approval_reader=forbidden,
            frozen_design_object_verifier=forbidden,
            reviewed_implementation_binder=forbidden,
            allocation_verification_resolver=forbidden,
            clock=forbidden,
        )
    assert excinfo.value.reason == "V8B_PIN_CREATION_CONFIRMATION_INVALID"


def test_public_entrypoint_rejects_wrong_confirmation():
    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        creation.create_v8b_trusted_allocation_pin_production(
            confirmation="not the real token",
            human_pin_authorization="whatever",
            allocation_artifact_path="/tmp/x",
            partition_manifest_path="/tmp/y",
            output_path="/tmp/z",
            authorization_note="note",
        )
    assert excinfo.value.reason == "V8B_PIN_CREATION_CONFIRMATION_INVALID"


def test_public_entrypoint_offers_no_consumption_state_root_or_review_reader_override():
    import inspect

    public_params = set(inspect.signature(creation.create_v8b_trusted_allocation_pin_production).parameters)
    assert "consumption_state_root" not in public_params
    assert "trust_pin_review_reader" not in public_params
    assert "clock" not in public_params


def test_private_seam_accepts_no_trust_pin_review_reader_parameter():
    import inspect

    private_params = set(
        inspect.signature(creation._create_v8b_trusted_allocation_pin_production_with_dependencies).parameters
    )
    assert "trust_pin_review_reader" not in private_params


# ---------------------------------------------------------------------------
# Provenance/freeze ordering
# ---------------------------------------------------------------------------


def test_git_provenance_failure_blocks_before_allocation_verification():
    def dirty_resolver():
        raise creation.V8BGitProvenanceBlocked("PRODUCTION_GIT_WORKTREE_DIRTY")

    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called")

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(git_commit_resolver=dirty_resolver, allocation_verification_resolver=forbidden)
    assert excinfo.value.reason == "PRODUCTION_GIT_WORKTREE_DIRTY"


def test_freeze_approval_failure_blocks():
    def failing_reader(head):
        raise creation.V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED")

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(design_freeze_approval_reader=failing_reader)
    assert excinfo.value.reason == "V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED"


def test_reviewed_implementation_binder_failure_blocks():
    def failing_binder(head):
        raise creation.V8BProductionProvenanceBlocked("V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING")

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(reviewed_implementation_binder=failing_binder)
    assert excinfo.value.reason == "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING"


# ---------------------------------------------------------------------------
# HIGH-2 (original round): the verification summary can only come from the
# real resolver, never an arbitrary caller-supplied mapping.
# ---------------------------------------------------------------------------


def test_allocation_verification_failure_blocks_before_human_authorization_check():
    def failing_resolver():
        raise verification.V8BAllocationVerificationBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(allocation_verification_resolver=failing_resolver, human_pin_authorization="anything")
    assert excinfo.value.reason == "V8B_ALLOCATION_VERIFICATION_FAILED:TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH"


def test_no_way_to_supply_a_favorable_mapping_directly():
    """There is no ``verification_result_summary`` parameter anywhere on
    either the public or private entrypoint -- the ONLY path to a pin is
    through calling the real resolver."""
    import inspect

    public_params = set(inspect.signature(creation.create_v8b_trusted_allocation_pin_production).parameters)
    assert "verification_result_summary" not in public_params
    assert "verification_result" not in public_params
    private_params = set(
        inspect.signature(creation._create_v8b_trusted_allocation_pin_production_with_dependencies).parameters
    )
    assert "verification_result_summary" not in private_params
    assert "verification_result" not in private_params


# ---------------------------------------------------------------------------
# HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION -- exact frozen grammar
# bound to the verified artifact's own hash.
# ---------------------------------------------------------------------------


def test_wrong_human_pin_authorization_blocks():
    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(human_pin_authorization="V8B_HUMAN_AUTHORIZE_T1B_ALLOCATION_PIN_AT_" + "0" * 64)
    assert excinfo.value.reason == "V8B_HUMAN_PIN_AUTHORIZATION_INVALID"


def test_human_pin_authorization_for_a_different_artifact_hash_blocks():
    """A plausible-looking, well-formed authorization for a DIFFERENT
    artifact hash must not authorize this one."""
    other_hash = "5" * 64
    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(human_pin_authorization=trust_pin.expected_human_gate(other_hash))
    assert excinfo.value.reason == "V8B_HUMAN_PIN_AUTHORIZATION_INVALID"


def test_arbitrary_nonempty_authorization_string_rejected():
    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(human_pin_authorization="I hereby authorize this pin")
    assert excinfo.value.reason == "V8B_HUMAN_PIN_AUTHORIZATION_INVALID"


# ---------------------------------------------------------------------------
# Repeat-round finding HIGH-1: this module no longer depends on
# INDEPENDENT_TRUST_PIN_REVIEW at all -- that gate is strictly downstream,
# verified at T1B acquisition time. Instead it durably, one-shot consumes
# the new HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION gate.
# ---------------------------------------------------------------------------


def test_pin_created_successfully_with_no_review_artifact_involved_at_all(tmp_path):
    output_path = tmp_path / "V8B_TRUSTED_ALLOCATION.json"
    result = run(output_path=output_path)
    assert result["authorization_status"] == "AUTHORIZED"


def test_early_gate_check_precedes_git_provenance_resolution():
    state_root = _fresh_state_root()
    from src.v8b_human_gate_consumption import GATE_PIN_VERIFIED_T1B_ALLOCATION, consume_gate_once
    from src.v8b_production_provenance import EXPECTED_V8B_FROZEN_DESIGN_COMMIT

    consume_gate_once(state_root, GATE_PIN_VERIFIED_T1B_ALLOCATION, EXPECTED_V8B_FROZEN_DESIGN_COMMIT, clock=clock_stub)

    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called -- gate check must precede git provenance resolution")

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(consumption_state_root=state_root, git_commit_resolver=forbidden)
    assert excinfo.value.reason == "V8B_HUMAN_GATE_ALREADY_CONSUMED:" + GATE_PIN_VERIFIED_T1B_ALLOCATION


def test_second_call_with_same_state_root_blocks_before_pin_write(tmp_path):
    state_root = _fresh_state_root()
    output_path_1 = tmp_path / "first" / "V8B_TRUSTED_ALLOCATION.json"
    output_path_2 = tmp_path / "second" / "V8B_TRUSTED_ALLOCATION.json"

    run(consumption_state_root=state_root, output_path=output_path_1)
    assert output_path_1.exists()

    from src.v8b_human_gate_consumption import GATE_PIN_VERIFIED_T1B_ALLOCATION

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(consumption_state_root=state_root, output_path=output_path_2)
    assert excinfo.value.reason == "V8B_HUMAN_GATE_ALREADY_CONSUMED:" + GATE_PIN_VERIFIED_T1B_ALLOCATION
    assert not output_path_2.exists()


def test_gate_consumed_strictly_before_write_never_after(tmp_path, monkeypatch):
    """If the pin write itself fails, the gate must already show consumed
    -- proving consumption happens strictly before, not after, the write."""
    state_root = _fresh_state_root()
    output_path = tmp_path / "private" / "V8B_TRUSTED_ALLOCATION.json"

    call_count = {"n": 0}
    real_fsync = creation.os.fsync

    def poisoned_fsync(fd):
        # Let the gate-consumption receipt's own fsync (step 6) succeed;
        # only the pin's own staging-write fsync (step 7, called second)
        # fails.
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise OSError("disk full")
        return real_fsync(fd)

    monkeypatch.setattr(creation.os, "fsync", poisoned_fsync)
    with pytest.raises(creation.V8BTrustPinCreationBlocked):
        run(consumption_state_root=state_root, output_path=output_path)

    from src.v8b_human_gate_consumption import GATE_PIN_VERIFIED_T1B_ALLOCATION, has_gate_been_consumed
    from src.v8b_production_provenance import EXPECTED_V8B_FROZEN_DESIGN_COMMIT

    assert has_gate_been_consumed(state_root, GATE_PIN_VERIFIED_T1B_ALLOCATION, EXPECTED_V8B_FROZEN_DESIGN_COMMIT)


def test_confirmation_failure_never_consumes_the_gate():
    state_root = _fresh_state_root()
    with pytest.raises(creation.V8BTrustPinCreationBlocked):
        run(consumption_state_root=state_root, confirmation="wrong")

    from src.v8b_human_gate_consumption import GATE_PIN_VERIFIED_T1B_ALLOCATION, has_gate_been_consumed
    from src.v8b_production_provenance import EXPECTED_V8B_FROZEN_DESIGN_COMMIT

    assert not has_gate_been_consumed(state_root, GATE_PIN_VERIFIED_T1B_ALLOCATION, EXPECTED_V8B_FROZEN_DESIGN_COMMIT)


def test_wrong_human_authorization_never_consumes_the_gate():
    state_root = _fresh_state_root()
    with pytest.raises(creation.V8BTrustPinCreationBlocked):
        run(consumption_state_root=state_root, human_pin_authorization="not the real authorization")

    from src.v8b_human_gate_consumption import GATE_PIN_VERIFIED_T1B_ALLOCATION, has_gate_been_consumed
    from src.v8b_production_provenance import EXPECTED_V8B_FROZEN_DESIGN_COMMIT

    assert not has_gate_been_consumed(state_root, GATE_PIN_VERIFIED_T1B_ALLOCATION, EXPECTED_V8B_FROZEN_DESIGN_COMMIT)


# ---------------------------------------------------------------------------
# Successful synthetic pin creation: write-once, safe return value.
# ---------------------------------------------------------------------------


def test_successful_synthetic_pin_creation_atomic_write_and_safe_return(tmp_path):
    output_path = tmp_path / "private" / "V8B_TRUSTED_ALLOCATION.json"
    result = run(output_path=output_path)
    assert result["authorization_status"] == "AUTHORIZED"
    assert set(result) == set(trust_pin.TRUST_PIN_FIELDS)
    # No ticker-identity field can ever appear -- schema-enforced.
    assert "t1b_tickers" not in result
    assert "remaining_t_spare_tickers" not in result
    assert "parent_t_spare_tickers" not in result

    written = trust_pin.validate_trust_pin(
        __import__("json").loads(output_path.read_bytes())
    )
    assert written == result


def test_pin_never_overwrites_existing_destination(tmp_path):
    output_path = tmp_path / "private" / "V8B_TRUSTED_ALLOCATION.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"pre-existing")

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(output_path=output_path)
    assert excinfo.value.reason == "V8B_TRUSTED_ALLOCATION_PIN_ALREADY_EXISTS"
    assert output_path.read_bytes() == b"pre-existing"


def test_staging_write_failure_never_leaks_private_path(tmp_path, monkeypatch):
    secret = "/very/secret/private/path"
    call_count = {"n": 0}
    real_fsync = creation.os.fsync

    def poisoned_fsync(fd):
        # The gate-consumption receipt is fsync'd first (step 6), strictly
        # before the pin's own staging write (step 7) -- only poison the
        # second-and-later fsync call so the gate consumption itself
        # succeeds and we're testing the pin write's own failure path.
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise OSError(f"disk full at {secret}")
        return real_fsync(fd)

    monkeypatch.setattr(creation.os, "fsync", poisoned_fsync)
    output_path = tmp_path / "private" / "V8B_TRUSTED_ALLOCATION.json"

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(output_path=output_path)
    assert excinfo.value.reason == "V8B_TRUSTED_ALLOCATION_PIN_STAGING_WRITE_FAILED"
    assert secret not in excinfo.value.reason


# ---------------------------------------------------------------------------
# Not executed against the real repository today.
# ---------------------------------------------------------------------------


def test_real_production_entrypoint_fails_closed_on_real_repo(tmp_path):
    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        creation.create_v8b_trusted_allocation_pin_production(
            confirmation=creation.PIN_CREATION_CONFIRMATION,
            human_pin_authorization="irrelevant-fails-earlier",
            allocation_artifact_path=tmp_path / "nonexistent.json",
            partition_manifest_path=tmp_path / "nonexistent2.json",
            output_path=tmp_path / "out.json",
            authorization_note="note",
        )
    assert excinfo.value.reason in {
        "PRODUCTION_GIT_WORKTREE_DIRTY",
        "PRODUCTION_GIT_HEAD_NOT_ORIGIN",
        "PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE",
        "V8B_DESIGN_FREEZE_APPROVAL_MISSING",
        "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING",
    }


def test_no_real_pin_created_in_this_repository():
    assert not (creation.CANONICAL_REPOSITORY_ROOT / creation.PIN_ARTIFACT_FILENAME).exists()
