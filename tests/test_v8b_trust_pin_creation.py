from __future__ import annotations

from pathlib import Path

import pytest

from src import v8b_allocation_verification as verification
from src import v8b_trust_pin as trust_pin
from src import v8b_trust_pin_creation as creation

SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_REVIEWED_COMMIT = "b" * 40


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
        trust_pin_review_reader=lambda head, artifact_hash: {"ok": True},
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
            trust_pin_review_reader=forbidden,
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


# ---------------------------------------------------------------------------
# Provenance/freeze/review ordering
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
# HIGH-2 core: the verification summary can only come from the real
# resolver, never an arbitrary caller-supplied mapping.
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
# INDEPENDENT_TRUST_PIN_REVIEW -- fresh, bound to the exact artifact hash.
# ---------------------------------------------------------------------------


def test_missing_trust_pin_review_blocks_after_human_authorization_passes():
    def missing_review(head, artifact_hash):
        raise creation.V8BProductionProvenanceBlocked("V8B_TRUST_PIN_INDEPENDENT_REVIEW_MISSING")

    with pytest.raises(creation.V8BTrustPinCreationBlocked) as excinfo:
        run(trust_pin_review_reader=missing_review)
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_MISSING"


def test_trust_pin_review_reader_called_with_the_exact_verified_artifact_hash(tmp_path):
    verification_result = _pass_verification_result(artifact_self_hash="6" * 64)
    seen: list[tuple] = []

    def recording_review(head, artifact_hash):
        seen.append((head, artifact_hash))

    run(
        verification_result=verification_result,
        trust_pin_review_reader=recording_review,
        output_path=tmp_path / "V8B_TRUSTED_ALLOCATION.json",
    )
    assert seen == [(SYNTHETIC_COMMIT, "6" * 64)]


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

    def poisoned_fsync(fd):
        raise OSError(f"disk full at {secret}")

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
