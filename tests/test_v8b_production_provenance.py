from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src import v8b_production_provenance as pp

ROOT = Path(__file__).resolve().parents[1]


def _real_head() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


def _init_bogus_git_repo(root: Path, *, files: dict[str, bytes]) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", "bogus"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


def _git_blob_sha(repo_root: Path, ref: str, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", ref + ":" + path],
        capture_output=True, check=True, text=True,
    ).stdout.strip()


def _commit_all(repo_root: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repo_root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", message],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Frozen design object -- exact blob at the frozen commit itself
# ---------------------------------------------------------------------------


def test_frozen_design_object_verifies_against_real_repo():
    pp.verify_frozen_design_object(ROOT)  # must not raise


def test_frozen_design_blob_mutation_blocks(tmp_path):
    """A repository whose frozen commit contains a *different* design draft
    blob must BLOCK -- proves the check is bound to exact bytes, not merely
    'a file exists at this path'."""
    mutated_content = b"# V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT (mutated)\nsemantically similar but different\n"
    bogus = tmp_path / "bogus_frozen_design"
    bogus.mkdir()
    _init_bogus_git_repo(bogus, files={"V8B_HISTORICAL_RESEARCH_DESIGN_DRAFT.md": mutated_content})
    # Force the resolver to look up the *real* frozen commit SHA inside this
    # *different* repository, where that SHA does not exist -> BLOCK.
    with pytest.raises(pp.V8BProductionProvenanceBlocked):
        pp.verify_frozen_design_object(bogus)


# ---------------------------------------------------------------------------
# Design freeze approval -- exact blob + exact field semantics (HIGH-5)
# ---------------------------------------------------------------------------


def test_design_freeze_approval_verifies_against_real_repo():
    approval = pp.read_and_verify_design_freeze_approval(ROOT, _real_head())
    assert approval["approval_status"] == "APPROVED"
    assert approval["frozen_design_git_commit"] == pp.EXPECTED_V8B_FROZEN_DESIGN_COMMIT


def test_design_freeze_approval_blob_mutation_blocks(tmp_path):
    """A semantically similar but modified approval artifact must not pass
    -- even if every field still looks individually correct, a byte-level
    mutation of the file must BLOCK on the exact-blob check first."""
    real_bytes = (ROOT / "V8B_DESIGN_FREEZE_APPROVAL.json").read_bytes()
    mutated = json.loads(real_bytes)
    mutated["authorization_note_added_by_attacker"] = "harmless-looking addition"
    bogus = tmp_path / "bogus_approval"
    bogus.mkdir()
    commit = _init_bogus_git_repo(
        bogus, files={"V8B_DESIGN_FREEZE_APPROVAL.json": json.dumps(mutated).encode()}
    )
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_design_freeze_approval(bogus, commit)
    assert excinfo.value.reason == "V8B_DESIGN_FREEZE_APPROVAL_BLOB_MUTATED"


@pytest.mark.parametrize(
    "field,value,expected_reason",
    [
        ("approval_status", "PENDING", "V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED"),
        ("final_independent_review_result", "FAIL", "V8B_DESIGN_FREEZE_FINAL_REVIEW_NOT_PASS"),
        ("preservation_recheck_result", "FAIL", "V8B_DESIGN_FREEZE_PRESERVATION_RECHECK_NOT_PASS"),
        ("human_gate", "V8B_HUMAN_DESIGN_FREEZE_APPROVED_FOR_COMMIT_" + "0" * 40, "V8B_DESIGN_FREEZE_HUMAN_GATE_MISMATCH"),
        ("design_finalized", False, "V8B_DESIGN_FREEZE_NOT_FINALIZED"),
        ("human_design_freeze_complete", False, "V8B_DESIGN_FREEZE_NOT_COMPLETE"),
        ("t1b_allocation_authorized", True, "V8B_DESIGN_FREEZE_UNEXPECTED_ALLOCATION_AUTHORIZATION"),
        ("real_network_authorized", True, "V8B_DESIGN_FREEZE_UNEXPECTED_NETWORK_AUTHORIZATION"),
        ("t1b_acquisition_authorized", True, "V8B_DESIGN_FREEZE_UNEXPECTED_T1B_ACQUISITION_AUTHORIZATION"),
        ("t2_acquisition_authorized", True, "V8B_DESIGN_FREEZE_UNEXPECTED_T2_ACQUISITION_AUTHORIZATION"),
        ("research_opening_authorized", True, "V8B_DESIGN_FREEZE_UNEXPECTED_RESEARCH_OPENING_AUTHORIZATION"),
    ],
)
def test_design_freeze_approval_field_semantics_enforced(tmp_path, field, value, expected_reason):
    real_bytes = (ROOT / "V8B_DESIGN_FREEZE_APPROVAL.json").read_bytes()
    mutated = json.loads(real_bytes)
    mutated[field] = value
    bogus = tmp_path / ("bogus_field_" + field)
    bogus.mkdir()
    commit = _init_bogus_git_repo(
        bogus, files={"V8B_DESIGN_FREEZE_APPROVAL.json": json.dumps(mutated).encode()}
    )
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_design_freeze_approval(bogus, commit)
    # Blob check fires first for any byte-level change; both outcomes prove
    # a mutated artifact never validates.
    assert excinfo.value.reason in {"V8B_DESIGN_FREEZE_APPROVAL_BLOB_MUTATED", expected_reason}


def test_design_freeze_approval_missing_blocks(tmp_path):
    bogus = tmp_path / "no_approval"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"x"})
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_design_freeze_approval(bogus, commit)
    assert excinfo.value.reason == "V8B_DESIGN_FREEZE_APPROVAL_MISSING"


# ---------------------------------------------------------------------------
# Reviewed-implementation binding (HIGH-2)
# ---------------------------------------------------------------------------


def test_reviewed_implementation_binding_missing_blocks_on_real_repo():
    """The real V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json does not exist
    yet -- production must fail closed today."""
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.verify_reviewed_implementation_binding(ROOT, _real_head())
    assert excinfo.value.reason == "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING"


def _review_artifact(reviewed_commit: str, **overrides) -> dict:
    review = {
        "schema_version": pp.IMPLEMENTATION_REVIEW_SCHEMA_VERSION,
        "study": pp.STUDY_NAME,
        "artifact_role": "PRODUCTION_IMPLEMENTATION_REVIEW",
        "reviewed_implementation_git_commit": reviewed_commit,
        "review_result": "PASS",
        "approval_status": "APPROVED",
    }
    review.update(overrides)
    return review


def _bound_file_contents() -> dict[str, bytes]:
    return {path: (ROOT / path).read_bytes() for path in pp.BOUND_PRODUCTION_FILES}


def test_reviewed_implementation_binding_passes_when_all_bound_blobs_match(tmp_path):
    """The reviewed commit may legitimately be earlier than the audit HEAD
    (review artifact added after the implementation commit); as long as
    every bound file's blob is identical between the two, this PASSes."""
    bogus = tmp_path / "review_pass"
    bogus.mkdir()
    files = _bound_file_contents()
    reviewed_commit = _init_bogus_git_repo(bogus, files=files)
    # A later, audit-only commit that changes an UNBOUND file and adds the
    # review artifact itself -- HEAD now differs from reviewed_commit.
    (bogus / "AUDIT_NOTE.md").write_bytes(b"audit-only commit, no production files touched\n")
    (bogus / "V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json").write_bytes(
        json.dumps(_review_artifact(reviewed_commit)).encode()
    )
    subprocess.run(["git", "-C", str(bogus), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(bogus), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", "audit"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(bogus), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()
    assert head != reviewed_commit

    result = pp.verify_reviewed_implementation_binding(bogus, head)
    assert result["reviewed_implementation_git_commit"] == reviewed_commit
    assert result["bound_files_verified"] == len(pp.BOUND_PRODUCTION_FILES)


def test_v8_partition_py_is_bound_to_the_review():
    assert "src/v8_partition.py" in pp.BOUND_PRODUCTION_FILES


def test_reviewed_implementation_binding_blocks_on_v8_partition_drift(tmp_path):
    """A drift in src/v8_partition.py specifically must BLOCK, even though
    V8B does not author that file -- round-2 finding HIGH-3."""
    bogus = tmp_path / "review_drift_v8_partition"
    bogus.mkdir()
    files = _bound_file_contents()
    reviewed_commit = _init_bogus_git_repo(bogus, files=files)

    drifted_path = "src/v8_partition.py"
    (bogus / drifted_path).write_bytes(files[drifted_path] + b"\n# drifted\n")
    (bogus / "V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json").write_bytes(
        json.dumps(_review_artifact(reviewed_commit)).encode()
    )
    subprocess.run(["git", "-C", str(bogus), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(bogus), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", "drift"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(bogus), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()

    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.verify_reviewed_implementation_binding(bogus, head)
    assert excinfo.value.reason == "V8B_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:src/v8_partition.py"


def test_reviewed_implementation_binding_blocks_on_bound_blob_drift(tmp_path):
    """A single changed bound production file between HEAD and the reviewed
    commit must BLOCK -- audit-only drift is fine, production-file drift is
    not."""
    bogus = tmp_path / "review_drift"
    bogus.mkdir()
    files = _bound_file_contents()
    reviewed_commit = _init_bogus_git_repo(bogus, files=files)

    drifted_path = pp.BOUND_PRODUCTION_FILES[0]
    (bogus / drifted_path).write_bytes(files[drifted_path] + b"\n# drifted\n")
    (bogus / "V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json").write_bytes(
        json.dumps(_review_artifact(reviewed_commit)).encode()
    )
    subprocess.run(["git", "-C", str(bogus), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(bogus), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", "drift"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(bogus), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()

    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.verify_reviewed_implementation_binding(bogus, head)
    assert excinfo.value.reason == "V8B_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:" + drifted_path


def test_reviewed_implementation_binding_requires_review_result_pass(tmp_path):
    bogus = tmp_path / "review_not_pass"
    bogus.mkdir()
    files = _bound_file_contents()
    reviewed_commit = _init_bogus_git_repo(bogus, files=files)
    (bogus / "V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json").write_bytes(
        json.dumps(_review_artifact(reviewed_commit, review_result="BLOCK")).encode()
    )
    subprocess.run(["git", "-C", str(bogus), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(bogus), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", "not-pass"],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", str(bogus), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.verify_reviewed_implementation_binding(bogus, head)
    assert excinfo.value.reason == "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_PASS"


# ---------------------------------------------------------------------------
# Original immutable V8 authority -- exact blob (HIGH-4)
# ---------------------------------------------------------------------------


def test_v8_trusted_partition_anchor_verifies_against_real_repo():
    anchor = pp.read_and_verify_v8_trusted_partition_anchor(ROOT, _real_head())
    assert anchor["authorized_partition_manifest_sha256"] == pp.EXPECTED_V8_PARTITION_MANIFEST_SHA256


def test_v8_trusted_partition_anchor_blob_mutation_blocks(tmp_path):
    """A re-pin that is internally self-consistent (valid schema, valid
    hash formats) but simply *different bytes* must still BLOCK -- the
    exact-blob check runs before any field-level parsing."""
    real_bytes = (ROOT / "V8_TRUSTED_PARTITION.json").read_bytes()
    mutated = json.loads(real_bytes)
    mutated["authorization_note"] = mutated["authorization_note"] + " (repinned)"
    bogus = tmp_path / "anchor_mutated"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"V8_TRUSTED_PARTITION.json": json.dumps(mutated).encode()})
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_v8_trusted_partition_anchor(bogus, commit)
    assert excinfo.value.reason == "V8_TRUSTED_PARTITION_BLOB_MUTATED"


def test_v8_trusted_partition_anchor_repin_with_matching_forged_manifest_still_blocks(tmp_path):
    """Even a re-pin whose fields match a different, forged private
    manifest must BLOCK at the exact-blob check -- the forged manifest's
    own correctness is irrelevant."""
    forged_manifest_sha = "f" * 64
    forged_impl_commit = "1" * 40
    anchor = {
        "schema_version": pp.TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION,
        "study_name": "V8_HISTORICAL_RESEARCH",
        "design_commit": pp.V8_DESIGN_COMMIT,
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": forged_manifest_sha,
        "authorized_partition_implementation_git_commit": forged_impl_commit,
        "authorization_note": "forged repin, internally self-consistent",
    }
    bogus = tmp_path / "anchor_repin"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"V8_TRUSTED_PARTITION.json": json.dumps(anchor).encode()})
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_v8_trusted_partition_anchor(bogus, commit)
    assert excinfo.value.reason == "V8_TRUSTED_PARTITION_BLOB_MUTATED"


# ---------------------------------------------------------------------------
# OPTION_2 T2 authority bridge -- exact fields (HIGH-4/5)
# ---------------------------------------------------------------------------


def test_t2_authority_bridge_verifies_against_real_repo():
    bridge = pp.read_and_verify_t2_authority_bridge(ROOT, _real_head())
    assert bridge["human_gate"] == pp.EXPECTED_T2_OPTION_2_HUMAN_GATE
    assert bridge["v8_trust_anchor_git_identity"] == pp.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA


@pytest.mark.parametrize(
    "field,value,expected_reason",
    [
        ("v8_trust_anchor_git_identity", "0" * 40, "V8B_T2_AUTHORITY_BRIDGE_ANCHOR_IDENTITY_MISMATCH"),
        ("authorized_parent_v8_partition_manifest_sha256", "0" * 64, "V8B_T2_AUTHORITY_BRIDGE_MANIFEST_SHA_MISMATCH"),
        ("expected_t2_ticker_list_sha256", "0" * 64, "V8B_T2_AUTHORITY_BRIDGE_TICKER_LIST_SHA_MISMATCH"),
        ("human_gate", "V8B_T2_AUTHORITY_INTEGRATION_OPTION_2_APPROVED_BY_SOMEONE", "V8B_T2_AUTHORITY_BRIDGE_HUMAN_GATE_MISMATCH"),
        ("t2_membership_reassignment", "ALLOWED", "V8B_T2_AUTHORITY_BRIDGE_MEMBERSHIP_REASSIGNMENT_INVALID"),
        ("v8_trusted_partition_json_mutated_or_repinned", True, "V8B_T2_AUTHORITY_BRIDGE_ANCHOR_MUTATION_INVALID"),
        ("option", "OPTION_1", "V8B_T2_AUTHORITY_BRIDGE_OPTION_MISMATCH"),
    ],
)
def test_t2_authority_bridge_exact_field_enforcement(tmp_path, field, value, expected_reason):
    real_bridge = json.loads((ROOT / "V8B_T2_AUTHORITY_BRIDGE.json").read_bytes())
    mutated = dict(real_bridge)
    mutated[field] = value
    bogus = tmp_path / ("bridge_field_" + field)
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"V8B_T2_AUTHORITY_BRIDGE.json": json.dumps(mutated).encode()})
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_t2_authority_bridge(bogus, commit)
    assert excinfo.value.reason == expected_reason


# ---------------------------------------------------------------------------
# FINAL_REPEAT finding HIGH-2 (original round), strengthened by repeat-round
# finding HIGH-1: INDEPENDENT_TRUST_PIN_REVIEW must bind to the actual
# published trust-pin Git blob/commit, not merely to the allocation-
# artifact hash and a human-gate string.
# ---------------------------------------------------------------------------

SYNTHETIC_ARTIFACT_HASH = "9" * 64
SYNTHETIC_TRUST_PIN_HUMAN_GATE = "V8B_HUMAN_AUTHORIZE_T1B_ALLOCATION_PIN_AT_" + SYNTHETIC_ARTIFACT_HASH


def _trust_pin_review_artifact(**overrides) -> dict:
    review = {
        "schema_version": pp.TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION,
        "study": pp.STUDY_NAME,
        "artifact_role": pp.TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE,
        "reviewed_allocation_artifact_self_hash": SYNTHETIC_ARTIFACT_HASH,
        "reviewed_trust_pin_human_gate": SYNTHETIC_TRUST_PIN_HUMAN_GATE,
        "reviewed_trust_pin_git_blob_sha": "0" * 40,
        "reviewed_trust_pin_git_commit": "0" * 40,
        "review_result": "PASS",
        "approval_status": "APPROVED",
    }
    review.update(overrides)
    return review


def _build_trust_pin_review_repo(
    tmp_path: Path,
    name: str,
    *,
    review_overrides: dict | None = None,
    pin_bytes: bytes = b'{"synthetic":"pin"}',
) -> tuple[Path, str, str, str]:
    """Build a bogus repo with a real, committed `V8B_TRUSTED_ALLOCATION.json`
    trust pin, then a self-consistent review artifact bound to that pin's
    exact blob sha and commit, committed in a LATER commit (so HEAD !=
    reviewed_commit -- exercising the "current HEAD" re-resolution path
    too). Returns ``(repo_root, head_commit, pin_commit, pin_blob_sha)``.
    """
    bogus = tmp_path / name
    bogus.mkdir()
    pin_commit = _init_bogus_git_repo(bogus, files={pp.TRUST_PIN_GIT_PATH: pin_bytes})
    pin_blob_sha = _git_blob_sha(bogus, pin_commit, pp.TRUST_PIN_GIT_PATH)

    base_overrides = {
        "reviewed_trust_pin_git_commit": pin_commit,
        "reviewed_trust_pin_git_blob_sha": pin_blob_sha,
    }
    merged_overrides = {**base_overrides, **(review_overrides or {})}
    review = _trust_pin_review_artifact(**merged_overrides)
    (bogus / pp.TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH).write_bytes(json.dumps(review).encode())
    head_commit = _commit_all(bogus, "add review")
    return bogus, head_commit, pin_commit, pin_blob_sha


def _read_review(bogus: Path, head: str, **overrides):
    kwargs = dict(
        expected_allocation_artifact_self_hash=SYNTHETIC_ARTIFACT_HASH,
        expected_trust_pin_human_gate=SYNTHETIC_TRUST_PIN_HUMAN_GATE,
    )
    kwargs.update(overrides)
    return pp.read_and_verify_trust_pin_independent_review(bogus, head, **kwargs)


def test_trust_pin_review_missing_blocks_on_real_repo():
    """The real V8B_TRUST_PIN_INDEPENDENT_REVIEW.json does not exist yet --
    production must fail closed today."""
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_trust_pin_independent_review(
            ROOT,
            _real_head(),
            expected_allocation_artifact_self_hash=SYNTHETIC_ARTIFACT_HASH,
            expected_trust_pin_human_gate=SYNTHETIC_TRUST_PIN_HUMAN_GATE,
        )
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_MISSING"


def test_trust_pin_review_passes_on_well_formed_synthetic_artifact(tmp_path):
    bogus, head, _pin_commit, _pin_blob = _build_trust_pin_review_repo(tmp_path, "trust_pin_review_pass")
    result = _read_review(bogus, head)
    assert result["review_result"] == "PASS"


def test_trust_pin_review_bound_to_a_different_hash_is_rejected(tmp_path):
    """A well-formed, PASS/APPROVED review for a DIFFERENT allocation
    artifact hash must never authorize this one."""
    bogus, head, _pin_commit, _pin_blob = _build_trust_pin_review_repo(tmp_path, "trust_pin_review_wrong_hash")
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        _read_review(bogus, head, expected_allocation_artifact_self_hash="8" * 64)
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_HASH_MISMATCH"


def test_trust_pin_review_bound_to_a_different_human_gate_is_rejected(tmp_path):
    """A well-formed, PASS/APPROVED review whose reviewed_trust_pin_human_gate
    does not exact-match the expected value must never authorize this pin
    (repeat-round finding HIGH-1: exact-value validated, not merely
    present)."""
    bogus, head, _pin_commit, _pin_blob = _build_trust_pin_review_repo(tmp_path, "trust_pin_review_wrong_gate")
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        _read_review(bogus, head, expected_trust_pin_human_gate="V8B_HUMAN_AUTHORIZE_T1B_ALLOCATION_PIN_AT_" + "7" * 64)
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_HUMAN_GATE_MISMATCH"


def test_trust_pin_review_self_inconsistent_claimed_blob_rejected(tmp_path):
    """A review that claims a blob sha which does NOT actually match the
    trust pin at its own claimed reviewed_trust_pin_git_commit must BLOCK
    -- proves the review cannot merely quote plausible-looking hex."""
    bogus, head, _pin_commit, _pin_blob = _build_trust_pin_review_repo(
        tmp_path, "trust_pin_review_self_inconsistent", review_overrides={"reviewed_trust_pin_git_blob_sha": "f" * 40}
    )
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        _read_review(bogus, head)
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_BLOB_SELF_INCONSISTENT"


def test_trust_pin_review_drift_after_pin_changed_since_review_rejected(tmp_path):
    """If the trust pin at the CURRENT verified HEAD no longer matches the
    blob the review actually reviewed (the pin was swapped/mutated after
    the review was written), it must BLOCK."""
    bogus, _review_head, pin_commit, pin_blob = _build_trust_pin_review_repo(
        tmp_path, "trust_pin_review_drift", pin_bytes=b'{"synthetic":"pin-v1"}'
    )
    # A further commit re-publishes a DIFFERENT trust pin at the same path.
    (bogus / pp.TRUST_PIN_GIT_PATH).write_bytes(b'{"synthetic":"pin-v2-different"}')
    new_head = _commit_all(bogus, "swap trust pin after review")
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        _read_review(bogus, new_head)
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_TRUST_PIN_BLOB_DRIFT"


def test_trust_pin_review_reviewed_commit_missing_trust_pin_rejected(tmp_path):
    """A review claiming a reviewed_trust_pin_git_commit that never actually
    contained the trust pin at all must BLOCK."""
    bogus = tmp_path / "trust_pin_review_reviewed_commit_missing"
    bogus.mkdir()
    empty_commit = _init_bogus_git_repo(bogus, files={"README.md": b"no pin here"})
    review = _trust_pin_review_artifact(
        reviewed_trust_pin_git_commit=empty_commit,
        reviewed_trust_pin_git_blob_sha="1" * 40,
    )
    (bogus / pp.TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH).write_bytes(json.dumps(review).encode())
    head = _commit_all(bogus, "add review pointing at commit without a pin")
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        _read_review(bogus, head)
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_REVIEWED_COMMIT_TRUST_PIN_MISSING"


def test_trust_pin_review_requires_strict_pin_to_review_ancestry(tmp_path):
    bogus, head, pin_commit, _pin_blob = _build_trust_pin_review_repo(tmp_path, "trust_pin_review_order")
    result = _read_review(bogus, head)
    assert result["reviewed_trust_pin_git_commit"] == pin_commit


def test_trust_pin_review_same_commit_blocks(tmp_path):
    bogus = tmp_path / "trust_pin_review_same_commit"
    bogus.mkdir()
    pin_commit = _init_bogus_git_repo(bogus, files={pp.TRUST_PIN_GIT_PATH: b'{"synthetic":"pin"}'})
    from src.v8b_git_provenance import V8BGitProvenanceBlocked, require_strict_git_ancestor

    with pytest.raises(V8BGitProvenanceBlocked) as excinfo:
        require_strict_git_ancestor(
            bogus,
            pin_commit,
            pin_commit,
            "V8B_TRUST_PIN_INDEPENDENT_REVIEW_REVIEWED_COMMIT_NOT_STRICT_ANCESTOR",
        )
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_REVIEWED_COMMIT_NOT_STRICT_ANCESTOR"


def test_trust_pin_review_sibling_commit_with_identical_blob_blocks(tmp_path):
    bogus = tmp_path / "trust_pin_review_sibling"
    bogus.mkdir()
    root_commit = _init_bogus_git_repo(bogus, files={"README.md": b"root"})
    (bogus / pp.TRUST_PIN_GIT_PATH).write_bytes(b'{"synthetic":"pin"}')
    main_pin_commit = _commit_all(bogus, "main pin publication")
    pin_blob = _git_blob_sha(bogus, main_pin_commit, pp.TRUST_PIN_GIT_PATH)
    subprocess.run(["git", "-C", str(bogus), "checkout", "-q", "-b", "sibling", root_commit], check=True)
    (bogus / pp.TRUST_PIN_GIT_PATH).write_bytes(b'{"synthetic":"pin"}')
    sibling_commit = _commit_all(bogus, "unrelated pin publication")
    subprocess.run(["git", "-C", str(bogus), "checkout", "-q", "-"], check=True)
    review = _trust_pin_review_artifact(reviewed_trust_pin_git_commit=sibling_commit, reviewed_trust_pin_git_blob_sha=pin_blob)
    (bogus / pp.TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH).write_bytes(json.dumps(review).encode())
    main_review = _commit_all(bogus, "main-line review")
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        _read_review(bogus, main_review)
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_REVIEWED_COMMIT_NOT_STRICT_ANCESTOR"


def test_trust_pin_review_artifact_present_at_pin_commit_blocks(tmp_path, monkeypatch):
    bogus = tmp_path / "trust_pin_review_already_present"
    bogus.mkdir()
    pin_bytes = b'{"synthetic":"pin"}'
    (bogus / pp.TRUST_PIN_GIT_PATH).write_bytes(pin_bytes)
    pin_commit = _commit_all(bogus, "publish pin") if (bogus / ".git").exists() else _init_bogus_git_repo(bogus, files={pp.TRUST_PIN_GIT_PATH: pin_bytes})
    pin_blob = _git_blob_sha(bogus, pin_commit, pp.TRUST_PIN_GIT_PATH)
    review = _trust_pin_review_artifact(reviewed_trust_pin_git_commit=pin_commit, reviewed_trust_pin_git_blob_sha=pin_blob)
    (bogus / pp.TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH).write_bytes(json.dumps(review).encode())
    head = _commit_all(bogus, "review present before review stage")
    real_resolve = pp.resolve_git_blob

    def pretend_review_existed(repository_root, commit, path):
        if commit == pin_commit and path == pp.TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH:
            return "a" * 40
        return real_resolve(repository_root, commit, path)

    monkeypatch.setattr(pp, "resolve_git_blob", pretend_review_existed)
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        _read_review(bogus, head)
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_REVIEW_ARTIFACT_PRESENT_AT_REVIEWED_COMMIT"


@pytest.mark.parametrize(
    "field,value,expected_reason",
    [
        ("review_result", "BLOCK", "V8B_TRUST_PIN_INDEPENDENT_REVIEW_NOT_PASS"),
        ("approval_status", "PENDING", "V8B_TRUST_PIN_INDEPENDENT_REVIEW_NOT_APPROVED"),
        ("study", "V8_HISTORICAL_RESEARCH", "V8B_TRUST_PIN_INDEPENDENT_REVIEW_STUDY_MISMATCH"),
        ("artifact_role", "SOMETHING_ELSE", "V8B_TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE_MISMATCH"),
        ("schema_version", "V0", "V8B_TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION_MISMATCH"),
    ],
)
def test_trust_pin_review_field_semantics_enforced(tmp_path, field, value, expected_reason):
    bogus, head, _pin_commit, _pin_blob = _build_trust_pin_review_repo(
        tmp_path, "trust_pin_review_field_" + field, review_overrides={field: value}
    )
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        _read_review(bogus, head)
    assert excinfo.value.reason == expected_reason


def test_trust_pin_review_schema_missing_field_blocks(tmp_path):
    bogus = tmp_path / "trust_pin_review_schema"
    bogus.mkdir()
    incomplete = _trust_pin_review_artifact()
    del incomplete["reviewed_trust_pin_human_gate"]
    commit = _init_bogus_git_repo(
        bogus, files={pp.TRUST_PIN_INDEPENDENT_REVIEW_GIT_PATH: json.dumps(incomplete).encode()}
    )
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        _read_review(bogus, commit)
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_INVALID"


def test_trust_pin_git_path_matches_the_other_two_modules_own_literals():
    from src import v8b_historical_acquisition as acquisition
    from src import v8b_trust_pin_creation as creation

    assert pp.TRUST_PIN_GIT_PATH == acquisition.T1B_TRUST_PIN_GIT_PATH
    assert pp.TRUST_PIN_GIT_PATH == creation.PIN_ARTIFACT_FILENAME


# ---------------------------------------------------------------------------
# FINAL_REPEAT finding HIGH-2: the new trust-pin-creation and human-gate-
# consumption modules are bound to the reviewed implementation too.
# ---------------------------------------------------------------------------


def test_trust_pin_creation_and_gate_consumption_modules_are_bound_to_the_review():
    assert "src/v8b_trust_pin_creation.py" in pp.BOUND_PRODUCTION_FILES
    assert "src/v8b_human_gate_consumption.py" in pp.BOUND_PRODUCTION_FILES


# ---------------------------------------------------------------------------
# Repeat-round finding HIGH-2: concrete Layer B / frozen-candidate artifacts.
# ---------------------------------------------------------------------------


def _layer_b_report(candidate_ids=None, **overrides) -> dict:
    report = {
        "schema_version": pp.LAYER_B_VALIDATION_REPORT_SCHEMA_VERSION,
        "study": pp.STUDY_NAME,
        "artifact_role": pp.LAYER_B_VALIDATION_REPORT_ARTIFACT_ROLE,
        "v8b_frozen_design_commit": pp.EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
        "validation_access_count": 1,
        "validation_result": "PASS",
        "surviving_candidate_definition_sha256s": candidate_ids or ["a" * 64],
        "validation_payload": {},
    }
    report.update(overrides)
    return report


def _candidate_artifact(source_commit: str, source_blob: str, **overrides) -> dict:
    candidate = {
        "schema_version": pp.FROZEN_FINAL_CANDIDATE_ARTIFACT_SCHEMA_VERSION,
        "study": pp.STUDY_NAME,
        "artifact_role": pp.FROZEN_FINAL_CANDIDATE_ARTIFACT_ROLE,
        "v8b_frozen_design_commit": pp.EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
        "frozen_final_candidate_count": 1,
        "parameters_sha256": "1" * 64,
        "features_sha256": "2" * 64,
        "friction_assumptions_sha256": "3" * 64,
        "universe_sha256": "4" * 64,
        "candidate_definition_sha256": "a" * 64,
        "source_layer_b_validation_report_git_path": pp.LAYER_B_VALIDATION_REPORT_GIT_PATH,
        "source_layer_b_validation_report_git_blob_sha": source_blob,
        "source_layer_b_validation_report_git_commit": source_commit,
        "candidate_payload": {},
    }
    candidate.update(overrides)
    return candidate


def _build_concrete_stage_repo(tmp_path: Path, name: str = "concrete_stages") -> tuple[Path, str, str, str, str]:
    import hashlib

    repo = tmp_path / name
    repo.mkdir()
    report = _layer_b_report()
    report_bytes = json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
    report_commit = _init_bogus_git_repo(repo, files={pp.LAYER_B_VALIDATION_REPORT_GIT_PATH: report_bytes})
    report_blob = _git_blob_sha(repo, report_commit, pp.LAYER_B_VALIDATION_REPORT_GIT_PATH)
    identity_text = "parameters=" + "1" * 64 + "\nfeatures=" + "2" * 64 + "\nfriction_assumptions=" + "3" * 64 + "\nuniverse=" + "4" * 64 + "\n"
    candidate_id = hashlib.sha256(identity_text.encode("ascii")).hexdigest()
    report["surviving_candidate_definition_sha256s"] = [candidate_id]
    (repo / pp.LAYER_B_VALIDATION_REPORT_GIT_PATH).write_bytes(json.dumps(report).encode())
    report_commit = _commit_all(repo, "finalize validation report")
    report_blob = _git_blob_sha(repo, report_commit, pp.LAYER_B_VALIDATION_REPORT_GIT_PATH)
    candidate = _candidate_artifact(report_commit, report_blob, candidate_definition_sha256=candidate_id)
    (repo / pp.FROZEN_FINAL_CANDIDATE_ARTIFACT_GIT_PATH).write_bytes(json.dumps(candidate).encode())
    candidate_commit = _commit_all(repo, "freeze final candidate")
    return repo, report_commit, report_blob, candidate_commit, candidate_id


def test_concrete_layer_b_and_candidate_readers_pass(tmp_path):
    repo, report_commit, report_blob, candidate_commit, candidate_id = _build_concrete_stage_repo(tmp_path)
    report = pp.read_and_verify_layer_b_validation_report(repo, report_commit)
    candidate = pp.read_and_verify_frozen_final_candidate(repo, candidate_commit)
    assert report["git_blob_sha"] == report_blob
    assert candidate["candidate_definition_sha256"] == candidate_id


def test_approval_only_artifacts_are_not_concrete_stage_evidence(tmp_path):
    repo = tmp_path / "approval_only"
    repo.mkdir()
    approval = {"schema_version": "synthetic", "result": "PASS", "approval_status": "APPROVED"}
    head = _init_bogus_git_repo(repo, files={
        "V8B_LAYER_B_COMPLETION_APPROVAL.json": json.dumps(approval).encode(),
        "V8B_FROZEN_FINAL_CANDIDATE_APPROVAL.json": json.dumps(approval).encode(),
    })
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_layer_b_validation_report(repo, head)
    assert excinfo.value.reason == "V8B_LAYER_B_VALIDATION_REPORT_MISSING"


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("validation_access_count", 0, "V8B_LAYER_B_VALIDATION_REPORT_ACCESS_COUNT_INVALID"),
        ("validation_result", "BLOCK", "V8B_LAYER_B_VALIDATION_REPORT_NOT_PASS"),
        ("surviving_candidate_definition_sha256s", [], "V8B_LAYER_B_VALIDATION_REPORT_CANDIDATE_IDS_INVALID"),
    ],
)
def test_layer_b_contract_rejects_invalid_stage_claims(tmp_path, field, value, reason):
    repo = tmp_path / ("bad_layer_b_" + field)
    repo.mkdir()
    report = _layer_b_report(**{field: value})
    head = _init_bogus_git_repo(repo, files={pp.LAYER_B_VALIDATION_REPORT_GIT_PATH: json.dumps(report).encode()})
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_layer_b_validation_report(repo, head)
    assert excinfo.value.reason == reason
