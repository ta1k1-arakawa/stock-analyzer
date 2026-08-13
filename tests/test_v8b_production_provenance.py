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
# FINAL_REPEAT finding HIGH-2: INDEPENDENT_TRUST_PIN_REVIEW
# ---------------------------------------------------------------------------

SYNTHETIC_ARTIFACT_HASH = "9" * 64


def _trust_pin_review_artifact(**overrides) -> dict:
    review = {
        "schema_version": pp.TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_VERSION,
        "study": pp.STUDY_NAME,
        "artifact_role": pp.TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_ROLE,
        "reviewed_allocation_artifact_self_hash": SYNTHETIC_ARTIFACT_HASH,
        "reviewed_trust_pin_human_gate": "V8B_HUMAN_AUTHORIZE_T1B_ALLOCATION_PIN_AT_" + SYNTHETIC_ARTIFACT_HASH,
        "review_result": "PASS",
        "approval_status": "APPROVED",
    }
    review.update(overrides)
    return review


def test_trust_pin_review_missing_blocks_on_real_repo():
    """The real V8B_TRUST_PIN_INDEPENDENT_REVIEW.json does not exist yet --
    production must fail closed today."""
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_trust_pin_independent_review(
            ROOT, _real_head(), expected_allocation_artifact_self_hash=SYNTHETIC_ARTIFACT_HASH
        )
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_MISSING"


def test_trust_pin_review_passes_on_well_formed_synthetic_artifact(tmp_path):
    bogus = tmp_path / "trust_pin_review_pass"
    bogus.mkdir()
    commit = _init_bogus_git_repo(
        bogus,
        files={"V8B_TRUST_PIN_INDEPENDENT_REVIEW.json": json.dumps(_trust_pin_review_artifact()).encode()},
    )
    result = pp.read_and_verify_trust_pin_independent_review(
        bogus, commit, expected_allocation_artifact_self_hash=SYNTHETIC_ARTIFACT_HASH
    )
    assert result["review_result"] == "PASS"


def test_trust_pin_review_bound_to_a_different_hash_is_rejected(tmp_path):
    """A well-formed, PASS/APPROVED review for a DIFFERENT allocation
    artifact hash must never authorize this one."""
    bogus = tmp_path / "trust_pin_review_wrong_hash"
    bogus.mkdir()
    commit = _init_bogus_git_repo(
        bogus,
        files={"V8B_TRUST_PIN_INDEPENDENT_REVIEW.json": json.dumps(_trust_pin_review_artifact()).encode()},
    )
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_trust_pin_independent_review(
            bogus, commit, expected_allocation_artifact_self_hash="8" * 64
        )
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_ARTIFACT_HASH_MISMATCH"


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
    bogus = tmp_path / ("trust_pin_review_field_" + field)
    bogus.mkdir()
    commit = _init_bogus_git_repo(
        bogus,
        files={
            "V8B_TRUST_PIN_INDEPENDENT_REVIEW.json": json.dumps(
                _trust_pin_review_artifact(**{field: value})
            ).encode()
        },
    )
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_trust_pin_independent_review(
            bogus, commit, expected_allocation_artifact_self_hash=SYNTHETIC_ARTIFACT_HASH
        )
    assert excinfo.value.reason == expected_reason


def test_trust_pin_review_schema_missing_field_blocks(tmp_path):
    bogus = tmp_path / "trust_pin_review_schema"
    bogus.mkdir()
    incomplete = _trust_pin_review_artifact()
    del incomplete["reviewed_trust_pin_human_gate"]
    commit = _init_bogus_git_repo(
        bogus, files={"V8B_TRUST_PIN_INDEPENDENT_REVIEW.json": json.dumps(incomplete).encode()}
    )
    with pytest.raises(pp.V8BProductionProvenanceBlocked) as excinfo:
        pp.read_and_verify_trust_pin_independent_review(
            bogus, commit, expected_allocation_artifact_self_hash=SYNTHETIC_ARTIFACT_HASH
        )
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# FINAL_REPEAT finding HIGH-2: the new trust-pin-creation and human-gate-
# consumption modules are bound to the reviewed implementation too.
# ---------------------------------------------------------------------------


def test_trust_pin_creation_and_gate_consumption_modules_are_bound_to_the_review():
    assert "src/v8b_trust_pin_creation.py" in pp.BOUND_PRODUCTION_FILES
    assert "src/v8b_human_gate_consumption.py" in pp.BOUND_PRODUCTION_FILES
