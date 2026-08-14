from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src import v8c_production_provenance as provenance

REPO_ROOT = Path(__file__).resolve().parents[1]


def _real_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


def test_exact_frozen_design_commit_and_blob_constants_match_task():
    assert provenance.EXPECTED_V8C_FROZEN_DESIGN_COMMIT == "c9c541ac7f7ba3bcca76db6250fe8273d9bb5756"
    assert provenance.EXPECTED_V8C_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT == "9c2cd4081a6f5ec2e48daab4a30d5ca78aea64d0"
    assert provenance.EXPECTED_V8C_DESIGN_FREEZE_APPROVAL_BLOB == "a43eed2274bdb433ac7314515b3b9c3492afbc57"
    assert provenance.EXPECTED_HUMAN_FREEZE_GATE == (
        "V8C_HUMAN_DESIGN_FREEZE_APPROVED_FOR_COMMIT_c9c541ac7f7ba3bcca76db6250fe8273d9bb5756"
    )


def test_exact_v8_authority_constants_match_task():
    assert provenance.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA == "61faade0625139cec3fb61216ab2f97f572a7028"
    assert provenance.EXPECTED_V8_PARTITION_MANIFEST_SHA256 == "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
    assert provenance.EXPECTED_T2_TICKER_LIST_SHA256 == "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500"
    assert provenance.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256 == "360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70"


def test_classifier_blob_constant_matches_frozen_v7_collector():
    assert provenance.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA == "76b57b077f3214e666ff9dc06d9c224afc16df9f"


def test_verify_frozen_design_object_passes_against_real_repository():
    # The frozen design commit is real, immutable history in this repo.
    provenance.verify_frozen_design_object(REPO_ROOT)


def test_verify_frozen_design_object_blocks_on_wrong_expected_blob(monkeypatch):
    monkeypatch.setattr(provenance, "EXPECTED_V8C_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT", "0" * 40)
    with pytest.raises(provenance.V8CProductionProvenanceBlocked) as excinfo:
        provenance.verify_frozen_design_object(REPO_ROOT)
    assert excinfo.value.reason == "V8C_FROZEN_DESIGN_OBJECT_MUTATED"


def test_read_and_verify_design_freeze_approval_passes_against_real_head():
    result = provenance.read_and_verify_design_freeze_approval(REPO_ROOT, _real_head())
    assert result["approval_status"] == "APPROVED"
    assert result["frozen_design_git_commit"] == provenance.EXPECTED_V8C_FROZEN_DESIGN_COMMIT


def test_read_and_verify_design_freeze_approval_blocks_on_wrong_blob_expectation(monkeypatch):
    monkeypatch.setattr(provenance, "EXPECTED_V8C_DESIGN_FREEZE_APPROVAL_BLOB", "0" * 40)
    with pytest.raises(provenance.V8CProductionProvenanceBlocked) as excinfo:
        provenance.read_and_verify_design_freeze_approval(REPO_ROOT, _real_head())
    assert excinfo.value.reason == "V8C_DESIGN_FREEZE_APPROVAL_BLOB_MUTATED"


def test_read_and_verify_design_freeze_approval_missing_file_fails_closed(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "a@b.c"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "t"], check=True)
    (tmp_path / "x.txt").write_text("x")
    subprocess.run(["git", "-C", str(tmp_path), "add", "x.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "init"], check=True)
    head = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()
    with pytest.raises(provenance.V8CProductionProvenanceBlocked) as excinfo:
        provenance.read_and_verify_design_freeze_approval(tmp_path, head)
    assert excinfo.value.reason == "V8C_DESIGN_FREEZE_APPROVAL_MISSING"


def test_verify_reviewed_implementation_binding_missing_fails_closed():
    """The real INDEPENDENT_V8C_PRODUCTION_IMPLEMENTATION_REVIEW.json does
    not exist in this repository -- this must fail closed today."""
    with pytest.raises(provenance.V8CProductionProvenanceBlocked) as excinfo:
        provenance.verify_reviewed_implementation_binding(REPO_ROOT, _real_head())
    assert excinfo.value.reason == "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING"


def test_verify_classifier_blob_accepts_exact_pin():
    provenance.verify_classifier_blob(provenance.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA)


def test_verify_classifier_blob_mismatch_blocks():
    with pytest.raises(provenance.V8CProductionProvenanceBlocked) as excinfo:
        provenance.verify_classifier_blob("0" * 40)
    assert excinfo.value.reason == "V8C_PRODUCTION_CLASSIFIER_VERSION_MISMATCH"
    assert excinfo.value.reason == provenance.CLASSIFIER_VERSION_MISMATCH_ERROR


def test_read_and_verify_v8_trusted_partition_anchor_passes_against_real_head():
    result = provenance.read_and_verify_v8_trusted_partition_anchor(REPO_ROOT, _real_head())
    assert result["authorization_status"] == "AUTHORIZED"
    assert result["authorized_partition_manifest_sha256"] == provenance.EXPECTED_V8_PARTITION_MANIFEST_SHA256


def test_trust_pin_independent_review_missing_fails_closed():
    with pytest.raises(provenance.V8CProductionProvenanceBlocked) as excinfo:
        provenance.read_and_verify_trust_pin_independent_review(
            REPO_ROOT, _real_head(),
            expected_allocation_artifact_self_hash="a" * 64,
            expected_trust_pin_human_gate="V8C_HUMAN_AUTHORIZE_T1C_ALLOCATION_PIN_AT_" + "a" * 64,
        )
    assert excinfo.value.reason == "V8C_TRUST_PIN_INDEPENDENT_REVIEW_MISSING"


def test_bound_production_files_all_exist_at_head():
    """Every file this module binds review to must actually exist -- a
    typo'd path would silently make ``verify_reviewed_implementation_
    binding`` vacuously trivial for that file."""
    for path in provenance.BOUND_PRODUCTION_FILES:
        assert (REPO_ROOT / path).is_file(), path
