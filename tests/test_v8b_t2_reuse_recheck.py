from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src import v8b_t2_reuse_recheck as recheck

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
    result = recheck.recheck_t2_reuse_conditions(_pass_metadata())
    assert result == {"result": "PASS", "block": "T2"}


def test_pure_evaluator_blocks_on_missing_field():
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.recheck_t2_reuse_conditions({"t2_acquired": False})
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_SAFE_METADATA"


def test_pure_evaluator_blocks_on_already_acquired():
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.recheck_t2_reuse_conditions(_pass_metadata(t2_acquired=True))
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_BLOCKED:T2_ACQUIRED"


def test_pure_evaluator_blocks_on_universe_definition_changed():
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.recheck_t2_reuse_conditions(_pass_metadata(t2_universe_definition_unchanged=False))
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_BLOCKED:T2_UNIVERSE_DEFINITION_UNCHANGED"


def test_pure_evaluator_module_defines_no_fallback_substitution():
    assert not any("spare" in name.lower() or "t3" in name.lower() for name in recheck.__all__)


# ---------------------------------------------------------------------------
# MEDIUM-2: production resolver -- must derive from a verified Git object,
# never an arbitrary caller-supplied mapping.
# ---------------------------------------------------------------------------


def test_production_resolver_signature_accepts_no_arbitrary_mapping():
    import inspect

    params = set(inspect.signature(recheck.resolve_and_recheck_t2_reuse_conditions).parameters)
    assert "safe_metadata" not in params
    assert params == {"repository_root", "verified_head"}


def test_production_resolver_blocks_on_real_repo_today():
    """The real V8B_T2_REUSE_CONDITIONS_RECHECK.json does not exist yet --
    the real post-Layer-B recheck has not been performed -- so production
    must fail closed today (round-2 finding HIGH-2 correction)."""
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.resolve_and_recheck_t2_reuse_conditions(ROOT, _real_head())
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_MISSING"


def test_module_no_longer_reads_the_pre_freeze_section_12_2_document():
    """Confirms the fix: this module's own field set does not name the old
    §12.2 markdown path, and no function accepts it."""
    assert "V8B_TSPARE_T2_T3_PRESERVATION_RECHECK.md" not in recheck.POST_FREEZE_RECHECK_GIT_PATH
    assert recheck.POST_FREEZE_RECHECK_GIT_PATH == "V8B_T2_REUSE_CONDITIONS_RECHECK.json"
    assert recheck.POST_FREEZE_RECHECK_STAGE == "POST_FREEZE"


def _post_freeze_artifact(**overrides) -> dict:
    artifact = {
        "schema_version": recheck.POST_FREEZE_RECHECK_SCHEMA_VERSION,
        "study": "V8B_HISTORICAL_RESEARCH",
        "gate": recheck.POST_FREEZE_RECHECK_GATE,
        "frozen_design_git_commit": "eedf198b93185b963b825170ed0be97e93f923b7",
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


def _write_json(path: Path, value: dict) -> bytes:
    import json

    raw = json.dumps(value).encode("utf-8")
    path.write_bytes(raw)
    return raw


def test_production_resolver_passes_on_well_formed_synthetic_post_freeze_artifact(tmp_path):
    bogus = tmp_path / "well_formed"
    bogus.mkdir()
    raw = _write_json(bogus / recheck.POST_FREEZE_RECHECK_GIT_PATH, _post_freeze_artifact())
    commit = _init_bogus_git_repo(bogus, files={recheck.POST_FREEZE_RECHECK_GIT_PATH: raw})
    result = recheck.resolve_and_recheck_t2_reuse_conditions(bogus, commit)
    assert result == {"result": "PASS", "block": "T2"}


def test_resolve_safe_metadata_matches_pure_evaluator_schema(tmp_path):
    bogus = tmp_path / "schema_check"
    bogus.mkdir()
    raw = _write_json(bogus / recheck.POST_FREEZE_RECHECK_GIT_PATH, _post_freeze_artifact())
    commit = _init_bogus_git_repo(bogus, files={recheck.POST_FREEZE_RECHECK_GIT_PATH: raw})
    safe_metadata = recheck.resolve_t2_reuse_safe_metadata_from_verified_head(bogus, commit)
    assert set(safe_metadata) == set(recheck.REQUIRED_SAFE_METADATA_FIELDS)
    for value in safe_metadata.values():
        assert isinstance(value, bool)


def test_production_resolver_missing_artifact_blocks(tmp_path):
    bogus = tmp_path / "no_doc"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"x"})
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.resolve_and_recheck_t2_reuse_conditions(bogus, commit)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_MISSING"


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
def test_production_resolver_field_semantics_enforced(tmp_path, field, value, expected_reason):
    bogus = tmp_path / ("field_" + field)
    bogus.mkdir()
    raw = _write_json(bogus / recheck.POST_FREEZE_RECHECK_GIT_PATH, _post_freeze_artifact(**{field: value}))
    commit = _init_bogus_git_repo(bogus, files={recheck.POST_FREEZE_RECHECK_GIT_PATH: raw})
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.resolve_and_recheck_t2_reuse_conditions(bogus, commit)
    assert excinfo.value.reason == expected_reason


def test_stale_pre_freeze_evidence_cannot_satisfy_the_post_freeze_gate(tmp_path):
    """A forged artifact that claims the OLD §12.2 pre-freeze evidence is
    good enough (stage=PRE_FREEZE) must not satisfy the §12.4 gate --
    this is exactly the bug round 2 found and this test locks the fix in."""
    bogus = tmp_path / "stale_evidence"
    bogus.mkdir()
    raw = _write_json(
        bogus / recheck.POST_FREEZE_RECHECK_GIT_PATH, _post_freeze_artifact(stage="PRE_FREEZE")
    )
    commit = _init_bogus_git_repo(bogus, files={recheck.POST_FREEZE_RECHECK_GIT_PATH: raw})
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.resolve_and_recheck_t2_reuse_conditions(bogus, commit)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_NOT_POST_FREEZE"


def test_duplicate_key_artifact_blocks(tmp_path):
    bogus = tmp_path / "dup_key"
    bogus.mkdir()
    raw = b'{"schema_version": "a", "schema_version": "b"}'
    commit = _init_bogus_git_repo(bogus, files={recheck.POST_FREEZE_RECHECK_GIT_PATH: raw})
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.resolve_and_recheck_t2_reuse_conditions(bogus, commit)
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_DUPLICATE_KEY"
