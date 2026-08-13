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


def test_production_resolver_passes_on_real_repo():
    result = recheck.resolve_and_recheck_t2_reuse_conditions(ROOT, _real_head())
    assert result == {"result": "PASS", "block": "T2"}


def test_resolve_safe_metadata_matches_pure_evaluator_schema():
    safe_metadata = recheck.resolve_t2_reuse_safe_metadata_from_verified_head(ROOT, _real_head())
    assert set(safe_metadata) == set(recheck.REQUIRED_SAFE_METADATA_FIELDS)
    for value in safe_metadata.values():
        assert isinstance(value, bool)


def test_production_resolver_doc_blob_mutation_blocks(tmp_path):
    """A doc claiming the same favorable PASS values, but with different
    surrounding bytes (mutated), must BLOCK at the exact-blob check --
    never trusted merely because it parses to the same fields."""
    real_bytes = (ROOT / recheck.PRESERVATION_RECHECK_GIT_PATH).read_bytes()
    mutated = real_bytes + b"\n<!-- attacker-added comment -->\n"
    bogus = tmp_path / "mutated_doc"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={recheck.PRESERVATION_RECHECK_GIT_PATH: mutated})
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.resolve_and_recheck_t2_reuse_conditions(bogus, commit)
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_DOC_MUTATED"


def test_production_resolver_missing_doc_blocks(tmp_path):
    bogus = tmp_path / "no_doc"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={"README.md": b"x"})
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.resolve_and_recheck_t2_reuse_conditions(bogus, commit)
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_DOC_MISSING"


def test_production_resolver_forged_favorable_doc_still_blocks_on_blob_check(tmp_path):
    """An attacker-forged doc that claims every condition PASSes, in the
    exact expected format, still cannot pass -- it does not match the real
    frozen blob, so the exact-blob check catches it before any field is
    ever parsed."""
    forged = (
        "```text\n"
        "result=PASS\n"
        "reviewed_design_commit=eedf198b93185b963b825170ed0be97e93f923b7\n"
        "```\n"
        "## B. `T2` recheck\n"
        "```text\n"
        "t2_acquired=false -- PASS\n"
        "t2_opened=false -- PASS\n"
        "t2_ticker_identities_exposed_to_human_public_research_loop=false -- PASS\n"
        "t2_market_data_raw_ohlcv_feature_outcome_research_exposure=false -- PASS\n"
        "t2_universe_definition_unchanged=true -- PASS\n"
        "t2_partition_algorithm_unchanged=true -- PASS\n"
        "v8b_f1_c1_production_policy_already_fixed_at_reviewed_design_sha=true -- PASS\n"
        "```\n"
        "## C. `T3` recheck\n"
    ).encode()
    bogus = tmp_path / "forged_favorable"
    bogus.mkdir()
    commit = _init_bogus_git_repo(bogus, files={recheck.PRESERVATION_RECHECK_GIT_PATH: forged})
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.resolve_and_recheck_t2_reuse_conditions(bogus, commit)
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_DOC_MUTATED"
