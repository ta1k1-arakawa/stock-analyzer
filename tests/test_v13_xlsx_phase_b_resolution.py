"""Offline authorization and one-shot resolution boundary checks."""

import json
from pathlib import Path

import pytest

from scripts import check_current_protected_environment as current
from scripts import v13_xlsx_phase_b_resolution as phase_b
from scripts import v13_xlsx_successor_contract as successor


def test_approval_exact_and_frozen_bindings() -> None:
    record = json.loads((phase_b.ROOT / phase_b.AUTH).read_text(encoding="utf-8"))
    phase_b._approval(record)
    assert phase_b._bound_file(phase_b.PHASE_A, record["current_authority_lock"])
    assert phase_b._bound_file(phase_b.PHASE_A, record["successor_direct_spec"])
    assert record["one_shot"] and not record["consumed"]
    assert not record["candidate_installation_allowed"]
    assert record["official_index_url"] == "https://pypi.org/simple"
    changed = dict(record, consumed=True)
    with pytest.raises(ValueError, match="AUTH_SCOPE_INVALID"):
        phase_b._approval(changed)


def test_current_authority_and_unresolved_successor_spec() -> None:
    authority = current.resolve_current_authority()
    assert authority["status"] == "PASS"
    assert authority["package_count"] == 27
    base = successor.validate_direct_spec()
    assert len(base) == 27
    assert "openpyxl" not in base
    assert "et-xmlfile" not in base


def test_rehearsal_has_no_side_effects(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["phase_b"])
    monkeypatch.setattr(phase_b, "preflight", lambda *a, **k: pytest.fail("preflight called"))
    monkeypatch.setattr(phase_b.subprocess, "run", lambda *a, **k: pytest.fail("subprocess called"))
    assert phase_b.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["package_index_requests"] == result["wheel_downloads"] == 0
    assert result["environment_mutations"] == 0


def test_existing_operation_root_blocks_second_resolution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(phase_b.sys, "platform", "win32")
    monkeypatch.setattr(phase_b, "preflight", lambda *a, **k: {"authorization_blob": "a" * 40})
    monkeypatch.setattr(phase_b.subprocess, "run", lambda *a, **k: pytest.fail("resolver called"))
    root = tmp_path / "consumed"
    root.mkdir()
    (root / "receipt.json").write_text('{"status":"PASS_CONSUMED"}', encoding="utf-8")
    with pytest.raises(ValueError, match="ONE_SHOT_ALREADY_STARTED"):
        phase_b.execute("b" * 40, root)


def test_resolver_command_is_download_only() -> None:
    source = (phase_b.ROOT / "scripts/v13_xlsx_phase_b_resolution.py").read_text(encoding="utf-8")
    assert '"-m", "pip", "download"' in source
    assert '"--index-url", INDEX' in source
    assert '"--constraint", str(ROOT / current.CURRENT_AUTHORITY_LOCK_PATH)' in source
    assert '"pip", "install"' not in source


def test_dirty_tree_blocks_before_remote_or_review(monkeypatch) -> None:
    observed = []

    def fake_git(*args):
        observed.append(args)
        if args == ("branch", "--show-current"):
            return (phase_b.BRANCH + "\n").encode()
        if args == ("rev-parse", "HEAD"):
            return ("a" * 40 + "\n").encode()
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return b" M PROJECT_STATE.md\n"
        pytest.fail("later Git operation reached")

    monkeypatch.setattr(phase_b, "_git", fake_git)
    monkeypatch.setattr(phase_b, "_review_pass", lambda head: pytest.fail("review lookup reached"))
    with pytest.raises(ValueError, match="DIRTY_WORKTREE"):
        phase_b.preflight("a" * 40)
    assert observed == [("branch", "--show-current"), ("rev-parse", "HEAD"),
                        ("status", "--porcelain", "--untracked-files=all")]


def test_review_pass_requires_exact_sha_and_issue(monkeypatch) -> None:
    class Result:
        stdout = json.dumps({"comments": [{"author": {"login": "ta1k1-arakawa"},
            "body": "MODEL=GPT-5.6_SOL\nMODE=GPT_EXACT_SHA_INDEPENDENT_REVIEW\n"
                    "GITHUB_ISSUE=100\nREVIEWED_SHA=" + "a" * 40 + "\nRESULT=PASS"}]}).encode()

    monkeypatch.setattr(phase_b.subprocess, "run", lambda *a, **k: Result())
    assert phase_b._review_pass("a" * 40)
    assert not phase_b._review_pass("b" * 40)
