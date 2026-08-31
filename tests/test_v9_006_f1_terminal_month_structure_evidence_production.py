from __future__ import annotations

from pathlib import Path

import pytest

from src import v9_006_f1_semantic_successor_public_acquisition_production as production


def test_constants_and_stable_state_root():
    root = Path(r"C:\work\stock-analyzer")
    assert production.derive_state_root(root) == root.parent / "v9-006-f1-successor-public-acquisition-state"
    assert production.DESIGN_BLOB == "6112b92f39f34c594d36a28d72072dcb255b9eee"


def test_check_bindings_requires_local_remote_clean_and_design_blob():
    expected = "a" * 40
    values = {
        ("branch", "--show-current"): production.AUTHORITATIVE_BRANCH,
        ("rev-parse", "HEAD"): expected,
        ("status", "--porcelain"): "",
        ("rev-parse", f"HEAD:{production.DESIGN_PATH}"): production.DESIGN_BLOB,
        ("rev-parse", f"origin/{production.AUTHORITATIVE_BRANCH}"): expected,
    }
    def fake_git(_root, *args):
        if args == ("fetch", "--no-tags", "origin", production.AUTHORITATIVE_BRANCH):
            return ""
        return values[args]
    production.check_bindings(expected, Path("."), git_output=fake_git)
    for key, replacement in [(("branch", "--show-current"), "other"), (("status", "--porcelain"), "dirty"), (("rev-parse", f"HEAD:{production.DESIGN_PATH}"), "bad")]:
        altered = dict(values); altered[key] = replacement
        def altered_git(_root, *args, altered=altered):
            if args == ("fetch", "--no-tags", "origin", production.AUTHORITATIVE_BRANCH):
                return ""
            return altered.get(args, values[args])
        with pytest.raises(ValueError):
            production.check_bindings(expected, Path("."), git_output=altered_git)


def test_parser_accepts_only_diagnostic_implementation_sha():
    parser = production.build_parser()
    assert parser.parse_args(["--diagnostic-implementation-git-sha", "a" * 40]).diagnostic_implementation_git_sha == "a" * 40
    with pytest.raises(SystemExit):
        parser.parse_args(["--state-root", "private"])
