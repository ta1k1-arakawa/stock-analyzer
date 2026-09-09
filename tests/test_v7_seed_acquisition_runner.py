from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.acquire_v7_forward_seed import build_parser
from src import v7_seed_acquisition as acquisition


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"C:\taiki\hobbies\v4-meta-label-env\Scripts\python.exe")
SCRIPT = ROOT / "scripts" / "acquire_v7_forward_seed.py"


def test_cli_has_exactly_the_six_authorized_options():
    actions = {action.dest for action in build_parser()._actions if action.dest != "help"}
    assert actions == {"output_dir", "universe_csv", "request_start", "request_end_exclusive", "seed_cutoff", "confirmation"}


def test_cli_rejects_activation_evaluate_and_order_options():
    parser = build_parser()
    for option in ("--activate", "--evaluate", "--order", "--portfolio", "--candidate"):
        with pytest.raises(SystemExit):
            parser.parse_args([option])


def test_cli_wrong_confirmation_fails_without_network_or_output(tmp_path):
    result = subprocess.run(
        [str(PYTHON), str(SCRIPT), "--output-dir", str(tmp_path / "out"), "--universe-csv", str(ROOT / "V4_UNIVERSE.csv"),
         "--request-start", "2025-07-01", "--request-end-exclusive", "2026-08-08", "--seed-cutoff", "2026-08-07", "--confirmation", "WRONG"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert json.loads(result.stdout)["reason"] == "CONFIRMATION_MISMATCH"
    assert not (tmp_path / "out").exists()


def test_constants_bind_fixed_lineage_and_no_activation():
    assert acquisition.DESIGN_COMMIT == "e3e1367efd913b601a70328a815d88c20af6d147"
    assert acquisition.LATEST_PREREGISTRATION_UTC == "2026-08-07T02:48:27Z"
    assert acquisition.COLLECTOR_COMMIT == "4ca41c53895e75910ae65809fea6018868929afa"
    assert acquisition.MODE == "PRE_ACTIVATION_SEED_ACQUISITION"
    assert acquisition.ACTIVATION_STATUS == "NOT_ACTIVATED"


def test_cli_module_has_no_network_execution_at_import():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen"]
    assert len(calls) == 1
    assert any(isinstance(node, ast.FunctionDef) and node.name == "_production_opener" for node in tree.body)
