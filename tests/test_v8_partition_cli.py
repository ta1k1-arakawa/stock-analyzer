from __future__ import annotations

import ast
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from scripts import build_v8_partition_manifest as cli
from src import v8_partition as partition

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_v8_partition_manifest.py"
PYTHON = sys.executable


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


@pytest.fixture(scope="module")
def synthetic_result():
    return cli.run_synthetic_partition_test()


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_has_exactly_one_authorized_option():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    options = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    }
    assert options == {"--synthetic-test"}


def test_cli_has_no_bypass_or_real_path_option():
    text = SCRIPT.read_text(encoding="utf-8")
    for flag in (
        "--skip-source-hash", "--force", "--ignore-parity", "--real",
        "--output-root", "--source-path", "--network",
    ):
        assert flag not in text


def test_cli_subprocess_requires_the_flag():
    result = subprocess.run([PYTHON, str(SCRIPT)], cwd=str(ROOT), capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert result.stdout == ""


def test_cli_subprocess_rejects_unknown_option():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--force"], cwd=str(ROOT), capture_output=True, text=True, timeout=60
    )
    assert result.returncode != 0


def test_cli_subprocess_exit_zero_and_reports_result():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--synthetic-test"], cwd=str(ROOT), capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["network_requests"] == 0
    assert payload["real_partition_created"] is False


def test_cli_leaves_no_manifest_in_repository():
    assert not list(ROOT.glob("**/partition_manifest.json"))
    assert not (ROOT / "V8_UNIVERSE_MANIFEST.json").exists()


# ---------------------------------------------------------------------------
# Synthetic result shape (in-process)
# ---------------------------------------------------------------------------


REQUIRED_RESULT_FIELDS = {
    "status", "mode", "source_reproduction_status", "block_sizes",
    "t1_role", "t2_role", "t3_role", "t3_price_acquisition_authorized",
    "manifest_sha256_verified", "write_once_enforced",
    "source_mismatch_blocks_before_allocation", "network_requests",
    "real_partition_created", "real_source_fetch_performed",
}


def test_synthetic_result_has_required_fields(synthetic_result):
    assert REQUIRED_RESULT_FIELDS <= set(synthetic_result)


def test_synthetic_result_values(synthetic_result):
    assert synthetic_result["status"] == "PASS"
    assert synthetic_result["mode"] == "STATIC_SYNTHETIC_ONLY"
    assert synthetic_result["source_reproduction_status"] == "PASS"
    assert synthetic_result["t1_role"] == "VALIDATION"
    assert synthetic_result["t2_role"] == "SEALED_HOLDOUT"
    assert synthetic_result["t3_role"] == "SEALED_RESERVE"
    assert synthetic_result["t3_price_acquisition_authorized"] is False
    assert synthetic_result["manifest_sha256_verified"] is True
    assert synthetic_result["write_once_enforced"] is True
    assert synthetic_result["source_mismatch_blocks_before_allocation"] is True
    assert synthetic_result["network_requests"] == 0
    assert synthetic_result["real_partition_created"] is False
    assert synthetic_result["real_source_fetch_performed"] is False


def test_synthetic_result_block_sizes_equal_for_t0_t1_t2_t3(synthetic_result):
    sizes = synthetic_result["block_sizes"]
    assert sizes["T0"] == sizes["T1"] == sizes["T2"] == sizes["T3"] == cli.SYNTHETIC_BLOCK_SIZE


# ---------------------------------------------------------------------------
# Static safety: src/v8_partition.py and this CLI
# ---------------------------------------------------------------------------


def test_module_source_has_no_network_imports():
    text = Path(partition.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip().lower() for line in text.splitlines() if line.strip().startswith(("import ", "from "))]
    for token in ("re" + "quests", "url" + "lib", "http" + "x", "aio" + "http", "so" + "cket", "y" + "finance"):
        assert not any(token in line for line in import_lines), token


def test_cli_source_has_no_urlopen_call():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen"
    ]
    assert calls == []


def test_module_never_touches_v7_files():
    text = Path(partition.__file__).read_text(encoding="utf-8")
    assert "v7_" not in text.lower()
    cli_text = SCRIPT.read_text(encoding="utf-8")
    assert "v7_" not in cli_text.lower()


def test_frozen_design_commit_matches_current_design():
    assert partition.DESIGN_COMMIT == "c414d3191cba356734d7ed08bdf1abc7d51fc384"


def test_frozen_block_size_and_p_hist_unchanged():
    assert partition.BLOCK_SIZE == 300
    assert partition.P_HIST_START == "2016-04-01"
    assert partition.P_HIST_END == "2025-12-31"
