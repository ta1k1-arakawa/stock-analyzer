from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v9_009_t0_top1_kill_screen.py"


def _invoke(tmp_path: Path, *extra: str):
    calendar = tmp_path / "calendar.json"
    calendar.write_text(json.dumps({"trading_dates": [day.strftime("%Y-%m-%d") for day in pd.bdate_range("2017-12-20", "2020-01-20")]}) + "\n", encoding="utf-8")
    command = [
        sys.executable,
        str(SCRIPT),
        "--training-cache",
        str(tmp_path / "missing-training-cache"),
        "--evaluation-cache",
        str(tmp_path / "missing-evaluation-cache"),
        "--universe-csv",
        str(tmp_path / "missing-universe.csv"),
        "--calendar-file",
        str(calendar),
        "--implementation-sha",
        "a" * 40,
        *extra,
    ]
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)


def test_cli_incompatible_cache_is_no_verdict_and_not_stop():
    temp_dir = Path(tempfile.mkdtemp(prefix="v9-009-cli-"))
    try:
        completed = _invoke(temp_dir)
    finally:
        shutil.rmtree(temp_dir)
    assert completed.returncode == 0
    assert completed.stderr == ""
    result = json.loads(completed.stdout)
    assert result["T0_RESULT"] == "NO_VERDICT_DATA_INCOMPATIBLE"
    assert result["validation"]["cache_identity"] is False
    assert result["validation"]["exact_v9_features"] is False
    assert result["T0_RESULT"] != "STOP"


def test_cli_safe_output_has_no_input_paths_or_outcome_fields():
    temp_dir = Path(tempfile.mkdtemp(prefix="v9-009-cli-"))
    try:
        completed = _invoke(temp_dir)
    finally:
        shutil.rmtree(temp_dir)
    assert completed.returncode == 0
    text = completed.stdout
    assert str(temp_dir) not in text
    assert "aggregate" not in text
    assert "yearly" not in text
    assert "percentile" not in text
    assert "ridge_score" not in text
    assert "0001" not in text
    assert "\\" not in text


def test_cli_unexpected_runtime_is_fixed_marker_only():
    temp_dir = Path(tempfile.mkdtemp(prefix="v9-009-cli-"))
    try:
        completed = _invoke(temp_dir, "--calendar-file", str(temp_dir / "does-not-exist.json"))
    finally:
        shutil.rmtree(temp_dir)
    assert completed.returncode == 3
    assert completed.stdout == ""
    assert completed.stderr.strip() == "V9_009_T0_TOP1_KILL_SCREEN_IMPLEMENTATION_FAILURE"
    assert str(temp_dir) not in completed.stderr
