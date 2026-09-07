from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v9_015_source_b_option_value_root_extractor_synthetic.py"


def test_synthetic_cli_imports_from_external_working_directory(tmp_path):
    environment = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
    completed = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--synthetic"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == "PASS"
    assert payload["candidate_years"] == ["2017", "2019", "2020", "2022", "2026"]
    assert payload["all_downstream_ok"] is True
    assert payload["network_requests"] == 0
