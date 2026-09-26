"""Synthetic/offline checks; no protected state or transport."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from scripts import v13_xlsx_readiness as readiness
from scripts import v13_xlsx_successor_contract as successor


def test_current_authority_and_direct_spec_are_exact() -> None:
    base = successor.validate_direct_spec()
    assert len(base) == 27
    assert "openpyxl" not in base and "et-xmlfile" not in base
    spec_lines = successor.SPEC.read_text(encoding="utf-8").splitlines()
    assert "openpyxl" in spec_lines and "et-xmlfile" not in spec_lines


def test_synthetic_fixture_has_xlsx_signature_and_no_external_input() -> None:
    raw = readiness.synthetic_jpx_xlsx()
    assert raw.startswith(b"PK\x03\x04")
    with zipfile.ZipFile(__import__("io").BytesIO(raw)) as archive:
        assert "xl/worksheets/sheet1.xml" in archive.namelist()


def test_missing_reader_fails_before_real_execution_inputs() -> None:
    wrapper = (Path(__file__).resolve().parents[1] / "scripts/run_v13_public_data_lock_direct_windows.ps1").read_text()
    probe = wrapper.index("-m scripts.v13_xlsx_readiness")
    assert probe < wrapper.index("--preflight-calendar") < wrapper.index("--t1-state")
    if importlib.util.find_spec("openpyxl") is None:
        result = subprocess.run([sys.executable, "-m", "scripts.v13_xlsx_readiness"],
                                capture_output=True, text=True, check=False)
        assert result.returncode != 0
        assert result.stdout.strip() == "JPX_XLSX_READINESS_PASS=false"


@pytest.mark.skipif(importlib.util.find_spec("openpyxl") is None,
                    reason="candidate XLSX reader is not installed in the development interpreter")
def test_actual_production_parse_jpx_xlsx_route() -> None:
    assert readiness.probe_production_xlsx_route() is True
