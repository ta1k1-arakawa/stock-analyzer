"""Repository-only tests. No consumed machine-local state or environment is opened."""

from __future__ import annotations

import json
import hashlib
import zipfile
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import v13_xlsx_successor_isolated_realization as run
from scripts import v13_xlsx_successor_freeze as freeze


def test_authorization_literal_scope_and_review_bindings():
    value = json.loads(run.AUTH.read_bytes())
    assert value["status"] == "AUTHORIZED"
    assert value["predecessor_gpt_reviewed_sha"] == run.PREDECESSOR
    assert value["predecessor_gpt_review_comment_id"] == 5847530715
    assert value["human_approval_comment_id"] == 5847574276
    assert value["human_approval_text"] == (
        "V13 XLSX successor の isolated candidate environment realization を承認します．"
        "既存の固定済み29-wheelのみを使用し，package-index再アクセス，active .venv-real-execution の変更，"
        "successor promotion は承認しません．")
    assert value["candidate_lock"]["package_count"] == 29
    assert value["freeze_record"]["wheel_count"] == 29
    assert value["current_authority_lock"]["package_count"] == 27
    assert value["source_phase_b_authorization_consumed"] is True
    assert value["source_phase_b_authorization_reusable"] is False
    assert value["one_shot"] is True and value["consumed"] is False
    assert value["reusable_after_durable_boundary"] is False
    assert all(value[key] is False for key in (
        "package_index_resolution_allowed", "network_download_allowed",
        "active_venv_real_execution_mutation_allowed", "successor_promotion_allowed",
        "current_authority_change_allowed", "jpx_yahoo_requests_allowed",
        "private_t1_reads_allowed", "selected_500_allowed",
        "model_fit_backtest_aq_paper_live_trading_allowed"))
    assert freeze.validate_freeze()["status"] == "PASS"


@pytest.mark.parametrize("records", [
    lambda x: x[:-1],
    lambda x: x + [{"name": "unknown", "version": "1"}],
    lambda x: [*x[:-1], {"name": x[-1]["name"], "version": "0"}],
    lambda x: [*x[:-1], x[0]],
])
def test_installed_metadata_rejects_missing_extra_drift_and_duplicate(records):
    expected = {"openpyxl": "3.1.5", "et-xmlfile": "2.0.0"}
    expected.update({f"pkg{i}": "1" for i in range(27)})
    clean = [{"name": name, "version": version} for name, version in expected.items()]
    run.validate_installed(clean, expected)
    with pytest.raises(ValueError):
        run.validate_installed(records(clean), expected)


def test_caller_cannot_select_wheelhouse_candidate_or_network(tmp_path):
    script = Path(run.__file__)
    for option in ("--wheelhouse", "--phase-b-root", "--candidate-root", "--index-url"):
        result = subprocess.run([sys.executable, "-m", "scripts.v13_xlsx_successor_isolated_realization",
                                 option, str(tmp_path)],
                                capture_output=True, text=True, cwd=run.ROOT)
        assert result.returncode != 0
    assert "--no-index" in run.realize.__code__.co_consts or "--no-index" in script.read_text()


def test_root_is_unique_and_exclusive_before_creation(tmp_path, monkeypatch):
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    monkeypatch.setattr(run, "ROOT", fake_repo)
    root = tmp_path / "candidate"
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    run.precheck_paths(root, wheelhouse)
    root.mkdir()
    with pytest.raises(ValueError, match="ONE_SHOT_ALREADY_STARTED"):
        run.precheck_paths(root, wheelhouse)


def test_rehearsal_has_no_mutation_and_execution_requires_explicit_head():
    result = subprocess.run([sys.executable, "-m", "scripts.v13_xlsx_successor_isolated_realization"],
                            capture_output=True, text=True, cwd=run.ROOT, check=True)
    value = json.loads(result.stdout)
    assert value == {"rehearsal": True, "network_requests": 0, "wheel_downloads": 0,
                     "package_installations": 0, "environment_mutations": 0,
                     "durable_candidate_creation": 0, "private_reads": 0,
                     "jpx_yahoo_requests": 0}
    result = subprocess.run([sys.executable, "-m", "scripts.v13_xlsx_successor_isolated_realization",
                             "--execute"], capture_output=True, text=True, cwd=run.ROOT)
    assert result.returncode != 0
    assert json.loads(result.stdout)["failure_class"] == "PRE_GATE_FAILURE"


def test_production_probe_is_actual_parser_route_and_fails_without_reader(monkeypatch):
    from scripts import v13_xlsx_readiness as readiness
    assert "parse_jpx" in readiness.probe_production_xlsx_route.__code__.co_names
    assert readiness.synthetic_jpx_xlsx().startswith(b"PK\x03\x04")
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    with pytest.raises((ImportError, ModuleNotFoundError)):
        readiness.probe_production_xlsx_route()


@pytest.mark.parametrize("break_case", [None, "missing", "extra", "version", "duplicate", "probe"])
def test_isolated_observer_requires_exact_set_and_production_probe(monkeypatch, tmp_path, break_case):
    expected = {"openpyxl": "3.1.5", "et-xmlfile": "2.0.0"}
    expected.update({f"pkg{i}": "1" for i in range(27)})
    records = [{"name": name, "version": version} for name, version in expected.items()]
    if break_case == "missing":
        records.pop()
    elif break_case == "extra":
        records.append({"name": "extra", "version": "1"})
    elif break_case == "version":
        records[0]["version"] = "3.1.4"
    elif break_case == "duplicate":
        records[-1] = records[0].copy()
    interpreter = tmp_path / "python.exe"
    observation = {"records": records, "ready": break_case != "probe",
                   "executable": str(interpreter), "isolated": True, "no_user_site": True,
                   "python_version": "3.12.10", "platform": "win-amd64",
                   "pythonpath_absent": True}
    class Result:
        returncode = 0
        stderr = ""
        stdout = json.dumps(observation)
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **kw: Result())
    if break_case is None:
        run.inspect_candidate(interpreter, expected)
    else:
        with pytest.raises(ValueError):
            run.inspect_candidate(interpreter, expected)


def test_synthetic_wheelhouse_rejects_missing_extra_and_tampered(tmp_path):
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    name = "sample-1.0-py3-none-any.whl"
    wheel = wheelhouse / name
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("sample-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: sample\nVersion: 1.0\n")
        archive.writestr("sample-1.0.dist-info/WHEEL", "Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    manifest = [{"name": "sample", "version": "1.0", "filename": name,
                 "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}]
    assert len(run.verify_wheelhouse(wheelhouse, manifest)) == 1
    wheel.unlink()
    with pytest.raises(ValueError):
        run.verify_wheelhouse(wheelhouse, manifest)
    wheel.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        run.verify_wheelhouse(wheelhouse, manifest)
    wheel.write_bytes(b"extra")
    (wheelhouse / "another.whl").write_bytes(b"extra")
    with pytest.raises(ValueError):
        run.verify_wheelhouse(wheelhouse, manifest)
