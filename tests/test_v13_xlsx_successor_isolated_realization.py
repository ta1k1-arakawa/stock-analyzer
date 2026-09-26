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


def test_current_process_identity_requires_exact_canonical_path(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical" / "python.exe"
    canonical.parent.mkdir()
    canonical.touch()
    other = tmp_path / "other" / "python.exe"
    other.parent.mkdir()
    other.touch()
    monkeypatch.setattr(run.current, "CANONICAL_INTERPRETER", canonical)
    monkeypatch.setattr(run.sys, "version_info", (3, 12, 10))
    monkeypatch.setattr(run.sysconfig, "get_platform", lambda: "win-amd64")
    run.canonical_process_interpreter(canonical)
    with pytest.raises(ValueError, match="PRE_GATE_WRONG_PYTHON_ENVIRONMENT"):
        run.canonical_process_interpreter(other)


def test_wrong_process_never_reaches_preflight_or_durable_root(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical.exe"
    other = tmp_path / "other.exe"
    canonical.touch()
    other.touch()
    monkeypatch.setattr(run.current, "CANONICAL_INTERPRETER", canonical)
    monkeypatch.setattr(run.sys, "executable", str(other))
    monkeypatch.setattr(run, "preflight", lambda _: pytest.fail("preflight reached"))
    monkeypatch.setattr(run.phase_b, "_prepare_operation_parent",
                        lambda _: pytest.fail("durable root reached"))
    with pytest.raises(ValueError, match="PRE_GATE_WRONG_PYTHON_ENVIRONMENT"):
        run.realize("synthetic-head")


@pytest.mark.parametrize("change", ["missing", "open_version", "xml_version", "failed", "false"])
def test_pre_gate_probe_fails_closed_on_manifest_or_parser_result(tmp_path, monkeypatch, change):
    canonical = tmp_path / "python.exe"
    canonical.touch()
    monkeypatch.setattr(run.current, "CANONICAL_INTERPRETER", canonical)
    wheels = ({"name": "openpyxl", "version": "3.1.5", "filename": "open.whl"},
              {"name": "et-xmlfile", "version": "2.0.0", "filename": "xml.whl"})
    if change == "missing":
        wheels = wheels[:1]
    elif change in {"open_version", "xml_version"}:
        target = 0 if change == "open_version" else 1
        wheels = tuple({**w, "version": "0"} if i == target else w
                       for i, w in enumerate(wheels))

    class Result:
        returncode = 1 if change == "failed" else 0
        stderr = ""
        stdout = json.dumps({"ready": change != "false", "executable": str(canonical),
                             "isolated": True})

    def child(argv, **kwargs):
        assert argv[0] == str(canonical)
        assert argv[1:3] == ["-I", "-B"]
        assert "--no-index" not in argv  # This probe never invokes pip.
        return Result()

    monkeypatch.setattr(run.subprocess, "run", child)
    with pytest.raises(ValueError, match="PRE_GATE_XLSX_"):
        run.probe_frozen_wheel_xlsx(tmp_path, wheels)


def test_pre_gate_probe_accepts_only_verified_archive_route(tmp_path, monkeypatch):
    canonical = tmp_path / "python.exe"
    canonical.touch()
    monkeypatch.setattr(run.current, "CANONICAL_INTERPRETER", canonical)
    wheels = ({"name": "openpyxl", "version": "3.1.5", "filename": "open.whl"},
              {"name": "et-xmlfile", "version": "2.0.0", "filename": "xml.whl"})

    def child(argv, **kwargs):
        assert argv[:4] == [str(canonical), "-I", "-B", "-c"]
        assert "probe_production_xlsx_route" in argv[4]
        assert argv[-2:] == [str(tmp_path / "open.whl"), str(tmp_path / "xml.whl")]
        assert kwargs["env"]["PIP_NO_INDEX"] == "1"
        return subprocess.CompletedProcess(argv, 0, json.dumps({
            "ready": True, "executable": str(canonical), "isolated": True}), "")

    monkeypatch.setattr(run.subprocess, "run", child)
    run.probe_frozen_wheel_xlsx(tmp_path, wheels)


def test_verified_wheels_precedes_probe_and_probe_precedes_root(tmp_path, monkeypatch):
    events = []
    candidate = tmp_path / "candidate"
    wheelhouse = tmp_path / "wheels"
    monkeypatch.setattr(run, "canonical_process_interpreter", lambda: events.append("identity"))
    monkeypatch.setattr(run, "preflight", lambda _: {})
    monkeypatch.setattr(run, "canonical_candidate_root", lambda: candidate)
    monkeypatch.setattr(run, "verified_wheels", lambda: (events.append("wheels") or
                                                          (wheelhouse, ())))
    monkeypatch.setattr(run, "precheck_paths", lambda *args: None)
    monkeypatch.setattr(run.sys, "_base_executable", str(tmp_path / "base.exe"), raising=False)
    (tmp_path / "base.exe").touch()
    monkeypatch.setattr(run.current, "CANONICAL_INTERPRETER", tmp_path / "canonical.exe")
    monkeypatch.setattr(run.sys, "version_info", (3, 12, 10))
    def probe(*args):
        events.append("probe")
        raise ValueError("PRE_GATE_XLSX_PROBE_FAILED")
    monkeypatch.setattr(run, "probe_frozen_wheel_xlsx", probe)
    monkeypatch.setattr(run.phase_b, "_prepare_operation_parent",
                        lambda _: pytest.fail("durable parent reached"))
    with pytest.raises(ValueError, match="PRE_GATE_XLSX_PROBE_FAILED"):
        run.realize("synthetic-head")
    assert events == ["identity", "wheels", "probe"]
    assert not candidate.exists()


def test_probe_failure_reports_reusable_pre_gate_authority(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["realize", "--execute", "--execution-head", "synthetic"])
    monkeypatch.setattr(run.sys, "platform", "win32")
    monkeypatch.setattr(run, "canonical_candidate_root", lambda: tmp_path / "candidate")
    monkeypatch.setattr(run, "realize", lambda _: (_ for _ in ()).throw(
        ValueError("PRE_GATE_XLSX_PROBE_FAILED")))
    assert run.main() == 1
    assert json.loads(capsys.readouterr().out) == {
        "status": "FAIL", "failure_class": "PRE_GATE_FAILURE",
        "reason": "PRE_GATE_XLSX_PROBE_FAILED", "authorization_reusable": True}
