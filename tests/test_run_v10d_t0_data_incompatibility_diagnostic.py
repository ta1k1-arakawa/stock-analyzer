from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import v10d_t0_data_incompatibility_diagnostic as diagnostic
from scripts import run_v10d_t0_data_incompatibility_diagnostic as runner


IMPLEMENTATION_SHA = "b" * 40


def _argv(tmp_path: Path, *mode: str) -> list[str]:
    return [
        *mode,
        "--repository-root",
        str(tmp_path / "repo"),
        "--training-cache",
        str(tmp_path / "training"),
        "--evaluation-cache",
        str(tmp_path / "evaluation"),
        "--universe-csv",
        str(tmp_path / "universe.csv"),
        "--implementation-sha",
        IMPLEMENTATION_SHA,
    ]


def test_production_cli_phase_a_dispatches_to_real_module_function(monkeypatch, tmp_path, capsys):
    observed = []
    metadata = diagnostic.PhaseAMetadata(SimpleNamespace(), IMPLEMENTATION_SHA, ("2018-01-01",))
    monkeypatch.setattr(
        diagnostic,
        "phase_a_metadata_preflight",
        lambda *args: observed.append(args) or metadata,
    )
    result_code = runner.main(_argv(tmp_path, "--phase-a-only"))
    captured = capsys.readouterr()
    assert result_code == 0
    assert len(observed) == 1
    output = json.loads(captured.out)
    assert output["phase_a_status"] == "PASS"
    assert output["payload_reads"] == 0
    assert output["parser_calls"] == 0
    assert output["authority_consumed"] is False


def test_production_cli_diagnostic_requires_guard_without_loader(monkeypatch, tmp_path, capsys):
    metadata = diagnostic.PhaseAMetadata(SimpleNamespace(), IMPLEMENTATION_SHA, ("2018-01-01",))
    monkeypatch.setattr(diagnostic, "phase_a_metadata_preflight", lambda *args: metadata)
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda *args, **kwargs: pytest.fail("guard bypass"))
    result_code = runner.main(_argv(tmp_path, "--diagnostic"))
    captured = capsys.readouterr()
    assert result_code == 4
    assert captured.err.strip() == "V10D_PRE_GATE_FAILURE"


def test_production_cli_diagnostic_uses_closed_safe_result(monkeypatch, tmp_path, capsys):
    metadata = diagnostic.PhaseAMetadata(SimpleNamespace(), IMPLEMENTATION_SHA, ("2018-01-01",))
    safe_result = diagnostic._safe_result(
        IMPLEMENTATION_SHA, diagnostic.RESULT_DATA, "UNKNOWN_DATA_INCOMPATIBILITY"
    )
    monkeypatch.setattr(diagnostic, "phase_a_metadata_preflight", lambda *args: metadata)
    monkeypatch.setattr(diagnostic, "run_diagnostic", lambda *args, **kwargs: safe_result)
    result_code = runner.main(
        [*_argv(tmp_path, "--diagnostic"), "--authority-boundary-token", diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN]
    )
    captured = capsys.readouterr()
    assert result_code == 0
    assert json.loads(captured.out)["result_class"] == diagnostic.RESULT_DATA


def test_production_cli_passes_phase_a_calendar_only(monkeypatch, tmp_path, capsys):
    metadata = diagnostic.PhaseAMetadata(SimpleNamespace(), IMPLEMENTATION_SHA, ("canonical-date",))
    observed = []
    safe_result = diagnostic._safe_result(
        IMPLEMENTATION_SHA, diagnostic.RESULT_DATA, "UNKNOWN_DATA_INCOMPATIBILITY"
    )
    monkeypatch.setattr(diagnostic, "phase_a_metadata_preflight", lambda *args: metadata)

    def observe(phase_a, *, authority_boundary_token):
        observed.append((phase_a, authority_boundary_token))
        return safe_result

    monkeypatch.setattr(diagnostic, "run_diagnostic", observe)
    result_code = runner.main(
        [*_argv(tmp_path, "--diagnostic"), "--authority-boundary-token", diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN]
    )

    assert result_code == 0
    assert observed[0][0].calendar_dates == ("canonical-date",)
    assert observed[0][1] == diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN


def test_cli_has_no_bypass_or_source_network_options():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "--force" not in source
    assert "--skip" not in source
    assert "urllib" not in source
    assert "requests" not in source


def test_diagnostic_cli_has_no_calendar_override_option():
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    assert "--calendar-json" not in source
