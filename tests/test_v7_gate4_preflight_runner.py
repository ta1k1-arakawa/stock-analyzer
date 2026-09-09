from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.preflight_v7_gate4 as runner
from src.v7_gate4_preflight import V7Gate4PreflightBlocked


ARGS = [
    "--seed-bundle", "seed", "--calendar-json", "calendar.json",
    "--calendar-raw", "calendar.html", "--universe-csv", "universe.csv",
    "--prospective-boundary", "2026-08-10",
]


def test_cli_exactly_exposes_read_only_inputs():
    parser = runner.build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert options == {"-h", "--help", "--seed-bundle", "--calendar-json", "--calendar-raw", "--universe-csv", "--prospective-boundary"}


def test_cli_rejects_activation_option():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(ARGS + ["--activate"])


def test_cli_rejects_network_option():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(ARGS + ["--network"])


def test_cli_rejects_authorization_option():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(ARGS + ["--authorization"])


def test_cli_passes_all_paths_and_boundary_to_core(monkeypatch, capsys):
    observed = {}

    def fake(**kwargs):
        observed.update(kwargs)
        return {"status": "PASS", "gate4_activation_ready": False}

    monkeypatch.setattr(runner, "run_gate4_preflight", fake)
    assert runner.main(ARGS) == 0
    assert observed == {
        "seed_bundle": "seed", "calendar_json": "calendar.json", "calendar_raw": "calendar.html",
        "universe_csv": "universe.csv", "prospective_boundary": "2026-08-10",
    }
    assert json.loads(capsys.readouterr().out) == {"gate4_activation_ready": False, "status": "PASS"}


def test_cli_blocked_core_is_fail_closed(monkeypatch, capsys):
    def blocked(**kwargs):
        raise V7Gate4PreflightBlocked("RAW_PAYLOAD_SHA_MISMATCH:3633")

    monkeypatch.setattr(runner, "run_gate4_preflight", blocked)
    assert runner.main(ARGS) == 2
    assert json.loads(capsys.readouterr().out) == {"reason": "RAW_PAYLOAD_SHA_MISMATCH:3633", "status": "BLOCKED"}


def test_cli_success_output_is_single_canonical_json_line(monkeypatch, capsys):
    monkeypatch.setattr(runner, "run_gate4_preflight", lambda **kwargs: {"z": 1, "a": 2})
    assert runner.main(ARGS) == 0
    output = capsys.readouterr().out
    assert output.endswith("\n") and output.count("\n") == 1
    assert output.index('"a"') < output.index('"z"')


def test_cli_requires_prospective_boundary():
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(ARGS[:-2])


def test_runner_source_has_no_network_transport():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "urlopen" not in source and "requests" not in source and "socket" not in source


def test_runner_source_does_not_create_activation_manifest():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "activation_manifest.json" not in source
    assert "activation_authorization_utc" not in source
