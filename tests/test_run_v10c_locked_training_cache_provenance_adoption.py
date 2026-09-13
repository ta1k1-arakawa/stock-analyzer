from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import src.v10c_locked_training_cache_provenance_adoption as core


def _load_cli():
    path = Path("scripts/run_v10c_locked_training_cache_provenance_adoption.py").resolve()
    spec = importlib.util.spec_from_file_location("v10c_adoption_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_surface_is_operational_only() -> None:
    cli = _load_cli()
    option_strings = {option for action in cli._parser()._actions for option in action.option_strings}
    assert option_strings == {"-h", "--help", "--candidate-root", "--implementation-sha", "--authorization-marker", "--receipt-path"}
    assert "--repo-root" not in option_strings
    assert cli.RUNNER_REPO_ROOT == Path(cli.__file__).resolve().parents[1]


def test_cli_governance_failure_has_no_scientific_result(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cli = _load_cli()

    def fail(*args: object, **kwargs: object) -> None:
        raise core.GovernanceProvenanceFailure("synthetic")

    monkeypatch.setattr(cli, "phase_a_preflight", fail)
    code = cli.main(["--candidate-root", str(tmp_path), "--implementation-sha", "a" * 40, "--authorization-marker", str(tmp_path / "m"), "--receipt-path", str(tmp_path / "r")])
    captured = capsys.readouterr()
    assert code == 4
    assert captured.out == ""
    assert captured.err == "V10C_OFFLINE_ADOPTION_GOVERNANCE_FAILURE\n"


def test_cli_implementation_failure_uses_bounded_implementation_surface(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cli = _load_cli()
    monkeypatch.setattr(cli, "phase_a_preflight", lambda *args, **kwargs: {"candidate_root": tmp_path, "ticker_order": []})

    def fail(*args: object, **kwargs: object) -> None:
        raise core.ImplementationFailure("synthetic")

    monkeypatch.setattr(cli, "phase_b_offline_adoption", fail)
    code = cli.main(["--candidate-root", str(tmp_path), "--implementation-sha", "a" * 40, "--authorization-marker", str(tmp_path / "m"), "--receipt-path", str(tmp_path / "r")])
    captured = capsys.readouterr()
    assert code == 3
    assert captured.out == ""
    assert captured.err == "V10C_OFFLINE_ADOPTION_IMPLEMENTATION_FAILURE\n"


def test_cli_unexpected_exception_uses_bounded_implementation_surface(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cli = _load_cli()

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("not safe to expose")

    monkeypatch.setattr(cli, "phase_a_preflight", fail)
    code = cli.main(["--candidate-root", str(tmp_path), "--implementation-sha", "a" * 40, "--authorization-marker", str(tmp_path / "m"), "--receipt-path", str(tmp_path / "r")])
    captured = capsys.readouterr()
    assert code == 3
    assert captured.out == ""
    assert captured.err == "V10C_OFFLINE_ADOPTION_IMPLEMENTATION_FAILURE\n"


def test_cli_success_prints_only_safe_receipt(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    cli = _load_cli()
    safe = {"schema_version": core.RECEIPT_SCHEMA, "execution_result": "PASS", "network_requests": 0}
    monkeypatch.setattr(cli, "phase_a_preflight", lambda *args, **kwargs: {"candidate_root": tmp_path, "ticker_order": []})
    monkeypatch.setattr(cli, "phase_b_offline_adoption", lambda *args, **kwargs: safe)
    code = cli.main(["--candidate-root", str(tmp_path), "--implementation-sha", "a" * 40, "--authorization-marker", str(tmp_path / "m"), "--receipt-path", str(tmp_path / "r")])
    captured = capsys.readouterr()
    assert code == 0
    assert "PASS" in captured.out
    assert str(tmp_path) not in captured.out
    assert captured.err == ""


def test_cli_rejects_calendar_or_source_override() -> None:
    cli = _load_cli()
    with pytest.raises(SystemExit):
        cli.main(["--candidate-root", "C:/candidate", "--implementation-sha", "a" * 40, "--authorization-marker", "C:/marker", "--receipt-path", "C:/receipt", "--provider", "other"])
