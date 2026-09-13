from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_v10b_training_cache_reacquisition.py"


def test_production_cli_exposes_only_operational_arguments():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--repo-root" in result.stdout
    assert "--attempt-root" in result.stdout
    assert "--implementation-sha" in result.stdout
    for forbidden in (
        "--provider",
        "--host",
        "--query",
        "--universe",
        "--ticker",
        "--fallback",
        "--retry",
        "--period",
    ):
        assert forbidden not in result.stdout


def test_legacy_or_methodology_override_arguments_are_rejected_without_network():
    for argument in ("--provider", "--universe", "--expected-r2-reviewed-sha"):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(REPO_ROOT),
                "--attempt-root",
                str(REPO_ROOT.parent / "synthetic-attempt"),
                "--implementation-sha",
                "0" * 40,
                argument,
                "synthetic",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 2
        assert "unrecognized arguments" in result.stderr


def test_cli_source_binds_fixed_endpoint_and_no_fallback_branch():
    source = SCRIPT.read_text(encoding="utf-8")
    core_source = (REPO_ROOT / "src" / "v10b_training_cache_reacquisition.py").read_text(
        encoding="utf-8"
    )
    assert "run_production" in source
    assert "query1.finance.yahoo.com" in core_source
    assert "fallback" not in source.lower()
    assert "provider" not in source.lower()
    assert "--attempt-root" in source
    assert "--implementation-sha" in source


def test_cli_governance_failure_is_bounded_and_does_not_emit_manifest(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(tmp_path / "not-a-repository"),
            "--attempt-root",
            str(tmp_path / "attempt"),
            "--implementation-sha",
            "0" * 40,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 4
    assert result.stdout == ""
    assert result.stderr.strip() == "V10B_ACQUISITION_PREFLIGHT_FAILURE"
    assert not (tmp_path / "attempt" / "cache_manifest.json").exists()


def test_cli_maps_explicit_post_boundary_failure_to_implementation_token(monkeypatch, capsys, tmp_path):
    spec = importlib.util.spec_from_file_location("v10b_cli_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    def fail_after_boundary(*args, **kwargs):
        raise __import__("src.v10b_training_cache_reacquisition", fromlist=["PostBoundaryFailure"]).PostBoundaryFailure(
            "synthetic post-boundary failure"
        )

    monkeypatch.setattr(cli, "run_production", fail_after_boundary)
    result = cli.main(
        [
            "--repo-root",
            str(tmp_path / "repo"),
            "--attempt-root",
            str(tmp_path / "attempt"),
            "--implementation-sha",
            "0" * 40,
        ]
    )
    captured = capsys.readouterr()
    assert result == 3
    assert captured.out == ""
    assert captured.err.strip() == "V10B_ACQUISITION_IMPLEMENTATION_FAILURE"
