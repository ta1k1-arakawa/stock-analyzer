"""No-network checks for the future direct-Windows entrypoint."""

import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest

from src import v8_jquants_identity_recovery as jq


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v8_jquants_identity_recovery_direct_windows.ps1"


def test_runner_has_closed_preboundary_order():
    source = SCRIPT.read_text(encoding="utf-8")
    checks = [
        "orca[\\/]workspaces", "branch', '--show-current'", "ls-remote",
        "status', '--porcelain'", "ExpectedScriptBlob", "ExpectedImplementationBlob",
        "check_current_protected_environment.py", "readiness_probe()",
        "inspect_state(private_root(Path.cwd()))", "JQUANTS_API_KEY",
        "ExecuteReviewedAcquisition", "-m src.v8_jquants_identity_recovery",
    ]
    positions = []
    for check in checks:
        if check == "orca[\\/]workspaces":
            positions.append(source.index("workspaces"))
        else:
            positions.append(source.index(check, positions[-1] + 1 if positions else 0))
    assert positions == sorted(positions)
    assert "Write-Output $result" in source
    assert "2>$null" in source
    assert "V8_JQUANTS_REVIEWED_HEAD', $oldHead" in source
    assert "V8_JQUANTS_REVIEWED_BLOB', $oldBlob" in source


def test_runner_generated_worktree_blocks_before_network():
    ps = shutil.which("powershell.exe") or shutil.which("pwsh.exe") or shutil.which("pwsh")
    if ps is None:
        pytest.skip("PowerShell unavailable")
    command = [ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
               "-ExpectedHead", "a" * 40, "-ExpectedScriptBlob", "b" * 40,
               "-ExpectedImplementationBlob", "c" * 40]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert result.stdout.strip() == ("JQUANTS_RECOVERY_RESULT=BLOCK NETWORK_BOUNDARY_CROSSED=false "
        "JQUANTS_LOGICAL_ACQUISITIONS=0 JQUANTS_HTTP_REQUESTS=0 STAGE=PRE_GATE "
        "REASON=PRE_GATE_REPOSITORY_BLOCK RAW_CONTENT_LOCK_PUBLISHED=false "
        "RECOVERY_ARTIFACT_PUBLISHED=false")
    assert result.stderr == ""


@pytest.mark.parametrize("fault,reason", [
    ("branch", "PRE_GATE_REPOSITORY_BLOCK"),
    ("head", "PRE_GATE_REPOSITORY_BLOCK"),
    ("remote", "PRE_GATE_REPOSITORY_BLOCK"),
    ("dirty", "PRE_GATE_REPOSITORY_BLOCK"),
    ("blob", "PRE_GATE_PROVENANCE_BLOCK"),
    ("environment", "PRE_GATE_ENVIRONMENT_BLOCK"),
    ("private", "PRE_GATE_PRIVATE_ROOT_BLOCK"),
])
def test_python_network_boundary_preflight_blocks(tmp_path, monkeypatch, fault, reason):
    (tmp_path / ".git").mkdir()
    expected_python = tmp_path / ".venv-real-execution" / "Scripts" / "python.exe"
    monkeypatch.setattr(jq.sys, "executable", str(expected_python))
    monkeypatch.setattr(jq, "readiness_probe", lambda: True)
    def root(_repo):
        if fault == "private":
            raise jq.Block("PRE_GATE_PRIVATE_ROOT_BLOCK")
        return tmp_path / "private"
    monkeypatch.setattr(jq, "private_root", root)
    monkeypatch.setattr(jq, "inspect_state", lambda _root: "absent")
    def run(args, **_kwargs):
        if args[0] != "git":
            return SimpleNamespace(returncode=1 if fault == "environment" else 0, stdout=b"")
        command = tuple(args[3:])
        if command == ("rev-parse", "--show-toplevel"):
            output = str(tmp_path)
        elif command == ("branch", "--show-current"):
            output = "wrong" if fault == "branch" else "v13-conditional-cross-sectional-short-horizon"
        elif command == ("rev-parse", "HEAD"):
            output = "0" * 40 if fault == "head" else "a" * 40
        elif command == ("status", "--porcelain"):
            output = " M dirty" if fault == "dirty" else ""
        elif command == ("remote", "get-url", "origin"):
            output = "https://github.com/ta1k1-arakawa/stock-analyzer.git"
        elif command[:2] == ("ls-remote", "--exit-code"):
            output = ("0" * 40 if fault == "remote" else "a" * 40) + "\trefs/heads/v13-conditional-cross-sectional-short-horizon"
        elif command[:1] == ("rev-parse",):
            output = "0" * 40 if fault == "blob" else ("b" * 40 if "src/" in command[1] else "c" * 40)
        elif command[:1] == ("hash-object",):
            output = "b" * 40 if "src/" in command[-1] else "c" * 40
        else:
            raise AssertionError(command)
        return SimpleNamespace(returncode=0, stdout=output)
    monkeypatch.setattr(jq.subprocess, "run", run)
    with pytest.raises(jq.Block) as exc:
        jq._protected_main_preflight(tmp_path, "a" * 40, "b" * 40, "c" * 40)
    assert exc.value.reason == reason
