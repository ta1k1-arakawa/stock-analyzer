from __future__ import annotations

from pathlib import Path
import re
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v9_006_f1_terminal_month_structure_evidence_production.ps1"
TEXT = SCRIPT.read_text(encoding="utf-8")


def _pwsh(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True)


def _quote(path: Path) -> str:
    return str(path).replace("'", "''")


def test_powershell_ast_and_parser_errors_collection():
    path = _quote(SCRIPT)
    command = (
        f"$text=Get-Content -Raw -LiteralPath '{path}'; $tokens=$null; $errors=$null; "
        "$ast=[System.Management.Automation.Language.Parser]::ParseInput($text,[ref]$tokens,[ref]$errors); "
        "if($null -eq $ast -or $errors.Count -ne 0){exit 1}; exit 0"
    )
    completed = _pwsh(command)
    assert completed.returncode == 0, completed.stderr


def test_script_parameter_surface_and_atomic_scope():
    assert re.search(r"(?m)^param\s*\(", TEXT)
    assert re.search(r"(?m)^&\s*\{", TEXT)
    assert '$ErrorActionPreference = "Stop"' in TEXT
    assert "$ExpectedHead" in TEXT and "$Authorize" in TEXT
    for forbidden in ("Error", "Args", "Input", "Host", "PID", "HOME", "Matches"):
        assert re.search(rf"\${forbidden}(?:\W|$)", TEXT, flags=re.IGNORECASE) is None
    parameter_block = TEXT[TEXT.index("param("):TEXT.index("& {")]
    assert re.findall(r"\$(ExpectedHead|Authorize)\b", parameter_block) == ["ExpectedHead", "Authorize"]
    for forbidden_parameter in ("state-root", "root-url", "terminal-url", "provider", "retry", "passthrough"):
        assert forbidden_parameter not in parameter_block.lower()


def test_fixed_environment_path_child_arguments_and_no_leakage():
    assert ".venv-real-execution\\Scripts\\python.exe" in TEXT
    assert ".venv\\Scripts\\python.exe" not in TEXT
    assert '"--diagnostic-implementation-git-sha", $ExpectedHead' in TEXT
    assert "Invoke-GitSafe" in TEXT and "origin/$authoritativeBranch" in TEXT
    assert "validate_safe_result" in TEXT and "canonical_json" in TEXT
    assert "$capturedStderr" in TEXT and "$capturedStderr" not in TEXT.split("catch", 1)[-1]
    assert "Write-Output $stateRoot" not in TEXT and "Write-Host $stateRoot" not in TEXT
    assert "--pull" not in TEXT and "merge" not in TEXT and "rebase" not in TEXT and "cherry-pick" not in TEXT and "--force" not in TEXT


def test_stable_state_identity_and_authorization_boundary():
    assert 'v9-006-f1-successor-public-acquisition-state' in TEXT
    state_definition = TEXT[TEXT.index("$stateRoot ="):TEXT.index("$canonicalInterpreter")]
    assert "$ExpectedHead" not in state_definition
    assert "if (-not $Authorize.IsPresent)" in TEXT
    assert "Write-Output $authMarker" in TEXT
    assert "Test-Path -LiteralPath $stateRoot -PathType Container" in TEXT
    assert "LISTED_ISSUES_PAGE_URL" not in TEXT
    assert "Invoke-WebRequest" not in TEXT and "urllib" not in TEXT.lower()


def test_environment_predicate_has_canonical_false_general_venv_semantics():
    assert "$readiness.GENERAL_PROJECT_VENV_REJECTED -eq $false" in TEXT
    assert "function Test-CanonicalEnvironmentPredicate" in TEXT
