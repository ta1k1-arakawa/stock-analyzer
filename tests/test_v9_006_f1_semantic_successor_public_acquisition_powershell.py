from __future__ import annotations

from pathlib import Path
import re
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v9_006_f1_semantic_successor_public_acquisition_production.ps1"
TEXT = SCRIPT.read_text(encoding="utf-8")


def test_powershell_ast_parse_succeeds_without_execution():
    command = ["pwsh", "-NoProfile", "-NonInteractive", "-Command", f"$text = Get-Content -Raw -LiteralPath '{SCRIPT}'; [System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$null, [ref]$null) | Out-Null"]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0 and completed.stderr == ""


def test_atomic_scope_stop_and_safe_parameter_surface():
    assert TEXT.lstrip().startswith("& {") and "$ErrorActionPreference = \"Stop\"" in TEXT
    assert "param(" in TEXT and "$ExpectedHead" in TEXT and "$Authorize" in TEXT
    for forbidden in ("Error", "Args", "Input", "Host", "PID", "HOME", "Matches"):
        assert re.search(rf"\${forbidden}(?:\W|$)", TEXT, flags=re.IGNORECASE) is None
    parameter_block = TEXT[TEXT.index("param("):TEXT.index("$ErrorActionPreference")]
    for forbidden_parameter in ("state-root", "root-url", "terminal-url", "provider", "retry", "passthrough"):
        assert forbidden_parameter not in parameter_block.lower()


def test_canonical_environment_and_exact_child_arguments_are_fixed():
    assert ".venv-real-execution\\Scripts\\python.exe" in TEXT
    assert ".venv\\Scripts\\python.exe" not in TEXT and '"python"' not in TEXT and '"py"' not in TEXT
    assert '"--state-root" $stateRoot "--implementation-git-sha" $ExpectedHead' in TEXT
    assert "Invoke-GitSafe" in TEXT and "fetch" in TEXT and "origin/$authoritativeBranch" in TEXT
    assert "--pull" not in TEXT and "merge" not in TEXT and "rebase" not in TEXT and "cherry-pick" not in TEXT and "--force" not in TEXT


def test_stable_private_state_and_preflight_before_authorization():
    assert 'Join-Path $stateParent "v9-006-f1-successor-public-acquisition-state"' in TEXT
    state_definition = TEXT[TEXT.index("$stateRoot ="):TEXT.index("$canonicalInterpreter")]
    assert "$ExpectedHead" not in state_definition
    assert "Test-Path -LiteralPath $stateRoot" in TEXT
    assert "if (-not $Authorize)" in TEXT and "Write-Output $authMarker" in TEXT
    assert "LISTED_ISSUES_PAGE_URL" not in TEXT
    assert "Write-Output $stateRoot" not in TEXT and "Write-Host $stateRoot" not in TEXT


def test_child_output_is_validated_without_echoing_stderr_or_real_execution():
    assert "2>&1 | Out-String" in TEXT
    assert "$childOutput" not in TEXT.split("catch", 1)[-1]
    assert "ConvertFrom-Json" in TEXT and "childLines.Count -ne 1" in TEXT
    assert "urllib" not in TEXT.lower() and "Invoke-WebRequest" not in TEXT


def test_low_2_argparse_path_is_unreachable_from_reviewed_invocation():
    invocation = TEXT.split("$childOutput", 1)[1]
    assert "--state-root" in invocation and "--implementation-git-sha" in invocation
    assert "@(" not in invocation.split("$canonicalInterpreter", 1)[-1].split("2>&1", 1)[0]
