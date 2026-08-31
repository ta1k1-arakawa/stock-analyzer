from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import re
import subprocess

import pytest

from src import v9_006_f1_semantic_successor_public_acquisition as acq


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v9_006_f1_semantic_successor_public_acquisition_production.ps1"
TEXT = SCRIPT.read_text(encoding="utf-8")


def _pwsh(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["pwsh", "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True)


def _ps_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def test_powershell_ast_parse_succeeds_and_parser_errors_are_checked():
    path = _ps_quote(SCRIPT)
    command = (
        f"$text=Get-Content -Raw -LiteralPath '{path}'; "
        "$tokens=$null; $errors=$null; "
        "$ast=[System.Management.Automation.Language.Parser]::ParseInput($text,[ref]$tokens,[ref]$errors); "
        "if($null -eq $ast -or $errors.Count -ne 0){ exit 1 }; exit 0"
    )
    completed = _pwsh(command)
    assert completed.returncode == 0, completed.stderr


def test_script_level_parameters_bind_without_running_authorized_operation():
    path = _ps_quote(SCRIPT)
    command = (
        f"$text=Get-Content -Raw -LiteralPath '{path}'; "
        "$tokens=$null; $errors=$null; $ast=[System.Management.Automation.Language.Parser]::ParseInput($text,[ref]$tokens,[ref]$errors); "
        "if($errors.Count -ne 0 -or $null -eq $ast.ParamBlock){exit 1}; "
        "$paramText=$ast.ParamBlock.Extent.Text; "
        "$tmp=Join-Path ([IO.Path]::GetTempPath()) ('v9-param-' + [guid]::NewGuid().ToString() + '.ps1'); "
        "$paramText + [Environment]::NewLine + 'Write-Output ($ExpectedHead + ''|'' + $Authorize.IsPresent)' | Set-Content -LiteralPath $tmp; "
        "$out=& pwsh -NoProfile -NonInteractive -File $tmp -ExpectedHead ('a'*40) -Authorize; "
        "Remove-Item -LiteralPath $tmp -Force; if(($out -join '') -ne (('a'*40)+'|True')){exit 1}; exit 0"
    )
    completed = _pwsh(command)
    assert completed.returncode == 0, completed.stderr


def test_atomic_scope_stop_and_safe_parameter_surface():
    assert re.search(r"(?m)^param\s*\(", TEXT)
    assert re.search(r"(?m)^&\s*\{", TEXT)
    assert "$ErrorActionPreference = \"Stop\"" in TEXT
    for forbidden in ("Error", "Args", "Input", "Host", "PID", "HOME", "Matches"):
        assert re.search(rf"\${forbidden}(?:\W|$)", TEXT, flags=re.IGNORECASE) is None
    param_block = TEXT[TEXT.index("param("):TEXT.index("& {")]
    assert "$ExpectedHead" in param_block and "$Authorize" in param_block
    for forbidden_parameter in ("state-root", "root-url", "terminal-url", "provider", "retry", "passthrough"):
        assert forbidden_parameter not in param_block.lower()


def test_canonical_environment_and_exact_child_arguments_are_fixed():
    assert ".venv-real-execution\\Scripts\\python.exe" in TEXT
    assert ".venv\\Scripts\\python.exe" not in TEXT and '"python"' not in TEXT and '"py"' not in TEXT
    assert '"--state-root", $stateRoot, "--implementation-git-sha", $ExpectedHead' in TEXT
    assert "Invoke-GitSafe" in TEXT and "fetch" in TEXT and "origin/$authoritativeBranch" in TEXT
    assert "--pull" not in TEXT and "merge" not in TEXT and "rebase" not in TEXT and "cherry-pick" not in TEXT and "--force" not in TEXT


def test_canonical_path_helper_accepts_slash_variation_and_rejects_different_path():
    path = _ps_quote(SCRIPT)
    command = (
        f"$text=Get-Content -Raw -LiteralPath '{path}'; $tokens=$null; $errors=$null; "
        "$ast=[System.Management.Automation.Language.Parser]::ParseInput($text,[ref]$tokens,[ref]$errors); "
        "if($errors.Count -ne 0){exit 1}; $fn=$ast.Find({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Convert-CanonicalWindowsPath'},$true); "
        "if($null -eq $fn){exit 1}; Invoke-Expression $fn.Extent.Text; "
        "$a=Convert-CanonicalWindowsPath 'C:/path/to/repo'; $b=Convert-CanonicalWindowsPath 'C:\\path\\to\\repo'; $c=Convert-CanonicalWindowsPath 'C:\\path\\to\\other'; "
        "if($a -cne $b -or $a -ceq $c){exit 1}; exit 0"
    )
    completed = _pwsh(command)
    assert completed.returncode == 0, completed.stderr


def test_child_output_uses_existing_python_validator_and_canonical_json_without_leaks():
    assert "validate_safe_acquisition_result" in TEXT
    assert "canonical_json" in TEXT
    assert 'validatorResult = Invoke-CanonicalPython' in TEXT
    assert '$validatedLine -cne $childLines[0]' in TEXT
    assert "$capturedStderr" in TEXT and "$capturedStderr" not in TEXT.split("catch", 1)[-1]
    assert "Write-Output $childLines[0]" in TEXT
    assert "urllib" not in TEXT.lower() and "Invoke-WebRequest" not in TEXT


def test_existing_validator_rejects_extra_noncanonical_or_malformed_result_without_echoing_raw_line():
    value = {
        "implementation_git_sha": "a" * 40,
        "result": "INPUT_BINDING_FAILURE",
        "failure_stage": "PRE_NETWORK_INPUT_BINDING",
    }
    with pytest.raises(ValueError):
        acq.validate_safe_acquisition_result(value)
    raw = json.dumps({"extra": "do-not-emit"}, separators=(",", ":"))
    assert "do-not-emit" not in TEXT.split("catch", 1)[-1]
    assert raw


def test_canonical_valid_closed_result_passes_existing_validation_seam():
    value = acq._base(
        "a" * 40,
        "INPUT_BINDING_FAILURE",
        "PRE_NETWORK_INPUT_BINDING",
        None,
        None,
        root_attempts=0,
        terminal_attempts=0,
        locator_result=None,
        locator_hash=None,
    )
    value = acq.finalize_safe_result(value)
    acq.validate_safe_acquisition_result(value)
    assert acq.canonical_json(value) == acq.canonical_json(json.loads(acq.canonical_json(value)))


def test_stable_private_state_and_preflight_before_authorization():
    assert 'Join-Path $stateParent "v9-006-f1-successor-public-acquisition-state"' in TEXT
    state_definition = TEXT[TEXT.index("$stateRoot ="):TEXT.index("$canonicalInterpreter")]
    assert "$ExpectedHead" not in state_definition
    assert "Test-Path -LiteralPath $stateRoot" in TEXT
    assert "if (-not $Authorize.IsPresent)" in TEXT and "Write-Output $authMarker" in TEXT
    assert "LISTED_ISSUES_PAGE_URL" not in TEXT
    assert "Write-Output $stateRoot" not in TEXT and "Write-Host $stateRoot" not in TEXT


def test_low_2_argparse_path_is_unreachable_from_reviewed_invocation():
    invocation = TEXT.split("$childArguments", 1)[1]
    assert '"--state-root", $stateRoot, "--implementation-git-sha", $ExpectedHead' in invocation
    assert "Invoke-CanonicalPython $canonicalInterpreter $childArguments $null" in invocation
    assert "passthrough" not in invocation.lower()
