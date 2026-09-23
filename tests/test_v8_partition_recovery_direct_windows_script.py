import base64
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v8_partition_recovery_direct_windows.ps1"


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_script_is_atomic_and_cleans_transient_state():
    source = _script()
    assert source.startswith("[CmdletBinding()]\nparam(")
    assert "[Parameter(Mandatory = $true)]" in source
    assert "[string]$ExpectedScriptBlob" in source
    assert "& {\n    $ErrorActionPreference = 'Stop'" in source
    assert "$ErrorActionPreference = 'Stop'" in source
    assert "    finally {" in source
    assert "SetEnvironmentVariable($pythonPayloadPath, $null, 'Process')" in source
    assert "SetEnvironmentVariable($pythonArtifactPath, $null, 'Process')" in source
    assert "SetEnvironmentVariable($pythonPayloadHash, $null, 'Process')" in source
    assert "[System.IO.File]::Delete($temporaryPayload)" in source


def test_all_preflight_gates_precede_network_boundary():
    source = _script()
    boundary = source.index("$networkBoundaryCrossed = $true")
    for gate in (
        "PRE_GATE_NOT_AUTHORITATIVE_CHECKOUT",
        "PRE_GATE_GENERATED_WORKTREE",
        "PRE_GATE_WRONG_BRANCH",
        "PRE_GATE_LOCAL_REMOTE_HEAD_MISMATCH",
        "PRE_GATE_DIRTY_WORKTREE",
        "PRE_GATE_RECOVERY_IMPLEMENTATION_ANCESTRY_MISMATCH",
        "PRE_GATE_REVIEWED_SCRIPT_BLOB_MISMATCH",
        "PRE_GATE_REVIEWED_SCRIPT_WORKTREE_BLOB_MISMATCH",
        "PRE_GATE_REVIEWED_BLOB_MISMATCH",
        "PRE_GATE_ARTIFACT_ROOT_INSIDE_REPOSITORY",
        "PRE_GATE_ARTIFACT_ALREADY_EXISTS",
        "PRE_GATE_PROTECTED_ENVIRONMENT_BLOCK",
        "PRE_GATE_OPERATION_PARSER_BLOCK",
    ):
        assert source.index(gate) < boundary
    assert "'ls-remote', '--exit-code'" in source
    assert "fetch" not in source.lower()
    assert "Invoke-OperationParserProbe $pythonExe $temporaryProbePath $parserProbe" in source
    assert "-c $parserProbe" not in source


def test_review_binding_requires_remote_local_equality_ancestor_and_exact_script_blobs():
    source = _script()
    boundary = source.index("$networkBoundaryCrossed = $true")
    preflight = source[:boundary]
    assert "$remoteHead -cne $localHead" in preflight
    assert "merge-base --is-ancestor $recoveryImplementationHead $localHead" in preflight
    assert "$recoveryImplementationHead = '13c7e6f30bf7f5be10e0f47f2b67a0410084d15c'" in preflight
    assert "${localHead}:$scriptRelativePath" in preflight
    assert "'hash-object', '--path', $scriptRelativePath, $scriptRelativePath" in preflight
    assert "$committedScriptBlob -cne $ExpectedScriptBlob" in preflight
    assert "$workingScriptBlob -cne $ExpectedScriptBlob" in preflight
    assert "PRE_GATE_LOCAL_HEAD_MISMATCH" not in source
    assert "git reset" not in source.lower()
    assert "git checkout" not in source.lower()
    assert "git pull" not in source.lower()
    assert "git merge " not in source.lower()
    assert "git rebase" not in source.lower()


def test_destination_is_mechanical_outside_repository_and_write_once():
    source = _script()
    assert "Join-Path $localAppData 'stock-analyzer\\private\\v8-recovery'" in source
    assert "GetFullPath" in source
    assert "StartsWith($repoPrefix" in source
    assert "CreateDirectory($artifactRoot)" in source
    assert source.index("CreateDirectory($artifactRoot)") < source.index("$networkBoundaryCrossed = $true")
    assert "PRE_GATE_ARTIFACT_ALREADY_EXISTS" in source
    assert "write_v8_partition_recovery_manifest_once" not in source
    assert "recover_and_publish_v8_partition_once(" in source


def test_request_guard_is_one_shot_and_no_redirect_or_retry():
    source = _script()
    assert source.count("$requestCount = 1") == 1
    assert source.count("$request.GetResponse()") == 1
    assert "$request.AllowAutoRedirect = $false" in source
    assert "ComputeHash($payloadBytes)" in source
    assert "hashlib.sha256(raw).hexdigest() != os.environ[\"V8_RECOVERY_TRANSIENT_SHA256\"]" in source
    boundary = source.index("$networkBoundaryCrossed = $true")
    network_operation = source[boundary:]
    assert "while (" not in network_operation.lower()
    assert "foreach (" not in network_operation.lower()
    assert "retry" not in network_operation.lower()


def test_safe_report_does_not_emit_paths_payload_or_member_assignments():
    source = _script()
    assert "RECOVERY_RESULT=PASS NETWORK_BOUNDARY_CROSSED=true JPX_SOURCE_REQUESTS=1" in source
    assert "SEALED_IDENTITIES_PUBLICLY_DISCLOSED=false" in source
    assert "Write-Output $artifactPath" not in source
    assert "Write-Output $temporaryPayload" not in source
    assert "Write-Output $payloadBytes" not in source
    assert "RECOVERY_RESULT=NOT_EXECUTED NETWORK_BOUNDARY_CROSSED=false JPX_SOURCE_REQUESTS=0" in source


def _powershell_executable() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def _run_parser_probe_with_powershell(probe_text: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    source = _script()
    helper_start = source.index("    function Invoke-OperationParserProbe(")
    helper_end = source.index("\n\n    try {", helper_start)
    helper = source[helper_start:helper_end]
    probe_match = re.search(r"\$parserProbe = @'\n(.*?)\n'@", source, re.DOTALL)
    assert probe_match is not None
    actual_probe = probe_match.group(1)
    if probe_text == "__ACTUAL_PROBE__":
        probe_text = actual_probe

    probe_path = tmp_path / "operation-parser-probe.py"
    python_exe = sys.executable.replace("'", "''")
    probe_path_ps = str(probe_path).replace("'", "''")
    probe_literal = "@'\n" + probe_text + "\n'@"
    harness = f"""
$ErrorActionPreference = 'Stop'
{helper}
$pythonExe = '{python_exe}'
$probePath = '{probe_path_ps}'
$probeText = {probe_literal}
try {{
    $result = Invoke-OperationParserProbe $pythonExe $probePath $probeText
    Write-Output $result
    Write-Output 'NETWORK_BOUNDARY_CROSSED=false JPX_SOURCE_REQUESTS=0'
    if (Test-Path -LiteralPath $probePath) {{ throw 'PROBE_FILE_NOT_CLEANED' }}
    Write-Output 'PROBE_FILE_CLEANED=true'
}}
catch {{
    $reason = [string]$_.Exception.Message
    if ($reason -notmatch '^PRE_GATE_[A-Z0-9_]+$') {{ $reason = 'PRE_GATE_OPERATION_PARSER_BLOCK' }}
    Write-Output $reason
    Write-Output 'NETWORK_BOUNDARY_CROSSED=false JPX_SOURCE_REQUESTS=0'
    if (Test-Path -LiteralPath $probePath) {{ throw 'PROBE_FILE_NOT_CLEANED' }}
    Write-Output 'PROBE_FILE_CLEANED=true'
    exit 1
}}
"""
    encoded = base64.b64encode(harness.encode("utf-16le")).decode("ascii")
    powershell = _powershell_executable()
    assert powershell is not None
    return subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_windows_actual_native_python_probe_path_passes_without_network():
    powershell = _powershell_executable()
    if os.name != "nt" or powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    dependencies = subprocess.run(
        [sys.executable, "-c", "import pandas, xlrd"], capture_output=True, check=False
    )
    if dependencies.returncode != 0:
        pytest.skip("The current Python environment lacks the parser dependencies")

    temp_root = Path(os.environ["LOCALAPPDATA"]) / "Temp"
    with tempfile.TemporaryDirectory(prefix="v8-parser-probe-test-", dir=temp_root) as directory:
        result = _run_parser_probe_with_powershell("__ACTUAL_PROBE__", Path(directory))
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "OPERATION_PARSER_PROBE_PASS" in output
        assert "NETWORK_BOUNDARY_CROSSED=false JPX_SOURCE_REQUESTS=0" in output
        assert "PROBE_FILE_CLEANED=true" in output


def test_windows_broken_probe_has_safe_pre_gate_reason_and_zero_requests():
    powershell = _powershell_executable()
    if os.name != "nt" or powershell is None:
        pytest.skip("Windows PowerShell is unavailable")

    temp_root = Path(os.environ["LOCALAPPDATA"]) / "Temp"
    with tempfile.TemporaryDirectory(prefix="v8-parser-probe-test-", dir=temp_root) as directory:
        result = _run_parser_probe_with_powershell("this is deliberately invalid Python !!!", Path(directory))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "PRE_GATE_OPERATION_PARSER_BLOCK" in output
        assert "NETWORK_BOUNDARY_CROSSED=false JPX_SOURCE_REQUESTS=0" in output
        assert "PROBE_FILE_CLEANED=true" in output
        assert "Traceback" not in output
