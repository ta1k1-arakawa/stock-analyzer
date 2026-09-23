import atexit
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import pytest
from test_v8_partition_recovery import _fixture


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
    assert "recover_and_publish_v8_partition_once(" not in source
    assert source.index("_validate_accepted_manifest(manifest)") < source.index("write_v8_partition_recovery_manifest_once(manifest,")


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
    assert "RECOVERY_RESULT=PASS NETWORK_BOUNDARY_CROSSED=true JPX_SOURCE_REQUESTS=1 STAGE=SUCCESSFUL_PUBLICATION" in source
    assert "SEALED_IDENTITIES_PUBLICLY_DISCLOSED=false" in source
    assert "Write-Output $artifactPath" not in source
    assert "Write-Output $temporaryPayload" not in source
    assert "Write-Output $payloadBytes" not in source
    assert "RECOVERY_RESULT=NOT_EXECUTED NETWORK_BOUNDARY_CROSSED=false JPX_SOURCE_REQUESTS=0" in source
    assert "if ($networkBoundaryCrossed) {\n            if ($postNetworkReason)" in source
    assert "REASON=EXECUTION_BLOCKED" not in source[source.index("$networkBoundaryCrossed = $true"):]


def test_closed_post_network_stage_pipeline_and_publication_order():
    source = _script()
    runner_start = source.index("$runner = @'\n") + len("$runner = @'\n")
    runner_end = source.index("\n'@", runner_start)
    runner = source[runner_start:runner_end]
    compile(runner, "embedded_v8_recovery_runner.py", "exec")
    steps = (
        '"SOURCE_PARSE", "SOURCE_PARSE_FAILED"',
        '"T0", "T0_IDENTITY_MISMATCH"',
        '"ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_COUNT_MISMATCH"',
        '"ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_HASH_MISMATCH"',
        '"BLOCK_IDENTITY", block_name.upper() + "_IDENTITY_MISMATCH"',
        'build_v8_partition_recovery_manifest(',
        '_validate_accepted_manifest(manifest)',
        'write_v8_partition_recovery_manifest_once(manifest,',
        '"SUCCESSFUL_PUBLICATION"',
    )
    offsets = [runner.index(step) for step in steps]
    assert offsets == sorted(offsets)
    assert 'except Exception:\n    emit_block("RECOVERY_PIPELINE", "POST_NETWORK_UNEXPECTED_FAILURE")' in runner
    assert '"sealed_identity_values_included": False' in runner
    assert '"network_requests": 1' in runner
    assert 'print(json.dumps({"schema_version": recovery.SCHEMA_VERSION, "status": "ACCEPTED"' in runner
    assert '"stage": "SUCCESSFUL_PUBLICATION", "reason": "RECOVERY_PUBLISHED"' in runner
    assert 'recovery.SCHEMA_VERSION' in runner
    assert "write_v8_partition_recovery_manifest_once(manifest" in runner


def test_synthetic_post_network_runner_reaches_successful_publication(tmp_path):
    """Execute the embedded runner with synthetic pins and its real safe-report gate."""
    frame, _rows, blocks, pins, _parser, _calls = _fixture()
    source = _script()
    runner_start = source.index("$runner = @'\n") + len("$runner = @'\n")
    runner_end = source.index("\n'@", runner_start)
    runner = source[runner_start:runner_end]

    raw = b"repository-safe synthetic publication fixture"
    payload_path = tmp_path / "synthetic-source.bin"
    frame_path = tmp_path / "synthetic-frame.pkl"
    publication_root = Path(tempfile.mkdtemp(prefix="v8-synthetic-publication-"))
    atexit.register(shutil.rmtree, publication_root, ignore_errors=True)
    artifact_path = publication_root / "published-recovery.json"
    payload_path.write_bytes(raw)
    frame.to_pickle(frame_path)

    # This prelude only substitutes the data parser and frozen production pins
    # at their module boundary. The extracted post-network runner remains intact.
    prelude = f'''import os, sys
sys.path.insert(0, os.getcwd())
from src import v8_partition_recovery as recovery
from src import v8_partition as historical
import pandas as pd
pd.read_excel = lambda *_args, **_kwargs: pd.read_pickle(os.environ["SYNTHETIC_FRAME_PATH"])
recovery.EXPECTED_ELIGIBLE_COUNT = {pins.eligible_count}
recovery.EXPECTED_ELIGIBLE_SHA256 = {pins.eligible_sha256!r}
recovery.EXPECTED_BLOCK_SHA256 = {dict(pins.block_sha256)!r}
synthetic_pins = recovery._TrustPins(
    eligible_count={pins.eligible_count},
    eligible_sha256={pins.eligible_sha256!r},
    t0_sha256={pins.t0_sha256!r},
    block_sha256={dict(pins.block_sha256)!r})
def synthetic_build(**kwargs):
    return recovery._build_recovery_manifest(**kwargs, _trust_pins=synthetic_pins)
recovery.build_v8_partition_recovery_manifest = synthetic_build
'''
    environment = os.environ.copy()
    environment.update({
        "V8_RECOVERY_TRANSIENT_PAYLOAD": str(payload_path),
        "V8_RECOVERY_TRANSIENT_ARTIFACT": str(artifact_path),
        "V8_RECOVERY_TRANSIENT_SHA256": hashlib.sha256(raw).hexdigest(),
        "SYNTHETIC_FRAME_PATH": str(frame_path),
    })
    execution = subprocess.run(
        [sys.executable, "-I", "-B", "-c", prelude + runner],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert execution.returncode == 0, execution.stderr
    assert len(execution.stdout.splitlines()) == 1
    safe_report = json.loads(execution.stdout)
    assert safe_report == {
        "schema_version": "V8_PARTITION_RECOVERY_MANIFEST_V1",
        "status": "ACCEPTED",
        "stage": "SUCCESSFUL_PUBLICATION",
        "reason": "RECOVERY_PUBLISHED",
        "network_requests": 1,
        "sealed_identity_values_included": False,
    }

    manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version", "original_trusted_manifest_sha256",
        "historical_partition_implementation_commit", "original_source_fingerprint",
        "recovery_source_fingerprint", "v4_provenance_fingerprint",
        "eligible_ticker_count", "eligible_ticker_list_sha256", "t0_reproduction_status",
        "t0_ticker_list_sha256", "trusted_block_ticker_list_sha256", "block_sizes",
        "block_assignments", "recovery_implementation_provenance", "recovery_timestamp_utc",
        "original_manifest_byte_exact_recovered", "original_partition_block_identity_recovered",
        "manifest_sha256",
    }
    assert manifest["schema_version"] == "V8_PARTITION_RECOVERY_MANIFEST_V1"
    assert manifest["schema_version"] != "V8_PARTITION_MANIFEST_V3"
    assert manifest["original_manifest_byte_exact_recovered"] is False
    assert manifest["original_partition_block_identity_recovered"] is True
    assert manifest["eligible_ticker_count"] == pins.eligible_count
    assert manifest["eligible_ticker_list_sha256"] == pins.eligible_sha256
    assert manifest["trusted_block_ticker_list_sha256"] == dict(pins.block_sha256)
    assert manifest["block_assignments"] == blocks
    assert len(list(publication_root.iterdir())) == 1
    assert "TICKER" not in execution.stdout and "ticker" not in execution.stdout
    assert str(artifact_path) not in execution.stdout

    powershell = _powershell_executable()
    assert powershell is not None
    ps_start = source.index("        $postNetworkStage = 'SAFE_REPORT_VALIDATION'")
    ps_end = source.index("        $terminalExitCode = 0", ps_start) + len("        $terminalExitCode = 0")
    validation = source[ps_start:ps_end]
    validation = validation.replace(
        "        $runnerOutput = & $pythonExe -I -B -c $runner 2>$null\n"
        "        $runnerExit = $LASTEXITCODE\n", "")
    encoded = base64.b64encode((
        "$ErrorActionPreference = 'Stop'\n"
        f"$runnerOutput = @('{json.dumps(safe_report, separators=(',', ':'))}')\n"
        "$runnerExit = 0\n"
        + validation
        + "\nWrite-Output $terminalReport\n"
    ).encode("utf-16le")).decode("ascii")
    publication = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert publication.returncode == 0, publication.stdout + publication.stderr
    assert publication.stdout.strip() == (
        "RECOVERY_RESULT=PASS NETWORK_BOUNDARY_CROSSED=true JPX_SOURCE_REQUESTS=1 "
        "STAGE=SUCCESSFUL_PUBLICATION REASON=RECOVERY_PUBLISHED "
        "SCHEMA=V8_PARTITION_RECOVERY_MANIFEST_V1 SEALED_IDENTITIES_PUBLICLY_DISCLOSED=false"
    )
    assert "C:\\" not in publication.stdout


def _run_failure_formatter_with_powershell(cases: list[tuple[str, str]], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    source = _script()
    helper_start = source.index("    function Format-PostNetworkFailure(")
    helper_end = source.index("\n\n    function Invoke-OperationParserProbe", helper_start)
    helper = source[helper_start:helper_end]
    ps_cases = ",\n".join(
        "@{{Stage='{}';Reason='{}'}}".format(stage, reason) for stage, reason in cases
    )
    harness = f"""
$ErrorActionPreference = 'Stop'
{helper}
$cases = @(
{ps_cases}
)
foreach ($case in $cases) {{
    Format-PostNetworkFailure $case.Stage $case.Reason 1
}}
"""
    encoded = base64.b64encode(harness.encode("utf-16le")).decode("ascii")
    powershell = _powershell_executable()
    assert powershell is not None
    result = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert list(tmp_path.iterdir()) == []
    return result


@pytest.mark.parametrize(
    ("stage", "reason"),
    [
        ("SOURCE_ACQUISITION", "SOURCE_TRANSPORT_OR_HTTP_FAILED"),
        ("SOURCE_BYTES", "SOURCE_BYTES_VALIDATION_FAILED"),
        ("SOURCE_BYTES_READY", "SOURCE_BYTES_HANDOFF_FAILED"),
        ("SOURCE_PARSE", "SOURCE_PARSE_FAILED"),
        ("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_EMPTY"),
        ("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_DUPLICATE"),
        ("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_CONSTRUCTION_FAILED"),
        ("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_COUNT_MISMATCH"),
        ("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_HASH_MISMATCH"),
        ("T0", "T0_IDENTITY_MISMATCH"),
        ("T0", "T0_REPRODUCTION_FAILED"),
        ("BLOCK_IDENTITY", "BLOCK_IDENTITY_CONSTRUCTION_FAILED"),
        ("BLOCK_IDENTITY", "T1_IDENTITY_MISMATCH"),
        ("BLOCK_IDENTITY", "T2_IDENTITY_MISMATCH"),
        ("BLOCK_IDENTITY", "T3_IDENTITY_MISMATCH"),
        ("BLOCK_IDENTITY", "T_SPARE_IDENTITY_MISMATCH"),
        ("RECOVERY_MANIFEST_CONSTRUCTION", "RECOVERY_MANIFEST_CONSTRUCTION_FAILED"),
        ("RECOVERY_MANIFEST_VALIDATION", "RECOVERY_MANIFEST_VALIDATION_FAILED"),
        ("DESTINATION_PUBLICATION", "DESTINATION_PUBLICATION_FAILED"),
        ("SAFE_REPORT_VALIDATION", "POST_NETWORK_SAFE_REPORT_INVALID"),
        ("RECOVERY_PIPELINE", "POST_NETWORK_UNEXPECTED_FAILURE"),
    ],
)
def test_synthetic_post_network_failures_have_exact_safe_terminal_reason(stage, reason, tmp_path):
    powershell = _powershell_executable()
    if os.name != "nt" or powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    result = _run_failure_formatter_with_powershell([(stage, reason)], tmp_path)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.strip() == (
        f"RECOVERY_RESULT=BLOCK NETWORK_BOUNDARY_CROSSED=true JPX_SOURCE_REQUESTS=1 STAGE={stage} REASON={reason}"
    )
    assert "TICKER" not in output and "ticker" not in output and "C:\\" not in output


def test_unexpected_post_network_failure_keeps_coarse_stage():
    source = _script()
    assert "$Stage = 'RECOVERY_PIPELINE'" in source
    assert "$Reason = 'POST_NETWORK_UNEXPECTED_FAILURE'" in source


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
