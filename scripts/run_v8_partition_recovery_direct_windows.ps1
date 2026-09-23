[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$ExpectedScriptBlob
)

& {
    $ErrorActionPreference = 'Stop'
    $expectedBranch = 'v13-conditional-cross-sectional-short-horizon'
    $recoveryImplementationHead = '13c7e6f30bf7f5be10e0f47f2b67a0410084d15c'
    $scriptRelativePath = 'scripts/run_v8_partition_recovery_direct_windows.ps1'
    $designBlob = 'ec94b32cfd1249e0635e0fcfee4e3c173d1d8e50'
    $recoveryBlob = '795912a8be6c8e79b7ed021014715dbd476e9668'
    $partitionBlob = '659473ee3ad3b8910225f53fe90f35849027d85e'
    $checkerBlob = 'd9403dcd278812373f88aa6882495814b3e0ca11'
    $sourceUrl = 'https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls'
    $artifactName = 'V8_PARTITION_RECOVERY_MANIFEST_V1.json'
    $requestCount = 0
    $networkBoundaryCrossed = $false
    $temporaryDirectory = $null
    $temporaryPayload = $null
    $pythonPayloadPath = 'V8_RECOVERY_TRANSIENT_PAYLOAD'
    $pythonArtifactPath = 'V8_RECOVERY_TRANSIENT_ARTIFACT'
    $pythonPayloadHash = 'V8_RECOVERY_TRANSIENT_SHA256'

    function Invoke-GitReadOnly([string[]]$GitArguments) {
        $result = & git @GitArguments 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'PRE_GATE_GIT_CHECK_FAILED' }
        return ($result -join "`n").Trim()
    }

    try {
        $repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
        $repoRoot = (Resolve-Path -LiteralPath $repoRoot).Path
        Push-Location -LiteralPath $repoRoot
        $gitDirectory = Join-Path $repoRoot '.git'
        if (-not (Test-Path -LiteralPath $gitDirectory -PathType Container)) { throw 'PRE_GATE_NOT_AUTHORITATIVE_CHECKOUT' }
        if ($repoRoot -match '(?i)[\\/]orca[\\/]workspaces[\\/]') { throw 'PRE_GATE_GENERATED_WORKTREE' }
        $top = Invoke-GitReadOnly @('-C', $repoRoot, 'rev-parse', '--show-toplevel')
        if ([System.IO.Path]::GetFullPath($top) -ne $repoRoot) { throw 'PRE_GATE_REPOSITORY_ROOT_MISMATCH' }
        $branch = Invoke-GitReadOnly @('-C', $repoRoot, 'branch', '--show-current')
        if ($branch -cne $expectedBranch) { throw 'PRE_GATE_WRONG_BRANCH' }
        $origin = Invoke-GitReadOnly @('-C', $repoRoot, 'remote', 'get-url', 'origin')
        if ($origin -notin @('https://github.com/ta1k1-arakawa/stock-analyzer.git', 'git@github.com:ta1k1-arakawa/stock-analyzer.git')) { throw 'PRE_GATE_WRONG_ORIGIN' }
        $remoteLine = Invoke-GitReadOnly @('-C', $repoRoot, 'ls-remote', '--exit-code', 'origin', "refs/heads/$expectedBranch")
        $remoteHead = ($remoteLine -split '\s+')[0]
        $localHead = Invoke-GitReadOnly @('-C', $repoRoot, 'rev-parse', 'HEAD')
        if ($remoteHead -cne $localHead) { throw 'PRE_GATE_LOCAL_REMOTE_HEAD_MISMATCH' }
        if ((Invoke-GitReadOnly @('-C', $repoRoot, 'status', '--porcelain')) -ne '') { throw 'PRE_GATE_DIRTY_WORKTREE' }
        & git -C $repoRoot merge-base --is-ancestor $recoveryImplementationHead $localHead 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'PRE_GATE_RECOVERY_IMPLEMENTATION_ANCESTRY_MISMATCH' }

        $committedScriptBlob = Invoke-GitReadOnly @('-C', $repoRoot, 'rev-parse', "${localHead}:$scriptRelativePath")
        if ($committedScriptBlob -cne $ExpectedScriptBlob) { throw 'PRE_GATE_REVIEWED_SCRIPT_BLOB_MISMATCH' }
        $workingScriptBlob = Invoke-GitReadOnly @('-C', $repoRoot, 'hash-object', '--path', $scriptRelativePath, $scriptRelativePath)
        if ($workingScriptBlob -cne $ExpectedScriptBlob) { throw 'PRE_GATE_REVIEWED_SCRIPT_WORKTREE_BLOB_MISMATCH' }

        if ((Invoke-GitReadOnly @('-C', $repoRoot, 'rev-parse', "${localHead}:V13_V8_PARTITION_RECONSTRUCTION_DESIGN.md")) -cne $designBlob -or
            (Invoke-GitReadOnly @('-C', $repoRoot, 'rev-parse', "${localHead}:src/v8_partition_recovery.py")) -cne $recoveryBlob -or
            (Invoke-GitReadOnly @('-C', $repoRoot, 'rev-parse', "${localHead}:src/v8_partition.py")) -cne $partitionBlob -or
            (Invoke-GitReadOnly @('-C', $repoRoot, 'rev-parse', "${localHead}:scripts/check_current_protected_environment.py")) -cne $checkerBlob) {
            throw 'PRE_GATE_REVIEWED_BLOB_MISMATCH'
        }
        foreach ($relativePath in @('V13_V8_PARTITION_RECONSTRUCTION_DESIGN.md', 'src/v8_partition_recovery.py', 'src/v8_partition.py', 'scripts/check_current_protected_environment.py')) {
            $worktreeBlob = (Invoke-GitReadOnly @('-C', $repoRoot, 'hash-object', $relativePath))
            $expectedBlob = switch ($relativePath) {
                'V13_V8_PARTITION_RECONSTRUCTION_DESIGN.md' { $designBlob }
                'src/v8_partition_recovery.py' { $recoveryBlob }
                'src/v8_partition.py' { $partitionBlob }
                'scripts/check_current_protected_environment.py' { $checkerBlob }
            }
            if ($worktreeBlob -cne $expectedBlob) { throw 'PRE_GATE_REVIEWED_WORKTREE_FILE_MISMATCH' }
        }

        $localAppData = $env:LOCALAPPDATA
        if ([string]::IsNullOrWhiteSpace($localAppData) -or $localAppData -notmatch '^(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\[^\\]+)') { throw 'PRE_GATE_LOCALAPPDATA_INVALID' }
        $localAppData = (Resolve-Path -LiteralPath $localAppData).Path
        $artifactRoot = [System.IO.Path]::GetFullPath((Join-Path $localAppData 'stock-analyzer\private\v8-recovery'))
        $repoPrefix = $repoRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        $existingAncestor = $artifactRoot
        while (-not (Test-Path -LiteralPath $existingAncestor)) {
            $parentInfo = [System.IO.Directory]::GetParent($existingAncestor)
            if ($null -eq $parentInfo) { throw 'PRE_GATE_ARTIFACT_ROOT_INVALID' }
            $existingAncestor = $parentInfo.FullName
        }
        $resolvedAncestor = (Resolve-Path -LiteralPath $existingAncestor).Path
        $unresolvedSuffix = $artifactRoot.Substring($existingAncestor.Length).TrimStart('\', '/')
        if ($unresolvedSuffix) { $artifactRoot = [System.IO.Path]::GetFullPath((Join-Path $resolvedAncestor $unresolvedSuffix)) }
        else { $artifactRoot = $resolvedAncestor }
        if ($artifactRoot.Equals($repoRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
            $artifactRoot.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'PRE_GATE_ARTIFACT_ROOT_INSIDE_REPOSITORY' }
        $artifactPath = Join-Path $artifactRoot $artifactName
        if (Test-Path -LiteralPath $artifactPath) { throw 'PRE_GATE_ARTIFACT_ALREADY_EXISTS' }

        $pythonExe = Join-Path $repoRoot '.venv-real-execution\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw 'PRE_GATE_CANONICAL_PYTHON_MISSING' }
        & $pythonExe scripts/check_current_protected_environment.py *> $null
        if ($LASTEXITCODE -ne 0) { throw 'PRE_GATE_PROTECTED_ENVIRONMENT_BLOCK' }

        $parserProbe = @'
import io, sys
from pathlib import Path
import pandas as pd
import xlrd
from src import v8_partition_recovery, v8_partition
fixture = Path("tests/fixtures/synthetic_jpx_source_snapshot.xls").read_bytes()
frame = pd.read_excel(io.BytesIO(fixture), engine="xlrd")
rows, _ = v8_partition.parse_eligible_universe(frame)
assert rows and len(rows) == 5
assert v8_partition_recovery.SCHEMA_VERSION == "V8_PARTITION_RECOVERY_MANIFEST_V1"
print("OPERATION_PARSER_PROBE_PASS")
'@
        $probeOutput = & $pythonExe -c $parserProbe 2>$null
        if ($LASTEXITCODE -ne 0 -or ($probeOutput -join '') -cne 'OPERATION_PARSER_PROBE_PASS') { throw 'PRE_GATE_OPERATION_PARSER_BLOCK' }

        # The root is created only after repository and environment preflight, before the request.
        [System.IO.Directory]::CreateDirectory($artifactRoot) | Out-Null
        if (Test-Path -LiteralPath $artifactPath) { throw 'PRE_GATE_ARTIFACT_ALREADY_EXISTS' }
        $temporaryDirectory = Join-Path $localAppData ('Temp\v8-recovery-' + [guid]::NewGuid().ToString('N'))
        [System.IO.Directory]::CreateDirectory($temporaryDirectory) | Out-Null
        $temporaryPayload = Join-Path $temporaryDirectory 'source.bin'

        # One non-redirecting HTTP request. A transport or semantic failure is terminal.
        $networkBoundaryCrossed = $true
        $requestCount = 1
        $request = [System.Net.HttpWebRequest]::Create($sourceUrl)
        $request.Method = 'GET'
        $request.AllowAutoRedirect = $false
        $request.Timeout = 120000
        $request.ReadWriteTimeout = 120000
        $request.UserAgent = 'stock-analyzer-v8-recovery/1.0'
        $response = $request.GetResponse()
        try {
            if ([int]$response.StatusCode -ne 200) { throw 'POST_GATE_SOURCE_HTTP_STATUS_BLOCK' }
            $sourceStream = $response.GetResponseStream()
            $payloadStream = [System.IO.MemoryStream]::new()
            try {
                $sourceStream.CopyTo($payloadStream)
                $payloadBytes = $payloadStream.ToArray()
            }
            finally {
                $sourceStream.Dispose()
                $payloadStream.Dispose()
            }
            if ($payloadBytes.Length -le 0) { throw 'POST_GATE_EMPTY_SOURCE_BLOCK' }
            $fileStream = [System.IO.File]::Open($temporaryPayload, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            try { $fileStream.Write($payloadBytes, 0, $payloadBytes.Length); $fileStream.Flush($true) }
            finally { $fileStream.Dispose() }
            $sha256 = [System.Security.Cryptography.SHA256]::Create()
            try {
                $payloadDigestBytes = $sha256.ComputeHash($payloadBytes)
                $payloadDigest = [System.BitConverter]::ToString($payloadDigestBytes).Replace('-', '').ToLowerInvariant()
            }
            finally { $sha256.Dispose() }
            [Array]::Clear($payloadBytes, 0, $payloadBytes.Length)
            $payloadBytes = $null
        }
        finally { $response.Dispose() }

        [System.Environment]::SetEnvironmentVariable($pythonPayloadPath, $temporaryPayload, 'Process')
        [System.Environment]::SetEnvironmentVariable($pythonArtifactPath, $artifactPath, 'Process')
        [System.Environment]::SetEnvironmentVariable($pythonPayloadHash, $payloadDigest, 'Process')
        $runner = @'
import io, json, os, sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, os.getcwd())
from src import v8_partition_recovery as recovery
try:
    raw = Path(os.environ["V8_RECOVERY_TRANSIENT_PAYLOAD"]).read_bytes()
    if not raw:
        raise recovery.V8PartitionRecoveryBlocked("RECOVERY_SOURCE_BYTES_INVALID")
    if hashlib.sha256(raw).hexdigest() != os.environ["V8_RECOVERY_TRANSIENT_SHA256"]:
        raise recovery.V8PartitionRecoveryBlocked("RECOVERY_CONTENT_LOCK_MISMATCH")
    import pandas as pd
    def parse_source_table(payload):
        return pd.read_excel(io.BytesIO(payload), engine="xlrd")
    head = __import__("subprocess").run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    recovery.recover_and_publish_v8_partition_once(
        raw_source_bytes=raw,
        parse_source_table=parse_source_table,
        v4_manifest_path="V4_UNIVERSE_MANIFEST.json",
        v4_universe_csv_path="V4_UNIVERSE.csv",
        recovery_source_url="https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls",
        recovery_source_acquisition_utc=datetime.now(timezone.utc),
        recovery_timestamp_utc=datetime.now(timezone.utc),
        recovery_implementation_commit=head,
        output_path=os.environ["V8_RECOVERY_TRANSIENT_ARTIFACT"],
        repository_root=os.getcwd(),
    )
    print(json.dumps(recovery.safe_recovery_status(accepted=True, eligible_ticker_count=recovery.EXPECTED_ELIGIBLE_COUNT, eligible_ticker_list_sha256=recovery.EXPECTED_ELIGIBLE_SHA256, block_hashes=recovery.EXPECTED_BLOCK_SHA256, network_requests=1), sort_keys=True))
except recovery.V8PartitionRecoveryBlocked as error:
    print(json.dumps({"schema_version": recovery.SCHEMA_VERSION, "status": "BLOCKED", "reason": error.reason, "network_requests": 1, "sealed_identity_values_included": False}, sort_keys=True))
    sys.exit(2)
except Exception:
    print(json.dumps({"schema_version": recovery.SCHEMA_VERSION, "status": "BLOCKED", "reason": "RECOVERY_EXECUTION_BLOCKED", "network_requests": 1, "sealed_identity_values_included": False}, sort_keys=True))
    sys.exit(3)
finally:
    raw = None
'@
        $runnerOutput = & $pythonExe -I -B -c $runner 2>$null
        $runnerExit = $LASTEXITCODE
        if ($runnerOutput.Count -ne 1) { throw 'POST_GATE_SAFE_REPORT_INVALID' }
        $safeReport = $runnerOutput[0] | ConvertFrom-Json
        if ($safeReport.network_requests -ne 1 -or $safeReport.sealed_identity_values_included -ne $false) { throw 'POST_GATE_SAFE_REPORT_INVALID' }
        if ($runnerExit -ne 0 -or $safeReport.status -ne 'ACCEPTED') {
            throw ('POST_GATE_' + [string]$safeReport.reason)
        }
        $terminalReport = 'RECOVERY_RESULT=PASS NETWORK_BOUNDARY_CROSSED=true JPX_SOURCE_REQUESTS=1 SCHEMA=V8_PARTITION_RECOVERY_MANIFEST_V1 SEALED_IDENTITIES_PUBLICLY_DISCLOSED=false'
        $terminalExitCode = 0
    }
    catch {
        $safeError = [string]$_.Exception.Message
        if ($safeError -notmatch '^(PRE_GATE_[A-Z0-9_]+|POST_GATE_[A-Z0-9_]+)$') { $safeError = 'EXECUTION_BLOCKED' }
        if (-not $terminalReport) {
            if ($networkBoundaryCrossed) {
                $terminalReport = "RECOVERY_RESULT=BLOCK NETWORK_BOUNDARY_CROSSED=true JPX_SOURCE_REQUESTS=$requestCount REASON=$safeError"
            }
            else {
                $terminalReport = "RECOVERY_RESULT=NOT_EXECUTED NETWORK_BOUNDARY_CROSSED=false JPX_SOURCE_REQUESTS=0 REASON=$safeError"
            }
            $terminalExitCode = 1
        }
    }
    finally {
        [System.Environment]::SetEnvironmentVariable($pythonPayloadPath, $null, 'Process')
        [System.Environment]::SetEnvironmentVariable($pythonArtifactPath, $null, 'Process')
        [System.Environment]::SetEnvironmentVariable($pythonPayloadHash, $null, 'Process')
        if ($temporaryPayload -and (Test-Path -LiteralPath $temporaryPayload)) { try { [System.IO.File]::Delete($temporaryPayload) } catch { } }
        if ($temporaryDirectory -and (Test-Path -LiteralPath $temporaryDirectory)) { try { [System.IO.Directory]::Delete($temporaryDirectory, $true) } catch { } }
        $request = $null
        $response = $null
        if ($payloadBytes) { [Array]::Clear($payloadBytes, 0, $payloadBytes.Length) }
        if ($payloadDigestBytes) { [Array]::Clear($payloadDigestBytes, 0, $payloadDigestBytes.Length) }
        $payloadBytes = $null
        $payloadDigestBytes = $null
        $sourceStream = $null
        $payloadStream = $null
        $fileStream = $null
        $payloadDigest = $null
        $artifactPath = $null
        $artifactRoot = $null
        $localAppData = $null
        $safeError = $null
        $safeReport = $null
        $runnerOutput = $null
        $temporaryPayload = $null
        $temporaryDirectory = $null
        Pop-Location -ErrorAction SilentlyContinue
    }
    Write-Output $terminalReport
    if ($terminalExitCode -ne 0) { exit $terminalExitCode }
}
