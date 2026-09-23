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
    $postNetworkStage = 'NOT_STARTED'
    $postNetworkReason = $null
    $temporaryDirectory = $null
    $temporaryPayload = $null
    $temporaryProbePath = $null
    $pythonPayloadPath = 'V8_RECOVERY_TRANSIENT_PAYLOAD'
    $pythonArtifactPath = 'V8_RECOVERY_TRANSIENT_ARTIFACT'
    $pythonPayloadHash = 'V8_RECOVERY_TRANSIENT_SHA256'

    function Invoke-GitReadOnly([string[]]$GitArguments) {
        $result = & git @GitArguments 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'PRE_GATE_GIT_CHECK_FAILED' }
        return ($result -join "`n").Trim()
    }

    function Get-SourceResponseOrThrow([System.Net.HttpWebRequest]$Request) {
        try {
            $response = $Request.GetResponse()
        }
        catch [System.Net.WebException] {
            $webException = $_.Exception
            if ($webException.Response -is [System.Net.HttpWebResponse]) {
                $statusCode = [int]$webException.Response.StatusCode
                $webException.Response.Dispose()
                if ($statusCode -ge 300 -and $statusCode -lt 400) { throw "SOURCE_REDIRECT_HTTP_$statusCode" }
                throw "SOURCE_HTTP_STATUS_$statusCode"
            }
            if ($webException.Status -eq [System.Net.WebExceptionStatus]::Timeout) { throw 'SOURCE_TIMEOUT' }
            throw 'SOURCE_TRANSPORT_FAILED'
        }
        catch {
            throw 'SOURCE_ACQUISITION_UNEXPECTED_FAILURE'
        }

        $statusCode = [int]$response.StatusCode
        if ($statusCode -ne 200) {
            $response.Dispose()
            if ($statusCode -ge 300 -and $statusCode -lt 400) { throw "SOURCE_REDIRECT_HTTP_$statusCode" }
            throw "SOURCE_HTTP_STATUS_$statusCode"
        }
        return $response
    }

    function Format-PostNetworkFailure([string]$Stage, [string]$Reason, [int]$Requests) {
        $knownStages = @('SOURCE_ACQUISITION', 'SOURCE_BYTES', 'SOURCE_PARSE', 'ELIGIBLE_UNIVERSE', 'T0',
            'BLOCK_IDENTITY', 'RECOVERY_MANIFEST_CONSTRUCTION', 'RECOVERY_MANIFEST_VALIDATION',
            'DESTINATION_PUBLICATION', 'RECOVERY_PIPELINE', 'SOURCE_BYTES_READY', 'SAFE_REPORT_VALIDATION',
            'REQUEST_INITIATED')
        $knownReasons = @('SOURCE_TIMEOUT', 'SOURCE_TRANSPORT_FAILED', 'SOURCE_ACQUISITION_UNEXPECTED_FAILURE', 'SOURCE_BYTES_HANDOFF_FAILED',
            'SOURCE_BYTES_VALIDATION_FAILED', 'SOURCE_PARSE_FAILED', 'ELIGIBLE_UNIVERSE_EMPTY',
            'ELIGIBLE_UNIVERSE_DUPLICATE', 'ELIGIBLE_UNIVERSE_CONSTRUCTION_FAILED',
            'ELIGIBLE_UNIVERSE_COUNT_MISMATCH', 'ELIGIBLE_UNIVERSE_HASH_MISMATCH',
            'T0_IDENTITY_MISMATCH', 'T0_REPRODUCTION_FAILED', 'BLOCK_IDENTITY_CONSTRUCTION_FAILED',
            'T1_IDENTITY_MISMATCH', 'T2_IDENTITY_MISMATCH', 'T3_IDENTITY_MISMATCH',
            'T_SPARE_IDENTITY_MISMATCH', 'RECOVERY_MANIFEST_CONSTRUCTION_FAILED',
            'RECOVERY_MANIFEST_VALIDATION_FAILED', 'DESTINATION_PUBLICATION_FAILED',
            'POST_NETWORK_UNEXPECTED_FAILURE', 'POST_NETWORK_SAFE_REPORT_INVALID')
        $statusReason = $Stage -eq 'SOURCE_ACQUISITION' -and $Reason -match '^SOURCE_(REDIRECT_HTTP|HTTP_STATUS)_\d{3}$'
        if ($Stage -notin $knownStages -or (-not $statusReason -and $Reason -notin $knownReasons) -or $Requests -ne 1) {
            $Stage = 'RECOVERY_PIPELINE'
            $Reason = 'POST_NETWORK_UNEXPECTED_FAILURE'
        }
        return "RECOVERY_RESULT=BLOCK NETWORK_BOUNDARY_CROSSED=true JPX_SOURCE_REQUESTS=1 STAGE=$Stage REASON=$Reason"
    }

    function Invoke-OperationParserProbe([string]$PythonExe, [string]$ProbePath, [string]$ProbeText) {
        try {
            [System.IO.File]::WriteAllText($ProbePath, $ProbeText, [System.Text.UTF8Encoding]::new($false))
            $output = & $PythonExe -I -B $ProbePath 2>$null
            if ($LASTEXITCODE -ne 0 -or ($output -join '') -cne 'OPERATION_PARSER_PROBE_PASS') {
                throw 'PRE_GATE_OPERATION_PARSER_BLOCK'
            }
            return ($output -join '')
        }
        catch {
            if ([string]$_.Exception.Message -match '^PRE_GATE_[A-Z0-9_]+$') { throw }
            throw 'PRE_GATE_OPERATION_PARSER_BLOCK'
        }
        finally {
            if (Test-Path -LiteralPath $ProbePath) { try { [System.IO.File]::Delete($ProbePath) } catch { } }
        }
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

        $temporaryDirectory = Join-Path $localAppData ('Temp\v8-recovery-' + [guid]::NewGuid().ToString('N'))
        [System.IO.Directory]::CreateDirectory($temporaryDirectory) | Out-Null
        $temporaryProbePath = Join-Path $temporaryDirectory 'operation-parser-probe.py'

        $parserProbe = @'
import io, sys
from pathlib import Path
import pandas as pd
import xlrd
sys.path.insert(0, str(Path.cwd()))
from src import v8_partition_recovery, v8_partition
fixture = Path("tests/fixtures/synthetic_jpx_source_snapshot.xls").read_bytes()
frame = pd.read_excel(io.BytesIO(fixture), engine="xlrd")
rows, _ = v8_partition.parse_eligible_universe(frame)
assert rows and len(rows) == 5
assert v8_partition_recovery.SCHEMA_VERSION == "V8_PARTITION_RECOVERY_MANIFEST_V1"
print("OPERATION_PARSER_PROBE_PASS")
'@
        $probeOutput = Invoke-OperationParserProbe $pythonExe $temporaryProbePath $parserProbe

        # The root is created only after repository and environment preflight, before the request.
        [System.IO.Directory]::CreateDirectory($artifactRoot) | Out-Null
        if (Test-Path -LiteralPath $artifactPath) { throw 'PRE_GATE_ARTIFACT_ALREADY_EXISTS' }
        $temporaryPayload = Join-Path $temporaryDirectory 'source.bin'

        # One non-redirecting HTTP request. A transport or semantic failure is terminal.
        $networkBoundaryCrossed = $true
        $requestCount = 1
        $postNetworkStage = 'SOURCE_ACQUISITION'
        $request = [System.Net.HttpWebRequest]::Create($sourceUrl)
        $request.Method = 'GET'
        $request.AllowAutoRedirect = $false
        $request.Timeout = 120000
        $request.ReadWriteTimeout = 120000
        $request.UserAgent = 'stock-analyzer-v8-recovery/1.0'
        try { $response = Get-SourceResponseOrThrow $request }
        catch {
            $acquisitionReason = [string]$_.Exception.Message
            if ($acquisitionReason -match '^SOURCE_(REDIRECT_HTTP|HTTP_STATUS)_\d{3}$' -or
                $acquisitionReason -in @('SOURCE_TIMEOUT', 'SOURCE_TRANSPORT_FAILED', 'SOURCE_ACQUISITION_UNEXPECTED_FAILURE')) {
                $postNetworkReason = $acquisitionReason
            }
            else { $postNetworkReason = 'SOURCE_ACQUISITION_UNEXPECTED_FAILURE' }
            throw 'POST_GATE_SOURCE_ACQUISITION_BLOCKED'
        }
        try {
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
            $postNetworkStage = 'SOURCE_BYTES_READY'
        }
        finally { $response.Dispose() }

        [System.Environment]::SetEnvironmentVariable($pythonPayloadPath, $temporaryPayload, 'Process')
        [System.Environment]::SetEnvironmentVariable($pythonArtifactPath, $artifactPath, 'Process')
        [System.Environment]::SetEnvironmentVariable($pythonPayloadHash, $payloadDigest, 'Process')
        $postNetworkStage = 'RECOVERY_PIPELINE'
        $runner = @'
import hashlib, io, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, os.getcwd())
from src import v8_partition_recovery as recovery
from src import v8_partition as historical

def emit_block(stage, reason):
    print(json.dumps({"schema_version": recovery.SCHEMA_VERSION, "status": "BLOCKED",
        "stage": stage, "reason": reason, "network_requests": 1,
        "sealed_identity_values_included": False}, sort_keys=True))
    sys.exit(2)

raw = None
try:
    raw = Path(os.environ["V8_RECOVERY_TRANSIENT_PAYLOAD"]).read_bytes()
    if not raw or hashlib.sha256(raw).hexdigest() != os.environ["V8_RECOVERY_TRANSIENT_SHA256"]:
        emit_block("SOURCE_BYTES", "SOURCE_BYTES_VALIDATION_FAILED")
    try:
        import pandas as pd
        frame = pd.read_excel(io.BytesIO(raw), engine="xlrd")
    except Exception:
        emit_block("SOURCE_PARSE", "SOURCE_PARSE_FAILED")

    try:
        eligible_rows, _excluded_counts = historical.parse_eligible_universe(frame)
        if not eligible_rows:
            emit_block("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_EMPTY")
        ordered_codes = historical.canonical_order([row["code"] for row in eligible_rows])
        rows_by_code = {row["code"]: row for row in eligible_rows}
        if len(rows_by_code) != len(eligible_rows):
            emit_block("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_DUPLICATE")
        ordered_rows = [rows_by_code[code] for code in ordered_codes]
        eligible_hash = historical.ticker_list_sha256(ordered_codes)
    except SystemExit:
        raise
    except Exception:
        emit_block("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_CONSTRUCTION_FAILED")
    try:
        provenance = historical.load_v4_provenance("V4_UNIVERSE_MANIFEST.json")
        historical.load_v4_universe_csv_bytes("V4_UNIVERSE.csv")
        t0 = historical.verify_t0_reproduction(ordered_rows, provenance)
        if historical.ticker_list_sha256(t0) != recovery.EXPECTED_T0_SHA256:
            emit_block("T0", "T0_IDENTITY_MISMATCH")
    except historical.V8PartitionBlocked as error:
        if error.reason == "V8_T0_REPRODUCTION_MISMATCH":
            emit_block("T0", "T0_IDENTITY_MISMATCH")
        emit_block("T0", "T0_REPRODUCTION_FAILED")
    except SystemExit:
        raise
    except Exception:
        emit_block("T0", "T0_REPRODUCTION_FAILED")

    if len(ordered_codes) != recovery.EXPECTED_ELIGIBLE_COUNT:
        emit_block("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_COUNT_MISMATCH")
    if eligible_hash != recovery.EXPECTED_ELIGIBLE_SHA256:
        emit_block("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_HASH_MISMATCH")

    try:
        blocks = historical.allocate_fresh_blocks(ordered_codes, t0)
        for block_name in ("T1", "T2", "T3", "T_spare"):
            if historical.ticker_list_sha256(blocks[block_name]) != recovery.EXPECTED_BLOCK_SHA256[block_name]:
                emit_block("BLOCK_IDENTITY", block_name.upper() + "_IDENTITY_MISMATCH")
    except SystemExit:
        raise
    except Exception:
        emit_block("BLOCK_IDENTITY", "BLOCK_IDENTITY_CONSTRUCTION_FAILED")

    def parse_source_table(_payload):
        return frame
    head = __import__("subprocess").run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    try:
        manifest = recovery.build_v8_partition_recovery_manifest(
            raw_source_bytes=raw, parse_source_table=parse_source_table,
            v4_manifest_path="V4_UNIVERSE_MANIFEST.json", v4_universe_csv_path="V4_UNIVERSE.csv",
            recovery_source_url="https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls",
            recovery_source_acquisition_utc=datetime.now(timezone.utc), recovery_timestamp_utc=datetime.now(timezone.utc),
            recovery_implementation_commit=head)
    except recovery.V8PartitionRecoveryBlocked as error:
        reason = error.reason
        if reason == "RECOVERY_T0_TRUST_PIN_MISMATCH":
            emit_block("T0", "T0_IDENTITY_MISMATCH")
        if reason == "RECOVERY_ELIGIBLE_COUNT_MISMATCH":
            emit_block("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_COUNT_MISMATCH")
        if reason == "RECOVERY_ELIGIBLE_UNIVERSE_HASH_MISMATCH":
            emit_block("ELIGIBLE_UNIVERSE", "ELIGIBLE_UNIVERSE_HASH_MISMATCH")
        for block_name in ("T1", "T2", "T3", "T_SPARE"):
            if reason == "RECOVERY_" + block_name + "_HASH_MISMATCH":
                emit_block("BLOCK_IDENTITY", block_name + "_IDENTITY_MISMATCH")
        emit_block("RECOVERY_MANIFEST_CONSTRUCTION", "RECOVERY_MANIFEST_CONSTRUCTION_FAILED")
    except Exception:
        emit_block("RECOVERY_MANIFEST_CONSTRUCTION", "RECOVERY_MANIFEST_CONSTRUCTION_FAILED")

    try:
        recovery._validate_accepted_manifest(manifest)
    except Exception:
        emit_block("RECOVERY_MANIFEST_VALIDATION", "RECOVERY_MANIFEST_VALIDATION_FAILED")
    try:
        recovery.write_v8_partition_recovery_manifest_once(manifest,
            os.environ["V8_RECOVERY_TRANSIENT_ARTIFACT"], os.getcwd())
    except Exception:
        emit_block("DESTINATION_PUBLICATION", "DESTINATION_PUBLICATION_FAILED")
    print(json.dumps({"schema_version": recovery.SCHEMA_VERSION, "status": "ACCEPTED",
        "stage": "SUCCESSFUL_PUBLICATION", "reason": "RECOVERY_PUBLISHED", "network_requests": 1,
        "sealed_identity_values_included": False}, sort_keys=True))
except SystemExit:
    raise
except Exception:
    emit_block("RECOVERY_PIPELINE", "POST_NETWORK_UNEXPECTED_FAILURE")
finally:
    raw = None
'@
        $runnerOutput = & $pythonExe -I -B -c $runner 2>$null
        $runnerExit = $LASTEXITCODE
        $postNetworkStage = 'SAFE_REPORT_VALIDATION'
        if ($runnerOutput.Count -ne 1) { throw 'POST_GATE_SAFE_REPORT_INVALID' }
        $safeReport = $runnerOutput[0] | ConvertFrom-Json
        if ($safeReport.network_requests -ne 1 -or $safeReport.sealed_identity_values_included -ne $false -or
            $safeReport.stage -notmatch '^(SOURCE_BYTES|SOURCE_PARSE|ELIGIBLE_UNIVERSE|T0|BLOCK_IDENTITY|RECOVERY_MANIFEST_CONSTRUCTION|RECOVERY_MANIFEST_VALIDATION|DESTINATION_PUBLICATION|RECOVERY_PIPELINE|SUCCESSFUL_PUBLICATION)$' -or
            $safeReport.reason -notmatch '^[A-Z0-9_]+$') { throw 'POST_GATE_SAFE_REPORT_INVALID' }
        if ($runnerExit -ne 0 -or $safeReport.status -ne 'ACCEPTED') {
            $postNetworkStage = [string]$safeReport.stage
            $postNetworkReason = [string]$safeReport.reason
            throw 'POST_GATE_PIPELINE_BLOCKED'
        }
        if ($safeReport.stage -cne 'SUCCESSFUL_PUBLICATION' -or $safeReport.reason -cne 'RECOVERY_PUBLISHED') { throw 'POST_GATE_SAFE_REPORT_INVALID' }
        $postNetworkStage = [string]$safeReport.stage
        $terminalReport = 'RECOVERY_RESULT=PASS NETWORK_BOUNDARY_CROSSED=true JPX_SOURCE_REQUESTS=1 STAGE=SUCCESSFUL_PUBLICATION REASON=RECOVERY_PUBLISHED SCHEMA=V8_PARTITION_RECOVERY_MANIFEST_V1 SEALED_IDENTITIES_PUBLICLY_DISCLOSED=false'
        $terminalExitCode = 0
    }
    catch {
        $safeError = [string]$_.Exception.Message
        if ($safeError -match '^PRE_GATE_[A-Z0-9_]+$') {
            $safeReason = $safeError
        }
        elseif ($networkBoundaryCrossed) {
            if ($postNetworkReason) { $safeReason = $postNetworkReason }
            elseif ($postNetworkStage -eq 'SOURCE_ACQUISITION') { $safeReason = 'SOURCE_ACQUISITION_UNEXPECTED_FAILURE' }
            elseif ($postNetworkStage -eq 'SOURCE_BYTES_READY') { $safeReason = 'SOURCE_BYTES_HANDOFF_FAILED' }
            elseif ($postNetworkStage -eq 'SAFE_REPORT_VALIDATION') { $safeReason = 'POST_NETWORK_SAFE_REPORT_INVALID' }
            else { $safeReason = 'POST_NETWORK_UNEXPECTED_FAILURE' }
            if (-not $postNetworkStage -or $postNetworkStage -eq 'NOT_STARTED') { $postNetworkStage = 'REQUEST_INITIATED' }
        }
        else { $safeReason = 'PRE_GATE_UNEXPECTED_FAILURE' }
        if (-not $terminalReport) {
            if ($networkBoundaryCrossed) {
                $terminalReport = Format-PostNetworkFailure $postNetworkStage $safeReason $requestCount
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
        if ($temporaryProbePath -and (Test-Path -LiteralPath $temporaryProbePath)) { try { [System.IO.File]::Delete($temporaryProbePath) } catch { } }
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
        $safeReason = $null
        $safeReport = $null
        $runnerOutput = $null
        $temporaryPayload = $null
        $temporaryProbePath = $null
        $temporaryDirectory = $null
        Pop-Location -ErrorAction SilentlyContinue
    }
    Write-Output $terminalReport
    if ($terminalExitCode -ne 0) { exit $terminalExitCode }
}
