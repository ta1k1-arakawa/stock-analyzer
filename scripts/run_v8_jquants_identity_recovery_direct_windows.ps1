[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedHead,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedScriptBlob,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedImplementationBlob,
    [switch]$ExecuteReviewedAcquisition
)

& {
    $ErrorActionPreference = 'Stop'
    $branch = 'v13-conditional-cross-sectional-short-horizon'
    $result = 'JQUANTS_RECOVERY_RESULT=BLOCK NETWORK_BOUNDARY_CROSSED=false JQUANTS_LOGICAL_ACQUISITIONS=0 JQUANTS_HTTP_REQUESTS=0 STAGE=PRE_GATE REASON=UNEXPECTED_FAILURE RAW_CONTENT_LOCK_PUBLISHED=false RECOVERY_ARTIFACT_PUBLISHED=false'
    $reason = 'PRE_GATE_REPOSITORY_BLOCK'
    $repoRoot = $null
    $oldHead = [Environment]::GetEnvironmentVariable('V8_JQUANTS_REVIEWED_HEAD', 'Process')
    $oldBlob = [Environment]::GetEnvironmentVariable('V8_JQUANTS_REVIEWED_BLOB', 'Process')
    $oldScriptBlob = [Environment]::GetEnvironmentVariable('V8_JQUANTS_REVIEWED_SCRIPT_BLOB', 'Process')
    function GitValue([string[]]$Argv) {
        $value = & git @Argv 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'BLOCK' }
        return ($value -join "`n").Trim()
    }
    try {
        $repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
        if ($repoRoot -match '(?i)[\\/]orca[\\/]workspaces[\\/]') { throw 'BLOCK' }
        if (-not (Test-Path -LiteralPath (Join-Path $repoRoot '.git') -PathType Container)) { throw 'BLOCK' }
        if ([IO.Path]::GetFullPath((GitValue @('-C', $repoRoot, 'rev-parse', '--show-toplevel'))) -cne $repoRoot) { throw 'BLOCK' }
        if ((GitValue @('-C', $repoRoot, 'branch', '--show-current')) -cne $branch) { throw 'BLOCK' }
        if ((GitValue @('-C', $repoRoot, 'remote', 'get-url', 'origin')) -notin @('https://github.com/ta1k1-arakawa/stock-analyzer.git', 'git@github.com:ta1k1-arakawa/stock-analyzer.git')) { throw 'BLOCK' }
        if ((GitValue @('-C', $repoRoot, 'rev-parse', 'HEAD')) -cne $ExpectedHead) { throw 'BLOCK' }
        $remote = GitValue @('-C', $repoRoot, 'ls-remote', '--exit-code', 'origin', "refs/heads/$branch")
        if (($remote -split '\s+')[0] -cne $ExpectedHead) { throw 'BLOCK' }
        if ((GitValue @('-C', $repoRoot, 'status', '--porcelain')) -ne '') { throw 'BLOCK' }
        $bindings = @{
            'scripts/run_v8_jquants_identity_recovery_direct_windows.ps1' = $ExpectedScriptBlob
            'src/v8_jquants_identity_recovery.py' = $ExpectedImplementationBlob
        }
        foreach ($path in $bindings.Keys) {
            if ((GitValue @('-C', $repoRoot, 'rev-parse', "${ExpectedHead}:$path")) -cne $bindings[$path]) { $reason = 'PRE_GATE_PROVENANCE_BLOCK'; throw 'BLOCK' }
            if ((GitValue @('-C', $repoRoot, 'hash-object', '--path', $path, $path)) -cne $bindings[$path]) { $reason = 'PRE_GATE_PROVENANCE_BLOCK'; throw 'BLOCK' }
        }
        Push-Location -LiteralPath $repoRoot
        try {
            $reason = 'PRE_GATE_ENVIRONMENT_BLOCK'
            $python = Join-Path $repoRoot '.venv-real-execution\Scripts\python.exe'
            if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'BLOCK' }
            & $python scripts/check_current_protected_environment.py *> $null
            if ($LASTEXITCODE -ne 0) { throw 'BLOCK' }
            $probe = 'from src.v8_jquants_identity_recovery import readiness_probe; import sys; sys.exit(0 if readiness_probe() else 1)'
            & $python -E -B -c $probe *> $null
            if ($LASTEXITCODE -ne 0) { throw 'BLOCK' }
            $reason = 'PRE_GATE_PRIVATE_ROOT_BLOCK'
            $rootProbe = @'
from pathlib import Path
from src.v8_jquants_identity_recovery import private_root, inspect_state_metadata, Block
try:
    state = inspect_state_metadata(private_root(Path.cwd()))
    print('READY' if state in ('absent', 'raw') else 'PRE_GATE_EXISTING_ARTIFACT_BLOCK')
except Block as exc:
    print(exc.reason if exc.reason in ('PRE_GATE_PRIVATE_ROOT_BLOCK', 'PRE_GATE_EXISTING_ARTIFACT_BLOCK') else 'PRE_GATE_EXISTING_ARTIFACT_BLOCK')
except Exception:
    print('PRE_GATE_PRIVATE_ROOT_BLOCK')
'@
            $rootResult = @(& $python -E -B -c $rootProbe 2>$null)
            if ($LASTEXITCODE -ne 0 -or $rootResult.Count -ne 1) { throw 'BLOCK' }
            if ($rootResult[0] -ne 'READY') {
                if ($rootResult[0] -eq 'PRE_GATE_EXISTING_ARTIFACT_BLOCK') { $reason = 'PRE_GATE_EXISTING_ARTIFACT_BLOCK' }
                throw 'BLOCK'
            }
            $reason = 'PRE_GATE_CREDENTIAL_BLOCK'
            if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable('JQUANTS_API_KEY', 'Process'))) { throw 'BLOCK' }
            $reason = 'PRE_GATE_PROVENANCE_BLOCK'
            if (-not $ExecuteReviewedAcquisition) { throw 'BLOCK' }
            [Environment]::SetEnvironmentVariable('V8_JQUANTS_REVIEWED_HEAD', $ExpectedHead, 'Process')
            [Environment]::SetEnvironmentVariable('V8_JQUANTS_REVIEWED_BLOB', $ExpectedImplementationBlob, 'Process')
            [Environment]::SetEnvironmentVariable('V8_JQUANTS_REVIEWED_SCRIPT_BLOB', $ExpectedScriptBlob, 'Process')
            $lines = @(& $python -E -B -m src.v8_jquants_identity_recovery 2>$null)
            if ($lines.Count -ne 1 -or $lines[0] -notmatch '^JQUANTS_RECOVERY_RESULT=(PASS|BLOCK) NETWORK_BOUNDARY_CROSSED=(true|false) JQUANTS_LOGICAL_ACQUISITIONS=[01] JQUANTS_HTTP_REQUESTS=\d+ STAGE=(PRE_GATE|SOURCE_ACQUISITION|RAW_CONTENT_LOCK|OFFLINE_SEMANTICS|RECOVERY_PUBLICATION|COMPLETE) REASON=[A-Z0-9_]+ RAW_CONTENT_LOCK_PUBLISHED=(true|false) RECOVERY_ARTIFACT_PUBLISHED=(true|false)( ELIGIBLE_COUNT=\d+)?( (ELIGIBLE|T0|T1|T2|T3|T_SPARE)_HASH_MATCH=(true|false|unknown))*$') {
                throw 'BLOCK'
            }
            $result = [string]$lines[0]
        }
        finally { Pop-Location }
    }
    catch {
        $result = "JQUANTS_RECOVERY_RESULT=BLOCK NETWORK_BOUNDARY_CROSSED=false JQUANTS_LOGICAL_ACQUISITIONS=0 JQUANTS_HTTP_REQUESTS=0 STAGE=PRE_GATE REASON=$reason RAW_CONTENT_LOCK_PUBLISHED=false RECOVERY_ARTIFACT_PUBLISHED=false"
    }
    finally {
        [Environment]::SetEnvironmentVariable('V8_JQUANTS_REVIEWED_HEAD', $oldHead, 'Process')
        [Environment]::SetEnvironmentVariable('V8_JQUANTS_REVIEWED_BLOB', $oldBlob, 'Process')
        [Environment]::SetEnvironmentVariable('V8_JQUANTS_REVIEWED_SCRIPT_BLOB', $oldScriptBlob, 'Process')
    }
    Write-Output $result
    if ($result -notmatch '^JQUANTS_RECOVERY_RESULT=PASS ') { exit 1 }
}
