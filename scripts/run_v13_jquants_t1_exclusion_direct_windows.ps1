[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedHead,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedScriptBlob,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedResolverBlob,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedHarnessBlob,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$FutureAuthorizationBlob,
    [Parameter(Mandatory=$true)][string]$FutureAuthorizationRelativePath,
    [switch]$ExecuteReviewedPrivateRead
)

& {
    $ErrorActionPreference = 'Stop'
    $reason = 'PRE_GATE_REPOSITORY_BLOCK'
    $harnessCalled = $false
    $report = 'EXECUTION_RESULT=PRE_GATE_STOP FAILURE_CLASS=PRE_GATE_REPOSITORY_BLOCK PRIVATE_BOUNDARY_CROSSED=false AUTHORIZATION_CONSUMED=false SOURCE_OPENS=0 NETWORK_REQUESTS=0 PRICE_PAYLOAD_READS=0 OUTCOME_READS=0 NON_T1_IDENTITIES_RETAINED=false V13_UNIVERSE_SELECTED=false'
    function GitValue([string[]]$Argv) {
        $value = & git @Argv 2>$null
        if ($LASTEXITCODE -ne 0) { throw 'BLOCK' }
        return ($value -join "`n").Trim()
    }
    try {
        $repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
        if ($repo -match '(?i)[\/]orca[\/]workspaces[\/]') { throw 'BLOCK' }
        if (-not (Test-Path -LiteralPath (Join-Path $repo '.git') -PathType Container)) { throw 'BLOCK' }
        if ((GitValue @('-C', $repo, 'branch', '--show-current')) -cne 'v13-conditional-cross-sectional-short-horizon') { throw 'BLOCK' }
        if ((GitValue @('-C', $repo, 'rev-parse', 'HEAD')) -cne $ExpectedHead) { throw 'BLOCK' }
        if ((GitValue @('-C', $repo, 'status', '--porcelain')) -ne '') { throw 'BLOCK' }
        if ((GitValue @('-C', $repo, 'remote', 'get-url', 'origin')) -notin @('https://github.com/ta1k1-arakawa/stock-analyzer.git', 'git@github.com:ta1k1-arakawa/stock-analyzer.git')) { throw 'BLOCK' }
        $remote = GitValue @('-C', $repo, 'ls-remote', '--exit-code', 'origin', 'refs/heads/v13-conditional-cross-sectional-short-horizon')
        if (($remote -split '\s+')[0] -cne $ExpectedHead) { throw 'BLOCK' }

        $reason = 'PRE_GATE_PROVENANCE_BLOCK'
        if ($FutureAuthorizationRelativePath -cne 'V13_JQUANTS_T1_EXCLUSION_POINT_OF_USE_AUTHORIZATION.json') { throw 'BLOCK' }
        $bindings = @{
            'scripts/run_v13_jquants_t1_exclusion_direct_windows.ps1' = $ExpectedScriptBlob
            'scripts/v13_resolve_jquants_t1_exclusion_state.py' = $ExpectedResolverBlob
            'scripts/v13_execute_jquants_t1_exclusion_private_read.py' = $ExpectedHarnessBlob
            'src/v8_jquants_identity_recovery.py' = 'f46ea0c304b0bbd2d230b9850acba9eada9f6908'
            'scripts/check_current_protected_environment.py' = 'd9403dcd278812373f88aa6882495814b3e0ca11'
            'V13_EXPOSURE_PROVENANCE_SUCCESSOR_DECISION.md' = '276feb56f417f9d5b931d594598a1d8591330bfd'
            'V13_EXPOSURE_PROVENANCE_SUCCESSOR_FREEZE_APPROVAL.json' = '175623c1fbf43a18eac298be03f163dab0754233'
            'V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json' = '280baea899a576cbc3b705db838e5988dbed5027'
            'V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md' = '3bfcd695c69f6dac480f8fc99ca4f3916f668e4a'
        }
        $bindings[$FutureAuthorizationRelativePath] = $FutureAuthorizationBlob
        foreach ($path in $bindings.Keys) {
            if ((GitValue @('-C', $repo, 'rev-parse', "${ExpectedHead}:$path")) -cne $bindings[$path]) { throw 'BLOCK' }
            if ((GitValue @('-C', $repo, 'hash-object', '--path', $path, $path)) -cne $bindings[$path]) { throw 'BLOCK' }
        }
        $oldAuth = Get-Content -LiteralPath (Join-Path $repo 'V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json') -Raw | ConvertFrom-Json
        if ($oldAuth.authorization_consumed -ne $false -or $oldAuth.t1_membership_identity_read_authorized -ne $true) { throw 'BLOCK' }
        $approval = Get-Content -LiteralPath (Join-Path $repo 'V13_EXPOSURE_PROVENANCE_SUCCESSOR_FREEZE_APPROVAL.json') -Raw | ConvertFrom-Json
        if ($approval.human_approved -ne $true -or
            $approval.approval_scope -cne 'V13_FULL_RECOVERED_T1_300_EXCLUSION_AMENDMENT_FREEZE_ONLY' -or
            $approval.successor_decision_commit -cne '98db82f100ba923a50bba236444a340d28fdd1fd' -or
            $approval.successor_decision_blob_sha1 -cne '276feb56f417f9d5b931d594598a1d8591330bfd' -or
            $approval.exclusion_disposition -cne 'EXCLUDE_FULL_RECOVERED_T1_BLOCK' -or
            $approval.t1_count -ne 300 -or
            $approval.t1_sha256 -cne '262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d') { throw 'BLOCK' }
        $futureAuth = Get-Content -LiteralPath (Join-Path $repo $FutureAuthorizationRelativePath) -Raw | ConvertFrom-Json
        if ($futureAuth.human_approved -ne $true -or $futureAuth.authorization_consumed -ne $false -or
            $futureAuth.authorization_scope -cne 'V13_JQUANTS_RECOVERED_T1_IDENTITY_ONLY_PRIVATE_READ' -or
            $futureAuth.source_schema -cne 'V8_JQUANTS_IDENTITY_RECOVERY_MANIFEST_V1' -or
            $futureAuth.source_commit -cne '7565ca723c76801d74d8d319d65d280a689b3cfa' -or
            $futureAuth.source_blob -cne 'f46ea0c304b0bbd2d230b9850acba9eada9f6908' -or
            $futureAuth.eligible_count -ne 3110 -or
            $futureAuth.eligible_sha256 -cne '37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405' -or
            $futureAuth.t1_count -ne 300 -or
            $futureAuth.t1_ticker_list_sha256 -cne '262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d') { throw 'BLOCK' }

        $reason = 'PRE_GATE_ENVIRONMENT_BLOCK'
        $python = Join-Path $repo '.venv-real-execution\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'BLOCK' }
        Push-Location -LiteralPath $repo
        try {
            & $python scripts/check_current_protected_environment.py *> $null
            if ($LASTEXITCODE -ne 0) { throw 'BLOCK' }
        }
        finally { Pop-Location }

        $reason = 'PRE_GATE_PRIVATE_TOPOLOGY_BLOCK'
        $localAppData = [Environment]::GetEnvironmentVariable('LOCALAPPDATA', 'Process')
        if ([string]::IsNullOrWhiteSpace($localAppData) -or -not [IO.Path]::IsPathRooted($localAppData)) { throw 'BLOCK' }
        $localAppData = [IO.Path]::GetFullPath($localAppData)
        $repoPrefix = $repo.TrimEnd('\', '/') + [IO.Path]::DirectorySeparatorChar
        if ($localAppData.Equals($repo, [StringComparison]::OrdinalIgnoreCase) -or
            $localAppData.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw 'BLOCK' }
        $privateBase = Join-Path $localAppData 'stock-analyzer\private'
        $sourceRoot = Join-Path $privateBase 'v8-jquants-identity-recovery'
        $outputRoot = Join-Path $privateBase 'v13-t1-exclusion-provenance'
        $source = Join-Path $sourceRoot 'recovery.json'
        $receipt = Join-Path $outputRoot 'consumed-receipt.json'
        $state = Join-Path $outputRoot 't1-exclusion-state.json'
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw 'BLOCK' }
        if ((Test-Path -LiteralPath $receipt) -or (Test-Path -LiteralPath $state)) { throw 'BLOCK' }
        if ((Test-Path -LiteralPath $outputRoot) -and @((Get-ChildItem -LiteralPath $outputRoot -Force)).Count -ne 0) { throw 'BLOCK' }
        foreach ($path in @($localAppData, $privateBase, $sourceRoot, $source, $outputRoot)) {
            $cursor = [IO.Path]::GetFullPath($path)
            while ($null -ne $cursor) {
                if (Test-Path -LiteralPath $cursor) {
                    if ((Get-Item -LiteralPath $cursor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'BLOCK' }
                }
                $parent = [IO.Directory]::GetParent($cursor)
                $cursor = if ($null -eq $parent) { $null } else { $parent.FullName }
            }
        }
        if (-not $ExecuteReviewedPrivateRead) { $reason = 'PRE_GATE_EXECUTION_SWITCH_REQUIRED'; throw 'BLOCK' }
        $report = ''
        $priorGate = [Environment]::GetEnvironmentVariable('V13_JQUANTS_T1_WRAPPER_GATE', 'Process')
        Push-Location -LiteralPath $repo
        try {
            [Environment]::SetEnvironmentVariable('V13_JQUANTS_T1_WRAPPER_GATE', 'REVIEWED_PRE_GATE_PASS', 'Process')
            $harnessCalled = $true
            $lines = @(& $python -E -B -m scripts.v13_execute_jquants_t1_exclusion_private_read --repository-root $repo 2>$null)
            $exit = $LASTEXITCODE
            $expectedReportKeys = @('PRE_GATE_STATUS', 'PRIVATE_BOUNDARY_CROSSED', 'AUTHORIZATION_CONSUMED',
                'AUTHORIZATION_REUSABLE', 'SOURCE_OPENS', 'PRIVATE_CONTENT_READS', 'NETWORK_REQUESTS',
                'PRICE_PAYLOAD_READS', 'OUTCOME_READS', 'NON_T1_IDENTITIES_RETAINED', 'V13_UNIVERSE_SELECTED',
                'SOURCE_BINDING_MATCH', 'T1_HASH_MATCH', 'CONSUMED_RECEIPT_WRITTEN', 'PRIVATE_STATE_WRITTEN',
                'EXECUTION_RESULT', 'FAILURE_CLASS', 'AUTOMATIC_RETRY', 'SECOND_PRIVATE_SOURCE_READ')
            if ($lines.Count -ne $expectedReportKeys.Count) { throw 'BLOCK' }
            for ($i = 0; $i -lt $expectedReportKeys.Count; $i++) {
                $safePattern = '^' + $expectedReportKeys[$i] + '=(?:[A-Z][A-Z0-9_]*|true|false|unknown|[01])$'
                if ([string]$lines[$i] -cnotmatch $safePattern) { throw 'BLOCK' }
            }
            $report = $lines -join "`n"
            if ($exit -ne 0 -and $report -match 'EXECUTION_RESULT=PASS') { throw 'BLOCK' }
        }
        finally {
            Pop-Location
            [Environment]::SetEnvironmentVariable('V13_JQUANTS_T1_WRAPPER_GATE', $priorGate, 'Process')
        }
    }
    catch {
        if (-not $harnessCalled) {
            $report = "EXECUTION_RESULT=PRE_GATE_STOP FAILURE_CLASS=$reason PRIVATE_BOUNDARY_CROSSED=false AUTHORIZATION_CONSUMED=false SOURCE_OPENS=0 NETWORK_REQUESTS=0 PRICE_PAYLOAD_READS=0 OUTCOME_READS=0 NON_T1_IDENTITIES_RETAINED=false V13_UNIVERSE_SELECTED=false"
        } elseif ($report -notmatch 'EXECUTION_RESULT=(PASS|PRE_GATE_STOP|PRE_BOUNDARY_FAILURE|POST_BOUNDARY_FAILURE)') {
            $report = 'EXECUTION_RESULT=POST_BOUNDARY_FAILURE FAILURE_CLASS=HARNESS_REPORT_UNKNOWN PRIVATE_BOUNDARY_CROSSED=unknown AUTHORIZATION_CONSUMED=unknown SOURCE_OPENS=unknown NETWORK_REQUESTS=0 PRICE_PAYLOAD_READS=0 OUTCOME_READS=0 NON_T1_IDENTITIES_RETAINED=false V13_UNIVERSE_SELECTED=false'
        }
    }
    Write-Output $report
    if ($report -notmatch 'EXECUTION_RESULT=PASS') { exit 1 }
}
