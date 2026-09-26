param(
    [switch] $Execute,
    [string] $ApprovedImplementationSha,
    [string] $ExecutionHead,
    [string] $AuthorizationRecord,
    [string] $PrivateT1State,
    [string] $CalendarLock,
    [string] $CalendarSha256,
    [string] $OutputDirectory
)

& {
    $ErrorActionPreference = 'Stop'
    $repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
    if (-not $Execute) {
        Write-Output 'V13_PUBLIC_DATALOCK_REHEARSAL=true'
        Write-Output 'NETWORK_REQUESTS=0'
        Write-Output 'PRIVATE_READS=0'
        Write-Output 'REAL_UNIVERSE_SELECTED=false'
        return
    }
    if ([string]::IsNullOrWhiteSpace($ApprovedImplementationSha) -or
        [string]::IsNullOrWhiteSpace($ExecutionHead) -or
        [string]::IsNullOrWhiteSpace($AuthorizationRecord) -or
        [string]::IsNullOrWhiteSpace($PrivateT1State) -or
        [string]::IsNullOrWhiteSpace($CalendarLock) -or
        [string]::IsNullOrWhiteSpace($CalendarSha256) -or
        [string]::IsNullOrWhiteSpace($OutputDirectory)) { throw 'BLOCK_MISSING_EXPLICIT_INPUT' }
    if ($ApprovedImplementationSha -cnotmatch '^[0-9a-f]{40}$' -or
        $ExecutionHead -cnotmatch '^[0-9a-f]{40}$' -or
        $CalendarSha256 -cne '30ad5d66c6c3b8bd2c71a814309e6133a03331437089103799551fa72150c44c') {
        throw 'BLOCK_INVALID_BINDING'
    }
    Set-Location -LiteralPath $repoRoot
    $branchName = (& git branch --show-current).Trim()
    $localHead = (& git rev-parse HEAD).Trim()
    $remoteHead = (& git ls-remote origin refs/heads/v13-conditional-cross-sectional-short-horizon)
    $treeStatus = & git status --porcelain
    if ($branchName -cne 'v13-conditional-cross-sectional-short-horizon' -or
        $localHead -cne $ExecutionHead -or
        @($remoteHead).Count -ne 1 -or
        (-not $remoteHead.StartsWith($ExecutionHead + "`t")) -or
        -not [string]::IsNullOrWhiteSpace(($treeStatus -join ''))) { throw 'BLOCK_REPOSITORY_PREFLIGHT' }
    $ancestor = (& git merge-base $ApprovedImplementationSha $ExecutionHead).Trim()
    if ($LASTEXITCODE -ne 0 -or $ancestor -cne $ApprovedImplementationSha) {
        throw 'BLOCK_IMPLEMENTATION_ANCESTRY'
    }
    $resultRelative = 'docs/v13/V13_MASTER_CALENDAR_REAL_GENERATION_SAFE_RESULT.json'
    $resultTree = & git ls-tree HEAD -- $resultRelative
    $resultPath = Join-Path $repoRoot $resultRelative
    if (@($resultTree).Count -ne 1 -or
        $resultTree -cnotmatch '^100644 blob 336e5dd6141230b95fa4548231b0a137be6a15cb\s' -or
        -not (Test-Path -LiteralPath $resultPath -PathType Leaf) -or
        (& git hash-object -- $resultPath).Trim() -cne '336e5dd6141230b95fa4548231b0a137be6a15cb') {
        throw 'BLOCK_MASTER_CALENDAR_RESULT_BLOB'
    }
    $authorization = Get-Content -LiteralPath $AuthorizationRecord -Raw | ConvertFrom-Json
    if ($authorization.schema -cne 'V13_PUBLIC_DATALOCK_POINT_OF_USE_AUTHORIZATION_V1' -or
        $authorization.reviewed_implementation_sha -cne $ApprovedImplementationSha -or
        $authorization.execution_head -cne $ExecutionHead -or
        $authorization.master_calendar_sha256 -cne $CalendarSha256 -or
        $authorization.master_calendar_safe_result_blob -cne '336e5dd6141230b95fa4548231b0a137be6a15cb' -or
        $authorization.public_datalock_execution_approved -cne $true) {
        throw 'BLOCK_AUTHORIZATION_OR_CALENDAR_SCOPE'
    }
    if (-not (Test-Path -LiteralPath $CalendarLock -PathType Leaf)) {
        throw 'BLOCK_CALENDAR_INPUT'
    }
    $calendarResolved = (Resolve-Path -LiteralPath $CalendarLock).Path
    if ($calendarResolved.StartsWith($repoRoot + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
        $calendarResolved -ceq $repoRoot) { throw 'BLOCK_CALENDAR_PATH_IN_REPOSITORY' }
    $pythonExe = Join-Path $repoRoot '.venv-real-execution\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw 'BLOCK_CANONICAL_ENVIRONMENT' }
    & $pythonExe scripts/check_current_protected_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'BLOCK_CANONICAL_ENVIRONMENT' }
    & $pythonExe scripts/v13_public_data_lock_execute.py --preflight-calendar $CalendarLock $CalendarSha256
    if ($LASTEXITCODE -ne 0) { throw 'BLOCK_MASTER_CALENDAR_INPUT' }
    if (-not (Test-Path -LiteralPath $PrivateT1State -PathType Leaf) -or
        ((Test-Path -LiteralPath $OutputDirectory) -and
         -not (Test-Path -LiteralPath $OutputDirectory -PathType Container))) {
        throw 'BLOCK_INPUT_OR_OUTPUT_TOPOLOGY'
    }
    $privateResolved = (Resolve-Path -LiteralPath $PrivateT1State).Path
    $outputParent = Split-Path -Parent $OutputDirectory
    if (-not (Test-Path -LiteralPath $outputParent -PathType Container)) { throw 'BLOCK_OUTPUT_PARENT' }
    $outputParentResolved = (Resolve-Path -LiteralPath $outputParent).Path
    foreach ($candidatePath in @($privateResolved, $outputParentResolved)) {
        if ($candidatePath.StartsWith($repoRoot + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
            $candidatePath -ceq $repoRoot) { throw 'BLOCK_PRIVATE_OR_RAW_PATH_IN_REPOSITORY' }
    }
    & $pythonExe scripts/v13_public_data_lock_execute.py --t1-state $PrivateT1State --v4-csv V4_UNIVERSE.csv --calendar-lock $CalendarLock --calendar-sha256 $CalendarSha256 --output $OutputDirectory --implementation-sha $ApprovedImplementationSha
    if ($LASTEXITCODE -ne 0) { throw 'BLOCK_PUBLIC_DATALOCK' }
}
