param(
    [switch] $Execute,
    [string] $ReviewedHead,
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
    if ([string]::IsNullOrWhiteSpace($ReviewedHead) -or
        [string]::IsNullOrWhiteSpace($AuthorizationRecord) -or
        [string]::IsNullOrWhiteSpace($PrivateT1State) -or
        [string]::IsNullOrWhiteSpace($CalendarLock) -or
        [string]::IsNullOrWhiteSpace($CalendarSha256) -or
        [string]::IsNullOrWhiteSpace($OutputDirectory)) { throw 'BLOCK_MISSING_EXPLICIT_INPUT' }
    if ($ReviewedHead -cnotmatch '^[0-9a-f]{40}$' -or $CalendarSha256 -cnotmatch '^[0-9a-f]{64}$') {
        throw 'BLOCK_INVALID_BINDING'
    }
    Set-Location -LiteralPath $repoRoot
    $branchName = (& git branch --show-current).Trim()
    $localHead = (& git rev-parse HEAD).Trim()
    $remoteHead = (& git ls-remote origin refs/heads/v13-conditional-cross-sectional-short-horizon)
    $treeStatus = & git status --porcelain
    if ($branchName -cne 'v13-conditional-cross-sectional-short-horizon' -or
        $localHead -cne $ReviewedHead -or
        @($remoteHead).Count -ne 1 -or
        (-not $remoteHead.StartsWith($ReviewedHead + "`t")) -or
        -not [string]::IsNullOrWhiteSpace(($treeStatus -join ''))) { throw 'BLOCK_REPOSITORY_PREFLIGHT' }
    $authorization = Get-Content -LiteralPath $AuthorizationRecord -Raw | ConvertFrom-Json
    if ($authorization.schema -cne 'V13_PUBLIC_DATALOCK_POINT_OF_USE_AUTHORIZATION_V1' -or
        $authorization.reviewed_implementation_sha -cne $ReviewedHead -or
        $authorization.calendar_scope_extension_approved -cne $true -or
        $authorization.public_datalock_execution_approved -cne $true) {
        throw 'BLOCK_AUTHORIZATION_OR_CALENDAR_SCOPE'
    }
    if (-not (Test-Path -LiteralPath $PrivateT1State -PathType Leaf) -or
        -not (Test-Path -LiteralPath $CalendarLock -PathType Leaf) -or
        (Test-Path -LiteralPath $OutputDirectory)) { throw 'BLOCK_INPUT_OR_OUTPUT_TOPOLOGY' }
    $privateResolved = (Resolve-Path -LiteralPath $PrivateT1State).Path
    $calendarResolved = (Resolve-Path -LiteralPath $CalendarLock).Path
    $outputParent = Split-Path -Parent $OutputDirectory
    if (-not (Test-Path -LiteralPath $outputParent -PathType Container)) { throw 'BLOCK_OUTPUT_PARENT' }
    $outputParentResolved = (Resolve-Path -LiteralPath $outputParent).Path
    foreach ($candidatePath in @($privateResolved, $calendarResolved, $outputParentResolved)) {
        if ($candidatePath.StartsWith($repoRoot + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
            $candidatePath -ceq $repoRoot) { throw 'BLOCK_PRIVATE_OR_RAW_PATH_IN_REPOSITORY' }
    }
    $pythonExe = Join-Path $repoRoot '.venv-real-execution\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw 'BLOCK_CANONICAL_ENVIRONMENT' }
    & $pythonExe scripts/check_current_protected_environment.py
    if ($LASTEXITCODE -ne 0) { throw 'BLOCK_CANONICAL_ENVIRONMENT' }
    & $pythonExe scripts/v13_public_data_lock_execute.py --t1-state $PrivateT1State --v4-csv V4_UNIVERSE.csv --calendar-lock $CalendarLock --calendar-sha256 $CalendarSha256 --output $OutputDirectory --implementation-sha $ReviewedHead
    if ($LASTEXITCODE -ne 0) { throw 'BLOCK_PUBLIC_DATALOCK' }
}
