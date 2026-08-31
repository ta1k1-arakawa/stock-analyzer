& {
    param(
        [Parameter(Mandatory = $true)]
        [ValidatePattern('^[0-9a-f]{40}$')]
        [string]$ExpectedHead,
        [switch]$Authorize
    )

    $ErrorActionPreference = "Stop"
    $authoritativeBranch = "v9-cross-sectional-close-auction-design"
    $designBlob = "ea612a777dd2915121f1747cdd3a14ff7f668efb"
    $designPath = "V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION_DESIGN.md"
    $repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $stateParent = [IO.Directory]::GetParent($repoRoot).FullName
    $stateRoot = Join-Path $stateParent "v9-006-f1-successor-public-acquisition-state"
    $canonicalInterpreter = Join-Path $repoRoot ".venv-real-execution\Scripts\python.exe"
    $readinessChecker = Join-Path $repoRoot "scripts\check_real_execution_env.py"
    $productionPython = Join-Path $repoRoot "scripts\run_v9_006_f1_semantic_successor_public_acquisition.py"
    $authMarker = "V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION_AUTHORIZATION_REQUIRED"
    $failureMarker = "V9_006_F1_SEMANTIC_SUCCESSOR_PUBLIC_ACQUISITION_IMPLEMENTATION_FAILURE"
    $exitCode = 3

    function Invoke-GitSafe([string[]]$gitArguments) {
        $gitOutput = & git -C $repoRoot @gitArguments 2>$null
        if ($LASTEXITCODE -ne 0) { throw "git preflight" }
        return (($gitOutput -join "`n").Trim())
    }

    try {
        if ((Invoke-GitSafe @("rev-parse", "--show-toplevel")) -ne $repoRoot) { throw "repository binding" }
        if ((Invoke-GitSafe @("branch", "--show-current")) -ne $authoritativeBranch) { throw "branch binding" }
        if ($ExpectedHead -cnotmatch '^[0-9a-f]{40}$') { throw "head format" }
        if ((Invoke-GitSafe @("rev-parse", "HEAD")) -cne $ExpectedHead) { throw "head binding" }
        if ((Invoke-GitSafe @("status", "--porcelain")) -ne "") { throw "worktree binding" }
        $null = Invoke-GitSafe @("fetch", "--no-tags", "origin", $authoritativeBranch)
        if ((Invoke-GitSafe @("rev-parse", "origin/$authoritativeBranch")) -cne $ExpectedHead) { throw "remote binding" }
        if ((Invoke-GitSafe @("rev-parse", "HEAD:$designPath")) -cne $designBlob) { throw "design binding" }
        foreach ($requiredPath in @("src/v9_006_f1_semantic_successor_public_acquisition_production.py", "src/v9_006_f1_semantic_successor_public_acquisition_runtime.py", "scripts/run_v9_006_f1_semantic_successor_public_acquisition.py")) {
            $null = Invoke-GitSafe @("cat-file", "-e", "HEAD:$requiredPath")
        }
        if (-not (Test-Path -LiteralPath $canonicalInterpreter -PathType Leaf)) { throw "canonical environment" }
        if (-not (Test-Path -LiteralPath $readinessChecker -PathType Leaf)) { throw "readiness checker" }
        $readinessOutput = (& $canonicalInterpreter $readinessChecker 2>&1 | Out-String)
        if ($LASTEXITCODE -ne 0) { throw "environment readiness" }
        $readiness = $readinessOutput | ConvertFrom-Json
        if ($readiness.REAL_EXECUTION_ENVIRONMENT_READY -ne $true -or $readiness.REAL_EXECUTION_ENVIRONMENT_FROZEN -ne $true -or $readiness.INTERPRETER_MATCH -ne $true -or $readiness.GENERAL_PROJECT_VENV_REJECTED -ne $true -or $readiness.DEPENDENCY_READINESS -ne "PASS" -or $readiness.ENVIRONMENT_LOCK_CHECK -ne "PASS" -or $readiness.ENVIRONMENT_FREEZE_CHECK -ne "PASS" -or $readiness.ENVIRONMENT_FREEZE_EVIDENCE_GIT_SHA256_MATCH -ne $true -or $readiness.REAL_NETWORK_REQUESTS -ne 0 -or $readiness.PRIVATE_READS -ne 0 -or $readiness.GATES_CONSUMED -ne 0) { throw "environment predicate" }
        if (Test-Path -LiteralPath $stateRoot) { throw "existing successor state" }
        if (-not $Authorize) {
            Write-Output $authMarker
            $exitCode = 4
        } else {
            if (-not (Test-Path -LiteralPath $productionPython -PathType Leaf)) { throw "production entrypoint" }
            if (Test-Path -LiteralPath $stateRoot) { throw "existing successor state" }
            $childOutput = (& $canonicalInterpreter $productionPython "--state-root" $stateRoot "--implementation-git-sha" $ExpectedHead 2>&1 | Out-String)
            $childExitCode = $LASTEXITCODE
            $childLines = @($childOutput -split "`r?`n" | Where-Object { $_.Length -gt 0 })
            if ($childExitCode -notin @(0, 2) -or $childLines.Count -ne 1) { throw "child boundary" }
            $safeResult = $childLines[0] | ConvertFrom-Json
            if ($null -eq $safeResult.result -or $null -eq $safeResult.failure_stage -or $null -eq $safeResult.structural_evidence_sha256) { throw "safe result" }
            Write-Output $childLines[0]
            $exitCode = $childExitCode
        }
    } catch {
        Write-Output $failureMarker
        $exitCode = 3
    }
    exit $exitCode
}
