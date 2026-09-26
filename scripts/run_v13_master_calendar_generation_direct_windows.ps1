param(
    [Parameter(Mandatory=$true)][string]$ExpectedReviewedHead,
    [Parameter(Mandatory=$true)][string]$OfficialWheelPath,
    [Parameter(Mandatory=$true)][string]$OutputRoot,
    [Parameter(Mandatory=$true)][string]$PointOfUseAuthorizationArtifact
)

& {
    $ErrorActionPreference = "Stop"
    $repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
    $authoritativeBranch = "v13-conditional-cross-sectional-short-horizon"
    $pythonExe = Join-Path $repoRoot ".venv-real-execution\Scripts\python.exe"
    $wheelName = "pandas_market_calendars-5.4.0-py3-none-any.whl"
    $wheelDigest = "bb2b93b28d496cab173b41c7d120fd5cd9d506b31f3bb0ad3d1d9f2b60d9d9e3"
    Set-Location -LiteralPath $repoRoot

    function Require-GitValue([string[]]$gitArgs, [string]$expected) {
        $actual = (& git @gitArgs 2>$null)
        if ($LASTEXITCODE -ne 0 -or $actual -ne $expected) { throw "PRE_GATE_GIT_MISMATCH" }
    }
    if ($ExpectedReviewedHead -cnotmatch '^[0-9a-f]{40}$') { throw "EXPECTED_HEAD_INVALID" }
    Require-GitValue @("remote", "get-url", "origin") "https://github.com/ta1k1-arakawa/stock-analyzer.git"
    Require-GitValue @("branch", "--show-current") $authoritativeBranch
    Require-GitValue @("rev-parse", "HEAD") $ExpectedReviewedHead
    $remoteLine = (& git ls-remote origin "refs/heads/$authoritativeBranch" 2>$null)
    if ($LASTEXITCODE -ne 0 -or $remoteLine -ne "$ExpectedReviewedHead`trefs/heads/$authoritativeBranch") { throw "REMOTE_HEAD_MISMATCH" }
    $dirty = (& git status --porcelain=v1 --untracked-files=all 2>$null)
    if ($LASTEXITCODE -ne 0 -or $dirty) { throw "WORKTREE_NOT_CLEAN" }

    Require-GitValue @("rev-parse", "HEAD:V13_MASTER_CALENDAR_AUTHORITY_DESIGN.md") "390a6ad8e8994f0ed6921183f581c43496eac1e3"
    Require-GitValue @("rev-parse", "HEAD:docs/v13/V13_MASTER_CALENDAR_AUTHORITY_DESIGN_HUMAN_FREEZE_APPROVAL.json") "f7b0e09b350acd365ef51e0f925fa71c88b5bae3"
    if (-not (Test-Path -LiteralPath "docs/v13/V13_MASTER_CALENDAR_AUTHORITY_DESIGN_HUMAN_FREEZE_APPROVAL.json" -PathType Leaf)) { throw "FREEZE_APPROVAL_MISSING" }
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw "CANONICAL_INTERPRETER_MISSING" }
    $null = & $pythonExe "scripts/check_current_protected_environment.py" 2>&1
    if ($LASTEXITCODE -ne 0) { throw "PROTECTED_ENVIRONMENT_CHECK_FAILED" }

    if (-not (Test-Path -LiteralPath $OfficialWheelPath -PathType Leaf)) { throw "OFFICIAL_WHEEL_MISSING" }
    $wheelFile = Get-Item -LiteralPath $OfficialWheelPath
    if ($wheelFile.Name -cne $wheelName -or (Get-FileHash -LiteralPath $wheelFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -cne $wheelDigest) { throw "OFFICIAL_WHEEL_MISMATCH" }
    $outputFull = [System.IO.Path]::GetFullPath($OutputRoot)
    if ($outputFull -eq $repoRoot -or $outputFull.StartsWith($repoRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw "OUTPUT_INSIDE_REPOSITORY" }
    if (Test-Path -LiteralPath $outputFull) { throw "OUTPUT_ROOT_ALREADY_EXISTS" }
    if ([string]::IsNullOrWhiteSpace($PointOfUseAuthorizationArtifact)) { throw "POINT_OF_USE_AUTHORITY_REQUIRED" }

    # The future execution Issue must replace this deliberate stop with an exact
    # reviewed, single-use authority verifier. This preparation has no such gate.
    throw "POINT_OF_USE_GATE_NOT_DEFINED"

    & $pythonExe "scripts/v13_master_calendar_generate.py" --official-wheel $wheelFile.FullName --output-root $outputFull --implementation-sha $ExpectedReviewedHead
    if ($LASTEXITCODE -ne 0) { throw "CALENDAR_GENERATION_FAILED" }
}
