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
    $approvedImplementation = "983c3c6da51865a46a8bb9361bb13604a5b0c112"
    $pythonExe = Join-Path $repoRoot ".venv-real-execution\Scripts\python.exe"
    $wheelName = "pandas_market_calendars-5.4.0-py3-none-any.whl"
    $wheelDigest = "bb2b93b28d496cab173b41c7d120fd5cd9d506b31f3bb0ad3d1d9f2b60d9d9e3"
    $authorizationPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "docs/v13/V13_MASTER_CALENDAR_POINT_OF_USE_AUTHORIZATION.json"))
    $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
    if ([string]::IsNullOrWhiteSpace($localAppData)) { throw "LOCAL_APP_DATA_UNAVAILABLE" }
    $gateDir = Join-Path $localAppData "stock-analyzer\v13-master-calendar"
    $gatePath = Join-Path $gateDir "V13_MASTER_CALENDAR_GENERATION_GATE.json"
    $resultPath = Join-Path $gateDir "V13_MASTER_CALENDAR_GENERATION_RESULT.json"
    Set-Location -LiteralPath $repoRoot

    function Require-GitValue([string[]]$gitArgs, [string]$expected) {
        $actual = (& git @gitArgs 2>$null)
        if ($LASTEXITCODE -ne 0 -or $actual -cne $expected) { throw "PRE_GATE_GIT_MISMATCH" }
    }
    function Write-NewDurableJson([string]$path, [object]$record) {
        $payload = [System.Text.Encoding]::UTF8.GetBytes(($record | ConvertTo-Json -Depth 5 -Compress) + "`n")
        $stream = [System.IO.FileStream]::new($path, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        try { $stream.Write($payload, 0, $payload.Length); $stream.Flush($true) }
        finally { $stream.Dispose() }
    }
    function Assert-DirectoryWritable([string]$directory) {
        $probePath = Join-Path $directory (".v13-calendar-preflight-" + [Guid]::NewGuid().ToString("N"))
        $stream = [System.IO.FileStream]::new($probePath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        try { $stream.WriteByte(0); $stream.Flush($true) }
        finally { $stream.Dispose(); [System.IO.File]::Delete($probePath) }
    }

    if ($ExpectedReviewedHead -cnotmatch '^[0-9a-f]{40}$') { throw "EXPECTED_HEAD_INVALID" }
    Require-GitValue @("remote", "get-url", "origin") "https://github.com/ta1k1-arakawa/stock-analyzer.git"
    Require-GitValue @("branch", "--show-current") $authoritativeBranch
    Require-GitValue @("rev-parse", "HEAD") $ExpectedReviewedHead
    $remoteLine = (& git ls-remote origin "refs/heads/$authoritativeBranch" 2>$null)
    if ($LASTEXITCODE -ne 0 -or $remoteLine -cne "$ExpectedReviewedHead`trefs/heads/$authoritativeBranch") { throw "REMOTE_HEAD_MISMATCH" }
    $dirty = (& git status --porcelain=v1 --untracked-files=all 2>$null)
    if ($LASTEXITCODE -ne 0 -or $dirty) { throw "WORKTREE_NOT_CLEAN" }
    Require-GitValue @("rev-parse", "HEAD:V13_MASTER_CALENDAR_AUTHORITY_DESIGN.md") "390a6ad8e8994f0ed6921183f581c43496eac1e3"
    Require-GitValue @("rev-parse", "HEAD:docs/v13/V13_MASTER_CALENDAR_AUTHORITY_DESIGN_HUMAN_FREEZE_APPROVAL.json") "f7b0e09b350acd365ef51e0f925fa71c88b5bae3"
    Require-GitValue @("rev-parse", "HEAD:docs/v13/V13_MASTER_CALENDAR_POINT_OF_USE_AUTHORIZATION.json") "2ccb3283fbc212d8f5d942237da79924e7a2ccf5"
    $null = & git merge-base --is-ancestor $approvedImplementation HEAD
    if ($LASTEXITCODE -ne 0) { throw "APPROVED_IMPLEMENTATION_NOT_ANCESTOR" }
    if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) { throw "CANONICAL_INTERPRETER_MISSING" }
    $null = & $pythonExe "scripts/check_current_protected_environment.py" 2>&1
    if ($LASTEXITCODE -ne 0) { throw "PROTECTED_ENVIRONMENT_CHECK_FAILED" }

    if (-not (Test-Path -LiteralPath $OfficialWheelPath -PathType Leaf)) { throw "OFFICIAL_WHEEL_MISSING" }
    $wheelFile = Get-Item -LiteralPath $OfficialWheelPath
    if ($wheelFile.Name -cne $wheelName -or (Get-FileHash -LiteralPath $wheelFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -cne $wheelDigest) { throw "OFFICIAL_WHEEL_MISMATCH" }
    $outputFull = [System.IO.Path]::GetFullPath($OutputRoot)
    if ($outputFull -eq $repoRoot -or $outputFull.StartsWith($repoRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) { throw "OUTPUT_INSIDE_REPOSITORY" }
    if (Test-Path -LiteralPath $outputFull) { throw "OUTPUT_ROOT_ALREADY_EXISTS" }
    if ($outputFull.StartsWith($gateDir + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
        $gateDir.StartsWith($outputFull + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase) -or
        [string]::Equals($outputFull, $gateDir, [System.StringComparison]::OrdinalIgnoreCase)) { throw "OUTPUT_GATE_PATH_COLLISION" }

    if ([string]::IsNullOrWhiteSpace($PointOfUseAuthorizationArtifact)) { throw "POINT_OF_USE_AUTHORITY_REQUIRED" }
    if (-not [string]::Equals([System.IO.Path]::GetFullPath($PointOfUseAuthorizationArtifact), $authorizationPath, [System.StringComparison]::OrdinalIgnoreCase)) { throw "POINT_OF_USE_AUTHORITY_PATH_MISMATCH" }
    if (-not (Test-Path -LiteralPath $authorizationPath -PathType Leaf)) { throw "POINT_OF_USE_AUTHORITY_MISSING" }
    $authorization = Get-Content -LiteralPath $authorizationPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $required = @{
        schema = "V13_MASTER_CALENDAR_POINT_OF_USE_AUTHORIZATION_V1"
        github_issue = 90
        predecessor_gpt_review_issue = 89
        predecessor_gpt_review_result = "PASS"
        reviewed_implementation_sha = $approvedImplementation
        frozen_design_file = "V13_MASTER_CALENDAR_AUTHORITY_DESIGN.md"
        frozen_design_git_blob = "390a6ad8e8994f0ed6921183f581c43496eac1e3"
        freeze_approval_file = "docs/v13/V13_MASTER_CALENDAR_AUTHORITY_DESIGN_HUMAN_FREEZE_APPROVAL.json"
        freeze_approval_git_blob = "f7b0e09b350acd365ef51e0f925fa71c88b5bae3"
        operation = "GENERATE_ONE_V13_MASTER_CALENDAR"
        calendar_source = "PANDAS_MARKET_CALENDARS_JPX_5_4_0_RELEASE_ARTIFACT"
        calendar_name = "JPX"
        official_pypi_wheel = $wheelName
        official_pypi_wheel_sha256 = $wheelDigest
        window_start = "2015-01-01"
        window_end = "2025-12-31"
        execution_environment = ".venv-real-execution"
        execution_mode = "DIRECT_WINDOWS_POWERSHELL"
        authorization_scope = "MASTER_CALENDAR_GENERATION_ONLY"
        selected_500_authorized = $false
        jpx_yahoo_market_data_acquisition_authorized = $false
        public_data_lock_authorized = $false
        model_fit_authorized = $false
        backtest_authorized = $false
        a_to_q_authorized = $false
        paper_trading_authorized = $false
        real_trading_authorized = $false
    }
    if (@($authorization.PSObject.Properties).Count -ne ($required.Count + 2)) { throw "POINT_OF_USE_AUTHORITY_SCHEMA_MISMATCH" }
    foreach ($key in $authorization.PSObject.Properties.Name) {
        if ($key -cne "human_approval_recorded_utc" -and $key -cne "human_approval_evidence" -and -not $required.ContainsKey($key)) { throw "POINT_OF_USE_AUTHORITY_SCHEMA_MISMATCH" }
    }
    if (([DateTimeOffset]$authorization.human_approval_recorded_utc).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") -cne "2026-09-26T02:55:23Z") { throw "POINT_OF_USE_AUTHORITY_TIME_MISMATCH" }
    foreach ($key in $required.Keys) {
        if ($authorization.$key -cne $required[$key]) { throw "POINT_OF_USE_AUTHORITY_MISMATCH" }
    }
    if (Test-Path -LiteralPath $gatePath) { throw "GENERATION_GATE_ALREADY_CONSUMED" }
    if (Test-Path -LiteralPath $resultPath) { throw "GENERATION_RESULT_ALREADY_EXISTS" }
    Assert-DirectoryWritable ([System.IO.Path]::GetDirectoryName($outputFull))
    $null = [System.IO.Directory]::CreateDirectory($gateDir)
    Assert-DirectoryWritable $gateDir

    # This path validates the same runtime, wheel, source bytes, and output
    # predicates as generation, while importing no provider or creating no calendar.
    $null = & $pythonExe "scripts/v13_master_calendar_generate.py" --official-wheel $wheelFile.FullName --output-root $outputFull --implementation-sha $ExpectedReviewedHead --preflight-only 2>&1
    if ($LASTEXITCODE -ne 0) { throw "PRE_GATE_SOURCE_OR_OUTPUT_PREFLIGHT_FAILED" }
    if (Test-Path -LiteralPath $gatePath) { throw "GENERATION_GATE_ALREADY_CONSUMED" }
    if (Test-Path -LiteralPath $resultPath) { throw "GENERATION_RESULT_ALREADY_EXISTS" }
    if (Test-Path -LiteralPath $outputFull) { throw "OUTPUT_ROOT_ALREADY_EXISTS" }

    # CreateNew is the irreversible single-use boundary. Even partial bytes
    # leave a collision, so an interrupted write never reopens this authority.
    Write-NewDurableJson $gatePath @{
        schema = "V13_MASTER_CALENDAR_GENERATION_GATE_V1"
        status = "CONSUMED"
        approval_issue = 90
        approved_implementation_sha = $approvedImplementation
        execution_sha = $ExpectedReviewedHead
        authorization_sha256 = (Get-FileHash -LiteralPath $authorizationPath -Algorithm SHA256).Hash.ToLowerInvariant()
        output_root = $outputFull
        consumed_utc = [DateTime]::UtcNow.ToString("o")
    }
    try {
        $null = & $pythonExe "scripts/v13_master_calendar_generate.py" --official-wheel $wheelFile.FullName --output-root $outputFull --implementation-sha $ExpectedReviewedHead 2>&1
        if ($LASTEXITCODE -ne 0) { throw "CALENDAR_GENERATION_FAILED" }
        $safeReceiptPath = Join-Path $outputFull "V13_MASTER_CALENDAR_SAFE_RECEIPT.json"
        $calendarPath = Join-Path $outputFull "V13_MASTER_CALENDAR.txt"
        if (-not (Test-Path -LiteralPath $safeReceiptPath -PathType Leaf)) { throw "CALENDAR_SAFE_RECEIPT_MISSING" }
        if (-not (Test-Path -LiteralPath $calendarPath -PathType Leaf)) { throw "CALENDAR_OUTPUT_MISSING" }
        $safeReceipt = Get-Content -LiteralPath $safeReceiptPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($safeReceipt.schema -cne "V13_MASTER_CALENDAR_SAFE_RECEIPT_V1" -or
            $safeReceipt.status -cne "PASS" -or
            $safeReceipt.implementation_sha -cne $ExpectedReviewedHead -or
            $safeReceipt.source_identity -cne "PANDAS_MARKET_CALENDARS_JPX_5_4_0_RELEASE_ARTIFACT" -or
            $safeReceipt.coverage_start -cne "2015-01-01" -or
            $safeReceipt.coverage_end -cne "2025-12-31" -or
            $safeReceipt.anchor_2020_10_01 -cne "INELIGIBLE" -or
            $safeReceipt.anchor_2020_10_02 -cne "ELIGIBLE" -or
            $safeReceipt.calendar_sha256 -cnotmatch '^[0-9a-f]{64}$' -or
            ($safeReceipt.session_count -isnot [int] -and $safeReceipt.session_count -isnot [long]) -or $safeReceipt.session_count -le 0 -or
            (Get-FileHash -LiteralPath $calendarPath -Algorithm SHA256).Hash.ToLowerInvariant() -cne $safeReceipt.calendar_sha256) { throw "CALENDAR_SAFE_RECEIPT_MISMATCH" }
        Write-NewDurableJson $resultPath @{
            schema = "V13_MASTER_CALENDAR_GENERATION_RESULT_V1"
            status = "PASS"
            failure_class = "NONE"
            execution_sha = $ExpectedReviewedHead
            calendar_sha256 = $safeReceipt.calendar_sha256
            session_count = $safeReceipt.session_count
            completed_utc = [DateTime]::UtcNow.ToString("o")
        }
        Write-Output "V13_MASTER_CALENDAR_GENERATION_PASS"
    } catch {
        if (-not (Test-Path -LiteralPath $resultPath)) {
            Write-NewDurableJson $resultPath @{
                schema = "V13_MASTER_CALENDAR_GENERATION_RESULT_V1"
                status = "FAIL"
                failure_class = "POST_GATE_FAILURE"
                execution_sha = $ExpectedReviewedHead
                calendar_sha256 = $null
                session_count = $null
                completed_utc = [DateTime]::UtcNow.ToString("o")
            }
        }
        throw "POST_GATE_FAILURE_NO_RETRY"
    }
}
