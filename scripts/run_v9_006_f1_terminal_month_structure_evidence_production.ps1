param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExpectedHead,
    [switch]$Authorize
)

& {
    $ErrorActionPreference = "Stop"
    $authoritativeBranch = "v9-cross-sectional-close-auction-design"
    $designBlob = "6112b92f39f34c594d36a28d72072dcb255b9eee"
    $designPath = "V9_006_F1_TERMINAL_MONTH_STRUCTURE_EVIDENCE_DESIGN.md"
    $repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    $stateParent = [IO.Directory]::GetParent($repoRoot).FullName
    $stateRoot = Join-Path $stateParent "v9-006-f1-successor-public-acquisition-state"
    $canonicalInterpreter = Join-Path $repoRoot ".venv-real-execution\Scripts\python.exe"
    $readinessChecker = Join-Path $repoRoot "scripts\check_real_execution_env.py"
    $productionPython = Join-Path $repoRoot "scripts\run_v9_006_f1_terminal_month_structure_evidence.py"
    $authMarker = "V9_006_F1_TERMINAL_MONTH_STRUCTURE_EVIDENCE_AUTHORIZATION_REQUIRED"
    $failureMarker = "V9_006_F1_TERMINAL_MONTH_STRUCTURE_EVIDENCE_IMPLEMENTATION_FAILURE"
    $exitCode = 3

    function Convert-CanonicalWindowsPath([string]$pathValue) {
        if ([string]::IsNullOrWhiteSpace($pathValue)) { throw "path" }
        $fullPath = [IO.Path]::GetFullPath($pathValue)
        return $fullPath.TrimEnd('\', '/').Replace('/', '\').ToUpperInvariant()
    }

    function Invoke-GitSafe([string[]]$gitArguments) {
        $gitOutput = & git -C $repoRoot @gitArguments 2>$null
        if ($LASTEXITCODE -ne 0) { throw "git preflight" }
        return (($gitOutput -join "`n").Trim())
    }

    function Invoke-CanonicalPython([string]$pythonPath, [string[]]$pythonArguments, [string]$stdinText) {
        $startInfo = [Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $pythonPath
        $startInfo.WorkingDirectory = $repoRoot
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardInput = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true
        foreach ($argument in $pythonArguments) { [void]$startInfo.ArgumentList.Add([string]$argument) }
        $process = [Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) { throw "python start" }
        if ($null -ne $stdinText) { $process.StandardInput.Write($stdinText) }
        $process.StandardInput.Close()
        $capturedStdout = $process.StandardOutput.ReadToEnd()
        $capturedStderr = $process.StandardError.ReadToEnd()
        $process.WaitForExit()
        $result = [pscustomobject]@{ ExitCode = $process.ExitCode; Stdout = $capturedStdout; Stderr = $capturedStderr }
        $process.Dispose()
        return $result
    }

    function Test-CanonicalEnvironmentPredicate($readiness) {
        return ($readiness.REAL_EXECUTION_ENVIRONMENT_READY -eq $true -and
            $readiness.REAL_EXECUTION_ENVIRONMENT_FROZEN -eq $true -and
            $readiness.INTERPRETER_MATCH -eq $true -and
            $readiness.GENERAL_PROJECT_VENV_REJECTED -eq $false -and
            $readiness.DEPENDENCY_READINESS -eq "PASS" -and
            $readiness.ENVIRONMENT_LOCK_CHECK -eq "PASS" -and
            $readiness.ENVIRONMENT_FREEZE_CHECK -eq "PASS" -and
            $readiness.ENVIRONMENT_FREEZE_EVIDENCE_GIT_SHA256_MATCH -eq $true -and
            $readiness.REAL_NETWORK_REQUESTS -eq 0 -and
            $readiness.PRIVATE_READS -eq 0 -and
            $readiness.GATES_CONSUMED -eq 0)
    }

    try {
        $canonicalRepoRoot = Convert-CanonicalWindowsPath $repoRoot
        $canonicalGitRoot = Convert-CanonicalWindowsPath (Invoke-GitSafe @("rev-parse", "--show-toplevel"))
        if ($canonicalGitRoot -cne $canonicalRepoRoot) { throw "repository binding" }
        if ((Invoke-GitSafe @("branch", "--show-current")) -cne $authoritativeBranch) { throw "branch binding" }
        if ($ExpectedHead -cnotmatch '^[0-9a-f]{40}$') { throw "head format" }
        if ((Invoke-GitSafe @("rev-parse", "HEAD")) -cne $ExpectedHead) { throw "head binding" }
        if ((Invoke-GitSafe @("status", "--porcelain")) -ne "") { throw "worktree binding" }
        $null = Invoke-GitSafe @("fetch", "--no-tags", "origin", $authoritativeBranch)
        if ((Invoke-GitSafe @("rev-parse", "origin/$authoritativeBranch")) -cne $ExpectedHead) { throw "remote binding" }
        if ((Invoke-GitSafe @("rev-parse", "HEAD:$designPath")) -cne $designBlob) { throw "design binding" }
        foreach ($requiredPath in @("src/v9_006_f1_terminal_month_structure_evidence.py", "src/v9_006_f1_terminal_month_structure_evidence_production.py", "scripts/run_v9_006_f1_terminal_month_structure_evidence.py")) {
            $null = Invoke-GitSafe @("cat-file", "-e", "HEAD:$requiredPath")
        }
        if (-not (Test-Path -LiteralPath $canonicalInterpreter -PathType Leaf)) { throw "canonical environment" }
        if (-not (Test-Path -LiteralPath $readinessChecker -PathType Leaf)) { throw "readiness checker" }
        $readinessResult = Invoke-CanonicalPython $canonicalInterpreter @($readinessChecker) $null
        if ($readinessResult.ExitCode -ne 0) { throw "environment readiness" }
        $readiness = $readinessResult.Stdout | ConvertFrom-Json
        if (-not (Test-CanonicalEnvironmentPredicate $readiness)) { throw "environment predicate" }
        if (-not (Test-Path -LiteralPath $stateRoot -PathType Container)) { throw "missing successor state" }
        if (-not $Authorize.IsPresent) {
            Write-Output $authMarker
            $exitCode = 4
        } else {
            $childArguments = @($productionPython, "--diagnostic-implementation-git-sha", $ExpectedHead)
            $childResult = Invoke-CanonicalPython $canonicalInterpreter $childArguments $null
            $childLines = @($childResult.Stdout -split "`r?`n" | Where-Object { $_.Length -gt 0 })
            if ($childResult.ExitCode -notin @(0, 2) -or $childLines.Count -ne 1) { throw "child boundary" }
            $validatorCode = "import json,sys; from src.v9_006_f1_terminal_month_structure_evidence import validate_safe_result,canonical_json; raw=sys.stdin.read(); parsed=json.loads(raw); validate_safe_result(parsed); sys.stdout.write(canonical_json(parsed))"
            $validatorResult = Invoke-CanonicalPython $canonicalInterpreter @("-c", $validatorCode) ($childLines[0] + "`n")
            $validatedLine = $validatorResult.Stdout.TrimEnd("`r", "`n")
            if ($validatorResult.ExitCode -ne 0 -or $validatedLine -cne $childLines[0]) { throw "safe result" }
            Write-Output $childLines[0]
            $exitCode = $childResult.ExitCode
        }
    } catch {
        Write-Output $failureMarker
        $exitCode = 3
    }
    exit $exitCode
}
