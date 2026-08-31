param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExpectedHead,
    [switch]$Authorize
)

& {
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
        $result = [pscustomobject]@{
            ExitCode = $process.ExitCode
            Stdout = $capturedStdout
            Stderr = $capturedStderr
        }
        $process.Dispose()
        return $result
    }

    try {
        if (Test-Path -LiteralPath $stateRoot) { throw "existing successor state" }
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
        foreach ($requiredPath in @("src/v9_006_f1_semantic_successor_public_acquisition_production.py", "src/v9_006_f1_semantic_successor_public_acquisition_runtime.py", "scripts/run_v9_006_f1_semantic_successor_public_acquisition.py")) {
            $null = Invoke-GitSafe @("cat-file", "-e", "HEAD:$requiredPath")
        }
        if (-not (Test-Path -LiteralPath $canonicalInterpreter -PathType Leaf)) { throw "canonical environment" }
        if (-not (Test-Path -LiteralPath $readinessChecker -PathType Leaf)) { throw "readiness checker" }
        $readinessResult = Invoke-CanonicalPython $canonicalInterpreter @($readinessChecker) $null
        if ($readinessResult.ExitCode -ne 0) { throw "environment readiness" }
        $readiness = $readinessResult.Stdout | ConvertFrom-Json
        if ($readiness.REAL_EXECUTION_ENVIRONMENT_READY -ne $true -or $readiness.REAL_EXECUTION_ENVIRONMENT_FROZEN -ne $true -or $readiness.INTERPRETER_MATCH -ne $true -or $readiness.GENERAL_PROJECT_VENV_REJECTED -ne $true -or $readiness.DEPENDENCY_READINESS -ne "PASS" -or $readiness.ENVIRONMENT_LOCK_CHECK -ne "PASS" -or $readiness.ENVIRONMENT_FREEZE_CHECK -ne "PASS" -or $readiness.ENVIRONMENT_FREEZE_EVIDENCE_GIT_SHA256_MATCH -ne $true -or $readiness.REAL_NETWORK_REQUESTS -ne 0 -or $readiness.PRIVATE_READS -ne 0 -or $readiness.GATES_CONSUMED -ne 0) { throw "environment predicate" }
        if (Test-Path -LiteralPath $stateRoot) { throw "existing successor state" }
        if (-not $Authorize.IsPresent) {
            Write-Output $authMarker
            $exitCode = 4
        } else {
            if (-not (Test-Path -LiteralPath $productionPython -PathType Leaf)) { throw "production entrypoint" }
            if (Test-Path -LiteralPath $stateRoot) { throw "existing successor state" }
            $childArguments = @($productionPython, "--state-root", $stateRoot, "--implementation-git-sha", $ExpectedHead)
            $childResult = Invoke-CanonicalPython $canonicalInterpreter $childArguments $null
            $childLines = @($childResult.Stdout -split "`r?`n" | Where-Object { $_.Length -gt 0 })
            if ($childResult.ExitCode -notin @(0, 2) -or $childLines.Count -ne 1) { throw "child boundary" }
            $validatorCode = "import json,sys; from src.v9_006_f1_semantic_successor_public_acquisition import validate_safe_acquisition_result,canonical_json; raw=sys.stdin.read(); parsed=json.loads(raw); validate_safe_acquisition_result(parsed); sys.stdout.write(canonical_json(parsed))"
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
