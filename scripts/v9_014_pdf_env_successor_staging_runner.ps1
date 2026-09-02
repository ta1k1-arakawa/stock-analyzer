# V9_014 PDF real-execution environment successor -- FUTURE Stage E5
# staging resolution runner.
#
# REVIEWED BUT NOT EXECUTED AT STAGE E2. This file is committed as part of
# Stage E2's offline implementation (per
# V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_DESIGN.md Section 5,
# Stage E2 / Stage E5) so its exact mechanics can be exact-SHA reviewed
# now. It is not run by Stage E2's own targeted checks, and Stage E2 does
# not invoke it, spawn it, or otherwise cause it to execute. Its actual
# execution is a separately authorized, later Stage E5 operation with its
# own fresh point-of-use authority (never reused from this commit's
# review).
#
# Scope (Section 2b, Section 5 Stage E5):
#   - creates and resolves into exactly ONE explicitly NON-CANONICAL
#     staging venv, under the caller-supplied -StagingRoot;
#   - writes evidence (captured freeze snapshots, stdout/stderr, exit
#     code) ONLY under the caller-supplied -OutputRoot;
#   - NEVER creates, writes to, deletes, or resets
#     ".venv-real-execution" (the canonical protected environment) or
#     ".venv" (the separate general project environment) -- neither
#     directory name is ever a write target anywhere in this file;
#   - performs NO PDF/source acquisition of any kind;
#   - performs exactly ONE successor-resolution pip command -- no retry,
#     no repair, no second resolution attempt, regardless of exit code;
#   - never deletes anything (this file contains no `Remove-Item` call
#     anywhere): a staging-path collision, or any other precondition
#     failure, is a fail-closed `throw`, never a deletion/reset;
#   - never touches JPX, Yahoo, or any other real production/research
#     network host.
#
# Run this file directly (a single reviewed .ps1), per
# AI_REAL_EXECUTION_RUNBOOK.md SS1 -- when a later, separately authorized
# Stage E5 operation actually runs it. Do not paste its body as
# independent line-by-line snippets.
#
#   powershell -File scripts\v9_014_pdf_env_successor_staging_runner.ps1 `
#       -ExpectedHead <exact reviewed Stage E1/E2 commit SHA> `
#       -StagingRoot  <absolute path, outside the repo, not named
#                       ".venv-real-execution" or ".venv"> `
#       -OutputRoot   <absolute path, outside the repo>

param(
    [Parameter(Mandatory = $true)][string]$ExpectedHead,
    [Parameter(Mandatory = $true)][string]$StagingRoot,
    [Parameter(Mandatory = $true)][string]$OutputRoot
)

& {
    $ErrorActionPreference = "Stop"

    $authoritativeBranch = "v9-cross-sectional-close-auction-design"
    $canonicalEnvironmentDirectoryName = ".venv-real-execution"
    $generalEnvironmentDirectoryName = ".venv"
    $canonicalPythonExactVersion = "3.12.10"
    $canonicalPlatformSystem = "Windows"

    $directSpecRelativePath = "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt"
    $predecessorLockRelativePath = "requirements-real-execution.lock.txt"

    # Reviewed predecessor lock identity (Section 1), hardcoded here --
    # not merely trusted from whatever the mutable lock file currently
    # says on disk -- so a tampered or stale predecessor lock is
    # independently, mechanically detectable before any staging
    # resolution begins.
    $reviewedPredecessorLockSha256 = "b5c063a1cca585fa100fdc0027d6cdbf4ef33ef5a7fe614230599fb882b51f96"

    # Frozen predecessor pins (Section 3a): constraints the successor
    # resolution must preserve exactly, never a starting point to drift
    # from.
    $predecessorPins = [ordered]@{
        "numpy"            = "2.5.2"
        "pandas"           = "3.0.5"
        "pip"              = "25.0.1"
        "python-dateutil"  = "2.9.0.post0"
        "six"              = "1.17.0"
        "tzdata"           = "2026.3"
        "xlrd"             = "2.0.2"
    }

    Write-Host "== V9_014 PDF environment successor -- Stage E5 staging resolution (NOT executed at E2) =="
    Write-Host "Target: a single NON-CANONICAL staging venv under -StagingRoot. '.venv-real-execution' and '.venv' are never touched."

    # ------------------------------------------------------------------
    # Preflight 1: repo/branch/exact-HEAD/clean-tree verification.
    # ------------------------------------------------------------------
    $currentBranch = (& git rev-parse --abbrev-ref HEAD).Trim()
    if ($currentBranch -ne $authoritativeBranch) {
        throw "PRE_GATE_BRANCH_MISMATCH: expected '$authoritativeBranch', found '$currentBranch'."
    }
    $currentHead = (& git rev-parse HEAD).Trim()
    if ($currentHead -ne $ExpectedHead) {
        throw "PRE_GATE_EXPECTED_HEAD_MISMATCH: expected '$ExpectedHead', found '$currentHead'."
    }
    $porcelainStatus = (& git status --porcelain)
    if ($porcelainStatus) {
        throw "PRE_GATE_DIRTY_WORKING_TREE: working tree is not clean; refusing to resolve from an unreviewed diff."
    }

    # ------------------------------------------------------------------
    # Preflight 2: predecessor artifact provenance -- the reviewed
    # predecessor lock's exact bytes, verified via canonical Git blob
    # bytes (never the working-tree checkout, which could carry CRLF
    # conversion), must match the hardcoded reviewed hash exactly.
    # ------------------------------------------------------------------
    function Get-GitBlobSha256 {
        param([Parameter(Mandatory = $true)][string]$GitRef)
        $processStartInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processStartInfo.FileName = "git"
        $processStartInfo.Arguments = "cat-file blob $GitRef"
        $processStartInfo.RedirectStandardOutput = $true
        $processStartInfo.RedirectStandardError = $true
        $processStartInfo.UseShellExecute = $false
        $processStartInfo.CreateNoWindow = $true
        $gitProcess = [System.Diagnostics.Process]::Start($processStartInfo)
        $memoryStream = New-Object System.IO.MemoryStream
        $gitProcess.StandardOutput.BaseStream.CopyTo($memoryStream)
        $gitProcess.WaitForExit()
        if ($gitProcess.ExitCode -ne 0) {
            throw "PRE_GATE_GIT_BLOB_READ_FAILURE: 'git cat-file blob $GitRef' failed."
        }
        $blobBytes = $memoryStream.ToArray()
        $sha256Provider = [System.Security.Cryptography.SHA256]::Create()
        $hashBytes = $sha256Provider.ComputeHash($blobBytes)
        return [System.BitConverter]::ToString($hashBytes).Replace("-", "").ToLowerInvariant()
    }

    $predecessorLockBlobSha256 = Get-GitBlobSha256 -GitRef "${ExpectedHead}:${predecessorLockRelativePath}"
    if ($predecessorLockBlobSha256 -ne $reviewedPredecessorLockSha256) {
        throw "PRE_GATE_PREDECESSOR_LOCK_PROVENANCE_MISMATCH: expected '$reviewedPredecessorLockSha256', found '$predecessorLockBlobSha256'."
    }

    # ------------------------------------------------------------------
    # Preflight 3: canonical Windows/CPython prerequisite.
    # ------------------------------------------------------------------
    $isWindowsHost = ($env:OS -eq "Windows_NT") -or ([System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT)
    if (-not $isWindowsHost) {
        throw "PRE_GATE_NON_WINDOWS_HOST: this staging runner requires $canonicalPlatformSystem."
    }
    $baseInterpreterVersion = (& python --version) 2>&1 | Out-String
    if ($baseInterpreterVersion -notmatch [regex]::Escape($canonicalPythonExactVersion)) {
        throw "PRE_GATE_NON_CANONICAL_PYTHON: expected CPython $canonicalPythonExactVersion on PATH, found '$($baseInterpreterVersion.Trim())'."
    }

    # ------------------------------------------------------------------
    # Preflight 4: staging-path collision check. A pre-existing
    # -StagingRoot is a fail-closed STOP, never a deletion/reset -- this
    # file contains no `Remove-Item` call anywhere.
    # ------------------------------------------------------------------
    if (-not [System.IO.Path]::IsPathRooted($StagingRoot)) {
        throw "PRE_GATE_STAGING_PATH_NOT_ABSOLUTE: -StagingRoot must be an absolute path."
    }
    $stagingLeafName = Split-Path -Path $StagingRoot -Leaf
    if ($stagingLeafName -eq $canonicalEnvironmentDirectoryName -or $stagingLeafName -eq $generalEnvironmentDirectoryName) {
        throw "PRE_GATE_STAGING_PATH_MATCHES_RESERVED_NAME: -StagingRoot must not be named '$canonicalEnvironmentDirectoryName' or '$generalEnvironmentDirectoryName'."
    }
    $repoRoot = (& git rev-parse --show-toplevel).Trim()
    $resolvedStagingRoot = [System.IO.Path]::GetFullPath($StagingRoot)
    $resolvedRepoRoot = [System.IO.Path]::GetFullPath($repoRoot)
    if ($resolvedStagingRoot.StartsWith($resolvedRepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_STAGING_PATH_INSIDE_REPO: -StagingRoot must be outside the repository tree."
    }
    if (Test-Path -LiteralPath $StagingRoot) {
        throw "PRE_GATE_STAGING_PATH_ALREADY_EXISTS: -StagingRoot already exists; refusing to reuse, reset, or delete it."
    }
    if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
        throw "PRE_GATE_OUTPUT_PATH_NOT_ABSOLUTE: -OutputRoot must be an absolute path."
    }

    Write-Host "All preflight checks passed. Proceeding with the frozen, single-attempt E5 sequence."

    # ==================================================================
    # Frozen Stage E5 sequence (Section 5, Stage E5). Fixed in source;
    # never reordered, retried, or repaired at execution time.
    # ==================================================================

    # E5 step 1: create a fresh staging venv under -StagingRoot only.
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    & python -m venv $StagingRoot
    if ($LASTEXITCODE -ne 0) {
        throw "E5_STEP1_STAGING_VENV_CREATION_FAILED: exit code $LASTEXITCODE."
    }
    $stagingInterpreter = Join-Path $StagingRoot "Scripts\python.exe"

    # E5 step 2: install the exact reviewed predecessor lock, --no-deps,
    # into the staging venv only.
    & $stagingInterpreter -m pip install --no-deps -r $predecessorLockRelativePath
    $predecessorInstallExitCode = $LASTEXITCODE
    if ($predecessorInstallExitCode -ne 0) {
        throw "E5_STEP2_PREDECESSOR_LOCK_INSTALL_FAILED: exit code $predecessorInstallExitCode. No retry."
    }

    # E5 step 3: capture the BEFORE `pip freeze --all` baseline and
    # require exactly the 7 predecessor pins.
    $beforeFreezeText = (& $stagingInterpreter -m pip freeze --all) | Out-String
    $beforeFreezePath = Join-Path $OutputRoot "before_freeze.txt"
    Set-Content -LiteralPath $beforeFreezePath -Value $beforeFreezeText -NoNewline -Encoding utf8

    # E5 step 4: exactly ONE successor resolution command, constrained
    # to the reviewed predecessor lock plus the reviewed direct spec.
    # This is the single resolution attempt this file ever performs --
    # there is no loop, no retry wrapper, and no alternate invocation
    # anywhere else in this file.
    $successorStdoutPath = Join-Path $OutputRoot "successor_resolution_stdout.txt"
    $successorStderrPath = Join-Path $OutputRoot "successor_resolution_stderr.txt"
    & $stagingInterpreter -m pip install `
        -c $predecessorLockRelativePath `
        -r $directSpecRelativePath `
        1> $successorStdoutPath 2> $successorStderrPath
    $successorResolutionExitCode = $LASTEXITCODE

    # E5 step 5: exit code is captured and preserved -- a nonzero exit is
    # NOT retried and NOT repaired; it proceeds directly to step 6/7 with
    # the failure evidence intact.
    $exitCodePath = Join-Path $OutputRoot "successor_resolution_exit_code.txt"
    Set-Content -LiteralPath $exitCodePath -Value "$successorResolutionExitCode" -NoNewline -Encoding utf8

    # E5 step 6: capture the AFTER `pip freeze --all` snapshot,
    # regardless of the resolution command's exit code.
    $afterFreezeText = (& $stagingInterpreter -m pip freeze --all) | Out-String
    $afterFreezePath = Join-Path $OutputRoot "after_freeze.txt"
    Set-Content -LiteralPath $afterFreezePath -Value $afterFreezeText -NoNewline -Encoding utf8

    # E5 step 7: no retry, no repair, no second resolution attempt. This
    # runner's job ends here; Stage E6's separately reviewed NO-NETWORK
    # inspection consumes the captured evidence above.
    Write-Host "Staging resolution complete. successor_resolution_exit_code=$successorResolutionExitCode"
    Write-Host "Evidence written under: $OutputRoot"
    Write-Host "'.venv-real-execution' and '.venv' were not touched by this run."
}
