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
#     code) ONLY under the caller-supplied -OutputRoot, created only
#     after every -StagingRoot AND -OutputRoot preflight check has
#     passed, and never via "-Force" (a previous evidence directory is
#     NEVER reused: -OutputRoot must not already exist, must not alias
#     or overlap -StagingRoot in either direction, must not be inside
#     the repository, and must not be, or be nested inside, either
#     reserved environment -- re-verified against both paths' COMPLETE
#     existing-ancestor chains, from the nearest existing ancestor all
#     the way to the filesystem root, never merely the nearest one; ANY
#     symlink/junction/reparse point anywhere in either chain is
#     rejected outright -- never followed and continued through);
#   - NEVER creates, writes to, deletes, or resets
#     ".venv-real-execution" (the canonical protected environment) or
#     ".venv" (the separate general project environment) -- neither
#     directory name is ever a write target anywhere in this file;
#   - performs NO PDF/source acquisition of any kind;
#   - mechanically enforces, via the reviewed Python semantic authority
#     in scripts\v9_014_pdf_env_successor.py
#     (`parse_pip_freeze_all` / `validate_predecessor_baseline`, through
#     its own `--validate-predecessor-baseline-file` CLI -- never a
#     second, independently invented comparison), that the captured
#     BEFORE freeze snapshot shows EXACTLY the 7 frozen predecessor
#     pins BEFORE the successor-resolution command is ever reached;
#   - performs exactly ONE successor-resolution pip command -- no retry,
#     no repair, no second resolution attempt, regardless of exit code;
#   - never deletes anything (this file contains no `Remove-Item` call
#     anywhere): a staging-path collision, an output-path collision, or
#     any other precondition failure, is a fail-closed `throw`, never a
#     deletion/reset;
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

    # ------------------------------------------------------------------
    # Shared helper: resolve the nearest EXISTING ancestor of a path,
    # under a CONSERVATIVE fail-closed policy. Finding the nearest
    # existing ancestor alone is not sufficient: an ordinary directory
    # can sit directly beneath a HIGHER symlink/junction/reparse point
    # (e.g. an external alias whose immediate child is a perfectly normal
    # directory, itself containing the not-yet-existing StagingRoot/
    # OutputRoot leaf) -- inspecting only the nearest existing ancestor
    # would silently miss that higher alias entirely. This helper instead
    # walks EVERY existing ancestor, from the nearest one up to the
    # filesystem root, and fails closed the instant ANY of them is a
    # reparse point of any kind. It never follows an alias's target and
    # "continues through" it optimistically: a detected reparse point
    # anywhere in the existing ancestor chain is CHATGPT_DECISION_REQUIRED,
    # full stop -- as is any failure to inspect an ancestor's attributes,
    # or finding no existing ancestor at all. Only once the ENTIRE
    # existing ancestor chain is proven reparse-point-free does this
    # return the nearest existing ancestor's own (unaliased) real path,
    # with the not-yet-existing suffix reattached, for the caller's
    # lexical repo/reserved/overlap checks to rely on. Used for BOTH
    # -StagingRoot and -OutputRoot, since both call this same helper.
    # ------------------------------------------------------------------
    function Resolve-ExistingAncestorRealPath {
        param([Parameter(Mandatory = $true)][string]$Path)
        $current = $Path
        $suffixParts = @()
        while (-not (Test-Path -LiteralPath $current)) {
            $suffixParts = ,(Split-Path -Path $current -Leaf) + $suffixParts
            $parent = Split-Path -Path $current -Parent
            if ([string]::IsNullOrEmpty($parent) -or $parent -eq $current) {
                throw "PRE_GATE_PATH_SAFETY_UNDETERMINED_CHATGPT_DECISION_REQUIRED: no existing ancestor found for '$Path'."
            }
            $current = $parent
        }

        # Walk the COMPLETE existing ancestor chain -- $current (the
        # nearest existing ancestor) and every existing directory above
        # it, all the way to the filesystem root -- inspecting each for
        # a reparse point. This never stops at the first existing
        # ancestor the way plain path resolution would.
        $nearestExistingItem = $null
        $chainNode = $current
        while ($true) {
            try {
                $chainItem = Get-Item -LiteralPath $chainNode -Force -ErrorAction Stop
            } catch {
                throw "PRE_GATE_PATH_SAFETY_UNDETERMINED_CHATGPT_DECISION_REQUIRED: unable to inspect existing ancestor '$chainNode' of '$Path': $($_.Exception.Message)"
            }
            if ($null -eq $nearestExistingItem) {
                $nearestExistingItem = $chainItem
            }
            if ($chainItem.LinkType -or ($chainItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
                throw "PRE_GATE_PATH_REPARSE_ANCESTOR_CHATGPT_DECISION_REQUIRED: existing ancestor '$chainNode' of '$Path' is a symlink/junction/reparse point; refusing to resolve through it."
            }
            $chainParent = Split-Path -Path $chainNode -Parent
            if ([string]::IsNullOrEmpty($chainParent) -or $chainParent -eq $chainNode) {
                break
            }
            $chainNode = $chainParent
        }

        $resolvedExisting = $nearestExistingItem.FullName
        foreach ($part in $suffixParts) {
            $resolvedExisting = Join-Path $resolvedExisting $part
        }
        return $resolvedExisting
    }

    # ------------------------------------------------------------------
    # Preflight 4b: StagingRoot REAL-PATH protected-state guard,
    # performed BEFORE any OutputRoot creation, staging venv creation,
    # pip invocation, or filesystem mutation of any kind. The lexical
    # checks above (leaf name, lexical inside-repo) are NOT sufficient by
    # themselves: an external parent symlink/junction whose REAL target
    # lands inside the repository, inside ".venv-real-execution", or
    # inside ".venv" must be rejected even though -StagingRoot itself
    # does not yet exist and its own leaf name looks unremarkable.
    # ------------------------------------------------------------------
    $realResolvedRepoRoot = Resolve-ExistingAncestorRealPath -Path $repoRoot
    $canonicalEnvironmentAbsolutePath = Join-Path $repoRoot $canonicalEnvironmentDirectoryName
    $generalEnvironmentAbsolutePath = Join-Path $repoRoot $generalEnvironmentDirectoryName
    $realResolvedCanonicalEnvironmentPath = Resolve-ExistingAncestorRealPath -Path $canonicalEnvironmentAbsolutePath
    $realResolvedGeneralEnvironmentPath = Resolve-ExistingAncestorRealPath -Path $generalEnvironmentAbsolutePath
    $realResolvedStagingRoot = Resolve-ExistingAncestorRealPath -Path $StagingRoot

    if ($realResolvedStagingRoot.StartsWith($realResolvedRepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_STAGING_PATH_REALPATH_INSIDE_REPO: -StagingRoot resolves inside the repository tree through an existing parent alias."
    }
    if ($realResolvedStagingRoot -eq $realResolvedCanonicalEnvironmentPath -or
        $realResolvedStagingRoot.StartsWith($realResolvedCanonicalEnvironmentPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_STAGING_PATH_REALPATH_INSIDE_CANONICAL_ENVIRONMENT: -StagingRoot resolves to, or inside, the canonical protected environment through an existing parent alias."
    }
    if ($realResolvedStagingRoot -eq $realResolvedGeneralEnvironmentPath -or
        $realResolvedStagingRoot.StartsWith($realResolvedGeneralEnvironmentPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_STAGING_PATH_REALPATH_INSIDE_GENERAL_ENVIRONMENT: -StagingRoot resolves to, or inside, the general project environment through an existing parent alias."
    }

    # ------------------------------------------------------------------
    # Preflight 5: OutputRoot fail-closed validation, performed BEFORE
    # any filesystem creation/write of any kind. A previous evidence
    # directory is NEVER reused: OutputRoot must not already exist, and
    # -OutputRoot must not alias or overlap -StagingRoot, the repository,
    # or either reserved environment in any direction.
    # ------------------------------------------------------------------
    if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
        throw "PRE_GATE_OUTPUT_PATH_NOT_ABSOLUTE: -OutputRoot must be an absolute path."
    }
    $resolvedOutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
    if ($resolvedOutputRoot.StartsWith($resolvedRepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_OUTPUT_PATH_INSIDE_REPO: -OutputRoot must be outside the repository tree."
    }
    $outputLeafName = Split-Path -Path $OutputRoot -Leaf
    if ($outputLeafName -eq $canonicalEnvironmentDirectoryName) {
        throw "PRE_GATE_OUTPUT_PATH_MATCHES_CANONICAL_NAME: -OutputRoot must not be named '$canonicalEnvironmentDirectoryName'."
    }
    if ($outputLeafName -eq $generalEnvironmentDirectoryName) {
        throw "PRE_GATE_OUTPUT_PATH_MATCHES_GENERAL_NAME: -OutputRoot must not be named '$generalEnvironmentDirectoryName'."
    }
    $outputPathSegments = $resolvedOutputRoot.Split([System.IO.Path]::DirectorySeparatorChar, [System.IO.Path]::AltDirectorySeparatorChar)
    if ($outputPathSegments -contains $canonicalEnvironmentDirectoryName -or $outputPathSegments -contains $generalEnvironmentDirectoryName) {
        throw "PRE_GATE_OUTPUT_PATH_INSIDE_RESERVED_ENVIRONMENT: -OutputRoot must not be nested inside '$canonicalEnvironmentDirectoryName' or '$generalEnvironmentDirectoryName'."
    }
    if ($resolvedOutputRoot -eq $resolvedStagingRoot) {
        throw "PRE_GATE_OUTPUT_PATH_EQUALS_STAGING_ROOT: -OutputRoot must not equal -StagingRoot."
    }
    if ($resolvedStagingRoot.StartsWith($resolvedOutputRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_OUTPUT_PATH_ANCESTOR_OF_STAGING_ROOT: -OutputRoot must not be an ancestor of -StagingRoot."
    }
    if ($resolvedOutputRoot.StartsWith($resolvedStagingRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_OUTPUT_PATH_DESCENDANT_OF_STAGING_ROOT: -OutputRoot must not be a descendant of -StagingRoot."
    }
    if (Test-Path -LiteralPath $OutputRoot) {
        throw "PRE_GATE_OUTPUT_PATH_ALREADY_EXISTS: -OutputRoot already exists; a previous evidence directory is never reused."
    }

    # Re-verify every OutputRoot overlap check above against REAL resolved
    # paths (unconditionally -- not merely when a real path happens to
    # differ from its lexical form, since relying on that comparison to
    # decide whether to even look is itself a way uncertainty could be
    # waved through). -StagingRoot's own real-path repo/reserved-
    # environment safety was already fully established in Preflight 4b
    # above; this block only needs to re-check REAL OutputRoot against
    # the repo, the reserved environments, and REAL StagingRoot.
    $realResolvedOutputRoot = Resolve-ExistingAncestorRealPath -Path $OutputRoot
    if ($realResolvedOutputRoot.StartsWith($realResolvedRepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_OUTPUT_PATH_REALPATH_INSIDE_REPO: -OutputRoot resolves inside the repository tree through an existing parent alias."
    }
    if ($realResolvedOutputRoot -eq $realResolvedCanonicalEnvironmentPath -or
        $realResolvedOutputRoot.StartsWith($realResolvedCanonicalEnvironmentPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_OUTPUT_PATH_REALPATH_INSIDE_CANONICAL_ENVIRONMENT: -OutputRoot resolves to, or inside, the canonical protected environment through an existing parent alias."
    }
    if ($realResolvedOutputRoot -eq $realResolvedGeneralEnvironmentPath -or
        $realResolvedOutputRoot.StartsWith($realResolvedGeneralEnvironmentPath + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_OUTPUT_PATH_REALPATH_INSIDE_GENERAL_ENVIRONMENT: -OutputRoot resolves to, or inside, the general project environment through an existing parent alias."
    }
    if ($realResolvedOutputRoot -eq $realResolvedStagingRoot) {
        throw "PRE_GATE_OUTPUT_PATH_EQUALS_STAGING_ROOT: -OutputRoot resolves to the same real path as -StagingRoot through an existing parent alias."
    }
    if ($realResolvedStagingRoot.StartsWith($realResolvedOutputRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_OUTPUT_PATH_ANCESTOR_OF_STAGING_ROOT: -OutputRoot resolves as an ancestor of -StagingRoot through an existing parent alias."
    }
    if ($realResolvedOutputRoot.StartsWith($realResolvedStagingRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "PRE_GATE_OUTPUT_PATH_DESCENDANT_OF_STAGING_ROOT: -OutputRoot resolves as a descendant of -StagingRoot through an existing parent alias."
    }

    Write-Host "All preflight checks passed, including StagingRoot and OutputRoot real-path fail-closed validation. Proceeding with the frozen, single-attempt E5 sequence."

    # ------------------------------------------------------------------
    # Native process capture helper: Windows PowerShell 5.1 can turn
    # native stderr records from `&` into a terminating NativeCommandError
    # while $ErrorActionPreference is Stop, even when the caller intends
    # to preserve the stream and inspect the process exit code. Invoke the
    # successor resolver through System.Diagnostics.Process instead. Both
    # redirected BaseStreams are drained concurrently so a verbose child
    # cannot deadlock on a full stdout or stderr pipe; the byte streams are
    # copied directly to their evidence files without PowerShell decoding
    # or rewriting them.
    # ------------------------------------------------------------------
    function Invoke-NativeProcessRawCapture {
        param(
            [Parameter(Mandatory = $true)][string]$FileName,
            [Parameter(Mandatory = $true)][string]$Arguments,
            [Parameter(Mandatory = $true)][string]$WorkingDirectory,
            [Parameter(Mandatory = $true)][string]$StdoutPath,
            [Parameter(Mandatory = $true)][string]$StderrPath
        )

        $processStartInfo = New-Object System.Diagnostics.ProcessStartInfo
        $processStartInfo.FileName = $FileName
        $processStartInfo.Arguments = $Arguments
        $processStartInfo.WorkingDirectory = $WorkingDirectory
        $processStartInfo.RedirectStandardOutput = $true
        $processStartInfo.RedirectStandardError = $true
        $processStartInfo.UseShellExecute = $false
        $processStartInfo.CreateNoWindow = $true

        $nativeProcess = New-Object System.Diagnostics.Process
        $nativeProcess.StartInfo = $processStartInfo
        $stdoutFileStream = $null
        $stderrFileStream = $null
        try {
            # OutputRoot was created only after its no-overwrite preflight;
            # CreateNew preserves that collision discipline for the two
            # stream evidence files as well.
            $stdoutFileStream = New-Object System.IO.FileStream(
                $StdoutPath,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::None
            )
            $stderrFileStream = New-Object System.IO.FileStream(
                $StderrPath,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::None
            )

            if (-not $nativeProcess.Start()) {
                throw "E5_STEP4_SUCCESSOR_PROCESS_START_FAILED: native successor process did not start."
            }

            # Start both copies before waiting for the process. Sequentially
            # draining stdout then stderr can deadlock when either pipe fills.
            $stdoutCopyTask = $nativeProcess.StandardOutput.BaseStream.CopyToAsync($stdoutFileStream)
            $stderrCopyTask = $nativeProcess.StandardError.BaseStream.CopyToAsync($stderrFileStream)
            $nativeProcess.WaitForExit()
            $capturedNativeExitCode = $nativeProcess.ExitCode
            $copyTasks = [System.Threading.Tasks.Task[]]@($stdoutCopyTask, $stderrCopyTask)
            [System.Threading.Tasks.Task]::WaitAll($copyTasks)
            $stdoutFileStream.Flush()
            $stderrFileStream.Flush()
            return $capturedNativeExitCode
        }
        finally {
            if ($null -ne $stdoutFileStream) {
                $stdoutFileStream.Dispose()
            }
            if ($null -ne $stderrFileStream) {
                $stderrFileStream.Dispose()
            }
            $nativeProcess.Dispose()
        }
    }

    # ==================================================================
    # Frozen Stage E5 sequence (Section 5, Stage E5). Fixed in source;
    # never reordered, retried, or repaired at execution time.
    # ==================================================================

    # OutputRoot is created ONLY now, after every StagingRoot and
    # OutputRoot preflight check above has passed -- never with "-Force"
    # (which would silently permit reusing/overwriting an existing
    # directory). If OutputRoot were somehow created by something else
    # between the Preflight 5 check and this line, New-Item without
    # -Force fails closed here rather than silently reusing it.
    New-Item -ItemType Directory -Path $OutputRoot | Out-Null

    # E5 step 1: create a fresh staging venv under -StagingRoot only.
    & python -m venv $StagingRoot
    if ($LASTEXITCODE -ne 0) {
        throw "E5_STEP1_STAGING_VENV_CREATION_FAILED: exit code $LASTEXITCODE."
    }
    $stagingInterpreter = Join-Path $StagingRoot "Scripts\python.exe"
    $offlineToolingPath = Join-Path (Get-Location) "scripts\v9_014_pdf_env_successor.py"

    # E5 step 2: install the exact reviewed predecessor lock, --no-deps,
    # into the staging venv only.
    & $stagingInterpreter -m pip install --no-deps -r $predecessorLockRelativePath
    $predecessorInstallExitCode = $LASTEXITCODE
    if ($predecessorInstallExitCode -ne 0) {
        throw "E5_STEP2_PREDECESSOR_LOCK_INSTALL_FAILED: exit code $predecessorInstallExitCode. No retry."
    }

    # E5 step 3: capture the BEFORE `pip freeze --all` baseline.
    $beforeFreezeText = (& $stagingInterpreter -m pip freeze --all) | Out-String
    $beforeFreezePath = Join-Path $OutputRoot "before_freeze.txt"
    Set-Content -LiteralPath $beforeFreezePath -Value $beforeFreezeText -NoNewline -Encoding utf8

    # E5 step 3 (mechanical enforcement): the captured BEFORE baseline
    # MUST be validated, via the reviewed Python semantic authority
    # (`parse_pip_freeze_all` / `validate_predecessor_baseline`, invoked
    # through this module's own `--validate-predecessor-baseline-file`
    # CLI -- never a second, independently invented comparison such as
    # grepping for seven package-name strings), to show EXACTLY the 7
    # frozen predecessor pins, BEFORE the successor-resolution command
    # below is ever reached. This uses the staging interpreter itself
    # (already confirmed CPython 3.12.10 via Preflight 3), performs no
    # network access and no environment mutation, and never imports
    # `pdfplumber`.
    & $stagingInterpreter $offlineToolingPath --validate-predecessor-baseline-file $beforeFreezePath
    $baselineValidationExitCode = $LASTEXITCODE
    if ($baselineValidationExitCode -ne 0) {
        throw "E5_STEP3_PREDECESSOR_BASELINE_VALIDATION_FAILED: exit code $baselineValidationExitCode. The successor-resolution command below is NOT executed."
    }

    # E5 step 4: exactly ONE successor resolution command, constrained
    # to the reviewed predecessor lock plus the reviewed direct spec, and
    # reached ONLY after E5 step 3's baseline validation above exited 0.
    # This is the single resolution attempt this file ever performs --
    # there is no loop, no retry wrapper, and no alternate invocation
    # anywhere else in this file.
    $successorStdoutPath = Join-Path $OutputRoot "successor_resolution_stdout.txt"
    $successorStderrPath = Join-Path $OutputRoot "successor_resolution_stderr.txt"
    $successorResolutionArguments = "-m pip install -c `"$predecessorLockRelativePath`" -r `"$directSpecRelativePath`""
    $successorResolutionExitCode = Invoke-NativeProcessRawCapture `
        -FileName $stagingInterpreter `
        -Arguments $successorResolutionArguments `
        -WorkingDirectory ((Get-Location).Path) `
        -StdoutPath $successorStdoutPath `
        -StderrPath $successorStderrPath

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
