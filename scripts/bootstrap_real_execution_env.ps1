# Real-execution Python environment bootstrap (environment setup stage ONLY).
#
# See REAL_EXECUTION_PYTHON_ENVIRONMENT.md for the human-readable contract
# and AI_REAL_EXECUTION_RUNBOOK.md SS15-19 for where this fits in the overall
# pre-authorization ordering.
#
# This script exclusively creates/verifies the CANONICAL_PROTECTED_REAL_
# EXECUTION_ENVIRONMENT, ".venv-real-execution". It never touches the
# repository's separate, existing ".venv"
# (GENERAL_PROJECT_ENVIRONMENT_NOT_AUTHORIZED_FOR_PROTECTED_EXECUTION, used
# for ordinary project development and the unrelated daily trading bot):
# this script does not read, alter, uninstall from, or copy packages out of
# ".venv" anywhere below.
#
# REAL_EXECUTION_ENVIRONMENT_LOCK_ENFORCEMENT: the reviewed
# "requirements-real-execution.lock.txt" is now the protected installation
# authority, not the unpinned "requirements-real-execution.txt". Protected
# packages are resolved and installed ONLY from the reviewed lock, with
# --no-deps, so pip cannot silently add anything outside the complete
# reviewed lock. Before any protected package installation, this script
# fails closed unless the reviewed lock candidate manifest
# (REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json), the lock file's own
# hash, the source requirements' canonical Git provenance, the canonical
# environment identity, and the live platform binding are all exactly the
# reviewed values -- hardcoded below, not merely trusted from whatever
# those mutable files currently say on disk.
#
# This script does NOT:
#   - consume any human research gate
#   - call JPX/Yahoo or any other real production network host
#   - access private/sealed data
#   - execute any V8I/V8J real acquisition
#   - touch the general ".venv" in any way
#
# The only network activity this script performs is ordinary PyPI package
# installation (pip), which is standard software-supply-chain activity, not
# the "real Yahoo/JPX/broker/production network" AI_REAL_EXECUTION_RUNBOOK.md
# is scoped to.
#
# Run this file directly (a single reviewed .ps1), per
# AI_REAL_EXECUTION_RUNBOOK.md SS1 ("one atomic `& { ... }` block, or one
# reviewed `.ps1` file"). Do not paste its body as independent line-by-line
# snippets.

& {
    $ErrorActionPreference = "Stop"

    $expectedRepositoryMarkerFiles = @(
        "src\v8i_source_snapshot.py",
        "requirement.txt",
        "requirements-real-execution.txt",
        "requirements-real-execution.lock.txt",
        "REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json",
        "AI_REAL_EXECUTION_RUNBOOK.md"
    )
    $canonicalPythonMajorMinor = "3.12"
    $canonicalPythonExactVersion = "3.12.10"
    $canonicalVenvDirectory = Join-Path (Get-Location) ".venv-real-execution"
    $canonicalInterpreterPath = Join-Path $canonicalVenvDirectory "Scripts\python.exe"
    $realExecutionLockPath = Join-Path (Get-Location) "requirements-real-execution.lock.txt"
    $lockCandidatePath = Join-Path (Get-Location) "REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json"
    $readinessCheckerPath = Join-Path (Get-Location) "scripts\check_real_execution_env.py"

    # Reviewed binding (REAL_EXECUTION_ENVIRONMENT_LOCK_ENFORCEMENT task) --
    # hardcoded here, not merely trusted from the mutable candidate/lock
    # files, so a tampered or stale file is independently detectable
    # before any protected package installation.
    $reviewedSourceGitSha = "b74e0f787599475cd9fe719d254202dc9bfc14d5"
    $reviewedLockSha256 = "b5c063a1cca585fa100fdc0027d6cdbf4ef33ef5a7fe614230599fb882b51f96"
    $reviewedSourceRequirementsGitSha256 = "2cdcfd7a87023c4e9c3ec463cf16f77d88f72ccc8d1f0e5de242e6c68b0cf601"
    $reviewedFixtureSha256 = "ca47744896a286e1c56d4d0c09260775772c7df0c01b80d81b7e9a515e6d6aa7"
    $reviewedPackageCount = 7
    $reviewedArtifactStatus = "CANDIDATE_NOT_FROZEN"
    $expectedCanonicalEnvironmentDirectory = ".venv-real-execution"
    $expectedCanonicalInterpreter = ".venv-real-execution\Scripts\python.exe"

    Write-Host "== Real-execution environment bootstrap (environment setup stage only) =="
    Write-Host "Target: CANONICAL_PROTECTED_REAL_EXECUTION_ENVIRONMENT = .venv-real-execution"
    Write-Host "This script never touches the separate general '.venv' project environment."
    Write-Host "Protected installation authority: requirements-real-execution.lock.txt (reviewed, exact-pinned)."

    # ------------------------------------------------------------------
    # Helper: exact Git blob bytes for "<sha>:<path>", captured via the
    # raw .NET process stdout byte stream -- NEVER through PowerShell's
    # text/console pipeline, which can otherwise reintroduce or lose
    # line-ending bytes. This is the canonical-Git-bytes, line-ending-
    # independent provenance mechanism: it bypasses the working-tree
    # checkout (and any CRLF conversion it applies) entirely.
    # ------------------------------------------------------------------
    function Get-GitBlobSha256 {
        param(
            [Parameter(Mandatory = $true)][string]$GitRef
        )
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
            throw "PRE_GATE_ENVIRONMENT_BLOCK: 'git cat-file blob $GitRef' failed -- cannot establish canonical Git-bytes provenance."
        }
        $blobBytes = $memoryStream.ToArray()
        $sha256Provider = [System.Security.Cryptography.SHA256]::Create()
        $hashBytes = $sha256Provider.ComputeHash($blobBytes)
        return [System.BitConverter]::ToString($hashBytes).Replace("-", "").ToLowerInvariant()
    }

    # ------------------------------------------------------------------
    # 1. Verify correct repository root.
    # ------------------------------------------------------------------
    foreach ($markerFile in $expectedRepositoryMarkerFiles) {
        $markerPath = Join-Path (Get-Location) $markerFile
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: repository marker file missing ($markerFile). Run this script from the repository root."
        }
    }
    Write-Host "Repository root verified (expected marker files present)."

    # ------------------------------------------------------------------
    # 2. Verify required base Python launcher/version. Exact 3.12.10 is
    #    required -- not merely 3.12 -- because the reviewed lock binds to
    #    the exact patch version.
    # ------------------------------------------------------------------
    $baseLauncherCommand = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -eq $baseLauncherCommand) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: the Python launcher (py) was not found. Install the official Python launcher for Windows before bootstrapping."
    }
    $baseLauncherVersionOutput = & py "-$canonicalPythonMajorMinor" -c "import sys; print('.'.join(str(part) for part in sys.version_info[:3]))"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($baseLauncherVersionOutput)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: py -$canonicalPythonMajorMinor did not resolve to a working interpreter. Install Python $canonicalPythonExactVersion (matching the reviewed environment lock) before bootstrapping."
    }
    $resolvedBaseVersion = $baseLauncherVersionOutput.Trim()
    if ($resolvedBaseVersion -ne $canonicalPythonExactVersion) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: py -$canonicalPythonMajorMinor resolved to Python $resolvedBaseVersion, expected exactly $canonicalPythonExactVersion (the reviewed environment lock's exact bound patch version). Install exactly Python $canonicalPythonExactVersion before bootstrapping."
    }
    Write-Host "Base launcher resolved: py -$canonicalPythonMajorMinor -> Python $resolvedBaseVersion (exact match)."

    # ------------------------------------------------------------------
    # 3/4. Create .venv-real-execution only if absent; if present, verify
    #      it belongs to this repository and the EXACT expected Python
    #      patch version rather than silently recreating it. Mismatch =>
    #      STOP, no auto-delete. The separate general ".venv" is never
    #      inspected, read from, or written to by this step or any other
    #      step below.
    # ------------------------------------------------------------------
    if (-not (Test-Path -LiteralPath $canonicalVenvDirectory -PathType Container)) {
        Write-Host "No existing .venv-real-execution found; creating one with py -$canonicalPythonMajorMinor ..."
        & py "-$canonicalPythonMajorMinor" -m venv $canonicalVenvDirectory
        if ($LASTEXITCODE -ne 0) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: 'py -$canonicalPythonMajorMinor -m venv' failed while creating .venv-real-execution."
        }
        Write-Host ".venv-real-execution created."
    }
    else {
        Write-Host "Existing .venv-real-execution found; verifying it (never auto-recreating on mismatch) ..."
        if (-not (Test-Path -LiteralPath $canonicalInterpreterPath -PathType Leaf)) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: .venv-real-execution exists but $canonicalInterpreterPath is missing. This does not look like a valid venv for this repository. Remediate manually; this script will not delete or recreate it automatically."
        }
        $existingVenvVersionOutput = & $canonicalInterpreterPath -c "import sys; print('.'.join(str(part) for part in sys.version_info[:3]))"
        if ($LASTEXITCODE -ne 0) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: existing .venv-real-execution interpreter at $canonicalInterpreterPath failed to run. Remediate manually; this script will not delete or recreate it automatically."
        }
        $existingVenvExactVersion = $existingVenvVersionOutput.Trim()
        if ($existingVenvExactVersion -ne $canonicalPythonExactVersion) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: existing .venv-real-execution is Python $existingVenvExactVersion, expected exactly $canonicalPythonExactVersion. This script will not delete or recreate a mismatched .venv-real-execution automatically -- remediate manually (e.g. remove .venv-real-execution yourself after confirming nothing depends on it, then re-run)."
        }
        Write-Host "Existing .venv-real-execution verified: Python $existingVenvExactVersion (exact match) at $canonicalInterpreterPath"
    }

    if (-not (Test-Path -LiteralPath $canonicalInterpreterPath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: canonical interpreter not found at $canonicalInterpreterPath after venv creation/verification."
    }

    # ------------------------------------------------------------------
    # 5. Reviewed-lock preflight. Fail closed BEFORE any protected package
    #    installation if the reviewed lock candidate manifest, the lock
    #    hash, the source-requirements Git provenance, the canonical
    #    environment identity, or the platform binding is invalid.
    # ------------------------------------------------------------------
    Write-Host "Running reviewed-lock preflight (before any protected package installation) ..."

    if (-not (Test-Path -LiteralPath $lockCandidatePath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json not found at $lockCandidatePath."
    }
    if (-not (Test-Path -LiteralPath $realExecutionLockPath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: requirements-real-execution.lock.txt not found at $realExecutionLockPath."
    }

    $lockCandidateJson = Get-Content -LiteralPath $lockCandidatePath -Raw | ConvertFrom-Json

    $requiredCandidateFields = @(
        "artifact_status", "canonical_environment_directory", "canonical_interpreter",
        "future_protected_execution_authorized", "gpt_exact_sha_independent_review_required",
        "pip_version", "private_or_sealed_reads", "python", "real_execution_environment_frozen",
        "real_execution_environment_ready_observed", "real_network_requests_to_protected_hosts",
        "requirements_real_execution", "research_gates_consumed", "resolved_lock", "schema_version",
        "source_git_sha", "synthetic_xls_fixture", "windows_grounded"
    )
    foreach ($requiredField in $requiredCandidateFields) {
        if (-not ($lockCandidateJson.PSObject.Properties.Name -contains $requiredField)) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json is missing required field '$requiredField' -- manifest is structurally invalid."
        }
    }

    if ($lockCandidateJson.artifact_status -ne $reviewedArtifactStatus) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: lock candidate artifact_status is '$($lockCandidateJson.artifact_status)', expected '$reviewedArtifactStatus'. Refusing to install from an unreviewed or promoted candidate."
    }
    if ($lockCandidateJson.source_git_sha -ne $reviewedSourceGitSha) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: lock candidate source_git_sha does not match the reviewed binding. Refusing to proceed."
    }
    if ($lockCandidateJson.canonical_environment_directory -ne $expectedCanonicalEnvironmentDirectory) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: lock candidate canonical_environment_directory does not match the reviewed canonical environment identity ($expectedCanonicalEnvironmentDirectory)."
    }
    if ($lockCandidateJson.canonical_interpreter -ne $expectedCanonicalInterpreter) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: lock candidate canonical_interpreter does not match the reviewed canonical interpreter identity ($expectedCanonicalInterpreter)."
    }
    if ($lockCandidateJson.real_execution_environment_frozen -ne $false) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: lock candidate real_execution_environment_frozen is not false -- refusing to proceed under an unexpected frozen claim."
    }
    if ($lockCandidateJson.resolved_lock.sha256 -ne $reviewedLockSha256) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: lock candidate resolved_lock.sha256 does not match the reviewed lock hash."
    }
    if ($lockCandidateJson.resolved_lock.package_count -ne $reviewedPackageCount) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: lock candidate resolved_lock.package_count is $($lockCandidateJson.resolved_lock.package_count), expected exactly $reviewedPackageCount."
    }
    if ($lockCandidateJson.requirements_real_execution.sha256 -ne $reviewedSourceRequirementsGitSha256) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: lock candidate requirements_real_execution.sha256 does not match the reviewed canonical Git-bytes source-requirements hash."
    }
    if ($lockCandidateJson.synthetic_xls_fixture.sha256 -ne $reviewedFixtureSha256) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: lock candidate synthetic_xls_fixture.sha256 does not match the reviewed fixture hash."
    }
    Write-Host "Lock candidate manifest structurally valid and matches the reviewed binding."

    # Independently recompute the on-disk lock file's SHA-256 -- never
    # trust the candidate JSON's self-reported hash alone.
    $lockFileHash = (Get-FileHash -LiteralPath $realExecutionLockPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($lockFileHash -ne $reviewedLockSha256) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: requirements-real-execution.lock.txt independently recomputed SHA-256 ($lockFileHash) does not match the reviewed lock hash ($reviewedLockSha256). Refusing to install from a mismatched lock file."
    }
    Write-Host "Lock file SHA-256 independently verified: $lockFileHash"

    # Independently recompute canonical Git object bytes for the source
    # requirements file at the reviewed source commit -- CRLF-independent,
    # never a checked-out working-tree copy (see Get-GitBlobSha256 above).
    $sourceRequirementsGitSha256 = Get-GitBlobSha256 -GitRef "$reviewedSourceGitSha`:requirements-real-execution.txt"
    if ($sourceRequirementsGitSha256 -ne $reviewedSourceRequirementsGitSha256) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: canonical Git-bytes SHA-256 for requirements-real-execution.txt at $reviewedSourceGitSha ($sourceRequirementsGitSha256) does not match the reviewed source-requirements hash ($reviewedSourceRequirementsGitSha256)."
    }
    Write-Host "Source requirements canonical Git-bytes provenance independently verified: $sourceRequirementsGitSha256"

    Write-Host "Reviewed-lock preflight PASSED."

    # ------------------------------------------------------------------
    # 6. Install the COMPLETE reviewed lock with exact versions, from
    #    requirements-real-execution.lock.txt ONLY (never the unpinned
    #    requirements-real-execution.txt), via the exact canonical
    #    .venv-real-execution interpreter (never bare pip/python/py, and
    #    never the general .venv's interpreter or its packages -- nothing
    #    is read from or copied out of ".venv"). --no-deps so pip cannot
    #    silently add any package outside the complete reviewed lock. Uses
    #    pip exactly as created by the Python 3.12.10 venv -- no separate
    #    "pip install --upgrade pip" step (the lock file itself pins the
    #    exact reviewed pip version).
    # ------------------------------------------------------------------
    Write-Host "Installing the complete reviewed lock via $canonicalInterpreterPath (--no-deps, requirements-real-execution.lock.txt only) ..."
    & $canonicalInterpreterPath -m pip install --no-deps -r $realExecutionLockPath
    if ($LASTEXITCODE -ne 0) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: '$canonicalInterpreterPath -m pip install --no-deps -r requirements-real-execution.lock.txt' failed."
    }
    Write-Host "Reviewed lock installation completed (.venv-real-execution only, from requirements-real-execution.lock.txt only, --no-deps)."

    # ------------------------------------------------------------------
    # 7. Run the readiness checker (no network, no private data, no gate
    #    consumption) via the exact canonical .venv-real-execution
    #    interpreter. It independently re-verifies everything checked
    #    above, plus live pip freeze --all package-set exactness.
    # ------------------------------------------------------------------
    if (-not (Test-Path -LiteralPath $readinessCheckerPath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: readiness checker not found at $readinessCheckerPath."
    }
    Write-Host "Running readiness checker ..."
    & $canonicalInterpreterPath $readinessCheckerPath
    $readinessExitCode = $LASTEXITCODE

    # ------------------------------------------------------------------
    # 8. Report only safe environment information. Never print private
    #    paths, ticker identities, or raw payloads (there are none here).
    # ------------------------------------------------------------------
    Write-Host "== Bootstrap complete =="
    Write-Host "CANONICAL_VENV_DIRECTORY=$canonicalVenvDirectory"
    Write-Host "CANONICAL_INTERPRETER=$canonicalInterpreterPath"
    Write-Host "GENERAL_VENV_TOUCHED=false"
    Write-Host "PROTECTED_INSTALL_AUTHORITY=requirements-real-execution.lock.txt"
    Write-Host "BASE_LAUNCHER_RESOLVED_VERSION=$resolvedBaseVersion"
    Write-Host "REVIEWED_LOCK_SHA256_VERIFIED=$lockFileHash"
    Write-Host "SOURCE_REQUIREMENTS_GIT_PROVENANCE_VERIFIED=$sourceRequirementsGitSha256"
    Write-Host "READINESS_CHECKER_EXIT_CODE=$readinessExitCode"
    Write-Host "REAL_NETWORK_REQUESTS_TO_PROTECTED_HOSTS=0"
    Write-Host "PRIVATE_READS=0"
    Write-Host "GATES_CONSUMED=0"

    if ($readinessExitCode -ne 0) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: readiness checker reported REAL_EXECUTION_ENVIRONMENT_READY=false (exit code $readinessExitCode). See its JSON output above. No protected boundary was crossed."
    }

    Write-Host "Environment bootstrap and readiness check both PASSED. This does NOT by itself authorize any gated real execution -- see AI_REAL_EXECUTION_RUNBOOK.md SS16 for the full required ordering. REAL_EXECUTION_ENVIRONMENT_FROZEN remains false until a separate, later, explicitly reviewed promotion task."
}
