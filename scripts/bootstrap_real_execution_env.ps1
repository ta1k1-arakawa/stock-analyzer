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
        "AI_REAL_EXECUTION_RUNBOOK.md"
    )
    $canonicalPythonMajorMinor = "3.12"
    $canonicalVenvDirectory = Join-Path (Get-Location) ".venv-real-execution"
    $canonicalInterpreterPath = Join-Path $canonicalVenvDirectory "Scripts\python.exe"
    $realExecutionRequirementsPath = Join-Path (Get-Location) "requirements-real-execution.txt"
    $readinessCheckerPath = Join-Path (Get-Location) "scripts\check_real_execution_env.py"

    Write-Host "== Real-execution environment bootstrap (environment setup stage only) =="
    Write-Host "Target: CANONICAL_PROTECTED_REAL_EXECUTION_ENVIRONMENT = .venv-real-execution"
    Write-Host "This script never touches the separate general '.venv' project environment."

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
    # 2. Verify required base Python launcher/version (used only to CREATE
    #    .venv-real-execution -- not itself the protected execution
    #    interpreter).
    # ------------------------------------------------------------------
    $baseLauncherCommand = Get-Command "py" -ErrorAction SilentlyContinue
    if ($null -eq $baseLauncherCommand) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: the Python launcher (py) was not found. Install the official Python launcher for Windows before bootstrapping."
    }
    $baseLauncherVersionOutput = & py "-$canonicalPythonMajorMinor" -c "import sys; print('.'.join(str(part) for part in sys.version_info[:3]))"
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($baseLauncherVersionOutput)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: py -$canonicalPythonMajorMinor did not resolve to a working interpreter. Install Python $canonicalPythonMajorMinor.x (matching .github/workflows/daily_ai_trade.yml) before bootstrapping."
    }
    $resolvedBaseVersion = $baseLauncherVersionOutput.Trim()
    Write-Host "Base launcher resolved: py -$canonicalPythonMajorMinor -> Python $resolvedBaseVersion"

    # ------------------------------------------------------------------
    # 3/4. Create .venv-real-execution only if absent; if present, verify
    #      it belongs to this repository and the expected Python version
    #      rather than silently recreating it. Mismatch => STOP, no
    #      auto-delete. The separate general ".venv" is never inspected,
    #      read from, or written to by this step or any other step below.
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
        $existingVenvVersionOutput = & $canonicalInterpreterPath -c "import sys; print('.'.join(str(part) for part in sys.version_info[:2]))"
        if ($LASTEXITCODE -ne 0) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: existing .venv-real-execution interpreter at $canonicalInterpreterPath failed to run. Remediate manually; this script will not delete or recreate it automatically."
        }
        $existingVenvMajorMinor = $existingVenvVersionOutput.Trim()
        if ($existingVenvMajorMinor -ne $canonicalPythonMajorMinor) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: existing .venv-real-execution is Python $existingVenvMajorMinor, expected $canonicalPythonMajorMinor. This script will not delete or recreate a mismatched .venv-real-execution automatically -- remediate manually (e.g. remove .venv-real-execution yourself after confirming nothing depends on it, then re-run)."
        }
        Write-Host "Existing .venv-real-execution verified: Python $existingVenvMajorMinor at $canonicalInterpreterPath"
    }

    if (-not (Test-Path -LiteralPath $canonicalInterpreterPath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: canonical interpreter not found at $canonicalInterpreterPath after venv creation/verification."
    }

    # ------------------------------------------------------------------
    # 5. Install/upgrade dependencies ONLY from the repository-controlled
    #    real-execution requirements specification, via the exact
    #    canonical .venv-real-execution interpreter (never bare
    #    pip/python/py, and never the general .venv's interpreter or its
    #    packages -- nothing is read from or copied out of ".venv").
    # ------------------------------------------------------------------
    if (-not (Test-Path -LiteralPath $realExecutionRequirementsPath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: requirements-real-execution.txt not found at $realExecutionRequirementsPath."
    }
    Write-Host "Installing/upgrading real-execution dependencies via $canonicalInterpreterPath ..."
    & $canonicalInterpreterPath -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: '$canonicalInterpreterPath -m pip install --upgrade pip' failed."
    }
    & $canonicalInterpreterPath -m pip install --upgrade -r $realExecutionRequirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: '$canonicalInterpreterPath -m pip install --upgrade -r requirements-real-execution.txt' failed."
    }
    Write-Host "Dependency installation completed (.venv-real-execution only)."

    # ------------------------------------------------------------------
    # 6. Run the readiness checker (no network, no private data, no gate
    #    consumption) via the exact canonical .venv-real-execution
    #    interpreter.
    # ------------------------------------------------------------------
    if (-not (Test-Path -LiteralPath $readinessCheckerPath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: readiness checker not found at $readinessCheckerPath."
    }
    Write-Host "Running readiness checker ..."
    & $canonicalInterpreterPath $readinessCheckerPath
    $readinessExitCode = $LASTEXITCODE

    # ------------------------------------------------------------------
    # 7. Report only safe environment information. Never print private
    #    paths, ticker identities, or raw payloads (there are none here).
    # ------------------------------------------------------------------
    Write-Host "== Bootstrap complete =="
    Write-Host "CANONICAL_VENV_DIRECTORY=$canonicalVenvDirectory"
    Write-Host "CANONICAL_INTERPRETER=$canonicalInterpreterPath"
    Write-Host "GENERAL_VENV_TOUCHED=false"
    Write-Host "BASE_LAUNCHER_RESOLVED_VERSION=$resolvedBaseVersion"
    Write-Host "READINESS_CHECKER_EXIT_CODE=$readinessExitCode"
    Write-Host "REAL_NETWORK_REQUESTS_TO_PROTECTED_HOSTS=0"
    Write-Host "PRIVATE_READS=0"
    Write-Host "GATES_CONSUMED=0"

    if ($readinessExitCode -ne 0) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: readiness checker reported REAL_EXECUTION_ENVIRONMENT_READY=false (exit code $readinessExitCode). See its JSON output above. No protected boundary was crossed."
    }

    Write-Host "Environment bootstrap and readiness check both PASSED. This does NOT by itself authorize any gated real execution -- see AI_REAL_EXECUTION_RUNBOOK.md SS16 for the full required ordering, including the still-separate Windows-grounded environment lock/fingerprint review (REAL_EXECUTION_ENVIRONMENT_FROZEN)."
}
