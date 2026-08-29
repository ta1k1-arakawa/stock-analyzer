# V9_006 Stage-A schema-discovery Phase-1 -- single atomic real-execution
# entrypoint (AI_REAL_EXECUTION_RUNBOOK.md SS1: one atomic `& { ... }` scope,
# or one reviewed .ps1 file). Do not paste this file's body as independent
# line-by-line snippets: a `throw` on one line does not stop a separately
# pasted later line from still running, which is exactly why this whole
# preflight lives inside one scope.
#
# Scope: V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_REAL_EXECUTION_DESIGN.md,
# Phase 1 only (F1 TERMINAL, F2 BASE, F3 YEAR, F4 BASE, F7 -- exactly 341
# evidence objects plus 12 support locks / 353 raw-lock pairs). This file
# implements only the reviewed PowerShell entrypoint; it never bypasses,
# duplicates, or replaces the Python one-shot boundary's receipt authority
# (src/v9_006_stage_a_schema_discovery.py). This script itself performs zero
# JPX/Yahoo network requests -- the sole real acquisition happens inside the
# canonical .venv-real-execution Python process invoked at the very end,
# after every applicable pre-gate check below has passed and fresh
# point-of-use human authorization has been obtained.
#
# This script does NOT bake any human authorization into code. The
# confirmation token is read interactively at point of use (Read-Host,
# below) and is compared only in-memory against the fixed contract string
# already defined in src/v9_006_stage_a_schema_discovery.py
# (SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_CONFIRMATION); it is never written to
# disk, never logged, is never a script parameter/argv value, and is cleared
# from the process environment (and from this script's own variable) in a
# `finally` block even on failure (AI_REAL_EXECUTION_RUNBOOK.md SS9).
#
# Before requesting that confirmation, this script mechanically verifies, in
# order (AI_REAL_EXECUTION_RUNBOOK.md SS2/SS13/SS16, REAL_EXECUTION_PYTHON_
# ENVIRONMENT.md, and V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_REAL_EXECUTION_
# DESIGN.md): the repository root (derived mechanically from this script's
# own location, never accepted as a parameter); every required reviewed
# file exists; the authoritative branch; that -ExpectedHead is an exact
# 40-lowercase-hex commit SHA; a clean working tree; that local HEAD and the
# fetched authoritative remote HEAD both equal -ExpectedHead exactly; that
# -OutputRoot is a fresh, not-yet-existing path; the canonical protected
# interpreter (.venv-real-execution\Scripts\python.exe, never the general
# .venv); and complete environment/dependency/lock/synthetic-parser readiness
# via the existing reviewed no-network checker
# (scripts\check_real_execution_env.py) -- proving, not merely trusting its
# process exit code, that REAL_EXECUTION_ENVIRONMENT_READY,
# REAL_EXECUTION_ENVIRONMENT_FROZEN, ENVIRONMENT_FREEZE_CHECK, and
# ENVIRONMENT_LOCK_FINGERPRINT_STATUS are all exactly their required PASS
# values from its parsed JSON output, so a ready-but-not-yet-frozen
# environment can never reach human authorization; and that the canonical
# task-global Phase-1 gate receipt is mechanically proven ABSENT via the
# exact reviewed Python reader (read_phase1_schema_discovery_gate_consumed_
# state) -- never a PowerShell reimplementation of that schema/parsing
# logic. Only when every one of these PASSES does this script request
# confirmation; after confirmation it reruns every applicable non-destructive
# binding once more -- including required-reviewed-file presence, the
# authoritative branch, and the same environment-freeze proof -- and only
# then briefly sets the confirmation environment variable
# immediately before invoking the single reviewed Python one-shot boundary
# (scripts\run_v9_006_stage_a_schema_discovery.py), which alone is
# responsible for OutputRoot creation and canonical receipt publication.
#
# No path in this script continues past a failed check: every failure is a
# `throw`, terminating the entire atomic scope. There is no retry path; a
# POST_GATE failure is never retried, and the receipt is never reset,
# deleted, or overwritten to obtain another attempt.
#
# Usage (run this whole file directly; do not copy fragments):
#   pwsh -File scripts\run_v9_006_stage_a_schema_discovery_phase1_real_execution.ps1 `
#       -ExpectedHead "<exact 40-hex reviewed execution SHA>" `
#       -OutputRoot "C:\path\to\a-fresh-not-yet-existing-directory"

param(
    [Parameter(Mandatory = $true)][string]$ExpectedHead,
    [Parameter(Mandatory = $true)][string]$OutputRoot
)

& {
    $ErrorActionPreference = "Stop"

    $authoritativeBranch = "v9-cross-sectional-close-auction-design"
    $confirmationContract = "V9_006_STAGE_A_SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_ONE_SHOT"
    $confirmationEnvironmentVariableName = "V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_CONFIRMATION"

    $requiredReviewedFiles = @(
        "scripts\run_v9_006_stage_a_schema_discovery.py",
        "scripts\run_v9_006_stage_a_schema_discovery_phase1_real_execution.ps1",
        "scripts\check_real_execution_env.py",
        "src\v9_006_stage_a_schema_discovery.py",
        "AI_REAL_EXECUTION_RUNBOOK.md",
        "AI_RESEARCH_EXECUTION_RULES.md",
        "REAL_EXECUTION_PYTHON_ENVIRONMENT.md",
        "V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_REAL_EXECUTION_DESIGN.md",
        "V9_006_STAGE_A_SCHEMA_DISCOVERY_IMPLEMENTATION.md",
        "PROJECT_STATE.md"
    )

    Write-Host "== V9_006 Stage-A schema-discovery Phase-1 real execution (atomic preflight + gated execution) =="

    # ------------------------------------------------------------------
    # Helper: canonical task-global Phase-1 receipt state, read only
    # through the exact reviewed Python function -- never reimplemented
    # here. Returns nothing; throws on anything but a mechanically proven
    # "False" (absent). Never prints the resolved receipt path.
    # ------------------------------------------------------------------
    function Assert-Phase1GlobalReceiptAbsent {
        param(
            [Parameter(Mandatory = $true)][string]$PythonInterpreter,
            [Parameter(Mandatory = $true)][string]$RepositoryRootPath
        )
        $receiptStateCheckCode = @'
import sys
sys.path.insert(0, sys.argv[1])
from src.v9_006_stage_a_schema_discovery import read_phase1_schema_discovery_gate_consumed_state
print(repr(read_phase1_schema_discovery_gate_consumed_state()))
'@
        $receiptStateOutputLines = $receiptStateCheckCode | & $PythonInterpreter - $RepositoryRootPath
        if ($LASTEXITCODE -ne 0) {
            throw "PRE_GATE_BLOCK: canonical Phase-1 global receipt-state reader failed to run (exit code $LASTEXITCODE)."
        }
        $receiptState = ($receiptStateOutputLines | Select-Object -Last 1).ToString().Trim()
        if ($receiptState -eq "True") {
            throw "POST_GATE_ALREADY_CONSUMED: the canonical Phase-1 global gate receipt already exists and is a valid consumed receipt. This one-shot gate has already been used for this study/task identity; it is never reset, deleted, or reused. Return to the GPT methodology authority and human authority for the next decision."
        }
        if ($receiptState -ne "False") {
            throw "PRE_GATE_BLOCK: canonical Phase-1 global receipt-state reader returned an ambiguous/uncertain state. Failing closed; do not proceed."
        }
    }

    # ------------------------------------------------------------------
    # Helper: proves the existing reviewed scripts\check_real_execution_env.py
    # checker reports a fully ready AND frozen canonical real-execution
    # environment. Never trusts the checker's process exit code alone --
    # REAL_EXECUTION_ENVIRONMENT_READY and REAL_EXECUTION_ENVIRONMENT_FROZEN
    # are computed and reported separately by that checker (exit code
    # depends only on READY), so a READY=true/FROZEN=false state must still
    # fail closed here before it can ever reach human authorization. Reused
    # by both the pre-authorization and post-confirmation checks so the
    # predicates cannot drift between them. Captures the checker's JSON
    # output only into a local variable (never Write-Host'd in full, to
    # avoid exposing local paths in its nested detail fields); on success it
    # prints only the four safe top-level status fields.
    # ------------------------------------------------------------------
    function Assert-RealExecutionEnvironmentFrozen {
        param(
            [Parameter(Mandatory = $true)][string]$PythonInterpreter,
            [Parameter(Mandatory = $true)][string]$CheckerPath
        )
        $checkerRawOutput = & $PythonInterpreter $CheckerPath | Out-String
        $checkerExitCode = $LASTEXITCODE
        if ($checkerExitCode -ne 0) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: scripts\check_real_execution_env.py exited non-zero ($checkerExitCode); REAL_EXECUTION_ENVIRONMENT_READY is not proven true. Failing closed."
        }
        try {
            $checkerResult = $checkerRawOutput | ConvertFrom-Json -ErrorAction Stop
        }
        catch {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: scripts\check_real_execution_env.py output could not be parsed as JSON. Failing closed."
        }
        if ($null -eq $checkerResult) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: scripts\check_real_execution_env.py produced no parsable JSON result. Failing closed."
        }
        $environmentReady = $checkerResult.REAL_EXECUTION_ENVIRONMENT_READY
        $environmentFrozen = $checkerResult.REAL_EXECUTION_ENVIRONMENT_FROZEN
        $freezeCheckStatus = $checkerResult.ENVIRONMENT_FREEZE_CHECK
        $lockFingerprintStatus = $checkerResult.ENVIRONMENT_LOCK_FINGERPRINT_STATUS
        if ($environmentReady -isnot [bool] -or $environmentReady -ne $true) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: REAL_EXECUTION_ENVIRONMENT_READY is not exactly boolean true (or missing/ambiguous). Failing closed."
        }
        if ($environmentFrozen -isnot [bool] -or $environmentFrozen -ne $true) {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: REAL_EXECUTION_ENVIRONMENT_FROZEN is not exactly boolean true (or missing/ambiguous). A ready-but-not-frozen environment must never reach human authorization. Failing closed."
        }
        if ($freezeCheckStatus -isnot [string] -or $freezeCheckStatus -cne "PASS") {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: ENVIRONMENT_FREEZE_CHECK is not exactly 'PASS' (or missing/ambiguous). Failing closed."
        }
        if ($lockFingerprintStatus -isnot [string] -or $lockFingerprintStatus -cne "FROZEN") {
            throw "PRE_GATE_ENVIRONMENT_BLOCK: ENVIRONMENT_LOCK_FINGERPRINT_STATUS is not exactly 'FROZEN' (or missing/ambiguous). Failing closed."
        }
        Write-Host "REAL_EXECUTION_ENVIRONMENT_READY=true"
        Write-Host "REAL_EXECUTION_ENVIRONMENT_FROZEN=true"
        Write-Host "ENVIRONMENT_FREEZE_CHECK=PASS"
        Write-Host "ENVIRONMENT_LOCK_FINGERPRINT_STATUS=FROZEN"
    }

    # ------------------------------------------------------------------
    # 1. Repository root, derived mechanically from this script's own
    #    location (never accepted as a parameter), and required reviewed
    #    files.
    # ------------------------------------------------------------------
    $repositoryRoot = Split-Path -Parent $PSScriptRoot
    if (-not (Test-Path -LiteralPath $repositoryRoot -PathType Container)) {
        throw "PRE_GATE_BLOCK: repository root derived from script location does not exist or is not a directory: $repositoryRoot"
    }
    Set-Location -LiteralPath $repositoryRoot
    foreach ($reviewedFile in $requiredReviewedFiles) {
        $reviewedFilePath = Join-Path (Get-Location) $reviewedFile
        if (-not (Test-Path -LiteralPath $reviewedFilePath -PathType Leaf)) {
            throw "PRE_GATE_BLOCK: required reviewed file missing ($reviewedFile). This does not look like a complete, reviewed stock-analyzer checkout."
        }
    }
    Write-Host "Repository root verified (mechanically derived from script location): $repositoryRoot"
    Write-Host "All required reviewed files present."

    # ------------------------------------------------------------------
    # 2. Authoritative branch.
    # ------------------------------------------------------------------
    $currentBranch = (git branch --show-current).Trim()
    if ($currentBranch -ne $authoritativeBranch) {
        throw "PRE_GATE_BLOCK: current branch '$currentBranch' is not the authoritative branch '$authoritativeBranch'."
    }
    Write-Host "Authoritative branch verified: $authoritativeBranch"

    # ------------------------------------------------------------------
    # 3. -ExpectedHead must be an exact 40-lowercase-hex commit SHA.
    # ------------------------------------------------------------------
    if (-not [regex]::IsMatch($ExpectedHead, "^[0-9a-f]{40}$")) {
        throw "PRE_GATE_BLOCK: -ExpectedHead is not an exact 40-lowercase-hex commit SHA."
    }
    Write-Host "Reviewed execution SHA format verified (40 lowercase hex)."

    # ------------------------------------------------------------------
    # 4. Clean working tree, exact local HEAD.
    # ------------------------------------------------------------------
    $workingTreeStatus = (git status --porcelain)
    if (-not [string]::IsNullOrEmpty($workingTreeStatus)) {
        throw "PRE_GATE_BLOCK: working tree is not clean. Commit, stash, or discard changes before running this real-execution entrypoint."
    }
    $localHead = (git rev-parse HEAD).Trim()
    if ($localHead -ne $ExpectedHead) {
        throw "EXPECTED_HEAD_MISMATCH: local HEAD ($localHead) does not equal -ExpectedHead ($ExpectedHead)."
    }
    Write-Host "Working tree clean; local HEAD verified: $localHead"

    # ------------------------------------------------------------------
    # 5. Exact authoritative remote HEAD.
    # ------------------------------------------------------------------
    git fetch origin $authoritativeBranch | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "PRE_GATE_BLOCK: 'git fetch origin $authoritativeBranch' failed."
    }
    $remoteHead = (git rev-parse "origin/$authoritativeBranch").Trim()
    if ($remoteHead -ne $ExpectedHead) {
        throw "EXPECTED_HEAD_MISMATCH: remote origin/$authoritativeBranch HEAD ($remoteHead) does not equal -ExpectedHead ($ExpectedHead)."
    }
    Write-Host "Remote authoritative HEAD verified: $remoteHead"

    # ------------------------------------------------------------------
    # 6. -OutputRoot must be fresh / not yet existing.
    # ------------------------------------------------------------------
    if (Test-Path -LiteralPath $OutputRoot) {
        throw "PRE_GATE_BLOCK: -OutputRoot already exists ($OutputRoot). Phase-1 output must be a fresh, not-yet-existing, exclusive directory -- this may indicate a prior attempt. Choose a new path; never overwrite or delete existing state automatically."
    }
    Write-Host "OutputRoot freshness verified (does not yet exist): $OutputRoot"

    # ------------------------------------------------------------------
    # 7. Canonical protected interpreter (never the general .venv, per
    #    AI_REAL_EXECUTION_RUNBOOK.md SS15 / REAL_EXECUTION_PYTHON_
    #    ENVIRONMENT.md).
    # ------------------------------------------------------------------
    $canonicalInterpreterPath = Join-Path (Get-Location) ".venv-real-execution\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $canonicalInterpreterPath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: canonical interpreter not found at $canonicalInterpreterPath. Run scripts\bootstrap_real_execution_env.ps1 first; this script will not fall back to the general .venv or a bare 'python'."
    }
    Write-Host "Canonical interpreter verified: $canonicalInterpreterPath"

    # ------------------------------------------------------------------
    # 8. Complete protected interpreter/dependency/environment-lock/
    #    synthetic-parser readiness AND the exact reviewed environment
    #    freeze, via the existing reviewed no-network checker. The checker's
    #    process exit code depends only on REAL_EXECUTION_ENVIRONMENT_READY,
    #    so this proves READY, FROZEN, ENVIRONMENT_FREEZE_CHECK, and
    #    ENVIRONMENT_LOCK_FINGERPRINT_STATUS independently from its parsed
    #    JSON output -- a ready-but-not-frozen environment must never reach
    #    human authorization. Zero real network occurs in this step.
    # ------------------------------------------------------------------
    $readinessCheckerPath = Join-Path (Get-Location) "scripts\check_real_execution_env.py"
    if (-not (Test-Path -LiteralPath $readinessCheckerPath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: readiness checker not found at $readinessCheckerPath."
    }
    Write-Host "Running canonical readiness checker (no network, no private data, no gate consumption) ..."
    Assert-RealExecutionEnvironmentFrozen -PythonInterpreter $canonicalInterpreterPath -CheckerPath $readinessCheckerPath
    Write-Host "Readiness checker PASSED: environment is ready AND frozen (see safe status fields above)."
    Write-Host "CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE=YES"

    # ------------------------------------------------------------------
    # 9. Canonical task-global Phase-1 gate receipt mechanically proven
    #    absent, via the exact reviewed Python reader.
    # ------------------------------------------------------------------
    Assert-Phase1GlobalReceiptAbsent -PythonInterpreter $canonicalInterpreterPath -RepositoryRootPath $repositoryRoot
    Write-Host "Canonical Phase-1 global gate receipt mechanically proven absent."

    Write-Host "== All pre-authorization checks PASSED =="

    # ------------------------------------------------------------------
    # 10. Fresh point-of-use human confirmation. Never a script parameter,
    #     never baked into this file or chat history.
    # ------------------------------------------------------------------
    $typedConfirmationToken = Read-Host -Prompt "Type the exact Phase-1 confirmation token to authorize real public JPX network requests (jpx.co.jp only)"
    if ($typedConfirmationToken -ne $confirmationContract) {
        $typedConfirmationToken = $null
        throw "PRE_GATE_BLOCK: confirmation token did not match. No network request was made."
    }
    Write-Host "Phase-1 confirmation token verified."

    # ------------------------------------------------------------------
    # 11. Post-confirmation, pre-consumption: rerun every applicable
    #     non-destructive provenance/readiness/durable-state binding --
    #     required reviewed files, the authoritative branch, working-tree
    #     cleanliness, local/remote HEAD, OutputRoot freshness, the
    #     canonical interpreter, environment readiness, and the receipt
    #     absence check -- before any gate consumption. Any change or
    #     uncertainty stops without consuming the gate, without setting the
    #     confirmation environment variable, and without invoking Python
    #     acquisition.
    # ------------------------------------------------------------------
    Write-Host "Re-running non-destructive bindings after confirmation, before gate consumption ..."

    foreach ($reviewedFile in $requiredReviewedFiles) {
        $reviewedFilePath = Join-Path (Get-Location) $reviewedFile
        if (-not (Test-Path -LiteralPath $reviewedFilePath -PathType Leaf)) {
            $typedConfirmationToken = $null
            throw "PRE_GATE_BLOCK: required reviewed file went missing between confirmation and consumption ($reviewedFile). Aborting without gate consumption."
        }
    }
    $postConfirmationBranch = (git branch --show-current).Trim()
    if ($postConfirmationBranch -ne $authoritativeBranch) {
        $typedConfirmationToken = $null
        throw "PRE_GATE_BLOCK: current branch changed between confirmation and consumption ('$postConfirmationBranch' != '$authoritativeBranch'). Aborting without gate consumption, without setting the confirmation environment variable, and without invoking Python acquisition."
    }

    $postConfirmationWorkingTreeStatus = (git status --porcelain)
    if (-not [string]::IsNullOrEmpty($postConfirmationWorkingTreeStatus)) {
        $typedConfirmationToken = $null
        throw "PRE_GATE_BLOCK: working tree became dirty between confirmation and consumption. Aborting without gate consumption."
    }
    $postConfirmationLocalHead = (git rev-parse HEAD).Trim()
    if ($postConfirmationLocalHead -ne $ExpectedHead) {
        $typedConfirmationToken = $null
        throw "EXPECTED_HEAD_MISMATCH: local HEAD changed between confirmation and consumption ($postConfirmationLocalHead != $ExpectedHead). Aborting without gate consumption."
    }
    git fetch origin $authoritativeBranch | Out-Null
    if ($LASTEXITCODE -ne 0) {
        $typedConfirmationToken = $null
        throw "PRE_GATE_BLOCK: post-confirmation 'git fetch origin $authoritativeBranch' failed. Aborting without gate consumption."
    }
    $postConfirmationRemoteHead = (git rev-parse "origin/$authoritativeBranch").Trim()
    if ($postConfirmationRemoteHead -ne $ExpectedHead) {
        $typedConfirmationToken = $null
        throw "EXPECTED_HEAD_MISMATCH: remote origin/$authoritativeBranch HEAD changed between confirmation and consumption ($postConfirmationRemoteHead != $ExpectedHead). Aborting without gate consumption."
    }
    if (Test-Path -LiteralPath $OutputRoot) {
        $typedConfirmationToken = $null
        throw "PRE_GATE_BLOCK: -OutputRoot came into existence between confirmation and consumption ($OutputRoot). Aborting without gate consumption."
    }
    if (-not (Test-Path -LiteralPath $canonicalInterpreterPath -PathType Leaf)) {
        $typedConfirmationToken = $null
        throw "PRE_GATE_ENVIRONMENT_BLOCK: canonical interpreter no longer present at $canonicalInterpreterPath. Aborting without gate consumption."
    }
    try {
        Assert-RealExecutionEnvironmentFrozen -PythonInterpreter $canonicalInterpreterPath -CheckerPath $readinessCheckerPath
    }
    catch {
        $typedConfirmationToken = $null
        throw
    }
    Assert-Phase1GlobalReceiptAbsent -PythonInterpreter $canonicalInterpreterPath -RepositoryRootPath $repositoryRoot

    Write-Host "Post-confirmation re-verification PASSED. Proceeding to the single reviewed Python one-shot boundary."

    # ------------------------------------------------------------------
    # 12. Consume the gate: invoke the single reviewed Python one-shot
    #     boundary. The confirmation environment variable is set only
    #     immediately before this call and is always cleared in `finally`,
    #     even on failure. Confirmation is never passed on argv. This
    #     PowerShell script never creates OutputRoot, never publishes the
    #     receipt itself, and never retries this call.
    # ------------------------------------------------------------------
    $cliScriptPath = Join-Path (Get-Location) "scripts\run_v9_006_stage_a_schema_discovery.py"
    try {
        [System.Environment]::SetEnvironmentVariable($confirmationEnvironmentVariableName, $typedConfirmationToken, "Process")
        Write-Host "Invoking the reviewed Python one-shot boundary (real jpx.co.jp-only network requests may now occur) ..."
        & $canonicalInterpreterPath $cliScriptPath --output-root $OutputRoot --execution-sha $ExpectedHead
        $cliExitCode = $LASTEXITCODE
    }
    finally {
        [System.Environment]::SetEnvironmentVariable($confirmationEnvironmentVariableName, $null, "Process")
        $typedConfirmationToken = $null
    }

    Write-Host "== Phase-1 real-execution run complete =="
    Write-Host "CLI_EXIT_CODE=$cliExitCode"
    Write-Host "See the safe JSON summary printed above for execution_result, gate_consumed, and safe counts/hashes."
    Write-Host "This run does NOT by itself authorize Phase 2, Stage B, T1, model fitting, backtesting, or V9 design freeze."

    if ($cliExitCode -ne 0) {
        throw "PHASE1_REAL_EXECUTION_DID_NOT_PASS: see the safe JSON summary above for failure_class and gate_consumed. This is a one-shot boundary; do not retry, reset, or delete the receipt. cliExitCode=$cliExitCode"
    }
}
