# V9_005 Stage-A free official JPX metadata/calendar probe -- single atomic
# real-execution entrypoint (AI_REAL_EXECUTION_RUNBOOK.md SS1: one atomic
# `& { ... }` scope, or one reviewed .ps1 file). Do not paste this file's
# body as independent line-by-line snippets: a `throw` on one line does not
# stop a separately pasted later line from still running, which is exactly
# why this whole preflight lives inside one scope.
#
# This script does NOT bake any human authorization into code. The Stage-A
# confirmation token is read interactively at point of use (Read-Host,
# below) and is compared only in-memory against the fixed contract string
# already defined in src/v9_005_stage_a_jpx_probe.py; it is never written to
# disk, never logged, and is cleared from the process environment in a
# `finally` block even on failure (AI_REAL_EXECUTION_RUNBOOK.md SS9).
#
# Before any network request, this script verifies (AI_REAL_EXECUTION_
# RUNBOOK.md SS2, SS13): correct repository; authoritative branch; exact
# expected local HEAD (supplied by -ExpectedHead, never hardcoded here);
# exact authoritative remote HEAD; clean working tree; the exact reviewed
# V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md blob binding
# (V9_005_HIGH_2B signal-grid contract); no existing Stage-A output/durable
# execution collision at -OutputRoot; and the explicit Stage-A confirmation
# token. No network occurs unless every one of these checks passes.
#
# Usage (run this whole file directly; do not copy fragments):
#   pwsh -File scripts\run_v9_005_stage_a_jpx_probe.ps1 `
#       -RepoRoot "C:\path\to\stock-analyzer" `
#       -ExpectedHead "<exact 40-hex commit sha the operator expects>" `
#       -OutputRoot "C:\path\to\a-fresh-not-yet-existing-directory"

param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$ExpectedHead,
    [Parameter(Mandatory = $true)][string]$OutputRoot
)

& {
    $ErrorActionPreference = "Stop"

    $authoritativeBranch = "v9-cross-sectional-close-auction-design"
    $boundSignalGridPath = "V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md"
    $boundSignalGridBlobSha = "9135183b7fc5097602fa40fcda8f1b0448220244"
    $stageAConfirmationToken = "V9_005_STAGE_A_HUMAN_AUTHORIZE_JPX_METADATA_PROBE"
    $confirmationEnvironmentVariableName = "V9_005_STAGE_A_CONFIRMATION"

    $expectedRepositoryMarkerFiles = @(
        "src\v9_005_stage_a_jpx_probe.py",
        "scripts\run_v9_005_stage_a_jpx_probe.py",
        "AI_REAL_EXECUTION_RUNBOOK.md",
        "V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md"
    )

    Write-Host "== V9_005 Stage-A free JPX metadata probe (atomic preflight + gated execution) =="

    # ------------------------------------------------------------------
    # 1. Repository root and marker files.
    # ------------------------------------------------------------------
    if (-not (Test-Path -LiteralPath $RepoRoot -PathType Container)) {
        throw "PRE_GATE_BLOCK: -RepoRoot does not exist or is not a directory: $RepoRoot"
    }
    Set-Location -LiteralPath $RepoRoot
    foreach ($markerFile in $expectedRepositoryMarkerFiles) {
        $markerPath = Join-Path (Get-Location) $markerFile
        if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
            throw "PRE_GATE_BLOCK: repository marker file missing ($markerFile). -RepoRoot does not look like the stock-analyzer repository."
        }
    }
    Write-Host "Repository root verified: $RepoRoot"

    # ------------------------------------------------------------------
    # 2. Authoritative branch, clean tree, exact expected local HEAD.
    # ------------------------------------------------------------------
    $currentBranch = (git branch --show-current).Trim()
    if ($currentBranch -ne $authoritativeBranch) {
        throw "PRE_GATE_BLOCK: current branch '$currentBranch' is not the authoritative branch '$authoritativeBranch'."
    }
    $workingTreeStatus = (git status --porcelain)
    if (-not [string]::IsNullOrEmpty($workingTreeStatus)) {
        throw "PRE_GATE_BLOCK: working tree is not clean. Commit, stash, or discard changes before running this probe."
    }
    $localHead = (git rev-parse HEAD).Trim()
    if ($localHead -ne $ExpectedHead) {
        throw "EXPECTED_HEAD_MISMATCH: local HEAD ($localHead) does not equal -ExpectedHead ($ExpectedHead)."
    }
    Write-Host "Branch/tree/local-HEAD verified: $authoritativeBranch @ $localHead (clean)"

    # ------------------------------------------------------------------
    # 3. Exact authoritative remote HEAD.
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
    # 4. V9_005_HIGH_2B signal-grid contract blob binding.
    # ------------------------------------------------------------------
    $actualSignalGridBlobSha = (git rev-parse "HEAD:$boundSignalGridPath").Trim()
    if ($actualSignalGridBlobSha -ne $boundSignalGridBlobSha) {
        throw "PROBE_SIGNAL_GRID_CONTRACT_MISMATCH: $boundSignalGridPath blob SHA ($actualSignalGridBlobSha) does not equal the bound reviewed SHA ($boundSignalGridBlobSha). Do not proceed; obtain a fresh GPT methodology review/rebinding before any further execution."
    }
    Write-Host "Signal-grid contract blob binding verified: $boundSignalGridPath @ $actualSignalGridBlobSha"

    # ------------------------------------------------------------------
    # 5. No existing Stage-A output/durable execution collision.
    # ------------------------------------------------------------------
    if (Test-Path -LiteralPath $OutputRoot) {
        throw "PRE_GATE_BLOCK: -OutputRoot already exists ($OutputRoot). Stage-A output must be a fresh, not-yet-existing, exclusive directory -- this may indicate a prior attempt. Choose a new path; never overwrite or delete existing state automatically."
    }
    Write-Host "Output-root collision check passed (fresh path): $OutputRoot"

    # ------------------------------------------------------------------
    # 6. Explicit Stage-A confirmation token, supplied at point of use.
    #    Never baked into this file or into chat history.
    # ------------------------------------------------------------------
    $typedConfirmationToken = Read-Host -Prompt "Type the exact Stage-A confirmation token to authorize real JPX network requests (jpx.co.jp only)"
    if ($typedConfirmationToken -ne $stageAConfirmationToken) {
        throw "PRE_GATE_BLOCK: confirmation token did not match. No network request was made."
    }
    Write-Host "Stage-A confirmation token verified."

    # ------------------------------------------------------------------
    # 7. Canonical protected-execution Python interpreter (never the
    #    general .venv, per AI_REAL_EXECUTION_RUNBOOK.md SS15).
    # ------------------------------------------------------------------
    $canonicalInterpreterPath = Join-Path (Get-Location) ".venv-real-execution\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $canonicalInterpreterPath -PathType Leaf)) {
        throw "PRE_GATE_ENVIRONMENT_BLOCK: canonical interpreter not found at $canonicalInterpreterPath. Run scripts\bootstrap_real_execution_env.ps1 first; this script will not fall back to the general .venv or a bare 'python'."
    }
    Write-Host "Canonical interpreter verified: $canonicalInterpreterPath"

    # ------------------------------------------------------------------
    # 8. Run the probe with the confirmation token passed only via a
    #    transient environment variable, cleared in `finally` even on
    #    failure. No other network operation occurs in this script.
    # ------------------------------------------------------------------
    $probeScriptPath = Join-Path (Get-Location) "scripts\run_v9_005_stage_a_jpx_probe.py"
    try {
        [System.Environment]::SetEnvironmentVariable($confirmationEnvironmentVariableName, $typedConfirmationToken, "Process")
        Write-Host "Running Stage-A probe (real jpx.co.jp-only network requests may now occur) ..."
        & $canonicalInterpreterPath $probeScriptPath --output-root $OutputRoot --repo-root (Get-Location)
        $probeExitCode = $LASTEXITCODE
    }
    finally {
        [System.Environment]::SetEnvironmentVariable($confirmationEnvironmentVariableName, $null, "Process")
        $typedConfirmationToken = $null
    }

    Write-Host "== Stage-A probe run complete =="
    Write-Host "PROBE_EXIT_CODE=$probeExitCode"
    Write-Host "See the safe JSON summary printed above, and OutputRoot\result.json / receipt.json for the durable safe evidence."
    Write-Host "This run does NOT by itself authorize Stage B, T1, model fitting, backtesting, or V9 design freeze."

    if ($probeExitCode -ne 0) {
        throw "STAGE_A_PROBE_DID_NOT_PASS: see the safe JSON summary above for failure_class and evidence booleans. probeExitCode=$probeExitCode"
    }
}
