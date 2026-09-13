[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CandidateRoot,
    [Parameter(Mandatory = $true)]
    [string]$ImplementationSha,
    [Parameter(Mandatory = $true)]
    [string]$AuthorizationMarker,
    [Parameter(Mandatory = $true)]
    [string]$ReceiptPath,
    [Parameter(Mandatory = $true)]
    [string]$CaptureRoot
)

$ErrorActionPreference = "Stop"

function Convert-ToFullPath([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or -not [IO.Path]::IsPathRooted($Value)) {
        throw "PATH_INVALID"
    }
    return [IO.Path]::GetFullPath($Value)
}

function Assert-NoReparseAncestor([string]$PathValue) {
    $current = [IO.Path]::GetFullPath($PathValue)
    while ($true) {
        try {
            $item = Get-Item -LiteralPath $current -Force -ErrorAction Stop
        }
        catch {
            throw "PATH_ANCESTOR_UNAVAILABLE"
        }
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "PATH_REPARSE_UNSAFE"
        }
        $parent = [IO.Directory]::GetParent($current)
        if ([string]::IsNullOrEmpty($parent) -or $parent -eq $current) {
            break
        }
        $current = $parent
    }
}

function Test-IsWithin([string]$PathValue, [string]$RootValue) {
    $pathWithSeparator = $PathValue.TrimEnd('\') + '\'
    $rootWithSeparator = $RootValue.TrimEnd('\') + '\'
    return $pathWithSeparator.StartsWith($rootWithSeparator, [StringComparison]::OrdinalIgnoreCase)
}

function Assert-External([string]$PathValue, [string[]]$ProtectedRoots) {
    foreach ($root in $ProtectedRoots) {
        if ([String]::Equals($PathValue, $root, [StringComparison]::OrdinalIgnoreCase) -or (Test-IsWithin $PathValue $root)) {
            throw "PATH_NOT_EXTERNAL"
        }
    }
}

function Assert-NewFile([string]$PathValue) {
    if ([IO.File]::Exists($PathValue) -or [IO.Directory]::Exists($PathValue)) {
        throw "CAPTURE_TARGET_EXISTS"
    }
    try {
        $existing = Get-Item -LiteralPath $PathValue -Force -ErrorAction Stop
        if ($null -ne $existing) {
            throw "CAPTURE_TARGET_EXISTS"
        }
    }
    catch [System.Management.Automation.ItemNotFoundException] {
    }
}

function Invoke-GitValue([string[]]$Arguments) {
    try {
        $output = & git -C $script:RepoRoot @Arguments 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "GIT_FAILURE"
        }
        return ([string]::Join([Environment]::NewLine, [string[]]$output)).Trim()
    }
    catch {
        throw "GIT_FAILURE"
    }
}

function Write-ExclusiveUtf8([string]$PathValue, [string]$TextValue) {
    $encoding = New-Object System.Text.UTF8Encoding($false)
    $bytes = $encoding.GetBytes($TextValue)
    $stream = $null
    try {
        $stream = [IO.File]::Open($PathValue, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $stream.Write($bytes, 0, $bytes.Length)
        $stream.Flush($true)
    }
    finally {
        if ($null -ne $stream) {
            $stream.Dispose()
        }
    }
}

try {
    $script:RepoRoot = Convert-ToFullPath (Join-Path $PSScriptRoot "..")
    $candidate = Convert-ToFullPath $CandidateRoot
    $marker = Convert-ToFullPath $AuthorizationMarker
    $receipt = Convert-ToFullPath $ReceiptPath
    $capture = Convert-ToFullPath $CaptureRoot
    $canonicalPython = Convert-ToFullPath (Join-Path $script:RepoRoot ".venv-real-execution\Scripts\python.exe")

    if ($ImplementationSha -notmatch '^[0-9a-f]{40}$') {
        throw "IMPLEMENTATION_SHA_INVALID"
    }
    if (-not [IO.File]::Exists($canonicalPython)) {
        throw "CANONICAL_PYTHON_UNAVAILABLE"
    }
    if (-not [IO.Directory]::Exists($candidate)) {
        throw "CANDIDATE_ROOT_UNAVAILABLE"
    }
    if (-not [IO.Directory]::Exists($capture)) {
        throw "CAPTURE_ROOT_UNAVAILABLE"
    }

    Assert-NoReparseAncestor $capture
    Assert-External $capture @($script:RepoRoot, $candidate)
    Assert-NewFile (Join-Path $capture "stdout.txt")
    Assert-NewFile (Join-Path $capture "stderr.txt")
    Assert-NewFile (Join-Path $capture "wrapper_evidence.json")

    $stdoutPath = Join-Path $capture "stdout.txt"
    $stderrPath = Join-Path $capture "stderr.txt"
    $evidencePath = Join-Path $capture "wrapper_evidence.json"

    if ((Invoke-GitValue @("config", "--get", "remote.origin.url")) -ne "https://github.com/ta1k1-arakawa/stock-analyzer.git") {
        throw "REPOSITORY_IDENTITY_MISMATCH"
    }
    if ((Invoke-GitValue @("rev-parse", "--abbrev-ref", "HEAD")) -ne "v9-cross-sectional-close-auction-design") {
        throw "BRANCH_MISMATCH"
    }
    if ((Invoke-GitValue @("rev-parse", "HEAD")) -ne $ImplementationSha) {
        throw "HEAD_MISMATCH"
    }
    if ((Invoke-GitValue @("rev-parse", "origin/v9-cross-sectional-close-auction-design")) -ne $ImplementationSha) {
        throw "REMOTE_HEAD_MISMATCH"
    }
    if ((Invoke-GitValue @("status", "--porcelain", "--untracked-files=all"))) {
        throw "WORKTREE_DIRTY"
    }

    $versionOutput = & $canonicalPython --version 2>$null
    if ($LASTEXITCODE -ne 0 -or ([string]::Join(" ", [string[]]$versionOutput)).Trim() -ne "Python 3.12.10") {
        throw "PYTHON_VERSION_MISMATCH"
    }

    $processStartAttempted = $true
    $processStarted = $true
    & $canonicalPython -m scripts.run_v10c_locked_training_cache_provenance_adoption --candidate-root $candidate --implementation-sha $ImplementationSha --authorization-marker $marker --receipt-path $receipt 1> $stdoutPath 2> $stderrPath
    $processExitCode = $LASTEXITCODE

    $evidenceJson = '{"automatic_retry_performed":false,"invocation_mode":"PYTHON_MODULE","network_requests_by_wrapper":0,"process_exit_code":' + $processExitCode + ',"process_start_attempted":true,"process_started":true,"reviewed_implementation_sha":"' + $ImplementationSha + '","runner_module":"scripts.run_v10c_locked_training_cache_provenance_adoption","schema_version":"V10C_PHASE_B_WRAPPER_EVIDENCE_V2","stderr_capture_exists":true,"stdout_capture_exists":true}' + "`n"
    Write-ExclusiveUtf8 $evidencePath $evidenceJson
    exit $processExitCode
}
catch {
    [Console]::Error.WriteLine("V10C_PHASE_B_WRAPPER_FAILURE")
    exit 3
}
