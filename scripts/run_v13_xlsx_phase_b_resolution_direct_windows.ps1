param(
    [switch] $Execute,
    [string] $ExecutionHead
)

& {
    $ErrorActionPreference = 'Stop'
    if (-not $Execute) {
        Write-Output 'V13_XLSX_PHASE_B_REHEARSAL=true'
        Write-Output 'PACKAGE_INDEX_REQUESTS=0'
        Write-Output 'WHEEL_DOWNLOADS=0'
        Write-Output 'ENVIRONMENT_MUTATIONS=0'
        Write-Output 'PRIVATE_READS=0'
        Write-Output 'JPX_YAHOO_REQUESTS=0'
        return
    }
    if ([string]::IsNullOrWhiteSpace($ExecutionHead)) {
        throw 'BLOCK_EXPLICIT_EXECUTION_INPUT_REQUIRED'
    }
    $repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
    $interpreter = Join-Path $repoRoot '.venv-real-execution\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $interpreter -PathType Leaf)) {
        throw 'BLOCK_CANONICAL_INTERPRETER_MISSING'
    }
    Set-Location -LiteralPath $repoRoot
    & $interpreter -m scripts.v13_xlsx_phase_b_resolution --execute --execution-head $ExecutionHead
    if ($LASTEXITCODE -ne 0) { throw 'BLOCK_PHASE_B_RESOLUTION_FAILED' }
}
