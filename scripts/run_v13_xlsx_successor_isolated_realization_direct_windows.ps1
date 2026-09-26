param(
    [switch] $Execute,
    [string] $ExecutionHead
)

& {
    $ErrorActionPreference = 'Stop'
    if (-not $Execute) {
        Write-Output 'V13_XLSX_ISOLATED_REALIZATION_REHEARSAL=true'
        Write-Output 'NETWORK_REQUESTS=0'
        Write-Output 'WHEEL_DOWNLOADS=0'
        Write-Output 'PACKAGE_INSTALLATIONS=0'
        Write-Output 'ENVIRONMENT_MUTATIONS=0'
        Write-Output 'DURABLE_CANDIDATE_CREATION=0'
        Write-Output 'PRIVATE_READS=0'
        Write-Output 'JPX_YAHOO_REQUESTS=0'
        return
    }
    if ([string]::IsNullOrWhiteSpace($ExecutionHead)) {
        throw 'BLOCK_EXPLICIT_EXECUTION_HEAD_REQUIRED'
    }
    if ($ExecutionHead -cnotmatch '^[0-9a-f]{40}$') {
        throw 'BLOCK_EXECUTION_HEAD_INVALID'
    }
    $repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
    $interpreter = Join-Path $repoRoot '.venv-real-execution\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $interpreter -PathType Leaf)) {
        throw 'BLOCK_CANONICAL_INTERPRETER_MISSING'
    }
    Set-Location -LiteralPath $repoRoot
    & $interpreter -B -m scripts.v13_xlsx_successor_isolated_realization --execute --execution-head $ExecutionHead
    if ($LASTEXITCODE -ne 0) { throw 'BLOCK_ISOLATED_REALIZATION_FAILED' }
}
