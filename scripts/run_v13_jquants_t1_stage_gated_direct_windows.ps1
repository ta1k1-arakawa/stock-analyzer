[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedHead,
    [string]$RepositoryRoot,
    [switch]$ExecuteReviewedPrivateRead
)

& {
    $ErrorActionPreference = 'Stop'
    $stage = 'POWERSHELL_VERSION'
    $privateCallStarted = $false
    $report = ''
    $gateName = 'V13_JQUANTS_T1_WRAPPER_GATE'
    $branchName = 'v13-conditional-cross-sectional-short-horizon'
    $runnerRelative = 'scripts/run_v13_jquants_t1_exclusion_direct_windows.ps1'
    $runnerBlob = 'b8e8d21d097ae6a408a7935f3470895323b28760'
    $resolverBlob = 'f859c7998954445e2afe0800ecd861a2423f6a8e'
    $harnessBlob = '50b997b761ed539e25334838e6e09fe67c09e051'
    $authorizationBlob = 'c36e9cbe84525a3be282fb68a8a0b17b7d828dc9'
    $authorizationRelative = 'V13_JQUANTS_T1_EXCLUSION_POINT_OF_USE_AUTHORIZATION.json'

    function Invoke-NativeChecked([string]$Executable, [string[]]$CommandArguments) {
        # Windows PowerShell 5.1 can turn redirected native stderr into an
        # ErrorRecord. The exit code, not stderr text, determines success.
        $savedPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            $outputLines = @(& $Executable @CommandArguments 2>$null)
            $exitCode = $LASTEXITCODE
        }
        finally { $ErrorActionPreference = $savedPreference }
        return [pscustomobject]@{ Lines = $outputLines; ExitCode = $exitCode }
    }

    function Git-Value([string[]]$CommandArguments) {
        $result = Invoke-NativeChecked 'git.exe' $CommandArguments
        if ($result.ExitCode -ne 0) { throw 'NATIVE_COMMAND_FAILED' }
        return (($result.Lines -join "`n").Trim())
    }

    function Assert-Repository([string]$Root, [string]$Head) {
        if (-not (Test-Path -LiteralPath (Join-Path $Root '.git') -PathType Container)) { throw 'REPOSITORY_BLOCK' }
        if ($Root -match '(?i)[\\/]orca[\\/]workspaces[\\/]') { throw 'GENERATED_WORKTREE_BLOCK' }
        if ((Git-Value @('-C', $Root, 'rev-parse', '--show-toplevel')).Replace('/', '\').TrimEnd('\') -cne $Root.Replace('/', '\').TrimEnd('\')) { throw 'REPOSITORY_BLOCK' }
        if ((Git-Value @('-C', $Root, 'branch', '--show-current')) -cne $branchName) { throw 'BRANCH_BLOCK' }
        if ((Git-Value @('-C', $Root, 'rev-parse', 'HEAD')) -cne $Head) { throw 'LOCAL_HEAD_BLOCK' }
        if ((Git-Value @('-C', $Root, 'status', '--porcelain')) -ne '') { throw 'DIRTY_TREE_BLOCK' }
        if ((Git-Value @('-C', $Root, 'remote', 'get-url', 'origin')) -notin @('https://github.com/ta1k1-arakawa/stock-analyzer.git', 'git@github.com:ta1k1-arakawa/stock-analyzer.git')) { throw 'ORIGIN_BLOCK' }
        $remoteLine = Git-Value @('-C', $Root, 'ls-remote', '--exit-code', 'origin', "refs/heads/$branchName")
        if (($remoteLine -split '\s+')[0] -cne $Head) { throw 'REMOTE_HEAD_BLOCK' }
        foreach ($binding in @(
            @($runnerRelative, $runnerBlob),
            @('scripts/v13_resolve_jquants_t1_exclusion_state.py', $resolverBlob),
            @('scripts/v13_execute_jquants_t1_exclusion_private_read.py', $harnessBlob),
            @($authorizationRelative, $authorizationBlob)
        )) {
            if ((Git-Value @('-C', $Root, 'rev-parse', "${Head}:$($binding[0])")) -cne $binding[1]) { throw 'BLOB_BLOCK' }
            if ((Git-Value @('-C', $Root, 'hash-object', '--path', $binding[0], $binding[0])) -cne $binding[1]) { throw 'WORKTREE_BLOB_BLOCK' }
        }
        $selfRelative = 'scripts/run_v13_jquants_t1_stage_gated_direct_windows.ps1'
        if (([IO.Path]::GetFullPath($PSCommandPath)) -eq ([IO.Path]::GetFullPath((Join-Path $Root $selfRelative)))) {
            $committed = Git-Value @('-C', $Root, 'rev-parse', "${Head}:$selfRelative")
            if ((Git-Value @('-C', $Root, 'hash-object', '--path', $selfRelative, $selfRelative)) -cne $committed) { throw 'SELF_BLOB_BLOCK' }
        }
    }

    function Assert-Report([string[]]$Lines, [string[]]$RequiredLines) {
        foreach ($requiredLine in $RequiredLines) {
            if (@($Lines | Where-Object { $_ -ceq $requiredLine }).Count -ne 1) { throw 'REPORT_BLOCK' }
        }
        if (@($Lines | Where-Object { $_ -notmatch '^[A-Z][A-Z0-9_]*=(?:[A-Z][A-Z0-9_]*|true|false|unknown|[0-9]+)(?: [A-Z][A-Z0-9_]*=(?:[A-Z][A-Z0-9_]*|true|false|unknown|[0-9]+))*$' }).Count -ne 0) { throw 'REPORT_BLOCK' }
    }

    try {
        if ($PSVersionTable.PSVersion.Major -ne 5 -or $PSVersionTable.PSVersion.Minor -ne 1) { throw 'POWERSHELL_VERSION_BLOCK' }
        Write-Output "POWERSHELL_VERSION=$($PSVersionTable.PSVersion.ToString())"
        $root = if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) { [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')) } else { [IO.Path]::GetFullPath($RepositoryRoot) }
        $trackedLauncher = [IO.Path]::GetFullPath((Join-Path $root 'scripts/run_v13_jquants_t1_stage_gated_direct_windows.ps1'))
        if ($ExecuteReviewedPrivateRead -and ([IO.Path]::GetFullPath($PSCommandPath) -cne $trackedLauncher)) { throw 'REVIEWED_ENTRYPOINT_BLOCK' }
        $stage = 'EXACT_LOCAL_REMOTE_HEAD_AND_CLEAN_PREFLIGHT'
        Assert-Repository $root $ExpectedHead
        Write-Output "$stage=PASS"

        $python = Join-Path $root '.venv-real-execution\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'PYTHON_ENVIRONMENT_BLOCK' }
        $testPackages = Join-Path $root '.venv\Lib\site-packages'
        if (-not (Test-Path -LiteralPath (Join-Path $testPackages 'pytest') -PathType Container)) { throw 'SYNTHETIC_TEST_DEPENDENCY_BLOCK' }
        # The frozen protected environment has no pytest. Import the existing
        # test tool only in this synthetic test process; do not mutate either
        # environment or pass this import path to the protected child.
        $escapedPackages = $testPackages.Replace("'", "''")
        $testCode = "import sys; sys.path.insert(0, r'$escapedPackages'); import pytest; raise SystemExit(pytest.main(sys.argv[1:]))"
        $testTemp = Join-Path ([IO.Path]::GetTempPath()) ('v13-issue82-synthetic-' + [guid]::NewGuid().ToString('N'))
        $stage = 'SAME_MACHINE_SYNTHETIC_TESTS'
        Push-Location -LiteralPath $root
        try {
            $testResult = Invoke-NativeChecked $python @('-E', '-B', '-c', $testCode, '-q', '-p', 'no:cacheprovider', '--basetemp', $testTemp, 'tests/test_v13_resolve_jquants_t1_exclusion_state.py', 'tests/test_v13_execute_jquants_t1_exclusion_private_read.py')
        }
        finally { Pop-Location }
        if ($testResult.ExitCode -ne 0) { throw 'SYNTHETIC_TEST_BLOCK' }
        Write-Output "$stage=PASS"

        $runner = Join-Path $root $runnerRelative
        $runnerArguments = @('-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', $runner,
            '-ExpectedHead', $ExpectedHead, '-ExpectedScriptBlob', $runnerBlob,
            '-ExpectedResolverBlob', $resolverBlob, '-ExpectedHarnessBlob', $harnessBlob,
            '-FutureAuthorizationBlob', $authorizationBlob,
            '-FutureAuthorizationRelativePath', $authorizationRelative)
        $stage = 'REVIEWED_WRAPPER_PRE_GATE_REHEARSAL'
        $wrapperResult = Invoke-NativeChecked 'powershell.exe' $runnerArguments
        if ($wrapperResult.ExitCode -ne 1) { throw 'WRAPPER_REHEARSAL_BLOCK' }
        Assert-Report $wrapperResult.Lines @('EXECUTION_RESULT=PRE_GATE_STOP FAILURE_CLASS=PRE_GATE_EXECUTION_SWITCH_REQUIRED PRIVATE_BOUNDARY_CROSSED=false AUTHORIZATION_CONSUMED=false AUTHORIZATION_REUSABLE=false SECOND_EXECUTION_ALLOWED=false SOURCE_OPENS=0 NETWORK_REQUESTS=0 PRICE_PAYLOAD_READS=0 OUTCOME_READS=0 NON_T1_IDENTITIES_RETAINED=false V13_UNIVERSE_SELECTED=false')
        Write-Output "$stage=PASS"

        $stage = 'WRAPPER_GATE_ENV_ABSENCE_PROOF_PROCESS_USER_MACHINE'
        foreach ($scope in @('Process', 'User', 'Machine')) {
            if ($null -ne [Environment]::GetEnvironmentVariable($gateName, $scope)) { throw 'WRAPPER_GATE_PRESENT_BLOCK' }
        }
        Write-Output "$stage=PASS"

        $stage = 'CHILD_PROCESS_REPORT_REHEARSAL'
        Push-Location -LiteralPath $root
        try {
            $childResult = Invoke-NativeChecked $python @('-E', '-B', '-m', 'scripts.v13_execute_jquants_t1_exclusion_private_read', '--repository-root', $root)
        }
        finally { Pop-Location }
        if ($childResult.ExitCode -ne 1) { throw 'CHILD_REHEARSAL_BLOCK' }
        Assert-Report $childResult.Lines @('PRE_GATE_STATUS=FAIL', 'PRIVATE_BOUNDARY_CROSSED=false', 'AUTHORIZATION_CONSUMED=false', 'SOURCE_OPENS=0', 'PRIVATE_CONTENT_READS=0', 'NETWORK_REQUESTS=0', 'PRICE_PAYLOAD_READS=0', 'OUTCOME_READS=0', 'NON_T1_IDENTITIES_RETAINED=false', 'V13_UNIVERSE_SELECTED=false', 'CONSUMED_RECEIPT_WRITTEN=false', 'PRIVATE_STATE_WRITTEN=false', 'EXECUTION_RESULT=PRE_GATE_STOP', 'FAILURE_CLASS=PRE_GATE_WRAPPER_REQUIRED', 'AUTOMATIC_RETRY=false', 'SECOND_PRIVATE_SOURCE_READ=false')
        if ($childResult.Lines.Count -ne 19) { throw 'CHILD_REPORT_SHAPE_BLOCK' }
        Write-Output "$stage=PASS"

        $stage = 'FINAL_HEAD_REMOTE_CLEAN_RECHECK'
        Assert-Repository $root $ExpectedHead
        Write-Output "$stage=PASS"
        if (-not $ExecuteReviewedPrivateRead) {
            Write-Output 'NO_PRIVATE_REHEARSAL_RESULT=PASS'
            Write-Output 'PRIVATE_BOUNDARY_CROSSED=false'
            Write-Output 'AUTHORIZATION_CONSUMED=false'
            Write-Output 'PRIVATE_CONTENT_READS=0'
            Write-Output 'NETWORK_REQUESTS_MARKET_DATA=0'
            return
        }

        $stage = 'ONE_REAL_T1_PRIVATE_READ'
        $privateCallStarted = $true
        $realResult = Invoke-NativeChecked 'powershell.exe' ($runnerArguments + @('-ExecuteReviewedPrivateRead'))
        $report = $realResult.Lines -join "`n"
        if ($realResult.ExitCode -ne 0 -or $report -cnotmatch '(?m)^EXECUTION_RESULT=PASS$') { throw 'REAL_REPORT_REQUIRES_ADJUDICATION' }
        Write-Output $report
    }
    catch {
        $failure = if ($privateCallStarted) { 'POST_BOUNDARY_STATUS_UNKNOWN' } else { 'PRE_GATE_STOP' }
        Write-Output "LAUNCHER_RESULT=$failure STAGE=$stage PRIVATE_BOUNDARY_CROSSED=$(if ($privateCallStarted) { 'unknown' } else { 'false' }) AUTHORIZATION_CONSUMED=$(if ($privateCallStarted) { 'unknown' } else { 'false' })"
        exit 1
    }
}
