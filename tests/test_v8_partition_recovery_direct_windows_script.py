from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v8_partition_recovery_direct_windows.ps1"


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_script_is_atomic_and_cleans_transient_state():
    source = _script()
    assert source.startswith("& {\n")
    assert "$ErrorActionPreference = 'Stop'" in source
    assert "    finally {" in source
    assert "SetEnvironmentVariable($pythonPayloadPath, $null, 'Process')" in source
    assert "SetEnvironmentVariable($pythonArtifactPath, $null, 'Process')" in source
    assert "SetEnvironmentVariable($pythonPayloadHash, $null, 'Process')" in source
    assert "[System.IO.File]::Delete($temporaryPayload)" in source


def test_all_preflight_gates_precede_network_boundary():
    source = _script()
    boundary = source.index("$networkBoundaryCrossed = $true")
    for gate in (
        "PRE_GATE_NOT_AUTHORITATIVE_CHECKOUT",
        "PRE_GATE_GENERATED_WORKTREE",
        "PRE_GATE_WRONG_BRANCH",
        "PRE_GATE_REMOTE_HEAD_MISMATCH",
        "PRE_GATE_LOCAL_HEAD_MISMATCH",
        "PRE_GATE_DIRTY_WORKTREE",
        "PRE_GATE_REVIEWED_BLOB_MISMATCH",
        "PRE_GATE_ARTIFACT_ROOT_INSIDE_REPOSITORY",
        "PRE_GATE_ARTIFACT_ALREADY_EXISTS",
        "PRE_GATE_PROTECTED_ENVIRONMENT_BLOCK",
        "PRE_GATE_OPERATION_PARSER_BLOCK",
    ):
        assert source.index(gate) < boundary
    assert "'ls-remote', '--exit-code'" in source
    assert "fetch" not in source.lower()


def test_destination_is_mechanical_outside_repository_and_write_once():
    source = _script()
    assert "Join-Path $localAppData 'stock-analyzer\\private\\v8-recovery'" in source
    assert "GetFullPath" in source
    assert "StartsWith($repoPrefix" in source
    assert "CreateDirectory($artifactRoot)" in source
    assert source.index("CreateDirectory($artifactRoot)") < source.index("$networkBoundaryCrossed = $true")
    assert "PRE_GATE_ARTIFACT_ALREADY_EXISTS" in source
    assert "write_v8_partition_recovery_manifest_once" not in source
    assert "recover_and_publish_v8_partition_once(" in source


def test_request_guard_is_one_shot_and_no_redirect_or_retry():
    source = _script()
    assert source.count("$requestCount = 1") == 1
    assert source.count("$request.GetResponse()") == 1
    assert "$request.AllowAutoRedirect = $false" in source
    assert "ComputeHash($payloadBytes)" in source
    assert "hashlib.sha256(raw).hexdigest() != os.environ[\"V8_RECOVERY_TRANSIENT_SHA256\"]" in source
    boundary = source.index("$networkBoundaryCrossed = $true")
    network_operation = source[boundary:]
    assert "while (" not in network_operation.lower()
    assert "foreach (" not in network_operation.lower()
    assert "retry" not in network_operation.lower()


def test_safe_report_does_not_emit_paths_payload_or_member_assignments():
    source = _script()
    assert "RECOVERY_RESULT=PASS NETWORK_BOUNDARY_CROSSED=true JPX_SOURCE_REQUESTS=1" in source
    assert "SEALED_IDENTITIES_PUBLICLY_DISCLOSED=false" in source
    assert "Write-Output $artifactPath" not in source
    assert "Write-Output $temporaryPayload" not in source
    assert "Write-Output $payloadBytes" not in source
    assert "RECOVERY_RESULT=NOT_EXECUTED NETWORK_BOUNDARY_CROSSED=false JPX_SOURCE_REQUESTS=0" in source
