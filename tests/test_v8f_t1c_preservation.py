from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8e_t1c_preservation as v8e_preservation
from src import v8_partition as v8_partition_module
from src import v8f_t1c_preservation as preservation
from src.v8c_git_provenance import CANONICAL_REPOSITORY_ROOT
from src.v8c_t1c_allocation import build_t1c_allocation_artifact


AUTHORIZATION = (
    preservation.V8F_AUTHORIZATION_PREFIX
    + preservation.V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT
    + preservation.V8F_AUTHORIZATION_SEPARATOR
    + preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH
)
REVIEWED_SUPPORT_SHA = "1" * 40


def _clock():
    return datetime(2026, 8, 18, tzinfo=timezone.utc)


def _preflight(**overrides):
    value = {
        "repository_identity": preservation.V8F_REPOSITORY_IDENTITY,
        "branch": preservation.V8F_PRODUCTION_BRANCH,
        "head": "1" * 40,
        "origin_head": "1" * 40,
        "worktree_clean": True,
        "reviewed_v8f_design_candidate_commit": preservation.V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "reviewed_v8f_design_blob_sha": preservation.V8F_DESIGN_CANDIDATE_BLOB_SHA,
        "v8e_terminal_commit": preservation.V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
        "v8e_terminal_blob_sha": preservation.V8F_V8E_TERMINAL_RECORD_BLOB_SHA,
        "trusted_partition_blob_sha": preservation.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "partition_manifest_sha256": preservation.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "partition_implementation_commit": preservation.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "v8c_allocation_implementation_commit": preservation.EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT,
        "parent_t_spare_ticker_count": preservation.EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
        "parent_t_spare_ticker_list_sha256": preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
    }
    value.update(overrides)
    return value


def _runtime_support_state(**overrides):
    value = {
        "branch": preservation.V8F_PRODUCTION_BRANCH,
        "head": REVIEWED_SUPPORT_SHA,
        "origin_head": REVIEWED_SUPPORT_SHA,
        "worktree_clean": True,
        "commits_after_reviewed_support_sha": 0,
    }
    value.update(overrides)
    return value


def _consume(state_root: Path):
    return preservation.consume_gate_once(state_root, AUTHORIZATION, clock=_clock)


def _receipt_raw(state_root: Path) -> tuple[str, bytes]:
    key = preservation.compute_receipt_key(AUTHORIZATION)
    return key, (state_root / f"{key}.json").read_bytes()


def _exact_artifact():
    return {
        "schema_version": "V8F_T1C_PRESERVATION_RECHECK_V1",
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "study": preservation.V8F_STUDY_NAME,
        "reviewed_v8f_design_candidate_commit": preservation.V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "source_v8e_terminal_commit": preservation.V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
        "allocation_artifact_self_hash": preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": preservation.EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": preservation.EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
        "parent_t_spare_ticker_list_sha256": preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "remaining_t_spare_ticker_list_sha256": preservation.EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "t1c_raw_acquisition_performed": False,
        "t1c_research_opened": False,
        "t1c_ohlcv_research_access": False,
        "t1c_feature_access": False,
        "t1c_outcome_access": False,
        "t1c_identities_publicly_exposed": False,
        "t1c_membership_reassigned": False,
        "allocation_self_hash_unchanged": True,
        "parent_v8_provenance_unchanged": True,
        "v8e_terminal_adjudication_authoritative": True,
        "preservation_recheck_result": "PASS",
    }


def _fresh_public_evidence(**overrides):
    value = {
        "schema_version": "V8F_T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE_V1",
        "evidence_role": "T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE",
        "study": preservation.V8F_STUDY_NAME,
        "reviewed_v8f_design_candidate_commit": preservation.V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "v8e_predecessor_terminal_commit": preservation.V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
        "v8e_terminal_disposition": preservation.V8F_V8E_TERMINAL_DISPOSITION,
        "v8e_terminal_failure_class": preservation.V8F_V8E_TERMINAL_FAILURE_CLASS,
        "v8e_terminal_artifact_blob_sha": preservation.V8F_V8E_TERMINAL_RECORD_BLOB_SHA,
        "allocation_artifact_self_hash": preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": preservation.EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": preservation.EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
        "parent_t_spare_ticker_list_sha256": preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "remaining_t_spare_ticker_list_sha256": preservation.EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "t1c_raw_acquisition_performed": False,
        "t1c_research_opened": False,
        "t1c_ohlcv_research_access": False,
        "t1c_feature_access": False,
        "t1c_outcome_access": False,
        "t1c_identities_publicly_exposed": False,
        "t1c_membership_reassigned": False,
        "allocation_self_hash_unchanged": True,
        "parent_v8_provenance_unchanged": True,
        "v8e_terminal_adjudication_authoritative": True,
        "fresh_public_preservation_evidence_result": "PASS",
    }
    value.update(overrides)
    return value


def _private_summary():
    return {
        "allocation_artifact_self_hash": preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": preservation.EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": preservation.EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
        "parent_t_spare_ticker_list_sha256": preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "remaining_t_spare_ticker_list_sha256": preservation.EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "t1c_membership_reassigned": False,
        "allocation_self_hash_unchanged": True,
        "parent_v8_provenance_unchanged": True,
    }


# ---------------------------------------------------------------------------
# Exact candidate/blob binding and V8F namespace exactness
# ---------------------------------------------------------------------------


def test_exact_candidate_and_authorization_grammar():
    preservation.validate_authorization_identity(AUTHORIZATION)
    assert preservation.authorization_identity_sha256(AUTHORIZATION) == hashlib.sha256(
        AUTHORIZATION.encode("utf-8")
    ).hexdigest()
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.validate_authorization_identity(AUTHORIZATION.replace("cd67a9", "ed67a9", 1))
    assert excinfo.value.reason == "V8F_AUTHORIZATION_GRAMMAR_MISMATCH"
    assert AUTHORIZATION not in str(excinfo.value)


def test_wrong_v8f_candidate_blocks():
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.validate_authorization_identity(
            AUTHORIZATION, reviewed_v8f_design_candidate_commit="0" * 40
        )
    assert excinfo.value.reason == "V8F_DESIGN_CANDIDATE_MISMATCH"


def test_namespace_literals_are_v8f_exact():
    assert preservation.V8F_STUDY_NAME == "V8F_HISTORICAL_RESEARCH"
    assert preservation.V8F_T1C_PRESERVATION_GATE == "HUMAN_V8F_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE"
    assert preservation.V8F_RECEIPT_SCHEMA_VERSION == "V8F_T1C_PRESERVATION_GATE_RECEIPT_V1"
    assert _exact_artifact()["schema_version"] == "V8F_T1C_PRESERVATION_RECHECK_V1"
    assert set(preservation.V8F_RECEIPT_FIELDS) == {
        "schema_version",
        "study",
        "artifact_role",
        "gate",
        "reviewed_v8f_design_candidate_commit",
        "authorization_identity_sha256",
        "authorized_allocation_artifact_self_hash",
        "consumed",
        "consumption_count",
        "consumption_boundary",
        "consumption_timestamp_utc",
    }


def test_consumption_boundary_is_exact():
    assert preservation.V8F_RECEIPT_CONSUMPTION_BOUNDARY == "IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ"


# ---------------------------------------------------------------------------
# Deterministic receipt key
# ---------------------------------------------------------------------------


def test_receipt_key_is_exact_and_deterministic():
    identity_hash = hashlib.sha256(AUTHORIZATION.encode("utf-8")).hexdigest()
    material = "|".join(
        (
            preservation.V8F_REPOSITORY_IDENTITY,
            preservation.V8F_T1C_PRESERVATION_GATE,
            preservation.V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
            identity_hash,
            preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        )
    )
    expected = hashlib.sha256(material.encode("utf-8")).hexdigest()
    assert preservation.compute_receipt_key(AUTHORIZATION) == expected
    assert preservation.compute_receipt_key(AUTHORIZATION) == expected


# ---------------------------------------------------------------------------
# Receipt: raw identity absent, duplicate/missing/extra rejection, one-shot
# ---------------------------------------------------------------------------


def test_receipt_has_exact_fields_and_never_persists_raw_identity(tmp_path):
    receipt = _consume(tmp_path)
    key, raw = _receipt_raw(tmp_path)
    assert set(receipt) == set(preservation.V8F_RECEIPT_FIELDS)
    assert set(preservation.read_gate_receipt(tmp_path, key)) == set(preservation.V8F_RECEIPT_FIELDS)
    assert AUTHORIZATION.encode("utf-8") not in raw
    assert preservation.V8F_RECEIPT_FIELDS == tuple(receipt)


@pytest.mark.parametrize("field", ["extra", "missing"])
def test_receipt_missing_or_extra_field_blocks(tmp_path, field):
    _consume(tmp_path)
    key, _ = _receipt_raw(tmp_path)
    path = tmp_path / f"{key}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if field == "extra":
        value["extra"] = True
    else:
        del value["consumption_timestamp_utc"]
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.read_gate_receipt(tmp_path, key)
    assert excinfo.value.reason == "V8F_RECEIPT_SCHEMA_INVALID"


def test_duplicate_receipt_field_blocks(tmp_path):
    _consume(tmp_path)
    key, raw = _receipt_raw(tmp_path)
    duplicate = raw.replace(b'"consumed":true', b'"consumed":true,"consumed":true', 1)
    (tmp_path / f"{key}.json").write_bytes(duplicate)
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.read_gate_receipt(tmp_path, key)
    assert excinfo.value.reason == "V8F_RECEIPT_DUPLICATE_KEY"


def test_one_shot_no_overwrite(tmp_path):
    _consume(tmp_path)
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        _consume(tmp_path)
    assert excinfo.value.reason == "V8F_GATE_ALREADY_CONSUMED"


# ---------------------------------------------------------------------------
# V8E authorization/receipt cannot authorize V8F
# ---------------------------------------------------------------------------


def test_v8e_authorization_identity_cannot_authorize_v8f():
    v8e_authorization = (
        v8e_preservation.V8E_AUTHORIZATION_PREFIX
        + v8e_preservation.V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT
        + v8e_preservation.V8E_AUTHORIZATION_SEPARATOR
        + v8e_preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH
    )
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.validate_authorization_identity(v8e_authorization)
    assert excinfo.value.reason in ("V8F_AUTHORIZATION_GRAMMAR_MISMATCH", "V8F_DESIGN_CANDIDATE_MISMATCH")


def test_v8e_receipt_cannot_be_read_as_v8f_receipt(tmp_path):
    v8e_receipt = v8e_preservation.consume_gate_once(
        tmp_path, v8e_preservation.V8E_AUTHORIZATION_PREFIX
        + v8e_preservation.V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT
        + v8e_preservation.V8E_AUTHORIZATION_SEPARATOR
        + v8e_preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        clock=_clock,
    )
    v8e_key = v8e_preservation.compute_receipt_key(
        v8e_preservation.V8E_AUTHORIZATION_PREFIX
        + v8e_preservation.V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT
        + v8e_preservation.V8E_AUTHORIZATION_SEPARATOR
        + v8e_preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH
    )
    assert v8e_receipt["gate"] != preservation.V8F_T1C_PRESERVATION_GATE
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._validate_receipt(v8e_receipt, v8e_key)
    # The V8E receipt's field set itself differs (reviewed_v8e_design_candidate_commit
    # vs reviewed_v8f_design_candidate_commit), so exact-schema rejection fires
    # before the gate-literal check is even reached -- V8E authority is rejected
    # at the earliest possible point, not merely at one specific field.
    assert excinfo.value.reason == "V8F_RECEIPT_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# Public preflight / reviewed-support-sha binding
# ---------------------------------------------------------------------------


def test_preflight_failure_has_zero_private_reads_and_zero_receipt(tmp_path):
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic")
    manifest.write_bytes(b"synthetic")
    reads = []
    bad = _preflight(reviewed_v8f_design_blob_sha="0" * 40)
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._execute_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
            state_root=tmp_path / "state",
            output_path=tmp_path / "artifact.json",
            allocation_artifact_path=allocation,
            partition_manifest_path=manifest,
            repository_root=tmp_path / "repo",
            public_preflight=lambda: bad,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(),
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=preservation.consume_gate_once,
            clock=_clock,
        )
    assert reads == []
    assert not (tmp_path / "state").exists()


def test_exact_reviewed_support_sha_runtime_binding_passes():
    assert preservation._validate_reviewed_support_implementation_binding(
        Path("synthetic-repository"),
        _preflight(),
        REVIEWED_SUPPORT_SHA,
        runtime_state_reader=lambda _root, _sha: _runtime_support_state(),
    ) == REVIEWED_SUPPORT_SHA


@pytest.mark.parametrize(
    "reviewed_sha,runtime_overrides",
    [
        ("malformed", {}),
        (REVIEWED_SUPPORT_SHA, {"head": "2" * 40}),
        (REVIEWED_SUPPORT_SHA, {"origin_head": "2" * 40}),
        (REVIEWED_SUPPORT_SHA, {"commits_after_reviewed_support_sha": 1}),
        (REVIEWED_SUPPORT_SHA, {"branch": "other-branch"}),
        (REVIEWED_SUPPORT_SHA, {"worktree_clean": False}),
    ],
)
def test_reviewed_support_sha_runtime_mismatch_blocks(reviewed_sha, runtime_overrides):
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._validate_reviewed_support_implementation_binding(
            Path("synthetic-repository"),
            _preflight(),
            reviewed_sha,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(**runtime_overrides),
        )


def test_historical_v8e_pass_is_not_v8f_fresh_authority():
    historical_only = {
        "schema_version": "V8E_T1C_READINESS_TERMINAL_ADJUDICATION_V1",
        "disposition": "BLOCK_CLOSED",
    }
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._validate_fresh_t1c_public_evidence(historical_only)


# ---------------------------------------------------------------------------
# Forbidden T1C access / reassignment flags block
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "t1c_raw_acquisition_performed",
        "t1c_research_opened",
        "t1c_ohlcv_research_access",
        "t1c_feature_access",
        "t1c_outcome_access",
        "t1c_identities_publicly_exposed",
        "t1c_membership_reassigned",
    ],
)
def test_fresh_public_evidence_rejects_each_absence_contradiction(field):
    evidence = _fresh_public_evidence(**{field: True})
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._validate_fresh_t1c_public_evidence(evidence)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_fresh_public_evidence_exact_schema(mutation):
    evidence = _fresh_public_evidence()
    if mutation == "missing":
        del evidence["t1c_outcome_access"]
    else:
        evidence["unexpected"] = False
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._validate_fresh_t1c_public_evidence(evidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("v8e_predecessor_terminal_commit", "0" * 40),
        ("v8e_terminal_disposition", "PASS"),
        ("reviewed_v8f_design_candidate_commit", "0" * 40),
        ("allocation_artifact_self_hash", "0" * 64),
        ("parent_v8_provenance_unchanged", False),
    ],
)
def test_fresh_public_evidence_bindings_fail_closed(field, value):
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._validate_fresh_t1c_public_evidence(_fresh_public_evidence(**{field: value}))


def test_public_artifact_absence_fields_are_from_fresh_evidence():
    artifact = preservation._build_public_artifact(_private_summary(), _fresh_public_evidence())
    for field in (
        "t1c_raw_acquisition_performed",
        "t1c_research_opened",
        "t1c_ohlcv_research_access",
        "t1c_feature_access",
        "t1c_outcome_access",
        "t1c_identities_publicly_exposed",
        "t1c_membership_reassigned",
    ):
        assert artifact[field] is False


def test_fresh_public_evidence_has_no_sensitive_values():
    public = json.dumps(_fresh_public_evidence(), sort_keys=True)
    assert AUTHORIZATION not in public
    assert "allocation_artifact_path" not in public
    assert "partition_manifest_path" not in public
    assert "raw_payload" not in public


# ---------------------------------------------------------------------------
# Private reader not called before gate boundary; gate consumed before reads
# ---------------------------------------------------------------------------


def test_public_fresh_evidence_failure_precedes_gate_and_private_read(tmp_path):
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic allocation")
    manifest.write_bytes(b"synthetic manifest")
    reads = []
    gate_calls = []
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._execute_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
            state_root=tmp_path / "state",
            output_path=tmp_path / "artifact.json",
            allocation_artifact_path=allocation,
            partition_manifest_path=manifest,
            repository_root=tmp_path / "repo",
            public_preflight=_preflight,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(),
            public_evidence_resolver=lambda _: {"bad": True},
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=lambda *args, **kwargs: gate_calls.append((args, kwargs)),
            clock=_clock,
        )
    assert reads == []
    assert gate_calls == []
    assert not (tmp_path / "state").exists()


def test_reviewed_support_failure_precedes_gate_private_read_and_receipt(tmp_path):
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic allocation")
    manifest.write_bytes(b"synthetic manifest")
    reads = []
    gate_calls = []
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._execute_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
            state_root=tmp_path / "state",
            output_path=tmp_path / "artifact.json",
            allocation_artifact_path=allocation,
            partition_manifest_path=manifest,
            repository_root=tmp_path / "repo",
            public_preflight=_preflight,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(
                head="2" * 40, commits_after_reviewed_support_sha=1
            ),
            public_evidence_resolver=lambda _: _fresh_public_evidence(),
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=lambda *args, **kwargs: gate_calls.append((args, kwargs)),
            clock=_clock,
        )
    assert reads == []
    assert gate_calls == []
    assert not (tmp_path / "state").exists()


def test_gate_consumed_immediately_before_first_synthetic_private_read(tmp_path):
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic allocation")
    manifest.write_bytes(b"synthetic manifest")
    reads = []
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._execute_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
            state_root=tmp_path / "state",
            output_path=tmp_path / "artifact.json",
            allocation_artifact_path=allocation,
            partition_manifest_path=manifest,
            repository_root=tmp_path / "repo",
            public_preflight=_preflight,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(),
            public_evidence_resolver=lambda _: _fresh_public_evidence(),
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=preservation.consume_gate_once,
            clock=_clock,
        )
    # Synthetic private reader is only invoked after synthetic gate consumption.
    assert len(reads) == 2
    assert list((tmp_path / "state").glob("*.json"))


def test_consumed_receipt_is_retained_after_post_consumption_failure(tmp_path):
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic allocation")
    manifest.write_bytes(b"synthetic manifest")
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._execute_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
            state_root=tmp_path / "state",
            output_path=tmp_path / "artifact.json",
            allocation_artifact_path=allocation,
            partition_manifest_path=manifest,
            repository_root=tmp_path / "repo",
            public_preflight=_preflight,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(),
            public_evidence_resolver=lambda _: _fresh_public_evidence(),
            private_reader=lambda path: path.read_bytes(),
            gate_consumer=preservation.consume_gate_once,
            clock=_clock,
        )
    key = preservation.compute_receipt_key(AUTHORIZATION)
    assert preservation.read_gate_receipt(tmp_path / "state", key)["consumed"] is True


def test_no_network_and_no_real_private_read_module_wide():
    import ast
    import inspect

    source = inspect.getsource(preservation)
    tree = ast.parse(source)
    forbidden = {"urlopen", "socket", "requests", "httpx"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (forbidden & names)
    assert not (forbidden & attrs)


# ---------------------------------------------------------------------------
# Independent artifact + receipt verification; T1C count/hash/provenance
# mismatch blocks
# ---------------------------------------------------------------------------


def test_artifact_independent_verification_exact_schema_and_receipt(tmp_path):
    _consume(tmp_path)
    key, receipt_raw = _receipt_raw(tmp_path)
    artifact_raw = json.dumps(_exact_artifact(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = preservation.verify_t1c_preservation_artifact_bytes(
        artifact_raw, receipt_raw, receipt_key=key
    )
    assert result["result"] == "PASS"
    assert result["receipt_validation_result"] == "PASS"
    assert len(result["gate_receipt_bytes_sha256"]) == 64


@pytest.mark.parametrize("mutation", ["missing", "extra", "candidate", "allocation"])
def test_artifact_schema_or_binding_mismatch_blocks(tmp_path, mutation):
    _consume(tmp_path)
    key, receipt_raw = _receipt_raw(tmp_path)
    artifact = _exact_artifact()
    if mutation == "missing":
        del artifact["preservation_recheck_result"]
    elif mutation == "extra":
        artifact["extra"] = True
    elif mutation == "candidate":
        artifact["reviewed_v8f_design_candidate_commit"] = "0" * 40
    else:
        artifact["allocation_artifact_self_hash"] = "0" * 64
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation.verify_t1c_preservation_artifact_bytes(
            json.dumps(artifact).encode("utf-8"), receipt_raw, receipt_key=key
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("t1c_ticker_count", 299),
        ("t1c_ticker_list_sha256", "0" * 64),
        ("parent_t_spare_ticker_list_sha256", "0" * 64),
        ("remaining_t_spare_ticker_list_sha256", "0" * 64),
        ("allocation_artifact_self_hash", "0" * 64),
    ],
)
def test_t1c_count_hash_provenance_mismatch_blocks(tmp_path, field, value):
    _consume(tmp_path)
    key, receipt_raw = _receipt_raw(tmp_path)
    artifact = _exact_artifact()
    artifact[field] = value
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation.verify_t1c_preservation_artifact_bytes(
            json.dumps(artifact).encode("utf-8"), receipt_raw, receipt_key=key
        )


def test_public_artifact_contains_no_identity_or_private_path():
    public = json.dumps(_exact_artifact())
    assert AUTHORIZATION not in public
    assert "allocation_artifact_path" not in public
    assert "partition_manifest_path" not in public


# ---------------------------------------------------------------------------
# V8F-PREFREEZE-HIGH-003: canonical/default fresh-public-evidence derivation
# from real committed public Git objects (not caller assertions / bare
# constants).  These exercise the real repository this session is running
# in: local Git object reads only, no network, no private data.
# ---------------------------------------------------------------------------


def _real_head() -> str:
    return subprocess.run(
        ["git", "-C", str(CANONICAL_REPOSITORY_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _real_preflight(head_sha: str):
    return {
        "repository_identity": preservation.V8F_REPOSITORY_IDENTITY,
        "branch": preservation.V8F_PRODUCTION_BRANCH,
        "head": head_sha,
        "origin_head": head_sha,
        "worktree_clean": True,
        "reviewed_v8f_design_candidate_commit": preservation.V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "reviewed_v8f_design_blob_sha": preservation.V8F_DESIGN_CANDIDATE_BLOB_SHA,
        "v8e_terminal_commit": preservation.V8F_V8E_PREDECESSOR_TERMINAL_COMMIT,
        "v8e_terminal_blob_sha": preservation.V8F_V8E_TERMINAL_RECORD_BLOB_SHA,
        "trusted_partition_blob_sha": preservation.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "partition_manifest_sha256": preservation.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "partition_implementation_commit": preservation.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "v8c_allocation_implementation_commit": preservation.EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT,
        "parent_t_spare_ticker_count": preservation.EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
        "parent_t_spare_ticker_list_sha256": preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
    }


def _real_runtime(head_sha: str):
    def reader(_root, _sha):
        return {
            "branch": preservation.V8F_PRODUCTION_BRANCH,
            "head": head_sha,
            "origin_head": head_sha,
            "worktree_clean": True,
            "commits_after_reviewed_support_sha": 0,
        }
    return reader


def test_default_fresh_evidence_is_derived_from_real_committed_git_objects():
    head_sha = _real_head()
    evidence = preservation._default_fresh_t1c_public_evidence(
        CANONICAL_REPOSITORY_ROOT,
        _real_preflight(head_sha),
        head_sha,
        runtime_state_reader=_real_runtime(head_sha),
    )
    assert evidence["fresh_public_preservation_evidence_result"] == "PASS"
    assert evidence["allocation_artifact_self_hash"] == preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH
    assert evidence["t1c_ticker_count"] == preservation.EXPECTED_V8F_T1C_TICKER_COUNT
    for field in (
        "t1c_raw_acquisition_performed",
        "t1c_research_opened",
        "t1c_ohlcv_research_access",
        "t1c_feature_access",
        "t1c_outcome_access",
        "t1c_identities_publicly_exposed",
        "t1c_membership_reassigned",
    ):
        assert evidence[field] is False


def test_default_fresh_evidence_has_no_caller_final_evidence_shortcut_parameter():
    """A caller-supplied final PASS mapping alone must never be sufficient
    for the canonical execution path: the default derivation's signature
    only accepts low-level chronology/runtime readers, never a pre-derived
    evidence mapping."""
    import inspect

    sig = inspect.signature(preservation._default_fresh_t1c_public_evidence)
    assert "evidence" not in sig.parameters
    assert "safe_evidence" not in sig.parameters
    assert "final_evidence" not in sig.parameters


@pytest.mark.parametrize(
    "tamper_path",
    [
        "V8_STATE_GIT_PATH",
        "V8C_TRUSTED_ALLOCATION_GIT_PATH",
        "V8F_V8E_TERMINAL_RECORD_GIT_PATH",
        "V8E_T1C_PRESERVATION_RECHECK_GIT_PATH",
    ],
)
def test_tampered_committed_blob_reference_blocks_before_gate(tamper_path):
    head_sha = _real_head()
    target_path = getattr(preservation, tamper_path)
    real_resolve = preservation.resolve_git_blob

    def tampered(root_, commit, path):
        if path == target_path:
            return "0" * 40
        return real_resolve(root_, commit, path)

    preservation.resolve_git_blob = tampered
    try:
        with pytest.raises(preservation.V8FT1CPreservationBlocked):
            preservation._default_fresh_t1c_public_evidence(
                CANONICAL_REPOSITORY_ROOT,
                _real_preflight(head_sha),
                head_sha,
                runtime_state_reader=_real_runtime(head_sha),
            )
    finally:
        preservation.resolve_git_blob = real_resolve


def test_empty_chronology_between_predecessor_and_candidate_blocks():
    head_sha = _real_head()
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._default_fresh_t1c_public_evidence(
            CANONICAL_REPOSITORY_ROOT,
            _real_preflight(head_sha),
            head_sha,
            runtime_state_reader=_real_runtime(head_sha),
            chronology_reader=lambda *args: [],
        )
    assert excinfo.value.reason == "V8F_PUBLIC_CHRONOLOGY_INVALID"


def test_malformed_chronology_record_blocks():
    head_sha = _real_head()
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._default_fresh_t1c_public_evidence(
            CANONICAL_REPOSITORY_ROOT,
            _real_preflight(head_sha),
            head_sha,
            runtime_state_reader=_real_runtime(head_sha),
            chronology_reader=lambda *args: [{"commit": "not-a-sha", "paths": ["x"]}],
        )


def _historical_t1c_record(**overrides):
    value = {
        "allocation_artifact_self_hash": preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "allocation_self_hash_unchanged": True,
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "parent_t_spare_ticker_list_sha256": preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "parent_v8_provenance_unchanged": True,
        "preservation_recheck_result": "PASS",
        "remaining_t_spare_ticker_list_sha256": preservation.EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "reviewed_v8e_design_candidate_commit": "6f672404b93a1003253915196dd635ca76fd2be1",
        "schema_version": "V8E_T1C_PRESERVATION_RECHECK_V1",
        "source_v8c_terminal_commit": "d18368c1ec1c26d752ea5862115ab9f4315d1780",
        "study": "V8E_HISTORICAL_RESEARCH",
        "t1c_feature_access": False,
        "t1c_identities_publicly_exposed": False,
        "t1c_membership_reassigned": False,
        "t1c_ohlcv_research_access": False,
        "t1c_outcome_access": False,
        "t1c_raw_acquisition_performed": False,
        "t1c_research_opened": False,
        "t1c_ticker_count": preservation.EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": preservation.EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
        "v8c_terminal_adjudication_authoritative": True,
    }
    value.update(overrides)
    return value


def test_historical_t1c_record_exactly_matches_real_committed_artifact():
    """This fixture matches the real, currently-committed
    V8E_T1C_PRESERVATION_RECHECK.json content exactly, and that real file
    validates successfully as V8F's historical evidence source."""
    assert set(_historical_t1c_record()) == preservation._V8E_T1C_PRESERVATION_RECHECK_FIELDS
    assert preservation._validate_historical_v8e_t1c_record(_historical_t1c_record())
    real_bytes = subprocess.run(
        ["git", "-C", str(CANONICAL_REPOSITORY_ROOT), "show",
         f"{preservation.V8E_T1C_PRESERVATION_RECHECK_COMMIT}:{preservation.V8E_T1C_PRESERVATION_RECHECK_GIT_PATH}"],
        capture_output=True, check=True,
    ).stdout
    real_record = json.loads(real_bytes)
    assert real_record == _historical_t1c_record()
    assert preservation._validate_historical_v8e_t1c_record(real_record)


def test_tampered_historical_t1c_record_field_blocks():
    tampered = _historical_t1c_record(t1c_membership_reassigned=True)
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._validate_historical_v8e_t1c_record(tampered)
    assert excinfo.value.reason == "V8F_V8E_HISTORICAL_T1C_VALUE_INVALID:t1c_membership_reassigned"


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_historical_t1c_record_schema_exactness(mutation):
    record = _historical_t1c_record()
    if mutation == "missing":
        del record["t1c_feature_access"]
    else:
        record["unexpected"] = True
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._validate_historical_v8e_t1c_record(record)
    assert excinfo.value.reason == "V8F_V8E_HISTORICAL_T1C_SCHEMA_INVALID"


def test_tampered_trusted_allocation_field_blocks():
    tampered = {
        "artifact_role": "TRUSTED_T1C_ALLOCATION_PIN",
        "authorization_note": "x",
        "authorization_status": "AUTHORIZED",
        "authorized_allocation_artifact_self_hash": "0" * 64,  # tampered
        "human_gate": "x",
        "logical_block": "T1C",
        "parent_t_spare_ticker_count": preservation.EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
        "parent_t_spare_ticker_list_sha256": preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "parent_v8_partition_implementation_commit": preservation.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "parent_v8_partition_manifest_sha256": preservation.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "predecessor_burned_count": 300,
        "remaining_t_spare_ticker_count": 1304,
        "remaining_t_spare_ticker_list_sha256": preservation.EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "schema_version": "V8C_TRUSTED_ALLOCATION_V1",
        "study_name": "V8C_HISTORICAL_RESEARCH",
        "t1c_ticker_count": preservation.EXPECTED_V8F_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": preservation.EXPECTED_V8F_T1C_TICKER_LIST_SHA256,
        "v8c_allocation_implementation_commit": preservation.EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT,
        "v8c_frozen_design_commit": preservation.EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        "v8c_reviewed_production_implementation_commit": preservation.EXPECTED_V8C_ALLOCATION_IMPLEMENTATION_COMMIT,
        "verification_result": "PASS",
    }
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._validate_current_trusted_t1c_allocation(tampered)
    assert excinfo.value.reason == "V8F_TRUSTED_ALLOCATION_VALUE_INVALID:authorized_allocation_artifact_self_hash"


def test_v8_state_t1_missing_section_blocks():
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._validate_current_v8_state_t1({"T1": {}, "last_real_t1_acquisition_attempt": {}})
    assert excinfo.value.reason.startswith("V8F_V8_STATE_T1_FIELD_MISSING")


def test_v8_state_t1_observed_access_flags_derived_correctly():
    state = {
        "T1": {
            "raw_data_acquired": False,
            "layer_b_opened": False,
            "validation_access_count": None,
            "ticker_count_frozen": preservation.EXPECTED_V8F_T1C_TICKER_COUNT,
        },
        "last_real_t1_acquisition_attempt": {
            "t1_successfully_acquired": False,
            "t1_opened_for_research": False,
            "validation_accessed": False,
        },
    }
    observed = preservation._validate_current_v8_state_t1(state)
    assert observed == {
        "raw_acquisition_performed": False,
        "research_opened": False,
        "ohlcv_research_access": False,
    }


def test_v8_state_t1_tampered_acquisition_flag_flips_observed_value():
    state = {
        "T1": {
            "raw_data_acquired": True,  # tampered
            "layer_b_opened": False,
            "validation_access_count": None,
            "ticker_count_frozen": preservation.EXPECTED_V8F_T1C_TICKER_COUNT,
        },
        "last_real_t1_acquisition_attempt": {
            "t1_successfully_acquired": False,
            "t1_opened_for_research": False,
            "validation_accessed": False,
        },
    }
    observed = preservation._validate_current_v8_state_t1(state)
    assert observed["raw_acquisition_performed"] is True


def test_pre_gate_default_evidence_failure_yields_zero_gate_and_private_reads(tmp_path):
    """Exercises the real default derivation path (public_evidence_resolver
    omitted) against this actual repository, forcing a pre-gate failure via
    a reviewed-support SHA that does not match the real runtime -- proving
    the default resolver never reaches private reads or gate consumption
    when its own binding checks fail."""
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic allocation")
    manifest.write_bytes(b"synthetic manifest")
    reads = []
    gate_calls = []
    head_sha = _real_head()
    wrong_sha = ("f" if head_sha[0] != "f" else "e") + head_sha[1:]

    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._execute_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=wrong_sha,
            state_root=tmp_path / "state",
            output_path=tmp_path / "artifact.json",
            allocation_artifact_path=allocation,
            partition_manifest_path=manifest,
            repository_root=CANONICAL_REPOSITORY_ROOT,
            public_preflight=lambda: _real_preflight(head_sha),
            runtime_state_reader=_real_runtime(head_sha),
            # public_evidence_resolver omitted -> exercises the real default
            # derivation path, which never even reaches evidence derivation
            # because the reviewed-support binding fails first.
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=lambda *args, **kwargs: gate_calls.append((args, kwargs)),
            clock=_clock,
        )
    assert reads == []
    assert gate_calls == []
    assert not (tmp_path / "state").exists()


def test_default_derivation_actually_reaches_pass_when_wired_through_execute(tmp_path):
    """Positive counterpart: with the real reviewed-support SHA and real
    default derivation (public_evidence_resolver omitted), the pipeline
    proceeds past evidence derivation into gate consumption and private
    reads -- proving the default path is genuinely wired in, not merely
    unreachable scaffolding."""
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic allocation")
    manifest.write_bytes(b"synthetic manifest")
    reads = []
    gate_calls = []
    head_sha = _real_head()

    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        # Still fails eventually (synthetic allocation bytes cannot pass the
        # real private-artifact verifier), but only *after* the real gate
        # consumption and private reads occur -- proving evidence derivation
        # itself passed using the real default resolver.
        preservation._execute_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=head_sha,
            state_root=tmp_path / "state",
            output_path=tmp_path / "artifact.json",
            allocation_artifact_path=allocation,
            partition_manifest_path=manifest,
            repository_root=CANONICAL_REPOSITORY_ROOT,
            public_preflight=lambda: _real_preflight(head_sha),
            runtime_state_reader=_real_runtime(head_sha),
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=lambda *args, **kwargs: gate_calls.append((args, kwargs)),
            clock=_clock,
        )
    assert len(reads) == 2
    assert len(gate_calls) == 1


# ---------------------------------------------------------------------------
# V8F-T1C-LOCATOR-001: post-gate content-addressed partition-manifest locator
# ---------------------------------------------------------------------------


def _synthetic_parent_tickers(count: int | None = None) -> list[str]:
    count = preservation.EXPECTED_PARENT_T_SPARE_TICKER_COUNT if count is None else count
    return [f"SYN{i:05d}" for i in range(count)]


def _synthetic_manifest_and_allocation():
    """A fully self-consistent, independently-verifiable synthetic manifest +
    allocation pair: every hash below is *computed*, never a bare literal
    standing in for real private production content."""
    parent = _synthetic_parent_tickers()
    t_spare_hash = v8_partition_module.ticker_list_sha256(parent)
    partition_commit = "a" * 40
    allocation_commit = "b" * 40
    design_commit = "c" * 40
    manifest_body = {
        "schema_version": v8_partition_module.SCHEMA_VERSION,
        "study_name": "V8_HISTORICAL_RESEARCH",
        "design_commit": "c414d3191cba356734d7ed08bdf1abc7d51fc384",
        "source_snapshot_semantics": v8_partition_module.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": v8_partition_module.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": partition_commit,
        "created_utc": "2026-01-01T00:00:00Z",
        "source_url": "https://example.invalid/synthetic",
        "source_host": "example.invalid",
        "source_acquisition_utc": "2026-01-01T00:00:00Z",
        "source_raw_sha256": "0" * 64,
        "source_raw_byte_count": 1,
        "v4_source_raw_sha256_reference": "0" * 64,
        "v4_raw_sha_equality_required": False,
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "eligible_ticker_count": len(parent),
        "eligible_ticker_list_sha256": t_spare_hash,
        "selection_rule": "synthetic",
        "deterministic_ordering_rule": "synthetic",
        "t0_ticker_list_sha256": v8_partition_module.ticker_list_sha256(["T0_A"]),
        "t1_ticker_list_sha256": v8_partition_module.ticker_list_sha256(["T1_A"]),
        "t2_ticker_list_sha256": v8_partition_module.ticker_list_sha256(["T2_A"]),
        "t3_ticker_list_sha256": v8_partition_module.ticker_list_sha256(["T3_A"]),
        "t_spare_ticker_list_sha256": t_spare_hash,
        "legacy_exclude_list": [],
        "legacy_exclude_list_sha256": v8_partition_module.ticker_list_sha256([]),
        "block_sizes": {"T0": 1, "T1": 1, "T2": 1, "T3": 1, "T_spare": len(parent)},
        "block_assignments": {
            "T0": ["T0_A"],
            "T1": ["T1_A"],
            "T2": ["T2_A"],
            "T3": ["T3_A"],
            "T_spare": parent,
        },
        "p_hist_start": "2018-01-01",
        "p_hist_end": "2025-12-31",
        "t1_role": "synthetic",
        "t2_role": "synthetic",
        "t3_role": "synthetic",
        "t3_price_acquisition_authorized": False,
    }
    manifest_sha256 = v8_partition_module.canonical_sha256(manifest_body)
    manifest = dict(manifest_body)
    manifest["manifest_sha256"] = manifest_sha256
    assert set(manifest) == set(v8_partition_module.MANIFEST_FIELDS)

    allocation = build_t1c_allocation_artifact(
        parent,
        parent_v8_partition_manifest_sha256=manifest_sha256,
        parent_v8_partition_implementation_commit=partition_commit,
        parent_t_spare_ticker_list_sha256=t_spare_hash,
        v8c_frozen_design_commit=design_commit,
        v8c_allocation_implementation_commit=allocation_commit,
        clock=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    meta = {
        "manifest_sha256": manifest_sha256,
        "partition_commit": partition_commit,
        "allocation_commit": allocation_commit,
        "design_commit": design_commit,
        "t_spare_hash": t_spare_hash,
    }
    return manifest, allocation, meta


def _retarget_manifest(manifest, **field_overrides):
    """A self-consistent but *differently-hashed* variant of a manifest."""
    mutated = dict(manifest)
    mutated.update(field_overrides)
    body = {k: v for k, v in mutated.items() if k != "manifest_sha256"}
    mutated["manifest_sha256"] = v8_partition_module.canonical_sha256(body)
    return mutated


# --- validate_candidate_partition_manifest_paths: pre-gate, metadata-only ---


def test_candidate_list_empty_blocks():
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.validate_candidate_partition_manifest_paths([], repository_root=Path("/repo"))
    assert excinfo.value.reason == "V8F_LOCATOR_CANDIDATE_LIST_EMPTY"


def test_candidate_list_bare_string_rejected():
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.validate_candidate_partition_manifest_paths(
            "/outside/a/partition_manifest.json", repository_root=Path("/repo")
        )
    assert excinfo.value.reason == "V8F_LOCATOR_CANDIDATE_LIST_INVALID"


def test_candidate_basename_must_be_exact():
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.validate_candidate_partition_manifest_paths(
            ["/outside/a/other_name.json"], repository_root=Path("/repo")
        )
    assert excinfo.value.reason == "V8F_LOCATOR_CANDIDATE_BASENAME_INVALID"


def test_candidate_inside_repo_blocks():
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.validate_candidate_partition_manifest_paths(
            ["/repo/nested/partition_manifest.json"], repository_root=Path("/repo")
        )
    assert excinfo.value.reason == "V8F_LOCATOR_CANDIDATE_PATH_INVALID"


def test_candidate_relative_path_blocks():
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.validate_candidate_partition_manifest_paths(
            ["relative/partition_manifest.json"], repository_root=Path("/repo")
        )
    assert excinfo.value.reason == "V8F_LOCATOR_CANDIDATE_PATH_INVALID"


def test_candidate_duplicate_after_normalization_blocks():
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation.validate_candidate_partition_manifest_paths(
            [
                "/outside/a/./partition_manifest.json",
                "/outside/a/partition_manifest.json",
            ],
            repository_root=Path("/repo"),
        )
    assert excinfo.value.reason == "V8F_LOCATOR_CANDIDATE_DUPLICATE_PATH"


def test_candidate_list_accepts_valid_unique_paths():
    result = preservation.validate_candidate_partition_manifest_paths(
        ["/outside/a/partition_manifest.json", "/outside/b/partition_manifest.json"],
        repository_root=Path("/repo"),
    )
    assert result == (
        Path("/outside/a/partition_manifest.json"),
        Path("/outside/b/partition_manifest.json"),
    )


# --- _locate_authorized_partition_manifest: post-gate content matching ---


def test_locator_selects_the_one_exact_match_among_several():
    manifest, _allocation, meta = _synthetic_manifest_and_allocation()
    manifest_raw = json.dumps(manifest).encode("utf-8")
    other_raw = json.dumps(_retarget_manifest(manifest, source_url="https://example.invalid/other")).encode(
        "utf-8"
    )
    reads = {
        Path("/outside/a/partition_manifest.json"): other_raw,
        Path("/outside/b/partition_manifest.json"): manifest_raw,
        Path("/outside/c/partition_manifest.json"): b"not json at all",
    }
    matched_raw, stats = preservation._locate_authorized_partition_manifest(
        lambda path: reads[path],
        tuple(reads.keys()),
        expected_partition_manifest_sha256=meta["manifest_sha256"],
        expected_partition_implementation_commit=meta["partition_commit"],
    )
    assert matched_raw == manifest_raw
    assert stats == {"candidate_count": 3, "candidates_read_count": 3, "exact_match_count": 1}


def test_locator_zero_matches_blocks():
    manifest, _allocation, meta = _synthetic_manifest_and_allocation()
    non_matching = json.dumps(
        _retarget_manifest(manifest, source_url="https://example.invalid/nonmatching")
    ).encode("utf-8")
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._locate_authorized_partition_manifest(
            lambda path: non_matching,
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=meta["manifest_sha256"],
            expected_partition_implementation_commit=meta["partition_commit"],
        )
    assert excinfo.value.reason == "V8F_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_locator_multiple_exact_copies_blocks():
    manifest, _allocation, meta = _synthetic_manifest_and_allocation()
    raw = json.dumps(manifest).encode("utf-8")
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._locate_authorized_partition_manifest(
            lambda path: raw,
            (Path("/outside/a/partition_manifest.json"), Path("/outside/b/partition_manifest.json")),
            expected_partition_manifest_sha256=meta["manifest_sha256"],
            expected_partition_implementation_commit=meta["partition_commit"],
        )
    assert excinfo.value.reason == "V8F_LOCATOR_MULTIPLE_MATCHING_CANDIDATES"


def test_locator_rejects_self_declared_hash_without_recomputation():
    manifest, _allocation, meta = _synthetic_manifest_and_allocation()
    tampered = dict(manifest)
    # Content changed, but the (now stale/incorrect) manifest_sha256 field is
    # left exactly equal to the expected hash -- a bare self-declaration.
    tampered["source_url"] = "https://example.invalid/tampered-but-claims-original-hash"
    raw = json.dumps(tampered).encode("utf-8")
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._locate_authorized_partition_manifest(
            lambda path: raw,
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=meta["manifest_sha256"],
            expected_partition_implementation_commit=meta["partition_commit"],
        )
    assert excinfo.value.reason == "V8F_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_locator_rejects_wrong_implementation_commit_despite_hash_match():
    manifest, _allocation, meta = _synthetic_manifest_and_allocation()
    raw = json.dumps(manifest).encode("utf-8")
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._locate_authorized_partition_manifest(
            lambda path: raw,
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=meta["manifest_sha256"],
            expected_partition_implementation_commit="d" * 40,
        )
    assert excinfo.value.reason == "V8F_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_locator_malformed_candidate_cannot_match():
    manifest, _allocation, meta = _synthetic_manifest_and_allocation()
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._locate_authorized_partition_manifest(
            lambda path: b"{not valid json",
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=meta["manifest_sha256"],
            expected_partition_implementation_commit=meta["partition_commit"],
        )
    assert excinfo.value.reason == "V8F_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_locator_unreadable_candidate_cannot_match():
    manifest, _allocation, meta = _synthetic_manifest_and_allocation()

    def private_reader(path):
        raise OSError("synthetic: candidate does not exist")

    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._locate_authorized_partition_manifest(
            private_reader,
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=meta["manifest_sha256"],
            expected_partition_implementation_commit=meta["partition_commit"],
        )
    assert excinfo.value.reason == "V8F_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_locator_error_never_leaks_candidate_path():
    manifest, _allocation, meta = _synthetic_manifest_and_allocation()
    non_matching = json.dumps(
        _retarget_manifest(manifest, source_url="https://example.invalid/nonmatching")
    ).encode("utf-8")
    secret_path = Path("/outside/SECRET_TICKER_MARKER/partition_manifest.json")
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._locate_authorized_partition_manifest(
            lambda path: non_matching,
            (secret_path,),
            expected_partition_manifest_sha256=meta["manifest_sha256"],
            expected_partition_implementation_commit=meta["partition_commit"],
        )
    assert "SECRET_TICKER_MARKER" not in str(excinfo.value)
    assert "SECRET_TICKER_MARKER" not in excinfo.value.reason


# --- Full existing verification chain, exercised end-to-end with synthetic,
# fully self-consistent data (never the frozen real production hash) ---


def test_locator_selected_candidate_passes_full_existing_verification():
    manifest, allocation, meta = _synthetic_manifest_and_allocation()
    allocation_raw = json.dumps(allocation).encode("utf-8")
    manifest_raw = json.dumps(manifest).encode("utf-8")

    matched_raw, stats = preservation._locate_authorized_partition_manifest(
        lambda path: manifest_raw,
        (Path("/outside/a/partition_manifest.json"),),
        expected_partition_manifest_sha256=meta["manifest_sha256"],
        expected_partition_implementation_commit=meta["partition_commit"],
    )
    assert stats["exact_match_count"] == 1

    summary = preservation._verify_private_artifacts(
        allocation_raw,
        matched_raw,
        expected_allocation_artifact_self_hash=allocation["artifact_self_hash"],
        expected_parent_t_spare_ticker_list_sha256=meta["t_spare_hash"],
        expected_t1c_ticker_list_sha256=allocation["t1c_ticker_list_sha256"],
        expected_remaining_t_spare_ticker_list_sha256=allocation["remaining_t_spare_ticker_list_sha256"],
        expected_partition_manifest_sha256=meta["manifest_sha256"],
        expected_partition_implementation_commit=meta["partition_commit"],
        expected_v8c_allocation_implementation_commit=meta["allocation_commit"],
        expected_v8c_frozen_design_commit=meta["design_commit"],
    )
    assert summary["allocation_artifact_self_hash"] == allocation["artifact_self_hash"]
    assert summary["t1c_membership_reassigned"] is False
    assert summary["allocation_self_hash_unchanged"] is True
    assert summary["parent_v8_provenance_unchanged"] is True


def test_locator_full_verification_rejects_wrong_parent_t_spare_binding():
    manifest, allocation, meta = _synthetic_manifest_and_allocation()
    allocation_raw = json.dumps(allocation).encode("utf-8")
    manifest_raw = json.dumps(manifest).encode("utf-8")
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._verify_private_artifacts(
            allocation_raw,
            manifest_raw,
            expected_allocation_artifact_self_hash=allocation["artifact_self_hash"],
            expected_parent_t_spare_ticker_list_sha256="0" * 64,
            expected_t1c_ticker_list_sha256=allocation["t1c_ticker_list_sha256"],
            expected_remaining_t_spare_ticker_list_sha256=allocation["remaining_t_spare_ticker_list_sha256"],
            expected_partition_manifest_sha256=meta["manifest_sha256"],
            expected_partition_implementation_commit=meta["partition_commit"],
            expected_v8c_allocation_implementation_commit=meta["allocation_commit"],
            expected_v8c_frozen_design_commit=meta["design_commit"],
        )


# --- Full DI execution boundary: gate/private-read ordering, pre/post-gate
# classification, one-shot no-retry semantics ---


def test_locator_pre_gate_failure_has_zero_gate_and_private_reads(tmp_path):
    allocation_path = tmp_path / "allocation.bin"
    allocation_path.write_bytes(b"synthetic allocation")
    reads = []
    gate_calls = []
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._execute_locator_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
            state_root=tmp_path / "state",
            allocation_artifact_path=allocation_path,
            candidate_partition_manifest_paths=[],
            repository_root=tmp_path / "repo",
            public_preflight=_preflight,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(),
            public_evidence_resolver=lambda _: _fresh_public_evidence(),
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=lambda *args, **kwargs: gate_calls.append((args, kwargs)),
            clock=_clock,
        )
    assert excinfo.value.reason == "V8F_LOCATOR_CANDIDATE_LIST_EMPTY"
    assert reads == []
    assert gate_calls == []
    assert not (tmp_path / "state").exists()


def test_locator_duplicate_candidate_paths_pre_gate_block(tmp_path):
    allocation_path = tmp_path / "allocation.bin"
    allocation_path.write_bytes(b"synthetic allocation")
    candidate = tmp_path.parent / "outside-candidates" / "partition_manifest.json"
    reads = []
    gate_calls = []
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._execute_locator_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
            state_root=tmp_path / "state",
            allocation_artifact_path=allocation_path,
            candidate_partition_manifest_paths=[candidate, candidate],
            repository_root=tmp_path / "repo",
            public_preflight=_preflight,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(),
            public_evidence_resolver=lambda _: _fresh_public_evidence(),
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=lambda *args, **kwargs: gate_calls.append((args, kwargs)),
            clock=_clock,
        )
    assert excinfo.value.reason == "V8F_LOCATOR_CANDIDATE_DUPLICATE_PATH"
    assert reads == []
    assert gate_calls == []


def test_locator_repo_internal_candidate_pre_gate_block(tmp_path):
    allocation_path = tmp_path / "allocation.bin"
    allocation_path.write_bytes(b"synthetic allocation")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    candidate = repo_root / "nested" / "partition_manifest.json"
    reads = []
    gate_calls = []
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._execute_locator_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
            state_root=tmp_path / "state",
            allocation_artifact_path=allocation_path,
            candidate_partition_manifest_paths=[candidate],
            repository_root=repo_root,
            public_preflight=_preflight,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(),
            public_evidence_resolver=lambda _: _fresh_public_evidence(),
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=lambda *args, **kwargs: gate_calls.append((args, kwargs)),
            clock=_clock,
        )
    assert excinfo.value.reason == "V8F_LOCATOR_CANDIDATE_PATH_INVALID"
    assert reads == []
    assert gate_calls == []


def test_locator_reviewed_support_failure_precedes_gate_and_private_read(tmp_path):
    allocation_path = tmp_path / "allocation.bin"
    allocation_path.write_bytes(b"synthetic allocation")
    candidate = tmp_path.parent / "outside-candidates-2" / "partition_manifest.json"
    reads = []
    gate_calls = []
    with pytest.raises(preservation.V8FT1CPreservationBlocked):
        preservation._execute_locator_with_dependencies(
            authorization_identity=AUTHORIZATION,
            reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
            state_root=tmp_path / "state",
            allocation_artifact_path=allocation_path,
            candidate_partition_manifest_paths=[candidate],
            repository_root=tmp_path / "repo",
            public_preflight=_preflight,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(
                head="2" * 40, commits_after_reviewed_support_sha=1
            ),
            public_evidence_resolver=lambda _: _fresh_public_evidence(),
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=lambda *args, **kwargs: gate_calls.append((args, kwargs)),
            clock=_clock,
        )
    assert reads == []
    assert gate_calls == []
    assert not (tmp_path / "state").exists()


def _locator_ordering_scenario(tmp_path, monkeypatch):
    """`validate_authorization_identity` always re-checks its allocation-hash
    argument against the real frozen `AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH`
    constant, so no fabricated authorization can ever reach the gate with a
    synthetic self-hash -- confirmed by `test_locator_pre_gate_...` style
    failures if attempted.  These plumbing/ordering tests therefore use the
    real `AUTHORIZATION` and monkeypatch only `read_t1c_allocation_artifact_bytes`
    to bypass the *cryptographic preimage* requirement (which cannot be
    fabricated and is already exercised directly and exhaustively via
    `_verify_private_artifacts` elsewhere in this file), so the call ordering
    itself -- gate before any private read, allocation before candidates --
    can be observed all the way through the candidate scan."""
    monkeypatch.setattr(
        preservation,
        "read_t1c_allocation_artifact_bytes",
        lambda raw: {
            "artifact_self_hash": preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
            "parent_v8_partition_manifest_sha256": preservation.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
            "parent_v8_partition_implementation_commit": preservation.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        },
    )
    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_bytes(b"synthetic allocation bytes")
    candidate_dir = tmp_path / "outside-candidates" / "a"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "partition_manifest.json"
    candidate.write_bytes(b"synthetic candidate bytes")
    state_root = tmp_path / "state"
    kwargs = dict(
        authorization_identity=AUTHORIZATION,
        reviewed_support_implementation_sha=REVIEWED_SUPPORT_SHA,
        state_root=state_root,
        allocation_artifact_path=allocation_path,
        candidate_partition_manifest_paths=[candidate],
        repository_root=tmp_path / "repo",
        public_preflight=_preflight,
        runtime_state_reader=lambda _root, _sha: _runtime_support_state(),
        public_evidence_resolver=lambda _: _fresh_public_evidence(),
        gate_consumer=preservation.consume_gate_once,
        clock=_clock,
    )
    return kwargs, state_root


def test_locator_candidate_and_allocation_bytes_only_read_after_gate(tmp_path, monkeypatch):
    kwargs, state_root = _locator_ordering_scenario(tmp_path, monkeypatch)
    receipt_seen_before_read = []

    def private_reader(path):
        receipt_seen_before_read.append(any(state_root.glob("*.json")))
        return path.read_bytes()

    kwargs["private_reader"] = private_reader
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._execute_locator_with_dependencies(**kwargs)
    # The only reachable post-gate BLOCK here is "no exact content match" --
    # there is no way to fabricate the frozen real partition-manifest hash --
    # but reaching it proves both the allocation read and the candidate scan
    # read actually happened, and only after gate consumption.
    assert excinfo.value.reason == "V8F_LOCATOR_ZERO_MATCHING_CANDIDATES"
    assert receipt_seen_before_read == [True, True]
    assert list(state_root.glob("*.json"))


def test_locator_post_gate_failure_receipt_persists_no_retry(tmp_path, monkeypatch):
    kwargs, state_root = _locator_ordering_scenario(tmp_path, monkeypatch)
    kwargs["private_reader"] = lambda path: path.read_bytes()
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo:
        preservation._execute_locator_with_dependencies(**kwargs)
    assert excinfo.value.reason == "V8F_LOCATOR_ZERO_MATCHING_CANDIDATES"

    key = preservation.compute_receipt_key(AUTHORIZATION)
    receipt = preservation.read_gate_receipt(state_root, key)
    assert receipt["consumed"] is True
    assert receipt["consumption_count"] == 1

    # Same authorization again: pre-gate checks pass again, but the gate is
    # already consumed -- no retry, no receipt reset, no second scan.
    with pytest.raises(preservation.V8FT1CPreservationBlocked) as excinfo_retry:
        preservation._execute_locator_with_dependencies(**kwargs)
    assert excinfo_retry.value.reason == "V8F_GATE_ALREADY_CONSUMED"


def test_locator_no_ticker_or_path_in_safe_result_fields():
    manifest, allocation, meta = _synthetic_manifest_and_allocation()
    manifest_raw = json.dumps(manifest).encode("utf-8")
    allocation_raw = json.dumps(allocation).encode("utf-8")
    matched_raw, stats = preservation._locate_authorized_partition_manifest(
        lambda path: manifest_raw,
        (Path("/outside/a/partition_manifest.json"),),
        expected_partition_manifest_sha256=meta["manifest_sha256"],
        expected_partition_implementation_commit=meta["partition_commit"],
    )
    summary = preservation._verify_private_artifacts(
        allocation_raw,
        matched_raw,
        expected_allocation_artifact_self_hash=allocation["artifact_self_hash"],
        expected_parent_t_spare_ticker_list_sha256=meta["t_spare_hash"],
        expected_t1c_ticker_list_sha256=allocation["t1c_ticker_list_sha256"],
        expected_remaining_t_spare_ticker_list_sha256=allocation["remaining_t_spare_ticker_list_sha256"],
        expected_partition_manifest_sha256=meta["manifest_sha256"],
        expected_partition_implementation_commit=meta["partition_commit"],
        expected_v8c_allocation_implementation_commit=meta["allocation_commit"],
        expected_v8c_frozen_design_commit=meta["design_commit"],
    )
    safe_fields = set(stats) | set(summary)
    assert "t1c_tickers" not in safe_fields
    assert "remaining_t_spare_tickers" not in safe_fields
    for ticker in allocation["t1c_tickers"]:
        assert ticker not in json.dumps(stats)
        assert ticker not in json.dumps(summary)


def test_partition_manifest_basename_constant_is_exact():
    assert preservation.PARTITION_MANIFEST_BASENAME == "partition_manifest.json"


def test_no_network_and_no_real_private_read_locator_seam():
    """Re-runs the existing module-wide AST scan; the locator seam adds no
    forbidden network/socket names."""
    import ast
    import inspect

    source = inspect.getsource(preservation)
    tree = ast.parse(source)
    forbidden = {"urlopen", "socket", "requests", "httpx"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (forbidden & names)
    assert not (forbidden & attrs)
