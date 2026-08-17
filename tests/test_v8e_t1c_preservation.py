from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8e_t1c_preservation as preservation


AUTHORIZATION = (
    preservation.V8E_AUTHORIZATION_PREFIX
    + preservation.V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT
    + preservation.V8E_AUTHORIZATION_SEPARATOR
    + preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH
)
REVIEWED_SUPPORT_SHA = "1" * 40


def _clock():
    return datetime(2026, 8, 15, tzinfo=timezone.utc)


def _preflight(**overrides):
    value = {
        "repository_identity": preservation.V8E_REPOSITORY_IDENTITY,
        "branch": preservation.V8E_PRODUCTION_BRANCH,
        "head": "1" * 40,
        "origin_head": "1" * 40,
        "worktree_clean": True,
        "reviewed_v8e_design_candidate_commit": preservation.V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "reviewed_v8e_design_blob_sha": preservation.V8E_DESIGN_CANDIDATE_BLOB_SHA,
        "v8c_terminal_commit": preservation.V8C_TERMINAL_COMMIT,
        "v8c_terminal_blob_sha": preservation.V8C_TERMINAL_ADJUDICATION_BLOB_SHA,
        "v8c_prefreeze_blob_sha": preservation.V8C_PREFREEZE_AUDIT_BLOB_SHA,
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
        "branch": preservation.V8E_PRODUCTION_BRANCH,
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
        "schema_version": "V8E_T1C_PRESERVATION_RECHECK_V1",
        "artifact_role": "T1C_PRESERVATION_RECHECK",
        "study": preservation.V8E_STUDY_NAME,
        "reviewed_v8e_design_candidate_commit": preservation.V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "source_v8c_terminal_commit": preservation.V8C_TERMINAL_COMMIT,
        "allocation_artifact_self_hash": preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": preservation.EXPECTED_V8E_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": preservation.EXPECTED_V8E_T1C_TICKER_LIST_SHA256,
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
        "v8c_terminal_adjudication_authoritative": True,
        "preservation_recheck_result": "PASS",
    }


def _fresh_public_evidence(**overrides):
    value = {
        "schema_version": "V8E_T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE_V1",
        "evidence_role": "T1C_FRESH_PUBLIC_PRESERVATION_EVIDENCE",
        "study": preservation.V8E_STUDY_NAME,
        "reviewed_v8e_design_candidate_commit": preservation.V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "v8d_predecessor_terminal_commit": preservation.V8E_V8D_PREDECESSOR_TERMINAL_COMMIT,
        "v8d_terminal_status": "BLOCK_CLOSED",
        "v8d_terminal_failure_class": "DESIGN_AUDITABILITY_FAILURE",
        "v8d_terminal_implementation_head": "a862efec34dcf4a89005c88b55b35c39be12b7bc",
        "v8d_historical_t1c_artifact_blob_sha": preservation.V8E_V8D_HISTORICAL_T1C_BLOB_SHA,
        "allocation_artifact_self_hash": preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": preservation.EXPECTED_V8E_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": preservation.EXPECTED_V8E_T1C_TICKER_LIST_SHA256,
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
        "v8c_terminal_adjudication_authoritative": True,
        "fresh_public_preservation_evidence_result": "PASS",
    }
    value.update(overrides)
    return value


def _private_summary():
    return {
        "allocation_artifact_self_hash": preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        "t1c_ticker_count": preservation.EXPECTED_V8E_T1C_TICKER_COUNT,
        "t1c_ticker_list_sha256": preservation.EXPECTED_V8E_T1C_TICKER_LIST_SHA256,
        "parent_t_spare_ticker_list_sha256": preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "remaining_t_spare_ticker_list_sha256": preservation.EXPECTED_REMAINING_T_SPARE_TICKER_LIST_SHA256,
        "t1c_membership_reassigned": False,
        "allocation_self_hash_unchanged": True,
        "parent_v8_provenance_unchanged": True,
    }


def test_exact_candidate_and_authorization_grammar():
    preservation.validate_authorization_identity(AUTHORIZATION)
    assert preservation.authorization_identity_sha256(AUTHORIZATION) == hashlib.sha256(
        AUTHORIZATION.encode("utf-8")
    ).hexdigest()
    with pytest.raises(preservation.V8ET1CPreservationBlocked) as excinfo:
        preservation.validate_authorization_identity(AUTHORIZATION.replace("6f6724", "7f6724", 1))
    assert excinfo.value.reason == "V8E_AUTHORIZATION_GRAMMAR_MISMATCH"
    assert AUTHORIZATION not in str(excinfo.value)


def test_receipt_key_is_exact_and_deterministic():
    identity_hash = hashlib.sha256(AUTHORIZATION.encode("utf-8")).hexdigest()
    material = "|".join(
        (
            preservation.V8E_REPOSITORY_IDENTITY,
            preservation.V8E_T1C_PRESERVATION_GATE,
            preservation.V8E_REVIEWED_DESIGN_CANDIDATE_COMMIT,
            identity_hash,
            preservation.AUTHORIZED_ALLOCATION_ARTIFACT_SELF_HASH,
        )
    )
    expected = hashlib.sha256(material.encode("utf-8")).hexdigest()
    assert preservation.compute_receipt_key(AUTHORIZATION) == expected
    assert preservation.compute_receipt_key(AUTHORIZATION) == expected


def test_receipt_has_exact_fields_and_never_persists_raw_identity(tmp_path):
    receipt = _consume(tmp_path)
    key, raw = _receipt_raw(tmp_path)
    assert set(receipt) == set(preservation.V8E_RECEIPT_FIELDS)
    assert set(preservation.read_gate_receipt(tmp_path, key)) == set(preservation.V8E_RECEIPT_FIELDS)
    assert AUTHORIZATION.encode("utf-8") not in raw
    assert preservation.V8E_RECEIPT_FIELDS == tuple(receipt)


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
    with pytest.raises(preservation.V8ET1CPreservationBlocked) as excinfo:
        preservation.read_gate_receipt(tmp_path, key)
    assert excinfo.value.reason == "V8E_RECEIPT_SCHEMA_INVALID"


def test_duplicate_receipt_field_blocks(tmp_path):
    _consume(tmp_path)
    key, raw = _receipt_raw(tmp_path)
    duplicate = raw.replace(b'"consumed":true', b'"consumed":true,"consumed":true', 1)
    (tmp_path / f"{key}.json").write_bytes(duplicate)
    with pytest.raises(preservation.V8ET1CPreservationBlocked) as excinfo:
        preservation.read_gate_receipt(tmp_path, key)
    assert excinfo.value.reason == "V8E_RECEIPT_DUPLICATE_KEY"


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-15T00:00:00Z",
        "2026-08-15T00:00:00.123456Z",
        "2024-02-29T00:00:00Z",
    ],
)
def test_timestamp_valid_forms_are_accepted(tmp_path, timestamp):
    _consume(tmp_path)
    key, _ = _receipt_raw(tmp_path)
    path = tmp_path / f"{key}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["consumption_timestamp_utc"] = timestamp
    path.write_text(json.dumps(value), encoding="utf-8")
    assert preservation.read_gate_receipt(tmp_path, key)["consumption_timestamp_utc"] == timestamp


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-08-15 00:00:00Z",
        "2026-08-15T00:00:00.12345Z",
        "2026-08-15T00:00:00.1234567Z",
        "2026-08-15T00:00:00+00:00",
        "2026-02-29T00:00:00.000000Z",
        "2026-08-15T00:00:00z",
    ],
)
def test_timestamp_invalid_forms_block(tmp_path, timestamp):
    _consume(tmp_path)
    key, _ = _receipt_raw(tmp_path)
    path = tmp_path / f"{key}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["consumption_timestamp_utc"] = timestamp
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(preservation.V8ET1CPreservationBlocked) as excinfo:
        preservation.read_gate_receipt(tmp_path, key)
    assert excinfo.value.reason == "V8E_RECEIPT_TIMESTAMP_INVALID"


def test_one_shot_no_overwrite(tmp_path):
    _consume(tmp_path)
    with pytest.raises(preservation.V8ET1CPreservationBlocked) as excinfo:
        _consume(tmp_path)
    assert excinfo.value.reason == "V8E_GATE_ALREADY_CONSUMED"


def test_preflight_failure_has_zero_private_reads_and_zero_receipt(tmp_path):
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic")
    manifest.write_bytes(b"synthetic")
    reads = []
    bad = _preflight(reviewed_v8e_design_blob_sha="0" * 40)
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
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
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
        preservation._validate_fresh_t1c_public_evidence(evidence)


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
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
        preservation._validate_reviewed_support_implementation_binding(
            Path("synthetic-repository"),
            _preflight(),
            reviewed_sha,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(**runtime_overrides),
        )


def test_allowlisted_source_filename_cannot_override_later_commit_failure():
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
        preservation._validate_reviewed_support_implementation_binding(
            Path("synthetic-repository"),
            _preflight(),
            REVIEWED_SUPPORT_SHA,
            runtime_state_reader=lambda _root, _sha: _runtime_support_state(
                head="2" * 40, commits_after_reviewed_support_sha=1
            ),
        )


def test_historical_v8d_pass_is_not_v8e_fresh_authority():
    historical_only = {
        "schema_version": "V8D_T1C_PRESERVATION_RECHECK_V1",
        "preservation_recheck_result": "PASS",
    }
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
        preservation._validate_fresh_t1c_public_evidence(historical_only)


@pytest.mark.parametrize(
    "chronology",
    [
        [{"commit": "1" * 40, "paths": ["src/v8e_t1c_preservation.py", "src/v8e_t1c_preservation.py"]}],
        [{"commit": "not-a-sha", "paths": ["src/v8e_t1c_preservation.py"]}],
    ],
)
def test_malformed_chronology_blocks(chronology):
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
        preservation._validate_public_chronology(chronology)


def test_historical_chronology_does_not_use_filename_allowlist():
    assert preservation._validate_public_chronology(
        [{"commit": "1" * 40, "paths": ["V8_STATE.json", "future-preservation-relevant-file"]}]
    )[0]["paths"] == ["V8_STATE.json", "future-preservation-relevant-file"]


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_fresh_public_evidence_exact_schema(mutation):
    evidence = _fresh_public_evidence()
    if mutation == "missing":
        del evidence["t1c_outcome_access"]
    else:
        evidence["unexpected"] = False
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
        preservation._validate_fresh_t1c_public_evidence(evidence)


@pytest.mark.parametrize(
    "field,value",
    [
        ("v8d_predecessor_terminal_commit", "0" * 40),
        ("v8d_terminal_status", "PASS"),
        ("reviewed_v8e_design_candidate_commit", "0" * 40),
        ("allocation_artifact_self_hash", "0" * 64),
        ("parent_v8_provenance_unchanged", False),
    ],
)
def test_fresh_public_evidence_bindings_fail_closed(field, value):
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
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


def test_public_fresh_evidence_failure_precedes_gate_and_private_read(tmp_path):
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic allocation")
    manifest.write_bytes(b"synthetic manifest")
    reads = []
    gate_calls = []
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
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
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
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
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
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
    assert len(reads) == 2
    assert list((tmp_path / "state").glob("*.json"))


def test_consumed_receipt_is_retained_after_post_consumption_failure(tmp_path):
    allocation = tmp_path / "allocation.bin"
    manifest = tmp_path / "manifest.bin"
    allocation.write_bytes(b"synthetic allocation")
    manifest.write_bytes(b"synthetic manifest")
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
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
        artifact["reviewed_v8e_design_candidate_commit"] = "0" * 40
    else:
        artifact["allocation_artifact_self_hash"] = "0" * 64
    with pytest.raises(preservation.V8ET1CPreservationBlocked):
        preservation.verify_t1c_preservation_artifact_bytes(
            json.dumps(artifact).encode("utf-8"), receipt_raw, receipt_key=key
        )


def test_public_artifact_contains_no_identity_or_private_path():
    public = json.dumps(_exact_artifact())
    assert AUTHORIZATION not in public
    assert "allocation_artifact_path" not in public
    assert "partition_manifest_path" not in public
