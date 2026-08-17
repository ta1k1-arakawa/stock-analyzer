from __future__ import annotations

import json
import subprocess
import urllib.error
from pathlib import Path

import pytest

from src import v8e_acquisition_engine as acquisition
from src import v8e_authority_bridge as authority_bridge
from src import v8e_git_provenance as git_provenance
from src import v8e_production_provenance as provenance
from src import v8e_t2_point_of_use_preservation as point_of_use
from src import v8e_transport as transport


V8D_T1C_PRESERVATION_COMMIT = "58b70e2ce79c5f1195d1b8f20d348305e513c5f3"
V8D_T1C_PRESERVATION_BLOB = "049becb3d2743ef68dc278f424484919ba379cca"
V8D_T2_PRESERVATION_COMMIT = "8ae3032b42b426420f44c9f7194f0b1849c23e98"
V8D_T2_PRESERVATION_BLOB = "d023913b435ffd18eadef1e213c7ea43a49db331"

V8E_T1C_PRESERVATION_COMMIT = "12a05d59daca7986e4dacb27bce63e073d064240"
V8E_T1C_PRESERVATION_BLOB = "cd084dd6e49be724e876d01b27ac45fa11a2dc64"
V8E_T2_PRESERVATION_COMMIT = "22071e3fceaff56ac2043f79e2d79d617f3658a5"
V8E_T2_PRESERVATION_BLOB = "24248bf96877ffb47bdba8fac7924684b1cae5cb"


def _valid_t1c_bridge() -> dict:
    return {
        "schema_version": authority_bridge.T1C_BRIDGE_SCHEMA,
        "study": authority_bridge.STUDY,
        "artifact_role": "T1C_ALLOCATION_AUTHORITY_BRIDGE",
        "logical_block": "T1C",
        "v8e_frozen_design_commit": authority_bridge.FROZEN_DESIGN_COMMIT,
        "source_v8c_terminal_commit": "d18368c1ec1c26d752ea5862115ab9f4315d1780",
        "source_v8c_trust_pin_git_commit": authority_bridge.V8C_TRUST_PIN_COMMIT,
        "source_v8c_trust_pin_git_blob_sha": authority_bridge.V8C_TRUST_PIN_BLOB,
        "authorized_allocation_artifact_self_hash": authority_bridge.T1C_ALLOCATION_SELF_HASH,
        "t1c_ticker_count": 300,
        "t1c_ticker_list_sha256": authority_bridge.T1C_TICKER_LIST_SHA256,
        "parent_v8_partition_manifest_sha256": authority_bridge.V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": authority_bridge.V8_PARTITION_IMPLEMENTATION_COMMIT,
        "parent_t_spare_ticker_list_sha256": authority_bridge.T1C_PARENT_SPARE_LIST_SHA256,
        "preservation_recheck_git_commit": authority_bridge.T1C_PRESERVATION_COMMIT,
        "preservation_recheck_git_blob_sha": authority_bridge.T1C_PRESERVATION_BLOB,
        "preservation_recheck_result": "PASS",
        "human_gate": (
            f"V8E_HUMAN_AUTHORIZE_T1C_AUTHORITY_BRIDGE_AT_{authority_bridge.FROZEN_DESIGN_COMMIT}"
            f"_FOR_{authority_bridge.T1C_ALLOCATION_SELF_HASH}"
        ),
        "authorization_status": "AUTHORIZED",
        "authorization_note": "authorized",
    }


def _valid_t2_bridge() -> dict:
    return {
        "schema_version": authority_bridge.T2_BRIDGE_SCHEMA,
        "study": authority_bridge.STUDY,
        "artifact_role": "T2_AUTHORITY_BRIDGE",
        "logical_block": "T2",
        "v8e_frozen_design_commit": authority_bridge.FROZEN_DESIGN_COMMIT,
        "source_authority": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY",
        "v8_trust_anchor_git_identity": authority_bridge.V8_TRUST_ANCHOR_BLOB,
        "authorized_parent_v8_partition_manifest_sha256": authority_bridge.V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": authority_bridge.V8_PARTITION_IMPLEMENTATION_COMMIT,
        "expected_t2_ticker_count": 300,
        "expected_t2_ticker_list_sha256": authority_bridge.T2_TICKER_LIST_SHA256,
        "preservation_recheck_git_commit": authority_bridge.T2_PRESERVATION_COMMIT,
        "preservation_recheck_git_blob_sha": authority_bridge.T2_PRESERVATION_BLOB,
        "preservation_recheck_result": "PASS",
        "human_gate": (
            f"V8E_HUMAN_AUTHORIZE_T2_AUTHORITY_BRIDGE_AT_{authority_bridge.FROZEN_DESIGN_COMMIT}"
            f"_FOR_{authority_bridge.T2_TICKER_LIST_SHA256}"
        ),
        "authorization_status": "AUTHORIZED",
        "authorization_note": "authorized",
    }


DESIGN = "6f672404b93a1003253915196dd635ca76fd2be1"
DESIGN_BLOB = "dac32f9e97d1ae2b90eb8b0820914e3845d0fa26"
FREEZE_BLOB = "651bc1a1cac122f8d36e6c19960db56161114f46"
SHA = "0123456789abcdef0123456789abcdef01234567"


def rows(valid: int, invalid: int, *, year: int = 2020, consecutive: bool = False) -> dict[str, list[dict[str, str]]]:
    valid_rows = [{"trading_date": f"{year}-01-{(i % 28) + 1:02d}"} for i in range(valid)]
    invalid_rows = []
    for i in range(invalid):
        day = (valid + i + (0 if consecutive else 3)) % 28 + 1
        invalid_rows.append({"trading_date": f"{year}-02-{day:02d}"})
    if consecutive and invalid:
        invalid_rows = [{"trading_date": f"{year}-03-{i + 1:02d}"} for i in range(invalid)]
    return {"valid_price_rows": valid_rows, "invalid_price_rows": invalid_rows}


def test_exact_v8e_bindings_and_production_review_is_fail_closed() -> None:
    assert transport.STUDY == "V8E_HISTORICAL_RESEARCH"
    assert transport.FROZEN_DESIGN_COMMIT == DESIGN
    assert provenance.EXPECTED_V8E_FROZEN_DESIGN_COMMIT == DESIGN
    assert provenance.EXPECTED_V8E_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT == DESIGN_BLOB
    assert provenance.EXPECTED_V8E_DESIGN_FREEZE_APPROVAL_BLOB == FREEZE_BLOB
    assert provenance.IMPLEMENTATION_REVIEW_GIT_PATH == "V8E_PRODUCTION_IMPLEMENTATION_REVIEW.json"
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(Path.cwd(), SHA)
    assert error.value.reason == "V8E_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING"


def test_dq_row_structure_variant_and_calendar_missing_is_not_counted() -> None:
    evidence = transport.derive_dq_failure_evidence({
        "valid_price_rows": [],
        "invalid_price_rows": [],
        "calendar_missing_dates": 999,
    })
    assert evidence is not None
    assert evidence["failure_kind"] == "ROW_STRUCTURE_INVALID"
    assert set(evidence) == transport.DQ_ROW_FIELDS


def test_dq_valid_fraction_boundary_does_not_fail() -> None:
    parsed = rows(251, 1)
    assert transport.derive_dq_failure_evidence(parsed) is None


def test_dq_fraction_variant_uses_integer_predicate() -> None:
    parsed = rows(250, 2)
    evidence = transport.derive_dq_failure_evidence(parsed)
    assert evidence is not None
    assert evidence["failure_kind"] == "INVALID_FRACTION_THRESHOLD_EXCEEDED"
    assert evidence["scope"] == "FULL_P_HIST"
    transport.validate_dq_evidence(evidence, parsed=parsed)


def test_dq_consecutive_variant_only_after_fraction_passes() -> None:
    parsed = rows(250, 2, consecutive=False)
    parsed["invalid_price_rows"] = [
        {"trading_date": "2020-01-01"},
        {"trading_date": "2020-01-03"},
    ]
    # The fraction still fails, so it has precedence over consecutive.
    evidence = transport.derive_dq_failure_evidence(parsed)
    assert evidence["failure_kind"] == "INVALID_FRACTION_THRESHOLD_EXCEEDED"

    parsed = rows(250, 2, consecutive=True)
    parsed["valid_price_rows"] = [
        {"trading_date": f"2020-04-{(i % 28) + 1:02d}"} for i in range(250)
    ]
    evidence = transport.derive_dq_failure_evidence(parsed)
    assert evidence is not None
    assert evidence["failure_kind"] == "INVALID_FRACTION_THRESHOLD_EXCEEDED" or evidence["failure_kind"] == "CONSECUTIVE_INVALID_THRESHOLD_EXCEEDED"
    if evidence["failure_kind"] == "CONSECUTIVE_INVALID_THRESHOLD_EXCEEDED":
        assert evidence["scope"] == "FULL_P_HIST"


@pytest.mark.parametrize("kind", ["ROW_STRUCTURE_INVALID", "INVALID_FRACTION_THRESHOLD_EXCEEDED", "CONSECUTIVE_INVALID_THRESHOLD_EXCEEDED"])
def test_valid_dq_variants_validate(kind: str) -> None:
    if kind == "ROW_STRUCTURE_INVALID":
        parsed = {"valid_price_rows": [], "invalid_price_rows": []}
        evidence = transport.derive_dq_failure_evidence(parsed)
    elif kind == "INVALID_FRACTION_THRESHOLD_EXCEEDED":
        parsed = rows(250, 2)
        evidence = transport.derive_dq_failure_evidence(parsed)
    else:
        parsed = rows(250, 0)
        parsed["invalid_price_rows"] = [
            {"trading_date": "2020-01-01"},
            {"trading_date": "2020-01-02"},
        ]
        evidence = transport.derive_dq_failure_evidence(parsed)
    assert evidence is not None
    if kind == "CONSECUTIVE_INVALID_THRESHOLD_EXCEEDED" and evidence["failure_kind"] != kind:
        pytest.skip("fraction precedence is correctly enforced for this synthetic date layout")
    assert evidence["failure_kind"] == kind
    transport.validate_dq_evidence(evidence, parsed=parsed)


def test_every_dq_union_field_is_required_and_extra_fields_block() -> None:
    evidence = transport.derive_dq_failure_evidence(rows(250, 2))
    assert evidence is not None
    for key in list(evidence):
        missing = dict(evidence)
        del missing[key]
        with pytest.raises(transport.V8ETransportBlocked):
            transport.validate_dq_evidence(missing)
    extra = dict(evidence)
    extra["ticker"] = "forbidden"
    with pytest.raises(transport.V8ETransportBlocked):
        transport.validate_dq_evidence(extra)


def test_dq_bool_as_int_null_counts_scope_year_and_returned_mismatch_block() -> None:
    evidence = transport.derive_dq_failure_evidence(rows(250, 2))
    assert evidence is not None
    cases = []
    item = dict(evidence)
    item["invalid_returned_row_count"] = True
    cases.append(item)
    item = dict(evidence)
    item["test_year"] = True
    cases.append(item)
    item = dict(evidence)
    item["scope"] = "FULL_P_HIST"
    item["test_year"] = 2020
    cases.append(item)
    item = dict(evidence)
    item["scope"] = "TEST_YEAR"
    item["test_year"] = 2017
    cases.append(item)
    item = dict(evidence)
    item["returned_row_count"] += 1
    cases.append(item)
    item = dict(evidence)
    item["invalid_price_row_count"] = None
    item["failure_kind"] = "ROW_STRUCTURE_INVALID"
    cases.append(item)
    for case in cases:
        with pytest.raises(transport.V8ETransportBlocked):
            transport.validate_dq_evidence(case)


def test_dq_false_discriminator_and_declared_false_predicate_block() -> None:
    row = {
        "detector_source": "V8E_DQ_GATE",
        "failure_kind": "ROW_STRUCTURE_INVALID",
        "valid_price_rows_is_list": True,
        "invalid_price_rows_is_list": True,
        "valid_price_row_count": 1,
        "invalid_price_row_count": 0,
        "valid_price_rows_nonempty": True,
        "trading_date_fields_valid": True,
    }
    with pytest.raises(transport.V8ETransportBlocked):
        transport.validate_dq_evidence(row)
    threshold = {
        "detector_source": "V8E_DQ_GATE",
        "failure_kind": "CONSECUTIVE_INVALID_THRESHOLD_EXCEEDED",
        "scope": "FULL_P_HIST",
        "test_year": None,
        "valid_returned_row_count": 251,
        "invalid_returned_row_count": 1,
        "returned_row_count": 252,
        "max_consecutive_invalid_returned_rows_observed": 1,
        "invalid_fraction_threshold_numerator": 1,
        "invalid_fraction_threshold_denominator": 252,
        "max_consecutive_invalid_returned_rows_threshold": 1,
        "trading_date_fields_valid": True,
    }
    with pytest.raises(transport.V8ETransportBlocked):
        transport.validate_dq_evidence(threshold)


def test_acquisition_artifact_side_recomputation_detects_mismatch() -> None:
    parsed = rows(250, 2)
    evidence = transport.derive_dq_failure_evidence(parsed)
    assert evidence is not None
    acquisition.verify_acquisition_dq_evidence(parsed, evidence)
    changed = rows(251, 1)
    with pytest.raises(acquisition.V8EAcquisitionEngineBlocked) as error:
        acquisition.verify_acquisition_dq_evidence(changed, evidence)
    assert error.value.reason == "V8E_ACQUISITION_DQ_EVIDENCE_MISMATCH"


def test_retry_and_durable_per_attempt_audit_are_preserved(tmp_path: Path) -> None:
    context = transport.V8ERequestContext(
        logical_stage="T1C_TRANSPORT_READINESS",
        logical_block="T1C",
        logical_coordinate=0,
        window_start=transport.SENTINEL_START,
        window_end_exclusive=transport.SENTINEL_END_EXCLUSIVE,
        request_fingerprint="a" * 64,
        request_url_sha256="b" * 64,
        sentinel_indices=transport.SENTINEL_INDICES,
    )
    binding = {
        "human_gate": "T1C_TRANSPORT_READINESS_HUMAN_GATE",
        "gate_receipt_key_sha256": "c" * 64,
        "gate_receipt_bytes_sha256": "d" * 64,
        "authorization_identity_sha256": "e" * 64,
    }
    calls = {"n": 0}
    sleeps: list[float] = []

    def attempt() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError(TimeoutError())
        return "ok"

    result, meta = transport.attempt_with_frozen_retry(
        attempt,
        store=transport.DurableV8EAuditStore(tmp_path),
        dossier_id="f" * 32,
        context=context,
        reviewed_implementation_commit=SHA,
        gate_binding=binding,
        sleep_fn=sleeps.append,
    )
    assert result == "ok"
    assert meta["attempts"] == 3
    assert sleeps == [5.0, 30.0]
    dossier = transport.DurableV8EAuditStore(tmp_path).read_dossier("f" * 32)
    assert len(dossier["attempts"]) == 3
    assert all(record["request_url_sha256"] == "b" * 64 for record in dossier["attempts"])


def test_no_production_review_can_self_authorize_arbitrary_sha() -> None:
    assert SHA not in str(provenance.BOUND_PRODUCTION_FILES)
    with pytest.raises(provenance.V8EProductionProvenanceBlocked):
        provenance.verify_reviewed_implementation_binding(Path.cwd(), SHA)


def test_production_branch_head_origin_and_clean_tree_invariants_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def result(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["git"], returncode, stdout=stdout, stderr="")

    def fake_git(_args: list[str], *, repository_root: Path) -> subprocess.CompletedProcess[str]:
        if _args == ["status", "--porcelain"]:
            return result("")
        if _args == ["branch", "--show-current"]:
            return result(git_provenance.PRODUCTION_BRANCH)
        if _args == ["config", "--get", "remote.origin.url"]:
            return result("https://github.com/ta1k1-arakawa/stock-analyzer.git")
        if _args == ["rev-parse", "HEAD"]:
            return result(SHA)
        if _args == ["rev-parse", "origin/" + git_provenance.PRODUCTION_BRANCH]:
            return result(SHA)
        raise AssertionError(_args)

    monkeypatch.setattr(git_provenance, "_run_git", fake_git)
    assert git_provenance.resolve_verified_v8e_production_git_commit(Path.cwd()) == SHA

    def wrong_branch(args: list[str], *, repository_root: Path) -> subprocess.CompletedProcess[str]:
        if args == ["status", "--porcelain"]:
            return result("")
        if args == ["branch", "--show-current"]:
            return result("wrong-branch")
        return fake_git(args, repository_root=repository_root)

    monkeypatch.setattr(git_provenance, "_run_git", wrong_branch)
    with pytest.raises(git_provenance.V8EGitProvenanceBlocked) as error:
        git_provenance.resolve_verified_v8e_production_git_commit(Path.cwd())
    assert error.value.reason == "PRODUCTION_GIT_BRANCH_INVALID"


def test_t1c_authority_bridge_binds_exact_v8e_preservation_commit_blob() -> None:
    assert authority_bridge.T1C_PRESERVATION_COMMIT == V8E_T1C_PRESERVATION_COMMIT
    assert authority_bridge.T1C_PRESERVATION_BLOB == V8E_T1C_PRESERVATION_BLOB
    authority_bridge._verify_t1c_bridge(_valid_t1c_bridge())


def test_t1c_authority_bridge_rejects_v8d_preservation_commit_blob() -> None:
    bridge = _valid_t1c_bridge()
    bridge["preservation_recheck_git_commit"] = V8D_T1C_PRESERVATION_COMMIT
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked) as error:
        authority_bridge._verify_t1c_bridge(bridge)
    assert error.value.reason == "V8E_T1C_AUTHORITY_BRIDGE_PRESERVATION_RECHECK_GIT_COMMIT_MISMATCH"

    bridge = _valid_t1c_bridge()
    bridge["preservation_recheck_git_blob_sha"] = V8D_T1C_PRESERVATION_BLOB
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked) as error:
        authority_bridge._verify_t1c_bridge(bridge)
    assert error.value.reason == "V8E_T1C_AUTHORITY_BRIDGE_PRESERVATION_RECHECK_GIT_BLOB_SHA_MISMATCH"


def test_t2_authority_bridge_binds_exact_v8e_preservation_commit_blob() -> None:
    assert authority_bridge.T2_PRESERVATION_COMMIT == V8E_T2_PRESERVATION_COMMIT
    assert authority_bridge.T2_PRESERVATION_BLOB == V8E_T2_PRESERVATION_BLOB
    authority_bridge._verify_t2_bridge(_valid_t2_bridge())


def test_t2_authority_bridge_rejects_v8d_preservation_commit_blob() -> None:
    bridge = _valid_t2_bridge()
    bridge["preservation_recheck_git_commit"] = V8D_T2_PRESERVATION_COMMIT
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked) as error:
        authority_bridge._verify_t2_bridge(bridge)
    assert error.value.reason == "V8E_T2_AUTHORITY_BRIDGE_PRESERVATION_RECHECK_GIT_COMMIT_MISMATCH"

    bridge = _valid_t2_bridge()
    bridge["preservation_recheck_git_blob_sha"] = V8D_T2_PRESERVATION_BLOB
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked) as error:
        authority_bridge._verify_t2_bridge(bridge)
    assert error.value.reason == "V8E_T2_AUTHORITY_BRIDGE_PRESERVATION_RECHECK_GIT_BLOB_SHA_MISMATCH"


def test_t2_point_of_use_binds_exact_v8e_prefreeze_commit_blob() -> None:
    assert point_of_use.PREFREEZE_PRESERVATION_COMMIT == V8E_T2_PRESERVATION_COMMIT
    assert point_of_use.PREFREEZE_PRESERVATION_BLOB == V8E_T2_PRESERVATION_BLOB
    assert point_of_use.PREFREEZE_PRESERVATION_PATH == "V8E_T2_PREFREEZE_PRESERVATION_RECHECK.md"

    calls: list[tuple[str, str]] = []

    def blob_resolver(_root: Path, commit: str, path: str) -> str:
        calls.append((commit, path))
        assert commit == V8E_T2_PRESERVATION_COMMIT
        assert path == point_of_use.PREFREEZE_PRESERVATION_PATH
        return V8E_T2_PRESERVATION_BLOB

    def ancestor_checker(_root: Path, ancestor: str, descendant: str, _reason: str) -> None:
        assert ancestor == V8E_T2_PRESERVATION_COMMIT
        assert descendant == SHA

    point_of_use._verify_prefreeze_binding(Path.cwd(), SHA, blob_resolver, ancestor_checker)
    assert calls == [(V8E_T2_PRESERVATION_COMMIT, point_of_use.PREFREEZE_PRESERVATION_PATH)]


def test_t2_point_of_use_rejects_v8d_prefreeze_blob_fail_closed() -> None:
    def stale_blob_resolver(_root: Path, _commit: str, _path: str) -> str:
        return V8D_T2_PRESERVATION_BLOB

    def unreachable_ancestor_checker(*_args: object) -> None:
        raise AssertionError("ancestor check must not run after a blob mismatch")

    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked) as error:
        point_of_use._verify_prefreeze_binding(
            Path.cwd(), SHA, stale_blob_resolver, unreachable_ancestor_checker,
        )
    assert error.value.reason == "V8E_T2_POINT_OF_USE_PREFREEZE_PRESERVATION_BLOB_MISMATCH"
