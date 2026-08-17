from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8e_acquisition_engine as acquisition
from src import v8e_audit
from src import v8e_authority_bridge as authority_bridge
from src import v8e_git_provenance as git_provenance
from src import v8e_human_gate_consumption as gate_consumption
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
IMPLEMENTATION_SHA = SHA
AUTH_IDENTITY = "synthetic-authorization-identity"
SAFE_URL = "https://query1.finance.yahoo.com/synthetic"


def _raise(error: BaseException):
    def call():
        raise error
    return call


def _fixed_clock() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def _gate_root(tmp_path: Path) -> Path:
    return tmp_path / "gate-state"


def _consume_gate(root: Path, *, stage: str, reviewed_commit: str = IMPLEMENTATION_SHA,
                   auth_identity: str = AUTH_IDENTITY,
                   frozen_design_commit: str = transport.FROZEN_DESIGN_COMMIT) -> dict[str, str]:
    """Durably consume the exact gate for ``stage`` under ``root`` (a
    per-test synthetic gate-receipt state root -- never production state)."""
    binding = gate_consumption.consume_gate_and_bind(
        root, logical_stage=stage, v8e_frozen_design_commit=frozen_design_commit,
        reviewed_production_implementation_commit=reviewed_commit,
        raw_authorization_identity=auth_identity, clock=_fixed_clock,
    )
    return {
        "human_gate": binding.human_gate,
        "gate_receipt_key_sha256": binding.gate_receipt_key_sha256,
        "gate_receipt_bytes_sha256": binding.gate_receipt_bytes_sha256,
        "authorization_identity_sha256": binding.authorization_identity_sha256,
    }


def _plan(stage: str, coordinate: int, request_fn) -> transport.V8ERequestPlan:
    block = "T1C" if stage.startswith("T1C") else "T2"
    readiness = "READINESS" in stage
    start = transport.SENTINEL_START if readiness else "2020-01-01"
    end = transport.SENTINEL_END_EXCLUSIVE if readiness else "2020-01-08"
    return transport.V8ERequestPlan(
        request_fn=request_fn,
        request_fingerprint=transport.make_request_fingerprint(
            logical_stage=stage, logical_block=block, logical_coordinate=coordinate,
            window_start=start, window_end_exclusive=end,
        ),
        request_url_sha256=transport.sha256_url(SAFE_URL),
    )


def _context(stage: str = "T1C_RAW_ACQUISITION", coordinate: int = 0) -> transport.V8ERequestContext:
    block = "T1C" if stage.startswith("T1C") else "T2"
    readiness = "READINESS" in stage
    start = transport.SENTINEL_START if readiness else "2020-01-01"
    end = transport.SENTINEL_END_EXCLUSIVE if readiness else "2020-01-08"
    return transport.V8ERequestContext(
        logical_stage=stage, logical_block=block, logical_coordinate=coordinate,
        window_start=start, window_end_exclusive=end,
        request_fingerprint=transport.make_request_fingerprint(
            logical_stage=stage, logical_block=block, logical_coordinate=coordinate,
            window_start=start, window_end_exclusive=end,
        ),
        request_url_sha256=transport.sha256_url(SAFE_URL),
        sentinel_indices=transport.SENTINEL_INDICES if readiness else None,
    )


def _single_attempt(tmp_path: Path, fn, *, stage: str = "T1C_RAW_ACQUISITION"):
    store = transport.DurableV8EAuditStore(tmp_path)
    dossier_id = store.new_id()
    gate_binding = _consume_gate(_gate_root(tmp_path), stage=stage)
    result = transport.attempt_with_frozen_retry(
        fn, store=store, dossier_id=dossier_id, context=_context(stage),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding,
        sleep_fn=lambda _seconds: None,
    )
    return result, store, dossier_id


def _rehashed_dossier(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value["audit_artifact_self_hash"] = transport.canonical_sha256(
        {key: item for key, item in value.items() if key != "audit_artifact_self_hash"}
    )
    path.write_bytes(transport.canonical_json_bytes(value))


def _git_config_commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", message],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()


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
        # 502 valid + 2 consecutive invalid = 504 total: 2*252 == 504 is NOT
        # > 504, so the invalid-fraction check does not fire first, and the
        # consecutive-invalid check (max run 2 > threshold 1) fires instead.
        parsed = rows(502, 2, consecutive=True)
        evidence = transport.derive_dq_failure_evidence(parsed)
    assert evidence is not None
    assert evidence["failure_kind"] == kind
    transport.validate_dq_evidence(evidence, parsed=parsed)


def test_dq_consecutive_invalid_valid_case_uses_exact_full_p_hist_numbers() -> None:
    """Mandatory regression: a mechanically valid CONSECUTIVE_INVALID_
    THRESHOLD_EXCEEDED case where the fraction check does NOT fail first.
    FULL_P_HIST with exactly 502 valid + 2 consecutive invalid returned
    rows (total 504): 2*252 == 504 is not > 504, so invalid-fraction does
    not exceed the threshold, while max consecutive=2 does exceed 1. No
    skip/xfail is used -- this must always emit exactly
    CONSECUTIVE_INVALID_THRESHOLD_EXCEEDED."""
    parsed = rows(502, 2, consecutive=True)
    observations = transport._dq_observations(parsed)
    assert len(observations) == 504
    invalid_count = sum(1 for _, valid in observations if not valid)
    assert invalid_count == 2
    assert invalid_count * 252 == 504  # exactly at the boundary: not > 504.

    evidence = transport.derive_dq_failure_evidence(parsed)
    assert evidence is not None
    assert evidence["failure_kind"] == "CONSECUTIVE_INVALID_THRESHOLD_EXCEEDED"
    assert evidence["scope"] == "FULL_P_HIST"
    assert evidence["test_year"] is None
    assert evidence["valid_returned_row_count"] == 502
    assert evidence["invalid_returned_row_count"] == 2
    assert evidence["returned_row_count"] == 504
    assert evidence["max_consecutive_invalid_returned_rows_observed"] == 2
    transport.validate_dq_evidence(evidence, parsed=parsed)


def test_dq_full_p_hist_scope_precedes_test_year_scope() -> None:
    # This data fails both at FULL_P_HIST scope and, in isolation, at
    # TEST_YEAR 2020 scope (all rows fall in 2020). The verifier must
    # report FULL_P_HIST -- it is checked first in the frozen scope order.
    parsed = rows(250, 2)
    observations = transport._dq_observations(parsed)
    year_scoped = [item for item in observations if item[0].startswith("2020-")]
    isolated_year_evidence = transport._dq_threshold_evidence("TEST_YEAR", 2020, year_scoped)
    assert isolated_year_evidence is not None
    assert isolated_year_evidence["failure_kind"] == "INVALID_FRACTION_THRESHOLD_EXCEEDED"

    evidence = transport.derive_dq_failure_evidence(parsed)
    assert evidence is not None
    assert evidence["scope"] == "FULL_P_HIST"


def test_dq_test_year_scope_order_is_ascending() -> None:
    # Both 2018 and 2019 independently fail (fraction), while a large
    # filler block elsewhere keeps the overall FULL_P_HIST scope PASS.
    # The verifier must report the lower year (2018) first.
    filler_valid = [{"trading_date": f"2021-01-{(i % 28) + 1:02d}"} for i in range(1010)]

    def year_block(year: int) -> tuple[list[dict], list[dict]]:
        valid = [{"trading_date": f"{year}-02-02"}]
        invalid = [{"trading_date": f"{year}-02-01"}, {"trading_date": f"{year}-02-03"}]
        return valid, invalid

    valid_2018, invalid_2018 = year_block(2018)
    valid_2019, invalid_2019 = year_block(2019)
    # A separator row dated between the two year blocks so the last invalid
    # row of 2018 and the first invalid row of 2019 are never adjacent in
    # the globally sorted FULL_P_HIST sequence (which would otherwise make
    # the overall consecutive-invalid check fail before TEST_YEAR is ever
    # reached).
    separator = [{"trading_date": "2018-06-01"}]
    parsed = {
        "valid_price_rows": filler_valid + valid_2018 + separator + valid_2019,
        "invalid_price_rows": invalid_2018 + invalid_2019,
    }
    full_evidence = transport._dq_threshold_evidence(
        "FULL_P_HIST", None, transport._dq_observations(parsed),
    )
    assert full_evidence is None

    evidence = transport.derive_dq_failure_evidence(parsed)
    assert evidence is not None
    assert evidence["scope"] == "TEST_YEAR"
    assert evidence["test_year"] == 2018
    transport.validate_dq_evidence(evidence, parsed=parsed)


def test_dq_calendar_missing_dates_never_manufacture_a_threshold_failure() -> None:
    parsed = rows(251, 1)
    parsed["calendar_missing_dates"] = 5000  # calendar gaps, not malformed returned rows
    assert transport.derive_dq_failure_evidence(parsed) is None


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


# ---------------------------------------------------------------------------
# Frozen retry matrix (synthetic/public-safe transport + durable audit only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_every_frozen_retryable_http_status_is_retried_and_audited(tmp_path: Path, status: int) -> None:
    calls: list[int] = []
    error = urllib.error.HTTPError(SAFE_URL, status, "synthetic message", {}, None)

    def request():
        calls.append(1)
        raise error

    with pytest.raises(urllib.error.HTTPError):
        _single_attempt(tmp_path, request)
    assert len(calls) == 3
    dossier = next(tmp_path.glob("dossier-*.json"))
    checked = v8e_audit.verify_dossier(
        dossier, gate_receipt_state_root=_gate_root(tmp_path), expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
    )
    assert [record["classification"] for record in checked["attempts"]] == [f"HTTP_{status}"] * 3
    assert [record["retryable"] for record in checked["attempts"]] == [True, True, True]
    assert checked["attempts"][-1]["terminal_state"] == "TERMINAL_FAILURE"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
def test_every_frozen_nonretryable_http_status_stops_after_one_attempt(tmp_path: Path, status: int) -> None:
    calls: list[int] = []
    error = urllib.error.HTTPError(SAFE_URL, status, "synthetic message", {}, None)

    def request():
        calls.append(1)
        raise error

    with pytest.raises(urllib.error.HTTPError):
        _single_attempt(tmp_path, request)
    assert len(calls) == 1
    checked = v8e_audit.verify_dossier(
        next(tmp_path.glob("dossier-*.json")), gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
    )
    assert checked["attempts"][0]["classification"] == f"HTTP_{status}"
    assert checked["attempts"][0]["retryable"] is False
    assert checked["attempts"][0]["terminal_state"] == "TERMINAL_FAILURE"


def test_retry_exhaustion_has_three_attempts_and_exact_backoffs(tmp_path: Path) -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def request():
        calls.append(1)
        raise TimeoutError("not persisted")

    with pytest.raises(TimeoutError):
        store = transport.DurableV8EAuditStore(tmp_path)
        gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
        transport.attempt_with_frozen_retry(
            request, store=store, dossier_id=store.new_id(), context=_context(),
            reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=sleeps.append,
        )
    assert len(calls) == 3
    assert transport.BACKOFF_SECONDS == (5, 30)
    assert sleeps == [5.0, 30.0]


def test_retryable_then_nonretryable_transition_stops_immediately(tmp_path: Path) -> None:
    calls: list[int] = []

    def request():
        calls.append(1)
        status = 429 if len(calls) == 1 else 400
        raise urllib.error.HTTPError(SAFE_URL, status, "synthetic", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        _single_attempt(tmp_path, request)
    assert len(calls) == 2
    checked = v8e_audit.verify_dossier(
        next(tmp_path.glob("dossier-*.json")), gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
    )
    assert [item["classification"] for item in checked["attempts"]] == ["HTTP_429", "HTTP_400"]
    assert [item["retryable"] for item in checked["attempts"]] == [True, False]


def test_success_after_retries_sleeps_only_between_failed_attempts(tmp_path: Path) -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def request():
        calls.append(len(calls) + 1)
        if len(calls) < 3:
            raise urllib.error.HTTPError(SAFE_URL, 429, "synthetic", {}, None)
        return "ok"

    store = transport.DurableV8EAuditStore(tmp_path)
    dossier_id = store.new_id()
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
    value, meta = transport.attempt_with_frozen_retry(
        request, store=store, dossier_id=dossier_id, context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=sleeps.append,
    )
    assert value == "ok"
    assert meta["attempts"] == 3
    assert sleeps == [5.0, 30.0]
    checked = v8e_audit.verify_dossier(
        store._path(dossier_id), gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
    )
    assert [item["terminal_state"] for item in checked["attempts"]] == [
        "RETRYABLE_FAILURE", "RETRYABLE_FAILURE", "SUCCESS",
    ]


# ---------------------------------------------------------------------------
# Durable audit persistence and tamper detection
# ---------------------------------------------------------------------------


def test_audit_write_failure_prevents_retry_or_success_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    sleeps: list[float] = []
    store = transport.DurableV8EAuditStore(tmp_path)
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")

    def request():
        calls.append(1)
        raise TimeoutError("must be audited first")

    def fail_write(*_args, **_kwargs):
        raise transport.V8EAuditPersistenceBlocked()

    monkeypatch.setattr(store, "write_attempt", fail_write)
    with pytest.raises(transport.V8EAuditPersistenceBlocked):
        transport.attempt_with_frozen_retry(
            request, store=store, dossier_id=store.new_id(), context=_context(),
            reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=sleeps.append,
        )
    assert calls == [1]
    assert sleeps == []
    assert not list(tmp_path.glob("aggregate-*.json"))


def test_failed_attempt_audit_is_durable_before_next_request(tmp_path: Path) -> None:
    order: list[str] = []

    def request():
        order.append("request")
        if order.count("request") == 1:
            raise urllib.error.HTTPError(SAFE_URL, 429, "hidden", {}, None)
        order.append("success")
        return "ok"

    class OrderedStore(transport.DurableV8EAuditStore):
        def write_attempt(self, *args, **kwargs):
            order.append("persist")
            return super().write_attempt(*args, **kwargs)

    store = OrderedStore(tmp_path)
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_RAW_ACQUISITION")
    transport.attempt_with_frozen_retry(
        request, store=store, dossier_id=store.new_id(), context=_context(),
        reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding,
        sleep_fn=lambda _seconds: order.append("sleep"),
    )
    assert order == ["request", "persist", "sleep", "request", "success", "persist"]


def test_tampered_rehashed_retry_chronology_blocks(tmp_path: Path) -> None:
    error = urllib.error.HTTPError(SAFE_URL, 429, "synthetic", {}, None)
    with pytest.raises(urllib.error.HTTPError):
        _single_attempt(tmp_path, _raise(error))
    dossier = next(tmp_path.glob("dossier-*.json"))

    def truncate_to_first_terminal(value):
        value["attempts"] = value["attempts"][:1]
        value["attempts"][0]["terminal_state"] = "TERMINAL_FAILURE"

    _rehashed_dossier(dossier, truncate_to_first_terminal)
    with pytest.raises(v8e_audit.V8EAuditVerificationBlocked) as error:
        v8e_audit.verify_dossier(
            dossier, gate_receipt_state_root=_gate_root(tmp_path),
            expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
        )
    assert error.value.reason == "V8E_DOSSIER_RETRYABLE_TERMINAL_NOT_EXHAUSTED"


def test_valid_retryable_terminal_after_exhaustion_passes(tmp_path: Path) -> None:
    error = urllib.error.HTTPError(SAFE_URL, 429, "synthetic", {}, None)
    with pytest.raises(urllib.error.HTTPError):
        _single_attempt(tmp_path, _raise(error))
    dossier = next(tmp_path.glob("dossier-*.json"))
    checked = v8e_audit.verify_dossier(
        dossier, gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA,
    )
    assert len(checked["attempts"]) == 3
    assert checked["attempts"][-1]["terminal_state"] == "TERMINAL_FAILURE"


def test_verify_dossier_and_aggregate_require_gate_receipt_state_root(tmp_path: Path) -> None:
    _result, store, dossier_id = _single_attempt(tmp_path, lambda: "ok")
    with pytest.raises(v8e_audit.V8EAuditVerificationBlocked) as error:
        v8e_audit.verify_dossier(store._path(dossier_id))
    assert error.value.reason == "V8E_GATE_RECEIPT_STATE_ROOT_REQUIRED"


def _success_readiness_artifacts(tmp_path: Path, *, stage: str = "T1C_TRANSPORT_READINESS") -> dict:
    gate_binding = _consume_gate(_gate_root(tmp_path), stage=stage)
    return transport.execute_v8e_stage(
        stage=stage, request_factory=lambda coordinate: _plan(stage, coordinate, lambda: "ok"),
        store=transport.DurableV8EAuditStore(tmp_path), reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=gate_binding, window_start=transport.SENTINEL_START,
        window_end_exclusive=transport.SENTINEL_END_EXCLUSIVE, request_count=len(transport.SENTINEL_INDICES),
        sleep_fn=lambda _seconds: None,
    )


def test_readiness_end_to_end_success_and_independent_aggregate_verification(tmp_path: Path) -> None:
    result = _success_readiness_artifacts(tmp_path)
    assert result["aggregate"]["result"] == "PASS"
    checked = v8e_audit.verify_aggregate(
        result["aggregate_path"], result["dossier_paths"], gate_receipt_state_root=_gate_root(tmp_path),
        expected_reviewed_implementation_commit=IMPLEMENTATION_SHA, expected_stage="T1C_TRANSPORT_READINESS",
    )
    assert checked["sentinel_count"] == 3 and checked["sentinel_pass_count"] == 3


def test_mixing_dossiers_from_two_different_stage_receipts_blocks(tmp_path: Path) -> None:
    shared_gate_root = tmp_path / "gate-state"
    result_t1c = transport.execute_v8e_stage(
        stage="T1C_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T1C_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        store=transport.DurableV8EAuditStore(tmp_path / "t1c"), reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=_consume_gate(shared_gate_root, stage="T1C_TRANSPORT_READINESS"),
        window_start=transport.SENTINEL_START, window_end_exclusive=transport.SENTINEL_END_EXCLUSIVE,
        request_count=len(transport.SENTINEL_INDICES), sleep_fn=lambda _seconds: None,
    )
    result_t2 = transport.execute_v8e_stage(
        stage="T2_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T2_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        store=transport.DurableV8EAuditStore(tmp_path / "t2"), reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=_consume_gate(shared_gate_root, stage="T2_TRANSPORT_READINESS"),
        window_start=transport.SENTINEL_START, window_end_exclusive=transport.SENTINEL_END_EXCLUSIVE,
        request_count=len(transport.SENTINEL_INDICES), sleep_fn=lambda _seconds: None,
    )
    mixed_dossier_paths = result_t1c["dossier_paths"][:2] + [result_t2["dossier_paths"][0]]
    with pytest.raises(v8e_audit.V8EAuditVerificationBlocked):
        v8e_audit.verify_aggregate(
            result_t1c["aggregate_path"], mixed_dossier_paths, gate_receipt_state_root=shared_gate_root,
        )


# ---------------------------------------------------------------------------
# Human gate consumption: schema, one-shot semantics, stage binding
# ---------------------------------------------------------------------------


def _valid_receipt_payload(*, stage: str = "T1C_TRANSPORT_READINESS", reviewed_commit: str = IMPLEMENTATION_SHA,
                            frozen_design_commit: str = transport.FROZEN_DESIGN_COMMIT, auth_hash: str = "0" * 64) -> dict:
    return {
        "schema_version": gate_consumption.SCHEMA_VERSION,
        "study": gate_consumption.STUDY_NAME,
        "repository": gate_consumption.REPOSITORY_IDENTITY,
        "gate": gate_consumption.STAGE_GATE[stage],
        "logical_stage": stage,
        "v8e_frozen_design_commit": frozen_design_commit,
        "reviewed_production_implementation_commit": reviewed_commit,
        "authorization_identity_sha256": auth_hash,
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": gate_consumption.CONSUMPTION_BOUNDARY,
        "consumed_at_utc": "2026-01-01T00:00:00Z",
    }


def _write_receipt(root: Path, *, stage: str = "T1C_TRANSPORT_READINESS", payload: dict | None = None) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    key = gate_consumption.compute_receipt_key(gate_consumption.STAGE_GATE[stage], transport.FROZEN_DESIGN_COMMIT)
    body = payload if payload is not None else _valid_receipt_payload(stage=stage)
    path = root / (key + ".json")
    path.write_bytes(json.dumps(body).encode())
    return path, key


def test_all_four_stage_gate_mappings_are_exact() -> None:
    assert gate_consumption.STAGE_GATE == {
        "T1C_TRANSPORT_READINESS": "T1C_TRANSPORT_READINESS_HUMAN_GATE",
        "T1C_RAW_ACQUISITION": "T1C_RAW_ACQUISITION_HUMAN_GATE",
        "T2_TRANSPORT_READINESS": "T2_TRANSPORT_READINESS_HUMAN_GATE",
        "T2_RAW_ACQUISITION": "T2_RAW_ACQUISITION_HUMAN_GATE",
    }
    assert len(gate_consumption.KNOWN_GATES) == 4 and len(set(gate_consumption.KNOWN_GATES)) == 4


def test_wrong_stage_gate_mapping_blocks_at_transport_layer(tmp_path: Path) -> None:
    gate_binding = _consume_gate(_gate_root(tmp_path), stage="T1C_TRANSPORT_READINESS")
    store = transport.DurableV8EAuditStore(tmp_path)
    with pytest.raises(transport.V8ETransportBlocked) as error:
        transport.attempt_with_frozen_retry(
            lambda: "ok", store=store, dossier_id=store.new_id(), context=_context("T1C_RAW_ACQUISITION"),
            reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=lambda _s: None,
        )
    assert error.value.reason == "V8E_GATE_BINDING_STAGE_MISMATCH"


@pytest.mark.parametrize("granted_stage,attempted_stage", [
    ("T1C_TRANSPORT_READINESS", "T1C_RAW_ACQUISITION"),
    ("T1C_TRANSPORT_READINESS", "T2_TRANSPORT_READINESS"),
    ("T1C_RAW_ACQUISITION", "T1C_TRANSPORT_READINESS"),
    ("T2_TRANSPORT_READINESS", "T2_RAW_ACQUISITION"),
    ("T2_RAW_ACQUISITION", "T2_TRANSPORT_READINESS"),
])
def test_a_stage_gate_receipt_cannot_authorize_a_different_stage(tmp_path: Path, granted_stage: str, attempted_stage: str) -> None:
    gate_binding = _consume_gate(_gate_root(tmp_path), stage=granted_stage)
    store = transport.DurableV8EAuditStore(tmp_path)
    with pytest.raises(transport.V8ETransportBlocked) as error:
        transport.attempt_with_frozen_retry(
            lambda: "ok", store=store, dossier_id=store.new_id(), context=_context(attempted_stage),
            reviewed_implementation_commit=IMPLEMENTATION_SHA, gate_binding=gate_binding, sleep_fn=lambda _s: None,
        )
    assert error.value.reason == "V8E_GATE_BINDING_STAGE_MISMATCH"


def test_wrong_stage_gate_mapping_in_receipt_blocks_independent_read(tmp_path: Path) -> None:
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload(stage="T1C_TRANSPORT_READINESS")
    payload["gate"] = gate_consumption.GATE_T2_RAW_ACQUISITION
    _path, key = _write_receipt(root, stage="T1C_TRANSPORT_READINESS", payload=payload)
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert error.value.reason == "V8E_HUMAN_GATE_STAGE_GATE_MISMATCH"


def test_empty_or_missing_authorization_identity_blocks(tmp_path: Path) -> None:
    for bad_identity in ("", None):
        with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
            gate_consumption.consume_gate_and_bind(
                _gate_root(tmp_path), logical_stage="T1C_TRANSPORT_READINESS",
                v8e_frozen_design_commit=transport.FROZEN_DESIGN_COMMIT,
                reviewed_production_implementation_commit=IMPLEMENTATION_SHA,
                raw_authorization_identity=bad_identity, clock=_fixed_clock,
            )
        assert error.value.reason == "V8E_HUMAN_GATE_AUTHORIZATION_IDENTITY_REQUIRED"


def test_raw_authorization_identity_never_appears_in_durable_artifacts(tmp_path: Path) -> None:
    secret_identity = "SUPER-SECRET-RAW-AUTH-IDENTITY-NOT-A-HASH"
    gate_root = _gate_root(tmp_path)
    gate_binding = _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS", auth_identity=secret_identity)
    result = transport.execute_v8e_stage(
        stage="T1C_TRANSPORT_READINESS",
        request_factory=lambda coordinate: _plan("T1C_TRANSPORT_READINESS", coordinate, lambda: "ok"),
        store=transport.DurableV8EAuditStore(tmp_path), reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=gate_binding, window_start=transport.SENTINEL_START,
        window_end_exclusive=transport.SENTINEL_END_EXCLUSIVE, request_count=len(transport.SENTINEL_INDICES),
        sleep_fn=lambda _seconds: None,
    )
    receipt_path = gate_root / (gate_binding["gate_receipt_key_sha256"] + ".json")
    receipt_raw = receipt_path.read_bytes()
    aggregate_raw = Path(result["aggregate_path"]).read_bytes()
    dossier_raw_all = b"".join(Path(p).read_bytes() for p in result["dossier_paths"])
    assert secret_identity.encode() not in receipt_raw
    assert secret_identity.encode() not in aggregate_raw
    assert secret_identity.encode() not in dossier_raw_all


def test_authorization_identity_sha256_is_correct(tmp_path: Path) -> None:
    identity = "check-this-exact-identity"
    binding = gate_consumption.consume_gate_and_bind(
        _gate_root(tmp_path), logical_stage="T1C_TRANSPORT_READINESS",
        v8e_frozen_design_commit=transport.FROZEN_DESIGN_COMMIT,
        reviewed_production_implementation_commit=IMPLEMENTATION_SHA, raw_authorization_identity=identity,
        clock=_fixed_clock,
    )
    assert binding.authorization_identity_sha256 == hashlib.sha256(identity.encode("utf-8")).hexdigest()


def test_receipt_exact_schema_enforced() -> None:
    assert gate_consumption.RECEIPT_FIELDS == (
        "schema_version", "study", "repository", "gate", "logical_stage",
        "v8e_frozen_design_commit", "reviewed_production_implementation_commit",
        "authorization_identity_sha256", "consumed", "consumption_count",
        "consumption_boundary", "consumed_at_utc",
    )


def test_duplicate_receipt_json_key_blocks(tmp_path: Path) -> None:
    root = tmp_path / "gate-state"
    root.mkdir()
    key = gate_consumption.compute_receipt_key(gate_consumption.GATE_T1C_TRANSPORT_READINESS, transport.FROZEN_DESIGN_COMMIT)
    raw = (
        b'{"schema_version": "' + gate_consumption.SCHEMA_VERSION.encode() + b'", '
        b'"schema_version": "DUPLICATE", "study": "x", "repository": "y"}'
    )
    (root / (key + ".json")).write_bytes(raw)
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_DUPLICATE_KEY"


@pytest.mark.parametrize("missing_field", gate_consumption.RECEIPT_FIELDS)
def test_missing_receipt_field_blocks(tmp_path: Path, missing_field: str) -> None:
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    del payload[missing_field]
    _path, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_SCHEMA_INVALID"


def test_extra_receipt_field_blocks(tmp_path: Path) -> None:
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    payload["unexpected_extra_field"] = "x"
    _path, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_SCHEMA_INVALID"


def test_consumed_flag_not_true_blocks(tmp_path: Path) -> None:
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    payload["consumed"] = False
    _path, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_CONSUMED_FLAG_INVALID"


def test_consumption_count_not_one_blocks(tmp_path: Path) -> None:
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    payload["consumption_count"] = 2
    _path, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_CONSUMPTION_COUNT_INVALID"


def test_wrong_consumption_boundary_blocks(tmp_path: Path) -> None:
    root = tmp_path / "gate-state"
    payload = _valid_receipt_payload()
    payload["consumption_boundary"] = "SOMETIME_LATER"
    _path, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_CONSUMPTION_BOUNDARY_INVALID"


def test_wrong_frozen_design_commit_in_receipt_blocks(tmp_path: Path) -> None:
    root = tmp_path / "gate-state"
    binding = gate_consumption.consume_gate_and_bind(
        root, logical_stage="T1C_TRANSPORT_READINESS", v8e_frozen_design_commit=transport.FROZEN_DESIGN_COMMIT,
        reviewed_production_implementation_commit=IMPLEMENTATION_SHA, raw_authorization_identity=AUTH_IDENTITY,
        clock=_fixed_clock,
    )
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(
            root, binding.gate_receipt_key_sha256, expected_gate=binding.human_gate,
            expected_v8e_frozen_design_commit="9" * 40,
        )
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_DESIGN_COMMIT_MISMATCH"


def test_receipt_stored_under_wrong_key_blocks(tmp_path: Path) -> None:
    root = tmp_path / "gate-state"
    binding = gate_consumption.consume_gate_and_bind(
        root, logical_stage="T1C_TRANSPORT_READINESS", v8e_frozen_design_commit=transport.FROZEN_DESIGN_COMMIT,
        reviewed_production_implementation_commit=IMPLEMENTATION_SHA, raw_authorization_identity=AUTH_IDENTITY,
        clock=_fixed_clock,
    )
    real_path = root / (binding.gate_receipt_key_sha256 + ".json")
    wrong_key = "0" * 64
    (root / (wrong_key + ".json")).write_bytes(real_path.read_bytes())
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, wrong_key)
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_KEY_CONTENT_MISMATCH"


def test_missing_receipt_blocks(tmp_path: Path) -> None:
    root = tmp_path / "gate-state"
    root.mkdir()
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, "0" * 64)
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_MISSING"


@pytest.mark.parametrize("timestamp", [
    "2026-01-01T00:00:00Z",
    "2026-01-01T00:00:00.123456Z",
])
def test_gate_receipt_accepts_canonical_utc_timestamps(tmp_path: Path, timestamp: str) -> None:
    payload = _valid_receipt_payload()
    payload["consumed_at_utc"] = timestamp
    root = tmp_path / "gate-state"
    _path, key = _write_receipt(root, payload=payload)
    assert gate_consumption.read_gate_consumption_receipt(root, key)["consumed_at_utc"] == timestamp


@pytest.mark.parametrize("timestamp", [
    "",
    "not-a-timestamp",
    "2026-01-01T00:00:00+00:00",
    "2026-01-01T00:00:00+01:00",
    "2026-01-01T00:00:00",
    "2026-01-01T00:00:00z",
    "2026-01-01 00:00:00Z",
    "2026-01-01T00:00:00.1Z",
    "2026-01-01T00:00:00.12345Z",
    "2026-01-01T00:00:00.1234567Z",
    "2026-02-29T00:00:00Z",
    " 2026-01-01T00:00:00Z",
    "2026-01-01T00:00:00Z ",
    "2026-01-01T00:00:00Ztrailing",
])
def test_gate_receipt_rejects_noncanonical_or_invalid_utc_timestamps(tmp_path: Path, timestamp: str) -> None:
    payload = _valid_receipt_payload()
    payload["consumed_at_utc"] = timestamp
    root = tmp_path / "gate-state"
    _path, key = _write_receipt(root, payload=payload)
    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        gate_consumption.read_gate_consumption_receipt(root, key)
    assert error.value.reason == "V8E_HUMAN_GATE_RECEIPT_TIMESTAMP_INVALID"


def test_one_shot_gate_cannot_be_reset_by_fresh_authorization_identity(tmp_path: Path) -> None:
    gate_root = tmp_path / "gate-state"
    _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS", auth_identity="first-authorization-identity")

    def consume_again(identity: str):
        return gate_consumption.consume_gate_and_bind(
            gate_root, logical_stage="T1C_TRANSPORT_READINESS",
            v8e_frozen_design_commit=transport.FROZEN_DESIGN_COMMIT,
            reviewed_production_implementation_commit=IMPLEMENTATION_SHA,
            raw_authorization_identity=identity, clock=_fixed_clock,
        )

    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        consume_again("first-authorization-identity")
    assert error.value.reason.startswith("V8E_HUMAN_GATE_ALREADY_CONSUMED")

    with pytest.raises(gate_consumption.V8EHumanGateConsumptionBlocked) as error:
        consume_again("a-completely-different-fresh-identity")
    assert error.value.reason.startswith("V8E_HUMAN_GATE_ALREADY_CONSUMED")


def test_no_receipt_reset_or_delete_api_exists() -> None:
    forbidden_substrings = ("delete", "reset", "remove", "clear", "revoke")
    public_names = [name for name in dir(gate_consumption) if not name.startswith("_")]
    for name in public_names:
        lowered = name.lower()
        assert not any(word in lowered for word in forbidden_substrings), name


def test_gate_receipt_exists_durably_before_first_request_fn_invocation(tmp_path: Path) -> None:
    gate_root = tmp_path / "gate-state"
    gate_binding = _consume_gate(gate_root, stage="T1C_TRANSPORT_READINESS")
    call_order: list[int] = []

    def request_factory(coordinate: int):
        def request_fn():
            key = gate_consumption.compute_receipt_key(gate_consumption.GATE_T1C_TRANSPORT_READINESS, transport.FROZEN_DESIGN_COMMIT)
            assert (gate_root / (key + ".json")).exists()
            call_order.append(coordinate)
            return "ok"
        return _plan("T1C_TRANSPORT_READINESS", coordinate, request_fn)

    transport.execute_v8e_stage(
        stage="T1C_TRANSPORT_READINESS", request_factory=request_factory,
        store=transport.DurableV8EAuditStore(tmp_path / "audit"), reviewed_implementation_commit=IMPLEMENTATION_SHA,
        gate_binding=gate_binding, window_start=transport.SENTINEL_START,
        window_end_exclusive=transport.SENTINEL_END_EXCLUSIVE, request_count=len(transport.SENTINEL_INDICES),
        sleep_fn=lambda _seconds: None,
    )
    assert call_order == list(transport.SENTINEL_INDICES)


def test_verify_dossier_without_gate_receipt_state_root_uses_generic_reason(tmp_path: Path) -> None:
    result = _success_readiness_artifacts(tmp_path)
    with pytest.raises(v8e_audit.V8EAuditVerificationBlocked) as error:
        v8e_audit.verify_aggregate(result["aggregate_path"], result["dossier_paths"])
    assert error.value.reason == "V8E_GATE_RECEIPT_STATE_ROOT_REQUIRED"


# ---------------------------------------------------------------------------
# Production provenance: reviewed-implementation binding (synthetic git only)
# ---------------------------------------------------------------------------


def _repo_with_raw_file(tmp_path: Path, name: str, relative_path: str, raw_bytes: bytes) -> tuple[Path, str]:
    repo = tmp_path / name
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    path = repo / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)
    head = _git_config_commit(repo, "raw file")
    return repo, head


def _valid_review_json(reviewed_commit: str) -> bytes:
    return json.dumps({
        "schema_version": provenance.IMPLEMENTATION_REVIEW_SCHEMA_VERSION,
        "study": provenance.STUDY_NAME,
        "artifact_role": provenance.IMPLEMENTATION_REVIEW_ARTIFACT_ROLE,
        "reviewed_implementation_git_commit": reviewed_commit,
        "review_result": "PASS",
        "approval_status": "APPROVED",
    }).encode()


def _build_bound_file_repo(tmp_path: Path, *, mutate_file: str | None, reviewed_commit_override: str | None = None):
    """A synthetic two-commit repository: commit 1 is the "reviewed"
    implementation state (every BOUND_PRODUCTION_FILES path present);
    commit 2 is the current "verified HEAD" state, carrying the review
    artifact plus either an identical or a mutated bound file."""
    repo = tmp_path / "bound_repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    base_content = {
        path: ("# " + path + " v1\n").encode() for path in provenance.BOUND_PRODUCTION_FILES
    }
    for relative_path, content in base_content.items():
        file_path = repo / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)
    reviewed_commit = _git_config_commit(repo, "reviewed implementation")

    if mutate_file is not None:
        (repo / mutate_file).write_bytes(base_content[mutate_file] + b"# mutated\n")
    else:
        (repo / "UNRELATED_DOC.md").write_text("docs-only change, no bound blob drift")
    review_commit_value = reviewed_commit_override or reviewed_commit
    review_path = repo / provenance.IMPLEMENTATION_REVIEW_GIT_PATH
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_bytes(_valid_review_json(review_commit_value))
    head_commit = _git_config_commit(repo, "current verified HEAD state")
    return repo, reviewed_commit, head_commit


def test_bound_production_files_all_exist_at_head() -> None:
    """Every file this module binds review to must actually exist -- a
    typo'd path would silently make `verify_reviewed_implementation_
    binding` vacuously trivial for that file."""
    for path in provenance.BOUND_PRODUCTION_FILES:
        assert (Path.cwd() / path).is_file(), path


def test_verify_reviewed_implementation_binding_malformed_json_blocks(tmp_path: Path) -> None:
    repo, head = _repo_with_raw_file(tmp_path, "malformed", provenance.IMPLEMENTATION_REVIEW_GIT_PATH, b"{not valid json")
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(repo, head)
    assert error.value.reason == "V8E_PRODUCTION_IMPLEMENTATION_REVIEW_INVALID_JSON"


def test_verify_reviewed_implementation_binding_duplicate_key_blocks(tmp_path: Path) -> None:
    raw = (
        b'{"schema_version": "' + provenance.IMPLEMENTATION_REVIEW_SCHEMA_VERSION.encode() + b'", '
        b'"schema_version": "DUPLICATE", '
        b'"study": "V8E_HISTORICAL_RESEARCH", '
        b'"artifact_role": "PRODUCTION_IMPLEMENTATION_REVIEW", '
        b'"reviewed_implementation_git_commit": "' + b"a" * 40 + b'", '
        b'"review_result": "PASS", "approval_status": "APPROVED"}'
    )
    repo, head = _repo_with_raw_file(tmp_path, "dup_key", provenance.IMPLEMENTATION_REVIEW_GIT_PATH, raw)
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(repo, head)
    assert error.value.reason == "V8E_PRODUCTION_IMPLEMENTATION_REVIEW_DUPLICATE_KEY"


def test_verify_reviewed_implementation_binding_extra_field_blocks(tmp_path: Path) -> None:
    payload = json.loads(_valid_review_json("a" * 40))
    payload["unexpected_extra_field"] = "x"
    repo, head = _repo_with_raw_file(
        tmp_path, "extra_field", provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode(),
    )
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(repo, head)
    assert error.value.reason == "V8E_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_INVALID"


@pytest.mark.parametrize("missing_field", provenance.IMPLEMENTATION_REVIEW_FIELDS)
def test_verify_reviewed_implementation_binding_missing_field_blocks(tmp_path: Path, missing_field: str) -> None:
    payload = json.loads(_valid_review_json("a" * 40))
    del payload[missing_field]
    repo, head = _repo_with_raw_file(
        tmp_path, "missing_field_" + missing_field, provenance.IMPLEMENTATION_REVIEW_GIT_PATH,
        json.dumps(payload).encode(),
    )
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(repo, head)
    assert error.value.reason == "V8E_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_INVALID"


def test_verify_reviewed_implementation_binding_review_result_not_pass_blocks(tmp_path: Path) -> None:
    payload = json.loads(_valid_review_json("a" * 40))
    payload["review_result"] = "FAIL"
    repo, head = _repo_with_raw_file(
        tmp_path, "not_pass", provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode(),
    )
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(repo, head)
    assert error.value.reason == "V8E_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_PASS"


def test_verify_reviewed_implementation_binding_approval_status_not_approved_blocks(tmp_path: Path) -> None:
    payload = json.loads(_valid_review_json("a" * 40))
    payload["approval_status"] = "PENDING"
    repo, head = _repo_with_raw_file(
        tmp_path, "not_approved", provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode(),
    )
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(repo, head)
    assert error.value.reason == "V8E_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_APPROVED"


def test_verify_reviewed_implementation_binding_invalid_commit_blocks(tmp_path: Path) -> None:
    payload = json.loads(_valid_review_json("a" * 40))
    payload["reviewed_implementation_git_commit"] = "not-a-valid-sha"
    repo, head = _repo_with_raw_file(
        tmp_path, "invalid_commit", provenance.IMPLEMENTATION_REVIEW_GIT_PATH, json.dumps(payload).encode(),
    )
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(repo, head)
    assert error.value.reason == "V8E_PRODUCTION_IMPLEMENTATION_REVIEW_COMMIT_INVALID"


def test_verify_reviewed_implementation_binding_bound_file_drift_blocks(tmp_path: Path) -> None:
    mutated_path = provenance.BOUND_PRODUCTION_FILES[0]
    repo, _reviewed_commit, head_commit = _build_bound_file_repo(tmp_path, mutate_file=mutated_path)
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(repo, head_commit)
    assert error.value.reason == "V8E_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:" + mutated_path


def test_verify_reviewed_implementation_binding_passes_when_blobs_identical(tmp_path: Path) -> None:
    repo, reviewed_commit, head_commit = _build_bound_file_repo(tmp_path, mutate_file=None)
    result = provenance.verify_reviewed_implementation_binding(repo, head_commit)
    assert result["reviewed_implementation_git_commit"] == reviewed_commit
    assert result["verified_head"] == head_commit
    assert result["bound_files_verified"] == len(provenance.BOUND_PRODUCTION_FILES)


def test_arbitrary_sha_without_committed_review_artifact_cannot_substitute_synthetic_git(tmp_path: Path) -> None:
    """The API accepts no caller-supplied "reviewed implementation commit"
    parameter at all: only ``repository_root`` and ``verified_head``. Naming
    an arbitrary, syntactically valid 40-hex SHA inside a forged review
    artifact -- one that was never actually committed as that reviewed
    state -- still BLOCKs, because the bound files can't be resolved at a
    commit that does not exist in this repository's history."""
    import inspect

    signature = inspect.signature(provenance.verify_reviewed_implementation_binding)
    assert list(signature.parameters) == ["repository_root", "verified_head"]

    arbitrary_unrelated_sha = "b" * 40
    repo, reviewed_commit, head_commit = _build_bound_file_repo(
        tmp_path, mutate_file=None, reviewed_commit_override=arbitrary_unrelated_sha,
    )
    assert arbitrary_unrelated_sha != reviewed_commit
    with pytest.raises(provenance.V8EProductionProvenanceBlocked) as error:
        provenance.verify_reviewed_implementation_binding(repo, head_commit)
    assert error.value.reason.startswith("V8E_BOUND_FILE_MISSING_AT_REVIEWED_COMMIT:")


# ---------------------------------------------------------------------------
# Authority bridge: valid synthetic pass, tamper/missing/duplicate blocks,
# and T1C/T2 non-interchangeability (synthetic git only)
# ---------------------------------------------------------------------------


def _build_synthetic_authority_bridge_repo(
    tmp_path: Path, logical_block: str, *, bridge_overrides: dict | None = None,
    bridge_remove: list[str] | None = None, bridge_extra: dict | None = None,
    review_overrides: dict | None = None, include_bridge: bool = True, include_review: bool = True,
):
    repo = tmp_path / f"authority_{logical_block}"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    bridge_path = Path(authority_bridge.T1C_BRIDGE_PATH if logical_block == "T1C" else authority_bridge.T2_BRIDGE_PATH)
    review_path = Path(authority_bridge.T1C_REVIEW_PATH if logical_block == "T1C" else authority_bridge.T2_REVIEW_PATH)
    if include_bridge:
        bridge = _valid_t1c_bridge() if logical_block == "T1C" else _valid_t2_bridge()
        bridge.update(bridge_overrides or {})
        for key in bridge_remove or ():
            bridge.pop(key, None)
        bridge.update(bridge_extra or {})
        bridge_file = repo / bridge_path
        bridge_file.parent.mkdir(parents=True, exist_ok=True)
        bridge_file.write_text(json.dumps(bridge, separators=(",", ":")), encoding="utf-8")
    else:
        (repo / "README.md").write_text("no bridge", encoding="utf-8")
    reviewed_commit = _git_config_commit(repo, "synthetic reviewed bridge")
    bridge_blob = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", f"{reviewed_commit}:{bridge_path.as_posix()}"], text=True,
    ).strip() if include_bridge else "0" * 40
    if include_review:
        review = {
            "schema_version": authority_bridge.REVIEW_SCHEMA,
            "study": authority_bridge.STUDY,
            "artifact_role": authority_bridge.REVIEW_ROLE,
            "logical_block": logical_block,
            "reviewed_bridge_git_commit": reviewed_commit,
            "reviewed_bridge_git_blob_sha": bridge_blob,
            "review_result": "PASS",
        }
        review.update(review_overrides or {})
        review_file = repo / review_path
        review_file.parent.mkdir(parents=True, exist_ok=True)
        review_file.write_text(json.dumps(review, separators=(",", ":")), encoding="utf-8")
        verified_head = _git_config_commit(repo, "synthetic bridge review")
    else:
        verified_head = reviewed_commit
    return repo, verified_head, reviewed_commit, bridge_blob


@pytest.mark.parametrize("logical_block,stage", [("T1C", "T1C_TRANSPORT_READINESS"), ("T2", "T2_TRANSPORT_READINESS")])
def test_valid_stage_specific_authority_bridge_and_review_pass_synthetic_git(tmp_path: Path, logical_block: str, stage: str) -> None:
    repo, head, reviewed_commit, reviewed_blob = _build_synthetic_authority_bridge_repo(tmp_path, logical_block)
    result = authority_bridge.verify_stage_authority_bridge(repo, head, stage)
    assert result["logical_block"] == logical_block
    assert result["reviewed_bridge_git_commit"] == reviewed_commit
    assert result["reviewed_bridge_git_blob_sha"] == reviewed_blob


def test_t1c_and_t2_bridges_are_not_interchangeable_synthetic_git(tmp_path: Path) -> None:
    t1c_repo, t1c_head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(tmp_path, "T1C")
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked):
        authority_bridge.verify_stage_authority_bridge(t1c_repo, t1c_head, "T2_TRANSPORT_READINESS")

    t2_repo, t2_head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(tmp_path, "T2")
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked):
        authority_bridge.verify_stage_authority_bridge(t2_repo, t2_head, "T1C_TRANSPORT_READINESS")


@pytest.mark.parametrize("logical_block,stage", [("T1C", "T1C_TRANSPORT_READINESS"), ("T2", "T2_TRANSPORT_READINESS")])
def test_stage_specific_bridge_missing_blocks_synthetic_git(tmp_path: Path, logical_block: str, stage: str) -> None:
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(
        tmp_path, logical_block, include_bridge=False, include_review=False,
    )
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked):
        authority_bridge.verify_stage_authority_bridge(repo, head, stage)


@pytest.mark.parametrize("field", ["schema_version", "logical_block", "preservation_recheck_result", "authorization_status", "human_gate"])
def test_t1c_bridge_frozen_binding_tamper_blocks_synthetic_git(tmp_path: Path, field: str) -> None:
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(
        tmp_path, "T1C", bridge_overrides={field: "tampered"},
    )
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked):
        authority_bridge.verify_stage_authority_bridge(repo, head, "T1C_TRANSPORT_READINESS")


@pytest.mark.parametrize("field", ["source_authority", "v8_trust_anchor_git_identity", "preservation_recheck_git_commit", "preservation_recheck_git_blob_sha", "human_gate"])
def test_t2_bridge_frozen_binding_tamper_blocks_synthetic_git(tmp_path: Path, field: str) -> None:
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(
        tmp_path, "T2", bridge_overrides={field: "tampered"},
    )
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked):
        authority_bridge.verify_stage_authority_bridge(repo, head, "T2_TRANSPORT_READINESS")


@pytest.mark.parametrize("override", [
    {"review_result": "BLOCK"},
    {"logical_block": "T2"},
    {"reviewed_bridge_git_commit": "f" * 40},
    {"reviewed_bridge_git_blob_sha": "f" * 40},
])
def test_independent_bridge_review_mismatch_blocks_synthetic_git(tmp_path: Path, override: dict) -> None:
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(
        tmp_path, "T1C", review_overrides=override,
    )
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked):
        authority_bridge.verify_stage_authority_bridge(repo, head, "T1C_TRANSPORT_READINESS")


@pytest.mark.parametrize("variant", ["extra", "duplicate"])
def test_independent_bridge_review_duplicate_and_extra_fields_block_synthetic_git(tmp_path: Path, variant: str) -> None:
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(tmp_path, "T1C")
    review_path = repo / authority_bridge.T1C_REVIEW_PATH
    if variant == "extra":
        raw = review_path.read_text(encoding="utf-8")
        review_path.write_text(raw[:-1] + ',"extra":true}', encoding="utf-8")
    else:
        review_path.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    new_head = _git_config_commit(repo, "extra review field")
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked):
        authority_bridge.verify_stage_authority_bridge(repo, new_head, "T1C_TRANSPORT_READINESS")


@pytest.mark.parametrize("kwargs", [
    {"bridge_remove": ["human_gate"]},
    {"bridge_extra": {"unexpected": True}},
])
def test_authority_bridge_exact_field_set_blocks_missing_or_extra_synthetic_git(tmp_path: Path, kwargs: dict) -> None:
    repo, head, _reviewed, _blob = _build_synthetic_authority_bridge_repo(tmp_path, "T2", **kwargs)
    with pytest.raises(authority_bridge.V8EAuthorityBridgeBlocked):
        authority_bridge.verify_stage_authority_bridge(repo, head, "T2_TRANSPORT_READINESS")


# ---------------------------------------------------------------------------
# T2 point-of-use preservation (synthetic/dependency-injected safe evidence)
# ---------------------------------------------------------------------------


def _t2_point_of_use_dependencies(tmp_path: Path, **overrides):
    repo = tmp_path / "synthetic-repo"
    safe_conditions = {
        "T2_real_data_acquired": False,
        "T2_opened": False,
        "T2_research_access_count": 0,
        "T2_features_observed": False,
        "T2_outcomes_observed": False,
        "T2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
    }
    anchor = {
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": provenance.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "authorized_partition_implementation_git_commit": provenance.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    }
    deps = {
        "git_commit_resolver": lambda: "b" * 40,
        "frozen_design_verifier": lambda _root: None,
        "freeze_approval_verifier": lambda _root, _head: provenance.EXPECTED_V8E_DESIGN_FREEZE_APPROVAL_BLOB,
        "reviewed_implementation_binder": lambda _root, _head: {
            "reviewed_implementation_git_commit": IMPLEMENTATION_SHA,
        },
        "anchor_reader": lambda _root, _head: anchor,
        "authority_bridge_verifier": lambda _root, _head, _stage: {
            "logical_block": "T2",
            "review_result": "PASS",
        },
        "readiness_reader": lambda: {
            "verification_stage": point_of_use.READINESS_VERIFICATION_STAGE,
            "logical_stage": point_of_use.READINESS_LOGICAL_STAGE,
            "verification_result": "PASS",
            "frozen_design_commit": point_of_use.FROZEN_DESIGN_COMMIT,
            "receipt_self_hash": "a" * 64,
        },
        "gate_consumption_checker": lambda _root, _gate, _design: False,
        "consumption_state_root": tmp_path / "synthetic-gate-state",
        "state_conditions_reader": lambda _root, _head: safe_conditions,
        "prefreeze_blob_resolver": lambda _root, _commit, _path: point_of_use.PREFREEZE_PRESERVATION_BLOB,
        "prefreeze_ancestor_checker": lambda _root, _ancestor, _descendant, _reason: None,
    }
    deps.update(overrides)
    return repo, deps


def _derive_synthetic_t2_point_of_use(tmp_path: Path, **overrides):
    repo, deps = _t2_point_of_use_dependencies(tmp_path, **overrides)
    return point_of_use._derive_t2_point_of_use_preservation_with_dependencies(repo, **deps)


def _review_document(*, commit: str = "c" * 40, blob: str = "d" * 40) -> dict:
    return {
        "schema_version": point_of_use.POINT_OF_USE_REVIEW_SCHEMA_VERSION,
        "study": point_of_use.STUDY,
        "artifact_role": point_of_use.POINT_OF_USE_REVIEW_ROLE,
        "checkpoint": point_of_use.POINT_OF_USE_REVIEW_CHECKPOINT,
        "v8e_frozen_design_commit": point_of_use.FROZEN_DESIGN_COMMIT,
        "reviewed_recheck_git_commit": commit,
        "reviewed_recheck_git_blob_sha": blob,
        "review_result": "PASS",
    }


def _synthetic_review_pass(tmp_path: Path, **dependency_overrides):
    repo, deps = _t2_point_of_use_dependencies(tmp_path, **dependency_overrides)
    _head, artifact = point_of_use._derive_t2_point_of_use_preservation_with_dependencies(repo, **deps)
    artifact_bytes = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    review = _review_document()
    review_deps = {
        "dependencies": deps,
        "review_reader": lambda _root, _head: review,
        "artifact_blob_resolver": lambda _root, _commit, _path: review["reviewed_recheck_git_blob_sha"],
        "artifact_reader": lambda _root, _commit, _path: artifact_bytes,
        "ancestor_checker": lambda _root, _ancestor, _descendant, _reason: None,
    }
    return repo, review_deps, artifact, artifact_bytes


def test_t2_point_of_use_valid_synthetic_safe_evidence_has_exact_contract(tmp_path: Path) -> None:
    _head, artifact = _derive_synthetic_t2_point_of_use(tmp_path)
    assert set(artifact) == set(point_of_use.POINT_OF_USE_ARTIFACT_FIELDS)
    assert artifact["t2_count"] == 300
    assert artifact["t2_ticker_list_sha256"] == point_of_use.T2_TICKER_LIST_SHA256
    assert artifact["point_of_use_preservation_result"] == "PASS"
    assert not (tmp_path / "synthetic-repo" / point_of_use.POINT_OF_USE_ARTIFACT_PATH).exists()


def test_t2_point_of_use_does_not_read_private_manifest_or_access_network(tmp_path: Path) -> None:
    accesses: list[str] = []

    def safe_state(_root, _head):
        accesses.append("V8_STATE.json")
        return {
            "T2_real_data_acquired": False,
            "T2_opened": False,
            "T2_research_access_count": 0,
            "T2_features_observed": False,
            "T2_outcomes_observed": False,
            "T2_membership_reassigned": False,
            "universe_definition_compatible": True,
            "partition_algorithm_compatible": True,
            "data_quality_policy_unchanged": True,
        }

    _derive_synthetic_t2_point_of_use(tmp_path, state_conditions_reader=safe_state)
    assert accesses == ["V8_STATE.json"]


def test_t2_point_of_use_missing_readiness_blocks(tmp_path: Path) -> None:
    def blocked():
        raise point_of_use.V8ET2PointOfUsePreservationBlocked("missing readiness")

    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        _derive_synthetic_t2_point_of_use(tmp_path, readiness_reader=blocked)


@pytest.mark.parametrize("bridge", [
    {"logical_block": "T1C", "review_result": "PASS"},
    {"logical_block": "T2", "review_result": "BLOCK"},
])
def test_t2_point_of_use_wrong_authority_bridge_or_review_blocks(tmp_path: Path, bridge: dict) -> None:
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        _derive_synthetic_t2_point_of_use(tmp_path, authority_bridge_verifier=lambda *_args: bridge)


def test_t2_point_of_use_wrong_prefreeze_review_result_or_design_blocks(tmp_path: Path) -> None:
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        _derive_synthetic_t2_point_of_use(tmp_path, freeze_approval_verifier=lambda *_args: "0" * 40)
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        _derive_synthetic_t2_point_of_use(
            tmp_path, frozen_design_verifier=lambda _root: (_ for _ in ()).throw(
                provenance.V8EProductionProvenanceBlocked("wrong design")
            ),
        )


@pytest.mark.parametrize("field,value", [
    ("T2_real_data_acquired", True),
    ("T2_opened", True),
    ("T2_research_access_count", 1),
    ("T2_features_observed", True),
    ("T2_outcomes_observed", True),
    ("T2_membership_reassigned", True),
    ("universe_definition_compatible", False),
    ("partition_algorithm_compatible", False),
    ("data_quality_policy_unchanged", False),
])
def test_t2_point_of_use_any_frozen_condition_wrong_blocks(tmp_path: Path, field: str, value: object) -> None:
    conditions = {
        "T2_real_data_acquired": False,
        "T2_opened": False,
        "T2_research_access_count": 0,
        "T2_features_observed": False,
        "T2_outcomes_observed": False,
        "T2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
    }
    conditions[field] = value
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        _derive_synthetic_t2_point_of_use(tmp_path, state_conditions_reader=lambda *_args: conditions)


def test_t2_point_of_use_gate_already_consumed_blocks(tmp_path: Path) -> None:
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        _derive_synthetic_t2_point_of_use(tmp_path, gate_consumption_checker=lambda *_args: True)


@pytest.mark.parametrize("wrong_anchor_field,wrong_value", [
    ("authorization_status", "BLOCK"),
    ("authorized_partition_manifest_sha256", "0" * 64),
    ("authorized_partition_implementation_git_commit", "0" * 40),
])
def test_t2_point_of_use_wrong_anchor_and_reviewed_implementation_block(
    tmp_path: Path, wrong_anchor_field: str, wrong_value: str,
) -> None:
    wrong_anchor = {
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": provenance.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "authorized_partition_implementation_git_commit": provenance.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    }
    wrong_anchor[wrong_anchor_field] = wrong_value
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        _derive_synthetic_t2_point_of_use(tmp_path, anchor_reader=lambda *_args: wrong_anchor)
    if wrong_anchor_field != "authorization_status":
        return
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        _derive_synthetic_t2_point_of_use(
            tmp_path,
            reviewed_implementation_binder=lambda *_args: {"reviewed_implementation_git_commit": "not-a-commit"},
        )


def test_t2_point_of_use_t1c_evidence_cannot_satisfy_t2(tmp_path: Path) -> None:
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        _derive_synthetic_t2_point_of_use(
            tmp_path,
            readiness_reader=lambda: {
                "verification_stage": "READ_ONLY_T1C_READINESS_TRANSPORT_AUDIT_VERIFICATION",
                "logical_stage": "T1C_TRANSPORT_READINESS",
                "verification_result": "PASS",
                "frozen_design_commit": point_of_use.FROZEN_DESIGN_COMMIT,
                "receipt_self_hash": "a" * 64,
            },
        )


def test_t2_point_of_use_public_signatures_have_no_override_parameters() -> None:
    import inspect

    assert not inspect.signature(point_of_use.resolve_and_recheck_t2_point_of_use_preservation).parameters
    assert not inspect.signature(point_of_use.derive_t2_point_of_use_preservation_artifact).parameters
    assert not inspect.signature(point_of_use.require_t2_point_of_use_preservation_review_pass).parameters


@pytest.mark.parametrize("mutator", [
    lambda artifact: artifact.pop("t2_count"),
    lambda artifact: artifact.update({"extra": True}),
])
def test_t2_point_of_use_artifact_schema_is_strict(tmp_path: Path, mutator) -> None:
    _head, artifact = _derive_synthetic_t2_point_of_use(tmp_path)
    mutator(artifact)
    raw = json.dumps(artifact, sort_keys=True).encode("utf-8")
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._validate_preservation_artifact(raw)


def test_t2_point_of_use_artifact_duplicate_key_blocks(tmp_path: Path) -> None:
    _head, artifact = _derive_synthetic_t2_point_of_use(tmp_path)
    first = json.dumps({"schema_version": artifact["schema_version"]})[1:-1]
    duplicate = first + "," + first + "," + json.dumps(
        {key: value for key, value in artifact.items() if key != "schema_version"}
    )[1:-1]
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._validate_preservation_artifact(("{" + duplicate + "}").encode("utf-8"))


@pytest.mark.parametrize("field,value", [("t2_count", 299), ("t2_ticker_list_sha256", "0" * 64)])
def test_t2_point_of_use_wrong_t2_count_or_list_hash_blocks(tmp_path: Path, field: str, value: object) -> None:
    _head, artifact = _derive_synthetic_t2_point_of_use(tmp_path)
    artifact[field] = value
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._validate_preservation_artifact(json.dumps(artifact).encode("utf-8"))


def test_t2_point_of_use_review_pass_resolves_exact_artifact_commit_and_blob(tmp_path: Path) -> None:
    repo, review_deps, artifact, _artifact_bytes = _synthetic_review_pass(tmp_path)
    result = point_of_use._require_t2_point_of_use_preservation_review_pass_with_dependencies(repo, **review_deps)
    assert result["review_result"] == "PASS"
    assert result["preservation_artifact"] == artifact
    assert result["reviewed_recheck_git_commit"] == "c" * 40
    assert result["reviewed_recheck_git_blob_sha"] == "d" * 40


def test_t2_point_of_use_missing_review_blocks(tmp_path: Path) -> None:
    repo, deps = _t2_point_of_use_dependencies(tmp_path)
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._require_t2_point_of_use_preservation_review_pass_with_dependencies(
            repo,
            dependencies=deps,
            review_reader=lambda *_args: (_ for _ in ()).throw(
                point_of_use.V8ET2PointOfUsePreservationBlocked("missing review")
            ),
            artifact_blob_resolver=lambda *_args: "d" * 40,
            artifact_reader=lambda *_args: b"{}",
            ancestor_checker=lambda *_args: None,
        )


@pytest.mark.parametrize("review_mutation", [
    lambda review: review.pop("review_result"),
    lambda review: review.update({"extra": True}),
    lambda review: review.update({"review_result": "BLOCK"}),
])
def test_t2_point_of_use_review_schema_and_pass_are_strict(tmp_path: Path, review_mutation, monkeypatch: pytest.MonkeyPatch) -> None:
    _repo, _deps = _t2_point_of_use_dependencies(tmp_path)
    review = _review_document()
    review_mutation(review)
    monkeypatch.setattr(
        point_of_use, "read_git_object_bytes",
        lambda *_args: json.dumps(review).encode("utf-8"),
    )
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._read_review_artifact(_repo, "b" * 40)


def test_t2_point_of_use_review_duplicate_key_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    review = _review_document()
    raw = json.dumps(review, sort_keys=True)[1:-1]
    duplicate = raw + ',"review_result":"PASS"'
    monkeypatch.setattr(point_of_use, "read_git_object_bytes", lambda *_args: ("{" + duplicate + "}").encode())
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._read_review_artifact(tmp_path / "repo", "b" * 40)


def test_t2_point_of_use_wrong_review_blob_or_tampered_artifact_blocks(tmp_path: Path) -> None:
    repo, deps = _t2_point_of_use_dependencies(tmp_path)
    _head, artifact = point_of_use._derive_t2_point_of_use_preservation_with_dependencies(repo, **deps)
    artifact_bytes = json.dumps(artifact).encode("utf-8")
    review = _review_document(blob="e" * 40)
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._require_t2_point_of_use_preservation_review_pass_with_dependencies(
            repo, dependencies=deps, review_reader=lambda *_args: review,
            artifact_blob_resolver=lambda *_args: "d" * 40,
            artifact_reader=lambda *_args: artifact_bytes,
            ancestor_checker=lambda *_args: None,
        )
    review = _review_document()
    tampered = dict(artifact)
    tampered["t2_count"] = 299
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._require_t2_point_of_use_preservation_review_pass_with_dependencies(
            repo, dependencies=deps, review_reader=lambda *_args: review,
            artifact_blob_resolver=lambda *_args: "d" * 40,
            artifact_reader=lambda *_args: json.dumps(tampered).encode("utf-8"),
            ancestor_checker=lambda *_args: None,
        )


def test_t2_point_of_use_wrong_reviewed_commit_blocks(tmp_path: Path) -> None:
    repo, review_deps, _artifact, _bytes = _synthetic_review_pass(tmp_path)

    def wrong_ancestor(_root, _ancestor, _descendant, _reason):
        raise git_provenance.V8EGitProvenanceBlocked("wrong reviewed commit")

    review_deps["ancestor_checker"] = wrong_ancestor
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._require_t2_point_of_use_preservation_review_pass_with_dependencies(repo, **review_deps)


def test_t2_point_of_use_review_blocks_after_acquisition_gate_consumption(tmp_path: Path) -> None:
    calls = {"count": 0}

    def consumed_after_first(_root, _gate, _design):
        calls["count"] += 1
        return calls["count"] > 1

    repo, review_deps, _artifact, _bytes = _synthetic_review_pass(tmp_path)
    review_deps["dependencies"]["gate_consumption_checker"] = consumed_after_first
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._require_t2_point_of_use_preservation_review_pass_with_dependencies(repo, **review_deps)


# ---------------------------------------------------------------------------
# T2 point-of-use: derivation from the real committed public V8_STATE.json
# (safe/public repository data only -- no private/sealed reads)
# ---------------------------------------------------------------------------


def _actual_v8_state_for_t2_test() -> tuple[str, dict]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    raw = git_provenance.read_git_object_bytes(Path.cwd(), head, "V8_STATE.json")
    return head, json.loads(raw.decode("utf-8"))


def _state_reader_from_mapping(state: dict):
    raw = json.dumps(state, separators=(",", ":")).encode("utf-8")
    return lambda _root, _head, _path: raw


def test_t2_point_of_use_derives_all_conditions_from_real_v8_state_schema() -> None:
    head, _state = _actual_v8_state_for_t2_test()
    conditions = point_of_use._derive_conditions_from_v8_state(
        Path.cwd(), head, git_provenance.read_git_object_bytes,
    )
    assert conditions == {
        "T2_real_data_acquired": False,
        "T2_opened": False,
        "T2_research_access_count": 0,
        "T2_features_observed": False,
        "T2_outcomes_observed": False,
        "T2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
    }


@pytest.mark.parametrize("mutation", [
    lambda state: state["real_partition_build_history"][0].update({"block_assignments_exposed": True}),
    lambda state: state["trust_anchor_pinning"].update({"block_assignments_exposed": True}),
    lambda state: state["real_partition_build_history"][0]["block_sizes"].update({"T2": 299}),
    lambda state: state["real_partition_build_history"][0]["block_sizes"].pop("T2"),
    lambda state: state["real_partition_build_history"][0].update({"t2_ticker_list_sha256": "0" * 64}),
    lambda state: state["real_partition_build_history"][0].pop("t2_ticker_list_sha256"),
    lambda state: state["real_partition_build_history"][0].update({"manifest_sha256": "0" * 64}),
    lambda state: state["real_partition_build_history"][0].pop("manifest_sha256"),
    lambda state: state["real_partition_build_history"][0].update({"partition_implementation_git_commit": "0" * 40}),
    lambda state: state["real_partition_build_history"][0].pop("partition_implementation_git_commit"),
])
def test_t2_point_of_use_partition_and_anchor_mismatch_blocks(mutation) -> None:
    head, state = _actual_v8_state_for_t2_test()
    mutation(state)
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._derive_conditions_from_v8_state(Path.cwd(), head, _state_reader_from_mapping(state))


@pytest.mark.parametrize("mutation", [
    lambda state: state["T2"].update({"raw_data_acquired": True}),
    lambda state: state["T2"].update({"opened_for_research": True}),
    lambda state: state["T2"].update({"sealed_holdout_access_count": 1}),
    lambda state: state.update({"backtests": 1}),
    lambda state: state.update({"models_fitted": 1}),
    lambda state: state.update({"profit_calculated": 1}),
    lambda state: state.update({"parameter_search": 1}),
    lambda state: state["T2"].update({"real_acquisition_authorized": True}),
])
def test_t2_point_of_use_positive_acquisition_or_research_state_blocks(mutation) -> None:
    head, state = _actual_v8_state_for_t2_test()
    mutation(state)
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._derive_conditions_from_v8_state(Path.cwd(), head, _state_reader_from_mapping(state))


@pytest.mark.parametrize("mutation", [
    lambda state: state.pop("real_partition_build_history"),
    lambda state: state.update({"real_partition_build_history": []}),
    lambda state: state["partition"].update({"block_assignments_recorded": True}),
    lambda state: state["malformed_ohlcv_policy_clarification"].update({
        "existing_partition_manifest_identity_unchanged": False,
    }),
])
def test_t2_point_of_use_contradictory_or_ambiguous_safe_state_blocks(mutation) -> None:
    head, state = _actual_v8_state_for_t2_test()
    mutation(state)
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._derive_conditions_from_v8_state(Path.cwd(), head, _state_reader_from_mapping(state))


def test_t2_point_of_use_duplicate_partition_history_key_blocks() -> None:
    head, state = _actual_v8_state_for_t2_test()
    raw = json.dumps(state, separators=(",", ":"))[:-1]
    raw += ',"real_partition_build_history":{}}'
    with pytest.raises(point_of_use.V8ET2PointOfUsePreservationBlocked):
        point_of_use._derive_conditions_from_v8_state(Path.cwd(), head, lambda *_args: raw.encode("utf-8"))
