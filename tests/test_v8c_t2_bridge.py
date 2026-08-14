from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8c_human_gate_consumption as gate_consumption
from src import v8c_t2_bridge as bridge
from src.v8c_production_provenance import (
    EXPECTED_T2_TICKER_COUNT,
    EXPECTED_T2_TICKER_LIST_SHA256,
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
)

SYNTHETIC_COMMIT = "b" * 40


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def _valid_kwargs(**overrides):
    kwargs = dict(
        v8_trust_anchor_git_identity=EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        authorized_parent_v8_partition_manifest_sha256=EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        reviewed_production_implementation_commit=SYNTHETIC_COMMIT,
        exact_human_bridge_authorization_identity=bridge.expected_human_gate(EXPECTED_V8C_FROZEN_DESIGN_COMMIT),
        authorization_note="note",
    )
    kwargs.update(overrides)
    return kwargs


def test_bridge_is_a_separate_git_path_from_v8b_bridge():
    assert bridge.BRIDGE_GIT_PATH == "V8C_T2_AUTHORITY_BRIDGE.json"
    assert bridge.BRIDGE_GIT_PATH != "V8B_T2_AUTHORITY_BRIDGE.json"


def test_build_bridge_pins_exact_authority_constants():
    built = bridge.build_v8c_t2_authority_bridge(**_valid_kwargs())
    assert built["v8_trust_anchor_git_identity"] == EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA
    assert built["authorized_parent_v8_partition_manifest_sha256"] == EXPECTED_V8_PARTITION_MANIFEST_SHA256
    assert built["expected_t2_ticker_count"] == EXPECTED_T2_TICKER_COUNT
    assert built["expected_t2_ticker_list_sha256"] == EXPECTED_T2_TICKER_LIST_SHA256
    assert built["study"] == "V8C_HISTORICAL_RESEARCH"
    assert built["exact_frozen_v8c_design_commit"] == EXPECTED_V8C_FROZEN_DESIGN_COMMIT


def test_build_bridge_rejects_fake_anchor_identity():
    with pytest.raises(bridge.V8CT2BridgeBlocked) as excinfo:
        bridge.build_v8c_t2_authority_bridge(**_valid_kwargs(v8_trust_anchor_git_identity="f" * 40))
    assert excinfo.value.reason == "V8_TRUST_ANCHOR_GIT_IDENTITY_MISMATCH"


def test_build_bridge_rejects_fake_manifest_sha():
    with pytest.raises(bridge.V8CT2BridgeBlocked) as excinfo:
        bridge.build_v8c_t2_authority_bridge(**_valid_kwargs(authorized_parent_v8_partition_manifest_sha256="0" * 64))
    assert excinfo.value.reason == "PARENT_V8_PARTITION_MANIFEST_SHA_MISMATCH"


def test_build_bridge_rejects_wrong_human_gate():
    with pytest.raises(bridge.V8CT2BridgeBlocked) as excinfo:
        bridge.build_v8c_t2_authority_bridge(**_valid_kwargs(exact_human_bridge_authorization_identity="not-the-real-gate"))
    assert excinfo.value.reason == "HUMAN_BRIDGE_AUTHORIZATION_IDENTITY_MISMATCH"


def test_validate_bridge_accepts_well_formed_bridge():
    built = bridge.build_v8c_t2_authority_bridge(**_valid_kwargs())
    validated = bridge.validate_v8c_t2_authority_bridge(built)
    assert validated == built


def test_validate_bridge_rejects_arbitrary_caller_crafted_mapping():
    """A fake authority mapping -- correct-looking but not derived from
    this module's own frozen constants -- must be rejected."""
    fake = {field: "FAKE" for field in bridge.BRIDGE_FIELDS}
    fake["schema_version"] = bridge.SCHEMA_VERSION
    fake["study"] = bridge.STUDY_NAME
    fake["role"] = bridge.ROLE
    fake["exact_frozen_v8c_design_commit"] = EXPECTED_V8C_FROZEN_DESIGN_COMMIT
    fake["source_authority"] = bridge.SOURCE_AUTHORITY
    fake["v8_trust_anchor_git_identity"] = "f" * 40  # fabricated, wrong
    fake["authorized_parent_v8_partition_manifest_sha256"] = "0" * 64
    fake["expected_t2_ticker_count"] = EXPECTED_T2_TICKER_COUNT
    fake["expected_t2_ticker_list_sha256"] = EXPECTED_T2_TICKER_LIST_SHA256
    fake["t2_membership_reassignment"] = "PROHIBITED"
    fake["v8_trusted_partition_json_mutated_or_repinned"] = False
    fake["t2_acquired_before_authorized_v8c_acquisition"] = False
    fake["t2_research_open_count_before_official_opening"] = 0
    fake["reviewed_production_implementation_commit"] = SYNTHETIC_COMMIT
    fake["exact_human_bridge_authorization_identity"] = bridge.expected_human_gate(EXPECTED_V8C_FROZEN_DESIGN_COMMIT)
    fake["authorization_note"] = "n"
    with pytest.raises(bridge.V8CT2BridgeBlocked) as excinfo:
        bridge.validate_v8c_t2_authority_bridge(fake)
    assert excinfo.value.reason == "BRIDGE_ANCHOR_IDENTITY_MISMATCH"


def test_v8b_bridge_content_rejected_as_v8c_authority():
    """The real, on-disk V8B_T2_AUTHORITY_BRIDGE.json artifact must never
    validate as V8C authority -- its study/schema differ."""
    v8b_bridge_path = Path(__file__).resolve().parents[1] / "V8B_T2_AUTHORITY_BRIDGE.json"
    v8b_bridge = json.loads(v8b_bridge_path.read_bytes())
    # Structurally it doesn't even have this module's required field set.
    with pytest.raises(bridge.V8CT2BridgeBlocked) as excinfo:
        bridge.validate_v8c_t2_authority_bridge(v8b_bridge)
    assert excinfo.value.reason == "BRIDGE_SCHEMA_INVALID"


def test_read_and_verify_fails_closed_when_bridge_file_absent(tmp_path):
    """No V8C_T2_AUTHORITY_BRIDGE.json exists in this repository -- the
    production reader must fail closed reading it from real Git HEAD."""
    from src.v8c_git_provenance import V8CGitProvenanceBlocked

    with pytest.raises((bridge.V8CT2BridgeBlocked, V8CGitProvenanceBlocked)):
        bridge.read_and_verify_v8c_t2_authority_bridge("f" * 40, "f" * 40)


# ---------------------------------------------------------------------------
# HIGH-2: at bridge-creation point, revalidate against CURRENT safe Git/
# trust/gate state (a fresh live recheck) before consuming
# HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE -- a stored durable PASS record alone
# is not sufficient production authority.
# ---------------------------------------------------------------------------


def _bridge_creation_base_deps(**overrides):
    deps = dict(
        confirmation=bridge.BRIDGE_CONFIRMATION,
        human_bridge_authorization=bridge.expected_human_gate(EXPECTED_V8C_FROZEN_DESIGN_COMMIT),
        output_path=Path(tempfile.gettempdir()) / ("v8c_bridge_out-" + uuid.uuid4().hex + ".json"),
        authorization_note="note",
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_COMMIT},
        anchor_reader=lambda head: {
            "authorization_status": "AUTHORIZED",
            "authorized_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        },
        clock=clock_stub,
        consumption_state_root=Path(tempfile.gettempdir()) / ("v8c_bridge_gate-" + uuid.uuid4().hex),
    )
    deps.update(overrides)
    return deps


def test_bridge_creation_blocks_when_live_recheck_disagrees_with_durable_record():
    deps = _bridge_creation_base_deps(
        t2_preservation_pass_reader=lambda commit: {"result": "PASS", "recheck_point": "recheck_2"},
        t2_preservation_live_resolver=lambda: {"result": "BLOCK", "recheck_point": "recheck_2"},
    )
    state_root = deps["consumption_state_root"]
    with pytest.raises(bridge.V8CT2BridgeBlocked) as excinfo:
        bridge._create_v8c_t2_authority_bridge_production_with_dependencies(**deps)
    assert excinfo.value.reason == "V8C_T2_PRESERVATION_RECHECK_PASS_REQUIRED"
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T2_AUTHORITY_BRIDGE, EXPECTED_V8C_FROZEN_DESIGN_COMMIT
    ) is False


def test_bridge_creation_blocks_when_live_recheck_raises():
    def failing_live_resolver():
        raise RuntimeError("live recheck dependency unavailable")

    deps = _bridge_creation_base_deps(
        t2_preservation_pass_reader=lambda commit: {"result": "PASS", "recheck_point": "recheck_2"},
        t2_preservation_live_resolver=failing_live_resolver,
    )
    state_root = deps["consumption_state_root"]
    with pytest.raises(bridge.V8CT2BridgeBlocked) as excinfo:
        bridge._create_v8c_t2_authority_bridge_production_with_dependencies(**deps)
    assert excinfo.value.reason == "V8C_T2_PRESERVATION_RECHECK_PASS_REQUIRED"
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T2_AUTHORITY_BRIDGE, EXPECTED_V8C_FROZEN_DESIGN_COMMIT
    ) is False


def test_bridge_creation_succeeds_when_both_durable_record_and_live_recheck_pass():
    deps = _bridge_creation_base_deps(
        t2_preservation_pass_reader=lambda commit: {"result": "PASS", "recheck_point": "recheck_2"},
        t2_preservation_live_resolver=lambda: {"result": "PASS", "recheck_point": "recheck_2"},
    )
    result = bridge._create_v8c_t2_authority_bridge_production_with_dependencies(**deps)
    assert result["reviewed_production_implementation_commit"] == SYNTHETIC_COMMIT
