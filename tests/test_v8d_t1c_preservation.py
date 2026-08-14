from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8_partition
from src import v8c_t1c_allocation as allocation
from src import v8d_t1c_preservation as preservation


SYNTHETIC_DESIGN = "2" * 40
SYNTHETIC_ALLOCATION_HASH = "3" * 64
SYNTHETIC_AUTHORIZATION = (
    preservation.V8D_AUTHORIZATION_PREFIX
    + SYNTHETIC_DESIGN
    + preservation.V8D_AUTHORIZATION_SEPARATOR
    + SYNTHETIC_ALLOCATION_HASH
)
SYNTHETIC_HEAD = "1" * 40
SYNTHETIC_IMPL = "b" * 40
SYNTHETIC_MANIFEST_SHA = "c" * 64
SYNTHETIC_V8C_DESIGN = "a" * 40


def _clock():
    return datetime(2026, 8, 15, tzinfo=timezone.utc)


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def _private_fixture():
    parent = _tickers("SPARE", 1904)
    blocks = {
        "T0": _tickers("T0", 300),
        "T1": _tickers("T1", 300),
        "T2": _tickers("T2", 300),
        "T3": _tickers("T3", 300),
        "T_spare": parent,
    }
    manifest = {
        "schema_version": v8_partition.SCHEMA_VERSION,
        "study_name": v8_partition.STUDY_NAME,
        "design_commit": v8_partition.DESIGN_COMMIT,
        "source_snapshot_semantics": v8_partition.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": v8_partition.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": SYNTHETIC_IMPL,
        "created_utc": "2026-08-15T00:00:00Z",
        "source_url": "https://example.invalid/source",
        "source_host": "example.invalid",
        "source_acquisition_utc": "2026-08-15T00:00:00Z",
        "source_raw_sha256": "d" * 64,
        "source_raw_byte_count": 1,
        "v4_source_raw_sha256_reference": "e" * 64,
        "v4_raw_sha_equality_required": False,
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "eligible_ticker_count": 3404,
        "eligible_ticker_list_sha256": "f" * 64,
        "selection_rule": "synthetic",
        "deterministic_ordering_rule": v8_partition.DETERMINISTIC_ORDERING_RULE,
        "t0_ticker_list_sha256": v8_partition.ticker_list_sha256(blocks["T0"]),
        "t1_ticker_list_sha256": v8_partition.ticker_list_sha256(blocks["T1"]),
        "t2_ticker_list_sha256": v8_partition.ticker_list_sha256(blocks["T2"]),
        "t3_ticker_list_sha256": v8_partition.ticker_list_sha256(blocks["T3"]),
        "t_spare_ticker_list_sha256": v8_partition.ticker_list_sha256(parent),
        "legacy_exclude_list": [],
        "legacy_exclude_list_sha256": v8_partition.ticker_list_sha256([]),
        "block_sizes": {key: len(value) for key, value in blocks.items()},
        "block_assignments": blocks,
        "p_hist_start": v8_partition.P_HIST_START,
        "p_hist_end": v8_partition.P_HIST_END,
        "t1_role": v8_partition.T1_ROLE,
        "t2_role": v8_partition.T2_ROLE,
        "t3_role": v8_partition.T3_ROLE,
        "t3_price_acquisition_authorized": False,
    }
    manifest["manifest_sha256"] = v8_partition.canonical_sha256(manifest)
    artifact = allocation.build_t1c_allocation_artifact(
        parent,
        parent_v8_partition_manifest_sha256=manifest["manifest_sha256"],
        parent_v8_partition_implementation_commit=SYNTHETIC_IMPL,
        parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        v8c_frozen_design_commit=SYNTHETIC_V8C_DESIGN,
        v8c_allocation_implementation_commit=SYNTHETIC_IMPL,
        clock=_clock,
    )
    return artifact, manifest


def _public_preflight():
    return {
        "repository_identity": "ta1k1-arakawa/stock-analyzer",
        "branch": preservation.V8D_PRODUCTION_BRANCH,
        "head": SYNTHETIC_HEAD,
        "origin_head": SYNTHETIC_HEAD,
        "worktree_clean": True,
        "frozen_design_commit": preservation.V8D_FROZEN_DESIGN_COMMIT,
        "frozen_design_blob_sha": preservation.V8D_FROZEN_DESIGN_BLOB_SHA,
        "v8c_terminal_commit": preservation.V8C_TERMINAL_COMMIT,
        "v8c_terminal_blob_sha": preservation.V8C_TERMINAL_ADJUDICATION_BLOB_SHA,
        "v8c_prefreeze_blob_sha": preservation.V8C_PREFREEZE_AUDIT_BLOB_SHA,
        "trusted_partition_blob_sha": preservation.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "partition_manifest_sha256": preservation.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "partition_implementation_commit": preservation.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "parent_t_spare_ticker_count": preservation.EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
        "parent_t_spare_ticker_list_sha256": preservation.EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
        "reviewed_implementation_commit": SYNTHETIC_IMPL,
        "v8c_frozen_design_commit": SYNTHETIC_V8C_DESIGN,
    }


def _synthetic_receipt_key():
    return preservation.compute_receipt_key(
        SYNTHETIC_AUTHORIZATION,
        SYNTHETIC_DESIGN,
        SYNTHETIC_ALLOCATION_HASH,
    )


def _consume_synthetic(state_root):
    return preservation.consume_gate_once(
        state_root,
        SYNTHETIC_AUTHORIZATION,
        clock=_clock,
        reviewed_design_candidate_commit=SYNTHETIC_DESIGN,
        authorized_allocation_artifact_self_hash=SYNTHETIC_ALLOCATION_HASH,
    )


def test_synthetic_authorization_grammar_passes():
    preservation.validate_authorization_identity(
        SYNTHETIC_AUTHORIZATION,
        SYNTHETIC_DESIGN,
        SYNTHETIC_ALLOCATION_HASH,
    )


@pytest.mark.parametrize(
    "authorization_identity",
    [
        "NONEMPTY-BUT-ARBITRARY",
        preservation.V8D_AUTHORIZATION_PREFIX + ("4" * 40) + preservation.V8D_AUTHORIZATION_SEPARATOR + SYNTHETIC_ALLOCATION_HASH,
        preservation.V8D_AUTHORIZATION_PREFIX + SYNTHETIC_DESIGN + preservation.V8D_AUTHORIZATION_SEPARATOR + ("4" * 64),
        " " + SYNTHETIC_AUTHORIZATION,
        SYNTHETIC_AUTHORIZATION + " ",
    ],
)
def test_authorization_grammar_variants_block(authorization_identity):
    with pytest.raises(preservation.V8DT1CPreservationBlocked) as excinfo:
        preservation.validate_authorization_identity(
            authorization_identity,
            SYNTHETIC_DESIGN,
            SYNTHETIC_ALLOCATION_HASH,
        )
    assert excinfo.value.reason == "V8D_AUTHORIZATION_GRAMMAR_MISMATCH"


def test_authorization_hash_and_receipt_key_are_exact_and_deterministic():
    expected_identity_hash = hashlib.sha256(SYNTHETIC_AUTHORIZATION.encode("utf-8")).hexdigest()
    assert preservation.authorization_identity_sha256(SYNTHETIC_AUTHORIZATION) == expected_identity_hash
    expected_material = "|".join(
        (
            "ta1k1-arakawa/stock-analyzer",
            preservation.V8D_T1C_PRESERVATION_GATE,
            SYNTHETIC_DESIGN,
            expected_identity_hash,
            SYNTHETIC_ALLOCATION_HASH,
        )
    )
    expected_key = hashlib.sha256(expected_material.encode("utf-8")).hexdigest()
    assert _synthetic_receipt_key() == expected_key
    assert _synthetic_receipt_key() == expected_key


def test_receipt_has_exact_fields_and_raw_identity_is_not_persisted(tmp_path):
    receipt = _consume_synthetic(tmp_path)
    assert set(receipt) == set(preservation.V8D_RECEIPT_FIELDS)
    key = _synthetic_receipt_key()
    raw = (tmp_path / f"{key}.json").read_bytes()
    assert SYNTHETIC_AUTHORIZATION.encode("utf-8") not in raw
    assert set(preservation.read_gate_receipt(tmp_path, key)) == set(preservation.V8D_RECEIPT_FIELDS)


def test_receipt_extra_or_missing_field_blocks(tmp_path):
    _consume_synthetic(tmp_path)
    key = _synthetic_receipt_key()
    path = tmp_path / f"{key}.json"
    data = json.loads(path.read_text())
    data["extra"] = True
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(preservation.V8DT1CPreservationBlocked) as excinfo:
        preservation.read_gate_receipt(tmp_path, key)
    assert excinfo.value.reason == "V8D_RECEIPT_SCHEMA_INVALID"


def test_duplicate_receipt_blocks(tmp_path):
    _consume_synthetic(tmp_path)
    with pytest.raises(preservation.V8DT1CPreservationBlocked) as excinfo:
        _consume_synthetic(tmp_path)
    assert excinfo.value.reason == "V8D_GATE_ALREADY_CONSUMED"


def test_wrong_design_sha_blocks_before_private_reader(tmp_path):
    allocation_path = tmp_path / "allocation.json"
    manifest_path = tmp_path / "manifest.json"
    allocation_path.write_bytes(b"synthetic allocation")
    manifest_path.write_bytes(b"synthetic manifest")
    reads: list[Path] = []
    bad = _public_preflight()
    bad["frozen_design_commit"] = "0" * 40
    with pytest.raises(preservation.V8DT1CPreservationBlocked):
        preservation._execute_with_dependencies(
            authorization_identity=SYNTHETIC_AUTHORIZATION,
            state_root=tmp_path / "state",
            output_path=tmp_path / "result.json",
            allocation_artifact_path=allocation_path,
            partition_manifest_path=manifest_path,
            repository_root=tmp_path / "repo",
            public_preflight=lambda: bad,
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=preservation.consume_gate_once,
            clock=_clock,
        )
    assert reads == []
    assert not (tmp_path / "state").exists()


def test_execution_consumes_synthetic_gate_before_first_private_read(tmp_path):
    allocation_path = tmp_path / "allocation.json"
    manifest_path = tmp_path / "manifest.json"
    allocation_path.write_bytes(b"synthetic allocation")
    manifest_path.write_bytes(b"synthetic manifest")
    reads: list[Path] = []
    with pytest.raises(preservation.V8DT1CPreservationBlocked):
        preservation._execute_with_dependencies(
            authorization_identity=SYNTHETIC_AUTHORIZATION,
            state_root=tmp_path / "state",
            output_path=tmp_path / "result.json",
            allocation_artifact_path=allocation_path,
            partition_manifest_path=manifest_path,
            repository_root=tmp_path / "repo",
            public_preflight=_public_preflight,
            private_reader=lambda path: reads.append(path) or path.read_bytes(),
            gate_consumer=preservation.consume_gate_once,
            clock=_clock,
            reviewed_design_candidate_commit=SYNTHETIC_DESIGN,
            authorized_allocation_artifact_self_hash=SYNTHETIC_ALLOCATION_HASH,
        )
    assert len(reads) == 2
    assert list((tmp_path / "state").glob("*.json"))


def test_authorization_mismatch_blocks_before_receipt_and_private_reader(tmp_path):
    allocation_path = tmp_path / "allocation.json"
    manifest_path = tmp_path / "manifest.json"
    allocation_path.write_bytes(b"synthetic allocation")
    manifest_path.write_bytes(b"synthetic manifest")
    private_reads = 0
    receipt_count = 0

    def count_receipt(*args, **kwargs):
        nonlocal receipt_count
        receipt_count += 1
        return {}

    def count_private_read(path):
        nonlocal private_reads
        private_reads += 1
        return path.read_bytes()

    wrong_authorization = (
        preservation.V8D_AUTHORIZATION_PREFIX
        + ("4" * 40)
        + preservation.V8D_AUTHORIZATION_SEPARATOR
        + SYNTHETIC_ALLOCATION_HASH
    )
    with pytest.raises(preservation.V8DT1CPreservationBlocked) as excinfo:
        preservation._execute_with_dependencies(
            authorization_identity=wrong_authorization,
            state_root=tmp_path / "state",
            output_path=tmp_path / "result.json",
            allocation_artifact_path=allocation_path,
            partition_manifest_path=manifest_path,
            repository_root=tmp_path / "repo",
            public_preflight=_public_preflight,
            private_reader=count_private_read,
            gate_consumer=count_receipt,
            clock=_clock,
            reviewed_design_candidate_commit=SYNTHETIC_DESIGN,
            authorized_allocation_artifact_self_hash=SYNTHETIC_ALLOCATION_HASH,
        )
    assert excinfo.value.reason == "V8D_AUTHORIZATION_GRAMMAR_MISMATCH"
    assert receipt_count == 0
    assert private_reads == 0


def test_malformed_private_artifact_blocks():
    with pytest.raises(preservation.V8DT1CPreservationBlocked):
        preservation._verify_private_artifacts(b"{}", b"{}")


def test_allocation_hash_mismatch_blocks():
    artifact, manifest = _private_fixture()
    with pytest.raises(preservation.V8DT1CPreservationBlocked):
        preservation._verify_private_artifacts(
            allocation.canonical_json_bytes(artifact),
            v8_partition.canonical_json_bytes(manifest),
            expected_allocation_artifact_self_hash="0" * 64,
            expected_partition_manifest_sha256=manifest["manifest_sha256"],
            expected_partition_implementation_commit=SYNTHETIC_IMPL,
            expected_reviewed_implementation_commit=SYNTHETIC_IMPL,
            expected_v8c_frozen_design_commit=SYNTHETIC_V8C_DESIGN,
        )


def test_t1c_count_and_hash_mismatch_block():
    artifact, manifest = _private_fixture()
    tampered = dict(artifact)
    tampered["t1c_ticker_count"] = 301
    with pytest.raises(preservation.V8DT1CPreservationBlocked):
        preservation._verify_private_artifacts(
            allocation.canonical_json_bytes(tampered),
            v8_partition.canonical_json_bytes(manifest),
            expected_allocation_artifact_self_hash=artifact["artifact_self_hash"],
            expected_partition_manifest_sha256=manifest["manifest_sha256"],
            expected_partition_implementation_commit=SYNTHETIC_IMPL,
            expected_reviewed_implementation_commit=SYNTHETIC_IMPL,
            expected_v8c_frozen_design_commit=SYNTHETIC_V8C_DESIGN,
        )
    with pytest.raises(preservation.V8DT1CPreservationBlocked):
        preservation._verify_private_artifacts(
            allocation.canonical_json_bytes(artifact),
            v8_partition.canonical_json_bytes(manifest),
            expected_allocation_artifact_self_hash=artifact["artifact_self_hash"],
            expected_t1c_ticker_list_sha256="0" * 64,
            expected_partition_manifest_sha256=manifest["manifest_sha256"],
            expected_partition_implementation_commit=SYNTHETIC_IMPL,
            expected_reviewed_implementation_commit=SYNTHETIC_IMPL,
            expected_v8c_frozen_design_commit=SYNTHETIC_V8C_DESIGN,
        )


def test_public_artifact_contains_no_private_identity_fields():
    artifact, manifest = _private_fixture()
    summary = preservation._verify_private_artifacts(
        allocation.canonical_json_bytes(artifact),
        v8_partition.canonical_json_bytes(manifest),
        expected_allocation_artifact_self_hash=artifact["artifact_self_hash"],
        expected_t1c_ticker_list_sha256=artifact["t1c_ticker_list_sha256"],
        expected_remaining_t_spare_ticker_list_sha256=artifact["remaining_t_spare_ticker_list_sha256"],
        expected_parent_t_spare_ticker_list_sha256=artifact["parent_t_spare_ticker_list_sha256"],
        expected_partition_manifest_sha256=manifest["manifest_sha256"],
        expected_partition_implementation_commit=SYNTHETIC_IMPL,
        expected_reviewed_implementation_commit=SYNTHETIC_IMPL,
        expected_v8c_frozen_design_commit=SYNTHETIC_V8C_DESIGN,
    )
    public = preservation._build_public_artifact(summary)
    assert set(public) == set(preservation.V8D_PRESERVATION_ARTIFACT_FIELDS)
    assert "t1c_tickers" not in public
    assert "remaining_t_spare_tickers" not in public
    assert SYNTHETIC_AUTHORIZATION not in json.dumps(public)
