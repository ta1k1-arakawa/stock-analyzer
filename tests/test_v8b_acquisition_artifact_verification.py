from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from src import v8_partition as partition
from src import v8b_acquisition_artifact_verification as artifact_verification
from src import v8b_allocation as allocation
from src import v8b_allocation_verification as verification
from src import v8b_historical_acquisition as acquisition
from src import v8b_trust_pin as trust_pin

SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_REVIEWED_COMMIT = "b" * 40


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def clock_stub():
    return datetime(2026, 8, 12, tzinfo=timezone.utc)


def _synthetic_payload(ticker: str) -> bytes:
    dates = [(2016, 4, 1), (2016, 4, 4), (2025, 12, 30)]
    timestamps = [int(datetime(*d, tzinfo=timezone.utc).timestamp()) for d in dates]
    price = 1000.0
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": timestamps,
        "indicators": {
            "quote": [{
                "open": [price] * len(timestamps),
                "high": [price + 2.0] * len(timestamps),
                "low": [price - 2.0] * len(timestamps),
                "close": [price] * len(timestamps),
                "volume": [10000.0] * len(timestamps),
            }],
            "adjclose": [{"adjclose": [price] * len(timestamps)}],
        },
        "events": {},
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


class FakeResponse:
    def __init__(self, payload: bytes, url: str) -> None:
        self.payload = payload
        self.status = 200
        self.url = url

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        pass


class FakeOpener:
    def __call__(self, request_obj: Any) -> FakeResponse:
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        return FakeResponse(_synthetic_payload(ticker), url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")


def build_and_run_t1b(tmp_path: Path) -> tuple[dict, Path]:
    parent = _tickers("TS", 1904)
    artifact = allocation.build_t1b_allocation_artifact(
        parent_t_spare_tickers=parent,
        parent_v8_partition_manifest_sha256="0" * 64,
        parent_v8_partition_implementation_commit=SYNTHETIC_COMMIT,
        parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
        v8b_allocation_implementation_commit=SYNTHETIC_REVIEWED_COMMIT,
        clock=clock_stub,
    )
    result = verification._verify_t1b_allocation_artifact(
        artifact=artifact,
        parent_t_spare_tickers=parent,
        t0_tickers=_tickers("T0", 300),
        old_t1_tickers=_tickers("OT1", 300),
        t2_tickers=_tickers("T2X", 300),
        t3_tickers=_tickers("T3X", 300),
        expected_parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
    )
    pin = trust_pin.build_trust_pin(verification_result_summary=result, authorization_note="test")
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_bytes(allocation.canonical_json_bytes(artifact))

    def _unreachable_t2_reuse_recheck_resolver(head):
        raise AssertionError("t2_reuse_recheck_resolver must not run for a T1B acquisition")

    deps = dict(
        output_root=tmp_path / "private",
        block="T1B",
        confirmation=acquisition.T1B_ACQUISITION_CONFIRMATION,
        partition_manifest_path=None,
        t1b_allocation_artifact_path=artifact_path,
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        classifier_blob_resolver=lambda head: acquisition.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
        zoneinfo_loader=lambda: object(),
        anchor_reader=lambda head: {},
        bridge_reader=lambda head: {},
        t2_reuse_recheck_resolver=_unreachable_t2_reuse_recheck_resolver,
        t1b_trust_pin_reader=lambda head: pin,
        trust_pin_review_reader=lambda head, artifact_hash, human_gate: {"ok": True},
        opener=FakeOpener(),
        clock=clock_stub,
        monotonic_clock=lambda: 0.0,
        sleep_fn=lambda _s: None,
        consumption_state_root=tmp_path / "gate_state",
    )
    acquisition._acquire_production_v8b_historical_block_bundle_with_dependencies(**deps)
    return artifact, tmp_path / "private", pin


def write_partition_manifest(path: Path, *, t2: list[str] | None = None) -> dict:
    """A genuine, self-hash-verified V8 partition manifest fixture (must
    satisfy src.v8_partition.read_partition_manifest's full schema)."""
    blocks = {
        "T0": _tickers("T0BLK", 300), "T1": _tickers("T1BLK", 300),
        "T2": list(t2 or _tickers("T2", 300)),
        "T3": _tickers("T3BLK", 300), "T_spare": _tickers("TSBLK", 300),
    }
    manifest = {
        "schema_version": partition.SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "source_snapshot_semantics": partition.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": partition.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": "6" * 40,
        "created_utc": "2026-08-09T00:00:00Z",
        "source_url": "https://www.jpx.co.jp/synthetic/data_j.xls",
        "source_host": "www.jpx.co.jp",
        "source_acquisition_utc": "2026-08-09T00:00:00Z",
        "source_raw_sha256": "0" * 64,
        "source_raw_byte_count": 0,
        "v4_source_raw_sha256_reference": "1" * 64,
        "v4_raw_sha_equality_required": partition.V4_RAW_SHA_EQUALITY_REQUIRED,
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "eligible_ticker_count": 1500,
        "eligible_ticker_list_sha256": partition.ticker_list_sha256(sum((blocks[k] for k in blocks), [])),
        "selection_rule": "synthetic fixture selection rule",
        "deterministic_ordering_rule": partition.DETERMINISTIC_ORDERING_RULE,
        "t0_ticker_list_sha256": partition.ticker_list_sha256(blocks["T0"]),
        "t1_ticker_list_sha256": partition.ticker_list_sha256(blocks["T1"]),
        "t2_ticker_list_sha256": partition.ticker_list_sha256(blocks["T2"]),
        "t3_ticker_list_sha256": partition.ticker_list_sha256(blocks["T3"]),
        "t_spare_ticker_list_sha256": partition.ticker_list_sha256(blocks["T_spare"]),
        "legacy_exclude_list": [],
        "legacy_exclude_list_sha256": partition.ticker_list_sha256([]),
        "block_sizes": {k: len(v) for k, v in blocks.items()},
        "block_assignments": blocks,
        "p_hist_start": partition.P_HIST_START,
        "p_hist_end": partition.P_HIST_END,
        "t1_role": partition.T1_ROLE,
        "t2_role": partition.T2_ROLE,
        "t3_role": partition.T3_ROLE,
        "t3_price_acquisition_authorized": False,
    }
    manifest["manifest_sha256"] = partition.canonical_sha256(manifest)
    assert set(manifest) == set(partition.MANIFEST_FIELDS)
    path.write_bytes(partition.canonical_json_bytes(manifest))
    return manifest


def build_and_run_t2(tmp_path: Path, monkeypatch) -> Path:
    """Build and publish a fully self-consistent, honest T2 acquisition
    bundle (manifest + raw/ + SEALED.json) via the real production DI seam
    -- used as the baseline for MEDIUM-3's SEALED.json tamper tests."""
    t2_tickers = _tickers("T2X", 300)
    manifest_path = tmp_path / "partition.json"
    partition_manifest = write_partition_manifest(manifest_path, t2=t2_tickers)
    t2_hash = partition.ticker_list_sha256(t2_tickers)
    monkeypatch.setattr(acquisition, "EXPECTED_T2_TICKER_COUNT", len(t2_tickers))
    monkeypatch.setattr(acquisition, "EXPECTED_T2_TICKER_LIST_SHA256", t2_hash)

    anchor = {
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": partition_manifest["manifest_sha256"],
        "authorized_partition_implementation_git_commit": partition_manifest["partition_implementation_git_commit"],
    }
    bridge = {
        "authorized_parent_v8_partition_manifest_sha256": partition_manifest["manifest_sha256"],
        "expected_t2_ticker_list_sha256": t2_hash,
        "human_gate": "V8B_T2_AUTHORITY_INTEGRATION_OPTION_2_APPROVED",
        "v8_trust_anchor_git_identity": "7" * 40,
    }

    def _unreachable_t1b_trust_pin_reader(head):
        raise AssertionError("t1b_trust_pin_reader must not run for a T2 acquisition")

    deps = dict(
        output_root=tmp_path / "private",
        block="T2",
        confirmation=acquisition.T2_ACQUISITION_CONFIRMATION,
        partition_manifest_path=manifest_path,
        t1b_allocation_artifact_path=None,
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        classifier_blob_resolver=lambda head: acquisition.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
        zoneinfo_loader=lambda: object(),
        anchor_reader=lambda head: anchor,
        bridge_reader=lambda head: bridge,
        t2_reuse_recheck_resolver=lambda: {"result": "PASS", "block": "T2"},
        t1b_trust_pin_reader=_unreachable_t1b_trust_pin_reader,
        trust_pin_review_reader=lambda head, artifact_hash, human_gate: {"ok": True},
        opener=FakeOpener(),
        clock=clock_stub,
        monotonic_clock=lambda: 0.0,
        sleep_fn=lambda _s: None,
        consumption_state_root=tmp_path / "gate_state",
    )
    acquisition._acquire_production_v8b_historical_block_bundle_with_dependencies(**deps)
    return tmp_path / "private"


def expected_kwargs_for(artifact: dict, pin: dict) -> dict:
    """The full ``expected_*`` kwarg set a *correctly-computed* production
    trust root would derive for this synthetic T1B artifact/pin pair --
    used to exercise the pure ``verify_acquisition_artifact`` checker
    directly with fake/synthetic values (round-3 finding HIGH-4)."""
    return dict(
        expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
        expected_reviewed_production_implementation_commit=SYNTHETIC_REVIEWED_COMMIT,
        expected_authority_chain="V8B_SUCCESSOR_ALLOCATION_AUTHORITY",
        expected_ticker_list_sha256=allocation.ticker_list_sha256(artifact["t1b_tickers"]),
        expected_authority_binding={
            "authorized_allocation_artifact_self_hash": artifact["artifact_self_hash"],
            "parent_v8_partition_manifest_sha256": artifact["parent_v8_partition_manifest_sha256"],
            "parent_v8_partition_implementation_commit": artifact["parent_v8_partition_implementation_commit"],
            "trust_pin_human_gate": pin["human_gate"],
        },
    )


def test_pass_on_honest_bundle(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    result = artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert result["result"] == "PASS"
    assert result["payload_manifest_record_count"] == 300
    assert result["ticker_count"] == 300


def test_detects_missing_payload_file(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    next(raw_dir.iterdir()).unlink()
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "RAW_PAYLOAD_MISSING"


def test_detects_extra_payload_file(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    (raw_dir / "EXTRA9999.json").write_bytes(b"{}")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "RAW_PAYLOAD_UNEXPECTED_EXTRA"


@pytest.mark.parametrize("directory_name", ["features", "research", "backtest", "model"])
def test_raw_directory_extra_or_nested_directory_blocks(tmp_path, directory_name):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    (raw_dir / directory_name).mkdir()
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "RAW_PAYLOAD_NON_REGULAR_ENTRY"


def test_expected_raw_payload_replaced_by_directory_blocks(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    expected = next(raw_dir.glob("*.json"))
    expected.unlink()
    expected.mkdir()
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "RAW_PAYLOAD_NON_REGULAR_ENTRY"


def test_expected_raw_payload_symlink_blocks_without_following(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    expected = next(raw_dir.glob("*.json"))
    target = tmp_path / "outside-payload.json"
    target.write_bytes(expected.read_bytes())
    expected.unlink()
    try:
        expected.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable in this test environment")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "RAW_PAYLOAD_NON_REGULAR_ENTRY"


def test_extra_raw_symlink_blocks(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    target = tmp_path / "outside-payload.json"
    target.write_bytes(b"{}")
    link = raw_dir / "EXTRA_LINK.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links unavailable in this test environment")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "RAW_PAYLOAD_NON_REGULAR_ENTRY"


def test_detects_modified_payload_bytes(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    victim = next(raw_dir.iterdir())
    victim.write_bytes(victim.read_bytes() + b"tampered")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "RAW_PAYLOAD_BYTE_COUNT_MISMATCH"


# ---------------------------------------------------------------------------
# MEDIUM-1: strengthened checks
# ---------------------------------------------------------------------------


def _load_manifest(output_root: Path, block: str) -> dict:
    manifest_path = output_root / acquisition.ACQUISITIONS_DIRNAME / block / acquisition.MANIFEST_FILENAME
    return json.loads(manifest_path.read_bytes())


def _rewrite_manifest(output_root: Path, block: str, manifest: dict) -> None:
    manifest_path = output_root / acquisition.ACQUISITIONS_DIRNAME / block / acquisition.MANIFEST_FILENAME
    manifest_path.write_bytes(acquisition.canonical_json_bytes(manifest))


def test_payload_manifest_hash_tampering_blocks(tmp_path):
    """MEDIUM-1: payload_manifest_sha256 is recomputed from the actual
    payload_manifest list, not merely trusted from the manifest's own
    stated field."""
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["payload_manifest_sha256"] = "0" * 64
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    # read_acquisition_manifest itself doesn't check payload_manifest_sha256
    # consistency, so this reaches the artifact verifier's own recompute check.
    assert excinfo.value.reason == "PAYLOAD_MANIFEST_SHA_MISMATCH"


def test_classifier_blob_tampering_blocks(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["canonical_parser_classifier_blob_sha"] = "0" * 40
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "CLASSIFIER_BLOB_MISMATCH"


def test_data_source_tampering_blocks(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["data_source_schema"] = "some other schema"
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "DATA_SOURCE_SCHEMA_MISMATCH"


def test_authority_binding_schema_mismatch_blocks(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    kwargs = expected_kwargs_for(artifact, pin)
    kwargs["expected_authority_chain"] = "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY_OPTION_2_BRIDGE"  # wrong chain for T1B
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **kwargs)
    assert excinfo.value.reason == "AUTHORITY_CHAIN_MISMATCH"


def test_authority_binding_field_set_must_match_block(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["authority_binding"]["extra_unexpected_field"] = "value"
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "AUTHORITY_BINDING_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# Round-3 HIGH-4: exact ticker_list_sha256 / authority_binding VALUE checks
# ---------------------------------------------------------------------------


def test_ticker_list_sha_mismatch_blocks(tmp_path):
    """A manifest whose own ticker_list_sha256 doesn't match the expected
    (Git-derived, in production) hash must BLOCK -- proves block membership
    authority is checked, not merely that the field is present."""
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    kwargs = expected_kwargs_for(artifact, pin)
    kwargs["expected_ticker_list_sha256"] = "0" * 64
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **kwargs)
    assert excinfo.value.reason == "TICKER_LIST_SHA_MISMATCH"


def test_authority_binding_matching_keys_but_wrong_values_blocks(tmp_path):
    """A forged authority_binding with the exact right KEY SET but one
    wrong VALUE must still BLOCK -- proves the check is a full value
    comparison, not merely a schema/key-set check (round-3 finding
    HIGH-4)."""
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["authority_binding"]["trust_pin_human_gate"] = "FORGED_BUT_SAME_KEY_SET"
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "AUTHORITY_BINDING_VALUE_MISMATCH"


def test_expected_authority_binding_wrong_schema_blocks(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    kwargs = expected_kwargs_for(artifact, pin)
    kwargs["expected_authority_binding"] = {"unexpected_field": "value"}
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **kwargs)
    assert excinfo.value.reason == "EXPECTED_AUTHORITY_BINDING_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# Repeat-round finding MEDIUM-3: READ_ONLY_T2_ACQUISITION_ARTIFACT_
# VERIFICATION must independently verify the actual on-disk SEALED.json
# bundle state, not merely the acquisition manifest's own self-reported
# sealed/research_access_authorized fields.
# ---------------------------------------------------------------------------


def expected_kwargs_for_t2(output_root: Path) -> dict:
    manifest = _load_manifest(output_root, "T2")
    return dict(
        expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
        expected_reviewed_production_implementation_commit=SYNTHETIC_REVIEWED_COMMIT,
        expected_authority_chain="ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY_OPTION_2_BRIDGE",
        expected_ticker_list_sha256=manifest["ticker_list_sha256"],
        expected_authority_binding=manifest["authority_binding"],
    )


def _sealed_path(output_root: Path) -> Path:
    return output_root / acquisition.ACQUISITIONS_DIRNAME / "T2" / acquisition.SEALED_FILENAME


def test_pass_on_honest_t2_bundle_verifies_the_real_sealed_record(tmp_path, monkeypatch):
    output_root = build_and_run_t2(tmp_path, monkeypatch)
    result = artifact_verification._verify_acquisition_artifact(output_root, "T2", **expected_kwargs_for_t2(output_root))
    assert result["result"] == "PASS"
    assert result["sealed"] is True


def test_t2_sealed_record_deleted_blocks(tmp_path, monkeypatch):
    output_root = build_and_run_t2(tmp_path, monkeypatch)
    _sealed_path(output_root).unlink()
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T2", **expected_kwargs_for_t2(output_root))
    assert excinfo.value.reason == "BLOCK_BUNDLE_TOP_LEVEL_ENTRIES_INVALID"


def test_t2_sealed_record_sealed_false_blocks(tmp_path, monkeypatch):
    output_root = build_and_run_t2(tmp_path, monkeypatch)
    record = json.loads(_sealed_path(output_root).read_bytes())
    record["sealed"] = False
    _sealed_path(output_root).write_bytes(acquisition.canonical_json_bytes(record))
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T2", **expected_kwargs_for_t2(output_root))
    assert excinfo.value.reason == "SEALED_RECORD_INVALID:SEALED_RECORD_SEALED_INVARIANT_VIOLATED"


def test_t2_sealed_record_research_access_authorized_true_blocks(tmp_path, monkeypatch):
    output_root = build_and_run_t2(tmp_path, monkeypatch)
    record = json.loads(_sealed_path(output_root).read_bytes())
    record["research_access_authorized"] = True
    _sealed_path(output_root).write_bytes(acquisition.canonical_json_bytes(record))
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T2", **expected_kwargs_for_t2(output_root))
    assert excinfo.value.reason == "SEALED_RECORD_INVALID:SEALED_RECORD_RESEARCH_ACCESS_INVARIANT_VIOLATED"


def test_t2_sealed_record_duplicate_json_key_blocks(tmp_path, monkeypatch):
    output_root = build_and_run_t2(tmp_path, monkeypatch)
    _sealed_path(output_root).write_bytes(
        b'{"sealed": true, "sealed": true, "research_access_authorized": false, "note": "x"}'
    )
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T2", **expected_kwargs_for_t2(output_root))
    assert excinfo.value.reason == "SEALED_RECORD_INVALID:SEALED_RECORD_DUPLICATE_KEY"


def test_t2_sealed_record_malformed_json_blocks(tmp_path, monkeypatch):
    output_root = build_and_run_t2(tmp_path, monkeypatch)
    _sealed_path(output_root).write_bytes(b"{not valid json")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T2", **expected_kwargs_for_t2(output_root))
    assert excinfo.value.reason == "SEALED_RECORD_INVALID:SEALED_RECORD_INVALID_JSON"


def test_t2_sealed_record_unexpected_extra_field_blocks(tmp_path, monkeypatch):
    output_root = build_and_run_t2(tmp_path, monkeypatch)
    record = json.loads(_sealed_path(output_root).read_bytes())
    record["extra_unexpected_field"] = "value"
    _sealed_path(output_root).write_bytes(acquisition.canonical_json_bytes(record))
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T2", **expected_kwargs_for_t2(output_root))
    assert excinfo.value.reason == "SEALED_RECORD_INVALID:SEALED_RECORD_SCHEMA_INVALID"


def test_t2_bundle_unexpected_extra_top_level_file_blocks(tmp_path, monkeypatch):
    output_root = build_and_run_t2(tmp_path, monkeypatch)
    (output_root / acquisition.ACQUISITIONS_DIRNAME / "T2" / "UNEXPECTED_EXTRA_FILE.json").write_bytes(b"{}")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T2", **expected_kwargs_for_t2(output_root))
    assert excinfo.value.reason == "BLOCK_BUNDLE_TOP_LEVEL_ENTRIES_INVALID"


def test_t2_bundle_unexpected_extra_top_level_directory_blocks(tmp_path, monkeypatch):
    output_root = build_and_run_t2(tmp_path, monkeypatch)
    (output_root / acquisition.ACQUISITIONS_DIRNAME / "T2" / "unexpected_dir").mkdir()
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T2", **expected_kwargs_for_t2(output_root))
    assert excinfo.value.reason == "BLOCK_BUNDLE_TOP_LEVEL_ENTRIES_INVALID"


def test_t1b_bundle_with_sealed_record_present_is_prohibited(tmp_path):
    """T1B must never carry the T2-only SEALED.json contract, even if
    every other T1B check would otherwise PASS."""
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    (output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.SEALED_FILENAME).write_bytes(
        acquisition.canonical_json_bytes({"sealed": True, "research_access_authorized": False, "note": "forged"})
    )
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "T1B_BUNDLE_MUST_NOT_CONTAIN_SEALED_RECORD"


# ---------------------------------------------------------------------------
# Round-3 HIGH-4: production resolver derives trust from verified Git only
# ---------------------------------------------------------------------------


def test_production_resolver_signature_accepts_no_caller_supplied_expected_values():
    import inspect

    params = set(inspect.signature(artifact_verification.resolve_and_verify_acquisition_artifact).parameters)
    assert params == {"output_root", "block"}
    for forbidden in (
        "repository_root",
        "expected_v8b_frozen_design_commit",
        "expected_reviewed_production_implementation_commit",
        "expected_authority_chain",
        "expected_ticker_list_sha256",
        "expected_authority_binding",
    ):
        assert forbidden not in params


def test_production_resolver_fails_closed_on_real_repo_today(tmp_path):
    """The real repository has no real published T1B/T2 acquisition bundle
    (and no real V8B_TRUSTED_ALLOCATION.json trust pin either) -- the real
    production resolver must fail closed today, proving zero real
    acquisition/allocation is required or performed by this phase."""
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked):
        artifact_verification.resolve_and_verify_acquisition_artifact(tmp_path / "private", "T1B")


def test_production_resolver_rejects_invalid_block():
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.resolve_and_verify_acquisition_artifact("/nonexistent", "T3")
    assert excinfo.value.reason == "BLOCK_INVALID"


# ---------------------------------------------------------------------------
# Repeat-round finding HIGH-1: READ_ONLY_T1B_ACQUISITION_ARTIFACT_
# VERIFICATION must re-establish the *complete* T1B authority chain,
# including INDEPENDENT_TRUST_PIN_REVIEW -- not merely the trust pin's own
# authorization_status. The frozen-commit/design-freeze/implementation-
# review checks cannot be satisfied by a fabricated repository (their
# expected Git object IDs are fixed literals that only exist in this
# repository's real history), so this exercises the private DI-testable
# seam with those specific steps monkeypatched to a synthetic PASS and only
# the trust-pin-review step left real.
# ---------------------------------------------------------------------------


def test_t1b_production_resolver_blocks_when_trust_pin_review_missing(tmp_path, monkeypatch):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)

    monkeypatch.setattr(
        artifact_verification, "resolve_verified_v8b_production_git_commit", lambda root: SYNTHETIC_COMMIT
    )
    monkeypatch.setattr(artifact_verification, "verify_frozen_design_object", lambda root: None)
    monkeypatch.setattr(
        artifact_verification, "read_and_verify_design_freeze_approval", lambda root, head: {"ok": True}
    )
    monkeypatch.setattr(
        artifact_verification,
        "verify_reviewed_implementation_binding",
        lambda root, head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
    )
    monkeypatch.setattr(artifact_verification, "read_t1b_trust_pin_from_verified_head", lambda root, head: pin)

    def missing_review(root, head, *, expected_allocation_artifact_self_hash, expected_trust_pin_human_gate):
        raise artifact_verification.V8BProductionProvenanceBlocked("V8B_TRUST_PIN_INDEPENDENT_REVIEW_MISSING")

    monkeypatch.setattr(artifact_verification, "read_and_verify_trust_pin_independent_review", missing_review)

    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._resolve_and_verify_acquisition_artifact_with_repository_root(
            output_root, "T1B", repository_root=tmp_path / "unused_repo_root"
        )
    assert excinfo.value.reason == "V8B_TRUST_PIN_INDEPENDENT_REVIEW_MISSING"


def test_t1b_production_resolver_binds_trust_pin_review_to_the_exact_pin_hash_and_gate(tmp_path, monkeypatch):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)

    monkeypatch.setattr(
        artifact_verification, "resolve_verified_v8b_production_git_commit", lambda root: SYNTHETIC_COMMIT
    )
    monkeypatch.setattr(artifact_verification, "verify_frozen_design_object", lambda root: None)
    monkeypatch.setattr(
        artifact_verification, "read_and_verify_design_freeze_approval", lambda root, head: {"ok": True}
    )
    monkeypatch.setattr(
        artifact_verification,
        "verify_reviewed_implementation_binding",
        lambda root, head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
    )
    monkeypatch.setattr(artifact_verification, "read_t1b_trust_pin_from_verified_head", lambda root, head: pin)

    seen: list[tuple] = []

    def recording_review(root, head, *, expected_allocation_artifact_self_hash, expected_trust_pin_human_gate):
        seen.append((expected_allocation_artifact_self_hash, expected_trust_pin_human_gate))
        return {"ok": True}

    monkeypatch.setattr(artifact_verification, "read_and_verify_trust_pin_independent_review", recording_review)

    result = artifact_verification._resolve_and_verify_acquisition_artifact_with_repository_root(
        output_root, "T1B", repository_root=tmp_path / "unused_repo_root"
    )
    assert result["result"] == "PASS"
    assert seen == [(pin["authorized_allocation_artifact_self_hash"], pin["human_gate"])]


# ---------------------------------------------------------------------------
# Round-3 repeat HIGH-2: block membership is bound to the concrete payload,
# not merely the manifest's own claimed ticker_list_sha256.
# ---------------------------------------------------------------------------


def test_forged_payload_with_correct_trusted_hash_but_different_ticker_set_blocks(tmp_path):
    """The exact scenario HIGH-2 describes: manifest.ticker_list_sha256
    keeps the correct trusted value, but payload_manifest/raw files are
    swapped to name a completely different 300-ticker set (with
    payload_manifest_sha256 recomputed to match the forged list, so that
    check alone cannot catch it). Only recomputing the membership hash
    from the concrete payload tickers catches this."""
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    original_ticker_list_sha256 = manifest["ticker_list_sha256"]

    forged_tickers = _tickers("TS", 1904)[300:600]  # a disjoint, equally well-formed 300-ticker set
    forged_payload_manifest = [
        {**entry, "ticker": forged_ticker}
        for entry, forged_ticker in zip(manifest["payload_manifest"], forged_tickers)
    ]
    manifest["payload_manifest"] = forged_payload_manifest
    manifest["payload_manifest_sha256"] = acquisition.sha256_bytes(
        acquisition.canonical_json_bytes(forged_payload_manifest)
    )
    # ticker_list_sha256 is deliberately left unchanged -- still the
    # correct, trusted value -- to prove that field alone is insufficient.
    assert manifest["ticker_list_sha256"] == original_ticker_list_sha256
    _rewrite_manifest(output_root, "T1B", manifest)

    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "PAYLOAD_TICKER_MEMBERSHIP_HASH_MISMATCH"


def test_payload_manifest_record_with_extra_field_blocks(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["payload_manifest"][0]["extra_unexpected_field"] = "value"
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "PAYLOAD_MANIFEST_RECORD_SCHEMA_INVALID"


def test_payload_manifest_record_missing_field_blocks(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    del manifest["payload_manifest"][0]["byte_count"]
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "PAYLOAD_MANIFEST_RECORD_SCHEMA_INVALID"


def test_non_canonical_ticker_spelling_blocks(tmp_path):
    """A ticker spelled in a non-canonical form (e.g. lowercase) must BLOCK
    even if every hash/count otherwise lines up -- membership is bound to
    canonical ticker identity."""
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    original_ticker = manifest["payload_manifest"][0]["ticker"]
    manifest["payload_manifest"][0]["ticker"] = original_ticker.lower()
    manifest["payload_manifest_sha256"] = acquisition.sha256_bytes(
        acquisition.canonical_json_bytes(manifest["payload_manifest"])
    )
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason in {"PAYLOAD_MANIFEST_TICKER_NOT_CANONICAL", "PAYLOAD_TICKER_MEMBERSHIP_HASH_MISMATCH"}


def test_duplicate_ticker_in_payload_manifest_blocks(tmp_path):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["payload_manifest"][1]["ticker"] = manifest["payload_manifest"][0]["ticker"]
    manifest["payload_manifest_sha256"] = acquisition.sha256_bytes(
        acquisition.canonical_json_bytes(manifest["payload_manifest"])
    )
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "PAYLOAD_MANIFEST_DUPLICATE_TICKER"


def test_membership_hash_recomputed_from_payload_order_not_resorted(tmp_path):
    """Recomputation must preserve payload_manifest's own order (the hash
    rule is order-sensitive) -- a bundle that is honest except for a
    reordering of payload_manifest entries must BLOCK, proving the check
    doesn't silently re-sort before hashing."""
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    reordered = list(reversed(manifest["payload_manifest"]))
    manifest["payload_manifest"] = reordered
    manifest["payload_manifest_sha256"] = acquisition.sha256_bytes(acquisition.canonical_json_bytes(reordered))
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "PAYLOAD_TICKER_MEMBERSHIP_HASH_MISMATCH"


# ---------------------------------------------------------------------------
# Round-3 repeat MEDIUM-2: pure/private helper is not a public production API
# ---------------------------------------------------------------------------


def test_private_pure_checker_is_not_publicly_exported():
    assert "verify_acquisition_artifact" not in artifact_verification.__all__
    assert not hasattr(artifact_verification, "verify_acquisition_artifact")
    assert hasattr(artifact_verification, "_verify_acquisition_artifact")


def test_only_the_production_resolver_and_safe_constants_are_public():
    assert set(artifact_verification.__all__) == {
        "CANONICAL_REPOSITORY_ROOT",
        "EXPECTED_AUTHORITY_CHAIN_BY_BLOCK",
        "V8BAcquisitionArtifactVerificationBlocked",
        "resolve_and_verify_acquisition_artifact",
    }


# ---------------------------------------------------------------------------
# Round-3 repeat HIGH-3: filesystem error privacy boundary
# ---------------------------------------------------------------------------

SECRET_PRIVATE_PATH_FRAGMENT = "/very/secret/private/acquisition/output"


def test_raw_payload_directory_unreadable_never_leaks_path(tmp_path, monkeypatch):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)

    real_scandir = artifact_verification.os.scandir

    def poisoned_scandir(path):
        if Path(path).name == acquisition.RAW_DIRNAME:
            raise OSError(f"permission denied listing {SECRET_PRIVATE_PATH_FRAGMENT}/raw")
        return real_scandir(path)

    monkeypatch.setattr(artifact_verification.os, "scandir", poisoned_scandir)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "RAW_PAYLOAD_DIRECTORY_UNREADABLE"
    assert SECRET_PRIVATE_PATH_FRAGMENT not in excinfo.value.reason


def test_raw_payload_read_failure_never_leaks_path_or_ticker(tmp_path, monkeypatch):
    artifact, output_root, pin = build_and_run_t1b(tmp_path)
    secret_ticker = artifact["t1b_tickers"][0]

    real_open = artifact_verification.os.open

    def poisoned_open(path, flags, *args):
        if Path(path).suffix == ".json" and Path(path).parent.name == acquisition.RAW_DIRNAME:
            raise OSError(f"permission denied reading {secret_ticker} at {SECRET_PRIVATE_PATH_FRAGMENT}")
        return real_open(path, flags, *args)

    monkeypatch.setattr(artifact_verification.os, "open", poisoned_open)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification._verify_acquisition_artifact(output_root, "T1B", **expected_kwargs_for(artifact, pin))
    assert excinfo.value.reason == "RAW_PAYLOAD_MISSING"
    assert SECRET_PRIVATE_PATH_FRAGMENT not in excinfo.value.reason
    assert secret_ticker not in excinfo.value.reason
