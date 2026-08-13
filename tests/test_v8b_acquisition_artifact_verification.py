from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

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
    result = verification.verify_t1b_allocation_artifact(
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

    deps = dict(
        output_root=tmp_path / "private",
        block="T1B",
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
        t1b_trust_pin_reader=lambda head: pin,
        opener=FakeOpener(),
        clock=clock_stub,
        monotonic_clock=lambda: 0.0,
        sleep_fn=lambda _s: None,
    )
    acquisition._acquire_production_v8b_historical_block_bundle_with_dependencies(**deps)
    return artifact, tmp_path / "private"


EXPECTED_KWARGS = dict(
    expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
    expected_reviewed_production_implementation_commit=SYNTHETIC_REVIEWED_COMMIT,
    expected_authority_chain="V8B_SUCCESSOR_ALLOCATION_AUTHORITY",
)


def test_pass_on_honest_bundle(tmp_path):
    artifact, output_root = build_and_run_t1b(tmp_path)
    result = artifact_verification.verify_acquisition_artifact(output_root, "T1B", **EXPECTED_KWARGS)
    assert result["result"] == "PASS"
    assert result["payload_manifest_record_count"] == 300
    assert result["ticker_count"] == 300


def test_detects_missing_payload_file(tmp_path):
    artifact, output_root = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    next(raw_dir.iterdir()).unlink()
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(output_root, "T1B", **EXPECTED_KWARGS)
    assert excinfo.value.reason == "RAW_PAYLOAD_MISSING"


def test_detects_extra_payload_file(tmp_path):
    artifact, output_root = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    (raw_dir / "EXTRA9999.json").write_bytes(b"{}")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(output_root, "T1B", **EXPECTED_KWARGS)
    assert excinfo.value.reason == "RAW_PAYLOAD_UNEXPECTED_EXTRA"


def test_detects_modified_payload_bytes(tmp_path):
    artifact, output_root = build_and_run_t1b(tmp_path)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    victim = next(raw_dir.iterdir())
    victim.write_bytes(victim.read_bytes() + b"tampered")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(output_root, "T1B", **EXPECTED_KWARGS)
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
    artifact, output_root = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["payload_manifest_sha256"] = "0" * 64
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(output_root, "T1B", **EXPECTED_KWARGS)
    # read_acquisition_manifest itself doesn't check payload_manifest_sha256
    # consistency, so this reaches the artifact verifier's own recompute check.
    assert excinfo.value.reason == "PAYLOAD_MANIFEST_SHA_MISMATCH"


def test_classifier_blob_tampering_blocks(tmp_path):
    artifact, output_root = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["canonical_parser_classifier_blob_sha"] = "0" * 40
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(output_root, "T1B", **EXPECTED_KWARGS)
    assert excinfo.value.reason == "CLASSIFIER_BLOB_MISMATCH"


def test_data_source_tampering_blocks(tmp_path):
    artifact, output_root = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["data_source_schema"] = "some other schema"
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(output_root, "T1B", **EXPECTED_KWARGS)
    assert excinfo.value.reason == "DATA_SOURCE_SCHEMA_MISMATCH"


def test_authority_binding_schema_mismatch_blocks(tmp_path):
    artifact, output_root = build_and_run_t1b(tmp_path)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(
            output_root, "T1B",
            expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
            expected_reviewed_production_implementation_commit=SYNTHETIC_REVIEWED_COMMIT,
            expected_authority_chain="ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY_OPTION_2_BRIDGE",  # wrong chain for T1B
        )
    assert excinfo.value.reason == "AUTHORITY_CHAIN_MISMATCH"


def test_authority_binding_field_set_must_match_block(tmp_path):
    artifact, output_root = build_and_run_t1b(tmp_path)
    manifest = _load_manifest(output_root, "T1B")
    manifest["authority_binding"]["extra_unexpected_field"] = "value"
    _rewrite_manifest(output_root, "T1B", manifest)
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(output_root, "T1B", **EXPECTED_KWARGS)
    assert excinfo.value.reason == "AUTHORITY_BINDING_SCHEMA_INVALID"
