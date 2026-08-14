from __future__ import annotations

import errno
import hashlib
import json
import socket
import tempfile
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from src import v8c_acquisition_artifact_verification as verification
from src import v8c_historical_acquisition as acquisition
from src import v8c_human_gate_consumption as gate_consumption
from src.v8c_production_provenance import CANONICAL_PARSER_CLASSIFIER_BLOB_SHA

SYNTHETIC_REVIEWED_COMMIT = "b" * 40


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def _epoch(year, month, day):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


DEFAULT_DATES = [(2016, 4, 1), (2016, 4, 4), (2025, 12, 30)]


def synthetic_payload(ticker, price=1000.0):
    timestamps = [_epoch(*d) for d in DEFAULT_DATES]
    result = {
        "meta": {"symbol": ticker + ".T"}, "timestamp": timestamps,
        "indicators": {
            "quote": [{"open": [price] * 3, "high": [price + 2] * 3, "low": [price - 2] * 3, "close": [price] * 3, "volume": [1.0] * 3}],
            "adjclose": [{"adjclose": [price] * 3}],
        },
        "events": {},
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


class FakeResponse:
    def __init__(self, payload, url, status=200):
        self.payload, self.url, self.status = payload, url, status

    def read(self):
        return self.payload

    def close(self):
        pass


class FakeOpener:
    def __init__(self):
        self.calls = []

    def __call__(self, request_obj: Any):
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        return FakeResponse(synthetic_payload(ticker), url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def _tickers(prefix, count):
    return [f"{prefix}{i:04d}" for i in range(count)]


def build_manifest(tmp_path) -> dict:
    synthetic_repository_root = tmp_path / "synthetic-repository"
    synthetic_repository_root.mkdir()
    return acquisition._acquire_v8c_block_bundle_with_validated_inputs(
        output_root=tmp_path,
        repository_root=synthetic_repository_root,
        block="T1C",
        tickers=_tickers("FAKE", 300),
        authority_binding={
            "authorized_allocation_artifact_self_hash": "0" * 64,
            "parent_v8_partition_manifest_sha256": "1" * 64,
            "parent_v8_partition_implementation_commit": "2" * 40,
            "trust_pin_human_gate": "V8C_HUMAN_AUTHORIZE_T1C_ALLOCATION_PIN_AT_" + "0" * 64,
        },
        implementation_git_commit=SYNTHETIC_REVIEWED_COMMIT,
        classifier_blob_sha=CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
        opener=FakeOpener(),
        clock=clock_stub,
        consumption_gate=gate_consumption.GATE_T1C_RAW_ACQUISITION,
        consumption_state_root=Path(tempfile.gettempdir()) / ("v8c_gate-" + uuid.uuid4().hex),
        sleep_fn=lambda s: None,
    )


def _expected_kwargs(manifest):
    return dict(
        expected_v8c_frozen_design_commit=manifest["v8c_frozen_design_commit"],
        expected_reviewed_production_implementation_commit=manifest["reviewed_production_implementation_commit"],
        expected_authority_chain=manifest["authority_chain"],
        expected_ticker_list_sha256=manifest["ticker_list_sha256"],
        expected_authority_binding=dict(manifest["authority_binding"]),
    )


def test_valid_bundle_passes(tmp_path):
    manifest = build_manifest(tmp_path)
    result = verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))
    assert result["result"] == "PASS"
    assert result["ticker_count"] == 300
    assert result["total_retry_count"] == 0
    assert result["total_request_attempts"] == 300


def test_wrong_expected_design_commit_blocks(tmp_path):
    manifest = build_manifest(tmp_path)
    kwargs = _expected_kwargs(manifest)
    kwargs["expected_v8c_frozen_design_commit"] = "f" * 40
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **kwargs)
    assert excinfo.value.reason == "FROZEN_DESIGN_COMMIT_MISMATCH"


def test_wrong_expected_ticker_list_hash_blocks(tmp_path):
    manifest = build_manifest(tmp_path)
    kwargs = _expected_kwargs(manifest)
    kwargs["expected_ticker_list_sha256"] = "0" * 64
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **kwargs)
    assert excinfo.value.reason == "TICKER_LIST_SHA_MISMATCH"


def test_wrong_authority_binding_value_blocks(tmp_path):
    manifest = build_manifest(tmp_path)
    kwargs = _expected_kwargs(manifest)
    kwargs["expected_authority_binding"]["authorized_allocation_artifact_self_hash"] = "f" * 64
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **kwargs)
    assert excinfo.value.reason == "AUTHORITY_BINDING_VALUE_MISMATCH"


def test_raw_payload_byte_count_tampering_detected(tmp_path):
    manifest = build_manifest(tmp_path)
    raw_dir = tmp_path / acquisition.ACQUISITIONS_DIRNAME / "T1C" / acquisition.RAW_DIRNAME
    first_file = sorted(raw_dir.iterdir())[0]
    first_file.write_bytes(first_file.read_bytes() + b"EXTRA")
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))
    assert excinfo.value.reason == "RAW_PAYLOAD_BYTE_COUNT_MISMATCH"


def test_raw_payload_sha256_tampering_with_same_length_detected(tmp_path):
    manifest = build_manifest(tmp_path)
    raw_dir = tmp_path / acquisition.ACQUISITIONS_DIRNAME / "T1C" / acquisition.RAW_DIRNAME
    first_file = sorted(raw_dir.iterdir())[0]
    original = bytearray(first_file.read_bytes())
    original[0] = original[0] ^ 0xFF  # flip a bit, same byte count
    first_file.write_bytes(bytes(original))
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))
    assert excinfo.value.reason == "RAW_PAYLOAD_SHA256_MISMATCH"


def test_missing_raw_payload_file_detected(tmp_path):
    manifest = build_manifest(tmp_path)
    raw_dir = tmp_path / acquisition.ACQUISITIONS_DIRNAME / "T1C" / acquisition.RAW_DIRNAME
    sorted(raw_dir.iterdir())[0].unlink()
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))
    assert excinfo.value.reason == "RAW_PAYLOAD_MISSING"


def test_extra_raw_payload_file_detected(tmp_path):
    manifest = build_manifest(tmp_path)
    raw_dir = tmp_path / acquisition.ACQUISITIONS_DIRNAME / "T1C" / acquisition.RAW_DIRNAME
    (raw_dir / "EXTRA9999.json").write_bytes(b"{}")
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))
    assert excinfo.value.reason == "RAW_PAYLOAD_UNEXPECTED_EXTRA"


def test_symlink_raw_payload_rejected(tmp_path):
    manifest = build_manifest(tmp_path)
    raw_dir = tmp_path / acquisition.ACQUISITIONS_DIRNAME / "T1C" / acquisition.RAW_DIRNAME
    entries = sorted(raw_dir.iterdir())
    target = entries[0]
    link_name = raw_dir / "LINKFAKE9999.json"
    payload_manifest = manifest["payload_manifest"]
    victim_ticker = payload_manifest[0]["ticker"]
    victim_path = raw_dir / (victim_ticker + ".json")
    victim_path.unlink()
    try:
        link_name.symlink_to(target)
    except OSError as error:
        pytest.skip(f"symlink privilege unavailable: {error}")
    link_name.rename(victim_path)
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))
    assert excinfo.value.reason == "RAW_PAYLOAD_NON_REGULAR_ENTRY"


def test_ticker_count_wrong_blocks(tmp_path):
    manifest = build_manifest(tmp_path)
    manifest_path = tmp_path / acquisition.ACQUISITIONS_DIRNAME / "T1C" / acquisition.MANIFEST_FILENAME
    tampered = dict(manifest)
    tampered["ticker_count"] = 4
    tampered["payload_manifest_sha256"] = acquisition.sha256_bytes(acquisition.canonical_json_bytes(tampered["payload_manifest"]))
    manifest_path.write_bytes(acquisition.canonical_json_bytes(tampered))
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked):
        verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))


# ---------------------------------------------------------------------------
# Retry audit invariants (§10.1)
# ---------------------------------------------------------------------------


def test_retry_audit_attempts_out_of_range_blocked(tmp_path):
    manifest = build_manifest(tmp_path)
    manifest_path = tmp_path / acquisition.ACQUISITIONS_DIRNAME / "T1C" / acquisition.MANIFEST_FILENAME
    tampered = dict(manifest)
    tampered_payload = [dict(entry) for entry in tampered["payload_manifest"]]
    tampered_payload[0]["attempts"] = 4  # exceeds MAXIMUM_ATTEMPTS_PER_TICKER
    tampered_payload[0]["retry_count"] = 3
    tampered["payload_manifest"] = tampered_payload
    tampered["total_retry_count"] = manifest["total_retry_count"] + 3
    tampered["total_request_attempts"] = 300 + tampered["total_retry_count"]
    tampered["payload_manifest_sha256"] = acquisition.sha256_bytes(acquisition.canonical_json_bytes(tampered_payload))
    manifest_path.write_bytes(acquisition.canonical_json_bytes(tampered))
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))
    assert excinfo.value.reason == "RETRY_AUDIT_ATTEMPTS_INVALID"


def test_retry_audit_total_retry_count_sum_mismatch_blocked(tmp_path):
    manifest = build_manifest(tmp_path)
    manifest_path = tmp_path / acquisition.ACQUISITIONS_DIRNAME / "T1C" / acquisition.MANIFEST_FILENAME
    tampered = dict(manifest)
    tampered["total_retry_count"] = 5  # doesn't match sum of per-ticker retry_count (all 0)
    tampered["total_request_attempts"] = 300 + 5
    manifest_path.write_bytes(acquisition.canonical_json_bytes(tampered))
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))
    assert excinfo.value.reason == "RETRY_AUDIT_TOTAL_RETRY_COUNT_MISMATCH"


def test_retry_policy_field_tampering_blocked(tmp_path):
    manifest = build_manifest(tmp_path)
    manifest_path = tmp_path / acquisition.ACQUISITIONS_DIRNAME / "T1C" / acquisition.MANIFEST_FILENAME
    tampered = dict(manifest)
    tampered["backoff_seconds"] = [1, 2]
    manifest_path.write_bytes(acquisition.canonical_json_bytes(tampered))
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_acquisition_artifact(tmp_path, "T1C", **_expected_kwargs(manifest))
    assert excinfo.value.reason == "ACQUISITION_MANIFEST_INVALID:ACQUISITION_MANIFEST_RETRY_POLICY_MISMATCH"


# ---------------------------------------------------------------------------
# MEDIUM-2: retry-audit concrete-exception re-derivation (§4.3), never
# trusting the recorded ``classification``/``retryable`` fields at face
# value merely because ``http_code``/``exception_type`` look plausible.
# ---------------------------------------------------------------------------


def _fingerprint(ticker: str) -> str:
    material = {
        "logical_request_identity": ticker,
        "request_start": "2016-04-01",
        "request_end_exclusive": "2026-01-01",
        "provider": verification.DATA_SOURCE,
        "host": verification.DATA_SOURCE_HOST,
        "request_parameters": {"interval": "1d", "events": "div,splits", "includeAdjustedClose": True},
    }
    return hashlib.sha256(verification.canonical_json_bytes(material)).hexdigest()


def _valid_two_attempt_audit(ticker: str, intermediate_entry: dict) -> list[dict]:
    fp = _fingerprint(ticker)
    entry = dict(intermediate_entry)
    entry["attempt"] = 1
    entry["request_fingerprint"] = fp
    return [
        entry,
        {"attempt": 2, "classification": "SUCCESS", "retryable": None,
         "classification_metadata": {"exception_type": None}, "request_fingerprint": fp},
    ]


def test_valid_network_timeout_intermediate_attempt_passes():
    audit = _valid_two_attempt_audit("AAAA", {
        "classification": "NETWORK_TIMEOUT", "retryable": True,
        "classification_metadata": {
            "exception_type": "TimeoutError", "reason_type": "TimeoutError",
            "errno": None, "classification": "NETWORK_TIMEOUT",
        },
    })
    retry_count, fp = verification._verify_member_transport_audit("AAAA", audit)
    assert retry_count == 1
    assert fp == audit[0]["request_fingerprint"]


def test_http_code_valid_but_exception_type_forged_blocks():
    audit = _valid_two_attempt_audit("AAAA", {
        "classification": "HTTP_503", "retryable": True,
        "classification_metadata": {"exception_type": "ForgedType", "http_code": 503},
    })
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_member_transport_audit("AAAA", audit)
    assert excinfo.value.reason == "RETRY_AUDIT_CLASSIFICATION_DERIVATION_MISMATCH"


def test_errno_matches_but_concrete_type_forged_blocks():
    audit = _valid_two_attempt_audit("AAAA", {
        "classification": "CONNECTION_RESET", "retryable": True,
        "classification_metadata": {
            "exception_type": "ForgedOSError", "reason_type": "ForgedOSError",
            "errno": errno.ECONNRESET, "classification": "CONNECTION_RESET",
        },
    })
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_member_transport_audit("AAAA", audit)
    assert excinfo.value.reason == "RETRY_AUDIT_CLASSIFICATION_DERIVATION_MISMATCH"


def test_named_condition_and_concrete_type_disagree_blocks():
    # A named condition is always nonretryable, so ``retryable=False``
    # (its only truthful value) is itself already caught by the earlier
    # retryable-class check before the named-condition/classification
    # cross-check is reached -- confirming a mismatched named condition can
    # never masquerade as a legitimate intermediate retry.
    audit = _valid_two_attempt_audit("AAAA", {
        "classification": "SYMBOL_MISMATCH", "retryable": False,
        "classification_metadata": {"exception_type": "V8CTransportNamedFailure", "named_condition": "PARSER_SCHEMA_FAILURE"},
    })
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_member_transport_audit("AAAA", audit)
    assert excinfo.value.reason == "RETRY_AUDIT_NONRETRYABLE_INTERMEDIATE_FAILURE"


def test_named_condition_forged_exception_type_blocks():
    # Every named condition is unconditionally nonretryable per §4, so it
    # can never legitimately be an intermediate (retried) attempt in the
    # first place -- the earlier retryable-class check already fail-closes
    # this exact shape before the named-condition schema/type cross-check
    # is even reached.
    audit = _valid_two_attempt_audit("AAAA", {
        "classification": "DATA_QUALITY_GATE_FAILURE", "retryable": True,
        "classification_metadata": {"exception_type": "ForgedNamedType", "named_condition": "DATA_QUALITY_GATE_FAILURE"},
    })
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_member_transport_audit("AAAA", audit)
    assert excinfo.value.reason == "RETRY_AUDIT_NONRETRYABLE_INTERMEDIATE_FAILURE"


def test_outer_classification_and_metadata_classification_disagree_blocks():
    audit = _valid_two_attempt_audit("AAAA", {
        "classification": "NETWORK_TIMEOUT", "retryable": True,
        "classification_metadata": {
            "exception_type": "TimeoutError", "reason_type": "TimeoutError",
            "errno": None, "classification": "CONNECTION_RESET",  # forged inner field
        },
    })
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_member_transport_audit("AAAA", audit)
    assert excinfo.value.reason == "RETRY_AUDIT_CLASSIFICATION_DERIVATION_MISMATCH"


def test_unexpected_extra_metadata_field_blocks():
    audit = _valid_two_attempt_audit("AAAA", {
        "classification": "HTTP_503", "retryable": True,
        "classification_metadata": {"exception_type": "HTTPError", "http_code": 503, "extra": "field"},
    })
    with pytest.raises(verification.V8CAcquisitionArtifactVerificationBlocked) as excinfo:
        verification._verify_member_transport_audit("AAAA", audit)
    assert excinfo.value.reason == "RETRY_AUDIT_CLASSIFICATION_METADATA_INVALID"
