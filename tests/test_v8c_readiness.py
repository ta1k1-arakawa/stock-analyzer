from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8c_human_gate_consumption as gate_consumption
from src import v8c_readiness as readiness

SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_REVIEWED_COMMIT = "b" * 40


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def _t0_manifest(tmp_path, t0_tickers):
    from src import v8_partition as partition

    blocks = {
        "T0": t0_tickers, "T1": _tickers("T1", 300), "T2": _tickers("T2", 300),
        "T3": _tickers("T3", 300), "T_spare": _tickers("SPARE", 1904),
    }
    manifest = {
        "schema_version": partition.SCHEMA_VERSION, "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT, "source_snapshot_semantics": partition.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": partition.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": "6" * 40, "created_utc": "2026-08-09T00:00:00Z",
        "source_url": "https://www.jpx.co.jp/synthetic/data_j.xls", "source_host": "www.jpx.co.jp",
        "source_acquisition_utc": "2026-08-09T00:00:00Z", "source_raw_sha256": "0" * 64, "source_raw_byte_count": 0,
        "v4_source_raw_sha256_reference": "1" * 64, "v4_raw_sha_equality_required": partition.V4_RAW_SHA_EQUALITY_REQUIRED,
        "source_reproduction_status": "PASS", "t0_reproduction_status": "PASS", "eligible_ticker_count": 1500,
        "eligible_ticker_list_sha256": partition.ticker_list_sha256(sum((blocks[k] for k in blocks), [])),
        "selection_rule": "synthetic", "deterministic_ordering_rule": partition.DETERMINISTIC_ORDERING_RULE,
        "t0_ticker_list_sha256": partition.ticker_list_sha256(blocks["T0"]),
        "t1_ticker_list_sha256": partition.ticker_list_sha256(blocks["T1"]),
        "t2_ticker_list_sha256": partition.ticker_list_sha256(blocks["T2"]),
        "t3_ticker_list_sha256": partition.ticker_list_sha256(blocks["T3"]),
        "t_spare_ticker_list_sha256": partition.ticker_list_sha256(blocks["T_spare"]),
        "legacy_exclude_list": [], "legacy_exclude_list_sha256": partition.ticker_list_sha256([]),
        "block_sizes": {k: len(v) for k, v in blocks.items()}, "block_assignments": blocks,
        "p_hist_start": partition.P_HIST_START, "p_hist_end": partition.P_HIST_END,
        "t1_role": partition.T1_ROLE, "t2_role": partition.T2_ROLE, "t3_role": partition.T3_ROLE,
        "t3_price_acquisition_authorized": False,
    }
    manifest["manifest_sha256"] = partition.canonical_sha256(manifest)
    path = tmp_path / "partition.json"
    path.write_bytes(partition.canonical_json_bytes(manifest))
    return path, manifest


def _valid_anchor_for(manifest):
    return {
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": manifest["manifest_sha256"],
        "authorized_partition_implementation_git_commit": manifest["partition_implementation_git_commit"],
    }


def _base_deps(**overrides):
    deps = dict(
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        classifier_blob_resolver=lambda head: readiness.CANONICAL_PARSER_CLASSIFIER_FILE and "76b57b077f3214e666ff9dc06d9c224afc16df9f",
        clock=clock_stub,
        sleep_fn=lambda seconds: None,
        opener=lambda request_obj: (_ for _ in ()).throw(AssertionError("opener must not be called")),
        consumption_state_root=Path(tempfile.gettempdir()) / ("v8c_readiness_state-" + uuid.uuid4().hex),
    )
    deps.update(overrides)
    return deps


def _patch_v8_authority_constants(monkeypatch, manifest):
    """The module also pins the real frozen V8 manifest SHA/commit as a
    defense-in-depth check alongside the DI-injected anchor; synthetic
    fixture manifests must have that pin retargeted to match them."""
    monkeypatch.setattr(readiness, "EXPECTED_V8_PARTITION_MANIFEST_SHA256", manifest["manifest_sha256"])
    monkeypatch.setattr(readiness, "EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT", manifest["partition_implementation_git_commit"])


def test_frozen_sentinel_constants():
    assert readiness.SENTINEL_INDICES == (0, 149, 299)
    assert readiness.SENTINEL_PROBE_START == "2025-12-01"
    assert readiness.SENTINEL_PROBE_END_EXCLUSIVE == "2025-12-08"


def test_stage_invalid_rejected(tmp_path):
    with pytest.raises(readiness.V8CReadinessBlocked) as excinfo:
        readiness._execute_transport_readiness_probe(
            stage="T3", human_authorization_identity="tok",
            partition_manifest_path=tmp_path / "x.json",
            **_base_deps(anchor_reader=lambda head: {}),
        )
    assert excinfo.value.reason == "V8C_READINESS_STAGE_INVALID"


def test_missing_authorization_identity_rejected(tmp_path):
    with pytest.raises(readiness.V8CReadinessBlocked) as excinfo:
        readiness._execute_transport_readiness_probe(
            stage="T1C", human_authorization_identity="",
            partition_manifest_path=tmp_path / "x.json",
            **_base_deps(anchor_reader=lambda head: {}),
        )
    assert excinfo.value.reason == "V8C_READINESS_HUMAN_AUTHORIZATION_IDENTITY_INVALID"


def test_same_authorization_identity_replay_blocks_before_provenance(tmp_path):
    state_root = tmp_path / "state"
    gate_consumption.consume_gate_once(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, readiness.EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        clock=clock_stub, authorization_identity="AUTH-1",
    )

    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called")

    with pytest.raises(readiness.V8CReadinessBlocked) as excinfo:
        readiness._execute_transport_readiness_probe(
            stage="T1C", human_authorization_identity="AUTH-1",
            partition_manifest_path=tmp_path / "unread.json",
            consumption_state_root=state_root,
            git_commit_resolver=forbidden, design_freeze_approval_reader=forbidden,
            frozen_design_object_verifier=forbidden, reviewed_implementation_binder=forbidden,
            classifier_blob_resolver=forbidden, anchor_reader=forbidden,
            opener=forbidden, sleep_fn=lambda s: None, clock=clock_stub,
        )
    assert excinfo.value.reason == "V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_T1C_TRANSPORT_READINESS
    assert not (tmp_path / "unread.json").exists()


def test_classifier_blob_mismatch_blocks_before_gate_consumption(tmp_path):
    state_root = tmp_path / "state"
    with pytest.raises(readiness.V8CReadinessBlocked) as excinfo:
        readiness._execute_transport_readiness_probe(
            stage="T1C", human_authorization_identity="AUTH-1",
            partition_manifest_path=tmp_path / "unread.json",
            **_base_deps(
                consumption_state_root=state_root,
            classifier_blob_resolver=lambda head: "0" * 40, anchor_reader=lambda head: {}),
        )
    assert excinfo.value.reason == "V8C_PRODUCTION_CLASSIFIER_VERSION_MISMATCH"
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, readiness.EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        authorization_identity="AUTH-1",
    ) is False


def test_all_three_sentinels_pass_reports_pass(tmp_path, monkeypatch):
    t0 = _tickers("T0", 300)
    manifest_path, manifest = _t0_manifest(tmp_path, t0)
    _patch_v8_authority_constants(monkeypatch, manifest)
    state_root = tmp_path / "state"

    def fake_opener(request_obj):
        raise AssertionError("real opener must not be invoked in this fake test")

    calls = []

    def fake_probe(ticker, opener, sleep_fn):
        calls.append(ticker)
        return {"result": {}, "audit": {"attempts": 1}}

    import src.v8c_readiness as readiness_module

    original = readiness_module._probe_one_sentinel
    readiness_module._probe_one_sentinel = fake_probe
    try:
        result = readiness._execute_transport_readiness_probe(
            stage="T1C", human_authorization_identity="AUTH-1",
            partition_manifest_path=manifest_path,
            **_base_deps(
                consumption_state_root=state_root,
            anchor_reader=lambda head: _valid_anchor_for(manifest), opener=fake_opener),
        )
    finally:
        readiness_module._probe_one_sentinel = original

    assert result["result"] == "PASS"
    assert result["sentinel_count"] == 3
    assert calls == [t0[0], t0[149], t0[299]]
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, readiness.EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        authorization_identity="AUTH-1",
    ) is True


def test_one_failing_sentinel_reports_block_but_still_consumes_gate(tmp_path, monkeypatch):
    t0 = _tickers("T0", 300)
    manifest_path, manifest = _t0_manifest(tmp_path, t0)
    _patch_v8_authority_constants(monkeypatch, manifest)
    state_root = tmp_path / "state"

    def fake_probe(ticker, opener, sleep_fn):
        if ticker == t0[149]:
            raise RuntimeError("simulated failure")
        return {"result": {}, "audit": {"attempts": 1}}

    import src.v8c_readiness as readiness_module

    original = readiness_module._probe_one_sentinel
    readiness_module._probe_one_sentinel = fake_probe
    try:
        result = readiness._execute_transport_readiness_probe(
            stage="T1C", human_authorization_identity="AUTH-1",
            partition_manifest_path=manifest_path,
            **_base_deps(
                consumption_state_root=state_root,
            anchor_reader=lambda head: _valid_anchor_for(manifest), opener=lambda r: None),
        )
    finally:
        readiness_module._probe_one_sentinel = original

    assert result["result"] == "BLOCK"
    assert result["sentinel_pass_count"] == 2
    assert result["readiness_failure_consumes_acquisition_gate"] is False
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T1C_TRANSPORT_READINESS, readiness.EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        authorization_identity="AUTH-1",
    ) is True
    assert gate_consumption.has_gate_been_consumed(
        state_root, gate_consumption.GATE_T1C_RAW_ACQUISITION, readiness.EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    ) is False


def test_fresh_authorization_permits_recheck_after_block(tmp_path, monkeypatch):
    t0 = _tickers("T0", 300)
    manifest_path, manifest = _t0_manifest(tmp_path, t0)
    _patch_v8_authority_constants(monkeypatch, manifest)
    state_root = tmp_path / "state"

    import src.v8c_readiness as readiness_module

    original = readiness_module._probe_one_sentinel
    readiness_module._probe_one_sentinel = lambda ticker, opener, sleep_fn: {"result": {}, "audit": {"attempts": 1}}
    try:
        readiness._execute_transport_readiness_probe(
            stage="T1C", human_authorization_identity="AUTH-1",
            partition_manifest_path=manifest_path, **_base_deps(
                consumption_state_root=state_root,
            anchor_reader=lambda head: _valid_anchor_for(manifest), opener=lambda r: None),
        )
        # A fresh authorization identity permits a new recheck probe.
        second = readiness._execute_transport_readiness_probe(
            stage="T1C", human_authorization_identity="AUTH-2",
            partition_manifest_path=manifest_path, **_base_deps(
                consumption_state_root=state_root,
            anchor_reader=lambda head: _valid_anchor_for(manifest), opener=lambda r: None),
        )
        assert second["result"] == "PASS"
    finally:
        readiness_module._probe_one_sentinel = original


def test_public_result_never_contains_ticker_identity(tmp_path, monkeypatch):
    t0 = _tickers("SECRET_T0", 300)
    manifest_path, manifest = _t0_manifest(tmp_path, t0)
    _patch_v8_authority_constants(monkeypatch, manifest)
    state_root = tmp_path / "state"

    import src.v8c_readiness as readiness_module

    original = readiness_module._probe_one_sentinel
    readiness_module._probe_one_sentinel = lambda ticker, opener, sleep_fn: {"result": {}, "audit": {"attempts": 1}}
    try:
        result = readiness._execute_transport_readiness_probe(
            stage="T2", human_authorization_identity="AUTH-1",
            partition_manifest_path=manifest_path, **_base_deps(
                consumption_state_root=state_root,
            anchor_reader=lambda head: _valid_anchor_for(manifest), opener=lambda r: None),
        )
    finally:
        readiness_module._probe_one_sentinel = original
    for value in result.values():
        if isinstance(value, str):
            assert "SECRET_T0" not in value
