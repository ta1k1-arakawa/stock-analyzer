from __future__ import annotations

import json
import os
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src import v8_partition as partition
from src import v8b_allocation as allocation
from src import v8b_allocation_verification as verification
from src import v8b_historical_acquisition as acquisition
from src import v8b_trust_pin as trust_pin

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


# ---------------------------------------------------------------------------
# Fake Yahoo transport (mirrors tests/test_v8_historical_acquisition.py)
# ---------------------------------------------------------------------------


def _epoch(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def synthetic_payload(
    ticker: str,
    dates: list[tuple[int, int, int]],
    price: float = 1000.0,
    *,
    bad_row_indices: list[int] | None = None,
) -> bytes:
    timestamps = [_epoch(*d) for d in dates]
    closes = [price] * len(timestamps)
    if bad_row_indices:
        for index in bad_row_indices:
            closes[index] = -1.0
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": timestamps,
        "indicators": {
            "quote": [{
                "open": [price] * len(timestamps),
                "high": [price + 2.0] * len(timestamps),
                "low": [price - 2.0] * len(timestamps),
                "close": closes,
                "volume": [10000.0] * len(timestamps),
            }],
            "adjclose": [{"adjclose": [price] * len(timestamps)}],
        },
        "events": {},
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


class FakeResponse:
    def __init__(self, payload: bytes, url: str, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.url = url

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        pass


class FakeOpener:
    def __init__(self, payload_fn, *, host: str = "query1.finance.yahoo.com") -> None:
        self.payload_fn = payload_fn
        self.host = host
        self.calls: list[str] = []

    def __call__(self, request_obj: Any) -> FakeResponse:
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        payload = self.payload_fn(ticker)
        return FakeResponse(payload, url=f"https://{self.host}/v8/finance/chart/{ticker}.T")


DEFAULT_DATES = [(2016, 4, 1), (2016, 4, 4), (2025, 12, 30)]


def _date_tuples(start: tuple[int, int, int], count: int) -> list[tuple[int, int, int]]:
    start_date = date(*start)
    return [
        ((start_date + timedelta(days=i)).year, (start_date + timedelta(days=i)).month, (start_date + timedelta(days=i)).day)
        for i in range(count)
    ]


def default_opener() -> FakeOpener:
    return FakeOpener(lambda ticker: synthetic_payload(ticker, DEFAULT_DATES))


def clock_stub():
    return datetime(2026, 8, 12, tzinfo=timezone.utc)


def forbidden_opener(*_args, **_kwargs):
    raise AssertionError("Yahoo opener must not run")


# ---------------------------------------------------------------------------
# T1B fixture: private allocation artifact + trust pin
# ---------------------------------------------------------------------------


def _t_spare(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def build_t1b_fixture(tmp_path: Path, *, parent_count: int = 1904, tamper_pin: dict | None = None):
    parent = _t_spare("TS", parent_count)
    artifact = allocation.build_t1b_allocation_artifact(
        parent_t_spare_tickers=parent,
        parent_v8_partition_manifest_sha256="0" * 64,
        parent_v8_partition_implementation_commit=SYNTHETIC_COMMIT,
        parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
        v8b_allocation_implementation_commit=SYNTHETIC_COMMIT,
        clock=clock_stub,
    )
    result = verification.verify_t1b_allocation_artifact(
        artifact=artifact,
        parent_t_spare_tickers=parent,
        t0_tickers=_t_spare("T0", 300),
        old_t1_tickers=_t_spare("OT1", 300),
        t2_tickers=_t_spare("T2X", 300),
        t3_tickers=_t_spare("T3X", 300),
        expected_parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
    )
    pin = trust_pin.build_trust_pin(
        verification_result_summary=result, human_gate="V8B_HUMAN_AUTHORIZE_T1B_PIN", authorization_note="test"
    )
    if tamper_pin:
        pin = {**pin, **tamper_pin}
    artifact_path = tmp_path / "t1b_allocation_artifact.json"
    artifact_path.write_bytes(allocation.canonical_json_bytes(artifact))
    pin_path = tmp_path / "t1b_trust_pin.json"
    pin_path.write_bytes(json.dumps(pin).encode("utf-8"))
    return artifact, pin, artifact_path, pin_path


def t1b_deps(artifact_path: Path, pin_path: Path, opener, **overrides):
    deps = dict(
        output_root=None,  # filled by caller
        block="T1B",
        partition_manifest_path=None,
        t1b_allocation_artifact_path=artifact_path,
        t1b_trust_pin_path=pin_path,
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda commit: {"ok": True},
        implementation_review_reader=lambda commit, reviewed: {"ok": True},
        classifier_blob_resolver=lambda commit: acquisition.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
        zoneinfo_loader=lambda: object(),
        git_anchor_reader=lambda commit: {},
        git_bridge_reader=lambda commit: {},
        opener=opener,
        clock=clock_stub,
        monotonic_clock=lambda: 0.0,
        sleep_fn=lambda _s: None,
    )
    deps.update(overrides)
    return deps


# ---------------------------------------------------------------------------
# T2 fixture: real V8 partition manifest + trusted anchor + OPTION_2 bridge
# ---------------------------------------------------------------------------


def _tickers(start: int) -> list[str]:
    return [f"{code:04d}" for code in range(start, start + 300)]


def write_partition_manifest(path: Path, *, t2=None) -> dict:
    blocks = {
        "T0": _tickers(4000), "T1": _tickers(1000), "T2": list(t2 or _tickers(2000)),
        "T3": _tickers(3000), "T_spare": _tickers(5000),
    }
    manifest = {
        "schema_version": partition.SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "source_snapshot_semantics": partition.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": partition.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": SYNTHETIC_COMMIT,
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


def write_trusted_anchor(path: Path, *, manifest: dict) -> dict:
    anchor = {
        "schema_version": acquisition.TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": manifest["manifest_sha256"],
        "authorized_partition_implementation_git_commit": manifest["partition_implementation_git_commit"],
        "authorization_note": "test-only anchor",
    }
    path.write_bytes(partition.canonical_json_bytes(anchor))
    return anchor


def build_bridge(*, manifest: dict, t2_tickers, tamper: dict | None = None) -> dict:
    bridge = {
        "schema_version": acquisition.T2_AUTHORITY_BRIDGE_SCHEMA_VERSION,
        "study": acquisition.STUDY_NAME,
        "role": "SEALED_HOLDOUT",
        "source_authority": "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY",
        "v8_trust_anchor_git_path": "V8_TRUSTED_PARTITION.json",
        "v8_trust_anchor_git_identity": "f" * 40,
        "authorized_parent_v8_partition_manifest_sha256": manifest["manifest_sha256"],
        "expected_t2_ticker_list_sha256": partition.ticker_list_sha256(t2_tickers),
        "t2_acquired_before_authorized_acquisition": False,
        "t2_research_open_count_before_official_opening": 0,
        "v8b_frozen_design_commit": acquisition.V8B_FROZEN_DESIGN_COMMIT,
        "t2_membership_reassignment": "PROHIBITED",
        "v8_trusted_partition_json_mutated_or_repinned": False,
        "option": "OPTION_2",
        "human_gate": "V8B_T2_AUTHORITY_INTEGRATION_OPTION_2_APPROVED",
        "authorization_note": "test-only bridge",
    }
    if tamper:
        bridge.update(tamper)
    return bridge


def t2_deps(partition_manifest_path: Path, anchor: dict, bridge: dict, opener, **overrides):
    deps = dict(
        output_root=None,
        block="T2",
        partition_manifest_path=partition_manifest_path,
        t1b_allocation_artifact_path=None,
        t1b_trust_pin_path=None,
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda commit: {"ok": True},
        implementation_review_reader=lambda commit, reviewed: {"ok": True},
        classifier_blob_resolver=lambda commit: acquisition.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
        zoneinfo_loader=lambda: object(),
        git_anchor_reader=lambda commit: anchor,
        git_bridge_reader=lambda commit: bridge,
        opener=opener,
        clock=clock_stub,
        monotonic_clock=lambda: 0.0,
        sleep_fn=lambda _s: None,
    )
    deps.update(overrides)
    return deps


def run(deps: dict, output_root: Path):
    deps = dict(deps)
    deps["output_root"] = output_root
    return acquisition._acquire_production_v8b_historical_block_bundle_with_dependencies(**deps)


# ---------------------------------------------------------------------------
# Public boundary shape
# ---------------------------------------------------------------------------


def test_public_production_signature_has_only_required_inputs():
    import inspect

    assert tuple(inspect.signature(acquisition.acquire_v8b_historical_block_bundle).parameters) == (
        "output_root", "block", "partition_manifest_path", "t1b_allocation_artifact_path", "t1b_trust_pin_path",
    )


@pytest.mark.parametrize("block", acquisition.PROHIBITED_ACQUISITION_BLOCKS)
def test_all_prohibited_blocks_rejected(tmp_path, block):
    deps = t1b_deps(tmp_path / "a.json", tmp_path / "p.json", forbidden_opener, block=block)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_BLOCK_ACQUISITION_PROHIBITED:" + block


# ---------------------------------------------------------------------------
# Pre-network ordering: zero opener calls on every pre-network failure
# ---------------------------------------------------------------------------


def test_step1_dirty_git_provenance_blocks_before_network(tmp_path):
    artifact_path, pin_path = tmp_path / "a.json", tmp_path / "p.json"
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)

    def dirty_resolver():
        raise acquisition.V8BHistoricalAcquisitionBlocked("PRODUCTION_GIT_WORKTREE_DIRTY")

    deps = t1b_deps(artifact_path, pin_path, forbidden_opener, git_commit_resolver=dirty_resolver)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "PRODUCTION_GIT_WORKTREE_DIRTY"


def test_step2_freeze_approval_failure_blocks_before_network(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)

    def failing_freeze_reader(commit):
        raise acquisition.V8BHistoricalAcquisitionBlocked("V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED")

    deps = t1b_deps(artifact_path, pin_path, forbidden_opener, design_freeze_approval_reader=failing_freeze_reader)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED"


def test_step3_missing_implementation_review_blocks_before_network(tmp_path):
    """The real V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json does not exist yet
    in this repository; production must fail closed here today."""
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition.acquire_v8b_historical_block_bundle(
            output_root=tmp_path / "private",
            block="T1B",
            t1b_allocation_artifact_path=artifact_path,
            t1b_trust_pin_path=pin_path,
        )
    assert excinfo.value.reason in {
        "PRODUCTION_GIT_WORKTREE_DIRTY",
        "PRODUCTION_GIT_HEAD_NOT_ORIGIN",
        "V8B_DESIGN_FREEZE_APPROVAL_READ_FAILED",
        "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING",
    }


def test_step4_classifier_mismatch_blocks_before_network(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    deps = t1b_deps(artifact_path, pin_path, forbidden_opener, classifier_blob_resolver=lambda commit: "0" * 40)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_PRODUCTION_CLASSIFIER_VERSION_MISMATCH"


def test_step4_classifier_match_passes_check():
    acquisition.verify_classifier_blob(acquisition.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA)


def test_step5_zoneinfo_unavailable_blocks_before_network(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)

    def failing_zoneinfo():
        raise LookupError("no tzdata")

    deps = t1b_deps(artifact_path, pin_path, forbidden_opener, zoneinfo_loader=failing_zoneinfo)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_ASIA_TOKYO_ZONEINFO_UNAVAILABLE"


def test_step5_zoneinfo_available_passes_check():
    acquisition.verify_asia_tokyo_zoneinfo_available(lambda: object())


def test_step6_t1b_trust_pin_not_authorized_blocks(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    not_authorized_pin = {field: None for field in trust_pin.TRUST_PIN_FIELDS} | {
        "schema_version": trust_pin.SCHEMA_VERSION,
        "study_name": trust_pin.STUDY_NAME,
        "artifact_role": trust_pin.ARTIFACT_ROLE,
        "logical_block": trust_pin.LOGICAL_BLOCK,
        "authorization_status": "NOT_AUTHORIZED",
    }
    pin_path.write_bytes(json.dumps(not_authorized_pin).encode("utf-8"))
    deps = t1b_deps(artifact_path, pin_path, forbidden_opener)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_TRUST_PIN_NOT_AUTHORIZED"


def test_step6_t1b_trust_pin_artifact_mismatch_blocks(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(
        tmp_path, tamper_pin={"authorized_allocation_artifact_self_hash": "9" * 64}
    )
    deps = t1b_deps(artifact_path, pin_path, forbidden_opener)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_TRUST_PIN_ALLOCATION_ARTIFACT_MISMATCH"


def test_step6_t2_bridge_ticker_hash_mismatch_blocks(tmp_path):
    manifest_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(manifest_path)
    anchor = write_trusted_anchor(tmp_path / "anchor.json", manifest=manifest)
    bridge = build_bridge(manifest=manifest, t2_tickers=_tickers(2000), tamper={"expected_t2_ticker_list_sha256": "f" * 64})
    deps = t2_deps(manifest_path, anchor, bridge, forbidden_opener)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_T2_AUTHORITY_BRIDGE_TICKER_LIST_SHA_MISMATCH"


def test_step6_t2_bridge_cannot_be_bypassed_wrong_manifest_binding(tmp_path):
    manifest_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(manifest_path)
    anchor = write_trusted_anchor(tmp_path / "anchor.json", manifest=manifest)
    bridge = build_bridge(
        manifest=manifest, t2_tickers=_tickers(2000), tamper={"authorized_parent_v8_partition_manifest_sha256": "e" * 64}
    )
    deps = t2_deps(manifest_path, anchor, bridge, forbidden_opener)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_T2_AUTHORITY_BRIDGE_MANIFEST_SHA_MISMATCH"


def test_step6_t2_anchor_not_authorized_blocks(tmp_path):
    manifest_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(manifest_path)
    anchor = write_trusted_anchor(tmp_path / "anchor.json", manifest=manifest)
    anchor = {**anchor, "authorization_status": "NOT_AUTHORIZED"}
    bridge = build_bridge(manifest=manifest, t2_tickers=_tickers(2000))
    deps = t2_deps(manifest_path, anchor, bridge, forbidden_opener)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "TRUSTED_PARTITION_NOT_AUTHORIZED"


def test_step8_output_path_inside_repository_blocks(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    deps = t1b_deps(artifact_path, pin_path, forbidden_opener)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, ROOT / "acquisitions_should_not_be_written_here")
    assert excinfo.value.reason == "OUTPUT_PATH_INSIDE_SOURCE_REPOSITORY"


# ---------------------------------------------------------------------------
# Fake success: exactly 300 opener calls, atomic publication
# ---------------------------------------------------------------------------


def test_t1b_fake_success_exactly_300_opener_calls(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    opener = default_opener()
    deps = t1b_deps(artifact_path, pin_path, opener)
    manifest = run(deps, tmp_path / "private")
    assert len(opener.calls) == 300
    assert manifest["request_count"] == 300
    assert manifest["success_transport_count"] == 300
    assert manifest["ticker_count"] == 300
    assert manifest["block"] == "T1B"
    assert manifest["role"] == "VALIDATION"
    assert manifest["sealed"] is False
    assert manifest["retry_count"] == 0
    assert manifest["authority_chain"] == "V8B_SUCCESSOR_ALLOCATION_AUTHORITY"
    assert manifest["v8b_frozen_design_commit"] == acquisition.V8B_FROZEN_DESIGN_COMMIT
    assert manifest["canonical_parser_classifier_blob_sha"] == acquisition.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA
    reread = acquisition.read_acquisition_manifest(tmp_path / "private", "T1B")
    assert reread == manifest


def test_t2_fake_success_exactly_300_opener_calls_and_sealed(tmp_path):
    manifest_path = tmp_path / "partition.json"
    manifest_fixture = write_partition_manifest(manifest_path)
    anchor = write_trusted_anchor(tmp_path / "anchor.json", manifest=manifest_fixture)
    bridge = build_bridge(manifest=manifest_fixture, t2_tickers=manifest_fixture["block_assignments"]["T2"])
    opener = default_opener()
    deps = t2_deps(manifest_path, anchor, bridge, opener)
    manifest = run(deps, tmp_path / "private")
    assert len(opener.calls) == 300
    assert manifest["block"] == "T2"
    assert manifest["role"] == "SEALED_HOLDOUT"
    assert manifest["sealed"] is True
    assert manifest["authority_chain"] == "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY_OPTION_2_BRIDGE"
    sealed_path = tmp_path / "private" / acquisition.ACQUISITIONS_DIRNAME / "T2" / acquisition.SEALED_FILENAME
    assert sealed_path.exists()


def test_t1b_atomic_no_partial_publication_on_mid_loop_block(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)

    def failing_payload(ticker: str) -> bytes:
        if ticker == artifact["t1b_tickers"][5]:
            return synthetic_payload(ticker, DEFAULT_DATES, bad_row_indices=[0, 1, 2])
        return synthetic_payload(ticker, DEFAULT_DATES)

    opener = FakeOpener(failing_payload)
    deps = t1b_deps(artifact_path, pin_path, opener)
    output_root = tmp_path / "private"
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked):
        run(deps, output_root)
    final_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B"
    assert not final_dir.exists()
    staging_entries = list((output_root / acquisition.ACQUISITIONS_DIRNAME).iterdir())
    assert staging_entries == []


# ---------------------------------------------------------------------------
# F1_C1 malformed-OHLCV thresholds: exact 1/252 and consecutive=1
# ---------------------------------------------------------------------------


PLACEHOLDER_TICKER = "9999"


def _observations(count_total: int, invalid_indices: list[int]) -> tuple[list[dict], list[dict]]:
    dates = _date_tuples((2020, 1, 1), count_total)
    valid, invalid = [], []
    for index, d in enumerate(dates):
        entry_date = date(*d).isoformat()
        if index in invalid_indices:
            invalid.append({"ticker": PLACEHOLDER_TICKER, "trading_date": entry_date, "reason": "NONFINITE_CLOSE"})
        else:
            valid.append({"ticker": PLACEHOLDER_TICKER, "trading_date": entry_date})
    return valid, invalid


def test_fraction_exactly_at_threshold_passes():
    # 252 observations, 1 invalid: 1*252 <= 252 -> PASS
    valid, invalid = _observations(252, [0])
    acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)


def test_fraction_one_beyond_threshold_blocks():
    # 251 observations, 1 invalid: 1*252 > 251 -> BLOCK
    valid, invalid = _observations(251, [0])
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


def test_consecutive_run_of_1_passes():
    valid, invalid = _observations(260, [10])
    acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)


def test_consecutive_run_of_2_blocks():
    # 600 observations keeps the fraction gate satisfied (2*252 <= 600) so
    # this isolates the consecutive-run gate specifically.
    valid, invalid = _observations(600, [10, 11])
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:CONSECUTIVE_EXCEEDED"


def test_mandatory_2018_test_year_checked_even_though_outside_calibration_window():
    """§7.6: production years are 2018-2025 (full P_hist), not the
    calibration-evidence 2019-2025 span. 2018 has only 10 returned
    observations with 2 non-consecutive invalid rows -- a fraction breach
    confined entirely to that one year's small window -- while every other
    year (and the full series) stays comfortably within tolerance. This
    must still BLOCK on the 2018-specific per-year check."""
    other_years_valid = [
        {"ticker": "9999", "trading_date": f"{year}-{offset // 28 + 1:02d}-{offset % 28 + 1:02d}"}
        for year in range(2016, 2026) if year != 2018
        for offset in range(60)
    ]
    year_2018 = [
        {"ticker": "9999", "trading_date": f"2018-01-{day:02d}"} for day in range(1, 11)
    ]
    invalid_2018 = [dict(year_2018[0], reason="NONFINITE_CLOSE"), dict(year_2018[5], reason="NONFINITE_CLOSE")]
    valid_2018 = [row for index, row in enumerate(year_2018) if index not in (0, 5)]

    full_valid = other_years_valid + valid_2018
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(full_valid, invalid_2018)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:TEST_YEAR_FRACTION_EXCEEDED"


def test_full_p_hist_check_is_independent_of_per_year_checks():
    """A full-series fraction breach must BLOCK even when every individual
    test year individually stays within tolerance (small per-year windows
    can mask an aggregate breach if only checked per-year)."""
    # 8 years * 30 observations = 240 total observations, all valid except one.
    valid = []
    for year in acquisition.MALFORMED_OHLCV_TEST_YEARS:
        for day in range(1, 31):
            valid.append({"ticker": "X", "trading_date": f"{year}-01-{day:02d}"})
    invalid = [{"ticker": "X", "trading_date": "2018-02-01", "reason": "NONFINITE_CLOSE"}]
    # 240 total, 1 invalid: 1*252 > 240 -> BLOCK on full series even though
    # this single invalid row is a tiny fraction of any one test year.
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


def test_policy_metadata_is_exact_and_never_float():
    metadata = acquisition._malformed_ohlcv_policy_metadata()
    assert metadata["invalid_fraction_numerator"] == 1
    assert metadata["invalid_fraction_denominator"] == 252
    assert isinstance(metadata["invalid_fraction_numerator"], int)
    assert isinstance(metadata["invalid_fraction_denominator"], int)
    assert metadata["max_consecutive_invalid_returned_rows"] == 1
    assert metadata["test_years"] == list(range(2018, 2026))
    assert metadata["policy_name"] == "POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE"


# ---------------------------------------------------------------------------
# No retry
# ---------------------------------------------------------------------------


def test_retry_count_is_repository_fixed_zero_not_caller_overridable():
    import inspect

    assert acquisition.RETRY_COUNT == 0
    assert "retry_count" not in inspect.signature(acquisition.acquire_v8b_historical_block_bundle).parameters


def test_single_ticker_failure_aborts_whole_acquisition_no_second_attempt(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    calls: list[str] = []

    def failing_payload(ticker: str) -> bytes:
        calls.append(ticker)
        if ticker == artifact["t1b_tickers"][0]:
            return synthetic_payload(ticker, DEFAULT_DATES, bad_row_indices=[0, 1])
        return synthetic_payload(ticker, DEFAULT_DATES)

    opener = FakeOpener(failing_payload)
    deps = t1b_deps(artifact_path, pin_path, opener)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked):
        run(deps, tmp_path / "private")
    # Only the first ticker was ever requested -- no automatic retry, and
    # the loop stops immediately rather than continuing past the BLOCK.
    assert calls == [artifact["t1b_tickers"][0]]


# ---------------------------------------------------------------------------
# No ticker/date/path leakage
# ---------------------------------------------------------------------------


def test_malformed_reason_strings_never_contain_a_ticker():
    valid, invalid = _observations(251, [0])
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert PLACEHOLDER_TICKER not in excinfo.value.reason
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


def test_prohibited_block_reason_never_leaks_a_file_path(tmp_path):
    deps = t1b_deps(tmp_path / "a.json", tmp_path / "p.json", forbidden_opener, block="T3")
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert str(tmp_path) not in excinfo.value.reason


# ---------------------------------------------------------------------------
# Malicious GIT_* environment isolation
# ---------------------------------------------------------------------------


def _init_bogus_git_repo(root: Path, *, classifier_content: bytes) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "src").mkdir()
    (root / "src" / "v7_yahoo_collector.py").write_bytes(classifier_content)
    (root / acquisition.DESIGN_FREEZE_APPROVAL_GIT_PATH).write_text(
        json.dumps({"approval_status": "APPROVED", "forged": True})
    )
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=a@b.c", "-c", "user.name=x", "commit", "-q", "-m", "bogus"],
        check=True,
    )


def test_malicious_git_dir_env_cannot_redirect_classifier_blob_resolution(monkeypatch, tmp_path):
    """A malicious GIT_DIR/GIT_WORK_TREE/GIT_INDEX_FILE must not be able to
    redirect this module's git-based reads to an attacker-controlled
    repository. Empirically, plain `git -C <real_root> ...` alone does NOT
    protect against this (GIT_DIR overrides -C's directory discovery) --
    this test proves the module's explicit env sanitization does. The
    forged repo's classifier blob is deliberately DIFFERENT bogus content
    (never the real pinned bytes), so if isolation ever regresses, this
    test fails loudly with a wrong-but-resolved hash rather than silently
    passing via a coincidental BLOCK."""
    bogus = tmp_path / "not_a_repo"
    bogus.mkdir()
    _init_bogus_git_repo(bogus, classifier_content=b"FORGED CLASSIFIER CONTENT")
    monkeypatch.setenv("GIT_DIR", str(bogus / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(bogus))
    monkeypatch.setenv("GIT_INDEX_FILE", str(bogus / ".git" / "index"))

    real_commit = "28e281c3ee30d6b4c2f981c5da3ddc983c09724d"
    blob_sha = acquisition._resolve_classifier_blob_sha_from_verified_head(real_commit)
    assert blob_sha == acquisition.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA


def test_malicious_git_dir_env_cannot_redirect_design_freeze_approval_read(monkeypatch, tmp_path):
    import subprocess

    real_head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()

    bogus = tmp_path / "not_a_repo2"
    bogus.mkdir()
    _init_bogus_git_repo(bogus, classifier_content=b"irrelevant")
    monkeypatch.setenv("GIT_DIR", str(bogus / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(bogus))

    raw = acquisition._git_show_bytes(
        real_head, acquisition.DESIGN_FREEZE_APPROVAL_GIT_PATH, read_failed_reason="READ_FAILED"
    )
    parsed = json.loads(raw)
    assert parsed.get("forged") is not True
    assert parsed.get("frozen_design_git_commit") == acquisition.V8B_FROZEN_DESIGN_COMMIT


def test_isolated_git_subprocess_env_strips_all_blocklisted_variables(monkeypatch):
    for key in acquisition._ISOLATED_GIT_ENV_BLOCKLIST:
        monkeypatch.setenv(key, "malicious-value")
    env = acquisition._isolated_git_subprocess_env()
    assert not (set(env) & set(acquisition._ISOLATED_GIT_ENV_BLOCKLIST))


def test_ambient_git_environment_context_manager_restores_prior_values(monkeypatch):
    monkeypatch.setenv("GIT_DIR", "original-value")
    with acquisition._isolated_ambient_git_environment():
        assert "GIT_DIR" not in os.environ
    assert os.environ["GIT_DIR"] == "original-value"


# ---------------------------------------------------------------------------
# Public production APIs expose no unsafe injection parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "forbidden",
    (
        "repository_root", "opener", "clock", "monotonic_clock", "sleep_fn",
        "git_commit_resolver", "classifier_blob_resolver", "zoneinfo_loader",
        "request_start", "request_end_exclusive", "trusted_partition_path",
    ),
)
def test_public_boundary_rejects_all_dependency_overrides(tmp_path, forbidden):
    kwargs = {
        "output_root": tmp_path / "private",
        "block": "T1B",
        forbidden: "override",
    }
    with pytest.raises(TypeError):
        acquisition.acquire_v8b_historical_block_bundle(**kwargs)


def test_sealed_holdout_guard_denies_before_authorization(tmp_path):
    manifest_path = tmp_path / "partition.json"
    manifest_fixture = write_partition_manifest(manifest_path)
    anchor = write_trusted_anchor(tmp_path / "anchor.json", manifest=manifest_fixture)
    bridge = build_bridge(manifest=manifest_fixture, t2_tickers=manifest_fixture["block_assignments"]["T2"])
    opener = default_opener()
    deps = t2_deps(manifest_path, anchor, bridge, opener)
    manifest = run(deps, tmp_path / "private")
    with pytest.raises(acquisition.V8BSealedHoldoutBlocked) as excinfo:
        acquisition.open_for_feature_generation(manifest)
    assert excinfo.value.reason == "SEALED_HOLDOUT_ACCESS_DENIED:feature_generation"


def test_t1b_guard_denies_before_authorization(tmp_path):
    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    opener = default_opener()
    deps = t1b_deps(artifact_path, pin_path, opener)
    manifest = run(deps, tmp_path / "private")
    with pytest.raises(acquisition.V8BSealedHoldoutBlocked) as excinfo:
        acquisition.open_for_validation(manifest)
    assert excinfo.value.reason == "RESEARCH_ACCESS_NOT_AUTHORIZED:validation"


# ---------------------------------------------------------------------------
# §12.6 raw acquisition artifact verification
# ---------------------------------------------------------------------------


def test_artifact_verification_pass_on_honest_t1b_bundle(tmp_path):
    from src import v8b_acquisition_artifact_verification as artifact_verification

    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    opener = default_opener()
    deps = t1b_deps(artifact_path, pin_path, opener)
    output_root = tmp_path / "private"
    run(deps, output_root)
    result = artifact_verification.verify_acquisition_artifact(
        output_root, "T1B",
        expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
        expected_reviewed_production_implementation_commit=SYNTHETIC_COMMIT,
        expected_authority_chain="V8B_SUCCESSOR_ALLOCATION_AUTHORITY",
    )
    assert result["result"] == "PASS"
    assert result["payload_manifest_record_count"] == 300


def test_artifact_verification_detects_missing_payload_file(tmp_path):
    from src import v8b_acquisition_artifact_verification as artifact_verification

    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    opener = default_opener()
    deps = t1b_deps(artifact_path, pin_path, opener)
    output_root = tmp_path / "private"
    run(deps, output_root)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    victim = next(raw_dir.iterdir())
    victim.unlink()
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(
            output_root, "T1B",
            expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
            expected_reviewed_production_implementation_commit=SYNTHETIC_COMMIT,
            expected_authority_chain="V8B_SUCCESSOR_ALLOCATION_AUTHORITY",
        )
    assert excinfo.value.reason == "RAW_PAYLOAD_MISSING"


def test_artifact_verification_detects_extra_payload_file(tmp_path):
    from src import v8b_acquisition_artifact_verification as artifact_verification

    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    opener = default_opener()
    deps = t1b_deps(artifact_path, pin_path, opener)
    output_root = tmp_path / "private"
    run(deps, output_root)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    (raw_dir / "EXTRA9999.json").write_bytes(b"{}")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(
            output_root, "T1B",
            expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
            expected_reviewed_production_implementation_commit=SYNTHETIC_COMMIT,
            expected_authority_chain="V8B_SUCCESSOR_ALLOCATION_AUTHORITY",
        )
    assert excinfo.value.reason == "RAW_PAYLOAD_UNEXPECTED_EXTRA"


def test_artifact_verification_detects_modified_payload_bytes(tmp_path):
    from src import v8b_acquisition_artifact_verification as artifact_verification

    artifact, pin, artifact_path, pin_path = build_t1b_fixture(tmp_path)
    opener = default_opener()
    deps = t1b_deps(artifact_path, pin_path, opener)
    output_root = tmp_path / "private"
    run(deps, output_root)
    raw_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B" / acquisition.RAW_DIRNAME
    victim = next(raw_dir.iterdir())
    original = victim.read_bytes()
    victim.write_bytes(original + b"tampered")
    with pytest.raises(artifact_verification.V8BAcquisitionArtifactVerificationBlocked) as excinfo:
        artifact_verification.verify_acquisition_artifact(
            output_root, "T1B",
            expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
            expected_reviewed_production_implementation_commit=SYNTHETIC_COMMIT,
            expected_authority_chain="V8B_SUCCESSOR_ALLOCATION_AUTHORITY",
        )
    assert excinfo.value.reason == "RAW_PAYLOAD_BYTE_COUNT_MISMATCH"


# ---------------------------------------------------------------------------
# §12.4 T2 reuse-conditions recheck
# ---------------------------------------------------------------------------


def test_t2_reuse_recheck_pass():
    from src import v8b_t2_reuse_recheck as recheck

    result = recheck.recheck_t2_reuse_conditions({
        "t2_acquired": False,
        "t2_opened": False,
        "t2_ticker_identities_exposed_to_human_public_research_loop": False,
        "t2_market_data_raw_ohlcv_feature_outcome_research_exposure": False,
        "t2_universe_definition_unchanged": True,
        "t2_partition_algorithm_unchanged": True,
        "t2_v8b_f1_c1_policy_fixed": True,
    })
    assert result == {"result": "PASS", "block": "T2"}


def test_t2_reuse_recheck_blocks_on_missing_field():
    from src import v8b_t2_reuse_recheck as recheck

    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.recheck_t2_reuse_conditions({"t2_acquired": False})
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_BLOCKED:MISSING_SAFE_METADATA"


def test_t2_reuse_recheck_blocks_on_already_acquired():
    from src import v8b_t2_reuse_recheck as recheck

    metadata = {
        "t2_acquired": True,
        "t2_opened": False,
        "t2_ticker_identities_exposed_to_human_public_research_loop": False,
        "t2_market_data_raw_ohlcv_feature_outcome_research_exposure": False,
        "t2_universe_definition_unchanged": True,
        "t2_partition_algorithm_unchanged": True,
        "t2_v8b_f1_c1_policy_fixed": True,
    }
    with pytest.raises(recheck.V8BT2PreservationRecheckBlocked) as excinfo:
        recheck.recheck_t2_reuse_conditions(metadata)
    assert excinfo.value.reason == "V8B_T2_PRESERVATION_RECHECK_BLOCKED:T2_ACQUIRED"


def test_t2_reuse_recheck_module_defines_no_fallback_substitution():
    from src import v8b_t2_reuse_recheck as recheck

    assert not any("spare" in name.lower() or "t3" in name.lower() for name in recheck.__all__)
