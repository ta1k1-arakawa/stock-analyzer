from __future__ import annotations

import json
import tempfile
import urllib.error
import urllib.request
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src import v8c_historical_acquisition as acquisition
from src import v8c_human_gate_consumption as gate_consumption
from src.v8c_production_provenance import CANONICAL_PARSER_CLASSIFIER_BLOB_SHA

SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_REVIEWED_COMMIT = "b" * 40


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def _epoch(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


DEFAULT_DATES = [(2016, 4, 1), (2016, 4, 4), (2025, 12, 30)]


def synthetic_payload(ticker: str, dates=DEFAULT_DATES, price: float = 1000.0) -> bytes:
    timestamps = [_epoch(*d) for d in dates]
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": timestamps,
        "indicators": {
            "quote": [{
                "open": [price] * len(timestamps), "high": [price + 2.0] * len(timestamps),
                "low": [price - 2.0] * len(timestamps), "close": [price] * len(timestamps),
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


class FailThenSucceedOpener:
    """Fails the first N calls per ticker with a retryable HTTP error, then succeeds."""

    def __init__(self, fail_count: int, payload_fn=None, error_code: int = 503) -> None:
        self.fail_count = fail_count
        self.payload_fn = payload_fn or (lambda ticker: synthetic_payload(ticker))
        self.error_code = error_code
        self.calls_by_ticker: dict[str, int] = {}
        self.calls: list[str] = []

    def __call__(self, request_obj: Any) -> FakeResponse:
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        self.calls_by_ticker[ticker] = self.calls_by_ticker.get(ticker, 0) + 1
        if self.calls_by_ticker[ticker] <= self.fail_count:
            raise urllib.error.HTTPError(request_obj.full_url, self.error_code, "retry me", {}, None)
        payload = self.payload_fn(ticker)
        return FakeResponse(payload, url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")


class AlwaysFailOpener:
    def __init__(self, code: int = 403) -> None:
        self.code = code
        self.calls: list[str] = []

    def __call__(self, request_obj: Any) -> FakeResponse:
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        raise urllib.error.HTTPError(request_obj.full_url, self.code, "nope", {}, None)


def forbidden_opener(*_args, **_kwargs):
    raise AssertionError("Yahoo opener must not run")


def clock_stub():
    return datetime(2026, 8, 14, tzinfo=timezone.utc)


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def _fresh_state_root(tmp_path=None) -> Path:
    return Path(tempfile.gettempdir()) / ("v8c_acq_gate_state-" + uuid.uuid4().hex)


def run_low_level(**overrides):
    defaults = dict(
        output_root=None,
        repository_root=acquisition.CANONICAL_REPOSITORY_ROOT,
        block="T1C",
        tickers=_tickers("FAKE", 3),
        authority_binding={"authorized_allocation_artifact_self_hash": "0" * 64},
        implementation_git_commit=SYNTHETIC_REVIEWED_COMMIT,
        classifier_blob_sha=CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
        opener=lambda req: (_ for _ in ()).throw(AssertionError("opener not set")),
        clock=clock_stub,
        consumption_gate=gate_consumption.GATE_T1C_RAW_ACQUISITION,
        consumption_state_root=_fresh_state_root(),
        sleep_fn=lambda s: None,
    )
    defaults.update(overrides)
    return acquisition._acquire_v8c_block_bundle_with_validated_inputs(**defaults)


# ---------------------------------------------------------------------------
# Confirmation / gate ordering (public production seam)
# ---------------------------------------------------------------------------


def test_confirmation_literals_are_frozen():
    assert acquisition.T1C_ACQUISITION_CONFIRMATION == "V8C_PRODUCTION_ACQUIRE_T1C"
    assert acquisition.T2_ACQUISITION_CONFIRMATION == "V8C_PRODUCTION_ACQUIRE_T2"


def test_wrong_confirmation_blocks_before_any_dependency_call(tmp_path):
    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called")

    with pytest.raises(acquisition.V8CHistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_production_v8c_historical_block_bundle_with_dependencies(
            output_root=tmp_path, block="T1C", confirmation="wrong",
            partition_manifest_path=None, t1c_allocation_artifact_path=None,
            git_commit_resolver=forbidden, design_freeze_approval_reader=forbidden,
            frozen_design_object_verifier=forbidden, reviewed_implementation_binder=forbidden,
            classifier_blob_resolver=forbidden, anchor_reader=forbidden, bridge_reader=forbidden,
            bridge_review_reader=forbidden, t2_preservation_recheck_resolver=forbidden,
            t1c_trust_pin_reader=forbidden, trust_pin_review_reader=forbidden,
            opener=forbidden_opener, clock=clock_stub, consumption_state_root=_fresh_state_root(),
        )
    assert excinfo.value.reason == "V8C_ACQUISITION_CONFIRMATION_INVALID"


def test_gate_already_consumed_blocks_before_provenance(tmp_path):
    state_root = _fresh_state_root()
    gate_consumption.consume_gate_once(state_root, gate_consumption.GATE_T1C_RAW_ACQUISITION, acquisition.V8C_FROZEN_DESIGN_COMMIT, clock=clock_stub)

    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called")

    with pytest.raises(acquisition.V8CHistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_production_v8c_historical_block_bundle_with_dependencies(
            output_root=tmp_path, block="T1C", confirmation=acquisition.T1C_ACQUISITION_CONFIRMATION,
            partition_manifest_path=None, t1c_allocation_artifact_path=None,
            git_commit_resolver=forbidden, design_freeze_approval_reader=forbidden,
            frozen_design_object_verifier=forbidden, reviewed_implementation_binder=forbidden,
            classifier_blob_resolver=forbidden, anchor_reader=forbidden, bridge_reader=forbidden,
            bridge_review_reader=forbidden, t2_preservation_recheck_resolver=forbidden,
            t1c_trust_pin_reader=forbidden, trust_pin_review_reader=forbidden,
            opener=forbidden_opener, clock=clock_stub, consumption_state_root=state_root,
        )
    assert excinfo.value.reason == "V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_T1C_RAW_ACQUISITION


def test_classifier_blob_mismatch_blocks_before_gate_consumption(tmp_path):
    state_root = _fresh_state_root()
    with pytest.raises(acquisition.V8CHistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_production_v8c_historical_block_bundle_with_dependencies(
            output_root=tmp_path, block="T1C", confirmation=acquisition.T1C_ACQUISITION_CONFIRMATION,
            partition_manifest_path=None, t1c_allocation_artifact_path=None,
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            classifier_blob_resolver=lambda head: "0" * 40,
            anchor_reader=lambda head: {}, bridge_reader=lambda head: {}, bridge_review_reader=lambda head, blob: {},
            t2_preservation_recheck_resolver=lambda: {"result": "PASS"},
            t1c_trust_pin_reader=lambda head: {}, trust_pin_review_reader=lambda head, h, g: {},
            opener=forbidden_opener, clock=clock_stub, consumption_state_root=state_root,
        )
    assert excinfo.value.reason == "V8C_PRODUCTION_CLASSIFIER_VERSION_MISMATCH"
    assert gate_consumption.has_gate_been_consumed(state_root, gate_consumption.GATE_T1C_RAW_ACQUISITION, acquisition.V8C_FROZEN_DESIGN_COMMIT) is False


# ---------------------------------------------------------------------------
# Low-level per-ticker loop: gate consumption boundary, retry, cleanup
# ---------------------------------------------------------------------------


def test_gate_consumed_exactly_at_first_opener_invocation(tmp_path):
    state_root = _fresh_state_root()
    opener = FailThenSucceedOpener(fail_count=0)

    assert gate_consumption.has_gate_been_consumed(state_root, gate_consumption.GATE_T1C_RAW_ACQUISITION, acquisition.V8C_FROZEN_DESIGN_COMMIT) is False
    run_low_level(output_root=tmp_path, opener=opener, consumption_state_root=state_root)
    assert gate_consumption.has_gate_been_consumed(state_root, gate_consumption.GATE_T1C_RAW_ACQUISITION, acquisition.V8C_FROZEN_DESIGN_COMMIT) is True
    assert len(opener.calls) == 3  # one real opener call per ticker, no retries needed


def test_no_gate_consumption_for_ticker_not_canonical(tmp_path):
    state_root = _fresh_state_root()
    with pytest.raises(acquisition.V8CHistoricalAcquisitionBlocked) as excinfo:
        run_low_level(output_root=tmp_path, tickers=["not.canonical.T"], opener=forbidden_opener, consumption_state_root=state_root)
    assert excinfo.value.reason == "V8C_TICKER_NOT_CANONICAL"
    assert gate_consumption.has_gate_been_consumed(state_root, gate_consumption.GATE_T1C_RAW_ACQUISITION, acquisition.V8C_FROZEN_DESIGN_COMMIT) is False


def test_no_gate_consumption_when_output_path_invalid(tmp_path):
    state_root = _fresh_state_root()
    with pytest.raises(acquisition.V8CHistoricalAcquisitionBlocked) as excinfo:
        run_low_level(output_root="relative/inside/repo", opener=forbidden_opener, consumption_state_root=state_root)
    assert gate_consumption.has_gate_been_consumed(state_root, gate_consumption.GATE_T1C_RAW_ACQUISITION, acquisition.V8C_FROZEN_DESIGN_COMMIT) is False


def test_retry_succeeds_after_transient_failure_and_records_audit(tmp_path):
    state_root = _fresh_state_root()
    opener = FailThenSucceedOpener(fail_count=1)  # first attempt per ticker fails, second succeeds
    manifest = run_low_level(output_root=tmp_path, opener=opener, consumption_state_root=state_root, sleep_fn=lambda s: None)
    assert manifest["total_retry_count"] == 3  # one retry per ticker * 3 tickers
    assert manifest["total_request_attempts"] == 3 + 3
    for entry in manifest["payload_manifest"]:
        assert entry["attempts"] == 2
        assert entry["retry_count"] == 1
    assert manifest["retry_audit_all_intermediate_failures_retryable"] is True


def test_nonretryable_failure_terminal_no_retry_and_atomic_cleanup(tmp_path):
    state_root = _fresh_state_root()
    opener = AlwaysFailOpener(code=403)  # nonretryable
    with pytest.raises(acquisition.V8CHistoricalAcquisitionBlocked) as excinfo:
        run_low_level(output_root=tmp_path, opener=opener, consumption_state_root=state_root)
    assert excinfo.value.reason == "TICKER_FETCH_BLOCKED:HTTP_403"
    assert excinfo.value.authorization_consumed is True
    assert len(opener.calls) == 1  # nonretryable: exactly one attempt, no retry
    # Atomic cleanup: no partial acquisition directory left visible.
    acquisitions_root = tmp_path / acquisition.ACQUISITIONS_DIRNAME
    if acquisitions_root.exists():
        remaining = list(acquisitions_root.iterdir())
        assert all(not entry.name.startswith("T1C.staging-") for entry in remaining)
        assert not (acquisitions_root / "T1C").exists()


def test_retryable_failure_exhausts_all_attempts_then_terminal(tmp_path):
    state_root = _fresh_state_root()
    opener = AlwaysFailOpener(code=503)  # retryable, but never recovers
    with pytest.raises(acquisition.V8CHistoricalAcquisitionBlocked) as excinfo:
        run_low_level(output_root=tmp_path, opener=opener, consumption_state_root=state_root, sleep_fn=lambda s: None)
    assert excinfo.value.reason == "TICKER_FETCH_BLOCKED:HTTP_503"
    assert len(opener.calls) == 3  # exactly MAXIMUM_ATTEMPTS_PER_TICKER for the first ticker


def test_data_quality_gate_failure_blocks_whole_acquisition(tmp_path):
    state_root = _fresh_state_root()
    bad_dates = [(2016, 4, 1), (2016, 4, 4), (2016, 4, 5), (2016, 4, 6)]

    def bad_payload(ticker):
        # 2 consecutive invalid rows > frozen max_consecutive=1.
        timestamps = [_epoch(*d) for d in bad_dates]
        price = 1000.0
        closes = [price, -1.0, -1.0, price]
        result = {
            "meta": {"symbol": ticker + ".T"}, "timestamp": timestamps,
            "indicators": {
                "quote": [{"open": [price] * 4, "high": [price + 2] * 4, "low": [price - 2] * 4, "close": closes, "volume": [1.0] * 4}],
                "adjclose": [{"adjclose": [price] * 4}],
            },
            "events": {},
        }
        return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")

    opener = FailThenSucceedOpener(fail_count=0, payload_fn=bad_payload)
    with pytest.raises(acquisition.V8CHistoricalAcquisitionBlocked) as excinfo:
        run_low_level(output_root=tmp_path, opener=opener, consumption_state_root=state_root)
    assert excinfo.value.reason.startswith("MALFORMED_OHLCV_QUALITY_GATE")
    assert excinfo.value.authorization_consumed is True


def test_public_summary_never_contains_payload_manifest(tmp_path):
    state_root = _fresh_state_root()
    opener = FailThenSucceedOpener(fail_count=0)
    manifest = run_low_level(output_root=tmp_path, opener=opener, consumption_state_root=state_root)
    summary = acquisition.public_acquisition_summary(manifest)
    assert "payload_manifest" not in summary
    assert summary["ticker_count"] == manifest["ticker_count"]


def test_frozen_retry_policy_fields_recorded_in_manifest(tmp_path):
    state_root = _fresh_state_root()
    opener = FailThenSucceedOpener(fail_count=0)
    manifest = run_low_level(output_root=tmp_path, opener=opener, consumption_state_root=state_root)
    assert manifest["max_attempts_per_ticker"] == 3
    assert manifest["max_retries"] == 2
    assert manifest["backoff_seconds"] == [5, 30]
    assert manifest["jitter"] is False
