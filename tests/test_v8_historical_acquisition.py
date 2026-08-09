from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from src import v8_historical_acquisition as acquisition
from src import v8_partition as partition

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_IMPLEMENTATION_GIT_COMMIT = "a" * 40


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def write_trusted_partition_anchor(
    path: Path,
    *,
    manifest: dict | None = None,
    manifest_sha256: str | None = None,
    implementation_git_commit: str | None = None,
    authorization_status: str | None = None,
) -> None:
    authorized = manifest is not None if authorization_status is None else authorization_status == "AUTHORIZED"
    path.write_bytes(partition.canonical_json_bytes({
        "schema_version": acquisition.TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "authorization_status": authorization_status or ("AUTHORIZED" if authorized else "NOT_AUTHORIZED"),
        "authorized_partition_manifest_sha256": (
            manifest_sha256 if manifest_sha256 is not None else (manifest["manifest_sha256"] if manifest else None)
        ),
        "authorized_partition_implementation_git_commit": (
            implementation_git_commit if implementation_git_commit is not None else (
                manifest["partition_implementation_git_commit"] if manifest else None
            )
        ),
        "authorization_note": "test-only anchor",
    }))


@pytest.fixture(autouse=True)
def test_trusted_partition_anchor(monkeypatch, tmp_path):
    anchor_path = tmp_path / "V8_TRUSTED_PARTITION.json"
    monkeypatch.setattr(acquisition, "TRUSTED_PARTITION_ANCHOR_PATH", anchor_path)
    write_trusted_partition_anchor(anchor_path)
    yield


# ---------------------------------------------------------------------------
# Fake Yahoo transport
# ---------------------------------------------------------------------------


def _epoch(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def synthetic_payload(
    ticker: str,
    dates: list[tuple[int, int, int]],
    price: float = 1000.0,
    *,
    bad_row_index: int | None = None,
    duplicate: bool = False,
    symbol_override: str | None = None,
    empty: bool = False,
) -> bytes:
    if empty:
        result = {"meta": {"symbol": (symbol_override or ticker) + ".T"}, "timestamp": [], "indicators": {}}
        return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")
    use_dates = list(dates)
    if duplicate:
        use_dates = use_dates + [use_dates[0]]
    timestamps = [_epoch(*d) for d in use_dates]
    closes = [price] * len(timestamps)
    if bad_row_index is not None:
        closes[bad_row_index] = -1.0
    result = {
        "meta": {"symbol": (symbol_override or ticker) + ".T"},
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
    """Deterministic fake Yahoo Chart opener; performs no network I/O."""

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


def default_opener() -> FakeOpener:
    return FakeOpener(lambda ticker: synthetic_payload(ticker, DEFAULT_DATES))


def clock_stub():
    return datetime(2026, 8, 9, tzinfo=timezone.utc)


def acquire_kwargs(output_root, block, tickers, opener, *, sleep_state=None):
    state = sleep_state if sleep_state is not None else {"now": 0.0}
    return dict(
        output_root=output_root,
        repository_root=ROOT,
        block=block,
        tickers=tickers,
        partition_manifest_sha256="s" * 64,
        implementation_git_commit=SYNTHETIC_IMPLEMENTATION_GIT_COMMIT,
        opener=opener,
        clock=clock_stub,
        monotonic_clock=lambda: state["now"],
        sleep_fn=lambda s: state.__setitem__("now", state["now"] + s),
    )


def _tickers(start: int) -> list[str]:
    return [f"{code:04d}" for code in range(start, start + 300)]


def write_partition_manifest(path: Path, *, t1=None, t2=None, mutation=None) -> dict:
    """Persist a self-hash-verified synthetic partition fixture."""
    blocks = {"T0": _tickers(4000), "T1": list(t1 or _tickers(1000)), "T2": list(t2 or _tickers(2000)),
              "T3": _tickers(3000), "T_spare": _tickers(5000)}
    manifest = {
        "schema_version": partition.SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "partition_implementation_git_commit": SYNTHETIC_IMPLEMENTATION_GIT_COMMIT,
        "created_utc": "2026-08-09T00:00:00Z",
        "source_url": "https://example.invalid/jpx",
        "source_host": "example.invalid",
        "source_acquisition_utc": "2026-08-09T00:00:00Z",
        "source_raw_sha256": "0" * 64,
        "source_raw_byte_count": 0,
        "expected_v4_source_raw_sha256": "0" * 64,
        "source_reproduction_status": "SYNTHETIC",
        "eligible_ticker_count": 1500,
        "eligible_ticker_list_sha256": partition.ticker_list_sha256(sum((blocks[key] for key in blocks), [])),
        "deterministic_ordering_rule": partition.DETERMINISTIC_ORDERING_RULE,
        "t0_ticker_list_sha256": partition.ticker_list_sha256(blocks["T0"]),
        "t1_ticker_list_sha256": partition.ticker_list_sha256(blocks["T1"]),
        "t2_ticker_list_sha256": partition.ticker_list_sha256(blocks["T2"]),
        "t3_ticker_list_sha256": partition.ticker_list_sha256(blocks["T3"]),
        "t_spare_ticker_list_sha256": partition.ticker_list_sha256(blocks["T_spare"]),
        "legacy_exclude_list": [],
        "legacy_exclude_list_sha256": partition.ticker_list_sha256([]),
        "block_sizes": {key: len(value) for key, value in blocks.items()},
        "block_assignments": blocks,
        "p_hist_start": partition.P_HIST_START,
        "p_hist_end": partition.P_HIST_END,
        "t1_role": partition.T1_ROLE,
        "t2_role": partition.T2_ROLE,
        "t3_role": partition.T3_ROLE,
        "t3_price_acquisition_authorized": False,
    }
    if mutation is not None:
        mutation(manifest)
    manifest["manifest_sha256"] = partition.canonical_sha256(manifest)
    assert set(manifest) == set(partition.MANIFEST_FIELDS)
    path.write_bytes(partition.canonical_json_bytes(manifest))
    write_trusted_partition_anchor(acquisition.TRUSTED_PARTITION_ANCHOR_PATH, manifest=manifest)
    return manifest


def bound_acquire_kwargs(output_root, partition_manifest_path, block, opener, *, resolver=None):
    return {
        "output_root": output_root,
        "repository_root": ROOT,
        "partition_manifest_path": partition_manifest_path,
        "block": block,
        "opener": opener,
        "clock": clock_stub,
        "implementation_git_commit_resolver": resolver or (lambda _: SYNTHETIC_IMPLEMENTATION_GIT_COMMIT),
        "monotonic_clock": lambda: 0.0,
        "sleep_fn": lambda _: None,
    }


# ---------------------------------------------------------------------------
# Validated partition binding
# ---------------------------------------------------------------------------


def test_unauthorized_canonical_trust_anchor_blocks_before_network(tmp_path):
    partition_path = tmp_path / "synthetic-partition.json"
    write_partition_manifest(partition_path)
    write_trusted_partition_anchor(acquisition.TRUSTED_PARTITION_ANCHOR_PATH)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener)
        )
    assert excinfo.value.reason == "TRUSTED_PARTITION_NOT_AUTHORIZED"
    assert opener.calls == []


def test_self_hashed_arbitrary_manifest_cannot_reach_network_without_authorization(tmp_path):
    partition_path = tmp_path / "forged-300-ticker-partition.json"
    forged = write_partition_manifest(partition_path, t1=_tickers(6000), t2=_tickers(7000))
    assert partition.read_partition_manifest(partition_path)["manifest_sha256"] == forged["manifest_sha256"]
    write_trusted_partition_anchor(acquisition.TRUSTED_PARTITION_ANCHOR_PATH)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener)
        )
    assert opener.calls == []


def test_trust_anchor_manifest_sha_mismatch_blocks_before_network(tmp_path):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(
        acquisition.TRUSTED_PARTITION_ANCHOR_PATH,
        manifest=manifest,
        manifest_sha256="0" * 64,
    )
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener)
        )
    assert excinfo.value.reason == "TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH"
    assert opener.calls == []


def test_trust_anchor_partition_implementation_mismatch_blocks_before_network(tmp_path):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(
        acquisition.TRUSTED_PARTITION_ANCHOR_PATH,
        manifest=manifest,
        implementation_git_commit="b" * 40,
    )
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener)
        )
    assert excinfo.value.reason == "TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH"
    assert opener.calls == []


def test_tampered_trust_anchor_blocks_before_network(tmp_path):
    partition_path = tmp_path / "partition.json"
    write_partition_manifest(partition_path)
    acquisition.TRUSTED_PARTITION_ANCHOR_PATH.write_text("{not json", encoding="utf-8")
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener)
        )
    assert excinfo.value.reason == "TRUSTED_PARTITION_ANCHOR_INVALID_JSON"
    assert opener.calls == []


def test_validated_partition_binding_reaches_fake_t1_transport(tmp_path):
    partition_path = tmp_path / "partition.json"
    expected = write_partition_manifest(partition_path)
    opener = default_opener()
    manifest = acquisition.acquire_historical_block_bundle(
        **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener)
    )
    assert opener.calls == expected["block_assignments"]["T1"]
    assert manifest["partition_manifest_sha256"] == expected["manifest_sha256"]
    assert manifest["implementation_git_commit"] == SYNTHETIC_IMPLEMENTATION_GIT_COMMIT


def test_validated_partition_binding_reaches_fake_t2_transport(tmp_path):
    partition_path = tmp_path / "partition.json"
    expected = write_partition_manifest(partition_path)
    opener = default_opener()
    manifest = acquisition.acquire_historical_block_bundle(
        **bound_acquire_kwargs(tmp_path / "private", partition_path, "T2", opener)
    )
    assert opener.calls == expected["block_assignments"]["T2"]
    assert manifest["partition_manifest_sha256"] == expected["manifest_sha256"]
    assert manifest["sealed"] is True


@pytest.mark.parametrize(
    ("label", "block", "mutation", "t1", "t2"),
    [
        ("wrong_schema", "T1", lambda value: value.__setitem__("schema_version", "WRONG"), None, None),
        ("wrong_study", "T1", lambda value: value.__setitem__("study_name", "WRONG"), None, None),
        ("wrong_design", "T1", lambda value: value.__setitem__("design_commit", "0" * 40), None, None),
        ("missing_assignment", "T1", lambda value: value.__setitem__("block_assignments", {}), None, None),
        ("299_tickers", "T1", None, _tickers(1000)[:-1], None),
        ("301_tickers", "T1", None, _tickers(1000) + ["1300"], None),
        ("ticker_hash_mismatch", "T1", lambda value: value.__setitem__("t1_ticker_list_sha256", "0" * 64), None, None),
    ],
)
def test_invalid_partition_binding_blocks_before_network(tmp_path, label, block, mutation, t1, t2):
    partition_path = tmp_path / f"{label}.json"
    write_partition_manifest(partition_path, t1=t1, t2=t2, mutation=mutation)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, block, opener)
        )
    assert opener.calls == []


def test_tampered_partition_self_hash_blocks_before_network(tmp_path):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    manifest["study_name"] = "TAMPERED"
    partition_path.write_bytes(partition.canonical_json_bytes(manifest))
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener)
        )
    assert excinfo.value.reason == "MANIFEST_SHA_MISMATCH"
    assert opener.calls == []


def test_missing_partition_manifest_blocks_before_network(tmp_path):
    write_trusted_partition_anchor(
        acquisition.TRUSTED_PARTITION_ANCHOR_PATH,
        authorization_status="AUTHORIZED",
        manifest_sha256="0" * 64,
        implementation_git_commit=SYNTHETIC_IMPLEMENTATION_GIT_COMMIT,
    )
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", tmp_path / "missing.json", "T1", opener)
        )
    assert excinfo.value.reason == "PARTITION_MANIFEST_READ_FAILED"
    assert opener.calls == []


@pytest.mark.parametrize("block", ("T3", "UNKNOWN"))
def test_prohibited_or_unknown_block_blocks_before_network(tmp_path, block):
    partition_path = tmp_path / "partition.json"
    write_partition_manifest(partition_path)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, block, opener)
        )
    assert opener.calls == []


def test_caller_cannot_spoof_partition_hash_or_substitute_tickers(tmp_path):
    partition_path = tmp_path / "partition.json"
    write_partition_manifest(partition_path)
    opener = default_opener()
    kwargs = bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener)
    with pytest.raises(TypeError):
        acquisition.acquire_historical_block_bundle(**kwargs, partition_manifest_sha256="0" * 64)
    with pytest.raises(TypeError):
        acquisition.acquire_historical_block_bundle(**kwargs, tickers=["9999"])
    with pytest.raises(TypeError):
        acquisition.acquire_historical_block_bundle(**kwargs, trusted_partition_anchor_path=tmp_path / "other.json")
    assert opener.calls == []


def test_implementation_provenance_failure_blocks_before_network(tmp_path):
    partition_path = tmp_path / "partition.json"
    write_partition_manifest(partition_path)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener, resolver=lambda _: "not-a-sha")
        )
    assert opener.calls == []


def test_implementation_provenance_unavailable_blocks_before_network(tmp_path):
    partition_path = tmp_path / "partition.json"
    write_partition_manifest(partition_path)
    opener = default_opener()

    def unavailable(_):
        raise acquisition.V8HistoricalAcquisitionBlocked("IMPLEMENTATION_GIT_COMMIT_UNAVAILABLE")

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.acquire_historical_block_bundle(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener, resolver=unavailable)
        )
    assert opener.calls == []


# ---------------------------------------------------------------------------
# Raw transport regression (private validated-input helper)
# ---------------------------------------------------------------------------


def test_t1_acquisition_allowed_and_manifest_shape(tmp_path):
    opener = default_opener()
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001", "1002"], opener))
    assert set(manifest) == set(acquisition.ACQUISITION_MANIFEST_FIELDS)
    assert manifest["block"] == "T1"
    assert manifest["role"] == "VALIDATION"
    assert manifest["status"] == "RAW_ACQUIRED_NOT_OPENED"
    assert manifest["sealed"] is False
    assert manifest["research_access_authorized"] is False
    assert opener.calls == ["1001", "1002"]


def test_t2_acquisition_allowed_and_sealed(tmp_path):
    opener = default_opener()
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T2", ["2001", "2002"], opener))
    assert manifest["block"] == "T2"
    assert manifest["role"] == "SEALED_HOLDOUT"
    assert manifest["status"] == "RAW_ACQUIRED_SEALED"
    assert manifest["sealed"] is True
    assert manifest["research_access_authorized"] is False


def test_t2_publishes_sealed_json(tmp_path):
    opener = default_opener()
    acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T2", ["2001"], opener))
    sealed_path = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME / "T2" / acquisition.SEALED_FILENAME
    assert sealed_path.exists()
    record = json.loads(sealed_path.read_bytes())
    assert record["sealed"] is True
    assert record["research_access_authorized"] is False


def test_t1_does_not_publish_sealed_json(tmp_path):
    opener = default_opener()
    acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    sealed_path = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME / "T1" / acquisition.SEALED_FILENAME
    assert not sealed_path.exists()


def test_t3_acquisition_always_blocked(tmp_path):
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T3", ["3001"], opener))
    assert excinfo.value.reason.startswith("V8_BLOCK_ACQUISITION_PROHIBITED")
    assert opener.calls == []


@pytest.mark.parametrize("block", ["T0", "T_spare", "all", "", "t1", None])
def test_only_t1_and_t2_are_ever_accepted(tmp_path, block):
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", block, ["1001"], opener))
    assert excinfo.value.reason.startswith("V8_BLOCK_ACQUISITION_PROHIBITED")
    assert opener.calls == []


# ---------------------------------------------------------------------------
# Transport / integrity BLOCKs
# ---------------------------------------------------------------------------


def test_wrong_response_host_blocks(tmp_path):
    opener = FakeOpener(lambda t: synthetic_payload(t, DEFAULT_DATES), host="evil.example.com")
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    assert "RESPONSE_HOST_MISMATCH" in excinfo.value.reason


def test_symbol_mismatch_blocks(tmp_path):
    opener = FakeOpener(lambda t: synthetic_payload(t, DEFAULT_DATES, symbol_override="9999"))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    assert "SYMBOL_MISMATCH" in excinfo.value.reason


def test_malformed_ohlcv_blocks(tmp_path):
    opener = FakeOpener(lambda t: synthetic_payload(t, DEFAULT_DATES, bad_row_index=0))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    assert excinfo.value.reason.startswith("MALFORMED_OHLCV")


def test_nonfinite_data_blocks(tmp_path):
    def payload(ticker: str) -> bytes:
        dates = DEFAULT_DATES
        timestamps = [_epoch(*d) for d in dates]
        result = {
            "meta": {"symbol": ticker + ".T"},
            "timestamp": timestamps,
            "indicators": {
                "quote": [{
                    "open": [1000.0] * len(timestamps), "high": [1002.0] * len(timestamps),
                    "low": [998.0] * len(timestamps),
                    "close": [1000.0, float("nan"), 1000.0],
                    "volume": [10000.0] * len(timestamps),
                }],
                "adjclose": [{"adjclose": [1000.0] * len(timestamps)}],
            },
            "events": {},
        }
        # json.dumps with allow_nan=True (default) so this is transport-valid
        # JSON carrying a NaN, matching what a real malformed payload could send.
        return json.dumps({"chart": {"error": None, "result": [result]}}, allow_nan=True).encode("utf-8")

    opener = FakeOpener(payload)
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    assert excinfo.value.reason.startswith("MALFORMED_OHLCV")


def test_duplicate_timestamp_blocks(tmp_path):
    opener = FakeOpener(lambda t: synthetic_payload(t, DEFAULT_DATES, duplicate=True))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))


def test_out_of_range_rows_block(tmp_path):
    out_of_range_dates = [(2015, 12, 31)]  # before REQUEST_START
    opener = FakeOpener(lambda t: synthetic_payload(t, out_of_range_dates))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    assert "OUT_OF_REQUEST_WINDOW" in excinfo.value.reason


def test_http_error_blocks_and_counts_429(tmp_path):
    import urllib.error

    class Http429Opener:
        def __init__(self):
            self.calls = []

        def __call__(self, request_obj):
            self.calls.append(request_obj)
            raise urllib.error.HTTPError(request_obj.full_url, 429, "Too Many Requests", {}, None)

    opener = Http429Opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    assert "HTTP_STATUS_429" in excinfo.value.reason


def test_empty_timestamp_response_blocks(tmp_path):
    opener = FakeOpener(lambda t: synthetic_payload(t, [], empty=True))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))


# ---------------------------------------------------------------------------
# Manifest integrity: raw SHA, atomic publish, no overwrite
# ---------------------------------------------------------------------------


def test_payload_manifest_records_exact_raw_sha(tmp_path):
    opener = default_opener()
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    record = manifest["payload_manifest"][0]
    raw_path = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME / "T1" / acquisition.RAW_DIRNAME / "1001.json"
    raw_bytes = raw_path.read_bytes()
    import hashlib

    assert record["payload_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert record["byte_count"] == len(raw_bytes)


def test_atomic_publish_no_staging_left_behind(tmp_path):
    opener = default_opener()
    acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    acquisitions_root = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME
    remaining = {entry.name for entry in acquisitions_root.iterdir()}
    assert remaining == {"T1"}


def test_atomic_publish_removes_staging_on_failure(tmp_path):
    opener = FakeOpener(lambda t: synthetic_payload(t, DEFAULT_DATES, bad_row_index=0))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    acquisitions_root = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME
    if acquisitions_root.exists():
        remaining = {entry.name for entry in acquisitions_root.iterdir()}
        assert remaining == set()


def test_overwrite_of_existing_final_bundle_blocked(tmp_path):
    opener = default_opener()
    acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], default_opener()))
    assert excinfo.value.reason.startswith("V8_ACQUISITION_ALREADY_EXISTS")


def test_output_root_inside_repository_blocked(tmp_path):
    opener = default_opener()
    inside_repo = ROOT / "tmp-v8-acquisition-test-root"
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(inside_repo, "T1", ["1001"], opener))
    assert excinfo.value.reason == "OUTPUT_PATH_INSIDE_SOURCE_REPOSITORY"
    assert not inside_repo.exists()


# ---------------------------------------------------------------------------
# Rate limiting / retry / no parallel path
# ---------------------------------------------------------------------------


def test_rate_limit_spacing_at_least_two_seconds(tmp_path):
    opener = default_opener()
    sleep_calls: list[float] = []
    state = {"now": 0.0}

    def sleep_fn(seconds):
        sleep_calls.append(seconds)
        state["now"] += seconds

    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(
        output_root=tmp_path / "root", repository_root=ROOT, block="T1",
        tickers=["1001", "1002", "1003"], partition_manifest_sha256="s" * 64,
        implementation_git_commit=SYNTHETIC_IMPLEMENTATION_GIT_COMMIT,
        opener=opener, clock=clock_stub, monotonic_clock=lambda: state["now"], sleep_fn=sleep_fn,
    )
    assert len(sleep_calls) == 2  # 3 tickers -> 2 gaps
    assert all(s >= acquisition.MIN_REQUEST_INTERVAL_SECONDS - 1e-9 for s in sleep_calls)
    assert manifest["retry_count"] == 0


def test_exactly_one_request_per_ticker(tmp_path):
    opener = default_opener()
    tickers = ["1001", "1002", "1003", "1004"]
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", tickers, opener))
    assert opener.calls == tickers
    assert manifest["request_count"] == len(tickers)
    assert manifest["success_transport_count"] == len(tickers)


def test_retry_count_always_zero_even_after_a_later_error(tmp_path):
    calls = []

    def payload_fn(ticker):
        calls.append(ticker)
        if ticker == "1002":
            raise TimeoutError("synthetic transport failure")
        return synthetic_payload(ticker, DEFAULT_DATES)

    class RaisingOpener:
        def __call__(self, request_obj):
            ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
            payload = payload_fn(ticker)
            return FakeResponse(payload, url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001", "1002", "1003"], RaisingOpener()))
    assert calls == ["1001", "1002"]  # stopped at first failure, never retried, never reached 1003


def test_no_parallel_path_calls_are_strictly_sequential(tmp_path):
    order: list[str] = []

    class SequentialTrackingOpener:
        def __call__(self, request_obj):
            ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
            order.append("start:" + ticker)
            payload = synthetic_payload(ticker, DEFAULT_DATES)
            order.append("end:" + ticker)
            return FakeResponse(payload, url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}.T")

    acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001", "1002"], SequentialTrackingOpener()))
    assert order == ["start:1001", "end:1001", "start:1002", "end:1002"]


# ---------------------------------------------------------------------------
# Access counters remain zero; no feature/profit fields
# ---------------------------------------------------------------------------


def test_t1_validation_access_count_remains_zero(tmp_path):
    opener = default_opener()
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    assert manifest["validation_access_count"] == 0
    assert manifest["feature_computation_count"] == 0
    assert manifest["outcome_access_count"] == 0


def test_t2_seal_counters_remain_zero(tmp_path):
    opener = default_opener()
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T2", ["2001"], opener))
    assert manifest["feature_computation_count"] == 0
    assert manifest["outcome_access_count"] == 0
    assert manifest["sealed_holdout_access_count"] == 0


def test_no_feature_or_profit_fields_in_manifest(tmp_path):
    opener = default_opener()
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    text = json.dumps(manifest, default=str).lower()
    for token in (
        "return_", "moving_average", "realized_volatility", "signal_",
        "candidate", "ranking", "trade_", "portfolio", "profit", "drawdown", "win_rate",
    ):
        assert token not in text, token


def test_no_raw_price_values_in_manifest(tmp_path):
    opener = default_opener()
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    text = json.dumps(manifest, default=str)
    assert "1000.0" not in text
    assert "1002.0" not in text


def test_read_acquisition_manifest_matches_written(tmp_path):
    opener = default_opener()
    written = acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    reread = acquisition.read_acquisition_manifest(tmp_path / "root", "T1")
    assert reread == written


# ---------------------------------------------------------------------------
# T2 sealed-holdout access guard
# ---------------------------------------------------------------------------


GUARD_FUNCTIONS = (
    acquisition.open_for_feature_generation,
    acquisition.open_for_candidate_generation,
    acquisition.open_for_validation,
    acquisition.open_for_backtest,
    acquisition.open_for_profit_evaluation,
)


@pytest.mark.parametrize("guard", GUARD_FUNCTIONS, ids=[fn.__name__ for fn in GUARD_FUNCTIONS])
def test_guard_blocks_sealed_t2_for_every_operation(tmp_path, guard):
    opener = default_opener()
    acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T2", ["2001"], opener))
    manifest = acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    with pytest.raises(acquisition.V8SealedHoldoutBlocked) as excinfo:
        guard(manifest)
    assert excinfo.value.reason.startswith("SEALED_HOLDOUT_ACCESS_DENIED")


@pytest.mark.parametrize("guard", GUARD_FUNCTIONS, ids=[fn.__name__ for fn in GUARD_FUNCTIONS])
def test_guard_blocks_unauthorized_t1_for_every_operation(tmp_path, guard):
    opener = default_opener()
    acquisition._acquire_historical_block_bundle_with_validated_inputs(**acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener))
    manifest = acquisition.read_acquisition_manifest(tmp_path / "root", "T1")
    with pytest.raises(acquisition.V8SealedHoldoutBlocked) as excinfo:
        guard(manifest)
    assert excinfo.value.reason.startswith("RESEARCH_ACCESS_NOT_AUTHORIZED")


def test_guard_permits_a_manifest_explicitly_authorized_and_unsealed():
    authorized_manifest = {"sealed": False, "research_access_authorized": True}
    for guard in GUARD_FUNCTIONS:
        guard(authorized_manifest)  # must not raise


def test_guard_rejects_manifest_missing_sealed_field():
    with pytest.raises(acquisition.V8SealedHoldoutBlocked) as excinfo:
        acquisition.open_for_validation({"research_access_authorized": True})
    assert excinfo.value.reason.startswith("ACQUISITION_MANIFEST_INVALID")


# ---------------------------------------------------------------------------
# Static safety
# ---------------------------------------------------------------------------


def test_module_has_no_direct_urlopen_call():
    text = Path(acquisition.__file__).read_text(encoding="utf-8")
    assert "urlopen(" not in text


def test_module_reuses_v7_yahoo_collector_read_only():
    text = Path(acquisition.__file__).read_text(encoding="utf-8")
    assert "from src.v7_yahoo_collector import" in text
    for forbidden in ("v7_daily_acquisition", "v7_seed_acquisition", "v7_activation_manifest", "v7_forward"):
        assert forbidden not in text


def test_module_never_writes_activation_manifest_or_touches_v7_study_root():
    text = Path(acquisition.__file__).read_text(encoding="utf-8")
    for token in ("write_activation_manifest_once", "durable_study_root", "ForwardStudyStore"):
        assert token not in text


def test_module_has_no_profit_or_feature_tokens_in_identifiers():
    import ast

    tree = ast.parse(Path(acquisition.__file__).read_text(encoding="utf-8"))
    docstrings = {
        node.body[0].value
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)
    }
    identifiers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node not in docstrings:
            identifiers.add(node.value)
    lowered = {value.lower() for value in identifiers}
    for token in ("profit_factor", "realized_net_profit", "win_rate", "moving_average", "candidate_rank"):
        offending = [v for v in lowered if token in v]
        assert offending == [], token
