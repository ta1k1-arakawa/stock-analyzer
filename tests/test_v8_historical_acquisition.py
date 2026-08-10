from __future__ import annotations

import json
import inspect
import re
import urllib.request
from datetime import date, datetime, timedelta, timezone
from email.message import Message
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


@pytest.fixture
def trusted_anchor_path(tmp_path) -> Path:
    path = tmp_path / "V8_TRUSTED_PARTITION.json"
    write_trusted_partition_anchor(path)
    return path


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
    bad_row_indices: list[int] | None = None,
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
    if bad_row_indices is not None:
        for index in bad_row_indices:
            closes[index] = -1.0
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


def _date_tuples(start: tuple[int, int, int], count: int) -> list[tuple[int, int, int]]:
    """``count`` consecutive calendar dates starting at ``start``.

    Only used to synthesize distinct Yahoo-returned trading_date values for
    the malformed-OHLCV quality-gate tests below; not a claim about a real
    JPX trading calendar.
    """
    start_date = date(*start)
    return [((start_date + timedelta(days=i)).year, (start_date + timedelta(days=i)).month,
              (start_date + timedelta(days=i)).day) for i in range(count)]


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


def write_partition_manifest(
    path: Path,
    *,
    t1=None,
    t2=None,
    mutation=None,
    source_reproduction_status: str = "PASS",
    source_url: str = "https://www.jpx.co.jp/synthetic/data_j.xls",
    source_host: str = "www.jpx.co.jp",
) -> dict:
    """Persist a self-hash-verified synthetic partition fixture."""
    blocks = {"T0": _tickers(4000), "T1": list(t1 or _tickers(1000)), "T2": list(t2 or _tickers(2000)),
              "T3": _tickers(3000), "T_spare": _tickers(5000)}
    manifest = {
        "schema_version": partition.SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "source_snapshot_semantics": partition.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": partition.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": SYNTHETIC_IMPLEMENTATION_GIT_COMMIT,
        "created_utc": "2026-08-09T00:00:00Z",
        "source_url": source_url,
        "source_host": source_host,
        "source_acquisition_utc": "2026-08-09T00:00:00Z",
        "source_raw_sha256": "0" * 64,
        "source_raw_byte_count": 0,
        "v4_source_raw_sha256_reference": "1" * 64,
        "v4_raw_sha_equality_required": partition.V4_RAW_SHA_EQUALITY_REQUIRED,
        "source_reproduction_status": source_reproduction_status,
        "t0_reproduction_status": source_reproduction_status,
        "eligible_ticker_count": 1500,
        "eligible_ticker_list_sha256": partition.ticker_list_sha256(sum((blocks[key] for key in blocks), [])),
        "selection_rule": "synthetic fixture selection rule",
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
    return manifest


def bound_acquire_kwargs(
    output_root,
    partition_manifest_path,
    block,
    opener,
    trusted_anchor_path,
    *,
    resolver=None,
):
    return {
        "output_root": output_root,
        "partition_manifest_path": partition_manifest_path,
        "block": block,
        "opener": opener,
        "clock": clock_stub,
        "git_commit_resolver": resolver or (lambda: SYNTHETIC_IMPLEMENTATION_GIT_COMMIT),
        "git_anchor_reader": lambda _: acquisition._read_trusted_partition_anchor_bytes(
            trusted_anchor_path.read_bytes()
        ),
        "monotonic_clock": lambda: 0.0,
        "sleep_fn": lambda _: None,
    }


def acquire_with_test_dependencies(**kwargs):
    return acquisition._acquire_production_historical_block_bundle_with_dependencies(**kwargs)


# ---------------------------------------------------------------------------
# Validated partition binding
# ---------------------------------------------------------------------------


def test_public_production_signature_has_only_required_inputs():
    assert tuple(inspect.signature(acquisition.acquire_historical_block_bundle).parameters) == (
        "output_root", "block", "partition_manifest_path"
    )


@pytest.mark.parametrize(
    "forbidden",
    ("repository_root", "implementation_git_commit_resolver", "request_start", "request_end_exclusive",
     "opener", "clock", "monotonic_clock", "sleep_fn", "trusted_partition_path",
     "trusted_manifest_sha", "trusted_implementation_sha"),
)
def test_public_production_boundary_rejects_all_overrides(tmp_path, forbidden):
    kwargs = {
        "output_root": tmp_path / "private",
        "block": "T1",
        "partition_manifest_path": tmp_path / "partition.json",
        forbidden: "override",
    }
    with pytest.raises(TypeError):
        acquisition.acquire_historical_block_bundle(**kwargs)


def test_public_path_blocks_before_network_without_valid_partition(tmp_path, monkeypatch):
    opener_calls = []

    def forbidden_trusted_yahoo_opener(request_obj):
        opener_calls.append(request_obj)
        raise AssertionError("trusted Yahoo opener must not run")

    monkeypatch.setattr(acquisition, "_default_trusted_yahoo_opener", forbidden_trusted_yahoo_opener)
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.acquire_historical_block_bundle(
            output_root=tmp_path / "private", block="T1", partition_manifest_path=tmp_path / "missing.json"
        )
    assert excinfo.value.reason in {
        "PRODUCTION_GIT_WORKTREE_DIRTY",
        "PRODUCTION_GIT_HEAD_NOT_ORIGIN",
        "TRUSTED_PARTITION_NOT_AUTHORIZED",
        "PARTITION_MANIFEST_READ_FAILED",
    }
    assert opener_calls == []


def test_git_resolution_precedes_anchor_read_and_committed_anchor_bytes_are_used(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "partition.json"
    expected = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=expected)
    events: list[str] = []
    opener = default_opener()

    def resolve() -> str:
        events.append("git")
        return SYNTHETIC_IMPLEMENTATION_GIT_COMMIT

    def anchor_reader(commit: str):
        events.append("anchor:" + commit)
        return acquisition._read_trusted_partition_anchor_bytes(trusted_anchor_path.read_bytes())

    manifest = acquire_with_test_dependencies(**{
        **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path),
        "git_commit_resolver": resolve,
        "git_anchor_reader": anchor_reader,
    })
    assert events == ["git", "anchor:" + SYNTHETIC_IMPLEMENTATION_GIT_COMMIT]
    assert opener.calls == expected["block_assignments"]["T1"]
    assert manifest["request_start"] == acquisition.REQUEST_START
    assert manifest["request_end_exclusive"] == acquisition.REQUEST_END_EXCLUSIVE


def test_git_show_anchor_reader_uses_verified_head_object(monkeypatch):
    anchor_bytes = partition.canonical_json_bytes({
        "schema_version": acquisition.TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "authorization_status": "NOT_AUTHORIZED",
        "authorized_partition_manifest_sha256": None,
        "authorized_partition_implementation_git_commit": None,
        "authorization_note": "committed bytes only",
    })

    class Result:
        returncode = 0
        stdout = anchor_bytes

    observed: list[list[str]] = []

    def fake_run(command, **_):
        observed.append(command)
        return Result()

    monkeypatch.setattr(acquisition.subprocess, "run", fake_run)
    anchor = acquisition._read_trusted_partition_anchor_from_verified_head("a" * 40)
    assert anchor["authorization_status"] == "NOT_AUTHORIZED"
    assert observed == [[
        "git", "-C", str(acquisition.CANONICAL_REPOSITORY_ROOT), "show",
        "a" * 40 + ":V8_TRUSTED_PARTITION.json",
    ]]


def test_worktree_anchor_difference_cannot_replace_committed_anchor_bytes(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest)
    committed_anchor_bytes = trusted_anchor_path.read_bytes()
    trusted_anchor_path.write_text("{not the committed anchor", encoding="utf-8")
    opener = default_opener()
    result = acquire_with_test_dependencies(**{
        **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path),
        "git_anchor_reader": lambda _: acquisition._read_trusted_partition_anchor_bytes(committed_anchor_bytes),
    })
    assert result["partition_manifest_sha256"] == manifest["manifest_sha256"]
    assert opener.calls == manifest["block_assignments"]["T1"]


@pytest.mark.parametrize(
    "reason",
    ("PRODUCTION_GIT_WORKTREE_DIRTY", "PRODUCTION_GIT_HEAD_UNAVAILABLE",
     "PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE", "PRODUCTION_GIT_HEAD_NOT_ORIGIN"),
)
def test_acquisition_git_provenance_failure_blocks_before_anchor_or_network(tmp_path, trusted_anchor_path, reason):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest)
    opener = default_opener()
    anchor_read = False

    def unavailable():
        raise acquisition.V8HistoricalAcquisitionBlocked(reason)

    def should_not_read(_: str):
        nonlocal anchor_read
        anchor_read = True
        raise AssertionError("anchor read after failed Git verification")

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquire_with_test_dependencies(**{
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path),
            "git_commit_resolver": unavailable,
            "git_anchor_reader": should_not_read,
        })
    assert excinfo.value.reason == reason
    assert anchor_read is False
    assert opener.calls == []


def test_unauthorized_anchor_blocks_before_network(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "partition.json"
    write_partition_manifest(partition_path)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquire_with_test_dependencies(**bound_acquire_kwargs(
            tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path
        ))
    assert excinfo.value.reason == "TRUSTED_PARTITION_NOT_AUTHORIZED"
    assert opener.calls == []


def test_authorized_synthetic_manifest_blocks_before_network(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "synthetic.json"
    manifest = write_partition_manifest(
        partition_path,
        source_reproduction_status="SYNTHETIC",
        source_url="https://example.invalid/data_j.xls",
        source_host="example.invalid",
    )
    write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquire_with_test_dependencies(**bound_acquire_kwargs(
            tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path
        ))
    # src.v8_partition.read_partition_manifest() now centrally validates
    # source_reproduction_status/t0_reproduction_status (V8_HISTORICAL_
    # RESEARCH_DESIGN.md §16) before acquisition's own redundant check ever
    # runs, so this BLOCKs earlier with the partition-module reason.
    assert excinfo.value.reason == "MANIFEST_SOURCE_REPRODUCTION_NOT_PASS"
    assert opener.calls == []


@pytest.mark.parametrize(
    ("source_url", "source_host", "reason"),
    (
        ("https://example.invalid/data_j.xls", "example.invalid", "PARTITION_MANIFEST_SOURCE_HOST_MISMATCH"),
        ("http://www.jpx.co.jp/data_j.xls", "www.jpx.co.jp", "PARTITION_MANIFEST_SOURCE_ORIGIN_INVALID"),
        ("https://user@www.jpx.co.jp/data_j.xls", "www.jpx.co.jp", "PARTITION_MANIFEST_SOURCE_ORIGIN_INVALID"),
        ("https://www.jpx.co.jp:444/data_j.xls", "www.jpx.co.jp", "PARTITION_MANIFEST_SOURCE_ORIGIN_INVALID"),
    ),
)
def test_authorized_manifest_requires_exact_jpx_metadata(
    tmp_path, trusted_anchor_path, source_url, source_host, reason
):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path, source_url=source_url, source_host=source_host)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquire_with_test_dependencies(**bound_acquire_kwargs(
            tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path
        ))
    assert excinfo.value.reason == reason
    assert opener.calls == []


def test_trust_anchor_manifest_sha_and_commit_mismatch_block_before_network(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    opener = default_opener()
    for field, value, reason in (
        ("manifest_sha256", "0" * 64, "TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH"),
        ("implementation_git_commit", "b" * 40, "TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH"),
    ):
        write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest, **{field: value})
        with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
            acquire_with_test_dependencies(**bound_acquire_kwargs(
                tmp_path / ("private-" + field), partition_path, "T1", opener, trusted_anchor_path
            ))
        assert excinfo.value.reason == reason
    assert opener.calls == []


def test_duplicate_anchor_json_key_blocks_before_network(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "partition.json"
    write_partition_manifest(partition_path)
    trusted_anchor_path.write_text(
        '{"schema_version":"V8_TRUSTED_PARTITION_V1","schema_version":"V8_TRUSTED_PARTITION_V1"}',
        encoding="utf-8",
    )
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquire_with_test_dependencies(**bound_acquire_kwargs(
            tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path
        ))
    assert excinfo.value.reason == "TRUSTED_PARTITION_ANCHOR_DUPLICATE_KEY"
    assert opener.calls == []


def test_validated_partition_binding_reaches_fake_t1_and_t2_transport(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "partition.json"
    expected = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=expected)
    for block in ("T1", "T2"):
        opener = default_opener()
        manifest = acquire_with_test_dependencies(**bound_acquire_kwargs(
            tmp_path / ("private-" + block), partition_path, block, opener, trusted_anchor_path
        ))
        assert opener.calls == expected["block_assignments"][block]
        assert manifest["partition_manifest_sha256"] == expected["manifest_sha256"]


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
def test_invalid_partition_binding_blocks_before_network(
    tmp_path, trusted_anchor_path, label, block, mutation, t1, t2
):
    partition_path = tmp_path / f"{label}.json"
    manifest = write_partition_manifest(partition_path, t1=t1, t2=t2, mutation=mutation)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquire_with_test_dependencies(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, block, opener, trusted_anchor_path)
        )
    assert opener.calls == []


def test_tampered_partition_self_hash_blocks_before_network(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest)
    manifest["study_name"] = "TAMPERED"
    partition_path.write_bytes(partition.canonical_json_bytes(manifest))
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquire_with_test_dependencies(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path)
        )
    assert excinfo.value.reason == "MANIFEST_SHA_MISMATCH"
    assert opener.calls == []


def test_missing_partition_manifest_blocks_before_network(tmp_path, trusted_anchor_path):
    write_trusted_partition_anchor(
        trusted_anchor_path,
        authorization_status="AUTHORIZED",
        manifest_sha256="0" * 64,
        implementation_git_commit=SYNTHETIC_IMPLEMENTATION_GIT_COMMIT,
    )
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquire_with_test_dependencies(
            **bound_acquire_kwargs(
                tmp_path / "private", tmp_path / "missing.json", "T1", opener, trusted_anchor_path
            )
        )
    assert excinfo.value.reason == "PARTITION_MANIFEST_READ_FAILED"
    assert opener.calls == []


@pytest.mark.parametrize("block", ("T3", "UNKNOWN"))
def test_prohibited_or_unknown_block_blocks_before_network(tmp_path, trusted_anchor_path, block):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquire_with_test_dependencies(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, block, opener, trusted_anchor_path)
        )
    assert opener.calls == []


def test_caller_cannot_spoof_partition_hash_or_substitute_tickers(tmp_path):
    partition_path = tmp_path / "partition.json"
    for forbidden in ("partition_manifest_sha256", "tickers", "trusted_partition_anchor_path"):
        with pytest.raises(TypeError):
            acquisition.acquire_historical_block_bundle(
                output_root=tmp_path / "private", block="T1", partition_manifest_path=partition_path,
                **{forbidden: "override"},
            )


def test_implementation_provenance_failure_blocks_before_network(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest)
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquire_with_test_dependencies(
            **bound_acquire_kwargs(
                tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path, resolver=lambda: "not-a-sha"
            )
        )
    assert opener.calls == []


def test_implementation_provenance_unavailable_blocks_before_network(tmp_path, trusted_anchor_path):
    partition_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(partition_path)
    write_trusted_partition_anchor(trusted_anchor_path, manifest=manifest)
    opener = default_opener()

    def unavailable():
        raise acquisition.V8HistoricalAcquisitionBlocked("IMPLEMENTATION_GIT_COMMIT_UNAVAILABLE")

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquire_with_test_dependencies(
            **bound_acquire_kwargs(tmp_path / "private", partition_path, "T1", opener, trusted_anchor_path, resolver=unavailable)
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
    assert "V8_YAHOO_SOURCE_ORIGIN_INVALID" in excinfo.value.reason


def test_trusted_yahoo_initial_request_and_standard_port_are_permitted():
    assert acquisition._require_trusted_yahoo_url(
        "https://query1.finance.yahoo.com/v8/finance/chart/1001.T"
    ).startswith("https://")
    assert acquisition._require_trusted_yahoo_url(
        "https://query1.finance.yahoo.com:443/v8/finance/chart/1001.T"
    ).startswith("https://")


@pytest.mark.parametrize("url", (
    "https://attacker.example/chart/1001.T",
    "http://query1.finance.yahoo.com/chart/1001.T",
    "https://query1.finance.yahoo.com:444/chart/1001.T",
    "https://user@query1.finance.yahoo.com/chart/1001.T",
))
def test_yahoo_redirect_rejected_before_off_origin_request(url):
    handler = acquisition._TrustedYahooRedirectHandler()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        handler.redirect_request(
            urllib.request.Request("https://query1.finance.yahoo.com/v8/finance/chart/1001.T"),
            None,
            302,
            "Found",
            Message(),
            url,
        )
    assert excinfo.value.reason == "V8_YAHOO_SOURCE_ORIGIN_INVALID"


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


# ---------------------------------------------------------------------------
# Malformed-OHLCV quality gate (POLICY_G_PRIME_V1_UNIFORM_RETURNED_ROW_
# QUALITY_GATE, V8_HISTORICAL_RESEARCH_DESIGN.md §17) -- integration-level
# behaviour through the full acquisition pipeline. Pure threshold-arithmetic
# tests (fraction/consecutive boundaries, per-test-year semantics, cross-year
# runs, reason-label uniformity) live in test_v8_malformed_ohlcv_policy.py.
# ---------------------------------------------------------------------------


def test_schema_version_bumped_for_malformed_ohlcv_policy_field():
    assert acquisition.SCHEMA_VERSION == "V8_HISTORICAL_ACQUISITION_V2"
    assert "malformed_ohlcv_policy" in acquisition.ACQUISITION_MANIFEST_FIELDS


def test_prohibited_blocks_unchanged_by_policy_addition():
    assert acquisition.PROHIBITED_ACQUISITION_BLOCKS == ("T0", "T3", "T_spare")
    assert acquisition.ALLOWED_ACQUISITION_BLOCKS == ("T1", "T2")


def test_accepted_invalid_rows_excluded_without_repair(tmp_path):
    dates = _date_tuples((2016, 4, 1), 151)
    opener = FakeOpener(lambda t: synthetic_payload(t, dates, bad_row_indices=[0]))
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(
        **acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener)
    )
    record = manifest["payload_manifest"][0]
    assert record["valid_price_row_count"] == 150
    assert record["invalid_price_row_count"] == 1
    assert manifest["valid_price_row_count"] == 150
    assert manifest["invalid_price_row_count"] == 1
    assert manifest["invalid_reason_counts"] == {"NONPOSITIVE_CLOSE": 1}
    # Raw payload bytes on disk are the untouched original wire bytes -- no
    # fill/repair/imputation is ever applied to an accepted invalid row.
    raw_path = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME / "T1" / acquisition.RAW_DIRNAME / "1001.json"
    assert raw_path.read_bytes() == synthetic_payload("1001", dates, bad_row_indices=[0])


def test_multi_ticker_invalid_reason_counts_accurate_and_policy_uniform_across_t1_t2(tmp_path):
    dates = _date_tuples((2016, 4, 1), 200)

    def payload_fn(ticker: str) -> bytes:
        return synthetic_payload(ticker, dates, bad_row_indices=[5])

    for block, tickers in (("T1", ["1001", "1002"]), ("T2", ["2001", "2002"])):
        opener = FakeOpener(payload_fn)
        manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(
            **acquire_kwargs(tmp_path / f"root-{block}", block, tickers, opener)
        )
        assert manifest["invalid_price_row_count"] == 2
        assert manifest["invalid_reason_counts"] == {"NONPOSITIVE_CLOSE": 2}
        assert manifest["valid_price_row_count"] == 2 * 199
        assert manifest["malformed_ohlcv_policy"]["policy_name"] == acquisition.MALFORMED_OHLCV_POLICY_NAME


def test_manifest_records_exact_policy_metadata(tmp_path):
    opener = default_opener()
    manifest = acquisition._acquire_historical_block_bundle_with_validated_inputs(
        **acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener)
    )
    assert manifest["malformed_ohlcv_policy"] == {
        "policy_name": "POLICY_G_PRIME_V1_UNIFORM_RETURNED_ROW_QUALITY_GATE",
        "invalid_fraction_threshold": 0.01,
        "max_consecutive_invalid_returned_rows": 5,
        "full_p_hist_check_required": True,
        "test_years": [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        "expected_calendar_missing_dates_treated_as_malformed": False,
        "threshold_exceedance_action": "BLOCK_WHOLE_ACQUISITION",
    }


def test_read_acquisition_manifest_fails_closed_on_missing_policy_metadata(tmp_path):
    opener = default_opener()
    acquisition._acquire_historical_block_bundle_with_validated_inputs(
        **acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener)
    )
    manifest_path = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME / "T1" / acquisition.MANIFEST_FILENAME
    tampered = json.loads(manifest_path.read_bytes())
    del tampered["malformed_ohlcv_policy"]
    manifest_path.write_bytes(json.dumps(tampered).encode("utf-8"))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T1")
    assert excinfo.value.reason == "MANIFEST_SCHEMA_INVALID"


def test_read_acquisition_manifest_fails_closed_on_wrong_policy_metadata(tmp_path):
    opener = default_opener()
    acquisition._acquire_historical_block_bundle_with_validated_inputs(
        **acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener)
    )
    manifest_path = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME / "T1" / acquisition.MANIFEST_FILENAME
    tampered = json.loads(manifest_path.read_bytes())
    tampered["malformed_ohlcv_policy"]["invalid_fraction_threshold"] = 0.05
    manifest_path.write_bytes(json.dumps(tampered).encode("utf-8"))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T1")
    assert excinfo.value.reason == "MALFORMED_OHLCV_POLICY_METADATA_MISMATCH"


def test_threshold_exceedance_blocks_whole_acquisition_no_partial_bundle(tmp_path):
    dates = _date_tuples((2016, 4, 1), 100)
    opener = FakeOpener(lambda t: synthetic_payload(t, dates, bad_row_indices=[0, 1]))  # 2/100 -> BLOCK
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(
            **acquire_kwargs(tmp_path / "root", "T1", ["1001"], opener)
        )
    assert excinfo.value.reason.startswith("MALFORMED_OHLCV_QUALITY_GATE")
    acquisitions_root = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME
    if acquisitions_root.exists():
        assert {entry.name for entry in acquisitions_root.iterdir()} == set()
    assert acquisition.RETRY_COUNT == 0


def test_threshold_exceedance_after_earlier_accepted_ticker_discards_everything(tmp_path):
    """No partial/fewer-than-300-ticker bundle may ever be published: even a
    prior ticker that individually passed the gate must be discarded whole
    when a later ticker in the same block exceeds the threshold."""
    dates = _date_tuples((2016, 4, 1), 100)

    def payload_fn(ticker: str) -> bytes:
        if ticker == "1001":
            return synthetic_payload(ticker, dates)  # clean, would pass alone
        return synthetic_payload(ticker, dates, bad_row_indices=[0, 1])  # 2/100 -> BLOCK

    opener = FakeOpener(payload_fn)
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition._acquire_historical_block_bundle_with_validated_inputs(
            **acquire_kwargs(tmp_path / "root", "T1", ["1001", "1002"], opener)
        )
    acquisitions_root = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME
    if acquisitions_root.exists():
        assert {entry.name for entry in acquisitions_root.iterdir()} == set()


def test_threshold_exceedance_reason_does_not_expose_ticker_or_date(tmp_path):
    dates = _date_tuples((2016, 4, 1), 100)
    secret_ticker = "1234"
    opener = FakeOpener(lambda t: synthetic_payload(t, dates, bad_row_indices=[0, 1]))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(
            **acquire_kwargs(tmp_path / "root", "T1", [secret_ticker], opener)
        )
    reason = excinfo.value.reason
    assert secret_ticker not in reason
    assert not re.search(r"\d{4}-\d{2}-\d{2}", reason)
    assert reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


# ---------------------------------------------------------------------------
# Private-ticker redaction on adjacent (non-quality-gate) failure paths
# ---------------------------------------------------------------------------


def test_noncanonical_ticker_reason_does_not_expose_ticker(tmp_path):
    secret_ticker = "1234"
    opener = default_opener()
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(
            **acquire_kwargs(tmp_path / "root", "T1", [secret_ticker + ".T"], opener)
        )
    assert excinfo.value.reason == "V8_TICKER_NOT_CANONICAL"
    assert secret_ticker not in excinfo.value.reason


def test_transport_failure_reason_does_not_expose_ticker(tmp_path):
    secret_ticker = "1234"
    opener = FakeOpener(lambda t: synthetic_payload(t, DEFAULT_DATES, symbol_override="9999"))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(
            **acquire_kwargs(tmp_path / "root", "T1", [secret_ticker], opener)
        )
    assert excinfo.value.reason.startswith("TICKER_FETCH_BLOCKED:")
    assert "SYMBOL_MISMATCH" in excinfo.value.reason
    assert secret_ticker not in excinfo.value.reason


def test_http_error_reason_does_not_expose_ticker(tmp_path):
    import urllib.error

    secret_ticker = "1234"

    class Http429Opener:
        def __call__(self, request_obj):
            raise urllib.error.HTTPError(request_obj.full_url, 429, "Too Many Requests", {}, None)

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition._acquire_historical_block_bundle_with_validated_inputs(
            **acquire_kwargs(tmp_path / "root", "T1", [secret_ticker], Http429Opener())
        )
    assert excinfo.value.reason.startswith("TICKER_FETCH_BLOCKED:")
    assert "HTTP_STATUS_429" in excinfo.value.reason
    assert secret_ticker not in excinfo.value.reason


def test_raw_payload_mismatch_reasons_are_static_string_literals_with_no_ticker_concatenation():
    """RAW_PAYLOAD_SHA_MISMATCH / RAW_PAYLOAD_BYTE_COUNT_MISMATCH are not
    reachable through any realistic fake opener: fetch_chart_once() itself
    rejects a non-bytes response.read() result (RESPONSE_BYTES_INVALID)
    before either check could ever observe a divergence, and a genuine
    bytes payload is captured byte-for-byte identically on both sides of
    the comparison by construction. This statically proves the redaction
    fix (no ' + ticker' concatenation) instead."""
    text = Path(acquisition.__file__).read_text(encoding="utf-8")
    assert 'V8HistoricalAcquisitionBlocked("RAW_PAYLOAD_SHA_MISMATCH")' in text
    assert 'V8HistoricalAcquisitionBlocked("RAW_PAYLOAD_BYTE_COUNT_MISMATCH")' in text
    assert '"RAW_PAYLOAD_SHA_MISMATCH:"' not in text
    assert '"RAW_PAYLOAD_BYTE_COUNT_MISMATCH:"' not in text


def test_no_ticker_concatenation_survives_for_any_redacted_reason_prefix():
    """Guards against regressing any of the four redacted reason formats
    back to their old ':<ticker>' or '<ticker>' suffix pattern."""
    text = Path(acquisition.__file__).read_text(encoding="utf-8")
    assert '"V8_TICKER_NOT_CANONICAL:" + str(ticker)' not in text
    assert '"TICKER_" + str(ticker) + ":"' not in text
    assert '"RAW_PAYLOAD_SHA_MISMATCH:" + ticker' not in text
    assert '"RAW_PAYLOAD_BYTE_COUNT_MISMATCH:" + ticker' not in text
