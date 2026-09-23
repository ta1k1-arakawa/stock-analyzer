from __future__ import annotations

import csv
import ast
import hashlib
import itertools
import json
import random
import string
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src import v8_partition as historical
from src import v8_partition_recovery as recovery


ROOT = Path(__file__).resolve().parents[1]
V4_MANIFEST = ROOT / "V4_UNIVERSE_MANIFEST.json"
V4_CSV = ROOT / "V4_UNIVERSE.csv"
RECOVERY_TIME = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)
IMPLEMENTATION_COMMIT = "b60b8eb985484dc4428e05bdc59c47e27cef9c71"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fixture():
    with V4_CSV.open(encoding="utf-8", newline="") as stream:
        t0_rows = list(csv.DictReader(stream))
    t0_codes = [row["ticker"] for row in t0_rows]
    cutoff = max(_digest(code) for code in t0_codes)
    legacy = list(historical.LEGACY_EXPOSED_TICKERS_OUTSIDE_T0)
    assert all(_digest(code) > cutoff for code in legacy)

    candidates = []
    alphabet = string.digits + string.ascii_uppercase
    for chars in itertools.product(alphabet, repeat=4):
        ticker = "".join(chars)
        if ticker in t0_codes or ticker in legacy or _digest(ticker) <= cutoff:
            continue
        candidates.append(ticker)
        if len(candidates) == 2803:
            break
    assert len(candidates) == 2803

    rows = [
        {"code": row["ticker"], "name": "fixture", "market": row["market"], "industry": row["industry"]}
        for row in t0_rows
    ]
    rows.extend(
        {"code": ticker, "name": "fixture", "market": "プライム 内国株式", "industry": "合成"}
        for ticker in legacy + candidates
    )
    frame = pd.DataFrame({
        "コード": [row["code"] for row in rows],
        "銘柄名": [row["name"] for row in rows],
        "市場・区分": [row["market"] for row in rows],
        "33業種区分": [row["industry"] for row in rows],
    })
    parsed, _ = historical.parse_eligible_universe(frame)
    ordered = historical.canonical_order([row["code"] for row in parsed])
    assert ordered[:300] == t0_codes
    blocks = historical.allocate_fresh_blocks(ordered, t0_codes)
    universe_sha = historical.ticker_list_sha256(ordered)
    block_hashes = {name: historical.ticker_list_sha256(blocks[name]) for name in recovery.EXPECTED_BLOCK_SHA256}
    pins = recovery._TrustPins(eligible_count=3110, eligible_sha256=universe_sha,
                               t0_sha256=recovery.EXPECTED_T0_SHA256,
                               block_sha256=block_hashes)

    calls = []

    def parse_source(raw: bytes):
        calls.append(raw)
        return frame.copy()

    return frame, rows, blocks, pins, parse_source, calls


def _build(pins, parser, **overrides):
    args = {
        "raw_source_bytes": b"repository-safe synthetic source bytes",
        "parse_source_table": parser,
        "v4_manifest_path": V4_MANIFEST,
        "v4_universe_csv_path": V4_CSV,
        "recovery_source_url": "synthetic://fixture/source",
        "recovery_source_acquisition_utc": RECOVERY_TIME,
        "recovery_timestamp_utc": RECOVERY_TIME,
        "recovery_implementation_commit": IMPLEMENTATION_COMMIT,
        "_trust_pins": pins,
    }
    args.update(overrides)
    return recovery._build_recovery_manifest(**args)


def test_ordering_is_deterministic_and_independent_of_rng():
    values = [" ABC1 ", "0001", "ZZZZ", "abc1", "0123"]
    random.seed(17)
    first = historical.canonical_order(values)
    random.seed(999)
    second = historical.canonical_order(values)
    expected = sorted({value.strip().upper() for value in values},
                      key=lambda value: (hashlib.sha256(value.encode()).hexdigest(), value))
    assert first == second == expected
    imported = {alias.name.split(".")[0]
                for node in ast.walk(ast.parse(Path(historical.__file__).read_text(encoding="utf-8")))
                for alias in (node.names if isinstance(node, (ast.Import, ast.ImportFrom)) else [])}
    assert "random" not in imported


def test_partition_implementation_source_matches_pinned_historical_commit():
    pinned = subprocess.run(
        ["git", "show", "36cbed941050e728f7f96ce2af505e81175cc02c:src/v8_partition.py"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    assert hashlib.sha256(pinned).digest() == hashlib.sha256(Path(historical.__file__).read_bytes()).digest()


def test_synthetic_recovery_passes_all_gates_and_records_exact_safe_schema():
    _frame, _rows, blocks, pins, parser, calls = _fixture()
    manifest = _build(pins, parser)
    assert calls == [b"repository-safe synthetic source bytes"]
    assert set(manifest) == recovery.MANIFEST_FIELDS
    assert manifest["schema_version"] == "V8_PARTITION_RECOVERY_MANIFEST_V1"
    assert manifest["schema_version"] != "V8_PARTITION_MANIFEST_V3"
    assert manifest["original_manifest_byte_exact_recovered"] is False
    assert manifest["original_partition_block_identity_recovered"] is True
    assert manifest["block_assignments"] == blocks
    assert manifest["recovery_implementation_provenance"]["historical_partition_source_git_commit"] == "36cbed941050e728f7f96ce2af505e81175cc02c"
    assert manifest["original_trusted_manifest_sha256"] == recovery.ORIGINAL_MANIFEST_SHA256
    assert manifest["eligible_ticker_count"] == 3110
    assert manifest["eligible_ticker_list_sha256"] == pins.eligible_sha256
    assert manifest["block_sizes"] == {"T0": 300, "T1": 300, "T2": 300, "T3": 300, "T_spare": 1903}
    fresh = [ticker for name in ("T1", "T2", "T3", "T_spare") for ticker in blocks[name]]
    assert not set(historical.LEGACY_EXPOSED_TICKERS_OUTSIDE_T0).intersection(fresh)
    ordered = historical.canonical_order([row["code"] for row in _rows])
    pool = [ticker for ticker in ordered
            if ticker not in set(blocks["T0"])
            and ticker not in set(historical.LEGACY_EXPOSED_TICKERS_OUTSIDE_T0)]
    assert blocks["T1"] == pool[0:300]
    assert blocks["T2"] == pool[300:600]
    assert blocks["T3"] == pool[600:900]
    assert blocks["T_spare"] == pool[900:]


def test_t0_reproduction_must_pass_before_allocation(tmp_path, monkeypatch):
    _frame, rows, _blocks, pins, parser, _calls = _fixture()
    changed = [dict(row) for row in rows]
    changed[0]["industry"] += "-changed"
    bad_frame = pd.DataFrame({
        "コード": [row["code"] for row in changed],
        "銘柄名": [row["name"] for row in changed],
        "市場・区分": [row["market"] for row in changed],
        "33業種区分": [row["industry"] for row in changed],
    })
    original_allocate = historical.allocate_fresh_blocks
    calls = []

    def forbidden_allocate(*args, **kwargs):
        calls.append(True)
        return original_allocate(*args, **kwargs)

    historical.allocate_fresh_blocks = forbidden_allocate
    monkeypatch.setattr(recovery, "build_v8_partition_recovery_manifest",
                        lambda **kwargs: _build(pins, lambda _raw: bad_frame))
    try:
        with pytest.raises(recovery.V8PartitionRecoveryBlocked, match="RECOVERY_HISTORICAL_GATE_FAILED"):
            recovery.recover_and_publish_v8_partition_once(
                raw_source_bytes=b"synthetic", parse_source_table=lambda _raw: bad_frame,
                v4_manifest_path=V4_MANIFEST, v4_universe_csv_path=V4_CSV,
                recovery_source_url="synthetic://fixture/source",
                recovery_source_acquisition_utc=RECOVERY_TIME,
                recovery_timestamp_utc=RECOVERY_TIME,
                recovery_implementation_commit=IMPLEMENTATION_COMMIT,
                output_path=tmp_path / "recovery.json", repository_root=ROOT,
            )
    finally:
        historical.allocate_fresh_blocks = original_allocate
    assert calls == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("pin_change, reason", [
    ("count", "RECOVERY_ELIGIBLE_COUNT_MISMATCH"),
    ("universe_hash", "RECOVERY_ELIGIBLE_UNIVERSE_HASH_MISMATCH"),
])
def test_universe_mismatch_fails_before_allocation(pin_change, reason, tmp_path, monkeypatch):
    _frame, _rows, _blocks, pins, parser, _calls = _fixture()
    if pin_change == "count":
        bad = recovery._TrustPins(pins.eligible_count + 1, pins.eligible_sha256, pins.t0_sha256, pins.block_sha256)
    else:
        bad = recovery._TrustPins(pins.eligible_count, "0" * 64, pins.t0_sha256, pins.block_sha256)
    original_allocate = historical.allocate_fresh_blocks
    calls = []

    def forbidden_allocate(*args, **kwargs):
        calls.append(True)
        return original_allocate(*args, **kwargs)

    historical.allocate_fresh_blocks = forbidden_allocate
    monkeypatch.setattr(recovery, "build_v8_partition_recovery_manifest",
                        lambda **kwargs: _build(bad, parser))
    try:
        with pytest.raises(recovery.V8PartitionRecoveryBlocked) as error:
            recovery.recover_and_publish_v8_partition_once(
                raw_source_bytes=b"synthetic", parse_source_table=parser,
                v4_manifest_path=V4_MANIFEST, v4_universe_csv_path=V4_CSV,
                recovery_source_url="synthetic://fixture/source",
                recovery_source_acquisition_utc=RECOVERY_TIME,
                recovery_timestamp_utc=RECOVERY_TIME,
                recovery_implementation_commit=IMPLEMENTATION_COMMIT,
                output_path=tmp_path / "recovery.json", repository_root=ROOT,
            )
    finally:
        historical.allocate_fresh_blocks = original_allocate
    assert error.value.reason == reason
    assert calls == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("block", ["T1", "T2", "T3", "T_spare"])
def test_each_trusted_block_hash_mismatch_fails_closed(block, tmp_path, monkeypatch):
    _frame, _rows, _blocks, pins, parser, _calls = _fixture()
    bad_hashes = dict(pins.block_sha256)
    bad_hashes[block] = "0" * 64
    bad = recovery._TrustPins(pins.eligible_count, pins.eligible_sha256, pins.t0_sha256, bad_hashes)
    artifact = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(recovery, "build_v8_partition_recovery_manifest",
                        lambda **kwargs: _build(bad, parser))
    with pytest.raises(recovery.V8PartitionRecoveryBlocked) as error:
        recovery.recover_and_publish_v8_partition_once(
            raw_source_bytes=b"safe fixture", parse_source_table=parser,
            v4_manifest_path=V4_MANIFEST, v4_universe_csv_path=V4_CSV,
            recovery_source_url="synthetic://fixture/source",
            recovery_source_acquisition_utc=RECOVERY_TIME,
            recovery_timestamp_utc=RECOVERY_TIME,
            recovery_implementation_commit=IMPLEMENTATION_COMMIT,
            output_path=artifact, repository_root=ROOT,
        )
    assert error.value.reason == f"RECOVERY_{block.upper()}_HASH_MISMATCH"
    assert not artifact.exists()
    assert not list(tmp_path.glob("*.staging-*"))


def test_atomic_write_once_and_artifact_provenance(tmp_path, monkeypatch):
    _frame, _rows, _blocks, pins, parser, _calls = _fixture()
    manifest = _build(pins, parser)
    monkeypatch.setattr(recovery, "EXPECTED_BLOCK_SHA256", dict(pins.block_sha256))
    monkeypatch.setattr(recovery, "EXPECTED_ELIGIBLE_SHA256", pins.eligible_sha256)
    monkeypatch.setattr(recovery, "EXPECTED_ELIGIBLE_COUNT", pins.eligible_count)
    monkeypatch.setattr(recovery, "EXPECTED_T0_SHA256", pins.t0_sha256)
    target = tmp_path / "recovery.json"
    written = recovery.write_v8_partition_recovery_manifest_once(manifest, target, ROOT)
    assert written == target
    parsed = json.loads(target.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "V8_PARTITION_RECOVERY_MANIFEST_V1"
    assert parsed["original_manifest_byte_exact_recovered"] is False
    assert parsed["original_partition_block_identity_recovered"] is True
    with pytest.raises(recovery.V8PartitionRecoveryBlocked, match="RECOVERY_ARTIFACT_ALREADY_EXISTS"):
        recovery.write_v8_partition_recovery_manifest_once(manifest, target, ROOT)


def test_writer_rejects_byte_exact_claim_and_bad_manifest_hash(tmp_path, monkeypatch):
    _frame, _rows, _blocks, pins, parser, _calls = _fixture()
    manifest = _build(pins, parser)
    monkeypatch.setattr(recovery, "EXPECTED_BLOCK_SHA256", dict(pins.block_sha256))
    monkeypatch.setattr(recovery, "EXPECTED_ELIGIBLE_COUNT", pins.eligible_count)
    monkeypatch.setattr(recovery, "EXPECTED_ELIGIBLE_SHA256", pins.eligible_sha256)
    monkeypatch.setattr(recovery, "EXPECTED_T0_SHA256", pins.t0_sha256)
    bad_claim = dict(manifest, original_manifest_byte_exact_recovered=True)
    with pytest.raises(recovery.V8PartitionRecoveryBlocked, match="RECOVERY_BYTE_EXACT_CLAIM_PROHIBITED"):
        recovery.write_v8_partition_recovery_manifest_once(bad_claim, tmp_path / "bad.json", ROOT)
    bad_hash = dict(manifest, manifest_sha256="0" * 64)
    with pytest.raises(recovery.V8PartitionRecoveryBlocked, match="RECOVERY_MANIFEST_HASH_MISMATCH"):
        recovery.write_v8_partition_recovery_manifest_once(bad_hash, tmp_path / "bad-hash.json", ROOT)
    assert not list(tmp_path.iterdir())


def test_safe_terminal_output_never_contains_recovered_identities():
    _frame, _rows, blocks, pins, parser, _calls = _fixture()
    manifest = _build(pins, parser)
    safe = recovery.safe_recovery_status(
        accepted=True,
        eligible_ticker_count=manifest["eligible_ticker_count"],
        eligible_ticker_list_sha256=manifest["eligible_ticker_list_sha256"],
        block_hashes=manifest["trusted_block_ticker_list_sha256"],
    )
    text = json.dumps(safe, ensure_ascii=False)
    assert safe["network_requests"] == 0
    assert safe["sealed_identity_values_included"] is False
    assert "block_assignments" not in safe
    assert all(not isinstance(value, list) for value in safe.values())
    assert "block_assignments" not in text


def test_synthetic_mode_is_caller_supplied_and_has_no_network_boundary():
    _frame, _rows, _blocks, pins, parser, calls = _fixture()
    manifest = _build(pins, parser, raw_source_bytes=b"synthetic only")
    assert calls == [b"synthetic only"]
    assert manifest["recovery_source_fingerprint"]["raw_sha256"] == hashlib.sha256(b"synthetic only").hexdigest()
    source = Path(recovery.__file__).read_text(encoding="utf-8")
    assert "urlopen(" not in source
    assert "requests.get(" not in source
