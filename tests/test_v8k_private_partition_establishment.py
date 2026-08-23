import hashlib
import hmac
import inspect
import json
import os
import socket
import urllib.request
import http.client
from datetime import datetime, timezone

import pandas as pd
import pytest

from src import v8k_private_partition_establishment as m
from src import v8_partition as partition
from scripts import run_v8k_private_partition_establishment as runner


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        raise AssertionError("REAL_NETWORK_FORBIDDEN")
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", blocked)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", blocked)


def auth(*, design_commit=None, support_sha="a" * 40, source_raw_sha256=None):
    return m.build_authorization_identity(
        design_commit=design_commit or m.FROZEN_DESIGN_COMMIT,
        support_sha=support_sha,
        source_raw_sha256=source_raw_sha256 or m.STAGE1_SOURCE_RAW_SHA256,
    )


def _synthetic_codes(total):
    pool = []
    for letter in "STUVWXYZ":
        for i in range(1000):
            pool.append(f"{letter}{i:03d}")
    ordered = sorted(pool, key=lambda code: hashlib.sha256(code.encode("utf-8")).hexdigest())
    assert total <= len(ordered)
    return ordered[:total]


def _build_fixture_from_codes(tmp_path, codes, *, block_size=300, tag="fixture"):
    ordered = partition.canonical_order(codes)
    t0_codes = ordered[:block_size]
    rows_by_code = {code: {"code": code, "market": "プライム（内国株式）", "industry": "SYN"} for code in ordered}
    t0_rows = [rows_by_code[code] for code in t0_codes]
    csv_bytes = partition.build_universe_csv_bytes(t0_rows)
    manifest_path = tmp_path / f"V4_UNIVERSE_MANIFEST_{tag}.json"
    universe_path = tmp_path / f"V4_UNIVERSE_{tag}.csv"
    manifest = {
        "source_host": "www.jpx.co.jp",
        "source_page": "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html",
        "raw_file_sha256": "a" * 64,
        "universe_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "ticker_list_sha256": partition.ticker_list_sha256(t0_codes),
        "selection_rule": "synthetic",
        "selected_count": block_size,
        "eligible_current_only": block_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    universe_path.write_bytes(csv_bytes)
    frame = pd.DataFrame(
        [
            {
                "コード": code,
                "銘柄名": "SYN",
                "市場・区分": rows_by_code[code]["market"],
                "33業種区分": rows_by_code[code]["industry"],
            }
            for code in ordered
        ]
    )
    raw = ("raw-bytes-" + tag + "-" + str(len(ordered))).encode("utf-8")
    eligible_list_sha256 = partition.ticker_list_sha256(ordered)
    return {
        "raw": raw,
        "parser": lambda _raw: frame,
        "manifest_path": manifest_path,
        "universe_path": universe_path,
        "eligible_count": len(ordered),
        "eligible_list_sha256": eligible_list_sha256,
        "source_raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _build_fixture(tmp_path, total, **kwargs):
    return _build_fixture_from_codes(tmp_path, _synthetic_codes(total), **kwargs)


def _patch_stage1_constants(monkeypatch, fixture):
    monkeypatch.setattr(m, "STAGE1_SOURCE_RAW_SHA256", fixture["source_raw_sha256"])
    monkeypatch.setattr(m, "STAGE1_ELIGIBLE_TICKER_COUNT", fixture["eligible_count"])
    monkeypatch.setattr(m, "STAGE1_ELIGIBLE_TICKER_LIST_SHA256", fixture["eligible_list_sha256"])


NOW = lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)


def _establish(tmp_path, fixture, *, gate_dir="gate", private_dir="private", seed_generator=None, support_sha="a" * 40):
    kwargs = dict(
        raw_authorization=auth(source_raw_sha256=fixture["source_raw_sha256"], support_sha=support_sha),
        support_sha=support_sha,
        raw_source_bytes=fixture["raw"],
        parse_source_table=fixture["parser"],
        v4_manifest_path=fixture["manifest_path"],
        v4_universe_csv_path=fixture["universe_path"],
        gate_state_root=tmp_path / gate_dir,
        private_state_root=tmp_path / private_dir,
        now=NOW,
    )
    if seed_generator is not None:
        kwargs["seed_generator"] = seed_generator
    return m._establish_for_test(**kwargs)


# --- frozen bindings / authorization grammar -------------------------------


def test_frozen_bindings_exact_values():
    assert m.FROZEN_DESIGN_COMMIT == "570d43ced5cb5268e31057231b9326779b09be58"
    assert m.FROZEN_DESIGN_BLOB == "e203ec6ade9d917d2e23d22528e0b41fed28c09a"
    assert m.STAGE1_SOURCE_RAW_SHA256 == "6e401867d9ddf2524e4752f08fd3e3e434cd308c6d423839ca6e24fc7b1e1653"
    assert m.STAGE1_ELIGIBLE_TICKER_COUNT == 3110
    assert m.STAGE1_ELIGIBLE_TICKER_LIST_SHA256 == "37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405"
    assert m.STAGE1_T0_REPRODUCTION_STATUS == "PASS"
    assert m.STAGE1_SUPPORT_SHA == "7fa38a6f74d631f7e1de37fae16fde944e18c580"
    assert m.GATE == "HUMAN_V8K_PRIVATE_PARTITION_GENERATION_GATE"
    assert (m.BLOCK_SIZE, m.MINIMUM_FRESH_POOL, m.SEED_BYTES) == (300, 900, 32)
    assert m.KEY_MATERIAL == (
        b"V8K_PRIVATE_PARTITION_GENERATION_GATE_RECEIPT_KEY_V1\x00"
        b"ta1k1-arakawa/stock-analyzer\x00V8K_HISTORICAL_RESEARCH\x00"
        b"HUMAN_V8K_PRIVATE_PARTITION_GENERATION_GATE"
    )
    assert m.receipt_key() == hashlib.sha256(m.KEY_MATERIAL).hexdigest()


def test_authorization_grammar_and_source_binding():
    identity = m.build_authorization_identity(design_commit=m.FROZEN_DESIGN_COMMIT, support_sha="b" * 40, source_raw_sha256="c" * 64)
    assert identity == (
        "V8K_HUMAN_AUTHORIZE_PRIVATE_PARTITION_GENERATION_AT_"
        + m.FROZEN_DESIGN_COMMIT
        + "_WITH_"
        + "b" * 40
        + "_SOURCE_"
        + "c" * 64
    )
    assert m.validate_authorization(
        identity, design_commit=m.FROZEN_DESIGN_COMMIT, support_sha="b" * 40, source_raw_sha256="c" * 64
    ) == hashlib.sha256(identity.encode()).hexdigest()
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="FROZEN_DESIGN_COMMIT_MISMATCH"):
        m.validate_authorization(identity, design_commit="d" * 40, support_sha="b" * 40, source_raw_sha256="c" * 64)
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="AUTHORIZATION_GRAMMAR_INVALID"):
        m.validate_authorization("wrong", design_commit=m.FROZEN_DESIGN_COMMIT, support_sha="b" * 40, source_raw_sha256="c" * 64)
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="AUTHORIZATION_GRAMMAR_INVALID"):
        m.build_authorization_identity(design_commit="short", support_sha="b" * 40, source_raw_sha256="c" * 64)


# --- Stage-1 source hash/list/count mismatch --------------------------------


def test_source_hash_list_count_mismatch_fail_closed(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1300, tag="mismatch")
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="STAGE1_SOURCE_RAW_SHA256_MISMATCH"):
        m.verify_stage1_binding_and_compute_fresh_pool(
            raw_source_bytes=fixture["raw"],
            parse_source_table=fixture["parser"],
            v4_manifest_path=fixture["manifest_path"],
            v4_universe_csv_path=fixture["universe_path"],
        )
    monkeypatch.setattr(m, "STAGE1_SOURCE_RAW_SHA256", fixture["source_raw_sha256"])
    monkeypatch.setattr(m, "STAGE1_ELIGIBLE_TICKER_COUNT", fixture["eligible_count"] + 1)
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="STAGE1_ELIGIBLE_TICKER_COUNT_MISMATCH"):
        m.verify_stage1_binding_and_compute_fresh_pool(
            raw_source_bytes=fixture["raw"],
            parse_source_table=fixture["parser"],
            v4_manifest_path=fixture["manifest_path"],
            v4_universe_csv_path=fixture["universe_path"],
        )
    monkeypatch.setattr(m, "STAGE1_ELIGIBLE_TICKER_COUNT", fixture["eligible_count"])
    monkeypatch.setattr(m, "STAGE1_ELIGIBLE_TICKER_LIST_SHA256", "f" * 64)
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="STAGE1_ELIGIBLE_TICKER_LIST_SHA256_MISMATCH"):
        m.verify_stage1_binding_and_compute_fresh_pool(
            raw_source_bytes=fixture["raw"],
            parse_source_table=fixture["parser"],
            v4_manifest_path=fixture["manifest_path"],
            v4_universe_csv_path=fixture["universe_path"],
        )


def test_t0_and_legacy_exclusions(tmp_path, monkeypatch):
    legacy_codes = list(partition.LEGACY_EXPOSED_TICKERS_OUTSIDE_T0)
    fixture = _build_fixture_from_codes(tmp_path, _synthetic_codes(1300) + legacy_codes, tag="legacy")
    _patch_stage1_constants(monkeypatch, fixture)
    ordered_codes, t0_tickers, fresh_pool = m.verify_stage1_binding_and_compute_fresh_pool(
        raw_source_bytes=fixture["raw"],
        parse_source_table=fixture["parser"],
        v4_manifest_path=fixture["manifest_path"],
        v4_universe_csv_path=fixture["universe_path"],
    )
    assert len(t0_tickers) == 300
    assert set(fresh_pool).isdisjoint(legacy_codes)
    assert len(fresh_pool) == len(ordered_codes) - len(set(t0_tickers) | set(legacy_codes))


def test_fresh_pool_insufficient_blocks_before_gate_and_seed(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1199, tag="insufficient")  # T0=300, fresh=899 < 900
    _patch_stage1_constants(monkeypatch, fixture)

    def seed_must_not_run():
        raise AssertionError("SEED_MUST_NOT_BE_GENERATED")

    gate_root = tmp_path / "gate"
    with pytest.raises(m.V8KPrivatePartitionBlocked) as excinfo:
        _establish(tmp_path, fixture, seed_generator=seed_must_not_run)
    assert excinfo.value.reason == "FRESH_POOL_INSUFFICIENT"
    assert excinfo.value.failure_class == "DATA_QUALITY_FAILURE"
    assert not gate_root.exists()


# --- HMAC allocation ---------------------------------------------------------


def test_exact_hmac_allocation_with_fixed_synthetic_seed():
    seed = bytes(range(32))
    fresh_pool = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF", "GGGG"]
    blocks = m.allocate_v8k_blocks(fresh_pool, seed, block_size=2)
    expected_keys = {
        code: hmac.new(seed, ("V8K_PARTITION_ASSIGN_V1\0" + code).encode("utf-8"), hashlib.sha256).digest()
        for code in fresh_pool
    }
    ordered = sorted(fresh_pool, key=lambda code: (expected_keys[code], code))
    assert blocks["T1"] == ordered[:2]
    assert blocks["T2"] == ordered[2:4]
    assert blocks["T3"] == ordered[4:6]
    assert blocks["T_spare"] == ordered[6:]


def test_allocate_blocks_sizes_300_300_300_remainder(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1300, tag="sizes")  # T0=300, fresh=1000
    _patch_stage1_constants(monkeypatch, fixture)
    evidence = _establish(tmp_path, fixture)
    assert (evidence["t0_count"], evidence["t1_count"], evidence["t2_count"], evidence["t3_count"], evidence["t_spare_count"]) == (
        300,
        300,
        300,
        300,
        100,
    )


def test_allocate_blocks_duplicate_ticker_fails_closed():
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="FRESH_POOL_DUPLICATE_TICKER"):
        m.allocate_v8k_blocks(["AAAA", "AAAA", "BBBB"], bytes(32), block_size=1)


def test_allocate_blocks_allocation_key_collision_fails_closed(monkeypatch):
    monkeypatch.setattr(m, "_allocation_key", lambda _seed, _code: b"\x00" * 32)
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="ALLOCATION_KEY_COLLISION"):
        m.allocate_v8k_blocks(["AAAA", "BBBB", "CCCC"], bytes(32), block_size=1)


def test_allocate_blocks_rejects_wrong_seed_length():
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="SEED_BYTES_INVALID"):
        m.allocate_v8k_blocks(["AAAA"], b"short", block_size=1)


# --- manifest mechanical verification ---------------------------------------


def _small_manifest(**overrides):
    fields = dict(
        support_sha="a" * 40,
        auth_hash="b" * 64,
        source_raw_sha256=m.STAGE1_SOURCE_RAW_SHA256,
        eligible_ticker_count=1300,
        eligible_ticker_list_sha256="c" * 64,
        seed=bytes(32),
        fresh_pool_count=1000,
        t0=[f"Z{i:03d}" for i in range(300)],
        t1=[f"A{i:03d}" for i in range(300)],
        t2=[f"B{i:03d}" for i in range(300)],
        t3=[f"C{i:03d}" for i in range(300)],
        t_spare=[f"D{i:03d}" for i in range(100)],
        now=NOW,
    )
    fields.update(overrides)
    return m._build_manifest(**fields)


def test_manifest_self_hash_mismatch_fails_closed():
    manifest = dict(_small_manifest())
    manifest["fresh_pool_count"] = 999
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="MANIFEST_SHA_MISMATCH"):
        m._verify_manifest(manifest)


def test_manifest_verification_detects_block_overlap():
    t1 = [f"A{i:03d}" for i in range(300)]
    t2 = [f"B{i:03d}" for i in range(299)] + [t1[0]]
    manifest = _small_manifest(t1=t1, t2=t2)
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="BLOCK_OVERLAP_DETECTED"):
        m._verify_manifest(manifest)


def test_manifest_verification_detects_block_size_invalid():
    manifest = dict(_small_manifest())
    manifest["block_sizes"] = dict(manifest["block_sizes"])
    manifest["block_sizes"]["T1"] = 299
    manifest["manifest_sha256"] = partition.canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="BLOCK_SIZE_INVALID"):
        m._verify_manifest(manifest)


def test_manifest_write_once_and_read_round_trip(tmp_path):
    manifest = _small_manifest()
    m._write_manifest_once(tmp_path, manifest)
    reread = m._read_manifest(tmp_path)
    assert reread == manifest
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="MANIFEST_ALREADY_EXISTS"):
        m._write_manifest_once(tmp_path, manifest)


# --- gate / seed one-shot discipline -----------------------------------------


def test_no_seed_before_durable_receipt(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1300, tag="order")
    _patch_stage1_constants(monkeypatch, fixture)
    calls = []
    real_consume = m._consume_gate

    def wrapped_consume(*args, **kwargs):
        calls.append("receipt")
        return real_consume(*args, **kwargs)

    monkeypatch.setattr(m, "_consume_gate", wrapped_consume)

    def seed_gen():
        calls.append("seed")
        return os.urandom(32)

    _establish(tmp_path, fixture, seed_generator=seed_gen)
    assert calls == ["receipt", "seed"]


def test_receipt_failure_prevents_seed_generation(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1300, tag="receiptfail")
    _patch_stage1_constants(monkeypatch, fixture)
    monkeypatch.setattr(m, "_consume_gate", lambda *a, **kw: (_ for _ in ()).throw(m.V8KPrivatePartitionBlocked("RECEIPT_WRITE_FAILED")))

    def seed_must_not_run():
        raise AssertionError("SEED_MUST_NOT_BE_GENERATED")

    with pytest.raises(m.V8KPrivatePartitionBlocked, match="RECEIPT_WRITE_FAILED"):
        _establish(tmp_path, fixture, seed_generator=seed_must_not_run)


def test_exactly_one_32_byte_seed_generated(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1300, tag="oneseed")
    _patch_stage1_constants(monkeypatch, fixture)
    calls = []

    def seed_gen():
        calls.append(1)
        return os.urandom(32)

    evidence = _establish(tmp_path, fixture, seed_generator=seed_gen)
    assert calls == [1]
    seed_file = m._seed_path(tmp_path / "private")
    data = seed_file.read_bytes()
    assert len(data) == 32
    assert evidence["seed_sha256"] == hashlib.sha256(data).hexdigest()


def test_seed_persistence_failure_after_receipt_is_block_closed_no_second_seed(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1300, tag="blockclosed")
    _patch_stage1_constants(monkeypatch, fixture)
    gate_root = tmp_path / "gate"
    private_root = tmp_path / "private_blocked"
    private_root.write_bytes(b"not-a-directory")  # forces mkdir/open failure below it

    with pytest.raises(m.V8KPrivatePartitionBlocked) as excinfo:
        m._establish_for_test(
            raw_authorization=auth(source_raw_sha256=fixture["source_raw_sha256"]),
            support_sha="a" * 40,
            raw_source_bytes=fixture["raw"],
            parse_source_table=fixture["parser"],
            v4_manifest_path=fixture["manifest_path"],
            v4_universe_csv_path=fixture["universe_path"],
            gate_state_root=gate_root,
            private_state_root=private_root,
            now=NOW,
        )
    assert excinfo.value.reason == "SEED_PERSISTENCE_FAILED_BLOCK_CLOSED"
    assert excinfo.value.failure_class == "IMPLEMENTATION_FAILURE"
    assert m._gate_receipt_path(gate_root).exists()

    seed_calls = []

    def seed_gen():
        seed_calls.append(1)
        return os.urandom(32)

    with pytest.raises(m.V8KPrivatePartitionBlocked, match="SEED_MISSING_AFTER_GATE_CONSUMED_BLOCK_CLOSED"):
        m._establish_for_test(
            raw_authorization=auth(source_raw_sha256=fixture["source_raw_sha256"]),
            support_sha="a" * 40,
            raw_source_bytes=fixture["raw"],
            parse_source_table=fixture["parser"],
            v4_manifest_path=fixture["manifest_path"],
            v4_universe_csv_path=fixture["universe_path"],
            gate_state_root=gate_root,
            private_state_root=private_root,
            now=NOW,
            seed_generator=seed_gen,
        )
    assert seed_calls == []


def test_deterministic_same_seed_continuation_no_reroll(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1300, tag="continuation")
    _patch_stage1_constants(monkeypatch, fixture)
    seed_calls = []

    def seed_gen():
        seed_calls.append(1)
        return os.urandom(32)

    first = _establish(tmp_path, fixture, seed_generator=seed_gen)
    second = _establish(tmp_path, fixture, seed_generator=seed_gen)
    assert seed_calls == [1]
    assert first == second


def test_existing_state_collision_before_gate_consumed_fails_closed(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1300, tag="collision")
    _patch_stage1_constants(monkeypatch, fixture)
    private_root = tmp_path / "private"
    private_root.mkdir(parents=True)
    (private_root / (m.receipt_key() + ".seed")).write_bytes(os.urandom(32))
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="EXISTING_STATE_COLLISION"):
        _establish(tmp_path, fixture)


# --- safe public evidence contract -------------------------------------------


def test_safe_evidence_never_contains_private_membership_seed_path_or_raw_auth(tmp_path, monkeypatch):
    fixture = _build_fixture(tmp_path, 1300, tag="safe")
    _patch_stage1_constants(monkeypatch, fixture)
    raw_auth = auth(source_raw_sha256=fixture["source_raw_sha256"])
    evidence = m._establish_for_test(
        raw_authorization=raw_auth,
        support_sha="a" * 40,
        raw_source_bytes=fixture["raw"],
        parse_source_table=fixture["parser"],
        v4_manifest_path=fixture["manifest_path"],
        v4_universe_csv_path=fixture["universe_path"],
        gate_state_root=tmp_path / "gate",
        private_state_root=tmp_path / "private",
        now=NOW,
    )
    assert set(evidence) == set(m.EVIDENCE_FIELDS)
    serialized = json.dumps(evidence)
    assert raw_auth not in serialized
    assert str(tmp_path) not in serialized
    for forbidden in ("block_assignments", "seed", "ticker_list", "raw_payload", "T0", "T1", "T2", "T3", "T_spare"):
        assert forbidden not in evidence
    assert len(evidence["seed_sha256"]) == 64


def test_production_api_exposes_only_raw_authorization():
    assert tuple(inspect.signature(m.establish_private_partition).parameters) == ("raw_authorization",)
    for keyword in (
        "state_root",
        "gate_state_root",
        "private_state_root",
        "seed_generator",
        "parser",
        "parse_source_table",
        "source_path",
        "v4_manifest_path",
        "v4_universe_csv_path",
        "gate_receipt",
        "support_sha",
        "now",
        "clock",
    ):
        with pytest.raises(TypeError):
            m.establish_private_partition(raw_authorization="x", **{keyword: None})


def test_production_provenance_failure_blocks_before_gate_and_seed(monkeypatch):
    monkeypatch.setattr(
        m, "production_provenance", lambda: (_ for _ in ()).throw(m.V8KPrivatePartitionBlocked("GOVERNANCE_FAILURE"))
    )
    monkeypatch.setattr(m, "_production_dependencies", lambda: (_ for _ in ()).throw(AssertionError("MUST_NOT_REACH_DEPENDENCIES")))
    with pytest.raises(m.V8KPrivatePartitionBlocked, match="GOVERNANCE_FAILURE"):
        m.establish_private_partition(raw_authorization="whatever")


def test_unexpected_exception_in_production_flow_fails_closed(monkeypatch):
    monkeypatch.setattr(m, "production_provenance", lambda: "a" * 40)
    monkeypatch.setattr(m, "_production_dependencies", lambda: (_ for _ in ()).throw(ImportError("synthetic missing dependency")))
    with pytest.raises(m.V8KPrivatePartitionBlocked) as excinfo:
        m.establish_private_partition(raw_authorization="whatever")
    assert excinfo.value.failure_class == "IMPLEMENTATION_FAILURE"
    assert "synthetic missing dependency" not in str(excinfo.value)


# --- runner: safe JSON, env-only auth, fail-closed unknown, no leakage -------


def test_runner_missing_auth_env_emits_safe_governance_json(monkeypatch, capsys):
    monkeypatch.delenv(runner.AUTH_ENV, raising=False)
    monkeypatch.setattr(runner, "establish_private_partition", lambda **_kw: (_ for _ in ()).throw(AssertionError("MUST_NOT_RUN")))
    with pytest.raises(SystemExit, match="GOVERNANCE_FAILURE"):
        runner.main()
    report = json.loads(capsys.readouterr().out)
    assert report == {
        "schema_version": "V8K_PRIVATE_PARTITION_ESTABLISHMENT_FAILURE_V1",
        "study": m.STUDY,
        "stage": "PRIVATE_PARTITION_ESTABLISHMENT",
        "execution_result": "BLOCKED",
        "failure_class": "GOVERNANCE_FAILURE",
    }


def test_runner_blocked_emits_safe_json_without_raw_reason_or_auth(monkeypatch, capsys):
    raw = "some-raw-authorization-value"
    monkeypatch.setenv(runner.AUTH_ENV, raw)
    monkeypatch.setattr(
        runner, "establish_private_partition", lambda **_kw: (_ for _ in ()).throw(runner.V8KPrivatePartitionBlocked("FRESH_POOL_INSUFFICIENT"))
    )
    with pytest.raises(SystemExit, match="DATA_QUALITY_FAILURE"):
        runner.main()
    output = capsys.readouterr().out
    assert "FRESH_POOL_INSUFFICIENT" not in output
    assert raw not in output
    assert json.loads(output) == {
        "schema_version": "V8K_PRIVATE_PARTITION_ESTABLISHMENT_FAILURE_V1",
        "study": m.STUDY,
        "stage": "PRIVATE_PARTITION_ESTABLISHMENT",
        "execution_result": "BLOCKED",
        "failure_class": "DATA_QUALITY_FAILURE",
    }


def test_runner_unexpected_exception_fails_closed_without_traceback(monkeypatch, capsys):
    raw = "auth-value"
    monkeypatch.setenv(runner.AUTH_ENV, raw)
    monkeypatch.setattr(runner, "establish_private_partition", lambda **_kw: (_ for _ in ()).throw(KeyError("some/private/leaked/path")))
    with pytest.raises(SystemExit, match="IMPLEMENTATION_FAILURE"):
        runner.main()
    output = capsys.readouterr().out
    assert "some/private/leaked/path" not in output
    assert "Traceback" not in output and "File \"" not in output and "KeyError" not in output
    assert raw not in output
    assert json.loads(output)["failure_class"] == "IMPLEMENTATION_FAILURE"


def test_runner_success_emits_safe_evidence(monkeypatch, capsys):
    raw = "auth-value"
    monkeypatch.setenv(runner.AUTH_ENV, raw)
    expected = {"schema_version": "V8K_PRIVATE_PARTITION_ESTABLISHMENT_EVIDENCE_V1", "result_classification": "COMPLETE"}
    monkeypatch.setattr(runner, "establish_private_partition", lambda **_kw: expected)
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_direct_runner_missing_authorization_is_governance_failure_without_import_error():
    import subprocess
    import sys
    from pathlib import Path

    environment = os.environ.copy()
    environment.pop(runner.AUTH_ENV, None)
    completed = subprocess.run(
        [sys.executable, "scripts/run_v8k_private_partition_establishment.py"],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "GOVERNANCE_FAILURE" in completed.stderr
    assert "ModuleNotFoundError" not in completed.stderr


# --- failure-class mapping ---------------------------------------------------


def test_failure_class_mapping_fail_closed_for_unknown_reason():
    assert m.public_failure_class("SOME_NEW_UNSPECIFIED_REASON") == "IMPLEMENTATION_FAILURE"
    exc = m.V8KPrivatePartitionBlocked("SOME_NEW_UNSPECIFIED_REASON")
    assert exc.failure_class == "IMPLEMENTATION_FAILURE"
