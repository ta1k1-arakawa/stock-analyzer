from __future__ import annotations

import hashlib
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src import v8_partition as partition

ROOT = Path(__file__).resolve().parents[1]

# Deliberately far smaller than the frozen production value (300) -- these
# tests exercise the partition-building *logic*, not production block-size
# semantics (V8_HISTORICAL_RESEARCH_DESIGN.md Sec 5.1).
BLOCK_SIZE = 5


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def _ticker_list_sha(tickers: list[str]) -> str:
    return hashlib.sha256(("\n".join(tickers) + "\n").encode("utf-8")).hexdigest()


def ordered_codes(total: int, *, start: int = 1000, pool: int = 3000) -> list[str]:
    """Codes pre-sorted by the module's own canonical order, so the first
    ``total`` are deterministic and independent of numeric magnitude."""
    candidates = [str(start + i) for i in range(pool)]
    return sorted(candidates, key=lambda code: hashlib.sha256(code.encode("utf-8")).hexdigest())[:total]


@pytest.fixture(scope="module")
def all_codes() -> list[str]:
    return ordered_codes(BLOCK_SIZE * 4 + 20)


@pytest.fixture(scope="module")
def t0_codes(all_codes) -> list[str]:
    return all_codes[:BLOCK_SIZE]


@pytest.fixture(scope="module")
def fresh_codes(all_codes) -> list[str]:
    return all_codes[BLOCK_SIZE:]


def t0_rows_for_csv(codes: list[str], market: str = "プライム（内国株式）") -> list[dict[str, str]]:
    return [{"code": code, "market": market, "industry": "SYN_INDUSTRY"} for code in codes]


def build_frame(t0_codes: list[str], fresh_codes: list[str], *, t0_market: str = "プライム（内国株式）") -> pd.DataFrame:
    rows = [
        {"コード": code, "銘柄名": "SYN", "市場・区分": t0_market, "33業種区分": "SYN_INDUSTRY"}
        for code in t0_codes
    ]
    rows += [
        {"コード": code, "銘柄名": "SYN", "市場・区分": "スタンダード（内国株式）", "33業種区分": "SYN_INDUSTRY"}
        for code in fresh_codes
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def v4_fixture(tmp_path_factory, t0_codes):
    workspace = tmp_path_factory.mktemp("v8-partition-v4fixture")
    csv_bytes = partition.build_universe_csv_bytes(t0_rows_for_csv(t0_codes))
    universe_csv_path = workspace / "V4_UNIVERSE.csv"
    universe_csv_path.write_bytes(csv_bytes)

    manifest_path = workspace / "V4_UNIVERSE_MANIFEST.json"
    import json

    manifest_path.write_bytes(json.dumps({
        "source_host": "www.jpx.co.jp",
        "source_page": "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html",
        "raw_file_sha256": hashlib.sha256(b"SYNTHETIC_RAW_SOURCE_BYTES").hexdigest(),
        "universe_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "ticker_list_sha256": _ticker_list_sha(t0_codes),
        "selection_rule": "synthetic fixture",
        "selected_count": BLOCK_SIZE,
        "eligible_current_only": BLOCK_SIZE * 4 + 20,
    }, ensure_ascii=False).encode("utf-8"))

    return {
        "manifest_path": manifest_path,
        "universe_csv_path": universe_csv_path,
        "raw_source_bytes": b"SYNTHETIC_RAW_SOURCE_BYTES",
        "universe_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
    }


def build_kwargs(v4_fixture, frame, *, raw_source_bytes=None):
    return dict(
        raw_source_bytes=raw_source_bytes if raw_source_bytes is not None else v4_fixture["raw_source_bytes"],
        parse_source_table=lambda _raw: frame,
        v4_manifest_path=v4_fixture["manifest_path"],
        v4_universe_csv_path=v4_fixture["universe_csv_path"],
        source_url="https://www.jpx.co.jp/synthetic/data_j.xls",
        source_acquisition_utc=datetime(2026, 8, 9, tzinfo=timezone.utc),
        clock=lambda: datetime(2026, 8, 9, 1, 0, 0, tzinfo=timezone.utc),
        partition_implementation_git_commit="a" * 40,
        block_size=BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Eligible-universe reconstruction / deterministic ordering
# ---------------------------------------------------------------------------


def test_canonical_order_is_deterministic_and_independent_of_input_order():
    codes = ["4188", "1570", "9999", "0001"]
    reversed_codes = list(reversed(codes))
    assert partition.canonical_order(codes) == partition.canonical_order(reversed_codes)


def test_canonical_order_matches_hash_key():
    codes = ["4188", "1570", "9999", "0001"]
    ordered = partition.canonical_order(codes)
    expected = sorted(codes, key=lambda c: (hashlib.sha256(c.encode("utf-8")).hexdigest(), c))
    assert ordered == expected


def test_parse_eligible_universe_matches_free_prototype_semantics(t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    rows, reasons = partition.parse_eligible_universe(frame)
    assert len(rows) == len(t0_codes) + len(fresh_codes)
    assert reasons["eligible_current_only"] == len(rows)
    assert reasons["excluded_non_prime_standard"] == 0


def test_parse_eligible_universe_excludes_non_prime_standard():
    frame = pd.DataFrame([
        {"コード": "1234", "銘柄名": "X", "市場・区分": "その他", "33業種区分": "IND"},
        {"コード": "5678", "銘柄名": "Y", "市場・区分": "プライム（内国株式）", "33業種区分": "IND"},
    ])
    rows, reasons = partition.parse_eligible_universe(frame)
    assert [r["code"] for r in rows] == ["5678"]
    assert reasons["excluded_non_prime_standard"] == 1


def test_parse_eligible_universe_deduplicates_codes():
    frame = pd.DataFrame([
        {"コード": "1234", "銘柄名": "X", "市場・区分": "プライム（内国株式）", "33業種区分": "IND"},
        {"コード": "1234", "銘柄名": "X2", "市場・区分": "プライム（内国株式）", "33業種区分": "IND"},
    ])
    rows, _ = partition.parse_eligible_universe(frame)
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Manifest construction: happy path, determinism
# ---------------------------------------------------------------------------


def test_build_partition_manifest_passes_and_has_correct_shape(v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    assert manifest["source_reproduction_status"] == "PASS"
    assert set(manifest) == set(partition.MANIFEST_FIELDS)
    assert manifest["block_sizes"] == {"T0": BLOCK_SIZE, "T1": BLOCK_SIZE, "T2": BLOCK_SIZE, "T3": BLOCK_SIZE, "T_spare": len(fresh_codes) - 3 * BLOCK_SIZE}


def test_identical_input_produces_byte_identical_manifest(v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    kwargs = build_kwargs(v4_fixture, frame)
    manifest_1 = partition.build_partition_manifest(**kwargs)
    manifest_2 = partition.build_partition_manifest(**kwargs)
    assert partition.canonical_json_bytes(manifest_1) == partition.canonical_json_bytes(manifest_2)


def test_source_provenance_fields_present_and_correct(v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    assert manifest["source_host"] == "www.jpx.co.jp"
    assert manifest["source_url"] == "https://www.jpx.co.jp/synthetic/data_j.xls"
    assert manifest["source_raw_sha256"] == hashlib.sha256(v4_fixture["raw_source_bytes"]).hexdigest()
    assert manifest["source_raw_byte_count"] == len(v4_fixture["raw_source_bytes"])
    assert manifest["expected_v4_source_raw_sha256"] == manifest["source_raw_sha256"]
    assert manifest["design_commit"] == partition.DESIGN_COMMIT
    assert manifest["partition_implementation_git_commit"] == "a" * 40
    assert manifest["study_name"] == partition.STUDY_NAME
    assert manifest["p_hist_start"] == "2016-04-01"
    assert manifest["p_hist_end"] == "2025-12-31"


# ---------------------------------------------------------------------------
# Fail-closed BLOCKs
# ---------------------------------------------------------------------------


def test_source_raw_hash_mismatch_blocks_before_any_allocation(v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.build_partition_manifest(**build_kwargs(v4_fixture, frame, raw_source_bytes=b"WRONG_BYTES"))
    assert excinfo.value.reason == "V8_PARTITION_SOURCE_NOT_REPRODUCIBLE"


def test_t0_parity_mismatch_blocks(v4_fixture, t0_codes, fresh_codes):
    # A different market string changes the reconstructed CSV bytes, so the
    # first BLOCK_SIZE tickers no longer byte-reproduce V4_UNIVERSE.csv.
    frame = build_frame(t0_codes, fresh_codes, t0_market="スタンダード（内国株式）")
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    assert excinfo.value.reason == "V8_T0_REPRODUCTION_MISMATCH"


def test_t0_parity_mismatch_from_missing_ticker_blocks(v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes[:-1], fresh_codes)
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    assert excinfo.value.reason == "V8_T0_REPRODUCTION_MISMATCH"


def test_failure_before_block_assignment_write(v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes, t0_market="スタンダード（内国株式）")
    with pytest.raises(partition.V8PartitionBlocked):
        manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
        # unreachable: build_partition_manifest must raise before returning
        assert "block_assignments" not in manifest  # pragma: no cover


def test_v4_universe_csv_provenance_mismatch_blocks(tmp_path, v4_fixture, t0_codes, fresh_codes):
    tampered_csv = tmp_path / "V4_UNIVERSE.csv"
    tampered_csv.write_bytes(b"ticker,market,industry\n9999,X,Y\n")
    frame = build_frame(t0_codes, fresh_codes)
    kwargs = build_kwargs(v4_fixture, frame)
    kwargs["v4_universe_csv_path"] = tampered_csv
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.build_partition_manifest(**kwargs)
    assert excinfo.value.reason == "V4_UNIVERSE_CSV_PROVENANCE_MISMATCH"


def test_v4_manifest_missing_field_blocks(tmp_path, t0_codes):
    import json

    incomplete = tmp_path / "incomplete_manifest.json"
    incomplete.write_bytes(json.dumps({"source_host": "www.jpx.co.jp"}).encode())
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.load_v4_provenance(incomplete)
    assert excinfo.value.reason == "V4_MANIFEST_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# allocate_fresh_blocks: sizing, disjointness, exclusions, duplicates
# ---------------------------------------------------------------------------


def test_allocate_fresh_blocks_exact_sizes_and_disjoint(t0_codes, fresh_codes):
    all_codes_ordered = t0_codes + fresh_codes
    blocks = partition.allocate_fresh_blocks(all_codes_ordered, t0_codes, block_size=BLOCK_SIZE)
    assert len(blocks["T0"]) == BLOCK_SIZE
    assert len(blocks["T1"]) == BLOCK_SIZE
    assert len(blocks["T2"]) == BLOCK_SIZE
    assert len(blocks["T3"]) == BLOCK_SIZE
    seen: set[str] = set()
    for name in ("T0", "T1", "T2", "T3", "T_spare"):
        block_set = set(blocks[name])
        assert not (seen & block_set), f"{name} overlaps a previous block"
        seen |= block_set


def test_allocate_fresh_blocks_t_spare_is_deterministic(t0_codes, fresh_codes):
    all_codes_ordered = t0_codes + fresh_codes
    blocks_1 = partition.allocate_fresh_blocks(all_codes_ordered, t0_codes, block_size=BLOCK_SIZE)
    blocks_2 = partition.allocate_fresh_blocks(all_codes_ordered, t0_codes, block_size=BLOCK_SIZE)
    assert blocks_1["T_spare"] == blocks_2["T_spare"]
    assert blocks_1 == blocks_2


def test_allocate_fresh_blocks_excludes_legacy_exposed_tickers(t0_codes):
    legacy = partition.LEGACY_EXPOSED_TICKERS_OUTSIDE_T0[0]
    fresh_with_legacy = [legacy] + [f"F{i:03d}" for i in range(BLOCK_SIZE * 3 + 5)]
    all_codes_ordered = t0_codes + fresh_with_legacy
    blocks = partition.allocate_fresh_blocks(all_codes_ordered, t0_codes, block_size=BLOCK_SIZE)
    for name in ("T1", "T2", "T3", "T_spare"):
        assert legacy not in blocks[name]


def test_allocate_fresh_blocks_duplicate_ticker_blocks(t0_codes):
    fresh_with_dup = [f"F{i:03d}" for i in range(BLOCK_SIZE * 3)] + ["F000"]
    all_codes_ordered = t0_codes + fresh_with_dup
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.allocate_fresh_blocks(all_codes_ordered, t0_codes, block_size=BLOCK_SIZE)
    assert excinfo.value.reason == "V8_ELIGIBLE_LIST_DUPLICATE_TICKER"


def test_allocate_fresh_blocks_insufficient_pool_blocks(t0_codes):
    too_few = [f"F{i:03d}" for i in range(BLOCK_SIZE)]  # not enough for 3 full blocks
    all_codes_ordered = t0_codes + too_few
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.allocate_fresh_blocks(all_codes_ordered, t0_codes, block_size=BLOCK_SIZE)
    assert excinfo.value.reason == "V8_ELIGIBLE_POOL_INSUFFICIENT"


# ---------------------------------------------------------------------------
# Manifest write-once / self-hash verification / output-root safety
# ---------------------------------------------------------------------------


def test_manifest_sha256_self_verifies(v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    stated = manifest["manifest_sha256"]
    recomputed = partition.canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    assert stated == recomputed


def test_write_partition_manifest_once_then_read_back(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    output_path = tmp_path / "output" / "partition_manifest.json"
    partition.write_partition_manifest_once(output_path, manifest, repository_root=ROOT)
    reread = partition.read_partition_manifest(output_path)
    assert reread == manifest


def test_write_partition_manifest_once_rejects_overwrite(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    output_path = tmp_path / "output" / "partition_manifest.json"
    partition.write_partition_manifest_once(output_path, manifest, repository_root=ROOT)
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.write_partition_manifest_once(output_path, manifest, repository_root=ROOT)
    assert excinfo.value.reason == "PARTITION_MANIFEST_ALREADY_EXISTS"


def test_write_partition_manifest_race_never_overwrites_existing_destination(
    tmp_path, v4_fixture, t0_codes, fresh_codes, monkeypatch
):
    frame = build_frame(t0_codes, fresh_codes)
    manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    output_path = tmp_path / "output" / "partition_manifest.json"
    original_link = partition.os.link
    competing_bytes = b"existing-formal-manifest-must-not-change"

    def competing_publish(src, dst):
        Path(dst).write_bytes(competing_bytes)
        return original_link(src, dst)

    monkeypatch.setattr(partition.os, "link", competing_publish)
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.write_partition_manifest_once(output_path, manifest, repository_root=ROOT)
    assert excinfo.value.reason == "PARTITION_MANIFEST_ALREADY_EXISTS"
    assert output_path.read_bytes() == competing_bytes


@pytest.mark.parametrize(
    ("status", "head", "origin", "reason"),
    [
        (" M src/v8_partition.py\n", "a" * 40, "a" * 40, "PRODUCTION_GIT_WORKTREE_DIRTY"),
        ("", None, "a" * 40, "PRODUCTION_GIT_HEAD_UNAVAILABLE"),
        ("", "a" * 40, None, "PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE"),
        ("", "a" * 40, "b" * 40, "PRODUCTION_GIT_HEAD_NOT_ORIGIN"),
    ],
)
def test_production_git_provenance_blocks_invalid_local_state(monkeypatch, status, head, origin, reason):
    class Result:
        def __init__(self, returncode, stdout):
            self.returncode = returncode
            self.stdout = stdout

    outcomes = iter((
        Result(0, status),
        Result(0 if head is not None else 1, (head or "") + "\n"),
        Result(0 if origin is not None else 1, (origin or "") + "\n"),
    ))
    monkeypatch.setattr(partition.subprocess, "run", lambda *args, **kwargs: next(outcomes))
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.resolve_verified_production_git_commit(ROOT)
    assert excinfo.value.reason == reason


def test_production_git_provenance_accepts_clean_matching_local_origin(monkeypatch):
    class Result:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    expected = "a" * 40
    outcomes = iter((Result(""), Result(expected + "\n"), Result(expected + "\n")))
    monkeypatch.setattr(partition.subprocess, "run", lambda *args, **kwargs: next(outcomes))
    assert partition.resolve_verified_production_git_commit(ROOT) == expected


def test_write_partition_manifest_once_rejects_in_repo_output(v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    inside_repo = ROOT / "tmp-v8-partition-test-output.json"
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.write_partition_manifest_once(inside_repo, manifest, repository_root=ROOT)
    assert excinfo.value.reason == "OUTPUT_PATH_INSIDE_SOURCE_REPOSITORY"
    assert not inside_repo.exists()


def test_write_partition_manifest_once_rejects_relative_path(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = build_frame(t0_codes, fresh_codes)
    manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.write_partition_manifest_once("relative/path.json", manifest, repository_root=ROOT)
    assert excinfo.value.reason == "OUTPUT_PATH_NOT_ABSOLUTE"


def test_read_partition_manifest_detects_tampered_hash(tmp_path, v4_fixture, t0_codes, fresh_codes):
    import json

    frame = build_frame(t0_codes, fresh_codes)
    manifest = partition.build_partition_manifest(**build_kwargs(v4_fixture, frame))
    output_path = tmp_path / "output" / "partition_manifest.json"
    partition.write_partition_manifest_once(output_path, manifest, repository_root=ROOT)

    tampered = json.loads(output_path.read_text(encoding="utf-8"))
    tampered["eligible_ticker_count"] = tampered["eligible_ticker_count"] + 1
    output_path.write_bytes(partition.canonical_json_bytes(tampered))

    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.read_partition_manifest(output_path)
    assert excinfo.value.reason == "MANIFEST_SHA_MISMATCH"


def test_read_partition_manifest_rejects_schema_drift(tmp_path):
    import json

    bad_path = tmp_path / "bad.json"
    bad_path.write_bytes(json.dumps({"only_one_field": True}).encode())
    with pytest.raises(partition.V8PartitionBlocked) as excinfo:
        partition.read_partition_manifest(bad_path)
    assert excinfo.value.reason == "MANIFEST_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# Static safety
# ---------------------------------------------------------------------------


def test_module_has_no_network_imports():
    text = Path(partition.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip().lower() for line in text.splitlines() if line.strip().startswith(("import ", "from "))]
    for token in ("re" + "quests", "url" + "lib", "http" + "x", "aio" + "http", "so" + "cket", "y" + "finance"):
        assert not any(token in line for line in import_lines), token


def test_module_never_touches_v7():
    text = Path(partition.__file__).read_text(encoding="utf-8")
    assert "v7_" not in text.lower()
