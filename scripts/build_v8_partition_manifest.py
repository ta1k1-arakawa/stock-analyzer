"""Synthetic-only V8 partition manifest builder check.

This CLI has no real-source, real-output-root, or real-network option. It
builds a fully local synthetic JPX-listing fixture and a synthetic
``V4_UNIVERSE_MANIFEST.json``/``V4_UNIVERSE.csv`` pair inside a temporary
workspace, drives the production partition builder end to end (source-hash
reproduction guard, eligible-universe reconstruction, T0 reproduction guard,
fresh-block allocation, write-once manifest publish), and never touches the
real JPX host, the real V4 universe files, or any real private V8 storage.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v8_partition import (
    V8PartitionBlocked,
    build_partition_manifest,
    read_partition_manifest,
    write_partition_manifest_once,
)

# Synthetic block size, deliberately far smaller than the frozen production
# value (300) -- this CLI proves the pipeline's *logic*, not production block
# size semantics, and must run in well under a second.
SYNTHETIC_BLOCK_SIZE = 5
SYNTHETIC_SOURCE_URL = "https://www.jpx.co.jp/synthetic-only/data_j.xls"


def _ordered_synthetic_codes(total: int) -> list[str]:
    """A pool of 4-digit codes pre-sorted by the same canonical order the
    partition builder itself uses, so the first ``SYNTHETIC_BLOCK_SIZE`` are
    guaranteed to land in T0 and the rest are guaranteed to sort after them
    -- computed here rather than assumed, since SHA-256 order has no
    relationship to numeric code order."""
    import hashlib

    candidates = [str(1000 + i) for i in range(2000)]
    ordered = sorted(candidates, key=lambda code: hashlib.sha256(code.encode("utf-8")).hexdigest())
    return ordered[:total]


def _t0_rows() -> list[dict[str, str]]:
    codes = _ordered_synthetic_codes(SYNTHETIC_BLOCK_SIZE)
    return [
        {"code": code, "market": "プライム（内国株式）", "industry": "SYN_INDUSTRY"}
        for code in codes
    ]


def _fresh_rows(count: int) -> list[dict[str, str]]:
    codes = _ordered_synthetic_codes(SYNTHETIC_BLOCK_SIZE + count)[SYNTHETIC_BLOCK_SIZE:]
    return [
        {"code": code, "market": "スタンダード（内国株式）", "industry": "SYN_INDUSTRY"}
        for code in codes
    ]


def build_workspace(workspace: Path) -> dict[str, Any]:
    import hashlib

    import pandas as pd

    from src.v8_partition import build_universe_csv_bytes, canonical_order

    t0_rows = _t0_rows()
    fresh_rows = _fresh_rows(SYNTHETIC_BLOCK_SIZE * 3 + 2)

    all_codes = [row["code"] for row in t0_rows + fresh_rows]
    ordered = canonical_order(all_codes)
    if ordered[:SYNTHETIC_BLOCK_SIZE] != [row["code"] for row in t0_rows]:
        raise AssertionError("SYNTHETIC_FIXTURE_ORDERING_INVALID")

    v4_universe_csv_bytes = build_universe_csv_bytes(t0_rows)
    ticker_list_sha256 = hashlib.sha256(
        ("\n".join(row["code"] for row in t0_rows) + "\n").encode("utf-8")
    ).hexdigest()
    universe_csv_sha256 = hashlib.sha256(v4_universe_csv_bytes).hexdigest()

    raw_source_bytes = b"SYNTHETIC_JPX_LISTING_RAW_BYTES_NOT_REAL_NETWORK_DATA"
    raw_file_sha256 = hashlib.sha256(raw_source_bytes).hexdigest()

    v4_manifest_path = workspace / "V4_UNIVERSE_MANIFEST.json"
    v4_manifest_path.write_bytes(json.dumps({
        "source_host": "www.jpx.co.jp",
        "source_page": "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html",
        "raw_file_sha256": raw_file_sha256,
        "universe_csv_sha256": universe_csv_sha256,
        "ticker_list_sha256": ticker_list_sha256,
        "selection_rule": "synthetic fixture mirrors the real V4 selection rule",
        "selected_count": SYNTHETIC_BLOCK_SIZE,
        "eligible_current_only": len(all_codes),
    }, ensure_ascii=False).encode("utf-8"))

    v4_universe_csv_path = workspace / "V4_UNIVERSE.csv"
    v4_universe_csv_path.write_bytes(v4_universe_csv_bytes)

    frame = pd.DataFrame([
        {"コード": row["code"], "銘柄名": "SYN", "市場・区分": row["market"], "33業種区分": row["industry"]}
        for row in t0_rows + fresh_rows
    ])

    return {
        "v4_manifest_path": v4_manifest_path,
        "v4_universe_csv_path": v4_universe_csv_path,
        "raw_source_bytes": raw_source_bytes,
        "frame": frame,
    }


def run_synthetic_partition_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v8-partition-") as temporary:
        workspace = Path(temporary)
        fixture = build_workspace(workspace)

        manifest = build_partition_manifest(
            raw_source_bytes=fixture["raw_source_bytes"],
            parse_source_table=lambda _raw: fixture["frame"],
            v4_manifest_path=fixture["v4_manifest_path"],
            v4_universe_csv_path=fixture["v4_universe_csv_path"],
            source_url=SYNTHETIC_SOURCE_URL,
            source_acquisition_utc=datetime(2026, 8, 9, tzinfo=timezone.utc),
            clock=lambda: datetime(2026, 8, 9, 1, 0, 0, tzinfo=timezone.utc),
            block_size=SYNTHETIC_BLOCK_SIZE,
        )
        if manifest["source_reproduction_status"] != "PASS":
            raise AssertionError("SOURCE_REPRODUCTION_NOT_PASS")

        output_path = workspace / "private-output" / "partition_manifest.json"
        write_partition_manifest_once(output_path, manifest, repository_root=ROOT)
        reread = read_partition_manifest(output_path)
        if reread != manifest:
            raise AssertionError("MANIFEST_ROUNDTRIP_MISMATCH")

        # A second write to the same path must BLOCK (write-once).
        try:
            write_partition_manifest_once(output_path, manifest, repository_root=ROOT)
            raise AssertionError("OVERWRITE_NOT_BLOCKED")
        except V8PartitionBlocked as error:
            if error.reason != "PARTITION_MANIFEST_ALREADY_EXISTS":
                raise

        # A source-hash mismatch must BLOCK before any block assignment.
        try:
            build_partition_manifest(
                raw_source_bytes=b"WRONG_BYTES_MUST_NOT_REPRODUCE",
                parse_source_table=lambda _raw: fixture["frame"],
                v4_manifest_path=fixture["v4_manifest_path"],
                v4_universe_csv_path=fixture["v4_universe_csv_path"],
                source_url=SYNTHETIC_SOURCE_URL,
                source_acquisition_utc=datetime(2026, 8, 9, tzinfo=timezone.utc),
                clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
                block_size=SYNTHETIC_BLOCK_SIZE,
            )
            raise AssertionError("SOURCE_MISMATCH_NOT_BLOCKED")
        except V8PartitionBlocked as error:
            if error.reason != "V8_PARTITION_SOURCE_NOT_REPRODUCIBLE":
                raise

    return {
        "status": "PASS",
        "mode": "STATIC_SYNTHETIC_ONLY",
        "source_reproduction_status": manifest["source_reproduction_status"],
        "block_sizes": manifest["block_sizes"],
        "t1_role": manifest["t1_role"],
        "t2_role": manifest["t2_role"],
        "t3_role": manifest["t3_role"],
        "t3_price_acquisition_authorized": manifest["t3_price_acquisition_authorized"],
        "manifest_sha256_verified": True,
        "write_once_enforced": True,
        "source_mismatch_blocks_before_allocation": True,
        "network_requests": 0,
        "real_partition_created": False,
        "real_source_fetch_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V8 partition manifest synthetic-only check")
    parser.add_argument("--synthetic-test", action="store_true", required=True)
    parser.parse_args(argv)
    result = run_synthetic_partition_test()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
