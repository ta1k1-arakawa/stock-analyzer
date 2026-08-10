"""V8 partition manifest builder CLI: synthetic check and production paths.

Two entirely separate code paths, selected by mutually exclusive flags:

* ``--synthetic-test`` -- unchanged from the original static-verification
  CLI. Builds a fully local synthetic JPX-listing fixture and a synthetic
  ``V4_UNIVERSE_MANIFEST.json``/``V4_UNIVERSE.csv`` pair inside a temporary
  workspace, drives the production partition builder end to end, and never
  touches the real JPX host, the real V4 universe files, or any real
  private V8 storage. ``network_requests`` is always 0 in this mode.

* ``--production-build-manifest`` -- fetches the official JPX listing at
  partition-implementation time (real network I/O when invoked with the
  default opener) and builds a real partition manifest. It reuses
  ``src.v8_partition.build_partition_manifest`` and
  ``write_partition_manifest_once`` completely unchanged. Per
  ``V8_HISTORICAL_RESEARCH_DESIGN.md`` §16 (``SOURCE_SNAPSHOT_SEMANTICS =
  "IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT"``), the fetched raw bytes are
  not required to hash-equal V4's 2026-08-03 ``raw_file_sha256`` -- the sole
  source-reproducibility fail-closed guard is exact ``T0`` reproduction
  against the already-frozen ``V4_UNIVERSE.csv``; a ``V8_T0_REPRODUCTION_
  MISMATCH`` BLOCKs before any block assignment is ever constructed. This
  mode requires an explicit ``--output-path`` (validated absolute, outside
  this repository, write-once) and an explicit ``--confirmation`` string,
  and persists nothing but the resulting manifest -- the raw JPX bytes
  themselves are never written anywhere, in this repository or otherwise.

* ``--production-source-preflight`` -- fetches the official JPX listing and
  verifies source/T0 reproduction only, under the same §16 semantics. It
  never allocates blocks, writes a partition manifest, or calls Yahoo. This
  is the separately authorized source-reproduction human gate.

No bypass flag of any kind exists for either mode: there is no
``--skip-t0-check``, ``--force``, or ``--ignore-parity``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v8_partition import (
    V8PartitionBlocked,
    build_partition_manifest,
    preflight_partition_manifest_output,
    read_partition_manifest,
    resolve_verified_production_git_commit,
    verify_partition_source_preflight,
    write_partition_manifest_once,
)

# Real official source. Only touched by --production-build-manifest. The
# public production runner always selects the strict canonical opener; fake
# openers exist only behind the private test seam below.
JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
JPX_SOURCE_HOST = "www.jpx.co.jp"
DATA_LINK_PATTERN = re.compile(r'href=["\']([^"\']*data_j\.xls)["\']', re.IGNORECASE)
PRODUCTION_USER_AGENT = "V8-Partition-Builder/1.0"
PRODUCTION_CONFIRMATION = "V8_PRODUCTION_PARTITION_BUILD"
PRODUCTION_SOURCE_PREFLIGHT_CONFIRMATION = "V8_PRODUCTION_JPX_SOURCE_PREFLIGHT"
V4_MANIFEST_PATH = ROOT / "V4_UNIVERSE_MANIFEST.json"
V4_UNIVERSE_CSV_PATH = ROOT / "V4_UNIVERSE.csv"

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
            partition_implementation_git_commit="a" * 40,
            block_size=SYNTHETIC_BLOCK_SIZE,
        )
        if manifest["source_reproduction_status"] != "PASS":
            raise AssertionError("SOURCE_REPRODUCTION_NOT_PASS")
        if manifest["v4_raw_sha_equality_required"] is not False:
            raise AssertionError("V4_RAW_SHA_EQUALITY_REQUIREMENT_UNEXPECTED")

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

        # V8_HISTORICAL_RESEARCH_DESIGN.md §16: a raw-hash mismatch against
        # V4's 2026-08-03 reference alone must NOT block -- only T0 mismatch
        # does. This proves the new semantics with a distinct (differing)
        # raw payload whose T0 still reproduces V4_UNIVERSE.csv exactly.
        differing_source_manifest = build_partition_manifest(
            raw_source_bytes=fixture["raw_source_bytes"] + b"_DIFFERENT_BUT_T0_STILL_REPRODUCES",
            parse_source_table=lambda _raw: fixture["frame"],
            v4_manifest_path=fixture["v4_manifest_path"],
            v4_universe_csv_path=fixture["v4_universe_csv_path"],
            source_url=SYNTHETIC_SOURCE_URL,
            source_acquisition_utc=datetime(2026, 8, 9, tzinfo=timezone.utc),
            clock=lambda: datetime(2026, 8, 9, 2, 0, 0, tzinfo=timezone.utc),
            partition_implementation_git_commit="a" * 40,
            block_size=SYNTHETIC_BLOCK_SIZE,
        )
        if differing_source_manifest["source_raw_sha256"] == manifest["v4_source_raw_sha256_reference"]:
            raise AssertionError("SYNTHETIC_FIXTURE_RAW_HASH_UNEXPECTEDLY_MATCHED")
        if differing_source_manifest["source_reproduction_status"] != "PASS":
            raise AssertionError("RAW_HASH_MISMATCH_INCORRECTLY_BLOCKED")

        # A T0 mismatch must still BLOCK before any block assignment -- the
        # sole remaining source-reproducibility guard.
        tampered_frame = fixture["frame"].copy()
        tampered_frame.loc[0, "市場・区分"] = "スタンダード（内国株式）"
        try:
            build_partition_manifest(
                raw_source_bytes=fixture["raw_source_bytes"],
                parse_source_table=lambda _raw: tampered_frame,
                v4_manifest_path=fixture["v4_manifest_path"],
                v4_universe_csv_path=fixture["v4_universe_csv_path"],
                source_url=SYNTHETIC_SOURCE_URL,
                source_acquisition_utc=datetime(2026, 8, 9, tzinfo=timezone.utc),
                clock=lambda: datetime(2026, 8, 9, tzinfo=timezone.utc),
                partition_implementation_git_commit="a" * 40,
                block_size=SYNTHETIC_BLOCK_SIZE,
            )
            raise AssertionError("T0_MISMATCH_NOT_BLOCKED")
        except V8PartitionBlocked as error:
            if error.reason != "V8_T0_REPRODUCTION_MISMATCH":
                raise

    return {
        "status": "PASS",
        "mode": "STATIC_SYNTHETIC_ONLY",
        "source_reproduction_status": manifest["source_reproduction_status"],
        "v4_raw_sha_equality_required": manifest["v4_raw_sha_equality_required"],
        "block_sizes": manifest["block_sizes"],
        "t1_role": manifest["t1_role"],
        "t2_role": manifest["t2_role"],
        "t3_role": manifest["t3_role"],
        "t3_price_acquisition_authorized": manifest["t3_price_acquisition_authorized"],
        "manifest_sha256_verified": True,
        "write_once_enforced": True,
        "raw_hash_mismatch_does_not_block": True,
        "t0_mismatch_blocks_before_allocation": True,
        "network_requests": 0,
        "real_partition_created": False,
        "real_source_fetch_performed": False,
    }


# ---------------------------------------------------------------------------
# Production path -- real JPX source, real partition manifest
# ---------------------------------------------------------------------------


def default_parse_source_table(raw_bytes: bytes) -> Any:
    """Real production parser: the raw bytes are the official JPX ``data_j.xls``
    listing. Tests must always inject a fake table-returning callable instead
    of exercising this function, since it depends on real spreadsheet bytes."""
    import io

    import pandas as pd

    return pd.read_excel(io.BytesIO(raw_bytes), dtype=str)


def _require_trusted_jpx_url(value: object) -> str:
    if not isinstance(value, str):
        raise V8PartitionBlocked("V8_PARTITION_SOURCE_FINAL_URL_INVALID")
    try:
        parsed = urllib.parse.urlparse(value)
        port = parsed.port
    except ValueError as error:
        raise V8PartitionBlocked("V8_PARTITION_SOURCE_HOST_INVALID") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != JPX_SOURCE_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise V8PartitionBlocked("V8_PARTITION_SOURCE_HOST_INVALID")
    return value


class TrustedJpxRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject a redirect before urllib issues an off-host request."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        _require_trusted_jpx_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_trusted_jpx_opener(request: Any) -> Any:
    """Default production opener with pre-request redirect host enforcement."""
    _require_trusted_jpx_url(getattr(request, "full_url", None))
    opener = urllib.request.build_opener(TrustedJpxRedirectHandler())
    return opener.open(request)


def _read_response(response: Any) -> bytes:
    response_url = getattr(response, "url", None)
    if response_url is None:
        geturl = getattr(response, "geturl", None)
        response_url = geturl() if callable(geturl) else None
    _require_trusted_jpx_url(response_url)
    try:
        payload = response.read()
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if not isinstance(payload, bytes):
        raise V8PartitionBlocked("V8_PARTITION_SOURCE_RESPONSE_INVALID")
    return payload


def _fetch_jpx_source_with_opener(*, opener: Callable[[Any], Any]) -> tuple[bytes, str]:
    """PRIVATE TEST SEAM ONLY -- fetch a JPX source through ``opener``.

    This dependency-injected helper is not a production public boundary.
    Production code must use ``fetch_real_jpx_source`` or
    ``run_production_partition_build`` instead.
    """
    page_request = urllib.request.Request(JPX_PAGE, headers={"User-Agent": PRODUCTION_USER_AGENT})
    try:
        page_response = opener(page_request)
    except urllib.error.URLError as error:
        raise V8PartitionBlocked("V8_PARTITION_SOURCE_PAGE_FETCH_FAILED:" + str(error.reason)) from error
    page_bytes = _read_response(page_response)

    match = DATA_LINK_PATTERN.search(page_bytes.decode("utf-8", errors="replace"))
    if not match:
        raise V8PartitionBlocked("V8_PARTITION_SOURCE_LINK_NOT_FOUND")
    source_url = urllib.parse.urljoin(JPX_PAGE, match.group(1))
    _require_trusted_jpx_url(source_url)

    xls_request = urllib.request.Request(source_url, headers={"User-Agent": PRODUCTION_USER_AGENT})
    try:
        xls_response = opener(xls_request)
    except urllib.error.URLError as error:
        raise V8PartitionBlocked("V8_PARTITION_SOURCE_XLS_FETCH_FAILED:" + str(error.reason)) from error
    raw_bytes = _read_response(xls_response)
    return raw_bytes, source_url


def fetch_real_jpx_source() -> tuple[bytes, str]:
    """Fetch the official JPX listing using the canonical strict opener."""
    return _fetch_jpx_source_with_opener(opener=_default_trusted_jpx_opener)


def _utc_clock() -> datetime:
    return datetime.now(timezone.utc)


def _run_production_source_preflight_with_dependencies(
    *,
    opener: Callable[[Any], Any],
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: Path,
    v4_universe_csv_path: Path,
    clock: Callable[[], datetime],
    repository_root: Path,
    git_commit_resolver: Callable[[Path], str],
) -> dict[str, Any]:
    """PRIVATE TEST SEAM ONLY -- source/T0 verification without publication.

    The Git check intentionally precedes the first JPX request. The raw
    source bytes exist only in memory and are passed to the source-only
    verifier, whose result contains no fresh-block assignments.
    """
    implementation_git_commit = git_commit_resolver(repository_root)
    raw_source_bytes, source_url = _fetch_jpx_source_with_opener(opener=opener)
    fetched_at = clock()
    return verify_partition_source_preflight(
        raw_source_bytes=raw_source_bytes,
        parse_source_table=parse_source_table,
        v4_manifest_path=v4_manifest_path,
        v4_universe_csv_path=v4_universe_csv_path,
        source_url=source_url,
        source_acquisition_utc=fetched_at,
        partition_implementation_git_commit=implementation_git_commit,
    )


def run_production_source_preflight() -> dict[str, Any]:
    """Run the canonical real-source/T0 preflight with no caller overrides.

    This public boundary intentionally accepts no arguments. Operators must
    perform ``git fetch origin`` immediately before invoking the CLI; the
    resolver then proves clean ``HEAD == origin/v8-partition-acquisition``
    before the strict JPX transport is allowed to issue its first request.
    """
    return _run_production_source_preflight_with_dependencies(
        opener=_default_trusted_jpx_opener,
        parse_source_table=default_parse_source_table,
        v4_manifest_path=V4_MANIFEST_PATH,
        v4_universe_csv_path=V4_UNIVERSE_CSV_PATH,
        clock=_utc_clock,
        repository_root=ROOT,
        git_commit_resolver=resolve_verified_production_git_commit,
    )


def _run_production_partition_build_with_dependencies(
    *,
    output_path: Path,
    opener: Callable[[Any], Any],
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: Path,
    v4_universe_csv_path: Path,
    clock: Callable[[], datetime],
    repository_root: Path,
    git_commit_resolver: Callable[[Path], str],
) -> dict[str, Any]:
    """PRIVATE TEST SEAM ONLY -- dependency-injected partition build.

    NOT PRODUCTION PUBLIC BOUNDARY. Tests may inject fixtures here without
    creating a production override path. The public runner below supplies
    only canonical production dependencies.
    """
    # These guards deliberately precede the first JPX request.
    destination = preflight_partition_manifest_output(output_path, repository_root)
    implementation_git_commit = git_commit_resolver(repository_root)

    raw_source_bytes, source_url = _fetch_jpx_source_with_opener(opener=opener)
    fetched_at = clock()

    manifest = build_partition_manifest(
        raw_source_bytes=raw_source_bytes,
        parse_source_table=parse_source_table,
        v4_manifest_path=v4_manifest_path,
        v4_universe_csv_path=v4_universe_csv_path,
        source_url=source_url,
        source_acquisition_utc=fetched_at,
        clock=clock,
        partition_implementation_git_commit=implementation_git_commit,
    )
    written_path = write_partition_manifest_once(destination, manifest, repository_root=repository_root)
    return {"manifest": manifest, "written_path": written_path}


def run_production_partition_build(*, output_path: Path) -> dict[str, Any]:
    """Build a production partition manifest with canonical dependencies only.

    Callers may supply only the destination path. JPX transport, parsing, V4
    provenance, UTC clock, repository root, and Git provenance resolution are
    intentionally fixed inside this formal production boundary.
    """
    return _run_production_partition_build_with_dependencies(
        output_path=output_path,
        opener=_default_trusted_jpx_opener,
        parse_source_table=default_parse_source_table,
        v4_manifest_path=V4_MANIFEST_PATH,
        v4_universe_csv_path=V4_UNIVERSE_CSV_PATH,
        clock=_utc_clock,
        repository_root=ROOT,
        git_commit_resolver=resolve_verified_production_git_commit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V8 partition manifest builder")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-test", action="store_true")
    mode.add_argument("--production-build-manifest", action="store_true")
    mode.add_argument("--production-source-preflight", action="store_true")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--confirmation", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.synthetic_test:
        result = run_synthetic_partition_test()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    if args.production_source_preflight:
        if args.output_path:
            parser.error("--production-source-preflight does not accept --output-path")
        if not args.confirmation:
            parser.error("--production-source-preflight requires --confirmation")
        if args.confirmation != PRODUCTION_SOURCE_PREFLIGHT_CONFIRMATION:
            print(json.dumps({"status": "BLOCKED", "reason": "CONFIRMATION_MISMATCH"}, sort_keys=True))
            return 2
        try:
            result = run_production_source_preflight()
        except V8PartitionBlocked as error:
            print(json.dumps({"status": "BLOCKED", "reason": error.reason}, sort_keys=True))
            return 2
        summary = {
            "status": "PASS",
            "mode": "PRODUCTION_SOURCE_PREFLIGHT",
            **result,
            "real_partition_created": False,
            "real_block_assignments_created": False,
            "partition_manifest_written": False,
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0

    # --production-build-manifest
    if not args.output_path or not args.confirmation:
        parser.error("--production-build-manifest requires --output-path and --confirmation")
    if args.confirmation != PRODUCTION_CONFIRMATION:
        print(json.dumps({"status": "BLOCKED", "reason": "CONFIRMATION_MISMATCH"}, sort_keys=True))
        return 2

    try:
        result = run_production_partition_build(output_path=Path(args.output_path))
    except V8PartitionBlocked as error:
        print(json.dumps({"status": "BLOCKED", "reason": error.reason}, sort_keys=True))
        return 2

    manifest = result["manifest"]
    summary = {
        "status": "PASS",
        "mode": "PRODUCTION",
        "written_path": str(result["written_path"]),
        "source_reproduction_status": manifest["source_reproduction_status"],
        "block_sizes": manifest["block_sizes"],
        "manifest_sha256": manifest["manifest_sha256"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
