"""V8 fresh cross-sectional ticker-block partition manifest builder.

This module never imports, reads, writes, or otherwise touches any V7
module, V7 activation manifest, or V7 durable study root. It is entirely
new V8-only code.

Its job is narrow and fail-closed: reconstruct the official JPX
eligible-universe listing using the exact selection semantics already
recorded as provenance in ``V4_UNIVERSE_MANIFEST.json`` (mirroring
``free_prototype.parse_current_jpx_universe`` / ``free_prototype.select_codes``
byte-for-byte, re-derived here rather than imported so that this raw,
ML-free partition builder does not have to pull in ``lightgbm``,
``scikit-learn``, ``scipy`` or ``requests``), prove that reconstruction
reproduces the existing frozen ``T0`` (``V4_UNIVERSE.csv``) exactly, and only
then allocate the frozen-size fresh ticker blocks ``T1``/``T2``/``T3``/
``T_spare`` required by ``V8_HISTORICAL_RESEARCH_DESIGN.md`` Decision 2.

If reproduction of the official source or of ``T0`` cannot be proven, this
module BLOCKs before any block assignment is written -- a partition that
cannot be re-derived is not a partition, it is an unverifiable claim about
which tickers were sealed.

Importing this module performs no I/O. The raw JPX source bytes and the
raw-bytes-to-table parser are both caller-supplied, so the whole pipeline is
exercisable with zero real network access; production wiring (the real
fetch and the real Excel parser) lives in ``scripts/build_v8_partition_manifest.py``
and is never invoked by any test in this phase.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

STUDY_NAME = "V8_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8_PARTITION_MANIFEST_V2"
DESIGN_COMMIT = "c414d3191cba356734d7ed08bdf1abc7d51fc384"
PRODUCTION_BRANCH = "v8-partition-acquisition"

BLOCK_SIZE = 300
P_HIST_START = "2016-04-01"
P_HIST_END = "2025-12-31"
T1_ROLE = "VALIDATION"
T2_ROLE = "SEALED_HOLDOUT"
T3_ROLE = "SEALED_RESERVE"
T3_PRICE_ACQUISITION_AUTHORIZED = False

DETERMINISTIC_ORDERING_RULE = (
    "sort eligible_current_only by (SHA-256(UTF-8 code), code) ascending"
)

# The seven LEGACY_8 codes (audit V8_DATA_EXPOSURE_AUDIT.md Sec 2.1) that are
# NOT inside T0 (`4188` is the only legacy code inside T0 and is excluded
# automatically by T0 membership itself). These seven are outcome-exposed via
# the committed reference replay artifacts and must never enter a fresh block.
LEGACY_EXPOSED_TICKERS_OUTSIDE_T0: tuple[str, ...] = (
    "1570", "4689", "5020", "7211", "7267", "8306", "9432",
)

UNIVERSE_CSV_COLUMNS = ("ticker", "market", "industry")

REQUIRED_V4_PROVENANCE_FIELDS = (
    "source_host",
    "source_page",
    "raw_file_sha256",
    "universe_csv_sha256",
    "ticker_list_sha256",
    "selection_rule",
    "selected_count",
    "eligible_current_only",
)

MANIFEST_FIELDS = (
    "schema_version",
    "study_name",
    "design_commit",
    "partition_implementation_git_commit",
    "created_utc",
    "source_url",
    "source_host",
    "source_acquisition_utc",
    "source_raw_sha256",
    "source_raw_byte_count",
    "expected_v4_source_raw_sha256",
    "source_reproduction_status",
    "eligible_ticker_count",
    "eligible_ticker_list_sha256",
    "deterministic_ordering_rule",
    "t0_ticker_list_sha256",
    "t1_ticker_list_sha256",
    "t2_ticker_list_sha256",
    "t3_ticker_list_sha256",
    "t_spare_ticker_list_sha256",
    "legacy_exclude_list",
    "legacy_exclude_list_sha256",
    "block_sizes",
    "block_assignments",
    "p_hist_start",
    "p_hist_end",
    "t1_role",
    "t2_role",
    "t3_role",
    "t3_price_acquisition_authorized",
    "manifest_sha256",
)


class V8PartitionBlocked(RuntimeError):
    """Fail-closed partition-manifest construction error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except ValueError as error:
        raise V8PartitionBlocked("NONFINITE_VALUE") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _utc_timestamp(value: Any, field: str) -> datetime:
    from datetime import timedelta

    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8PartitionBlocked("UTC_TIMESTAMP_INVALID:" + field)
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _ticker_list_sha(tickers: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(tickers) + "\n").encode("utf-8"))


def ticker_list_sha256(tickers: Sequence[str]) -> str:
    """Return the frozen V8 canonical ticker-list SHA-256.

    This public wrapper is the single authoritative reusable implementation
    for consumers that need to verify a persisted partition assignment.
    """
    return _ticker_list_sha(tickers)


def require_git_commit(value: object, reason: str = "IMPLEMENTATION_GIT_COMMIT_INVALID") -> str:
    """Require a full lowercase Git object ID suitable for provenance."""
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8PartitionBlocked(reason)
    return value


def resolve_verified_production_git_commit(
    repository_root: str | os.PathLike[str],
) -> str:
    """Resolve a clean checkout exactly matching the local origin branch ref.

    This deliberately performs no fetch.  Production operators must fetch
    separately; this guard only proves the local checkout is exactly the
    already-fetched GitHub-tracking state before any production network I/O.
    """
    root = Path(repository_root)
    commands = (
        ("status", ["git", "-C", str(root), "status", "--porcelain"]),
        ("head", ["git", "-C", str(root), "rev-parse", "HEAD"]),
        ("origin", ["git", "-C", str(root), "rev-parse", "origin/" + PRODUCTION_BRANCH]),
    )
    results: dict[str, subprocess.CompletedProcess[str]] = {}
    try:
        for name, command in commands:
            results[name] = subprocess.run(
                command, capture_output=True, check=False, text=True, timeout=10
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8PartitionBlocked("PRODUCTION_GIT_PROVENANCE_UNAVAILABLE") from error
    if results["status"].returncode != 0:
        raise V8PartitionBlocked("PRODUCTION_GIT_PROVENANCE_UNAVAILABLE")
    if results["status"].stdout.strip():
        raise V8PartitionBlocked("PRODUCTION_GIT_WORKTREE_DIRTY")
    if results["head"].returncode != 0:
        raise V8PartitionBlocked("PRODUCTION_GIT_HEAD_UNAVAILABLE")
    if results["origin"].returncode != 0:
        raise V8PartitionBlocked("PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE")
    head = require_git_commit(results["head"].stdout.strip())
    origin = require_git_commit(results["origin"].stdout.strip())
    if head != origin:
        raise V8PartitionBlocked("PRODUCTION_GIT_HEAD_NOT_ORIGIN")
    return head


# ---------------------------------------------------------------------------
# Official V4 provenance (read, never re-hardcoded)
# ---------------------------------------------------------------------------


def load_v4_provenance(v4_manifest_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the official source provenance V4 already recorded and verified.

    This deliberately reads from ``V4_UNIVERSE_MANIFEST.json`` at call time
    rather than re-declaring the same hashes as separate constants in this
    module, so there is exactly one place those values can drift from.
    """
    path = Path(v4_manifest_path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8PartitionBlocked("V4_MANIFEST_READ_FAILED") from error
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8PartitionBlocked("V4_MANIFEST_INVALID_JSON") from error
    if not isinstance(data, Mapping):
        raise V8PartitionBlocked("V4_MANIFEST_INVALID_JSON")
    missing = set(REQUIRED_V4_PROVENANCE_FIELDS) - set(data)
    if missing:
        raise V8PartitionBlocked("V4_MANIFEST_SCHEMA_INVALID")
    return dict(data)


def load_v4_universe_csv_bytes(v4_universe_csv_path: str | os.PathLike[str]) -> bytes:
    path = Path(v4_universe_csv_path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8PartitionBlocked("V4_UNIVERSE_CSV_READ_FAILED") from error
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


# ---------------------------------------------------------------------------
# Eligible-universe reconstruction (mirrors free_prototype.py exactly)
# ---------------------------------------------------------------------------


def _find_column(columns: list[str], needles: tuple[str, ...]) -> str:
    for column in columns:
        compact = re.sub(r"\s+", "", str(column))
        if all(needle in compact for needle in needles):
            return column
    raise V8PartitionBlocked("JPX_SOURCE_COLUMN_MISSING")


def parse_eligible_universe(frame: Any) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Reproduce ``free_prototype.parse_current_jpx_universe`` exactly.

    ``frame`` is any pandas-DataFrame-like object with the raw JPX listing
    columns (code, name, market, and optionally the 33-sector column). This
    function does not fetch or parse raw bytes itself -- that step is
    injected by the caller (production: ``pandas.read_excel``; tests: a
    small in-memory frame), keeping this function's selection logic testable
    without any Excel-parsing dependency.

    Returns eligible rows sorted by code (mergesort, matching the source
    algorithm) as plain dicts with keys ``code``/``name``/``market``/
    ``industry``, plus the same exclusion-reason counters the source
    algorithm reports.
    """
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise V8PartitionBlocked("JPX_SOURCE_FRAME_INVALID")
    columns = [str(c) for c in frame.columns]
    code_col = _find_column(columns, ("コード",))
    name_col = _find_column(columns, ("銘柄名",))
    market_col = _find_column(columns, ("市場", "区分"))
    sector_col = next((c for c in columns if "33業種区分" in re.sub(r"\s+", "", c)), None)
    work = frame.rename(columns={code_col: "code", name_col: "name", market_col: "market"}).copy()
    if sector_col:
        work = work.rename(columns={sector_col: "industry"})
    else:
        work["industry"] = "MISSING"
    work["code"] = work["code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.upper()
    work["market"] = work["market"].astype(str).str.strip()
    prime_standard = work["market"].str.contains("プライム|Prime|スタンダード|Standard", case=False, regex=True)
    domestic = work["market"].str.contains("内国株式|Domestic Stocks", case=False, regex=True)
    ordinary_code = work["code"].str.fullmatch(r"[0-9A-Z]{4}")
    reasons = {
        "input_rows": int(len(work)),
        "excluded_non_prime_standard": int((~prime_standard).sum()),
        "excluded_non_domestic_stock": int((prime_standard & ~domestic).sum()),
        "excluded_non_four_character_code": int((prime_standard & domestic & ~ordinary_code).sum()),
    }
    eligible = work.loc[prime_standard & domestic & ordinary_code, ["code", "name", "market", "industry"]].drop_duplicates("code")
    reasons["eligible_current_only"] = int(len(eligible))
    ordered = eligible.sort_values("code", kind="mergesort").reset_index(drop=True)
    rows = [
        {"code": str(row["code"]), "market": str(row["market"]), "industry": str(row["industry"])}
        for row in ordered.to_dict(orient="records")
    ]
    return rows, reasons


def canonical_order(tickers: Sequence[str]) -> list[str]:
    """Mirror ``free_prototype.select_codes`` ordering with no size limit."""
    normalized = sorted({str(code).strip().upper() for code in tickers if str(code).strip()})
    return sorted(normalized, key=lambda code: (hashlib.sha256(code.encode("utf-8")).hexdigest(), code))


def build_universe_csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    """Byte-identical writer to the one that produced ``V4_UNIVERSE.csv``:
    header ``ticker,market,industry``, LF line endings, rows kept in the
    order given (no re-sorting -- ordering is the caller's responsibility)."""
    import csv
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=UNIVERSE_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "ticker": row["code"],
            "market": row["market"],
            "industry": row["industry"],
        })
    return stream.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# T0 reproduction guard
# ---------------------------------------------------------------------------


def verify_t0_reproduction(
    eligible_rows_ordered: Sequence[Mapping[str, str]],
    v4_provenance: Mapping[str, Any],
    *,
    block_size: int = BLOCK_SIZE,
) -> list[str]:
    """Require that the first ``block_size`` reconstructed tickers reproduce
    V4_UNIVERSE.csv byte-for-byte, in the same order. BLOCKs with
    V8_T0_REPRODUCTION_MISMATCH on any divergence -- never silently accepts
    a "close enough" reconstruction.

    ``block_size`` defaults to the frozen production value (300) but is
    overridable so synthetic tests can exercise this guard against small
    fixtures without reproducing full production block-size semantics."""
    if len(eligible_rows_ordered) < block_size:
        raise V8PartitionBlocked("V8_T0_REPRODUCTION_MISMATCH")
    t0_rows = list(eligible_rows_ordered[:block_size])
    t0_tickers = [row["code"] for row in t0_rows]
    ticker_list_sha = _ticker_list_sha(t0_tickers)
    csv_bytes = build_universe_csv_bytes(t0_rows)
    csv_sha = sha256_bytes(csv_bytes)
    if ticker_list_sha != v4_provenance["ticker_list_sha256"]:
        raise V8PartitionBlocked("V8_T0_REPRODUCTION_MISMATCH")
    if csv_sha != v4_provenance["universe_csv_sha256"]:
        raise V8PartitionBlocked("V8_T0_REPRODUCTION_MISMATCH")
    return t0_tickers


def _source_preflight_core(
    *,
    raw_source_bytes: bytes,
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    source_url: str,
    source_acquisition_utc: datetime,
    partition_implementation_git_commit: str,
    block_size: int = BLOCK_SIZE,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Run only source/T0 reproduction and return private continuation data.

    The two ticker lists returned after the public result are intentionally
    private continuation data for the full manifest builder.  The public
    ``verify_partition_source_preflight`` wrapper discards them, so a
    source-only run cannot expose or construct fresh-block assignments.
    """
    implementation_git_commit = require_git_commit(partition_implementation_git_commit)
    if not isinstance(raw_source_bytes, (bytes, bytearray)):
        raise V8PartitionBlocked("RAW_SOURCE_BYTES_INVALID")
    raw_bytes = bytes(raw_source_bytes)
    v4_provenance = load_v4_provenance(v4_manifest_path)

    committed_csv_bytes = load_v4_universe_csv_bytes(v4_universe_csv_path)
    if sha256_bytes(committed_csv_bytes) != v4_provenance["universe_csv_sha256"]:
        raise V8PartitionBlocked("V4_UNIVERSE_CSV_PROVENANCE_MISMATCH")

    source_raw_sha256 = sha256_bytes(raw_bytes)
    expected = v4_provenance["raw_file_sha256"]
    if source_raw_sha256 != expected:
        raise V8PartitionBlocked("V8_PARTITION_SOURCE_NOT_REPRODUCIBLE")

    frame = parse_source_table(raw_bytes)
    eligible_rows, _reasons = parse_eligible_universe(frame)
    if not eligible_rows:
        raise V8PartitionBlocked("V8_ELIGIBLE_UNIVERSE_EMPTY")

    ordered_codes = canonical_order([row["code"] for row in eligible_rows])
    rows_by_code = {row["code"]: row for row in eligible_rows}
    if len(rows_by_code) != len(eligible_rows):
        raise V8PartitionBlocked("V8_ELIGIBLE_LIST_DUPLICATE_TICKER")
    eligible_rows_ordered = [rows_by_code[code] for code in ordered_codes]
    t0_tickers = verify_t0_reproduction(eligible_rows_ordered, v4_provenance, block_size=block_size)
    acquired = _utc_timestamp(source_acquisition_utc, "source_acquisition_utc")

    result = {
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "source_url": source_url,
        "source_host": v4_provenance["source_host"],
        "source_raw_sha256": source_raw_sha256,
        "expected_source_raw_sha256": expected,
        "source_raw_byte_count": len(raw_bytes),
        "source_acquisition_utc": _timestamp_text(acquired),
        "eligible_ticker_count": len(ordered_codes),
        "eligible_ticker_list_sha256": _ticker_list_sha(ordered_codes),
        "t0_ticker_list_sha256": _ticker_list_sha(t0_tickers),
        "partition_implementation_git_commit": implementation_git_commit,
    }
    return result, ordered_codes, t0_tickers


def verify_partition_source_preflight(
    *,
    raw_source_bytes: bytes,
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    source_url: str,
    source_acquisition_utc: datetime,
    partition_implementation_git_commit: str,
    block_size: int = BLOCK_SIZE,
) -> dict[str, Any]:
    """Verify official-source and T0 reproduction without partition allocation.

    This is the source-only human-gate primitive.  It validates raw bytes,
    V4 provenance, canonical eligible-universe reconstruction, ordering, and
    T0 reproduction, then returns audit metadata.  It never calls
    ``allocate_fresh_blocks`` and never constructs T1/T2/T3/T_spare data.
    """
    result, _ordered_codes, _t0_tickers = _source_preflight_core(
        raw_source_bytes=raw_source_bytes,
        parse_source_table=parse_source_table,
        v4_manifest_path=v4_manifest_path,
        v4_universe_csv_path=v4_universe_csv_path,
        source_url=source_url,
        source_acquisition_utc=source_acquisition_utc,
        partition_implementation_git_commit=partition_implementation_git_commit,
        block_size=block_size,
    )
    return result


# ---------------------------------------------------------------------------
# Fresh block allocation
# ---------------------------------------------------------------------------


def allocate_fresh_blocks(
    eligible_ordered_tickers: Sequence[str],
    t0_tickers: Sequence[str],
    *,
    block_size: int = BLOCK_SIZE,
) -> dict[str, list[str]]:
    """Allocate T1/T2/T3/T_spare from the deterministic order, excluding T0
    and the legacy-exposed codes outside T0. BLOCKs on any duplicate,
    insufficient pool size, or invariant violation before returning.

    ``block_size`` defaults to the frozen production value (300); see
    ``verify_t0_reproduction`` for why it is overridable."""
    if len(set(eligible_ordered_tickers)) != len(eligible_ordered_tickers):
        raise V8PartitionBlocked("V8_ELIGIBLE_LIST_DUPLICATE_TICKER")
    t0_set = set(t0_tickers)
    if len(t0_set) != len(t0_tickers) or len(t0_set) != block_size:
        raise V8PartitionBlocked("V8_T0_SIZE_INVALID")
    legacy_set = set(LEGACY_EXPOSED_TICKERS_OUTSIDE_T0)
    exclude = t0_set | legacy_set
    fresh_pool = [ticker for ticker in eligible_ordered_tickers if ticker not in exclude]
    if len(fresh_pool) != len(set(fresh_pool)):
        raise V8PartitionBlocked("V8_FRESH_POOL_DUPLICATE_TICKER")
    if len(fresh_pool) < block_size * 3:
        raise V8PartitionBlocked("V8_ELIGIBLE_POOL_INSUFFICIENT")

    t1 = fresh_pool[:block_size]
    t2 = fresh_pool[block_size:2 * block_size]
    t3 = fresh_pool[2 * block_size:3 * block_size]
    t_spare = fresh_pool[3 * block_size:]

    blocks = {"T0": list(t0_tickers), "T1": t1, "T2": t2, "T3": t3, "T_spare": t_spare}

    if len(t1) != block_size or len(t2) != block_size or len(t3) != block_size:
        raise V8PartitionBlocked("V8_BLOCK_SIZE_INVALID")

    all_assigned: list[str] = []
    for name in ("T0", "T1", "T2", "T3", "T_spare"):
        all_assigned.extend(blocks[name])
    if len(set(all_assigned)) != len(all_assigned):
        raise V8PartitionBlocked("V8_BLOCK_OVERLAP_DETECTED")

    for name in ("T1", "T2", "T3", "T_spare"):
        if legacy_set & set(blocks[name]):
            raise V8PartitionBlocked("V8_LEGACY_TICKER_IN_FRESH_BLOCK")
        if t0_set & set(blocks[name]):
            raise V8PartitionBlocked("V8_T0_TICKER_IN_FRESH_BLOCK")

    return blocks


# ---------------------------------------------------------------------------
# Output-root safety (reject destinations inside this repository)
# ---------------------------------------------------------------------------


def require_absolute_output_path_outside_repository(
    value: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
) -> Path:
    text = str(value)
    if not text.strip() or text != text.strip():
        raise V8PartitionBlocked("OUTPUT_PATH_INVALID")
    candidate = Path(text)
    if not candidate.is_absolute():
        raise V8PartitionBlocked("OUTPUT_PATH_NOT_ABSOLUTE")
    try:
        repository = Path(repository_root).resolve()
    except OSError as error:
        raise V8PartitionBlocked("REPOSITORY_ROOT_INVALID") from error
    try:
        resolved_parent = candidate.parent.resolve()
    except OSError:
        resolved_parent = candidate.parent
    resolved_candidate = resolved_parent / candidate.name
    if resolved_candidate == repository or resolved_candidate.is_relative_to(repository):
        raise V8PartitionBlocked("OUTPUT_PATH_INSIDE_SOURCE_REPOSITORY")
    return candidate


def preflight_partition_manifest_output(
    output_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
) -> Path:
    """Validate and prepare a production manifest destination before fetch.

    The write path repeats these checks immediately before publish.  This
    preflight exists specifically so invalid, in-repository, or already-used
    destinations cannot cause a JPX request.
    """
    destination = require_absolute_output_path_outside_repository(output_path, repository_root)
    if destination.exists():
        raise V8PartitionBlocked("PARTITION_MANIFEST_ALREADY_EXISTS")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8PartitionBlocked("OUTPUT_PATH_PARENT_INVALID") from error
    if not destination.parent.is_dir():
        raise V8PartitionBlocked("OUTPUT_PATH_PARENT_INVALID")
    return destination


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def build_partition_manifest(
    *,
    raw_source_bytes: bytes,
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    source_url: str,
    source_acquisition_utc: datetime,
    clock: Callable[[], datetime],
    partition_implementation_git_commit: str,
    block_size: int = BLOCK_SIZE,
) -> dict[str, Any]:
    """Build (but do not write) the complete partition manifest.

    Fails closed with ``V8_PARTITION_SOURCE_NOT_REPRODUCIBLE`` if the raw
    source bytes do not hash-match the ``raw_file_sha256`` already recorded
    in ``V4_UNIVERSE_MANIFEST.json``, and with ``V8_T0_REPRODUCTION_MISMATCH``
    if the reconstructed universe's first 300 tickers do not byte-reproduce
    ``V4_UNIVERSE.csv``. Neither failure writes anything.
    """
    source_result, ordered_codes, t0_tickers = _source_preflight_core(
        raw_source_bytes=raw_source_bytes,
        parse_source_table=parse_source_table,
        v4_manifest_path=v4_manifest_path,
        v4_universe_csv_path=v4_universe_csv_path,
        source_url=source_url,
        source_acquisition_utc=source_acquisition_utc,
        partition_implementation_git_commit=partition_implementation_git_commit,
        block_size=block_size,
    )
    implementation_git_commit = source_result["partition_implementation_git_commit"]
    source_raw_sha256 = source_result["source_raw_sha256"]
    expected = source_result["expected_source_raw_sha256"]
    v4_provenance = load_v4_provenance(v4_manifest_path)
    raw_bytes = bytes(raw_source_bytes)
    blocks = allocate_fresh_blocks(ordered_codes, t0_tickers, block_size=block_size)

    legacy_list = sorted(LEGACY_EXPOSED_TICKERS_OUTSIDE_T0)
    started = _utc_timestamp(clock(), "created_utc")
    acquired = _utc_timestamp(source_acquisition_utc, "source_acquisition_utc")

    block_sizes = {name: len(members) for name, members in blocks.items()}

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "design_commit": DESIGN_COMMIT,
        "partition_implementation_git_commit": implementation_git_commit,
        "created_utc": _timestamp_text(started),
        "source_url": source_url,
        "source_host": v4_provenance["source_host"],
        "source_acquisition_utc": _timestamp_text(acquired),
        "source_raw_sha256": source_raw_sha256,
        "source_raw_byte_count": len(raw_bytes),
        "expected_v4_source_raw_sha256": expected,
        "source_reproduction_status": "PASS",
        "eligible_ticker_count": len(ordered_codes),
        "eligible_ticker_list_sha256": _ticker_list_sha(ordered_codes),
        "deterministic_ordering_rule": DETERMINISTIC_ORDERING_RULE,
        "t0_ticker_list_sha256": _ticker_list_sha(blocks["T0"]),
        "t1_ticker_list_sha256": _ticker_list_sha(blocks["T1"]),
        "t2_ticker_list_sha256": _ticker_list_sha(blocks["T2"]),
        "t3_ticker_list_sha256": _ticker_list_sha(blocks["T3"]),
        "t_spare_ticker_list_sha256": _ticker_list_sha(blocks["T_spare"]),
        "legacy_exclude_list": legacy_list,
        "legacy_exclude_list_sha256": _ticker_list_sha(legacy_list),
        "block_sizes": block_sizes,
        "block_assignments": blocks,
        "p_hist_start": P_HIST_START,
        "p_hist_end": P_HIST_END,
        "t1_role": T1_ROLE,
        "t2_role": T2_ROLE,
        "t3_role": T3_ROLE,
        "t3_price_acquisition_authorized": T3_PRICE_ACQUISITION_AUTHORIZED,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    if set(manifest) != set(MANIFEST_FIELDS):
        raise V8PartitionBlocked("MANIFEST_SCHEMA_INVALID")
    return manifest


def write_partition_manifest_once(
    output_path: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    repository_root: str | os.PathLike[str],
) -> Path:
    """Atomically publish a complete manifest without replacing a destination.

    The staging file is fsynced first, then linked into its final name.
    ``os.link`` is atomic with respect to destination creation on supported
    filesystems: if another process wins the race, this operation blocks and
    never falls back to a replacement operation.
    """
    destination = require_absolute_output_path_outside_repository(output_path, repository_root)
    if destination.exists():
        raise V8PartitionBlocked("PARTITION_MANIFEST_ALREADY_EXISTS")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8PartitionBlocked("OUTPUT_PATH_PARENT_INVALID") from error
    if not destination.parent.is_dir():
        raise V8PartitionBlocked("OUTPUT_PATH_PARENT_INVALID")
    if not isinstance(manifest, Mapping) or set(manifest) != set(MANIFEST_FIELDS):
        raise V8PartitionBlocked("MANIFEST_SCHEMA_INVALID")
    stated = manifest.get("manifest_sha256")
    recomputed = canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if stated != recomputed:
        raise V8PartitionBlocked("MANIFEST_SHA_MISMATCH")
    payload = canonical_json_bytes(dict(manifest))
    staging = destination.parent / (destination.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(staging, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(staging), str(destination))
        except FileExistsError as error:
            raise V8PartitionBlocked("PARTITION_MANIFEST_ALREADY_EXISTS") from error
        except OSError as error:
            # Never replace an existing destination as a fallback.  A
            # filesystem without atomic no-overwrite publication support is
            # fail-closed rather than weakening write-once semantics.
            raise V8PartitionBlocked("PARTITION_MANIFEST_ATOMIC_PUBLISH_FAILED") from error
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass
    return destination


def read_partition_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read-only load with full self-hash re-verification."""
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
    except OSError as error:
        raise V8PartitionBlocked("PARTITION_MANIFEST_READ_FAILED") from error
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8PartitionBlocked("PARTITION_MANIFEST_DUPLICATE_KEY")
            result[key] = value
        return result
    try:
        manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8PartitionBlocked("PARTITION_MANIFEST_INVALID_JSON") from error
    if not isinstance(manifest, Mapping) or set(manifest) != set(MANIFEST_FIELDS):
        raise V8PartitionBlocked("MANIFEST_SCHEMA_INVALID")
    stated = manifest["manifest_sha256"]
    recomputed = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    if stated != recomputed:
        raise V8PartitionBlocked("MANIFEST_SHA_MISMATCH")
    require_git_commit(manifest["partition_implementation_git_commit"])
    return dict(manifest)


__all__ = [
    "BLOCK_SIZE",
    "DESIGN_COMMIT",
    "DETERMINISTIC_ORDERING_RULE",
    "LEGACY_EXPOSED_TICKERS_OUTSIDE_T0",
    "MANIFEST_FIELDS",
    "P_HIST_END",
    "P_HIST_START",
    "REQUIRED_V4_PROVENANCE_FIELDS",
    "SCHEMA_VERSION",
    "STUDY_NAME",
    "T1_ROLE",
    "T2_ROLE",
    "T3_PRICE_ACQUISITION_AUTHORIZED",
    "T3_ROLE",
    "UNIVERSE_CSV_COLUMNS",
    "V8PartitionBlocked",
    "allocate_fresh_blocks",
    "build_partition_manifest",
    "build_universe_csv_bytes",
    "canonical_json_bytes",
    "canonical_order",
    "canonical_sha256",
    "load_v4_provenance",
    "load_v4_universe_csv_bytes",
    "parse_eligible_universe",
    "read_partition_manifest",
    "require_absolute_output_path_outside_repository",
    "sha256_bytes",
    "ticker_list_sha256",
    "verify_partition_source_preflight",
    "verify_t0_reproduction",
    "write_partition_manifest_once",
]
