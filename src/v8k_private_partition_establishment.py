"""V8K Stage-2 private-partition-establishment support
(`V8K_LAYER_B_T1_PARTITION_AND_POINT_OF_USE_AUTHORITY_DESIGN_DRAFT.md` §3).

Implements the frozen Stage-2 `PRIVATE_PARTITION_ESTABLISHMENT` stage: the
`HUMAN_V8K_PRIVATE_PARTITION_GENERATION_GATE` fixed one-shot gate (frozen
authorization grammar, fixed receipt-key material, durable exclusive/
no-overwrite receipt publication strictly before seed creation), inherited
`src/v8_partition.py` source/T0/list-hash reuse (never its old deterministic
positional allocation, which this stage's HMAC-keyed allocation deliberately
replaces), the fresh-eligible-pool ( >= 900 ) fail-closed check, exactly-once
32-byte OS-CSPRNG seed generation and exclusive/no-overwrite private
persistence, HMAC-SHA256-keyed T1/T2/T3/T_spare allocation, and the private
`V8K_PARTITION_MANIFEST_V1` manifest / safe public evidence producers.

This is a fresh V8K-only implementation. It reuses `src/v8_partition.py`
only for reusable source/T0/list-hash semantics (`parse_eligible_universe`,
`canonical_order`, `verify_t0_reproduction`, `ticker_list_sha256`,
`canonical_json_bytes`/`canonical_sha256`, `LEGACY_EXPOSED_TICKERS_OUTSIDE_T0`)
and `src/v8k_public_source_preparation.py` only for the identical frozen
design-commit/blob binding and repository-provenance check both stages of
this design share (`production_provenance`) and for the fixed pointer to
Stage-1's own locked-raw-bytes location (`CANONICAL_V8K_PUBLIC_SOURCE_STATE_
ROOT`, `receipt_key`). It never reuses `allocate_fresh_blocks` -- that
function's plain canonical-order slicing is not this stage's frozen HMAC-
keyed allocation contract.

Importing this module performs no I/O, no network access, and no gate
consumption. The production entry point (`establish_private_partition`)
reads Stage-1's real machine-local locked raw bytes and the real canonical
private-partition state roots; this implementation task grants zero real
execution authority and is never invoked against real state by any test in
this file. All test coverage goes through the private `_establish_for_test`
seam with fully injected, temporary state roots and injected source bytes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.v8_partition import (
    LEGACY_EXPOSED_TICKERS_OUTSIDE_T0,
    V8PartitionBlocked,
    canonical_json_bytes,
    canonical_order,
    canonical_sha256,
    load_v4_provenance,
    load_v4_universe_csv_bytes,
    parse_eligible_universe,
    sha256_bytes,
    ticker_list_sha256,
    verify_t0_reproduction,
)
from src.v8c_human_gate_consumption import CANONICAL_CONSUMPTION_STATE_ROOT
from src.v8k_public_source_preparation import (
    CANONICAL_V8K_PUBLIC_SOURCE_STATE_ROOT,
    FROZEN_DESIGN_BLOB,
    FROZEN_DESIGN_COMMIT,
    STUDY,
    production_provenance,
    receipt_key as _stage1_receipt_key,
)

CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

GATE = "HUMAN_V8K_PRIVATE_PARTITION_GENERATION_GATE"
BLOCK_SIZE = 300
MINIMUM_FRESH_POOL = 900
SEED_BYTES = 32

AUTH_PREFIX = "V8K_HUMAN_AUTHORIZE_PRIVATE_PARTITION_GENERATION_AT_"
AUTH_WITH = "_WITH_"
AUTH_SOURCE = "_SOURCE_"

KEY_MATERIAL = (
    "V8K_PRIVATE_PARTITION_GENERATION_GATE_RECEIPT_KEY_V1\0"
    "ta1k1-arakawa/stock-analyzer\0" + STUDY + "\0" + GATE
).encode()

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Stage-1 (V8K_PUBLIC_SOURCE_PREPARATION) frozen factual binding. This is
# never trusted merely because it is declared here -- every real execution
# re-derives the eligible universe/T0/list-hash from the exact locked
# Stage-1 raw bytes and requires exact equality against these values before
# any authorization/gate/seed step proceeds.
STAGE1_SOURCE_RAW_SHA256 = "6e401867d9ddf2524e4752f08fd3e3e434cd308c6d423839ca6e24fc7b1e1653"
STAGE1_ELIGIBLE_TICKER_COUNT = 3110
STAGE1_ELIGIBLE_TICKER_LIST_SHA256 = "37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405"
STAGE1_T0_REPRODUCTION_STATUS = "PASS"
STAGE1_SUPPORT_SHA = "7fa38a6f74d631f7e1de37fae16fde944e18c580"

CANONICAL_V8K_PRIVATE_PARTITION_GATE_STATE_ROOT = (
    CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8k-private-partition-gate-state"
)
CANONICAL_V8K_PRIVATE_PARTITION_PRIVATE_STATE_ROOT = (
    CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8k-private-partition-private-state"
)

RECEIPT_SCHEMA_VERSION = "V8K_PRIVATE_PARTITION_GENERATION_GATE_RECEIPT_V1"
CONSUMPTION_BOUNDARY = "IMMEDIATELY_BEFORE_AUTHORITATIVE_SEED_CREATION"
RECEIPT_FIELDS = (
    "schema_version",
    "study",
    "gate",
    "receipt_key_sha256",
    "authorization_identity_sha256",
    "consumed",
    "consumption_count",
    "consumption_boundary",
    "consumption_timestamp_utc",
)

MANIFEST_SCHEMA_VERSION = "V8K_PARTITION_MANIFEST_V1"
MANIFEST_FIELDS = (
    "schema_version",
    "study",
    "gate",
    "frozen_design_commit",
    "frozen_design_blob",
    "reviewed_support_implementation_sha",
    "authorization_identity_sha256",
    "receipt_key_sha256",
    "source_raw_sha256",
    "eligible_ticker_count",
    "eligible_ticker_list_sha256",
    "seed_sha256",
    "fresh_pool_count",
    "t0_ticker_list_sha256",
    "t1_ticker_list_sha256",
    "t2_ticker_list_sha256",
    "t3_ticker_list_sha256",
    "t_spare_ticker_list_sha256",
    "block_sizes",
    "block_assignments",
    "created_utc",
    "manifest_sha256",
)

EVIDENCE_SCHEMA_VERSION = "V8K_PRIVATE_PARTITION_ESTABLISHMENT_EVIDENCE_V1"
EVIDENCE_ARTIFACT_ROLE = "PRIVATE_PARTITION_ESTABLISHMENT_EVIDENCE"
EVIDENCE_FIELDS = (
    "schema_version",
    "artifact_role",
    "study",
    "stage",
    "gate",
    "frozen_design_commit",
    "frozen_design_blob",
    "reviewed_support_implementation_sha",
    "authorization_identity_sha256",
    "receipt_key_sha256",
    "source_raw_sha256",
    "eligible_ticker_count",
    "eligible_ticker_list_sha256",
    "seed_sha256",
    "private_manifest_sha256",
    "t0_ticker_list_sha256",
    "t1_ticker_list_sha256",
    "t2_ticker_list_sha256",
    "t3_ticker_list_sha256",
    "t_spare_ticker_list_sha256",
    "t0_count",
    "t1_count",
    "t2_count",
    "t3_count",
    "t_spare_count",
    "receipt_consumed",
    "consumption_count",
    "result_classification",
)

DATA_QUALITY_FAILURE = "DATA_QUALITY_FAILURE"
GOVERNANCE_FAILURE = "GOVERNANCE_FAILURE"
IMPLEMENTATION_FAILURE = "IMPLEMENTATION_FAILURE"
PUBLIC_FAILURE_CLASSES = frozenset({DATA_QUALITY_FAILURE, GOVERNANCE_FAILURE, IMPLEMENTATION_FAILURE})

# Explicit fail-closed mapping from internal reasons to the frozen public
# failure classes (`V8K_LAYER_B_T1_PARTITION_AND_POINT_OF_USE_AUTHORITY_
# DESIGN_DRAFT.md` §5). Any reason not listed here is IMPLEMENTATION_FAILURE.
_INTERNAL_REASON_TO_PUBLIC_FAILURE_CLASS: dict[str, str] = {
    "DATA_QUALITY_FAILURE": DATA_QUALITY_FAILURE,
    "STAGE1_LOCKED_RAW_MISSING": DATA_QUALITY_FAILURE,
    "STAGE1_SOURCE_RAW_SHA256_MISMATCH": DATA_QUALITY_FAILURE,
    "STAGE1_ELIGIBLE_TICKER_COUNT_MISMATCH": DATA_QUALITY_FAILURE,
    "STAGE1_ELIGIBLE_TICKER_LIST_SHA256_MISMATCH": DATA_QUALITY_FAILURE,
    "STAGE1_T0_REPRODUCTION_STATUS_MISMATCH": DATA_QUALITY_FAILURE,
    "V8_T0_REPRODUCTION_MISMATCH": DATA_QUALITY_FAILURE,
    "V8_ELIGIBLE_UNIVERSE_EMPTY": DATA_QUALITY_FAILURE,
    "ELIGIBLE_UNIVERSE_EMPTY": DATA_QUALITY_FAILURE,
    "FRESH_POOL_INSUFFICIENT": DATA_QUALITY_FAILURE,
    "GOVERNANCE_FAILURE": GOVERNANCE_FAILURE,
    "AUTHORIZATION_GRAMMAR_INVALID": GOVERNANCE_FAILURE,
    "FROZEN_DESIGN_COMMIT_MISMATCH": GOVERNANCE_FAILURE,
    "GATE_ALREADY_CONSUMED": GOVERNANCE_FAILURE,
    "RECEIPT_AUTHORIZATION_MISMATCH": GOVERNANCE_FAILURE,
    "IMPLEMENTATION_FAILURE": IMPLEMENTATION_FAILURE,
}


def public_failure_class(reason: str) -> str:
    """Fail-closed mapping: any unrecognized internal reason is IMPLEMENTATION_FAILURE."""
    return _INTERNAL_REASON_TO_PUBLIC_FAILURE_CLASS.get(reason, IMPLEMENTATION_FAILURE)


class V8KPrivatePartitionBlocked(RuntimeError):
    """Internal reason stays in .reason/str(exc); only .failure_class is public-safe."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.failure_class = public_failure_class(reason)


def sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def receipt_key() -> str:
    return sha256(KEY_MATERIAL)


# ---------------------------------------------------------------------------
# §5 -- Human authorization grammar
# ---------------------------------------------------------------------------


def build_authorization_identity(*, design_commit: str, support_sha: str, source_raw_sha256: str) -> str:
    if not HEX40.fullmatch(design_commit) or not HEX40.fullmatch(support_sha) or not HEX64.fullmatch(source_raw_sha256):
        raise V8KPrivatePartitionBlocked("AUTHORIZATION_GRAMMAR_INVALID")
    return AUTH_PREFIX + design_commit + AUTH_WITH + support_sha + AUTH_SOURCE + source_raw_sha256


def validate_authorization(raw: str, *, design_commit: str, support_sha: str, source_raw_sha256: str) -> str:
    if design_commit != FROZEN_DESIGN_COMMIT:
        raise V8KPrivatePartitionBlocked("FROZEN_DESIGN_COMMIT_MISMATCH")
    expected = build_authorization_identity(
        design_commit=design_commit, support_sha=support_sha, source_raw_sha256=source_raw_sha256
    )
    if not isinstance(raw, str) or raw != expected:
        raise V8KPrivatePartitionBlocked("AUTHORIZATION_GRAMMAR_INVALID")
    return sha256(raw)


# ---------------------------------------------------------------------------
# Stage-1 source binding + fresh-pool computation (reuses src/v8_partition.py
# only for reusable source/T0/list-hash semantics; never its old allocation)
# ---------------------------------------------------------------------------


def verify_stage1_binding_and_compute_fresh_pool(
    *,
    raw_source_bytes: bytes,
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    minimum_fresh_pool: int = MINIMUM_FRESH_POOL,
    block_size: int = BLOCK_SIZE,
) -> tuple[list[str], list[str], list[str]]:
    """Independently re-derive the eligible universe/T0/fresh pool from the
    exact locked Stage-1 raw bytes and require exact equality against this
    reviewed implementation's frozen Stage-1 factual binding. Returns
    ``(ordered_codes, t0_tickers, fresh_pool)`` -- callers must never log,
    print, or publicly return any of these three lists."""
    if not isinstance(raw_source_bytes, (bytes, bytearray)):
        raise V8KPrivatePartitionBlocked("STAGE1_RAW_SOURCE_BYTES_INVALID")
    raw_bytes = bytes(raw_source_bytes)
    if sha256_bytes(raw_bytes) != STAGE1_SOURCE_RAW_SHA256:
        raise V8KPrivatePartitionBlocked("STAGE1_SOURCE_RAW_SHA256_MISMATCH")

    try:
        v4_provenance = load_v4_provenance(v4_manifest_path)
        committed_csv_bytes = load_v4_universe_csv_bytes(v4_universe_csv_path)
    except V8PartitionBlocked as error:
        raise V8KPrivatePartitionBlocked("V8_" + error.reason) from error
    if sha256_bytes(committed_csv_bytes) != v4_provenance["universe_csv_sha256"]:
        raise V8KPrivatePartitionBlocked("V4_UNIVERSE_CSV_PROVENANCE_MISMATCH")

    try:
        frame = parse_source_table(raw_bytes)
        eligible_rows, _reasons = parse_eligible_universe(frame)
    except V8PartitionBlocked as error:
        raise V8KPrivatePartitionBlocked("V8_" + error.reason) from error
    if not eligible_rows:
        raise V8KPrivatePartitionBlocked("ELIGIBLE_UNIVERSE_EMPTY")

    ordered_codes = canonical_order([row["code"] for row in eligible_rows])
    rows_by_code = {row["code"]: row for row in eligible_rows}
    if len(rows_by_code) != len(eligible_rows):
        raise V8KPrivatePartitionBlocked("ELIGIBLE_LIST_DUPLICATE_TICKER")
    eligible_rows_ordered = [rows_by_code[code] for code in ordered_codes]

    try:
        t0_tickers = verify_t0_reproduction(eligible_rows_ordered, v4_provenance, block_size=block_size)
    except V8PartitionBlocked as error:
        raise V8KPrivatePartitionBlocked("V8_" + error.reason) from error

    if len(ordered_codes) != STAGE1_ELIGIBLE_TICKER_COUNT:
        raise V8KPrivatePartitionBlocked("STAGE1_ELIGIBLE_TICKER_COUNT_MISMATCH")
    if ticker_list_sha256(ordered_codes) != STAGE1_ELIGIBLE_TICKER_LIST_SHA256:
        raise V8KPrivatePartitionBlocked("STAGE1_ELIGIBLE_TICKER_LIST_SHA256_MISMATCH")
    if STAGE1_T0_REPRODUCTION_STATUS != "PASS":
        raise V8KPrivatePartitionBlocked("STAGE1_T0_REPRODUCTION_STATUS_MISMATCH")

    legacy_set = set(LEGACY_EXPOSED_TICKERS_OUTSIDE_T0)
    t0_set = set(t0_tickers)
    if len(t0_set) != len(t0_tickers) or len(t0_set) != block_size:
        raise V8KPrivatePartitionBlocked("T0_SIZE_INVALID")
    exclude = t0_set | legacy_set
    fresh_pool = [code for code in ordered_codes if code not in exclude]
    if len(fresh_pool) != len(set(fresh_pool)):
        raise V8KPrivatePartitionBlocked("FRESH_POOL_DUPLICATE_TICKER")
    if len(fresh_pool) < minimum_fresh_pool:
        raise V8KPrivatePartitionBlocked("FRESH_POOL_INSUFFICIENT")
    return ordered_codes, t0_tickers, fresh_pool


# ---------------------------------------------------------------------------
# §3 -- V8K HMAC-keyed fresh-block allocation (never the old V8 allocation)
# ---------------------------------------------------------------------------


def _allocation_key(seed: bytes, ticker_code: str) -> bytes:
    return hmac.new(seed, ("V8K_PARTITION_ASSIGN_V1\0" + ticker_code).encode("utf-8"), hashlib.sha256).digest()


def allocate_v8k_blocks(fresh_pool: Sequence[str], seed: bytes, *, block_size: int = BLOCK_SIZE) -> dict[str, list[str]]:
    """Allocate T1/T2/T3/T_spare from ``fresh_pool`` using the frozen
    HMAC-SHA256(seed, "V8K_PARTITION_ASSIGN_V1\\0"+ticker) keyed sort. Fails
    closed on any duplicate ticker, allocation-key collision, or invariant
    violation before returning."""
    if not isinstance(seed, (bytes, bytearray)) or len(seed) != SEED_BYTES:
        raise V8KPrivatePartitionBlocked("SEED_BYTES_INVALID")
    seed_bytes = bytes(seed)
    if len(fresh_pool) != len(set(fresh_pool)):
        raise V8KPrivatePartitionBlocked("FRESH_POOL_DUPLICATE_TICKER")
    keyed = [(_allocation_key(seed_bytes, code), code) for code in fresh_pool]
    keys_only = [key for key, _ in keyed]
    if len(set(keys_only)) != len(keys_only):
        raise V8KPrivatePartitionBlocked("ALLOCATION_KEY_COLLISION")
    ordered = sorted(keyed, key=lambda pair: (pair[0], pair[1]))
    ordered_codes = [code for _, code in ordered]

    t1 = ordered_codes[:block_size]
    t2 = ordered_codes[block_size : 2 * block_size]
    t3 = ordered_codes[2 * block_size : 3 * block_size]
    t_spare = ordered_codes[3 * block_size :]
    if len(t1) != block_size or len(t2) != block_size or len(t3) != block_size:
        raise V8KPrivatePartitionBlocked("BLOCK_SIZE_INVALID")

    blocks = {"T1": t1, "T2": t2, "T3": t3, "T_spare": t_spare}
    all_assigned: list[str] = []
    for name in ("T1", "T2", "T3", "T_spare"):
        all_assigned.extend(blocks[name])
    if len(set(all_assigned)) != len(all_assigned):
        raise V8KPrivatePartitionBlocked("BLOCK_OVERLAP_DETECTED")
    return blocks


# ---------------------------------------------------------------------------
# Durable-publication primitive (write-once, exclusive/no-overwrite)
# ---------------------------------------------------------------------------


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_once(payload: bytes, output: Path, already_exists_reason: str, write_failed_reason: str) -> None:
    if output.exists():
        raise V8KPrivatePartitionBlocked(already_exists_reason)
    staging = output.parent / (output.name + ".staging-" + os.urandom(8).hex())
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(staging, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output)
        except FileExistsError as error:
            raise V8KPrivatePartitionBlocked(already_exists_reason) from error
        except OSError as error:
            raise V8KPrivatePartitionBlocked(write_failed_reason) from error
        _fsync_directory(output.parent)
    except V8KPrivatePartitionBlocked:
        raise
    except OSError as error:
        raise V8KPrivatePartitionBlocked(write_failed_reason) from error
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# §6 -- Fixed one-shot gate receipt (published strictly before seed creation)
# ---------------------------------------------------------------------------


def _gate_receipt_path(gate_state_root: str | os.PathLike[str]) -> Path:
    return Path(gate_state_root) / (receipt_key() + ".receipt.json")


def _validate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != set(RECEIPT_FIELDS):
        raise V8KPrivatePartitionBlocked("RECEIPT_SCHEMA_INVALID")
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION or receipt["study"] != STUDY or receipt["gate"] != GATE:
        raise V8KPrivatePartitionBlocked("RECEIPT_SCHEMA_INVALID")
    if receipt["receipt_key_sha256"] != receipt_key():
        raise V8KPrivatePartitionBlocked("RECEIPT_SCHEMA_INVALID")
    if type(receipt["consumed"]) is not bool or receipt["consumed"] is not True:
        raise V8KPrivatePartitionBlocked("RECEIPT_SCHEMA_INVALID")
    if type(receipt["consumption_count"]) is not int or receipt["consumption_count"] != 1:
        raise V8KPrivatePartitionBlocked("RECEIPT_SCHEMA_INVALID")
    if receipt["consumption_boundary"] != CONSUMPTION_BOUNDARY:
        raise V8KPrivatePartitionBlocked("RECEIPT_SCHEMA_INVALID")
    identity_hash = receipt["authorization_identity_sha256"]
    if not isinstance(identity_hash, str) or not HEX64.fullmatch(identity_hash):
        raise V8KPrivatePartitionBlocked("RECEIPT_SCHEMA_INVALID")
    if not isinstance(receipt["consumption_timestamp_utc"], str) or not receipt["consumption_timestamp_utc"]:
        raise V8KPrivatePartitionBlocked("RECEIPT_SCHEMA_INVALID")
    return dict(receipt)


def _read_receipt(gate_state_root: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = _gate_receipt_path(gate_state_root)
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
    except Exception as error:
        raise V8KPrivatePartitionBlocked("RECEIPT_SCHEMA_INVALID") from error
    return _validate_receipt(parsed)


def _consume_gate(gate_state_root: str | os.PathLike[str], auth_hash: str, now: Callable[[], datetime]) -> dict[str, Any]:
    path = _gate_receipt_path(gate_state_root)
    if path.exists():
        raise V8KPrivatePartitionBlocked("GATE_ALREADY_CONSUMED")
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "study": STUDY,
        "gate": GATE,
        "receipt_key_sha256": receipt_key(),
        "authorization_identity_sha256": auth_hash,
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": CONSUMPTION_BOUNDARY,
        "consumption_timestamp_utc": now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    payload = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _atomic_publish_once(payload, path, "GATE_ALREADY_CONSUMED", "RECEIPT_WRITE_FAILED")
    return receipt


# ---------------------------------------------------------------------------
# §6 -- Exactly-once 32-byte OS-CSPRNG seed; exclusive/no-overwrite publish
# ---------------------------------------------------------------------------


def _generate_seed() -> bytes:
    return os.urandom(SEED_BYTES)


def _seed_path(private_state_root: str | os.PathLike[str]) -> Path:
    return Path(private_state_root) / (receipt_key() + ".seed")


def _persist_seed_once(private_state_root: str | os.PathLike[str], seed: bytes) -> None:
    if not isinstance(seed, (bytes, bytearray)) or len(seed) != SEED_BYTES:
        raise V8KPrivatePartitionBlocked("SEED_BYTES_INVALID")
    path = _seed_path(private_state_root)
    try:
        _atomic_publish_once(bytes(seed), path, "SEED_ALREADY_EXISTS", "SEED_PERSISTENCE_FAILED_BLOCK_CLOSED")
    except V8KPrivatePartitionBlocked as error:
        if error.reason == "SEED_ALREADY_EXISTS":
            raise
        raise V8KPrivatePartitionBlocked("SEED_PERSISTENCE_FAILED_BLOCK_CLOSED") from error


def _read_seed(private_state_root: str | os.PathLike[str]) -> bytes | None:
    path = _seed_path(private_state_root)
    if not path.exists():
        return None
    try:
        data = path.read_bytes()
    except OSError as error:
        raise V8KPrivatePartitionBlocked("SEED_READ_FAILED") from error
    if len(data) != SEED_BYTES:
        raise V8KPrivatePartitionBlocked("SEED_BYTES_INVALID")
    return data


# ---------------------------------------------------------------------------
# Private manifest (V8K_PARTITION_MANIFEST_V1) -- mechanically re-verified,
# never merely trusted, on every later read.
# ---------------------------------------------------------------------------


def _manifest_path(private_state_root: str | os.PathLike[str]) -> Path:
    return Path(private_state_root) / (receipt_key() + ".manifest.json")


def _build_manifest(
    *,
    support_sha: str,
    auth_hash: str,
    source_raw_sha256: str,
    eligible_ticker_count: int,
    eligible_ticker_list_sha256: str,
    seed: bytes,
    fresh_pool_count: int,
    t0: list[str],
    t1: list[str],
    t2: list[str],
    t3: list[str],
    t_spare: list[str],
    now: Callable[[], datetime],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "study": STUDY,
        "gate": GATE,
        "frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "frozen_design_blob": FROZEN_DESIGN_BLOB,
        "reviewed_support_implementation_sha": support_sha,
        "authorization_identity_sha256": auth_hash,
        "receipt_key_sha256": receipt_key(),
        "source_raw_sha256": source_raw_sha256,
        "eligible_ticker_count": eligible_ticker_count,
        "eligible_ticker_list_sha256": eligible_ticker_list_sha256,
        "seed_sha256": sha256(seed),
        "fresh_pool_count": fresh_pool_count,
        "t0_ticker_list_sha256": ticker_list_sha256(t0),
        "t1_ticker_list_sha256": ticker_list_sha256(t1),
        "t2_ticker_list_sha256": ticker_list_sha256(t2),
        "t3_ticker_list_sha256": ticker_list_sha256(t3),
        "t_spare_ticker_list_sha256": ticker_list_sha256(t_spare),
        "block_sizes": {"T0": len(t0), "T1": len(t1), "T2": len(t2), "T3": len(t3), "T_spare": len(t_spare)},
        "block_assignments": {"T0": list(t0), "T1": list(t1), "T2": list(t2), "T3": list(t3), "T_spare": list(t_spare)},
        "created_utc": now().astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    if set(manifest) != set(MANIFEST_FIELDS):
        raise V8KPrivatePartitionBlocked("MANIFEST_SCHEMA_INVALID")
    return manifest


_BLOCK_HASH_FIELD = {
    "T0": "t0_ticker_list_sha256",
    "T1": "t1_ticker_list_sha256",
    "T2": "t2_ticker_list_sha256",
    "T3": "t3_ticker_list_sha256",
    "T_spare": "t_spare_ticker_list_sha256",
}


def _verify_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanically re-verify schema, self-hash, source binding, sizes,
    overlaps/duplicates, and every block-list hash. A persisted manifest's
    self-declared correctness is never trusted."""
    if not isinstance(manifest, Mapping) or set(manifest) != set(MANIFEST_FIELDS):
        raise V8KPrivatePartitionBlocked("MANIFEST_SCHEMA_INVALID")
    recomputed = canonical_sha256({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    if manifest["manifest_sha256"] != recomputed:
        raise V8KPrivatePartitionBlocked("MANIFEST_SHA_MISMATCH")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION or manifest["study"] != STUDY or manifest["gate"] != GATE:
        raise V8KPrivatePartitionBlocked("MANIFEST_SCHEMA_INVALID")
    if manifest["frozen_design_commit"] != FROZEN_DESIGN_COMMIT or manifest["frozen_design_blob"] != FROZEN_DESIGN_BLOB:
        raise V8KPrivatePartitionBlocked("MANIFEST_SCHEMA_INVALID")
    if manifest["receipt_key_sha256"] != receipt_key():
        raise V8KPrivatePartitionBlocked("MANIFEST_SCHEMA_INVALID")
    if manifest["source_raw_sha256"] != STAGE1_SOURCE_RAW_SHA256:
        raise V8KPrivatePartitionBlocked("STAGE1_SOURCE_RAW_SHA256_MISMATCH")

    blocks = manifest["block_assignments"]
    sizes = manifest["block_sizes"]
    if not isinstance(blocks, Mapping) or set(blocks) != set(_BLOCK_HASH_FIELD):
        raise V8KPrivatePartitionBlocked("MANIFEST_SCHEMA_INVALID")
    if not isinstance(sizes, Mapping) or set(sizes) != set(_BLOCK_HASH_FIELD):
        raise V8KPrivatePartitionBlocked("MANIFEST_SCHEMA_INVALID")
    for name in ("T0", "T1", "T2", "T3"):
        if sizes.get(name) != BLOCK_SIZE or len(blocks[name]) != BLOCK_SIZE:
            raise V8KPrivatePartitionBlocked("BLOCK_SIZE_INVALID")
    if sizes.get("T_spare") != len(blocks["T_spare"]):
        raise V8KPrivatePartitionBlocked("BLOCK_SIZE_INVALID")

    all_assigned: list[str] = []
    for name in ("T0", "T1", "T2", "T3", "T_spare"):
        all_assigned.extend(blocks[name])
    if len(set(all_assigned)) != len(all_assigned):
        raise V8KPrivatePartitionBlocked("BLOCK_OVERLAP_DETECTED")

    for name, hash_field in _BLOCK_HASH_FIELD.items():
        if manifest[hash_field] != ticker_list_sha256(blocks[name]):
            raise V8KPrivatePartitionBlocked("MANIFEST_SCHEMA_INVALID")
    return dict(manifest)


def _write_manifest_once(private_state_root: str | os.PathLike[str], manifest: Mapping[str, Any]) -> None:
    path = _manifest_path(private_state_root)
    _atomic_publish_once(canonical_json_bytes(manifest), path, "MANIFEST_ALREADY_EXISTS", "MANIFEST_WRITE_FAILED")


def _read_manifest(private_state_root: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = _manifest_path(private_state_root)
    if not path.exists():
        return None

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8KPrivatePartitionBlocked("MANIFEST_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        raw = path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except V8KPrivatePartitionBlocked:
        raise
    except Exception as error:
        raise V8KPrivatePartitionBlocked("MANIFEST_SCHEMA_INVALID") from error
    return _verify_manifest(parsed)


# ---------------------------------------------------------------------------
# Safe public evidence (hashes/counts/booleans/enums only -- never ticker
# identities/order, seed bytes, private paths, raw manifest, raw
# authorization, or raw payload)
# ---------------------------------------------------------------------------


def _build_evidence(manifest: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    blocks = manifest["block_assignments"]
    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "artifact_role": EVIDENCE_ARTIFACT_ROLE,
        "study": STUDY,
        "stage": "PRIVATE_PARTITION_ESTABLISHMENT",
        "gate": GATE,
        "frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "frozen_design_blob": FROZEN_DESIGN_BLOB,
        "reviewed_support_implementation_sha": manifest["reviewed_support_implementation_sha"],
        "authorization_identity_sha256": manifest["authorization_identity_sha256"],
        "receipt_key_sha256": manifest["receipt_key_sha256"],
        "source_raw_sha256": manifest["source_raw_sha256"],
        "eligible_ticker_count": manifest["eligible_ticker_count"],
        "eligible_ticker_list_sha256": manifest["eligible_ticker_list_sha256"],
        "seed_sha256": manifest["seed_sha256"],
        "private_manifest_sha256": manifest["manifest_sha256"],
        "t0_ticker_list_sha256": manifest["t0_ticker_list_sha256"],
        "t1_ticker_list_sha256": manifest["t1_ticker_list_sha256"],
        "t2_ticker_list_sha256": manifest["t2_ticker_list_sha256"],
        "t3_ticker_list_sha256": manifest["t3_ticker_list_sha256"],
        "t_spare_ticker_list_sha256": manifest["t_spare_ticker_list_sha256"],
        "t0_count": len(blocks["T0"]),
        "t1_count": len(blocks["T1"]),
        "t2_count": len(blocks["T2"]),
        "t3_count": len(blocks["T3"]),
        "t_spare_count": len(blocks["T_spare"]),
        "receipt_consumed": receipt["consumed"],
        "consumption_count": receipt["consumption_count"],
        "result_classification": "COMPLETE",
    }
    if set(evidence) != set(EVIDENCE_FIELDS):
        raise V8KPrivatePartitionBlocked("EVIDENCE_SCHEMA_INVALID")
    return evidence


# ---------------------------------------------------------------------------
# Full DI execution boundary (private test seam) + narrow production API
# ---------------------------------------------------------------------------


def _establish_for_test(
    *,
    raw_authorization: str,
    support_sha: str,
    raw_source_bytes: bytes,
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    gate_state_root: str | os.PathLike[str],
    private_state_root: str | os.PathLike[str],
    now: Callable[[], datetime],
    seed_generator: Callable[[], bytes] = _generate_seed,
) -> dict[str, Any]:
    """Private test seam. Production callers must use
    establish_private_partition(), never any of these parameters."""
    ordered_codes, t0_tickers, fresh_pool = verify_stage1_binding_and_compute_fresh_pool(
        raw_source_bytes=raw_source_bytes,
        parse_source_table=parse_source_table,
        v4_manifest_path=v4_manifest_path,
        v4_universe_csv_path=v4_universe_csv_path,
    )
    eligible_ticker_list_sha256 = ticker_list_sha256(ordered_codes)
    auth_hash = validate_authorization(
        raw_authorization,
        design_commit=FROZEN_DESIGN_COMMIT,
        support_sha=support_sha,
        source_raw_sha256=STAGE1_SOURCE_RAW_SHA256,
    )

    receipt = _read_receipt(gate_state_root)
    existing_manifest = _read_manifest(private_state_root)
    existing_seed = _read_seed(private_state_root)

    if receipt is None:
        # Pre-gate: an already-present seed or manifest is an unexplained
        # invariant violation, never silently reused or overwritten.
        if existing_seed is not None or existing_manifest is not None:
            raise V8KPrivatePartitionBlocked("EXISTING_STATE_COLLISION")
        receipt = _consume_gate(gate_state_root, auth_hash, now)
        seed = seed_generator()
        _persist_seed_once(private_state_root, seed)
    else:
        if receipt["authorization_identity_sha256"] != auth_hash:
            raise V8KPrivatePartitionBlocked("RECEIPT_AUTHORIZATION_MISMATCH")
        if existing_seed is None:
            # Gate consumed but seed persistence never succeeded: the design
            # is explicit that this is BLOCK_CLOSED -- never a second seed.
            raise V8KPrivatePartitionBlocked("SEED_MISSING_AFTER_GATE_CONSUMED_BLOCK_CLOSED")
        seed = existing_seed

    blocks = allocate_v8k_blocks(fresh_pool, seed)

    if existing_manifest is not None:
        # Deterministic continuation: prove recomputing from the exact same
        # persisted seed and locked source reproduces the exact same
        # already-persisted manifest -- never reroll, never silently trust.
        if (
            existing_manifest["source_raw_sha256"] != STAGE1_SOURCE_RAW_SHA256
            or existing_manifest["eligible_ticker_list_sha256"] != eligible_ticker_list_sha256
            or existing_manifest["seed_sha256"] != sha256(seed)
            or existing_manifest["authorization_identity_sha256"] != auth_hash
            or existing_manifest["t0_ticker_list_sha256"] != ticker_list_sha256(t0_tickers)
            or existing_manifest["t1_ticker_list_sha256"] != ticker_list_sha256(blocks["T1"])
            or existing_manifest["t2_ticker_list_sha256"] != ticker_list_sha256(blocks["T2"])
            or existing_manifest["t3_ticker_list_sha256"] != ticker_list_sha256(blocks["T3"])
            or existing_manifest["t_spare_ticker_list_sha256"] != ticker_list_sha256(blocks["T_spare"])
        ):
            raise V8KPrivatePartitionBlocked("MANIFEST_SHA_MISMATCH")
        manifest = existing_manifest
    else:
        manifest = _build_manifest(
            support_sha=support_sha,
            auth_hash=auth_hash,
            source_raw_sha256=STAGE1_SOURCE_RAW_SHA256,
            eligible_ticker_count=len(ordered_codes),
            eligible_ticker_list_sha256=eligible_ticker_list_sha256,
            seed=seed,
            fresh_pool_count=len(fresh_pool),
            t0=t0_tickers,
            t1=blocks["T1"],
            t2=blocks["T2"],
            t3=blocks["T3"],
            t_spare=blocks["T_spare"],
            now=now,
        )
        _write_manifest_once(private_state_root, manifest)

    return _build_evidence(manifest, receipt)


def _production_dependencies() -> tuple[Callable[[bytes], Any], Path, Path]:
    from scripts.build_v8_partition_manifest import default_parse_source_table

    return (
        default_parse_source_table,
        CANONICAL_REPOSITORY_ROOT / "V4_UNIVERSE_MANIFEST.json",
        CANONICAL_REPOSITORY_ROOT / "V4_UNIVERSE.csv",
    )


def _read_locked_stage1_raw_bytes() -> bytes:
    """Read Stage-1's real machine-local locked raw bytes. Never invoked by
    any test in this file -- there is no real Stage-1 lock in this
    environment, so a real call here fails closed on a missing file."""
    path = CANONICAL_V8K_PUBLIC_SOURCE_STATE_ROOT / (_stage1_receipt_key() + ".raw")
    try:
        return path.read_bytes()
    except OSError as error:
        raise V8KPrivatePartitionBlocked("STAGE1_LOCKED_RAW_MISSING") from error


def establish_private_partition(*, raw_authorization: str) -> dict[str, Any]:
    """Production-facing API: fixed canonical machine-local state only.

    Exposes no test override for state roots, seed generator, parser,
    source path, gate receipt, or private paths -- every one of those is
    wired internally to the canonical/reviewed value."""
    try:
        support_sha = production_provenance()
        parser, v4_manifest_path, v4_universe_csv_path = _production_dependencies()
        raw_source_bytes = _read_locked_stage1_raw_bytes()
        return _establish_for_test(
            raw_authorization=raw_authorization,
            support_sha=support_sha,
            raw_source_bytes=raw_source_bytes,
            parse_source_table=parser,
            v4_manifest_path=v4_manifest_path,
            v4_universe_csv_path=v4_universe_csv_path,
            gate_state_root=CANONICAL_V8K_PRIVATE_PARTITION_GATE_STATE_ROOT,
            private_state_root=CANONICAL_V8K_PRIVATE_PARTITION_PRIVATE_STATE_ROOT,
            now=lambda: datetime.now(timezone.utc),
        )
    except V8KPrivatePartitionBlocked:
        raise
    except Exception as error:
        # Total fail-closed boundary: any unexpected implementation/runtime
        # exception not already classified above must still resolve to a
        # safe public failure class rather than escape raw.
        raise V8KPrivatePartitionBlocked("IMPLEMENTATION_FAILURE") from error
