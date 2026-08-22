"""V8J source-snapshot acquisition support
(`V8J_SOURCE_SNAPSHOT_ENVIRONMENT_SUCCESSOR_DESIGN_DRAFT.md` sections 3-8).

This module implements the frozen V8J source-snapshot acquisition stage:
the `HUMAN_V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE` one-shot gate (design's
own frozen authorization grammar and deterministic fixed receipt key,
durable exclusive/no-overwrite receipt publication), inherited V8/V8H
eligible-universe reconstruction/T0-reproduction reuse (via
`src.v8_partition`'s existing public primitives, unmodified), the
fresh-eligible-count ( >= 900 ) fail-closed check, private raw-source-bytes
preservation outside Git, and the safe public source-snapshot evidence
artifact producer/verifier. It does not implement V8J partition
generation -- that is explicitly out of this narrower task's scope
(design §7's closing note; design §9's CHATGPT_DECISION_REQUIRED list).
It performs no partition allocation, no seed generation, no membership
disclosure, and no research opening.

Bound to the exact independently reviewed and human-frozen V8J design
candidate:

    reviewed_v8j_design_candidate_commit = 0eeb207108235cd305c6a8f8f90253a3896c9165
    design_blob = b3e03ca0bef40380fef66083d91d74ebd79e617e
    freeze_approval_blob = 9796595a8783ec075fbc6e757fc0063ffbb0d0e2

Design §3 explicitly and deliberately freezes a two-artifact split to
resolve the exact receipt-content-timing contradiction GPT independent
review found in V8H's own frozen design §7.1
(`FROZEN_DESIGN_RECEIPT_CONTRACT_DEVIATION_WITHOUT_AUTHORITY`, BLOCK on
commit `ecf74d7fb3a093bce5ceae372bd2d02c8499e43d`): (A) a minimal
pre-request one-shot gate receipt, published strictly before the one
authorized JPX request and containing only values knowable at that
instant, and (B) a distinct, cryptographically bound, post-request
public-safe evidence artifact carrying every value the request and its
parsing produce. This module implements exactly that frozen split under
a fresh V8J-only namespace -- its gate name, receipt-key material, and
authorization grammar never reuse V8H's or V8G's. No raw ticker code,
private path, or raw payload byte is ever a field in either artifact.

This is a fresh V8J-namespaced implementation. It does not import from,
edit, or treat as operational authority the V8I implementation or any V8I
gate, receipt, key, or private-state location; V8I remains historical
`BLOCK_CLOSED` evidence only. `src.v8_partition` and
`src.v8c_git_provenance` are reused only as generic/public primitives.

Importing this module performs no I/O, no network access, and no gate
consumption. The production entry point requires an explicit
`jpx_fetcher` and `parse_source_table` with no default wiring to a real
network call -- this implementation task grants zero real JPX request
authority, and no default production fetcher is wired here.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v8_partition import (
    LEGACY_EXPOSED_TICKERS_OUTSIDE_T0,
    SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
    SOURCE_SNAPSHOT_SEMANTICS,
    V4_RAW_SHA_EQUALITY_REQUIRED,
    V8PartitionBlocked,
    canonical_order,
    load_v4_provenance,
    load_v4_universe_csv_bytes,
    parse_eligible_universe,
    sha256_bytes,
    ticker_list_sha256,
    verify_t0_reproduction,
)
from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    read_git_object_bytes,
    require_strict_git_ancestor,
    resolve_git_blob,
)
from src.v8c_human_gate_consumption import CANONICAL_CONSUMPTION_STATE_ROOT

V8J_STUDY_NAME = "V8J_HISTORICAL_RESEARCH"
V8J_REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
# The authoritative *remote* branch this study's every task has verified
# against (`origin/<this>` == expected HEAD). The Claude Cloud *local*
# checkout branch name is explicitly irrelevant per repository governance
# and is never checked here -- only the remote-tracking ref matters.
V8J_AUTHORITATIVE_BRANCH = "v8g-private-partition-locator-successor-design"

# The exact independently reviewed V8J design candidate this
# implementation is bound to.
REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT = "0eeb207108235cd305c6a8f8f90253a3896c9165"
V8J_DESIGN_CANDIDATE_BLOB_SHA = "b3e03ca0bef40380fef66083d91d74ebd79e617e"
V8J_DESIGN_DRAFT_GIT_PATH = "V8J_SOURCE_SNAPSHOT_ENVIRONMENT_SUCCESSOR_DESIGN_DRAFT.md"

# The freeze-approval artifact is a *separate* file with its own,
# necessarily *later* history: V8J_DESIGN_FREEZE_APPROVAL.json did not
# exist at REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT at all (it is created only
# after that design candidate is independently reviewed) and reaches its
# final `APPROVED_FROZEN` content only at its own later commit. Binding
# both the design and the freeze artifact to the same design-candidate
# commit is therefore not merely imprecise but impossible to satisfy
# against real repository history -- `_default_public_preflight` would
# always fail resolving the freeze blob from that commit. The exact
# freeze-record commit below was found mechanically
# (`git log --follow -- V8J_DESIGN_FREEZE_APPROVAL.json`) as the commit
# that introduced the final `APPROVED_FROZEN` content, and independently
# confirmed a real ancestor of this branch's current HEAD. It is bound
# and verified completely separately from, and never conflated with,
# REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT.
V8J_FREEZE_APPROVAL_GIT_PATH = "V8J_DESIGN_FREEZE_APPROVAL.json"
REVIEWED_V8J_FREEZE_RECORD_COMMIT = "e09188ffc4e138b62399cd3c47f4d619c0feccfc"
V8J_FREEZE_APPROVAL_BLOB_SHA = "9796595a8783ec075fbc6e757fc0063ffbb0d0e2"
V8J_ENVIRONMENT_FREEZE_PROMOTION_COMMIT = "f26c4138bd7b1fb9ea1394ed04a1a600a3fee425"
V8J_CANONICAL_INTERPRETER_RELATIVE_PATH = ".venv-real-execution\\Scripts\\python.exe"
V8J_ENVIRONMENT_CHECKER_GIT_PATH = "scripts/check_real_execution_env.py"
V8J_ENVIRONMENT_CRITICAL_GIT_PATHS = (
    "REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json",
    "REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json",
    "REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json",
    "requirements-real-execution.lock.txt",
    "requirements-real-execution.txt",
    "scripts/check_real_execution_env.py",
    "scripts/bootstrap_real_execution_env.ps1",
    "tests/fixtures/synthetic_jpx_source_snapshot.xls",
)

BLOCK_SIZE = 300
MINIMUM_FRESH_ELIGIBLE_COUNT = 900  # 3 x BLOCK_SIZE (inherited from V8H design §7)

V8J_SOURCE_SNAPSHOT_GATE = "HUMAN_V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE"
V8J_SOURCE_SNAPSHOT_GATE_CONSUMPTION_BOUNDARY = "IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST"

V8J_AUTHORIZATION_PREFIX = "V8J_HUMAN_AUTHORIZE_SOURCE_SNAPSHOT_ACQUISITION_AT_"
V8J_AUTHORIZATION_WITH = "_WITH_"

V8J_SOURCE_SNAPSHOT_RECEIPT_SCHEMA_VERSION = "V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_V1"
V8J_SOURCE_SNAPSHOT_RECEIPT_ARTIFACT_ROLE = "SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT"
V8J_SOURCE_SNAPSHOT_RECEIPT_FIELDS = (
    "schema_version",
    "artifact_role",
    "study",
    "gate",
    "reviewed_v8j_design_candidate_commit",
    "reviewed_source_snapshot_support_implementation_sha",
    "authorization_identity_sha256",
    "consumed",
    "consumption_count",
    "consumption_boundary",
    "consumption_timestamp_utc",
)

V8J_SOURCE_SNAPSHOT_EVIDENCE_SCHEMA_VERSION = "V8J_SOURCE_SNAPSHOT_ACQUISITION_EVIDENCE_V1"
V8J_SOURCE_SNAPSHOT_EVIDENCE_ARTIFACT_ROLE = "SOURCE_SNAPSHOT_ACQUISITION_EVIDENCE"
V8J_SOURCE_SNAPSHOT_EVIDENCE_FIELDS = (
    "schema_version",
    "artifact_role",
    "study",
    "reviewed_v8j_design_candidate_commit",
    "reviewed_source_snapshot_support_implementation_sha",
    "source_snapshot_gate_receipt_key_sha256",
    "source_snapshot_gate_receipt_bytes_sha256",
    "source_snapshot_semantics",
    "source_snapshot_clarification_commit",
    "v4_raw_sha_equality_required",
    "source_raw_sha256",
    "source_raw_byte_count",
    "source_acquisition_utc",
    "t0_reproduction_status",
    "eligible_ticker_count",
    "eligible_ticker_list_sha256",
    "t0_ticker_list_sha256",
    "fresh_eligible_count",
    "ticker_identities_exposed",
    "private_path_exposed",
    "raw_payload_exposed",
    "historical_price_raw_acquisition_performed",
    "partition_generation_authorized",
    "membership_disclosure_authorized",
    "research_opened",
    "source_snapshot_result",
    "source_snapshot_artifact_self_sha256",
)

CANONICAL_V8J_SOURCE_SNAPSHOT_GATE_STATE_ROOT = CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8j-source-snapshot-gate-state"
CANONICAL_V8J_SOURCE_SNAPSHOT_PRIVATE_STATE_ROOT = (
    CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8j-source-snapshot-private-state"
)

_HEX = re.compile(r"^[0-9a-f]+$")
_TIMESTAMP_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_TIMESTAMP_MICROS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


class V8JSourceSnapshotBlocked(RuntimeError):
    """Fail-closed V8J source-snapshot acquisition error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise V8JSourceSnapshotBlocked(reason)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V8JSourceSnapshotBlocked("V8J_NONFINITE_OR_UNSERIALIZABLE") from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_object(raw: bytes, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8JSourceSnapshotBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8JSourceSnapshotBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8JSourceSnapshotBlocked(invalid_reason)
    return parsed


def _timestamp_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8JSourceSnapshotBlocked("V8J_CLOCK_INVALID")
    utc = value.astimezone(timezone.utc)
    if utc.microsecond:
        return utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_clock() -> datetime:
    """The sole production post-gate clock, using reviewed stdlib only."""
    return datetime.now(timezone.utc)


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or not (_TIMESTAMP_SECONDS.fullmatch(value) or _TIMESTAMP_MICROS.fullmatch(value)):
        raise V8JSourceSnapshotBlocked("V8J_TIMESTAMP_INVALID")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ" if "." not in value else "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as error:
        raise V8JSourceSnapshotBlocked("V8J_TIMESTAMP_INVALID") from error
    return value


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
    """Write-once atomic publication shared by the receipt and the evidence
    artifact: staging write, fsync file, atomic no-overwrite link, fsync
    directory, staging cleanup. Never replaces an existing destination."""
    if output.exists():
        raise V8JSourceSnapshotBlocked(already_exists_reason)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (output.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(staging, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output)
        except FileExistsError as error:
            raise V8JSourceSnapshotBlocked(already_exists_reason) from error
        except OSError as error:
            raise V8JSourceSnapshotBlocked(write_failed_reason) from error
        _fsync_directory(output.parent)
    except V8JSourceSnapshotBlocked:
        raise
    except OSError as error:
        raise V8JSourceSnapshotBlocked(write_failed_reason) from error
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# §5 -- Human authorization grammar
# ---------------------------------------------------------------------------


def build_authorization_identity(
    *,
    reviewed_v8j_design_candidate_commit: str,
    reviewed_source_snapshot_support_implementation_sha: str,
) -> str:
    candidate = _require_hex(reviewed_v8j_design_candidate_commit, 40, "V8J_DESIGN_CANDIDATE_INVALID")
    implementation = _require_hex(
        reviewed_source_snapshot_support_implementation_sha, 40, "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_SHA_INVALID"
    )
    return V8J_AUTHORIZATION_PREFIX + candidate + V8J_AUTHORIZATION_WITH + implementation


def authorization_identity_sha256(authorization_identity: str) -> str:
    """Return only the SHA-256; never persist or expose the raw identity."""
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8JSourceSnapshotBlocked("V8J_AUTHORIZATION_IDENTITY_REQUIRED")
    return hashlib.sha256(authorization_identity.encode("utf-8")).hexdigest()


def validate_authorization_identity(
    authorization_identity: str,
    *,
    reviewed_source_snapshot_support_implementation_sha: str,
    reviewed_v8j_design_candidate_commit: str = REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
) -> None:
    """Require the exact V8J source-snapshot authorization grammar and binding."""
    candidate = _require_hex(reviewed_v8j_design_candidate_commit, 40, "V8J_DESIGN_CANDIDATE_INVALID")
    implementation = _require_hex(
        reviewed_source_snapshot_support_implementation_sha, 40, "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_SHA_INVALID"
    )
    if candidate != REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT:
        raise V8JSourceSnapshotBlocked("V8J_DESIGN_CANDIDATE_MISMATCH")
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8JSourceSnapshotBlocked("V8J_AUTHORIZATION_GRAMMAR_MISMATCH")
    expected = build_authorization_identity(
        reviewed_v8j_design_candidate_commit=candidate,
        reviewed_source_snapshot_support_implementation_sha=implementation,
    )
    if authorization_identity != expected:
        raise V8JSourceSnapshotBlocked("V8J_AUTHORIZATION_GRAMMAR_MISMATCH")


# ---------------------------------------------------------------------------
# §4 -- Deterministic one-shot receipt key (independent of authorization,
# design candidate, and implementation SHA -- design's explicit rationale;
# fresh V8J-only namespace, never reuses V8H's or V8G's material)
# ---------------------------------------------------------------------------

_RECEIPT_KEY_MATERIAL = (
    "V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_KEY_V1\0"
    + V8J_REPOSITORY_IDENTITY
    + "\0"
    + V8J_STUDY_NAME
    + "\0"
    + V8J_SOURCE_SNAPSHOT_GATE
).encode("utf-8")


def compute_source_snapshot_gate_receipt_key() -> str:
    """Fixed, deterministic receipt key: repository + study + gate only.

    Deliberately takes no arguments -- the authorization identity, the
    reviewed design candidate commit, and the reviewed source-snapshot-
    support implementation SHA must never affect this key, so the gate can
    be durably consumed at most once for the entire life of the V8J study.
    """
    return hashlib.sha256(_RECEIPT_KEY_MATERIAL).hexdigest()


def _receipt_path(state_root: str | os.PathLike[str]) -> Path:
    return Path(state_root) / (compute_source_snapshot_gate_receipt_key() + ".json")


def _validate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if set(receipt) != set(V8J_SOURCE_SNAPSHOT_RECEIPT_FIELDS):
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_SCHEMA_INVALID")
    if receipt["schema_version"] != V8J_SOURCE_SNAPSHOT_RECEIPT_SCHEMA_VERSION:
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_SCHEMA_VERSION_MISMATCH")
    if receipt["artifact_role"] != V8J_SOURCE_SNAPSHOT_RECEIPT_ARTIFACT_ROLE:
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_IDENTITY_INVALID")
    if receipt["study"] != V8J_STUDY_NAME or receipt["gate"] != V8J_SOURCE_SNAPSHOT_GATE:
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_IDENTITY_INVALID")
    _require_hex(receipt["reviewed_v8j_design_candidate_commit"], 40, "V8J_RECEIPT_DESIGN_CANDIDATE_INVALID")
    _require_hex(
        receipt["reviewed_source_snapshot_support_implementation_sha"],
        40,
        "V8J_RECEIPT_IMPLEMENTATION_SHA_INVALID",
    )
    _require_hex(receipt["authorization_identity_sha256"], 64, "V8J_RECEIPT_AUTHORIZATION_HASH_INVALID")
    if (
        receipt["consumed"] is not True
        or type(receipt["consumption_count"]) is not int
        or receipt["consumption_count"] != 1
    ):
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_CONSUMPTION_INVALID")
    if receipt["consumption_boundary"] != V8J_SOURCE_SNAPSHOT_GATE_CONSUMPTION_BOUNDARY:
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_CONSUMPTION_BOUNDARY_INVALID")
    _validate_timestamp(receipt["consumption_timestamp_utc"])
    return dict(receipt)


def read_gate_receipt(state_root: str | os.PathLike[str]) -> dict[str, Any]:
    path = _receipt_path(state_root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_MISSING") from error
    return _validate_receipt(_strict_json_object(raw, "V8J_RECEIPT_INVALID_JSON", "V8J_RECEIPT_DUPLICATE_KEY"))


def gate_receipt_bytes_sha256(state_root: str | os.PathLike[str]) -> str:
    """Validate and hash the exact durable receipt bytes externally."""
    path = _receipt_path(state_root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_MISSING") from error
    _validate_receipt(_strict_json_object(raw, "V8J_RECEIPT_INVALID_JSON", "V8J_RECEIPT_DUPLICATE_KEY"))
    return hashlib.sha256(raw).hexdigest()


def consume_gate_once(
    state_root: str | os.PathLike[str],
    authorization_identity: str,
    *,
    clock: Callable[[], datetime],
    reviewed_v8j_design_candidate_commit: str,
    reviewed_source_snapshot_support_implementation_sha: str,
) -> dict[str, Any]:
    """Durably publish exactly one V8J source-snapshot receipt.

    Published strictly before any JPX byte is fetched (design §3.1's
    `consumption_boundary=IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST`), so it
    can contain only pre-known binding values -- never a value that
    depends on the fetch this receipt is about to authorize (design §3.1
    explicitly prohibits `source_raw_sha256`, `source_acquisition_utc`,
    `eligible_ticker_count`, `eligible_ticker_list_sha256`, and
    `t0_reproduction_status` from ever appearing here). No reset or
    replay API exists.
    """
    validate_authorization_identity(
        authorization_identity,
        reviewed_source_snapshot_support_implementation_sha=reviewed_source_snapshot_support_implementation_sha,
        reviewed_v8j_design_candidate_commit=reviewed_v8j_design_candidate_commit,
    )
    root = Path(state_root)
    path = _receipt_path(root)
    if path.exists():
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED")
    receipt = {
        "schema_version": V8J_SOURCE_SNAPSHOT_RECEIPT_SCHEMA_VERSION,
        "artifact_role": V8J_SOURCE_SNAPSHOT_RECEIPT_ARTIFACT_ROLE,
        "study": V8J_STUDY_NAME,
        "gate": V8J_SOURCE_SNAPSHOT_GATE,
        "reviewed_v8j_design_candidate_commit": reviewed_v8j_design_candidate_commit,
        "reviewed_source_snapshot_support_implementation_sha": reviewed_source_snapshot_support_implementation_sha,
        "authorization_identity_sha256": authorization_identity_sha256(authorization_identity),
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": V8J_SOURCE_SNAPSHOT_GATE_CONSUMPTION_BOUNDARY,
        "consumption_timestamp_utc": _timestamp_text(clock()),
    }
    if set(receipt) != set(V8J_SOURCE_SNAPSHOT_RECEIPT_FIELDS):
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_SCHEMA_INVALID")
    try:
        _atomic_publish_once(
            _canonical_json_bytes(receipt),
            path,
            "V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED",
            "V8J_RECEIPT_STORAGE_WRITE_FAILED",
        )
    except V8JSourceSnapshotBlocked:
        raise
    return dict(receipt)


def _read_and_bind_gate_receipt(
    state_root: str | os.PathLike[str],
    *,
    reviewed_v8j_design_candidate_commit: str,
    reviewed_source_snapshot_support_implementation_sha: str,
    authorization_identity: str,
) -> tuple[dict[str, Any], str]:
    """Post-gate, pre-evidence-publication receipt semantic binding.

    Reads the exact durable receipt once, validates its structural schema,
    then mechanically requires exact equality between every one of its
    bound fields and this execution's own authorized values -- never
    trusting a structurally well-formed but semantically stale receipt.
    Returns the validated receipt together with the SHA-256 of its exact
    durable bytes -- this is the "point-of-use authorization verification"
    design §3.2 requires before evidence publication.
    """
    path = _receipt_path(state_root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_MISSING") from error
    receipt = _validate_receipt(_strict_json_object(raw, "V8J_RECEIPT_INVALID_JSON", "V8J_RECEIPT_DUPLICATE_KEY"))
    if receipt["reviewed_v8j_design_candidate_commit"] != reviewed_v8j_design_candidate_commit:
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_DESIGN_CANDIDATE_MISMATCH")
    if (
        receipt["reviewed_source_snapshot_support_implementation_sha"]
        != reviewed_source_snapshot_support_implementation_sha
    ):
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_IMPLEMENTATION_SHA_MISMATCH")
    if receipt["authorization_identity_sha256"] != authorization_identity_sha256(authorization_identity):
        raise V8JSourceSnapshotBlocked("V8J_RECEIPT_AUTHORIZATION_HASH_MISMATCH")
    return receipt, hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# §7 -- Inherited V8/V8H eligible-universe reconstruction / T0 reproduction /
# fresh-eligible-count fail-closed check (no allocation, no seed, no tier
# assignment -- explicitly out of this narrower task's scope)
# ---------------------------------------------------------------------------


def _perform_source_snapshot_acquisition(
    *,
    raw_source_bytes: bytes,
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    source_acquisition_utc: datetime,
    block_size: int = BLOCK_SIZE,
    minimum_fresh_eligible_count: int = MINIMUM_FRESH_ELIGIBLE_COUNT,
) -> dict[str, Any]:
    """Reuses `src.v8_partition`'s existing public primitives unchanged to
    reconstruct the eligible universe, verify exact T0 reproduction, and
    compute the fresh-eligible pool size after inherited exclusions.
    Returns only safe counts/hashes -- no ticker code, T0/legacy set, or
    fresh pool is ever included in the return value or logged/printed.
    """
    if not isinstance(raw_source_bytes, (bytes, bytearray)):
        raise V8JSourceSnapshotBlocked("V8J_RAW_SOURCE_BYTES_INVALID")
    raw_bytes = bytes(raw_source_bytes)
    try:
        v4_provenance = load_v4_provenance(v4_manifest_path)
        committed_csv_bytes = load_v4_universe_csv_bytes(v4_universe_csv_path)
    except V8PartitionBlocked as error:
        raise V8JSourceSnapshotBlocked("V8J_" + error.reason) from error
    if sha256_bytes(committed_csv_bytes) != v4_provenance["universe_csv_sha256"]:
        raise V8JSourceSnapshotBlocked("V8J_V4_UNIVERSE_CSV_PROVENANCE_MISMATCH")

    source_raw_sha256 = sha256_bytes(raw_bytes)

    try:
        frame = parse_source_table(raw_bytes)
        eligible_rows, _reasons = parse_eligible_universe(frame)
    except V8PartitionBlocked as error:
        raise V8JSourceSnapshotBlocked("V8J_" + error.reason) from error
    if not eligible_rows:
        raise V8JSourceSnapshotBlocked("V8J_ELIGIBLE_UNIVERSE_EMPTY")

    ordered_codes = canonical_order([row["code"] for row in eligible_rows])
    rows_by_code = {row["code"]: row for row in eligible_rows}
    if len(rows_by_code) != len(eligible_rows):
        raise V8JSourceSnapshotBlocked("V8J_ELIGIBLE_LIST_DUPLICATE_TICKER")
    eligible_rows_ordered = [rows_by_code[code] for code in ordered_codes]

    try:
        t0_tickers = verify_t0_reproduction(eligible_rows_ordered, v4_provenance, block_size=block_size)
    except V8PartitionBlocked as error:
        raise V8JSourceSnapshotBlocked("V8J_" + error.reason) from error

    legacy_set = set(LEGACY_EXPOSED_TICKERS_OUTSIDE_T0)
    t0_set = set(t0_tickers)
    if len(t0_set) != len(t0_tickers) or len(t0_set) != block_size:
        raise V8JSourceSnapshotBlocked("V8J_T0_SIZE_INVALID")
    exclude = t0_set | legacy_set
    fresh_pool = [ticker for ticker in ordered_codes if ticker not in exclude]
    fresh_eligible_count = len(fresh_pool)
    if fresh_eligible_count < minimum_fresh_eligible_count:
        raise V8JSourceSnapshotBlocked("V8J_FRESH_ELIGIBLE_POOL_INSUFFICIENT")

    acquired = source_acquisition_utc
    if not isinstance(acquired, datetime) or acquired.tzinfo is None or acquired.utcoffset() != timedelta(0):
        raise V8JSourceSnapshotBlocked("V8J_ACQUISITION_TIMESTAMP_INVALID")

    return {
        "source_snapshot_semantics": SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "v4_raw_sha_equality_required": V4_RAW_SHA_EQUALITY_REQUIRED,
        "source_raw_sha256": source_raw_sha256,
        "source_raw_byte_count": len(raw_bytes),
        "source_acquisition_utc": _timestamp_text(acquired),
        "t0_reproduction_status": "PASS",
        "eligible_ticker_count": len(ordered_codes),
        "eligible_ticker_list_sha256": ticker_list_sha256(ordered_codes),
        "t0_ticker_list_sha256": ticker_list_sha256(t0_tickers),
        "fresh_eligible_count": fresh_eligible_count,
    }


# ---------------------------------------------------------------------------
# Private raw-source-bytes preservation (outside Git, exclusive/no-overwrite)
# ---------------------------------------------------------------------------


def preserve_raw_source_bytes_once(
    private_state_root: str | os.PathLike[str],
    source_raw_sha256: str,
    raw_source_bytes: bytes,
) -> Path:
    """Durably preserve the raw source bytes privately, outside Git.

    Content-addressed by the caller-supplied, already-verified
    `source_raw_sha256` (never recomputed here from an untrusted value --
    callers must pass the value this module itself computed). Exclusive,
    no-overwrite publication with flush/fsync, mirroring every other
    durable-publication primitive in this module. Never returns or logs
    the raw bytes; only the destination path.
    """
    digest = _require_hex(source_raw_sha256, 64, "V8J_SOURCE_RAW_SHA_INVALID")
    if not isinstance(raw_source_bytes, (bytes, bytearray)):
        raise V8JSourceSnapshotBlocked("V8J_RAW_SOURCE_BYTES_INVALID")
    destination = Path(private_state_root) / (digest + ".raw")
    _atomic_publish_once(
        bytes(raw_source_bytes),
        destination,
        "V8J_PRIVATE_RAW_SOURCE_ALREADY_PRESERVED",
        "V8J_PRIVATE_RAW_SOURCE_WRITE_FAILED",
    )
    return destination


# ---------------------------------------------------------------------------
# §3.2 -- Safe public source-snapshot evidence artifact producer/verifier
# ---------------------------------------------------------------------------


def _build_source_snapshot_evidence(
    *,
    reviewed_v8j_design_candidate_commit: str,
    reviewed_source_snapshot_support_implementation_sha: str,
    source_snapshot_gate_receipt_key_sha256_value: str,
    source_snapshot_gate_receipt_bytes_sha256_value: str,
    acquisition_result: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema_version": V8J_SOURCE_SNAPSHOT_EVIDENCE_SCHEMA_VERSION,
        "artifact_role": V8J_SOURCE_SNAPSHOT_EVIDENCE_ARTIFACT_ROLE,
        "study": V8J_STUDY_NAME,
        "reviewed_v8j_design_candidate_commit": reviewed_v8j_design_candidate_commit,
        "reviewed_source_snapshot_support_implementation_sha": reviewed_source_snapshot_support_implementation_sha,
        "source_snapshot_gate_receipt_key_sha256": source_snapshot_gate_receipt_key_sha256_value,
        "source_snapshot_gate_receipt_bytes_sha256": source_snapshot_gate_receipt_bytes_sha256_value,
        "source_snapshot_semantics": acquisition_result["source_snapshot_semantics"],
        "source_snapshot_clarification_commit": acquisition_result["source_snapshot_clarification_commit"],
        "v4_raw_sha_equality_required": acquisition_result["v4_raw_sha_equality_required"],
        "source_raw_sha256": acquisition_result["source_raw_sha256"],
        "source_raw_byte_count": acquisition_result["source_raw_byte_count"],
        "source_acquisition_utc": acquisition_result["source_acquisition_utc"],
        "t0_reproduction_status": acquisition_result["t0_reproduction_status"],
        "eligible_ticker_count": acquisition_result["eligible_ticker_count"],
        "eligible_ticker_list_sha256": acquisition_result["eligible_ticker_list_sha256"],
        "t0_ticker_list_sha256": acquisition_result["t0_ticker_list_sha256"],
        "fresh_eligible_count": acquisition_result["fresh_eligible_count"],
        "ticker_identities_exposed": False,
        "private_path_exposed": False,
        "raw_payload_exposed": False,
        "historical_price_raw_acquisition_performed": False,
        "partition_generation_authorized": False,
        "membership_disclosure_authorized": False,
        "research_opened": False,
        "source_snapshot_result": "PASS",
    }
    self_hash = canonical_sha256(body)
    artifact = dict(body)
    artifact["source_snapshot_artifact_self_sha256"] = self_hash
    if set(artifact) != set(V8J_SOURCE_SNAPSHOT_EVIDENCE_FIELDS):
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_SCHEMA_INVALID")
    return artifact


def _validate_source_snapshot_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or set(evidence) != set(V8J_SOURCE_SNAPSHOT_EVIDENCE_FIELDS):
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_SCHEMA_INVALID")
    exact = {
        "schema_version": V8J_SOURCE_SNAPSHOT_EVIDENCE_SCHEMA_VERSION,
        "artifact_role": V8J_SOURCE_SNAPSHOT_EVIDENCE_ARTIFACT_ROLE,
        "study": V8J_STUDY_NAME,
        "source_snapshot_semantics": SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "v4_raw_sha_equality_required": False,
        "t0_reproduction_status": "PASS",
        "ticker_identities_exposed": False,
        "private_path_exposed": False,
        "raw_payload_exposed": False,
        "historical_price_raw_acquisition_performed": False,
        "partition_generation_authorized": False,
        "membership_disclosure_authorized": False,
        "research_opened": False,
        "source_snapshot_result": "PASS",
    }
    for key, expected in exact.items():
        if evidence[key] != expected or (isinstance(expected, bool) and type(evidence[key]) is not bool):
            raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_VALUE_INVALID:" + key)
    _require_hex(
        evidence["reviewed_v8j_design_candidate_commit"], 40, "V8J_SOURCE_SNAPSHOT_EVIDENCE_DESIGN_CANDIDATE_INVALID"
    )
    _require_hex(
        evidence["reviewed_source_snapshot_support_implementation_sha"],
        40,
        "V8J_SOURCE_SNAPSHOT_EVIDENCE_IMPLEMENTATION_SHA_INVALID",
    )
    _require_hex(
        evidence["source_snapshot_gate_receipt_key_sha256"],
        64,
        "V8J_SOURCE_SNAPSHOT_EVIDENCE_RECEIPT_KEY_INVALID",
    )
    _require_hex(
        evidence["source_snapshot_gate_receipt_bytes_sha256"],
        64,
        "V8J_SOURCE_SNAPSHOT_EVIDENCE_RECEIPT_BYTES_INVALID",
    )
    _require_hex(evidence["source_raw_sha256"], 64, "V8J_SOURCE_SNAPSHOT_EVIDENCE_SOURCE_RAW_SHA_INVALID")
    _require_hex(
        evidence["eligible_ticker_list_sha256"], 64, "V8J_SOURCE_SNAPSHOT_EVIDENCE_ELIGIBLE_HASH_INVALID"
    )
    _require_hex(evidence["t0_ticker_list_sha256"], 64, "V8J_SOURCE_SNAPSHOT_EVIDENCE_T0_HASH_INVALID")
    _validate_timestamp(evidence["source_acquisition_utc"])
    if type(evidence["source_raw_byte_count"]) is not int or evidence["source_raw_byte_count"] < 0:
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_BYTE_COUNT_INVALID")
    if type(evidence["eligible_ticker_count"]) is not int or evidence["eligible_ticker_count"] < 1:
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_ELIGIBLE_COUNT_INVALID")
    if type(evidence["fresh_eligible_count"]) is not int or evidence["fresh_eligible_count"] < 0:
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_FRESH_COUNT_INVALID")
    recomputed_self_hash = canonical_sha256(
        {key: value for key, value in evidence.items() if key != "source_snapshot_artifact_self_sha256"}
    )
    if evidence["source_snapshot_artifact_self_sha256"] != recomputed_self_hash:
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_SELF_HASH_MISMATCH")
    return dict(evidence)


def verify_source_snapshot_evidence_binding(
    evidence: Mapping[str, Any],
    *,
    authorized_reviewed_v8j_design_candidate_commit: str,
    authorized_reviewed_source_snapshot_support_implementation_sha: str,
) -> dict[str, Any]:
    """Point-of-use verification for a later stage consuming this artifact.

    Independently requires exact equality on the design-candidate binding
    and the source-snapshot-support implementation binding, requires the
    artifact's own receipt-key hash to equal this fixed key (rejecting a
    stale or substituted receipt binding), and recomputes the artifact's
    own self-hash rather than trusting the self-declared field alone.
    """
    validated = _validate_source_snapshot_evidence(evidence)
    if validated["reviewed_v8j_design_candidate_commit"] != authorized_reviewed_v8j_design_candidate_commit:
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_DESIGN_CANDIDATE_MISMATCH")
    if (
        validated["reviewed_source_snapshot_support_implementation_sha"]
        != authorized_reviewed_source_snapshot_support_implementation_sha
    ):
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_IMPLEMENTATION_MISMATCH")
    if validated["source_snapshot_gate_receipt_key_sha256"] != compute_source_snapshot_gate_receipt_key():
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_RECEIPT_KEY_MISMATCH")
    return validated


def _publish_source_snapshot_evidence(evidence: Mapping[str, Any], output: Path) -> dict[str, Any]:
    _atomic_publish_once(
        _canonical_json_bytes(evidence),
        output,
        "V8J_SOURCE_SNAPSHOT_EVIDENCE_ALREADY_EXISTS",
        "V8J_SOURCE_SNAPSHOT_EVIDENCE_WRITE_FAILED",
    )
    return dict(evidence)


# ---------------------------------------------------------------------------
# Authority/design/implementation binding (public preflight + reviewed
# source-snapshot-support implementation runtime binding + freeze-artifact
# verification)
# ---------------------------------------------------------------------------

_PUBLIC_PREFLIGHT_FIELDS = frozenset(
    {
        "repository_identity",
        "head",
        "authoritative_remote_head",
        "worktree_clean",
        "reviewed_v8j_design_candidate_commit",
        "reviewed_v8j_design_blob_sha",
        "freeze_record_commit",
        "freeze_approval_blob_sha",
        "freeze_approved_frozen",
    }
)


def _validate_public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(preflight, Mapping) or set(preflight) != _PUBLIC_PREFLIGHT_FIELDS:
        raise V8JSourceSnapshotBlocked("V8J_PUBLIC_PREFLIGHT_SCHEMA_INVALID")
    if preflight["repository_identity"] != V8J_REPOSITORY_IDENTITY:
        raise V8JSourceSnapshotBlocked("V8J_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    if preflight["worktree_clean"] is not True:
        raise V8JSourceSnapshotBlocked("V8J_PUBLIC_GIT_BINDING_INVALID")
    head = _require_hex(preflight["head"], 40, "V8J_PUBLIC_HEAD_INVALID")
    if preflight["authoritative_remote_head"] != head:
        raise V8JSourceSnapshotBlocked("V8J_PUBLIC_HEAD_NOT_AUTHORITATIVE_REMOTE")
    if preflight["reviewed_v8j_design_candidate_commit"] != REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT:
        raise V8JSourceSnapshotBlocked("V8J_DESIGN_CANDIDATE_MISMATCH")
    if preflight["reviewed_v8j_design_blob_sha"] != V8J_DESIGN_CANDIDATE_BLOB_SHA:
        raise V8JSourceSnapshotBlocked("V8J_DESIGN_CANDIDATE_BLOB_MISMATCH")
    # Independent binding #2: the human-freeze record. Deliberately never
    # compared to, derived from, or substituted for the design-candidate
    # binding above -- the freeze artifact has its own, necessarily later,
    # commit (it cannot exist at the pre-freeze design-candidate commit).
    if preflight["freeze_record_commit"] != REVIEWED_V8J_FREEZE_RECORD_COMMIT:
        raise V8JSourceSnapshotBlocked("V8J_FREEZE_RECORD_COMMIT_MISMATCH")
    if preflight["freeze_approval_blob_sha"] != V8J_FREEZE_APPROVAL_BLOB_SHA:
        raise V8JSourceSnapshotBlocked("V8J_FREEZE_APPROVAL_BLOB_MISMATCH")
    if preflight["freeze_approved_frozen"] is not True:
        raise V8JSourceSnapshotBlocked("V8J_FREEZE_NOT_APPROVED")
    return dict(preflight)


def _validate_freeze_approval_content(raw: bytes) -> bool:
    """Parse and mechanically verify the freeze-approval artifact content
    itself (not merely its blob identity) binds to this exact design
    candidate and records a completed human freeze with a clean GPT
    independent review. Returns True only if every check passes."""
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8JSourceSnapshotBlocked("V8J_FREEZE_APPROVAL_INVALID_JSON") from error
    if not isinstance(payload, Mapping):
        raise V8JSourceSnapshotBlocked("V8J_FREEZE_APPROVAL_INVALID_JSON")
    required = (
        payload.get("study") == V8J_STUDY_NAME
        and payload.get("frozen_design_git_commit") == REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT
        and payload.get("frozen_design_git_blob_sha") == V8J_DESIGN_CANDIDATE_BLOB_SHA
        and payload.get("approval_status") == "APPROVED_FROZEN"
        and payload.get("human_approval_received") is True
        and payload.get("human_design_freeze_complete") is True
        and payload.get("final_independent_design_review_result") == "PASS"
        and payload.get("critical") == 0
        and payload.get("high") == 0
        and payload.get("medium") == 0
        and payload.get("jpx_acquisition_authorized") is False
        and payload.get("future_profitability_established") is False
    )
    return bool(required)


def _git_text(repository_root: Path, args: list[str], reason: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8JSourceSnapshotBlocked("V8J_PUBLIC_GIT_UNAVAILABLE") from error
    if result.returncode != 0:
        raise V8JSourceSnapshotBlocked(reason)
    return result.stdout.strip()


def _default_public_preflight(repository_root: Path = CANONICAL_REPOSITORY_ROOT) -> dict[str, Any]:
    status = _git_text(repository_root, ["status", "--porcelain"], "V8J_PUBLIC_GIT_UNAVAILABLE")
    head = _git_text(repository_root, ["rev-parse", "HEAD"], "V8J_PUBLIC_HEAD_UNAVAILABLE")
    authoritative_remote_head = _git_text(
        repository_root,
        ["rev-parse", "origin/" + V8J_AUTHORITATIVE_BRANCH],
        "V8J_PUBLIC_ORIGIN_UNAVAILABLE",
    )
    origin_url = _git_text(repository_root, ["config", "--get", "remote.origin.url"], "V8J_PUBLIC_ORIGIN_UNAVAILABLE")
    if origin_url not in {
        "https://github.com/ta1k1-arakawa/stock-analyzer",
        "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "git@github.com:ta1k1-arakawa/stock-analyzer.git",
        "ssh://git@github.com/ta1k1-arakawa/stock-analyzer.git",
    }:
        raise V8JSourceSnapshotBlocked("V8J_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    # Binding #1: the frozen design, resolved from the exact reviewed
    # design-candidate commit.
    try:
        design_blob = resolve_git_blob(repository_root, REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT, V8J_DESIGN_DRAFT_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise V8JSourceSnapshotBlocked("V8J_PUBLIC_PROVENANCE_INVALID") from error
    if design_blob != V8J_DESIGN_CANDIDATE_BLOB_SHA:
        raise V8JSourceSnapshotBlocked("V8J_PUBLIC_PROVENANCE_INVALID")

    # Binding #2: the human-freeze record, resolved from its own, separate,
    # necessarily later commit -- never from REVIEWED_V8J_DESIGN_CANDIDATE_
    # COMMIT, at which V8J_DESIGN_FREEZE_APPROVAL.json does not exist at
    # all. The freeze-record commit must itself be a genuine, currently
    # reachable ancestor of this branch's authoritative remote HEAD (or that
    # exact HEAD while this support implementation is first being prepared).
    try:
        if authoritative_remote_head != REVIEWED_V8J_FREEZE_RECORD_COMMIT:
            require_strict_git_ancestor(
                repository_root,
                REVIEWED_V8J_FREEZE_RECORD_COMMIT,
                authoritative_remote_head,
                "V8J_FREEZE_RECORD_NOT_ANCESTOR",
            )
    except V8CGitProvenanceBlocked as error:
        raise V8JSourceSnapshotBlocked("V8J_FREEZE_RECORD_NOT_ANCESTOR") from error
    try:
        freeze_blob = resolve_git_blob(
            repository_root, REVIEWED_V8J_FREEZE_RECORD_COMMIT, V8J_FREEZE_APPROVAL_GIT_PATH
        )
    except V8CGitProvenanceBlocked as error:
        raise V8JSourceSnapshotBlocked("V8J_PUBLIC_PROVENANCE_INVALID") from error
    freeze_approved_frozen = False
    if freeze_blob == V8J_FREEZE_APPROVAL_BLOB_SHA:
        try:
            freeze_raw = read_git_object_bytes(
                repository_root, REVIEWED_V8J_FREEZE_RECORD_COMMIT, V8J_FREEZE_APPROVAL_GIT_PATH
            )
        except V8CGitProvenanceBlocked as error:
            raise V8JSourceSnapshotBlocked("V8J_PUBLIC_PROVENANCE_INVALID") from error
        freeze_approved_frozen = _validate_freeze_approval_content(freeze_raw)
    return _validate_public_preflight(
        {
            "repository_identity": V8J_REPOSITORY_IDENTITY,
            "head": head,
            "authoritative_remote_head": authoritative_remote_head,
            "worktree_clean": status == "",
            "reviewed_v8j_design_candidate_commit": REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            "reviewed_v8j_design_blob_sha": design_blob,
            "freeze_record_commit": REVIEWED_V8J_FREEZE_RECORD_COMMIT,
            "freeze_approval_blob_sha": freeze_blob,
            "freeze_approved_frozen": freeze_approved_frozen,
        }
    )


_REVIEWED_IMPLEMENTATION_RUNTIME_FIELDS = frozenset(
    {"head", "authoritative_remote_head", "worktree_clean", "commits_after_reviewed_implementation_sha"}
)


def _default_reviewed_source_snapshot_support_runtime_state(
    repository_root: Path, reviewed_source_snapshot_support_implementation_sha: str
) -> dict[str, Any]:
    reviewed_sha = _require_hex(
        reviewed_source_snapshot_support_implementation_sha, 40, "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_SHA_MALFORMED"
    )
    resolved_sha = _git_text(
        repository_root,
        ["rev-parse", "--verify", f"{reviewed_sha}^{{commit}}"],
        "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_SHA_UNRESOLVABLE",
    )
    if resolved_sha != reviewed_sha:
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_SHA_UNRESOLVABLE")
    count_text = _git_text(
        repository_root,
        ["rev-list", "--count", f"{reviewed_sha}..HEAD"],
        "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_CHRONOLOGY_INVALID",
    )
    if not count_text.isdecimal():
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_CHRONOLOGY_INVALID")
    return {
        "head": _git_text(repository_root, ["rev-parse", "HEAD"], "V8J_PUBLIC_HEAD_UNAVAILABLE"),
        "authoritative_remote_head": _git_text(
            repository_root,
            ["rev-parse", "origin/" + V8J_AUTHORITATIVE_BRANCH],
            "V8J_PUBLIC_ORIGIN_UNAVAILABLE",
        ),
        "worktree_clean": _git_text(repository_root, ["status", "--porcelain"], "V8J_PUBLIC_GIT_UNAVAILABLE") == "",
        "commits_after_reviewed_implementation_sha": int(count_text),
    }


def _validate_reviewed_source_snapshot_support_implementation_binding(
    repository_root: Path,
    reviewed_source_snapshot_support_implementation_sha: str,
    *,
    runtime_state_reader: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> str:
    reviewed_sha = _require_hex(
        reviewed_source_snapshot_support_implementation_sha, 40, "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_SHA_MALFORMED"
    )
    runtime = (runtime_state_reader or _default_reviewed_source_snapshot_support_runtime_state)(
        repository_root, reviewed_sha
    )
    if not isinstance(runtime, Mapping) or set(runtime) != _REVIEWED_IMPLEMENTATION_RUNTIME_FIELDS:
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_RUNTIME_SCHEMA_INVALID")
    if (
        runtime["head"] != reviewed_sha
        or runtime["authoritative_remote_head"] != reviewed_sha
        or runtime["worktree_clean"] is not True
        or type(runtime["commits_after_reviewed_implementation_sha"]) is not int
        or runtime["commits_after_reviewed_implementation_sha"] != 0
    ):
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_RUNTIME_BINDING_INVALID")
    return reviewed_sha


# ---------------------------------------------------------------------------
# Frozen V8J pre-gate environment readiness (design §3)
# ---------------------------------------------------------------------------

_ENVIRONMENT_PREFLIGHT_FIELDS = frozenset(
    {
        "canonical_interpreter",
        "checker_exit_code",
        "REAL_EXECUTION_ENVIRONMENT_READY",
        "ENVIRONMENT_LOCK_CHECK",
        "ENVIRONMENT_FREEZE_CHECK",
        "ENVIRONMENT_FREEZE_EVIDENCE_GIT_SHA256_MATCH",
        "ENVIRONMENT_LOCK_FINGERPRINT_STATUS",
        "REAL_EXECUTION_ENVIRONMENT_FROZEN",
        "REAL_NETWORK_REQUESTS",
        "PRIVATE_READS",
        "GATES_CONSUMED",
    }
)


def _canonical_interpreter_path(repository_root: Path) -> Path:
    return (repository_root / ".venv-real-execution" / "Scripts" / "python.exe").resolve(strict=False)


def _validate_environment_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete, public-safe V8J environment prerequisite.

    This intentionally accepts no partial or extra state: every required
    condition must be mechanically known before the V8J receipt can exist.
    """
    if not isinstance(preflight, Mapping) or set(preflight) != _ENVIRONMENT_PREFLIGHT_FIELDS:
        raise V8JSourceSnapshotBlocked("V8J_ENVIRONMENT_PREFLIGHT_SCHEMA_INVALID")
    expected = {
        "canonical_interpreter": V8J_CANONICAL_INTERPRETER_RELATIVE_PATH,
        "checker_exit_code": 0,
        "REAL_EXECUTION_ENVIRONMENT_READY": True,
        "ENVIRONMENT_LOCK_CHECK": "PASS",
        "ENVIRONMENT_FREEZE_CHECK": "PASS",
        "ENVIRONMENT_FREEZE_EVIDENCE_GIT_SHA256_MATCH": True,
        "ENVIRONMENT_LOCK_FINGERPRINT_STATUS": "FROZEN",
        "REAL_EXECUTION_ENVIRONMENT_FROZEN": True,
        "REAL_NETWORK_REQUESTS": 0,
        "PRIVATE_READS": 0,
        "GATES_CONSUMED": 0,
    }
    for key, value in expected.items():
        if preflight[key] != value or type(preflight[key]) is not type(value):
            raise V8JSourceSnapshotBlocked("V8J_ENVIRONMENT_PRE_GATE_BLOCK")
    return dict(preflight)


def _verify_environment_critical_git_blobs(repository_root: Path) -> None:
    """Require Git-object (never working-tree-byte) equality to the frozen
    environment promotion baseline for every V8J-critical path."""
    current_commit = _git_text(repository_root, ["rev-parse", "HEAD"], "V8J_ENVIRONMENT_GIT_PROVENANCE_UNAVAILABLE")
    _require_hex(current_commit, 40, "V8J_ENVIRONMENT_GIT_PROVENANCE_UNAVAILABLE")
    for git_path in V8J_ENVIRONMENT_CRITICAL_GIT_PATHS:
        try:
            baseline_blob = resolve_git_blob(repository_root, V8J_ENVIRONMENT_FREEZE_PROMOTION_COMMIT, git_path)
            current_blob = resolve_git_blob(repository_root, current_commit, git_path)
        except V8CGitProvenanceBlocked as error:
            raise V8JSourceSnapshotBlocked("V8J_ENVIRONMENT_GIT_PROVENANCE_UNAVAILABLE") from error
        if baseline_blob != current_blob:
            raise V8JSourceSnapshotBlocked("V8J_ENVIRONMENT_CRITICAL_BLOB_MISMATCH")


def _default_environment_preflight(repository_root: Path) -> dict[str, Any]:
    canonical_interpreter = _canonical_interpreter_path(repository_root)
    try:
        running_interpreter = Path(sys.executable).resolve(strict=False)
    except OSError as error:
        raise V8JSourceSnapshotBlocked("V8J_CANONICAL_INTERPRETER_MISMATCH") from error
    if running_interpreter != canonical_interpreter or not canonical_interpreter.is_file():
        raise V8JSourceSnapshotBlocked("V8J_CANONICAL_INTERPRETER_MISMATCH")
    checker = repository_root / V8J_ENVIRONMENT_CHECKER_GIT_PATH
    try:
        result = subprocess.run(
            [str(canonical_interpreter), str(checker)],
            cwd=str(repository_root),
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8JSourceSnapshotBlocked("V8J_ENVIRONMENT_CHECKER_UNAVAILABLE") from error
    try:
        payload = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8JSourceSnapshotBlocked("V8J_ENVIRONMENT_CHECKER_INVALID_JSON") from error
    if not isinstance(payload, Mapping):
        raise V8JSourceSnapshotBlocked("V8J_ENVIRONMENT_CHECKER_INVALID_JSON")
    values: dict[str, Any] = {
        "canonical_interpreter": V8J_CANONICAL_INTERPRETER_RELATIVE_PATH,
        "checker_exit_code": result.returncode,
    }
    for field in _ENVIRONMENT_PREFLIGHT_FIELDS - set(values):
        if field not in payload:
            raise V8JSourceSnapshotBlocked("V8J_ENVIRONMENT_CHECKER_REQUIRED_FIELD_MISSING")
        values[field] = payload[field]
    return _validate_environment_preflight(values)


def _require_frozen_environment_pre_gate(
    repository_root: Path,
    *,
    environment_preflight: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _verify_environment_critical_git_blobs(repository_root)
    return _validate_environment_preflight((environment_preflight or _default_environment_preflight)(repository_root))


CANONICAL_JPX_SOURCE_MODULE = "scripts.build_v8_partition_manifest"
CANONICAL_JPX_SOURCE_GIT_PATH = "scripts/build_v8_partition_manifest.py"


def _require_canonical_post_gate_callable_binding(
    *,
    jpx_fetcher: Callable[..., Any],
    parse_source_table: Callable[..., Any],
    clock: Callable[[], datetime],
    repository_root: Path,
    reviewed_source_snapshot_support_implementation_sha: str,
) -> dict[str, str]:
    """Bind the only production post-gate callables before receipt publication.

    Internal test DI is deliberately outside this function. The public entry
    point invokes it unconditionally, so arbitrary callbacks cannot convert
    dependency readiness into a pre-gate affirmative result.
    """
    try:
        from scripts.build_v8_partition_manifest import default_parse_source_table, fetch_real_jpx_source
    except ImportError as error:
        raise V8JSourceSnapshotBlocked("V8J_CANONICAL_CALLABLE_IMPORT_UNAVAILABLE") from error
    if parse_source_table is not default_parse_source_table:
        raise V8JSourceSnapshotBlocked("V8J_CANONICAL_PARSER_BINDING_INVALID")
    if jpx_fetcher is not fetch_real_jpx_source:
        raise V8JSourceSnapshotBlocked("V8J_CANONICAL_FETCHER_BINDING_INVALID")
    if clock is not _utc_clock:
        raise V8JSourceSnapshotBlocked("V8J_CANONICAL_CLOCK_BINDING_INVALID")

    canonical_path = (repository_root / CANONICAL_JPX_SOURCE_GIT_PATH).resolve(strict=False)
    for callable_value, expected_name in (
        (default_parse_source_table, "default_parse_source_table"),
        (fetch_real_jpx_source, "fetch_real_jpx_source"),
    ):
        if (
            getattr(callable_value, "__module__", None) != CANONICAL_JPX_SOURCE_MODULE
            or getattr(callable_value, "__name__", None) != expected_name
        ):
            raise V8JSourceSnapshotBlocked("V8J_CANONICAL_CALLABLE_PROVENANCE_INVALID")
        try:
            source_path = Path(inspect.getsourcefile(callable_value) or "").resolve(strict=False)
        except (OSError, TypeError) as error:
            raise V8JSourceSnapshotBlocked("V8J_CANONICAL_CALLABLE_PROVENANCE_INVALID") from error
        if source_path != canonical_path:
            raise V8JSourceSnapshotBlocked("V8J_CANONICAL_CALLABLE_PROVENANCE_INVALID")

    reviewed_sha = _require_hex(
        reviewed_source_snapshot_support_implementation_sha,
        40,
        "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_SHA_MALFORMED",
    )
    try:
        source_blob = resolve_git_blob(repository_root, reviewed_sha, CANONICAL_JPX_SOURCE_GIT_PATH)
        head = _git_text(repository_root, ["rev-parse", "HEAD"], "V8J_CANONICAL_CALLABLE_GIT_PROVENANCE_UNAVAILABLE")
        head_blob = resolve_git_blob(repository_root, head, CANONICAL_JPX_SOURCE_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise V8JSourceSnapshotBlocked("V8J_CANONICAL_CALLABLE_GIT_PROVENANCE_UNAVAILABLE") from error
    if source_blob != head_blob:
        raise V8JSourceSnapshotBlocked("V8J_CANONICAL_CALLABLE_GIT_PROVENANCE_MISMATCH")
    return {
        "CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE": "YES"
    }


# ---------------------------------------------------------------------------
# Full DI execution boundary
# ---------------------------------------------------------------------------


def _require_safe_external_path(value: str | os.PathLike[str], repository_root: Path, reason: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise V8JSourceSnapshotBlocked(reason)
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(Path(repository_root).resolve(strict=False))
    except ValueError:
        return resolved
    raise V8JSourceSnapshotBlocked(reason)


def _execute_source_snapshot_acquisition_with_dependencies(
    *,
    authorization_identity: str,
    gate_state_root: str | os.PathLike[str],
    private_state_root: str | os.PathLike[str],
    evidence_output_path: str | os.PathLike[str],
    jpx_fetcher: Callable[[], Any],
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    repository_root: Path,
    public_preflight: Callable[[], Mapping[str, Any]],
    gate_consumer: Callable[..., Mapping[str, Any]],
    clock: Callable[[], datetime],
    reviewed_source_snapshot_support_implementation_sha: str,
    runtime_state_reader: Callable[[Path, str], Mapping[str, Any]] | None = None,
    environment_preflight: Callable[[Path], Mapping[str, Any]] | None = None,
    require_canonical_post_gate_callables: bool = False,
    reviewed_v8j_design_candidate_commit: str = REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
    block_size: int = BLOCK_SIZE,
    minimum_fresh_eligible_count: int = MINIMUM_FRESH_ELIGIBLE_COUNT,
) -> dict[str, Any]:
    """DI-only future execution boundary; never called with a real network
    `jpx_fetcher` by this implementation task. `jpx_fetcher` is invoked
    exactly once, strictly after the gate's durable receipt is published
    (matching `IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST` literally) -- this is
    the one authorized official-JPX snapshot request. Any failure from
    that instant onward is terminal (design §6) for this V8J
    source-snapshot attempt: no retry, no second request, no receipt
    reset/deletion, no provider/date substitution.
    """
    _validate_public_preflight(public_preflight())
    reviewed_impl_sha = _validate_reviewed_source_snapshot_support_implementation_binding(
        repository_root, reviewed_source_snapshot_support_implementation_sha, runtime_state_reader=runtime_state_reader
    )
    _require_frozen_environment_pre_gate(repository_root, environment_preflight=environment_preflight)
    if require_canonical_post_gate_callables:
        _require_canonical_post_gate_callable_binding(
            jpx_fetcher=jpx_fetcher,
            parse_source_table=parse_source_table,
            clock=clock,
            repository_root=repository_root,
            reviewed_source_snapshot_support_implementation_sha=reviewed_impl_sha,
        )
    validate_authorization_identity(
        authorization_identity,
        reviewed_source_snapshot_support_implementation_sha=reviewed_impl_sha,
        reviewed_v8j_design_candidate_commit=reviewed_v8j_design_candidate_commit,
    )

    gate_state = _require_safe_external_path(gate_state_root, repository_root, "V8J_GATE_STATE_PATH_INVALID")
    private_state = _require_safe_external_path(private_state_root, repository_root, "V8J_PRIVATE_STATE_PATH_INVALID")
    evidence_output = _require_safe_external_path(evidence_output_path, repository_root, "V8J_OUTPUT_PATH_INVALID")
    if evidence_output.exists():
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_EVIDENCE_ALREADY_EXISTS")
    try:
        gate_state.mkdir(parents=True, exist_ok=True)
        private_state.mkdir(parents=True, exist_ok=True)
        evidence_output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8JSourceSnapshotBlocked("V8J_OUTPUT_OR_STATE_PREPARATION_FAILED") from error
    if (gate_state / (compute_source_snapshot_gate_receipt_key() + ".json")).exists():
        raise V8JSourceSnapshotBlocked("V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED")

    # Exact frozen boundary: no JPX byte is fetched before the durable receipt.
    gate_consumer(
        gate_state,
        authorization_identity,
        clock=clock,
        reviewed_v8j_design_candidate_commit=reviewed_v8j_design_candidate_commit,
        reviewed_source_snapshot_support_implementation_sha=reviewed_impl_sha,
    )

    # The one authorized official-JPX snapshot request. Called exactly once.
    raw_bytes = jpx_fetcher()
    if require_canonical_post_gate_callables:
        if (
            not isinstance(raw_bytes, tuple)
            or len(raw_bytes) != 2
            or not isinstance(raw_bytes[0], (bytes, bytearray))
            or not isinstance(raw_bytes[1], str)
        ):
            raise V8JSourceSnapshotBlocked("V8J_CANONICAL_FETCHER_RESULT_INVALID")
        raw_bytes = raw_bytes[0]
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise V8JSourceSnapshotBlocked("V8J_RAW_SOURCE_BYTES_INVALID")
    raw_bytes = bytes(raw_bytes)
    fetch_utc = clock()

    acquisition_result = _perform_source_snapshot_acquisition(
        raw_source_bytes=raw_bytes,
        parse_source_table=parse_source_table,
        v4_manifest_path=v4_manifest_path,
        v4_universe_csv_path=v4_universe_csv_path,
        source_acquisition_utc=fetch_utc,
        block_size=block_size,
        minimum_fresh_eligible_count=minimum_fresh_eligible_count,
    )

    preserve_raw_source_bytes_once(private_state, acquisition_result["source_raw_sha256"], raw_bytes)

    # Post-gate, pre-evidence-publication: the durable receipt must be
    # semantically bound to this exact execution's authorized values.
    _receipt, receipt_bytes_sha = _read_and_bind_gate_receipt(
        gate_state,
        reviewed_v8j_design_candidate_commit=reviewed_v8j_design_candidate_commit,
        reviewed_source_snapshot_support_implementation_sha=reviewed_impl_sha,
        authorization_identity=authorization_identity,
    )
    receipt_key = compute_source_snapshot_gate_receipt_key()
    evidence = _build_source_snapshot_evidence(
        reviewed_v8j_design_candidate_commit=reviewed_v8j_design_candidate_commit,
        reviewed_source_snapshot_support_implementation_sha=reviewed_impl_sha,
        source_snapshot_gate_receipt_key_sha256_value=receipt_key,
        source_snapshot_gate_receipt_bytes_sha256_value=receipt_bytes_sha,
        acquisition_result=acquisition_result,
    )
    _publish_source_snapshot_evidence(evidence, evidence_output)
    return {
        "result": "PASS",
        "source_snapshot_gate_receipt_key_sha256": receipt_key,
        "eligible_ticker_count": acquisition_result["eligible_ticker_count"],
        "fresh_eligible_count": acquisition_result["fresh_eligible_count"],
        "evidence_written": True,
    }


def resolve_and_acquire_source_snapshot(
    authorization_identity: str,
    *,
    reviewed_source_snapshot_support_implementation_sha: str,
    jpx_fetcher: Callable[[], Any],
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    evidence_output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Prepared future entry point; not executed by this support task.

    `jpx_fetcher` and `parse_source_table` are required, non-defaulted
    keyword arguments -- there is no wired default real-network production
    fetcher in this module. This implementation task grants zero real JPX
    request authority; a future task wires and reviews the real fetcher
    separately.
    """
    return _execute_source_snapshot_acquisition_with_dependencies(
        authorization_identity=authorization_identity,
        gate_state_root=CANONICAL_V8J_SOURCE_SNAPSHOT_GATE_STATE_ROOT,
        private_state_root=CANONICAL_V8J_SOURCE_SNAPSHOT_PRIVATE_STATE_ROOT,
        evidence_output_path=evidence_output_path,
        jpx_fetcher=jpx_fetcher,
        parse_source_table=parse_source_table,
        v4_manifest_path=v4_manifest_path,
        v4_universe_csv_path=v4_universe_csv_path,
        repository_root=CANONICAL_REPOSITORY_ROOT,
        public_preflight=lambda: _default_public_preflight(CANONICAL_REPOSITORY_ROOT),
        gate_consumer=consume_gate_once,
        clock=_utc_clock,
        reviewed_source_snapshot_support_implementation_sha=reviewed_source_snapshot_support_implementation_sha,
        require_canonical_post_gate_callables=True,
    )


__all__ = [
    "BLOCK_SIZE",
    "CANONICAL_V8J_SOURCE_SNAPSHOT_GATE_STATE_ROOT",
    "CANONICAL_V8J_SOURCE_SNAPSHOT_PRIVATE_STATE_ROOT",
    "MINIMUM_FRESH_ELIGIBLE_COUNT",
    "REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT",
    "REVIEWED_V8J_FREEZE_RECORD_COMMIT",
    "V8J_AUTHORITATIVE_BRANCH",
    "V8J_DESIGN_CANDIDATE_BLOB_SHA",
    "V8J_FREEZE_APPROVAL_BLOB_SHA",
    "V8J_SOURCE_SNAPSHOT_EVIDENCE_FIELDS",
    "V8J_SOURCE_SNAPSHOT_GATE",
    "V8J_SOURCE_SNAPSHOT_GATE_CONSUMPTION_BOUNDARY",
    "V8J_SOURCE_SNAPSHOT_RECEIPT_FIELDS",
    "V8J_STUDY_NAME",
    "V8JSourceSnapshotBlocked",
    "authorization_identity_sha256",
    "build_authorization_identity",
    "canonical_sha256",
    "compute_source_snapshot_gate_receipt_key",
    "consume_gate_once",
    "gate_receipt_bytes_sha256",
    "preserve_raw_source_bytes_once",
    "read_gate_receipt",
    "resolve_and_acquire_source_snapshot",
    "validate_authorization_identity",
    "verify_source_snapshot_evidence_binding",
]
