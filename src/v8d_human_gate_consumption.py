"""Durable, fail-closed, one-shot consumption receipts for V8D's four
Yahoo-request-bearing production human gates.

`V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md`. Mirrors
`src.v8c_human_gate_consumption`'s proven durable-receipt pattern (an
fsync'd, atomically-created-never-overwritten JSON receipt on durable local
storage, keyed by a SHA-256 over safe, already-hashed components only), but
with V8D-specific constants, an exact 12-field receipt schema, and simpler
one-shot semantics: **every** V8D gate below is one-shot for the life of the
frozen V8D design commit -- unlike V8C's readiness gates, there is no
per-authorization-identity exception. A fresh authorization identity must
never reset an already-consumed V8D stage; the deterministic receipt key is
therefore derived from ``(repository, gate, v8d_frozen_design_commit)``
alone and deliberately excludes both the authorization identity and the
reviewed implementation commit (see `compute_receipt_key`).

``consume_gate_and_bind`` is the sole function that ever creates a receipt.
It is fail-closed: an existing receipt for the exact same key blocks a
second consumption. There is deliberately no deletion/reset API -- a
consumed gate stays consumed for the life of the frozen design commit.

This module performs no Git access, no network access, and never reads or
writes a ticker identity, private path, raw URL/payload/price, or raw
exception message. The raw human authorization identity is used only
transiently (to compute its SHA-256) and is never persisted. Importing this
module performs no state-ledger read/write; production root resolution uses
only the fixed machine-local OS location described below (never
HOME/USERPROFILE).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

STUDY_NAME = "V8D_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8D_HUMAN_GATE_CONSUMPTION_RECEIPT_V1"
REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
CONSUMPTION_BOUNDARY = "IMMEDIATELY_BEFORE_FIRST_YAHOO_REQUEST"

GATE_T1C_TRANSPORT_READINESS = "T1C_TRANSPORT_READINESS_HUMAN_GATE"
GATE_T1C_RAW_ACQUISITION = "T1C_RAW_ACQUISITION_HUMAN_GATE"
GATE_T2_TRANSPORT_READINESS = "T2_TRANSPORT_READINESS_HUMAN_GATE"
GATE_T2_RAW_ACQUISITION = "T2_RAW_ACQUISITION_HUMAN_GATE"

# The exact, sole valid stage -> gate mapping (frozen design's four
# Yahoo-request-bearing production stages). No other stage/gate combination
# is valid; a receipt or binding whose (logical_stage, gate) pair disagrees
# with this mapping fails closed.
STAGE_GATE: dict[str, str] = {
    "T1C_TRANSPORT_READINESS": GATE_T1C_TRANSPORT_READINESS,
    "T1C_RAW_ACQUISITION": GATE_T1C_RAW_ACQUISITION,
    "T2_TRANSPORT_READINESS": GATE_T2_TRANSPORT_READINESS,
    "T2_RAW_ACQUISITION": GATE_T2_RAW_ACQUISITION,
}
GATE_STAGE: dict[str, str] = {gate: stage for stage, gate in STAGE_GATE.items()}
KNOWN_STAGES: tuple[str, ...] = tuple(STAGE_GATE.keys())
KNOWN_GATES: tuple[str, ...] = tuple(STAGE_GATE.values())

CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_POSIX_MACHINE_STATE_BASE = Path("/var/lib/stock-analyzer")


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


_FOLDERID_PROGRAM_DATA = _GUID(
    0x62AB5D82,
    0xFDC1,
    0x4DC3,
    (ctypes.c_ubyte * 8)(0xA9, 0xDD, 0x07, 0x0D, 0x1D, 0x49, 0x5D, 0x97),
)


def _resolve_windows_program_data_directory() -> Path:
    try:
        path_ptr = ctypes.c_wchar_p()
        result = ctypes.windll.shell32.SHGetKnownFolderPath(
            ctypes.byref(_FOLDERID_PROGRAM_DATA), 0, None, ctypes.byref(path_ptr)
        )
        if result != 0 or not path_ptr.value:
            raise RuntimeError
        path = Path(path_ptr.value)
        ctypes.windll.ole32.CoTaskMemFree(path_ptr)
        return path
    except (AttributeError, OSError, TypeError, RuntimeError) as error:
        raise RuntimeError("V8D_HUMAN_GATE_STATE_ROOT_UNAVAILABLE") from error


def _default_consumption_state_root() -> Path:
    if os.name == "nt":
        base = _resolve_windows_program_data_directory()
        return base / "stock-analyzer" / "v8d-human-gate-state"
    return _POSIX_MACHINE_STATE_BASE / "v8d-human-gate-state"


CANONICAL_CONSUMPTION_STATE_ROOT = _default_consumption_state_root()


class V8DHumanGateConsumptionBlocked(RuntimeError):
    """Fail-closed durable one-shot V8D human-gate consumption error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_known_stage(value: object) -> str:
    if value not in KNOWN_STAGES:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_STAGE_UNKNOWN")
    return value


def _require_known_gate(value: object) -> str:
    if value not in KNOWN_GATES:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_UNKNOWN")
    return value


def _require_stage_gate_pair(stage: object, gate: object) -> tuple[str, str]:
    stage = _require_known_stage(stage)
    gate = _require_known_gate(gate)
    if STAGE_GATE[stage] != gate:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_STAGE_GATE_MISMATCH")
    return stage, gate


def _require_git_commit(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8DHumanGateConsumptionBlocked(reason)
    return value


def _require_hex64(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V8DHumanGateConsumptionBlocked(reason)
    return value


def _identity_sha256(raw_authorization_identity: str) -> str:
    """The one-way hash of a raw human authorization identity. This is the
    ONLY form of the identity ever persisted -- the raw identity itself is
    used transiently, as a function argument, and never written to durable
    storage or returned to any caller."""
    return hashlib.sha256(raw_authorization_identity.encode("utf-8")).hexdigest()


def compute_receipt_key(gate: str, v8d_frozen_design_commit: str) -> str:
    """Deterministic receipt key from only safe components: repository,
    gate, and the exact frozen V8D design commit -- deliberately excluding
    both the authorization identity and the reviewed implementation commit
    (see module docstring), so neither a fresh authorization nor a later
    implementation re-review can reset an already-consumed one-shot gate."""
    _require_known_gate(gate)
    _require_git_commit(v8d_frozen_design_commit, "V8D_HUMAN_GATE_DESIGN_COMMIT_INVALID")
    material = REPOSITORY_IDENTITY + "|" + gate + "|" + v8d_frozen_design_commit
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _receipt_path(state_root: str | os.PathLike[str], gate: str, v8d_frozen_design_commit: str) -> Path:
    return Path(state_root) / (compute_receipt_key(gate, v8d_frozen_design_commit) + ".json")


RECEIPT_FIELDS = (
    "schema_version",
    "study",
    "repository",
    "gate",
    "logical_stage",
    "v8d_frozen_design_commit",
    "reviewed_production_implementation_commit",
    "authorization_identity_sha256",
    "consumed",
    "consumption_count",
    "consumption_boundary",
    "consumed_at_utc",
)


@dataclass(frozen=True)
class V8DGateReceiptBinding:
    """A validated safe binding, derived only from an actual durable
    receipt successfully created and independently read back from disk --
    never from a caller's own claims. Contains no raw authorization
    identity. Production entrypoints obtain this only from
    `consume_gate_and_bind`."""

    human_gate: str
    logical_stage: str
    gate_receipt_key_sha256: str
    gate_receipt_bytes_sha256: str
    authorization_identity_sha256: str
    reviewed_production_implementation_commit: str


def has_gate_been_consumed(state_root: str | os.PathLike[str], gate: str, v8d_frozen_design_commit: str) -> bool:
    """Read-only existence check -- never raises for "not yet consumed";
    only raises for a malformed ``gate``/``v8d_frozen_design_commit`` or an
    unreadable state root."""
    path = _receipt_path(state_root, gate, v8d_frozen_design_commit)
    try:
        return path.exists()
    except OSError as error:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_STATE_UNAVAILABLE") from error


def require_gate_not_yet_consumed(state_root: str | os.PathLike[str], gate: str, v8d_frozen_design_commit: str) -> None:
    """Fail-fast, read-only pre-check: BLOCK immediately if ``gate`` (under
    this exact key) has already been durably consumed."""
    if has_gate_been_consumed(state_root, gate, v8d_frozen_design_commit):
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_ALREADY_CONSUMED:" + gate)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_DUPLICATE_KEY")
        result[key] = value
    return result


def _parse_and_validate_receipt_bytes(raw: bytes) -> tuple[dict[str, Any], str]:
    """Strictly parse and validate a receipt's exact schema/field semantics
    from its raw durable bytes, and return the independently recomputed
    deterministic key derivable purely from the receipt's own validated
    safe content. Never trusts the filename it was read from."""
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_INVALID_JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != set(RECEIPT_FIELDS):
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_SCHEMA_INVALID")
    if parsed["schema_version"] != SCHEMA_VERSION:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_SCHEMA_VERSION_MISMATCH")
    if parsed["study"] != STUDY_NAME:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_STUDY_MISMATCH")
    if parsed["repository"] != REPOSITORY_IDENTITY:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_REPOSITORY_MISMATCH")
    _require_stage_gate_pair(parsed["logical_stage"], parsed["gate"])
    _require_git_commit(parsed["v8d_frozen_design_commit"], "V8D_HUMAN_GATE_RECEIPT_DESIGN_COMMIT_INVALID")
    _require_git_commit(
        parsed["reviewed_production_implementation_commit"], "V8D_HUMAN_GATE_RECEIPT_IMPLEMENTATION_COMMIT_INVALID"
    )
    _require_hex64(parsed["authorization_identity_sha256"], "V8D_HUMAN_GATE_RECEIPT_AUTHORIZATION_HASH_INVALID")
    if parsed["consumed"] is not True:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_CONSUMED_FLAG_INVALID")
    if type(parsed["consumption_count"]) is not int or parsed["consumption_count"] != 1:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_CONSUMPTION_COUNT_INVALID")
    if parsed["consumption_boundary"] != CONSUMPTION_BOUNDARY:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_CONSUMPTION_BOUNDARY_INVALID")
    _require_canonical_utc_timestamp(parsed["consumed_at_utc"])

    recomputed_key = compute_receipt_key(parsed["gate"], parsed["v8d_frozen_design_commit"])
    return dict(parsed), recomputed_key


def _read_and_validate_receipt(state_root: str | os.PathLike[str], receipt_key: str) -> tuple[dict[str, Any], bytes]:
    if not isinstance(receipt_key, str) or len(receipt_key) != 64 or any(
        char not in "0123456789abcdef" for char in receipt_key
    ):
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_KEY_INVALID")
    path = Path(state_root) / (receipt_key + ".json")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_MISSING") from error
    parsed, recomputed_key = _parse_and_validate_receipt_bytes(raw)
    # Not merely internally self-consistent field-by-field: the requested
    # ``receipt_key`` (the filename this receipt was located at) must be
    # exactly the canonical key the receipt's OWN safe content derives. A
    # well-formed receipt copied to -- or fabricated at -- an arbitrary/
    # wrong 64-hex filename fails this recomputation and BLOCKs.
    if recomputed_key != receipt_key:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_KEY_CONTENT_MISMATCH")
    return parsed, raw


def read_gate_consumption_receipt(
    state_root: str | os.PathLike[str],
    receipt_key: str,
    *,
    expected_gate: str | None = None,
    expected_v8d_frozen_design_commit: str | None = None,
) -> dict[str, Any]:
    """Read-only: mechanically read and strictly validate the exact durable
    gate-consumption receipt located at ``receipt_key``. Fails closed on a
    missing, malformed, duplicate-keyed, schema-invalid, or wrong-key
    receipt."""
    parsed, _raw = _read_and_validate_receipt(state_root, receipt_key)
    if expected_gate is not None and parsed["gate"] != expected_gate:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_GATE_MISMATCH")
    if expected_v8d_frozen_design_commit is not None and parsed["v8d_frozen_design_commit"] != expected_v8d_frozen_design_commit:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_DESIGN_COMMIT_MISMATCH")
    return parsed


def read_gate_consumption_receipt_with_bytes_hash(
    state_root: str | os.PathLike[str],
    receipt_key: str,
    *,
    expected_gate: str | None = None,
    expected_v8d_frozen_design_commit: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Like `read_gate_consumption_receipt`, but also returns the SHA-256
    over the exact durable raw receipt bytes actually read -- computed from
    the very same bytes that were parsed (no separate re-read), so the
    result is never subject to a time-of-check/time-of-use mismatch."""
    parsed, raw = _read_and_validate_receipt(state_root, receipt_key)
    if expected_gate is not None and parsed["gate"] != expected_gate:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_GATE_MISMATCH")
    if expected_v8d_frozen_design_commit is not None and parsed["v8d_frozen_design_commit"] != expected_v8d_frozen_design_commit:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_DESIGN_COMMIT_MISMATCH")
    return parsed, hashlib.sha256(raw).hexdigest()


def _utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_CLOCK_INVALID")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


_CANONICAL_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{6})?Z$"
)


def _require_canonical_utc_timestamp(value: object) -> str:
    if not isinstance(value, str) or not _CANONICAL_UTC_TIMESTAMP.fullmatch(value):
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_TIMESTAMP_INVALID")
    try:
        format_string = "%Y-%m-%dT%H:%M:%S.%fZ" if "." in value else "%Y-%m-%dT%H:%M:%SZ"
        parsed = datetime.strptime(value, format_string).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_TIMESTAMP_INVALID") from error
    if _timestamp_text(parsed) != value:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_TIMESTAMP_INVALID")
    return value


def consume_gate_and_bind(
    state_root: str | os.PathLike[str],
    *,
    logical_stage: str,
    v8d_frozen_design_commit: str,
    reviewed_production_implementation_commit: str,
    raw_authorization_identity: str,
    clock: Callable[[], datetime],
) -> V8DGateReceiptBinding:
    """The sole function that durably, atomically, fail-closed consumes a
    V8D human gate exactly once. The caller must invoke this exactly once,
    strictly before the first real Yahoo request it authorizes, and must
    treat any exception here as "not consumed" (no request may proceed).

    Uses exclusive-creation (``os.link``, never overwriting an existing
    receipt), fsync's the receipt before publishing it, and -- crucially --
    never returns a binding built from the in-memory receipt dict just
    written: it independently re-reads the actual durable bytes back from
    disk and re-validates them (`read_gate_consumption_receipt_with_bytes_
    hash`) before returning the binding, so any interference on the write
    path (a filesystem quirk, a mid-flight tamper) cannot silently
    substitute an incorrect binding for the real evidence.

    There is deliberately no deletion/reset API: a consumed gate stays
    consumed for the life of the frozen design commit, and a fresh
    ``raw_authorization_identity`` cannot bypass an existing receipt --
    `compute_receipt_key` never includes the authorization identity.
    """
    stage = _require_known_stage(logical_stage)
    gate = STAGE_GATE[stage]
    _require_git_commit(v8d_frozen_design_commit, "V8D_HUMAN_GATE_DESIGN_COMMIT_INVALID")
    _require_git_commit(
        reviewed_production_implementation_commit, "V8D_HUMAN_GATE_IMPLEMENTATION_COMMIT_INVALID"
    )
    if not isinstance(raw_authorization_identity, str) or not raw_authorization_identity:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_AUTHORIZATION_IDENTITY_REQUIRED")

    path = _receipt_path(state_root, gate, v8d_frozen_design_commit)
    if path.exists():
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_ALREADY_CONSUMED:" + gate)

    identity_hash = _identity_sha256(raw_authorization_identity)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY_NAME,
        "repository": REPOSITORY_IDENTITY,
        "gate": gate,
        "logical_stage": stage,
        "v8d_frozen_design_commit": v8d_frozen_design_commit,
        "reviewed_production_implementation_commit": reviewed_production_implementation_commit,
        "authorization_identity_sha256": identity_hash,
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": CONSUMPTION_BOUNDARY,
        "consumed_at_utc": _timestamp_text(_utc_timestamp(clock() if callable(clock) else clock)),
    }
    payload = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_STATE_UNAVAILABLE") from error

    staging = path.parent / (path.name + ".staging-" + os.urandom(8).hex())
    try:
        try:
            with open(staging, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_STATE_WRITE_FAILED") from error
        try:
            os.link(str(staging), str(path))
        except FileExistsError as error:
            raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_ALREADY_CONSUMED:" + gate) from error
        except OSError as error:
            raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_STATE_WRITE_FAILED") from error
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass

    receipt_key = compute_receipt_key(gate, v8d_frozen_design_commit)
    validated, bytes_hash = read_gate_consumption_receipt_with_bytes_hash(
        state_root, receipt_key, expected_gate=gate, expected_v8d_frozen_design_commit=v8d_frozen_design_commit,
    )
    if validated["logical_stage"] != stage:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_STAGE_MISMATCH")
    if validated["reviewed_production_implementation_commit"] != reviewed_production_implementation_commit:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_IMPLEMENTATION_MISMATCH")
    if validated["authorization_identity_sha256"] != identity_hash:
        raise V8DHumanGateConsumptionBlocked("V8D_HUMAN_GATE_RECEIPT_AUTHORIZATION_MISMATCH")

    return V8DGateReceiptBinding(
        human_gate=gate,
        logical_stage=stage,
        gate_receipt_key_sha256=receipt_key,
        gate_receipt_bytes_sha256=bytes_hash,
        authorization_identity_sha256=validated["authorization_identity_sha256"],
        reviewed_production_implementation_commit=validated["reviewed_production_implementation_commit"],
    )


__all__ = [
    "CANONICAL_CONSUMPTION_STATE_ROOT",
    "CANONICAL_REPOSITORY_ROOT",
    "CONSUMPTION_BOUNDARY",
    "GATE_STAGE",
    "GATE_T1C_RAW_ACQUISITION",
    "GATE_T1C_TRANSPORT_READINESS",
    "GATE_T2_RAW_ACQUISITION",
    "GATE_T2_TRANSPORT_READINESS",
    "KNOWN_GATES",
    "KNOWN_STAGES",
    "RECEIPT_FIELDS",
    "REPOSITORY_IDENTITY",
    "SCHEMA_VERSION",
    "STAGE_GATE",
    "STUDY_NAME",
    "V8DGateReceiptBinding",
    "V8DHumanGateConsumptionBlocked",
    "compute_receipt_key",
    "consume_gate_and_bind",
    "has_gate_been_consumed",
    "read_gate_consumption_receipt",
    "read_gate_consumption_receipt_with_bytes_hash",
    "require_gate_not_yet_consumed",
]
