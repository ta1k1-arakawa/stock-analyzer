"""Durable, fail-closed, one-shot consumption receipts for V8C's named
one-time human gates (§6, §12).

Mirrors `src.v8b_human_gate_consumption`'s proven pattern: a "consumption"
is a small, fsync'd, atomically-created (never overwritten) receipt file on
durable local storage, keyed by the exact repository identity, gate name,
and the exact frozen V8C design commit -- never by anything ticker-, date-,
or path-derived, so a receipt itself carries no private information.
``require_gate_not_yet_consumed`` is a read-only, non-mutating early
fail-fast check; ``consume_gate_once`` is the sole function that ever
creates a receipt, and it is fail-closed: an existing receipt for the exact
same key blocks a second consumption, whether from an earlier call in this
process, a previous process, a previous restart, or a different checkout of
this same repository on the same machine. There is deliberately no
deletion/reset API: a consumed gate stays consumed.

**Readiness gates are per-authorization, not forever-one-shot.**
`V8C_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §3/§4.1 explicitly requires that
"[a] prior readiness authorization never authorizes a later probe" while
also requiring that "[a] blocked readiness result may be rechecked ... only
after a new explicit readiness authorization" -- i.e. readiness gates must
support being consumed again under a *fresh* human authorization, unlike
every other V8C one-shot gate (allocation, pin, raw acquisition, research
opening, T2 authority bridge), which -- exactly like every V8B one-shot
gate -- may only ever be consumed once, permanently, for the life of the
frozen design commit. This module models that distinction by keying a
readiness gate's receipt on ``(repository, gate, design_commit,
authorization_identity)`` -- the exact human authorization token/identity
string, supplied by the caller and never invented here -- so replaying the
*same* authorization token BLOCKs, while a genuinely new, distinct
authorization identity is a fresh key and may be consumed. Every other gate
is keyed on ``(repository, gate, design_commit)`` alone, exactly like
`src.v8b_human_gate_consumption`, and rejects any caller-supplied
``authorization_identity``.

This module performs no Git access, no network access, and never reads or
writes a ticker identity, private path, or raw OHLCV value. Importing it
performs no state-ledger read/write; production root resolution uses only
the fixed machine-local OS location described below (never HOME/USERPROFILE).
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

STUDY_NAME = "V8C_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8C_HUMAN_GATE_CONSUMPTION_RECEIPT_V1"

REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"

GATE_ALLOCATE_T1C = "ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1C"
GATE_PIN_T1C_ALLOCATION = "HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1C_ALLOCATION"
GATE_T1C_TRANSPORT_READINESS = "T1C_TRANSPORT_READINESS_HUMAN_GATE"
GATE_T1C_RAW_ACQUISITION = "T1C_RAW_ACQUISITION_HUMAN_GATE"
GATE_T1C_RESEARCH_OPENING = "SEPARATE_T1C_RESEARCH_OPENING_GATE"
GATE_T2_AUTHORITY_BRIDGE = "HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE"
GATE_T2_TRANSPORT_READINESS = "T2_TRANSPORT_READINESS_HUMAN_GATE"
GATE_T2_RAW_ACQUISITION = "T2_RAW_ACQUISITION_HUMAN_GATE"
GATE_T2_RESEARCH_OPENING = "SEPARATE_T2_RESEARCH_OPENING_GATE"

KNOWN_GATES = (
    GATE_ALLOCATE_T1C,
    GATE_PIN_T1C_ALLOCATION,
    GATE_T1C_TRANSPORT_READINESS,
    GATE_T1C_RAW_ACQUISITION,
    GATE_T1C_RESEARCH_OPENING,
    GATE_T2_AUTHORITY_BRIDGE,
    GATE_T2_TRANSPORT_READINESS,
    GATE_T2_RAW_ACQUISITION,
    GATE_T2_RESEARCH_OPENING,
)

# Gates whose receipt is additionally keyed by the exact human authorization
# identity, so a fresh authorization is a fresh key (see module docstring).
PER_AUTHORIZATION_GATES = frozenset({GATE_T1C_TRANSPORT_READINESS, GATE_T2_TRANSPORT_READINESS})

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
        raise RuntimeError("V8C_HUMAN_GATE_STATE_ROOT_UNAVAILABLE") from error


def _default_consumption_state_root() -> Path:
    if os.name == "nt":
        base = _resolve_windows_program_data_directory()
        return base / "stock-analyzer" / "v8c-human-gate-state"
    else:
        base = _POSIX_MACHINE_STATE_BASE
        return base / "v8c-human-gate-state"


CANONICAL_CONSUMPTION_STATE_ROOT = _default_consumption_state_root()


class V8CHumanGateConsumptionBlocked(RuntimeError):
    """Fail-closed durable one-shot human-gate consumption error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_known_gate(gate: str) -> str:
    if gate not in KNOWN_GATES:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_UNKNOWN")
    return gate


def _require_git_commit(value: object) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_DESIGN_COMMIT_INVALID")
    return value


def _require_authorization_identity(gate: str, authorization_identity: str | None) -> str | None:
    if gate in PER_AUTHORIZATION_GATES:
        if not isinstance(authorization_identity, str) or not authorization_identity:
            raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_AUTHORIZATION_IDENTITY_REQUIRED")
        return authorization_identity
    if authorization_identity is not None:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_AUTHORIZATION_IDENTITY_NOT_APPLICABLE")
    return None


def _identity_sha256(raw_authorization_identity: str) -> str:
    """The one-way hash of a raw human authorization identity. This is the
    ONLY form of the identity ever persisted (in a receipt's own content or
    in a receipt key) -- the raw identity itself is used transiently, as a
    function argument, and never written to durable storage."""
    return hashlib.sha256(raw_authorization_identity.encode("utf-8")).hexdigest()


def _require_identity_sha256(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_AUTHORIZATION_IDENTITY_HASH_INVALID")
    return value


def _receipt_key_from_identity_hash(
    gate: str, v8c_frozen_design_commit: str, authorization_identity_sha256: str | None
) -> str:
    """Derive the durable receipt key from only SAFE, already-hashed
    components -- never the raw authorization identity. Used both to
    compute the key at consumption time (from a freshly-hashed identity)
    and to independently RECOMPUTE the key from a receipt's own persisted
    content at read time, so a receipt's filename/key can never merely be
    asserted -- it must be mechanically derivable from what the receipt
    itself safely declares."""
    _require_known_gate(gate)
    _require_git_commit(v8c_frozen_design_commit)
    key_material = REPOSITORY_IDENTITY + "|" + gate + "|" + v8c_frozen_design_commit
    if gate in PER_AUTHORIZATION_GATES:
        key_material += "|" + _require_identity_sha256(authorization_identity_sha256)
    elif authorization_identity_sha256 is not None:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_AUTHORIZATION_IDENTITY_NOT_APPLICABLE")
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


def _receipt_key(gate: str, v8c_frozen_design_commit: str, authorization_identity: str | None) -> str:
    identity = _require_authorization_identity(gate, authorization_identity)
    identity_hash = _identity_sha256(identity) if identity is not None else None
    return _receipt_key_from_identity_hash(gate, v8c_frozen_design_commit, identity_hash)


def _receipt_path(
    state_root: str | os.PathLike[str],
    gate: str,
    v8c_frozen_design_commit: str,
    authorization_identity: str | None,
) -> Path:
    return Path(state_root) / (_receipt_key(gate, v8c_frozen_design_commit, authorization_identity) + ".json")


def compute_receipt_key(
    gate: str, v8c_frozen_design_commit: str, authorization_identity: str | None = None
) -> str:
    """Public, safe wrapper around the internal receipt key derivation.
    Other V8C modules (e.g. durable stage evidence) may record this key --
    a one-way hash, never the raw authorization identity -- so a later
    reader can deterministically locate and mechanically re-verify the
    exact real consumed receipt without ever needing the raw identity
    itself."""
    return _receipt_key(gate, v8c_frozen_design_commit, authorization_identity)


RECEIPT_FIELDS = (
    "schema_version",
    "study_name",
    "repository",
    "gate",
    "v8c_frozen_design_commit",
    "per_authorization_gate",
    "authorization_identity_sha256",
    "consumed_at_utc",
)


def read_gate_consumption_receipt(state_root: str | os.PathLike[str], receipt_key: str) -> dict[str, Any]:
    """Read-only: mechanically read and strictly validate the exact durable
    gate-consumption receipt located at ``receipt_key`` (as returned by
    ``compute_receipt_key``/produced by ``consume_gate_once``) -- never
    inferred from another artifact's claims about the receipt's content.
    Fails closed on a missing, malformed, duplicate-keyed, or schema-
    invalid receipt."""
    if not isinstance(receipt_key, str) or len(receipt_key) != 64 or any(
        char not in "0123456789abcdef" for char in receipt_key
    ):
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_KEY_INVALID")
    path = Path(state_root) / (receipt_key + ".json")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_MISSING") from error

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_DUPLICATE_KEY")
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_INVALID_JSON") from error
    if not isinstance(parsed, dict) or set(parsed) != set(RECEIPT_FIELDS):
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_SCHEMA_INVALID")
    if parsed["schema_version"] != SCHEMA_VERSION:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_SCHEMA_VERSION_MISMATCH")
    if parsed["study_name"] != STUDY_NAME:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_STUDY_MISMATCH")
    if parsed["repository"] != REPOSITORY_IDENTITY:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_REPOSITORY_MISMATCH")
    if parsed["gate"] not in KNOWN_GATES:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_GATE_INVALID")
    _require_git_commit(parsed["v8c_frozen_design_commit"])
    if not isinstance(parsed["per_authorization_gate"], bool):
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_PER_AUTHORIZATION_FLAG_INVALID")
    if parsed["per_authorization_gate"] != (parsed["gate"] in PER_AUTHORIZATION_GATES):
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_PER_AUTHORIZATION_FLAG_MISMATCH")
    identity_hash = parsed["authorization_identity_sha256"]
    if parsed["per_authorization_gate"]:
        if not isinstance(identity_hash, str) or len(identity_hash) != 64 or any(
            char not in "0123456789abcdef" for char in identity_hash
        ):
            raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_AUTHORIZATION_IDENTITY_HASH_INVALID")
    elif identity_hash is not None:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_AUTHORIZATION_IDENTITY_HASH_INVALID")
    if not isinstance(parsed["consumed_at_utc"], str) or not parsed["consumed_at_utc"]:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_TIMESTAMP_INVALID")

    # Not merely internally self-consistent field-by-field: the requested
    # ``receipt_key`` (the filename this receipt was located at) must be
    # exactly the canonical key the receipt's OWN safe content (repository,
    # gate, design commit, authorization identity hash) derives. A well-
    # formed receipt copied to -- or fabricated at -- an arbitrary/wrong
    # 64-hex filename fails this recomputation and BLOCKs, even though
    # every individual field independently validates.
    try:
        recomputed_key = _receipt_key_from_identity_hash(
            parsed["gate"], parsed["v8c_frozen_design_commit"], identity_hash
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_KEY_CONTENT_MISMATCH") from error
    if recomputed_key != receipt_key:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_RECEIPT_KEY_CONTENT_MISMATCH")

    return dict(parsed)


def _utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_CLOCK_INVALID")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def has_gate_been_consumed(
    state_root: str | os.PathLike[str],
    gate: str,
    v8c_frozen_design_commit: str,
    *,
    authorization_identity: str | None = None,
) -> bool:
    """Read-only existence check -- never raises for "not yet consumed";
    only raises for a malformed ``gate``/``v8c_frozen_design_commit``/
    ``authorization_identity`` or an unreadable state root."""
    path = _receipt_path(state_root, gate, v8c_frozen_design_commit, authorization_identity)
    try:
        return path.exists()
    except OSError as error:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_STATE_UNAVAILABLE") from error


def require_gate_not_yet_consumed(
    state_root: str | os.PathLike[str],
    gate: str,
    v8c_frozen_design_commit: str,
    *,
    authorization_identity: str | None = None,
) -> None:
    """Fail-fast, read-only pre-check: BLOCK immediately if ``gate`` (under
    this exact key) has already been durably consumed. Call this before any
    provenance/private-access step so a replay attempt never even reaches
    Git resolution."""
    if has_gate_been_consumed(state_root, gate, v8c_frozen_design_commit, authorization_identity=authorization_identity):
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate)


def consume_gate_once(
    state_root: str | os.PathLike[str],
    gate: str,
    v8c_frozen_design_commit: str,
    *,
    clock: Callable[[], datetime],
    authorization_identity: str | None = None,
) -> None:
    """Atomically, durably mark ``gate`` consumed under this exact key.
    Raises -- never silently succeeds -- if a receipt already exists,
    including the benign race where two callers attempt this at once (the
    underlying ``os.link`` no-overwrite publish is atomic). The caller must
    invoke this exactly once, strictly before the actual private-access/
    network action it authorizes, and must treat any exception here as
    "not consumed" (no private/network action may proceed).
    """
    path = _receipt_path(state_root, gate, v8c_frozen_design_commit, authorization_identity)
    if path.exists():
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "repository": REPOSITORY_IDENTITY,
        "gate": gate,
        "v8c_frozen_design_commit": v8c_frozen_design_commit,
        "per_authorization_gate": gate in PER_AUTHORIZATION_GATES,
        "authorization_identity_sha256": (
            _identity_sha256(authorization_identity) if authorization_identity is not None else None
        ),
        "consumed_at_utc": _timestamp_text(_utc_timestamp(clock() if callable(clock) else clock)),
    }
    payload = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_STATE_UNAVAILABLE") from error

    staging = path.parent / (path.name + ".staging-" + os.urandom(8).hex())
    try:
        try:
            with open(staging, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_STATE_WRITE_FAILED") from error
        try:
            os.link(str(staging), str(path))
        except FileExistsError as error:
            raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_ALREADY_CONSUMED:" + gate) from error
        except OSError as error:
            raise V8CHumanGateConsumptionBlocked("V8C_HUMAN_GATE_STATE_WRITE_FAILED") from error
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass


__all__ = [
    "CANONICAL_CONSUMPTION_STATE_ROOT",
    "CANONICAL_REPOSITORY_ROOT",
    "GATE_ALLOCATE_T1C",
    "GATE_PIN_T1C_ALLOCATION",
    "GATE_T1C_RAW_ACQUISITION",
    "GATE_T1C_RESEARCH_OPENING",
    "GATE_T1C_TRANSPORT_READINESS",
    "GATE_T2_AUTHORITY_BRIDGE",
    "GATE_T2_RAW_ACQUISITION",
    "GATE_T2_RESEARCH_OPENING",
    "GATE_T2_TRANSPORT_READINESS",
    "KNOWN_GATES",
    "PER_AUTHORIZATION_GATES",
    "RECEIPT_FIELDS",
    "REPOSITORY_IDENTITY",
    "SCHEMA_VERSION",
    "STUDY_NAME",
    "V8CHumanGateConsumptionBlocked",
    "compute_receipt_key",
    "consume_gate_once",
    "has_gate_been_consumed",
    "read_gate_consumption_receipt",
    "require_gate_not_yet_consumed",
]
