"""Durable, fail-closed, one-shot consumption receipts for V8B's named
one-time human gates (§12): `ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B`,
`HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION`,
`T1B_RAW_ACQUISITION_HUMAN_GATE`, `T2_RAW_ACQUISITION_HUMAN_GATE`.

`FINAL_REPEAT_INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` finding
HIGH-1 (original round): before this module existed, ``authorization_consumed``
was only an in-memory boolean local to a single function call -- nothing
durable ever recorded that a one-time gate had already fired, so a second
call, a new process, or a process restart could silently repeat the exact
same one-time action under the exact same authorization. Every function in
`src/v8b_t1b_allocator.py`, `src/v8b_trust_pin_creation.py` and
`src/v8b_historical_acquisition.py` that consumes one of the named gates
above now does so through this module instead of a bare local variable.

Repeat-round finding HIGH-1: the frozen §12 sequence is
READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION ->
HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION ->
CREATE_V8B_TRUSTED_ALLOCATION_PIN -> INDEPENDENT_TRUST_PIN_REVIEW ->
T1B_RAW_ACQUISITION_HUMAN_GATE. The new
``GATE_PIN_VERIFIED_T1B_ALLOCATION`` gate is the durable, one-shot,
fail-closed human authorization to *pin* the already-verified allocation --
distinct from, and consumed strictly before, trust-pin creation, and wholly
independent of the *later* INDEPENDENT_TRUST_PIN_REVIEW of the pin that
creation produces.

Repeat-round finding MEDIUM-1: ``CANONICAL_CONSUMPTION_STATE_ROOT`` used to
be derived from this module file's own checkout path
(``Path(__file__).resolve().parents[1]``), which means a second clone or
worktree of this same repository at a different filesystem path resolved to
a *different*, empty receipt directory -- silently defeating the one-shot
guarantee above for any caller running from that second checkout. The
canonical state root is now derived only from fixed, trust-bearing values
that do not vary by checkout path: the user's home directory (falling back
to a fixed non-checkout path if home resolution fails) joined with a
namespace derived from this repository's fixed identity string
(``REPOSITORY_IDENTITY``). Every checkout of this same repository, at any
path, on the same machine, now resolves to the exact same canonical ledger.

A "consumption" here is a small, fsync'd, atomically-created (never
overwritten) receipt file on durable local storage, keyed by the exact
repository identity, gate name, and the exact frozen V8B design commit it
was consumed under -- never by anything ticker-, date-, or path-derived, so
a receipt itself carries no private information. ``require_gate_not_yet_consumed``
is a read-only, non-mutating check any caller may run as an early fail-fast
gate; ``consume_gate_once`` is the sole function that ever creates a
receipt, and it is fail-closed: if a receipt for this exact
``(gate, v8b_frozen_design_commit)`` pair already exists (whether created
by an earlier call in this process, a previous process, a previous restart,
or a different checkout of this same repository), it raises rather than
silently succeeding, and it never overwrites or deletes an existing
receipt. Every production caller invokes ``consume_gate_once`` exactly
once, at the exact point that already constituted "consumption" before this
module existed (the first private partition read for the allocator; the
pin-write step of trust-pin creation; the first trusted Yahoo opener
invocation for acquisition) -- strictly before the actual private-access /
network action, so a second invocation using the same authorization BLOCKs
before that private access / network request, never merely after it. There
is deliberately no deletion/reset API: a consumed gate stays consumed.

This module performs no Git access, no network access, and never reads or
writes a ticker identity, private path, or raw OHLCV value. Importing it
performs no I/O beyond resolving the home directory path (no read/write).
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

STUDY_NAME = "V8B_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8B_HUMAN_GATE_CONSUMPTION_RECEIPT_V2"

# The fixed repository identity this ledger is scoped to. Deliberately a
# literal, not derived from the checkout path -- MEDIUM-1 (repeat round).
REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"

# The exact one-time human gates this module names. No other gate name is
# accepted by this module.
GATE_ALLOCATE_T1B = "ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B"
GATE_PIN_VERIFIED_T1B_ALLOCATION = "HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION"
GATE_T1B_RAW_ACQUISITION = "T1B_RAW_ACQUISITION_HUMAN_GATE"
GATE_T2_RAW_ACQUISITION = "T2_RAW_ACQUISITION_HUMAN_GATE"

KNOWN_GATES = (
    GATE_ALLOCATE_T1B,
    GATE_PIN_VERIFIED_T1B_ALLOCATION,
    GATE_T1B_RAW_ACQUISITION,
    GATE_T2_RAW_ACQUISITION,
)

# Informational only (no longer used to derive the state root -- MEDIUM-1).
CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_REPOSITORY_IDENTITY_NAMESPACE = hashlib.sha256(REPOSITORY_IDENTITY.encode("utf-8")).hexdigest()[:16]

# A fixed, non-checkout-path-derived fallback used only if the home
# directory cannot be resolved at all (e.g. no passwd entry / no HOME env).
_FALLBACK_HOME_DIRECTORY = Path("/var/lib")


def _resolve_home_directory() -> Path:
    try:
        return Path.home()
    except RuntimeError:
        return _FALLBACK_HOME_DIRECTORY


def _default_consumption_state_root() -> Path:
    return _resolve_home_directory() / ".v8b_human_gate_state" / _REPOSITORY_IDENTITY_NAMESPACE


# The one fixed, non-overridable production consumption-state root. Derived
# solely from the machine-local home directory and a namespace fixed by
# REPOSITORY_IDENTITY -- never from this module's own checkout path, so
# every clone/worktree of this repository on this machine resolves to the
# exact same durable ledger (MEDIUM-1, repeat round). This is deliberately
# not caller-suppliable on any public production entrypoint, mirroring this
# repository's existing no-caller-override convention for other
# trust-bearing roots.
CANONICAL_CONSUMPTION_STATE_ROOT = _default_consumption_state_root()


class V8BHumanGateConsumptionBlocked(RuntimeError):
    """Fail-closed durable one-shot human-gate consumption error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_known_gate(gate: str) -> str:
    if gate not in KNOWN_GATES:
        raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_UNKNOWN")
    return gate


def _require_git_commit(value: object) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_DESIGN_COMMIT_INVALID")
    return value


def _receipt_key(gate: str, v8b_frozen_design_commit: str) -> str:
    _require_known_gate(gate)
    _require_git_commit(v8b_frozen_design_commit)
    digest = hashlib.sha256(
        (REPOSITORY_IDENTITY + "|" + gate + "|" + v8b_frozen_design_commit).encode("utf-8")
    ).hexdigest()
    return digest


def _receipt_path(state_root: str | os.PathLike[str], gate: str, v8b_frozen_design_commit: str) -> Path:
    return Path(state_root) / (_receipt_key(gate, v8b_frozen_design_commit) + ".json")


def _utc_timestamp(value: Any) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_CLOCK_INVALID")
    return value.astimezone(timezone.utc)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def has_gate_been_consumed(state_root: str | os.PathLike[str], gate: str, v8b_frozen_design_commit: str) -> bool:
    """Read-only existence check -- never raises for "not yet consumed";
    only raises for a malformed ``gate``/``v8b_frozen_design_commit`` or an
    unreadable state root. Safe to call as an authoritative fact source by
    other production checks (e.g. the §12.4 T2 reuse recheck), not only as
    a pre-flight gate."""
    path = _receipt_path(state_root, gate, v8b_frozen_design_commit)
    try:
        return path.exists()
    except OSError as error:
        raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_STATE_UNAVAILABLE") from error


def require_gate_not_yet_consumed(
    state_root: str | os.PathLike[str], gate: str, v8b_frozen_design_commit: str
) -> None:
    """Fail-fast, read-only pre-check: BLOCK immediately if ``gate`` has
    already been durably consumed under ``v8b_frozen_design_commit`` --
    whether by an earlier call in this process, a previous process, or a
    previous restart. Call this before any provenance/private-access step
    so a replay attempt never even reaches Git resolution."""
    if has_gate_been_consumed(state_root, gate, v8b_frozen_design_commit):
        raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_ALREADY_CONSUMED:" + gate)


def consume_gate_once(
    state_root: str | os.PathLike[str],
    gate: str,
    v8b_frozen_design_commit: str,
    *,
    clock: Callable[[], datetime],
) -> None:
    """Atomically, durably mark ``gate`` consumed under
    ``v8b_frozen_design_commit``. Raises -- never silently succeeds -- if a
    receipt already exists, including the benign race where two callers
    attempt this at once (the underlying ``os.link`` no-overwrite publish
    is atomic). The caller must invoke this exactly once, strictly before
    the actual private-access/network action it authorizes, and must treat
    any exception here as "not consumed" (no private/network action may
    proceed).
    """
    path = _receipt_path(state_root, gate, v8b_frozen_design_commit)
    if path.exists():
        raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_ALREADY_CONSUMED:" + gate)

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "study_name": STUDY_NAME,
        "repository": REPOSITORY_IDENTITY,
        "gate": gate,
        "v8b_frozen_design_commit": v8b_frozen_design_commit,
        "consumed_at_utc": _timestamp_text(_utc_timestamp(clock() if callable(clock) else clock)),
    }
    payload = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_STATE_UNAVAILABLE") from error

    staging = path.parent / (path.name + ".staging-" + os.urandom(8).hex())
    try:
        try:
            with open(staging, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_STATE_WRITE_FAILED") from error
        try:
            os.link(str(staging), str(path))
        except FileExistsError as error:
            raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_ALREADY_CONSUMED:" + gate) from error
        except OSError as error:
            raise V8BHumanGateConsumptionBlocked("V8B_HUMAN_GATE_STATE_WRITE_FAILED") from error
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass


__all__ = [
    "CANONICAL_CONSUMPTION_STATE_ROOT",
    "CANONICAL_REPOSITORY_ROOT",
    "GATE_ALLOCATE_T1B",
    "GATE_PIN_VERIFIED_T1B_ALLOCATION",
    "GATE_T1B_RAW_ACQUISITION",
    "GATE_T2_RAW_ACQUISITION",
    "KNOWN_GATES",
    "REPOSITORY_IDENTITY",
    "SCHEMA_VERSION",
    "STUDY_NAME",
    "V8BHumanGateConsumptionBlocked",
    "consume_gate_once",
    "has_gate_been_consumed",
    "require_gate_not_yet_consumed",
]
