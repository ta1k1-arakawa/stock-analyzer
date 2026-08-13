"""Durable, fail-closed, one-shot consumption receipts for V8B's named
one-time human gates (§12): `ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B`,
`T1B_RAW_ACQUISITION_HUMAN_GATE`, `T2_RAW_ACQUISITION_HUMAN_GATE`.

`FINAL_REPEAT_INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` finding
HIGH-1: before this module existed, ``authorization_consumed`` was only an
in-memory boolean local to a single function call -- nothing durable ever
recorded that a one-time gate had already fired, so a second call, a new
process, or a process restart could silently repeat the exact same
one-time action under the exact same authorization. Every function in
`src/v8b_t1b_allocator.py` and `src/v8b_historical_acquisition.py` that
consumes one of the three gates above now does so through this module
instead of a bare local variable.

A "consumption" here is a small, fsync'd, atomically-created (never
overwritten) receipt file on durable local storage, keyed by the exact
gate name and the exact frozen V8B design commit it was consumed under --
never by anything ticker-, date-, or path-derived, so a receipt itself
carries no private information. ``require_gate_not_yet_consumed`` is a
read-only, non-mutating check any caller may run as an early fail-fast
gate; ``consume_gate_once`` is the sole function that ever creates a
receipt, and it is fail-closed: if a receipt for this exact
``(gate, v8b_frozen_design_commit)`` pair already exists (whether created
by an earlier call in this process, a previous process, or a previous
restart), it raises rather than silently succeeding, and it never
overwrites or deletes an existing receipt. Every production caller invokes
``consume_gate_once`` exactly once, at the exact point that already
constituted "consumption" before this module existed (the first private
partition read for the allocator; the first trusted Yahoo opener
invocation for acquisition) -- strictly before the actual private-access /
network action, so a second invocation using the same authorization BLOCKs
before that private access / network request, never merely after it.

This module performs no Git access, no network access, and never reads or
writes a ticker identity, private path, or raw OHLCV value. Importing it
performs no I/O.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

STUDY_NAME = "V8B_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8B_HUMAN_GATE_CONSUMPTION_RECEIPT_V1"

# The exact three one-time human gates HIGH-1 names. No other gate name is
# accepted by this module.
GATE_ALLOCATE_T1B = "ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B"
GATE_T1B_RAW_ACQUISITION = "T1B_RAW_ACQUISITION_HUMAN_GATE"
GATE_T2_RAW_ACQUISITION = "T2_RAW_ACQUISITION_HUMAN_GATE"

KNOWN_GATES = (GATE_ALLOCATE_T1B, GATE_T1B_RAW_ACQUISITION, GATE_T2_RAW_ACQUISITION)

# The one fixed, non-overridable production consumption-state root: a
# sibling directory of the repository checkout, outside the Git worktree
# (so it can never dirty `git status --porcelain`, which V8B's own
# provenance check requires to be clean) and outside any private V8/V8B
# data directory. Every production call from the same checkout resolves to
# exactly this same directory, so a receipt written by one process/call is
# durably visible to the next -- this is deliberately not caller-suppliable
# on any public production entrypoint, mirroring this repository's existing
# no-caller-override convention for other trust-bearing roots.
CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONSUMPTION_STATE_ROOT = (
    CANONICAL_REPOSITORY_ROOT.parent / (CANONICAL_REPOSITORY_ROOT.name + ".v8b_human_gate_state")
)


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
    digest = hashlib.sha256((gate + "|" + v8b_frozen_design_commit).encode("utf-8")).hexdigest()
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
    "GATE_ALLOCATE_T1B",
    "GATE_T1B_RAW_ACQUISITION",
    "GATE_T2_RAW_ACQUISITION",
    "KNOWN_GATES",
    "SCHEMA_VERSION",
    "STUDY_NAME",
    "V8BHumanGateConsumptionBlocked",
    "consume_gate_once",
    "has_gate_been_consumed",
    "require_gate_not_yet_consumed",
]
