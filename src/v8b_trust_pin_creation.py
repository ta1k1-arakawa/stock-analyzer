"""Production-gated `CREATE_V8B_TRUSTED_ALLOCATION_PIN` boundary (§11.3.C).

Frozen §12 gate sequence: READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION ->
HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION ->
CREATE_V8B_TRUSTED_ALLOCATION_PIN -> INDEPENDENT_TRUST_PIN_REVIEW ->
T1B_RAW_ACQUISITION_HUMAN_GATE.

`FINAL_REPEAT_INDEPENDENT_V8B_PRODUCTION_IMPLEMENTATION_REVIEW` finding
HIGH-2 (original round): `src/v8b_trust_pin.py::build_trust_pin` accepts
any caller-supplied mapping shaped like a PASS verification summary --
nothing previously required that mapping to have actually come from a
real, Git-grounded `READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION` call,
so an arbitrary caller-fabricated ``{"result": "PASS", ...}`` dict was
sufficient to build an "AUTHORIZED" pin object. This module closes that
gap: its sole entrypoint, ``create_v8b_trusted_allocation_pin_production``,
obtains the verification summary **only** by calling the real production
resolver (`src.v8b_allocation_verification.resolve_and_verify_t1b_
allocation_artifact`), and requires an exact human-authorization token
matching the frozen `HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION`
grammar bound to that exact verified artifact's ``artifact_self_hash``
(`src.v8b_trust_pin.expected_human_gate`).

Repeat-round finding HIGH-1: the implementation previously required a
fresh `INDEPENDENT_TRUST_PIN_REVIEW` artifact **before** this module would
write the pin -- backwards from the frozen sequence above, which places
INDEPENDENT_TRUST_PIN_REVIEW strictly *after* CREATE_V8B_TRUSTED_
ALLOCATION_PIN (a review of the pin cannot possibly precede the pin's own
existence). This module now depends on `INDEPENDENT_TRUST_PIN_REVIEW`
**not at all**: pin creation happens strictly between the
HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION human gate and the
(now-later) INDEPENDENT_TRUST_PIN_REVIEW gate, which is instead verified
downstream, at T1B acquisition time
(`src/v8b_historical_acquisition.py`, `src/v8b_acquisition_artifact_
verification.py`) -- the earliest point at which the *published* pin
(this module's own write) actually exists to be reviewed.

In its place, this module durably, fail-closed, one-shot-consumes the new
`HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION` gate
(`src.v8b_human_gate_consumption`) strictly after allocation verification
and human-authorization-token validation, and strictly before writing the
pin -- so a second call using the same authorization can never write a
second pin, and a rejected/failed call (wrong confirmation, wrong human
authorization, failed allocation verification) never consumes the gate.

This module is **not executed** by this implementation phase -- no real
`T1B` allocation, allocation verification, or human pin authorization has
occurred, so every real invocation of
``create_v8b_trusted_allocation_pin_production`` fails closed today by
construction: its prerequisite Git-tracked artifact
(`V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json`) does not exist, and its
prerequisite private `T1B` allocation artifact does not exist either.
Every test exercising this module is fake/synthetic-only. Importing this
module performs no I/O and no network access; it never imports
`src/v7_yahoo_collector.py`.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v8b_allocation import canonical_json_bytes
from src.v8b_allocation_verification import (
    V8BAllocationVerificationBlocked,
    resolve_and_verify_t1b_allocation_artifact,
)
from src.v8b_git_provenance import (
    V8BGitProvenanceBlocked,
    resolve_verified_v8b_production_git_commit,
)
from src.v8b_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_PIN_VERIFIED_T1B_ALLOCATION,
    V8BHumanGateConsumptionBlocked,
    consume_gate_once,
    require_gate_not_yet_consumed,
)
from src.v8b_production_provenance import (
    EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
    V8BProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8b_trust_pin import (
    V8BTrustPinBlocked,
    build_trust_pin,
    expected_human_gate,
)

# Frozen production confirmation literal (§12's `CREATE_V8B_TRUSTED_
# ALLOCATION_PIN` gate). Mechanical anti-fat-finger syntax only, mirroring
# this repository's existing convention -- not itself real human
# authorization.
PIN_CREATION_CONFIRMATION = "V8B_PRODUCTION_CREATE_TRUSTED_ALLOCATION_PIN"

CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PIN_ARTIFACT_FILENAME = "V8B_TRUSTED_ALLOCATION.json"


class V8BTrustPinCreationBlocked(RuntimeError):
    """Fail-closed production trust-pin-creation error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8BTrustPinCreationBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8BTrustPinCreationBlocked(missing_reason)
    if isinstance(reason, str):
        return V8BTrustPinCreationBlocked(reason)
    return V8BTrustPinCreationBlocked("PROVENANCE_CHECK_FAILED")


def _write_pin_once(destination: Path, pin_bytes: bytes) -> Path:
    """Atomically publish, never replacing an existing destination.

    Mirrors `src/v8b_t1b_allocator.py::_write_allocation_artifact_once`'s
    proven fsync-then-``os.link`` no-overwrite pattern.
    """
    if destination.exists():
        raise V8BTrustPinCreationBlocked("V8B_TRUSTED_ALLOCATION_PIN_ALREADY_EXISTS")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8BTrustPinCreationBlocked("OUTPUT_PATH_PARENT_INVALID") from error
    if not destination.parent.is_dir():
        raise V8BTrustPinCreationBlocked("OUTPUT_PATH_PARENT_INVALID")
    staging = destination.parent / (destination.name + ".staging-" + os.urandom(8).hex())
    try:
        try:
            with open(staging, "wb") as stream:
                stream.write(pin_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise V8BTrustPinCreationBlocked("V8B_TRUSTED_ALLOCATION_PIN_STAGING_WRITE_FAILED") from error
        try:
            os.link(str(staging), str(destination))
        except FileExistsError as error:
            raise V8BTrustPinCreationBlocked("V8B_TRUSTED_ALLOCATION_PIN_ALREADY_EXISTS") from error
        except OSError as error:
            raise V8BTrustPinCreationBlocked("V8B_TRUSTED_ALLOCATION_PIN_ATOMIC_PUBLISH_FAILED") from error
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass
    return destination


def create_v8b_trusted_allocation_pin_production(
    *,
    confirmation: str,
    human_pin_authorization: str,
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    authorization_note: str,
) -> dict[str, Any]:
    """Sole production entrypoint. **Not executed** by this implementation
    phase -- see module docstring. Returns only the safe pin fields (no
    ticker identities by schema construction); the pin itself is written
    to ``output_path``.
    """
    return _create_v8b_trusted_allocation_pin_production_with_dependencies(
        confirmation=confirmation,
        human_pin_authorization=human_pin_authorization,
        allocation_artifact_path=allocation_artifact_path,
        partition_manifest_path=partition_manifest_path,
        output_path=output_path,
        authorization_note=authorization_note,
        git_commit_resolver=lambda: resolve_verified_v8b_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        design_freeze_approval_reader=lambda head: read_and_verify_design_freeze_approval(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        frozen_design_object_verifier=lambda: verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        allocation_verification_resolver=lambda: resolve_and_verify_t1b_allocation_artifact(
            allocation_artifact_path, partition_manifest_path
        ),
        clock=lambda: datetime.now(timezone.utc),
        consumption_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
    )


def _create_v8b_trusted_allocation_pin_production_with_dependencies(
    *,
    confirmation: str,
    human_pin_authorization: str,
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    authorization_note: str,
    git_commit_resolver: Callable[[], str],
    design_freeze_approval_reader: Callable[[str], Mapping[str, Any]],
    frozen_design_object_verifier: Callable[[], None],
    reviewed_implementation_binder: Callable[[str], Mapping[str, Any]],
    allocation_verification_resolver: Callable[[], Mapping[str, Any]],
    clock: Callable[[], datetime],
    consumption_state_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Private fake-only seam. No OHLCV, no network, no retry: exactly one
    deterministic attempt per call -- either it succeeds once or raises.

    This function does **not** depend on ``INDEPENDENT_TRUST_PIN_REVIEW`` --
    that gate is strictly downstream of this module's own write (repeat-
    round finding HIGH-1); see module docstring.
    """
    # (0) explicit, exact pin-creation confirmation token
    if confirmation != PIN_CREATION_CONFIRMATION:
        raise V8BTrustPinCreationBlocked("V8B_PIN_CREATION_CONFIRMATION_INVALID")

    # (0.5) fail fast, read-only: HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_
    # ALLOCATION must not already have been durably consumed.
    try:
        require_gate_not_yet_consumed(
            consumption_state_root, GATE_PIN_VERIFIED_T1B_ALLOCATION, EXPECTED_V8B_FROZEN_DESIGN_COMMIT
        )
    except V8BHumanGateConsumptionBlocked as error:
        raise V8BTrustPinCreationBlocked(error.reason) from error

    # (1) repo/provenance -- V8B's own branch
    try:
        verified_head = git_commit_resolver()
    except V8BGitProvenanceBlocked as error:
        raise _wrap(error) from error

    # (2) frozen design object + freeze approval (exact blob + fields)
    try:
        frozen_design_object_verifier()
        design_freeze_approval_reader(verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    # (3) reviewed implementation binding
    try:
        reviewed_implementation_binder(verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error, "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error

    # (4) READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION -- the verification
    # summary can only come from the real, Git-grounded production
    # resolver, never an arbitrary caller-supplied mapping (HIGH-2).
    try:
        verification_result = allocation_verification_resolver()
    except V8BAllocationVerificationBlocked as error:
        raise V8BTrustPinCreationBlocked("V8B_ALLOCATION_VERIFICATION_FAILED:" + error.reason) from error

    artifact_self_hash = verification_result.get("artifact_self_hash")
    if not isinstance(artifact_self_hash, str):
        raise V8BTrustPinCreationBlocked("V8B_ALLOCATION_VERIFICATION_RESULT_INVALID")

    # (5) HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION -- exact frozen
    # grammar bound to this exact verified artifact hash, never an
    # arbitrary nonempty string.
    if human_pin_authorization != expected_human_gate(artifact_self_hash):
        raise V8BTrustPinCreationBlocked("V8B_HUMAN_PIN_AUTHORIZATION_INVALID")

    # (6) durably, fail-closed, one-shot consume the
    # HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1B_ALLOCATION gate -- strictly
    # after every check above (so a rejected call never consumes it) and
    # strictly before the pin write below (so a second call with the same
    # authorization BLOCKs before it could ever produce a second pin).
    try:
        consume_gate_once(
            consumption_state_root, GATE_PIN_VERIFIED_T1B_ALLOCATION, EXPECTED_V8B_FROZEN_DESIGN_COMMIT, clock=clock
        )
    except V8BHumanGateConsumptionBlocked as error:
        raise V8BTrustPinCreationBlocked(error.reason) from error

    # (7) build (never trusts a caller-supplied summary) and write-once
    try:
        pin = build_trust_pin(verification_result_summary=verification_result, authorization_note=authorization_note)
    except V8BTrustPinBlocked as error:
        raise V8BTrustPinCreationBlocked("V8B_TRUST_PIN_BUILD_FAILED:" + error.reason) from error

    destination = Path(output_path)
    _write_pin_once(destination, canonical_json_bytes(pin))
    return dict(pin)


__all__ = [
    "CANONICAL_REPOSITORY_ROOT",
    "PIN_ARTIFACT_FILENAME",
    "PIN_CREATION_CONFIRMATION",
    "V8BTrustPinCreationBlocked",
    "create_v8b_trusted_allocation_pin_production",
]
