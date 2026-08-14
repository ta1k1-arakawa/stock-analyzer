"""Production-gated `CREATE_V8C_TRUSTED_ALLOCATION_PIN` boundary.

Frozen §12 gate sequence: READ_ONLY_T1C_ALLOCATION_ARTIFACT_VERIFICATION ->
HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1C_ALLOCATION ->
CREATE_V8C_TRUSTED_ALLOCATION_PIN -> INDEPENDENT_TRUST_PIN_REVIEW ->
T1C_TRANSPORT_READINESS_HUMAN_GATE -> T1C_RAW_ACQUISITION_HUMAN_GATE.

This module's sole entrypoint, ``create_v8c_trusted_allocation_pin_production``,
obtains the verification summary **only** by calling the real production
resolver (`src.v8c_t1c_allocation_verification.resolve_and_verify_t1c_
allocation_artifact`), and requires an exact human-authorization token
matching the frozen `HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1C_ALLOCATION`
grammar bound to that exact verified artifact's ``artifact_self_hash``.

This module durably, fail-closed, one-shot-consumes
`HUMAN_AUTHORIZATION_TO_PIN_VERIFIED_T1C_ALLOCATION`
(`src.v8c_human_gate_consumption`) strictly after allocation verification
and human-authorization-token validation, and strictly before writing the
pin. It does not depend on `INDEPENDENT_TRUST_PIN_REVIEW`, which is
strictly downstream of this module's own write (that review requires the
*published* pin to exist first).

This module is **not executed** by this implementation phase -- no real
`T1C` allocation, allocation verification, or human pin authorization has
occurred, so every real invocation fails closed today by construction.
Every test exercising this module is fake/synthetic-only. Importing this
module performs no I/O and no network access; it never imports
`src/v7_yahoo_collector.py`.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v8c_t1c_allocation import canonical_json_bytes
from src.v8c_t1c_allocation_verification import (
    V8CAllocationVerificationBlocked,
    resolve_and_verify_t1c_allocation_artifact,
)
from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    resolve_verified_v8c_production_git_commit,
)
from src.v8c_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_PIN_T1C_ALLOCATION,
    V8CHumanGateConsumptionBlocked,
    consume_gate_once,
    require_gate_not_yet_consumed,
)
from src.v8c_production_provenance import (
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    V8CProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8c_trust_pin import V8CTrustPinBlocked, build_trust_pin, expected_human_gate

PIN_CREATION_CONFIRMATION = "V8C_PRODUCTION_CREATE_TRUSTED_ALLOCATION_PIN"

PIN_ARTIFACT_FILENAME = "V8C_TRUSTED_ALLOCATION.json"


class V8CTrustPinCreationBlocked(RuntimeError):
    """Fail-closed production trust-pin-creation error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8CTrustPinCreationBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8CTrustPinCreationBlocked(missing_reason)
    if isinstance(reason, str):
        return V8CTrustPinCreationBlocked(reason)
    return V8CTrustPinCreationBlocked("PROVENANCE_CHECK_FAILED")


def _write_pin_once(destination: Path, pin_bytes: bytes) -> Path:
    if destination.exists():
        raise V8CTrustPinCreationBlocked("V8C_TRUSTED_ALLOCATION_PIN_ALREADY_EXISTS")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8CTrustPinCreationBlocked("OUTPUT_PATH_PARENT_INVALID") from error
    if not destination.parent.is_dir():
        raise V8CTrustPinCreationBlocked("OUTPUT_PATH_PARENT_INVALID")
    staging = destination.parent / (destination.name + ".staging-" + os.urandom(8).hex())
    try:
        try:
            with open(staging, "wb") as stream:
                stream.write(pin_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise V8CTrustPinCreationBlocked("V8C_TRUSTED_ALLOCATION_PIN_STAGING_WRITE_FAILED") from error
        try:
            os.link(str(staging), str(destination))
        except FileExistsError as error:
            raise V8CTrustPinCreationBlocked("V8C_TRUSTED_ALLOCATION_PIN_ALREADY_EXISTS") from error
        except OSError as error:
            raise V8CTrustPinCreationBlocked("V8C_TRUSTED_ALLOCATION_PIN_ATOMIC_PUBLISH_FAILED") from error
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass
    return destination


def create_v8c_trusted_allocation_pin_production(
    *,
    confirmation: str,
    human_pin_authorization: str,
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    t1b_allocation_artifact_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    authorization_note: str,
) -> dict[str, Any]:
    """Sole production entrypoint. **Not executed** by this implementation phase."""
    return _create_v8c_trusted_allocation_pin_production_with_dependencies(
        confirmation=confirmation,
        human_pin_authorization=human_pin_authorization,
        allocation_artifact_path=allocation_artifact_path,
        partition_manifest_path=partition_manifest_path,
        t1b_allocation_artifact_path=t1b_allocation_artifact_path,
        output_path=output_path,
        authorization_note=authorization_note,
        git_commit_resolver=lambda: resolve_verified_v8c_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        design_freeze_approval_reader=lambda head: read_and_verify_design_freeze_approval(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        frozen_design_object_verifier=lambda: verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        allocation_verification_resolver=lambda: resolve_and_verify_t1c_allocation_artifact(
            allocation_artifact_path, partition_manifest_path, t1b_allocation_artifact_path
        ),
        clock=lambda: datetime.now(timezone.utc),
        consumption_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
    )


def _create_v8c_trusted_allocation_pin_production_with_dependencies(
    *,
    confirmation: str,
    human_pin_authorization: str,
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    t1b_allocation_artifact_path: str | os.PathLike[str],
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
    if confirmation != PIN_CREATION_CONFIRMATION:
        raise V8CTrustPinCreationBlocked("V8C_PIN_CREATION_CONFIRMATION_INVALID")

    try:
        require_gate_not_yet_consumed(
            consumption_state_root, GATE_PIN_T1C_ALLOCATION, EXPECTED_V8C_FROZEN_DESIGN_COMMIT
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CTrustPinCreationBlocked(error.reason) from error

    try:
        verified_head = git_commit_resolver()
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error) from error

    try:
        frozen_design_object_verifier()
        design_freeze_approval_reader(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    try:
        reviewed_implementation_binder(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error

    try:
        verification_result = allocation_verification_resolver()
    except V8CAllocationVerificationBlocked as error:
        raise V8CTrustPinCreationBlocked("V8C_ALLOCATION_VERIFICATION_FAILED:" + error.reason) from error

    artifact_self_hash = verification_result.get("artifact_self_hash")
    if not isinstance(artifact_self_hash, str):
        raise V8CTrustPinCreationBlocked("V8C_ALLOCATION_VERIFICATION_RESULT_INVALID")

    if human_pin_authorization != expected_human_gate(artifact_self_hash):
        raise V8CTrustPinCreationBlocked("V8C_HUMAN_PIN_AUTHORIZATION_INVALID")

    try:
        consume_gate_once(
            consumption_state_root, GATE_PIN_T1C_ALLOCATION, EXPECTED_V8C_FROZEN_DESIGN_COMMIT, clock=clock
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CTrustPinCreationBlocked(error.reason) from error

    try:
        pin = build_trust_pin(verification_result_summary=verification_result, authorization_note=authorization_note)
    except V8CTrustPinBlocked as error:
        raise V8CTrustPinCreationBlocked("V8C_TRUST_PIN_BUILD_FAILED:" + error.reason) from error

    destination = Path(output_path)
    _write_pin_once(destination, canonical_json_bytes(pin))
    return dict(pin)


__all__ = [
    "CANONICAL_REPOSITORY_ROOT",
    "PIN_ARTIFACT_FILENAME",
    "PIN_CREATION_CONFIRMATION",
    "V8CTrustPinCreationBlocked",
    "create_v8c_trusted_allocation_pin_production",
]
