"""V8D synthetic-injectable transport readiness catch paths, plus the
fail-closed production readiness entrypoints
(`V8D_PROD_HIGH_1B_GATE_CONSUMPTION_RECEIPT_BINDING`).

``execute_transport_readiness_probe`` remains transport-only: it requires
an already-derived ``reviewed_implementation_commit`` and an already-
consumed ``gate_binding`` supplied by the caller, and performs no
provenance resolution or gate consumption of its own -- it is the shared
synthetic-injectable catch path used by both tests and the production
entrypoints below.

``execute_t1c_transport_readiness_production`` / ``execute_t2_transport_
readiness_production`` are the sole production entrypoints. Unlike the
transport-only function above, they never accept an authority-bearing
``reviewed_implementation_commit`` from the caller: they resolve verified
V8D Git HEAD (`src.v8d_git_provenance`), verify the frozen design and
freeze approval, derive the reviewed implementation commit through the
HIGH-1A `src.v8d_production_provenance.verify_reviewed_implementation_
binding`, and durably consume the exact applicable one-shot human gate
(`src.v8d_human_gate_consumption`) -- strictly before the first Yahoo
request -- before ever invoking the transport stage.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any

from src.v8d_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8DGitProvenanceBlocked,
    resolve_verified_v8d_production_git_commit,
)
from src.v8d_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    V8DHumanGateConsumptionBlocked,
    consume_gate_and_bind,
)
from src.v8d_production_provenance import (
    EXPECTED_V8D_FROZEN_DESIGN_COMMIT,
    V8DProductionProvenanceBlocked,
    verify_design_freeze_approval_blob,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8d_transport import (
    DurableV8DAuditStore,
    READINESS_STAGES,
    SENTINEL_END_EXCLUSIVE,
    SENTINEL_INDICES,
    SENTINEL_START,
    V8DRequestPlan,
    V8DTransportBlocked,
    execute_v8d_stage,
    make_request_fingerprint,
    sha256_url,
)


class V8DReadinessBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _wrap(error: BaseException) -> V8DReadinessBlocked:
    reason = getattr(error, "reason", None)
    return V8DReadinessBlocked(reason if isinstance(reason, str) else "V8D_READINESS_PROVENANCE_CHECK_FAILED")


def execute_transport_readiness_probe(*, stage: str, request_factory: Callable[[int], V8DRequestPlan],
                                      audit_root: str | Path, reviewed_implementation_commit: str,
                                      gate_binding: Any, sleep_fn: Callable[[float], None]) -> dict[str, Any]:
    if stage not in READINESS_STAGES:
        raise V8DReadinessBlocked("V8D_READINESS_STAGE_INVALID")
    try:
        return execute_v8d_stage(
            stage=stage,
            request_factory=request_factory,
            store=DurableV8DAuditStore(audit_root),
            reviewed_implementation_commit=reviewed_implementation_commit,
            gate_binding=gate_binding,
            window_start=SENTINEL_START,
            window_end_exclusive=SENTINEL_END_EXCLUSIVE,
            request_count=len(SENTINEL_INDICES),
            sleep_fn=sleep_fn,
        )
    except V8DTransportBlocked as error:
        raise V8DReadinessBlocked(error.reason) from error


def _execute_production_transport_readiness(
    *,
    stage: str,
    human_authorization_identity: str,
    request_factory: Callable[[int], V8DRequestPlan],
    audit_root: str | Path,
    repository_root: str | Path = CANONICAL_REPOSITORY_ROOT,
    consumption_state_root: str | Path = CANONICAL_CONSUMPTION_STATE_ROOT,
    git_commit_resolver: Callable[[], str] | None = None,
    frozen_design_object_verifier: Callable[[], None] | None = None,
    design_freeze_approval_verifier: Callable[[str], None] | None = None,
    reviewed_implementation_binder: Callable[[str], Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    if stage not in READINESS_STAGES:
        raise V8DReadinessBlocked("V8D_READINESS_STAGE_INVALID")
    if not isinstance(human_authorization_identity, str) or not human_authorization_identity:
        raise V8DReadinessBlocked("V8D_READINESS_HUMAN_AUTHORIZATION_IDENTITY_REQUIRED")

    resolver = git_commit_resolver or (lambda: resolve_verified_v8d_production_git_commit(repository_root))
    try:
        verified_head = resolver()
    except V8DGitProvenanceBlocked as error:
        raise _wrap(error) from error

    frozen_verifier = frozen_design_object_verifier or (lambda: verify_frozen_design_object(repository_root))
    approval_verifier = design_freeze_approval_verifier or (lambda head: verify_design_freeze_approval_blob(repository_root, head))
    try:
        frozen_verifier()
        approval_verifier(verified_head)
    except V8DProductionProvenanceBlocked as error:
        raise _wrap(error) from error

    binder = reviewed_implementation_binder or (lambda head: verify_reviewed_implementation_binding(repository_root, head))
    try:
        review_binding = binder(verified_head)
    except V8DProductionProvenanceBlocked as error:
        raise _wrap(error) from error
    reviewed_commit = review_binding["reviewed_implementation_git_commit"]

    try:
        binding = consume_gate_and_bind(
            consumption_state_root,
            logical_stage=stage,
            v8d_frozen_design_commit=EXPECTED_V8D_FROZEN_DESIGN_COMMIT,
            reviewed_production_implementation_commit=reviewed_commit,
            raw_authorization_identity=human_authorization_identity,
            clock=clock,
        )
    except V8DHumanGateConsumptionBlocked as error:
        raise V8DReadinessBlocked(error.reason) from error

    gate_binding = {
        "human_gate": binding.human_gate,
        "gate_receipt_key_sha256": binding.gate_receipt_key_sha256,
        "gate_receipt_bytes_sha256": binding.gate_receipt_bytes_sha256,
        "authorization_identity_sha256": binding.authorization_identity_sha256,
    }

    return execute_transport_readiness_probe(
        stage=stage,
        request_factory=request_factory,
        audit_root=audit_root,
        reviewed_implementation_commit=reviewed_commit,
        gate_binding=gate_binding,
        sleep_fn=sleep_fn,
    )


def execute_t1c_transport_readiness_production(
    *, human_authorization_identity: str, request_factory: Callable[[int], V8DRequestPlan], audit_root: str | Path,
) -> dict[str, Any]:
    """Sole production entrypoint for `T1C_TRANSPORT_READINESS_HUMAN_GATE`."""
    return _execute_production_transport_readiness(
        stage="T1C_TRANSPORT_READINESS", human_authorization_identity=human_authorization_identity,
        request_factory=request_factory, audit_root=audit_root,
    )


def execute_t2_transport_readiness_production(
    *, human_authorization_identity: str, request_factory: Callable[[int], V8DRequestPlan], audit_root: str | Path,
) -> dict[str, Any]:
    """Sole production entrypoint for `T2_TRANSPORT_READINESS_HUMAN_GATE`."""
    return _execute_production_transport_readiness(
        stage="T2_TRANSPORT_READINESS", human_authorization_identity=human_authorization_identity,
        request_factory=request_factory, audit_root=audit_root,
    )


def synthetic_request_plan(*, stage: str, coordinate: int, url: str,
                           request_fn: Callable[[], Any],
                           request_parameters: dict[str, Any] | None = None) -> V8DRequestPlan:
    """Build a plan for tests or a caller that already owns request creation.

    Only the URL digest is retained; the URL itself is never written to an
    audit or aggregate artifact.
    """
    block = "T1C" if stage.startswith("T1C") else "T2"
    fingerprint = make_request_fingerprint(
        logical_stage=stage,
        logical_block=block,
        logical_coordinate=coordinate,
        window_start=SENTINEL_START,
        window_end_exclusive=SENTINEL_END_EXCLUSIVE,
        request_parameters=request_parameters,
    )
    return V8DRequestPlan(request_fn=request_fn, request_fingerprint=fingerprint, request_url_sha256=sha256_url(url))


__all__ = [
    "V8DReadinessBlocked",
    "execute_t1c_transport_readiness_production",
    "execute_t2_transport_readiness_production",
    "execute_transport_readiness_probe",
    "synthetic_request_plan",
]
