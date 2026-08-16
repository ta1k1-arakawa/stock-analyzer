"""V8D raw-acquisition transport catch path, plus the fail-closed production
raw-acquisition entrypoints
(`V8D_PROD_HIGH_1B_GATE_CONSUMPTION_RECEIPT_BINDING`).

``execute_raw_acquisition_transport`` remains transport-only: it requires an
already-derived ``reviewed_implementation_commit`` and an already-consumed
``gate_binding`` supplied by the caller, and performs no private reads, no
identity access, no provenance resolution, and no gate consumption of its
own -- it is the shared synthetic-injectable catch path used by both tests
and the production entrypoints below.

``execute_t1c_raw_acquisition_production`` / ``execute_t2_raw_acquisition_
production`` are the sole production entrypoints. See
`src.v8d_readiness`'s equivalent docstring for the exact fail-closed
ordering they follow: verified V8D Git HEAD -> frozen design/freeze
approval -> HIGH-1A reviewed-implementation binding -> durable one-shot
gate consumption -> transport stage. No caller-supplied authority-bearing
``reviewed_implementation_commit`` is ever accepted.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

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
    ACQUISITION_STAGES,
    DurableV8DAuditStore,
    V8DRequestPlan,
    V8DTransportBlocked,
    execute_v8d_stage,
    make_request_fingerprint,
    sha256_url,
)


class V8DAcquisitionBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _wrap(error: BaseException) -> V8DAcquisitionBlocked:
    reason = getattr(error, "reason", None)
    return V8DAcquisitionBlocked(reason if isinstance(reason, str) else "V8D_ACQUISITION_PROVENANCE_CHECK_FAILED")


def execute_raw_acquisition_transport(*, stage: str, request_factory: Callable[[int], V8DRequestPlan],
                                       audit_root: str | Path, reviewed_implementation_commit: str,
                                       gate_binding: Any,
                                       request_start: str, request_end_exclusive: str,
                                       request_count: int,
                                       sleep_fn: Callable[[float], None]) -> dict[str, Any]:
    if stage not in ACQUISITION_STAGES:
        raise V8DAcquisitionBlocked("V8D_ACQUISITION_STAGE_INVALID")
    try:
        return execute_v8d_stage(
            stage=stage,
            request_factory=request_factory,
            store=DurableV8DAuditStore(audit_root),
            reviewed_implementation_commit=reviewed_implementation_commit,
            gate_binding=gate_binding,
            window_start=request_start,
            window_end_exclusive=request_end_exclusive,
            request_count=request_count,
            sleep_fn=sleep_fn,
        )
    except V8DTransportBlocked as error:
        raise V8DAcquisitionBlocked(error.reason) from error


def _execute_production_raw_acquisition(
    *,
    stage: str,
    human_authorization_identity: str,
    request_factory: Callable[[int], V8DRequestPlan],
    audit_root: str | Path,
    request_start: str,
    request_end_exclusive: str,
    request_count: int,
    repository_root: str | Path = CANONICAL_REPOSITORY_ROOT,
    consumption_state_root: str | Path = CANONICAL_CONSUMPTION_STATE_ROOT,
    git_commit_resolver: Callable[[], str] | None = None,
    frozen_design_object_verifier: Callable[[], None] | None = None,
    design_freeze_approval_verifier: Callable[[str], None] | None = None,
    reviewed_implementation_binder: Callable[[str], Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    if stage not in ACQUISITION_STAGES:
        raise V8DAcquisitionBlocked("V8D_ACQUISITION_STAGE_INVALID")
    if not isinstance(human_authorization_identity, str) or not human_authorization_identity:
        raise V8DAcquisitionBlocked("V8D_ACQUISITION_HUMAN_AUTHORIZATION_IDENTITY_REQUIRED")

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
        raise V8DAcquisitionBlocked(error.reason) from error

    gate_binding = {
        "human_gate": binding.human_gate,
        "gate_receipt_key_sha256": binding.gate_receipt_key_sha256,
        "gate_receipt_bytes_sha256": binding.gate_receipt_bytes_sha256,
        "authorization_identity_sha256": binding.authorization_identity_sha256,
    }

    return execute_raw_acquisition_transport(
        stage=stage,
        request_factory=request_factory,
        audit_root=audit_root,
        reviewed_implementation_commit=reviewed_commit,
        gate_binding=gate_binding,
        request_start=request_start,
        request_end_exclusive=request_end_exclusive,
        request_count=request_count,
        sleep_fn=sleep_fn,
    )


def execute_t1c_raw_acquisition_production(
    *, human_authorization_identity: str, partition_manifest_path: str | Path,
    t1c_allocation_artifact_path: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    """Fixed production entrypoint for the T1C raw-acquisition stage."""
    from src.v8d_acquisition_engine import V8DAcquisitionEngineBlocked, _execute_fixed_production_acquisition

    try:
        return _execute_fixed_production_acquisition(
            stage="T1C_RAW_ACQUISITION",
            human_authorization_identity=human_authorization_identity,
            partition_manifest_path=partition_manifest_path,
            t1c_allocation_artifact_path=t1c_allocation_artifact_path,
            output_root=output_root,
        )
    except V8DAcquisitionEngineBlocked as error:
        raise V8DAcquisitionBlocked(error.reason) from error


def execute_t2_raw_acquisition_production(
    *, human_authorization_identity: str, partition_manifest_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Fixed production entrypoint for the T2 raw-acquisition stage."""
    from src.v8d_acquisition_engine import V8DAcquisitionEngineBlocked, _execute_fixed_production_acquisition

    try:
        return _execute_fixed_production_acquisition(
            stage="T2_RAW_ACQUISITION",
            human_authorization_identity=human_authorization_identity,
            partition_manifest_path=partition_manifest_path,
            output_root=output_root,
        )
    except V8DAcquisitionEngineBlocked as error:
        raise V8DAcquisitionBlocked(error.reason) from error


def synthetic_request_plan(*, stage: str, coordinate: int, request_start: str,
                           request_end_exclusive: str, url: str,
                           request_fn: Callable[[], Any],
                           request_parameters: dict[str, Any] | None = None) -> V8DRequestPlan:
    block = "T1C" if stage.startswith("T1C") else "T2"
    fingerprint = make_request_fingerprint(
        logical_stage=stage,
        logical_block=block,
        logical_coordinate=coordinate,
        window_start=request_start,
        window_end_exclusive=request_end_exclusive,
        request_parameters=request_parameters,
    )
    return V8DRequestPlan(request_fn=request_fn, request_fingerprint=fingerprint, request_url_sha256=sha256_url(url))


__all__ = [
    "V8DAcquisitionBlocked",
    "execute_raw_acquisition_transport",
    "execute_t1c_raw_acquisition_production",
    "execute_t2_raw_acquisition_production",
    "synthetic_request_plan",
]
