"""V8D raw-acquisition transport catch path and fixed production entrypoints.

``execute_raw_acquisition_transport`` remains transport-only: it requires an
already-derived ``reviewed_implementation_commit`` and an already-consumed
``gate_binding`` supplied by the caller, and performs no private reads, no
identity access, no provenance resolution, and no gate consumption of its
own -- it is the shared synthetic-injectable catch path.

``execute_t1c_raw_acquisition_production`` / ``execute_t2_raw_acquisition_
production`` are the sole production entrypoints. The fixed implementation
in ``src.v8d_acquisition_engine`` is the sole raw-acquisition gate-consuming
path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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
