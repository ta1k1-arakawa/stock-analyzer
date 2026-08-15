"""V8D raw-acquisition transport catch path, without gate execution.

The caller supplies the already-authorized production request factory.  This
stage implementation itself performs no private reads, no identity access,
and no network call until the injected factory is invoked by the caller.
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
            window_start=request_start,
            window_end_exclusive=request_end_exclusive,
            request_count=request_count,
            sleep_fn=sleep_fn,
        )
    except V8DTransportBlocked as error:
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


__all__ = ["V8DAcquisitionBlocked", "execute_raw_acquisition_transport", "synthetic_request_plan"]
