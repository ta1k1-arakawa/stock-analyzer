"""V8D synthetic-injectable transport readiness catch paths.

The public functions require an injected request factory.  They do not
consume readiness authorization and do not perform real execution in this
implementation stage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Any

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


def execute_transport_readiness_probe(*, stage: str, request_factory: Callable[[int], V8DRequestPlan],
                                      audit_root: str | Path, reviewed_implementation_commit: str,
                                      sleep_fn: Callable[[float], None]) -> dict[str, Any]:
    if stage not in READINESS_STAGES:
        raise V8DReadinessBlocked("V8D_READINESS_STAGE_INVALID")
    try:
        return execute_v8d_stage(
            stage=stage,
            request_factory=request_factory,
            store=DurableV8DAuditStore(audit_root),
            reviewed_implementation_commit=reviewed_implementation_commit,
            window_start=SENTINEL_START,
            window_end_exclusive=SENTINEL_END_EXCLUSIVE,
            request_count=len(SENTINEL_INDICES),
            sleep_fn=sleep_fn,
        )
    except V8DTransportBlocked as error:
        raise V8DReadinessBlocked(error.reason) from error


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


__all__ = ["V8DReadinessBlocked", "execute_transport_readiness_probe", "synthetic_request_plan"]
