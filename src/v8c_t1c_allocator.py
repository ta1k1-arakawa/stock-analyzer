"""Production-gated `T1C` allocation boundary (§2, §9.2, §12's
`EXECUTE_T1C_ALLOCATION` gate).

This module is **not executed** by this implementation phase.
`ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1C` has not occurred; the only
entrypoint below deliberately requires an exact, repository-fixed
confirmation literal precisely so it cannot be invoked by accident, mirroring
this repository's existing convention (`src.v8b_t1b_allocator`'s
``V8B_PRODUCTION_ALLOCATE_T1B``). Every test exercising this module uses a
synthetic parent `T_spare` fixture and a synthetic partition manifest, never
the real private V8 partition. Importing this module performs no I/O and no
network access of any kind; it never imports `src/v7_yahoo_collector.py`.

`ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1C`'s one-shot consumption is
durable, not merely an in-memory boolean --
`src.v8c_human_gate_consumption.consume_gate_once` fsync's a receipt
strictly before the private partition-manifest read, and every call first
checks `require_gate_not_yet_consumed` before any provenance step. A second
call, a new process, or a restart, using the same authorization under the
same frozen design commit, BLOCKs before the private read -- it never
repeats the allocation.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v8_partition import (
    STUDY_NAME as V8_STUDY_NAME,
    V8PartitionBlocked,
    read_partition_manifest,
    require_absolute_output_path_outside_repository,
    ticker_list_sha256,
)
from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    resolve_verified_v8c_production_git_commit,
)
from src.v8c_production_provenance import (
    EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
    EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    V8CProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    read_and_verify_v8_trusted_partition_anchor,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8c_t1c_allocation import (
    V8CAllocationBlocked,
    build_t1c_allocation_artifact,
    canonical_json_bytes,
    public_allocation_summary,
)
from src.v8c_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_ALLOCATE_T1C,
    V8CHumanGateConsumptionBlocked,
    consume_gate_once,
    require_gate_not_yet_consumed,
)

V8_DESIGN_COMMIT = "c414d3191cba356734d7ed08bdf1abc7d51fc384"

ALLOCATION_CONFIRMATION = "V8C_PRODUCTION_ALLOCATE_T1C"


class V8CT1CAllocatorBlocked(RuntimeError):
    """Fail-closed production T1C allocation error.

    ``authorization_consumed`` is ``False`` for every pre-private-access
    failure and ``True`` for any failure at or after the first private
    partition read -- a safe boolean, never a ticker or path.
    """

    def __init__(self, reason: str, *, authorization_consumed: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.authorization_consumed = authorization_consumed


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8CT1CAllocatorBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8CT1CAllocatorBlocked(missing_reason)
    if isinstance(reason, str):
        return V8CT1CAllocatorBlocked(reason)
    return V8CT1CAllocatorBlocked("PROVENANCE_CHECK_FAILED")


def _write_allocation_artifact_once(destination: Path, artifact_bytes: bytes) -> Path:
    """Atomically publish, never replacing an existing destination."""
    if destination.exists():
        raise V8CT1CAllocatorBlocked("V8C_ALLOCATION_ARTIFACT_ALREADY_EXISTS")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8CT1CAllocatorBlocked("OUTPUT_PATH_PARENT_INVALID") from error
    if not destination.parent.is_dir():
        raise V8CT1CAllocatorBlocked("OUTPUT_PATH_PARENT_INVALID")
    staging = destination.parent / (destination.name + ".staging-" + os.urandom(8).hex())
    try:
        try:
            with open(staging, "wb") as stream:
                stream.write(artifact_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise V8CT1CAllocatorBlocked("V8C_ALLOCATION_ARTIFACT_STAGING_WRITE_FAILED") from error
        try:
            os.link(str(staging), str(destination))
        except FileExistsError as error:
            raise V8CT1CAllocatorBlocked("V8C_ALLOCATION_ARTIFACT_ALREADY_EXISTS") from error
        except OSError as error:
            raise V8CT1CAllocatorBlocked("V8C_ALLOCATION_ARTIFACT_ATOMIC_PUBLISH_FAILED") from error
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass
    return destination


def allocate_t1c_production(
    *,
    confirmation: str,
    partition_manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Sole production entrypoint. **Not executed** by this implementation
    phase -- see module docstring."""
    return _allocate_t1c_production_with_dependencies(
        confirmation=confirmation,
        partition_manifest_path=partition_manifest_path,
        output_path=output_path,
        git_commit_resolver=lambda: resolve_verified_v8c_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        design_freeze_approval_reader=lambda head: read_and_verify_design_freeze_approval(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        frozen_design_object_verifier=lambda: verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        anchor_reader=lambda head: read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, head),
        clock=lambda: datetime.now(timezone.utc),
        consumption_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
    )


def _allocate_t1c_production_with_dependencies(
    *,
    confirmation: str,
    partition_manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    git_commit_resolver: Callable[[], str],
    design_freeze_approval_reader: Callable[[str], Mapping[str, Any]],
    frozen_design_object_verifier: Callable[[], None],
    reviewed_implementation_binder: Callable[[str], Mapping[str, Any]],
    anchor_reader: Callable[[str], Mapping[str, Any]],
    clock: Callable[[], datetime],
    consumption_state_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Private fake-only seam. No OHLCV, no network, no retry: exactly one
    deterministic attempt per call -- either it succeeds once or raises."""
    if confirmation != ALLOCATION_CONFIRMATION:
        raise V8CT1CAllocatorBlocked("V8C_ALLOCATION_CONFIRMATION_INVALID")

    try:
        require_gate_not_yet_consumed(consumption_state_root, GATE_ALLOCATE_T1C, EXPECTED_V8C_FROZEN_DESIGN_COMMIT)
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CT1CAllocatorBlocked(error.reason) from error

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
        review_binding = reviewed_implementation_binder(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    reviewed_commit = review_binding["reviewed_implementation_git_commit"]

    try:
        anchor = anchor_reader(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error
    if anchor["authorization_status"] != "AUTHORIZED":
        raise V8CT1CAllocatorBlocked("TRUSTED_PARTITION_NOT_AUTHORIZED")

    consumed = False
    try:
        try:
            consume_gate_once(
                consumption_state_root, GATE_ALLOCATE_T1C, EXPECTED_V8C_FROZEN_DESIGN_COMMIT, clock=clock
            )
        except V8CHumanGateConsumptionBlocked as error:
            raise V8CT1CAllocatorBlocked(error.reason) from error
        consumed = True

        partition_manifest = read_partition_manifest(partition_manifest_path)

        manifest_sha = partition_manifest["manifest_sha256"]
        if manifest_sha != anchor["authorized_partition_manifest_sha256"]:
            raise V8CT1CAllocatorBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
        partition_implementation_commit = partition_manifest["partition_implementation_git_commit"]
        if partition_implementation_commit != anchor["authorized_partition_implementation_git_commit"]:
            raise V8CT1CAllocatorBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")
        if partition_manifest["study_name"] != V8_STUDY_NAME:
            raise V8CT1CAllocatorBlocked("PARTITION_MANIFEST_STUDY_NAME_MISMATCH")
        if partition_manifest["design_commit"] != V8_DESIGN_COMMIT:
            raise V8CT1CAllocatorBlocked("PARTITION_MANIFEST_DESIGN_COMMIT_MISMATCH")

        assignments = partition_manifest["block_assignments"]
        if not isinstance(assignments, Mapping) or "T_spare" not in assignments:
            raise V8CT1CAllocatorBlocked("PARTITION_BLOCK_ASSIGNMENT_MISSING:T_SPARE")
        parent_assignment = assignments["T_spare"]
        if not isinstance(parent_assignment, list):
            raise V8CT1CAllocatorBlocked("PARTITION_BLOCK_ASSIGNMENT_INVALID:T_SPARE")
        parent_tickers = list(parent_assignment)

        if len(parent_tickers) != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
            raise V8CT1CAllocatorBlocked("PARENT_T_SPARE_TICKER_COUNT_INVALID")
        computed_hash = ticker_list_sha256(parent_tickers)
        if computed_hash != EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256:
            raise V8CT1CAllocatorBlocked("PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH")
        if computed_hash != partition_manifest["t_spare_ticker_list_sha256"]:
            raise V8CT1CAllocatorBlocked("PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH")

        destination = require_absolute_output_path_outside_repository(output_path, CANONICAL_REPOSITORY_ROOT)

        try:
            artifact = build_t1c_allocation_artifact(
                parent_t_spare_tickers=parent_tickers,
                parent_v8_partition_manifest_sha256=manifest_sha,
                parent_v8_partition_implementation_commit=partition_implementation_commit,
                parent_t_spare_ticker_list_sha256=computed_hash,
                v8c_frozen_design_commit=EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
                v8c_allocation_implementation_commit=reviewed_commit,
                clock=clock,
            )
        except V8CAllocationBlocked as error:
            raise V8CT1CAllocatorBlocked("V8C_ALLOCATION_ARTIFACT_BUILD_FAILED:" + error.reason) from error

        _write_allocation_artifact_once(destination, canonical_json_bytes(artifact))
        return public_allocation_summary(artifact)
    except V8PartitionBlocked as error:
        raise V8CT1CAllocatorBlocked(error.reason, authorization_consumed=consumed) from error
    except V8CT1CAllocatorBlocked as error:
        error.authorization_consumed = consumed
        raise


__all__ = [
    "ALLOCATION_CONFIRMATION",
    "V8CT1CAllocatorBlocked",
    "allocate_t1c_production",
]
