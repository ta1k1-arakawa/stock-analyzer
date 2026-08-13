"""Production-gated `T1B` allocation boundary (§11.3.B, §12's
`EXECUTE_T1B_ALLOCATION` gate).

This module is **not executed** by this implementation phase.
`ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B` (§12) has not occurred; the
only entrypoint below deliberately requires an exact, repository-fixed
confirmation literal precisely so it cannot be invoked by accident, exactly
mirroring `V8_PRODUCTION_ACQUIRE_T1`/`V8_PRODUCTION_ACQUIRE_T2`'s existing
convention in `scripts/acquire_v8_historical.py`. Every test exercising
this module uses a synthetic parent `T_spare` fixture and a synthetic
partition manifest, never the real private V8 partition. Importing this
module performs no I/O and no network access of any kind; it never
imports `src/v7_yahoo_collector.py`.
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
from src.v8b_git_provenance import (
    V8BGitProvenanceBlocked,
    resolve_verified_v8b_production_git_commit,
)
from src.v8b_production_provenance import (
    EXPECTED_PARENT_T_SPARE_TICKER_COUNT,
    EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256,
    EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
    V8BProductionProvenanceBlocked,
    V8_DESIGN_COMMIT,
    read_and_verify_design_freeze_approval,
    read_and_verify_v8_trusted_partition_anchor,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8b_allocation import (
    V8BAllocationBlocked,
    build_t1b_allocation_artifact,
    canonical_json_bytes,
    public_allocation_summary,
)

# Frozen production confirmation literal (§12's ONE_TIME_HUMAN_
# AUTHORIZATION_TO_ALLOCATE_T1B). Mirrors this repository's existing,
# already-reviewed "operator must type this exact fixed token" convention
# (`scripts/acquire_v8_historical.py --confirmation V8_PRODUCTION_ACQUIRE_
# T1`) -- a mechanical anti-fat-finger safeguard, not a methodology
# decision: no threshold, partition, or selection-rule content is encoded
# in it.
ALLOCATION_CONFIRMATION = "V8B_PRODUCTION_ALLOCATE_T1B"

CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class V8BT1BAllocatorBlocked(RuntimeError):
    """Fail-closed production T1B allocation error.

    ``authorization_consumed`` is ``False`` for every pre-private-access
    failure (confirmation, provenance, freeze, review, anchor) and
    ``True`` for any failure at or after the first private partition read
    -- a safe boolean, never a ticker or path (round-2 finding HIGH-1).
    """

    def __init__(self, reason: str, *, authorization_consumed: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.authorization_consumed = authorization_consumed


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8BT1BAllocatorBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8BT1BAllocatorBlocked(missing_reason)
    if isinstance(reason, str):
        return V8BT1BAllocatorBlocked(reason)
    return V8BT1BAllocatorBlocked("PROVENANCE_CHECK_FAILED")


def _write_allocation_artifact_once(destination: Path, artifact_bytes: bytes) -> Path:
    """Atomically publish, never replacing an existing destination.

    Mirrors `src/v8_partition.py::write_partition_manifest_once`'s proven
    fsync-then-``os.link`` no-overwrite pattern (read-only precedent, not
    imported/modified).
    """
    if destination.exists():
        raise V8BT1BAllocatorBlocked("V8B_ALLOCATION_ARTIFACT_ALREADY_EXISTS")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8BT1BAllocatorBlocked("OUTPUT_PATH_PARENT_INVALID") from error
    if not destination.parent.is_dir():
        raise V8BT1BAllocatorBlocked("OUTPUT_PATH_PARENT_INVALID")
    staging = destination.parent / (destination.name + ".staging-" + os.urandom(8).hex())
    try:
        try:
            with open(staging, "wb") as stream:
                stream.write(artifact_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            # Round-3 repeat finding HIGH-3: never let a raw OSError
            # (which could carry the private staging/output path) escape.
            raise V8BT1BAllocatorBlocked("V8B_ALLOCATION_ARTIFACT_STAGING_WRITE_FAILED") from error
        try:
            os.link(str(staging), str(destination))
        except FileExistsError as error:
            raise V8BT1BAllocatorBlocked("V8B_ALLOCATION_ARTIFACT_ALREADY_EXISTS") from error
        except OSError as error:
            raise V8BT1BAllocatorBlocked("V8B_ALLOCATION_ARTIFACT_ATOMIC_PUBLISH_FAILED") from error
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass
    return destination


def allocate_t1b_production(
    *,
    confirmation: str,
    partition_manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Sole production entrypoint. **Not executed** by this implementation
    phase -- see module docstring. Returns only a safe public summary
    (hashes/counts, never ticker identities); the private artifact itself
    is written to ``output_path``, never returned or logged in full.
    """
    return _allocate_t1b_production_with_dependencies(
        confirmation=confirmation,
        partition_manifest_path=partition_manifest_path,
        output_path=output_path,
        git_commit_resolver=lambda: resolve_verified_v8b_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        design_freeze_approval_reader=lambda head: read_and_verify_design_freeze_approval(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        frozen_design_object_verifier=lambda: verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(
            CANONICAL_REPOSITORY_ROOT, head
        ),
        anchor_reader=lambda head: read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, head),
        clock=lambda: datetime.now(timezone.utc),
    )


def _allocate_t1b_production_with_dependencies(
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
) -> dict[str, Any]:
    """Private fake-only seam. No OHLCV, no network, no retry: exactly one
    deterministic attempt per call -- either it succeeds once or raises."""
    # (0) explicit, exact allocation-gate confirmation token
    if confirmation != ALLOCATION_CONFIRMATION:
        raise V8BT1BAllocatorBlocked("V8B_ALLOCATION_CONFIRMATION_INVALID")

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
        review_binding = reviewed_implementation_binder(verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error, "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    reviewed_commit = review_binding["reviewed_implementation_git_commit"]

    # (4) original immutable V8 authority (exact anchor blob, §11.1)
    try:
        anchor = anchor_reader(verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error) from error
    if anchor["authorization_status"] != "AUTHORIZED":
        raise V8BT1BAllocatorBlocked("TRUSTED_PARTITION_NOT_AUTHORIZED")

    # (5) onward: the first private action (reading the private V8 partition
    # manifest) is about to begin -- authorization is consumed as of this
    # exact point, regardless of what happens next. No automatic or manual
    # retry is authorized by this implementation: this function makes
    # exactly one attempt per call.
    try:
        partition_manifest = read_partition_manifest(partition_manifest_path)

        manifest_sha = partition_manifest["manifest_sha256"]
        if manifest_sha != anchor["authorized_partition_manifest_sha256"]:
            raise V8BT1BAllocatorBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
        partition_implementation_commit = partition_manifest["partition_implementation_git_commit"]
        if partition_implementation_commit != anchor["authorized_partition_implementation_git_commit"]:
            raise V8BT1BAllocatorBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")
        if partition_manifest["study_name"] != V8_STUDY_NAME:
            raise V8BT1BAllocatorBlocked("PARTITION_MANIFEST_STUDY_NAME_MISMATCH")
        if partition_manifest["design_commit"] != V8_DESIGN_COMMIT:
            raise V8BT1BAllocatorBlocked("PARTITION_MANIFEST_DESIGN_COMMIT_MISMATCH")

        assignments = partition_manifest["block_assignments"]
        if not isinstance(assignments, Mapping) or "T_spare" not in assignments:
            raise V8BT1BAllocatorBlocked("PARTITION_BLOCK_ASSIGNMENT_MISSING:T_SPARE")
        parent_assignment = assignments["T_spare"]
        if not isinstance(parent_assignment, list):
            raise V8BT1BAllocatorBlocked("PARTITION_BLOCK_ASSIGNMENT_INVALID:T_SPARE")
        parent_tickers = list(parent_assignment)

        # (6) exact frozen parent T_spare count/hash pin (HIGH-7)
        if len(parent_tickers) != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
            raise V8BT1BAllocatorBlocked("PARENT_T_SPARE_TICKER_COUNT_INVALID")
        computed_hash = ticker_list_sha256(parent_tickers)
        if computed_hash != EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256:
            raise V8BT1BAllocatorBlocked("PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH")
        if computed_hash != partition_manifest["t_spare_ticker_list_sha256"]:
            raise V8BT1BAllocatorBlocked("PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH")

        # (7) output path safety (private, outside repository)
        destination = require_absolute_output_path_outside_repository(output_path, CANONICAL_REPOSITORY_ROOT)

        # (8) deterministic §4 zero-offset slice -- no OHLCV, no network
        try:
            artifact = build_t1b_allocation_artifact(
                parent_t_spare_tickers=parent_tickers,
                parent_v8_partition_manifest_sha256=manifest_sha,
                parent_v8_partition_implementation_commit=partition_implementation_commit,
                parent_t_spare_ticker_list_sha256=computed_hash,
                v8b_frozen_design_commit=EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
                v8b_allocation_implementation_commit=reviewed_commit,
                clock=clock,
            )
        except V8BAllocationBlocked as error:
            raise V8BT1BAllocatorBlocked("V8B_ALLOCATION_ARTIFACT_BUILD_FAILED:" + error.reason) from error

        _write_allocation_artifact_once(destination, canonical_json_bytes(artifact))
        return public_allocation_summary(artifact)
    except V8PartitionBlocked as error:
        raise V8BT1BAllocatorBlocked(error.reason, authorization_consumed=True) from error
    except V8BT1BAllocatorBlocked as error:
        error.authorization_consumed = True
        raise


__all__ = [
    "ALLOCATION_CONFIRMATION",
    "V8BT1BAllocatorBlocked",
    "allocate_t1b_production",
]
