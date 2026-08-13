"""`READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION` -- §11.4 invariants.

Independently verifies every invariant `V8B_HISTORICAL_RESEARCH_DESIGN_
DRAFT.md` §11.4 requires against a *concrete* `T1B` allocation artifact
(`src/v8b_allocation.py`'s output) -- not merely against the implementation
code that produced it. Any single invariant failing is `BLOCK`: no trust
pin may be created and no acquisition may proceed (§11.4, §12's
`READ_ONLY_T1B_ALLOCATION_ARTIFACT_VERIFICATION` gate).

Two, and only two, ways to call this module (round-3 finding MEDIUM-2,
tightened further in round 3's repeat review):

- ``_verify_t1b_allocation_artifact`` -- the **private/pure invariant
  evaluator** (round-3 repeat finding MEDIUM-2: not part of the production
  public surface -- fake/synthetic tests import and call it directly as an
  internal helper). It performs no I/O and no network access, and accepts
  every trusted comparison input (the parent `T_spare`/`T0`/old-`T1`/`T2`/
  `T3` ticker lists, the expected parent hash, the expected frozen design
  commit) directly from the caller -- exactly the seam fake/synthetic
  tests need. It is not, by itself, a safe *production* trust root: a
  caller could supply a favorable but wrong mapping.
- ``resolve_and_verify_t1b_allocation_artifact`` -- the sole **public
  production boundary**. It resolves verified V8B Git HEAD from the one
  fixed, non-overridable production repository root (round-3 repeat
  finding HIGH-1: no ``repository_root`` parameter exists on this public
  function; a private DI-testable variant carries that parameter for
  fake/synthetic tests only); verifies the frozen design object, freeze
  approval, and reviewed-implementation binding; verifies the exact
  immutable V8 anchor; reads the private V8 partition manifest and the
  private `T1B` allocation artifact from caller-supplied paths (private
  data, so a path parameter remains appropriate, exactly like
  `src/v8b_t1b_allocator.py` and the `T1B` branch of
  `src/v8b_historical_acquisition.py`); derives every block's ticker
  assignment internally from that one verified manifest; checks the
  artifact's parent manifest SHA/implementation-commit and the exact
  frozen parent `T_spare` count/hash; checks the artifact's
  ``v8b_allocation_implementation_commit`` equals the reviewed
  implementation commit; and only then invokes the pure evaluator above.
  It never accepts a caller-supplied expected hash/commit/repository-root
  as the trust root, performs no network access, and its return value is
  the pure evaluator's own safe aggregate result -- hashes/counts/status
  only, never a ticker identity or private path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.v8_partition import (
    STUDY_NAME as V8_STUDY_NAME,
    V8PartitionBlocked,
    read_partition_manifest,
)
from src.v8b_allocation import (
    ALLOCATION_ARTIFACT_FIELDS,
    ARTIFACT_ROLE,
    LOGICAL_BLOCK,
    PARENT_STUDY_NAME,
    SCHEMA_VERSION as ALLOCATION_SCHEMA_VERSION,
    SELECTION_RULE_ID,
    SELECTION_RULE_TEXT,
    STUDY_NAME as ALLOCATION_STUDY_NAME,
    T1B_OFFSET_WITHIN_PARENT_T_SPARE,
    T1B_SLICE_END_EXCLUSIVE,
    T1B_SLICE_START_INCLUSIVE,
    T1B_TICKER_COUNT,
    V8BAllocationBlocked,
    ticker_list_sha256,
    verify_allocation_artifact_self_hash,
)
from src.v8b_git_provenance import V8BGitProvenanceBlocked, resolve_verified_v8b_production_git_commit
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

CANONICAL_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_BLOCK_KEYS = ("T0", "T1", "T2", "T3", "T_spare")


class V8BAllocationVerificationBlocked(RuntimeError):
    """Fail-closed §11.4 invariant verification error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _verify_t1b_allocation_artifact(
    artifact: Mapping[str, Any],
    *,
    parent_t_spare_tickers: Sequence[str],
    t0_tickers: Sequence[str],
    old_t1_tickers: Sequence[str],
    t2_tickers: Sequence[str],
    t3_tickers: Sequence[str],
    expected_parent_t_spare_ticker_list_sha256: str,
    expected_v8b_frozen_design_commit: str,
) -> dict[str, Any]:
    """Verify every §11.4 invariant; return a safe public PASS summary.

    ``parent_t_spare_tickers``/``t0_tickers``/``old_t1_tickers``/
    ``t2_tickers``/``t3_tickers`` are supplied by the caller from whatever
    already-verified, trusted source it holds them (in production: the
    real, private trusted V8 partition manifest; in tests: synthetic
    fixtures). This function performs no partition/trust-anchor resolution
    of its own -- it only checks the invariants over what it is given.
    """
    try:
        verified = verify_allocation_artifact_self_hash(artifact)
    except V8BAllocationBlocked as error:
        raise V8BAllocationVerificationBlocked("ARTIFACT_SELF_HASH_INVALID:" + error.reason) from error

    # MEDIUM-1 (FINAL_REPEAT finding): exact-bind every trust-bearing
    # allocation-semantics field to its single frozen expected value --
    # never merely "present and self-hash-consistent". A forged artifact
    # that recomputes its own ``artifact_self_hash`` to match a wrong
    # semantic field (e.g. a different ``artifact_role``, a shifted
    # ``t1b_slice_start_inclusive``, or a substituted ``selection_rule_
    # id``) still passes ``verify_allocation_artifact_self_hash`` above --
    # only this exact-value check catches it.
    if verified["schema_version"] != ALLOCATION_SCHEMA_VERSION:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_SCHEMA_VERSION_MISMATCH")
    if verified["study_name"] != ALLOCATION_STUDY_NAME:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_STUDY_NAME_MISMATCH")
    if verified["artifact_role"] != ARTIFACT_ROLE:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_ROLE_MISMATCH")
    if verified["logical_block"] != LOGICAL_BLOCK:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_LOGICAL_BLOCK_MISMATCH")
    if verified["parent_study"] != PARENT_STUDY_NAME:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_PARENT_STUDY_MISMATCH")
    if verified["selection_rule_id"] != SELECTION_RULE_ID:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_SELECTION_RULE_ID_MISMATCH")
    if verified["t1b_offset_within_parent_t_spare"] != T1B_OFFSET_WITHIN_PARENT_T_SPARE:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_T1B_OFFSET_MISMATCH")
    if verified["t1b_slice_start_inclusive"] != T1B_SLICE_START_INCLUSIVE:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_T1B_SLICE_START_MISMATCH")
    if verified["t1b_slice_end_exclusive"] != T1B_SLICE_END_EXCLUSIVE:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_T1B_SLICE_END_MISMATCH")

    parent = list(parent_t_spare_tickers)
    if len(set(parent)) != len(parent):
        raise V8BAllocationVerificationBlocked("PARENT_T_SPARE_DUPLICATE_TICKER")

    # MEDIUM-1: the artifact's own claimed parent_t_spare_ticker_count must
    # exactly equal the caller-supplied (already trust-anchored) parent
    # T_spare sequence's length -- never merely internally self-consistent
    # with the artifact's own (possibly forged) ticker lists.
    if verified["parent_t_spare_ticker_count"] != len(parent):
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_PARENT_T_SPARE_COUNT_MISMATCH")

    t1b = verified["t1b_tickers"]
    remaining = verified["remaining_t_spare_tickers"]

    # T1B = original_parent_T_spare[0:300]; remaining_T_spare = original_parent_T_spare[300:]
    if t1b != parent[:300]:
        raise V8BAllocationVerificationBlocked("T1B_NOT_EXACT_ZERO_OFFSET_SLICE")
    if remaining != parent[300:]:
        raise V8BAllocationVerificationBlocked("REMAINING_T_SPARE_NOT_EXACT_TAIL_SLICE")

    # len(T1B) = 300
    if len(t1b) != T1B_TICKER_COUNT:
        raise V8BAllocationVerificationBlocked("T1B_SIZE_INVALID")

    # len(T1B) + len(remaining_T_spare) = len(original_parent_T_spare)
    if len(t1b) + len(remaining) != len(parent):
        raise V8BAllocationVerificationBlocked("T1B_REMAINING_ACCOUNTING_INVALID")

    t1b_set = set(t1b)
    remaining_set = set(remaining)
    parent_set = set(parent)

    # T1B ∩ remaining_T_spare = ∅
    if t1b_set & remaining_set:
        raise V8BAllocationVerificationBlocked("T1B_REMAINING_NOT_DISJOINT")

    # T1B ∪ remaining_T_spare = original_parent_T_spare
    if (t1b_set | remaining_set) != parent_set:
        raise V8BAllocationVerificationBlocked("T1B_REMAINING_UNION_MISMATCH")

    # T1B disjoint from T0 / old T1 / T2 / T3
    if t1b_set & set(t0_tickers):
        raise V8BAllocationVerificationBlocked("T1B_NOT_DISJOINT_FROM_T0")
    if t1b_set & set(old_t1_tickers):
        raise V8BAllocationVerificationBlocked("T1B_NOT_DISJOINT_FROM_OLD_T1")
    if t1b_set & set(t2_tickers):
        raise V8BAllocationVerificationBlocked("T1B_NOT_DISJOINT_FROM_T2")
    if t1b_set & set(t3_tickers):
        raise V8BAllocationVerificationBlocked("T1B_NOT_DISJOINT_FROM_T3")

    # parent_t_spare_ticker_list_sha256 matches the original trusted V8 partition manifest
    computed_parent_hash = ticker_list_sha256(parent)
    if computed_parent_hash != expected_parent_t_spare_ticker_list_sha256:
        raise V8BAllocationVerificationBlocked("PARENT_T_SPARE_HASH_MISMATCH_TRUSTED_ANCHOR")
    if verified["parent_t_spare_ticker_list_sha256"] != computed_parent_hash:
        raise V8BAllocationVerificationBlocked("PARENT_T_SPARE_HASH_MISMATCH_ARTIFACT")

    # t1b_ticker_list_sha256 / remaining_t_spare_ticker_list_sha256 match the artifact
    if ticker_list_sha256(t1b) != verified["t1b_ticker_list_sha256"]:
        raise V8BAllocationVerificationBlocked("T1B_TICKER_LIST_SHA_MISMATCH")
    if ticker_list_sha256(remaining) != verified["remaining_t_spare_ticker_list_sha256"]:
        raise V8BAllocationVerificationBlocked("REMAINING_T_SPARE_TICKER_LIST_SHA_MISMATCH")

    # selection_rule_canonical_text_or_hash exactly matches the frozen V8B design (§4)
    if verified["selection_rule_canonical_text_or_hash"] != SELECTION_RULE_TEXT:
        raise V8BAllocationVerificationBlocked("SELECTION_RULE_TEXT_MISMATCH")

    # v8b_frozen_design_commit matches the authorized/frozen V8B design
    if verified["v8b_frozen_design_commit"] != expected_v8b_frozen_design_commit:
        raise V8BAllocationVerificationBlocked("V8B_FROZEN_DESIGN_COMMIT_MISMATCH")

    if set(verified) != set(ALLOCATION_ARTIFACT_FIELDS):
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_SCHEMA_INVALID")

    return {
        "result": "PASS",
        "logical_block": verified["logical_block"],
        "study_name": verified["study_name"],
        "parent_t_spare_ticker_count": verified["parent_t_spare_ticker_count"],
        "parent_t_spare_ticker_list_sha256": verified["parent_t_spare_ticker_list_sha256"],
        "t1b_ticker_count": verified["t1b_ticker_count"],
        "t1b_ticker_list_sha256": verified["t1b_ticker_list_sha256"],
        "remaining_t_spare_ticker_count": verified["remaining_t_spare_ticker_count"],
        "remaining_t_spare_ticker_list_sha256": verified["remaining_t_spare_ticker_list_sha256"],
        "artifact_self_hash": verified["artifact_self_hash"],
        "v8b_frozen_design_commit": verified["v8b_frozen_design_commit"],
        "v8b_allocation_implementation_commit": verified["v8b_allocation_implementation_commit"],
        "parent_v8_partition_manifest_sha256": verified["parent_v8_partition_manifest_sha256"],
        "parent_v8_partition_implementation_commit": verified["parent_v8_partition_implementation_commit"],
        "no_membership_choice_based_on_ohlcv_or_data_quality_outcomes": True,
    }


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8BAllocationVerificationBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8BAllocationVerificationBlocked(missing_reason)
    if isinstance(reason, str):
        return V8BAllocationVerificationBlocked(reason)
    return V8BAllocationVerificationBlocked("PROVENANCE_CHECK_FAILED")


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8BAllocationVerificationBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8BAllocationVerificationBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8BAllocationVerificationBlocked(invalid_reason)
    return parsed


def _resolve_and_verify_t1b_allocation_artifact_with_repository_root(
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
    *,
    repository_root,
) -> dict[str, Any]:
    """Private DI-testable implementation -- fake/synthetic tests only, not
    a production API (round-3 repeat finding HIGH-1). ``repository_root``
    is caller-injectable here so fake tests can exercise this ordering
    against a bogus/synthetic repository; the public
    ``resolve_and_verify_t1b_allocation_artifact`` below is the only
    production entrypoint, and it always passes
    ``CANONICAL_REPOSITORY_ROOT`` -- never a caller-suppliable value. See
    module docstring for the full ordering.

    ``allocation_artifact_path``/``partition_manifest_path`` are private
    data, so caller-supplied paths remain appropriate here -- exactly the
    same convention `src/v8b_t1b_allocator.py` and the `T1B` branch of
    `src/v8b_historical_acquisition.py` already use. Every *trust* value
    (the reviewed implementation commit, the immutable V8 anchor, the
    frozen parent `T_spare` count/hash) is derived from verified Git
    objects or this module's own frozen constants, never from the artifact
    or manifest's own self-reported fields. No network access; never
    prints a ticker identity or a private path.
    """
    root = repository_root

    try:
        verified_head = resolve_verified_v8b_production_git_commit(root)
    except V8BGitProvenanceBlocked as error:
        raise _wrap(error) from error

    try:
        verify_frozen_design_object(root)
        read_and_verify_design_freeze_approval(root, verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    try:
        review_binding = verify_reviewed_implementation_binding(root, verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error, "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    reviewed_commit = review_binding["reviewed_implementation_git_commit"]

    try:
        anchor = read_and_verify_v8_trusted_partition_anchor(root, verified_head)
    except (V8BProductionProvenanceBlocked, V8BGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    try:
        partition_manifest = read_partition_manifest(partition_manifest_path)
    except V8PartitionBlocked as error:
        raise V8BAllocationVerificationBlocked(error.reason) from error

    manifest_sha = partition_manifest["manifest_sha256"]
    if manifest_sha != anchor["authorized_partition_manifest_sha256"]:
        raise V8BAllocationVerificationBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    partition_implementation_commit = partition_manifest["partition_implementation_git_commit"]
    if partition_implementation_commit != anchor["authorized_partition_implementation_git_commit"]:
        raise V8BAllocationVerificationBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")
    if partition_manifest["study_name"] != V8_STUDY_NAME:
        raise V8BAllocationVerificationBlocked("PARTITION_MANIFEST_STUDY_NAME_MISMATCH")
    if partition_manifest["design_commit"] != V8_DESIGN_COMMIT:
        raise V8BAllocationVerificationBlocked("PARTITION_MANIFEST_DESIGN_COMMIT_MISMATCH")

    assignments = partition_manifest["block_assignments"]
    if not isinstance(assignments, Mapping) or set(_REQUIRED_BLOCK_KEYS) - set(assignments):
        raise V8BAllocationVerificationBlocked("PARTITION_BLOCK_ASSIGNMENT_MISSING")
    blocks: dict[str, list[str]] = {}
    for key in _REQUIRED_BLOCK_KEYS:
        value = assignments[key]
        if not isinstance(value, list):
            raise V8BAllocationVerificationBlocked("PARTITION_BLOCK_ASSIGNMENT_INVALID:" + key)
        blocks[key] = list(value)

    # Exact frozen parent T_spare count/hash pin -- never merely "whatever
    # this particular manifest's own T_spare list happens to be".
    parent_tickers = blocks["T_spare"]
    if len(parent_tickers) != EXPECTED_PARENT_T_SPARE_TICKER_COUNT:
        raise V8BAllocationVerificationBlocked("PARENT_T_SPARE_TICKER_COUNT_INVALID")
    computed_parent_hash = ticker_list_sha256(parent_tickers)
    if computed_parent_hash != EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256:
        raise V8BAllocationVerificationBlocked("PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH")
    if computed_parent_hash != partition_manifest["t_spare_ticker_list_sha256"]:
        raise V8BAllocationVerificationBlocked("PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH")

    try:
        artifact_raw = Path(allocation_artifact_path).read_bytes()
    except OSError as error:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_READ_FAILED") from error
    artifact = _strict_json_object(
        artifact_raw,
        invalid_reason="ALLOCATION_ARTIFACT_INVALID_JSON",
        duplicate_reason="ALLOCATION_ARTIFACT_DUPLICATE_KEY",
    )

    # Exact parent manifest SHA/implementation-commit binding -- the
    # artifact must claim the same verified parent this call just derived,
    # not merely a self-consistent one of its own choosing.
    if artifact.get("parent_v8_partition_manifest_sha256") != manifest_sha:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_PARENT_MANIFEST_SHA_MISMATCH")
    if artifact.get("parent_v8_partition_implementation_commit") != partition_implementation_commit:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_PARENT_IMPLEMENTATION_COMMIT_MISMATCH")

    # The artifact must record the *actually reviewed* implementation
    # commit, never merely a later (possibly audit-drifted) HEAD.
    if artifact.get("v8b_allocation_implementation_commit") != reviewed_commit:
        raise V8BAllocationVerificationBlocked("ALLOCATION_ARTIFACT_IMPLEMENTATION_COMMIT_NOT_REVIEWED")

    return _verify_t1b_allocation_artifact(
        artifact,
        parent_t_spare_tickers=parent_tickers,
        t0_tickers=blocks["T0"],
        old_t1_tickers=blocks["T1"],
        t2_tickers=blocks["T2"],
        t3_tickers=blocks["T3"],
        expected_parent_t_spare_ticker_list_sha256=computed_parent_hash,
        expected_v8b_frozen_design_commit=EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
    )


def resolve_and_verify_t1b_allocation_artifact(
    allocation_artifact_path: str | os.PathLike[str],
    partition_manifest_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """The sole public production `READ_ONLY_T1B_ALLOCATION_ARTIFACT_
    VERIFICATION` boundary (round-3 repeat finding HIGH-1). Always
    resolves trust from ``CANONICAL_REPOSITORY_ROOT`` -- this signature
    deliberately exposes no ``repository_root`` (or any other trust-root)
    override.
    """
    return _resolve_and_verify_t1b_allocation_artifact_with_repository_root(
        allocation_artifact_path, partition_manifest_path, repository_root=CANONICAL_REPOSITORY_ROOT
    )


__all__ = [
    "CANONICAL_REPOSITORY_ROOT",
    "V8BAllocationVerificationBlocked",
    "resolve_and_verify_t1b_allocation_artifact",
]
