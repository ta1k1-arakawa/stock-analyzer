"""V8D production provenance: exact-blob-bound freeze/reviewed-implementation
verification.

`V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md`, `V8D_DESIGN_FREEZE_
APPROVAL.json`. Mirrors `src.v8c_production_provenance`'s exact pattern:
every function reads its input from a caller-supplied *verified* Git commit
(`src.v8d_git_provenance.resolve_verified_v8d_production_git_commit`'s
return value) via `read_git_object_bytes`/`resolve_git_blob` -- never from
the working tree directly, and never from a caller-suppliable path. Every
check compares against a hardcoded literal constant, so a self-consistent
but mutated artifact still BLOCKs. Performs no network access and no
private-data access.

This subtask (`V8D_PROD_HIGH_1A_REVIEWED_IMPLEMENTATION_BINDING`)
implements exactly:

- mechanical verification that the frozen design document blob at the
  frozen design commit, and the design-freeze-approval blob at the current
  verified HEAD, equal their independently reviewed literal values; and
- reviewed-implementation binding: a caller-supplied arbitrary 40-hex Git
  commit SHA is never, by itself, sufficient evidence that a V8D production
  implementation was independently reviewed. The authoritative reviewed
  implementation commit is derived only from the committed
  `V8D_PRODUCTION_IMPLEMENTATION_REVIEW.json` artifact at a verified V8D
  Git HEAD, and every bound production file's blob must be byte-for-byte
  identical between that reviewed commit and current verified HEAD.

The `V8D_PRODUCTION_IMPLEMENTATION_REVIEW.json` artifact does not exist in
this repository yet -- `verify_reviewed_implementation_binding` therefore
fails closed today, by construction, exactly as
`V8C_PRODUCTION_IMPLEMENTATION_REVIEW.json` does for V8C. Human-gate
receipt binding is explicitly out of scope for this subtask (a separate
follow-up subtask, `V8D_PROD_HIGH_1B`).

The readiness consumer also uses this module for the original immutable V8
partition anchor. That is V8 provenance, not V8C study authority: its exact
Git blob and exact frozen manifest/implementation bindings are verified here
before any private partition bytes are read.
"""

from __future__ import annotations

import json
from typing import Any

from src.v8b_git_provenance import V8BGitProvenanceBlocked
from src.v8d_git_provenance import (
    V8DGitProvenanceBlocked,
    read_git_object_bytes,
    require_git_commit as _require_git_commit_raw,
    resolve_git_blob,
)

STUDY_NAME = "V8D_HISTORICAL_RESEARCH"

EXPECTED_V8D_FROZEN_DESIGN_COMMIT = "eda657cde2383718d986c4c4bfaae794784fe04d"
EXPECTED_V8D_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT = "9577a88c7bf46483b941aec3301c6064d9734c1f"
EXPECTED_V8D_DESIGN_FREEZE_APPROVAL_BLOB = "67e3e1ab1e252b5c8f7583eb0605ec0333e487f6"

DESIGN_DRAFT_GIT_PATH = "V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md"
DESIGN_FREEZE_APPROVAL_GIT_PATH = "V8D_DESIGN_FREEZE_APPROVAL.json"

# --- Future INDEPENDENT_V8D_PRODUCTION_IMPLEMENTATION_REVIEW binding -------

IMPLEMENTATION_REVIEW_GIT_PATH = "V8D_PRODUCTION_IMPLEMENTATION_REVIEW.json"
IMPLEMENTATION_REVIEW_SCHEMA_VERSION = "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_V1"
IMPLEMENTATION_REVIEW_ARTIFACT_ROLE = "PRODUCTION_IMPLEMENTATION_REVIEW"
IMPLEMENTATION_REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "reviewed_implementation_git_commit",
    "review_result",
    "approval_status",
)

# --- Original immutable V8 authority used by the fixed V8D T0 probe ---------

V8_DESIGN_COMMIT = "c414d3191cba356734d7ed08bdf1abc7d51fc384"
EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA = "61faade0625139cec3fb61216ab2f97f572a7028"
EXPECTED_V8_PARTITION_MANIFEST_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT = "36cbed941050e728f7f96ce2af505e81175cc02c"
TRUSTED_PARTITION_ANCHOR_GIT_PATH = "V8_TRUSTED_PARTITION.json"
TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION = "V8_TRUSTED_PARTITION_V1"
TRUSTED_PARTITION_ANCHOR_FIELDS = (
    "schema_version",
    "study_name",
    "design_commit",
    "authorization_status",
    "authorized_partition_manifest_sha256",
    "authorized_partition_implementation_git_commit",
    "authorization_note",
)

# Every production-relevant V8D source file that must be byte-for-byte
# identical between current verified HEAD and the exact commit
# INDEPENDENT_V8D_PRODUCTION_IMPLEMENTATION_REVIEW reviewed. A later
# docs/audit-only commit may move HEAD forward without invalidating review
# of these files, as long as none of their blobs actually changed.
BOUND_PRODUCTION_FILES: tuple[str, ...] = (
    "src/v8d_git_provenance.py",
    "src/v8d_production_provenance.py",
    "src/v8d_transport.py",
    "src/v8d_readiness.py",
    "src/v8d_historical_acquisition.py",
    "src/v8d_audit.py",
    "src/v8d_human_gate_consumption.py",
    "src/v8d_authority_bridge.py",
)


class V8DProductionProvenanceBlocked(RuntimeError):
    """Fail-closed exact-blob production provenance error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def require_git_commit(value: object, reason: str = "GIT_COMMIT_INVALID") -> str:
    """Require a full lowercase 40-hex Git object ID, raising this module's
    own ``V8DProductionProvenanceBlocked`` (not the generic-primitive
    ``V8BGitProvenanceBlocked``) so every caller's ``except
    V8DProductionProvenanceBlocked`` clause actually catches it."""
    try:
        return _require_git_commit_raw(value, reason)
    except V8BGitProvenanceBlocked as error:
        raise V8DProductionProvenanceBlocked(error.reason) from error


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8DProductionProvenanceBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8DProductionProvenanceBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8DProductionProvenanceBlocked(invalid_reason)
    return parsed


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap_git_provenance_error(
    error: V8DGitProvenanceBlocked, missing_reason: str | None = None
) -> V8DProductionProvenanceBlocked:
    if missing_reason is not None and error.reason in _GIT_OBJECT_MISSING_REASONS:
        return V8DProductionProvenanceBlocked(missing_reason)
    return V8DProductionProvenanceBlocked(error.reason)


# ---------------------------------------------------------------------------
# Frozen design object (verified at the frozen commit itself)
# ---------------------------------------------------------------------------


def verify_frozen_design_object(repository_root) -> None:
    """Verify the frozen `V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md`
    object *at the frozen design commit itself* -- not at current HEAD --
    exactly matches the blob independently reviewed."""
    try:
        blob = resolve_git_blob(repository_root, EXPECTED_V8D_FROZEN_DESIGN_COMMIT, DESIGN_DRAFT_GIT_PATH)
    except V8DGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error) from error
    if blob != EXPECTED_V8D_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT:
        raise V8DProductionProvenanceBlocked("V8D_FROZEN_DESIGN_OBJECT_MUTATED")


# ---------------------------------------------------------------------------
# Design freeze approval (exact blob binding at current verified HEAD)
# ---------------------------------------------------------------------------


def verify_design_freeze_approval_blob(repository_root, verified_head: str) -> str:
    """Verify `V8D_DESIGN_FREEZE_APPROVAL.json` at the current *verified*
    HEAD -- never the working tree -- resolves to the exact blob
    independently reviewed. Returns that blob SHA."""
    commit = require_git_commit(verified_head, "DESIGN_FREEZE_APPROVAL_HEAD_INVALID")
    try:
        blob = resolve_git_blob(repository_root, commit, DESIGN_FREEZE_APPROVAL_GIT_PATH)
    except V8DGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8D_DESIGN_FREEZE_APPROVAL_MISSING") from error
    if blob != EXPECTED_V8D_DESIGN_FREEZE_APPROVAL_BLOB:
        raise V8DProductionProvenanceBlocked("V8D_DESIGN_FREEZE_APPROVAL_BLOB_MUTATED")
    return blob


def read_and_verify_v8_trusted_partition_anchor(repository_root, verified_head: str) -> dict[str, Any]:
    """Verify the original V8 trust anchor from the verified Git HEAD.

    The blob is checked against its independent frozen SHA before its JSON is
    interpreted. The returned object contains only public provenance and
    hashes; it never contains a partition assignment or ticker identity.
    """
    commit = require_git_commit(verified_head, "TRUSTED_PARTITION_ANCHOR_HEAD_INVALID")
    try:
        blob = resolve_git_blob(repository_root, commit, TRUSTED_PARTITION_ANCHOR_GIT_PATH)
    except V8DGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "TRUSTED_PARTITION_ANCHOR_MISSING") from error
    if blob != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8DProductionProvenanceBlocked("V8_TRUSTED_PARTITION_BLOB_MUTATED")
    try:
        raw = read_git_object_bytes(repository_root, commit, TRUSTED_PARTITION_ANCHOR_GIT_PATH)
    except V8DGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "TRUSTED_PARTITION_ANCHOR_MISSING") from error
    anchor = _strict_json_object(
        raw,
        invalid_reason="TRUSTED_PARTITION_ANCHOR_INVALID_JSON",
        duplicate_reason="TRUSTED_PARTITION_ANCHOR_DUPLICATE_KEY",
    )
    if set(anchor) != set(TRUSTED_PARTITION_ANCHOR_FIELDS):
        raise V8DProductionProvenanceBlocked("TRUSTED_PARTITION_ANCHOR_SCHEMA_INVALID")
    if anchor["schema_version"] != TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION:
        raise V8DProductionProvenanceBlocked("TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION_MISMATCH")
    if anchor["design_commit"] != V8_DESIGN_COMMIT:
        raise V8DProductionProvenanceBlocked("TRUSTED_PARTITION_ANCHOR_DESIGN_COMMIT_MISMATCH")
    if anchor["authorization_status"] != "AUTHORIZED":
        raise V8DProductionProvenanceBlocked("TRUSTED_PARTITION_NOT_AUTHORIZED")
    if anchor["authorized_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8DProductionProvenanceBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    if anchor["authorized_partition_implementation_git_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8DProductionProvenanceBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")
    return dict(anchor)


# ---------------------------------------------------------------------------
# Reviewed-implementation binding (future INDEPENDENT_V8D_PRODUCTION_
# IMPLEMENTATION_REVIEW artifact -- does not exist in this repository yet,
# so this fails closed today by construction)
# ---------------------------------------------------------------------------


def verify_reviewed_implementation_binding(repository_root, verified_head: str) -> dict[str, Any]:
    """Bind production to the exact commit `INDEPENDENT_V8D_PRODUCTION_
    IMPLEMENTATION_REVIEW` actually reviewed, re-derived from the committed
    review artifact itself -- never assumed to equal current HEAD, and
    never satisfiable by a caller-supplied arbitrary SHA alone."""
    commit = require_git_commit(verified_head, "IMPLEMENTATION_REVIEW_HEAD_INVALID")
    try:
        raw = read_git_object_bytes(repository_root, commit, IMPLEMENTATION_REVIEW_GIT_PATH)
    except V8DGitProvenanceBlocked as error:
        raise _wrap_git_provenance_error(error, "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    review = _strict_json_object(
        raw,
        invalid_reason="V8D_PRODUCTION_IMPLEMENTATION_REVIEW_INVALID_JSON",
        duplicate_reason="V8D_PRODUCTION_IMPLEMENTATION_REVIEW_DUPLICATE_KEY",
    )
    if set(review) != set(IMPLEMENTATION_REVIEW_FIELDS):
        raise V8DProductionProvenanceBlocked("V8D_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_INVALID")
    if review["schema_version"] != IMPLEMENTATION_REVIEW_SCHEMA_VERSION:
        raise V8DProductionProvenanceBlocked("V8D_PRODUCTION_IMPLEMENTATION_REVIEW_SCHEMA_VERSION_MISMATCH")
    if review["study"] != STUDY_NAME:
        raise V8DProductionProvenanceBlocked("V8D_PRODUCTION_IMPLEMENTATION_REVIEW_STUDY_MISMATCH")
    if review["artifact_role"] != IMPLEMENTATION_REVIEW_ARTIFACT_ROLE:
        raise V8DProductionProvenanceBlocked("V8D_PRODUCTION_IMPLEMENTATION_REVIEW_ARTIFACT_ROLE_MISMATCH")
    if review["review_result"] != "PASS":
        raise V8DProductionProvenanceBlocked("V8D_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_PASS")
    if review["approval_status"] != "APPROVED":
        raise V8DProductionProvenanceBlocked("V8D_PRODUCTION_IMPLEMENTATION_REVIEW_NOT_APPROVED")

    reviewed_commit = require_git_commit(
        review["reviewed_implementation_git_commit"], "V8D_PRODUCTION_IMPLEMENTATION_REVIEW_COMMIT_INVALID"
    )

    for path in BOUND_PRODUCTION_FILES:
        try:
            blob_head = resolve_git_blob(repository_root, commit, path)
        except V8DGitProvenanceBlocked as error:
            raise _wrap_git_provenance_error(error, "V8D_BOUND_FILE_MISSING_AT_HEAD:" + path) from error
        try:
            blob_reviewed = resolve_git_blob(repository_root, reviewed_commit, path)
        except V8DGitProvenanceBlocked as error:
            raise _wrap_git_provenance_error(error, "V8D_BOUND_FILE_MISSING_AT_REVIEWED_COMMIT:" + path) from error
        if blob_head != blob_reviewed:
            raise V8DProductionProvenanceBlocked("V8D_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:" + path)

    return {
        "reviewed_implementation_git_commit": reviewed_commit,
        "verified_head": commit,
        "bound_files_verified": len(BOUND_PRODUCTION_FILES),
    }


__all__ = [
    "BOUND_PRODUCTION_FILES",
    "DESIGN_DRAFT_GIT_PATH",
    "DESIGN_FREEZE_APPROVAL_GIT_PATH",
    "EXPECTED_V8D_DESIGN_DRAFT_BLOB_AT_FROZEN_COMMIT",
    "EXPECTED_V8D_DESIGN_FREEZE_APPROVAL_BLOB",
    "EXPECTED_V8D_FROZEN_DESIGN_COMMIT",
    "IMPLEMENTATION_REVIEW_ARTIFACT_ROLE",
    "IMPLEMENTATION_REVIEW_FIELDS",
    "IMPLEMENTATION_REVIEW_GIT_PATH",
    "IMPLEMENTATION_REVIEW_SCHEMA_VERSION",
    "STUDY_NAME",
    "EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT",
    "EXPECTED_V8_PARTITION_MANIFEST_SHA256",
    "EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA",
    "TRUSTED_PARTITION_ANCHOR_FIELDS",
    "TRUSTED_PARTITION_ANCHOR_GIT_PATH",
    "TRUSTED_PARTITION_ANCHOR_SCHEMA_VERSION",
    "V8_DESIGN_COMMIT",
    "V8DProductionProvenanceBlocked",
    "verify_design_freeze_approval_blob",
    "verify_frozen_design_object",
    "verify_reviewed_implementation_binding",
    "read_and_verify_v8_trusted_partition_anchor",
]
