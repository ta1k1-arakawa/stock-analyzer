"""V8C-specific `T2` authority bridge (§7.2).

`V8B_T2_AUTHORITY_BRIDGE.json` is scoped to `V8B_HISTORICAL_RESEARCH` and
must never be silently reused as V8C authority
(``existing_V8B_T2_authority_bridge_authorizes_V8C=false``). This module
implements the schema, builder, production-gated creation boundary, and
verifier for a **separate**, V8C-specific bridge artifact
(`V8C_T2_AUTHORITY_BRIDGE.json`) that re-binds the same original, immutable
V8 `T2` authority to V8C's own study identity and frozen design commit --
never modifying, reinterpreting, or re-pinning `V8_TRUSTED_PARTITION.json`
or the original V8 partition manifest.

`CREATE_V8C_T2_AUTHORITY_BRIDGE` is **not executed** by this implementation
phase -- `HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE` has not occurred, and this
repository does not contain a `V8C_T2_AUTHORITY_BRIDGE.json` file. Every
real invocation of ``create_v8c_t2_authority_bridge_production`` and
``resolve_and_verify_v8c_t2_authority_bridge`` fails closed today by
construction. Every test exercising this module is fake/synthetic-only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    read_git_object_bytes,
    require_git_commit,
    resolve_git_blob,
    resolve_verified_v8c_production_git_commit,
)
from src.v8c_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_T2_AUTHORITY_BRIDGE,
    V8CHumanGateConsumptionBlocked,
    consume_gate_once,
    require_gate_not_yet_consumed,
)
from src.v8c_production_provenance import (
    EXPECTED_T2_TICKER_COUNT,
    EXPECTED_T2_TICKER_LIST_SHA256,
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
    V8CProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    read_and_verify_v8_trusted_partition_anchor,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8c_stage_state import read_valid_t2_recheck_pass, V8CStageEvidenceBlocked

STUDY_NAME = "V8C_HISTORICAL_RESEARCH"
SCHEMA_VERSION = "V8C_T2_AUTHORITY_BRIDGE_V1"
ROLE = "SEALED_HOLDOUT"
SOURCE_AUTHORITY = "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY"

# Deliberately its own, separate Git path -- never `V8B_T2_AUTHORITY_
# BRIDGE.json`. A production reader that only ever looks at this path
# structurally cannot be satisfied by the V8B bridge artifact.
BRIDGE_GIT_PATH = "V8C_T2_AUTHORITY_BRIDGE.json"

HUMAN_GATE_PREFIX = "V8C_HUMAN_AUTHORIZE_T2_AUTHORITY_BRIDGE_FOR_DESIGN_"


def expected_human_gate(exact_frozen_v8c_design_commit: str) -> str:
    return HUMAN_GATE_PREFIX + exact_frozen_v8c_design_commit


BRIDGE_CONFIRMATION = "V8C_PRODUCTION_CREATE_T2_AUTHORITY_BRIDGE"

BRIDGE_FIELDS = (
    "schema_version",
    "study",
    "role",
    "exact_frozen_v8c_design_commit",
    "source_authority",
    "v8_trust_anchor_git_identity",
    "authorized_parent_v8_partition_manifest_sha256",
    "expected_t2_ticker_count",
    "expected_t2_ticker_list_sha256",
    "t2_membership_reassignment",
    "v8_trusted_partition_json_mutated_or_repinned",
    "t2_acquired_before_authorized_v8c_acquisition",
    "t2_research_open_count_before_official_opening",
    "reviewed_production_implementation_commit",
    "exact_human_bridge_authorization_identity",
    "authorization_note",
)

# The future INDEPENDENT_V8C_T2_AUTHORITY_BRIDGE_REVIEW artifact.
BRIDGE_REVIEW_GIT_PATH = "V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW.json"
BRIDGE_REVIEW_SCHEMA_VERSION = "V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_V1"
BRIDGE_REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "exact_bridge_git_commit",
    "exact_bridge_git_blob_sha",
    "exact_frozen_v8c_design_commit",
    "v8_trust_anchor_git_identity",
    "authorized_parent_v8_partition_manifest_sha256",
    "exact_human_bridge_authorization_identity",
    "reviewed_production_implementation_commit",
    "review_result",
    "approval_status",
)


class V8CT2BridgeBlocked(RuntimeError):
    """Fail-closed V8C T2 authority-bridge construction/read/creation error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_git_commit(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise V8CT2BridgeBlocked(reason)
    return value


def _require_sha256_hex(value: object, reason: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise V8CT2BridgeBlocked(reason)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except ValueError as error:
        raise V8CT2BridgeBlocked("NONFINITE_VALUE") from error


def build_v8c_t2_authority_bridge(
    *,
    v8_trust_anchor_git_identity: str,
    authorized_parent_v8_partition_manifest_sha256: str,
    reviewed_production_implementation_commit: str,
    exact_human_bridge_authorization_identity: str,
    authorization_note: str,
) -> dict[str, Any]:
    """Build (never writes) the V8C-specific `T2` authority bridge.

    Every field is pinned to this module's own frozen constants except the
    reviewed implementation commit (derived from the real production
    reviewed-implementation binding) and the human authorization identity
    (the exact `HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE` token) -- never a
    caller-suppliable arbitrary value for the authority fields themselves.
    """
    if v8_trust_anchor_git_identity != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8CT2BridgeBlocked("V8_TRUST_ANCHOR_GIT_IDENTITY_MISMATCH")
    manifest_sha = _require_sha256_hex(
        authorized_parent_v8_partition_manifest_sha256, "PARENT_V8_PARTITION_MANIFEST_SHA_INVALID"
    )
    if manifest_sha != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8CT2BridgeBlocked("PARENT_V8_PARTITION_MANIFEST_SHA_MISMATCH")
    reviewed_commit = _require_git_commit(
        reviewed_production_implementation_commit, "REVIEWED_PRODUCTION_IMPLEMENTATION_COMMIT_INVALID"
    )
    if not isinstance(exact_human_bridge_authorization_identity, str) or not exact_human_bridge_authorization_identity:
        raise V8CT2BridgeBlocked("HUMAN_BRIDGE_AUTHORIZATION_IDENTITY_INVALID")
    if exact_human_bridge_authorization_identity != expected_human_gate(EXPECTED_V8C_FROZEN_DESIGN_COMMIT):
        raise V8CT2BridgeBlocked("HUMAN_BRIDGE_AUTHORIZATION_IDENTITY_MISMATCH")
    if not isinstance(authorization_note, str):
        raise V8CT2BridgeBlocked("AUTHORIZATION_NOTE_INVALID")

    bridge: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study": STUDY_NAME,
        "role": ROLE,
        "exact_frozen_v8c_design_commit": EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        "source_authority": SOURCE_AUTHORITY,
        "v8_trust_anchor_git_identity": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "authorized_parent_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "expected_t2_ticker_count": EXPECTED_T2_TICKER_COUNT,
        "expected_t2_ticker_list_sha256": EXPECTED_T2_TICKER_LIST_SHA256,
        "t2_membership_reassignment": "PROHIBITED",
        "v8_trusted_partition_json_mutated_or_repinned": False,
        "t2_acquired_before_authorized_v8c_acquisition": False,
        "t2_research_open_count_before_official_opening": 0,
        "reviewed_production_implementation_commit": reviewed_commit,
        "exact_human_bridge_authorization_identity": exact_human_bridge_authorization_identity,
        "authorization_note": authorization_note,
    }
    if set(bridge) != set(BRIDGE_FIELDS):
        raise V8CT2BridgeBlocked("BRIDGE_SCHEMA_INVALID")
    return bridge


def validate_v8c_t2_authority_bridge(bridge: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless ``bridge`` is a well-formed V8C-specific bridge,
    exactly pinned to this module's frozen authority constants -- never
    merely internally self-consistent."""
    if not isinstance(bridge, Mapping) or set(bridge) != set(BRIDGE_FIELDS):
        raise V8CT2BridgeBlocked("BRIDGE_SCHEMA_INVALID")
    if bridge["schema_version"] != SCHEMA_VERSION:
        raise V8CT2BridgeBlocked("BRIDGE_SCHEMA_VERSION_MISMATCH")
    if bridge["study"] != STUDY_NAME:
        raise V8CT2BridgeBlocked("BRIDGE_STUDY_MISMATCH")
    if bridge["role"] != ROLE:
        raise V8CT2BridgeBlocked("BRIDGE_ROLE_MISMATCH")
    if bridge["exact_frozen_v8c_design_commit"] != EXPECTED_V8C_FROZEN_DESIGN_COMMIT:
        raise V8CT2BridgeBlocked("BRIDGE_DESIGN_COMMIT_MISMATCH")
    if bridge["source_authority"] != SOURCE_AUTHORITY:
        raise V8CT2BridgeBlocked("BRIDGE_SOURCE_AUTHORITY_MISMATCH")
    if bridge["v8_trust_anchor_git_identity"] != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8CT2BridgeBlocked("BRIDGE_ANCHOR_IDENTITY_MISMATCH")
    if bridge["authorized_parent_v8_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8CT2BridgeBlocked("BRIDGE_MANIFEST_SHA_MISMATCH")
    if bridge["expected_t2_ticker_count"] != EXPECTED_T2_TICKER_COUNT:
        raise V8CT2BridgeBlocked("BRIDGE_TICKER_COUNT_MISMATCH")
    if bridge["expected_t2_ticker_list_sha256"] != EXPECTED_T2_TICKER_LIST_SHA256:
        raise V8CT2BridgeBlocked("BRIDGE_TICKER_LIST_SHA_MISMATCH")
    if bridge["t2_membership_reassignment"] != "PROHIBITED":
        raise V8CT2BridgeBlocked("BRIDGE_MEMBERSHIP_REASSIGNMENT_INVALID")
    if bridge["v8_trusted_partition_json_mutated_or_repinned"] is not False:
        raise V8CT2BridgeBlocked("BRIDGE_ANCHOR_MUTATION_INVALID")
    if bridge["t2_acquired_before_authorized_v8c_acquisition"] is not False:
        raise V8CT2BridgeBlocked("BRIDGE_ACQUIRED_BEFORE_INVALID")
    if bridge["t2_research_open_count_before_official_opening"] != 0:
        raise V8CT2BridgeBlocked("BRIDGE_OPEN_COUNT_INVALID")
    _require_git_commit(bridge["reviewed_production_implementation_commit"], "BRIDGE_REVIEWED_COMMIT_INVALID")
    if bridge["exact_human_bridge_authorization_identity"] != expected_human_gate(EXPECTED_V8C_FROZEN_DESIGN_COMMIT):
        raise V8CT2BridgeBlocked("BRIDGE_HUMAN_GATE_MISMATCH")
    if not isinstance(bridge["authorization_note"], str):
        raise V8CT2BridgeBlocked("BRIDGE_AUTHORIZATION_NOTE_INVALID")
    return dict(bridge)


# ---------------------------------------------------------------------------
# Production-gated CREATE_V8C_T2_AUTHORITY_BRIDGE boundary
# ---------------------------------------------------------------------------

_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8CT2BridgeBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8CT2BridgeBlocked(missing_reason)
    if isinstance(reason, str):
        return V8CT2BridgeBlocked(reason)
    return V8CT2BridgeBlocked("PROVENANCE_CHECK_FAILED")


def _write_bridge_once(destination: Path, bridge_bytes: bytes) -> Path:
    if destination.exists():
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_ALREADY_EXISTS")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8CT2BridgeBlocked("OUTPUT_PATH_PARENT_INVALID") from error
    staging = destination.parent / (destination.name + ".staging-" + os.urandom(8).hex())
    try:
        try:
            with open(staging, "wb") as stream:
                stream.write(bridge_bytes)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_STAGING_WRITE_FAILED") from error
        try:
            os.link(str(staging), str(destination))
        except FileExistsError as error:
            raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_ALREADY_EXISTS") from error
        except OSError as error:
            raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_ATOMIC_PUBLISH_FAILED") from error
    finally:
        try:
            if staging.exists():
                staging.unlink()
        except OSError:
            pass
    return destination


def create_v8c_t2_authority_bridge_production(
    *,
    confirmation: str,
    human_bridge_authorization: str,
    output_path: str | os.PathLike[str],
    authorization_note: str,
) -> dict[str, Any]:
    """Sole production entrypoint for `CREATE_V8C_T2_AUTHORITY_BRIDGE`.
    **Not executed** by this implementation phase."""
    return _create_v8c_t2_authority_bridge_production_with_dependencies(
        confirmation=confirmation,
        human_bridge_authorization=human_bridge_authorization,
        output_path=output_path,
        authorization_note=authorization_note,
        git_commit_resolver=lambda: resolve_verified_v8c_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        design_freeze_approval_reader=lambda head: read_and_verify_design_freeze_approval(CANONICAL_REPOSITORY_ROOT, head),
        frozen_design_object_verifier=lambda: verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(CANONICAL_REPOSITORY_ROOT, head),
        anchor_reader=lambda head: read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, head),
        clock=lambda: datetime.now(timezone.utc),
        consumption_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
        t2_preservation_pass_reader=lambda implementation_commit: read_valid_t2_recheck_pass(
            CANONICAL_CONSUMPTION_STATE_ROOT,
            frozen_design_commit=EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
            reviewed_implementation_commit=implementation_commit,
        ),
        t2_preservation_live_resolver=_resolve_and_recheck_t2_preservation_live,
    )


def _resolve_and_recheck_t2_preservation_live() -> dict[str, Any]:
    from src.v8c_t2_preservation_recheck import resolve_and_recheck_t2_preservation

    return resolve_and_recheck_t2_preservation()


def _create_v8c_t2_authority_bridge_production_with_dependencies(
    *,
    confirmation: str,
    human_bridge_authorization: str,
    output_path: str | os.PathLike[str],
    authorization_note: str,
    git_commit_resolver: Callable[[], str],
    design_freeze_approval_reader: Callable[[str], Mapping[str, Any]],
    frozen_design_object_verifier: Callable[[], None],
    reviewed_implementation_binder: Callable[[str], Mapping[str, Any]],
    anchor_reader: Callable[[str], Mapping[str, Any]],
    clock: Callable[[], datetime],
    consumption_state_root: str | os.PathLike[str],
    t2_preservation_pass_reader: Callable[[str], Mapping[str, Any]] | None = None,
    t2_preservation_live_resolver: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if confirmation != BRIDGE_CONFIRMATION:
        raise V8CT2BridgeBlocked("V8C_BRIDGE_CREATION_CONFIRMATION_INVALID")

    try:
        require_gate_not_yet_consumed(
            consumption_state_root, GATE_T2_AUTHORITY_BRIDGE, EXPECTED_V8C_FROZEN_DESIGN_COMMIT
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CT2BridgeBlocked(error.reason) from error

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
        raise V8CT2BridgeBlocked("TRUSTED_PARTITION_NOT_AUTHORIZED")

    if human_bridge_authorization != expected_human_gate(EXPECTED_V8C_FROZEN_DESIGN_COMMIT):
        raise V8CT2BridgeBlocked("V8C_HUMAN_BRIDGE_AUTHORIZATION_INVALID")

    if t2_preservation_pass_reader is not None:
        try:
            preservation = t2_preservation_pass_reader(reviewed_commit)
            if preservation.get("result") != "PASS" or preservation.get("recheck_point") != "recheck_2":
                raise V8CT2BridgeBlocked("V8C_T2_PRESERVATION_RECHECK_PASS_REQUIRED")
        except (V8CStageEvidenceBlocked, V8CT2BridgeBlocked) as error:
            raise V8CT2BridgeBlocked("V8C_T2_PRESERVATION_RECHECK_PASS_REQUIRED") from error

    # Revalidate against CURRENT safe Git/trust/gate state immediately
    # before consuming HUMAN_V8C_T2_AUTHORITY_BRIDGE_GATE -- the durable
    # recheck_2 record above proves a real execution once produced PASS,
    # but a stored record alone is not production authority. Re-running the
    # live resolver here re-derives every condition fresh (anchor, reviewed
    # implementation binding, gate-consumption state, pre-freeze baseline
    # blob, design freeze approval) at bridge-creation time, so drift since
    # the durable record was written still BLOCKs.
    if t2_preservation_live_resolver is not None:
        try:
            live = t2_preservation_live_resolver()
            if live.get("result") != "PASS" or live.get("recheck_point") != "recheck_2":
                raise V8CT2BridgeBlocked("V8C_T2_PRESERVATION_RECHECK_PASS_REQUIRED")
        except Exception as error:  # noqa: BLE001 - any live-recheck failure BLOCKs bridge creation
            if isinstance(error, V8CT2BridgeBlocked):
                raise
            raise V8CT2BridgeBlocked("V8C_T2_PRESERVATION_RECHECK_PASS_REQUIRED") from error

    try:
        consume_gate_once(
            consumption_state_root, GATE_T2_AUTHORITY_BRIDGE, EXPECTED_V8C_FROZEN_DESIGN_COMMIT, clock=clock
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CT2BridgeBlocked(error.reason) from error

    bridge = build_v8c_t2_authority_bridge(
        v8_trust_anchor_git_identity=EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        authorized_parent_v8_partition_manifest_sha256=anchor["authorized_partition_manifest_sha256"],
        reviewed_production_implementation_commit=reviewed_commit,
        exact_human_bridge_authorization_identity=human_bridge_authorization,
        authorization_note=authorization_note,
    )
    destination = Path(output_path)
    _write_bridge_once(destination, canonical_json_bytes(bridge))
    return dict(bridge)


# ---------------------------------------------------------------------------
# Read-only verification from a verified Git object -- the sole production
# T2-authority resolution path for V8C. Reads ONLY `BRIDGE_GIT_PATH`
# (`V8C_T2_AUTHORITY_BRIDGE.json`), never `V8B_T2_AUTHORITY_BRIDGE.json`.
# ---------------------------------------------------------------------------


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8CT2BridgeBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8CT2BridgeBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8CT2BridgeBlocked(invalid_reason)
    return parsed


def read_and_verify_v8c_t2_authority_bridge(repository_root, verified_head: str) -> dict[str, Any]:
    """Read the V8C-specific bridge from a verified Git object at
    ``BRIDGE_GIT_PATH`` only. Fails closed with
    ``V8C_T2_AUTHORITY_BRIDGE_MISSING`` today -- this repository contains
    no such file."""
    commit = require_git_commit(verified_head, "T2_AUTHORITY_BRIDGE_HEAD_INVALID")
    try:
        raw = read_git_object_bytes(repository_root, commit, BRIDGE_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error, "V8C_T2_AUTHORITY_BRIDGE_MISSING") from error
    bridge = _strict_json_object(
        raw, invalid_reason="V8C_T2_AUTHORITY_BRIDGE_INVALID_JSON", duplicate_reason="V8C_T2_AUTHORITY_BRIDGE_DUPLICATE_KEY"
    )
    validated = validate_v8c_t2_authority_bridge(bridge)
    try:
        from src.v8c_production_provenance import verify_reviewed_implementation_binding
        expected_impl = verify_reviewed_implementation_binding(repository_root, commit)["reviewed_implementation_git_commit"]
    except Exception as error:  # noqa: BLE001
        raise V8CT2BridgeBlocked("V8C_T2_BRIDGE_IMPLEMENTATION_REVIEW_BINDING_FAILED") from error
    if validated["reviewed_production_implementation_commit"] != expected_impl:
        raise V8CT2BridgeBlocked("V8C_T2_BRIDGE_REVIEWED_IMPLEMENTATION_MISMATCH")
    return validated


def read_and_verify_v8c_t2_authority_bridge_independent_review(
    repository_root,
    verified_head: str,
    *,
    expected_bridge_git_blob_sha: str,
) -> dict[str, Any]:
    """Read and verify the future `INDEPENDENT_V8C_T2_AUTHORITY_BRIDGE_
    REVIEW` artifact from a verified Git object. Does not exist yet, so
    this fails closed today by construction."""
    commit = require_git_commit(verified_head, "T2_AUTHORITY_BRIDGE_REVIEW_HEAD_INVALID")
    try:
        raw = read_git_object_bytes(repository_root, commit, BRIDGE_REVIEW_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error, "V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_MISSING") from error
    review = _strict_json_object(
        raw,
        invalid_reason="V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_INVALID_JSON",
        duplicate_reason="V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_DUPLICATE_KEY",
    )
    if set(review) != set(BRIDGE_REVIEW_FIELDS):
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_SCHEMA_INVALID")
    if review["schema_version"] != BRIDGE_REVIEW_SCHEMA_VERSION:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_SCHEMA_VERSION_MISMATCH")
    if review["study"] != STUDY_NAME:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_STUDY_MISMATCH")
    if review["artifact_role"] != "INDEPENDENT_V8C_T2_AUTHORITY_BRIDGE_REVIEW":
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_ROLE_MISMATCH")
    if review["exact_frozen_v8c_design_commit"] != EXPECTED_V8C_FROZEN_DESIGN_COMMIT:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_DESIGN_COMMIT_MISMATCH")
    if review["v8_trust_anchor_git_identity"] != EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_ANCHOR_MISMATCH")
    if review["authorized_parent_v8_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_MANIFEST_MISMATCH")
    if review["exact_bridge_git_blob_sha"] != expected_bridge_git_blob_sha:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_BLOB_MISMATCH")
    reviewed_bridge_commit = _require_git_commit(review["exact_bridge_git_commit"], "V8C_T2_AUTHORITY_BRIDGE_REVIEW_COMMIT_INVALID")
    try:
        reviewed_bridge_blob = resolve_git_blob(repository_root, reviewed_bridge_commit, BRIDGE_GIT_PATH)
        current_bridge_blob = resolve_git_blob(repository_root, verified_head, BRIDGE_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error, "V8C_T2_AUTHORITY_BRIDGE_REVIEWED_BLOB_MISSING") from error
    if reviewed_bridge_blob != review["exact_bridge_git_blob_sha"]:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_REVIEWED_BLOB_SELF_MISMATCH")
    if current_bridge_blob != review["exact_bridge_git_blob_sha"]:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_CURRENT_BLOB_DRIFT")
    bridge = read_and_verify_v8c_t2_authority_bridge(repository_root, verified_head)
    if review["exact_human_bridge_authorization_identity"] != bridge["exact_human_bridge_authorization_identity"]:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_REVIEW_HUMAN_GATE_MISMATCH")
    try:
        from src.v8c_production_provenance import verify_reviewed_implementation_binding
        expected_impl = verify_reviewed_implementation_binding(repository_root, verified_head)["reviewed_implementation_git_commit"]
    except Exception as error:  # noqa: BLE001
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_REVIEW_IMPLEMENTATION_BINDING_FAILED") from error
    if review["reviewed_production_implementation_commit"] != expected_impl:
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_REVIEW_IMPLEMENTATION_MISMATCH")
    if review["review_result"] != "PASS":
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_NOT_PASS")
    if review["approval_status"] != "APPROVED":
        raise V8CT2BridgeBlocked("V8C_T2_AUTHORITY_BRIDGE_INDEPENDENT_REVIEW_NOT_APPROVED")
    return dict(review)


__all__ = [
    "BRIDGE_CONFIRMATION",
    "BRIDGE_FIELDS",
    "BRIDGE_GIT_PATH",
    "BRIDGE_REVIEW_FIELDS",
    "BRIDGE_REVIEW_GIT_PATH",
    "BRIDGE_REVIEW_SCHEMA_VERSION",
    "HUMAN_GATE_PREFIX",
    "ROLE",
    "SCHEMA_VERSION",
    "SOURCE_AUTHORITY",
    "STUDY_NAME",
    "V8CT2BridgeBlocked",
    "build_v8c_t2_authority_bridge",
    "create_v8c_t2_authority_bridge_production",
    "expected_human_gate",
    "read_and_verify_v8c_t2_authority_bridge",
    "read_and_verify_v8c_t2_authority_bridge_independent_review",
    "validate_v8c_t2_authority_bridge",
]
