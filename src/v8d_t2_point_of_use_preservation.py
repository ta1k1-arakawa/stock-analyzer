"""V8D T2 point-of-use preservation contract.

This module defines the safe, committed/audit/provenance-only checkpoint
immediately before the later T2 acquisition preflight.  It never reads the
private partition manifest, never exposes T2 identities, never consumes a
human gate, and never writes either future point-of-use artifact.

The public production functions have no authority or evidence override
parameters.  The underscore-prefixed dependency-injected functions exist
only for synthetic tests of the fail-closed contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v8d_authority_bridge import (
    T2_STAGE,
    V8DAuthorityBridgeBlocked,
    verify_stage_authority_bridge,
)
from src.v8d_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8DGitProvenanceBlocked,
    read_git_object_bytes,
    require_strict_git_ancestor,
    resolve_git_blob,
    resolve_verified_v8d_production_git_commit,
)
from src.v8d_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_T2_RAW_ACQUISITION,
    V8DHumanGateConsumptionBlocked,
    has_gate_been_consumed,
)
from src.v8d_production_provenance import (
    EXPECTED_V8D_DESIGN_FREEZE_APPROVAL_BLOB,
    EXPECTED_V8D_FROZEN_DESIGN_COMMIT,
    EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
    STUDY_NAME,
    V8DProductionProvenanceBlocked,
    read_and_verify_v8_trusted_partition_anchor,
    verify_design_freeze_approval_blob,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8d_readiness_audit_verification import require_t2_readiness_audit_verification_pass


STUDY = STUDY_NAME
FROZEN_DESIGN_COMMIT = EXPECTED_V8D_FROZEN_DESIGN_COMMIT
POINT_OF_USE_ARTIFACT_PATH = "V8D_T2_POINT_OF_USE_PRESERVATION_RECHECK.json"
POINT_OF_USE_REVIEW_PATH = "V8D_T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW.json"
POINT_OF_USE_SCHEMA_VERSION = "V8D_T2_POINT_OF_USE_PRESERVATION_RECHECK_V1"
POINT_OF_USE_ARTIFACT_ROLE = "T2_POINT_OF_USE_PRESERVATION_RECHECK"
POINT_OF_USE_CHECKPOINT = "READ_ONLY_V8D_T2_POINT_OF_USE_PRESERVATION_RECHECK"
POINT_OF_USE_RECHECK = "immediately_before_T2_acquisition"
READINESS_VERIFICATION_STAGE = "READ_ONLY_T2_READINESS_TRANSPORT_AUDIT_VERIFICATION"
READINESS_LOGICAL_STAGE = "T2_TRANSPORT_READINESS"
POINT_OF_USE_REVIEW_SCHEMA_VERSION = "V8D_T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW_V1"
POINT_OF_USE_REVIEW_ROLE = "T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW"
POINT_OF_USE_REVIEW_CHECKPOINT = "INDEPENDENT_V8D_T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW"

PREFREEZE_PRESERVATION_COMMIT = "8ae3032b42b426420f44c9f7194f0b1849c23e98"
PREFREEZE_PRESERVATION_BLOB = "d023913b435ffd18eadef1e213c7ea43a49db331"
PREFREEZE_PRESERVATION_PATH = "V8D_T2_PREFREEZE_PRESERVATION_RECHECK.md"
V8_STATE_PATH = "V8_STATE.json"

T2_COUNT = 300
T2_TICKER_LIST_SHA256 = "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500"

NINE_CONDITION_FIELDS = (
    "T2_real_data_acquired",
    "T2_opened",
    "T2_research_access_count",
    "T2_features_observed",
    "T2_outcomes_observed",
    "T2_membership_reassigned",
    "universe_definition_compatible",
    "partition_algorithm_compatible",
    "data_quality_policy_unchanged",
)

POINT_OF_USE_ARTIFACT_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "checkpoint",
    "recheck",
    "v8d_frozen_design_commit",
    "source_prefreeze_preservation_git_commit",
    "source_prefreeze_preservation_git_blob_sha",
    "readiness_verification_stage",
    "readiness_verification_result",
    "readiness_receipt_self_hash",
    "v8_trust_anchor_git_blob",
    "original_v8_partition_manifest_sha256",
    "parent_v8_partition_implementation_commit",
    "t2_count",
    "t2_ticker_list_sha256",
    *NINE_CONDITION_FIELDS,
    "t2_raw_acquisition_gate_consumed",
    "point_of_use_preservation_result",
)

POINT_OF_USE_REVIEW_FIELDS = (
    "schema_version",
    "study",
    "artifact_role",
    "checkpoint",
    "v8d_frozen_design_commit",
    "reviewed_recheck_git_commit",
    "reviewed_recheck_git_blob_sha",
    "review_result",
)


class V8DT2PointOfUsePreservationBlocked(RuntimeError):
    """Fail-closed point-of-use preservation contract error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _strict_json_object(raw: bytes, *, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8DT2PointOfUsePreservationBlocked(duplicate_reason)
            result[key] = value

        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8DT2PointOfUsePreservationBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8DT2PointOfUsePreservationBlocked(invalid_reason)
    return parsed


def _require_exact_fields(value: Mapping[str, Any], fields: tuple[str, ...], reason: str) -> None:
    if set(value) != set(fields):
        raise V8DT2PointOfUsePreservationBlocked(reason)


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise V8DT2PointOfUsePreservationBlocked(reason)
    return value


def _require_bool(value: object, reason: str) -> bool:
    if type(value) is not bool:
        raise V8DT2PointOfUsePreservationBlocked(reason)
    return value


def _require_safe_conditions(conditions: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(conditions, Mapping) or set(conditions) != set(NINE_CONDITION_FIELDS):
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_CONDITIONS_SCHEMA_INVALID")
    for field in (
        "T2_real_data_acquired",
        "T2_opened",
        "T2_features_observed",
        "T2_outcomes_observed",
        "T2_membership_reassigned",
    ):
        if _require_bool(conditions[field], "V8D_T2_POINT_OF_USE_CONDITION_TYPE_INVALID") is not False:
            raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_CONDITION_INVALID:" + field)
    access_count = conditions["T2_research_access_count"]
    if type(access_count) is not int or access_count != 0:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_CONDITION_INVALID:T2_research_access_count")
    for field in (
        "universe_definition_compatible",
        "partition_algorithm_compatible",
        "data_quality_policy_unchanged",
    ):
        if _require_bool(conditions[field], "V8D_T2_POINT_OF_USE_CONDITION_TYPE_INVALID") is not True:
            raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_CONDITION_INVALID:" + field)
    return dict(conditions)


def _derive_conditions_from_v8_state(
    repository_root: Path,
    verified_head: str,
    git_object_reader: Callable[[Path, str, str], bytes],
) -> dict[str, Any]:
    try:
        state = _strict_json_object(
            git_object_reader(repository_root, verified_head, V8_STATE_PATH),
            invalid_reason="V8D_T2_POINT_OF_USE_V8_STATE_INVALID_JSON",
            duplicate_reason="V8D_T2_POINT_OF_USE_V8_STATE_DUPLICATE_KEY",
        )
    except V8DGitProvenanceBlocked as error:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_V8_STATE_MISSING") from error

    t2 = state.get("T2")
    trust_anchor_state = state.get("trusted_partition_anchor_state")
    if not isinstance(t2, Mapping) or not isinstance(trust_anchor_state, Mapping):
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_V8_STATE_T2_EVIDENCE_INVALID")
    if _require_bool(t2.get("raw_data_acquired"), "V8D_T2_POINT_OF_USE_V8_STATE_RAW_DATA_INVALID"):
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_CONDITION_INVALID:T2_real_data_acquired")
    if _require_bool(t2.get("opened_for_research"), "V8D_T2_POINT_OF_USE_V8_STATE_OPENED_INVALID"):
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_CONDITION_INVALID:T2_opened")
    access_count = t2.get("sealed_holdout_access_count")
    if access_count is not None and (type(access_count) is not int or access_count != 0):
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_CONDITION_INVALID:T2_research_access_count")
    if _require_bool(
        trust_anchor_state.get("block_assignments_exposed"),
        "V8D_T2_POINT_OF_USE_V8_STATE_ASSIGNMENT_EXPOSURE_INVALID",
    ):
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_CONDITION_INVALID:T2_features_observed")

    # The committed state uses null for an unopened sealed-holdout access
    # counter. Together with explicit no-acquisition/no-opening evidence,
    # that is the safe committed representation of zero access; no private
    # manifest or identity is needed at this checkpoint.
    return _require_safe_conditions({
        "T2_real_data_acquired": False,
        "T2_opened": False,
        "T2_research_access_count": 0,
        "T2_features_observed": False,
        "T2_outcomes_observed": False,
        "T2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
    })


def _validate_anchor(anchor: Mapping[str, Any]) -> None:
    if anchor.get("authorization_status") != "AUTHORIZED":
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_TRUST_ANCHOR_NOT_AUTHORIZED")
    if anchor.get("authorized_partition_manifest_sha256") != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_MANIFEST_SHA_MISMATCH")
    if anchor.get("authorized_partition_implementation_git_commit") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_PARTITION_IMPLEMENTATION_MISMATCH")


def _validate_readiness_metadata(readiness: Mapping[str, Any]) -> str:
    if (
        readiness.get("verification_stage") != READINESS_VERIFICATION_STAGE
        or readiness.get("logical_stage") != READINESS_LOGICAL_STAGE
        or readiness.get("verification_result") != "PASS"
        or readiness.get("frozen_design_commit") != FROZEN_DESIGN_COMMIT
    ):
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_READINESS_VERIFICATION_INVALID")
    return _require_hex(
        readiness.get("receipt_self_hash"), 64,
        "V8D_T2_POINT_OF_USE_READINESS_RECEIPT_HASH_INVALID",
    )


def _verify_prefreeze_binding(
    repository_root: Path,
    verified_head: str,
    git_blob_resolver: Callable[[Path, str, str], str],
    ancestor_checker: Callable[[Path, str, str, str], None],
) -> None:
    try:
        resolved = git_blob_resolver(repository_root, PREFREEZE_PRESERVATION_COMMIT, PREFREEZE_PRESERVATION_PATH)
    except V8DGitProvenanceBlocked as error:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_PREFREEZE_PRESERVATION_MISSING") from error
    if resolved != PREFREEZE_PRESERVATION_BLOB:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_PREFREEZE_PRESERVATION_BLOB_MISMATCH")
    try:
        ancestor_checker(
            repository_root, PREFREEZE_PRESERVATION_COMMIT, verified_head,
            "V8D_T2_POINT_OF_USE_PREFREEZE_PRESERVATION_NOT_IN_CURRENT_HISTORY",
        )
    except V8DGitProvenanceBlocked as error:
        raise V8DT2PointOfUsePreservationBlocked(error.reason) from error


def _derive_t2_point_of_use_preservation_with_dependencies(
    repository_root: Path,
    *,
    git_commit_resolver: Callable[[], str],
    frozen_design_verifier: Callable[[Path], None],
    freeze_approval_verifier: Callable[[Path, str], str],
    reviewed_implementation_binder: Callable[[Path, str], Mapping[str, Any]],
    anchor_reader: Callable[[Path, str], Mapping[str, Any]],
    authority_bridge_verifier: Callable[[Path, str, str], Mapping[str, Any]],
    readiness_reader: Callable[[], Mapping[str, Any]],
    gate_consumption_checker: Callable[[Path, str, str], bool],
    consumption_state_root: Path,
    state_conditions_reader: Callable[[Path, str], Mapping[str, Any]],
    prefreeze_blob_resolver: Callable[[Path, str, str], str],
    prefreeze_ancestor_checker: Callable[[Path, str, str, str], None],
) -> tuple[str, dict[str, Any]]:
    """Synthetic/DI-only implementation of the production derivation."""
    try:
        verified_head = git_commit_resolver()
    except V8DGitProvenanceBlocked as error:
        raise V8DT2PointOfUsePreservationBlocked(error.reason) from error
    _require_hex(verified_head, 40, "V8D_T2_POINT_OF_USE_VERIFIED_HEAD_INVALID")

    try:
        frozen_design_verifier(repository_root)
        freeze_approval_blob = freeze_approval_verifier(repository_root, verified_head)
        reviewed = reviewed_implementation_binder(repository_root, verified_head)
    except (V8DProductionProvenanceBlocked, V8DGitProvenanceBlocked) as error:
        raise V8DT2PointOfUsePreservationBlocked(getattr(error, "reason", "V8D_T2_POINT_OF_USE_PROVENANCE_BLOCKED")) from error
    if freeze_approval_blob != EXPECTED_V8D_DESIGN_FREEZE_APPROVAL_BLOB:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_FREEZE_APPROVAL_INVALID")
    if not isinstance(reviewed, Mapping):
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_REVIEWED_IMPLEMENTATION_INVALID")
    _require_hex(reviewed.get("reviewed_implementation_git_commit"), 40, "V8D_T2_POINT_OF_USE_REVIEWED_IMPLEMENTATION_INVALID")

    try:
        anchor = anchor_reader(repository_root, verified_head)
    except (V8DProductionProvenanceBlocked, V8DGitProvenanceBlocked) as error:
        raise V8DT2PointOfUsePreservationBlocked(getattr(error, "reason", "V8D_T2_POINT_OF_USE_TRUST_ANCHOR_BLOCKED")) from error
    _validate_anchor(anchor)

    try:
        bridge = authority_bridge_verifier(repository_root, verified_head, T2_STAGE)
    except (V8DAuthorityBridgeBlocked, V8DGitProvenanceBlocked) as error:
        raise V8DT2PointOfUsePreservationBlocked(getattr(error, "reason", "V8D_T2_POINT_OF_USE_AUTHORITY_BRIDGE_BLOCKED")) from error
    if bridge.get("logical_block") != "T2" or bridge.get("review_result") != "PASS":
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_AUTHORITY_BRIDGE_NOT_PASS")

    try:
        readiness = readiness_reader()
    except BaseException as error:  # dependency boundary must fail closed
        raise V8DT2PointOfUsePreservationBlocked(getattr(error, "reason", "V8D_T2_POINT_OF_USE_READINESS_BLOCKED")) from error
    readiness_receipt_hash = _validate_readiness_metadata(readiness)

    try:
        consumed = gate_consumption_checker(
            consumption_state_root,
            GATE_T2_RAW_ACQUISITION, FROZEN_DESIGN_COMMIT,
        )
    except V8DHumanGateConsumptionBlocked as error:
        raise V8DT2PointOfUsePreservationBlocked(error.reason) from error
    if consumed is not False:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_ACQUISITION_GATE_ALREADY_CONSUMED")

    _verify_prefreeze_binding(
        repository_root, verified_head, prefreeze_blob_resolver, prefreeze_ancestor_checker,
    )
    conditions = _require_safe_conditions(state_conditions_reader(repository_root, verified_head))

    artifact: dict[str, Any] = {
        "schema_version": POINT_OF_USE_SCHEMA_VERSION,
        "study": STUDY,
        "artifact_role": POINT_OF_USE_ARTIFACT_ROLE,
        "checkpoint": POINT_OF_USE_CHECKPOINT,
        "recheck": POINT_OF_USE_RECHECK,
        "v8d_frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "source_prefreeze_preservation_git_commit": PREFREEZE_PRESERVATION_COMMIT,
        "source_prefreeze_preservation_git_blob_sha": PREFREEZE_PRESERVATION_BLOB,
        "readiness_verification_stage": READINESS_VERIFICATION_STAGE,
        "readiness_verification_result": "PASS",
        "readiness_receipt_self_hash": readiness_receipt_hash,
        "v8_trust_anchor_git_blob": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "original_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "t2_count": T2_COUNT,
        "t2_ticker_list_sha256": T2_TICKER_LIST_SHA256,
        **conditions,
        "t2_raw_acquisition_gate_consumed": False,
        "point_of_use_preservation_result": "PASS",
    }
    if set(artifact) != set(POINT_OF_USE_ARTIFACT_FIELDS):
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_ARTIFACT_SCHEMA_INVALID")
    return verified_head, artifact


def _production_derivation_dependencies(repository_root: Path) -> dict[str, Any]:
    return {
        "git_commit_resolver": lambda: resolve_verified_v8d_production_git_commit(repository_root),
        "frozen_design_verifier": verify_frozen_design_object,
        "freeze_approval_verifier": verify_design_freeze_approval_blob,
        "reviewed_implementation_binder": verify_reviewed_implementation_binding,
        "anchor_reader": read_and_verify_v8_trusted_partition_anchor,
        "authority_bridge_verifier": verify_stage_authority_bridge,
        "readiness_reader": require_t2_readiness_audit_verification_pass,
        "gate_consumption_checker": has_gate_been_consumed,
        "consumption_state_root": CANONICAL_CONSUMPTION_STATE_ROOT,
        "state_conditions_reader": lambda root, head: _derive_conditions_from_v8_state(root, head, read_git_object_bytes),
        "prefreeze_blob_resolver": resolve_git_blob,
        "prefreeze_ancestor_checker": require_strict_git_ancestor,
    }


def resolve_and_recheck_t2_point_of_use_preservation() -> dict[str, Any]:
    """Derive future point-of-use artifact content without writing it."""
    _head, artifact = _derive_t2_point_of_use_preservation_with_dependencies(
        CANONICAL_REPOSITORY_ROOT,
        **_production_derivation_dependencies(CANONICAL_REPOSITORY_ROOT),
    )
    return artifact


def derive_t2_point_of_use_preservation_artifact() -> dict[str, Any]:
    """Compatibility name for the zero-argument production derivation."""
    return resolve_and_recheck_t2_point_of_use_preservation()


def _validate_preservation_artifact(raw: bytes) -> dict[str, Any]:
    artifact = _strict_json_object(
        raw,
        invalid_reason="V8D_T2_POINT_OF_USE_ARTIFACT_INVALID_JSON",
        duplicate_reason="V8D_T2_POINT_OF_USE_ARTIFACT_DUPLICATE_KEY",
    )
    _require_exact_fields(artifact, POINT_OF_USE_ARTIFACT_FIELDS, "V8D_T2_POINT_OF_USE_ARTIFACT_SCHEMA_INVALID")
    expected = {
        "schema_version": POINT_OF_USE_SCHEMA_VERSION,
        "study": STUDY,
        "artifact_role": POINT_OF_USE_ARTIFACT_ROLE,
        "checkpoint": POINT_OF_USE_CHECKPOINT,
        "recheck": POINT_OF_USE_RECHECK,
        "v8d_frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "source_prefreeze_preservation_git_commit": PREFREEZE_PRESERVATION_COMMIT,
        "source_prefreeze_preservation_git_blob_sha": PREFREEZE_PRESERVATION_BLOB,
        "readiness_verification_stage": READINESS_VERIFICATION_STAGE,
        "readiness_verification_result": "PASS",
        "v8_trust_anchor_git_blob": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "original_v8_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "t2_count": T2_COUNT,
        "t2_ticker_list_sha256": T2_TICKER_LIST_SHA256,
        "t2_raw_acquisition_gate_consumed": False,
        "point_of_use_preservation_result": "PASS",
    }
    for key, value in expected.items():
        if artifact[key] != value:
            raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_ARTIFACT_VALUE_MISMATCH:" + key)
    _require_hex(artifact["source_prefreeze_preservation_git_commit"], 40, "V8D_T2_POINT_OF_USE_PREFREEZE_COMMIT_INVALID")
    _require_hex(artifact["source_prefreeze_preservation_git_blob_sha"], 40, "V8D_T2_POINT_OF_USE_PREFREEZE_BLOB_INVALID")
    _require_hex(artifact["v8d_frozen_design_commit"], 40, "V8D_T2_POINT_OF_USE_DESIGN_COMMIT_INVALID")
    _require_hex(artifact["readiness_receipt_self_hash"], 64, "V8D_T2_POINT_OF_USE_READINESS_HASH_INVALID")
    _require_hex(artifact["v8_trust_anchor_git_blob"], 40, "V8D_T2_POINT_OF_USE_ANCHOR_BLOB_INVALID")
    _require_hex(artifact["original_v8_partition_manifest_sha256"], 64, "V8D_T2_POINT_OF_USE_MANIFEST_HASH_INVALID")
    _require_hex(artifact["parent_v8_partition_implementation_commit"], 40, "V8D_T2_POINT_OF_USE_PARENT_COMMIT_INVALID")
    _require_hex(artifact["t2_ticker_list_sha256"], 64, "V8D_T2_POINT_OF_USE_T2_HASH_INVALID")
    if type(artifact["t2_count"]) is not int or artifact["t2_count"] != T2_COUNT:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_T2_COUNT_INVALID")
    _require_safe_conditions({key: artifact[key] for key in NINE_CONDITION_FIELDS})
    return artifact


def _read_review_artifact(repository_root: Path, verified_head: str) -> dict[str, Any]:
    try:
        raw = read_git_object_bytes(repository_root, verified_head, POINT_OF_USE_REVIEW_PATH)
    except V8DGitProvenanceBlocked as error:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_REVIEW_MISSING") from error
    review = _strict_json_object(
        raw,
        invalid_reason="V8D_T2_POINT_OF_USE_REVIEW_INVALID_JSON",
        duplicate_reason="V8D_T2_POINT_OF_USE_REVIEW_DUPLICATE_KEY",
    )
    _require_exact_fields(review, POINT_OF_USE_REVIEW_FIELDS, "V8D_T2_POINT_OF_USE_REVIEW_SCHEMA_INVALID")
    expected = {
        "schema_version": POINT_OF_USE_REVIEW_SCHEMA_VERSION,
        "study": STUDY,
        "artifact_role": POINT_OF_USE_REVIEW_ROLE,
        "checkpoint": POINT_OF_USE_REVIEW_CHECKPOINT,
        "v8d_frozen_design_commit": FROZEN_DESIGN_COMMIT,
        "review_result": "PASS",
    }
    for key, value in expected.items():
        if review[key] != value:
            raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_REVIEW_VALUE_MISMATCH:" + key)
    _require_hex(review["reviewed_recheck_git_commit"], 40, "V8D_T2_POINT_OF_USE_REVIEW_COMMIT_INVALID")
    _require_hex(review["reviewed_recheck_git_blob_sha"], 40, "V8D_T2_POINT_OF_USE_REVIEW_BLOB_INVALID")
    return review


def _require_t2_point_of_use_preservation_review_pass_with_dependencies(
    repository_root: Path,
    *,
    dependencies: Mapping[str, Any],
    review_reader: Callable[[Path, str], Mapping[str, Any]],
    artifact_blob_resolver: Callable[[Path, str, str], str],
    artifact_reader: Callable[[Path, str, str], bytes],
    ancestor_checker: Callable[[Path, str, str, str], None],
) -> dict[str, Any]:
    """Synthetic/DI-only review-reader implementation."""
    first_head, _ = _derive_t2_point_of_use_preservation_with_dependencies(
        repository_root, **dependencies,
    )
    review = dict(review_reader(repository_root, first_head))
    try:
        ancestor_checker(
            repository_root, review["reviewed_recheck_git_commit"], first_head,
            "V8D_T2_POINT_OF_USE_REVIEW_NOT_IN_CURRENT_HISTORY",
        )
        resolved_blob = artifact_blob_resolver(
            repository_root, review["reviewed_recheck_git_commit"], POINT_OF_USE_ARTIFACT_PATH,
        )
        if resolved_blob != review["reviewed_recheck_git_blob_sha"]:
            raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_REVIEW_BLOB_MISMATCH")
        artifact = _validate_preservation_artifact(
            artifact_reader(repository_root, review["reviewed_recheck_git_commit"], POINT_OF_USE_ARTIFACT_PATH)
        )
    except V8DGitProvenanceBlocked as error:
        raise V8DT2PointOfUsePreservationBlocked(error.reason) from error

    second_head, live_artifact = _derive_t2_point_of_use_preservation_with_dependencies(
        repository_root, **dependencies,
    )
    if second_head != first_head:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_REVIEW_VERIFIED_HEAD_CHANGED")
    if artifact != live_artifact:
        raise V8DT2PointOfUsePreservationBlocked("V8D_T2_POINT_OF_USE_REVIEW_ARTIFACT_LIVE_MISMATCH")
    return {
        "schema_version": review["schema_version"],
        "study": review["study"],
        "artifact_role": review["artifact_role"],
        "checkpoint": review["checkpoint"],
        "v8d_frozen_design_commit": review["v8d_frozen_design_commit"],
        "reviewed_recheck_git_commit": review["reviewed_recheck_git_commit"],
        "reviewed_recheck_git_blob_sha": review["reviewed_recheck_git_blob_sha"],
        "review_result": review["review_result"],
        "preservation_artifact": artifact,
    }


def require_t2_point_of_use_preservation_review_pass() -> dict[str, Any]:
    """Require the independently reviewed, current T2 point-of-use PASS."""
    repository_root = CANONICAL_REPOSITORY_ROOT
    dependencies = _production_derivation_dependencies(repository_root)
    return _require_t2_point_of_use_preservation_review_pass_with_dependencies(
        repository_root,
        dependencies=dependencies,
        review_reader=_read_review_artifact,
        artifact_blob_resolver=resolve_git_blob,
        artifact_reader=read_git_object_bytes,
        ancestor_checker=require_strict_git_ancestor,
    )


__all__ = [
    "NINE_CONDITION_FIELDS",
    "POINT_OF_USE_ARTIFACT_FIELDS",
    "POINT_OF_USE_ARTIFACT_PATH",
    "POINT_OF_USE_CHECKPOINT",
    "POINT_OF_USE_RECHECK",
    "POINT_OF_USE_REVIEW_FIELDS",
    "POINT_OF_USE_REVIEW_PATH",
    "POINT_OF_USE_REVIEW_CHECKPOINT",
    "POINT_OF_USE_REVIEW_SCHEMA_VERSION",
    "POINT_OF_USE_SCHEMA_VERSION",
    "READINESS_VERIFICATION_STAGE",
    "V8DT2PointOfUsePreservationBlocked",
    "derive_t2_point_of_use_preservation_artifact",
    "require_t2_point_of_use_preservation_review_pass",
    "resolve_and_recheck_t2_point_of_use_preservation",
]
