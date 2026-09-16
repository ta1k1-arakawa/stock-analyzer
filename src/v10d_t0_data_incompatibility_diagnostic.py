"""V10D data-incompatibility diagnostic successor.

This module is deliberately diagnostic-only.  Phase A validates repository
and cache metadata without opening price payloads.  The protected diagnostic
path reuses the reviewed V10C cache loader, then checks only the inherited
structural contracts needed to localize the first data incompatibility.  It
never fits a model, creates predictions, computes scores, or emits a T0
decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src import v10c_t0_successor_training_input_binding as v10c
from src import v9_009_t0_top1_kill_screen as v9


STUDY_IDENTITY = "V10D_T0_DATA_INCOMPATIBILITY_DIAGNOSTIC_SUCCESSOR"
AUTHORITATIVE_BRANCH = "v9-cross-sectional-close-auction-design"
EXPECTED_REPOSITORY_URL = "https://github.com/ta1k1-arakawa/stock-analyzer.git"

DESIGN_FILE = "V10D_T0_DATA_INCOMPATIBILITY_DIAGNOSTIC_SUCCESSOR_DESIGN_DRAFT.md"
DESIGN_COMMIT = "c83bd856c24d59b5230908da22d7790d51a9f802"
DESIGN_BLOB = "5f1bcf90228bab0f9fa8dd463a2d9544aa74457d"
DESIGN_SHA256 = "42be313100bf384e0d81a33e2b5a3cf2ee4353773018f84c1f8e66f91d3dbc13"
FREEZE_APPROVAL_FILE = "V10D_DESIGN_FREEZE_APPROVAL.json"
FREEZE_APPROVAL_BLOB = "66f6400ec9f1ea8eb2307afa26ab9c5d864c8b32"
FREEZE_APPROVAL_SHA256 = "068e99c9c159b6eff115a9bc749a843ec489d33a8177496110b30eda6bbd22f7"

V10C_BINDING_FILE = "src/v10c_t0_successor_training_input_binding.py"
V10C_BINDING_BLOB = "8ad18fe101aa0afe7aaeb7e0456d77ea86e9cb0b"
EVALUATION_MANIFEST_SHA256 = "797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5"
TRAINING_MANIFEST_SHA256 = "887c031a004f91a080fa53ab511711fff92c92527cb119878ab2c295ee13cd44"
V4_UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
V4_TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
V10A_CALENDAR_SHA256 = "2e9fbfbf64777d448e5a98dd85d5bb4c679cd22b19b80a07b78deac9aad507e0"
EVALUATION_PAYLOAD_COUNT = 300
TRAINING_SUCCESS_COUNT = 283
TRAINING_FAILED_COUNT = 17

DIAGNOSTIC_BOUNDARY_TOKEN = "V10D_PROTECTED_DIAGNOSTIC_BOUNDARY_ATTEMPT_1"
SAFE_RESULT_SCHEMA = "V10D_DATA_INCOMPATIBILITY_DIAGNOSTIC_SAFE_RESULT_V1"
PHASE_A_SCHEMA = "V10D_PHASE_A_METADATA_PREFLIGHT_V1"
RESULT_DATA = "DATA_INCOMPATIBILITY_DIAGNOSTIC"
RESULT_IMPLEMENTATION = "IMPLEMENTATION_FAILURE"
DATA_STAGES = (
    "INPUT_BYTE_OR_FILESET_CONTRACT",
    "PARSER_NORMALIZATION_CONTRACT",
    "COMBINED_SERIES_CONTRACT",
    "FEATURE_TARGET_DATASET_CONTRACT",
    "FORMAL_SCORING_PRECONDITION_CONTRACT",
    "POST_SCORING_STRUCTURAL_TARGET_CONTRACT",
    "UNKNOWN_DATA_INCOMPATIBILITY",
)

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class V10DPreflightFailure(RuntimeError):
    """A safe repository/metadata failure before protected payload access."""


class V10DDiagnosticImplementationFailure(RuntimeError):
    """A diagnostic implementation or wrapper failure."""


@dataclass(frozen=True)
class PhaseAMetadata:
    successor_metadata: v10c.SuccessorMetadata
    implementation_sha: str


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_sha1(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA1.fullmatch(value))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _is_reparse(stat_result: Any) -> bool:
    return bool(getattr(stat_result, "st_file_attributes", 0) & 0x400) or bool(
        getattr(stat_result, "st_reparse_tag", 0)
    )


def _safe_file(path: Path, failure_type: type[Exception]) -> None:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise failure_type("REQUIRED_FILE_UNAVAILABLE") from exc
    if path.is_symlink() or _is_reparse(stat_result) or not path.is_file():
        raise failure_type("REQUIRED_FILE_UNSAFE")


def _strict_json(raw: bytes, failure_type: type[Exception]) -> dict[str, Any]:
    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise failure_type("JSON_METADATA_INVALID") from exc
    if not isinstance(value, dict):
        raise failure_type("JSON_METADATA_SCHEMA_INVALID")
    return value


def _read_strict_json(path: Path, failure_type: type[Exception]) -> tuple[bytes, dict[str, Any]]:
    _safe_file(path, failure_type)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise failure_type("JSON_METADATA_UNAVAILABLE") from exc
    return raw, _strict_json(raw, failure_type)


def _run_git(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise V10DPreflightFailure("GIT_PROVENANCE_UNAVAILABLE") from exc
    if completed.returncode != 0:
        raise V10DPreflightFailure("GIT_PROVENANCE_UNAVAILABLE")
    output = completed.stdout
    if not isinstance(output, bytes):
        raise V10DPreflightFailure("GIT_OUTPUT_TYPE_INVALID")
    try:
        return output.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise V10DPreflightFailure("GIT_OUTPUT_UTF8_INVALID") from exc


def _repository_url_is_authoritative(value: str) -> bool:
    return value.strip().rstrip("/") in {
        EXPECTED_REPOSITORY_URL,
        "https://github.com/ta1k1-arakawa/stock-analyzer",
    }


_APPROVAL_KEYS = {
    "schema_version",
    "artifact_role",
    "study",
    "design_document",
    "approval_scope",
    "approval_status",
    "human_design_freeze_complete",
    "frozen_design_git_commit",
    "frozen_design_git_blob_sha1",
    "frozen_design_sha256",
    "final_independent_review_authority",
    "final_independent_review_design_commit",
    "final_independent_review_result",
    "methodology_change_after_freeze_requires",
    "implementation_phase_may_begin_only_after_approval_record_gpt_review",
    "implementation_authorized_by_this_artifact",
    "diagnostic_execution_authorized",
    "protected_payload_read_authorized",
    "historical_evaluation_authorized",
    "model_fit_authorized",
    "t0_authorized",
    "network_access_authorized",
    "refetch_authorized",
    "cache_repair_authorized",
    "substitution_authorized",
    "private_sealed_access_authorized",
    "future_point_of_use_authority_required",
    "future_profitability_established",
    "next_required_action",
}


def _validate_freeze_approval(approval: Mapping[str, Any], design_sha256: str) -> None:
    if set(approval) != _APPROVAL_KEYS:
        raise V10DPreflightFailure("FREEZE_APPROVAL_SCHEMA_INVALID")
    expected = {
        "schema_version": "V10D_DESIGN_FREEZE_APPROVAL_V1",
        "artifact_role": "DESIGN_FREEZE_APPROVAL",
        "study": STUDY_IDENTITY,
        "design_document": DESIGN_FILE,
        "approval_scope": "DESIGN_FREEZE_ONLY",
        "approval_status": "APPROVED",
        "human_design_freeze_complete": True,
        "frozen_design_git_commit": DESIGN_COMMIT,
        "frozen_design_git_blob_sha1": DESIGN_BLOB,
        "frozen_design_sha256": design_sha256,
        "final_independent_review_authority": "GPT-5.6 Sol",
        "final_independent_review_design_commit": DESIGN_COMMIT,
        "final_independent_review_result": "PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_1",
        "methodology_change_after_freeze_requires": "NEW_STUDY_REQUIRED",
        "implementation_phase_may_begin_only_after_approval_record_gpt_review": True,
        "implementation_authorized_by_this_artifact": False,
        "diagnostic_execution_authorized": False,
        "protected_payload_read_authorized": False,
        "historical_evaluation_authorized": False,
        "model_fit_authorized": False,
        "t0_authorized": False,
        "network_access_authorized": False,
        "refetch_authorized": False,
        "cache_repair_authorized": False,
        "substitution_authorized": False,
        "private_sealed_access_authorized": False,
        "future_point_of_use_authority_required": True,
        "future_profitability_established": False,
        "next_required_action": "GPT_EXACT_SHA_V10D_DESIGN_FREEZE_APPROVAL_REVIEW",
    }
    if any(approval[key] != value for key, value in expected.items()):
        raise V10DPreflightFailure("FREEZE_APPROVAL_SEMANTICS_INVALID")
    if not _is_sha256(approval["frozen_design_sha256"]):
        raise V10DPreflightFailure("FREEZE_APPROVAL_DESIGN_SHA256_INVALID")


def _verify_repository_and_freeze(repository_root: Path, implementation_sha: str) -> None:
    if not _is_sha1(implementation_sha):
        raise V10DPreflightFailure("IMPLEMENTATION_SHA_INVALID")
    try:
        if not _repository_url_is_authoritative(_run_git(repository_root, "remote", "get-url", "origin")):
            raise V10DPreflightFailure("REPOSITORY_IDENTITY_MISMATCH")
        if _run_git(repository_root, "rev-parse", "--abbrev-ref", "HEAD") != AUTHORITATIVE_BRANCH:
            raise V10DPreflightFailure("BRANCH_MISMATCH")
        if _run_git(repository_root, "rev-parse", "HEAD") != implementation_sha:
            raise V10DPreflightFailure("IMPLEMENTATION_HEAD_MISMATCH")
        if _run_git(repository_root, "rev-parse", f"refs/remotes/origin/{AUTHORITATIVE_BRANCH}") != implementation_sha:
            raise V10DPreflightFailure("REMOTE_HEAD_MISMATCH")
        if _run_git(repository_root, "status", "--porcelain", "--untracked-files=all"):
            raise V10DPreflightFailure("WORKTREE_DIRTY")
    except V10DPreflightFailure:
        raise
    except Exception as exc:
        raise V10DPreflightFailure("REPOSITORY_PROVENANCE_UNAVAILABLE") from exc

    design_path = repository_root / DESIGN_FILE
    approval_path = repository_root / FREEZE_APPROVAL_FILE
    _safe_file(design_path, V10DPreflightFailure)
    _safe_file(approval_path, V10DPreflightFailure)
    try:
        design_raw = design_path.read_bytes()
        approval_raw = approval_path.read_bytes()
    except OSError as exc:
        raise V10DPreflightFailure("FROZEN_ARTIFACT_UNAVAILABLE") from exc
    if _run_git(repository_root, "rev-parse", f"HEAD:{DESIGN_FILE}") != DESIGN_BLOB:
        raise V10DPreflightFailure("DESIGN_BLOB_MISMATCH")
    if _run_git(repository_root, "rev-parse", f"HEAD:{FREEZE_APPROVAL_FILE}") != FREEZE_APPROVAL_BLOB:
        raise V10DPreflightFailure("FREEZE_APPROVAL_BLOB_MISMATCH")
    if _sha256(design_raw) != DESIGN_SHA256:
        raise V10DPreflightFailure("DESIGN_SHA256_MISMATCH")
    if _sha256(approval_raw) != FREEZE_APPROVAL_SHA256:
        raise V10DPreflightFailure("FREEZE_APPROVAL_SHA256_MISMATCH")
    approval = _strict_json(approval_raw, V10DPreflightFailure)
    _validate_freeze_approval(approval, _sha256(design_raw))

    inherited_blobs = {
        v10c.DESIGN_FILE: v10c.DESIGN_BLOB,
        V10C_BINDING_FILE: V10C_BINDING_BLOB,
        v10c.V10A_DESIGN_FILE: v10c.V10A_DESIGN_BLOB,
        v10c.V10B_SOURCE_FILE: v10c.V10B_SOURCE_BLOB,
        v10c.V9_CORE_FILE: v10c.V9_CORE_BLOB,
        v10c.ADOPTION_RECORD_FILE: v10c.ADOPTION_RECORD_BLOB,
        v10c.DESIGN_FREEZE_APPROVAL_FILE: v10c.DESIGN_FREEZE_APPROVAL_BLOB,
    }
    for path, blob in inherited_blobs.items():
        if _run_git(repository_root, "rev-parse", f"HEAD:{path}") != blob:
            raise V10DPreflightFailure("INHERITED_DEPENDENCY_BLOB_MISMATCH")


def phase_a_metadata_preflight(
    repository_root: Path,
    training_root: Path,
    evaluation_root: Path,
    universe_csv: Path,
    implementation_sha: str,
) -> PhaseAMetadata:
    """Run metadata-only preflight; no payload parser is reachable here."""
    _verify_repository_and_freeze(repository_root, implementation_sha)
    try:
        metadata = v10c.phase_a_metadata_preflight(
            repository_root,
            training_root,
            evaluation_root,
            universe_csv,
            implementation_sha,
        )
    except v10c.SuccessorPreflightFailure as exc:
        raise V10DPreflightFailure("INHERITED_PHASE_A_FAILURE") from exc
    except Exception as exc:
        raise V10DPreflightFailure("INHERITED_PHASE_A_FAILURE") from exc
    return PhaseAMetadata(metadata, implementation_sha)


_INPUT_REASONS = {
    "CACHE_MANIFEST_INVALID",
    "CACHE_MANIFEST_IDENTITY_MISMATCH",
    "CACHE_MANIFEST_MODE_INVALID",
    "CACHE_MANIFEST_UNIVERSE_MISMATCH",
    "CACHE_PAYLOAD_SET_INVALID",
    "CACHE_PAYLOAD_SCHEMA_INVALID",
    "CACHE_PAYLOAD_PATH_INVALID",
    "CACHE_PAYLOAD_MISSING",
    "TRAINING_PAYLOAD_HASH_CLOSURE_INVALID",
    "CACHE_DIRECTORY_UNREADABLE",
    "CACHE_PAYLOAD_READ_INVALID",
}
_PARSER_REASONS = {
    "CACHE_PAYLOAD_HASH_MISMATCH",
    "CACHE_PAYLOAD_JSON_INVALID",
    "CACHE_PAYLOAD_LENGTH_INVALID",
    "CACHE_PAYLOAD_DATE_INVALID",
    "CACHE_PAYLOAD_SPLIT_EVENT_INVALID",
    "CACHE_PAYLOAD_SCHEMA_INVALID",
    "CANONICAL_CODE_INVALID",
    "DATE_INVALID",
    "OHLCV_FRAME_INVALID",
    "OHLCV_REQUIRED_COLUMNS_MISSING",
    "DUPLICATE_OR_INVALID_PRICE_DATE",
    "NONFINITE_OR_NONPOSITIVE_OHLCV",
    "SPLIT_ACTION_SCHEMA_INVALID",
    "SPLIT_RATIO_INVALID",
    "D0_DATE_INVALID",
    "NONFINITE_FEATURE",
    "FEATURE_HISTORY_UNAVAILABLE",
    "FEATURE_ATR_UNAVAILABLE",
    "D0_PRICE_MISSING",
    "D0_NOT_IN_CALENDAR",
    "D1_D3_CALENDAR_TAIL_MISSING",
    "TARGET_PRICE_MISSING",
    "NONFINITE_TARGET",
}
_COMBINED_REASONS = {
    "EVALUATION_PAYLOAD_MISSING",
    "COMBINED_PRICE_EMPTY",
    "DUPLICATE_COMBINED_PRICE_DATE",
    "DUPLICATE_COMBINED_SPLIT_EVENT",
}


def _stage_for_cache_reason(reason: object) -> str:
    if reason in _INPUT_REASONS:
        return "INPUT_BYTE_OR_FILESET_CONTRACT"
    if reason in _PARSER_REASONS:
        return "PARSER_NORMALIZATION_CONTRACT"
    if reason in _COMBINED_REASONS:
        return "COMBINED_SERIES_CONTRACT"
    return "UNKNOWN_DATA_INCOMPATIBILITY"


def _structural_formal_preconditions(
    dataset: pd.DataFrame, calendar_dates: Sequence[object]
) -> pd.DataFrame:
    """Reproduce score_formal_dataset's pre-fit checks without fitting."""
    validated = v9._validate_dataset(dataset)
    calendar = v9.normalize_calendar(calendar_dates)
    formal = validated[validated["d0"].between(v9.FORMAL_SIGNAL_START, v9.FORMAL_SIGNAL_END)].copy()
    if formal.empty or set(formal["d0"].dt.year.unique()) != set(v9.FORMAL_YEARS):
        raise v9.T0DataIncompatible("FORMAL_SIGNAL_YEARS_INCOMPLETE")
    months = sorted({(int(day.year), int(day.month)) for day in formal["d0"]})
    for year, month in months:
        v9.month_start(calendar, year, month)
        training = v9.causal_training_rows(validated, calendar, year, month)
        if training.empty:
            raise v9.T0DataIncompatible("INSUFFICIENT_CAUSAL_TRAINING_DATA")
        feature_values = training.loc[:, list(v9.FEATURE_COLUMNS)].to_numpy(dtype=float)
        target_values = training["target_percentile"].to_numpy(dtype=float)
        if not np.isfinite(feature_values).all() or not np.isfinite(target_values).all():
            raise v9.T0DataIncompatible("NONFINITE_TRAINING_VALUE")
    return validated


def _structural_post_scoring_conditions(dataset: pd.DataFrame) -> None:
    """Check only structural equivalents of post-scoring T0 conditions."""
    validated = v9._validate_dataset(dataset)
    formal = validated[validated["d0"].dt.year.isin(v9.KILL_SCREEN_YEARS)]
    if "target_available" in validated.columns:
        if not formal["target_available"].all():
            raise v9.T0DataIncompatible("FORMAL_TARGET_UNAVAILABLE")
    if formal["target_percentile"].isna().any():
        raise v9.T0DataIncompatible("FORMAL_TARGET_UNAVAILABLE")
    for year in v9.KILL_SCREEN_YEARS:
        if formal[formal["d0"].dt.year == year].empty:
            raise v9.T0DataIncompatible("KILL_SCREEN_YEAR_INCOMPLETE")


def _safe_provenance() -> dict[str, Any]:
    return {
        "design_git_commit": DESIGN_COMMIT,
        "design_git_blob_sha1": DESIGN_BLOB,
        "design_sha256": DESIGN_SHA256,
        "freeze_approval_git_blob_sha1": FREEZE_APPROVAL_BLOB,
        "freeze_approval_sha256": FREEZE_APPROVAL_SHA256,
        "training_manifest_sha256": TRAINING_MANIFEST_SHA256,
        "evaluation_manifest_sha256": EVALUATION_MANIFEST_SHA256,
        "v4_universe_csv_sha256": V4_UNIVERSE_CSV_SHA256,
        "v4_ticker_list_sha256": V4_TICKER_LIST_SHA256,
        "v10a_calendar_sha256": V10A_CALENDAR_SHA256,
    }


def _safe_result(
    implementation_sha: str,
    result_class: str,
    first_failed_stage: str | None,
) -> dict[str, Any]:
    result = {
        "schema_version": SAFE_RESULT_SCHEMA,
        "study": STUDY_IDENTITY,
        "implementation_sha": implementation_sha,
        "provenance": _safe_provenance(),
        "counts": {
            "training_success_count": TRAINING_SUCCESS_COUNT,
            "training_failed_count": TRAINING_FAILED_COUNT,
            "evaluation_payload_count": EVALUATION_PAYLOAD_COUNT,
        },
        "result_class": result_class,
        "first_failed_stage": first_failed_stage,
        "validation": {
            "phase_a_metadata_only": True,
            "protected_payload_boundary_crossed": True,
            "localized_data_incompatibility": result_class == RESULT_DATA,
        },
        "authority_consumed": True,
        "retry_authorized": False,
        "execution_counters": {
            "network_requests": 0,
            "model_fits": 0,
            "t0_runs": 0,
        },
        "future_profitability_established": False,
    }
    return validate_safe_result(result)


_SAFE_RESULT_KEYS = {
    "schema_version",
    "study",
    "implementation_sha",
    "provenance",
    "counts",
    "result_class",
    "first_failed_stage",
    "validation",
    "authority_consumed",
    "retry_authorized",
    "execution_counters",
    "future_profitability_established",
}
_SAFE_PROVENANCE_KEYS = set(_safe_provenance())
_SAFE_COUNT_KEYS = {"training_success_count", "training_failed_count", "evaluation_payload_count"}
_SAFE_VALIDATION_KEYS = {
    "phase_a_metadata_only",
    "protected_payload_boundary_crossed",
    "localized_data_incompatibility",
}
_SAFE_COUNTER_KEYS = {"network_requests", "model_fits", "t0_runs"}


def validate_safe_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping) or set(result) != _SAFE_RESULT_KEYS:
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_SCHEMA_INVALID")
    if result["schema_version"] != SAFE_RESULT_SCHEMA or result["study"] != STUDY_IDENTITY:
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_IDENTITY_INVALID")
    if not _is_sha1(result["implementation_sha"]):
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_IMPLEMENTATION_SHA_INVALID")
    if set(result["provenance"]) != _SAFE_PROVENANCE_KEYS:
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_PROVENANCE_SCHEMA_INVALID")
    if any(not _is_sha256(value) and key.endswith("sha256") for key, value in result["provenance"].items()):
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_PROVENANCE_SHA_INVALID")
    if not _is_sha1(result["provenance"]["design_git_blob_sha1"]) or not _is_sha1(
        result["provenance"]["freeze_approval_git_blob_sha1"]
    ):
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_PROVENANCE_BLOB_INVALID")
    if set(result["counts"]) != _SAFE_COUNT_KEYS or any(
        type(value) is not int or value < 0 for value in result["counts"].values()
    ):
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_COUNTS_INVALID")
    if result["result_class"] not in {RESULT_DATA, RESULT_IMPLEMENTATION}:
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_CLASS_INVALID")
    stage = result["first_failed_stage"]
    if result["result_class"] == RESULT_DATA:
        if stage not in DATA_STAGES:
            raise V10DDiagnosticImplementationFailure("SAFE_RESULT_STAGE_INVALID")
    elif stage is not None:
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_IMPLEMENTATION_STAGE_INVALID")
    if set(result["validation"]) != _SAFE_VALIDATION_KEYS or any(
        type(value) is not bool for value in result["validation"].values()
    ):
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_VALIDATION_INVALID")
    if result["validation"]["phase_a_metadata_only"] is not True or result["validation"][
        "protected_payload_boundary_crossed"
    ] is not True:
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_BOUNDARY_INVALID")
    if result["validation"]["localized_data_incompatibility"] is not (result["result_class"] == RESULT_DATA):
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_LOCALIZATION_INVALID")
    if type(result["authority_consumed"]) is not bool or result["authority_consumed"] is not True:
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_AUTHORITY_INVALID")
    if type(result["retry_authorized"]) is not bool or result["retry_authorized"] is not False:
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_RETRY_INVALID")
    if set(result["execution_counters"]) != _SAFE_COUNTER_KEYS or any(
        type(value) is not int or value < 0 for value in result["execution_counters"].values()
    ):
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_COUNTERS_INVALID")
    if type(result["future_profitability_established"]) is not bool or result[
        "future_profitability_established"
    ] is not False:
        raise V10DDiagnosticImplementationFailure("SAFE_RESULT_PROFITABILITY_INVALID")
    return dict(result)


def phase_a_safe_result(implementation_sha: str) -> dict[str, Any]:
    if not _is_sha1(implementation_sha):
        raise V10DPreflightFailure("IMPLEMENTATION_SHA_INVALID")
    return {
        "schema_version": PHASE_A_SCHEMA,
        "study": STUDY_IDENTITY,
        "implementation_sha": implementation_sha,
        "phase_a_status": "PASS",
        "payload_reads": 0,
        "parser_calls": 0,
        "network_requests": 0,
        "authority_consumed": False,
        "protected_payload_read_authorized": False,
        "future_profitability_established": False,
    }


def run_diagnostic(
    phase_a: PhaseAMetadata,
    calendar_dates: Sequence[object],
    *,
    authority_boundary_token: str | None,
) -> dict[str, Any]:
    """Localize one exact fixed-artifact failure after the explicit guard."""
    if authority_boundary_token != DIAGNOSTIC_BOUNDARY_TOKEN:
        raise V10DPreflightFailure("DIAGNOSTIC_AUTHORITY_BOUNDARY_REQUIRED")
    implementation_sha = phase_a.implementation_sha
    try:
        frames, actions, _provenance, universe = v10c.load_successor_cache_pair(
            phase_a.successor_metadata
        )
    except v9.T0DataIncompatible as exc:
        return _safe_result(implementation_sha, RESULT_DATA, _stage_for_cache_reason(exc.reason))
    except Exception:
        return _safe_result(implementation_sha, RESULT_IMPLEMENTATION, None)

    try:
        dataset = v9.build_dataset(frames, universe, calendar_dates, actions)
    except v9.T0DataIncompatible:
        return _safe_result(implementation_sha, RESULT_DATA, "FEATURE_TARGET_DATASET_CONTRACT")
    except Exception:
        return _safe_result(implementation_sha, RESULT_IMPLEMENTATION, None)

    try:
        _structural_formal_preconditions(dataset, calendar_dates)
    except v9.T0DataIncompatible:
        return _safe_result(implementation_sha, RESULT_DATA, "FORMAL_SCORING_PRECONDITION_CONTRACT")
    except Exception:
        return _safe_result(implementation_sha, RESULT_IMPLEMENTATION, None)

    try:
        _structural_post_scoring_conditions(dataset)
    except v9.T0DataIncompatible:
        return _safe_result(implementation_sha, RESULT_DATA, "POST_SCORING_STRUCTURAL_TARGET_CONTRACT")
    except (v9.T0ImplementationFailure, Exception):
        return _safe_result(implementation_sha, RESULT_IMPLEMENTATION, None)

    return _safe_result(implementation_sha, RESULT_IMPLEMENTATION, None)


def _read_calendar_dates(path: Path) -> list[str]:
    raw, value = _read_strict_json(path, V10DPreflightFailure)
    del raw
    dates = value.get("trading_dates")
    if not isinstance(dates, list) or any(type(item) is not str for item in dates):
        raise V10DPreflightFailure("CALENDAR_METADATA_INVALID")
    return dates


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V10D safe data-incompatibility diagnostic")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--phase-a-only", action="store_true")
    mode.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--training-cache", type=Path, required=True)
    parser.add_argument("--evaluation-cache", type=Path, required=True)
    parser.add_argument("--universe-csv", type=Path, required=True)
    parser.add_argument("--implementation-sha", required=True)
    parser.add_argument("--calendar-json", type=Path)
    parser.add_argument("--authority-boundary-token")
    return parser.parse_args(argv)


def _write_json(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _arguments(argv)
        phase_a = phase_a_metadata_preflight(
            arguments.repository_root,
            arguments.training_cache,
            arguments.evaluation_cache,
            arguments.universe_csv,
            arguments.implementation_sha,
        )
        if arguments.phase_a_only:
            _write_json(phase_a_safe_result(phase_a.implementation_sha))
            return 0
        if arguments.calendar_json is None:
            raise V10DPreflightFailure("CALENDAR_JSON_REQUIRED")
        if arguments.authority_boundary_token != DIAGNOSTIC_BOUNDARY_TOKEN:
            raise V10DPreflightFailure("DIAGNOSTIC_AUTHORITY_BOUNDARY_REQUIRED")
        calendar_dates = _read_calendar_dates(arguments.calendar_json)
        diagnostic_result = run_diagnostic(
            phase_a,
            calendar_dates,
            authority_boundary_token=arguments.authority_boundary_token,
        )
        _write_json(validate_safe_result(diagnostic_result))
        return 0
    except V10DPreflightFailure:
        print("V10D_PRE_GATE_FAILURE", file=sys.stderr)
        return 4
    except V10DDiagnosticImplementationFailure:
        print("V10D_IMPLEMENTATION_FAILURE", file=sys.stderr)
        return 3
    except Exception:
        print("V10D_IMPLEMENTATION_FAILURE", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITATIVE_BRANCH",
    "DATA_STAGES",
    "DESIGN_BLOB",
    "DESIGN_COMMIT",
    "DESIGN_FILE",
    "DIAGNOSTIC_BOUNDARY_TOKEN",
    "FREEZE_APPROVAL_BLOB",
    "FREEZE_APPROVAL_FILE",
    "PhaseAMetadata",
    "V10DPreflightFailure",
    "V10DDiagnosticImplementationFailure",
    "main",
    "phase_a_metadata_preflight",
    "phase_a_safe_result",
    "run_diagnostic",
    "validate_safe_result",
]
