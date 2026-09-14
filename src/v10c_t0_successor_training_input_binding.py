"""V10C successor input binding for the reviewed V9 TOP1 T0.

This adapter is deliberately separate from the historical V9 cache loader.
Phase A validates only repository, manifest, receipt, directory and file-set
metadata.  ``load_successor_cache_pair`` is the later full-load boundary: it
is the only function in this module that reads locked/evaluation payload
bytes and it reuses the inherited V9 chart parser and V10B hash validator.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src import v10b_training_cache_reacquisition as v10b
from src.v9_009_t0_top1_kill_screen import (
    T0DataIncompatible,
    _parse_chart_payload,
    build_dataset,
    load_fixed_universe,
    make_safe_result,
    score_formal_dataset,
    screen_top1,
)


STUDY_IDENTITY = "V10C_LOCKED_TRAINING_CACHE_PROVENANCE_ADOPTION_SUCCESSOR"
DESIGN_COMMIT = "0e65b170caef9958c31b7efa6802aa0313571d57"
DESIGN_BLOB = "52572c53934f9de1a182a7f580df91e52086de9c"
DESIGN_FILE = "V10C_T0_SUCCESSOR_TRAINING_INPUT_BINDING_DESIGN_DRAFT.md"
DESIGN_FREEZE_APPROVAL_FILE = "V10C_DESIGN_FREEZE_APPROVAL.json"
DESIGN_FREEZE_APPROVAL_BLOB = "4676eb87c10dfc47fc19be387bfcb2ca17ebc59d"
ADOPTION_RECORD_FILE = "V10C_TRAINING_PROVENANCE_ADOPTION_RECORD.json"
ADOPTION_RECORD_BLOB = "7febddd4af7c82fe00ef7ba618403f4dcf8f6758"
ADOPTION_RECORD_SHA256 = "9f850c793434f574655b1a37592dc435d7b085a1798457a3b589621983badb68"
V9_CORE_FILE = "src/v9_009_t0_top1_kill_screen.py"
V9_CORE_BLOB = "42753ddc75c6d7c016ac148991935bf316d9d14e"
V10B_SOURCE_FILE = "src/v10b_training_cache_reacquisition.py"
V10B_SOURCE_BLOB = "abb17129870241f18eba2f31a49c5606475a1a0d"
V10A_DESIGN_FILE = "V10A_T0_CALENDAR_INPUT_BINDING_BRIDGE_DESIGN_DRAFT.md"
V10A_DESIGN_BLOB = "6df95aa8354c3d335a51747ee98ed9f2741c2410"
TRAINING_MANIFEST_SHA256 = "887c031a004f91a080fa53ab511711fff92c92527cb119878ab2c295ee13cd44"
OLD_TRAINING_MANIFEST_SHA256 = "72ae3db1186f2c9c113b1bafe1d37fb74a5627ac7ceed1dfc2473a24e060de85"
ATTEMPT_RECEIPT_SHA256 = "44387e68bfc5de8bf54de8ca9694ba9f12bad2be779224946888eaea36d96eb7"
EVALUATION_MANIFEST_SHA256 = "797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5"
UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
TICKER_LIST_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
TICKER_COUNT = 300
SUCCESS_COUNT = 283
FAILED_COUNT = 17
TRAINING_SCHEMA = "V10B_TRAINING_CACHE_MANIFEST_V1"
ATTEMPT_SCHEMA = "V10B_ACQUISITION_ATTEMPT_RECEIPT_V1"
EVALUATION_SCHEMA = 1
LOCKED_RAW = "locked_raw"
RAW = "raw"
MANIFEST_FILE = "cache_manifest.json"
ATTEMPT_RECEIPT_FILE = "V10B_ACQUISITION_ATTEMPT_RECEIPT.json"
PRICE_FROM = "2015-01-01"
PRICE_TO = "2019-12-31"
EVAL_START_WITH_TRAINING = pd.Timestamp("2020-01-01")
EVAL_START_WITHOUT_TRAINING = pd.Timestamp("2019-01-01")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")


class SuccessorPreflightFailure(RuntimeError):
    """Governance/provenance failure before any payload byte read."""


@dataclass(frozen=True)
class SuccessorMetadata:
    candidate_root: Path
    evaluation_root: Path
    universe_csv: Path
    ticker_order: tuple[str, ...]
    training_manifest: dict[str, Any]
    evaluation_manifest: dict[str, Any]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _is_reparse(stat_result: os.stat_result) -> bool:
    return bool(getattr(stat_result, "st_file_attributes", 0) & 0x400) or bool(
        getattr(stat_result, "st_reparse_tag", 0)
    )


def _assert_existing_chain(path: Path, *, error: type[Exception] = SuccessorPreflightFailure) -> None:
    current = path
    while True:
        try:
            stat_result = current.lstat()
        except OSError as exc:
            raise error("PATH_ANCESTOR_UNAVAILABLE") from exc
        if current.is_symlink() or _is_reparse(stat_result):
            raise error("PATH_ANCESTOR_UNSAFE")
        if current.parent == current:
            return
        current = current.parent


def _safe_file(path: Path, *, error: type[Exception]) -> None:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise error("REQUIRED_FILE_UNAVAILABLE") from exc
    if path.is_symlink() or _is_reparse(stat_result) or not path.is_file():
        raise error("REQUIRED_FILE_UNSAFE")


def _safe_directory(path: Path, *, error: type[Exception]) -> None:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise error("REQUIRED_DIRECTORY_UNAVAILABLE") from exc
    if path.is_symlink() or _is_reparse(stat_result) or not path.is_dir():
        raise error("REQUIRED_DIRECTORY_UNSAFE")


def _safe_root(path: Path, repo_root: Path) -> Path:
    if not path.is_absolute():
        raise SuccessorPreflightFailure("CACHE_ROOT_NOT_ABSOLUTE")
    _assert_existing_chain(path)
    _safe_directory(path, error=SuccessorPreflightFailure)
    resolved = path.resolve(strict=True)
    repo = repo_root.resolve(strict=True)
    if resolved == repo or resolved.is_relative_to(repo):
        raise SuccessorPreflightFailure("CACHE_ROOT_INSIDE_REPOSITORY")
    return resolved


def _safe_child(root: Path, relative: str, *, error: type[Exception]) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise error("PAYLOAD_PATH_INVALID")
    candidate = root / relative
    resolved = candidate.resolve(strict=False)
    if resolved != root and not resolved.is_relative_to(root.resolve(strict=False)):
        raise error("PAYLOAD_PATH_ESCAPE")
    if candidate.is_symlink():
        raise error("PAYLOAD_PATH_SYMLINK")
    return candidate


def _read_json(path: Path, *, error: type[Exception]) -> tuple[bytes, dict[str, Any]]:
    _safe_file(path, error=error)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error("JSON_METADATA_INVALID") from exc
    if not isinstance(value, dict):
        raise error("JSON_METADATA_SCHEMA_INVALID")
    return raw, value


def _git(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SuccessorPreflightFailure("GIT_PROVENANCE_UNAVAILABLE") from exc
    return result.stdout.strip()


def _repository_url_is_authoritative(value: str) -> bool:
    return value.strip().rstrip("/") in {
        "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "https://github.com/ta1k1-arakawa/stock-analyzer",
    }


def _verify_successor_repo(repo_root: Path, implementation_sha: str) -> None:
    if not isinstance(implementation_sha, str) or not _SHA1.fullmatch(implementation_sha):
        raise SuccessorPreflightFailure("IMPLEMENTATION_SHA_INVALID")
    if not _repository_url_is_authoritative(_git(repo_root, "remote", "get-url", "origin")):
        raise SuccessorPreflightFailure("REPOSITORY_IDENTITY_MISMATCH")
    if _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD") != "v9-cross-sectional-close-auction-design":
        raise SuccessorPreflightFailure("BRANCH_MISMATCH")
    if _git(repo_root, "rev-parse", "HEAD") != implementation_sha:
        raise SuccessorPreflightFailure("IMPLEMENTATION_HEAD_MISMATCH")
    if _git(repo_root, "rev-parse", "refs/remotes/origin/v9-cross-sectional-close-auction-design") != implementation_sha:
        raise SuccessorPreflightFailure("REMOTE_HEAD_MISMATCH")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise SuccessorPreflightFailure("WORKTREE_DIRTY")
    expected = {
        DESIGN_FILE: DESIGN_BLOB,
        V10A_DESIGN_FILE: V10A_DESIGN_BLOB,
        V10B_SOURCE_FILE: V10B_SOURCE_BLOB,
        V9_CORE_FILE: V9_CORE_BLOB,
    }
    for path, blob in expected.items():
        if _git(repo_root, "rev-parse", f"HEAD:{path}") != blob:
            raise SuccessorPreflightFailure("SUCCESSOR_PROVENANCE_BLOB_MISMATCH")
    if _git(repo_root, "rev-parse", f"HEAD:{ADOPTION_RECORD_FILE}") != ADOPTION_RECORD_BLOB:
        raise SuccessorPreflightFailure("ADOPTION_RECORD_BLOB_MISMATCH")
    if _git(repo_root, "rev-parse", f"HEAD:{DESIGN_FREEZE_APPROVAL_FILE}") != DESIGN_FREEZE_APPROVAL_BLOB:
        raise SuccessorPreflightFailure("DESIGN_FREEZE_APPROVAL_BLOB_MISMATCH")
    adoption_path = repo_root / ADOPTION_RECORD_FILE
    raw, adoption = _read_json(adoption_path, error=SuccessorPreflightFailure)
    if _sha256(raw) != ADOPTION_RECORD_SHA256:
        raise SuccessorPreflightFailure("ADOPTION_RECORD_SHA256_MISMATCH")
    if adoption.get("schema_version") != "V10C_TRAINING_PROVENANCE_ADOPTION_RECORD_V1":
        raise SuccessorPreflightFailure("ADOPTION_RECORD_SCHEMA_INVALID")
    if (
        adoption.get("manifest_sha256") != TRAINING_MANIFEST_SHA256
        or adoption.get("promotion_status") != "ADOPTED_V10C_PROVENANCE_ONLY"
        or adoption.get("historical_v9_009_cache_recovered") is not False
    ):
        raise SuccessorPreflightFailure("ADOPTION_RECORD_BINDING_INVALID")


def _validate_attempt_receipt(receipt: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": ATTEMPT_SCHEMA,
        "study_identity": "V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR",
        "frozen_design_git_commit": "1260538c5bef899478806f74ae32d9e4be7b023b",
        "frozen_design_git_blob_sha": "bc57720b6e73cc8c4cf793258a98f78a635af783",
        "freeze_approval_git_blob_sha": "dde0418589088932559b289b3ed26a40e29b62b6",
        "implementation_sha": "b2172723df28b3edfce386c77ee79ce38a716925",
        "attempt_started": True,
        "network_acquisition_scope": "V10B_PUBLIC_YAHOO_ONLY",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise SuccessorPreflightFailure("ATTEMPT_RECEIPT_BINDING_INVALID")


def _validate_file_set(directory: Path, expected_names: set[str]) -> None:
    _safe_directory(directory, error=SuccessorPreflightFailure)
    try:
        entries = list(directory.iterdir())
    except OSError as exc:
        raise SuccessorPreflightFailure("CACHE_DIRECTORY_UNREADABLE") from exc
    if {entry.name for entry in entries} != expected_names:
        raise SuccessorPreflightFailure("CACHE_PAYLOAD_FILE_SET_INVALID")
    for entry in entries:
        _safe_file(entry, error=SuccessorPreflightFailure)


def _validate_training_metadata(root: Path, ticker_order: Sequence[str]) -> dict[str, Any]:
    receipt_raw, receipt = _read_json(root / ATTEMPT_RECEIPT_FILE, error=SuccessorPreflightFailure)
    if _sha256(receipt_raw) != ATTEMPT_RECEIPT_SHA256:
        raise SuccessorPreflightFailure("ATTEMPT_RECEIPT_SHA256_MISMATCH")
    _validate_attempt_receipt(receipt)
    manifest_raw, manifest = _read_json(root / MANIFEST_FILE, error=SuccessorPreflightFailure)
    manifest_sha = _sha256(manifest_raw)
    if manifest_sha == OLD_TRAINING_MANIFEST_SHA256:
        raise SuccessorPreflightFailure("OLD_TRAINING_MANIFEST_REJECTED")
    if manifest_sha != TRAINING_MANIFEST_SHA256:
        raise SuccessorPreflightFailure("TRAINING_MANIFEST_SHA256_MISMATCH")
    try:
        validated = v10c_validate_manifest_structure(manifest, ticker_order)
    except Exception as exc:
        raise SuccessorPreflightFailure("TRAINING_MANIFEST_METADATA_INVALID") from exc
    payloads = validated["payloads"]
    _validate_file_set(root / LOCKED_RAW, {f"{item['ticker']}.json" for item in payloads})
    return validated


def _validate_evaluation_metadata(root: Path, ticker_order: Sequence[str]) -> dict[str, Any]:
    raw, manifest = _read_json(root / MANIFEST_FILE, error=SuccessorPreflightFailure)
    if _sha256(raw) != EVALUATION_MANIFEST_SHA256:
        raise SuccessorPreflightFailure("EVALUATION_MANIFEST_SHA256_MISMATCH")
    if (
        manifest.get("schema_version") != EVALUATION_SCHEMA
        or manifest.get("complete") is not True
        or manifest.get("universe_mode") != "FIXED_V4_300"
        or manifest.get("ticker_count") != TICKER_COUNT
        or manifest.get("ticker_order") != list(ticker_order)
    ):
        raise SuccessorPreflightFailure("EVALUATION_MANIFEST_BINDING_INVALID")
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list) or len(payloads) != TICKER_COUNT:
        raise SuccessorPreflightFailure("EVALUATION_PAYLOAD_SET_INVALID")
    seen: list[str] = []
    for item in payloads:
        if not isinstance(item, dict) or not {"ticker", "relative_path", "sha256", "byte_count"}.issubset(item):
            raise SuccessorPreflightFailure("EVALUATION_PAYLOAD_SCHEMA_INVALID")
        ticker = item["ticker"]
        if ticker not in ticker_order or ticker in seen or item["relative_path"] != f"{RAW}/{ticker}.json":
            raise SuccessorPreflightFailure("EVALUATION_PAYLOAD_BINDING_INVALID")
        if not _SHA256.fullmatch(str(item["sha256"])) or type(item["byte_count"]) is not int or item["byte_count"] <= 0:
            raise SuccessorPreflightFailure("EVALUATION_PAYLOAD_METADATA_INVALID")
        seen.append(ticker)
    if seen != list(ticker_order):
        raise SuccessorPreflightFailure("EVALUATION_PAYLOAD_ORDER_INVALID")
    _validate_file_set(root / RAW, {f"{ticker}.json" for ticker in ticker_order})
    return manifest


def v10c_validate_manifest_structure(manifest: Mapping[str, Any], ticker_order: Sequence[str]) -> dict[str, Any]:
    """Call the reviewed V10C no-payload-byte manifest validator."""
    from src.v10c_locked_training_cache_provenance_adoption import validate_manifest_structure

    return validate_manifest_structure(manifest, ticker_order)


def phase_a_metadata_preflight(
    repository_root: Path,
    training_root: Path,
    evaluation_root: Path,
    universe_csv: Path,
    implementation_sha: str,
) -> SuccessorMetadata:
    """Validate successor identities without opening any price payload bytes."""
    _verify_successor_repo(repository_root, implementation_sha)
    training = _safe_root(training_root, repository_root)
    evaluation = _safe_root(evaluation_root, repository_root)
    if training == evaluation:
        raise SuccessorPreflightFailure("CACHE_ROOTS_NOT_DISTINCT")
    try:
        universe = load_fixed_universe(universe_csv)
    except Exception as exc:
        raise SuccessorPreflightFailure("UNIVERSE_IDENTITY_INVALID") from exc
    ticker_order = tuple(universe["ticker"].tolist())
    if len(ticker_order) != TICKER_COUNT:
        raise SuccessorPreflightFailure("UNIVERSE_CARDINALITY_INVALID")
    train_manifest = _validate_training_metadata(training, ticker_order)
    eval_manifest = _validate_evaluation_metadata(evaluation, ticker_order)
    return SuccessorMetadata(training, evaluation, Path(universe_csv), ticker_order, train_manifest, eval_manifest)


def _read_payload_bytes(path: Path) -> bytes:
    """The sole payload-byte read seam, used only after the authority boundary."""
    try:
        return path.read_bytes()
    except OSError as exc:
        raise T0DataIncompatible("CACHE_PAYLOAD_READ_INVALID") from exc


def _parse_payload_file(path: Path, expected_hash: str, expected_size: int) -> tuple[pd.DataFrame, dict[pd.Timestamp, float]]:
    body = _read_payload_bytes(path)
    if len(body) != expected_size or _sha256(body) != expected_hash:
        raise T0DataIncompatible("CACHE_PAYLOAD_HASH_MISMATCH")
    return _parse_chart_payload(body)


def _load_evaluation_payloads(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, tuple[pd.DataFrame, dict[pd.Timestamp, float]]]:
    parsed: dict[str, tuple[pd.DataFrame, dict[pd.Timestamp, float]]] = {}
    for item in manifest["payloads"]:
        path = _safe_child(root, item["relative_path"], error=T0DataIncompatible)
        _safe_file(path, error=T0DataIncompatible)
        parsed[item["ticker"]] = _parse_payload_file(path, item["sha256"], item["byte_count"])
    return parsed


def _combine_frames(
    ticker_order: Sequence[str],
    training: dict[str, tuple[pd.DataFrame, dict[pd.Timestamp, float]]],
    evaluation: dict[str, tuple[pd.DataFrame, dict[pd.Timestamp, float]]],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[pd.Timestamp, float]]]:
    frames: dict[str, pd.DataFrame] = {}
    actions: dict[str, dict[pd.Timestamp, float]] = {}
    for ticker in ticker_order:
        parts: list[pd.DataFrame] = []
        action_parts: list[dict[pd.Timestamp, float]] = []
        if ticker in training:
            train_frame, train_actions = training[ticker]
            parts.append(train_frame.loc[train_frame.index <= pd.Timestamp("2019-12-31")])
            action_parts.append(train_actions)
        if ticker not in evaluation:
            raise T0DataIncompatible("EVALUATION_PAYLOAD_MISSING")
        eval_frame, eval_actions = evaluation[ticker]
        start = EVAL_START_WITH_TRAINING if ticker in training else EVAL_START_WITHOUT_TRAINING
        parts.append(eval_frame.loc[eval_frame.index >= start])
        action_parts.append(eval_actions)
        if not parts:
            raise T0DataIncompatible("COMBINED_PRICE_EMPTY")
        combined = pd.concat(parts).sort_index()
        if combined.index.duplicated().any():
            raise T0DataIncompatible("DUPLICATE_COMBINED_PRICE_DATE")
        merged: dict[pd.Timestamp, float] = {}
        for source in action_parts:
            for day, ratio in source.items():
                if day in merged and merged[day] != ratio:
                    raise T0DataIncompatible("DUPLICATE_COMBINED_SPLIT_EVENT")
                merged[day] = ratio
        frames[ticker] = combined
        actions[ticker] = merged
    return frames, actions


def load_successor_cache_pair(
    metadata: SuccessorMetadata,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[pd.Timestamp, float]], dict[str, Any], pd.DataFrame]:
    """Full-load boundary for a later authorized execution.

    The V10B validator is intentionally called only here, after Phase A and
    the caller's separate point-of-use authority boundary.
    """
    try:
        v10b.validate_manifest(metadata.training_manifest, metadata.candidate_root, metadata.ticker_order)
    except v10b.ManifestValidationError as exc:
        raise T0DataIncompatible("TRAINING_PAYLOAD_HASH_CLOSURE_INVALID") from exc
    training: dict[str, tuple[pd.DataFrame, dict[pd.Timestamp, float]]] = {}
    for item in metadata.training_manifest["payloads"]:
        path = _safe_child(metadata.candidate_root, item["relative_path"], error=T0DataIncompatible)
        _safe_file(path, error=T0DataIncompatible)
        training[item["ticker"]] = _parse_payload_file(path, item["sha256"], item["byte_count"])
    evaluation = _load_evaluation_payloads(metadata.evaluation_root, metadata.evaluation_manifest)
    frames, actions = _combine_frames(metadata.ticker_order, training, evaluation)
    universe = load_fixed_universe(metadata.universe_csv)
    provenance = {
        "universe_mode": "FIXED_V4_300",
        "universe_csv_sha256": UNIVERSE_CSV_SHA256,
        "ticker_list_sha256": TICKER_LIST_SHA256,
        "universe_ticker_count": TICKER_COUNT,
        "training_cache_manifest_sha256": TRAINING_MANIFEST_SHA256,
        "evaluation_cache_manifest_sha256": EVALUATION_MANIFEST_SHA256,
        "training_payload_count": SUCCESS_COUNT,
        "evaluation_payload_count": TICKER_COUNT,
    }
    return frames, actions, provenance, universe


def run_successor_from_cache(
    repository_root: Path,
    training_root: Path,
    evaluation_root: Path,
    universe_csv: Path,
    calendar_dates: Sequence[object],
    implementation_sha: str,
) -> dict[str, Any]:
    metadata = phase_a_metadata_preflight(
        repository_root, training_root, evaluation_root, universe_csv, implementation_sha
    )
    frames, actions, provenance, universe = load_successor_cache_pair(metadata)
    dataset = build_dataset(frames, universe, calendar_dates, actions)
    scored = score_formal_dataset(dataset, calendar_dates)
    return make_safe_result(screen_top1(scored), implementation_sha, provenance)


__all__ = [
    "ADOPTION_RECORD_BLOB",
    "ADOPTION_RECORD_SHA256",
    "ATTEMPT_RECEIPT_SHA256",
    "DESIGN_BLOB",
    "DESIGN_COMMIT",
    "DESIGN_FILE",
    "DESIGN_FREEZE_APPROVAL_BLOB",
    "EVALUATION_MANIFEST_SHA256",
    "OLD_TRAINING_MANIFEST_SHA256",
    "SuccessorMetadata",
    "SuccessorPreflightFailure",
    "TRAINING_MANIFEST_SHA256",
    "T0DataIncompatible",
    "load_successor_cache_pair",
    "phase_a_metadata_preflight",
    "run_successor_from_cache",
    "v10c_validate_manifest_structure",
]
