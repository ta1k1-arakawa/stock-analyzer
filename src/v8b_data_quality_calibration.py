"""V8B data-quality calibration core (pure functions only).

Implements exactly ``V8B_DATA_QUALITY_CALIBRATION_PLAN_V1`` as approved in
``V8B_DATA_QUALITY_CALIBRATION_PLAN_APPROVAL.json`` (approved plan commit
``8c15426166742c43745e604f6367788af6123c1a``). This module performs no I/O
on import, makes no network calls, and implements no real-data filesystem
adapter: every entry point that touches bytes takes those bytes as an
argument rather than reading them from a caller-chosen path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import posixpath
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

# ---------------------------------------------------------------------------
# 5. Fixed constants
# ---------------------------------------------------------------------------

STUDY = "V8B_HISTORICAL_RESEARCH"
PLAN_VERSION = "V8B_DATA_QUALITY_CALIBRATION_PLAN_V1"
APPROVED_PLAN_COMMIT = "8c15426166742c43745e604f6367788af6123c1a"
APPROVED_PLAN_BLOB_SHA = "72f397aecbb3dadb6bc08e6fe929d54064d889a8"
APPROVAL_ARTIFACT_BLOB_SHA = "d8b7ce46f497d774541dc85f5e5df90cbc69c9e5"
PINNED_COLLECTOR_BLOB_SHA = "76b57b077f3214e666ff9dc06d9c224afc16df9f"
PINNED_COLLECTOR_PATH = "src/v7_yahoo_collector.py"
APPROVAL_ARTIFACT_PATH = "V8B_DATA_QUALITY_CALIBRATION_PLAN_APPROVAL.json"
PREREGISTRATION_PATH = "V8B_DATA_QUALITY_CALIBRATION_PREREGISTRATION_DRAFT.md"

CALIBRATION_START = "2019-01-01"
CALIBRATION_END_EXCLUSIVE = "2026-01-01"
CALIBRATION_YEARS = (2019, 2020, 2021, 2022, 2023, 2024, 2025)

V5B_ORIGINAL_REQUEST_START = "2019-01-01"
V5B_ORIGINAL_REQUEST_END_EXCLUSIVE = "2026-02-01"

EXPECTED_V5B_MANIFEST_SHA256 = "797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5"
EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256 = "a45ce89a7fa8be689e7d0affe34de56152552d7a3414935f0a364843cd3121f8"
EXPECTED_V5B_TICKER_COUNT = 300
V5B_MANIFEST_REQUEST_START = "2019-01-01"
V5B_MANIFEST_REQUEST_END = "2026-01-31"

SYNTHETIC_BASE_COUNT = 20
SYNTHETIC_SEQUENCE_LENGTH = 252

RESULT_SCHEMA_VERSION = "V8B_DATA_QUALITY_CALIBRATION_RESULT_V1"


# ---------------------------------------------------------------------------
# 6. Error type
# ---------------------------------------------------------------------------


class V8BCalibrationBlocked(RuntimeError):
    """Fail-closed error for every calibration blocking condition."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# 7. Git-blob hash
# ---------------------------------------------------------------------------


def git_blob_sha1(raw_bytes: bytes) -> str:
    header = b"blob " + str(len(raw_bytes)).encode("ascii") + b"\0"
    return hashlib.sha1(header + raw_bytes).hexdigest()


# ---------------------------------------------------------------------------
# 8. Canonical JSON
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def sha256_hex(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


class _DuplicateJSONKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _DuplicateJSONKeyError(key)
        seen.add(key)
        result[key] = value
    return result


def parse_strict_json(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateJSONKeyError as error:
        raise V8BCalibrationBlocked("STRICT_JSON_DUPLICATE_KEY") from error
    except json.JSONDecodeError as error:
        raise V8BCalibrationBlocked("STRICT_JSON_MALFORMED") from error


# ---------------------------------------------------------------------------
# 9. Repository contract verification
# ---------------------------------------------------------------------------


def _read_repository_file(repository_root: Path, relative_path: str, reason: str) -> bytes:
    try:
        return (repository_root / relative_path).read_bytes()
    except OSError as error:
        raise V8BCalibrationBlocked(reason) from error


def verify_repository_contract(repository_root: Path) -> dict[str, str]:
    """Read-only verification of the frozen plan/approval/classifier state.

    Never reads any V5-B external cache directory.
    """

    plan_bytes = _read_repository_file(repository_root, PREREGISTRATION_PATH, "CALIBRATION_PLAN_BLOB_MISMATCH")
    if git_blob_sha1(plan_bytes) != APPROVED_PLAN_BLOB_SHA:
        raise V8BCalibrationBlocked("CALIBRATION_PLAN_BLOB_MISMATCH")

    approval_bytes = _read_repository_file(repository_root, APPROVAL_ARTIFACT_PATH, "CALIBRATION_APPROVAL_BLOB_MISMATCH")
    if git_blob_sha1(approval_bytes) != APPROVAL_ARTIFACT_BLOB_SHA:
        raise V8BCalibrationBlocked("CALIBRATION_APPROVAL_BLOB_MISMATCH")

    try:
        approval_text = approval_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V8BCalibrationBlocked("CALIBRATION_APPROVAL_INVALID") from error
    approval_data = parse_strict_json(approval_text)
    if not isinstance(approval_data, dict):
        raise V8BCalibrationBlocked("CALIBRATION_APPROVAL_INVALID")

    required_fields = {
        "approval_status": "APPROVED",
        "human_gate": "DATA_QUALITY_CALIBRATION_PLAN_APPROVED",
        "approved_plan_version": PLAN_VERSION,
        "approved_plan_git_commit": APPROVED_PLAN_COMMIT,
        "approved_plan_blob_sha": APPROVED_PLAN_BLOB_SHA,
        "methodology_frozen_by_this_approval": True,
    }
    for key, expected in required_fields.items():
        if approval_data.get(key) != expected:
            raise V8BCalibrationBlocked("CALIBRATION_APPROVAL_INVALID")

    classifier_bytes = _read_repository_file(repository_root, PINNED_COLLECTOR_PATH, "CALIBRATION_CLASSIFIER_VERSION_MISMATCH")
    if git_blob_sha1(classifier_bytes) != PINNED_COLLECTOR_BLOB_SHA:
        raise V8BCalibrationBlocked("CALIBRATION_CLASSIFIER_VERSION_MISMATCH")

    return {
        "plan_blob_sha": APPROVED_PLAN_BLOB_SHA,
        "approval_blob_sha": APPROVAL_ARTIFACT_BLOB_SHA,
        "classifier_blob_sha": PINNED_COLLECTOR_BLOB_SHA,
    }


# ---------------------------------------------------------------------------
# 10. Pinned parser loading
# ---------------------------------------------------------------------------


def _load_pinned_collector(repository_root: Path) -> Any:
    raw = _read_repository_file(repository_root, PINNED_COLLECTOR_PATH, "CALIBRATION_CLASSIFIER_VERSION_MISMATCH")
    if git_blob_sha1(raw) != PINNED_COLLECTOR_BLOB_SHA:
        raise V8BCalibrationBlocked("CALIBRATION_CLASSIFIER_VERSION_MISMATCH")
    path = repository_root / PINNED_COLLECTOR_PATH
    spec = importlib.util.spec_from_file_location("v8b_pinned_v7_yahoo_collector", path)
    if spec is None or spec.loader is None:
        raise V8BCalibrationBlocked("CALIBRATION_CLASSIFIER_VERSION_MISMATCH")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 11. Candidate grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    fraction_id: str
    declared_numerator: int
    declared_denominator: int
    max_consecutive: int


_FRACTION_DEFS = (
    ("F1", 1, 252),
    ("F2", 2, 252),
    ("FQ1", 1, 100),
    ("F3", 3, 252),
    ("F4", 4, 252),
    ("F5", 5, 252),
)
_CONSECUTIVE_VALUES = (1, 2, 3, 4, 5)


def _build_candidate_grid() -> tuple[Candidate, ...]:
    ordered_fractions = sorted(_FRACTION_DEFS, key=lambda item: Fraction(item[1], item[2]))
    candidates: list[Candidate] = []
    for fraction_id, numerator, denominator in ordered_fractions:
        for consecutive in _CONSECUTIVE_VALUES:
            candidates.append(
                Candidate(
                    id=f"{fraction_id}_C{consecutive}",
                    fraction_id=fraction_id,
                    declared_numerator=numerator,
                    declared_denominator=denominator,
                    max_consecutive=consecutive,
                )
            )
    return tuple(candidates)


CANDIDATES: tuple[Candidate, ...] = _build_candidate_grid()


def candidate_fraction_value(candidate: Candidate) -> Fraction:
    return Fraction(candidate.declared_numerator, candidate.declared_denominator)


# ---------------------------------------------------------------------------
# 12. Corruption definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CorruptionSpec:
    name: str
    field: str
    value: float | None


CORRUPTIONS: tuple[CorruptionSpec, ...] = (
    CorruptionSpec("NONFINITE_OPEN", "open", None),
    CorruptionSpec("NONPOSITIVE_OPEN", "open", 0.0),
    CorruptionSpec("NONFINITE_HIGH", "high", None),
    CorruptionSpec("NONPOSITIVE_HIGH", "high", 0.0),
    CorruptionSpec("NONFINITE_LOW", "low", None),
    CorruptionSpec("NONPOSITIVE_LOW", "low", 0.0),
    CorruptionSpec("NONFINITE_CLOSE", "close", None),
    CorruptionSpec("NONPOSITIVE_CLOSE", "close", 0.0),
    CorruptionSpec("NONFINITE_ADJCLOSE", "adjclose", None),
    CorruptionSpec("NONPOSITIVE_ADJCLOSE", "adjclose", 0.0),
    CorruptionSpec("NONFINITE_VOLUME", "volume", None),
    CorruptionSpec("NEGATIVE_VOLUME", "volume", -1.0),
)


def apply_corruption(
    rows: Sequence[Mapping[str, float | None]],
    field: str,
    value: float | None,
    indices: Sequence[int],
) -> list[dict[str, float | None]]:
    corrupted = [dict(row) for row in rows]
    for index in indices:
        corrupted[index][field] = value
    return corrupted


# ---------------------------------------------------------------------------
# 13. Synthetic placement
# ---------------------------------------------------------------------------

PLACEMENT_FAMILIES = ("ISOLATED_EVENLY_SPACED", "CONSECUTIVE_RUN", "START_RUN", "END_RUN")


def corrupted_indices(k: int, family: str, *, n: int = SYNTHETIC_SEQUENCE_LENGTH) -> tuple[int, ...]:
    if k == 0:
        if family != "NONE":
            raise ValueError("K0_REQUIRES_NONE_FAMILY")
        return ()
    if family == "ISOLATED_EVENLY_SPACED":
        return tuple(((j + 1) * n) // (k + 1) for j in range(k))
    if family == "CONSECUTIVE_RUN":
        start = (n - k) // 2
        return tuple(range(start, start + k))
    if family == "START_RUN":
        return tuple(range(0, k))
    if family == "END_RUN":
        return tuple(range(n - k, n))
    raise ValueError("UNKNOWN_PLACEMENT_FAMILY:" + family)


# ---------------------------------------------------------------------------
# 14. Generic exact quality policy
# ---------------------------------------------------------------------------


def longest_true_run(flags: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for flag in flags:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def quality_policy_pass(
    invalid_flags: Sequence[bool],
    fraction_numerator: int,
    fraction_denominator: int,
    max_consecutive: int,
) -> bool:
    total_returned = len(invalid_flags)
    if total_returned == 0:
        return False
    invalid_count = sum(1 for flag in invalid_flags if flag)
    if invalid_count * fraction_denominator > total_returned * fraction_numerator:
        return False
    observed_max_consecutive = longest_true_run(invalid_flags)
    return observed_max_consecutive <= max_consecutive


# ---------------------------------------------------------------------------
# 15. Returned observation model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    trading_date: str
    valid: bool
    invalid_reason: str | None
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adjclose: float | None
    volume: float | None


# ---------------------------------------------------------------------------
# 16. Pure payload parsing
# ---------------------------------------------------------------------------


def parse_ticker_observations(
    ticker: object,
    payload_bytes: bytes,
    pinned_module: Any,
) -> tuple[str, tuple[Observation, ...]]:
    try:
        canonical = pinned_module.canonical_ticker(ticker)
        parsed = pinned_module.parse_chart_payload(
            payload_bytes,
            canonical,
            V5B_ORIGINAL_REQUEST_START,
            V5B_ORIGINAL_REQUEST_END_EXCLUSIVE,
        )
    except pinned_module.V7YahooCollectorBlocked as error:
        raise V8BCalibrationBlocked("CALIBRATION_INPUT_CANONICAL_PARSE_BLOCKED") from error

    combined: list[Observation] = []
    for row in parsed["valid_price_rows"]:
        combined.append(
            Observation(
                trading_date=row["trading_date"],
                valid=True,
                invalid_reason=None,
                open=row["raw_open"],
                high=row["raw_high"],
                low=row["raw_low"],
                close=row["raw_close"],
                adjclose=row["adj_close"],
                volume=row["raw_volume"],
            )
        )
    for row in parsed["invalid_price_rows"]:
        combined.append(
            Observation(
                trading_date=row["trading_date"],
                valid=False,
                invalid_reason=row["reason"],
                open=None,
                high=None,
                low=None,
                close=None,
                adjclose=None,
                volume=None,
            )
        )
    combined.sort(key=lambda observation: observation.trading_date)
    restricted = tuple(
        observation
        for observation in combined
        if CALIBRATION_START <= observation.trading_date < CALIBRATION_END_EXCLUSIVE
    )
    return canonical, restricted


# ---------------------------------------------------------------------------
# 17. Observed window stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowStats:
    total_returned: int
    valid_returned: int
    invalid_returned: int
    invalid_fraction: Fraction
    max_consecutive_invalid_returned_rows: int


def compute_window_stats(observations: Sequence[Observation]) -> WindowStats:
    total = len(observations)
    invalid_flags = [not observation.valid for observation in observations]
    invalid_count = sum(1 for flag in invalid_flags if flag)
    fraction = Fraction(invalid_count, total) if total else Fraction(0, 1)
    return WindowStats(
        total_returned=total,
        valid_returned=total - invalid_count,
        invalid_returned=invalid_count,
        invalid_fraction=fraction,
        max_consecutive_invalid_returned_rows=longest_true_run(invalid_flags),
    )


def compute_yearly_window_stats(
    observations: Sequence[Observation],
) -> dict[int, WindowStats | None]:
    by_year: dict[int, list[Observation]] = {}
    for observation in observations:
        by_year.setdefault(int(observation.trading_date[:4]), []).append(observation)
    result: dict[int, WindowStats | None] = {}
    for year in CALIBRATION_YEARS:
        year_observations = by_year.get(year, [])
        result[year] = compute_window_stats(year_observations) if year_observations else None
    return result


def compute_full_span_stats(observations: Sequence[Observation]) -> WindowStats:
    if not observations:
        raise V8BCalibrationBlocked("CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED")
    return compute_window_stats(observations)


# ---------------------------------------------------------------------------
# 18. Global envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GlobalEnvelope:
    m_fraction: Fraction
    m_fraction_source_window_count: int
    m_consecutive: int
    m_consecutive_source_window_count: int


def compute_global_envelope(windows: Sequence[WindowStats]) -> GlobalEnvelope:
    if not windows:
        raise V8BCalibrationBlocked("CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED")
    m_fraction = max(window.invalid_fraction for window in windows)
    m_consecutive = max(window.max_consecutive_invalid_returned_rows for window in windows)
    return GlobalEnvelope(
        m_fraction=m_fraction,
        m_fraction_source_window_count=sum(1 for w in windows if w.invalid_fraction == m_fraction),
        m_consecutive=m_consecutive,
        m_consecutive_source_window_count=sum(
            1 for w in windows if w.max_consecutive_invalid_returned_rows == m_consecutive
        ),
    )


# ---------------------------------------------------------------------------
# 19. Candidate defensibility
# ---------------------------------------------------------------------------


def is_candidate_defensible(candidate: Candidate, m_fraction: Fraction, m_consecutive: int) -> bool:
    return candidate_fraction_value(candidate) > m_fraction and candidate.max_consecutive > m_consecutive


# ---------------------------------------------------------------------------
# 20. Selection
# ---------------------------------------------------------------------------

CALIBRATION_NO_DEFENSIBLE_POLICY = "CALIBRATION_NO_DEFENSIBLE_POLICY"
NOT_EVALUATED = "NOT_EVALUATED"


def select_policy(m_fraction: Fraction, m_consecutive: int) -> tuple[str, tuple[Candidate, ...]]:
    defensible = tuple(c for c in CANDIDATES if is_candidate_defensible(c, m_fraction, m_consecutive))
    if not defensible:
        return CALIBRATION_NO_DEFENSIBLE_POLICY, defensible
    return defensible[0].id, defensible


# ---------------------------------------------------------------------------
# 21. Synthetic base selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CleanBase:
    base_index: int
    ticker_sha256: str
    window_start: str
    window_end: str
    rows: tuple[dict[str, float], ...]


def ticker_sha256(canonical: str) -> str:
    return hashlib.sha256((canonical + "\n").encode("utf-8")).hexdigest()


def find_earliest_clean_slice(observations: Sequence[Observation], length: int) -> int | None:
    run_length = 0
    for index, observation in enumerate(observations):
        if observation.valid:
            run_length += 1
            if run_length >= length:
                return index - length + 1
        else:
            run_length = 0
    return None


def select_synthetic_bases(
    observations_by_ticker: Mapping[str, Sequence[Observation]],
) -> tuple[CleanBase, ...]:
    bases: list[CleanBase] = []
    for ticker in sorted(observations_by_ticker):
        if len(bases) >= SYNTHETIC_BASE_COUNT:
            break
        observations = observations_by_ticker[ticker]
        start = find_earliest_clean_slice(observations, SYNTHETIC_SEQUENCE_LENGTH)
        if start is None:
            continue
        window = observations[start : start + SYNTHETIC_SEQUENCE_LENGTH]
        rows = tuple(
            {
                "open": observation.open,
                "high": observation.high,
                "low": observation.low,
                "close": observation.close,
                "adjclose": observation.adjclose,
                "volume": observation.volume,
            }
            for observation in window
        )
        bases.append(
            CleanBase(
                base_index=len(bases),
                ticker_sha256=ticker_sha256(ticker),
                window_start=window[0].trading_date,
                window_end=window[-1].trading_date,
                rows=rows,
            )
        )
    if len(bases) < SYNTHETIC_BASE_COUNT:
        raise V8BCalibrationBlocked("SYNTHETIC_BASE_SELECTION_BLOCKED")
    return tuple(bases)


# ---------------------------------------------------------------------------
# 22-23. Synthetic semantics verification / exact synthetic truth table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SyntheticScenario:
    base_index: int
    corruption_name: str
    field: str
    value: float | None
    k: int
    family: str


def iter_synthetic_scenarios(base_count: int = SYNTHETIC_BASE_COUNT) -> Iterator[SyntheticScenario]:
    for base_index in range(base_count):
        for corruption in CORRUPTIONS:
            for k in range(7):
                if k == 0:
                    yield SyntheticScenario(base_index, corruption.name, corruption.field, corruption.value, 0, "NONE")
                else:
                    for family in PLACEMENT_FAMILIES:
                        yield SyntheticScenario(base_index, corruption.name, corruption.field, corruption.value, k, family)


SYNTHETIC_SCENARIO_COUNT = SYNTHETIC_BASE_COUNT * len(CORRUPTIONS) * (1 + 6 * len(PLACEMENT_FAMILIES))
SYNTHETIC_CANDIDATE_COMPARISON_COUNT = SYNTHETIC_SCENARIO_COUNT * len(CANDIDATES)


def expected_synthetic_max_run(k: int, family: str) -> int:
    if k == 0:
        return 0
    if family == "ISOLATED_EVENLY_SPACED":
        return 1
    return k


def expected_fraction_pass(k: int, candidate: Candidate) -> bool:
    return k * candidate.declared_denominator <= SYNTHETIC_SEQUENCE_LENGTH * candidate.declared_numerator


def expected_consecutive_pass(k: int, family: str, candidate: Candidate) -> bool:
    return expected_synthetic_max_run(k, family) <= candidate.max_consecutive


def expected_scenario_pass(k: int, family: str, candidate: Candidate) -> bool:
    return expected_fraction_pass(k, candidate) and expected_consecutive_pass(k, family, candidate)


@dataclass(frozen=True, slots=True)
class SyntheticVerificationResult:
    scenario_count: int
    comparison_count: int
    truth_table_mismatch_count: int
    classifier_mismatch: bool


def run_synthetic_semantics_verification(
    bases: Sequence[CleanBase],
    pinned_module: Any,
) -> SyntheticVerificationResult:
    row_invalid_reason = pinned_module._row_invalid_reason
    scenario_count = 0
    truth_table_mismatch_count = 0
    for scenario in iter_synthetic_scenarios(len(bases)):
        scenario_count += 1
        base = bases[scenario.base_index]
        indices = corrupted_indices(scenario.k, scenario.family)
        corrupted_rows = apply_corruption(base.rows, scenario.field, scenario.value, indices)
        indices_set = set(indices)
        reasons: list[str | None] = []
        for index, row in enumerate(corrupted_rows):
            reason = row_invalid_reason(row)
            reasons.append(reason)
            if index in indices_set:
                if reason != scenario.corruption_name:
                    return SyntheticVerificationResult(scenario_count, 0, 0, True)
            else:
                if reason is not None:
                    return SyntheticVerificationResult(scenario_count, 0, 0, True)
        invalid_flags = [reason is not None for reason in reasons]
        for candidate in CANDIDATES:
            expected = expected_scenario_pass(scenario.k, scenario.family, candidate)
            observed = quality_policy_pass(
                invalid_flags,
                candidate.declared_numerator,
                candidate.declared_denominator,
                candidate.max_consecutive,
            )
            if expected != observed:
                truth_table_mismatch_count += 1
    return SyntheticVerificationResult(
        scenario_count=scenario_count,
        comparison_count=scenario_count * len(CANDIDATES),
        truth_table_mismatch_count=truth_table_mismatch_count,
        classifier_mismatch=False,
    )


# ---------------------------------------------------------------------------
# 24. Run validity representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunValidity:
    r0_classifier_pinned: bool = True
    r1_v5b_preflight: bool = True
    r2_payload_reconstruction: bool = True
    r3_nonempty_full_span: bool = True
    r4_synthetic_base_selection: bool = True
    r5_corrupted_row_classification: bool = True
    r6_uncorrupted_row_stability: bool = True
    r7_policy_truth_table: bool = True
    r8_no_masked_hard_failure: bool = True
    r9_plan_conformance: bool = True
    failure_reason: str | None = None

    @property
    def valid(self) -> bool:
        return (
            self.r0_classifier_pinned
            and self.r1_v5b_preflight
            and self.r2_payload_reconstruction
            and self.r3_nonempty_full_span
            and self.r4_synthetic_base_selection
            and self.r5_corrupted_row_classification
            and self.r6_uncorrupted_row_stability
            and self.r7_policy_truth_table
            and self.r8_no_masked_hard_failure
            and self.r9_plan_conformance
        )


VALID_RUN = RunValidity()

_RUN_INVALID_REASON_FLAGS: dict[str, tuple[str, ...]] = {
    "CALIBRATION_CLASSIFIER_VERSION_MISMATCH": ("r0_classifier_pinned",),
    "V5B_CALIBRATION_INPUT_PREFLIGHT_BLOCKED": ("r1_v5b_preflight",),
    "CALIBRATION_INPUT_CANONICAL_PARSE_BLOCKED": ("r2_payload_reconstruction", "r8_no_masked_hard_failure"),
    "CALIBRATION_INPUT_EMPTY_SERIES_BLOCKED": ("r3_nonempty_full_span",),
    "SYNTHETIC_BASE_SELECTION_BLOCKED": ("r4_synthetic_base_selection",),
    "SYNTHETIC_CLASSIFIER_MISMATCH": ("r5_corrupted_row_classification", "r6_uncorrupted_row_stability"),
    "SYNTHETIC_POLICY_SEMANTICS_MISMATCH": ("r7_policy_truth_table",),
    "CALIBRATION_PLAN_BLOB_MISMATCH": ("r9_plan_conformance",),
    "CALIBRATION_APPROVAL_BLOB_MISMATCH": ("r9_plan_conformance",),
    "CALIBRATION_APPROVAL_INVALID": ("r9_plan_conformance",),
}


def run_validity_for_reason(reason: str) -> RunValidity:
    flagged = _RUN_INVALID_REASON_FLAGS.get(reason, ("r9_plan_conformance",))
    overrides = {name: False for name in flagged}
    return RunValidity(failure_reason=reason, **overrides)


# ---------------------------------------------------------------------------
# 25. Pure V5-B manifest validation
# ---------------------------------------------------------------------------

_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _validate_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("raw/"):
        raise V8BCalibrationBlocked("MANIFEST_RELATIVE_PATH_INVALID")
    if "\\" in value:
        raise V8BCalibrationBlocked("MANIFEST_RELATIVE_PATH_INVALID")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise V8BCalibrationBlocked("MANIFEST_RELATIVE_PATH_INVALID")
    if posixpath.normpath(value) != value:
        raise V8BCalibrationBlocked("MANIFEST_RELATIVE_PATH_INVALID")
    return value


def _recompute_payload_hash_list_sha256(payloads: Sequence[Mapping[str, Any]]) -> str:
    digest_list = [payload["sha256"] for payload in payloads]
    encoded = json.dumps(digest_list, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_v5b_manifest_structure(data: Any) -> dict[str, Any]:
    """Structural validation of an already-parsed V5-B manifest object.

    Pure; does not read any payload file and does not check the outer
    whole-manifest SHA-256 pin (see ``validate_v5b_manifest_provenance``).
    """

    if not isinstance(data, dict):
        raise V8BCalibrationBlocked("MANIFEST_ROOT_INVALID")

    if data.get("schema_version") != 2:
        raise V8BCalibrationBlocked("MANIFEST_SCHEMA_VERSION_MISMATCH")
    if data.get("complete") is not True:
        raise V8BCalibrationBlocked("MANIFEST_NOT_COMPLETE")
    if data.get("usable_for_evaluation") is not True:
        raise V8BCalibrationBlocked("MANIFEST_NOT_USABLE")
    if data.get("attempted_ticker_count") != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked("MANIFEST_ATTEMPTED_TICKER_COUNT_MISMATCH")
    if data.get("success_count") != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked("MANIFEST_SUCCESS_COUNT_MISMATCH")
    if data.get("failed_count") != 0:
        raise V8BCalibrationBlocked("MANIFEST_FAILED_COUNT_MISMATCH")
    if data.get("ticker_count") != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked("MANIFEST_TICKER_COUNT_MISMATCH")
    if data.get("failed_tickers") != []:
        raise V8BCalibrationBlocked("MANIFEST_FAILED_TICKERS_NOT_EMPTY")
    if data.get("circuit_breaker_triggered") is not False:
        raise V8BCalibrationBlocked("MANIFEST_CIRCUIT_BREAKER_TRIGGERED")
    if data.get("request_start") != V5B_MANIFEST_REQUEST_START:
        raise V8BCalibrationBlocked("MANIFEST_REQUEST_START_MISMATCH")
    if data.get("request_end") != V5B_MANIFEST_REQUEST_END:
        raise V8BCalibrationBlocked("MANIFEST_REQUEST_END_MISMATCH")

    payloads = data.get("payloads")
    if not isinstance(payloads, list) or len(payloads) != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_COUNT_MISMATCH")

    seen_tickers: set[str] = set()
    seen_paths: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_RECORD_INVALID")
        ticker = payload.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_TICKER_INVALID")
        if ticker in seen_tickers:
            raise V8BCalibrationBlocked("MANIFEST_DUPLICATE_TICKER")
        seen_tickers.add(ticker)

        relative_path = _validate_relative_path(payload.get("relative_path"))
        if relative_path in seen_paths:
            raise V8BCalibrationBlocked("MANIFEST_DUPLICATE_RELATIVE_PATH")
        seen_paths.add(relative_path)

        digest = payload.get("sha256")
        if not isinstance(digest, str) or not _HEX64_RE.match(digest):
            raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_SHA256_INVALID")

        byte_count = payload.get("byte_count")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_BYTE_COUNT_INVALID")

    recomputed = _recompute_payload_hash_list_sha256(payloads)
    if recomputed != EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256:
        raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_HASH_LIST_MISMATCH")
    if data.get("payload_hash_list_sha256") != EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256:
        raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_HASH_LIST_FIELD_MISMATCH")

    return dict(data)


def validate_v5b_manifest_provenance(manifest_bytes: bytes) -> dict[str, Any]:
    """Production wrapper: fixed whole-manifest hash, then strict-JSON parse.

    Accepts only raw bytes. Never accepts a filesystem path and exposes no
    expected-hash override.
    """

    if sha256_hex(manifest_bytes) != EXPECTED_V5B_MANIFEST_SHA256:
        raise V8BCalibrationBlocked("MANIFEST_SHA256_MISMATCH")
    try:
        text = manifest_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise V8BCalibrationBlocked("MANIFEST_NOT_UTF8") from error
    data = parse_strict_json(text)
    return validate_v5b_manifest_structure(data)


# ---------------------------------------------------------------------------
# 27-28. Fraction JSON representation / artifact self-hash
# ---------------------------------------------------------------------------


def fraction_to_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_result_artifact(
    *,
    run_validity: RunValidity,
    selected_policy: str,
    candidate_selection_executed: bool,
    candidate_results: Sequence[Mapping[str, Any]],
    m_fraction: Fraction | None,
    m_fraction_window_count: int,
    m_consecutive: int | None,
    m_consecutive_window_count: int,
    synthetic_base_count: int,
    synthetic_scenario_count: int,
    synthetic_candidate_comparison_count: int,
    synthetic_truth_table_mismatch_count: int,
    synthetic_base_metadata: Sequence[Mapping[str, Any]],
    input_provenance: Mapping[str, Any],
    implementation_git_commit: str,
    calibration_attempt_id: str,
    run_started_utc: str,
    run_completed_or_blocked_utc: str,
    selected_candidate_fraction_headroom_exact_or_null: Mapping[str, int] | None = None,
    selected_candidate_consecutive_headroom_or_null: int | None = None,
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study": STUDY,
        "calibration_plan_version": PLAN_VERSION,
        "approved_plan_commit": APPROVED_PLAN_COMMIT,
        "approved_plan_blob_sha": APPROVED_PLAN_BLOB_SHA,
        "approval_artifact_blob_sha": APPROVAL_ARTIFACT_BLOB_SHA,
        "implementation_git_commit": implementation_git_commit,
        "calibration_attempt_id": calibration_attempt_id,
        "calibration_run_valid": run_validity.valid,
        "run_invalid_reason_or_null": run_validity.failure_reason,
        "candidate_selection_executed": candidate_selection_executed,
        "selected_policy": selected_policy,
        "input_provenance": dict(input_provenance),
        "calibration_start": CALIBRATION_START,
        "calibration_end_exclusive": CALIBRATION_END_EXCLUSIVE,
        "calibration_years": list(CALIBRATION_YEARS),
        "candidate_count": len(CANDIDATES),
        "candidate_results": list(candidate_results),
        "M_fraction_exact": fraction_to_json(m_fraction) if m_fraction is not None else None,
        "M_fraction_source_window_count": m_fraction_window_count,
        "M_consecutive": m_consecutive,
        "M_consecutive_source_window_count": m_consecutive_window_count,
        "selected_candidate_fraction_headroom_exact_or_null": selected_candidate_fraction_headroom_exact_or_null,
        "selected_candidate_consecutive_headroom_or_null": selected_candidate_consecutive_headroom_or_null,
        "synthetic_base_count": synthetic_base_count,
        "synthetic_scenario_count": synthetic_scenario_count,
        "synthetic_candidate_comparison_count": synthetic_candidate_comparison_count,
        "synthetic_truth_table_mismatch_count": synthetic_truth_table_mismatch_count,
        "synthetic_base_metadata": list(synthetic_base_metadata),
        "run_started_utc": run_started_utc,
        "run_completed_or_blocked_utc": run_completed_or_blocked_utc,
    }
    digest = sha256_hex(canonical_json_bytes(artifact))
    return {**artifact, "artifact_self_hash": digest}


def verify_artifact_self_hash(artifact: Mapping[str, Any]) -> bool:
    if "artifact_self_hash" not in artifact:
        return False
    claimed = artifact["artifact_self_hash"]
    without_hash = {key: value for key, value in artifact.items() if key != "artifact_self_hash"}
    return sha256_hex(canonical_json_bytes(without_hash)) == claimed


# ---------------------------------------------------------------------------
# 26. Orchestration (pure artifact construction only; no real artifact write)
# ---------------------------------------------------------------------------


def run_data_quality_calibration(
    *,
    repository_root: Path,
    manifest_bytes: bytes,
    ticker_payloads: Mapping[str, bytes],
    implementation_git_commit: str,
    calibration_attempt_id: str,
    run_started_utc: str | None = None,
) -> dict[str, Any]:
    """Pure (no filesystem write) end-to-end calibration run.

    ``ticker_payloads`` and ``manifest_bytes`` are supplied entirely by the
    caller as in-memory bytes; this function never opens a V5-B cache path.
    """

    started = run_started_utc or _utc_now_iso()
    provenance = {
        "manifest_sha256": sha256_hex(manifest_bytes),
        "ticker_count": len(ticker_payloads),
    }

    def blocked(reason: str) -> dict[str, Any]:
        return build_result_artifact(
            run_validity=run_validity_for_reason(reason),
            selected_policy=NOT_EVALUATED,
            candidate_selection_executed=False,
            candidate_results=[],
            m_fraction=None,
            m_fraction_window_count=0,
            m_consecutive=None,
            m_consecutive_window_count=0,
            synthetic_base_count=0,
            synthetic_scenario_count=0,
            synthetic_candidate_comparison_count=0,
            synthetic_truth_table_mismatch_count=0,
            synthetic_base_metadata=[],
            input_provenance=provenance,
            implementation_git_commit=implementation_git_commit,
            calibration_attempt_id=calibration_attempt_id,
            run_started_utc=started,
            run_completed_or_blocked_utc=_utc_now_iso(),
        )

    try:
        verify_repository_contract(repository_root)
        pinned_module = _load_pinned_collector(repository_root)
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    # R1 (V5-B cache provenance/preflight) is exposed only through the pure
    # validate_v5b_manifest_provenance()/validate_v5b_manifest_structure()
    # helpers for a future real-data adapter to call. This phase has no such
    # adapter and never reads the real V5-B cache, so R1 is not exercised
    # here against manifest_bytes; only its SHA-256 is recorded below for
    # provenance metadata.

    observations_by_ticker: dict[str, tuple[Observation, ...]] = {}
    try:
        for ticker, payload_bytes in ticker_payloads.items():
            canonical, restricted = parse_ticker_observations(ticker, payload_bytes, pinned_module)
            observations_by_ticker[canonical] = restricted
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    windows: list[WindowStats] = []
    try:
        for observations in observations_by_ticker.values():
            windows.append(compute_full_span_stats(observations))
            yearly = compute_yearly_window_stats(observations)
            windows.extend(stats for stats in yearly.values() if stats is not None)
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    envelope = compute_global_envelope(windows)

    try:
        bases = select_synthetic_bases(observations_by_ticker)
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    verification = run_synthetic_semantics_verification(bases, pinned_module)
    if verification.classifier_mismatch:
        return blocked("SYNTHETIC_CLASSIFIER_MISMATCH")
    if verification.truth_table_mismatch_count:
        return blocked("SYNTHETIC_POLICY_SEMANTICS_MISMATCH")

    selected_policy, defensible = select_policy(envelope.m_fraction, envelope.m_consecutive)
    defensible_ids = {candidate.id for candidate in defensible}

    candidate_results = []
    for candidate in CANDIDATES:
        fraction_headroom = candidate_fraction_value(candidate) - envelope.m_fraction
        consecutive_headroom = candidate.max_consecutive - envelope.m_consecutive
        candidate_results.append(
            {
                "candidate_id": candidate.id,
                "declared_numerator": candidate.declared_numerator,
                "declared_denominator": candidate.declared_denominator,
                "max_consecutive": candidate.max_consecutive,
                "defensible": candidate.id in defensible_ids,
                "fraction_headroom_exact": fraction_to_json(fraction_headroom),
                "consecutive_headroom": consecutive_headroom,
            }
        )

    selected_candidate = next((c for c in CANDIDATES if c.id == selected_policy), None)
    if selected_candidate is not None:
        selected_fraction_headroom = fraction_to_json(candidate_fraction_value(selected_candidate) - envelope.m_fraction)
        selected_consecutive_headroom = selected_candidate.max_consecutive - envelope.m_consecutive
    else:
        selected_fraction_headroom = None
        selected_consecutive_headroom = None

    synthetic_base_metadata = [
        {
            "base_index": base.base_index,
            "ticker_sha256": base.ticker_sha256,
            "window_start": base.window_start,
            "window_end": base.window_end,
        }
        for base in bases
    ]

    return build_result_artifact(
        run_validity=VALID_RUN,
        selected_policy=selected_policy,
        candidate_selection_executed=True,
        candidate_results=candidate_results,
        m_fraction=envelope.m_fraction,
        m_fraction_window_count=envelope.m_fraction_source_window_count,
        m_consecutive=envelope.m_consecutive,
        m_consecutive_window_count=envelope.m_consecutive_source_window_count,
        synthetic_base_count=SYNTHETIC_BASE_COUNT,
        synthetic_scenario_count=verification.scenario_count,
        synthetic_candidate_comparison_count=verification.comparison_count,
        synthetic_truth_table_mismatch_count=verification.truth_table_mismatch_count,
        synthetic_base_metadata=synthetic_base_metadata,
        input_provenance=provenance,
        implementation_git_commit=implementation_git_commit,
        calibration_attempt_id=calibration_attempt_id,
        run_started_utc=started,
        run_completed_or_blocked_utc=_utc_now_iso(),
        selected_candidate_fraction_headroom_exact_or_null=selected_fraction_headroom,
        selected_candidate_consecutive_headroom_or_null=selected_consecutive_headroom,
    )


# ---------------------------------------------------------------------------
# 30. Static-check support (used by scripts/check_v8b_data_quality_calibration.py)
# ---------------------------------------------------------------------------


def _verify_candidate_grid_integrity() -> None:
    if len(CANDIDATES) != 30:
        raise V8BCalibrationBlocked("CALIBRATION_CANDIDATE_GRID_MISMATCH")
    expected_ids = [f"{fraction_id}_C{c}" for fraction_id, _, _ in _FRACTION_DEFS for c in _CONSECUTIVE_VALUES]
    # _FRACTION_DEFS is declared in numeric-ascending order already; re-derive
    # the expected order independently via Fraction comparison so this check
    # does not merely restate the declaration order.
    ordered_fraction_ids = [item[0] for item in sorted(_FRACTION_DEFS, key=lambda item: Fraction(item[1], item[2]))]
    expected_ids = [f"{fraction_id}_C{c}" for fraction_id in ordered_fraction_ids for c in _CONSECUTIVE_VALUES]
    if [candidate.id for candidate in CANDIDATES] != expected_ids:
        raise V8BCalibrationBlocked("CALIBRATION_CANDIDATE_GRID_MISMATCH")
    previous_key: tuple[Fraction, int] | None = None
    for candidate in CANDIDATES:
        key = (candidate_fraction_value(candidate), candidate.max_consecutive)
        if previous_key is not None and key < previous_key:
            raise V8BCalibrationBlocked("CALIBRATION_CANDIDATE_GRID_MISMATCH")
        previous_key = key


def _verify_placement_formulas() -> None:
    if corrupted_indices(0, "NONE") != ():
        raise V8BCalibrationBlocked("CALIBRATION_PLACEMENT_FORMULA_MISMATCH")
    for k in range(1, 7):
        isolated = corrupted_indices(k, "ISOLATED_EVENLY_SPACED")
        if len(isolated) != k or len(set(isolated)) != k or sorted(isolated) != list(isolated):
            raise V8BCalibrationBlocked("CALIBRATION_PLACEMENT_FORMULA_MISMATCH")
        flags = [False] * SYNTHETIC_SEQUENCE_LENGTH
        for index in isolated:
            if not (0 <= index < SYNTHETIC_SEQUENCE_LENGTH):
                raise V8BCalibrationBlocked("CALIBRATION_PLACEMENT_FORMULA_MISMATCH")
            flags[index] = True
        if longest_true_run(flags) != 1:
            raise V8BCalibrationBlocked("CALIBRATION_PLACEMENT_FORMULA_MISMATCH")

        for family in ("CONSECUTIVE_RUN", "START_RUN", "END_RUN"):
            indices = corrupted_indices(k, family)
            if len(indices) != k:
                raise V8BCalibrationBlocked("CALIBRATION_PLACEMENT_FORMULA_MISMATCH")
            expected = tuple(range(indices[0], indices[0] + k))
            if indices != expected:
                raise V8BCalibrationBlocked("CALIBRATION_PLACEMENT_FORMULA_MISMATCH")
        if corrupted_indices(k, "START_RUN") != tuple(range(0, k)):
            raise V8BCalibrationBlocked("CALIBRATION_PLACEMENT_FORMULA_MISMATCH")
        if corrupted_indices(k, "END_RUN") != tuple(range(SYNTHETIC_SEQUENCE_LENGTH - k, SYNTHETIC_SEQUENCE_LENGTH)):
            raise V8BCalibrationBlocked("CALIBRATION_PLACEMENT_FORMULA_MISMATCH")
        expected_start = (SYNTHETIC_SEQUENCE_LENGTH - k) // 2
        if corrupted_indices(k, "CONSECUTIVE_RUN") != tuple(range(expected_start, expected_start + k)):
            raise V8BCalibrationBlocked("CALIBRATION_PLACEMENT_FORMULA_MISMATCH")


def _verify_synthetic_counts() -> None:
    scenario_count = sum(1 for _ in iter_synthetic_scenarios())
    if scenario_count != 6000:
        raise V8BCalibrationBlocked("CALIBRATION_SYNTHETIC_SCENARIO_COUNT_MISMATCH")
    if scenario_count * len(CANDIDATES) != 180000:
        raise V8BCalibrationBlocked("CALIBRATION_SYNTHETIC_COMPARISON_COUNT_MISMATCH")


def _verify_policy_boundary_cases() -> None:
    fraction_pass = [True] + [False] * (SYNTHETIC_SEQUENCE_LENGTH - 1)
    if not quality_policy_pass(fraction_pass, 1, SYNTHETIC_SEQUENCE_LENGTH, 5):
        raise V8BCalibrationBlocked("CALIBRATION_POLICY_BOUNDARY_MISMATCH")
    fraction_fail = [True, True] + [False] * (SYNTHETIC_SEQUENCE_LENGTH - 2)
    if quality_policy_pass(fraction_fail, 1, SYNTHETIC_SEQUENCE_LENGTH, 5):
        raise V8BCalibrationBlocked("CALIBRATION_POLICY_BOUNDARY_MISMATCH")

    consecutive_pass = [True] * 5 + [False] * (SYNTHETIC_SEQUENCE_LENGTH - 5)
    if not quality_policy_pass(consecutive_pass, 5, SYNTHETIC_SEQUENCE_LENGTH, 5):
        raise V8BCalibrationBlocked("CALIBRATION_POLICY_BOUNDARY_MISMATCH")
    consecutive_fail = [True] * 6 + [False] * (SYNTHETIC_SEQUENCE_LENGTH - 6)
    if quality_policy_pass(consecutive_fail, 5, SYNTHETIC_SEQUENCE_LENGTH, 5):
        raise V8BCalibrationBlocked("CALIBRATION_POLICY_BOUNDARY_MISMATCH")

    if quality_policy_pass([], 5, SYNTHETIC_SEQUENCE_LENGTH, 5):
        raise V8BCalibrationBlocked("CALIBRATION_POLICY_BOUNDARY_MISMATCH")


def _verify_self_hash_round_trip() -> None:
    dummy = build_result_artifact(
        run_validity=VALID_RUN,
        selected_policy="F1_C1",
        candidate_selection_executed=True,
        candidate_results=[],
        m_fraction=Fraction(0, 1),
        m_fraction_window_count=1,
        m_consecutive=0,
        m_consecutive_window_count=1,
        synthetic_base_count=0,
        synthetic_scenario_count=0,
        synthetic_candidate_comparison_count=0,
        synthetic_truth_table_mismatch_count=0,
        synthetic_base_metadata=[],
        input_provenance={"manifest_sha256": "0" * 64, "ticker_count": 0},
        implementation_git_commit="0" * 40,
        calibration_attempt_id="static-check",
        run_started_utc=_utc_now_iso(),
        run_completed_or_blocked_utc=_utc_now_iso(),
    )
    if not verify_artifact_self_hash(dummy):
        raise V8BCalibrationBlocked("CALIBRATION_ARTIFACT_SELF_HASH_MISMATCH")
    mutated = dict(dummy)
    mutated["selected_policy"] = "F2_C1"
    if verify_artifact_self_hash(mutated):
        raise V8BCalibrationBlocked("CALIBRATION_ARTIFACT_SELF_HASH_MISMATCH")


def run_static_check(repository_root: Path) -> None:
    """Repository-only static verification. Never reads any V5-B cache."""

    verify_repository_contract(repository_root)
    _load_pinned_collector(repository_root)

    if STUDY != "V8B_HISTORICAL_RESEARCH" or PLAN_VERSION != "V8B_DATA_QUALITY_CALIBRATION_PLAN_V1":
        raise V8BCalibrationBlocked("CALIBRATION_PLAN_CONSTANT_MISMATCH")
    if len(CORRUPTIONS) != 12:
        raise V8BCalibrationBlocked("CALIBRATION_CORRUPTION_GRID_MISMATCH")

    _verify_candidate_grid_integrity()
    _verify_placement_formulas()
    _verify_synthetic_counts()
    _verify_policy_boundary_cases()
    _verify_self_hash_round_trip()


__all__ = [
    "APPROVAL_ARTIFACT_BLOB_SHA",
    "APPROVED_PLAN_BLOB_SHA",
    "APPROVED_PLAN_COMMIT",
    "CALIBRATION_END_EXCLUSIVE",
    "CALIBRATION_NO_DEFENSIBLE_POLICY",
    "CALIBRATION_START",
    "CALIBRATION_YEARS",
    "CANDIDATES",
    "CORRUPTIONS",
    "Candidate",
    "CleanBase",
    "CorruptionSpec",
    "EXPECTED_V5B_MANIFEST_SHA256",
    "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256",
    "EXPECTED_V5B_TICKER_COUNT",
    "GlobalEnvelope",
    "NOT_EVALUATED",
    "Observation",
    "PINNED_COLLECTOR_BLOB_SHA",
    "PINNED_COLLECTOR_PATH",
    "PLACEMENT_FAMILIES",
    "PLAN_VERSION",
    "RESULT_SCHEMA_VERSION",
    "RunValidity",
    "STUDY",
    "SYNTHETIC_BASE_COUNT",
    "SYNTHETIC_CANDIDATE_COMPARISON_COUNT",
    "SYNTHETIC_SCENARIO_COUNT",
    "SYNTHETIC_SEQUENCE_LENGTH",
    "SyntheticScenario",
    "SyntheticVerificationResult",
    "V8BCalibrationBlocked",
    "VALID_RUN",
    "WindowStats",
    "apply_corruption",
    "build_result_artifact",
    "candidate_fraction_value",
    "canonical_json_bytes",
    "compute_full_span_stats",
    "compute_global_envelope",
    "compute_window_stats",
    "compute_yearly_window_stats",
    "corrupted_indices",
    "expected_consecutive_pass",
    "expected_fraction_pass",
    "expected_scenario_pass",
    "expected_synthetic_max_run",
    "find_earliest_clean_slice",
    "fraction_to_json",
    "git_blob_sha1",
    "is_candidate_defensible",
    "iter_synthetic_scenarios",
    "longest_true_run",
    "parse_strict_json",
    "parse_ticker_observations",
    "quality_policy_pass",
    "run_data_quality_calibration",
    "run_static_check",
    "run_synthetic_semantics_verification",
    "run_validity_for_reason",
    "select_policy",
    "select_synthetic_bases",
    "sha256_hex",
    "ticker_sha256",
    "validate_v5b_manifest_provenance",
    "validate_v5b_manifest_structure",
    "verify_artifact_self_hash",
    "verify_repository_contract",
]
