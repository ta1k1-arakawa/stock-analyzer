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
import json
import posixpath
import re
import types
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

SYNTHETIC_BASE_SELECTION_RULE_VERSION = "V8B_SYNTHETIC_BASE_SELECTION_V1"
SYNTHETIC_PLACEMENT_FORMULAS_VERSION = "V8B_SYNTHETIC_PLACEMENT_FORMULAS_V1"

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
    """Verify the pinned collector's exact bytes once, then execute those
    exact bytes. The file is never reopened after verification: the object
    compiled and exec'd is built directly from the ``raw`` bytes that were
    hashed, so a TOCTOU swap of the on-disk file cannot change what runs.
    """

    raw = _read_repository_file(repository_root, PINNED_COLLECTOR_PATH, "CALIBRATION_CLASSIFIER_VERSION_MISMATCH")
    if git_blob_sha1(raw) != PINNED_COLLECTOR_BLOB_SHA:
        raise V8BCalibrationBlocked("CALIBRATION_CLASSIFIER_VERSION_MISMATCH")
    module = types.ModuleType("v8b_pinned_v7_yahoo_collector")
    module.__file__ = str(repository_root / PINNED_COLLECTOR_PATH)
    try:
        code_object = compile(raw, module.__file__, "exec")
        exec(code_object, module.__dict__)
    except V8BCalibrationBlocked:
        raise
    except Exception as error:
        raise V8BCalibrationBlocked("CALIBRATION_CLASSIFIER_VERSION_MISMATCH") from error
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


def _window_passes_candidate(window: WindowStats, candidate: Candidate) -> bool:
    """Exactly equivalent to quality_policy_pass() on the window's original
    per-row flags: it depends only on total/invalid-count/max-run, which
    WindowStats already captures losslessly for this purpose."""

    if window.total_returned == 0:
        return False
    if window.invalid_returned * candidate.declared_denominator > window.total_returned * candidate.declared_numerator:
        return False
    return window.max_consecutive_invalid_returned_rows <= candidate.max_consecutive


def _failed_criterion_ids(candidate: Candidate, m_fraction: Fraction, m_consecutive: int) -> list[str]:
    failed: list[str] = []
    if not (candidate_fraction_value(candidate) > m_fraction):
        failed.append("D1")
    if not (candidate.max_consecutive > m_consecutive):
        failed.append("D2")
    return failed


# ---------------------------------------------------------------------------
# 20. Selection
# ---------------------------------------------------------------------------

CALIBRATION_NO_DEFENSIBLE_POLICY = "CALIBRATION_NO_DEFENSIBLE_POLICY"
NOT_EVALUATED = "NOT_EVALUATED"


def select_policy(
    run_validity: RunValidity,
    m_fraction: Fraction | None,
    m_consecutive: int | None,
) -> tuple[str, tuple[Candidate, ...]]:
    if not run_validity.valid or m_fraction is None or m_consecutive is None:
        raise V8BCalibrationBlocked("CALIBRATION_SELECTION_REQUIRES_VALID_RUN")
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
    pinned_module: Any,
) -> tuple[CleanBase, ...]:
    canonical_map: dict[str, Sequence[Observation]] = {}
    for raw_ticker, observations in observations_by_ticker.items():
        try:
            canonical = pinned_module.canonical_ticker(raw_ticker)
        except pinned_module.V7YahooCollectorBlocked as error:
            raise V8BCalibrationBlocked("CALIBRATION_INPUT_CANONICAL_TICKER_COLLISION") from error
        if canonical in canonical_map:
            raise V8BCalibrationBlocked("CALIBRATION_INPUT_CANONICAL_TICKER_COLLISION")
        canonical_map[canonical] = observations

    bases: list[CleanBase] = []
    for ticker in sorted(canonical_map):
        if len(bases) >= SYNTHETIC_BASE_COUNT:
            break
        observations = canonical_map[ticker]
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

    def __post_init__(self) -> None:
        if self.valid:
            if self.failure_reason is not None:
                raise V8BCalibrationBlocked("CALIBRATION_RUN_VALIDITY_STATE_INVALID")
        else:
            if not self.failure_reason:
                raise V8BCalibrationBlocked("CALIBRATION_RUN_VALIDITY_STATE_INVALID")


VALID_RUN = RunValidity()

_RUN_INVALID_REASON_FLAGS: dict[str, tuple[str, ...]] = {
    "CALIBRATION_CLASSIFIER_VERSION_MISMATCH": ("r0_classifier_pinned",),
    "V5B_CALIBRATION_INPUT_PREFLIGHT_BLOCKED": ("r1_v5b_preflight",),
    "CALIBRATION_INPUT_PAYLOAD_SET_MISMATCH": ("r1_v5b_preflight",),
    "CALIBRATION_INPUT_PAYLOAD_PATH_MISMATCH": ("r1_v5b_preflight",),
    "CALIBRATION_INPUT_PAYLOAD_BYTE_COUNT_MISMATCH": ("r1_v5b_preflight",),
    "CALIBRATION_INPUT_PAYLOAD_SHA256_MISMATCH": ("r1_v5b_preflight",),
    "CALIBRATION_INPUT_CANONICAL_TICKER_COLLISION": ("r1_v5b_preflight",),
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
    if reason in _RUN_INVALID_REASON_FLAGS:
        flagged = _RUN_INVALID_REASON_FLAGS[reason]
    elif reason.startswith("MANIFEST_"):
        flagged = ("r1_v5b_preflight",)
    else:
        flagged = ("r9_plan_conformance",)
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

    if type(data.get("schema_version")) is not int or data.get("schema_version") != 2:
        raise V8BCalibrationBlocked("MANIFEST_SCHEMA_VERSION_MISMATCH")
    if data.get("complete") is not True:
        raise V8BCalibrationBlocked("MANIFEST_NOT_COMPLETE")
    if data.get("usable_for_evaluation") is not True:
        raise V8BCalibrationBlocked("MANIFEST_NOT_USABLE")
    if type(data.get("attempted_ticker_count")) is not int or data.get("attempted_ticker_count") != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked("MANIFEST_ATTEMPTED_TICKER_COUNT_MISMATCH")
    if type(data.get("success_count")) is not int or data.get("success_count") != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked("MANIFEST_SUCCESS_COUNT_MISMATCH")
    if type(data.get("failed_count")) is not int or data.get("failed_count") != 0:
        raise V8BCalibrationBlocked("MANIFEST_FAILED_COUNT_MISMATCH")
    if type(data.get("ticker_count")) is not int or data.get("ticker_count") != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked("MANIFEST_TICKER_COUNT_MISMATCH")
    if data.get("failed_tickers") != []:
        raise V8BCalibrationBlocked("MANIFEST_FAILED_TICKERS_NOT_EMPTY")
    if data.get("circuit_breaker_triggered") is not False:
        raise V8BCalibrationBlocked("MANIFEST_CIRCUIT_BREAKER_TRIGGERED")
    if data.get("request_start") != V5B_MANIFEST_REQUEST_START:
        raise V8BCalibrationBlocked("MANIFEST_REQUEST_START_MISMATCH")
    if data.get("request_end") != V5B_MANIFEST_REQUEST_END:
        raise V8BCalibrationBlocked("MANIFEST_REQUEST_END_MISMATCH")

    for optional_counter in ("retry_count", "http_429_count", "http_5xx_count"):
        if optional_counter in data:
            value = data[optional_counter]
            if type(value) is not int or value < 0:
                raise V8BCalibrationBlocked(f"MANIFEST_{optional_counter.upper()}_INVALID")

    payloads = data.get("payloads")
    if not isinstance(payloads, list) or len(payloads) != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_COUNT_MISMATCH")

    seen_tickers: set[str] = set()
    seen_paths: set[str] = set()
    normalized_payloads: list[dict[str, Any]] = []
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
        normalized_digest = digest.lower()

        byte_count = payload.get("byte_count")
        if type(byte_count) is not int or byte_count < 0:
            raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_BYTE_COUNT_INVALID")

        normalized_payloads.append({**payload, "sha256": normalized_digest})

    # Normalize every valid payload SHA to lowercase BEFORE recomputation.
    recomputed = _recompute_payload_hash_list_sha256(normalized_payloads)
    if recomputed != EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256:
        raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_HASH_LIST_MISMATCH")

    stored_hash_list = data.get("payload_hash_list_sha256")
    if not isinstance(stored_hash_list, str) or not _HEX64_RE.match(stored_hash_list):
        raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_HASH_LIST_FIELD_MISMATCH")
    normalized_stored_hash_list = stored_hash_list.lower()
    if normalized_stored_hash_list != EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256:
        raise V8BCalibrationBlocked("MANIFEST_PAYLOAD_HASH_LIST_FIELD_MISMATCH")

    result = dict(data)
    result["payloads"] = normalized_payloads
    result["payload_hash_list_sha256"] = normalized_stored_hash_list
    return result


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
# Payload binding: every supplied in-memory payload must bind exactly to a
# record in the R1-validated manifest before any parsing happens.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InMemoryPayload:
    relative_path: str
    payload_bytes: bytes


def bind_payloads_to_manifest(
    manifest: Mapping[str, Any],
    ticker_payloads: Mapping[str, InMemoryPayload],
    pinned_module: Any,
) -> dict[str, bytes]:
    """Bind caller-supplied in-memory payloads to a validated manifest.

    Both ``manifest`` (already R1-validated) and ``ticker_payloads`` are
    caller-supplied in-memory data; no filesystem access happens here. Every
    manifest payload record must match exactly one supplied payload by
    canonical ticker, with exact relative_path/byte_count/SHA-256 agreement.
    """

    manifest_payloads = manifest.get("payloads")
    if not isinstance(manifest_payloads, list):
        raise V8BCalibrationBlocked("CALIBRATION_INPUT_PAYLOAD_SET_MISMATCH")

    manifest_by_canonical: dict[str, Mapping[str, Any]] = {}
    for record in manifest_payloads:
        try:
            canonical = pinned_module.canonical_ticker(record.get("ticker"))
        except pinned_module.V7YahooCollectorBlocked as error:
            raise V8BCalibrationBlocked("CALIBRATION_INPUT_PAYLOAD_SET_MISMATCH") from error
        if canonical in manifest_by_canonical:
            raise V8BCalibrationBlocked("CALIBRATION_INPUT_CANONICAL_TICKER_COLLISION")
        manifest_by_canonical[canonical] = record

    supplied_by_canonical: dict[str, InMemoryPayload] = {}
    for raw_ticker, payload in ticker_payloads.items():
        try:
            canonical = pinned_module.canonical_ticker(raw_ticker)
        except pinned_module.V7YahooCollectorBlocked as error:
            raise V8BCalibrationBlocked("CALIBRATION_INPUT_PAYLOAD_SET_MISMATCH") from error
        if canonical in supplied_by_canonical:
            raise V8BCalibrationBlocked("CALIBRATION_INPUT_CANONICAL_TICKER_COLLISION")
        supplied_by_canonical[canonical] = payload

    if set(manifest_by_canonical) != set(supplied_by_canonical):
        raise V8BCalibrationBlocked("CALIBRATION_INPUT_PAYLOAD_SET_MISMATCH")

    bound: dict[str, bytes] = {}
    for canonical, record in manifest_by_canonical.items():
        payload = supplied_by_canonical[canonical]
        if payload.relative_path != record.get("relative_path"):
            raise V8BCalibrationBlocked("CALIBRATION_INPUT_PAYLOAD_PATH_MISMATCH")
        if len(payload.payload_bytes) != record.get("byte_count"):
            raise V8BCalibrationBlocked("CALIBRATION_INPUT_PAYLOAD_BYTE_COUNT_MISMATCH")
        if sha256_hex(payload.payload_bytes) != record.get("sha256"):
            raise V8BCalibrationBlocked("CALIBRATION_INPUT_PAYLOAD_SHA256_MISMATCH")
        bound[canonical] = payload.payload_bytes
    return bound


# ---------------------------------------------------------------------------
# 27-28. Fraction JSON representation / artifact self-hash
# ---------------------------------------------------------------------------


def fraction_to_json(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_LOWER_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_utc_z(value: Any, reason: str) -> datetime:
    if not isinstance(value, str) or not _TIMESTAMP_RE.match(value):
        raise V8BCalibrationBlocked(reason)
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise V8BCalibrationBlocked(reason) from error


def _validate_provenance_fields(
    implementation_git_commit: str,
    calibration_attempt_id: str,
    run_started_utc: str,
    run_completed_or_blocked_utc: str,
) -> None:
    if not isinstance(implementation_git_commit, str) or not _COMMIT_RE.match(implementation_git_commit):
        raise V8BCalibrationBlocked("CALIBRATION_PROVENANCE_COMMIT_INVALID")
    if (
        not isinstance(calibration_attempt_id, str)
        or not calibration_attempt_id
        or len(calibration_attempt_id) > 128
        or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in calibration_attempt_id)
    ):
        raise V8BCalibrationBlocked("CALIBRATION_PROVENANCE_ATTEMPT_ID_INVALID")
    started = _parse_utc_z(run_started_utc, "CALIBRATION_PROVENANCE_TIMESTAMP_INVALID")
    completed = _parse_utc_z(run_completed_or_blocked_utc, "CALIBRATION_PROVENANCE_TIMESTAMP_INVALID")
    if completed < started:
        raise V8BCalibrationBlocked("CALIBRATION_PROVENANCE_TIMESTAMP_INVALID")


def _compute_candidate_results(
    yearly_windows: Sequence[WindowStats],
    full_span_windows: Sequence[WindowStats],
    m_fraction: Fraction,
    m_consecutive: int,
) -> list[dict[str, Any]]:
    """The single source of truth for candidate-result rows: every field is
    derived purely from CANDIDATES + the windows + the envelope, so this is
    called both to construct a VALID artifact's rows and to independently
    recompute the expected rows when validating one before hashing."""

    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        fraction_headroom = candidate_fraction_value(candidate) - m_fraction
        consecutive_headroom = candidate.max_consecutive - m_consecutive
        year_pass_count = sum(1 for window in yearly_windows if _window_passes_candidate(window, candidate))
        span_pass_count = sum(1 for window in full_span_windows if _window_passes_candidate(window, candidate))
        rows.append(
            {
                "candidate_id": candidate.id,
                "exact_fraction_rational": fraction_to_json(candidate_fraction_value(candidate)),
                "declared_fraction": {
                    "declared_numerator": candidate.declared_numerator,
                    "declared_denominator": candidate.declared_denominator,
                },
                "max_consecutive": candidate.max_consecutive,
                "observed_ticker_year_pass_count_over_denominator": {
                    "pass_count": year_pass_count,
                    "denominator": len(yearly_windows),
                },
                "observed_full_ticker_pass_count_over_denominator": {
                    "pass_count": span_pass_count,
                    "denominator": len(full_span_windows),
                },
                "DEFENSIBLE": is_candidate_defensible(candidate, m_fraction, m_consecutive),
                "failed_criterion_ids": _failed_criterion_ids(candidate, m_fraction, m_consecutive),
                "fraction_headroom_exact": fraction_to_json(fraction_headroom),
                "consecutive_headroom": consecutive_headroom,
            }
        )
    return rows


def _compute_selected_headrooms(
    selected_policy: str, m_fraction: Fraction, m_consecutive: int
) -> tuple[dict[str, int] | None, int | None]:
    if selected_policy == CALIBRATION_NO_DEFENSIBLE_POLICY:
        return None, None
    selected_candidate = next((c for c in CANDIDATES if c.id == selected_policy), None)
    if selected_candidate is None:
        return None, None
    fraction_headroom = fraction_to_json(candidate_fraction_value(selected_candidate) - m_fraction)
    consecutive_headroom = selected_candidate.max_consecutive - m_consecutive
    return fraction_headroom, consecutive_headroom


# ---------------------------------------------------------------------------
# Independent reference computation for semantic validation.
#
# These functions deliberately reimplement the candidate-row and selection
# arithmetic from scratch, using direct integer/Fraction comparisons, and
# never call _compute_candidate_results, _window_passes_candidate,
# is_candidate_defensible, or _failed_criterion_ids. This is what makes
# semantic validation a genuine cross-check rather than a circular
# construction-equals-construction comparison: a bug in the construction
# helpers would not also be present here, and vice versa.
# ---------------------------------------------------------------------------


def _reference_window_pass(window: WindowStats, numerator: int, denominator: int, max_consecutive: int) -> bool:
    if window.total_returned == 0:
        return False
    if window.invalid_returned * denominator > window.total_returned * numerator:
        return False
    return window.max_consecutive_invalid_returned_rows <= max_consecutive


def _reference_candidate_row(
    candidate: Candidate,
    yearly_windows: Sequence[WindowStats],
    full_span_windows: Sequence[WindowStats],
    m_fraction: Fraction,
    m_consecutive: int,
) -> dict[str, Any]:
    numerator = candidate.declared_numerator
    denominator = candidate.declared_denominator
    max_consecutive = candidate.max_consecutive
    exact_fraction = Fraction(numerator, denominator)

    year_pass_count = sum(
        1 for window in yearly_windows if _reference_window_pass(window, numerator, denominator, max_consecutive)
    )
    span_pass_count = sum(
        1 for window in full_span_windows if _reference_window_pass(window, numerator, denominator, max_consecutive)
    )

    fraction_defensible = exact_fraction > m_fraction
    consecutive_defensible = max_consecutive > m_consecutive
    failed_criterion_ids: list[str] = []
    if not fraction_defensible:
        failed_criterion_ids.append("D1")
    if not consecutive_defensible:
        failed_criterion_ids.append("D2")

    fraction_headroom = exact_fraction - m_fraction
    consecutive_headroom = max_consecutive - m_consecutive

    return {
        "candidate_id": candidate.id,
        "exact_fraction_rational": fraction_to_json(exact_fraction),
        "declared_fraction": {"declared_numerator": numerator, "declared_denominator": denominator},
        "max_consecutive": max_consecutive,
        "observed_ticker_year_pass_count_over_denominator": {
            "pass_count": year_pass_count,
            "denominator": len(yearly_windows),
        },
        "observed_full_ticker_pass_count_over_denominator": {
            "pass_count": span_pass_count,
            "denominator": len(full_span_windows),
        },
        "DEFENSIBLE": fraction_defensible and consecutive_defensible,
        "failed_criterion_ids": failed_criterion_ids,
        "fraction_headroom_exact": fraction_to_json(fraction_headroom),
        "consecutive_headroom": consecutive_headroom,
    }


def _verify_candidate_rows_independently(
    candidate_results: Sequence[Mapping[str, Any]],
    yearly_windows: Sequence[WindowStats],
    full_span_windows: Sequence[WindowStats],
    m_fraction: Fraction,
    m_consecutive: int,
) -> None:
    reason = "CALIBRATION_RESULT_STATE_INVALID"
    if len(candidate_results) != len(CANDIDATES):
        raise V8BCalibrationBlocked(reason)
    seen_ids: set[str] = set()
    for candidate, row in zip(CANDIDATES, candidate_results):
        if not isinstance(row, Mapping) or row.get("candidate_id") != candidate.id:
            raise V8BCalibrationBlocked(reason)
        if candidate.id in seen_ids:
            raise V8BCalibrationBlocked(reason)
        seen_ids.add(candidate.id)
        expected_row = _reference_candidate_row(candidate, yearly_windows, full_span_windows, m_fraction, m_consecutive)
        if dict(row) != expected_row:
            raise V8BCalibrationBlocked(reason)
    if seen_ids != {candidate.id for candidate in CANDIDATES}:
        raise V8BCalibrationBlocked(reason)


def _reference_select_policy(m_fraction: Fraction, m_consecutive: int) -> str:
    for candidate in CANDIDATES:
        exact_fraction = Fraction(candidate.declared_numerator, candidate.declared_denominator)
        if exact_fraction > m_fraction and candidate.max_consecutive > m_consecutive:
            return candidate.id
    return CALIBRATION_NO_DEFENSIBLE_POLICY


def _reference_synthetic_base_metadata(bases: Sequence[CleanBase]) -> list[dict[str, Any]]:
    return [
        {
            "base_index": base.base_index,
            "ticker_sha256": base.ticker_sha256,
            "window_start": base.window_start,
            "window_end": base.window_end,
        }
        for base in bases
    ]


def _verify_synthetic_base_metadata_against_bases(
    synthetic_base_metadata: Sequence[Mapping[str, Any]],
    bases: Sequence[CleanBase],
) -> None:
    reason = "CALIBRATION_RESULT_STATE_INVALID"
    if len(bases) != SYNTHETIC_BASE_COUNT:
        raise V8BCalibrationBlocked(reason)
    _validate_synthetic_base_metadata(synthetic_base_metadata)
    expected = _reference_synthetic_base_metadata(bases)
    if list(synthetic_base_metadata) != expected:
        raise V8BCalibrationBlocked(reason)


def _validate_synthetic_base_metadata(rows: Sequence[Mapping[str, Any]]) -> None:
    reason = "CALIBRATION_RESULT_STATE_INVALID"
    if len(rows) != SYNTHETIC_BASE_COUNT:
        raise V8BCalibrationBlocked(reason)
    seen_indices: set[int] = set()
    for row in rows:
        if set(row) != {"base_index", "ticker_sha256", "window_start", "window_end"}:
            raise V8BCalibrationBlocked(reason)
        base_index = row.get("base_index")
        if type(base_index) is not int or not (0 <= base_index < SYNTHETIC_BASE_COUNT):
            raise V8BCalibrationBlocked(reason)
        if base_index in seen_indices:
            raise V8BCalibrationBlocked(reason)
        seen_indices.add(base_index)

        ticker_hash = row.get("ticker_sha256")
        if not isinstance(ticker_hash, str) or not _LOWER_HEX64_RE.match(ticker_hash):
            raise V8BCalibrationBlocked(reason)

        start = row.get("window_start")
        end = row.get("window_end")
        if not isinstance(start, str) or not _ISO_DATE_RE.match(start):
            raise V8BCalibrationBlocked(reason)
        if not isinstance(end, str) or not _ISO_DATE_RE.match(end):
            raise V8BCalibrationBlocked(reason)
        try:
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")
        except ValueError as error:
            raise V8BCalibrationBlocked(reason) from error
        if not (CALIBRATION_START <= start < CALIBRATION_END_EXCLUSIVE):
            raise V8BCalibrationBlocked(reason)
        if not (CALIBRATION_START <= end < CALIBRATION_END_EXCLUSIVE):
            raise V8BCalibrationBlocked(reason)
        if start > end:
            raise V8BCalibrationBlocked(reason)
    if seen_indices != set(range(SYNTHETIC_BASE_COUNT)):
        raise V8BCalibrationBlocked(reason)


def _expected_error_counts_from_manifest(manifest: Mapping[str, Any]) -> dict[str, int]:
    return {
        "failed_count": manifest.get("failed_count", 0),
        "retry_count": manifest.get("retry_count", 0),
        "http_429_count": manifest.get("http_429_count", 0),
        "http_5xx_count": manifest.get("http_5xx_count", 0),
    }


def _validate_result_state(
    *,
    run_validity: RunValidity,
    selected_policy: str,
    candidate_selection_executed: bool,
    candidate_results: Sequence[Mapping[str, Any]],
    yearly_windows: Sequence[WindowStats],
    full_span_windows: Sequence[WindowStats],
    synthetic_bases: Sequence[CleanBase],
    manifest_bytes: bytes | None,
    m_fraction: Fraction | None,
    m_fraction_window_count: int,
    m_consecutive: int | None,
    m_consecutive_window_count: int,
    synthetic_base_count: int,
    synthetic_scenario_count: int,
    synthetic_candidate_comparison_count: int,
    synthetic_truth_table_mismatch_count: int,
    synthetic_base_metadata: Sequence[Mapping[str, Any]],
    input_provenance_hashes: Mapping[str, Any],
    error_counts: Mapping[str, Any],
    selected_candidate_fraction_headroom_exact_or_null: Mapping[str, int] | None,
    selected_candidate_consecutive_headroom_or_null: int | None,
) -> None:
    reason = "CALIBRATION_RESULT_STATE_INVALID"

    if not run_validity.valid:
        if (
            not run_validity.failure_reason
            or candidate_selection_executed is not False
            or selected_policy != NOT_EVALUATED
            or len(candidate_results) != 0
            or len(yearly_windows) != 0
            or len(full_span_windows) != 0
            or len(synthetic_bases) != 0
            or m_fraction is not None
            or m_consecutive is not None
            or m_fraction_window_count != 0
            or m_consecutive_window_count != 0
            or synthetic_base_count != 0
            or synthetic_scenario_count != 0
            or synthetic_candidate_comparison_count != 0
            or synthetic_truth_table_mismatch_count != 0
            or len(synthetic_base_metadata) != 0
            or selected_candidate_fraction_headroom_exact_or_null is not None
            or selected_candidate_consecutive_headroom_or_null is not None
            or dict(input_provenance_hashes) != {"invalid_reason_count": 1}
            or dict(error_counts) != {"invalid_reason_count": 1}
        ):
            raise V8BCalibrationBlocked(reason)
        return

    # --- VALID branch: independently recompute everything derivable and
    # require it to exactly match what the caller supplied. Caller-provided
    # candidate rows / envelope / selection / synthetic metadata / error
    # counts are never trusted at face value — each is cross-checked against
    # its own independent source of truth (windows, CANDIDATES, the actual
    # CleanBase objects, and the R1-validated manifest bytes).
    if run_validity.failure_reason is not None or candidate_selection_executed is not True:
        raise V8BCalibrationBlocked(reason)
    if m_fraction is None or m_consecutive is None:
        raise V8BCalibrationBlocked(reason)
    if len(full_span_windows) != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked(reason)
    if not yearly_windows:
        raise V8BCalibrationBlocked(reason)

    recomputed_envelope = compute_global_envelope(list(yearly_windows) + list(full_span_windows))
    if (
        m_fraction != recomputed_envelope.m_fraction
        or m_fraction_window_count != recomputed_envelope.m_fraction_source_window_count
        or m_consecutive != recomputed_envelope.m_consecutive
        or m_consecutive_window_count != recomputed_envelope.m_consecutive_source_window_count
    ):
        raise V8BCalibrationBlocked(reason)

    _verify_candidate_rows_independently(
        candidate_results, yearly_windows, full_span_windows, recomputed_envelope.m_fraction, recomputed_envelope.m_consecutive
    )

    expected_selected_policy = _reference_select_policy(recomputed_envelope.m_fraction, recomputed_envelope.m_consecutive)
    if selected_policy != expected_selected_policy:
        raise V8BCalibrationBlocked(reason)

    if selected_policy == CALIBRATION_NO_DEFENSIBLE_POLICY:
        if (
            selected_candidate_fraction_headroom_exact_or_null is not None
            or selected_candidate_consecutive_headroom_or_null is not None
        ):
            raise V8BCalibrationBlocked(reason)
    else:
        selected_candidate = next((c for c in CANDIDATES if c.id == selected_policy), None)
        if selected_candidate is None:
            raise V8BCalibrationBlocked(reason)
        expected_row = _reference_candidate_row(
            selected_candidate, yearly_windows, full_span_windows, recomputed_envelope.m_fraction, recomputed_envelope.m_consecutive
        )
        if not expected_row["DEFENSIBLE"]:
            raise V8BCalibrationBlocked(reason)
        if (
            selected_candidate_fraction_headroom_exact_or_null != expected_row["fraction_headroom_exact"]
            or selected_candidate_consecutive_headroom_or_null != expected_row["consecutive_headroom"]
        ):
            raise V8BCalibrationBlocked(reason)

    if (
        synthetic_base_count != SYNTHETIC_BASE_COUNT
        or synthetic_scenario_count != SYNTHETIC_SCENARIO_COUNT
        or synthetic_candidate_comparison_count != SYNTHETIC_CANDIDATE_COMPARISON_COUNT
        or synthetic_truth_table_mismatch_count != 0
    ):
        raise V8BCalibrationBlocked(reason)

    _verify_synthetic_base_metadata_against_bases(synthetic_base_metadata, synthetic_bases)

    provenance = dict(input_provenance_hashes)
    if set(provenance) != {"manifest_sha256", "payload_hash_list_sha256", "manifest_payload_count", "bound_payload_count"}:
        raise V8BCalibrationBlocked(reason)
    if type(provenance.get("manifest_payload_count")) is not int or type(provenance.get("bound_payload_count")) is not int:
        raise V8BCalibrationBlocked(reason)
    if (
        provenance.get("manifest_sha256") != EXPECTED_V5B_MANIFEST_SHA256
        or provenance.get("payload_hash_list_sha256") != EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256
        or provenance.get("manifest_payload_count") != EXPECTED_V5B_TICKER_COUNT
        or provenance.get("bound_payload_count") != EXPECTED_V5B_TICKER_COUNT
    ):
        raise V8BCalibrationBlocked(reason)

    if manifest_bytes is None:
        raise V8BCalibrationBlocked(reason)
    validated_manifest = validate_v5b_manifest_provenance(manifest_bytes)
    expected_errors = _expected_error_counts_from_manifest(validated_manifest)
    errors = dict(error_counts)
    if set(errors) != {"failed_count", "retry_count", "http_429_count", "http_5xx_count"}:
        raise V8BCalibrationBlocked(reason)
    for value in errors.values():
        if type(value) is not int or value < 0:
            raise V8BCalibrationBlocked(reason)
    if errors != expected_errors:
        raise V8BCalibrationBlocked(reason)


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
    input_provenance_hashes: Mapping[str, Any],
    error_counts: Mapping[str, Any],
    implementation_git_commit: str,
    calibration_attempt_id: str,
    run_started_utc: str,
    run_completed_or_blocked_utc: str,
    yearly_windows: Sequence[WindowStats] = (),
    full_span_windows: Sequence[WindowStats] = (),
    synthetic_bases: Sequence[CleanBase] = (),
    manifest_bytes: bytes | None = None,
    selected_candidate_fraction_headroom_exact_or_null: Mapping[str, int] | None = None,
    selected_candidate_consecutive_headroom_or_null: int | None = None,
) -> dict[str, Any]:
    _validate_provenance_fields(
        implementation_git_commit, calibration_attempt_id, run_started_utc, run_completed_or_blocked_utc
    )
    _validate_result_state(
        run_validity=run_validity,
        selected_policy=selected_policy,
        candidate_selection_executed=candidate_selection_executed,
        candidate_results=candidate_results,
        yearly_windows=yearly_windows,
        full_span_windows=full_span_windows,
        synthetic_bases=synthetic_bases,
        manifest_bytes=manifest_bytes,
        m_fraction=m_fraction,
        m_fraction_window_count=m_fraction_window_count,
        m_consecutive=m_consecutive,
        m_consecutive_window_count=m_consecutive_window_count,
        synthetic_base_count=synthetic_base_count,
        synthetic_scenario_count=synthetic_scenario_count,
        synthetic_candidate_comparison_count=synthetic_candidate_comparison_count,
        synthetic_truth_table_mismatch_count=synthetic_truth_table_mismatch_count,
        synthetic_base_metadata=synthetic_base_metadata,
        input_provenance_hashes=input_provenance_hashes,
        error_counts=error_counts,
        selected_candidate_fraction_headroom_exact_or_null=selected_candidate_fraction_headroom_exact_or_null,
        selected_candidate_consecutive_headroom_or_null=selected_candidate_consecutive_headroom_or_null,
    )
    artifact: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study": STUDY,
        "calibration_plan_version": PLAN_VERSION,
        "calibration_plan_commit_or_hash": APPROVED_PLAN_COMMIT,
        "approved_plan_commit": APPROVED_PLAN_COMMIT,
        "approved_plan_blob_sha": APPROVED_PLAN_BLOB_SHA,
        "approval_artifact_blob_sha": APPROVAL_ARTIFACT_BLOB_SHA,
        "implementation_git_commit": implementation_git_commit,
        "calibration_attempt_id": calibration_attempt_id,
        "calibration_run_valid": run_validity.valid,
        "run_invalid_reason_or_null": run_validity.failure_reason,
        "candidate_selection_executed": candidate_selection_executed,
        "selected_policy": selected_policy,
        "mechanically_selected_candidate_or_NO_DEFENSIBLE_POLICY_or_NOT_EVALUATED": selected_policy,
        "input_provenance_hashes": dict(input_provenance_hashes),
        "error_counts": dict(error_counts),
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
        "synthetic_base_ticker_count": synthetic_base_count,
        "synthetic_base_selection_rule": SYNTHETIC_BASE_SELECTION_RULE_VERSION,
        "exact_synthetic_placement_formulas_version": SYNTHETIC_PLACEMENT_FORMULAS_VERSION,
        "synthetic_scenario_count": synthetic_scenario_count,
        "synthetic_candidate_comparison_count": synthetic_candidate_comparison_count,
        "full_expected_vs_observed_synthetic_truth_table_mismatch_count": synthetic_truth_table_mismatch_count,
        "synthetic_base_window_start_and_end_metadata": list(synthetic_base_metadata),
        "run_started_utc": run_started_utc,
        "run_completed_or_blocked_utc": run_completed_or_blocked_utc,
    }
    digest = sha256_hex(canonical_json_bytes(artifact))
    return {**artifact, "artifact_self_hash": digest}


def verify_artifact_self_hash(artifact: Mapping[str, Any]) -> bool:
    """INTEGRITY CHECK ONLY — not an acceptance check.

    This only proves the artifact's fields are internally self-consistent
    with its own recorded hash (i.e. nothing was corrupted/edited without
    also updating the hash). It says nothing about whether the *content* is
    scientifically correct: an attacker (or a bug) can mutate a semantic
    field and simply recompute a new, mathematically valid self-hash over
    the mutated content, and this function will still return True. The
    public acceptance API is ``validate_result_artifact_semantics``, which
    independently re-derives every semantic field from its own source of
    truth (windows, CANDIDATES, the actual CleanBase objects, and the
    R1-validated manifest bytes) instead of trusting the artifact's content.
    """

    if "artifact_self_hash" not in artifact:
        return False
    claimed = artifact["artifact_self_hash"]
    without_hash = {key: value for key, value in artifact.items() if key != "artifact_self_hash"}
    return sha256_hex(canonical_json_bytes(without_hash)) == claimed


# ---------------------------------------------------------------------------
# Public acceptance API: full persisted-artifact semantic verification.
#
# This is deliberately NOT implemented by calling build_result_artifact()
# and diffing the result — it is a verification path over the artifact's
# own JSON-shaped fields (as returned by run_data_quality_calibration, or
# as loaded back from wherever it was persisted), re-deriving every
# semantic field from its own independent source of truth. Self-hash
# integrity (verify_artifact_self_hash) is checked first but is not by
# itself sufficient: an attacker who mutates content and re-signs the hash
# passes that check while still failing everything below.
# ---------------------------------------------------------------------------


def _fraction_from_json(value: Any) -> Fraction:
    reason = "CALIBRATION_RESULT_STATE_INVALID"
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise V8BCalibrationBlocked(reason)
    numerator, denominator = value.get("numerator"), value.get("denominator")
    if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
        raise V8BCalibrationBlocked(reason)
    return Fraction(numerator, denominator)


def _verify_invalid_artifact_fields(artifact: Mapping[str, Any]) -> None:
    reason = "CALIBRATION_RESULT_STATE_INVALID"
    if (
        not artifact.get("run_invalid_reason_or_null")
        or artifact.get("candidate_selection_executed") is not False
        or artifact.get("selected_policy") != NOT_EVALUATED
        or artifact.get("mechanically_selected_candidate_or_NO_DEFENSIBLE_POLICY_or_NOT_EVALUATED") != NOT_EVALUATED
        or artifact.get("candidate_results") != []
        or artifact.get("M_fraction_exact") is not None
        or artifact.get("M_consecutive") is not None
        or artifact.get("M_fraction_source_window_count") != 0
        or artifact.get("M_consecutive_source_window_count") != 0
        or artifact.get("synthetic_base_count") != 0
        or artifact.get("synthetic_base_ticker_count") != 0
        or artifact.get("synthetic_scenario_count") != 0
        or artifact.get("synthetic_candidate_comparison_count") != 0
        or artifact.get("full_expected_vs_observed_synthetic_truth_table_mismatch_count") != 0
        or artifact.get("synthetic_base_window_start_and_end_metadata") != []
        or artifact.get("selected_candidate_fraction_headroom_exact_or_null") is not None
        or artifact.get("selected_candidate_consecutive_headroom_or_null") is not None
        or artifact.get("input_provenance_hashes") != {"invalid_reason_count": 1}
        or artifact.get("error_counts") != {"invalid_reason_count": 1}
    ):
        raise V8BCalibrationBlocked(reason)


def _verify_valid_artifact_fields(
    artifact: Mapping[str, Any],
    yearly_windows: Sequence[WindowStats],
    full_span_windows: Sequence[WindowStats],
    synthetic_bases: Sequence[CleanBase],
    manifest_bytes: bytes | None,
) -> None:
    reason = "CALIBRATION_RESULT_STATE_INVALID"

    if artifact.get("run_invalid_reason_or_null") is not None:
        raise V8BCalibrationBlocked(reason)
    if artifact.get("candidate_selection_executed") is not True:
        raise V8BCalibrationBlocked(reason)
    if len(full_span_windows) != EXPECTED_V5B_TICKER_COUNT:
        raise V8BCalibrationBlocked(reason)
    if not yearly_windows:
        raise V8BCalibrationBlocked(reason)

    recomputed_envelope = compute_global_envelope(list(yearly_windows) + list(full_span_windows))

    m_fraction = _fraction_from_json(artifact.get("M_fraction_exact"))
    if m_fraction != recomputed_envelope.m_fraction:
        raise V8BCalibrationBlocked(reason)
    if artifact.get("M_fraction_source_window_count") != recomputed_envelope.m_fraction_source_window_count:
        raise V8BCalibrationBlocked(reason)
    m_consecutive = artifact.get("M_consecutive")
    if type(m_consecutive) is not int or m_consecutive != recomputed_envelope.m_consecutive:
        raise V8BCalibrationBlocked(reason)
    if artifact.get("M_consecutive_source_window_count") != recomputed_envelope.m_consecutive_source_window_count:
        raise V8BCalibrationBlocked(reason)

    candidate_results = artifact.get("candidate_results")
    if not isinstance(candidate_results, list):
        raise V8BCalibrationBlocked(reason)
    _verify_candidate_rows_independently(
        candidate_results, yearly_windows, full_span_windows, recomputed_envelope.m_fraction, recomputed_envelope.m_consecutive
    )

    selected_policy = artifact.get("selected_policy")
    expected_selected_policy = _reference_select_policy(recomputed_envelope.m_fraction, recomputed_envelope.m_consecutive)
    if selected_policy != expected_selected_policy:
        raise V8BCalibrationBlocked(reason)
    if artifact.get("mechanically_selected_candidate_or_NO_DEFENSIBLE_POLICY_or_NOT_EVALUATED") != selected_policy:
        raise V8BCalibrationBlocked(reason)

    if selected_policy == CALIBRATION_NO_DEFENSIBLE_POLICY:
        if (
            artifact.get("selected_candidate_fraction_headroom_exact_or_null") is not None
            or artifact.get("selected_candidate_consecutive_headroom_or_null") is not None
        ):
            raise V8BCalibrationBlocked(reason)
    else:
        matching_row = next((row for row in candidate_results if row.get("candidate_id") == selected_policy), None)
        if matching_row is None or matching_row.get("DEFENSIBLE") is not True:
            raise V8BCalibrationBlocked(reason)
        if (
            artifact.get("selected_candidate_fraction_headroom_exact_or_null") != matching_row.get("fraction_headroom_exact")
            or artifact.get("selected_candidate_consecutive_headroom_or_null") != matching_row.get("consecutive_headroom")
        ):
            raise V8BCalibrationBlocked(reason)

    if (
        artifact.get("synthetic_base_count") != SYNTHETIC_BASE_COUNT
        or artifact.get("synthetic_base_ticker_count") != SYNTHETIC_BASE_COUNT
        or artifact.get("synthetic_scenario_count") != SYNTHETIC_SCENARIO_COUNT
        or artifact.get("synthetic_candidate_comparison_count") != SYNTHETIC_CANDIDATE_COMPARISON_COUNT
        or artifact.get("full_expected_vs_observed_synthetic_truth_table_mismatch_count") != 0
    ):
        raise V8BCalibrationBlocked(reason)

    synthetic_metadata = artifact.get("synthetic_base_window_start_and_end_metadata")
    if not isinstance(synthetic_metadata, list):
        raise V8BCalibrationBlocked(reason)
    _verify_synthetic_base_metadata_against_bases(synthetic_metadata, synthetic_bases)

    provenance = artifact.get("input_provenance_hashes")
    if not isinstance(provenance, dict) or set(provenance) != {
        "manifest_sha256",
        "payload_hash_list_sha256",
        "manifest_payload_count",
        "bound_payload_count",
    }:
        raise V8BCalibrationBlocked(reason)
    if type(provenance.get("manifest_payload_count")) is not int or type(provenance.get("bound_payload_count")) is not int:
        raise V8BCalibrationBlocked(reason)
    if (
        provenance.get("manifest_sha256") != EXPECTED_V5B_MANIFEST_SHA256
        or provenance.get("payload_hash_list_sha256") != EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256
        or provenance.get("manifest_payload_count") != EXPECTED_V5B_TICKER_COUNT
        or provenance.get("bound_payload_count") != EXPECTED_V5B_TICKER_COUNT
    ):
        raise V8BCalibrationBlocked(reason)

    if manifest_bytes is None:
        raise V8BCalibrationBlocked(reason)
    validated_manifest = validate_v5b_manifest_provenance(manifest_bytes)
    expected_errors = _expected_error_counts_from_manifest(validated_manifest)
    error_counts = artifact.get("error_counts")
    if not isinstance(error_counts, dict) or set(error_counts) != {
        "failed_count",
        "retry_count",
        "http_429_count",
        "http_5xx_count",
    }:
        raise V8BCalibrationBlocked(reason)
    for value in error_counts.values():
        if type(value) is not int or value < 0:
            raise V8BCalibrationBlocked(reason)
    if dict(error_counts) != expected_errors:
        raise V8BCalibrationBlocked(reason)


def validate_result_artifact_semantics(
    artifact: Mapping[str, Any],
    *,
    yearly_windows: Sequence[WindowStats] = (),
    full_span_windows: Sequence[WindowStats] = (),
    synthetic_bases: Sequence[CleanBase] = (),
    manifest_bytes: bytes | None = None,
) -> None:
    """The public acceptance API for a V8B calibration result artifact.

    Raises ``V8BCalibrationBlocked`` on any semantic inconsistency; returns
    ``None`` (accepts) only if every field is independently verifiable
    against its own source of truth. Checks, in order:

    1. self-hash integrity (necessary, not sufficient — see
       ``verify_artifact_self_hash``'s docstring);
    2. legal run-state shape (INVALID / VALID-D-empty / VALID-D-nonempty);
    3. for a VALID artifact: the global envelope recomputed from
       ``yearly_windows``/``full_span_windows``; all 30 candidate rows
       recomputed independently (see ``_reference_candidate_row``); the
       selected policy recomputed independently (see
       ``_reference_select_policy``); the synthetic base metadata bound to
       the actual ``synthetic_bases`` (``CleanBase``) objects used in the
       run; and ``error_counts`` bound to a fresh
       ``validate_v5b_manifest_provenance(manifest_bytes)`` re-run against
       the exact manifest bytes, not the caller's unverified claim.

    For an INVALID artifact, ``yearly_windows``/``full_span_windows``/
    ``synthetic_bases``/``manifest_bytes`` are not required and are ignored.
    """

    if not verify_artifact_self_hash(artifact):
        raise V8BCalibrationBlocked("CALIBRATION_ARTIFACT_SELF_HASH_MISMATCH")

    calibration_run_valid = artifact.get("calibration_run_valid")
    if calibration_run_valid is False:
        _verify_invalid_artifact_fields(artifact)
        return
    if calibration_run_valid is not True:
        raise V8BCalibrationBlocked("CALIBRATION_RESULT_STATE_INVALID")

    _verify_valid_artifact_fields(artifact, yearly_windows, full_span_windows, synthetic_bases, manifest_bytes)


# ---------------------------------------------------------------------------
# 26. Orchestration (pure artifact construction only; no real artifact write)
# ---------------------------------------------------------------------------


def run_data_quality_calibration(
    *,
    repository_root: Path,
    manifest_bytes: bytes,
    ticker_payloads: Mapping[str, InMemoryPayload],
    implementation_git_commit: str,
    calibration_attempt_id: str,
    run_started_utc: str | None = None,
) -> dict[str, Any]:
    """Pure (no filesystem write) end-to-end calibration run.

    ``manifest_bytes`` and every payload in ``ticker_payloads`` are supplied
    entirely by the caller as in-memory bytes; this function never opens a
    V5-B cache path. R1 (manifest provenance) is enforced here, before any
    payload is parsed, and every supplied payload must bind exactly to the
    R1-validated manifest (ticker/relative_path/byte_count/SHA-256) before
    parsing — a future adapter cannot route data around either check.
    """

    started = run_started_utc or _utc_now_iso()

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
            input_provenance_hashes={"invalid_reason_count": 1},
            error_counts={"invalid_reason_count": 1},
            implementation_git_commit=implementation_git_commit,
            calibration_attempt_id=calibration_attempt_id,
            run_started_utc=started,
            run_completed_or_blocked_utc=_utc_now_iso(),
        )

    # 1. repository contract
    try:
        verify_repository_contract(repository_root)
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    # 2. pinned parser (verified bytes compiled and exec'd directly)
    try:
        pinned_module = _load_pinned_collector(repository_root)
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    # 3. R1: manifest provenance against the fixed, non-overridable real hash
    try:
        manifest = validate_v5b_manifest_provenance(manifest_bytes)
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    # 4. bind every supplied payload to the R1-validated manifest
    try:
        bound_payloads = bind_payloads_to_manifest(manifest, ticker_payloads, pinned_module)
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    # 5. parse payloads
    observations_by_ticker: dict[str, tuple[Observation, ...]] = {}
    try:
        for canonical, payload_bytes in bound_payloads.items():
            _, restricted = parse_ticker_observations(canonical, payload_bytes, pinned_module)
            observations_by_ticker[canonical] = restricted
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    # 6. stats / envelope
    yearly_windows: list[WindowStats] = []
    full_span_windows: list[WindowStats] = []
    try:
        for observations in observations_by_ticker.values():
            full_span_windows.append(compute_full_span_stats(observations))
            yearly = compute_yearly_window_stats(observations)
            yearly_windows.extend(stats for stats in yearly.values() if stats is not None)
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    envelope = compute_global_envelope(yearly_windows + full_span_windows)

    # 7. synthetic verification
    try:
        bases = select_synthetic_bases(observations_by_ticker, pinned_module)
    except V8BCalibrationBlocked as error:
        return blocked(error.reason)

    verification = run_synthetic_semantics_verification(bases, pinned_module)
    if verification.classifier_mismatch:
        return blocked("SYNTHETIC_CLASSIFIER_MISMATCH")
    if verification.truth_table_mismatch_count:
        return blocked("SYNTHETIC_POLICY_SEMANTICS_MISMATCH")

    # 8. establish valid run
    run_validity = VALID_RUN

    # 9. candidate selection (selection itself requires a valid run)
    selected_policy, _ = select_policy(run_validity, envelope.m_fraction, envelope.m_consecutive)
    candidate_results = _compute_candidate_results(
        yearly_windows, full_span_windows, envelope.m_fraction, envelope.m_consecutive
    )
    selected_fraction_headroom, selected_consecutive_headroom = _compute_selected_headrooms(
        selected_policy, envelope.m_fraction, envelope.m_consecutive
    )

    synthetic_base_metadata = [
        {
            "base_index": base.base_index,
            "ticker_sha256": base.ticker_sha256,
            "window_start": base.window_start,
            "window_end": base.window_end,
        }
        for base in bases
    ]

    input_provenance_hashes = {
        "manifest_sha256": EXPECTED_V5B_MANIFEST_SHA256,
        "payload_hash_list_sha256": EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256,
        "manifest_payload_count": EXPECTED_V5B_TICKER_COUNT,
        "bound_payload_count": len(bound_payloads),
    }
    error_counts = {
        "failed_count": manifest.get("failed_count", 0),
        "retry_count": manifest.get("retry_count", 0),
        "http_429_count": manifest.get("http_429_count", 0),
        "http_5xx_count": manifest.get("http_5xx_count", 0),
    }

    # 10. artifact
    return build_result_artifact(
        run_validity=run_validity,
        selected_policy=selected_policy,
        candidate_selection_executed=True,
        candidate_results=candidate_results,
        yearly_windows=yearly_windows,
        full_span_windows=full_span_windows,
        synthetic_bases=bases,
        manifest_bytes=manifest_bytes,
        m_fraction=envelope.m_fraction,
        m_fraction_window_count=envelope.m_fraction_source_window_count,
        m_consecutive=envelope.m_consecutive,
        m_consecutive_window_count=envelope.m_consecutive_source_window_count,
        synthetic_base_count=SYNTHETIC_BASE_COUNT,
        synthetic_scenario_count=verification.scenario_count,
        synthetic_candidate_comparison_count=verification.comparison_count,
        synthetic_truth_table_mismatch_count=verification.truth_table_mismatch_count,
        synthetic_base_metadata=synthetic_base_metadata,
        input_provenance_hashes=input_provenance_hashes,
        error_counts=error_counts,
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
        run_validity=run_validity_for_reason("SYNTHETIC_BASE_SELECTION_BLOCKED"),
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
        input_provenance_hashes={"invalid_reason_count": 1},
        error_counts={"invalid_reason_count": 1},
        implementation_git_commit="0" * 40,
        calibration_attempt_id="static-check",
        run_started_utc=_utc_now_iso(),
        run_completed_or_blocked_utc=_utc_now_iso(),
    )
    if not verify_artifact_self_hash(dummy):
        raise V8BCalibrationBlocked("CALIBRATION_ARTIFACT_SELF_HASH_MISMATCH")
    mutated = dict(dummy)
    mutated["calibration_attempt_id"] = "mutated"
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
    "InMemoryPayload",
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
    "SYNTHETIC_BASE_SELECTION_RULE_VERSION",
    "SYNTHETIC_CANDIDATE_COMPARISON_COUNT",
    "SYNTHETIC_PLACEMENT_FORMULAS_VERSION",
    "SYNTHETIC_SCENARIO_COUNT",
    "SYNTHETIC_SEQUENCE_LENGTH",
    "SyntheticScenario",
    "SyntheticVerificationResult",
    "V8BCalibrationBlocked",
    "VALID_RUN",
    "WindowStats",
    "apply_corruption",
    "bind_payloads_to_manifest",
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
    "validate_result_artifact_semantics",
    "validate_v5b_manifest_provenance",
    "validate_v5b_manifest_structure",
    "verify_repository_contract",
]

# NOTE: verify_artifact_self_hash is deliberately NOT exported here. It is
# an integrity-only check (see its docstring) and must not be mistaken for
# the acceptance API. It remains directly importable
# (``from src.v8b_data_quality_calibration import verify_artifact_self_hash``)
# for callers/tests that specifically want the integrity check, but the
# public acceptance API advertised by this module is
# ``validate_result_artifact_semantics``.
