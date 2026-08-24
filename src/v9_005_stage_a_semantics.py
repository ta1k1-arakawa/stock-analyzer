"""V9_006_HIGH_2_SEMANTIC_VALIDATION_IMPLEMENTATION.

Implements, in code, the exact GPT-bound methodology recorded in
`V9_006_STAGE_A_SEMANTIC_VALIDATION_METHODOLOGY.md` for V9_005's
`SECURITY_TYPE`, `CANONICAL_IDENTITY`, `LISTING_TRANSITIONS`/
`DELISTING_TRANSITIONS`/`MARKET_TRANSITIONS`, `EFFECTIVE_DATE`, and
`RECONSTRUCTION` Stage-A evidence items.

This module implements no F1-F7 network traversal, fetch, or parser: it
operates only on already-structured `SemanticEvent`/`TerminalIdentityState`
input supplied by a caller. A future, separately reviewed F2-F7
acquisition/parser-integration task is responsible for turning locked raw
JPX bytes into that structured input. This module performs zero network
access and imports nothing from `src.v9_005_stage_a_jpx_probe` (the
dependency runs the other way).

Design note on "no evidence acquired" (production fail-closed default):
`compute_semantic_validation_result` returns an all-`False` result when
`terminal_identities` is empty, mirroring the existing repository
convention that empty/absent evidence fails closed rather than passing
vacuously (see `src.v9_005_stage_a_jpx_probe._family_fully_covered`). A
non-empty `terminal_identities` with zero events for a given
code/dimension is a different, legitimate case (a genuine absence of any
recorded transition for that code/dimension) and is evaluated on its own
merits, not forced to fail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date as _date
from typing import Any, Mapping, Sequence

# --- Canonical code grammar (V9_006_STAGE_A_SEMANTIC_VALIDATION_METHODOLOGY.md #1) -----

# Position 2 and 4 may be a digit or one of these ASCII letters (JPX's
# official four-character stock specific-name-code grammar, effective for
# codes assigned from 2024 onward). No other letter is valid there, and
# position 1/3 must always be a digit.
_ALLOWED_ALT_CHARS = frozenset("ACDFGHJKLMNPRSTUWXY")
_NUMERIC_FLOAT_ARTIFACT_RE = re.compile(r"^\d+\.0$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

INVALID_CANONICAL_CODE = "INVALID_CANONICAL_CODE"
AMBIGUOUS_REUSED_SECURITY_CODE = "AMBIGUOUS_REUSED_SECURITY_CODE"
CONFLICTING_TRANSITION_EVIDENCE = "CONFLICTING_TRANSITION_EVIDENCE"
AMBIGUOUS_EFFECTIVE_DATE = "AMBIGUOUS_EFFECTIVE_DATE"
INVALID_SEMANTIC_EVENT = "INVALID_SEMANTIC_EVENT"
INVALID_TERMINAL_IDENTITY_STATE = "INVALID_TERMINAL_IDENTITY_STATE"
DUPLICATE_CANONICAL_IDENTITY = "DUPLICATE_CANONICAL_IDENTITY"


class SemanticValidationError(ValueError):
    """Raised for a single malformed canonical code, date, or event. Never
    raised out of `compute_semantic_validation_result` itself -- that
    function catches these per-item and folds them into the returned
    pass/fail booleans, since one bad input item must fail only the
    evidence items it actually taints, not raise and abort the whole
    Stage-A evidence computation."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def normalize_canonical_code(raw: object) -> str:
    """Mechanically remove spreadsheet representation artifacts only when
    unambiguous: surrounding whitespace, and an integral numeric-cell
    trailing ".0" (e.g. "1301.0" -> "1301"). Then uppercase ASCII letters.
    Any other shape is left as-is for `validate_canonical_code` to reject."""
    if not isinstance(raw, str):
        raise SemanticValidationError(INVALID_CANONICAL_CODE)
    text = raw.strip()
    if _NUMERIC_FLOAT_ARTIFACT_RE.fullmatch(text):
        text = text[:-2]
    return text.upper()


def validate_canonical_code(raw: object) -> str:
    """Exact 4-character grammar: position 1 and 3 must be digits; position
    2 and 4 may be a digit or one of the allowed letters. Never
    digits-only-assumed (JPX began assigning letters from 2024). A
    5-character code (using a reserved security-type character) is
    rejected here -- length != 4 -- and is therefore never accepted as the
    canonical ordinary-common identity."""
    code = normalize_canonical_code(raw)
    if len(code) != 4 or not code.isascii():
        raise SemanticValidationError(INVALID_CANONICAL_CODE)
    if not code[0].isdigit() or not code[2].isdigit():
        raise SemanticValidationError(INVALID_CANONICAL_CODE)
    for i in (1, 3):
        ch = code[i]
        if not (ch.isdigit() or ch in _ALLOWED_ALT_CHARS):
            raise SemanticValidationError(INVALID_CANONICAL_CODE)
    return code


def _validate_iso_date(value: object) -> str:
    if not isinstance(value, str) or not _ISO_DATE_RE.fullmatch(value):
        raise SemanticValidationError(AMBIGUOUS_EFFECTIVE_DATE)
    try:
        year, month, day = (int(part) for part in value.split("-"))
        _date(year, month, day)
    except ValueError as exc:
        raise SemanticValidationError(AMBIGUOUS_EFFECTIVE_DATE) from exc
    return value


# --- Structured semantic events (#2, #3, #5) ---------------------------------

DIMENSION_LISTED_STATE = "LISTED_STATE"
DIMENSION_MARKET_STATE = "MARKET_STATE"
DIMENSION_SECURITY_TYPE_STATE = "SECURITY_TYPE_STATE"
VALID_DIMENSIONS = frozenset({DIMENSION_LISTED_STATE, DIMENSION_MARKET_STATE, DIMENSION_SECURITY_TYPE_STATE})

SECURITY_TYPE_ELIGIBLE = "ELIGIBLE_DOMESTIC_ORDINARY_COMMON"
SECURITY_TYPE_INELIGIBLE = "EXPLICITLY_INELIGIBLE"
SECURITY_TYPE_UNKNOWN = "UNKNOWN"
VALID_SECURITY_TYPE_STATES = frozenset({SECURITY_TYPE_ELIGIBLE, SECURITY_TYPE_INELIGIBLE, SECURITY_TYPE_UNKNOWN})


@dataclass(frozen=True)
class SemanticEvent:
    """One structured, already-parsed official transition record. Never
    constructed by this module from raw bytes -- callers (a future F2-F7
    acquisition/parser task) supply these from actual observed official
    evidence; this module never synthesizes an unassigned code or an event
    that was not actually observed."""

    canonical_code: str
    effective_date: str
    dimension: str
    before_state: Any
    after_state: Any
    source_family: str


@dataclass(frozen=True)
class TerminalIdentityState:
    """One reconstructed candidate identity's state as of the terminal
    seed date `T` -- not necessarily "currently listed": a code delisted
    before `T` has `listed_state=False` here, and its listing/delisting
    history is still reconstructed via `SemanticEvent`s below."""

    listed_state: bool
    market_state: str
    security_type_state: str


def canonical_bytes(value: Any) -> bytes:
    """Same canonical JSON convention as
    `src.v9_005_stage_a_jpx_probe.canonical_bytes` (sort_keys, compact
    separators, UTF-8, trailing LF) -- duplicated locally so this module
    stays free of any dependency on that module."""
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _validate_event(event: SemanticEvent) -> SemanticEvent:
    code = validate_canonical_code(event.canonical_code)
    effective_date = _validate_iso_date(event.effective_date)
    if not isinstance(event.source_family, str) or not event.source_family:
        raise SemanticValidationError(INVALID_SEMANTIC_EVENT)
    if event.dimension == DIMENSION_LISTED_STATE:
        if not isinstance(event.before_state, bool) or not isinstance(event.after_state, bool):
            raise SemanticValidationError(INVALID_SEMANTIC_EVENT)
        if event.before_state == event.after_state:
            raise SemanticValidationError(INVALID_SEMANTIC_EVENT)
    elif event.dimension == DIMENSION_SECURITY_TYPE_STATE:
        if event.before_state not in VALID_SECURITY_TYPE_STATES or event.after_state not in VALID_SECURITY_TYPE_STATES:
            raise SemanticValidationError(INVALID_SEMANTIC_EVENT)
    elif event.dimension == DIMENSION_MARKET_STATE:
        if not isinstance(event.before_state, str) or not event.before_state:
            raise SemanticValidationError(INVALID_SEMANTIC_EVENT)
        if not isinstance(event.after_state, str) or not event.after_state:
            raise SemanticValidationError(INVALID_SEMANTIC_EVENT)
    else:
        raise SemanticValidationError(INVALID_SEMANTIC_EVENT)
    return SemanticEvent(code, effective_date, event.dimension, event.before_state, event.after_state, event.source_family)


@dataclass(frozen=True)
class _TerminalStateValidity:
    """Per-field validity of one `TerminalIdentityState` before semantic
    use (V9_006_HIGH_2_SEM_IMPL_HIGH_1). Each field is checked
    independently so that, e.g., a bad `security_type_state` fails only
    the gates that field is responsible for, without discarding a
    perfectly valid `listed_state`/`market_state` reconstruction."""

    listed_state_valid: bool
    market_state_valid: bool
    security_type_state_valid: bool

    @property
    def all_valid(self) -> bool:
        return self.listed_state_valid and self.market_state_valid and self.security_type_state_valid


def _validate_terminal_identity_state(state: object) -> _TerminalStateValidity:
    """Never raises -- returns per-field validity so the caller can fail
    closed on exactly the affected evidence items. A value that is not
    even a `TerminalIdentityState` instance fails all three fields (there
    is nothing trustworthy to read). `listed_state` must be an actual
    `bool` -- `isinstance(1, bool)` is `False` in Python, so an int like
    `1` is correctly rejected even though `1 == True`. `market_state` must
    be a non-empty `str` (no market enumeration is invented here).
    `security_type_state` must be exactly one of
    `VALID_SECURITY_TYPE_STATES`."""
    if not isinstance(state, TerminalIdentityState):
        return _TerminalStateValidity(False, False, False)
    listed_state_valid = isinstance(state.listed_state, bool)
    market_state_valid = isinstance(state.market_state, str) and state.market_state != ""
    security_type_state_valid = state.security_type_state in VALID_SECURITY_TYPE_STATES
    return _TerminalStateValidity(listed_state_valid, market_state_valid, security_type_state_valid)


def _dedupe_and_detect_conflicts(
    events: Sequence[SemanticEvent],
) -> tuple[dict[tuple[str, str, str], SemanticEvent], frozenset[str]]:
    """Group by (canonical_code, effective_date, dimension). Identical
    corroborating events (e.g. matching F2 and F3 records) collapse
    deterministically into one. A differing (before_state, after_state)
    pair for the same key is `CONFLICTING_TRANSITION_EVIDENCE` -- dropped
    from the collapsed map and its dimension recorded as conflicted."""
    grouped: dict[tuple[str, str, str], list[SemanticEvent]] = {}
    for event in events:
        key = (event.canonical_code, event.effective_date, event.dimension)
        grouped.setdefault(key, []).append(event)

    collapsed: dict[tuple[str, str, str], SemanticEvent] = {}
    conflict_dimensions: set[str] = set()
    for key, group in grouped.items():
        distinct = {(item.before_state, item.after_state) for item in group}
        if len(distinct) > 1:
            conflict_dimensions.add(key[2])
            continue
        collapsed[key] = group[0]
    return collapsed, frozenset(conflict_dimensions)


def _reconstruct_dimension_timeline(
    events_for_code_dim: Sequence[SemanticEvent],
    terminal_value: Any,
) -> tuple[list[tuple[str | None, Any]], bool]:
    """#7: begin from the terminal value, replay backward (descending
    effective date) through the given events, then replay forward
    (ascending, the exact same events) from the recovered earliest value.
    Returns (timeline, consistent) where timeline is
    [(None, earliest_value), (date_1, value_after_event_1), ...] sorted
    ascending, and `consistent` is True iff every step's before/after
    matched and the final forward-replayed value byte-equals
    `terminal_value`. A mismatch at any step (a broken chain, or a
    terminal value that disagrees with what the events actually imply)
    fails closed immediately -- there is no reconciliation."""
    ordered = sorted(events_for_code_dim, key=lambda e: e.effective_date)

    current = terminal_value
    for event in reversed(ordered):
        if event.after_state != current:
            return [], False
        current = event.before_state
    earliest = current

    timeline: list[tuple[str | None, Any]] = [(None, earliest)]
    current = earliest
    for event in ordered:
        if event.before_state != current:
            return [], False
        current = event.after_state
        timeline.append((event.effective_date, current))

    return timeline, current == terminal_value


def _value_at(timeline: Sequence[tuple[str | None, Any]], at_date: str | None) -> Any:
    value = timeline[0][1]
    if at_date is None:
        return value
    for entry_date, entry_value in timeline[1:]:
        if entry_date is not None and entry_date <= at_date:
            value = entry_value
    return value


def _unknown_security_type_while_listed(
    listed_timeline: Sequence[tuple[str | None, Any]],
    security_timeline: Sequence[tuple[str | None, Any]],
) -> bool:
    """#4: any point where the code is listed and its security type
    resolves to UNKNOWN fails `security_type_pass`."""
    checkpoints = sorted(
        {entry_date for entry_date, _ in listed_timeline if entry_date is not None}
        | {entry_date for entry_date, _ in security_timeline if entry_date is not None}
    )
    if listed_timeline[0][1] is True and security_timeline[0][1] == SECURITY_TYPE_UNKNOWN:
        return True
    for checkpoint in checkpoints:
        if _value_at(listed_timeline, checkpoint) is True and _value_at(security_timeline, checkpoint) == SECURITY_TYPE_UNKNOWN:
            return True
    return False


def _fail_closed_result() -> dict[str, Any]:
    """Production/"no evidence acquired" default. Every gate is `False`;
    never a vacuous pass over zero identities."""
    return {
        "listing_transition_pass": False,
        "delisting_transition_pass": False,
        "market_transition_pass": False,
        "security_type_pass": False,
        "canonical_identity_pass": False,
        "effective_date_pass": False,
        "deterministic_reconstruction_pass": False,
        "reconstructed_identity_count": 0,
        "canonical_state": {},
        "reasons": (),
    }


def compute_semantic_validation_result(
    *,
    terminal_identities: Mapping[str, TerminalIdentityState],
    events: Sequence[SemanticEvent],
) -> dict[str, Any]:
    """Compute the semantic validation result -- `listing_transition_pass`,
    `delisting_transition_pass`, `market_transition_pass`,
    `security_type_pass`, `canonical_identity_pass`, `effective_date_pass`,
    `deterministic_reconstruction_pass`, plus `reconstructed_identity_count`
    and the reconstructed `canonical_state` -- derived only from the given
    structured `terminal_identities`/`events`. Never a monthly-family
    coverage proxy and never a caller-supplied arbitrary pass boolean.

    With no terminal identities at all (no semantic evidence acquired --
    the current production state, since F2-F7 acquisition/parsing is not
    yet implemented) this fails closed: see `_fail_closed_result`."""
    if not terminal_identities:
        return _fail_closed_result()

    reasons: set[str] = set()
    any_invalid_listing = False
    any_invalid_delisting = False
    any_invalid_market = False
    any_ambiguous_date = False
    unattributed_invalid = False

    valid_events: list[SemanticEvent] = []
    for event in events:
        try:
            valid_events.append(_validate_event(event))
        except SemanticValidationError as exc:
            reasons.add(exc.reason)
            if exc.reason == AMBIGUOUS_EFFECTIVE_DATE:
                any_ambiguous_date = True
            dimension = event.dimension if event.dimension in VALID_DIMENSIONS else None
            if dimension == DIMENSION_LISTED_STATE:
                # Malformed LISTED_STATE events can't be attributed to
                # listing vs. delisting with confidence -- fail both.
                any_invalid_listing = True
                any_invalid_delisting = True
            elif dimension == DIMENSION_MARKET_STATE:
                any_invalid_market = True
            elif dimension != DIMENSION_SECURITY_TYPE_STATE:
                unattributed_invalid = True

    # V9_006_HIGH_2_SEM_IMPL_HIGH_1: group by canonical code AFTER
    # normalization so that two distinct raw terminal_identities keys
    # normalizing to the same canonical_code (e.g. "1301" and " 1301 ", or
    # "130a" and "130A") are detected as a collision -- never silently
    # overwritten. Also validate each surviving state's fields
    # independently before any semantic use.
    normalized_groups: dict[str, list[str]] = {}
    any_invalid_identity = False
    for raw_code in terminal_identities:
        try:
            code = validate_canonical_code(raw_code)
        except SemanticValidationError:
            any_invalid_identity = True
            reasons.add(INVALID_CANONICAL_CODE)
            continue
        normalized_groups.setdefault(code, []).append(raw_code)

    identities: dict[str, TerminalIdentityState] = {}
    any_invalid_listed_state = False
    any_invalid_market_state = False
    any_invalid_security_type_state = False
    for code, raw_keys in normalized_groups.items():
        if len(raw_keys) > 1:
            any_invalid_identity = True
            reasons.add(DUPLICATE_CANONICAL_IDENTITY)
            continue
        state = terminal_identities[raw_keys[0]]
        validity = _validate_terminal_identity_state(state)
        if not validity.all_valid:
            any_invalid_identity = True
            reasons.add(INVALID_TERMINAL_IDENTITY_STATE)
            if not validity.listed_state_valid:
                any_invalid_listed_state = True
            if not validity.market_state_valid:
                any_invalid_market_state = True
            if not validity.security_type_state_valid:
                any_invalid_security_type_state = True
            continue
        identities[code] = state

    collapsed, conflict_dimensions = _dedupe_and_detect_conflicts(valid_events)
    if conflict_dimensions:
        reasons.add(CONFLICTING_TRANSITION_EVIDENCE)
    listed_conflict = DIMENSION_LISTED_STATE in conflict_dimensions
    market_conflict = DIMENSION_MARKET_STATE in conflict_dimensions
    security_type_conflict = DIMENSION_SECURITY_TYPE_STATE in conflict_dimensions

    by_code_dim: dict[tuple[str, str], list[SemanticEvent]] = {}
    for (code, _effective_date, dimension), event in collapsed.items():
        by_code_dim.setdefault((code, dimension), []).append(event)

    reused_code_violation = False
    reconstruction_consistent = True
    security_type_ok = not security_type_conflict
    canonical_state: dict[str, dict[str, Any]] = {}

    for code, terminal_state in identities.items():
        listed_events = by_code_dim.get((code, DIMENSION_LISTED_STATE), [])
        listing_episode_starts = [e for e in listed_events if e.after_state is True]
        if len(listing_episode_starts) > 1:
            reused_code_violation = True

        dimension_timelines: dict[str, list[tuple[str | None, Any]]] = {}
        for dimension, terminal_value in (
            (DIMENSION_LISTED_STATE, terminal_state.listed_state),
            (DIMENSION_MARKET_STATE, terminal_state.market_state),
            (DIMENSION_SECURITY_TYPE_STATE, terminal_state.security_type_state),
        ):
            dim_events = by_code_dim.get((code, dimension), [])
            timeline, consistent = _reconstruct_dimension_timeline(dim_events, terminal_value)
            if not consistent:
                reconstruction_consistent = False
                if dimension == DIMENSION_LISTED_STATE:
                    # Per methodology #2, a broken LISTED_STATE chain is
                    # itself "conflicting identity evidence" for this code.
                    reused_code_violation = True
                dimension_timelines[dimension] = [(None, terminal_value)]
            else:
                dimension_timelines[dimension] = timeline

        if _unknown_security_type_while_listed(
            dimension_timelines[DIMENSION_LISTED_STATE], dimension_timelines[DIMENSION_SECURITY_TYPE_STATE],
        ):
            security_type_ok = False

        canonical_state[code] = {
            "listed_state": terminal_state.listed_state,
            "market_state": terminal_state.market_state,
            "security_type_state": terminal_state.security_type_state,
        }

    if reused_code_violation:
        reasons.add(AMBIGUOUS_REUSED_SECURITY_CODE)
    if unattributed_invalid:
        reasons.add(INVALID_SEMANTIC_EVENT)

    # V9_006_HIGH_2_SEM_IMPL_HIGH_1: any invalid terminal-identity field
    # (of any code) or any normalized-code collision is folded into
    # any_invalid_identity below, which fails canonical_identity_pass and
    # deterministic_reconstruction_pass outright -- semantic validation
    # cannot PASS with an invalid or colliding terminal identity in the
    # input, even if every other identity/event is perfectly clean. Each
    # invalid field additionally fails its own more specific gate(s).
    listing_ok = not any_invalid_listing and not listed_conflict and not unattributed_invalid and not any_invalid_listed_state
    delisting_ok = (
        not any_invalid_delisting and not listed_conflict and not unattributed_invalid and not any_invalid_listed_state
    )
    market_ok = not any_invalid_market and not market_conflict and not unattributed_invalid and not any_invalid_market_state
    security_type_ok = security_type_ok and not any_invalid_security_type_state
    canonical_identity_ok = not any_invalid_identity and not reused_code_violation
    deterministic_reconstruction_ok = reconstruction_consistent and not any_invalid_identity
    effective_date_ok = not any_ambiguous_date

    return {
        "listing_transition_pass": bool(listing_ok),
        "delisting_transition_pass": bool(delisting_ok),
        "market_transition_pass": bool(market_ok),
        "security_type_pass": bool(security_type_ok),
        "canonical_identity_pass": bool(canonical_identity_ok),
        "effective_date_pass": bool(effective_date_ok),
        "deterministic_reconstruction_pass": bool(deterministic_reconstruction_ok),
        "reasons": tuple(sorted(reasons)),
        "reconstructed_identity_count": len(identities),
        "canonical_state": canonical_state,
    }


__all__ = [
    "AMBIGUOUS_EFFECTIVE_DATE",
    "AMBIGUOUS_REUSED_SECURITY_CODE",
    "CONFLICTING_TRANSITION_EVIDENCE",
    "DIMENSION_LISTED_STATE",
    "DIMENSION_MARKET_STATE",
    "DIMENSION_SECURITY_TYPE_STATE",
    "INVALID_CANONICAL_CODE",
    "INVALID_SEMANTIC_EVENT",
    "SECURITY_TYPE_ELIGIBLE",
    "SECURITY_TYPE_INELIGIBLE",
    "SECURITY_TYPE_UNKNOWN",
    "SemanticEvent",
    "SemanticValidationError",
    "TerminalIdentityState",
    "VALID_DIMENSIONS",
    "VALID_SECURITY_TYPE_STATES",
    "canonical_bytes",
    "compute_semantic_validation_result",
    "normalize_canonical_code",
    "validate_canonical_code",
]
