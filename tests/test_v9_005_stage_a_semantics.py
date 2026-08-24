from __future__ import annotations

import pytest

import src.v9_005_stage_a_semantics as sem

TIS = sem.TerminalIdentityState
E = sem.SemanticEvent


# --- 1. Canonical code grammar (methodology #1) ------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1301", "1301"),
        (" 1301 ", "1301"),
        ("1301.0", "1301"),
        ("130a", "130A"),
        ("130A", "130A"),
        ("1a30", "1A30"),
    ],
)
def test_valid_numeric_and_alphanumeric_canonical_codes(raw: str, expected: str) -> None:
    assert sem.validate_canonical_code(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "13B0",  # 'B' is not in the allowed alt-character set
        "13E0",
        "13I0",
        "13O0",
        "13Q0",
        "13V0",
        "13Z0",
    ],
)
def test_illegal_letter_rejected(raw: str) -> None:
    with pytest.raises(sem.SemanticValidationError) as excinfo:
        sem.validate_canonical_code(raw)
    assert excinfo.value.reason == sem.INVALID_CANONICAL_CODE


@pytest.mark.parametrize(
    "raw",
    [
        "A301",  # position 1 must be a digit
        "13A0",  # position 3 must be a digit
        "A30A",  # both position 1 and 3 invalid
    ],
)
def test_illegal_position_rejected(raw: str) -> None:
    with pytest.raises(sem.SemanticValidationError) as excinfo:
        sem.validate_canonical_code(raw)
    assert excinfo.value.reason == sem.INVALID_CANONICAL_CODE


@pytest.mark.parametrize("raw", ["13010", "1301A5", "1", "", "13-1"])
def test_five_character_and_malformed_codes_rejected(raw: str) -> None:
    """No 5-character code (e.g. one using a reserved fifth security-type
    character) is ever accepted as the canonical ordinary-common identity."""
    with pytest.raises(sem.SemanticValidationError) as excinfo:
        sem.validate_canonical_code(raw)
    assert excinfo.value.reason == sem.INVALID_CANONICAL_CODE


def test_normalize_only_strips_unambiguous_artifacts() -> None:
    assert sem.normalize_canonical_code("1301.0") == "1301"
    assert sem.normalize_canonical_code("  130a  ") == "130A"
    # A non-integral float-looking artifact is not silently coerced -- it
    # is left for grammar validation to reject.
    assert sem.normalize_canonical_code("1301.5") == "1301.5"


# --- Fixtures -----------------------------------------------------------------

def _single_identity(
    *, listed_state: bool = True, market_state: str = "PRIME", security_type_state: str = sem.SECURITY_TYPE_ELIGIBLE,
) -> dict[str, TIS]:
    return {"1301": TIS(listed_state=listed_state, market_state=market_state, security_type_state=security_type_state)}


# --- 2. Reused codes ----------------------------------------------------------

def test_reused_code_disjoint_episodes_fails() -> None:
    identities = _single_identity(listed_state=True)
    events = [
        E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2"),
        E("1301", "2018-06-01", sem.DIMENSION_LISTED_STATE, True, False, "F3"),
        E("1301", "2019-01-01", sem.DIMENSION_LISTED_STATE, False, True, "F2"),
    ]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["canonical_identity_pass"] is False
    assert sem.AMBIGUOUS_REUSED_SECURITY_CODE in result["reasons"]


def test_single_listing_episode_is_not_a_reuse_violation() -> None:
    identities = _single_identity(listed_state=True)
    events = [E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2")]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["canonical_identity_pass"] is True


def test_continuous_listing_with_name_change_is_not_reuse() -> None:
    """A mere company-name change while continuously listed is out of
    scope for this engine (it never consumes company names at all) -- a
    single, uninterrupted listing episode with no LISTED_STATE events
    other than the original listing must not be flagged as reused."""
    identities = _single_identity(listed_state=True)
    events = [
        E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2"),
        E("1301", "2020-06-01", sem.DIMENSION_MARKET_STATE, "FIRST_SECTION", "PRIME", "F2"),
    ]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["canonical_identity_pass"] is True


# --- 3 & 5. Point-in-time state / transition evidence -------------------------

def test_simple_listing_and_delisting_chronology() -> None:
    identities = {"1301": TIS(listed_state=False, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE)}
    events = [
        E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2"),
        E("1301", "2020-06-01", sem.DIMENSION_LISTED_STATE, True, False, "F3"),
    ]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["listing_transition_pass"] is True
    assert result["delisting_transition_pass"] is True
    assert result["deterministic_reconstruction_pass"] is True
    assert result["canonical_identity_pass"] is True


def test_market_transition_requires_exact_effective_date() -> None:
    identities = _single_identity(market_state="PRIME")
    events = [E("1301", "2022-04-04", sem.DIMENSION_MARKET_STATE, "FIRST_SECTION", "PRIME", "F2")]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["market_transition_pass"] is True
    assert result["deterministic_reconstruction_pass"] is True


def test_market_transition_pass_is_not_a_proxy_for_listing_transition_pass() -> None:
    """A malformed LISTED_STATE event must not drag market_transition_pass
    down with it, and vice versa -- the two must be computed
    independently, never merely equal by definition."""
    identities = _single_identity(listed_state=True, market_state="PRIME")
    bad_market_event = E("1301", "2022-04-04", sem.DIMENSION_MARKET_STATE, "FIRST_SECTION", "", "F2")  # empty after_state invalid
    good_listing_event = E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2")
    result = sem.compute_semantic_validation_result(
        terminal_identities=identities, events=[bad_market_event, good_listing_event],
    )
    assert result["market_transition_pass"] is False
    assert result["listing_transition_pass"] is True


def test_f2_f3_identical_corroboration_accepted() -> None:
    identities = {"1301": TIS(listed_state=False, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE)}
    events = [
        E("1301", "2020-06-01", sem.DIMENSION_LISTED_STATE, True, False, "F2"),
        E("1301", "2020-06-01", sem.DIMENSION_LISTED_STATE, True, False, "F3"),
    ]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["delisting_transition_pass"] is True
    assert sem.CONFLICTING_TRANSITION_EVIDENCE not in result["reasons"]


def test_f2_f3_conflict_fails() -> None:
    identities = _single_identity(listed_state=True, market_state="PRIME")
    events = [
        E("1301", "2020-06-01", sem.DIMENSION_MARKET_STATE, "FIRST_SECTION", "PRIME", "F2"),
        E("1301", "2020-06-01", sem.DIMENSION_MARKET_STATE, "FIRST_SECTION", "STANDARD", "F3"),
    ]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["market_transition_pass"] is False
    assert sem.CONFLICTING_TRANSITION_EVIDENCE in result["reasons"]


def test_same_date_different_dimension_events_are_deterministic() -> None:
    """Different dimensions on the same date are simultaneous and need no
    cross-dimension ordering: the result must be identical regardless of
    input event order."""
    identities = _single_identity(listed_state=True, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE)
    events = [
        E("1301", "2020-01-06", sem.DIMENSION_MARKET_STATE, "FIRST_SECTION", "PRIME", "F2"),
        E("1301", "2019-01-01", sem.DIMENSION_LISTED_STATE, False, True, "F2"),
    ]
    forward = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    backward = sem.compute_semantic_validation_result(terminal_identities=identities, events=list(reversed(events)))
    assert sem.canonical_bytes(forward) == sem.canonical_bytes(backward)
    assert forward["market_transition_pass"] is True
    assert forward["listing_transition_pass"] is True


def test_ambiguous_effective_date_fails() -> None:
    identities = _single_identity(listed_state=True)
    events = [E("1301", "not-a-date", sem.DIMENSION_LISTED_STATE, False, True, "F2")]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["effective_date_pass"] is False
    assert result["listing_transition_pass"] is False
    assert sem.AMBIGUOUS_EFFECTIVE_DATE in result["reasons"]


def test_impossible_calendar_date_is_ambiguous() -> None:
    identities = _single_identity(listed_state=True)
    events = [E("1301", "2021-02-30", sem.DIMENSION_LISTED_STATE, False, True, "F2")]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["effective_date_pass"] is False


# --- 4. Security type ----------------------------------------------------------

def test_unknown_security_type_while_listed_fails() -> None:
    identities = _single_identity(listed_state=True, security_type_state=sem.SECURITY_TYPE_UNKNOWN)
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=[])
    assert result["security_type_pass"] is False


def test_unknown_security_type_while_never_listed_does_not_fail() -> None:
    identities = _single_identity(listed_state=False, security_type_state=sem.SECURITY_TYPE_UNKNOWN)
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=[])
    assert result["security_type_pass"] is True


def test_eligible_security_type_while_listed_passes() -> None:
    identities = _single_identity(listed_state=True, security_type_state=sem.SECURITY_TYPE_ELIGIBLE)
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=[])
    assert result["security_type_pass"] is True


def test_explicitly_ineligible_is_not_unknown_and_does_not_fail_that_gate() -> None:
    identities = _single_identity(listed_state=True, security_type_state=sem.SECURITY_TYPE_INELIGIBLE)
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=[])
    assert result["security_type_pass"] is True


# --- 7. Deterministic reconstruction (reverse/forward) -------------------------

def test_reverse_forward_terminal_byte_equality_passes() -> None:
    identities = {"1301": TIS(listed_state=True, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE)}
    events = [
        E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2"),
        E("1301", "2019-04-01", sem.DIMENSION_MARKET_STATE, "FIRST_SECTION", "PRIME", "F2"),
    ]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["deterministic_reconstruction_pass"] is True
    assert result["canonical_state"]["1301"] == {
        "listed_state": True, "market_state": "PRIME", "security_type_state": sem.SECURITY_TYPE_ELIGIBLE,
    }


def test_tampered_terminal_state_fails_reverse_forward_check() -> None:
    """The declared terminal state disagrees with what the event chain
    actually implies -- no reconciliation, this must fail closed."""
    identities = {"1301": TIS(listed_state=False, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE)}
    events = [E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2")]  # implies listed=True at terminal
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["deterministic_reconstruction_pass"] is False


def test_inconsistent_chained_events_fail_reverse_forward_check() -> None:
    """Two listing events back-to-back with no intervening delisting is an
    internally inconsistent chain."""
    identities = {"1301": TIS(listed_state=True, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE)}
    events = [
        E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2"),
        E("1301", "2018-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2"),
    ]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["deterministic_reconstruction_pass"] is False
    assert result["canonical_identity_pass"] is False


def test_repeated_reconstruction_is_byte_identical() -> None:
    identities = {"1301": TIS(listed_state=True, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE)}
    events = [E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2")]
    first = sem.canonical_bytes(sem.compute_semantic_validation_result(terminal_identities=identities, events=events))
    second = sem.canonical_bytes(sem.compute_semantic_validation_result(terminal_identities=identities, events=events))
    assert first == second


# --- Production fail-closed default ("no evidence acquired") -------------------

def test_no_terminal_identities_fails_closed_not_vacuous_pass() -> None:
    """Dummy/empty evidence must never make any semantic gate PASS -- this
    is the production seam used while F2-F7 acquisition is not yet
    implemented."""
    result = sem.compute_semantic_validation_result(terminal_identities={}, events=())
    for key in (
        "listing_transition_pass", "delisting_transition_pass", "market_transition_pass",
        "security_type_pass", "canonical_identity_pass", "effective_date_pass",
        "deterministic_reconstruction_pass",
    ):
        assert result[key] is False, key
    assert result["reconstructed_identity_count"] == 0


def test_events_with_no_terminal_identities_still_fails_closed() -> None:
    """Events alone (with no declared terminal identity) must not smuggle
    a pass through -- the empty-terminal-identities fail-closed default
    applies regardless of what `events` contains."""
    events = [E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2")]
    result = sem.compute_semantic_validation_result(terminal_identities={}, events=events)
    assert result["listing_transition_pass"] is False
    assert result["canonical_identity_pass"] is False


# --- Invalid identity / invalid event structure --------------------------------

def test_invalid_terminal_identity_code_fails_canonical_identity_pass() -> None:
    identities = {"13B0": TIS(listed_state=True, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE)}
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=())
    assert result["canonical_identity_pass"] is False
    assert sem.INVALID_CANONICAL_CODE in result["reasons"]


def test_malformed_listed_state_event_before_equals_after_is_invalid() -> None:
    identities = _single_identity(listed_state=True)
    events = [E("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, True, True, "F2")]
    result = sem.compute_semantic_validation_result(terminal_identities=identities, events=events)
    assert result["listing_transition_pass"] is False
    assert result["delisting_transition_pass"] is False
