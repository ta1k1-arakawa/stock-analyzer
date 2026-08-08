from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.check_v7_activation_manifest import (
    CALENDAR_PATH,
    SEED_CUTOFF,
    SYNTHETIC_ACQUISITION_WINDOW_JST,
    SYNTHETIC_AUTHORIZATION_UTC,
    SYNTHETIC_SEED_ACQUISITION_UTC,
    UNIVERSE_CSV,
    build_synthetic_fixture,
    seed_observation_days,
    synthetic_candidate,
    synthetic_seed_rows,
    universe_tickers,
    write_synthetic_seed_csv,
)
from src import v7_activation_manifest as manifest_module
from src.v7_activation_manifest import (
    ARM_PARITY_FIELDS,
    HUMAN_ACTIVATION_CONFIRMATION,
    MANIFEST_FIELDS,
    PRODUCTION_SEED_PROVENANCE,
    PROHIBITION_FIELDS,
    SHARED_RULES,
    SeedProvenanceExpectation,
    V7ActivationManifestBlocked,
    build_activation_manifest_candidate,
    canonical_sha256,
    compute_manifest_sha256,
    expected_activation_boundary,
    validate_acquisition_window,
    validate_activation_manifest_candidate,
    validate_authorization_and_boundary,
    validate_calendar_binding,
    validate_output_root,
    validate_seed_provenance,
)
from src.v7_jpx_calendar import load_calendar_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = "2026-08-10"


@pytest.fixture(scope="module")
def calendar_snapshot():
    return load_calendar_snapshot(json.loads(CALENDAR_PATH.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    """Synthetic Gate 4 inputs built once; nothing here is a study decision."""
    workspace = tmp_path_factory.mktemp("v7-activation-fixture")
    return build_synthetic_fixture(workspace)


@pytest.fixture(scope="module")
def candidate(fixture):
    return synthetic_candidate(fixture)


def full_validate(fixture, manifest, **overrides) -> dict[str, Any]:
    kwargs = dict(
        repository_root=fixture["repository_root"],
        calendar_path=CALENDAR_PATH,
        universe_csv=UNIVERSE_CSV,
        seed_csv=fixture["seed_csv"],
        seed_acquisition_manifest=fixture["seed_acquisition_manifest"],
        expected_seed_provenance=fixture["seed_expectation"],
    )
    kwargs.update(overrides)
    return validate_activation_manifest_candidate(manifest, **kwargs)


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------


def test_candidate_has_exact_schema(candidate):
    assert set(candidate) == set(MANIFEST_FIELDS)


def test_candidate_field_count_is_stable(candidate):
    assert len(MANIFEST_FIELDS) == len(set(MANIFEST_FIELDS))
    assert len(candidate) == len(MANIFEST_FIELDS)


def test_unknown_field_blocked(fixture, candidate):
    tampered = {**candidate, "unexpected_field": 1}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason.startswith("MANIFEST_UNKNOWN_FIELD:")


def test_missing_field_blocked(fixture, candidate):
    tampered = {key: value for key, value in candidate.items() if key != "output_root"}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason.startswith("MANIFEST_MISSING_FIELD:")


def test_non_mapping_manifest_blocked(fixture):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, ["not", "a", "mapping"])
    assert excinfo.value.reason == "MANIFEST_INVALID"


# ---------------------------------------------------------------------------
# Frozen identity
# ---------------------------------------------------------------------------


FROZEN_EXPECTATIONS = {
    "schema_version": "V7_FORWARD_ACTIVATION_V1",
    "mode": "FORWARD_ONLY_EXPLORATORY_PAPER_STUDY",
    "study_name": "V7_FORWARD_CAPACITY",
    "activation_status": "ACTIVATED",
    "design_commit": "e3e1367efd913b601a70328a815d88c20af6d147",
    "preregistration_utc": "2026-08-07T02:48:27Z",
    "implementation_commit": "98b328ec905837fc1c7dfca91950529c573bc5db",
    "collector_commit": "4ca41c53895e75910ae65809fea6018868929afa",
    "calendar_commit": "03ce048b0eedca632f79ad925a627cb9e967d78d",
    "seed_generation_commit": "0facf819c14e681036d2a081db0a5208c14b7cf9",
    "calendar_source": "JPX_OFFICIAL_MARKET_HOLIDAYS",
    "calendar_timezone": "Asia/Tokyo",
    "calendar_definition_version": "V7_JPX_OFFICIAL_MARKET_HOLIDAYS_V1",
    "calendar_snapshot_sha256": "6114094de84f9f9833ceddaa9fb4a46290662423f425b3e24be1b60eb00968a0",
    "data_source": "Yahoo Chart",
    "data_source_host": "query1.finance.yahoo.com",
    "data_source_schema": "V7_YAHOO_CHART_DAILY_RAW_OHLCV_V1",
    "seed_data_source": "Yahoo Chart",
    "seed_data_schema": "V7_YAHOO_CHART_DAILY_RAW_OHLCV_V1",
    "universe_csv_sha256": "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997",
    "ticker_list_sha256": "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7",
    "ticker_count": 300,
    "arm_a_parameters_sha256": "0ace638e6c40a222cd5b4ca107ddf6012c1f4e40e45dd45d49f37a1673b71b41",
    "arm_b_parameters_sha256": "d505d325d1c573595b9af26e141564f69a6ac8efdb8e6388d7eb61d50440a779",
    "single_changed_parameter": "max_open_positions",
    "seed_validation_result": "PASS",
}


@pytest.mark.parametrize("field,expected", sorted(FROZEN_EXPECTATIONS.items()))
def test_frozen_identity_value(candidate, field, expected):
    assert candidate[field] == expected


@pytest.mark.parametrize(
    "field", sorted(set(FROZEN_EXPECTATIONS) - {"calendar_snapshot_sha256"})
)
def test_frozen_identity_tamper_blocked(fixture, candidate, field):
    tampered = {**candidate, field: "TAMPERED" if isinstance(candidate[field], str) else 999}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason.startswith("FROZEN_FIELD_MISMATCH:")


@pytest.mark.parametrize("field,expected", sorted(PROHIBITION_FIELDS.items()))
def test_historical_prohibition_booleans(candidate, field, expected):
    assert candidate[field] is expected


@pytest.mark.parametrize("field", sorted(PROHIBITION_FIELDS))
def test_prohibition_flip_blocked(fixture, candidate, field):
    tampered = {**candidate, field: not PROHIBITION_FIELDS[field]}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason.startswith("PROHIBITION_FIELD_INVALID:")


def test_survivorship_bias_is_true(candidate):
    assert candidate["survivorship_bias"] is True


@pytest.mark.parametrize("field", ARM_PARITY_FIELDS)
def test_arm_parity_fields_true(candidate, field):
    assert candidate[field] is True


@pytest.mark.parametrize("field", ARM_PARITY_FIELDS)
def test_arm_parity_false_blocked(fixture, candidate, field):
    tampered = {**candidate, field: False}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason.startswith("ARM_PARITY_FIELD_INVALID:")


def test_arm_parameter_hashes_match_engine():
    from src.v7_capacity_engine import V7EngineParameters

    assert manifest_module.ARM_A_PARAMETERS_SHA256 == V7EngineParameters.control().sha256()
    assert manifest_module.ARM_B_PARAMETERS_SHA256 == V7EngineParameters.capacity_3().sha256()


def test_single_changed_parameter_is_max_open_positions(candidate):
    from src.v7_capacity_engine import V7EngineParameters, validate_single_parameter_difference

    assert candidate["single_changed_parameter"] == "max_open_positions"
    assert validate_single_parameter_difference(
        V7EngineParameters.control(), V7EngineParameters.capacity_3()
    ) is True


# ---------------------------------------------------------------------------
# Shared rules hash
# ---------------------------------------------------------------------------


def test_shared_rules_hash_is_deterministic():
    assert manifest_module.shared_rules_sha256() == manifest_module.shared_rules_sha256()
    assert manifest_module.SHARED_RULES_SHA256 == canonical_sha256(SHARED_RULES)


def test_shared_rules_hash_covers_exact_declared_fields():
    assert set(SHARED_RULES) == {
        "starting_cash", "quantity", "cash_reserve", "capital_limit_per_position",
        "same_industry_concurrent", "duplicate_ticker_concurrent", "same_day_proceeds_reuse",
        "entry_source", "entry_gap_multiplier", "entry_slippage",
        "exit_source", "exit_slippage", "exit_reason", "stop_loss",
        "candidate_rules", "market_gate", "ranking_rules", "top_candidates_per_signal_day",
    }


def test_shared_rules_values_match_frozen_study_rules():
    assert SHARED_RULES["starting_cash"] == 400000
    assert SHARED_RULES["quantity"] == 100
    assert SHARED_RULES["cash_reserve"] == 40000
    assert SHARED_RULES["capital_limit_per_position"] == 220000
    assert SHARED_RULES["entry_source"] == "D1_RAW_OPEN"
    assert SHARED_RULES["exit_source"] == "D10_RAW_OPEN"
    assert SHARED_RULES["stop_loss"] == "NONE"
    assert SHARED_RULES["top_candidates_per_signal_day"] == 20


def test_shared_rules_hash_tamper_blocked(fixture, candidate):
    tampered = {**candidate, "shared_rules_sha256": "0" * 64}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason.startswith("FROZEN_FIELD_MISMATCH:")


# ---------------------------------------------------------------------------
# Calendar binding
# ---------------------------------------------------------------------------


def test_calendar_raw_sha_matches_repository_file():
    result = validate_calendar_binding(CALENDAR_PATH)
    assert result["calendar_snapshot_sha256"] == manifest_module.CALENDAR_SNAPSHOT_SHA256


def test_calendar_wrong_expected_sha_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_calendar_binding(CALENDAR_PATH, expected_snapshot_sha256="1" * 64)
    assert excinfo.value.reason == "CALENDAR_SNAPSHOT_SHA_MISMATCH"


def test_calendar_missing_file_blocked(tmp_path):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_calendar_binding(tmp_path / "absent.json")
    assert excinfo.value.reason == "CALENDAR_FILE_READ_FAILED"


def test_calendar_altered_file_blocked(tmp_path):
    altered = tmp_path / "calendar.json"
    altered.write_bytes(CALENDAR_PATH.read_bytes() + b"\n")
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_calendar_binding(altered)
    assert excinfo.value.reason == "CALENDAR_SNAPSHOT_SHA_MISMATCH"


def test_calendar_identity_fields_present():
    payload = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    assert payload["calendar_source"] == "JPX_OFFICIAL_MARKET_HOLIDAYS"
    assert payload["calendar_timezone"] == "Asia/Tokyo"
    assert payload["calendar_definition_version"] == "V7_JPX_OFFICIAL_MARKET_HOLIDAYS_V1"


def test_calendar_snapshot_loads_via_accepted_loader():
    result = validate_calendar_binding(CALENDAR_PATH)
    assert result["snapshot"].coverage_start == date(2026, 1, 1)
    assert result["snapshot"].coverage_end == date(2027, 12, 31)


def test_manifest_calendar_hash_tamper_blocked(fixture, candidate):
    tampered = {**candidate, "calendar_snapshot_sha256": "2" * 64}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason == "CALENDAR_SNAPSHOT_SHA_MISMATCH"


# ---------------------------------------------------------------------------
# Authorization UTC
# ---------------------------------------------------------------------------


def _auth(calendar_snapshot, authorization, boundary=BOUNDARY, seed_utc=SYNTHETIC_SEED_ACQUISITION_UTC):
    return validate_authorization_and_boundary(
        snapshot=calendar_snapshot,
        activation_authorization_utc=authorization,
        activation_boundary_first_jpx_trading_date=boundary,
        seed_acquisition_utc=seed_utc,
    )


def test_authorization_aware_utc_pass(calendar_snapshot):
    result = _auth(calendar_snapshot, SYNTHETIC_AUTHORIZATION_UTC)
    assert result["expected_activation_boundary"] == BOUNDARY


def test_authorization_naive_timestamp_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, "2026-08-07T09:00:00")
    assert excinfo.value.reason.startswith("UTC_TIMESTAMP_INVALID:")


def test_authorization_non_utc_offset_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, "2026-08-07T18:00:00+09:00")
    assert excinfo.value.reason.startswith("UTC_TIMESTAMP_INVALID:")


def test_authorization_before_preregistration_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, "2026-08-06T09:00:00Z", seed_utc="2026-08-05T00:00:00Z")
    assert excinfo.value.reason == "AUTHORIZATION_NOT_AFTER_PREREGISTRATION"


def test_authorization_equal_to_preregistration_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, manifest_module.PREREGISTRATION_UTC, seed_utc="2026-08-01T00:00:00Z")
    assert excinfo.value.reason == "AUTHORIZATION_NOT_AFTER_PREREGISTRATION"


def test_authorization_before_seed_acquisition_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, "2026-08-07T03:00:00Z", seed_utc="2026-08-07T05:00:00Z")
    assert excinfo.value.reason == "AUTHORIZATION_NOT_AFTER_SEED_ACQUISITION"


def test_authorization_equal_to_seed_acquisition_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, "2026-08-07T09:00:00Z", seed_utc="2026-08-07T09:00:00Z")
    assert excinfo.value.reason == "AUTHORIZATION_NOT_AFTER_SEED_ACQUISITION"


# ---------------------------------------------------------------------------
# Activation boundary
# ---------------------------------------------------------------------------


def test_boundary_is_first_jpx_trading_day_after_authorization_jst_date(calendar_snapshot):
    assert expected_activation_boundary(calendar_snapshot, "2026-08-07T09:00:00Z") == "2026-08-10"


def test_boundary_rolls_over_when_authorization_is_late_jst(calendar_snapshot):
    # 2026-08-07T23:30Z is 2026-08-08 08:30 JST (Saturday) -> Monday 2026-08-10
    assert expected_activation_boundary(calendar_snapshot, "2026-08-07T23:30:00Z") == "2026-08-10"


def test_boundary_same_jst_day_as_authorization_blocked(calendar_snapshot):
    # 2026-08-10T01:00Z is 2026-08-10 10:00 JST; same-day activation is prohibited
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, "2026-08-10T01:00:00Z", boundary="2026-08-10")
    assert excinfo.value.reason == "ACTIVATION_BOUNDARY_NOT_AFTER_AUTHORIZATION_JST_DATE"


def test_boundary_not_first_next_jpx_trading_day_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, SYNTHETIC_AUTHORIZATION_UTC, boundary="2026-08-12")
    assert excinfo.value.reason == "ACTIVATION_BOUNDARY_NOT_FIRST_JPX_TRADING_DAY_AFTER_AUTHORIZATION"


def test_boundary_on_weekend_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, SYNTHETIC_AUTHORIZATION_UTC, boundary="2026-08-15")
    assert excinfo.value.reason == "ACTIVATION_BOUNDARY_NOT_JPX_TRADING_DAY"


def test_boundary_on_market_holiday_blocked(calendar_snapshot):
    # 2026-08-11 is Mountain Day, an official JPX market holiday.
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, SYNTHETIC_AUTHORIZATION_UTC, boundary="2026-08-11")
    assert excinfo.value.reason == "ACTIVATION_BOUNDARY_NOT_JPX_TRADING_DAY"


def test_boundary_outside_calendar_coverage_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, SYNTHETIC_AUTHORIZATION_UTC, boundary="2028-01-04")
    assert excinfo.value.reason == "ACTIVATION_BOUNDARY_OUTSIDE_CALENDAR_COVERAGE"


def test_boundary_invalid_date_blocked(calendar_snapshot):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        _auth(calendar_snapshot, SYNTHETIC_AUTHORIZATION_UTC, boundary="2026-8-10")
    assert excinfo.value.reason.startswith("INVALID_DATE:")


# ---------------------------------------------------------------------------
# Acquisition window
# ---------------------------------------------------------------------------


def test_acquisition_window_valid():
    assert validate_acquisition_window("17:00-18:00 Asia/Tokyo")["start_minutes"] == 17 * 60


def test_acquisition_window_market_close_boundary_allowed():
    assert validate_acquisition_window("15:30-16:00 Asia/Tokyo")["start_minutes"] == 15 * 60 + 30


@pytest.mark.parametrize("value", [
    "17:00-18:00",
    "17:00-18:00 UTC",
    "17:00-18:00 Asia/Seoul",
    "1700-1800 Asia/Tokyo",
    "17:00 - 18:00 Asia/Tokyo",
    "7:00-18:00 Asia/Tokyo",
])
def test_acquisition_window_syntax_blocked(value):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_acquisition_window(value)
    assert excinfo.value.reason == "ACQUISITION_WINDOW_SYNTAX_INVALID"


def test_acquisition_window_before_market_close_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_acquisition_window("14:00-15:00 Asia/Tokyo")
    assert excinfo.value.reason == "ACQUISITION_WINDOW_BEFORE_MARKET_CLOSE"


def test_acquisition_window_reversed_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_acquisition_window("18:00-17:00 Asia/Tokyo")
    assert excinfo.value.reason == "ACQUISITION_WINDOW_ORDER_INVALID"


def test_acquisition_window_equal_start_end_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_acquisition_window("17:00-17:00 Asia/Tokyo")
    assert excinfo.value.reason == "ACQUISITION_WINDOW_ORDER_INVALID"


def test_acquisition_window_non_string_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_acquisition_window(1700)
    assert excinfo.value.reason == "ACQUISITION_WINDOW_INVALID"


def test_manifest_acquisition_window_tamper_blocked(fixture, candidate):
    tampered = {**candidate, "acquisition_window_jst": "09:00-10:00 Asia/Tokyo"}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason == "ACQUISITION_WINDOW_BEFORE_MARKET_CLOSE"


# ---------------------------------------------------------------------------
# Output root
# ---------------------------------------------------------------------------


def test_output_root_absolute_outside_repo_pass(tmp_path):
    assert validate_output_root(str(tmp_path.resolve()), REPO_ROOT) == str(tmp_path.resolve())


def test_output_root_relative_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root("study/output", REPO_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_NOT_ABSOLUTE"


def test_output_root_empty_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root("", REPO_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_INVALID"


def test_output_root_equal_to_repository_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root(str(REPO_ROOT.resolve()), REPO_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


def test_output_root_inside_repository_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root(str((REPO_ROOT / "data" / "study").resolve()), REPO_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


def test_output_root_windows_absolute_allowed():
    assert validate_output_root(r"C:\v7-study\output", REPO_ROOT) == r"C:\v7-study\output"


def test_output_root_windows_relative_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root(r"v7-study\output", REPO_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_NOT_ABSOLUTE"


def test_output_root_uri_allowed():
    assert validate_output_root("s3://v7-forward-study/root", REPO_ROOT).startswith("s3://")


def test_output_root_file_uri_inside_repository_blocked():
    value = "file://" + (REPO_ROOT / "data").resolve().as_posix()
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root(value, REPO_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


# ---------------------------------------------------------------------------
# Windows file:// URI containment (cross-platform: exercised on any host OS,
# since PureWindowsPath never touches the real filesystem)
# ---------------------------------------------------------------------------


WINDOWS_REPOSITORY_ROOT = r"C:\repo"


@pytest.mark.parametrize("value", [
    r"C:\repo\data",
    "file:///C:/repo",
    "file:///C:/repo/data",
    "file://C:/repo",
    "file://C:/repo/data",
    "file:///c:/repo/data",
])
def test_windows_file_uri_inside_repository_blocked(value):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root(value, WINDOWS_REPOSITORY_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


def test_windows_file_uri_exact_repository_root_blocked():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root("file:///C:/repo", WINDOWS_REPOSITORY_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


@pytest.mark.parametrize("value", [
    r"D:\v7-study",
    "file:///D:/v7-study",
    "file://D:/v7-study",
])
def test_windows_file_uri_outside_repository_passes(value):
    assert validate_output_root(value, WINDOWS_REPOSITORY_ROOT) == value


def test_windows_file_uri_sibling_directory_not_blocked():
    # "C:\repository-archive" is not inside "C:\repo" despite sharing a prefix.
    assert (
        validate_output_root(r"C:\repository-archive\output", WINDOWS_REPOSITORY_ROOT)
        == r"C:\repository-archive\output"
    )


def test_windows_drive_letter_case_does_not_bypass_containment():
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root("file:///c:/repo/data", r"c:\repo")
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root("file:///C:/repo/data", r"c:\repo")
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


def test_posix_file_uri_behavior_unchanged_by_windows_fix():
    value = "file:///repo/data"
    assert validate_output_root(value, "/elsewhere") == value
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root(value, "/repo")
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


POSIX_REPOSITORY_ROOT = "/repo"


@pytest.mark.parametrize("value", [
    "file:///repo",
    "file:///repo/data",
    "/repo/data",
])
def test_posix_containment_is_host_independent_blocked(value):
    """These must BLOCK identically whether this validator runs on a POSIX
    or a Windows host -- the whole point of this fix is that repository_root
    is compared by its own declared syntax, not by feeding it through the
    *host's* native path resolver."""
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root(value, POSIX_REPOSITORY_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


@pytest.mark.parametrize("value", [
    "file:///elsewhere",
    "/elsewhere",
])
def test_posix_containment_is_host_independent_passes(value):
    assert validate_output_root(value, POSIX_REPOSITORY_ROOT) == value


def test_posix_string_repository_root_is_not_resolved_against_real_cwd():
    """A synthetic POSIX-style repository_root string must be compared
    lexically, not turned into a host-native resolved path -- otherwise a
    string like "/repo" silently becomes e.g. "C:\\repo" on Windows or an
    unrelated cwd-relative path, defeating containment entirely."""
    from src.v7_activation_manifest import _explicit_repository_root_flavor
    from pathlib import PurePosixPath

    assert _explicit_repository_root_flavor("/repo") == (PurePosixPath, "/repo")


def test_real_path_repository_root_still_uses_host_resolution(tmp_path):
    """A genuine pathlib.Path (the normal caller shape) keeps the original
    resolve()-based safety -- only explicit foreign-flavor strings bypass it."""
    nested = tmp_path / "nested"
    nested.mkdir()
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_output_root(str((nested / "study").resolve()), tmp_path)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


def test_non_file_uri_still_passes_through_unchanged():
    assert validate_output_root("s3://v7-forward-study/root", WINDOWS_REPOSITORY_ROOT) == (
        "s3://v7-forward-study/root"
    )


def test_manifest_windows_output_root_inside_repo_blocked(fixture, candidate):
    tampered = {**candidate, "output_root": r"file://C:/repo/study-output"}
    tampered["manifest_sha256"] = compute_manifest_sha256(tampered)
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered, repository_root=WINDOWS_REPOSITORY_ROOT)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


def test_manifest_output_root_inside_repo_blocked(fixture, candidate):
    tampered = {**candidate, "output_root": str((REPO_ROOT / "tmp-study").resolve())}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"


def test_manifest_output_root_relative_blocked(fixture, candidate):
    tampered = {**candidate, "output_root": "relative/output"}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason == "OUTPUT_ROOT_NOT_ABSOLUTE"


# ---------------------------------------------------------------------------
# Placeholders / unresolved human decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("placeholder", ["NOT_SET", "UNRESOLVED_HUMAN_GATE", "TBD", "TODO", "UNKNOWN", ""])
@pytest.mark.parametrize("field", list(manifest_module.HUMAN_DECISION_FIELDS))
def test_placeholder_human_decision_blocked_at_build(fixture, field, placeholder):
    kwargs = {
        "activation_authorization_utc": SYNTHETIC_AUTHORIZATION_UTC,
        "activation_boundary_first_jpx_trading_date": fixture["activation_boundary"],
        "acquisition_window_jst": SYNTHETIC_ACQUISITION_WINDOW_JST,
        "output_root": fixture["output_root"],
        "seed_acquisition_utc": SYNTHETIC_SEED_ACQUISITION_UTC,
        "seed_provenance": fixture["seed_provenance"],
    }
    kwargs[field] = placeholder
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        build_activation_manifest_candidate(**kwargs)
    assert excinfo.value.reason == "HUMAN_DECISION_UNRESOLVED:" + field


def test_placeholder_blocked_at_validation(fixture, candidate):
    tampered = {**candidate, "output_root": "NOT_SET"}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason == "HUMAN_DECISION_UNRESOLVED:output_root"


def test_none_human_decision_blocked(fixture):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        build_activation_manifest_candidate(
            activation_authorization_utc=SYNTHETIC_AUTHORIZATION_UTC,
            activation_boundary_first_jpx_trading_date=None,
            acquisition_window_jst=SYNTHETIC_ACQUISITION_WINDOW_JST,
            output_root=fixture["output_root"],
            seed_acquisition_utc=SYNTHETIC_SEED_ACQUISITION_UTC,
            seed_provenance=fixture["seed_provenance"],
        )
    assert excinfo.value.reason.startswith("HUMAN_DECISION_UNRESOLVED:")


# ---------------------------------------------------------------------------
# Seed provenance semantics
# ---------------------------------------------------------------------------


def test_seed_source_payload_hash_is_derived_from_acquisition_manifest(fixture):
    from src.v7_activation_manifest import hash_source_payload_manifest

    expected = hash_source_payload_manifest(fixture["seed_acquisition_manifest"]["payload_manifest"])
    assert fixture["seed_provenance"]["seed_source_payload_manifest_sha256"] == expected


def test_seed_ticker_manifest_hash_is_derived_from_selected_rows(fixture):
    from src.v7_forward_protocol import validate_seed_rows
    from src.v7_activation_manifest import read_seed_csv_rows

    rows, _ = read_seed_csv_rows(fixture["seed_csv"])
    validation = validate_seed_rows(rows, universe_tickers(), fixture["activation_boundary"])
    assert fixture["seed_provenance"]["seed_ticker_manifest_sha256"] == validation["seed_payload_manifest_sha256"]


def test_seed_two_hash_semantics_are_distinct(fixture):
    assert (
        fixture["seed_provenance"]["seed_source_payload_manifest_sha256"]
        != fixture["seed_provenance"]["seed_ticker_manifest_sha256"]
    )


def test_seed_hash_swap_blocked(fixture, candidate):
    tampered = {
        **candidate,
        "seed_source_payload_manifest_sha256": candidate["seed_ticker_manifest_sha256"],
        "seed_ticker_manifest_sha256": candidate["seed_source_payload_manifest_sha256"],
    }
    tampered["manifest_sha256"] = compute_manifest_sha256(tampered)
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason == "SEED_FIELD_MISMATCH:seed_source_payload_manifest_sha256"


def test_production_seed_pin_holds_known_actual_values():
    assert PRODUCTION_SEED_PROVENANCE.seed_source_payload_manifest_sha256 == (
        "f71446043ad88e1688069ce1f438b11fa0e5172ca5ab21e96fe679ff1b74043f"
    )
    assert PRODUCTION_SEED_PROVENANCE.seed_ticker_manifest_sha256 == (
        "edd06a02103f36b22552124d73f81f9826f609ea10a327d817ccd2c4281d0eff"
    )
    assert PRODUCTION_SEED_PROVENANCE.seed_canonical_csv_sha256 == (
        "8ac3adde3be58ea62072bb6fd7af242ba8c7c5701df1cc67ca2f3b411cde84d3"
    )
    assert PRODUCTION_SEED_PROVENANCE.seed_ticker_count == 300
    assert PRODUCTION_SEED_PROVENANCE.seed_row_count == 75600
    assert PRODUCTION_SEED_PROVENANCE.seed_cutoff_trading_date == "2026-08-07"


def test_production_seed_pin_is_the_validation_default():
    import inspect

    signature = inspect.signature(validate_activation_manifest_candidate)
    assert signature.parameters["expected_seed_provenance"].default is PRODUCTION_SEED_PROVENANCE


def test_seed_csv_sha_binds_actual_file(fixture):
    from src.v7_activation_manifest import sha256_bytes

    assert fixture["seed_provenance"]["seed_canonical_csv_sha256"] == sha256_bytes(
        Path(fixture["seed_csv"]).read_bytes()
    )


def test_seed_row_and_ticker_counts(fixture):
    assert fixture["seed_provenance"]["seed_ticker_count"] == 300
    assert fixture["seed_provenance"]["seed_row_count"] == 75600


def test_seed_cutoff_matches_last_observation(fixture):
    assert fixture["seed_provenance"]["seed_cutoff_trading_date"] == SEED_CUTOFF


def test_seed_row_on_or_after_boundary_blocked(tmp_path, fixture):
    tickers = universe_tickers()
    days = seed_observation_days(3)
    rows = synthetic_seed_rows(tickers, days)
    rows.append({**rows[0], "trading_date": fixture["activation_boundary"]})
    seed_csv = tmp_path / "seed.csv"
    write_synthetic_seed_csv(seed_csv, rows)
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_seed_provenance(
            universe_csv=UNIVERSE_CSV,
            seed_csv=seed_csv,
            seed_acquisition_manifest=fixture["seed_acquisition_manifest"],
            activation_boundary_first_jpx_trading_date=fixture["activation_boundary"],
            expected=None,
        )
    assert excinfo.value.reason.startswith("SEED_VALIDATION_FAILED:")


def test_seed_expectation_mismatch_blocked(tmp_path, fixture):
    wrong = SeedProvenanceExpectation(
        seed_source_payload_manifest_sha256="3" * 64,
        seed_ticker_manifest_sha256=fixture["seed_expectation"].seed_ticker_manifest_sha256,
        seed_canonical_csv_sha256=fixture["seed_expectation"].seed_canonical_csv_sha256,
        seed_ticker_count=300,
        seed_row_count=75600,
        seed_cutoff_trading_date=SEED_CUTOFF,
    )
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_seed_provenance(
            universe_csv=UNIVERSE_CSV,
            seed_csv=fixture["seed_csv"],
            seed_acquisition_manifest=fixture["seed_acquisition_manifest"],
            activation_boundary_first_jpx_trading_date=fixture["activation_boundary"],
            expected=wrong,
        )
    assert excinfo.value.reason == "SEED_PROVENANCE_MISMATCH:seed_source_payload_manifest_sha256"


def test_seed_acquisition_manifest_without_payload_manifest_blocked(fixture):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_seed_provenance(
            universe_csv=UNIVERSE_CSV,
            seed_csv=fixture["seed_csv"],
            seed_acquisition_manifest={"mode": "PRE_ACTIVATION_SEED_ACQUISITION"},
            activation_boundary_first_jpx_trading_date=fixture["activation_boundary"],
            expected=None,
        )
    assert excinfo.value.reason == "SEED_SOURCE_PAYLOAD_MANIFEST_INVALID"


def test_seed_csv_schema_invalid_blocked(tmp_path, fixture):
    bad = tmp_path / "seed.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        validate_seed_provenance(
            universe_csv=UNIVERSE_CSV,
            seed_csv=bad,
            seed_acquisition_manifest=fixture["seed_acquisition_manifest"],
            activation_boundary_first_jpx_trading_date=fixture["activation_boundary"],
            expected=None,
        )
    assert excinfo.value.reason == "SEED_CSV_SCHEMA_INVALID"


def test_manifest_seed_field_mismatch_blocked(fixture, candidate):
    tampered = {**candidate, "seed_row_count": 75599}
    tampered["manifest_sha256"] = compute_manifest_sha256(tampered)
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason.startswith("SEED_FIELD_MISMATCH:") or excinfo.value.reason.startswith(
        "SEED_PROVENANCE_MISMATCH:"
    )


# ---------------------------------------------------------------------------
# Manifest hash
# ---------------------------------------------------------------------------


def test_manifest_sha_is_deterministic(candidate):
    assert compute_manifest_sha256(candidate) == compute_manifest_sha256(dict(reversed(list(candidate.items()))))


def test_manifest_sha_excludes_itself(candidate):
    body = {key: candidate[key] for key in MANIFEST_FIELDS if key != "manifest_sha256"}
    assert canonical_sha256(body) == candidate["manifest_sha256"]


def test_manifest_sha_tamper_blocked(fixture, candidate):
    tampered = {**candidate, "manifest_sha256": "4" * 64}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason == "MANIFEST_SHA_MISMATCH"


def test_manifest_sha_invalid_format_blocked(fixture, candidate):
    tampered = {**candidate, "manifest_sha256": "not-a-sha"}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        full_validate(fixture, tampered)
    assert excinfo.value.reason == "MANIFEST_SHA_INVALID"


def test_clean_candidate_validates(fixture, candidate):
    result = full_validate(fixture, candidate)
    assert result["status"] == "PASS"
    assert result["manifest_sha256"] == candidate["manifest_sha256"]


# ---------------------------------------------------------------------------
# Candidate builder purity
# ---------------------------------------------------------------------------


def test_candidate_builder_writes_no_file(tmp_path, fixture):
    before = sorted(entry.name for entry in tmp_path.iterdir())
    build_activation_manifest_candidate(
        activation_authorization_utc=SYNTHETIC_AUTHORIZATION_UTC,
        activation_boundary_first_jpx_trading_date=fixture["activation_boundary"],
        acquisition_window_jst=SYNTHETIC_ACQUISITION_WINDOW_JST,
        output_root=str(tmp_path.resolve()),
        seed_acquisition_utc=SYNTHETIC_SEED_ACQUISITION_UTC,
        seed_provenance=fixture["seed_provenance"],
    )
    assert sorted(entry.name for entry in tmp_path.iterdir()) == before


def test_candidate_builder_is_repeatable(fixture, candidate):
    assert synthetic_candidate(fixture) == candidate


def test_candidate_builder_requires_seed_provenance_fields(fixture):
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        build_activation_manifest_candidate(
            activation_authorization_utc=SYNTHETIC_AUTHORIZATION_UTC,
            activation_boundary_first_jpx_trading_date=fixture["activation_boundary"],
            acquisition_window_jst=SYNTHETIC_ACQUISITION_WINDOW_JST,
            output_root=fixture["output_root"],
            seed_acquisition_utc=SYNTHETIC_SEED_ACQUISITION_UTC,
            seed_provenance={"seed_row_count": 75600},
        )
    assert excinfo.value.reason.startswith("SEED_PROVENANCE_MISSING_FIELD:")


def test_validation_creates_no_activation_artifact(tmp_path, fixture, candidate):
    full_validate(fixture, candidate)
    assert list(tmp_path.iterdir()) == []
    assert not (Path(fixture["output_root"]) / "activation_manifest.json").exists()
