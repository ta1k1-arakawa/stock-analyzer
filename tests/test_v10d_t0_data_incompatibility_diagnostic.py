from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from src import v10d_t0_data_incompatibility_diagnostic as diagnostic
from src import v10c_t0_successor_training_input_binding as v10c
from src import v9_009_t0_top1_kill_screen as v9


IMPLEMENTATION_SHA = "a" * 40


def _phase_a() -> diagnostic.PhaseAMetadata:
    return diagnostic.PhaseAMetadata(SimpleNamespace(), IMPLEMENTATION_SHA, ("2018-01-01",))


def _raise(reason: str):
    def raiser(_metadata):
        raise v9.T0DataIncompatible(reason)

    return raiser


def test_phase_a_reuses_v10c_metadata_preflight_without_payload_or_parser(monkeypatch):
    calls = []
    metadata = object()
    monkeypatch.setattr(diagnostic, "_verify_repository_and_freeze", lambda *_: calls.append("repo"))
    monkeypatch.setattr(
        diagnostic.v10a_calendar_bridge,
        "load_fixed_calendar_binding",
        lambda *args: calls.append("calendar") or ["2018-01-01"],
    )
    monkeypatch.setattr(v10c, "phase_a_metadata_preflight", lambda *args: calls.append(args) or metadata)
    monkeypatch.setattr(v10c, "_read_payload_bytes", lambda *_: pytest.fail("payload read"))
    monkeypatch.setattr(v10c, "_parse_payload_file", lambda *_: pytest.fail("parser call"))

    result = diagnostic.phase_a_metadata_preflight(
        tmp_path := __import__("pathlib").Path("synthetic-repo"),
        __import__("pathlib").Path("training"),
        __import__("pathlib").Path("evaluation"),
        __import__("pathlib").Path("universe.csv"),
        IMPLEMENTATION_SHA,
    )

    assert result.successor_metadata is metadata
    assert result.implementation_sha == IMPLEMENTATION_SHA
    assert result.calendar_dates == ("2018-01-01",)
    assert calls[0] == "repo"
    assert calls[1] == "calendar"
    assert len(calls) == 3


def test_phase_a_calendar_binding_failure_is_preflight_and_stops_before_v10c(monkeypatch):
    monkeypatch.setattr(diagnostic, "_verify_repository_and_freeze", lambda *_: None)
    monkeypatch.setattr(
        diagnostic.v10a_calendar_bridge,
        "load_fixed_calendar_binding",
        lambda *_: (_ for _ in ()).throw(RuntimeError("calendar drift")),
    )
    monkeypatch.setattr(v10c, "phase_a_metadata_preflight", lambda *_: pytest.fail("v10c called"))

    with pytest.raises(diagnostic.V10DPreflightFailure, match="CALENDAR"):
        diagnostic.phase_a_metadata_preflight(
            __import__("pathlib").Path("repo"),
            __import__("pathlib").Path("training"),
            __import__("pathlib").Path("evaluation"),
            __import__("pathlib").Path("universe.csv"),
            IMPLEMENTATION_SHA,
        )


def test_inherited_phase_a_failure_remains_preflight_failure(monkeypatch):
    monkeypatch.setattr(diagnostic, "_verify_repository_and_freeze", lambda *_: None)
    monkeypatch.setattr(
        diagnostic.v10a_calendar_bridge,
        "load_fixed_calendar_binding",
        lambda *_: ["2018-01-01"],
    )

    def fail(*_args):
        raise v10c.SuccessorPreflightFailure("metadata")

    monkeypatch.setattr(v10c, "phase_a_metadata_preflight", fail)
    with pytest.raises(diagnostic.V10DPreflightFailure, match="INHERITED"):
        diagnostic.phase_a_metadata_preflight(
            __import__("pathlib").Path("repo"),
            __import__("pathlib").Path("training"),
            __import__("pathlib").Path("evaluation"),
            __import__("pathlib").Path("universe.csv"),
            IMPLEMENTATION_SHA,
        )


def test_diagnostic_requires_boundary_before_loader(monkeypatch):
    monkeypatch.setattr(v10c, "load_successor_cache_pair", lambda *_: pytest.fail("payload loader called"))
    with pytest.raises(diagnostic.V10DPreflightFailure, match="BOUNDARY"):
        diagnostic.run_diagnostic(_phase_a(), authority_boundary_token=None)


@pytest.mark.parametrize(
    ("reason", "stage"),
    [
        ("TRAINING_PAYLOAD_HASH_CLOSURE_INVALID", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("PAYLOAD_PATH_INVALID", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("PAYLOAD_PATH_ESCAPE", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("PAYLOAD_PATH_SYMLINK", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("REQUIRED_FILE_UNAVAILABLE", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("REQUIRED_FILE_UNSAFE", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("CACHE_PAYLOAD_READ_INVALID", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("CACHE_PAYLOAD_HASH_MISMATCH", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("UNIVERSE_READ_INVALID", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("UNIVERSE_IDENTITY_MISMATCH", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("UNIVERSE_TICKER_IDENTITY_MISMATCH", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("CANONICAL_CODE_INVALID", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("CACHE_PAYLOAD_JSON_INVALID", "PARSER_NORMALIZATION_CONTRACT"),
        ("CACHE_PAYLOAD_SCHEMA_INVALID", "PARSER_NORMALIZATION_CONTRACT"),
        ("CACHE_PAYLOAD_LENGTH_INVALID", "PARSER_NORMALIZATION_CONTRACT"),
        ("CACHE_PAYLOAD_DATE_INVALID", "PARSER_NORMALIZATION_CONTRACT"),
        ("OHLCV_FRAME_INVALID", "PARSER_NORMALIZATION_CONTRACT"),
        ("OHLCV_REQUIRED_COLUMNS_MISSING", "PARSER_NORMALIZATION_CONTRACT"),
        ("DUPLICATE_OR_INVALID_PRICE_DATE", "PARSER_NORMALIZATION_CONTRACT"),
        ("NONFINITE_OR_NONPOSITIVE_OHLCV", "PARSER_NORMALIZATION_CONTRACT"),
        ("SPLIT_EVENT_SCHEMA_INVALID", "PARSER_NORMALIZATION_CONTRACT"),
        ("SPLIT_EVENT_RATIO_MISSING", "PARSER_NORMALIZATION_CONTRACT"),
        ("SPLIT_EVENT_RATIO_INVALID", "PARSER_NORMALIZATION_CONTRACT"),
        ("EVALUATION_PAYLOAD_MISSING", "COMBINED_SERIES_CONTRACT"),
        ("COMBINED_PRICE_EMPTY", "COMBINED_SERIES_CONTRACT"),
        ("DUPLICATE_COMBINED_PRICE_DATE", "COMBINED_SERIES_CONTRACT"),
        ("DUPLICATE_COMBINED_SPLIT_EVENT", "COMBINED_SERIES_CONTRACT"),
        ("UNMAPPED_REASON", "UNKNOWN_DATA_INCOMPATIBILITY"),
    ],
)
def test_cache_data_failures_map_only_known_reasons(monkeypatch, reason, stage):
    monkeypatch.setattr(v10c, "load_successor_cache_pair", _raise(reason))
    result = diagnostic.run_diagnostic(
        _phase_a(), authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert result["result_class"] == diagnostic.RESULT_DATA
    assert result["first_failed_stage"] == stage
    assert "reason" not in str(result)


def test_cache_reason_sets_are_pairwise_disjoint_and_exclude_feature_reasons():
    assert diagnostic._INPUT_REASONS.isdisjoint(diagnostic._PARSER_REASONS)
    assert diagnostic._INPUT_REASONS.isdisjoint(diagnostic._COMBINED_REASONS)
    assert diagnostic._PARSER_REASONS.isdisjoint(diagnostic._COMBINED_REASONS)
    assert "FEATURE_HISTORY_UNAVAILABLE" not in diagnostic._INPUT_REASONS
    assert "FEATURE_HISTORY_UNAVAILABLE" not in diagnostic._PARSER_REASONS
    assert "FEATURE_HISTORY_UNAVAILABLE" not in diagnostic._COMBINED_REASONS
    assert diagnostic._stage_for_cache_reason("CACHE_PAYLOAD_HASH_MISMATCH") == (
        "INPUT_BYTE_OR_FILESET_CONTRACT"
    )
    assert diagnostic._stage_for_cache_reason("CACHE_PAYLOAD_SCHEMA_INVALID") == (
        "PARSER_NORMALIZATION_CONTRACT"
    )


def test_dataset_data_failure_maps_to_feature_target_stage(monkeypatch):
    monkeypatch.setattr(v10c, "load_successor_cache_pair", lambda *_: ({}, {}, {}, object()))
    monkeypatch.setattr(v9, "build_dataset", lambda *args: (_ for _ in ()).throw(v9.T0DataIncompatible("DATASET_SCHEMA_INVALID")))
    result = diagnostic.run_diagnostic(
        _phase_a(), authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert result["first_failed_stage"] == "FEATURE_TARGET_DATASET_CONTRACT"


def test_formal_and_post_structural_failures_are_separate(monkeypatch):
    monkeypatch.setattr(v10c, "load_successor_cache_pair", lambda *_: ({}, {}, {}, object()))
    monkeypatch.setattr(v9, "build_dataset", lambda *args: object())
    monkeypatch.setattr(
        diagnostic,
        "_structural_formal_preconditions",
        lambda *args: (_ for _ in ()).throw(v9.T0DataIncompatible("INSUFFICIENT_CAUSAL_TRAINING_DATA")),
    )
    formal_result = diagnostic.run_diagnostic(
        _phase_a(), authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert formal_result["first_failed_stage"] == "FORMAL_SCORING_PRECONDITION_CONTRACT"

    monkeypatch.setattr(diagnostic, "_structural_formal_preconditions", lambda *args: object())
    monkeypatch.setattr(
        diagnostic,
        "_structural_post_scoring_conditions",
        lambda *args: (_ for _ in ()).throw(v9.T0DataIncompatible("FORMAL_TARGET_UNAVAILABLE")),
    )
    post_result = diagnostic.run_diagnostic(
        _phase_a(), authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert post_result["first_failed_stage"] == "POST_SCORING_STRUCTURAL_TARGET_CONTRACT"


@pytest.mark.parametrize("exception", [v9.T0ImplementationFailure("x"), RuntimeError("x")])
def test_implementation_exceptions_never_become_unknown_data(monkeypatch, exception):
    def fail(*_args):
        raise exception

    monkeypatch.setattr(v10c, "load_successor_cache_pair", fail)
    result = diagnostic.run_diagnostic(
        _phase_a(), authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert result["result_class"] == diagnostic.RESULT_IMPLEMENTATION
    assert result["first_failed_stage"] is None


def test_successful_localization_is_not_a_data_pass(monkeypatch):
    monkeypatch.setattr(v10c, "load_successor_cache_pair", lambda *_: ({}, {}, {}, object()))
    monkeypatch.setattr(v9, "build_dataset", lambda *args: object())
    monkeypatch.setattr(diagnostic, "_structural_formal_preconditions", lambda *args: object())
    monkeypatch.setattr(diagnostic, "_structural_post_scoring_conditions", lambda *args: None)
    result = diagnostic.run_diagnostic(
        _phase_a(), authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert result["result_class"] == diagnostic.RESULT_IMPLEMENTATION
    assert result["first_failed_stage"] is None
    assert result["future_profitability_established"] is False


def test_safe_result_validator_is_closed_and_private_reason_free():
    result = diagnostic._safe_result(
        IMPLEMENTATION_SHA, diagnostic.RESULT_DATA, "PARSER_NORMALIZATION_CONTRACT"
    )
    assert diagnostic.validate_safe_result(result) == result
    with pytest.raises(diagnostic.V10DDiagnosticImplementationFailure):
        diagnostic.validate_safe_result({**result, "unexpected": True})
    with pytest.raises(diagnostic.V10DDiagnosticImplementationFailure):
        diagnostic.validate_safe_result({**result, "authority_consumed": False})
    assert "CACHE_PAYLOAD_HASH_MISMATCH" not in str(result)


@pytest.mark.parametrize(
    "key",
    sorted(
        {
            "design_git_commit",
            "design_git_blob_sha1",
            "design_sha256",
            "freeze_approval_git_blob_sha1",
            "freeze_approval_sha256",
            "training_manifest_sha256",
            "evaluation_manifest_sha256",
            "v4_universe_csv_sha256",
            "v4_ticker_list_sha256",
            "v10a_calendar_sha256",
        }
    ),
)
def test_safe_result_requires_exact_frozen_provenance(key):
    result = diagnostic._safe_result(
        IMPLEMENTATION_SHA, diagnostic.RESULT_DATA, "PARSER_NORMALIZATION_CONTRACT"
    )
    tampered = copy.deepcopy(result)
    original = tampered["provenance"][key]
    tampered["provenance"][key] = ("0" * len(original)) if original != ("0" * len(original)) else ("1" * len(original))
    with pytest.raises(diagnostic.V10DDiagnosticImplementationFailure):
        diagnostic.validate_safe_result(tampered)


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_safe_result_provenance_key_set_is_exact(operation):
    result = diagnostic._safe_result(
        IMPLEMENTATION_SHA, diagnostic.RESULT_DATA, "PARSER_NORMALIZATION_CONTRACT"
    )
    tampered = copy.deepcopy(result)
    if operation == "missing":
        tampered["provenance"].pop("design_sha256")
    else:
        tampered["provenance"]["unexpected"] = "0" * 64
    with pytest.raises(diagnostic.V10DDiagnosticImplementationFailure):
        diagnostic.validate_safe_result(tampered)


def test_safe_result_requires_exact_frozen_counts_and_counters():
    result = diagnostic._safe_result(
        IMPLEMENTATION_SHA, diagnostic.RESULT_DATA, "PARSER_NORMALIZATION_CONTRACT"
    )
    for key, value in {
        "training_success_count": 282,
        "training_failed_count": 18,
        "evaluation_payload_count": 299,
    }.items():
        tampered = copy.deepcopy(result)
        tampered["counts"][key] = value
        with pytest.raises(diagnostic.V10DDiagnosticImplementationFailure):
            diagnostic.validate_safe_result(tampered)
    for key in ("network_requests", "model_fits", "t0_runs"):
        tampered = copy.deepcopy(result)
        tampered["execution_counters"][key] = 1
        with pytest.raises(diagnostic.V10DDiagnosticImplementationFailure):
            diagnostic.validate_safe_result(tampered)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("training_success_count", True),
        ("training_failed_count", "17"),
        ("evaluation_payload_count", None),
        ("network_requests", True),
        ("model_fits", "0"),
        ("t0_runs", None),
    ],
)
def test_safe_result_rejects_wrong_count_and_counter_types(key, value):
    result = diagnostic._safe_result(
        IMPLEMENTATION_SHA, diagnostic.RESULT_DATA, "PARSER_NORMALIZATION_CONTRACT"
    )
    tampered = copy.deepcopy(result)
    target = "counts" if key in diagnostic._SAFE_COUNT_KEYS else "execution_counters"
    tampered[target][key] = value
    with pytest.raises(diagnostic.V10DDiagnosticImplementationFailure):
        diagnostic.validate_safe_result(tampered)


def test_safe_result_expected_contracts_are_fresh_not_mutable_aliases():
    result = diagnostic._safe_result(
        IMPLEMENTATION_SHA, diagnostic.RESULT_DATA, "PARSER_NORMALIZATION_CONTRACT"
    )
    expected_provenance = diagnostic._safe_provenance()
    expected_provenance["design_sha256"] = "0" * 64
    expected_counts = diagnostic._safe_counts()
    expected_counts["training_success_count"] = 0
    expected_counters = diagnostic._safe_execution_counters()
    expected_counters["t0_runs"] = 1
    assert diagnostic.validate_safe_result(result) == result


def test_safe_result_rejects_profitability_claim():
    result = diagnostic._safe_result(
        IMPLEMENTATION_SHA, diagnostic.RESULT_DATA, "PARSER_NORMALIZATION_CONTRACT"
    )
    tampered = copy.deepcopy(result)
    tampered["future_profitability_established"] = True
    with pytest.raises(diagnostic.V10DDiagnosticImplementationFailure):
        diagnostic.validate_safe_result(tampered)


def test_no_model_or_scientific_screening_calls_in_diagnostic_source():
    source = __import__("inspect").getsource(diagnostic)
    assert ".fit(" not in source
    assert ".predict(" not in source
    assert "screen_top1" not in source
    assert "top1_metrics" not in source
    assert "STOP" not in source
    assert "CONTINUE" not in source
