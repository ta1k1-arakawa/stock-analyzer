from __future__ import annotations

from types import SimpleNamespace

import pytest

from src import v10d_t0_data_incompatibility_diagnostic as diagnostic
from src import v10c_t0_successor_training_input_binding as v10c
from src import v9_009_t0_top1_kill_screen as v9


IMPLEMENTATION_SHA = "a" * 40


def _phase_a() -> diagnostic.PhaseAMetadata:
    return diagnostic.PhaseAMetadata(SimpleNamespace(), IMPLEMENTATION_SHA)


def _raise(reason: str):
    def raiser(_metadata):
        raise v9.T0DataIncompatible(reason)

    return raiser


def test_phase_a_reuses_v10c_metadata_preflight_without_payload_or_parser(monkeypatch):
    calls = []
    metadata = object()
    monkeypatch.setattr(diagnostic, "_verify_repository_and_freeze", lambda *_: calls.append("repo"))
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
    assert calls[0] == "repo"
    assert len(calls) == 2


def test_inherited_phase_a_failure_remains_preflight_failure(monkeypatch):
    monkeypatch.setattr(diagnostic, "_verify_repository_and_freeze", lambda *_: None)

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
        diagnostic.run_diagnostic(_phase_a(), [], authority_boundary_token=None)


@pytest.mark.parametrize(
    ("reason", "stage"),
    [
        ("CACHE_PAYLOAD_SET_INVALID", "INPUT_BYTE_OR_FILESET_CONTRACT"),
        ("CACHE_PAYLOAD_HASH_MISMATCH", "PARSER_NORMALIZATION_CONTRACT"),
        ("DUPLICATE_COMBINED_PRICE_DATE", "COMBINED_SERIES_CONTRACT"),
        ("UNMAPPED_REASON", "UNKNOWN_DATA_INCOMPATIBILITY"),
    ],
)
def test_cache_data_failures_map_only_known_reasons(monkeypatch, reason, stage):
    monkeypatch.setattr(v10c, "load_successor_cache_pair", _raise(reason))
    result = diagnostic.run_diagnostic(
        _phase_a(), [], authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert result["result_class"] == diagnostic.RESULT_DATA
    assert result["first_failed_stage"] == stage
    assert "reason" not in str(result)


def test_dataset_data_failure_maps_to_feature_target_stage(monkeypatch):
    monkeypatch.setattr(v10c, "load_successor_cache_pair", lambda *_: ({}, {}, {}, object()))
    monkeypatch.setattr(v9, "build_dataset", lambda *args: (_ for _ in ()).throw(v9.T0DataIncompatible("DATASET_SCHEMA_INVALID")))
    result = diagnostic.run_diagnostic(
        _phase_a(), [], authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
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
        _phase_a(), [], authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert formal_result["first_failed_stage"] == "FORMAL_SCORING_PRECONDITION_CONTRACT"

    monkeypatch.setattr(diagnostic, "_structural_formal_preconditions", lambda *args: object())
    monkeypatch.setattr(
        diagnostic,
        "_structural_post_scoring_conditions",
        lambda *args: (_ for _ in ()).throw(v9.T0DataIncompatible("FORMAL_TARGET_UNAVAILABLE")),
    )
    post_result = diagnostic.run_diagnostic(
        _phase_a(), [], authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert post_result["first_failed_stage"] == "POST_SCORING_STRUCTURAL_TARGET_CONTRACT"


@pytest.mark.parametrize("exception", [v9.T0ImplementationFailure("x"), RuntimeError("x")])
def test_implementation_exceptions_never_become_unknown_data(monkeypatch, exception):
    def fail(*_args):
        raise exception

    monkeypatch.setattr(v10c, "load_successor_cache_pair", fail)
    result = diagnostic.run_diagnostic(
        _phase_a(), [], authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
    )
    assert result["result_class"] == diagnostic.RESULT_IMPLEMENTATION
    assert result["first_failed_stage"] is None


def test_successful_localization_is_not_a_data_pass(monkeypatch):
    monkeypatch.setattr(v10c, "load_successor_cache_pair", lambda *_: ({}, {}, {}, object()))
    monkeypatch.setattr(v9, "build_dataset", lambda *args: object())
    monkeypatch.setattr(diagnostic, "_structural_formal_preconditions", lambda *args: object())
    monkeypatch.setattr(diagnostic, "_structural_post_scoring_conditions", lambda *args: None)
    result = diagnostic.run_diagnostic(
        _phase_a(), [], authority_boundary_token=diagnostic.DIAGNOSTIC_BOUNDARY_TOKEN
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


def test_no_model_or_scientific_screening_calls_in_diagnostic_source():
    source = __import__("inspect").getsource(diagnostic)
    assert ".fit(" not in source
    assert ".predict(" not in source
    assert "screen_top1" not in source
    assert "top1_metrics" not in source
    assert "STOP" not in source
    assert "CONTINUE" not in source
