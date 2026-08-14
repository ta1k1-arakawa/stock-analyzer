from __future__ import annotations

from src import v8c_research_opening_guard as guard


def test_research_opening_gates_are_the_expected_two():
    assert guard.RESEARCH_OPENING_GATE_BY_BLOCK == {
        "T1C": "SEPARATE_T1C_RESEARCH_OPENING_GATE",
        "T2": "SEPARATE_T2_RESEARCH_OPENING_GATE",
    }


def test_invalid_block_rejected():
    import pytest

    with pytest.raises(guard.V8CResearchOpeningGuardBlocked) as excinfo:
        guard.verify_point_of_use_before_opening("/tmp/nonexistent", "T3")
    assert excinfo.value.reason == "V8C_RESEARCH_OPENING_BLOCK_INVALID"


def test_no_open_for_api_exists_in_bound_acquisition_module():
    """Mirrors src.v8b_historical_acquisition's own security invariant:
    no function that returns raw OHLCV/feature/outcome data exists."""
    assert guard.no_research_opening_api_exists() is True


def test_verify_point_of_use_never_authorizes_opening_or_consumes_gate(tmp_path, monkeypatch):
    """Even a synthetic 'PASS' from the underlying resolver must never
    cause this function to claim authorization or gate consumption."""
    monkeypatch.setattr(
        guard, "resolve_and_verify_acquisition_artifact", lambda output_root, block: {"result": "PASS"}
    )
    result = guard.verify_point_of_use_before_opening(tmp_path, "T1C")
    assert result["result"] == "PASS"
    assert result["authorizes_opening"] is False
    assert result["consumes_research_opening_gate"] is False


def test_verify_point_of_use_propagates_verification_failure(tmp_path, monkeypatch):
    def failing(output_root, block):
        raise guard.V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_SHA256_MISMATCH")

    monkeypatch.setattr(guard, "resolve_and_verify_acquisition_artifact", failing)
    import pytest

    with pytest.raises(guard.V8CResearchOpeningGuardBlocked) as excinfo:
        guard.verify_point_of_use_before_opening(tmp_path, "T1C")
    assert excinfo.value.reason == "POINT_OF_USE_VERIFICATION_FAILED:RAW_PAYLOAD_SHA256_MISMATCH"


def test_point_of_use_detects_tampering_introduced_after_an_earlier_call(tmp_path, monkeypatch):
    """The guard must never cache/trust an earlier PASS -- tampering that
    occurs strictly between two calls must be independently re-detected on
    the second call."""
    call_count = {"n": 0}

    def sometimes_tampered(output_root, block):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"result": "PASS"}
        raise guard.V8CAcquisitionArtifactVerificationBlocked("RAW_PAYLOAD_SHA256_MISMATCH")

    monkeypatch.setattr(guard, "resolve_and_verify_acquisition_artifact", sometimes_tampered)

    first = guard.verify_point_of_use_before_opening(tmp_path, "T1C")
    assert first["result"] == "PASS"

    import pytest

    with pytest.raises(guard.V8CResearchOpeningGuardBlocked) as excinfo:
        guard.verify_point_of_use_before_opening(tmp_path, "T1C")
    assert excinfo.value.reason == "POINT_OF_USE_VERIFICATION_FAILED:RAW_PAYLOAD_SHA256_MISMATCH"
    assert call_count["n"] == 2  # re-verified fresh both times, never cached


def test_no_write_or_open_api_exists_in_this_module():
    for name in guard.__all__:
        assert not name.startswith("open_")
        assert "write" not in name.lower()
