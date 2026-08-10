"""Fake-only security regression tests for the V8 acquisition-manifest trust
boundary.

These remediate the CRITICAL finding from the independent
POLICY_G_PRIME_V1 implementation review: read_acquisition_manifest() used
plain json.loads() (silently resolving duplicate keys to their last
occurrence) and never re-validated a manifest's immutable acquisition-time
invariants by value, allowing a tampered persisted acquisition_manifest.json
to defeat every official T2 sealed-holdout access guard
(open_for_feature_generation / candidate_generation / validation / backtest
/ profit_evaluation).

No network. No private partition manifest or raw data. No T1 attempt #2 or
T2/T3 production acquisition -- every acquisition here is fake-only, via the
existing private _acquire_historical_block_bundle_with_validated_inputs seam.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import v8_historical_acquisition as acquisition
from test_v8_historical_acquisition import acquire_kwargs, default_opener

GUARD_FUNCTIONS = (
    acquisition.open_for_feature_generation,
    acquisition.open_for_candidate_generation,
    acquisition.open_for_validation,
    acquisition.open_for_backtest,
    acquisition.open_for_profit_evaluation,
)


# ---------------------------------------------------------------------------
# Helpers -- raw JSON byte/text manipulation only, never Python dicts, for
# the duplicate-key scenarios (a dict cannot represent a duplicate key).
# ---------------------------------------------------------------------------


def _acquire_genuine_manifest_text(tmp_path: Path, block: str = "T2", ticker: str = "2001") -> tuple[Path, str]:
    opener = default_opener()
    acquisition._acquire_historical_block_bundle_with_validated_inputs(
        **acquire_kwargs(tmp_path / "root", block, [ticker], opener)
    )
    manifest_path = tmp_path / "root" / acquisition.ACQUISITIONS_DIRNAME / block / acquisition.MANIFEST_FILENAME
    return manifest_path, manifest_path.read_text(encoding="utf-8")


def _append_duplicate_top_level_key(raw_text: str, key: str, json_literal: str) -> str:
    """Insert a duplicate top-level ``"key": json_literal`` as the LAST
    top-level key. Plain last-wins JSON parsing (if duplicate rejection
    were absent) would resolve to THIS value -- so a test asserting BLOCK
    here proves strict rejection actually fired, not that the tamper
    happened to be harmless."""
    stripped = raw_text.rstrip("\n")
    assert stripped.endswith("}")
    return stripped[:-1] + f',"{key}":{json_literal}}}\n'


def _append_duplicate_nested_policy_key(raw_text: str, key: str, json_literal: str) -> str:
    marker = '"malformed_ohlcv_policy":{'
    start = raw_text.index(marker) + len(marker)
    end = raw_text.index("}", start)  # malformed_ohlcv_policy has no nested braces of its own
    return raw_text[:end] + f',"{key}":{json_literal}' + raw_text[end:]


def _set_single_top_level_value(raw_text: str, key: str, value: object) -> bytes:
    """Re-serialize with exactly one occurrence of ``key`` changed -- not a
    duplicate-key scenario, exercises the separate value-invariant check."""
    manifest = json.loads(raw_text)
    manifest[key] = value
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _assert_all_guards_block(manifest_like: object) -> None:
    for guard in GUARD_FUNCTIONS:
        with pytest.raises(acquisition.V8SealedHoldoutBlocked):
            guard(manifest_like)


# ---------------------------------------------------------------------------
# Section 3 -- duplicate top-level and nested JSON key rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "json_literal"),
    (
        ("sealed", "false"),
        ("research_access_authorized", "true"),
        ("schema_version", '"TAMPERED"'),
    ),
)
def test_duplicate_top_level_key_rejected(tmp_path, key, json_literal):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    manifest_path.write_text(_append_duplicate_top_level_key(raw_text, key, json_literal), encoding="utf-8")
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "ACQUISITION_MANIFEST_DUPLICATE_KEY"


def test_duplicate_nested_malformed_ohlcv_policy_key_rejected(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    tampered = _append_duplicate_nested_policy_key(raw_text, "invalid_fraction_threshold", "0.5")
    manifest_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "ACQUISITION_MANIFEST_DUPLICATE_KEY"


def test_duplicate_key_rejection_does_not_reject_the_genuine_untampered_manifest(tmp_path):
    """Sanity check: the strict parser must not false-positive on a real,
    non-duplicated manifest."""
    manifest_path, _ = _acquire_genuine_manifest_text(tmp_path)
    reread = acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert reread["sealed"] is True
    assert reread["research_access_authorized"] is False


# ---------------------------------------------------------------------------
# Section 4 -- direct (single, non-duplicate) manifest tampering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value", "expected_reason"),
    (
        ("schema_version", "V8_HISTORICAL_ACQUISITION_V1", "ACQUISITION_MANIFEST_SCHEMA_VERSION_MISMATCH"),
        ("schema_version", "ARBITRARY_TEXT", "ACQUISITION_MANIFEST_SCHEMA_VERSION_MISMATCH"),
        ("study_name", "WRONG_STUDY", "ACQUISITION_MANIFEST_STUDY_NAME_MISMATCH"),
        ("design_commit", "0" * 40, "ACQUISITION_MANIFEST_DESIGN_COMMIT_MISMATCH"),
        ("role", "WRONG_ROLE", "ACQUISITION_MANIFEST_ROLE_MISMATCH"),
        ("status", "WRONG_STATUS", "ACQUISITION_MANIFEST_STATUS_MISMATCH"),
        ("sealed", False, "ACQUISITION_MANIFEST_SEALED_MISMATCH"),
        ("research_access_authorized", True, "ACQUISITION_MANIFEST_RESEARCH_ACCESS_INVARIANT_VIOLATED"),
        ("data_source", "WRONG", "ACQUISITION_MANIFEST_DATA_SOURCE_MISMATCH"),
        ("data_source_host", "evil.example.com", "ACQUISITION_MANIFEST_DATA_SOURCE_HOST_MISMATCH"),
        ("data_source_schema", "WRONG", "ACQUISITION_MANIFEST_DATA_SOURCE_SCHEMA_MISMATCH"),
        ("request_start", "2000-01-01", "ACQUISITION_MANIFEST_REQUEST_START_MISMATCH"),
        ("request_end_exclusive", "2099-01-01", "ACQUISITION_MANIFEST_REQUEST_END_EXCLUSIVE_MISMATCH"),
        ("retry_count", 1, "ACQUISITION_MANIFEST_RETRY_COUNT_MISMATCH"),
    ),
)
def test_single_field_invariant_tampering_rejected(tmp_path, key, value, expected_reason):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    manifest_path.write_bytes(_set_single_top_level_value(raw_text, key, value))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == expected_reason


@pytest.mark.parametrize("field", acquisition.ACQUISITION_MANIFEST_ZERO_ACCESS_COUNTER_FIELDS)
def test_access_counter_tampering_rejected(tmp_path, field):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    manifest_path.write_bytes(_set_single_top_level_value(raw_text, field, 1))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "ACQUISITION_MANIFEST_ACCESS_COUNTER_INVARIANT_VIOLATED"


def test_access_counter_boolean_false_not_silently_accepted_as_zero(tmp_path):
    """0 == False in Python; the invariant check must not be fooled by a
    boolean masquerading as the integer zero."""
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    manifest_path.write_bytes(_set_single_top_level_value(raw_text, "validation_access_count", False))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "ACQUISITION_MANIFEST_ACCESS_COUNTER_INVARIANT_VIOLATED"


def test_block_field_changed_to_t1_inside_persisted_t2_file_rejected(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    manifest_path.write_bytes(_set_single_top_level_value(raw_text, "block", "T1"))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "ACQUISITION_MANIFEST_BLOCK_MISMATCH"


def test_genuine_untampered_manifest_passes_all_invariant_checks(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    reread = acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert reread["block"] == "T2"
    assert reread["role"] == "SEALED_HOLDOUT"
    assert reread["sealed"] is True
    assert reread["research_access_authorized"] is False


# ---------------------------------------------------------------------------
# Policy-metadata and top-level field-set edges (section 8)
# ---------------------------------------------------------------------------


def test_policy_metadata_extra_field_rejected(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    manifest = json.loads(raw_text)
    manifest["malformed_ohlcv_policy"]["extra_field"] = "unexpected"
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "MALFORMED_OHLCV_POLICY_METADATA_SCHEMA_INVALID"


def test_policy_metadata_missing_field_rejected(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    manifest = json.loads(raw_text)
    del manifest["malformed_ohlcv_policy"]["max_consecutive_invalid_returned_rows"]
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "MALFORMED_OHLCV_POLICY_METADATA_SCHEMA_INVALID"


def test_top_level_extra_field_rejected(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    manifest = json.loads(raw_text)
    manifest["unexpected_extra_field"] = "x"
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "MANIFEST_SCHEMA_INVALID"


def test_top_level_missing_field_rejected(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    manifest = json.loads(raw_text)
    del manifest["retry_count"]
    manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "MANIFEST_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# Section 12 -- mandatory exploit regression, A-E exactly as specified
# ---------------------------------------------------------------------------


def test_section12_a_duplicate_sealed_true_then_false_blocks(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    tampered_text = _append_duplicate_top_level_key(raw_text, "sealed", "false")
    manifest_path.write_text(tampered_text, encoding="utf-8")

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")

    # Defense in depth: even a caller that bypasses read_acquisition_manifest
    # entirely and plain-parses the tampered bytes itself (last-wins) must
    # still be denied by every official guard.
    _assert_all_guards_block(json.loads(tampered_text))


def test_section12_b_duplicate_research_access_authorized_false_then_true_blocks(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    tampered_text = _append_duplicate_top_level_key(raw_text, "research_access_authorized", "true")
    manifest_path.write_text(tampered_text, encoding="utf-8")

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")

    _assert_all_guards_block(json.loads(tampered_text))


def test_section12_c_single_sealed_false_blocks(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    tampered_bytes = _set_single_top_level_value(raw_text, "sealed", False)
    manifest_path.write_bytes(tampered_bytes)

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")

    _assert_all_guards_block(json.loads(tampered_bytes))


def test_section12_d_single_research_access_authorized_true_blocks(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    tampered_bytes = _set_single_top_level_value(raw_text, "research_access_authorized", True)
    manifest_path.write_bytes(tampered_bytes)

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")

    _assert_all_guards_block(json.loads(tampered_bytes))


def test_section12_e_block_changed_to_t1_inside_persisted_t2_file_blocks(tmp_path):
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    tampered_bytes = _set_single_top_level_value(raw_text, "block", "T1")
    manifest_path.write_bytes(tampered_bytes)

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked):
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")

    # Guards never inspect "block" at all -- sealed/research_access_authorized
    # are untouched (still genuinely sealed) in this scenario, so they must
    # still independently block too.
    _assert_all_guards_block(json.loads(tampered_bytes))


def test_full_combined_bypass_from_original_review_now_blocked_at_read(tmp_path):
    """The exact exploit empirically demonstrated during the independent
    review: duplicate 'sealed' (true->false) and duplicate
    'research_access_authorized' (false->true) together used to defeat
    every guard when read through read_acquisition_manifest(). Must now
    BLOCK before any guard is ever reached."""
    manifest_path, raw_text = _acquire_genuine_manifest_text(tmp_path)
    stripped = raw_text.rstrip("\n")
    assert stripped.endswith("}")
    tampered_text = stripped[:-1] + ',"sealed":false,"research_access_authorized":true}\n'
    manifest_path.write_text(tampered_text, encoding="utf-8")

    with pytest.raises(acquisition.V8HistoricalAcquisitionBlocked) as excinfo:
        acquisition.read_acquisition_manifest(tmp_path / "root", "T2")
    assert excinfo.value.reason == "ACQUISITION_MANIFEST_DUPLICATE_KEY"
