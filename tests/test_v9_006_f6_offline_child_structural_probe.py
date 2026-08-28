from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from src import v9_006_f6_offline_child_structural_probe as probe
from src.v9_005_stage_a_jpx_probe import SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
from scripts import run_v9_006_f6_offline_child_structural_probe as cli


def _bindings(root: Path, payload: bytes) -> probe.ProbeBindings:
    return probe.ProbeBindings(
        output_root_id_sha256=probe.output_root_id_sha256(root),
        child_sha256=sha256(payload).hexdigest(),
        child_byte_length=len(payload),
    )


def _fixture(tmp_path: Path, payload: bytes = b"PK\x03\x04synthetic") -> tuple[Path, Path, probe.ProbeBindings, Path, str]:
    parent = tmp_path / "protected-parent"
    root = parent / "protected-output"
    raw = root / "raw"
    raw.mkdir(parents=True)
    (root / probe.RECEIPT_FILENAME).write_text(json.dumps({
        "schema_version": probe.RECEIPT_SCHEMA, "task": probe.RECEIPT_TASK,
        "confirmation_contract": probe.RECEIPT_CONTRACT, "gate_consumed": True,
        "consumption_timestamp_utc": "2026-08-27T00:00:00Z",
    }), encoding="utf-8")
    # A canonical synthetic candidate: a valid JPX-domain URL and a metadata
    # filename equal to the real repository raw-lock key derived from it, per
    # the canonical semantics in src/v9_005_stage_a_jpx_probe.py.
    synthetic_url = "https://www.jpx.co.jp/english/markets/indices/topix/synthetic-fixture-child"
    key = probe.source_object_slot_id(probe.SOURCE_FAMILY, probe.APPLICABLE_PERIOD, synthetic_url)
    meta_path = raw / (key + ".json")
    bin_path = meta_path.with_suffix(".bin")
    bin_path.write_bytes(payload)
    meta_path.write_text(json.dumps({
        "schema_version": "V9_005_STAGE_A_RAW_LOCK_V1", "source_family": probe.SOURCE_FAMILY,
        "applicable_period": probe.APPLICABLE_PERIOD, "requested_url": synthetic_url,
        "resolved_url": synthetic_url, "http_status": 200,
        "retrieval_timestamp_utc": "2026-08-27T00:00:00Z", "byte_length": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }), encoding="utf-8")
    return parent, root, _bindings(root, payload), bin_path, synthetic_url


def _read_meta(root: Path) -> tuple[Path, dict[str, object]]:
    meta_path = next((root / "raw").glob("*.json"))
    return meta_path, json.loads(meta_path.read_text(encoding="utf-8"))


def _write_meta(meta_path: Path, value: dict[str, object]) -> None:
    meta_path.write_text(json.dumps(value), encoding="utf-8")


def _assert_phase_a_blocked_without_bin_read(
    parent: Path, root: Path, bindings: probe.ProbeBindings, bin_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("bin read")) if self == bin_path else original_read(self))
    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.locate_metadata_only(production_state_parent=parent, output_root=root, bindings=bindings)
    # Every Phase-A rejection must preserve accurate false/false provenance:
    # no CHILD byte was ever read.
    assert raised.value.raw_bytes_read_for_integrity is False
    assert raised.value.child_content_inspected is False


def test_metadata_locator_and_default_unsupported_are_safe(tmp_path: Path) -> None:
    parent, root, bindings, _bin, url = _fixture(tmp_path)
    result = probe.run_offline_child_structural_probe(production_state_parent=parent, output_root=root, bindings=bindings)
    assert result["status"] == "STRUCTURAL_FORMAT_UNSUPPORTED"
    assert result["network_request_count"] == 0
    assert result["coverage_evaluated"] is False
    assert result["child_content_inspected"] is True
    public = json.dumps(result)
    assert str(parent) not in public and str(root) not in public and url not in public


@pytest.mark.parametrize("mutate", ["hash", "outside"])
def test_output_root_binding_failures_stop_before_child_read(tmp_path: Path, mutate: str, monkeypatch: pytest.MonkeyPatch) -> None:
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    if mutate == "hash":
        bindings = probe.ProbeBindings(**{**bindings.__dict__, "output_root_id_sha256": "0" * 64})
    else:
        parent = tmp_path / "other-parent"
        parent.mkdir()
    original = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("bin read")) if self == bin_path else original(self))
    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.locate_metadata_only(production_state_parent=parent, output_root=root, bindings=bindings)
    assert raised.value.outcome == "CHATGPT_DECISION_REQUIRED"
    assert raised.value.raw_bytes_read_for_integrity is False
    assert raised.value.child_content_inspected is False


@pytest.mark.parametrize("kind", ["missing-receipt", "duplicate-meta", "malformed-meta"])
def test_metadata_failures_do_not_read_bin(tmp_path: Path, kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    if kind == "missing-receipt":
        (root / probe.RECEIPT_FILENAME).unlink()
    elif kind == "duplicate-meta":
        original = next((root / "raw").glob("*.json"))
        (root / "raw" / ("b" * 64 + ".json")).write_text(original.read_text(encoding="utf-8"), encoding="utf-8")
        (root / "raw" / ("b" * 64 + ".bin")).write_bytes(b"different")
    else:
        next((root / "raw").glob("*.json")).write_text("{", encoding="utf-8")
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("bin read")) if self == bin_path else original_read(self))
    with pytest.raises(probe.ProbeBlocked):
        probe.locate_metadata_only(production_state_parent=parent, output_root=root, bindings=bindings)


def test_off_domain_requested_url_rejected_before_bin_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    meta_path, value = _read_meta(root)
    value["requested_url"] = "https://private.example.invalid/child"
    _write_meta(meta_path, value)
    _assert_phase_a_blocked_without_bin_read(parent, root, bindings, bin_path, monkeypatch)


def test_off_domain_resolved_url_rejected_before_bin_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    meta_path, value = _read_meta(root)
    value["resolved_url"] = "https://private.example.invalid/child"
    _write_meta(meta_path, value)
    _assert_phase_a_blocked_without_bin_read(parent, root, bindings, bin_path, monkeypatch)


@pytest.mark.parametrize(
    "bad_timestamp",
    ["2026-08-27 00:00:00Z", "2026-08-27T00:00:00", "2026/08/27T00:00:00Z", "not-a-timestamp", "2026-08-27T00:00:00+00:00"],
)
def test_noncanonical_timestamp_rejected_before_bin_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_timestamp: str) -> None:
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    meta_path, value = _read_meta(root)
    value["retrieval_timestamp_utc"] = bad_timestamp
    _write_meta(meta_path, value)
    _assert_phase_a_blocked_without_bin_read(parent, root, bindings, bin_path, monkeypatch)


@pytest.mark.parametrize("bad_status", [99, 600, True, "200", 0])
def test_invalid_http_status_rejected_before_bin_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_status: object) -> None:
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    meta_path, value = _read_meta(root)
    value["http_status"] = bad_status
    _write_meta(meta_path, value)
    _assert_phase_a_blocked_without_bin_read(parent, root, bindings, bin_path, monkeypatch)


def test_wrong_metadata_filename_key_rejected_before_bin_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Metadata content is otherwise entirely valid; only the filename stem /
    # raw-lock key diverges from the canonical key derived from its own
    # source_family + applicable_period + requested_url.
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    meta_path, _value = _read_meta(root)
    wrong_stem = "f" * 64
    new_meta_path = meta_path.parent / (wrong_stem + ".json")
    new_bin_path = meta_path.parent / (wrong_stem + ".bin")
    meta_path.rename(new_meta_path)
    bin_path.rename(new_bin_path)
    _assert_phase_a_blocked_without_bin_read(parent, root, bindings, new_bin_path, monkeypatch)


@pytest.mark.parametrize("bad_sha", ["0" * 63, "0" * 65, "g" * 64, "A" * 64, "0" * 63 + "Z"])
def test_malformed_sha_rejected_before_bin_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_sha: str) -> None:
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    meta_path, value = _read_meta(root)
    value["sha256"] = bad_sha
    _write_meta(meta_path, value)
    _assert_phase_a_blocked_without_bin_read(parent, root, bindings, bin_path, monkeypatch)


def test_canonical_synthetic_candidate_reaches_phase_b(tmp_path: Path) -> None:
    # A fully canonical synthetic candidate (valid JPX-domain URLs, canonical
    # timestamp/status/SHA format, and a filename equal to the real
    # repository raw-lock key) must pass Phase A and reach Phase B.
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    meta_path, meta, raw_path = probe.locate_metadata_only(production_state_parent=parent, output_root=root, bindings=bindings)
    assert raw_path == bin_path
    raw = probe.content_blind_integrity_read(raw_path, meta, bindings=bindings)
    assert raw == bin_path.read_bytes()


@pytest.mark.parametrize("kind", ["sha", "length", "meta"])
def test_content_blind_integrity_mismatches_block_before_structural(tmp_path: Path, kind: str) -> None:
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    if kind == "sha":
        bindings = probe.ProbeBindings(**{**bindings.__dict__, "child_sha256": "0" * 64})
    elif kind == "length":
        bindings = probe.ProbeBindings(**{**bindings.__dict__, "child_byte_length": bindings.child_byte_length + 1})
    else:
        meta = next((root / "raw").glob("*.json")); value = json.loads(meta.read_text(encoding="utf-8")); value["sha256"] = "0" * 64; meta.write_text(json.dumps(value), encoding="utf-8")
    called = False
    def inspector(_: bytes) -> dict[str, object]:
        nonlocal called
        called = True
        return {"status": "STRUCTURAL_FORMAT_CAPTURED"}
    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.run_offline_child_structural_probe(production_state_parent=parent, output_root=root, bindings=bindings, structural_inspector=inspector)
    assert raised.value.outcome == "IMPLEMENTATION_FAILURE"
    assert called is False
    # The exact CHILD bytes were read before this mismatch was detected, but
    # structural inspection was never reached.
    assert raised.value.raw_bytes_read_for_integrity is True
    assert raised.value.child_content_inspected is False


def test_phase_b_read_exception_reports_unknown_not_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    _meta_path, meta, raw_path = probe.locate_metadata_only(production_state_parent=parent, output_root=root, bindings=bindings)
    assert raw_path == bin_path
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(OSError("simulated read failure")))
    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.content_blind_integrity_read(raw_path, meta, bindings=bindings)
    assert raised.value.outcome == "IMPLEMENTATION_FAILURE"
    # A failed read attempt does not prove bytes were never exposed; must
    # never be fabricated False.
    assert raised.value.raw_bytes_read_for_integrity == "unknown"
    assert raised.value.child_content_inspected is False


def test_phase_c_inspector_exception_reports_true_true(tmp_path: Path) -> None:
    parent, root, bindings, _bin, _url = _fixture(tmp_path)

    def raising_inspector(_raw: bytes) -> dict[str, object]:
        raise ValueError("simulated inspector crash")

    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.run_offline_child_structural_probe(production_state_parent=parent, output_root=root, bindings=bindings, structural_inspector=raising_inspector)
    assert raised.value.outcome == "IMPLEMENTATION_FAILURE"
    assert raised.value.raw_bytes_read_for_integrity is True
    assert raised.value.child_content_inspected is True


def test_phase_c_safe_evidence_validation_failure_reports_true_true(tmp_path: Path) -> None:
    parent, root, bindings, _bin, _url = _fixture(tmp_path)

    def bad_inspector(_raw: bytes) -> dict[str, object]:
        return {"status": "STRUCTURAL_FORMAT_CAPTURED", "not_an_allowed_field": "leak"}

    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.run_offline_child_structural_probe(production_state_parent=parent, output_root=root, bindings=bindings, structural_inspector=bad_inspector)
    assert raised.value.outcome == "IMPLEMENTATION_FAILURE"
    assert raised.value.raw_bytes_read_for_integrity is True
    assert raised.value.child_content_inspected is True


@pytest.mark.parametrize("status", ["STRUCTURAL_FORMAT_CAPTURED", "STRUCTURAL_FORMAT_AMBIGUOUS"])
def test_synthetic_structural_outcomes_after_integrity(tmp_path: Path, status: str) -> None:
    # Only the strict frozen enums are used here -- no test-only production
    # enum such as "SYNTHETIC".
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    result = probe.run_offline_child_structural_probe(
        production_state_parent=parent, output_root=root, bindings=bindings,
        structural_inspector=lambda _raw: {
            "status": status,
            "container_format": "ZIP_CONTAINER",
            "open_parse_status": "OPEN_PARSE_OK",
            "sheet_table_count": 1,
            "candidate_header_column_count": 2,
            "candidate_date_column_count": 1,
            "candidate_value_column_count": 1,
            "structural_dimensions": [{"ordinal": 1, "row_count": 2, "column_count": 3, "visibility": "VISIBLE", "object_type": "WORKSHEET"}],
        },
    )
    assert result["status"] == status
    assert result["coverage_evaluated"] is False


def _assert_phase_c_evidence_rejected(parent: Path, root: Path, bindings: probe.ProbeBindings, raw_evidence: object) -> None:
    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.run_offline_child_structural_probe(
            production_state_parent=parent, output_root=root, bindings=bindings,
            structural_inspector=lambda _raw: raw_evidence,
        )
    assert raised.value.outcome == "IMPLEMENTATION_FAILURE"
    # Every Phase-C rejection must prove the CHILD bytes were read and
    # structural inspection was reached, per the reviewed MEDIUM-1 contract.
    assert raised.value.raw_bytes_read_for_integrity is True
    assert raised.value.child_content_inspected is True


@pytest.mark.parametrize(
    "bad_evidence",
    [
        # arbitrary container_format
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "container_format": "MICROSOFT_XLSX"},
        # arbitrary open_parse_status
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "open_parse_status": "OK"},
        # arbitrary string injected into structural_dimensions
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": ["not-a-dict"]},
        # nested list injected into a structural_dimensions item slot
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [["nested", "list"]]},
        # nested dict injected as a structural_dimensions item value
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "row_count": {"nested": "dict"}}]},
        # structural_dimensions itself not a list
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": {"ordinal": 1}},
        # extra nested dimension key
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "sheet_name": "Sheet1"}]},
        # date-like string smuggled under container_format
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "container_format": "2024-01-01"},
        # URL-like string smuggled under open_parse_status
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "open_parse_status": "https://example.com/leak"},
        # header/name-like string smuggled under a dimension's visibility field
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "visibility": "Header Row 2024"}]},
        # path-like string smuggled under a dimension's object_type field
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "object_type": "/var/data/2024.xlsx"}]},
        # bool where an integer count is expected
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "sheet_table_count": True},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "candidate_header_column_count": False},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "row_count": True}]},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": True}]},
        # negative count
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "sheet_table_count": -1},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "candidate_date_column_count": -5},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "column_count": -3}]},
        # ordinal below the required minimum (1)
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 0}]},
        # duplicate ordinal
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1}, {"ordinal": 1}]},
        # status outside OUTCOMES
        {"status": "NOT_A_REAL_OUTCOME"},
        {"status": None},
        # extra top-level key
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "raw_url": "https://private.example.invalid/leak"},
    ],
)
def test_malformed_structural_evidence_rejected_true_true(tmp_path: Path, bad_evidence: object) -> None:
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    _assert_phase_c_evidence_rejected(parent, root, bindings, bad_evidence)


# --- MEDIUM-3A: closed-set enum membership checks must be total for
# arbitrary (including unhashable) Python objects -- `x in a_frozenset`
# raises TypeError for a list/dict, which must never escape as an ordinary
# exception. Every case here must raise ProbeBlocked(true, true), never
# TypeError. ---------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_evidence",
    [
        {"status": []},
        {"status": {}},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "container_format": []},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "container_format": {}},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "open_parse_status": []},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "open_parse_status": {}},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "visibility": []}]},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "visibility": {}}]},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "object_type": []}]},
        {"status": "STRUCTURAL_FORMAT_CAPTURED", "structural_dimensions": [{"ordinal": 1, "object_type": {}}]},
    ],
)
def test_unhashable_enum_value_rejected_as_probeblocked_never_typeerror(tmp_path: Path, bad_evidence: object) -> None:
    # pytest.raises(probe.ProbeBlocked) inside the shared helper lets any
    # other exception type (e.g. TypeError from an unguarded frozenset
    # membership check) propagate and fail the test, so this also proves
    # the rejection is specifically ProbeBlocked, never TypeError.
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    _assert_phase_c_evidence_rejected(parent, root, bindings, bad_evidence)


def test_unexpected_safe_evidence_exception_after_phase_c_reports_true_true_not_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates a genuinely unanticipated bug inside safe-evidence validation
    # (not one of the specifically-handled malformed-value cases): it must
    # still be converted to a Phase-C ProbeBlocked(true, true) by
    # run_offline_child_structural_probe's own boundary handling, not allowed
    # to escape as a raw exception. If it escaped, the CLI's generic
    # exception handler (see test_cli_unproven_phase_exception_fails_closed_
    # to_unknown) would incorrectly report unknown/false for a failure that
    # in fact occurred after Phase C began.
    parent, root, bindings, _bin, _url = _fixture(tmp_path)

    def _raise_unexpected(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated unanticipated safe-evidence bug")

    monkeypatch.setattr(probe, "_safe_structural_evidence", _raise_unexpected)
    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.run_offline_child_structural_probe(
            production_state_parent=parent, output_root=root, bindings=bindings,
            structural_inspector=lambda _raw: {"status": "STRUCTURAL_FORMAT_CAPTURED"},
        )
    assert raised.value.outcome == "IMPLEMENTATION_FAILURE"
    assert raised.value.raw_bytes_read_for_integrity is True
    assert raised.value.child_content_inspected is True


def test_cli_failure_is_json_only_and_does_not_leak_paths_or_url(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parent, root, _bindings_value, _bin, url = _fixture(tmp_path)
    assert cli.main(["--production-state-parent", str(parent), "--output-root", str(root)]) == 2
    output = capsys.readouterr().out
    assert output.startswith("{") and str(parent) not in output and str(root) not in output and url not in output
    assert "https://" not in output and "2026-08-27" not in output


def test_cli_phase_a_failure_reports_false_false(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The CLI always runs against the real frozen production bindings, so any
    # synthetic tmp_path fixture fails the Phase A output-root hash check
    # before any CHILD byte is read -- a genuine Phase A CLI failure.
    parent, root, _bindings_value, _bin, _url = _fixture(tmp_path)
    assert cli.main(["--production-state-parent", str(parent), "--output-root", str(root)]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["execution_result"] == "BLOCKED"
    assert result["raw_bytes_read_for_integrity"] is False
    assert result["child_content_inspected"] is False


@pytest.mark.parametrize(
    "raised_exc,expected_raw,expected_inspected",
    [
        (probe.ProbeBlocked("IMPLEMENTATION_FAILURE", raw_bytes_read_for_integrity=True, child_content_inspected=False), True, False),
        (probe.ProbeBlocked("IMPLEMENTATION_FAILURE", raw_bytes_read_for_integrity="unknown", child_content_inspected=False), "unknown", False),
        (probe.ProbeBlocked("IMPLEMENTATION_FAILURE", raw_bytes_read_for_integrity=True, child_content_inspected=True), True, True),
    ],
)
def test_cli_forwards_module_phase_provenance_exactly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    raised_exc: probe.ProbeBlocked,
    expected_raw: bool | str,
    expected_inspected: bool,
) -> None:
    # Proves the CLI forwards the module's exact phase-provenance fields
    # rather than hardcoding false/false for every failure, regardless of
    # which phase raised the ProbeBlocked.
    def _raise(**_kwargs: object) -> None:
        raise raised_exc

    monkeypatch.setattr(cli, "run_offline_child_structural_probe", _raise)
    parent, root, _bindings_value, _bin, url = _fixture(tmp_path)
    assert cli.main(["--production-state-parent", str(parent), "--output-root", str(root)]) == 2
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["raw_bytes_read_for_integrity"] == expected_raw
    assert result["child_content_inspected"] is expected_inspected
    assert str(parent) not in output and str(root) not in output and url not in output


def test_cli_unproven_phase_exception_fails_closed_to_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # An exception outside the module's tracked ProbeBlocked phases (e.g. an
    # unanticipated bug) carries no provable boundary; the CLI must fail
    # closed with "unknown" rather than fabricate a false "no bytes read"
    # claim.
    def _raise(**_kwargs: object) -> None:
        raise RuntimeError("simulated unanticipated failure")

    monkeypatch.setattr(cli, "run_offline_child_structural_probe", _raise)
    parent, root, _bindings_value, _bin, url = _fixture(tmp_path)
    assert cli.main(["--production-state-parent", str(parent), "--output-root", str(root)]) == 2
    output = capsys.readouterr().out
    result = json.loads(output)
    assert result["status"] == "IMPLEMENTATION_FAILURE"
    assert result["raw_bytes_read_for_integrity"] == "unknown"
    assert result["child_content_inspected"] is False
    assert str(parent) not in output and str(root) not in output and url not in output


# --- MEDIUM-4: source_family binding must equal the real production value,
# never the identifier-name string it was previously mistaken for. ---------

def test_canonical_value_is_topix_historical_index_value() -> None:
    assert SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE == "TOPIX_HISTORICAL_INDEX_VALUE"


def test_frozen_bindings_source_family_equals_canonical_v9_005_constant() -> None:
    assert probe.FROZEN_BINDINGS.source_family == SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
    assert probe.SOURCE_FAMILY == SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
    # Never the erroneous identifier-name string this binding used to hold.
    assert probe.FROZEN_BINDINGS.source_family != "SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE"


def test_candidate_using_production_constant_passes_phase_a_to_phase_b(tmp_path: Path) -> None:
    # A synthetic candidate built with the real production source_family
    # constant (as the actual F6 raw acquisition writes it) must pass Phase A
    # and reach Phase B.
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    meta_path, value = _read_meta(root)
    assert value["source_family"] == SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
    _meta_path, meta, raw_path = probe.locate_metadata_only(production_state_parent=parent, output_root=root, bindings=bindings)
    assert raw_path == bin_path
    raw = probe.content_blind_integrity_read(raw_path, meta, bindings=bindings)
    assert raw == bin_path.read_bytes()


def test_candidate_using_erroneous_identifier_name_string_rejected_before_bin_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A candidate whose source_family field holds the old, erroneous
    # identifier-name-as-string ("SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE")
    # instead of the real production value must never be accepted.
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    meta_path, value = _read_meta(root)
    value["source_family"] = "SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE"
    _write_meta(meta_path, value)
    _assert_phase_a_blocked_without_bin_read(parent, root, bindings, bin_path, monkeypatch)
