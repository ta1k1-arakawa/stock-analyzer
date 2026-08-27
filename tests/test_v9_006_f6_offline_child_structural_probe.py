from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from src import v9_006_f6_offline_child_structural_probe as probe
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
    secret_url = "https://private.example.invalid/child"
    meta_path = raw / ("a" * 64 + ".json")
    bin_path = meta_path.with_suffix(".bin")
    bin_path.write_bytes(payload)
    meta_path.write_text(json.dumps({
        "schema_version": "V9_005_STAGE_A_RAW_LOCK_V1", "source_family": probe.SOURCE_FAMILY,
        "applicable_period": probe.APPLICABLE_PERIOD, "requested_url": secret_url,
        "resolved_url": secret_url, "http_status": 200,
        "retrieval_timestamp_utc": "2026-08-27T00:00:00Z", "byte_length": len(payload),
        "sha256": sha256(payload).hexdigest(),
    }), encoding="utf-8")
    return parent, root, _bindings(root, payload), bin_path, secret_url


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
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    result = probe.run_offline_child_structural_probe(
        production_state_parent=parent, output_root=root, bindings=bindings,
        structural_inspector=lambda _raw: {"status": status, "container_format": "SYNTHETIC", "open_parse_status": "SYNTHETIC", "sheet_table_count": 1, "structural_dimensions": [{"ordinal": 1, "row_count": 2, "column_count": 3}]},
    )
    assert result["status"] == status
    assert result["coverage_evaluated"] is False


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
