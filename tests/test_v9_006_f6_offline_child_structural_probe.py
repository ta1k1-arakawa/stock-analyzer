from __future__ import annotations

import copy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from src import v9_006_f6_offline_child_structural_probe as probe
from src.v9_005_stage_a_jpx_probe import SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
from scripts import run_v9_006_f6_offline_child_structural_probe as cli
from scripts import generate_synthetic_jpx_xls_fixture as synthetic_xls_fixture


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


# --- OLE/BIFF structural parser implementation fixtures --------------------
# MEDIUM-1 remediation (V9_006_F6_OLE_BIFF_STRUCTURAL_PARSER_IMPL_MEDIUM_1_
# UNDECLARED_XLWT_TEST_DEPENDENCY_BREAKS_REPRODUCIBILITY): this module must
# not import xlwt -- it is not a declared test/runtime dependency in
# requirement.txt, only a manually installed fixture-generator-only tool.
# Genuine end-to-end xlrd/OLE-BIFF integration coverage instead reads the
# already-committed synthetic fixture bytes at
# tests/fixtures/synthetic_jpx_source_snapshot.xls (see
# test_default_inspector_genuine_ole_biff_fixture_is_captured_and_deterministic
# below). Importing scripts.generate_synthetic_jpx_xls_fixture for its
# committed-identity constant does not itself require xlwt: that module's own
# `import xlwt` is local to its build_workbook_bytes() function body, not
# module-level, so this import alone never pulls xlwt into this test
# module's dependency surface. Every other real-xlrd-API test below (cell
# types, nrows/ncols/EMPTY/BLANK semantics, visibility 0/1/2) uses a
# deterministic fake Book/Sheet standing in for xlrd's own Book/Sheet
# surface (nrows/ncols/visibility/cell_type only -- never cell_value/
# row_values/col_values, matching design section 5.10), so no test needs to
# actually build a workbook.

class _FakeSheet:
    """A minimal stand-in for an xlrd Sheet: nrows/ncols/visibility plus
    cell_type(row, col), the only Sheet surface the reviewed parser ever
    touches. `cell_type_grid`, if given, supplies a full (row x col) matrix
    of xlrd cell-type codes for exact per-cell control; `cell_type_code`
    supplies one code for every cell (used by the two fail-closed-only
    tests, where the exact cell type is irrelevant)."""

    def __init__(
        self,
        nrows: int,
        ncols: int,
        visibility: int,
        cell_type_grid: list[list[int]] | None = None,
        cell_type_code: int | None = None,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.visibility = visibility
        self._cell_type_grid = cell_type_grid
        self._cell_type_code = cell_type_code

    def cell_type(self, row: int, col: int) -> int:
        if self._cell_type_grid is not None:
            return self._cell_type_grid[row][col]
        return self._cell_type_code


class _FakeBook:
    def __init__(self, sheets: list[_FakeSheet]) -> None:
        self._sheets = sheets
        self.nsheets = len(sheets)

    def sheet_by_index(self, index: int) -> _FakeSheet:
        return self._sheets[index]


# A canonical, fully valid CAPTURED payload satisfying the strict frozen
# six-key contract (design sections 5.3-5.5): sheet 1 has one column summing
# to its row_count, sheet 2 has two columns each summing to its row_count,
# and cell_type_profiles is in strict (sheet_ordinal, column_ordinal)
# ascending canonical order.
_VALID_CAPTURED_PAYLOAD: dict[str, object] = {
    "status": "STRUCTURAL_FORMAT_CAPTURED",
    "container_format": "OLE_COMPOUND_FILE",
    "open_parse_status": "OPEN_PARSE_OK",
    "sheet_table_count": 2,
    "structural_dimensions": [
        {"ordinal": 1, "row_count": 2, "column_count": 1, "visibility": "VISIBLE", "object_type": "WORKSHEET"},
        {"ordinal": 2, "row_count": 1, "column_count": 2, "visibility": "HIDDEN", "object_type": "WORKSHEET"},
    ],
    "cell_type_profiles": [
        {"sheet_ordinal": 1, "column_ordinal": 1, "cell_type_counts": {"EMPTY": 0, "BLANK": 0, "TEXT": 2, "NUMBER": 0, "DATE": 0, "BOOLEAN": 0, "ERROR": 0}},
        {"sheet_ordinal": 2, "column_ordinal": 1, "cell_type_counts": {"EMPTY": 0, "BLANK": 0, "TEXT": 1, "NUMBER": 0, "DATE": 0, "BOOLEAN": 0, "ERROR": 0}},
        {"sheet_ordinal": 2, "column_ordinal": 2, "cell_type_counts": {"EMPTY": 0, "BLANK": 0, "TEXT": 0, "NUMBER": 1, "DATE": 0, "BOOLEAN": 0, "ERROR": 0}},
    ],
}


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
    # Genuine non-Excel garbage bytes: empirically verified to raise a clean
    # xlrd.XLRDError before any Book exists (design section 2.1), exercising
    # the real default OLE/BIFF parser's UNSUPPORTED path end-to-end.
    parent, root, bindings, _bin, url = _fixture(
        tmp_path, payload=b"not an excel file at all, just plain garbage bytes"
    )
    result = probe.run_offline_child_structural_probe(production_state_parent=parent, output_root=root, bindings=bindings)
    assert result["status"] == "STRUCTURAL_FORMAT_UNSUPPORTED"
    assert result["network_request_count"] == 0
    assert result["coverage_evaluated"] is False
    assert result["child_content_inspected"] is True
    assert result["structural_evidence"]["open_parse_status"] == "OPEN_PARSE_UNSUPPORTED"
    public = json.dumps(result)
    assert str(parent) not in public and str(root) not in public and url not in public


def test_default_inspector_other_open_exception_is_implementation_failure(tmp_path: Path) -> None:
    # Empirically verified xlrd 2.0.2 behavior: bytes that merely look like a
    # ZIP magic header but are not a real zip raise zipfile.BadZipFile, NOT
    # xlrd.XLRDError, from inside xlrd.open_workbook. Per design section 2.1,
    # only xlrd.XLRDError before a Book exists maps to
    # STRUCTURAL_FORMAT_UNSUPPORTED; every other open/extraction exception
    # must map to IMPLEMENTATION_FAILURE via the real default inspector.
    parent, root, bindings, _bin, _url = _fixture(tmp_path, payload=b"PK\x03\x04synthetic")
    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.run_offline_child_structural_probe(production_state_parent=parent, output_root=root, bindings=bindings)
    assert raised.value.outcome == "IMPLEMENTATION_FAILURE"
    assert raised.value.raw_bytes_read_for_integrity is True
    assert raised.value.child_content_inspected is True


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


def test_synthetic_ambiguous_outcome_after_integrity(tmp_path: Path) -> None:
    # Only the strict frozen enums are used here -- no test-only production
    # enum such as "SYNTHETIC". STRUCTURAL_FORMAT_AMBIGUOUS is not
    # STRUCTURAL_FORMAT_CAPTURED, so the strict six-key contract (and
    # cell_type_profiles) does not apply to it.
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    result = probe.run_offline_child_structural_probe(
        production_state_parent=parent, output_root=root, bindings=bindings,
        structural_inspector=lambda _raw: {
            "status": "STRUCTURAL_FORMAT_AMBIGUOUS",
            "container_format": "ZIP_CONTAINER",
            "open_parse_status": "OPEN_PARSE_OK",
            "sheet_table_count": 1,
            "candidate_header_column_count": 2,
            "candidate_date_column_count": 1,
            "candidate_value_column_count": 1,
            "structural_dimensions": [{"ordinal": 1, "row_count": 2, "column_count": 3, "visibility": "VISIBLE", "object_type": "WORKSHEET"}],
        },
    )
    assert result["status"] == "STRUCTURAL_FORMAT_AMBIGUOUS"
    assert result["coverage_evaluated"] is False


def test_exact_six_key_captured_payload_is_accepted(tmp_path: Path) -> None:
    # design sections 5.3-5.5: STRUCTURAL_FORMAT_CAPTURED requires exactly
    # six keys, fixed container_format/open_parse_status values, and fully
    # cross-validated structural_dimensions/cell_type_profiles topology.
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    result = probe.run_offline_child_structural_probe(
        production_state_parent=parent, output_root=root, bindings=bindings,
        structural_inspector=lambda _raw: copy.deepcopy(_VALID_CAPTURED_PAYLOAD),
    )
    assert result["status"] == "STRUCTURAL_FORMAT_CAPTURED"
    assert result["structural_evidence"] == _VALID_CAPTURED_PAYLOAD
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


# --- design sections 5.3-5.7: the strict CAPTURED six-key contract and the
# full structural_dimensions/cell_type_profiles topology/cardinality/order
# cross-validation. Each mutator starts from a fully valid CAPTURED payload
# (_VALID_CAPTURED_PAYLOAD) and breaks exactly one contract element, so a
# rejection here proves the specific cross-validation rule, not merely a
# missing/wrong top-level field. ---------------------------------------------

def _mutated_captured_payload(mutate) -> dict[str, object]:
    payload = copy.deepcopy(_VALID_CAPTURED_PAYLOAD)
    mutate(payload)
    return payload


_CAPTURED_TOPOLOGY_VIOLATIONS: list[tuple[str, object]] = [
    ("missing_required_key", lambda p: p.pop("sheet_table_count")),
    ("extra_key_beyond_required", lambda p: p.__setitem__("candidate_header_column_count", 1)),
    ("wrong_fixed_container_format", lambda p: p.__setitem__("container_format", "ZIP_CONTAINER")),
    ("wrong_fixed_open_parse_status", lambda p: p.__setitem__("open_parse_status", "OPEN_PARSE_AMBIGUOUS")),
    ("sheet_table_count_dimension_mismatch", lambda p: p.__setitem__("sheet_table_count", 3)),
    ("wrong_per_sheet_cardinality", lambda p: p["structural_dimensions"][0].__setitem__("column_count", 2)),
    ("cell_type_counts_sum_mismatch", lambda p: p["cell_type_profiles"][0]["cell_type_counts"].__setitem__("TEXT", 3)),
    ("duplicate_sheet_column_pair", lambda p: p["cell_type_profiles"].insert(1, copy.deepcopy(p["cell_type_profiles"][0]))),
    ("wrong_canonical_order", lambda p: p.__setitem__("cell_type_profiles", list(reversed(p["cell_type_profiles"])))),
    ("out_of_range_column_ordinal", lambda p: p["cell_type_profiles"][0].__setitem__("column_ordinal", 5)),
    ("unknown_sheet_ordinal_reference", lambda p: p["cell_type_profiles"][-1].__setitem__("sheet_ordinal", 99)),
    ("missing_cell_type_count_key", lambda p: p["cell_type_profiles"][0]["cell_type_counts"].pop("ERROR")),
    ("extra_cell_type_count_key", lambda p: p["cell_type_profiles"][0]["cell_type_counts"].__setitem__("UNKNOWN_TYPE", 0)),
    ("extra_profile_item_key", lambda p: p["cell_type_profiles"][0].__setitem__("sheet_name", "Sheet1")),
    ("missing_profile_item_key", lambda p: p["cell_type_profiles"][0].pop("column_ordinal")),
    ("cell_type_profiles_present_on_unsupported_status", lambda p: p.__setitem__("status", "STRUCTURAL_FORMAT_UNSUPPORTED")),
    ("cell_type_profiles_present_on_ambiguous_status", lambda p: p.__setitem__("status", "STRUCTURAL_FORMAT_AMBIGUOUS")),
]


@pytest.mark.parametrize(
    "mutate", [mutate for _label, mutate in _CAPTURED_TOPOLOGY_VIOLATIONS],
    ids=[label for label, _mutate in _CAPTURED_TOPOLOGY_VIOLATIONS],
)
def test_captured_topology_contract_violations_rejected_true_true(tmp_path: Path, mutate) -> None:
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    _assert_phase_c_evidence_rejected(parent, root, bindings, _mutated_captured_payload(mutate))


# --- MEDIUM-3A: closed-set enum membership checks must be total for
# arbitrary (including unhashable) Python objects -- `x in a_frozenset`
# raises TypeError for a list/dict, which must never escape as an ordinary
# exception. Every case here must raise ProbeBlocked(true, true), never
# TypeError. ---------------------------------------------------------------

# A structurally valid CAPTURED payload except for its cell_type_profiles
# value -- lets the unhashable-injection cases below reach past the exact
# six-key/fixed-value/structural_dimensions checks and exercise
# _is_valid_captured_cell_type_profiles / _is_valid_cell_type_profile_item /
# _is_valid_cell_type_counts directly with unhashable input.
_BASE_VALID_CAPTURED_MINUS_PROFILES: dict[str, object] = {
    "status": "STRUCTURAL_FORMAT_CAPTURED",
    "container_format": "OLE_COMPOUND_FILE",
    "open_parse_status": "OPEN_PARSE_OK",
    "sheet_table_count": 1,
    "structural_dimensions": [{"ordinal": 1, "row_count": 1, "column_count": 1, "visibility": "VISIBLE", "object_type": "WORKSHEET"}],
}


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
        # cell_type_profiles itself an unhashable non-list / malformed-item
        # shape, reached with an otherwise-valid six-key CAPTURED payload.
        {**_BASE_VALID_CAPTURED_MINUS_PROFILES, "cell_type_profiles": "not-a-list"},
        {**_BASE_VALID_CAPTURED_MINUS_PROFILES, "cell_type_profiles": {}},
        {**_BASE_VALID_CAPTURED_MINUS_PROFILES, "cell_type_profiles": [[]]},
        {**_BASE_VALID_CAPTURED_MINUS_PROFILES, "cell_type_profiles": [{}]},
        {**_BASE_VALID_CAPTURED_MINUS_PROFILES, "cell_type_profiles": [{"sheet_ordinal": [], "column_ordinal": 1, "cell_type_counts": {}}]},
        {**_BASE_VALID_CAPTURED_MINUS_PROFILES, "cell_type_profiles": [{"sheet_ordinal": 1, "column_ordinal": {}, "cell_type_counts": {}}]},
        {**_BASE_VALID_CAPTURED_MINUS_PROFILES, "cell_type_profiles": [{"sheet_ordinal": 1, "column_ordinal": 1, "cell_type_counts": []}]},
        {**_BASE_VALID_CAPTURED_MINUS_PROFILES, "cell_type_profiles": [{"sheet_ordinal": 1, "column_ordinal": 1, "cell_type_counts": {"TEXT": []}}]},
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


# --- V9_006_STAGE_A_F6_OLE_BIFF_STRUCTURAL_PARSER_DESIGN.md implementation:
# the real xlrd-based _default_structural_inspector, exercised end-to-end
# against the already-committed genuine legacy `.xls` fixture bytes for real
# OLE/BIFF integration coverage, and against a deterministic fake Book/Sheet
# (see _FakeSheet/_FakeBook above) for every other case, including the two
# fail-closed branches (unrecognized visibility/cell-type code) that no
# valid BIFF file can contain. This module does not import xlwt (MEDIUM-1
# remediation: xlwt is not a declared test/runtime dependency). ------------

def test_default_inspector_calls_xlrd_open_workbook_with_frozen_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # design section 2: the exact frozen xlrd.open_workbook call shape.
    captured: dict[str, object] = {}

    def fake_open_workbook(**kwargs: object) -> _FakeBook:
        captured.update(kwargs)
        return _FakeBook([])

    monkeypatch.setattr(probe.xlrd, "open_workbook", fake_open_workbook)
    parent, root, bindings, bin_path, _url = _fixture(tmp_path)
    result = probe.run_offline_child_structural_probe(production_state_parent=parent, output_root=root, bindings=bindings)
    assert result["status"] == "STRUCTURAL_FORMAT_CAPTURED"
    assert set(captured) == {"file_contents", "formatting_info", "on_demand", "ragged_rows"}
    assert captured["file_contents"] == bin_path.read_bytes()
    assert captured["formatting_info"] is True
    assert captured["on_demand"] is False
    assert captured["ragged_rows"] is False


def test_default_inspector_genuine_ole_biff_fixture_is_captured_and_deterministic() -> None:
    # Genuine OLE/BIFF xlrd integration coverage (MEDIUM-1 remediation):
    # reads the already-committed synthetic legacy .xls fixture -- no xlwt
    # import required -- verifies its committed identity against the
    # existing repository fixture-identity constant, then proves the real
    # xlrd-based parser captures it and is deterministic across repeated
    # calls (design section 7).
    fixture_bytes = synthetic_xls_fixture.FIXTURE_PATH.read_bytes()
    assert sha256(fixture_bytes).hexdigest() == synthetic_xls_fixture.EXPECTED_FIXTURE_SHA256
    first = probe._default_structural_inspector(fixture_bytes)
    assert first["status"] == "STRUCTURAL_FORMAT_CAPTURED"
    assert first["container_format"] == "OLE_COMPOUND_FILE"
    assert first["open_parse_status"] == "OPEN_PARSE_OK"
    second = probe._default_structural_inspector(fixture_bytes)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    # Cross-validated by the real safe-evidence validator too, not merely
    # shaped like a valid payload.
    assert probe._safe_structural_evidence(first) == first


def test_default_inspector_covers_all_seven_cell_type_buckets(monkeypatch: pytest.MonkeyPatch) -> None:
    # design section 5.10: exercises XL_CELL_TEXT/NUMBER/DATE/BOOLEAN/ERROR/
    # BLANK/EMPTY -- every entry in _CELL_TYPE_COUNT_KEYS -- via a
    # deterministic fake Book/Sheet whose cell_type(row, col) mechanically
    # supplies each of xlrd's own seven documented type codes (MEDIUM-1
    # remediation: no xlwt-built workbook is used).
    grid = [[
        probe.xlrd.XL_CELL_TEXT, probe.xlrd.XL_CELL_NUMBER, probe.xlrd.XL_CELL_DATE,
        probe.xlrd.XL_CELL_BOOLEAN, probe.xlrd.XL_CELL_ERROR, probe.xlrd.XL_CELL_BLANK,
        probe.xlrd.XL_CELL_EMPTY,
    ]]
    fake_sheet = _FakeSheet(nrows=1, ncols=7, visibility=0, cell_type_grid=grid)
    monkeypatch.setattr(probe.xlrd, "open_workbook", lambda **_kwargs: _FakeBook([fake_sheet]))
    result = probe._default_structural_inspector(b"irrelevant-bytes-open-is-mocked")
    assert result["status"] == "STRUCTURAL_FORMAT_CAPTURED"
    profiles = {item["column_ordinal"]: item["cell_type_counts"] for item in result["cell_type_profiles"]}
    assert profiles[1]["TEXT"] == 1
    assert profiles[2]["NUMBER"] == 1
    assert profiles[3]["DATE"] == 1
    assert profiles[4]["BOOLEAN"] == 1
    assert profiles[5]["ERROR"] == 1
    assert profiles[6]["BLANK"] == 1
    assert profiles[7]["EMPTY"] == 1
    dims = result["structural_dimensions"][0]
    assert dims["row_count"] == 1
    assert dims["column_count"] == 7
    # Cross-validated by the real safe-evidence validator too, not merely
    # shaped like a valid payload.
    assert probe._safe_structural_evidence(result) == result


def test_default_inspector_nrows_ncols_exact_with_blanks(monkeypatch: pytest.MonkeyPatch) -> None:
    # design section 5.9: row_count/column_count must equal xlrd's own
    # sheet.nrows/sheet.ncols exactly, and section 5.10's per-cell typing
    # must distinguish EMPTY/BLANK/NUMBER purely from cell_type() codes,
    # never a value access -- a deterministic fake Book/Sheet cell-type
    # matrix proves both without building a real workbook.
    grid = [
        [probe.xlrd.XL_CELL_TEXT, probe.xlrd.XL_CELL_EMPTY],
        [probe.xlrd.XL_CELL_TEXT, probe.xlrd.XL_CELL_BLANK],
        [probe.xlrd.XL_CELL_TEXT, probe.xlrd.XL_CELL_NUMBER],
    ]
    fake_sheet = _FakeSheet(nrows=3, ncols=2, visibility=0, cell_type_grid=grid)
    monkeypatch.setattr(probe.xlrd, "open_workbook", lambda **_kwargs: _FakeBook([fake_sheet]))
    result = probe._default_structural_inspector(b"irrelevant-bytes-open-is-mocked")
    dims = result["structural_dimensions"][0]
    assert dims["row_count"] == 3
    assert dims["column_count"] == 2
    profiles = {item["column_ordinal"]: item["cell_type_counts"] for item in result["cell_type_profiles"]}
    assert profiles[1]["TEXT"] == 3
    assert profiles[2]["EMPTY"] == 1
    assert profiles[2]["BLANK"] == 1
    assert profiles[2]["NUMBER"] == 1
    assert sum(profiles[1].values()) == 3
    assert sum(profiles[2].values()) == 3
    assert probe._safe_structural_evidence(result) == result


def test_default_inspector_visibility_zero_one_two_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    # design section 5.2: xlrd Sheet.visibility 0/1/2 must map exactly to
    # VISIBLE/HIDDEN/VERY_HIDDEN -- proved via deterministic fake sheets
    # rather than an xlwt-built workbook (MEDIUM-1 remediation).
    single_cell_grid = [[probe.xlrd.XL_CELL_TEXT]]
    sheets = [
        _FakeSheet(nrows=1, ncols=1, visibility=0, cell_type_grid=single_cell_grid),
        _FakeSheet(nrows=1, ncols=1, visibility=1, cell_type_grid=single_cell_grid),
        _FakeSheet(nrows=1, ncols=1, visibility=2, cell_type_grid=single_cell_grid),
    ]
    monkeypatch.setattr(probe.xlrd, "open_workbook", lambda **_kwargs: _FakeBook(sheets))
    result = probe._default_structural_inspector(b"irrelevant-bytes-open-is-mocked")
    visibility_by_ordinal = {item["ordinal"]: item["visibility"] for item in result["structural_dimensions"]}
    assert visibility_by_ordinal == {1: "VISIBLE", 2: "HIDDEN", 3: "VERY_HIDDEN"}
    assert probe._safe_structural_evidence(result) == result


def test_default_inspector_invalid_visibility_is_implementation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # design section 5.2: an unrecognized xlrd sheet.visibility code must
    # fail closed to IMPLEMENTATION_FAILURE, never a guessed UNKNOWN. No
    # valid BIFF file can carry this, so a fake Book/Sheet is used.
    fake_book = _FakeBook([_FakeSheet(nrows=0, ncols=0, visibility=3, cell_type_code=0)])
    monkeypatch.setattr(probe.xlrd, "open_workbook", lambda **_kwargs: fake_book)
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.run_offline_child_structural_probe(production_state_parent=parent, output_root=root, bindings=bindings)
    assert raised.value.outcome == "IMPLEMENTATION_FAILURE"
    assert raised.value.raw_bytes_read_for_integrity is True
    assert raised.value.child_content_inspected is True


def test_default_inspector_unknown_cell_type_is_implementation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # design section 5.10: an unrecognized xlrd cell_type() code must fail
    # closed to IMPLEMENTATION_FAILURE. No valid BIFF file can carry this,
    # so a fake Book/Sheet is used.
    fake_book = _FakeBook([_FakeSheet(nrows=1, ncols=1, visibility=0, cell_type_code=99)])
    monkeypatch.setattr(probe.xlrd, "open_workbook", lambda **_kwargs: fake_book)
    parent, root, bindings, _bin, _url = _fixture(tmp_path)
    with pytest.raises(probe.ProbeBlocked) as raised:
        probe.run_offline_child_structural_probe(production_state_parent=parent, output_root=root, bindings=bindings)
    assert raised.value.outcome == "IMPLEMENTATION_FAILURE"
    assert raised.value.raw_bytes_read_for_integrity is True
    assert raised.value.child_content_inspected is True
