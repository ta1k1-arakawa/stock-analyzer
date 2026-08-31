from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from src import v9_006_f1_semantic_successor_public_acquisition as acq
from src import v9_006_f1_terminal_month_structure_evidence as diagnostic
from src import v9_006_f1_terminal_month_structure_evidence_production as production
from src.v9_005_stage_a_jpx_probe import LISTED_ISSUES_PAGE_URL


DIAG_IMPL = "a" * 40
ROOT_URL = LISTED_ISSUES_PAGE_URL
TERM_URL = ROOT_URL.rsplit("/", 1)[0] + "/synthetic.xls"
ROOT_BYTES = b"root"
TERM_BYTES = b"terminal"


def _fake_bindings(_implementation: str, _repo: Path) -> None:
    return None


def _safe_acquisition(root_lock: acq.VerifiedLock, terminal_lock: acq.VerifiedLock) -> dict[str, object]:
    value = acq._base(
        diagnostic.ACQUISITION_IMPLEMENTATION_GIT_SHA,
        "SUCCESS",
        "NONE",
        root_lock,
        terminal_lock,
        root_attempts=1,
        terminal_attempts=1,
        locator_result="SUCCESSOR_LOCATOR_MATCHED",
        locator_hash="b" * 64,
    )
    return acq.finalize_safe_result(value)


def _state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, dict[str, object]]:
    monkeypatch.setattr(diagnostic, "TERMINAL_PAYLOAD_SHA256", sha256(TERM_BYTES).hexdigest())
    monkeypatch.setattr(diagnostic, "TERMINAL_BYTE_LENGTH", len(TERM_BYTES))
    root_lock = acq.VerifiedLock(acq.ROOT_FAMILY, acq.ROOT_PERIOD, 200, sha256(ROOT_BYTES).hexdigest(), len(ROOT_BYTES), ROOT_URL)
    terminal_lock = acq.VerifiedLock(acq.ROOT_FAMILY, acq.TERMINAL_PERIOD, 200, sha256(TERM_BYTES).hexdigest(), len(TERM_BYTES), TERM_URL)
    monkeypatch.setattr(diagnostic, "RAW_LOCK_SET_SHA256", acq.raw_lock_set_sha256(root_lock, terminal_lock))
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    state_root = production.derive_state_root(repo_root)
    (state_root / "raw" / acq.ROOT_PERIOD).mkdir(parents=True)
    (state_root / "raw" / acq.TERMINAL_PERIOD).mkdir(parents=True)
    for period, payload, lock in ((acq.ROOT_PERIOD, ROOT_BYTES, root_lock), (acq.TERMINAL_PERIOD, TERM_BYTES, terminal_lock)):
        directory = state_root / "raw" / period
        (directory / "payload.bin").write_bytes(payload)
        metadata = {
            "source_family": lock.source_family, "applicable_period": lock.applicable_period,
            "http_status": lock.http_status, "byte_length": lock.byte_length,
            "payload_sha256": lock.payload_sha256, "resolved_url": lock.resolved_url,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    safe = _safe_acquisition(root_lock, terminal_lock)
    (state_root / "safe-result.json").write_text(acq.canonical_json(safe), encoding="utf-8")
    return repo_root, state_root, safe


def _profile(_raw: bytes) -> dict[str, object]:
    counts = [{key: 0 for key in diagnostic.schema._CELL_TYPES}]
    counts[0]["TEXT"] = 1
    return {
        "container_format": diagnostic.schema.FORMAT_OLE_BIFF,
        "sheet_count": 1,
        "sheets": [{"sheet_ordinal": 1, "visibility": "VISIBLE", "row_count": 1, "column_count": 1, "column_cell_type_counts": counts, "sheet_name_date_text": "January 2026", "sheet_name_was_redacted": False}],
        "text_neighborhood": [{"sheet_ordinal": 1, "row_ordinal": 1, "cells": [{"column_ordinal": 1, "cell_type": "TEXT", "text": "January 2026"}]}],
        "neighborhood_truncated": False,
    }


def _rewrite_metadata(state_root: Path, period: str, **changes: object) -> None:
    path = state_root / "raw" / period / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(changes)
    path.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def test_imports_actual_new_production_module():
    assert production.__name__ == "src.v9_006_f1_terminal_month_structure_evidence_production"


def test_check_bindings_enforces_reviewed_repository_contract():
    expected = "a" * 40
    values = {
        ("branch", "--show-current"): production.AUTHORITATIVE_BRANCH,
        ("rev-parse", "HEAD"): expected,
        ("status", "--porcelain"): "",
        ("rev-parse", f"HEAD:{production.DESIGN_PATH}"): production.DESIGN_BLOB,
        ("rev-parse", f"origin/{production.AUTHORITATIVE_BRANCH}"): expected,
    }

    def fake_git(_root: Path, *args: str) -> str:
        if args == ("fetch", "--no-tags", "origin", production.AUTHORITATIVE_BRANCH):
            return ""
        return values[args]

    production.check_bindings(expected, Path("."), git_output=fake_git)
    for key, replacement in [
        (("branch", "--show-current"), "other"),
        (("status", "--porcelain"), "dirty"),
        (("rev-parse", f"HEAD:{production.DESIGN_PATH}"), "bad"),
    ]:
        altered = dict(values)
        altered[key] = replacement

        def altered_git(_root: Path, *args: str, altered=altered) -> str:
            if args == ("fetch", "--no-tags", "origin", production.AUTHORITATIVE_BRANCH):
                return ""
            return altered[args]

        with pytest.raises(ValueError):
            production.check_bindings(expected, Path("."), git_output=altered_git)


def test_run_from_state_synthetic_success_uses_actual_validator_and_is_read_only(tmp_path, monkeypatch):
    repo_root, state_root, before = _state(tmp_path, monkeypatch)
    snapshot = {path.relative_to(state_root).as_posix(): path.read_bytes() for path in state_root.rglob("*") if path.is_file()}
    result = production.run_from_state(DIAG_IMPL, repo_root, binding_check=_fake_bindings, profiler=_profile)
    diagnostic.validate_safe_result(result)
    assert result["diagnostic_result"] == "EVIDENCE_CAPTURED"
    assert result["structural_evidence_sha256"] == diagnostic.structural_evidence_sha256(result)
    assert before["result"] == "SUCCESS"
    assert snapshot == {path.relative_to(state_root).as_posix(): path.read_bytes() for path in state_root.rglob("*") if path.is_file()}


def test_missing_or_malformed_acquisition_safe_result_fails_before_payload(tmp_path, monkeypatch):
    repo_root, state_root, _ = _state(tmp_path, monkeypatch)
    (state_root / "safe-result.json").unlink()
    result = production.run_from_state(DIAG_IMPL, repo_root, binding_check=_fake_bindings, profiler=lambda _: pytest.fail("payload read"))
    assert (result["diagnostic_result"], result["failure_stage"]) == ("INPUT_BINDING_FAILURE", "PRE_READ_BINDING")
    (state_root / "safe-result.json").write_text("{}", encoding="utf-8")
    result = production.run_from_state(DIAG_IMPL, repo_root, binding_check=_fake_bindings, profiler=lambda _: pytest.fail("payload read"))
    assert result["failure_stage"] == "PRE_READ_BINDING"


@pytest.mark.parametrize("mutation", ["terminal_hash", "terminal_length", "raw_lock_set", "missing_raw", "malformed_raw"])
def test_lock_and_raw_state_corruption_is_read_only_fail_closed(tmp_path, monkeypatch, mutation):
    repo_root, state_root, safe = _state(tmp_path, monkeypatch)
    if mutation == "terminal_hash":
        _rewrite_metadata(state_root, acq.TERMINAL_PERIOD, payload_sha256="0" * 64)
    elif mutation == "terminal_length":
        _rewrite_metadata(state_root, acq.TERMINAL_PERIOD, byte_length=99)
    elif mutation == "raw_lock_set":
        monkeypatch.setattr(acq, "raw_lock_set_sha256", lambda *_locks: "0" * 64)
    elif mutation == "missing_raw":
        (state_root / "raw" / acq.TERMINAL_PERIOD / "payload.bin").unlink()
    else:
        (state_root / "raw" / acq.TERMINAL_PERIOD / "metadata.json").write_text("not-json", encoding="utf-8")
    before = {path.relative_to(state_root).as_posix(): path.read_bytes() for path in state_root.rglob("*") if path.is_file()}
    result = production.run_from_state(DIAG_IMPL, repo_root, binding_check=_fake_bindings, profiler=lambda _: pytest.fail("payload read"))
    assert result["diagnostic_result"] == "INPUT_BINDING_FAILURE"
    assert before == {path.relative_to(state_root).as_posix(): path.read_bytes() for path in state_root.rglob("*") if path.is_file()}


def test_no_source_network_path_in_production_module():
    text = Path(production.__file__).read_text(encoding="utf-8")
    assert "urllib" not in text.lower() and "urlopen" not in text.lower() and "requests" not in text.lower()
