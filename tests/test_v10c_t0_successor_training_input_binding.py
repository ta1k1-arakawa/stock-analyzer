from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src import v10c_t0_successor_training_input_binding as binding
from src import v10c_locked_training_cache_provenance_adoption as adoption
from src import v10b_training_cache_reacquisition as v10b


def _codes() -> list[str]:
    return [f"T{i:03d}" for i in range(300)]


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _receipt() -> dict[str, object]:
    return {
        "schema_version": "V10B_ACQUISITION_ATTEMPT_RECEIPT_V1",
        "study_identity": "V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR",
        "frozen_design_git_commit": "1260538c5bef899478806f74ae32d9e4be7b023b",
        "frozen_design_git_blob_sha": "bc57720b6e73cc8c4cf793258a98f78a635af783",
        "freeze_approval_git_blob_sha": "dde0418589088932559b289b3ed26a40e29b62b6",
        "implementation_sha": "b2172723df28b3edfce386c77ee79ce38a716925",
        "attempt_started": True,
        "network_acquisition_scope": "V10B_PUBLIC_YAHOO_ONLY",
    }


def _audit(ticker: str, *, success: bool) -> dict[str, object]:
    body = b"x"
    return {
        "ticker": ticker,
        "attempt": 1,
        "scheme": "https",
        "host": "query1.finance.yahoo.com",
        "path": f"/v8/finance/chart/{ticker}.T",
        "query_specification": [[key, value] for key, value in v10b.QUERY_SPECIFICATION],
        "status": 200 if success else 404,
        "error_type": None if success else "HTTP_ERROR",
        "redirect_detected": False,
        "body_byte_count": 1 if success else 0,
        "payload_sha256": _sha(body) if success else None,
        "retry": False,
        "final": True,
        "success": success,
    }


def _training_manifest(codes: list[str]) -> dict[str, object]:
    body = b"x"
    payloads = [
        {"ticker": ticker, "relative_path": f"locked_raw/{ticker}.json", "sha256": _sha(body), "byte_count": 1}
        for ticker in codes[:283]
    ]
    return {
        "schema_version": "V10B_TRAINING_CACHE_MANIFEST_V1",
        "complete": True,
        "universe_mode": "FIXED_V4_300",
        "universe_csv_sha256": binding.UNIVERSE_CSV_SHA256,
        "ticker_list_sha256": binding.TICKER_LIST_SHA256,
        "ticker_count": 300,
        "ticker_order": codes,
        "price_from": "2015-01-01",
        "price_to": "2019-12-31",
        "query_specification": [[key, value] for key, value in v10b.QUERY_SPECIFICATION],
        "payloads": payloads,
        "network_audit": [_audit(ticker, success=ticker in codes[:283]) for ticker in codes],
        "successful_ticker_count": 283,
        "failed_tickers": codes[283:],
        "payload_hash_list_sha256": _sha(adoption.canonical_json_bytes(payloads)),
    }


def _evaluation_manifest(codes: list[str]) -> dict[str, object]:
    body = b"e"
    return {
        "schema_version": 1,
        "complete": True,
        "universe_mode": "FIXED_V4_300",
        "universe_csv_sha256": binding.UNIVERSE_CSV_SHA256,
        "ticker_list_sha256": binding.TICKER_LIST_SHA256,
        "ticker_count": 300,
        "ticker_order": codes,
        "payloads": [
            {"ticker": ticker, "relative_path": f"raw/{ticker}.json", "sha256": _sha(body), "byte_count": 1}
            for ticker in codes
        ],
    }


def _write_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, list[str]]:
    codes = _codes()
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    repo = tmp_path / "repo"
    training.mkdir()
    evaluation.mkdir()
    repo.mkdir()
    (training / "locked_raw").mkdir()
    (evaluation / "raw").mkdir()
    for ticker in codes[:283]:
        (training / "locked_raw" / f"{ticker}.json").write_bytes(b"x")
    for ticker in codes:
        (evaluation / "raw" / f"{ticker}.json").write_bytes(b"e")
    receipt_raw = adoption.canonical_json_bytes(_receipt())
    manifest_raw = adoption.canonical_json_bytes(_training_manifest(codes))
    evaluation_raw = adoption.canonical_json_bytes(_evaluation_manifest(codes))
    (training / "V10B_ACQUISITION_ATTEMPT_RECEIPT.json").write_bytes(receipt_raw)
    (training / "cache_manifest.json").write_bytes(manifest_raw)
    (evaluation / "cache_manifest.json").write_bytes(evaluation_raw)
    monkeypatch.setattr(binding, "_verify_successor_repo", lambda *_: None)
    monkeypatch.setattr(
        binding,
        "load_fixed_universe",
        lambda _path: pd.DataFrame({"ticker": codes, "market": ["X"] * 300, "industry": ["Y"] * 300}),
    )
    monkeypatch.setattr(binding, "ATTEMPT_RECEIPT_SHA256", _sha(receipt_raw))
    monkeypatch.setattr(binding, "TRAINING_MANIFEST_SHA256", _sha(manifest_raw))
    monkeypatch.setattr(binding, "EVALUATION_MANIFEST_SHA256", _sha(evaluation_raw))
    return training, evaluation, codes


def test_fixed_successor_identities_and_old_identity_rejection():
    assert binding.TRAINING_MANIFEST_SHA256 == "887c031a004f91a080fa53ab511711fff92c92527cb119878ab2c295ee13cd44"
    assert binding.OLD_TRAINING_MANIFEST_SHA256 != binding.TRAINING_MANIFEST_SHA256
    assert binding.EVALUATION_MANIFEST_SHA256 == "797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5"
    assert binding.DESIGN_COMMIT == "0e65b170caef9958c31b7efa6802aa0313571d57"
    assert binding.DESIGN_BLOB == "52572c53934f9de1a182a7f580df91e52086de9c"
    assert binding.ADOPTION_RECORD_BLOB == "7febddd4af7c82fe00ef7ba618403f4dcf8f6758"


def _mocked_successor_git_values() -> dict[tuple[str, ...], str]:
    return {
        ("remote", "get-url", "origin"): "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        ("rev-parse", "--abbrev-ref", "HEAD"): "v9-cross-sectional-close-auction-design",
        ("rev-parse", "HEAD"): "a" * 40,
        ("rev-parse", "refs/remotes/origin/v9-cross-sectional-close-auction-design"): "a" * 40,
        ("status", "--porcelain", "--untracked-files=all"): "",
        ("rev-parse", "HEAD:V10C_T0_SUCCESSOR_TRAINING_INPUT_BINDING_DESIGN_DRAFT.md"): "52572c53934f9de1a182a7f580df91e52086de9c",
        ("rev-parse", "HEAD:V10A_T0_CALENDAR_INPUT_BINDING_BRIDGE_DESIGN_DRAFT.md"): "6df95aa8354c3d335a51747ee98ed9f2741c2410",
        ("rev-parse", "HEAD:src/v10b_training_cache_reacquisition.py"): "abb17129870241f18eba2f31a49c5606475a1a0d",
        ("rev-parse", "HEAD:src/v9_009_t0_top1_kill_screen.py"): "42753ddc75c6d7c016ac148991935bf316d9d14e",
        ("rev-parse", "HEAD:V10C_TRAINING_PROVENANCE_ADOPTION_RECORD.json"): "7febddd4af7c82fe00ef7ba618403f4dcf8f6758",
        ("rev-parse", "HEAD:V10C_DESIGN_FREEZE_APPROVAL.json"): "4676eb87c10dfc47fc19be387bfcb2ca17ebc59d",
    }


def test_successor_repo_provenance_uses_independent_exact_design_literal(monkeypatch):
    values = _mocked_successor_git_values()
    monkeypatch.setattr(binding, "_git", lambda _root, *parts: values[parts])
    binding._verify_successor_repo(Path(__file__).resolve().parents[1], "a" * 40)


def test_old_v10c_adoption_design_blob_is_rejected_independently(monkeypatch):
    values = _mocked_successor_git_values()
    values[("rev-parse", "HEAD:V10C_T0_SUCCESSOR_TRAINING_INPUT_BINDING_DESIGN_DRAFT.md")] = "d4e6e9b15dfee423aabc0a4052e68e4970739319"
    monkeypatch.setattr(binding, "_git", lambda _root, *parts: values[parts])
    with pytest.raises(binding.SuccessorPreflightFailure, match="SUCCESSOR_PROVENANCE_BLOB_MISMATCH"):
        binding._verify_successor_repo(Path(__file__).resolve().parents[1], "a" * 40)


def test_phase_a_accepts_valid_metadata_and_reads_zero_payload_bytes(tmp_path, monkeypatch):
    training, evaluation, codes = _write_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(binding, "_read_payload_bytes", lambda _path: (_ for _ in ()).throw(AssertionError("payload read")))
    metadata = binding.phase_a_metadata_preflight(
        tmp_path / "repo", training, evaluation, tmp_path / "V4_UNIVERSE.csv", "a" * 40
    )
    assert metadata.training_manifest["successful_ticker_count"] == 283
    assert len(metadata.training_manifest["failed_tickers"]) == 17
    assert metadata.ticker_order == tuple(codes)


def test_phase_a_does_not_invoke_inherited_parser(tmp_path, monkeypatch):
    training, evaluation, _ = _write_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(binding, "_parse_chart_payload", lambda _body: (_ for _ in ()).throw(AssertionError("parse")))
    binding.phase_a_metadata_preflight(tmp_path / "repo", training, evaluation, tmp_path / "V4_UNIVERSE.csv", "a" * 40)


@pytest.mark.parametrize("field", ["schema_version", "complete", "successful_ticker_count", "failed_tickers"])
def test_training_metadata_contract_rejects_tampering(tmp_path, monkeypatch, field):
    training, evaluation, codes = _write_fixture(tmp_path, monkeypatch)
    manifest = _training_manifest(codes)
    manifest[field] = "wrong" if field == "schema_version" else (False if field == "complete" else 282)
    raw = adoption.canonical_json_bytes(manifest)
    (training / "cache_manifest.json").write_bytes(raw)
    monkeypatch.setattr(binding, "TRAINING_MANIFEST_SHA256", _sha(raw))
    with pytest.raises(binding.SuccessorPreflightFailure):
        binding.phase_a_metadata_preflight(tmp_path / "repo", training, evaluation, tmp_path / "V4_UNIVERSE.csv", "a" * 40)


def test_old_manifest_identity_is_rejected_even_if_payload_shape_is_valid(tmp_path, monkeypatch):
    training, evaluation, codes = _write_fixture(tmp_path, monkeypatch)
    manifest = _training_manifest(codes)
    raw = adoption.canonical_json_bytes(manifest)
    (training / "cache_manifest.json").write_bytes(raw)
    original_sha = binding._sha256
    monkeypatch.setattr(
        binding,
        "_sha256",
        lambda value: binding.OLD_TRAINING_MANIFEST_SHA256 if value == raw else original_sha(value),
    )
    with pytest.raises(binding.SuccessorPreflightFailure, match="OLD_TRAINING_MANIFEST_REJECTED"):
        binding.phase_a_metadata_preflight(tmp_path / "repo", training, evaluation, tmp_path / "V4_UNIVERSE.csv", "a" * 40)


def test_path_escape_and_unsafe_payload_entries_rejected():
    root = Path("C:/synthetic-training")
    with pytest.raises(binding.SuccessorPreflightFailure):
        binding._safe_child(root, "../outside.json", error=binding.SuccessorPreflightFailure)


def test_missing_extra_locked_files_rejected(tmp_path, monkeypatch):
    training, evaluation, _ = _write_fixture(tmp_path, monkeypatch)
    (training / "locked_raw" / "EXTRA.json").write_bytes(b"x")
    with pytest.raises(binding.SuccessorPreflightFailure, match="CACHE_PAYLOAD_FILE_SET_INVALID"):
        binding.phase_a_metadata_preflight(tmp_path / "repo", training, evaluation, tmp_path / "V4_UNIVERSE.csv", "a" * 40)


def test_symlink_candidate_or_payload_path_is_rejected(tmp_path, monkeypatch):
    training, evaluation, _ = _write_fixture(tmp_path, monkeypatch)
    link = training / "locked_raw" / "T000.json.link"
    try:
        link.symlink_to(training / "locked_raw" / "T000.json")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation unavailable")
    with pytest.raises(binding.SuccessorPreflightFailure):
        binding.phase_a_metadata_preflight(tmp_path / "repo", training, evaluation, tmp_path / "V4_UNIVERSE.csv", "a" * 40)


def test_full_payload_seam_hashes_before_inherited_parser(tmp_path, monkeypatch):
    payload = tmp_path / "T000.json"
    payload.write_bytes(b"payload")
    seen: list[bytes] = []
    monkeypatch.setattr(binding, "_parse_chart_payload", lambda body: (seen.append(body), ({}, {}))[1])
    frame, actions = binding._parse_payload_file(payload, _sha(b"payload"), len(b"payload"))
    assert seen == [b"payload"]
    assert frame == {}
    assert actions == {}


def test_full_loader_uses_reviewed_v10b_validator_and_inherited_parser(monkeypatch):
    source = Path(binding.__file__).read_text(encoding="utf-8")
    assert "v10b.validate_manifest(" in source
    assert "_parse_chart_payload(body)" in source


def test_no_network_or_refetch_surface_in_successor_module():
    source = Path(binding.__file__).read_text(encoding="utf-8")
    assert "urllib" not in source
    assert "requests" not in source
    assert "refetch" not in source.lower()
