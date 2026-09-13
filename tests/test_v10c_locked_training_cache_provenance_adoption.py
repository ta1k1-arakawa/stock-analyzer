from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

import src.v10c_locked_training_cache_provenance_adoption as v10c


TICKERS = [f"T{i:03d}" for i in range(v10c.CANDIDATE_TICKER_COUNT)]


def _audit_entry(ticker: str, *, accepted: bool) -> dict[str, object]:
    body = b"not-json-" + ticker.encode("ascii")
    return {
        "ticker": ticker,
        "attempt": 1,
        "scheme": v10c.YAHOO_SCHEME,
        "host": v10c.YAHOO_HOST,
        "path": f"{v10c.YAHOO_PATH_PREFIX}{ticker}.T",
        "query_specification": v10c._query_as_lists(),
        "status": 200 if accepted else 404,
        "error_type": None if accepted else "HTTP_ERROR",
        "redirect_detected": False,
        "body_byte_count": len(body) if accepted else 0,
        "payload_sha256": v10c.sha256_bytes(body) if accepted else None,
        "retry": False,
        "final": True,
        "success": accepted,
    }


def _attempt_receipt() -> dict[str, object]:
    return {
        "schema_version": v10c.ATTEMPT_RECEIPT_SCHEMA,
        "study_identity": "V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR",
        "frozen_design_git_commit": "1260538c5bef899478806f74ae32d9e4be7b023b",
        "frozen_design_git_blob_sha": "bc57720b6e73cc8c4cf793258a98f78a635af783",
        "freeze_approval_git_blob_sha": "dde0418589088932559b289b3ed26a40e29b62b6",
        "implementation_sha": v10c.V10B_ACQUISITION_IMPLEMENTATION_SHA,
        "attempt_started": True,
        "network_acquisition_scope": "V10B_PUBLIC_YAHOO_ONLY",
    }


@pytest.fixture
def synthetic_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    accepted = TICKERS[: v10c.CANDIDATE_SUCCESS_COUNT]
    failed = TICKERS[v10c.CANDIDATE_SUCCESS_COUNT :]
    payloads: list[dict[str, object]] = []
    for ticker in accepted:
        body = b"not-json-" + ticker.encode("ascii")
        payloads.append(
            {
                "ticker": ticker,
                "relative_path": f"{v10c.LOCKED_RAW_DIRECTORY}/{ticker}.json",
                "sha256": v10c.sha256_bytes(body),
                "byte_count": len(body),
            }
        )
    audit = [_audit_entry(ticker, accepted=ticker in accepted) for ticker in TICKERS]
    manifest = {
        "schema_version": v10c.MANIFEST_SCHEMA,
        "complete": True,
        "universe_mode": "FIXED_V4_300",
        "universe_csv_sha256": v10c.UNIVERSE_CSV_SHA256,
        "ticker_list_sha256": v10c.TICKER_LIST_SHA256,
        "ticker_count": v10c.CANDIDATE_TICKER_COUNT,
        "ticker_order": TICKERS,
        "price_from": v10c.PRICE_FROM,
        "price_to": v10c.PRICE_TO,
        "query_specification": v10c._query_as_lists(),
        "payloads": payloads,
        "network_audit": audit,
        "successful_ticker_count": len(payloads),
        "failed_tickers": failed,
        "payload_hash_list_sha256": v10c.sha256_bytes(v10c.canonical_json_bytes(payloads)),
    }
    root = tmp_path / "candidate"
    locked_raw = root / v10c.LOCKED_RAW_DIRECTORY
    locked_raw.mkdir(parents=True)
    for item in payloads:
        (root / item["relative_path"]).write_bytes(b"not-json-" + item["ticker"].encode("ascii"))
    manifest_raw = v10c.canonical_json_bytes(manifest)
    receipt = _attempt_receipt()
    receipt_raw = v10c.canonical_json_bytes(receipt)
    root.mkdir(exist_ok=True)
    (root / v10c.MANIFEST_FILE).write_bytes(manifest_raw)
    (root / v10c.ATTEMPT_RECEIPT_FILE).write_bytes(receipt_raw)
    monkeypatch.setattr(v10c, "CANDIDATE_MANIFEST_SHA256", v10c.sha256_bytes(manifest_raw))
    monkeypatch.setattr(v10c, "CANDIDATE_ATTEMPT_RECEIPT_SHA256", v10c.sha256_bytes(receipt_raw))
    return {"root": root, "manifest": manifest, "order": TICKERS, "payloads": payloads, "accepted": accepted, "failed": failed}


def _marker(path: Path, implementation_sha: str, manifest_sha: str) -> None:
    marker = {
        "schema_version": v10c.AUTHORIZATION_MARKER_SCHEMA,
        "study": v10c.STUDY_IDENTITY,
        "authorization_scope": v10c.AUTHORIZATION_SCOPE,
        "reviewed_v10c_implementation_sha": implementation_sha,
        "candidate_manifest_sha256": manifest_sha,
        "human_authorization_confirmed": True,
    }
    path.write_bytes(v10c.canonical_json_bytes(marker))


def test_valid_synthetic_closure_pass_and_300_success_not_required(synthetic_candidate: dict[str, object], tmp_path: Path) -> None:
    root = synthetic_candidate["root"]
    order = synthetic_candidate["order"]
    manifest = v10c.validate_candidate_metadata(root, order)
    assert manifest["complete"] is True
    assert manifest["successful_ticker_count"] == 283
    assert len(manifest["failed_tickers"]) == 17
    assert manifest["successful_ticker_count"] + len(manifest["failed_tickers"]) == 300
    marker = tmp_path / "marker.json"
    _marker(marker, "a" * 40, v10c.CANDIDATE_MANIFEST_SHA256)
    receipt = v10c.phase_b_offline_adoption(root, marker, "a" * 40, order)
    assert receipt["execution_result"] == "PASS"
    assert receipt["semantic_payload_parsing"] is False


def test_phase_a_reads_zero_locked_raw_bytes(synthetic_candidate: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    root = synthetic_candidate["root"]
    observed: list[str] = []
    original = Path.read_bytes

    def recording(path: Path) -> bytes:
        observed.append(str(path))
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", recording)
    v10c.validate_candidate_metadata(root, synthetic_candidate["order"])
    assert not any(v10c.LOCKED_RAW_DIRECTORY in path for path in observed)


def test_phase_b_authorization_precedes_payload_read(synthetic_candidate: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = synthetic_candidate["root"]
    observed: list[str] = []
    original = Path.read_bytes

    def recording(path: Path) -> bytes:
        observed.append(str(path))
        if v10c.LOCKED_RAW_DIRECTORY in str(path):
            raise AssertionError("locked payload read before authorization")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", recording)
    with pytest.raises(v10c.GovernanceProvenanceFailure):
        v10c.phase_b_offline_adoption(root, tmp_path / "missing-marker.json", "a" * 40, synthetic_candidate["order"])
    assert not any(v10c.LOCKED_RAW_DIRECTORY in path for path in observed)


def test_hash_only_closure_does_not_parse_payload_json(synthetic_candidate: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    marker = tmp_path / "marker.json"
    _marker(marker, "a" * 40, v10c.CANDIDATE_MANIFEST_SHA256)
    original_loads = v10c.json.loads

    def guarded(value: object, *args: object, **kwargs: object) -> object:
        assert value != "not-json"
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(v10c.json, "loads", guarded)
    receipt = v10c.phase_b_offline_adoption(synthetic_candidate["root"], marker, "a" * 40, synthetic_candidate["order"])
    assert receipt["locked_payload_hash_closure"] == "PASS"


def test_payload_hash_list_uses_canonical_ordered_entry_json(synthetic_candidate: dict[str, object]) -> None:
    manifest = synthetic_candidate["manifest"]
    assert manifest["payload_hash_list_sha256"] == v10c.sha256_bytes(v10c.canonical_json_bytes(manifest["payloads"]))


@pytest.mark.parametrize("field", ["schema_version", "complete", "ticker_order", "payload_hash_list_sha256"])
def test_manifest_mutation_rejected(synthetic_candidate: dict[str, object], field: str) -> None:
    manifest = copy.deepcopy(synthetic_candidate["manifest"])
    manifest[field] = "bad" if field != "complete" else False
    with pytest.raises(v10c.LockedArtifactIntegrityFailure):
        v10c.validate_manifest_structure(manifest, synthetic_candidate["order"])


def test_partition_and_count_mutations_rejected(synthetic_candidate: dict[str, object]) -> None:
    manifest = copy.deepcopy(synthetic_candidate["manifest"])
    manifest["failed_tickers"] = manifest["failed_tickers"][:-1]
    with pytest.raises(v10c.LockedArtifactIntegrityFailure):
        v10c.validate_manifest_structure(manifest, synthetic_candidate["order"])

    manifest = copy.deepcopy(synthetic_candidate["manifest"])
    manifest["successful_ticker_count"] = 300
    with pytest.raises(v10c.LockedArtifactIntegrityFailure):
        v10c.validate_manifest_structure(manifest, synthetic_candidate["order"])


def test_audit_invalid_rejected(synthetic_candidate: dict[str, object]) -> None:
    manifest = copy.deepcopy(synthetic_candidate["manifest"])
    manifest["network_audit"][0]["retry"] = True
    with pytest.raises(v10c.LockedArtifactIntegrityFailure):
        v10c.validate_manifest_structure(manifest, synthetic_candidate["order"])


def test_missing_extra_and_payload_metadata_failures(synthetic_candidate: dict[str, object]) -> None:
    root = synthetic_candidate["root"]
    manifest = v10c.validate_candidate_metadata(root, synthetic_candidate["order"])
    missing = root / v10c.LOCKED_RAW_DIRECTORY / f"{synthetic_candidate['accepted'][0]}.json"
    missing.unlink()
    with pytest.raises(v10c.LockedArtifactIntegrityFailure):
        v10c.validate_locked_payload_closure(root, manifest)

    extra = root / v10c.LOCKED_RAW_DIRECTORY / "EXTRA.json"
    extra.write_bytes(b"extra")
    with pytest.raises(v10c.LockedArtifactIntegrityFailure):
        v10c.validate_locked_payload_closure(root, manifest)


def test_payload_hash_and_byte_count_failures(synthetic_candidate: dict[str, object]) -> None:
    root = synthetic_candidate["root"]
    manifest = v10c.validate_candidate_metadata(root, synthetic_candidate["order"])
    path = root / manifest["payloads"][0]["relative_path"]
    path.write_bytes(b"tampered")
    with pytest.raises(v10c.LockedArtifactIntegrityFailure):
        v10c.validate_locked_payload_closure(root, manifest)


def test_metadata_hash_failures(synthetic_candidate: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v10c, "CANDIDATE_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(v10c.LockedArtifactIntegrityFailure):
        v10c.validate_candidate_metadata(synthetic_candidate["root"], synthetic_candidate["order"])


def test_attempt_receipt_hash_failure(synthetic_candidate: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(v10c, "CANDIDATE_ATTEMPT_RECEIPT_SHA256", "0" * 64)
    with pytest.raises(v10c.LockedArtifactIntegrityFailure):
        v10c.validate_candidate_metadata(synthetic_candidate["root"], synthetic_candidate["order"])


def test_terminal_predecessor_binding_is_fail_closed() -> None:
    bad = {"study": "wrong", "adjudication_result": "BLOCK"}
    with pytest.raises(v10c.GovernanceProvenanceFailure):
        v10c._validate_terminal_adjudication(bad)


def test_freeze_approval_mutation_is_rejected(tmp_path: Path) -> None:
    approval = json.loads(Path("V10C_DESIGN_FREEZE_APPROVAL.json").read_text(encoding="utf-8"))
    approval["approval_scope"] = "WRONG_SCOPE"
    (tmp_path / v10c.DESIGN_FREEZE_APPROVAL_FILE).write_bytes(v10c.canonical_json_bytes(approval))
    with pytest.raises(v10c.GovernanceProvenanceFailure):
        v10c._validate_freeze_approval(tmp_path)


def test_repository_provenance_mutations_are_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    approval = Path("V10C_DESIGN_FREEZE_APPROVAL.json").read_bytes()
    (tmp_path / v10c.DESIGN_FREEZE_APPROVAL_FILE).write_bytes(approval)
    (tmp_path / v10c.V10B_TERMINAL_ADJUDICATION_FILE).write_bytes(
        Path(v10c.V10B_TERMINAL_ADJUDICATION_FILE).read_bytes()
    )
    monkeypatch.setattr(v10c, "load_fixed_universe", lambda repo_root: TICKERS)
    original_git = v10c._git

    def fake_git(repo_root: Path, *args: str) -> str:
        if args == ("config", "--get", "remote.origin.url"):
            return v10c.EXPECTED_REPOSITORY_URL
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return v10c.AUTHORITATIVE_BRANCH
        if args == ("rev-parse", "HEAD") or args == ("rev-parse", f"origin/{v10c.AUTHORITATIVE_BRANCH}"):
            return "a" * 40
        if args == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        if args == ("rev-parse", f"HEAD:{v10c.DESIGN_FILE}"):
            return v10c.FROZEN_V10C_DESIGN_BLOB
        if args == ("rev-parse", f"HEAD:{v10c.DESIGN_FREEZE_APPROVAL_FILE}"):
            return v10c.DESIGN_FREEZE_APPROVAL_BLOB
        if args == ("merge-base", "--is-ancestor", v10c.FROZEN_V10C_DESIGN_COMMIT, "HEAD"):
            return ""
        if args == ("merge-base", "--is-ancestor", v10c.DESIGN_FREEZE_APPROVAL_COMMIT, "HEAD"):
            return ""
        if args == ("rev-parse", f"HEAD:{v10c.V10B_SOURCE_FILE}"):
            return v10c.V10B_SOURCE_BLOB
        if args == ("rev-parse", f"HEAD:{v10c.V10B_TERMINAL_ADJUDICATION_FILE}"):
            return v10c.V10B_TERMINAL_ADJUDICATION_BLOB
        if args == ("merge-base", "--is-ancestor", v10c.V10B_TERMINAL_ADJUDICATION_COMMIT, "HEAD"):
            return ""
        return original_git(repo_root, *args)

    monkeypatch.setattr(v10c, "_git", fake_git)
    assert v10c.validate_repository_preflight(tmp_path, "a" * 40) == TICKERS

    def wrong_design(repo_root: Path, *args: str) -> str:
        if args == ("rev-parse", f"HEAD:{v10c.DESIGN_FILE}"):
            return "0" * 40
        return fake_git(repo_root, *args)

    monkeypatch.setattr(v10c, "_git", wrong_design)
    with pytest.raises(v10c.GovernanceProvenanceFailure):
        v10c.validate_repository_preflight(tmp_path, "a" * 40)


def test_marker_mismatch_is_governance_failure(synthetic_candidate: dict[str, object], tmp_path: Path) -> None:
    marker = tmp_path / "marker.json"
    _marker(marker, "b" * 40, v10c.CANDIDATE_MANIFEST_SHA256)
    with pytest.raises(v10c.GovernanceProvenanceFailure):
        v10c.validate_authorization_marker(marker, "a" * 40)


def test_safe_receipt_is_bounded_and_deterministic(synthetic_candidate: dict[str, object], tmp_path: Path) -> None:
    manifest = v10c.validate_candidate_metadata(synthetic_candidate["root"], synthetic_candidate["order"])
    first = v10c.build_safe_receipt("a" * 40, manifest, authorization_consumed=True)
    second = v10c.build_safe_receipt("a" * 40, manifest, authorization_consumed=True)
    assert v10c.canonical_json_bytes(first) == v10c.canonical_json_bytes(second)
    rendered = json.dumps(first, sort_keys=True)
    assert str(synthetic_candidate["root"]) not in rendered
    assert "T000" not in rendered
    assert "not-json" not in rendered
    assert "human_authorization" not in rendered


def test_candidate_root_symlink_is_rejected_when_supported(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")
    with pytest.raises(v10c.GovernanceProvenanceFailure):
        v10c.validate_candidate_metadata(link, TICKERS)


def test_no_network_or_payload_parser_in_production_source() -> None:
    source = Path("src/v10c_locked_training_cache_provenance_adoption.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported = {
        (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
        for node in imports
        for alias in getattr(node, "names", [])
    }
    assert "requests" not in imported
    assert "urllib" in imported
    assert "urlopen" not in source
    assert "parse_v4" not in source


def test_fixed_provenance_constants_are_not_v10b_identity() -> None:
    assert v10c.STUDY_IDENTITY == "V10C_LOCKED_TRAINING_CACHE_PROVENANCE_ADOPTION_SUCCESSOR"
    assert v10c.CANDIDATE_MANIFEST_SHA256 != "72ae3db1186f2c9c113b1bafe1d37fb74a5627ac7ceed1dfc2473a24e060de85"
