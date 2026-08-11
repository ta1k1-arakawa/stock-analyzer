from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from pathlib import Path

import pytest

from src import v8b_data_quality_calibration as calib
from src import v8b_v5b_calibration_input_preflight as preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_COMMIT = "a" * 40
WRONG_COMMIT = "b" * 40


# ---------------------------------------------------------------------------
# Synthetic V5-B cache fixture builder. All fixtures below are temporary,
# synthetic, and never touch the real V5-B cache.
# ---------------------------------------------------------------------------


def _write_synthetic_cache(tmp_path: Path, *, count: int = 300, corrupt_index: int | None = None, corrupt_kind: str | None = None):
    root = tmp_path / "synthetic_v5b_cache"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)

    payload_records = []
    for index in range(count):
        ticker = f"T{index:04d}"
        content = json.dumps({"marker": index, "chart": {"result": [{"synthetic": True}]}}).encode("utf-8")
        (raw_dir / f"{ticker}.json").write_bytes(content)
        payload_records.append(
            {
                "ticker": ticker,
                "relative_path": f"raw/{ticker}.json",
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        )

    if corrupt_index is not None:
        ticker = f"T{corrupt_index:04d}"
        path = raw_dir / f"{ticker}.json"
        if corrupt_kind == "missing":
            path.unlink()
        elif corrupt_kind == "byte_count_mismatch":
            path.write_bytes(path.read_bytes() + b"EXTRA")
        elif corrupt_kind == "sha_mismatch":
            original = path.read_bytes()
            tampered = original[:-1] + (b"0" if original[-1:] != b"0" else b"1")
            assert len(tampered) == len(original)
            path.write_bytes(tampered)
        elif corrupt_kind == "symlink_escape":
            outside_target = tmp_path / "outside_secret.txt"
            outside_target.write_bytes(b"not part of the designated cache")
            path.unlink()
            path.symlink_to(outside_target)
        else:
            raise AssertionError(f"unknown corrupt_kind {corrupt_kind}")

    manifest = {
        "schema_version": 2,
        "complete": True,
        "usable_for_evaluation": True,
        "attempted_ticker_count": count,
        "success_count": count,
        "failed_count": 0,
        "ticker_count": count,
        "failed_tickers": [],
        "circuit_breaker_triggered": False,
        "request_start": "2019-01-01",
        "request_end": "2026-01-31",
        "payloads": payload_records,
    }
    manifest["payload_hash_list_sha256"] = calib._recompute_payload_hash_list_sha256(payload_records)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    (root / "cache_manifest.json").write_bytes(manifest_bytes)
    return root, manifest_bytes, manifest


def _patch_expected_hashes(monkeypatch, manifest_bytes: bytes, manifest: dict) -> None:
    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(manifest_bytes).hexdigest())
    monkeypatch.setattr(calib, "EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256", manifest["payload_hash_list_sha256"])


def _no_leakage(value) -> bool:
    blob = json.dumps(value, default=str)
    if "raw/" in blob:
        return False
    for index in range(300):
        if f"T{index:04d}" in blob:
            return False
    return True


# ---------------------------------------------------------------------------
# Synthetic Git repository fixture builder (finding 2). Never touches the
# real repository's Git state; always a throwaway `git init` under tmp_path.
# ---------------------------------------------------------------------------


def _run_git_or_fail(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    return completed


def _build_synthetic_repo(
    tmp_path: Path,
    *,
    mutate_relative_path: str | None = None,
    omit_relative_path: str | None = None,
) -> tuple[Path, str]:
    repo_root = tmp_path / "synthetic_repo"
    repo_root.mkdir()
    _run_git_or_fail(["init", "-q"], repo_root)
    _run_git_or_fail(["config", "user.email", "preflight-test@example.invalid"], repo_root)
    _run_git_or_fail(["config", "user.name", "Preflight Test"], repo_root)

    for relative_path in preflight._RELEVANT_IMPLEMENTATION_RELATIVE_PATHS:
        if relative_path == omit_relative_path:
            continue
        target = repo_root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"synthetic committed content for {relative_path}\n", encoding="utf-8")

    _run_git_or_fail(["add", "-A"], repo_root)
    _run_git_or_fail(["commit", "-q", "-m", "synthetic commit"], repo_root)
    actual_head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    if mutate_relative_path is not None:
        (repo_root / mutate_relative_path).write_text("mutated, uncommitted content\n", encoding="utf-8")

    return repo_root, actual_head


@pytest.fixture
def gated_head(tmp_path, monkeypatch) -> str:
    """A clean synthetic Git repository, with _REPO_ROOT monkeypatched to
    it, returning the exact commit that satisfies the Git-HEAD binding
    check (finding 2)."""
    repo_root, actual_head = _build_synthetic_repo(tmp_path)
    monkeypatch.setattr(preflight, "_REPO_ROOT", repo_root)
    return actual_head


def _pass_gate(**overrides):
    kwargs = {"confirmation": preflight.PREFLIGHT_GATE_CONFIRMATION}
    kwargs.update(overrides)
    return preflight.run_production_v5b_calibration_input_preflight(**kwargs)


# ---------------------------------------------------------------------------
# §1 Fixed production input / API surface (finding 1: no ungated bypass)
# ---------------------------------------------------------------------------


def test_fixed_cache_root_matches_declared_local_path():
    assert preflight.FIXED_V5B_CACHE_ROOT_WINDOWS_PATH == r"C:\taiki\hobbies\v5-b-evaluation-cache-retry1"
    assert str(preflight.V5B_CACHE_ROOT) == preflight.FIXED_V5B_CACHE_ROOT_WINDOWS_PATH


def test_production_entry_point_exposes_no_path_override_parameter():
    params = set(inspect.signature(preflight.run_production_v5b_calibration_input_preflight).parameters)
    assert params == {"confirmation", "implementation_git_commit"}


def test_old_ungated_public_entry_point_no_longer_exists():
    assert not hasattr(preflight, "run_v5b_calibration_input_preflight")
    assert "run_v5b_calibration_input_preflight" not in preflight.__all__
    assert "run_v5b_calibration_input_preflight" not in dir(preflight)


def test_private_filesystem_helper_is_not_exported():
    assert "_run_v5b_calibration_input_preflight_against_root" not in preflight.__all__
    assert "_verify_implementation_matches_repository_head" not in preflight.__all__


def test_no_exported_callable_accepts_arbitrary_filesystem_override():
    """Adversarial API-surface test (finding 1): nothing this module
    exports may be called with a cache_root/path/manifest_path/input_dir/
    dataset argument -- the only way to reach real filesystem I/O is the
    confirmation- and Git-HEAD-gated production entry point."""
    forbidden_param_names = {"cache_root", "path", "manifest_path", "input_dir", "dataset"}
    for name in preflight.__all__:
        candidate = getattr(preflight, name)
        if not callable(candidate) or inspect.isclass(candidate):
            continue
        params = set(inspect.signature(candidate).parameters)
        assert params.isdisjoint(forbidden_param_names), f"{name} exposes {params & forbidden_param_names}"


# ---------------------------------------------------------------------------
# §2 Human gate
# ---------------------------------------------------------------------------


def test_gate_token_exact_value():
    assert preflight.PREFLIGHT_GATE_CONFIRMATION == "V5B_CALIBRATION_INPUT_PREFLIGHT_GATE"


def test_wrong_gate_confirmation_blocks_before_touching_cache_root(monkeypatch):
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", Path("/definitely/does/not/exist/anywhere"))
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight.run_production_v5b_calibration_input_preflight(
            confirmation="NOT_THE_TOKEN", implementation_git_commit=VALID_COMMIT
        )
    assert excinfo.value.reason == preflight.PREFLIGHT_BLOCKER
    assert excinfo.value.detail == "PREFLIGHT_GATE_CONFIRMATION_REQUIRED"


def test_missing_gate_confirmation_blocks():
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight.run_production_v5b_calibration_input_preflight(
            confirmation="", implementation_git_commit=VALID_COMMIT
        )
    assert excinfo.value.detail == "PREFLIGHT_GATE_CONFIRMATION_REQUIRED"


def test_correct_gate_confirmation_reaches_synthetic_fixture(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    result = _pass_gate(implementation_git_commit=gated_head)
    assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# Finding 2: implementation_git_commit bound to actual repository Git HEAD
# ---------------------------------------------------------------------------


def test_verify_implementation_head_accepts_clean_matching_repo(tmp_path):
    repo_root, actual_head = _build_synthetic_repo(tmp_path)
    returned_head = preflight._verify_implementation_matches_repository_head(
        repo_root=repo_root, implementation_git_commit=actual_head
    )
    assert returned_head == actual_head


def test_verify_implementation_head_rejects_wrong_commit(tmp_path):
    repo_root, actual_head = _build_synthetic_repo(tmp_path)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight._verify_implementation_matches_repository_head(
            repo_root=repo_root, implementation_git_commit=WRONG_COMMIT
        )
    assert excinfo.value.detail == "IMPLEMENTATION_COMMIT_HEAD_MISMATCH"
    assert excinfo.value.reason == preflight.PREFLIGHT_BLOCKER


@pytest.mark.parametrize("relative_path", list(preflight._RELEVANT_IMPLEMENTATION_RELATIVE_PATHS))
def test_verify_implementation_head_rejects_dirty_relevant_file(tmp_path, relative_path):
    repo_root, actual_head = _build_synthetic_repo(tmp_path, mutate_relative_path=relative_path)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight._verify_implementation_matches_repository_head(
            repo_root=repo_root, implementation_git_commit=actual_head
        )
    assert excinfo.value.detail == "IMPLEMENTATION_FILE_DIRTY"


def test_verify_implementation_head_rejects_missing_committed_file(tmp_path):
    omitted = preflight._RELEVANT_IMPLEMENTATION_RELATIVE_PATHS[0]
    repo_root, actual_head = _build_synthetic_repo(tmp_path, omit_relative_path=omitted)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight._verify_implementation_matches_repository_head(
            repo_root=repo_root, implementation_git_commit=actual_head
        )
    assert excinfo.value.detail == "IMPLEMENTATION_FILE_UNVERIFIABLE"


def test_verify_implementation_head_rejects_unresolvable_git_state(tmp_path):
    not_a_repo = tmp_path / "not_a_git_repo"
    not_a_repo.mkdir()
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight._verify_implementation_matches_repository_head(
            repo_root=not_a_repo, implementation_git_commit=VALID_COMMIT
        )
    assert excinfo.value.detail == "GIT_HEAD_UNRESOLVABLE"


def test_wrong_git_head_blocks_before_cache_access_via_gated_entry_point(tmp_path, monkeypatch):
    repo_root, actual_head = _build_synthetic_repo(tmp_path)
    monkeypatch.setattr(preflight, "_REPO_ROOT", repo_root)
    # Cache root deliberately points nowhere: if the implementation ever
    # reached cache access, this would fail with a cache-specific reason
    # instead of the expected Git-HEAD-mismatch reason.
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", tmp_path / "never_reached")
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=WRONG_COMMIT)
    assert excinfo.value.detail == "IMPLEMENTATION_COMMIT_HEAD_MISMATCH"
    assert excinfo.value.result is not None
    assert "checked_payload_count" not in excinfo.value.result


def test_arbitrary_syntactically_valid_sha_cannot_be_recorded_as_provenance(tmp_path, monkeypatch):
    repo_root, actual_head = _build_synthetic_repo(tmp_path)
    monkeypatch.setattr(preflight, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", tmp_path / "never_reached")
    arbitrary_but_syntactically_valid_sha = "f" * 40
    assert arbitrary_but_syntactically_valid_sha != actual_head
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=arbitrary_but_syntactically_valid_sha)
    assert excinfo.value.detail == "IMPLEMENTATION_COMMIT_HEAD_MISMATCH"
    # No PASS result was ever produced, so no output anywhere claims this
    # arbitrary SHA as accepted provenance.
    assert excinfo.value.result["status"] == "BLOCK"


@pytest.mark.parametrize("relative_path", list(preflight._RELEVANT_IMPLEMENTATION_RELATIVE_PATHS))
def test_dirty_relevant_file_blocks_before_cache_access_via_gated_entry_point(tmp_path, monkeypatch, relative_path):
    repo_root, actual_head = _build_synthetic_repo(tmp_path, mutate_relative_path=relative_path)
    monkeypatch.setattr(preflight, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", tmp_path / "never_reached")
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=actual_head)
    assert excinfo.value.detail == "IMPLEMENTATION_FILE_DIRTY"
    assert "checked_payload_count" not in (excinfo.value.result or {})


def test_git_head_unresolvable_blocks_before_cache_access_via_gated_entry_point(tmp_path, monkeypatch):
    not_a_repo = tmp_path / "not_a_git_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(preflight, "_REPO_ROOT", not_a_repo)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", tmp_path / "never_reached")
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=VALID_COMMIT)
    assert excinfo.value.detail == "GIT_HEAD_UNRESOLVABLE"


# ---------------------------------------------------------------------------
# §5 input validation: genuine PASS, via the gated entry point
# ---------------------------------------------------------------------------


def test_genuine_synthetic_pass(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    result = _pass_gate(implementation_git_commit=gated_head)
    assert result["status"] == "PASS"
    assert result["detail_reason"] is None
    assert result["expected_payload_count"] == 300
    assert result["checked_payload_count"] == 300
    assert result["byte_count_mismatch_count"] == 0
    assert result["sha256_mismatch_count"] == 0
    assert result["missing_or_unreadable_count"] == 0
    assert result["observed_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert result["observed_payload_hash_list_sha256"] == manifest["payload_hash_list_sha256"]
    assert result["role"] == "R1_V5B_CALIBRATION_INPUT_PREFLIGHT"
    assert result["schema_version"] == "V5B_CALIBRATION_INPUT_PREFLIGHT_RESULT_V1"
    assert isinstance(result["artifact_self_hash"], str) and len(result["artifact_self_hash"]) == 64


def test_pass_result_never_parses_payload_body_as_json(tmp_path, monkeypatch, gated_head):
    # Payload content is deliberately NOT valid JSON. If the preflight ever
    # tried to json-parse payload bodies, this would raise inside it; it
    # must instead PASS purely on byte-length/SHA-256 binding.
    root = tmp_path / "synthetic_v5b_cache"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    payload_records = []
    for index in range(300):
        ticker = f"T{index:04d}"
        content = b"NOT { VALID JSON AT ALL ]]]" + str(index).encode("ascii")
        (raw_dir / f"{ticker}.json").write_bytes(content)
        payload_records.append(
            {
                "ticker": ticker,
                "relative_path": f"raw/{ticker}.json",
                "sha256": hashlib.sha256(content).hexdigest(),
                "byte_count": len(content),
            }
        )
    manifest = {
        "schema_version": 2,
        "complete": True,
        "usable_for_evaluation": True,
        "attempted_ticker_count": 300,
        "success_count": 300,
        "failed_count": 0,
        "ticker_count": 300,
        "failed_tickers": [],
        "circuit_breaker_triggered": False,
        "request_start": "2019-01-01",
        "request_end": "2026-01-31",
        "payloads": payload_records,
    }
    manifest["payload_hash_list_sha256"] = calib._recompute_payload_hash_list_sha256(payload_records)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    (root / "cache_manifest.json").write_bytes(manifest_bytes)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)

    result = _pass_gate(implementation_git_commit=gated_head)
    assert result["status"] == "PASS"


def test_pass_result_has_no_ticker_or_path_leakage(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    result = _pass_gate(implementation_git_commit=gated_head)
    assert _no_leakage(result)


# ---------------------------------------------------------------------------
# §5 / §7 adversarial byte-binding failures, via the gated entry point
# ---------------------------------------------------------------------------


def test_missing_manifest_blocks(tmp_path, monkeypatch, gated_head):
    root = tmp_path / "empty_cache"
    root.mkdir()
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.reason == preflight.PREFLIGHT_BLOCKER
    assert excinfo.value.detail == "MANIFEST_UNREADABLE"
    assert excinfo.value.result["status"] == "BLOCK"
    assert _no_leakage(excinfo.value.result)


def test_wrong_manifest_hash_blocks(tmp_path, monkeypatch, gated_head):
    # Deliberately do NOT patch the expected hash constants: the synthetic
    # manifest's real SHA-256 will never equal the frozen production pin.
    root, _manifest_bytes, _manifest = _write_synthetic_cache(tmp_path)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "MANIFEST_PROVENANCE_INVALID:MANIFEST_SHA256_MISMATCH"


def test_duplicate_key_manifest_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    duplicate_key_bytes = b'{"a": 1, "a": 2}'
    (root / "cache_manifest.json").write_bytes(duplicate_key_bytes)
    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(duplicate_key_bytes).hexdigest())
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "MANIFEST_PROVENANCE_INVALID:STRICT_JSON_DUPLICATE_KEY"


def test_malformed_json_manifest_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    malformed_bytes = b"{not valid json"
    (root / "cache_manifest.json").write_bytes(malformed_bytes)
    monkeypatch.setattr(calib, "EXPECTED_V5B_MANIFEST_SHA256", hashlib.sha256(malformed_bytes).hexdigest())
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "MANIFEST_PROVENANCE_INVALID:STRICT_JSON_MALFORMED"


def test_missing_payload_file_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, corrupt_index=7, corrupt_kind="missing")
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "PAYLOAD_BINDING_FAILED"
    result = excinfo.value.result
    assert result["missing_or_unreadable_count"] == 1
    assert result["checked_payload_count"] == 299
    assert _no_leakage(result)


def test_path_escape_via_symlink_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, corrupt_index=3, corrupt_kind="symlink_escape")
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "PAYLOAD_PATH_ESCAPE_DETECTED"
    assert excinfo.value.result is not None
    assert _no_leakage(excinfo.value.result)


def test_manifest_level_path_traversal_is_rejected_before_binding(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    payloads = list(manifest["payloads"])
    payloads[0] = dict(payloads[0], relative_path="raw/../escape.json")
    manifest = dict(manifest, payloads=payloads)
    manifest["payload_hash_list_sha256"] = calib._recompute_payload_hash_list_sha256(payloads)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    (root / "cache_manifest.json").write_bytes(manifest_bytes)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "MANIFEST_PROVENANCE_INVALID:MANIFEST_RELATIVE_PATH_INVALID"


def test_byte_count_mismatch_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, corrupt_index=11, corrupt_kind="byte_count_mismatch")
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "PAYLOAD_BINDING_FAILED"
    result = excinfo.value.result
    assert result["byte_count_mismatch_count"] == 1
    assert result["checked_payload_count"] == 300


def test_payload_sha_mismatch_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, corrupt_index=42, corrupt_kind="sha_mismatch")
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "PAYLOAD_BINDING_FAILED"
    result = excinfo.value.result
    assert result["sha256_mismatch_count"] == 1
    assert result["byte_count_mismatch_count"] == 0
    assert result["checked_payload_count"] == 300


@pytest.mark.parametrize("payload_count", [299, 301])
def test_designated_payload_count_mismatch_blocks(tmp_path, monkeypatch, gated_head, payload_count):
    # attempted_ticker_count / success_count / ticker_count stay at the
    # expected 300 so this exercises specifically the payload LIST length
    # check (not the earlier, unrelated ticker-count-field checks).
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, count=300)
    payloads = list(manifest["payloads"])
    if payload_count == 299:
        payloads = payloads[:-1]
    else:
        extra = dict(payloads[-1])
        extra["ticker"] = "T0300"
        extra["relative_path"] = "raw/T0300.json"
        (root / "raw" / "T0300.json").write_bytes((root / "raw" / "T0000.json").read_bytes())
        extra["sha256"] = payloads[0]["sha256"]
        extra["byte_count"] = payloads[0]["byte_count"]
        payloads = payloads + [extra]
    manifest = dict(manifest, payloads=payloads)
    manifest["payload_hash_list_sha256"] = calib._recompute_payload_hash_list_sha256(payloads)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    (root / "cache_manifest.json").write_bytes(manifest_bytes)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", root)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "MANIFEST_PROVENANCE_INVALID:MANIFEST_PAYLOAD_COUNT_MISMATCH"


def test_cache_root_missing_blocks(tmp_path, monkeypatch, gated_head):
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", tmp_path / "does_not_exist")
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "CACHE_ROOT_INACCESSIBLE"


def test_cache_root_not_a_directory_blocks(tmp_path, monkeypatch, gated_head):
    not_a_dir = tmp_path / "just_a_file"
    not_a_dir.write_text("x")
    monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", not_a_dir)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "CACHE_ROOT_NOT_A_DIRECTORY"


@pytest.mark.parametrize("bad_commit", ["", "not-hex", "A" * 40, "0" * 39, None, 12345])
def test_invalid_implementation_commit_format_blocks(bad_commit):
    # Format rejection happens before any Git or cache lookup, so no repo
    # or cache fixture is needed here.
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        _pass_gate(implementation_git_commit=bad_commit)
    assert excinfo.value.detail == "IMPLEMENTATION_COMMIT_INVALID"


# ---------------------------------------------------------------------------
# §4 forbidden actions -- source-level and functional checks
# ---------------------------------------------------------------------------


def _functional_source() -> str:
    source = (REPO_ROOT / "src" / "v8b_v5b_calibration_input_preflight.py").read_text(encoding="utf-8")
    # Exclude run_static_check()'s own body: it necessarily names these
    # tokens as literal strings in order to check for them elsewhere, and
    # would otherwise always self-match.
    return source[: source.index("\ndef run_static_check")]


def test_module_source_has_no_forbidden_parsing_or_calibration_calls():
    source = _functional_source()
    forbidden_calls = [
        "parse_ticker_observations(",
        "run_data_quality_calibration(",
        "_row_invalid_reason(",
        "select_synthetic_bases(",
        "compute_global_envelope(",
        "select_policy(",
        "apply_corruption(",
    ]
    for token in forbidden_calls:
        assert token not in source, f"forbidden call found: {token}"


def test_module_source_has_no_network_strings():
    source = _functional_source()
    forbidden = ["urllib", "requests", "yfinance", "query1.finance.yahoo.com", "http://", "https://"]
    for token in forbidden:
        assert token not in source, f"forbidden token found: {token}"


def test_module_reuses_existing_manifest_provenance_validator():
    source = (REPO_ROOT / "src" / "v8b_v5b_calibration_input_preflight.py").read_text(encoding="utf-8")
    assert "validate_v5b_manifest_provenance" in source
    assert "def validate_v5b_manifest_provenance" not in source


# ---------------------------------------------------------------------------
# §7 exhaustive-failure blocker constant
# ---------------------------------------------------------------------------


def test_blocker_constant_matches_calibration_run_validity_r1_reason():
    assert preflight.PREFLIGHT_BLOCKER == "V5B_CALIBRATION_INPUT_PREFLIGHT_BLOCKED"
    assert preflight.PREFLIGHT_BLOCKER in calib._RUN_INVALID_REASON_FLAGS


def test_all_blocked_exceptions_carry_the_single_generic_reason(tmp_path, monkeypatch, gated_head):
    for cache_root in (tmp_path / "missing", tmp_path):
        monkeypatch.setattr(preflight, "V5B_CACHE_ROOT", cache_root)
        with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
            _pass_gate(implementation_git_commit=gated_head)
        assert excinfo.value.reason == preflight.PREFLIGHT_BLOCKER


# ---------------------------------------------------------------------------
# §3 (LOW finding): run_static_check() is meaningful, not a no-op
# ---------------------------------------------------------------------------


def test_static_check_passes_cleanly_on_the_real_module():
    preflight.run_static_check()  # must not raise


def test_static_check_detects_cache_root_drift(monkeypatch):
    monkeypatch.setattr(preflight, "FIXED_V5B_CACHE_ROOT_WINDOWS_PATH", r"C:\somewhere\else")
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_CACHE_ROOT_DRIFT"


def test_static_check_detects_gate_token_drift(monkeypatch):
    monkeypatch.setattr(preflight, "PREFLIGHT_GATE_CONFIRMATION", "SOMETHING_ELSE")
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_GATE_TOKEN_DRIFT"


def test_static_check_detects_payload_count_drift(monkeypatch):
    monkeypatch.setattr(preflight, "EXPECTED_V5B_TICKER_COUNT", 299)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_PAYLOAD_COUNT_DRIFT"


def test_static_check_detects_manifest_validator_drift(monkeypatch):
    monkeypatch.setattr(preflight, "validate_v5b_manifest_provenance", lambda manifest_bytes: {})
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_MANIFEST_VALIDATOR_DRIFT"


def test_static_check_detects_reintroduced_ungated_export(monkeypatch):
    monkeypatch.setattr(preflight, "__all__", list(preflight.__all__) + ["run_v5b_calibration_input_preflight"])
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_UNGATED_FILESYSTEM_RUNNER_EXPORTED"


def test_static_check_detects_exported_function_with_cache_root_param(monkeypatch):
    def fake_runner(cache_root):  # pragma: no cover - never actually called
        return cache_root

    monkeypatch.setattr(preflight, "fake_runner", fake_runner, raising=False)
    monkeypatch.setattr(preflight, "__all__", list(preflight.__all__) + ["fake_runner"])
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_UNGATED_FILESYSTEM_RUNNER_EXPORTED"


def test_static_check_detects_production_api_surface_drift(monkeypatch):
    def fake_production(*, confirmation, implementation_git_commit, cache_root=None):
        raise AssertionError("never called")

    monkeypatch.setattr(preflight, "run_production_v5b_calibration_input_preflight", fake_production)
    with pytest.raises(preflight.V5BCalibrationInputPreflightBlocked) as excinfo:
        preflight.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_PRODUCTION_API_SURFACE_DRIFT"
