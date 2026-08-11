from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import v8b_data_quality_calibration as calib
from src import v8b_data_quality_calibration_execution as execution
from src import v8b_v5b_calibration_input_preflight as preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_COMMIT = "a" * 40
WRONG_COMMIT = "b" * 40


# ---------------------------------------------------------------------------
# Synthetic (trivial, non-Yahoo-shaped) V5-B cache fixture builder. All
# fixtures below are temporary and synthetic; none ever touches the real
# fixed V5-B cache root.
# ---------------------------------------------------------------------------


def _write_synthetic_cache(
    tmp_path: Path, *, count: int = 300, corrupt_index: int | None = None, corrupt_kind: str | None = None
):
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
            try:
                path.symlink_to(outside_target)
            except OSError as error:
                if getattr(error, "winerror", None) == 1314:
                    pytest.skip("Windows symlink creation requires unavailable privilege")
                raise
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
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
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
        if f"T{index:04d}" in blob or f"TICK{index:03d}" in blob:
            return False
    return True


# ---------------------------------------------------------------------------
# Full, Yahoo-chart-shaped synthetic cache: used only for the genuine
# reach-and-succeed tests (a real call into run_data_quality_calibration()).
# ---------------------------------------------------------------------------


def _epoch(day: date, hour: int = 0) -> int:
    return int(datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc).timestamp())


def _consecutive_days(start: date, count: int) -> list[date]:
    return [start + timedelta(days=index) for index in range(count)]


def _yahoo_payload_bytes(symbol: str, days: list[date]) -> bytes:
    n = len(days)
    quote = {
        "open": [100.0 + index for index in range(n)],
        "high": [101.0 + index for index in range(n)],
        "low": [99.0 + index for index in range(n)],
        "close": [100.5 + index for index in range(n)],
        "volume": [1000.0 + index for index in range(n)],
    }
    indicators = {"quote": [quote], "adjclose": [{"adjclose": [100.25 + index for index in range(n)]}]}
    body = {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"symbol": symbol},
                    "timestamp": [_epoch(day) for day in days],
                    "indicators": indicators,
                    "events": {},
                }
            ],
        }
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_synthetic_full_cache(tmp_path: Path):
    """300 canonical tickers TICK000..TICK299. First 20 (sorted) get a full
    clean 252-day run in 2019 so they qualify as synthetic bases; the rest
    get a short (5-day) valid window -- mirrors the calibration core's own
    full-pipeline integration fixture."""

    root = tmp_path / "synthetic_v5b_full_cache"
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)

    tickers = [f"TICK{index:03d}" for index in range(300)]
    payload_records = []
    for index, ticker in enumerate(tickers):
        if index < 20:
            days = _consecutive_days(date(2019, 1, 2), calib.SYNTHETIC_SEQUENCE_LENGTH)
        else:
            days = _consecutive_days(date(2019, 1, 2), 5)
        content = _yahoo_payload_bytes(f"{ticker}.T", days)
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
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    (root / "cache_manifest.json").write_bytes(manifest_bytes)
    return root, manifest_bytes, manifest


# ---------------------------------------------------------------------------
# Synthetic Git repository fixture builder. Never touches the real
# repository's Git state; always a throwaway `git init` under tmp_path.
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
    include_real_calibration_core_dependencies: bool = False,
) -> tuple[Path, str]:
    repo_root = tmp_path / "synthetic_repo"
    repo_root.mkdir()
    _run_git_or_fail(["init", "-q"], repo_root)
    _run_git_or_fail(["config", "user.email", "execution-test@example.invalid"], repo_root)
    _run_git_or_fail(["config", "user.name", "Execution Test"], repo_root)

    for relative_path in execution._RELEVANT_IMPLEMENTATION_RELATIVE_PATHS:
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

    if include_real_calibration_core_dependencies:
        # These three files are NOT part of _RELEVANT_IMPLEMENTATION_RELATIVE_
        # PATHS (they are separately, independently pinned by
        # verify_repository_contract() every time run_data_quality_calibration
        # runs). run_data_quality_calibration() reads them via plain
        # Path.read_bytes(), never through Git, so copying the real repo's
        # current bytes here (uncommitted) is sufficient for that check to
        # pass without affecting the synthetic Git-HEAD verification above.
        (repo_root / "src").mkdir(parents=True, exist_ok=True)
        (repo_root / calib.PREREGISTRATION_PATH).write_bytes((REPO_ROOT / calib.PREREGISTRATION_PATH).read_bytes())
        (repo_root / calib.APPROVAL_ARTIFACT_PATH).write_bytes((REPO_ROOT / calib.APPROVAL_ARTIFACT_PATH).read_bytes())
        (repo_root / calib.PINNED_COLLECTOR_PATH).write_bytes((REPO_ROOT / calib.PINNED_COLLECTOR_PATH).read_bytes())

    return repo_root, actual_head


@pytest.fixture
def gated_head(tmp_path, monkeypatch) -> str:
    """A clean synthetic Git repository, with _REPO_ROOT monkeypatched to
    it, returning the exact commit that satisfies the Git-HEAD binding
    check. Does NOT carry the real calibration-core dependency files, so if
    a test reaches run_data_quality_calibration() with this fixture, the
    core's own verify_repository_contract() will (correctly) block it."""
    repo_root, actual_head = _build_synthetic_repo(tmp_path)
    monkeypatch.setattr(execution, "_REPO_ROOT", repo_root)
    return actual_head


@pytest.fixture
def dependency_bound_head(tmp_path, monkeypatch) -> str:
    """Like `gated_head`, but additionally carries real copies of the three
    files verify_repository_contract() independently pins, so a genuine
    full run_data_quality_calibration() call can succeed all the way
    through."""
    repo_root, actual_head = _build_synthetic_repo(tmp_path, include_real_calibration_core_dependencies=True)
    monkeypatch.setattr(execution, "_REPO_ROOT", repo_root)
    return actual_head


def _pass_gate(**overrides):
    kwargs = {
        "confirmation": execution.EXECUTION_GATE_CONFIRMATION,
        "calibration_attempt_id": "test-attempt-0001",
    }
    kwargs.update(overrides)
    return execution.run_production_v8b_data_quality_calibration(**kwargs)


# ---------------------------------------------------------------------------
# Section 1: fixed production input / API surface -- no ungated bypass.
# ---------------------------------------------------------------------------


def test_fixed_cache_root_matches_declared_local_path():
    assert execution.FIXED_V5B_CACHE_ROOT_WINDOWS_PATH == r"C:\taiki\hobbies\v5-b-evaluation-cache-retry1"
    assert str(execution.V5B_CACHE_ROOT) == execution.FIXED_V5B_CACHE_ROOT_WINDOWS_PATH


def test_production_entry_point_exposes_no_path_override_parameter():
    params = set(inspect.signature(execution.run_production_v8b_data_quality_calibration).parameters)
    assert params == {"confirmation", "implementation_git_commit", "calibration_attempt_id"}


def test_no_module_level_name_for_the_nested_cache_walker():
    for name in (
        "_walk_and_execute",
        "run_v8b_data_quality_calibration",
        "read_verified_file",
        "verify_root",
        "is_reparse_point",
        "reject_reparse_components",
        "normalized_path",
        "is_within",
    ):
        assert not hasattr(execution, name)
        assert name not in dir(execution)


def _module_level_callables(module):
    for name in dir(module):
        if name.startswith("__"):
            continue
        candidate = getattr(module, name)
        if not callable(candidate) or inspect.isclass(candidate):
            continue
        if getattr(candidate, "__module__", None) != module.__name__:
            continue  # reused from elsewhere (calibration core / preflight), not this module's own surface
        yield name, candidate


def test_no_module_level_callable_accepts_arbitrary_filesystem_override():
    forbidden_param_names = {"cache_root", "path", "manifest_path", "input_dir", "dataset"}
    for name, candidate in _module_level_callables(execution):
        params = set(inspect.signature(candidate).parameters)
        assert params.isdisjoint(forbidden_param_names), f"{name} exposes {params & forbidden_param_names}"


def test_only_production_entry_point_is_filesystem_capable_by_name():
    v5b_cache_related_names = {"cache_root", "manifest_path", "path", "input_dir", "dataset"}
    for name, candidate in _module_level_callables(execution):
        if name == "run_production_v8b_data_quality_calibration":
            continue
        params = set(inspect.signature(candidate).parameters)
        assert params.isdisjoint(v5b_cache_related_names), f"{name} exposes {params & v5b_cache_related_names}"


def test_module_reuses_preflight_git_verifier_not_reimplemented():
    source = (REPO_ROOT / "src" / "v8b_data_quality_calibration_execution.py").read_text(encoding="utf-8")
    assert "_verify_implementation_matches_repository_head" in source
    assert "def _verify_implementation_matches_repository_head" not in source


def test_module_reuses_existing_manifest_provenance_validator():
    source = (REPO_ROOT / "src" / "v8b_data_quality_calibration_execution.py").read_text(encoding="utf-8")
    assert "validate_v5b_manifest_provenance" in source
    assert "def validate_v5b_manifest_provenance" not in source


def test_module_reuses_existing_calibration_entry_point():
    source = (REPO_ROOT / "src" / "v8b_data_quality_calibration_execution.py").read_text(encoding="utf-8")
    assert "def run_data_quality_calibration" not in source


# ---------------------------------------------------------------------------
# Section 2: human gate.
# ---------------------------------------------------------------------------


def test_gate_token_exact_value():
    assert execution.EXECUTION_GATE_CONFIRMATION == "V8B_DATA_QUALITY_CALIBRATION_EXECUTION_GATE"


def test_wrong_gate_confirmation_blocks_before_touching_cache_root(monkeypatch):
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", Path("/definitely/does/not/exist/anywhere"))
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_production_v8b_data_quality_calibration(
            confirmation="NOT_THE_TOKEN",
            implementation_git_commit=VALID_COMMIT,
            calibration_attempt_id="attempt-1",
        )
    assert excinfo.value.reason == execution.EXECUTION_BLOCKER
    assert excinfo.value.detail == "EXECUTION_GATE_CONFIRMATION_REQUIRED"


def test_missing_gate_confirmation_blocks():
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_production_v8b_data_quality_calibration(
            confirmation="", implementation_git_commit=VALID_COMMIT, calibration_attempt_id="attempt-1"
        )
    assert excinfo.value.detail == "EXECUTION_GATE_CONFIRMATION_REQUIRED"


@pytest.mark.parametrize("bad_commit", ["", "not-hex", "A" * 40, "0" * 39, None, 12345])
def test_invalid_implementation_commit_format_blocks(bad_commit):
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=bad_commit)
    assert excinfo.value.detail == "IMPLEMENTATION_COMMIT_INVALID"


@pytest.mark.parametrize(
    "bad_attempt_id", ["", None, 12345, "x" * 129, "bad\x00id", "bad\x7fid", "bad\nid"]
)
def test_invalid_calibration_attempt_id_format_blocks(bad_attempt_id):
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=VALID_COMMIT, calibration_attempt_id=bad_attempt_id)
    assert excinfo.value.detail == "CALIBRATION_ATTEMPT_ID_INVALID"


def test_confirmation_checked_before_commit_and_attempt_id_format():
    # Wrong confirmation surfaces first, even though the commit and attempt
    # id are also malformed -- order is fixed and security-relevant.
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_production_v8b_data_quality_calibration(
            confirmation="WRONG", implementation_git_commit="not-hex", calibration_attempt_id=""
        )
    assert excinfo.value.detail == "EXECUTION_GATE_CONFIRMATION_REQUIRED"


# ---------------------------------------------------------------------------
# Section 3: implementation Git-HEAD binding, via the gated entry point.
# ---------------------------------------------------------------------------


def test_wrong_git_head_blocks_before_cache_access(tmp_path, monkeypatch):
    repo_root, actual_head = _build_synthetic_repo(tmp_path)
    monkeypatch.setattr(execution, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", tmp_path / "never_reached")
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=WRONG_COMMIT)
    assert excinfo.value.detail == "IMPLEMENTATION_COMMIT_HEAD_MISMATCH"
    assert excinfo.value.result is not None
    assert excinfo.value.result["checked_payload_count"] == 0


@pytest.mark.parametrize("relative_path", list(execution._RELEVANT_IMPLEMENTATION_RELATIVE_PATHS))
def test_dirty_relevant_file_blocks_before_cache_access(tmp_path, monkeypatch, relative_path):
    repo_root, actual_head = _build_synthetic_repo(tmp_path, mutate_relative_path=relative_path)
    monkeypatch.setattr(execution, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", tmp_path / "never_reached")
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=actual_head)
    assert excinfo.value.detail == "IMPLEMENTATION_FILE_DIRTY"


def test_git_head_unresolvable_blocks_before_cache_access(tmp_path, monkeypatch):
    not_a_repo = tmp_path / "not_a_git_repo"
    not_a_repo.mkdir()
    monkeypatch.setattr(execution, "_REPO_ROOT", not_a_repo)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", tmp_path / "never_reached")
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=VALID_COMMIT)
    assert excinfo.value.detail == "GIT_HEAD_UNRESOLVABLE"


def test_missing_committed_relevant_file_blocks(tmp_path, monkeypatch):
    omitted = execution._RELEVANT_IMPLEMENTATION_RELATIVE_PATHS[0]
    repo_root, actual_head = _build_synthetic_repo(tmp_path, omit_relative_path=omitted)
    monkeypatch.setattr(execution, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", tmp_path / "never_reached")
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=actual_head)
    assert excinfo.value.detail == "IMPLEMENTATION_FILE_UNVERIFIABLE"


def test_calibration_core_and_preflight_module_are_in_the_bound_path_set():
    assert "src/v8b_data_quality_calibration.py" in execution._RELEVANT_IMPLEMENTATION_RELATIVE_PATHS
    assert "src/v8b_v5b_calibration_input_preflight.py" in execution._RELEVANT_IMPLEMENTATION_RELATIVE_PATHS


def test_dirty_calibration_core_dependency_blocks_via_gated_entry_point(tmp_path, monkeypatch):
    repo_root, actual_head = _build_synthetic_repo(
        tmp_path, mutate_relative_path="src/v8b_data_quality_calibration.py"
    )
    monkeypatch.setattr(execution, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", tmp_path / "never_reached")
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=actual_head)
    assert excinfo.value.detail == "IMPLEMENTATION_FILE_DIRTY"


def test_git_routing_environment_cannot_redirect_head_or_commit_reads(tmp_path, monkeypatch):
    repo_root, actual_head = _build_synthetic_repo(tmp_path)
    external_parent = tmp_path / "external"
    external_parent.mkdir()
    external_root, _external_head = _build_synthetic_repo(external_parent)
    (external_root / "external-marker.txt").write_bytes(b"external repository")
    _run_git_or_fail(["add", "external-marker.txt"], external_root)
    _run_git_or_fail(["commit", "-q", "-m", "external repository commit"], external_root)
    external_head = subprocess.run(
        ["git", "-C", str(external_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    monkeypatch.setattr(execution, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", tmp_path / "never_reached")

    for variable, value in {
        "GIT_DIR": str(external_root / ".git"),
        "GIT_WORK_TREE": str(external_root),
        "GIT_INDEX_FILE": str(external_root / ".git" / "index"),
        "GIT_OBJECT_DIRECTORY": str(external_root / ".git" / "objects"),
        "GIT_COMMON_DIR": str(external_root / ".git"),
    }.items():
        monkeypatch.setenv(variable, value)
        assert preflight._resolve_actual_git_head(repo_root) == actual_head
        with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
            _pass_gate(implementation_git_commit=external_head)
        assert excinfo.value.detail == "IMPLEMENTATION_COMMIT_HEAD_MISMATCH"
        assert excinfo.value.result["checked_payload_count"] == 0
        monkeypatch.delenv(variable, raising=False)


# ---------------------------------------------------------------------------
# Section 4: real-cache read semantics -- missing root/manifest/payload,
# symlink/reparse/path escape, byte-count/SHA mismatch.
# ---------------------------------------------------------------------------


def test_cache_root_missing_blocks(tmp_path, monkeypatch, gated_head):
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", tmp_path / "does_not_exist")
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "CACHE_ROOT_INACCESSIBLE"


def test_cache_root_not_a_directory_blocks(tmp_path, monkeypatch, gated_head):
    not_a_dir = tmp_path / "just_a_file"
    not_a_dir.write_text("x")
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", not_a_dir)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "CACHE_ROOT_NOT_A_DIRECTORY"


def test_root_symlink_is_rejected_before_manifest_read(tmp_path, monkeypatch, gated_head):
    real_root, _manifest_bytes, _manifest = _write_synthetic_cache(tmp_path)
    alias = tmp_path / "cache-alias"
    try:
        alias.symlink_to(real_root, target_is_directory=True)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires unavailable privilege")
        raise
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", alias)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "CACHE_ROOT_REPARSE_POINT"


def test_missing_manifest_blocks(tmp_path, monkeypatch, gated_head):
    root = tmp_path / "empty_cache"
    root.mkdir()
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.reason == execution.EXECUTION_BLOCKER
    assert excinfo.value.detail == "MANIFEST_UNREADABLE"
    assert excinfo.value.result["status"] == "BLOCKED"
    assert _no_leakage(excinfo.value.result)


def test_manifest_symlink_is_rejected_before_manifest_read(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, _manifest = _write_synthetic_cache(tmp_path)
    manifest_path = root / "cache_manifest.json"
    outside = tmp_path / "manifest-outside.json"
    outside.write_bytes(manifest_bytes)
    manifest_path.unlink()
    try:
        manifest_path.symlink_to(outside)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink creation requires unavailable privilege")
        raise
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "MANIFEST_REPARSE_POINT"


def test_wrong_manifest_hash_blocks(tmp_path, monkeypatch, gated_head):
    # Deliberately do NOT patch the expected hash constants: the synthetic
    # manifest's real SHA-256 will never equal the frozen production pin.
    root, _manifest_bytes, _manifest = _write_synthetic_cache(tmp_path)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "MANIFEST_PROVENANCE_INVALID:MANIFEST_SHA256_MISMATCH"


def test_manifest_level_path_traversal_is_rejected_before_binding(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    payloads = list(manifest["payloads"])
    payloads[0] = dict(payloads[0], relative_path="raw/../escape.json")
    manifest = dict(manifest, payloads=payloads)
    manifest["payload_hash_list_sha256"] = calib._recompute_payload_hash_list_sha256(payloads)
    manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    (root / "cache_manifest.json").write_bytes(manifest_bytes)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "MANIFEST_PROVENANCE_INVALID:MANIFEST_RELATIVE_PATH_INVALID"


@pytest.mark.parametrize("payload_count", [299, 301])
def test_designated_payload_count_mismatch_blocks(tmp_path, monkeypatch, gated_head, payload_count):
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
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "MANIFEST_PROVENANCE_INVALID:MANIFEST_PAYLOAD_COUNT_MISMATCH"


def test_missing_payload_file_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, corrupt_index=7, corrupt_kind="missing")
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "PAYLOAD_BINDING_FAILED"
    result = excinfo.value.result
    assert result["missing_or_unreadable_count"] == 1
    assert result["checked_payload_count"] == 299
    assert _no_leakage(result)


def test_path_escape_via_symlink_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, corrupt_index=3, corrupt_kind="symlink_escape")
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    # The symlinked payload itself is caught by the earlier, more specific
    # reparse-point check before the separate path-escape/containment check
    # ever runs.
    assert excinfo.value.detail == "PAYLOAD_REPARSE_POINT"
    assert _no_leakage(excinfo.value.result)


def test_byte_count_mismatch_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(
        tmp_path, corrupt_index=11, corrupt_kind="byte_count_mismatch"
    )
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "PAYLOAD_BINDING_FAILED"
    result = excinfo.value.result
    assert result["byte_count_mismatch_count"] == 1
    assert result["checked_payload_count"] == 300


def test_payload_sha_mismatch_blocks(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, corrupt_index=42, corrupt_kind="sha_mismatch")
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    assert excinfo.value.detail == "PAYLOAD_BINDING_FAILED"
    result = excinfo.value.result
    assert result["sha256_mismatch_count"] == 1
    assert result["byte_count_mismatch_count"] == 0
    assert result["checked_payload_count"] == 300


# ---------------------------------------------------------------------------
# Section 5: the frozen calibration core is never invoked before the
# adapter's own manifest/payload byte-binding fully succeeds.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "corrupt_kind",
    ["missing", "byte_count_mismatch", "sha_mismatch"],
)
def test_calibration_core_never_invoked_on_binding_failure(tmp_path, monkeypatch, gated_head, corrupt_kind):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, corrupt_index=5, corrupt_kind=corrupt_kind)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)

    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        raise AssertionError("run_data_quality_calibration must not be called on a binding failure")

    monkeypatch.setattr(execution, "run_data_quality_calibration", spy)

    with pytest.raises(execution.V8BCalibrationExecutionBlocked):
        _pass_gate(implementation_git_commit=gated_head)
    assert calls == []


def test_calibration_core_never_invoked_on_wrong_manifest_hash(tmp_path, monkeypatch, gated_head):
    root, _manifest_bytes, _manifest = _write_synthetic_cache(tmp_path)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)

    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        raise AssertionError("must not be called")

    monkeypatch.setattr(execution, "run_data_quality_calibration", spy)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked):
        _pass_gate(implementation_git_commit=gated_head)
    assert calls == []


def test_calibration_core_never_invoked_before_git_verification_succeeds(tmp_path, monkeypatch):
    repo_root, actual_head = _build_synthetic_repo(tmp_path)
    monkeypatch.setattr(execution, "_REPO_ROOT", repo_root)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", tmp_path / "never_reached")

    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        raise AssertionError("must not be called")

    monkeypatch.setattr(execution, "run_data_quality_calibration", spy)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=WRONG_COMMIT)
    assert excinfo.value.detail == "IMPLEMENTATION_COMMIT_HEAD_MISMATCH"
    assert calls == []


# ---------------------------------------------------------------------------
# Section 6: genuine reach-and-succeed -- the frozen calibration core is
# actually invoked, exactly once, with the same in-memory bytes that were
# just verified, and its return value is passed through unmodified.
# ---------------------------------------------------------------------------


def test_calibration_core_invoked_exactly_once_with_bound_bytes_on_success(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)

    calls = []
    real = execution.run_data_quality_calibration

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(execution, "run_data_quality_calibration", spy)

    result = _pass_gate(implementation_git_commit=gated_head, calibration_attempt_id="reach-test")

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["manifest_bytes"] == manifest_bytes
    assert kwargs["repository_root"] == execution._REPO_ROOT
    assert kwargs["implementation_git_commit"] == gated_head
    assert kwargs["calibration_attempt_id"] == "reach-test"
    expected_tickers = {record["ticker"] for record in manifest["payloads"]}
    assert set(kwargs["ticker_payloads"]) == expected_tickers
    for record in manifest["payloads"]:
        supplied = kwargs["ticker_payloads"][record["ticker"]]
        assert supplied.relative_path == record["relative_path"]
        assert supplied.payload_bytes == (root / record["relative_path"]).read_bytes()

    # No dependency-file copies were supplied by `gated_head`, so the
    # frozen core's own verify_repository_contract() correctly blocks the
    # run -- proving the adapter genuinely reached and invoked it, and
    # passed its (unmodified) return value straight through, rather than
    # intercepting or reshaping it.
    assert result["schema_version"] == calib.RESULT_SCHEMA_VERSION
    assert result["schema_version"] != execution.EXECUTION_STATUS_SCHEMA_VERSION
    assert result["calibration_run_valid"] is False
    assert result["run_invalid_reason_or_null"] == "CALIBRATION_PLAN_BLOB_MISMATCH"


@pytest.mark.slow
def test_full_happy_path_produces_a_genuinely_valid_calibration_result(tmp_path, monkeypatch, dependency_bound_head):
    root, manifest_bytes, manifest = _write_synthetic_full_cache(tmp_path)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)

    result = _pass_gate(implementation_git_commit=dependency_bound_head, calibration_attempt_id="full-happy-path")

    assert result["schema_version"] == calib.RESULT_SCHEMA_VERSION
    assert result["calibration_run_valid"] is True
    assert result["run_invalid_reason_or_null"] is None
    assert result["candidate_selection_executed"] is True
    assert result["selected_policy"] == "F1_C1"
    assert result["synthetic_base_count"] == 20
    assert result["input_provenance_hashes"]["bound_payload_count"] == 300
    assert result["input_provenance_hashes"]["manifest_payload_count"] == 300
    assert calib.verify_artifact_self_hash(result) is True
    serialized = json.dumps(result, default=str)
    assert "TICK000" not in serialized
    assert "raw/" not in serialized


# ---------------------------------------------------------------------------
# Section 7: execution status semantics validator.
# ---------------------------------------------------------------------------


def _valid_payload_binding_failed_status() -> dict:
    return execution._canonical_execution_status(
        "PAYLOAD_BINDING_FAILED",
        implementation_git_commit="a" * 40,
        calibration_attempt_id="attempt-1",
        observed_manifest_sha256=calib.EXPECTED_V5B_MANIFEST_SHA256,
        observed_payload_hash_list_sha256=calib.EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256,
        checked_payload_count=299,
        missing_or_unreadable_count=1,
    )


def _rehash(result: dict) -> dict:
    mutated = dict(result)
    mutated.pop("artifact_self_hash")
    mutated["artifact_self_hash"] = execution.sha256_hex(execution.canonical_json_bytes(mutated))
    return mutated


def test_semantic_validator_requires_expected_commit_argument():
    result = _valid_payload_binding_failed_status()
    with pytest.raises(TypeError):
        execution.validate_execution_status_semantics(
            result, expected_calibration_attempt_id="attempt-1"
        )  # type: ignore[call-arg]


def test_semantic_validator_requires_expected_attempt_id_argument():
    result = _valid_payload_binding_failed_status()
    with pytest.raises(TypeError):
        execution.validate_execution_status_semantics(
            result, expected_implementation_git_commit="a" * 40
        )  # type: ignore[call-arg]


def test_semantic_validator_accepts_genuine_block_artifact():
    result = _valid_payload_binding_failed_status()
    execution.validate_execution_status_semantics(
        result, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
    )  # must not raise


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.update(checked_payload_count=300, missing_or_unreadable_count=0),
        lambda item: item.update(observed_manifest_sha256="0" * 64),
        lambda item: item.update(observed_payload_hash_list_sha256="0" * 64),
        lambda item: item.update(implementation_git_commit="b" * 40),
        lambda item: item.update(checked_payload_count=True),
        lambda item: item.update(missing_or_unreadable_count=1.0),
        lambda item: item.pop("detail_reason"),
        lambda item: item.update(extra_field="unexpected"),
        lambda item: item.update(run_completed_utc="2020-01-01T00:00:00Z"),
        lambda item: item.update(detail_reason="UNKNOWN_DETAIL"),
        lambda item: item.update(status="PASS"),
        lambda item: item.update(calibration_attempt_id="a" * 200),
        lambda item: item.update(calibration_attempt_id="attempt-EVIL"),
        # Forging the clean-bind counts alone does not relabel the detail:
        # this must still be rejected as an ordinary payload-stage failure
        # falsely claiming a clean bind, not accepted as CALIBRATION_CORE_
        # BLOCKED (which requires the detail itself to carry that prefix).
        lambda item: item.update(
            checked_payload_count=300,
            byte_count_mismatch_count=0,
            sha256_mismatch_count=0,
            missing_or_unreadable_count=0,
        ),
    ],
)
def test_rehashed_semantic_mutations_are_rejected(mutation):
    mutated = _valid_payload_binding_failed_status()
    mutation(mutated)
    mutated = _rehash(mutated)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked):
        execution.validate_execution_status_semantics(
            mutated, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
        )


def test_semantic_validator_rejects_rehashed_commit_substitution():
    genuine = _valid_payload_binding_failed_status()
    execution.validate_execution_status_semantics(
        genuine, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
    )

    mutated_fields = dict(genuine)
    mutated_fields["implementation_git_commit"] = "b" * 40
    del mutated_fields["artifact_self_hash"]
    mutated_fields["artifact_self_hash"] = execution.sha256_hex(execution.canonical_json_bytes(mutated_fields))
    execution._verify_execution_status_self_hash(mutated_fields)  # integrity alone would pass

    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.validate_execution_status_semantics(
            mutated_fields, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
        )
    assert excinfo.value.detail == "EXECUTION_STATUS_COMMIT_MISMATCH"


def test_semantic_validator_rejects_rehashed_attempt_id_substitution():
    genuine = _valid_payload_binding_failed_status()  # calibration_attempt_id == "attempt-1"
    execution.validate_execution_status_semantics(
        genuine, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
    )

    # A different, still well-formed attempt id, with the self-hash
    # recomputed so integrity alone would pass.
    mutated_fields = dict(genuine)
    mutated_fields["calibration_attempt_id"] = "attempt-EVIL"
    del mutated_fields["artifact_self_hash"]
    mutated_fields["artifact_self_hash"] = execution.sha256_hex(execution.canonical_json_bytes(mutated_fields))
    execution._verify_execution_status_self_hash(mutated_fields)  # integrity alone would pass

    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.validate_execution_status_semantics(
            mutated_fields, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
        )
    assert excinfo.value.detail == "EXECUTION_STATUS_ATTEMPT_ID_MISMATCH"


def test_semantic_validator_rejects_a_fully_clean_bind_state_outside_calibration_core_blocked():
    # A gate-level status may claim a fully clean 300/300 bind ONLY under
    # the CALIBRATION_CORE_BLOCKED:* detail category; for every other
    # detail (including PAYLOAD_BINDING_FAILED here) that state means
    # calibration WAS invoked and returned a canonical RESULT artifact (the
    # other schema), never this one.
    forged = execution._canonical_execution_status(
        "PAYLOAD_BINDING_FAILED",
        implementation_git_commit="a" * 40,
        calibration_attempt_id="attempt-1",
        observed_manifest_sha256=calib.EXPECTED_V5B_MANIFEST_SHA256,
        observed_payload_hash_list_sha256=calib.EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256,
        checked_payload_count=300,
        missing_or_unreadable_count=0,
        byte_count_mismatch_count=0,
        sha256_mismatch_count=0,
    )
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.validate_execution_status_semantics(
            forged, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
        )
    assert excinfo.value.detail == "EXECUTION_STATUS_STATE_INVALID"


def test_calibration_core_blocked_requires_a_fully_clean_bind():
    # The converse of the rule above: CALIBRATION_CORE_BLOCKED:* can only
    # legitimately be produced immediately after a fully clean 300/300
    # bind (the frozen core is invoked only then), so anything less is
    # internally inconsistent and must be rejected too.
    forged = execution._canonical_execution_status(
        "CALIBRATION_CORE_BLOCKED:SYNTHETIC_CLASSIFIER_MISMATCH",
        implementation_git_commit="a" * 40,
        calibration_attempt_id="attempt-1",
        observed_manifest_sha256=calib.EXPECTED_V5B_MANIFEST_SHA256,
        observed_payload_hash_list_sha256=calib.EXPECTED_V5B_PAYLOAD_HASH_LIST_SHA256,
        checked_payload_count=299,
        missing_or_unreadable_count=1,
    )
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.validate_execution_status_semantics(
            forged, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
        )
    assert excinfo.value.detail == "EXECUTION_STATUS_STATE_INVALID"


def test_rehashed_forgery_from_ordinary_gate_failure_to_calibration_core_blocked_is_rejected():
    # An ordinary pre-manifest gate failure (no bytes ever bound) rehashed
    # into claiming CALIBRATION_CORE_BLOCKED:* must still fail: that detail
    # category requires a fully clean bind, which this state never had.
    genuine = execution._canonical_execution_status(
        "MANIFEST_UNREADABLE", implementation_git_commit="a" * 40, calibration_attempt_id="attempt-1"
    )
    execution.validate_execution_status_semantics(
        genuine, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
    )

    mutated = dict(genuine)
    mutated["detail_reason"] = "CALIBRATION_CORE_BLOCKED:SYNTHETIC_CLASSIFIER_MISMATCH"
    del mutated["artifact_self_hash"]
    mutated["artifact_self_hash"] = execution.sha256_hex(execution.canonical_json_bytes(mutated))
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.validate_execution_status_semantics(
            mutated, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
        )
    assert excinfo.value.detail == "EXECUTION_STATUS_STATE_INVALID"


@pytest.mark.parametrize("bad_expected", ["", "not-hex", "A" * 40, "0" * 39, "0" * 41])
def test_malformed_expected_commit_argument_is_rejected(bad_expected):
    genuine = _valid_payload_binding_failed_status()
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.validate_execution_status_semantics(
            genuine, expected_implementation_git_commit=bad_expected, expected_calibration_attempt_id="attempt-1"
        )
    assert excinfo.value.detail == "EXECUTION_STATUS_EXPECTED_COMMIT_INVALID"


@pytest.mark.parametrize("bad_expected", ["", "x" * 129, "bad\x00id", "bad\x7fid", "bad\nid"])
def test_malformed_expected_attempt_id_argument_is_rejected(bad_expected):
    genuine = _valid_payload_binding_failed_status()
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.validate_execution_status_semantics(
            genuine, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id=bad_expected
        )
    assert excinfo.value.detail == "EXECUTION_STATUS_EXPECTED_ATTEMPT_ID_INVALID"


def test_early_gate_status_allows_none_commit_and_attempt_id_but_still_requires_both_trusted_arguments():
    early_block = execution._canonical_execution_status("EXECUTION_GATE_CONFIRMATION_REQUIRED")
    assert early_block["implementation_git_commit"] is None
    assert early_block["calibration_attempt_id"] is None
    with pytest.raises(TypeError):
        execution.validate_execution_status_semantics(
            early_block, expected_calibration_attempt_id="whatever-1"
        )  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        execution.validate_execution_status_semantics(
            early_block, expected_implementation_git_commit="d" * 40
        )  # type: ignore[call-arg]
    execution.validate_execution_status_semantics(
        early_block, expected_implementation_git_commit="d" * 40, expected_calibration_attempt_id="whatever-1"
    )


def test_manifest_provenance_invalid_detail_requires_exact_recognized_inner_reason():
    genuine = execution._canonical_execution_status(
        "MANIFEST_PROVENANCE_INVALID:MANIFEST_SHA256_MISMATCH",
        implementation_git_commit="a" * 40,
        calibration_attempt_id="attempt-1",
        observed_manifest_sha256="0" * 64,
    )
    execution.validate_execution_status_semantics(
        genuine, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
    )

    fabricated = execution._canonical_execution_status(
        "MANIFEST_PROVENANCE_INVALID:TOTALLY_MADE_UP_REASON",
        implementation_git_commit="a" * 40,
        calibration_attempt_id="attempt-1",
        observed_manifest_sha256="0" * 64,
    )
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.validate_execution_status_semantics(
            fabricated, expected_implementation_git_commit="a" * 40, expected_calibration_attempt_id="attempt-1"
        )
    assert excinfo.value.detail == "EXECUTION_STATUS_DETAIL_INVALID"


def test_validator_result_from_real_gate_failure_round_trips(tmp_path, monkeypatch, gated_head):
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path, corrupt_index=7, corrupt_kind="missing")
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head)
    execution.validate_execution_status_semantics(
        excinfo.value.result,
        expected_implementation_git_commit=gated_head,
        expected_calibration_attempt_id="test-attempt-0001",
    )  # must not raise -- the adapter's own real output satisfies its own validator


def test_calibration_core_blocked_round_trip_after_full_clean_bind(tmp_path, monkeypatch, gated_head):
    """Adversarial round-trip for Finding 1: (1) reach a full, clean 300/300
    byte bind using synthetic fixtures; (2) force the core call to raise a
    recognized V8BCalibrationBlocked; (3) capture the adapter-emitted
    execution status; (4) require validate_execution_status_semantics() to
    accept that genuine status -- construction and validation must agree."""

    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)

    def raising_core(**kwargs):
        raise calib.V8BCalibrationBlocked("SYNTHETIC_CLASSIFIER_MISMATCH")

    monkeypatch.setattr(execution, "run_data_quality_calibration", raising_core)

    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        _pass_gate(implementation_git_commit=gated_head, calibration_attempt_id="core-raise-test")

    assert excinfo.value.detail == "CALIBRATION_CORE_BLOCKED:SYNTHETIC_CLASSIFIER_MISMATCH"
    result = excinfo.value.result
    assert result["schema_version"] == execution.EXECUTION_STATUS_SCHEMA_VERSION
    assert result["observed_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert result["observed_payload_hash_list_sha256"] == manifest["payload_hash_list_sha256"]
    assert result["checked_payload_count"] == 300
    assert result["byte_count_mismatch_count"] == 0
    assert result["sha256_mismatch_count"] == 0
    assert result["missing_or_unreadable_count"] == 0
    assert _no_leakage(result)

    # This is Finding 1's core requirement: the adapter's own genuine
    # construction of this status must be accepted by its own validator.
    execution.validate_execution_status_semantics(
        result,
        expected_implementation_git_commit=gated_head,
        expected_calibration_attempt_id="core-raise-test",
    )


# ---------------------------------------------------------------------------
# Section 8: static-check mode (repository only).
# ---------------------------------------------------------------------------


def test_static_check_passes_cleanly_on_the_real_module():
    execution.run_static_check()  # must not raise


def test_static_check_detects_cache_root_drift(monkeypatch):
    monkeypatch.setattr(execution, "FIXED_V5B_CACHE_ROOT_WINDOWS_PATH", r"C:\somewhere\else")
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_CACHE_ROOT_DRIFT"


def test_static_check_detects_gate_token_drift(monkeypatch):
    monkeypatch.setattr(execution, "EXECUTION_GATE_CONFIRMATION", "SOMETHING_ELSE")
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_GATE_TOKEN_DRIFT"


def test_static_check_detects_production_api_surface_drift(monkeypatch):
    def fake_production(*, confirmation, implementation_git_commit, calibration_attempt_id, cache_root=None):
        raise AssertionError("never called")

    monkeypatch.setattr(execution, "run_production_v8b_data_quality_calibration", fake_production)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_PRODUCTION_API_SURFACE_DRIFT"


def test_static_check_detects_reintroduced_ungated_export(monkeypatch):
    monkeypatch.setattr(execution, "__all__", list(execution.__all__) + ["_walk_and_execute"])
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_UNGATED_FILESYSTEM_RUNNER_EXPORTED"


def test_static_check_detects_module_level_function_with_cache_root_param(monkeypatch):
    def fake_runner(cache_root):  # pragma: no cover - never actually called
        return cache_root

    fake_runner.__module__ = execution.__name__
    monkeypatch.setattr(execution, "fake_runner", fake_runner, raising=False)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_UNGATED_FILESYSTEM_RUNNER_EXPORTED"


def test_static_check_detects_payload_count_drift(monkeypatch):
    monkeypatch.setattr(execution, "EXPECTED_V5B_TICKER_COUNT", 299)
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_PAYLOAD_COUNT_DRIFT"


def test_static_check_detects_manifest_validator_drift(monkeypatch):
    monkeypatch.setattr(execution, "validate_v5b_manifest_provenance", lambda manifest_bytes: {})
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_MANIFEST_VALIDATOR_DRIFT"


def test_static_check_detects_calibration_entry_point_drift(monkeypatch):
    monkeypatch.setattr(execution, "run_data_quality_calibration", lambda **kwargs: {})
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_CALIBRATION_ENTRY_POINT_DRIFT"


def test_static_check_detects_git_verifier_drift(monkeypatch):
    class _FakePreflightModule:
        @staticmethod
        def _verify_implementation_matches_repository_head(*, repo_root, implementation_git_commit):
            raise AssertionError("never called")

    monkeypatch.setattr(execution, "_v8b_preflight_module", _FakePreflightModule())
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_GIT_VERIFIER_DRIFT"


def test_static_check_detects_forbidden_source_token(tmp_path, monkeypatch):
    real_source = (REPO_ROOT / "src" / "v8b_data_quality_calibration_execution.py").read_text(encoding="utf-8")
    marker = "\ndef run_static_check"
    index = real_source.index(marker)
    tampered_source = real_source[:index] + "\nimport requests\n" + real_source[index:]
    tampered_file = tmp_path / "tampered_execution_module.py"
    tampered_file.write_text(tampered_source, encoding="utf-8")
    monkeypatch.setattr(execution, "__file__", str(tampered_file))
    with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
        execution.run_static_check()
    assert excinfo.value.detail == "STATIC_CHECK_FORBIDDEN_SOURCE_TOKEN"


def test_module_source_has_no_forbidden_parsing_calls_or_network_strings():
    source = (REPO_ROOT / "src" / "v8b_data_quality_calibration_execution.py").read_text(encoding="utf-8")
    functional_source = source[: source.index("\ndef run_static_check")]
    forbidden = [
        "parse_ticker_observations(",
        "_row_invalid_reason(",
        "select_synthetic_bases(",
        "compute_global_envelope(",
        "select_policy(",
        "apply_corruption(",
        "urllib",
        "requests",
        "yfinance",
        "query1.finance.yahoo.com",
        "http://",
        "https://",
    ]
    for token in forbidden:
        assert token not in functional_source, f"forbidden token found: {token}"


# ---------------------------------------------------------------------------
# Section 9: CLI smoke tests (in-process, no subprocess/git overhead --
# the Git-HEAD/dirty-file machinery is already covered above and in the
# preflight CLI's own test suite).
# ---------------------------------------------------------------------------


def test_cli_static_check_success(monkeypatch, capsys):
    from scripts import run_v8b_data_quality_calibration as cli

    exit_code = cli.main(["--static-check"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == cli.STATIC_SUCCESS_MESSAGE


def test_cli_static_check_rejects_extra_arguments(capsys):
    from scripts import run_v8b_data_quality_calibration as cli

    exit_code = cli.main(["--static-check", "--implementation-git-commit", VALID_COMMIT])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "STATIC_CHECK_TAKES_NO_OTHER_ARGUMENTS" in captured.err


def test_cli_confirm_requires_implementation_commit(capsys):
    from scripts import run_v8b_data_quality_calibration as cli

    exit_code = cli.main(["--confirm", execution.EXECUTION_GATE_CONFIRMATION, "--calibration-attempt-id", "x"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "IMPLEMENTATION_COMMIT_REQUIRED" in captured.err


def test_cli_confirm_requires_calibration_attempt_id(capsys):
    from scripts import run_v8b_data_quality_calibration as cli

    exit_code = cli.main(
        ["--confirm", execution.EXECUTION_GATE_CONFIRMATION, "--implementation-git-commit", VALID_COMMIT]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "CALIBRATION_ATTEMPT_ID_REQUIRED" in captured.err


def test_cli_wrong_confirmation_emits_canonical_json_status(monkeypatch, capsys):
    from scripts import run_v8b_data_quality_calibration as cli

    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", Path("/definitely/does/not/exist"))
    exit_code = cli.main(
        [
            "--confirm",
            "WRONG",
            "--implementation-git-commit",
            VALID_COMMIT,
            "--calibration-attempt-id",
            "test1",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 2
    payload = json.loads(captured.out)
    assert payload["status"] == "BLOCKED"
    assert payload["detail_reason"] == "EXECUTION_GATE_CONFIRMATION_REQUIRED"
    assert payload["calibration_attempt_id"] is None


def test_cli_mutually_exclusive_static_check_and_confirm():
    from scripts import run_v8b_data_quality_calibration as cli

    with pytest.raises(SystemExit):
        cli.main(["--static-check", "--confirm", execution.EXECUTION_GATE_CONFIRMATION])


# ---------------------------------------------------------------------------
# Section 10: exhaustive-failure single blocker reason + no leakage.
# ---------------------------------------------------------------------------


def test_all_blocked_exceptions_carry_the_single_generic_reason(tmp_path, monkeypatch, gated_head):
    for cache_root in (tmp_path / "missing", tmp_path):
        monkeypatch.setattr(execution, "V5B_CACHE_ROOT", cache_root)
        with pytest.raises(execution.V8BCalibrationExecutionBlocked) as excinfo:
            _pass_gate(implementation_git_commit=gated_head)
        assert excinfo.value.reason == execution.EXECUTION_BLOCKER


def test_pass_through_result_never_parses_payload_body_as_json(tmp_path, monkeypatch, gated_head):
    # Payload content is deliberately NOT valid Yahoo-chart JSON. Since
    # verify_repository_contract() (the very first thing the frozen core
    # does) blocks first here, this proves the adapter's own binding step
    # never attempted to interpret payload content as anything other than
    # opaque bytes for hashing/length checks.
    root, manifest_bytes, manifest = _write_synthetic_cache(tmp_path)
    _patch_expected_hashes(monkeypatch, manifest_bytes, manifest)
    monkeypatch.setattr(execution, "V5B_CACHE_ROOT", root)
    result = _pass_gate(implementation_git_commit=gated_head)
    assert result["schema_version"] == calib.RESULT_SCHEMA_VERSION
