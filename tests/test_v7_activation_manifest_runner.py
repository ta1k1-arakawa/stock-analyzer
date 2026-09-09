from __future__ import annotations

import ast
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from scripts import check_v7_activation_manifest as cli
from src import v7_activation_manifest as manifest_module
from src.v7_activation_manifest import (
    HUMAN_ACTIVATION_CONFIRMATION,
    V7ActivationManifestBlocked,
    canonical_json_bytes,
    compute_manifest_sha256,
    read_activation_manifest,
    write_activation_manifest_once,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_v7_activation_manifest.py"
PYTHON = sys.executable


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("v7-activation-runner")
    return cli.build_synthetic_fixture(workspace)


@pytest.fixture(scope="module")
def candidate(fixture):
    return cli.synthetic_candidate(fixture)


def write(fixture, manifest, path: Path, **overrides) -> dict[str, Any]:
    kwargs = dict(
        output_path=path,
        manifest=manifest,
        repository_root=fixture["repository_root"],
        confirmation=HUMAN_ACTIVATION_CONFIRMATION,
        calendar_path=cli.CALENDAR_PATH,
        universe_csv=cli.UNIVERSE_CSV,
        seed_csv=fixture["seed_csv"],
        seed_acquisition_manifest=fixture["seed_acquisition_manifest"],
        expected_seed_provenance=fixture["seed_expectation"],
    )
    kwargs.update(overrides)
    return write_activation_manifest_once(**kwargs)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_has_exactly_one_authorized_option():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    options = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
        and node.args[0].value.startswith("--")
    }
    assert options == {"--synthetic-activation-contract-test"}


def test_cli_has_no_real_path_network_or_activation_option():
    text = SCRIPT.read_text(encoding="utf-8")
    for flag in ("--output-path", "--output-root", "--activate", "--network", "--real", "--study-root", "--seed-csv"):
        assert flag not in text


def test_cli_performs_no_urlopen():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "urlopen"
    ]
    assert calls == []


def test_cli_subprocess_requires_the_flag():
    result = subprocess.run([PYTHON, str(SCRIPT)], cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    assert result.returncode != 0
    assert result.stdout == ""


def test_cli_subprocess_rejects_unknown_option():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--activate"], cwd=str(ROOT), capture_output=True, text=True, timeout=300
    )
    assert result.returncode != 0


def test_cli_synthetic_contract_test_passes():
    result = cli.run_synthetic_activation_contract_test()
    assert result == {
        "status": "PASS",
        "mode": "STATIC_SYNTHETIC_ONLY",
        "candidate_validation": "PASS",
        "manifest_hash_pass": True,
        "write_once_pass": True,
        "duplicate_write_blocked": True,
        "tamper_detection_pass": True,
        "network_requests": 0,
        "collector_enabled": False,
        "forward_processing": 0,
        "actual_activation_created": False,
    }


def test_cli_subprocess_exit_zero_and_reports_no_activation():
    result = subprocess.run(
        [PYTHON, str(SCRIPT), "--synthetic-activation-contract-test"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PASS"
    assert payload["actual_activation_created"] is False
    assert payload["network_requests"] == 0


def test_cli_leaves_no_manifest_in_repository():
    cli.run_synthetic_activation_contract_test()
    assert not (ROOT / "activation_manifest.json").exists()
    assert not list((ROOT / "data").glob("activation_manifest*.json"))


# ---------------------------------------------------------------------------
# Write-once
# ---------------------------------------------------------------------------


def test_write_once_creates_manifest(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    result = write(fixture, candidate, path)
    assert result["status"] == "WRITTEN"
    assert path.exists()
    assert result["manifest_sha256"] == candidate["manifest_sha256"]


def test_write_once_bytes_are_canonical(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    write(fixture, candidate, path)
    assert path.read_bytes() == canonical_json_bytes(dict(candidate))


def test_read_back_is_byte_deterministic(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    write(fixture, candidate, path)
    read_back = read_activation_manifest(path)
    assert canonical_json_bytes(read_back) == canonical_json_bytes(dict(candidate))
    assert compute_manifest_sha256(read_back) == candidate["manifest_sha256"]


def test_second_write_blocked(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    write(fixture, candidate, path)
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        write(fixture, candidate, path)
    assert excinfo.value.reason == "ACTIVATION_MANIFEST_ALREADY_EXISTS"


def test_second_write_does_not_change_bytes(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    write(fixture, candidate, path)
    before = path.read_bytes()
    with pytest.raises(V7ActivationManifestBlocked):
        write(fixture, candidate, path)
    assert path.read_bytes() == before


def test_preexisting_file_blocked(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        write(fixture, candidate, path)
    assert excinfo.value.reason == "ACTIVATION_MANIFEST_ALREADY_EXISTS"
    assert path.read_text(encoding="utf-8") == "{}"


def test_missing_parent_directory_blocked(tmp_path, fixture, candidate):
    path = tmp_path / "absent" / "activation_manifest.json"
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        write(fixture, candidate, path)
    assert excinfo.value.reason == "ACTIVATION_MANIFEST_PARENT_MISSING"
    assert not path.exists()


def test_failed_validation_writes_no_file(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    tampered = {**candidate, "manifest_sha256": "5" * 64}
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        write(fixture, tampered, path)
    assert excinfo.value.reason == "MANIFEST_SHA_MISMATCH"
    assert not path.exists()


def test_failed_validation_leaves_no_staging_remnant(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    tampered = {**candidate, "ticker_count": 299}
    with pytest.raises(V7ActivationManifestBlocked):
        write(fixture, tampered, path)
    assert [entry.name for entry in tmp_path.iterdir() if ".staging-" in entry.name] == []


def test_successful_write_leaves_no_staging_remnant(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    write(fixture, candidate, path)
    assert [entry.name for entry in tmp_path.iterdir() if ".staging-" in entry.name] == []
    assert sorted(entry.name for entry in tmp_path.iterdir()) == ["activation_manifest.json"]


def test_write_into_repository_root_blocked(tmp_path, fixture, candidate):
    path = tmp_path / "activation_manifest.json"
    tampered = {**candidate, "output_root": str(ROOT.resolve())}
    tampered["manifest_sha256"] = compute_manifest_sha256(tampered)
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        write(fixture, tampered, path)
    assert excinfo.value.reason == "OUTPUT_ROOT_INSIDE_SOURCE_REPOSITORY"
    assert not path.exists()


# ---------------------------------------------------------------------------
# Human activation confirmation
# ---------------------------------------------------------------------------


def test_confirmation_literal_is_exact():
    assert HUMAN_ACTIVATION_CONFIRMATION == "V7_GATE4_HUMAN_ACTIVATION_APPROVED"


@pytest.mark.parametrize("confirmation", [
    "", "APPROVED", "v7_gate4_human_activation_approved",
    "V7_GATE4_HUMAN_ACTIVATION_APPROVED ", "YES", None,
])
def test_wrong_confirmation_blocked(tmp_path, fixture, candidate, confirmation):
    path = tmp_path / "activation_manifest.json"
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        write(fixture, candidate, path, confirmation=confirmation)
    assert excinfo.value.reason == "HUMAN_ACTIVATION_CONFIRMATION_REQUIRED"
    assert not path.exists()


def test_confirmation_is_a_required_keyword_argument():
    import inspect

    parameter = inspect.signature(write_activation_manifest_once).parameters["confirmation"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_wrong_confirmation_checked_before_any_validation(tmp_path, fixture):
    path = tmp_path / "activation_manifest.json"
    with pytest.raises(V7ActivationManifestBlocked) as excinfo:
        write(fixture, {"clearly": "invalid"}, path, confirmation="WRONG")
    assert excinfo.value.reason == "HUMAN_ACTIVATION_CONFIRMATION_REQUIRED"


# ---------------------------------------------------------------------------
# No overwrite / correction API exists
# ---------------------------------------------------------------------------


def test_module_exposes_no_overwrite_or_update_api():
    exported = set(manifest_module.__all__)
    for forbidden in ("overwrite", "update", "amend", "append", "patch", "delete", "replace_manifest"):
        assert not any(forbidden in name.lower() for name in exported), forbidden


def test_write_api_has_no_overwrite_or_force_parameter():
    import inspect

    parameters = set(inspect.signature(write_activation_manifest_once).parameters)
    for forbidden in ("overwrite", "force", "replace", "allow_existing"):
        assert forbidden not in parameters


def _filesystem_writer_functions() -> set[str]:
    """Functions that can create or modify a file, by AST inspection."""
    tree = ast.parse(Path(manifest_module.__file__).read_text(encoding="utf-8"))
    writers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            if isinstance(func, ast.Attribute):
                module = func.value.id if isinstance(func.value, ast.Name) else None
                if func.attr in {"write_bytes", "write_text", "mkdir", "unlink", "fsync"}:
                    writers.add(node.name)
                elif module == "os" and func.attr == "replace":
                    writers.add(node.name)
                elif module == "tempfile" and func.attr in {"mkstemp", "mkdtemp"}:
                    writers.add(node.name)
                elif module == "os" and func.attr == "fdopen":
                    writers.add(node.name)
            elif isinstance(func, ast.Name) and func.id == "open":
                writers.add(node.name)
    return writers


def test_only_write_function_creates_files():
    assert _filesystem_writer_functions() <= {"write_activation_manifest_once"}


# ---------------------------------------------------------------------------
# No side effects from build/validate
# ---------------------------------------------------------------------------


def test_module_source_has_no_network_imports():
    text = Path(manifest_module.__file__).read_text(encoding="utf-8")
    import_lines = [line.strip().lower() for line in text.splitlines() if line.strip().startswith(("import ", "from "))]
    for token in ("re" + "quests", "url" + "lib", "http" + "x", "aio" + "http", "so" + "cket", "y" + "finance"):
        assert not any(token in line for line in import_lines), token


def test_module_source_has_no_collector_or_processing_calls():
    text = Path(manifest_module.__file__).read_text(encoding="utf-8")
    for token in (
        "fetch_chart_once", "acquire_daily_bundle", "acquire_seed_bundle",
        "generate_forward_candidates_for_day", "process_forward_day",
        "ForwardStudyStore", "CausalEventEngine(", "place_order", "submit_order",
    ):
        assert token not in text, token


def test_real_orders_remain_prohibited():
    assert manifest_module.PROHIBITION_FIELDS["real_orders_allowed"] is False
    assert manifest_module.PROHIBITION_FIELDS["deployment_allowed"] is False


def test_module_source_has_no_profit_or_evaluation_tokens():
    text = Path(manifest_module.__file__).read_text(encoding="utf-8").lower()
    for token in ("realized_net_profit", "profit_factor", "win_rate", "drawdown", "formal_evaluation"):
        assert token not in text, token


def test_validation_creates_no_forward_store(tmp_path, fixture, candidate):
    from scripts.check_v7_activation_manifest import CALENDAR_PATH, UNIVERSE_CSV
    from src.v7_activation_manifest import validate_activation_manifest_candidate

    validate_activation_manifest_candidate(
        candidate,
        repository_root=fixture["repository_root"],
        calendar_path=CALENDAR_PATH,
        universe_csv=UNIVERSE_CSV,
        seed_csv=fixture["seed_csv"],
        seed_acquisition_manifest=fixture["seed_acquisition_manifest"],
        expected_seed_provenance=fixture["seed_expectation"],
    )
    assert not (Path(fixture["output_root"]) / "days").exists()
    assert not (Path(fixture["output_root"]) / "acquisitions").exists()
    assert list(tmp_path.iterdir()) == []


def test_candidate_build_touches_no_filesystem(tmp_path, fixture):
    cli.synthetic_candidate(fixture)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The accepted DRY_RUN_ONLY validator must remain untouched
# ---------------------------------------------------------------------------


def test_dry_run_validator_still_requires_dry_run_only_mode():
    from src.v7_forward_protocol import ProtocolBlocked, validate_activation_manifest

    with pytest.raises(ProtocolBlocked):
        validate_activation_manifest({"mode": manifest_module.MODE})


def test_dry_run_validator_schema_is_unchanged():
    from src.v7_forward_protocol import ACTIVATION_MANIFEST_FIELDS

    assert "seed_payload_manifest_sha256" in ACTIVATION_MANIFEST_FIELDS
    assert "seed_source_payload_manifest_sha256" not in ACTIVATION_MANIFEST_FIELDS
    assert "manifest_sha256" not in ACTIVATION_MANIFEST_FIELDS


def test_production_schema_is_independent_of_dry_run_schema():
    from src.v7_forward_protocol import ACTIVATION_MANIFEST_FIELDS

    assert set(manifest_module.MANIFEST_FIELDS) != set(ACTIVATION_MANIFEST_FIELDS)
    assert "manifest_sha256" in manifest_module.MANIFEST_FIELDS


def test_production_manifest_is_rejected_by_dry_run_validator(candidate):
    from src.v7_forward_protocol import ProtocolBlocked, validate_activation_manifest

    with pytest.raises(ProtocolBlocked):
        validate_activation_manifest(candidate)


def test_new_module_does_not_import_dry_run_validator():
    tree = ast.parse(Path(manifest_module.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert "validate_activation_manifest" not in imported


def test_new_module_never_calls_dry_run_validator():
    tree = ast.parse(Path(manifest_module.__file__).read_text(encoding="utf-8"))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "validate_activation_manifest" not in called


def test_seed_hash_field_names_are_unambiguous():
    assert "seed_source_payload_manifest_sha256" in manifest_module.MANIFEST_FIELDS
    assert "seed_ticker_manifest_sha256" in manifest_module.MANIFEST_FIELDS
    assert "seed_payload_manifest_sha256" not in manifest_module.MANIFEST_FIELDS
