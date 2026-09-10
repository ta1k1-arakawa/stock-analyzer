from __future__ import annotations

import copy
import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.v10_environment_extension_contract import (
    FROZEN_V10_DESIGN_SHA,
    PREDECESSOR_LOCK_BLOB_SHA1,
    PREDECESSOR_LOCK_SHA256,
    PREDECESSOR_PACKAGE_SET,
    ContractValidationError,
    build_exact_delta_install_argv,
    derive_package_sets,
    inspect_wheel_file,
    validate_mutation_preflight_receipt,
    validate_resolution_evidence,
    validate_successor_lock_candidate,
    verify_reviewed_wheelhouse,
)


EXTENSION_SHA = "a" * 40
IMPLEMENTATION_SHA = "b" * 40
DIRECT_BLOB_SHA = "c" * 40
DIRECT_SHA256 = "d" * 64
CANDIDATE_SHA256 = "e" * 64
MIGRATION_BLOB_SHA = "f" * 40
GENERIC_LOCK_BLOB_SHA = "1" * 40


def _wheel_filename(name: str, version: str) -> str:
    return f"{name.replace('-', '_')}-{version}-py3-none-any.whl"


def _candidate() -> dict:
    packages = [{"name": name, "version": version} for name, version in PREDECESSOR_PACKAGE_SET]
    packages.extend(
        [
            {"name": "exchange-calendars", "version": "5.0.0"},
            {"name": "pandas-market-calendars", "version": "5.4.0"},
        ]
    )
    packages.sort(key=lambda item: item["name"])
    wheels = []
    for package in packages:
        filename = _wheel_filename(package["name"], package["version"])
        wheels.append(
            {
                "name": package["name"],
                "version": package["version"],
                "filename": filename,
                "sha256": hashlib.sha256(filename.encode()).hexdigest(),
            }
        )
    return {
        "schema_version": "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE_V2",
        "artifact_status": "WINDOWS_RESOLUTION_CANDIDATE_NOT_INSTALL_AUTHORITY",
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": EXTENSION_SHA,
        "reviewed_resolution_implementation_git_sha": IMPLEMENTATION_SHA,
        "direct_spec_git_blob_sha1": DIRECT_BLOB_SHA,
        "direct_spec_sha256": DIRECT_SHA256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "predecessor_package_count": 15,
        "python_version": "3.12.10",
        "platform_system": "Windows",
        "platform_machine": "AMD64",
        "sysconfig_platform": "win-amd64",
        "resolution_policy_id": "PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1",
        "resolved_packages": packages,
        "resolved_package_count": len(packages),
        "resolved_wheels": wheels,
        "predecessor_pin_drift_count": 0,
        "pandas_market_calendars_version": "5.4.0",
        "exchange_calendars_version": "5.0.0",
    }


def _validate_candidate(candidate: dict) -> dict:
    return validate_successor_lock_candidate(
        candidate,
        expected_extension_design_sha=EXTENSION_SHA,
        expected_reviewed_resolution_implementation_sha=IMPLEMENTATION_SHA,
    )


def test_valid_candidate_and_delta_derivation() -> None:
    candidate = _candidate()
    package_sets = _validate_candidate(candidate)
    assert package_sets["delta"] == (
        ("exchange-calendars", "5.0.0"),
        ("pandas-market-calendars", "5.4.0"),
    )
    assert derive_package_sets(candidate["resolved_packages"])["delta"] == package_sets["delta"]


@pytest.mark.parametrize("mutation", ["missing_key", "extra_key"])
def test_candidate_exact_top_level_keys(mutation: str) -> None:
    candidate = _candidate()
    if mutation == "missing_key":
        del candidate["resolved_wheels"]
    else:
        candidate["unexpected"] = True
    with pytest.raises(ContractValidationError):
        _validate_candidate(candidate)


@pytest.mark.parametrize("mutation", ["duplicate_name", "unsorted_packages", "unsorted_wheels"])
def test_candidate_duplicate_and_order_rejection(mutation: str) -> None:
    candidate = _candidate()
    if mutation == "duplicate_name":
        candidate["resolved_packages"].append(copy.deepcopy(candidate["resolved_packages"][0]))
        candidate["resolved_package_count"] += 1
    elif mutation == "unsorted_packages":
        candidate["resolved_packages"][0], candidate["resolved_packages"][1] = (
            candidate["resolved_packages"][1],
            candidate["resolved_packages"][0],
        )
    else:
        candidate["resolved_wheels"][0], candidate["resolved_wheels"][1] = (
            candidate["resolved_wheels"][1],
            candidate["resolved_wheels"][0],
        )
    with pytest.raises(ContractValidationError):
        _validate_candidate(candidate)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_filename", "path_separator", "bad_sha", "version_mismatch", "missing_wheel", "extra_wheel"],
)
def test_candidate_wheel_manifest_rejection(mutation: str) -> None:
    candidate = _candidate()
    if mutation == "duplicate_filename":
        candidate["resolved_wheels"][1]["filename"] = candidate["resolved_wheels"][0]["filename"]
    elif mutation == "path_separator":
        candidate["resolved_wheels"][0]["filename"] = "nested/" + candidate["resolved_wheels"][0]["filename"]
    elif mutation == "bad_sha":
        candidate["resolved_wheels"][0]["sha256"] = "A" * 64
    elif mutation == "version_mismatch":
        candidate["resolved_wheels"][0]["version"] = "9.9.9"
    elif mutation == "missing_wheel":
        candidate["resolved_wheels"].pop()
    else:
        candidate["resolved_wheels"].append(copy.deepcopy(candidate["resolved_wheels"][-1]))
        candidate["resolved_wheels"][-1]["name"] = "new-package"
        candidate["resolved_wheels"][-1]["version"] = "1.0"
        candidate["resolved_wheels"][-1]["filename"] = "new_package-1.0-py3-none-any.whl"
    with pytest.raises(ContractValidationError):
        _validate_candidate(candidate)


def test_predecessor_pin_drift_is_rejected() -> None:
    candidate = _candidate()
    candidate["resolved_packages"][0]["version"] = "999.0"
    with pytest.raises(ContractValidationError, match="PREDECESSOR_PIN_DRIFT"):
        _validate_candidate(candidate)


def _write_wheel(root: Path, name: str, version: str) -> Path:
    filename = _wheel_filename(name, version)
    path = root / filename
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n".encode()
    wheel = b"Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", wheel)
    return path


def test_synthetic_wheel_inspection_and_integrity(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, "demo-package", "1.0")
    inspected = inspect_wheel_file(wheel)
    result = verify_reviewed_wheelhouse(
        tmp_path, [inspected], [{"name": "demo-package", "version": "1.0"}]
    )
    assert result["ok"] is True
    assert result["wheelhouse_integrity_verified"] is True
    assert result["delta_wheel_count"] == 1
    assert result["delta_wheel_paths"] == (wheel,)

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("tamper.txt", "changed bytes")
    tampered = verify_reviewed_wheelhouse(
        tmp_path, [inspected], [{"name": "demo-package", "version": "1.0"}]
    )
    assert tampered["ok"] is False
    assert tampered["failure_code"] == "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE"
    assert tampered["wheelhouse_integrity_verified"] is False


def test_wheelhouse_missing_and_extra_files_fail(tmp_path: Path) -> None:
    wheel = _write_wheel(tmp_path, "demo-package", "1.0")
    manifest = [inspect_wheel_file(wheel)]
    wheel.unlink()
    missing = verify_reviewed_wheelhouse(
        tmp_path, manifest, [{"name": "demo-package", "version": "1.0"}]
    )
    assert missing["wheelhouse_integrity_verified"] is False
    _write_wheel(tmp_path, "demo-package", "1.0")
    _write_wheel(tmp_path, "extra-package", "1.0")
    extra = verify_reviewed_wheelhouse(
        tmp_path, manifest, [{"name": "demo-package", "version": "1.0"}]
    )
    assert extra["failure_code"] == "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE"


def _resolution_evidence() -> dict:
    return {
        "schema_version": "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE_V1",
        "artifact_status": "WINDOWS_RESOLUTION_EVIDENCE",
        "status": "PASS",
        "failure_code": "NONE",
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": EXTENSION_SHA,
        "reviewed_resolution_implementation_git_sha": IMPLEMENTATION_SHA,
        "direct_spec_git_blob_sha1": DIRECT_BLOB_SHA,
        "direct_spec_sha256": DIRECT_SHA256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "resolution_policy_id": "PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1",
        "process_started": True,
        "process_exit_code": 0,
        "resolution_completed": True,
        "candidate_artifact_created": True,
        "successor_lock_candidate_sha256": CANDIDATE_SHA256,
        "resolved_package_count": 17,
        "package_index_id": "PYPI_OFFICIAL_SIMPLE",
        "package_resolution_process_invocations": 1,
        "human_authority_consumed": True,
        "package_installations": 0,
        "alternate_venv_created": False,
        "calendar_imports": 0,
        "calendar_dates_inspected": 0,
    }


def _validate_resolution(evidence: dict) -> None:
    validate_resolution_evidence(
        evidence,
        expected_extension_design_sha=EXTENSION_SHA,
        expected_reviewed_resolution_implementation_sha=IMPLEMENTATION_SHA,
    )


def test_resolution_process_start_semantics() -> None:
    evidence = _resolution_evidence()
    _validate_resolution(evidence)

    launch_failure = copy.deepcopy(evidence)
    launch_failure.update(
        status="FAIL",
        failure_code="RESOLUTION_PROCESS_FAILURE",
        process_started=False,
        process_exit_code=None,
        resolution_completed=False,
        candidate_artifact_created=False,
        successor_lock_candidate_sha256=None,
        resolved_package_count=None,
        package_resolution_process_invocations=0,
    )
    _validate_resolution(launch_failure)

    sentinel = copy.deepcopy(launch_failure)
    sentinel["process_exit_code"] = -1
    with pytest.raises(ContractValidationError):
        _validate_resolution(sentinel)

    nonzero = copy.deepcopy(evidence)
    nonzero.update(
        status="FAIL",
        failure_code="RESOLUTION_PROCESS_FAILURE",
        process_exit_code=7,
        resolution_completed=False,
        candidate_artifact_created=False,
        successor_lock_candidate_sha256=None,
        resolved_package_count=None,
    )
    _validate_resolution(nonzero)


def _receipt(status: str, failure: str, integrity: bool | None, delta_count: int | None) -> dict:
    return {
        "schema_version": "V10_CANONICAL_ENVIRONMENT_MUTATION_PREFLIGHT_RECEIPT_V1",
        "artifact_status": "V10_CANONICAL_ENVIRONMENT_MUTATION_PREFLIGHT_RECEIPT",
        "status": status,
        "failure_code": failure,
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": EXTENSION_SHA,
        "reviewed_successor_lock_candidate_sha256": CANDIDATE_SHA256,
        "migration_authority_git_blob_sha1": MIGRATION_BLOB_SHA,
        "generic_lock_git_blob_sha1": GENERIC_LOCK_BLOB_SHA,
        "wheelhouse_integrity_verified": integrity,
        "delta_wheel_count": delta_count,
        "mutation_authority_consumed": False,
        "mutation_started": False,
    }


def _validate_receipt(receipt: dict) -> None:
    validate_mutation_preflight_receipt(
        receipt,
        expected_extension_design_sha=EXTENSION_SHA,
        expected_successor_lock_candidate_sha256=CANDIDATE_SHA256,
        expected_migration_authority_git_blob_sha1=MIGRATION_BLOB_SHA,
        expected_generic_lock_git_blob_sha1=GENERIC_LOCK_BLOB_SHA,
    )


def test_preflight_receipt_nullable_integrity_semantics() -> None:
    _validate_receipt(_receipt("FAIL", "PROVENANCE_BINDING_FAILURE", None, None))
    _validate_receipt(_receipt("FAIL", "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE", False, 2))
    _validate_receipt(_receipt("PASS", "NONE", True, 2))

    invalid_not_checked = _receipt("FAIL", "PROVENANCE_BINDING_FAILURE", False, None)
    with pytest.raises(ContractValidationError):
        _validate_receipt(invalid_not_checked)


def test_exact_offline_install_argv() -> None:
    argv = build_exact_delta_install_argv(
        r"C:\venv\Scripts\python.exe",
        [r"C:\wheelhouse\exchange_calendars-5.0.0-py3-none-any.whl"],
    )
    assert argv == [
        r"C:\venv\Scripts\python.exe",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-index",
        r"C:\wheelhouse\exchange_calendars-5.0.0-py3-none-any.whl",
    ]
    assert "--find-links" not in argv
    assert all("==" not in item for item in argv)
    assert all("index-url" not in item for item in argv)
