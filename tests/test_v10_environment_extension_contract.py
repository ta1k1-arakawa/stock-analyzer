from __future__ import annotations

import copy
import hashlib
import zipfile
from pathlib import Path

import pytest

from scripts.v10_environment_extension_contract import (
    FROZEN_V10_DESIGN_SHA,
    PREDECESSOR_LOCK_BLOB_SHA1,
    PREDECESSOR_FREEZE_RECORD_GIT_BLOB_SHA1,
    PREDECESSOR_LOCK_CANDIDATE_GIT_BLOB_SHA1,
    PREDECESSOR_LOCK_SHA256,
    PREDECESSOR_PACKAGE_SET,
    GENERIC_SUCCESSOR_LOCK_GIT_BLOB_SHA1,
    GENERIC_SUCCESSOR_LOCK_PACKAGE_COUNT,
    GENERIC_SUCCESSOR_LOCK_SHA256,
    REVIEWED_RESOLUTION_EVIDENCE_GIT_BLOB_SHA1,
    REVIEWED_RESOLUTION_EVIDENCE_GIT_SHA,
    REVIEWED_SUCCESSOR_CANDIDATE_GIT_BLOB_SHA1,
    REVIEWED_SUCCESSOR_CANDIDATE_GIT_SHA,
    ContractValidationError,
    build_exact_delta_install_argv,
    derive_package_sets,
    inspect_wheel_file,
    validate_mutation_preflight_receipt,
    validate_generic_migration_authority,
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


def _migration_authority() -> dict:
    return {
        "schema_version": "V10_CANONICAL_ENVIRONMENT_GENERIC_MIGRATION_AUTHORITY_V1",
        "artifact_status": "REVIEWED_INSTALL_AUTHORITY_NOT_LIVE_FROZEN",
        "canonical_environment_state": "V10_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED",
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "predecessor_generic_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_generic_lock_sha256": PREDECESSOR_LOCK_SHA256,
        "predecessor_generic_lock_package_count": 15,
        "predecessor_generic_lock_candidate_git_blob_sha1": PREDECESSOR_LOCK_CANDIDATE_GIT_BLOB_SHA1,
        "predecessor_generic_freeze_record_git_blob_sha1": PREDECESSOR_FREEZE_RECORD_GIT_BLOB_SHA1,
        "reviewed_v10_successor_lock_candidate_git_sha": REVIEWED_SUCCESSOR_CANDIDATE_GIT_SHA,
        "reviewed_v10_successor_lock_candidate_git_blob_sha1": REVIEWED_SUCCESSOR_CANDIDATE_GIT_BLOB_SHA1,
        "reviewed_v10_resolution_evidence_git_sha": REVIEWED_RESOLUTION_EVIDENCE_GIT_SHA,
        "reviewed_v10_resolution_evidence_git_blob_sha1": REVIEWED_RESOLUTION_EVIDENCE_GIT_BLOB_SHA1,
        "new_generic_lock_git_blob_sha1": GENERIC_SUCCESSOR_LOCK_GIT_BLOB_SHA1,
        "new_generic_lock_sha256": GENERIC_SUCCESSOR_LOCK_SHA256,
        "new_generic_lock_package_count": GENERIC_SUCCESSOR_LOCK_PACKAGE_COUNT,
        "live_environment_successor_match": False,
        "future_protected_execution_authorized": False,
    }


def _validate_migration_authority(authority: dict) -> None:
    validate_generic_migration_authority(
        authority,
        expected_reviewed_successor_lock_candidate_git_sha=REVIEWED_SUCCESSOR_CANDIDATE_GIT_SHA,
        expected_reviewed_successor_lock_candidate_git_blob_sha1=REVIEWED_SUCCESSOR_CANDIDATE_GIT_BLOB_SHA1,
        expected_reviewed_resolution_evidence_git_sha=REVIEWED_RESOLUTION_EVIDENCE_GIT_SHA,
        expected_reviewed_resolution_evidence_git_blob_sha1=REVIEWED_RESOLUTION_EVIDENCE_GIT_BLOB_SHA1,
        expected_new_generic_lock_git_blob_sha1=GENERIC_SUCCESSOR_LOCK_GIT_BLOB_SHA1,
        expected_new_generic_lock_sha256=GENERIC_SUCCESSOR_LOCK_SHA256,
        expected_new_generic_lock_package_count=GENERIC_SUCCESSOR_LOCK_PACKAGE_COUNT,
    )


def test_generic_migration_authority_exact_18_key_schema() -> None:
    _validate_migration_authority(_migration_authority())


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_commit", "wrong_blob", "truthy_bool", "bool_count"])
def test_generic_migration_authority_rejects_schema_or_binding_drift(mutation: str) -> None:
    authority = _migration_authority()
    if mutation == "missing":
        del authority["new_generic_lock_sha256"]
    elif mutation == "extra":
        authority["unexpected"] = True
    elif mutation == "wrong_commit":
        authority["reviewed_v10_successor_lock_candidate_git_sha"] = "0" * 40
    elif mutation == "wrong_blob":
        authority["new_generic_lock_git_blob_sha1"] = "0" * 40
    elif mutation == "truthy_bool":
        authority["live_environment_successor_match"] = 0
    else:
        authority["new_generic_lock_package_count"] = True
    with pytest.raises(ContractValidationError):
        _validate_migration_authority(authority)


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


def _successor_packages() -> list[dict[str, str]]:
    packages = [{"name": name, "version": version} for name, version in PREDECESSOR_PACKAGE_SET]
    packages.extend(
        [
            {"name": "exchange-calendars", "version": "5.0.0"},
            {"name": "pandas-market-calendars", "version": "5.4.0"},
        ]
    )
    packages.sort(key=lambda item: item["name"])
    return packages


def _write_successor_wheelhouse(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    successor = _successor_packages()
    manifest = []
    for package in successor:
        manifest.append(inspect_wheel_file(_write_wheel(root, package["name"], package["version"])))
    return successor, manifest


def test_synthetic_wheel_inspection_and_integrity(tmp_path: Path) -> None:
    successor, manifest = _write_successor_wheelhouse(tmp_path)
    result = verify_reviewed_wheelhouse(tmp_path, manifest, successor)
    assert result["ok"] is True
    assert result["wheelhouse_integrity_verified"] is True
    assert result["delta_packages"] == (
        ("exchange-calendars", "5.0.0"),
        ("pandas-market-calendars", "5.4.0"),
    )
    assert result["delta_wheel_count"] == len(result["delta_packages"])
    assert [path.name for path in result["delta_wheel_paths"]] == [
        "exchange_calendars-5.0.0-py3-none-any.whl",
        "pandas_market_calendars-5.4.0-py3-none-any.whl",
    ]
    argv = build_exact_delta_install_argv(
        r"C:\venv\Scripts\python.exe", result["delta_wheel_paths"]
    )
    assert argv[-2:] == [str(path) for path in result["delta_wheel_paths"]]

    wheel = tmp_path / "exchange_calendars-5.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("tamper.txt", "changed bytes")
    tampered = verify_reviewed_wheelhouse(tmp_path, manifest, successor)
    assert tampered["ok"] is False
    assert tampered["failure_code"] == "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE"
    assert tampered["wheelhouse_integrity_verified"] is False
    assert tampered["delta_wheel_count"] == 2
    assert tampered["delta_packages"] == result["delta_packages"]
    assert tampered["delta_wheel_paths"] == ()
    _validate_receipt(
        _receipt(
            "FAIL",
            "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE",
            tampered["wheelhouse_integrity_verified"],
            tampered["delta_wheel_count"],
        )
    )


def test_wheelhouse_missing_and_extra_files_fail(tmp_path: Path) -> None:
    successor, manifest = _write_successor_wheelhouse(tmp_path)
    wheel = tmp_path / "exchange_calendars-5.0.0-py3-none-any.whl"
    wheel.unlink()
    missing = verify_reviewed_wheelhouse(tmp_path, manifest, successor)
    assert missing["wheelhouse_integrity_verified"] is False
    assert missing["delta_wheel_count"] == 2
    assert missing["delta_packages"] == (
        ("exchange-calendars", "5.0.0"),
        ("pandas-market-calendars", "5.4.0"),
    )
    _write_wheel(tmp_path, "exchange-calendars", "5.0.0")
    _write_wheel(tmp_path, "extra-package", "1.0")
    extra = verify_reviewed_wheelhouse(tmp_path, manifest, successor)
    assert extra["failure_code"] == "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE"
    assert extra["delta_wheel_count"] == 2
    assert extra["delta_packages"] == missing["delta_packages"]


def test_malformed_wheel_after_delta_derivation_preserves_observation(tmp_path: Path) -> None:
    successor, manifest = _write_successor_wheelhouse(tmp_path)
    wheel = tmp_path / "exchange_calendars-5.0.0-py3-none-any.whl"
    wheel.write_bytes(b"not a wheel")
    result = verify_reviewed_wheelhouse(tmp_path, manifest, successor)
    assert result["ok"] is False
    assert result["failure_code"] == "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE"
    assert result["wheelhouse_integrity_verified"] is False
    assert result["delta_wheel_count"] == 2
    assert result["delta_packages"] == (
        ("exchange-calendars", "5.0.0"),
        ("pandas-market-calendars", "5.4.0"),
    )
    assert result["delta_wheel_paths"] == ()


def test_invalid_successor_fails_before_wheelhouse_observation(tmp_path: Path) -> None:
    successor, manifest = _write_successor_wheelhouse(tmp_path)
    invalid_successor = copy.deepcopy(successor)
    invalid_successor[0]["version"] = "999.0.0"
    with pytest.raises(ContractValidationError, match="PREDECESSOR_PIN_DRIFT"):
        verify_reviewed_wheelhouse(tmp_path, manifest, invalid_successor)


def test_wheelhouse_delta_is_derived_and_legacy_subset_is_not_accepted(tmp_path: Path) -> None:
    successor, manifest = _write_successor_wheelhouse(tmp_path)
    with pytest.raises(TypeError):
        verify_reviewed_wheelhouse(tmp_path, manifest, successor, successor[:1])
    result = verify_reviewed_wheelhouse(tmp_path, manifest, successor)
    assert result["delta_packages"] == (
        ("exchange-calendars", "5.0.0"),
        ("pandas-market-calendars", "5.4.0"),
    )
    assert all(name not in {item[0] for item in PREDECESSOR_PACKAGE_SET} for name, _ in result["delta_packages"])


@pytest.mark.parametrize("mutation", [
    "missing_delta_wheel",
    "wheel_not_in_successor",
    "successor_version_mismatch",
])
def test_wheelhouse_exact_successor_delta_binding_rejection(tmp_path: Path, mutation: str) -> None:
    successor, manifest = _write_successor_wheelhouse(tmp_path)
    if mutation == "missing_delta_wheel":
        manifest.pop()
    elif mutation == "wheel_not_in_successor":
        manifest[-1] = {
            **manifest[-1],
            "name": "unreviewed-package",
            "version": "1.0.0",
            "filename": "unreviewed_package-1.0.0-py3-none-any.whl",
        }
    elif mutation == "successor_version_mismatch":
        manifest[-1] = {**manifest[-1], "version": "5.4.1"}
    result = verify_reviewed_wheelhouse(tmp_path, manifest, successor)
    assert result["ok"] is False
    assert result["failure_code"] == "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE"
    assert result["delta_wheel_paths"] == ()
    assert result["delta_packages"] == (
        ("exchange-calendars", "5.0.0"),
        ("pandas-market-calendars", "5.4.0"),
    )
    assert result["delta_wheel_count"] == 2


def test_wrong_delta_version_is_rejected_after_delta_derivation(tmp_path: Path) -> None:
    successor, manifest = _write_successor_wheelhouse(tmp_path)
    index = next(i for i, package in enumerate(successor) if package["name"] == "pandas-market-calendars")
    successor[index] = {**successor[index], "version": "99.0.0"}
    result = verify_reviewed_wheelhouse(tmp_path, manifest, successor)
    assert result["ok"] is False
    assert result["failure_code"] == "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE"
    assert result["wheelhouse_integrity_verified"] is False
    assert result["delta_wheel_count"] == 2
    assert result["delta_packages"] == (
        ("exchange-calendars", "5.0.0"),
        ("pandas-market-calendars", "99.0.0"),
    )
    assert result["delta_wheel_paths"] == ()


def test_predecessor_cannot_be_added_as_a_delta_package(tmp_path: Path) -> None:
    successor, manifest = _write_successor_wheelhouse(tmp_path)
    successor.append({"name": "cffi", "version": "2.1.1"})
    with pytest.raises(ContractValidationError):
        verify_reviewed_wheelhouse(tmp_path, manifest, successor)


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
        expected_direct_spec_git_blob_sha1=DIRECT_BLOB_SHA,
        expected_direct_spec_sha256=DIRECT_SHA256,
        expected_successor_lock_candidate_sha256=CANDIDATE_SHA256,
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


@pytest.mark.parametrize("field,value", [
    ("package_installations", 1),
    ("alternate_venv_created", True),
    ("calendar_imports", 1),
    ("calendar_dates_inspected", 1),
])
def test_resolution_pass_rejects_forbidden_operations(field: str, value: object) -> None:
    evidence = _resolution_evidence()
    evidence[field] = value
    with pytest.raises(ContractValidationError):
        _validate_resolution(evidence)


def test_resolution_unauthorized_installation_precedence() -> None:
    evidence = _resolution_evidence()
    evidence.update(
        status="FAIL",
        failure_code="UNAUTHORIZED_INSTALLATION",
        package_installations=1,
        resolution_completed=False,
        candidate_artifact_created=False,
        successor_lock_candidate_sha256=None,
        resolved_package_count=None,
    )
    _validate_resolution(evidence)

    invalid = copy.deepcopy(evidence)
    invalid["package_installations"] = 0
    with pytest.raises(ContractValidationError):
        _validate_resolution(invalid)

    both = copy.deepcopy(evidence)
    both["alternate_venv_created"] = True
    _validate_resolution(both)


def test_resolution_unauthorized_alternate_environment_precedence() -> None:
    evidence = _resolution_evidence()
    evidence.update(
        status="FAIL",
        failure_code="UNAUTHORIZED_ALTERNATE_ENVIRONMENT",
        alternate_venv_created=True,
        resolution_completed=False,
        candidate_artifact_created=False,
        successor_lock_candidate_sha256=None,
        resolved_package_count=None,
    )
    _validate_resolution(evidence)

    invalid = copy.deepcopy(evidence)
    invalid["alternate_venv_created"] = False
    with pytest.raises(ContractValidationError):
        _validate_resolution(invalid)


@pytest.mark.parametrize("field", [
    "predecessor_lock_git_blob_sha1",
    "predecessor_lock_sha256",
])
def test_resolution_requires_exact_predecessor_provenance(field: str) -> None:
    evidence = _resolution_evidence()
    evidence[field] = "0" * (40 if field.endswith("sha1") else 64)
    with pytest.raises(ContractValidationError):
        _validate_resolution(evidence)


@pytest.mark.parametrize("argument,wrong", [
    ("expected_direct_spec_git_blob_sha1", "0" * 40),
    ("expected_direct_spec_sha256", "0" * 64),
    ("expected_extension_design_sha", "0" * 40),
    ("expected_reviewed_resolution_implementation_sha", "0" * 40),
])
def test_resolution_requires_caller_bound_provenance(argument: str, wrong: str) -> None:
    evidence = _resolution_evidence()
    kwargs = {
        "expected_extension_design_sha": EXTENSION_SHA,
        "expected_reviewed_resolution_implementation_sha": IMPLEMENTATION_SHA,
        "expected_direct_spec_git_blob_sha1": DIRECT_BLOB_SHA,
        "expected_direct_spec_sha256": DIRECT_SHA256,
        "expected_successor_lock_candidate_sha256": CANDIDATE_SHA256,
    }
    kwargs[argument] = wrong
    with pytest.raises(ContractValidationError):
        validate_resolution_evidence(evidence, **kwargs)


def test_resolution_pass_requires_caller_bound_candidate_sha() -> None:
    evidence = _resolution_evidence()
    with pytest.raises(ContractValidationError):
        validate_resolution_evidence(
            evidence,
            expected_extension_design_sha=EXTENSION_SHA,
            expected_reviewed_resolution_implementation_sha=IMPLEMENTATION_SHA,
            expected_direct_spec_git_blob_sha1=DIRECT_BLOB_SHA,
            expected_direct_spec_sha256=DIRECT_SHA256,
            expected_successor_lock_candidate_sha256="0" * 64,
        )


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
