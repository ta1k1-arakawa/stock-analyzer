from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import v10c_t0_ml_environment_contract as contract


def _wheel(root: Path, name: str, version: str) -> dict[str, str]:
    filename_name = name.replace("-", "_")
    filename = f"{filename_name}-{version}-py3-none-any.whl"
    path = root / filename
    dist_info = f"{filename_name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{dist_info}/METADATA", f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n")
        archive.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
    inspected = contract.inspect_wheel_file(path)
    return inspected


def _packages() -> list[dict[str, str]]:
    packages = [{"name": name, "version": version} for name, version in contract.PREDECESSOR_PACKAGE_SET] + [
        {"name": "lightgbm", "version": "4.6.0"},
        {"name": "scikit-learn", "version": "1.9.0"},
    ]
    return sorted(packages, key=lambda item: (item["name"], item["version"]))


def test_constants_and_direct_spec_are_frozen() -> None:
    assert contract.FROZEN_DESIGN_SHA == "840094e09f89569b8e6345bd2e19f43298e9ddfe"
    assert contract.FROZEN_DESIGN_BLOB == "8efe1ef1d789ea22b5c070593610f158d2fad53e"
    assert contract.APPROVAL_RECORD_BLOB == "0df6ad1ae9000cafc09bab08c31195a51a346a07"
    assert contract.PREDECESSOR_LOCK_BLOB == "99395e7a5be752fb3ea92fd31be0334f38792261"
    assert contract.PREDECESSOR_LOCK_SHA256 == "eb325ac5e3417e6407400b18c8d90ca734a32e852056926e5bcd2a635e43c444"
    assert contract.validate_direct_spec_bytes(contract.DIRECT_SPEC_BYTES) == contract.DIRECT_SPEC_SHA256
    assert contract.DIRECT_SPEC_BYTES.decode() == (
        "pandas\nxlrd==2.0.2\npdfplumber==0.11.10\npandas-market-calendars==5.4.0\n"
        "lightgbm==4.6.0\nscikit-learn==1.9.0\n"
    )


def test_predecessor_and_successor_package_sets_are_exact() -> None:
    assert len(contract.PREDECESSOR_PACKAGE_SET) == 20
    result = contract.validate_resolved_packages(_packages())
    assert result["predecessor"] == contract.PREDECESSOR_PACKAGE_SET
    assert result["delta"] == (("lightgbm", "4.6.0"), ("scikit-learn", "1.9.0"))


@pytest.mark.parametrize("mutation", ["wrong_lightgbm", "wrong_sklearn", "missing_predecessor", "drift_predecessor"])
def test_pin_and_predecessor_drift_rejected(mutation: str) -> None:
    packages = _packages()
    if mutation == "wrong_lightgbm":
        packages[-2]["version"] = "4.6.1"
    elif mutation == "wrong_sklearn":
        packages[-1]["version"] = "1.9.1"
    elif mutation == "missing_predecessor":
        packages = packages[1:]
    else:
        packages[0]["version"] = "9.9.9"
    with pytest.raises(contract.ContractValidationError):
        contract.validate_resolved_packages(packages)


def test_duplicate_normalized_distribution_rejected() -> None:
    packages = _packages()
    packages.append({"name": "light_gbm", "version": "4.6.0"})
    with pytest.raises(contract.ContractValidationError):
        contract.validate_resolved_packages(packages)


def test_wheel_filename_metadata_and_hash_are_checked(tmp_path: Path) -> None:
    wheel = _wheel(tmp_path, "scikit-learn", "1.9.0")
    assert wheel["name"] == "scikit-learn"
    assert wheel["version"] == "1.9.0"
    assert len(wheel["sha256"]) == 64
    broken = dict(wheel, version="1.9.1")
    with pytest.raises(contract.ContractValidationError):
        contract.validate_wheel_manifest([broken])


def test_wheelhouse_rejects_source_distribution_and_missing_direct_package(tmp_path: Path) -> None:
    (tmp_path / "lightgbm-4.6.0.tar.gz").write_bytes(b"source")
    code, _ = contract.inspect_wheelhouse(tmp_path)
    assert code == "SOURCE_DISTRIBUTION_REQUIRED"
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _ = contract.inspect_wheelhouse(empty)
    assert code == "REQUIRED_DIRECT_DISTRIBUTION_MISSING"


def test_candidate_requires_all_packages_and_wheels(tmp_path: Path) -> None:
    wheels = tuple(_wheel(tmp_path, item["name"], item["version"]) for item in _packages())
    assert tuple((item["name"], item["version"]) for item in wheels) == tuple(sorted((item["name"], item["version"]) for item in wheels))
    with pytest.raises(contract.ContractValidationError):
        contract.validate_wheel_manifest(list(wheels) + [dict(wheels[0])])


def test_no_ml_packages_are_imported_by_contract() -> None:
    source = Path(contract.__file__).read_text(encoding="utf-8")
    assert "import lightgbm" not in source
    assert "import sklearn" not in source


def test_candidate_schema_is_closed() -> None:
    assert len(contract.CANDIDATE_KEYS) == 23
    assert len(contract.EVIDENCE_KEYS) == 27
    assert contract.CANDIDATE_SCHEMA.endswith("_V1")
    assert contract.EVIDENCE_SCHEMA.endswith("_V1")
