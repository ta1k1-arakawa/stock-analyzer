"""Pure, synthetic-contract validators for the reviewed V10 environment design.

This module deliberately performs no installation, subprocess execution, or
network access.  Callers provide all future artifact identities explicitly.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FROZEN_V10_DESIGN_SHA = "8c923ed1734c6bdfe95a743cd9e15a5156d62c03"
PREDECESSOR_LOCK_BLOB_SHA1 = "5e9d15caa822bd39e751a49cd0758db6eaf04bdf"
PREDECESSOR_LOCK_SHA256 = "ddd505cc01ac4a3a798cdf7ed9c35b3a9e56db569a421aef98c02d013dd286b7"
PREDECESSOR_PACKAGE_SET = (
    ("cffi", "2.1.1"),
    ("charset-normalizer", "3.5.1"),
    ("cryptography", "50.0.1"),
    ("numpy", "2.5.2"),
    ("pandas", "3.0.5"),
    ("pdfminer-six", "20260107"),
    ("pdfplumber", "0.11.10"),
    ("pillow", "12.3.0"),
    ("pip", "25.0.1"),
    ("pycparser", "3.0"),
    ("pypdfium2", "5.13.0"),
    ("python-dateutil", "2.9.0.post0"),
    ("six", "1.17.0"),
    ("tzdata", "2026.3"),
    ("xlrd", "2.0.2"),
)

LOCK_CANDIDATE_KEYS = frozenset(
    {
        "schema_version", "artifact_status", "frozen_v10_design_git_sha",
        "extension_design_git_sha", "reviewed_resolution_implementation_git_sha",
        "direct_spec_git_blob_sha1", "direct_spec_sha256",
        "predecessor_lock_git_blob_sha1", "predecessor_lock_sha256",
        "predecessor_package_count", "python_version", "platform_system",
        "platform_machine", "sysconfig_platform", "resolution_policy_id",
        "resolved_packages", "resolved_package_count", "resolved_wheels",
        "predecessor_pin_drift_count", "pandas_market_calendars_version",
        "exchange_calendars_version",
    }
)
RESOLUTION_EVIDENCE_KEYS = frozenset(
    {
        "schema_version", "artifact_status", "status", "failure_code",
        "frozen_v10_design_git_sha", "extension_design_git_sha",
        "reviewed_resolution_implementation_git_sha", "direct_spec_git_blob_sha1",
        "direct_spec_sha256", "predecessor_lock_git_blob_sha1",
        "predecessor_lock_sha256", "resolution_policy_id", "process_started",
        "process_exit_code", "resolution_completed", "candidate_artifact_created",
        "successor_lock_candidate_sha256", "resolved_package_count",
        "package_index_id", "package_resolution_process_invocations",
        "human_authority_consumed", "package_installations", "alternate_venv_created",
        "calendar_imports", "calendar_dates_inspected",
    }
)
PREFLIGHT_RECEIPT_KEYS = frozenset(
    {
        "schema_version", "artifact_status", "status", "failure_code",
        "frozen_v10_design_git_sha", "extension_design_git_sha",
        "reviewed_successor_lock_candidate_sha256",
        "migration_authority_git_blob_sha1", "generic_lock_git_blob_sha1",
        "wheelhouse_integrity_verified", "delta_wheel_count",
        "mutation_authority_consumed", "mutation_started",
    }
)
WHEEL_FIELDS = frozenset({"name", "version", "filename", "sha256"})
RESOLUTION_FAILURES = frozenset(
    {
        "NONE", "RESOLUTION_PROCESS_FAILURE", "RESOLUTION_REPORT_INVALID",
        "PREDECESSOR_PIN_DRIFT", "REQUIRED_DISTRIBUTION_MISSING",
        "SOURCE_DISTRIBUTION_REQUIRED", "UNAUTHORIZED_INSTALLATION",
        "UNAUTHORIZED_ALTERNATE_ENVIRONMENT",
    }
)
PREFLIGHT_FAILURES = frozenset(
    {
        "NONE", "PROVENANCE_BINDING_FAILURE",
        "PREDECESSOR_LIVE_BASELINE_MISMATCH",
        "GENERIC_SUCCESSOR_LOCK_MISMATCH",
        "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE",
    }
)
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractValidationError(ValueError):
    """Raised when a synthetic artifact violates the frozen contract."""


def normalize_distribution_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ContractValidationError("distribution name must be nonempty")
    normalized = re.sub(r"[-_.]+", "-", name.lower())
    if not normalized:
        raise ContractValidationError("distribution name normalizes empty")
    return normalized


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractValidationError(message)


def _strict_int(value: Any, message: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), message)
    return value


def _strict_bool(value: Any, message: str) -> bool:
    _require(isinstance(value, bool), message)
    return value


def _string(value: Any, message: str) -> str:
    _require(isinstance(value, str) and bool(value), message)
    return value


def _sha(value: Any, pattern: re.Pattern[str], message: str) -> str:
    _require(isinstance(value, str) and pattern.fullmatch(value) is not None, message)
    return value


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(set(value) == set(expected), f"{label} has unexpected keys")


def _validate_package_array(packages: Any, label: str) -> tuple[tuple[str, str], ...]:
    _require(isinstance(packages, list), f"{label} must be an array")
    result: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in packages:
        _exact_keys(item, frozenset({"name", "version"}), f"{label} entry")
        name = _string(item["name"], f"{label} name")
        version = _string(item["version"], f"{label} version")
        normalized = normalize_distribution_name(name)
        _require(name == normalized, f"{label} name is not normalized")
        _require(normalized not in names, f"duplicate {label} name")
        names.add(normalized)
        result.append((normalized, version))
    _require(result == sorted(result), f"{label} is not sorted")
    return tuple(result)


def derive_package_sets(
    successor_packages: Any,
    predecessor_packages: Sequence[tuple[str, str]] = PREDECESSOR_PACKAGE_SET,
) -> dict[str, tuple[tuple[str, str], ...]]:
    successor = _validate_package_array(successor_packages, "resolved_packages")
    predecessor = tuple(predecessor_packages)
    predecessor_names = {name for name, _ in predecessor}
    successor_map = dict(successor)
    _require(len(predecessor) == 15, "predecessor package set must contain 15 packages")
    for name, version in predecessor:
        if successor_map.get(name) != version:
            raise ContractValidationError("PREDECESSOR_PIN_DRIFT")
    delta = tuple((name, version) for name, version in successor if name not in predecessor_names)
    return {"predecessor": predecessor, "successor": successor, "delta": delta}


def _parse_wheel_filename(filename: str) -> tuple[str, str]:
    _require("/" not in filename and "\\" not in filename, "wheel filename must be basename")
    _require(filename.lower().endswith(".whl"), "wheel filename must end in .whl")
    parts = filename[:-4].split("-")
    _require(len(parts) in (5, 6), "malformed wheel filename")
    distribution = parts[0]
    version = parts[1]
    _require(distribution and version, "malformed wheel identity")
    return normalize_distribution_name(distribution.replace("_", "-")), version


def _validate_wheel_manifest(wheels: Any) -> tuple[dict[str, Any], ...]:
    _require(isinstance(wheels, list), "resolved_wheels must be an array")
    names: set[str] = set()
    filenames: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in wheels:
        _exact_keys(item, WHEEL_FIELDS, "resolved_wheels entry")
        name = _string(item["name"], "wheel name")
        version = _string(item["version"], "wheel version")
        filename = _string(item["filename"], "wheel filename")
        digest = _sha(item["sha256"], SHA256_RE, "wheel sha256")
        normalized = normalize_distribution_name(name)
        _require(name == normalized, "wheel name is not normalized")
        _require(normalized not in names, "duplicate normalized wheel name")
        filename_key = filename.casefold()
        _require(filename_key not in filenames, "duplicate wheel filename")
        parsed_name, parsed_version = _parse_wheel_filename(filename)
        _require(parsed_name == normalized and parsed_version == version, "wheel filename identity mismatch")
        names.add(normalized)
        filenames.add(filename_key)
        result.append({"name": normalized, "version": version, "filename": filename, "sha256": digest})
    _require([item["name"] for item in result] == sorted(item["name"] for item in result), "resolved_wheels is not sorted")
    return tuple(result)


def validate_successor_lock_candidate(
    candidate: Mapping[str, Any],
    *,
    expected_extension_design_sha: str,
    expected_reviewed_resolution_implementation_sha: str,
) -> dict[str, tuple[tuple[str, str], ...]]:
    _exact_keys(candidate, LOCK_CANDIDATE_KEYS, "lock candidate")
    _require(candidate["schema_version"] == "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE_V2", "candidate schema")
    _require(candidate["artifact_status"] == "WINDOWS_RESOLUTION_CANDIDATE_NOT_INSTALL_AUTHORITY", "candidate status")
    _require(candidate["frozen_v10_design_git_sha"] == FROZEN_V10_DESIGN_SHA, "frozen design SHA")
    _require(candidate["extension_design_git_sha"] == expected_extension_design_sha, "extension design SHA")
    _require(candidate["reviewed_resolution_implementation_git_sha"] == expected_reviewed_resolution_implementation_sha, "resolution implementation SHA")
    _sha(candidate["extension_design_git_sha"], SHA1_RE, "extension design SHA")
    _sha(candidate["reviewed_resolution_implementation_git_sha"], SHA1_RE, "resolution implementation SHA")
    _sha(candidate["frozen_v10_design_git_sha"], SHA1_RE, "frozen design SHA")
    _sha(candidate["direct_spec_git_blob_sha1"], SHA1_RE, "direct spec blob SHA")
    _sha(candidate["predecessor_lock_git_blob_sha1"], SHA1_RE, "predecessor lock blob SHA")
    _sha(candidate["direct_spec_sha256"], SHA256_RE, "direct spec SHA")
    _require(candidate["predecessor_lock_git_blob_sha1"] == PREDECESSOR_LOCK_BLOB_SHA1, "predecessor lock blob")
    _require(candidate["predecessor_lock_sha256"] == PREDECESSOR_LOCK_SHA256, "predecessor lock SHA")
    _sha(candidate["predecessor_lock_sha256"], SHA256_RE, "predecessor lock SHA")
    _require(candidate["predecessor_package_count"] == 15, "predecessor count")
    _require(candidate["python_version"] == "3.12.10", "python version")
    _require(candidate["platform_system"] == "Windows", "platform system")
    _require(candidate["platform_machine"] == "AMD64", "platform machine")
    _require(candidate["sysconfig_platform"] == "win-amd64", "sysconfig platform")
    _require(candidate["resolution_policy_id"] == "PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1", "resolution policy")
    packages = _validate_package_array(candidate["resolved_packages"], "resolved_packages")
    _require(candidate["resolved_package_count"] == len(packages), "resolved package count")
    _require(candidate["resolved_package_count"] > 15, "successor package count")
    _require(candidate["predecessor_pin_drift_count"] == 0, "predecessor pin drift count")
    package_sets = derive_package_sets(candidate["resolved_packages"])
    package_map = dict(packages)
    _require(package_map.get("pandas-market-calendars") == "5.4.0", "pandas-market-calendars pin")
    _require("exchange-calendars" in package_map, "exchange-calendars missing")
    _require(candidate["pandas_market_calendars_version"] == "5.4.0", "pandas-market-calendars version")
    _require(candidate["exchange_calendars_version"] == package_map["exchange-calendars"], "exchange-calendars version")
    wheels = _validate_wheel_manifest(candidate["resolved_wheels"])
    _require(len(wheels) == len(packages), "wheel/package count mismatch")
    wheel_pairs = tuple((item["name"], item["version"]) for item in wheels)
    _require(wheel_pairs == packages, "wheel/package identity mismatch")
    return package_sets


def _validate_common_sha_fields(value: Mapping[str, Any], expected: Mapping[str, str] | None = None) -> None:
    for key, pattern in (
        ("frozen_v10_design_git_sha", SHA1_RE),
        ("extension_design_git_sha", SHA1_RE),
        ("reviewed_resolution_implementation_git_sha", SHA1_RE),
        ("direct_spec_git_blob_sha1", SHA1_RE),
        ("predecessor_lock_git_blob_sha1", SHA1_RE),
        ("direct_spec_sha256", SHA256_RE),
        ("predecessor_lock_sha256", SHA256_RE),
    ):
        _sha(value[key], pattern, key)
    if expected:
        for key, expected_value in expected.items():
            _require(value[key] == expected_value, f"unexpected {key}")


def validate_resolution_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_extension_design_sha: str,
    expected_reviewed_resolution_implementation_sha: str,
    expected_direct_spec_git_blob_sha1: str,
    expected_direct_spec_sha256: str,
    expected_successor_lock_candidate_sha256: str | None,
) -> None:
    _exact_keys(evidence, RESOLUTION_EVIDENCE_KEYS, "resolution evidence")
    _require(evidence["schema_version"] == "V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE_V1", "resolution schema")
    _require(evidence["artifact_status"] == "WINDOWS_RESOLUTION_EVIDENCE", "resolution artifact status")
    _require(evidence["status"] in {"PASS", "FAIL"}, "resolution status")
    _require(evidence["failure_code"] in RESOLUTION_FAILURES, "resolution failure code")
    for value, pattern, label in (
        (expected_extension_design_sha, SHA1_RE, "expected extension design SHA"),
        (expected_reviewed_resolution_implementation_sha, SHA1_RE, "expected resolution implementation SHA"),
        (expected_direct_spec_git_blob_sha1, SHA1_RE, "expected direct spec blob SHA"),
        (expected_direct_spec_sha256, SHA256_RE, "expected direct spec SHA"),
    ):
        _sha(value, pattern, label)
    expected = {
        "frozen_v10_design_git_sha": FROZEN_V10_DESIGN_SHA,
        "extension_design_git_sha": expected_extension_design_sha,
        "reviewed_resolution_implementation_git_sha": expected_reviewed_resolution_implementation_sha,
        "direct_spec_git_blob_sha1": expected_direct_spec_git_blob_sha1,
        "direct_spec_sha256": expected_direct_spec_sha256,
        "predecessor_lock_git_blob_sha1": PREDECESSOR_LOCK_BLOB_SHA1,
        "predecessor_lock_sha256": PREDECESSOR_LOCK_SHA256,
    }
    _validate_common_sha_fields(evidence, expected)
    _require(evidence["resolution_policy_id"] == "PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1", "resolution policy")
    _require(evidence["package_index_id"] == "PYPI_OFFICIAL_SIMPLE", "package index")
    process_started = _strict_bool(evidence["process_started"], "process_started")
    exit_code = evidence["process_exit_code"]
    if process_started:
        _strict_int(exit_code, "started process exit code")
    else:
        _require(exit_code is None, "non-started process must have null exit code")
    for key in ("resolution_completed", "candidate_artifact_created", "human_authority_consumed", "alternate_venv_created"):
        _strict_bool(evidence[key], key)
    for key in ("package_installations", "calendar_imports", "calendar_dates_inspected"):
        _require(_strict_int(evidence[key], key) >= 0, key)
    invocations = _strict_int(evidence["package_resolution_process_invocations"], "invocation count")
    _require(invocations in {0, 1}, "invocation count")
    candidate_sha = evidence["successor_lock_candidate_sha256"]
    package_count = evidence["resolved_package_count"]
    installations = _strict_int(evidence["package_installations"], "package installations")
    alternate_venv_created = evidence["alternate_venv_created"]

    # The frozen precedence records unauthorized operations before any
    # resolver/process or Phase-C failure classification.
    if installations > 0:
        _require(evidence["status"] == "FAIL", "unauthorized installation must be FAIL")
        _require(evidence["failure_code"] == "UNAUTHORIZED_INSTALLATION", "installation precedence")
        _require(not evidence["candidate_artifact_created"], "unauthorized installation candidate")
        _require(candidate_sha is None and package_count is None, "unauthorized installation candidate fields")
        return
    _require(evidence["failure_code"] != "UNAUTHORIZED_INSTALLATION", "installation failure requires installation")
    if alternate_venv_created:
        _require(evidence["status"] == "FAIL", "unauthorized alternate environment must be FAIL")
        _require(evidence["failure_code"] == "UNAUTHORIZED_ALTERNATE_ENVIRONMENT", "alternate environment precedence")
        _require(not evidence["candidate_artifact_created"], "unauthorized alternate environment candidate")
        _require(candidate_sha is None and package_count is None, "unauthorized alternate environment candidate fields")
        return
    _require(
        evidence["failure_code"] != "UNAUTHORIZED_ALTERNATE_ENVIRONMENT",
        "alternate environment failure requires alternate environment",
    )
    if evidence["status"] == "PASS":
        _require(process_started and exit_code == 0, "PASS requires started zero-exit process")
        _require(evidence["failure_code"] == "NONE", "PASS failure code")
        _require(invocations == 1 and evidence["human_authority_consumed"], "PASS attempt semantics")
        _require(evidence["resolution_completed"] and evidence["candidate_artifact_created"], "PASS completion")
        _require(installations == 0, "PASS package installations")
        _require(not alternate_venv_created, "PASS alternate environment")
        _require(evidence["calendar_imports"] == 0, "PASS calendar imports")
        _require(evidence["calendar_dates_inspected"] == 0, "PASS calendar dates inspected")
        _sha(candidate_sha, SHA256_RE, "candidate SHA")
        _require(expected_successor_lock_candidate_sha256 is not None, "PASS expected candidate SHA")
        _sha(expected_successor_lock_candidate_sha256, SHA256_RE, "expected candidate SHA")
        _require(candidate_sha == expected_successor_lock_candidate_sha256, "unexpected candidate SHA")
        _require(_strict_int(package_count, "package count") > 15, "PASS package count")
    elif not process_started:
        _require(evidence["failure_code"] == "RESOLUTION_PROCESS_FAILURE", "launch failure code")
        _require(invocations == 0 and evidence["human_authority_consumed"], "launch failure attempt semantics")
        _require(not evidence["resolution_completed"] and not evidence["candidate_artifact_created"], "launch failure completion")
        _require(candidate_sha is None and package_count is None, "launch failure candidate fields")
    else:
        _require(invocations == 1 and evidence["human_authority_consumed"], "started failure attempt semantics")
        _require(not evidence["resolution_completed"] and not evidence["candidate_artifact_created"], "failed resolution completion")
        _require(candidate_sha is None and package_count is None, "failed resolution candidate fields")
        if exit_code != 0:
            _require(evidence["failure_code"] == "RESOLUTION_PROCESS_FAILURE", "nonzero exit failure code")
        else:
            _require(evidence["failure_code"] not in {"NONE", "RESOLUTION_PROCESS_FAILURE"}, "zero-exit failed evidence code")


def validate_mutation_preflight_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_extension_design_sha: str,
    expected_successor_lock_candidate_sha256: str,
    expected_migration_authority_git_blob_sha1: str,
    expected_generic_lock_git_blob_sha1: str,
) -> None:
    _exact_keys(receipt, PREFLIGHT_RECEIPT_KEYS, "mutation preflight receipt")
    _require(receipt["schema_version"] == "V10_CANONICAL_ENVIRONMENT_MUTATION_PREFLIGHT_RECEIPT_V1", "receipt schema")
    _require(receipt["artifact_status"] == "V10_CANONICAL_ENVIRONMENT_MUTATION_PREFLIGHT_RECEIPT", "receipt artifact status")
    _require(receipt["status"] in {"PASS", "FAIL"}, "receipt status")
    _require(receipt["failure_code"] in PREFLIGHT_FAILURES, "receipt failure code")
    _require(receipt["frozen_v10_design_git_sha"] == FROZEN_V10_DESIGN_SHA, "receipt frozen design SHA")
    _sha(receipt["frozen_v10_design_git_sha"], SHA1_RE, "receipt frozen design SHA")
    for key, pattern, expected in (
        ("extension_design_git_sha", SHA1_RE, expected_extension_design_sha),
        ("reviewed_successor_lock_candidate_sha256", SHA256_RE, expected_successor_lock_candidate_sha256),
        ("migration_authority_git_blob_sha1", SHA1_RE, expected_migration_authority_git_blob_sha1),
        ("generic_lock_git_blob_sha1", SHA1_RE, expected_generic_lock_git_blob_sha1),
    ):
        _sha(receipt[key], pattern, key)
        _require(receipt[key] == expected, f"receipt {key}")
    integrity = receipt["wheelhouse_integrity_verified"]
    _require(integrity is None or isinstance(integrity, bool), "wheelhouse integrity nullable boolean")
    count = receipt["delta_wheel_count"]
    _require(count is None or (isinstance(count, int) and not isinstance(count, bool) and count >= 0), "delta wheel count")
    _require(receipt["mutation_authority_consumed"] is False, "mutation authority consumed")
    _require(receipt["mutation_started"] is False, "mutation started")
    failure = receipt["failure_code"]
    if receipt["status"] == "PASS":
        _require(failure == "NONE" and integrity is True and count is not None, "receipt PASS semantics")
    else:
        _require(failure != "NONE", "receipt FAIL semantics")
        if failure in {"PROVENANCE_BINDING_FAILURE", "PREDECESSOR_LIVE_BASELINE_MISMATCH", "GENERIC_SUCCESSOR_LOCK_MISMATCH"}:
            _require(integrity is None and count is None, "early receipt failure null semantics")
        elif failure == "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE":
            _require(integrity is False and count is not None, "wheelhouse receipt failure semantics")


def inspect_wheel_file(wheel_path: str | Path) -> dict[str, str]:
    path = Path(wheel_path)
    filename = path.name
    filename_name, filename_version = _parse_wheel_filename(filename)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ContractValidationError("wheel ZIP integrity failure")
            members = archive.infolist()
            metadata_members = [m for m in members if m.filename.endswith(".dist-info/METADATA")]
            wheel_members = [m for m in members if m.filename.endswith(".dist-info/WHEEL")]
            _require(len(metadata_members) == 1 and len(wheel_members) == 1, "wheel metadata cardinality")
            metadata = BytesParser(policy=compat32).parsebytes(archive.read(metadata_members[0]))
            wheel_metadata = BytesParser(policy=compat32).parsebytes(archive.read(wheel_members[0]))
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as error:
        raise ContractValidationError("unreadable wheel") from error
    names = metadata.get_all("Name") or []
    versions = metadata.get_all("Version") or []
    _require(len(names) == 1 and len(versions) == 1, "wheel Name/Version fields")
    _require(len(wheel_metadata.get_all("Wheel-Version") or []) == 1, "wheel version metadata")
    name = normalize_distribution_name(str(names[0]))
    version = str(versions[0])
    _require(version != "", "wheel metadata version")
    _require(name == filename_name and version == filename_version, "wheel metadata/filename identity")
    return {"name": name, "version": version, "filename": filename, "sha256": digest}


def verify_reviewed_wheelhouse(
    wheelhouse: str | Path,
    resolved_wheels: Sequence[Mapping[str, Any]],
    successor_packages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        package_sets = derive_package_sets(list(successor_packages))
    except (ContractValidationError, TypeError, ValueError) as error:
        # No wheelhouse observation occurred if the successor/delta authority
        # could not be established.  Let the enclosing preflight classify the
        # earlier prerequisite failure rather than mislabeling it as a
        # checked-and-failed wheelhouse.
        if isinstance(error, ContractValidationError):
            raise
        raise ContractValidationError("invalid successor package set") from error

    delta = package_sets["delta"]
    try:
        successor = package_sets["successor"]
        expected = _validate_wheel_manifest(list(resolved_wheels))
        wheel_pairs = tuple((item["name"], item["version"]) for item in expected)
        _require(wheel_pairs == successor, "wheel/package identity mismatch")
        root = Path(wheelhouse)
        _require(root.is_dir(), "wheelhouse missing")
        entries = list(root.iterdir())
        _require(all(entry.is_file() for entry in entries), "wheelhouse contains directory")
        actual_by_casefold = {entry.name.casefold(): entry for entry in entries}
        _require(len(actual_by_casefold) == len(entries), "duplicate wheel filename")
        expected_by_casefold = {item["filename"].casefold(): item for item in expected}
        _require(set(actual_by_casefold) == set(expected_by_casefold), "wheelhouse file set mismatch")
        for item in expected:
            actual = actual_by_casefold[item["filename"].casefold()]
            inspected = inspect_wheel_file(actual)
            _require(inspected == item, "wheelhouse wheel identity or hash mismatch")
        paths: list[Path] = []
        for pair in delta:
            matches = [item for item in expected if (item["name"], item["version"]) == pair]
            _require(len(matches) == 1, "delta wheel identity is not one-to-one")
            paths.append(actual_by_casefold[matches[0]["filename"].casefold()])
        _require(len(paths) == len(delta), "delta wheel count mismatch")
        _require(
            tuple((item["name"], item["version"]) for item in expected if (item["name"], item["version"]) in delta)
            == delta,
            "delta wheel ordering mismatch",
        )
        return {
            "ok": True,
            "failure_code": "NONE",
            "wheelhouse_integrity_verified": True,
            "delta_wheel_count": len(paths),
            "delta_packages": delta,
            "delta_wheel_paths": tuple(paths),
        }
    except (OSError, ContractValidationError, TypeError, ValueError) as error:
        return {
            "ok": False,
            "failure_code": "REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE",
            "wheelhouse_integrity_verified": False,
            "delta_wheel_count": len(delta),
            "delta_packages": delta,
            "delta_wheel_paths": tuple(),
            "reason": str(error),
        }


def build_exact_delta_install_argv(canonical_python: str | Path, delta_wheel_paths: Iterable[str | Path]) -> list[str]:
    interpreter = str(canonical_python)
    _require(interpreter != "", "canonical interpreter")
    paths = [str(path) for path in delta_wheel_paths]
    _require(paths, "delta wheel paths must be nonempty")
    for path in paths:
        _require(path and not path.startswith("-"), "wheel path must be local")
        _require("==" not in path and "://" not in path, "package specifier or URL is forbidden")
        _require(Path(path).name.lower().endswith(".whl"), "install path must be a wheel")
    return [interpreter, "-m", "pip", "install", "--no-deps", "--no-index", *paths]
