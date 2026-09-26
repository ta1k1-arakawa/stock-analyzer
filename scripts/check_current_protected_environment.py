"""Resolve and validate the current protected real-execution environment.

This checker is deliberately separate from ``check_real_execution_env.py``.
The latter remains the historical V9_014 generic 15-package checker and is
kept independently testable for provenance and regression purposes.  This
module resolves the current V10C 27-package authority from its reviewed
promotion/final-freeze chain and never treats the historical checker or its
lock as the current authority.

The checker is no-network, no-private-data, and read-only. It launches an
isolated child through the exact canonical interpreter; that child observes
installed package identity through ``importlib.metadata`` and runs the
reviewed synthetic PDF operational probe. It never installs packages, opens
research payloads, or consumes a gate. A missing, ambiguous, stale, or
contradictory authority fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_STATE_PATH = REPO_ROOT / "PROJECT_STATE.md"
CURRENT_LOCK_PATH = REPO_ROOT / "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK.txt"
FINAL_FREEZE_RECORD_PATH = (
    REPO_ROOT / "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_RECORD_CANDIDATE.json"
)
FINAL_FREEZE_EVIDENCE_PATH = (
    REPO_ROOT / "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_SAFE_EVIDENCE.json"
)
RESOLUTION_PROMOTION_PATH = (
    REPO_ROOT / "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_RESOLUTION_PROMOTION.json"
)

CURRENT_AUTHORITY_REVIEWED_SHA = "92cc4a964af798f9543155e32eb83a12e7352151"
CURRENT_AUTHORITY_REVIEWED_PARENT_SHA = "d574015cb148f9e172d73b2ad74b5f99e02ff5ea"
CURRENT_AUTHORITY_STUDY = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR"
CURRENT_AUTHORITY_STATE = "CANONICAL_FROZEN"
CURRENT_AUTHORITY_LOCK_BLOB_SHA1 = "13636e58fbe40071be04cbfa57c3990c1d8ff2e0"
CURRENT_AUTHORITY_LOCK_SHA256 = "f38dd4c7319465bb7e6ff429e8dff4a476d9966c744b19e50264dcc0b18e8300"
CURRENT_AUTHORITY_LOCK_PACKAGE_COUNT = 27
CURRENT_AUTHORITY_LOCK_PATH = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK.txt"
CURRENT_AUTHORITY_RECORD_BLOB_SHA1 = "23d2f8c5b1dc12f22aed262a918b0ba18fceeaf1"
CURRENT_AUTHORITY_EVIDENCE_BLOB_SHA1 = "bffa5180ebe2930fca5279eac76e3cba83cea855"
CURRENT_AUTHORITY_PROMOTION_BLOB_SHA1 = "b866e6d77508d6366569ee6a229587c59c3c8be2"
CURRENT_AUTHORITY_PREDECESSOR_LOCK_BLOB_SHA1 = "99395e7a5be752fb3ea92fd31be0334f38792261"
CURRENT_AUTHORITY_PREDECESSOR_LOCK_SHA256 = "eb325ac5e3417e6407400b18c8d90ca734a32e852056926e5bcd2a635e43c444"
CURRENT_AUTHORITY_PREDECESSOR_PACKAGE_COUNT = 20
CURRENT_AUTHORITY_SAFE_EVIDENCE_SHA256 = "b83d02921e571545511e477d4783ed408b8700b790d52c1f7d51d93d850d8ba3"

CANONICAL_VENV_DIR = REPO_ROOT / ".venv-real-execution"
CANONICAL_INTERPRETER = CANONICAL_VENV_DIR / "Scripts" / "python.exe"
CANONICAL_PYTHON_VERSION = (3, 12, 10)

PDF_PROBE_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "v9_014_synthetic_pdf_env_probe.pdf"
PDF_PROBE_EXPECTED_FIXTURE_SHA256 = "5eecb758a50e829af16bd42833f89a8329bfaaaa561aee209fbd2249b507b413"
PDF_PROBE_REQUIRED_PDFPLUMBER_VERSION = "0.11.10"
PDF_PROBE_EXPECTED_PAGE_COUNT = 1
PDF_PROBE_PASS = "SYNTHETIC_PDF_PROBE_PASS"
ISOLATED_CHILD_SCHEMA = "V12_CURRENT_PROTECTED_ENVIRONMENT_ISOLATED_OBSERVER_V1"
ISOLATED_CHILD_TOP_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "isolated",
        "no_user_site",
        "pythonpath_env_absent",
        "bytecode_disabled",
        "python_implementation",
        "python_version",
        "executable",
        "platform_system",
        "platform_machine",
        "sysconfig_platform",
        "packages",
        "package_observation_status",
        "package_failure",
        "pdf_probe",
    }
)
ISOLATED_CHILD_PDF_KEYS = frozenset(
    {"status", "fixture_sha256", "pdfplumber_version", "page_count"}
)

_PINNED_LINE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^\s#]+)$")
_KEY_LINE = re.compile(r"^(?P<key>[A-Z0-9_]+)=(?P<value>.*)$")

GitBlobReader = Callable[[Path, str], bytes | None]


def _git_blob_bytes(repo_root: Path, git_ref: str) -> bytes | None:
    """Read one already-present Git object; this function never fetches."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "blob", git_ref],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git_blob_sha1(blob_bytes: bytes) -> str:
    header = f"blob {len(blob_bytes)}\0".encode("ascii")
    return hashlib.sha1(header + blob_bytes).hexdigest()


def _working_blob_sha1(repo_root: Path, relative_name: str, path: Path) -> str | None:
    """Hash a tracked worktree artifact using its Git path's clean rules."""

    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "hash-object", "--path", relative_name,
             "--", str(path)],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    blob = result.stdout.decode("ascii", errors="replace").strip()
    return blob if re.fullmatch(r"[0-9a-f]{40}", blob) else None


def _git_commit_parent(repo_root: Path, commit_sha: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", "-s", "--format=%P", commit_sha],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    parents = result.stdout.decode("ascii", errors="strict").strip().split()
    return parents[0] if len(parents) == 1 else None


def _normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_pinned_lines(text: str) -> tuple[dict[str, str], list[str], list[str]]:
    packages: dict[str, str] = {}
    invalid: list[str] = []
    duplicates: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _PINNED_LINE.fullmatch(line)
        if match is None:
            invalid.append(raw_line)
            continue
        normalized = _normalize_package_name(match.group("name"))
        if normalized in packages:
            duplicates.append(raw_line)
            continue
        packages[normalized] = match.group("version")
    return packages, invalid, duplicates


def _state_values(state_text: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for line in state_text.splitlines():
        match = _KEY_LINE.fullmatch(line.strip())
        if match is None:
            continue
        values.setdefault(match.group("key"), []).append(match.group("value"))
    return values


def _one_state_value(values: dict[str, list[str]], key: str) -> tuple[str | None, str | None]:
    matches = values.get(key, [])
    if not matches:
        return None, "CURRENT_AUTHORITY_STATE_KEY_MISSING"
    if len(matches) != 1:
        return None, "CURRENT_AUTHORITY_STATE_KEY_AMBIGUOUS"
    return matches[0], None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _current_chain_state_ok(values: dict[str, list[str]]) -> tuple[bool, str | None]:
    expected = {
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_STATE": "CANONICAL_FROZEN",
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_PROMOTED": "true",
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_ENVIRONMENT_FROZEN": "true",
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_ENVIRONMENT_STATE": "CANONICAL_FROZEN",
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_GLOBAL_T0_READINESS": "NO",
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_HISTORICAL_EVALUATION_AUTHORIZED": "false",
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_PRIVATE_SEALED_ACCESS_AUTHORIZED": "false",
        "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_FUTURE_PROFITABILITY_ESTABLISHED": "false",
        "V10C_T0_ML_RESOLUTION_PROMOTION": "GPT_REVIEWED_PASS",
        "V10C_T0_ML_SUCCESSOR_LOCK_PROMOTED": "true",
        "V10C_T0_ML_SUCCESSOR_FINAL_FREEZE_F9_REVIEWED_SHA": CURRENT_AUTHORITY_REVIEWED_SHA,
        "V10C_T0_ML_SUCCESSOR_FINAL_FREEZE_F9_REVIEW_RESULT": "PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_0",
        "V10C_T0_ML_SUCCESSOR_FINAL_FREEZE_F9_PARENT_SHA": CURRENT_AUTHORITY_REVIEWED_PARENT_SHA,
        "V10C_T0_ML_SUCCESSOR_FINAL_FREEZE_F9_SAFE_EVIDENCE_REVIEW": "PASS",
    }
    for key, expected_value in expected.items():
        actual, reason = _one_state_value(values, key)
        if reason is not None:
            return False, reason
        if actual != expected_value:
            return False, "CURRENT_AUTHORITY_STATE_VALUE_MISMATCH"
    return True, None


def resolve_current_authority(
    *,
    repo_root: Path = REPO_ROOT,
    state_path: Path = PROJECT_STATE_PATH,
    lock_path: Path = CURRENT_LOCK_PATH,
    record_path: Path = FINAL_FREEZE_RECORD_PATH,
    evidence_path: Path = FINAL_FREEZE_EVIDENCE_PATH,
    promotion_path: Path = RESOLUTION_PROMOTION_PATH,
    git_blob_reader: GitBlobReader = _git_blob_bytes,
) -> dict[str, Any]:
    """Resolve exactly one reviewed current authority, failing closed."""

    try:
        state_values = _state_values(state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_STATE_UNREADABLE"}

    state_ok, state_reason = _current_chain_state_ok(state_values)
    if not state_ok:
        return {"status": "FAIL", "reason": state_reason}

    if _git_commit_parent(repo_root, CURRENT_AUTHORITY_REVIEWED_SHA) != CURRENT_AUTHORITY_REVIEWED_PARENT_SHA:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_REVIEW_COMMIT_PARENT_MISMATCH"}

    artifact_paths = {
        "lock": (CURRENT_AUTHORITY_LOCK_PATH, lock_path, CURRENT_AUTHORITY_LOCK_BLOB_SHA1),
        "final_freeze_record": (str(FINAL_FREEZE_RECORD_PATH.name), record_path, CURRENT_AUTHORITY_RECORD_BLOB_SHA1),
        "final_freeze_evidence": (str(FINAL_FREEZE_EVIDENCE_PATH.name), evidence_path, CURRENT_AUTHORITY_EVIDENCE_BLOB_SHA1),
        "resolution_promotion": (str(RESOLUTION_PROMOTION_PATH.name), promotion_path, CURRENT_AUTHORITY_PROMOTION_BLOB_SHA1),
    }
    artifacts: dict[str, bytes] = {}
    for label, (relative_name, path, expected_blob) in artifact_paths.items():
        reviewed = git_blob_reader(repo_root, f"{CURRENT_AUTHORITY_REVIEWED_SHA}:{relative_name}")
        if reviewed is None or _git_blob_sha1(reviewed) != expected_blob:
            return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_REVIEWED_ARTIFACT_UNAVAILABLE", "artifact": label}
        if not path.is_file():
            return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_WORKING_ARTIFACT_UNREADABLE", "artifact": label}
        if _working_blob_sha1(repo_root, relative_name, path) != expected_blob:
            return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_WORKING_ARTIFACT_MISMATCH", "artifact": label}
        artifacts[label] = reviewed

    lock_text = artifacts["lock"].decode("utf-8")
    package_map, invalid_lines, duplicate_lines = _parse_pinned_lines(lock_text)
    if invalid_lines or duplicate_lines or len(package_map) != CURRENT_AUTHORITY_LOCK_PACKAGE_COUNT:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_LOCK_SCHEMA_INVALID"}
    if hashlib.sha256(artifacts["lock"]).hexdigest() != CURRENT_AUTHORITY_LOCK_SHA256:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_LOCK_SHA256_MISMATCH"}

    record = json.loads(artifacts["final_freeze_record"].decode("utf-8"))
    promotion = json.loads(artifacts["resolution_promotion"].decode("utf-8"))
    evidence = json.loads(artifacts["final_freeze_evidence"].decode("utf-8"))
    if not isinstance(record, dict) or not isinstance(promotion, dict) or not isinstance(evidence, dict):
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_ARTIFACT_SCHEMA_INVALID"}
    if (
        record.get("study") != CURRENT_AUTHORITY_STUDY
        or record.get("successor_lock_git_blob_sha1") != CURRENT_AUTHORITY_LOCK_BLOB_SHA1
        or record.get("successor_lock_sha256") != CURRENT_AUTHORITY_LOCK_SHA256
        or record.get("successor_package_count") != CURRENT_AUTHORITY_LOCK_PACKAGE_COUNT
        or record.get("predecessor_lock_git_blob_sha1") != CURRENT_AUTHORITY_PREDECESSOR_LOCK_BLOB_SHA1
        or record.get("predecessor_lock_sha256") != CURRENT_AUTHORITY_PREDECESSOR_LOCK_SHA256
        or record.get("predecessor_package_count") != CURRENT_AUTHORITY_PREDECESSOR_PACKAGE_COUNT
        or record.get("environment_state") != "MUTATED_VALIDATED_NOT_FROZEN"
        or record.get("environment_frozen") is not False
    ):
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_FINAL_FREEZE_BINDING_MISMATCH"}
    if (
        promotion.get("successor_lock_sha256") != CURRENT_AUTHORITY_LOCK_SHA256
        or promotion.get("successor_lock_package_count") != CURRENT_AUTHORITY_LOCK_PACKAGE_COUNT
        or promotion.get("predecessor_package_count") != CURRENT_AUTHORITY_PREDECESSOR_PACKAGE_COUNT
        or promotion.get("successor_lock_file") != CURRENT_AUTHORITY_LOCK_PATH
    ):
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_PROMOTION_BINDING_MISMATCH"}
    if (
        evidence.get("study") != CURRENT_AUTHORITY_STUDY
        or evidence.get("package_count") != CURRENT_AUTHORITY_LOCK_PACKAGE_COUNT
        or evidence.get("canonical_environment_promoted") is not False
        or evidence.get("environment_frozen") is not False
        or evidence.get("global_t0_readiness") != "NO"
        or evidence.get("t0_authorized") is not False
        or evidence.get("future_profitability_established") is not False
        or hashlib.sha256(artifacts["final_freeze_evidence"]).hexdigest() != CURRENT_AUTHORITY_SAFE_EVIDENCE_SHA256
    ):
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_FINAL_FREEZE_EVIDENCE_MISMATCH"}

    return {
        "status": "PASS",
        "study": CURRENT_AUTHORITY_STUDY,
        "authority_state": CURRENT_AUTHORITY_STATE,
        "reviewed_sha": CURRENT_AUTHORITY_REVIEWED_SHA,
        "lock_path": CURRENT_AUTHORITY_LOCK_PATH,
        "lock_sha256": CURRENT_AUTHORITY_LOCK_SHA256,
        "package_count": CURRENT_AUTHORITY_LOCK_PACKAGE_COUNT,
        "package_map": package_map,
        "historical_predecessor_package_count": CURRENT_AUTHORITY_PREDECESSOR_PACKAGE_COUNT,
        "historical_predecessor_lock_blob_sha1": CURRENT_AUTHORITY_PREDECESSOR_LOCK_BLOB_SHA1,
    }


def check_interpreter_identity() -> dict[str, Any]:
    actual = Path(sys.executable).resolve(strict=False)
    expected = CANONICAL_INTERPRETER.resolve(strict=False)
    interpreter_match = os.path.normcase(str(actual)) == os.path.normcase(str(expected))
    version = tuple(sys.version_info[:3])
    version_match = version == CANONICAL_PYTHON_VERSION
    platform_match = (
        os.name == "nt"
        and platform.python_implementation() == "CPython"
        and platform.system() == "Windows"
        and platform.machine() == "AMD64"
        and sysconfig.get_platform() == "win-amd64"
    )
    identity_match = interpreter_match and version_match and platform_match
    return {
        "status": "PASS" if identity_match else "FAIL",
        "interpreter_match": interpreter_match,
        "python_version": ".".join(str(item) for item in version),
        "python_patch_match": version_match,
        "platform_match": platform_match,
        "failure_class": None if identity_match else "PRE_GATE_WRONG_PYTHON_ENVIRONMENT",
    }


def _package_map_from_installed_metadata(distributions: Any) -> tuple[dict[str, str] | None, str | None]:
    actual: dict[str, str] = {}
    try:
        for distribution in distributions:
            name = distribution.metadata.get("Name")
            version = distribution.version
            if not isinstance(name, str) or not isinstance(version, str) or not name or not version:
                return None, "CURRENT_AUTHORITY_INSTALLED_METADATA_MALFORMED"
            normalized = _normalize_package_name(name)
            if normalized in actual:
                return None, "CURRENT_AUTHORITY_INSTALLED_METADATA_DUPLICATE"
            actual[normalized] = version
    except Exception:
        return None, "CURRENT_AUTHORITY_INSTALLED_METADATA_OBSERVATION_FAILED"
    return actual, None


def check_live_package_set(
    expected_packages: dict[str, str], observed_distributions: Any = None
) -> dict[str, Any]:
    """Validate injected metadata records; production uses the isolated child."""

    if observed_distributions is None:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_ISOLATED_CHILD_REQUIRED"}
    actual, observation_reason = _package_map_from_installed_metadata(observed_distributions)
    if observation_reason is not None or actual is None:
        return {"status": "FAIL", "reason": observation_reason}
    if actual != expected_packages:
        return {
            "status": "FAIL",
            "reason": "CURRENT_AUTHORITY_PACKAGE_SET_MISMATCH",
            "missing_packages": sorted(set(expected_packages) - set(actual)),
            "extra_packages": sorted(set(actual) - set(expected_packages)),
            "version_mismatches": sorted(
                key for key in set(expected_packages) & set(actual) if expected_packages[key] != actual[key]
            ),
        }
    return {"status": "PASS", "package_count": len(actual)}


def _isolated_child_script() -> str:
    """Return the reviewed-F6-style isolated observer program."""

    return r'''
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
import sysconfig
from pathlib import Path

SCHEMA = "V12_CURRENT_PROTECTED_ENVIRONMENT_ISOLATED_OBSERVER_V1"
PDF_PASS = "SYNTHETIC_PDF_PROBE_PASS"


def emit(status, packages, package_status, package_failure, pdf_probe):
    result = {
        "schema_version": SCHEMA,
        "status": status,
        "isolated": type(sys.flags.isolated) is int and sys.flags.isolated == 1,
        "no_user_site": type(sys.flags.no_user_site) is int and sys.flags.no_user_site == 1,
        "pythonpath_env_absent": "PYTHONPATH" not in os.environ,
        "bytecode_disabled": sys.dont_write_bytecode is True,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "executable": str(Path(sys.executable).resolve(strict=False)),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "sysconfig_platform": sysconfig.get_platform(),
        "packages": packages,
        "package_observation_status": package_status,
        "package_failure": package_failure,
        "pdf_probe": pdf_probe,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


def empty_pdf_probe():
    return {"status": "FAIL", "fixture_sha256": None, "pdfplumber_version": None, "page_count": None}


packages = []
package_status = "PASS"
package_failure = None
pdf_probe = empty_pdf_probe()
try:
    if len(sys.argv) != 2:
        raise ValueError("REPOSITORY_ROOT_ARGUMENT_INVALID")
    repository_root = Path(sys.argv[1]).resolve(strict=True)
    probe_module_path = repository_root / "scripts" / "v9_014_pdf_env_successor.py"
    fixture_path = repository_root / "tests" / "fixtures" / "v9_014_synthetic_pdf_env_probe.pdf"
    if not probe_module_path.is_file() or not fixture_path.is_file():
        raise OSError("REVIEWED_PROBE_ARTIFACT_MISSING")

    names = set()
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(name, str) or not isinstance(version, str) or not name or not version:
            package_status = "FAIL"
            package_failure = "MALFORMED_METADATA"
            break
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized in names:
            package_status = "FAIL"
            package_failure = "DUPLICATE_NORMALIZED_METADATA"
            break
        names.add(normalized)
        packages.append({"name": name, "version": version})

    module_spec = importlib.util.spec_from_file_location("_v12_reviewed_pdf_probe", probe_module_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError("REVIEWED_PROBE_LOAD_FAILED")
    probe_module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = probe_module
    module_spec.loader.exec_module(probe_module)
    probe_result = probe_module.run_synthetic_pdf_operational_probe(fixture_path=fixture_path)
    pdf_probe = {
        "status": getattr(probe_result, "status", None),
        "fixture_sha256": getattr(probe_result, "observed_fixture_sha256", None),
        "pdfplumber_version": getattr(probe_result, "observed_pdfplumber_version", None),
        "page_count": getattr(probe_result, "observed_page_count", None),
    }
except Exception:
    if package_status == "PASS":
        package_status = "FAIL"
        package_failure = "ISOLATED_OBSERVER_EXCEPTION"
    if pdf_probe["status"] == "FAIL" and pdf_probe["fixture_sha256"] is None:
        pdf_probe = empty_pdf_probe()

identity_pass = (
    platform.python_implementation() == "CPython"
    and platform.python_version() == "3.12.10"
    and platform.system() == "Windows"
    and platform.machine() == "AMD64"
    and sysconfig.get_platform() == "win-amd64"
)
overall_status = "PASS" if identity_pass and package_status == "PASS" and pdf_probe["status"] == PDF_PASS else "FAIL"
emit(overall_status, packages, package_status, package_failure, pdf_probe)
'''


def _controlled_child_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in {"PYTHONPATH", "PYTHONHOME"}
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    return environment


def _validate_isolated_child_json(value: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, dict) or set(value) != ISOLATED_CHILD_TOP_KEYS:
        return None, "CURRENT_AUTHORITY_CHILD_JSON_SCHEMA_INVALID"
    if value.get("schema_version") != ISOLATED_CHILD_SCHEMA:
        return None, "CURRENT_AUTHORITY_CHILD_JSON_SCHEMA_INVALID"
    for key in ("isolated", "no_user_site", "pythonpath_env_absent", "bytecode_disabled"):
        if type(value.get(key)) is not bool:
            return None, "CURRENT_AUTHORITY_CHILD_JSON_TYPES_INVALID"
    for key in (
        "status",
        "python_implementation",
        "python_version",
        "executable",
        "platform_system",
        "platform_machine",
        "sysconfig_platform",
        "package_observation_status",
    ):
        if not isinstance(value.get(key), str):
            return None, "CURRENT_AUTHORITY_CHILD_JSON_TYPES_INVALID"
    if value.get("status") not in {"PASS", "FAIL"} or value.get("package_observation_status") not in {"PASS", "FAIL"}:
        return None, "CURRENT_AUTHORITY_CHILD_JSON_ENUM_INVALID"
    if value.get("package_failure") is not None and not isinstance(value.get("package_failure"), str):
        return None, "CURRENT_AUTHORITY_CHILD_JSON_TYPES_INVALID"
    packages = value.get("packages")
    if not isinstance(packages, list):
        return None, "CURRENT_AUTHORITY_CHILD_JSON_TYPES_INVALID"
    for package in packages:
        if not isinstance(package, dict) or set(package) != {"name", "version"}:
            return None, "CURRENT_AUTHORITY_CHILD_PACKAGE_SCHEMA_INVALID"
        if not isinstance(package["name"], str) or not package["name"] or not isinstance(package["version"], str) or not package["version"]:
            return None, "CURRENT_AUTHORITY_CHILD_PACKAGE_TYPES_INVALID"
    pdf_probe = value.get("pdf_probe")
    if not isinstance(pdf_probe, dict) or set(pdf_probe) != ISOLATED_CHILD_PDF_KEYS:
        return None, "CURRENT_AUTHORITY_CHILD_PDF_SCHEMA_INVALID"
    if not isinstance(pdf_probe["status"], str):
        return None, "CURRENT_AUTHORITY_CHILD_PDF_TYPES_INVALID"
    for key in ("fixture_sha256", "pdfplumber_version"):
        if pdf_probe[key] is not None and not isinstance(pdf_probe[key], str):
            return None, "CURRENT_AUTHORITY_CHILD_PDF_TYPES_INVALID"
    if pdf_probe["page_count"] is not None and (type(pdf_probe["page_count"]) is not int or isinstance(pdf_probe["page_count"], bool)):
        return None, "CURRENT_AUTHORITY_CHILD_PDF_TYPES_INVALID"
    return value, None


def _run_isolated_observer() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                str(CANONICAL_INTERPRETER),
                "-I",
                "-B",
                "-c",
                _isolated_child_script(),
                str(REPO_ROOT),
            ],
            capture_output=True,
            text=True,
            check=False,
            env=_controlled_child_environment(),
        )
    except OSError:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_CHILD_LAUNCH_FAILED"}
    if result.returncode != 0:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_CHILD_NONZERO_EXIT"}
    if result.stderr.strip():
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_CHILD_STDERR_NOT_EMPTY"}
    try:
        value = json.loads(result.stdout)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_CHILD_JSON_INVALID"}
    validated, reason = _validate_isolated_child_json(value)
    if reason is not None or validated is None:
        return {"status": "FAIL", "reason": reason}
    return {"status": "PASS", "evidence": validated}


def _validate_child_identity(evidence: dict[str, Any]) -> dict[str, Any]:
    try:
        executable_match = os.path.normcase(str(Path(evidence["executable"]).resolve(strict=False))) == os.path.normcase(
            str(CANONICAL_INTERPRETER.resolve(strict=False))
        )
    except OSError:
        executable_match = False
    passed = (
        evidence["status"] == "PASS"
        and evidence["isolated"] is True
        and evidence["no_user_site"] is True
        and evidence["pythonpath_env_absent"] is True
        and evidence["bytecode_disabled"] is True
        and evidence["python_implementation"] == "CPython"
        and evidence["python_version"] == "3.12.10"
        and evidence["platform_system"] == "Windows"
        and evidence["platform_machine"] == "AMD64"
        and evidence["sysconfig_platform"] == "win-amd64"
        and executable_match
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "reason": None if passed else "CURRENT_AUTHORITY_CHILD_IDENTITY_MISMATCH",
    }


def _package_map_from_records(records: list[dict[str, str]]) -> tuple[dict[str, str] | None, str | None]:
    actual: dict[str, str] = {}
    for record in records:
        normalized = _normalize_package_name(record["name"])
        if normalized in actual:
            return None, "CURRENT_AUTHORITY_INSTALLED_METADATA_DUPLICATE"
        actual[normalized] = record["version"]
    return actual, None


def _validate_child_package_set(expected_packages: dict[str, str], evidence: dict[str, Any]) -> dict[str, Any]:
    if evidence["package_observation_status"] != "PASS":
        return {
            "status": "FAIL",
            "reason": evidence["package_failure"] or "CURRENT_AUTHORITY_INSTALLED_METADATA_OBSERVATION_FAILED",
        }
    actual, reason = _package_map_from_records(evidence["packages"])
    if reason is not None or actual is None:
        return {"status": "FAIL", "reason": reason}
    if actual != expected_packages:
        return {
            "status": "FAIL",
            "reason": "CURRENT_AUTHORITY_PACKAGE_SET_MISMATCH",
            "missing_packages": sorted(set(expected_packages) - set(actual)),
            "extra_packages": sorted(set(actual) - set(expected_packages)),
            "version_mismatches": sorted(
                key for key in set(expected_packages) & set(actual) if expected_packages[key] != actual[key]
            ),
        }
    return {"status": "PASS", "package_count": len(actual)}


def _validate_child_pdf_probe(evidence: dict[str, Any]) -> dict[str, Any]:
    probe = evidence["pdf_probe"]
    if probe["status"] != PDF_PROBE_PASS:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_PDF_PROBE_FAILED"}
    if probe["fixture_sha256"] != PDF_PROBE_EXPECTED_FIXTURE_SHA256:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_PDF_PROBE_FIXTURE_IDENTITY_MISMATCH"}
    if probe["pdfplumber_version"] != PDF_PROBE_REQUIRED_PDFPLUMBER_VERSION:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_PDF_PROBE_VERSION_MISMATCH"}
    if probe["page_count"] != PDF_PROBE_EXPECTED_PAGE_COUNT:
        return {"status": "FAIL", "reason": "CURRENT_AUTHORITY_PDF_PROBE_PAGE_COUNT_MISMATCH"}
    return {
        "status": "PASS",
        "fixture_sha256": probe["fixture_sha256"],
        "pdfplumber_version": probe["pdfplumber_version"],
        "page_count": probe["page_count"],
    }


def run_current_readiness() -> dict[str, Any]:
    authority = resolve_current_authority()
    interpreter = check_interpreter_identity()
    isolated = (
        _run_isolated_observer()
        if authority["status"] == "PASS" and interpreter["status"] == "PASS"
        else {"status": "NOT_RUN", "reason": "CURRENT_AUTHORITY_PRECONDITION_FAILED"}
    )
    if isolated["status"] == "PASS":
        evidence = isolated["evidence"]
        child_identity = _validate_child_identity(evidence)
        packages = _validate_child_package_set(authority["package_map"], evidence)
        pdf_probe = _validate_child_pdf_probe(evidence)
    else:
        child_identity = {"status": "NOT_RUN", "reason": isolated["reason"]}
        packages = {"status": "NOT_RUN", "reason": isolated["reason"]}
        pdf_probe = {"status": "NOT_RUN", "reason": isolated["reason"]}
    ready = (
        authority["status"] == "PASS"
        and interpreter["status"] == "PASS"
        and isolated["status"] == "PASS"
        and child_identity["status"] == "PASS"
        and packages["status"] == "PASS"
        and pdf_probe["status"] == "PASS"
    )
    return {
        "CURRENT_PROTECTED_ENVIRONMENT_AUTHORITY": authority,
        "INTERPRETER": interpreter,
        "ISOLATED_RUNTIME": {"status": isolated["status"], "reason": isolated.get("reason")},
        "ISOLATED_CHILD_IDENTITY": child_identity,
        "LIVE_PACKAGE_SET": packages,
        "PDF_OPERATIONAL_PROBE": pdf_probe,
        "CURRENT_ENVIRONMENT_READY": ready,
        "REAL_NETWORK_REQUESTS": 0,
        "PRIVATE_READS": 0,
        "ENVIRONMENT_MUTATIONS": 0,
        "PACKAGE_INSTALLATIONS": 0,
        "GATES_CONSUMED": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve and validate the current protected environment")
    parser.parse_args()
    result = run_current_readiness()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["CURRENT_ENVIRONMENT_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
