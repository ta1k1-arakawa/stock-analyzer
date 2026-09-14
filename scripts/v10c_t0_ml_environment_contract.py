"""Pure validators for the frozen V10C canonical ML environment successor.

The contract module has no import-time I/O, subprocess execution, package
imports, installation behavior, or network behavior.  All observations are
provided by callers so synthetic tests can exercise the exact successor
contract without touching the canonical environment.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from email.parser import BytesParser
from email.policy import compat32
from pathlib import Path
from typing import Any, Mapping, Sequence


STUDY = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR"
FROZEN_DESIGN_SHA = "840094e09f89569b8e6345bd2e19f43298e9ddfe"
FROZEN_DESIGN_BLOB = "8efe1ef1d789ea22b5c070593610f158d2fad53e"
APPROVAL_RECORD_BLOB = "0df6ad1ae9000cafc09bab08c31195a51a346a07"
PREDECESSOR_LOCK_BLOB = "99395e7a5be752fb3ea92fd31be0334f38792261"
PREDECESSOR_LOCK_SHA256 = "eb325ac5e3417e6407400b18c8d90ca734a32e852056926e5bcd2a635e43c444"
PREDECESSOR_PACKAGE_SET = (
    ("cffi", "2.1.1"),
    ("charset-normalizer", "3.5.1"),
    ("cryptography", "50.0.1"),
    ("exchange-calendars", "4.13.2"),
    ("korean-lunar-calendar", "0.4.0"),
    ("numpy", "2.5.2"),
    ("pandas", "3.0.5"),
    ("pandas-market-calendars", "5.4.0"),
    ("pdfminer-six", "20260107"),
    ("pdfplumber", "0.11.10"),
    ("pillow", "12.3.0"),
    ("pip", "25.0.1"),
    ("pycparser", "3.0"),
    ("pyluach", "2.3.0"),
    ("pypdfium2", "5.13.0"),
    ("python-dateutil", "2.9.0.post0"),
    ("six", "1.17.0"),
    ("toolz", "1.1.0"),
    ("tzdata", "2026.3"),
    ("xlrd", "2.0.2"),
)
DIRECT_SPEC_BYTES = (
    b"pandas\n"
    b"xlrd==2.0.2\n"
    b"pdfplumber==0.11.10\n"
    b"pandas-market-calendars==5.4.0\n"
    b"lightgbm==4.6.0\n"
    b"scikit-learn==1.9.0\n"
)
DIRECT_SPEC_SHA256 = hashlib.sha256(DIRECT_SPEC_BYTES).hexdigest()
LIGHTGBM_PIN = "4.6.0"
SCIKIT_LEARN_PIN = "1.9.0"
RESOLUTION_POLICY_ID = "PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1"
PACKAGE_INDEX_ID = "PYPI_OFFICIAL_SIMPLE"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CANDIDATE_SCHEMA = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE_V1"
EVIDENCE_SCHEMA = "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE_V1"
CANDIDATE_KEYS = frozenset(
    {
        "schema_version", "artifact_status", "study", "frozen_design_git_sha",
        "frozen_design_git_blob_sha1", "approval_record_git_blob_sha1",
        "reviewed_resolution_implementation_git_sha", "direct_spec_git_blob_sha1",
        "direct_spec_sha256", "predecessor_lock_git_blob_sha1",
        "predecessor_lock_sha256", "predecessor_package_count", "python_version",
        "platform_system", "platform_machine", "sysconfig_platform",
        "resolution_policy_id", "resolved_packages", "resolved_package_count",
        "resolved_wheels", "predecessor_pin_drift_count", "lightgbm_version",
        "scikit_learn_version", "resolved_dependency_metadata",
    }
)
EVIDENCE_KEYS = frozenset(
    {
        "schema_version", "artifact_status", "status", "failure_code", "study",
        "frozen_design_git_sha", "frozen_design_git_blob_sha1", "approval_record_git_blob_sha1",
        "reviewed_resolution_implementation_git_sha", "direct_spec_git_blob_sha1",
        "direct_spec_sha256", "predecessor_lock_git_blob_sha1",
        "predecessor_lock_sha256", "resolution_policy_id", "package_index_id",
        "process_started", "process_exit_code", "resolution_completed",
        "candidate_artifact_created", "successor_lock_candidate_sha256",
        "resolved_package_count", "package_resolution_process_invocations",
        "human_authority_consumed", "package_installations",
        "alternate_venv_created", "t0_runs", "payload_reads",
    }
)
APPROVAL_KEYS = frozenset(
    {
        "schema_version", "study", "artifact_role", "frozen_design_git_commit",
        "frozen_design_git_blob_sha1", "design_document", "final_independent_review_result",
        "final_independent_review_design_commit", "final_independent_review_authority",
        "review_low_1", "review_low_1_resolved_by_this_artifact", "approval_status",
        "human_design_freeze_complete", "approval_scope",
        "approval_artifact_commit_is_not_the_frozen_design_commit",
        "approval_artifact_authorizes_implementation_phase_only", "predecessor_package_count",
        "predecessor_lock_git_blob_sha1", "predecessor_lock_sha256", "lightgbm_direct_pin",
        "scikit_learn_direct_pin", "transitive_versions_preselected", "package_resolution_authorized",
        "package_index_network_access_authorized", "wheel_acquisition_authorized",
        "canonical_environment_mutation_authorized", "package_installation_authorized",
        "training_payload_read_authorized", "evaluation_payload_read_authorized",
        "model_fit_authorized", "t0_authorized", "historical_evaluation_authorized",
        "private_sealed_access_authorized", "future_profitability_established",
        "methodology_change_after_freeze_requires", "next_required_action",
    }
)
FAILURE_CODES = frozenset(
    {
        "NONE", "UNAUTHORIZED_INSTALLATION", "UNAUTHORIZED_ALTERNATE_ENVIRONMENT",
        "RESOLUTION_PROCESS_FAILURE", "RESOLUTION_REPORT_INVALID",
        "PREDECESSOR_PIN_DRIFT", "REQUIRED_DIRECT_DISTRIBUTION_MISSING",
        "SOURCE_DISTRIBUTION_REQUIRED", "WHEEL_PROVENANCE_FAILURE",
    }
)
WHEEL_FIELDS = frozenset({"name", "version", "filename", "sha256"})
DEPENDENCY_METADATA_FIELDS = frozenset({"name", "version", "requires_dist", "requires_python"})
TARGET_ENVIRONMENT = {
    "python_version": "3.12",
    "python_full_version": "3.12.10",
    "platform_system": "Windows",
    "platform_machine": "AMD64",
    "platform_python_implementation": "CPython",
    "implementation_name": "cpython",
    "sys_platform": "win32",
    "os_name": "nt",
    "extra": "",
}


class ContractValidationError(ValueError):
    """Raised when a synthetic candidate/evidence object violates the contract."""


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"


def git_blob_sha1(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


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


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(set(value) == set(expected), f"{label} fieldset mismatch")


def _sha(value: Any, pattern: re.Pattern[str], label: str) -> str:
    _require(isinstance(value, str) and pattern.fullmatch(value) is not None, f"{label} invalid")
    return value


def _string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} invalid")
    return value


def _strict_int(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} invalid")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    _require(isinstance(value, bool), f"{label} invalid")
    return value


def _version_key(value: str) -> tuple[Any, ...]:
    match = re.fullmatch(
        r"(?i)(\d+(?:\.\d+)*)(?:(a|alpha|b|beta|rc|c)(\d*))?(?:(?:\.|-)?(post|rev|r)(\d*))?(?:(?:\.|-)?dev(\d*))?",
        value,
    )
    _require(match is not None, "unsupported version syntax")
    release_parts = [int(part) for part in match.group(1).split(".")]
    while len(release_parts) > 1 and release_parts[-1] == 0:
        release_parts.pop()
    release = tuple(release_parts)
    stage = {None: 3, "a": 1, "alpha": 1, "b": 2, "beta": 2, "rc": 2, "c": 2}[match.group(2)]
    stage_serial = int(match.group(3) or 0)
    post = int(match.group(5) or 0)
    dev = match.group(6)
    return (release, stage, stage_serial, post, -1 if dev is not None else 0, int(dev or 0))


def _version_satisfies(version: str, specifier: str) -> bool:
    actual = _version_key(version)
    for raw_part in specifier.split(","):
        part = raw_part.strip()
        _require(bool(part), "empty version specifier")
        match = re.fullmatch(r"(===|==|!=|<=|>=|~=|<|>)[ ]*(.+)", part)
        _require(match is not None, "unsupported version specifier")
        operator, expected_text = match.groups()
        if operator in {"==", "!="} and expected_text.endswith(".*"):
            prefix = expected_text[:-2].split(".")
            equal = tuple(str(item) for item in _version_key(version)[0][: len(prefix)]) == tuple(prefix)
            if (operator == "==") != equal:
                return False
            continue
        expected = _version_key(expected_text)
        if operator == "==" or operator == "===":
            result = actual == expected
        elif operator == "!=":
            result = actual != expected
        elif operator == ">=":
            result = actual >= expected
        elif operator == ">":
            result = actual > expected
        elif operator == "<=":
            result = actual <= expected
        elif operator == "<":
            result = actual < expected
        else:
            expected_release = expected[0]
            upper = (expected_release[:-1] + (expected_release[-1] + 1,)) if len(expected_release) > 1 else (expected_release[0] + 1,)
            result = actual >= expected and actual < (upper, 3, 0, 0, 0, 0)
        if not result:
            return False
    return True


_MARKER_TOKEN = re.compile(r"\s*(?:(and|or|not|in)|(!=|==|<=|>=|~=|<|>|\(|\))|([A-Za-z_][A-Za-z0-9_]*)|('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"))")


def _marker_value(token: tuple[str, str], environment: Mapping[str, str]) -> str:
    kind, value = token
    if kind == "identifier":
        _require(value in environment, "unsupported environment marker variable")
        return environment[value]
    if kind == "string":
        import ast

        parsed = ast.literal_eval(value)
        _require(isinstance(parsed, str), "invalid environment marker string")
        return parsed
    raise ContractValidationError("invalid environment marker operand")


class _MarkerParser:
    def __init__(self, raw: str, environment: Mapping[str, str]) -> None:
        self.environment = environment
        self.tokens: list[tuple[str, str]] = []
        position = 0
        while position < len(raw):
            match = _MARKER_TOKEN.match(raw, position)
            _require(match is not None, "invalid environment marker")
            if match.group(1):
                self.tokens.append(("word", match.group(1)))
            elif match.group(2):
                self.tokens.append(("operator", match.group(2)))
            elif match.group(3):
                self.tokens.append(("identifier", match.group(3)))
            else:
                self.tokens.append(("string", match.group(4)))
            position = match.end()
        self.index = 0

    def _peek(self, value: str | None = None) -> bool:
        if self.index >= len(self.tokens):
            return False
        return value is None or self.tokens[self.index][1] == value

    def _take(self, value: str | None = None) -> tuple[str, str]:
        _require(self.index < len(self.tokens), "unexpected end of environment marker")
        token = self.tokens[self.index]
        _require(value is None or token[1] == value, "invalid environment marker grammar")
        self.index += 1
        return token

    def parse(self) -> bool:
        result = self._parse_or()
        _require(self.index == len(self.tokens), "trailing environment marker")
        return result

    def _parse_or(self) -> bool:
        result = self._parse_and()
        while self._peek("or"):
            self._take("or")
            right = self._parse_and()
            result = result or right
        return result

    def _parse_and(self) -> bool:
        result = self._parse_not()
        while self._peek("and"):
            self._take("and")
            right = self._parse_not()
            result = result and right
        return result

    def _parse_not(self) -> bool:
        if self._peek("not"):
            self._take("not")
            return not self._parse_not()
        return self._parse_atom()

    def _parse_atom(self) -> bool:
        if self._peek("("):
            self._take("(")
            result = self._parse_or()
            self._take(")")
            return result
        left = _marker_value(self._take(), self.environment)
        if self._peek():
            operator = self._take()[1]
            if operator == "not":
                self._take("in")
                operator = "not in"
            elif operator == "in":
                pass
            else:
                _require(operator in {"!=", "==", "<=", ">=", "~=", "<", ">"}, "invalid marker operator")
            right = _marker_value(self._take(), self.environment)
            if operator == "in":
                return left in right
            if operator == "not in":
                return left not in right
            if operator in {"<", "<=", ">", ">=", "~="} and re.fullmatch(r"\d+(?:\.\d+)*", left) and re.fullmatch(r"\d+(?:\.\d+)*", right):
                left_key = _version_key(left)
                right_key = _version_key(right)
                if operator == "<":
                    return left_key < right_key
                if operator == "<=":
                    return left_key <= right_key
                if operator == ">":
                    return left_key > right_key
                if operator == ">=":
                    return left_key >= right_key
                if operator == "~=":
                    return _version_satisfies(left, "~=" + right)
            return {"==": left == right, "!=": left != right, "<": left < right, "<=": left <= right, ">": left > right, ">=": left >= right, "~=": left == right}[operator]
        return bool(left)


def _marker_applies(raw: str | None) -> bool:
    return True if raw is None else _MarkerParser(raw.strip(), TARGET_ENVIRONMENT).parse()


def _parse_requirement(raw: str) -> tuple[str, str, str | None, str | None]:
    _require(isinstance(raw, str) and bool(raw.strip()), "invalid dependency requirement")
    requirement, separator, marker = raw.partition(";")
    match = re.fullmatch(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([^]]+)\])?\s*(.*)\s*", requirement)
    _require(match is not None, "invalid dependency requirement")
    extras = match.group(2)
    specifier = match.group(3).strip()
    if specifier.startswith("(") and specifier.endswith(")"):
        specifier = specifier[1:-1].strip()
    _require(not separator or bool(marker.strip()), "invalid dependency marker")
    return normalize_distribution_name(match.group(1)), specifier, marker.strip() if separator else None, extras


def _validate_dependency_closure(packages: Sequence[Mapping[str, Any]], metadata: Any) -> None:
    package_pairs = tuple((item["name"], item["version"]) for item in packages)
    _require(isinstance(metadata, list), "RESOLUTION_REPORT_INVALID")
    parsed: list[tuple[str, str, tuple[str, ...], str | None]] = []
    for item in metadata:
        _exact_keys(item, DEPENDENCY_METADATA_FIELDS, "dependency metadata")
        name = _string(item["name"], "dependency name")
        version = _string(item["version"], "dependency version")
        _require(name == normalize_distribution_name(name), "dependency name is not normalized")
        requirements = item["requires_dist"]
        _require(isinstance(requirements, list) and all(isinstance(value, str) and bool(value) for value in requirements), "dependency requirements invalid")
        _require(tuple(requirements) == tuple(sorted(requirements)), "dependency requirements are not sorted")
        requires_python = item["requires_python"]
        _require(requires_python is None or (isinstance(requires_python, str) and bool(requires_python)), "Requires-Python invalid")
        parsed.append((name, version, tuple(requirements), requires_python))
    _require(tuple((name, version) for name, version, _, _ in parsed) == package_pairs, "dependency metadata/package mismatch")
    package_map = dict(package_pairs)
    metadata_map = {(name, version): (requirements, requires_python) for name, version, requirements, requires_python in parsed}
    roots = {name for name, _ in PREDECESSOR_PACKAGE_SET} | {"lightgbm", "scikit-learn"}
    reachable: set[str] = set()
    pending = sorted(roots)
    while pending:
        name = pending.pop(0)
        if name in reachable:
            continue
        _require(name in package_map, "RESOLUTION_REPORT_INVALID")
        reachable.add(name)
        requirements, requires_python = metadata_map[(name, package_map[name])]
        if requires_python is not None:
            _require(_version_satisfies("3.12.10", requires_python), "RESOLUTION_REPORT_INVALID")
        for raw in requirements:
            dependency, specifier, marker, extras = _parse_requirement(raw)
            if not _marker_applies(marker):
                continue
            _require(extras is None, "dependency extras are not closed")
            _require(dependency in package_map, "RESOLUTION_REPORT_INVALID")
            if specifier:
                _require(_version_satisfies(package_map[dependency], specifier), "RESOLUTION_REPORT_INVALID")
            if dependency not in reachable:
                pending.append(dependency)
                pending.sort()
    _require(reachable == set(package_map), "RESOLUTION_REPORT_INVALID")


def validate_direct_spec_bytes(raw: bytes) -> str:
    _require(raw == DIRECT_SPEC_BYTES, "DIRECT_SPEC_BYTES_MISMATCH")
    return hashlib.sha256(raw).hexdigest()


def validate_approval_record(record: Mapping[str, Any]) -> None:
    """Validate the exact reviewed design-freeze approval semantics."""

    _exact_keys(record, APPROVAL_KEYS, "design-freeze approval")
    expected_strings = {
        "schema_version": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DESIGN_FREEZE_APPROVAL_V1",
        "study": STUDY,
        "artifact_role": "DESIGN_FREEZE_APPROVAL",
        "frozen_design_git_commit": FROZEN_DESIGN_SHA,
        "frozen_design_git_blob_sha1": FROZEN_DESIGN_BLOB,
        "design_document": "V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DESIGN_DRAFT.md",
        "final_independent_review_result": "PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_1",
        "final_independent_review_design_commit": FROZEN_DESIGN_SHA,
        "final_independent_review_authority": "GPT-5.6 Sol",
        "review_low_1": "DESIGN_FREEZE_APPROVAL_RECORD_SCHEMA_NOT_YET_EXACTLY_ENUMERATED_SAFE_DEFERRED",
        "approval_status": "APPROVED",
        "approval_scope": "DESIGN_FREEZE_ONLY",
        "methodology_change_after_freeze_requires": "NEW_STUDY_REQUIRED",
        "next_required_action": "GPT_EXACT_SHA_V10C_T0_CANONICAL_ML_ENVIRONMENT_SUCCESSOR_DESIGN_FREEZE_APPROVAL_REVIEW",
    }
    for key, expected in expected_strings.items():
        _require(record[key] == expected, f"approval {key}")
    _require(record["predecessor_package_count"] == len(PREDECESSOR_PACKAGE_SET), "approval predecessor count")
    _require(record["predecessor_lock_git_blob_sha1"] == PREDECESSOR_LOCK_BLOB, "approval predecessor blob")
    _require(record["predecessor_lock_sha256"] == PREDECESSOR_LOCK_SHA256, "approval predecessor SHA")
    _require(record["lightgbm_direct_pin"] == LIGHTGBM_PIN, "approval LightGBM pin")
    _require(record["scikit_learn_direct_pin"] == SCIKIT_LEARN_PIN, "approval scikit-learn pin")
    _require(record["transitive_versions_preselected"] is False, "approval transitive selection")
    _require(record["review_low_1_resolved_by_this_artifact"] is True, "approval low finding")
    _require(record["human_design_freeze_complete"] is True, "approval human freeze")
    _require(record["approval_artifact_commit_is_not_the_frozen_design_commit"] is True, "approval commit distinction")
    _require(record["approval_artifact_authorizes_implementation_phase_only"] is True, "approval implementation scope")
    for key in (
        "package_resolution_authorized", "package_index_network_access_authorized", "wheel_acquisition_authorized",
        "canonical_environment_mutation_authorized", "package_installation_authorized", "training_payload_read_authorized",
        "evaluation_payload_read_authorized", "model_fit_authorized", "t0_authorized",
        "historical_evaluation_authorized", "private_sealed_access_authorized", "future_profitability_established",
    ):
        _require(record[key] is False, f"approval authority {key}")


def validate_predecessor_packages(packages: Any) -> tuple[tuple[str, str], ...]:
    _require(isinstance(packages, list), "predecessor package set must be an array")
    parsed: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in packages:
        _exact_keys(item, frozenset({"name", "version"}), "predecessor package")
        name = _string(item["name"], "predecessor name")
        version = _string(item["version"], "predecessor version")
        normalized = normalize_distribution_name(name)
        _require(name == normalized, "predecessor name is not normalized")
        _require(normalized not in names, "duplicate predecessor distribution")
        names.add(normalized)
        parsed.append((normalized, version))
    _require(tuple(parsed) == PREDECESSOR_PACKAGE_SET, "PREDECESSOR_PIN_DRIFT")
    return tuple(parsed)


def validate_resolved_packages(packages: Any) -> dict[str, tuple[tuple[str, str], ...]]:
    _require(isinstance(packages, list), "resolved packages must be an array")
    parsed: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in packages:
        _exact_keys(item, frozenset({"name", "version"}), "resolved package")
        name = _string(item["name"], "resolved name")
        version = _string(item["version"], "resolved version")
        normalized = normalize_distribution_name(name)
        _require(name == normalized, "resolved name is not normalized")
        _require(normalized not in names, "duplicate normalized distribution")
        names.add(normalized)
        parsed.append((normalized, version))
    _require(tuple(parsed) == tuple(sorted(parsed)), "resolved packages are not sorted")
    package_map = dict(parsed)
    drift = [name for name, version in PREDECESSOR_PACKAGE_SET if package_map.get(name) != version]
    _require(not drift, "PREDECESSOR_PIN_DRIFT")
    _require(package_map.get("lightgbm") == LIGHTGBM_PIN, "REQUIRED_DIRECT_DISTRIBUTION_MISSING")
    _require(package_map.get("scikit-learn") == SCIKIT_LEARN_PIN, "REQUIRED_DIRECT_DISTRIBUTION_MISSING")
    delta = tuple(pair for pair in parsed if pair[0] not in {name for name, _ in PREDECESSOR_PACKAGE_SET})
    return {"predecessor": PREDECESSOR_PACKAGE_SET, "successor": tuple(parsed), "delta": delta}


def _parse_wheel_filename(filename: str) -> tuple[str, str]:
    _require("/" not in filename and "\\" not in filename, "wheel filename must be a basename")
    _require(filename.lower().endswith(".whl"), "wheel filename must end in .whl")
    parts = filename[:-4].split("-")
    _require(len(parts) >= 5, "malformed wheel filename")
    for version_index in range(1, len(parts) - 3):
        remaining = len(parts) - version_index
        version = parts[version_index]
        if remaining not in (4, 5) or not re.match(r"^[0-9]", version):
            continue
        name = normalize_distribution_name("-".join(parts[:version_index]).replace("_", "-"))
        _require(bool(name and version), "malformed wheel identity")
        return name, version
    raise ContractValidationError("malformed wheel identity")


def validate_wheel_manifest(wheels: Any) -> tuple[dict[str, str], ...]:
    _require(isinstance(wheels, list), "resolved wheels must be an array")
    result: list[dict[str, str]] = []
    names: set[str] = set()
    filenames: set[str] = set()
    for item in wheels:
        _exact_keys(item, WHEEL_FIELDS, "resolved wheel")
        name = _string(item["name"], "wheel name")
        version = _string(item["version"], "wheel version")
        filename = _string(item["filename"], "wheel filename")
        digest = _sha(item["sha256"], SHA256_RE, "wheel sha256")
        normalized = normalize_distribution_name(name)
        _require(name == normalized, "wheel name is not normalized")
        _require(normalized not in names, "duplicate wheel distribution")
        _require(filename.casefold() not in filenames, "duplicate wheel filename")
        parsed_name, parsed_version = _parse_wheel_filename(filename)
        _require(parsed_name == normalized and parsed_version == version, "wheel filename identity mismatch")
        names.add(normalized)
        filenames.add(filename.casefold())
        result.append({"name": normalized, "version": version, "filename": filename, "sha256": digest})
    _require(tuple((item["name"], item["version"]) for item in result) == tuple(sorted((item["name"], item["version"]) for item in result)), "resolved wheels are not sorted")
    return tuple(result)


def _validate_common_bindings(value: Mapping[str, Any], expected_reviewed_sha: str, expected_direct_blob: str, expected_direct_sha: str) -> None:
    _require(value["study"] == STUDY, "study binding")
    _require(value["frozen_design_git_sha"] == FROZEN_DESIGN_SHA, "design commit binding")
    _require(value["frozen_design_git_blob_sha1"] == FROZEN_DESIGN_BLOB, "design blob binding")
    _require(value["approval_record_git_blob_sha1"] == APPROVAL_RECORD_BLOB, "approval blob binding")
    _require(value["reviewed_resolution_implementation_git_sha"] == expected_reviewed_sha, "implementation binding")
    _require(value["direct_spec_git_blob_sha1"] == expected_direct_blob, "direct spec blob binding")
    _require(value["direct_spec_sha256"] == expected_direct_sha, "direct spec SHA binding")
    _require(value["predecessor_lock_git_blob_sha1"] == PREDECESSOR_LOCK_BLOB, "predecessor lock blob binding")
    _require(value["predecessor_lock_sha256"] == PREDECESSOR_LOCK_SHA256, "predecessor lock SHA binding")
    for key, pattern in (
        ("frozen_design_git_sha", SHA1_RE), ("frozen_design_git_blob_sha1", SHA1_RE),
        ("approval_record_git_blob_sha1", SHA1_RE), ("reviewed_resolution_implementation_git_sha", SHA1_RE),
        ("direct_spec_git_blob_sha1", SHA1_RE), ("direct_spec_sha256", SHA256_RE),
        ("predecessor_lock_git_blob_sha1", SHA1_RE), ("predecessor_lock_sha256", SHA256_RE),
    ):
        _sha(value[key], pattern, key)


def validate_lock_candidate(candidate: Mapping[str, Any], *, expected_reviewed_sha: str, expected_direct_blob: str, expected_direct_sha: str) -> dict[str, tuple[tuple[str, str], ...]]:
    _exact_keys(candidate, CANDIDATE_KEYS, "lock candidate")
    _require(candidate["schema_version"] == CANDIDATE_SCHEMA, "candidate schema")
    _require(candidate["artifact_status"] == "WINDOWS_RESOLUTION_CANDIDATE_NOT_INSTALL_AUTHORITY", "candidate status")
    _validate_common_bindings(candidate, expected_reviewed_sha, expected_direct_blob, expected_direct_sha)
    _require(candidate["predecessor_package_count"] == len(PREDECESSOR_PACKAGE_SET), "predecessor package count")
    _require(candidate["python_version"] == "3.12.10", "python version")
    _require(candidate["platform_system"] == "Windows", "platform system")
    _require(candidate["platform_machine"] == "AMD64", "platform machine")
    _require(candidate["sysconfig_platform"] == "win-amd64", "sysconfig platform")
    _require(candidate["resolution_policy_id"] == RESOLUTION_POLICY_ID, "resolution policy")
    packages = validate_resolved_packages(candidate["resolved_packages"])
    _require(candidate["resolved_package_count"] == len(packages["successor"]), "resolved package count")
    _require(candidate["resolved_package_count"] > len(PREDECESSOR_PACKAGE_SET), "successor package count")
    _require(candidate["predecessor_pin_drift_count"] == 0, "predecessor pin drift count")
    package_map = dict(packages["successor"])
    _require(candidate["lightgbm_version"] == LIGHTGBM_PIN == package_map["lightgbm"], "lightgbm version")
    _require(candidate["scikit_learn_version"] == SCIKIT_LEARN_PIN == package_map["scikit-learn"], "scikit-learn version")
    wheels = validate_wheel_manifest(candidate["resolved_wheels"])
    _require(tuple((item["name"], item["version"]) for item in wheels) == packages["successor"], "wheel/package identity")
    _validate_dependency_closure(candidate["resolved_packages"], candidate["resolved_dependency_metadata"])
    return packages


def validate_evidence(evidence: Mapping[str, Any], *, expected_reviewed_sha: str, expected_direct_blob: str, expected_direct_sha: str, expected_candidate_sha: str | None = None) -> None:
    _exact_keys(evidence, EVIDENCE_KEYS, "resolution evidence")
    _require(evidence["schema_version"] == EVIDENCE_SCHEMA, "evidence schema")
    _require(evidence["artifact_status"] == "WINDOWS_RESOLUTION_EVIDENCE", "evidence status")
    _require(evidence["study"] == STUDY, "evidence study")
    _validate_common_bindings(evidence, expected_reviewed_sha, expected_direct_blob, expected_direct_sha)
    _require(evidence["resolution_policy_id"] == RESOLUTION_POLICY_ID, "evidence policy")
    _require(evidence["package_index_id"] == PACKAGE_INDEX_ID, "evidence index")
    _require(evidence["status"] in {"PASS", "FAIL"}, "evidence result")
    _require(evidence["failure_code"] in FAILURE_CODES, "evidence failure code")
    started = _strict_bool(evidence["process_started"], "process_started")
    exit_code = evidence["process_exit_code"]
    if started:
        _strict_int(exit_code, "process exit code")
    else:
        _require(exit_code is None, "unstarted process exit code")
    for key in ("resolution_completed", "candidate_artifact_created", "human_authority_consumed", "alternate_venv_created"):
        _strict_bool(evidence[key], key)
    installation_count = _strict_int(evidence["package_installations"], "package installations")
    _require(installation_count >= 0, "package installations")
    _require(_strict_int(evidence["t0_runs"], "t0 runs") == 0, "t0 runs")
    _require(_strict_int(evidence["payload_reads"], "payload reads") == 0, "payload reads")
    invocations = _strict_int(evidence["package_resolution_process_invocations"], "process invocations")
    _require(invocations in (0, 1), "process invocation count")
    candidate_sha = evidence["successor_lock_candidate_sha256"]
    if candidate_sha is not None:
        _sha(candidate_sha, SHA256_RE, "candidate SHA")
    package_count = evidence["resolved_package_count"]
    if package_count is not None:
        _strict_int(package_count, "resolved package count")
    if evidence["status"] == "PASS":
        _require(installation_count == 0 and evidence["alternate_venv_created"] is False, "PASS unauthorized activity")
        _require(started and exit_code == 0 and invocations == 1, "PASS process semantics")
        _require(evidence["failure_code"] == "NONE", "PASS failure code")
        _require(evidence["human_authority_consumed"] and evidence["resolution_completed"] and evidence["candidate_artifact_created"], "PASS completion")
        _require(candidate_sha is not None and expected_candidate_sha is not None and candidate_sha == expected_candidate_sha, "PASS candidate binding")
    elif evidence["failure_code"] == "UNAUTHORIZED_INSTALLATION":
        _require(installation_count > 0, "installation failure precedence")
        _require(evidence["human_authority_consumed"] and not evidence["candidate_artifact_created"], "installation failure completion")
        _require(candidate_sha is None and package_count is None, "installation failure candidate fields")
        _require(started and exit_code == 0 and invocations == 1 and evidence["resolution_completed"] is True, "installation failure process semantics")
    elif evidence["failure_code"] == "UNAUTHORIZED_ALTERNATE_ENVIRONMENT":
        _require(installation_count == 0 and evidence["alternate_venv_created"] is True, "alternate environment failure precedence")
        _require(evidence["human_authority_consumed"] and not evidence["candidate_artifact_created"], "alternate environment failure completion")
        _require(candidate_sha is None and package_count is None, "alternate environment failure candidate fields")
        _require(started and exit_code == 0 and invocations == 1 and evidence["resolution_completed"] is True, "alternate environment failure process semantics")
    elif evidence["failure_code"] == "RESOLUTION_PROCESS_FAILURE":
        _require(evidence["human_authority_consumed"] and not evidence["candidate_artifact_created"], "process failure completion")
        _require(candidate_sha is None and package_count is None, "process failure candidate fields")
        if started:
            _require(exit_code != 0 and invocations == 1 and evidence["resolution_completed"] is False, "started process failure semantics")
        else:
            _require(exit_code is None and invocations == 0 and evidence["resolution_completed"] is False, "launch failure semantics")
    else:
        _require(installation_count == 0 and evidence["alternate_venv_created"] is False, "offline inspection unauthorized activity")
        _require(evidence["failure_code"] != "NONE", "failed evidence failure code")
        _require(evidence["human_authority_consumed"] and not evidence["candidate_artifact_created"], "failed evidence completion")
        _require(started and exit_code == 0 and invocations == 1 and evidence["resolution_completed"] is True, "offline inspection failure semantics")
        _require(candidate_sha is None and package_count is None, "offline inspection candidate fields")


def _inspect_wheel_file_metadata(wheel_path: str | Path) -> dict[str, Any]:
    path = Path(wheel_path)
    parsed_name, parsed_version = _parse_wheel_filename(path.name)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        with zipfile.ZipFile(path) as archive:
            _require(archive.testzip() is None, "wheel ZIP integrity")
            metadata_members = [item for item in archive.infolist() if item.filename.endswith(".dist-info/METADATA")]
            wheel_members = [item for item in archive.infolist() if item.filename.endswith(".dist-info/WHEEL")]
            _require(len(metadata_members) == 1 and len(wheel_members) == 1, "wheel metadata cardinality")
            metadata = BytesParser(policy=compat32).parsebytes(archive.read(metadata_members[0]))
            wheel_metadata = BytesParser(policy=compat32).parsebytes(archive.read(wheel_members[0]))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise ContractValidationError("WHEEL_PROVENANCE_FAILURE") from error
    names = metadata.get_all("Name") or []
    versions = metadata.get_all("Version") or []
    _require(len(names) == 1 and len(versions) == 1, "wheel Name/Version fields")
    _require(len(wheel_metadata.get_all("Wheel-Version") or []) == 1, "wheel metadata")
    metadata_name = normalize_distribution_name(str(names[0]))
    metadata_version = str(versions[0])
    _require(metadata_name == parsed_name and metadata_version == parsed_version, "wheel identity mismatch")
    requires_python_values = metadata.get_all("Requires-Python") or []
    _require(len(requires_python_values) <= 1, "duplicate Requires-Python")
    requirements = tuple(sorted(str(value).strip() for value in (metadata.get_all("Requires-Dist") or [])))
    _require(all(requirements), "empty dependency requirement")
    return {
        "name": parsed_name,
        "version": parsed_version,
        "filename": path.name,
        "sha256": digest,
        "requires_dist": requirements,
        "requires_python": str(requires_python_values[0]).strip() if requires_python_values else None,
    }


def inspect_wheel_file(wheel_path: str | Path) -> dict[str, str]:
    inspected = _inspect_wheel_file_metadata(wheel_path)
    return {key: inspected[key] for key in WHEEL_FIELDS}


def inspect_wheelhouse(wheelhouse: str | Path) -> tuple[str, tuple[dict[str, Any], ...] | None]:
    root = Path(wheelhouse)
    try:
        entries = list(root.iterdir())
    except OSError:
        return "RESOLUTION_REPORT_INVALID", None
    if not entries:
        return "PREDECESSOR_PIN_DRIFT", None
    if any(not entry.is_file() for entry in entries):
        return "RESOLUTION_REPORT_INVALID", None
    source_present = any(not entry.name.lower().endswith(".whl") for entry in entries)
    try:
        wheels = tuple(sorted((_inspect_wheel_file_metadata(entry) for entry in entries if entry.name.lower().endswith(".whl")), key=lambda item: (item["name"], item["version"])))
        public_wheels = [{key: item[key] for key in WHEEL_FIELDS} for item in wheels]
        validate_wheel_manifest(public_wheels)
    except (ContractValidationError, OSError, ValueError, KeyError):
        return "RESOLUTION_REPORT_INVALID", None
    package_map = {item["name"]: item["version"] for item in wheels}
    if any(package_map.get(name) != version for name, version in PREDECESSOR_PACKAGE_SET):
        return "PREDECESSOR_PIN_DRIFT", wheels
    if package_map.get("lightgbm") != LIGHTGBM_PIN or package_map.get("scikit-learn") != SCIKIT_LEARN_PIN:
        return "REQUIRED_DIRECT_DISTRIBUTION_MISSING", wheels
    try:
        _validate_dependency_closure(
            [{"name": item["name"], "version": item["version"]} for item in wheels],
            [{"name": item["name"], "version": item["version"], "requires_dist": list(item["requires_dist"]), "requires_python": item["requires_python"]} for item in wheels],
        )
    except ContractValidationError:
        return "RESOLUTION_REPORT_INVALID", None
    if source_present:
        return "SOURCE_DISTRIBUTION_REQUIRED", wheels
    return "NONE", wheels


def verify_wheelhouse(wheelhouse: str | Path, expected_wheels: Sequence[Mapping[str, Any]]) -> tuple[dict[str, str], ...]:
    """Verify a wheelhouse has exactly the independently supplied manifest."""

    expected = validate_wheel_manifest([dict(item) for item in expected_wheels])
    root = Path(wheelhouse)
    try:
        entries = list(root.iterdir())
    except OSError as error:
        raise ContractValidationError("WHEEL_PROVENANCE_FAILURE") from error
    _require(all(entry.is_file() for entry in entries), "wheelhouse contains non-file")
    _require(all(entry.name.lower().endswith(".whl") for entry in entries), "SOURCE_DISTRIBUTION_REQUIRED")
    actual = tuple(sorted((inspect_wheel_file(entry) for entry in entries), key=lambda item: (item["name"], item["version"])))
    _require(actual == expected, "wheelhouse file set or hash mismatch")
    return actual


def derive_package_sets(packages: Sequence[Mapping[str, Any]]) -> dict[str, tuple[tuple[str, str], ...]]:
    return validate_resolved_packages(list(packages))
