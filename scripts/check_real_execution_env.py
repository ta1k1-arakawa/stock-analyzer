"""No-network, no-private-data readiness checker for the canonical
V8-lineage real-execution Python environment.

See `REAL_EXECUTION_PYTHON_ENVIRONMENT.md` for the human-readable contract
this script mechanically enforces, and `AI_REAL_EXECUTION_RUNBOOK.md` §15-19
for where this fits in the overall pre-authorization ordering. This script
is environment-readiness only: it never consumes a human research gate,
never calls JPX/Yahoo, never accesses private/sealed data, and never
executes any V8I/V8J real acquisition. It never opens a network socket and
never reads, writes, resets, or deletes any real gate receipt or evidence
artifact. The filesystem probe imports and calls the real production
durable/exclusive publication primitive itself
(`src.v8i_source_snapshot._atomic_publish_once`) -- not a reimplementation
of it -- but only ever on a freshly created, disposable temporary path that
is mechanically proven, via the real gate/private state root path
*constants* also imported from `src.v8i_source_snapshot`, never to overlap
real durable state.

This script is safe to run on any platform for static/structural
validation, but it can only ever report a Windows-grounded
`REAL_EXECUTION_ENVIRONMENT_READY=true` when actually run on Windows, inside
the canonical `.venv-real-execution`, via
`.venv-real-execution\\Scripts\\python.exe`. When run anywhere else
(including this repository's own Claude Code Cloud / Linux sessions),
`platform_windows_grounded` is always `false` and
`REAL_EXECUTION_ENVIRONMENT_READY` is always `false`, regardless of what
every other individual check reports -- this script must never claim
Windows-grounded readiness from a non-Windows run.

This repository also has an existing, separate `.venv` used for ordinary
project development and the unrelated daily trading bot -- it mixes in
dependencies (`yfinance`, `lightgbm`, `pytest`, `requests`, `curl_cffi`,
`scikit-learn`, and more) that have nothing to do with, and were never
reviewed for, protected real execution.

```text
.venv                 = GENERAL_PROJECT_ENVIRONMENT_NOT_AUTHORIZED_FOR_PROTECTED_EXECUTION
.venv-real-execution   = CANONICAL_PROTECTED_REAL_EXECUTION_ENVIRONMENT
```

`.venv` is never accepted for protected execution, even when it happens to
be Python 3.12 with pandas/xlrd installed and every other probe would
otherwise pass: `check_interpreter_identity` rejects any interpreter that
is not the exact resolved `.venv-real-execution\Scripts\python.exe` path,
and reports `interpreter_failure_class="PRE_GATE_WRONG_PYTHON_ENVIRONMENT"`
whenever a Windows run resolves to a different interpreter -- including,
explicitly, the general `.venv`.

The JPX ".xls" operational parser probe runs against the committed, wholly
synthetic fixture `tests/fixtures/synthetic_jpx_source_snapshot.xls` and
drives the real production parsing functions end to end, so it can now
report a genuine `PASS` (resolving the former
`CHATGPT_DECISION_REQUIRED: REAL_EXECUTION_XLS_SYNTHETIC_FIXTURE_STRATEGY`).
A `PASS` there still only reports what actually ran -- it is never
fabricated, and it does not by itself make the overall environment ready.

`check_environment_lock` (REAL_EXECUTION_ENVIRONMENT_LOCK_ENFORCEMENT) is a
mechanical environment-lock check that `REAL_EXECUTION_ENVIRONMENT_READY`
requires to PASS. It binds to hardcoded REVIEWED_* constants (the reviewed
`REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json` candidate, the reviewed
`requirements-real-execution.lock.txt` lock hash, and the reviewed source
requirements' canonical Git object SHA-256) rather than merely trusting
whatever those mutable files currently say, then further requires the live
interpreter, the live platform (exact CPython 3.12.10 / Windows / AMD64 /
win-amd64), and the live `python -m pip freeze --all` package set to match
exactly -- no extra package, no missing package, no version drift, and
every non-empty freeze line must itself be a valid exact `name==version`
pin (a direct-URL, editable/VCS, malformed, or duplicate-named entry is
never silently skipped -- it is a hard FAIL of the exact-set check).
Source-requirements provenance is established from canonical Git object
bytes (`git cat-file blob <sha>:<path>`), never from a checked-out
working-tree copy, so Windows CRLF line-ending conversion can never
silently pass or fail this check. `REAL_EXECUTION_ENVIRONMENT_FROZEN`
remains hardcoded `false` regardless of this check's result -- promoting
the environment to frozen is a separate, later, explicitly reviewed task,
not something this checker declares on its own.

Exit code is 0 only if `REAL_EXECUTION_ENVIRONMENT_READY` is true; nonzero
otherwise, always before any protected boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import ssl
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# CANONICAL_PROTECTED_REAL_EXECUTION_ENVIRONMENT -- the only environment
# ever accepted for protected real/network/private/human-gated execution.
CANONICAL_VENV_DIR = REPO_ROOT / ".venv-real-execution"
CANONICAL_WINDOWS_INTERPRETER = CANONICAL_VENV_DIR / "Scripts" / "python.exe"

# GENERAL_PROJECT_ENVIRONMENT_NOT_AUTHORIZED_FOR_PROTECTED_EXECUTION -- the
# repository's existing, separate general-development/trading-bot
# environment. Never deleted, modified, or reinterpreted by this script;
# referenced here ONLY so the interpreter-identity check can explicitly
# detect and reject it (see `check_interpreter_identity`).
GENERAL_PROJECT_VENV_DIR = REPO_ROOT / ".venv"

REQUIREMENTS_REAL_EXECUTION_FILE = REPO_ROOT / "requirements-real-execution.txt"
CANONICAL_PYTHON_MAJOR_MINOR = (3, 12)  # .github/workflows/daily_ai_trade.yml: python-version: '3.12'
SYNTHETIC_XLS_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "synthetic_jpx_source_snapshot.xls"

_REQUIRED_DIRECT_DEPENDENCIES = ("pandas", "xlrd")

# ---------------------------------------------------------------------------
# Environment-lock check (REAL_EXECUTION_ENVIRONMENT_LOCK_ENFORCEMENT):
# reviewed binding, hardcoded here -- not merely trusted from the mutable
# REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json / requirements-real-
# execution.lock.txt files on disk -- so a tampered or stale candidate/lock
# is independently, mechanically detectable. Mirrors the same
# hardcoded-reviewed-constant pattern already used by
# src/v8i_source_snapshot.py's REVIEWED_V8I_DESIGN_CANDIDATE_COMMIT.
# ---------------------------------------------------------------------------

LOCK_CANDIDATE_PATH = REPO_ROOT / "REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json"
LOCK_FILE_PATH = REPO_ROOT / "requirements-real-execution.lock.txt"

REVIEWED_LOCK_CANDIDATE_GIT_SHA = "107430894723c2bdc2f8493cb12c467fccd8665e"
REVIEWED_SOURCE_GIT_SHA = "b74e0f787599475cd9fe719d254202dc9bfc14d5"
REVIEWED_LOCK_SHA256 = "b5c063a1cca585fa100fdc0027d6cdbf4ef33ef5a7fe614230599fb882b51f96"
REVIEWED_SOURCE_REQUIREMENTS_GIT_SHA256 = "2cdcfd7a87023c4e9c3ec463cf16f77d88f72ccc8d1f0e5de242e6c68b0cf601"
REVIEWED_FIXTURE_SHA256 = "ca47744896a286e1c56d4d0c09260775772c7df0c01b80d81b7e9a515e6d6aa7"
REVIEWED_PACKAGE_COUNT = 7
REVIEWED_ARTIFACT_STATUS = "CANDIDATE_NOT_FROZEN"

# ---------------------------------------------------------------------------
# V9_014 successor promotion preparation (E8): these identities bind the
# reviewed E7 candidate/evidence closure.  They do not assert that the live
# canonical environment has changed; live validation is permitted only in
# SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED at later E10.
# ---------------------------------------------------------------------------
V9_014_E7_LOCK_CANDIDATE_PATH = REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json"
V9_014_E7_WINDOWS_EVIDENCE_PATH = REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_WINDOWS_VALIDATION_EVIDENCE.json"
V9_014_E7_LOCK_CANDIDATE_SHA256 = "b7c30ccded8009a6df122fd51889e5ac2deb3a8db9b1d49d7dccd528a87c633e"
V9_014_E7_WINDOWS_EVIDENCE_SHA256 = "6d201a5a33e696e92fb76341fbe1b91b578f136d81ba26d7e0056a0015f0fe86"
V9_014_PREDECESSOR_CANONICAL_FROZEN = "PREDECESSOR_CANONICAL_FROZEN"
V9_014_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED = "SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED"
V9_014_SUCCESSOR_CANONICAL_FROZEN = "SUCCESSOR_CANONICAL_FROZEN"
V9_014_PROMOTION_STATES = frozenset(
    {
        V9_014_PREDECESSOR_CANONICAL_FROZEN,
        V9_014_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED,
        V9_014_SUCCESSOR_CANONICAL_FROZEN,
    }
)

# Complete semantic content of the reviewed candidate at
# REVIEWED_LOCK_CANDIDATE_GIT_SHA.  This is deliberately an exhaustive,
# nested object rather than a partial selection of fields: after the exact
# schema check below, semantic equality binds every reviewed field/value and
# rejects any otherwise-schema-valid mutation.  It is a Python semantic
# object, not a hash of working-tree bytes, so CRLF conversion is irrelevant.
REVIEWED_LOCK_CANDIDATE_SEMANTIC_CONTENT: dict[str, Any] = {
    "artifact_status": "CANDIDATE_NOT_FROZEN",
    "canonical_environment_directory": ".venv-real-execution",
    "canonical_interpreter": ".venv-real-execution\\Scripts\\python.exe",
    "future_protected_execution_authorized": False,
    "gpt_exact_sha_independent_review_required": True,
    "pip_version": "25.0.1",
    "private_or_sealed_reads": 0,
    "python": {
        "implementation": "CPython",
        "os_name": "nt",
        "platform_machine": "AMD64",
        "platform_release": "11",
        "platform_system": "Windows",
        "sysconfig_platform": "win-amd64",
        "version": "3.12.10",
    },
    "real_execution_environment_frozen": False,
    "real_execution_environment_ready_observed": True,
    "real_network_requests_to_protected_hosts": 0,
    "requirements_real_execution": {
        "path": "requirements-real-execution.txt",
        "sha256": "2cdcfd7a87023c4e9c3ec463cf16f77d88f72ccc8d1f0e5de242e6c68b0cf601",
    },
    "research_gates_consumed": 0,
    "resolved_lock": {
        "generated_from": ".venv-real-execution\\Scripts\\python.exe -m pip freeze --all",
        "package_count": 7,
        "path": "requirements-real-execution.lock.txt",
        "sha256": "b5c063a1cca585fa100fdc0027d6cdbf4ef33ef5a7fe614230599fb882b51f96",
    },
    "schema_version": 1,
    "source_git_sha": "b74e0f787599475cd9fe719d254202dc9bfc14d5",
    "synthetic_xls_fixture": {
        "path": "tests/fixtures/synthetic_jpx_source_snapshot.xls",
        "sha256": "ca47744896a286e1c56d4d0c09260775772c7df0c01b80d81b7e9a515e6d6aa7",
    },
    "windows_grounded": True,
}

# Plain literal strings, not derived from a live pathlib.Path -- a Path's
# str() renders with OS-native separators, which would silently mismatch
# the JSON's fixed Windows-style backslash literal on a non-Windows run.
EXPECTED_CANDIDATE_CANONICAL_ENVIRONMENT_DIRECTORY = ".venv-real-execution"
EXPECTED_CANDIDATE_CANONICAL_INTERPRETER = ".venv-real-execution\\Scripts\\python.exe"

CANONICAL_PYTHON_EXACT_VERSION = (3, 12, 10)
CANONICAL_PYTHON_IMPLEMENTATION = "CPython"
CANONICAL_PLATFORM_MACHINE = "AMD64"
CANONICAL_PLATFORM_SYSTEM = "Windows"
CANONICAL_SYSCONFIG_PLATFORM = "win-amd64"

_CANDIDATE_TOP_LEVEL_FIELDS = frozenset(
    {
        "artifact_status",
        "canonical_environment_directory",
        "canonical_interpreter",
        "future_protected_execution_authorized",
        "gpt_exact_sha_independent_review_required",
        "pip_version",
        "private_or_sealed_reads",
        "python",
        "real_execution_environment_frozen",
        "real_execution_environment_ready_observed",
        "real_network_requests_to_protected_hosts",
        "requirements_real_execution",
        "research_gates_consumed",
        "resolved_lock",
        "schema_version",
        "source_git_sha",
        "synthetic_xls_fixture",
        "windows_grounded",
    }
)
_CANDIDATE_PYTHON_FIELDS = frozenset(
    {
        "implementation",
        "os_name",
        "platform_machine",
        "platform_release",
        "platform_system",
        "sysconfig_platform",
        "version",
    }
)
_CANDIDATE_REQUIREMENTS_FIELDS = frozenset({"path", "sha256"})
_CANDIDATE_RESOLVED_LOCK_FIELDS = frozenset({"generated_from", "package_count", "path", "sha256"})
_CANDIDATE_FIXTURE_FIELDS = frozenset({"path", "sha256"})

# ---------------------------------------------------------------------------
# Freeze-record check (REAL_EXECUTION_ENVIRONMENT_FREEZE_PROMOTION):
# promotes the reviewed, Windows-validated lock candidate from
# CANDIDATE_VERIFIED_NOT_FROZEN to mechanically enforced FROZEN. Binds to
# hardcoded REVIEWED_* constants -- not merely to whatever
# REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json currently says -- and
# independently re-derives the canonical Git blob SHA-256 of the reviewed
# Windows validation evidence artifact, never trusting the freeze record's
# self-reported hash alone. REAL_EXECUTION_ENVIRONMENT_FROZEN becomes true
# only when this check passes AND the full existing readiness/lock checks
# also pass -- freezing never weakens or replaces any of them.
# ---------------------------------------------------------------------------

FREEZE_RECORD_PATH = REPO_ROOT / "REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json"
WINDOWS_VALIDATION_EVIDENCE_PATH = REPO_ROOT / "REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json"

REVIEWED_TESTED_IMPLEMENTATION_GIT_SHA = "84d4512d800b18b858b6f129be9a4ba0ea73d4ca"
REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_SHA = "f52f31ab6305e321cd9e8e9855d6efd83238f552"
REVIEWED_WINDOWS_VALIDATION_EVIDENCE_CANONICAL_GIT_SHA256 = (
    "c0c54866a54cd4901d029b9cc2a8eaed65d51f342c94858b144637b89956afe4"
)
REVIEWED_FREEZE_ARTIFACT_STATUS = "REAL_EXECUTION_ENVIRONMENT_FROZEN"

# Complete semantic content of the reviewed freeze record. As with
# REVIEWED_LOCK_CANDIDATE_SEMANTIC_CONTENT, this is an exhaustive nested
# object compared with _type_strict_semantic_equal after the exact-schema
# check -- any mutation of any field (frozen True->False, authorization
# False->True, a tampered SHA, a dropped/added package, a changed platform
# value, etc.) fails this binding.
REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT: dict[str, Any] = {
    "artifact_status": REVIEWED_FREEZE_ARTIFACT_STATUS,
    "canonical_environment_directory": ".venv-real-execution",
    "canonical_interpreter": ".venv-real-execution\\Scripts\\python.exe",
    "future_protected_execution_authorized": False,
    "gpt_exact_sha_independent_review_required": True,
    "package_set": [
        "numpy==2.5.2",
        "pandas==3.0.5",
        "pip==25.0.1",
        "python-dateutil==2.9.0.post0",
        "six==1.17.0",
        "tzdata==2026.3",
        "xlrd==2.0.2",
    ],
    "python": {
        "implementation": "CPython",
        "os_name": "nt",
        "platform_machine": "AMD64",
        "platform_system": "Windows",
        "sysconfig_platform": "win-amd64",
        "version": "3.12.10",
    },
    "real_execution_environment_frozen": True,
    "real_execution_environment_ready": True,
    "resolved_lock": {
        "package_count": 7,
        "path": "requirements-real-execution.lock.txt",
        "sha256": "b5c063a1cca585fa100fdc0027d6cdbf4ef33ef5a7fe614230599fb882b51f96",
    },
    "reviewed_lock_candidate_git_sha": "107430894723c2bdc2f8493cb12c467fccd8665e",
    "reviewed_windows_validation_evidence": {
        "canonical_git_sha256": REVIEWED_WINDOWS_VALIDATION_EVIDENCE_CANONICAL_GIT_SHA256,
        "path": "REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json",
    },
    "reviewed_windows_validation_evidence_git_sha": REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_SHA,
    "safety": {
        "jpx_requests": 0,
        "private_or_sealed_reads": 0,
        "protected_network_requests": 0,
        "research_gates_consumed": 0,
        "yahoo_requests": 0,
    },
    "schema_version": 1,
    "source_git_sha": "b74e0f787599475cd9fe719d254202dc9bfc14d5",
    "source_requirements": {
        "canonical_git_sha256": "2cdcfd7a87023c4e9c3ec463cf16f77d88f72ccc8d1f0e5de242e6c68b0cf601",
        "path": "requirements-real-execution.txt",
    },
    "synthetic_xls_fixture": {
        "path": "tests/fixtures/synthetic_jpx_source_snapshot.xls",
        "sha256": "ca47744896a286e1c56d4d0c09260775772c7df0c01b80d81b7e9a515e6d6aa7",
    },
    "tested_implementation_git_sha": REVIEWED_TESTED_IMPLEMENTATION_GIT_SHA,
}

_FREEZE_RECORD_TOP_LEVEL_FIELDS = frozenset(REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT)
_FREEZE_RECORD_PYTHON_FIELDS = frozenset(REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT["python"])
_FREEZE_RECORD_RESOLVED_LOCK_FIELDS = frozenset(REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT["resolved_lock"])
_FREEZE_RECORD_EVIDENCE_REF_FIELDS = frozenset(REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT["reviewed_windows_validation_evidence"])
_FREEZE_RECORD_SAFETY_FIELDS = frozenset(REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT["safety"])
_FREEZE_RECORD_SOURCE_REQUIREMENTS_FIELDS = frozenset(REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT["source_requirements"])
_FREEZE_RECORD_FIXTURE_FIELDS = frozenset(REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT["synthetic_xls_fixture"])

# Exact schema the reviewed Windows validation evidence artifact
# (REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json) must have,
# and the exact fields inside its "validation"/"safety" blocks this check
# requires to be PASS/true/zero. Read from the evidence's own canonical Git
# blob bytes at REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_SHA -- never the
# working-tree copy -- alongside the working-tree copy for a defense-in-
# depth cross-check that both agree.
_EVIDENCE_REQUIRED_VALIDATION_TRUE_FIELDS = (
    "git_preflight_pass",
    "read_only_checker_ready",
    "environment_lock_check_pass",
    "bootstrap_pass",
    "working_tree_clean_after",
)
_EVIDENCE_REQUIRED_VALIDATION_FALSE_FIELDS = ("package_state_drift", "pip_network_allowed")
_EVIDENCE_REQUIRED_VALIDATION_ZERO_EXIT_FIELDS = ("read_only_checker_exit_code", "bootstrap_exit_code")
_EVIDENCE_REQUIRED_SAFETY_ZERO_FIELDS = (
    "protected_network_requests",
    "jpx_requests",
    "yahoo_requests",
    "private_or_sealed_reads",
    "research_gates_consumed",
)


def _type_strict_semantic_equal(actual: Any, expected: Any) -> bool:
    """Return whether two JSON-semantic values match in both type and value.

    Ordinary Python equality is insufficient for a reviewed JSON artifact:
    `True == 1`, `False == 0`, and `1 == 1.0` all evaluate true.  This
    comparator rejects those type-confusion cases recursively.  JSON object
    keys are strings after `json.loads`; their exact sets, every nested
    value, list order, and scalar type/value are all bound here.
    """
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _type_strict_semantic_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _type_strict_semantic_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return actual == expected


def _parse_pinned_requirement(requirements_path: Path, package_name: str) -> str | None:
    """Read an exact `name==version` pin out of a requirements file.

    Returns None if the file is missing, the package is present but
    unpinned, or the package is absent -- callers distinguish those cases
    themselves rather than this helper guessing.
    """
    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(package_name + "==") :
            return stripped.split("==", 1)[1].strip()
    return None


def check_interpreter_identity() -> dict[str, Any]:
    """Interpreter identity for protected execution.

    Accepts ONLY the exact resolved `.venv-real-execution\\Scripts\\python.exe`
    path. This explicitly and mechanically rejects the repository's separate
    general-development `.venv` -- even when that `.venv` happens to be
    Python 3.12 with pandas/xlrd installed -- because `interpreter_match`
    compares the resolved executable path only, never version or package
    state. A Windows run from any interpreter other than the canonical one
    (including, explicitly, the general `.venv`) reports
    `interpreter_failure_class="PRE_GATE_WRONG_PYTHON_ENVIRONMENT"`.
    """
    is_windows = os.name == "nt"
    actual_executable = str(Path(sys.executable).resolve())
    expected_executable = str(CANONICAL_WINDOWS_INTERPRETER.resolve()) if is_windows else None
    general_project_venv_executable = (
        str((GENERAL_PROJECT_VENV_DIR / "Scripts" / "python.exe").resolve()) if is_windows else None
    )

    if is_windows:
        interpreter_match = actual_executable.casefold() == (expected_executable or "").casefold()
        general_project_venv_rejected = (
            not interpreter_match
            and general_project_venv_executable is not None
            and actual_executable.casefold() == general_project_venv_executable.casefold()
        )
    else:
        # The canonical protected interpreter is a Windows path by design
        # (REAL_EXECUTION_PYTHON_ENVIRONMENT.md §2). A non-Windows run can
        # never match it; this is reported explicitly, not silently skipped.
        interpreter_match = False
        general_project_venv_rejected = False

    # Distinct from "not running on Windows at all" (platform_windows_
    # grounded=false, handled separately): this specifically flags a
    # Windows run that resolved to the wrong interpreter.
    interpreter_failure_class = "PRE_GATE_WRONG_PYTHON_ENVIRONMENT" if is_windows and not interpreter_match else None

    version_info = sys.version_info
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    python_major_minor_match = (version_info.major, version_info.minor) == CANONICAL_PYTHON_MAJOR_MINOR
    # Stricter than python_major_minor_match: the reviewed environment lock
    # binds to exact CPython 3.12.10, not merely "3.12" (a 3.12.x other
    # than .10 must not silently satisfy the reviewed lock binding).
    python_patch_match = (version_info.major, version_info.minor, version_info.micro) == CANONICAL_PYTHON_EXACT_VERSION

    return {
        "platform_windows_grounded": is_windows,
        "actual_interpreter": actual_executable,
        "expected_interpreter": expected_executable,
        "interpreter_match": interpreter_match,
        "general_project_venv_rejected": general_project_venv_rejected,
        "interpreter_failure_class": interpreter_failure_class,
        "python_version": python_version,
        "python_major_minor_match": python_major_minor_match,
        "python_patch_match": python_patch_match,
    }


def check_dependency_readiness() -> dict[str, Any]:
    results: dict[str, Any] = {"packages": {}, "status": "PASS"}
    for package_name in _REQUIRED_DIRECT_DEPENDENCIES:
        try:
            module = __import__(package_name)
        except ImportError as error:
            results["packages"][package_name] = {"importable": False, "error": str(error)}
            results["status"] = "FAIL"
            continue
        version = getattr(module, "__version__", "UNKNOWN")
        entry: dict[str, Any] = {"importable": True, "version": version}
        pinned = _parse_pinned_requirement(REQUIREMENTS_REAL_EXECUTION_FILE, package_name)
        if pinned is not None:
            entry["required_pin"] = pinned
            entry["pin_satisfied"] = version == pinned
            if not entry["pin_satisfied"]:
                results["status"] = "FAIL"
        results["packages"][package_name] = entry
    return results


def check_jpx_xls_parser_synthetic_probe() -> dict[str, Any]:
    """Operational (not merely `import pandas`) probe for the real JPX
    ".xls" parsing path, run against the committed, wholly synthetic
    fixture `tests/fixtures/synthetic_jpx_source_snapshot.xls`.

    This drives the REAL production functions end to end and never
    reimplements `pandas.read_excel` here:

      1. read the committed synthetic ".xls" bytes;
      2. `scripts.build_v8_partition_manifest.default_parse_source_table`;
      3. `src.v8_partition.parse_eligible_universe`.

    That mechanically proves, before any protected boundary, that the
    canonical interpreter works, pandas imports, the `xlrd` engine imports
    and can actually parse legacy OLE2/BIFF ".xls" bytes, the production
    `default_parse_source_table` works, and the downstream JPX column
    detection / eligible-universe reconstruction initializes successfully.

    The fixture contains no real JPX payload, no real or private ticker
    membership, and no prices or outcomes -- only obviously synthetic
    placeholder rows (see `scripts/generate_synthetic_jpx_xls_fixture.py`).
    Only safe synthetic properties are verified and reported: no network
    request, no private read, and no gate consumption occurs here.
    """
    if not SYNTHETIC_XLS_FIXTURE_PATH.exists():
        return {
            "status": "FAIL",
            "reason": "SYNTHETIC_XLS_FIXTURE_MISSING",
            "fixture_path_checked": str(SYNTHETIC_XLS_FIXTURE_PATH),
        }
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        # Expected synthetic properties come from the generator module, so
        # the fixture's contents and this probe's expectations cannot drift.
        # Importing this module also re-runs its own module-level
        # `_assert_synthetic_namespace` guard, which raises at import time
        # if any fixture code does not start with the synthetic namespace
        # prefix -- so a later edit cannot silently reintroduce an ordinary
        # numeric JPX-looking code.
        from scripts.generate_synthetic_jpx_xls_fixture import (
            EXPECTED_ELIGIBLE_CODES,
            EXPECTED_ELIGIBLE_ROW_COUNT,
            EXPECTED_FIXTURE_SHA256,
            EXPECTED_TOTAL_ROW_COUNT,
            SYNTHETIC_TICKER_NAMESPACE_PREFIX,
        )
    except ImportError as error:
        return {"status": "FAIL", "reason": "SYNTHETIC_FIXTURE_EXPECTATIONS_UNAVAILABLE", "error": str(error)}
    except AssertionError as error:
        return {"status": "FAIL", "reason": "SYNTHETIC_FIXTURE_CODE_OUTSIDE_NAMESPACE", "error": str(error)}

    # Defense-in-depth restatement of the same namespace guarantee, against
    # the exact expected codes this probe is about to assert on.
    for expected_code in EXPECTED_ELIGIBLE_CODES:
        if not expected_code.startswith(SYNTHETIC_TICKER_NAMESPACE_PREFIX):
            return {
                "status": "FAIL",
                "reason": "SYNTHETIC_FIXTURE_CODE_OUTSIDE_NAMESPACE",
                "offending_code": expected_code,
            }

    raw_bytes = SYNTHETIC_XLS_FIXTURE_PATH.read_bytes()
    fixture_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # Mechanically verify the committed fixture bytes match the recorded
    # canonical identity BEFORE parsing anything -- a fixture whose bytes
    # were tampered with (or a stale/mismatched EXPECTED_FIXTURE_SHA256
    # after a regeneration) must FAIL here, not silently parse whatever is
    # on disk.
    if fixture_sha256 != EXPECTED_FIXTURE_SHA256:
        return {
            "status": "FAIL",
            "reason": "FIXTURE_SHA256_MISMATCH",
            "fixture_sha256": fixture_sha256,
            "expected_fixture_sha256": EXPECTED_FIXTURE_SHA256,
        }

    try:
        import pandas as pd
    except ImportError as error:
        return {"status": "FAIL", "reason": "PANDAS_UNAVAILABLE", "error": str(error)}

    try:
        from scripts.build_v8_partition_manifest import default_parse_source_table
        from src.v8_partition import parse_eligible_universe
    except ImportError as error:
        return {"status": "FAIL", "reason": "PRODUCTION_PARSER_IMPORT_FAILED", "error": str(error)}

    # Step 2: the exact production ".xls" byte parser.
    try:
        frame = default_parse_source_table(raw_bytes)
    except Exception as error:  # noqa: BLE001 -- any parser failure is a probe FAIL, not a crash
        return {"status": "FAIL", "reason": "SYNTHETIC_XLS_PARSE_FAILED", "error": str(error)}
    if not isinstance(frame, pd.DataFrame):
        return {"status": "FAIL", "reason": "SYNTHETIC_XLS_PARSE_RESULT_NOT_DATAFRAME"}
    parsed_row_count = int(len(frame))
    if parsed_row_count != EXPECTED_TOTAL_ROW_COUNT:
        return {
            "status": "FAIL",
            "reason": "SYNTHETIC_XLS_PARSED_ROW_COUNT_UNEXPECTED",
            "parsed_row_count": parsed_row_count,
            "expected_row_count": EXPECTED_TOTAL_ROW_COUNT,
        }

    # Step 3: the exact production eligible-universe reconstruction.
    try:
        eligible_rows, reasons = parse_eligible_universe(frame)
    except Exception as error:  # noqa: BLE001 -- any reconstruction failure is a probe FAIL
        return {"status": "FAIL", "reason": "SYNTHETIC_ELIGIBLE_UNIVERSE_RECONSTRUCTION_FAILED", "error": str(error)}

    eligible_codes = tuple(row["code"] for row in eligible_rows)
    if eligible_codes != tuple(EXPECTED_ELIGIBLE_CODES):
        return {
            "status": "FAIL",
            "reason": "SYNTHETIC_ELIGIBLE_CODES_UNEXPECTED",
            "eligible_row_count": len(eligible_codes),
            "expected_eligible_row_count": EXPECTED_ELIGIBLE_ROW_COUNT,
        }
    if int(reasons.get("eligible_current_only", -1)) != EXPECTED_ELIGIBLE_ROW_COUNT:
        return {
            "status": "FAIL",
            "reason": "SYNTHETIC_ELIGIBLE_REASON_COUNT_UNEXPECTED",
            "reported_eligible_current_only": reasons.get("eligible_current_only"),
            "expected_eligible_row_count": EXPECTED_ELIGIBLE_ROW_COUNT,
        }

    return {
        "status": "PASS",
        "fixture_path": str(SYNTHETIC_XLS_FIXTURE_PATH),
        "fixture_sha256": fixture_sha256,
        "fixture_sha256_verified_against_canonical": True,
        "fixture_is_synthetic_non_sensitive": True,
        "synthetic_ticker_namespace_prefix": SYNTHETIC_TICKER_NAMESPACE_PREFIX,
        "synthetic_namespace_verified": True,
        "production_functions_exercised": [
            "scripts.build_v8_partition_manifest.default_parse_source_table",
            "src.v8_partition.parse_eligible_universe",
        ],
        "parsed_row_count": parsed_row_count,
        "eligible_row_count": len(eligible_codes),
        "legacy_xls_engine_proven_operational": True,
    }


def check_tls_stdlib_initialization() -> dict[str, Any]:
    try:
        context = ssl.create_default_context()
        _ = context.protocol
    except Exception as error:  # noqa: BLE001 -- report, never crash the checker
        return {"status": "FAIL", "error": str(error)}
    return {"status": "PASS"}


def check_trusted_host_request_construction() -> dict[str, Any]:
    """Prove the real production trusted-host/request-construction code
    initializes without issuing any request. Reuses the actual production
    module (`scripts/build_v8_partition_manifest.py`), never a
    reimplementation. Importing that module performs no I/O (guarded by
    `if __name__ == "__main__":`); this function performs no network call.
    """
    try:
        import urllib.request

        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from scripts.build_v8_partition_manifest import (
            JPX_PAGE,
            PRODUCTION_USER_AGENT,
            TrustedJpxRedirectHandler,
        )

        handler = TrustedJpxRedirectHandler()
        request = urllib.request.Request(JPX_PAGE, headers={"User-Agent": PRODUCTION_USER_AGENT})
        if request.full_url != JPX_PAGE or handler is None:
            raise AssertionError("TRUSTED_HOST_REQUEST_CONSTRUCTION_UNEXPECTED_STATE")
    except Exception as error:  # noqa: BLE001 -- report, never crash the checker
        return {"status": "FAIL", "error": str(error)}
    return {"status": "PASS"}


def check_filesystem_durable_publication_probe() -> dict[str, Any]:
    """Exercise the REAL production durable/exclusive publication primitive
    (`src.v8i_source_snapshot._atomic_publish_once` -- staging write,
    mandatory `os.fsync` of the file, atomic no-overwrite `os.link`, a
    best-effort directory fsync via the production `_fsync_directory()`
    semantics, staging cleanup) on a disposable temporary path, rather than
    merely doing an ordinary write/read/unlink on some unrelated file.

    The probe path is mechanically proven never to overlap any real V8I
    gate or private state root before anything is written. This function
    never reads, writes, resets, or deletes any real gate receipt, real
    evidence artifact, or real private state -- every path touched here is
    freshly created inside a `tempfile.TemporaryDirectory()` and destroyed
    when this function returns.

    Proves, by reusing the actual primitive rather than reimplementing it:
      - exclusive/no-overwrite creation semantics (first publish succeeds);
      - a second publication to the same destination correctly fails, via
        the primitive's own real collision guard, not a simulated one;
      - durable byte round-trip;
      - the file-level `os.fsync` the primitive performs unconditionally;
      - the directory fsync the primitive *attempts* via `_fsync_directory()`,
        which is deliberately best-effort: that helper returns silently when
        the platform cannot `os.open()` a directory, so a successful probe
        proves the code path ran, NOT that a directory-entry fsync actually
        reached the disk on every OS;
      - cleanup of the disposable probe artifact only.
    """
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import src.v8i_source_snapshot as v8i_module
        from src.v8i_source_snapshot import V8ISourceSnapshotBlocked, _atomic_publish_once

        durable_roots = {
            str(v8i_module.CANONICAL_V8I_SOURCE_SNAPSHOT_GATE_STATE_ROOT.resolve()),
            str(v8i_module.CANONICAL_V8I_SOURCE_SNAPSHOT_PRIVATE_STATE_ROOT.resolve()),
        }
        with tempfile.TemporaryDirectory(prefix="real-execution-env-probe-") as probe_directory:
            resolved_probe_root = Path(probe_directory).resolve()
            for durable_root in durable_roots:
                if str(resolved_probe_root) == durable_root or str(resolved_probe_root).startswith(
                    durable_root + os.sep
                ):
                    raise AssertionError("FILESYSTEM_PROBE_COLLIDES_WITH_DURABLE_STATE_ROOT")

            probe_output_path = resolved_probe_root / "durable-publication-probe.bin"
            probe_payload = os.urandom(64)

            # Exclusive creation + fsync(file) + atomic no-overwrite link +
            # fsync(directory) -- the exact real production primitive, not
            # a reimplementation of it.
            _atomic_publish_once(probe_payload, probe_output_path, "PROBE_ALREADY_EXISTS", "PROBE_WRITE_FAILED")
            if not probe_output_path.exists():
                raise AssertionError("DURABLE_PUBLICATION_PROBE_OUTPUT_MISSING_AFTER_PUBLISH")

            # Durable byte round-trip.
            read_back_payload = probe_output_path.read_bytes()
            if read_back_payload != probe_payload:
                raise AssertionError("DURABLE_PUBLICATION_PROBE_BYTE_ROUNDTRIP_MISMATCH")

            # A second publication to the same destination must fail --
            # exercises the primitive's own real collision guard.
            collision_correctly_blocked = False
            try:
                _atomic_publish_once(
                    probe_payload, probe_output_path, "PROBE_ALREADY_EXISTS", "PROBE_WRITE_FAILED"
                )
            except V8ISourceSnapshotBlocked as error:
                collision_correctly_blocked = error.reason == "PROBE_ALREADY_EXISTS"
            if not collision_correctly_blocked:
                raise AssertionError("DURABLE_PUBLICATION_PROBE_COLLISION_NOT_BLOCKED")

            # Cleanup of the disposable probe artifact only -- real durable
            # state is never touched by this function in the first place.
            probe_output_path.unlink()
            if probe_output_path.exists():
                raise AssertionError("DURABLE_PUBLICATION_PROBE_CLEANUP_FAILED")
    except AssertionError as error:
        return {"status": "FAIL", "reason": str(error)}
    except Exception as error:  # noqa: BLE001 -- report, never crash the checker
        return {"status": "FAIL", "error": str(error)}
    return {
        "status": "PASS",
        "primitive_reused": "src.v8i_source_snapshot._atomic_publish_once",
        "exclusive_creation_verified": True,
        "collision_second_publish_blocked": True,
        "byte_roundtrip_verified": True,
        "file_fsync_mandatory_in_primitive": True,
        "directory_fsync_attempted_best_effort": True,
        "directory_fsync_guaranteed_on_every_platform": False,
        "cleanup_verified": True,
    }


def _git_blob_bytes(repo_root: Path, git_ref: str) -> bytes | None:
    """Canonical Git object bytes for a `<sha>:<path>` ref.

    Bypasses any working-tree checkout entirely -- including whatever
    line-ending conversion a Windows checkout might apply -- by asking Git
    for the exact committed blob bytes. This is the line-ending-independent
    provenance mechanism: never compare against a checked-out working-tree
    copy for Git-bound provenance. Returns None on any failure (git
    unavailable, ref unresolvable, timeout) rather than raising, so the
    caller can report a safe FAIL reason instead of crashing.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "blob", git_ref],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


_SEPARATOR_RUN_PATTERN = re.compile(r"[-_.]+")


def _normalize_package_name(name: str) -> str:
    """Proper PEP 503 normalization: collapse any run of `-`, `_`, `.` into
    a single `-`, then lowercase (https://peps.python.org/pep-0503/#normalized-names).
    `foo__bar`, `foo.bar`, and `foo--bar` all normalize to the same
    `foo-bar`, matching how PyPI/pip themselves treat these as identical.
    """
    return _SEPARATOR_RUN_PATTERN.sub("-", name.strip()).lower()


def _parse_pinned_lock_lines(text: str) -> dict[str, str]:
    """Parse `name==version` lines out of the REVIEWED lock file into
    {normalized_name: exact_version}. Handles both LF and CRLF line
    endings via `splitlines()`, and strips any stray `\\r`.

    This lenient parser (silently skipping any non-`==` line) is safe ONLY
    because it is applied exclusively to `requirements-real-execution.lock.txt`,
    whose exact bytes are independently SHA-256-verified against the
    reviewed hash immediately before this function is ever called, and
    whose exact reviewed content is known to be clean `name==version`
    lines with no comments or blanks. It must NEVER be used to parse live
    `pip freeze --all` output -- use `_parse_exact_pinned_freeze_lines`
    for that, which fails closed on anything that is not an exact pinned
    entry instead of silently skipping it.
    """
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "==" not in stripped:
            continue
        name, version = stripped.split("==", 1)
        parsed[_normalize_package_name(name)] = version.strip()
    return parsed


# A live `pip freeze --all` entry is accepted ONLY in the exact pinned
# `name==version` form pip itself emits for a normal installed
# distribution. This deliberately rejects, by construction (no `==`
# substring can appear inside a version token that also contains `@`,
# whitespace, or `/`):
#   - direct URL forms:   "name @ file:///..." / "name @ https://..."
#   - editable/VCS forms: "-e ..." / "git+https://..."
#   - any other malformed or non-pinned entry (e.g. a bare "name" with no
#     version, "name == version" with stray whitespace, a `-f`/`--find-links`
#     option line, etc.)
# so none of those can silently vanish from the exact-set comparison the
# way a lenient "skip if no '==' substring" parser would let them.
_EXACT_PIN_LINE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)==(?P<version>[A-Za-z0-9](?:[A-Za-z0-9._+!-]*[A-Za-z0-9])?)$"
)


def _parse_exact_pinned_freeze_lines(text: str) -> tuple[dict[str, str], list[str], list[str]]:
    """Strictly parse live `pip freeze --all` output.

    Returns `(packages, invalid_lines, duplicate_lines)`:
      - `packages`: {normalized_name: exact_version} for every line that IS
        an exact `name==version` pinned entry;
      - `invalid_lines`: every non-empty, non-comment line that is NOT an
        exact pinned entry (direct URL, editable/VCS, malformed, or any
        other non-`name==version` form) -- never silently dropped;
      - `duplicate_lines`: every line whose normalized name already
        appeared earlier -- never silently overwrites the first occurrence.

    The caller must treat ANY non-empty `invalid_lines` or `duplicate_lines`
    as a hard FAIL of the exact-set check, not merely omit them from the
    comparison.
    """
    packages: dict[str, str] = {}
    invalid_lines: list[str] = []
    duplicate_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _EXACT_PIN_LINE_PATTERN.match(stripped)
        if match is None:
            invalid_lines.append(stripped)
            continue
        normalized_name = _normalize_package_name(match.group("name"))
        if normalized_name in packages:
            duplicate_lines.append(stripped)
            continue
        packages[normalized_name] = match.group("version")
    return packages, invalid_lines, duplicate_lines


def check_environment_lock(interpreter: dict[str, Any]) -> dict[str, Any]:
    """Mechanical environment-lock check.

    Binds to the hardcoded REVIEWED_* constants above -- not merely to
    whatever `REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json` and
    `requirements-real-execution.lock.txt` currently say on disk -- so a
    tampered, stale, or accidentally-promoted candidate/lock is
    independently, mechanically detectable rather than silently trusted.

    Fails closed unless, in order:
      1. the candidate manifest and lock file both exist;
      2. the candidate manifest is structurally valid (exact schema, every
         required field present, no unexpected extra field);
      3. the candidate's complete semantic content recursively matches the
         reviewed candidate binding with exact JSON types and values;
      4. the on-disk lock file's independently recomputed SHA-256 matches
         the reviewed lock hash;
      5. the on-disk lock file parses to exactly the reviewed package
         count;
      6. the source requirements file's canonical Git object bytes at the
         reviewed source commit (via `git cat-file blob`, never a
         checked-out working-tree copy) independently hash to the reviewed
         source-requirements SHA-256;
      7. the committed fixture's raw bytes independently hash to the
         reviewed fixture SHA-256;
      8. the live interpreter is the exact canonical
         `.venv-real-execution\\Scripts\\python.exe`;
      9. the live platform is exactly CPython 3.12.10 / Windows / AMD64 /
         win-amd64;
      10. the live `python -m pip freeze --all` package set exactly equals
          the reviewed seven entries -- no extra package, no missing
          package, no version drift.
    """
    detail: dict[str, Any] = {}

    if not LOCK_CANDIDATE_PATH.exists():
        return {"status": "FAIL", "reason": "LOCK_CANDIDATE_MISSING", "detail": detail}
    if not LOCK_FILE_PATH.exists():
        return {"status": "FAIL", "reason": "LOCK_FILE_MISSING", "detail": detail}

    try:
        candidate = json.loads(LOCK_CANDIDATE_PATH.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"status": "FAIL", "reason": "LOCK_CANDIDATE_INVALID_JSON", "error": str(error), "detail": detail}

    if not isinstance(candidate, dict) or set(candidate) != _CANDIDATE_TOP_LEVEL_FIELDS:
        return {"status": "FAIL", "reason": "LOCK_CANDIDATE_SCHEMA_INVALID", "detail": detail}
    python_block = candidate.get("python")
    resolved_lock_block = candidate.get("resolved_lock")
    requirements_block = candidate.get("requirements_real_execution")
    fixture_block = candidate.get("synthetic_xls_fixture")
    if (
        not isinstance(python_block, dict)
        or set(python_block) != _CANDIDATE_PYTHON_FIELDS
        or not isinstance(resolved_lock_block, dict)
        or set(resolved_lock_block) != _CANDIDATE_RESOLVED_LOCK_FIELDS
        or not isinstance(requirements_block, dict)
        or set(requirements_block) != _CANDIDATE_REQUIREMENTS_FIELDS
        or not isinstance(fixture_block, dict)
        or set(fixture_block) != _CANDIDATE_FIXTURE_FIELDS
    ):
        return {"status": "FAIL", "reason": "LOCK_CANDIDATE_SCHEMA_INVALID", "detail": detail}
    detail["candidate_structurally_valid"] = True
    detail["candidate_status"] = candidate.get("artifact_status")
    detail["candidate_package_count"] = resolved_lock_block.get("package_count")

    candidate_semantic_match = _type_strict_semantic_equal(candidate, REVIEWED_LOCK_CANDIDATE_SEMANTIC_CONTENT)
    detail["candidate_semantic_match"] = candidate_semantic_match
    # Retain the existing detail key for callers that consume prior
    # readiness-check output; it now denotes complete semantic equality.
    detail["candidate_matches_reviewed_binding"] = candidate_semantic_match
    if not candidate_semantic_match:
        return {"status": "FAIL", "reason": "LOCK_CANDIDATE_DOES_NOT_MATCH_REVIEWED_BINDING", "detail": detail}

    # Independently recompute the on-disk lock file's SHA-256 -- never
    # trust the candidate JSON's self-reported hash alone.
    lock_bytes = LOCK_FILE_PATH.read_bytes()
    lock_sha256_recomputed = hashlib.sha256(lock_bytes).hexdigest()
    detail["lock_sha256_recomputed"] = lock_sha256_recomputed
    lock_sha256_match = lock_sha256_recomputed == REVIEWED_LOCK_SHA256
    detail["lock_sha256_match"] = lock_sha256_match
    if not lock_sha256_match:
        return {"status": "FAIL", "reason": "LOCK_SHA256_MISMATCH", "detail": detail}

    lock_packages = _parse_pinned_lock_lines(lock_bytes.decode("utf-8"))
    detail["lock_package_count_recomputed"] = len(lock_packages)
    if len(lock_packages) != REVIEWED_PACKAGE_COUNT:
        return {"status": "FAIL", "reason": "LOCK_PACKAGE_COUNT_UNEXPECTED", "detail": detail}

    # Independently recompute canonical Git object bytes for the source
    # requirements file at the reviewed source commit -- CRLF-independent.
    git_blob = _git_blob_bytes(REPO_ROOT, f"{REVIEWED_SOURCE_GIT_SHA}:requirements-real-execution.txt")
    if git_blob is None:
        return {"status": "FAIL", "reason": "SOURCE_REQUIREMENTS_GIT_PROVENANCE_UNAVAILABLE", "detail": detail}
    source_requirements_git_sha256 = hashlib.sha256(git_blob).hexdigest()
    detail["source_requirements_git_sha256_recomputed"] = source_requirements_git_sha256
    source_requirements_provenance_match = source_requirements_git_sha256 == REVIEWED_SOURCE_REQUIREMENTS_GIT_SHA256
    detail["source_requirements_provenance_match"] = source_requirements_provenance_match
    if not source_requirements_provenance_match:
        return {"status": "FAIL", "reason": "SOURCE_REQUIREMENTS_GIT_PROVENANCE_MISMATCH", "detail": detail}

    # Independently recompute the fixture's raw-byte SHA-256.
    if not SYNTHETIC_XLS_FIXTURE_PATH.exists():
        return {"status": "FAIL", "reason": "SYNTHETIC_XLS_FIXTURE_MISSING", "detail": detail}
    fixture_sha256_recomputed = hashlib.sha256(SYNTHETIC_XLS_FIXTURE_PATH.read_bytes()).hexdigest()
    detail["fixture_sha256_recomputed"] = fixture_sha256_recomputed
    fixture_sha256_match = fixture_sha256_recomputed == REVIEWED_FIXTURE_SHA256
    detail["fixture_sha256_match"] = fixture_sha256_match
    if not fixture_sha256_match:
        return {"status": "FAIL", "reason": "FIXTURE_SHA256_MISMATCH", "detail": detail}

    # Live interpreter identity: exact canonical .venv-real-execution path.
    detail["interpreter_match"] = interpreter["interpreter_match"]
    if interpreter["interpreter_match"] is not True:
        return {"status": "FAIL", "reason": "INTERPRETER_NOT_CANONICAL", "detail": detail}

    # Live platform: exact CPython 3.12.10 / Windows / AMD64 / win-amd64.
    detail["python_patch_match"] = interpreter["python_patch_match"]
    platform_binding_match = (
        interpreter["python_patch_match"] is True
        and platform.python_implementation() == CANONICAL_PYTHON_IMPLEMENTATION
        and platform.system() == CANONICAL_PLATFORM_SYSTEM
        and platform.machine() == CANONICAL_PLATFORM_MACHINE
        and sysconfig.get_platform() == CANONICAL_SYSCONFIG_PLATFORM
        and os.name == "nt"
    )
    detail["platform_binding_match"] = platform_binding_match
    detail["live_platform_system"] = platform.system()
    detail["live_platform_machine"] = platform.machine()
    detail["live_sysconfig_platform"] = sysconfig.get_platform()
    if not platform_binding_match:
        return {"status": "FAIL", "reason": "PLATFORM_BINDING_MISMATCH", "detail": detail}

    # Live `pip freeze --all` must contain EXACTLY the reviewed seven
    # entries: no extra package, no missing package, no version drift, and
    # -- per PIP_FREEZE_EXACT_SET_CHECK_IGNORES_NON_EQUALS_ENTRIES -- every
    # single non-empty line must itself be a valid exact `name==version`
    # pin; a direct-URL, editable/VCS, malformed, or duplicate-named entry
    # is never silently skipped, it is a hard FAIL.
    try:
        freeze_result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "FAIL", "reason": "PIP_FREEZE_UNAVAILABLE", "error": str(error), "detail": detail}
    if freeze_result.returncode != 0:
        return {"status": "FAIL", "reason": "PIP_FREEZE_FAILED", "detail": detail}

    live_packages, invalid_freeze_lines, duplicate_freeze_lines = _parse_exact_pinned_freeze_lines(
        freeze_result.stdout
    )
    extra_packages = sorted(set(live_packages) - set(lock_packages))
    missing_packages = sorted(set(lock_packages) - set(live_packages))
    version_mismatched_packages = sorted(
        name for name in (set(lock_packages) & set(live_packages)) if lock_packages[name] != live_packages[name]
    )
    package_set_match = (
        not extra_packages
        and not missing_packages
        and not version_mismatched_packages
        and not invalid_freeze_lines
        and not duplicate_freeze_lines
    )
    detail["live_package_count"] = len(live_packages)
    detail["extra_packages"] = extra_packages
    detail["missing_packages"] = missing_packages
    detail["version_mismatched_packages"] = version_mismatched_packages
    detail["invalid_freeze_lines"] = invalid_freeze_lines
    detail["duplicate_freeze_lines"] = duplicate_freeze_lines
    detail["package_set_match"] = package_set_match
    if not package_set_match:
        return {"status": "FAIL", "reason": "PIP_FREEZE_PACKAGE_SET_MISMATCH", "detail": detail}

    return {
        "status": "PASS",
        "reviewed_lock_candidate_git_sha": REVIEWED_LOCK_CANDIDATE_GIT_SHA,
        "detail": detail,
    }


def _validate_evidence_schema(evidence: Any) -> bool:
    if not isinstance(evidence, dict):
        return False
    top_fields = {
        "artifact_status",
        "canonical_environment_directory",
        "canonical_interpreter",
        "future_protected_execution_authorized",
        "gpt_exact_sha_independent_review_required",
        "package_set",
        "python",
        "real_execution_environment_frozen",
        "real_execution_environment_ready",
        "resolved_lock",
        "reviewed_lock_candidate_git_sha",
        "safety",
        "schema_version",
        "source_git_sha",
        "source_requirements",
        "synthetic_xls_fixture",
        "tested_git_sha",
        "validation",
    }
    if set(evidence) != top_fields:
        return False
    validation_block = evidence.get("validation")
    safety_block = evidence.get("safety")
    if not isinstance(validation_block, dict) or set(validation_block) != {
        "bootstrap_exit_code",
        "bootstrap_pass",
        "environment_lock_check_pass",
        "git_preflight_pass",
        "package_state_drift",
        "pip_network_allowed",
        "read_only_checker_exit_code",
        "read_only_checker_ready",
        "working_tree_clean_after",
    }:
        return False
    if not isinstance(safety_block, dict) or set(safety_block) != set(_EVIDENCE_REQUIRED_SAFETY_ZERO_FIELDS):
        return False
    return True


def _check_evidence_content(evidence: dict[str, Any], detail: dict[str, Any]) -> str | None:
    """Validate the reviewed Windows validation evidence's own claimed
    content. Returns a FAIL reason string, or None if every requirement is
    satisfied. Populates `detail` regardless of outcome.
    """
    if not _validate_evidence_schema(evidence):
        return "WINDOWS_VALIDATION_EVIDENCE_SCHEMA_INVALID"
    validation_block = evidence["validation"]
    safety_block = evidence["safety"]

    detail["evidence_tested_git_sha"] = evidence.get("tested_git_sha")
    if evidence.get("tested_git_sha") != REVIEWED_TESTED_IMPLEMENTATION_GIT_SHA:
        return "WINDOWS_VALIDATION_EVIDENCE_TESTED_GIT_SHA_MISMATCH"

    for field in _EVIDENCE_REQUIRED_VALIDATION_TRUE_FIELDS:
        if validation_block.get(field) is not True:
            detail["evidence_validation_failure_field"] = field
            return "WINDOWS_VALIDATION_EVIDENCE_VALIDATION_FIELD_NOT_TRUE"
    for field in _EVIDENCE_REQUIRED_VALIDATION_FALSE_FIELDS:
        if validation_block.get(field) is not False:
            detail["evidence_validation_failure_field"] = field
            return "WINDOWS_VALIDATION_EVIDENCE_VALIDATION_FIELD_NOT_FALSE"
    for field in _EVIDENCE_REQUIRED_VALIDATION_ZERO_EXIT_FIELDS:
        value = validation_block.get(field)
        if type(value) is not int or value != 0:
            detail["evidence_validation_failure_field"] = field
            return "WINDOWS_VALIDATION_EVIDENCE_EXIT_CODE_NOT_ZERO"
    for field in _EVIDENCE_REQUIRED_SAFETY_ZERO_FIELDS:
        value = safety_block.get(field)
        if type(value) is not int or value != 0:
            detail["evidence_safety_failure_field"] = field
            return "WINDOWS_VALIDATION_EVIDENCE_SAFETY_COUNT_NOT_ZERO"

    if evidence.get("real_execution_environment_ready") is not True:
        return "WINDOWS_VALIDATION_EVIDENCE_NOT_READY"
    if evidence.get("real_execution_environment_frozen") is not False:
        return "WINDOWS_VALIDATION_EVIDENCE_UNEXPECTED_FROZEN_CLAIM"
    if evidence.get("future_protected_execution_authorized") is not False:
        return "WINDOWS_VALIDATION_EVIDENCE_UNEXPECTED_AUTHORIZATION_CLAIM"

    detail["evidence_content_valid"] = True
    return None


def check_freeze_record(interpreter: dict[str, Any], lock_check: dict[str, Any]) -> dict[str, Any]:
    """Mechanical freeze-record check
    (REAL_EXECUTION_ENVIRONMENT_FREEZE_PROMOTION).

    `REAL_EXECUTION_ENVIRONMENT_FROZEN` is true only when this check PASSes
    AND the full existing environment-lock check (`check_environment_lock`)
    also PASSes -- freezing is strictly additional, it never weakens or
    bypasses any existing readiness/lock check.

    Fails closed unless, in order:
      1. the live environment-lock check itself currently PASSes (a stale
         or invalid lock/candidate/platform means nothing can be frozen);
      2. `REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json` exists and is
         structurally valid (exact schema, no missing/extra field);
      3. the freeze record's complete semantic content recursively,
         type-strictly matches the reviewed freeze binding -- any mutation
         (frozen True->False, authorization False->True, a tampered SHA, a
         changed package/platform value, etc.) fails this;
      4. the freeze record's lock/source-requirements/fixture/platform
         identities match the SAME hardcoded REVIEWED_* constants
         `check_environment_lock` itself uses -- not merely its own,
         separately-editable copy of them;
      5. the freeze record's own `package_set` array, independently
         parsed, matches the on-disk reviewed lock file's independently
         parsed package set exactly;
      6. the reviewed Windows validation evidence artifact's canonical Git
         object bytes at `REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_SHA`
         (via `git cat-file blob`, never a checked-out working-tree copy)
         independently hash to `REVIEWED_WINDOWS_VALIDATION_EVIDENCE_
         CANONICAL_GIT_SHA256` -- never trusting the freeze record's own
         claimed hash alone;
      7. that same canonical Git blob, parsed as JSON, has
         `tested_git_sha == REVIEWED_TESTED_IMPLEMENTATION_GIT_SHA`,
         reports checker/bootstrap PASS with exit code 0,
         `package_state_drift == false`, `pip_network_allowed == false`,
         and every safety count is exactly zero.
    """
    detail: dict[str, Any] = {}

    detail["lock_check_status"] = lock_check.get("status")
    if lock_check.get("status") != "PASS":
        return {"status": "FAIL", "reason": "ENVIRONMENT_LOCK_CHECK_NOT_PASSING", "detail": detail}

    if not FREEZE_RECORD_PATH.exists():
        return {"status": "FAIL", "reason": "FREEZE_RECORD_MISSING", "detail": detail}

    try:
        freeze_record = json.loads(FREEZE_RECORD_PATH.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"status": "FAIL", "reason": "FREEZE_RECORD_INVALID_JSON", "error": str(error), "detail": detail}

    if not isinstance(freeze_record, dict) or set(freeze_record) != _FREEZE_RECORD_TOP_LEVEL_FIELDS:
        return {"status": "FAIL", "reason": "FREEZE_RECORD_SCHEMA_INVALID", "detail": detail}
    python_block = freeze_record.get("python")
    resolved_lock_block = freeze_record.get("resolved_lock")
    evidence_ref_block = freeze_record.get("reviewed_windows_validation_evidence")
    safety_block = freeze_record.get("safety")
    source_requirements_block = freeze_record.get("source_requirements")
    fixture_block = freeze_record.get("synthetic_xls_fixture")
    if (
        not isinstance(python_block, dict)
        or set(python_block) != _FREEZE_RECORD_PYTHON_FIELDS
        or not isinstance(resolved_lock_block, dict)
        or set(resolved_lock_block) != _FREEZE_RECORD_RESOLVED_LOCK_FIELDS
        or not isinstance(evidence_ref_block, dict)
        or set(evidence_ref_block) != _FREEZE_RECORD_EVIDENCE_REF_FIELDS
        or not isinstance(safety_block, dict)
        or set(safety_block) != _FREEZE_RECORD_SAFETY_FIELDS
        or not isinstance(source_requirements_block, dict)
        or set(source_requirements_block) != _FREEZE_RECORD_SOURCE_REQUIREMENTS_FIELDS
        or not isinstance(fixture_block, dict)
        or set(fixture_block) != _FREEZE_RECORD_FIXTURE_FIELDS
        or not isinstance(freeze_record.get("package_set"), list)
    ):
        return {"status": "FAIL", "reason": "FREEZE_RECORD_SCHEMA_INVALID", "detail": detail}
    detail["freeze_record_structurally_valid"] = True

    freeze_record_semantic_match = _type_strict_semantic_equal(freeze_record, REVIEWED_FREEZE_RECORD_SEMANTIC_CONTENT)
    detail["freeze_record_semantic_match"] = freeze_record_semantic_match
    if not freeze_record_semantic_match:
        return {"status": "FAIL", "reason": "FREEZE_RECORD_DOES_NOT_MATCH_REVIEWED_BINDING", "detail": detail}

    # Cross-check against the SAME hardcoded reviewed constants
    # check_environment_lock itself uses -- not merely the freeze record's
    # own, separately-editable copy of the same values.
    cross_check_match = (
        freeze_record["reviewed_lock_candidate_git_sha"] == REVIEWED_LOCK_CANDIDATE_GIT_SHA
        and resolved_lock_block["sha256"] == REVIEWED_LOCK_SHA256
        and resolved_lock_block["package_count"] == REVIEWED_PACKAGE_COUNT
        and freeze_record["source_git_sha"] == REVIEWED_SOURCE_GIT_SHA
        and source_requirements_block["canonical_git_sha256"] == REVIEWED_SOURCE_REQUIREMENTS_GIT_SHA256
        and fixture_block["sha256"] == REVIEWED_FIXTURE_SHA256
        and freeze_record["canonical_environment_directory"] == EXPECTED_CANDIDATE_CANONICAL_ENVIRONMENT_DIRECTORY
        and freeze_record["canonical_interpreter"] == EXPECTED_CANDIDATE_CANONICAL_INTERPRETER
        and python_block["implementation"] == CANONICAL_PYTHON_IMPLEMENTATION
        and python_block["platform_system"] == CANONICAL_PLATFORM_SYSTEM
        and python_block["platform_machine"] == CANONICAL_PLATFORM_MACHINE
        and python_block["sysconfig_platform"] == CANONICAL_SYSCONFIG_PLATFORM
        and python_block["os_name"] == "nt"
        and python_block["version"] == "3.12.10"
        and evidence_ref_block["canonical_git_sha256"] == REVIEWED_WINDOWS_VALIDATION_EVIDENCE_CANONICAL_GIT_SHA256
        and freeze_record["reviewed_windows_validation_evidence_git_sha"]
        == REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_SHA
        and freeze_record["tested_implementation_git_sha"] == REVIEWED_TESTED_IMPLEMENTATION_GIT_SHA
    )
    detail["cross_check_against_lock_check_constants_match"] = cross_check_match
    if not cross_check_match:
        return {"status": "FAIL", "reason": "FREEZE_RECORD_CROSS_CHECK_MISMATCH", "detail": detail}

    # The freeze record's own package_set array, independently parsed,
    # must match the on-disk reviewed lock file's independently parsed
    # package set exactly (internal self-consistency, not merely a static
    # string comparison against the reviewed constant list).
    try:
        freeze_record_packages = _parse_pinned_lock_lines("\n".join(freeze_record["package_set"]) + "\n")
    except (TypeError, AttributeError):
        return {"status": "FAIL", "reason": "FREEZE_RECORD_PACKAGE_SET_UNPARSEABLE", "detail": detail}
    detail["freeze_record_package_count"] = len(freeze_record_packages)
    if len(freeze_record_packages) != REVIEWED_PACKAGE_COUNT:
        return {"status": "FAIL", "reason": "FREEZE_RECORD_PACKAGE_SET_COUNT_UNEXPECTED", "detail": detail}

    if not LOCK_FILE_PATH.exists():
        return {"status": "FAIL", "reason": "LOCK_FILE_MISSING", "detail": detail}
    on_disk_lock_packages = _parse_pinned_lock_lines(LOCK_FILE_PATH.read_bytes().decode("utf-8"))
    package_sets_match = freeze_record_packages == on_disk_lock_packages
    detail["freeze_record_package_set_matches_lock_file"] = package_sets_match
    if not package_sets_match:
        return {"status": "FAIL", "reason": "FREEZE_RECORD_PACKAGE_SET_DOES_NOT_MATCH_LOCK_FILE", "detail": detail}

    # Independently recompute the canonical Git blob SHA-256 of the
    # reviewed Windows validation evidence artifact -- never trust the
    # freeze record's or the working-tree copy's self-reported hash alone.
    evidence_git_blob = _git_blob_bytes(
        REPO_ROOT,
        f"{REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_SHA}:REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json",
    )
    if evidence_git_blob is None:
        return {"status": "FAIL", "reason": "WINDOWS_VALIDATION_EVIDENCE_GIT_PROVENANCE_UNAVAILABLE", "detail": detail}
    evidence_git_sha256_recomputed = hashlib.sha256(evidence_git_blob).hexdigest()
    detail["evidence_git_sha256_recomputed"] = evidence_git_sha256_recomputed
    evidence_git_sha256_match = evidence_git_sha256_recomputed == REVIEWED_WINDOWS_VALIDATION_EVIDENCE_CANONICAL_GIT_SHA256
    detail["evidence_git_sha256_match"] = evidence_git_sha256_match
    if not evidence_git_sha256_match:
        return {"status": "FAIL", "reason": "WINDOWS_VALIDATION_EVIDENCE_GIT_PROVENANCE_MISMATCH", "detail": detail}

    # Parse and validate the evidence's own claimed content from those
    # exact canonical Git bytes (not the working-tree copy).
    try:
        evidence_from_git = json.loads(evidence_git_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return {
            "status": "FAIL",
            "reason": "WINDOWS_VALIDATION_EVIDENCE_INVALID_JSON",
            "error": str(error),
            "detail": detail,
        }
    evidence_failure_reason = _check_evidence_content(evidence_from_git, detail)
    if evidence_failure_reason is not None:
        return {"status": "FAIL", "reason": evidence_failure_reason, "detail": detail}

    # Defense-in-depth: the working-tree copy, if present, must agree.
    if WINDOWS_VALIDATION_EVIDENCE_PATH.exists():
        try:
            evidence_working_tree = json.loads(WINDOWS_VALIDATION_EVIDENCE_PATH.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            return {
                "status": "FAIL",
                "reason": "WINDOWS_VALIDATION_EVIDENCE_WORKING_TREE_INVALID_JSON",
                "error": str(error),
                "detail": detail,
            }
        working_tree_matches_git_blob = _type_strict_semantic_equal(evidence_working_tree, evidence_from_git)
        detail["evidence_working_tree_matches_git_blob"] = working_tree_matches_git_blob
        if not working_tree_matches_git_blob:
            return {"status": "FAIL", "reason": "WINDOWS_VALIDATION_EVIDENCE_WORKING_TREE_DRIFTED", "detail": detail}

    return {
        "status": "PASS",
        "reviewed_windows_validation_evidence_git_sha": REVIEWED_WINDOWS_VALIDATION_EVIDENCE_GIT_SHA,
        "tested_implementation_git_sha": REVIEWED_TESTED_IMPLEMENTATION_GIT_SHA,
        "detail": detail,
    }


def check_v9_014_e7_bundle() -> dict[str, Any]:
    """Validate the reviewed E7 closure without touching an environment.

    The result is an input gate for future E9/E10 tooling, not evidence that
    the canonical interpreter has been mutated or validated.
    """
    detail: dict[str, Any] = {}
    try:
        lock_bytes = V9_014_E7_LOCK_CANDIDATE_PATH.read_bytes()
        evidence_bytes = V9_014_E7_WINDOWS_EVIDENCE_PATH.read_bytes()
    except OSError as error:
        return {"status": "FAIL", "reason": "V9_014_E7_ARTIFACT_MISSING", "error": str(error), "detail": detail}

    detail["lock_candidate_sha256"] = hashlib.sha256(lock_bytes).hexdigest()
    detail["windows_evidence_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    if detail["lock_candidate_sha256"] != V9_014_E7_LOCK_CANDIDATE_SHA256:
        return {"status": "FAIL", "reason": "V9_014_E7_LOCK_CANDIDATE_HASH_MISMATCH", "detail": detail}
    if detail["windows_evidence_sha256"] != V9_014_E7_WINDOWS_EVIDENCE_SHA256:
        return {"status": "FAIL", "reason": "V9_014_E7_WINDOWS_EVIDENCE_HASH_MISMATCH", "detail": detail}

    try:
        from scripts import v9_014_pdf_env_successor as successor

        lock_candidate = json.loads(lock_bytes.decode("utf-8"))
        windows_evidence = json.loads(evidence_bytes.decode("utf-8"))
    except (ImportError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return {"status": "FAIL", "reason": "V9_014_E7_ARTIFACT_INVALID", "error": str(error), "detail": detail}

    if not successor.validate_lock_candidate_schema(lock_candidate):
        return {"status": "FAIL", "reason": "V9_014_E7_LOCK_CANDIDATE_INVALID", "detail": detail}
    if not successor.validate_windows_evidence_schema(windows_evidence):
        return {"status": "FAIL", "reason": "V9_014_E7_WINDOWS_EVIDENCE_INVALID", "detail": detail}
    if not successor.validate_e7_candidate_bundle(lock_candidate, windows_evidence):
        return {"status": "FAIL", "reason": "V9_014_E7_BUNDLE_IDENTITY_MISMATCH", "detail": detail}
    detail["resolved_package_count"] = len(lock_candidate["resolved_packages"])
    detail["predecessor_package_count"] = len(windows_evidence["before_freeze"])
    return {
        "status": "PASS",
        "detail": detail,
        "lock_candidate": lock_candidate,
        "windows_evidence": windows_evidence,
    }


def _v9_014_probe_status(probe: Any) -> str | None:
    if isinstance(probe, dict):
        return probe.get("status")
    return getattr(probe, "status", None)


def _v9_014_probe_field(probe: Any, field: str) -> Any:
    if isinstance(probe, dict):
        return probe.get(field)
    return getattr(probe, field, None)


def check_v9_014_successor_promotion(
    promotion_state: str,
    *,
    live_packages: dict[str, str] | None = None,
    platform_evidence: dict[str, Any] | None = None,
    xls_probe: dict[str, Any] | None = None,
    pdf_probe: Any = None,
) -> dict[str, Any]:
    """Fail-closed future-E9/E10 checker for the reviewed successor closure.

    E8 uses this only as offline tooling.  The migration-state branch is
    deliberately the sole branch that can assess a live successor package
    set; the final frozen state remains unavailable until Stage E15 review.
    """
    if promotion_state not in V9_014_PROMOTION_STATES:
        return {"status": "FAIL", "reason": "V9_014_PROMOTION_STATE_INVALID", "promotion_state": promotion_state}

    bundle = check_v9_014_e7_bundle()
    if bundle["status"] != "PASS":
        return {"status": "FAIL", "reason": bundle["reason"], "promotion_state": promotion_state, "bundle": bundle}

    if promotion_state == V9_014_PREDECESSOR_CANONICAL_FROZEN:
        return {
            "status": "PASS",
            "promotion_state": promotion_state,
            "canonical_environment_mutation_authorized": False,
            "successor_live_validation": "NOT_RUN_PREDECESSOR_STATE",
            "bundle": bundle["detail"],
        }
    if promotion_state == V9_014_SUCCESSOR_CANONICAL_FROZEN:
        return {
            "status": "FAIL",
            "reason": "V9_014_SUCCESSOR_FROZEN_REQUIRES_E15_REVIEW",
            "promotion_state": promotion_state,
            "bundle": bundle["detail"],
        }

    expected_packages = bundle["lock_candidate"]["resolved_packages"]
    if live_packages is None:
        try:
            freeze_result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze", "--all"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return {"status": "FAIL", "reason": "V9_014_SUCCESSOR_PIP_FREEZE_UNAVAILABLE", "error": str(error)}
        if freeze_result.returncode != 0:
            return {"status": "FAIL", "reason": "V9_014_SUCCESSOR_PIP_FREEZE_FAILED"}
        live_packages, invalid_lines, duplicate_lines = _parse_exact_pinned_freeze_lines(freeze_result.stdout)
        if invalid_lines or duplicate_lines:
            return {
                "status": "FAIL",
                "reason": "V9_014_SUCCESSOR_PIP_FREEZE_NOT_EXACT",
                "invalid_lines": invalid_lines,
                "duplicate_lines": duplicate_lines,
            }
    if live_packages != expected_packages:
        return {
            "status": "FAIL",
            "reason": "V9_014_SUCCESSOR_PACKAGE_SET_MISMATCH",
            "expected_package_count": len(expected_packages),
            "actual_package_count": len(live_packages),
        }

    if platform_evidence is None:
        interpreter = check_interpreter_identity()
        platform_evidence = {
            "implementation": platform.python_implementation(),
            "version": list(sys.version_info[:3]),
            "os_name": os.name,
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "sysconfig_platform": sysconfig.get_platform(),
            "interpreter_match": interpreter["interpreter_match"],
        }
    expected_platform = dict(bundle["windows_evidence"]["platform"])
    actual_platform = {key: platform_evidence.get(key) for key in expected_platform}
    if actual_platform != expected_platform or platform_evidence.get("interpreter_match") is not True:
        return {"status": "FAIL", "reason": "V9_014_SUCCESSOR_PLATFORM_BINDING_MISMATCH"}

    if xls_probe is None:
        xls_probe = check_jpx_xls_parser_synthetic_probe()
    if _v9_014_probe_status(xls_probe) != "PASS":
        return {"status": "FAIL", "reason": "V9_014_SUCCESSOR_XLS_PROBE_FAILED"}

    if pdf_probe is None:
        from scripts.v9_014_pdf_env_successor import run_synthetic_pdf_operational_probe

        pdf_probe = run_synthetic_pdf_operational_probe()
    expected_pdf_probe = bundle["windows_evidence"]["synthetic_pdf_probe"]
    if (
        _v9_014_probe_status(pdf_probe) != "SYNTHETIC_PDF_PROBE_PASS"
        or _v9_014_probe_field(pdf_probe, "observed_fixture_sha256") != expected_pdf_probe["fixture_sha256"]
        or _v9_014_probe_field(pdf_probe, "observed_pdfplumber_version") != expected_pdf_probe["pdfplumber_version"]
        or _v9_014_probe_field(pdf_probe, "observed_page_count") != expected_pdf_probe["page_count"]
    ):
        return {"status": "FAIL", "reason": "V9_014_SUCCESSOR_PDF_PROBE_FAILED"}

    return {
        "status": "PASS",
        "promotion_state": promotion_state,
        "canonical_environment_mutation_authorized": False,
        "successor_package_count": len(expected_packages),
        "predecessor_package_count": len(bundle["windows_evidence"]["before_freeze"]),
        "e7_bundle": bundle["detail"],
        "safety": bundle["windows_evidence"]["safety"],
    }


def run_readiness_checks() -> dict[str, Any]:
    interpreter = check_interpreter_identity()
    dependencies = check_dependency_readiness()
    xls_probe = check_jpx_xls_parser_synthetic_probe()
    tls_probe = check_tls_stdlib_initialization()
    trusted_host_probe = check_trusted_host_request_construction()
    filesystem_probe = check_filesystem_durable_publication_probe()
    lock_check = check_environment_lock(interpreter)
    lock_detail = lock_check.get("detail", {})

    ready = (
        interpreter["platform_windows_grounded"] is True
        and interpreter["interpreter_match"] is True
        and interpreter["python_patch_match"] is True
        and dependencies["status"] == "PASS"
        and xls_probe["status"] == "PASS"
        and tls_probe["status"] == "PASS"
        and trusted_host_probe["status"] == "PASS"
        and filesystem_probe["status"] == "PASS"
        and lock_check["status"] == "PASS"
    )

    # Freeze is strictly additional to, and requires, full readiness --
    # REAL_EXECUTION_ENVIRONMENT_FROZEN can never be true unless `ready` is
    # also true (dependency/xls/tls/trusted-host/filesystem/lock checks all
    # PASS on the live, Windows-grounded canonical environment), plus the
    # freeze record itself binds correctly. On any non-Windows run `ready`
    # is always False, so FROZEN is always False here too.
    freeze_check = check_freeze_record(interpreter, lock_check)
    freeze_detail = freeze_check.get("detail", {})
    frozen = ready and freeze_check["status"] == "PASS"

    return {
        "REAL_EXECUTION_ENVIRONMENT_READY": ready,
        "STATIC_CLOUD_VALIDATION_ONLY": not interpreter["platform_windows_grounded"],
        "INTERPRETER_MATCH": interpreter["interpreter_match"],
        "INTERPRETER_FAILURE_CLASS": interpreter["interpreter_failure_class"],
        "GENERAL_PROJECT_VENV_REJECTED": interpreter["general_project_venv_rejected"],
        "PYTHON_VERSION": interpreter["python_version"],
        "PYTHON_MAJOR_MINOR_MATCH": interpreter["python_major_minor_match"],
        "PYTHON_PATCH_MATCH": interpreter["python_patch_match"],
        "DEPENDENCY_READINESS": dependencies["status"],
        "DEPENDENCY_DETAIL": dependencies["packages"],
        "JPX_XLS_PARSER_SYNTHETIC_PROBE": xls_probe["status"],
        "JPX_XLS_PARSER_SYNTHETIC_PROBE_DETAIL": xls_probe,
        "TLS_STDLIB_PROBE": tls_probe["status"],
        "TRUSTED_HOST_REQUEST_CONSTRUCTION_PROBE": trusted_host_probe["status"],
        "FILESYSTEM_PROBE": filesystem_probe["status"],
        "FILESYSTEM_PROBE_DETAIL": filesystem_probe,
        "ENVIRONMENT_LOCK_CHECK": lock_check["status"],
        "ENVIRONMENT_LOCK_CHECK_DETAIL": lock_check,
        "ENVIRONMENT_LOCK_FINGERPRINT_STATUS": (
            "FROZEN"
            if frozen
            else "CANDIDATE_VERIFIED_NOT_FROZEN"
            if lock_check["status"] == "PASS"
            else "CANDIDATE_INVALID_OR_UNVERIFIED"
        ),
        "ENVIRONMENT_LOCK_PACKAGE_SET_MATCH": lock_detail.get("package_set_match"),
        "ENVIRONMENT_LOCK_PACKAGE_COUNT": lock_detail.get(
            "lock_package_count_recomputed", lock_detail.get("candidate_package_count")
        ),
        "ENVIRONMENT_LOCK_SHA256_MATCH": lock_detail.get("lock_sha256_match"),
        "ENVIRONMENT_FREEZE_CHECK": freeze_check["status"],
        "ENVIRONMENT_FREEZE_CHECK_DETAIL": freeze_check,
        "ENVIRONMENT_FREEZE_EVIDENCE_GIT_SHA256_MATCH": freeze_detail.get("evidence_git_sha256_match"),
        "REAL_EXECUTION_ENVIRONMENT_FROZEN": frozen,
        "REAL_NETWORK_REQUESTS": 0,
        "PRIVATE_READS": 0,
        "GATES_CONSUMED": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="No-network canonical environment readiness checker")
    parser.add_argument(
        "--v9-014-promotion-state",
        choices=sorted(V9_014_PROMOTION_STATES),
        help="Run the future E9/E10 successor validator in the declared lifecycle state.",
    )
    args = parser.parse_args()
    if args.v9_014_promotion_state is not None:
        result = check_v9_014_successor_promotion(args.v9_014_promotion_state)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if result["status"] == "PASS" else 1
    result = run_readiness_checks()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["REAL_EXECUTION_ENVIRONMENT_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
