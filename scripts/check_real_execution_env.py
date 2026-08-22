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

Exit code is 0 only if `REAL_EXECUTION_ENVIRONMENT_READY` is true; nonzero
otherwise, always before any protected boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import sys
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

    return {
        "platform_windows_grounded": is_windows,
        "actual_interpreter": actual_executable,
        "expected_interpreter": expected_executable,
        "interpreter_match": interpreter_match,
        "general_project_venv_rejected": general_project_venv_rejected,
        "interpreter_failure_class": interpreter_failure_class,
        "python_version": python_version,
        "python_major_minor_match": python_major_minor_match,
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


def run_readiness_checks() -> dict[str, Any]:
    interpreter = check_interpreter_identity()
    dependencies = check_dependency_readiness()
    xls_probe = check_jpx_xls_parser_synthetic_probe()
    tls_probe = check_tls_stdlib_initialization()
    trusted_host_probe = check_trusted_host_request_construction()
    filesystem_probe = check_filesystem_durable_publication_probe()

    ready = (
        interpreter["platform_windows_grounded"] is True
        and interpreter["interpreter_match"] is True
        and interpreter["python_major_minor_match"] is True
        and dependencies["status"] == "PASS"
        and xls_probe["status"] == "PASS"
        and tls_probe["status"] == "PASS"
        and trusted_host_probe["status"] == "PASS"
        and filesystem_probe["status"] == "PASS"
    )

    return {
        "REAL_EXECUTION_ENVIRONMENT_READY": ready,
        "STATIC_CLOUD_VALIDATION_ONLY": not interpreter["platform_windows_grounded"],
        "INTERPRETER_MATCH": interpreter["interpreter_match"],
        "INTERPRETER_FAILURE_CLASS": interpreter["interpreter_failure_class"],
        "GENERAL_PROJECT_VENV_REJECTED": interpreter["general_project_venv_rejected"],
        "PYTHON_VERSION": interpreter["python_version"],
        "PYTHON_MAJOR_MINOR_MATCH": interpreter["python_major_minor_match"],
        "DEPENDENCY_READINESS": dependencies["status"],
        "DEPENDENCY_DETAIL": dependencies["packages"],
        "JPX_XLS_PARSER_SYNTHETIC_PROBE": xls_probe["status"],
        "JPX_XLS_PARSER_SYNTHETIC_PROBE_DETAIL": xls_probe,
        "TLS_STDLIB_PROBE": tls_probe["status"],
        "TRUSTED_HOST_REQUEST_CONSTRUCTION_PROBE": trusted_host_probe["status"],
        "FILESYSTEM_PROBE": filesystem_probe["status"],
        "FILESYSTEM_PROBE_DETAIL": filesystem_probe,
        "ENVIRONMENT_LOCK_FINGERPRINT_STATUS": "NOT_YET_ESTABLISHED",
        "REAL_EXECUTION_ENVIRONMENT_FROZEN": False,
        "REAL_NETWORK_REQUESTS": 0,
        "PRIVATE_READS": 0,
        "GATES_CONSUMED": 0,
    }


def main() -> int:
    result = run_readiness_checks()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["REAL_EXECUTION_ENVIRONMENT_READY"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
