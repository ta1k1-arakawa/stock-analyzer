"""No-network, no-private-data readiness checker for the canonical
V8-lineage real-execution Python environment.

See `REAL_EXECUTION_PYTHON_ENVIRONMENT.md` for the human-readable contract
this script mechanically enforces, and `AI_REAL_EXECUTION_RUNBOOK.md` §15-19
for where this fits in the overall pre-authorization ordering. This script
is environment-readiness only: it never consumes a human research gate,
never calls JPX/Yahoo, never accesses private/sealed data, and never
executes any V8I/V8J real acquisition. It never opens a network socket and
never reads or writes any real gate receipt or evidence artifact -- it only
references their canonical path *constants* (imported from
`src.v8i_source_snapshot`) to prove its own disposable filesystem probe
never collides with them.

This script is safe to run on any platform for static/structural
validation, but it can only ever report a Windows-grounded
`REAL_EXECUTION_ENVIRONMENT_READY=true` when actually run on Windows, inside
the canonical `.venv`, via `.venv\\Scripts\\python.exe`. When run anywhere
else (including this repository's own Claude Code Cloud / Linux sessions),
`platform_windows_grounded` is always `false` and
`REAL_EXECUTION_ENVIRONMENT_READY` is always `false`, regardless of what
every other individual check reports -- this script must never claim
Windows-grounded readiness from a non-Windows run.

Known open item: the JPX ".xls" operational parser probe currently reports
`CHATGPT_DECISION_REQUIRED`, not `PASS`, because this repository has no
genuine synthetic ".xls" (OLE2/BIFF) fixture and none can be produced
without either a new dependency (e.g. `xlwt`) or risky hand-rolled binary
construction -- see `REAL_EXECUTION_PYTHON_ENVIRONMENT.md` §6. This script
does not fabricate a `PASS` for that probe.

Exit code is 0 only if `REAL_EXECUTION_ENVIRONMENT_READY` is true; nonzero
otherwise, always before any protected boundary.
"""

from __future__ import annotations

import io
import json
import os
import ssl
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_VENV_DIR = REPO_ROOT / ".venv"
CANONICAL_WINDOWS_INTERPRETER = CANONICAL_VENV_DIR / "Scripts" / "python.exe"
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
    is_windows = os.name == "nt"
    actual_executable = str(Path(sys.executable).resolve())
    expected_executable = str(CANONICAL_WINDOWS_INTERPRETER.resolve()) if is_windows else None

    if is_windows:
        interpreter_match = actual_executable.casefold() == (expected_executable or "").casefold()
    else:
        # The canonical protected interpreter is a Windows path by design
        # (REAL_EXECUTION_PYTHON_ENVIRONMENT.md §2). A non-Windows run can
        # never match it; this is reported explicitly, not silently skipped.
        interpreter_match = False

    version_info = sys.version_info
    python_version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    python_major_minor_match = (version_info.major, version_info.minor) == CANONICAL_PYTHON_MAJOR_MINOR

    return {
        "platform_windows_grounded": is_windows,
        "actual_interpreter": actual_executable,
        "expected_interpreter": expected_executable,
        "interpreter_match": interpreter_match,
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
    ".xls" parsing path. Mirrors default_parse_source_table's exact call
    (`pandas.read_excel(io.BytesIO(raw_bytes), dtype=str)`, no `engine=`
    override) against a genuine local synthetic ".xls" fixture, if one
    exists and is independently reviewed and repository-provided.

    No such fixture currently exists in this repository -- see
    REAL_EXECUTION_PYTHON_ENVIRONMENT.md §6. This function reports
    CHATGPT_DECISION_REQUIRED rather than fabricating a PASS or silently
    skipping the check.
    """
    if not SYNTHETIC_XLS_FIXTURE_PATH.exists():
        return {
            "status": "CHATGPT_DECISION_REQUIRED",
            "reason": "REAL_EXECUTION_XLS_SYNTHETIC_FIXTURE_STRATEGY",
            "fixture_path_checked": str(SYNTHETIC_XLS_FIXTURE_PATH),
        }
    try:
        import pandas as pd
    except ImportError as error:
        return {"status": "FAIL", "reason": "PANDAS_UNAVAILABLE", "error": str(error)}
    try:
        raw_bytes = SYNTHETIC_XLS_FIXTURE_PATH.read_bytes()
        frame = pd.read_excel(io.BytesIO(raw_bytes), dtype=str)
    except Exception as error:  # noqa: BLE001 -- any parser failure is a probe FAIL, not a crash
        return {"status": "FAIL", "reason": "SYNTHETIC_XLS_PARSE_FAILED", "error": str(error)}
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {"status": "FAIL", "reason": "SYNTHETIC_XLS_PARSE_RESULT_EMPTY_OR_INVALID"}
    return {"status": "PASS", "fixture_path": str(SYNTHETIC_XLS_FIXTURE_PATH), "parsed_row_count": int(len(frame))}


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


def check_filesystem_probe() -> dict[str, Any]:
    """Prove disposable filesystem primitives are usable, on a temporary
    probe path that is mechanically proven never to collide with any real
    durable gate/private V8I state root. Never reads or writes real gate
    receipt or evidence state.
    """
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        import src.v8i_source_snapshot as v8i_module

        durable_roots = {
            str(v8i_module.CANONICAL_V8I_SOURCE_SNAPSHOT_GATE_STATE_ROOT.resolve()),
            str(v8i_module.CANONICAL_V8I_SOURCE_SNAPSHOT_PRIVATE_STATE_ROOT.resolve()),
        }
        with tempfile.TemporaryDirectory(prefix="real-execution-env-probe-") as probe_directory:
            resolved_probe = Path(probe_directory).resolve()
            for durable_root in durable_roots:
                if str(resolved_probe) == durable_root or str(resolved_probe).startswith(durable_root + os.sep):
                    raise AssertionError("FILESYSTEM_PROBE_COLLIDES_WITH_DURABLE_STATE_ROOT")
            probe_file = resolved_probe / "probe.txt"
            probe_file.write_text("real-execution-env-probe", encoding="utf-8")
            read_back = probe_file.read_text(encoding="utf-8")
            if read_back != "real-execution-env-probe":
                raise AssertionError("FILESYSTEM_PROBE_ROUNDTRIP_MISMATCH")
            probe_file.unlink()
    except Exception as error:  # noqa: BLE001 -- report, never crash the checker
        return {"status": "FAIL", "error": str(error)}
    return {"status": "PASS"}


def run_readiness_checks() -> dict[str, Any]:
    interpreter = check_interpreter_identity()
    dependencies = check_dependency_readiness()
    xls_probe = check_jpx_xls_parser_synthetic_probe()
    tls_probe = check_tls_stdlib_initialization()
    trusted_host_probe = check_trusted_host_request_construction()
    filesystem_probe = check_filesystem_probe()

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
        "PYTHON_VERSION": interpreter["python_version"],
        "PYTHON_MAJOR_MINOR_MATCH": interpreter["python_major_minor_match"],
        "DEPENDENCY_READINESS": dependencies["status"],
        "DEPENDENCY_DETAIL": dependencies["packages"],
        "JPX_XLS_PARSER_SYNTHETIC_PROBE": xls_probe["status"],
        "JPX_XLS_PARSER_SYNTHETIC_PROBE_DETAIL": xls_probe,
        "TLS_STDLIB_PROBE": tls_probe["status"],
        "TRUSTED_HOST_REQUEST_CONSTRUCTION_PROBE": trusted_host_probe["status"],
        "FILESYSTEM_PROBE": filesystem_probe["status"],
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
