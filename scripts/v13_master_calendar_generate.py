"""Dormant V13 calendar generator. Real execution requires a later reviewed gate.

Importing this module never imports or constructs the JPX calendar provider.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.v13_public_data_lock import (
    CALENDAR_PROVENANCE, RawLock, digest, parse_canonical_calendar,
    serialize_calendar_schedule, validate_calendar_anchors,
    validate_calendar_release_artifact,
)

CALENDAR_NAME = "V13_MASTER_CALENDAR.txt"
RECEIPT_NAME = "V13_MASTER_CALENDAR_SAFE_RECEIPT.json"


def _outside_repo(output_root: Path) -> Path:
    root = output_root.resolve(strict=False)
    if not output_root.is_absolute() or root == REPO_ROOT or REPO_ROOT in root.parents:
        raise ValueError("OUTPUT_ROOT_MUST_BE_OUTSIDE_REPOSITORY")
    if not root.parent.is_dir():
        raise ValueError("OUTPUT_PARENT_MISSING")
    if root.exists() or root.is_symlink():
        raise FileExistsError("OUTPUT_ROOT_ALREADY_EXISTS")
    return root


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_BINARY, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def publish_calendar(schedule: object, output_root: Path, implementation_sha: str) -> dict[str, object]:
    """Pure publication boundary; synthetic schedules can exercise this offline."""
    root = _outside_repo(output_root)
    if len(implementation_sha) != 40 or any(c not in "0123456789abcdef" for c in implementation_sha):
        raise ValueError("IMPLEMENTATION_SHA_INVALID")
    payload = serialize_calendar_schedule(schedule)
    sessions, manifest = parse_canonical_calendar(RawLock.from_bytes(payload))
    validate_calendar_anchors(sessions)
    calendar_sha = digest(payload)
    if manifest["session_sha256"] != calendar_sha:
        raise ValueError("CALENDAR_CANONICAL_BYTES_MISMATCH")
    receipt: dict[str, object] = {
        "schema": "V13_MASTER_CALENDAR_SAFE_RECEIPT_V1",
        "status": "PASS",
        "failure_class": "NONE",
        "implementation_sha": implementation_sha,
        "source_identity": dict(CALENDAR_PROVENANCE)["source_identity"],
        "coverage_start": "2015-01-01",
        "coverage_end": "2025-12-31",
        "calendar_sha256": calendar_sha,
        "session_count": len(sessions),
        "anchor_2020_10_01": "INELIGIBLE",
        "anchor_2020_10_02": "ELIGIBLE",
    }
    root.mkdir(parents=False, exist_ok=False)
    _write_new(root / CALENDAR_NAME, payload)
    _write_new(root / RECEIPT_NAME, (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
    return receipt


def _publish_failure(output_root: Path, implementation_sha: str) -> None:
    """Record a post-source-validation failure without exposing schedule content."""
    root = output_root.resolve(strict=False)
    if not output_root.is_absolute() or root == REPO_ROOT or REPO_ROOT in root.parents:
        raise ValueError("OUTPUT_ROOT_MUST_BE_OUTSIDE_REPOSITORY")
    root.mkdir(parents=False, exist_ok=False)
    receipt_path = root / RECEIPT_NAME
    receipt = {
        "schema": "V13_MASTER_CALENDAR_SAFE_RECEIPT_V1",
        "status": "FAIL",
        "failure_class": "IMPLEMENTATION_FAILURE",
        "implementation_sha": implementation_sha,
        "source_identity": dict(CALENDAR_PROVENANCE)["source_identity"],
        "coverage_start": "2015-01-01",
        "coverage_end": "2025-12-31",
        "calendar_sha256": None,
        "session_count": None,
        "anchor_2020_10_01": "NOT_CONFIRMED",
        "anchor_2020_10_02": "NOT_CONFIRMED",
    }
    _write_new(receipt_path, (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def generate(wheel_path: Path, output_root: Path, implementation_sha: str) -> dict[str, object]:
    """Production path. The reviewed PowerShell wrapper must pass every preflight first."""
    canonical_python = REPO_ROOT / ".venv-real-execution" / "Scripts" / "python.exe"
    if Path(sys.executable).resolve() != canonical_python.resolve():
        raise ValueError("CANONICAL_INTERPRETER_REQUIRED")
    _outside_repo(output_root)
    if len(implementation_sha) != 40 or any(c not in "0123456789abcdef" for c in implementation_sha):
        raise ValueError("IMPLEMENTATION_SHA_INVALID")
    expected = dict(CALENDAR_PROVENANCE)
    if wheel_path.name != expected["official_pypi_wheel"] or not wheel_path.is_file():
        raise ValueError("OFFICIAL_WHEEL_REQUIRED")
    for package, version in (("pandas-market-calendars", "5.4.0"),
                             ("exchange-calendars", "4.13.2"), ("pandas", "3.0.5")):
        if importlib.metadata.version(package) != version:
            raise ValueError("PROTECTED_PACKAGE_VERSION_MISMATCH")
    distribution = importlib.metadata.distribution("pandas-market-calendars")
    installed_sources = {}
    for key in ("jpx_source_file", "holiday_source_file"):
        name = expected[key]
        installed_sources[name] = Path(distribution.locate_file(name)).read_bytes()
    wheel = wheel_path.read_bytes()
    validate_calendar_release_artifact(wheel, installed_sources, expected)

    # The provider import and construction are strictly beyond source validation.
    try:
        import pandas_market_calendars as mcal
        schedule = mcal.get_calendar("JPX").schedule(start_date="2015-01-01", end_date="2025-12-31")
        return publish_calendar(schedule, output_root, implementation_sha)
    except Exception:
        _publish_failure(output_root, implementation_sha)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-wheel", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--implementation-sha", required=True)
    args = parser.parse_args()
    try:
        receipt = generate(args.official_wheel, args.output_root, args.implementation_sha)
    except Exception:
        print(json.dumps({"status": "FAIL", "failure_class": "GENERATION_OR_PREFLIGHT_FAILURE"}, sort_keys=True))
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
