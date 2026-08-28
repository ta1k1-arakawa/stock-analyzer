"""Synthetic-only V7 Gate 4 activation manifest contract check.

This CLI has no real path, network, collector, or activation option.  It
builds a fully synthetic fixture in a temporary directory, exercises the
candidate/validate/write-once contract there, and never creates a production
activation artifact.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v7_activation_manifest import (
    HUMAN_ACTIVATION_CONFIRMATION,
    SeedProvenanceExpectation,
    V7ActivationManifestBlocked,
    build_activation_manifest_candidate,
    canonical_json_bytes,
    expected_activation_boundary,
    hash_source_payload_manifest,
    read_activation_manifest,
    sha256_bytes,
    validate_activation_manifest_candidate,
    validate_seed_provenance,
    write_activation_manifest_once,
)
from src.v7_jpx_calendar import load_calendar_snapshot
from src.v7_seed_acquisition import validate_universe_file

UNIVERSE_CSV = ROOT / "V4_UNIVERSE.csv"
CALENDAR_PATH = ROOT / "data" / "v7_jpx_calendar_2026_2027.json"

# Synthetic (fake) human Gate 4 values.  These are NOT study decisions.
SYNTHETIC_AUTHORIZATION_UTC = "2026-08-07T09:00:00Z"
SYNTHETIC_SEED_ACQUISITION_UTC = "2026-08-07T03:10:00Z"
SYNTHETIC_ACQUISITION_WINDOW_JST = "17:00-18:00 Asia/Tokyo"
SEED_CUTOFF = "2026-08-07"
SEED_OBSERVATION_COUNT = 252
SEED_CSV_COLUMNS = (
    "ticker", "trading_date", "raw_open", "raw_high", "raw_low",
    "raw_close", "adj_close", "raw_volume",
)


def universe_tickers() -> list[str]:
    return validate_universe_file(UNIVERSE_CSV)["tickers"]


def seed_observation_days(count: int = SEED_OBSERVATION_COUNT) -> list[str]:
    """Business days ending at the seed cutoff; synthetic pre-activation history."""
    days: list[str] = []
    current = date.fromisoformat(SEED_CUTOFF)
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current -= timedelta(days=1)
    return sorted(days)


def synthetic_seed_rows(tickers: Sequence[str], days: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        for index, day in enumerate(days):
            price = 1000.0 + float(index)
            rows.append({
                "ticker": ticker,
                "trading_date": day,
                "raw_open": price,
                "raw_high": price + 2.0,
                "raw_low": price - 2.0,
                "raw_close": price,
                "adj_close": price,
                "raw_volume": 100000.0,
            })
    return rows


def write_synthetic_seed_csv(path: Path, rows: Sequence[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=SEED_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (str(item["ticker"]), str(item["trading_date"]))):
        writer.writerow({column: row[column] for column in SEED_CSV_COLUMNS})
    payload = stream.getvalue().encode("utf-8")
    path.write_bytes(payload)
    return payload


def synthetic_seed_acquisition_manifest(tickers: Sequence[str]) -> dict[str, Any]:
    """A synthetic stand-in for the raw Yahoo acquisition-side provenance manifest."""
    return {
        "mode": "PRE_ACTIVATION_SEED_ACQUISITION",
        "payload_manifest": [
            {
                "ticker": ticker,
                "payload_sha256": sha256_bytes(("synthetic-payload:" + ticker).encode("utf-8")),
                "byte_count": 1024 + index,
            }
            for index, ticker in enumerate(tickers)
        ],
    }


def build_synthetic_fixture(workspace: Path) -> dict[str, Any]:
    """Create every input the contract needs, entirely inside ``workspace``."""
    tickers = universe_tickers()
    days = seed_observation_days()
    rows = synthetic_seed_rows(tickers, days)
    seed_csv = workspace / "seed.csv"
    write_synthetic_seed_csv(seed_csv, rows)
    acquisition_manifest = synthetic_seed_acquisition_manifest(tickers)

    snapshot = load_calendar_snapshot(read_activation_manifest(CALENDAR_PATH))
    boundary = expected_activation_boundary(snapshot, SYNTHETIC_AUTHORIZATION_UTC)

    seed_provenance = validate_seed_provenance(
        universe_csv=UNIVERSE_CSV,
        seed_csv=seed_csv,
        seed_acquisition_manifest=acquisition_manifest,
        activation_boundary_first_jpx_trading_date=boundary,
        expected=None,
    )
    expectation = SeedProvenanceExpectation(
        seed_source_payload_manifest_sha256=seed_provenance["seed_source_payload_manifest_sha256"],
        seed_ticker_manifest_sha256=seed_provenance["seed_ticker_manifest_sha256"],
        seed_canonical_csv_sha256=seed_provenance["seed_canonical_csv_sha256"],
        seed_ticker_count=seed_provenance["seed_ticker_count"],
        seed_row_count=seed_provenance["seed_row_count"],
        seed_cutoff_trading_date=seed_provenance["seed_cutoff_trading_date"],
    )
    output_root = workspace / "study-output"
    output_root.mkdir()
    return {
        "workspace": workspace,
        "tickers": tickers,
        "seed_csv": seed_csv,
        "seed_acquisition_manifest": acquisition_manifest,
        "seed_provenance": seed_provenance,
        "seed_expectation": expectation,
        "activation_boundary": boundary,
        "output_root": str(output_root.resolve()),
        "repository_root": ROOT,
    }


def synthetic_candidate(fixture: dict[str, Any]) -> dict[str, Any]:
    return build_activation_manifest_candidate(
        activation_authorization_utc=SYNTHETIC_AUTHORIZATION_UTC,
        activation_boundary_first_jpx_trading_date=fixture["activation_boundary"],
        acquisition_window_jst=SYNTHETIC_ACQUISITION_WINDOW_JST,
        output_root=fixture["output_root"],
        seed_acquisition_utc=SYNTHETIC_SEED_ACQUISITION_UTC,
        seed_provenance=fixture["seed_provenance"],
    )


def validate_synthetic_candidate(fixture: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return validate_activation_manifest_candidate(
        manifest,
        repository_root=fixture["repository_root"],
        calendar_path=CALENDAR_PATH,
        universe_csv=UNIVERSE_CSV,
        seed_csv=fixture["seed_csv"],
        seed_acquisition_manifest=fixture["seed_acquisition_manifest"],
        expected_seed_provenance=fixture["seed_expectation"],
    )


def write_synthetic_manifest(fixture: dict[str, Any], manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    return write_activation_manifest_once(
        output_path=path,
        manifest=manifest,
        repository_root=fixture["repository_root"],
        confirmation=HUMAN_ACTIVATION_CONFIRMATION,
        calendar_path=CALENDAR_PATH,
        universe_csv=UNIVERSE_CSV,
        seed_csv=fixture["seed_csv"],
        seed_acquisition_manifest=fixture["seed_acquisition_manifest"],
        expected_seed_provenance=fixture["seed_expectation"],
    )


def run_synthetic_activation_contract_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="v7-activation-contract-") as temporary:
        workspace = Path(temporary)
        fixture = build_synthetic_fixture(workspace)

        manifest = synthetic_candidate(fixture)
        validation = validate_synthetic_candidate(fixture, manifest)
        if validation["status"] != "PASS":
            raise AssertionError("CANDIDATE_VALIDATION_NOT_PASS")

        manifest_path = workspace / "activation_manifest.json"
        if manifest_path.exists():
            raise AssertionError("SYNTHETIC_MANIFEST_ALREADY_PRESENT")
        written = write_synthetic_manifest(fixture, manifest, manifest_path)
        if written["status"] != "WRITTEN":
            raise AssertionError("WRITE_ONCE_NOT_WRITTEN")

        read_back = read_activation_manifest(manifest_path)
        if canonical_json_bytes(read_back) != canonical_json_bytes(manifest):
            raise AssertionError("READ_BACK_NOT_BYTE_DETERMINISTIC")
        validate_synthetic_candidate(fixture, read_back)

        duplicate_write_blocked = False
        try:
            write_synthetic_manifest(fixture, manifest, manifest_path)
        except V7ActivationManifestBlocked as error:
            duplicate_write_blocked = error.reason == "ACTIVATION_MANIFEST_ALREADY_EXISTS"
        if not duplicate_write_blocked:
            raise AssertionError("DUPLICATE_WRITE_NOT_BLOCKED")

        tampered = dict(read_back)
        tampered["ticker_count"] = 299
        tamper_detected = False
        try:
            validate_synthetic_candidate(fixture, tampered)
        except V7ActivationManifestBlocked:
            tamper_detected = True

        hash_tampered = dict(read_back)
        hash_tampered["manifest_sha256"] = "0" * 64
        hash_tamper_detected = False
        try:
            validate_synthetic_candidate(fixture, hash_tampered)
        except V7ActivationManifestBlocked as error:
            hash_tamper_detected = error.reason == "MANIFEST_SHA_MISMATCH"

        if not tamper_detected or not hash_tamper_detected:
            raise AssertionError("TAMPER_DETECTION_NOT_PASS")

        remaining_staging = [
            entry.name for entry in workspace.iterdir() if ".staging-" in entry.name
        ]
        if remaining_staging:
            raise AssertionError("STAGING_REMNANT_PRESENT")

    return {
        "status": "PASS",
        "mode": "STATIC_SYNTHETIC_ONLY",
        "candidate_validation": "PASS",
        "manifest_hash_pass": True,
        "write_once_pass": True,
        "duplicate_write_blocked": True,
        "tamper_detection_pass": True,
        "network_requests": 0,
        "collector_enabled": False,
        "forward_processing": 0,
        "actual_activation_created": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V7 Gate 4 activation manifest synthetic-only check")
    parser.add_argument("--synthetic-activation-contract-test", action="store_true", required=True)
    parser.parse_args(argv)
    result = run_synthetic_activation_contract_test()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
