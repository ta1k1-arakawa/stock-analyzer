"""Offline, fail-closed recovery of the frozen V8 partition identity.

The caller supplies source bytes and their parser. This module has no source
acquisition or network path. Real-source execution remains out of scope for
this implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from src import v8_partition as historical


SCHEMA_VERSION = "V8_PARTITION_RECOVERY_MANIFEST_V1"
ORIGINAL_MANIFEST_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
HISTORICAL_IMPLEMENTATION_COMMIT = "36cbed941050e728f7f96ce2af505e81175cc02c"
HISTORICAL_IMPLEMENTATION_SOURCE_SHA256 = "d1f85f25d64628c745c25cf5d6cbaf593a5b28ee09c5befd92ae5ed0ea09da10"
ORIGINAL_SOURCE_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
ORIGINAL_SOURCE_ACQUISITION_UTC = "2026-08-10T03:00:51.733526Z"
ORIGINAL_SOURCE_BYTE_COUNT = 830464
ORIGINAL_SOURCE_SHA256 = "6e401867d9ddf2524e4752f08fd3e3e434cd308c6d423839ca6e24fc7b1e1653"
EXPECTED_ELIGIBLE_COUNT = 3110
EXPECTED_ELIGIBLE_SHA256 = "37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405"
EXPECTED_T0_SHA256 = "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7"
EXPECTED_V4_UNIVERSE_CSV_SHA256 = "d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997"
EXPECTED_BLOCK_SHA256 = MappingProxyType({
    "T1": "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d",
    "T2": "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500",
    "T3": "43a585f4c3341307e7c67561c54780322b0f253fefa628a7c6129773901a7b7a",
    "T_spare": "360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70",
})

MANIFEST_FIELDS = frozenset({
    "schema_version", "original_trusted_manifest_sha256",
    "historical_partition_implementation_commit", "original_source_fingerprint",
    "recovery_source_fingerprint", "v4_provenance_fingerprint",
    "eligible_ticker_count", "eligible_ticker_list_sha256", "t0_reproduction_status",
    "t0_ticker_list_sha256", "trusted_block_ticker_list_sha256",
    "block_sizes", "block_assignments", "recovery_implementation_provenance",
    "recovery_timestamp_utc", "original_manifest_byte_exact_recovered",
    "original_partition_block_identity_recovered", "manifest_sha256",
})


class V8PartitionRecoveryBlocked(RuntimeError):
    """Safe, identity-free failure from a recovery gate."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _TrustPins:
    eligible_count: int = EXPECTED_ELIGIBLE_COUNT
    eligible_sha256: str = EXPECTED_ELIGIBLE_SHA256
    t0_sha256: str = EXPECTED_T0_SHA256
    block_sha256: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.block_sha256 is None:
            object.__setattr__(self, "block_sha256", MappingProxyType(dict(EXPECTED_BLOCK_SHA256)))


_FROZEN_PINS = _TrustPins()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V8PartitionRecoveryBlocked("RECOVERY_MANIFEST_SERIALIZATION_FAILED") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_text(value: datetime) -> str:
    if (not isinstance(value, datetime) or value.tzinfo is None
            or value.utcoffset() != timedelta(0)):
        raise V8PartitionRecoveryBlocked("RECOVERY_TIMESTAMP_INVALID")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _fail(reason: str) -> None:
    raise V8PartitionRecoveryBlocked(reason)


def _build_recovery_manifest(
    *,
    raw_source_bytes: bytes,
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    recovery_source_url: str,
    recovery_source_acquisition_utc: datetime,
    recovery_timestamp_utc: datetime,
    recovery_implementation_commit: str,
    _trust_pins: _TrustPins = _FROZEN_PINS,
) -> dict[str, Any]:
    """Internal seam; `_trust_pins` is only for non-production synthetic tests."""
    try:
        historical.require_git_commit(recovery_implementation_commit)
        historical.require_git_commit(HISTORICAL_IMPLEMENTATION_COMMIT)
        if _sha256(Path(historical.__file__).read_bytes()) != HISTORICAL_IMPLEMENTATION_SOURCE_SHA256:
            _fail("RECOVERY_HISTORICAL_IMPLEMENTATION_SOURCE_MISMATCH")
        if not isinstance(raw_source_bytes, (bytes, bytearray)) or not raw_source_bytes:
            _fail("RECOVERY_SOURCE_BYTES_INVALID")
        if not isinstance(recovery_source_url, str) or not recovery_source_url.strip():
            _fail("RECOVERY_SOURCE_URL_INVALID")

        provenance = historical.load_v4_provenance(v4_manifest_path)
        committed_csv = historical.load_v4_universe_csv_bytes(v4_universe_csv_path)
        if (_sha256(committed_csv) != provenance.get("universe_csv_sha256")
                or provenance.get("universe_csv_sha256") != EXPECTED_V4_UNIVERSE_CSV_SHA256
                or provenance.get("selected_count") != 300
                or provenance.get("eligible_current_only") != 3115):
            _fail("V4_COMMITTED_UNIVERSE_PROVENANCE_MISMATCH")
        if provenance.get("ticker_list_sha256") != _trust_pins.t0_sha256:
            _fail("V4_T0_TRUST_PIN_MISMATCH")

        # This reproduces historical V8's normalization, eligibility,
        # ordering, and exact V4 T0 checks. Fresh allocation has not occurred.
        frame = parse_source_table(bytes(raw_source_bytes))
        eligible_rows, _excluded_counts = historical.parse_eligible_universe(frame)
        if not eligible_rows:
            _fail("RECOVERY_ELIGIBLE_UNIVERSE_EMPTY")
        ordered_codes = historical.canonical_order([row["code"] for row in eligible_rows])
        rows_by_code = {row["code"]: row for row in eligible_rows}
        if len(rows_by_code) != len(eligible_rows):
            _fail("RECOVERY_ELIGIBLE_UNIVERSE_DUPLICATE")
        ordered_rows = [rows_by_code[code] for code in ordered_codes]
        t0 = historical.verify_t0_reproduction(ordered_rows, provenance)
        t0_sha256 = historical.ticker_list_sha256(t0)
        if t0_sha256 != _trust_pins.t0_sha256:
            _fail("RECOVERY_T0_TRUST_PIN_MISMATCH")

        # These frozen universe gates must pass before allocate_fresh_blocks.
        eligible_sha256 = historical.ticker_list_sha256(ordered_codes)
        if len(ordered_codes) != _trust_pins.eligible_count:
            _fail("RECOVERY_ELIGIBLE_COUNT_MISMATCH")
        if eligible_sha256 != _trust_pins.eligible_sha256:
            _fail("RECOVERY_ELIGIBLE_UNIVERSE_HASH_MISMATCH")

        blocks = historical.allocate_fresh_blocks(ordered_codes, t0)
        block_hashes = {name: historical.ticker_list_sha256(blocks[name])
                        for name in ("T1", "T2", "T3", "T_spare")}
        for name, expected in _trust_pins.block_sha256.items():
            if block_hashes.get(name) != expected:
                _fail("RECOVERY_" + name.upper() + "_HASH_MISMATCH")
        if set(block_hashes) != set(EXPECTED_BLOCK_SHA256):
            _fail("RECOVERY_BLOCK_HASH_SCHEMA_MISMATCH")

        assigned = [ticker for name in ("T0", "T1", "T2", "T3", "T_spare")
                    for ticker in blocks[name]]
        if len(assigned) != len(set(assigned)):
            _fail("RECOVERY_BLOCK_OVERLAP")
        if any(len(blocks[name]) != historical.BLOCK_SIZE for name in ("T1", "T2", "T3")):
            _fail("RECOVERY_BLOCK_SIZE_MISMATCH")
        if set(historical.LEGACY_EXPOSED_TICKERS_OUTSIDE_T0).intersection(
                ticker for name in ("T1", "T2", "T3", "T_spare") for ticker in blocks[name]):
            _fail("RECOVERY_LEGACY_EXCLUSION_MISMATCH")

        acquired_at = _utc_text(recovery_source_acquisition_utc)
        recovered_at = _utc_text(recovery_timestamp_utc)
        module_bytes = Path(__file__).read_bytes()
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "original_trusted_manifest_sha256": ORIGINAL_MANIFEST_SHA256,
            "historical_partition_implementation_commit": HISTORICAL_IMPLEMENTATION_COMMIT,
            "original_source_fingerprint": {
                "source_url": ORIGINAL_SOURCE_URL,
                "source_acquisition_utc": ORIGINAL_SOURCE_ACQUISITION_UTC,
                "raw_byte_count": ORIGINAL_SOURCE_BYTE_COUNT,
                "raw_sha256": ORIGINAL_SOURCE_SHA256,
            },
            "recovery_source_fingerprint": {
                "source_url": str(recovery_source_url),
                "source_acquisition_utc": acquired_at,
                "raw_byte_count": len(raw_source_bytes),
                "raw_sha256": _sha256(bytes(raw_source_bytes)),
            },
            "v4_provenance_fingerprint": {
                "manifest_raw_sha256": _sha256(
                    Path(v4_manifest_path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                ),
                "universe_csv_sha256": _sha256(committed_csv),
            },
            "eligible_ticker_count": len(ordered_codes),
            "eligible_ticker_list_sha256": eligible_sha256,
            "t0_reproduction_status": "PASS",
            "t0_ticker_list_sha256": t0_sha256,
            "trusted_block_ticker_list_sha256": dict(_trust_pins.block_sha256),
            "block_sizes": {name: len(members) for name, members in blocks.items()},
            "block_assignments": blocks,
            "recovery_implementation_provenance": {
                "implementation_git_commit": recovery_implementation_commit,
                "implementation_source_sha256": _sha256(module_bytes),
                "historical_partition_source_git_commit": HISTORICAL_IMPLEMENTATION_COMMIT,
            },
            "recovery_timestamp_utc": recovered_at,
            "original_manifest_byte_exact_recovered": False,
            "original_partition_block_identity_recovered": True,
        }
        if set(manifest) != MANIFEST_FIELDS - {"manifest_sha256"}:
            _fail("RECOVERY_MANIFEST_SCHEMA_MISMATCH")
        manifest["manifest_sha256"] = _sha256(_canonical_json_bytes(manifest))
        return manifest
    except V8PartitionRecoveryBlocked:
        raise
    except historical.V8PartitionBlocked as error:
        _fail("RECOVERY_HISTORICAL_GATE_FAILED")
    except Exception as error:
        # Do not surface parser exception text: it may contain source data.
        raise V8PartitionRecoveryBlocked("RECOVERY_INPUT_VALIDATION_FAILED") from error


def build_v8_partition_recovery_manifest(
    *,
    raw_source_bytes: bytes,
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    recovery_source_url: str,
    recovery_source_acquisition_utc: datetime,
    recovery_timestamp_utc: datetime,
    recovery_implementation_commit: str,
) -> dict[str, Any]:
    """Build an accepted, private-use manifest from explicitly supplied bytes."""
    return _build_recovery_manifest(
        raw_source_bytes=raw_source_bytes,
        parse_source_table=parse_source_table,
        v4_manifest_path=v4_manifest_path,
        v4_universe_csv_path=v4_universe_csv_path,
        recovery_source_url=recovery_source_url,
        recovery_source_acquisition_utc=recovery_source_acquisition_utc,
        recovery_timestamp_utc=recovery_timestamp_utc,
        recovery_implementation_commit=recovery_implementation_commit,
    )


def _validate_accepted_manifest(manifest: Mapping[str, Any]) -> bytes:
    if not isinstance(manifest, Mapping) or set(manifest) != MANIFEST_FIELDS:
        raise V8PartitionRecoveryBlocked("RECOVERY_MANIFEST_SCHEMA_INVALID")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise V8PartitionRecoveryBlocked("RECOVERY_MANIFEST_SCHEMA_INVALID")
    if manifest.get("original_manifest_byte_exact_recovered") is not False:
        raise V8PartitionRecoveryBlocked("RECOVERY_BYTE_EXACT_CLAIM_PROHIBITED")
    if manifest.get("original_partition_block_identity_recovered") is not True:
        raise V8PartitionRecoveryBlocked("RECOVERY_IDENTITY_GATE_NOT_PASSED")
    if (manifest.get("original_trusted_manifest_sha256") != ORIGINAL_MANIFEST_SHA256
            or manifest.get("historical_partition_implementation_commit") != HISTORICAL_IMPLEMENTATION_COMMIT
            or manifest.get("eligible_ticker_count") != EXPECTED_ELIGIBLE_COUNT
            or manifest.get("eligible_ticker_list_sha256") != EXPECTED_ELIGIBLE_SHA256
            or manifest.get("t0_reproduction_status") != "PASS"
            or manifest.get("t0_ticker_list_sha256") != EXPECTED_T0_SHA256):
        raise V8PartitionRecoveryBlocked("RECOVERY_PROVENANCE_GATE_NOT_PASSED")
    trusted_hashes = manifest.get("trusted_block_ticker_list_sha256")
    if not isinstance(trusted_hashes, Mapping) or set(trusted_hashes) != set(EXPECTED_BLOCK_SHA256):
        raise V8PartitionRecoveryBlocked("RECOVERY_TRUST_PIN_SCHEMA_INVALID")
    assignments = manifest.get("block_assignments")
    sizes = manifest.get("block_sizes")
    if (not isinstance(assignments, Mapping)
            or set(assignments) != {"T0", "T1", "T2", "T3", "T_spare"}
            or not isinstance(sizes, Mapping)
            or set(sizes) != set(assignments)):
        raise V8PartitionRecoveryBlocked("RECOVERY_BLOCK_ASSIGNMENT_SCHEMA_INVALID")
    flattened: list[str] = []
    for name, members in assignments.items():
        if (not isinstance(members, list) or any(not isinstance(value, str) for value in members)
                or sizes.get(name) != len(members)):
            raise V8PartitionRecoveryBlocked("RECOVERY_BLOCK_ASSIGNMENT_SCHEMA_INVALID")
        flattened.extend(members)
    if len(flattened) != len(set(flattened)):
        raise V8PartitionRecoveryBlocked("RECOVERY_BLOCK_OVERLAP")
    if (len(assignments["T0"]) != historical.BLOCK_SIZE
            or any(len(assignments[name]) != historical.BLOCK_SIZE for name in ("T1", "T2", "T3"))
            or historical.ticker_list_sha256(assignments["T0"]) != EXPECTED_T0_SHA256):
        raise V8PartitionRecoveryBlocked("RECOVERY_BLOCK_SIZE_OR_T0_MISMATCH")
    legacy = set(historical.LEGACY_EXPOSED_TICKERS_OUTSIDE_T0)
    if any(legacy.intersection(assignments[name]) for name in ("T1", "T2", "T3", "T_spare")):
        raise V8PartitionRecoveryBlocked("RECOVERY_LEGACY_EXCLUSION_MISMATCH")
    for name, expected in trusted_hashes.items():
        if (expected != EXPECTED_BLOCK_SHA256[name]
                or historical.ticker_list_sha256(assignments.get(name, [])) != expected):
            raise V8PartitionRecoveryBlocked("RECOVERY_BLOCK_HASH_MISMATCH")
    if (manifest.get("original_source_fingerprint") != {
            "source_url": ORIGINAL_SOURCE_URL,
            "source_acquisition_utc": ORIGINAL_SOURCE_ACQUISITION_UTC,
            "raw_byte_count": ORIGINAL_SOURCE_BYTE_COUNT,
            "raw_sha256": ORIGINAL_SOURCE_SHA256,
    }):
        raise V8PartitionRecoveryBlocked("RECOVERY_ORIGINAL_SOURCE_PROVENANCE_MISMATCH")
    source_fp = manifest.get("recovery_source_fingerprint")
    if (not isinstance(source_fp, Mapping)
            or not isinstance(source_fp.get("raw_sha256"), str)
            or len(source_fp["raw_sha256"]) != 64
            or not isinstance(source_fp.get("raw_byte_count"), int)
            or source_fp["raw_byte_count"] <= 0):
        raise V8PartitionRecoveryBlocked("RECOVERY_SOURCE_PROVENANCE_INVALID")
    impl = manifest.get("recovery_implementation_provenance")
    if (not isinstance(impl, Mapping)
            or impl.get("historical_partition_source_git_commit") != HISTORICAL_IMPLEMENTATION_COMMIT
            or not isinstance(impl.get("implementation_git_commit"), str)
            or len(impl["implementation_git_commit"]) != 40
            or not isinstance(impl.get("implementation_source_sha256"), str)
            or len(impl["implementation_source_sha256"]) != 64):
        raise V8PartitionRecoveryBlocked("RECOVERY_IMPLEMENTATION_PROVENANCE_INVALID")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _sha256(_canonical_json_bytes(unsigned)):
        raise V8PartitionRecoveryBlocked("RECOVERY_MANIFEST_HASH_MISMATCH")
    return _canonical_json_bytes(dict(manifest))


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_v8_partition_recovery_manifest_once(
    manifest: Mapping[str, Any],
    output_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
) -> Path:
    """Atomically publish only a fully accepted artifact, without overwrite."""
    try:
        output = historical.require_absolute_output_path_outside_repository(output_path, repository_root)
    except historical.V8PartitionBlocked as error:
        raise V8PartitionRecoveryBlocked("RECOVERY_OUTPUT_PATH_INVALID") from error
    try:
        payload = _validate_accepted_manifest(manifest)
    except V8PartitionRecoveryBlocked:
        raise
    if output.exists():
        raise V8PartitionRecoveryBlocked("RECOVERY_ARTIFACT_ALREADY_EXISTS")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        if not output.parent.is_dir():
            raise OSError("parent not a directory")
        staging = output.parent / (output.name + ".staging-" + os.urandom(8).hex())
        try:
            with open(staging, "xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(staging, output)
            except FileExistsError as error:
                raise V8PartitionRecoveryBlocked("RECOVERY_ARTIFACT_ALREADY_EXISTS") from error
            except OSError as error:
                raise V8PartitionRecoveryBlocked("RECOVERY_ARTIFACT_ATOMIC_PUBLISH_FAILED") from error
            _fsync_directory(output.parent)
        finally:
            if staging.exists():
                staging.unlink()
    except V8PartitionRecoveryBlocked:
        raise
    except OSError as error:
        raise V8PartitionRecoveryBlocked("RECOVERY_ARTIFACT_WRITE_FAILED") from error
    return output


def safe_recovery_status(
    *,
    accepted: bool,
    eligible_ticker_count: int | None = None,
    eligible_ticker_list_sha256: str | None = None,
    block_hashes: Mapping[str, str] | None = None,
    network_requests: int = 0,
) -> dict[str, Any]:
    """Identity-free terminal summary for logs and public-safe reports."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ACCEPTED" if accepted else "BLOCKED",
        "eligible_ticker_count": eligible_ticker_count,
        "eligible_ticker_list_sha256": eligible_ticker_list_sha256,
        "trusted_block_ticker_list_sha256": dict(EXPECTED_BLOCK_SHA256),
        "original_manifest_byte_exact_recovered": False,
        "original_partition_block_identity_recovered": bool(accepted),
        "network_requests": network_requests,
        "sealed_identity_values_included": False,
        "block_hashes": dict(block_hashes or {}),
    }


def recover_and_publish_v8_partition_once(
    *,
    raw_source_bytes: bytes,
    parse_source_table: Callable[[bytes], Any],
    v4_manifest_path: str | os.PathLike[str],
    v4_universe_csv_path: str | os.PathLike[str],
    recovery_source_url: str,
    recovery_source_acquisition_utc: datetime,
    recovery_timestamp_utc: datetime,
    recovery_implementation_commit: str,
    output_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build every gate in memory, then publish only the complete artifact."""
    manifest = build_v8_partition_recovery_manifest(
        raw_source_bytes=raw_source_bytes,
        parse_source_table=parse_source_table,
        v4_manifest_path=v4_manifest_path,
        v4_universe_csv_path=v4_universe_csv_path,
        recovery_source_url=recovery_source_url,
        recovery_source_acquisition_utc=recovery_source_acquisition_utc,
        recovery_timestamp_utc=recovery_timestamp_utc,
        recovery_implementation_commit=recovery_implementation_commit,
    )
    write_v8_partition_recovery_manifest_once(manifest, output_path, repository_root)
    return safe_recovery_status(
        accepted=True,
        eligible_ticker_count=manifest["eligible_ticker_count"],
        eligible_ticker_list_sha256=manifest["eligible_ticker_list_sha256"],
        block_hashes=manifest["trusted_block_ticker_list_sha256"],
        network_requests=0,
    )


__all__ = [
    "EXPECTED_BLOCK_SHA256", "EXPECTED_ELIGIBLE_COUNT", "EXPECTED_ELIGIBLE_SHA256",
    "HISTORICAL_IMPLEMENTATION_COMMIT", "ORIGINAL_MANIFEST_SHA256", "SCHEMA_VERSION",
    "V8PartitionRecoveryBlocked", "build_v8_partition_recovery_manifest",
    "recover_and_publish_v8_partition_once", "safe_recovery_status",
    "write_v8_partition_recovery_manifest_once",
]
