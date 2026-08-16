"""Fixed V8D production raw-acquisition orchestration.

The public functions in :mod:`src.v8d_historical_acquisition` deliberately
have no request, authority, timing, or repository injection seams.  This
module contains the production-only boundary and the private bundle writer;
the existing V8D transport stage remains the sole retry/audit engine.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.v8c_t1c_allocation import read_t1c_allocation_artifact_bytes, ticker_list_sha256
from src.v8d_authority_bridge import T1C_STAGE as T1C_READINESS_STAGE
from src.v8d_authority_bridge import T2_STAGE as T2_READINESS_STAGE
from src.v8d_authority_bridge import verify_stage_authority_bridge
from src.v8d_git_provenance import CANONICAL_REPOSITORY_ROOT, resolve_verified_v8d_production_git_commit
from src.v8d_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_T1C_RAW_ACQUISITION,
    GATE_T2_RAW_ACQUISITION,
    consume_gate_and_bind,
    require_gate_not_yet_consumed,
)
from src.v8d_production_provenance import (
    EXPECTED_V8D_FROZEN_DESIGN_COMMIT,
    EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
    V8_DESIGN_COMMIT,
    read_and_verify_v8_trusted_partition_anchor,
    verify_design_freeze_approval_blob,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8d_readiness_audit_verification import (
    require_t1c_readiness_audit_verification_pass,
    require_t2_readiness_audit_verification_pass,
)
from src.v8d_t2_point_of_use_preservation import require_t2_point_of_use_preservation_review_pass
from src.v8d_transport import (
    CANONICAL_PARSER_CLASSIFIER_BLOB,
    CANONICAL_PARSER_CLASSIFIER_COMMIT,
    DurableV8DAuditStore,
    V8DNamedFailure,
    V8DRequestPlan,
    build_yahoo_request_plan,
    canonical_json_bytes,
    canonical_sha256,
    default_trusted_yahoo_opener,
    execute_v8d_stage,
)
from src.v8_partition import BLOCK_SIZE, read_partition_manifest, require_absolute_output_path_outside_repository
from src.v7_yahoo_collector import FRAME_FIELDS


REQUEST_START = "2016-04-01"
REQUEST_END_EXCLUSIVE = "2026-01-01"
REQUEST_COUNT = 300
DQ_POLICY_NAME = "POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE"
DQ_INVALID_NUMERATOR = 1
DQ_INVALID_DENOMINATOR = 252
DQ_MAX_CONSECUTIVE_INVALID = 1
DQ_FULL_P_HIST = True
DQ_TEST_YEARS = tuple(range(2018, 2026))
DQ_CALENDAR_MISSING_IS_MALFORMED = False
DQ_THRESHOLD_ACTION = "BLOCK_WHOLE_ACQUISITION"
T1C_LIST_SHA256 = "85a06d4b88698915315f5cf72e0d3e04dfacafb5403786ac3bb613e14b0deb54"
T1C_PARENT_SPARE_SHA256 = "360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70"
T2_LIST_SHA256 = "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500"
T1C_ALLOCATION_SELF_HASH = "16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c"
T1C_SLICE_START = 300
T1C_SLICE_END = 600

ACQUISITION_MANIFEST_SCHEMA = "V8D_RAW_ACQUISITION_BUNDLE_V1"
PRODUCTION_BINDING_SCHEMA = "V8D_PRODUCTION_RAW_ACQUISITION_EXECUTION_BINDING_V1"
PRODUCTION_BINDING_ROOT = CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8d-acquisition-production-execution-state"
PRODUCTION_BINDING_FILENAMES = frozenset({
    "t1c-raw-acquisition-execution-binding.json",
    "t2-raw-acquisition-execution-binding.json",
})


class V8DAcquisitionEngineBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _block(reason: str, error: BaseException | None = None) -> None:
    if error is None:
        raise V8DAcquisitionEngineBlocked(reason)
    raise V8DAcquisitionEngineBlocked(reason) from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_private_file(path_value: str | os.PathLike[str], label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute() or path.is_symlink():
        _block(f"V8D_ACQUISITION_{label}_LOCATOR_INVALID")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            _block(f"V8D_ACQUISITION_{label}_LOCATOR_INVALID")
        return resolved
    except OSError as error:
        _block(f"V8D_ACQUISITION_{label}_LOCATOR_INVALID", error)


def _private_tickers(manifest: Mapping[str, Any], block: str) -> tuple[str, ...]:
    assignments = manifest.get("block_assignments")
    if not isinstance(assignments, Mapping) or block not in assignments:
        _block("V8D_ACQUISITION_PARTITION_ASSIGNMENT_MISSING")
    values = assignments[block]
    if not isinstance(values, list) or len(values) != REQUEST_COUNT or any(not isinstance(v, str) or not v for v in values):
        _block("V8D_ACQUISITION_PARTITION_COUNT_INVALID")
    if len(set(values)) != len(values):
        _block("V8D_ACQUISITION_PARTITION_DUPLICATE_TICKER")
    expected = T1C_LIST_SHA256 if block == "T1C" else T2_LIST_SHA256
    field = "t1c_ticker_list_sha256" if block == "T1C" else "t2_ticker_list_sha256"
    if ticker_list_sha256(values) != expected or manifest.get(field) != expected:
        _block("V8D_ACQUISITION_PARTITION_LIST_HASH_INVALID")
    return tuple(values)


def _read_and_validate_partition(path: Path, block: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        manifest = read_partition_manifest(path)
    except Exception as error:  # noqa: BLE001 - private evidence is fail-closed
        _block("V8D_ACQUISITION_PARTITION_MANIFEST_INVALID", error)
    if manifest.get("manifest_sha256") != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        _block("V8D_ACQUISITION_PARTITION_MANIFEST_SHA_MISMATCH")
    if manifest.get("study_name") != "V8_HISTORICAL_RESEARCH" or manifest.get("design_commit") != V8_DESIGN_COMMIT:
        _block("V8D_ACQUISITION_PARTITION_DESIGN_MISMATCH")
    if manifest.get("partition_implementation_git_commit") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        _block("V8D_ACQUISITION_PARTITION_IMPLEMENTATION_MISMATCH")
    size_key = "T_spare" if block == "T1C" else block
    if manifest.get("block_sizes", {}).get(size_key) != (1904 if block == "T1C" else BLOCK_SIZE):
        _block("V8D_ACQUISITION_PARTITION_BLOCK_SIZE_INVALID")
    if block == "T1C":
        spare = manifest.get("block_assignments", {}).get("T_spare")
        if not isinstance(spare, list) or len(spare) != 1904 or ticker_list_sha256(spare) != T1C_PARENT_SPARE_SHA256 or manifest.get("t_spare_ticker_list_sha256") != T1C_PARENT_SPARE_SHA256:
            _block("V8D_ACQUISITION_PARENT_SPARE_HASH_INVALID")
        return manifest, ()
    return manifest, _private_tickers(manifest, block)


def _validate_t1c_allocation(path: Path, manifest: Mapping[str, Any]) -> tuple[str, ...]:
    try:
        artifact = read_t1c_allocation_artifact_bytes(path.read_bytes())
    except Exception as error:  # noqa: BLE001
        _block("V8D_ACQUISITION_T1C_ALLOCATION_INVALID", error)
    if artifact.get("artifact_self_hash") != T1C_ALLOCATION_SELF_HASH:
        _block("V8D_ACQUISITION_T1C_ALLOCATION_SELF_HASH_MISMATCH")
    if artifact.get("parent_v8_partition_manifest_sha256") != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        _block("V8D_ACQUISITION_T1C_ALLOCATION_MANIFEST_MISMATCH")
    if artifact.get("parent_v8_partition_implementation_commit") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        _block("V8D_ACQUISITION_T1C_ALLOCATION_IMPLEMENTATION_MISMATCH")
    spare = manifest.get("block_assignments", {}).get("T_spare")
    if not isinstance(spare, list) or ticker_list_sha256(spare) != T1C_PARENT_SPARE_SHA256 or manifest.get("t_spare_ticker_list_sha256") != T1C_PARENT_SPARE_SHA256:
        _block("V8D_ACQUISITION_PARENT_SPARE_HASH_INVALID")
    expected = spare[T1C_SLICE_START:T1C_SLICE_END]
    remaining = spare[T1C_SLICE_END:]
    if artifact.get("t1c_tickers") != expected or artifact.get("remaining_t_spare_tickers") != remaining:
        _block("V8D_ACQUISITION_T1C_MEMBERSHIP_MISMATCH")
    if artifact.get("t1c_ticker_list_sha256") != T1C_LIST_SHA256 or len(expected) != REQUEST_COUNT:
        _block("V8D_ACQUISITION_T1C_LIST_HASH_INVALID")
    return tuple(expected)


def _dq_metadata() -> dict[str, Any]:
    return {
        "policy_name": DQ_POLICY_NAME,
        "invalid_fraction_numerator": DQ_INVALID_NUMERATOR,
        "invalid_fraction_denominator": DQ_INVALID_DENOMINATOR,
        "max_consecutive_invalid_returned_rows": DQ_MAX_CONSECUTIVE_INVALID,
        "full_p_hist_check_required": DQ_FULL_P_HIST,
        "test_years": list(DQ_TEST_YEARS),
        "expected_calendar_missing_dates_treated_as_malformed": DQ_CALENDAR_MISSING_IS_MALFORMED,
        "threshold_exceedance_action": DQ_THRESHOLD_ACTION,
    }


def _require_dq(parsed: Mapping[str, Any]) -> None:
    valid = parsed.get("valid_price_rows")
    invalid = parsed.get("invalid_price_rows")
    if not isinstance(valid, list) or not isinstance(invalid, list) or not valid:
        raise V8DNamedFailure("DATA_QUALITY_GATE_FAILURE", evidence={
            "nonempty_timestamp": False,
            "valid_price_row_count": len(valid) if isinstance(valid, list) else 0,
            "trading_date_fields_valid": False,
        })
    observations: list[tuple[str, bool]] = []
    for row in valid:
        if not isinstance(row, Mapping) or not isinstance(row.get("trading_date"), str):
            raise V8DNamedFailure("DATA_QUALITY_GATE_FAILURE", evidence={
                "nonempty_timestamp": bool(valid),
                "valid_price_row_count": len(valid),
                "trading_date_fields_valid": False,
            })
        observations.append((row["trading_date"], True))
    for row in invalid:
        if not isinstance(row, Mapping) or not isinstance(row.get("trading_date"), str):
            raise V8DNamedFailure("DATA_QUALITY_GATE_FAILURE", evidence={
                "nonempty_timestamp": bool(valid),
                "valid_price_row_count": len(valid),
                "trading_date_fields_valid": False,
            })
        observations.append((row["trading_date"], False))
    observations.sort()
    for window in (observations, *([x for x in observations if x[0].startswith(f"{year}-")] for year in DQ_TEST_YEARS)):
        total = len(window)
        bad = sum(not ok for _, ok in window)
        if bad * DQ_INVALID_DENOMINATOR > total * DQ_INVALID_NUMERATOR:
            raise V8DNamedFailure("DATA_QUALITY_GATE_FAILURE", evidence={
                "nonempty_timestamp": True,
                "valid_price_row_count": len(valid),
                "trading_date_fields_valid": True,
            })
        run = 0
        for _, ok in window:
            run = 0 if ok else run + 1
            if run > DQ_MAX_CONSECUTIVE_INVALID:
                raise V8DNamedFailure("DATA_QUALITY_GATE_FAILURE", evidence={
                    "nonempty_timestamp": True,
                    "valid_price_row_count": len(valid),
                    "trading_date_fields_valid": True,
                })


class _RecordingResponse:
    def __init__(self, response: Any, capture: bytearray) -> None:
        self._response = response
        self._capture = capture

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        value = self._response.read(*args, **kwargs)
        if isinstance(value, bytes):
            self._capture.extend(value)
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


def _canonical_rows(parsed_values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted([{field: row[field] for field in FRAME_FIELDS} for row in parsed_values], key=lambda row: (row["ticker"], row["trading_date"]))


def _exclusive_publish_file(destination: Path, payload: bytes) -> None:
    destination = Path(destination)
    try:
        canonical_destination = destination.resolve(strict=False)
        canonical_root = PRODUCTION_BINDING_ROOT.resolve(strict=False)
    except OSError as error:
        _block("V8D_ACQUISITION_PUBLICATION_PATH_INVALID", error)
    if canonical_destination.parent == canonical_root and canonical_destination.name in PRODUCTION_BINDING_FILENAMES:
        _block("V8D_ACQUISITION_PRODUCTION_BINDING_PUBLICATION_SEAM_BLOCKED")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f"{destination.name}.staging-{os.urandom(8).hex()}"
    try:
        with open(staging, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(str(staging), str(destination))
        except FileExistsError as error:
            _block("V8D_ACQUISITION_BUNDLE_ALREADY_EXISTS", error)
        except OSError as error:
            _block("V8D_ACQUISITION_BUNDLE_ATOMIC_PUBLISH_FAILED", error)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _build_bundle(
    *, stage: str, block: str, output_root: Path, staging_root: Path, tickers: tuple[str, ...], captured: Mapping[int, tuple[bytes, Mapping[str, Any]]],
    execution: Mapping[str, Any], reviewed_commit: str, gate: Mapping[str, Any], membership_hash: str,
) -> dict[str, Any]:
    if execution["aggregate"]["result"] != "PASS" or len(captured) != REQUEST_COUNT:
        _block("V8D_ACQUISITION_TRANSPORT_BLOCK")
    raw_dir = staging_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    payload_manifest: list[dict[str, Any]] = []
    all_rows: list[Mapping[str, Any]] = []
    all_splits: list[Mapping[str, Any]] = []
    for coordinate, ticker in enumerate(tickers):
        raw_bytes, parsed = captured[coordinate]
        raw_path = raw_dir / f"payload-{coordinate:04d}.bin"
        with open(raw_path, "xb") as stream:
            stream.write(raw_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        rows = parsed.get("valid_price_rows", [])
        splits = parsed.get("canonical_split_events", [])
        all_rows.extend(rows)
        all_splits.extend(splits)
        payload_manifest.append({
            "logical_coordinate": coordinate,
            "ticker": ticker,
            "raw_filename": raw_path.name,
            "payload_sha256": _sha256_bytes(raw_bytes),
            "payload_byte_count": len(raw_bytes),
            "valid_price_row_count": len(rows),
            "invalid_price_row_count": len(parsed.get("invalid_price_rows", [])),
            "canonical_price_rows_sha256": canonical_sha256(rows),
            "canonical_split_events_sha256": canonical_sha256(splits),
        })
    payload_manifest_bytes = canonical_json_bytes(payload_manifest)
    canonical_rows = _canonical_rows(all_rows)
    canonical_splits = sorted([dict(item) for item in all_splits], key=lambda item: canonical_json_bytes(item))
    manifest: dict[str, Any] = {
        "schema_version": ACQUISITION_MANIFEST_SCHEMA,
        "study": "V8D_HISTORICAL_RESEARCH",
        "logical_stage": stage,
        "logical_block": block,
        "frozen_design_commit": EXPECTED_V8D_FROZEN_DESIGN_COMMIT,
        "reviewed_production_implementation_commit": reviewed_commit,
        "v8_trust_anchor_git_blob": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "partition_implementation_git_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "gate_receipt_key_sha256": gate["gate_receipt_key_sha256"],
        "gate_receipt_bytes_sha256": gate["gate_receipt_bytes_sha256"],
        "authorization_identity_sha256": gate["authorization_identity_sha256"],
        "membership_count": REQUEST_COUNT,
        "membership_list_sha256": membership_hash,
        "request_start": REQUEST_START,
        "request_end_exclusive": REQUEST_END_EXCLUSIVE,
        "request_count": REQUEST_COUNT,
        "provider": "Yahoo Chart",
        "host": "query1.finance.yahoo.com",
        "parser": "V7_CANONICAL_YAHOO_CHART",
        "canonical_parser_classifier_commit": CANONICAL_PARSER_CLASSIFIER_COMMIT,
        "canonical_parser_classifier_blob": CANONICAL_PARSER_CLASSIFIER_BLOB,
        "dq_policy": _dq_metadata(),
        "transport_aggregate_filename": Path(execution["aggregate_path"]).name,
        "transport_aggregate_self_hash": execution["aggregate"]["aggregate_self_hash"],
        "transport_dossier_count": len(execution["dossier_paths"]),
        "transport_total_attempts": execution["aggregate"]["total_request_attempts"],
        "transport_retry_count": execution["aggregate"]["retry_count"],
        "payload_manifest": payload_manifest,
        "payload_manifest_sha256": _sha256_bytes(payload_manifest_bytes),
        "canonical_price_rows_sha256": canonical_sha256(canonical_rows),
        "canonical_split_events_sha256": canonical_sha256(canonical_splits),
        "valid_price_row_count": len(canonical_rows),
        "invalid_price_row_count": sum(item["invalid_price_row_count"] for item in payload_manifest),
        "T2_raw_acquired_sealed": block == "T2",
        "T1C_raw_acquired_not_opened": block == "T1C",
        "research_access_count": 0,
        "features_observed": False,
        "outcomes_observed": False,
        "result": "PASS",
    }
    manifest["manifest_self_hash"] = canonical_sha256(manifest)
    _exclusive_publish_file(staging_root / "V8D_ACQUISITION_MANIFEST.json", canonical_json_bytes(manifest))
    output_root.mkdir(parents=True, exist_ok=True)
    final_root = output_root / ("T1C_RAW_ACQUISITION" if block == "T1C" else "T2_RAW_ACQUISITION")
    if final_root.exists() or final_root.is_symlink():
        _block("V8D_ACQUISITION_BUNDLE_ALREADY_EXISTS")
    try:
        os.rename(str(staging_root), str(final_root))
    except FileExistsError as error:
        _block("V8D_ACQUISITION_BUNDLE_ALREADY_EXISTS", error)
    except OSError as error:
        _block("V8D_ACQUISITION_BUNDLE_ATOMIC_PUBLISH_FAILED", error)
    return {key: value for key, value in manifest.items() if key != "payload_manifest"}


def _execute_fixed_production_acquisition(
    *, stage: str, human_authorization_identity: str, partition_manifest_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str], t1c_allocation_artifact_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    if stage not in {"T1C_RAW_ACQUISITION", "T2_RAW_ACQUISITION"}:
        _block("V8D_ACQUISITION_STAGE_INVALID")
    if not isinstance(human_authorization_identity, str) or not human_authorization_identity:
        _block("V8D_ACQUISITION_HUMAN_AUTHORIZATION_IDENTITY_REQUIRED")
    staging: Path | None = None
    try:
        verified_head = resolve_verified_v8d_production_git_commit(CANONICAL_REPOSITORY_ROOT)
        verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT)
        verify_design_freeze_approval_blob(CANONICAL_REPOSITORY_ROOT, verified_head)
        binding = verify_reviewed_implementation_binding(CANONICAL_REPOSITORY_ROOT, verified_head)
        reviewed_commit = binding["reviewed_implementation_git_commit"]
        verify_stage_authority_bridge(CANONICAL_REPOSITORY_ROOT, verified_head, T1C_READINESS_STAGE if stage.startswith("T1C") else T2_READINESS_STAGE)
        if stage.startswith("T1C"):
            require_t1c_readiness_audit_verification_pass()
        else:
            require_t2_readiness_audit_verification_pass()
            require_t2_point_of_use_preservation_review_pass()
        gate_name = GATE_T1C_RAW_ACQUISITION if stage.startswith("T1C") else GATE_T2_RAW_ACQUISITION
        require_gate_not_yet_consumed(CANONICAL_CONSUMPTION_STATE_ROOT, gate_name, EXPECTED_V8D_FROZEN_DESIGN_COMMIT)
        anchor = read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, verified_head)
        if anchor.get("authorized_partition_manifest_sha256") != EXPECTED_V8_PARTITION_MANIFEST_SHA256 or anchor.get("authorized_partition_implementation_git_commit") != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
            _block("V8D_ACQUISITION_TRUST_ANCHOR_MISMATCH")
        out = require_absolute_output_path_outside_repository(output_root, CANONICAL_REPOSITORY_ROOT)
        partition_path = _strict_private_file(partition_manifest_path, "PARTITION_MANIFEST")
        manifest, tickers = _read_and_validate_partition(partition_path, "T1C" if stage.startswith("T1C") else "T2")
        if stage.startswith("T1C"):
            if t1c_allocation_artifact_path is None:
                _block("V8D_ACQUISITION_T1C_ALLOCATION_REQUIRED")
            tickers = _validate_t1c_allocation(_strict_private_file(t1c_allocation_artifact_path, "T1C_ALLOCATION"), manifest)
        out.mkdir(parents=True, exist_ok=True)
        final_root = out / ("T1C_RAW_ACQUISITION" if stage.startswith("T1C") else "T2_RAW_ACQUISITION")
        if final_root.exists() or final_root.is_symlink():
            _block("V8D_ACQUISITION_BUNDLE_ALREADY_EXISTS")
        staging = Path(tempfile.mkdtemp(prefix=final_root.name + ".staging-", dir=str(out)))
        captured: dict[int, tuple[bytes, Mapping[str, Any]]] = {}
        plans: list[V8DRequestPlan] = []
        for coordinate, ticker in enumerate(tickers):
            capture = bytearray()
            def opener(request: Any, capture: bytearray = capture) -> Any:
                return _RecordingResponse(default_trusted_yahoo_opener(request), capture)
            def validate(parsed: Mapping[str, Any], coordinate: int = coordinate, capture: bytearray = capture) -> None:
                _require_dq(parsed)
                captured[coordinate] = (bytes(capture), parsed)
            plans.append(build_yahoo_request_plan(logical_stage=stage, logical_block="T1C" if stage.startswith("T1C") else "T2", logical_coordinate=coordinate, ticker=ticker, request_start=REQUEST_START, request_end_exclusive=REQUEST_END_EXCLUSIVE, opener=opener, request_parameters={"interval": "1d", "events": "div,splits", "includeAdjustedClose": True}, validate_result=validate))
        gate_result = consume_gate_and_bind(CANONICAL_CONSUMPTION_STATE_ROOT, logical_stage=stage, v8d_frozen_design_commit=EXPECTED_V8D_FROZEN_DESIGN_COMMIT, reviewed_production_implementation_commit=reviewed_commit, raw_authorization_identity=human_authorization_identity, clock=lambda: datetime.now(timezone.utc))
        gate = {"human_gate": gate_result.human_gate, "gate_receipt_key_sha256": gate_result.gate_receipt_key_sha256, "gate_receipt_bytes_sha256": gate_result.gate_receipt_bytes_sha256, "authorization_identity_sha256": gate_result.authorization_identity_sha256}
        def request_factory(coordinate: int) -> V8DRequestPlan:
            return plans[coordinate]
        audit_root = CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8d-acquisition-transport-audit-state" / ("t1c" if stage.startswith("T1C") else "t2")
        execution = execute_v8d_stage(stage=stage, request_factory=request_factory, store=DurableV8DAuditStore(audit_root), reviewed_implementation_commit=reviewed_commit, gate_binding=gate, window_start=REQUEST_START, window_end_exclusive=REQUEST_END_EXCLUSIVE, request_count=REQUEST_COUNT)
        # This binding is deliberately constructed in the fixed production
        # scope, after the durable gate binding and fixed V8D execution exist.
        # There is no module-level constructor or publisher that synthetic
        # callers can invoke with look-alike evidence.
        aggregate_path = Path(execution["aggregate_path"])
        aggregate = execution["aggregate"]
        if aggregate.get("result") not in {"PASS", "BLOCK"}:
            _block("V8D_ACQUISITION_PRODUCTION_BINDING_RESULT_INVALID")
        aggregate_hash = aggregate.get("aggregate_self_hash")
        if aggregate_hash != canonical_sha256({key: value for key, value in aggregate.items() if key != "aggregate_self_hash"}):
            _block("V8D_ACQUISITION_PRODUCTION_BINDING_AGGREGATE_HASH_INVALID")
        dossier_bindings = []
        for path_value in execution["dossier_paths"]:
            path = Path(path_value)
            raw = path.read_bytes()
            dossier = json.loads(raw.decode("utf-8"))
            dossier_hash = dossier.get("audit_artifact_self_hash")
            if dossier_hash != canonical_sha256({key: value for key, value in dossier.items() if key != "audit_artifact_self_hash"}):
                _block("V8D_ACQUISITION_PRODUCTION_BINDING_DOSSIER_HASH_INVALID")
            dossier_bindings.append({
                "filename": path.name,
                "audit_artifact_self_hash": dossier_hash,
                "logical_coordinate": dossier["logical_coordinate"],
            })
        binding_body: dict[str, Any] = {
            "schema_version": PRODUCTION_BINDING_SCHEMA,
            "study": "V8D_HISTORICAL_RESEARCH",
            "artifact_role": "V8D_RAW_ACQUISITION_PRODUCTION_EXECUTION_BINDING",
            "logical_stage": stage,
            "frozen_design_commit": EXPECTED_V8D_FROZEN_DESIGN_COMMIT,
            "reviewed_production_implementation_commit": reviewed_commit,
            "membership_count": REQUEST_COUNT,
            "membership_list_sha256": T1C_LIST_SHA256 if stage.startswith("T1C") else T2_LIST_SHA256,
            "request_start": REQUEST_START,
            "request_end_exclusive": REQUEST_END_EXCLUSIVE,
            "request_count": REQUEST_COUNT,
            "aggregate_filename": aggregate_path.name,
            "aggregate_artifact_self_hash": aggregate_hash,
            "dossier_bindings": dossier_bindings,
            "gate_receipt_key_sha256": gate["gate_receipt_key_sha256"],
            "gate_receipt_bytes_sha256": gate["gate_receipt_bytes_sha256"],
            "authorization_identity_sha256": gate["authorization_identity_sha256"],
            "execution_result": aggregate["result"],
        }
        binding_body["binding_self_hash"] = canonical_sha256(binding_body)
        binding_path = PRODUCTION_BINDING_ROOT / (
            ("t1c" if stage.startswith("T1C") else "t2") + "-raw-acquisition-execution-binding.json"
        )
        binding_payload = canonical_json_bytes(binding_body)
        binding_staging: Path | None = None
        try:
            binding_path.parent.mkdir(parents=True, exist_ok=True)
            if binding_path.is_symlink() or binding_path.exists():
                raise FileExistsError
            binding_staging = binding_path.parent / (binding_path.name + ".staging-" + os.urandom(8).hex())
            with open(binding_staging, "xb") as stream:
                stream.write(binding_payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(str(binding_staging), str(binding_path))
        except FileExistsError as error:
            _block("V8D_ACQUISITION_PRODUCTION_BINDING_ALREADY_PUBLISHED", error)
        except OSError as error:
            _block("V8D_ACQUISITION_PRODUCTION_BINDING_ATOMIC_PUBLISH_FAILED", error)
        finally:
            if binding_staging is not None:
                try:
                    if binding_staging.exists():
                        binding_staging.unlink()
                except OSError:
                    pass
        try:
            return _build_bundle(stage=stage, block="T1C" if stage.startswith("T1C") else "T2", output_root=out, staging_root=staging, tickers=tickers, captured=captured, execution=execution, reviewed_commit=reviewed_commit, gate=gate, membership_hash=T1C_LIST_SHA256 if stage.startswith("T1C") else T2_LIST_SHA256)
        except Exception:
            import shutil
            shutil.rmtree(staging, ignore_errors=True)
            raise
    except V8DAcquisitionEngineBlocked:
        raise
    except Exception as error:  # noqa: BLE001
        _block(getattr(error, "reason", "V8D_ACQUISITION_PRODUCTION_BLOCKED"), error)
    finally:
        if staging is not None and staging.exists():
            import shutil
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "REQUEST_COUNT", "REQUEST_END_EXCLUSIVE", "REQUEST_START", "V8DAcquisitionEngineBlocked",
    "_execute_fixed_production_acquisition",
]
