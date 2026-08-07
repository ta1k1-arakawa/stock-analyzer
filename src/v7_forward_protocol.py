"""Offline V7 forward protocol primitives.

No network, seed acquisition, activation, cache, replay, or formal evaluation
is present in this module. All validators operate on caller-provided local or
synthetic values.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .v7_capacity_engine import (
        V7EngineParameters,
        CausalEventEngine,
        canonical_sha256,
        validate_single_parameter_difference,
    )
except ImportError:
    from v7_capacity_engine import (
        V7EngineParameters,
        CausalEventEngine,
        canonical_sha256,
        validate_single_parameter_difference,
    )


DESIGN_COMMIT = "e3e1367efd913b601a70328a815d88c20af6d147"
DESIGN_PREREGISTRATION_UTC = "2026-08-07T02:48:27Z"
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
CHECKPOINT_FILENAME_RE = re.compile(r"^checkpoint-(\d{4}-\d{2}-\d{2})\.json$")

SEED_REQUIRED_FIELDS = (
    "ticker",
    "trading_date",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "adj_close",
    "raw_volume",
)
SEED_CANONICAL_COLUMNS = (
    "ticker",
    "trading_date",
    "raw_open",
    "raw_high",
    "raw_low",
    "raw_close",
    "adj_close",
    "raw_volume",
)

ACTIVATION_MANIFEST_FIELDS = frozenset({
    "mode",
    "design_commit",
    "implementation_commit",
    "collector_commit",
    "activation_authorization_utc",
    "activation_boundary_first_jpx_trading_date",
    "calendar_source",
    "calendar_version",
    "calendar_timezone",
    "data_source",
    "data_source_schema",
    "acquisition_window_jst",
    "universe_csv_sha",
    "ticker_list_sha",
    "arm_a_parameters_sha256",
    "arm_b_parameters_sha256",
    "shared_rules_sha256",
    "output_root",
    "seed_data_source",
    "seed_data_schema",
    "seed_acquisition_utc",
    "seed_cutoff_trading_date",
    "seed_ticker_count",
    "seed_row_count",
    "seed_payload_manifest_sha256",
    "seed_canonical_csv_sha256",
    "seed_generation_commit",
    "seed_validation_result",
    "arm_seed_hash_equal",
    "arm_candidate_input_hash_equal",
    "arm_market_gate_input_hash_equal",
})


class ProtocolBlocked(ValueError):
    """Raised for fail-closed causal, manifest, or checkpoint violations."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_date(value: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolBlocked("INVALID_DATE")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as error:
        raise ProtocolBlocked("INVALID_DATE") from error
    if parsed.strftime("%Y-%m-%d") != value:
        raise ProtocolBlocked("INVALID_DATE")
    return parsed


def _parse_utc_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise ProtocolBlocked("UTC_TIMESTAMP_INVALID:" + field_name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProtocolBlocked("UTC_TIMESTAMP_INVALID:" + field_name) from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ProtocolBlocked("UTC_TIMESTAMP_MUST_BE_AWARE_UTC:" + field_name)
    return parsed


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _valid_commit(value: Any) -> bool:
    return isinstance(value, str) and COMMIT_RE.fullmatch(value) is not None


def _row_for_hash(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): row[key] for key in sorted(row)}


def _canonical_seed_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [_row_for_hash(row) for row in rows],
        key=lambda row: (str(row["ticker"]), str(row["trading_date"])),
    )


def _canonical_seed_csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=SEED_CANONICAL_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in _canonical_seed_rows(rows):
        writer.writerow({field: row[field] for field in SEED_CANONICAL_COLUMNS})
    return stream.getvalue().encode("utf-8")


def _payload_hash(row: Mapping[str, Any]) -> str:
    supplied = row.get("payload_sha256")
    if supplied is not None:
        if not _valid_sha(supplied):
            raise ProtocolBlocked("SEED_PAYLOAD_SHA_INVALID")
        return str(supplied).lower()
    without_payload = {key: value for key, value in row.items() if key != "payload_sha256"}
    return sha256_bytes(canonical_json_bytes(_row_for_hash(without_payload)))


def validate_seed_rows(
    rows: Sequence[Mapping[str, Any]],
    fixed_universe: Sequence[str] | set[str],
    activation_boundary_first_jpx_trading_date: str,
    *,
    expected_seed_canonical_sha256: str | None = None,
    expected_seed_payload_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate local/synthetic seed rows and return deterministic metadata."""
    boundary = _parse_date(activation_boundary_first_jpx_trading_date)
    universe = tuple(sorted({str(ticker) for ticker in fixed_universe}))
    if not universe:
        raise ProtocolBlocked("EMPTY_FIXED_UNIVERSE")
    seen: set[tuple[str, str]] = set()
    valid_rows: list[dict[str, Any]] = []
    for source_row in rows:
        if not isinstance(source_row, Mapping):
            raise ProtocolBlocked("SEED_ROW_NOT_MAPPING")
        missing = sorted(set(SEED_REQUIRED_FIELDS).difference(source_row))
        if missing:
            raise ProtocolBlocked("SEED_SCHEMA_MISSING:" + ",".join(missing))
        row = dict(source_row)
        ticker = str(row["ticker"])
        trading_date = str(row["trading_date"])
        if ticker not in universe:
            raise ProtocolBlocked("TICKER_OUTSIDE_FIXED_UNIVERSE")
        parsed_date = _parse_date(trading_date)
        if parsed_date >= boundary:
            raise ProtocolBlocked("SEED_ROW_ON_OR_AFTER_ACTIVATION_BOUNDARY")
        key = (ticker, trading_date)
        if key in seen:
            raise ProtocolBlocked("DUPLICATE_TICKER_DATE")
        seen.add(key)
        for field_name in ("raw_open", "raw_high", "raw_low", "raw_close", "adj_close", "raw_volume"):
            value = row[field_name]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                reason = "SEED_NONFINITE_ADJ_CLOSE" if field_name == "adj_close" else "SEED_NONFINITE_OHLCV"
                raise ProtocolBlocked(reason)
            if field_name != "raw_volume" and float(value) <= 0:
                reason = "SEED_NONPOSITIVE_ADJ_CLOSE" if field_name == "adj_close" else "SEED_NONPOSITIVE_PRICE"
                raise ProtocolBlocked(reason)
            if field_name == "raw_volume" and float(value) < 0:
                raise ProtocolBlocked("SEED_NEGATIVE_VOLUME")
        valid_rows.append(row)

    by_ticker: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in universe}
    for row in valid_rows:
        by_ticker[str(row["ticker"])].append(row)
    selected_rows: list[dict[str, Any]] = []
    ticker_manifest: list[dict[str, Any]] = []
    for ticker in universe:
        ordered = sorted(by_ticker[ticker], key=lambda row: str(row["trading_date"]))
        selected = ordered[-252:]
        selected_rows.extend(selected)
        ticker_manifest.append({
            "ticker": ticker,
            "first_seed_trading_date": selected[0]["trading_date"] if selected else None,
            "last_seed_trading_date": selected[-1]["trading_date"] if selected else None,
            "valid_observation_count": len(selected),
            "ticker_payload_sha256": sha256_bytes(_canonical_seed_csv_bytes(selected)),
            "eligibility_at_activation": len(selected) == 252,
        })

    canonical_rows = _canonical_seed_rows(selected_rows)
    seed_canonical_sha256 = sha256_bytes(_canonical_seed_csv_bytes(canonical_rows))
    payload_manifest = [
        {
            "ticker": item["ticker"],
            "ticker_payload_sha256": item["ticker_payload_sha256"],
        }
        for item in ticker_manifest
    ]
    seed_payload_manifest_sha256 = sha256_bytes(canonical_json_bytes(payload_manifest))
    if expected_seed_canonical_sha256 is not None and expected_seed_canonical_sha256 != seed_canonical_sha256:
        raise ProtocolBlocked("SEED_CANONICAL_HASH_MISMATCH")
    if expected_seed_payload_manifest_sha256 is not None and expected_seed_payload_manifest_sha256 != seed_payload_manifest_sha256:
        raise ProtocolBlocked("SEED_PAYLOAD_MANIFEST_HASH_MISMATCH")

    selected_dates = [str(row["trading_date"]) for row in canonical_rows]
    return {
        "seed_validation_result": "PASS",
        "ticker_count": len(universe),
        "row_count": len(canonical_rows),
        "eligible_ticker_count": sum(
            1 for item in ticker_manifest if item["eligibility_at_activation"]
        ),
        "ineligible_ticker_count": sum(
            1 for item in ticker_manifest if not item["eligibility_at_activation"]
        ),
        "seed_cutoff_trading_date": max(selected_dates) if selected_dates else None,
        "ticker_manifest": ticker_manifest,
        "canonical_rows": canonical_rows,
        "seed_canonical_sha256": seed_canonical_sha256,
        "seed_payload_manifest_sha256": seed_payload_manifest_sha256,
    }


def _seed_validation_passed(value: Any) -> bool:
    if value == "PASS":
        return True
    if isinstance(value, Mapping):
        result = value.get("seed_validation_result", value.get("status"))
        return result in {"PASS", True}
    return False


def validate_activation_manifest(
    manifest: Mapping[str, Any],
    *,
    control: V7EngineParameters | None = None,
    variant: V7EngineParameters | None = None,
    seed_validation: Mapping[str, Any] | None = None,
    expected_design_commit: str = DESIGN_COMMIT,
) -> dict[str, Any]:
    """Validate only a DRY_RUN_ONLY manifest; never creates or activates one."""
    if not isinstance(manifest, Mapping):
        raise ProtocolBlocked("MANIFEST_NOT_MAPPING")
    unknown = sorted(set(manifest).difference(ACTIVATION_MANIFEST_FIELDS))
    missing = sorted(ACTIVATION_MANIFEST_FIELDS.difference(manifest))
    if unknown:
        raise ProtocolBlocked("MANIFEST_UNKNOWN_FIELD:" + ",".join(unknown))
    if missing:
        raise ProtocolBlocked("MANIFEST_MISSING_FIELD:" + ",".join(missing))
    if manifest["mode"] != "DRY_RUN_ONLY":
        raise ProtocolBlocked("MANIFEST_MODE_NOT_DRY_RUN_ONLY")
    if manifest["design_commit"] != expected_design_commit:
        raise ProtocolBlocked("DESIGN_COMMIT_MISMATCH")
    preregistration = _parse_utc_timestamp(
        DESIGN_PREREGISTRATION_UTC, "design_preregistration"
    )
    seed_acquisition = _parse_utc_timestamp(
        manifest["seed_acquisition_utc"], "seed_acquisition_utc"
    )
    activation_authorization = _parse_utc_timestamp(
        manifest["activation_authorization_utc"], "activation_authorization_utc"
    )
    if not preregistration < seed_acquisition:
        raise ProtocolBlocked("SEED_ACQUISITION_NOT_AFTER_PREREGISTRATION")
    if not seed_acquisition < activation_authorization:
        raise ProtocolBlocked("ACTIVATION_TIME_ORDER_INVALID")
    for key in (
        "implementation_commit",
        "collector_commit",
        "seed_generation_commit",
    ):
        if not _valid_commit(manifest[key]):
            raise ProtocolBlocked("MANIFEST_COMMIT_INVALID:" + key)
    for key in (
        "universe_csv_sha",
        "ticker_list_sha",
        "arm_a_parameters_sha256",
        "arm_b_parameters_sha256",
        "shared_rules_sha256",
        "seed_payload_manifest_sha256",
        "seed_canonical_csv_sha256",
    ):
        if not _valid_sha(manifest[key]):
            raise ProtocolBlocked("MANIFEST_SHA_INVALID:" + key)
    if manifest["calendar_timezone"] != "Asia/Tokyo":
        raise ProtocolBlocked("CALENDAR_TIMEZONE_MUST_BE_ASIA_TOKYO")
    boundary = _parse_date(manifest["activation_boundary_first_jpx_trading_date"])
    seed_cutoff = _parse_date(manifest["seed_cutoff_trading_date"])
    if seed_cutoff >= boundary:
        raise ProtocolBlocked("SEED_CUTOFF_NOT_BEFORE_ACTIVATION")
    if not _seed_validation_passed(manifest["seed_validation_result"]):
        raise ProtocolBlocked("SEED_VALIDATION_FAILURE")
    for key in ("arm_seed_hash_equal", "arm_candidate_input_hash_equal", "arm_market_gate_input_hash_equal"):
        if manifest[key] is not True:
            raise ProtocolBlocked("ARM_INPUT_HASH_MISMATCH:" + key)
    if not isinstance(manifest["output_root"], str) or not manifest["output_root"]:
        raise ProtocolBlocked("OUTPUT_ROOT_INVALID")
    if not isinstance(manifest["data_source"], str) or not manifest["data_source"]:
        raise ProtocolBlocked("DATA_SOURCE_INVALID")
    for key in ("seed_ticker_count", "seed_row_count"):
        if not isinstance(manifest[key], int) or manifest[key] < 0:
            raise ProtocolBlocked("MANIFEST_COUNT_INVALID:" + key)
    if control is not None or variant is not None:
        if control is None or variant is None:
            raise ProtocolBlocked("ARM_PARAMETER_PAIR_REQUIRED")
        validate_single_parameter_difference(control, variant)
        if manifest["arm_a_parameters_sha256"] != control.sha256():
            raise ProtocolBlocked("CONTROL_PARAMETER_HASH_MISMATCH")
        if manifest["arm_b_parameters_sha256"] != variant.sha256():
            raise ProtocolBlocked("VARIANT_PARAMETER_HASH_MISMATCH")
    if seed_validation is not None:
        if seed_validation.get("seed_validation_result") != "PASS":
            raise ProtocolBlocked("SEED_VALIDATION_FAILURE")
        if manifest["seed_cutoff_trading_date"] != seed_validation.get(
            "seed_cutoff_trading_date"
        ):
            raise ProtocolBlocked("SEED_CUTOFF_MISMATCH")
        for key in ("seed_canonical_sha256", "seed_payload_manifest_sha256"):
            manifest_key = (
                "seed_canonical_csv_sha256"
                if key == "seed_canonical_sha256"
                else "seed_payload_manifest_sha256"
            )
            if manifest[manifest_key] != seed_validation[key]:
                raise ProtocolBlocked("SEED_HASH_MISMATCH:" + manifest_key)
        if seed_validation.get("seed_cutoff_trading_date") is not None:
            if _parse_date(str(seed_validation["seed_cutoff_trading_date"])) >= boundary:
                raise ProtocolBlocked("SEED_ROW_ON_OR_AFTER_ACTIVATION_BOUNDARY")
        if manifest["seed_ticker_count"] != seed_validation["ticker_count"]:
            raise ProtocolBlocked("SEED_TICKER_COUNT_MISMATCH")
        if manifest["seed_row_count"] != seed_validation["row_count"]:
            raise ProtocolBlocked("SEED_ROW_COUNT_MISMATCH")
    return {"status": "PASS", "mode": "DRY_RUN_ONLY", "activation_created": False}


@dataclass(frozen=True)
class ArmInputHashes:
    seed_hash: str
    price_snapshot_hash: str
    candidate_snapshot_hash: str
    market_gate_snapshot_hash: str

    def as_dict(self) -> dict[str, str]:
        return {
            "seed_hash": self.seed_hash,
            "price_snapshot_hash": self.price_snapshot_hash,
            "candidate_snapshot_hash": self.candidate_snapshot_hash,
            "market_gate_snapshot_hash": self.market_gate_snapshot_hash,
        }


def validate_arm_input_hashes(
    control: ArmInputHashes, variant: ArmInputHashes
) -> bool:
    if control.as_dict() != variant.as_dict():
        raise ProtocolBlocked("ARM_INPUT_HASH_MISMATCH")
    for value in control.as_dict().values():
        if not _valid_sha(value):
            raise ProtocolBlocked("ARM_INPUT_SHA_INVALID")
    return True


@dataclass
class DualArmStudy:
    control: CausalEventEngine
    variant: CausalEventEngine
    control_input_hashes: ArmInputHashes
    variant_input_hashes: ArmInputHashes

    def __post_init__(self) -> None:
        validate_arm_input_hashes(self.control_input_hashes, self.variant_input_hashes)
        if self.control is self.variant or self.control.state is self.variant.state:
            raise ProtocolBlocked("CROSS_ARM_STATE_REFERENCE")
        if self.control.parameters.max_open_positions != 2 or self.variant.parameters.max_open_positions != 3:
            raise ProtocolBlocked("ARM_MAX_POSITION_CONTRACT_INVALID")
        validate_single_parameter_difference(self.control.parameters, self.variant.parameters)

    def run(self) -> "DualArmStudy":
        self.control.run()
        self.variant.run()
        return self

    def state_objects_are_independent(self) -> bool:
        return (
            self.control is not self.variant
            and self.control.state is not self.variant.state
            and self.control.state.open_positions is not self.variant.state.open_positions
            and self.control.state.pending_orders_by_entry_date is not self.variant.state.pending_orders_by_entry_date
            and self.control.state.pending_proceeds_by_available_date is not self.variant.state.pending_proceeds_by_available_date
            and self.control.state.event_audit is not self.variant.state.event_audit
            and self.control.state.completed_trades is not self.variant.state.completed_trades
            and self.control.state.daily_equity is not self.variant.state.daily_equity
        )

    def state_snapshots(self) -> dict[str, dict[str, Any]]:
        if not self.state_objects_are_independent():
            raise ProtocolBlocked("CROSS_ARM_STATE_REFERENCE")
        return {
            "CONTROL": self.control.state_snapshot(),
            "CAPACITY_3": self.variant.state_snapshot(),
        }


def create_dual_arm_study(
    frames: Mapping[str, Mapping[str, Mapping[str, float]]],
    calendar: Sequence[str],
    candidates: Sequence[Mapping[str, Any]],
    control_input_hashes: ArmInputHashes,
    variant_input_hashes: ArmInputHashes,
    control_parameters: V7EngineParameters | None = None,
    variant_parameters: V7EngineParameters | None = None,
    split_events_by_day: Mapping[str, Sequence[str]] | None = None,
) -> DualArmStudy:
    control = control_parameters or V7EngineParameters.control()
    variant = variant_parameters or V7EngineParameters.capacity_3()
    validate_single_parameter_difference(control, variant)
    return DualArmStudy(
        CausalEventEngine(
            copy.deepcopy(frames),
            tuple(calendar),
            copy.deepcopy(candidates),
            control,
            split_events_by_day=copy.deepcopy(split_events_by_day),
        ),
        CausalEventEngine(
            copy.deepcopy(frames),
            tuple(calendar),
            copy.deepcopy(candidates),
            variant,
            split_events_by_day=copy.deepcopy(split_events_by_day),
        ),
        control_input_hashes,
        variant_input_hashes,
    )


CHECKPOINT_FIELDS = (
    "previous_checkpoint_sha256",
    "current_checkpoint_sha256",
    "last_completed_engine_day",
    "arm_a_state_sha256",
    "arm_b_state_sha256",
    "candidate_snapshot_sha256",
    "price_snapshot_sha256",
    "collector_commit",
    "status",
)


class CheckpointWriter:
    """Append-only COMPLETE checkpoint writer for synthetic/tmp paths only."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _files(self) -> list[Path]:
        return sorted(self.root.glob("checkpoint-*.json"))

    def _staging_files(self) -> list[Path]:
        return sorted(self.root.glob("*.staging-*"))

    def _canonical_body(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: record[key]
            for key in CHECKPOINT_FIELDS
            if key != "current_checkpoint_sha256"
        }

    def _verify_record(self, path: Path) -> dict[str, Any]:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProtocolBlocked("CHECKPOINT_READ_FAILED") from error
        if set(record) != set(CHECKPOINT_FIELDS):
            raise ProtocolBlocked("CHECKPOINT_SCHEMA_INVALID")
        match = CHECKPOINT_FILENAME_RE.fullmatch(path.name)
        if match is None:
            raise ProtocolBlocked("CHECKPOINT_FILENAME_INVALID")
        if record.get("last_completed_engine_day") != match.group(1):
            raise ProtocolBlocked("CHECKPOINT_FILENAME_DATE_MISMATCH")
        if record["status"] != "COMPLETE":
            raise ProtocolBlocked("PARTIAL_CHECKPOINT_BLOCKED")
        for key in (
            "current_checkpoint_sha256",
            "arm_a_state_sha256",
            "arm_b_state_sha256",
            "candidate_snapshot_sha256",
            "price_snapshot_sha256",
        ):
            if not _valid_sha(record[key]):
                raise ProtocolBlocked("CHECKPOINT_SHA_INVALID:" + key)
        if record["previous_checkpoint_sha256"] is not None and not _valid_sha(
            record["previous_checkpoint_sha256"]
        ):
            raise ProtocolBlocked("CHECKPOINT_SHA_INVALID:previous_checkpoint_sha256")
        if not _valid_commit(record["collector_commit"]):
            raise ProtocolBlocked("CHECKPOINT_COMMIT_INVALID")
        _parse_date(record["last_completed_engine_day"])
        expected = sha256_bytes(canonical_json_bytes(self._canonical_body(record)))
        if expected != record["current_checkpoint_sha256"]:
            raise ProtocolBlocked("CHECKPOINT_HASH_MISMATCH")
        return record

    def _verified_records(self) -> list[dict[str, Any]]:
        records = [self._verify_record(path) for path in self._files()]
        previous = None
        previous_day = None
        for record in records:
            if record["previous_checkpoint_sha256"] != previous:
                raise ProtocolBlocked("CHECKPOINT_PREVIOUS_HASH_MISMATCH")
            current_day = _parse_date(record["last_completed_engine_day"])
            if previous_day is not None and current_day <= previous_day:
                raise ProtocolBlocked("CHECKPOINT_ENGINE_DAY_NOT_INCREASING")
            previous_day = current_day
            previous = record["current_checkpoint_sha256"]
        return records

    def write_complete(self, **values: Any) -> dict[str, Any]:
        required_input_fields = set(CHECKPOINT_FIELDS) - {"current_checkpoint_sha256"}
        missing = [
            key for key in required_input_fields
            if key not in values
            or (values[key] is None and key != "previous_checkpoint_sha256")
        ]
        if missing:
            raise ProtocolBlocked("CHECKPOINT_MISSING_FIELD:" + ",".join(missing))
        if values.get("status") != "COMPLETE":
            raise ProtocolBlocked("PARTIAL_CHECKPOINT_BLOCKED")
        records = self._verified_records()
        previous = records[-1]["current_checkpoint_sha256"] if records else None
        if values["previous_checkpoint_sha256"] != previous:
            raise ProtocolBlocked("CHECKPOINT_PREVIOUS_HASH_MISMATCH")
        day = str(values["last_completed_engine_day"])
        _parse_date(day)
        for key in (
            "arm_a_state_sha256",
            "arm_b_state_sha256",
            "candidate_snapshot_sha256",
            "price_snapshot_sha256",
        ):
            if not _valid_sha(values[key]):
                raise ProtocolBlocked("CHECKPOINT_SHA_INVALID:" + key)
        if not _valid_commit(values["collector_commit"]):
            raise ProtocolBlocked("CHECKPOINT_COMMIT_INVALID")
        if values["previous_checkpoint_sha256"] is not None and not _valid_sha(
            values["previous_checkpoint_sha256"]
        ):
            raise ProtocolBlocked("CHECKPOINT_SHA_INVALID:previous_checkpoint_sha256")
        final_path = self.root / f"checkpoint-{day}.json"
        if final_path.exists():
            raise ProtocolBlocked("DUPLICATE_ENGINE_DAY_PROCESSING")
        if records and _parse_date(day) <= _parse_date(
            records[-1]["last_completed_engine_day"]
        ):
            raise ProtocolBlocked("CHECKPOINT_ENGINE_DAY_NOT_INCREASING")
        body = {
            key: values[key]
            for key in CHECKPOINT_FIELDS
            if key != "current_checkpoint_sha256"
        }
        body["current_checkpoint_sha256"] = sha256_bytes(canonical_json_bytes(body))
        if self._staging_files():
            raise ProtocolBlocked("PARTIAL_CHECKPOINT_BLOCKED")
        fd, staging_name = tempfile.mkstemp(
            prefix=".checkpoint-", suffix=".staging-" + day, dir=self.root
        )
        staging_path = Path(staging_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(canonical_json_bytes(body).decode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staging_path, final_path)
        finally:
            if staging_path.exists():
                staging_path.unlink()
        return self._verify_record(final_path)

    def load_last_complete(self) -> dict[str, Any] | None:
        records = self._verified_records()
        return records[-1] if records else None

    def restart_from_last_checkpoint(self) -> dict[str, Any] | None:
        return self.load_last_complete()
