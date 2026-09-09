"""Offline-only V7 forward-study append-only persistence and restart.

This module owns no network, collector, activation, or real-order path.  It
persists caller-supplied study-day snapshots and engine runtime state to a
local append-only ``days/YYYY-MM-DD/`` study root, and restores a
:class:`~src.v7_capacity_engine.CausalEventEngine` runtime from that store so
that a study may resume exactly where it stopped.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from src.v7_capacity_engine import (
    CausalEventEngine,
    EngineState,
    OpenPosition,
    PendingOrder,
    PendingProceeds,
)


SCHEMA_VERSION = 1

DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")

DAY_FILE_PRICE = "price_snapshot.json"
DAY_FILE_CANDIDATE = "candidate_snapshot.json"
DAY_FILE_MARKET_GATE = "market_gate_snapshot.json"
DAY_FILE_ARM_A = "arm_a_runtime.json"
DAY_FILE_ARM_B = "arm_b_runtime.json"
DAY_FILE_CHECKPOINT = "checkpoint.json"

DAY_FILES = (
    DAY_FILE_PRICE,
    DAY_FILE_CANDIDATE,
    DAY_FILE_MARKET_GATE,
    DAY_FILE_ARM_A,
    DAY_FILE_ARM_B,
    DAY_FILE_CHECKPOINT,
)

CHECKPOINT_FIELDS = (
    "activation_manifest_sha256",
    "previous_checkpoint_sha256",
    "current_checkpoint_sha256",
    "last_completed_engine_day",
    "arm_a_state_sha256",
    "arm_b_state_sha256",
    "price_snapshot_sha256",
    "candidate_snapshot_sha256",
    "market_gate_snapshot_sha256",
    "collector_commit",
    "status",
)

RUNTIME_FIELDS = (
    "schema_version",
    "parameters_sha256",
    "engine_day",
    "available_cash",
    "open_positions",
    "pending_orders_by_entry_date",
    "pending_proceeds_by_available_date",
    "completed_trades",
    "daily_equity",
    "event_audit",
    "safety_counters",
    "skip_reason_counts",
)


class V7ForwardPersistenceBlocked(RuntimeError):
    """Fail-closed forward-persistence or restart boundary violation."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


def _parse_iso_date(value: str) -> date:
    if not isinstance(value, str):
        raise V7ForwardPersistenceBlocked("INVALID_DATE_FORMAT")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError) as error:
        raise V7ForwardPersistenceBlocked("INVALID_DATE_FORMAT") from error
    if parsed.isoformat() != value:
        raise V7ForwardPersistenceBlocked("INVALID_DATE_FORMAT")
    return parsed


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except ValueError as error:
        raise V7ForwardPersistenceBlocked("NONFINITE_VALUE") from error


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _valid_commit(value: Any) -> bool:
    return isinstance(value, str) and bool(COMMIT_RE.fullmatch(value))


def _atomic_write(path: Path, data: bytes) -> None:
    with open(path, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


# ---------------------------------------------------------------------------
# Engine runtime export / restore
# ---------------------------------------------------------------------------


def export_engine_runtime(engine: CausalEventEngine) -> dict[str, Any]:
    state: EngineState = engine.state
    return {
        "schema_version": SCHEMA_VERSION,
        "parameters_sha256": engine.parameters.sha256(),
        "engine_day": state.engine_day,
        "available_cash": state.available_cash,
        "open_positions": [asdict(item) for item in state.open_positions],
        "pending_orders_by_entry_date": {
            day: [asdict(item) for item in items]
            for day, items in state.pending_orders_by_entry_date.items()
        },
        "pending_proceeds_by_available_date": {
            day: [asdict(item) for item in items]
            for day, items in state.pending_proceeds_by_available_date.items()
        },
        "completed_trades": copy.deepcopy(state.completed_trades),
        "daily_equity": copy.deepcopy(state.daily_equity),
        "event_audit": copy.deepcopy(state.event_audit),
        "safety_counters": engine.safety_counters(),
        "skip_reason_counts": engine.skip_reason_counts(),
    }


def restore_engine_runtime(engine: CausalEventEngine, payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise V7ForwardPersistenceBlocked("RUNTIME_PAYLOAD_INVALID")
    missing = sorted(set(RUNTIME_FIELDS) - set(payload))
    unknown = sorted(set(payload) - set(RUNTIME_FIELDS))
    if missing:
        raise V7ForwardPersistenceBlocked("RUNTIME_FIELD_MISSING:" + ",".join(missing))
    if unknown:
        raise V7ForwardPersistenceBlocked("RUNTIME_FIELD_UNKNOWN:" + ",".join(unknown))
    canonical_json_bytes(payload)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise V7ForwardPersistenceBlocked("RUNTIME_SCHEMA_VERSION_MISMATCH")
    if payload["parameters_sha256"] != engine.parameters.sha256():
        raise V7ForwardPersistenceBlocked("RUNTIME_PARAMETERS_MISMATCH")

    try:
        open_positions = [OpenPosition(**item) for item in payload["open_positions"]]
        pending_orders = {
            day: [PendingOrder(**item) for item in items]
            for day, items in payload["pending_orders_by_entry_date"].items()
        }
        pending_proceeds = {
            day: [PendingProceeds(**item) for item in items]
            for day, items in payload["pending_proceeds_by_available_date"].items()
        }
    except TypeError as error:
        raise V7ForwardPersistenceBlocked("RUNTIME_FIELD_SCHEMA_INVALID") from error

    engine.state.engine_day = payload["engine_day"]
    engine.state.available_cash = payload["available_cash"]
    engine.state.open_positions = open_positions
    engine.state.pending_orders_by_entry_date = pending_orders
    engine.state.pending_proceeds_by_available_date = pending_proceeds
    engine.state.completed_trades = copy.deepcopy(list(payload["completed_trades"]))
    engine.state.daily_equity = copy.deepcopy(list(payload["daily_equity"]))
    engine.state.event_audit = copy.deepcopy(list(payload["event_audit"]))

    for name, count in payload["safety_counters"].items():
        if not count:
            continue
        try:
            engine.record_safety_violation(name, count)
        except ValueError:
            continue  # derived counter: recomputed from restored state, not sticky

    # No public setter exists for skip-reason accounting; engine source is immutable.
    engine._skip_reason_counts = dict(payload["skip_reason_counts"])


# ---------------------------------------------------------------------------
# Append-only day-directory study store
# ---------------------------------------------------------------------------


class ForwardStudyStore:
    """Append-only ``days/YYYY-MM-DD/`` study root with atomic day publish."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.days_root = self.root / "days"
        self.days_root.mkdir(parents=True, exist_ok=True)

    def _day_dir(self, day: str) -> Path:
        return self.days_root / day

    def _final_days(self) -> list[str]:
        return sorted(
            entry.name
            for entry in self.days_root.iterdir()
            if entry.is_dir() and DAY_DIR_RE.fullmatch(entry.name)
        )

    def _staging_entries(self) -> list[Path]:
        return sorted(
            entry for entry in self.days_root.iterdir() if ".staging-" in entry.name
        )

    def _ensure_no_staging_remnants(self) -> None:
        if self._staging_entries():
            raise V7ForwardPersistenceBlocked("PARTIAL_DAY_COMMIT")

    def _read_json(self, path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise V7ForwardPersistenceBlocked("DAY_FILE_READ_FAILED:" + path.name) from error

    def _verify_chain(
        self,
        *,
        expected_activation_manifest_sha256: str | None = None,
        expected_collector_commit: str | None = None,
    ) -> list[dict[str, Any]]:
        self._ensure_no_staging_remnants()
        previous_sha: str | None = None
        previous_date: date | None = None
        records: list[dict[str, Any]] = []
        for day in self._final_days():
            day_dir = self._day_dir(day)
            actual_files = {entry.name for entry in day_dir.iterdir()}
            if actual_files != set(DAY_FILES):
                raise V7ForwardPersistenceBlocked("DAY_SCHEMA_INVALID:" + day)
            checkpoint = self._read_json(day_dir / DAY_FILE_CHECKPOINT)
            if set(checkpoint) != set(CHECKPOINT_FIELDS):
                raise V7ForwardPersistenceBlocked("CHECKPOINT_SCHEMA_INVALID:" + day)
            if checkpoint["status"] != "COMPLETE":
                raise V7ForwardPersistenceBlocked("PARTIAL_DAY_COMMIT:" + day)
            if checkpoint["last_completed_engine_day"] != day:
                raise V7ForwardPersistenceBlocked("DAY_DIRECTORY_MISMATCH:" + day)
            current_date = _parse_iso_date(day)
            if previous_date is not None and current_date <= previous_date:
                raise V7ForwardPersistenceBlocked("ENGINE_DAY_NOT_INCREASING:" + day)
            if checkpoint["previous_checkpoint_sha256"] != previous_sha:
                raise V7ForwardPersistenceBlocked("CHECKPOINT_CHAIN_MISMATCH:" + day)
            if not _valid_commit(checkpoint["collector_commit"]):
                raise V7ForwardPersistenceBlocked("COLLECTOR_COMMIT_INVALID:" + day)
            if not _valid_sha(checkpoint["activation_manifest_sha256"]):
                raise V7ForwardPersistenceBlocked("ACTIVATION_MANIFEST_SHA_INVALID:" + day)
            if (
                expected_activation_manifest_sha256 is not None
                and checkpoint["activation_manifest_sha256"] != expected_activation_manifest_sha256
            ):
                raise V7ForwardPersistenceBlocked("ACTIVATION_MANIFEST_MISMATCH:" + day)
            if (
                expected_collector_commit is not None
                and checkpoint["collector_commit"] != expected_collector_commit
            ):
                raise V7ForwardPersistenceBlocked("COLLECTOR_COMMIT_MISMATCH:" + day)
            for filename, hash_field in (
                (DAY_FILE_PRICE, "price_snapshot_sha256"),
                (DAY_FILE_CANDIDATE, "candidate_snapshot_sha256"),
                (DAY_FILE_MARKET_GATE, "market_gate_snapshot_sha256"),
                (DAY_FILE_ARM_A, "arm_a_state_sha256"),
                (DAY_FILE_ARM_B, "arm_b_state_sha256"),
            ):
                if not _valid_sha(checkpoint[hash_field]):
                    raise V7ForwardPersistenceBlocked("CHECKPOINT_SHA_INVALID:" + day + ":" + hash_field)
                payload = self._read_json(day_dir / filename)
                digest = sha256_bytes(canonical_json_bytes(payload))
                if digest != checkpoint[hash_field]:
                    raise V7ForwardPersistenceBlocked(
                        "SNAPSHOT_HASH_MISMATCH:" + day + ":" + filename
                    )
            expected_checkpoint_sha = sha256_bytes(
                canonical_json_bytes(
                    {key: checkpoint[key] for key in CHECKPOINT_FIELDS if key != "current_checkpoint_sha256"}
                )
            )
            if not _valid_sha(checkpoint["current_checkpoint_sha256"]):
                raise V7ForwardPersistenceBlocked("CHECKPOINT_SHA_INVALID:" + day + ":current_checkpoint_sha256")
            if expected_checkpoint_sha != checkpoint["current_checkpoint_sha256"]:
                raise V7ForwardPersistenceBlocked("CHECKPOINT_HASH_MISMATCH:" + day)
            previous_sha = checkpoint["current_checkpoint_sha256"]
            previous_date = current_date
            records.append(checkpoint)
        return records

    def load_latest_checkpoint(self) -> dict[str, Any] | None:
        records = self._verify_chain()
        return records[-1] if records else None

    def write_day(
        self,
        day: str,
        *,
        price_snapshot: Any,
        candidate_snapshot: Any,
        market_gate_snapshot: Any,
        arm_a_runtime: Mapping[str, Any],
        arm_b_runtime: Mapping[str, Any],
        activation_manifest_sha256: str,
        collector_commit: str,
        fault_injector: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        if not DAY_DIR_RE.fullmatch(day):
            raise V7ForwardPersistenceBlocked("DAY_FORMAT_INVALID")
        _parse_iso_date(day)
        if not _valid_sha(activation_manifest_sha256):
            raise V7ForwardPersistenceBlocked("ACTIVATION_MANIFEST_SHA_INVALID")
        if not _valid_commit(collector_commit):
            raise V7ForwardPersistenceBlocked("COLLECTOR_COMMIT_INVALID")
        if isinstance(arm_a_runtime, Mapping) and arm_a_runtime.get("engine_day") not in (day, None):
            raise V7ForwardPersistenceBlocked("ARM_A_RUNTIME_DAY_MISMATCH")
        if isinstance(arm_b_runtime, Mapping) and arm_b_runtime.get("engine_day") not in (day, None):
            raise V7ForwardPersistenceBlocked("ARM_B_RUNTIME_DAY_MISMATCH")

        records = self._verify_chain()
        previous_record = records[-1] if records else None
        final_dir = self._day_dir(day)
        if final_dir.exists():
            raise V7ForwardPersistenceBlocked("DUPLICATE_ENGINE_DAY_PROCESSING")
        if previous_record is not None:
            if _parse_iso_date(day) <= _parse_iso_date(previous_record["last_completed_engine_day"]):
                raise V7ForwardPersistenceBlocked("ENGINE_DAY_NOT_INCREASING")
            if previous_record["activation_manifest_sha256"] != activation_manifest_sha256:
                raise V7ForwardPersistenceBlocked("ACTIVATION_MANIFEST_MISMATCH")
            if previous_record["collector_commit"] != collector_commit:
                raise V7ForwardPersistenceBlocked("COLLECTOR_COMMIT_MISMATCH")

        price_bytes = canonical_json_bytes(price_snapshot)
        candidate_bytes = canonical_json_bytes(candidate_snapshot)
        market_gate_bytes = canonical_json_bytes(market_gate_snapshot)
        arm_a_bytes = canonical_json_bytes(arm_a_runtime)
        arm_b_bytes = canonical_json_bytes(arm_b_runtime)

        checkpoint_body = {
            "activation_manifest_sha256": activation_manifest_sha256,
            "previous_checkpoint_sha256": previous_record["current_checkpoint_sha256"] if previous_record else None,
            "last_completed_engine_day": day,
            "arm_a_state_sha256": sha256_bytes(arm_a_bytes),
            "arm_b_state_sha256": sha256_bytes(arm_b_bytes),
            "price_snapshot_sha256": sha256_bytes(price_bytes),
            "candidate_snapshot_sha256": sha256_bytes(candidate_bytes),
            "market_gate_snapshot_sha256": sha256_bytes(market_gate_bytes),
            "collector_commit": collector_commit,
            "status": "COMPLETE",
        }
        checkpoint_record = dict(checkpoint_body)
        checkpoint_record["current_checkpoint_sha256"] = sha256_bytes(canonical_json_bytes(checkpoint_body))
        checkpoint_bytes = canonical_json_bytes(checkpoint_record)

        # Staging lives directly under days_root so the final publish is a single
        # same-directory atomic rename; nothing under DAY_DIR_RE is visible until then.
        staging_dir = Path(tempfile.mkdtemp(prefix=f"{day}.staging-", dir=self.days_root))
        _atomic_write(staging_dir / DAY_FILE_PRICE, price_bytes)
        if fault_injector is not None:
            fault_injector("after_price_write")
        _atomic_write(staging_dir / DAY_FILE_CANDIDATE, candidate_bytes)
        if fault_injector is not None:
            fault_injector("after_candidate_write")
        _atomic_write(staging_dir / DAY_FILE_MARKET_GATE, market_gate_bytes)
        _atomic_write(staging_dir / DAY_FILE_ARM_A, arm_a_bytes)
        if fault_injector is not None:
            fault_injector("after_arm_a_write")
        _atomic_write(staging_dir / DAY_FILE_ARM_B, arm_b_bytes)
        if fault_injector is not None:
            fault_injector("before_checkpoint_write")
        _atomic_write(staging_dir / DAY_FILE_CHECKPOINT, checkpoint_bytes)
        os.replace(staging_dir, final_dir)
        return checkpoint_record

    def load_latest_runtime(self) -> dict[str, Any] | None:
        records = self._verify_chain()
        if not records:
            return None
        checkpoint = records[-1]
        day = checkpoint["last_completed_engine_day"]
        day_dir = self._day_dir(day)
        arm_a_runtime = self._read_json(day_dir / DAY_FILE_ARM_A)
        arm_b_runtime = self._read_json(day_dir / DAY_FILE_ARM_B)
        return {
            "day": day,
            "checkpoint": checkpoint,
            "arm_a_runtime": arm_a_runtime,
            "arm_b_runtime": arm_b_runtime,
        }


def verify_forward_store(
    root: str | Path,
    expected_activation_manifest_sha256: str,
    expected_collector_commit: str,
) -> dict[str, Any]:
    if not _valid_sha(expected_activation_manifest_sha256):
        raise V7ForwardPersistenceBlocked("ACTIVATION_MANIFEST_SHA_INVALID")
    if not _valid_commit(expected_collector_commit):
        raise V7ForwardPersistenceBlocked("COLLECTOR_COMMIT_INVALID")
    store = ForwardStudyStore(root)
    records = store._verify_chain(
        expected_activation_manifest_sha256=expected_activation_manifest_sha256,
        expected_collector_commit=expected_collector_commit,
    )
    return {
        "status": "PASS",
        "verified_days": [record["last_completed_engine_day"] for record in records],
        "day_count": len(records),
        "latest_checkpoint_sha256": records[-1]["current_checkpoint_sha256"] if records else None,
    }


def load_latest_runtime(root: str | Path) -> dict[str, Any] | None:
    return ForwardStudyStore(root).load_latest_runtime()
