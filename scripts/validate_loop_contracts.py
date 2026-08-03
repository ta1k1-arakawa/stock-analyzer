"""Read-only validator for the Phase A loop-control bootstrap files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATES = frozenset({
    "NEW", "PLANNED", "READY", "IMPLEMENTING", "VERIFYING", "RETRY_ALLOWED",
    "HUMAN_GATE", "ACCEPTED", "REJECTED", "BLOCKED", "CANCELLED", "DONE",
})
PHASE_A_NEXT_STATES = ("PLANNED", "CANCELLED", "HUMAN_GATE")
EMPTY_TASK_HASH = hashlib.sha256(b"").hexdigest()
REQUIRED_CONTROL_FILES = (
    "LOOP_SPEC.md",
    "loop_state.json",
    "evaluation_contract.json",
    "loop_history.jsonl",
    "human_approvals.jsonl",
    "PHASE_A_MANUAL_CHECKLIST.md",
    "schemas/loop_state.schema.json",
    "schemas/evaluation_contract.schema.json",
    "schemas/loop_history_event.schema.json",
    "schemas/human_approval.schema.json",
)
PROHIBITED_PHASE_A_PATHS = (
    "scripts/run_once.py",
    "scripts/run_loop.py",
    "scripts/loop_runner.py",
    "scripts/schedule_loop.py",
    "loop_control/run_once.py",
    "loop_control/lock.py",
)
SECRET_PATTERNS = (
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bJQUANTS_API_KEY\s*=\s*[^\s]+"),
)


class ValidationFailure(ValueError):
    """Raised when any Phase A contract rule is not satisfied."""


@dataclass(frozen=True)
class ValidationResult:
    summary_hash: str
    checked_files: tuple[str, ...]
    history_count: int
    approval_count: int


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"invalid JSON: {path.as_posix()}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValidationFailure(f"cannot read JSONL: {path.as_posix()}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationFailure(f"invalid JSONL line {line_number}: {path.name}") from exc
        if not isinstance(value, dict):
            raise ValidationFailure(f"JSONL line {line_number} is not an object: {path.name}")
        records.append(value)
    return records


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    reference = schema.get("$ref")
    if reference is None:
        return schema
    if not reference.startswith("#/$defs/"):
        raise ValidationFailure(f"unsupported schema reference: {reference}")
    name = reference.removeprefix("#/$defs/")
    try:
        return root["$defs"][name]
    except KeyError as exc:
        raise ValidationFailure(f"missing schema definition: {name}") from exc


def _is_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValidationFailure(f"unsupported schema type: {expected}")


def _is_datetime(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _validate_instance(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str) -> None:
    schema = _resolve(schema, root)
    if "anyOf" in schema:
        errors: list[ValidationFailure] = []
        for option in schema["anyOf"]:
            try:
                _validate_instance(value, option, root, path)
                return
            except ValidationFailure as exc:
                errors.append(exc)
        raise ValidationFailure(f"{path}: no allowed schema alternative matched") from errors[-1]
    expected_type = schema.get("type")
    if expected_type is not None and not _is_type(value, expected_type):
        raise ValidationFailure(f"{path}: expected {expected_type}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValidationFailure(f"{path}: value is not in enum")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValidationFailure(f"{path}: string is too short")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            raise ValidationFailure(f"{path}: string does not match pattern")
        if schema.get("format") == "date-time" and not _is_datetime(value):
            raise ValidationFailure(f"{path}: invalid ISO 8601 datetime")
    if isinstance(value, int) and not isinstance(value, bool) and value < schema.get("minimum", value):
        raise ValidationFailure(f"{path}: number is below minimum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValidationFailure(f"{path}: array is too short")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            raise ValidationFailure(f"{path}: array items must be unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_instance(item, item_schema, root, f"{path}[{index}]")
    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            raise ValidationFailure(f"{path}: missing required fields: {', '.join(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ValidationFailure(f"{path}: unknown fields: {', '.join(unknown)}")
        for name, child_schema in properties.items():
            if name in value:
                _validate_instance(value[name], child_schema, root, f"{path}.{name}")


def _sha256_normalized_text(path: Path) -> str:
    """Hash UTF-8 text after universal-newline normalization for stable receipts."""
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _require_phase_a_semantics(
    state: dict[str, Any], contract: dict[str, Any], history: list[dict[str, Any]],
    approvals: list[dict[str, Any],], spec_text: str, root: Path,
) -> None:
    if state["current_state"] != "NEW":
        raise ValidationFailure("Phase A current_state must remain NEW")
    if tuple(state["allowed_next_states"]) != PHASE_A_NEXT_STATES:
        raise ValidationFailure("Phase A allowed_next_states are not fixed")
    if state["current_task"] != "" or state["task_hash"] != EMPTY_TASK_HASH:
        raise ValidationFailure("Phase A must not register an executable task")
    if state["attempt"] != 0 or state["max_attempts"] != 0:
        raise ValidationFailure("Phase A attempts must be zero")
    if state["human_gate"] != {
        "required": False, "approval_id": None, "requested_action": None, "return_state": None,
    }:
        raise ValidationFailure("Phase A must not request or consume approval")
    if state["last_verified_commit"] is not None:
        raise ValidationFailure("Phase A last_verified_commit must be null")
    if contract["active"] is not False:
        raise ValidationFailure("Phase A evaluation contract must be inactive")
    if contract["task_hash"] != state["task_hash"]:
        raise ValidationFailure("state and contract task_hash differ")
    for label, budget in (("state", state["budget_remaining"]), ("contract", contract["budget"])):
        if any(value != 0 for value in budget.values()):
            raise ValidationFailure(f"Phase A {label} budget must be all zero")
    if contract["allowed_network_hosts"]:
        raise ValidationFailure("Phase A allows no network hosts")
    required_closed = (
        "research_status: CLOSED", "deployment_status: NO_CANDIDATE", "shadow_status: DISABLED",
        "paid_data_decision: DO_NOT_PURCHASE", "further_loop_on_same_data: PROHIBITED",
        "2db8e08833e8fc4b96e93c36e0f1b2fc74c5f158",
    )
    if any(marker not in spec_text for marker in required_closed):
        raise ValidationFailure("stock research closure status is missing or changed")
    if len(history) != 1:
        raise ValidationFailure("Phase A history must contain exactly one initialization event")
    event = history[0]
    if event["event_type"] != "INITIALIZED" or event["start_state"] != "NEW" or event["end_state"] != "NEW":
        raise ValidationFailure("Phase A history is not an initialization event")
    if event["state_transition"] is not None or event["output_commit"] is not None:
        raise ValidationFailure("Phase A initialization must not transition state or set output_commit")
    if event["loop_id"] != state["loop_id"] or event["task_hash"] != state["task_hash"]:
        raise ValidationFailure("Phase A history does not match state")
    if event["input_commit"] != state["base_commit"]:
        raise ValidationFailure("Phase A history input_commit does not match base_commit")
    if event["network_calls"] != 0 or event["model_fits"] != 0 or event["evaluations"] != 0:
        raise ValidationFailure("Phase A initialization cannot record execution activity")
    for approval in approvals:
        if approval["used"]:
            raise ValidationFailure("used approval cannot be reused in Phase A")
        expiry = datetime.fromisoformat(approval["expires_at"].replace("Z", "+00:00"))
        if expiry <= datetime.now(timezone.utc):
            raise ValidationFailure("expired approval cannot be used in Phase A")
        raise ValidationFailure("Phase A must not contain a human approval record")
    for relative in PROHIBITED_PHASE_A_PATHS:
        if (root / relative).exists():
            raise ValidationFailure(f"Phase A runner, scheduler, or lock is present: {relative}")


def _check_secret_patterns(paths: tuple[Path, ...]) -> None:
    for path in paths:
        if path.suffix not in {".json", ".jsonl", ".md", ".py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise ValidationFailure(f"potential secret detected: {path.as_posix()}")


def validate(root: Path) -> ValidationResult:
    """Validate without mutating files, Git state, environment, or network state."""
    root = root.resolve()
    control = root / "loop_control"
    files = tuple(control / relative for relative in REQUIRED_CONTROL_FILES)
    missing = [path.relative_to(root).as_posix() for path in files if not path.is_file()]
    if missing:
        raise ValidationFailure(f"missing required files: {', '.join(missing)}")
    schemas = {
        "state": _read_json(control / "schemas/loop_state.schema.json"),
        "contract": _read_json(control / "schemas/evaluation_contract.schema.json"),
        "history": _read_json(control / "schemas/loop_history_event.schema.json"),
        "approval": _read_json(control / "schemas/human_approval.schema.json"),
    }
    state = _read_json(control / "loop_state.json")
    contract = _read_json(control / "evaluation_contract.json")
    history = _read_jsonl(control / "loop_history.jsonl")
    approvals = _read_jsonl(control / "human_approvals.jsonl")
    _validate_instance(state, schemas["state"], schemas["state"], "loop_state")
    _validate_instance(contract, schemas["contract"], schemas["contract"], "evaluation_contract")
    for index, event in enumerate(history):
        _validate_instance(event, schemas["history"], schemas["history"], f"history[{index}]")
    for index, approval in enumerate(approvals):
        _validate_instance(approval, schemas["approval"], schemas["approval"], f"approval[{index}]")
    spec_path = control / "LOOP_SPEC.md"
    _require_phase_a_semantics(state, contract, history, approvals, spec_path.read_text(encoding="utf-8"), root)
    checked_paths = files + (root / "scripts/validate_loop_contracts.py",)
    _check_secret_patterns(checked_paths)
    canonical = {
        "approval_count": len(approvals),
        "contract_task_hash": contract["task_hash"],
        "current_state": state["current_state"],
        "file_hashes": {
            path.relative_to(root).as_posix(): _sha256_normalized_text(path)
            for path in checked_paths
        },
        "history_count": len(history),
        "task_hash": state["task_hash"],
    }
    summary_hash = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ValidationResult(
        summary_hash=summary_hash,
        checked_files=tuple(path.relative_to(root).as_posix() for path in checked_paths),
        history_count=len(history),
        approval_count=len(approvals),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only Phase A loop-control validator")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        result = validate(args.root)
    except ValidationFailure as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        "PASS: "
        f"history_count={result.history_count} approval_count={result.approval_count} "
        f"summary_hash={result.summary_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
