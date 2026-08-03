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
STATIC_ALLOWED_TRANSITIONS = {
    "NEW": ("PLANNED", "CANCELLED", "HUMAN_GATE"),
    "PLANNED": ("READY", "BLOCKED", "CANCELLED", "HUMAN_GATE"),
    "READY": ("IMPLEMENTING", "BLOCKED", "CANCELLED", "HUMAN_GATE"),
    "IMPLEMENTING": ("VERIFYING", "BLOCKED", "CANCELLED", "HUMAN_GATE"),
    "VERIFYING": ("ACCEPTED", "REJECTED", "RETRY_ALLOWED", "BLOCKED", "HUMAN_GATE"),
    "RETRY_ALLOWED": ("IMPLEMENTING", "BLOCKED", "CANCELLED", "HUMAN_GATE"),
    "ACCEPTED": ("DONE", "HUMAN_GATE"),
    "REJECTED": ("HUMAN_GATE",),
    "BLOCKED": ("HUMAN_GATE",),
    "CANCELLED": (),
    "DONE": (),
}
PHASE_A_NEXT_STATES = STATIC_ALLOWED_TRANSITIONS["NEW"]
EMPTY_TASK_HASH = hashlib.sha256(b"").hexdigest()
BOOTSTRAP_LOOP_ID = "phase-a-bootstrap"
BOOTSTRAP_ORIGIN_COMMIT = "c8552e30539f062fa76c4ac77d767039b6a7903e"
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


def allowed_next_states(state: str, return_state: str | None = None) -> tuple[str, ...]:
    """Return the fixed, deterministic transition order for one current state."""
    if state == "HUMAN_GATE":
        if return_state not in STATES or return_state in {"CANCELLED", "BLOCKED"}:
            raise ValidationFailure("HUMAN_GATE requires a non-terminal return_state")
        return (return_state, "CANCELLED", "BLOCKED")
    return STATIC_ALLOWED_TRANSITIONS[state]


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


def _require_closed_stock_research(spec_text: str, root: Path) -> None:
    required_closed = (
        "research_status: CLOSED", "deployment_status: NO_CANDIDATE", "shadow_status: DISABLED",
        "paid_data_decision: DO_NOT_PURCHASE", "further_loop_on_same_data: PROHIBITED",
        "2db8e08833e8fc4b96e93c36e0f1b2fc74c5f158",
    )
    if any(marker not in spec_text for marker in required_closed):
        raise ValidationFailure("stock research closure status is missing or changed")
    for relative in PROHIBITED_PHASE_A_PATHS:
        if (root / relative).exists():
            raise ValidationFailure(f"Phase A runner, scheduler, or lock is present: {relative}")


def _as_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_initialization(event: dict[str, Any], state: dict[str, Any]) -> None:
    if event["event_type"] != "INITIALIZED" or event["start_state"] != "NEW" or event["end_state"] != "NEW":
        raise ValidationFailure("first history event must be INITIALIZED NEW->NEW")
    if event["state_transition"] is not None or event["output_commit"] is not None:
        raise ValidationFailure("initialization must have null state_transition and output_commit")
    if event["loop_id"] != state["loop_id"]:
        raise ValidationFailure("initialization loop_id does not match loop state")
    expected_origin = BOOTSTRAP_ORIGIN_COMMIT if state["loop_id"] == BOOTSTRAP_LOOP_ID else state["base_commit"]
    if event["input_commit"] != expected_origin:
        raise ValidationFailure("initialization input_commit does not match bootstrap origin")
    if event["task_hash"] != EMPTY_TASK_HASH:
        raise ValidationFailure("initialization must use empty task hash")


def _validate_bootstrap(
    state: dict[str, Any], contract: dict[str, Any], history: list[dict[str, Any]],
    approvals: list[dict[str, Any]],
) -> None:
    if state["current_state"] != "NEW" or tuple(state["allowed_next_states"]) != PHASE_A_NEXT_STATES:
        raise ValidationFailure("bootstrap state must be NEW with fixed allowed_next_states")
    if state["base_commit"] != BOOTSTRAP_ORIGIN_COMMIT:
        raise ValidationFailure("bootstrap base_commit must equal bootstrap origin")
    if state["current_task"] != "" or state["task_hash"] != EMPTY_TASK_HASH:
        raise ValidationFailure("bootstrap must not register an executable task")
    if state["attempt"] != 0 or state["max_attempts"] != 0 or state["last_verified_commit"] is not None:
        raise ValidationFailure("bootstrap attempts and last_verified_commit must be empty")
    if state["human_gate"] != {"required": False, "approval_id": None, "requested_action": None, "return_state": None}:
        raise ValidationFailure("bootstrap must not request or consume approval")
    if contract["active"] is not False or contract["task_hash"] != EMPTY_TASK_HASH:
        raise ValidationFailure("bootstrap contract task_hash must be empty and inactive")
    if any(value != 0 for value in state["budget_remaining"].values()) or any(value != 0 for value in contract["budget"].values()):
        raise ValidationFailure("bootstrap budgets must be all zero")
    if contract["allowed_network_hosts"]:
        raise ValidationFailure("bootstrap allows no network hosts")
    if approvals:
        for approval in approvals:
            if approval["used"]:
                raise ValidationFailure("used approval cannot be reused in bootstrap")
            if _as_datetime(approval["expires_at"]) <= datetime.now(timezone.utc):
                raise ValidationFailure("expired approval cannot be used in bootstrap")
        raise ValidationFailure("bootstrap must not contain a human approval record")
    if len(history) != 1:
        raise ValidationFailure("bootstrap requires one initialization event")
    _validate_initialization(history[0], state)


def _validate_approvals(
    approvals: list[dict[str, Any]], state: dict[str, Any], history: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    usage: dict[str, int] = {}
    for approval in approvals:
        approval_id = approval["approval_id"]
        if approval_id in by_id:
            raise ValidationFailure("approval_id is duplicated")
        if approval["loop_id"] != state["loop_id"] or approval["task_hash"] != state["task_hash"]:
            raise ValidationFailure("approval loop_id or task_hash does not match state")
        approved_at = _as_datetime(approval["approved_at"])
        expires_at = _as_datetime(approval["expires_at"])
        if expires_at <= approved_at:
            raise ValidationFailure("approval expires_at must be after approved_at")
        if approval["used"] and approval["used_at"] is None:
            raise ValidationFailure("used approval requires used_at")
        if not approval["used"] and approval["used_at"] is not None:
            raise ValidationFailure("unused approval must have null used_at")
        if approval["used"] and _as_datetime(approval["used_at"]) < approved_at:
            raise ValidationFailure("approval used_at precedes approved_at")
        by_id[approval_id] = approval
        usage[approval_id] = 0
    for event in history:
        approval_id = event["human_approval_id"]
        if approval_id is not None:
            if approval_id not in by_id:
                raise ValidationFailure("history references missing human approval")
            usage[approval_id] += 1
    for approval_id, count in usage.items():
        if count > 1:
            raise ValidationFailure("human approval is reused by multiple events")
        if by_id[approval_id]["used"] != (count == 1):
            raise ValidationFailure("approval used flag does not match history usage")
    return by_id


def _validate_history(
    state: dict[str, Any], history: list[dict[str, Any]], approvals: dict[str, dict[str, Any]],
) -> None:
    if not history:
        raise ValidationFailure("history cannot be empty")
    _validate_initialization(history[0], state)
    run_ids: set[str] = set()
    previous: dict[str, Any] | None = None
    previous_time: datetime | None = None
    for index, event in enumerate(history):
        if event["run_id"] in run_ids:
            raise ValidationFailure("run_id is duplicated")
        run_ids.add(event["run_id"])
        timestamp = _as_datetime(event["timestamp"])
        if previous_time is not None and timestamp < previous_time:
            raise ValidationFailure("history timestamp is decreasing")
        previous_time = timestamp
        if event["loop_id"] != state["loop_id"]:
            raise ValidationFailure("history loop_id does not match state")
        if any(event[name] < 0 for name in ("network_calls", "model_fits", "evaluations")):
            raise ValidationFailure("history activity counts must be nonnegative")
        if event["output_commit"] is not None and re.fullmatch(r"[0-9a-f]{40}", event["output_commit"]) is None:
            raise ValidationFailure("output_commit must be null or lowercase commit hash")
        if index == 0:
            previous = event
            continue
        if event["input_commit"] != state["base_commit"]:
            raise ValidationFailure("post-initialization input_commit does not match task base_commit")
        if index == 1 and (
            event["event_type"] != "STATE_TRANSITION"
            or event["start_state"] != "NEW"
            or event["end_state"] != "PLANNED"
        ):
            raise ValidationFailure("first state transition must be NEW->PLANNED")
        if previous is None:
            raise AssertionError("unreachable")
        if previous["end_state"] in {"DONE", "CANCELLED"}:
            raise ValidationFailure("no event may follow DONE or CANCELLED")
        if event["start_state"] != previous["end_state"]:
            raise ValidationFailure("history start_state does not match previous end_state")
        if event["task_hash"] != state["task_hash"]:
            raise ValidationFailure("task-registered history event has mismatched task_hash")
        if event["event_type"] == "EVIDENCE":
            if event["start_state"] != event["end_state"] or event["state_transition"] is not None:
                raise ValidationFailure("EVIDENCE event cannot change state")
        elif event["event_type"] == "STATE_TRANSITION":
            transition = event["state_transition"]
            if transition is None:
                raise ValidationFailure("STATE_TRANSITION requires state_transition")
            if transition["from"] != event["start_state"] or transition["to"] != event["end_state"]:
                raise ValidationFailure("state_transition does not match start/end state")
            if event["end_state"] not in allowed_next_states(event["start_state"], event.get("gate_return_state")):
                raise ValidationFailure("state transition is not allowed")
            if event["start_state"] == "HUMAN_GATE":
                approval_id = event["human_approval_id"]
                if approval_id is None or approval_id not in approvals:
                    raise ValidationFailure("HUMAN_GATE transition requires a matching approval")
                approval = approvals[approval_id]
                if not approval["used"] or event.get("gate_requested_action") != approval["approved_action"]:
                    raise ValidationFailure("HUMAN_GATE approval action is not valid")
                if event.get("gate_return_state") != event["end_state"] or approval["permitted_return_state"] != event["end_state"]:
                    raise ValidationFailure("HUMAN_GATE approval return_state is not valid")
                event_time = _as_datetime(event["timestamp"])
                if not (_as_datetime(approval["approved_at"]) <= event_time <= _as_datetime(approval["expires_at"])):
                    raise ValidationFailure("HUMAN_GATE approval is outside its validity period")
        else:
            raise ValidationFailure("only INITIALIZED, STATE_TRANSITION, and EVIDENCE are allowed")
        previous = event
    if history[-1]["end_state"] != state["current_state"]:
        raise ValidationFailure("last history state does not match loop_state")


def _validate_manual(
    state: dict[str, Any], contract: dict[str, Any], history: list[dict[str, Any]], approvals: list[dict[str, Any]],
) -> None:
    if state["current_state"] == "NEW":
        raise ValidationFailure("task-registered NEW is not allowed")
    if not state["current_task"] or state["task_hash"] != hashlib.sha256(state["current_task"].encode("utf-8")).hexdigest():
        raise ValidationFailure("manual task and task_hash must match")
    if contract["active"] is not True or contract["task_hash"] != state["task_hash"] or contract["contract_version"] < 1:
        raise ValidationFailure("manual loop requires an active matching contract")
    for name in ("pass_conditions", "failure_conditions", "allowed_files", "forbidden_files", "required_tests"):
        if not contract[name]:
            raise ValidationFailure(f"manual contract {name} cannot be empty")
    if state["attempt"] > state["max_attempts"]:
        raise ValidationFailure("attempt exceeds max_attempts")
    for name, remaining in state["budget_remaining"].items():
        if remaining > contract["budget"][name]:
            raise ValidationFailure("budget_remaining exceeds contract budget")
    gate = state["human_gate"]
    if state["current_state"] == "HUMAN_GATE":
        if not gate["required"] or gate["requested_action"] is None or gate["return_state"] is None:
            raise ValidationFailure("HUMAN_GATE requires requested_action and return_state")
    elif gate != {"required": False, "approval_id": None, "requested_action": None, "return_state": None}:
        raise ValidationFailure("non-HUMAN_GATE state must have an empty human_gate")
    expected_next = allowed_next_states(state["current_state"], gate["return_state"])
    if tuple(state["allowed_next_states"]) != expected_next:
        raise ValidationFailure("allowed_next_states do not match fixed transition table")
    approvals_by_id = _validate_approvals(approvals, state, history)
    _validate_history(state, history, approvals_by_id)
    last = history[-1]
    current = state["current_state"]
    if current == "PLANNED" and (state["attempt"] != 0 or state["last_verified_commit"] is not None):
        raise ValidationFailure("PLANNED requires attempt zero and null last_verified_commit")
    if current == "READY" and not any(event["end_state"] == "PLANNED" for event in history):
        raise ValidationFailure("READY requires a PLANNED history event")
    if current == "IMPLEMENTING" and last["start_state"] not in {"READY", "RETRY_ALLOWED"}:
        raise ValidationFailure("IMPLEMENTING must follow READY or RETRY_ALLOWED")
    if current == "VERIFYING" and last["start_state"] != "IMPLEMENTING":
        raise ValidationFailure("VERIFYING must follow IMPLEMENTING")
    if current == "RETRY_ALLOWED" and (last["start_state"] != "VERIFYING" or state["attempt"] >= state["max_attempts"]):
        raise ValidationFailure("RETRY_ALLOWED requires VERIFYING predecessor and remaining attempt")
    if current == "ACCEPTED" and (last["start_state"] != "VERIFYING" or last["verification_result"] != "PASS"):
        raise ValidationFailure("ACCEPTED requires PASS evidence from VERIFYING")
    if current == "REJECTED" and (last["start_state"] != "VERIFYING" or last["verification_result"] != "FAIL"):
        raise ValidationFailure("REJECTED requires FAIL evidence from VERIFYING")
    if current == "BLOCKED" and last["verification_result"] != "BLOCKED" and not last["failure_reason"]:
        raise ValidationFailure("BLOCKED requires BLOCKED evidence or failure_reason")
    if current == "DONE" and last["start_state"] != "ACCEPTED":
        raise ValidationFailure("DONE must follow ACCEPTED")


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
    _require_closed_stock_research(spec_path.read_text(encoding="utf-8"), root)
    if state["current_state"] == "NEW":
        _validate_bootstrap(state, contract, history, approvals)
    else:
        _validate_manual(state, contract, history, approvals)
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
