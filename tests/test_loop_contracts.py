from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from pathlib import Path

import pytest

from scripts import validate_loop_contracts as validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_copy(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "phase-a"
    shutil.copytree(PROJECT_ROOT / "loop_control", root / "loop_control")
    (root / "scripts").mkdir(parents=True)
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "validate_loop_contracts.py",
        root / "scripts" / "validate_loop_contracts.py",
    )
    return root


def _read_json(root: Path, name: str) -> dict:
    return json.loads((root / "loop_control" / name).read_text(encoding="utf-8"))


def _write_json(root: Path, name: str, value: dict) -> None:
    (root / "loop_control" / name).write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _approval(*, used: bool = False, expires_at: str = "2099-01-01T00:00:00Z") -> dict:
    return {
        "approval_id": "approval-1",
        "loop_id": "phase-a-bootstrap",
        "task_hash": validator.EMPTY_TASK_HASH,
        "approved_action": "fictional-action",
        "approver_id": "human",
        "approved_at": "2026-08-03T00:00:00Z",
        "expires_at": expires_at,
        "used": used,
        "used_at": "2026-08-03T00:01:00Z" if used else None,
        "permitted_return_state": "PLANNED",
        "corresponding_commit": None,
    }


def _write_approval(root: Path, approval: dict) -> None:
    (root / "loop_control" / "human_approvals.jsonl").write_text(
        json.dumps(approval, sort_keys=True) + "\n", encoding="utf-8"
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_phase_a_bootstrap_validates(tmp_path: Path) -> None:
    result = validator.validate(_bootstrap_copy(tmp_path))
    assert result.history_count == 1
    assert result.approval_count == 0
    assert len(result.summary_hash) == 64


def test_schema_documents_are_valid_json() -> None:
    schema_dir = PROJECT_ROOT / "loop_control" / "schemas"
    schema_paths = sorted(schema_dir.glob("*.schema.json"))
    assert len(schema_paths) == 4
    for path in schema_paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False


def test_unknown_state_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    state = _read_json(root, "loop_state.json")
    state["current_state"] = "UNKNOWN"
    _write_json(root, "loop_state.json", state)
    with pytest.raises(validator.ValidationFailure):
        validator.validate(root)


def test_unallowed_next_state_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    state = _read_json(root, "loop_state.json")
    state["allowed_next_states"] = ["DONE"]
    _write_json(root, "loop_state.json", state)
    with pytest.raises(validator.ValidationFailure, match="allowed_next_states"):
        validator.validate(root)


def test_task_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    contract = _read_json(root, "evaluation_contract.json")
    contract["task_hash"] = "0" * 64
    _write_json(root, "evaluation_contract.json", contract)
    with pytest.raises(validator.ValidationFailure, match="task_hash"):
        validator.validate(root)


def test_invalid_commit_hash_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    state = _read_json(root, "loop_state.json")
    state["base_commit"] = "not-a-commit"
    _write_json(root, "loop_state.json", state)
    with pytest.raises(validator.ValidationFailure, match="pattern"):
        validator.validate(root)


def test_negative_budget_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    state = _read_json(root, "loop_state.json")
    state["budget_remaining"]["max_api_calls"] = -1
    _write_json(root, "loop_state.json", state)
    with pytest.raises(validator.ValidationFailure, match="minimum"):
        validator.validate(root)


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    state = _read_json(root, "loop_state.json")
    state["surprise"] = True
    _write_json(root, "loop_state.json", state)
    with pytest.raises(validator.ValidationFailure, match="unknown fields"):
        validator.validate(root)


def test_invalid_jsonl_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    (root / "loop_control" / "loop_history.jsonl").write_text("not-json\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure, match="invalid JSONL"):
        validator.validate(root)


def test_fictional_approval_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    _write_approval(root, _approval())
    with pytest.raises(validator.ValidationFailure, match="must not contain"):
        validator.validate(root)


def test_used_approval_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    _write_approval(root, _approval(used=True))
    with pytest.raises(validator.ValidationFailure, match="used approval"):
        validator.validate(root)


def test_expired_approval_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    _write_approval(root, _approval(expires_at="2000-01-01T00:00:00Z"))
    with pytest.raises(validator.ValidationFailure, match="expired approval"):
        validator.validate(root)


def test_active_contract_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    contract = _read_json(root, "evaluation_contract.json")
    contract["active"] = True
    _write_json(root, "evaluation_contract.json", contract)
    with pytest.raises(validator.ValidationFailure, match="inactive"):
        validator.validate(root)


@pytest.mark.parametrize("budget_name", ["max_model_fits", "max_api_calls"])
def test_nonzero_phase_a_budget_is_rejected(tmp_path: Path, budget_name: str) -> None:
    root = _bootstrap_copy(tmp_path)
    contract = _read_json(root, "evaluation_contract.json")
    contract["budget"][budget_name] = 1
    _write_json(root, "evaluation_contract.json", contract)
    with pytest.raises(validator.ValidationFailure, match="budget"):
        validator.validate(root)


def test_reopened_stock_research_is_rejected(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    spec = root / "loop_control" / "LOOP_SPEC.md"
    spec.write_text(
        spec.read_text(encoding="utf-8").replace("research_status: CLOSED", "research_status: OPEN"),
        encoding="utf-8",
    )
    with pytest.raises(validator.ValidationFailure, match="closure status"):
        validator.validate(root)


def test_validator_is_read_only(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    before = _tree_hashes(root)
    validator.validate(root)
    assert _tree_hashes(root) == before


def test_validator_uses_no_subprocess_or_network_modules() -> None:
    source = inspect.getsource(validator)
    assert "import subprocess" not in source
    assert "import socket" not in source
    assert "import requests" not in source
    assert "urllib.request" not in source
    assert "os.environ" not in source


def test_validator_summary_hash_is_deterministic(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    first = validator.validate(root).summary_hash
    second = validator.validate(root).summary_hash
    assert first == second


def test_phase_a_state_remains_new() -> None:
    state = json.loads((PROJECT_ROOT / "loop_control" / "loop_state.json").read_text(encoding="utf-8"))
    assert state["current_state"] == "NEW"
    assert state["current_task"] == ""
    assert state["task_hash"] == validator.EMPTY_TASK_HASH


def _transition(
    start: str, end: str, index: int, task_hash: str, *, gate: bool = False,
    input_commit: str = validator.BOOTSTRAP_ORIGIN_COMMIT,
) -> dict:
    result = "NOT_RUN"
    failure_reason = None
    if end == "ACCEPTED":
        result = "PASS"
    elif end == "REJECTED":
        result = "FAIL"
    elif end == "BLOCKED":
        result = "BLOCKED"
        failure_reason = "recorded blocker"
    event = {
        "run_id": f"manual-{index}", "loop_id": "phase-a-bootstrap",
        "event_type": "STATE_TRANSITION", "start_state": start, "end_state": end,
        "input_commit": input_commit,
        "output_commit": None, "task_hash": task_hash,
        "command_summary": "manual record only", "changed_files": [], "test_results": {},
        "verification_result": result, "state_transition": {"from": start, "to": end},
        "failure_reason": failure_reason, "human_approval_id": None,
        "network_calls": 0, "model_fits": 0, "evaluations": 0,
        "timestamp": f"2026-08-03T00:{index:02d}:00Z",
    }
    if gate:
        event.update({
            "human_approval_id": "approval-gate",
            "gate_requested_action": "resume-approved-pilot",
            "gate_return_state": end,
        })
    return event


def _path_to(state: str) -> list[tuple[str, str]]:
    paths = {
        "NEW": [],
        "PLANNED": [("NEW", "PLANNED")],
        "READY": [("NEW", "PLANNED"), ("PLANNED", "READY")],
        "IMPLEMENTING": [("NEW", "PLANNED"), ("PLANNED", "READY"), ("READY", "IMPLEMENTING")],
        "VERIFYING": [("NEW", "PLANNED"), ("PLANNED", "READY"), ("READY", "IMPLEMENTING"), ("IMPLEMENTING", "VERIFYING")],
        "RETRY_ALLOWED": [("NEW", "PLANNED"), ("PLANNED", "READY"), ("READY", "IMPLEMENTING"), ("IMPLEMENTING", "VERIFYING"), ("VERIFYING", "RETRY_ALLOWED")],
        "ACCEPTED": [("NEW", "PLANNED"), ("PLANNED", "READY"), ("READY", "IMPLEMENTING"), ("IMPLEMENTING", "VERIFYING"), ("VERIFYING", "ACCEPTED")],
        "REJECTED": [("NEW", "PLANNED"), ("PLANNED", "READY"), ("READY", "IMPLEMENTING"), ("IMPLEMENTING", "VERIFYING"), ("VERIFYING", "REJECTED")],
        "BLOCKED": [("NEW", "PLANNED"), ("PLANNED", "BLOCKED")],
        "HUMAN_GATE": [("NEW", "PLANNED"), ("PLANNED", "HUMAN_GATE")],
        "CANCELLED": [("NEW", "PLANNED"), ("PLANNED", "CANCELLED")],
        "DONE": [("NEW", "PLANNED"), ("PLANNED", "READY"), ("READY", "IMPLEMENTING"), ("IMPLEMENTING", "VERIFYING"), ("VERIFYING", "ACCEPTED"), ("ACCEPTED", "DONE")],
    }
    return paths[state]


def _manual_root(
    tmp_path: Path, transitions: list[tuple[str, str]],
    *, task_base: str = validator.BOOTSTRAP_ORIGIN_COMMIT,
) -> Path:
    root = _bootstrap_copy(tmp_path)
    task = "non-confidential documentation pilot"
    task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()
    state = _read_json(root, "loop_state.json")
    contract = _read_json(root, "evaluation_contract.json")
    contract["active"] = True
    contract["task_hash"] = task_hash
    for name in contract["budget"]:
        contract["budget"][name] = 1
        state["budget_remaining"][name] = 1
    state.update({
        "current_task": task, "task_hash": task_hash, "max_attempts": 1,
        "base_commit": task_base,
    })
    history_path = root / "loop_control" / "loop_history.jsonl"
    initial = json.loads(history_path.read_text(encoding="utf-8"))
    events = [initial]
    approvals: list[dict] = []
    for index, (start, end) in enumerate(transitions, start=1):
        gate = start == "HUMAN_GATE"
        events.append(_transition(start, end, index, task_hash, gate=gate, input_commit=task_base))
        if gate:
            approvals.append({
                "approval_id": "approval-gate", "loop_id": state["loop_id"], "task_hash": task_hash,
                "approved_action": "resume-approved-pilot", "approver_id": "human",
                "approved_at": "2026-08-03T00:00:30Z", "expires_at": "2099-01-01T00:00:00Z",
                "used": True, "used_at": f"2026-08-03T00:{index:02d}:00Z",
                "permitted_return_state": end, "corresponding_commit": None,
            })
    final_state = transitions[-1][1]
    state["current_state"] = final_state
    if final_state == "HUMAN_GATE":
        state["human_gate"] = {
            "required": True, "approval_id": None,
            "requested_action": "resume-approved-pilot", "return_state": "READY",
        }
        state["allowed_next_states"] = list(validator.allowed_next_states("HUMAN_GATE", "READY"))
    else:
        state["human_gate"] = {"required": False, "approval_id": None, "requested_action": None, "return_state": None}
        state["allowed_next_states"] = list(validator.allowed_next_states(final_state))
    _write_json(root, "loop_state.json", state)
    _write_json(root, "evaluation_contract.json", contract)
    history_path.write_text("\n".join(json.dumps(event, sort_keys=True) for event in events) + "\n", encoding="utf-8")
    (root / "loop_control" / "human_approvals.jsonl").write_text(
        "\n".join(json.dumps(approval, sort_keys=True) for approval in approvals) + ("\n" if approvals else ""),
        encoding="utf-8",
    )
    return root


def test_manual_planned_record_validates(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")])
    result = validator.validate(root)
    assert result.history_count == 2
    assert result.approval_count == 0


def test_manual_planned_record_allows_task_base_after_bootstrap_origin(tmp_path: Path) -> None:
    task_base = "eaf0e982646885e490f12e85c0ddd67ec2f9bbb4"
    root = _manual_root(tmp_path, [("NEW", "PLANNED")], task_base=task_base)
    state = _read_json(root, "loop_state.json")
    initial, planned = [
        json.loads(line)
        for line in (root / "loop_control" / "loop_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert initial["input_commit"] == validator.BOOTSTRAP_ORIGIN_COMMIT
    assert state["base_commit"] == task_base
    assert planned["input_commit"] == task_base
    validator.validate(root)


@pytest.mark.parametrize(
    "change",
    ["origin_rewritten", "origin_invalid", "origin_loop", "origin_state", "origin_transition"],
)
def test_initialization_origin_is_immutable(tmp_path: Path, change: str) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")], task_base="eaf0e982646885e490f12e85c0ddd67ec2f9bbb4")
    history_path = root / "loop_control" / "loop_history.jsonl"
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    if change == "origin_rewritten":
        events[0]["input_commit"] = "eaf0e982646885e490f12e85c0ddd67ec2f9bbb4"
    elif change == "origin_invalid":
        events[0]["input_commit"] = "invalid"
    elif change == "origin_loop":
        events[0]["loop_id"] = "other-loop"
    elif change == "origin_state":
        events[0]["end_state"] = "PLANNED"
    else:
        events[0]["state_transition"] = {"from": "NEW", "to": "NEW"}
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure):
        validator.validate(root)


def test_post_initialization_commit_must_equal_task_base(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")], task_base="eaf0e982646885e490f12e85c0ddd67ec2f9bbb4")
    history_path = root / "loop_control" / "loop_history.jsonl"
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    events[1]["input_commit"] = validator.BOOTSTRAP_ORIGIN_COMMIT
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure, match="task base_commit"):
        validator.validate(root)


def test_post_initialization_commit_cannot_change_mid_history(tmp_path: Path) -> None:
    task_base = "eaf0e982646885e490f12e85c0ddd67ec2f9bbb4"
    root = _manual_root(tmp_path, [("NEW", "PLANNED"), ("PLANNED", "READY")], task_base=task_base)
    history_path = root / "loop_control" / "loop_history.jsonl"
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    events[2]["input_commit"] = validator.BOOTSTRAP_ORIGIN_COMMIT
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure, match="task base_commit"):
        validator.validate(root)


@pytest.mark.parametrize("field", ["task_hash", "loop_id"])
def test_post_initialization_task_identity_cannot_change(tmp_path: Path, field: str) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")])
    history_path = root / "loop_control" / "loop_history.jsonl"
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    events[1][field] = ("0" * 64) if field == "task_hash" else "other-loop"
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure):
        validator.validate(root)


def test_first_post_initialization_event_must_plan(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, [("NEW", "READY")])
    with pytest.raises(validator.ValidationFailure, match="first state transition"):
        validator.validate(root)


def test_bootstrap_mode_requires_matching_origin_and_base(tmp_path: Path) -> None:
    root = _bootstrap_copy(tmp_path)
    state = _read_json(root, "loop_state.json")
    state["base_commit"] = "eaf0e982646885e490f12e85c0ddd67ec2f9bbb4"
    _write_json(root, "loop_state.json", state)
    with pytest.raises(validator.ValidationFailure, match="bootstrap base_commit"):
        validator.validate(root)


def test_all_static_allowed_transitions_validate(tmp_path: Path) -> None:
    checked = 0
    for source, targets in validator.STATIC_ALLOWED_TRANSITIONS.items():
        for target in targets:
            if source == "NEW" and target != "PLANNED":
                continue
            root = _manual_root(tmp_path / f"{source}-{target}", _path_to(source) + [(source, target)])
            validator.validate(root)
            checked += 1
    assert checked == sum(len(targets) for targets in validator.STATIC_ALLOWED_TRANSITIONS.values()) - 2


def test_human_gate_return_transition_validates(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED"), ("PLANNED", "HUMAN_GATE"), ("HUMAN_GATE", "READY")])
    validator.validate(root)


def test_all_forbidden_transitions_fail(tmp_path: Path) -> None:
    checked = 0
    for source in sorted(validator.STATES):
        allowed = set(validator.allowed_next_states(source, "READY"))
        for target in sorted(validator.STATES - allowed):
            root = _manual_root(tmp_path / f"bad-{source}-{target}", _path_to(source) + [(source, target)])
            if source == "HUMAN_GATE":
                history_path = root / "loop_control" / "loop_history.jsonl"
                events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
                events[-1]["gate_return_state"] = "READY"
                history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
                approvals_path = root / "loop_control" / "human_approvals.jsonl"
                approvals = [json.loads(line) for line in approvals_path.read_text(encoding="utf-8").splitlines()]
                approvals[0]["permitted_return_state"] = "READY"
                approvals_path.write_text(json.dumps(approvals[0]) + "\n", encoding="utf-8")
            with pytest.raises(validator.ValidationFailure):
                validator.validate(root)
            checked += 1
    assert checked > 0


def test_history_cannot_skip_a_state(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, [("NEW", "READY")])
    with pytest.raises(validator.ValidationFailure, match="first state transition"):
        validator.validate(root)


@pytest.mark.parametrize("terminal", ["DONE", "CANCELLED"])
def test_terminal_state_cannot_have_another_event(tmp_path: Path, terminal: str) -> None:
    root = _manual_root(tmp_path, _path_to(terminal) + [(terminal, "HUMAN_GATE")])
    with pytest.raises(validator.ValidationFailure, match="no event may follow"):
        validator.validate(root)


def test_rejected_cannot_return_to_planned(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, _path_to("REJECTED") + [("REJECTED", "PLANNED")])
    with pytest.raises(validator.ValidationFailure):
        validator.validate(root)


def test_blocked_cannot_return_to_implementing(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, _path_to("BLOCKED") + [("BLOCKED", "IMPLEMENTING")])
    with pytest.raises(validator.ValidationFailure):
        validator.validate(root)


def test_manual_task_hash_must_match(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")])
    state = _read_json(root, "loop_state.json")
    state["task_hash"] = "0" * 64
    _write_json(root, "loop_state.json", state)
    with pytest.raises(validator.ValidationFailure, match="task_hash"):
        validator.validate(root)


def test_manual_planned_requires_active_contract(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")])
    contract = _read_json(root, "evaluation_contract.json")
    contract["active"] = False
    _write_json(root, "evaluation_contract.json", contract)
    with pytest.raises(validator.ValidationFailure, match="active"):
        validator.validate(root)


@pytest.mark.parametrize("change", ["budget", "attempt"])
def test_manual_budget_and_attempt_limits(tmp_path: Path, change: str) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")])
    state = _read_json(root, "loop_state.json")
    if change == "budget":
        state["budget_remaining"]["max_api_calls"] = 2
    else:
        state["attempt"] = 2
    _write_json(root, "loop_state.json", state)
    with pytest.raises(validator.ValidationFailure):
        validator.validate(root)


def test_history_chain_run_ids_and_timestamps_are_checked(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED"), ("PLANNED", "READY")])
    history_path = root / "loop_control" / "loop_history.jsonl"
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    events[2]["run_id"] = events[1]["run_id"]
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure, match="run_id"):
        validator.validate(root)
    events[2]["run_id"] = "unique"
    events[2]["timestamp"] = "2026-08-02T00:00:00Z"
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure, match="timestamp"):
        validator.validate(root)


def test_evidence_event_preserves_state_and_transition_shape_is_checked(tmp_path: Path) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")])
    history_path = root / "loop_control" / "loop_history.jsonl"
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    evidence = dict(events[-1])
    evidence.update({
        "run_id": "evidence-1", "event_type": "EVIDENCE", "start_state": "PLANNED",
        "end_state": "PLANNED", "state_transition": None,
        "timestamp": "2026-08-03T00:02:00Z",
    })
    events.append(evidence)
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    validator.validate(root)
    events[-1]["state_transition"] = {"from": "PLANNED", "to": "READY"}
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure, match="EVIDENCE"):
        validator.validate(root)


@pytest.mark.parametrize("event_change", ["evidence_changes_state", "transition_is_null"])
def test_event_shape_rules_are_checked(tmp_path: Path, event_change: str) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")])
    history_path = root / "loop_control" / "loop_history.jsonl"
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    if event_change == "evidence_changes_state":
        events[1]["event_type"] = "EVIDENCE"
    else:
        events[1]["state_transition"] = None
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure):
        validator.validate(root)


@pytest.mark.parametrize("change", ["missing", "expired", "task", "action", "return", "reused"])
def test_human_gate_approval_rules(tmp_path: Path, change: str) -> None:
    root = _manual_root(tmp_path, [("NEW", "HUMAN_GATE"), ("HUMAN_GATE", "READY")])
    history_path = root / "loop_control" / "loop_history.jsonl"
    approvals_path = root / "loop_control" / "human_approvals.jsonl"
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    approvals = [json.loads(line) for line in approvals_path.read_text(encoding="utf-8").splitlines()]
    if change == "missing":
        events[2]["human_approval_id"] = None
    elif change == "expired":
        approvals[0]["expires_at"] = "2026-08-03T00:01:30Z"
    elif change == "task":
        approvals[0]["task_hash"] = "f" * 64
    elif change == "action":
        approvals[0]["approved_action"] = "other"
    elif change == "return":
        approvals[0]["permitted_return_state"] = "PLANNED"
    else:
        duplicate = dict(events[2])
        duplicate["run_id"] = "reused-approval"
        duplicate["start_state"] = "READY"
        duplicate["end_state"] = "HUMAN_GATE"
        duplicate["state_transition"] = {"from": "READY", "to": "HUMAN_GATE"}
        duplicate["gate_requested_action"] = None
        duplicate["gate_return_state"] = None
        duplicate["timestamp"] = "2026-08-03T00:03:00Z"
        events.append(duplicate)
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    approvals_path.write_text("\n".join(json.dumps(approval) for approval in approvals) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure):
        validator.validate(root)


@pytest.mark.parametrize("change", ["unused_with_timestamp", "used_without_timestamp", "used_before_approval"])
def test_approval_timestamp_semantics_are_checked(tmp_path: Path, change: str) -> None:
    root = _manual_root(tmp_path, [("NEW", "PLANNED")])
    approval = _approval()
    approval["task_hash"] = _read_json(root, "loop_state.json")["task_hash"]
    if change == "unused_with_timestamp":
        approval["used_at"] = "2026-08-03T00:01:00Z"
    elif change == "used_without_timestamp":
        approval["used"] = True
    else:
        approval["used"] = True
        approval["used_at"] = "2026-08-02T00:00:00Z"
    _write_approval(root, approval)
    with pytest.raises(validator.ValidationFailure):
        validator.validate(root)


def test_accept_reject_and_done_evidence_rules(tmp_path: Path) -> None:
    accepted = _manual_root(tmp_path / "accepted", _path_to("ACCEPTED"))
    validator.validate(accepted)
    done = _manual_root(tmp_path / "done", _path_to("DONE"))
    validator.validate(done)
    rejected = _manual_root(tmp_path / "rejected", _path_to("REJECTED"))
    validator.validate(rejected)
    history_path = rejected / "loop_control" / "loop_history.jsonl"
    events = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    events[-1]["verification_result"] = "PASS"
    history_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure, match="REJECTED requires"):
        validator.validate(rejected)


@pytest.mark.parametrize("relative", ["scripts/run_once.py", "scripts/schedule_loop.py", "loop_control/lock.py"])
def test_runner_scheduler_and_lock_are_rejected(tmp_path: Path, relative: str) -> None:
    root = _bootstrap_copy(tmp_path)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("forbidden", encoding="utf-8")
    with pytest.raises(validator.ValidationFailure, match="runner, scheduler, or lock"):
        validator.validate(root)
