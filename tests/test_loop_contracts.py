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
