# Phase A manual checklist

Phase A has no runner, automatic state update, lock, CI, scheduler, or external authority. This checklist is a human review aid and does not itself authorize a transition.

## Review before accepting a Phase A change

- [ ] `base_commit` is `c8552e30539f062fa76c4ac77d767039b6a7903e`.
- [ ] `current_state` is `NEW`.
- [ ] `current_task` is empty and `task_hash` is the SHA-256 of the empty UTF-8 string.
- [ ] `allowed_next_states` are only `PLANNED`, `CANCELLED`, and `HUMAN_GATE`.
- [ ] Every budget value is zero.
- [ ] `evaluation_contract.json` has `active: false`.
- [ ] There is no human approval record; `human_approvals.jsonl` is empty.
- [ ] The initialization history has one event and no state transition.
- [ ] `python scripts/validate_loop_contracts.py` reports `PASS` and record its summary hash.
- [ ] Re-running the validator produces the same summary hash.
- [ ] `pytest -q`, `python -m unittest discover -s tests -v`, and `git diff --check` pass.
- [ ] `git diff --name-only` shows only the Phase A allowed files.
- [ ] No forbidden file, raw data, secret, model, stock evaluation result, or shadow file changed.
- [ ] No runner, scheduler, GitHub Actions change, lock, automatic commit, or automatic push exists.
- [ ] `LOOP_SPEC.md` still says `research_status: CLOSED` and `shadow_status: DISABLED`.
- [ ] The worktree is clean after commit and only the intended Phase A branch was pushed.

## Before Phase B

A human must separately choose and approve a small, non-confidential, non-investment pilot task. The pilot must have a new immutable task and evaluation contract. Phase A does not authorize Phase B, stock-analyzer research, paid data, model training, backtesting, shadow activation, or any live order.

## Manual-record constraint before Phase B

Manual Phase A history records may use `output_commit: null`. A Git commit cannot safely contain its own final SHA before that commit exists, and the validator must never infer or fill this value from Git history. No follow-up commit may be created merely to backfill it. Before Phase B adds any `run_once` implementation, a separate human-approved design must choose the commit-evidence format.
