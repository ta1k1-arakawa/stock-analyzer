# Human-in-the-loop engineering loop design

## 0. Status and scope

### Confirmed facts

- The stock-analyzer research cycle is closed at commit `2db8e08833e8fc4b96e93c36e0f1b2fc74c5f158`.
- `research_status: CLOSED`
- `deployment_status: NO_CANDIDATE`
- `shadow_status: DISABLED`
- `paid_data_decision: DO_NOT_PURCHASE`
- `further_loop_on_same_data: PROHIBITED`
- This design is a reusable, small development-loop framework. It is not a stock-strategy engine.

### Purpose

Define a fail-closed, human-in-the-loop outer loop that reads repository-owned state, allows an agent to complete exactly one authorized step, independently verifies evidence, and then stops at a terminal state or a human approval gate.

### Non-goals

- Reopen, evaluate, modify, deploy, or schedule stock-analyzer research.
- Generate hypotheses, relax evaluation criteria, or optimize from observed results.
- Execute live orders, manage money, or handle brokerage credentials.
- Provide an unattended infinite loop or an autonomous product manager.
- Implement the files or automation described here. This document is design only.

## 1. Current manual loop and the engineering-loop definition

### Current manual loop

```text
ChatGPT creates instructions
        -> user copies them to Codex
        -> Codex performs work
        -> user copies results to ChatGPT
```

The manual hand-offs naturally provide pauses, but the task contract, allowable state transitions, evidence, retry budget, and approval decisions are not yet represented in a machine-checkable form.

### Engineering-loop definition

An engineering loop is a state machine with four separate roles, immutable task/evaluation contracts, explicit budgets, evidence-based verification, and a single-state-transition `run_once` command model. The loop can be reused for a future project only after that project defines its own scope and contracts.

## 2. Role separation

| Role | May do | Must not do | Output |
|---|---|---|---|
| Planner | Read state, task registry, and pre-registered work; select exactly one permitted next task. | Invent a hypothesis, alter an evaluation contract, select an unregistered task, or advance a state. | One task proposal and its deterministic `task_hash`. |
| Implementer | Work only in the stated isolated branch/worktree; perform the one selected task; run required tests; create the task commit when allowed. | Continue to a second task, modify scope/criteria, merge, tag, deploy, or self-certify verification. | Commit, command summaries, and raw evidence paths. |
| Verifier | Independently inspect Git, tests, hashes, files, data boundaries, network evidence, and budgets; produce `PASS`, `FAIL`, or `BLOCKED`. | Trust self-report alone, repair code, amend commits, or choose a new task. | Verifier report with reproducible evidence. |
| State Controller | Validate the verifier report and authorization; write one permitted transition only. | Infer approval, skip a gate, transition twice in one run, or modify project code. | Updated state and append-only history event. |

No role may impersonate another role's evidence. A single agent may execute different roles in separate `run_once` invocations, but the role, inputs, evidence, and state transition must remain distinct in `loop_history.jsonl`.

## 3. State machine

### State diagram

```text
NEW -> PLANNED -> READY -> IMPLEMENTING -> VERIFYING -> ACCEPTED -> DONE
                     ^             |             |
                     |             v             +-> REJECTED
                     +--- RETRY_ALLOWED <--------+       |
                              |                           v
                              +-> IMPLEMENTING       HUMAN_GATE

Any active state --(blocked evidence)--> BLOCKED
Any non-terminal state --(cancel)--> CANCELLED
Any state requiring explicit approval --> HUMAN_GATE
HUMAN_GATE --(valid, unused approval)--> only its predeclared return state
```

`REJECTED`, `BLOCKED`, `CANCELLED`, and `DONE` are terminal for automatic execution. `REJECTED` may only move to `HUMAN_GATE`, never directly to `PLANNED` or `READY`.

### State definitions and transitions

| State | Meaning and permitted processing | Allowed next states | State writer |
|---|---|---|---|
| `NEW` | A loop identifier exists but no task or contract is accepted. Planner may inspect registered tasks. | `PLANNED`, `CANCELLED`, `HUMAN_GATE` | State Controller after Planner output |
| `PLANNED` | One pre-registered task and immutable contract are bound by hash. Preflight checks may run. | `READY`, `BLOCKED`, `CANCELLED`, `HUMAN_GATE` | State Controller |
| `READY` | Isolated branch/worktree, clean base commit, contract, budget, and lock are verified. | `IMPLEMENTING`, `BLOCKED`, `CANCELLED`, `HUMAN_GATE` | State Controller |
| `IMPLEMENTING` | Implementer may perform only `current_task`; it may commit only its authorized files. | `VERIFYING`, `BLOCKED`, `CANCELLED`, `HUMAN_GATE` | State Controller after implementation evidence |
| `VERIFYING` | Verifier runs independent checks against the stated commit and contract. | `ACCEPTED`, `REJECTED`, `RETRY_ALLOWED`, `BLOCKED`, `HUMAN_GATE` | State Controller after verifier report |
| `RETRY_ALLOWED` | Exactly one implementation-bug correction to the same task is permitted; no requirement or method change. | `IMPLEMENTING`, `BLOCKED`, `CANCELLED`, `HUMAN_GATE` | State Controller |
| `HUMAN_GATE` | Automation is stopped awaiting a valid approval record for the exact task/hash/action. | Predeclared return state, `CANCELLED`, `BLOCKED` | State Controller after approval validation |
| `ACCEPTED` | Contract was satisfied; no new task is selected automatically. | `DONE`, `HUMAN_GATE` | State Controller |
| `REJECTED` | Contract failed or task result is unacceptable; no automatic alternate method is allowed. | `HUMAN_GATE` only | State Controller |
| `BLOCKED` | Safety, budget, environment, or evidence issue prevents progress. | `HUMAN_GATE` only | State Controller |
| `CANCELLED` | Human or contract stopped the loop before completion. | none | State Controller |
| `DONE` | Accepted work was archived and the loop is finished. | none | State Controller |

### Forbidden transitions

- Any state directly to `DONE` except `ACCEPTED -> DONE`.
- `REJECTED -> PLANNED`, `REJECTED -> READY`, or `REJECTED -> IMPLEMENTING` without a human-gated, separately approved new loop.
- `BLOCKED -> IMPLEMENTING` without an approved gate resolution.
- Any automatic transition out of `HUMAN_GATE`.
- Any transition that changes `task_hash`, evaluation contract, base commit, or budget in place.
- Any transition that reopens an explicitly closed research project.

### One-transition rule, idempotency, and recovery

Each `run_once` invocation may write **zero or one** state transition. A run reads `current_state` and an immutable state revision, acquires a lock, performs only the operation allowed in that state, appends a history event, makes at most one transition, commits the state evidence if state files are versioned, and stops.

An identical retry with the same `(loop_id, current_state, task_hash, input_commit, attempt)` must detect the prior history event and return the existing outcome rather than duplicate a commit, API call, or state change. If a process ends after implementation but before transition, the next run reconciles the expected commit/evidence with `loop_history.jsonl`; ambiguity is `BLOCKED`, not inferred success.

## 4. Mandatory human approval gates

The State Controller must transition to `HUMAN_GATE` and stop before any of the following:

- paid contract, API key/credential addition, external-service registration;
- design, purpose, evaluation-condition, or experiment-budget change;
- a new hypothesis;
- merge into an existing branch, tag creation, deploy, schedule enablement, shadow enablement, or live order;
- data deletion or overwrite of existing results;
- a retry after `REJECTED` that uses a different method;
- reopening a `CLOSED` research program.

The gate is passable only when `human_approvals.jsonl` contains an unexpired, unused approval whose `loop_id`, `task_hash`, requested action, and permitted return state match exactly. User prose outside that record is not sufficient evidence for an automated controller. Consuming an approval is itself a single, auditable state action.

## 5. Proposed machine-readable contracts

These are proposed formats only; this design does not create them.

### `LOOP_SPEC.md`

Human-readable, version-controlled loop charter. It includes purpose, non-goals, permitted and forbidden work, stopping conditions, human gates, and each budget. Its canonical hash is stored in state and history; changing it requires a new loop or a human gate.

### `loop_state.json`

Example schema:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["project_id", "loop_id", "current_state", "allowed_next_states", "base_branch", "base_commit", "work_branch", "worktree_path", "current_task", "task_hash", "attempt", "max_attempts", "budget_remaining", "human_gate", "last_verified_commit", "created_at", "updated_at"],
  "properties": {
    "project_id": {"type": "string"},
    "loop_id": {"type": "string"},
    "current_state": {"enum": ["NEW", "PLANNED", "READY", "IMPLEMENTING", "VERIFYING", "RETRY_ALLOWED", "HUMAN_GATE", "ACCEPTED", "REJECTED", "BLOCKED", "CANCELLED", "DONE"]},
    "allowed_next_states": {"type": "array", "items": {"type": "string"}},
    "base_branch": {"type": "string"},
    "base_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
    "work_branch": {"type": "string"},
    "worktree_path": {"type": "string"},
    "current_task": {"type": "string"},
    "task_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "attempt": {"type": "integer", "minimum": 0},
    "max_attempts": {"type": "integer", "minimum": 0},
    "budget_remaining": {"type": "object"},
    "human_gate": {"type": "object"},
    "last_verified_commit": {"type": ["string", "null"]},
    "created_at": {"type": "string", "format": "date-time"},
    "updated_at": {"type": "string", "format": "date-time"}
  },
  "additionalProperties": false
}
```

### `evaluation_contract.json`

The contract is immutable for a task and contains metrics, pass/fail criteria, data bounds, prohibited periods, allowed network hosts, allowed/forbidden files, required tests, deterministic-output conditions, and the budget ceilings. A concise example:

```json
{
  "contract_version": 1,
  "task_hash": "sha256-of-canonical-task",
  "metrics": [{"name": "test_pass_rate", "operator": "=", "value": 1}],
  "pass_conditions": ["all_required_tests_pass", "allowed_files_only"],
  "failure_conditions": ["future_data_access", "secret_detected", "dirty_base_worktree"],
  "data_boundary": {"date_to": "YYYY-MM-DD", "labels_confirmed_by": "signal_date"},
  "prohibited_periods": ["future or reference dates when the contract says so"],
  "allowed_network_hosts": [],
  "allowed_files": ["exact paths or path globs"],
  "forbidden_files": ["models/**", "raw_data/**"],
  "required_tests": ["pytest -q", "git diff --check"],
  "determinism": {"runs": 2, "artifact_hashes_must_match": true},
  "budget": {"max_agent_runs": 0, "max_retries": 0, "max_api_calls": 0, "max_download_bytes": 0, "max_changed_files": 0, "max_model_fits": 0, "max_evaluations": 0}
}
```

### `loop_history.jsonl`

Append-only, one canonical JSON object per run. Required fields are `run_id`, `loop_id`, `start_state`, `end_state`, `input_commit`, `output_commit`, `command_summary`, `changed_files`, `test_results`, `verification_result`, `state_transition`, `failure_reason`, `human_approval_id`, and `timestamp`. The event must also include `task_hash`, state revision before/after, observed network counts, and deterministic artifact hashes. No raw credentials, raw market data, or unredacted command environment is allowed.

### `human_approvals.jsonl`

Append-only approval evidence with `approval_id`, `loop_id`, `task_hash`, exact approved action, approver identity, approved timestamp, expiry, `used`, `used_at`, permitted return state, and optional corresponding commit. An approval cannot be reused across loops, task hashes, or actions.

## 6. `run_once` protocol

```text
1. Read loop_state, LOOP_SPEC, evaluation_contract, history, and approvals.
2. Verify JSON schema, contract/task hashes, base branch/commit, budget, and allowed transition.
3. Acquire a per-loop exclusive lock using atomic creation.
4. Fail closed if the worktree is dirty, base commit differs, lock is live, or state/history disagree.
5. Execute only the operation permitted by current_state and current_task.
6. Preserve command receipts and non-secret evidence outside the agent narrative.
7. Have Verifier independently inspect the stated output commit and contract.
8. State Controller writes at most one allowed transition and appends one history event.
9. Commit only the authorized state/evidence files when the contract permits it.
10. Release the lock and stop. Never select the next task in the same run.
```

### Locking and crash handling

- Lock contents include `loop_id`, state revision, process identity, host, start time, intended input commit, and a nonce.
- A live lock always blocks concurrent execution. The system never guesses that a holder is dead.
- A stale lock can only be cleared at `HUMAN_GATE` after inspecting the last durable history event and Git state.
- The controller checks for an existing task commit with the same task hash before creating another. More than one matching commit is `BLOCKED`.
- External calls must be recorded with a request id and cache key before invocation. Retrying an uncertain external call is `BLOCKED` unless the provider supports an idempotency key and the contract allows it.
- A state file that points to a commit different from the verified commit, or a state transition without the matching history event, is `BLOCKED`.

## 7. Verifier contract

Verifier evidence is obtained independently from the Implementer report. At minimum it checks:

| Area | Required evidence | Fail-closed result |
|---|---|---|
| Scope | `git diff --name-only`, commit file list, allowed/forbidden path matching | Unauthorized or forbidden file |
| Secrets/raw data | secret scanning and tracked-file inspection; raw-data path policy | Secret or raw data committed |
| Quality | Required `pytest`/`unittest` receipts and `git diff --check` | Missing or failing command |
| Git integrity | clean worktree, expected base/input/output commits, branch/tag ref comparison, no merge or force push | Dirty/unexpected refs/history |
| Data/temporal rules | data manifest hashes, row/date boundary assertions, label confirmation cutoff | Future/forbidden data access |
| Network | host allowlist receipt and call counts | Unapproved host or count |
| Budget | counted runs, retries, API calls, bytes, changed files, model fits, evaluations | Any ceiling exceeded |
| Reproducibility | required repeated-run hashes and deterministic ordering | Hash mismatch |
| Human authority | matching, unexpired, unused approval record | Missing/mismatched approval |

The Verifier reports only `PASS`, `FAIL`, or `BLOCKED`, with command output references and hashes. `PASS` means contract evidence passed; it does not mean a strategy, product, or deployment is approved.

## 8. Retry and budget policy

### Fixed retry rule

- One implementation-bug correction to the same task is allowed at most once.
- The correction must preserve `task_hash`, contract hash, evaluation criteria, data boundary, and method.
- A specification change, alternate model, different method, or relaxed condition is not a retry; it requires `HUMAN_GATE` and normally a new loop.
- Reaching `max_attempts`, `max_retries`, or any budget ceiling transitions to `BLOCKED`.
- `REJECTED` never returns automatically to `PLANNED`.
- REFERENCE outcomes must not create subsequent hypotheses.

### Required budget fields

Every contract must set nonnegative ceilings for:

- maximum agent runs;
- maximum retries;
- maximum external API calls;
- maximum downloaded bytes;
- maximum changed files;
- maximum model fits;
- maximum evaluations.

For a documentation-only loop, the recommended values are one agent run, zero retries, zero API calls, zero downloaded bytes, one changed file, zero model fits, and zero evaluations.

## 9. Execution-mode comparison

| Mode | Environment/cost | Safety and stopping | Phone-only operation | PC requirement | Secrets/logs/reproducibility |
|---|---|---|---|---|---|
| A. Current manual | ChatGPT, Codex, user copy/paste; lowest setup cost | Strong human pauses, but contracts and evidence are informal | Yes for instruction hand-off; local work still needs a PC | Yes for local execution | User must preserve logs manually; secret exposure depends on copied text |
| B. Semi-automatic `run_once` | Repository state/contract files plus user command to Codex; modest implementation cost | Strong fail-closed checks and one-transition stopping; human remains in control | Yes to ask for a run; PC needed for worktree execution | Yes | Versioned state/history make audit and reproduction practical; secrets remain out of state files |
| C. Automatic orchestrator | Local service or CI, credentialed execution environment, monitoring; highest cost | Fastest but greatest misconfiguration/credential/loop risk; requires robust gates and incident handling | Possibly for approval only | CI can avoid an always-on PC; local orchestration cannot | Requires dedicated secret manager, durable logs, locks, monitoring, and audited runner identities |

### Recommendation

Adopt Mode B first. It preserves the useful human pause in the existing process while making scope, evidence, budgets, and stopping conditions explicit. Mode C must not be introduced merely to reduce copy/paste; it requires a separate security and operational decision at `HUMAN_GATE`.

## 10. Staged adoption plan

### Phase A — contracts and state only (recommended first implementation)

- Implementation: add `LOOP_SPEC.md`, schema-validated state/contract/history/approval formats, and documentation templates. No agent runner, no scheduler, no automatic state update.
- Outputs: a manual checklist plus example contracts for a non-sensitive, non-research maintenance task.
- Completion: schemas validate; a human can audit a planned task and manually record one transition.
- Stop: ambiguous state, missing approval evidence, or any attempt to attach it to closed stock research.
- Advance condition: a human approves a small, unrelated pilot loop after reviewing the templates.
- User operation: explicitly chooses each task and manually requests a single review or execution.

### Phase B — Codex `run_once`

- Implementation: a local command that validates state, obtains a lock, executes one role-specific step, and stops without scheduling.
- Outputs: idempotent history receipts and fail-closed recovery behavior.
- Completion: repeated invocation neither duplicates work nor advances more than one state.
- Stop: lock ambiguity, dirty worktree, contract mismatch, or exceeded budget.
- Advance condition: Phase B has passed a separately approved, non-sensitive pilot.
- User operation: invokes `run_once` and handles every `HUMAN_GATE`.

### Phase C — independent Verifier process

- Implementation: separate verifier command/environment that reads commits and contracts but has no code-edit authority.
- Outputs: signed or hashed verifier reports and required evidence receipts.
- Completion: verifier catches seeded scope, future-data, and dirty-worktree violations in a pilot.
- Stop: verifier cannot independently reproduce the result.
- Advance condition: human reviews verifier reliability and approves automation scope.
- User operation: requests verification and resolves gates.

### Phase D — scheduled execution outside gates

- Implementation: only after separate approval, a scheduler starts eligible `run_once` invocations with a minimal credential set and no gate bypass.
- Outputs: schedule configuration, monitoring, failure alerts, and durable lock/audit logs.
- Completion: a non-sensitive pilot demonstrates safe stop/recovery and no duplicate runs.
- Stop: any unexpected network, state, budget, or authorization event.
- Advance condition: none is implied; deployment remains a separate human decision.
- User operation: approves gates, monitors alerts, and can disable the scheduler.

Phase A is the sole recommended initial scope because it captures the safety contract without creating a new execution authority. It can be reviewed on a phone and does not require persistent local infrastructure or credentials.

## 11. Stock-analyzer application limits

stock-analyzer is an example of a completed research program, not an eligible work queue. This framework must reject any task that would:

- reevaluate `FREE_PROTOTYPE_NOT_PROMISING`;
- change features, models, periods, thresholds, or evaluation conditions;
- contract J-Quants or another paid data source;
- start LOOP-005, restart v3, or enable existing shadow execution;
- generate, backtest, deploy, schedule, or place an investment strategy.

For stock-analyzer, the only permitted example use is passive documentation of the already closed outcome. Any proposal to reopen it requires a `HUMAN_GATE` and a separately defined project; this document grants no such authority.

## 12. Fail-closed conditions

The controller must stop in `BLOCKED` or `HUMAN_GATE` when any of the following occurs:

- unknown state, schema failure, missing history event, or contract/task hash mismatch;
- dirty worktree, wrong base commit, unexpected branch/tag, merge, or force push;
- missing/expired/mismatched approval;
- unauthorized file, secret, raw data, network host, data period, model fit, evaluation, or external call;
- budget exhaustion, duplicate task commit, duplicate external request, non-deterministic output, or ambiguous lock;
- a request conflicts with the closed stock-analyzer research status.

There is no "best effort" path around any of these conditions.

## 13. Decisions before implementation

### Confirmed in this design

- State transitions are limited to one per `run_once`.
- Contracts and budgets are immutable per task.
- Human gates are mandatory for material authority changes.
- Verifier is independent and read-only.
- Phase A is the recommended first implementation.

### Proposed, requiring user selection

- Where state and history live: repository, a separate control repository, or an external append-only store.
- Whether Phase A uses JSON Schema validation locally, in CI, or both.
- The identity and evidence format for a human approver.
- Lock implementation and stale-lock review procedure.
- Secret-scanning tool and approved network-evidence mechanism.
- Whether future use is local-only, CI-assisted, or both.

### Not decided and intentionally out of scope

- A specific orchestration product or Codex automation integration.
- CI provider, runner trust model, and secret manager.
- Any project-specific evaluation metric, model, strategy, deployment, or schedule.

## 14. Final design conclusion

The safe next move is not automation. It is Phase A: explicitly record intent, authority, contracts, budgets, and evidence while retaining human invocation and review. Only a separate, human-approved pilot on a non-sensitive project should determine whether Phase B or later is warranted.

This design does not alter the closed stock-analyzer outcome and must not be interpreted as authorization to resume investment research or automation.
