# V9_006 Stage-A schema-discovery Phase-1 real-execution design

Status: governance contract only. It creates no implementation, PowerShell
entrypoint, receipt, result, authorization, or network activity, and awaits
GPT exact-SHA review before a future implementation task.

## Operation scope

Phase 1 only: F1 TERMINAL (1), F2 BASE (108), F3 YEAR (9), F4 BASE (108),
and F7 (115: 108 BASE plus seven ENVELOPE_EXTRA), exactly **341** evidence
objects. The 12 support locks are F1 discovery root; F2/F4 shared root; nine
F2/F4 shared year pages; and F3 discovery root. Support locks are never
profile inputs; a successful fresh run has exactly **353** canonical raw-lock
pairs. F2 BRIDGE, F5, and F6 are excluded. No `T` parsing/inference occurs.
Phase 2 is separately reviewed/authorized and cannot reuse a Phase-1
authority or receipt.

## Authority and mandatory pre-gate ordering

This public acquisition is an explicitly designed `ONE_SHOT` human-gated
operation. Its sole confirmation contract is
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_ONE_SHOT`; no other V9,
V8, or F6 confirmation satisfies it. Implementation review PASS never
authorizes execution. Each invocation requires fresh point-of-use human
authorization after all pre-gate checks pass; a typed confirmation is never
reused.

Before asking for human authorization, a future entrypoint must prove:

1. correct repository/authoritative branch, exact local HEAD, exact
   authoritative remote HEAD, clean tree, and reviewed implementation binding;
2. fresh, nonexistent OutputRoot and no conflicting receipt, result, or
   durable state;
3. `.venv-real-execution\Scripts\python.exe` exists, every exact protected
   interpreter/readiness/dependency/environment-lock check in
   `AI_REAL_EXECUTION_RUNBOOK.md` passes, and synthetic operational readiness
   is PASS;
4. `CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE=YES`.

After confirmation, but before consumption, every applicable non-destructive
provenance/readiness/durable-state binding is rerun. A failure before receipt
is `PRE_GATE`, has `gate_consumed=false` and zero network requests wherever
mechanically provable, and creates no execution authority.

## Gate boundary and receipt

The dedicated receipt is
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_GATE_RECEIPT.json`, schema
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_GATE_RECEIPT_V1`, with exactly:

- `schema_version`
- `task`
- `confirmation_contract`
- `execution_sha`
- `gate_consumed`
- `consumption_timestamp_utc`

`task` is exactly `V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_REAL_EXECUTION`.
`execution_sha` is the exact independently reviewed implementation SHA used
by the real invocation. The receipt contains no raw confirmation, path, URL,
payload, ticker identity, or private value.

The fresh OutputRoot is created and the receipt atomically/durably published
without overwrite strictly before the first JPX request. Publication is the
one-shot consumption boundary. Existing OutputRoot, receipt, or result stops:
never delete, reset, overwrite, or automatically reuse state. After receipt,
any failure is `POST_GATE`; receipt remains durable; no second execution,
reset, deletion, or authority reuse is allowed. Unknown partial-core request
counts must be reported as `unknown`, never fabricated.

## Successful durable result

The sole successful result is canonical/no-overwrite
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_RESULT.json`, schema
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PHASE1_RESULT_V1`, with only:

- `schema_version`, `task`, `execution_sha`, `execution_result`, and
  `gate_consumed`
- `evidence_count`, `support_raw_lock_count`, `total_raw_lock_pair_count`,
  and `network_attempt_count`
- `evidence_slot_ids`, `safe_profiles`, and `representative_safe_profiles`

Success requires evidence count 341, support raw-lock count 12, total
raw-lock-pair count 353, and valid canonical provenance for every raw lock.
It contains no raw bytes, raw URLs, OutputRoot path, confirmation, private
identity, historical-evaluation output, or `T`. Stdout is a privacy-safe
summary/hash/count report only.

## Future topology and unchanged state

A later reviewed implementation uses the existing reviewed production fetcher
and UTC clock with `run_phase1_schema_discovery_core`; confirmation is not on
argv. Actual execution is through an atomic reviewed Windows PowerShell
entrypoint and canonical `.venv-real-execution`, never an AI-session-dependent
process. Creating this design authorizes no execution.

`V9_design_frozen=false`; historical evaluation/private/sealed access remain
unauthorized; `PUBLIC_ACQUISITION_EXECUTED=false`, `HUMAN_GATE_CONSUMED=false`,
and `OVERALL_STAGE_A_IMPLEMENTATION_READY=false`; future profitability remains
unestablished.
