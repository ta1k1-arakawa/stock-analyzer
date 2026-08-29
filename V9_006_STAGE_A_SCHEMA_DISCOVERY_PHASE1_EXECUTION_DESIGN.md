# V9_006 Stage-A schema-discovery Phase-1 execution design

Status: execution binding only; no executor, receipt, runner, PowerShell
entrypoint, acquisition, or human authorization is created by this document.

## Purpose and phase boundary

Phase 1 is the prefreeze public-source structure-acquisition phase whose
object identities are fully deterministic without parsing the F1 TERMINAL
content. Its future real execution acquires and profiles only the evidence
objects below. It does not derive the F1 terminal month `T`, call
`f2_bridge_months`, enumerate or pre-create F2 BRIDGE slots.

This design-remediation task performs zero network requests and creates no
authority. The future Phase-1 executor implementation and its synthetic tests
also do not authorize or perform real acquisition. Only after that
implementation and wrapper receive exact-SHA GPT review PASS and a human
supplies fresh explicit authorization under the dedicated Phase-1 one-shot
contract may the real Phase-1 execution make the public JPX network requests
required by the already-bound helpers. That run remains governed by
`AI_REAL_EXECUTION_RUNBOOK.md`; no documentation, implementation, test, or
prior confirmation alone is authorization to acquire.

The later Phase 2 may derive `T` only from the exact locked F1 TERMINAL
payload using a separately reviewed parser and boundary.  It alone may then
call `f2_bridge_months(T)`, acquire the resulting bridge objects, and do so
under a fresh authority that cannot be satisfied by any Phase-1 receipt.
`T` must never be inferred from a date, retrieval time, URL, F7 envelope,
filename, or another proxy.

## Exact Phase-1 raw-lock inventory

| Class | Family/domain | Count | Treatment |
| --- | --- | ---: | --- |
| Evidence | F1 / TERMINAL | 1 | Profiled |
| Evidence | F2 / BASE | 108 | Profiled |
| Evidence | F3 / YEAR | 9 | Profiled once each |
| Evidence | F4 / BASE | 108 | Profiled |
| Evidence | F7 / BASE | 108 | Profiled |
| Evidence | F7 / ENVELOPE_EXTRA | 7 | Profiled |
| Support | F1 discovery root | 1 | Locked only; never profiler input |
| Support | F2/F4 shared root | 1 | Locked only; never profiler input |
| Support | F2/F4 shared year pages, 2017--2025 | 9 | Locked only; never profiler input |
| Support | F3 discovery root | 1 | Locked only; never profiler input |

The exact evidence total is **341**: F1 1, F2 BASE 108, F3 YEAR 9, F4 BASE
108, and F7 115 (108 BASE plus seven ENVELOPE_EXTRA).  The exact support total
is **12**, so a complete Phase-1 run has **353 canonical raw-lock pairs**.
Support provenance objects are not schema evidence objects and must never be
materialized as any of the 341 profiler inputs.  F5 and F6 are excluded.
F2 BRIDGE is excluded.

## Exact acquisition binding

### F1 TERMINAL

A future implementation may factor the reviewed F1 portion of `run_stage_a`
into this narrow behavior-preserving helper:

```text
acquire_f1_terminal_evidence(output_root, *, fetcher, sleep, clock) -> tuple[str, int]
```

It must retain precisely the existing sequence: lock the F1 discovery root;
call `extract_data_j_xls_url(locked_discovery["raw"],
locked_discovery["resolved_url"])`; then lock the F1 `TERMINAL` object using
that URL.  Its result is only the TERMINAL evidence slot ID and aggregate
network-attempt count.  It must not parse the TERMINAL payload or infer `T`.

### F2 and F4 BASE

Phase 1 binds only `acquire_f2_f4_monthly_evidence`.  It iterates exact
`inventory_months()` in ascending order and, for each month, calls F2 then
F4, preserving the reviewed base traversal order.  It must not call
`acquire_f2_f4_required_slots`, because that seam calls `f2_bridge_months`
and includes F2 BRIDGE.

### F3 YEAR

Phase 1 binds `acquire_f3_required_slots`.  It locks exactly nine unique YEAR
objects for 2017--2025 and fans each object to twelve BASE references.  Schema
discovery profiles the nine unique YEAR objects only, never 108 duplicated
fan-out references.

### F7 envelope

Phase 1 binds `acquire_f7_required_slots`: exactly 108 inventory-month BASE
objects and the seven `calendar_envelope_extra_months()` ENVELOPE_EXTRA
objects, 115 in total.

## Evidence-to-profiler binding

Only the exact 341 evidence locks above may be materialized as
`VerifiedLockedObject` values and passed to `profile_verified_lock`.  The
domain mapping is closed:

- F1 TERMINAL maps to `TERMINAL`.
- F2 and F4 `inventory_months()` map to `BASE`.
- F3 years 2017--2025 map to `YEAR`.
- F7 `inventory_months()` map to `BASE`.
- F7 envelope extras map to `ENVELOPE_EXTRA`.

Existing `select_representatives` operates only on those safe profiles.
Support roots and F2/F4 year pages are not evidence and cannot enter
selection.

## Authority and deferred wrapper

This binding creates no consumable authority.  The existing identity
`V9_006_STAGE_A_SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_ONE_SHOT` is merely
documented here; receipt, PowerShell, and human-gate mechanics are deferred
to a later wrapper task. Once independently implemented, reviewed, and
freshly authorized, that dedicated Phase-1 contract governs the real public
JPX acquisition described above. No prior confirmation identity can authorize
a Phase-2 bridge run, and Phase-1 authority cannot be reused for Phase 2.

`V9_design_frozen=false`; historical evaluation, private/sealed access, live
or order execution remain unauthorized; future profitability is unestablished;
`PUBLIC_ACQUISITION_EXECUTED=false`; `HUMAN_GATE_CONSUMED=false`; and
`OVERALL_STAGE_A_IMPLEMENTATION_READY=false`.

## Recorded independent review

GPT-5.6 Sol reviewed `df9f00a2aedf627c08d7f9d011589497422147fd` with
`CRITICAL=0`, `HIGH=0`, `MEDIUM=0`, `LOW=1`, `RESULT=PASS`.
`V9_006_STAGE_A_SCHEMA_DISCOVERY_FOUNDATION=PASS` and M1/M2/M3/M4 are
resolved.  `LOW_1=SAFE_VALIDATOR_NEGATIVE_REGRESSION_COVERAGE_NOT_EXHAUSTIVE`
remains deferred; this design-only task does not remediate it.
