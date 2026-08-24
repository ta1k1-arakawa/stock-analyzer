# V9_006 Stage-A locator implementation review

```text
REVIEWED_SHA=7c5abbee11b02406b202d413c917f2ed523e5d13
CRITICAL=0
HIGH=3
MEDIUM=0
RESULT=BLOCK

FINDINGS:
1. V9_006_LOCATOR_IMPL_HIGH_1_KNOWN_INCOMPLETE_ACQUISITION_CROSSES_NETWORK
2. V9_006_LOCATOR_IMPL_HIGH_2_F1_EXACT_ROOT_CONTRACT_MISMATCH
3. V9_006_LOCATOR_IMPL_HIGH_3_SECURITY_TYPE_GATE_OUT_OF_SCOPE_WEAKENING

TARGET_FINDING_STATUS=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW
```

This records GPT's independent exact-SHA `BLOCK` review of the Stage-A
locator/inventory-contract implementation at reviewed commit
`7c5abbee11b02406b202d413c917f2ed523e5d13`, and this task's remediation of
exactly one of its three HIGH findings.

## Finding 1 (remediated this task)

`V9_006_LOCATOR_IMPL_HIGH_1_KNOWN_INCOMPLETE_ACQUISITION_CROSSES_NETWORK`:
`verify_locator_contract_complete()` now genuinely passes for the reviewed
`LOCATOR_STRATEGIES` registry, but that is locator-*strategy* completeness,
not acquisition *implementation* completeness -- no F2-F7 traversal/fetch
code exists yet. Left unguarded, a real `run_stage_a()` run would cross the
JPX network boundary, fetch only the two objects that do have an
implemented fetch path (F1's terminal snapshot, the calendar page), and
report the remaining 648 monthly-coverage slots `MISSING` -- a knowingly
incomplete acquisition run, the same category of problem as the doomed-run
issue `V9_006_HIGH_1` already forbade for the locator-methodology gate.

**Remediation implemented this task:** a new, separate, pre-network
fail-closed gate, `verify_acquisition_implementation_ready()`, called in
`run_stage_a()` immediately after `verify_locator_contract_complete()` and
before output-root creation, before any git call, and before any fetcher
call. It raises `V9005StageABlocked(STAGE_A_ACQUISITION_IMPLEMENTATION_
INCOMPLETE)` (public `failure_class=CHATGPT_DECISION_REQUIRED`) while the
module-level flag `ACQUISITION_IMPLEMENTATION_COMPLETE` is `False`, which
this task hardcodes it to be. The flag flips to `True` only via a future,
separately reviewed task that actually implements the complete F1-F7
acquisition pipeline. `verify_locator_contract_complete()` itself is
unchanged and continues to pass; the 648-record matrix, F1 `TERMINAL_SEED`
role, F2 bridge derivation, F3 `YEAR` strategy, F4/F5/F6/F7 strategies, and
the retry policy are all unchanged. See
`V9_006_STAGE_A_IMPLEMENTATION.md`'s "Acquisition-implementation readiness"
section for the full description, and
`tests/test_v9_005_stage_a_jpx_probe.py`'s
`test_acquisition_implementation_is_not_yet_complete` and
`test_run_stage_a_valid_confirmation_still_stops_before_any_network_or_git`
for the proof.

`V9_006_LOCATOR_IMPL_HIGH_1=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW`

## Finding 2 (OPEN -- explicitly out of scope this task)

`V9_006_LOCATOR_IMPL_HIGH_2_F1_EXACT_ROOT_CONTRACT_MISMATCH`: not
remediated. `LISTED_ISSUES_PAGE_URL` / F1's root were not changed by this
task, per this task's explicit prohibition. `V9_006_LOCATOR_IMPL_HIGH_2=OPEN`.

## Finding 3 (OPEN -- explicitly out of scope this task)

`V9_006_LOCATOR_IMPL_HIGH_3_SECURITY_TYPE_GATE_OUT_OF_SCOPE_WEAKENING`: not
remediated. `security_type_pass` and its semantics were not changed by this
task, per this task's explicit prohibition. `V9_006_LOCATOR_IMPL_HIGH_3=OPEN`.

## What this review closure does not do

This is not a GPT review -- it records the BLOCK review this task responds
to and this task's own remediation claim for finding 1 only. It creates no
network, data, T1, or design-freeze authority, and does not by itself
authorize any Stage-A execution, which remains `BLOCK`ed pending: GPT's
independent exact-SHA review of this remediation
(`GPT_EXACT_SHA_V9_006_LOCATOR_IMPL_HIGH_1_REVIEW`); remediation and PASS of
findings 2 and 3; a future, separately reviewed F2-F7 acquisition-pipeline
implementation task; and a fresh, separate, explicit point-of-use human
network authorization obtained after all of the above.
