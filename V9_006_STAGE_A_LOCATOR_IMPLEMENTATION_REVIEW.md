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

## HIGH_1 exact-SHA remediation review

```text
REVIEWED_SHA=afc59fb285e09aa8c7225ce6f855d16801c67584
CRITICAL=0
HIGH=0
MEDIUM=0
RESULT=PASS
```

GPT's independent exact-SHA review of the `V9_006_LOCATOR_IMPL_HIGH_1`
remediation (reviewed commit `afc59fb285e09aa8c7225ce6f855d16801c67584`,
parent `7c5abbee11b02406b202d413c917f2ed523e5d13`) is `PASS`.
`V9_006_LOCATOR_IMPL_HIGH_1=RESOLVED`.

## Finding 2 status update

`V9_006_LOCATOR_IMPL_HIGH_2_F1_EXACT_ROOT_CONTRACT_MISMATCH`: remediated in
task `V9_006_LOCATOR_IMPL_HIGH_2_F1_EXACT_ROOT`. `LISTED_ISSUES_PAGE_URL`
(and, derived from it, `LOCATOR_STRATEGIES[F1].root_url`) is now bound to
the exact authoritative English root
`https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html`
per `V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md` -- no alias,
fallback root, redirect-based substitution, non-English alternative, or
guessed historical root was introduced. The reviewed traversal rule (a
unique same-domain `data_j.xls` link from the official F1 page) is
unchanged; a relative link continues to resolve against this exact bound
root via `urllib.parse.urljoin`, proven for a genuinely relative href by
`test_extract_data_j_xls_url_relative_link_resolves_against_english_root`.
`ACQUISITION_IMPLEMENTATION_COMPLETE` remains `False` and
`verify_acquisition_implementation_ready()` is unchanged, so a valid real
run still stops before any filesystem/git/network access.

`V9_006_LOCATOR_IMPL_HIGH_2=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW`

## Finding 3 (still OPEN -- explicitly out of scope this task)

`V9_006_LOCATOR_IMPL_HIGH_3_SECURITY_TYPE_GATE_OUT_OF_SCOPE_WEAKENING`
remains `OPEN`; not remediated by this task, per this task's explicit
prohibition. `security_type_pass` and its semantics were not touched.

## HIGH_2 exact-SHA remediation review

```text
REVIEWED_SHA=ed70bc8f42beabef5aac76242a7aaba9c9ab1b6a
CRITICAL=0
HIGH=0
MEDIUM=0
RESULT=PASS
```

GPT's independent exact-SHA review of the `V9_006_LOCATOR_IMPL_HIGH_2`
remediation (reviewed commit `ed70bc8f42beabef5aac76242a7aaba9c9ab1b6a`,
parent `afc59fb285e09aa8c7225ce6f855d16801c67584`) is `PASS`.
`V9_006_LOCATOR_IMPL_HIGH_2=RESOLVED`.

## Finding 3 status update

`V9_006_LOCATOR_IMPL_HIGH_3_SECURITY_TYPE_GATE_OUT_OF_SCOPE_WEAKENING`:
remediated in task `V9_006_LOCATOR_IMPL_HIGH_3_SECURITY_TYPE_FAIL_CLOSED`.
The previous `security_type_pass = bool(terminal_snapshot_locked)` falsely
equated "a terminal object exists" with V9_005's `SECURITY_TYPE` evidence
requirement (domestic ordinary-common-stock eligibility must be
determinable for every reconstructed identity/date needed by V9 without
future security state; `UNKNOWN` fails). `compute_stage_a_evidence()` now
takes an explicit `security_type_validation_pass: bool` input, and
`security_type_pass = bool(security_type_validation_pass)` -- never
inferred from `terminal_snapshot_locked`, family coverage, row count, or
any other proxy. Production `run_stage_a()` passes
`security_type_validation_pass=False` (statically proven by
`test_production_security_type_validation_pass_is_hardcoded_false`),
because the actual semantic security-type validator has not yet been
implemented or independently reviewed; this `False` will only be replaced
by that future validator's real result. `terminal_snapshot_pass` remains an
independent gate based solely on terminal-snapshot locking -- the two gates
are proven not to be conflated by
`test_terminal_snapshot_locked_true_with_security_type_validation_false_fails_security_type`,
`test_terminal_snapshot_locked_alone_can_never_make_security_type_pass`,
and
`test_synthetic_security_type_validation_true_feeds_conjunction_independent_of_terminal_lock`.
This task removes only the unsafe proxy; it does NOT implement the actual
semantic security-type parser/classifier, and does NOT touch
`canonical_identity_pass`'s formula, `effective_date_pass`,
`reconstruct_security_state`, or `reconstruction_is_deterministic` --
`canonical_identity_pass = bool(terminal_snapshot_locked) and
security_type_pass` is textually unchanged and simply now correctly
reflects the fixed `security_type_pass`. Original V9_006 HIGH_2 (full
semantic reconstruction/validation), original HIGH_3 (raw
provenance/content-lock), and original HIGH_4 (redirect handling) remain
explicitly OPEN and out of scope. `ACQUISITION_IMPLEMENTATION_COMPLETE`
remains `False` and `verify_acquisition_implementation_ready()` is
unchanged, so a valid real run still stops before any
filesystem/git/network access.

`V9_006_LOCATOR_IMPL_HIGH_3=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW`

## What this review closure does not do

This is not a GPT review -- it records GPT's independent PASS of the
HIGH_1 and HIGH_2 remediations, and this task's own remediation claim for
finding 3 only. It creates no network, data, T1, or design-freeze
authority, and does not by itself authorize any Stage-A execution, which
remains `BLOCK`ed pending: GPT's independent exact-SHA review of this
HIGH_3 remediation (`GPT_EXACT_SHA_V9_006_LOCATOR_IMPL_HIGH_3_REVIEW`);
remediation and PASS of the original V9_006 HIGH_2 (full semantic
reconstruction/validation, including the actual security-type
parser/classifier), original HIGH_3 (raw provenance/content-lock), and
original HIGH_4 (redirect handling) findings, none of which this task
remediated; a future, separately reviewed F2-F7 acquisition-pipeline
implementation task; and a fresh, separate, explicit point-of-use human
network authorization obtained after all of the above.
