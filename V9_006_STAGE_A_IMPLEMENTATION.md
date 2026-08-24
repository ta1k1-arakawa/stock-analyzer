# V9_006 Stage-A implementation

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
task=V9_006_STAGE_A_PUBLIC_JPX_PROBE_IMPLEMENTATION
network_executed=false
network_authorized_by_this_task=false
t1_or_design_freeze_authority_created=false
```

This implements, without executing any real network request, the reviewed
V9_005 Stage-A free official JPX metadata/calendar probe defined in
`V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md` (independently
reviewed PASS: `V9_005_HIGH_2B_REVIEW.md`, reviewed SHA
`137e6ba50b916720adeef66f09049010185534d8`). It creates a durable,
fail-closed implementation suitable for later direct Windows PowerShell
execution under a fresh, separately obtained Stage-A human network
authorization -- it does not consume the authorization a human already gave
in chat, and it makes zero real network requests itself.

## Files

- `src/v9_005_stage_a_jpx_probe.py` -- Stage-A logic: off-domain rejection,
  first-complete-payload raw locking with full provenance, the deterministic
  monthly `SOURCE_INVENTORY`, the reconstruction/validation evidence items,
  the exact `FREE_JPX_METADATA_PROBE_PASS` conjunction, the V9_005_HIGH_2B
  signal-grid blob-binding point-of-use check, and the mechanical
  `FINAL_SIGNAL_D0` / `STAGE_B_GLOBAL_END_EXCLUSIVE` derivation.
- `scripts/run_v9_005_stage_a_jpx_probe.py` -- production entrypoint; reads
  the Stage-A confirmation token only from the
  `V9_005_STAGE_A_CONFIRMATION` environment variable (never argv, never
  hardcoded), and prints only the safe JSON summary.
- `scripts/run_v9_005_stage_a_jpx_probe.ps1` -- single atomic `& { ... }`
  Windows entrypoint per `AI_REAL_EXECUTION_RUNBOOK.md` SS1. Performs the
  full preflight (repository, authoritative branch, exact expected local
  HEAD, exact remote HEAD, clean tree, the V9_005_HIGH_2B design-blob
  binding, output-root collision check, and an interactively typed Stage-A
  confirmation token) before any network request; no chat-supplied
  authorization is baked into the file.
- `tests/test_v9_005_stage_a_jpx_probe.py` -- 71 tests, entirely offline
  (synthetic fixtures and injected fake fetchers/git callables only).

## Source-locator discipline (contract item 4)

No JPX endpoint URL in this implementation was guessed; every locator is
reused verbatim from `V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md`
and `V9_006_F1_TERMINAL_SEED_PREFREEZE_AMENDMENT.md`'s exact GPT-reviewed
bindings, represented in code as the `LOCATOR_STRATEGIES` registry
(`src/v9_005_stage_a_jpx_probe.py`):

- **F1** (`TERMINAL`): the listed-issues page + `data_j.xls`
  link-extraction pattern, reused from
  `src/v8k_public_source_preparation.py` / `scripts/build_v8_partition_manifest.py`.
  F1 is `TERMINAL_SEED` only and has zero `MONTHLY_COVERAGE_MATRIX` cells.
- **F2** (`MONTHLY`) / **F4** (`MONTHLY`): a shared Monthly Statistics root,
  distinguished by semantic row label (`F2_SEMANTIC_ROW_LABEL` /
  `F4_SEMANTIC_ROW_LABEL`).
- **F3** (`YEAR`): the delisted-company archive root; one `YEAR` object's
  strategy identically supports all 12 months of its year.
- **F5** (`MONTHLY`, `auxiliary=true`): the listing/co root.
- **F6** (`GLOBAL`): the TOPIX root, exactly one object under the
  "Historical Index Value" section.
- **F7** (`MONTHLY`): the exact GPT-bound per-month template
  `https://www.jpx.co.jp/calendar/{YYYY}{MM:02d}.html`, envelope
  2016-09 through 2026-03 inclusive.

**V9_006_HIGH_1 / HIGH_1A / SOURCE_SLOT_LOCATOR / STAGE_A_LOCATOR_CONTRACT
remediation chain.** The original implementation assumed all seven
families occupied the monthly grid and had no reviewed locator strategy at
all, so it either crossed the network boundary before a guaranteed FAIL
(HIGH_1) or later stopped unconditionally with `CHATGPT_DECISION_REQUIRED`
because no family had a *concrete resolved URL* per month (HIGH_1 fix).
GPT's subsequent methodology work established: (a) F1 is `TERMINAL_SEED`
only (`V9_006_F1_TERMINAL_SEED_PREFREEZE_AMENDMENT.md`), shrinking the base
matrix to 648 records (F2-F7 x 108 months); and (b) the pre-network
completeness gate must verify only that a *reviewed deterministic locator
strategy* (root + semantic traversal, or F7's exact template) is bound per
family -- never that the concrete per-month/per-year child URL is already
known, since discovering that child URL requires traversing a locked
official root response, which is real Stage-A network work.

This task implements that exact contract: `MONTHLY_COVERAGE_FAMILIES` (F2-
F7) drives `build_source_inventory`'s 648-record matrix; `LOCATOR_
STRATEGIES` binds a `LocatorStrategy` (`slot_kind` in `MONTHLY`/`YEAR`/
`TERMINAL`/`GLOBAL` only -- no `MONTHLY_AUXILIARY`) for every one of the
seven families; `resolve_month_locator` returns the bound strategy (not a
concrete URL) for every required monthly-coverage cell; `f2_bridge_months`
mechanically derives F2's post-2025 bridge slots from the terminal snapshot
month `T`; `calendar_envelope_months`/`calendar_envelope_extra_months`
mechanically derive F7's envelope slots outside 2017-2025.
`verify_locator_contract_complete()` therefore now passes without raising
under the currently bound registry -- `LOCATOR_CONTRACT_COMPLETE=true` --
though a real Stage-A run today would still report `FREE_JPX_METADATA_
PROBE_FAIL` (`required_inventory_missing_count=648`), because the actual
F2-F7 root-traversal fetch implementation (parsing a locked official page
to find its child object) is separate, future, authorized work: this task
implements the locator/inventory *contract*, not the traversal *fetcher*.
See `V9_006_HIGH_1_REVIEW.md`, `V9_006_SOURCE_SLOT_LOCATOR_HIGH_1_REVIEW.md`,
and `V9_006_F1_TERMINAL_SEED_AMENDMENT_REVIEW.md` for the full finding and
remediation chain.

`SOURCE_OR_DATA_FEASIBILITY_FAILURE` remains reserved for a genuine result
produced after real Stage-A execution actually attempts (and fails) the
reviewed traversal; it is exercised in tests via the existing F1 + calendar-
page fetch path (`test_run_stage_a_offline_reports_fail_with_safe_
evidence`), which still only fetches the two non-monthly artifacts that
have always had a concrete root -- the single locked calendar page there
covers only 2026-2027 (the same years `src/v7_jpx_calendar.py` already
handles), which is insufficient to mechanically derive `FINAL_SIGNAL_D0`
(requires coverage back to 2018-01-01); `derive_stage_b_global_end_
exclusive` fails closed with `SOURCE_OR_DATA_FEASIBILITY_FAILURE` in that
case rather than compute a wrong index from a narrower window.

## Acquisition-implementation readiness (V9_006_LOCATOR_IMPL_HIGH_1)

GPT's exact-SHA review of the locator/inventory-contract implementation
(reviewed SHA `7c5abbee11b02406b202d413c917f2ed523e5d13`) found `RESULT=BLOCK`
with three HIGH findings; this task remediates exactly one:
`V9_006_LOCATOR_IMPL_HIGH_1_KNOWN_INCOMPLETE_ACQUISITION_CROSSES_NETWORK`.
The other two (`HIGH_2`: F1 exact-root contract mismatch; `HIGH_3`:
`security_type_pass` semantic gate weakening) remain `OPEN` and are
explicitly out of scope here -- see
`V9_006_STAGE_A_LOCATOR_IMPLEMENTATION_REVIEW.md`.

The finding: `verify_locator_contract_complete()` now genuinely passes
(every one of the seven source families has a reviewed deterministic
locator *strategy* bound), but that is a separate thing from the actual
acquisition *implementation* -- no code in this module yet walks a locked
official F2-F7 root response to find each required child object for every
base/bridge/envelope slot. Left unguarded, a real `run_stage_a()` run today
would cross the network boundary, fetch only the two objects that do have
an implemented fetch path (F1's terminal snapshot, the calendar page), and
then report the remaining 648 slots `MISSING` -- a knowingly incomplete
acquisition run, not materially different from the knowingly-doomed-run
problem `V9_006_HIGH_1` already forbade for the locator-methodology gate.

The fix adds a second, independent, pre-network gate,
`verify_acquisition_implementation_ready()`, called in `run_stage_a()`
immediately after `verify_locator_contract_complete()` and before
output-root creation, before any git call, and before any fetcher call. It
raises `V9005StageABlocked(STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE)`
(public `failure_class=CHATGPT_DECISION_REQUIRED`, never
`SOURCE_OR_DATA_FEASIBILITY_FAILURE`) unconditionally while the module-level
flag `ACQUISITION_IMPLEMENTATION_COMPLETE` is `False`, which it is
hardcoded to be by this task. That flag flips to `True` only when a future,
separately reviewed task actually implements the complete F1-F7 acquisition
pipeline (base 648-record matrix, F1's mandatory `TERMINAL` object, F2's
post-2025 bridge slots, F7's envelope slots). This task implements only the
guard, not that pipeline: `verify_locator_contract_complete()` continues to
pass unchanged, the 648-record matrix, F1 `TERMINAL_SEED` role, F2 bridge
derivation, F3 `YEAR` strategy, F4/F5/F6/F7 strategies, and the retry policy
are all unchanged, and `security_type_pass`/F1 root/HIGH_2/HIGH_3/HIGH_4 are
untouched.

Tests: `test_acquisition_implementation_is_not_yet_complete` proves the flag
is `False` and the guard raises with the correct reason/failure class;
`test_run_stage_a_valid_confirmation_still_stops_before_any_network_or_git`
calls `run_stage_a` with a valid confirmation against the real, unmocked,
complete `LOCATOR_STRATEGIES` registry and real (`False`)
`ACQUISITION_IMPLEMENTATION_COMPLETE`, asserting zero fetcher calls, zero
git calls, and no output-root directory created. The two existing offline
regression tests that exercise the fetch/lock/evidence pipeline below both
gates (`test_run_stage_a_offline_reports_fail_with_safe_evidence`,
`test_run_stage_a_wrong_signal_grid_blob_stops_before_any_fetch`) now each
force `ACQUISITION_IMPLEMENTATION_COMPLETE=True` via `monkeypatch`, clearly
commented as forcing (not claiming) completeness, so that neither test
normalizes a "fetch some objects then 648-MISSING FAIL" production run
under the real, still-incomplete state.

## F1 exact-root binding (V9_006_LOCATOR_IMPL_HIGH_2)

GPT's exact-SHA review of the HIGH_1 remediation (`RESULT=PASS`, reviewed
SHA `afc59fb285e09aa8c7225ce6f855d16801c67584`) resolved
`V9_006_LOCATOR_IMPL_HIGH_1`; this task remediates the second finding from
the original locator-implementation review:
`V9_006_LOCATOR_IMPL_HIGH_2_F1_EXACT_ROOT_CONTRACT_MISMATCH`. `HIGH_3`
(`security_type_pass` semantic gate weakening) remains `OPEN` and is
explicitly out of scope here -- see
`V9_006_STAGE_A_LOCATOR_IMPLEMENTATION_REVIEW.md`.

The finding: `LISTED_ISSUES_PAGE_URL` (F1's root) had drifted from the
exact authoritative root bound in
`V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md`
(`root=https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html`),
using the non-English page instead. The fix corrects
`LISTED_ISSUES_PAGE_URL` to that exact reviewed URL; because
`LOCATOR_STRATEGIES[F1].root_url` derives from this constant, it is
corrected automatically. No alias, fallback root, redirect-based
substitution, non-English alternative, or guessed historical root was
introduced -- the reviewed traversal rule (a unique same-domain
`data_j.xls` link from the official F1 page) and its relative-link
resolution via `urllib.parse.urljoin` are otherwise unchanged; a relative
link now correctly resolves under the `/english/` root when it is not an
absolute-path link. `ACQUISITION_IMPLEMENTATION_COMPLETE` remains `False`
and `verify_acquisition_implementation_ready()` is unchanged, so a valid
real run still stops before any filesystem/git/network access -- the F1
root fix does not, by itself, enable any Stage-A network call.
`security_type_pass`, `reconstruct_security_state`, retry policy, and the
F2-F7 locator strategies/bridge/envelope are all unchanged.

Tests: `test_f1_root_is_exact_bound_english_root` and
`test_f1_locator_strategy_root_matches_bound_constant` prove the exact
binding; `test_extract_data_j_xls_url_relative_link_resolves_against_english_root`
proves a genuinely relative `data_j.xls` href resolves under the new root
and stays same-domain. The pre-existing
`test_run_stage_a_valid_confirmation_still_stops_before_any_network_or_git`
(unchanged, exercised against the real registry) continues to prove zero
network calls, zero git calls, and no output-root creation under a valid
confirmation.

## Security-type evidence fail-closed (V9_006_LOCATOR_IMPL_HIGH_3)

GPT's exact-SHA review of the HIGH_2 remediation (`RESULT=PASS`, reviewed
SHA `ed70bc8f42beabef5aac76242a7aaba9c9ab1b6a`) resolved
`V9_006_LOCATOR_IMPL_HIGH_2`; this task remediates the third and final
finding from the original locator-implementation review:
`V9_006_LOCATOR_IMPL_HIGH_3_SECURITY_TYPE_GATE_OUT_OF_SCOPE_WEAKENING`.
Original V9_006 HIGH_2 (full semantic reconstruction/validation), original
HIGH_3 (raw provenance/content-lock), and original HIGH_4 (redirect
handling) remain `OPEN` and are explicitly out of scope here -- see
`V9_006_STAGE_A_LOCATOR_IMPLEMENTATION_REVIEW.md`.

The finding: `security_type_pass = bool(terminal_snapshot_locked)` falsely
equated "F1's mandatory terminal object was successfully locked" with
V9_005's actual `SECURITY_TYPE` evidence requirement -- domestic
ordinary-common-stock eligibility must be determinable for every
reconstructed identity/date needed by V9 without future security state,
with `UNKNOWN` failing. A locked terminal object proves nothing about
whether that classification is actually determinable.

The fix, WITHOUT implementing the full semantic validator: `compute_stage_a_
evidence()` gains an explicit `security_type_validation_pass: bool`
parameter, and `security_type_pass = bool(security_type_validation_pass)`
-- never inferred from `terminal_snapshot_locked`, family coverage, row
count, or any other proxy. Production `run_stage_a()` passes
`security_type_validation_pass=False` (with a comment explaining this is
intentionally fail-closed pending the future independently reviewed
validator), because that validator has not yet been implemented or
reviewed. `terminal_snapshot_pass` remains an independent gate based solely
on terminal-snapshot locking, unconflated with `security_type_pass`.
`canonical_identity_pass`'s formula (`bool(terminal_snapshot_locked) and
security_type_pass`) is textually unchanged -- it simply now correctly
reflects the fixed, no-longer-falsely-derived `security_type_pass`, per
this task's explicit instruction not to touch canonical-identity/effective-
date/reconstruction semantics. `ACQUISITION_IMPLEMENTATION_COMPLETE`
remains `False` and `verify_acquisition_implementation_ready()` is
unchanged, so a valid real run still stops before any
filesystem/git/network access.

Tests: `test_terminal_snapshot_locked_true_with_security_type_validation_false_fails_security_type`
proves a locked terminal snapshot cannot make `security_type_pass` true on
its own; `test_terminal_snapshot_locked_alone_can_never_make_security_type_pass`
sweeps both values of `terminal_snapshot_locked` with
`security_type_validation_pass=False` to prove `security_type_pass` stays
`False` either way;
`test_synthetic_security_type_validation_true_feeds_conjunction_independent_of_terminal_lock`
proves a synthetic `True` can drive `security_type_pass=True` even with
`terminal_snapshot_locked=False`, showing the two gates are independent
(this synthetic input tests conjunction mechanics only -- it is not a claim
that production semantic validation exists);
`test_production_security_type_validation_pass_is_hardcoded_false` is a
static-source proof (via `inspect.getsource`) that `run_stage_a()` always
passes `security_type_validation_pass=False`. All pre-existing
`compute_stage_a_evidence` call sites and the `_full_evidence` test helper
were updated to pass the new required keyword (the helper defaults it to a
clearly-commented synthetic `True`, so the pre-existing PASS/FAIL
conjunction tests continue to isolate their own conditions under test), and
a new parametrized case (`security_type_validation_pass: False`) was added
to `test_exact_pass_conjunction_false_if_any_single_condition_fails`.

## Semantic validation methodology binding (V9_006_HIGH_2, docs-only)

GPT's exact-SHA review of the HIGH_3 remediation (`RESULT=PASS`, reviewed
SHA `b2d91bd226aadfe3366d7272f868762925909b13`) resolved
`V9_006_LOCATOR_IMPL_HIGH_3`, closing the locator/inventory-contract
implementation review chain (`V9_006_STAGE_A_LOCATOR_IMPLEMENTATION=PASS`,
`V9_006_SOURCE_SLOT_LOCATOR_METHODOLOGY=PASS`). This task is a separate,
docs-only GPT methodology binding for original V9_006 HIGH_2 (full semantic
reconstruction/validation) -- it implements no parser or semantic
validator and makes no code change.

`V9_006_STAGE_A_SEMANTIC_VALIDATION_METHODOLOGY.md` records the exact GPT-
bound methodology for the `SECURITY_TYPE`, `CANONICAL_IDENTITY`,
`LISTING_TRANSITIONS`/`DELISTING_TRANSITIONS`/`MARKET_TRANSITIONS`,
`EFFECTIVE_DATE`, and `RECONSTRUCTION` Stage-A evidence items: the exact
4-character `canonical_code` serialization and grammar (including
JPX's 2024+ alphanumeric codes and the 5th-character security-type
exclusion); `AMBIGUOUS_REUSED_SECURITY_CODE` fail-closed handling for
disjoint listing episodes sharing a code; point-in-time `listed_state`/
market/`security_type_state` reconstruction with exact half-open effective-
date semantics; the three-state `security_type_state` classifier
(`ELIGIBLE_DOMESTIC_ORDINARY_COMMON` / `EXPLICITLY_INELIGIBLE` / `UNKNOWN`,
with any needed `UNKNOWN` failing `security_type_pass`); the requirement
that `listing_transition_pass`/`delisting_transition_pass`/
`market_transition_pass` be computed from parsed official transition
events (not the currently-implemented family-coverage-only proxies, and
`market_transition_pass` must not equal `listing_transition_pass` merely
as a proxy); `canonical_identity_pass`'s exact conjunction; and
`deterministic_reconstruction_pass`'s reverse/forward consistency check
(replay backward then forward from the same locked seed/transitions; the
recovered terminal state must byte-match the canonical terminal state, no
reconciliation on mismatch).

This methodology binding does not implement any of the above in code:
`src/v9_005_stage_a_jpx_probe.py`'s `reconstruct_security_state`,
`reconstruction_is_deterministic`, and `compute_stage_a_evidence`'s
`listing_transition_pass`/`delisting_transition_pass`/
`market_transition_pass`/`canonical_identity_pass`/`effective_date_pass`
computations are unchanged by this task. `security_type_pass`'s unsafe
proxy was already removed in `V9_006_LOCATOR_IMPL_HIGH_3` (production
`run_stage_a()` already hardcodes `security_type_validation_pass=False`
pending exactly the validator this document's section 4 now
methodologically binds); this task does not flip that flag. Original
V9_006 HIGH_2 remains `OPEN` pending a separate, future implementation
task that codes this exact binding, itself subject to its own GPT
exact-SHA review. The 648-record matrix, F1 `TERMINAL_SEED` role, F2
bridge derivation, F3 `YEAR` strategy, F4 ratio orientation, F5/F6/F7
strategies, retry policy, `ACQUISITION_IMPLEMENTATION_COMPLETE`, original
HIGH_3 (raw provenance), and original HIGH_4 (redirect handling) are all
unchanged.

## Signal-grid binding (contract item 5)

`verify_signal_grid_binding` is called immediately after output-root
initialization and before any Stage-A network request. It re-derives
`V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md`'s exact Git blob SHA at
the current HEAD and compares it to the bound reviewed value
`9135183b7fc5097602fa40fcda8f1b0448220244`
(`BOUND_SIGNAL_GRID_BLOB_SHA`). A mismatch raises
`V9005StageABlocked(PROBE_SIGNAL_GRID_CONTRACT_MISMATCH)` before any
network call; the `.ps1` entrypoint performs the identical check with `git
rev-parse` before its own network step. `FINAL_SIGNAL_D0` is derived
mechanically from whatever JPX trading-day calendar was actually locked --
never hard-coded -- using exactly the bound rule: `j0` = calendar index of
the first JPX trading day `>= 2018-01-01`; a date at index `j` is a V9
signal-grid `D0` iff `(j - j0) mod 3 == 0`; `FINAL_SIGNAL_D0` = the last such
`D0 <= 2025-12-31`.

## Raw content locking (contract item 3)

`lock_first_complete_payload` persists raw bytes and a full-provenance JSON
record (requested URL, resolved URL, HTTP status, retrieval timestamp, byte
length, SHA-256, source family, applicable period) atomically via
hard-link-based no-overwrite creation (the same technique already reviewed
in `src/v8k_public_source_preparation.py`); a second attempt at the same key
fails closed rather than overwriting. `ensure_locked_payload` always checks
for an existing lock before fetching, so a parser/semantic repair path
reprocesses the exact same locked bytes and never triggers a second network
request. Transport failures are classified only via the already-reviewed
`src/v8c_transport.classify_transport_exception`, retried up to the frozen
attempt/backoff policy, and unresolved retryable failures surface as
`PLUMBING_FAILURE_RETRIABLE` per `AI_REAL_EXECUTION_RUNBOOK.md`.

## Off-domain rejection (contract item 1)

`validate_jpx_url` accepts only `https://jpx.co.jp` or an exact subdomain of
it, with no credentials, no nonstandard port, and no fragment.
`fetch_once_with_retry` applies this check to both the requested URL (before
any request) and the response's final/resolved URL (before any content is
consumed), so an off-domain redirect is rejected before its body is ever
read.

## Durable output and atomic PowerShell preflight (contract items 6-7)

`initialize_output_root` requires a fresh, not-yet-existing output
directory (a durable-execution collision check per
`AI_REAL_EXECUTION_RUNBOOK.md` SS8); it is never silently reused or
overwritten. `scripts/run_v9_005_stage_a_jpx_probe.ps1` is a single
`& { $ErrorActionPreference = "Stop" ... }` scope that verifies, in order,
the repository, authoritative branch, clean tree, exact expected local
HEAD, exact remote HEAD, the design-blob binding, the output-root collision
check, and an interactively typed Stage-A confirmation token, before running
the canonical `.venv-real-execution` interpreter against the Python
entrypoint; the confirmation token is cleared from the process environment
in a `finally` block even on failure, and is never written to disk, logged,
or embedded in this file.

## What this task does not do

- No real network request was made (`NETWORK_REQUESTS=0`).
- The human's existing chat-given Stage-A authorization was not consumed or
  referenced by any code path.
- No JPX/Yahoo/J-Quants/broker data was acquired; no price or outcome was
  inspected.
- No T1 partition was generated or opened; no model was fit; no backtest
  ran; no V9 design freeze occurred; no Stage B artifact was produced.
- This implementation task itself did not modify
  `V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md`, and did not
  change retry classification/policy, F1/F2-F7 methodology, F4's ratio
  orientation, or the F7 acquisition envelope -- it only wires the already
  GPT-reviewed locator/inventory contract into code. (Earlier, separate
  methodology tasks did amend that design draft -- the
  `F1_TERMINAL_SEED_PREFREEZE_AMENDMENT` and the retry-policy binding; see
  `PROJECT_DECISION_LOG.md` for that history.)

## Next action

`GPT_EXACT_SHA_V9_006_HIGH_2_SEMANTIC_METHODOLOGY_REVIEW`: obtain GPT's
independent exact-SHA review of this docs-only semantic-validation
methodology binding
(`V9_006_STAGE_A_SEMANTIC_VALIDATION_METHODOLOGY.md`) before any
implementation task codes it. `V9_006_LOCATOR_IMPL_HIGH_1`,
`V9_006_LOCATOR_IMPL_HIGH_2`, and `V9_006_LOCATOR_IMPL_HIGH_3` are already
`RESOLVED` (GPT exact-SHA `PASS`, reviewed SHAs
`afc59fb285e09aa8c7225ce6f855d16801c67584`,
`ed70bc8f42beabef5aac76242a7aaba9c9ab1b6a`, and
`b2d91bd226aadfe3366d7272f868762925909b13` respectively), closing the
locator/inventory-contract implementation review chain
(`V9_006_STAGE_A_LOCATOR_IMPLEMENTATION=PASS`,
`V9_006_SOURCE_SLOT_LOCATOR_METHODOLOGY=PASS`). Real execution
additionally requires: this semantic-methodology review's PASS; a future,
separately reviewed implementation task that codes this exact methodology
into `src/v9_005_stage_a_jpx_probe.py` (resolving original V9_006 HIGH_2
and replacing production `run_stage_a()`'s hardcoded
`security_type_validation_pass=False`), itself subject to its own GPT
exact-SHA review PASS; PASS of the original HIGH_3 finding (raw
provenance/content-lock boundary) and the original HIGH_4 finding
(redirect-before-body-consumption), neither remediated by this task; a
future, separately reviewed task that actually implements the complete
F2-F7 acquisition pipeline and flips `ACQUISITION_IMPLEMENTATION_COMPLETE`
to `True` (not built by this task); the environment readiness ordering in
`AI_REAL_EXECUTION_RUNBOOK.md` SS16-19; and a fresh, separate, explicit
point-of-use human network authorization obtained after all of the above
(not the authorization already given in chat, which this task did not
consume).
