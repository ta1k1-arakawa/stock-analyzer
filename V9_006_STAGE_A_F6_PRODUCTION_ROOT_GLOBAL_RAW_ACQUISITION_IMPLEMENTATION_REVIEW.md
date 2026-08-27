# V9_006 Stage-A F6 production ROOT/GLOBAL raw acquisition implementation review

~~~text
task=V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_IMPLEMENTATION
status=PASS
implementation_parent_sha=0a798cc6e6d5996b458c7bda829cec0cb982b0bc
implementation_reviewed_sha=05456096909d7da30700776066f3cee94ae2d9cb
offline_only=true
real_raw_lock_execution=false
real_network_request=false
human_authorization_consumed=false
source_data_network_requests=0
GLOBAL_CHILD_FETCH_AUTHORIZED=false
GLOBAL_CHILD_FETCHED=false
GLOBAL_CHILD_CONTENT_INSPECTED=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
~~~

This document records the execution-agent implementation boundary for the
reviewed `V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_DESIGN.md`
(`REVIEWED_SHA=0a798cc6e6d5996b458c7bda829cec0cb982b0bc`, `RESULT=PASS`). It
does not call the implementation PASS. GPT-5.6 Sol remains the final
independent reviewer.

## Method implemented

`src/v9_005_stage_a_jpx_probe.py` adds:

- `TOPIX_DISCOVERY_ROOT` and `TOPIX_GLOBAL_2017_2025`: the two production
  raw-lock `applicable_period` identities from the reviewed design's section
  2, permanently distinct from the existing `F6_ROOT_STRUCTURE_DIAGNOSTIC`
  identity;
- `F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION`, the exact
  dedicated one-shot confirmation identity from the design's section 3B
  (`V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_ONE_SHOT`), and
  `F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_FILENAME`/
  `..._SCHEMA_VERSION`, the dedicated durable receipt identity;
- `_write_f6_production_acquisition_gate_receipt`, which durably publishes
  the receipt (`schema_version`, `task`, `confirmation_contract`,
  `gate_consumed=true`, `consumption_timestamp_utc`) using the same atomic,
  no-overwrite `_atomic_create` primitive already used for every raw lock in
  this module. It persists only the fixed, publicly known
  confirmation-contract identity, never the raw confirmation value the
  caller supplied;
- `run_f6_production_root_global_raw_acquisition_network`, the real-execution
  entrypoint implementing the design's mandatory sequence exactly:
  1. reject a missing/wrong confirmation before any filesystem access;
  2. `initialize_output_root` -- fails closed if `output_root` already
     exists, so an existing output root, receipt, or raw lock is rejected
     before any fetch, never silently reused or overwritten;
  3. durably publish the gate receipt before the first fetch;
  4. fetch exactly `TOPIX_ROOT_URL` once via the existing, unmodified
     `fetch_once_with_retry` policy, then immediately raw-lock the first
     complete payload as `TOPIX_DISCOVERY_ROOT` before any locator or
     semantic step; this ROOT is never refetched afterward;
  5. run the exact reviewed `parse_f6_global_child_locator` on the newly
     locked production ROOT bytes -- never
     `read_f6_root_structure_diagnostic_lock`, and never the previously
     observed diagnostic child href/URL;
  6. on locator failure of either already-reviewed failure class
     (`CHATGPT_DECISION_REQUIRED` identity/methodology-resolution failures,
     or `IMPLEMENTATION_FAILURE` malformed/corrupt/invalid-UTF-8/DOM
     failures), the ROOT lock is preserved, zero CHILD requests are made,
     and the locator's own `V9005StageABlocked` propagates unchanged
     (only its `network_request_count` is corrected to include the ROOT
     fetch, which the locator itself performs no network for);
  7. on locator success, fetch exactly its mechanically resolved child URL
     once via the identical retry policy, then immediately raw-lock the
     first complete CHILD payload as `TOPIX_GLOBAL_2017_2025`;
  8. STOP -- the function never opens, decodes, or otherwise inspects CHILD
     content, never proves coverage, never populates F6 `AVAILABLE`/
     `MISSING`, and never calls `run_stage_a` or any F1-F5/F7 acquisition
     path.

`scripts/run_v9_006_f6_production_root_global_raw_acquisition.py` is the new
CLI entrypoint. It accepts only `--output-root`; the confirmation is read
exclusively from the dedicated
`V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION` environment
variable, never argv. It reuses the exact, already-reviewed production-safe
`_production_fetcher`/`_utc_clock` from `run_v9_005_stage_a_jpx_probe.py`
verbatim -- no new transport policy is invented. Its stdout is a whitelisted
safe-field dict containing only hashes (`sha256` of the ROOT/CHILD payload
bytes and of the requested/resolved URL text), counts, statuses, and
equality booleans; it never prints a raw requested/resolved URL or any
payload content. It performs no environment/dependency bootstrap itself, per
the reviewed design's explicit deferral of that check to a future Windows
PowerShell entrypoint executed before the confirmation variable is ever
supplied.

## Authority boundary

Neither the module function nor the CLI script has a real fetcher, sleep,
clock, or credential wired to a live network by this task. Running the CLI
script directly against a real network is NOT authorized by this
implementation task. Future real execution still requires a fresh, separate,
explicit one-shot human authorization, obtained only after GPT exact-SHA
review of this implementation, per the reviewed design's section 3B.

## Targeted verification

Synthetic locked/synthetic-fetcher bytes only; no real socket, no real raw
lock, no real JPX request:

~~~text
PYTHONPATH=. pytest tests/test_v9_005_stage_a_jpx_probe.py -q
323 passed in 5.11s
git diff --check
clean
SOURCE_DATA_NETWORK_REQUESTS=0
~~~

The added tests cover, at minimum: a missing/wrong confirmation making zero
filesystem/fetch calls (including that neither the F6 diagnostic's nor
production Stage-A's own confirmation satisfies this dedicated gate); an
existing output root/receipt failing closed before any fetch; the receipt
existing durably at the moment of the first fetch call; the ROOT request
targeting exactly `TOPIX_ROOT_URL`; the ROOT raw lock existing before the
locator is invoked; the locator resolving against the production ROOT's
final `resolved_url` (not its `requested_url`, proven with a same-domain
redirect) and never invoking the diagnostic reader/identity; two different
synthetic child hrefs producing two different mechanically resolved child
URLs (proving no filename is hardcoded); a `CHATGPT_DECISION_REQUIRED`
locator failure (zero candidate anchors) and an off-domain child href,
each preserving the ROOT lock with zero CHILD requests; an
`IMPLEMENTATION_FAILURE` locator failure (invalid UTF-8 ROOT bytes)
likewise preserving the ROOT lock with zero CHILD requests; a rerun against
an already-consumed output root making zero new fetches; CHILD bytes that
are deliberately invalid UTF-8/non-spreadsheet garbage still locking
successfully, proving CHILD content is never parsed or decoded after lock;
the unchanged 3-attempt/2-retry/(5, 30)-second/no-jitter retry policy
constants, exercised end-to-end against both the ROOT and CHILD fetch
phases; that `run_stage_a` is never invoked and F6 inventory stays
`MISSING`/`ACQUISITION_IMPLEMENTATION_COMPLETE` stays `false`; that no real
socket is used; and, at the CLI layer, that missing/wrong confirmation
yields zero fetch calls and that successful safe stdout contains no raw
`TOPIX_ROOT_URL` text, no raw child href/URL text, and no payload content.

## Supplied prior design review

~~~text
REVIEWED_SHA=0a798cc6e6d5996b458c7bda829cec0cb982b0bc
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_F6_PRODUCTION_RAW_DESIGN_MEDIUM_1_LOCATOR_FAILURE_CLASS_COLLAPSE=RESOLVED
V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_DESIGN=PASS
~~~

The tuple above is the supplied exact-SHA review of the design this
implementation follows. It is not an independent PASS of this
implementation. This implementation does not change the design's root ->
locator -> child order, diagnostic-URL non-promotion, raw identities, retry
policy, the fresh one-shot human gate, pre-gate environment readiness, the
child raw-lock-then-STOP rule, the child-content-parsing prohibition, or any
F1/F2/F3/F4/F5/F7 rule.

## High-1 remediation: post-gate safe-report provenance

~~~text
finding=V9_006_F6_PRODUCTION_RAW_IMPL_HIGH_1_POST_GATE_SAFE_REPORT_PROVENANCE
REVIEWED_SHA=defe46f14fe3403e62315b145ae9e0484e599bab
CRITICAL=0
HIGH=1
MEDIUM=0
LOW=1
RESULT=BLOCK
~~~

The reviewed implementation SHA above correctly implemented the acquisition
methodology itself, but its safe failure/report provenance had three gaps:

1. **Cumulative network counts.** `run_f6_production_root_global_raw_
   acquisition_network` reported only `root_requests` (correcting a
   locator failure's `network_request_count`), but a CHILD-stage
   `fetch_once_with_retry` failure propagated with only that call's own
   partial count -- never the ROOT requests already made -- and any
   unexpected (non-`V9005StageABlocked`) exception after a completed fetch
   was not converted at all, so it would surface as
   `network_request_count=0` purely because it was not a
   `V9005StageABlocked`.
2. **Gate consumption in safe failure output.** The CLI's failure path
   never exposed `gate_consumed`, so a post-receipt failure was
   indistinguishable from a genuine pre-gate failure.
3. **Raw machine path in stdout.** Success stdout printed `receipt_path`,
   a raw local filesystem path derived from `output_root`.

The remediation is entirely inside `run_f6_production_root_global_raw_
acquisition_network`, `scripts/run_v9_006_f6_production_root_global_raw_
acquisition.py`, and the new read-only
`read_f6_production_acquisition_gate_consumed_state`:

- The executor now tracks `cumulative_requests`, updated only immediately
  after a fetch call itself returns successfully (never inside it). Every
  stage (receipt write + ROOT fetch/lock; locator; CHILD fetch/lock) is
  wrapped so a `V9005StageABlocked` has its `network_request_count`
  corrected to `cumulative_requests + exc.network_request_count` (additive,
  not overwritten -- a `fetch_once_with_retry` exception already carries
  that call's own accurate partial-attempt count) and any other exception
  is converted, fail-closed, into `V9005StageABlocked(IMPLEMENTATION_
  FAILURE, network_request_count=cumulative_requests)`. No fetch/retry
  methodology changed -- only already-mechanically-known counts are now
  correctly propagated and never silently dropped to 0.
- `read_f6_production_acquisition_gate_consumed_state` is a new read-only,
  safe helper: it never authorizes, skips, deletes, or resets anything, and
  is used only for reporting. It returns `True` only when the exact
  reviewed receipt is present and structurally valid with
  `gate_consumed == True`; `False` when the receipt clearly does not exist;
  or `None` ("unknown" in the safe report) when the path exists but cannot
  be conclusively read/validated -- never a fabricated value either way.
  The CLI derives `gate_consumed` from this durable, on-disk state for
  every failure after the confirmation check itself passes, so a
  post-receipt failure can never look `PRE_GATE` merely because the Python
  call raised. For a missing/wrong confirmation specifically -- the one
  case genuinely before any filesystem access -- the CLI reports
  `gate_consumed=false` directly, without even statting the path, so an
  unrelated pre-existing receipt at the same `output_root` can never leak
  into that deterministic pre-gate report. `authorization_reusable=false`
  and `second_execution_allowed=false` are now included, unconditionally,
  in every report.
- CLI stdout (both success and failure) no longer includes `receipt_path`
  or any other raw machine path/URL/payload; internal filesystem use
  (writing the receipt under `output_root`) is unchanged.

Verified test additions cover: ROOT exhaustion (`network_request_count ==
MAX_ATTEMPTS`); ROOT success + CHILD exhaustion (`1 + MAX_ATTEMPTS`); ROOT
retry-success + a CHILD non-retryable failure on its second attempt
(`2 + 2 == 4`); an unexpected (plain `RuntimeError`, not
`V9005StageABlocked`) exception injected at each of three post-receipt
points -- between ROOT fetch and ROOT lock, inside the locator stage, and
between CHILD fetch and CHILD lock -- each asserting the correct cumulative
count and `IMPLEMENTATION_FAILURE`; the existing locator-failure and
rerun-after-consumed-receipt tests now also assert
`read_f6_production_acquisition_gate_consumed_state(...) is True`; the new
reader's tri-state behavior (absent, present-but-no-receipt, and corrupt
receipt) directly; and, at the CLI layer, that `gate_consumed`/
`authorization_reusable`/`second_execution_allowed` appear in every
failure and success report, that a post-receipt CLI failure reports
`gate_consumed=true` with the correct count, and that neither
`receipt_path` nor the local `output_root` path string appears anywhere in
success or failure stdout.

~~~text
PYTHONPATH=. pytest tests/test_v9_005_stage_a_jpx_probe.py -q
331 passed in 5.48s
git diff --check
clean
SOURCE_DATA_NETWORK_REQUESTS=0
~~~

This remediation does not change the confirmation identity, gate-receipt
schema/atomic/no-overwrite behavior, the ROOT -> lock -> locator -> CHILD
ordering, `TOPIX_DISCOVERY_ROOT`/`TOPIX_GLOBAL_2017_2025`, locator
methodology/failure taxonomy, diagnostic-URL non-use, retry policy, the
child-content prohibition, inventory/fanout,
`ACQUISITION_IMPLEMENTATION_COMPLETE=false`, or any F1-F5/F7 behavior. It
is not self-called PASS by the execution agent. The next action is GPT
exact-SHA independent review of this remediation.

## High-1A remediation: fail-closed gate-state reader

~~~text
finding=V9_006_F6_PRODUCTION_RAW_IMPL_HIGH_1A_GATE_STATE_READER_FAIL_CLOSED
REVIEWED_SHA=29562ec0bf081d3d8430b6488508af7c320501a4
CRITICAL=0
HIGH=1
MEDIUM=0
RESULT=BLOCK
~~~

The supplied review found that the reporting-only durable gate-state reader
was not genuinely fail-closed: an unguarded receipt existence probe could
raise on ordinary filesystem uncertainty, and a structurally invalid
receipt with gate_consumed=false (or another non-bool value) could be
reported as false. False must mean only that receipt absence was
mechanically proven; every present receipt whose exact schema, task,
confirmation identity, consumed value, or timestamp cannot be proven valid
must be unknown.

The one-finding remediation changes only the reader and the CLI reporting
boundary. The reader now uses guarded lstat/read/JSON validation, accepts
only a regular receipt file, checks the exact field set, schema version,
task, confirmation contract, canonical UTC timestamp, and gate_consumed is
True, and returns None for any uncertainty or invalidity. The CLI now
contains a defense-in-depth reporting wrapper: even if the reader itself
unexpectedly raises, failure stdout remains whitelist JSON with
gate_consumed=unknown and no exception detail or local path.

Synthetic offline tests prove absent receipt=false and exact valid receipt=true.
Malformed JSON, false/non-bool gate_consumed, wrong task, identity/schema,
invalid timestamp, lstat/read PermissionError, and other unproven state
return None. A CLI failure whose reader raises still emits safe JSON with
gate_consumed=unknown and no output-root path, error text, or traceback.
The targeted command used the repository development virtual environment
because pytest was not on PATH:

~~~text
PYTHONPATH=. .venv\Scripts\python.exe -m pytest tests/test_v9_005_stage_a_jpx_probe.py -q
341 passed in 8.76s
git diff --check
clean
SOURCE_DATA_NETWORK_REQUESTS=0
~~~

No confirmation identity, receipt-write schema/atomic no-overwrite
behavior, ROOT->lock->locator->CHILD ordering, locator methodology, retry
policy, network semantics, output-root semantics, F1-F5/F7 behavior,
human-gate state, or real acquisition changed. This remediation is
REMEDIATED_AWAITING_GPT_REVIEW and is not self-called PASS by the execution
agent.

## GPT final independent implementation review

~~~text
REVIEWED_SHA=05456096909d7da30700776066f3cee94ae2d9cb
CRITICAL=0
HIGH=0
MEDIUM=0
RESULT=PASS
V9_006_F6_PRODUCTION_RAW_IMPL_HIGH_1A_GATE_STATE_READER_FAIL_CLOSED=RESOLVED
V9_006_F6_PRODUCTION_RAW_IMPL_HIGH_1_POST_GATE_SAFE_REPORT_PROVENANCE=RESOLVED
V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_IMPLEMENTATION=PASS
~~~

The supplied GPT-5.6 Sol exact-SHA adjudication closes the implementation
review chain. This PASS creates no execution authority: the current stage is
post-implementation-review, pre-human-gate/pre-execution readiness;
`GLOBAL_CHILD_FETCH_AUTHORIZED=false`, network and acquisition flags remain
false, `ACQUISITION_IMPLEMENTATION_COMPLETE=false`, `V9_design_frozen=false`,
and `future_profitability_established=false`.
