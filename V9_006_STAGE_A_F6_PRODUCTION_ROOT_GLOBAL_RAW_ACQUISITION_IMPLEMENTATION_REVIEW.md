# V9_006 Stage-A F6 production ROOT/GLOBAL raw acquisition implementation review

~~~text
task=V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_IMPLEMENTATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
implementation_parent_sha=0a798cc6e6d5996b458c7bda829cec0cb982b0bc
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
F1/F2/F3/F4/F5/F7 rule. The next action is GPT exact-SHA independent review
of this implementation; the execution agent does not call it PASS.
