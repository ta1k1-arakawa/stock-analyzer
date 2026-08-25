# V9_006 Stage-A F6 root structure probe network executor review

```text
task=V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_NETWORK_EXECUTOR_IMPLEMENTATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
network_executed=false
network_authorization_flag_set=false
human_authorization_consumed=false
production_stage_a_authorized=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

This implements only the executable network-request plumbing for the F6
root-structure diagnostic. It performs no real network request during
implementation or tests, sets no network-authorization flag, consumes no
human authorization, and does not authorize production Stage A, a
GLOBAL-child fetch, or F5/any other source. Future real execution still
requires its own fresh, explicit, one-shot human authorization obtained
AFTER GPT exact-SHA review of this commit, exactly as fixed by
`V9_006_STAGE_A_F6_PUBLIC_ROOT_STRUCTURE_PROBE_DESIGN.md`.

## What was added (`src/v9_005_stage_a_jpx_probe.py`)

- `F6_ROOT_STRUCTURE_PROBE_CONFIRMATION = "V9_006_F6_ROOT_STRUCTURE_PROBE_ONE_SHOT"`,
  a dedicated one-shot confirmation constant, distinct from production
  Stage-A's `CONFIRMATION`. The production token does not satisfy this
  gate (`test_f6_root_structure_network_production_stage_a_confirmation_
  does_not_satisfy_this_gate`).
- `run_f6_root_structure_probe_network(*, output_root, confirmation,
  fetcher, sleep, clock)`: the network executor.
  1. Checks `confirmation == F6_ROOT_STRUCTURE_PROBE_CONFIRMATION` first,
     before any filesystem access or fetcher call; a mismatch raises
     `V9005StageABlocked(GOVERNANCE_FAILURE)`.
  2. Calls `initialize_output_root(output_root)` exactly once. That
     existing function itself fails closed if `output_root` already
     exists, so rerunning against an already-used `output_root` fails
     closed rather than acquiring/refetching -- no new code was needed for
     this.
  3. Requests exactly `TOPIX_ROOT_URL` via the existing, unmodified
     `fetch_once_with_retry(TOPIX_ROOT_URL, fetcher, sleep)` -- the exact
     reviewed retry/backoff/redirect/off-domain-rejection policy, verbatim.
     No other URL is ever constructed, discovered, or requested; no href is
     ever followed; no child object is ever fetched.
  4. On the first complete payload, immediately calls the existing
     `lock_first_complete_payload(...)` with
     `source_family=SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE`,
     `applicable_period=F6_ROOT_STRUCTURE_DIAGNOSTIC`,
     `requested_url=TOPIX_ROOT_URL` -- before any parsing.
  5. Only then calls the already-reviewed offline seam,
     `run_f6_root_structure_probe_offline(root)`, to parse the just-locked
     bytes. No parser logic is duplicated. A parser/extraction failure
     after the raw lock exists never triggers a refetch or a child
     request -- the raw lock and whatever deterministic artifact the
     offline seam produces (`STRUCTURE_CAPTURED`, `STRUCTURE_AMBIGUOUS`, or
     `STRUCTURE_EXTRACTION_FAILED`) are both preserved.
  6. Returns the offline artifact plus `network_request_count` (the exact
     `fetch_once_with_retry` attempt count) -- no raw payload, page text,
     or index value is added.

## What was added (`scripts/run_v9_006_f6_root_structure_probe.py`, new)

A thin CLI wrapper, mirroring `scripts/run_v9_005_stage_a_jpx_probe.py`'s
reviewed pattern:

- `--output-root` is the only required CLI input.
- The confirmation token is read only from the
  `V9_006_F6_ROOT_STRUCTURE_PROBE_CONFIRMATION` environment variable --
  never accepted on argv, never defaulted to approved, never hardcoded.
  Missing or wrong confirmation prints a safe `GOVERNANCE_FAILURE` failure
  report and makes zero fetch calls.
- The real HTTP fetcher and UTC clock are NOT duplicated: this script
  imports `_production_fetcher` and `_utc_clock` directly from
  `scripts/run_v9_005_stage_a_jpx_probe.py` (same directory, unmodified),
  reusing the exact already-reviewed production-safe fetch/redirect
  primitive rather than loosening or reimplementing that policy.
- It never imports or calls production Stage-A's orchestration entrypoint
  (`run_stage_a`) -- proven both by a static source-text check and by every
  live CLI test.
- stdout prints only `status`, `label_occurrence_count`, `requested_url`,
  `resolved_url`, `http_status`, `byte_length`, `sha256`,
  `retrieval_timestamp_utc`, `network_request_count`, and the derived
  `artifact_path` -- built from an explicit safe-field allowlist, so
  `occurrences`, `anchors`, raw `href`, raw payload bytes, page text, and
  index values can never reach stdout even when a future field is added to
  the artifact. They remain only in the durable diagnostic artifact file
  for later GPT review.

## Tests (`tests/test_v9_005_stage_a_jpx_probe.py`)

Two new offline-only sections (every fetcher is synthetic; no test opens a
real socket or performs a real network request) prove, at minimum: wrong
confirmation fails closed before any filesystem or fetcher call; production
Stage-A's confirmation does not satisfy this gate; exactly `TOPIX_ROOT_URL`
is requested, exactly once, with no second/child URL ever requested; a
synthetic same-domain redirect preserves `requested_url=TOPIX_ROOT_URL`
distinct from the redirected `resolved_url`; the raw lock exists before the
offline parser seam is invoked; a successful payload produces the reviewed
offline-seam artifact on disk; `STRUCTURE_AMBIGUOUS` still produces a
durable artifact with no refetch; `STRUCTURE_EXTRACTION_FAILED` after a
complete payload preserves the raw lock with no refetch; a retryable
transport failure before the payload uses the existing retry count/backoff
unchanged; an exhausted transport failure produces no diagnostic artifact
claiming structure; an off-domain redirect is rejected under the existing
policy; rerunning against an already-existing `output_root` fails closed
rather than acquiring/refetching; `run_stage_a` is never invoked; no real
socket is ever opened; and `ACQUISITION_IMPLEMENTATION_COMPLETE` remains
`False`. The CLI section separately proves: it never imports
`run_stage_a` (static check); missing and wrong (including the production)
confirmation each make zero fetch calls and print only a safe
`GOVERNANCE_FAILURE` report; and safe stdout -- on both a captured and an
extraction-failure result -- excludes `occurrences`, `anchors`, the raw
`href` (entity spelling included), the matched label text, anchor visible
text, and unrelated numeric/date table text, while the durable artifact
file itself still exists on disk.

`PYTHONPATH=. pytest tests/test_v9_005_stage_a_jpx_probe.py -q`: 217 passed
(196 existing + 21 new). `git diff --check`: clean.
`SOURCE_DATA_NETWORK_REQUESTS=0` for this implementation task.

## Exact GPT review preceding this implementation

```text
REVIEWED_SHA=f236be6774859151acb0f5328d269bedf8fef2d5
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_OFFLINE_IMPLEMENTATION=PASS
V9_006_F6_ROOT_OFFLINE_MEDIUM_1_DOUBLE_HTML_ENTITY_DECODE=RESOLVED
```

## What this implementation does not do

No real network request was made. No network-authorization flag was set
`true`; `V9_006_STAGE_A_NETWORK_AUTHORIZED` remains `false`. No human
authorization was consumed. No production Stage-A execution, GLOBAL-child
fetch, or F5/other-source fetch is authorized or possible through this
code path. `ACQUISITION_IMPLEMENTATION_COMPLETE` remains `False`;
`V9_006_STAGE_A_IMPLEMENTATION` remains `BLOCK`. This does not itself
authorize any future real acquisition of the diagnostic raw payload --
that still requires its own fresh, explicit, one-shot human authorization
at the point of use, exactly as fixed by
`V9_006_STAGE_A_F6_PUBLIC_ROOT_STRUCTURE_PROBE_DESIGN.md`, obtained only
after independent GPT exact-SHA review of this commit.
