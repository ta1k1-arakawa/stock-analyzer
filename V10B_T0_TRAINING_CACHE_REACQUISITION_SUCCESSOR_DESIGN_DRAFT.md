# V10B T0 Training-Cache Reacquisition Successor Design Draft

```text
study_identity=V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR
predecessor=V9_009_T0 / V10A_T0_CALENDAR_INPUT_BINDING_BRIDGE
design_status=FROZEN_CANDIDATE_AWAITING_GPT_EXACT_SHA_REVIEW
evidence_role=TRAINING_INPUT_PROVENANCE_ONLY
profitability_evidential_capacity=ZERO
V10B_NETWORK_ACQUISITION_AUTHORIZED=false
V10B_T0_AUTHORIZED=false
V10B_HISTORICAL_EVALUATION_AUTHORIZED=false
PRIVATE_SEALED_ACCESS_AUTHORIZED=false
future_profitability_established=false
```

## 1. Purpose and terminal predecessor boundary

V10B is a new successor study whose sole changed component is the
replacement of an unavailable historical training-cache artifact with a
newly acquired and prospectively locked training-cache identity. It is not a
retry, repair, reinterpretation, or continuation of the missing V9_009
cache.

The exact historical training manifest identity remains immutable historical
provenance:

```text
V9_009_TRAINING_MANIFEST_SHA256=72ae3db1186f2c9c113b1bafe1d37fb74a5627ac7ceed1dfc2473a24e060de85
V9_009_EXPECTED_HISTORICAL_TRAINING_PAYLOAD_COUNT=283
```

That manifest hash and expected count must never be overwritten, relabeled as
the V10B manifest, or used to imply that the old cache was recovered.

The approved metadata-only recovery established that the original known
training root was absent; the evaluation root existed with raw-file count
`300`; `V4_UNIVERSE.csv` existed; there were zero exact-name training
directory candidates, zero transfer ZIP candidates, zero extracted transfer
directory candidates, and zero generic candidates containing
`cache_manifest.json` plus `283` raw JSON files. One local/removable drive
was searched. No candidate paths were printed and no cache manifest, price
payload, universe content, or ZIP content was read. The recovery recorded
zero network requests, zero T0 runs, and zero human-gate consumption.

Accordingly:

```text
MISSING_REQUIRED_FROZEN_INPUT=TRAINING_CACHE
V9_009_V10A_EXISTING_INPUT_T0=TERMINAL_BLOCKED
V9_009_V10A_T0_SCIENTIFIC_RESULT=NOT_OBSERVED
V9_009_V10A_DATA_QUALITY_OR_STRATEGY_FAILURE=false
V9_009_V10A_PROFITABILITY_FAILURE=false
```

The old evaluation identity remains separate and unchanged. Its contents are
not inspected or reacquired by this design:

```text
EVALUATION_MANIFEST_SHA256=797265bf671af2245a342051ffad02aa2929d67ba885945e7762149649148aa5
```

The V10A canonical calendar and its reviewed input-binding bridge remain the
sole calendar authority. V10B does not regenerate, reopen, or reinterpret
that calendar study.

## 2. Inherited scientific methodology

V10B inherits the V9_009 and V10A scientific methodology unchanged. In
particular, this design does not change:

- the fixed `V4_UNIVERSE.csv` identity or ticker-list identity;
- V9_009 feature definitions, causal D0 scoreability, or D1/D2/D3 target
  semantics;
- signal years, including the inherited evaluation roles;
- T0 years `2020..2025`;
- Ridge or LightGBM parameters;
- pooled cross-sectional scoring, `TOP1` selection, or the T0 STOP/CONTINUE
  thresholds;
- the V10A JPX calendar authority, session/date semantics, or calendar input
  binding;
- evaluation-cache identity, costs, slippage, portfolio mechanics, or the
  stopping rule; or
- the no-hyperparameter-search rule.

The only successor change is the identity and acquisition provenance of the
training cache. A different observed successful-payload count is not a
parameter, threshold, or tuning target. If the acquired cache cannot support
the inherited T0 semantics, the result is `DATA_INCOMPATIBLE`; it is not
permission to acquire again, choose a preferred subset, or modify the
methodology.

The fixed public universe identities remain:

```text
V4_UNIVERSE_CSV_SHA256=d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997
V4_TICKER_LIST_SHA256=12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7
V4_UNIVERSE_TICKER_COUNT=300
```

The exact ticker order is the canonical order supplied by the existing V4
universe/ticker-list identity. It is not redrawn, sorted by observed
availability, filtered by outcomes, or otherwise selected during V10B.

## 3. Frozen public acquisition source

The only source is the fixed Yahoo Finance chart endpoint and the fixed
historical request semantics below:

```text
provider=Yahoo_Finance_chart
host=query1.finance.yahoo.com
path=/v8/finance/chart/{ticker}.T
period1=1420070400
period2=1577836800
interval=1d
events=div,splits
includeAdjustedClose=true
logical_range=2015-01-01..2019-12-31
```

The endpoint, query, period, interval, event settings, and fixed `300`
universe are part of the V10B identity. There is no alternate provider,
ticker substitution, period substitution, favorable subset, provider
fallback, or post-outcome refetch.

The request path is formed only from the fixed canonical ticker order. Any
future implementation must validate the fixed public universe identity
before network access and must not accept a caller-supplied provider,
universe, ticker list, date window, or query override.

## 4. Operation class and exact stage sequence

V10B public transport is classified as `RETRIABLE_PUBLIC_PLUMBING`. This
classification permits only the bounded transport behavior in Section 5; it
does not authorize private/sealed access, T0, historical evaluation,
production trading, or profitability claims.

The complete successor sequence is:

```text
1  DESIGN
2  GPT exact-SHA design review
3  DESIGN_REVIEW_PASS only if CRITICAL=0/HIGH=0/MEDIUM=0
4  explicit human V10B design-freeze approval bound to that exact reviewed design commit
5  durable V10B design-freeze approval record committed to Git
6  GPT exact-SHA review of the freeze-approval record/provenance
7  acquisition implementation/reuse audit
8  targeted synthetic tests
9  GPT exact-SHA implementation review
10 Phase A NO-NETWORK preflight
11 fresh point-of-use human authorization for V10B public acquisition
12 Phase B bounded real Yahoo acquisition exactly once
13 Phase C NO-NETWORK safe inspection
14 GPT acquisition adjudication
15 promote exact new training-cache provenance
16 successor T0 input-binding implementation requiring that exact manifest
17 GPT exact-SHA review of that successor implementation
18 only then a fresh T0 Phase A may be considered
```

No T0, evaluation, model fit, backtest, outcome read, or profitability
calculation is permitted in steps 1--17. This design itself consumes no
human gate and grants no network authority.

### Design review and human freeze approval

V10B is a new successor study. GPT must first independently review the exact
design commit and return `CRITICAL=0`, `HIGH=0`, and `MEDIUM=0`; only that
result is `DESIGN_REVIEW_PASS`. A GPT PASS alone does not freeze the design
and does not authorize implementation.

After that PASS, an explicit human V10B design-freeze approval must bind all
of the following safe values to the exact reviewed design:

```text
study_identity=V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR
exact_gpt_reviewed_design_commit=<exact GPT-reviewed PASS design SHA>
exact_design_git_blob=<exact reviewed design blob SHA>
approval_scope=DESIGN_FREEZE_ONLY
implementation_phase_only=true
network_acquisition_authorized=false
T0_authorized=false
historical_evaluation_authorized=false
private_sealed_access_authorized=false
```

Vague approval of a latest design, branch, or task is not sufficient. The
approval must be recorded as a durable `V10B_DESIGN_FREEZE_APPROVAL.json`
artifact committed separately from the frozen design commit. This artifact
is not created in this design task. Its required future schema is:

```text
schema_version=V10B_DESIGN_FREEZE_APPROVAL_V1
study=V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR
artifact_role=DESIGN_FREEZE_APPROVAL
frozen_design_git_commit=<same exact GPT-reviewed PASS design SHA>
frozen_design_git_blob_sha=<same exact reviewed design blob SHA>
design_document=V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR_DESIGN_DRAFT.md
final_independent_review_result=PASS_CRITICAL_0_HIGH_0_MEDIUM_0_LOW_0
final_independent_review_design_commit=<same exact design SHA>
approval_status=APPROVED
human_design_freeze_complete=true
approval_scope=DESIGN_FREEZE_ONLY
approval_artifact_authorizes_implementation_phase_only=true
network_acquisition_authorized=false
t0_authorized=false
historical_evaluation_authorized=false
private_sealed_access_authorized=false
implementation_performed_by_this_artifact=false
methodology_change_after_freeze_requires=NEW_STUDY_REQUIRED
```

The approval artifact's own Git commit is not the frozen design commit. GPT
must independently exact-SHA review the approval artifact and its
provenance. Only that review PASS permits the acquisition implementation/
reuse audit and synthetic-test work. The design-freeze authority and the
later real-network acquisition authority are distinct and cannot be reused
for each other.

Once frozen, any semantic change to provider, query, range, retry policy,
universe, content locking, T0 methodology, or another frozen design element
requires `NEW_STUDY_REQUIRED`.

### Phase A: no-network acquisition preflight

Before the fresh authorization and before any Yahoo request, a future
reviewed implementation must verify, without reading research payload
contents:

- the authoritative repository, branch, exact reviewed implementation
  binding, clean tree, and current Git provenance;
- the exact V4 universe and ticker-list file identities and canonical order;
- the V10A calendar artifact/receipt authority remains unchanged;
- exclusive new attempt, output, payload, manifest, and evidence roots do
  not already exist or conflict;
- the fixed endpoint/query/range and maximum-attempt policy are the only
  operational inputs; and
- the canonical protected environment and synthetic parser readiness are
  available as required by the repository runbook.

Any governance, provenance, path-safety, or readiness failure stops before
the network boundary. It is not a data-quality result and does not consume
the future acquisition authorization.

### Phase B: one bounded acquisition

After Phase A passes, fresh point-of-use human authorization is required.
The authorization is scoped to exactly one V10B public acquisition under
this design. Phase B invokes the reviewed implementation once. Its internal
transport attempts are bounded per ticker as specified below; they are not
separate acquisition executions.

### Phase C and adjudication

Phase C is no-network and read-only. It may inspect only safe aggregate
facts, exact hashes/sizes, manifest structure, payload-file metadata, and
repository provenance needed to verify the frozen acquisition contract. It
must not open evaluation/outcome data, private/sealed data, or use observed
results to choose a cache.

GPT adjudicates the safe Phase-B/Phase-C facts. Only after a GPT PASS may
the exact new manifest and its exact payload set be promoted as V10B
training-input provenance. A later T0 input-binding implementation must
bind that exact promoted manifest identity; it may not bind the historical
V9_009 manifest as a substitute.

## 5. Bounded transport, retry, and content-lock rules

The maximum is exactly three transport attempts per fixed ticker. A retry is
permitted only when the preceding attempt fails by:

```text
transport exception OR HTTP 429 OR HTTP 5xx
```

No retry is permitted for a complete successful payload, HTTP/semantic/data-
quality failure, parser failure, T0 incompatibility, or observed ticker
failure. There is no provider fallback, date-window fallback, repair,
redraw, or retry-to-obtain-a-preferred-success-count rule.

For each ticker, the first complete successful response body is immediately
content-locked as raw bytes before parsing or semantic use. The future
implementation must persist those exact bytes under the exclusive attempt
root and compute their SHA-256 and byte count before interpreting the
payload. The same raw bytes, not a refetched response, are the only input to
later parsing or software repair.

An HTTP success is not a license to refetch. If its complete payload cannot
support the fixed parser/semantic contract, that ticker's failure is the
mechanically observed result of the frozen policy. The final acquired set
is authoritative even when some fixed tickers fail. A cache that is
incomplete or otherwise unable to support inherited T0 is
`DATA_INCOMPATIBLE`, not a reason to start a second acquisition.

Every fixed ticker is attempted in the same canonical order under the same
rules. `successful_ticker_count` and `failed_tickers` are observed facts,
not acceptance thresholds and not selection controls. The implementation
must not stop early after a favorable count or refetch after an unfavorable
count.

## 6. V10B manifest and payload identity

The new manifest is a deterministic provenance artifact. Its exact final
SHA-256 and the observed payload count are unknown until the authorized
acquisition completes and must never be guessed or prefilled. The manifest
contains at least these exact top-level fields:

```text
schema_version
complete
universe_mode
universe_csv_sha256
ticker_list_sha256
ticker_count
ticker_order
price_from
price_to
query_specification
payloads
network_audit
successful_ticker_count
failed_tickers
payload_hash_list_sha256
```

The future implementation must freeze the schema version as a new V10B
value before acquisition and must reject extra or missing fields once that
implementation contract is reviewed. The listed values bind as follows:

```text
universe_mode=FIXED_V4_300
universe_csv_sha256=d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997
ticker_list_sha256=12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7
ticker_count=300
price_from=2015-01-01
price_to=2019-12-31
```

`complete=false` means that acquisition has not terminally processed every
one of the `ticker_count` fixed tickers under the frozen retry policy, or
that a crash or interruption occurred before the full fixed universe reached
terminal transport states. `complete=true` means that every fixed canonical
ticker reached exactly one terminal transport outcome under the frozen
policy. It does not require `ticker_count` successful payloads, and
`failed_tickers` may be non-empty.

The terminal manifest must mechanically satisfy all of the following:

```text
successful_ticker_count + len(failed_tickers) == 300
accepted_ticker_set ∩ failed_ticker_set == ∅
accepted_ticker_set ∪ failed_ticker_set == exact FIXED_V4_300 ticker set
```

The accepted-payload ticker set and failed-ticker set are the exact observed
partition produced after all 300 fixed tickers reach terminal states. Both
lists preserve canonical V4 order where ordering applies, and no ticker may
be manually added or removed. `successful_ticker_count` is the exact
observed count of accepted first-complete payloads; acquisition imposes no
minimum success count and does not require the historical count `283`.

This preserves the inherited V4 acquisition meaning of `complete` and does
not change inclusion or exclusion methodology. After terminal acquisition,
the cache identity, observed success count, failed membership, and exact
payload bytes are frozen as provenance. Whether that cache can support the
inherited T0 semantics is a later `DATA_INCOMPATIBLE`/input-binding
decision, and that decision can never trigger reacquisition or refetch.

Each accepted payload entry binds exactly:

```text
ticker
relative_path
sha256
byte_count
```

`relative_path` is a deterministic path beneath the exclusive V10B payload
root and is never an absolute path or a path containing traversal. The
payload list and failed-ticker list preserve canonical V4 ticker order.
`network_audit` records only safe request-attempt/result facts needed to
verify the fixed transport policy; it contains no raw response bytes,
credentials, private identities, research outcomes, prices, or unbounded
exception text.

The manifest canonical bytes are UTF-8 JSON with `ensure_ascii=false`,
`sort_keys=true`, `separators=(',', ':')`, `allow_nan=false`, and exactly
one final LF. No manifest self-hash is inserted into the bytes. The
`payload_hash_list_sha256` is the SHA-256 of the same canonical JSON
serialization applied to the ordered payload-entry list, with no
reordering or normalization after hashing. The final manifest file
SHA-256 is computed over its exact durable bytes.

The exact raw payload bytes, their sizes, their SHA-256 values, the exact
manifest bytes, and the final successful/failed ticker aggregates are the
only authoritative V10B acquisition identity. No raw payload, price, or
outcome is copied into `PROJECT_STATE.md` or `PROJECT_DECISION_LOG.md`.

## 7. Failure and stopping semantics

The failure taxonomy is operational/data-boundary classification, not a
scientific verdict:

```text
PLUMBING_FAILURE_RETRIABLE = transport exception, HTTP 429, or HTTP 5xx before a complete payload
DATA_INCOMPATIBLE = fixed cache cannot support inherited T0 after the frozen acquisition
GOVERNANCE_FAILURE = provenance, authority, path, gate, or scope failure
IMPLEMENTATION_FAILURE = unexpected reviewed-tooling or wrapper defect
```

Transport retries are exhausted only within the per-ticker maximum. A
complete payload is never refetched because parsing, semantics, cache
completeness, or a later T0 check is unfavorable. A Phase-B crash, partial
output, or consumed authorization does not authorize deletion, reset,
overwrite, or reuse of the attempt. The applicable durable evidence is
inspected by Phase C and returned to GPT for adjudication.

No classification above emits a T0 `STOP` or `CONTINUE` result. T0 remains
outside this design until the exact new manifest has been promoted and a
separately reviewed successor T0 input-binding implementation has passed.

## 8. Evaluation and calendar boundaries

V10B does not reacquire the existing evaluation cache and does not inspect
its contents. If later point-of-use validation proves its exact identity is
not the recorded one, the process stops and returns to GPT; no automatic
repair or reacquisition is authorized by this design.

The V10A canonical calendar remains unchanged and authoritative for any
later T0 input-binding bridge. V10B does not generate, inspect, or modify
calendar dates, sessions, anchors, or market-close values.

## 9. Authority and non-claims

```text
V10B_NETWORK_ACQUISITION_AUTHORIZED=false
V10B_T0_AUTHORIZED=false
V10B_HISTORICAL_EVALUATION_AUTHORIZED=false
PRIVATE_SEALED_ACCESS_AUTHORIZED=false
V10A_T0_AUTHORIZED=false
future_profitability_established=false
```

The design-freeze approval plus its GPT exact-SHA PASS are required before
acquisition implementation/reuse audit and synthetic tests begin. A future
fresh point-of-use human authority is required immediately before the single
Phase-B public acquisition. No authorization from V9, V10, or V10A is
reused, and design-freeze approval is not network authority.

Training-cache reacquisition has zero profitability evidential capacity. No
T0 outcome, strategy result, data-quality strategy failure, or profitability
conclusion is established by this design or by the metadata-only recovery.

## 10. Design acceptance and next action

GPT exact-SHA review must verify that the fixed provider/query/range,
universe identity/order, per-ticker retry boundary, first-complete-payload
content lock, deterministic manifest, safe evidence surface, no-refetch
stopping rule, successor stage sequence, exact design-freeze approval
binding, and separation of design-freeze versus acquisition authority are
mechanically closed without changing V9/V10A methodology.

After design PASS, the next action is explicit human design-freeze approval
for the exact reviewed design, followed by its durable approval record and
GPT exact-SHA provenance review. Only then may the separately reviewed
acquisition implementation/reuse audit and synthetic tests begin. Until all
of those steps and a fresh acquisition human gate are complete, no network
request, cache creation, T0, historical evaluation, private/sealed read, or
profitability operation may occur.

```text
V10B_DESIGN_FROZEN=false
V10B_HUMAN_DESIGN_FREEZE_APPROVED=false
V10B_NETWORK_ACQUISITION_AUTHORIZED=false
V10B_T0_AUTHORIZED=false
V10B_HISTORICAL_EVALUATION_AUTHORIZED=false
T0=NOT_RUN
HISTORICAL_EVALUATION=NOT_PERFORMED
FUTURE_PROFITABILITY=UNESTABLISHED
NEXT_ACTION=GPT_EXACT_SHA_V10B_T0_TRAINING_CACHE_REACQUISITION_SUCCESSOR_DESIGN_REVIEW
```
