# V9_004 free-data source feasibility audit

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
audit_scope=SOURCE_AND_IMPLEMENTATION_FEASIBILITY_ONLY
design_change_authorized=false
price_outcomes_observed=false
```

## Scope and evidence boundary

This audit inspected repository code and the supplied official-source facts;
it made no Yahoo, JPX, J-Quants, broker, private, sealed, or T1 request. It
does not approve the candidate architecture, alter the design, or establish
profitability.

## GPT-supplied official source evidence

The supplied official JPX evidence is recorded in
`V9_004_EXTERNAL_SOURCE_EVIDENCE.md`. It establishes availability of historical
listing-change archives, approximately 11 years of delisting history,
split/right-treatment archives, and TOPIX historical index values. It does not
demonstrate a complete free official individual-stock OHLCV dataset for
2016-09 through 2025.

## Proposed free-first architecture assessment

| Component | Candidate source | Classification | Basis and limit |
| --- | --- | --- | --- |
| Security master / PIT membership | Official JPX archives and monthly listing-change materials | PARTIAL_NOT_PROVEN | Listing and delisting history exists, but a complete daily source contract is not yet demonstrated. |
| Corporate-action ratios | Official JPX split/right-treatment archives | PARTIAL_NOT_PROVEN | Archive availability is supplied; ratio completeness, effective-date binding, and mapping to canonical identities remain unproven. |
| TOPIX | Official JPX historical index values | FEASIBLE_FROM_EXISTING_EVIDENCE | Supplied evidence identifies historical index values; later source-contract binding is still required. |
| Individual OHLCV | Existing Yahoo Chart/raw-OHLCV infrastructure | REQUIRES_PUBLIC_NETWORK_COVERAGE_PROBE | Parser/integrity infrastructure exists, but 2016-2025 coverage for all needed active and delisted identities is not proven. |

## Point-in-time universe feasibility

`FREE_PIT_UNIVERSE_FEASIBILITY=PARTIAL_NOT_PROVEN`.

The supplied JPX archives support the inference that a reconstruction path may
exist without retrospectively using today's constituent list. They do not yet
prove a complete, deterministic daily 2017-2025 membership history. Required
source-contract elements remain unresolved:

| Requirement | Status | Reason |
| --- | --- | --- |
| 2017-01-01 starting-state reconstruction | PARTIAL_NOT_PROVEN | Listing-change archives alone do not demonstrate an authoritative starting snapshot and deterministic replay rule. |
| New listings and delistings | PARTIAL_NOT_PROVEN | Historical materials are available, but completeness and exact effective-date semantics are not bound. |
| Market-segment changes | NOT_PROVEN | No reviewed source contract establishes complete historical effective dates. |
| Ordinary-common-stock classification | PARTIAL_NOT_PROVEN | Current/recent listed-issues information is insufficient by itself for the full historical master. |
| Code reuse / canonical identity | NOT_PROVEN | No source contract yet proves identity disambiguation across reuse. |
| Delisted names | PARTIAL_NOT_PROVEN | Approximately 11-year history is supplied, but joinability and completeness for the master are unproven. |
| Exact effective dates | PARTIAL_NOT_PROVEN | Archive availability does not establish a daily effective-date convention. |
| Archive completeness | NOT_PROVEN | No mechanically reviewed archive inventory or gap rule exists. |
| Revision/version/hash preservation | NOT_PROVEN | No V9 archive-locking contract has been defined. |

Therefore no complete-PIT claim is warranted.

## Existing Yahoo/raw-OHLCV implementation audit

`src/v7_yahoo_collector.py` builds a Yahoo Chart request with `interval=1d`,
`events=div,splits`, and adjusted-close inclusion. Its parser retains separate
raw O/H/L/C/volume and `adj_close`, computes a SHA-256 over raw payload bytes,
canonical hashes for price rows and split events, rejects duplicate trading
dates, validates finite/nonnegative fields, records invalid rows, and parses
split-event ratios. `fetch_chart_once` is a single transport attempt.

`src/v8_historical_acquisition.py` demonstrates reusable *generic mechanics*
only: it captures first returned bytes during an acquisition attempt, verifies
payload hash/length, writes raw bytes, records payload and canonical hashes in
a manifest, rejects duplicate ticker/date rows, uses `RETRY_COUNT=0`, and
publishes a completed bundle atomically. It is V8 T1/T2-specific and does not
authorize or itself constitute a V9 source contract. `src/fetchers/yfinance.py`
is weaker legacy infrastructure: it applies adjusted-close factors to OHLC and
drops all-null rows, so it is not suitable as V9's raw/adjusted separation
without a future authorized adapter.

| Question | Classification | Repository evidence and limitation |
| --- | --- | --- |
| Raw OHLCV extraction | PROVEN | V7 parser emits separate raw OHLCV fields. |
| Adjusted-close separation | PROVEN | V7 retains `adj_close` separately; legacy fetcher overwrites OHLC and is not reusable as-is. |
| Split-event extraction | PROVEN_FOR_RETURNED_PAYLOAD | V7 parses Yahoo-returned `events.splits`; provider completeness is not established. |
| Raw payload / canonical row hashing | PROVEN | V7 computes payload/canonical hashes; V8 persists them in a manifest. |
| Duplicate/date checks | PROVEN | V7 rejects duplicate dates; V8 rejects duplicate ticker/date pairs. |
| Missing-bar behavior | PROVEN_FOR_RETURNED_ROWS | V7 records invalid returned rows; it does not prove expected-calendar completeness. |
| Retry/network behavior | PROVEN | V7 has one attempt; V8 fixes retry count to zero and fail-closes. |
| Historical revisions | NOT_PROVEN | Existing hashes identify bytes obtained, not Yahoo's revision history or revision policy. |
| Current/delisted ticker assumptions | PARTIAL | Canonical ticker syntax exists, not coverage/identity proof. |
| Code-reuse ambiguity | NOT_PROVEN | Canonical ticker normalization does not establish security-identity continuity. |
| Immutable first-complete payload cache | PARTIAL | V8 has atomic raw persistence for its own authorized blocks; no V9 locking/source contract exists. |

```text
YAHOO_2016_2025_ACTIVE_TICKER_COVERAGE=NOT_PROVEN
YAHOO_2016_2025_DELISTED_TICKER_COVERAGE=NOT_PROVEN
YAHOO_RAW_OHLCV_REPRODUCIBILITY=PARTIAL
YAHOO_SPLIT_EVENT_COMPLETENESS=NOT_PROVEN
FREE_OHLCV_FEASIBILITY=PARTIAL_NOT_PROVEN
```

Prior Yahoo use for historical tickers proves neither fresh V9 identity nor
delisted/full-window coverage.

## Required future public coverage probe — not executed

`PUBLIC_NETWORK_COVERAGE_PROBE_REQUIRED=true`. Before any execution, a fresh
human authorization is required if the then-applicable governance requires a
public-network gate. A future GPT-approved probe protocol must preregister a
public sample independent of T1 membership, before any request, and must not
print price values or returns. It must make no result-dependent sample choice,
replacement, or retry.

The fixed probe observation set is limited to HTTP success, row count and date
coverage, response schema, duplicate-date result, split-event field/presence,
and raw/canonical hashes. Its PASS rule is: every preregistered request must
return the expected schema, have HTTP success, contain no duplicate dates,
cover its preregistered permitted listing interval, preserve the requested
raw/adjusted/split fields, and produce recorded raw and canonical hashes. Any
other result is FAIL. This only tests transport/coverage feasibility; it does
not prove PIT membership, outcome quality, or profitability.

## Free-source reproducibility

`FREE_REPRODUCIBILITY=PARTIAL_NOT_PROVEN`. Existing code can support immutable
payload hashes, request provenance embedded in the V7 parsed result, canonical
normalization/hashing, and deterministic regeneration from locked V8-style raw
bytes. The unresolved V9 gap is a frozen source contract that binds a V9
request manifest, first-complete-payload lock, no-silent-refetch rule, and
revision-detection procedure across every selected identity and official JPX
archive input.

## Economic source decision

```text
FREE_FIRST_POLICY=true
FREE_SOURCE_STATUS=REQUIRES_COVERAGE_PROBE
JQUANTS_STANDARD_STATUS=FALLBACK_ONLY_NOT_AUTHORIZED
JQUANTS_STANDARD_NEEDED_NOW=false
JQUANTS_PURCHASE_AUTHORIZED=false
```

The paid source is not authorized and may become preferred only if free
coverage or reproducibility is materially inadequate after an authorized,
scientifically defensible assessment. No expected V9 profit was calculated.

## Files inspected

- `AGENTS.md`
- `PROJECT_STATE.md`
- `PROJECT_DECISION_LOG.md`
- `AI_RESEARCH_EXECUTION_RULES.md`
- `AI_REAL_EXECUTION_RUNBOOK.md`
- `V8_DATA_EXPOSURE_AUDIT.md`
- `V9_001_REUSE_DATA_AND_EXECUTION_FEASIBILITY_AUDIT.md`
- `V9_002_EXTERNAL_SOURCE_EVIDENCE.md`
- `V9_003_FULL_DESIGN_INDEPENDENT_REVIEW.md`
- `V9_003_HIGH_FULL_1_REVIEW.md`
- `V9_003_HIGH_FULL_2_REVIEW.md`
- `V9_CROSS_SECTIONAL_CLOSE_AUCTION_DESIGN_DRAFT.md`
- `src/v7_yahoo_collector.py`
- `src/v8_historical_acquisition.py`
- `src/fetchers/yfinance.py`

## Next safe action

Obtain GPT review of this audit. If continued, request a separate, fresh human
authorization for a preregistered public-network coverage probe; do not access
prices/outcomes, generate T1 membership, or change the design beforehand.

FREE_PIT_UNIVERSE_FEASIBILITY=PARTIAL_NOT_PROVEN
FREE_OHLCV_FEASIBILITY=PARTIAL_NOT_PROVEN
FREE_CORPORATE_ACTION_FEASIBILITY=PARTIAL_NOT_PROVEN
FREE_TOPIX_FEASIBILITY=FEASIBLE_FROM_EXISTING_EVIDENCE
FREE_REPRODUCIBILITY=PARTIAL_NOT_PROVEN
FREE_SOURCE_STATUS=REQUIRES_COVERAGE_PROBE
PRIMARY_BLOCKER=UNPROVEN_2016_2025_YAHOO_COVERAGE_FOR_V9_ACTIVE_AND_DELISTED_CANONICAL_IDENTITIES
PUBLIC_NETWORK_COVERAGE_PROBE_REQUIRED=true
JQUANTS_STANDARD_NEEDED_NOW=false
JQUANTS_PURCHASE_AUTHORIZED=false
DESIGN_CHANGE_AUTHORIZED=false
PRICE_OUTCOMES_OBSERVED=false
