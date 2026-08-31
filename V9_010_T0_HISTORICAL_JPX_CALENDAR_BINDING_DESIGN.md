# V9_010 Historical JPX Calendar Binding Design

```text
document_role=V9_IMPLEMENTATION_DESIGN
task=V9_010_T0_HISTORICAL_JPX_CALENDAR_BINDING_DESIGN
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
status=DESIGN_AWAITING_GPT_EXACT_SHA_REVIEW
T0_STATUS=NOT_RUN
network_requests=0
real_cache_reads=0
real_outcome_calculations=0
```

## 1. Purpose and authority

This design specifies the smallest future official-source procedure for one
canonical historical JPX cash-equity trading-date artifact used by V9_009.
It does not authorize acquisition, cache reading, T0 execution, or any
research verdict. The artifact is an input-binding object, not evidence of
profitability.

The calendar is authoritative for all of the following V9 operations:

- the global three-JPX-trading-day signal grid and its anchor;
- D1, D2, and D3 lookup from each D0; and
- the first JPX trading day of each month used as the monthly causal training
  cutoff.

During real V9_009 execution, the calendar may not be selected by a caller.
`pd.bdate_range`, generic weekday logic, Yahoo observed dates, an operating
system locale calendar, and an arbitrary caller-supplied calendar file are not
authoritative inputs.

## 2. Frozen coverage contract

The canonical artifact must contain every required JPX cash-equity trading day
from `2017-01-01` through `2026-01-31`, inclusive of the date bounds. The
coverage contract is therefore:

```text
CALENDAR_COVERAGE_START=2017-01-01
CALENDAR_COVERAGE_END=2026-01-31
REQUIRED_SOURCE_MONTHS=2017-01..2026-01 inclusive
REQUIRED_SOURCE_MONTH_COUNT=109
```

This covers backward V9 signal-grid operation in 2017, T0 evaluation signals
in 2020..2025, and D1/D3 realization for final 2025 signals. A larger tail is
not permitted by implementation convenience. If a later implementation
proves that a larger bounded tail is mechanically required, it must stop with
`CHATGPT_DECISION_REQUIRED`; it may not silently extend coverage.

## 3. Official source family and finite source contract

Only public official Japan Exchange Group / Tokyo Stock Exchange sources may
define historical cash-equity calendar semantics. The source family is:

```text
CALENDAR_SOURCE_FAMILY=OFFICIAL_JPX_TSE_CASH_EQUITY_CALENDAR
CALENDAR_SOURCE_HOST_ALLOWLIST=www.jpx.co.jp
CALENDAR_SOURCE_SCHEME=https
```

V9_005 established source-family feasibility and verified that historical
monthly JPX calendar pages exist, including an official historical page. It
did not establish complete archive coverage for every required month. This
design therefore does not claim archive completeness.

The future acquisition implementation must bind one finite, predeclared
source manifest with exactly one official source object for each of the 109
required month slots. The manifest must have the closed form:

```text
source_slot=YYYY-MM for every YYYY-MM in 2017-01..2026-01
source_object_count=109
source_object_per_slot=1
fallback_source_objects=0
```

Before any network execution, Stage A of the future implementation must freeze
either the exact 109-entry HTTPS JPX URL manifest or an exact deterministic
manifest-generation rule. Its expansion must be mechanically checked to be
exactly 109 unique HTTPS `www.jpx.co.jp` URLs, one for each required month
slot. Stage A freezes endpoint identity only; it does not require DOM
selectors, holiday-table selectors, payload-specific semantic anchors, or
parser rules before raw acquisition. It must reject a missing, duplicate, or
ambiguous slot rather than discover a neighboring page, substitute a
different source, or infer archive coverage. This is a future implementation
requirement, not permission to acquire pages in V9_010.

No Yahoo, broker, price, outcome, or non-official holiday source may supply or
repair the calendar.

## 4. Three ordered future stages

Future acquisition is classified as:

```text
operation_class=RETRIABLE_PUBLIC_PLUMBING
```

The future workflow is strictly ordered. A later stage may not run until the
earlier stage has succeeded.

### Stage A: raw acquisition

Before the first network request, freeze the exact finite 109-slot endpoint
manifest or the exact deterministic generation rule described in section 3.
Compute and persist `source_manifest_sha256` over the exact ordered manifest:

```text
canonical_json_bytes([
  {"source_slot": "YYYY-MM", "source_url": "https://...",
   "source_url_sha256": "..."},
  ... exactly 109 entries in source_slot order ...
])
```

`source_url_sha256` is SHA-256 of the exact UTF-8 URL string. The endpoint
manifest is an internal acquisition input; raw URLs are not emitted in the
public canonical calendar artifact. The manifest digest is bound before any
request and is never changed in response to payload content.

The acquisition retry/stopping contract is fixed as follows:

```text
operation_class=RETRIABLE_PUBLIC_PLUMBING
MAX_PRE_COMPLETE_ATTEMPTS=3
STOP_AFTER_FIRST_COMPLETE_HTTP_200=true
RETRY_AFTER_COMPLETE_PAYLOAD=false
RETRY_AFTER_PARSER_OR_SEMANTIC_FAILURE=false
```

For each manifest object, a transport attempt may continue only until one
complete HTTP-200 payload is received or the bounded attempt count is
exhausted. Immediately upon receiving that first complete HTTP-200 payload,
lock the exact bytes before any semantic inspection. Create one durable
raw-lock record containing at minimum:

```text
source_slot
source_url_sha256
http_status
byte_count
payload_sha256
```

The record must use the exact payload byte count and SHA-256. A non-200,
partial, or interrupted response is not a locked payload and cannot enter the
raw-lock-set digest. Once a complete payload is locked, the slot is closed to
network access permanently, including after parser, semantic, DQ, or
canonicalization failure.

After all 109 payloads are locked, compute:

```text
raw_lock_set_sha256 = SHA256(canonical_json_bytes(
  [raw_lock_record_1, ..., raw_lock_record_109]
))
```

The records are ordered by `source_slot`; all 109 slots must occur exactly
once. Stage A stops on any manifest, endpoint, transport, lock, byte-count,
or payload-digest failure. It does not parse holiday semantics.

### Stage B: offline structure and parser freeze

Only after Stage A has successfully locked all 109 payloads may the locked
bytes be inspected. Structure discovery, selector identification, semantic
anchors, and parser rules must use exactly those locked bytes and no network.
The parser must be frozen only after this offline inspection. Parser design
and parser implementation each require a separate GPT exact-SHA review.

No parser development or parser execution may refetch a source object. If the
locked payloads cannot support unambiguous official cash-equity closure
semantics, the result is `SOURCE_OR_DATA_FEASIBILITY`; it is not permission to
refetch until a parser passes and it is not a V9 STOP/CONTINUE verdict.

### Stage C: canonical calendar generation

Parse only the exact 109-payload locked set using the separately reviewed
Stage-B parser. Generate and validate the canonical calendar artifact from
those bytes and the locked provenance records. Stage C performs no network
request and cannot add, replace, or reorder source objects.

Transport failure, source-identity failure, parser failure, and calendar
semantic/data-quality failure remain distinct failure classes. A failed or
partial response never becomes a semantic calendar input, and no partial bytes
may enter the payload-set or raw-lock-set digest.

V9_010 itself performs zero network requests. Any future network execution
requires a separately reviewed implementation and fresh human authorization.

## 5. Canonical artifact

After all 109 source objects pass, the future parser must emit one deterministic
canonical artifact containing only these fields:

```text
schema_version
calendar_source_family
covered_start
covered_end
trading_dates
source_payload_set_sha256
source_manifest_sha256
raw_lock_set_sha256
canonical_calendar_sha256
source_object_count
source_month_count
source_payload_total_byte_count
acquisition_design_git_sha
acquisition_implementation_git_sha
parser_design_git_sha
parser_implementation_git_sha
```

All four Git provenance fields must be exact lowercase 40-hex Git SHAs. The
acquisition implementation SHA is the exact independently reviewed
repository SHA used for Stage-A raw acquisition. The parser implementation
SHA is the exact independently reviewed repository SHA used on the locked
payloads. There is no caller-selectable provenance.

`trading_dates` is the sorted exact JPX cash-equity trading-date list in
`YYYY-MM-DD` form. The public artifact must not contain page HTML, source
URLs, private paths, ticker identities, or unrelated JPX event content.
Source hashes and safe counts are provenance, not outcome data.

The canonical byte procedure is frozen as UTF-8 JSON with `ensure_ascii=False`,
`sort_keys=True`, separators `(',', ':')`, `allow_nan=False`, and one final LF.
The canonical calendar digest is SHA-256 of those exact canonical bytes after
excluding the digest field itself. The payload-set digest input is a canonical
JSON array ordered by `source_slot`, with each entry containing only
`source_slot`, `payload_sha256`, and `byte_count`; its SHA-256 is recorded as
`source_payload_set_sha256`. The exact artifact field ordering is supplied by
the canonical JSON procedure, not source-page order. The payload-set digest
must be derived from the same exact ordered locked payload records and cannot
be recomputed from a later refetch.

The artifact digest is computed only after all validation succeeds. No
partially valid artifact is authoritative.

## 6. Stage-B parsing and calendar semantics

After Stage A lock success, the parser must read only the official source
representation that is
mechanically identified as the JPX/TSE cash-equity market-holiday/calendar
section for its manifest month. It must distinguish that section from
derivatives, futures, options, commodities, listings, corporate events,
announcements, and unrelated page text. If that distinction is not unique,
the source object fails closed.

For each source slot, the parser must establish the slot year and month from
the official page structure or an exact equivalent semantic anchor. A date
without unambiguous year/month context is not assigned by guessing. A source
object that cannot establish complete month semantics is invalid even if some
dates appear plausible.

The canonical trading-date set is formed only after the official cash-equity
closure set has been parsed for every required slot:

```text
trading_dates = weekdays within the fixed coverage bounds
                minus every explicitly identified official cash-equity closure
```

An exceptional exchange-wide closure, if represented by the official
historical source, is a cash-equity closure row and is included in the closure
set exactly like an ordinary market holiday. The parser must not assume that
weekday-minus-national-holidays is sufficient. If the official source uses a
special-closure, exchange-closure, or equivalent label, it must be recognized
as closure semantics; if the label or scope is ambiguous, the parser fails
closed. No exceptional closure may be inferred from a neighboring month,
weekday pattern, price absence, or another source.

## 7. Required fail-closed validation

Before canonical calendar generation, the future implementation must
mechanically verify all acquisition and parser provenance bindings:

- the manifest digest equals the frozen expected `source_manifest_sha256`;
- all 109 source slots occur exactly once;
- every raw-lock endpoint identity matches its manifest slot;
- `raw_lock_set_sha256` equals the digest of the exact ordered records;
- every locked payload’s bytes match its recorded SHA-256 and byte count;
- acquisition design/implementation SHAs are exact reviewed lowercase 40-hex
  values;
- parser design/implementation SHAs are exact reviewed lowercase 40-hex
  values; and
- the parser consumed exactly the locked payload set and performed no network
  request.

Any mismatch fails before calendar acceptance and cannot be converted into a
V9 research verdict.

The future parser/artifact validator must reject all of the following:

- a missing required source object or month;
- duplicate source month/slot or more than one accepted object for a slot;
- wrong host, non-HTTPS source identity, or unrelated official content;
- malformed or ambiguous market-holiday/cash-equity semantics;
- a date with invalid syntax, invalid calendar value, or ambiguous context;
- duplicate closure date or duplicate trading date;
- truncated or incomplete source content;
- a date outside `2017-01-01..2026-01-31`;
- an unsorted source-derived or canonical trading-date list;
- any weekend in `trading_dates`;
- incomplete boundary coverage or an artifact outside the fixed bounds;
- inability to distinguish cash-equity closures from listing, event,
  derivatives, or other unrelated JPX information; and
- any digest, count, schema, provenance, or canonical-byte mismatch.

The validator must not infer a missing month from neighboring months and must
not convert any of these failures into a V9 STOP or CONTINUE research
verdict. They are source/parser/input-binding failures.

## 8. Binding into V9_009 after artifact PASS

The later V9_009 HIGH_2 remediation must make the canonical artifact the sole
production calendar authority. Before any FIXED_V4_300 cache read or outcome
calculation, it must:

1. load and validate the exact artifact schema and canonical digest;
2. verify `covered_start`, `covered_end`, source-set digest, parser/design/
   implementation provenance, sorted dates, weekday exclusion, and exact
   required coverage;
3. bind the exact reviewed canonical-calendar SHA and schema version;
4. derive the V9 three-day grid, D1/D2/D3 lookup, and `MONTH_START` only from
   the artifact’s `trading_dates`; and
5. emit `exact_calendar_grid=true` only after all exact binding checks pass.

Any binding, coverage, digest, or provenance failure must stop before cache or
outcome access and map to the existing non-research input-binding/fail-closed
state, such as `NO_VERDICT_DATA_INCOMPATIBLE`. It must never silently accept a
caller calendar, generic business-day calendar, or substitute source.

Synthetic tests may continue to pass explicit synthetic calendars to pure
internal helpers. That test convenience does not authorize an arbitrary
calendar during real T0 execution.

## 9. Non-actions and provenance

This design was created without inspecting new external source payloads and
without observing a new V9 T0 outcome. It makes no profitability claim and
does not freeze the whole V9 study.

```text
T0_STATUS=NOT_RUN
network_requests=0
real_cache_reads=0
real_outcome_calculations=0
real_terminal_reads=0
human_gates_consumed=0
JQUANTS_PURCHASE_AUTHORIZED=false
V9_design_frozen=false
future_profitability_established=false
```

The future exact URL manifest, payload selectors, and finite pre-complete
retry bound remain implementation-time bindings that must be frozen and
independently reviewed before any network request. No implementation,
acquisition, calendar artifact, V9_009 modification, F1/T/F2 action, or T0
execution is authorized by V9_010.

## 10. Required next action

```text
NEXT_ACTION=GPT_EXACT_SHA_DESIGN_REVIEW
```
