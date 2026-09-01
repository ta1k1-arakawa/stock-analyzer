# V9_012 Actual TSE Trading-Day Authority Successor Design

## Status and study identity

```text
study_id=V9_012_ACTUAL_TSE_TRADING_DAY_AUTHORITY_SUCCESSOR
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
design_status=AWAITING_GPT_REVIEW
```

V9_012 is a new study identity and is not a retry of V9_011. V9_011 remains
terminal and immutable after its official-calendar semantic sentinel failure.
This design defines the successor procedure for producing the canonical
actual TSE cash-equity auction trading-date artifact required by V9_009.

This design authorizes no implementation, Python execution, J-Quants/API
request, private credential read, raw-payload read or reuse, materialization
rerun, T0/cache/outcome/backtest/model access, or durable production-state
access.

## Frozen coverage and authority boundary

The required inclusive coverage is exactly `2017-01-01..2026-01-31`. The
artifact is the sole future calendar authority for the V9 global signal grid,
D1/D2/D3 lookup, and monthly causal `MONTH_START` cutoffs after V9_012
design, implementation, real execution, and GPT exact-result review all
pass.

The caller may not select a calendar, provide a replacement calendar, or
derive dates from generic weekdays, `pd.bdate_range`, Yahoo observed dates,
OS locale calendars, or any other source. V9_009 may consume only the
accepted V9_012 canonical `trading_dates` artifact; until then
`V9_009_HIGH_2=OPEN_REQUIRES_HISTORICAL_JPX_CALENDAR_BINDING`.

## Independent official source roles

Both sources are frozen J-Quants V2 official API sources. They have separate
complete finite page chains, separate raw-byte locks, and separate source
chain provenance. No source is authoritative by itself for actual trading
dates.

### SOURCE_A — scheduled TSE business-day superset

```text
endpoint=https://api.jquants.com/v2/markets/calendar
role=SCHEDULED_TSE_BUSINESS_DAY_SUPERSET
base_query={"from":"2017-01-01","to":"2026-01-31"}
hol_div=omitted
scheduled_open_predicate=HolDiv in {"1","2"}
```

`scheduled_open_dates` is the set of valid `Date` values from SOURCE_A whose
`HolDiv` satisfies the frozen predicate. SOURCE_A is only a scheduled
business-day superset and must not define actual trading dates by itself.

### SOURCE_B — actual TSE activity evidence

```text
endpoint=https://api.jquants.com/v2/indices/bars/daily/topix
role=ACTUAL_TSE_MARKET_ACTIVITY_DATE_EVIDENCE
base_query={"from":"2017-01-01","to":"2026-01-31"}
```

SOURCE_B uses `Date` and only the null/non-null/type validity of `O`, `H`,
`L`, and `C`. TOPIX numeric magnitudes have zero research role. Exact raw
response bytes remain protected source evidence and are never projected into
the public artifact.

`TOPIX_ACTIVE_DATE` is exactly one row with a valid `Date` and `O`, `H`, `L`,
and `C` all present, non-null, finite real numeric values; booleans are
invalid numeric values. A row with all four OHLC values null is inactive.
Mixed null/non-null OHLC, duplicate `Date`, malformed required fields, or an
out-of-coverage `Date` is a data-quality failure.

## Exact-set adjudication

The following relation is frozen before any V9_012 acquisition:

```text
EXPECTED_EXCEPTION_SET={"2020-10-01"}

scheduled_open_dates - topix_active_dates
    == EXPECTED_EXCEPTION_SET

topix_active_dates - scheduled_open_dates
    == empty set
```

The mandatory neighboring checks are:

```text
2020-09-30 in topix_active_dates
2020-10-01 not in topix_active_dates
2020-10-02 in topix_active_dates
```

`observed_exception_dates` is the first set difference above. PASS requires
both `expected_exception_dates` and `observed_exception_dates` to be exactly
`["2020-10-01"]` in that order. Any exact-set or neighbor/sentinel failure
is terminal `ACTUAL_TRADING_DAY_AUTHORITY_FAILURE`. No missing date may be
accepted, imputed, manually corrected, added to the expected exception set,
or removed after observation. There is no manual override, post-hoc date
edit, alternate ticker/index/source substitution, or redraw-until-PASS.

Only after all exact-set and data-quality checks pass is the canonical
`trading_dates` sequence formed from `topix_active_dates`, strictly ascending.

This explicitly captures an exceptional exchange-wide closure: the official
scheduled superset may list `2020-10-01`, while the independently observed
actual TSE activity source must omit it. Weekday-minus-national-holiday
inference is never used as a substitute for this relation.

## Ordered acquisition and content locking

V9_012 requires a new non-colliding durable acquisition root and fresh,
point-of-use human network/API authorization after its own Phase-A PASS. It
must not reuse V9_011 raw acquisition state, authorization, or durable roots.
This design authorizes zero network requests.

For each source, acquisition is a complete ordered finite page chain for one
exact frozen base query. Page 1 uses exactly the source base query. Each later
page is requested only with the exact non-empty server-issued pagination key
from the immediately preceding locked page; page order, key continuity, and
terminal-page reachability are mechanically verified. Manual, substituted,
repeated, skipped, reordered, malformed, null, or empty pagination metadata
fails closed.

For each exact page request:

1. Apply `MAX_PRE_COMPLETE_ATTEMPTS=3` bounded pre-complete attempts with
   backoff `[5,30]` seconds and no jitter. Retryable and nonretryable
   transport classes are inherited from the reviewed V9_011 design without
   modification.
2. Reject redirects and bind the response to the exact endpoint/query
   request. The first complete HTTP-200 response is immediately persisted as
   exact raw bytes and locked before JSON or semantic inspection.
3. Treat every locked page as immutable. A parser, semantic, or data-quality
   failure after a complete lock never authorizes refetch. Transport failure
   remains distinct from source, parser, and data-quality failure.
4. Do not inspect `Date`, `HolDiv`, or TOPIX OHLC semantic values until the
   corresponding complete source page chain is locked and its pagination
   continuity is proven.

All page locks bind the source/page identity, safe request identity, HTTP
status, byte count, payload SHA-256, and chain position. Raw pagination keys
and raw payloads are protected durable evidence, not public artifact content.

## Deterministic source order, restart, and authorization

The real acquisition order is frozen as `SOURCE_A` first, then `SOURCE_B`.
The two source states are separate and source-identity bound. A page, lock,
request, continuation key, or terminal marker from one source can never
satisfy, resume, or be used as state for the other source.

Once an HTTP-200 page has its exact raw bytes and an immutable valid lock, that
raw/lock pair is authoritative durable state. A locked page is never fetched
again. A completed terminal source chain is immutable across process restart.

If a process, wrapper, timeout, or transport operation is interrupted before
both source chains are complete, all durable state is preserved. It is never
deleted, reset, or overwritten merely to restart. Restart validates the
locked raw/lock pairs and reconstructs pagination only from those validated
pairs, using transport-envelope and lock metadata as in the V9_011
crash-safe restart procedure. Restart does not inspect `Date`, `HolDiv`, or
`O`/`H`/`L`/`C` values merely to decide where to resume.

Continuation is allowed only at the first missing page of the first
incomplete source. If SOURCE_A is partial, only its first missing page may be
requested; SOURCE_B cannot start. If SOURCE_A is terminally complete,
SOURCE_A is never requested again and continuation may proceed only with
SOURCE_B. If both chains are already complete, restart performs zero network
requests and proceeds directly to offline validation.

Every invocation that can issue a new HTTP request requires the ordered
authority sequence `Phase A PASS -> GPT adjudication -> fresh point-of-use
human network/API authorization`. Authorization is one-shot for that
invocation and is consumed when the private credential/network boundary is
crossed, even if the invocation produces zero lockable payloads. Consumed
authorization is never reused after a crash, wrapper failure, transport
failure, timeout, nonzero exit, or partial acquisition. A resumed invocation
that can issue any new request requires fresh authorization. An offline
restart over complete locked state has no network path and requires no
network authorization.

The maximum remains `MAX_PRE_COMPLETE_ATTEMPTS=3` for one exact page request
within one authorized invocation, with backoff `[5,30]` seconds and no
jitter. A page is never retried after its HTTP-200 payload is locked. A
nonretryable transport result is immediately terminal for that invocation;
exhaustion of the three retryable attempts stops that invocation. No
automatic second invocation or cross-invocation retry loop is allowed. A
later continuation invocation is possible only after GPT reviews safe
preserved-state evidence and a human supplies fresh point-of-use
authorization, and it resumes only the first missing page without refetching
any locked page.

After both chains are complete, all `Date`/`HolDiv`/TOPIX OHLC semantic,
parser, schema, data-quality, exact-set, and sentinel processing is offline
only. Any such failure is terminal `ACTUAL_TRADING_DAY_AUTHORITY_FAILURE`
for V9_012 and authorizes no refetch, new network attempt, source
substitution, or durable-state reset.

Safe restart evidence may expose only source role/id, page counts, lock
counts, payload SHA-256 and byte counts, terminal booleans, first missing
page index, request count, process exit code, authorization-consumed
boolean, and safe failure class. It must never expose raw payloads, raw
pagination keys, API keys, TOPIX values, or private paths.

## Offline validation and canonical artifact

After both complete source chains are locked, all parsing and adjudication is
offline and uses exactly those locked bytes. No network or refetch is
available during parser development or execution. A source/parser/data-
quality failure is terminal; it cannot trigger a new acquisition attempt.

The accepted public canonical artifact contains exactly these frozen fields:

```text
schema_version
covered_start
covered_end
trading_dates
scheduled_calendar_source_chain_sha256
topix_source_chain_sha256
scheduled_open_count
actual_trading_date_count
expected_exception_dates
observed_exception_dates
scheduled_calendar_source_api_identity
topix_source_api_identity
scheduled_calendar_base_query_sha256
topix_base_query_sha256
acquisition_design_git_sha
acquisition_implementation_git_sha
```

The acquisition design and implementation Git SHA fields must be exact
lowercase 40-hex repository SHAs from the independently reviewed code/design
actually used. Every SHA-256 field is exact lowercase 64-hex. The
source/API identity fields equal the exact API identity strings defined below;
base-query identities bind the exact query objects below, and source-chain
digests bind the complete locked page chains.

Public canonical JSON is deterministic: UTF-8, `ensure_ascii=false`,
`sort_keys=true`, `separators=(',',':')`, `allow_nan=false`, and exactly one
final LF. The canonical artifact SHA is external metadata over exactly those
final-LF artifact bytes and is not embedded in the artifact itself.

The artifact contains no TOPIX numeric values, ticker identities, API key,
raw responses, pagination keys, prices, private paths, or unrelated source
content. It contains no caller-selectable calendar or provenance.

Before accepting the artifact, the implementation must verify both complete
source chains, exact page/request/lock/payload bindings, exact coverage,
field validity, duplicate and range rules, the exact set relation, the three
neighbor checks, sorted unique `trading_dates`, exact artifact schema, and
all digest/provenance fields. Any mismatch fails before V9_009 can read or
consume the artifact.

## Exact provenance and hash-domain closure

The following definitions are frozen before V9_012 implementation. The
primitive `CANONICAL_JSON_NO_LF(value)` is UTF-8 JSON serialized with
`ensure_ascii=false`, `sort_keys=true`, `separators=(',',':')`,
`allow_nan=false`, no BOM, and no final LF. Every SHA-256 value is exact
lowercase 64-hex.

The exact API identity strings are:

```text
SCHEDULED_CALENDAR_SOURCE_API_IDENTITY=https://api.jquants.com/v2/markets/calendar
TOPIX_SOURCE_API_IDENTITY=https://api.jquants.com/v2/indices/bars/daily/topix
```

The public artifact fields with those names must equal the corresponding
strings exactly. The exact base-query object for both sources is:

```json
{"from":"2017-01-01","to":"2026-01-31"}
```

`hol_div` is intentionally absent, not null, for SOURCE_A. SOURCE_B uses the
same exact object. For each source:

```text
BASE_QUERY_SHA256=SHA256(CANONICAL_JSON_NO_LF(BASE_QUERY_OBJECT))
```

The two base-query SHA-256 values are therefore equal; source/API identity and
source role bind the otherwise equal query objects independently.

For an exact server-issued pagination-key string:

```text
PAGINATION_KEY_SHA256=SHA256(UTF8(exact string))
```

There is no trimming, normalization, case conversion, decoding substitution,
or reserialization. The exact page-request identity object is:

```json
{
  "base_query_sha256":"<64hex>",
  "continuation_key_sha256":"<64hex or null>",
  "page_index":<positive integer>,
  "source_api_identity":"<exact API identity string>",
  "source_role":"<exact frozen source role>"
}
```

SOURCE_A uses `SCHEDULED_TSE_BUSINESS_DAY_SUPERSET`; SOURCE_B uses
`ACTUAL_TSE_MARKET_ACTIVITY_DATE_EVIDENCE`. Page 1 has
`continuation_key_sha256=null`; every later request uses the exact hash of the
server-issued key from the immediately preceding page. The request identity
digest is:

```text
PAGE_REQUEST_IDENTITY_SHA256=
  SHA256(CANONICAL_JSON_NO_LF(page-request identity object))
```

For each source, the exact source-chain manifest object is:

```json
{
  "base_query_sha256":"<64hex>",
  "page_count":<positive integer>,
  "pages":[
    {
      "byte_count":<nonnegative integer>,
      "continuation_issued":<boolean>,
      "continuation_key_sha256":"<64hex or null>",
      "page_index":<positive integer>,
      "page_request_identity_sha256":"<64hex>",
      "payload_sha256":"<64hex>"
    }
  ],
  "source_api_identity":"<exact API identity string>",
  "source_role":"<exact frozen source role>",
  "terminal_page_index":<positive integer>
}
```

Manifest requirements are exact: page indices are `1..page_count` in strict
order; `terminal_page_index == page_count`; only the terminal page has
`continuation_issued=false` and `continuation_key_sha256=null`; every
nonterminal page has `continuation_issued=true` and a 64-hex continuation-key
identity; `payload_sha256` hashes exact locked raw HTTP response bytes; and
`byte_count` equals exact raw payload byte length. No raw pagination key
appears in the manifest. The manifest and every page are source-identity
bound.

```text
SOURCE_CHAIN_MANIFEST_BYTES=CANONICAL_JSON_NO_LF(manifest)
SOURCE_CHAIN_SHA256=SHA256(SOURCE_CHAIN_MANIFEST_BYTES)
```

The canonical fields `scheduled_calendar_source_chain_sha256` and
`topix_source_chain_sha256` must equal the corresponding source-specific
`SOURCE_CHAIN_SHA256` values exactly.

The accepted canonical artifact field set remains exact, with no missing or
extra fields:

```text
schema_version
covered_start
covered_end
trading_dates
scheduled_calendar_source_chain_sha256
topix_source_chain_sha256
scheduled_open_count
actual_trading_date_count
expected_exception_dates
observed_exception_dates
scheduled_calendar_source_api_identity
topix_source_api_identity
scheduled_calendar_base_query_sha256
topix_base_query_sha256
acquisition_design_git_sha
acquisition_implementation_git_sha
```

Its exact value rules are:

```text
schema_version=V9_012_CANONICAL_ACTUAL_TSE_TRADING_DAYS_V1
covered_start="2017-01-01"
covered_end="2026-01-31"
expected_exception_dates=["2020-10-01"]
observed_exception_dates=["2020-10-01"] for PASS
scheduled_open_count=len(sorted unique scheduled_open_dates)
actual_trading_date_count=len(trading_dates)
```

`trading_dates` is the strictly ascending unique TOPIX_ACTIVE_DATE sequence,
formed only after all exact-set, data-quality, and neighbor/sentinel checks
pass. `acquisition_design_git_sha` and `acquisition_implementation_git_sha`
are exact lowercase 40-hex SHAs for the independently GPT-reviewed V9_012
design and implementation actually used.

The exact artifact byte and digest domain is:

```text
CANONICAL_ARTIFACT_CONTENT=the exact field-set object above
CANONICAL_ARTIFACT_BYTES=
  UTF-8 JSON with ensure_ascii=false,
  sort_keys=true,
  separators=(',',':'),
  allow_nan=false,
  no BOM,
  and exactly one final LF
CANONICAL_ARTIFACT_SHA256=SHA256(CANONICAL_ARTIFACT_BYTES)
```

The artifact must not contain its own SHA. The external receipt object has
exactly these fields:

```text
schema_version
status
canonical_artifact_sha256
```

with exact values:

```text
schema_version=V9_012_CANONICAL_HASH_RECEIPT_V1
status=COMPLETE
canonical_artifact_sha256=exact lowercase SHA256 of CANONICAL_ARTIFACT_BYTES
```

Receipt bytes use the same deterministic public JSON serialization with
exactly one final LF; the receipt has no self-reference.

The public artifact, receipt, and project state contain no raw TOPIX numeric
`O`/`H`/`L`/`C` values, raw responses, raw pagination keys, API key, private
paths, prices, ticker identities, or unrelated source data.

Before acceptance, implementation must reject any alternative hash domain or
serializer, missing or extra artifact/receipt/manifest field, a SOURCE_A
query containing `hol_div:null`, reordered or incomplete page chain,
source-identity mismatch, or any noncanonical public artifact. It must verify
both complete source chains, exact page/request/lock/payload bindings, exact
coverage, field validity, duplicate and range rules, the exact set relation,
the three neighbor checks, sorted unique `trading_dates`, and every digest
and provenance field before V9_009 can read or consume the artifact.

## V9_009 binding gate

The later V9_009 HIGH_2 remediation must remove arbitrary real-execution
calendar authority and mechanically bind the exact accepted V9_012 artifact
schema, canonical SHA, and coverage before any V9_009 cache or outcome read.
It must derive the signal grid, D1/D3 lookup, and `MONTH_START` solely from
that artifact, and emit `exact_calendar_grid=true` only after exact binding
succeeds. Binding failure is `NO_VERDICT` / input-binding failure before any
research verdict.

V9_012 design, implementation, real execution, and exact result review are
all prerequisites. No F1/T/F2/Phase2 action, J-Quants purchase, T0,
cache/outcome access, or profitability claim is created by this design.

## Non-actions and execution boundary

This design does not acquire official pages, inspect V9_011 raw payloads,
read private credentials, or establish V9_012 success. The exact V9_012
network execution requires a fresh Phase-A PASS and fresh point-of-use human
authorization. The provider, endpoints, coverage, pagination, retry policy,
content-lock rules, exact-set relation, sentinel, and artifact schema above
are frozen; no implementation-time source substitution or semantic weakening
is permitted.

The next implementation must use the two independent source roles exactly as
specified. If the frozen official API responses cannot support unambiguous
actual TSE cash-equity date semantics or the exact relation, it must fail
`ACTUAL_TRADING_DAY_AUTHORITY_FAILURE`; it must not refetch until pass or
invent a replacement methodology.
