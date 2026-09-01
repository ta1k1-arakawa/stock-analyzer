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
actually used. Every SHA-256 field is exact lowercase 64-hex. Source/API
identity values are frozen stable role identifiers; base-query identities bind
the exact query objects above. Source-chain digests bind the complete locked
page chains.

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
