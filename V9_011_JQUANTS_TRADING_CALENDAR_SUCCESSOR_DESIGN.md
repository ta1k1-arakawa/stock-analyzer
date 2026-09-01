# V9_011 J-Quants Trading Calendar Successor Design

```text
document_type=V9_SUCCESSOR_ACQUISITION_DESIGN
study=V9_011_JQUANTS_TRADING_CALENDAR_SUCCESSOR
status=AWAITING_GPT_EXACT_SHA_REVIEW
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
```

## 1. Purpose and study boundary

V9_011 is a new successor acquisition design created after the terminal
V9_010 Stage-A HTTP-404 result. Its sole purpose is to produce the canonical
TSE cash-equity trading-date artifact required to remediate
`V9_009_HIGH_2`. It is not a retry or continuation of V9_010 and is not
`ATTEMPT_3`.

The artifact is input binding only. It has zero profitability evidential
capacity and does not authorize cache reads, outcome calculations, T0,
model fitting, backtests, or any strategy conclusion.

V9_010 remains immutable and terminal:

- `V9_010_STAGE_A_REAL_ACQUISITION=FAIL`
- `ATTEMPT_2_RUNNER_SAFE_REASON=HTTP_404`
- `ATTEMPT_2_REEXECUTION_AUTHORIZED=false`
- no V9_010 `ATTEMPT_3` is authorized

## 2. Frozen coverage and authoritative source

The required inclusive coverage is `2017-01-01` through `2026-01-31`.

The authoritative source is the official JPX J-Quants API V2 Trading
Calendar:

```text
endpoint_method=GET
endpoint_identity=https://api.jquants.com/v2/markets/calendar
logical_query_from=2017-01-01
logical_query_to=2026-01-31
logical_query_hol_div=omitted
api_contract_version=V2
```

The one logical source object is the exact requested calendar response.
Authentication uses the private `x-api-key` header. The API key is never
committed, printed, hashed into a public report, placed in a public artifact,
or exposed through command-line output. A future real execution may obtain it
only at point of use under `AI_REAL_EXECUTION_RUNBOOK.md`.

The current official access contract supplied for this design is as of
`2026-09-01`: Trading Calendar storage begins `2008-01-01`, and Standard
provides ten years of historical data plus data through the end of the next
year. Therefore:

```text
MINIMUM_EXPECTED_PLAN=STANDARD
PURCHASE_AUTHORIZED=false
NETWORK_AUTHORIZED=false
```

A future Phase A must re-check the current official plan and coverage before
any purchase or network boundary. If the official coverage or API contract
has changed, classify `PLAN_OR_API_CONTRACT_CHANGED`, stop, and require a new
ChatGPT decision. No automatic upgrade or purchase is permitted.

F6 TOPIX annual `DATE` fields are not a daily-calendar substitute and cannot
replace this authoritative Trading Calendar source.

## 3. Response projection and calendar semantics

The semantic projection uses only the response fields `Date` and `HolDiv`.
Any additional API fields are preserved only as locked source bytes and do
not become calendar semantics.

`HolDiv` is interpreted exactly as follows:

| `HolDiv` | TSE cash-equity meaning |
| --- | --- |
| `"0"` | TSE non-trading date |
| `"1"` | TSE trading date |
| `"2"` | TSE half-day trading date |
| `"3"` | TSE non-trading date; OSE holiday trading exists |

The canonical TSE trading-date predicate is exactly `HolDiv in {"1","2"}`.
`HolDiv in {"0","3"}` is not a TSE cash-equity trading date.

## 4. Mandatory semantic sentinel

The official TSE cash-equity trading-system failure on `2020-10-01` is a
mandatory semantic sentinel. All listed symbols were halted for the entire
day, so the J-Quants response must classify `2020-10-01` as non-TSE-trading:
its `HolDiv` must not be `"1"` or `"2"`.

If the sentinel is classified as trading, the result is
`CALENDAR_SEMANTIC_SENTINEL_FAILURE`: stop immediately, do not manually
override it, do not remove the date post hoc, and do not access T0, caches, or
outcomes. The sentinel is not assumed to pass before execution.

## 5. Validation contract

After the exact source payload has been locked, project and validate only the
following fields and rules:

- `Date` is exactly `YYYY-MM-DD`.
- `HolDiv` is a string exactly one of `"0"`, `"1"`, `"2"`, or `"3"`.
- Every calendar date in the inclusive coverage occurs exactly once.
- Duplicate dates are invalid.
- Dates outside the inclusive coverage are invalid.
- The projected rows are deterministically ordered chronologically.
- No calendar date is missing.
- No caller-supplied calendar is accepted.
- Yahoo does not provide date authority.
- Generic business-day or calendar libraries do not provide date authority.
- No date is silently imputed.
- The `2020-10-01` semantic sentinel passes as non-TSE-trading.

Any validation, schema, semantic, or data-quality failure after content lock
is terminal for this logical source object. It never authorizes a refetch or
a source substitution.

## 6. Transport and content-lock boundary

The existing reviewed shared transport policy is inherited without
modification for the pre-complete phase:

```text
maximum_attempts=3
backoff_seconds=[5,30]
jitter=false
```

Only these conditions are retryable before the first complete payload:

- timeout
- connection reset
- temporary DNS failure
- HTTP `408`, `425`, `429`, `500`, `502`, `503`, or `504`

HTTP `400`, `401`, `403`, `404`, and every other nonretryable response stop
immediately. `401` and `403` are classified separately as
`AUTH_OR_PLAN_FAILURE`. The endpoint, query, plan, provider, and source
identity cannot change after observing a result.

The first transport-complete exact-request HTTP-200 payload, including an
empty payload, must be durably preserved and SHA-256 locked before any
projection, schema, semantic, or data-quality inspection. After that lock,
parser/schema/semantic/data-quality failure cannot cause a refetch.

## 7. Canonical artifact contract

The canonical artifact must bind at minimum:

```text
schema_version
calendar_source_family
covered_start
covered_end
trading_dates
source_payload_sha256
projected_calendar_sha256
canonical_calendar_sha256
source_row_count
trading_date_count
acquisition_design_git_sha
acquisition_implementation_git_sha
api_contract_version=V2
endpoint_identity_sha256
```

The canonical JSON byte procedure is:

```text
encoding=UTF-8
ensure_ascii=false
sort_keys=true
separators=(',', ':')
allow_nan=false
final_byte=LF
```

The public canonical artifact must contain no API key, raw response bytes,
URL containing credentials, prices, ticker identities, or unrelated J-Quants
data. The endpoint identity digest binds the exact endpoint identity without
creating credential material. Implementation and design Git SHAs must be
lowercase exact 40-hex provenance values bound at the reviewed execution
boundary.

## 8. Future execution phases and authorization gates

No real execution occurs under this design task. A future sequence is:

1. This design receives GPT-5.6 Sol exact-SHA PASS.
2. The implementation receives GPT-5.6 Sol exact-SHA PASS.
3. Any required Standard subscription or purchase receives explicit human
   approval. Purchase authority and network authority are separate and
   non-transferable.
4. Fresh point-of-use human network/API authorization is obtained.
5. Phase A no-network preflight passes using the real protected environment.
6. Phase B performs only the minimal frozen API acquisition and immediately
   locks the first complete HTTP-200 bytes.
7. Phase C performs projection and validation with no network and no refetch.

Until both GPT implementation review and the required future gates pass,
`V9_009_HIGH_2` and `V9_009_MEDIUM_1` remain open. This design creates no
purchase authority and no network authority.

## 9. Deferred research state

```text
V9_011_JQUANTS_TRADING_CALENDAR_SUCCESSOR_DESIGN=AWAITING_GPT_REVIEW
MINIMUM_EXPECTED_PLAN=STANDARD
PURCHASE_AUTHORIZED=false
NETWORK_AUTHORIZED=false
T0_STATUS=NOT_RUN
real_cache_reads=0
real_outcome_calculations=0
future_profitability_established=false
```

