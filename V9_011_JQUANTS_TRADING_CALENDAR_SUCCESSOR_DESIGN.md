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

The logical source object is the complete ordered finite locked page chain
generated from the one frozen base query and server-issued pagination
continuation. A single HTTP response is a `PAGE_PAYLOAD`, not the logical
source object. If pagination is absent, the logical source object is a
one-page chain.

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

Before any subscription purchase, GPT methodology authority must independently
re-check the current official public J-Quants documentation for the V2
Trading Calendar endpoint and contract, required historical coverage, and
plan availability sufficient for this coverage. This is public contract
verification only: it accesses no API key and no protected study data. If
the official coverage or API contract has changed or is insufficient,
classify `PLAN_OR_API_CONTRACT_CHANGED`, stop, and require a new ChatGPT
decision. No automatic upgrade or purchase is permitted.

Future Phase A is entirely no-network and must not perform this live plan or
coverage re-check. It relies on the preceding GPT public-contract
verification.

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

## 7. Deterministic pagination and content-lock contract

The `LOGICAL_SOURCE_OBJECT` is the complete ordered finite page chain for the
one frozen base query. Pagination is continuation of the same logical source
acquisition, not a retry, refetch, provider substitution, or source change.

The page-1 request is exactly the frozen base query with no
`pagination_key` parameter:

```text
page_1_query_from=2017-01-01
page_1_query_to=2026-01-31
page_1_query_hol_div=omitted
page_1_pagination_key=absent
```

For every page N, the existing reviewed pre-complete retry policy is applied
to that exact page request only: at most three attempts with `[5,30]`
backoff and no jitter. On the first transport-complete HTTP-200 response for
that page, the exact raw bytes are durably persisted and SHA-256 locked before
JSON inspection. Once locked, that page is immutable and can never be
refetched.

After a page is locked, acquisition may inspect only the transport envelope
needed to determine whether pagination continues. It must not inspect
`Date`, `HolDiv`, projected rows, or any calendar semantic value. Semantic
calendar processing begins only after the entire page chain has reached a
terminal page and every page is durably locked.

If a locked page has no `pagination_key` member, it is the terminal page and
the chain is complete. If the member is present, it must be a non-empty
string. Its exact value, without trimming, normalization, or substitution,
must be used as the only pagination key in the next request. The next request
must retain the exact base query and add exactly that server-issued key; no
other query parameter, endpoint, provider, plan, or source identity may
change. The key must come only from the immediately preceding locked page.

Page indices advance strictly by one. A pagination key already observed in
the chain, including a key repeated by a later page, is invalid. Page
skipping, page reordering, a manually supplied key, a substituted key, or any
page-order ambiguity is invalid. A present-but-null key, present-but-empty
key, malformed pagination metadata, or inability to prove the exact chain is
also invalid. These failures stop before any `Date`/`HolDiv` semantic calendar
processing and never authorize a refetch.

The safe page-chain provenance is deterministic and contains no raw
pagination key:

```text
schema_version=V9_011_PAGE_CHAIN_PROVENANCE_V1
base_query_identity_sha256
endpoint_identity_sha256
page_count
pages
terminal_page_index
terminal_page_reached=true
chain_lock_status=COMPLETE
semantic_processing_precondition=ALL_PAGES_LOCKED_BEFORE_DATE_HOLDIV_INSPECTION
```

`pages` is ordered by the one-based `page_index` and each page entry binds:

```text
page_index
page_request_identity_sha256
byte_count
payload_sha256
continuation_issued
continuation_key_sha256
```

For a terminal page, `continuation_issued=false` and
`continuation_key_sha256=null`. For a continuing page,
`continuation_issued=true` and `continuation_key_sha256` is the lowercase
SHA-256 of the exact server-issued pagination-key bytes. The raw key remains
machine-local durable state only. `page_request_identity_sha256` is the
canonical digest of the exact endpoint identity, frozen base-query identity,
page index, and continuation-key identity (or null for page 1); it is not a
second source-selection mechanism.

The provenance is emitted only after the terminal page is locked and all
ordered page entries are present. This ordering proves page count, per-page
byte counts, per-page payload digests, exact base-query identity, continuation
status, non-secret key identity, terminal-page reachability, and complete
chain locking before semantic processing. Parser, schema, semantic, or DQ
repair may reprocess only this complete locked page chain and may never
refetch a page.

## 8. Canonical artifact contract

All SHA-256 values below are lowercase hexadecimal. Define `UTF8(s)` as the
exact UTF-8 encoding of `s` with no BOM.

The endpoint identity digest is:

```text
ENDPOINT_IDENTITY_SHA256 = SHA256(UTF8("https://api.jquants.com/v2/markets/calendar"))
```

The base-query identity object is exactly:

```json
{"from":"2017-01-01","hol_div":null,"to":"2026-01-31"}
```

`BASE_QUERY_IDENTITY_BYTES` is the canonical JSON serialization of that
object with `ensure_ascii=false`, `sort_keys=true`,
`separators=(',', ':')`, and `allow_nan=false`, with no final LF.

```text
BASE_QUERY_IDENTITY_SHA256 = SHA256(BASE_QUERY_IDENTITY_BYTES)
```

For a server-issued continuation key:

```text
CONTINUATION_KEY_SHA256 = SHA256(UTF8(exact raw server-issued pagination_key string))
```

No trimming or normalization is applied. The exact page-request identity
object is:

```json
{
  "base_query_identity_sha256": "<64hex>",
  "continuation_key_sha256": "<64hex or null>",
  "endpoint_identity_sha256": "<64hex>",
  "page_index": 1
}
```

`PAGE_REQUEST_IDENTITY_SHA256` is the SHA-256 of its canonical JSON bytes
using the same identity procedure and no final LF. `page_index` is the
one-based page index. Page 1 uses a null continuation-key identity; later
pages use the identity of the exact key issued by the immediately preceding
locked page.

`PAGE_PAYLOAD_SHA256` remains the SHA-256 of the exact locked raw HTTP
response bytes. It is never replaced with a digest of parsed or concatenated
content.

The logical source-chain manifest object is exactly:

```json
{
  "base_query_identity_sha256": "<64hex>",
  "endpoint_identity_sha256": "<64hex>",
  "page_count": 1,
  "pages": [
    {
      "byte_count": 0,
      "continuation_issued": false,
      "continuation_key_sha256": null,
      "page_index": 1,
      "page_request_identity_sha256": "<64hex>",
      "payload_sha256": "<64hex>"
    }
  ],
  "terminal_page_index": 1
}
```

The values shown are structural examples: the actual `page_count`, byte
count, and digests are determined by the locked chain. `pages` is strictly
ordered with `page_index=1..page_count`, `terminal_page_index == page_count`,
and only the terminal page has `continuation_issued=false` and
`continuation_key_sha256=null`.

`SOURCE_CHAIN_MANIFEST_BYTES` is the canonical JSON serialization of this
object with `ensure_ascii=false`, `sort_keys=true`,
`separators=(',', ':')`, and `allow_nan=false`, with no final LF.

```text
SOURCE_CHAIN_SHA256 = SHA256(SOURCE_CHAIN_MANIFEST_BYTES)
```

The deterministic page-chain provenance artifact defined in Section 7 is
serialized as canonical UTF-8 JSON with the same key ordering and no BOM,
including exactly one final LF for its file format. Its public provenance
digest is:

```text
SOURCE_PAGE_CHAIN_PROVENANCE_SHA256 =
SHA256(exact canonical page-chain provenance artifact file bytes)
```

Raw pagination keys remain outside that artifact; only their SHA-256
identities are present. No synthetic concatenation of raw page bytes is used
as source-chain identity.

The canonical calendar artifact content object must bind at minimum:

```text
schema_version
calendar_source_family
covered_start
covered_end
trading_dates
source_chain_sha256
source_page_chain_provenance_sha256
projected_calendar_sha256
source_row_count
trading_date_count
acquisition_design_git_sha
acquisition_implementation_git_sha
api_contract_version=V2
endpoint_identity_sha256
base_query_identity_sha256
```

The content object must not contain `canonical_calendar_sha256`. Its bytes are
the complete canonical artifact serialization:

```text
encoding=UTF-8
ensure_ascii=false
sort_keys=true
separators=(',', ':')
allow_nan=false
exactly_one_final_LF=true
```

`CANONICAL_CALENDAR_SHA256` is external provenance metadata accompanying the
artifact, not a field in the hashed content:

```text
CANONICAL_ARTIFACT_BYTES = exact canonical artifact content bytes
CANONICAL_CALENDAR_SHA256 = SHA256(CANONICAL_ARTIFACT_BYTES)
```

No self-reference or fixed-point construction is permitted.

The projected calendar object is exactly:

```json
{
  "covered_end":"2026-01-31",
  "covered_start":"2017-01-01",
  "rows":[
    {"Date":"YYYY-MM-DD","HolDiv":"0|1|2|3"}
  ]
}
```

Its `rows` already satisfy the frozen validation contract and are strictly
chronologically ordered. `PROJECTED_CALENDAR_BYTES` is its canonical JSON
serialization using `ensure_ascii=false`, `sort_keys=true`,
`separators=(',', ':')`, and `allow_nan=false`, with no final LF.

```text
PROJECTED_CALENDAR_SHA256 = SHA256(PROJECTED_CALENDAR_BYTES)
```

`source_row_count` is the total number of projected `Date`/`HolDiv` rows
across the complete validated page chain before filtering to TSE trading
dates. `trading_date_count` is `len(trading_dates)`, and `trading_dates` are
the strictly ascending `Date` values whose `HolDiv` is exactly `"1"` or
`"2"`.

The public canonical artifact must contain no API key, raw response bytes,
URL containing credentials, prices, ticker identities, or unrelated J-Quants
data. Any inability to reproduce one of these exact byte domains or hashes is
`IMPLEMENTATION_FAILURE` and stops the process. No fallback serialization,
alternate ordering, or executor-inferred hash domain is permitted.

The endpoint, coverage, HolDiv semantics, `2020-10-01` sentinel, pagination
continuation, retry policy, content lock, purchase/network ordering, and
authorization semantics are unchanged by these hash-domain definitions.

## 9. Future execution phases and authorization gates

No real execution occurs under this design task. A future sequence is:

1. This design receives GPT-5.6 Sol exact-SHA PASS.
2. The implementation receives GPT-5.6 Sol exact-SHA PASS.
3. Before any purchase, GPT methodology authority re-checks the current
   official public endpoint, contract, coverage, and plan sufficiency. This
   is public contract verification only and creates no API/data authority.
   A changed or insufficient contract is `PLAN_OR_API_CONTRACT_CHANGED` and
   requires a new ChatGPT decision.
4. If a paid subscription is actually required, explicit human purchase
   approval is obtained. Purchase authority authorizes only subscription
   purchase/account provisioning; it does not authorize API acquisition,
   cache/outcome/T0 access, and is not reusable as network authorization.
5. After required account/API-key provisioning exists, Phase A performs an
   entirely no-network preflight using the real protected environment. It
   binds the exact repository provenance, reviewed design and implementation,
   required files, environment readiness, a new non-colliding durable root,
   the private credential's existence/unique resolution without reading its
   content, purchase prerequisite if applicable, and unopened T0/cache/
   outcome state. Phase A does not re-check live public documentation.
6. Phase-A output is returned to GPT, which must adjudicate Phase-A PASS
   before any network/API authorization is requested.
7. Only after Phase-A PASS is fresh point-of-use human network/API
   authorization obtained. It is scope-bound to the reviewed V9_011 Trading
   Calendar acquisition, separate from purchase authority, and not
   authorization for cache/outcomes/T0/private/sealed data.
8. Immediately before Phase B, if GPT methodology authority determines that
   the previously checked public API/plan contract may have materially
   changed, official public documentation is re-checked. This creates no
   API/data-acquisition authority; a change stops the process.
9. Phase B performs only the minimal frozen API acquisition and immediately
   locks the first complete HTTP-200 bytes.
10. Phase C performs projection and validation with no network and no refetch.

Until both GPT implementation review and the required future gates pass,
`V9_009_HIGH_2` and `V9_009_MEDIUM_1` remain open. This design creates no
purchase authority and no network authority.

## 10. Deferred research state

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
