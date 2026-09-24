# V13 V8 J-Quants Real-Execution and Private Content-Lock Design

```text
document_type=V8_JQUANTS_REAL_EXECUTION_CONTENT_LOCK_DESIGN
status=FROZEN_AWAITING_GPT_EXACT_SHA_REVIEW
issue=66
design_owner=GPT-5.6_SOL
semantic_design=V13_V8_JQUANTS_IDENTITY_RECOVERY_DESIGN.md
semantic_design_reviewed_sha=4e6909acd35c52e1fa1188f23d8e5b3f4cc7273f
real_jquants_execution=false
jquants_api_requests=0
```

## 1. Scope and authority

This freezes execution plumbing for the already accepted J-Quants Standard
semantic and block-identity route. Issue #65's exact-SHA independent review
of `4e6909acd35c52e1fa1188f23d8e5b3f4cc7273f` returned PASS with zero
critical, high, medium, and low findings. This design performs no source
request, credential read, private-content read, fit, backtest, or trade. It
does not authorize real execution. Implementation, independent exact-SHA
review, applicable environment readiness, and a separate point-of-use human
authorization must precede the direct-Windows acquisition.

The frozen source and methodology are unchanged:

```text
RECOVERY_STRATEGY=JQUANTS_STANDARD_SEMANTIC_AND_BLOCK_IDENTITY
JQUANTS_PLAN=STANDARD
JQUANTS_API_VERSION=V2
JQUANTS_ENDPOINT=https://api.jquants.com/v2/equities/master
JQUANTS_QUERY_DATE=20260731
JQUANTS_AUTH_HEADER=x-api-key
MARKET_CODES_ACCEPTED={0111,0112}
PRODUCT_CATEGORY_ACCEPTED={011}
ELIGIBLE_CANONICAL_ORDER=sort by (SHA-256(UTF-8 code), code) ascending
ELIGIBLE_COUNT_REQUIRED=3110
ELIGIBLE_AND_ALL_BLOCK_HASH_PINS=UNCHANGED_FROM_SEMANTIC_DESIGN
ADAPTIVE_FILTERING=false
ALTERNATE_DATE_QUERY=false
REPLACEMENT_PARTITION=false
```

The earlier semantic design's prohibition on public or committed raw rows
continues. Its allowance for a later separately reviewed private artifact
contract is exercised here: raw pages are retained only under the protected
local root below. This is not a change to the accepted source or gates.

## 2. Direct-Windows preflight and secret boundary

The future protected runner must use an atomic PowerShell scope or reviewed
`.ps1` entrypoint, the exact authoritative branch and reviewed HEAD/blob
bindings, a clean tree, and the canonical `.venv-real-execution` interpreter
with `scripts/check_current_protected_environment.py`. It must finish all
repository, source-contract, environment, private-root containment, and
existing-artifact checks before the first network request. The protected
runner must reject generated Orca task worktrees, as the existing V8 direct
runner does. A failed or ambiguous preflight blocks before the network
boundary.

```text
API_KEY_SOURCE=PROCESS_ENVIRONMENT_JQUANTS_API_KEY_ONLY
API_KEY_PRINTED=false
API_KEY_COMMITTED=false
API_KEY_PERSISTED=false
API_KEY_PASSED_ON_COMMAND_LINE=false
```

Preflight may check only that `JQUANTS_API_KEY` exists and is non-empty. It
must never report the value, length, hash, prefix, suffix, or raw exception
text that might include it. The key is supplied only as the `x-api-key`
request header. No command-line argument, artifact, log, traceback, or safe
report may contain it. Clear any transient child-process environment copy in
`finally`; do not modify the caller's process environment merely to clear a
copy that the runner did not create.

## 3. Single logical acquisition and bounded transport

One logical acquisition is the entire result of the fixed
`/v2/equities/master?date=20260731` query, including all continuation pages.
Page 1 uses exactly the fixed endpoint and date. Each later page preserves
them and adds only the server-provided continuation parameter required by
the same query. A continuation token is opaque and private. Redirects to
another endpoint, date changes, extra filters, or user-supplied pagination
values block. The runner tracks previously used tokens and rejects reuse,
cycles, or a token that would require page 101. The page count includes page
1. Pagination is part of the same logical acquisition, not a new acquisition.

```text
LOGICAL_ACQUISITION_COUNT_MAX=1
MAX_PAGES=100
MAX_TRANSPORT_ATTEMPTS_PER_PAGE=3
RETRYABLE=TIMEOUT,CONNECTION_OR_TLS_FAILURE,HTTP_429,HTTP_5XX
NON_RETRYABLE=OTHER_HTTP_4XX,INVALID_RESPONSE_SCHEMA,PAGINATION_LOOP_OR_LIMIT
BACKOFF_SECONDS=2,5
```

Each page gets at most three total HTTP attempts, with a 2-second wait after
the first retryable failure and a 5-second wait after the second. Every
attempt uses the same page URL and request semantics; the request counter
counts actual HTTP attempts. HTTP 3xx is not followed to another endpoint
and blocks under source response failure. A response with an invalid
pagination envelope, malformed body, or inconsistent continuation is not a
transport retry. No retry changes source, date, filters, or methodology.
After a complete raw content lock exists, no parser, software, semantic,
publication, or reporting failure authorizes a fetch; replay is offline from
that exact locked artifact. A semantic/hash mismatch is terminal for this
mapping.

The transport layer may parse only the response envelope fields required to
validate page-chain shape and obtain the next continuation token. It must
not inspect, filter, count, hash, log, or disclose ticker membership during
acquisition. It preserves exact response body bytes separately from any
envelope parse. Any invalid required envelope blocks before a complete
content lock is declared.

## 4. Private raw content lock

```text
PRIVATE_ROOT=%LOCALAPPDATA%\stock-analyzer\private\v8-jquants-identity-recovery
RAW_ARTIFACT_ID=eq-master-20260731
RAW_ARTIFACT_WRITE_ONCE=true
RAW_ARTIFACT_OUTSIDE_REPOSITORY=true
RAW_ARTIFACT_PUBLIC=false
RAW_ARTIFACT_COMMITTED=false
```

The implementation derives exact staging, raw-final, and recovery-final
subpaths mechanically from this root; it takes no user-supplied artifact
path. Before network access, resolve the local application-data anchor and
all existing ancestors, reject reparse points, symlinks, and junctions in
the path ancestry, and prove the resulting root lies outside the repository.
Repeat relevant containment and collision checks before publication. If a
complete raw artifact already exists, acquisition must not run again; a
valid artifact may be processed offline. A pre-existing incomplete,
ambiguous, or conflicting state blocks and cannot be silently deleted,
overwritten, or treated as complete.

For each successful page, write its exact HTTP response body bytes to a
new file in a private staging directory with exclusive/create-new semantics.
Privately record sequential page index, byte count, SHA-256 of those exact
bytes, and only the pagination state needed for deterministic chain replay.
Continuation tokens may occur only in this private page/manifest state, not
in the public report. Flush each page and its metadata durably before
advancing. Reject duplicate page index, token reuse, missing/extra page,
count/hash mismatch, or a non-terminating final page. A complete private
manifest must bind the ordered page records, fixed query identity, terminal
pagination state, and the expected page count. Its self-hash uses canonical
serialization excluding the self-hash field.

After the final page is durably written, re-read the staged bytes, validate
all page hashes/counts and page-chain consistency, and durably write the
complete manifest. Publish the entire staged directory to a previously
absent final location with an atomic no-overwrite operation, then verify its
completed state. If the filesystem cannot provide the required durable,
no-overwrite publication semantics, block. A staging directory or a final
directory without a verified complete manifest is never semantic input.
Leave failed or partial staging state private for read-only adjudication;
never reinterpret it as a content lock. The finished artifact must support
deterministic offline replay with zero J-Quants requests.

## 5. Offline semantic boundary and recovered identity

Only after the complete raw artifact is published and validated may the
implementation read its pages for semantic parsing. All semantic input comes
from locked bytes, never live response objects. The acceptance sequence in
`V13_V8_JQUANTS_IDENTITY_RECOVERY_DESIGN.md` remains authoritative:

```text
1. every accepted row Date == 2026-07-31
2. filter Mkt in {0111,0112}
3. filter ProdCat == 011
4. frozen Code normalization
5. canonical SHA-order
6. eligible count == 3110
7. eligible hash pin match
8. T0 hash pin match
9. T1 hash pin match
10. T2 hash pin match
11. T3 hash pin match
12. T_spare hash pin match
```

This list expresses the existing acceptance pipeline, including filtering
and normalization steps; it creates no new filter or pin. Any date, count,
eligible hash, or block hash mismatch is `DATA_QUALITY_FAILURE` and terminal
for this mapping. No refetch, alternate date, filter adjustment, or
replacement partition follows.

On full PASS only, publish a second private write-once artifact with the
actual eligible and T0/T1/T2/T3/T_spare block assignments needed to resume
V13. It has schema `V8_JQUANTS_IDENTITY_RECOVERY_MANIFEST_V1` and records the
frozen contract identifier, fixed query/date/filter/order, ordered raw page
counts and hashes, eligible count/hash, all block sizes and hashes, actual
block assignments, implementation commit and source/blob provenance,
recovery timestamp, and a self-hash computed over canonical bytes excluding
the self-hash field. Validate it against the raw lock and all pins before
atomic durable no-overwrite publication. Existing conflicting output blocks;
no overwrite or public copy is allowed. It contains no API key or raw rows.

```text
ORIGINAL_MANIFEST_BYTE_EXACT_RECOVERED=false
ORIGINAL_RAW_XLS_BYTE_EXACT_RECOVERED=false
ORIGINAL_ELIGIBLE_UNIVERSE_IDENTITY_RECOVERED=true
ORIGINAL_PARTITION_BLOCK_IDENTITY_RECOVERED=true
RECOVERY_EQUIVALENCE_CLASS=SEMANTIC_UNIVERSE_PLUS_ALL_BLOCK_HASHES_EXACT
```

These success assertions apply only after all gates and private publication
pass. The public terminal report never contains memberships or assignments.

## 6. Closed safe report and failure taxonomy

The runner emits exactly one safe terminal line with fixed-order fields from
closed enums, booleans, nonnegative counters, and approved derived hashes.
The report contains no URL query, continuation token, response/body, API
key, exception text, private path, ticker, or block membership. Optional
count/match fields are emitted only after their gate is reached. A report
validation failure maps to a closed failure code; it never prints the
offending value.

```text
JQUANTS_RECOVERY_RESULT=PASS|BLOCK
NETWORK_BOUNDARY_CROSSED=true|false
JQUANTS_LOGICAL_ACQUISITIONS=0|1
JQUANTS_HTTP_REQUESTS=<nonnegative integer>
STAGE=PRE_GATE|SOURCE_ACQUISITION|RAW_CONTENT_LOCK|OFFLINE_SEMANTICS|RECOVERY_PUBLICATION|COMPLETE
REASON=NONE|PRE_GATE_REPOSITORY_BLOCK|PRE_GATE_PROVENANCE_BLOCK|PRE_GATE_ENVIRONMENT_BLOCK|PRE_GATE_CREDENTIAL_BLOCK|PRE_GATE_PRIVATE_ROOT_BLOCK|PRE_GATE_EXISTING_ARTIFACT_BLOCK|SOURCE_TIMEOUT|SOURCE_TRANSPORT_FAILED|SOURCE_HTTP_429_EXHAUSTED|SOURCE_HTTP_5XX_EXHAUSTED|SOURCE_HTTP_4XX|SOURCE_RESPONSE_SCHEMA_INVALID|PAGINATION_LOOP|PAGINATION_LIMIT|RAW_CONTENT_LOCK_PUBLICATION_FAILED|EFFECTIVE_DATE_MISMATCH|ELIGIBLE_COUNT_MISMATCH|ELIGIBLE_HASH_MISMATCH|T0_HASH_MISMATCH|T1_HASH_MISMATCH|T2_HASH_MISMATCH|T3_HASH_MISMATCH|T_SPARE_HASH_MISMATCH|RECOVERY_ARTIFACT_PUBLICATION_FAILED|UNEXPECTED_FAILURE
RAW_CONTENT_LOCK_PUBLISHED=true|false
RECOVERY_ARTIFACT_PUBLISHED=true|false
ELIGIBLE_COUNT=<nonnegative integer or omitted>
ELIGIBLE_HASH_MATCH=true|false|unknown
T0_HASH_MATCH=true|false|unknown
T1_HASH_MATCH=true|false|unknown
T2_HASH_MATCH=true|false|unknown
T3_HASH_MATCH=true|false|unknown
T_SPARE_HASH_MATCH=true|false|unknown
```

`NONE` occurs only on PASS. A non-retryable HTTP 3xx maps to
`SOURCE_RESPONSE_SCHEMA_INVALID`; other non-retryable 4xx maps to
`SOURCE_HTTP_4XX`. A retryable timeout/transport/429/5xx failure is reported
under its corresponding closed code only after attempts are exhausted.
Malformed envelope or page-chain schema maps to
`SOURCE_RESPONSE_SCHEMA_INVALID`; token reuse maps to `PAGINATION_LOOP` and
page 101 to `PAGINATION_LIMIT`. Raw-lock failures after receipt of pages
map to `RAW_CONTENT_LOCK_PUBLICATION_FAILED`. Unexpected exceptions are
sanitized to `UNEXPECTED_FAILURE`. The stage and counters must reflect the
last proven boundary and durable state, never an inferred PASS.

## 7. Next implementation checkpoint

After GPT exact-SHA PASS of this design, a separate implementation Issue
should add a testable Python module using historical `src.v8_partition`
ordering, hashing, and allocation primitives; a protected direct-Windows
entrypoint with exact branch/HEAD/blob preflight and the canonical protected
environment checker; private raw-page and recovery-artifact publication;
and synthetic or loopback/injected-transport tests for pagination, retries,
content lock, semantic PASS, every terminal mismatch, and terminal-output
privacy. Implementation and tests make zero J-Quants requests. Real source
execution remains a later, separately reviewed and human-authorized step.

```text
REAL_JQUANTS_EXECUTION=false
JQUANTS_API_REQUESTS=0
NETWORK_REQUESTS_MARKET_DATA=0
PRIVATE_CREDENTIAL_READS=0
PRIVATE_CONTENT_READS=0
MODEL_FITS=0
BACKTESTS=0
REAL_TRADING=0
```
