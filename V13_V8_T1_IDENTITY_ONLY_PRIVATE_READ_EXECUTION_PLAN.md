# V13 V8 T1 Identity-Only Private-Read Execution Plan

```text
TASK=V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_EXECUTION_HARNESS
ISSUE41_REVIEWED_SHA=63fa6b7541694565eb171a98d11df0092ab20034
HARNESS_BASE_SHA=43f80b76bd82d03d02779d890cc6ddde552f49e9
AUTHORIZATION_REVIEWED_SHA=18a99fbb3740ecb827abc14513fad1402f7632ec
V13_FROZEN_DESIGN_BLOB_SHA1=3bfcd695c69f6dac480f8fc99ca4f3916f668e4a
PRIVATE_READ_EXECUTION_IN_THIS_ISSUE=false
AUTHORIZATION_CONSUMED=false
```

## Purpose and scope

This plan freezes the later, separately reviewed operation that may use the
one-shot authorization from Issue #38. Issues #41 and #42 implement and
synthetically prove the selective resolver and execution harness only. They
do not locate or read a real private/sealed file, consume authorization,
request market data, or open price/outcome data.

The implementation is a standard-library-only selective JSON scanner and
execution harness. It opens one explicitly supplied manifest source once, parses the fixed public
manifest structure, retains only `block_assignments.T1` and the public binding
scalars needed for verification, and writes the T1-only state described below.
It must not call `src.v8_partition.read_partition_manifest()`, call
`json.loads()` on a full object, or recompute the full manifest self-hash.
Non-T1 assignment arrays are scanned only to validate JSON/string-array
structure; their strings are not decoded, compared, hashed, returned, printed,
or persisted.

## Frozen bindings

```text
SOURCE_MANIFEST_STATED_SHA256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
STATED_T1_TICKER_LIST_SHA256=262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d
COMPUTED_T1_TICKER_LIST_SHA256=SHA256(UTF8(join(T1,"\n") + "\n"))
T1_COUNT=300
T1_CODE_FORMAT=[0-9A-Z]{4}
KNOWN_DEFINITELY_ACQUIRED_PREFIX_COUNT=297
```

The resolver checks the manifest's *stated* self-hash against the public
binding. Recomputing the self-hash would require including unauthorized block
arrays and is prohibited. It separately checks the stated T1 hash, computes
the canonical T1 hash from the extracted ordered list, requires equality with
the public expected hash, checks the manifest schema and T1 block-size scalar,
and requires 300 unique canonical T1 codes.

## One-shot boundary and failure state

```text
PRIVATE_BOUNDARY=FIRST_CONTENT_BYTE_READ_FROM_VERIFIED_PRIVATE_V8_PARTITION_MANIFEST
PRE_BOUNDARY_FAILURE=AUTHORIZATION_UNCONSUMED
POST_BOUNDARY_FAILURE=AUTHORIZATION_CONSUMED_NO_RETRY
AUTOMATIC_RETRY=false
SECOND_PRIVATE_SOURCE_READ=false
```

Before the later boundary, the direct Windows execution must finish all
authorization, exact-SHA, checkout, source-binding, output-path, no-overwrite,
environment, and safe-report preflight. At the first content byte, record the
one-shot authority as consumed. Any later parse, binding, write, or reporting
failure terminates that attempt; it must not automatically retry or read the
source again. No real boundary was crossed in Issue #41.

## Private state and output safety

The later caller supplies explicit absolute private-state and consumed-receipt
destinations outside the repository. Both must be distinct, outside-repository
paths with existing writable parent directories. Reject relative paths,
collisions, existing destinations (including symlinks), and existing durable
state before opening the source. No output path or source path is printed.

The resolver opens the source once and reads exactly one byte. Empty or
unreadable sources fail before the boundary and do not invoke the boundary
callback. Immediately after a non-empty first byte, the callback atomically
publishes the consumed receipt with no overwrite. Only after receipt
publication succeeds does the resolver read the remainder from the same open
stream and pass the combined bytes to the existing selective scanner. No
second source open or read is allowed. Receipt-publication failure is
POST_BOUNDARY and is never retried; the report must say the statistical
boundary was crossed even though the receipt was not published. Any later
parse, binding, private-state publication, or reporting failure leaves an
already-published receipt in place and forbids automatic retry. A catastrophic
loss between the first byte and receipt publication is intrinsically
ambiguous; all future execution must fail closed and may not assume PRE_GATE.

The receipt schema is `V13_V8_T1_PRIVATE_READ_CONSUMED_RECEIPT_V1` and
contains only the fixed public values below. No path, ticker, raw byte,
identity sample, price, outcome, feature, metric, or private-derived prefix
hash is permitted.

```text
schema=V13_V8_T1_PRIVATE_READ_CONSUMED_RECEIPT_V1
study=V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON
operation_class=STATISTICALLY_IRREVERSIBLE_GATE
boundary=FIRST_CONTENT_BYTE_READ_FROM_VERIFIED_PRIVATE_V8_PARTITION_MANIFEST
authorization_reviewed_sha=18a99fbb3740ecb827abc14513fad1402f7632ec
resolver_reviewed_sha=63fa6b7541694565eb171a98d11df0092ab20034
expected_partition_manifest_stated_sha256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
expected_t1_ticker_list_sha256=262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d
expected_t1_count=300
authorization_consumed=true
```

Before invoking this harness, the later separately reviewed
`DIRECT_WINDOWS_POWERSHELL` caller must finish Git branch/HEAD/clean-tree and
reviewed-commit checks, authorization and reviewed-binding checks, unique
private source metadata resolution, both output/durable-state collision
checks, and its safe-report preflight. It supplies paths transiently and does
not print them. This harness independently checks the public authorization
artifact and frozen design blob, absolute/external source and outputs,
distinct/no-overwrite destinations, and output-parent writability before it
opens the source. It never discovers a private path or runs Git/network code.

Publish the T1-only private state with the existing atomic no-overwrite
operation; do not store raw manifest bytes or T2/T3/T_spare membership.

The private state schema is exactly:

```text
schema=V13_V8_T1_IDENTITY_STATE_V1
source_partition_manifest_stated_sha256=<public expected hash>
t1_ticker_list_sha256=<public expected hash>
t1_count=300
known_definitely_acquired_prefix_count=297
t1_membership=<private ordered 300-code list>
```

Do not decide whether the eventual V13 exclusion uses the known 297 prefix or
all 300 T1 members. Attempt #2 request count remains unknown/not persisted.
The eventual exclusion treatment is a later GPT/human decision if required.

## Safe reporting

The runner prints only safe fields: `PRE_GATE_STATUS`,
`PRIVATE_BOUNDARY_CROSSED`, `GATE_CONSUMED`, `PRIVATE_READS`,
`NETWORK_REQUESTS`, `PRICE_PAYLOAD_READS`, `OUTCOME_READS`, `T1_COUNT`,
`T1_HASH_MATCH`, `MANIFEST_STATED_SHA_MATCH`, `PRIVATE_STATE_WRITTEN`,
`CONSUMED_RECEIPT_WRITTEN`, `T2_T3_TSPARE_IDENTITIES_RETAINED`,
`EXECUTION_RESULT`, `FAILURE_CLASS`, `AUTHORIZATION_REUSABLE`, and
`SECOND_EXECUTION_ALLOWED`. Never emit input/output paths, block identities,
raw manifest bytes, identity samples, private-derived prefix hashes, prices,
outcomes, features, or metrics. The resolver itself does not print.

## Synthetic acceptance coverage

Tests use generated synthetic four-character identities and temporary files
only. In addition to the #41 selective-parser cases, they prove one source
open, first-byte read before exactly one callback, remainder parsing through
that open stream only after callback success, no callback for empty/unreadable
input, POST_BOUNDARY callback/parser/hash/count/state-write failure handling,
receipt persistence/no retry, existing receipt/state pre-read blocking,
relative/repository-contained/colliding/reused destination rejection, safe
receipt fields, sentinel non-leakage, safe reports, no network/full-object
private-manifest parser path, and exact public reviewed bindings. Synthetic
bindings are confined to the private test seam; production uses the fixed
Issue #41 bindings.

## Issue #41 boundary record

```text
PRIVATE_CONTENT_READS=0
PRIVATE_PATH_DISCOVERY=0
T1_REAL_IDENTITIES_READ=0
NETWORK_REQUESTS=0
PRICE_PAYLOAD_READS=0
OUTCOME_READS=0
MODEL_FITS=0
BACKTESTS=0
AUTHORIZATION_CONSUMED=false
```

The actual private read remains a separate `DIRECT_WINDOWS_POWERSHELL` task
after GPT exact-SHA PASS for this implementation.
