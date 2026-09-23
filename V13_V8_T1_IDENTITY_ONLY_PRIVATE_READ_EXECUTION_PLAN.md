# V13 V8 T1 Identity-Only Private-Read Execution Plan

```text
TASK=V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_RESOLVER_IMPLEMENTATION
AUTHORIZATION_REVIEWED_SHA=18a99fbb3740ecb827abc14513fad1402f7632ec
V13_FROZEN_DESIGN_BLOB_SHA1=3bfcd695c69f6dac480f8fc99ca4f3916f668e4a
PRIVATE_READ_EXECUTION_IN_THIS_ISSUE=false
AUTHORIZATION_CONSUMED=false
```

## Purpose and scope

This plan freezes the later, separately reviewed operation that may use the
one-shot authorization from Issue #38. Issue #41 implements and synthetically
proves the resolver only. It does not locate or read a real private/sealed
file, consume authorization, request market data, or open price/outcome data.

The implementation is a standard-library-only selective JSON scanner. It
reads one explicitly supplied manifest source once, parses the fixed public
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

The later caller supplies an explicit absolute output path outside the source
repository. Reject an existing destination and any path resolving inside the
repository before opening the source. Publish one file with an atomic
no-overwrite operation; do not store raw manifest bytes or T2/T3/T_spare
membership.

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

Only safe fields may be emitted: status, boundary boolean, T1 count/hash match,
stated-hash match booleans, state-written boolean, request/read counters, and
the false T2/T3/T_spare-retention boolean. Never emit input/output paths,
block identities, raw manifest bytes, identity samples, private-derived
prefix hashes, prices, outcomes, features, or metrics. No stdout output is
needed from the resolver.

## Synthetic acceptance coverage

Tests use generated synthetic four-character identities and temporary files
only. They cover a valid T1-only state; wrong stated hashes; actual T1 hash,
count, uniqueness, and format failures; duplicate/ambiguous keys; malformed
input; output collision and repository-contained paths; leakage absence from
return data, state, exceptions, and stdout; a T2/T3/T_spare-only sentinel; and
absence of a network/full-object JSON path. Synthetic bindings are supplied
only to the private test seam; the production entry point is fixed to the
public Issue #41 bindings.

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
