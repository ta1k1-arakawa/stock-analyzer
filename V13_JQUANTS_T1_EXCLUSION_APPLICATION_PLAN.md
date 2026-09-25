# V13 recovered T1 exclusion application preparation

```text
STATUS=PRE_GATE_DURABLE_STATE_CLASSIFICATION_REMEDIATED_AWAITING_GPT_EXACT_SHA_REVIEW
SOURCE_SCHEMA=V8_JQUANTS_IDENTITY_RECOVERY_MANIFEST_V1
SOURCE_CONTRACT=V13_V8_JQUANTS_IDENTITY_RECOVERY_DESIGN
QUERY_DATE=20260731
EFFECTIVE_DATE=2026-07-31
CANONICAL_ORDER=SHA256_UTF8_CODE_THEN_CODE_ASC
BLOCK_SIZE=300
ELIGIBLE_COUNT=3110
ELIGIBLE_SHA256=37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405
T1_COUNT=300
T1_SHA256=262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d
SOURCE_IMPLEMENTATION_REVIEWED_SHA=7565ca723c76801d74d8d319d65d280a689b3cfa
SOURCE_IMPLEMENTATION_BLOB_SHA1=f46ea0c304b0bbd2d230b9850acba9eada9f6908
EXCLUSION_DISPOSITION=EXCLUDE_FULL_RECOVERED_T1_BLOCK
KNOWN_DEFINITELY_ACQUIRED_PREFIX_COUNT=297
PRIVATE_BOUNDARY=FIRST_CONTENT_BYTE_READ_FROM_VERIFIED_JQUANTS_RECOVERY_ARTIFACT
REAL_PRIVATE_READ_IN_ISSUE75=false
```

## Source, output, and authority

The future source is only `%LOCALAPPDATA%\stock-analyzer\private\v8-jquants-identity-recovery\recovery.json`. Its actual absolute path and contents remain private. Raw J-Quants pages are outside this operation. The state and consumed receipt belong in a separate `%LOCALAPPDATA%\stock-analyzer\private\v13-t1-exclusion-provenance` root. Both outputs are write-once. Existing output or unexpected root topology stops before source content read; neither output is reset or overwritten.

Issue #73 supplied human approval of the full recovered T1 300 exclusion amendment, bound to decision blob `276feb56f417f9d5b931d594598a1d8591330bfd`. The older V8 T1 identity-only authorization is unconsumed, but its source binding names the lost partition manifest. It supplies semantic limits and reviewed durability primitives only. It does not authorize the recovered J-Quants source. Future real execution requires a fresh point-of-use human authorization, exact reviewed implementation and authorization blobs, and a separately reviewed `DIRECT_WINDOWS_POWERSHELL` Issue. The preparation here performs no real private read or path discovery.

## Narrow parser and one-shot boundary

The stdlib scanner reads the source once. It parses only approved public root scalars, `block_sizes.T1`, `block_hashes.T1`, and `assignments.T1`; it validates syntax while skipping `assignments.eligible`, `T0`, `T2`, `T3`, `T_spare`, raw page metadata, and other fields without making their values into Python objects. It rejects duplicate relevant keys, missing or extra structural keys, malformed JSON, incorrect public bindings, wrong T1 stated count/hash, actual T1 count/hash mismatch, duplicate T1 codes, and codes outside `[0-9A-Z]{4}`. The actual T1 hash is SHA-256 over ASCII codes joined by newline with a final newline. Production does not use full recovery `json.loads`, `_exact_json`, or `validate_recovery`.

All repository/branch/HEAD/remote/clean/blob/environment/authority and path-topology checks precede source content read. The source opens once. On a nonempty first byte, the operation is consumed. A no-overwrite receipt is published and durably flushed using the previously reviewed Windows write-through and `FlushFileBuffers` primitives before any remainder byte is read. Receipt publication failure after first byte is a terminal post-boundary failure; no automatic retry or second source read exists. A current-invocation pre-gate stop can report unconsumed only after the durable output topology is proven clear. Existing receipt, state, unexpected output-root contents, or any unresolved prior-boundary topology stops with unknown boundary and consumption status, non-reusable authority, and no second execution pending adjudication. Durable state is never reset or overwritten to recover.

The private state contains only safe source bindings and T1 membership with `schema=V13_JQUANTS_T1_EXCLUSION_STATE_V1`, the pinned T1 hash/count, the 297 definitely acquired prefix count, and `EXCLUDE_FULL_RECOVERED_T1_BLOCK`. The receipt uses `V13_JQUANTS_T1_EXCLUSION_CONSUMED_RECEIPT_V1` and contains public bindings, counts, enums, and booleans only. Neither file contains non-T1 membership, raw source bytes, prices, outcomes, features, metrics, or API material. Terminal reports contain safe gate, boundary, consumption, read count, hash/binding, output, and result fields plus `NETWORK_REQUESTS=0`, `PRICE_PAYLOAD_READS=0`, `OUTCOME_READS=0`, `NON_T1_IDENTITIES_RETAINED=false`, and `V13_UNIVERSE_SELECTED=false`.

## Preparation and next action

`scripts/run_v13_jquants_t1_exclusion_direct_windows.ps1` is a future pre-gate wrapper. It requires exact HEAD and blob arguments, checked public authorization records, canonical protected environment, metadata-only source/output topology, the separately reviewed `V13_JQUANTS_T1_EXCLUSION_POINT_OF_USE_AUTHORIZATION.json` record, and `-ExecuteReviewedPrivateRead`. It is not executed on real private data in this task. Synthetic tests use temporary files and injected streams only.

Issue #76 GPT exact-SHA review of `515130991ea023896520a446eab2621287480148` passed C0/H0/M0/L0 and registered the active branch-wide advisory. The advisory is consulted. The frozen V13 methodology remains unchanged for the first measurement. P2-a and P2-b are not implemented; process cadence should be compressed while preserving all private and one-shot gates. The next action after this task is GPT exact-SHA independent review. No ticker selection, model fit, backtest, forward-paper, or trading authority is created.

Issue #75 GPT exact-SHA review of `f2f0878cabfe7d4e15ce709f5c1ec34f7755a4fe` blocked with HIGH_1 `PRE_GATE_DURABLE_STATE_COLLISION_CAN_BE_MISREPORTED_AS_AUTHORIZATION_UNCONSUMED`. Issue #77 remediates the direct-Windows wrapper classification and makes a contradictory child exit/report fail closed to a non-PASS unknown-boundary report. Both changes await independent GPT exact-SHA review. They create no point-of-use authorization and perform no real private read.
