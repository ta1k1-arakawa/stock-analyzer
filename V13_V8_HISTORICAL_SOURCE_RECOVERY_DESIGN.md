# V13 V8 Historical Source Recovery Design

```text
document_type=V8_HISTORICAL_SOURCE_RECOVERY_DESIGN
status=FROZEN_AWAITING_GPT_EXACT_SHA_REVIEW
issue=61
design_owner=GPT-5.6_SOL
real_source_execution=false
additional_jpx_request_authorized=false
recovery_pipeline_entered=false
```

## 1. Objective and acceptance rule

Recover the historical source snapshot that was originally acquired on
2026-08-10 and used for V8 partition recovery. The accepted source is the
original raw payload by exact byte identity, regardless of which public
archive, cache, or mirror may provide it. Provider identity alone does not
establish equivalence.

The pinned original source identity is:

```text
HISTORICAL_SOURCE_RECOVERY_STRATEGY=EXACT_RAW_BYTE_IDENTITY
ORIGINAL_ACQUISITION_UTC=2026-08-10T03:00:51.733526Z
ORIGINAL_RAW_BYTE_COUNT=830464
ORIGINAL_RAW_SHA256=6e401867d9ddf2524e4752f08fd3e3e434cd308c6d423839ca6e24fc7b1e1653
ORIGINAL_SOURCE_URL=https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls
```

The fixed current attachment URL is not an acceptable recovery target after
the deterministic HTTP 404. JPX replaces the Excel attachment over time, so
a current-month payload cannot be presumed to represent the original source
snapshot.

## 2. Exact frozen controls

```text
CURRENT_MOVING_ATTACHMENT_RETRY=PROHIBITED
CURRENT_MONTH_PAYLOAD_SUBSTITUTION=PROHIBITED
ARCHIVE_OR_MIRROR_PAYLOAD_ACCEPTANCE=ONLY_IF_RAW_BYTE_COUNT_AND_SHA256_EXACT_MATCH
HASH_MISMATCH_SEMANTIC_PARSE=PROHIBITED
HASH_MISMATCH_RECOVERY_PIPELINE_ENTRY=PROHIBITED
NEW_PARTITION_SUBSTITUTION=PROHIBITED
```

Both the complete raw byte count and SHA-256 must match the pinned values
before any parsing. A mismatch rejects the candidate immediately. Do not
inspect ticker rows, parse semantic content, or enter recovery stages for a
mismatching candidate.

## 3. Candidate discovery and future acquisition boundary

1. Prefer a historical snapshot of the exact original JPX URL at or nearest
   to the original acquisition time/date.
2. Metadata-only candidate discovery may identify public archives, caches,
   or mirrors. This Issue performs no external candidate acquisition.
3. Any future raw-payload acquisition requires separate explicit human
   authorization and a reviewed direct-execution contract.
4. Hash the first complete candidate payload before parsing it.
5. If both raw-byte pins match, preserve the matched payload as the single
   authoritative recovered historical source under a separately reviewed
   write-once/private artifact contract, then run the already-frozen V8
   recovery pipeline offline against those exact bytes.
6. No source-provider substitution may weaken the exact-byte acceptance
   gate.
7. If the exact historical bytes cannot be recovered, stop for GPT/human
   adjudication. Do not fall back to current-month data or construct a new
   partition.

## 4. Relationship to the frozen V8 recovery design

This design only defines how the original raw source bytes may be recovered.
It does not change the existing V8 semantic block-identity recovery method,
its T0 and eligible-universe gates, its T1/T2/T3/T_spare identity pins, or
its recovery-artifact contract. The recovery pipeline may run only offline
after a candidate passes both exact raw-byte identity pins and after the
matched source is preserved under the separately reviewed artifact
contract.

## 5. Authority and current disposition

Issue #60's third one-shot authorization was consumed by one JPX request,
which ended in source acquisition with `SOURCE_HTTP_STATUS_404`. No complete
source payload was accepted, the recovery pipeline was not entered, no
recovery artifact was published, and no partition identity mismatch or
data-quality, strategy, or profitability failure was observed.

This design is frozen pending GPT exact-SHA review. It grants no source
acquisition, network, archive request, private-data, recovery execution,
model-fit, backtest, forward-paper, or trading authority. A later task must
obtain the required separate human authorization and reviewed direct
execution contract before acquiring any candidate payload.

```text
REAL_SOURCE_EXECUTION=false
ADDITIONAL_JPX_REQUEST_AUTHORIZED=false
ARCHIVE_PAYLOAD_REQUESTS=0
MARKET_DATA_NETWORK_REQUESTS=0
PRIVATE_CONTENT_READS=0
SEALED_IDENTITY_READS=0
MODEL_FITS=0
BACKTESTS=0
REAL_TRADING=0
CURRENT_MONTH_SOURCE_ACCEPTED=false
NEW_PARTITION_SUBSTITUTED=false
```
