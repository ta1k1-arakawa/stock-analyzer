# V13 V8 J-Quants Identity Recovery Design

```text
document_type=V8_JQUANTS_SEMANTIC_AND_BLOCK_IDENTITY_RECOVERY_DESIGN
status=REMEDIATED_AWAITING_GPT_EXACT_SHA_REVIEW
issue=64
design_owner=GPT-5.6_SOL
real_jquants_execution=false
jquants_api_requests=0
```

## 1. Objective and relationship to prior recovery design

Reconstruct the original V8 eligible-universe identity and deterministic
partition block identities using the user's existing J-Quants Standard
access. This successor design replaces exact raw-XLS byte recovery as the
current recovery route. The previous exact-byte design and its historical
facts remain immutable provenance; this route must never be described as
recovery of the original manifest bytes or raw XLS bytes.

The original V8 identity anchors remain unchanged:

```text
ORIGINAL_SOURCE_ACQUISITION_UTC=2026-08-10T03:00:51.733526Z
ORIGINAL_RAW_BYTE_COUNT_AUDIT_ONLY=830464
ORIGINAL_RAW_SHA256_AUDIT_ONLY=6e401867d9ddf2524e4752f08fd3e3e434cd308c6d423839ca6e24fc7b1e1653
ORIGINAL_ELIGIBLE_TICKER_COUNT=3110
ORIGINAL_ELIGIBLE_TICKER_LIST_SHA256=37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405
ORIGINAL_T0_TICKER_LIST_SHA256=12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7
ORIGINAL_T1_TICKER_LIST_SHA256=262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d
ORIGINAL_T2_TICKER_LIST_SHA256=e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500
ORIGINAL_T3_TICKER_LIST_SHA256=43a585f4c3341307e7c67561c54780322b0f253fefa628a7c6129773901a7b7a
ORIGINAL_T_SPARE_TICKER_LIST_SHA256=360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70
```

The original raw byte count and SHA-256 are audit facts only and are not
acceptance gates for this semantic route.

## 2. Frozen source and query

```text
RECOVERY_STRATEGY=JQUANTS_STANDARD_SEMANTIC_AND_BLOCK_IDENTITY
JQUANTS_PLAN=STANDARD
JQUANTS_API_VERSION=V2
JQUANTS_ENDPOINT=https://api.jquants.com/v2/equities/master
JQUANTS_QUERY_DATE=20260731
JQUANTS_AUTH_HEADER=x-api-key
JQUANTS_RAW_ROWS_PUBLICLY_PERSISTED=false
JQUANTS_RAW_ROWS_COMMITTED=false
```

The query date represents the month-end listing basis of the lost JPX
monthly listing. Every accepted row must have effective `Date` exactly
`2026-07-31`. If the service resolves the query to any other effective date,
block. Pagination, if required, is part of this single logical acquisition.

## 3. Frozen normalization and eligibility mapping

The historical V8 selector was Prime or Standard, Domestic Stocks, and a
normalized four-character `[0-9A-Z]{4}` security code. Use this fixed
J-Quants mapping without tuning:

```text
MARKET_CODES_ACCEPTED={0111,0112}
PRODUCT_CATEGORY_ACCEPTED={011}
```

Normalize `Code` as follows:

1. Convert to uppercase string and trim surrounding whitespace.
2. If it is exactly five `[0-9A-Z]` characters ending in `0`, strip that
   final `0`.
3. If it is already exactly four `[0-9A-Z]` characters, retain it.
4. Otherwise exclude it as non-canonical.
5. Require the resulting code to match `[0-9A-Z]{4}`.
6. De-duplicate by normalized code.
7. Sort normalized eligible codes by `(SHA-256(UTF-8 code), code)` ascending
   for canonical eligible-list hashing and partition construction.

Use only ticker membership for recovery. Company names, margin category,
industry names, price/outcome fields, and other attributes do not affect
acceptance or partition identity. Do not broaden product categories, include
Growth, or vary date/filter rules to explain a mismatch.

## 4. Canonical hashes and deterministic partition construction

Preserve the existing V8 hash exactly:

```text
SHA256(UTF8("ticker1\nticker2\n...\n"))
```

Use the same historical V8 canonical order for the eligible-list hash and
partition construction:

```text
eligible_ordered = sort normalized eligible by (SHA-256(UTF-8 code), code) ascending
ELIGIBLE_TICKER_LIST_SHA256 = SHA256(UTF8("\n".join(eligible_ordered) + "\n"))
T0 = first 300 of eligible_ordered
```

Rebuild the remaining blocks using the original deterministic algorithm:

```text
DETERMINISTIC_ORDERING_RULE=sort eligible by (SHA-256(UTF-8 code), code) ascending
BLOCK_SIZE=300
T0=first 300 of eligible_ordered
LEGACY_EXPOSED_OUTSIDE_T0={1570,4689,5020,7211,7267,8306,9432}
FRESH_POOL=ordered eligible excluding T0 and the seven legacy-exposed-outside-T0 codes
T1=first 300 fresh
T2=next 300 fresh
T3=next 300 fresh
T_SPARE=remaining fresh
```

No RNG, redraw, substitution, fallback partition, nearest-match, or adaptive
filtering is allowed.

## 5. Ordered acceptance gates

A later reviewed acquisition/reconstruction passes only if all gates succeed
in this order:

```text
1. JQUANTS_EFFECTIVE_DATE_EXACT_MATCH=true  # 2026-07-31
2. ELIGIBLE_COUNT=3110
3. ELIGIBLE_TICKER_LIST_SHA256=37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405
4. T0_TICKER_LIST_SHA256=12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7
5. T1_TICKER_LIST_SHA256=262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d
6. T2_TICKER_LIST_SHA256=e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500
7. T3_TICKER_LIST_SHA256=43a585f4c3341307e7c67561c54780322b0f253fefa628a7c6129773901a7b7a
8. T_SPARE_TICKER_LIST_SHA256=360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70
```

Any mismatch is terminal `BLOCK` for this mapping. It does not authorize an
adjacent date, alternate category mapping, changed normalization, inclusion
of Growth/foreign/ETF rows, or a replacement partition. Return to GPT/human
adjudication before any methodology change.

Only if all gates pass may the result be recorded as:

```text
ORIGINAL_MANIFEST_BYTE_EXACT_RECOVERED=false
ORIGINAL_RAW_XLS_BYTE_EXACT_RECOVERED=false
ORIGINAL_ELIGIBLE_UNIVERSE_IDENTITY_RECOVERED=true
ORIGINAL_PARTITION_BLOCK_IDENTITY_RECOVERED=true
RECOVERY_EQUIVALENCE_CLASS=SEMANTIC_UNIVERSE_PLUS_ALL_BLOCK_HASHES_EXACT
```

## 6. Later execution boundary

The human approved the J-Quants Standard source choice. This design issue
itself performs zero requests and does not access or request an API key. Any
later direct-Windows execution must use only private local configuration or
environment for the key and must never print, commit, log, or paste it.

A later reviewed execution must query only the frozen V2 endpoint/date,
including required pagination for the same logical acquisition. Transport
retry is permitted only for public-plumbing failures before a complete
logical payload under a separately reviewed bounded policy. A semantic or
hash mismatch cannot justify re-acquisition or alternate-date querying. Only
safe derived counts and hashes may be publicly reported; raw rows are not
persisted or committed, and T1/T2/T3/T_spare memberships are not disclosed.
No model fit, backtest, forward-paper, or real trading is authorized here.

```text
REAL_JQUANTS_EXECUTION=false
JQUANTS_API_REQUESTS=0
JPX_SOURCE_REQUESTS=0
ARCHIVE_PAYLOAD_REQUESTS=0
MARKET_DATA_NETWORK_REQUESTS=0
PRIVATE_CREDENTIAL_READS=0
PRIVATE_CONTENT_READS=0
SEALED_IDENTITY_READS=0
RAW_JQUANTS_ROWS_PERSISTED=false
TICKER_MEMBERSHIPS_PUBLICLY_DISCLOSED=false
MODEL_FITS=0
BACKTESTS=0
REAL_TRADING=0
REPLACEMENT_PARTITION_CREATED=false
ADAPTIVE_FILTERING=false
ALTERNATE_DATE_QUERY=false
```
