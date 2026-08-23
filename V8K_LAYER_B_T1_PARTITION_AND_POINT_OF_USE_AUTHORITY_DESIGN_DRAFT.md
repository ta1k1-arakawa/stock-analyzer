# V8K Layer B T1 Partition and Point-of-Use Authority Design Draft

```text
document_type=V8K_LAYER_B_T1_PARTITION_AND_POINT_OF_USE_AUTHORITY_DESIGN
status=PROPOSED_NOT_FROZEN
study=V8K_HISTORICAL_RESEARCH
layer_b_reviewed_freeze_sha=9f91ad00ce86eba3b0b990476744f98253d11474
candidate_preregistration_blob=1a087bb93b947be4e93855b3c0b2d5b1b5b1b2d1
t1_confirmation_design_blob=0b7f35f744ad44111e80ffd80ab5bcc063fbcedb
candidate_id=V8K_PULLBACK_VOLUME_DRY_UP_FIXED_V1
T1_ACCESS_AUTHORIZED=false
T1_CONSUMED=false
T2_AUTHORIZED=false
future_profitability_established=false
deployment_allowed=false
```

This proposed design binds the human-frozen Layer-B candidate and T1 confirmation methodology above. It grants no public-network, private/sealed-data, partition-generation, membership-disclosure, price-acquisition, research-opening, T2, T3, deployment, or production authority. No V8H seed, membership, receipt, authorization, or private state is reused.

## 1. Frozen stage order

```text
1. PUBLIC_SOURCE_PREPARATION
2. PRIVATE_PARTITION_ESTABLISHMENT
3. T1_POINT_OF_USE_OPENING_AND_CONFIRMATION

SOURCE_COMPLETE_AND_LOCKED before PARTITION_SEED_CREATION
PARTITION_SEALED_AND_PRESERVED before T1_MEMBERSHIP_READ
T1_POINT_OF_USE_GATE before FIRST_PRIVATE_T1_MEMBERSHIP_CONTENT_READ
```

A completed stage proves only that stage's specified result; it grants no authority to the next stage. Each implementation, exact-SHA independent review, freeze, and applicable fresh human authorization remains separately required.

## 2. Public source preparation

```text
stage=PUBLIC_SOURCE_PREPARATION
operation_class=RETRIABLE_PUBLIC_PLUMBING
gate=HUMAN_V8K_PUBLIC_SOURCE_PREPARATION_GATE
source_snapshot_semantics=IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT
source_snapshot_clarification_commit=266999a8e48c77905dd7c7312fd41c7f38241d78
```

The exact inherited official-JPX provider/endpoint semantics are those reviewed for the V8 source snapshot: request only `https://www.jpx.co.jp/markets/statistics-equities/misc/01.html`; extract its official `data_j.xls` link; resolve it against that page; then request the resolved XLS URL. Both requests must be HTTPS to `www.jpx.co.jp`, with no credentials, nonstandard port, provider, or date substitution. This transcribes the existing `JPX_PAGE`, `DATA_LINK_PATTERN`, and trusted same-host endpoint validation semantics in `scripts/build_v8_partition_manifest.py`.

A later explicit human authorization for this named gate is standing only for the frozen public-plumbing scope. Inherited per-invocation retry policy applies to transport/setup failures only until the first complete raw XLS payload. Immediately preserve the raw bytes outside Git and record SHA-256 before semantic inspection. Thereafter:

```text
FIRST_COMPLETE_PAYLOAD_LOCKED=true
NO_REFETCH=true
NO_PROVIDER_SUBSTITUTION=true
NO_DATE_SUBSTITUTION=true
```

Parser, T0 reproduction, eligible-count, or other semantic/data-quality failure is `DATA_QUALITY_FAILURE` and stops the stage; it never authorizes fetch-until-PASS. Only pre-payload transport/setup failure is `PLUMBING_FAILURE_RETRIABLE`.

Reuse exact `src/v8_partition.py` universe/T0 semantics: `parse_eligible_universe`, canonical ordering, `build_universe_csv_bytes`, and `verify_t0_reproduction`. T0 must exactly reproduce the exposed 300 in `V4_UNIVERSE.csv`; V4 raw-byte equality is not required. Public evidence may contain only `source_raw_sha256`, `eligible_ticker_count`, `eligible_ticker_list_sha256`, T0 reproduction status, and safe Git/support provenance. It contains no ticker identities, raw payload, private path, seed, or block membership.

Structural authorization identity:

```text
V8K_HUMAN_AUTHORIZE_PUBLIC_SOURCE_PREPARATION_AT_
  <reviewed_frozen_partition_authority_design_commit>_WITH_
  <reviewed_public_source_support_implementation_sha>
```

Only SHA-256 of raw authorization may be durably recorded. Fixed receipt-key material is UTF-8 of:

```text
V8K_PUBLIC_SOURCE_PREPARATION_GATE_RECEIPT_KEY_V1\0
ta1k1-arakawa/stock-analyzer\0
V8K_HISTORICAL_RESEARCH\0
HUMAN_V8K_PUBLIC_SOURCE_PREPARATION_GATE
```

It excludes authorization text, attempt count, timestamps, payload hashes, and every other attempt-varying value.

## 3. Private partition establishment

```text
stage=PRIVATE_PARTITION_ESTABLISHMENT
generation_contract=V8K_PARTITION_GENERATION_V1
manifest_schema=V8K_PARTITION_MANIFEST_V1
gate=HUMAN_V8K_PRIVATE_PARTITION_GENERATION_GATE
consumption_boundary=IMMEDIATELY_BEFORE_AUTHORITATIVE_SEED_CREATION
partition_seed_bytes=32
```

This is a fresh V8K partition and reads no price, feature, label, candidate, or strategy-outcome information. Its source is the locked stage-1 snapshot.

```text
T0=existing_exposed_300
T1=fresh_validation_300
T2=fresh_sealed_holdout_300
T3=fresh_sealed_reserve_300
T_spare=remainder
fresh_pool=eligible_current_only - T0 - LEGACY_EXPOSED_TICKERS_OUTSIDE_T0
minimum_fresh_pool=900
```

`LEGACY_EXPOSED_TICKERS_OUTSIDE_T0` and canonical universe/T0 semantics are exactly those in `src/v8_partition.py`. Fewer than 900 fresh tickers is fail-closed `DATA_QUALITY_FAILURE`; no balancing, substitution, or manual adjustment exists.

After a fixed one-shot durable gate receipt is published, generate exactly 32 bytes from the OS CSPRNG and immediately private-durably publish the seed with exclusive/no-overwrite semantics. If authoritative seed persistence never succeeds after gate consumption, the result is `BLOCK_CLOSED / CHATGPT_DECISION_REQUIRED`; no second seed may be generated. The raw seed never enters Git, logs, stdout, public evidence, prompts, or exception messages.

For each canonical fresh ticker:

```text
allocation_key = HMAC-SHA256(
  key=partition_seed_bytes,
  msg=UTF8("V8K_PARTITION_ASSIGN_V1\0" + ticker_code)
)
sort=(allocation_key, ticker_code)
first_300=T1
next_300=T2
next_300=T3
remainder=T_spare
```

Duplicate ticker or allocation key, overlap, invalid block size, hash mismatch, or preservation failure blocks the stage. No reroll, reshuffle, reset, manual change, or membership substitution is allowed. Once the exact seed is durably persisted, a crash during allocation/manifest processing may continue only as `DETERMINISTIC_DURABLE_STATE`, using that same seed and locked source snapshot, without a new gate or new seed.

Canonical ticker/list hashing and canonical JSON exactly reuse `src/v8_partition.py`: list SHA-256 is `SHA256(UTF8("\n".join(tickers) + "\n"))`, and the manifest self-hash is canonical JSON excluding only its own self-hash field. The seed and `V8K_PARTITION_MANIFEST_V1` are machine-local private durable state. Public evidence has only hashes/counts/booleans/enums/safe provenance.

Structural authorization identity:

```text
V8K_HUMAN_AUTHORIZE_PRIVATE_PARTITION_GENERATION_AT_
  <reviewed_frozen_partition_authority_design_commit>_WITH_
  <reviewed_partition_generation_support_implementation_sha>_SOURCE_
  <source_raw_sha256>
```

Only its SHA-256 is recorded. Fixed receipt-key material is UTF-8 of:

```text
V8K_PRIVATE_PARTITION_GENERATION_GATE_RECEIPT_KEY_V1\0
ta1k1-arakawa/stock-analyzer\0
V8K_HISTORICAL_RESEARCH\0
HUMAN_V8K_PRIVATE_PARTITION_GENERATION_GATE
```

Every later point of use recomputes the private manifest self-hash, seed SHA-256 commitment, all five block-list hashes, and locked source-snapshot binding. A prior PASS artifact alone is insufficient.

## 4. T1 point-of-use opening and confirmation

```text
stage=T1_POINT_OF_USE_OPENING_AND_CONFIRMATION
gate=HUMAN_V8K_T1_CONFIRMATION_OPENING_GATE
operation_class=STATISTICALLY_IRREVERSIBLE_GATE
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_PRIVATE_T1_MEMBERSHIP_CONTENT_READ
max_validation_access=1
```

The future one-shot gate may authorize only partition verification, private T1 membership read by reviewed local execution code, required fixed historical-price acquisition for exactly T1, baseline-versus-frozen-candidate evaluation, and one frozen T1 confirmation decision. It binds the exact future frozen partition/authority-design commit, reviewed T1-support implementation SHA, the Layer-B freeze SHA and both bound blobs above, candidate ID, source snapshot hash, private manifest hash, and T1 ticker-list hash.

Structural authorization identity:

```text
V8K_HUMAN_AUTHORIZE_T1_CONFIRMATION_OPENING_AT_
  <reviewed_frozen_partition_authority_design_commit>_WITH_
  <reviewed_t1_support_implementation_sha>_LAYER_B_FREEZE_
  9f91ad00ce86eba3b0b990476744f98253d11474_CANDIDATE_PREREGISTRATION_BLOB_
  1a087bb93b947be4e93855b3c0b2d5b1b5b1b2d1_T1_DESIGN_BLOB_
  0b7f35f744ad44111e80ffd80ab5bcc063fbcedb_SOURCE_
  <source_raw_sha256>_PRIVATE_MANIFEST_<private_manifest_sha256>_T1_LIST_
  <t1_ticker_list_sha256>
```

Raw authorization, identities, membership lists, private paths, and raw price payloads are never printed or publicly persisted. Fixed receipt-key material is UTF-8 of:

```text
V8K_T1_CONFIRMATION_OPENING_GATE_RECEIPT_KEY_V1\0
ta1k1-arakawa/stock-analyzer\0
V8K_HISTORICAL_RESEARCH\0
HUMAN_V8K_T1_CONFIRMATION_OPENING_GATE
```

Durable publication/consumption of this fixed one-shot receipt is the
irreversible T1 consumption event. It occurs immediately before the first
private T1 membership-content read and must durably establish
`T1_CONSUMED=true` and `consumption_count=1` before that read. A failure
before durable receipt publication is `PRE_GATE_FAILURE` and leaves
`T1_CONSUMED=false`. Every failure after receipt consumption is `POST_GATE`
with respect to this T1 gate: authorization never becomes reusable; the
receipt cannot be deleted, reset, or replaced; and no second T1 opening is
permitted. The absence of later membership or result exposure does not restore
authorization. Membership/result exposure remains relevant to
`max_validation_access=1` accounting, but cannot delay or undo consumption.

### T1 public-safe confirmation evidence

The public-safe T1 confirmation evidence artifact must be sufficient for an
exact-SHA independent review of the frozen six criteria without revealing T1
membership. It may contain only safe aggregates and provenance: schema/version
and artifact role; study and candidate identifiers; exact reviewed/frozen Git
commit and blob bindings; `authorization_identity_sha256`; gate receipt-key
SHA-256, consumed boolean, and consumption count; `source_raw_sha256`;
`eligible_ticker_list_sha256` where applicable; `private_manifest_sha256`;
`t1_ticker_list_sha256`; T1 ticker count `300`; safe timestamps, counts,
booleans, hashes, and failure-classification enums.

It may also contain the baseline and candidate aggregate metrics required by
the frozen primary criteria: net profit, profit factor, MTM maximum drawdown,
book-cost maximum drawdown, and positive-year count; each of the six
primary-condition PASS/FAIL booleans; and `T1_RESULT`. The frozen secondary
aggregate diagnostics listed in `V8K_LAYER_B_T1_CONFIRMATION_DESIGN_DRAFT.md`
may be recorded, but remain nondecisional.

Public persistence or output is prohibited for ticker identities, membership
lists or ordering, private paths, raw seed, raw partition manifest, raw
authorization text, raw Yahoo/price payloads, per-ticker price rows,
per-ticker or per-trade private outcomes that could reveal membership, and any
other membership-reconstructing information.

Evaluation uses the six primary criteria already frozen in
`V8K_LAYER_B_T1_CONFIRMATION_DESIGN_DRAFT.md`, without restatement or
reinterpretation. Secondary diagnostics remain nondecisional. A scientifically
valid result with any primary criterion failing is `T1_CONFIRMATION_REJECT`;
transport, data-quality, governance, or implementation failure is not strategy
rejection.

No redraw, resample, successor selection, or post-T1 tuning is allowed. A T1 pass does not establish future profitability or authorize T2, T3, deployment, production, candidate modification, or another candidate.

## 5. Failure and authority boundary

```text
PLUMBING_FAILURE_RETRIABLE=public_transport_or_setup_before_first_complete_payload
DATA_QUALITY_FAILURE=locked_payload_or_source_semantic_failure
GOVERNANCE_FAILURE=missing_or_invalid_required_authority_or_receipt
IMPLEMENTATION_FAILURE=reviewed_code_or_durable_state_invariant_failure
STRATEGY_FAILURE=scientifically_valid_frozen_T1_evaluation_fails_primary_criteria
PROFITABILITY_FAILURE=scientifically_valid_result_only_when_the_frozen_result_so_classifies_it
```

Transport, private-state, or governance failure is never strategy or profitability failure. No classification authorizes T2/T3. This design supplies no human authorization string and consumes no gate. Its approval, independent review, and later support implementations are all required before any real execution.

```text
network_requests=0
private_reads=0
sealed_reads=0
partition_generation=false
gate_consumption=0
T1_ACCESS_AUTHORIZED=false
T1_CONSUMED=false
T2_AUTHORIZED=false
NEXT_ACTION=GPT_EXACT_SHA_V8K_T1_PARTITION_AUTHORITY_DESIGN_REVIEW
```
