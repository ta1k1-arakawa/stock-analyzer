# V13 V8 Partition Reconstruction Design

```text
document_type=V8_PARTITION_RECOVERY_DESIGN
status=DESIGN_COMPLETE_AWAITING_GPT_EXACT_SHA_REVIEW
issue=46
base_sha=b0c152013c8d5c4b6ddeb69cc572ba0e6c8d0df0
design_owner=GPT-5.6_SOL
implementation_authorized=false
reconstruction_executed=false
```

## 1. Objective and non-negotiable identity rule

Recover the original trusted V8 partition identity after loss of the private manifest. This is recovery，not creation of a new experimental partition. The trusted public anchors remain:

```text
ORIGINAL_MANIFEST_SHA256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
ORIGINAL_PARTITION_IMPLEMENTATION_GIT_COMMIT=36cbed941050e728f7f96ce2af505e81175cc02c
ORIGINAL_SCHEMA=V8_PARTITION_MANIFEST_V3
T1_SHA256=262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d
T2_SHA256=e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500
T3_SHA256=43a585f4c3341307e7c67561c54780322b0f253fefa628a7c6129773901a7b7a
T_SPARE_SHA256=360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70
```

A reconstruction that does not reproduce the recorded block hashes is not the original V8 partition and MUST be rejected. A generated manifest whose full canonical manifest SHA does not equal the original manifest SHA MUST NOT be represented as byte-exact recovery.

## 2. Mechanically recovered original construction contract

At the trusted implementation commit，`src/v8_partition.py` defines the complete partition algorithm. Eligible codes are normalized and ordered by `(SHA-256(UTF-8 code), code)` ascending. T0 is the first 300 entries and must reproduce the committed V4 universe exactly. The seven legacy-exposed tickers outside T0 are excluded. The remaining deterministic fresh pool is sliced without RNG:

```text
T1 = fresh_pool[0:300]
T2 = fresh_pool[300:600]
T3 = fresh_pool[600:900]
T_spare = fresh_pool[900:]
```

Therefore there is no random seed，PRNG state，or stochastic library dependency in block allocation.

The original accepted source snapshot is publicly fingerprinted as:

```text
source_url=https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls
source_acquisition_utc=2026-08-10T03:00:51.733526Z
source_raw_byte_count=830464
source_raw_sha256=6e401867d9ddf2524e4752f08fd3e3e434cd308c6d423839ca6e24fc7b1e1653
eligible_ticker_count=3110
eligible_ticker_list_sha256=37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405
t0_ticker_list_sha256=12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7
```

The original raw JPX bytes were explicitly not persisted. A future current JPX download is not presumed identical to those bytes and MUST NOT silently substitute for them.

## 3. Manifest byte-exactness analysis

The original manifest contains dynamic fields in addition to deterministic block membership，including `created_utc` and `source_acquisition_utc`. The builder computes `manifest_sha256` over canonical JSON of all manifest fields except `manifest_sha256` itself，using UTF-8，sorted keys，compact separators and one trailing LF.

`source_acquisition_utc` is known from public state. However，the original manifest's `created_utc` value is not recorded in the public provenance inspected for this recovery，and the original raw source bytes were not persisted. The raw source hash and length are known but do not permit recovery of the bytes themselves.

Consequently the original complete manifest bytes cannot currently be regenerated from repository-recorded provenance alone. The trusted full manifest SHA is an acceptance oracle，but it cannot supply the missing preimage fields.

## 4. Disposition

```text
RECONSTRUCTION_DISPOSITION=SEMANTIC_RECONSTRUCTION_ONLY_REQUIRES_HUMAN_DECISION
BYTE_EXACT_RECONSTRUCTION_DESIGN_READY=false
ORIGINAL_BLOCK_IDENTITY_RECONSTRUCTION_POTENTIALLY_VERIFIABLE=true
ORIGINAL_MANIFEST_BYTE_EXACT_RECONSTRUCTION_PROVEN=false
```

This disposition is deliberately fail-closed. Issue #45 prohibits silently substituting a new partition. Therefore no implementation may be authorized merely to fetch today's JPX file and rebuild a new partition.

## 5. What can be recovered safely

The block allocation itself is deterministic if and only if an eligible ticker universe can be obtained whose canonical ordered ticker-list SHA equals the original recorded value `37630f8f...63405`. Once that equality holds，the trusted implementation algorithm mechanically yields block assignments that can be independently checked against all four recorded T1/T2/T3/T_spare hashes.

Thus a future recovery can prove **semantic/original block identity** without recovering the original manifest bytes，provided all of these gates pass:

1. Use the exact partition logic from Git commit `36cbed941050e728f7f96ce2af505e81175cc02c` or a mechanically demonstrated equivalent implementation.
2. Reproduce T0 exactly against committed V4 provenance.
3. Require eligible ticker count `3110`.
4. Require eligible ticker-list SHA `37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405` before allocating fresh blocks.
5. Allocate with the exact deterministic no-RNG slicing contract above.
6. Require exact equality of T1，T2，T3 and T_spare ticker-list hashes to their recorded trusted hashes.
7. Reject on any mismatch. No partial acceptance，redraw，new snapshot semantics，or alternative ordering is allowed.

Passing these gates proves the recovered block assignments have the same canonical ticker-list identities as the original trusted V8 partition. It does **not** prove the replacement manifest bytes equal the lost manifest.

## 6. Human methodology gate required before implementation

Because Issue #45 requires byte-exact recovery as the target and prohibits a semantic substitute without an explicit later human methodology decision，implementation remains blocked.

The required human decision is narrowly:

> Permit recovery of the original V8 partition by exact block-identity reproduction，where every original block's canonical ticker-list SHA must match the already-pinned public hashes，while acknowledging that the replacement recovery manifest will have a new schema/provenance identity and MUST NOT claim the lost `V8_PARTITION_MANIFEST_V3` full-manifest SHA.

If the human does not approve this change，recovery remains blocked unless the original manifest or enough missing provenance to reproduce its exact bytes is recovered.

## 7. Design for the later implementation if the human approves

A later frozen amendment SHALL define a new recovery artifact，not counterfeit the lost V3 manifest. Recommended schema name:

```text
V8_PARTITION_RECOVERY_MANIFEST_V1
```

It SHALL contain，at minimum，the original trusted manifest SHA，original implementation commit，original source fingerprints，eligible-universe hash，all original block hashes，the recovered block assignments，recovery implementation SHA，recovery timestamp，and explicit fields:

```text
original_manifest_byte_exact_recovered=false
original_partition_block_identity_recovered=true
```

The implementation SHALL be mechanical enough for `CHEAP_CODEX_AGENT_OK` after amendment freeze. It SHALL have a repository-only/synthetic test mode and a separately authorized real-source execution mode. Any real JPX network request remains a separate human-gated execution task.

The real recovery path SHALL NOT publish an artifact until every eligible-universe and block-hash gate has passed. A current JPX source whose canonical eligible ticker-list hash differs from the original must terminate with a recovery-source mismatch and produce no accepted recovery manifest.

## 8. Acceptance tests for later implementation

Synthetic/repository-safe tests SHALL cover:

- exact deterministic ordering and no RNG dependency;
- T0 reproduction gate;
- legacy exclusion;
- eligible-universe hash mismatch fails before accepted allocation;
- each individual block hash mismatch fails closed;
- exact block-size and disjointness invariants;
- recovery artifact cannot claim byte-exact original manifest recovery;
- write-once/safe publication behavior;
- no network in synthetic mode.

Real execution acceptance SHALL require all trusted hashes to match and shall expose no sealed identities to public repository output or terminal logs beyond the already authorized safe reporting contract.

## 9. Authority boundaries

This document performs no reconstruction and creates no network，private-data，model-fit，backtest，or trading authority. It does not authorize implementation because the required semantic-recovery methodology decision has not yet been granted.

```text
PRIVATE_READ_AUTHORIZATION_CONSUMED=false
PRIVATE_CONTENT_READS=0
PRIVATE_PATH_DISCOVERY=0
NETWORK_REQUESTS_MARKET_DATA=0
MODEL_FITS=0
BACKTESTS=0
RECONSTRUCTION_EXECUTED=false
REPLACEMENT_PARTITION_ACCEPTED=false
IMPLEMENTATION_AUTHORIZED=false
```
