# V8H Fresh Private Partition Successor Design Draft

This is a successor-study design draft only. It creates no implementation
authority, network authority, private-data authority, partition-generation
authority, allocation authority, raw-acquisition authority, human
authorization, gate receipt, or research-opening authority.

## 1. Study identity and predecessor disposition

```text
study=V8H_HISTORICAL_RESEARCH
study_type=SUCCESSOR_STUDY
predecessor=V8G_HISTORICAL_RESEARCH
predecessor_terminal_commit=5f0d7eff2df1728abcf58b7eddd16329a9010f8e
predecessor_terminal_artifact=V8G_PRIVATE_PARTITION_LOCATOR_TERMINAL_ADJUDICATION.json
predecessor_terminal_artifact_blob=13771d6dd5cf5d9a8445b3d7123d1cb7f5118a2e
predecessor_terminal_stage=V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT
predecessor_reviewed_v8g_design_candidate_commit=b9c7014ba72b72efadb1a4be6c5aa4aa71201518
predecessor_reviewed_locator_support_implementation_sha=928ab622e0a3d1cd34f021e0248bf725d9cf2e66
predecessor_execution_result=BLOCK
predecessor_failure_class=PRIVATE_PARTITION_LOCATOR_PROVENANCE_FAILURE
predecessor_failure_reason=V8G_LOCATOR_ZERO_MATCHING_CANDIDATES
predecessor_disposition=BLOCK_CLOSED
predecessor_gate_consumed=true
predecessor_authorization_reusable=false
predecessor_second_execution_allowed=false
predecessor_successor_study_decision_required=true
```

V8G is, and permanently remains, `BLOCK_CLOSED`. Its one-shot
`HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE` is durably
consumed; per its own frozen contract
(`V8G_PRIVATE_PARTITION_LOCATOR_SUCCESSOR_DESIGN_DRAFT.md` §2.1.3) that
consumption is permanent for the life of the V8G study — no retry, reset,
delete, reuse, candidate substitution, or alternate-root search is ever
permitted under the V8G authorization, by any actor, under any
circumstance. `V8G_LOCATOR_EXECUTION_INCIDENTS.md` records that the
locator's operational wrapper incidents were all `PRE_GATE` and did not
consume authorization, and that the terminal `BLOCK` itself was the
scientific V8G result, not an operational error.

This draft does not reopen, retry, amend, or continue V8G under any name.
V8H is a new, independent study identity. It does not search again for the
historical V8/V8B–V8G private partition (the one whose manifest hash is
`0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62`, pinned
at `V8_TRUSTED_PARTITION.json`), does not broaden the V8G locator's root
scope, does not substitute or add candidates to any prior enumeration,
does not reset or reuse the V8G locator gate receipt, and does not reuse
any V8G human authorization for any V8H purpose. No V8G identifier —
gate, receipt, authorization identity, candidate hash, or artifact —
authorizes any V8H stage.

No historical holdout or private membership from V8, V8B, V8C, V8D, V8E,
V8F, or V8G may be treated as V8H authority, may seed V8H partition
generation, or may be read, referenced, or compared against during V8H
generation. The historical partition manifest hash and implementation
commit above are cited here purely as inert historical identifiers, to
state unambiguously what V8H is *not* attempting to recover.

## 2. Rationale: provenance/availability failure, not strategy or profitability failure

```text
V8G_failure_class=PRIVATE_PARTITION_LOCATOR_PROVENANCE_FAILURE
V8G_strategy_failure=false
V8G_profitability_failure=false
V8G_future_profitability_established=false
V8H_addresses_provenance_availability_only=true
V8H_does_not_address_strategy_or_profitability=true
```

The V8G terminal result states only that a metadata-only, content-addressed
candidate enumeration for that one authorized execution contained zero
files whose independently recomputed canonical hash matched the
historically authorized partition-manifest identity. It is silent on, and
establishes nothing about:

- whether any strategy, labeling scheme, cost model, or promotion rule
  performs well or poorly;
- whether the underlying market data, had it been located, would have
  passed any data-quality gate;
- readiness, transport, or acquisition semantics for any partition;
- future profitability of any strategy, historical or prospective.

This is a **provenance/availability failure**: the historical artifact
that would have been read was, at the moment of the one authorized V8G
execution, not discoverable at any location the frozen locator contract
was permitted to search. It is categorically distinct from a **strategy
failure** (a hypothesis or rule underperforms once evaluated) or a
**profitability failure** (a strategy is evaluated and found unprofitable
under the frozen evaluation contract). Because V8/V8B–V8G never reached a
private-data read that could produce either of those latter outcomes, no
such outcome exists for V8H to inherit, avoid, or correct for. V8H's
existence is justified exclusively by the unavailability finding above; it
carries forward no scientific evidence about strategy quality or
profitability, favorable or unfavorable, from any predecessor study.

## 3. Single change relative to V8G

```text
V8H_single_change=ABANDON_HISTORICAL_MANIFEST_RECOVERY_AND_GENERATE_FRESH_PARTITION
```

V8H changes exactly one methodological element relative to the
V8/V8B–V8G lineage: it abandons any attempt to recover, locate, or
reconstruct the historical private partition manifest referenced in §1,
and instead establishes an entirely new, freshly generated private
partition under a newly preregistered `V8H` partition-generation
contract. No other element of the inherited methodology changes (§4).

This is a strictly narrower and more conservative change than it may
first appear: it does not relax any acceptance criterion, does not permit
any manual or outcome-informed membership choice, and does not create any
new route by which a previously blocked V8G attempt could be revived under
a different name. It creates a wholly new partition identity, generated
under its own frozen, preregistered generation contract (§6), with its
own fresh one-shot human gates (§7), fully independent of every V8G
locator artifact, receipt, or authorization.

## 4. Unchanged inherited methodology

V8H inherits the V8/V8B/V8C/V8D/V8E/V8F/V8G methodology unchanged except
for the single change in §3. In particular, this task does not change,
and no future V8H stage may silently change:

- the evaluation period or research-selection cutoff;
- labels or target definitions;
- the strategy, hypothesis, or candidate-rule family;
- transaction costs, slippage, or portfolio rules;
- the promotion threshold or any acceptance criterion for a strategy;
- the search space or grid;
- the stopping rule;
- any profitability criterion;
- the Yahoo provider/host, `interval=1d`, request headers, `events`,
  `includeAdjustedClose`;
- readiness sentinels `[0, 149, 299]`, sentinel count 3;
- the readiness window, exactly `2025-12-01` to `2025-12-08` (exclusive);
- maximum attempts=3; maximum retries=2; backoff=`[5, 30]`; jitter=false;
  the retry classifier;
- the V8B DQ evidence policy and its exact thresholds
  (`POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE`,
  `invalid_fraction_threshold=1/252`,
  `max_consecutive_invalid_returned_rows=1`, `full_P_hist_check=true`,
  `test_years=2018..2025`,
  `calendar_missing_dates_are_not_malformed_returned_rows=true`,
  `threshold_failure_action=BLOCK_WHOLE_ACQUISITION`);
- research-opening rules other than the fresh gates this draft inserts
  before any V8H membership is revealed (§7).

No threshold recalibration, provider substitution, retry-policy change,
evaluation-window change, or stopping-rule change is authorized or implied
by this draft.

## 5. Preregistration-before-generation ordering

```text
ordering_invariant=PREREGISTRATION_STRICTLY_PRECEDES_GENERATION
ordering_invariant=GENERATION_STRICTLY_PRECEDES_MEMBERSHIP_DISCLOSURE
ordering_invariant=MEMBERSHIP_DISCLOSURE_STRICTLY_PRECEDES_RESEARCH_OPENING
```

The V8H partition-generation contract (§6) — its generation mechanism,
source-snapshot identity, deterministic/randomization commitment,
allocation rules, `T0`/`T1`/`T2`/`T3`/`T_spare` semantics, hashing scheme,
preservation rules, and all acceptance/failure semantics — must be fully
frozen, independently reviewed, and human-authorized *before* any private
partition membership is generated. Partition membership must in turn be
fully generated, hashed, and sealed *before* any human, ChatGPT prompt,
Claude/Codex execution, or downstream artifact is permitted to observe
which specific identities were assigned to which tier. Membership must
remain sealed *before* any research-opening gate for V8H is even
approached.

Reordering any of the above — generating before the contract is frozen,
revealing membership before generation is sealed, or opening research
before membership is sealed — is prohibited under all circumstances and
is not a decision this draft delegates to any execution agent.

## 6. Partition-generation commitment requirements (frozen: `V8H_PRIVATE_PARTITION_GENERATION_V1`)

```text
generation_uses_outcomes=false
generation_uses_prices_features_or_returns=false
generation_uses_manual_selection=false
generation_uses_performance_based_inclusion_or_exclusion=false
generation_is_fixed_before_any_private_byte_is_read=true
generation_contract=V8H_PRIVATE_PARTITION_GENERATION_V1
```

The new V8H partition must be generated by a mechanism that is entirely
free of outcome-based or manual membership selection: no candidate
identity may be included in, excluded from, or reassigned between tiers
because of its observed or anticipated price behavior, strategy
performance, or any other outcome-linked signal, at generation time or
subsequently. Generation must be either fully deterministic from a frozen,
publicly committed rule applied to a frozen source snapshot, or
randomized under a frozen, publicly pre-committed randomization scheme —
never a mechanism whose outcome could be steered, tuned, or retried by
any party after seeing a preliminary result.

The elements below are now frozen in full by explicit ChatGPT methodology
decision (task `V8H_PARTITION_GENERATION_CONTRACT_COMPLETION`). No value
in this section is invented by an execution agent; each is the binding
decision supplied by that task. Once frozen, this contract is subject to
the same staleness discipline V8G established for its own frozen contract
(V8G draft §2.3's `design_candidate_staleness_rule`): any later amendment
invalidates every authorization granted against the prior version, and
requires a fresh independent review before it can authorize anything.

### 6.1 Source universe (inherited from V8, unchanged)

V8H does not invent a new source universe. It reuses, unchanged:

- the official JPX implementation-time snapshot semantics already frozen
  by V8 (`src/v8_partition.py`'s `SOURCE_SNAPSHOT_SEMANTICS =
  "IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT"`,
  `SOURCE_SNAPSHOT_CLARIFICATION_COMMIT =
  266999a8e48c77905dd7c7312fd41c7f38241d78`): the source artifact is
  whichever official JPX snapshot is fetched at the moment of the one
  authorized V8H source-snapshot acquisition (§7.1) — there is no
  separate fixed calendar as-of date to preregister beyond that;
- the exact V8 eligible-universe selection/reconstruction semantics
  already implemented in `src/v8_partition.py`
  (`parse_eligible_universe`, `canonical_order`,
  `build_universe_csv_bytes`), unmodified;
- exact `T0` reproduction against the existing frozen V4/T0 evidence
  (`V4_UNIVERSE.csv` / `V4_UNIVERSE_MANIFEST.json`,
  `verify_t0_reproduction`) remains mandatory: if the reconstructed
  universe's first 300 tickers do not byte-reproduce `V4_UNIVERSE.csv`
  exactly, generation `BLOCK`s (`V8_T0_REPRODUCTION_MISMATCH`) before any
  fresh allocation is attempted;
- V4 raw-byte equality remains NOT required
  (`V4_RAW_SHA_EQUALITY_REQUIRED = False`), consistent with V8: the fresh
  snapshot's own raw bytes are never compared for equality against V4's
  original 2026-08-03 raw bytes, only recorded as an audit reference.

At the one authorized V8H source-snapshot acquisition, the following
execution-provenance values must be recorded: raw source bytes (held
privately, never published), acquisition UTC, raw SHA-256, eligible
ticker count, and eligible ticker-list SHA-256. These are runtime
provenance values, not methodological choices — they become immutable
the instant the one authorized snapshot acquisition completes (§7.1,
§10). No source substitution, second snapshot, fallback provider, or
favorable-date substitution is permitted after that snapshot authority is
consumed.

### 6.2 Fresh eligible set

V8 block meaning is unchanged:

```text
T0 = existing exposed 300
T1 = fresh validation 300
T2 = fresh sealed holdout 300
T3 = fresh sealed reserve 300
T_spare = remainder
```

The fresh allocation universe for V8H is:

```text
fresh_allocation_universe =
    eligible_current_only - T0 - every known legacy/outcome-exposed
    ticker outside T0
```

using the exact legacy-exclusion semantics already encoded in
`src/v8_partition.py` (`LEGACY_EXPOSED_TICKERS_OUTSIDE_T0`, the seven
codes `1570`, `4689`, `5020`, `7211`, `7267`, `8306`, `9432`), unchanged.
`T0` itself is the same already-publicly-reproducible 300 (`V4_UNIVERSE.
csv`, reproduced per §6.1) — not sealed, and not V8/V8B–V8G private
membership. Because the historical V8/V8B–V8G partition manifest was
never located or read (§1, §2), its `T1`/`T2`/`T3`/`T_spare` membership
is unknown to this repository and cannot be, and is not, referenced by
this exclusion rule; the only membership V8H's generation universe
excludes is the publicly reproducible `T0` and the seven publicly known
legacy-exposed codes.

At least 900 eligible fresh tickers (`3 × 300`, matching the unchanged
block size) must remain after exclusions. If fewer than 900 remain,
generation cannot create `T1`/`T2`/`T3` and must `BLOCK` — this mirrors
the existing `V8_ELIGIBLE_POOL_INSUFFICIENT` check
(`len(fresh_pool) < block_size * 3`) already implemented in
`src/v8_partition.py`. Block size remains fixed at 300; this draft does
not change it.

Inherited tier semantics (`V8_HISTORICAL_RESEARCH_DESIGN.md` §§1.4–1.6,
unchanged, restated here for V8H only as an unambiguous cross-reference,
not as a new decision):

- `T1`: fresh validation block; `max_validation_access = 1` — the access
  count becomes 1 the moment any result is seen, however partially; the
  shortlist and every parameter must be frozen before `T1` is opened;
- `T2`: fresh sealed holdout for exactly one `FROZEN_FINAL_CANDIDATE` —
  comparing or selecting among multiple candidates on `T2` is prohibited;
- `T3`: fresh sealed reserve, `SEALED_RESERVE_NOT_USED_IN_INITIAL_V8H`
  (mirroring V8's own `T3_PRICE_ACQUISITION_AUTHORIZED = False`);
- `T_spare`: remainder, unallocated, available only under a future
  separately authorized successor/reallocation gate (§7.4).

### 6.3 Randomization and commitment: secret CSPRNG seed

The old public deterministic `SHA256(code)` ordering V8 used for its
original partition is explicitly NOT reused for V8H fresh membership,
because a public universe plus a public deterministic rule would make
sealed membership derivable by anyone. V8H instead freezes:

```text
partition_seed_bytes = exactly 32 bytes from the operating system
                        cryptographic RNG
```

The production implementation must use the OS CSPRNG equivalent of
Python's `os.urandom(32)`. No PRNG seeded with a user-selected integer is
permitted under any circumstance.

The seed is generated exactly once, only **after** the
`HUMAN_V8H_PRIVATE_PARTITION_GENERATION_GATE` durable receipt (§7.2) has
been successfully published. It must be immediately persisted in
canonical machine-local **private** durable state (§12) using exclusive,
no-overwrite durable publication. The raw seed must never enter Git,
logs, stdout, public artifacts, exception messages, or prompts, at any
point, under any failure mode.

Public commitment:

```text
partition_seed_sha256 = SHA256(partition_seed_bytes)
```

Only `partition_seed_sha256` — never `partition_seed_bytes` — is safe
public evidence (§11).

### 6.4 Deterministic allocation from the secret seed

For each canonical ticker code in the validated fresh eligible set
(§6.2), compute:

```text
allocation_key =
  HMAC-SHA256(
    key=partition_seed_bytes,
    msg=UTF8("V8H_PARTITION_ASSIGN_V1\0" + ticker_code)
  )
```

Sort all fresh eligible ticker codes ascending by
`(allocation_key hexadecimal bytes/value, ticker_code)`.

Any duplicate canonical ticker, or any duplicate `allocation_key`, is a
fail-closed `BLOCK` — never silently deduplicated after the eligible-set
construction of §6.2 has already been validated (that construction
already rejects duplicate tickers; a duplicate surviving to this step, or
a duplicate `allocation_key` collision, is itself a schema/invariant
violation).

Allocate from the sorted order:

```text
first 300  -> T1
next  300  -> T2
next  300  -> T3
remainder  -> T_spare
```

No reroll, reshuffle, seed replacement, manual adjustment, industry
balancing, performance balancing, or membership substitution is
permitted, under any circumstance, at any point after the seed is
generated.

### 6.5 Canonical ticker/list hashing (inherited from V8, unchanged)

V8 canonical ticker representation and ticker-list hashing are reused
wherever applicable: canonical ticker normalization is the existing
`str(code).strip().upper()` form already applied by
`src/v8_partition.py`'s eligible-set construction and `canonical_order`.
This inherited normalization has been checked against V8H's needs and is
sufficient; it is not altered, and no new normalization is invented.

For each block, the inherited newline-list digest semantics are
preserved exactly:

```text
SHA256(UTF8("\n".join(tickers) + "\n"))
```

— i.e., `src/v8_partition.py`'s existing `ticker_list_sha256` /
`_ticker_list_sha`, reused unchanged. The order hashed for each block is
its frozen allocation order from §6.4 (the HMAC-sorted cut order), never
a later human-selected or re-sorted ordering.

### 6.6 Manifest canonicalization: `V8H_PARTITION_MANIFEST_V1`

A new schema namespace is used: conceptually
`V8H_PARTITION_MANIFEST_V1`, distinct from V8's own
`V8_PARTITION_MANIFEST_V3`. V8 canonical JSON semantics are reused
exactly:

```text
json.dumps(..., ensure_ascii=False, sort_keys=True,
           separators=(",", ":"), allow_nan=False) + "\n"
```

— i.e., `src/v8_partition.py`'s existing `canonical_json_bytes`, reused
unchanged. The manifest self-hash is SHA-256 over the canonical manifest
object excluding only its own `manifest_sha256` field, following the
existing V8 `canonical_sha256` pattern exactly.

Two distinct manifest objects exist, and are never merged:

- the **PRIVATE** manifest, which may contain the actual block
  assignments and the raw partition seed (or an exact private reference
  sufficient for deterministic reproduction) — neither may appear in any
  public evidence, ever;
- the **PUBLIC** generation evidence, which may contain only safe
  hashes/counts/booleans/enums/Git provenance (§11).

### 6.7 Acceptance and failure semantics

```text
PRE_GATE_BLOCK_CONDITIONS (before HUMAN_V8H_PRIVATE_PARTITION_GENERATION_GATE
consumption; gate remains unconsumed where mechanically possible, per
AI_REAL_EXECUTION_RUNBOOK.md):
  - Git/provenance mismatch
  - source-snapshot artifact mismatch
  - generation-support implementation mismatch
  - dirty working tree
  - malformed authorization (wrong grammar, wrong length, wrong case,
    non-hex character)
  - already-consumed generation-gate receipt
  - insufficient fresh eligible count (< 900, §6.2)
  - missing prerequisite artifact (e.g. no PASS'd source-snapshot
    artifact)

POST_GATE_BLOCK_CLOSED_CONDITIONS (after HUMAN_V8H_PRIVATE_PARTITION_
GENERATION_GATE durable receipt publication; ANY failure here is
permanent):
  - CSPRNG failure
  - seed-persistence failure
  - allocation failure (duplicate ticker/key, wrong block sizes, overlap)
  - manifest-publication failure
  - hash mismatch (self-hash, ticker-list hash, or binding hash)
  - preservation failure

no_second_seed=true
no_retry=true
no_reset=true
no_receipt_deletion=true
no_regeneration=true
no_membership_reassignment=true
no_candidate_substitution=true
no_same_study_repair_requiring_another_generation=true
successor_study_decision_required=true
```

A `PRE_GATE` failure never consumes the generation gate. A `POST_GATE`
failure of any kind — including operational failures unrelated to the
scientific mechanism, such as a CSPRNG or preservation failure — is
`BLOCK_CLOSED` for the entire V8H study; it is never treated as a
retryable operational incident, mirroring V8G's own
`V8G_LOCATOR_EXECUTION_INCIDENTS.md` PRE_GATE/POST_GATE separation but
applied more strictly here: after this gate, there is no PRE_GATE/
POST_GATE distinction left to draw, because everything past this point is
scientific generation, not enumeration.

## 7. Fresh human-gate requirements

```text
V8H_gates_are_fresh=true
V8H_gates_reuse_no_V8G_receipt=true
V8H_gates_reuse_no_V8G_authorization=true
```

Source acquisition and partition generation are separate stages, each
with its own fresh one-shot gate, independent of every V8G gate, receipt,
and authorization, and independent of each other. Every V8H gate is
one-shot, durably receipted outside this repository per the existing
canonical durable-state pattern (`CANONICAL_CONSUMPTION_STATE_ROOT`), and
bound to its own exact reviewed design-candidate commit (prefreeze) or
frozen design commit (postfreeze), following the same stage-aware
prefreeze/postfreeze design-candidate binding discipline V8G froze in its
own §2.3, applied here under the V8H namespace. No V8H gate may be
consumed using a V8G receipt, a V8G authorization identity, or any
V8G-derived binding value, and no fixed receipt key below reuses any V8G
receipt-key literal.

**Implementation boundary.** Mirroring V8G's own two-module split (V8G
draft §2.2), source-snapshot acquisition and partition generation each
require their own fresh, separately reviewed, V8H-namespaced support
module — a `reviewed_source_snapshot_support_implementation_sha` and a
distinct `reviewed_generation_support_implementation_sha`. These two
implementation-commit bindings are never merged, renamed, or compared to
each other. Neither module may edit `src/v8_partition.py` in place; that
module remains valid, unmodified V8 evidence and is only imported/reused
(§6.1, §6.5) by the new V8H-namespaced modules, never renamed to V8H.

### 7.1 V8H source-snapshot acquisition gate

```text
gate=HUMAN_V8H_SOURCE_SNAPSHOT_ACQUISITION_GATE
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST
```

This gate's one-shot authorization permits only the one V8H
source-snapshot acquisition and its T0-reproduction validation (§6.1). It
does **not** authorize partition generation, membership disclosure, raw
historical price acquisition, or research opening. A successful
source-snapshot stage freezes its exact bytes' hash and the eligible
ticker-list hash for later use by V8H generation (§6.1, §7.2); those
values become immutable the instant this gate's post-gate work
completes. Pre-network failures are handled only per
`AI_REAL_EXECUTION_RUNBOOK.md`. Once this gate's one-shot network
authority is consumed, no silent substitution of another snapshot,
provider, or date is permitted, under any circumstance.

Binding (mechanically re-verified before consumption, mismatch on any
field is `PRE_GATE BLOCK`):

```text
source_snapshot_gate_authority_binding:
  repository                                     (= "ta1k1-arakawa/stock-analyzer")
  study                                           (= "V8H_HISTORICAL_RESEARCH")
  gate                                            (= "HUMAN_V8H_SOURCE_SNAPSHOT_ACQUISITION_GATE")
  reviewed_v8h_design_candidate_commit            (this design's own reviewed candidate, §13)
  reviewed_source_snapshot_support_implementation_sha
```

**Human authorization grammar.** A single deterministic string:

```text
authorization_identity =
    "V8H_HUMAN_AUTHORIZE_SOURCE_SNAPSHOT_ACQUISITION_AT_"
  + reviewed_v8h_design_candidate_commit
  + "_WITH_"
  + reviewed_source_snapshot_support_implementation_sha
```

Both components are exactly 40 lowercase hex characters (a full Git
commit object id). Any component of the wrong length, wrong case, or
containing a non-hex character is a grammar mismatch and a `PRE_GATE
BLOCK`, never post-gate, never coerced. Only
`authorization_identity_sha256 = SHA256(UTF8(authorization_identity))`
may ever be persisted or appear in a receipt or safe result; the raw
string is never printed, logged, or persisted.

**Deterministic one-shot receipt key**, fixed the moment repository,
study, and gate name are fixed — excluding every attempt-varying value,
so a second attempt cannot unlock this gate under any authorization,
candidate, or implementation:

```text
receipt_key_material =
    "V8H_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_KEY_V1\0"
  + "ta1k1-arakawa/stock-analyzer"
  + "\0"
  + "V8H_HISTORICAL_RESEARCH"
  + "\0"
  + "HUMAN_V8H_SOURCE_SNAPSHOT_ACQUISITION_GATE"

source_snapshot_gate_receipt_key_sha256 = SHA256(UTF8(receipt_key_material))
```

This gate may be durably consumed at most once for the entire life of the
V8H study. An existing receipt at this fixed key is a `PRE_GATE BLOCK`
regardless of which authorization/candidate/implementation is presented
afterward; a malformed existing receipt is also `BLOCK` and is never
deleted, repaired, or replaced.

The receipt body (not the key) must record, at minimum: `schema_version`,
`artifact_role`, `study`, `gate`, `reviewed_v8h_design_candidate_commit`,
`reviewed_source_snapshot_support_implementation_sha`,
`authorization_identity_sha256`, `source_raw_sha256`,
`source_acquisition_utc`, `eligible_ticker_count`,
`eligible_ticker_list_sha256`, `t0_reproduction_status`, `consumed=true`,
`consumption_count=1`, `consumption_boundary`, `consumption_timestamp_utc`
— the same class of safe Git-commit/hash/enum/boolean/timestamp values
V8's and V8G's existing receipts already safely persist. No raw ticker
identity, private path, or raw manifest byte is ever a field here.

### 7.2 V8H partition-generation gate

```text
gate=HUMAN_V8H_PRIVATE_PARTITION_GENERATION_GATE
consumption_boundary=IMMEDIATELY_BEFORE_PARTITION_SEED_GENERATION
```

This gate's one-shot authorization permits only the one V8H partition
generation (§6.3–§6.4: CSPRNG seed generation, HMAC-based allocation, and
manifest construction/preservation). It does not authorize membership
disclosure, raw historical price acquisition, `T1`/`T2`/`T3` opening, or
research opening (§11).

Binding (mechanically re-verified before consumption, mismatch on any
field is `PRE_GATE BLOCK`):

```text
generation_gate_authority_binding:
  repository                                      (= "ta1k1-arakawa/stock-analyzer")
  study                                            (= "V8H_HISTORICAL_RESEARCH")
  gate                                             (= "HUMAN_V8H_PRIVATE_PARTITION_GENERATION_GATE")
  reviewed_v8h_design_candidate_commit
  reviewed_generation_support_implementation_sha
  source_snapshot_artifact_self_sha256             (the frozen §7.1 artifact's own self-hash)
  eligible_ticker_list_sha256                      (frozen at §7.1, reused unchanged)
```

**Human authorization grammar.**

```text
authorization_identity =
    "V8H_HUMAN_AUTHORIZE_PRIVATE_PARTITION_GENERATION_AT_"
  + reviewed_v8h_design_candidate_commit
  + "_WITH_"
  + reviewed_generation_support_implementation_sha
  + "_FOR_SNAPSHOT_"
  + source_snapshot_artifact_self_sha256
  + "_ELIGIBLE_"
  + eligible_ticker_list_sha256
```

`reviewed_v8h_design_candidate_commit`,
`reviewed_generation_support_implementation_sha`, and
`source_snapshot_artifact_self_sha256` are each exactly 40 or 64
lowercase hex characters as applicable to a Git commit id or a SHA-256
digest respectively; `eligible_ticker_list_sha256` is exactly 64
lowercase hex characters. Any component of the wrong length, wrong case,
or containing a non-hex character is a grammar mismatch and a `PRE_GATE
BLOCK`. Only `authorization_identity_sha256` is ever persisted; the raw
string is never printed, logged, or persisted.

**Deterministic one-shot receipt key**, deliberately excluding every
attempt-varying value (the authorization identity/hash, the reviewed
design candidate commit, the generation-support implementation SHA, the
source-snapshot artifact hash, and the eligible ticker-list hash) so that
no combination of a fresh authorization, a newly reviewed candidate, or a
newly reviewed implementation can ever unlock a second generation attempt:

```text
receipt_key_material =
    "V8H_PRIVATE_PARTITION_GENERATION_GATE_RECEIPT_KEY_V1\0"
  + "ta1k1-arakawa/stock-analyzer"
  + "\0"
  + "V8H_HISTORICAL_RESEARCH"
  + "\0"
  + "HUMAN_V8H_PRIVATE_PARTITION_GENERATION_GATE"

generation_gate_receipt_key_sha256 = SHA256(UTF8(receipt_key_material))
```

This gate may be durably consumed at most once for the entire life of the
V8H study. An existing receipt at this fixed key is a `PRE_GATE BLOCK`
regardless of which authorization/candidate/implementation is presented
afterward; a malformed existing receipt is also `BLOCK` and is never
deleted, repaired, or replaced. After this receipt is durably published,
the seed is generated (§6.3); any failure from that instant onward is
`POST_GATE BLOCK_CLOSED` per §6.7, permanently, for the entire V8H study.

The receipt body (not the key) must record, at minimum: `schema_version`,
`artifact_role`, `study`, `gate`,
`reviewed_v8h_design_candidate_commit`,
`reviewed_generation_support_implementation_sha`,
`source_snapshot_artifact_self_sha256`, `eligible_ticker_list_sha256`,
`authorization_identity_sha256`, `consumed=true`, `consumption_count=1`,
`consumption_boundary`, `consumption_timestamp_utc`. It never records the
raw seed, the private manifest, or any ticker identity or block
assignment.

### 7.3 Later gates (deferred, not resolved by this task)

A separate gate before any private membership is disclosed to any actor
or artifact beyond what §11 permits as public-safe evidence, and a
separate gate before any V8H research-opening stage (strictly after
membership is sealed and preserved (§12) and after the independent
exact-SHA reviews required by §13), remain required. Their exact names,
receipt-key material, and authorization grammar are not resolved by this
task and remain `CHATGPT_DECISION_REQUIRED` (§15) — generation PASS does
not itself authorize either of them (§11).

### 7.4 Generation PASS grants no further authority

```text
generation_pass_authorizes_membership_disclosure=false
generation_pass_authorizes_historical_price_acquisition=false
generation_pass_authorizes_T1_opening=false
generation_pass_authorizes_T2_opening=false
generation_pass_authorizes_T3_opening=false
generation_pass_authorizes_research_opening=false
generation_pass_authorizes_production_trading=false
```

A `HUMAN_V8H_PRIVATE_PARTITION_GENERATION_GATE` PASS establishes only
that a fresh partition was generated under this frozen contract (§14). It
does not, by itself, authorize membership disclosure, historical-price
acquisition, `T1`/`T2`/`T3` opening, research opening, or production
trading. Each of those keeps its own separate, fresh authority boundary,
consumed only under its own future gate (§7.3). Inherited V8 access
budgets apply unchanged once opened: `T1` retains `max_validation_access
= 1`; `T2` remains a sealed holdout for exactly one frozen final
candidate; `T3` remains sealed reserve; `T_spare` remains remainder,
available only under a future, separately authorized successor/
reallocation gate already compatible with the inherited V8 rules cited in
§6.2.

## 8. One-shot / fail-closed semantics

```text
one_shot=true
retry_allowed=false
reset_allowed=false
delete_allowed=false
reuse_allowed=false
candidate_substitution_allowed=false
alternate_root_allowed=false
membership_reassignment_after_generation_allowed=false
```

Every V8H gate defined under this draft or a future V8H amendment is
one-shot in the same sense V8G froze for its own locator gate: a durable
receipt at a fixed key blocks any second attempt for the life of the V8H
study, regardless of which authorization, reviewed candidate, or
implementation is presented afterward. A post-gate failure of any kind is
permanent for that authorization; it does not restore authorization,
does not permit a retry, reset, deletion, or reuse of the same gate, and
does not permit membership to be regenerated, reassigned, or
supplemented after generation is sealed. Any repair to a failed V8H stage
that would change methodology or the execution contract requires its own
successor-study decision, exactly as `AI_REAL_EXECUTION_RUNBOOK.md` §10
already requires generically.

## 9. Leakage and post-outcome-selection prohibitions

```text
post_outcome_selection_prohibited=true
leakage_into_generation_prohibited=true
generation_may_not_read_prices_features_or_labels=true
generation_may_not_read_strategy_outputs=true
membership_may_not_be_adjusted_after_any_evaluation=true
```

No step of V8H partition generation may read, or be informed directly or
indirectly by, price data, engineered features, labels, strategy outputs,
backtest results, or any other outcome-linked signal for any candidate
identity. No V8H tier's membership may be adjusted, resampled, or
selectively regenerated after any evaluation — partial, preliminary, or
final — has been observed for any candidate. A generation process that
could be rerun, tuned, or steered based on a downstream result is not a
valid instantiation of the frozen contract in §6, regardless of how the
mechanism is otherwise described. This prohibition applies for the entire
life of the V8H study, including any future amendment: an amendment made
after any evaluation has been observed must itself be treated as
outcome-informed and is prohibited from retroactively altering already-
sealed membership.

## 10. Source-snapshot immutability requirement

```text
source_snapshot_must_be_frozen_before_generation=true
source_snapshot_identity_must_be_content_addressed=true
source_snapshot_may_not_be_substituted_after_freeze=true
```

The source snapshot from which V8H membership candidates are drawn must
be frozen and content-addressed strictly before generation begins,
exactly as the frozen historical `manifest_sha256` anchor pattern already
established for the V8/V8B–V8G partition (§1). Once frozen, the source
snapshot may never be substituted, extended, re-scraped, or re-scoped for
the purposes of this generation — any change to the source snapshot after
it is frozen requires the same successor-study-level decision as any
other frozen-contract amendment (§6, §9).

The source snapshot's identity is now resolved (§6.1, §7.1): it is
whichever official JPX implementation-time snapshot is fetched at the one
authorized `HUMAN_V8H_SOURCE_SNAPSHOT_ACQUISITION_GATE` execution, and it
is content-addressed by `source_raw_sha256` (SHA-256 of its raw bytes),
frozen and immutable from the instant that one authorized acquisition
completes. There is no separate fixed calendar as-of date to preregister
beyond that instant — this mirrors V8's own
`IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT` semantics exactly, and is not
a new decision.

## 11. Membership confidentiality / public-safe evidence rules

```text
ticker_identities_exposed=false
block_assignments_exposed=false
raw_seed_exposed=false
research_opened=false
raw_or_private_payload_persisted_publicly=false
```

The set of safe public V8H generation evidence is now resolved in full
(task `V8H_PARTITION_GENERATION_CONTRACT_COMPLETION`). Public V8H
generation, gate, and evidence artifacts may expose only, at minimum:

```text
source_raw_sha256
eligible_ticker_list_sha256
eligible_ticker_count
t0_ticker_count / t1_ticker_count / t2_ticker_count / t3_ticker_count / t_spare_ticker_count
t0_ticker_list_sha256 / t1_ticker_list_sha256 / t2_ticker_list_sha256 /
  t3_ticker_list_sha256 / t_spare_ticker_list_sha256
partition_seed_sha256
private_manifest_sha256
reviewed_v8h_design_candidate_commit
reviewed_source_snapshot_support_implementation_sha
reviewed_generation_support_implementation_sha
source_snapshot_gate_receipt_key_sha256 / generation_gate_receipt_key_sha256
generation_result
ticker_identities_exposed=false
block_assignments_exposed=false
raw_seed_exposed=false
research_opened=false
```

plus the fixed contract identifiers
(`V8H_PRIVATE_PARTITION_GENERATION_V1`, `V8H_PARTITION_MANIFEST_V1`) and
Git commit/blob provenance. No ticker identity, private path, tier/block
assignment (the actual membership lists), raw manifest byte, raw seed,
price, feature, or outcome may appear in any public V8H artifact at any
stage, including the generation artifact itself, any gate receipt, or any
independent review report — only the aggregate per-block *hash* and
*count* of each tier are public; the tier *contents* never are. This
design task performs no membership generation and therefore inspects,
reads, or exposes no ticker identity, private path, sealed membership, or
protected raw payload of any kind, for either V8G or any future V8H
candidate.

## 12. Preservation and point-of-use verification requirements

```text
preservation_required_before_any_reuse=true
point_of_use_reverification_required=true
preservation_self_hash_recomputation_required=true
```

Once generated, the private seed and the private partition manifest must
be durably preserved outside the Git working tree, in canonical
machine-local **private** durable state — reusing the existing repository
pattern exactly (`src/v8c_human_gate_consumption.py`'s
`CANONICAL_CONSUMPTION_STATE_ROOT` resolution), under its own
`v8h-partition-generation-state` subdirectory, mirroring V8G's own
`v8g-locator-gate-state` naming convention under the V8H namespace; a
future implementation must not invent a new or per-checkout state root.
Publication is exclusive and no-overwrite, with flush/fsync on the
artifact and its containing directory, exactly per
`AI_REAL_EXECUTION_RUNBOOK.md`'s durable-publication discipline.

Every later point of use must independently recompute and check, never
trusting a cached or previously verified result alone:

- the private manifest's own self-hash (never trusting the self-declared
  `manifest_sha256` field alone, exactly as `src/v8_partition.py`'s
  `read_partition_manifest` self-hash recomputation already requires);
- the partition seed's `SHA256` commitment against the publicly recorded
  `partition_seed_sha256`;
- all five block ticker-list hashes (`t0`…`t_spare`) against the publicly
  recorded values;
- binding to the exact frozen source-snapshot artifact
  (`source_snapshot_artifact_self_sha256`, §7.2).

A prior PASS artifact alone is never sufficient — any stage that later
reads V8H membership (transport readiness, raw acquisition, research
opening, or any T2-equivalent stage) must perform this full
re-verification itself before proceeding.

## 13. Independent exact-SHA review requirements before any subsequent gated stage

```text
independent_review_required_before_generation_implementation=true
independent_review_required_before_generation_gate_consumption=true
independent_review_required_before_membership_disclosure=true
independent_review_required_before_research_opening=true
review_binds_to_exact_git_commit=true
review_recommendation_is_not_self_authorizing=true
```

No V8H stage after this design draft — generation-support
implementation, the generation gate, membership disclosure of any kind,
preservation, or research opening — may proceed on the authority of this
draft alone. Each requires its own independent, exact-SHA-bound
methodology review, performed and reported by the reviewing authority
(GPT-5.6 Sol, per this task's own final-review assignment, or whichever
authority a future task names), before that stage's implementation may be
authorized, exactly mirroring the review chain V8G's own §7 minimum stage
order already froze (`INDEPENDENT_V8G_DESIGN_REVIEW`,
`INDEPENDENT_V8G_LOCATOR_SUPPORT_REVIEW`, and so on for every subsequent
stage). A review recommendation is never automatically adopted or
self-authorizing; per `AI_RESEARCH_EXECUTION_RULES.md` §5, only an
explicit human or ChatGPT decision converts a review finding into
authority. This draft itself requires exactly such an independent
exact-SHA review, against the precise commit that introduces this file,
before any V8H implementation work of any kind begins.

## 14. Future profitability remains unestablished

```text
V8H_future_profitability_established=false
V8H_design_draft_establishes_profitability=false
V8H_design_draft_establishes_data_quality=false
V8H_design_draft_establishes_readiness=false
V8H_design_draft_establishes_strategy_validity=false
```

This draft establishes nothing about the profitability, historical or
future, of any strategy under V8H. It does not establish data quality,
transport readiness, or strategy validity for any V8H partition, because
no V8H partition yet exists: this draft precedes generation entirely.
Exactly as V8G's own §4 recorded for its locator contract, a future V8H
partition-generation PASS — once the now-frozen contract in §6 is
implemented, reviewed, authorized, and executed — would establish only
that a fresh partition was generated under the frozen, preregistered,
non-outcome-based contract; it would say nothing by itself about data
quality, readiness, or any strategy's profitability (§7.4).

## 15. Explicit unresolved decisions (CHATGPT_DECISION_REQUIRED)

Task `V8H_PARTITION_GENERATION_CONTRACT_COMPLETION` resolved the
following items previously listed here, using the binding decisions it
supplied (§6, §7.1, §7.2, §10, §11, §12): the exact V8H
partition-generation mechanism; the exact source-snapshot identity; the
exact deterministic/randomization commitment scheme; the exact allocation
rules across tiers; the exact `T0`/`T1`/`T2`/`T3`/`T_spare` semantics and
sizing; the exact manifest hashing/canonicalization scheme; the exact
preservation storage location/mechanism; the exact acceptance and failure
semantics for generation; and the exact name, receipt-key material, and
authorization grammar for the source-snapshot and partition-generation
gates specifically. None of these values was invented by an execution
agent; each is the binding decision the task supplied.

The following methodological choices remain genuinely open. They are not
fixed by this draft and must not be silently decided by any execution
agent. Each requires an explicit future ChatGPT methodology decision,
stated in full, before the corresponding V8H stage may be implemented or
reviewed:

```text
CHATGPT_DECISION_REQUIRED: exact name, receipt-key material, and
  authorization grammar for the V8H membership-disclosure gate (§7.3)
CHATGPT_DECISION_REQUIRED: exact name, receipt-key material, and
  authorization grammar for the V8H research-opening gate (§7.3)
CHATGPT_DECISION_REQUIRED: exact minimum V8H stage order and namespace
  substitution beyond the two gates this task resolved (§7.1, §7.2) --
  i.e. the independent-review stage names and any T1C/T2-equivalent
  authority-bridge, readiness, acquisition, and research-opening stage
  structure for V8H, analogous to V8G draft §7
CHATGPT_DECISION_REQUIRED: exact future successor/reallocation gate
  semantics governing any later use of T_spare (§6.2, §7.4)
```

No value is invented for any item above. Where this draft states a
requirement a future decision must satisfy (§5, §6, §9, §10, §12), that
requirement is binding; the specific mechanism or values that satisfy it
remain open.

## 16. Design task scope boundary

This design task itself performs:

```text
network=0
private_reads=0
gate_consumption=0
Yahoo=0
JPX=0
raw_acquisition=0
research_opening=0
implementation=0
partition_generation=0
allocation=0
```

```text
design_finalized=false
human_design_freeze_complete=false
source_snapshot_support_implemented=false
generation_support_implemented=false
approval_artifact_created=false
network_access_authorized=false
private_data_access_authorized=false
partition_generation_authorized=false
allocation_authorized=false
raw_acquisition_authorized=false
research_opening_authorized=false
production_authorized=false
human_gate_consumed=false
V8G_gate_or_receipt_reused=false
V8G_authorization_reused=false
```

This draft does not implement a generation mechanism or its verifier,
create an approval or freeze artifact, generate or read any private
membership, inspect any ticker identity, access Yahoo or JPX, consume any
gate, acquire raw data, or open research. It grants zero real network
authority, zero private/sealed-data authority, zero partition-generation
authority, zero allocation authority, zero raw-acquisition authority, zero
research-opening authority, and zero production authority. A future V8H
implementation task requires its own independent exact-SHA review (§13)
before any real private generation, private read, or gate consumption
begins.

```text
next_action=GPT_EXACT_SHA_INDEPENDENT_REVIEW
```
