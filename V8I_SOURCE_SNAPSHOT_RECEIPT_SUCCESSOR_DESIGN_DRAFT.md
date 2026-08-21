# V8I Source-Snapshot Receipt Successor Design Draft

This is a successor-study design draft only. It creates no implementation
authority, network authority, private-data authority, partition-generation
authority, allocation authority, raw-acquisition authority, human
authorization, gate receipt, or research-opening authority.

## 1. Study identity and predecessor disposition

```text
study=V8I_HISTORICAL_RESEARCH
study_type=SUCCESSOR_STUDY
predecessor=V8H_HISTORICAL_RESEARCH
predecessor_frozen_design_commit=543dedf19fe15a94121520e91e441cb751551bc1
predecessor_frozen_design_blob=c23eb3bdddf320861922e35aa172fa30908a866b
predecessor_freeze_approval_artifact=V8H_DESIGN_FREEZE_APPROVAL.json
predecessor_freeze_status=APPROVED_FROZEN
predecessor_freeze_amended_by_this_task=false
```

V8H's design freeze is unchanged and unamended by this draft. V8H remains,
and permanently remains, `APPROVED_FROZEN` exactly as recorded in
`V8H_DESIGN_FREEZE_APPROVAL.json`. This draft does not edit, reinterpret,
or weaken that record.

```text
predecessor_implementation_reviewed_commit=ecf74d7fb3a093bce5ceae372bd2d02c8499e43d
predecessor_implementation_review_result=BLOCK
predecessor_implementation_review_critical=0
predecessor_implementation_review_high=1
predecessor_implementation_review_medium=0
predecessor_implementation_review_high_finding=FROZEN_DESIGN_RECEIPT_CONTRACT_DEVIATION_WITHOUT_AUTHORITY
predecessor_real_jpx_requests=0
predecessor_real_yahoo_requests=0
predecessor_private_or_sealed_reads=0
predecessor_source_snapshot_gate_consumed=false
predecessor_partition_generation_gate_consumed=false
predecessor_partition_seeds_generated=0
predecessor_scientific_attempt_occurred=false
```

GPT independent review of the source-snapshot support implementation at
commit `ecf74d7fb3a093bce5ceae372bd2d02c8499e43d`
(`src/v8h_source_snapshot.py`, `tests/test_v8h_source_snapshot.py`) found
that V8H's own frozen design §7.1 simultaneously requires (a) durable
source-snapshot gate receipt publication at
`IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST`, and (b) that same immutable
receipt body to contain values only knowable after that request
(`source_raw_sha256`, `source_acquisition_utc`, `eligible_ticker_count`,
`eligible_ticker_list_sha256`, `t0_reproduction_status`). The
implementation correctly detected this contradiction and resolved it
architecturally (a pre-request receipt plus a separate post-request
evidence artifact), but per `AI_RESEARCH_EXECUTION_RULES.md` §2 an
execution agent is not authorized to resolve a frozen-design ambiguity on
its own initiative — only an explicit human or ChatGPT decision may do
that. GPT-5.6 Sol accordingly returned `RESULT=BLOCK` with exactly one
HIGH finding (`FROZEN_DESIGN_RECEIPT_CONTRACT_DEVIATION_WITHOUT_AUTHORITY`)
and zero CRITICAL/MEDIUM findings.

No real JPX request, real Yahoo request, private or sealed production
read, source-snapshot gate consumption, partition-generation gate
consumption, or partition-seed generation occurred at that commit. There
is no consumed scientific attempt to preserve. This is a **governance/
design-contract failure**, not a strategy failure and not a profitability
failure: nothing about label design, cost model, promotion rule, search
space, or any evaluated hypothesis was ever reached. V8I must not, and
does not, claim any V8H strategy or profitability failure.

```text
V8I_addresses_receipt_contract_authority_only=true
V8I_does_not_address_strategy_or_profitability=true
V8H_strategy_failure=false
V8H_profitability_failure=false
V8H_future_profitability_established=false
```

V8I is a new, independent study identity. It is **not** an amendment,
retry, reset, or continuation of the frozen V8H study. It does not edit
`V8H_FRESH_PRIVATE_PARTITION_SUCCESSOR_DESIGN_DRAFT.md`,
`V8H_DESIGN_FREEZE_APPROVAL.json`, `src/v8h_source_snapshot.py`, or
`tests/test_v8h_source_snapshot.py` — none of those are modified or
deleted by this task, and none of V8H's identifiers (its gate names,
receipt-key material, authorization grammar, design-candidate/frozen
commit, or freeze-approval binding) are reused by any V8I stage. Every
V8I prerequisite requires fresh, V8I-specific authorization.

`ecf74d7fb3a093bce5ceae372bd2d02c8499e43d` remains BLOCKED implementation
evidence. It is not, and this draft does not declare it,
V8I-production-approved. Its architectural pre-receipt/post-evidence
split is consistent with the direction this draft now freezes for V8I
(§2–§6 below), and may inform a future V8I implementation task, but that
consistency is informational only — it grants no authority. V8I requires
its own fresh implementation task, under its own fresh V8I-namespaced
module, and its own fresh GPT exact-SHA independent review, after V8I's
own design freeze.

## 2. Single change relative to V8H

```text
V8I_single_change=RESOLVE_SOURCE_SNAPSHOT_RECEIPT_CONTENT_TIMING_CONTRADICTION
```

V8I changes exactly one methodological element relative to V8H: it
resolves the source-snapshot gate's receipt-publication contradiction
identified in §1, by explicit binding decision (§3–§6), rather than
leaving it for an execution agent to resolve. No other element of the
V8H-inherited methodology changes (§7).

This is the same kind of narrowly-scoped, single-change succession V8G
made relative to V8F and V8H made relative to V8G: it does not relax any
acceptance criterion, does not permit any manual or outcome-informed
membership choice, and does not create any route by which the blocked
V8H implementation attempt could be revived under a different name. It
creates a wholly fresh V8I gate and evidence contract, fully independent
of every V8H receipt, artifact, or authorization.

## 3. Frozen contract: `V8I_SOURCE_SNAPSHOT_RECEIPT_EVIDENCE_SPLIT_V1`

```text
generation_uses_outcomes=false
source_snapshot_uses_prices_features_or_outcomes=false
V8I_source_snapshot_contract=V8I_SOURCE_SNAPSHOT_RECEIPT_EVIDENCE_SPLIT_V1
```

The source-snapshot publication model is split into two distinct,
separately published artifacts: (A) a minimal pre-request one-shot gate
receipt, and (B) a distinct post-request public-safe evidence artifact
that cryptographically binds to that exact receipt. This mirrors the
pattern already established by every other "immediately before X"
boundary gate in this repository (`src/v8f_t1c_preservation.py`'s
`V8F_RECEIPT_FIELDS`, `src/v8g_private_partition_locator.py`'s locator
receipt/artifact split) and resolves §1's contradiction by construction:
the receipt is fixed and published before anything the request could
produce is known, and everything the request produces lives only in the
separate, later artifact.

### 3.1 (A) Pre-request one-shot gate receipt

```text
gate=HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST
```

This gate's one-shot authorization permits only the one V8I
source-snapshot acquisition and its T0-reproduction validation (§7). It
does **not** authorize partition generation, membership disclosure, raw
historical price acquisition, or research opening.

The durable, exclusive, no-overwrite receipt is published strictly
**before** the first authorized JPX request. It contains, at minimum,
only values knowable at that instant:

```text
schema_version                                  (= "V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_V1")
artifact_role                                    (= "SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT")
study                                            (= "V8I_HISTORICAL_RESEARCH")
gate                                             (= "HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE")
reviewed_v8i_design_candidate_commit             (this design's own reviewed candidate, stage-appropriate
                                                   per the prefreeze/postfreeze binding discipline §8 inherits)
reviewed_source_snapshot_support_implementation_sha
authorization_identity_sha256
consumed                                         (= true)
consumption_count                                (= 1)
consumption_boundary                             (= "IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST")
consumption_timestamp_utc
```

It **must not** contain `source_raw_sha256`, `source_acquisition_utc`,
`eligible_ticker_count`, `eligible_ticker_list_sha256`,
`t0_reproduction_status`, or any other value unknowable at consumption
time. A receipt containing any such field, or missing any field above, or
carrying any extra field, is schema-invalid and fail-closed.

### 3.2 (B) Post-request source-snapshot evidence

After the one authorized JPX request completes and is parsed, publish a
distinct, canonical, write-once, public-safe evidence artifact:

```text
schema_version                                   (= "V8I_SOURCE_SNAPSHOT_ACQUISITION_EVIDENCE_V1")
artifact_role                                     (= "SOURCE_SNAPSHOT_ACQUISITION_EVIDENCE")
study                                             (= "V8I_HISTORICAL_RESEARCH")
reviewed_v8i_design_candidate_commit
reviewed_source_snapshot_support_implementation_sha
source_snapshot_gate_receipt_key_sha256           (§4, the fixed deterministic key)
source_snapshot_gate_receipt_bytes_sha256         (SHA-256 of the exact published receipt's own bytes)
source_snapshot_semantics                         (inherited unchanged, §7)
source_snapshot_clarification_commit              (inherited unchanged, §7)
v4_raw_sha_equality_required                      (= false, inherited unchanged, §7)
source_raw_sha256
source_raw_byte_count
source_acquisition_utc
t0_reproduction_status
eligible_ticker_count
eligible_ticker_list_sha256
t0_ticker_list_sha256
fresh_eligible_count
ticker_identities_exposed                         (= false)
private_path_exposed                              (= false)
raw_payload_exposed                               (= false)
historical_price_raw_acquisition_performed         (= false)
partition_generation_authorized                    (= false)
membership_disclosure_authorized                   (= false)
research_opened                                    (= false)
source_snapshot_result                              (= "PASS")
source_snapshot_artifact_self_sha256                (canonical self-hash, excluding only this field itself)
```

No ticker identity, private path, raw payload byte, partition membership,
or raw human authorization identity may appear in this artifact, or in
any log, exception message, or stdout output, under any circumstance.

Before publishing this artifact, the post-request stage must:

- read the exact durable receipt published in §3.1, mechanically
  re-validate its structural schema, and mechanically require exact
  equality between its bound fields (`reviewed_v8i_design_candidate_commit`,
  `reviewed_source_snapshot_support_implementation_sha`,
  `authorization_identity_sha256`) and this execution's own currently
  authorized values — never trusting a structurally well-formed but
  semantically stale receipt;
- compute `source_snapshot_gate_receipt_bytes_sha256` from the receipt's
  exact validated durable bytes, only after that semantic-binding check
  passes;
- compute `source_snapshot_gate_receipt_key_sha256` from the fixed,
  argument-independent key function (§4);
- perform the inherited V8/V8H source-universe reconstruction, T0
  reproduction, and fresh-eligible-count check (§7) before this artifact
  may report `source_snapshot_result=PASS`.

This binds the evidence artifact cryptographically to the exact immutable
pre-request receipt via both the receipt-key hash and the exact
receipt-bytes hash — a forged or substituted receipt, or a receipt from a
different execution, fails this binding and blocks evidence publication.

## 4. One-shot receipt key: fresh V8I-only namespace

```text
receipt_key_material =
    "V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_KEY_V1\0"
  + "ta1k1-arakawa/stock-analyzer"
  + "\0"
  + "V8I_HISTORICAL_RESEARCH"
  + "\0"
  + "HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE"

source_snapshot_gate_receipt_key_sha256 = SHA256(UTF8(receipt_key_material))
```

This key is fixed for the entire life of the V8I study the moment
repository identity, study name, and gate name are fixed. It deliberately
excludes every attempt-varying value — the authorization identity/hash,
the reviewed V8I design candidate commit, and the reviewed
source-snapshot-support implementation SHA — so that no fresh
authorization, no newly reviewed design candidate, and no newly reviewed
implementation can ever unlock a second request under this key. Once any
V8I source-snapshot gate receipt exists at all, no later attempt — under
any candidate, any implementation, any authorization, any timestamp — can
ever consume this gate again.

This domain-separation string, and every literal composed into it, is
distinct from V8H's own `V8H_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_KEY_V1`
material and from V8G's own `V8G_PRIVATE_PARTITION_LOCATOR_GATE_RECEIPT_KEY_V1`
material (`src/v8h_source_snapshot.py`, `src/v8g_private_partition_locator.py`).
No V8I receipt key ever equals, derives from, or may be substituted for
either predecessor's key.

## 5. Authorization grammar: fresh V8I-only namespace

```text
authorization_identity =
    "V8I_HUMAN_AUTHORIZE_SOURCE_SNAPSHOT_ACQUISITION_AT_"
  + reviewed_v8i_design_candidate_commit
  + "_WITH_"
  + reviewed_source_snapshot_support_implementation_sha
```

Both components are exactly 40 lowercase hex characters (a full Git
commit object id). Any component of the wrong length, wrong case, or
containing a non-hex character is a grammar mismatch and a `PRE_GATE
BLOCK`, never post-gate, never coerced. Only

```text
authorization_identity_sha256 = SHA256(UTF8(authorization_identity))
```

may ever be persisted or appear in a receipt, evidence artifact,
exception message, or any other output. The raw `authorization_identity`
string itself is never printed, logged, or persisted. This grammar is
bound at minimum to the exact V8I design identity
(`reviewed_v8i_design_candidate_commit`) and the exact independently
reviewed source-snapshot-support implementation SHA. It does not reuse,
derive from, or accept V8H's own `V8H_HUMAN_AUTHORIZE_SOURCE_SNAPSHOT_
ACQUISITION_AT_...` grammar or any V8H human authorization; a V8H
authorization identity or its hash can never satisfy this grammar.

## 6. Failure semantics

```text
PRE_GATE_BLOCK_CONDITIONS (before HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE
durable receipt publication; gate remains unconsumed where mechanically
possible under AI_REAL_EXECUTION_RUNBOOK.md):
  - Git/provenance mismatch
  - design-candidate/frozen-design or freeze-approval-artifact mismatch
  - implementation-binding mismatch
  - dirty working tree
  - malformed authorization (wrong grammar, wrong length, wrong case,
    non-hex character)
  - already-consumed receipt at the fixed key
  - missing prerequisite artifact

POST_GATE_TERMINAL_CONDITIONS (after HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_
GATE durable receipt publication; ANY failure here is terminal for this
V8I source-snapshot attempt):
  - request failure
  - private raw-source preservation failure
  - parsing failure
  - T0 reproduction failure
  - fresh-eligible-count failure (< 900, §7)
  - hashing failure
  - evidence-publication failure

no_second_jpx_request=true
no_provider_or_date_substitution=true
no_receipt_deletion_or_reset=true
no_same_study_source_snapshot_retry=true
successor_study_decision_required_on_post_gate_failure=true
```

A `PRE_GATE` failure never consumes the gate. Any `POST_GATE` failure —
of any kind, including an operational failure unrelated to the scientific
mechanism — is terminal for this V8I source-snapshot attempt: no second
JPX request, no provider or date substitution, no receipt deletion or
reset, and no same-study source-snapshot retry is permitted under any
circumstance. If the post-gate stage cannot produce a valid, fully
verified `V8I_SOURCE_SNAPSHOT_ACQUISITION_EVIDENCE_V1` artifact, V8I's
source-snapshot stage fails closed and requires a successor-study
decision, exactly mirroring the permanence discipline V8G froze for its
own locator gate and V8H inherited unchanged.

## 7. Inherited unchanged methodology

V8I inherits the following unchanged from the frozen V8H design, with no
threshold recalibration, provider substitution, retry-policy change,
partition-generation-contract change, evaluation-window change, or
stopping-rule change of any kind:

- the official JPX source-universe semantics
  (`SOURCE_SNAPSHOT_SEMANTICS = "IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT"`,
  `SOURCE_SNAPSHOT_CLARIFICATION_COMMIT =
  266999a8e48c77905dd7c7312fd41c7f38241d78`);
- exact T0 reproduction requirements against the existing frozen V4/T0
  evidence;
- the V4 raw-byte-equality policy (`V4_RAW_SHA_EQUALITY_REQUIRED = False`);
- the legacy/outcome-exposed ticker exclusions
  (`LEGACY_EXPOSED_TICKERS_OUTSIDE_T0`, unchanged);
- the minimum fresh eligible count of 900 (`3 x 300`) and its fail-closed
  BLOCK on insufficiency;
- `T0`/`T1`/`T2`/`T3`/`T_spare` semantics exactly as V8H froze them
  (`T1` `max_validation_access = 1`; `T2` sealed holdout for exactly one
  frozen final candidate; `T3` sealed reserve; `T_spare` remainder,
  reallocable only under a future separately authorized gate);
- block size = 300;
- the 32-byte OS-CSPRNG `partition_seed_bytes` commitment
  (`os.urandom(32)`-equivalent, no user-selected seed);
- HMAC-SHA256 deterministic allocation from the secret seed
  (`V8H_PARTITION_ASSIGN_V1` domain separation and the sort/cut rule);
- canonical ticker/list hashing (`SHA256(UTF8("\n".join(tickers)+"\n"))`);
- canonical JSON rules (`json.dumps(..., ensure_ascii=False,
  sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"`);
- the private/public partition-manifest separation (private manifest may
  contain block assignments and the raw seed; public evidence never
  does);
- the partition-generation gate concept and its post-generation failure
  semantics (§6's `POST_GATE_TERMINAL_CONDITIONS` pattern, generalized);
- every leakage and post-outcome-selection prohibition;
- the prohibition on outcome-based or manual membership selection;
- preservation and point-of-use re-verification requirements;
- the evaluation period, labels/targets, strategy, cost/slippage, DQ
  thresholds, promotion thresholds, search space, and stopping rules;
- forward-performance interpretation.

```text
future_profitability_established=false
```

The general repository namespace-substitution rule already established
across V8F→V8G→V8H (`V8H_<TOKEN>` → `V8I_<TOKEN>`, `study=
V8H_HISTORICAL_RESEARCH` → `study=V8I_HISTORICAL_RESEARCH`,
`reviewed_v8h_design_candidate_commit` →
`reviewed_v8i_design_candidate_commit`) applies mechanically to every
inherited item above, including the eventual V8I-namespaced
partition-generation gate, whenever that stage is implemented under V8I.
This draft freezes only the source-snapshot receipt/evidence contract
(§3–§6); it does not re-specify a new partition-generation gate name,
receipt-key material, or authorization grammar — those remain V8H's own
frozen `HUMAN_V8H_PRIVATE_PARTITION_GENERATION_GATE` contract, inherited
unchanged in substance and namespace-substituted mechanically if and when
a V8I partition-generation stage is implemented, which is outside this
narrower task's scope.

## 8. Stage-aware design-authority binding (inherited)

V8I inherits, unchanged in mechanism and applied under its own namespace,
the same prefreeze/postfreeze design-candidate binding discipline V8G
froze in its own §2.3 and V8H continued: before `HUMAN_V8I_DESIGN_FREEZE`
completes, every V8I stage's authority binds to
`reviewed_v8i_design_candidate_commit` (the exact reviewed Git commit of
*this* design draft, or a later independently re-reviewed amendment of
it) — never to a nonexistent "frozen design commit." The same
design-candidate staleness rule applies: any new commit to this draft
immediately stales every existing `reviewed_v8i_design_candidate_commit`
binding tied to the prior content, and a materially amended candidate
requires its own fresh independent review before it can authorize
anything.

## 9. Explicit unresolved decisions (CHATGPT_DECISION_REQUIRED)

The following methodological choices are not required to complete this
narrower source-snapshot successor design and are not fixed by this
draft. They must not be silently decided by any execution agent, and
this draft does not attempt to resolve them:

```text
CHATGPT_DECISION_REQUIRED: exact name, receipt-key material, and
  authorization grammar for the V8I membership-disclosure gate
CHATGPT_DECISION_REQUIRED: exact name, receipt-key material, and
  authorization grammar for the V8I research-opening gate
CHATGPT_DECISION_REQUIRED: exact minimum V8I stage order and namespace
  substitution beyond the source-snapshot gate this draft resolved --
  including the V8I-namespaced partition-generation gate's own exact
  name/receipt-key/grammar, the independent-review stage names, and any
  T1C/T2-equivalent authority-bridge, readiness, acquisition, and
  research-opening stage structure for V8I, analogous to V8G draft §7
CHATGPT_DECISION_REQUIRED: exact future successor/reallocation gate
  semantics governing any later use of T_spare under V8I
```

No value is invented for any item above. Where this draft states a
requirement a future decision must satisfy (§6, §7), that requirement is
binding; the specific mechanism or values that satisfy it remain open.

## 10. Design task scope boundary

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
V8H_frozen_design_edited=false
V8H_freeze_approval_edited=false
V8H_blocked_implementation_edited=false
V8H_blocked_implementation_deleted=false
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
V8H_gate_or_receipt_reused=false
V8H_authorization_reused=false
V8G_gate_or_receipt_reused=false
```

This draft does not implement the source-snapshot receipt/evidence split
or its verifier, create an approval or freeze artifact, generate or read
any private data, inspect any ticker identity, access Yahoo or JPX,
consume any gate, acquire raw data, or open research. It grants zero real
network authority, zero private/sealed-data authority, zero
partition-generation authority, zero allocation authority, zero
raw-acquisition authority, zero research-opening authority, and zero
production authority. A future V8I source-snapshot implementation task
requires its own independent exact-SHA review before any real private
acquisition, private read, or gate consumption begins.

```text
next_action=GPT_EXACT_SHA_INDEPENDENT_REVIEW
```
