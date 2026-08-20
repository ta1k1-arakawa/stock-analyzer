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
- research-opening rules other than the fresh gate this draft inserts
  before any V8H membership is revealed (§8).

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

## 6. Partition-generation commitment requirements

```text
generation_uses_outcomes=false
generation_uses_prices_features_or_returns=false
generation_uses_manual_selection=false
generation_uses_performance_based_inclusion_or_exclusion=false
generation_is_fixed_before_any_private_byte_is_read=true
```

The new V8H partition must be generated by a mechanism that is entirely
free of outcome-based or manual membership selection: no candidate
identity may be included in, excluded from, or reassigned between tiers
because of its observed or anticipated price behavior, strategy
performance, or any other outcome-linked signal, at generation time or
subsequently. Generation must be either fully deterministic from a frozen,
publicly committed rule applied to a frozen source snapshot, or
randomized under a frozen, publicly pre-committed randomization scheme
(e.g., a committed seed or seed-derivation rule fixed and hashed before
generation) — never a mechanism whose outcome could be steered, tuned, or
retried by any party after seeing a preliminary result.

The following elements must each be frozen, in full, before any V8H
private membership is generated, and each remains, at the level of exact
values, algorithm, and thresholds, an open methodological decision this
draft does not itself make:

```text
generation_mechanism: CHATGPT_DECISION_REQUIRED
source_snapshot_identity: CHATGPT_DECISION_REQUIRED
deterministic_or_randomization_commitment_scheme: CHATGPT_DECISION_REQUIRED
allocation_rules: CHATGPT_DECISION_REQUIRED
T0_T1_T2_T3_Tspare_semantics: CHATGPT_DECISION_REQUIRED
tier_sizes_or_sizing_rule: CHATGPT_DECISION_REQUIRED
manifest_hashing_and_canonicalization_scheme: CHATGPT_DECISION_REQUIRED
preservation_rules: CHATGPT_DECISION_REQUIRED
acceptance_and_failure_semantics: CHATGPT_DECISION_REQUIRED
```

This draft intentionally does not invent values for any of the above. Per
`AI_RESEARCH_EXECUTION_RULES.md` §2, an execution agent must not
independently choose a data partition, allocation rule, or acceptance
criterion; each item above requires an explicit ChatGPT methodology
decision, stated in full, in a future `V8H` design amendment or successor
draft, before any generation implementation may be reviewed or
authorized. Whatever mechanism is eventually chosen must, at minimum,
satisfy every requirement already fixed elsewhere in this section and in
§5, §9, §10, and §11 — those requirements bound the choice without
prescribing it.

Once frozen, the generation contract itself becomes subject to the same
staleness discipline V8G established for its own frozen contract (V8G
draft §2.3's `design_candidate_staleness_rule`): any later amendment to
the frozen V8H generation contract invalidates every authorization
granted against the prior version, and requires a fresh independent
review before it can authorize anything.

## 7. Fresh human-gate requirements

```text
V8H_gates_are_fresh=true
V8H_gates_reuse_no_V8G_receipt=true
V8H_gates_reuse_no_V8G_authorization=true
```

V8H requires its own, entirely fresh set of one-shot human gates,
independent of every V8G gate, receipt, and authorization:

- a gate immediately before the frozen generation contract may be applied
  to produce actual membership (analogous in role, never in identity or
  receipt key, to V8G's
  `HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE`) — its exact
  name, receipt-key material, and authorization grammar are
  `CHATGPT_DECISION_REQUIRED`, to be fixed alongside the generation
  contract itself (§6) so the gate binds to the exact frozen contract it
  authorizes, following the same stage-aware, prefreeze/postfreeze design-
  candidate binding discipline V8G froze in its own §2.3;
- a separate gate before any private membership is disclosed to any actor
  or artifact beyond what §10 permits as public-safe evidence;
- a separate gate before any V8H research-opening stage, strictly after
  membership is sealed and preserved (§11) and after the independent
  exact-SHA reviews required by §12.

Every V8H gate must be one-shot, durably receipted outside this
repository per the existing canonical durable-state pattern
(`CANONICAL_CONSUMPTION_STATE_ROOT`), and bound to its own exact reviewed
design-candidate commit (prefreeze) or frozen design commit (postfreeze),
exactly as V8G's own gates were bound in its §2.3. No V8H gate may be
consumed using a V8G receipt, a V8G authorization identity, or any
V8G-derived binding value.

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
be frozen and content-addressed (e.g., by a committed hash of the
snapshot's own canonical serialization) strictly before generation
begins, exactly as the frozen historical `manifest_sha256` anchor pattern
already established for the V8/V8B–V8G partition (§1). Once frozen, the
source snapshot may never be substituted, extended, re-scraped, or
re-scoped for the purposes of this generation — any change to the source
snapshot after it is frozen requires the same successor-study-level
decision as any other frozen-contract amendment (§6, §9). The exact
identity of the source snapshot (what it is drawn from, its as-of date,
and its content-addressing scheme) is `CHATGPT_DECISION_REQUIRED` (§6);
this section fixes only the immutability requirement once that identity
is chosen.

## 11. Membership confidentiality / public-safe evidence rules

```text
ticker_identities_exposed=false
block_or_tier_assignments_exposed=false
raw_or_private_payload_persisted_publicly=false
```

Public V8H generation, gate, and evidence artifacts may expose only:

- fixed contract identifiers and the frozen hashing/canonicalization
  scheme once §6 is decided;
- safe nonnegative integer counts (e.g., tier sizes, candidate counts);
- safe SHA-256 hex digests explicitly designated as safe by the frozen
  contract (e.g., a manifest self-hash, a receipt-key hash, a
  candidate-set hash);
- safe booleans and enums (generation result, gate-consumption state,
  research-opened, etc.);
- Git commit/blob provenance.

No ticker identity, private path, tier/block assignment, raw manifest
byte, price, feature, or outcome may appear in any public V8H artifact at
any stage, including the generation artifact itself, any gate receipt, or
any independent review report. This design task performs no membership
generation and therefore inspects, reads, or exposes no ticker identity,
private path, sealed membership, or protected raw payload of any kind, for
either V8G or any future V8H candidate.

## 12. Preservation and point-of-use verification requirements

```text
preservation_required_before_any_reuse=true
point_of_use_reverification_required=true
preservation_self_hash_recomputation_required=true
```

Once generated, sealed V8H membership must be durably preserved under the
same discipline the V8/V8B–V8G lineage already applies to preservation
artifacts: exclusive, no-overwrite publication with flush/fsync on the
artifact and its containing directory; a recomputed self-hash verified
independently of any self-declared field (never trusting a declared hash
alone, exactly as the existing `_read_partition_manifest_bytes` pattern
already requires); and re-verification against the frozen manifest hash
at every future point of use, not only at generation time. Any stage that
later reads V8H membership (transport readiness, raw acquisition,
research opening, or any T2-equivalent stage) must independently
recompute and check this binding before proceeding — reading a cached or
previously verified result is never a substitute for point-of-use
re-verification. The exact preservation storage location and mechanism
are `CHATGPT_DECISION_REQUIRED` (§6), but must in all cases exclude this
repository's own working tree and any location this draft's own scope
boundary (§13) would otherwise disallow.

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
partition-generation PASS — once the open decisions in §6 are resolved,
reviewed, authorized, and executed — would establish only that a fresh
partition was generated under the frozen, preregistered, non-outcome-based
contract; it would say nothing by itself about data quality, readiness, or
any strategy's profitability.

## 15. Explicit unresolved decisions (CHATGPT_DECISION_REQUIRED)

The following methodological choices are not fixed by this draft and must
not be silently decided by any execution agent. Each requires an explicit
future ChatGPT methodology decision, stated in full, before the
corresponding V8H stage may be implemented or reviewed:

```text
CHATGPT_DECISION_REQUIRED: exact V8H partition-generation mechanism
CHATGPT_DECISION_REQUIRED: exact source-snapshot identity and as-of date
CHATGPT_DECISION_REQUIRED: exact deterministic-or-randomization commitment
  scheme (including any seed/seed-derivation commitment)
CHATGPT_DECISION_REQUIRED: exact allocation rules across tiers
CHATGPT_DECISION_REQUIRED: exact T0/T1/T2/T3/T_spare semantics and sizing
CHATGPT_DECISION_REQUIRED: exact manifest hashing/canonicalization scheme
CHATGPT_DECISION_REQUIRED: exact preservation storage location/mechanism
CHATGPT_DECISION_REQUIRED: exact acceptance and failure semantics for
  generation (including what constitutes a generation-stage BLOCK)
CHATGPT_DECISION_REQUIRED: exact name, receipt-key material, and
  authorization grammar for every fresh V8H human gate
CHATGPT_DECISION_REQUIRED: exact minimum V8H stage order and namespace
  substitution, analogous to V8G draft §7, once the above are fixed
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
