# V8G Private Partition Locator Successor Design Draft

This is a successor-study design draft only. It creates no implementation
authority, network authority, private-data authority, allocation authority,
human authorization, gate receipt, or research-opening authority.

## 1. Study identity and predecessor disposition

```text
study=V8G_HISTORICAL_RESEARCH
study_type=SUCCESSOR_STUDY
predecessor=V8F_HISTORICAL_RESEARCH
predecessor_terminal_commit=d1447df86b0caa7a5240d45cba8f01f8829a940c
predecessor_terminal_artifact=V8F_T1C_PRESERVATION_TERMINAL_ADJUDICATION.json
predecessor_terminal_artifact_blob=91572d706c7ccb6f6f2e3c840791cb14c7eb8bca
```

V8G is a new study identity. It is not a V8F retry, amendment, continuation,
or repair under the V8F frozen design.

The V8F terminal disposition is preserved exactly, unchanged, as historical
evidence only:

```text
execution_result=BLOCK
disposition=BLOCK_CLOSED
failure_class=PRIVATE_PRESERVATION_PROVENANCE_LOCATOR_FAILURE
failure_reason=V8F_LOCATOR_ZERO_MATCHING_CANDIDATES
strategy_failure=false
profitability_failure=false
future_profitability_established=false
```

That disposition is private-provenance-locator evidence only. It is not
strategy evidence, profitability evidence, data-quality evidence, or T1C/T2
readiness evidence. It states only that the V8F content-addressed locator's
metadata-only candidate snapshot for that one execution contained zero
partition manifests whose independently recomputed canonical hash matched
the frozen authorized identity. This draft does not reinterpret, upgrade, or
downgrade that finding; it is carried forward exactly as recorded.

No V8F human authorization, gate, receipt, preservation result, locator
result, or research-opening authority authorizes any V8G stage. Every V8G
prerequisite requires fresh, V8G-specific authorization, bound — per stage,
exactly as §2.3 freezes — to either the exact independently reviewed V8G
design candidate commit applicable to that stage (every stage before
`HUMAN_V8G_DESIGN_FREEZE`) or the exact frozen V8G design commit (every
stage from `HUMAN_V8G_DESIGN_FREEZE` onward).

## 2. Single change relative to V8F

V8G changes exactly one operational/private-provenance contract relative to
V8F: it introduces a separate, explicit
`V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT` stage, with its own one-shot
gate, strictly **before** V8G T1C preservation. No strategy, evaluation,
data-quality, or transport parameter is changed by this draft.

The V8F content-addressed locator (`V8F-T1C-LOCATOR-001` /
`V8F-T1C-LOCATOR-HIGH-001`) resolved candidate manifests, gated, and
verified as one combined step inside the T1C preservation-recheck gate
itself. V8G separates that concern into its own named stage, its own
one-shot human gate, and its own durable, independently reviewable public
artifact, produced and reviewed *before* the T1C preservation gate is ever
approached. This is a workflow/stage-boundary change only; it does not
change what "authorized partition identity" means, how a candidate is
verified, or any downstream preservation/readiness/acquisition semantics.

### 2.1 Frozen contract: `PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_V1`

The historical authorized partition identity is unchanged and remains
exactly:

```text
manifest_sha256=0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62
partition_implementation_commit=36cbed941050e728f7f96ce2af505e81175cc02c
```

The real production partition-manifest path was historically intentionally
not publicly recorded (per `V8_STATE.json`'s
`partition_manifest_path_recorded=false`). V8G does not invent, guess, or
assume it. The frozen contract below exists precisely to resolve it, once,
by content address, from a metadata-only enumerated candidate set — never
from a hardcoded or asserted path.

#### 2.1.1 Metadata-only candidate snapshot

Before locator gate consumption, a future V8G locator support
implementation must:

- enumerate every accessible file with exact basename
  `partition_manifest.json`;
- scope the enumeration to every ready local filesystem volume visible to
  the Windows production process whose `DriveType` is `Fixed` or
  `Removable`;
- exclude network volumes and CD/optical volumes from the scope;
- exclude this repository's own working-tree subtree from the scope;
- read no candidate content and inspect no ticker identity during
  enumeration;
- never print, log, or persist any candidate path;
- normalize every discovered path to its canonical form (§2.1.2) and
  deduplicate the candidate set on that normalized form;
- freeze the exact resulting candidate path list in memory for this one
  execution only;
- require `candidate_count >= 1`; zero candidates is a `PRE_GATE` `BLOCK`.

No candidate count is hardcoded or assumed by this design. The candidate
universe for a given execution is exactly whatever that execution's
deterministic enumeration produces at that moment — never a previously
observed count, and never supplemented after gate consumption. Adding,
removing, or substituting a candidate after the gate is consumed is
prohibited under all circumstances (§2.1.3).

#### 2.1.2 Safe path-hash contract

To identify a candidate publicly without ever disclosing it, every
normalized candidate path is reduced to a domain-separated SHA-256 digest
before any public evidence is produced. This canonicalization and hashing
scheme is frozen exactly as follows and must not be altered by any future
implementation task without a new successor-study decision:

```text
canonical_path_text =
  NFC(
    str(Path(path).resolve(strict=True))
      .replace("\\", "/")
      .casefold()
  )

locator_path_sha256 =
  SHA256(
    UTF8(
      "V8G_PRIVATE_PARTITION_LOCATOR_PATH_V1\0"
      + canonical_path_text
    )
  )
```

`canonical_path_text` itself must never be published, logged, or persisted
in any public artifact, receipt, or exception message — only
`locator_path_sha256` (and, where explicitly listed as safe in §2.1.4, the
selected candidate's hash) may appear in public evidence.

The candidate-set digest is frozen exactly as follows:

```text
candidate_set_serialization_v1(hash_list) =
    "V8G_PRIVATE_PARTITION_LOCATOR_CANDIDATE_SET_V1\n"
  + str(len(hash_list)) + "\n"
  + "\n".join(hash_list)
  + "\n"

  where hash_list is the full list of locator_path_sha256 values for every
  normalized, deduplicated candidate, sorted ascending as lowercase hex
  strings; require every element unique (a duplicate after path
  normalization is already excluded by §2.1.1, so a duplicate hash here is
  itself a fail-closed schema violation, never silently deduplicated again).

candidate_set_sha256 = SHA256(UTF8(candidate_set_serialization_v1(hash_list)))
```

The leading domain-separation line and explicit count line are frozen parts
of the serialization; a future implementation must reproduce them exactly
byte-for-byte. Public evidence may contain only: `candidate_count`,
`candidate_set_sha256`, `selected_locator_path_sha256` (§2.1.4, once
established), and other explicitly approved safe hashes/counts/booleans.
Raw paths, ticker identities, block assignments, and raw manifest bytes are
never safe evidence at any point in this contract.

#### 2.1.3 Locator human gate

A fresh, V8G-only, one-shot gate is frozen:

```text
gate=HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_CANDIDATE_PARTITION_BYTE_READ
```

This gate's authority is a prefreeze authority (§2.3): it binds exactly to

```text
locator_gate_authority_binding:
  reviewed_v8g_design_candidate_commit         (§2.3; the candidate commit
                                                  this locator stage's own
                                                  independent review PASS
                                                  actually approved)
  reviewed_locator_support_implementation_sha  (the exact reviewed Git
                                                  commit of the locator-
                                                  support module, §2.2)
  study                                        (= "V8G_HISTORICAL_RESEARCH")
  gate = "HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE"
  expected_partition_manifest_sha256           (§2.1, frozen identity)
  expected_partition_implementation_commit     (§2.1, frozen identity)
```

Mechanically re-verifying that binding — the executing implementation's own
commit and every field above equal exactly what was independently reviewed
and authorized for this stage, exactly as the existing
`_validate_reviewed_support_implementation_binding` /
`_validate_public_preflight` pattern already does for V8F's T1C
preservation gate — is itself a prerequisite check before this gate may be
consumed; a mismatch on any field is a `PRE_GATE` `BLOCK`, never a
post-gate condition.

**Human authorization grammar.** The human authorization that consumes this
gate is a single deterministic string, frozen exactly, binding human
authority to the exact reviewed candidate, exact reviewed implementation,
and frozen partition identity above:

```text
authorization_identity =
    "V8G_HUMAN_AUTHORIZE_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_AT_"
  + reviewed_v8g_design_candidate_commit
  + "_WITH_"
  + reviewed_locator_support_implementation_sha
  + "_FOR_MANIFEST_"
  + expected_partition_manifest_sha256
  + "_IMPL_"
  + expected_partition_implementation_commit
```

Every component must be exact lowercase hexadecimal of its frozen length:
`reviewed_v8g_design_candidate_commit`, `reviewed_locator_support_implementation_sha`,
and `expected_partition_implementation_commit` are each exactly 40
lowercase hex characters (a full Git commit object id);
`expected_partition_manifest_sha256` is exactly 64 lowercase hex characters
(a SHA-256 digest). Any component of the wrong length, wrong case, or
containing a non-hex character is a grammar mismatch and a `PRE_GATE`
`BLOCK` — never a post-gate condition, and never silently coerced (no
case-folding, no trimming).

The raw `authorization_identity` string must never be printed, logged,
persisted publicly, or included in any exception message, receipt, or
artifact, at any point, under any failure mode. Only its digest,

```text
authorization_identity_sha256 = SHA256(UTF8(authorization_identity))
```

may ever be persisted or appear in any receipt or safe result.

**Deterministic one-shot receipt key.** The receipt key that determines
whether this gate has already been consumed is frozen to be exactly:

```text
receipt_key_material =
    "V8G_PRIVATE_PARTITION_LOCATOR_GATE_RECEIPT_KEY_V1\0"
  + "ta1k1-arakawa/stock-analyzer"
  + "\0"
  + "V8G_HISTORICAL_RESEARCH"
  + "\0"
  + "HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE"

locator_gate_receipt_key_sha256 = SHA256(UTF8(receipt_key_material))
```

This deliberately, and unlike V8F's own T1C preservation receipt key
(`reviewed_v8f_design_candidate_commit` + `authorization_identity_sha256` +
`authorized_allocation_artifact_self_hash`), excludes every value that
could ever change across attempts: the authorization identity/hash, the
design candidate commit, the locator-support implementation SHA, the
partition manifest SHA, the selected path hash, and the candidate-set hash.
None of those may ever appear in `receipt_key_material`. This is
intentional: `HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE` may
be durably consumed **at most once for the entire life of the V8G study**
— the receipt key is fixed the moment the repository identity, study name,
and gate name are fixed, and stays fixed regardless of which authorization,
which reviewed design candidate, or which reviewed implementation was used
at the one execution that consumed it. Once any V8G locator gate receipt
exists at all, no later fresh human authorization, no later amended and
independently re-reviewed design candidate, and no later locator-support
implementation can ever obtain a second locator execution in this study —
the fixed-key receipt itself blocks it, before the authorization-grammar
and binding checks above are even reached. This is strictly stronger than,
and does not weaken, the binding checks above: those checks still ensure
that whichever single execution *does* consume this fixed key used a
genuinely reviewed and current candidate/implementation; the fixed key
ensures no second execution — under any candidate, any implementation, any
authorization — can ever happen at all. A post-gate `BLOCK` under this
gate therefore always requires a successor study, exactly as this
subsection's existing failure semantics already freeze below.

The exact reviewed candidate/implementation/partition-identity values used
at the one execution that does consume this key belong in the receipt
**body**, never the key, precisely so they remain independently verifiable
without ever being able to unlock a second consumption. The receipt body
must contain and validate at minimum:

```text
schema_version
artifact_role
study                                   (= "V8G_HISTORICAL_RESEARCH")
gate                                    (= "HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE")
reviewed_v8g_design_candidate_commit
reviewed_locator_support_implementation_sha
expected_partition_manifest_sha256
expected_partition_implementation_commit
authorization_identity_sha256
consumed=true
consumption_count=1
consumption_boundary
consumption_timestamp_utc
```

The exact JSON field ordering/typing is frozen by the future implementation
that builds this receipt; the field *set* above is frozen by this design.
None of these fields is a private path, a ticker identity, a raw manifest
byte, or the raw authorization identity — they are exactly the same class
of safe Git-commit/hash/enum/boolean/timestamp values `V8F_RECEIPT_FIELDS`
already safely persists. Unlike the V8F combined gate, this receipt never
records an allocation-artifact hash: the locator stage resolves and
verifies only the partition manifest, never the T1C allocation, which
remains the separate, later V8G T1C preservation gate's exclusive concern.

**Durable state.** The receipt lives in canonical machine-local durable
state, outside this repository, shared across every checkout of
`ta1k1-arakawa/stock-analyzer` on that machine — never a per-checkout or
HOME/USERPROFILE-derived location. A future implementation must reuse the
existing repository pattern for this exactly
(`src/v8c_human_gate_consumption.py`'s `CANONICAL_CONSUMPTION_STATE_ROOT`
resolution: the Windows `FOLDERID_ProgramData` known-folder path via
`SHGetKnownFolderPath`, with a POSIX `/var/lib/stock-analyzer` fallback for
local development/testing), under its own `v8g-locator-gate-state`
subdirectory; it must not invent a new or per-checkout state root. Receipt
publication is exclusive and no-overwrite, with flush/fsync on the receipt
file and fsync on its containing directory, exactly per
`AI_REAL_EXECUTION_RUNBOOK.md`'s durable-publication discipline. There is
no reset/delete/reuse API for this receipt, ever. An existing receipt at
the fixed key is a `PRE_GATE` `BLOCK` before any candidate content is
read, regardless of which authorization/candidate/implementation the
caller now presents. A malformed existing receipt (wrong schema, wrong
field set, unparseable, or any other structural defect) is also `BLOCK`,
and is never deleted, repaired, or replaced — it is left exactly as found,
and only a successor-study decision may address it. The receipt's own
filesystem path is itself machine-local private operational detail; it
never needs to appear in, and must not appear in, any public evidence.

Before this gate's durable receipt is published: `candidate_content_reads =
0`. Only the metadata-only enumeration and hashing of §2.1.1–§2.1.2 may
occur pre-gate.

After the durable receipt is published, and only then:

- the frozen candidate list from §2.1.1 is scanned exactly once; no
  re-enumeration and no second scan under any circumstance;
- each candidate may be read at most once;
- for each candidate, its self-declared `manifest_sha256` is never trusted
  alone — the canonical manifest hash must be independently recomputed from
  the candidate's own bytes, exactly as the existing
  `_read_partition_manifest_bytes` self-hash recomputation already does for
  V8F, and the recomputed hash is what is compared, never the self-declared
  field;
- the recomputed hash must equal the frozen `manifest_sha256` (§2.1) exactly;
- the candidate's `partition_implementation_git_commit` must equal the
  frozen `partition_implementation_commit` (§2.1) exactly;
- every existing manifest schema, frozen-binding, and provenance check this
  repository already applies to a partition manifest (schema exactness,
  `source_snapshot_semantics`, `source_snapshot_clarification_commit`,
  `block_assignments` presence, etc.) continues to apply unchanged;
- no candidate path, ticker identity, block assignment, or raw manifest byte
  is ever logged, printed, or persisted publicly at any point in this scan.

Outcome:

```text
exact_match_count == 1  => locator PASS
exact_match_count == 0  => BLOCK_CLOSED
exact_match_count  > 1  => BLOCK_CLOSED
```

Any post-gate failure, of any kind, is permanent for this authorization:

```text
locator_authorization_consumed=true
retry_allowed=false
reset_allowed=false
delete_allowed=false
reuse_allowed=false
candidate_substitution_allowed=false
alternate_root_allowed=false
successor_study_decision_required=true
```

This mirrors, and does not weaken, the fail-closed one-shot discipline
already frozen for the V8F/V8E/V8D preservation and readiness gates
(`AI_REAL_EXECUTION_RUNBOOK.md` §4, §8, §10).

#### 2.1.4 Safe locator artifact

On locator PASS, a future implementation must create exactly one durable,
public artifact — conceptually `V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT.json`
— containing at minimum:

```text
schema_version
artifact_role
study
reviewed_v8g_design_candidate_commit
reviewed_locator_support_implementation_sha
predecessor_terminal_commit
predecessor_terminal_artifact_blob
locator_contract              (= "PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_V1")
candidate_count
candidate_set_sha256
selected_locator_path_sha256
expected_partition_manifest_sha256
expected_partition_implementation_commit
locator_gate_receipt_key_sha256
locator_gate_receipt_bytes_sha256
ticker_identities_exposed=false
block_assignments_exposed=false
raw_or_private_payload_persisted_publicly=false
research_opened=false
raw_acquisition_performed=false
locator_result=PASS
```

No private path, ticker identity, block assignment, or raw manifest byte
may appear in this artifact under any field. Independent, exact-SHA review
of this public artifact by the methodology authority is mandatory before
any V8G T1C preservation authority may be granted; the locator stage does
not itself authorize preservation.

The locator artifact verifier — both this independent review step and any
later stage's own pre-gate preflight that consumes this artifact (§2.1.5)
— must require exact equality between the artifact's
`reviewed_v8g_design_candidate_commit` / `reviewed_locator_support_implementation_sha`
and that consuming stage's own currently authorized/reviewed values. A
locator artifact produced under a different (including a later-amended,
per the §2.3 staleness rule) design candidate commit, an unreviewed
locator-support implementation, a locator-support implementation the
artifact's own review never covered, or any predecessor study's
identifiers must `BLOCK` on that mismatch alone and must never authorize
V8G T1C preservation, regardless of how the artifact's own
`exact_match_count` or `locator_result` fields read.

#### 2.1.5 Future preservation use

V8G T1C preservation is itself a prefreeze stage (§2.3): its own authority
binds to its own `reviewed_v8g_design_candidate_commit` and its own
reviewed T1C-preservation-support implementation commit, exactly as the
locator stage's does (§2.1.3) — never to a frozen design commit, since none
exists yet at this stage either.

Before doing anything else — and strictly before the manifest-path
resolution below is even attempted — V8G T1C preservation must mechanically
re-verify that the reviewed locator artifact's
`reviewed_v8g_design_candidate_commit` and
`reviewed_locator_support_implementation_sha` (§2.1.4) exactly equal V8G
T1C preservation's own currently authorized/reviewed values. Any mismatch
is a `PRE_GATE` `BLOCK`, per §2.1.4's verifier requirement; a stale locator
artifact (one reviewed against an earlier, since-amended design candidate)
never authorizes T1C preservation, no matter how it resolved.

After the locator artifact reaches PASS and independent review PASS, *and*
that binding re-verification passes, a future V8G T1C preservation
execution resolves the private manifest path as follows, strictly before
its own preservation gate:

- repeat the same metadata-only enumeration and canonicalization contract
  of §2.1.1–§2.1.2 (same scope, same exclusions, same hashing scheme);
- compute `locator_path_sha256` for each freshly enumerated candidate —
  content is never read at this step either;
- require exactly one currently-enumerated path whose `locator_path_sha256`
  equals the frozen `selected_locator_path_sha256` recorded in the reviewed
  locator artifact (§2.1.4);
- zero or more than one such path is a `PRE_GATE` `BLOCK` — the separate
  fresh V8G T1C preservation gate is never consumed in that case.

Only once exactly one such path is confirmed does the separate, fresh V8G
T1C preservation one-shot gate apply, with its boundary unchanged from the
V8F/V8E precedent:

```text
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ
```

After that gate only: read the authorized T1C allocation through the
existing private dependency-injected bytes boundary; read the single
resolved partition manifest (the one path just confirmed, never a broader
scan); fully verify both artifacts and their bindings using the existing
verification chain; produce the V8G T1C preservation artifact. No broad
content scan, and no re-enumeration of candidates, is permitted inside the
preservation gate itself — path resolution is strictly a pre-gate,
metadata-only step, and content verification is strictly a post-gate,
single-path step.

If the resolved candidate's content later fails exact manifest verification
post-gate (self-hash mismatch, wrong implementation commit, wrong
provenance binding, or any other existing check failure): `BLOCK_CLOSED`,
no retry, following the same permanent-consumption discipline as §2.1.3.

### 2.2 Implementation boundary

A future V8G implementation must introduce study-scoped code: a fresh
V8G-namespaced locator-support module for §2.1.1–§2.1.4, and a separate
fresh V8G-namespaced T1C preservation module for §2.1.5, each independently
reviewed. Neither may edit `src/v8f_t1c_preservation.py`,
`src/v8e_t1c_preservation.py`, `src/v8d_t1c_preservation.py`, or any of
their historical tests in place — the same non-in-place-modification rule
V8F applied to the historical V7 transport source. Those historical modules
remain valid, unmodified evidence of their own predecessor studies; they
are never renamed to V8G and never substitute for V8G's own fresh
implementation or authority.

### 2.3 Stage-aware design authority binding

No frozen V8G design commit exists until `HUMAN_V8G_DESIGN_FREEZE` (§7)
completes. Every stage in §7's minimum stage order that occurs *before*
`HUMAN_V8G_DESIGN_FREEZE` — `V8G_LOCATOR_SUPPORT_IMPLEMENTATION`,
`HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE`,
`V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT`,
`V8G_T1C_PRESERVATION_SUPPORT_IMPLEMENTATION`,
`V8G_T1C_PRESERVATION_AUTHORITY_GATE`, `V8G_T1C_PRESERVATION_RECHECK`, and
`V8G_T2_PREFREEZE_PRESERVATION_RECHECK` — therefore cannot bind its
authority to a frozen design commit. This section is the exact, frozen
resolution of that gap; every other reference in this draft to authority
"bound to the exact V8G frozen design commit" means exactly what this
section defines, not a nonexistent pre-freeze frozen commit.

```text
prefreeze_stages = {
  V8G_LOCATOR_SUPPORT_IMPLEMENTATION,
  HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE,
  V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT,
  V8G_T1C_PRESERVATION_SUPPORT_IMPLEMENTATION,
  V8G_T1C_PRESERVATION_AUTHORITY_GATE,
  V8G_T1C_PRESERVATION_RECHECK,
  V8G_T2_PREFREEZE_PRESERVATION_RECHECK,
}

postfreeze_stages = {
  V8G_TRANSPORT_PRODUCTION_IMPLEMENTATION,
  V8G_T1C_AUTHORITY_BRIDGE_GATE,
  V8G_T1C_READINESS_HUMAN_GATE,
  EXECUTE_FIXED_V8G_T1C_TRANSPORT_READINESS,
  V8G_T1C_RAW_ACQUISITION_HUMAN_GATE,
  EXECUTE_V8G_T1C_RAW_ACQUISITION,
  SEPARATE_V8G_T1C_RESEARCH_OPENING_GATE,
  V8G_T1C_RESEARCH_OPENING,
  ... and every analogous T2 stage, once frozen per §7's closing note
}
```

For every stage in `prefreeze_stages`, authority binds to
`reviewed_v8g_design_candidate_commit`: the exact Git commit of *this*
design draft that underwent independent methodology review and was
explicitly approved as current for that specific stage. It never binds to
a "frozen design commit," because none exists yet at that point.

`reviewed_v8g_design_candidate_commit` is frozen per stage, not globally.
Different prefreeze stages may be reviewed and authorized against
different candidate commits over time, if this draft is amended and
independently re-reviewed between them. Each stage's own authorization
records exactly which candidate commit it was reviewed and authorized
against; it is never inferred, defaulted, or assumed equal to any other
stage's binding.

```text
design_candidate_staleness_rule:
  the instant this design draft's content changes (any new commit to
  V8G_PRIVATE_PARTITION_LOCATOR_SUCCESSOR_DESIGN_DRAFT.md), every existing
  reviewed_v8g_design_candidate_commit binding tied to the prior content
  becomes stale immediately.
  A stale candidate authority may not authorize, and may never be treated
  as authorizing, any stage reviewed against the amended (later) candidate.
  A materially amended candidate requires its own fresh independent review
  before it can authorize anything, for every stage whose authority it
  would newly cover.
```

For every stage in `postfreeze_stages` (and every later stage this draft
has not yet frozen), authority binds to the exact frozen
`v8g_frozen_design_commit` established at `HUMAN_V8G_DESIGN_FREEZE` — the
same binding V8F's own T1C preservation stage already used
(`reviewed_v8f_design_candidate_commit`), generalized here across every
V8G prefreeze stage and then superseded by frozen-commit binding once
`HUMAN_V8G_DESIGN_FREEZE` completes.

No prefreeze authority — no prefreeze human gate consumption, no prefreeze
one-shot receipt, no prefreeze `reviewed_v8g_design_candidate_commit`
binding — carries through `HUMAN_V8G_DESIGN_FREEZE` to become post-freeze
execution authority for any stage, under any circumstance. The one
exception is narrower than "carrying forward as authority": a prefreeze
stage's own durable public artifact (e.g. the locator artifact, the V8G
T1C preservation artifact) remains valid historical evidence that a later,
independently gated stage's own contract may explicitly require and read
as input (exactly as §2.1.5 requires V8G T1C preservation to read the
reviewed locator artifact) — but reading that historical evidence is never
itself an authorization, consumes no gate, and never substitutes for that
later stage's own fresh, independently authorized gate.

## 3. Unchanged inherited methodology

V8G inherits the V8F/V8E/V8D/V8C/V8B methodology unchanged except for the
one locator-stage-boundary change in §2. In particular, V8G does not
change:

- `JST_EXCHANGE_LOCAL_MIDNIGHT_REQUEST_BOUNDARY_V1` (V8F §2.1);
- the Yahoo provider/host;
- `interval=1d`; request headers, `events`, `includeAdjustedClose`;
- readiness sentinels `[0, 149, 299]`, sentinel count 3;
- the readiness window, exactly `2025-12-01` to `2025-12-08` (exclusive);
- maximum attempts=3; maximum retries=2; backoff=`[5, 30]`; jitter=false;
- the retry classifier;
- labels or target definitions;
- transaction costs or slippage;
- portfolio rules; the search space, stopping rules, promotion criteria,
  robustness rules;
- the historical research period;
- universe or partition definition, or T1C/T2 membership;
- the V8B DQ evidence policy and its exact thresholds
  (`POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE`,
  `invalid_fraction_threshold=1/252`,
  `max_consecutive_invalid_returned_rows=1`, `full_P_hist_check=true`,
  `test_years=2018..2025`,
  `calendar_missing_dates_are_not_malformed_returned_rows=true`,
  `threshold_failure_action=BLOCK_WHOLE_ACQUISITION`);
- research-opening rules.

No threshold recalibration, provider substitution, retry-policy change,
universe redraw, alternate partition, or stopping-rule change is permitted
by this draft.

## 4. Anti-overfitting constraints and interpretation

```text
locator_uses_prices_features_outcomes_profits=false
ticker_selected_or_substituted_for_performance=false
V8F_BLOCK_supplies_V8G_PASS_evidence=false
locator_success_establishes_provenance_only=true
locator_success_establishes_data_quality=false
locator_success_establishes_readiness=false
locator_success_establishes_strategy_validity=false
locator_success_establishes_profitability=false
future_profitability_remains_unestablished=true
```

The `PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_V1` contract is a predeclared,
fixed-before-execution content-addressing scheme, stated in full in this
frozen draft. It is not tuned, grid-searched, or selected by testing
alternative canonicalizations or scopes against the V8F 0-match outcome or
any other real execution result. V8F's `V8F_LOCATOR_ZERO_MATCHING_CANDIDATES`
BLOCK motivates why this successor stage exists (the real production
manifest path was never publicly recorded and the prior locator's candidate
snapshot for that one execution did not include it); it supplies no V8G
evidence and authorizes no V8G execution. A locator PASS establishes only
that exactly one currently-enumerated file's content is byte-identical (up
to SHA-256 collision, i.e. cryptographically certain) to the historically
authorized partition manifest. It says nothing about that data's quality,
about transport readiness, about any strategy's validity, or about
profitability, historical or future.

## 5. Mandatory future synthetic tests

Before any V8G locator-support or T1C-preservation production implementation
review, synthetic tests must cover at minimum:

- `canonical_path_text`/`locator_path_sha256` computed exactly per §2.1.2
  for representative synthetic paths, including mixed-case and
  backslash/forward-slash inputs that must canonicalize identically;
- `candidate_set_sha256` computed exactly per the frozen serialization for a
  representative sorted synthetic hash list, including the domain-separation
  and count lines;
- a duplicate `locator_path_sha256` within an otherwise-deduplicated
  candidate set is rejected, never silently merged;
- zero candidates pre-gate => `BLOCK`, `gate_consumption=0`,
  `candidate_content_reads=0`;
- no candidate content is read before the locator gate's durable receipt
  exists;
- the exact authorization grammar accepts the exact tuple of
  `reviewed_v8g_design_candidate_commit` / `reviewed_locator_support_implementation_sha`
  / `expected_partition_manifest_sha256` / `expected_partition_implementation_commit`
  it was constructed from, and only that tuple;
- any single component mismatch (wrong commit, wrong manifest SHA, wrong
  length, wrong case, non-hex character) rejects the authorization
  `PRE_GATE`, never post-gate;
- the raw `authorization_identity` string never appears in any log,
  exception message, receipt, or artifact — only `authorization_identity_sha256`
  does;
- `locator_gate_receipt_key_sha256` is deterministic: recomputing
  `receipt_key_material` from the same repository/study/gate literals
  always yields the same digest;
- changing only the authorization identity, only the reviewed design
  candidate commit, or only the reviewed locator-support implementation SHA
  — independently or in any combination — never changes
  `locator_gate_receipt_key_sha256`;
- a second locator execution attempted after a receipt already exists at
  the fixed key `BLOCK`s even when presented with a fresh, otherwise-valid
  authorization identity, a newly reviewed design candidate, or a newly
  reviewed implementation;
- a malformed existing receipt at the fixed key `BLOCK`s and is never
  deleted, repaired, or overwritten by a subsequent attempt;
- exactly one exact synthetic match among several non-matching/malformed
  candidates => locator PASS;
- zero exact matches post-gate => `BLOCK_CLOSED`;
- more than one exact match post-gate => `BLOCK_CLOSED`;
- a candidate with a self-declared `manifest_sha256` equal to the expected
  value but whose recomputed canonical hash disagrees => rejected, never
  trusted from the declared field alone;
- a post-gate locator failure leaves `locator_authorization_consumed=true`
  with no retry, no reset, and a second execution attempt fails closed;
- a locator gate/artifact whose `reviewed_v8g_design_candidate_commit` or
  `reviewed_locator_support_implementation_sha` disagrees with the
  currently authorized/reviewed values is rejected pre-gate (for the gate)
  or by the artifact verifier (for the artifact), never treated as PASS;
- a stale locator artifact (reviewed against an earlier, since-amended
  design candidate per §2.3's staleness rule) is rejected by V8G T1C
  preservation's own binding re-verification before manifest-path
  resolution is even attempted;
- the §2.1.5 pre-preservation-gate path-hash resolution: zero or more than
  one currently-enumerated match for the frozen `selected_locator_path_sha256`
  => `PRE_GATE` `BLOCK`, preservation gate not consumed;
- exactly one currently-enumerated match, followed by a genuine post-gate
  content-verification failure on that single resolved path => `BLOCK_CLOSED`,
  no retry;
- no ticker identity, candidate path, block assignment, or raw manifest byte
  appears in any safe result, log, or exception message, in any of the above;
- `network_requests=0` and no real filesystem-wide discovery outside a
  synthetic/temporary fixture root, for every test above.

All tests must use synthetic or fake data and temporary state only. No test
may use real network access, private identities, or private production
data.

## 6. Privacy-safe boundary

Public V8G locator and preservation design, gate, and evidence artifacts
may expose only:

- the fixed contract identifiers (`PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_V1`
  and the frozen path/candidate-set hashing scheme);
- safe nonnegative integer counts (`candidate_count`, etc.);
- safe SHA-256 hex digests explicitly listed as safe in §2.1.4;
- safe booleans and enums (`locator_result`, `research_opened`, etc.);
- fixed threshold and provenance constants inherited unchanged from V8F/V8E;
- Git commit/blob provenance.

No ticker identity, URL, raw payload, private path, `canonical_path_text`,
price, feature, or outcome may appear in public evidence. No private
identity is inspected or surfaced by this design task.

## 7. Authority: fresh V8G namespace, inserted locator stage

V8G reuses the full V8F/V8E stage-and-gate discipline mechanically, with the
namespace-substitution rule V8F used relative to V8E, and inserts the new
locator-establishment stage before T1C preservation:

```text
V8G_<CURRENT_STUDY_TOKEN> -> replaces the V8F_<CURRENT_STUDY_TOKEN> analogue
study=V8F_HISTORICAL_RESEARCH -> study=V8G_HISTORICAL_RESEARCH
reviewed_v8f_design_candidate_commit -> reviewed_v8g_design_candidate_commit
v8f_frozen_design_commit -> v8g_frozen_design_commit
```

This applies to every current-study `schema_version`, `artifact_role`,
gate, review, stage, receipt, and freeze-status literal, including the
minimum preserved stage order:

```text
CREATE_V8G_DESIGN_DRAFT
INDEPENDENT_V8G_DESIGN_REVIEW

V8G_LOCATOR_SUPPORT_IMPLEMENTATION
INDEPENDENT_V8G_LOCATOR_SUPPORT_REVIEW

HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE
V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT
INDEPENDENT_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_REVIEW

only if locator establishment PASS and its independent review PASS:
V8G_T1C_PRESERVATION_SUPPORT_IMPLEMENTATION
INDEPENDENT_V8G_T1C_PRESERVATION_SUPPORT_REVIEW (as required)

V8G_T1C_PRESERVATION_AUTHORITY_GATE
V8G_T1C_PRESERVATION_RECHECK
INDEPENDENT_V8G_T1C_PRESERVATION_RECHECK_REVIEW

V8G_T2_PREFREEZE_PRESERVATION_RECHECK
INDEPENDENT_V8G_T2_PREFREEZE_PRESERVATION_RECHECK_REVIEW

V8G_DESIGN_FINALIZED
HUMAN_V8G_DESIGN_FREEZE

V8G_TRANSPORT_PRODUCTION_IMPLEMENTATION
INDEPENDENT_V8G_PRODUCTION_IMPLEMENTATION_REVIEW

V8G_T1C_AUTHORITY_BRIDGE_GATE
CREATE_V8G_T1C_AUTHORITY_BRIDGE
INDEPENDENT_V8G_T1C_AUTHORITY_BRIDGE_REVIEW
V8G_T1C_READINESS_HUMAN_GATE
EXECUTE_FIXED_V8G_T1C_TRANSPORT_READINESS
READ_ONLY_V8G_T1C_READINESS_TRANSPORT_AUDIT_VERIFICATION

only if readiness PASS and its audit verification PASS:
V8G_T1C_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_V8G_T1C_RAW_ACQUISITION
READ_ONLY_V8G_T1C_ACQUISITION_ARTIFACT_VERIFICATION
READ_ONLY_V8G_T1C_ACQUISITION_TRANSPORT_AUDIT_VERIFICATION

only if raw acquisition PASS and both acquisition verifications PASS:
SEPARATE_V8G_T1C_RESEARCH_OPENING_GATE
V8G_T1C_RESEARCH_OPENING
```

This is the *minimum* frozen order; it does not itself authorize, and is
not itself, the full T2 stage sequence, which remains governed by the same
inherited V8F/V8E stage discipline and must be frozen in the same manner
before any T2-stage work begins. No stage above may be skipped or silently
merged. No locator establishment substitutes for T1C preservation
verification; no T1C preservation verification substitutes for readiness;
no readiness verification substitutes for acquisition verification; no
acquisition verification substitutes for a research-opening authorization.

Every stage above binds its authority exactly per §2.3:
`reviewed_v8g_design_candidate_commit` before `HUMAN_V8G_DESIGN_FREEZE`,
the frozen `v8g_frozen_design_commit` from `HUMAN_V8G_DESIGN_FREEZE`
onward. No stage's authority is ever inferred from, defaulted from, or
assumed equal to, another stage's binding.

### 7.1 No V8F authority carries forward

No V8F human authorization, gate, preservation result, locator result,
implementation review, freeze approval, allocation-authority bridge, or
readiness receipt authorizes any V8G stage. Every V8G prerequisite requires
fresh, V8G-specific authorization, bound per §2.3 to the reviewed design
candidate commit applicable to that stage before `HUMAN_V8G_DESIGN_FREEZE`,
and to the exact frozen V8G design commit from `HUMAN_V8G_DESIGN_FREEZE`
onward. Historical V8F, V8E, V8D, V8C, V8B, and V8 identifiers (the
original partition hash and implementation commit, trust-pin
commits/blobs, T1C/T2 membership hashes, and the V8F terminal adjudication
commit/blob cited in §1) remain historical evidence only; they are never
renamed to V8G and never substitute for V8G authority.

### 7.2 Inherited security semantics

The following semantics are copied without weakening or reinterpretation,
per `AI_REAL_EXECUTION_RUNBOOK.md` and the V8F/V8E precedent:

- every locator, preservation, authority, readiness, acquisition, and
  research-opening gate is one-shot;
- each receipt is durably published with flush/fsync and exclusive
  no-overwrite rules;
- `consumed=true` and `consumption_count=1` are required for a valid PASS;
- authorization reset, deletion, replay, and reuse are prohibited;
- a failed or malformed receipt is fail-closed and does not restore
  authorization;
- exact receipt-key and receipt-byte bindings are independently
  recomputed;
- raw authorization identities, ticker identities, private paths
  (including `canonical_path_text`), raw payloads, prices, features, and
  outcomes remain prohibited from public evidence;
- missing, duplicate, extra, malformed, mismatched, or unverifiable
  evidence is `BLOCK`, never an implicit PASS.

## 8. Design task scope boundary

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
```

```text
design_finalized=false
human_design_freeze_complete=false
locator_support_implemented=false
t1c_preservation_support_implemented=false
approval_artifact_created=false
network_access_authorized=false
private_data_access_authorized=false
human_gate_consumed=false
```

This draft does not implement the locator or the preservation producer or
verifier, create an approval or freeze artifact, enumerate or read any
private path or candidate, inspect any ticker identity, access Yahoo or
JPX, consume any gate, acquire raw data, or open research. Future V8G
locator-support and T1C-preservation-support implementation tasks require
their own independent exact-SHA reviews before any real private
enumeration, private read, or gate consumption begins.
