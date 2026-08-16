# V8E Independently Verifiable Data-Quality Evidence Successor Design Draft

This is a successor-study design draft only. It creates no implementation
authority, network authority, private-data authority, allocation authority,
human authorization, gate receipt, or research-opening authority.

## 1. Study identity and predecessor disposition

```text
study=V8E_HISTORICAL_RESEARCH
study_type=SUCCESSOR_STUDY
predecessor=V8D_HISTORICAL_RESEARCH
predecessor_terminal_commit=b8f8d0d500d349ccaa5d3e49294f351dc53ea7e8
normative_inherited_design=V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md
normative_inherited_design_commit=eda657cde2383718d986c4c4bfaae794784fe04d
```

V8E is a new study identity. It is not a V8D retry, amendment, continuation,
or repair under the V8D frozen design. The V8D terminal status is
`BLOCK_CLOSED` with failure class `DESIGN_AUDITABILITY_FAILURE`. That status
is design-audit evidence only; it is not strategy, profitability, T1C, or T2
data-quality evidence.

V8E changes exactly one methodological contract: the frozen V8D
`DATA_QUALITY_GATE_FAILURE` audit evidence contract is replaced by the
independently re-derivable privacy-safe contract in §4. The raw data-quality
decision, thresholds, acquisition policy, and all research methodology remain
unchanged.

No V8D human authorization, gate, preservation result, implementation review,
or freeze approval authorizes V8E. All V8E authority must be fresh and
V8E-specific.

## 2. Unchanged inherited methodology

V8E inherits the V8D/V8C/V8B methodology unchanged except for the one audit
evidence-contract change stated above. In particular, V8E does not change:

- the historical period;
- labels or target definition;
- walk-forward, causality, or leakage rules;
- transaction costs or slippage;
- portfolio rules;
- search space;
- stopping rules;
- promotion criteria;
- robustness rules;
- partition or universe definition;
- T1C or T2 membership;
- Yahoo as provider;
- the canonical parser;
- the retry classifier or retry policy;
- readiness sentinels or readiness window;
- sample inclusion or exclusion; or
- research-opening rules.

The frozen data-quality policy remains exactly:

```text
policy=POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE
invalid_fraction_threshold=1/252
max_consecutive_invalid_returned_rows=1
full_P_hist_check=true
test_years=2018..2025
calendar_missing_dates_are_not_malformed_returned_rows=true
threshold_failure_action=BLOCK_WHOLE_ACQUISITION
```

The fraction comparison is an exact integer comparison. No floating-point
comparison, threshold recalibration, provider substitution, retry change,
redraw, alternate partition, or stopping-rule change is permitted.

## 3. Conditional preservation and fresh authority

The exact V8D T1C and T2 memberships may be considered for V8E only through
fresh V8E preservation checks. The existing V8D T1C preservation record and
V8D T2 prefreeze record are historical evidence; neither is V8E authority.

Reuse is conditional on fresh, privacy-safe V8E verification of the exact
existing memberships, original V8 provenance, and absence of intervening
acquisition, opening, feature, outcome, or membership-reassignment evidence.
No redraw, T3 substitution, alternate slice, or automatic replacement is
allowed. If exact preservation cannot be established, V8E stops for a new
methodology decision.

V8E requires fresh, separately reviewed and authorized stages for:

- T1C preservation authority, recheck, and independent review before freeze;
- T2 prefreeze preservation authority, recheck, and independent review before
  freeze;
- V8E design review and human design freeze;
- V8E production implementation review;
- T1C and T2 authority bridges and their independent reviews;
- T1C and T2 readiness gates and transport-audit verification;
- T1C and T2 acquisition gates and artifact/transport verification; and
- T1C and T2 research-opening gates.

The preservation, readiness, acquisition, transport-audit, artifact-
verification, and research-opening stages are distinct and
non-substitutable. A V8D PASS cannot satisfy a V8E prerequisite.

This design task itself performs no private read, network request, gate
consumption, allocation, raw acquisition, or research opening.

## 3A. Successor-authority rebinding

Except for the explicitly replaced §4 `DATA_QUALITY_GATE_FAILURE` evidence
contract, every inherited V8D governance and security contract is semantically
unchanged. V8E does not leave token selection to an execution agent. The
following substitutions are mechanical and exhaustive for current-study
authority.

### 3A.1 Current-study identity rules

```text
study=V8D_HISTORICAL_RESEARCH
  -> study=V8E_HISTORICAL_RESEARCH

reviewed_design_candidate_commit
  -> reviewed_v8e_design_candidate_commit

v8d_frozen_design_commit
  -> v8e_frozen_design_commit
```

Every current-study design binding must resolve to the exact one reviewed
V8E design-candidate SHA before freeze and to the exact one frozen V8E design
SHA after freeze. A branch name, current HEAD, latest HEAD, working-tree
state, moving tag, or unreviewed commit is not an authority substitute.

Where an inherited freeze-approval schema retains the generic field name
`frozen_design_git_commit`, its value is the exact V8E frozen design SHA. The
current-study candidate field is always
`reviewed_v8e_design_candidate_commit`; the current-study bridge field is
always `v8e_frozen_design_commit`. A current-study receipt, artifact, review,
bridge, gate, readiness, acquisition, or research-opening binding referring
to a V8D current-study design SHA is invalid for V8E.

### 3A.2 Namespace substitution and historical values

For a token that identifies the current study, the exact namespace rule is:

```text
V8D_<CURRENT_STUDY_TOKEN>
  -> V8E_<CURRENT_STUDY_TOKEN>
```

This applies to current-study `schema_version`, `artifact_role`, gate,
review, stage, receipt, and freeze-status literals. Examples include:

```text
V8D_T1C_PRESERVATION_GATE_RECEIPT_V1
  -> V8E_T1C_PRESERVATION_GATE_RECEIPT_V1
V8D_T1C_PRESERVATION_RECHECK_V1
  -> V8E_T1C_PRESERVATION_RECHECK_V1
V8D_T1C_ALLOCATION_AUTHORITY_BRIDGE_V1
  -> V8E_T1C_ALLOCATION_AUTHORITY_BRIDGE_V1
V8D_T2_AUTHORITY_BRIDGE_V1
  -> V8E_T2_AUTHORITY_BRIDGE_V1
V8D_DESIGN_FINALIZED
  -> V8E_DESIGN_FINALIZED
HUMAN_V8D_DESIGN_FREEZE
  -> HUMAN_V8E_DESIGN_FREEZE
```

The same rule applies to the V8D-specific preservation, authority-bridge,
readiness, acquisition, transport-audit, artifact-verification, and
research-opening stage names listed in the V8D sequence. The §4 DQ contract
is not a namespace-only substitution: it is the one explicitly replaced
contract and is governed solely by the exact V8E union defined there.

The substitution rule MUST NOT rename, repin, or rewrite historical
provenance. Exact values referring to original V8, V8B, V8C, or terminal V8D
evidence remain unchanged, including original partition hashes and
implementation commits, V8C terminal and trust-pin identifiers, source V8C
artifact identifiers, and the V8D terminal commit and adjudication. The
normative inherited design remains the exact historical
`V8D_TRANSPORT_AUDIT_SUCCESSOR_DESIGN_DRAFT.md` at commit
`eda657cde2383718d986c4c4bfaae794784fe04d`; its filename and SHA are not
renamed to V8E. Historical provenance is evidence, never V8E authority.

The repository value remains exactly:

```text
repository=ta1k1-arakawa/stock-analyzer
```

### 3A.3 Inherited security semantics that do not change

The following semantics are copied without weakening or reinterpretation:

- every preservation, authority, readiness, acquisition, and research-opening
  gate is one-shot;
- the receipt is durably published with the inherited flush/fsync and
  exclusive no-overwrite rules;
- the consumption boundary remains
  `IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ`;
- `consumed=true` and `consumption_count=1` are required for a valid PASS;
- authorization reset, deletion, replay, and reuse are prohibited;
- a failed or malformed receipt is fail-closed and does not restore
  authorization;
- exact receipt-key and receipt-byte bindings are independently recomputed;
- raw authorization identities, ticker identities, private paths, raw
  payloads, prices, features, and outcomes remain prohibited from public
  evidence; and
- missing, duplicate, extra, malformed, mismatched, or unverifiable evidence
  is BLOCK, never an implicit PASS.

V8E-specific names do not create an additional attempt, reset an old gate, or
convert a historical V8D receipt into V8E authority.

### 3A.4 Frozen V8E T1C prefreeze preservation authority

The next T1C preservation authority contract is frozen as follows:

```text
gate=HUMAN_V8E_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE
authorization_identity="V8E_HUMAN_AUTHORIZE_T1C_PRESERVATION_VERIFY_AT_"
                      + reviewed_v8e_design_candidate_commit
                      + "_FOR_"
                      + allocation_artifact_self_hash
```

The exact inherited allocation binding remains:

```text
allocation_artifact_self_hash=16e3c2b026e4aaf4382d88e5bce25c2a52f0bb7ebbc03838679c3c6e84daaf7c
```

The identity grammar is evaluated as one exact string with no whitespace or
normalization changes. Its raw value is never recorded or exposed; only its
SHA-256 may appear in privacy-safe evidence. The gate authorizes exactly one
minimum read-only verification of the existing T1C allocation and original
V8 provenance. It authorizes no ticker display, raw OHLCV access, feature or
outcome access, network, allocation, redraw, substitution, or research
opening. It is consumed durably immediately before the first private byte
read.

The V8E durable receipt is the V8D preservation receipt contract with only
the current-study rebinding below. This is an exact field set: no extra or
omitted fields are permitted.

```text
schema_version=V8E_T1C_PRESERVATION_GATE_RECEIPT_V1
study=V8E_HISTORICAL_RESEARCH
artifact_role=T1C_PRESERVATION_PRIVATE_GATE_RECEIPT
gate=HUMAN_V8E_T1C_PRESERVATION_PRIVATE_VERIFICATION_GATE
reviewed_v8e_design_candidate_commit
authorization_identity_sha256
authorized_allocation_artifact_self_hash
consumed=true
consumption_count=1
consumption_boundary=IMMEDIATELY_BEFORE_FIRST_PRIVATE_BYTE_READ
consumption_timestamp_utc
```

`consumption_timestamp_utc` must be a canonical UTC timestamp in exactly one
of these forms:

```text
YYYY-MM-DDTHH:MM:SSZ
YYYY-MM-DDTHH:MM:SS.ffffffZ
```

The timestamp has exact uppercase `T` and `Z`, zero UTC offset, valid
calendar/time values, and either no fraction or exactly six fractional
digits. No whitespace, offset spelling, lowercase `z`, naive timestamp,
trailing text, or shortened fraction is valid. No chronology or age policy is
added.

There is no receipt self-hash inside the receipt:

```text
receipt_self_hash_inside_receipt=PROHIBITED
```

The receipt key is constructed exactly as in the inherited contract, with the
V8E gate and candidate field:

```text
gate_receipt_key_sha256=SHA256(
  repository + "|" + gate + "|" +
  reviewed_v8e_design_candidate_commit + "|" +
  authorization_identity_sha256 + "|" +
  authorized_allocation_artifact_self_hash
)
```

The input is the exact UTF-8 string, with no added whitespace or newline, and
`repository` is exactly `ta1k1-arakawa/stock-analyzer`. The receipt storage
key or filename is derived from that hash. An independent reader recomputes
the key from the validated receipt and requires equality with the actual
storage key; a copied receipt under another key is BLOCK.

After durable publication, the exact receipt bytes may be independently
hashed as `gate_receipt_bytes_sha256`. That byte hash is external to the
receipt and is not a replacement for receipt validation. Publication remains
flush/fsync, atomic, exclusive, and no-overwrite. If the deterministic receipt
destination already exists, a second consumption attempt MUST BLOCK, even
when the existing receipt bytes are identical and strictly valid. Existing
receipt bytes may be read and independently validated for read-only review or
provenance purposes only. Such read-only validation MUST NOT make another
consume operation succeed and MUST NOT authorize another private-byte read.
No reset, deletion, replay, or reuse of the same authorization identity is
allowed. A failure or crash after durable receipt consumption does not restore
authority. A later private preservation verification requires a genuinely
fresh explicit human authorization under the frozen V8E rules; it is not
recovery or idempotent replay of the consumed identity.

The V8E T1C preservation artifact and independent review use the corresponding
current-study bindings:

```text
schema_version=V8E_T1C_PRESERVATION_RECHECK_V1
study=V8E_HISTORICAL_RESEARCH
reviewed_v8e_design_candidate_commit
preservation_recheck_result=PASS
```

All inherited safe allocation/provenance fields and frozen preservation
conditions remain unchanged, including the exact allocation self-hash above;
historical `source_v8c_*` values remain historical exact values. Its
independent review must resolve the exact artifact commit/blob, independently
validate this V8E receipt and key, compute the exact receipt byte hash, and
require the same V8E candidate SHA. Producer-declared PASS alone is never
authority.

### 3A.5 Authority bridges and later current-study contracts

The same mechanical rebinding applies to the later current-study authority
contracts. Their inherited schemas and fields remain exact except for the
current-study identity substitutions in this section:

```text
V8E_T1C_ALLOCATION_AUTHORITY_BRIDGE_V1
study=V8E_HISTORICAL_RESEARCH
artifact_role=T1C_ALLOCATION_AUTHORITY_BRIDGE
v8e_frozen_design_commit
human_gate=V8E_HUMAN_AUTHORIZE_T1C_AUTHORITY_BRIDGE_AT_
            + v8e_frozen_design_commit
            + "_FOR_"
            + authorized_allocation_artifact_self_hash

V8E_T2_PREFREEZE_PRESERVATION_RECHECK_V1
study=V8E_HISTORICAL_RESEARCH
reviewed_v8e_design_candidate_commit
preservation_recheck_result=PASS

V8E_T2_AUTHORITY_BRIDGE_V1
study=V8E_HISTORICAL_RESEARCH
artifact_role=T2_AUTHORITY_BRIDGE
v8e_frozen_design_commit
human_gate=V8E_HUMAN_AUTHORIZE_T2_AUTHORITY_BRIDGE_AT_
            + v8e_frozen_design_commit
            + "_FOR_"
            + expected_t2_ticker_list_sha256
```

The T1C and T2 bridges retain the exact original V8 authority/provenance
values, exact inherited membership hashes and counts, and safe preservation
artifact commit/blob bindings; only the current V8E candidate/frozen design,
study, namespace, and fresh V8E gate literals are rebound. The T2 prefreeze
and later point-of-use checks remain distinct and neither authorizes the other.
All V8E readiness, acquisition, transport-audit, artifact-verification, and
research-opening receipts follow the same rule: current-study V8E namespace
and exact V8E design binding, inherited schema semantics, and fresh V8E gate;
historical source identifiers remain byte-for-byte historical values.

## 4. V8E DQ failure evidence model

`DATA_QUALITY_GATE_FAILURE` uses a discriminated evidence union. The audit
record has one exact union variant. The discriminator is not proof by itself;
the verifier must validate the exact variant fields and independently derive
the failure predicate.

Every variant has exactly these two common fields:

```text
detector_source=V8E_DQ_GATE
failure_kind=<exact enum defined below>
```

No `schema_version`, producer status, named-condition assertion, message,
URL, ticker, payload, price, private path, or other field may be added to the
variant evidence. Any surrounding audit envelope remains governed by the
inherited audit schema; the union body has the exact field sets below.

### 4.1 Variant A: row or schema structure invalid

```text
failure_kind=ROW_STRUCTURE_INVALID
detector_source=V8E_DQ_GATE
valid_price_rows_is_list: bool
invalid_price_rows_is_list: bool
valid_price_row_count: nonnegative integer or null
invalid_price_row_count: nonnegative integer or null
valid_price_rows_nonempty: bool
trading_date_fields_valid: bool
```

The exact null/count consistency rules are:

1. If `valid_price_rows_is_list=true`,
   `valid_price_row_count` is a nonnegative integer and
   `valid_price_rows_nonempty == (valid_price_row_count > 0)`.
2. If `valid_price_rows_is_list=false`,
   `valid_price_row_count=null` and `valid_price_rows_nonempty=false`.
3. If `invalid_price_rows_is_list=true`,
   `invalid_price_row_count` is a nonnegative integer.
4. If `invalid_price_rows_is_list=false`,
   `invalid_price_row_count=null`.
5. Boolean values are not accepted as integer counts.

The verifier derives `ROW_STRUCTURE_INVALID` if and only if at least one of
the following is true:

```text
valid_price_rows_is_list=false
invalid_price_rows_is_list=false
valid_price_rows_nonempty=false
trading_date_fields_valid=false
```

The verifier rejects the variant if any null/count rule fails, if the
declared `failure_kind` is not exact, if an extra or missing field exists, or
if all four failure predicates are false. The verifier does not trust the
producer's discriminator.

### 4.2 Threshold variants: exact common schema

Both threshold variants use one exact field set:

```text
detector_source=V8E_DQ_GATE
failure_kind=INVALID_FRACTION_THRESHOLD_EXCEEDED
  or CONSECUTIVE_INVALID_THRESHOLD_EXCEEDED
scope=FULL_P_HIST or TEST_YEAR
test_year=null or integer 2018..2025
valid_returned_row_count: nonnegative integer
invalid_returned_row_count: nonnegative integer
returned_row_count: nonnegative integer
max_consecutive_invalid_returned_rows_observed: nonnegative integer
invalid_fraction_threshold_numerator=1
invalid_fraction_threshold_denominator=252
max_consecutive_invalid_returned_rows_threshold=1
trading_date_fields_valid=true
```

The exact type and consistency requirements are:

- `scope=FULL_P_HIST` requires `test_year=null`.
- `scope=TEST_YEAR` requires an integer `test_year` in 2018 through 2025.
- Boolean values are not accepted as integer counts or years.
- All four counts are nonnegative integers.
- `returned_row_count` equals
  `valid_returned_row_count + invalid_returned_row_count` exactly.
- The three threshold constants have exactly the values and integer types
  shown above.
- `trading_date_fields_valid` must be the boolean `true`.
- No field outside this exact set is accepted, and no listed field may be
  omitted.

The invalid-fraction failure predicate is derived with integer arithmetic
only:

```text
invalid_returned_row_count * 252
>
returned_row_count * 1
```

`INVALID_FRACTION_THRESHOLD_EXCEEDED` is valid if and only if that predicate
is true and every schema/type/consistency rule passes.

`CONSECUTIVE_INVALID_THRESHOLD_EXCEEDED` is valid if and only if:

```text
max_consecutive_invalid_returned_rows_observed > 1
```

and every schema/type/consistency rule passes. Because fraction checking has
precedence within a scope, a consecutive-invalid variant is invalid if the
integer invalid-fraction predicate is also true for that same scope. A
fraction variant may have either observed consecutive value because the
fraction condition is checked first.

The threshold union cannot represent a row-structure failure. The producer
must complete the row-structure checks first, and the independent acquisition
artifact verifier must establish that the threshold evidence is based on
structurally valid canonical parsed evidence.

## 5. Deterministic producer and verifier precedence

The producer and verifier use this exact scope order:

```text
1. row/schema validity
2. FULL_P_HIST
3. TEST_YEAR 2018
4. TEST_YEAR 2019
5. TEST_YEAR 2020
6. TEST_YEAR 2021
7. TEST_YEAR 2022
8. TEST_YEAR 2023
9. TEST_YEAR 2024
10. TEST_YEAR 2025
```

Within every threshold scope, the order is:

```text
1. invalid-fraction check
2. consecutive-invalid check
```

The first failing condition in that total order is the sole emitted failure
evidence. If no condition fails, the acquisition DQ result is unchanged and
may proceed under the inherited rules. If any condition fails, the result is
`BLOCK_WHOLE_ACQUISITION`.

The transport-audit verifier validates the exact union schema, local integer
predicate, scope/year constraints, constant values, count equality, and
discriminator agreement. It rejects any producer-declared failure whose
evidence does not prove the declared predicate. It never treats
`failure_kind` alone as authority.

For an acquisition execution, the later acquisition-artifact verifier
recomputes the relevant canonical parsed-evidence counts and maximum
consecutive-invalid-row value. It requires those independently derived
values to equal the DQ audit evidence and requires the evidence to be the
first failure in the frozen order above. It must also recheck the applicable
scope (`FULL_P_HIST` or the exact fixed test year) and the exact fixed
threshold constants before any research-opening gate can be reached.

Calendar-missing dates remain outside the malformed returned-row counts and
cannot be used to manufacture a DQ failure.

## 6. Privacy-safe boundary

The DQ union may expose only:

- the fixed detector source;
- the exact failure kind;
- safe booleans;
- nonnegative safe counts;
- fixed threshold constants;
- `FULL_P_HIST` or a fixed year from 2018 through 2025; and
- no more than the exact fields defined in §4.

It must never contain a ticker identity, URL, raw payload, price or OHLCV
value, raw exception message, private path, exact private membership content,
or a mapping from safe DQ evidence to a ticker. No private identity is
inspected or surfaced by this design task.

## 7. Preserved stage order and separation

V8E preserves the V8D stage ordering and separation, with each stage renamed
and bound to the exact V8E design candidate:

```text
CREATE_V8E_DESIGN_DRAFT
INDEPENDENT_V8E_DESIGN_REVIEW

V8E_T1C_PRESERVATION_AUTHORITY_GATE
V8E_T1C_PRESERVATION_RECHECK
INDEPENDENT_V8E_T1C_PRESERVATION_RECHECK_REVIEW

V8E_T2_PREFREEZE_PRESERVATION_RECHECK
INDEPENDENT_V8E_T2_PREFREEZE_PRESERVATION_RECHECK_REVIEW

V8E_DESIGN_FINALIZED
HUMAN_V8E_DESIGN_FREEZE

V8E_TRANSPORT_AUDIT_IMPLEMENTATION
INDEPENDENT_V8E_PRODUCTION_IMPLEMENTATION_REVIEW

V8E_T1C_AUTHORITY_BRIDGE_GATE
CREATE_V8E_T1C_AUTHORITY_BRIDGE
INDEPENDENT_V8E_T1C_AUTHORITY_BRIDGE_REVIEW
V8E_T1C_READINESS_HUMAN_GATE
EXECUTE_FIXED_V8E_T1C_TRANSPORT_READINESS
READ_ONLY_V8E_T1C_READINESS_TRANSPORT_AUDIT_VERIFICATION

only if readiness PASS and its audit verification PASS:
V8E_T1C_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_V8E_T1C_RAW_ACQUISITION
READ_ONLY_V8E_T1C_ACQUISITION_ARTIFACT_VERIFICATION
READ_ONLY_V8E_T1C_ACQUISITION_TRANSPORT_AUDIT_VERIFICATION

only if raw acquisition PASS and both acquisition verifications PASS:
SEPARATE_V8E_T1C_RESEARCH_OPENING_GATE
V8E_T1C_RESEARCH_OPENING

V8E_T2_AUTHORITY_BRIDGE_GATE
CREATE_V8E_T2_AUTHORITY_BRIDGE
INDEPENDENT_V8E_T2_AUTHORITY_BRIDGE_REVIEW
V8E_T2_READINESS_HUMAN_GATE
EXECUTE_FIXED_V8E_T2_TRANSPORT_READINESS
READ_ONLY_V8E_T2_READINESS_TRANSPORT_AUDIT_VERIFICATION

only if T2 readiness PASS and its audit verification PASS:
READ_ONLY_V8E_T2_POINT_OF_USE_PRESERVATION_RECHECK
INDEPENDENT_V8E_T2_POINT_OF_USE_PRESERVATION_RECHECK_REVIEW

only if both T2 point-of-use preservation stages PASS:
V8E_T2_RAW_ACQUISITION_HUMAN_GATE
EXECUTE_V8E_T2_RAW_ACQUISITION
READ_ONLY_V8E_T2_ACQUISITION_ARTIFACT_VERIFICATION
READ_ONLY_V8E_T2_ACQUISITION_TRANSPORT_AUDIT_VERIFICATION

only if T2 raw acquisition PASS and both acquisition verifications PASS:
SEPARATE_V8E_T2_RESEARCH_OPENING_GATE
V8E_T2_RESEARCH_OPENING
```

No readiness verification substitutes for acquisition verification. No
acquisition verification substitutes for a research-opening authorization.
The T2 prefreeze and point-of-use preservation checkpoints remain distinct;
the point-of-use checkpoint remains immediately before the T2 acquisition
gate and remains safe committed/audit/provenance evidence only. No T2
private manifest read, ticker identity inspection, network request, raw
acquisition, or research opening occurs in this design task.

## 8. Fresh V8E governance and exact-SHA freeze

Every V8E prerequisite binds to one exact design-candidate commit. The
candidate identity is:

```text
design_candidate_binding=EXACT_ONE_40_HEX_GIT_COMMIT_SHA
moving_branch_binding=INVALID
latest_HEAD_binding=INVALID
working_tree_binding=INVALID
reviewed_v8e_design_candidate_commit=the exact commit containing this V8E draft
```

The V8E independent design review, T1C preservation recheck and review, and
T2 prefreeze preservation recheck and review must all name the same exact
candidate SHA and must all PASS before `V8E_DESIGN_FINALIZED` or the human
design freeze. V8D approvals and V8D preservation records cannot be carried
forward as V8E PASS evidence.

The later V8E freeze approval is a separate, non-self-referential,
privacy-safe artifact. Its exact minimum field set is:

```text
schema_version=V8E_DESIGN_FREEZE_APPROVAL_V1
study=V8E_HISTORICAL_RESEARCH
frozen_design_git_commit
design_document=V8E_DQ_EVIDENCE_SUCCESSOR_DESIGN_DRAFT.md
final_independent_design_review_result=PASS
final_independent_design_review_commit
t1c_preservation_recheck_result=PASS
t1c_preservation_recheck_design_commit
t1c_preservation_independent_review_result=PASS
t1c_preservation_independent_review_design_commit
t2_prefreeze_preservation_recheck_result=PASS
t2_prefreeze_preservation_recheck_design_commit
t2_prefreeze_preservation_independent_review_result=PASS
t2_prefreeze_preservation_independent_review_design_commit
approval_status=APPROVED
human_gate=HUMAN_V8E_DESIGN_FREEZE
```

All design-commit fields in that approval record must resolve to the same
exact frozen design commit. The approval record must not bind to its own Git
commit or blob and cannot authorize an unreviewed working tree.

Any semantic change to this V8E draft after a prerequisite review or recheck
creates a new exact candidate SHA. Prior results then become invalid for the
new candidate and the affected review/recheck chains must be repeated. After
`HUMAN_V8E_DESIGN_FREEZE`, semantic changes are prohibited unless this design
explicitly pre-authorizes the exact change; otherwise a new successor study
is required.

## 9. Design status and scope boundary

```text
design_finalized=false
human_design_freeze_complete=false
implementation_created=false
approval_artifact_created=false
network_access_authorized=false
private_data_access_authorized=false
human_gate_consumed=false
```

This draft does not implement the producer or verifier, create an approval or
freeze artifact, read private files, inspect ticker identities, access Yahoo
or JPX, consume a gate, acquire raw data, open research, or evaluate strategy
profitability. Implementation may begin only after fresh V8E design review,
fresh V8E preservation PASS/reviews, exact-SHA human freeze, and the remaining
V8E authority and gate sequence.
