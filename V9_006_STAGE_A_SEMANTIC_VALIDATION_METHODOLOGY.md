# V9_006 Stage-A semantic validation methodology

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
task=V9_006_HIGH_2_SEMANTIC_VALIDATION_METHODOLOGY_BINDING
methodology_authority=GPT-5.6_Sol
document_role=PREFREEZE_METHODOLOGY_REFINEMENT_RECORD
network_authorized_by_this_task=false
v9_design_frozen=false
parsers_or_semantic_validators_implemented_by_this_task=false
```

This records, exactly as decided by GPT methodology authority, the semantic
validation methodology for V9_005's `SECURITY_TYPE`, `CANONICAL_IDENTITY`,
`LISTING_TRANSITIONS`/`DELISTING_TRANSITIONS`/`MARKET_TRANSITIONS`,
`EFFECTIVE_DATE`, and `RECONSTRUCTION` Stage-A evidence items
(`V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md`, "Required
Stage-A evidence" items 2-8). It is a PREFREEZE methodology refinement, not
a V9 design freeze, and creates no network, data, T1, or design-freeze
authority. The execution agent records this methodology exactly; it does
not extend, reinterpret, or implement it in this task -- no parser or
semantic validator is implemented here. Original V9_006 HIGH_2
(`V9_006_LOCATOR_IMPL_HIGH_2` review context's parent finding: full
semantic reconstruction/validation) remains `OPEN` pending a separate,
future implementation task that codes this exact methodology.

## 1. `canonical_security_identity`

```text
canonical_security_identity = canonical_code
```

The exact serialized text/bytes of `canonical_security_identity` are the
exact 4-character `canonical_code` itself, UTF-8/ASCII, with no prefix,
suffix, newline, company name, market, or date.

The parser may mechanically remove spreadsheet representation artifacts
only when unambiguous (surrounding whitespace; integral numeric-cell
`.0`) and uppercase ASCII letters. Any other ambiguity is a semantic
validation FAIL.

The canonical common-stock issue code is exactly 4 characters. Do not
assume digits-only: JPX began assigning letters from 2024. Characters must
obey the official four-character stock specific-name-code grammar; an
invalid or ambiguous code is a semantic validation FAIL. A five-character
stock code that uses a reserved security-type character is NOT the
canonical ordinary-common identity for V9.

## 2. Reused codes

The same `canonical_code` must not represent more than one disjoint
genuine listing episode over the reconstructed chronology. If two disjoint
episodes, or conflicting identity evidence, occur for the same
`canonical_code`:

```text
reason=AMBIGUOUS_REUSED_SECURITY_CODE
```

Stage A FAILs. Do not disambiguate using company-name similarity, future
data, prices, outcomes, or manual judgment. A mere company-name change
while continuously listed does not create a new identity.

## 3. Point-in-time state

For each `canonical_code`, reconstruct, over the required chronology:

- `listed_state`
- exact official market/product-division state
- `security_type_state`

Listing effective date `e`: `listed_state=false` strictly before `e`,
`true` on and after `e`.

Delisting effective date `e`: `listed_state=true` strictly before `e`,
`false` on and after `e`.

Market/security-state transition effective date `e`: the new state applies
on and after `e`.

A missing or ambiguous effective date sets `effective_date_pass=false`.
Conflicting same-code/same-date transitions without a mechanically unique
official ordering FAIL Stage A; there is no manual ordering.

## 4. Security type

`security_type_state` is exactly one of:

```text
ELIGIBLE_DOMESTIC_ORDINARY_COMMON
EXPLICITLY_INELIGIBLE
UNKNOWN
```

`ELIGIBLE_DOMESTIC_ORDINARY_COMMON` only when official JPX metadata
mechanically proves both:

- domestic stock/TSE listing category; and
- ordinary common stock, represented by the four-character common-stock
  specific-name code without a reserved fifth security-type character.

An ETF/ETP, REIT, foreign issue, preferred/class/special/new-share
security, or another explicitly non-common product is
`EXPLICITLY_INELIGIBLE`.

Insufficient or conflicting metadata is `UNKNOWN`. Any `UNKNOWN` needed for
a reconstructed V9 identity/date makes `security_type_pass=false`.

Do not infer eligibility from price availability, TOPIX membership,
survival, name text, today's state, or outcomes.

## 5. Transition evidence

`listing_transition_pass` must be computed from parsed official transition
events, not merely F2 family coverage.

`delisting_transition_pass` must be computed from parsed official events;
where F2 and F3 both provide authoritative evidence, a conflict between
them FAILs.

`market_transition_pass` must prove every encountered market/segment
change has an exact representable effective date; it must not equal
`listing_transition_pass` merely as a proxy.

(This supersedes, for `listing_transition_pass`, `delisting_transition_
pass`, and `market_transition_pass`, the currently-implemented family-
coverage-only proxies in `src/v9_005_stage_a_jpx_probe.py`'s
`compute_stage_a_evidence` -- not changed by this docs-only task; see
"What this task does not decide" below.)

## 6. `canonical_identity_pass`

```text
canonical_identity_pass = true iff, for every reconstructed candidate
identity:
  - it has a valid canonical_code under section 1's exact grammar and
    serialization;
  - no duplicate identity exists on the same date; and
  - no ambiguous/reused-code condition (section 2) exists.
```

## 7. `deterministic_reconstruction_pass`

Begin from the locked terminal seed and locked official transitions. Two
independent deterministic reconstructions from identical locked bytes must
produce byte-identical canonical state output.

Additionally perform a reverse/forward consistency check: replay backward
through the required chronology, then replay forward using the same
canonical events. The recovered terminal state must byte-match the
canonical terminal state. A mismatch sets
`deterministic_reconstruction_pass=false`; there is no reconciliation.

## 8. `effective_date_pass`

`effective_date_pass=true` only when every consumed state transition has
one exact, mechanically derived effective JPX date. An unknown or
ambiguous date sets `effective_date_pass=false`.

## What this task does not decide

This methodology does not yet implement parsers or semantic validators.
Original V9_006 HIGH_2 (full semantic reconstruction/validation) remains
`OPEN` pending a separate implementation task that codes this exact
binding into `src/v9_005_stage_a_jpx_probe.py` -- including
`reconstruct_security_state`, `reconstruction_is_deterministic`, and
`compute_stage_a_evidence`'s `listing_transition_pass`/
`delisting_transition_pass`/`market_transition_pass`/
`canonical_identity_pass`/`effective_date_pass` computations, none of which
this task changes.

This task does not alter: the 648-record `MONTHLY_COVERAGE_MATRIX`, F1's
`TERMINAL_SEED` role, F2's post-2025 bridge derivation, F3's `YEAR`
strategy, F4's ratio orientation, F5/F6/F7's strategies, or the retry
policy; any cost/slippage/evaluation/model/label/search-space definition;
Stage B sample sizes; historical periods; the F2-F7 acquisition
implementation (`ACQUISITION_IMPLEMENTATION_COMPLETE` remains `False`);
original HIGH_3 (raw provenance/content-lock boundary); or original HIGH_4
(redirect-before-body-consumption behavior). It makes no network request,
no code change, and consumes no human authorization, including the
Stage-A authorization already given in chat.

`security_type_pass`'s previous unsafe proxy
(`bool(terminal_snapshot_locked)`) was already removed in
`V9_006_LOCATOR_IMPL_HIGH_3` (see `V9_006_STAGE_A_LOCATOR_IMPLEMENTATION_
REVIEW.md`), which introduced the explicit, currently-hardcoded-`False`
`security_type_validation_pass` input pending exactly the semantic
validator this document's section 4 now methodologically binds. This task
does not flip that flag; it remains `False` until the future
implementation task both codes this methodology and receives its own
independent GPT exact-SHA review PASS.

## Next action

`GPT_EXACT_SHA_V9_006_HIGH_2_SEMANTIC_METHODOLOGY_REVIEW`: obtain GPT's
independent exact-SHA review of this semantic-validation methodology
binding. A future, separately authorized implementation task would then
code this exact binding (canonical-code grammar/serialization, reused-code
detection, point-in-time state reconstruction, the three-state
`security_type_state` classifier, transition-evidence parsing,
`canonical_identity_pass`, the reverse/forward
`deterministic_reconstruction_pass` consistency check, and
`effective_date_pass`) into `src/v9_005_stage_a_jpx_probe.py`, itself
subject to a separate GPT exact-SHA review PASS -- still without executing
any real network request until a fresh, separate, explicit Stage-A human
network authorization is obtained after that implementation's review PASS,
in addition to the still-open original HIGH_3/HIGH_4 findings and the
separate F2-F7 acquisition-implementation task.
