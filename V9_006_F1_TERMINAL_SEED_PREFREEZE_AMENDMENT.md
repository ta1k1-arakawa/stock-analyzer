# V9_006 F1 terminal-seed prefreeze amendment

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
task=V9_006_F1_TERMINAL_SEED_PREFREEZE_AMENDMENT
methodology_authority=GPT-5.6_Sol
document_role=PREFREEZE_METHODOLOGY_AMENDMENT_RECORD
probe_executed=false
network_authorized_by_this_task=false
v9_design_frozen=false
v9_study_identity_changed=false
```

This records, exactly as decided by GPT methodology authority, a PREFREEZE
amendment to `V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md`'s
Stage-A deterministic inventory semantics: source family F1
(`LISTED_ISSUES_MONTH_END`, List of TSE-listed Issues / month-end
listed-issue material) is `TERMINAL_SEED` only. It is a PREFREEZE amendment
made before any Stage-A real network probe has executed
(`probe_executed=false`) and before `V9_design_frozen`; it is not a
response to an observed Stage-A probe result, and it retains the same V9
study identity. The execution agent records this decision exactly; it does
not extend, reinterpret, or implement it in this task.

## The decision

1. `SOURCE_OBJECT_INVENTORY` remains heterogeneous and still includes a
   mandatory F1 `TERMINAL` object.
2. `MONTHLY_COVERAGE_MATRIX` now includes source families F2 through F7
   only: 6 families × 108 months (2017-01 through 2025-12) = exactly 648
   records.
3. F1 has **no** base `MONTHLY_COVERAGE_MATRIX` record for any month. It
   is not represented as `NOT_APPLICABLE_BY_SOURCE_CONTRACT` or `MISSING`
   -- it is simply outside the required monthly reconstruction contract.
4. F1's mandatory `TERMINAL` object is unchanged in every other respect:
   root/traversal unchanged; a unique current rolling object is required;
   terminal snapshot month `T` is parsed mechanically; an absent or
   ambiguous terminal object sets `terminal_snapshot_pass=false`, which
   fails Stage A; terminal raw provenance remains mandatory.
5. Historical F1 monthly snapshots are not required. Stage A must not
   search for, fetch, consume, or use them -- including as post-hoc
   corroboration or crosscheck if later discovered by any other means. A
   future study/amendment would be required to add them.
6. This amendment must not, and does not, weaken any substantive
   point-in-time (PIT) reconstruction gate (see below).
7. `required_inventory_missing_count` now refers to the 648-record F2-F7
   `MONTHLY_COVERAGE_MATRIX`. Every other existing separate mandatory-
   object/pass gate remains in force unchanged: `terminal_snapshot_pass`,
   `listing_transition_pass` (including F2 bridge slots),
   `delisting_transition_pass`, `market_transition_pass`,
   `security_type_pass`, `canonical_identity_pass`, `effective_date_pass`,
   `trading_calendar_pass` (including the F7 envelope), `deterministic_
   reconstruction_pass`, the month-end crosscheck rule, and
   `raw_provenance_pass`.
8. F2-F7 mappings, roots, traversal rules, F4 ratio orientation, the F7
   acquisition envelope, the V9_005_HIGH_2B signal-grid binding, and all
   other previously bound methodology are unchanged.
9. This is a PREFREEZE amendment before any Stage-A real network probe.
   `probe_executed=false` and `V9_design_frozen=false` are unchanged. The
   same V9 study identity is retained; this is not a response to an
   observed Stage-A probe result.
10. Retry count/backoff remains undecided. This amendment neither alters
    nor approves the existing invented retry policy in
    `src/v9_005_stage_a_jpx_probe.py`.

## Rationale

`V9_005_FREE_SOURCE_PUBLIC_NETWORK_PROBE_DESIGN_DRAFT.md`'s "Required
Stage-A evidence" section already defines the reconstruction algorithm as:
a locked `TERMINAL_SNAPSHOT` seed, plus locked official transition records,
deterministically producing the same security state on repeated runs
(evidence item 8, `RECONSTRUCTION`). That algorithm's only two required
inputs are the terminal seed and the transition records supplied by F2
(listing/market changes), F3 (delistings), and F4 (corporate actions) --
never a series of historical F1 monthly snapshots. A historical F1 monthly
snapshot would, at most, provide *redundant* corroboration of state that F2
and F3's transition records already mechanically determine from the
terminal seed forward and backward; it was never load-bearing for the
reconstruction algorithm as originally specified. Requiring it anyway (as
the prior `V9_006_SOURCE_SLOT_LOCATOR_HIGH_1A` remediation did, by leaving
F1's base-month cells as `MISSING` pending an unestablished historical
locator) manufactured a guaranteed, unresolvable Stage-A blocker out of
evidence the reconstruction contract never actually needed.

## Why this does not weaken substantive PIT reconstruction criteria

Every substantive point-in-time reconstruction requirement remains bound
exactly as before, sourced from F2/F3 and the other unchanged families,
not from F1:

- **All listings and exact effective dates** -- still required from F2
  (`LISTING_TRANSITIONS`, evidence item 2), unchanged.
- **All delistings and exact effective dates** -- still required from F3
  (`DELISTING_TRANSITIONS`, evidence item 3), unchanged.
- **All required market/segment transitions** -- still required
  (`MARKET_TRANSITIONS`, evidence item 4), unchanged.
- **Domestic ordinary-common security type at every required date** --
  still required (`SECURITY_TYPE`, evidence item 5); `UNKNOWN` still fails
  Stage A, unchanged.
- **Canonical identity without future state** -- still required
  (`CANONICAL_IDENTITY`, evidence item 6), unchanged.
- **Exact effective dates** -- still required (`EFFECTIVE_DATE`, evidence
  item 7); an ambiguous date still fails Stage A, unchanged.
- **Deterministic reconstruction** -- still required (`RECONSTRUCTION`,
  evidence item 8), unchanged; it still begins from the locked terminal
  seed and applies only locked official transition records.
- **Unknown/unrepresentable/ambiguous still fails.** Nothing in this
  amendment converts an unknown, unrepresentable, or ambiguous
  reconstruction input into a pass. F1's base-month cells are removed from
  the grid entirely -- they are not silently marked `AVAILABLE`,
  reinterpreted as non-blocking `MISSING`, or granted any other favorable
  status. The one thing that changes is that a historical F1 monthly
  snapshot is no longer *demanded as an input the algorithm never used*.

What this amendment removes is not reconstruction evidence -- it is a
demand for evidence that was never wired into the reconstruction algorithm
in the first place, and that (per `V9_006_SOURCE_SLOT_LOCATOR_HIGH_1_
REVIEW.md`) has no established official locator under the currently bound
F1 root. Leaving that demand in place would not have made the
reconstruction more rigorous; it would only have guaranteed Stage-A FAIL on
an input the algorithm does not consume.

## Scope discipline

`F2` through `F7`'s roots, traversal semantics, monthly-coverage mappings,
F4's ratio orientation, F7's acquisition envelope, and the V9_005_HIGH_2B
signal-grid binding are all unchanged by this amendment. The retry/backoff
policy remains a separate, still-undecided methodology question. No code
was changed, no probe was executed, and no network request was made.

## Authority created

```text
NETWORK_REQUESTS=0
DATA_ACQUIRED=false
CODE_CHANGED=false
PROBE_EXECUTED=false
HUMAN_GATE_CONSUMED=false
RETRY_POLICY_DECIDED=false
HISTORICAL_F1_DISCOVERY_PERFORMED=false
T1_OR_DESIGN_FREEZE_AUTHORITY_CREATED=false
```

This amendment is a docs-only methodology decision. It does not authorize
network access, data acquisition, T1 membership generation or opening,
model fitting, backtesting, profit calculation, or V9 design freeze, and
does not consume the human's existing chat-given Stage-A authorization.

## Next action

`GPT_EXACT_SHA_V9_006_F1_TERMINAL_SEED_AMENDMENT_REVIEW`: obtain GPT's
independent exact-SHA review of this amendment. A future, separately
authorized implementation task would then wire the amended F1-F7 semantics
-- F1 now `TERMINAL_SEED`-only, F2-F7 unchanged -- into
`src/v9_005_stage_a_jpx_probe.py`'s `resolve_month_locator` seam under this
exact binding, still without executing any real network request until a
fresh, separate, explicit Stage-A human network authorization is obtained
after that implementation's own GPT exact-SHA review PASS, and only after
the retry/backoff policy question is separately resolved.
