# V10 Calendar Authority Inheritance Correction Amendment

```text
document_role=V10_FROZEN_DESIGN_CORRECTION_AMENDMENT
status=DRAFT_AWAITING_GPT_EXACT_SHA_REVIEW
base_frozen_design_git_sha=8c923ed1734c6bdfe95a743cd9e15a5156d62c03
v9_008_gpt_reviewed_sha=8079bb0956c105a3972e59f0e4ba21ea5b81b14a
changed_component=NON_CALENDAR_INHERITANCE_PRECEDENCE_CORRECTION
new_economic_methodology_introduced=false
calendar_methodology_changed=false
study_identity_changed=false
```

## Purpose and authority correction

V10 remains the study identity `V10_CALENDAR_AUTHORITY_SUCCESSOR`. Its
scientific intent is limited to replacing the historical TSE
session/date-authority mechanism. It does not intentionally revert later
passed V9 economic methodology.

`V9_008_SINGLE_POSITION_OBJECTIVE_PREFREEZE_AMENDMENT.md` predates V10 and
was GPT-reviewed at
`8079bb0956c105a3972e59f0e4ba21ea5b81b14a`. Its single-position portfolio
and objective rules therefore supersede the conflicting ten-position text
carried into frozen V10 Section 3. V10's explicit purpose was calendar-only
authority replacement; the ten-position text is stale carry-over and is not
authority to reverse V9_008.

The effective V10 inherited portfolio/objective authority is:

```text
MAX_CONCURRENT_POSITIONS=1
SELECTED_NAMES_PER_ENTRY_CYCLE=1
POSITION_CONCENTRATION=SINGLE_BEST_NAME
BEST_ONE_OR_CASH=true
PRIMARY_RANK_ESTIMAND=RANK_TOP1_EDGE
INDEPENDENT_PER_TICKER_MODELS=false
```

Accordingly, the following stale V10 Section 3 mechanics are not effective
authority: ten maximum concurrent positions; equal-notional ten-name
allocation; 90% invested / 10% cash-buffer mechanics; ten-name sector-cap
or diversification mechanics; and `target_notional=0.90*equity/10` or an
equivalent rule.

This amendment does not invent replacement portfolio mechanics.
`V9_008_T1_SINGLE_POSITION_CRITERIA=NOT_YET_FROZEN` remains authoritative.
Exact causal D0 quantity reservation, D1 cash-buffer/quantity mechanics,
the abstention threshold, and final single-position T1 portfolio criteria
remain `CHATGPT_DECISION_REQUIRED_BEFORE_PORTFOLIO_EXECUTION` and cannot be
filled by the executor.

## Effect and gates

This amendment changes inheritance precedence only. It does not change the
calendar methodology, source identity, coverage, anchors, runtime-lock
design, costs/slippage, target semantics, periods, labels, models,
thresholds, or any unrelated later PASS V9/V10 decision. It does not
authorize T0, T1, historical evaluation, model fit, backtest, calendar
generation or date inspection, environment mutation, network/private/sealed
access, or profitability evaluation.

The amendment requires GPT exact-SHA independent-review `PASS` and fresh
human freeze approval before it becomes effective execution authority. The
prior human approval for the `8c923ed1734c6bdfe95a743cd9e15a5156d62c03`
V10 base design freeze is already consumed, is scoped only to that base
design-freeze transition, and must not be reused for this amendment.
