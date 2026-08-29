# V9 prefreeze proportional-strictness and F6 coverage amendment

```text
status=POST_EXPOSURE_TRANSPARENT_PREFREEZE_AMENDMENT
authority=HUMAN_AUTHORIZED
strictness=STRICT_ON_INFERENCE_PRAGMATIC_ON_PLUMBING
operating_rule=CLAIM_LEVEL_FAIL_CLOSED_PROJECT_LEVEL_GRACEFUL_DEGRADATION
```

## Historical record preserved

The historical F6 old-rule production result remains exactly
`F6_YEAR_COVERAGE_AMBIGUOUS`. It is not rewritten as a pass, and the old
full-history-histogram rule remains part of the immutable execution record.
V9 is not terminated solely by that old-rule ambiguity.

## Non-negotiable inference controls

Nothing in this amendment relaxes holdout/future leakage controls,
post-outcome strategy or model tuning prohibitions, private/sealed
boundaries, cost/slippage treatment, promotion criteria, frozen selection or
validation periods, or provider/ticker/period cherry-picking. Exact-SHA,
provenance, and human-gate controls remain in force.

For public source/data plumbing and coverage, inability to mechanically prove
a cell/year/month leaves that claim `MISSING`/`UNPROVEN`. It does not by
itself terminate V9 unless the missing evidence is indispensable to a frozen
downstream inference. Severity tracks inferential/statistical
irreversibility: warning or partial data-quality conditions remain distinct
from study-fatal failures.

## F6 successor required-period rule

For `R={2017,...,2025}`, a required year is covered only if mechanically
present in **both** preregistered date columns from the same locked official
F6 object:

```text
covered_required_years = R intersect year_set_col4 intersect year_set_col6
```

A disagreement outside `R` is an out-of-scope warning/data-quality
diagnostic, not an F6 required-period coverage failure. A required year
absent from either column is `MISSING` for that year only. Structural or
source-identity ambiguity preventing required-year membership determination
remains fail-closed. Full-history histogram-count equality outside `R` is
not required merely to establish 2017--2025 membership.

## Exposure transparency and confirmation debt

Because the real F6 histograms were exposed before this amendment, they are
`DEVELOPMENT_EVIDENCE`, not retroactive confirmatory validation. This rule
was not preregistered before observing those histograms. Any promotion claim
materially relying on this amended F6 rule requires later fresh, forward, or
independent confirmation. This amendment creates no historical-evaluation,
private/sealed-access, design-freeze, live-trading, or profitability
authority.

```text
V9_design_frozen=false
V9_historical_evaluation_authorized=false
V9_private_or_sealed_access_authorized=false
future_profitability_established=false
live_trading_authorized=false
```
