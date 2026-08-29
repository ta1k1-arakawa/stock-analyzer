# V9_006 F6 required-period coverage successor parser design

```text
status=AWAITING_GPT_REVIEW
scope=SUCCESSOR_PARSER_DESIGN_ONLY
historical_evidence_class=DEVELOPMENT_EVIDENCE
confirmation_debt=true
```

## Binding and unchanged safeguards

This successor reuses the exact locked official F6 CHILD, inherited Phase A/B
provenance and integrity checks, expected structural-profile SHA, reviewed
structural inspector/safe validator, DATE columns `[4,6]`, deterministic
`xlrd` opening/decoding, and cell-type-before-value discipline. It permits no
network, path-based alternate input, refetch/provider substitution, union,
preferred column, one-column fallback, neighboring-year inference, or
interpolation. Inherited `CHATGPT_DECISION_REQUIRED` and phase-total
fail-closed provenance remain unchanged.

## Required-period rule

Let `R={2017,...,2025}`. For each frozen DATE column, mechanically derive
only its year membership from the same verified object. Then:

```text
covered_required_years = R intersect year_set_col4 intersect year_set_col6
missing_required_years = R minus covered_required_years
```

For each required year, presence in both columns is `COVERED`; absence in
either is `MISSING` for that year only. No full-history histogram or count
equality is required to decide required-period membership. Out-of-scope
differences must appear as a bounded safe diagnostic `out_of_scope_disagreement`
boolean, but do not change required-period coverage. Structural/source identity
ambiguity that prevents required-year membership remains fail-closed.

## Results and fan-out

`SUCCESSOR_REQUIRED_PERIOD_COVERAGE_CAPTURED` means required membership is
determinate, `coverage_evaluated=true`, `coverage_result_accepted=true`, and
the covered/missing lists are exact complements within R; all-covered is true
iff missing is empty. `SUCCESSOR_REQUIRED_PERIOD_COVERAGE_PARTIAL` has the
same determinate/accepted provenance but one or more missing required years;
it is not automatically study-terminal. `AMBIGUOUS`, `IMPLEMENTATION_FAILURE`,
and inherited `CHATGPT_DECISION_REQUIRED` occur only where determination or
an existing integrity boundary fails, never merely because a required year is
missing.

Each covered required year fans the existing GLOBAL slot ID to its twelve
months. Each missing required year leaves exactly those twelve months
`MISSING`; no monthly object or refetch is created.

## Exposure governance

The old `F6_YEAR_COVERAGE_AMBIGUOUS` result is immutable. Its exposed
histograms are DEVELOPMENT_EVIDENCE, not confirmatory evidence or retroactive
preregistration of this successor. Any promotion materially relying on this
amended rule requires fresh, forward, or independent confirmation. This design
authorizes no implementation, execution, evaluation, private access, model
fit, backtest, or profitability claim.
