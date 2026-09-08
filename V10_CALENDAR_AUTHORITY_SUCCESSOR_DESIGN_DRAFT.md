# V10 Calendar Authority Successor Design Draft

```text
study_id=V10_CALENDAR_AUTHORITY_SUCCESSOR
predecessor=V9_CROSS_SECTIONAL_CLOSE_AUCTION
design_status=DRAFT_AWAITING_GPT_METHODOLOGY_REVIEW
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
execution_authorized=false
network_requests=0
```

## 1. Purpose and successor boundary

V10 is a new successor feasibility/design study. It is not a retry, repair,
rerun, or reinterpretation of V9. V9 remains terminal because its frozen and
authorized calendar/source lineage did not establish the historical actual
TSE-session authority required by `V9_009_HIGH_2`.

The only permitted methodological change in V10 is the mechanism used to
establish historical TSE cash-equity session/date eligibility. The V9
cross-sectional close-auction economic hypothesis and all non-calendar
methodology remain inherited unchanged. V10 has no authority to turn the V9
terminal result into a strategy or profitability result.

This draft performs no source acquisition, empirical source probing, raw
payload read, private read, network request, T0 operation, outcome read,
model fit, backtest, or profitability calculation.

## 2. Predecessor terminal facts

The current V9 closure is preserved exactly:

```text
V9_009_HIGH_2=UNRESOLVED_TERMINAL
V9_CALENDAR_AUTHORITY=FAIL_TERMINAL
V9_CALENDAR_FAILURE_CLASS=CALENDAR_FEASIBILITY_FAILURE
V9_017_TERMINAL_RESULT=DATA_QUALITY_FAILURE/FIXED_EIGHT_LOCATOR_FAILURE
V9_017_TERMINAL_RESULT_CHANGED=false
T0=NOT_RUN
HISTORICAL_EVALUATION=NOT_PERFORMED
FUTURE_PROFITABILITY=UNESTABLISHED
```

The V9_017 Source-B terminal result is a distinct data-quality failure. It is
not reclassified as a calendar failure, and V10 does not authorize a V9_017
rerun, refetch, locator repair, or same-study Source-B repair. V9_009 HIGH_2
was not declared impossible in every external source; it was unresolved
within the frozen/authorized V9 lineage.

## 3. Preserved V9 hypothesis and economics

V10 preserves the V9 scientific question: whether a pooled cross-sectional
score can support a robust Japanese-equity close-auction / close-to-close
architecture under the stated capital constraint. In particular, V10 does
not change:

- Japanese domestic ordinary common-stock scope, point-in-time membership,
  survivorship treatment, daily eligibility, and the fixed feature-history
  boundary;
- pooled cross-sectional scoring with ticker identity excluded from model
  features, the Ridge control, the shallow LightGBM candidate, mandatory
  TOPIX/equal-weight/random-K reporting baselines, and the fixed no-search
  model protocol;
- 1-share granularity, approximately equal-notional allocation, ten maximum
  concurrent positions, the 400,000 JPY primary case, the 300,000 JPY
  robustness case, the 90% invested / 10% cash-buffer mechanics, and the
  sector-cap and carry semantics;
- causal D0 information, the three-trading-day signal cadence, D1 entry,
  D3 planned exit, two close-to-close intervals, no optimistic stop-price
  execution, no same-close exit proceeds for same-close entries, and the
  frozen no-fill, cost, slippage, corporate-action, and unresolved-exit
  mechanics;
- the ten causal factors, split-neutral ex-dividend price-return target,
  same-date cross-sectional percentile ranks, and the fixed training/model
  roles;
- the V9 time roles: 2017 pre-evaluation training only, formal development
  evaluation from 2018-01-01 through 2025-12-31, and no change to the frozen
  T1/OOS, holdout, model-selection, threshold, benchmark, or promotion
  criteria; and
- the V9 requirements for transaction-cost disclosure, robustness, drawdown,
  reproducibility, and no profitability claim before the later authorized
  evaluation stages.

The V9 design and charter remain the authority for every item above. If any
non-calendar item cannot be inherited unambiguously, the correct state is
`CHATGPT_DECISION_REQUIRED`; this draft does not choose it.

## 4. The single changed component

Only this component changes:

```text
V10_CHANGED_COMPONENT=HISTORICAL_TSE_SESSION_DATE_AUTHORITY_MECHANISM
```

V10 replaces the failed V9 calendar/source feasibility route with a new,
separately designed authority mechanism. It does not alter the economic
hypothesis, target, data partitions, evaluation period, labels, thresholds,
cost/slippage assumptions, position sizing, universe, search space, or
profitability criteria.

## 5. Authority and source policy proposed for GPT review

The proposed acceptable authority class is limited to a source published by
the exchange or an exchange-designated official body that directly declares
or mechanically identifies historical TSE cash-equity trading-session/date
eligibility. The source must be capable of answering the frozen V9 question
for the required date range; a generic weekday/business-day calendar or an
index-price observation is insufficient by itself.

The following are not acceptable authority classes for V10:

- Yahoo, Stooq, broker calendars, vendor-derived calendars, OS/locale
  calendars, generic weekday logic, or `pd.bdate_range`;
- a source selected only because its dates make a later outcome result more
  favorable;
- any source whose historical publication/version identity, coverage, or
  session semantics cannot be bound before acquisition; or
- any new source introduced after observing a missing date, disagreement, or
  evaluation result.

This draft intentionally does not select a concrete provider, endpoint,
publication, API, document series, or source version. Those are
methodological choices that GPT must freeze before acquisition:

```text
V10_CONCRETE_SOURCE_IDENTITY=CHATGPT_DECISION_REQUIRED
V10_ACCEPTED_SOURCE_CLASS_EXACT_FORM=CHATGPT_DECISION_REQUIRED
V10_SOURCE_COUNT_AND_PRIORITY=CHATGPT_DECISION_REQUIRED_IF_MORE_THAN_ONE
V10_SOURCE_PUBLICATION_VERSION_BINDING=CHATGPT_DECISION_REQUIRED
```

Until those fields are frozen, V10 has no acquisition authority. The draft
proposal is one predeclared official-authority envelope, not permission to
probe candidates or to promote a source.

Fallback is not permitted in this draft. If GPT later authorizes more than
one source, the exact source order, relation, and disagreement rule must be
frozen before acquisition; a later source may not be tried merely because it
produces a more favorable calendar.

## 6. Coverage and exact session/date semantics

The inherited required coverage remains exactly:

```text
coverage_start=2017-01-01
coverage_end=2026-01-31
```

For each date in that inclusive range, the future accepted authority must
produce one mechanically bound state for the actual TSE cash-equity session
relevant to the frozen close-auction schedule:

```text
ELIGIBLE = an actual TSE cash-equity trading session is established
           by the frozen authority for that date
INELIGIBLE = the authority establishes that no such session occurred
UNRESOLVED = the authority does not establish exactly one of the above
```

Only `ELIGIBLE` dates may enter the V9 signal calendar. `INELIGIBLE` dates
must not enter it. `UNRESOLVED`, a missing date, an out-of-range date, or a
date whose session scope is ambiguous is a terminal feasibility failure. No
date may be inferred from weekdays, neighboring dates, price absence, or a
favorable later result.

The accepted calendar must support the inherited first-session-of-month
cutoff and the D0/D1/D2/D3 lookup. The treatment of partial, shortened, or
otherwise nonstandard sessions is not specified by the supplied V9
authority and therefore remains an explicit pre-freeze decision:

```text
V10_PARTIAL_SESSION_ELIGIBILITY=CHATGPT_DECISION_REQUIRED
```

GPT must resolve that item using an exact source-defined rule before design
PASS. No implementation may silently decide it from observed source content.

Structural requirements are exact: dates are unique, valid, within coverage,
and emitted in deterministic ascending order. A source record that contains
duplicate, malformed, contradictory, or incompletely scoped date/session
records fails closed.

## 7. Provenance and public evidence boundary

Before any source acquisition, the later frozen design must bind an exact
finite source manifest or an exact deterministic manifest-generation rule.
The binding must include, as applicable:

- source authority class, provider/endpoint or publication identity, version
  or effective date, coverage, and the exact source-object order;
- a canonical manifest digest and exact design/implementation Git SHAs;
- for every locked source object, its role/slot, status, byte count, payload
  SHA-256, and source-object identity; and
- the exact parser/schema version used for offline eligibility extraction.

Raw source bytes remain protected evidence. Public safe receipts may expose
only approved hashes, counts, coverage, booleans, closed failure codes, and
reviewed provenance SHAs. They must not expose raw payloads, private paths,
credentials, URLs when the approved artifact contract excludes them, ticker
identities, prices, outcomes, or exception text.

No source bytes or semantic labels are inspected by this design task.

## 8. Outcome-blind feasibility gate

Calendar feasibility must complete before any operation that can expose or
calculate an outcome:

```text
source binding and session eligibility
    -> safe no-network inspection and GPT adjudication
    -> only if PASS, later V10 input-binding acceptance
    -> only after separately authorized later stages, T0/outcome/model work
```

The feasibility runner must have no access path to prices, returns, labels,
model inputs, target values, trading outcomes, or profitability metrics. It
must not choose a calendar by comparing outcome results or by selecting the
calendar with the most eligible dates.

## 9. Bounded feasibility process and stopping rule

The infrastructure budget is finite and preregistered:

```text
V10_FEASIBILITY_EXECUTION_LIMIT=1
V10_AUTOMATIC_SUCCESSOR=false
V10_OUTCOME_DEPENDENT_RETRY=false
V10_FAVORABLE_CALENDAR_SELECTION=false
V10_FALLBACK_ALLOWED=false
```

The only later execution sequence permitted after this design is frozen,
implemented, and independently reviewed is:

1. A no-network provenance preflight verifies the exact repository,
   frozen-design SHA, reviewed implementation SHA, clean state, and exact
   source-manifest/input authority prerequisites.
2. One bounded feasibility execution acquires or reads only the exact
   predeclared authority objects. There is no source discovery, candidate
   expansion, or second execution.
3. Each complete public object is content-locked before semantic inspection.
   For public transport, the inherited V9 bounded pre-complete plumbing
   discipline may be reused only if GPT binds it to the selected source:
   maximum three attempts per exact request, retryable statuses limited to
   `{408,429,500,502,503,504}`, and no retry after a complete payload. A
   transport policy not covered by that inherited rule is
   `CHATGPT_DECISION_REQUIRED` before design PASS.
4. Offline processing uses only the exact locked objects. A parser, schema,
   missing-date, duplicate, disagreement, or eligibility failure is terminal
   and never authorizes refetch or source substitution.
5. A safe receipt is inspected without network. GPT adjudicates only the
   bounded authority evidence and does not use any outcome information.

The single feasibility budget is consumed when the bound source bytes first
enter semantic session/date processing. A pre-semantic operational failure
does not create an automatic rerun; it stops and requires
`CHATGPT_DECISION_REQUIRED`. A post-boundary failure never restores a gate,
permits a refetch, or permits a new source under V10.

If the one execution cannot establish the full required coverage and exact
session semantics, V10 is:

```text
V10_CALENDAR_FEASIBILITY=FAIL_TERMINAL
T0=NOT_RUN
HISTORICAL_EVALUATION=NOT_PERFORMED
```

The archive/source remediation route then stops. No unbounded PDF, locator,
label, calendar, or provider loop is allowed.

## 10. Missing, disagreement, duplicate, and malformed handling

The following are terminal feasibility failures:

- any required date absent from the bound source coverage;
- more than one record for a required date without a pre-frozen exact
  deduplication identity;
- conflicting eligibility/session states for a date;
- malformed dates, invalid coverage, incomplete source pages, or a source
  object whose publication/version identity cannot be verified; and
- inability to distinguish an actual TSE cash-equity session from a generic
  business-day or unrelated market/event record.

No missing date is imputed. No duplicate is resolved by first/last choice.
No disagreement is resolved in favor of a larger or more favorable eligible
set. No fallback source, alternate endpoint, alias, neighboring date, or
post-observation repair is permitted. If GPT later authorizes multiple
independent authority roles, their exact relation and disagreement result
must be specified before design PASS; unresolved disagreement remains
terminal.

## 11. Later authority and execution requirements

This draft grants no authority. Before any later source access, all of the
following are required:

1. GPT methodology PASS on this exact design and a frozen exact Git SHA.
2. Synthetic-only implementation and targeted tests for the selected
   source-binding/eligibility contract.
3. GPT exact-SHA PASS on that implementation.
4. A no-network preflight proving the exact frozen design, implementation,
   source manifest, and clean repository state.
5. A fresh point-of-use human authorization for any real network, private,
   sealed, credential, or protected durable-state boundary required by the
   final design. No V9 authorization is reusable.
6. One bounded execution and one safe no-network receipt inspection.

No human gate is consumed by this draft. A future public source must still
obey the final approved source, content-lock, retry, redirect, and stopping
contract; public transport plumbing does not authorize a methodology change.

## 12. Explicit non-actions and prohibitions

This task performs zero network requests, zero source acquisition, zero
private/sealed reads, zero protected Source-A reads, and zero semantic
inspection of historical outcomes or preserved production inputs. It does
not implement a runner or parser, select a source, freeze a session grammar,
consume a human gate, run T0, fit a model, run a backtest, or calculate
profitability.

V10 does not authorize Yahoo/Stooq, J-Quants whole-market daily quotes,
provider substitution, ticker-universe changes, PIT/delisted rules, outage
composites, new thresholds, new calendars, or any future Astra
recommendation. It does not change the V9 evaluation period, holdout,
labels/targets, costs, slippage, search space, stopping criteria, or
profitability gates.

## 13. Design-freeze PASS criteria and next stage

GPT may mark this design `PASS_FROZEN` only when the exact source authority
class and identity, coverage, session/partial-session semantics, manifest
and publication/version binding, source order, missing/duplicate/disagreement
rules, fallback policy, retry policy, failure codes, one-shot budget, safe
receipt schema, and later authority sequence are all mechanically closed.
No `CHATGPT_DECISION_REQUIRED` item may remain unresolved.

If GPT passes and freezes this draft, the next stage is a synthetic-only
implementation of the frozen minimal calendar/session-binding runner,
followed by targeted tests and an independent exact-SHA implementation
review. That stage still performs no real acquisition. Only a later,
separately authorized execution may evaluate V10 calendar feasibility.

V10 feasibility PASS would establish input-binding readiness only. It would
not establish profitability and would not itself authorize T0, historical
evaluation, model fitting, outcome access, or strategy promotion.

```text
V10_STATUS=DRAFT_AWAITING_GPT_METHODOLOGY_REVIEW
V10_EXECUTION_AUTHORIZED=false
T0=NOT_RUN
HISTORICAL_EVALUATION=NOT_PERFORMED
FUTURE_PROFITABILITY=UNESTABLISHED
NEXT_ACTION=GPT_EXACT_SHA_INDEPENDENT_REVIEW
```
