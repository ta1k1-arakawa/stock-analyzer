# V13 Implementation Remediation Policy Amendment

```text
document_role=V13_IMPLEMENTATION_PROCESS_AND_ANTI_SPRAW_POLICY_AMENDMENT
human_decision_date=2026-09-23
HUMAN_APPROVED=true
scope=IMPLEMENTATION_PROCESS_AND_ANTI_SPRAW_ONLY
SCIENTIFIC_METHODOLOGY_CHANGED=false
EXECUTION_AUTHORITY_CHANGED=false
```

## Binding and scope

This amendment records the human decision that ordinary implementation
correction must be separated from scientific stopping. It is limited to
implementation process and anti-sprawl governance. It does not amend the
frozen V13 hypothesis, universe, data source, features, models or
hyperparameters, timing, costs, comparators, metrics, A-Q acceptance
criteria, outcome partition, or scientifically meaningful tie-break or
fallback semantics.

The amendment is bound to the frozen design as follows:

```text
BASE_FROZEN_DESIGN_COMMIT=61268237494e2968562983e456ea40e6f821d066
BASE_FROZEN_DESIGN_BLOB_SHA1=3bfcd695c69f6dac480f8fc99ca4f3916f668e4a
PRIOR_V13_IMPLEMENTATION_REMEDIATION_HARD_CAP=2
PRIOR_V13_ACTIVE_IMPLEMENTATION_BUDGET_MINUTES=240
NEW_IMPLEMENTATION_REMEDIATION_HARD_CAP=NONE_FOR_METHODOLOGY_PRESERVING_PRE_MEASUREMENT_FIXES
NEW_240_MINUTE_SEMANTICS=ENGINEERING_ESCALATION_THRESHOLD_ONLY
```

The prior two-round and 240-minute hard-stop interpretation is superseded
only for ordinary methodology-preserving implementation correction before a
relevant measurement, irreversible, or outcome boundary. The frozen base
design and all stricter requirements remain binding.

## Replacement policy

### 1. Mechanical implementation defects do not terminate a study

The following do not by themselves consume a scientific-remediation budget or
force V13 to pause when caught before a relevant irreversible or outcome
boundary:

- `IMPLEMENTATION_FAILURE` or a deterministic code defect;
- missing or insufficient targeted test coverage;
- a contract-matrix coverage gap;
- generated-worktree, interpreter, dependency-discovery, or other plumbing;
- documentation, bookkeeping, or current-state correction;
- deterministic serialization or reporting defect;
- a missing mechanical fail-closed assertion; and
- another methodology-preserving implementation fix.

These defects still require correction and GPT exact-SHA PASS before V13
advances. There is no fixed count limit for ordinary methodology-preserving
pre-measurement implementation fixes. The distinction does not authorize
scope expansion, concealment of findings, relaxed tests, or a methodology
change.

### 2. 240 minutes is engineering escalation only

Active implementation minutes continue to be tracked. Above 240 cumulative
active minutes, set:

```text
ENGINEERING_BUDGET_ESCALATION=true
```

The threshold triggers decomposition or scope reconsideration; it does not
automatically pause or terminate V13. Correctness, review, and scientific
requirements must never be relaxed to save time.

### 3. Scientific and irreversible stops remain strict

The following remain hard stops:

- methodology ambiguity requiring a design decision;
- a frozen-methodology change without explicit amendment;
- leakage or future-information use;
- unauthorized data, network, private, or sealed access;
- irreversible outcome exposure outside a human gate;
- corrupted or unverifiable provenance that cannot be safely reconstructed;
- authority or human-gate violation;
- real historical A-Q failure or `STOP_HYPOTHESIS_NOT_PROMOTED`; and
- a task-specific safety or runbook STOP.

An implementation defect at or affecting such a boundary remains governed as
a strict scientific or safety remediation.

### 4. Scientific methodology changes remain separately governed

Changing the hypothesis, universe, source, features, models or
hyperparameters, horizon, costs, comparators, thresholds, A-Q criteria,
outcome partition, or scientifically meaningful tie-break or fallback
semantics is not an ordinary implementation fix. It requires explicit
methodology handling and the applicable human/GPT authority.

### 5. Current counters are engineering diagnostics

Historical counts are preserved, but their hard-stop meaning is removed for
the scope above:

```text
V13_SUBSTANTIVE_REMEDIATION_ROUNDS_USED=2
V13_SUBSTANTIVE_REMEDIATION_ROUNDS_REMAINING=NOT_APPLICABLE_NO_HARD_CAP
V13_IMPLEMENTATION_REMEDIATION_HARD_CAP=false
V13_ACTIVE_IMPLEMENTATION_BUDGET_HARD_STOP=false
V13_ENGINEERING_ESCALATION_THRESHOLD_MINUTES=240
V13_ACTIVE_IMPLEMENTATION_MINUTES_USED=46
```

This amendment creates no new execution, data, model, backtest, trading, or
human-gate authority. Real-market and outcome authorities remain false, and
future profitability remains unestablished. Synthetic feasibility evidence
remains synthetic and is not a profitability or historical-viability claim.

## Next bounded action

After GPT exact-SHA PASS of this amendment, perform a bounded,
methodology-preserving remediation of Issue #31 findings H1, H2, H3, and M1.
Further mechanical BLOCKs may be corrected under this policy. Scientific or
irreversible failures remain strict and are not converted into engineering
diagnostics.
