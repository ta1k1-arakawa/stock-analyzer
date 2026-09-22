# V12 Root-Cause Postmortem

## Outcome classification

V12 reached the frozen `PAUSE_V12` disposition before the first historical
screen. This was a process/readiness failure, not an established profitability
failure. The frozen design allowed `ACTIVE_IMPLEMENTATION_BUDGET_MINUTES=120`
and `MAX_SUBSTANTIVE_REMEDIATION_ROUNDS=1`.

## Causal timeline

1. Issue #12 received a GPT exact-SHA `BLOCK` with two HIGH findings: the
   current package observer did not preserve the reviewed installed-identity
   semantics, and readiness omitted the reachable synthetic PDF probe.
2. Issue #13 was substantive remediation round 1. It corrected those findings,
   but GPT found `HIGH_1=CURRENT_READINESS_DOES_NOT_REPRODUCE_REVIEWED_F6_ISOLATED_RUNTIME`.
3. The one allowed substantive remediation round was therefore exhausted
   without GPT PASS. The frozen pause condition was already met.
4. Issue #14 was a second substantive remediation round. It later received
   GPT PASS, but that success was out of budget and could not retroactively
   erase the triggered `PAUSE_V12` condition.
5. Issue #15 recorded the durable `PAUSE_V12` state.

## Root-cause hierarchy

- Direct trigger: the frozen remediation budget was exhausted before GPT
  independent PASS.
- Primary process cause: orchestration mechanically created the next
  remediation Issue after `BLOCK` without first checking the applicable frozen
  budget and stopping rule.
- Contributing engineering/review causes: Issue #12 used `pip freeze --all`
  presentation semantics instead of the reviewed `importlib.metadata`
  installed-identity semantics and omitted the reachable PDF probe. Issue #13
  fixed those points but did not reproduce the reviewed F6 isolated runtime.
  These are first-pass quality factors, not model-blame conclusions.
- Planning contributor: a one-round anti-sprawl ceiling was frozen while
  nontrivial protected PRE_GATE plumbing still had unresolved implementation
  risk. The ceiling correctly stopped sprawl; feasibility and residual risk
  should have been accounted for before freezing such a small budget.

## Was Codex the problem?

Codex contributed to first-pass implementation quality through the defects
that increased remediation burden, but Codex was not the primary root cause.
The process must remain safe when an executor is imperfect; the missing
pre-remediation budget gate was the preventable failure.

## Was GitHub Issue-first the problem?

No. Exact-SHA Issue-first traceability preserved the review evidence and made
the defect chain visible. The incomplete generic workflow lacked an explicit
frozen-budget gate before spawning remediation.

## Prevention rules added

- A `BLOCK` now requires a frozen-budget/stopping-rule gate before any
  substantive remediation Issue is created or activated.
- `EXHAUSTED` requires the terminal/pause/state-record action; `UNKNOWN`
  requires `CHATGPT_DECISION_REQUIRED`.
- Checkpoint cadence is subordinate to stricter study-specific anti-sprawl
  rules, and bookkeeping-only corrections are classified separately.
- Complex protected PRE_GATE work now uses a lightweight
  `PRE_IMPLEMENTATION_CONTRACT_MATRIX`, including predecessor semantics,
  reachable probes, isolation, fail-closed cases, and budget status.
- Anti-sprawl studies track remediation rounds used and remaining from round
  zero in durable state or the active Issue.

## What should happen next time

Before implementation, verify readiness feasibility and record unresolved
plumbing risk against the frozen budget. After every GPT `BLOCK`, evaluate the
budget gate before following the normal remediation cadence. If the budget is
exhausted, record the required terminal disposition immediately; do not let a
later successful out-of-budget fix rewrite the earlier state.
