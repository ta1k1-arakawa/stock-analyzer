# AI Research Checkpoint Workflow

```text
document_type=REPOSITORY_ORCHESTRATION_CADENCE_CONTRACT
status=ACTIVE
scope=TASK_BATCHING_AND_REVIEW_CHECKPOINT_CADENCE_ONLY
```

## Purpose

This document exists solely to reduce orchestration round trips between the
GPT methodology/review authority, the human operator, and the Claude Code /
Codex execution agent, **without weakening**:

- research integrity;
- provenance (exact-SHA bindings, safe-evidence contracts, durable-state
  checks);
- human-gate authority or discipline;
- STOP discipline (`EXPECTED_HEAD_MISMATCH`, dirty tree, unexpected
  files/history, frozen-blob mismatch, missing authority/artifact); or
- GPT exact-SHA independent review authority.

It is orchestration-only. It changes batching and checkpoint cadence; it
changes no methodology, no authority, and no frozen rule. Where anything
below appears to conflict with `AGENTS.md`, `AI_RESEARCH_EXECUTION_RULES.md`,
`AI_REAL_EXECUTION_RUNBOOK.md`, a task-specific frozen design, or any
stricter existing rule, those documents win (section 9).

## 1. Design checkpoint

One substantive design task produces one artifact for one GPT exact-SHA
review. If that review returns `BLOCK`, the remediation task that follows
may address the explicit named finding plus, at the execution agent's
discretion, one bounded, scope-local mechanical closure sweep over that
same design document (see section 5). Default one finding per remediation
task; at most two only when they are strongly coupled (the same root cause,
the same section, or fixing one is mechanically impossible without the
other).

## 2. Implementation checkpoint

One execution-agent task should normally bundle: the implementation change,
targeted/short tests proving it, the required state/evidence/doc updates,
and the commit + push -- as one unit, followed by one GPT exact-SHA
independent review of that unit. This is the existing pattern already used
throughout this repository's `V9_006` implementation remediation chain; this
document only names it as the default, it does not introduce it.

## 3. Real execution checkpoint

Protected, private, real-network, or long-running execution stays on direct
Windows PowerShell exactly where `AI_REAL_EXECUTION_RUNBOOK.md` and
`REAL_EXECUTION_PYTHON_ENVIRONMENT.md` already require it. GPT reviews the
safe result (safe evidence, counts, hashes, booleans -- never raw/private
content) before methodology or implementation work progresses on top of it.
This checkpoint's shape is unchanged by this document; it is restated here
only so the cadence contract is complete in one place.

## 4. Record-only commits

Do not create a standalone commit merely to record a prior PASS/result
unless an authority boundary, a human gate, a frozen design, an
audit/provenance requirement, or a stricter existing repo rule requires
immediate durable recording. Otherwise, fold the recording of a prior GPT
review or a prior safe execution's evidence into the next substantive
authorized repo-writing task, rather than spending a round trip on a
recording-only commit. When in doubt, record immediately -- silence is never
the safe default for an authority-boundary or human-gate event; batching is
only for genuinely deferrable bookkeeping.

## 5. Closure sweep is not self-review

A closure sweep lets the execution agent repair only the **mechanical**
implications of already-frozen intent within the current design/task scope
-- for example: exact API/mechanical extraction semantics; closed
schema/key/type/cardinality/order enforcement; deterministic
iteration/ordering; exact closed-enum mapping; fail-closed exception
classification; phase provenance; or privacy-safe output validation. It may
never invent methodology, choose between competing methodological options,
or resolve an authority/scope ambiguity. Any such ambiguity encountered
during a closure sweep is `CHATGPT_DECISION_REQUIRED`: stop, name it, and do
not guess. A closure sweep is not, and never substitutes for, GPT-5.6 Sol's
own independent exact-SHA review; the final `PASS` for the design or
implementation always remains GPT's call, never the execution agent's
self-call.

## 6. Remediation granularity

One finding equals one task by default. At most two findings in one task
only when they are strongly coupled (section 1). Do not batch unrelated
findings into one remediation merely to save a round trip -- that trades
reviewability for speed in exactly the wrong direction; each finding should
remain independently traceable to its own fix and its own review.

## 7. Long work

Full `pytest`/regression/backtest/walk-forward runs, real data acquisition,
and portfolio simulation must not depend on a cloud AI session's lifetime.
Use durable state and direct Windows execution where the existing
governance already requires it, exactly as before; this document does not
relax any of that.

## 8. Provenance and STOP discipline

Every existing STOP condition remains exactly as strict as before this
document: an expected-HEAD mismatch, a dirty working tree, unexpected files
or history, a frozen-blob mismatch, or a missing required authority or
artifact all still mean STOP, mechanically, before any further action --
this workflow does not add a faster path around any of them.

## 9. Precedence

`AGENTS.md`, `AI_RESEARCH_EXECUTION_RULES.md`, `AI_REAL_EXECUTION_RUNBOOK.md`,
any task-specific frozen design, and any rule stricter than this document
always override it. This document only changes batching and checkpoint
cadence; it creates no new authority, weakens no existing rule, and is not
itself a methodology or review-authority document.
