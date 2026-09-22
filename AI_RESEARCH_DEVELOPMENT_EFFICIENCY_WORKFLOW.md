# AI Research Development Efficiency Workflow

```text
document_type=REPOSITORY_ORCHESTRATION_EFFICIENCY_AND_QUALITY_METRICS_CONTRACT
status=ACTIVE
scope=MEASUREMENT_AND_FIRST_PASS_QUALITY_IMPROVEMENT_ONLY
```

## Purpose and precedence

This contract provides stable, comparable measures for development throughput
and first-pass engineering quality. It changes no research methodology,
statistical criterion, frozen design, data scope, or acceptance criterion. It
creates no execution, network, private-data, durable-state, or human-gate
authority.

It weakens no GPT exact-SHA review, PASS criterion, human gate, STOP
discipline, provenance, test discipline, or frozen design. Existing stricter
governance always wins, including `AGENTS.md`,
`AI_RESEARCH_EXECUTION_RULES.md`, `AI_REAL_EXECUTION_RUNBOOK.md`,
`AI_RESEARCH_CHECKPOINT_WORKFLOW.md`, and task-specific designs.

Never optimize raw commit count itself. Never hide findings, avoid a BLOCK,
weaken tests, or batch unrelated findings to improve these metrics. Metrics
describe process quality; they never justify a change in methodology or an
exception to an existing safety rule.

## Primary metrics

### FIRST_PASS_PASS_RATE

For new substantive design or implementation units, report:

`number whose first GPT exact-SHA review is PASS / number receiving their first GPT exact-SHA review in the measurement window`.

Report numerator and denominator. Report remediation units separately: a PASS
on remediation never inflates this rate for a new substantive unit.

### REMEDIATION_BURDEN

Report all of the following:

- remediation task count;
- remediation count per originating substantive unit;
- finding counts by CRITICAL, HIGH, MEDIUM, and LOW; and
- repeated-remediation chains separately from one-pass remediations.

The purpose is defect learning, not suppressing legitimate findings or
forcing unrelated fixes into one task.

### STAGE_ADVANCEMENT

Count durable, meaningful transitions rather than commits. Examples are a
design PASS, implementation PASS, validated execution checkpoint,
promotion/rejection decision, or forward-evidence milestone. Do not count
bookkeeping alone as stage advancement.

### PRE_BOUNDARY_DEFECT_INTERCEPTION

Count CRITICAL/HIGH/MEDIUM defects caught before the relevant irreversible,
private, real-network, or production boundary. Preserve severity and record
whether the defect would have caused deterministic post-gate/post-boundary
failure when that fact is mechanically supported.

### CYCLE_TIME

When timestamps are mechanically available, report time from a substantive
task commit to its accepted PASS and the duration of each remediation chain.
Never invent unavailable review timestamps; mark those values `UNAVAILABLE`.

## Secondary diagnostics

Report raw commit count, record-only/admin commit count, and substantive
commit count as diagnostics only. They are never optimization targets and do
not replace any primary metric.

## Pre-implementation contract and remediation tracking

For complex protected `PRE_GATE`, environment-checker, parser, or runtime
boundary work, complete a lightweight `PRE_IMPLEMENTATION_CONTRACT_MATRIX`
before implementation or commit. It must map the task to:

- exact reviewed predecessor semantics to reuse;
- reachable post-boundary dependencies and required synthetic probes;
- process-isolation and runtime-identity requirements;
- task-specific runbook constraints;
- negative and fail-closed cases; and
- current frozen remediation-budget status.

The matrix is a closure aid, not a new review authority and not boilerplate
that duplicates the full design. It should expose obvious predecessor or
runbook omissions before the first GPT exact-SHA review. Any unresolved
methodology, authority, or scope choice remains `CHATGPT_DECISION_REQUIRED`.

Anti-sprawl studies must track remediation rounds from round zero in durable
state or the active Issue:

```text
REMEDIATION_ROUNDS_USED=
REMEDIATION_ROUNDS_REMAINING=
FROZEN_REMEDIATION_BUDGET_STATUS=WITHIN_LIMIT|EXHAUSTED|NOT_APPLICABLE|UNKNOWN
```

Do not reconstruct these values only after a later review. Bookkeeping-only
corrections are tracked separately from substantive remediation.

### Engineering diagnostics versus scientific stops

Remediation count and active time are engineering diagnostics unless a human-
approved, study-specific rule explicitly freezes them as scientific stopping
conditions. Ordinary methodology-preserving implementation, plumbing,
targeted-test, deterministic reporting, and bookkeeping corrections do not
automatically consume a scientific-remediation round or force study pause.
They still require correction, appropriate checks, and exact-SHA review before
advancement. If an engineering threshold is exceeded, set the corresponding
escalation diagnostic and reconsider decomposition or scope; do not terminate
the study automatically and do not relax correctness or scientific
requirements to improve throughput. Scientific ambiguity, methodology
change, leakage, unauthorized access, irreversible-boundary failure, and
other frozen hard stops remain strict.

## Measurement windows and interpretation

Use these windows:

- active development day, using Asia/Tokyo for human-facing daily reports;
- rolling 7-day window; and
- per-stage summary.

Always show raw numerator/denominator. When a denominator is small, flag
`SMALL_SAMPLE` rather than overinterpreting a percentage.

Improving first-pass PASS rate toward roughly 70--80%+ over a meaningful
multi-unit sample is a nonbinding engineering objective, not a PASS gate.
Research integrity and every existing quality/safety rule dominate it.

## Standard report template

```text
WINDOW=
SUBSTANTIVE_NEW_UNITS=
FIRST_PASS_PASS=
FIRST_PASS_PASS_RATE=
REMEDIATION_UNITS=
REMEDIATION_PER_BASE_UNIT=
FINDINGS_CRITICAL=
FINDINGS_HIGH=
FINDINGS_MEDIUM=
FINDINGS_LOW=
STAGE_ADVANCEMENTS=
PRE_BOUNDARY_C_H_M_CAUGHT=
RAW_COMMITS_DIAGNOSTIC_ONLY=
RECORD_ONLY_COMMITS=
CYCLE_TIME=
SAMPLE_SIZE_WARNING=
PRIMARY_BOTTLENECK=
NEXT_PROCESS_IMPROVEMENT=
```

## Proportional pre-commit first-pass-quality checklist

For ordinary small tasks, confirm:

- exact expected HEAD, clean tree, and allowed scope;
- targeted short tests or static checks appropriate to the change;
- `git diff --check`;
- exact command reporting; and
- no unrelated changes.

For validators, acceptance boundaries, one-shot gates, durable state,
parser/schema work, real-execution plumbing, or similarly high-risk
implementation, additionally:

- perform the mechanical closure sweep already permitted by
  `AI_RESEARCH_CHECKPOINT_WORKFLOW.md`;
- pass at least one canonical valid synthetic fixture through the real
  production acceptance/closure validator without monkeypatching that
  validator away;
- keep ordering-only harnesses from being the sole success-path coverage;
- add failure-surface negative cases proportional to the code;
- for filesystem/one-shot boundaries, consider as applicable absence, valid
  state, malformed state, non-regular entry, symlink/dangling symlink,
  read/stat uncertainty, no-overwrite, race/ambiguity, and pre-/post-gate
  failure semantics;
- for schemas/parsers, consider exact keys, types, cardinality, ordering,
  provenance, and malformed nested values; and
- verify frozen canonical counts and identities in success closure where
  applicable.

Targeted tests remain sufficient when they cover the change; do not run a
full pytest/regression suite automatically merely for this checklist. These
are engineering-quality checks only and do not become new research
methodology or statistical criteria.
