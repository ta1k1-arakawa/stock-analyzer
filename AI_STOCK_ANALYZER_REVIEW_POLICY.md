document_type=STOCK_ANALYZER_REVIEW_AND_RESEARCH_PRIORITY_POLICY
status=ACTIVE
scope=ta1k1-arakawa/stock-analyzer
authority=GPT_METHODOLOGY_AND_INDEPENDENT_REVIEW
supersedable_only_by=EXPLICIT_HUMAN_DECISION_OR_STRICTER_FROZEN_DESIGN

This is the stable cross-chat review and research-priority policy for
stock-analyzer. A task-specific frozen design wins over this general policy;
stricter repository governance also wins. Reading this policy grants no
network, private, T0, production, package, environment, human-gate, or
other execution authority.

## 1. Cross-chat bootstrap

For an important request such as continuing the project, reviewing a SHA,
deciding what to do next, preparing execution, or assessing profitability,
a fresh ChatGPT conversation must, when connected GitHub is available:

1. inspect the actual repository;
2. read `AGENTS.md`, `AI_RESEARCH_EXECUTION_RULES.md`, and this policy;
3. read `PROJECT_STATE.md`;
4. read `AI_REAL_EXECUTION_RUNBOOK.md` when real, private, or gated work is relevant;
5. read the applicable frozen design, approval, review, and adjudication artifacts;
6. verify the authoritative branch and exact remote HEAD;
7. recover the current stage and open findings from repository evidence;
8. independently review the current HEAD when the repository says `AWAITING_GPT_REVIEW`;
9. do not ask the human to re-enter state recoverable from the repository.

Repository and frozen artifacts override conversational summaries. A useful
state report is:

```text
CURRENT_STAGE=
REMOTE_HEAD=
LATEST_REVIEWED_SHA=
LATEST_RESULT=
CRITICAL=
HIGH=
MEDIUM=
OPEN_FINDINGS=
NEXT_TASK=
EXECUTOR=
HUMAN_GATE=
PROHIBITIONS=
NEXT_PASS_CONDITION=
```

## 2. Exact-SHA independent review

Every important review targets an exact 40-hex SHA and verifies the
authoritative remote HEAD, parent and ancestry, changed-file scope, actual
diff, relevant complete source/artifacts, frozen provenance, governance,
targeted tests, and available CI. The reviewer distinguishes executor
self-report from GPT independent review, searches for newly visible defects,
and fails closed on provenance ambiguity.

Required verdict:

```text
REVIEWED_SHA=...
CRITICAL=...
HIGH=...
MEDIUM=...
RESULT=PASS/BLOCK
```

PASS requires CRITICAL=0, HIGH=0, and MEDIUM=0. A safe LOW may be deferred
only when it does not affect authority, provenance, methodology, or safety.

## 3. Primary research objective

The objective is not maximizing historical profit. The primary objective is
credible expected future real-world profitability while preserving leak-free,
reproducible, scientifically valid evaluation. Prioritize forward-only or
OOS evidence, realistic net profit, drawdown, costs, slippage, robustness,
capacity, data quality, reproducibility, overfitting risk, and execution
realism. Validation PASS and T0 CONTINUE are not proof of profitability.

Until sufficient forward-only evidence exists:

```text
future_profitability_established=false
future profitabilityは未確立
```

## 4. Real-world executability

At relevant future stages assess 100-share-lot purchasability where
applicable, actual budget, highest-ranked versus actually purchasable
security, no-fill and skip behavior, capital lock-up, unresolved positions,
costs, slippage, liquidity, capacity, exit delay, concentration, loss
concentration, drawdown, and comparable candidate/baseline assumptions. Do
not assume the highest model score is the best executable trade.

## 5. Profit evidence hierarchy

Evidence strength increases from implementation correctness, through
data-quality/readiness, frozen OOS/T0 evidence, realistic cost-aware
evaluation, robustness/capacity evidence, to forward-only evidence. A
historical profit increase alone is insufficient.

## 6. Overfitting and multiple testing

Where applicable, freeze comparator/baseline, inclusion/exclusion, targets,
thresholds, costs/slippage, search space, stopping rule, acceptance criteria,
tie-break, and partitions before outcome observation. Do not expand grids,
select favorable securities or periods after outcomes, redraw validation,
optimize holdouts, silently change methodology, or cherry-pick favorable
runs. More tried models or strategies increase overfitting risk.

## 7. Failure classification

Keep distinct:

```text
PLUMBING_OR_TRANSPORT_FAILURE
DATA_QUALITY_FAILURE
GOVERNANCE_FAILURE
IMPLEMENTATION_FAILURE
STRATEGY_FAILURE
PROFITABILITY_FAILURE
```

Do not infer strategy failure from transport, parser, or environment
failure, and do not use strategy or profitability failure as permission to
modify frozen methodology.

## 8. Development efficiency

Prefer one authoritative current-state key per fact in `PROJECT_STATE.md`,
historical chronology in `PROJECT_DECISION_LOG.md`, duplicate-key and
contradiction checks, targeted tests before full regression,
production-entrypoint-to-safe-evidence synthetic tests, exact schema/type
validation, tampered-evidence fail-closed handling, deterministic state
reuse, and one finding per remediation by default. These practices never
weaken a frozen contract retroactively.

## 9. Authority

This policy grants no authority for network access, private or sealed access,
historical evaluation, T0, package or environment mutation, production
trading, one-shot authority reuse, or human-gate consumption. Human
point-of-use gates and frozen governance remain authoritative.

## 10. V11 and next-study priorities

After V10C is correctly completed, future research may prioritize actually
purchasable 100-share candidates, budget-aware selection, no-fill and skip
semantics, position and capital-lock semantics, cost/slippage-aware net PnL,
concentration and exit risk, fair baselines, pre-frozen adoption criteria,
and feasible security-universe data. This section does not modify V10C or
authorize V11. The normal IDEA -> DESIGN -> FREEZE -> IMPLEMENTATION ->
REVIEW sequence remains required.

## 11. Review questions

Reviewers should ask:

- Could future or holdout information leak?
- Could tampered or contradictory evidence be accepted?
- Is authority silently broadened?
- Is provenance exact and reproducible?
- Does the result survive realistic costs and execution?
- Is improvement robust or selection noise?
- Are we measuring what the user can trade?
- Is forward profitability established or only historical/OOS performance?
- Does the next task materially shorten the path to a trustworthy investment decision?
