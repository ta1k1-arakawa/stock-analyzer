# Stock Analyzer Research Viability Checkpoint

```text
document_type=PROJECT_RESEARCH_VIABILITY_CHECKPOINT
status=ACTIVE_AWAITING_GPT_REVIEW
scope=ta1k1-arakawa/stock-analyzer
purpose=MINIMAL_SCIENTIFICALLY_CREDIBLE_ACTIONABLE_NOTIFICATION_SYSTEM
feature_expansion_frozen=true
automatic_successor_creation_allowed=false
active_work_budget_minutes=480
active_work_consumed_minutes=0
active_work_remaining_minutes=480
budget_exhaustion_disposition=PAUSE_CURRENT_ROUTE
global_t0_readiness=NO
t0_authorized=false
future_profitability_established=false
```

This checkpoint records an explicit human project decision to stop the
unbounded successor-diagnostic pattern and assess whether the practical
research route is viable. The objective is not to build a general stock-
research platform or maximize historical profit. The smallest credible future
system that can eventually produce actionable `BUY`, `NO-TRADE`, and `EXIT`
notifications may be sufficient, including a system for one fixed security or
a very small fixed universe in a genuinely new study.

This checkpoint does not modify the V10C scientific methodology, reinterpret
the terminal V10C result, authorize T0, authorize protected data, or authorize
environment mutation. Frozen task designs and stricter repository governance
remain authoritative.

## 1. Current boundary

V10C T0 Attempt 1 remains terminal safe evidence with result class
`NO_VERDICT_DATA_INCOMPATIBLE`. It is not `STOP`, `CONTINUE`, strategy
failure, or profitability failure. Its one-shot authority was consumed and
its retry, refetch, cache-repair, substitution, and methodology-change flags
remain false. V10C must not be rerun or repaired under this checkpoint.

V10D remains a diagnostic-only successor. Its design and implementation are
not a reason to create further unbounded diagnostic successors. The
V10C runtime-recovery draft was never frozen and is superseded by this
checkpoint:

```text
V10C_RUNTIME_RECOVERY_DRAFT=SUPERSEDED_UNFROZEN_BY_RESEARCH_VIABILITY_CHECKPOINT
V10D_EXECUTION_AUTHORIZED=false
V10D_PROTECTED_PAYLOAD_READ_AUTHORIZED=false
V10D_REFETCH_AUTHORIZED=false
V10D_T0_AUTHORIZED=false
```

The already-promoted V10C canonical environment identity remains frozen, but
runtime feasibility is an operational question below. Nothing here unfreezes
or reverses that identity.

## 2. Active scope freeze

Until this checkpoint is resolved, do not add:

- models, features, hyperparameter searches, or new signal grids;
- security or universe expansion;
- data providers, acquisition routes, or cache substitutions;
- generalized framework or platform features;
- unrelated infrastructure improvements;
- another successor study merely because a technical blocker appears.

The only active technical questions are:

```text
Q1_ENVIRONMENT_FEASIBILITY
Q2_FIXED_DATA_FEASIBILITY
```

Q1 asks whether the required research environment can run reproducibly on the
target Windows machine without disabling or bypassing security controls. Q2
asks whether the already-fixed V10C/V10D inputs satisfy the frozen research
contract, and, if not, whether the exact incompatibility can be deterministically
localized and classified.

## 3. Active-work budget

The active-work budget is a project-management limit, not a research metric,
profitability criterion, or acceptance threshold. It begins only after this
checkpoint commit receives GPT exact-SHA PASS.

```text
ACTIVE_WORK_BUDGET_MINUTES=480
ACTIVE_WORK_CONSUMED_MINUTES=0
ACTIVE_WORK_REMAINING_MINUTES=480
```

Count conservatively, rounding uncertain active work up:

- Codex technical implementation or remediation;
- approved Windows diagnosis or execution;
- GPT-required technical remediation.

Do not count passive waiting. Maintain the consumed and remaining values in
durable state at later authorized checkpoints.

If the budget is exhausted before both Q1 and Q2 are resolved, the only
automatic disposition is:

```text
PROJECT_DISPOSITION=PAUSE_CURRENT_ROUTE
```

That disposition is a development cost/benefit decision. It is not strategy
failure, profitability failure, or evidence that the strategy loses money.
A future restart requires a new explicit human project decision.

## 4. Q1 — environment feasibility

Environment and software diagnosis is operational plumbing and may be
proportional to its statistical irreversibility. Within the active budget,
read-only local checks may be repeated when safe, including:

- Python and package metadata inspection;
- installed-package RECORD inspection;
- filesystem metadata;
- bounded Python import probes;
- bounded Windows CodeIntegrity/AppLocker event inspection;
- reproducibility checks.

These checks must not read protected stock payloads, expose outcome/T0
results, change methodology, bypass security controls, or overwrite frozen
evidence. They require no one-shot statistical human gate solely because
they are read-only plumbing observations.

Environment mutation is not implied. Package download, reinstall, uninstall,
resolution, environment recreation, security-policy modification,
`Unblock-File`, or comparable mutation requires a separate GPT decision before
execution and any required human gate.

Q1 PASS requires a reproducible supported solution on the target machine, not
an accidental one-off workaround and not a security-control bypass. A
runtime import success without reproducibility evidence is insufficient.

## 5. Q2 — fixed-data feasibility

Q2 may use only the already-fixed V10C/V10D inputs and frozen V10C
methodology. It must not refetch until success, substitute a provider, ticker,
period, cache, parser, feature, target, model, threshold, cost, slippage,
partition, or stopping rule. It must not redraw validation or repair data based
on an outcome.

Q2 has exactly two acceptable resolutions:

```text
FIXED_INPUT_CONTRACT_COMPATIBLE
FIXED_INPUT_INTRINSIC_INCOMPATIBILITY_LOCALIZED
```

`FIXED_INPUT_CONTRACT_COMPATIBLE` means the fixed inputs can proceed through
the required frozen contract. `FIXED_INPUT_INTRINSIC_INCOMPATIBILITY_LOCALIZED`
means they cannot satisfy that contract and the exact intrinsic reason has
been deterministically established. The latter does not authorize repairing
the inputs until they pass. If exact localization is not possible, Q2 is not
resolved and the route remains paused rather than guessing.

If Q2 resolves to intrinsic incompatibility, the present historical route is
paused unless a new explicit human decision creates a genuinely new study.
The result is not silently relabeled as strategy or profitability failure.

## 6. Resolution sequence and authority

This checkpoint itself is documentation and bookkeeping only. It grants no
execution authority. After its GPT exact-SHA PASS, work must still follow the
repository governance and any applicable frozen design:

1. Resolve Q1 using bounded, no-bypass, read-only plumbing checks where safe.
2. Resolve Q2 using only fixed inputs and the unchanged frozen contract.
3. Record both outcomes with exact provenance, safe evidence, and active-work
   accounting.
4. Return to GPT for independent adjudication before any new authority or
   methodology decision.

No result of this checkpoint issues a human gate, permits package/environment
mutation, authorizes protected payload access, or authorizes T0. No authority
may be silently reused.

## 7. Only allowed next goal after Q1 and Q2

Even if both questions are resolved, do not return to broad system
development. The next allowed goal is only:

```text
MINIMAL_FIXED_STRATEGY_EVALUATION
```

Its purpose is to determine whether continued investment is justified using
credible out-of-sample or forward-only, cost-aware, drawdown-aware,
robustness-aware, and execution-aware evidence. A later small production
target, such as one fixed security or a very small fixed universe, may be
considered only after that evidence and only in a separately designed and
reviewed study.

For any eventual actionable-notification design, assess what the user can
actually trade: 100-share-lot purchasability where applicable, available
capital, highest-ranked actually purchasable security, no-fill and skip
behavior, capital lock-up, unresolved positions, exit delay, costs, slippage,
liquidity, capacity, concentration, loss concentration, drawdown, and fair
baseline assumptions. Do not assume the highest model score is the best
executable trade.

## 8. Research integrity and profitability language

Keep unchanged the V10C/V9 periods, labels and targets, TOP1 estimand,
D1/D2/D3 semantics, feature set, Ridge/LightGBM parameters and random
states, scaler behavior, signal grid, thresholds, stopping rule, costs,
slippage, portfolio rules, promoted `283/17` provenance, evaluation identity,
and V10A calendar authority. No post-outcome tuning or T0 observation is
authorized by this checkpoint.

Implementation correctness and environment or data readiness are not
profitability evidence. T0 CONTINUE alone is not proof of profitability;
validation PASS alone is not proof of future profitability. Until credible
forward-only evidence supports it:

```text
future_profitability_established=false
```

The project should state plainly: `future profitabilityは未確立`.
