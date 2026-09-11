# V11 SBI 100-Share Operation Policy Design Draft

```text
document_role=V11_PREFREEZE_DESIGN_DRAFT
task=V11_001_SBI_100_SHARE_OPERATION_POLICY_DESIGN_DRAFT
study=V11_SBI_100_SHARE_SINGLE_POSITION_OPERATION
successor_of=V10_CALENDAR_AUTHORITY_SUCCESSOR
authoritative_branch=v9-cross-sectional-close-auction-design
base_reviewed_head=d0e0ee33bd18580e2932f405c288c3e659aeacdd
status=DESIGN_ONLY_AWAITING_GPT_EXACT_SHA_REVIEW
V11_DESIGN_FROZEN=false
V11_EXECUTION_AUTHORIZED=false
V11_BACKTEST_AUTHORIZED=false
V11_REAL_TRADING_AUTHORIZED=false
FUTURE_PROFITABILITY=UNESTABLISHED
```

This document is the durable design basis for the planned V11 successor. It
is a prefreeze draft, not a frozen methodology, evaluation result, trading
instruction, or profitability claim. The labels below are normative:

- `HUMAN_FIXED` records conditions fixed by the user. They are not AI
  recommendations.
- `ADOPTED_DIRECTION` records the current design direction authorized for
  drafting and inheritance analysis.
- `GPT_RECOMMENDED_NOT_YET_FROZEN` records a GPT recommendation that is
  explicitly not adopted as methodology by this draft.
- `UNRESOLVED_BEFORE_FREEZE` records a closure required before V11 may be
  frozen or executed.

## 1. Purpose and study boundary

V10 remains the calendar/runtime/environment successor study and remains the
current study in `PROJECT_STATE.md`. V11 is a new economic/execution
successor study for the user's intended low-operation-burden, fixed-100-share
cash-equity operation. Creating V11 does not discard prior work. V11 may
inherit only reviewed V9/V10 infrastructure that is compatible with this
study and is explicitly bound during a later freeze; it does not silently
inherit incompatible economic or execution assumptions.

V10 calendar/environment PASS is not profitability evidence. V11 future
profitability remains `UNESTABLISHED`.

No V11 evaluation, model fit, backtest, data acquisition, broker access, or
real order is authorized by this draft. V11 is not `PASS` or `FROZEN`.

## 2. HUMAN_FIXED conditions

The following conditions are fixed by the human user and must be preserved
unless the human explicitly changes them. They are not recommendations made
by Codex or GPT.

```text
PRIMARY_OPERATION_CAPITAL=400000_JPY
BROKER=SBI_SECURITIES
JAPANESE_CASH_EQUITIES_ONLY=true
MARGIN=false
BORROWING=false
SHORT_SELLING=false
MAX_CONCURRENT_POSITIONS=1
NEW_ORDER_QUANTITY=100_SHARES_FIXED
POSITION_SCALE_UP=false
AVERAGING_DOWN=false
NEW_POSITION_WHILE_PRIOR_POSITION_OPEN=false
PROFIT_DOES_NOT_INCREASE_ORDER_QUANTITY=true
OPERATION_BURDEN=LOW_MONITORING_LOW_OPERATION_BURDEN
APPROX_MAX_BUY_ORDER_OPERATIONS_PER_TRADING_DAY=1
APPROX_MAX_SELL_ORDER_OPERATIONS_PER_TRADING_DAY=1
DAILY_TRADE_REQUIREMENT=false
CASH_WAITING_ALLOWED=true
```

The user's willingness to lose the full `400,000 JPY` is risk tolerance only.
It is not a stop-rule definition, permission to maximize losses, or
production-trading authorization.

## 3. ADOPTED_DIRECTION

The current direction for the eventual V11 freeze is:

1. V10 remains responsible for calendar, runtime, and environment work.
2. V11 changes the economic/execution design for the fixed 100-share SBI
   cash-equity operation.
3. V11 creation does not discard V9 or V10 work. Compatible reviewed
   infrastructure may be inherited explicitly and with provenance.
4. Backtest/evaluation/real orders remain prohibited until V11 design freeze
   and all required reviews and authorities are complete.
5. A main full-universe backtest is not authorized now.

The following compatible prior direction is preserved for V11 unless a later
reviewed V11 design explicitly changes it before protected evaluation:

- causal information only at D0;
- a three-trading-day decision cadence;
- D1 planned entry;
- D3 planned exit;
- `BEST_ONE_OR_CASH` single-position architecture;
- no same-close exit proceeds reused for same-close entry;
- conservative no-fill and unresolved-exit treatment;
- no outcome-driven redesign; and
- existing frozen periods.

The prior single-position objective is an inheritance anchor, not authority
to fill V11's still-open 100-share execution, affordability, signal-gate, or
portfolio criteria.

## 4. V11_HIGH_1 — SBI afternoon closing-order semantics

Status: `UNRESOLVED_BEFORE_FREEZE`.

SBI closing-auction execution must be bound before freeze to all of the
following:

- order type;
- market/session;
- valid submission window; and
- acceptance and cutoff semantics.

V11 must not assume that an order placed the previous night automatically
targets the next day's afternoon close. The known review fact is that a
closing-condition order effective for the morning close does not simply carry
into the afternoon close. The exact operational submission window remains:

```text
TO_BE_FROZEN_FROM_OFFICIAL_SBI_SPEC
```

Any convenient human work window is only a proposal until the official SBI
specification and the V11 design freeze bind it. This closure must be
resolved before a real order, execution simulation, or protected evaluation.

## 5. V11_HIGH_2 — order condition is not guaranteed fill

Status: `UNRESOLVED_BEFORE_FREEZE`.

V11 must represent at least these distinct states:

```text
ORDER_NOT_SUBMITTED
ORDER_REJECTED
ORDER_ACCEPTED
FULL_FILL
NO_FILL
UNRESOLVED_FILL
```

An order condition is not a fill guarantee. V11 must not encode
`close <= buy_limit => guaranteed fill`. Closing-auction existence, trading
halt, order acceptance, same-price queue/allocation ambiguity, and every
other available execution signal must be handled conservatively. Daily OHLC
alone must not silently convert uncertain execution into `FULL_FILL`.

The later frozen execution contract must define the evidence and precedence
needed to distinguish these states, including conservative treatment of an
unresolved exit and the resulting inability to open a second position.

## 6. V11_HIGH_3 — historical 100-share eligibility

Status: `UNRESOLVED_BEFORE_FREEZE`.

V11's new-order quantity is always exactly 100 shares. For each historical
point in time:

```text
historical normal board lot == 100 shares
    => may proceed to the remaining eligibility rules

historical normal board lot != 100 shares
    => INELIGIBLE for this V11 strategy
```

V11 must not replace 100 shares with 1,000 shares for older names whose
historical normal board lot was 1,000 shares. The formal period must not be
shortened merely because this eligibility path is difficult. Historical
point-in-time board-lot feasibility must be established; if it cannot be
established, the relevant data path fails closed.

## 7. V11_HIGH_4 — executable-TOP1 estimand alignment

Status: `UNRESOLVED_BEFORE_FREEZE`.

The required ordering is:

1. construct the full frozen eligible universe;
2. compute full-universe causal features and ranks;
3. score all names;
4. apply causal execution, 100-share, and affordability eligibility;
5. choose the highest-ranked executable name; and
6. hold `CASH` if none qualify or the abstention rule rejects.

V11 must not compute cross-sectional features only within cheap or affordable
names. However, the primary estimand and the model-selection estimand must
still be closed before freeze because actual affordability can become
portfolio-path-dependent: model choices can change later cash.

The following are explicitly unresolved design decisions:

```text
V11_T0_PRIMARY_ESTIMAND=UNRESOLVED_BEFORE_FREEZE
V11_MODEL_SELECTION_PRIMARY_ESTIMAND=UNRESOLVED_BEFORE_FREEZE
V11_PRE_T1_SIGNAL_GATE_ESTIMAND=UNRESOLVED_BEFORE_FREEZE
V11_RANDOM1_ELIGIBLE_SET=UNRESOLVED_BEFORE_FREEZE
```

Current GPT recommendation, not adopted methodology:

```text
REFERENCE_AFFORDABLE_TOP1=GPT_RECOMMENDED_NOT_YET_FROZEN
```

The candidate would use a pre-frozen causal `400,000 JPY` reference
affordability rule for signal and model-selection stages, while actual
evolving cash, fills, carry, and costs are evaluated later in portfolio/T1.
This is recorded for a later ChatGPT decision and must not be implemented,
used for model selection, or treated as frozen authority by the executor.

## 8. V11_MEDIUM_1 — portfolio halt and restart

Status: `UNRESOLVED_BEFORE_FREEZE`.

No `300,000 JPY` halt is adopted merely because a `300,000 JPY` robustness
case exists.

```text
PORTFOLIO_HALT_THRESHOLD=UNRESOLVED
AUTO_RESTART=false
```

Before freeze, V11 must define:

- the equity measurement timestamp;
- cash plus marked-position treatment;
- unrealized P&L treatment;
- fees and slippage treatment;
- behavior of an existing position at halt;
- whether halt affects new entries only; and
- restart authority.

The preferred policy wording is: “record the halt state and do not restart
automatically; restart requires a separate human/methodology review.”

## 9. Capital and affordability — unresolved mechanics

Both internally available cash and broker buy-power constraints matter. D0
must not use unknown D1 close information. A previous close at or below
`4,000 JPY` is not sufficient proof that a 100-share D1 order is executable.
Unsettled or unconfirmed sale proceeds are not guaranteed new-entry capital.

The following mechanics remain:

```text
D0_RESERVATION_RULE=CHATGPT_DECISION_REQUIRED_BEFORE_V11_FREEZE
D1_CASH_BUFFER_RULE=CHATGPT_DECISION_REQUIRED_BEFORE_V11_FREEZE
MAXIMUM_PRICE_MECHANICS=CHATGPT_DECISION_REQUIRED_BEFORE_V11_FREEZE
```

No executor may invent a reservation, cash buffer, maximum-price rule,
quantity resize, or broker-buy-power substitute. These choices must be
closed by ChatGPT under the applicable human-fixed conditions before V11
freeze.

## 10. Eventual evaluation requirements

After V11 is frozen and the required authorities are separately complete, the
eventual evaluation must report at least:

- executable post-cost P&L;
- maximum drawdown;
- no-fill and unresolved-fill frequency;
- cash fraction;
- operation burden;
- fixed data/service costs separately;
- pre-tax and post-tax results without mixing them;
- a cash benchmark;
- an appropriate market benchmark;
- executable Random-1;
- robustness; and
- reproducibility.

Formal evaluation criteria, portfolio criteria, the abstention threshold, and
single-position T1 promotion/rejection rules must be frozen before protected
evaluation and must not be chosen from evaluation outcomes. Random-1 must
use the same frozen causal eligibility, 100-share, capital, execution, and
carry mechanics required for a fair comparison once its eligible set is
closed.

## 11. Prohibitions and current authorization

This draft authorizes none of the following:

- access to the SBI account or any broker account;
- order placement or production trading;
- new market, private, or sealed data acquisition;
- inspection of protected evaluation outcomes;
- T0 or T1;
- model fitting;
- backtests or the main full-universe backtest;
- changing V10 frozen methodology;
- mutating the canonical environment;
- changing evaluation periods, thresholds, costs, or slippage;
- claiming profitability; or
- marking V11 `PASS` or `FROZEN`.

The current task is design-only. No human gate is consumed by this draft.
The V10 calendar/runtime/environment work remains governed by V10 and its
reviewed compatible artifacts. V11 execution can begin only after the design
closures above, required exact-SHA review, human/methodology authorities,
and any later implementation reviews are complete.

## 12. Freeze gate and next action

V11 remains a planned successor design awaiting GPT exact-SHA review. A GPT
review must evaluate this exact design artifact and its provenance. A later
freeze record must preserve the distinction between human-fixed conditions,
adopted direction, GPT recommendations, and unresolved decisions. Until that
sequence is complete:

```text
V11_DESIGN=CREATED_AWAITING_GPT_REVIEW
V11_DESIGN_FROZEN=false
V11_BACKTEST_AUTHORIZED=false
V11_REAL_TRADING_AUTHORIZED=false
V11_FUTURE_PROFITABILITY=UNESTABLISHED
NEXT_ACTION=GPT_EXACT_SHA_V11_DESIGN_REVIEW
```
