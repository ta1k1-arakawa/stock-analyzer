# V13 Pre-Implementation Determinism Amendment Draft

```text
AMENDMENT_STATUS=DRAFT_UNFROZEN
IMPLEMENTATION_BLOCKED_PENDING_AMENDMENT_GPT_PASS=true
IMPLEMENTATION_BLOCKED_PENDING_HUMAN_AMENDMENT_FREEZE_APPROVAL=true
```

This is an additive, methodology-bearing amendment to the frozen V13 design.
It closes deterministic mechanics that materially affect criteria A–Q before
offline/synthetic implementation begins. It does not modify the frozen base
design or `V13_DESIGN_FREEZE_APPROVAL.json`, and it does not authorize
implementation, acquisition, historical viability, model fitting,
backtesting, forward paper, or trading.

## Base binding

```text
BASE_FROZEN_DESIGN_COMMIT=61268237494e2968562983e456ea40e6f821d066
BASE_FROZEN_DESIGN_BLOB_SHA1=3bfcd695c69f6dac480f8fc99ca4f3916f668e4a
BASE_APPROVAL_RECORD_REVIEWED_SHA=a5b5a919f669221bbe2fd0010d08a3ed8690eab4
```

The future viability engine consumes an explicit frozen ordered
exchange-session calendar supplied by the reproducible data-lock stage.
`t+1`, `t+2`, and `t+3` are always the next first, second, and third dates in
that calendar, never the next available ticker observation. Existing
ticker-valid-observation rolling windows remain unchanged.

## 1. Exchange-session calendar and study-end boundary

For portfolio opening, if signal date `t` does not have both `t+1` and `t+3`
inside the frozen raw-price study window, create no pending entry and record
`END_OF_STUDY_NO_ENTRY`. This is not `NO_FILL`; never read 2026 prices or
extend the raw window.

For IC and top-decile diagnostics, target-missing dates remain undefined and
are excluded under the base design.

At `t+1` Open, skip a ranked candidate with missing, non-finite, or
non-positive raw Open and continue traversal. If none executes, record
`NO_FILL`.

Once a trade executes, missing, non-finite, or non-positive raw Close on its
exact scheduled calendar-date `t+3` terminates the run as
`DATA_QUALITY_FAILURE`. Never drop, substitute, or delay the exit.

## 2. Comparator rankings

Use these exact rankings on `RANK_ELIGIBLE(t)`:

```text
LIGHTGBM:
  predicted base-cost net-return score descending
  exact tie -> canonical 4-character code ascending by ASCII/UTF-8 byte order
  qualify only score > 0

RIDGE:
  predicted base-cost net-return score descending
  exact tie -> canonical 4-character code ascending by ASCII/UTF-8 byte order
  qualify only score > 0

SECTOR_REL_REVERSAL_1D:
  sector_rel_ret_1 ascending
  exact tie -> canonical 4-character code ascending by ASCII/UTF-8 byte order
  no sign gate

SECTOR_REL_MOMENTUM_20D:
  sector_rel_ret_20 descending
  exact tie -> canonical 4-character code ascending by ASCII/UTF-8 byte order
  no sign gate
```

All non-cash comparators use identical next-Open traversal, affordability
fallback, event order, lot size, and single-position rules.

## 3. RANDOM_500 deterministic ranking

For decimal seed `s` in `202609220000..202609220499`, signal date formatted
`YYYY-MM-DD`, and ticker code:

```text
random_key(s,t,code) =
SHA256(UTF8(decimal_seed + "|" + YYYY-MM-DD + "|" + code))
```

For each signal date and path:

1. Include every `RANK_ELIGIBLE(t)` ticker.
2. Sort `random_key` ascending as lowercase 64-hex.
3. Break an exact hash tie with canonical 4-character code ascending by ASCII/UTF-8 byte order.
4. Apply no sign or score gate.
5. When flat, freeze the full ranking after Close and traverse it at next
   Open with the same missing-Open and affordability fallback.

Do not use Python `random`, NumPy RNG, shuffle, sampling-with-replacement, or
any alternate PRNG rule. Base and stress reuse the same random rankings;
stress never rerandomizes. Criterion H remains the linear-interpolation 95th
percentile of the 500 base-cost `RANDOM_500` total net profits.

## 4. Calendar-year PnL and criteria B/N

Define end-of-session marked equity after the Close phase:

- If still open after Close, mark at raw Close with no pretend liquidation
  friction.
- If exiting at Close, use actual friction-adjusted exit proceeds.
- Exit proceeds count in end-of-session equity immediately, while remaining
  unavailable for new entry until the next session Open.

For year `y`:

```text
year_net_pnl_y =
equity_after_close(last exchange session in y)
- equity_after_close(last exchange session before y)
```

For 2020, the prior-year baseline is exactly 300000 JPY immediately before
the first 2020 OOS session.

```text
positive_calendar_year_y = year_net_pnl_y > 0
B = positive years among 2020..2025 >= 4
N = max_y(max(year_net_pnl_y,0)) / sum_y(max(year_net_pnl_y,0))
```

A zero denominator for `N` fails closed.

## 5. Reporting metric closure

For every closed trade:

```text
closed_trade_pnl =
friction-adjusted exit proceeds - friction-adjusted entry cost

closed_trade_net_return =
100 * (exit_exec / entry_exec - 1)

win = closed_trade_pnl > 0
win_rate = wins / closed_trade_count

gross_positive_profit = sum(max(closed_trade_pnl,0))
gross_negative_loss_abs = abs(sum(min(closed_trade_pnl,0)))
profit_factor = gross_positive_profit / gross_negative_loss_abs
```

If `closed_trade_count=0`, win rate is undefined. If
`gross_negative_loss_abs=0`, positive gross profit produces profit-factor
status `POSITIVE_INFINITY`, never a non-finite JSON number; otherwise profit
factor is undefined. Mean and median trade net return use
`closed_trade_net_return`.

Exposure fraction is the number of OOS exchange sessions during which a
position was held for any portion of the regular Open-to-Close session,
divided by the number of OOS exchange sessions. Both an Open-entry day and a
scheduled Close-exit day count as exposed.

## 6. Exact Q invariant set

Q is true if and only if all of the following are true:

```text
Q1_FEATURE_CAUSALITY
Q2_MONTHLY_ASOF_LABEL_CUTOFF
Q3_NO_CURRENT_MONTH_LEARNING
Q4_STAGE_B_REFERENCE_FROZEN
Q5_NO_2026_PRICE_READ
Q6_RANKING_FROZEN_BEFORE_OPEN
Q7_SINGLE_POSITION_AND_CASH_SAFETY
Q8_EXIT_EVENT_ORDER
Q9_REQUIRED_EXIT_DATA
Q10_STRESS_NO_RETRAIN_OR_RERANK
Q11_MODEL_FEATURE_CONTRACT
Q12_COMPARATOR_CONTRACT
```

The invariant meanings are:

- `Q1`: no feature or cross-sectional transform reads after signal Close.
- `Q2`: every training label's scheduled exit is fully known by the prior
  calendar month's final exchange session.
- `Q3`: no prediction-month outcome enters fit/refit; exactly one fit per
  model per prediction month.
- `Q4`: Stage B is computed once per date and never recomputed after
  transform or omission.
- `Q5`: no raw or adjusted 2026 price is read or used.
- `Q6`: next-Open order uses ranking frozen after the prior Close; Open never
  reranks.
- `Q7`: max one position, nonnegative 100-share-multiple quantity, no
  negative available cash, overlap, duplicate allocation, or double use of
  proceeds.
- `Q8`: Close exit precedes same-date after-Close signal generation; no
  same-session re-entry; proceeds become entry-available at next Open.
- `Q9`: every executed trade has an exact scheduled `t+3` valid raw Close;
  violation is `DATA_QUALITY_FAILURE`.
- `Q10`: stress reuses base-trained predictions and frozen rankings; only
  friction, affordability, quantity, and fallback may differ.
- `Q11`: exact frozen features, models, target, windows, transforms, and
  no-imputation contract; no tuning or substitution.
- `Q12`: comparators use the base design plus this amendment exactly.

If Q fails, preserve the base overall non-promotion semantics and also report
a separate `FAILURE_CLASS`, such as `DATA_QUALITY_FAILURE`,
`GOVERNANCE_FAILURE`, or `IMPLEMENTATION_FAILURE`. Never call a Q failure an
established profitability failure.

## Amendment status and required sequence

This remains the same V13 study because no outcome-bearing acquisition or run
or acceptance result has occurred. It is methodology-bearing and therefore
requires new human amendment-freeze approval.

```text
AMENDMENT_STATUS=DRAFT_UNFROZEN
IMPLEMENTATION_BLOCKED_PENDING_AMENDMENT_GPT_PASS=true
IMPLEMENTATION_BLOCKED_PENDING_HUMAN_AMENDMENT_FREEZE_APPROVAL=true
```

The next sequence is: GPT exact-SHA review; if PASS, explicit human
amendment-freeze approval; a separate amendment approval record plus GPT
review; then offline/synthetic implementation may begin.

No outcome data was accessed while preparing this draft. No base design or
approval artifact was changed.
