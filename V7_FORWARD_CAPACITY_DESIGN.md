# V7 Forward Capacity Study — Preregistration

## 1. Study identity and research position

This document preregisters the design of V7. It is a design artifact only. No
implementation, data collection, historical evaluation, portfolio simulation,
formal evaluation, or artifact generation is performed by this preregistration.

```text
study_name=V7_FORWARD_CAPACITY
study_type=FORWARD_ONLY_EXPLORATORY_PAPER_STUDY
historical_backtest_allowed=false
historical_replay_allowed=false
same_data_tuning_allowed=false
real_orders_allowed=false
deployment_allowed=false
ai_used=false
survivorship_bias=true
```

V6-A-R2 formal results are recorded only as motivation for this next study:

```text
V6-A-R2 net profit=93503.80047546465
V6-A-R2 PF=1.3422746366162215
V6-A-R2 MTM DD=14.94765934733312
V6-A-R2 closed trades=132
V6-A-R2 MAX_OPEN_POSITIONS skips=373
V6-A-R2 verdict=EXPLORATORY_NOT_PROMISING
```

These results do not establish that `max_open_positions=3` is advantageous.
In particular, V7 does not assert that `max3` is favorable based on V6-A-R2.
The V6-A-R2 results are not an outcome, prior pass, or tuning signal for V7.

## 2. Preregistered hypothesis

On strictly forward-only data collected after preregistration, increasing
the maximum number of concurrent positions from two to three, while
holding every other V6-A-R2 signal, ranking, execution, sizing, cash,
industry, and provenance rule fixed, can increase aggregate net profit
without materially worsening drawdown.

The hypothesis is exploratory. It does not authorize a historical comparison,
parameter search, deployment, or real-order connection.

## 3. Comparison arms

### Arm A — CONTROL

Arm A uses frozen V6-A-R2 exactly as preregistered:

```text
arm_id=ARM_A
arm_name=CONTROL
max_open_positions=2
```

### Arm B — CAPACITY_3

Arm B differs from Arm A in exactly one parameter:

```text
arm_id=ARM_B
arm_name=CAPACITY_3
max_open_positions=3
```

All of the following are identical between both arms:

```text
starting_cash=400000
quantity=100
cash_reserve=40000
capital_limit_per_position=220000
same_industry_concurrent=false
duplicate_ticker_concurrent=false
same_day_proceeds_reuse=false

entry=D1 raw open
entry_gap_limit=D0 close * 1.02
entry_slippage=0.0003

exit=D10 raw open
exit_slippage=0.0003
exit_reason=TIME
stop_loss=none

candidate_rules=frozen V6-A-R2
market_gate=frozen V6-A-R2
ranking_rules=frozen V6-A-R2
top_candidates_per_signal_day=20
```

The same candidate snapshot and price snapshot are supplied to both arms.
Any difference other than `max_open_positions=2` versus `3` is a protocol
failure. The study must fail closed if any such difference is detected or
cannot be demonstrated absent.

## 4. Forward boundary and causal rules

The study start date is the first JPX trading day after a human approves the
implementation and the forward collector is enabled following the design
commit. The activation boundary is not selected or hard-coded at the design
stage. After implementation, it must be fixed by a separate implementation
commit and a UTC timestamp, with human approval recorded before collection.

The following rules are absolute:

- no signal before activation boundary
- no retroactive signal generation
- no historical candidate backfill
- no historical portfolio reconstruction
- missed collection day remains missing
- future observations are append-only

Forward collection begins only after the activation boundary. A dry run before
activation must not create an active study boundary or produce study signals.

## 5. Universe and survivorship statement

The existing fixed 300-stock universe is used without modification:

```text
universe_csv=V4_UNIVERSE.csv
universe_csv_sha=d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997
ticker_list_sha=12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7
universe_changes_during_study=0
```

The SHA values are the fixed values of the existing V4 universe and ticker
list. No new universe selection, replacement, or survivorship correction is
permitted. `survivorship_bias=true` remains part of the study identity.

Delistings, missing data, suspended or otherwise untradeable days, and other
availability failures do not authorize adding or replacing a ticker. Such
days are either fail-closed or skipped under an auditable reason code. The
absence must remain visible in the daily snapshot manifest and relevant
safety/data-integrity audit.

## 6. Append-only data and audit storage

For every collected trading day, at minimum, the following fields are stored
append-only:

```text
acquisition_utc
source
ticker
trading_date
raw_open
raw_high
raw_low
raw_close
raw_volume
payload_sha256
daily_snapshot_sha256
collector_commit
```

The design also requires these append-only records:

- candidate snapshot
- market-gate snapshot
- both-arm event audit
- orders
- trades
- daily equity
- safety counters

Once a date's data is stored, it must not be overwritten without authorization.
If a correction is necessary, the old record is retained and a revision record
is appended with the reason, timestamp, source payload hash, prior record hash,
new record hash, and responsible collector commit. A revision is not a silent
rewrite and does not erase the original observation.

The candidate and market-gate snapshots must identify the input snapshot hashes
and the frozen rule/provenance identifiers used for both arms. The event audit
must make the same-day ordering of signal, gate, candidate, order, fill, skip,
cash, and position events auditable.

## 7. Arm-state independence

Candidate and price snapshots are shared inputs, but all portfolio and audit
state is fully separate by arm. The following state is independently initialized,
updated, persisted, and audited for Arm A and Arm B:

```text
available_cash
open_positions
pending_orders
pending_proceeds
completed_trades
daily_equity
event_audit
safety_counters
```

Neither arm may read or use the other arm's cash, positions, pending proceeds,
orders, completed trades, daily equity, event audit, or skip result. A skip in
one arm is not a skip in the other arm. Cross-arm state leakage is a safety
failure and forces a blocked verdict.

## 8. Evaluation timing and interim reporting

Formal evaluation occurs at the later of:

1. 12 calendar months after activation; and
2. 30 `CLOSED` trades in each arm.

The maximum horizon is 24 calendar months after activation. At 24 months, if
either arm has fewer than 30 `CLOSED` trades, the verdict is:

```text
V7_FORWARD_CAPACITY_INCONCLUSIVE_INSUFFICIENT_TRADES
```

Before formal evaluation, monthly reports are limited to counts, data-missing
status, and safety audits. They must not compare profit or use profit comparison
to stop, change, tune, or adopt either arm.

## 9. Fixed evaluation metrics

The following are calculated separately for both arms using the same fixed
forward horizon and deterministic definitions:

```text
net_profit
ending_equity
closed_trade_count
skipped_trade_count
win_rate
profit_factor
monthly_win_rate
MTM maximum drawdown
book-cost maximum drawdown
positive_month_count
maximum_open_positions
skip_reason_counts
top5_positive_profit_share
maximum_industry_positive_profit_share
```

The preregistered comparison values are:

```text
B_minus_A_net_profit
B_minus_A_profit_factor
B_minus_A_MTM_DD
B_minus_A_closed_trades
B_minus_A_positive_months
```

Drawdown thresholds are expressed as percentage points in the same fixed
measurement convention for both arms. `MTM maximum drawdown` is calculated
from daily mark-to-market equity; `book-cost maximum drawdown` is retained as
a separate diagnostic and does not replace the MTM gate.

## 10. Fixed decision gate

The gate order is fixed and must be evaluated in this sequence:

1. `data_integrity_pass`
2. `all_safety_counters_zero_both_arms`
3. `minimum_30_closed_trades_both_arms`
4. `B_net_profit_gt_A`
5. `B_profit_factor_ge_A`
6. `B_MTM_DD_le_20_percent`
7. `B_MTM_DD_le_A_plus_5_percentage_points`
8. `B_positive_months_ge_A`
9. `two_pass_final_artifacts_byte_identical`

Only when all nine conditions PASS is the verdict:

```text
V7_FORWARD_CAPACITY_SUPPORTED
```

If the required trade count and time condition are met but any condition 4
through 8 FAILs, the verdict is:

```text
V7_FORWARD_CAPACITY_NOT_SUPPORTED
```

Data integrity, causal-safety, or final-artifact reproducibility problems
produce:

```text
V7_FORWARD_CAPACITY_BLOCKED
```

Insufficient trade count at the maximum horizon produces the inconclusive
verdict specified in Section 8. A blocked condition takes precedence over a
performance verdict whenever both would otherwise appear applicable.

## 11. Safety requirements

For both arms, each of the following counters must be exactly zero:

```text
future_price_access
negative_cash
same_day_proceeds_reuse
duplicate_order
duplicate_ticker_open
same_industry_overlap
max_position_violation
cash_reserve_violation
capital_limit_violation
D0_state_mutation
historical_backfill
snapshot_rewrite
cross_arm_state_leakage
```

Any nonzero counter, missing counter, unverifiable counter, or unexplained
counter reset is a causal-safety/data-integrity failure and forces
`V7_FORWARD_CAPACITY_BLOCKED`.

## 12. Final evaluation artifacts

At final evaluation, the fixed artifact set is exactly:

```text
summary.json
arm_a_trades.csv
arm_b_trades.csv
arm_a_daily_equity.csv
arm_b_daily_equity.csv
candidates.csv
daily_snapshot_manifest.csv
event_audit.csv
```

Every artifact must use a deterministic record and column order, canonical
serialization, UTF-8 encoding, and SHA-256 recording. Each file must be
written atomically. Final artifact generation must be performed twice by
independent generation runs, and corresponding files must be byte-identical.
Failure of any hash, ordering, atomic-write, or byte-identity requirement is
a blocked outcome. The artifact set contains no unlisted generated file as a
study deliverable.

## 13. Human gates

The following human gates are mandatory and ordered:

1. Gate 1 — design review
2. Gate 2 — implementation and synthetic causal review
3. Gate 3 — forward collector dry run with no activation
4. Gate 4 — human activation of forward boundary
5. Gate 5 — final evaluation after fixed horizon

Without explicit human approval, the collector must not be enabled, formal
evaluation must not be performed, and no real-order connection may be made.

## 14. Explicit prohibitions for this preregistration

This change performs none of the following and authorizes none of them:

- `src` implementation
- tests implementation
- collector implementation
- GitHub Actions implementation
- network access
- data acquisition
- historical cache read
- historical backtest
- portfolio simulation
- formal evaluation
- artifact generation
- `PROJECT_STATE` changes
- V6 artifact changes
- real-order code

The design phase has no forward activation boundary. It creates no signal,
candidate, order, trade, equity, or evaluation artifact.

