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

## 15. Gate 1 correction addendum: frozen V6-A-R2 lineage and event boundary

The V6-A-R2 lineage used by both arms is frozen as follows:

```text
v6_a_r2_design_commit=eae60d7a472c1365afb8f8da69db7878dbf3c6a0
v6_a_r2_engine_commit=548288f9e16739fe0bff2d21996a7c53274f3e54
v6_a_r2_static_evaluator_commit=ae6e70921e0883f6f06f5fc6c1f94bc38fd48d47
v7_base_project_state_commit=cd337c952efa58abf937bdc27fb4570673b681a1
```

The following event phase order is common and frozen for both arms:

1. release proceeds available on current engine day
2. process D1 entries before same-day exits
3. process D10 exits at current raw open
4. record end-of-day equity
5. queue D0 signals for next trading day

D0 signal processing must not mutate cash, positions, pending proceeds, or
equity. The price read guard is:

```text
requested_date > engine_day => fail closed
```

A violation of lineage, phase order, or the price read guard is a causal-safety
failure and forces `V7_FORWARD_CAPACITY_BLOCKED`.

## 16. Immutable historical feature seed and forward-only study events

Historical portfolio state, historical candidate generation, historical signal
generation, historical orders, historical trades, historical profit, and
historical evaluation remain prohibited. A separately defined immutable seed is
permitted only to initialize candidate and market-gate feature history.

```text
historical_feature_seed_allowed=true
historical_feature_seed_role=FEATURE_INITIALIZATION_ONLY
historical_candidate_generation_allowed=false
historical_signal_generation_allowed=false
historical_order_generation_allowed=false
historical_portfolio_replay_allowed=false
historical_profit_calculation_allowed=false

seed_acquisition_must_occur_after_preregistration=true
seed_cutoff_before_activation_boundary=true
seed_required_valid_observations_per_ticker=252
seed_shared_identically_between_arms=true
seed_manifest_immutable_after_activation=true
seed_snapshot_replacement_allowed=false
seed_revision_recomputes_study_state=false

pre_activation_candidates=0
pre_activation_signals=0
pre_activation_orders=0
pre_activation_trades=0
pre_activation_equity_rows=0
```

The seed is used only for initial candidate-feature and market-gate-feature
history. Candidate selection, ranking, portfolio simulation, orders, fills,
profit/loss, and evaluation must not be run over the seed period. Formal V7
observations begin only at the activation boundary.

For each ticker, the seed contains only the latest 252 valid trading
observations before activation:

```text
seed_max_trading_date < activation_boundary_first_jpx_trading_date
```

Activation-day or post-activation prices must not enter the seed. If a ticker
has fewer than 252 valid observations before activation, it is ineligible at
activation. It may become eligible only after adding valid forward observations
collected after activation. Missing historical seed observations may not be
retrieved after activation.

```text
post_activation_seed_backfill_allowed=false
```

The 12-calendar-month and 24-calendar-month clocks still start at activation,
exactly as specified in this design. The study accepts that a ticker's delayed
eligibility can reduce trade counts; the study period is not extended afterward.

The market-breadth denominator continues to follow the frozen V6-A-R2 rule: it
uses only fixed-universe tickers with the required history available by that day
and for which the breadth calculation is computable. Seed and forward-history
eligibility are auditable per engine day.

## 17. Fixed study calendar and activation inputs

The study calendar is independent of individual ticker data availability. Before
activation, Gate 3 must fix the JPX trading-calendar source, timezone, calendar
version or commit, and the method used to generate target engine days in the
activation manifest.

```text
calendar_timezone=Asia/Tokyo
missing_ticker_data_does_not_remove_engine_day=true
individual_ticker_missing_data=fail_closed_or_audited_skip
calendar_rewrite_after_activation=false
```

The activation manifest must also fix the collector source, daily acquisition
window, and schema mapping. None of these may be changed after human Gate 4.
Changing the source or acquisition time blocks the study and requires a new
preregistration as a new study. A change to schema mapping or calendar
definition is treated the same way.

## 18. Canonical snapshots and revisions

For each ticker and trading date, the first snapshot that passes validation and
is appended becomes the study's canonical observation.

```text
first_validated_snapshot_is_canonical=true
canonical_snapshot_replacement_allowed=false
revision_recomputes_candidates=false
revision_recomputes_orders=false
revision_recomputes_trades=false
revision_recomputes_equity=false
missed_snapshot_backfill_allowed=false
```

A later correction may be stored as a revision record for audit purposes only.
It must not change historical candidates, orders, fills, cash, positions, or
equity. If the first acquisition is missing or invalid, that ticker/date is
fixed as missing; a later acquisition may not backfill or reconstruct it.

## 19. Daily acquisition and processing boundary

A D-day candidate is calculated only from the canonical daily snapshot acquired
and validated after the D-day market close. The D-day candidate is queued as a
pending D1 order for the next JPX trading day.

D1 open and D10 open are supplied to the paper engine when the canonical
snapshot for that engine day has been acquired. Their event timestamps and
logical phase order nevertheless remain the frozen V6-A-R2 order in Section 15.
No OHLC value from a date later than the current engine day may be read.

If the collector cannot complete daily processing, that day is recorded as
missing. The study must not later reconstruct that day's signal or portfolio
event.

## 20. Persistence and restart

The activation manifest and daily checkpoints are persisted append-only. At
minimum, each activation or checkpoint record fixes:

```text
activation_manifest_sha256
previous_checkpoint_sha256
current_checkpoint_sha256
last_completed_engine_day
arm_a_state_sha256
arm_b_state_sha256
candidate_snapshot_sha256
price_snapshot_sha256
collector_commit
```

A restart may resume only from the last normally completed checkpoint. Duplicate
processing of an engine day, partial reuse of an incomplete checkpoint, and any
state sharing between arms fail closed.

```text
duplicate_engine_day_processing=BLOCKED
checkpoint_hash_mismatch=BLOCKED
partial_day_commit=BLOCKED
```

## 21. Evaluation cutoff and trailing exits

The first day on which the formal evaluation conditions are satisfied, or the
24-calendar-month upper-limit day when the conditions have not been satisfied,
is fixed as `signal_cutoff_date`.

After D0 processing on `signal_cutoff_date`, no new signal is queued. Positions
queued or filled before the cutoff continue through their frozen D10 exit and
proceeds release. Forward collection continues until those terminal events are
complete, but no new signal is generated during that trailing period.

```text
signals_after_cutoff=0
new_entries_from_post_cutoff_signals=0
pre_cutoff_positions_follow_frozen_D10_exit=true
formal_evaluation_requires_terminal_state=true
```

Gate 5 formal evaluation occurs only after every position, pending order, and
pending proceeds record in both arms is terminal.

If, at the 24-calendar-month cutoff, either arm has fewer than 30 `CLOSED`
trades, performance gates are not evaluated and the verdict is:

```text
V7_FORWARD_CAPACITY_INCONCLUSIVE_INSUFFICIENT_TRADES
```

A data-integrity or safety violation still takes precedence and produces
`V7_FORWARD_CAPACITY_BLOCKED`.

## 22. Activation manifest

Gate 4 fixes one immutable activation manifest with these required fields:

```text
design_commit
implementation_commit
collector_commit
activation_authorization_utc
activation_boundary_first_jpx_trading_date
calendar_source
calendar_version
data_source
data_source_schema
acquisition_window_jst
universe_csv_sha
ticker_list_sha
arm_a_parameters_sha256
arm_b_parameters_sha256
shared_rules_sha256
output_root
```

The activation manifest is created and fixed only at Gate 4. It is not created
by this design correction. Once created, it cannot be changed. Any required
change after activation requires blocking the study and preregistering a new
study.

## 23. Design acceptance checks

This correction preserves the existing hypothesis, two arms, all common
conditions, study periods, nine gates, and verdict definitions. The following
machine-readable acceptance checks are explicit:

```text
single_changed_parameter=max_open_positions
historical_feature_seed=true
seed_feature_initialization_only=true
historical_candidate_generation=false
historical_portfolio_replay=false
pre_activation_study_events=0
seed_shared_between_arms=true
seed_immutable_after_activation=true
calendar_independent_of_ticker_availability=true
first_forward_snapshot_canonical=true
revision_changes_study_state=false
event_phase_order_frozen=true
terminal_state_required=true
activation_status=NOT_ACTIVATED
```

No activation manifest is created, no activation boundary is selected, and no
collector, candidate engine, paper engine, evaluation, or artifact generator
is run by this correction.

## 24. Historical feature seed cutoff and activation manifest

Gate 3 dry run must validate the seed creation procedure without activating the
study or generating study events. Gate 4 fixes the seed inputs and hashes in
the immutable activation manifest. The seed is acquired only after this
preregistration and before the activation boundary.

The seed-related activation-manifest fields are:

```text
seed_data_source
seed_data_schema
seed_acquisition_utc
seed_cutoff_trading_date
seed_ticker_count
seed_row_count
seed_payload_manifest_sha256
seed_canonical_csv_sha256
seed_generation_commit
seed_validation_result
```

The ticker-level seed manifest contains at minimum:

```text
ticker
first_seed_trading_date
last_seed_trading_date
valid_observation_count
ticker_payload_sha256
eligibility_at_activation
```

A missing or mismatched seed manifest, canonical seed, or ticker hash blocks
activation. The seed is not created, acquired, or read by this design change.

## 25. Seed identity, arm identity, and forward boundary

Both arms use exactly the same seed, candidate snapshot, and market-gate
snapshot. These are required data-integrity checks, not additional performance
gates:

```text
arm_seed_hash_equal=true
arm_candidate_input_hash_equal=true
arm_market_gate_input_hash_equal=true
```

Any mismatch produces:

```text
V7_FORWARD_CAPACITY_BLOCKED
```

The study clock and signal boundary are fixed as follows:

```text
study_clock_start=activation_boundary_first_jpx_trading_date
signals_allowed_from_activation_boundary=true
```

Signal generation for a ticker is permitted only after that ticker has 252 valid
observations available from the immutable pre-activation seed plus valid
forward observations. D0 candidates, D1 entries, D10 exits, cash, positions,
and equity record only events after the activation boundary. Seed rows never
become portfolio or evaluation events.

## 26. Canonical seed and correction policy

The seed approved at Gate 4 is canonical. Activation-time data corrections must
not change it. A correction value may be stored as a separate audit revision,
but revisions must not change any of:

```text
candidate
market gate
order
trade
cash
position
equity
evaluation
```

A material defect in the seed blocks the study. The seed must not be corrected
in place and the study must not continue with a replacement seed. Any
continuation requires a new preregistration.
