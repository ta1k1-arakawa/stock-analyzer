# V8K Layer B T1 Confirmation Design

```text
document_type=V8K_LAYER_B_T1_CONFIRMATION_DESIGN
status=PROPOSED_NOT_FROZEN
T1_ACCESS_AUTHORIZED=false
T1_CONSUMED=false
future_profitability_established=false
deployment_allowed=false
```

This is a proposed design for one untouched confirmation of the single fixed
candidate. It authorizes no T1 access, partition establishment, membership
read, raw acquisition, or human gate consumption. V8H is referenced only for
inherited T1/T2/T3 semantic background; no V8H authority, receipt, membership,
or human authorization is reused.

## Bound preregistration and source identity

```text
preregistration_commit=08bae8fd4626b71a1223657ead4ffbbaa9d0e03c
preregistration_blob=6575070833c502c5b2a71f1669cc9d9f567e2518
candidate_id=V8K_PULLBACK_VOLUME_DRY_UP_FIXED_V1
candidate_source=b954f527941fc652ad8af8b5048974e598c17e46
common_source=7e9e519056d07bcb22236b38d471862fb560637c
v5b_source=89e5ebdd3c26aeaba8ae4a45296f9270ad370a6f
universe=5a19ea918be6773e0d43d98eb5a9f3afc9920346
```

The exact candidate, ranking, and execution are those bound in the reviewed
preregistration and are not restated differently here.

## T1 semantic block

```text
purpose=ONE_UNTOUCHED_CONFIRMATION_OF_SINGLE_FIXED_CANDIDATE
block=fresh_validation_block
block_size=300
max_validation_access=1
candidate_comparison_search_tuning_on_T1=false
```

Actual V8K partition establishment and point-of-use authority remain separate
stages and are not authorized by this task. Baseline and candidate use
identical T1 tickers, prices, splits, evaluation period, capital,
costs/slippage, and execution.

## Primary T1 pass criteria

All six full-precision conditions are required, without tolerance, rounding
margin, rescue band, or borderline category:

1. `candidate.net_profit > 0`
2. `candidate.profit_factor > 1.0`
3. `candidate.net_profit >= baseline.net_profit`
4. `candidate.mtm_maximum_drawdown < baseline.mtm_maximum_drawdown`
5. `candidate.book_cost_maximum_drawdown < baseline.book_cost_maximum_drawdown`
6. `candidate.positive_year_count >= baseline.positive_year_count`

```text
all_six_primary_conditions=true => T1_RESULT=T1_CONFIRMATION_PASS
scientifically_valid_and_any_primary_condition_false=true => T1_RESULT=T1_CONFIRMATION_REJECT
```

## Secondary diagnostics

The following are recorded but never affect PASS: win rate; average win/loss;
monthly win rate; filled trade count; yearly net profit; yearly MTM/book DD;
turnover/exposure; top5 positive-profit share; maximum industry
positive-profit share; pooled/yearly Spearman; pooled score quintiles; top20
mechanism; and fill mechanism.

```text
secondary_diagnostics_nondecisional=true
```

## Failure domain and post-T1 boundary

Transport, data-quality, governance, or implementation failure is not T1
strategy rejection. Once any T1 result/outcome information is exposed, T1
access is consumed. No new fetch, resample, partition redraw, or substitute
block is permitted after exposure. Deterministic reprocessing may later be
allowed only from exactly the same durably preserved bytes/state under frozen
V8K rules; this task authorizes no such execution.

A T1 REJECT cannot tune 004, change 5/20 windows, invert the score, add
thresholds/features, substitute another Layer-A family, rerun/redraw/reallocate
T1, or use the same T1 to select a successor. Any successor-study response is
`CHATGPT_DECISION_REQUIRED`.

T1 PASS does not establish future profitability, authorize deployment, T2, or
production, or permit candidate modification. T2 remains separately gated.

## Explicitly unresolved decisions

```text
T1_CONFIRMATION_PARTITION_GENERATION_AUTHORIZATION=CHATGPT_DECISION_REQUIRED
T1_HUMAN_AUTHORIZATION_GRAMMAR=CHATGPT_DECISION_REQUIRED
T1_RECEIPT_KEY=CHATGPT_DECISION_REQUIRED
T1_PRIVATE_PATHS=CHATGPT_DECISION_REQUIRED
T1_MEMBERSHIP=CHATGPT_DECISION_REQUIRED
T2_CRITERIA=CHATGPT_DECISION_REQUIRED
PRODUCTION_CRITERIA=CHATGPT_DECISION_REQUIRED
```

This draft invents none of those mechanisms or values.
