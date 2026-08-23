# V8K Layer B T1 Candidate Preregistration

```text
document_type=V8K_LAYER_B_T1_CANDIDATE_PREREGISTRATION
status=PROPOSED_NOT_FROZEN
evidence_capacity=ZERO
future_profitability_established=false
deployment_allowed=false
T1_access_authorized=false
T1_consumed=false
LAYER_A_SIMPLE_RANKING_EXPLORATION=CLOSED
NO_MORE_LAYER_A_SIMPLE_RANKING_SEARCH=true
NO_POST_SELECTION_TUNING=true
```

This records the completed Layer A search and proposes an exact candidate for
a later human-reviewed freeze. It authorizes no T1 access. Selection across
five exposed Layer-A families creates search/selection overfitting risk.
Layer-A results have ZERO confirmatory evidence.

## Completed Layer A search

| Measurement | Family | Decision | Scorecard SHA-256 |
|---|---|---|---|
| 001 | INDUSTRY_RELATIVE_RESIDUAL_MOMENTUM | REJECT | F90C4B36064265408D198D52A4A2D07628742B76A0D7E78CE0E0C802D88E083E |
| 002 | INDUSTRY_RELATIVE_SHORT_HORIZON_REVERSAL | REJECT | CD5A53128E36A827FE67A0100A424D82A8376DD968D327829D468170C9D5C104 |
| 003 | VOLATILITY_ADJUSTED_60D_MOMENTUM | REJECT | 35615575918D7F4D4F6596D7935F44A169457841B102AB2DBA53C997652E6DD4 |
| 004 | PULLBACK_VOLUME_DRY_UP | SURVIVES_VIABILITY_SCREEN_WEAK | 0ED36916F2B686DD26281A2293E3A93E8143A1378A09A6E015B7A9D2B6093511 |
| 005 | TREND_PERSISTENCE_10D | REJECT | FCB9905B821173BEABADF003FF03E5CA9E0433DEC57F8027994F82ACDADD4B08 |

```text
search_family_count=5
selected_candidate_count=1
```

## Selected candidate

```text
candidate_id=V8K_PULLBACK_VOLUME_DRY_UP_FIXED_V1
source_family=Measurement_004_PULLBACK_VOLUME_DRY_UP
model=NONE
fitted_parameters=NONE
threshold=NONE
fallback=NONE
industry_adjustment=NONE
alternate_volume_window=NONE
```

At exact signal date `D0`:

```text
volume_ratio_5_20 =
  mean(raw Volume over last 5 observations including D0)
  / mean(raw Volume over last 20 observations including D0)

volume_dryup_score = 1.0 - volume_ratio_5_20
higher_score_ranks_higher=true
```

Eligibility is exact frozen V5-A2/V5-B eligibility. Baseline ranking is
`return_60d DESC`, `return_20d DESC`, `ticker ASC`, top 20 per signal date.
Candidate ranking is `volume_dryup_score DESC`, `return_60d DESC`,
`return_20d DESC`, `ticker ASC`, top 20 per signal date.

Execution is exact existing V5-B/V5-A2 fixed-100 D1-D5 execution, including
the existing gap, cost/slippage, cash, maximum-position, industry/ticker
duplication, and same-day-proceeds rules.

## Bound source identity

```text
repository_commit_at_draft=a4f2b29de46142857c05456e1a6c7807802838ff
src/v8k_layer_a_pullback_volume_dryup.py=b954f527941fc652ad8af8b5048974e598c17e46
src/v8k_layer_a_volatility_adjusted_momentum.py=7e9e519056d07bcb22236b38d471862fb560637c
src/v5_b_candidate_ranker.py=89e5ebdd3c26aeaba8ae4a45296f9270ad370a6f
V4_UNIVERSE.csv=5a19ea918be6773e0d43d98eb5a9f3afc9920346
```

## Why 004 survived the viability screen

These observations are descriptive, not tuned thresholds or a profitability
claim.

| Metric | Baseline | 004 |
|---|---:|---:|
| net profit | 120114.48709488308 | 109327.60155540849 |
| profit factor | 1.1114860129807256 | 1.1059452001766283 |
| MTM maximum drawdown | 26.782565969991488 | 18.19919317974997 |
| book-cost maximum drawdown | 26.54899660560596 | 18.19919317974997 |
| positive years | 3 | 4 |
| win rate | 0.49733570159857904 | 0.5115452930728241 |

```text
all_eligible_pooled_spearman=0.009806632523639723
```

- Historical net profit did NOT improve.
- Score discrimination is weak.
- The meaningful observed feature was materially lower drawdown while retaining
  most historical net profit.
- This is only a reason to submit the exact frozen candidate to untouched
  confirmation, not evidence that it will work in future.

## Freeze boundary

After human approval of a later reviewed preregistration, the following are
forbidden before and after T1: changing the 5/20 volume windows or score sign;
adding thresholds; combining Measurement 001-005 scores; changing eligibility,
top20, execution/cost/slippage, or capacity; feature engineering based on T1;
or retry/redraw/substitution based on outcome.

```text
T1_CONFIRMATION_DESIGN=CHATGPT_DECISION_REQUIRED
T1_HUMAN_AUTHORIZATION=REQUIRED_FRESH_AT_POINT_OF_USE
```

This draft does not invent T1 pass thresholds, T1 partition/allocation, T1
gate grammar, T1 retry rules, or promotion rules. A separate ChatGPT
methodology decision is required before T1.
