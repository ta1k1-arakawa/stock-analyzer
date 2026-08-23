# V9 Cross-Sectional Close-Auction Charter

```text
study=V9_CROSS_SECTIONAL_CLOSE_AUCTION
status=PREFREEZE_DRAFT
design_frozen=false
execution_authorized=false
parent_decision=V8K termination reviewed SHA 3791dfa421c54555acc066346c810ee4cf8c95b5
future_profitability_established=false
```

## Objective

Maximize prospective real-world net expected value and robustness under a
30–40万円 capital constraint, rather than historical backtest profit.

## Chosen architectural direction

- Japanese equities using an SBI S-share / odd-lot compatible portfolio concept.
- Close-auction / close-to-close execution architecture, 1-share granularity,
  and approximately equal-notional allocation.
- Pooled cross-sectional scoring rather than per-ticker models; ticker identity
  is excluded from model features.
- Short multi-day holding architecture.
- Ridge is the mandatory simple baseline; shallow LightGBM is a candidate.
- TOPIX, equal-weight-universe, and random-K baselines are required.
- No optimistic stop-price execution model. Measure fill/non-fill and
  operational execution mismatch.
- Costs, turnover, drawdown, robustness, capacity, and reproducibility matter.

Planning heuristics such as `+3.25% alpha` or `3x expected profit` are not
frozen assumptions or evidence. Future profitability is not established.

## Prior-evidence treatment

- `V8_DATA_EXPOSURE_AUDIT.md` is inherited as provenance.
- Existing outcome-exposed historical data may be development evidence only
  where a future frozen design permits; it must never be relabeled untouched
  OOS.
- Current-constituent historical universes carry survivorship bias. Point-in-
  time universe treatment must be resolved before formal historical claims.
- The audit identifies cross-sectional, not temporal, evidence as the strongest
  unused historical axis.
- V7/V8 forward, private, and sealed boundaries are not automatically inherited.

## V8K non-inheritance boundary

V9 must not inherit or reuse V8K Stage-2 authorization, T1 authorization,
private-partition gate, seed, private partition, or one-shot gate state. Safe
public provenance or infrastructure may be adopted only after V9's own design
explicitly binds it.

## GPT methodology decisions required before freeze

No model fitting, backtest, profit calculation, Yahoo/JPX/broker network,
private/sealed read, human-gate consumption, or V9 design freeze is authorized.
For every item below, if the choice is not explicitly supplied:
`CHATGPT_DECISION_REQUIRED`.

1. Exact universe construction and point-in-time/survivorship treatment.
2. Exact fresh-validation, sealed-OOS, and forward evidence architecture.
3. Exact number of concurrent positions.
4. Exact rebalance schedule.
5. Exact primary holding period.
6. Exact label/target definition.
7. Exact feature set and treatment of `volume_dryup`.
8. Exact Ridge and LightGBM hyperparameters.
9. Exact transaction-cost/fill model.
10. Exact benchmarks and random-K protocol.
11. Exact promotion/rejection criteria.
12. Stop/stopping rule.
13. Paper-trading and small-capital forward gates.
