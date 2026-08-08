# Fixed baseline vs evaluator-v2 diagnostic report

## Scope and fairness

Both evaluators used the same immutable adjusted-OHLCV snapshot. The baseline code ran as-is at detached commit `2975e3375c615052bd3a1ab2e5a24e723e94c46b` with only its data fetch method temporarily replaced in-process. Network access was forbidden. evaluator-v2 outputs were read without changing selected rules.

The legacy research score uses validation profit *and research-internal test profit*, and its saved adoption result also inspects the reference period. Therefore `legacy_selection_uses_non_validation_data` is true. Its reference result is not a fair estimate of unknown-data performance. The v2 reference interval is also previously observed and is diagnostic only.

## Method differences

- A (`legacy_as_is`) preserves legacy labels, models, rules, stop behavior, and independent per-signal budgets.
- B (`legacy_signals_v2_portfolio`) holds legacy signals/rules fixed and applies v2 execution plus one shared portfolio.
- C (`v2_signals_independent_budget`) holds v2 signals/rules fixed but gives every signal an independent full budget; it is diagnostic only.
- D (`v2_full`) is the recorded v2 shared-portfolio result.

A-to-D differences contain interactions and are **not** claimed to be a complete additive attribution.

## Main observations

- Legacy as-is profit: 248300.00; trades: 582; max drawdown: -145155.00.
- v2 full profit: -110765.80; trades: 140; max drawdown: -123801.99.
- Legacy maximum simultaneously committed notional: 2962810.59.
- Legacy maximum capital overlap above one budget: 2662810.59.
- Legacy maximum simultaneous position equivalents: 10.
- Legacy trades participating in duplicated-capital intervals: 581.
- Alignment categories: `{"GAP_STOP_DIFFERENCE": 8, "LABEL_OR_MODEL_DIFFERENCE": 29, "NO_MATCHING_SIGNAL": 16, "PORTFOLIO_CONFLICT": 253, "RULE_DIFFERENCE": 493, "SAME": 1326, "SLIPPAGE_OR_COMMISSION_DIFFERENCE": 71, "STOP_EXECUTION_DIFFERENCE": 11}`.

Legacy can overstate attainable profit when concurrent trades each reuse the full budget, when competing same-day signals ignore rank and position limits, or when same-day exit proceeds are effectively reusable too early. It can also understate a trade when its normal-stop-price fill is worse than another convention in a non-gap case; gap-down handling can overstate fills because legacy fills at the stop rather than the lower opening-price basis. Commission and slippage rounding can move either direction.

Label/model differences are visible through changed probabilities and signals; stop-execution differences through stop categories and aligned exits; rule-selection changes through differing fixed rule tuples. Portfolio conflicts and insufficient cash explain signals that cannot become v2 trades.

The current v2 result establishes a feasible shared-cash execution path with deterministic ordering, but it does not establish superior predictive performance. Remaining uncertainty includes prior observation of the reference interval, model estimation error, market-impact realism beyond configured costs, and interactions among changed labels, rules, execution, and portfolio constraints.
