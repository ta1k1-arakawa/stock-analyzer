# V5-A2 Fixed-100 / Stop Exploratory Mechanistic Study

This is an exploratory, already-viewed-period mechanistic comparison. It is not
a final holdout evaluation and does not fit AI models or add a new candidate rule.
`exploratory_only=true`, `unused_holdout=false`, `deployment_allowed=false`, and
`ai_used=false` are fixed in every formal summary.

## Arms

Both arms consume the same V5-A candidate frame and deterministic ranked top-20
rows. Both use the V5-A D1 raw-open entry, 1% gap-up rejection, 0.03% entry and
normal-exit slippage, ¥400,000 independent fold starts, two slots, ¥40,000 cash
reserve, ¥220,000 per-entry cap, duplicate ticker/industry restrictions, pending
same-day proceeds, and signal-date fold ownership including cross-year exits.

* `FIXED100_CURRENT_STOP` uses exactly the V5-A ATR14 stop (`clamp(1.8 ATR /
  adjusted close, 4%, 8%)`) and therefore retains STOP/GAP_STOP/TIME semantics.
* `FIXED100_D5_ONLY` has no intraday or gap stop. Every accepted candidate exits
  at planned D5 raw open × 0.9997.

Both arms always buy exactly 100 shares. Risk-budget lot sizing is not used;
orders require both `entry_cost <= ¥220,000` and `entry_cost <= cash - ¥40,000`.
No candidate, slot, or trade set is reused from the V5-A formal result.

## Measurement

The event calendar extends through the maximum planned exit for each signal-owned
fold. Daily equity contains available cash, pending cash, locked entry cost,
raw-close position market value, book equity, mark-to-market equity, and open
positions. Primary drawdown is MTM; book-cost drawdown remains an audit value.
The comparison records arm-level metrics, fold metrics, skips, safety audits,
common/arm-only fills, and exit differences. Formal output is an atomic set of
`summary.json`, `trades.csv`, `daily_equity.csv`, and `comparison.csv`; production
cache execution is deliberately not run in this implementation task.
