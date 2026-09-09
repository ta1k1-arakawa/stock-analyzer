# V6-A Confirmed Breakout Baseline

## hypothesis

In a broadly positive market regime, Japanese stocks breaking above
their prior 20-day high after volatility contraction, with strong
volume confirmation, can produce positive 10-trading-day realized net
returns more consistently than the V5 shallow-pullback baseline.

日本語: 広くポジティブな市場レジームにおいて、ボラティリティ収縮後に過去20日高値を上抜け、強い出来高確認を伴う日本株は、V5のshallow-pullback baselineよりも、10取引日の実現純収益をより一貫してプラスにできるかを検証する。

## exploratory status

```text
exploratory_only=true
unused_holdout=false
deployment_allowed=false
ai_used=false
survivorship_bias=true
```

2020–2025年は既に別実験で確認済みであり、未使用holdoutではない。これは探索的な事前登録である。

## fixed caches and hashes

V5-Bと同じsource-aware価格frameを使用する。今回はキャッシュを読み込まず、評価も実行しない。

Training cache: `C:\taiki\hobbies\v4-meta-label-formal-cache`

- manifest SHA: `72AE3DB1186F2C9C113B1BAFE1D37FB74A5627AC7CEED1DFC2473A24E060DE85`

Evaluation cache: `C:\taiki\hobbies\v5-b-evaluation-cache-retry1`

- manifest SHA: `797265BF671AF2245A342051FFAD02AA2929D67BA885945E7762149649148AA5`

## source boundary

For tickers present in the training cache, use training-cache prices through 2019-12-31 and evaluation-cache prices from 2020-01-01 onward. For evaluation-only tickers, use evaluation-cache prices from 2019-01-01 onward.

Evaluation signals are 2020-01-01 through 2025-12-31. January 2026 prices may be used only as needed for D10 exits of 2025 year-end signals. 2026 signals are prohibited.

## universe

`V4_UNIVERSE.csv`, fixed 300 stocks.

## candidate eligibility

Feature calculation and signal determination use adjusted prices; execution uses raw prices. On each signal date all of the following are required:

1. adjusted-price history of at least 252 trading days
2. raw 60-day median turnover >= 100,000,000 yen
3. raw 60-day median volume >= 50,000 shares
4. signal raw close * 100 <= 220,000 yen
5. all required features are finite
6. raw opens required for D1 entry and D10 exit exist
7. no split between entry and exit

## market gate

The gate is computed from all Universe stocks for which 252-day history and MA60 are calculable, not only from eligible candidates.

`breadth_above_ma60 = number of stocks with AdjClose > MA60 / number of stocks for which MA60 is calculable`

Signal permission requires `breadth_above_ma60 >= 0.50` and `cross-sectional median return20 > 0`. If the denominator is fewer than 100 stocks, the day is `market gate BLOCKED` and no signals are generated.

## trend conditions

All individual conditions must hold:

- AdjClose > MA60
- MA20 > MA60
- return60 > 0

## volatility contraction

From daily returns of adjusted close:

- `volatility10 = standard deviation of the previous 10 daily returns`
- `volatility60 = standard deviation of the previous 60 daily returns`
- condition: `volatility10 <= 0.80 * volatility60`

## breakout conditions

`prior_high20` is the highest adjusted close in the preceding 20 trading days, excluding the signal day. Required conditions:

- `AdjClose(signal) > prior_high20`
- `0 < return1 <= 0.06`

## volume confirmation

`volume_surprise = signal-day raw Volume / median raw Volume over the previous 20 trading days, excluding the signal day`.

Required condition: `volume_surprise >= 1.50`.

## ranking

```text
ATR14_percent = adjusted ATR14 / AdjClose
breakout_strength_atr = (AdjClose / prior_high20 - 1) / ATR14_percent
```

Same-day candidates are ordered by this fixed dictionary order:

1. `breakout_strength_atr` descending
2. `volume_surprise` descending
3. `return60` descending
4. `ticker` ascending

Only the top 20 candidates per day are passed to the portfolio engine. No weighted score, learning model, or optimization coefficient is used.

## execution

`D0` is the signal day. `D1` is the next trading-day raw open. `D10` is the raw open on the tenth trading day after D0, counting D1 as trading-day 1. Entry is at D1 raw open and exit is at D10 raw open.

Gap condition: `D1 raw open <= D0 raw close * 1.02`. If exceeded, skip with `ENTRY_GAP_TOO_HIGH`; alternative candidates are tried in fixed ranking order.

- entry price: `D1 raw open * 1.0003`
- exit price: `D10 raw open * 0.9997`
- 10-trading-day TIME exit only
- no stop loss
- no take profit
- no trailing stop
- sale proceeds become usable from the next trading day
- entry slippage=0.03%
- exit slippage=0.03%

## portfolio

- starting cash=400,000 yen
- quantity=100 shares
- maximum open positions=2
- cash reserve=40,000 yen
- per-ticker capital limit=220,000 yen
- duplicate ticker holdings prohibited
- same-industry concurrent holdings prohibited
- same-day sale proceeds reuse prohibited
- commission=0

## fold ownership

Evaluation folds are 2020, 2021, 2022, 2023, 2024, and 2025. Each year starts independently with starting cash=400,000 yen. No prior-year cash or positions carry over.

## cross-year exits

If a December signal's D10 exit enters the following year, process through the exit and attribute the profit to the signal year. January 2026 prices are permitted only for such 2025 signal exits; 2026 signals are prohibited.

## metrics

For aggregate and each year, report:

- net profit
- ending-equity equivalent
- filled trade count
- win rate
- profit factor
- average profit
- average loss
- maximum profit
- maximum loss
- monthly win rate
- mark-to-market maximum drawdown
- book-cost maximum drawdown
- average holding period
- maximum open positions
- skip reason counts
- yearly profit

Also report top 5 positive-trades profit share, maximum industry positive-profit share, signal day count, candidate count, market-gate pass day count, and market-gate blocked day count.

MTM DD and book-cost DD must be calculated separately; the same formula must not be reused for both.

## fixed V5-B comparator

V5-B must not be rerun. The fixed comparison target is:

- net profit: 122,536.15709488306 yen
- profit factor: 1.1138514271409448
- MTM DD: 26.782565969991488%
- filled trades: 569
- positive years: 3/6

Yearly profit:

| Year | Profit |
|---|---:|
| 2020 | -27,792.634676513204 |
| 2021 | -106,195.98642242365 |
| 2022 | -45,253.59194076466 |
| 2023 | +114,181.43414215161 |
| 2024 | +102,867.2727392584 |
| 2025 | +84,729.66325317451 |

Compare aggregate net profit difference, profit factor difference, MTM DD difference, filled-trade difference, yearly net-profit differences, and positive-year-count difference.

## acceptance gates

V6-A is PROMISING only if all 20 conditions pass:

1. aggregate net profit > 0
2. aggregate profit factor > 1.05
3. positive years >= 4/6
4. aggregate MTM DD <= 20%
5. filled trades >= 100
6. every year has at least 10 filled trades
7. V6 net profit > V5-B baseline net profit
8. V6 profit factor > V5-B baseline profit factor
9. V6 MTM DD < V5-B baseline MTM DD
10. V6 beats the V5-B baseline in at least 4/6 years
11. top 5 positive-trades profit share <= 50%
12. maximum industry positive-profit share <= 40%
13. negative cash count = 0
14. same-day proceeds reuse count = 0
15. duplicate order count = 0
16. max-position violation count = 0
17. cash-reserve violation count = 0
18. industry-overlap violation count = 0
19. 2026 signal count = 0
20. two independent passes are byte-identical

## verdicts

Allowed verdicts are exactly:

- `V6_A_BREAKOUT_BASELINE_EXPLORATORY_PROMISING`
- `V6_A_BREAKOUT_BASELINE_EXPLORATORY_NOT_PROMISING`
- `V6_A_BREAKOUT_BASELINE_EXPLORATORY_BLOCKED`

If there is a problem with data, execution, hash, causality, or artifact generation, do not make a performance judgment; use `V6_A_BREAKOUT_BASELINE_EXPLORATORY_BLOCKED`.

## blocked conditions

Any data, execution, hash, causality, or artifact-generation problem blocks the evaluation. A failure caused by the environment that does not reach scientific evaluation is not a formal result. No threshold, candidate condition, or frozen rule may be changed after observing a result.

## planned output schemas

At formal evaluation, exactly these four files are planned:

- `summary.json`
- `trades.csv`
- `candidates.csv`
- `daily_equity.csv`

No planned artifact is created in this preregistration stage.

## one-shot policy

- one preregistered production attempt
- one deterministic internal rerun
- maximum one genuine implementation-bug retry
- no parameter tuning
- no threshold changes
- no candidate-condition changes after seeing the result
- no AI
- no deployment

If the verdict is NOT_PROMISING, end this breakout family without fine-grained threshold tuning.

## no-AI declaration

`ai_used=false`. No AI fit, prediction, learned ranker, weighted score, or optimization coefficient is part of V6-A.

## no-deployment declaration

`deployment_allowed=false`. No deployment, live trading, automatic orders, broker integration, or order placement is authorized by this design.

