# V6-A-R2 Confirmed Breakout Causal Execution Baseline

```text
experiment_name=V6-A-R2 Confirmed Breakout Causal Execution Baseline
experiment_type=NEW_PREREGISTERED_EXPERIMENT
derived_from=V6-A scientific hypothesis and frozen strategy rules
reason_for_new_experiment=V6-A engineering block after retry
exploratory_only=true
unused_holdout=false
deployment_allowed=false
ai_used=false
survivorship_bias=true
```

V6-A-R2 is not a parameter-tuned variant.
Its scientific candidate, ranking, execution-price, portfolio,
evaluation-period, comparison, and acceptance rules remain identical
to the frozen V6-A design.

The new registration exists solely to implement the intended causal
D0/D1/D10 event timing in a new event engine.

## Frozen scientific specification

### Hypothesis

In a broadly positive market regime, Japanese stocks breaking above
their prior 20-day high after volatility contraction, with strong
volume confirmation, can produce positive 10-trading-day realized net
returns more consistently than the V5 shallow-pullback baseline.

日本語: 広くポジティブな市場レジームにおいて、ボラティリティ収縮後に過去20日高値を上抜け、強い出来高確認を伴う日本株は、V5のshallow-pullback baselineよりも、10取引日の実現純収益をより一貫してプラスにできるかを検証する。

`exploratory_only=true`, `unused_holdout=false`, `deployment_allowed=false`,
`ai_used=false`, `survivorship_bias=true`。2020–2025年は既に別実験で確認済みであり、未使用holdoutではない。これは探索的な事前登録である。

### Fixed universe, caches, and source-aware boundary

`V4_UNIVERSE.csv`、固定300 stocksを使用する。Training cacheは
`C:\taiki\hobbies\v4-meta-label-formal-cache`（manifest SHA:
`72AE3DB1186F2C9C113B1BAFE1D37FB74A5627AC7CEED1DFC2473A24E060DE85`）、
Evaluation cacheは `C:\taiki\hobbies\v5-b-evaluation-cache-retry1`
（manifest SHA: `797265BF671AF2245A342051FFAD02AA2929D67BA885945E7762149649148AA5`）
とする。今回はキャッシュを変更・再取得せず、評価も実行しない。

Training cacheに存在するtickerは2019-12-31までtraining-cache prices、
2020-01-01以降evaluation-cache pricesを使う。evaluation-only tickerは
2019-01-01以降evaluation-cache pricesを使う。

Evaluation signalsは2020-01-01～2025-12-31。2026年は2025 signalのD10 exitに必要な価格だけを使い、2026 signalは禁止する。

### Candidate and signal rules

Feature calculationとsignal determinationはadjusted prices、executionはraw pricesを使う。各signal dateで次をすべて要求する。

1. adjusted-price historyが少なくとも252 trading days
2. raw 60-day median turnover >= 100,000,000 yen
3. raw 60-day median volume >= 50,000 shares
4. signal raw close * 100 <= 220,000 yen
5. required featuresがすべてfinite
6. D1 entryとD10 exitのraw opensが存在
7. entryとexitの間にsplitがない（split-span確認）

Market gateは、252-day historyとMA60を計算できる全Universe stocksから計算する。`breadth_above_ma60 = AdjClose > MA60 の銘柄数 / MA60計算可能銘柄数`。signal permissionは `breadth_above_ma60 >= 0.50` かつ `cross-sectional median return20 > 0`。分母100未満は `market gate BLOCKED` とし、signalを生成しない。

Individual trendは `AdjClose > MA60`、`MA20 > MA60`、`return60 > 0` をすべて満たすこと。

Volatility contractionはdaily adjusted-close returnsについて、`volatility10 = previous 10 daily returnsのstandard deviation`、`volatility60 = previous 60 daily returnsのstandard deviation`、条件 `volatility10 <= 0.80 * volatility60`。

Breakoutはsignal dayを除くpreceding 20 trading daysのadjusted close最高値を`prior_high20`とし、`AdjClose(signal) > prior_high20` かつ `0 < return1 <= 0.06`。

Volume confirmationは `volume_surprise = signal-day raw Volume / signal dayを除くprevious 20 trading daysのraw Volume median`、条件 `volume_surprise >= 1.50`。

Rankingは `ATR14_percent = adjusted ATR14 / AdjClose`、
`breakout_strength_atr = (AdjClose / prior_high20 - 1) / ATR14_percent`。
同日候補は、`breakout_strength_atr` descending、`volume_surprise` descending、
`return60` descending、`ticker` ascendingの固定辞書順で並べ、dayごとtop20のみportfolio engineへ渡す。weighted score、learning model、optimization coefficientは使わない。

### Frozen execution and portfolio rules

`D0`はsignal day、`D1`はnext trading-day raw open、`D10`はD0後10番目のtrading day（D1を1日目として数える）のraw open。entryはD1 raw open、exitはD10 raw open。

Gap conditionは `D1 raw open <= D0 raw close * 1.02`。超過時は
`ENTRY_GAP_TOO_HIGH`でskipする。entry slippageは0.03%、entry priceは
`D1 raw open * 1.0003`、quantityは100 shares。exitはD10 raw open、
exit slippageは0.03%、exit priceは `D10 raw open * 0.9997`。
TIME exitのみで、stop loss、take profit、trailing stopはない。

Starting cashは400,000円、maximum open positionsは2、cash reserveは40,000円、
per-ticker capital limitは220,000円、duplicate ticker holdingsは禁止、
same-industry concurrent holdingsは禁止、commissionは0円。sale proceedsは
次のtrading dayから利用可能で、同日sale proceeds再利用は禁止する。

Evaluation foldsは2020、2021、2022、2023、2024、2025のindependent yearly folds。各年は400,000円から開始し、年をまたいでcash/positionsをcarryしない。December signalのexitが翌年に入る場合はsignal year foldへ帰属する。

Metricsはnet profit、ending-equity equivalent、filled trade count、win rate、profit factor、average profit/loss、maximum profit/loss、monthly win rate、MTM maximum drawdown、book-cost maximum drawdown、average holding period、maximum open positions、skip reason counts、yearly profit、top 5 positive-trades profit share、maximum industry positive-profit share、signal day count、candidate count、market-gate pass/blocked day countを報告する。MTM DDとbook-cost DDは独立計算し、aggregate DDはfold別DDの最大値とする。

V5-B固定比較値は、net profit `122,536.15709488306`円、profit factor `1.1138514271409448`、MTM DD `26.782565969991488%`、filled trades `569`、positive years `3/6`。yearly profitは2020 `-27,792.634676513204`、2021 `-106,195.98642242365`、2022 `-45,253.59194076466`、2023 `+114,181.43414215161`、2024 `+102,867.2727392584`、2025 `+84,729.66325317451`円。比較項目はaggregate net profit差、profit factor差、MTM DD差、filled-trade差、yearly net-profit差、positive-year-count差。

V6-Aの20 acceptance gatesは変更しない。順に、aggregate net profit > 0、aggregate PF > 1.05、positive years >= 4/6、aggregate MTM DD <= 20%、filled trades >= 100、全year >=10 filled trades、V6 net profitがV5-B超、V6 PFがV5-B超、V6 MTM DDがV5-B未満、4/6年以上でV5-B超、top5 positive-trades profit share <=50%、maximum industry positive-profit share <=40%、negative cash count=0、same-day proceeds reuse=0、duplicate order count=0、max-position violation count=0、cash-reserve violation count=0、industry-overlap violation count=0、2026 signal count=0、two independent passes byte-identical。許可verdictは `V6_A_BREAKOUT_BASELINE_EXPLORATORY_PROMISING`、`V6_A_BREAKOUT_BASELINE_EXPLORATORY_NOT_PROMISING`、`V6_A_BREAKOUT_BASELINE_EXPLORATORY_BLOCKED` の3種類のみ。formal outputはsummary.json、trades.csv、candidates.csv、daily_equity.csvの4 artifactsのみであり、本設計段階では生成しない。

## Causal event engine contract

Engine stateは次の7つを明示的に持つ。

```text
available_cash
open_positions
pending_orders_by_entry_date
pending_proceeds_by_available_date
completed_trades
daily_equity
event_audit
```

### Candidate meaning and D0 restrictions

D0 candidate rowは売買そのものではない。固定する日付は次のとおり。

```text
signal_date=D0
order_created_date=D0
entry_attempt_date=D1
planned_exit_date=D10
```

D0で許可されるportfolio operationはpending orderをD1 queueへ登録することだけ。D0ではD1 openの読取り、cash控除、position追加、slot消費、industry枠消費、entry FILLED判定、entry SKIPPED判定を禁止する。alternative candidateは使わない。

### Fixed daily processing order

Common market calendarの各日Tを、必ず以下の順で処理する。

1. **Phase 1 — proceeds release:** `availability_date == T` のpending proceedsだけをavailable cashへ移す。Tにexitするpositionの代金は含めない。
2. **Phase 2 — D1 entry attempts:** `entry_attempt_date == T` のpending ordersを、D0で固定されたrank順、ticker tie-breakで処理する。Tにexit予定の既存positionもopen position、slot占有、industry占有として扱い、T exit proceedsは使えない。読める価格はsignal_dateのraw closeとTのraw openのみ。`T raw open <= D0 raw close * 1.02`、entry price `T raw open * 1.0003`、quantity=100、entry cost <=220000、entry後available cash >=40000、open positions <2、same ticker not open、same industry not openを要求する。FILLED時だけTにcashを控除しpositionを追加しentry timestampを記録する。SKIPPED時はposition/cashを変更しない。
3. **Phase 3 — D10 exits:** `planned_exit_date == T` の既存positionをT raw openでexitする。exit priceは `T raw open * 0.9997`。positionはTに削除し、proceedsはavailable cashへ加えず、`availability_date = Tの次のcommon-calendar trading day`としてpending proceedsへ登録する。
4. **Phase 4 — end-of-day equity:** T終了後、book equity=`available cash + open position entry cost + pending proceeds`、MTM equity=`available cash + open positionのT raw close時価 + pending proceeds`を記録する。pending ordersには価値を付けない。
5. **Phase 5 — D0 signal queue:** T closeまでの情報だけで成立したaccepted top20 candidateを、`entry_attempt_date = next common-calendar trading day`のpending orderへ登録する。このphaseでcash、position、equityを変更しない。

### Future-read guard

実装時の必須契約として、価格読取り関数は次の形式とする。

```text
read_price(ticker, requested_date, field, engine_day)
```

`requested_date > engine_day`なら必ず `FUTURE_PRICE_ACCESS_PROHIBITED` でfail closedする。D0でD1 openを読むのはfail、D1でD1 openを読むのはallowed、D10でD10 openを読むのはallowed、TでT closeを読むのはallowed。candidate生成時のD1/D10日付存在確認とsplit-span確認はportfolio state transitionから分離し、portfolio engineが未来のprice valueを読むことは禁止する。

### Required event ledger and invariants

formal `trades.csv`または内部event auditには最低限、`signal_date`、`order_created_date`、`entry_attempt_date`、`entry_state_transition_date`、`entry_price_source_date`、`planned_exit_date`、`exit_execution_date`、`exit_price_source_date`、`proceeds_available_date`、`status`、`skip_reason`、`cash_before_entry`、`cash_after_entry`、`position_count_before_entry`、`position_count_after_entry`を記録できる設計とする。

固定invariantは、`order_created_date == signal_date`、FILLED時の
`entry_state_transition_date == entry_attempt_date`、
`entry_price_source_date == entry_attempt_date`、
`exit_execution_date == planned_exit_date`、
`exit_price_source_date == planned_exit_date`、
`proceeds_available_date > exit_execution_date`。D0にFILLEDまたはcash控除が存在した場合はfail closedする。

### Golden timeline

| Engine day | Required events | Cash / position rule |
|---|---|---|
| D0 | order queued | cash unchanged; positions unchanged; equity unchanged |
| D1 | entry attempted; D1 open read | cash deducted and position opened only if FILLED |
| D10 | entry phase first; existing exit position still occupies slot; exit executed afterward | exit proceeds pending |
| D11 | proceeds released | proceeds become available |

## Acceptance test contract before implementation

次の30 testsを実装前の受入契約として固定する。

1. D0はorder queueだけが変化し、cash/positions/equityは変化しない
2. D0処理中のD1 open読取りがfuture-read guardで失敗する
3. D1当日に初めてentry fillとcash控除が起きる
4. D1 openを変更してもD0 state/D0 equityがbyte-identical
5. D1より後の価格を変更してもD0/D1 entry decisionが変化しない
6. D10当日にだけexitする
7. D10 proceedsは同日にentryへ再利用されない
8. D10 proceedsは次営業日に利用可能になる
9. 同日exit positionがentry phase中はslotを占有する
10. 同日exit positionがentry phase中はindustryを占有する
11. duplicate tickerを同時保有しない
12. max positions 2を超えない
13. cash reserve 40,000円を下回らない
14. cross-year exitはsignal year foldへ帰属する
15. D0/D1/D10のevent ledger datesが完全一致する
16. book equityとMTM equityを独立計算する
17. aggregate DDはfold別DDの最大値
18. 実測safety counters
19. 実測concentration metrics
20. accepted candidate key集合が旧V6-A preflightと完全一致
21. accepted candidate数608
22. signal days346
23. year別candidate数109/107/63/118/87/124
24. market gate pass691日/blocked774日
25. 正式4 artifactsのみ
26. two-pass byte equality
27. formal pathは明示confirmationなしで停止
28. preflightはreal-cache portfolio simulationを呼ばない
29. network処理が存在しない
30. 2026 signalをfail closed

## Human gates, review, and retry policy

**Gate 1:** design commitを人間が確認するまで実装禁止。

**Gate 2:** 実装後、synthetic golden timelineとfuture-read testsを確認するまでreal-cache preflight禁止。

**Gate 3:** candidate parityとformal-path static reviewを確認するまでformal evaluation禁止。

**Gate 4:** formal evaluationは人間が明示した1回だけ。

正式評価前のレビューでは、最低限、コード行単位で「どの日付でpending orderを作るか」「どの日付でpriceを読むか」「どの日付でcashを変更するか」「どの日付でpositionを追加/削除するか」「どの日付でproceedsを解放するか」を確認する。

retry policyは `formal attempts=1`、`internal deterministic rerun=1`、
`genuine implementation bug retry before formal run=max 1`、
`parameter tuning=0`、`post-result code correction=0`。implementation retryを使用した場合は、原因、影響、修正commitをPROJECT_STATE.mdへ記録する。

## Explicit non-actions in this design phase

Python実装、runner実装、test実装、旧V6-A code/design修正、formal evaluation、
real-cache portfolio simulation、artifact生成、cache変更・再取得、network、
AI fit、prediction、amend、rebase、force pushは行わない。
