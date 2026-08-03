# V4 Meta-Label Prototype — Pre-registered Design

## Status and scope

- `design_status`: `PRE_REGISTERED_NOT_EXECUTED`
- `research_series`: `V4_META_LABEL`
- `base_commit`: `0dd5d11cc0d118592b5a2dab5807dd10d1144b1a`
- `universe_count`: `300`
- `universe_csv_sha256`: `d40b1fcfd824822c7511f0d4f99445640706b7f5dfae08155636624704c41997`
- `ticker_list_sha256`: `12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7`
- `evaluation_type`: `SURVIVORSHIP_BIASED_RESEARCH_ONLY`
- `formal_backtest`: `false`
- `deployment_allowed`: `false`
- `shadow_allowed`: `false`
- `real_orders_allowed`: `false`
- `paid_data_allowed`: `false`

これは固定済み v4 universe を用いる独立した v4 研究であり、旧研究の再開ではない。旧研究の `CLOSED`、`NO_CANDIDATE`、`SHADOW DISABLED` を維持し、旧研究の結果、終了判定、モデル、データ、loop control を書き換えない。研究用途のみであり、売買推奨、利益保証、shadow、deploy、schedule、実注文を認めない。

## Fixed hypothesis

```text
V4 Hypothesis:

A pooled meta-label classifier can improve a deterministic
20-day momentum strategy by abstaining from trades with
low predicted probability of positive realized net return.
```

日本語説明：

全銘柄共通のメタラベル分類モデルは，20日モメンタム戦略が生成した取引候補のうち，実現純損益がプラスになる確率が低い取引を見送ることで，AIを使用しない基準戦略よりも成績を改善できる．

仮説は1件だけである。結果確認後に仮説、条件、モデル、特徴量、閾値、期間を変更しない。

## Data boundary and universe

- Universe は固定済み `V4_UNIVERSE.csv` の300銘柄である。universe取得時点は `2026-08-03` である。
- point-in-time universe ではない。現在銘柄を過去へ適用するため、生存者バイアスを持つ `SURVIVORSHIP_BIASED_RESEARCH_ONLY` であり、正式な過去時点バックテストではない。
- 価格提供元は Yahoo Finance chart API のみとし、許可hostは `query1.finance.yahoo.com` のみとする。redirectは禁止する。
- J-Quants、有料データ、旧モデル予測、旧損益、reference replay、shadow結果を入力に使用しない。
- raw cache は repository 外へ保存し、raw価格データをGitへ追加しない。
- 価格取得開始は `2015-01-01`、価格取得終了は `2019-12-31`、Yahoo の `period2` は排他的境界の `2020-01-01` とする。
- signal対象期間は `2016-04-01` から `2019-12-31` とする。
- `2020-01-01` 以降の価格行、特徴量、ラベル、結果を使用しない。

2015年のデータは、252営業日の履歴条件と60日特徴量を2016年から計算するためのwarm-upだけに使用する。

Yahooから取得する内容は raw Open、High、Low、Close、Volume、Adjusted Close、split events とする。特徴量用OHLCは `Adjusted Close / raw Close` の係数で調整し、Volumeは raw Volume とする。約定計算には raw OHLC を用いる。entryからexitまでにsplit eventが含まれる候補は除外する。

## Past-information eligibility

signal日の情報だけから、次をすべて満たす銘柄を適格とする。

1. `History_Count >= 252`
2. 過去60営業日の `raw Close × Volume` の中央値が `100,000,000円` 以上。
3. 過去60営業日の raw Volume 中央値が `50,000株` 以上。
4. signal日の `raw Close × 100 <= 300,000円`。
5. 必要な全特徴量が finite。
6. 翌営業日のentryと固定期間内のexitを確定できる。
7. entryからexitまでsplit eventをまたがない。

上場前の銘柄や履歴不足の銘柄は ticker一覧から削除せず、そのsignal日の候補からだけ除外する。

## Baseline candidate

各signal日について、次の順序で1候補だけを生成する。

1. 適格銘柄を抽出する。
2. Adjusted Close を使い、`signal日の値 ÷ 20営業日前の値 − 1` で `return_20d` を計算する。
3. `return_20d` の降順に並べる。
4. 同率は ticker 昇順で決定する。
5. 先頭1銘柄をその日の基準候補とする。

候補がない日は `NO_CANDIDATE` とする。翌日価格が初期資金条件を超えた場合も、2位銘柄へ置き換えない。

## Fixed execution rules

commit `8658ba62e3c7c02b3ddab725cbd9183ff5e4eee7` 時点の `src/free_prototype.py` と `src/trade_simulator.py` の意味に合わせ、次を固定する。

- 初期資金：`300,000円`
- 売買単位：`100株`
- 最大保有：`1銘柄`
- signal翌営業日の raw Open でentryする。
- entry slippage：`0.03%`
- stop loss：`5.00%`
- stop slippage：`0.10%`
- 通常exit slippage：`0.03%`
- commission：`0.00%`
- `future_days`：`2`
- TIME exit はentry日を1日目として2営業日目の raw Close とする。
- stop価格は slippage反映後entry価格 `× 0.95` とする。
- 当日 raw Low がstop以下ならSTOPとする。当日 raw Open がstop以下ならGAP STOPとする。
- GAP STOP基準価格は当日 raw Open、通常STOP基準価格はstop価格とする。
- STOPとGAP STOPの基準価格へ `0.10%` のexit側slippageを適用する。
- TIME Closeへ `0.03%` のexit側slippageを適用する。
- 売却代金はexit当日には再利用せず、次の処理日から利用する。
- 同日資金二重利用、negative cash、重複注文を禁止する。
- 同順位を含め ticker順序は決定的に処理する。

実現純損益率は次で固定する。

```text
realized_net_return_percent =
    (exit_price - entry_price - commission_cost) / entry_price × 100
```

commissionは `0%` だが、計算式と出力項目には残す。

## Fixed features

signal日の終了時点で確定している次の15特徴量だけを使用する。割合、return、volatilityは小数単位で統一し、途中で百分率へ変換しない。

1. `return_5d`：Adjusted Close ÷ 5営業日前Adjusted Close − 1。
2. `return_20d`：Adjusted Close ÷ 20営業日前Adjusted Close − 1。
3. `return_60d`：Adjusted Close ÷ 60営業日前Adjusted Close − 1。
4. `volatility_20d`：Adjusted Closeの日次単純収益率の過去20営業日標準偏差、`ddof=0`、年率化係数 `sqrt(252)`。
5. `volume_ratio_5d_20d`：過去5営業日の raw Volume 平均 ÷ 過去20営業日の raw Volume 平均。
6. `close_to_ma20`：Adjusted Close ÷ 過去20営業日Adjusted Close平均 − 1。
7. `close_to_ma60`：Adjusted Close ÷ 過去60営業日Adjusted Close平均 − 1。
8. `high_low_range_20d`：過去20営業日の adjusted High 最大値 ÷ adjusted Low 最小値 − 1。
9. `required_cash_ratio`：signal日の raw Close × 100 ÷ 300,000。
10. `momentum_20d_percentile_rank`：同日の適格銘柄内での `return_20d` percentile rank、`method=average`、0から1。
11. `relative_momentum_20d`：候補銘柄の `return_20d` − 同日の適格銘柄 `return_20d` 中央値。
12. `cross_section_median_return_20d`：同日の適格銘柄 `return_20d` 中央値。
13. `cross_section_breadth_above_ma20`：同日の適格銘柄のうち `Adjusted Close > MA20` である割合。
14. `cross_section_median_volatility_20d`：同日の適格銘柄 `volatility_20d` 中央値。
15. `cross_section_eligible_count`：同日の適格銘柄数。

ticker、ticker embedding、企業名、industry、market、将来のentry価格、将来のexit情報、財務情報、ニュース、SNS、旧モデル予測を特徴量に使用しない。特徴量の追加、削除、組合せ探索を禁止する。

## Labels, learning samples, and common comparison opportunities

各日の基準候補に固定約定ルールを適用する。

- positive：`realized_net_return_percent > 0`
- negative：`realized_net_return_percent <= 0`
- モデル出力：`P(realized_net_return_percent > 0)`
- ラベル確定日：`ExitDate`

各foldの学習にはtest開始日前に `ExitDate` が確定した行だけを使用する。学習にはtrain期間内に生成されたすべての日次基準候補を使用し、ポートフォリオが別取引を保有していたかどうかを学習候補の生成条件に含めない。

test評価では、まずAIを使わずに各foldを初期資金300,000円から実行し、実際に約定したBaseline取引一覧を固定する。各取引で次を固定する。

- fold
- signal_date
- ticker
- entry_date
- exit_date
- entry_price
- exit_price
- exit_reason
- realized_net_return_percent
- realized_net_profit_yen
- 15特徴量

V4はこの固定Baseline取引一覧だけを対象に、採用または見送りを判断する。V4が見送った期間中に別日や別銘柄の取引を追加せず、見送った候補を別銘柄へ置き換えず、entry価格、exit価格、exit reasonを再計算しない。V4側でも資金不足なら約定しない。違いはmeta-labelによる見送りと、それに伴う資金経路だけとする。

## Fixed model

全銘柄共通の LightGBM binary classifier を各foldにつき1個学習する。モデル仕様は1件だけである。

```text
objective = binary
n_estimators = 300
learning_rate = 0.03
num_leaves = 15
max_depth = -1
min_child_samples = 40
subsample = 0.8
subsample_freq = 1
colsample_bytree = 0.8
reg_alpha = 0.0
reg_lambda = 1.0
random_state = 20260803
n_jobs = 1
deterministic = true
force_col_wise = true
verbosity = -1
class_weight = null
```

hyperparameter tuning、モデル種類比較、class weight探索、threshold探索、probability calibration、ticker別モデル、回帰への変更、early stopping、test結果を使った再学習を禁止する。

## Walk-forward evaluation

| Fold | Train signal | Test signal |
|---|---|---|
| 1 | 2016-04-01～2016-12-31 | 2017-01-01～2017-12-31 |
| 2 | 2016-04-01～2017-12-31 | 2018-01-01～2018-12-31 |
| 3 | 2016-04-01～2018-12-31 | 2019-01-01～2019-12-31 |

test開始日前に `ExitDate` が確定していないtrain候補は禁止する。test期間の特徴量分布、ラベル率、結果を学習設定へ反映しない。OOF評価対象は2017年から2019年である。

## V4 decision rule

固定Baseline取引について、OOF probability が次を満たす場合だけV4で採用する。

```text
probability >= 0.55
```

`0.55` 未満は `ABSTAIN` とする。閾値は1件だけであり、結果確認後に変更しない。

## Data-sufficiency BLOCKED conditions

次のいずれかに該当した場合は成績判定を行わず、`FREE_META_LABEL_PROTOTYPE_BLOCKED` とする。これは成績不振を意味しない。

1. 価格取得成功tickerが150未満。
2. いずれかのfoldのtrain候補が100件未満。
3. いずれかのfoldのtrainラベルが1クラスだけ。
4. いずれかのfoldのpositiveまたはnegativeが20件未満。
5. いずれかのfoldのBaseline closed tradesが40件未満。
6. いずれかのfoldのtestラベルが1クラスだけ。
7. 必要な価格、特徴量、labelのhashを固定できない。
8. `2020-01-01` 以降の価格行を検出する。
9. 許可外network hostを検出する。
10. 決定性を確認できない。

## Required reporting metrics

固定Baseline OOF取引を対象に、分類指標として sample count、positive rate、overall ROC-AUC、fold別ROC-AUC、Brier score、log loss、probability minimum、probability maximum、probability mean、probability median、`0.55` 以上の割合、採用候補の平均realized net return、見送り候補の平均realized net return、採用候補と見送り候補の平均差を報告する。

BaselineとV4は同じ形式で、次のポートフォリオ指標を報告する。

- aggregate net profit
- aggregate ending-equity equivalent：`300,000円 + 3fold利益合計`
- 最大drawdown：3foldの最大値
- fold別net profit、fold別ending equity、fold別max drawdown
- closed trades、win rate、monthly win rate、年別net profit
- model acceptance count、model abstain count、model acceptance rate
- insufficient cash count、STOP count、GAP STOP count、TIME count
- 最大1銘柄positive profit share、上位5銘柄positive profit share、最大1業種positive profit share
- negative cash count、capital reuse count、duplicate order count

各foldは独立して300,000円から開始する。profit share の分母は銘柄または業種別に集計したpositive profitの合計とし、分母が0の場合は0とする。

## Pre-registered acceptance conditions

次の17件をすべて満たした場合だけ `FREE_META_LABEL_PROTOTYPE_PROMISING` と判定する。

1. V4の3fold合計net profitがBaselineより大きい。
2. V4の3fold合計net profitが0円より大きい。
3. V4の最大drawdownがBaselineより小さい。
4. 3fold中2fold以上でV4 net profitがBaselineを上回る。
5. 全foldでV4 max drawdownがBaseline以下。
6. V4 win rateがBaselineより高い。
7. V4 closed tradesが100件以上。
8. model acceptance rateが20%以上80%以下。
9. overall ROC-AUCが0.52より大きい。
10. 3fold中2fold以上でROC-AUCが0.50より大きい。
11. V4の最大1銘柄positive profit shareが35%以下。
12. V4の上位5銘柄positive profit shareが60%以下。
13. V4の最大1業種positive profit shareが50%以下。
14. negative cash、capital reuse、duplicate orderがBaselineとV4で全て0。
15. 2回の完全実行結果がbyte-identical。
16. `2020-01-01` 以降の価格行、特徴量、ラベル、結果が0件。
17. 許可外network通信が0件。

1件でも満たさない場合は `FREE_META_LABEL_PROTOTYPE_NOT_PROMISING` と判定する。合格条件は結果確認後に変更しない。

## Experiment budget

- 仮説：1件。
- universe：固定済み1件。
- 特徴量集合：1件。
- モデル仕様：1件。
- threshold：`0.55` の1件。
- walk-forward設計：1件。
- 完全評価：1回。
- 決定性確認の同一再実行：1回。
- 実装bug修正retry：最大1回。仕様変更ではなく、事前登録設計を実装できていない場合だけ許可する。
- 別モデル、別threshold、別特徴量、別期間、同一データでの追加loopを禁止する。
- `2020-01-01` 以降への拡張、有料データ、shadow、deploy、schedule、実注文を禁止する。

## Required reproducibility evidence at execution

将来の実装は次を保存する。

- universe hash
- 各ticker raw payload hash
- payload hash一覧のhash
- 使用価格行全体のmanifest hash
- 特徴量定義hash
- 候補一覧hash
- Baseline取引一覧hash
- OOF予測一覧hash
- model parameter hash
- 結果summary hash
- network audit
- future access audit
- 決定的な再実行比較
- 取得失敗ticker一覧
- データ除外理由別件数

raw価格そのもの、モデルpickle、注文情報はGitへcommitしない。

## Explicit non-actions for this design step

この設計作成では、JPX通信、Yahoo通信、J-Quants通信、価格取得、raw cache作成、Pythonコード作成、既存コード変更、モデル学習、バックテスト、結果ファイル作成、loop control変更、Phase B、自動化、shadow、deploy、schedule、注文を行わない。

## Design verification requirements

この設計の受入時には、変更ファイルが `V4_META_LABEL_DESIGN.md` だけであること、`V4_UNIVERSE.csv` と `V4_UNIVERSE_MANIFEST.json` がbyte-identicalであること、旧研究終了報告が不変であることを確認する。仮説1件、特徴量15件、モデル仕様1件、threshold `0.55` 1件、fold3件、BLOCKED条件10件、合格条件17件、実験予算を明示する。未決定を示す記載および文字コード破損を含めない。`2020-01-01` 以降の価格利用を許可する記述を含めない。`git diff --check` はPASSでなければならない。
