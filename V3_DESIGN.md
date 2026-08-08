# stock-analyzer v3 設計書

## 0. 文書の位置付け

- **[確定]** 設計基準は `train-label-alignment-fix` の `dab43d97887900dda7052013caf2d866bb2a19cd` とする。
- **[確定]** 本書は設計のみを扱う。コード、設定、データ、モデル、既存評価結果、既存タグを変更しない。
- **[確定]** v1/v2、LOOP-000〜004、reference replay、shadowの結果は履歴として保存し、後から意味や数値を書き換えない。
- **[確定]** 文中のラベルは次の意味を持つ。
  - **[確定]**: 既存監査、既存運用制約、またはユーザー指示で固定済み。
  - **[提案]**: v3実装前に採否を決める設計案。結果を見て変更しない。
  - **[未決]**: 実装開始前にユーザー判断またはデータ提供可否の確認が必要。

## 1. 現在のv1/v2系で確定した問題

1. **[確定] 固定8銘柄への依存**: 開発・評価対象が8銘柄だけであり、銘柄固有の偶然、業種偏り、ETF混在の影響を分離できていない。
2. **[確定] 銘柄別モデル**: 銘柄ごとに別モデルを学習したため、1モデル当たりのサンプル数が少なく、新規銘柄へ一般化できない。
3. **[確定] base rateと出力分布の差**: 銘柄ごとにpositive label率、raw probabilityの平均・分散・範囲が大きく異なった。
4. **[確定] raw probabilityはNOT_COMPARABLE**: 銘柄別モデルのraw probabilityを、同じ成功確率を表す数値として銘柄間で直接比較できなかった。
5. **[確定] 識別力不足**: 多くの銘柄・foldでROC-AUCが0.5前後だった。
6. **[確定] 収益との関係不足**: probabilityと実現純収益率のSpearman相関はほぼ0だった。
7. **[確定] LOOP-003の意味不整合**: `target_net_return`を含む期待値式が、実際には存在しないtarget利確出口を前提としていた。実約定はSTOP、GAP STOP、TIMEであり、scoreと実現可能な出口が一致していなかった。
8. **[確定] LOOP-004はREJECT**: nested OOFのPlatt校正はBrier、log loss、ECEを概ね改善したが、ポートフォリオ利益、月次勝率、銘柄依存率を改善せず、事前条件を満たさなかった。
9. **[確定] referenceの非独立性**: 2025-04-01〜2026-07-31は結果確認済みのREFERENCE ONLY期間であり、完全な未観測テストではない。
10. **[確定] shadowはBLOCKED**: 現在のshadow scheduleは有効化してはいけない。既存shadow候補をv3へ黙って継承しない。
11. **[確定] 日次ラベル引数**: `train.py`からstop loss、stop slippage、commissionがラベル生成へ渡らない不一致は `dab43d97887900dda7052013caf2d866bb2a19cd` で修正済み。

## 2. v3の目的と非目的

### 2.1 目的

v3は次の6問題を順序どおり分離して解く。

1. 各時点で監視可能だった銘柄集合をpoint-in-timeで定義する。
2. 各銘柄について、実際の約定結果と一致する将来純収益を予測する。
3. 異なる銘柄を、意味の一致した共通の収益率尺度で比較する。
4. 30万円、100株、最大1ポジションの制約下で、最高順位を買うか何も買わないかを決める。
5. STOP、GAP STOP、TIME決済、費用をラベルとバックテストで同じ共通関数から算出する。
6. point-in-time universeとnested walk-forwardにより、研究期間への過適合を測定する。

### 2.2 非目的

- **[確定]** 実注文、証券口座接続、資金移動はv3の範囲外。
- **[確定]** 複数同時保有、信用取引、空売り、日中足を使った執行最適化は扱わない。
- **[確定]** target価格到達による自動利確出口を追加しない。
- **[確定]** 2025-04-01〜2026-07-31の結果を使ってモデル、特徴量、閾値、universe、採用条件を改善しない。
- **[確定]** v1/v2の結果、タグ、LOOP判定を再評価・改称しない。
- **[確定]** 「順位式だけを微修正する」実験系列を継続しない。

## 3. v3第一候補

### 3.1 採用する第一候補

- **[提案] モデル設計**: 案B「複数銘柄の共通LightGBM回帰モデル」。全point-in-time eligible銘柄の行を一つの学習表へまとめる。
- **[提案] 予測対象**: 手数料、entry/exit/stop slippage、通常STOP、GAP STOP、TIME決済を含む、1取引の`realized_net_return_percent`。
- **[提案] 売買判断**: 各signal dateで購入可能な候補を予測純収益率降順、同値は銘柄コード昇順に並べる。最高値が0%以下なら取引しない。最高候補が寄付き時点で資金不足なら次順位を順に検討する。
- **[提案] 識別子**: 第一候補ではraw銘柄コードや銘柄embeddingを使わず、point-in-time業種、時価総額区分、流動性等を使う。これにより未学習銘柄への適用可能性を残す。銘柄IDは最大4特徴量群の一つとしてのみ事前登録可能とする。

この案はLOOP-001の「8個の銘柄別回帰モデル」とは異なる。データ共有、広いpoint-in-time universe、共通ラベル、共通モデルによって、銘柄間比較をモデル構造と予測単位の両方で統一する。

### 3.2 第一候補を選ぶ設計上の理由

1. 目的変数が全銘柄で同じ「費用控除後%収益」であり、確率校正を介さず比較できる。
2. 銘柄別モデルより学習行数が増え、低頻度positive labelへの依存もなくなる。
3. 案Cより実装・監査が単純で、STOP/GAP STOP/TIMEとの一致を行単位で検証できる。
4. 30万円・1ポジションではtop-1の順序が重要だが、回帰値は「取引しない」という0%基準も自然に表現できる。
5. 新規銘柄でも、履歴特徴量とpoint-in-time属性が揃えば推論できる。

## 4. 設計候補の比較

| 観点 | 案A: 銘柄別モデル | 案B: 複数銘柄共通モデル | 案C: クロスセクショナル順位学習 |
|---|---|---|---|
| 予測対象 | 銘柄ごとの確率または収益 | 全銘柄共通の実現純収益率 | 同日内の収益順位、pair、top銘柄 |
| 必要データ量 | 銘柄ごとに十分な長期履歴 | 銘柄数×日数を共有可能 | 各日に十分な同時候補が必要 |
| 銘柄間比較 | 校正・尺度変換が別途必要 | 共通目的変数なら自然 | 学習目的自体が比較なので最も自然 |
| 過学習リスク | 銘柄固有ノイズが高い | 大型銘柄・特定日・業種への偏り | 日付group、universe変動、pair増幅への過適合 |
| 実装難易度 | 低い | 中程度 | 高い |
| 説明可能性 | 銘柄ごとには説明可能、横比較は困難 | 共通SHAPと属性別誤差を説明可能 | pairwise/LTR scoreの金銭的意味が弱い |
| 30万円運用との相性 | 横比較が弱い | 予測%でtop-1とno-tradeを決めやすい | top-1には適するが絶対的なno-trade判断が別途必要 |
| 新規銘柄 | モデルがなく対応不可 | ID非依存特徴なら対応可能 | 学習時と同等の同日候補集合が必要 |
| 主な欠点 | データ不足、校正差、保守コスト | regime差、横断面相関、属性欠損への対処が必要 | 複雑、group漏洩、絶対収益が負でも1位になり得る |
| 採用時の検証 | 銘柄別OOF校正と横断比較 | purged nested walk-forward、銘柄/業種holdout | date-grouped purged CV、NDCG/top-k、絶対収益ゲート |
| v3での扱い | 第一候補にしない | **第一候補** | 実験予算内の第2候補。B失敗後もREFERENCEは使わない |

案Aは既知のNOT_COMPARABLE問題を再発させるため不採用。案Cは目的との整合性は高いが、point-in-time universeとdate groupingが完成する前に導入すると誤ったgroup比較を学習する。したがってBを先に監査し、Cは研究期間内の事前登録された1候補に限定する。

## 5. 銘柄ユニバース

### 5.1 対象と定期更新

- **[提案] 対象市場**: 東京証券取引所PrimeおよびStandardの普通株。Growth、ETF、ETN、REIT、優先株、外国株は第一候補から除外する。
- **[提案] 銘柄数目安**: 各月のpoint-in-time条件適用後で150〜400銘柄。数を目的に条件を緩めず、実数と除外理由を毎月保存する。
- **[提案] universe更新**: 月1回、前月最終取引日終了時点で翌月の基本universeを固定。取引停止、上場廃止、価格・資金不足は日次overlayでfail-closed除外する。
- **[提案] 最低売買代金**: 過去60取引日の中央値が1億円/日以上。
- **[提案] 最低出来高**: 過去60取引日の中央値が5万株/日以上。
- **[提案] 上場期間**: signal date時点で252取引日以上。ただし企業再編後の履歴接続可否は別監査する。
- **[提案] 購入可能性**: signal dateに既知の終値で100株＋想定entry費用が30万円以下を候補条件とする。実際の翌日始値で不足した場合は`SKIPPED_INSUFFICIENT_CASH`として次順位へ進む。
- **[提案] 業種偏り**: 業種を除外条件にはせず、学習weightと評価指標で偏りを監視する。特定業種がeligible universeの30%を超えた月は警告する。

閾値は実装前に固定し、研究結果を見て変更しない。上記値はデータ取得前の提案値であり、ユーザー承認が必要である。

### 5.2 異常・企業行動

- **[確定]** 取引停止日は新規注文を作らず、理由を保存する。既存positionの価格が得られない日は推測決済しない。
- **[提案]** 整理銘柄指定または上場廃止がsignal時点で公表済みなら新規候補から除外する。将来の上場廃止を知って過去行を削除しない。
- **[提案]** 上場廃止銘柄の過去データも研究snapshotに残し、delisting returnまたは最終取引価格の扱いをmanifestで明示する。
- **[確定]** 株式分割・併合はOHLCVと出来高を一貫して調整する。価格値の調整方式、corporate action、取得時刻、hashをmanifestへ保存する。
- **[提案]** 特徴量用価格は、feature date時点までに発効したcorporate actionだけから再構築するpoint-in-time adjusted seriesを優先する。将来の分割係数で過去snapshotを書き換えない。
- **[確定]** 必須日欠測、重複、OHLC不整合、timezone不一致、hash不一致は当該銘柄をfail-closedにし、欠測理由を保存する。
- **[提案]** ETFは個別株と混ぜない。後に扱う場合は別universe・別モデル・別評価として事前登録する。

### 5.3 survivorship bias対策

- **[提案] 第一選択**: 過去時点の上場・市場区分・業種・取引停止・上場廃止・corporate actionを持つpoint-in-time security masterを使用する。各signal dateで、その時点に存在し条件を満たした銘柄だけを候補にする。
- **[確定]** 現在の構成銘柄一覧を過去全期間へ遡及適用しない。
- **[提案] 代替策**: point-in-time構成を取得できない場合、研究期間中に一度でも観測された銘柄の和集合を作り、上場日・最終取引日・価格存在期間だけで各日universeを近似する。上場廃止銘柄を可能な限り追加し、coverage reportを出す。
- **[限界]** 代替策では、過去の市場区分、整理指定、停止、社名・コード変更、未取得の上場廃止銘柄を完全再現できず、survivorship biasは残る。この場合の結果は`UNIVERSE_BIASED_DIAGNOSTIC`でありrelease candidate採用に使わない。

## 6. データソース要件

### 6.1 必須データ

1. 調整前OHLCV、corporate action、取引カレンダー。
2. point-in-time上場銘柄master、市場区分、上場・廃止日、取引停止状態。
3. point-in-time業種分類、発行済株式数または時価総額区分。
4. TOPIX等の市場指数OHLCVと、可能なら業種指数。
5. 企業情報を使う場合は公表日時・利用可能日時を含むpoint-in-time値。

### 6.2 snapshot要件

- **[確定]** source、取得時刻、requested/actual期間、timezone、列、調整方式、ファイル別SHA-256、全体snapshot hashを保存する。
- **[確定]** 学習・評価中にAPIへfallbackしない。不足・hash不一致・期間不足は即時停止する。
- **[提案]** raw、point-in-time変換後、feature-readyの3層を分け、それぞれimmutable manifestを持つ。
- **[提案]** license、訂正履歴、delisted coverage、corporate-action revisionをデータ監査成果物へ記録する。

## 7. 特徴量設計

最大4特徴量群を次の順で事前登録する。

1. **G1 既存価格・テクニカル [提案]**: 既存SMA乖離、RSI、MACD比率、BB位置、ATR比率、1/3/5日return、出来高変化。計算はsignal日の確定終値まで。
2. **G2 流動性・リスク [提案]**: trailing売買代金、出来高、実現volatility、ATR、overnight gap、価格水準、100株必要資金。全て過去rolling値。
3. **G3 市場相対・regime [提案]**: 市場指数・業種指数に対する1/5/20/60日相対return、rolling beta、market volatility、breadth。point-in-time universeだけで算出する。
4. **G4 point-in-time属性 [提案]**: 業種、時価総額bucket、市場区分、上場経過日数。銘柄コード/embeddingを試すならG4内の1回に限定する。

- **[確定]** 将来公表される企業情報、将来universe、将来調整係数を特徴量へ入れない。
- **[提案]** 横断面rankやz-scoreは、そのsignal dateのeligible universeだけで算出する。
- **[提案]** 欠測補完値は各foldのtraining dataだけでfitし、欠測indicatorを併設する。全期間統計で補完しない。
- **[提案]** 第一候補はG1+G2+G3を基本とし、G4追加は特徴量予算内の1比較とする。

## 8. 予測対象と約定仕様

### 8.1 共通約定

signal dateを`t`、翌取引日をentry dayとする。

1. **[確定] Entry**: `t+1`の始値にentry slippageを加え、100株単位で購入する。
2. **[確定] 通常STOP**: 当日始値がstop価格より上で安値がstop以下なら、stop価格からstop slippageを引いた価格で決済する。
3. **[確定] GAP STOP**: 当日始値がstop価格以下なら、stop価格ではなく当日始値からstop slippageを引いた価格で決済する。
4. **[確定] TIME**: STOPしなければ固定holding horizon最終日の終値からexit slippageを引いて決済する。
5. **[確定] Fee**: entry/exit両側の既存手数料を控除する。
6. **[確定]** target_percentを自動利確価格または出口として使用しない。
7. **[確定]** 売却代金は翌営業日の寄付き注文から利用可能にする。

ラベルは次で固定する。

```text
realized_net_return_percent =
    (actual_exit_value - actual_entry_value
     - entry_commission - exit_commission)
    / actual_entry_value * 100
```

`actual_exit_value`はSTOP、GAP STOP、TIMEのうち共通約定関数が実際に選んだものとする。ラベル、評価器、paper tradeは同じversioned execution関数とcost manifestを呼ぶ。

### 8.2 予測対象候補

| 対象 | 長所 | 短所 | v3での扱い |
|---|---|---|---|
| 実現純収益率の回帰 | 金銭的意味と横比較が直接一致 | outlier、heteroskedasticity | **第一候補** |
| 下方リスク控除後の純収益率 | 安定性を目的へ入れられる | risk penalty定義が新しい調整項目になる | 最大3モデル案の候補 |
| 同日候補内の収益順位 | top-1目的に近い | 全候補が負でも1位、絶対収益gateが別途必要 | 案Cでのみ検討 |
| 一定純収益以上の確率 | no-trade閾値が分かりやすい | threshold依存、payoff幅を失う | 第一候補にしない |
| 利益/中立/損失の多クラス | 非対称損失を表せる | class境界が追加探索になる | 第一候補にしない |

**[提案] 第一候補**は実現純収益率回帰。STOP/TIME結果のclass分解や仮想target収益を経由せず、実際に実現する%を直接予測するためである。

## 9. 学習方法

- **[提案]** 全eligible銘柄×signal dateの行を結合し、1つのLightGBM regressorを学習する。
- **[提案]** 各signal dateの総weightを一定にし、銘柄数が多い日だけが過度に重くならないようにする。銘柄・業種別weight上限もtraining fold内で固定する。
- **[提案]** primary lossは実装前に`L2`または`Huber`の一方へ固定する。両方を結果で選ばない。推奨はoutlier耐性のあるHuberだが、loss parameterを増やすためユーザー決定事項とする。
- **[確定]** 特徴量順、モデル設定、seed、学習cutoff、ラベルcutoff、universe snapshot hashをmodel manifestへ保存する。
- **[確定]** validation/reference/shadow予測を学習へ戻さない。
- **[提案]** research時はouter foldごとに再学習し、shadow用release candidateは固定モデルpackageを一度だけ生成する。shadow中は再学習しない。
- **[未決]** release candidate最終fitにREFERENCE ONLY期間を「選択には使わずtraining contextとして」含めるか。公平性重視なら2025-03-31 cutoffのまま、鮮度重視なら全設計固定後に最新label-confirmed日まで一度だけrefitする。後者ではREFERENCE成績との同一モデル比較は成立しない。

## 10. 時系列分割とリーク防止

### 10.1 期間の役割

- **[確定] Research/selection**: 2025-03-31までにラベルが確定したデータだけを、モデル・特徴量群・設定・no-trade条件の選択に使う。
- **[提案] Outer validation**: research内で5個のexpanding walk-forward foldを作り、各foldは約6か月の連続期間とする。利用可能な履歴長に応じて最初のtrainingを最低3年確保する。
- **[確定] REFERENCE ONLY**: 2025-04-01〜2026-07-31。設計・採用条件・次の改善へ使用せず、一回だけ診断する。完全な未観測テストと表現しない。
- **[確定] 完全未来shadow**: v3モデル、ルール、評価器、release candidateを固定してcommit/tag/pushした後の、次の日本株営業日から開始する。開始日前をバックフィルしない。

具体的なouter fold日付は、point-in-time universeと全データの利用可能開始日を監査後、結果を見る前にmanifestへ固定する。

### 10.2 行単位の時点管理

- **[確定]** feature timestampはsignal dateの市場終了時。利用できる価格はその日の確定OHLCVまで。
- **[確定]** label confirmed timestampは実際のSTOP/GAP STOP/TIME exitが確定した市場終了時。
- **[確定]** 各予測のtraining rowは`feature_date < signal_date`かつ`label_confirmed_date < signal_date`に限定する。
- **[提案]** holding intervalがvalidation開始と重なるtraining rowをpurgeする。最低embargoは最大holding horizon取引日数とし、inner/outer foldの双方へ適用する。
- **[提案]** 横断面特徴量とranking metricはsignal dateをgroupとして分割し、同日の別銘柄をtrainとvalidationへ分けない。
- **[提案]** universe membershipは前月末までに確定した情報だけで翌月分を決める。日次overlayも当日注文前に判明した情報だけを使う。
- **[提案]** 企業情報は公表日時の次の取引日から利用可能とし、決算期ではなく実公表日時でjoinする。
- **[提案]** corporate actionはeffective dateと公表日を保持し、as-of再構築をテストする。
- **[提案]** research内のwalk-forward再学習は各outer fold開始時のみ。paper shadowは固定モデルとし、再学習頻度を0にする。

## 11. ポートフォリオ判断

- **[確定]** 初期資金30万円、100株単位、最大同時保有1銘柄。
- **[提案]** signal dateごとにeligible候補を`predicted_realized_net_return_percent`降順、同値はコード昇順で固定ソートする。
- **[提案]** 予測値が0%以下なら`NO_TRADE_NON_POSITIVE_EXPECTATION`とする。0%は費用控除後ラベルの経済的に固定された境界で、validationから調整しない。
- **[確定]** entry dayの注文は前日までに確定したavailable cashだけを使う。
- **[確定]** 寄付き価格不正、100株購入不能、position上限、データ不足は理由を保存して次候補へ進む。
- **[確定]** 当日STOP/TIMEの売却代金を同日寄付き注文へ使わない。
- **[確定]** cash、pending cash、locked capital、position、注文順位、skipを日次ledgerへ残す。

## 12. 評価方法

### 12.1 ポートフォリオ指標

- 全体実現純利益、最大DD、月次勝率、取引数。
- 銘柄別・業種別利益、最大銘柄利益依存率、最大業種利益依存率。
- 年別・outer fold別成績。
- STOP、GAP STOP、TIME件数と損益。
- costを0.5倍、1倍、1.5倍、2倍にした感度。ただし採用scoreは事前固定した1倍のみ。
- 欠測日、現金負値、資金二重使用、重複注文、同時保有違反、時間逆転。

### 12.2 予測・ranking指標

- 日次Spearman ICとそのfold別分布。
- Kendall rank correlation、NDCG@1/@5、top-1/top-5平均実現純収益率。
- 予測decile別平均実現純収益率と単調性。
- top候補が0%超だった割合、取引しなかった割合。
- ex-post bestとの差（regret）は診断のみとし、未来選択には使わない。

### 12.3 比較baseline

1. 何も取引しない（利益0、DD0）。
2. eligible候補からランダムに1銘柄選択。
3. 単純な直近20日相対return順。
4. 既存LOOP-000。ただし8銘柄限定であり、full-universe比較とは分け、共通8銘柄sliceでのみ比較する。
5. 市場指数buy-and-hold、およびeligible universe equal-weight指数。30万円/100株制約と異なるため参考benchmarkと明記する。

ランダムbaselineは**[提案]** 固定seed一覧100個で100回評価し、平均、中央値、5/25/75/95 percentileを保存する。v3は少なくともランダム95 percentileを上回ることを合格条件案とする。seed一覧は評価前にmanifestへ固定する。

## 13. 合格条件案

以下は**[提案]**であり、データを見ない段階でユーザー承認後に固定する。

1. 全outer fold合計の実現純利益が正で、5fold中4fold以上が正。
2. no-trade、単純momentumを上回り、固定100-seedランダムbaselineの95 percentileを上回る。
3. 最大DDが15.5%以下、月次勝率が60%以上。
4. 最大銘柄利益依存率20%以下、最大業種利益依存率35%以下。
5. outer validation合計200取引以上。
6. top-1平均実現純収益率が全eligible候補平均を上回る。
7. 日次Spearman ICのfold中央値が正で、5fold中4fold以上が非負。
8. 1.5倍costでも純利益が正。2倍costは診断として保存する。
9. 欠測理由未記録、未来参照、現金負値、資金二重使用、重複注文、同時保有違反が全て0。
10. 固定入力から決定的出力が2回一致する。

REFERENCE ONLY結果はこの合格条件へ入れない。REFERENCEで不具合を発見した場合はv3候補を無効化し、新versionを別研究として開始するが、REFERENCE期間を再び未知testとして使わない。

## 14. 実験予算と過学習防止

- **[提案] モデル案上限**: 3案。B共通回帰、C date-group ranking、残り1案は下方リスク目的または単純linear/regularized baselineのどちらかを事前選択する。
- **[提案] 特徴量群上限**: 上記G1〜G4の4群。追加・削除順を事前固定し、任意subsetの総当たりをしない。
- **[提案] ハイパーパラメータ探索**: 全モデル合計24 trialまで。nested inner foldだけで実施し、outer foldやREFERENCEをscoreへ使わない。
- **[提案] 論理実験上限**: 12回。データ不備修正は実験に数えないが、修正前結果は無効として記録する。
- **[提案] 同じ期間での失敗後再試行**: 同一仮説につき1回だけ。再試行は実装バグまたは事前仕様の曖昧さ修正に限定し、数式・基準変更は禁止する。
- **[確定]** REFERENCE結果を次の特徴量、モデル、universe、閾値、採用条件へ使用しない。
- **[確定]** 未来shadow中はモデル、特徴量、universe規則、cost、ranking、no-trade規則を変更しない。
- **[確定]** 採用条件は結果確認後に変更しない。
- **[提案]** 予算を使い切って合格候補がなければ`NO_V3_CANDIDATE`で停止し、同じ研究期間でLOOPを継続しない。

## 15. Phase別開発計画

| Phase | 入力 | 成果物 | 完了条件 | 停止条件 | 次へ進んではいけない条件 |
|---|---|---|---|---|---|
| Phase 0 データとuniverse監査 | raw OHLCV、security master、corporate action、calendar | immutable snapshot、point-in-time universe、coverage/survivorship監査 | 全hash一致、日付/重複/調整検証、delisted coverage明示 | point-in-time構成不能、license不明、欠測過多 | 現在銘柄を過去へ遡及、将来action混入、外部fallback |
| Phase 1 共通約定・ラベル固定 | Phase 0価格、既存execution仕様 | versioned execution spec、golden fixtures、label manifest | STOP/GAP STOP/TIME・費用がlabel/evaluatorで一致 | 日足で確定不能な規則、境界テスト失敗 | labelとportfolioで別ロジック、仮想target出口 |
| Phase 2 単純baseline | 固定universe、execution | no-trade、random分布、momentum、index基準 | seed・候補・出力schema固定、2回一致 | safety invariant違反 | baselineを見てuniverse/costを変更 |
| Phase 3 第一候補モデル | research期間のみ、G1〜G3、固定label | pooled regression実装、model manifest、unit tests | training cutoff、特徴量時点、ID非依存推論PASS | leakage、学習不足、非有限予測 | outer/reference結果を使った調整 |
| Phase 4 nested walk-forward | purged inner/outer folds、実験予算 | fold予測、portfolio ledger、ranking/calibration診断、候補判定 | 全テスト・安全条件・事前合格条件PASS | 予算超過、1条件でもFAIL | 失敗後の無制限再試行、REFERENCE閲覧 |
| Phase 5 一回限りREFERENCE | 完全固定candidate、2025-04-01〜2026-07-31 | `REFERENCE_ONLY` report | 1回実行、未来/選択逆流0、結果hash保存 | invariant/実装不具合 | 結果を使った修正・再採用・再評価 |
| Phase 6 完全未来shadow | commit/tag/push済みRC、開始後データ | 日次signal/rank/order/trade/ledger/missing log | 最低期間・取引数、安全条件を満たし正式レビュー | hash不一致、通信逸脱、資金違反、欠測未説明 | バックフィル、観察中変更、自動実注文 |

## 16. 完全未来shadowと自動運用安全性

- **[確定]** release candidateのcommit、tag、model hash、rules hash、feature schema、data cutoffをpush後に固定する。
- **[確定]** 最初のsignal dateは固定・push後の次の日本株営業日。過去signal、取引、利益をバックフィルしない。
- **[提案]** 正式判断は最低6か月かつ50件の決済済み取引の両方を満たした後とする。
- **[確定]** 30万円、100株、最大1position、売却代金翌営業日解放を維持する。
- **[確定]** ペーパートレードのみ。実注文API、証券認証情報、資金移動、売買指示通知を実装しない。
- **[確定]** 同一営業日の再実行はidempotent key（strategy/model/signal date/code）で重複を防ぐ。
- **[確定]** model、rules、snapshot、source dataの日次hashを検証し、不一致はfail-closedにする。
- **[提案]** network allowlistは承認済み市場データhost一つとGitHub Actionsの必要通信に分け、redirectと任意hostを拒否する。
- **[確定]** データ取得失敗、休場判定不能、部分欠測、内容改変は推測せず`missing_days`へ記録する。
- **[確定]** scheduleを有効化する操作はv3設計・実装とは別承認にする。

## 17. リスク一覧

| リスク | 影響 | 緩和策 |
|---|---|---|
| point-in-time universe不足 | survivorship bias | vendor master、delisted coverage、biased診断への格下げ |
| corporate actionの後日改訂 | future-adjustment leakage | raw/action保存、as-of再構築、snapshot不変化 |
| 日足STOP順序の限界 | gap以外のintraday path不明 | target出口を置かず、STOP判定規則を固定。必要なら別versionで細足検証 |
| 共通モデルの大型株偏重 | 小型株一般化低下 | date/sector/stock weight、属性別誤差、profit dependency |
| regime shift | 過去foldの再現性低下 | 年/fold/regime別評価、固定future shadow |
| 収益outlier | 回帰不安定 | lossを事前固定、winsorizationを行うならtraining fold内だけで固定 |
| universe数変動 | 横断面特徴量変動 | date-group計算、月次universe manifest |
| 価格上昇による購入不能 | 候補と約定の乖離 | signal時precheck、entry時skip、次順位処理 |
| データ欠測の非ランダム性 | 特定銘柄・時期bias | 欠測indicator、除外理由、coverage threshold |
| 実験多重性 | research過適合 | 12実験・24trial上限、REFERENCE逆流禁止 |
| stale model | future性能低下 | 最終refit方針を事前決定し、shadow中は固定 |
| baseline非同条件 | 誤った優劣 | full-universeと共通8銘柄sliceを分離表示 |

## 18. 未決事項（実装開始前にユーザー判断が必要）

1. **[未決] データ提供元**: point-in-time上場master、上場廃止、corporate action、業種、時価総額の履歴を取得できるsourceとlicense。
2. **[未決] 市場範囲**: Prime+Standardで開始するか、Growthも別stratumとして含めるか。
3. **[未決] universe閾値**: 60日中央値の売買代金1億円、出来高5万株、上場252日を承認するか。
4. **[未決] データ開始年**: 最低3年training+5foldを満たすため、2015年または取得可能最古年から開始するか。
5. **[未決] holding horizonとstop値**: 既存値をそのままv3全universeへ固定するか。ここを探索する場合は別の限定予算が必要。
6. **[未決] primary regression loss**: Huberを固定採用するか、説明性の高いL2にするか。
7. **[未決] G4**: point-in-time業種・sizeだけを使うか、銘柄ID/embeddingを1回だけ比較するか。
8. **[未決] 最終refit cutoff**: 2025-03-31のresearch-only modelをshadowへ出すか、全設計固定後にREFERENCE期間をperformance評価なしのtraining contextとして含めた別modelを作るか。
9. **[未決] 合格条件**: 5fold中4fold、DD15.5%、月次60%、銘柄20%、業種35%、200取引、random 95 percentileを承認するか。
10. **[未決] shadow最低観察量**: 6か月かつ50決済を承認するか。
11. **[未決] 許可市場データhostと実行基盤**: allowlist、credentialを持たない取得方法、手動からscheduleへ移る承認手順。

## 19. 実装開始ゲート

次をすべて満たすまでv3コードを実装しない。

1. 未決事項1〜11をユーザーが決定する。
2. Phase 0で必要なpoint-in-timeデータの取得権限と保存方針を承認する。
3. universe規則、execution version、feature群、モデル案、実験予算、合格条件を結果を見る前にcommitする。
4. REFERENCE ONLYのアクセス制御と完全未来shadow開始手順をテスト仕様として固定する。
5. v1/v2/shadow/referenceの既存branch・tagを変更しない隔離repo/worktree方針を決める。

本書の承認だけでは、データ取得、モデル学習、バックテスト、reference実行、shadow schedule有効化、実注文を許可しない。
