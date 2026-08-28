# stock-analyzer v3 Phase 0 データソース監査

## 0. 文書の位置付け

- **監査日:** 2026-08-03（JST）
- **設計基準:** `v3-design` / `9e3d30bdb8d946fdf70e4108605a632f42de4400`
- **監査範囲:** 公式の仕様、料金、利用条件ページの閲覧だけ。データ本体、API、ログイン、登録、購入、モデル学習、バックテストは実行していない。
- **判定:** **DATA_SOURCE_DECISION_REQUIRED**
- **理由:** JPX公式データを組み合わせればpoint-in-time universeを高い精度で構成できる見込みはある。しかし、個人向けJ-Quants APIだけでは安定企業ID、全企業行動、上場・廃止属性変更、取引停止の完全な履歴を公式仕様から保証できない。最も完全性が高いJ-Quants Proは法人向けで、契約資格、月額費用、保存・GitHub利用許諾のユーザー判断とJPXIへの確認が必要である。

本書の「推奨」はデータ品質と再現性だけによる提案であり、バックテスト結果に基づく選択ではない。現在上場中の銘柄一覧を過去へ遡及する方法は正式評価に採用しない。

## 1. 判定語

- `FULL_POINT_IN_TIME`: 上場廃止、市場区分変更、証券種別、企業行動を含め、各時点のuniverseを再現できる。
- `PARTIAL_POINT_IN_TIME`: 一部を再現できるが補助データ、契約範囲の確認、または明示的仮定が必要。
- `CURRENT_ONLY`: 現在銘柄中心で、正式な過去評価には不適切。
- `UNKNOWN`: 公開された公式仕様だけでは必要範囲を確定できない。

## 2. 必要データと監査結果

| 必要項目 | universeでの用途 | J-Quants API（個人） | J-Quants Pro/JPX有償 | 現時点の不足・注意 |
|---|---|---|---|---|
| 日次OHLCV | 特徴量、約定、流動性 | あり | あり、2008-05-07以降 | 無取引日のnullと欠測を区別する必要 |
| adjusted OHLCV | 連続収益・特徴量 | split等の調整値あり | 調整前後と係数あり | Pro仕様でも調整は遡及更新。as-of再構築にはrawと企業行動を保存する |
| 分割・併合・配当等 | 価格・出来高・株数調整 | split/配当はあるが全イベント履歴は不足 | Corporate Action Dataあり。配当2013-02-20、その他2015-05-08以降 | 外国証券等は企業行動coverage外。第一版は普通株に限定 |
| 銘柄コード | join、注文単位 | あり | あり | コードは企業の永続IDではない |
| 安定企業識別子 | コード変更・再上場の連結 | 公開仕様で保証なし | 変更前後コード等はあるが永続issuer IDは要確認 | 内部IDを変更履歴から生成する場合も合併・承継ルールの監査が必要 |
| 上場日・上場廃止日 | 252日条件、survivorship排除 | 過去masterはあるが専用履歴の完全性不明 | 上場銘柄情報、個別銘柄属性変更、上場廃止情報あり | Proの属性変更・廃止はSFTP/Snowflake中心。契約対象を確認 |
| 市場区分変更履歴 | Prime/Standard PIT条件 | 過去日masterで相当程度可能 | Listed Issue Informationと属性変更で可能 | 2022再編前は旧市場区分。Prime/Standardへ過去を遡及しない |
| Prime/Standard所属履歴 | 月次universe | 過去日masterで相当程度可能 | 日次履歴2008年以降 | 2022-04-04以前は東証一部/二部等として別regimeを保持 |
| 証券種別 | 普通株とETF等の分離 | masterで一部判別可能 | ETP/REITを含むmaster、参照データ | 優先株、外国株、JDR等の網羅的type codeを契約前に確認 |
| 取引停止・売買不能 | 日次overlay | OHLC欠損だけでは停止理由を識別不能 | reference/corporate-action補助が必要 | 公式仕様上、完全な日次停止履歴は未確認。価格欠損はfail-closed可能 |
| 業種・変更履歴 | sector特徴量、偏り評価 | 過去日masterの17/33業種 | masterと属性変更に旧新業種あり | 公表・適用時刻を保持し、effective date前に使わない |
| 発行済株式数 | 時価総額 | 財務開示時点値は利用可能 | Listed Shares Flash/Factors、財務データ | 後日訂正・更新がある。availability timestamp付きsnapshotが必要 |
| 時価総額・区分 | size特徴量 | OHLC×利用可能株数で推計 | 個別株時価総額、株数データあり | point-in-time株数が確保できない契約では使用不可 |
| 出来高 | 流動性閾値 | あり | あり | split時に価格と同じ方針で整合調整 |
| 売買代金 | 1億円条件 | 日次データの項目確認が必要 | Stock Pricesにtrading valueあり | なければraw OHLCVから近似せず、公式turnover valueを優先 |
| 呼値・売買単位 | 100株購入可否、丸め | masterの取引単位は要plan確認 | Listed Issue Informationにtrading unit | 呼値テーブル変更履歴は別資料が必要。第一段階は公式約定価格を使用 |
| 100株購入可能性 | 30万円制約 | 終値×単元＋費用で事前判定 | 同左 | 翌日始値では再判定し資金不足を記録する |
| 利用可能日時 | 情報リーク防止 | 当日/翌営業日データの配信仕様あり | datasetごとに更新時刻・reporting lagあり | 過去の訂正前versionをAPIが返す保証はない。取得時snapshotを不変保存する |

## 3. 候補データソース

### 3.1 JPX/JPMR J-Quants API（個人向け）

- **提供者・データ名:** JPX Market Innovation & Research, Inc.、J-Quants API。
- **取得方法:** REST API v2、API key必須。登録・API発行は今回未実施。
- **料金・履歴:** Free 0円（2年、12週遅延）、Light 1,650円/月（5年）、Standard 3,300円/月（10年）、Premium 16,500円/月（最大20年）。公式料金ページの2026-08-03閲覧時点。
- **収録:** Listed Issue Master、日次OHLCV、出来高、財務。Premiumは配当、財務明細等を追加。過去、当日、翌営業日時点の銘柄情報を取得可能と公式説明にある。
- **上場廃止・変更履歴:** 過去日masterにより在籍状態と市場/業種を再現できる可能性は高いが、専用の上場廃止・コード変更・全企業行動・停止履歴を個人APIだけで完全に取得できるとは公開仕様から断定できない。
- **調整:** split等の調整値・係数を提供するが、adjusted seriesは後日の企業行動で遡及更新され得る。研究snapshot作成時の値は固定できるが、当時見えていたadjusted historyそのものの再現にはraw/action履歴が必要。
- **時点再現:** 配信日/開示時刻を持つ財務項目はas-of join可能。後日訂正前のAPI応答を恒久的に再取得できる保証はない。
- **API制限:** plan別。Freeは5 calls/minと公式料金ページに表示。全planの詳細上限は契約前に再確認する。
- **保存・再利用:** 個人的分析は可能だが、取得データを閲覧可能な形で第三者へ配布・共有することは禁止と公式FAQに明記。raw CSVをpublic/privateを問わずGitHubへ保存することは、第三者アクセスが成立し得るため、書面確認なしでは不可とする。hash/manifestのみをGit管理し、rawは暗号化されたアクセス制御storageへ置く案が必要。
- **用途制約:** 個人の私的利用限定。法人利用や分析サービス提供は不可。
- **停止時再現性:** 契約中にimmutable raw snapshotと利用条件の証跡を保存できれば評価の再現性は保てるが、再配布・別環境復元はlicense制約を受ける。
- **判定:** **PARTIAL_POINT_IN_TIME**。
- **正式評価:** JPXIが欠ける属性のcoverageと保存方法を確認し、JPX参照データで補完できる場合に限り候補。
- **既知の欠点:** 個人利用限定、履歴長がplan依存、完全な企業行動・停止・永続IDが不足、adjusted値の遡及改訂、GitHub保存不可の可能性。

### 3.2 J-Quants Pro（法人向け）＋JPX参照データ

- **提供者・データ名:** JPXI、Listed Issue Information、Stock Prices (OHLC)、Corporate Action Data、Listed Shares Flash Data/Corporate Action Factors、Financial Summary/Statements。必要に応じJPX Reference Data/Master File。
- **取得方法:** REST API、SFTP、Snowflake。Corporate Action Dataと一部属性変更/廃止はSFTP/Snowflakeのみ。
- **料金・登録:** 法人契約と審査が必要。公開料金の単一法人内部利用は、Listed Issue Information 50,000円/月、Stock Prices 150,000円/月、Corporate Action Data 300,000円/月、Financial Statements 150,000円/月（各税別）。必要最小bundleは少なくとも50万円/月、point-in-time株数まで含めると最大65万円/月程度を見込む。実契約額・最低利用期間・初期費は見積確認が必要。
- **履歴:** Listed Issue InformationとOHLCは2008-05-07以降。Corporate Actionは配当2013-02-20、その他2015-05-08以降。財務summaryは2008-07-07以降。Listed Shares系は2008-05-07以降。
- **coverage:** 全TSE上場株、ETP、REITを含む。masterは会社名、コード、市場、17/33業種、取引単位等を日次履歴で提供。属性変更には発表日、適用日、変更前後コード、市場、業種、売買単位がある。delisting datasetもある。
- **企業行動:** split、reverse split、配当、社名/証券master変更等。外国会社・外国商品は一部対象外であり、普通株限定案とは整合する。
- **株数・時価総額:** Listed Shares Flash/Factorsと財務データを利用可能。ただし報告・開示から更新まで1〜2営業日または月次更新となるイベントがあり、availability timestamp基準で使用する。
- **OHLCV:** 調整前後、adjustment factor、volume、trading value。調整値は企業行動に応じて遡及調整される。raw＋actionからas-of seriesを再構築する必要がある。
- **2022市場再編:** 日次market codeを保持し、2022-04-04より前は旧区分、以後はPrime/Standard/Growthとして扱える。旧東証一部をPrimeへ遡及変換しない。
- **利用可能時点:** masterは17:30に翌営業日分、OHLCは16:30、財務はnear-real-time等、dataset別に明示。取得時刻とannouncement/effective dateを保存する。
- **保存・GitHub:** 内部利用は契約範囲内。第三者配布、公表、GitHub保存には別途External Distribution契約/承認が必要。公開料金表でdatasetごとに許可区分が異なるため、raw dataをGitへ入れない設計を既定とする。
- **サービス停止時:** 契約上許されるimmutable snapshotを保管すれば強い再現性を持つが、契約終了後の保有・利用可否を契約書で確認する必要がある。
- **判定:** **PARTIAL_POINT_IN_TIME（契約・項目確認後はFULL_POINT_IN_TIME候補）**。停止履歴、永続issuer ID、証券種別の完全なcode set、契約終了後保有が未確認なので現段階ではFULLと断定しない。
- **正式評価:** 最有力。上記4点をsample schemaと契約書で確認できれば2016年以降の正式評価に使用可能。
- **既知の欠点:** 法人向け、高額、最低利用期間、SFTP/Snowflake必須項目、再配布制限、個人が契約できない可能性。

### 3.3 JPX Historical Data / J-Quants DataCube / 公開一覧

- **提供者・取得方法:** JPX/JPXI。過去データファイルを案件ごとに購入、またはJPX公開Webページを閲覧。
- **料金・登録:** DataCubeは個別見積・購入。公開の上場廃止一覧等は無料。
- **収録:** 株価、月間相場、個別株時価総額等のhistorical files。Reference Dataには国内取引所の銘柄masterと企業行動サービスがある。
- **上場廃止:** JPX公開一覧は日付、銘柄、コード、市場、理由を含むが、11年より前に上場廃止した会社はWeb一覧で利用できない。
- **時点再現:** 複数の公式資料を組み合わせれば補完できるが、毎月universe、停止、コード連結、業種履歴を一つの公開schemaで再現できない。
- **保存・再利用:** DataCubeは利用目的・再配布条件を見積時に確認。GitHub保存は明示許可なしでは不可。
- **判定:** **PARTIAL_POINT_IN_TIME**。
- **正式評価:** J-Quants API/Proの欠損補助には使える。公開Webだけを単独の正式sourceにはしない。
- **既知の欠点:** ファイル単位・見積制、更新処理が分散、公開delistingの保持年限、availability timestampの統一困難。

### 3.4 Yahoo Finance / yfinance（現行リポジトリの取得経路）

- **提供者・取得方法:** Yahoo Financeを非公式Python library `yfinance`経由で取得。
- **料金・登録:** 通常無料、登録不要の範囲があるが、安定した正式API契約ではない。
- **収録:** 日次OHLCV、adjusted価格、split/dividendの一部。
- **不足:** 過去時点の全上場銘柄master、上場廃止、旧市場区分、証券種別、停止、業種変更、安定ID、正確なavailability timestampを公式に保証しない。
- **調整定義・訂正:** 提供値の遡及改訂やlibrary仕様変更を長期に固定できない。公式取引所sourceとの一致保証がない。
- **API/保存/利用条件:** rate/availabilityに保証がなく、データ再配布・GitHub保存はYahooの利用条件確認が必要。
- **判定:** universeは **CURRENT_ONLY**、価格は **PARTIAL_POINT_IN_TIME**。
- **正式評価:** 不可。価格の探索的cross-checkに限定し、v3正式snapshotへfallbackしない。

### 3.5 日経NEEDS / NEEDS SPOT

- **提供者・取得方法:** 日本経済新聞社。NEEDS bulk/FinancialQUEST、または一回購入のNEEDS SPOT（CSV/Excel/Access等）。
- **料金・登録:** 有料、見積・契約が必要。公開定価は監査時に確認できない。
- **収録:** 長期の企業・財務・株式・債券データ。NEEDS SPOTは全上場会社の日次データ等のhistorical order実績を明記。
- **PIT項目:** 上場廃止、企業行動、旧コード、市場・業種履歴、availability timestampの粒度は公開製品ページだけでは確定できない。
- **保存・GitHub:** 契約次第。rawのGitHub保存は許諾確認前は禁止。
- **判定:** **UNKNOWN（有力な有料代替）**。
- **正式評価:** sample data dictionary、delisted coverage、as-of/revision policy、利用許諾を営業窓口から文書取得後に判断。
- **既知の欠点:** 費用・schema・PIT性・配布権が公開情報だけで判断不能。

### 3.6 LSEG Workspace/Data Platform、FactSet、QUICK等

- **提供者・取得方法:** 各商用vendorのterminal/API/data feed。契約・認証必須。
- **収録:** LSEGは公式developer資料でcorporate actions、identifier change、historical constituentsを扱う。FactSet/QUICKも広範なsecurity master/price/fundamentalを提供するとされるが、日本株の必要項目を満たす具体的契約schemaは公開ページだけでは確認できない。
- **料金・制限:** 見積。API量、保存期間、派生データ、再配布は契約別。
- **PIT性:** LSEGはpoint-in-time構成を扱える可能性が高いが、本監査対象の「全TSE ordinary sharesの日次master」のcoverageを公式公開仕様だけで確定できない。FactSet/QUICKも同様。
- **判定:** **UNKNOWN**。
- **正式評価:** vendor comparison用RFPに必須項目表を添え、sample extractとlicense回答を得た場合のみ候補。
- **既知の欠点:** 高額、vendor固有ID、ブラックボックスな訂正・調整、契約終了後の再現性、GitHub配布不可。

## 4. point-in-time再現性と2022年市場再編

| source | 判定 | 2022年前後 | 正式評価への条件 |
|---|---|---|---|
| J-Quants API Premium/Standard | PARTIAL_POINT_IN_TIME | 日付指定masterで旧/新区分を保持できる見込み | delisting、code/type変更、停止、保存権をJPXIに確認し補助sourceを固定 |
| J-Quants Pro bundle | PARTIAL_POINT_IN_TIME（FULL候補） | 2008年以降の日次market code＋属性変更で再現可能 | 契約schemaで停止、永続ID、全type、訂正履歴を確認 |
| JPX DataCube/公開資料 | PARTIAL_POINT_IN_TIME | 市場再編資料と個別fileで補完可能 | 欠損・availability・ライセンスをmanifest化 |
| Yahoo/yfinance | CURRENT_ONLY | 正式な日次security masterがない | 正式評価には使用しない |
| NEEDS | UNKNOWN | 提供可能性はある | sample/schema/license確認 |
| LSEG/FactSet/QUICK | UNKNOWN | 提供可能性はある | 日本株coverageの文書回答とsample確認 |

2022-04-04の市場再編は単純な名称変更として扱わない。再編前の各月は当時の市場区分を保存する。v3の「Prime/Standard普通株」という規則を2016年まで遡及適用すると未来の新区分を使うため、次のいずれかを実装前に固定する必要がある。

1. **推奨:** 2022-03までを旧東証一部・二部のpoint-in-time条件、2022-04以降をPrime・Standard条件とし、対応表を規則として事前固定する。
2. 2022-04から研究を開始する。ただし履歴とfold数が大幅に不足する。
3. 公式の再編移行判定日情報を使用する。判定公表前の過去日へ新市場区分を戻さない。

## 5. 無料で実現できる範囲と選択肢

無料で得られるのは、短い遅延OHLC、限定的master、JPX公開の上場・廃止資料である。2016/2018開始の全ordinary sharesについて、上場廃止、属性変更、全企業行動、株数、availability timestampを一貫したschemaで保存することは無料sourceだけでは確認できない。したがって無料だけで`FULL_POINT_IN_TIME`正式評価を構築できるとは判定しない。

| 案 | 費用 | 再現性 | バイアス | 実装難易度 | 研究信頼性 | 未来shadow |
|---|---:|---|---|---|---|---|
| A. 有料データ契約 | J-Quants API 3,300〜16,500円/月、Pro bundle概算50〜65万円/月税別、他vendor見積 | 最も高い | 最小化可能 | 中〜高 | 正式評価候補 | 契約継続・日次配信があれば可 |
| B. 取得可能期間から開始 | 個人API料金程度 | 期間内は中〜高 | 期間短縮によるregime bias | 中 | PIT coverage確認後なら可 | 可。ただし学習量不足を明示 |
| C. 公式に履歴再現できる限定集合 | 中 | 高 | universe縮小・代表性bias | 中 | 限定scopeの正式評価 | 同じscopeなら可 |
| D. survivorship bias明示の参考評価 | 無料〜低 | 低 | 明確に残る | 低 | `UNIVERSE_BIASED_DIAGNOSTIC`のみ | RC採用・shadow開始に使用不可 |
| E. NO_DATAで停止 | 0 | 評価しない | 誤評価なし | 低 | 最も誠実な停止 | 不可 |

**推奨順:** A（契約可能ならPro/公式bundle）→ B（個人APIで公式に保証できる最古期間）→ C。Dを正式評価へ昇格させない。A〜Cのいずれも利用許諾とcoverageを確定できなければEで停止する。

## 6. データ開始年の比較

| 開始案 | データ量・regime | 市場再編 | 企業行動・廃止coverage | 費用・欠測 | 判断 |
|---|---|---|---|---|---|
| 2016 | 約9年のresearch、COVID等を含む | 旧市場→新市場を跨ぐ | Proの「その他企業行動」開始後をほぼ包含 | Proなら強い。個人Standard 10年は境界が近い | **品質上の推奨** |
| 2018 | 約7年、複数regime | 跨ぐ | 個人API planでも扱いやすいが属性変更の完全性は未保証 | Standard範囲内 | 個人向け現実案 |
| 2020 | 約5年 | 旧市場期間が短い | 上場廃止・actionは比較的揃う | Light/Standardで低コスト | foldとregime多様性が不足 |
| 取得可能最古 | 最大 | 複数制度変更 | datasetごとに開始日が違い、欠測・定義変更が増える | 高い清掃・契約コスト | 「最古」を目的にせず共通coverage開始を使う |

**推奨開始年は2016年。** 理由は、J-Quants ProのOHLC/masterは2008年から、配当以外の企業行動は2015-05-08からであり、2016年は主要datasetの共通完全年として監査しやすいからである。2015年以前を混ぜて行数を増やすより、2016年からの一貫したdefinitionを優先する。個人向けsourceしか採用できない場合は2018年へ変更するのではなく、契約時点で取得可能な最古日と必要項目coverageを再監査し、開始日を結果を見る前に固定する。

## 7. 暫定universe条件とデータ監査上の推奨

| 項目 | 暫定推奨 | 根拠・留保 |
|---|---|---|
| 市場 | Prime/Standard。2022以前は事前固定した旧一部/二部対応 | 新区分を過去へ遡及しない |
| Growth | 初期版から除外 | 上場期間、流動性、分布差を減らす。将来は別stratum |
| ETF等 | 普通株と分離 | 企業行動・業種・fundamentalの意味が異なる |
| 最低売買代金 | 60取引日中央値1億円/日 | V3設計案を維持。official trading valueを使う |
| 最低出来高 | 60取引日中央値5万株/日 | split整合後の値を使う |
| 最低上場期間 | 252取引日 | 上場日ではなく実取引日数も監査 |
| 100株購入可能 | signal日終値×100＋費用≤30万円、entry始値で再判定 | point-in-time売買単位を必須とする |
| 更新頻度 | 月次、日次overlay | masterは前月末as-of、停止・価格・資金は日次 |
| 時価総額特徴量 | PIT株数とavailabilityが契約で確定した場合のみ | 確定しなければG4から除外。現在株数の遡及は禁止 |
| 業種履歴 | Pro属性変更または日次masterが確保できれば使用 | effective date前に使わない |
| split等 | raw OHLCV＋公表/発効日付きactionからversioned調整 | vendor adjusted値だけを盲信しない |
| コード変更 | vendor変更前後code＋内部immutable security ID | 4桁コードを永続企業IDにしない |
| 開始年 | 2016 | source決定後にcoverage auditで最終承認 |

想定銘柄数は各月150〜400銘柄だが、これは設計上の容量見積であり、銘柄一覧を取得して算出した値ではない。実数を合わせるために流動性条件を変更しない。

## 8. データソース決定前後の未決事項

### 8.1 データソース決定前に固定できる

- holding horizon: STOP/GAP STOP/TIMEラベルの確定日とpurge幅を決めるため、データ取得前に固定可能。
- stop値: 約定・ラベル仕様としてデータ結果を見る前に固定可能。
- loss: 第一提案は**Huber**。実現純収益の外れ値耐性が理由で、データ結果によるL2との選択はしない。
- 銘柄ID/embedding: 第一提案では**使用しない**。共通尺度と新規銘柄一般化を優先する。
- 合格条件: 指標定義と閾値はデータ取得前に事前登録可能。
- shadow最低観察量: 6か月かつ50決済というV3設計案をデータ結果前に固定可能。

### 8.2 データソース決定後でなければ固定できない

- 最終fit cutoff: 取得可能最終日、availability、REFERENCE隔離方針が必要。
- 実行基盤: licenseが許す保管場所、SFTP/Snowflake/API、credential管理で決まる。
- 許可する市場データhost: 採用vendorの正式endpointとredirect/CDN仕様の確認後に固定する。
- 具体的walk-forward fold: 全datasetの共通開始日、欠測、ラベル確定可能日が必要。
- 時価総額・業種特徴量の採否: PIT履歴とavailability timestampが契約項目に含まれることを確認してから固定する。

### 8.3 今回ユーザーが決める必要がある項目

1. 法人としてJ-Quants Proを契約可能か。月額50〜65万円税別規模を調査・見積してよいか。
2. 個人向けJ-Quants APIを選ぶ場合、Standard/Premiumのどちらを許容するか。
3. raw dataをGitHubへ置かず、アクセス制御された外部storage＋Git管理hashにすることを承認するか。
4. JPXIへ、delisted coverage、停止履歴、永続ID、証券type、訂正履歴、契約終了後保有を問い合わせてよいか。
5. 2016開始と、2022以前の旧一部/二部対応規則を承認するか。
6. 暫定universe条件（Growth除外、ETF分離、流動性、252日、100株条件）を承認するか。
7. Pro不可の場合、B（取得可能期間短縮）、C（限定集合）、E（停止）のどれを選ぶか。Dは正式評価に使わない。
8. Huber固定、銘柄ID/embedding不使用をPhase 0後の第一モデル仕様として維持するか。

## 9. Phase 0開始ゲート

**現時点ではPhase 0実装へ進まない。** 次を満たせば`DATA_READY`へ再判定できる。

1. 採用sourceと契約planを一つに決める。
2. sample data dictionaryまたは契約仕様で、日次master、delisting、market/industry/type/trading-unit変更、企業行動、OHLCV/trading value、株数、availability timestampのcoverageを確認する。
3. 2016年以降の普通株について、上場廃止を含む月次universeを再構成可能であることを、データ本体取得前のacceptance checklistとして固定する。
4. 2022市場再編の旧/新区分規則を事前承認する。
5. raw保存期間、暗号化、アクセス権、契約終了後利用、GitHub非保存、hash公開可否を法的に確認する。
6. 欠ける項目と補助sourceを固定し、現在銘柄による代用や外部fallbackを禁止する。
7. 取得量、rate limit、再取得、訂正時versioning、サービス停止時archive方針を決める。

これらが満たせなければ、正式v3評価は`NOT_FEASIBLE/NO_DATA`として停止する。biased diagnosticを正式評価やfuture shadowの候補選択へ使用しない。

## 10. 閲覧した公式資料

確認日は全て2026-08-03。ページ上で更新日が明示されたものは併記する。

1. JPX, [J-Quants API](https://www.jpx.co.jp/english/markets/other-data-services/j-quants-api/index.html) — サービス概要、個人向け。
2. J-Quants, [公式トップ・料金・FAQ](https://jpx-jquants.com/) — plan、履歴長、API制限、私的利用・配布制限。
3. JPX, [Historical Data](https://www.jpx.co.jp/english/markets/paid-info-equities/historical/index.html) — DataCube、J-Quants価格、利用区分。
4. JPX, [J-Quants Pro](https://www.jpx.co.jp/english/markets/other-data-services/j-quants-pro/) — 法人向け提供形態。
5. J-Quants Pro, [Listed Issue Information](https://pro.jpx-jquants.com/datasets/4) — 2008-05-07以降、17:30、翌営業日情報。
6. J-Quants Pro, [Stock Prices (OHLC)](https://pro.jpx-jquants.com/datasets/9) — OHLCV、trading value、調整前後、係数。
7. J-Quants Pro, [Corporate Action Data](https://pro.jpx-jquants.com/datasets/14) — 配当2013-02-20、その他2015-05-08以降。
8. J-Quants Pro, [Listed Shares Flash Data](https://pro.jpx-jquants.com/datasets/16) / [Corporate Action Factors](https://pro.jpx-jquants.com/datasets/17) — 株数と更新時点。
9. J-Quants Pro, [Financial Summary/Statements](https://pro.jpx-jquants.com/datasets/5) — 開示データ、履歴、更新時刻。
10. J-Quants Pro, [Data history/specification](https://jpx.gitbook.io/j-quants-pro/data-spec) — dataset別開始時期。
11. J-Quants Pro, [Individual Stock Attribute Changes](https://jpx.gitbook.io/j-quants-pro-ja/api-reference/corporate_action/change_of_stock_detail) — 変更前後code、市場、業種、単位。
12. J-Quants Pro, [Delisting](https://jpx.gitbook.io/j-quants-pro/api-reference/corporate_action/delisting) — 公表日時と廃止日。
13. J-Quants Pro, [Pricing and Usage Table](https://pro.jpx-jquants.com/pdfs/appendix-1-2-pricing-and-usage-table-en.pdf) — 2026-01-20更新版、月額・配布区分。
14. J-Quants Pro, [Data License Agreement](https://pro.jpx-jquants.com/pdfs/j-quants-pro-data-license-agreement-en.pdf) / [External Distribution](https://pro.jpx-jquants.com/pdfs/appendix-4-external-distribution-en.pdf) — 内部利用、第三者配布。
15. JPX, [Reference Data](https://www.jpx.co.jp/english/markets/paid-info-equities/reference/index.html) — master/corporate action service。
16. JPX, [Delisted Companies](https://www.jpx.co.jp/english/listing/stocks/delisted/) — 公開廃止一覧、11年保持上限。
17. Nikkei, [NEEDS Data](https://needs.nikkei.co.jp/needs-data/) / [NEEDS SPOT](https://nkbb.nikkei.co.jp/en/service/needs-spot/) — historical dataの個別提供。
18. LSEG Developer Portal, [Corporate Actions Content Set](https://developers.lseg.com/en/article-catalog/article/workspace-corporate-actions-content-set-guide) / [Historical Constituents](https://developers.lseg.com/en/article-catalog/article/building-historical-index-constituents) — corporate action/identifier/PIT構成の機能例。

## 11. 最終結論

- **最終判定:** `DATA_SOURCE_DECISION_REQUIRED`
- **最有力:** 法人契約可能ならJ-Quants ProのListed Issue Information＋Stock Prices＋Corporate Action Data＋Listed Shares/Financial Data。個人利用ならJ-Quants API Standard/Premium＋JPX公式補助データだが、FULL_POINT_IN_TIMEを保証する追加確認が必須。
- **無料で実現可能か:** 正式なFULL_POINT_IN_TIME評価は不可と判断する。
- **point-in-time再現可能期間:** Pro bundleの公開仕様上、主要企業行動が揃う2016-01-01以降が第一候補。個人APIは契約planと不足属性確認後に確定する。
- **推奨開始年:** 2016年。
- **推奨universe:** PITの旧一部/二部→Prime/Standard普通株、Growth除外、ETF等分離、60日中央値売買代金1億円・出来高5万株、上場252取引日、30万円で100株購入可能、月次更新＋日次overlay。
- **想定規模:** 月150〜400銘柄（未取得データに基づかない容量見積）。
- **主な不足:** 永続issuer ID、完全な取引停止履歴、証券type codeの網羅性、個人APIのdelisting/属性変更coverage、訂正前version、保存/契約終了後利用権。

本監査はPhase 0の実装、データ取得、モデル学習、バックテストを許可しない。ユーザーがsource、費用、license、開始年、universe規則を決めた後に、データdictionaryと契約回答だけを用いた最終source acceptanceを別途行う。
