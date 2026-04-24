# stock-analyzer アーキテクチャ

LightGBM による日本株の買いシグナル予測と Slack 通知を、GitHub Actions 上で完全自動運用するプロジェクト。
ローカルではバックテスト（walk-forward グリッドサーチ）と週次レポートを実行する。

---

## 全体像

```mermaid
flowchart LR
    subgraph Batch["① 日次バッチ（GitHub Actions, 平日 16:00 JST）"]
        direction TB
        Train[train.py]
        Predict[main.py]
        Train --> Predict
    end

    subgraph Local["② ローカル実行（手動）"]
        direction TB
        BT[backtest.py]
        WR[weekly_report.py]
    end

    YF[(Yahoo Finance<br/>yfinance)]
    Model[(models/<br/>stock_ai_model.pkl)]
    Log[(data/<br/>trade_log.csv)]
    Slack[/Slack Webhook/]
    Repo[(GitHub Repo<br/>コミット & push)]

    Train -->|学習用 OHLCV 取得| YF
    Train -->|モデル保存| Model
    Predict -->|最新 OHLCV 取得| YF
    Predict -->|モデル読込| Model
    Predict -->|シグナル追記 & 答合わせ| Log
    Predict -->|通知送信| Slack
    Predict -->|CSV を push| Repo

    BT -->|候補銘柄の履歴取得| YF
    WR -->|集計| Log
```

---

## ① 日次予測フロー（本番運用）

`.github/workflows/daily_ai_trade.yml` が平日 16:00 JST（UTC 7:00）に起動し、
`train.py` → `main.py` の順に実行される。成績 CSV はワークフロー末尾で自動コミットされる。

```mermaid
sequenceDiagram
    autonumber
    participant GA as GitHub Actions
    participant TR as train.py
    participant YF as Yahoo Finance
    participant MD as models/*.pkl
    participant MN as main.py (predict)
    participant LG as data/trade_log.csv
    participant SL as Slack

    GA->>TR: 1. 学習ジョブ起動
    TR->>YF: 2. 過去データ取得（2020-01-01〜当日）
    YF-->>TR: OHLCV
    TR->>TR: 3. 指標計算 + 正解ラベル生成
    TR->>MD: 4. LightGBM 学習モデルを保存

    GA->>MN: 5. 予測ジョブ起動
    MN->>YF: 6. 直近 365 日分を取得
    YF-->>MN: OHLCV
    MN->>MD: 7. モデル読込
    MN->>MN: 8. 指標計算 → 最新行で predict_proba
    MN->>LG: 9. PENDING 行の答え合わせ（buy/sell/profit 更新）
    alt 買い確率 >= threshold
        MN->>LG: 10a. 新規 BUY シグナル追記
        MN->>SL: 10b. 「AI買いシグナル」通知
    else 未満
        MN->>SL: 10c. 「様子見」通知
    end
    GA->>GA: 11. trade_log.csv を commit & push
```

**判定ロジック**（[src/predict.py:106](src/predict.py#L106)）

| 項目 | 値 | 出所 |
| --- | --- | --- |
| 対象銘柄 | 8306 三菱UFJ | [config.yaml:16-19](config.yaml#L16-L19) |
| 買い閾値 | 確率 ≥ 15% | `ai_params.threshold` |
| 保有期間 | 翌営業日始値で買い、2日後終値で売り | `ai_params.future_days` |
| 損切り目安 | エントリー -3% | `ai_params.stop_loss_percent` |

---

## ② 学習フロー（train.py）

```mermaid
flowchart TB
    A[config.yaml<br/>target_stock / feature_columns] --> B[YFinanceFetcher<br/>日次 OHLCV]
    B --> C[sanitize_ohlcv<br/>数値化]
    C --> D[calculate_indicators<br/>SMA/RSI/MACD/BB/ATR/ADX<br/>→ Rate 変換]
    D --> E[create_target_variable<br/>翌日始値買い → N日後終値売り<br/>≥ target_percent を Target=1]
    E --> F[dropna<br/>特徴量 11 列 + Target]
    F --> G[LightGBM<br/>LGBMClassifier.fit]
    G --> H[(models/<br/>stock_ai_model.pkl)]
```

**特徴量 11 個**（[config.yaml:29-40](config.yaml#L29-L40)）
`SMA_5_Rate / SMA_25_Rate / RSI_14 / MACD_Rate / BB_Position / ATR_Rate / ADX_14 / Change_Rate_{1,3,5} / Volume_Change_1`

---

## ③ バックテストフロー（backtest.py）

パラメータを **val で選定 → test で評価** する walk-forward 方式で、
候補銘柄×(target_percent × stop_loss × threshold)=75 通りをグリッドサーチする。

```mermaid
flowchart LR
    subgraph PerStock["銘柄ごとに繰り返し"]
        direction TB
        S1[OHLCV 取得] --> S2[指標計算]
        S2 --> S3[target_percent ごとに<br/>ラベル付けデータ生成]
        S3 --> Fold
        subgraph Fold["各 fold (3通り)"]
            direction TB
            F1[train で LGBM 学習]
            F2[val で stop×thr 最良組を選定]
            F3[test でその組を評価<br/>損切りは日次 Low で再現]
            F1 --> F2 --> F3
        end
        Fold --> S4[fold 平均・最悪 fold・勝率を集計]
    end

    C[(config.yaml<br/>backtest_candidates)] --> PerStock
    PerStock --> R[推奨設定レポート<br/>tp / stop / thr / 合計利益]
```

**fold 分割**（[backtest.py:22-27](backtest.py#L22-L27)）
`(0-60%, 60-70%, 70-80%)` / `(0-70%, 70-80%, 80-90%)` / `(0-80%, 80-90%, 90-100%)`

---

## ④ 週次レポート（weekly_report.py）

```mermaid
flowchart LR
    L[(data/trade_log.csv)] --> P[status=DONE で絞り込み]
    P --> W[直近7日統計]
    P --> T[通算統計]
    W --> O[コンソール出力]
    T --> O
    T --> J{月末判定基準<br/>勝率≥45% / 利益≥-5,000円 / ≥15件}
    J -->|OK| OK[継続条件クリア]
    J -->|NG| NG[要注意]
```

---

## モジュール責務

| レイヤ | ファイル | 責務 |
| --- | --- | --- |
| エントリ | [main.py](main.py) | 日次予測のエントリ。`run_prediction` を呼ぶだけ |
| エントリ | [train.py](train.py) | フル学習モードの実行。モデル pkl を出力 |
| エントリ | [backtest.py](backtest.py) | walk-forward グリッドサーチ |
| エントリ | [weekly_report.py](weekly_report.py) | trade_log の集計 |
| 予測 | [src/predict.py](src/predict.py) | 指標計算 → 予測 → tracker/notifier 呼び出し |
| データ取得 | [src/fetchers/base.py](src/fetchers/base.py) | `StockDataFetcher` Protocol |
| データ取得 | [src/fetchers/yfinance.py](src/fetchers/yfinance.py) | yfinance 実装 |
| データ取得 | [src/fetchers/jquants.py](src/fetchers/jquants.py) | J-Quants 実装（予備） |
| 分析 | [src/analysis.py](src/analysis.py) | テクニカル指標計算 + 正解ラベル生成 |
| 記録 | [src/tracker.py](src/tracker.py) | trade_log.csv の追記・答え合わせ・集計メッセージ |
| 通知 | [src/notifier.py](src/notifier.py) | Slack Incoming Webhook 送信 |
| 設定 | [src/config.py](src/config.py) | config.yaml / .env 読込、`AppConfig` & ロガー |
| ルール判定 | [src/signal.py](src/signal.py) | ルールベース判定（現在は未使用、ハイブリッド用に保持） |

---

## 外部依存

| カテゴリ | 依存先 | 用途 |
| --- | --- | --- |
| データソース | Yahoo Finance（yfinance） | 日次 OHLCV |
| 機械学習 | LightGBM | 二値分類（買う/買わない） |
| 特徴量 | pandas-ta | SMA/RSI/MACD/BBands/ATR/ADX |
| 通知 | Slack Incoming Webhook | 環境変数 `SLACK_WEBHOOK_URL` |
| 実行環境 | GitHub Actions | 日次スケジュール実行（平日 16:00 JST） |
| ストレージ | Git リポジトリ | `data/trade_log.csv` を自動コミット |
