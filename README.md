# stock-analyzer

日本株の売買判断を補助するための、AI株価分析・通知ツールです。

Yahoo Finance から株価データを取得し、テクニカル指標を作り、LightGBM の機械学習モデルで「買いシグナル」の確率を出します。買い条件を満たした場合は、Slack に通知し、結果を `data/trade_log.csv` に記録します。

## できること

- 指定した日本株の過去株価を取得する
- SMA、RSI、MACD、ボリンジャーバンド、ATR、ADX などの指標を計算する
- AIモデルを学習し、買いシグナルの確率を予測する
- 買いシグナルが出たら Slack に通知する
- 売買シグナルの結果を CSV に記録する
- 過去データでバックテストし、銘柄やパラメータを比較する
- 週次レポートで勝率や損益を確認する

## 要約

このツールは「今日この銘柄を買うべき可能性がどれくらいあるか」を、過去の値動きから機械的にチェックするものです。

現在の設定では、主に次の流れで使います。

1. 過去の株価データを集める
2. AIモデルを学習する
3. 今日の株価データから買い確率を出す
4. 確率が設定した基準以上なら Slack に通知する
5. 後日、そのシグナルが利益になったかを記録する

記録は `data/trade_log.csv` に保存されます。週次レポートを実行すると、これまでの勝率、損益、保留中のシグナル数などを確認できます。

## 主なファイル

| ファイル | 役割 |
| --- | --- |
| `config.yaml` | 監視する銘柄、AIのしきい値、特徴量、バックテスト対象などの設定 |
| `train.py` | AIモデルを学習して `models/stock_ai_model.pkl` に保存 |
| `main.py` | 最新データを使って予測し、必要なら Slack 通知とログ記録 |
| `backtest.py` | 過去データで銘柄・パラメータを検証 |
| `weekly_report.py` | `data/trade_log.csv` から週次・累計成績を表示 |
| `src/analysis.py` | テクニカル指標と学習用ラベルを作成 |
| `src/predict.py` | 予測、通知、売買ログ更新の中心処理 |
| `src/tracker.py` | 売買シグナルと結果を CSV に記録 |
| `src/notifier.py` | Slack 通知 |
| `.github/workflows/daily_ai_trade.yml` | GitHub Actions で平日に自動実行する設定 |

## セットアップ

Python 3.12 を想定しています。

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install pandas lightgbm yfinance pandas-ta joblib scikit-learn pyyaml requests python-dotenv
```

Slack 通知を使う場合は、プロジェクトルートに `.env` を作り、Incoming Webhook URL を設定します。

```env
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

Slack 通知が不要な場合、`.env` がなくても予測処理自体は実行できます。

## 使い方

バックテストを実行します。

```bash
python backtest.py
```

AIモデルを学習します。

```bash
python train.py
```

最新データで予測し、条件を満たせば Slack に通知します。

```bash
python main.py
```

週次レポートを表示します。

```bash
python weekly_report.py
```

`weekly_report.py` は8306とログ専用銘柄をまとめて読み込み、銘柄ごとの勝率、通算損益、PENDING件数を損益順に表示します。

## ログ専用の複数銘柄運用

`config.yaml` の `log_only_stocks` に指定した銘柄は、8306と同じ条件で銘柄別にモデルを学習・予測します。これらの銘柄はSlack通知を行わず、仮想売買の結果だけを個別CSVへ保存します。

| 対象 | モデル | 売買ログ | Slack通知 |
| --- | --- | --- | --- |
| `target_stock`（8306） | `models/stock_ai_model.pkl` | `data/trade_log.csv` | あり |
| `log_only_stocks` | `models/monitoring/stock_ai_model_<code>.pkl` | `data/monitoring/trade_log_<code>.csv` | なし |

追加銘柄の `ai_params` を省略した場合は、`target_stock.ai_params` と同じ条件を引き継ぎます。銘柄ごとに条件を変えたい場合だけ、その銘柄の下へ `ai_params` を追加してください。

モデルファイルは実行のたびに全銘柄分を再学習して一時生成し、Gitには保存しません。学習に失敗した場合は古いモデルへフォールバックせず、その日の予測処理を停止します。

## 設定の見方

主な設定は `config.yaml` にあります。

| 設定 | 意味 |
| --- | --- |
| `target_stock.code` | 予測対象の銘柄コード |
| `target_stock.name` | 通知やログに表示する銘柄名 |
| `ai_params.budget` | 仮想売買で使う予算 |
| `ai_params.future_days` | 何営業日後までを判定対象にするか |
| `ai_params.target_percent` | 何%以上の上昇を「成功」とみなすか |
| `ai_params.threshold` | 買いシグナルを出す最低確率 |
| `ai_params.stop_loss_percent` | バックテスト上の損切り率 |
| `feature_columns` | AIモデルに入力する特徴量 |
| `backtest_candidates` | バックテストで比較する銘柄コード |
| `log_only_stocks` | 通知せず、銘柄別CSVへ成績を残す監視銘柄 |

しきい値を上げると、通知は減りますが、より慎重な判定になります。しきい値を下げると、通知は増えますが、誤判定も増える可能性があります。

## 自動実行

`.github/workflows/daily_ai_trade.yml` により、GitHub Actions 上で平日 16:00 JST 頃に次の処理が走る想定です。

1. 依存ライブラリをインストール
2. `python train.py` でモデル学習
3. `python main.py` で予測と通知
4. `data/trade_log.csv` に変更があればコミットして push

GitHub Actions で Slack 通知を使う場合は、リポジトリの Secrets に `SLACK_WEBHOOK_URL` を登録してください。

## 出力されるもの

| パス | 内容 |
| --- | --- |
| `models/stock_ai_model.pkl` | 学習済みAIモデル |
| `data/trade_log.csv` | 買いシグナルと結果の記録 |
| `log/` | 実行ログ |

## 開発メモ

- データ取得は `yfinance` を使っています。
- 日本株コードは `8306` のように指定し、内部で `8306.T` に変換されます。
- AIモデルは `lightgbm.LGBMClassifier` です。
- テクニカル指標は `pandas-ta` で計算します。
- 学習・予測・バックテストで同じ特徴量生成処理を使うため、`src/analysis.py` が共通部品になっています。
