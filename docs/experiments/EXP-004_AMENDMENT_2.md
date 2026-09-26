# EXP-004 変更2：無料のデータ源と Claude の契約に切り替える（開始前に固定）

```text
status=FIXED_BEFORE_START（2026-09-26 に記録。開始は 2026-09-28）
reason=ユーザー依頼：J-Quants の契約を続けず、Claude API の従量課金も使わない
code=experiments/free_data.py、experiments/exp004_forward.py、.github/workflows/exp004_forward.yml
```

## 変更の内容

| 役割 | 変更前 | 変更後 |
|---|---|---|
| 株価・売買代金 | J-Quants | yfinance（Yahoo Finance、`<コード>.T`）。売買代金は 終値×出来高 で近似 |
| 純資産・純利益予想・発行株式数 | J-Quants | `forward/fundamentals.csv`（J-Quants の契約中に、計算に必要な数字だけを抜き出して保存）に、TDnet の決算短信・業績予想の修正の XBRL サマリーを毎日追加 |
| 時価総額 | J-Quants の MktCap | 発行株式数（自己株式を含む）× 終値。開示後の株式分割を補正 |
| TOPIX | J-Quants の指数 | TOPIX 連動 ETF（1306）で代用。1306 は ETF で、V13 の候補（個別株）ではない |
| 営業日 | J-Quants のカレンダー | 平日から `forward/tse_holidays.txt`（J-Quants から作成、2027年末まで）の休業日を除く |
| Claude の評価 | Claude API（`claude-opus-5`） | Claude Code を GitHub Actions で動かし、ユーザーの Claude の契約の利用枠を使う（`CLAUDE_CODE_OAUTH_TOKEN`）。モデルは契約の既定。1回の実行で最大25通 |

## 変更前との違いの確認（開始前）

- J-Quants の株価を yfinance の形式に並べ替えて新しい読み込み処理に入れ、R の銘柄の順位を比べた。
  順位相関は 0.995〜0.998、上位20銘柄の一致は 18〜20銘柄、時価総額は完全に一致した
- 違いの主な原因は売買代金の近似（終値×出来高）で、流動性1億円の境目の銘柄が入れ替わる
- 2026-09-28 の最初の注文は、J-Quants のデータで決めたもの（`forward/orders/2026-09-28.md`）をそのまま使う。以降は無料データで決める
- yfinance と TDnet は、この作業環境からは接続できなかったため、実データでの動作は GitHub Actions の初回実行で確認する
- LightGBM（M）のモデルは J-Quants のデータで学習して固定したもの。入力の特徴量は無料データから同じ定義で作る

## 変えないもの

判定の方法（変更1を含む）、ポートフォリオのルール、開始日、評価日は変えない。
