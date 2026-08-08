# V4 Meta-Label MVP Protocol

研究仮説は「pooled meta-label classifier が、低い正の実現純利益率確率の取引を棄権することで、決定的な20日モメンタム戦略を改善できる」です。固定条件は `V4_META_LABEL_DESIGN.md` を完全に継承します。

実装上のバグ修正回数は制限しません。一方、実市場結果を見た後に特徴量、threshold、期間、モデル、合格条件を変更することは禁止です。MVPの正式な実データ評価はコード完成後に1回だけ行います。有望な結果であっても deployment または実注文を意味しません。

正式成果物は `summary.json`、`trades.csv`、`predictions.csv` の3件です。今回のPhase 1では実データ評価を行いません。
