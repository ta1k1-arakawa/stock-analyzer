---
name: stock-analyzer-direction-review
description: stock-analyzer の現在の研究・開発方向を、将来の実運用における credible expected future real-world profitability への寄与という観点で独立レビューする。明示的に呼び出された場合だけ使用する。
disable-model-invocation: true
---

# stock-analyzer direction review

このSkillの目的は、過去の作業を正当化することではなく、ここから先の時間を何に使えば将来利益性について最も価値の高い情報を得られるかを、repository evidence に基づいて独立に判断することである。past engineering effort は sunk cost として扱う。Version、study、branch、コード量、文書量、AI利用、historical PnL最大化、モデルの複雑化そのものを目的にしない。

## authority と安全境界

レビュー開始時に、必ず次をこの順に行う。

1. 最初に `AGENTS.md` を読む。
2. `AGENTS.md` が要求する `AI_RESEARCH_EXECUTION_RULES.md`、`AI_STOCK_ANALYZER_REVIEW_POLICY.md`、`AI_GITHUB_ISSUE_ORCHESTRATION_WORKFLOW.md`、`AI_REAL_EXECUTION_RUNBOOK.md`、必要に応じて `AI_RESEARCH_CHECKPOINT_WORKFLOW.md`、`AI_RESEARCH_DEVELOPMENT_EFFICIENCY_WORKFLOW.md`、`CLAUDE.md` を読む。
3. `PROJECT_STATE.md` を読み、`authoritative_branch`、`current_study`、`current_stage`、直近のレビュー状態、open finding、next action を確認する。
4. current study の design、review、凍結・承認記録、関連する GitHub Issue または repository 内の Issue mirror が、`PROJECT_STATE.md` と整合するか確認する。READY Issue が実在する場合はその実行単位仕様も読む。会話の記憶で補わない。
5. `PROJECT_STATE.md` など権威ある repository state から `authoritative_branch` を復元する。`main`、default branch、現在 checkout 中のbranch、最終更新日時が新しいbranch、過去チャットのbranch、Claude/Codex の一時branchを推測で採用してはならない。`main` は state が明示的に `authoritative_branch=main` の場合だけ対象にできる。
6. その branch が remote に実在することを確認し、remote branch の最新 HEAD を取得・確認する。レビュー基準点はその remote HEAD であり、local checkout の都合ではない。`git ls-remote` で存在と最新SHAを確認し、必要なら `git fetch origin <authoritative_branch>` でその exact SHA を安全に取得して読む。
7. state 内の複数キー、design、Issue、remote branch が相互に矛盾する、branch名やHEADを一意に決められない、remote branch が存在しない、または local branch/HEAD が対象 remote HEAD と一致しない場合は推測・branch切替・mergeをせず停止する。branchの権威が不明な場合は `AUTHORITATIVE_BRANCH_STATUS=AMBIGUOUS` と報告する。local sync 不整合も明示してレビューを実質開始しない。

レビューだけを目的とする読み取りでは、ネットワーク市場データ、private/sealed data、ticker選択、model fit、backtest、historical screen、forward paper、real trading、Slack送信を実行しない。既存の frozen methodology、execution authority、human gate、研究結果、現在 study の意味を変更しない。レビューの提案は authority の付与、gate の消費、方法論の変更、studyの自動継続を意味しない。方法論上の選択が必要で、既存文書に指定がなければ `CHATGPT_DECISION_REQUIRED` とし、勝手に選ばない。

`PROJECT_STATE.md` の現在値、対象 remote HEAD、current study の design/review、実装、テスト、provenance、利用可能な成果物だけを根拠にする。implementation PASS は profitability の証明ではなく、historical profit 増加も将来利益性の証明ではない。`future profitability` が forward evidence で確立されていない限り、確立済みと表現しない。

## レビューの問い

現在 route が、最終的に次の end-to-end system に実質的に近づいているかを評価する。

`株価・市場データ取得 → 将来リターン／売買機会の予測 → コスト・リスク込みの経済評価 → trade / no-trade → 十分に魅力的な機会だけSlack通知`

最低限、次を具体的な evidence と不足に分けて検討する。

- 最終目的との整合性：現在のnext taskが、実運用で信頼できる正の期待利益に近づくか。
- 最大のprofitability uncertainty：本当に将来利益が出るかを判断する最大の未解決問題は何か。current task がそれを直接減らすか。
- information gain：結果が continue、stop、simplify、pivot、forward evaluation の判断を変え得るか。
- opportunity cost：同じ時間でより大きな profitability information を得る研究がないか。既済作業は sunk cost。
- scientific credibility：future leakage、holdout tuning、outcome-driven変更、favorable ticker/period選択、multiple testing、overfitting、再現性、OOS/forward evidence。
- economic realism：fees、slippage、lot size、required capital、liquidity、fillability、capital lock、no-fill、skip、exit delay、drawdown、concentration、実際に取引可能か。prediction accuracy単独で評価しない。
- plumbing と overengineering：environment、parser、transport、provenance、governanceの必要最小限と、profitability evidenceをほぼ増やさない複雑化を分ける。governanceを無視する提案はしない。
- model/AI：高度さを目的にせず、単純な方法のrobustness、reproducibility、forward evidence、net profitabilityを評価する。richer ML、cross-sectional ranking、ensemble、追加features、fundamentals、news、alternative data、regime detectionは、現在 frozen study を変更せず、information gainの根拠がある将来候補としてのみ扱う。
- Slack：signalの価値が十分に確認される前に、integrationやproduction polishを主作業にしない。

証拠は次の階層で重み付けする（下位のPASSを上位の利益性証明に読み替えない）。

`implementation correctness < data readiness / data quality < causal historical / OOS evidence < realistic cost-aware OOS evidence < robustness / capacity evidence < forward paper evidence < 実運用に近いforward evidence`

現在 study が存在することを、完了まで続ける理由にしない。frozen methodology や human gate を破らずに、study継続より pivot、pause、簡略化の方が価値が高い場合は明示する。

## 判定とnext step

次のいずれか1つを `DIRECTION_VERDICT` として返す。

- `ON_TRACK`：方向が合理的で、重要な利益性不確実性を効率よく減らしている。
- `ON_TRACK_BUT_SIMPLIFY`：方向は正しいが、実装・研究・ガバナンスが情報価値に対して複雑化している。
- `PIVOT_RECOMMENDED`：別の研究・戦略・モデル・評価方法の方が、同程度の時間で大きな利益性情報を得そうである。
- `PAUSE_RECOMMENDED`：現在 route を続ける追加価値が低い。
- `INSUFFICIENT_EVIDENCE`：方向性判断に必要な repository evidence が不足している。

原則として、次の1つだけを `HIGHEST_VALUE_NEXT_STEP` にする。候補は、`profitability impact × information gain × 重要な不確実性を解消する確率` が大きく、`implementation cost + research time + overfitting risk + operational complexity + governance cost` が小さいものを優先する。根拠のない数値expected valueは作らない。提案は現行のauthority範囲内の安全な次のレビュー・実装単位として記述し、追加の方法論、data source、threshold、ticker、gate、executionを暗黙に決めない。

## 標準出力契約

以下の形式を維持し、値が不明なら推測せず `UNKNOWN`、不整合なら上記の `AMBIGUOUS` とする。`AUTHORITATIVE_BRANCH_SOURCE` は state と設計・Issueのどの権威資料から判断したかを簡潔に示す。`REMOTE_HEAD` と `REVIEWED_HEAD` は原則として authoritative remote branch の最新HEADに一致させる。

```text
AUTHORITATIVE_BRANCH_STATUS=
AUTHORITATIVE_BRANCH=
AUTHORITATIVE_BRANCH_SOURCE=
REMOTE_HEAD=
REVIEWED_HEAD=

CURRENT_STUDY=
CURRENT_STAGE=

END_GOAL=credible expected future real-world profitability
CURRENT_ROUTE=

DIRECTION_VERDICT=

WHY=
- ...
- ...
- ...

BIGGEST_UNRESOLVED_PROFITABILITY_QUESTION=

HIGHEST_VALUE_NEXT_STEP=
WHY_THIS_STEP=

CURRENT_WORK_TO_KEEP=
CURRENT_WORK_TO_SIMPLIFY_OR_DEFER=

END_TO_END_GAPS=
DATA_ACQUISITION=
PREDICTION=
ECONOMIC_DECISION_RULE=
OOS_OR_FORWARD_EVIDENCE=
SLACK_NOTIFICATION=

SCIENTIFIC_RISK=
EXECUTION_REALISM_GAP=
OPPORTUNITY_COST=

WHAT_RESULT_WOULD_CHANGE_THE_DIRECTION=

DEFERRED_IDEAS=

AUTHORITY_BOUNDARY=
```

最後に、過去のVersionを守ることではなく、stock-analyzerの最終目的に近づくことを優先したかを確認する。ただし、このSkill自身は研究methodology、frozen design、execution authority、human gate、研究結果、study identityを変更しない。
