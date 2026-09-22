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
7. state 内の複数キー、design、Issue、remote branch が相互に矛盾する、branch名やHEADを一意に決められない、remote branch が存在しない、または authoritative remote HEAD を一意に取得できない場合は推測せず停止する。branchの権威が不明な場合は `AUTHORITATIVE_BRANCH_STATUS=AMBIGUOUS` と報告する。local branch/HEAD が authoritative remote branch/HEAD と一致しないことだけでは停止しない。local checkoutを自動でswitch、checkout、reset、merge、rebaseせず、必要なら `LOCAL_CHECKOUT_STATUS=MISMATCH` と補足し、取得したremote HEADのrepository evidenceをread-onlyでレビューする。
8. 証拠の状態を混同しない。`current authoritative remote repository state`、アクセス可能な `live GitHub Issue state`、`PROJECT_STATE` / `PROJECT_DECISION_LOG` などの repository mirror、過去の review record をそれぞれ別の層として記録する。live Issue にアクセスできる場合は、その現在の open/closed、status、next action と mirror を照合し、stale な next-action 文言を検出する。live Issue にアクセスできない場合はその制約を明記し、stale かもしれない mirror から live Issue の現在状態を事実として断定しない。stale な `CURRENT_STAGE`、`last_gpt_reviewed_sha`、next-action は、より新しい exact-SHA review、closed Issue、またはより厳格な frozen artifact を上書きしない。
9. `DIRECTION_VERDICT` または `HIGHEST_VALUE_NEXT_STEP` を選ぶ前に、適用される frozen design、approval、current Issue、terminal disposition を明示的に確認する。implementation/time budget、maximum remediation rounds、stopping rule、terminal disposition rule、one-shot/gate constraint、authority limitation の有無と状態を調べる。`FROZEN_STOPPING_RULE_STATUS` は `NOT_APPLICABLE`、`WITHIN_LIMIT`、`LIMIT_EXCEEDED`、`UNKNOWN` のいずれかで回答する。binding な frozen condition が pause、stop、terminal disposition を要求している場合、route の科学的な形が合理的でも同じ study の継続を推奨してはならない。route quality と現在の authorized/disposition state は別物であることを説明し、verdict と next step に binding な stop/pause を反映する。この場合の highest-value next step は追加の implementation や execution ではなく、必要な state transition、review、recording action とし、frozen rule を再解釈・免除・waive してはならない。`LIMIT_EXCEEDED` の場合は、正確な repository evidence を示し、同じ frozen study の継続を防止する。

レビューだけを目的とする読み取りでは、ネットワーク市場データ、private/sealed data、ticker選択、model fit、backtest、historical screen、forward paper、real trading、Slack送信を実行しない。既存の frozen methodology、execution authority、human gate、研究結果、現在 study の意味を変更しない。レビューの提案は authority の付与、gate の消費、方法論の変更、studyの自動継続を意味しない。方法論上の選択が必要で、既存文書に指定がなければ `CHATGPT_DECISION_REQUIRED` とし、勝手に選ばない。

## 出力言語

機械可読な field name、固定 enum、status token、branch name、SHA、study ID、task/status token は正確な表記を維持する。それ以外の free-text value、箇条書き、説明、根拠、要約、caveat、recommendation は、必ず自然な日本語で記述する。repository terminology として必要な英語 technical term は日本語の文中に含めてよいが、参照資料が英語であることを理由に英語の説明文を返してはならない。

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

ただし、上記の frozen stop-rule inspection が `LIMIT_EXCEEDED`、または binding な pause/stop/terminal disposition を示す場合は、この原則より frozen rule を優先する。route が合理的であることを理由に、停止後の同じ study の implementation、execution、継続調査を `HIGHEST_VALUE_NEXT_STEP` にしてはならない。

## 標準出力契約

以下の形式を維持し、値が不明なら推測せず `UNKNOWN`、不整合なら上記の `AMBIGUOUS` とする。`AUTHORITATIVE_BRANCH_SOURCE` は state と設計・Issueのどの権威資料から判断したかを簡潔に示す。`REMOTE_HEAD` と `REVIEWED_HEAD` は原則として authoritative remote branch の最新HEADに一致させる。

```text
AUTHORITATIVE_BRANCH_STATUS=
AUTHORITATIVE_BRANCH=
AUTHORITATIVE_BRANCH_SOURCE=
REMOTE_HEAD=
REVIEWED_HEAD=

FROZEN_STOPPING_RULE_STATUS=
FROZEN_STOPPING_RULE_EVIDENCE=

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

最後に、過去のVersionを守ることではなく、stock-analyzerの最終目的に近づくことを優先したか、live Issue と repository mirror の差異を確認したか、frozen stop-rule を先に評価したかを確認する。ただし、このSkill自身は研究methodology、frozen design、execution authority、human gate、研究結果、study identityを変更しない。
