# Loop Engineering Phase A specification

## project_scope

`loop-engineering-framework-only`

## Purpose

将来の非機密・非投資プロジェクトについて、人間が起動する安全な開発ループを構築する。その前段として、Phase Aは状態、契約、履歴、人間承認をGit管理された機械可読形式で表現し、読み取り専用で検証できるようにする。

## Non-goals

- stock-analyzer研究の再開
- 投資戦略の生成
- モデル学習
- バックテスト
- J-Quants契約
- shadow有効化
- 実注文
- 無人の無限ループ
- 自動での仮説生成

## Closed stock-analyzer status

- `research_status: CLOSED`
- `deployment_status: NO_CANDIDATE`
- `shadow_status: DISABLED`
- `paid_data_decision: DO_NOT_PURCHASE`
- `further_loop_on_same_data: PROHIBITED`
- closure commit: `2db8e08833e8fc4b96e93c36e0f1b2fc74c5f158`

このPhase Aは終了済みのstock-analyzer研究に作業を許可しない。終了済み事例として参照するだけであり、評価、モデル、データ、戦略、shadow、実注文を再開しない。

## Allowed work

- `loop_control/` 内の状態・契約・履歴・承認形式の手動記録
- 読み取り専用validatorによるスキーマと整合性の確認
- 人間によるGit差分、commit、branch、tagの確認

## Forbidden work

- run_once実行器、状態自動更新、自動commit、自動push
- Codex SDK、Codex Automation、GitHub Actions、schedule、常駐処理、lock実装
- 外部API、ネットワーク通信、秘密情報の追加または読取り
- データ取得、モデル学習、バックテスト、株式戦略変更
- stock-analyzer研究の再開、J-Quants、shadow有効化、実注文
- 既存branchへのmerge、tag作成、force push

## Mandatory human approval gate

次の操作は必ず明示的な人間承認を必要とする。

- 有料契約、認証情報追加、外部サービス登録
- 目的・設計・評価条件の変更、予算追加、新仮説追加
- merge、tag、deploy、schedule、shadow、実注文
- データ削除、既存結果の上書き、REJECTED後の別方式
- CLOSED研究の再開

Phase Aでは承認要求は存在しない。`human_approvals.jsonl` は空であり、承認を消費しない。

## Phase A stopping conditions

- stateまたは契約のスキーマ不正
- task hashまたは状態遷移の不整合
- `active: true`、0以外の予算、または実行タスクの登録
- stock-analyzer閉鎖状態の不一致
- runner、scheduler、lock、外部通信、秘密情報の追加
- 手動監査で許可外ファイルまたはGit参照変更が見つかった場合

## Phase A budget

全予算は0である。Phase Aは実験でも実行でもなく、bootstrap契約の読み取り検証だけである。

## Not implemented until Phase B or later

ローカル半自動`run_once`、排他lock、状態の自動遷移、独立Verifierの自動実行、CI、scheduleは実装しない。
