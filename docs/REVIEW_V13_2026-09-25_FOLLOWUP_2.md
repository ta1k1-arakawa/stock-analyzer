# V13 方針レビュー 追記2：追記1の採否と実行前の補足 (2026-09-25)

```text
document_type=EXTERNAL_ADVISORY_REVIEW_FOLLOWUP
author=Claude Code (ユーザー依頼による外部レビュー)
parent_review=docs/REVIEW_V13_2026-09-25.md
previous_followup=docs/REVIEW_V13_2026-09-25_FOLLOWUP.md (source commit ab721031753fce5371e5da2ec60236f10440d42b)
reviewed_branch=v13-conditional-cross-sectional-short-horizon
reviewed_head=95721330bc6d89118b471218f4f3c17e0bb70965
authority_created=none
frozen_artifacts_modified=false
v13_branch_modified=false
market_data_requests=0
private_content_reads=0
scope_excludes=main branch daily notification (human decided to keep main as is)
```

本書は、追記1に対する人間の採否を記録し、その採否を前提に、実行前に追加コストなしで入れられる補足をまとめたものである。
これまでと同じく助言のみであり、凍結済みの設計・状態・権限を変更せず、いかなる実行権限も生まない。

main ブランチの日次通知（親レビュー P4）は、人間が現状維持と判断したため本書の対象外とする。

---

## 1. 追記1に対する人間の採否（ユーザーから伝えられた内容）

| 追記1 | 採否 | 人間の判断の要旨 |
|---|---|---|
| §4.1 最初の測定まで設計変更しない | **採用** | P2-a/P2-b も今は入れない |
| §4.2 同じ環境での予行演習 | **採用** | 本番直前に、同じWindows機・同じレビュー済みラッパーで、execute switch なしの PRE_GATE 予行演習を行う。新しい production code は追加しない。`PRE_GATE_EXECUTION_SWITCH_REQUIRED`・source open 0・boundary 未到達を確認してから、本番を1回だけ実行する |
| §4.3 技術的失敗時の再試行 | **今回は不採用** | 同じ固定 T1 成果物の再読は、科学的には必ずしも bias を増やさない。ただし現在の人間の許可とレビュー済み契約は single-use/no-retry であり、今変更すると凍結・承認・レビューが増える。将来の identity-only 処理は、最初から決定的で再利用可能な処理として設計する余地がある |
| §4.4 工程の統合 | **強く採用** | T1 読み込み成功後、「除外適用 → 500銘柄 → 取得 → 固定 → preflight」の互換な工程をできるだけまとめる。ただし、今回の private T1 の1回限りの読み込みと、最終的な A〜Q の statistically irreversible な実行は、独立した関門として残す |
| I/J/K/L の位置づけ | **採用** | 親レビューの「検出力不足」は V13 全体ではなく、主に単一ポジションの PnL 系への批判と捉える。I/J/K/L はより情報量の多い cross-sectional signal diagnostics である。ただし、同じ V13 の STOP を後から救済する材料には使わない |
| §4.5 期限を日付で置く | 未決 | — |

本書はこれらの採否に同意する。特に §4.3 の不採用は「今契約を変えるコスト」と「失敗した場合のコスト × 失敗確率」の比較として妥当である。

---

## 2. 補足A：§4.2 の予行演習が通らない区間と、コード追加なしの補完

### 2.1 予行演習の到達範囲

`scripts/run_v13_jquants_t1_exclusion_direct_windows.ps1`（`9572133`）の処理順:

| 行 | 処理 | execute switch なしの予行演習 |
|---|---|---|
| 1〜109 | git の状態、blob、許可記録、実行環境、private フォルダ構成の確認 | 通る |
| 110 | execute switch がなければ `PRE_GATE_EXECUTION_SWITCH_REQUIRED` で停止 | ここで停止 |
| 117 | `$ErrorActionPreference='Stop'` の下で、`& $python -E -B -m scripts.v13_execute_jquants_t1_exclusion_private_read ... 2>$null` を実行し、stdout を行配列で受け取る | **通らない** |
| 118〜131 | 終了コードの取得、19行の報告の形式検査、終了コードと `EXECUTION_RESULT` の整合確認 | **通らない** |

予行演習は、設定ミスが起きやすい区間（1〜109行）を確認でき、有効である。
一方、117行以降（子プロセスの起動、出力の受け取り、報告の解析）は予行演習では一度も実行されない。

この区間は、1回限りの読み込みが実際に消費される区間であり、次の前例がある区間でもある。

- `PROJECT_DECISION_LOG.md` 2026-09-01 V9_010 Stage-A attempt 1: `UNRECOVERABLE_BECAUSE_EXTERNAL_POWERSHELL_WRAPPER_LOST_RUNNER_STDERR`、人間の許可が消費され再実行不可となった

### 2.2 コード追加なし・private 無接触で 117〜131 行を試す方法

ハーネス `scripts/v13_execute_jquants_t1_exclusion_private_read.py` の `main()`（130行）は、134行目で
環境変数 `V13_JQUANTS_T1_WRAPPER_GATE` が `REVIEWED_PRE_GATE_PASS` でなければ、次の報告を出して終了コード1で戻る。

- `EXECUTION_RESULT=PRE_GATE_STOP`
- `FAILURE_CLASS=PRE_GATE_WRAPPER_REQUIRED`
- `SOURCE_OPENS=0`

これは `LOCALAPPDATA` の参照とソースのパス解決（`execute()` の呼び出し）より前に起きる。
また、この報告の19個のキーは、ラッパーの `$expectedReportKeys` と同じ順序である。

したがって、同じ Windows 機・同じ PowerShell で、環境変数を**設定せずに**117行目と同じ呼び出しを手動で行えば、次を確認できる。
private ソースには触れず、新しいコードも要らない。

- 同じ Python・同じフラグ（`-E -B -m`）での起動とモジュール読み込み
- `$ErrorActionPreference='Stop'` と `2>$null` の組み合わせで例外が起きないこと
- stdout が19行の配列として受け取れ、各行がラッパーの安全パターンに合うこと
- 終了コード1と `EXECUTION_RESULT=PRE_GATE_STOP` が整合すること

手順例（リポジトリのルートで実行）:

```powershell
Remove-Item Env:V13_JQUANTS_T1_WRAPPER_GATE -ErrorAction SilentlyContinue
$PSVersionTable.PSVersion
& {
    $ErrorActionPreference = 'Stop'
    $repo = (Get-Location).Path
    $python = Join-Path $repo '.venv-real-execution\Scripts\python.exe'
    $lines = @(& $python -E -B -m scripts.v13_execute_jquants_t1_exclusion_private_read --repository-root $repo 2>$null)
    "LINES=$($lines.Count) EXIT=$LASTEXITCODE"
    $lines
}
```

期待される結果:

- 例外なし
- `LINES=19 EXIT=1`
- `EXECUTION_RESULT=PRE_GATE_STOP`
- `FAILURE_CLASS=PRE_GATE_WRAPPER_REQUIRED`
- `SOURCE_OPENS=0`
- `PRIVATE_CONTENT_READS=0`

### 2.3 PowerShell のバージョン確認

Windows PowerShell 5.1 では、`$ErrorActionPreference='Stop'` の下で外部コマンドの stderr をリダイレクトすると、
stderr への出力がエラーレコードとして扱われ、終了エラーになる場合がある（PowerShell 7.2 以降は既定でこの挙動がない）。
ラッパーの catch 節では、117行目以降で起きた例外は `POST_BOUNDARY_FAILURE` / `HARNESS_REPORT_UNKNOWN`（許可の消費状態は不明、再利用不可）として扱われる。

- 5.1 で実行する場合、本番成功時に子プロセスが stderr に何も書かないことが前提になる。
- `execute()` は例外を捕捉して報告に変換するため、成功経路で stderr 出力が起きる可能性は低い。ただし、実行する PowerShell のバージョンは本番前に記録しておく価値がある。

### 2.4 既存の合成テストを同じ環境で実行する

`tests/test_v13_execute_jquants_t1_exclusion_private_read.py` と `tests/test_v13_resolve_jquants_t1_exclusion_state.py` を、
本番と同じ Windows 機・同じ `.venv-real-execution` で実行する。受領書の書き込みなど、Python 側の境界後の処理を実機で確認できる。
新しいコードは要らない。

---

## 3. 補足B：§4.3 不採用の前提で、失敗時の手順だけ先に決めておく

契約（single-use/no-retry）は変えずに、実行 Issue に次のような一文を書いておくことを提案する。

> 本実行が、市場データ・価格・アウトカムを読んでいない技術的失敗（`POST_BOUNDARY_FAILURE` を含む）で終わった場合、
> 次の手順は「同じレビュー済みコード（必要なら最小修正）に対する、新しい1回限りの使用時許可」とする。後継の設計は作らない。

失敗しなければコストはゼロである。失敗した場合でも、停止を「新しい後継設計を含む数日〜数週間」から「許可とレビューの1サイクル」に抑えられる。

参考: 受領書 `_receipt()` の `operation_class` は、この identity-only の読み込みにも `STATISTICALLY_IRREVERSIBLE_GATE` を使っている。
これは V13 本番の A〜Q 判定と同じ区分である。人間が §4.3 で述べた「将来の identity-only 処理は最初から再利用可能な処理として設計する」を実行する際には、
この区分を分けることが出発点になる。

---

## 4. 補足C：§4.4 の統合 Issue では、再試行してよい失敗を最初に明記する

V13 設計書（`V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md` 325行）は、
公開データ取得を `RETRIABLE_PUBLIC_PLUMBING`、最初のアウトカム実行を `STATISTICALLY_IRREVERSIBLE_GATE` と区別している。

「除外適用 → 500銘柄 → 取得 → 固定 → preflight」を1つにまとめる Issue でも、冒頭に次を書いておくことを提案する。

- 再試行してよい失敗: Yahoo/JPX の通信失敗、タイムアウト、一時的な HTTP エラーなど、アウトカムを読んでいない公開データ取得の失敗
- 再試行してはいけない失敗: 価格アウトカムを使った計算に到達した後の失敗

これにより、通信の一時的な失敗が新しい関門や後継設計を生むことを防げる。
人間が残すと判断した2つの独立関門（T1 の1回限りの読み込み、A〜Q の実行）はそのまま維持する。

---

## 5. 補足D：I/J/K/L の使い方を、結果を見る前に一文で残す

人間の判断（I/J/K/L は同じ V13 の STOP を救済する材料には使わない）に同意する。

加えて、結果を見る前に、次の研究での使い方を一文で記録しておくことを提案する。例:

> A〜Q の総合判定が STOP で、I（平均IC）と K（上位10%と全体の差）がともに正だった場合、
> それは「単一ポジションでは検出できないが、横断的なシグナルが存在する可能性」の証拠として扱う。
> 次の研究で分散ポートフォリオ型を新しい設計・新しい事前登録で検討する根拠の一つとする。
> V13 の判定は変更しない。

結果の後に書くと、事後の解釈になる。結果の前に書けば、事前の方針になる。

---

## 6. 残る未決事項

- **期限（追記1 §4.5）**: 例として「2026-10-09 までに最初の A〜Q 判定」。
  期限までに届かなかった場合は、原因を記録し、工程の簡略化を人間が判断する。

---

## 7. GPTレビュアーへの依頼事項

1. §2.1 の行番号と処理順、§2.2 の「環境変数なしではソースに触れずに `PRE_GATE_WRAPPER_REQUIRED` で戻る」という読みが、`9572133` のコードと一致するか確認してください。
2. §2.2 の手動確認は、現在の許可・契約・ガバナンスの下で実行してよいものですか。production code の変更や private の読み込みに当たらないか判断してください。
3. §2.3 の PowerShell 5.1 の stderr の挙動について、本番の実行環境で問題になりうるか見解をください。
4. §3 の「失敗時の手順を実行 Issue に書く」ことは、契約の変更に当たらず追加のレビューなしに入れられますか。
5. §4 の再試行区分と、§5 の I/J/K/L の事前の使い方の記録について、V13 の凍結設計と矛盾しないか確認してください。
