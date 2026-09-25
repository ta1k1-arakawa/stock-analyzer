# V13 方針レビュー 追記1：最新コミットの評価 (2026-09-25)

```text
document_type=EXTERNAL_ADVISORY_REVIEW_FOLLOWUP
author=Claude Code (ユーザー依頼による外部レビュー)
parent_review=docs/REVIEW_V13_2026-09-25.md (source commit b0940e3006d0b9135642e09d51bd470e70615fb8)
reviewed_branch=v13-conditional-cross-sectional-short-horizon
reviewed_range=bfde3d609cf965d3be3666264de73ea13966e04e..95721330bc6d89118b471218f4f3c17e0bb70965
reviewed_head=95721330bc6d89118b471218f4f3c17e0bb70965
authority_created=none
frozen_artifacts_modified=false
v13_branch_modified=false
market_data_requests=0
private_content_reads=0
```

本書は、最初のレビュー（以下「親レビュー」）の後にv13ブランチへ追加された8コミットの評価である。
親レビューと同じく助言のみであり、凍結済みの設計・状態・権限を変更せず、いかなる実行権限も生まない。

---

## 0. 結論

- **方向: 概ね良い。** 作業は最初の測定への一本道（除外の適用 → 500銘柄選定）の上にある。
- **効率: 変わっていない。** 300銘柄の識別子リストを読むだけの工程に、約900行のコードと
  レビューの往復（BLOCK 1回を含む）がかかっている。
- **直近で最大のリスク:** 次の工程が1回限り・再試行なしの設計になっていること。
  過去にはWindows/PowerShellのラッパー不具合で、1回限りの許可が消費された前例がある（§3.3）。
- 親レビューのP2（結果観測前の設計修正）を採用しなかった判断は妥当であり、支持する（§2.3）。

---

## 1. 対象コミット

| コミット | 内容 | GPTレビュー |
|---|---|---|
| `4c40fea`, `459c363`, `01d0bd0`, `ee20c2d` | レビューツールが誤って作成した空ファイル（`__noop__`, `SHOULD_NOT_CREATE`）の追加と削除。Issue #74でツール事故として記録 | — |
| `5151309` | 親レビューを `NON_AUTHORITATIVE_ADVISORY` として取り込み、`AGENTS.md` に参照ルールを追記（Issue #76） | PASS C0/H0/M0/L0 |
| `f2f0878` | 回収済みJ-Quants成果物からT1の300件だけを読む仕組み（T1だけを読むストリーミング走査、1回限りの消費受領書、Windows直接実行ラッパー）（Issue #75） | **BLOCK** C0/H1/M0/L0 |
| `5e265ee` | 上記HIGH_1の修正：事前ゲートで停止した際の「許可未消費」誤報告を「不明・再利用不可」に変更（Issue #77） | PASS C0/H0/M0/L0 |
| `9572133` | 回収済みT1を読むための1回限りの使用時許可の記録（Issue #78） | 待ち |

`bfde3d6..9572133` の `scripts/`, `src/`, `tests/` への追加: 5ファイル、884行。

現在の段階: `V13_JQUANTS_T1_POINT_OF_USE_AUTHORIZATION_RECORD_AWAITING_GPT_EXACT_SHA_REVIEW`。
次は、この許可記録のGPTレビューと、別Issueでの `DIRECT_WINDOWS_POWERSHELL` 実行。

---

## 2. 良い点

### 2.1 止まっていた工程が前進した

親レビュー時点では、除外リストが確定しないため銘柄選定に進めなかった。
今回、全300件除外の承認がGPT PASSとなり、適用の仕組みも実装とレビューを通過して実行直前まで来ている。

### 2.2 親レビューが実際に参照されている

`AGENTS.md` に次の文が追加された。

> When a review identifies process overhead, consider the shortest scientifically valid path to the next
> informative measurement without weakening real, private, or one-shot safety boundaries.

また `f2f0878` の決定ログには「process-delay finding is accepted for cadence planning」と記録されている。

### 2.3 P2を採用しなかった判断は妥当

`f2f0878` は「frozen V13 methodology stays unchanged for the first measurement. P2-a and P2-b were not implemented」と記録している。
この判断を支持する。親レビュー §2.2(a) の補足として次の2点を挙げる。

- 凍結済みの判定条件には I/J（日次の順位相関IC）と K/L（予測上位10%と全体の平均差）が含まれる。
  これらは上位銘柄群をまとめて評価する指標であり、P2-a（上位N銘柄ポートフォリオ）が狙う検出力の多くをすでに担っている。
- 設計修正にはさらに1回の凍結サイクルが要る。先に1回測定する方が、学習速度の面で有利である。

したがって、親レビューが示した「1銘柄集中による検出力不足」は、判定A（損益）などの**単一ポジションの損益系の条件**に主に当てはまる。
I/J/K/L の結果は、それとは別に重視して読むべきである（§4.4）。

---

## 3. 問題点

### 3.1 工程の目的に対して実装が過大

この工程の目的は「ローカルの回収成果物から300個の銘柄コードを読み、除外に使う」ことである。
実装（`f2f0878`, `5e265ee`）には次が含まれる。

- 関係するフィールド以外をPythonオブジェクトにせず構文だけ検証する、標準ライブラリだけのストリーミングJSON走査
- 最初の1バイトで消費扱いとし、残りを読む前に受領書をWindowsの書き込みスルーと `FlushFileBuffers` で永続化する仕組み
- 出力ルートの構造確認、既存出力での停止、PowerShellラッパー

守ろうとしているもの（V8の他区画 T0/T2/T3/T_spare の識別子を読まないこと、T1の識別子を公開しないこと）は理解できる。
ただし、V8系列の研究は終了済み（`V8K_TERMINATION_RECORD.md`、`TERMINATED_PRE_PRIVATE_PARTITION`）であり、
読む対象は「除外すべき銘柄のリスト」にすぎない。
守る価値に対して工数と失敗面が大きい。

### 3.2 複雑さが新しいBLOCKを生んでいる

`f2f0878` のHIGH_1（`PRE_GATE_DURABLE_STATE_COLLISION_CAN_BE_MISREPORTED_AS_AUTHORIZATION_UNCONSUMED`）は、
除外の中身の誤りではない。1回限りの消費管理という仕組みそのものの不具合である。
V11とV12も、この種の「仕組みの不具合 → 修正 → 再レビュー」の往復で予算を使い切り、`PAUSE` した（`V12_ROOT_CAUSE_POSTMORTEM.md`）。

### 3.3 1回限り・再試行なしの設計が、最大の停止リスクになっている

`V13_JQUANTS_T1_EXCLUSION_POINT_OF_USE_AUTHORIZATION.json` は `single_use=true`, `automatic_retry=false`, `second_private_source_read=false` である。
最初の1バイトを読んだ後の失敗は、終端扱いとなる。

過去の記録では、市場と無関係なWindows/PowerShellのラッパー不具合が繰り返し起きている。

| 記録 | 内容 |
|---|---|
| `PROJECT_DECISION_LOG.md` 2026-09-01 V9_010 Stage-A attempt 1 | `UNRECOVERABLE_BECAUSE_EXTERNAL_POWERSHELL_WRAPPER_LOST_RUNNER_STDERR`。**人間の許可が消費され、再実行不可となった** |
| `PROJECT_DECISION_LOG.md` 2026-09-04 V9_014 E5 HIGH_1 | Windows PowerShellのネイティブstderr取得の修正 |
| `RESEARCH_VIABILITY_CHECKPOINT.md` | `PHASE_C_INSPECTION_WRAPPER_EMPTY_STDERR_NULL_HANDLING_BUG` |

同種の失敗が今回起きれば、新しい許可と後継の設計が必要になり、V13は再び数日〜数週間止まる。
市場の証拠を一切得ないまま研究が止まる経路として、現時点で最も確率が高いのはこれである。

### 3.4 レビューの頻度は変わっていない

親レビューのP1（レビューを「設計・修正の固定」と「1回限りの本番実行前コード」の2点に絞る）は、運用に反映されていない。
`9572133`（許可を記録するだけのコミット）も、単独でGPTの厳密SHAレビュー待ちになっている。
空ファイルのコミット4件（§1）についても、独立したIssueで扱われた。

---

## 4. 提案（いずれも人間の判断が必要）

### 4.1 設計は凍結のまま、最初の測定まで新しい設計を追加しない

現在の判断を維持する。P2は最初の測定の後に、新しい研究として検討すればよい。

### 4.2 本番の読み込み前に、同じ環境で予行演習を1回行う

本番と同じWindows機・同じPowerShellラッパー・同じ呼び出し方で、ダミーの回収成果物（一時ルート）を対象に実行する。
次の3点を確認する。

- 正常終了時の標準出力・標準エラーの取得
- 終了コードの取得
- 受領書の書き込み

ラッパーのパスが固定されていて一時ルートを指定できない場合は、その指定手段を追加すること自体を最小の変更として検討する。

### 4.3 許可条件を「技術的失敗なら同じ許可で再試行可」に変更することを検討する

読む対象は終了済みV8のT1識別子であり、読み込みを繰り返しても研究の統計的な妥当性は損なわれない。
1回限りの厳格さが意味を持つのは、結果を観測するV13本番の実行（設計書の `STATISTICALLY_IRREVERSIBLE_GATE`）である。
識別子の読み込みにまで同じ扱いを適用する必要性は低い。

変更案:

- 市場データやアウトカムを読んでいないことが確認できる技術的失敗は、同じ許可での再試行を1〜2回まで認める
- 上限を超えた場合は人間の判断に戻す

### 4.4 残りの工程をまとめ、GPTレビューを1回に絞る

「除外の適用 → 500銘柄選定と宇宙マニフェスト → Yahoo取得 → データ固定 → 実行前検証」を、
可能な限り1つのIssueにまとめる。GPTの厳密SHAレビューは、1回限りの本番実行（A〜Q判定）の直前に1回だけ行う。

結果を読むときの注意（§2.3より）:

- 単一ポジションの損益系の条件（A、B、C、Mなど）は検出力が低い
- I/J/K/L は検出力が相対的に高い
- A〜Qの総合判定が `STOP` であっても、I/J/K/L の値は次の研究の事前確率として記録・参照する価値がある
  （同じ研究の救済には使わない）

### 4.5 期限を日付で置く

例: 2026-10-09 までに最初のA〜Q判定を出す。期限までに届かない場合は、原因を市場以外の技術的要因として記録し、
工程の簡略化を人間が判断する。

---

## 5. GPTレビュアーへの依頼事項

1. §1の事実関係（コミット、判定、許可の条件）に誤りがないか確認してください。
2. §3.3のリスク評価に同意しますか。同意しない場合、V9_010 attempt 1 と同種の失敗が今回は起きないと言える根拠を示してください。
3. §4.3（識別子の読み込みを技術的失敗に限り再試行可とする）は、V13の科学的な妥当性と、公開文書に私的識別子を出さない方針を損ないますか。
4. §4.4の「残りの工程を1 Issueにまとめ、本番直前のレビュー1回にする」案について、省略すると検証の妥当性が実際に落ちるゲートがあれば、それだけを挙げてください。
5. §2.3の補足（I/J/K/L が検出力を担っている）を踏まえ、親レビュー§2.2(a)の評価の修正が必要か確認してください。
