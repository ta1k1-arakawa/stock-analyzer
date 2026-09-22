# GitHub-Issue-First Orchestration Workflow

```text
document_type=REPOSITORY_GITHUB_ISSUE_EXECUTION_ORCHESTRATION_CONTRACT
status=ACTIVE
scope=CROSS_CHAT_TASK_DISPATCH_AND_EXACT_SHA_REVIEW_CADENCE_ONLY
```

This workflow persists the repository-first execution loop for ChatGPT,
GPT-5.6 Sol, Codex, Claude Code Cloud, and Orca-managed agents. It changes
no research methodology, threshold, data source, authority, frozen design,
study result, or human gate. `AGENTS.md`,
`AI_RESEARCH_EXECUTION_RULES.md`, `AI_REAL_EXECUTION_RUNBOOK.md`, applicable
frozen designs, human gates, and stricter task-specific rules always win.

## 1. Default control loop

```text
GPT connected-GitHub state restoration
-> GPT creates/updates exact GitHub execution Issue
-> execution agent reads repository governance + Issue
-> execution agent edits/tests/commits/pushes
-> GPT exact-SHA independent review from GitHub
-> PASS: GPT comments review and closes task
-> BLOCK: GPT comments findings and creates bounded remediation Issue(s)
```

The GitHub Issue is the authoritative task specification for that execution
unit, subordinate to repository governance, frozen designs, human gates, and
stricter task-specific artifacts. The human should normally need to send only
a short dispatcher prompt such as `GITHUB_ISSUE=<n>`; the executor reads the
task body from GitHub rather than relying on a copied conversational prompt.

## 2. Required executable Issue fields

Every `READY` execution Issue must define:

- repository;
- authoritative branch;
- exact 40-hex `EXPECTED_HEAD`;
- task ID;
- task status;
- executor class;
- read-first documents;
- exact objective;
- allowed files and actions;
- explicit prohibitions and gates;
- required targeted and static checks;
- commit and push conditions;
- required final report; and
- GPT final-review authority.

An Issue without an exact executable `EXPECTED_HEAD` must be `QUEUED` or
`BLOCKED`. An executor must never infer a missing expected HEAD.

## 3. Executor routing

Use semantic executor classes rather than depending on one permanent model
name:

- `CHEAP_CODEX_AGENT_OK`: default for closed, mechanical, well-specified
  implementation or remediation with limited scope and no methodology
  discretion.
- `STRONG_CODEX_REQUIRED`: nontrivial integration or reasoning where
  first-pass correctness materially benefits from the stronger executor.
- `STRONG_CODEX_PARENT_WITH_CHEAP_SUBAGENTS`: decomposable work with
  genuinely independent subscopes; the strong parent owns decomposition and
  integration and cheap agents perform bounded subtasks.
- `CLAUDE_CODE_CLOUD`: only where explicitly selected and appropriate under
  repository governance.
- `DIRECT_WINDOWS_POWERSHELL`: long-running, real-network, private, sealed,
  human-gated, or durable-state operations where the runbook requires direct
  Windows execution.

Cheap Codex is the default when the Issue is sufficiently closed. Do not use
a strong parent merely for a tiny bookkeeping task.

## 4. Subagents and one-writer branch discipline

Subagents may be used only for bounded tasks with clear interfaces and
non-overlapping scopes. Do not parallelize tasks that touch the same files
without an explicit integration plan, tasks where one output defines another
task's interface, methodology decisions, human-gated/private/real execution
boundaries, or tasks whose commits would race on the same authoritative
branch.

The strong parent remains responsible for integration, targeted checks, final
diff, commit, push, and report.

By default, only one repository-writing task may be `READY` or executing
against an authoritative branch at a time. For ordinary/local checkout mode,
before editing and again before push, the executor must verify:

- branch matches the Issue;
- local HEAD equals the exact `EXPECTED_HEAD`;
- authoritative remote HEAD equals the exact `EXPECTED_HEAD`;
- working tree is clean; and
- no unexpected files or history exist.

Any ordinary/local checkout mismatch is `EXPECTED_HEAD_MISMATCH` and requires
STOP. If `CODEX_MANAGED_TASK_WORKSPACE` applies, route to §5 before treating
the generated branch/worktree name or base/local-HEAD mismatch as a STOP;
after §5 realignment, the same exact-HEAD checks apply. A future task is
activated only after GPT reviews the preceding exact SHA and updates the next
Issue with the new exact parent.

## 5. CODEX_MANAGED_TASK_WORKSPACE mode

When an ordinary repository-writing task is launched from the GitHub/Codex
Start button, Codex may receive a generated task branch/worktree based on a
branch other than the authoritative branch. The generated name or base is not
by itself a mismatch. This compatibility mode changes workspace alignment
only; it creates no research, network, private-data, production, trading,
human-gate, or other execution authority.

Before editing, the executor must:

1. record the generated task branch/worktree identity;
2. require a clean working tree;
3. fetch/read Git metadata for the exact authoritative branch as needed;
4. require the authoritative remote HEAD to equal the Issue `EXPECTED_HEAD`;
5. require the exact `EXPECTED_HEAD` commit to be locally available;
6. realign only the generated task branch/worktree to that exact
   `EXPECTED_HEAD`; never check out or mutate another worktree's
   authoritative local branch;
7. evaluate target files and scope only after that exact-head realignment.

After editing and the required checks, the executor must create exactly one
task commit whose parent is the Issue `EXPECTED_HEAD`, unless the Issue
explicitly specifies otherwise. Immediately before pushing, it must verify
that the authoritative remote HEAD still equals the original `EXPECTED_HEAD`,
then push non-force with an explicit refspec from the task HEAD to the
authoritative branch:

```text
git push origin HEAD:refs/heads/<authoritative-branch>
```

If the remote moved, the parent or history is wrong or unexpected, the tree
is dirty, the push is non-fast-forward, or any required fact is ambiguous,
STOP. Force push is prohibited. This mode must never be used to perform a
protected/direct-real-execution operation that requires the direct Windows
PowerShell runbook; those stricter rules remain in force.

## 6. Methodology, authority, and STOP discipline

Execution agents never infer missing research choices or broaden authority.
If the Issue or repository leaves a materially methodological choice
unspecified, return `CHATGPT_DECISION_REQUIRED` and stop before implementing
that choice.

GitHub connectivity, file access, or tool capability never implies authority
for network research, private or sealed data, ticker selection, historical
screening, model fitting, backtesting, forward paper, production, or trading.
Human point-of-use gates and frozen designs remain authoritative.

An execution agent must stop for an expected-HEAD mismatch, dirty tree,
unexpected file/history, frozen-blob mismatch, missing artifact or authority,
scope conflict, or methodology ambiguity. It must not repair the blocker by
guessing, broadening scope, or changing methodology.

## 7. Review lifecycle

The execution agent reports the exact commit SHA, parent, changed-file scope,
tests, push result, remote HEAD, clean-tree status, and boundary counts
required by the Issue. It does not self-declare the final review PASS.

GPT independently verifies the authoritative remote HEAD, exact commit and
diff, changed files, relevant complete source/artifacts, frozen provenance,
governance, targeted tests, and available CI. GPT's review targets one exact
40-hex SHA and requires `CRITICAL=0`, `HIGH=0`, and `MEDIUM=0` for PASS. GPT
posts the review to the GitHub Issue and closes the Issue only after PASS.

On BLOCK, keep the originating task traceable. Normally create one bounded
remediation Issue per finding, at most two only when findings are strongly
coupled. Each remediation Issue gets its own exact `EXPECTED_HEAD` after the
prior commit, and the execution/review loop repeats.

## 8. Cross-chat bootstrap

When a new ChatGPT chat continues stock-analyzer and connected GitHub is
available:

1. read `AGENTS.md`, `AI_RESEARCH_EXECUTION_RULES.md`,
   `AI_STOCK_ANALYZER_REVIEW_POLICY.md`, `PROJECT_STATE.md`, and this
   workflow;
2. verify the authoritative branch and remote HEAD;
3. inspect relevant open GitHub Issues;
4. distinguish `READY` from `QUEUED` and `BLOCKED` Issues; and
5. independently review the current HEAD when repository state says it is
   awaiting GPT review.

Do not ask the human to re-enter task state recoverable from the repository.
Repository and frozen artifacts override Issue text if they conflict;
stricter authority wins.

## 9. Prompt minimization

Once this workflow is active, normal executor dispatch should contain only:

- mode and executor class;
- repository;
- GitHub Issue number;
- instruction to read and exactly execute the `READY` Issue;
- STOP conditions for mismatch and methodology ambiguity; and
- the required report convention.

Stable governance and frozen methodology belong in version-controlled
repository documents, not in every dispatcher prompt. A short prompt must
not create ambiguity; missing rules must be stated explicitly or added to
the canonical repository document before being relied upon.

## 10. Activation checks and precedence

When this workflow is activated for a task, perform the exact branch, HEAD,
remote, clean-tree, and scope preflight; change only the Issue-planned files;
resolve internal governance-document references; confirm that no
methodology or authority semantics changed; and run `git diff --check`.
Documentation-only governance tasks do not require full pytest, regression,
backtest, or research-data access unless their Issue explicitly requires it.

`AGENTS.md`, `AI_RESEARCH_EXECUTION_RULES.md`,
`AI_REAL_EXECUTION_RUNBOOK.md`, applicable frozen designs, human gates, and
stricter task-specific rules override this workflow. This document changes
only cross-chat task dispatch and review cadence; it creates no authority and
is not a methodology or review-authority document.

## 11. Required final report

When an Issue specifies this workflow's governance task, report:

```text
MODEL=
MODE=
STARTING_HEAD=
TASK=GITHUB_ISSUE_ORCHESTRATION_WORKFLOW_GOVERNANCE
GITHUB_ISSUE=
CHANGED_FILES=
STATIC_CHECK=
METHODOLOGY_CHANGED=false
EXECUTION_AUTHORITY_CHANGED=false
FROZEN_DESIGN_CHANGED=false
COMMIT_SHA=
PARENT_SHA=
PUSH_RESULT=
REMOTE_HEAD=
WORKING_TREE_CLEAN=
NEXT_ACTION=GPT_EXACT_SHA_INDEPENDENT_REVIEW
```

GPT-5.6 Sol remains the methodology authority and final independent review
authority.
