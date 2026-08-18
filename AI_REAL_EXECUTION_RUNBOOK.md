# AI Real Execution Runbook

```text
document_type=REPOSITORY_REAL_EXECUTION_SAFETY_RUNBOOK
status=ACTIVE
scope=REAL_NETWORK_PRIVATE_DATA_HUMAN_GATE_DURABLE_STATE
```

This runbook is mandatory whenever a task can reach any of the following:

- real Yahoo, JPX, broker, or other external production network;
- private or sealed data, identities, or inputs;
- human-gated one-shot execution;
- durable machine-local production, gate, or audit state;
- raw acquisition or research opening; or
- direct Windows PowerShell prepared for any of the above.

It applies to the task author, ChatGPT, GPT-5.6 Sol, Codex, Claude Code
Cloud, and reviewers when they prepare, inspect, authorize, execute, or
report an in-scope operation.

## Authority and boundary

The frozen task or study design remains authoritative for methodology,
including provider, sentinel, windows, retry policy, thresholds, partitions,
sample rules, stopping rules, and gate semantics. The human gate remains the
ultimate authority. A task-specific stricter rule wins. This runbook supplies
stable operational safety only; it must never silently relax, reinterpret, or
replace a frozen study design. If a repair would change methodology or the
execution contract, stop and obtain a successor-study or other explicit
methodology decision from the GPT methodology authority and human authority.

Reusable real-execution safety belongs in this version-controlled runbook,
not in conversational memory or repeated long prompts. Repository-first and
delta-only prompt rules remain in `AI_RESEARCH_EXECUTION_RULES.md`.

## 1. Atomic PowerShell execution

A fail-closed real-execution command supplied to a user MUST be one atomic
PowerShell script block, normally `& { ... }`, or a pre-created reviewed
`.ps1` file. Do not provide independent lines where a `throw` can fail while
the user can continue pasting later commands.

The atomic scope MUST set the stop mode inside the scope:

```powershell
& {
    $ErrorActionPreference = "Stop"
    # Read-only provenance and state preflight.
    # Stop before the private/network/gate boundary unless every check passes.
    # Consume the one-shot gate only at the frozen boundary.
    try {
        # Authorized operation.
    }
    finally {
        # Clear any transient private environment values.
    }
}
```

Every preflight failure must terminate the whole block before any private,
network, or gate boundary. If an interactive continuation prompt `>>` is
present unexpectedly, instruct the user to press Ctrl+C before any further
execution. Long or one-shot execution must not depend on an AI-agent session
remaining alive.

## 2. Mandatory preflight before a boundary

Before any irreversible, private, network, or gate boundary, mechanically
verify every item applicable to the frozen design:

- correct repository;
- authoritative branch;
- exact expected local HEAD;
- exact authoritative remote HEAD;
- clean working tree;
- required commit, blob, and artifact bindings;
- frozen-design binding;
- reviewed-implementation binding;
- human-authority prerequisites;
- one-shot gate not already consumed;
- no conflicting durable execution binding, audit, or receipt;
- required private input exists and resolves uniquely.

Only after all required checks PASS may execution continue. A missing,
ambiguous, malformed, mismatched, stale, or unverifiable prerequisite is a
STOP condition. The preflight must not print private values while proving
these predicates.

## 3. Stale local checkout recovery

Never interpret a merely stale local checkout as a research or study failure.
First fetch `origin`. If, and only if, the authoritative remote HEAD equals
the exact expected SHA, the working tree is clean, the branch is correct, and
the current local HEAD is an ancestor of that expected SHA, a sync-only
pre-execution operation is allowed:

```powershell
git reset --hard <exact-expected-SHA>
```

This creates no methodology authority and consumes no gate. If ancestry,
branch, remote HEAD, cleanliness, or history is not exact, stop with
`EXPECTED_HEAD_MISMATCH` or `LOCAL_HISTORY_MISMATCH`, as applicable. Never
use `git pull`, merge, rebase, cherry-pick, or force as recovery. Re-run the
entire preflight from the beginning after an allowed sync.

## 4. Pre-gate versus post-gate failure

Classify the boundary before deciding authorization status or recovery. First
perform read-only state verification.

`PRE_GATE_FAILURE` means the one-shot gate was not consumed, the applicable
private or network boundary was not crossed, and real request count is zero
where that count is provable. Do not automatically declare a still-unused
authorization consumed. Only non-methodological, safe preflight repair may
be proposed; then run preflight again from the beginning.

`POST_GATE_FAILURE` means a durable one-shot gate was consumed. A crash or
BLOCK does not restore authorization. Do not retry, reset, delete a receipt,
generate a fresh nonce to bypass the gate, or reinterpret the same human
authorization. Follow the frozen stopping rule and return to the GPT
methodology authority and human authority for the next decision.

When the exact boundary cannot be proven, fail closed and do not assume
`PRE_GATE_FAILURE`.

## 5. Private path and file discovery

Never guess a private path. Never ask the user to paste a private path into
public or chat output when avoidable. Before the applicable authorized
content boundary, discovery is metadata-only.

Resolve candidate paths privately and count them without printing identities
or paths:

- zero candidates: STOP;
- exactly one candidate: it may proceed after all other checks PASS;
- more than one candidate: STOP for ambiguity.

Do not print ticker identities, private paths, raw payloads, prices,
features, outcomes, or raw authorization identity. Content reads require the
applicable authorization boundary. Safe evidence may expose only approved
hashes, counts, booleans, and safe aggregates.

## 6. PowerShell automatic variables

PowerShell variable names are case-insensitive. Do not use ordinary task
variables whose names collide with automatic variables. At minimum prohibit:

```text
Matches / matches
Error / error
Args / args
Input / input
Host / host
PID / pid
HOME / home
```

Prefer descriptive names such as `regexMatch`, `candidatePaths`,
`executionError`, and `privateManifestPath`. Apply this rule to every
variable in a prepared script, including variables in `catch`, `finally`,
and helper functions.

## 7. Execution environment

For real network, private-data, human-gated, or durable-machine-state
execution, default to direct Windows PowerShell. Claude Code Cloud and
Codex may prepare and statically inspect the command, but execution must not
depend on the lifetime of an AI-agent session. Long or one-shot work must be
prepared so it can continue on the Windows side after an SSH or agent
connection ends, when the frozen design permits that topology.

Local Claude Code remains prohibited by repository governance. Normal local
Codex editing and short validation are separate from direct execution and do
not authorize a private or network boundary.

## 8. Durable-state collision check

Before consuming a gate, read-only check every relevant existing durable
state location, including:

- gate receipt;
- production execution binding;
- audit aggregate or dossier where a deterministic conflict matters;
- verification receipt; and
- stage-specific durable state.

If existing state indicates a prior attempt or execution, STOP. Never delete,
reset, overwrite, or otherwise alter that state to obtain another attempt.
Durable publication must remain exclusive and no-overwrite when required by
the frozen design.

## 9. Transient secrets and private values

Raw human authorization identities and private paths must not be printed.
If a private value is temporarily placed in an environment variable, clear it
in a `finally` block even when the operation fails. Prefer public or durable
safe evidence consisting of SHA-256 values, counts, booleans, or safe
aggregates. Private ticker identities, raw URLs, and raw payloads must not
leak into logs, reports, exceptions, or receipts.

## 10. Frozen failure discipline

After a real BLOCK, the execution agent or ChatGPT must not, within the same
frozen study, silently:

- change provider;
- change a sentinel;
- change a readiness or acquisition window;
- add or change retry behavior;
- change a threshold;
- substitute another ticker or block;
- redraw a partition;
- alter sample inclusion or exclusion; or
- relax validation.

Follow the frozen stopping rule. If repair changes methodology or the
contract and was not explicitly pre-authorized, require a successor-study
decision. Operational recovery is not permission to retry a consumed
experiment.

## 11. Safe execution report contract

For real execution, report only safe values, where applicable, using this
contract:

```text
PRE_GATE_STATUS=<PASS|FAIL|UNKNOWN>
LOCAL_HEAD=<safe exact SHA or omitted when unavailable>
REMOTE_HEAD=<safe exact SHA or omitted when unavailable>
WORKING_TREE_CLEAN=<true|false|unknown>
GATE_CONSUMED=<true|false|unknown>
PRIVATE_READS=<nonnegative count|unknown>
NETWORK_REQUESTS=<nonnegative count|unknown>
YAHOO_REQUESTS=<nonnegative count|unknown>
JPX_REQUESTS=<nonnegative count|unknown>
EXECUTION_RESULT=<safe enum>
FAILURE_CLASS=<safe enum or NONE>
DURABLE_EVIDENCE=<hashes/counts/booleans/safe aggregates only>
AUTHORIZATION_REUSABLE=<true|false|unknown>
SECOND_EXECUTION_ALLOWED=<true|false|unknown>
RAW_ACQUISITION_ALLOWED=<true|false|unknown>
RESEARCH_OPENING_ALLOWED=<true|false|unknown>
NEXT_ACTION=<safe stopping or authority action>
```

Never include prohibited private values in a report, including private paths,
ticker identities, raw URLs, raw payloads, prices, features, outcomes, or raw
authorization identity.

## 12. Recovery state machine

The generic recovery flow is:

```text
STOP
  -> determine whether private/network/gate boundary was reached
  -> read-only inspect durable state
  -> classify PRE_GATE or POST_GATE
  -> PRE_GATE: propose only non-methodological safe preflight repair
  -> PRE_GATE: re-run preflight from the beginning
  -> POST_GATE: no second execution unless frozen design explicitly authorizes it
  -> return to GPT methodology authority and human authority for next study decision
```

If a state transition is not provable, STOP. A POST_GATE crash, BLOCK, or
partial result never becomes a PRE_GATE failure through cleanup.

## 13. Command quality checklist

Before giving a real-execution PowerShell block, ChatGPT or the execution
agent preparing it must statically check:

- one atomic `& { }` scope or one reviewed `.ps1`;
- `$ErrorActionPreference = "Stop"` inside that scope;
- no automatic-variable name collisions;
- no path continues after a failed preflight;
- no private or network operation before required preflight;
- gate-consumption boundary matches the frozen design;
- no raw secrets are printed;
- `finally` cleanup exists for transient private environment values;
- second-execution behavior is explicit;
- failure output is privacy-safe; and
- no methodology choice is hidden in recovery logic.

## 14. Generic incident rationale

Stale local Git state must be repaired before interpreting a study outcome.
Shell error handling must be atomic because interactive line-by-line
execution can continue after a thrown error. PowerShell's case-insensitive
automatic variables can corrupt ordinary state if reused. One-shot gates
require explicit pre- and post-boundary classification. These are
operational safety rules, not permission to retry a consumed experiment.

This rationale is generic and does not establish any study-specific root
cause or convert a hypothesis into a confirmed fact.
