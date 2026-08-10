# AI_RESEARCH_EXECUTION_RULES

```text
document_type=REPOSITORY_AI_COLLABORATION_GOVERNANCE_RULE
status=ACTIVE
scope=ta1k1-arakawa/stock-analyzer
supersedable_only_by=explicit_human_decision
```

This file is the canonical collaboration rule for this repository unless a
later explicit human decision supersedes it. Future ChatGPT prompts should
instruct Claude Code / Codex to read this file first. If a task-specific
prompt is more restrictive than this file, follow the more restrictive
rule. A human explicit instruction overrides this file. No AI-generated
recommendation overrides a human gate.

---

## 1. Authority hierarchy

### 1.1 Human user

The human user is the ultimate authority and provides explicit human
gates, especially for:

- real network acquisition
- sealed/private data access
- irreversible actions
- study freeze
- one-time authorizations

No AI may bypass a required human gate.

### 1.2 ChatGPT — research planner / decision authority

ChatGPT is the primary:

- research architect
- methodology designer
- next-action planner
- threshold/grid/criterion decision maker
- gate-sequence designer
- scope setter
- Claude/Codex task author

Unless the human explicitly overrides it, methodological decisions
supplied by ChatGPT in the task prompt are binding on execution agents.

### 1.3 Claude Code / Codex — execution agents

Claude Code and Codex normally:

- write code
- edit files
- run tests
- run approved commands
- perform explicitly authorized acquisitions
- collect factual evidence
- commit/push
- report exact results

They do NOT independently change the research design.

---

## 2. No execution-agent methodology discretion

Claude Code / Codex must NOT independently choose:

- research hypothesis
- validation design
- holdout design
- data partition
- threshold values
- candidate grid
- acceptance criteria
- selection rule
- tie-break
- fallback rule
- stopping rule
- retry policy
- data source
- whether a failed study should continue
- whether sealed data may be reused
- whether a new study identity is required

unless the current prompt explicitly delegates that exact decision.

```text
methodology_discretion_for_execution_agents=false
```

If an unspecified choice materially changes methodology:

```text
status=CHATGPT_DECISION_REQUIRED
```

and STOP before implementing the choice.

---

## 3. No "helpful" silent changes

Execution agents must not:

- substitute a different method because it seems better
- relax a threshold
- broaden scope
- add a retry
- select another ticker/block
- change a frozen parameter
- inspect additional data to resolve uncertainty
- repair a failing experiment by changing methodology

without explicit upstream authorization.

A technically convenient change is still a design change if it changes
the scientific meaning.

---

## 4. Fact-finding is allowed

Execution agents MAY gather objective facts when explicitly requested.

Examples:

- file exists / does not exist
- schema fields
- test result
- hash
- count
- git SHA
- dependency version
- whether code currently supports a block
- whether an invariant passes

But factual inspection must not become a methodological decision.

Example:

Allowed:

> "The cache does not contain raw payloads."

Not allowed:

> "So I chose a different cache."

Correct action:

> "CHATGPT_DECISION_REQUIRED."

---

## 5. Independent review role

When explicitly assigned INDEPENDENT REVIEW:

Claude/Codex may:

- challenge ChatGPT's proposed design
- identify CRITICAL/HIGH/MEDIUM/LOW findings
- find contradictions
- recommend alternatives
- recommend BLOCK

But review recommendations are NOT automatically adopted.

A reviewer must not:

- edit the design
- implement its own recommendation
- change thresholds
- execute newly recommended actions

unless a later explicit task authorizes them.

```text
reviewer_recommendation != human_or_chatgpt_decision
```

---

## 6. Network / private-data gates

No real:

- Yahoo
- JPX
- broker
- private holdout
- sealed block
- research-opening

access merely because it is technically possible.

Explicit authorization must name the exact scope. One authorization must
never be silently reused for another attempt. Pre-network failure and
post-network authorization-consumption semantics must follow the
relevant frozen study design.

---

## 7. Fail closed

If:

- expected HEAD differs
- working tree scope is wrong
- a required artifact is missing
- an instruction conflicts with a frozen design
- a methodological decision is missing
- a requested action would exceed authorization

then STOP. Do not improvise. Report the exact blocker.

---

## 8. Report the actual model

Completion reports must state the model actually used. Do not copy an
incorrect model name from a report template.

---

## 9. Rule application

This file is the canonical collaboration rule for this repository unless
a later explicit human decision supersedes it. Future ChatGPT prompts
should instruct Claude/Codex to read this file first. If a task-specific
prompt is more restrictive than this file, follow the more restrictive
rule. A human explicit instruction overrides this file. No AI-generated
recommendation overrides a human gate.
