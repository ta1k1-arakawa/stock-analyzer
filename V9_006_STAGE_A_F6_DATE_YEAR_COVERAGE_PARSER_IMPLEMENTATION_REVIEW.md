# V9_006 F6 date/year coverage parser implementation review

```text
implementation_status=CANDIDATE_AWAITING_GPT_REVIEW
scope=OFFLINE_SYNTHETIC_TESTED_ONLY
network_requests=0
production_child_reads=0
coverage_executed=false
```

Implements the reviewed date/year design as a separate offline module. The
production entry point preserves the inherited Phase A/B binding, recomputes
the existing reviewed structural evidence, and permits DATE-value reads only
after the exact canonical structural-profile hash gate. It uses only the two
frozen columns, produces no alternate ambiguity-resolution mode, validates a
closed safe schema, and emits deterministic JSON from the CLI. This candidate
does not self-call PASS; GPT-5.6 Sol remains final review authority.

## GPT implementation review and remediation

```text
REVIEWED_SHA=8dc08d2d989b01e522d976bc5acc5ee9ef4917c0
PARENT_SHA=56ad61ee6ebd8bdac79414830cf2e59eea60a766
CRITICAL=0
HIGH=0
MEDIUM=2
LOW=0
RESULT=BLOCK
MEDIUM_1=V9_006_F6_DATE_YEAR_IMPL_INHERITED_PHASE_A_CHATGPT_DECISION_REQUIRED_COLLAPSED
MEDIUM_2=V9_006_F6_DATE_YEAR_IMPL_SAFE_VALIDATOR_FAILURE_PROVENANCE_FALSE_NEGATIVE
```

The remediation preserves inherited `ProbeBlocked` status and both Phase A/B
provenance values exactly. It also supplies runtime-known provenance to a
safe-validator rejection, so a matched structural hash, DATE read, completed
comparison, and raw/content boundaries are never reset to a false pre-hash
state. The validator now rejects expected-SHA plus unverified status, closing
the structural-hash three-state invariant. This remains
`BLOCK`/`REMEDIATED_AWAITING_GPT_REVIEW`, not self-called PASS.
