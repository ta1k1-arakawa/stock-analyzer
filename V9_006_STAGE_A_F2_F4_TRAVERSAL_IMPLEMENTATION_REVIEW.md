# V9_006 Stage-A F2/F4 traversal implementation review

```text
task=V9_006_STAGE_A_F2_F4_MONTHLY_STATISTICS_TRAVERSAL_FOUNDATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

Pure stdlib HTML traversal helpers now resolve a unique official archive-year
link from locked Monthly Statistics root bytes, then resolve a unique linked
evidence object from the exact F2 or F4 semantic row and exact requested
`YYYY-MM` header column in locked year-page bytes. Relative links are resolved
against the supplied locked-page URL and validated as JPX HTTPS URLs. Ambiguous,
missing, malformed, unsafe, unsupported-family, and year-mismatch input fails
closed. These helpers do not fetch, write raw locks, create coverage references,
parse F2 events/F4 ratios, or integrate acquisition.

```text
REVIEWED_SHA=302c4abe46df9bf5be80ef0bb6df809d2426bb1b
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_ACQ_OBJECT_IMPL_MEDIUM_1=RESOLVED
V9_006_STAGE_A_ACQUISITION_OBJECT_MODEL_IMPLEMENTATION=PASS
```

## MEDIUM_1 review and remediation

```text
REVIEWED_SHA=e03b959b149852c50a17576a204592d2a3ddb51f
PARENT_SHA=302c4abe46df9bf5be80ef0bb6df809d2426bb1b
CRITICAL=0
HIGH=0
MEDIUM=1
LOW=1
RESULT=BLOCK
FINDING=V9_006_F2_F4_TRAVERSAL_MEDIUM_1_MALFORMED_HTML_NOT_TOTAL_FAIL_CLOSED
```

Traversal-relevant `table`, `tr`, `th`, `td`, and `a` tags now use one strict
matching stack. Orphan or mismatched closes, invalid parent nesting, nested
cell/anchor tags, premature table closure, and unclosed tags fail closed as
`IMPLEMENTATION_FAILURE`; unrelated ordinary tags remain non-structural.

`V9_006_STAGE_A_F2_F4_TRAVERSAL=REMEDIATION_IMPLEMENTED_AWAITING_GPT_REVIEW`.
