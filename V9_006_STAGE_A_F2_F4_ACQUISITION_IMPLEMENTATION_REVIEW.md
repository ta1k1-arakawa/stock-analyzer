# V9_006 Stage-A F2/F4 single-slot acquisition implementation review

```text
task=V9_006_STAGE_A_F2_F4_SINGLE_SLOT_ACQUISITION_FOUNDATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

`acquire_f2_f4_monthly_evidence` is a low-level single-slot seam only. It
locks the F2-owned shared Monthly Statistics root as
`MONTHLY_STATISTICS_DISCOVERY_ROOT`, resolves and locks the F2-owned selected
year support page as `MONTHLY_STATISTICS_DISCOVERY_YEAR_YYYY`, then locks the
unique requested F2/F4 child under its own family and exact `YYYY-MM` period.
It returns only the child raw-lock slot ID and total attempts. Existing locks
are reused; corrupt/orphan support locks and traversal failure fail closed.

No base-month enumeration, bridge enumeration, content parsing, matrix
mutation, production `run_stage_a` integration, or real network execution is
implemented.

The traversal base for each locked support object is that object's recorded
`resolved_url`: root HTML resolves its selected year link against the locked
root final URL, and year-page HTML resolves its child link against the locked
year-page final URL. Raw-lock identities remain keyed by requested URLs, and
the returned child slot ID remains keyed by the mechanically resolved child
requested URL.

```text
REVIEWED_SHA=770321aadfc696de515a7576534da4fdc0086a4f
PARENT_SHA=4b9c082bde0f7ace7bd4867b621f2e505b8d18de
CRITICAL=0
HIGH=0
MEDIUM=1
LOW=1
RESULT=BLOCK
FINDING=V9_006_F2_F4_ACQ_MEDIUM_1_RELATIVE_LINK_BASE_IGNORES_RESOLVED_URL
```

```text
REVIEWED_SHA=4b9c082bde0f7ace7bd4867b621f2e505b8d18de
PARENT_SHA=04455334511f49ec8f8029d2a07022d78d8b758f
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F2_F4_SHARED_SUPPORT_OBJECT_METHODOLOGY=PASS
V9_006_STAGE_A_F2_F4_TRAVERSAL=PASS
V9_006_STAGE_A_ACQUISITION_OBJECT_MODEL_IMPLEMENTATION=PASS
```
