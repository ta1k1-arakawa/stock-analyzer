# V9_006 Stage-A F2/F4 required-slot enumeration implementation review

```text
task=V9_006_STAGE_A_F2_F4_REQUIRED_SLOT_ENUMERATION_FOUNDATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

`acquire_f2_f4_required_slots` deterministically invokes only the reviewed
single-slot F2/F4 seam: ascending 2017-01 through 2025-12, F2 then F4 for
each month, then ascending F2-only bridge months from `f2_bridge_months`.
It returns separate base coverage references and bridge references, plus the
aggregate attempted-fetch count. Before returning, it requires the exact
216-key base set and independently verified child locks with matching family
and exact applicable period. Support root/year locks cannot satisfy either
reference domain.

No content parsing, production `run_stage_a` integration, F1/F3/F5/F6/F7
acquisition, real network execution, or acquisition-readiness change is
implemented.

```text
REVIEWED_SHA=ae62c1dd1e5aa7753a03a765fc40dcfb6e7adc6f
PARENT_SHA=770321aadfc696de515a7576534da4fdc0086a4f
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_F2_F4_ACQ_MEDIUM_1=RESOLVED
V9_006_STAGE_A_F2_F4_ACQUISITION=PASS
```
