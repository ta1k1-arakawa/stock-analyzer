# V9_006 Stage-A F3 YEAR acquisition implementation review

```text
task=V9_006_STAGE_A_F3_YEAR_ACQUISITION_AND_FANOUT_IMPLEMENTATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

`resolve_delisted_company_year_url` uses the existing strict locked-HTML
anchor parser and existing JPX URL validation to resolve exactly one F3
archive-year link. It resolves relative links against the supplied locked
root final URL and fails closed on malformed anchors, ambiguity, missing
years, or unsafe URLs.

`acquire_f3_required_slots` locks/reuses the F3-owned discovery root, then
locks exactly nine selected YEAR evidence objects for 2017--2025. It fans each
verified YEAR raw-lock key to precisely that year's twelve F3 base cells and
requires the exact 108-cell set, one same ID per year, nine distinct year IDs,
and matching F3/year metadata before returning.

No delisting-event parsing, conflict-semantics change, `run_stage_a`
integration, other-family acquisition, real network execution, or readiness
flag change is implemented.

```text
REVIEWED_SHA=3b55ca8f34b6a2d9ccc565ad1cea25228a363e0f
PARENT_SHA=edb71bb969f694b29e63c395ad16bae65d7311f1
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F3_YEAR_COVERAGE_METHODOLOGY=PASS
```
