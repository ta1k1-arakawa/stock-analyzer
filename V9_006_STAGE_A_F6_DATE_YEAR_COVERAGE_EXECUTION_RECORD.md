# V9_006 F6 date/year coverage production execution record

```text
REVIEWED_IMPLEMENTATION_SHA=89e7fbbea7c24a7cc4749da97fa9b8c1bb5f19c5
PRE_EXECUTION_PROVENANCE=PASS
PROTECTED_ENVIRONMENT_PREFLIGHT=PASS
REAL_EXECUTION_ENVIRONMENT_FROZEN=true
CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE=YES
COVERAGE_EXIT_CODE=0
EXECUTION_RESULT=COMPLETE
STATUS=F6_YEAR_COVERAGE_AMBIGUOUS
NETWORK_REQUESTS=0
RAW_BYTES_READ_FOR_INTEGRITY=true
CHILD_CONTENT_INSPECTED=true
STRUCTURAL_PROFILE_SHA256=4332d0b27a1e35256abef4c0e240b2c576c20122a264374ea0c5da3729beacce
STRUCTURAL_PROFILE_HASH_VERIFIED=true
DATE_COLUMN_ORDINALS=[4,6]
DATE_YEAR_VALUE_READ=true
COVERAGE_EVALUATED=true
COVERAGE_RESULT_ACCEPTED=false
```

```text
YEAR_HISTOGRAM_COLUMN_4=2007:1,2008:1,2009:1,2010:1,2011:1,2012:1,2014:2,2015:1,2016:1,2017:1,2018:1,2019:1,2020:1,2021:1,2022:1,2023:1,2024:1,2025:1
YEAR_HISTOGRAM_COLUMN_6=2007:1,2008:1,2009:1,2010:1,2011:1,2012:1,2013:1,2014:1,2015:1,2016:1,2017:1,2018:1,2019:1,2020:1,2021:1,2022:1,2023:1,2024:1,2025:1
HISTOGRAM_COUNT_COLUMN_4=19
HISTOGRAM_COUNT_COLUMN_6=19
```

Mechanical difference: column 4 has no 2013 entry and has `2014:2`; column
6 has `2013:1` and `2014:1`. No covered-year, required-year, missing-year,
or all-years-covered field is emitted for this AMBIGUOUS result.

This production parser execution is COMPLETE/PASS and the safe evidence is
valid terminal `F6_YEAR_COVERAGE_AMBIGUOUS` evidence. It records evaluation,
not acceptance. No union, intersection, column preference, required-years-
only comparison, rerun, refetch, reselection, or provider substitution is
permitted within this current preregistered V9 study. A materially different
rule requires a successor study/new preregistration. This is neither a
strategy nor profitability failure; `future_profitability_established=false`.
The acquisition gate remains consumed/nonreusable; this run consumed zero
human gates and made zero network requests.
