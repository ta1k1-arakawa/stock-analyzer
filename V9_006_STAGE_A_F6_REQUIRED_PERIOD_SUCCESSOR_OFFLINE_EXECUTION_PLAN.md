# V9_006 F6 required-period successor offline execution plan

```text
status=PREP_AWAITING_GPT_REVIEW
execution_class=DETERMINISTIC_REPROCESSING_OF_ALREADY_LOCKED_PUBLIC_PAYLOAD
reviewed_implementation_sha=d065cefc45ac3330ccba99da2b88665767ea0351
successor_source_blob=4b9c06a7952b48ace7a0949cea950a491a7e6cce
successor_cli_blob=606d67f113565051520fdfb7973e97d96f8c2407
locked_f6_child_sha256=060d74a7f5a3b413d351de05ed07f412d093a3ebf41f6ea3d4e0de3f313b4b0c
locked_f6_child_byte_length=36352
expected_structural_profile_sha256=4332d0b27a1e35256abef4c0e240b2c576c20122a264374ea0c5da3729beacce
date_column_ordinals=[4,6]
required_years=2017..2025
network_requests_allowed=0
refetch_allowed=false
provider_substitution_allowed=false
new_human_gate_consumption_required=false
private_or_sealed_access=false
```

## Execution boundary

The later execution is bound only to the reviewed successor CLI and the exact
already-locked F6 CHILD above. It must use the repository
`.venv\Scripts\python.exe` interpreter, re-run the existing metadata and
content-blind integrity checks, and read no other object. It must make no
HTTP or other network request, reacquire nothing, replace nothing, and inspect
no provider, object, or path as a fallback.

Actual runtime path values are neither committed nor unnecessarily printed.
GPT will supply a single atomic PowerShell runtime-resolution/preflight block
only after this preparation commit receives exact-SHA PASS. This plan neither
contains nor authorizes that block or a production runner invocation.

## Result handling and safe output

Record whichever valid safe successor result occurs without tuning or rerun:
`SUCCESSOR_REQUIRED_PERIOD_COVERAGE_CAPTURED` and
`SUCCESSOR_REQUIRED_PERIOD_COVERAGE_PARTIAL` are both recordable outcomes.
Inherited `CHATGPT_DECISION_REQUIRED` or `IMPLEMENTATION_FAILURE` stops for
GPT adjudication. No automatic rerun or refetch is permitted under any
result.

Safe output remains limited to the reviewed successor schema. It must never
contain raw bytes, a local raw path, payload values, full dates, URLs, or
exception text. `out_of_scope_disagreement` remains a diagnostic only and
cannot alter coverage acceptance.

## Scientific interpretation and retained prohibitions

This future deterministic execution produces DEVELOPMENT_EVIDENCE only. The
historical old-rule `F6_YEAR_COVERAGE_AMBIGUOUS` result remains immutable; a
successor result does not retroactively preregister the amended rule.
Confirmation debt remains true. This plan creates no authority for promotion,
model fitting, historical backtest, private or sealed access, live trading, or
profitability claims; future profitability remains unestablished.

## GPT final implementation review record

```text
REVIEWED_SHA=d065cefc45ac3330ccba99da2b88665767ea0351
PARENT_SHA=2ba026bd67e4cd7bb1e28f660cccbace4bb28129
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_F6_REQUIRED_PERIOD_SUCCESSOR_PARSER_DESIGN=PASS
V9_006_F6_REQUIRED_PERIOD_SUCCESSOR_PARSER_IMPLEMENTATION=PASS
TARGETED_TESTS=18/18_PASS
OLD_PARSER_MODIFIED=false
```

This preparation does not execute the parser, read the raw CHILD, read any
DATE/year value, make a network request, or consume a human gate.
