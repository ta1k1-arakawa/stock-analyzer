# V9_006 Stage-A F7 calendar coverage acquisition implementation review

```text
task=V9_006_STAGE_A_F7_CALENDAR_COVERAGE_ACQUISITION_FOUNDATION
status=IMPLEMENTED_AWAITING_GPT_REVIEW
network_executed=false
ACQUISITION_IMPLEMENTATION_COMPLETE=false
```

`acquire_f7_required_slots` iterates only the existing ascending
`calendar_envelope_months()` and derives every requested URL only through
`resolve_f7_calendar_url`. It locks each F7 monthly page under its exact
month, returning separate 108-cell base and seven-month envelope-extra
reference mappings plus total attempts. Before returning, it independently
requires exact domain sets and verified F7 family/month/requested-key metadata.

No discovery support object, calendar content parsing, trading-day derivation,
`FINAL_SIGNAL_D0` work, `run_stage_a` integration, or readiness change is
implemented. Same-domain redirects retain requested-URL raw-lock identity.

```text
REVIEWED_SHA=7682fe67d20f7ad8028df3fca82d82f85b686bc3
PARENT_SHA=a9168df38b793525a56aef60699e0ece8e804c7e
CRITICAL=0
HIGH=0
MEDIUM=0
LOW=1
RESULT=PASS
V9_006_STAGE_A_F5_COVERAGE_COMPARABILITY_METHODOLOGY=PASS
```
