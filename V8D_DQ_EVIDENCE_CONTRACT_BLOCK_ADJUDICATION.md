# V8D DQ Evidence Contract Block Adjudication

This document records the terminal V8D design-auditability disposition using
privacy-safe facts and public Git provenance only. It records no ticker
identity, private path, raw payload, price, private manifest content, or human
authorization identity.

## Terminal record

```text
study=V8D_HISTORICAL_RESEARCH
terminal_status=BLOCK_CLOSED
failure_class=DESIGN_AUDITABILITY_FAILURE
terminal_implementation_head=a862efec34dcf4a89005c88b55b35c39be12b7bc
```

The attempted synthetic E2E work made no real Yahoo requests, performed no
private production reads, accessed no ticker identities, consumed no real
human gates, performed no real acquisition, and opened no research data.
No T1C/T2 outcomes or features were observed. Strategy profitability was not
evaluated, and future profitability is unestablished.

## Exact auditability contradiction

The frozen `DATA_QUALITY_GATE_FAILURE` evidence contract contains only these
fields:

```text
nonempty_timestamp
valid_price_row_count
trading_date_fields_valid
```

All three fields can be valid while the independently frozen acquisition data
quality policy fails because either:

```text
invalid_fraction > 1/252
max_consecutive_invalid_returned_rows > 1
```

Those three recorded fields do not contain the invalid-row counts, returned-row
denominator, invalid fraction, or consecutive-invalid-run evidence needed for
an independent verifier to re-derive either threshold failure. The verifier
therefore cannot establish the threshold failure without either trusting the
producer-declared failure or recording additional threshold evidence. The
first option weakens independent verification; the second changes the frozen
semantic audit contract. Neither change was authorized after the V8D design
freeze.

This is a design-auditability defect, not a strategy failure, profitability
finding, or T1C/T2 data-quality result.

## Frozen-boundary disposition

The following actions are explicitly not taken inside V8D:

- Do not falsify one of the three existing evidence fields merely to make the
  verifier pass.
- Do not weaken the verifier to trust `named_condition` or another
  producer-declared failure.
- Do not silently extend the frozen evidence schema inside V8D.
- Do not change retry, redraw, provider, threshold, or other methodology
  semantics.

V8D is therefore terminally closed as `BLOCK_CLOSED`. No V8D producer or
verifier semantics, frozen design, network boundary, private-data boundary,
human-gate state, or research state is changed by this adjudication.

## Successor recommendation

`V8E_HISTORICAL_RESEARCH` is required as a successor study. V8E should
inherit all unchanged V8D/V8C methodology and fix only the privacy-safe,
independently re-derivable data-quality threshold evidence contract. The V8E
design is not created and no network or private-data execution is performed
in this adjudication.
