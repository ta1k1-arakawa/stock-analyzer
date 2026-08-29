# V9_006 Stage-A completion audit

```text
audit_basis=ACTUAL_CURRENT_CODE_AND_TARGETED_TEST_SURFACES
reviewed_execution_checkpoint_sha=923bb4e7f79156aea59d3e6b294ebc0e108357b5
stage_a_executed=false
acquisition_implementation_complete=false
```

## Mechanical finding

`src/v9_005_stage_a_jpx_probe.py` has reviewed locator/inventory contracts,
raw-lock helpers, evidence construction, and `run_stage_a()`, but its
`verify_acquisition_implementation_ready()` deliberately blocks before any
fetch or output creation while `ACQUISITION_IMPLEMENTATION_COMPLETE=false`.
Accordingly, reviewed family seams are not production Stage-A integration.

## Family audit

Legend: `C` complete/reviewed; `I` mechanical integration remaining; `R`
implementation remaining under existing methodology; `N` not required; `M`
methodology decision required. Columns are A locator, B acquisition/traversal,
C enumeration, D raw provenance, E transformation, F matrix/fanout, G semantic
adapter, H production integration, I tests, J methodology.

| Family | A | B | C | D | E | F | G | H | I | J | Current code / minimum test surface |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 TERMINAL | C | I | N | C | R | N | R | I | I | N | `run_stage_a`, terminal lock/reconstruction; synthetic terminal parse-to-semantic input |
| F2 MONTHLY | C | R | R | R | R | R | R | I | R | N | `LOCATOR_STRATEGIES`, `build_source_inventory`, `f2_bridge_months`; root traversal/bridge fanout |
| F3 YEAR | C | R | R | R | R | R | R | I | R | N | `build_source_inventory`; one year object -> exactly 12 monthly cells |
| F4 MONTHLY | C | R | R | R | R | R | R | I | R | N | `LOCATOR_STRATEGIES`; locked traversal and matrix mutation |
| F5 auxiliary | C | R | R | R | R | R | R | I | R | N | F5 methodology; auxiliary coverage plus independent comparability evidence |
| F6 GLOBAL | C | C | C | C | C | I | N | I | C | N | successor parser/record; GLOBAL covered year -> 12 cells, no old-rule equality gate |
| F7 MONTHLY | C | R | R | R | R | R | R | I | R | N | `calendar_envelope_months`; envelope/base calendar parse and FINAL_SIGNAL_D0 input |

The existing semantic engine (`src/v9_005_stage_a_semantics.py`) is reviewed
at its own input boundary, but no current production adapter constructs its
official parsed events from the F1--F7 locked objects. Its existence is not a
production semantic integration PASS.

## Cross-family gaps and dependency order

1. **R — acquisition/enumeration/matrix wiring.** Implement locked-root
   traversal and raw-lock provenance for F1--F5/F7, F2 bridge, F3 twelve-month
   fanout, F5 comparability evidence, F6 required-year fanout, and exact
   `MONTHLY_COVERAGE_MATRIX` mutation. Authority: acquisition-object model,
   F2/F4/F3/F5/F7 reviews, and successor F6 design. Tests: synthetic root/
   child fixtures, redirect/provenance assertions, and exact 648-cell matrix
   coverage. Depends on no new methodology.
2. **R — content-to-semantic/calendar integration.** Parse the locked family
   payloads into semantic-engine inputs, terminal snapshot/reconstruction
   inputs, F7 base/envelope calendar evidence, and `FINAL_SIGNAL_D0` input.
   Tests: safe synthetic payload transformations and semantic/calendar end-to-
   end inputs. Depends on gap 1's locked-object interface.
3. **I — final production integration/readiness.** Wire all family results to
   `run_stage_a`, derive `FREE_JPX_METADATA_PROBE_PASS`, preserve safe output,
   and permit `ACQUISITION_IMPLEMENTATION_COMPLETE=true` only after gaps 1--2
   pass their targeted offline tests. Tests: synthetic full Stage-A success,
   each fail-closed boundary, zero-network preflight, and safe-output checks.
   Depends on gaps 1 and 2.

No `METHODOLOGY_DECISION_REQUIRED` gap was found: the listed work is already
specified by the cited methodology/review chain. The minimum sensible plan is
therefore three substantive checkpoints (A acquisition/matrix, B
content/semantic/calendar, C final integration), in that dependency order.

## Readiness gate

`ACQUISITION_IMPLEMENTATION_COMPLETE` MUST remain false until all seven
families' required slots are acquired from their reviewed locator strategies,
raw-lock/provenance validation and transformations are wired, F2/F3/F5/F6/F7
fanout semantics are mechanically tested, semantic/calendar inputs are
constructed, `run_stage_a` derives the full matrix and
`FREE_JPX_METADATA_PROBE_PASS`, and the targeted offline end-to-end safe-output
suite passes. This audit creates no execution, network, private, model,
backtest, promotion, or profitability authority.
