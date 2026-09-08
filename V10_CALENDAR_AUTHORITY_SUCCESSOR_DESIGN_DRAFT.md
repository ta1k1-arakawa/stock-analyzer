# V10 Calendar Authority Successor Design Draft

```text
study_id=V10_CALENDAR_AUTHORITY_SUCCESSOR
predecessor=V9_CROSS_SECTIONAL_CLOSE_AUCTION
design_status=FREEZE_CANDIDATE_AWAITING_GPT_EXACT_SHA_REVIEW
evidence_role=INPUT_BINDING_ONLY
profitability_evidential_capacity=ZERO
execution_authorized=false
network_requests=0
```

## 1. Purpose and successor boundary

V10 is a new successor feasibility/design study. It is not a retry, repair,
rerun, or reinterpretation of V9. V9 remains terminal because its frozen and
authorized calendar/source lineage did not establish the historical actual
TSE-session authority required by `V9_009_HIGH_2`.

The only permitted methodological change in V10 is the mechanism used to
establish historical TSE cash-equity session/date eligibility. The V9
cross-sectional close-auction economic hypothesis and all non-calendar
methodology remain inherited unchanged. V10 has no authority to turn the V9
terminal result into a strategy or profitability result.

This draft performs no source acquisition, empirical source probing, raw
payload read, private read, network request, T0 operation, outcome read,
model fit, backtest, or profitability calculation.

## 2. Predecessor terminal facts

The current V9 closure is preserved exactly:

```text
V9_009_HIGH_2=UNRESOLVED_TERMINAL
V9_CALENDAR_AUTHORITY=FAIL_TERMINAL
V9_CALENDAR_FAILURE_CLASS=CALENDAR_FEASIBILITY_FAILURE
V9_017_TERMINAL_RESULT=DATA_QUALITY_FAILURE/FIXED_EIGHT_LOCATOR_FAILURE
V9_017_TERMINAL_RESULT_CHANGED=false
T0=NOT_RUN
HISTORICAL_EVALUATION=NOT_PERFORMED
FUTURE_PROFITABILITY=UNESTABLISHED
```

The V9_017 Source-B terminal result is a distinct data-quality failure. It is
not reclassified as a calendar failure, and V10 does not authorize a V9_017
rerun, refetch, locator repair, or same-study Source-B repair. V9_009 HIGH_2
was not declared impossible in every external source; it was unresolved
within the frozen/authorized V9 lineage.

## 3. Preserved V9 hypothesis and economics

V10 preserves the V9 scientific question: whether a pooled cross-sectional
score can support a robust Japanese-equity close-auction / close-to-close
architecture under the stated capital constraint. In particular, V10 does
not change:

- Japanese domestic ordinary common-stock scope, point-in-time membership,
  survivorship treatment, daily eligibility, and the fixed feature-history
  boundary;
- pooled cross-sectional scoring with ticker identity excluded from model
  features, the Ridge control, the shallow LightGBM candidate, mandatory
  TOPIX/equal-weight/random-K reporting baselines, and the fixed no-search
  model protocol;
- 1-share granularity, approximately equal-notional allocation, ten maximum
  concurrent positions, the 400,000 JPY primary case, the 300,000 JPY
  robustness case, the 90% invested / 10% cash-buffer mechanics, and the
  sector-cap and carry semantics;
- causal D0 information, the three-trading-day signal cadence, D1 entry,
  D3 planned exit, two close-to-close intervals, no optimistic stop-price
  execution, no same-close exit proceeds for same-close entries, and the
  frozen no-fill, cost, slippage, corporate-action, and unresolved-exit
  mechanics;
- the ten causal factors, split-neutral ex-dividend price-return target,
  same-date cross-sectional percentile ranks, and the fixed training/model
  roles;
- the V9 time roles: 2017 pre-evaluation training only, formal development
  evaluation from 2018-01-01 through 2025-12-31, and no change to the frozen
  T1/OOS, holdout, model-selection, threshold, benchmark, or promotion
  criteria; and
- the V9 requirements for transaction-cost disclosure, robustness, drawdown,
  reproducibility, and no profitability claim before the later authorized
  evaluation stages.

The V9 design and charter remain the authority for every item above. This
draft does not alter any non-calendar item. Any future ambiguity outside the
closed calendar mechanism must stop for a separate methodology decision.

## 4. The single changed component

Only this component changes:

```text
V10_CHANGED_COMPONENT=HISTORICAL_TSE_SESSION_DATE_AUTHORITY_MECHANISM
```

V10 replaces the failed V9 calendar/source feasibility route with a new,
separately designed authority mechanism. It does not alter the economic
hypothesis, target, data partitions, evaluation period, labels, thresholds,
cost/slippage assumptions, position sizing, universe, search space, or
profitability criteria.

## 5. Authority and source policy

The frozen deterministic calendar generator is:

```text
V10_CONCRETE_SOURCE_IDENTITY=PANDAS_MARKET_CALENDARS_JPX_5_4_0
V10_ACCEPTED_SOURCE_CLASS_EXACT_FORM=FROZEN_DERIVED_JPX_SESSION_CALENDAR_WITH_OFFICIAL_JPX_TSE_PROVENANCE
V10_SOURCE_COUNT_AND_PRIORITY=ONE_CANONICAL_GENERATOR_NO_FALLBACK
calendar_generator=pandas_market_calendars
package=pandas_market_calendars==5.4.0
calendar_name=JPX
calendar_alias_selection_is_not_caller_configurable=true
```

`pandas_market_calendars` is third-party deterministic code and is not an
official JPX product. Official JPX/TSE publications provide provenance for
the relevant market rules and exceptional-session facts. The pinned package
and calendar name are the sole generator inputs; Yahoo, Stooq, broker
calendars, vendor calendars, OS/locale calendars, generic weekday logic,
`pd.bdate_range`, J-Quants calendars, and JPX monthly PDFs are not fallback
or repair sources.

The exact upstream provenance is frozen as follows:

```text
upstream_repo=rsheftel/pandas_market_calendars
upstream_commit=ce73d50c85d773f96b1b712b6b16dd94b0e028b3
calendar_source_file=pandas_market_calendars/calendars/jpx.py
calendar_source_blob=0c2041b1300d1dbbd505202b00ac0ada38c712e1
holiday_source_file=pandas_market_calendars/holidays/jp.py
holiday_source_blob=4c34214d06862e02ac22e946757463f748074fde
```

The supplied methodology evidence establishes that the pinned JPX calendar
contains the `EquityTradingSystemFailure` exclusion for 2020-10-01, that its
upstream source cites the official JPX/TSE system-failure publication, that
the upstream JPX tests expect 2020-10-01 as a trading-system-failure holiday,
and that the calendar contains the 2024-11-05 close-time transition. These
facts are methodology evidence supplied for this freeze candidate; this task
does not reconfirm them by network or package execution.

The source order is a single canonical generator. No second source is
permitted, no source priority can be selected after observation, and no
fallback or favorable-calendar selection exists.

## 6. Coverage and exact session/date semantics

The inherited required coverage remains exactly:

```text
coverage_start=2017-01-01
coverage_end=2026-01-31
```

For each date in that inclusive range, the future accepted authority must
produce one mechanically bound state for the actual TSE cash-equity session
relevant to the frozen close-auction schedule:

```text
ELIGIBLE = the pinned JPX generator emits the date exactly once as a session
           label AND that emitted session has a valid market_close
INELIGIBLE = the pinned JPX generator emits no session label for the date
UNRESOLVED_FAIL_TERMINAL = generator failure, duplicate emitted session label,
           malformed emitted session label, emitted label outside coverage,
           invalid/missing market_close, runtime provenance mismatch, anchor
           failure, or canonicalization failure
```

Only `ELIGIBLE` dates may enter the V9 signal calendar. `INELIGIBLE` dates
must not enter it. An ordinary date that the pinned generator does not emit
is `INELIGIBLE`, not a missing-source failure. `UNRESOLVED_FAIL_TERMINAL` is
reserved for
generator failure, duplicate or malformed emitted labels, emitted labels
outside coverage, invalid or missing `market_close`, runtime provenance
mismatch, anchor failure, or canonicalization failure. No date may be
inferred from weekdays, neighboring dates, price absence, or a favorable
later result.

The accepted calendar must support the inherited first-session-of-month
cutoff and the D0/D1/D2/D3 lookup. A shortened, early, or partial session is
still eligible when the pinned calendar emits a valid session label and the
generated schedule has a valid `market_close`. There is no separate
discretionary normal-session filter:

```text
V10_PARTIAL_SESSION_ELIGIBILITY=VALID_EMITTED_SESSION_WITH_VALID_MARKET_CLOSE
```

Structural requirements are exact: the generator's emitted session labels are
unique, valid, within coverage, and canonicalized into deterministic
ascending order. A duplicate, malformed, contradictory, or incompletely
scoped emitted session record fails closed. Non-emission remains the exact
`INELIGIBLE` result above.

The following two pre-outcome regression anchors are checked against the
generator output and are not local calendar patches:

```text
2020-10-01=INELIGIBLE
2020-10-02=ELIGIBLE
V10_LOCAL_DATE_PATCH_ALLOWED=false
```

The first anchor is supported by the official TSE system-failure evidence
and the pinned generator's corresponding `EquityTradingSystemFailure`
holiday; the second is supported by the official resumption notice. If
either anchor fails, feasibility is terminal and no repair is permitted. No
additional anchors may be added from later observations.

## 7. Runtime provenance and public evidence boundary

All durable V10 Python execution is bound to the repository's canonical
protected real-execution environment, not to a V10-specific virtual
environment:

```text
V10_PROTECTED_RUNTIME_ENVIRONMENT=.venv-real-execution
V10_PROTECTED_RUNTIME_INTERPRETER=.venv-real-execution\Scripts\python.exe
V10_RUNTIME_ENVIRONMENT_AUTHORITY=REPOSITORY_CANONICAL_PROTECTED_ENVIRONMENT
V10_ALTERNATE_VIRTUALENV_ALLOWED=false
V10_INSTALL_OUTSIDE_PROTECTED_LOCK_ALLOWED=false
```

The protected installation/runtime authority is the reviewed
`requirements-real-execution.lock.txt` together with the applicable
`AI_REAL_EXECUTION_RUNBOOK.md` and `REAL_EXECUTION_PYTHON_ENVIRONMENT.md`
environment freeze and validation contract. The explicit interpreter above
must be used and its resolved identity must be verified before any protected
V10 boundary.

The study-specific `V10_RUNTIME_ENVIRONMENT_LOCK.json` is an additional V10
provenance snapshot only. It is not an installation authority and does not
authorize a separate virtual environment or interpreter:

```text
V10_RUNTIME_LOCK_ROLE=ADDITIONAL_PROVENANCE_SNAPSHOT_NOT_INSTALLATION_AUTHORITY
V10_RUNTIME_LOCK_INSTALLATION_AUTHORITY=false
V10_RUNTIME_LOCK_SEPARATE_INTERPRETER_AUTHORITY=false
```

The current reviewed `requirements-real-execution.lock.txt` does not contain
the V10-required `pandas-market-calendars` / `exchange-calendars` stack.
Therefore the V10 runtime-lock creation and semantic calendar generation
are currently not environment-ready:

```text
V10_CURRENT_CANONICAL_ENVIRONMENT_V10_READY=false
V10_RUNTIME_LOCK_CREATION_READY=false
V10_SEMANTIC_GENERATION_READY=false
```

Before `V10_RUNTIME_ENVIRONMENT_LOCK.json` can be created, a separate future
canonical-environment-extension stage is required under
`AI_REAL_EXECUTION_RUNBOOK.md` and `REAL_EXECUTION_PYTHON_ENVIRONMENT.md`.
That stage must operate only on `.venv-real-execution` through its explicit
interpreter, establish an exact-pinned reviewed package authority containing
the V10 stack, preserve all existing protected-environment readiness
requirements, and receive the required Windows-grounded validation,
environment-freeze promotion, and GPT exact-SHA review before V10 continues.
While selecting, provisioning, or locking dependencies, it must not import
or run the JPX generator, generate or inspect calendar dates, inspect
research data, or perform historical-price, private/sealed, T0, model,
backtest, or profitability operations. Any real software acquisition needed
by repository governance requires separate point-of-use authority. No
alternate V10 virtual environment is permitted, and no installation outside
the reviewed canonical protected lock is permitted.

If the canonical environment cannot be extended while preserving
`pandas_market_calendars==5.4.0`, `calendar_name=JPX`, the frozen source blob
identities, and repository protected-environment readiness, the process must
stop as `CHATGPT_DECISION_REQUIRED` before runtime-lock creation or
generator execution.

Only after that future canonical environment extension has passed its
Windows validation/environment-freeze promotion and GPT exact-SHA review
may the later implementation complete a no-network runtime provenance
preflight before calendar generation. It must mechanically verify all of the
following against the frozen values:

- installed distribution name is exactly `pandas_market_calendars`;
- installed distribution version is exactly `5.4.0`;
- the only calendar name is exactly `JPX` and is not caller configurable;
- the installed `pandas_market_calendars/calendars/jpx.py` bytes have Git
  blob identity `0c2041b1300d1dbbd505202b00ac0ada38c712e1`;
- the installed `pandas_market_calendars/holidays/jp.py` bytes have Git blob
  identity `4c34214d06862e02ac22e946757463f748074fde`; and
- the exact Python version, `pandas_market_calendars` version, pandas
  version, `exchange-calendars` version, and every other runtime distribution
  mechanically required by the installed calendar stack are recorded in one
  deterministic sorted distribution/version mapping with a SHA-256 digest.

Git-blob identity means standard Git blob hashing over the exact file bytes,
not a plain-file SHA-1 relabeled as a Git blob. A mismatch is
`RUNTIME_CALENDAR_PROVENANCE_MISMATCH` and is `FAIL_TERMINAL`; there is no
repair, version comparison, or reinstall-and-retry after feasibility
generation begins. Dependency versions are not selected by inspecting
generated dates.

The sole runtime provenance mechanism is:

```text
V10_RUNTIME_PROVENANCE_MECHANISM=SEPARATE_CANONICAL_RUNTIME_ENVIRONMENT_LOCK
V10_RUNTIME_LOCK_FILENAME=V10_RUNTIME_ENVIRONMENT_LOCK.json
V10_DIRECT_RUNTIME_MAPPING_ALTERNATIVE=false
```

Before the one calendar feasibility execution, a separate
`V10_RUNTIME_ENVIRONMENT_LOCK.json` must be created as a provenance
snapshot from that already reviewed canonical protected environment. Its
creation is not an installation step and it must not select a different
interpreter or environment. The runtime-lock stage must not import or run
the JPX generator, generate or inspect calendar dates, or inspect prices,
returns, outcomes, or other research data. The runtime lock stage requires
its own GPT exact-SHA/provenance review before the one semantic calendar
generation.

The runtime-lock JSON has exactly this required contract:

```text
schema_version=V10_RUNTIME_ENVIRONMENT_LOCK_V1
python_version
calendar_distribution_name
calendar_distribution_version
calendar_name
calendar_source_blob
holiday_source_blob
runtime_distributions
runtime_distribution_count
```

The fixed values are `calendar_distribution_name=pandas_market_calendars`,
`calendar_distribution_version=5.4.0`, `calendar_name=JPX`, and the two
frozen source blobs above. `runtime_distributions` is the complete installed
Python distribution set in the repository's canonical protected real-execution
environment at lock time, represented as an array of objects with exactly
`name` and
`version` fields. For each installed metadata name, lower-case it and replace
each maximal run of `-`, `_`, or `.` with `-`; reject empty normalized names
and duplicate normalized names. Sort the array lexicographically by
normalized `name`, and require `runtime_distribution_count` to equal its
length. The set must include at least `pandas-market-calendars`, `pandas`,
and `exchange-calendars`, plus every other installed distribution.

`python_version` is exactly the string
`f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"`.
No other Python-version representation is permitted.

The canonical runtime-lock bytes are UTF-8 with `ensure_ascii=false`,
`sort_keys=true`, `separators=(',', ':')`, `allow_nan=false`, and exactly one
final LF. The lock JSON contains no self-hash. The later safe receipt and
canonical calendar artifact record:

```text
runtime_environment_lock_sha256=SHA256(exact canonical V10_RUNTIME_ENVIRONMENT_LOCK.json bytes)
```

There is no direct runtime-mapping embedding alternative. Public safe
receipts may expose only approved hashes, counts, coverage, booleans, closed
failure codes, and reviewed provenance SHAs. They must not expose raw
payloads, private paths, credentials, URLs, ticker identities, prices,
outcomes, or exception text.

No package is installed or executed, and no calendar is generated, by this
design task.

## 8. Canonical generator and artifact contract

The later implementation must use exactly the pinned package/version and
`calendar_name="JPX"`, generate only the fixed coverage, and persist one
reviewed canonical artifact rather than silently regenerating under a newer
library. The artifact has exactly the following top-level fields and no
extras:

```text
schema_version=V10_CANONICAL_CALENDAR_V1
calendar_method=PANDAS_MARKET_CALENDARS_JPX_5_4_0
calendar_package=pandas_market_calendars
calendar_package_version=5.4.0
upstream_commit
calendar_source_blob
holiday_source_blob
calendar_name=JPX
runtime_environment_lock_sha256
coverage_start=2017-01-01
coverage_end=2026-01-31
trading_dates
trading_date_count
python_version
pandas_version
canonical_calendar_sha256
generator_implementation_git_sha
```

`runtime_environment_lock_sha256` must equal the SHA-256 of the exact
reviewed canonical runtime-lock bytes; the complete runtime distribution
mapping is bound only through that separate lock. `python_version` and
`pandas_version` remain required scalar provenance fields and must equal the
reviewed lock. `trading_dates` is the sorted unique `YYYY-MM-DD`
session-label sequence.
Every emitted session must have a valid `market_close`; generation failure,
duplicate or malformed session labels, out-of-coverage labels, invalid
close values, or noncanonical serialization is unresolved and fails closed.
The canonical JSON bytes are UTF-8 with `ensure_ascii=false`,
`sort_keys=true`, `separators=(',', ':')`, `allow_nan=false`, and exactly one
final LF. `canonical_calendar_sha256` is calculated over the canonical object
with that field excluded, then recorded in the artifact. `pandas_version`
equals the exact version for normalized distribution name `pandas` in the
reviewed runtime lock. `python_version` equals the reviewed runtime lock's
exact `python_version`; no alternate representation is permitted. The exact
artifact field set above admits no additional fields.

Session-label canonicalization is mechanical for every schedule index label:

1. Convert the label with `pandas.Timestamp(label)`; conversion failure is
   `MALFORMED_SESSION_LABEL`.
2. Require `tzinfo is None`, `hour=0`, `minute=0`, `second=0`,
   `microsecond=0`, and `nanosecond=0`; any violation is
   `MALFORMED_SESSION_LABEL`.
3. Canonicalize as `timestamp.date().isoformat()` and require the resulting
   string to be exact `YYYY-MM-DD`.
4. Classify duplicate canonical labels as `DUPLICATE_SESSION_LABEL` and
   labels outside the inclusive frozen coverage as
   `OUT_OF_COVERAGE_SESSION_LABEL`.

No timezone shift, neighboring-date correction, or other date repair is
permitted. For every emitted schedule row, `market_close` is valid only when
`pandas.Timestamp(value)` succeeds, the result is not `NaT`, and its
`tzinfo` is not `None`; otherwise the failure code is
`INVALID_MARKET_CLOSE`. The time of `market_close` is not used to add or
remove an otherwise emitted session.

The sole safe receipt schema is:

```text
V10_SAFE_RECEIPT_FILENAME=V10_CALENDAR_FEASIBILITY_SAFE_RECEIPT.json
V10_SAFE_RECEIPT_SCHEMA=V10_CALENDAR_FEASIBILITY_SAFE_RECEIPT_V1
```

Its exact top-level fields are:

```text
schema_version
status
failure_code
design_git_sha
generator_implementation_git_sha
runtime_environment_lock_sha256
calendar_name
coverage_start
coverage_end
calendar_artifact_created
canonical_calendar_sha256
trading_date_count
anchor_2020_10_01
anchor_2020_10_02
research_data_network_requests
historical_calendar_data_acquisition
private_or_sealed_reads
human_gate_consumed
t0_run
```

The exact semantic constraints are:

```text
schema_version=V10_CALENDAR_FEASIBILITY_SAFE_RECEIPT_V1
calendar_name=JPX
coverage_start=2017-01-01
coverage_end=2026-01-31
```

`status` is exactly `PASS` or `FAIL`. `failure_code` is exactly one of:

```text
NONE
RUNTIME_CALENDAR_PROVENANCE_MISMATCH
CALENDAR_GENERATOR_FAILURE
DUPLICATE_SESSION_LABEL
MALFORMED_SESSION_LABEL
OUT_OF_COVERAGE_SESSION_LABEL
INVALID_MARKET_CLOSE
ANCHOR_2020_10_01_FAILURE
ANCHOR_2020_10_02_FAILURE
CANONICALIZATION_FAILURE
DURABLE_ARTIFACT_WRITE_FAILURE
```

There is exactly one receipt schema and no alternate receipt mechanism. A
`PASS` receipt requires `failure_code=NONE`,
`calendar_artifact_created=true`, a lower-case 64-hex
`canonical_calendar_sha256`, an integer `trading_date_count` greater than
zero, `anchor_2020_10_01=INELIGIBLE`, and
`anchor_2020_10_02=ELIGIBLE`. A `FAIL` receipt requires
`failure_code` not equal to `NONE`.

Both anchor fields are exactly one of `INELIGIBLE`, `ELIGIBLE`, or
`NOT_CHECKED`. They record the observed generator classification only when
that anchor has actually been checked; an execution that fails before a
check records `NOT_CHECKED`. Anchor checks occur only in this order:

1. `2020-10-01`
2. `2020-10-02`

No additional anchor is permitted. A checked mismatch at the first anchor
uses `ANCHOR_2020_10_01_FAILURE`. A checked mismatch at the second anchor,
after the first passed, uses `ANCHOR_2020_10_02_FAILURE`.

If failure occurs before valid canonical calendar artifact bytes exist,
`calendar_artifact_created=false`,
`canonical_calendar_sha256=null`, and `trading_date_count=null` are required.
If canonical artifact bytes were fully and validly created before a later
durable artifact-handling failure, the mechanically known artifact fields
are preserved rather than replaced with fabricated nulls. The closed
`DURABLE_ARTIFACT_WRITE_FAILURE` code refers only to failure to durably
create the canonical calendar artifact after canonical bytes were prepared.
A failure to durably write the safe receipt cannot be represented inside a
receipt that does not exist; it is an execution/orchestration observability
incident for Phase-C inspection and does not authorize retry.

`design_git_sha` and `generator_implementation_git_sha` are exact lower-case
40-hex reviewed Git SHAs. `runtime_environment_lock_sha256` is an exact
lower-case 64-hex SHA-256 and equals the reviewed canonical runtime-lock
SHA-256. The operation-status fields are fixed:

```text
research_data_network_requests=0
historical_calendar_data_acquisition=0
private_or_sealed_reads=0
human_gate_consumed=0
t0_run=NOT_RUN
```

The receipt contains no raw paths, package metadata dump, prices, ticker
identities, outcomes, or profitability fields. Its canonical JSON bytes are
UTF-8 with `ensure_ascii=false`, `sort_keys=true`,
`separators=(',', ':')`, `allow_nan=false`, and exactly one final LF.

When more than one failure is mechanically observable, the receipt records
only the highest-precedence applicable code in this exact order:

```text
1. RUNTIME_CALENDAR_PROVENANCE_MISMATCH
2. CALENDAR_GENERATOR_FAILURE
3. DUPLICATE_SESSION_LABEL
4. MALFORMED_SESSION_LABEL
5. OUT_OF_COVERAGE_SESSION_LABEL
6. INVALID_MARKET_CLOSE
7. ANCHOR_2020_10_01_FAILURE
8. ANCHOR_2020_10_02_FAILURE
9. CANONICALIZATION_FAILURE
10. DURABLE_ARTIFACT_WRITE_FAILURE
```

No failure code authorizes retry or repair.

## 9. Outcome-blind feasibility gate

Calendar feasibility must complete before any operation that can expose or
calculate an outcome:

```text
runtime binding and session eligibility
    -> safe no-network inspection and GPT adjudication
    -> only if PASS, later V10 input-binding acceptance
    -> only after separately authorized later stages, T0/outcome/model work
```

The feasibility runner must have no access path to prices, returns, labels,
model inputs, target values, trading outcomes, or profitability metrics. It
must not choose a calendar by comparing outcome results or by selecting the
calendar with the most eligible dates.

## 10. Bounded feasibility process and stopping rule

The infrastructure budget is finite and preregistered:

```text
V10_FEASIBILITY_EXECUTION_LIMIT=1
V10_AUTOMATIC_SUCCESSOR=false
V10_OUTCOME_DEPENDENT_RETRY=false
V10_FAVORABLE_CALENDAR_SELECTION=false
V10_FALLBACK_ALLOWED=false
```

The only later execution sequence permitted after this design is frozen,
implemented, and independently reviewed is:

1. A separate canonical-environment-extension stage operates only on
   `.venv-real-execution` through
   `.venv-real-execution\Scripts\python.exe`, under repository environment
   governance, if the V10 stack is not already present. It performs no JPX
   generator import, date generation/inspection, research-data operation, or
   outcome operation.
2. That canonical environment extension must establish an exact-pinned
   reviewed package authority, preserve all existing readiness requirements,
   pass Windows-grounded validation and environment-freeze promotion, and
   receive GPT exact-SHA review before V10 continues. If it cannot preserve
   the frozen V10 package/version, `JPX` name, source blobs, and protected
   readiness contract, stop with `CHATGPT_DECISION_REQUIRED`.
3. The V10 runtime-lock stage creates the canonical
   `V10_RUNTIME_ENVIRONMENT_LOCK.json` as an additional provenance snapshot
   from the reviewed canonical environment, without importing or running
   the JPX generator and without generating or inspecting calendar dates.
4. GPT reviews and binds the exact runtime-lock provenance before the one
   semantic calendar generation.
5. Synthetic-only implementation and targeted tests are completed.
6. GPT performs the exact-SHA implementation review. If repository
   governance requires this implementation review before runtime-lock
   creation, that stricter ordering is retained; the lock must still be
   reviewed before generation.
7. Phase A no-network provenance preflight verifies the exact reviewed
   runtime lock, its canonical SHA-256, the frozen design, reviewed
   implementation, clean state, canonical protected interpreter identity,
   pinned package/version, fixed `JPX` name, exact Git-blob identities, and
   exact runtime distribution mapping.
8. Exactly one offline feasibility execution invokes only the pinned
   generator over the fixed coverage. It reads no historical calendar
   payload, performs no HTTP request, and has no source discovery, candidate
   expansion, fallback, or second execution.
9. The execution durably persists one canonical calendar artifact and one
   safe receipt containing the same runtime-lock SHA-256, bound provenance,
   canonical sorted session labels, and closed failure/result fields.
10. A safe receipt is inspected without network.
11. GPT adjudicates only the bounded generator evidence and does not use any
    outcome information.

The single feasibility budget is consumed when the pinned generator first
begins the one feasibility execution. A pre-generation operational failure
does not create an automatic rerun; it stops for methodology authority. A
post-boundary failure never restores a gate, permits a retry, a second lock,
environment repair, or a new source under V10.

If the one execution cannot establish the full required coverage and exact
session semantics, V10 is:

```text
V10_CALENDAR_FEASIBILITY=FAIL_TERMINAL
T0=NOT_RUN
HISTORICAL_EVALUATION=NOT_PERFORMED
```

The calendar feasibility route then stops. No unbounded PDF, locator, label,
calendar, provider, or dependency-version loop is allowed.

## 11. Missing, disagreement, duplicate, and malformed handling

The following are terminal feasibility failures:

- generator failure;
- duplicate emitted session labels;
- malformed emitted session labels or labels outside requested coverage;
- missing or invalid `market_close` for an emitted session;
- runtime provenance mismatch; or
- failure of either frozen regression anchor or canonicalization.

An ordinary date not emitted by the pinned generator is exactly
`INELIGIBLE`; it is not a missing-date failure and is not imputed from any
other rule. No duplicate is resolved by first/last choice. There is no
second source, disagreement resolution, alternate endpoint, alias,
neighboring-date inference, fallback, or post-observation repair. A valid
emitted shortened, early, or partial session remains `ELIGIBLE` under the
single `market_close` rule.

## 12. Later authority and execution requirements

This draft grants no authority. The ten-step sequence in Section 10 is the
sole later execution order. GPT methodology PASS on this exact design and a
frozen exact Git SHA is its prerequisite; the sequence then requires
software/environment provisioning if mechanically necessary, creation and
GPT review of the exact canonical `V10_RUNTIME_ENVIRONMENT_LOCK.json`,
synthetic implementation/tests and GPT implementation review, Phase A
no-network preflight, exactly one offline pinned-generator execution,
durable artifact/receipt, no-network inspection, and GPT adjudication. No
historical calendar payload is acquired or read.

Calendar generation itself requires no historical calendar-data acquisition
or private calendar access:

```text
V10_CALENDAR_NETWORK_DATA_ACQUISITION_REQUIRED=false
V10_PRIVATE_OR_SEALED_CALENDAR_ACCESS_REQUIRED=false
V10_CALENDAR_HUMAN_GATE_REQUIRED=false
```

If package provisioning is needed later, it is software provisioning only and
must follow repository/environment governance; it does not authorize research
data access. No human gate is consumed by this draft. The pinned generator
must be used exactly as frozen, and provisioning cannot change its version,
calendar name, source blobs, runtime lock, or methodology based on output.

## 13. Explicit non-actions and prohibitions

This task performs zero network requests, zero source acquisition, zero
private/sealed reads, zero protected Source-A reads, and zero semantic
inspection of historical outcomes or preserved production inputs. It does
not implement a runner or parser, select a source, freeze a session grammar,
consume a human gate, run T0, fit a model, run a backtest, or calculate
profitability.

V10 does not authorize Yahoo/Stooq, J-Quants whole-market daily quotes,
provider substitution, ticker-universe changes, PIT/delisted rules, outage
composites, new thresholds, new calendars, or any future Astra
recommendation. It does not change the V9 evaluation period, holdout,
labels/targets, costs, slippage, search space, stopping criteria, or
profitability gates.

## 14. Design-freeze PASS criteria and next stage

GPT may mark this design `PASS_FROZEN` only when the exact source authority
class and identity, coverage, session/partial-session semantics, the separate
runtime-lock schema/canonical bytes/GPT binding, package/version and Git-blob
provenance, exact dependency-set binding, duplicate/malformed/out-of-coverage/
market-close/anchor/canonicalization rules, non-emission semantics, fallback
policy, no-retry/no-repair policy, failure codes, one-shot budget, safe
receipt schema, and later offline execution sequence are all mechanically
closed.

If GPT passes and freezes this draft, the next stage is a synthetic-only
implementation of the frozen minimal calendar/session-binding runner,
followed by targeted tests and an independent exact-SHA implementation
review. That stage still performs no real acquisition. Only a later,
separately authorized execution may evaluate V10 calendar feasibility.

V10 feasibility PASS would establish input-binding readiness only. It would
not establish profitability and would not itself authorize T0, historical
evaluation, model fitting, outcome access, or strategy promotion.

```text
V10_STATUS=FREEZE_CANDIDATE_AWAITING_GPT_EXACT_SHA_REVIEW
V10_EXECUTION_AUTHORIZED=false
T0=NOT_RUN
HISTORICAL_EVALUATION=NOT_PERFORMED
FUTURE_PROFITABILITY=UNESTABLISHED
NEXT_ACTION=GPT_EXACT_SHA_INDEPENDENT_REVIEW
```
