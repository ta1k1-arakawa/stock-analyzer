# V10 Canonical Environment Extension Design Draft

```text
study_id=V10_CANONICAL_ENVIRONMENT_EXTENSION
document_role=OPERATIONAL_REPRODUCIBILITY_SUCCESSOR
status=DRAFT_AWAITING_GPT_EXACT_SHA_REVIEW
parent_scientific_design=V10_CALENDAR_AUTHORITY_SUCCESSOR
V10_FROZEN_DESIGN_GIT_SHA=8c923ed1734c6bdfe95a743cd9e15a5156d62c03
V10_FREEZE_RECORD_GIT_SHA=4e925f83edff4cbb15dd9e8e8fdf97630123f3a1
V10_EXECUTION_AUTHORIZED=false
```

## 1. Scope and non-methodology boundary

This is an operational/reproducibility successor under the frozen V10
scientific design. It establishes only the future process for extending the
canonical protected Python environment so that V10 may later create its
separate runtime-provenance snapshot. It changes no V10 or V9 economics,
calendar methodology, coverage, labels, partitions, thresholds, costs, search
space, stopping rules, or profitability criteria.

This draft grants no package-resolution, package-installation, environment
mutation, calendar, research-data, private/sealed, T0, historical-evaluation,
model, backtest, or profitability authority.

## 2. Immutable predecessor authority

The only canonical protected predecessor environment is:

```text
canonical_environment=.venv-real-execution
canonical_interpreter=.venv-real-execution\Scripts\python.exe
predecessor_package_count=15
predecessor_lock_git_blob_sha1=5e9d15caa822bd39e751a49cd0758db6eaf04bdf
predecessor_lock_canonical_git_sha256=ddd505cc01ac4a3a798cdf7ed9c35b3a9e56db569a421aef98c02d013dd286b7
```

The predecessor's exact 15 resolved pins are constraints for the V10
successor candidate:

```text
cffi==2.1.1
charset-normalizer==3.5.1
cryptography==50.0.1
numpy==2.5.2
pandas==3.0.5
pdfminer-six==20260107
pdfplumber==0.11.10
pillow==12.3.0
pip==25.0.1
pycparser==3.0
pypdfium2==5.13.0
python-dateutil==2.9.0.post0
six==1.17.0
tzdata==2026.3
xlrd==2.0.2
```

All V9_014 PDF environment-successor artifacts remain immutable historical
evidence. V10 does not overwrite, reinterpret, or use the former
non-canonical staging-venv topology.

```text
V10_ALT_STAGING_VENV_ALLOWED=false
V10_RESOLUTION_INTERPRETER=.venv-real-execution\Scripts\python.exe
```

Invocation through the canonical interpreter for future resolution does not
authorize mutation of that environment.

## 3. Dependency delta and fail-closed constraints

The sole newly specified direct dependency is:

```text
pandas-market-calendars==5.4.0
```

No new transitive dependency version is selected in this draft. A later
Windows-grounded resolution candidate must contain the unchanged predecessor
15 pins, `pandas-market-calendars==5.4.0`, `pandas==3.0.5`,
`exchange-calendars=<exact Windows-resolved version>`, and every mechanically
required new transitive dependency.

If satisfying the new direct dependency would change or remove any predecessor
pin, stop with `CHATGPT_DECISION_REQUIRED`. If the resulting closure omits
`exchange-calendars`, stop with `CHATGPT_DECISION_REQUIRED`.

## 4. Future Phase A/B/C software-resolution boundary

### Phase A — no-network preflight

Before any resolution operation, a future reviewed implementation must
read-only verify the frozen V10 design SHA, predecessor lock provenance and
15-package identity, canonical Python `3.12.10` / Windows AMD64 identity,
clean repository state, required artifacts, and a durable-resolution-root
non-collision condition. Phase A performs no network, installation, calendar
import, calendar generation, or date inspection.

### Phase B — one bounded resolution/acquisition operation

If real package-index/software network access is mechanically necessary,
Phase B requires fresh, point-of-use human authority. It uses only
`.venv-real-execution\Scripts\python.exe`, without installation and without
creating an alternate virtual environment.

The future implementation must use a reviewed pip dry-run/download/report
mechanism with the predecessor lock as constraints and a V10 direct-spec
candidate containing `pandas-market-calendars==5.4.0`. Wheel-only acquisition
and no persistent pip cache are preferred. If an exact required wheel is
unavailable and continuing would require an sdist/build-policy change, stop
with `CHATGPT_DECISION_REQUIRED`.

The first complete Windows-grounded resolution result fixes the candidate.
No complete result is re-resolved to obtain a preferable dependency set. A
Phase B failure receives no automatic retry; Phase C returns only safe
evidence to GPT. No JPX/calendar import or execution is permitted.

### Phase C — no-network inspection

Phase C inspects only safe package names, versions, counts, hashes, exit code,
and provenance. It performs no package repair, re-resolution, calendar
operation, or research-data operation.

## 5. Future successor artifacts and output safety

No successor artifact is created by this task. Later reviewed work must define
the exact schemas and create, at minimum:

```text
V10_CANONICAL_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt
V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json
V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE.json
V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LIVE_VALIDATION_EVIDENCE.json
V10_CANONICAL_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_VERIFICATION_EVIDENCE.json
```

Their public provenance is limited to package names, versions, counts, hashes,
Git SHAs, Python/platform identity, closed status/failure enums, and
mechanically known network/package-resolution counts. Public artifacts must
not contain private paths, raw stdout, raw stderr, or raw URLs.

## 5A. Exact successor-artifact schema closure

### Common JSON canonicalization

For every V10 extension JSON artifact in this design, including
`V10_CANONICAL_ENVIRONMENT_GENERIC_MIGRATION_AUTHORITY.json`, exactly the
listed keys are allowed; missing or extra keys are invalid. Canonical bytes
are UTF-8 with `ensure_ascii=false`, `sort_keys=true`, separators
`(',', ':')`, `allow_nan=false`, and exactly one final LF. SHA-256 values are
lowercase 64-hex; Git SHAs and Git-blob SHA-1 values are lowercase 40-hex.
JSON booleans are booleans, never `0`/`1`; counts are nonnegative integers,
never booleans.

### Direct specification

The exact future bytes of
`V10_CANONICAL_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt` are UTF-8 with LF line
endings, exactly one final LF, no blank lines, and no comments:

```text
pandas
xlrd==2.0.2
pdfplumber==0.11.10
pandas-market-calendars==5.4.0
```

It is direct-dependency specification only and never installation authority.

### Successor lock candidate

`V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE.json` has exactly:

```text
schema_version
artifact_status
frozen_v10_design_git_sha
extension_design_git_sha
reviewed_resolution_implementation_git_sha
direct_spec_git_blob_sha1
direct_spec_sha256
predecessor_lock_git_blob_sha1
predecessor_lock_sha256
predecessor_package_count
python_version
platform_system
platform_machine
sysconfig_platform
resolution_policy_id
resolved_packages
resolved_package_count
predecessor_pin_drift_count
pandas_market_calendars_version
exchange_calendars_version
```

Fixed values are:

```text
schema_version=V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE_V1
artifact_status=WINDOWS_RESOLUTION_CANDIDATE_NOT_INSTALL_AUTHORITY
frozen_v10_design_git_sha=8c923ed1734c6bdfe95a743cd9e15a5156d62c03
predecessor_lock_git_blob_sha1=5e9d15caa822bd39e751a49cd0758db6eaf04bdf
predecessor_lock_sha256=ddd505cc01ac4a3a798cdf7ed9c35b3a9e56db569a421aef98c02d013dd286b7
predecessor_package_count=15
python_version=3.12.10
platform_system=Windows
platform_machine=AMD64
sysconfig_platform=win-amd64
predecessor_pin_drift_count=0
pandas_market_calendars_version=5.4.0
```

`extension_design_git_sha` equals the final GPT-reviewed extension-design SHA;
`reviewed_resolution_implementation_git_sha` equals its later GPT-reviewed
SHA. `resolution_policy_id` equals the exact identifier frozen only when the
open resolution-mode finding is resolved; this design does not choose it.

`resolved_packages` is an array of objects containing only `name` and
`version`. Distribution names are lowercased with every maximal run of `-`,
`_`, or `.` replaced by `-`; empty or duplicate normalized names are invalid.
The array is sorted lexicographically by normalized name and
`resolved_package_count == len(resolved_packages)`. It contains all
predecessor 15 name/version pairs unchanged,
`pandas-market-calendars==5.4.0`, and exactly one `exchange-calendars` entry;
`exchange_calendars_version` equals that entry's version.

### Windows resolution evidence

`V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE.json` has
exactly:

```text
schema_version
artifact_status
status
failure_code
frozen_v10_design_git_sha
extension_design_git_sha
reviewed_resolution_implementation_git_sha
direct_spec_git_blob_sha1
direct_spec_sha256
predecessor_lock_git_blob_sha1
predecessor_lock_sha256
resolution_policy_id
process_exit_code
resolution_completed
candidate_artifact_created
successor_lock_candidate_sha256
resolved_package_count
package_index_network_requests
human_authority_consumed
package_installations
alternate_venv_created
calendar_imports
calendar_dates_inspected
```

```text
schema_version=V10_CANONICAL_ENVIRONMENT_SUCCESSOR_WINDOWS_RESOLUTION_EVIDENCE_V1
artifact_status=WINDOWS_RESOLUTION_EVIDENCE
status=PASS|FAIL
failure_code={NONE,RESOLUTION_PROCESS_FAILURE,RESOLUTION_REPORT_INVALID,PREDECESSOR_PIN_DRIFT,REQUIRED_DISTRIBUTION_MISSING,SOURCE_DISTRIBUTION_REQUIRED,UNAUTHORIZED_INSTALLATION,UNAUTHORIZED_ALTERNATE_ENVIRONMENT}
```

For `PASS`, `failure_code=NONE`, `process_exit_code=0`,
`resolution_completed=true`, `candidate_artifact_created=true`,
`successor_lock_candidate_sha256` is lowercase 64-hex,
`resolved_package_count>15`, `package_installations=0`,
`alternate_venv_created=false`, `calendar_imports=0`, and
`calendar_dates_inspected=0`. For `FAIL`, `failure_code != NONE`,
`candidate_artifact_created=false`, `successor_lock_candidate_sha256=null`,
and `resolved_package_count=null`; an invalid/failed resolution creates no
accepted candidate. Retry and exact pip mechanics remain open.

### Live validation evidence

`V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LIVE_VALIDATION_EVIDENCE.json` has
exactly:

```text
schema_version
artifact_status
status
failure_code
frozen_v10_design_git_sha
extension_design_git_sha
reviewed_generic_authority_transition_git_sha
migration_authority_git_blob_sha1
generic_lock_git_blob_sha1
generic_lock_sha256
generic_lock_package_count
observed_packages
observed_package_count
python_version
platform_system
platform_machine
sysconfig_platform
pandas_market_calendars_version
exchange_calendars_version
jpx_source_blob_match
holiday_source_blob_match
xls_probe_status
pdf_probe_status
package_index_network_requests
package_installations
calendar_object_creations
calendar_dates_inspected
protected_or_private_reads
t0_run
```

```text
schema_version=V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LIVE_VALIDATION_EVIDENCE_V1
artifact_status=V10_SUCCESSOR_LIVE_VALIDATION_EVIDENCE
status=PASS|FAIL
failure_code={NONE,PROVENANCE_BINDING_FAILURE,LIVE_PACKAGE_SET_MISMATCH,PYTHON_PLATFORM_MISMATCH,PMC_VERSION_MISMATCH,EXCHANGE_CALENDARS_VERSION_MISMATCH,JPX_SOURCE_BLOB_MISMATCH,HOLIDAY_SOURCE_BLOB_MISMATCH,XLS_PROBE_FAILURE,PDF_PROBE_FAILURE,UNAUTHORIZED_OPERATION_OBSERVED}
```

For `PASS`, all provenance bindings are exact; the observed package set equals
the reviewed generic successor lock with equal count; Python/platform are
`3.12.10` / `Windows` / `AMD64` / `win-amd64`;
`pandas-market-calendars==5.4.0`; exchange-calendars equals the reviewed
resolved version; both source-blob matches are true; XLS/PDF probes are
`PASS`; and package-index network requests, installations, calendar-object
creations, calendar-date inspections, and protected/private reads are zero,
with `t0_run=false`. `PASS` requires `failure_code=NONE`; `FAIL` requires
`failure_code != NONE`.

### Final freeze-verification evidence

`V10_CANONICAL_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_VERIFICATION_EVIDENCE.json`
has exactly:

```text
schema_version
artifact_status
status
failure_code
frozen_v10_design_git_sha
extension_design_git_sha
reviewed_live_validation_evidence_git_sha
reviewed_live_validation_evidence_git_blob_sha1
reviewed_generic_authority_transition_git_sha
migration_authority_git_blob_sha1
generic_lock_git_blob_sha1
generic_lock_sha256
generic_lock_package_count
generic_lock_candidate_git_blob_sha1
generic_windows_validation_evidence_git_blob_sha1
generic_freeze_record_git_blob_sha1
checker_git_blob_sha1
bootstrap_git_blob_sha1
observed_package_count
live_package_set_match
environment_freeze_check
pandas_market_calendars_version
exchange_calendars_version
jpx_source_blob_match
holiday_source_blob_match
xls_probe_status
pdf_probe_status
network_requests
package_installations
environment_mutations
calendar_object_creations
calendar_dates_inspected
protected_or_private_reads
t0_run
future_protected_execution_authorized
```

```text
schema_version=V10_CANONICAL_ENVIRONMENT_SUCCESSOR_FINAL_FREEZE_VERIFICATION_EVIDENCE_V1
artifact_status=V10_SUCCESSOR_FINAL_FREEZE_VERIFICATION_EVIDENCE
status=PASS|FAIL
future_protected_execution_authorized=false
failure_code={NONE,GIT_OR_PROVENANCE_BINDING_FAILURE,LIVE_PACKAGE_SET_MISMATCH,ENVIRONMENT_FREEZE_CHECK_FAILURE,PMC_VERSION_MISMATCH,EXCHANGE_CALENDARS_VERSION_MISMATCH,JPX_SOURCE_BLOB_MISMATCH,HOLIDAY_SOURCE_BLOB_MISMATCH,XLS_PROBE_FAILURE,PDF_PROBE_FAILURE,UNAUTHORIZED_OPERATION_OBSERVED}
```

For `PASS`, all Git/blob/hash bindings are exact,
`live_package_set_match=true`, `environment_freeze_check=PASS`,
`pandas-market-calendars==5.4.0`, exchange-calendars equals the reviewed
resolved version, both source-blob matches are true, XLS/PDF probes are
`PASS`, and network requests, installations, environment mutations,
calendar-object creations, calendar-date inspections, protected/private reads
are zero with `t0_run=false` and
`future_protected_execution_authorized=false`. `PASS` requires
`failure_code=NONE`; `FAIL` requires `failure_code != NONE`.

### Relational bindings

A `PASS` resolution-evidence hash refers to the exact canonical bytes of its
lock candidate. Migration authority binds the exact GPT-reviewed
candidate/evidence commit and blobs. Live validation binds the exact reviewed
migration-authority transition and generic successor lock. Final verification
binds the exact GPT-reviewed live-validation evidence and final generic
artifacts. No artifact claims a later-stage status before that stage occurs,
and a `FAIL` artifact never satisfies a `PASS` prerequisite.

## 6. Generic installation-authority transition

Before canonical mutation, the current generic live-observation/freeze
artifacts are immutable predecessor evidence for the observed 15-package
canonical environment. They must not be rewritten to claim a not-yet-observed
successor environment:

```text
REAL_EXECUTION_ENVIRONMENT_LOCK_CANDIDATE.json
REAL_EXECUTION_ENVIRONMENT_WINDOWS_VALIDATION_EVIDENCE.json
REAL_EXECUTION_ENVIRONMENT_FREEZE_RECORD.json
```

The V10 successor candidate is not installation authority merely because it
exists. After candidate/evidence exact-SHA GPT PASS, a separate reviewed
pre-mutation generic-authority-transition may update
`requirements-real-execution.lock.txt` to the reviewed successor package
set. That lock then becomes the reviewed generic installation authority, but
the transition is not a Windows-grounded live-environment observation.

The transition must create and bind, in later reviewed work, exactly this
additional provenance role (not in this task):

```text
V10_CANONICAL_ENVIRONMENT_GENERIC_MIGRATION_AUTHORITY.json
```

Its role is reviewed pre-mutation installation-authority provenance, not
live-environment evidence and not environment-freeze evidence. Its exact
schema fields are:

```text
schema_version
artifact_status
canonical_environment_state
frozen_v10_design_git_sha
predecessor_generic_lock_git_blob_sha1
predecessor_generic_lock_sha256
predecessor_generic_lock_package_count
predecessor_generic_lock_candidate_git_blob_sha1
predecessor_generic_freeze_record_git_blob_sha1
reviewed_v10_successor_lock_candidate_git_sha
reviewed_v10_successor_lock_candidate_git_blob_sha1
reviewed_v10_resolution_evidence_git_sha
reviewed_v10_resolution_evidence_git_blob_sha1
new_generic_lock_git_blob_sha1
new_generic_lock_sha256
new_generic_lock_package_count
live_environment_successor_match
future_protected_execution_authorized
```

Its fixed values are:

```text
schema_version=V10_CANONICAL_ENVIRONMENT_GENERIC_MIGRATION_AUTHORITY_V1
artifact_status=REVIEWED_INSTALL_AUTHORITY_NOT_LIVE_FROZEN
canonical_environment_state=V10_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED
frozen_v10_design_git_sha=8c923ed1734c6bdfe95a743cd9e15a5156d62c03
predecessor_generic_lock_git_blob_sha1=5e9d15caa822bd39e751a49cd0758db6eaf04bdf
predecessor_generic_lock_sha256=ddd505cc01ac4a3a798cdf7ed9c35b3a9e56db569a421aef98c02d013dd286b7
predecessor_generic_lock_package_count=15
live_environment_successor_match=false
future_protected_execution_authorized=false
```

All remaining Git/blob/SHA/count fields are exact mechanically derived values
from their reviewed future artifacts.

From that transition until final live-freeze PASS:

```text
CANONICAL_ENVIRONMENT_STATE=V10_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED
```

No protected, private, or research execution is permitted in that state. A
later canonical mutation installs only from the reviewed generic
`requirements-real-execution.lock.txt` using `--no-deps`; it must never
install directly from a V10 successor-candidate artifact.

Because the generic installation authority and observed live environment no
longer match in this state, `REAL_EXECUTION_ENVIRONMENT_FROZEN` must not be
accepted as current live successor readiness. Existing predecessor history
remains valid historical evidence. Only after canonical mutation may later
no-network live validation produce successor live-observation evidence. Only
the later final generic freeze/tooling closure may replace or update the three
generic live-observation/freeze artifacts above, grounded in actual canonical
Windows observation rather than the pre-mutation resolution candidate.

## 7. Later canonical mutation and validation

Canonical mutation is a separate Phase A/B/C direct-Windows operation with
fresh point-of-use authority. Neither the V10 scientific design freeze nor
the resolution/acquisition authority is reusable for it. Failure does not
authorize rollback, reset, reinstall, repair, or retry.

Before mutation, separately reviewed migration-mode bootstrap/checker tooling
must verify the predecessor live 15-package baseline, reviewed V10 successor
candidate/evidence bindings, reviewed migration-authority artifact, and the
reviewed successor generic lock. It must reject any successor live-ready or
frozen claim. The mutation command is only:

```text
.venv-real-execution\Scripts\python.exe -m pip install --no-deps -r requirements-real-execution.lock.txt
```

After mutation, no-network live validation must require all of the following
without creating a JPX calendar object or inspecting generated dates:

- exact reviewed successor package-set match;
- existing XLS synthetic readiness remains `PASS`;
- existing PDF synthetic readiness remains `PASS`;
- `pandas-market-calendars==5.4.0`;
- `exchange-calendars` equals the reviewed resolved version;
- installed `jpx.py` and `jp.py` Git-blob identities equal the frozen V10
  source blobs.

It performs no JPX calendar object creation, session/date generation or
inspection, research-data operation, T0, model, backtest, or profitability
operation.

## 8. Promotion order and continuing boundary

The required order is:

1. candidate/evidence exact-SHA GPT PASS;
2. generic authority-transition exact-SHA GPT PASS;
3. separately authorized canonical mutation;
4. no-network live validation;
5. evidence commit and GPT exact-SHA PASS;
6. final generic freeze-record/tooling closure;
7. final no-network live reverification; and
8. GPT exact-SHA PASS.

Only then may:

```text
V10_CURRENT_CANONICAL_ENVIRONMENT_V10_READY=true
REAL_EXECUTION_ENVIRONMENT_FROZEN=true
```

That promotion does not authorize V10 runtime-lock creation or semantic
calendar generation. Those stages continue only in the order frozen by the
V10 scientific design.

## 9. Current status and prohibitions

```text
V10_CANONICAL_ENVIRONMENT_EXTENSION_DESIGN=DRAFT_AWAITING_GPT_EXACT_SHA_REVIEW
V10_CURRENT_CANONICAL_ENVIRONMENT_V10_READY=false
V10_EXECUTION_AUTHORIZED=false
T0=NOT_RUN
HISTORICAL_EVALUATION=NOT_PERFORMED
FUTURE_PROFITABILITY=UNESTABLISHED
```

This task performs no package resolution, pip/network operation, environment
mutation, Python/calendar import, calendar generation/date inspection,
private/sealed access, human-gate consumption, T0, model, backtest, or
profitability calculation.

## 10. Review criteria and next stage

GPT may approve this operational design only if predecessor immutability,
canonical-interpreter-only topology, unchanged-pin constraints, future
resolution and mutation gates, candidate/artifact roles, generic transition,
validation criteria, and final promotion order are mechanically closed.

If approved, the next stage is synthetic-only implementation of the reviewed
resolution/preflight contract. No real package-resolution or environment
mutation is authorized by that implementation stage without separate
point-of-use authority.
