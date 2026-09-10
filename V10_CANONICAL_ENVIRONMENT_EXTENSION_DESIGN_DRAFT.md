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

This remediation closes only the provenance gap between the first reviewed
Windows wheelhouse and later canonical installation. It does not remediate
the separate `PROCESS_LAUNCH_FAILURE_EXIT_CODE_SCHEMA_NOT_REPRESENTABLE`
finding.

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

The exact resolution policy is:

```text
resolution_policy_id=PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1
resolver_distribution=pip
resolver_version=25.0.1
resolver_interpreter=.venv-real-execution\Scripts\python.exe
resolver_command=pip download
alternate_venv_allowed=false
package_installations_allowed=false
source_distributions_allowed=false
persistent_pip_cache_allowed=false
automatic_retry_allowed=false
package_index_id=PYPI_OFFICIAL_SIMPLE
package_index_url=https://pypi.org/simple
```

Phase A must verify the canonical interpreter and exact `pip==25.0.1`
before Phase B. The package index identity is fixed to
`PYPI_OFFICIAL_SIMPLE`; public artifacts record only that identifier, not the
URL. No alternate or extra index, trusted-host/TLS-verification bypass,
pre-release override, editable/VCS/local-project requirement, source
distribution, alternate interpreter, or alternate virtual environment is
allowed. The actual running canonical Python `3.12.10` Windows/AMD64
environment is the target; no cross-platform `--platform`, `--python-version`,
or `--abi` simulation is permitted.

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
creating an alternate virtual environment. Exactly one subprocess invocation
of that interpreter runs `python -m pip download` with this exact argument
vector and order (the durable wheelhouse is empty immediately before launch):

```text
.venv-real-execution\Scripts\python.exe -m pip download
  --dest <EMPTY_DURABLE_WHEELHOUSE>
  --only-binary=:all:
  --no-cache-dir
  --disable-pip-version-check
  --no-input
  --progress-bar off
  --retries 0
  --timeout 15
  --index-url https://pypi.org/simple
  --requirement requirements-real-execution.lock.txt
  --requirement V10_CANONICAL_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt
  --constraint requirements-real-execution.lock.txt
```

The child process environment is sanitized by removing every inherited
variable whose name begins with `PIP_`, then setting child
`PIP_CONFIG_FILE` to Windows `NUL`. Parent machine/user pip configuration is
not changed. Standard OS proxy/TLS variables are transport plumbing only and
must not change provider identity; no trusted-host or TLS-verification bypass
is allowed.

`--only-binary=:all:` is mandatory. Every completed download must be a wheel;
a source archive, local source tree, VCS reference, build operation, or
missing acceptable wheel is `SOURCE_DISTRIBUTION_REQUIRED` and terminal for
this attempt. The predecessor lock is both a requirement input and a
constraint, so the wheelhouse represents the complete prospective successor
environment. Do not use `pip install`, `pip wheel`, `pip --dry-run`, another
index, `extra-index-url`, `find-links`, or any unlisted resolver option.

The first complete Windows-grounded resolution result fixes the candidate.
No complete result is re-resolved to obtain a preferable dependency set. A
Phase B failure receives no automatic retry; Phase C returns only safe
evidence to GPT. No JPX/calendar import or execution is permitted.

After the first complete resolution is accepted, its durable wheelhouse is
the immutable reviewed input for candidate/evidence creation, GPT review,
generic migration-authority transition, and canonical mutation. No wheel may
be replaced, added, deleted, re-downloaded, or re-resolved as a silent repair.
If a later required wheel is missing or its bytes do not match the reviewed
SHA-256, execution stops fail-closed and requires a GPT decision; no
replacement is acquired under the accepted candidate.

The durable resolution root must not already exist and must not alias the
protected environment, `.venv-real-execution`, V9/V10 durable state, or
another governed path. Parent/root realpath and reparse protections follow
the applicable repository rules. Once the subprocess is launched,
`human_authority_consumed=true` for this attempt. A nonzero exit receives no
second invocation, retry, root deletion, reset, or recreation; partial
wheelhouse and stdout/stderr are preserved separately for Phase C. A zero
exit fixes the first complete result and permits no re-resolution.

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

### Exact scalar and nullable-field domains

Every field in each exact field list below has the following mechanical
domain. `schema_version`, `artifact_status`, `canonical_environment_state`,
`resolution_policy_id`, and every evaluated package/name/version/platform
string are nonempty JSON strings. `status` is exactly `PASS` or `FAIL`, and
each `failure_code` is exactly one member of that artifact's already-listed
closed enum. Every field ending `_git_sha` and every
`reviewed_*_implementation_git_sha` is a lowercase 40-hex JSON string; every
field ending `_git_blob_sha1` is a lowercase 40-hex JSON string; and every
field ending `_sha256` is a lowercase 64-hex JSON string, except where the
artifact's fail-state rule expressly permits `null` before that value is
observed or constructed.

All named booleans, including completion, creation, match, source-blob,
authorization, and operation flags, are JSON booleans only, except where the
live-validation or final-verification fail-state rules below expressly permit
`null` for an unperformed later check. Every `*_count`, every operation
counter, and every package-count field is a nonnegative JSON integer and
never a boolean, except where the fail-state rules expressly permit `null`
for a count derived from an unperformed observation. `process_exit_code` is a
JSON integer. Probe and environment-status fields use only their explicitly
defined closed enums. These domains apply to the successor lock candidate,
Windows resolution evidence, generic migration authority, live validation
evidence, and final freeze-verification evidence; no mapping, string-list, or
implicit alternate representation is permitted.

For live validation only, `observed_packages` permits `null` exclusively
under the unperformed-observation `FAIL` rule below. It is otherwise the
defined JSON array; it never uses an empty array as a stand-in for an
unperformed observation.

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
resolved_wheels
predecessor_pin_drift_count
pandas_market_calendars_version
exchange_calendars_version
```

Fixed values are:

```text
schema_version=V10_CANONICAL_ENVIRONMENT_SUCCESSOR_LOCK_CANDIDATE_V2
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
SHA. `resolution_policy_id` is exactly
`PIP_25_0_1_WINDOWS_WHEEL_DOWNLOAD_V1`.

`resolved_packages` is an array of objects containing only `name` and
`version`. Distribution names are lowercased with every maximal run of `-`,
`_`, or `.` replaced by `-`; empty or duplicate normalized names are invalid.
The array is sorted lexicographically by normalized name and
`resolved_package_count == len(resolved_packages)`. It contains all
predecessor 15 name/version pairs unchanged,
`pandas-market-calendars==5.4.0`, and exactly one `exchange-calendars` entry;
`exchange_calendars_version` equals that entry's version.

`resolved_wheels` is an array whose elements have exactly the four string
fields `name`, `version`, `filename`, and `sha256`. `name` uses the same
normalized distribution-name rule as `resolved_packages`; `version` is the
exact nonempty resolved version string; `filename` is a nonempty basename
with no path separator and a case-insensitive `.whl` suffix; and `sha256` is
the lowercase 64-hex SHA-256 of the exact wheel-file bytes. The array is
sorted lexicographically by normalized `name`. It contains exactly one wheel
for every normalized resolved package:

```text
len(resolved_wheels) == resolved_package_count
each resolved_packages {name,version} pair has exactly one matching resolved_wheels entry
```

Duplicate normalized names, duplicate filenames, missing or extra wheels,
malformed wheel names, unreadable or malformed wheel metadata, or any
package/version disagreement is `RESOLUTION_REPORT_INVALID`. The canonical
successor-candidate artifact hash therefore commits to every reviewed wheel
filename and SHA-256. Wheel inspection uses only wheel filenames and
standard-library `METADATA`/`WHEEL` parsing; no wheel or package is imported.

The package-set roles are mechanically defined as follows:

```text
PREDECESSOR_PACKAGE_SET=the existing exact reviewed 15 normalized name/version pairs
SUCCESSOR_PACKAGE_SET=the reviewed successor resolved_packages array
SUCCESSOR_DELTA_PACKAGE_SET=SUCCESSOR_PACKAGE_SET entries whose normalized name is absent from PREDECESSOR_PACKAGE_SET
```

Every predecessor package must remain present at its reviewed version. A
predecessor version change or disappearance is terminal
`PREDECESSOR_PIN_DRIFT`. The predecessor 15 packages are not reinstalled to
establish V10; their already-frozen live environment remains their
byte/provenance authority. Only `SUCCESSOR_DELTA_PACKAGE_SET` is installed
by the later V10 mutation.

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
package_index_id
package_resolution_process_invocations
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
failure_code={NONE,RESOLUTION_PROCESS_FAILURE,RESOLUTION_REPORT_INVALID,PREDECESSOR_PIN_DRIFT,REQUIRED_DISTRIBUTION_MISSING,SOURCE_DISTRIBUTION_REQUIRED,REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE,UNAUTHORIZED_INSTALLATION,UNAUTHORIZED_ALTERNATE_ENVIRONMENT}
package_index_id=PYPI_OFFICIAL_SIMPLE
```

For `PASS`, `failure_code=NONE`, `process_exit_code=0`,
`resolution_completed=true`, `candidate_artifact_created=true`,
`successor_lock_candidate_sha256` is lowercase 64-hex,
`resolved_package_count>15`, `package_installations=0`,
`alternate_venv_created=false`, `calendar_imports=0`, and
`calendar_dates_inspected=0`, and
`package_resolution_process_invocations=1`. For `FAIL`, `failure_code != NONE`,
`candidate_artifact_created=false`, `successor_lock_candidate_sha256=null`,
and `resolved_package_count=null`; an invalid/failed resolution creates no
accepted candidate. The invocation counter is a nonnegative integer, never
boolean: it is `0` before Phase B launch, exactly `1` after the single
subprocess is launched, and can never exceed `1`. A pre-launch failure is
recorded only by Phase-A/preflight mechanics, not fabricated as completed
Windows-resolution evidence. A nonzero exit or launch failure after the
attempt boundary is `RESOLUTION_PROCESS_FAILURE`; a zero exit whose
wheelhouse cannot produce the exact candidate is `RESOLUTION_REPORT_INVALID`.

For the exact resolution policy, Phase C derives candidates offline from
wheel filename plus wheel `METADATA`/`WHEEL` using reviewed standard-library
tooling only; downloaded packages are never imported. Each wheel must yield
exactly one normalized name/version and its exact file SHA-256; duplicate
names, duplicate filenames, missing/extra wheel artifacts, or
malformed/unreadable metadata are `RESOLUTION_REPORT_INVALID`. The candidate
is the exact sorted package set and `resolved_wheels` manifest represented by
the completed wheelhouse, with no silent insertion from the old environment.
No resolver fallback or second execution is permitted.

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
reviewed_successor_lock_candidate_sha256
installed_delta_packages
installed_delta_package_count
installed_delta_wheel_count
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

Successful mutation evidence additionally binds
`reviewed_successor_lock_candidate_sha256` to the exact reviewed successor
candidate whose canonical bytes contain `resolved_wheels`. This is the
wheel-manifest identity through the candidate; no alternate wheel manifest
or path-based identity is permitted. `installed_delta_packages` is an array
of objects with only `name` and `version`, normalized and sorted exactly like
`resolved_packages`; it must equal `SUCCESSOR_DELTA_PACKAGE_SET`, and its
count must equal its length. `installed_delta_wheel_count` must equal the
delta package count. These fields prove which reviewed delta was installed;
wheel hashes prove byte identity/reproducibility, not package safety.

For a `PASS` live-validation artifact, these three delta fields and
`reviewed_successor_lock_candidate_sha256` are non-null; the candidate SHA
must be the exact reviewed candidate identity, the delta array must be the
exact sorted normalized difference defined above, and both counts must be
exact nonnegative integers with the stated equalities. For a `FAIL` before
successful mutation evidence exists, the candidate SHA, delta array, and both
delta counts are `null`; after successful mutation evidence exists they are
retained mechanically even if a later live-validation check fails.

`observed_packages` is exactly the same representation as
`resolved_packages`: a JSON array whose every element is an object with only
the string fields `name` and `version`. The installed metadata distribution
name is lowercased and every maximal run of `-`, `_`, or `.` is replaced by
`-`; an empty normalized name or a duplicate normalized name is invalid. The
array is sorted lexicographically by normalized `name`, and
`observed_package_count == len(observed_packages)`. For `PASS`, the array is
exactly equal, element-for-element, to the normalized and sorted package
array mechanically derived from the reviewed generic successor lock. No
mapping or string-list alternative is permitted.

If the live package-set observation was mechanically performed,
`observed_packages` is that normalized and sorted array and
`observed_package_count == len(observed_packages)`. If an earlier terminal
failure occurred before live package-set observation,
`observed_packages=null` and `observed_package_count=null`. `[]` is permitted
only when an observation actually occurred and mechanically observed zero
distributions; it never represents `NOT_OBSERVED`.

A `FAIL` that occurs after package observation retains the mechanically
observed normalized/sorted array and its exact count even when the observed
package set later proves mismatched. `LIVE_PACKAGE_SET_MISMATCH` may be
selected only after package observation, so its `observed_packages` and
`observed_package_count` are non-null. `PROVENANCE_BINDING_FAILURE` and
`UNAUTHORIZED_OPERATION_OBSERVED` may stop before package observation, in
which case the required pair is `null`. A `PASS` requires non-null
`observed_packages` and `observed_package_count`, their exact length equality,
and the exact reviewed-lock element-for-element equality above.

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

### Fail-state and not-checked semantics

A `FAIL` artifact records only mechanically established observations. For
live validation, a version/string observation not evaluated because an
earlier terminal failure occurred is `null`; an unperformed boolean match is
`null`; and a count derived from an unperformed observation is `null`.
`xls_probe_status` and `pdf_probe_status` are each exactly one of
`PASS`, `FAIL`, or `NOT_CHECKED`. For final freeze verification,
`environment_freeze_check`, `xls_probe_status`, and `pdf_probe_status` are
each exactly one of `PASS`, `FAIL`, or `NOT_CHECKED`; later unperformed
version/string observations, boolean matches, and derived counts are
respectively `null`, `null`, and `null`.

For a `PASS` live-validation or final-verification artifact, no required
observation may be `null` and no required status may be `NOT_CHECKED`. A
Windows resolution `FAIL` before a valid completed candidate exists remains
closed as already specified: `candidate_artifact_created=false`,
`successor_lock_candidate_sha256=null`, and
`resolved_package_count=null`; no candidate may be fabricated from an
incomplete or failed report.

### Deterministic fail-closed evaluation order

For Windows resolution evidence, record the first applicable failure in this
exact order: (1) `UNAUTHORIZED_INSTALLATION`; (2)
`UNAUTHORIZED_ALTERNATE_ENVIRONMENT`; (3) `RESOLUTION_PROCESS_FAILURE`; (4)
`RESOLUTION_REPORT_INVALID`; (5) `REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE`;
(6) `PREDECESSOR_PIN_DRIFT`; (7)
`REQUIRED_DISTRIBUTION_MISSING`; (8) `SOURCE_DISTRIBUTION_REQUIRED`; (9)
`NONE` only when all `PASS` requirements hold.

For live validation evidence, record the first applicable failure in this
exact order: (1) `UNAUTHORIZED_OPERATION_OBSERVED`; (2)
`PROVENANCE_BINDING_FAILURE`; (3) `LIVE_PACKAGE_SET_MISMATCH`; (4)
`PYTHON_PLATFORM_MISMATCH`; (5) `PMC_VERSION_MISMATCH`; (6)
`EXCHANGE_CALENDARS_VERSION_MISMATCH`; (7) `JPX_SOURCE_BLOB_MISMATCH`; (8)
`HOLIDAY_SOURCE_BLOB_MISMATCH`; (9) `XLS_PROBE_FAILURE`; (10)
`PDF_PROBE_FAILURE`; (11) `NONE` only when all `PASS` requirements hold.

For final freeze-verification evidence, record the first applicable failure
in this exact order: (1) `UNAUTHORIZED_OPERATION_OBSERVED`; (2)
`GIT_OR_PROVENANCE_BINDING_FAILURE`; (3) `LIVE_PACKAGE_SET_MISMATCH`; (4)
`ENVIRONMENT_FREEZE_CHECK_FAILURE`; (5) `PMC_VERSION_MISMATCH`; (6)
`EXCHANGE_CALENDARS_VERSION_MISMATCH`; (7) `JPX_SOURCE_BLOB_MISMATCH`; (8)
`HOLIDAY_SOURCE_BLOB_MISMATCH`; (9) `XLS_PROBE_FAILURE`; (10)
`PDF_PROBE_FAILURE`; (11) `NONE` only when all `PASS` requirements hold.

Once the first applicable failure is established, later unsafe or unnecessary
checks are not required merely to populate an artifact; the null and
`NOT_CHECKED` domains above apply. No retry, repair, or alternate resolver is
created by any failure code.

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

`requirements-real-execution.lock.txt` remains the full reviewed successor
normalized name/version package-set authority, but it is not sufficient by
itself to select installation bytes. Canonical mutation authority is
conjunctive: (A) the reviewed generic successor package lock, (B) the
reviewed candidate/evidence/migration-authority bindings, (C) the exact
`resolved_wheels` filename/SHA-256 manifest committed by the reviewed
candidate, and (D) the exact local wheel bytes passing the pre-install hash
gate.

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

`new_generic_lock_package_count` exactly equals the reviewed V10 successor
lock candidate's `resolved_package_count`. `new_generic_lock_sha256` and
`new_generic_lock_git_blob_sha1` identify the exact canonical
`requirements-real-execution.lock.txt` bytes whose normalized name/version
package set is exactly the reviewed successor lock-candidate package set.
`live_environment_successor_match` remains `false` in this pre-mutation
artifact. This migration-authority artifact can never satisfy live-ready or
frozen-environment evidence.

From that transition until final live-freeze PASS:

```text
CANONICAL_ENVIRONMENT_STATE=V10_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED
```

No protected, private, or research execution is permitted in that state. A
later canonical mutation uses the reviewed generic lock only as the complete
successor package-set authority and installs no predecessor package. It must
install only the exact validated local wheel paths corresponding to
`SUCCESSOR_DELTA_PACKAGE_SET`; it must never install directly from a V10
successor-candidate artifact or reacquire package bytes from an index.

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
frozen claim. It must also verify, before mutation, that the predecessor live
environment exactly matches its reviewed 15-package baseline, the generic
successor lock equals `SUCCESSOR_PACKAGE_SET`, the candidate/evidence/
migration-authority SHA bindings pass, the preserved wheelhouse contains
exactly every `resolved_wheels` file and no extra selectable wheel, every
wheel SHA-256 recomputes exactly, and every delta wheel identity equals
`SUCCESSOR_DELTA_PACKAGE_SET`.

Immediately before launching pip, it must recompute the SHA-256 of every
delta wheel. Any missing, extra, tampered, unparseable, or otherwise
mismatched reviewed wheel is `REVIEWED_WHEELHOUSE_INTEGRITY_FAILURE` and
stops before mutation. No authority is consumed and no automatic repair,
replacement download, or re-resolution is permitted. The exact local path
list is constructed mechanically from the reviewed manifest and preserved
wheelhouse; no package name/version specifier may cause pip to select another
artifact. The only permitted mutation semantics are equivalent to:

```text
.venv-real-execution\Scripts\python.exe -m pip install --no-deps --no-index <EXACT_REVIEWED_LOCAL_DELTA_WHEEL_PATHS>
```

`NETWORK_REQUESTS=0` is required for canonical mutation. `--find-links` is
not used as an artifact search mechanism, and no PyPI/index/fallback path is
allowed.

After mutation, no-network live validation must require all of the following
without creating a JPX calendar object or inspecting generated dates:

- exact reviewed successor package-set match;
- existing XLS synthetic readiness remains `PASS`;
- existing PDF synthetic readiness remains `PASS`;
- `pandas-market-calendars==5.4.0`;
- `exchange-calendars` equals the reviewed resolved version;
- successful mutation evidence binds the reviewed successor candidate SHA,
  its `resolved_wheels` manifest, the exact normalized delta package set, and
  the exact delta wheel count;
- installed `jpx.py` and `jp.py` Git-blob identities equal the frozen V10
  source blobs.

It performs no JPX calendar object creation, session/date generation or
inspection, research-data operation, T0, model, backtest, or profitability
operation.

## 8. Promotion order and continuing boundary

The required order is:

1. candidate/evidence exact-SHA GPT PASS;
2. generic authority-transition exact-SHA GPT PASS;
3. separately authorized pre-mutation exact-wheel integrity gate;
4. separately authorized canonical mutation;
5. no-network live validation;
6. evidence commit and GPT exact-SHA PASS;
7. final generic freeze-record/tooling closure;
8. final no-network live reverification; and
9. GPT exact-SHA PASS.

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
