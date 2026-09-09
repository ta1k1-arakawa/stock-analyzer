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

## 6. Generic installation-authority transition

The V10 successor candidate is not installation authority merely because it
exists. After candidate/evidence exact-SHA GPT PASS, a separate reviewed
generic-authority-transition commit must make the reviewed successor package
set the contents and provenance authority of
`requirements-real-execution.lock.txt` and update all mechanically coupled
generic candidate, checker, bootstrap, tests, and documentation.

From that transition until final live-freeze PASS:

```text
CANONICAL_ENVIRONMENT_STATE=V10_SUCCESSOR_MIGRATION_IN_PROGRESS_NOT_AUTHORIZED
```

No protected, private, or research execution is permitted in that state. A
later canonical mutation installs only from the reviewed generic
`requirements-real-execution.lock.txt` using `--no-deps`; it must never
install directly from a V10 successor-candidate artifact.

## 7. Later canonical mutation and validation

Canonical mutation is a separate Phase A/B/C direct-Windows operation with
fresh point-of-use authority. Neither the V10 scientific design freeze nor
the resolution/acquisition authority is reusable for it. Failure does not
authorize rollback, reset, reinstall, repair, or retry.

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
