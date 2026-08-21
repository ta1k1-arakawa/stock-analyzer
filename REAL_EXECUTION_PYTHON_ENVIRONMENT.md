# Real-Execution Python Environment Contract

```text
document_type=REPOSITORY_REAL_EXECUTION_ENVIRONMENT_CONTRACT
status=ACTIVE
scope=REAL_NETWORK_PRIVATE_DATA_HUMAN_GATE_DURABLE_STATE_PYTHON_ENVIRONMENT
governs_task=REPRODUCIBLE_REAL_EXECUTION_ENVIRONMENT_AND_RUNBOOK_HARDENING
```

This document is the canonical, human-readable environment contract for any
real network, private-data, human-gated, or durable-machine-state Python
execution in this repository (`AI_REAL_EXECUTION_RUNBOOK.md`'s scope). It is
operational/governance hardening only. It does not reopen any study, does
not authorize any acquisition, and does not change research methodology.
`AI_REAL_EXECUTION_RUNBOOK.md` remains the authoritative operational-safety
document; this file explains the environment piece of that contract in one
place, and `scripts/check_real_execution_env.py` is the mechanical
enforcement of it.

## 1. Canonical environment directory

```text
canonical_environment_directory = ".venv"   (repository root)
```

`.venv` is never committed to Git. `.gitignore` already ignores `.venv/`
(verified; no change was required for this task).

## 2. Canonical interpreter invocation

```text
canonical_windows_interpreter = ".venv\Scripts\python.exe"   (exact, repo-root-relative)
```

All future Windows real/network/private/human-gated Python execution MUST
invoke this exact interpreter path explicitly. It MUST NOT rely on:

- `python`
- `python3`
- `py`
- PATH activation (`.venv\Scripts\Activate.ps1`)
- whatever interpreter happens to be active in the current shell

Activation may be convenient for a human operator, but it is never the
security/provenance mechanism: an activated shell's `python` can silently
resolve to something other than `.venv\Scripts\python.exe` (a stale PATH
entry, a WindowsApps alias, a differently-versioned interpreter). Production
code must instead verify `sys.executable` itself resolves to the repository's
exact `.venv\Scripts\python.exe` before any gate/network/private boundary.
A mismatch is `PRE_GATE_WRONG_PYTHON_ENVIRONMENT` (see
`AI_REAL_EXECUTION_RUNBOOK.md` §17).

## 3. Canonical Python version

```text
canonical_python_major_minor = "3.12"
canonical_python_version_source = ".github/workflows/daily_ai_trade.yml (python-version: '3.12')"
canonical_python_version_derivation = "MECHANICALLY_DERIVED_FROM_EXISTING_REPOSITORY_CI"
```

This repository already establishes Python 3.12 as its supported major/minor
version, in `.github/workflows/daily_ai_trade.yml`'s
`actions/setup-python@v4` step (`python-version: '3.12'`). Per the task's own
"Python version decision discipline," this existing repository evidence is
reused rather than a version being freshly invented because of whatever
Python happens to be installed on a given Windows machine (e.g. a
locally-installed 3.14.5 must not be silently adopted as canonical).

The exact resolved patch version (e.g. `3.12.x`) is whatever the real Windows
`py -3.12` launcher resolves to at bootstrap time. That exact patch version is
recorded, not fabricated, by `scripts/bootstrap_real_execution_env.ps1` and by
the future Windows-grounded environment lock (§7).

## 4. Direct real-execution dependencies

```text
direct_dependency_specification = "requirements-real-execution.txt"
```

Mechanically inspecting the complete import closure reachable from the real
V8-lineage source-snapshot execution path
(`src/v8i_source_snapshot.py` → `src/v8_partition.py`,
`src/v8c_git_provenance.py` → `src/v8b_git_provenance.py`,
`src/v8c_human_gate_consumption.py`; and the production parser reference
implementation, `scripts/build_v8_partition_manifest.py`'s
`default_parse_source_table`), the only non-stdlib direct dependencies are:

```text
pandas   (unpinned; see §7 -- no authoritative pandas version is
          established anywhere in this repository yet, and this task
          deliberately does not invent one)
xlrd==2.0.2   (pinned; reused verbatim from the repository's own existing
               requirement.txt -- not invented here)
```

Everything else reachable from that path is Python standard library only:
`hashlib`, `json`, `os`, `re`, `subprocess`, `datetime`, `pathlib`, `typing`,
`urllib.request`, `urllib.parse`, `urllib.error`, `ctypes`, `tempfile`,
`argparse`, `sys`.

This is deliberately narrower than the top-level `requirement.txt`, which
also serves the unrelated daily automated trading bot (`lightgbm`,
`scikit-learn`, `scipy`, `joblib`, `PyYAML`, `requests`, `python-dotenv`,
`pytest`, ...). None of those are imported anywhere on the real V8-lineage
source-snapshot path, and the task explicitly requires the real-execution
specification to reflect the real path, not merely what unit tests import.
Notably, `requests` is *not* on this path either: the real JPX fetch
reference implementation
(`scripts/build_v8_partition_manifest.py`'s `fetch_real_jpx_source`) uses
only `urllib.request` with a trusted-host redirect handler
(`TrustedJpxRedirectHandler`), not `requests`.

## 5. Required JPX Excel engine

```text
required_excel_engine = "xlrd"
required_excel_engine_version = "2.0.2"
required_excel_engine_derivation = "MECHANICALLY_DERIVED_FROM_EXISTING_REPOSITORY_EVIDENCE"
```

The official JPX listing is `data_j.xls` (see
`scripts/build_v8_partition_manifest.py`'s `DATA_LINK_PATTERN`,
`SYNTHETIC_SOURCE_URL`) -- the legacy OLE2/BIFF `.xls` binary format, not
`.xlsx`. `default_parse_source_table` calls
`pandas.read_excel(io.BytesIO(raw_bytes), dtype=str)` with no explicit
`engine=` argument; pandas selects the engine by sniffing the byte content
(OLE2 compound-file signature vs. `.xlsx`'s zip/OOXML signature), and reading
that OLE2/BIFF signature requires the `xlrd` package to be installed. Because
`xlrd >= 2.0` intentionally dropped `.xlsx` support and reads *only* the
legacy `.xls` format, `xlrd==2.0.2` -- already pinned in this repository's
own `requirement.txt` -- is exactly the correct, already-established engine
version for this real path. This task reuses that existing pin verbatim; it
does not invent a new one.

## 6. Bootstrap and readiness procedure

```text
bootstrap_script  = "scripts/bootstrap_real_execution_env.ps1"
readiness_checker = "scripts/check_real_execution_env.py"
```

`scripts/bootstrap_real_execution_env.ps1` is a fail-closed, environment-only
PowerShell script: it creates or verifies `.venv`, installs only from
`requirements-real-execution.txt` via
`.venv\Scripts\python.exe -m pip ...`, and finally runs the readiness
checker. It never consumes a human research gate, never calls JPX/Yahoo,
never accesses private/sealed data, and never executes any V8I/V8J real
acquisition.

`scripts/check_real_execution_env.py` is a no-network, no-private-data
readiness checker. It verifies interpreter identity, dependency
presence/version, an *operational* parser probe (not merely `import
pandas`), TLS/stdlib initialization, trusted-host request-construction
initialization (reusing the real
`scripts/build_v8_partition_manifest.py` production code, not a
reimplementation), and filesystem/durable-publication primitive usability on
a disposable probe path that never touches real gate/receipt/evidence state.
See its module docstring for the exact safe result contract.

### Operational JPX `.xls` parser probe -- open item

```text
CHATGPT_DECISION_REQUIRED: REAL_EXECUTION_XLS_SYNTHETIC_FIXTURE_STRATEGY
```

The readiness checker's JPX `.xls` operational probe currently reports
`CHATGPT_DECISION_REQUIRED` rather than `PASS`, because this repository has
no genuine synthetic `.xls` (OLE2/BIFF) binary fixture, and none can be
mechanically produced without either (a) adding a new dependency capable of
*writing* legacy `.xls` bytes (e.g. `xlwt`, which pandas itself no longer
bundles as a `to_excel` writer engine), or (b) hand-constructing raw
OLE2/BIFF bytes by hand, which risks silently producing an unrepresentative
or invalid fixture that would give false confidence rather than real proof.
This exact gap already exists in the repository today:
`scripts/build_v8_partition_manifest.py`'s own synthetic-test path
(`run_synthetic_partition_test`) explicitly injects a fake
DataFrame-returning callable in place of `default_parse_source_table`,
precisely because, in its own words, that function "depends on real
spreadsheet bytes." This task does not unilaterally resolve that gap by
picking a new dependency on its own initiative -- per this task's own
explicit instruction, it stops with `CHATGPT_DECISION_REQUIRED` here instead.
Until this is resolved and independently reviewed, the readiness checker
cannot report `REAL_EXECUTION_ENVIRONMENT_READY=true`, and no future
human-gated real execution should be authorized on the strength of this
environment contract alone.

## 7. Exact environment lock/fingerprint

```text
REAL_EXECUTION_ENVIRONMENT_FROZEN = false
```

This task (Claude Code Cloud) does not and cannot produce a Windows-grounded
package-resolution lock. A later, separate, explicitly reviewed task must run
on the real target Windows machine, using the canonical interpreter:

```powershell
.venv\Scripts\python.exe -m pip freeze --all
```

and commit the exact resolved package set as a dedicated environment
lock/fingerprint artifact, subject to its own GPT exact-SHA independent
review, before any future real execution is authorized. Until that
Windows-grounded lock exists and is independently reviewed,
`REAL_EXECUTION_ENVIRONMENT_FROZEN` remains `false` and no future
human-gated real execution should be authorized, regardless of what the
readiness checker itself reports.

## 8. Authorization ordering

No real human authorization may be requested or accepted before environment
readiness `PASS`. The full required ordering is frozen in
`AI_REAL_EXECUTION_RUNBOOK.md` §16.

## 9. V8I permanence (prospective only)

This environment contract is prospective infrastructure only. It does not:

- reopen V8I;
- authorize a V8I retry or a second JPX request;
- reset, delete, or reuse the V8I `HUMAN_V8I_SOURCE_SNAPSHOT_ACQUISITION_GATE`
  receipt;
- reuse the V8I human authorization;
- reconstruct or preserve the lost V8I raw bytes; or
- change V8I's `BLOCK_CLOSED` disposition.

V8I remains permanently `BLOCK_CLOSED`, exactly as recorded in
`V8I_SOURCE_SNAPSHOT_TERMINAL_ADJUDICATION.json` and
`V8I_SOURCE_SNAPSHOT_EXECUTION_INCIDENTS.md`. Any future V8-lineage
source-snapshot attempt (e.g. under a successor study such as V8J) is a
fresh, independent study identity with its own fresh gate, receipt key, and
authorization grammar; this contract merely ensures that a future attempt
cannot again discover a missing software dependency only after its own gate
is consumed.
