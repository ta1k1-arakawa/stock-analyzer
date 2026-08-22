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
reimplementation), and durable-publication readiness by directly invoking
the real production exclusive/atomic publication primitive
(`src.v8i_source_snapshot._atomic_publish_once` -- staging write,
mandatory `os.fsync` of the file, atomic no-overwrite `os.link`, then a
best-effort directory fsync) on a disposable probe path that is
mechanically proven never to overlap real gate/receipt/evidence state. This
proves, not merely asserts, exclusive/no-overwrite creation, that a second
publication to the same destination correctly fails, durable byte
round-trip, and cleanup of the disposable probe artifact only -- an
ordinary write/read/unlink on an unrelated temp file would not exercise any
of that. See its module docstring for the exact safe result contract.

Precise durability semantics, stated exactly as the code behaves:

```text
file_fsync_mandatory_in_atomic_publish_once     = true
directory_fsync_attempted_best_effort           = true
directory_fsync_guaranteed_on_every_platform    = false
```

The file-level `os.fsync` is unconditional inside `_atomic_publish_once`.
The directory fsync goes through the production `_fsync_directory()`
helper, which returns silently when the platform cannot `os.open()` a
directory -- so a passing probe proves that code path executed, not that a
directory-entry fsync actually reached the disk on every OS. This document
does not claim otherwise.

### Operational JPX `.xls` parser probe -- RESOLVED

```text
former_open_item = "CHATGPT_DECISION_REQUIRED: REAL_EXECUTION_XLS_SYNTHETIC_FIXTURE_STRATEGY"
status           = RESOLVED_BY_BINDING_GPT_DECISION
resolution       = COMMITTED_SYNTHETIC_LEGACY_XLS_FIXTURE
```

The former open item is resolved by binding GPT methodology/operational
decision: this repository now commits an entirely synthetic, non-sensitive
legacy `.xls` fixture used solely for pre-gate environment readiness.

```text
fixture_path   = "tests/fixtures/synthetic_jpx_source_snapshot.xls"
fixture_sha256 = "ca47744896a286e1c56d4d0c09260775772c7df0c01b80d81b7e9a515e6d6aa7"
fixture_format = "legacy OLE2/BIFF .xls (verified D0CF11E0A1B11AE1 signature)"
```

```text
finding_resolved = "REAL_EXECUTION_XLS_FIXTURE_MEDIUM_1_REAL_TICKER_COLLISION"
```

An earlier revision of this fixture (SHA-256
`c51e3a766534820529a8946bec5c2093d7c90c593ccf0e99556b91d539cbd7cb`) used
plain `9001`-`9007`-style numeric placeholder codes, which collide with real
JPX security codes (`9001` and `9003` are real listed-instrument
identities) and therefore did not satisfy "no real ticker identities." That
SHA-256 is superseded and is **not** current canonical identity. Every code
is now drawn from an unmistakably artificial `ZZ`-prefixed namespace
(`ZZA1`, `ZZB2`, `ZZC3`, `ZZD4`, `ZZE5`, `ZZG6`, `ZZF7`, `ZZZZ8`) -- visibly
synthetic from the code value itself, not merely from the paired company
name -- and mechanically enforced: both the generator module (at import
time, via `_assert_synthetic_namespace`) and the readiness checker (before
parsing) fail if any fixture code does not start with the `ZZ` prefix, so a
later edit cannot silently reintroduce an ordinary numeric JPX-looking code.

**Completely synthetic and non-sensitive.** The fixture contains no real JPX
payload, no real or private ticker membership, no prices, and no outcomes.
Every row pairs a `ZZ`-namespace code with a `SYNTHETIC_*` name and
`SYNTHETIC_SECTOR_*` industry; none is asserted to be, or derived from, any
real listed instrument. Its columns are exactly the minimum the real parser
path needs -- `コード`, `銘柄名`, `市場・商品区分` (which satisfies the
production `_find_column(..., ("市場", "区分"))` detection), and
`33業種区分`.

It carries 8 synthetic rows: 5 that the production filter must accept
(`ZZA1`, `ZZB2`, `ZZC3`, `ZZD4`, `ZZE5`), plus one row per exclusion branch
-- non-prime/standard (`ZZG6`), non-domestic (`ZZF7`), and a
non-four-character code (`ZZZZ8`) -- so the probe exercises both the accept
and reject paths rather than merely proving that parsing returned
something.

**The checker mechanically verifies the fixture SHA-256 before parsing.**
`check_jpx_xls_parser_synthetic_probe` computes the committed fixture
bytes' SHA-256 and compares it against `EXPECTED_FIXTURE_SHA256` (recorded
in `scripts/generate_synthetic_jpx_xls_fixture.py`, the single source of
truth for the canonical identity) before calling any production parser
function. A mismatch -- tampered bytes, a stale recorded SHA after
regeneration, or any other divergence -- fails the probe immediately as
`FIXTURE_SHA256_MISMATCH`, rather than silently parsing whatever happens to
be on disk.

**Exact production functions exercised** by the readiness probe, in order,
with no reimplementation of `pandas.read_excel` in the checker:

1. read the committed synthetic `.xls` bytes;
2. `scripts.build_v8_partition_manifest.default_parse_source_table(raw_bytes)`;
3. `src.v8_partition.parse_eligible_universe(frame)`.

That mechanically proves the canonical `.venv` Python works, pandas
imports, `xlrd` imports, pandas can actually parse legacy `.xls` bytes, the
production `default_parse_source_table` works, and the downstream JPX
column-detection / eligible-universe reconstruction initializes
successfully. The checker verifies only safe synthetic properties (a
DataFrame was returned, the expected synthetic row count, the expected
eligible reconstruction) and performs no network request, no private read,
and no gate consumption.

**Generator (dev tool only).**

```text
generator_script       = "scripts/generate_synthetic_jpx_xls_fixture.py"
generator_dependency   = "xlwt==1.3.0"
xlwt_is_production_dep = false
```

`xlwt==1.3.0` is a fixture-generation/dev dependency ONLY and is
deliberately **not** added to `requirements-real-execution.txt`: production
*reads* `.xls` (pandas + `xlrd`) and never writes it, so nothing on the
real execution path imports `xlwt`. The production direct-dependency set is
unchanged by this decision (`pandas`, `xlrd==2.0.2`).

**Byte determinism is not claimed.** Regenerating the workbook reproduced
identical bytes across repeated runs and across fresh interpreter processes
on the platform and library versions used to author it, but this repository
has not established that `xlwt`'s output is byte-stable across other
platforms or versions. Per the governing decision, the committed, reviewed
fixture bytes and the `fixture_sha256` recorded above are therefore the
canonical identity; the generator is explanatory/reconstruction tooling
only. `python3 scripts/generate_synthetic_jpx_xls_fixture.py --check`
reports whether a fresh rebuild happens to match the committed bytes
without asserting that it must.

**This resolves the parser probe only.** A genuine
`JPX_XLS_PARSER_SYNTHETIC_PROBE=PASS` does not by itself make the
environment ready: Windows-grounded execution inside the canonical `.venv`
(§2-§3) is still required, and the Windows-grounded environment
lock/fingerprint (§7) is still required. `REAL_EXECUTION_ENVIRONMENT_FROZEN`
remains `false` until that lock exists and is independently reviewed, and no
future human-gated real execution should be authorized before then.

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
