# V9 Stage A — Checkpoint A implementation

```text
status=IMPLEMENTED_AWAITING_GPT_REVIEW
reviewed_parent_sha=94af3c50bf40857b8445c06098fa2b70c55e365a
F5_auxiliary_nonblocking_amendment=PASS
```

Checkpoint A adds a deterministic, offline-only 648-cell integration seam.
It consumes the reviewed F2/F4, F3, F7 acquisition-result objects and the
reviewed generic F6 successor coverage result without invoking any
acquisition helper, fetcher, clock, or parser. F2 bridge and F7 envelope
objects remain mandatory outside the base matrix and are never inserted as
base cells. F3 preserves the reviewed YEAR-to-twelve-month references.

The F6 seam accepts a locked GLOBAL slot plus an accepted exact partition of
required years 2017--2025. Covered years fan that one slot to twelve cells;
missing years remain truthful `MISSING`. No observed production outcome is
embedded and no F6 raw data is read or refetched.

F5 has no selector or acquisition implementation. Its 108 base cells remain
`MISSING` with empty references unless a future caller supplies independently
valid raw-lock coverage. The required-missing count now uses only the generic
`LocatorStrategy.auxiliary` attribute: F5 stays diagnostic/nonblocking while
every non-auxiliary `MISSING` still blocks.

`ACQUISITION_IMPLEMENTATION_COMPLETE=true` records that mandatory acquisition
responsibilities are reviewed and implemented. It is intentionally distinct
from `OVERALL_STAGE_A_IMPLEMENTATION_READY=false`. `run_stage_a()` verifies
locator completeness, acquisition readiness, then the overall readiness guard
before any filesystem, git, clock, or network effect; the final guard maps to
`CHATGPT_DECISION_REQUIRED` and cannot be bypassed by confirmation.

No Stage-A execution, network request, production raw/private/sealed read,
human-gate consumption, F5 acquisition, parser integration, or methodology
change occurred in this checkpoint.
