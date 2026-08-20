# V8G Locator Execution Incidents

This record separates operational wrapper and environment incidents from the
scientific V8G locator result. No PRE_GATE failure below consumed the V8G
authorization. The final post-gate private-read count is not reported as zero;
it remains unknown/not safely reported.

## Chronological record

1. **Operational — SSH/Tailscale continuation prompt.** A long PowerShell
   command pasted through SSH/Tailscale entered the continuation prompt `>>`.
   Execution was aborted and disconnected. Read-only durable checks proved
   `receipt=false` and `artifact=false` before continuation. This was a
   PRE_GATE failure and did not consume authorization.

2. **Operational — stale local checkout.** The local checkout was initially
   on the V8F branch/old HEAD while the authoritative remote V8G branch was
   `928ab622e0a3d1cd34f021e0248bf725d9cf2e66`. It was safely synchronized
   before proceeding. No gate or private read occurred.

3. **Operational — malformed read-only path check.** One read-only path check
   was malformed (`$env\...` rather than the ProgramData location). It was
   corrected using absolute ProgramData paths and was not used as authority.

4. **Operational — WindowsApps interpreter launch.** The initial launcher
   selected `WindowsApps python.exe` and failed with Access Denied before
   runner execution. The durable check afterward proved
   `receipt=false`, `report=false`, and `artifact=false`. This was a PRE_GATE
   failure and did not consume authorization.

5. **Operational — `.pytest_cache` ACL.** `.pytest_cache` had abnormal
   Windows ACLs and caused `git status` to fail with Permission Denied.
   Non-admin `takeown` failed at PRE_GATE; admin-only cache ownership/ACL
   repair and deletion succeeded. No research file was changed. This was an
   operational preflight repair, not gate consumption.

6. **Operational — pinned interpreter verification.** The explicit Python
   3.12.10 interpreter at
   `C:\Users\taiki\AppData\Local\Programs\Python\Python312\python.exe`
   was verified. Module import passed and the receipt key matched.

7. **Operational — final preflight.** Branch, local HEAD, and remote HEAD
   exactly matched `928ab622e0a3d1cd34f021e0248bf725d9cf2e66`; the working tree
   was clean; runner SHA was
   `bd221e81fd674c11afb4f8ba72eb3a78cebbafe90593ea4fbc85786ddb2b38e3`;
   the gate was unconsumed; and pre-gate private reads were `0`. Preflight
   passed.

8. **Scientific V8G result — terminal locator block.** The one-shot
   production locator consumed the V8G gate and returned
   `V8G_LOCATOR_ZERO_MATCHING_CANDIDATES` / `BLOCK`. This is the scientific
   V8G locator result, not an operational shell error. The gate is consumed,
   the disposition is `BLOCK_CLOSED`, and there is no same-study retry,
   reset, rescan, or candidate substitution. The final private-read count is
   `UNKNOWN_NOT_SAFELY_REPORTED`.

## Lessons and future runbook notes

- One-shot private/gated operations should run directly in Windows
  PowerShell, not by long interactive paste through SSH.
- Pin the exact real Python interpreter; never rely on WindowsApps alias
  resolution.
- Verify git-status accessibility and temporary-cache ACLs before gated
  preflight.
- After every PRE_GATE wrapper failure, verify durable receipt absence
  read-only before any same-stage continuation.
- Once a receipt exists, absolutely no rerun, reset, rescan, or substitution.
- Operational wrapper failures must not be misclassified as
  strategy/profitability failures.
