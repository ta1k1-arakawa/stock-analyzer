"""V7 production operations day runner.

For exactly one JPX engine day, this module orchestrates the already
accepted primitives in fail-closed order:

    load + validate the activation manifest (must be ACTIVATED)
    -> verify engine_day is on/after the activation boundary, is a JPX
       trading day, and is within the manifest's acquisition window
    -> probe existing durable-root state for this engine_day
       (nothing / fully committed / partial) and fail closed on partial
    -> if nothing exists yet: run v7_daily_acquisition once, verify the
       bundle, run v7_forward_day_processing once, verify the processed day
    -> if already fully committed: re-verify (never re-mutate) and return
       ALREADY_COMMITTED

It creates no activation manifest, authorizes nothing, and never performs a
network request itself -- every network-capable call is the caller-supplied
``opener`` threaded straight through to ``v7_daily_acquisition``.  It never
computes or exposes profit, drawdown, profit factor, win rate, or any arm
performance comparison.

Before it ever calls the acquisition opener, this module also enforces two
safety properties that belong to *this* orchestration layer rather than to
any of the primitives it calls:

* the engine_day being freshly acquired must equal the current JST calendar
  date -- forward-only means no fetching a stale, already-past day just
  because it happens to fall on/after the activation boundary; and
* a per-(durable_root, engine_day) cross-process exclusive lock (an atomic
  directory create under ``.v7_forward_operations_locks/``) is held across
  the whole state-probe-through-persist critical section, so two runner
  processes racing on the same engine_day can never both observe "nothing
  acquired yet" and both start acquiring.  A pre-existing lock is always
  treated as held -- this module never inspects its age or removes it.

That lock metadata is the one piece of durable-root filesystem mutation this
module performs directly; every study artifact (acquisitions/, days/) is
still written exclusively by the lower-layer modules it calls.
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from src.v7_activation_manifest import (
    PRODUCTION_SEED_PROVENANCE,
    SeedProvenanceExpectation,
    V7ActivationManifestBlocked,
    read_activation_manifest,
    read_seed_csv_rows,
    validate_acquisition_window,
    validate_activation_manifest_candidate,
    validate_output_root,
)
from src.v7_daily_acquisition import (
    ACQUISITIONS_DIRNAME,
    CALENDAR_COMMIT as EXPECTED_CALENDAR_COMMIT,
    COLLECTOR_COMMIT as EXPECTED_COLLECTOR_COMMIT,
    V7DailyAcquisitionBlocked,
    acquire_daily_bundle,
    verify_daily_acquisition_bundle,
)
from src.v7_forward_day_processing import (
    V7ForwardDayProcessingBlocked,
    process_forward_day,
    verify_processed_forward_day,
)
from src.v7_jpx_calendar import V7JpxCalendarBlocked, is_jpx_trading_day, load_calendar_snapshot

JST = ZoneInfo("Asia/Tokyo")
FORWARD_DAYS_DIRNAME = "days"


class V7ForwardOperationsBlocked(RuntimeError):
    """Fail-closed production operations boundary violation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _blocked(reason: str) -> V7ForwardOperationsBlocked:
    return V7ForwardOperationsBlocked(reason)


def _parse_iso_date(value: Any, field: str) -> date:
    if not isinstance(value, str):
        raise _blocked("INVALID_DATE:" + field)
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise _blocked("INVALID_DATE:" + field) from error
    if parsed.isoformat() != value:
        raise _blocked("INVALID_DATE:" + field)
    return parsed


# ---------------------------------------------------------------------------
# Activation manifest gate
# ---------------------------------------------------------------------------


def load_and_verify_activation_manifest(
    *,
    activation_manifest_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    calendar_path: str | os.PathLike[str],
    universe_csv: str | os.PathLike[str],
    seed_csv: str | os.PathLike[str],
    seed_acquisition_manifest: Mapping[str, Any],
    expected_seed_provenance: SeedProvenanceExpectation | None = PRODUCTION_SEED_PROVENANCE,
) -> dict[str, Any]:
    """Read-only load and full re-validation.  Creates and authorizes nothing."""
    try:
        manifest = read_activation_manifest(activation_manifest_path)
    except V7ActivationManifestBlocked as error:
        raise _blocked("ACTIVATION_MANIFEST_READ_FAILED:" + error.reason) from error
    try:
        validate_activation_manifest_candidate(
            manifest,
            repository_root=repository_root,
            calendar_path=calendar_path,
            universe_csv=universe_csv,
            seed_csv=seed_csv,
            seed_acquisition_manifest=seed_acquisition_manifest,
            expected_seed_provenance=expected_seed_provenance,
        )
    except V7ActivationManifestBlocked as error:
        raise _blocked("ACTIVATION_MANIFEST_VALIDATION_FAILED:" + error.reason) from error
    if manifest["activation_status"] != "ACTIVATED":
        raise _blocked("ACTIVATION_STATUS_NOT_ACTIVATED")
    return manifest


def build_processing_activation_context(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project the 48-field activation manifest onto v7_forward_day_processing's
    narrower ACTIVATION_CONTEXT_FIELDS shape.  Field *names* differ between the
    two modules for the same values; get this mapping wrong and processing
    binds to the wrong provenance silently, so it lives in exactly one place.
    """
    return {
        "activation_manifest_sha256": manifest["manifest_sha256"],
        "activation_boundary_first_jpx_trading_date": manifest["activation_boundary_first_jpx_trading_date"],
        "implementation_commit": manifest["implementation_commit"],
        "collector_commit": manifest["collector_commit"],
        "expected_seed_canonical_sha256": manifest["seed_canonical_csv_sha256"],
        "expected_seed_ticker_manifest_sha256": manifest["seed_ticker_manifest_sha256"],
    }


# ---------------------------------------------------------------------------
# Durable output root
# ---------------------------------------------------------------------------


def require_durable_output_root(
    durable_output_root: str | os.PathLike[str],
    manifest: Mapping[str, Any],
    repository_root: str | os.PathLike[str],
) -> Path:
    durable_value = str(durable_output_root)
    if durable_value != manifest["output_root"]:
        raise _blocked("DURABLE_OUTPUT_ROOT_MISMATCH")
    try:
        validate_output_root(durable_value, repository_root)
    except V7ActivationManifestBlocked as error:
        raise _blocked("DURABLE_OUTPUT_ROOT_INVALID:" + error.reason) from error
    root = Path(durable_output_root)
    if not root.is_dir():
        raise _blocked("DURABLE_OUTPUT_ROOT_NOT_FOUND")
    if not os.access(root, os.W_OK):
        raise _blocked("DURABLE_OUTPUT_ROOT_NOT_WRITABLE")
    return root


# ---------------------------------------------------------------------------
# Engine day / acquisition window gate
# ---------------------------------------------------------------------------


def require_engine_day_ready(
    engine_day: str, manifest: Mapping[str, Any], calendar_path: str | os.PathLike[str]
) -> None:
    try:
        calendar = load_calendar_snapshot(calendar_path)
    except V7JpxCalendarBlocked as error:
        raise _blocked("CALENDAR_SNAPSHOT_INVALID:" + error.reason) from error
    try:
        trading = is_jpx_trading_day(calendar, engine_day)
    except V7JpxCalendarBlocked as error:
        raise _blocked("ENGINE_DAY_OUTSIDE_CALENDAR_COVERAGE") from error
    if not trading:
        raise _blocked("ENGINE_DAY_NOT_JPX_TRADING_DAY")
    if engine_day < manifest["activation_boundary_first_jpx_trading_date"]:
        raise _blocked("ENGINE_DAY_BEFORE_ACTIVATION_BOUNDARY")


def require_within_acquisition_window(now_utc: datetime, acquisition_window_jst: str) -> None:
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None or now_utc.utcoffset() != timedelta(0):
        raise _blocked("OPERATIONS_CLOCK_INVALID")
    try:
        window = validate_acquisition_window(acquisition_window_jst)
    except V7ActivationManifestBlocked as error:
        raise _blocked("ACQUISITION_WINDOW_INVALID:" + error.reason) from error
    now_jst = now_utc.astimezone(JST)
    minutes = now_jst.hour * 60 + now_jst.minute
    if not (window["start_minutes"] <= minutes < window["end_minutes"]):
        raise _blocked("ACQUISITION_WINDOW_NOT_OPEN")


def require_current_jst_date_matches_engine_day(now_utc: datetime, engine_day: str) -> None:
    """No-historical-backfill guard for a *fresh* acquisition only.

    A COMPLETE day is re-verified (never re-acquired) regardless of when or
    from what day it is re-checked, so this must only ever be called on the
    NONE-state path, immediately before the acquisition opener is touched.
    """
    if not isinstance(now_utc, datetime) or now_utc.tzinfo is None or now_utc.utcoffset() != timedelta(0):
        raise _blocked("OPERATIONS_CLOCK_INVALID")
    current_jst_date = now_utc.astimezone(JST).date().isoformat()
    if current_jst_date != engine_day:
        raise _blocked("ENGINE_DAY_NOT_CURRENT_JST_DATE")


# ---------------------------------------------------------------------------
# Engine-day state probe (idempotence / atomicity)
# ---------------------------------------------------------------------------


def probe_engine_day_state(durable_root: str | os.PathLike[str], engine_day: str) -> str:
    """Return "NONE", "PARTIAL", or "COMPLETE" for this engine_day.

    "COMPLETE" only means both the acquisition bundle and the processed
    forward day *final directories* exist; it does not itself imply either
    one is uncorrupted -- callers must still re-verify before trusting it.
    """
    root = Path(durable_root)
    acquisition_exists = (root / ACQUISITIONS_DIRNAME / engine_day).is_dir()
    forward_exists = (root / FORWARD_DAYS_DIRNAME / engine_day).is_dir()
    if acquisition_exists and forward_exists:
        return "COMPLETE"
    if acquisition_exists or forward_exists:
        return "PARTIAL"
    return "NONE"


# ---------------------------------------------------------------------------
# Per-engine-day cross-process exclusive lock
# ---------------------------------------------------------------------------

LOCK_DIRNAME = ".v7_forward_operations_locks"


class _EngineDayLock:
    """Atomic per-(durable_root, engine_day) exclusive lock.

    Directory creation (``os.mkdir``) is atomic on every filesystem this
    project targets: exactly one competing creator succeeds, the rest get
    ``FileExistsError``.  A pre-existing lock directory -- whether held by a
    live process or abandoned by a crashed one -- is always treated as held.
    Nothing here inspects its age or removes it; only a human clearing it by
    hand can unblock a stale lock.  Cleanup on our own exit is best-effort
    (a release failure must never mask the real result/exception).
    """

    def __init__(self, durable_root: str | os.PathLike[str], engine_day: str) -> None:
        self._path = Path(durable_root) / LOCK_DIRNAME / f"{engine_day}.lock"
        self._held = False

    def __enter__(self) -> "_EngineDayLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.mkdir()
        except FileExistsError as error:
            raise _blocked("ENGINE_DAY_LOCK_HELD") from error
        except OSError as error:
            raise _blocked("ENGINE_DAY_LOCK_ACQUIRE_FAILED:" + str(error)) from error
        self._held = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._held:
            try:
                self._path.rmdir()
            except OSError:
                pass
            self._held = False
        return False


# ---------------------------------------------------------------------------
# Public one-day production runner
# ---------------------------------------------------------------------------


def run_forward_operations_day(
    *,
    activation_manifest_path: str | os.PathLike[str],
    durable_output_root: str | os.PathLike[str],
    universe_csv: str | os.PathLike[str],
    calendar_path: str | os.PathLike[str],
    seed_csv: str | os.PathLike[str],
    seed_acquisition_manifest: Mapping[str, Any],
    engine_day: str,
    repository_root: str | os.PathLike[str],
    opener: Callable[[Any], Any],
    clock: Callable[[], datetime],
    monotonic_clock: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
    expected_seed_provenance: SeedProvenanceExpectation | None = PRODUCTION_SEED_PROVENANCE,
) -> dict[str, Any]:
    _parse_iso_date(engine_day, "engine_day")

    manifest = load_and_verify_activation_manifest(
        activation_manifest_path=activation_manifest_path,
        repository_root=repository_root,
        calendar_path=calendar_path,
        universe_csv=universe_csv,
        seed_csv=seed_csv,
        seed_acquisition_manifest=seed_acquisition_manifest,
        expected_seed_provenance=expected_seed_provenance,
    )
    durable_root = require_durable_output_root(durable_output_root, manifest, repository_root)
    require_engine_day_ready(engine_day, manifest, calendar_path)

    activation_context = build_processing_activation_context(manifest)

    with _EngineDayLock(durable_root, engine_day):
        state = probe_engine_day_state(durable_root, engine_day)

        if state == "PARTIAL":
            raise _blocked("PARTIAL_ENGINE_DAY_STATE")

        if state == "COMPLETE":
            # Re-verification only: no network, no mutation, so neither the
            # current JST date nor the acquisition window applies here.
            try:
                verify_daily_acquisition_bundle(
                    durable_root, engine_day, EXPECTED_CALENDAR_COMMIT, EXPECTED_COLLECTOR_COMMIT, universe_csv
                )
            except V7DailyAcquisitionBlocked as error:
                raise _blocked("ACQUISITION_VERIFICATION_FAILED:" + error.reason) from error
            try:
                verification = verify_processed_forward_day(
                    study_root=durable_root,
                    engine_day=engine_day,
                    universe_csv=universe_csv,
                    activation_context=activation_context,
                )
            except V7ForwardDayProcessingBlocked as error:
                raise _blocked("PROCESSING_VERIFICATION_FAILED:" + error.reason) from error
            return {
                "status": "ALREADY_COMMITTED",
                "engine_day": engine_day,
                "activation_manifest_verified": True,
                "acquisition_verified": True,
                "processing_verified": True,
                "persistence_verified": True,
                "already_committed": True,
                "accepted_candidate_count": verification["accepted_candidate_count"],
                "valid_d0_count": verification["valid_d0_count"],
                "missing_d0_count": verification["missing_d0_count"],
                "control": None,
                "variant": None,
            }

        # state == "NONE": a fresh acquisition is about to happen -- gate it
        # on "today" before the opener is ever touched (no historical
        # backfill), then confirm the acquisition window, then run the
        # pipeline exactly once.
        now_utc = clock()
        require_current_jst_date_matches_engine_day(now_utc, engine_day)
        require_within_acquisition_window(now_utc, manifest["acquisition_window_jst"])

        try:
            acquire_daily_bundle(
                output_root=durable_root,
                universe_csv=universe_csv,
                calendar_snapshot=calendar_path,
                engine_day=engine_day,
                opener=opener,
                clock=clock,
                monotonic_clock=monotonic_clock,
                sleep_fn=sleep_fn,
            )
        except V7DailyAcquisitionBlocked as error:
            raise _blocked("ACQUISITION_FAILED:" + error.reason) from error

        try:
            verify_daily_acquisition_bundle(
                durable_root, engine_day, EXPECTED_CALENDAR_COMMIT, EXPECTED_COLLECTOR_COMMIT, universe_csv
            )
        except V7DailyAcquisitionBlocked as error:
            raise _blocked("ACQUISITION_VERIFICATION_FAILED:" + error.reason) from error

        try:
            rows, _raw = read_seed_csv_rows(seed_csv)
        except V7ActivationManifestBlocked as error:
            raise _blocked("SEED_READ_FAILED:" + error.reason) from error

        try:
            summary = process_forward_day(
                study_root=durable_root,
                engine_day=engine_day,
                universe_csv=universe_csv,
                calendar_snapshot=calendar_path,
                seed_rows=rows,
                activation_context=activation_context,
            )
        except V7ForwardDayProcessingBlocked as error:
            raise _blocked("PROCESSING_FAILED:" + error.reason) from error

        try:
            verify_processed_forward_day(
                study_root=durable_root,
                engine_day=engine_day,
                universe_csv=universe_csv,
                activation_context=activation_context,
            )
        except V7ForwardDayProcessingBlocked as error:
            raise _blocked("PROCESSING_VERIFICATION_FAILED:" + error.reason) from error

        return {
            "status": "PASS",
            "engine_day": engine_day,
            "activation_manifest_verified": True,
            "acquisition_verified": True,
            "processing_verified": True,
            "persistence_verified": True,
            "already_committed": False,
            "accepted_candidate_count": summary["accepted_candidate_count"],
            "valid_d0_count": summary["valid_d0_count"],
            "missing_d0_count": summary["missing_d0_count"],
            "control": summary["control"],
            "variant": summary["variant"],
        }


__all__ = [
    "EXPECTED_CALENDAR_COMMIT",
    "EXPECTED_COLLECTOR_COMMIT",
    "LOCK_DIRNAME",
    "V7ForwardOperationsBlocked",
    "build_processing_activation_context",
    "load_and_verify_activation_manifest",
    "probe_engine_day_state",
    "require_current_jst_date_matches_engine_day",
    "require_durable_output_root",
    "require_engine_day_ready",
    "require_within_acquisition_window",
    "run_forward_operations_day",
]
