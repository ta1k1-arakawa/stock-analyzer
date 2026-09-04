"""V9_014 PDF real-execution environment successor -- Stage E2 offline
tooling.

Implements ONLY Stage E2 of
`V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_DESIGN.md` Section 5:
deterministic, offline validation/construction helpers needed by the
LATER, separately reviewed Stages E3-E7 (staging preflight, staging
resolution, staging inspection, candidate/evidence commit) and the
synthetic PDF operational-readiness probe used at Stages E6/E10/E14
(Section 5a). It does not perform, and cannot by itself trigger, any of
those later stages.

This module is pure and offline for every function EXCEPT
`run_synthetic_pdf_operational_probe`, which lazily imports `pdfplumber`
only when explicitly called (never at import time, never as a side effect
of importing this module or of any other function here). No other
function in this module reads a package index, resolves a dependency,
installs anything, or creates/mutates any real or staging Python
environment. `.venv-real-execution` (the canonical predecessor/successor
environment) and any future staging venv are never touched by this
module -- Stage E2 performs no environment creation or mutation of any
kind.

Frozen scope, per the reviewed design:
  - the successor direct-spec candidate (`validate_direct_spec_bytes`,
    bound to `DIRECT_SPEC_PATH`)
  - strict `pip freeze --all` parsing (`parse_pip_freeze_all`)
  - predecessor-baseline and successor-after-resolution validation
    against the seven frozen predecessor pins plus
    `pdfplumber==0.11.10` (`validate_predecessor_baseline`,
    `validate_successor_after_resolution`)
  - before/after delta derivation with predecessor-pin-drift detection
    (`compute_before_after_delta`)
  - Windows/platform evidence schema validation
    (`validate_platform_evidence`)
  - deterministic construction/validation helpers for the FUTURE Stage E7
    lock-candidate and Windows-evidence artifacts
    (`build_lock_candidate_payload`, `build_windows_evidence_payload`,
    `validate_lock_candidate_schema`, `validate_windows_evidence_schema`)
    -- called only by a later, separately reviewed stage; E2 does not
    invoke them to write any artifact
  - staging-path validation (`validate_staging_path`)
  - the synthetic PDF operational-readiness probe
    (`run_synthetic_pdf_operational_probe`)

This module does not resolve V9_014 design LOW_1, does not authorize any
SOURCE_B PDF calibration acquisition, does not materialize
`trading_dates`, and makes no profitability claim of any kind.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Section 3: successor direct dependency spec ----------------------------
DIRECT_SPEC_PATH = REPO_ROOT / "V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_DIRECT_SPEC.txt"
EXPECTED_DIRECT_SPEC_BYTES = b"pandas\nxlrd==2.0.2\npdfplumber==0.11.10\n"

# --- Section 3a / Section 1: frozen predecessor pins (constraints, never
# altered by successor resolution) and the sole new direct dependency ------
PREDECESSOR_PINS: dict[str, str] = {
    "numpy": "2.5.2",
    "pandas": "3.0.5",
    "pip": "25.0.1",
    "python-dateutil": "2.9.0.post0",
    "six": "1.17.0",
    "tzdata": "2026.3",
    "xlrd": "2.0.2",
}
PREDECESSOR_PIN_COUNT = 7
assert len(PREDECESSOR_PINS) == PREDECESSOR_PIN_COUNT

NEW_DIRECT_PACKAGE_NAME = "pdfplumber"
NEW_DIRECT_PACKAGE_VERSION = "0.11.10"

# --- Section 2: canonical/staging environment identity ----------------------
CANONICAL_ENVIRONMENT_DIRECTORY_NAME = ".venv-real-execution"
GENERAL_PROJECT_ENVIRONMENT_DIRECTORY_NAME = ".venv"

# --- Predecessor freeze-record platform binding (Section 1) -----------------
CANONICAL_PYTHON_IMPLEMENTATION = "CPython"
CANONICAL_PYTHON_EXACT_VERSION = (3, 12, 10)
CANONICAL_PLATFORM_SYSTEM = "Windows"
CANONICAL_PLATFORM_MACHINE = "AMD64"
CANONICAL_SYSCONFIG_PLATFORM = "win-amd64"
CANONICAL_OS_NAME = "nt"


# =============================================================================
# Section 3: direct-spec byte validation
# =============================================================================


@dataclass(frozen=True)
class DirectSpecValidation:
    status: str
    actual_bytes: Optional[bytes] = None


DIRECT_SPEC_OK = "DIRECT_SPEC_OK"
DIRECT_SPEC_MISMATCH_FAILURE = "DIRECT_SPEC_MISMATCH_DQ_FAILURE"


def validate_direct_spec_bytes(data: bytes) -> DirectSpecValidation:
    """Exact byte-for-byte comparison against the frozen direct-spec
    content (UTF-8/LF, trailing LF, no comments, no extra dependency).
    Any deviation -- a comment, a reordered line, an extra blank line, a
    CRLF line ending, a missing trailing newline, an added/removed
    dependency -- fails closed.
    """
    if data == EXPECTED_DIRECT_SPEC_BYTES:
        return DirectSpecValidation(status=DIRECT_SPEC_OK)
    return DirectSpecValidation(status=DIRECT_SPEC_MISMATCH_FAILURE, actual_bytes=data)


# =============================================================================
# Strict `pip freeze --all` parsing (reused pattern/discipline from
# `scripts/check_real_execution_env.py`'s own strict freeze parser -- not
# imported from it, so this module's successor-only tooling stays fully
# independent of the canonical checker per the reviewed design's Section 4
# distinct-identity requirement; reimplemented here rather than shared).
# =============================================================================

_SEPARATOR_RUN_PATTERN = re.compile(r"[-_.]+")


def _normalize_package_name(name: str) -> str:
    """PEP 503 normalization: collapse any run of `-`, `_`, `.` into a
    single `-`, then lowercase.
    """
    return _SEPARATOR_RUN_PATTERN.sub("-", name.strip()).lower()


# Accepts ONLY the exact `name==version` form `pip freeze --all` emits for
# a normal installed distribution -- rejects, by construction, direct-URL
# forms (`name @ file://...`), editable/VCS forms (`-e ...`,
# `git+https://...`), and any other malformed or non-pinned line.
_EXACT_PIN_LINE_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)==(?P<version>[A-Za-z0-9](?:[A-Za-z0-9._+!-]*[A-Za-z0-9])?)$"
)


@dataclass(frozen=True)
class PipFreezeParseResult:
    packages: Mapping[str, str]
    invalid_lines: Sequence[str]
    duplicate_lines: Sequence[str]

    @property
    def is_clean(self) -> bool:
        return not self.invalid_lines and not self.duplicate_lines


def parse_pip_freeze_all(text: str) -> PipFreezeParseResult:
    """Strictly parse live `pip freeze --all` output.

    Returns `{normalized_name: exact_version}` for every exact
    `name==version` line, plus every non-empty, non-comment line that is
    NOT an exact pinned entry (`invalid_lines`) and every line whose
    normalized name already appeared earlier (`duplicate_lines`) --
    neither is ever silently dropped. Callers must treat any non-empty
    `invalid_lines`/`duplicate_lines` as a hard failure of the exact-set
    check, never merely omit them from comparison.
    """
    packages: dict[str, str] = {}
    invalid_lines: list[str] = []
    duplicate_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _EXACT_PIN_LINE_PATTERN.match(stripped)
        if match is None:
            invalid_lines.append(stripped)
            continue
        normalized_name = _normalize_package_name(match.group("name"))
        if normalized_name in packages:
            duplicate_lines.append(stripped)
            continue
        packages[normalized_name] = match.group("version")
    return PipFreezeParseResult(packages=packages, invalid_lines=tuple(invalid_lines), duplicate_lines=tuple(duplicate_lines))


# =============================================================================
# Predecessor-baseline / successor-after-resolution validation
# =============================================================================


@dataclass(frozen=True)
class BaselineValidation:
    status: str
    missing: Sequence[str] = ()
    version_mismatches: Mapping[str, str] = field(default_factory=dict)
    unexpected_extra: Sequence[str] = ()


BASELINE_OK = "PREDECESSOR_BASELINE_OK"
BASELINE_FAILURE = "PREDECESSOR_BASELINE_DQ_FAILURE"


def validate_predecessor_baseline(freeze_result: PipFreezeParseResult) -> BaselineValidation:
    """Stage E5 step 3: after installing the reviewed predecessor lock
    with `--no-deps` and nothing else, `pip freeze --all` must show
    EXACTLY the 7 frozen predecessor pins -- no more, no less, no drift.
    Fails closed on any invalid/duplicate freeze line, any missing pin,
    any version mismatch, or any unexpected extra package.
    """
    if not freeze_result.is_clean:
        return BaselineValidation(status=BASELINE_FAILURE)

    packages = freeze_result.packages
    missing = tuple(name for name in PREDECESSOR_PINS if name not in packages)
    version_mismatches = {
        name: packages[name]
        for name in PREDECESSOR_PINS
        if name in packages and packages[name] != PREDECESSOR_PINS[name]
    }
    unexpected_extra = tuple(name for name in packages if name not in PREDECESSOR_PINS)

    if missing or version_mismatches or unexpected_extra:
        return BaselineValidation(
            status=BASELINE_FAILURE,
            missing=missing,
            version_mismatches=version_mismatches,
            unexpected_extra=unexpected_extra,
        )
    return BaselineValidation(status=BASELINE_OK)


BASELINE_FILE_MISSING_FAILURE = "PREDECESSOR_BASELINE_FILE_MISSING_DQ_FAILURE"
BASELINE_FILE_DECODE_FAILURE = "PREDECESSOR_BASELINE_FILE_DECODE_DQ_FAILURE"


def read_freeze_file_text(path: Path) -> str:
    """Read a captured `pip freeze --all` output file as text, standard
    library only. Decodes as UTF-8 and transparently strips a leading
    UTF-8 byte-order mark if present (`"utf-8-sig"` handles both a
    BOM-prefixed file, as PowerShell's default UTF-8 encoding on Windows
    commonly writes, and a plain BOM-less UTF-8 file identically -- there
    is no separate "Windows path" and "normal path": both decode through
    the exact same call). Raises `UnicodeDecodeError` on genuinely
    non-UTF-8 bytes; callers (`validate_predecessor_baseline_file`) treat
    that as a hard failure, never a silent skip.
    """
    return path.read_bytes().decode("utf-8-sig")


def validate_predecessor_baseline_file(path: Path) -> BaselineValidation:
    """Read and validate a captured BEFORE `pip freeze --all` snapshot
    file against the exact 7 frozen predecessor pins, reusing the
    existing `parse_pip_freeze_all` / `validate_predecessor_baseline`
    semantic authority verbatim -- no second, independently invented
    package-comparison methodology. This is the exact function the future
    Stage E5 staging runner invokes, via this module's CLI (see `main`),
    after capturing `before_freeze.txt` and before attempting the single
    successor-resolution command.

    Performs no package/environment mutation, no network access, and
    never imports `pdfplumber`. Fails closed (a dedicated status, never
    `BASELINE_OK`) if the file is missing or cannot be decoded as text.
    """
    if not path.exists():
        return BaselineValidation(status=BASELINE_FILE_MISSING_FAILURE)
    try:
        text = read_freeze_file_text(path)
    except UnicodeDecodeError:
        return BaselineValidation(status=BASELINE_FILE_DECODE_FAILURE)
    freeze_result = parse_pip_freeze_all(text)
    return validate_predecessor_baseline(freeze_result)


@dataclass(frozen=True)
class SuccessorValidation:
    status: str
    missing_predecessor_pins: Sequence[str] = ()
    predecessor_version_drift: Mapping[str, str] = field(default_factory=dict)
    new_package_missing: bool = False
    new_package_version_wrong: Optional[str] = None


SUCCESSOR_OK = "SUCCESSOR_RESOLUTION_OK"
SUCCESSOR_FAILURE = "SUCCESSOR_RESOLUTION_DQ_FAILURE"


def validate_successor_after_resolution(freeze_result: PipFreezeParseResult) -> SuccessorValidation:
    """Stage E5/E6: after the one-shot successor resolution, ALL 7
    predecessor pins must remain present and byte-for-byte unchanged from
    `PREDECESSOR_PINS`, and `pdfplumber==0.11.10` must be present. Any
    other package (pdfplumber's own transitive closure) may additionally
    exist -- this function does not reject additions, only predecessor-pin
    absence/drift and the new package's absence/wrong version. Fails
    closed on any invalid/duplicate freeze line.
    """
    if not freeze_result.is_clean:
        return SuccessorValidation(status=SUCCESSOR_FAILURE)

    packages = freeze_result.packages
    missing_predecessor_pins = tuple(name for name in PREDECESSOR_PINS if name not in packages)
    predecessor_version_drift = {
        name: packages[name]
        for name in PREDECESSOR_PINS
        if name in packages and packages[name] != PREDECESSOR_PINS[name]
    }
    new_package_missing = NEW_DIRECT_PACKAGE_NAME not in packages
    new_package_version_wrong: Optional[str] = None
    if not new_package_missing and packages[NEW_DIRECT_PACKAGE_NAME] != NEW_DIRECT_PACKAGE_VERSION:
        new_package_version_wrong = packages[NEW_DIRECT_PACKAGE_NAME]

    if missing_predecessor_pins or predecessor_version_drift or new_package_missing or new_package_version_wrong:
        return SuccessorValidation(
            status=SUCCESSOR_FAILURE,
            missing_predecessor_pins=missing_predecessor_pins,
            predecessor_version_drift=predecessor_version_drift,
            new_package_missing=new_package_missing,
            new_package_version_wrong=new_package_version_wrong,
        )
    return SuccessorValidation(status=SUCCESSOR_OK)


# =============================================================================
# Before/after delta derivation (Section 3a: added packages may exist only
# in AFTER; predecessor package versions may never drift)
# =============================================================================


@dataclass(frozen=True)
class DeltaResult:
    status: str
    added: Mapping[str, str] = field(default_factory=dict)
    removed: Sequence[str] = ()
    changed: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    predecessor_pin_drift: Mapping[str, tuple[str, str]] = field(default_factory=dict)


DELTA_OK = "DELTA_OK"
DELTA_PREDECESSOR_DRIFT_FAILURE = "DELTA_PREDECESSOR_PIN_DRIFT_DQ_FAILURE"
DELTA_REMOVED_PACKAGE_FAILURE = "DELTA_PACKAGE_REMOVED_DQ_FAILURE"


def compute_before_after_delta(before: Mapping[str, str], after: Mapping[str, str]) -> DeltaResult:
    """Derive the exact set-delta between a BEFORE (predecessor-baseline)
    and AFTER (post-successor-resolution) freeze snapshot.

    `added`: present only in AFTER (the intended shape for
    `pdfplumber==0.11.10` and its transitive closure).
    `removed`: present in BEFORE but absent from AFTER -- never permitted;
    forces `DELTA_REMOVED_PACKAGE_FAILURE`.
    `changed`: present in both with a different version -- reported for
    every such package, but only a change touching one of the 7 frozen
    `PREDECESSOR_PINS` (`predecessor_pin_drift`) forces
    `DELTA_PREDECESSOR_DRIFT_FAILURE`; a version change in a package that
    is not one of the 7 frozen predecessor pins is reported but does not
    by itself fail this function (predecessor-baseline validation already
    guarantees BEFORE contains only the 7 pins, so in practice `changed`
    outside `predecessor_pin_drift` should never occur when BEFORE was
    itself validated by `validate_predecessor_baseline`).
    """
    added = {name: version for name, version in after.items() if name not in before}
    removed = tuple(name for name in before if name not in after)
    changed = {
        name: (before[name], after[name])
        for name in before
        if name in after and before[name] != after[name]
    }
    predecessor_pin_drift = {
        name: versions for name, versions in changed.items() if name in PREDECESSOR_PINS
    }

    if removed:
        return DeltaResult(
            status=DELTA_REMOVED_PACKAGE_FAILURE,
            added=added,
            removed=removed,
            changed=changed,
            predecessor_pin_drift=predecessor_pin_drift,
        )
    if predecessor_pin_drift:
        return DeltaResult(
            status=DELTA_PREDECESSOR_DRIFT_FAILURE,
            added=added,
            removed=removed,
            changed=changed,
            predecessor_pin_drift=predecessor_pin_drift,
        )
    return DeltaResult(status=DELTA_OK, added=added, removed=removed, changed=changed, predecessor_pin_drift=predecessor_pin_drift)


# =============================================================================
# Windows/platform evidence schema validation
# =============================================================================

_PLATFORM_EVIDENCE_REQUIRED_KEYS = frozenset(
    {"implementation", "os_name", "platform_machine", "platform_system", "sysconfig_platform", "version"}
)


@dataclass(frozen=True)
class PlatformValidation:
    status: str
    unexpected_extra_keys: Sequence[str] = ()
    missing_keys: Sequence[str] = ()


PLATFORM_OK = "PLATFORM_EVIDENCE_OK"
PLATFORM_SCHEMA_FAILURE = "PLATFORM_EVIDENCE_SCHEMA_DQ_FAILURE"
PLATFORM_VALUE_MISMATCH_FAILURE = "PLATFORM_EVIDENCE_VALUE_MISMATCH_DQ_FAILURE"


def validate_platform_evidence(payload: Mapping[str, object]) -> PlatformValidation:
    """Exact-schema validation for a Windows/platform evidence payload:
    the key set must equal `_PLATFORM_EVIDENCE_REQUIRED_KEYS` exactly (no
    missing, no extra), and every value must equal the frozen canonical
    binding exactly -- CPython 3.12.10, Windows/AMD64/win-amd64, `os_name
    == "nt"`. `version` must be the exact 3-tuple
    `CANONICAL_PYTHON_EXACT_VERSION`.
    """
    actual_keys = frozenset(payload.keys())
    missing_keys = tuple(sorted(_PLATFORM_EVIDENCE_REQUIRED_KEYS - actual_keys))
    unexpected_extra_keys = tuple(sorted(actual_keys - _PLATFORM_EVIDENCE_REQUIRED_KEYS))
    if missing_keys or unexpected_extra_keys:
        return PlatformValidation(status=PLATFORM_SCHEMA_FAILURE, unexpected_extra_keys=unexpected_extra_keys, missing_keys=missing_keys)

    version = payload.get("version")
    values_match = (
        payload.get("implementation") == CANONICAL_PYTHON_IMPLEMENTATION
        and payload.get("os_name") == CANONICAL_OS_NAME
        and payload.get("platform_machine") == CANONICAL_PLATFORM_MACHINE
        and payload.get("platform_system") == CANONICAL_PLATFORM_SYSTEM
        and payload.get("sysconfig_platform") == CANONICAL_SYSCONFIG_PLATFORM
        and isinstance(version, (tuple, list))
        and tuple(version) == CANONICAL_PYTHON_EXACT_VERSION
    )
    if not values_match:
        return PlatformValidation(status=PLATFORM_VALUE_MISMATCH_FAILURE)
    return PlatformValidation(status=PLATFORM_OK)


# =============================================================================
# Future Stage E7 lock-candidate / Windows-evidence construction and schema
# validation. Pure functions only -- E2 never calls the `build_*` helpers
# to write any file; that happens only at the later, separately reviewed
# Stage E7.
# =============================================================================

_LOCK_CANDIDATE_REQUIRED_KEYS = frozenset(
    {"schema_version", "direct_spec_sha256", "resolved_packages", "predecessor_pins_preserved", "new_direct_package"}
)
_WINDOWS_EVIDENCE_REQUIRED_KEYS = frozenset(
    {"schema_version", "platform", "before_freeze", "after_freeze", "delta_status"}
)


def build_lock_candidate_payload(
    *,
    direct_spec_sha256: str,
    resolved_packages: Mapping[str, str],
) -> dict[str, object]:
    """Deterministically construct the FUTURE Stage E7 successor lock
    candidate payload shape. Pure function: no file I/O, no environment
    access. Not written to disk by this module.
    """
    return {
        "schema_version": 1,
        "direct_spec_sha256": direct_spec_sha256,
        "resolved_packages": dict(sorted(resolved_packages.items())),
        "predecessor_pins_preserved": dict(sorted(PREDECESSOR_PINS.items())),
        "new_direct_package": f"{NEW_DIRECT_PACKAGE_NAME}=={NEW_DIRECT_PACKAGE_VERSION}",
    }


def build_windows_evidence_payload(
    *,
    platform_evidence: Mapping[str, object],
    before_freeze: Mapping[str, str],
    after_freeze: Mapping[str, str],
    delta_status: str,
) -> dict[str, object]:
    """Deterministically construct the FUTURE Stage E7 Windows validation
    evidence payload shape. Pure function: no file I/O, no environment
    access. Not written to disk by this module.
    """
    return {
        "schema_version": 1,
        "platform": dict(platform_evidence),
        "before_freeze": dict(sorted(before_freeze.items())),
        "after_freeze": dict(sorted(after_freeze.items())),
        "delta_status": delta_status,
    }


def validate_lock_candidate_schema(payload: Mapping[str, object]) -> bool:
    """Exact key-set validation for a future Stage E7 lock-candidate
    payload. Schema-only; does not verify semantic correctness of values.
    """
    return frozenset(payload.keys()) == _LOCK_CANDIDATE_REQUIRED_KEYS


def validate_windows_evidence_schema(payload: Mapping[str, object]) -> bool:
    """Exact key-set validation for a future Stage E7 Windows-evidence
    payload. Schema-only; does not verify semantic correctness of values.
    """
    return frozenset(payload.keys()) == _WINDOWS_EVIDENCE_REQUIRED_KEYS


# =============================================================================
# Staging-path validation (Section 2b)
# =============================================================================


@dataclass(frozen=True)
class StagingPathValidation:
    status: str


STAGING_PATH_OK = "STAGING_PATH_OK"
STAGING_PATH_NOT_ABSOLUTE_FAILURE = "STAGING_PATH_NOT_ABSOLUTE_DQ_FAILURE"
STAGING_PATH_INSIDE_REPO_FAILURE = "STAGING_PATH_INSIDE_REPO_DQ_FAILURE"
STAGING_PATH_MATCHES_CANONICAL_NAME_FAILURE = "STAGING_PATH_MATCHES_CANONICAL_ENVIRONMENT_NAME_DQ_FAILURE"
STAGING_PATH_MATCHES_GENERAL_NAME_FAILURE = "STAGING_PATH_MATCHES_GENERAL_ENVIRONMENT_NAME_DQ_FAILURE"
STAGING_PATH_IS_SYMLINK_FAILURE = "STAGING_PATH_IS_SYMLINK_DQ_FAILURE"
STAGING_PATH_ALREADY_EXISTS_FAILURE = "STAGING_PATH_ALREADY_EXISTS_DQ_FAILURE"


def validate_staging_path(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    require_not_exists: bool = False,
) -> StagingPathValidation:
    """Validate a caller-supplied staging-venv path against Section 2b's
    frozen constraints: absolute; outside the repository tree; not named
    `.venv-real-execution` or `.venv` (the canonical and general
    environments); no symlink anywhere along the path; and, when
    `require_not_exists=True` (the future Stage E5 creation-time check),
    the path must not already exist. Purely a path-string/filesystem-
    metadata check -- performs no environment creation or mutation.
    """
    if not path.is_absolute():
        return StagingPathValidation(status=STAGING_PATH_NOT_ABSOLUTE_FAILURE)

    resolved_repo_root = repo_root.resolve()
    try:
        path.resolve().relative_to(resolved_repo_root)
        inside_repo = True
    except ValueError:
        inside_repo = False
    if inside_repo:
        return StagingPathValidation(status=STAGING_PATH_INSIDE_REPO_FAILURE)

    if path.name == CANONICAL_ENVIRONMENT_DIRECTORY_NAME:
        return StagingPathValidation(status=STAGING_PATH_MATCHES_CANONICAL_NAME_FAILURE)
    if path.name == GENERAL_PROJECT_ENVIRONMENT_DIRECTORY_NAME:
        return StagingPathValidation(status=STAGING_PATH_MATCHES_GENERAL_NAME_FAILURE)

    node = path
    while True:
        if node.is_symlink():
            return StagingPathValidation(status=STAGING_PATH_IS_SYMLINK_FAILURE)
        if node.parent == node:
            break
        node = node.parent

    if require_not_exists and path.exists():
        return StagingPathValidation(status=STAGING_PATH_ALREADY_EXISTS_FAILURE)

    return StagingPathValidation(status=STAGING_PATH_OK)


# =============================================================================
# Output-root validation (Section 5, Stage E5's -OutputRoot). Distinct from
# staging-path validation: OutputRoot additionally may never alias or
# overlap StagingRoot in either direction, and -- unlike a staging path,
# which may legitimately be reused across attempts prior to a one-shot
# resolution -- an OutputRoot must NEVER already exist: a previous
# evidence directory is never reused, appended to, or overwritten.
# =============================================================================


@dataclass(frozen=True)
class OutputPathValidation:
    status: str


OUTPUT_PATH_OK = "OUTPUT_PATH_OK"
OUTPUT_PATH_NOT_ABSOLUTE_FAILURE = "OUTPUT_PATH_NOT_ABSOLUTE_DQ_FAILURE"
OUTPUT_PATH_INSIDE_REPO_FAILURE = "OUTPUT_PATH_INSIDE_REPO_DQ_FAILURE"
OUTPUT_PATH_MATCHES_CANONICAL_NAME_FAILURE = "OUTPUT_PATH_MATCHES_CANONICAL_ENVIRONMENT_NAME_DQ_FAILURE"
OUTPUT_PATH_MATCHES_GENERAL_NAME_FAILURE = "OUTPUT_PATH_MATCHES_GENERAL_ENVIRONMENT_NAME_DQ_FAILURE"
OUTPUT_PATH_INSIDE_RESERVED_ENVIRONMENT_FAILURE = "OUTPUT_PATH_INSIDE_RESERVED_ENVIRONMENT_DQ_FAILURE"
OUTPUT_PATH_EQUALS_STAGING_ROOT_FAILURE = "OUTPUT_PATH_EQUALS_STAGING_ROOT_DQ_FAILURE"
OUTPUT_PATH_ANCESTOR_OF_STAGING_ROOT_FAILURE = "OUTPUT_PATH_ANCESTOR_OF_STAGING_ROOT_DQ_FAILURE"
OUTPUT_PATH_DESCENDANT_OF_STAGING_ROOT_FAILURE = "OUTPUT_PATH_DESCENDANT_OF_STAGING_ROOT_DQ_FAILURE"
OUTPUT_PATH_ALREADY_EXISTS_FAILURE = "OUTPUT_PATH_ALREADY_EXISTS_DQ_FAILURE"
OUTPUT_PATH_IS_SYMLINK_FAILURE = "OUTPUT_PATH_IS_SYMLINK_DQ_FAILURE"
OUTPUT_PATH_SAFETY_UNDETERMINED_FAILURE = "OUTPUT_PATH_SAFETY_UNDETERMINED_CHATGPT_DECISION_REQUIRED"

_RESERVED_ENVIRONMENT_NAMES = (CANONICAL_ENVIRONMENT_DIRECTORY_NAME, GENERAL_PROJECT_ENVIRONMENT_DIRECTORY_NAME)


def _contains_reserved_environment_component(path: Path) -> bool:
    """True if ANY path component (not just the leaf) is literally named
    `.venv-real-execution` or `.venv` -- catches an OutputRoot nested
    underneath a reserved-named directory, not merely equal to one.
    """
    return any(part in _RESERVED_ENVIRONMENT_NAMES for part in path.parts)


def _is_path_inside(candidate: Path, other: Path) -> bool:
    try:
        candidate.relative_to(other)
        return True
    except ValueError:
        return False


def _resolve_existing_ancestor(path: Path) -> Path:
    """Resolve the nearest EXISTING ancestor of `path` (following any real
    symlinks/junctions the OS reports) and reattach the non-existing
    suffix, producing a best-effort fully resolved candidate for
    aliasing/overlap re-verification -- without ever creating anything.
    """
    node = path
    suffix: list[str] = []
    while not node.exists():
        suffix.append(node.name)
        parent = node.parent
        if parent == node:
            break
        node = parent
    resolved_existing = node.resolve()
    for name in reversed(suffix):
        resolved_existing = resolved_existing / name
    return resolved_existing


def validate_output_path(
    path: Path,
    *,
    staging_root: Path,
    repo_root: Path = REPO_ROOT,
) -> OutputPathValidation:
    """Validate a caller-supplied Stage E5 `-OutputRoot` path. Fails
    closed, in order, unless: absolute; outside the repository tree; not
    named (nor nested inside a directory named) `.venv-real-execution` or
    `.venv`; not equal to, an ancestor of, or a descendant of
    `staging_root`; does not already exist (a previous evidence directory
    is NEVER reused); and no existing ancestor is a symlink. As a final
    step, the nearest existing ancestor is resolved (following real
    symlinks/junctions) and every overlap check above is re-applied to
    that resolved candidate, so a parent-directory symlink cannot
    silently alias OutputRoot into the repo, a reserved environment, or
    StagingRoot. If resolution itself cannot be completed (an OS-level
    error), this fails closed with `OUTPUT_PATH_SAFETY_UNDETERMINED_FAILURE`
    rather than assuming safety.

    Purely a path-string/filesystem-metadata check -- performs no
    directory creation, deletion, or mutation of any kind.
    """
    if not path.is_absolute():
        return OutputPathValidation(status=OUTPUT_PATH_NOT_ABSOLUTE_FAILURE)

    resolved_repo_root = repo_root.resolve()
    if _is_path_inside(path, resolved_repo_root):
        return OutputPathValidation(status=OUTPUT_PATH_INSIDE_REPO_FAILURE)

    if path.name == CANONICAL_ENVIRONMENT_DIRECTORY_NAME:
        return OutputPathValidation(status=OUTPUT_PATH_MATCHES_CANONICAL_NAME_FAILURE)
    if path.name == GENERAL_PROJECT_ENVIRONMENT_DIRECTORY_NAME:
        return OutputPathValidation(status=OUTPUT_PATH_MATCHES_GENERAL_NAME_FAILURE)
    if _contains_reserved_environment_component(path):
        return OutputPathValidation(status=OUTPUT_PATH_INSIDE_RESERVED_ENVIRONMENT_FAILURE)

    if path == staging_root:
        return OutputPathValidation(status=OUTPUT_PATH_EQUALS_STAGING_ROOT_FAILURE)
    if _is_path_inside(staging_root, path):
        return OutputPathValidation(status=OUTPUT_PATH_ANCESTOR_OF_STAGING_ROOT_FAILURE)
    if _is_path_inside(path, staging_root):
        return OutputPathValidation(status=OUTPUT_PATH_DESCENDANT_OF_STAGING_ROOT_FAILURE)

    if path.exists():
        return OutputPathValidation(status=OUTPUT_PATH_ALREADY_EXISTS_FAILURE)

    node = path
    while True:
        if node.is_symlink():
            return OutputPathValidation(status=OUTPUT_PATH_IS_SYMLINK_FAILURE)
        if node.parent == node:
            break
        node = node.parent

    try:
        resolved_candidate = _resolve_existing_ancestor(path)
    except OSError:
        return OutputPathValidation(status=OUTPUT_PATH_SAFETY_UNDETERMINED_FAILURE)

    if resolved_candidate != path:
        if _is_path_inside(resolved_candidate, resolved_repo_root):
            return OutputPathValidation(status=OUTPUT_PATH_INSIDE_REPO_FAILURE)
        if _contains_reserved_environment_component(resolved_candidate):
            return OutputPathValidation(status=OUTPUT_PATH_INSIDE_RESERVED_ENVIRONMENT_FAILURE)
        resolved_staging_root: Optional[Path]
        try:
            resolved_staging_root = _resolve_existing_ancestor(staging_root)
        except OSError:
            return OutputPathValidation(status=OUTPUT_PATH_SAFETY_UNDETERMINED_FAILURE)
        if resolved_candidate == resolved_staging_root:
            return OutputPathValidation(status=OUTPUT_PATH_EQUALS_STAGING_ROOT_FAILURE)
        if _is_path_inside(resolved_staging_root, resolved_candidate):
            return OutputPathValidation(status=OUTPUT_PATH_ANCESTOR_OF_STAGING_ROOT_FAILURE)
        if _is_path_inside(resolved_candidate, resolved_staging_root):
            return OutputPathValidation(status=OUTPUT_PATH_DESCENDANT_OF_STAGING_ROOT_FAILURE)

    return OutputPathValidation(status=OUTPUT_PATH_OK)


# =============================================================================
# Section 5a: synthetic PDF operational-readiness probe
# =============================================================================

PROBE_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "v9_014_synthetic_pdf_env_probe.pdf"
PROBE_EXPECTED_FIXTURE_SHA256 = "5eecb758a50e829af16bd42833f89a8329bfaaaa561aee209fbd2249b507b413"
PROBE_REQUIRED_PDFPLUMBER_VERSION = "0.11.10"
PROBE_EXPECTED_PAGE_COUNT = 1
PROBE_EXPECTED_TEXT = "SYNTHETIC_KEY SYNTHETIC_VALUE\nALPHA 11\nBETA 22"
PROBE_EXPECTED_TABLE: list[list[str]] = [
    ["SYNTHETIC_KEY", "SYNTHETIC_VALUE"],
    ["ALPHA", "11"],
    ["BETA", "22"],
]
# Fixed synthetic-only extraction settings. These are NOT, and must never
# be described or reused as, real JPX PDF parser/table settings, and do
# not resolve V9_014 design LOW_1 -- they exist solely to prove this
# pinned pdfplumber build extracts a known, fully synthetic table
# correctly.
PROBE_TEXT_EXTRACTION_TOLERANCES: dict[str, int] = {"x_tolerance": 3, "y_tolerance": 3}
PROBE_TABLE_SETTINGS: dict[str, object] = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 3,
    "join_tolerance": 3,
    "intersection_tolerance": 3,
}

PROBE_PASS = "SYNTHETIC_PDF_PROBE_PASS"
PROBE_FIXTURE_MISSING_FAILURE = "SYNTHETIC_PDF_PROBE_FIXTURE_MISSING_FAILURE"
PROBE_FIXTURE_HASH_MISMATCH_FAILURE = "SYNTHETIC_PDF_PROBE_FIXTURE_HASH_MISMATCH_FAILURE"
PROBE_VERSION_MISMATCH_FAILURE = "SYNTHETIC_PDF_PROBE_PDFPLUMBER_VERSION_MISMATCH_FAILURE"
PROBE_PAGE_COUNT_MISMATCH_FAILURE = "SYNTHETIC_PDF_PROBE_PAGE_COUNT_MISMATCH_FAILURE"
PROBE_TEXT_MISMATCH_FAILURE = "SYNTHETIC_PDF_PROBE_TEXT_MISMATCH_FAILURE"
PROBE_TABLE_MISMATCH_FAILURE = "SYNTHETIC_PDF_PROBE_TABLE_MISMATCH_FAILURE"


@dataclass(frozen=True)
class SyntheticPdfProbeResult:
    status: str
    observed_fixture_sha256: Optional[str] = None
    observed_pdfplumber_version: Optional[str] = None
    observed_page_count: Optional[int] = None
    observed_text: Optional[str] = None
    observed_table: Optional[list] = None


def run_synthetic_pdf_operational_probe(
    fixture_path: Path = PROBE_FIXTURE_PATH,
    *,
    pdfplumber_module: object = None,
) -> SyntheticPdfProbeResult:
    """Environment-readiness probe only. Proves the pinned
    `pdfplumber==0.11.10` build opens the committed wholly synthetic
    fixture and extracts its exact predetermined text/table -- nothing
    more. Never observes any of the 8 real SOURCE_B calibration PDFs and
    never resolves or amends V9_014 design LOW_1.

    `pdfplumber` is imported lazily, INSIDE this function body, ONLY when
    this function is explicitly called -- never at module import time and
    never as a side effect of any other function in this module. Pass
    `pdfplumber_module` to inject a fake/mock module for offline testing
    (Stage E2's own test suite never imports the real installed
    `pdfplumber`); leaving it `None` triggers the real lazy `import
    pdfplumber` (exercised only at the later, separately reviewed Stage
    E6/E10/E14).

    The fixture's SHA-256 is verified FIRST, before any `pdfplumber`
    import is attempted, so a tampered or wrong fixture fails closed
    without ever touching the PDF-parsing boundary.
    """
    if not fixture_path.exists():
        return SyntheticPdfProbeResult(status=PROBE_FIXTURE_MISSING_FAILURE)

    fixture_bytes = fixture_path.read_bytes()
    observed_fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    if observed_fixture_sha256 != PROBE_EXPECTED_FIXTURE_SHA256:
        return SyntheticPdfProbeResult(status=PROBE_FIXTURE_HASH_MISMATCH_FAILURE, observed_fixture_sha256=observed_fixture_sha256)

    if pdfplumber_module is None:
        import pdfplumber as pdfplumber_module  # lazy, real import -- only reached here

    observed_pdfplumber_version = getattr(pdfplumber_module, "__version__", None)
    if observed_pdfplumber_version != PROBE_REQUIRED_PDFPLUMBER_VERSION:
        return SyntheticPdfProbeResult(
            status=PROBE_VERSION_MISMATCH_FAILURE,
            observed_fixture_sha256=observed_fixture_sha256,
            observed_pdfplumber_version=observed_pdfplumber_version,
        )

    with pdfplumber_module.open(str(fixture_path)) as pdf:
        observed_page_count = len(pdf.pages)
        if observed_page_count != PROBE_EXPECTED_PAGE_COUNT:
            return SyntheticPdfProbeResult(
                status=PROBE_PAGE_COUNT_MISMATCH_FAILURE,
                observed_fixture_sha256=observed_fixture_sha256,
                observed_pdfplumber_version=observed_pdfplumber_version,
                observed_page_count=observed_page_count,
            )

        page = pdf.pages[0]
        observed_text = page.extract_text(**PROBE_TEXT_EXTRACTION_TOLERANCES)
        if observed_text != PROBE_EXPECTED_TEXT:
            return SyntheticPdfProbeResult(
                status=PROBE_TEXT_MISMATCH_FAILURE,
                observed_fixture_sha256=observed_fixture_sha256,
                observed_pdfplumber_version=observed_pdfplumber_version,
                observed_page_count=observed_page_count,
                observed_text=observed_text,
            )

        observed_table = page.extract_table(PROBE_TABLE_SETTINGS)
        if observed_table != PROBE_EXPECTED_TABLE:
            return SyntheticPdfProbeResult(
                status=PROBE_TABLE_MISMATCH_FAILURE,
                observed_fixture_sha256=observed_fixture_sha256,
                observed_pdfplumber_version=observed_pdfplumber_version,
                observed_page_count=observed_page_count,
                observed_text=observed_text,
                observed_table=observed_table,
            )

    return SyntheticPdfProbeResult(
        status=PROBE_PASS,
        observed_fixture_sha256=observed_fixture_sha256,
        observed_pdfplumber_version=observed_pdfplumber_version,
        observed_page_count=observed_page_count,
        observed_text=observed_text,
        observed_table=observed_table,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    With no arguments: a no-op informational message, exit 0 -- unchanged
    from Stage E2's original scope.

    `--validate-predecessor-baseline-file PATH`: the ONLY real action this
    CLI performs. Reads the captured BEFORE `pip freeze --all` snapshot at
    PATH and validates it against the exact 7 frozen predecessor pins via
    `validate_predecessor_baseline_file` (which itself reuses
    `parse_pip_freeze_all` / `validate_predecessor_baseline` verbatim --
    no independently invented comparison). This is the exact invocation
    the future Stage E5 staging runner makes, with the staging
    interpreter, immediately after capturing `before_freeze.txt` and
    strictly before attempting the single successor-resolution command.
    Prints the resulting status (and, on failure, the specific missing/
    drifted/extra package detail) and exits 0 only on `BASELINE_OK` --
    nonzero on every other status. Performs no package/environment
    mutation, no network access, and never imports `pdfplumber`.
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-predecessor-baseline-file",
        metavar="PATH",
        help="Validate a captured BEFORE 'pip freeze --all' file against the exact 7 frozen predecessor pins.",
    )
    args = parser.parse_args(argv)

    if args.validate_predecessor_baseline_file is not None:
        result = validate_predecessor_baseline_file(Path(args.validate_predecessor_baseline_file))
        print(f"status={result.status}")
        if result.missing:
            print(f"missing={','.join(result.missing)}")
        if result.version_mismatches:
            print(f"version_mismatches={result.version_mismatches}")
        if result.unexpected_extra:
            print(f"unexpected_extra={','.join(result.unexpected_extra)}")
        print("real_pdfplumber_imported=false")
        print("network_requests=0")
        print("environment_mutations=0")
        return 0 if result.status == BASELINE_OK else 1

    print("V9_014 PDF real-execution environment successor -- Stage E2 offline tooling module.")
    print("This module performs no action when run directly. See module docstring for scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
