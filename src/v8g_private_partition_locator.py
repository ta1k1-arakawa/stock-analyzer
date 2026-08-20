"""V8G private partition locator support
(`V8G_PRIVATE_PARTITION_LOCATOR_SUCCESSOR_DESIGN_DRAFT.md` sections 2.1.1-2.1.4).

This module implements ONLY the V8G locator-support contract: metadata-only
candidate normalization/hashing, the frozen human-authorization grammar, the
frozen fixed one-shot receipt-key derivation, canonical machine-local
durable receipt support, exact locator authority/design/implementation
binding, canonical partition-manifest hash/provenance verification, and the
safe public locator artifact producer/verifier. It does not implement V8G
T1C preservation (design section 2.1.5) -- that is a separate, independently
reviewed fresh module per section 2.2, never merged with this one.

Bound to the exact independently reviewed design candidate:

    reviewed_v8g_design_candidate_commit = b9c7014ba72b72efadb1a4be6c5aa4aa71201518
    design_blob = fefbf898a1dda01d852d8d36b1ed8e086c748c7d

Section 2.1.1 assigns the metadata-only candidate enumeration itself to
"a future V8G locator support implementation" -- this module, not an
external PowerShell/runbook step (unlike V8F's locator, which explicitly
delegated enumeration externally; V8G's design contains no such
delegation). `enumerate_candidate_partition_manifest_paths` is the
authoritative implementation: every ready Fixed/Removable volume,
recursively, for files named exactly `partition_manifest.json`, excluding
this repository's own subtree, reading no content and inspecting no ticker
identity. The production entry point always derives its candidate list from
this enumerator; a normal caller cannot supply or override it. A
dependency-injected volume/walk provider exists solely for tests and
internal DI -- this implementation session and its test suite never invoke
the real, Windows-only default provider against a real filesystem.
Importing this module performs no I/O, no network access, and no gate
consumption.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import string
import subprocess
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.v8_partition import (
    MANIFEST_FIELDS,
    SCHEMA_VERSION as V8_PARTITION_SCHEMA_VERSION,
    SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
    SOURCE_SNAPSHOT_SEMANTICS,
    V8PartitionBlocked,
    canonical_sha256 as v8_canonical_sha256,
    require_git_commit as require_v8_git_commit,
)
from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    resolve_git_blob,
)
from src.v8c_human_gate_consumption import CANONICAL_CONSUMPTION_STATE_ROOT


V8G_STUDY_NAME = "V8G_HISTORICAL_RESEARCH"
V8G_REPOSITORY_IDENTITY = "ta1k1-arakawa/stock-analyzer"
V8G_PRODUCTION_BRANCH = "v8g-private-partition-locator-successor-design"

# The exact independently reviewed V8G design candidate this implementation
# is bound to (design sections 2.3, 2.1.3). Never a "frozen design commit":
# none exists until HUMAN_V8G_DESIGN_FREEZE.
REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT = "b9c7014ba72b72efadb1a4be6c5aa4aa71201518"
V8G_DESIGN_CANDIDATE_BLOB_SHA = "fefbf898a1dda01d852d8d36b1ed8e086c748c7d"
V8G_DESIGN_DRAFT_GIT_PATH = "V8G_PRIVATE_PARTITION_LOCATOR_SUCCESSOR_DESIGN_DRAFT.md"

# Historical V8F predecessor terminal evidence (design section 1). Historical
# evidence only; never renamed to V8G and never V8G authority.
V8F_PREDECESSOR_TERMINAL_COMMIT = "d1447df86b0caa7a5240d45cba8f01f8829a940c"
V8F_PREDECESSOR_TERMINAL_ARTIFACT_BLOB = "91572d706c7ccb6f6f2e3c840791cb14c7eb8bca"

# The frozen historical authorized partition identity (design section 2.1).
# The real production manifest path was never publicly recorded; this module
# never invents, guesses, or assumes it -- it only resolves it, once, by
# content address, from an externally supplied candidate list.
EXPECTED_V8_PARTITION_MANIFEST_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT = "36cbed941050e728f7f96ce2af505e81175cc02c"

PARTITION_MANIFEST_BASENAME = "partition_manifest.json"

V8G_LOCATOR_GATE = "HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE"
V8G_LOCATOR_GATE_CONSUMPTION_BOUNDARY = "IMMEDIATELY_BEFORE_FIRST_CANDIDATE_PARTITION_BYTE_READ"

V8G_AUTHORIZATION_PREFIX = "V8G_HUMAN_AUTHORIZE_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_AT_"
V8G_AUTHORIZATION_WITH = "_WITH_"
V8G_AUTHORIZATION_FOR_MANIFEST = "_FOR_MANIFEST_"
V8G_AUTHORIZATION_IMPL = "_IMPL_"

V8G_LOCATOR_RECEIPT_SCHEMA_VERSION = "V8G_PRIVATE_PARTITION_LOCATOR_GATE_RECEIPT_V1"
V8G_LOCATOR_RECEIPT_ARTIFACT_ROLE = "PRIVATE_PARTITION_LOCATOR_GATE_RECEIPT"
V8G_LOCATOR_RECEIPT_FIELDS = (
    "schema_version",
    "artifact_role",
    "study",
    "gate",
    "reviewed_v8g_design_candidate_commit",
    "reviewed_locator_support_implementation_sha",
    "expected_partition_manifest_sha256",
    "expected_partition_implementation_commit",
    "authorization_identity_sha256",
    "consumed",
    "consumption_count",
    "consumption_boundary",
    "consumption_timestamp_utc",
)

V8G_LOCATOR_ARTIFACT_SCHEMA_VERSION = "V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_V1"
V8G_LOCATOR_ARTIFACT_ROLE = "PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT"
V8G_LOCATOR_CONTRACT = "PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_V1"
V8G_LOCATOR_ARTIFACT_FIELDS = (
    "schema_version",
    "artifact_role",
    "study",
    "reviewed_v8g_design_candidate_commit",
    "reviewed_locator_support_implementation_sha",
    "predecessor_terminal_commit",
    "predecessor_terminal_artifact_blob",
    "locator_contract",
    "candidate_count",
    "candidate_set_sha256",
    "selected_locator_path_sha256",
    "expected_partition_manifest_sha256",
    "expected_partition_implementation_commit",
    "locator_gate_receipt_key_sha256",
    "locator_gate_receipt_bytes_sha256",
    "ticker_identities_exposed",
    "block_assignments_exposed",
    "raw_or_private_payload_persisted_publicly",
    "research_opened",
    "raw_acquisition_performed",
    "locator_result",
)

CANONICAL_V8G_LOCATOR_GATE_STATE_ROOT = CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8g-locator-gate-state"

_PATH_HASH_DOMAIN = "V8G_PRIVATE_PARTITION_LOCATOR_PATH_V1\0"
_CANDIDATE_SET_DOMAIN = "V8G_PRIVATE_PARTITION_LOCATOR_CANDIDATE_SET_V1\n"

_HEX = re.compile(r"^[0-9a-f]+$")
_TIMESTAMP_SECONDS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_TIMESTAMP_MICROS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
_REQUIRED_BLOCK_KEYS = ("T0", "T1", "T2", "T3", "T_spare")


class V8GPrivatePartitionLocatorBlocked(RuntimeError):
    """Fail-closed V8G private partition locator error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _require_hex(value: object, length: int, reason: str) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise V8GPrivatePartitionLocatorBlocked(reason)
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_NONFINITE_OR_UNSERIALIZABLE") from error


def _strict_json_object(raw: bytes, invalid_reason: str, duplicate_reason: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise V8GPrivatePartitionLocatorBlocked(duplicate_reason)
            result[key] = value
        return result

    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V8GPrivatePartitionLocatorBlocked(invalid_reason) from error
    if not isinstance(parsed, dict):
        raise V8GPrivatePartitionLocatorBlocked(invalid_reason)
    return parsed


def _timestamp_text(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_CLOCK_INVALID")
    utc = value.astimezone(timezone.utc)
    if utc.microsecond:
        return utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str) or not (_TIMESTAMP_SECONDS.fullmatch(value) or _TIMESTAMP_MICROS.fullmatch(value)):
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_TIMESTAMP_INVALID")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ" if "." not in value else "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_TIMESTAMP_INVALID") from error
    return value


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


# ---------------------------------------------------------------------------
# 2.1.2 -- Safe path-hash contract
# ---------------------------------------------------------------------------


def _canonicalize_resolved_path_text(resolved_path_text: str) -> str:
    """Pure string transform: backslash-to-forward-slash + casefold + NFC.

    Split out from `canonical_path_text` so the mixed-case/separator
    equivalence the design requires can be tested without depending on a
    case-insensitive filesystem (this repository's test/CI hosts are
    case-sensitive POSIX, while the production target is Windows).
    """
    folded = resolved_path_text.replace("\\", "/").casefold()
    return unicodedata.normalize("NFC", folded)


def canonical_path_text(path: str | os.PathLike[str]) -> str:
    """`NFC(str(Path(path).resolve(strict=True)).replace("\\\\","/").casefold())`."""
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_CANDIDATE_PATH_UNAVAILABLE") from error
    return _canonicalize_resolved_path_text(str(resolved))


def _path_hash_from_canonical_text(canonical_text: str) -> str:
    return hashlib.sha256((_PATH_HASH_DOMAIN + canonical_text).encode("utf-8")).hexdigest()


def locator_path_sha256(path: str | os.PathLike[str]) -> str:
    text = canonical_path_text(path)
    return _path_hash_from_canonical_text(text)


def candidate_set_serialization_v1(hash_list: Sequence[str]) -> bytes:
    if isinstance(hash_list, (str, bytes)):
        raise V8GPrivatePartitionLocatorBlocked("V8G_CANDIDATE_SET_INVALID")
    try:
        ordered = list(hash_list)
    except TypeError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_CANDIDATE_SET_INVALID") from error
    for value in ordered:
        _require_hex(value, 64, "V8G_CANDIDATE_SET_HASH_INVALID")
    if ordered != sorted(ordered):
        raise V8GPrivatePartitionLocatorBlocked("V8G_CANDIDATE_SET_NOT_SORTED")
    if len(set(ordered)) != len(ordered):
        raise V8GPrivatePartitionLocatorBlocked("V8G_CANDIDATE_SET_DUPLICATE_HASH")
    text = _CANDIDATE_SET_DOMAIN + str(len(ordered)) + "\n" + "\n".join(ordered) + "\n"
    return text.encode("utf-8")


def candidate_set_sha256(hash_list: Sequence[str]) -> str:
    return hashlib.sha256(candidate_set_serialization_v1(hash_list)).hexdigest()


# ---------------------------------------------------------------------------
# 2.1.1 -- Metadata-only candidate snapshot (pre-gate, no content read)
# ---------------------------------------------------------------------------


def _require_outside_repository(resolved: Path, repository_root: Path, reason: str) -> Path:
    try:
        resolved.relative_to(Path(repository_root).resolve(strict=False))
    except ValueError:
        return resolved
    raise V8GPrivatePartitionLocatorBlocked(reason)


def validate_candidate_partition_manifest_paths(
    candidate_paths: Sequence[str | os.PathLike[str]],
    repository_root: Path = CANONICAL_REPOSITORY_ROOT,
) -> tuple[Path, ...]:
    """Pre-gate, metadata-only candidate validation. Reads no candidate bytes.

    Every candidate must exist (its metadata is stat'd to canonicalize and
    hash it -- never its content), have exact basename
    `partition_manifest.json`, and lie outside the repository. The
    normalized (canonical-path-hash) candidate set must be non-empty.
    Per §2.1.1, a candidate whose canonical path text exactly repeats an
    already-observed candidate (e.g. the same path supplied twice, or a
    dot/dot-dot alias resolving to the same location) is a benign duplicate
    and is silently merged into a single frozen candidate -- never
    double-counted, never scanned twice. A candidate whose
    `locator_path_sha256` collides with an already-observed candidate's hash
    while its canonical path text differs is, per §2.1.2, a fail-closed
    schema violation rather than a duplicate: this is rejected with a
    generic collision reason. No candidate path or canonical path text is
    ever printed, logged, persisted, or returned in an exception message.
    """
    if isinstance(candidate_paths, (str, bytes)):
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_CANDIDATE_LIST_INVALID")
    try:
        candidate_list = list(candidate_paths)
    except TypeError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_CANDIDATE_LIST_INVALID") from error
    if len(candidate_list) == 0:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_CANDIDATE_LIST_EMPTY")
    normalized: list[Path] = []
    # In-memory only, never persisted/printed/logged: path_hash -> the
    # canonical_path_text that produced it, so a genuine repeated path can be
    # distinguished from a same-hash/different-text collision.
    seen_canonical_text_by_hash: dict[str, str] = {}
    for value in candidate_list:
        try:
            resolved = Path(value).resolve(strict=True)
        except OSError as error:
            raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_CANDIDATE_PATH_UNAVAILABLE") from error
        if resolved.name != PARTITION_MANIFEST_BASENAME:
            raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_CANDIDATE_BASENAME_INVALID")
        _require_outside_repository(resolved, repository_root, "V8G_LOCATOR_CANDIDATE_PATH_INVALID")
        canonical_text = _canonicalize_resolved_path_text(str(resolved))
        path_hash = _path_hash_from_canonical_text(canonical_text)
        existing_text = seen_canonical_text_by_hash.get(path_hash)
        if existing_text is not None:
            if existing_text == canonical_text:
                continue  # benign duplicate: same normalized path, merged silently
            raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_CANDIDATE_HASH_COLLISION")
        seen_canonical_text_by_hash[path_hash] = canonical_text
        normalized.append(resolved)
    return tuple(normalized)


# ---------------------------------------------------------------------------
# 2.1.1 -- Authoritative metadata-only candidate enumeration
# ---------------------------------------------------------------------------

# Win32 GetDriveTypeW return values (metadata only; no content is read to
# determine either drive type or readiness).
_WINDOWS_DRIVE_TYPE_LABELS = {2: "Removable", 3: "Fixed", 4: "Network", 5: "CDROM"}
_ALLOWED_ENUMERATION_DRIVE_TYPES = frozenset({"Fixed", "Removable"})


def _default_raw_volume_provider() -> tuple[tuple[Path, str], ...]:
    """Windows-only production volume provider.

    Enumerates every currently assigned drive letter and its Win32
    `DriveType` via the metadata-only `GetDriveTypeW` API -- no file content
    of any kind is read. Every drive type is returned unfiltered, including
    Network and CD/optical; `_filter_allowed_volumes` applies the
    Fixed/Removable-only inclusion policy separately, so that policy stays
    independently unit-testable without depending on this Windows-only call.
    Never invoked by this implementation's own test suite.
    """
    if os.name != "nt":
        raise V8GPrivatePartitionLocatorBlocked("V8G_ENUMERATION_WINDOWS_ONLY")
    get_drive_type = ctypes.windll.kernel32.GetDriveTypeW  # type: ignore[attr-defined]
    volumes: list[tuple[Path, str]] = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        drive_type = get_drive_type(ctypes.c_wchar_p(root))
        label = _WINDOWS_DRIVE_TYPE_LABELS.get(drive_type)
        if label is not None:
            volumes.append((Path(root), label))
    return tuple(volumes)


def _filter_allowed_volumes(volumes: Sequence[tuple[Path, str]]) -> tuple[Path, ...]:
    """Pure, metadata-only `DriveType` filter (design section 2.1.1):
    `Fixed` and `Removable` are kept; `Network` and CD/optical are excluded
    by construction. Independent of the Windows-only raw provider so the
    inclusion/exclusion policy itself is fully unit-testable.
    """
    return tuple(root for root, drive_type in volumes if drive_type in _ALLOWED_ENUMERATION_DRIVE_TYPES)


def _default_file_walker(volume_root: Path) -> Iterable[Path]:
    """Production file walker: recursively lists every accessible file
    under `volume_root`. A volume that is not currently ready, or any
    subdirectory that is not accessible, is silently skipped -- never
    logged, never included in any exception -- via `os.walk`'s permissive
    `onerror` handling; no file content is ever read here. Never invoked by
    this implementation's own test suite.
    """
    for dirpath, _dirnames, filenames in os.walk(volume_root, onerror=lambda _error: None):
        for filename in filenames:
            yield Path(dirpath) / filename


def enumerate_candidate_partition_manifest_paths(
    *,
    repository_root: Path = CANONICAL_REPOSITORY_ROOT,
    raw_volume_provider: Callable[[], Sequence[tuple[Path, str]]] | None = None,
    file_walker: Callable[[Path], Iterable[Path]] | None = None,
) -> tuple[Path, ...]:
    """The authoritative design section 2.1.1 metadata-only candidate
    enumerator.

    Scans every ready `Fixed`/`Removable` volume, recursively, for a file
    named exactly `partition_manifest.json`, excluding this repository's
    own working-tree subtree. Reads no candidate content, inspects no
    ticker identity, and never prints/logs/persists a candidate path. The
    resulting candidate list is normalized, deduplicated, and validated
    through the same `validate_candidate_partition_manifest_paths` pre-gate
    check the rest of this module already uses -- a genuine duplicate or an
    empty result is fail-closed exactly as it already is there. The
    returned tuple is frozen for the caller's one execution; nothing may be
    added to or removed from it afterward.

    `raw_volume_provider`/`file_walker` exist solely for tests and internal
    DI -- production callers must never override them (only the
    `resolve_and_locate_authorized_partition_manifest` internals may, and
    they always resolve to the real, Windows-only defaults).
    """
    provider = raw_volume_provider or _default_raw_volume_provider
    walker = file_walker or _default_file_walker
    resolved_repository_root = Path(repository_root).resolve(strict=False)
    discovered: list[Path] = []
    for volume_root in _filter_allowed_volumes(provider()):
        try:
            for candidate in walker(volume_root):
                if candidate.name != PARTITION_MANIFEST_BASENAME:
                    continue
                try:
                    resolved_candidate = candidate.resolve(strict=False)
                except OSError:
                    continue
                try:
                    resolved_candidate.relative_to(resolved_repository_root)
                except ValueError:
                    discovered.append(candidate)
                # else: inside the repository working-tree subtree -- excluded
                # from scope entirely, never appended.
        except OSError:
            continue
    return validate_candidate_partition_manifest_paths(discovered, repository_root)


# ---------------------------------------------------------------------------
# 2.1.3 -- Human authorization grammar
# ---------------------------------------------------------------------------


def build_authorization_identity(
    *,
    reviewed_v8g_design_candidate_commit: str,
    reviewed_locator_support_implementation_sha: str,
    expected_partition_manifest_sha256: str,
    expected_partition_implementation_commit: str,
) -> str:
    candidate = _require_hex(reviewed_v8g_design_candidate_commit, 40, "V8G_DESIGN_CANDIDATE_INVALID")
    implementation = _require_hex(
        reviewed_locator_support_implementation_sha, 40, "V8G_LOCATOR_IMPLEMENTATION_SHA_INVALID"
    )
    manifest = _require_hex(expected_partition_manifest_sha256, 64, "V8G_MANIFEST_SHA_INVALID")
    impl_commit = _require_hex(
        expected_partition_implementation_commit, 40, "V8G_PARTITION_IMPLEMENTATION_COMMIT_INVALID"
    )
    return (
        V8G_AUTHORIZATION_PREFIX
        + candidate
        + V8G_AUTHORIZATION_WITH
        + implementation
        + V8G_AUTHORIZATION_FOR_MANIFEST
        + manifest
        + V8G_AUTHORIZATION_IMPL
        + impl_commit
    )


def authorization_identity_sha256(authorization_identity: str) -> str:
    """Return only the SHA-256; never persist or expose the raw identity."""
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8GPrivatePartitionLocatorBlocked("V8G_AUTHORIZATION_IDENTITY_REQUIRED")
    return hashlib.sha256(authorization_identity.encode("utf-8")).hexdigest()


def validate_authorization_identity(
    authorization_identity: str,
    *,
    reviewed_locator_support_implementation_sha: str,
    reviewed_v8g_design_candidate_commit: str = REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
    expected_partition_manifest_sha256: str = EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    expected_partition_implementation_commit: str = EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
) -> None:
    """Require the exact V8G locator authorization grammar and binding."""
    candidate = _require_hex(reviewed_v8g_design_candidate_commit, 40, "V8G_DESIGN_CANDIDATE_INVALID")
    implementation = _require_hex(
        reviewed_locator_support_implementation_sha, 40, "V8G_LOCATOR_IMPLEMENTATION_SHA_INVALID"
    )
    manifest = _require_hex(expected_partition_manifest_sha256, 64, "V8G_MANIFEST_SHA_INVALID")
    impl_commit = _require_hex(
        expected_partition_implementation_commit, 40, "V8G_PARTITION_IMPLEMENTATION_COMMIT_INVALID"
    )
    if candidate != REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT:
        raise V8GPrivatePartitionLocatorBlocked("V8G_DESIGN_CANDIDATE_MISMATCH")
    if manifest != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8GPrivatePartitionLocatorBlocked("V8G_MANIFEST_IDENTITY_MISMATCH")
    if impl_commit != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PARTITION_IMPLEMENTATION_IDENTITY_MISMATCH")
    if not isinstance(authorization_identity, str) or not authorization_identity:
        raise V8GPrivatePartitionLocatorBlocked("V8G_AUTHORIZATION_GRAMMAR_MISMATCH")
    expected = build_authorization_identity(
        reviewed_v8g_design_candidate_commit=candidate,
        reviewed_locator_support_implementation_sha=implementation,
        expected_partition_manifest_sha256=manifest,
        expected_partition_implementation_commit=impl_commit,
    )
    if authorization_identity != expected:
        raise V8GPrivatePartitionLocatorBlocked("V8G_AUTHORIZATION_GRAMMAR_MISMATCH")


# ---------------------------------------------------------------------------
# 2.1.3 -- Deterministic one-shot receipt key (independent of authorization,
# design candidate, and implementation SHA -- see the design's rationale)
# ---------------------------------------------------------------------------

_RECEIPT_KEY_MATERIAL = (
    "V8G_PRIVATE_PARTITION_LOCATOR_GATE_RECEIPT_KEY_V1\0"
    + V8G_REPOSITORY_IDENTITY
    + "\0"
    + V8G_STUDY_NAME
    + "\0"
    + V8G_LOCATOR_GATE
).encode("utf-8")


def compute_locator_gate_receipt_key() -> str:
    """Fixed, deterministic receipt key: repository + study + gate only.

    Deliberately takes no arguments -- the authorization identity, the
    reviewed design candidate commit, and the reviewed locator-support
    implementation SHA must never affect this key, so the gate can be
    durably consumed at most once for the entire life of the V8G study.
    """
    return hashlib.sha256(_RECEIPT_KEY_MATERIAL).hexdigest()


def _receipt_path(state_root: str | os.PathLike[str]) -> Path:
    return Path(state_root) / (compute_locator_gate_receipt_key() + ".json")


def _validate_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if set(receipt) != set(V8G_LOCATOR_RECEIPT_FIELDS):
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_SCHEMA_INVALID")
    if receipt["schema_version"] != V8G_LOCATOR_RECEIPT_SCHEMA_VERSION:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_SCHEMA_VERSION_MISMATCH")
    if receipt["artifact_role"] != V8G_LOCATOR_RECEIPT_ARTIFACT_ROLE:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_IDENTITY_INVALID")
    if receipt["study"] != V8G_STUDY_NAME or receipt["gate"] != V8G_LOCATOR_GATE:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_IDENTITY_INVALID")
    _require_hex(receipt["reviewed_v8g_design_candidate_commit"], 40, "V8G_RECEIPT_DESIGN_CANDIDATE_INVALID")
    _require_hex(
        receipt["reviewed_locator_support_implementation_sha"], 40, "V8G_RECEIPT_IMPLEMENTATION_SHA_INVALID"
    )
    _require_hex(receipt["expected_partition_manifest_sha256"], 64, "V8G_RECEIPT_MANIFEST_SHA_INVALID")
    _require_hex(
        receipt["expected_partition_implementation_commit"], 40, "V8G_RECEIPT_IMPLEMENTATION_COMMIT_INVALID"
    )
    _require_hex(receipt["authorization_identity_sha256"], 64, "V8G_RECEIPT_AUTHORIZATION_HASH_INVALID")
    if (
        receipt["consumed"] is not True
        or type(receipt["consumption_count"]) is not int
        or receipt["consumption_count"] != 1
    ):
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_CONSUMPTION_INVALID")
    if receipt["consumption_boundary"] != V8G_LOCATOR_GATE_CONSUMPTION_BOUNDARY:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_CONSUMPTION_BOUNDARY_INVALID")
    _validate_timestamp(receipt["consumption_timestamp_utc"])
    return dict(receipt)


def read_gate_receipt(state_root: str | os.PathLike[str]) -> dict[str, Any]:
    path = _receipt_path(state_root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_MISSING") from error
    return _validate_receipt(_strict_json_object(raw, "V8G_RECEIPT_INVALID_JSON", "V8G_RECEIPT_DUPLICATE_KEY"))


def gate_receipt_bytes_sha256(state_root: str | os.PathLike[str]) -> str:
    """Validate and hash the exact durable receipt bytes externally."""
    path = _receipt_path(state_root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_MISSING") from error
    _validate_receipt(_strict_json_object(raw, "V8G_RECEIPT_INVALID_JSON", "V8G_RECEIPT_DUPLICATE_KEY"))
    return hashlib.sha256(raw).hexdigest()


def _read_and_bind_gate_receipt(
    state_root: str | os.PathLike[str],
    *,
    reviewed_v8g_design_candidate_commit: str,
    reviewed_locator_support_implementation_sha: str,
    expected_partition_manifest_sha256: str,
    expected_partition_implementation_commit: str,
    authorization_identity: str,
) -> tuple[dict[str, Any], str]:
    """Post-gate, pre-artifact-publication receipt semantic binding.

    Reads the exact durable receipt once, validates its structural schema
    (``_validate_receipt``), then mechanically requires exact equality
    between every one of its bound fields and this execution's own
    authorized values -- a structurally well-formed receipt that is bound
    to a different design candidate, locator-support implementation,
    expected manifest/implementation identity, or authorization identity is
    a fail-closed ``POST_GATE`` ``BLOCK``, never silently accepted. The raw
    authorization identity is hashed locally for comparison and never
    persisted or logged; the receipt itself is only ever read here, never
    replaced, reset, or deleted. Returns the validated receipt together
    with the SHA-256 of the exact validated durable bytes.
    """
    path = _receipt_path(state_root)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_MISSING") from error
    receipt = _validate_receipt(_strict_json_object(raw, "V8G_RECEIPT_INVALID_JSON", "V8G_RECEIPT_DUPLICATE_KEY"))
    if receipt["reviewed_v8g_design_candidate_commit"] != reviewed_v8g_design_candidate_commit:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_DESIGN_CANDIDATE_MISMATCH")
    if receipt["reviewed_locator_support_implementation_sha"] != reviewed_locator_support_implementation_sha:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_IMPLEMENTATION_SHA_MISMATCH")
    if receipt["expected_partition_manifest_sha256"] != expected_partition_manifest_sha256:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_MANIFEST_SHA_MISMATCH")
    if receipt["expected_partition_implementation_commit"] != expected_partition_implementation_commit:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_IMPLEMENTATION_COMMIT_MISMATCH")
    if receipt["authorization_identity_sha256"] != authorization_identity_sha256(authorization_identity):
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_AUTHORIZATION_HASH_MISMATCH")
    return receipt, hashlib.sha256(raw).hexdigest()


def consume_gate_once(
    state_root: str | os.PathLike[str],
    authorization_identity: str,
    *,
    clock: Callable[[], datetime],
    reviewed_v8g_design_candidate_commit: str,
    reviewed_locator_support_implementation_sha: str,
    expected_partition_manifest_sha256: str,
    expected_partition_implementation_commit: str,
) -> dict[str, Any]:
    """Durably publish exactly one V8G locator receipt; no reset/replay API."""
    validate_authorization_identity(
        authorization_identity,
        reviewed_locator_support_implementation_sha=reviewed_locator_support_implementation_sha,
        reviewed_v8g_design_candidate_commit=reviewed_v8g_design_candidate_commit,
        expected_partition_manifest_sha256=expected_partition_manifest_sha256,
        expected_partition_implementation_commit=expected_partition_implementation_commit,
    )
    root = Path(state_root)
    path = _receipt_path(root)
    if path.exists():
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_GATE_ALREADY_CONSUMED")
    receipt = {
        "schema_version": V8G_LOCATOR_RECEIPT_SCHEMA_VERSION,
        "artifact_role": V8G_LOCATOR_RECEIPT_ARTIFACT_ROLE,
        "study": V8G_STUDY_NAME,
        "gate": V8G_LOCATOR_GATE,
        "reviewed_v8g_design_candidate_commit": reviewed_v8g_design_candidate_commit,
        "reviewed_locator_support_implementation_sha": reviewed_locator_support_implementation_sha,
        "expected_partition_manifest_sha256": expected_partition_manifest_sha256,
        "expected_partition_implementation_commit": expected_partition_implementation_commit,
        "authorization_identity_sha256": authorization_identity_sha256(authorization_identity),
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": V8G_LOCATOR_GATE_CONSUMPTION_BOUNDARY,
        "consumption_timestamp_utc": _timestamp_text(clock()),
    }
    payload = _canonical_json_bytes(receipt)
    try:
        root.mkdir(parents=True, exist_ok=True)
        staging = root / (path.name + ".staging-" + os.urandom(8).hex())
        with open(staging, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, path)
        except FileExistsError as error:
            raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_GATE_ALREADY_CONSUMED") from error
        except OSError as error:
            raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_STORAGE_WRITE_FAILED") from error
        _fsync_directory(root)
    except V8GPrivatePartitionLocatorBlocked:
        raise
    except OSError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_RECEIPT_STORAGE_WRITE_FAILED") from error
    finally:
        staging_path = locals().get("staging")
        if isinstance(staging_path, Path):
            try:
                if staging_path.exists():
                    staging_path.unlink()
            except OSError:
                pass
    return dict(receipt)


# ---------------------------------------------------------------------------
# Canonical partition-manifest hash/provenance verification (post-gate only)
# ---------------------------------------------------------------------------


def _read_partition_manifest_bytes(raw: bytes) -> dict[str, Any]:
    manifest = _strict_json_object(raw, "V8G_PARTITION_MANIFEST_INVALID_JSON", "V8G_PARTITION_MANIFEST_DUPLICATE_KEY")
    if set(manifest) != set(MANIFEST_FIELDS):
        raise V8GPrivatePartitionLocatorBlocked("V8G_PARTITION_MANIFEST_SCHEMA_INVALID")
    if manifest["manifest_sha256"] != v8_canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    ):
        raise V8GPrivatePartitionLocatorBlocked("V8G_PARTITION_MANIFEST_SHA_MISMATCH")
    try:
        require_v8_git_commit(manifest["partition_implementation_git_commit"])
    except V8PartitionBlocked as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PARTITION_IMPLEMENTATION_COMMIT_INVALID") from error
    if (
        manifest["schema_version"] != V8_PARTITION_SCHEMA_VERSION
        or manifest["source_snapshot_semantics"] != SOURCE_SNAPSHOT_SEMANTICS
        or manifest["source_snapshot_clarification_commit"] != SOURCE_SNAPSHOT_CLARIFICATION_COMMIT
        or manifest["v4_raw_sha_equality_required"] is not False
        or manifest["source_reproduction_status"] != "PASS"
        or manifest["t0_reproduction_status"] != "PASS"
    ):
        raise V8GPrivatePartitionLocatorBlocked("V8G_PARTITION_MANIFEST_FROZEN_BINDING_INVALID")
    assignments = manifest["block_assignments"]
    if not isinstance(assignments, Mapping) or set(_REQUIRED_BLOCK_KEYS) - set(assignments):
        raise V8GPrivatePartitionLocatorBlocked("V8G_PARTITION_BLOCK_ASSIGNMENT_INVALID")
    for key in _REQUIRED_BLOCK_KEYS:
        if not isinstance(assignments[key], list):
            raise V8GPrivatePartitionLocatorBlocked("V8G_PARTITION_BLOCK_ASSIGNMENT_INVALID")
    return manifest


def _locate_authorized_partition_manifest(
    private_reader: Callable[[Path], bytes],
    candidate_paths: Sequence[Path],
    *,
    expected_partition_manifest_sha256: str,
    expected_partition_implementation_commit: str,
) -> tuple[Path, bytes, dict[str, int]]:
    """Single post-gate content-addressed scan over pre-validated candidates.

    A candidate can only ever fail to become a match; an unreadable,
    malformed, or non-matching candidate never aborts the scan of the
    remaining candidates, and this never trusts a candidate's self-declared
    ``manifest_sha256`` -- ``_read_partition_manifest_bytes`` always
    recomputes it first. Exactly one exact match is required; zero or more
    than one is fail-closed. Never returns or logs a candidate path outside
    this module's own private, in-memory control flow.
    """
    candidates_read_count = 0
    matches: list[tuple[Path, bytes]] = []
    for candidate_path in candidate_paths:
        try:
            candidate_raw = private_reader(candidate_path)
        except OSError:
            continue
        candidates_read_count += 1
        try:
            manifest = _read_partition_manifest_bytes(candidate_raw)
        except V8GPrivatePartitionLocatorBlocked:
            continue
        if (
            manifest["manifest_sha256"] == expected_partition_manifest_sha256
            and manifest["partition_implementation_git_commit"] == expected_partition_implementation_commit
        ):
            matches.append((candidate_path, candidate_raw))
    stats = {
        "candidate_count": len(candidate_paths),
        "candidates_read_count": candidates_read_count,
        "exact_match_count": len(matches),
    }
    if len(matches) == 0:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ZERO_MATCHING_CANDIDATES")
    if len(matches) > 1:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_MULTIPLE_MATCHING_CANDIDATES")
    matched_path, matched_raw = matches[0]
    return matched_path, matched_raw, stats


# ---------------------------------------------------------------------------
# 2.1.4 -- Safe locator artifact producer/verifier
# ---------------------------------------------------------------------------


def _build_locator_artifact(
    *,
    reviewed_v8g_design_candidate_commit: str,
    reviewed_locator_support_implementation_sha: str,
    candidate_count: int,
    candidate_set_sha256_value: str,
    selected_locator_path_sha256_value: str,
    expected_partition_manifest_sha256: str,
    expected_partition_implementation_commit: str,
    locator_gate_receipt_key_sha256_value: str,
    locator_gate_receipt_bytes_sha256_value: str,
) -> dict[str, Any]:
    artifact = {
        "schema_version": V8G_LOCATOR_ARTIFACT_SCHEMA_VERSION,
        "artifact_role": V8G_LOCATOR_ARTIFACT_ROLE,
        "study": V8G_STUDY_NAME,
        "reviewed_v8g_design_candidate_commit": reviewed_v8g_design_candidate_commit,
        "reviewed_locator_support_implementation_sha": reviewed_locator_support_implementation_sha,
        "predecessor_terminal_commit": V8F_PREDECESSOR_TERMINAL_COMMIT,
        "predecessor_terminal_artifact_blob": V8F_PREDECESSOR_TERMINAL_ARTIFACT_BLOB,
        "locator_contract": V8G_LOCATOR_CONTRACT,
        "candidate_count": candidate_count,
        "candidate_set_sha256": candidate_set_sha256_value,
        "selected_locator_path_sha256": selected_locator_path_sha256_value,
        "expected_partition_manifest_sha256": expected_partition_manifest_sha256,
        "expected_partition_implementation_commit": expected_partition_implementation_commit,
        "locator_gate_receipt_key_sha256": locator_gate_receipt_key_sha256_value,
        "locator_gate_receipt_bytes_sha256": locator_gate_receipt_bytes_sha256_value,
        "ticker_identities_exposed": False,
        "block_assignments_exposed": False,
        "raw_or_private_payload_persisted_publicly": False,
        "research_opened": False,
        "raw_acquisition_performed": False,
        "locator_result": "PASS",
    }
    if set(artifact) != set(V8G_LOCATOR_ARTIFACT_FIELDS):
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_SCHEMA_INVALID")
    return artifact


def _validate_locator_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(artifact, Mapping) or set(artifact) != set(V8G_LOCATOR_ARTIFACT_FIELDS):
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_SCHEMA_INVALID")
    exact = {
        "schema_version": V8G_LOCATOR_ARTIFACT_SCHEMA_VERSION,
        "artifact_role": V8G_LOCATOR_ARTIFACT_ROLE,
        "study": V8G_STUDY_NAME,
        "predecessor_terminal_commit": V8F_PREDECESSOR_TERMINAL_COMMIT,
        "predecessor_terminal_artifact_blob": V8F_PREDECESSOR_TERMINAL_ARTIFACT_BLOB,
        "locator_contract": V8G_LOCATOR_CONTRACT,
        "ticker_identities_exposed": False,
        "block_assignments_exposed": False,
        "raw_or_private_payload_persisted_publicly": False,
        "research_opened": False,
        "raw_acquisition_performed": False,
        "locator_result": "PASS",
    }
    for key, expected in exact.items():
        if artifact[key] != expected or (isinstance(expected, bool) and type(artifact[key]) is not bool):
            raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_VALUE_INVALID:" + key)
    _require_hex(
        artifact["reviewed_v8g_design_candidate_commit"], 40, "V8G_LOCATOR_ARTIFACT_DESIGN_CANDIDATE_INVALID"
    )
    _require_hex(
        artifact["reviewed_locator_support_implementation_sha"],
        40,
        "V8G_LOCATOR_ARTIFACT_IMPLEMENTATION_SHA_INVALID",
    )
    _require_hex(artifact["expected_partition_manifest_sha256"], 64, "V8G_LOCATOR_ARTIFACT_MANIFEST_SHA_INVALID")
    _require_hex(
        artifact["expected_partition_implementation_commit"],
        40,
        "V8G_LOCATOR_ARTIFACT_IMPLEMENTATION_COMMIT_INVALID",
    )
    _require_hex(artifact["candidate_set_sha256"], 64, "V8G_LOCATOR_ARTIFACT_CANDIDATE_SET_HASH_INVALID")
    _require_hex(artifact["selected_locator_path_sha256"], 64, "V8G_LOCATOR_ARTIFACT_PATH_HASH_INVALID")
    _require_hex(artifact["locator_gate_receipt_key_sha256"], 64, "V8G_LOCATOR_ARTIFACT_RECEIPT_KEY_INVALID")
    _require_hex(artifact["locator_gate_receipt_bytes_sha256"], 64, "V8G_LOCATOR_ARTIFACT_RECEIPT_BYTES_INVALID")
    if type(artifact["candidate_count"]) is not int or artifact["candidate_count"] < 1:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_CANDIDATE_COUNT_INVALID")
    return dict(artifact)


def verify_locator_artifact_binding(
    artifact: Mapping[str, Any],
    *,
    authorized_reviewed_v8g_design_candidate_commit: str,
    authorized_reviewed_locator_support_implementation_sha: str,
) -> dict[str, Any]:
    """The locator artifact verifier (design section 2.1.4).

    Independently requires exact equality on two distinct bindings and never
    compares them to each other: the artifact's own
    ``reviewed_v8g_design_candidate_commit`` against whatever design
    candidate commit is currently authorized for the caller's stage (the
    staleness check), and the artifact's own
    ``reviewed_locator_support_implementation_sha`` against the exact
    locator-support implementation SHA that *this artifact's own*
    independent review approved. This function only ever compares the
    locator-support SHA to a caller-supplied "authorized locator-support
    SHA" -- never to any other module's own reviewed implementation SHA
    (e.g. a future T1C-preservation-support SHA), so it cannot be misused to
    conflate the two.

    In addition, this verifier mechanically requires the artifact's own
    ``expected_partition_manifest_sha256``, ``expected_partition_implementation_commit``,
    and ``locator_gate_receipt_key_sha256`` to exactly equal this module's
    frozen expected values -- a structurally valid artifact bound to a
    different manifest identity, partition implementation, or a wrong
    deterministic receipt key is rejected here, before any future T1C
    preservation gate could be consumed on the strength of it.
    """
    validated = _validate_locator_artifact(artifact)
    if validated["reviewed_v8g_design_candidate_commit"] != authorized_reviewed_v8g_design_candidate_commit:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_DESIGN_CANDIDATE_MISMATCH")
    if (
        validated["reviewed_locator_support_implementation_sha"]
        != authorized_reviewed_locator_support_implementation_sha
    ):
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_IMPLEMENTATION_MISMATCH")
    if validated["expected_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_MANIFEST_SHA_MISMATCH")
    if validated["expected_partition_implementation_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_IMPLEMENTATION_COMMIT_MISMATCH")
    if validated["locator_gate_receipt_key_sha256"] != compute_locator_gate_receipt_key():
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_RECEIPT_KEY_MISMATCH")
    return validated


def _publish_locator_artifact(artifact: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Write-once atomic publication: canonical JSON, staging write, fsync
    file, atomic no-overwrite link, fsync directory, staging cleanup. Never
    replaces an existing destination."""
    if output.exists():
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_ALREADY_EXISTS")
    payload = _canonical_json_bytes(artifact)
    staging = output.parent / (output.name + ".staging-" + os.urandom(8).hex())
    try:
        with open(staging, "xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, output)
        except FileExistsError as error:
            raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_ALREADY_EXISTS") from error
        except OSError as error:
            raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_ATOMIC_PUBLISH_FAILED") from error
        _fsync_directory(output.parent)
    except V8GPrivatePartitionLocatorBlocked:
        raise
    except OSError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_ARTIFACT_WRITE_FAILED") from error
    finally:
        if staging.exists():
            try:
                staging.unlink()
            except OSError:
                pass
    return dict(artifact)


# ---------------------------------------------------------------------------
# Locator authority/design/implementation binding (public preflight +
# reviewed locator-support implementation runtime binding)
# ---------------------------------------------------------------------------

_PUBLIC_PREFLIGHT_FIELDS = frozenset(
    {
        "repository_identity",
        "branch",
        "head",
        "origin_head",
        "worktree_clean",
        "reviewed_v8g_design_candidate_commit",
        "reviewed_v8g_design_blob_sha",
    }
)


def _validate_public_preflight(preflight: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(preflight, Mapping) or set(preflight) != _PUBLIC_PREFLIGHT_FIELDS:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PUBLIC_PREFLIGHT_SCHEMA_INVALID")
    if preflight["repository_identity"] != V8G_REPOSITORY_IDENTITY:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    if preflight["branch"] != V8G_PRODUCTION_BRANCH or preflight["worktree_clean"] is not True:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PUBLIC_GIT_BINDING_INVALID")
    head = _require_hex(preflight["head"], 40, "V8G_PUBLIC_HEAD_INVALID")
    if preflight["origin_head"] != head:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PUBLIC_HEAD_NOT_ORIGIN")
    if preflight["reviewed_v8g_design_candidate_commit"] != REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT:
        raise V8GPrivatePartitionLocatorBlocked("V8G_DESIGN_CANDIDATE_MISMATCH")
    if preflight["reviewed_v8g_design_blob_sha"] != V8G_DESIGN_CANDIDATE_BLOB_SHA:
        raise V8GPrivatePartitionLocatorBlocked("V8G_DESIGN_CANDIDATE_BLOB_MISMATCH")
    return dict(preflight)


def _git_text(repository_root: Path, args: list[str], reason: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PUBLIC_GIT_UNAVAILABLE") from error
    if result.returncode != 0:
        raise V8GPrivatePartitionLocatorBlocked(reason)
    return result.stdout.strip()


def _default_public_preflight(repository_root: Path = CANONICAL_REPOSITORY_ROOT) -> dict[str, Any]:
    status = _git_text(repository_root, ["status", "--porcelain"], "V8G_PUBLIC_GIT_UNAVAILABLE")
    branch = _git_text(repository_root, ["branch", "--show-current"], "V8G_PUBLIC_BRANCH_UNAVAILABLE")
    head = _git_text(repository_root, ["rev-parse", "HEAD"], "V8G_PUBLIC_HEAD_UNAVAILABLE")
    origin_head = _git_text(
        repository_root, ["rev-parse", "origin/" + V8G_PRODUCTION_BRANCH], "V8G_PUBLIC_ORIGIN_UNAVAILABLE"
    )
    origin_url = _git_text(repository_root, ["config", "--get", "remote.origin.url"], "V8G_PUBLIC_ORIGIN_UNAVAILABLE")
    if origin_url not in {
        "https://github.com/ta1k1-arakawa/stock-analyzer",
        "https://github.com/ta1k1-arakawa/stock-analyzer.git",
        "git@github.com:ta1k1-arakawa/stock-analyzer.git",
        "ssh://git@github.com/ta1k1-arakawa/stock-analyzer.git",
    }:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PUBLIC_REPOSITORY_IDENTITY_MISMATCH")
    try:
        design_blob = resolve_git_blob(repository_root, REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT, V8G_DESIGN_DRAFT_GIT_PATH)
    except V8CGitProvenanceBlocked as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PUBLIC_PROVENANCE_INVALID") from error
    if design_blob != V8G_DESIGN_CANDIDATE_BLOB_SHA:
        raise V8GPrivatePartitionLocatorBlocked("V8G_PUBLIC_PROVENANCE_INVALID")
    return _validate_public_preflight(
        {
            "repository_identity": V8G_REPOSITORY_IDENTITY,
            "branch": branch,
            "head": head,
            "origin_head": origin_head,
            "worktree_clean": status == "",
            "reviewed_v8g_design_candidate_commit": REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
            "reviewed_v8g_design_blob_sha": design_blob,
        }
    )


_REVIEWED_IMPLEMENTATION_RUNTIME_FIELDS = frozenset(
    {"branch", "head", "origin_head", "worktree_clean", "commits_after_reviewed_implementation_sha"}
)


def _default_reviewed_locator_support_runtime_state(
    repository_root: Path, reviewed_locator_support_implementation_sha: str
) -> dict[str, Any]:
    reviewed_sha = _require_hex(
        reviewed_locator_support_implementation_sha, 40, "V8G_LOCATOR_IMPLEMENTATION_SHA_MALFORMED"
    )
    resolved_sha = _git_text(
        repository_root,
        ["rev-parse", "--verify", f"{reviewed_sha}^{{commit}}"],
        "V8G_LOCATOR_IMPLEMENTATION_SHA_UNRESOLVABLE",
    )
    if resolved_sha != reviewed_sha:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_IMPLEMENTATION_SHA_UNRESOLVABLE")
    count_text = _git_text(
        repository_root,
        ["rev-list", "--count", f"{reviewed_sha}..HEAD"],
        "V8G_LOCATOR_IMPLEMENTATION_CHRONOLOGY_INVALID",
    )
    if not count_text.isdecimal():
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_IMPLEMENTATION_CHRONOLOGY_INVALID")
    return {
        "branch": _git_text(repository_root, ["branch", "--show-current"], "V8G_PUBLIC_BRANCH_UNAVAILABLE"),
        "head": _git_text(repository_root, ["rev-parse", "HEAD"], "V8G_PUBLIC_HEAD_UNAVAILABLE"),
        "origin_head": _git_text(
            repository_root, ["rev-parse", "origin/" + V8G_PRODUCTION_BRANCH], "V8G_PUBLIC_ORIGIN_UNAVAILABLE"
        ),
        "worktree_clean": _git_text(repository_root, ["status", "--porcelain"], "V8G_PUBLIC_GIT_UNAVAILABLE") == "",
        "commits_after_reviewed_implementation_sha": int(count_text),
    }


def _validate_reviewed_locator_support_implementation_binding(
    repository_root: Path,
    reviewed_locator_support_implementation_sha: str,
    *,
    runtime_state_reader: Callable[[Path, str], Mapping[str, Any]] | None = None,
) -> str:
    reviewed_sha = _require_hex(
        reviewed_locator_support_implementation_sha, 40, "V8G_LOCATOR_IMPLEMENTATION_SHA_MALFORMED"
    )
    runtime = (runtime_state_reader or _default_reviewed_locator_support_runtime_state)(repository_root, reviewed_sha)
    if not isinstance(runtime, Mapping) or set(runtime) != _REVIEWED_IMPLEMENTATION_RUNTIME_FIELDS:
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_IMPLEMENTATION_RUNTIME_SCHEMA_INVALID")
    if (
        runtime["branch"] != V8G_PRODUCTION_BRANCH
        or runtime["head"] != reviewed_sha
        or runtime["origin_head"] != reviewed_sha
        or runtime["worktree_clean"] is not True
        or type(runtime["commits_after_reviewed_implementation_sha"]) is not int
        or runtime["commits_after_reviewed_implementation_sha"] != 0
    ):
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_IMPLEMENTATION_RUNTIME_BINDING_INVALID")
    return reviewed_sha


# ---------------------------------------------------------------------------
# Full DI execution boundary
# ---------------------------------------------------------------------------


def _require_safe_external_path(value: str | os.PathLike[str], repository_root: Path, reason: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise V8GPrivatePartitionLocatorBlocked(reason)
    resolved = path.resolve(strict=False)
    return _require_outside_repository(resolved, repository_root, reason)


def _prepare_locator_execution_paths(
    *,
    state_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    candidates: Sequence[Path],
    repository_root: Path,
) -> tuple[Path, Path]:
    state = _require_safe_external_path(state_root, repository_root, "V8G_STATE_PATH_INVALID")
    output = _require_safe_external_path(output_path, repository_root, "V8G_OUTPUT_PATH_INVALID")
    if output in candidates:
        raise V8GPrivatePartitionLocatorBlocked("V8G_OUTPUT_PATH_COLLISION")
    try:
        state.mkdir(parents=True, exist_ok=True)
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise V8GPrivatePartitionLocatorBlocked("V8G_OUTPUT_OR_STATE_PREPARATION_FAILED") from error
    if not state.is_dir() or not output.parent.is_dir() or output.exists():
        raise V8GPrivatePartitionLocatorBlocked("V8G_OUTPUT_OR_STATE_PREPARATION_FAILED")
    if (state / (compute_locator_gate_receipt_key() + ".json")).exists():
        raise V8GPrivatePartitionLocatorBlocked("V8G_LOCATOR_GATE_ALREADY_CONSUMED")
    return state, output


def _execute_locator_with_dependencies(
    *,
    authorization_identity: str,
    state_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    candidate_enumerator: Callable[[], Sequence[str | os.PathLike[str]]],
    repository_root: Path,
    public_preflight: Callable[[], Mapping[str, Any]],
    private_reader: Callable[[Path], bytes],
    gate_consumer: Callable[..., Mapping[str, Any]],
    clock: Callable[[], datetime],
    reviewed_locator_support_implementation_sha: str,
    runtime_state_reader: Callable[[Path, str], Mapping[str, Any]] | None = None,
    reviewed_v8g_design_candidate_commit: str = REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
    expected_partition_manifest_sha256: str = EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    expected_partition_implementation_commit: str = EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
) -> dict[str, Any]:
    """DI-only future execution boundary; never called with real private
    candidate paths by this implementation task."""
    _validate_public_preflight(public_preflight())
    reviewed_impl_sha = _validate_reviewed_locator_support_implementation_binding(
        repository_root, reviewed_locator_support_implementation_sha, runtime_state_reader=runtime_state_reader
    )
    validate_authorization_identity(
        authorization_identity,
        reviewed_locator_support_implementation_sha=reviewed_impl_sha,
        reviewed_v8g_design_candidate_commit=reviewed_v8g_design_candidate_commit,
        expected_partition_manifest_sha256=expected_partition_manifest_sha256,
        expected_partition_implementation_commit=expected_partition_implementation_commit,
    )
    # Pre-gate, metadata-only: authoritative enumeration, then
    # normalize/dedupe/basename/outside-repo validation.
    raw_candidates = candidate_enumerator()
    candidates = validate_candidate_partition_manifest_paths(raw_candidates, repository_root)
    state, output = _prepare_locator_execution_paths(
        state_root=state_root, output_path=output_path, candidates=candidates, repository_root=repository_root
    )

    # Exact frozen boundary: no private reader is called before durable receipt.
    gate_consumer(
        state,
        authorization_identity,
        clock=clock,
        reviewed_v8g_design_candidate_commit=reviewed_v8g_design_candidate_commit,
        reviewed_locator_support_implementation_sha=reviewed_impl_sha,
        expected_partition_manifest_sha256=expected_partition_manifest_sha256,
        expected_partition_implementation_commit=expected_partition_implementation_commit,
    )

    matched_path, _matched_raw, stats = _locate_authorized_partition_manifest(
        private_reader,
        candidates,
        expected_partition_manifest_sha256=expected_partition_manifest_sha256,
        expected_partition_implementation_commit=expected_partition_implementation_commit,
    )

    receipt_key = compute_locator_gate_receipt_key()
    # Post-gate, pre-artifact-publication: the durable receipt must be
    # semantically bound to this exact execution's authorized values, not
    # merely structurally well-formed.
    _receipt, receipt_bytes_sha = _read_and_bind_gate_receipt(
        state,
        reviewed_v8g_design_candidate_commit=reviewed_v8g_design_candidate_commit,
        reviewed_locator_support_implementation_sha=reviewed_impl_sha,
        expected_partition_manifest_sha256=expected_partition_manifest_sha256,
        expected_partition_implementation_commit=expected_partition_implementation_commit,
        authorization_identity=authorization_identity,
    )
    sorted_hash_list = sorted(locator_path_sha256(candidate) for candidate in candidates)
    artifact = _build_locator_artifact(
        reviewed_v8g_design_candidate_commit=reviewed_v8g_design_candidate_commit,
        reviewed_locator_support_implementation_sha=reviewed_impl_sha,
        candidate_count=stats["candidate_count"],
        candidate_set_sha256_value=candidate_set_sha256(sorted_hash_list),
        selected_locator_path_sha256_value=locator_path_sha256(matched_path),
        expected_partition_manifest_sha256=expected_partition_manifest_sha256,
        expected_partition_implementation_commit=expected_partition_implementation_commit,
        locator_gate_receipt_key_sha256_value=receipt_key,
        locator_gate_receipt_bytes_sha256_value=receipt_bytes_sha,
    )
    _publish_locator_artifact(artifact, output)
    return {
        "result": "PASS",
        "candidate_count": stats["candidate_count"],
        "candidates_read_count": stats["candidates_read_count"],
        "exact_match_count": stats["exact_match_count"],
        "artifact_written": True,
        "expected_partition_manifest_sha256": expected_partition_manifest_sha256,
        "locator_gate_receipt_key_sha256": receipt_key,
    }


def resolve_and_locate_authorized_partition_manifest(
    authorization_identity: str,
    *,
    reviewed_locator_support_implementation_sha: str,
    output_path: str | os.PathLike[str],
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Prepared future entry point; not executed by this support task.

    Candidates are always derived from the authoritative
    `enumerate_candidate_partition_manifest_paths` enumerator (design
    section 2.1.1); a normal production caller cannot supply or override an
    arbitrary candidate list.
    """
    return _execute_locator_with_dependencies(
        authorization_identity=authorization_identity,
        state_root=CANONICAL_V8G_LOCATOR_GATE_STATE_ROOT,
        output_path=output_path,
        candidate_enumerator=lambda: enumerate_candidate_partition_manifest_paths(
            repository_root=CANONICAL_REPOSITORY_ROOT
        ),
        repository_root=CANONICAL_REPOSITORY_ROOT,
        public_preflight=lambda: _default_public_preflight(CANONICAL_REPOSITORY_ROOT),
        private_reader=lambda path: path.read_bytes(),
        gate_consumer=consume_gate_once,
        clock=clock or (lambda: datetime.now(timezone.utc)),
        reviewed_locator_support_implementation_sha=reviewed_locator_support_implementation_sha,
    )


__all__ = [
    "CANONICAL_V8G_LOCATOR_GATE_STATE_ROOT",
    "EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT",
    "EXPECTED_V8_PARTITION_MANIFEST_SHA256",
    "PARTITION_MANIFEST_BASENAME",
    "REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT",
    "V8F_PREDECESSOR_TERMINAL_ARTIFACT_BLOB",
    "V8F_PREDECESSOR_TERMINAL_COMMIT",
    "V8G_DESIGN_CANDIDATE_BLOB_SHA",
    "V8G_LOCATOR_ARTIFACT_FIELDS",
    "V8G_LOCATOR_CONTRACT",
    "V8G_LOCATOR_GATE",
    "V8G_LOCATOR_GATE_CONSUMPTION_BOUNDARY",
    "V8G_LOCATOR_RECEIPT_FIELDS",
    "V8G_STUDY_NAME",
    "V8GPrivatePartitionLocatorBlocked",
    "authorization_identity_sha256",
    "build_authorization_identity",
    "candidate_set_sha256",
    "candidate_set_serialization_v1",
    "canonical_path_text",
    "compute_locator_gate_receipt_key",
    "consume_gate_once",
    "enumerate_candidate_partition_manifest_paths",
    "gate_receipt_bytes_sha256",
    "locator_path_sha256",
    "read_gate_receipt",
    "resolve_and_locate_authorized_partition_manifest",
    "validate_authorization_identity",
    "validate_candidate_partition_manifest_paths",
    "verify_locator_artifact_binding",
]
