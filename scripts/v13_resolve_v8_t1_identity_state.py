"""Selectively resolve V8 T1 membership for the later approved V13 gate.

This module has no command-line entry point and imports only the standard
library. The public resolver is bound to the reviewed public manifest and T1
hashes. Synthetic tests use the private lower-level helper with synthetic
bindings; the private source is never read by this module during this task.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable


EXPECTED_MANIFEST_STATED_SHA256 = "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
EXPECTED_T1_TICKER_LIST_SHA256 = "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d"
EXPECTED_T1_COUNT = 300
KNOWN_DEFINITELY_ACQUIRED_PREFIX_COUNT = 297
EXPECTED_SCHEMA_VERSION = "V8_PARTITION_MANIFEST_V3"

_ASSIGNMENT_KEYS = frozenset({"T0", "T1", "T2", "T3", "T_spare"})
_BLOCK_SIZE_KEYS = _ASSIGNMENT_KEYS
_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "study_name",
        "design_commit",
        "source_snapshot_semantics",
        "source_snapshot_clarification_commit",
        "partition_implementation_git_commit",
        "created_utc",
        "source_url",
        "source_host",
        "source_acquisition_utc",
        "source_raw_sha256",
        "source_raw_byte_count",
        "v4_source_raw_sha256_reference",
        "v4_raw_sha_equality_required",
        "source_reproduction_status",
        "t0_reproduction_status",
        "eligible_ticker_count",
        "eligible_ticker_list_sha256",
        "selection_rule",
        "deterministic_ordering_rule",
        "t0_ticker_list_sha256",
        "t1_ticker_list_sha256",
        "t2_ticker_list_sha256",
        "t3_ticker_list_sha256",
        "t_spare_ticker_list_sha256",
        "legacy_exclude_list",
        "legacy_exclude_list_sha256",
        "block_sizes",
        "block_assignments",
        "p_hist_start",
        "p_hist_end",
        "t1_role",
        "t2_role",
        "t3_role",
        "t3_price_acquisition_authorized",
        "manifest_sha256",
    }
)
_STRING_ROOT_KEYS = _ROOT_KEYS - {
    "source_raw_byte_count",
    "eligible_ticker_count",
    "v4_raw_sha_equality_required",
    "legacy_exclude_list",
    "block_sizes",
    "block_assignments",
    "t3_price_acquisition_authorized",
}
_INTEGER_ROOT_KEYS = frozenset({"source_raw_byte_count", "eligible_ticker_count"})
_BOOLEAN_ROOT_KEYS = frozenset(
    {"v4_raw_sha_equality_required", "t3_price_acquisition_authorized"}
)
_TICKER_PATTERN = re.compile(r"[0-9A-Z]{4}\Z")


class IdentityResolutionBlocked(RuntimeError):
    """Fail-closed error whose reason never contains source identities/paths."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class _ExpectedBindings:
    manifest_stated_sha256: str
    t1_ticker_list_sha256: str
    t1_count: int


_PRODUCTION_BINDINGS = _ExpectedBindings(
    manifest_stated_sha256=EXPECTED_MANIFEST_STATED_SHA256,
    t1_ticker_list_sha256=EXPECTED_T1_TICKER_LIST_SHA256,
    t1_count=EXPECTED_T1_COUNT,
)


class _SelectiveManifestScanner:
    """Strict scanner that materializes only selected public scalars and T1."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw
        self._position = 0

    def _block(self) -> None:
        raise IdentityResolutionBlocked("MANIFEST_INVALID_OR_AMBIGUOUS")

    def _skip_whitespace(self) -> None:
        raw = self._raw
        while self._position < len(raw) and raw[self._position] in b" \t\r\n":
            self._position += 1

    def _expect(self, token: bytes) -> None:
        self._skip_whitespace()
        if not self._raw.startswith(token, self._position):
            self._block()
        self._position += len(token)

    def _scan_utf8_scalar(self, position: int) -> int:
        raw = self._raw
        first = raw[position]
        if 0xC2 <= first <= 0xDF:
            width = 2
        elif 0xE0 <= first <= 0xEF:
            width = 3
        elif 0xF0 <= first <= 0xF4:
            width = 4
        else:
            self._block()
        end = position + width
        if end > len(raw):
            self._block()
        second = raw[position + 1]
        for index in range(position + 1, end):
            if raw[index] < 0x80 or raw[index] > 0xBF:
                self._block()
        if width == 3 and first == 0xE0 and second < 0xA0:
            self._block()
        if width == 3 and first == 0xED and second > 0x9F:
            self._block()
        if width == 4 and first == 0xF0 and second < 0x90:
            self._block()
        if width == 4 and first == 0xF4 and second > 0x8F:
            self._block()
        return end

    def _skip_hex_quad(self) -> None:
        start = self._position
        end = start + 4
        if end > len(self._raw):
            self._block()
        for index in range(start, end):
            byte = self._raw[index]
            if not (
                ord("0") <= byte <= ord("9")
                or ord("a") <= byte <= ord("f")
                or ord("A") <= byte <= ord("F")
            ):
                self._block()
        self._position = end

    def _scan_string(self) -> tuple[int, int]:
        self._skip_whitespace()
        if self._position >= len(self._raw) or self._raw[self._position] != ord('"'):
            self._block()
        token_start = self._position
        self._position += 1
        while self._position < len(self._raw):
            byte = self._raw[self._position]
            if byte == ord('"'):
                token_end = self._position + 1
                self._position = token_end
                return token_start, token_end
            if byte < 0x20:
                self._block()
            if byte == ord("\\"):
                self._position += 1
                if self._position >= len(self._raw):
                    self._block()
                escaped = self._raw[self._position]
                self._position += 1
                if escaped in b'"\\/bfnrt':
                    continue
                if escaped != ord("u"):
                    self._block()
                self._skip_hex_quad()
                continue
            if byte < 0x80:
                self._position += 1
            else:
                self._position = self._scan_utf8_scalar(self._position)
        self._block()

    def _read_ascii_string(self, *, decode: bool) -> str | None:
        start, end = self._scan_string()
        if not decode:
            return None
        contents = self._raw[start + 1 : end - 1]
        if b"\\" in contents or any(byte >= 0x80 for byte in contents):
            self._block()
        return contents.decode("ascii")

    def _read_object_key(self) -> str:
        key = self._read_ascii_string(decode=True)
        assert key is not None
        self._skip_whitespace()
        self._expect(b":")
        return key

    def _read_integer(self) -> int:
        start, end = self._scan_integer()
        try:
            value = int(self._raw[start:end])
        except ValueError:
            self._block()
        return value

    def _skip_integer(self) -> None:
        self._scan_integer()

    def _scan_integer(self) -> tuple[int, int]:
        self._skip_whitespace()
        start = self._position
        raw = self._raw
        if self._position < len(raw) and raw[self._position] == ord("-"):
            self._position += 1
        if self._position >= len(raw):
            self._block()
        if raw[self._position] == ord("0"):
            self._position += 1
            if self._position < len(raw) and ord("0") <= raw[self._position] <= ord("9"):
                self._block()
        elif ord("1") <= raw[self._position] <= ord("9"):
            self._position += 1
            while self._position < len(raw) and ord("0") <= raw[self._position] <= ord("9"):
                self._position += 1
        else:
            self._block()
        end = self._position
        if raw[start] == ord("-"):
            self._block()
        return start, end

    def _read_boolean(self) -> bool:
        self._skip_whitespace()
        if self._raw.startswith(b"true", self._position):
            self._position += 4
            return True
        if self._raw.startswith(b"false", self._position):
            self._position += 5
            return False
        self._block()

    def _read_string_array(self, *, collect: bool) -> list[str] | None:
        self._expect(b"[")
        result: list[str] | None = [] if collect else None
        self._skip_whitespace()
        if self._position < len(self._raw) and self._raw[self._position] == ord("]"):
            self._position += 1
            return result
        while True:
            if collect:
                value = self._read_ascii_string(decode=True)
                assert value is not None and result is not None
                result.append(value)
            else:
                self._read_ascii_string(decode=False)
            self._skip_whitespace()
            if self._position >= len(self._raw):
                self._block()
            delimiter = self._raw[self._position]
            self._position += 1
            if delimiter == ord("]"):
                return result
            if delimiter != ord(","):
                self._block()

    def _read_assignments(self) -> list[str]:
        self._expect(b"{")
        seen: set[str] = set()
        t1_membership: list[str] | None = None
        self._skip_whitespace()
        if self._position < len(self._raw) and self._raw[self._position] == ord("}"):
            self._block()
        while True:
            key = self._read_object_key()
            if key in seen or key not in _ASSIGNMENT_KEYS:
                self._block()
            seen.add(key)
            values = self._read_string_array(collect=(key == "T1"))
            if key == "T1":
                assert values is not None
                t1_membership = values
            self._skip_whitespace()
            if self._position >= len(self._raw):
                self._block()
            delimiter = self._raw[self._position]
            self._position += 1
            if delimiter == ord("}"):
                break
            if delimiter != ord(","):
                self._block()
        if seen != _ASSIGNMENT_KEYS or t1_membership is None:
            self._block()
        return t1_membership

    def _read_block_sizes(self) -> int:
        self._expect(b"{")
        seen: set[str] = set()
        t1_count: int | None = None
        self._skip_whitespace()
        if self._position < len(self._raw) and self._raw[self._position] == ord("}"):
            self._block()
        while True:
            key = self._read_object_key()
            if key in seen or key not in _BLOCK_SIZE_KEYS:
                self._block()
            seen.add(key)
            if key == "T1":
                t1_count = self._read_integer()
            else:
                self._skip_integer()
            self._skip_whitespace()
            if self._position >= len(self._raw):
                self._block()
            delimiter = self._raw[self._position]
            self._position += 1
            if delimiter == ord("}"):
                break
            if delimiter != ord(","):
                self._block()
        if seen != _BLOCK_SIZE_KEYS or t1_count is None:
            self._block()
        return t1_count

    def read_t1(self) -> tuple[str, str, str, int, list[str]]:
        self._expect(b"{")
        seen: set[str] = set()
        stated_manifest_sha: str | None = None
        stated_t1_sha: str | None = None
        schema_version: str | None = None
        t1_membership: list[str] | None = None
        block_sizes_t1_count: int | None = None
        self._skip_whitespace()
        if self._position < len(self._raw) and self._raw[self._position] == ord("}"):
            self._block()
        while True:
            key = self._read_object_key()
            if key in seen or key not in _ROOT_KEYS:
                self._block()
            seen.add(key)
            if key in _STRING_ROOT_KEYS:
                value = self._read_ascii_string(
                    decode=key in {"schema_version", "manifest_sha256", "t1_ticker_list_sha256"}
                )
                if key == "schema_version":
                    schema_version = value
                elif key == "manifest_sha256":
                    stated_manifest_sha = value
                elif key == "t1_ticker_list_sha256":
                    stated_t1_sha = value
            elif key in _INTEGER_ROOT_KEYS:
                self._skip_integer()
            elif key in _BOOLEAN_ROOT_KEYS:
                value = self._read_boolean()
                if key in {"v4_raw_sha_equality_required", "t3_price_acquisition_authorized"} and value:
                    self._block()
            elif key == "legacy_exclude_list":
                self._read_string_array(collect=False)
            elif key == "block_sizes":
                block_sizes_t1_count = self._read_block_sizes()
            elif key == "block_assignments":
                t1_membership = self._read_assignments()
            else:
                self._block()
            self._skip_whitespace()
            if self._position >= len(self._raw):
                self._block()
            delimiter = self._raw[self._position]
            self._position += 1
            if delimiter == ord("}"):
                break
            if delimiter != ord(","):
                self._block()
        self._skip_whitespace()
        if self._position != len(self._raw):
            self._block()
        if seen != _ROOT_KEYS:
            self._block()
        if (
            schema_version != EXPECTED_SCHEMA_VERSION
            or stated_manifest_sha is None
            or stated_t1_sha is None
            or t1_membership is None
            or block_sizes_t1_count is None
        ):
            self._block()
        return stated_manifest_sha, stated_t1_sha, schema_version, block_sizes_t1_count, t1_membership


def _validate_membership(
    members: list[str], block_sizes_t1_count: int, bindings: _ExpectedBindings
) -> str:
    if len(members) != bindings.t1_count or block_sizes_t1_count != bindings.t1_count:
        raise IdentityResolutionBlocked("T1_COUNT_MISMATCH")
    seen: set[str] = set()
    for member in members:
        if _TICKER_PATTERN.fullmatch(member) is None:
            raise IdentityResolutionBlocked("T1_CODE_FORMAT_INVALID")
        if member in seen:
            raise IdentityResolutionBlocked("T1_DUPLICATE_CODE")
        seen.add(member)
    try:
        actual_sha = hashlib.sha256(("\n".join(members) + "\n").encode("utf-8")).hexdigest()
    except (UnicodeEncodeError, TypeError):
        raise IdentityResolutionBlocked("T1_HASH_COMPUTATION_FAILED") from None
    if actual_sha != bindings.t1_ticker_list_sha256:
        raise IdentityResolutionBlocked("T1_HASH_MISMATCH")
    return actual_sha


def _validate_output_destination(
    output_path: str | os.PathLike[str], repository_root: str | os.PathLike[str]
) -> Path:
    try:
        candidate = Path(output_path)
        repository = Path(repository_root).resolve()
        if not candidate.is_absolute() or candidate.name in {"", ".", ".."}:
            raise IdentityResolutionBlocked("OUTPUT_PATH_INVALID")
        resolved_parent = candidate.parent.resolve()
        destination = resolved_parent / candidate.name
    except IdentityResolutionBlocked:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise IdentityResolutionBlocked("OUTPUT_PATH_INVALID") from None
    try:
        destination.relative_to(repository)
    except ValueError:
        return destination
    else:
        raise IdentityResolutionBlocked("OUTPUT_PATH_INSIDE_REPOSITORY")


def _ensure_output_does_not_exist(destination: Path) -> None:
    try:
        exists = os.path.lexists(destination)
    except (OSError, TypeError, ValueError):
        raise IdentityResolutionBlocked("OUTPUT_PATH_INVALID") from None
    if exists:
        raise IdentityResolutionBlocked("OUTPUT_ALREADY_EXISTS")


def _publish_windows_write_through(staging_path: str, destination: Path) -> None:
    """Publish by same-directory no-replace rename and wait for disk commit."""
    import ctypes

    move_file_ex = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
    move_file_ex.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    move_file_ex.restype = ctypes.c_int
    # MOVEFILE_WRITE_THROUGH; omitting REPLACE_EXISTING preserves no-overwrite.
    if not move_file_ex(staging_path, str(destination), 0x00000008):
        error = ctypes.get_last_error()
        if error in {80, 183}:  # ERROR_FILE_EXISTS / ERROR_ALREADY_EXISTS
            raise FileExistsError(error, "destination already exists", str(destination))
        raise OSError(error, "durable Windows publication failed", str(destination))


def _flush_windows_file(destination: Path) -> None:
    """Flush the published file after its write-through namespace move."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
    )
    create_file.restype = ctypes.c_void_p
    flush_file = kernel32.FlushFileBuffers
    flush_file.argtypes = (ctypes.c_void_p,)
    flush_file.restype = ctypes.c_int
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (ctypes.c_void_p,)
    close_handle.restype = ctypes.c_int
    # GENERIC_WRITE; share read/write/delete; OPEN_EXISTING.
    handle = create_file(str(destination), 0x40000000, 0x00000001 | 0x00000002 | 0x00000004,
                         None, 3, 0x00000080, None)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        raise OSError(ctypes.get_last_error(), "published file open for flush failed", str(destination))
    try:
        if not flush_file(handle):
            raise OSError(ctypes.get_last_error(), "published file flush failed", str(destination))
    finally:
        close_handle(handle)


def _flush_directory(directory: Path) -> None:
    """Persist POSIX directory-entry changes after publication."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_once(destination: Path, payload: bytes) -> None:
    staging_path: str | None = None
    descriptor: int | None = None
    published = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _ensure_output_does_not_exist(destination)
        descriptor, staging_path = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=str(destination.parent)
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                _publish_windows_write_through(staging_path, destination)
                staging_path = None  # MoveFileExW consumed the staging name.
                published = True
                _flush_windows_file(destination)
            else:
                os.link(staging_path, destination)
                published = True
                os.unlink(staging_path)
                staging_path = None
                _flush_directory(destination.parent)
        except FileExistsError:
            raise IdentityResolutionBlocked("OUTPUT_ALREADY_EXISTS") from None
        except OSError:
            reason = "OUTPUT_DURABILITY_FLUSH_FAILED" if published else "ATOMIC_OUTPUT_PUBLISH_FAILED"
            raise IdentityResolutionBlocked(reason) from None
    except IdentityResolutionBlocked:
        raise
    except (OSError, TypeError, ValueError):
        raise IdentityResolutionBlocked("OUTPUT_WRITE_FAILED") from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if staging_path is not None:
            try:
                os.unlink(staging_path)
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _resolve_identity_state(
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    *,
    bindings: _ExpectedBindings,
    on_first_byte: Callable[[], None] | None = None,
    on_state_written: Callable[[], None] | None = None,
    source_opener: Callable[[Path], BinaryIO] | None = None,
) -> dict[str, Any]:
    """Private seam used by synthetic tests; production uses fixed bindings."""
    destination = _validate_output_destination(output_path, repository_root)
    _ensure_output_does_not_exist(destination)
    try:
        source = Path(manifest_path)
        opener = source_opener or (lambda path: path.open("rb"))
        with opener(source) as stream:
            first_byte = stream.read(1)
            if not first_byte:
                raise IdentityResolutionBlocked("MANIFEST_INVALID_OR_AMBIGUOUS")
            if on_first_byte is not None:
                try:
                    on_first_byte()
                except Exception:
                    raise IdentityResolutionBlocked("POST_BOUNDARY_RECEIPT_PUBLISH_FAILED") from None
            try:
                raw = first_byte + stream.read()
            except Exception:
                raise IdentityResolutionBlocked("SOURCE_READ_FAILED_POST_BOUNDARY") from None
    except IdentityResolutionBlocked:
        raise
    except (OSError, TypeError, ValueError):
        raise IdentityResolutionBlocked("SOURCE_READ_FAILED") from None
    scanner = _SelectiveManifestScanner(raw)
    stated_manifest_sha, stated_t1_sha, _schema, block_sizes_t1_count, members = scanner.read_t1()
    if stated_manifest_sha != bindings.manifest_stated_sha256:
        raise IdentityResolutionBlocked("MANIFEST_STATED_SHA_MISMATCH")
    if stated_t1_sha != bindings.t1_ticker_list_sha256:
        raise IdentityResolutionBlocked("T1_STATED_SHA_MISMATCH")
    actual_t1_sha = _validate_membership(members, block_sizes_t1_count, bindings)
    state: dict[str, Any] = {
        "schema": "V13_V8_T1_IDENTITY_STATE_V1",
        "source_partition_manifest_stated_sha256": stated_manifest_sha,
        "t1_ticker_list_sha256": actual_t1_sha,
        "t1_count": len(members),
        "known_definitely_acquired_prefix_count": KNOWN_DEFINITELY_ACQUIRED_PREFIX_COUNT,
        "t1_membership": members,
    }
    payload = (json.dumps(state, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    _write_once(destination, payload)
    if on_state_written is not None:
        try:
            on_state_written()
        except Exception:
            raise IdentityResolutionBlocked("POST_BOUNDARY_REPORTING_FAILED") from None
    return state


def resolve_identity_state(
    manifest_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    *,
    on_first_byte: Callable[[], None] | None = None,
    on_state_written: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Resolve only T1 and atomically persist the bound private state once.

    The manifest path is explicit. No network request, full-object JSON load,
    full-manifest hash, or console output is performed.
    """
    return _resolve_identity_state(
        manifest_path,
        output_path,
        repository_root,
        bindings=_PRODUCTION_BINDINGS,
        on_first_byte=on_first_byte,
        on_state_written=on_state_written,
    )


__all__ = [
    "EXPECTED_MANIFEST_STATED_SHA256",
    "EXPECTED_T1_COUNT",
    "EXPECTED_T1_TICKER_LIST_SHA256",
    "IdentityResolutionBlocked",
    "resolve_identity_state",
]
