"""Dependency-injected durable-state adapter for the Stage-1 acquisition core."""
from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Callable

from src import v9_006_f1_semantic_successor_public_acquisition as acquisition
from src import v9_006_f1_semantic_successor_locator as locator
from src.v9_005_stage_a_jpx_probe import validate_jpx_url

_RECEIPT = "execution-start-receipt.json"
_RAW = "raw"
_PAYLOAD = "payload.bin"
_METADATA = "metadata.json"
_PERIODS = frozenset({acquisition.ROOT_PERIOD, acquisition.TERMINAL_PERIOD})


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_exclusive(path: Path, content: bytes) -> bool:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except (FileExistsError, OSError):
        return False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        return False
    return True


def _verify_existing_durable(path: Path, expected: bytes) -> bool:
    """Re-prove an exact preserved component without changing its pathname."""
    try:
        descriptor = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        with os.fdopen(descriptor, "r+b", closefd=True) as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return False
            if handle.read() != expected:
                return False
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        return False
    return True


def _receipt(implementation_git_sha: str) -> dict[str, object]:
    return {
        "task": acquisition.TASK,
        "design_git_sha": acquisition.DESIGN_GIT_SHA,
        "implementation_git_sha": implementation_git_sha,
        "operation_class": acquisition.OPERATION_CLASS,
        "execution_started": True,
    }


def _conflict(implementation_git_sha: str) -> dict[str, object]:
    return acquisition.finalize_safe_result(acquisition._base(implementation_git_sha, "GOVERNANCE_FAILURE", "EXECUTION_BINDING_CONFLICT"))


class DurableState:
    """Private raw-lock state rooted at a caller-supplied local directory."""

    def __init__(self, root: Path):
        self.root = root

    def start(self, implementation_git_sha: str) -> bool:
        if self.root.exists():
            if not self.root.is_dir() or any(self.root.iterdir()):
                return False
        else:
            try:
                self.root.mkdir(parents=True)
            except OSError:
                return False
        return _write_exclusive(self.root / _RECEIPT, _canonical(_receipt(implementation_git_sha)))

    def _directory(self, period: str) -> Path:
        return self.root / _RAW / period

    def _metadata(self, period: str, payload: bytes, resolved_url: str) -> dict[str, object]:
        return {
            "source_family": acquisition.ROOT_FAMILY,
            "applicable_period": period,
            "http_status": 200,
            "byte_length": len(payload),
            "payload_sha256": sha256(payload).hexdigest(),
            "resolved_url": resolved_url,
        }

    def _read(self, period: str, expected_payload: bytes | None = None, expected_url: str | None = None) -> acquisition.PersistedObject | None:
        directory = self._directory(period)
        try:
            if set(item.name for item in directory.iterdir()) != {_PAYLOAD, _METADATA}:
                return None
            payload = (directory / _PAYLOAD).read_bytes()
            metadata = json.loads((directory / _METADATA).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if type(payload) is not bytes or type(metadata) is not dict or set(metadata) != {"source_family", "applicable_period", "http_status", "byte_length", "payload_sha256", "resolved_url"}:
            return None
        if type(metadata["resolved_url"]) is not str or type(metadata["source_family"]) is not str or type(metadata["applicable_period"]) is not str or type(metadata["http_status"]) is not int or type(metadata["byte_length"]) is not int or type(metadata["payload_sha256"]) is not str:
            return None
        try:
            validate_jpx_url(metadata["resolved_url"])
        except Exception:
            return None
        if metadata["source_family"] != acquisition.ROOT_FAMILY or metadata["applicable_period"] != period or metadata["http_status"] != 200 or metadata["byte_length"] != len(payload) or metadata["payload_sha256"] != sha256(payload).hexdigest():
            return None
        if expected_payload is not None and payload != expected_payload:
            return None
        if expected_url is not None and metadata["resolved_url"] != expected_url:
            return None
        return acquisition.PersistedObject(acquisition.VerifiedLock(**metadata), payload)

    def persist(self, family: str, period: str, payload: bytes, resolved_url: str, _attempt: int) -> acquisition.PersistedObject | None:
        if family != acquisition.ROOT_FAMILY or period not in _PERIODS or type(payload) is not bytes or type(resolved_url) is not str:
            return None
        try:
            validate_jpx_url(resolved_url)
            directory = self._directory(period)
            directory.mkdir(parents=True, exist_ok=True)
            if set(item.name for item in directory.iterdir()) - {_PAYLOAD, _METADATA}:
                return None
        except OSError:
            return None
        payload_path, metadata_path = directory / _PAYLOAD, directory / _METADATA
        if payload_path.exists():
            if not _verify_existing_durable(payload_path, payload):
                return None
        elif not _write_exclusive(payload_path, payload):
            return None
        metadata = self._metadata(period, payload, resolved_url)
        metadata_bytes = _canonical(metadata)
        if metadata_path.exists():
            if not _verify_existing_durable(metadata_path, metadata_bytes):
                return None
        elif not _write_exclusive(metadata_path, metadata_bytes):
            return None
        return self._read(period, payload, resolved_url)

    def verify_final(self, root: acquisition.VerifiedLock, terminal: acquisition.VerifiedLock) -> bool:
        try:
            raw = self.root / _RAW
            if set(item.name for item in raw.iterdir()) != _PERIODS:
                return False
        except OSError:
            return False
        persisted_root = self._read(acquisition.ROOT_PERIOD)
        persisted_terminal = self._read(acquisition.TERMINAL_PERIOD)
        if persisted_root is None or persisted_terminal is None or persisted_root.lock != root or persisted_terminal.lock != terminal:
            return False
        return acquisition.raw_lock_set_sha256(persisted_root.lock, persisted_terminal.lock) == acquisition.raw_lock_set_sha256(root, terminal)


def run_durable_acquisition(implementation_git_sha: str, state_root: Path, root_url: str, root_fetch: Callable[[str, int], acquisition.FetchOutcome], terminal_fetch: Callable[[str, int], acquisition.FetchOutcome], delay: Callable[[int], None] = lambda _seconds: None, locator_runner: Callable[[bytes, str, str, int], tuple[dict[str, object], str | None]] = locator.run_fresh_root_locator) -> dict[str, object]:
    """Start once, then run the pure core through exclusive private locks."""
    if not acquisition._hex(implementation_git_sha, 40):
        raise ValueError("implementation_git_sha")
    state = DurableState(Path(state_root))
    if not state.start(implementation_git_sha):
        return _conflict(implementation_git_sha)
    return acquisition.run_pure_acquisition(implementation_git_sha, root_url, root_fetch, terminal_fetch, state.persist, delay, locator_runner, state.verify_final)
