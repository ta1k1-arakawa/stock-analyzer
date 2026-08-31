from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import v9_006_f1_semantic_successor_public_acquisition as acq
from src import v9_006_f1_semantic_successor_public_acquisition_runtime as runtime
from src import v9_006_f1_semantic_successor_locator as locator
from src.v9_005_stage_a_jpx_probe import LISTED_ISSUES_PAGE_URL

SHA = "a" * 40
ROOT_URL = LISTED_ISSUES_PAGE_URL
TERM_URL = ROOT_URL.rsplit("/", 1)[0] + "/a.xls"
ROOT_BYTES = b"List of TSE-listed Issues as of previous month-end is available.<p>List of TSE-listed Issues (Jan. 2026)</p><a href='a.xls'>x</a>"
TERMINAL_BYTES = b"terminal"


def outcome(status, payload=None, complete=False, url=ROOT_URL):
    return acq.FetchOutcome(status, payload, complete, url)


def locator_success(payload, url, digest, length):
    return locator.run_fresh_root_locator(payload, url, digest, length)


def run_success(state_root: Path, root_fetch=None, terminal_fetch=None, locator_runner=locator_success):
    return runtime.run_durable_acquisition(
        SHA,
        state_root,
        ROOT_URL,
        root_fetch or (lambda *_: outcome(200, ROOT_BYTES, True, ROOT_URL)),
        terminal_fetch or (lambda *_: outcome(200, TERMINAL_BYTES, True, TERM_URL)),
        locator_runner=locator_runner,
    )


def test_clean_receipt_is_durable_and_precedes_first_fetch(tmp_path):
    state_root, observed = tmp_path / "state", []
    def root_fetch(*_):
        observed.append((state_root / runtime._RECEIPT).exists())
        return outcome(200, ROOT_BYTES, True, ROOT_URL)
    result = run_success(state_root, root_fetch=root_fetch)
    assert result["result"] == "SUCCESS" and observed == [True]
    assert json.loads((state_root / runtime._RECEIPT).read_text()) == {"task": acq.TASK, "design_git_sha": acq.DESIGN_GIT_SHA, "implementation_git_sha": SHA, "operation_class": acq.OPERATION_CLASS, "execution_started": True}


@pytest.mark.parametrize("prepare", [lambda root: root.mkdir() or (root / runtime._RECEIPT).write_text("{}"), lambda root: root.mkdir() or (root / "unexpected").write_text("x")])
def test_existing_or_unexpected_durable_state_stops_before_fetch(tmp_path, prepare):
    state_root, calls = tmp_path / "state", []
    prepare(state_root)
    result = run_success(state_root, root_fetch=lambda *_: calls.append(True), terminal_fetch=lambda *_: calls.append(True))
    assert (result["result"], result["failure_stage"]) == ("GOVERNANCE_FAILURE", "EXECUTION_BINDING_CONFLICT") and calls == []


def test_second_invocation_and_receipt_creation_failure_stop_before_fetch(tmp_path, monkeypatch):
    state_root, calls = tmp_path / "state", []
    first = run_success(state_root)
    second = run_success(state_root, root_fetch=lambda *_: calls.append(True), terminal_fetch=lambda *_: calls.append(True))
    assert first["result"] == "SUCCESS" and second["failure_stage"] == "EXECUTION_BINDING_CONFLICT" and calls == []
    monkeypatch.setattr(runtime, "_write_exclusive", lambda *_: False)
    failed = run_success(tmp_path / "failed", root_fetch=lambda *_: calls.append(True), terminal_fetch=lambda *_: calls.append(True))
    assert failed["failure_stage"] == "EXECUTION_BINDING_CONFLICT" and calls == []


@pytest.mark.parametrize("period, payload, url", [(acq.ROOT_PERIOD, ROOT_BYTES, ROOT_URL), (acq.TERMINAL_PERIOD, TERMINAL_BYTES, TERM_URL)])
def test_exclusive_lock_writes_and_returns_durable_reread(tmp_path, period, payload, url):
    state = runtime.DurableState(tmp_path / "state")
    assert state.start(SHA)
    persisted = state.persist(acq.ROOT_FAMILY, period, payload, url, 1)
    directory = state._directory(period)
    assert persisted is not None and type(persisted.payload) is bytes and persisted.payload == payload
    assert set(item.name for item in directory.iterdir()) == {runtime._PAYLOAD, runtime._METADATA}
    assert persisted.lock.payload_sha256 == acq.sha256(payload).hexdigest() and persisted.lock.resolved_url == url


def test_exact_partial_lock_can_finish_but_mismatched_partial_lock_is_preserved(tmp_path):
    state = runtime.DurableState(tmp_path / "state")
    assert state.start(SHA)
    directory = state._directory(acq.ROOT_PERIOD); directory.mkdir(parents=True)
    payload_path = directory / runtime._PAYLOAD
    payload_path.write_bytes(ROOT_BYTES)
    assert state.persist(acq.ROOT_FAMILY, acq.ROOT_PERIOD, ROOT_BYTES, ROOT_URL, 1) is not None
    assert payload_path.read_bytes() == ROOT_BYTES and (directory / runtime._METADATA).exists()
    other = runtime.DurableState(tmp_path / "other")
    assert other.start(SHA)
    bad_directory = other._directory(acq.ROOT_PERIOD); bad_directory.mkdir(parents=True)
    bad_payload = bad_directory / runtime._PAYLOAD; bad_payload.write_bytes(b"mismatch")
    assert other.persist(acq.ROOT_FAMILY, acq.ROOT_PERIOD, ROOT_BYTES, ROOT_URL, 1) is None
    assert bad_payload.read_bytes() == b"mismatch" and not (bad_directory / runtime._METADATA).exists()


def test_runtime_persistence_retries_receive_the_same_authoritative_bytes(tmp_path, monkeypatch):
    original_write, identities, failures = runtime._write_exclusive, [], [True]
    def fail_first_metadata(path, content):
        if path.name == runtime._METADATA and failures and failures.pop():
            return False
        return original_write(path, content)
    original_persist = runtime.DurableState.persist
    def recording_persist(self, family, period, payload, url, attempt):
        if period == acq.ROOT_PERIOD:
            identities.append(id(payload))
        return original_persist(self, family, period, payload, url, attempt)
    monkeypatch.setattr(runtime, "_write_exclusive", fail_first_metadata)
    monkeypatch.setattr(runtime.DurableState, "persist", recording_persist)
    assert run_success(tmp_path / "state")["result"] == "SUCCESS"
    assert identities == [id(ROOT_BYTES), id(ROOT_BYTES)]


def test_locator_receives_exact_durable_reread_and_corrupt_reread_skips_it(tmp_path, monkeypatch):
    supplied, located = [], []
    original_persist = runtime.DurableState.persist
    def reread_copy(self, *args):
        persisted = original_persist(self, *args)
        if persisted is not None and args[1] == acq.ROOT_PERIOD:
            return acq.PersistedObject(persisted.lock, bytes(bytearray(persisted.payload)))
        return persisted
    monkeypatch.setattr(runtime.DurableState, "persist", reread_copy)
    def root_fetch(*_):
        supplied.append(ROOT_BYTES)
        return outcome(200, ROOT_BYTES, True, ROOT_URL)
    def recording_locator(payload, *args):
        located.append(payload)
        return locator_success(payload, *args)
    assert run_success(tmp_path / "good", root_fetch=root_fetch, locator_runner=recording_locator)["result"] == "SUCCESS"
    assert located[0] == ROOT_BYTES and located[0] is not supplied[0]
    monkeypatch.setattr(runtime.DurableState, "persist", original_persist)
    monkeypatch.setattr(runtime.DurableState, "_read", lambda *_args, **_kwargs: None)
    result = run_success(tmp_path / "corrupt", locator_runner=lambda *_: located.append(True))
    assert result["failure_stage"] == "ROOT_PERSISTENCE_EXHAUSTED" and located == [ROOT_BYTES]


@pytest.mark.parametrize("corrupt", ["extra", "metadata"])
def test_final_provenance_failure_preserves_two_locks_and_never_refetches(tmp_path, monkeypatch, corrupt):
    original_verify = runtime.DurableState.verify_final
    def corrupted_verify(self, root, terminal):
        raw = self.root / runtime._RAW
        if corrupt == "extra":
            (raw / "extra").mkdir()
        else:
            (self._directory(acq.TERMINAL_PERIOD) / runtime._METADATA).write_text("{}")
        return original_verify(self, root, terminal)
    monkeypatch.setattr(runtime.DurableState, "verify_final", corrupted_verify)
    root_calls, terminal_calls = [], []
    result = run_success(tmp_path / corrupt, root_fetch=lambda *_: root_calls.append(True) or outcome(200, ROOT_BYTES, True, ROOT_URL), terminal_fetch=lambda *_: terminal_calls.append(True) or outcome(200, TERMINAL_BYTES, True, TERM_URL))
    assert (result["result"], result["failure_stage"], result["raw_lock_count"], result["safe_provenance_verified"]) == ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_POST_TERMINAL_PRE_PROVENANCE", 2, False)
    assert root_calls == [True] and terminal_calls == [True]
    rendered = acq.canonical_json(result)
    assert ROOT_URL not in rendered and TERM_URL not in rendered and str(tmp_path) not in rendered
