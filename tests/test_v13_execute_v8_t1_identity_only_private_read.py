from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from scripts import v13_execute_v8_t1_identity_only_private_read as runner
from scripts import v13_resolve_v8_t1_identity_state as resolver

_REPOSITORY = Path(__file__).resolve().parents[1]


def _codes(count: int = 300) -> list[str]:
    return [f"{value:04X}" for value in range(count)]


def _synthetic_manifest(members: list[str] | None = None) -> tuple[bytes, resolver._ExpectedBindings]:
    t1 = members or _codes()
    t1_hash = hashlib.sha256(("\n".join(t1) + "\n").encode()).hexdigest()
    manifest_hash = hashlib.sha256(b"synthetic stated manifest binding").hexdigest()
    codes = _codes(300)
    obj = {
        "schema_version": "V8_PARTITION_MANIFEST_V3", "study_name": "synthetic-study",
        "design_commit": "1" * 40, "source_snapshot_semantics": "SYNTHETIC",
        "source_snapshot_clarification_commit": "2" * 40,
        "partition_implementation_git_commit": "3" * 40, "created_utc": "2026-01-01T00:00:00Z",
        "source_url": "https://synthetic.invalid/", "source_host": "synthetic.invalid",
        "source_acquisition_utc": "2026-01-01T00:00:00Z", "source_raw_sha256": "4" * 64,
        "source_raw_byte_count": 1, "v4_source_raw_sha256_reference": "5" * 64,
        "v4_raw_sha_equality_required": False, "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS", "eligible_ticker_count": 2000,
        "eligible_ticker_list_sha256": "6" * 64, "selection_rule": "synthetic",
        "deterministic_ordering_rule": "synthetic", "t0_ticker_list_sha256": "7" * 64,
        "t1_ticker_list_sha256": t1_hash, "t2_ticker_list_sha256": "8" * 64,
        "t3_ticker_list_sha256": "9" * 64, "t_spare_ticker_list_sha256": "a" * 64,
        "legacy_exclude_list": ["1570"], "legacy_exclude_list_sha256": "b" * 64,
        "block_sizes": {"T0": 300, "T1": len(t1), "T2": 300, "T3": 300, "T_spare": 1},
        "block_assignments": {"T0": codes, "T1": t1, "T2": ["T2XX"] + codes[1:],
                              "T3": ["T3XX"] + codes[1:], "T_spare": ["SPAR"]},
        "p_hist_start": "2016-04-01", "p_hist_end": "2025-12-31", "t1_role": "VALIDATION",
        "t2_role": "SEALED_HOLDOUT", "t3_role": "SEALED_RESERVE",
        "t3_price_acquisition_authorized": False, "manifest_sha256": manifest_hash,
    }
    bindings = resolver._ExpectedBindings(manifest_hash, t1_hash, 300)
    return json.dumps(obj, separators=(",", ":")).encode(), bindings


class _ObservedStream:
    def __init__(self, raw: bytes, events: list[str]):
        self._stream = __import__("io").BytesIO(raw)
        self.events = events

    def __enter__(self):
        self.events.append("open")
        return self

    def __exit__(self, *_args):
        self.events.append("close")

    def read(self, size: int = -1):
        chunk = self._stream.read(size)
        self.events.append("read1" if size == 1 else "read_rest")
        return chunk


@pytest.fixture
def synthetic_paths(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copyfile(_REPOSITORY / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json",
                    repo / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json")
    shutil.copyfile(_REPOSITORY / "V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md",
                    repo / "V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md")
    outside = tmp_path / "outside"
    outside.mkdir()
    source = outside / "synthetic.json"
    raw, bindings = _synthetic_manifest()
    source.write_bytes(raw)
    return repo, outside, source, bindings


def _resolve(source: Path, output: Path, repo: Path, bindings, **kwargs):
    return resolver._resolve_identity_state(source, output, repo, bindings=bindings, **kwargs)


def test_valid_one_open_first_byte_receipt_then_same_stream_remainder_and_state(synthetic_paths, monkeypatch):
    repo, outside, source, bindings = synthetic_paths
    raw = source.read_bytes()
    events: list[str] = []
    receipt = outside / "consumed.json"

    if __import__("os").name == "nt":
        original_publish = resolver._publish_windows_write_through
        original_flush = resolver._flush_windows_file

        def observed_durable_publish(staging_path, destination):
            events.append("publish_begin")
            original_publish(staging_path, destination)
            events.append("publish_done")

        def observed_flush(destination):
            events.append("durability_begin")
            original_flush(destination)
            events.append("durability_done")

        monkeypatch.setattr(resolver, "_publish_windows_write_through", observed_durable_publish)
        monkeypatch.setattr(resolver, "_flush_windows_file", observed_flush)
    else:
        original_link = resolver.os.link
        original_flush = resolver._flush_directory

        def observed_link(staging_path, destination):
            original_link(staging_path, destination)
            events.append("publish")

        def observed_flush(directory):
            events.append("durability_begin")
            original_flush(directory)
            events.append("durability_done")

        monkeypatch.setattr(resolver.os, "link", observed_link)
        monkeypatch.setattr(resolver, "_flush_directory", observed_flush)

    def opened(path: Path):
        assert path == source
        return _ObservedStream(raw, events)

    def boundary():
        assert events == ["open", "read1"]
        runner._write_once(receipt, runner._receipt_bytes())
        events.append("receipt")

    state = _resolve(source, outside / "state.json", repo, bindings,
                     on_first_byte=boundary, source_opener=opened)
    assert events.index("read1") < events.index("publish_begin" if __import__("os").name == "nt" else "publish")
    if __import__("os").name == "nt":
        assert events.index("publish_done") < events.index("durability_begin")
    assert events.index("durability_done") < events.index("receipt")
    assert events.index("receipt") < events.index("read_rest")
    assert events.count("open") == 1
    assert json.loads(receipt.read_text(encoding="ascii"))["authorization_consumed"] is True
    assert state["t1_count"] == 300
    assert (outside / "state.json").exists()


def test_durability_failure_after_publication_stops_before_remainder_and_is_post_boundary(
    synthetic_paths, monkeypatch
):
    repo, outside, source, bindings = synthetic_paths
    raw = source.read_bytes()
    events: list[str] = []
    receipt = outside / "consumed.json"

    if __import__("os").name == "nt":
        original_publish = resolver._publish_windows_write_through
        def observed_publish(staging_path, destination):
            original_publish(staging_path, destination)
            events.append("published")
        monkeypatch.setattr(resolver, "_publish_windows_write_through", observed_publish)
        monkeypatch.setattr(resolver, "_flush_windows_file", lambda _path: (_ for _ in ()).throw(
            OSError("synthetic durability failure")
        ))
    else:
        original_link = resolver.os.link
        def observed_link(staging_path, destination):
            original_link(staging_path, destination)
            events.append("published")
        monkeypatch.setattr(resolver.os, "link", observed_link)
        monkeypatch.setattr(resolver, "_flush_directory", lambda _path: (_ for _ in ()).throw(
            OSError("synthetic durability failure")
        ))

    def opener(_path):
        return _ObservedStream(raw, events)

    report = runner.execute(
        source, outside / "state.json", receipt, repo,
        repo / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json",
        resolver=lambda source_arg, output_arg, repo_arg, *, on_first_byte, on_state_written: resolver._resolve_identity_state(
            source_arg, output_arg, repo_arg, bindings=bindings, on_first_byte=on_first_byte,
            on_state_written=on_state_written, source_opener=opener,
        ),
    )
    assert "PRIVATE_BOUNDARY_CROSSED=true" in report
    assert "AUTHORIZATION_REUSABLE=false" in report
    assert "SECOND_EXECUTION_ALLOWED=false" in report
    assert "POST_BOUNDARY_RECEIPT_PUBLISH_FAILED" in report
    assert events.index("published") < events.index("close")
    assert "read_rest" not in events


def test_runner_valid_synthetic_execution_emits_only_safe_report(synthetic_paths, monkeypatch):
    repo, outside, source, bindings = synthetic_paths
    raw = source.read_bytes()
    events: list[str] = []
    state_path = outside / "state.json"
    receipt_path = outside / "receipt.json"
    auth_path = repo / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json"
    original_write_once = runner._write_once

    def observed_write_once(destination, payload):
        if destination == receipt_path:
            events.append("receipt")
        return original_write_once(destination, payload)

    monkeypatch.setattr(runner, "_write_once", observed_write_once)

    def opener(_path):
        return _ObservedStream(raw, events)

    def synthetic_resolver(source_arg, output_arg, repo_arg, *, on_first_byte, on_state_written):
        return resolver._resolve_identity_state(
            source_arg, output_arg, repo_arg, bindings=bindings,
            on_first_byte=on_first_byte, on_state_written=on_state_written, source_opener=opener,
        )

    report = runner.execute(
        source, state_path, receipt_path, repo, auth_path, resolver=synthetic_resolver,
    )
    assert "EXECUTION_RESULT=PASS" in report
    assert "PRIVATE_READS=1" in report
    assert "PRIVATE_STATE_WRITTEN=true" in report
    assert "CONSUMED_RECEIPT_WRITTEN=true" in report
    assert events.count("open") == 1
    assert events.index("read1") < events.index("receipt") < events.index("read_rest")
    assert "receipt.json" not in report and str(source) not in report
    assert json.loads(receipt_path.read_text(encoding="ascii"))["authorization_consumed"] is True
    assert json.loads(state_path.read_text(encoding="utf-8"))["t1_count"] == 300


@pytest.mark.parametrize("raw", [b"", None])
def test_empty_or_unreadable_source_does_not_call_boundary(tmp_path: Path, raw):
    repo = tmp_path / "repo"
    repo.mkdir()
    called: list[bool] = []
    source = tmp_path / "missing.json"
    if raw is not None:
        source.write_bytes(raw)
    with pytest.raises(resolver.IdentityResolutionBlocked):
        _resolve(source, tmp_path / "state.json", repo, _synthetic_manifest()[1],
                 on_first_byte=lambda: called.append(True))
    assert called == []


def test_callback_called_exactly_once_after_one_nonempty_byte(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    raw, bindings = _synthetic_manifest()
    source = tmp_path / "synthetic.json"
    source.write_bytes(raw)
    events: list[str] = []
    def callback():
        assert events == ["open", "read1"]
        events.append("callback")
    def opener(_path):
        return _ObservedStream(raw, events)
    _resolve(source, tmp_path / "state.json", repo, bindings, on_first_byte=callback, source_opener=opener)
    assert events.count("callback") == 1


def test_callback_failure_is_post_boundary_and_does_not_read_remainder_or_retry(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    raw, bindings = _synthetic_manifest()
    source = tmp_path / "synthetic.json"
    source.write_bytes(raw)
    events: list[str] = []
    def callback():
        events.append("callback")
        raise OSError("private path and sentinel must not leak")
    def opener(_path):
        return _ObservedStream(raw, events)
    with pytest.raises(resolver.IdentityResolutionBlocked, match="POST_BOUNDARY_RECEIPT_PUBLISH_FAILED"):
        _resolve(source, tmp_path / "state.json", repo, bindings, on_first_byte=callback, source_opener=opener)
    assert events == ["open", "read1", "callback", "close"]


def test_runner_reports_receipt_publish_failure_as_crossed_without_retry(synthetic_paths, monkeypatch):
    repo, outside, source, _ = synthetic_paths
    monkeypatch.setattr(runner, "_write_once", lambda *_args: (_ for _ in ()).throw(OSError("private")))
    report = runner.execute(
        source, outside / "state.json", outside / "receipt.json", repo,
        repo / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json",
    )
    assert "PRE_GATE_STATUS=PASS" in report
    assert "PRIVATE_BOUNDARY_CROSSED=true" in report
    assert "GATE_CONSUMED=true" in report
    assert "PRIVATE_READS=1" in report
    assert "CONSUMED_RECEIPT_WRITTEN=false" in report
    assert "FAILURE_CLASS=POST_BOUNDARY_RECEIPT_PUBLISH_FAILED" in report
    assert "AUTHORIZATION_REUSABLE=false" in report
    assert "SECOND_EXECUTION_ALLOWED=false" in report


@pytest.mark.parametrize("failure", ["malformed", "hash", "count", "state_write"])
def test_post_boundary_failure_keeps_receipt_and_forbids_retry(tmp_path: Path, failure, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    raw, bindings = _synthetic_manifest()
    if failure == "malformed":
        raw = b"{"
    elif failure == "hash":
        bindings = resolver._ExpectedBindings(bindings.manifest_stated_sha256, "f" * 64, 300)
    elif failure == "count":
        bindings = resolver._ExpectedBindings(bindings.manifest_stated_sha256, bindings.t1_ticker_list_sha256, 299)
    source = tmp_path / "synthetic.json"
    source.write_bytes(raw)
    receipt = tmp_path / "consumed.json"
    calls: list[int] = []
    def boundary():
        calls.append(1)
        runner._write_once(receipt, runner._receipt_bytes())
    state_path = tmp_path / "state.json"
    if failure == "state_write":
        monkeypatch.setattr(resolver, "_write_once", lambda *_args: (_ for _ in ()).throw(
            resolver.IdentityResolutionBlocked("OUTPUT_WRITE_FAILED")
        ))
    with pytest.raises(resolver.IdentityResolutionBlocked):
        _resolve(source, state_path, repo, bindings, on_first_byte=boundary)
    assert receipt.exists()
    assert calls == [1]


@pytest.mark.parametrize("target", ["receipt", "state"])
def test_existing_durable_output_blocks_before_source_content_read(synthetic_paths, target):
    repo, outside, source, _ = synthetic_paths
    state = outside / "state.json"
    receipt = outside / "receipt.json"
    (receipt if target == "receipt" else state).write_text("existing", encoding="utf-8")
    resolver_called: list[bool] = []
    def forbidden_resolver(*_args, **_kwargs):
        resolver_called.append(True)
        raise AssertionError("source content path must not be entered")
    report = runner.execute(
        source, state, receipt, repo,
        repo / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json",
        resolver=forbidden_resolver,
    )
    assert resolver_called == []
    assert "PRIVATE_BOUNDARY_CROSSED=false" in report
    assert "PRIVATE_READS=0" in report
    assert "AUTHORIZATION_REUSABLE=false" in report


def test_post_boundary_parse_failure_retains_receipt_and_later_run_stops(synthetic_paths):
    repo, outside, source, _ = synthetic_paths
    source.write_bytes(b"{")
    state = outside / "state.json"
    receipt = outside / "receipt.json"
    auth = repo / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json"
    report = runner.execute(source, state, receipt, repo, auth)
    assert "PRE_GATE_STATUS=PASS" in report
    assert "PRIVATE_BOUNDARY_CROSSED=true" in report
    assert "CONSUMED_RECEIPT_WRITTEN=true" in report
    assert "PRIVATE_STATE_WRITTEN=false" in report
    assert receipt.exists()
    resolver_called: list[bool] = []
    def forbidden_resolver(*_args, **_kwargs):
        resolver_called.append(True)
        raise AssertionError("existing receipt must stop before source content")
    retry_report = runner.execute(source, state, receipt, repo, auth, resolver=forbidden_resolver)
    assert resolver_called == []
    assert "PRE_GATE_STATUS=FAIL" in retry_report
    assert "PRIVATE_READS=0" in retry_report
    assert "AUTHORIZATION_REUSABLE=false" in retry_report
    assert "SECOND_EXECUTION_ALLOWED=false" in retry_report


def test_preflight_rejects_relative_inside_collision_and_reuse_paths(synthetic_paths):
    repo, outside, source, _ = synthetic_paths
    auth = repo / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json"
    cases = [
        ("relative-state", "state.json", outside / "receipt.json"),
        ("relative-receipt", outside / "state.json", "receipt.json"),
        ("inside-state", repo / "state.json", outside / "receipt.json"),
        ("inside-receipt", outside / "state.json", repo / "receipt.json"),
        ("collision", outside / "same.json", outside / "same.json"),
    ]
    for _label, state, receipt in cases:
        report = runner.execute(source, state, receipt, repo, auth)
        assert "PRIVATE_READS=0" in report
        assert "PRIVATE_BOUNDARY_CROSSED=false" in report
    (outside / "reused.json").write_text("old state", encoding="utf-8")
    report = runner.execute(source, outside / "reused.json", outside / "new-receipt.json", repo, auth)
    assert "FAILURE_CLASS=OUTPUT_ALREADY_EXISTS" in report


def test_receipt_contains_only_safe_allowlisted_fields(synthetic_paths):
    _, outside, _source, _bindings = synthetic_paths
    sentinel = "NO_PRIVATE_PATH_OR_IDENTITY_SENTINEL"
    data = runner._receipt_bytes().decode("ascii")
    receipt = json.loads(data)
    assert set(receipt) == {
        "schema", "study", "operation_class", "boundary", "authorization_reviewed_sha",
        "resolver_reviewed_sha", "expected_partition_manifest_stated_sha256",
        "expected_t1_ticker_list_sha256", "expected_t1_count", "authorization_consumed",
    }
    assert sentinel not in data
    assert str(outside) not in data


def test_t2_t3_spare_sentinel_does_not_leak_from_selective_resolver(tmp_path: Path):
    raw, bindings = _synthetic_manifest()
    sentinel = "T2_T3_SPARE_SENTINEL"
    raw = raw.replace(b"T2XX", sentinel.encode()).replace(b"T3XX", sentinel.encode()).replace(b"SPAR", sentinel.encode())
    source = tmp_path / "synthetic.json"
    source.write_bytes(raw)
    state_path = tmp_path / "state.json"
    repo = tmp_path / "repo"
    repo.mkdir()
    state = _resolve(source, state_path, repo, bindings)
    assert sentinel not in repr(state)
    assert sentinel not in state_path.read_text(encoding="utf-8")


def test_runner_static_contract_no_network_or_full_manifest_parser():
    path = _REPOSITORY / "scripts" / "v13_execute_v8_t1_identity_only_private_read.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    forbidden = {"http", "httpx", "requests", "socket", "subprocess", "urllib", "urllib3"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
    assert "read_partition_manifest" not in text
    assert text.count("json.loads(") == 1
    assert "json.loads(authorization_bytes.decode" in text  # Authorization only; source goes to selective resolver.
    assert "source, state, repository, on_first_byte=consume" in text
    assert "stream.read()" not in text


def test_resolver_and_runner_bindings_match_issue_42():
    assert resolver.EXPECTED_MANIFEST_STATED_SHA256 == "0a8632804eb1b629ca2d5f3c3b679e3f9b1094b668a7f44b00b35acc2b70ca62"
    assert resolver.EXPECTED_T1_TICKER_LIST_SHA256 == "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d"
    assert resolver.EXPECTED_T1_COUNT == 300
    assert runner.AUTHORIZATION_REVIEWED_SHA == "18a99fbb3740ecb827abc14513fad1402f7632ec"
    assert runner.RESOLVER_REVIEWED_SHA == "63fa6b7541694565eb171a98d11df0092ab20034"
    assert runner.DESIGN_BLOB_SHA1 == "3bfcd695c69f6dac480f8fc99ca4f3916f668e4a"
    assert runner.AUTHORIZATION_BLOB_SHA1 == "280baea899a576cbc3b705db838e5988dbed5027"


def test_safe_report_never_contains_any_supplied_paths(synthetic_paths):
    repo, outside, source, _ = synthetic_paths
    missing = outside / "does-not-exist-private-path.json"
    state = outside / "state-private-path.json"
    receipt = outside / "receipt-private-path.json"
    report = runner.execute(
        missing, state, receipt, repo,
        repo / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json",
    )
    assert "PRE_GATE_STATUS=FAIL" in report
    assert "PRIVATE_READS=0" in report
    for path in (str(missing), str(state), str(receipt), str(source), str(repo)):
        assert path not in report
