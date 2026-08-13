from __future__ import annotations

import tempfile
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8_partition as partition
from src import v8b_allocation as allocation
from src import v8b_human_gate_consumption as gate_consumption
from src import v8b_t1b_allocator as allocator

SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_REVIEWED_COMMIT = "b" * 40


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """The allocator must never open a network connection of any kind."""

    def forbidden(*args, **kwargs):
        raise AssertionError("real network call executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def clock_stub():
    return datetime(2026, 8, 12, tzinfo=timezone.utc)


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


def write_partition_manifest(path: Path, *, t_spare: list[str]) -> dict:
    blocks = {
        "T0": _tickers("T0BLK", 300), "T1": _tickers("T1BLK", 300),
        "T2": _tickers("T2BLK", 300), "T3": _tickers("T3BLK", 300),
        "T_spare": list(t_spare),
    }
    manifest = {
        "schema_version": partition.SCHEMA_VERSION,
        "study_name": partition.STUDY_NAME,
        "design_commit": partition.DESIGN_COMMIT,
        "source_snapshot_semantics": partition.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": partition.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": "6" * 40,
        "created_utc": "2026-08-09T00:00:00Z",
        "source_url": "https://www.jpx.co.jp/synthetic/data_j.xls",
        "source_host": "www.jpx.co.jp",
        "source_acquisition_utc": "2026-08-09T00:00:00Z",
        "source_raw_sha256": "0" * 64,
        "source_raw_byte_count": 0,
        "v4_source_raw_sha256_reference": "1" * 64,
        "v4_raw_sha_equality_required": partition.V4_RAW_SHA_EQUALITY_REQUIRED,
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "eligible_ticker_count": 1500,
        "eligible_ticker_list_sha256": partition.ticker_list_sha256(sum((blocks[k] for k in blocks), [])),
        "selection_rule": "synthetic fixture selection rule",
        "deterministic_ordering_rule": partition.DETERMINISTIC_ORDERING_RULE,
        "t0_ticker_list_sha256": partition.ticker_list_sha256(blocks["T0"]),
        "t1_ticker_list_sha256": partition.ticker_list_sha256(blocks["T1"]),
        "t2_ticker_list_sha256": partition.ticker_list_sha256(blocks["T2"]),
        "t3_ticker_list_sha256": partition.ticker_list_sha256(blocks["T3"]),
        "t_spare_ticker_list_sha256": partition.ticker_list_sha256(blocks["T_spare"]),
        "legacy_exclude_list": [],
        "legacy_exclude_list_sha256": partition.ticker_list_sha256([]),
        "block_sizes": {k: len(v) for k, v in blocks.items()},
        "block_assignments": blocks,
        "p_hist_start": partition.P_HIST_START,
        "p_hist_end": partition.P_HIST_END,
        "t1_role": partition.T1_ROLE,
        "t2_role": partition.T2_ROLE,
        "t3_role": partition.T3_ROLE,
        "t3_price_acquisition_authorized": False,
    }
    manifest["manifest_sha256"] = partition.canonical_sha256(manifest)
    assert set(manifest) == set(partition.MANIFEST_FIELDS)
    path.write_bytes(partition.canonical_json_bytes(manifest))
    return manifest


def run(**overrides):
    # Each call defaults to its own fresh, isolated consumption-state
    # directory so existing tests (which exercise unrelated failure paths)
    # remain independent of one another; tests specifically exercising
    # HIGH-1's durable one-shot consumption pass an explicit, shared
    # ``consumption_state_root`` across two calls instead.
    overrides.setdefault(
        "consumption_state_root", Path(tempfile.gettempdir()) / ("v8b_gate_state-" + uuid.uuid4().hex)
    )
    return allocator._allocate_t1b_production_with_dependencies(**overrides)


# ---------------------------------------------------------------------------
# Confirmation token (HIGH-7)
# ---------------------------------------------------------------------------


def test_confirmation_token_is_frozen_literal():
    assert allocator.ALLOCATION_CONFIRMATION == "V8B_PRODUCTION_ALLOCATE_T1B"


def test_wrong_confirmation_blocks_before_any_dependency_call(tmp_path):
    def forbidden(*_a, **_kw):
        raise AssertionError("must not be called")

    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation="wrong",
            partition_manifest_path=tmp_path / "missing.json",
            output_path=tmp_path / "out.json",
            git_commit_resolver=forbidden,
            design_freeze_approval_reader=forbidden,
            frozen_design_object_verifier=forbidden,
            reviewed_implementation_binder=forbidden,
            anchor_reader=forbidden,
            clock=clock_stub,
        )
    assert excinfo.value.reason == "V8B_ALLOCATION_CONFIRMATION_INVALID"
    assert excinfo.value.authorization_consumed is False


def test_public_entrypoint_rejects_wrong_confirmation():
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        allocator.allocate_t1b_production(
            confirmation="not the real token", partition_manifest_path="/tmp/x", output_path="/tmp/y"
        )
    assert excinfo.value.reason == "V8B_ALLOCATION_CONFIRMATION_INVALID"
    assert excinfo.value.authorization_consumed is False


# ---------------------------------------------------------------------------
# Provenance/freeze/review binding ordering, all before private data access
# ---------------------------------------------------------------------------


def test_git_provenance_failure_blocks_before_private_manifest_read(tmp_path):
    def dirty_resolver():
        raise allocator.V8BGitProvenanceBlocked("PRODUCTION_GIT_WORKTREE_DIRTY")

    def forbidden_manifest_path(*_a, **_kw):
        raise AssertionError("private manifest must not be read")

    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=forbidden_manifest_path,
            output_path=tmp_path / "out.json",
            git_commit_resolver=dirty_resolver,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: {},
            clock=clock_stub,
        )
    assert excinfo.value.reason == "PRODUCTION_GIT_WORKTREE_DIRTY"
    assert excinfo.value.authorization_consumed is False


def test_freeze_approval_failure_blocks_before_private_manifest_read(tmp_path):
    def failing_reader(head):
        raise allocator.V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED")

    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=tmp_path / "unread.json",
            output_path=tmp_path / "out.json",
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=failing_reader,
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: {},
            clock=clock_stub,
        )
    assert excinfo.value.reason == "V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED"
    assert excinfo.value.authorization_consumed is False
    assert not (tmp_path / "unread.json").exists()  # never even written by test setup -- proves no read attempted crashes differently


def test_reviewed_implementation_binder_failure_blocks(tmp_path):
    def failing_binder(head):
        raise allocator.V8BProductionProvenanceBlocked("V8B_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:src/v8b_t1b_allocator.py")

    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=tmp_path / "x.json",
            output_path=tmp_path / "out.json",
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=failing_binder,
            anchor_reader=lambda head: {},
            clock=clock_stub,
        )
    assert excinfo.value.reason == "V8B_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:src/v8b_t1b_allocator.py"
    assert excinfo.value.authorization_consumed is False


def test_anchor_not_authorized_blocks_before_private_manifest_read(tmp_path):
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=tmp_path / "unread.json",
            output_path=tmp_path / "out.json",
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: {"authorization_status": "NOT_AUTHORIZED"},
            clock=clock_stub,
        )
    assert excinfo.value.reason == "TRUSTED_PARTITION_NOT_AUTHORIZED"
    assert excinfo.value.authorization_consumed is False
    assert not (tmp_path / "unread.json").exists()


# ---------------------------------------------------------------------------
# Exact parent T_spare count/hash pin
# ---------------------------------------------------------------------------


def _valid_anchor_for(manifest: dict) -> dict:
    return {
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": manifest["manifest_sha256"],
        "authorized_partition_implementation_git_commit": manifest["partition_implementation_git_commit"],
    }


def test_wrong_study_name_on_otherwise_matching_manifest_blocks(tmp_path, monkeypatch):
    manifest_path = tmp_path / "partition.json"
    t_spare = _tickers("TS", 1904)
    manifest = write_partition_manifest(manifest_path, t_spare=t_spare)
    tampered = dict(manifest)
    tampered["study_name"] = "NOT_V8_HISTORICAL_RESEARCH"
    tampered["manifest_sha256"] = partition.canonical_sha256(
        {k: v for k, v in tampered.items() if k != "manifest_sha256"}
    )
    manifest_path.write_bytes(partition.canonical_json_bytes(tampered))
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_COUNT", 1904)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256", partition.ticker_list_sha256(t_spare))
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=manifest_path,
            output_path=tmp_path / "out.json",
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: {
                "authorization_status": "AUTHORIZED",
                "authorized_partition_manifest_sha256": tampered["manifest_sha256"],
                "authorized_partition_implementation_git_commit": tampered["partition_implementation_git_commit"],
            },
            clock=clock_stub,
        )
    assert excinfo.value.reason == "PARTITION_MANIFEST_STUDY_NAME_MISMATCH"
    assert excinfo.value.authorization_consumed is True


def test_parent_t_spare_count_mismatch_blocks(tmp_path):
    manifest_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(manifest_path, t_spare=_tickers("TS", 1903))  # one short of frozen 1904
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=manifest_path,
            output_path=tmp_path / "out.json",
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: _valid_anchor_for(manifest),
            clock=clock_stub,
        )
    assert excinfo.value.reason == "PARENT_T_SPARE_TICKER_COUNT_INVALID"
    assert excinfo.value.authorization_consumed is True


def test_synthetic_parent_never_matches_real_frozen_hash_by_accident(tmp_path):
    """Any synthetic (non-real) 1904-ticker T_spare set, even the right
    size, must still BLOCK -- proves the hash pin is a real content check,
    not merely a length check."""
    manifest_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(manifest_path, t_spare=_tickers("TS", 1904))
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=manifest_path,
            output_path=tmp_path / "out.json",
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: _valid_anchor_for(manifest),
            clock=clock_stub,
        )
    assert excinfo.value.reason == "PARENT_T_SPARE_TICKER_LIST_SHA_MISMATCH"
    assert excinfo.value.authorization_consumed is True


# ---------------------------------------------------------------------------
# Successful synthetic allocation: deterministic slice, atomic write, no retry
# ---------------------------------------------------------------------------


def test_successful_synthetic_allocation_atomic_write_and_public_summary(tmp_path, monkeypatch):
    manifest_path = tmp_path / "partition.json"
    t_spare = _tickers("TS", 1904)
    manifest = write_partition_manifest(manifest_path, t_spare=t_spare)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_COUNT", 1904)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256", partition.ticker_list_sha256(t_spare))
    output_path = tmp_path / "private" / "t1b_allocation_artifact.json"

    result = run(
        confirmation=allocator.ALLOCATION_CONFIRMATION,
        partition_manifest_path=manifest_path,
        output_path=output_path,
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        anchor_reader=lambda head: _valid_anchor_for(manifest),
        clock=clock_stub,
    )

    # Public summary never leaks ticker identities.
    assert "t1b_tickers" not in result
    assert "remaining_t_spare_tickers" not in result
    assert result["t1b_ticker_count"] == 300

    # The written private artifact IS the deterministic §4 zero-offset slice.
    written = allocation.read_t1b_allocation_artifact_bytes(output_path.read_bytes())
    assert written["t1b_tickers"] == t_spare[:300]
    assert written["remaining_t_spare_tickers"] == t_spare[300:]
    assert written["v8b_allocation_implementation_commit"] == SYNTHETIC_REVIEWED_COMMIT


def test_allocation_artifact_never_overwrites_existing_destination(tmp_path, monkeypatch):
    manifest_path = tmp_path / "partition.json"
    t_spare = _tickers("TS", 1904)
    manifest = write_partition_manifest(manifest_path, t_spare=t_spare)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_COUNT", 1904)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256", partition.ticker_list_sha256(t_spare))
    output_path = tmp_path / "private" / "t1b_allocation_artifact.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"pre-existing")

    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=manifest_path,
            output_path=output_path,
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: _valid_anchor_for(manifest),
            clock=clock_stub,
        )
    assert excinfo.value.reason == "V8B_ALLOCATION_ARTIFACT_ALREADY_EXISTS"
    assert excinfo.value.authorization_consumed is True
    assert output_path.read_bytes() == b"pre-existing"


def test_output_path_inside_repository_blocks(tmp_path, monkeypatch):
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    manifest_path = tmp_path / "partition.json"
    t_spare = _tickers("TS", 1904)
    manifest = write_partition_manifest(manifest_path, t_spare=t_spare)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_COUNT", 1904)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256", partition.ticker_list_sha256(t_spare))

    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=manifest_path,
            output_path=root / "should_not_write_here.json",
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: _valid_anchor_for(manifest),
            clock=clock_stub,
        )
    assert excinfo.value.reason == "OUTPUT_PATH_INSIDE_SOURCE_REPOSITORY"
    assert excinfo.value.authorization_consumed is True


# ---------------------------------------------------------------------------
# Round-3 HIGH-1: one-shot authorization_consumed semantics for the allocator
# ---------------------------------------------------------------------------


def test_authorization_consumed_true_at_first_private_manifest_read_failure(tmp_path):
    """A failure reading the private partition manifest itself (the very
    first private action, step (5)) must already report
    authorization_consumed=True -- consumption begins at the attempt, not
    at a later success."""
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=tmp_path / "does_not_exist.json",
            output_path=tmp_path / "out.json",
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: {"authorization_status": "AUTHORIZED"},
            clock=clock_stub,
        )
    assert excinfo.value.authorization_consumed is True


def test_no_automatic_or_manual_retry_after_authorization_consumed(tmp_path):
    """Calling the same dependency-injected entrypoint twice with identical
    deps re-attempts the full sequence from confirmation each time -- there
    is no hidden resume/retry state carried between calls."""
    manifest_path = tmp_path / "partition.json"
    manifest = write_partition_manifest(manifest_path, t_spare=_tickers("TS", 1903))
    kwargs = dict(
        confirmation=allocator.ALLOCATION_CONFIRMATION,
        partition_manifest_path=manifest_path,
        output_path=tmp_path / "out.json",
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        anchor_reader=lambda head: _valid_anchor_for(manifest),
        clock=clock_stub,
    )
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as first:
        run(**kwargs)
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as second:
        run(**kwargs)
    assert first.value.reason == second.value.reason == "PARENT_T_SPARE_TICKER_COUNT_INVALID"
    assert first.value.authorization_consumed is True
    assert second.value.authorization_consumed is True
    assert not (tmp_path / "out.json").exists()


# ---------------------------------------------------------------------------
# FINAL_REPEAT finding HIGH-1: durable, fail-closed, one-shot consumption of
# ONE_TIME_HUMAN_AUTHORIZATION_TO_ALLOCATE_T1B -- must survive a second
# call, a new process, and a restart, not merely an in-memory flag.
# ---------------------------------------------------------------------------


def _successful_kwargs(tmp_path, manifest, output_path, *, consumption_state_root):
    return dict(
        confirmation=allocator.ALLOCATION_CONFIRMATION,
        partition_manifest_path=tmp_path / "partition.json",
        output_path=output_path,
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        anchor_reader=lambda head: _valid_anchor_for(manifest),
        clock=clock_stub,
        consumption_state_root=consumption_state_root,
    )


def test_second_call_with_same_state_root_blocks_before_private_read(tmp_path, monkeypatch):
    """A second call sharing the SAME durable consumption_state_root (as a
    real second invocation, new process, or restart would) must BLOCK
    before the private partition-manifest read is ever attempted -- it
    must not repeat the allocation."""
    manifest_path = tmp_path / "partition.json"
    t_spare = _tickers("TS", 1904)
    manifest = write_partition_manifest(manifest_path, t_spare=t_spare)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_COUNT", 1904)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256", partition.ticker_list_sha256(t_spare))
    shared_state_root = tmp_path / "gate_state"

    first_output = tmp_path / "private" / "attempt1.json"
    result = run(**_successful_kwargs(tmp_path, manifest, first_output, consumption_state_root=shared_state_root))
    assert result["t1b_ticker_count"] == 300

    read_attempts: list[str] = []
    real_read = allocator.read_partition_manifest

    def counting_read(*args, **kwargs):
        read_attempts.append("called")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(allocator, "read_partition_manifest", counting_read)

    second_output = tmp_path / "private" / "attempt2.json"
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(**_successful_kwargs(tmp_path, manifest, second_output, consumption_state_root=shared_state_root))
    assert excinfo.value.reason == "V8B_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_ALLOCATE_T1B
    assert excinfo.value.authorization_consumed is False
    assert read_attempts == []  # the private manifest was never read a second time
    assert not second_output.exists()


def test_consumption_receipt_is_a_durable_file_readable_by_a_fresh_module_state(tmp_path, monkeypatch):
    """Simulates "a new process, or restart": the receipt is plain,
    durable, fsync'd bytes on disk -- checked fresh from disk each call,
    never from any Python-process-lifetime state."""
    manifest_path = tmp_path / "partition.json"
    t_spare = _tickers("TS", 1904)
    manifest = write_partition_manifest(manifest_path, t_spare=t_spare)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_COUNT", 1904)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256", partition.ticker_list_sha256(t_spare))
    shared_state_root = tmp_path / "gate_state"

    run(**_successful_kwargs(tmp_path, manifest, tmp_path / "private" / "a.json", consumption_state_root=shared_state_root))

    receipts = list(Path(shared_state_root).glob("*.json"))
    assert len(receipts) == 1
    import json as _json

    receipt = _json.loads(receipts[0].read_bytes())
    assert receipt["gate"] == gate_consumption.GATE_ALLOCATE_T1B
    assert receipt["v8b_frozen_design_commit"] == allocator.EXPECTED_V8B_FROZEN_DESIGN_COMMIT
    # No ticker identity, path, or raw data of any kind in the receipt.
    assert set(receipt) == {
        "schema_version",
        "study_name",
        "repository",
        "gate",
        "v8b_frozen_design_commit",
        "consumed_at_utc",
    }
    assert receipt["repository"] == gate_consumption.REPOSITORY_IDENTITY

    # A brand-new call -- standing in for a new process/restart -- reusing
    # only the durable state root (no Python object shared with the call
    # above) still sees the same consumed state.
    assert gate_consumption.has_gate_been_consumed(
        shared_state_root, gate_consumption.GATE_ALLOCATE_T1B, allocator.EXPECTED_V8B_FROZEN_DESIGN_COMMIT
    )
    with pytest.raises(gate_consumption.V8BHumanGateConsumptionBlocked):
        gate_consumption.require_gate_not_yet_consumed(
            shared_state_root, gate_consumption.GATE_ALLOCATE_T1B, allocator.EXPECTED_V8B_FROZEN_DESIGN_COMMIT
        )


def test_early_gate_check_precedes_git_provenance_resolution(tmp_path):
    """The durable consumption pre-check must run before git_commit_resolver
    -- an already-consumed gate blocks even when the git resolver would
    itself raise, proving the check is not merely folded in later."""
    shared_state_root = tmp_path / "gate_state"
    gate_consumption.consume_gate_once(
        shared_state_root, gate_consumption.GATE_ALLOCATE_T1B, allocator.EXPECTED_V8B_FROZEN_DESIGN_COMMIT,
        clock=clock_stub,
    )

    def unreachable_resolver():
        raise AssertionError("git_commit_resolver must not run once already consumed")

    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=tmp_path / "unread.json",
            output_path=tmp_path / "out.json",
            git_commit_resolver=unreachable_resolver,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: {"authorization_status": "AUTHORIZED"},
            clock=clock_stub,
            consumption_state_root=shared_state_root,
        )
    assert excinfo.value.reason == "V8B_HUMAN_GATE_ALREADY_CONSUMED:" + gate_consumption.GATE_ALLOCATE_T1B
    assert excinfo.value.authorization_consumed is False


def test_public_entrypoint_uses_fixed_non_overridable_state_root():
    import inspect

    assert "consumption_state_root" not in inspect.signature(allocator.allocate_t1b_production).parameters


# ---------------------------------------------------------------------------
# Round-3 repeat HIGH-3: filesystem error privacy boundary
# ---------------------------------------------------------------------------

SECRET_PRIVATE_PATH_FRAGMENT = "/very/secret/private/allocation/output"


def test_staging_write_failure_never_leaks_private_path(tmp_path, monkeypatch):
    manifest_path = tmp_path / "partition.json"
    t_spare = _tickers("TS", 1904)
    manifest = write_partition_manifest(manifest_path, t_spare=t_spare)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_COUNT", 1904)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256", partition.ticker_list_sha256(t_spare))

    # The one-shot gate-consumption receipt (HIGH-1) is fsync'd first and
    # must succeed; only the allocation-artifact staging fsync (the second
    # fsync call) is poisoned.
    call_count = {"n": 0}

    def poisoned_fsync(fd):
        call_count["n"] += 1
        if call_count["n"] > 1:
            raise OSError(f"disk full while writing staging file at {SECRET_PRIVATE_PATH_FRAGMENT}")

    monkeypatch.setattr(allocator.os, "fsync", poisoned_fsync)
    output_path = tmp_path / "private" / "t1b_allocation_artifact.json"

    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=manifest_path,
            output_path=output_path,
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: _valid_anchor_for(manifest),
            clock=clock_stub,
        )
    assert excinfo.value.reason == "V8B_ALLOCATION_ARTIFACT_STAGING_WRITE_FAILED"
    assert SECRET_PRIVATE_PATH_FRAGMENT not in excinfo.value.reason
    assert excinfo.value.authorization_consumed is True


def test_link_publish_failure_never_leaks_private_path(tmp_path, monkeypatch):
    manifest_path = tmp_path / "partition.json"
    t_spare = _tickers("TS", 1904)
    manifest = write_partition_manifest(manifest_path, t_spare=t_spare)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_COUNT", 1904)
    monkeypatch.setattr(allocator, "EXPECTED_PARENT_T_SPARE_TICKER_LIST_SHA256", partition.ticker_list_sha256(t_spare))

    def poisoned_link(src, dst):
        # Only poison the link publishing the allocation artifact itself --
        # the one-shot gate-consumption receipt (HIGH-1) publishes via
        # ``os.link`` too, into a completely different directory, and must
        # still succeed.
        if "t1b_allocation_artifact" in str(dst):
            raise OSError(f"cross-device link from {src} to {SECRET_PRIVATE_PATH_FRAGMENT}")
        real_link(src, dst)

    real_link = allocator.os.link
    monkeypatch.setattr(allocator.os, "link", poisoned_link)
    output_path = tmp_path / "private" / "t1b_allocation_artifact.json"

    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        run(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=manifest_path,
            output_path=output_path,
            git_commit_resolver=lambda: SYNTHETIC_COMMIT,
            design_freeze_approval_reader=lambda head: {"ok": True},
            frozen_design_object_verifier=lambda: None,
            reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
            anchor_reader=lambda head: _valid_anchor_for(manifest),
            clock=clock_stub,
        )
    assert excinfo.value.reason == "V8B_ALLOCATION_ARTIFACT_ATOMIC_PUBLISH_FAILED"
    assert SECRET_PRIVATE_PATH_FRAGMENT not in excinfo.value.reason
    assert str(output_path) not in excinfo.value.reason
    assert excinfo.value.authorization_consumed is True


def test_no_retry_parameter_exists():
    import inspect

    assert "retry" not in " ".join(inspect.signature(allocator.allocate_t1b_production).parameters).lower()


def test_allocator_module_performs_zero_network_by_construction():
    """This module never imports src/v7_yahoo_collector.py or urllib at
    module scope -- structurally incapable of a Yahoo/JPX request."""
    import sys

    assert "src.v7_yahoo_collector" not in getattr(allocator, "__dict__", {})
    source_module = sys.modules[allocator.__name__]
    assert not hasattr(source_module, "fetch_chart_once")
    assert not hasattr(source_module, "urlopen")


def test_real_production_entrypoint_fails_closed_on_real_repo(tmp_path):
    """The real allocate_t1b_production, called against the real repo
    state, must fail closed today (dirty worktree mid-implementation, and
    the real review artifact does not exist yet either way) -- proves this
    phase performs zero real allocation."""
    with pytest.raises(allocator.V8BT1BAllocatorBlocked) as excinfo:
        allocator.allocate_t1b_production(
            confirmation=allocator.ALLOCATION_CONFIRMATION,
            partition_manifest_path=tmp_path / "nonexistent.json",
            output_path=tmp_path / "out.json",
        )
    assert excinfo.value.reason in {
        "PRODUCTION_GIT_WORKTREE_DIRTY",
        "PRODUCTION_GIT_HEAD_NOT_ORIGIN",
        "PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE",
        "V8B_DESIGN_FREEZE_APPROVAL_MISSING",
        "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING",
    }
