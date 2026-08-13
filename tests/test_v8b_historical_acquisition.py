from __future__ import annotations

import json
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src import v8_partition as partition
from src import v8b_historical_acquisition as acquisition
from src import v8b_allocation as allocation
from src import v8b_allocation_verification as verification
from src import v8b_trust_pin as trust_pin

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_COMMIT = "a" * 40
SYNTHETIC_REVIEWED_COMMIT = "b" * 40


@pytest.fixture(autouse=True)
def no_real_urlopen(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("real urlopen executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


# ---------------------------------------------------------------------------
# Fake Yahoo transport
# ---------------------------------------------------------------------------


def _epoch(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp())


def synthetic_payload(
    ticker: str,
    dates: list[tuple[int, int, int]],
    price: float = 1000.0,
    *,
    bad_row_indices: list[int] | None = None,
) -> bytes:
    timestamps = [_epoch(*d) for d in dates]
    closes = [price] * len(timestamps)
    if bad_row_indices:
        for index in bad_row_indices:
            closes[index] = -1.0
    result = {
        "meta": {"symbol": ticker + ".T"},
        "timestamp": timestamps,
        "indicators": {
            "quote": [{
                "open": [price] * len(timestamps),
                "high": [price + 2.0] * len(timestamps),
                "low": [price - 2.0] * len(timestamps),
                "close": closes,
                "volume": [10000.0] * len(timestamps),
            }],
            "adjclose": [{"adjclose": [price] * len(timestamps)}],
        },
        "events": {},
    }
    return json.dumps({"chart": {"error": None, "result": [result]}}).encode("utf-8")


class FakeResponse:
    def __init__(self, payload: bytes, url: str, status: int = 200) -> None:
        self.payload = payload
        self.status = status
        self.url = url

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        pass


class FakeOpener:
    def __init__(self, payload_fn, *, host: str = "query1.finance.yahoo.com") -> None:
        self.payload_fn = payload_fn
        self.host = host
        self.calls: list[str] = []

    def __call__(self, request_obj: Any) -> FakeResponse:
        ticker = request_obj.full_url.split("/chart/")[1].split(".T?")[0]
        self.calls.append(ticker)
        payload = self.payload_fn(ticker)
        return FakeResponse(payload, url=f"https://{self.host}/v8/finance/chart/{ticker}.T")


DEFAULT_DATES = [(2016, 4, 1), (2016, 4, 4), (2025, 12, 30)]
PLACEHOLDER_TICKER = "9999"


def _date_tuples(start: tuple[int, int, int], count: int) -> list[tuple[int, int, int]]:
    start_date = date(*start)
    return [
        ((start_date + timedelta(days=i)).year, (start_date + timedelta(days=i)).month, (start_date + timedelta(days=i)).day)
        for i in range(count)
    ]


def default_opener() -> FakeOpener:
    return FakeOpener(lambda ticker: synthetic_payload(ticker, DEFAULT_DATES))


def forbidden_opener(*_args, **_kwargs):
    raise AssertionError("Yahoo opener must not run")


def clock_stub():
    return datetime(2026, 8, 12, tzinfo=timezone.utc)


def _tickers(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i:04d}" for i in range(count)]


# ---------------------------------------------------------------------------
# T1B fixture: private allocation artifact + Git-sourced trust pin
# ---------------------------------------------------------------------------


def build_t1b_fixture(tmp_path: Path, *, parent_count: int = 1904, tamper_pin: dict | None = None):
    parent = _tickers("TS", parent_count)
    artifact = allocation.build_t1b_allocation_artifact(
        parent_t_spare_tickers=parent,
        parent_v8_partition_manifest_sha256="0" * 64,
        parent_v8_partition_implementation_commit=SYNTHETIC_COMMIT,
        parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
        v8b_allocation_implementation_commit=SYNTHETIC_REVIEWED_COMMIT,
        clock=clock_stub,
    )
    result = verification.verify_t1b_allocation_artifact(
        artifact=artifact,
        parent_t_spare_tickers=parent,
        t0_tickers=_tickers("T0", 300),
        old_t1_tickers=_tickers("OT1", 300),
        t2_tickers=_tickers("T2X", 300),
        t3_tickers=_tickers("T3X", 300),
        expected_parent_t_spare_ticker_list_sha256=allocation.ticker_list_sha256(parent),
        expected_v8b_frozen_design_commit=acquisition.V8B_FROZEN_DESIGN_COMMIT,
    )
    pin = trust_pin.build_trust_pin(verification_result_summary=result, authorization_note="test")
    if tamper_pin:
        pin = {**pin, **tamper_pin}
    artifact_path = tmp_path / "t1b_allocation_artifact.json"
    artifact_path.write_bytes(allocation.canonical_json_bytes(artifact))
    return artifact, pin, artifact_path


def not_authorized_pin() -> dict:
    return {field: None for field in trust_pin.TRUST_PIN_FIELDS} | {
        "schema_version": trust_pin.SCHEMA_VERSION,
        "study_name": trust_pin.STUDY_NAME,
        "artifact_role": trust_pin.ARTIFACT_ROLE,
        "logical_block": trust_pin.LOGICAL_BLOCK,
        "authorization_status": "NOT_AUTHORIZED",
    }


def default_deps(
    *,
    block,
    opener,
    artifact_path=None,
    pin=None,
    partition_manifest_path=None,
    anchor=None,
    bridge=None,
    confirmation=None,
    t2_reuse_recheck_resolver=None,
    **overrides,
):
    deps = dict(
        output_root=None,
        block=block,
        confirmation=(
            confirmation
            if confirmation is not None
            else acquisition.ACQUISITION_CONFIRMATION_BY_BLOCK.get(block, "")
        ),
        partition_manifest_path=partition_manifest_path,
        t1b_allocation_artifact_path=artifact_path,
        git_commit_resolver=lambda: SYNTHETIC_COMMIT,
        design_freeze_approval_reader=lambda head: {"ok": True},
        frozen_design_object_verifier=lambda: None,
        reviewed_implementation_binder=lambda head: {"reviewed_implementation_git_commit": SYNTHETIC_REVIEWED_COMMIT},
        classifier_blob_resolver=lambda head: acquisition.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA,
        zoneinfo_loader=lambda: object(),
        anchor_reader=lambda head: anchor or {},
        bridge_reader=lambda head: bridge or {},
        t2_reuse_recheck_resolver=t2_reuse_recheck_resolver or (lambda head: {"result": "PASS", "block": "T2"}),
        t1b_trust_pin_reader=lambda head: pin or {},
        opener=opener,
        clock=clock_stub,
        monotonic_clock=lambda: 0.0,
        sleep_fn=lambda _s: None,
    )
    deps.update(overrides)
    return deps


def run(deps: dict, output_root: Path):
    deps = dict(deps)
    deps["output_root"] = output_root
    return acquisition._acquire_production_v8b_historical_block_bundle_with_dependencies(**deps)


# ---------------------------------------------------------------------------
# T2 fixture: synthetic partition manifest + anchor + bridge (self-consistent)
# ---------------------------------------------------------------------------


def write_partition_manifest(path: Path, *, t2=None) -> dict:
    """A genuine, self-hash-verified V8 partition manifest fixture (must
    satisfy src.v8_partition.read_partition_manifest's full schema)."""
    blocks = {
        "T0": _tickers("T0BLK", 300), "T1": _tickers("T1BLK", 300),
        "T2": list(t2 or _tickers("T2", 300)),
        "T3": _tickers("T3BLK", 300), "T_spare": _tickers("TSBLK", 300),
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


def build_t2_fixture(t2_tickers: list[str], manifest_path: Path) -> tuple[dict, dict, dict]:
    """A fully self-consistent T2 fixture -- manifest/anchor/bridge all agree
    with each other, but NOT with the real frozen HIGH-4 constants unless
    the caller also monkeypatches those (see ``patch_t2_expected_constants``).
    """
    manifest = write_partition_manifest(manifest_path, t2=t2_tickers)
    t2_hash = partition.ticker_list_sha256(t2_tickers)
    anchor = {
        "authorization_status": "AUTHORIZED",
        "authorized_partition_manifest_sha256": manifest["manifest_sha256"],
        "authorized_partition_implementation_git_commit": manifest["partition_implementation_git_commit"],
    }
    bridge = {
        "authorized_parent_v8_partition_manifest_sha256": manifest["manifest_sha256"],
        "expected_t2_ticker_list_sha256": t2_hash,
        "human_gate": "V8B_T2_AUTHORITY_INTEGRATION_OPTION_2_APPROVED",
        "v8_trust_anchor_git_identity": "7" * 40,
    }
    return manifest, anchor, bridge


def patch_t2_expected_constants(monkeypatch, t2_tickers: list[str]) -> None:
    """Redirect the HIGH-4 frozen T2 pin to match a synthetic ticker list,
    for tests that need to exercise a full successful T2 acquisition
    without any real private V8 data. Only ever used in synthetic tests;
    the real production constants (src/v8b_production_provenance.py) are
    never modified."""
    monkeypatch.setattr(acquisition, "EXPECTED_T2_TICKER_COUNT", len(t2_tickers))
    monkeypatch.setattr(acquisition, "EXPECTED_T2_TICKER_LIST_SHA256", partition.ticker_list_sha256(t2_tickers))


# ---------------------------------------------------------------------------
# Public boundary shape
# ---------------------------------------------------------------------------


def test_public_production_signature_has_only_required_inputs():
    import inspect

    assert tuple(inspect.signature(acquisition.acquire_v8b_historical_block_bundle).parameters) == (
        "output_root", "block", "confirmation", "partition_manifest_path", "t1b_allocation_artifact_path",
    )


def test_public_boundary_has_no_t1b_trust_pin_path_parameter():
    """HIGH-3: production must not accept a caller-supplied trust-pin path."""
    import inspect

    assert "t1b_trust_pin_path" not in inspect.signature(acquisition.acquire_v8b_historical_block_bundle).parameters


@pytest.mark.parametrize(
    "forbidden",
    (
        "repository_root", "opener", "clock", "monotonic_clock", "sleep_fn",
        "git_commit_resolver", "classifier_blob_resolver", "zoneinfo_loader",
        "request_start", "request_end_exclusive", "trusted_partition_path",
        "t1b_trust_pin_path",
    ),
)
def test_public_boundary_rejects_all_dependency_overrides(tmp_path, forbidden):
    kwargs = {
        "output_root": tmp_path / "private",
        "block": "T1B",
        forbidden: "override",
    }
    with pytest.raises(TypeError):
        acquisition.acquire_v8b_historical_block_bundle(**kwargs)


@pytest.mark.parametrize("block", acquisition.PROHIBITED_ACQUISITION_BLOCKS)
def test_all_prohibited_blocks_rejected(tmp_path, block):
    deps = default_deps(block=block, opener=forbidden_opener)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_BLOCK_ACQUISITION_PROHIBITED:" + block
    assert excinfo.value.authorization_consumed is False


# ---------------------------------------------------------------------------
# Pre-network ordering: zero opener calls on every pre-network failure
# ---------------------------------------------------------------------------


def test_step1_git_provenance_failure_blocks_before_network(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)

    def dirty_resolver():
        raise acquisition.V8BGitProvenanceBlocked("PRODUCTION_GIT_WORKTREE_DIRTY")

    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin, git_commit_resolver=dirty_resolver)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "PRODUCTION_GIT_WORKTREE_DIRTY"
    assert excinfo.value.authorization_consumed is False


def test_step2_frozen_design_object_failure_blocks_before_network(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)

    def failing_verifier():
        raise acquisition.V8BProductionProvenanceBlocked("V8B_FROZEN_DESIGN_OBJECT_MUTATED")

    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin, frozen_design_object_verifier=failing_verifier)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_FROZEN_DESIGN_OBJECT_MUTATED"
    assert excinfo.value.authorization_consumed is False


def test_step2_freeze_approval_failure_blocks_before_network(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)

    def failing_freeze_reader(head):
        raise acquisition.V8BProductionProvenanceBlocked("V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED")

    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin, design_freeze_approval_reader=failing_freeze_reader)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_DESIGN_FREEZE_APPROVAL_NOT_APPROVED"
    assert excinfo.value.authorization_consumed is False


def test_step3_missing_implementation_review_blocks_before_network_on_real_repo(tmp_path):
    """The real V8B_PRODUCTION_IMPLEMENTATION_REVIEW.json does not exist
    yet; the real public entrypoint must fail closed today."""
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition.acquire_v8b_historical_block_bundle(
            output_root=tmp_path / "private",
            block="T1B",
            confirmation=acquisition.T1B_ACQUISITION_CONFIRMATION,
            t1b_allocation_artifact_path=artifact_path,
        )
    assert excinfo.value.reason in {
        "PRODUCTION_GIT_WORKTREE_DIRTY",
        "PRODUCTION_GIT_HEAD_NOT_ORIGIN",
        "PRODUCTION_GIT_ORIGIN_REF_UNAVAILABLE",
        "V8B_DESIGN_FREEZE_APPROVAL_MISSING",
        "V8B_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING",
    }
    assert excinfo.value.authorization_consumed is False


def test_step3_reviewed_implementation_binder_failure_blocks_before_network(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)

    def failing_binder(head):
        raise acquisition.V8BProductionProvenanceBlocked("V8B_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:src/v8b_allocation.py")

    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin, reviewed_implementation_binder=failing_binder)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_REVIEWED_IMPLEMENTATION_BLOB_DRIFT:src/v8b_allocation.py"
    assert excinfo.value.authorization_consumed is False


def test_step4_classifier_mismatch_blocks_before_network(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin, classifier_blob_resolver=lambda head: "0" * 40)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_PRODUCTION_CLASSIFIER_VERSION_MISMATCH"
    assert excinfo.value.authorization_consumed is False


def test_step5_zoneinfo_unavailable_blocks_before_network(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)

    def failing_zoneinfo():
        raise LookupError("no tzdata")

    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin, zoneinfo_loader=failing_zoneinfo)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_ASIA_TOKYO_ZONEINFO_UNAVAILABLE"
    assert excinfo.value.authorization_consumed is False


def test_step6_t1b_trust_pin_not_authorized_blocks(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=not_authorized_pin())
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_TRUST_PIN_NOT_AUTHORIZED"
    assert excinfo.value.authorization_consumed is False


def test_step6_t1b_trust_pin_read_from_git_missing_blocks(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)

    def missing_pin_reader(head):
        raise acquisition.V8BGitProvenanceBlocked("GIT_OBJECT_READ_FAILED")

    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin, t1b_trust_pin_reader=missing_pin_reader)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_TRUSTED_ALLOCATION_MISSING"
    assert excinfo.value.authorization_consumed is False


def test_step6_t1b_allocation_artifact_self_hash_tampered_blocks(tmp_path):
    """A forged local trust pin file cannot authorize acquisition: even if
    a caller tampers the private allocation artifact on disk, the artifact
    self-hash check catches it before the (Git-sourced) pin's own checks."""
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    tampered = json.loads(artifact_path.read_bytes())
    tampered["t1b_tickers"][0] = "FORGED"
    artifact_path.write_bytes(json.dumps(tampered).encode())
    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason.startswith("V8B_ALLOCATION_ARTIFACT_INVALID:")
    assert excinfo.value.authorization_consumed is False


def test_step6_t1b_trust_pin_artifact_mismatch_blocks(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    tampered_pin = {**pin, "authorized_allocation_artifact_self_hash": "9" * 64}
    with pytest.raises(trust_pin.V8BTrustPinBlocked):
        trust_pin.validate_trust_pin(tampered_pin)  # human_gate no longer matches its own hash
    # Construct a pin that is internally well-formed (grammar matches its
    # own claimed hash) but points at a DIFFERENT artifact than the one on
    # disk -- this must still be rejected by the acquisition-level binding.
    other_pin = trust_pin.build_trust_pin(
        verification_result_summary={
            "result": "PASS",
            "parent_v8_partition_manifest_sha256": "0" * 64,
            "parent_v8_partition_implementation_commit": SYNTHETIC_COMMIT,
            "parent_t_spare_ticker_count": 1904,
            "parent_t_spare_ticker_list_sha256": "1" * 64,
            "t1b_ticker_count": 300,
            "t1b_ticker_list_sha256": "2" * 64,
            "remaining_t_spare_ticker_count": 1604,
            "remaining_t_spare_ticker_list_sha256": "3" * 64,
            "v8b_frozen_design_commit": acquisition.V8B_FROZEN_DESIGN_COMMIT,
            "v8b_allocation_implementation_commit": SYNTHETIC_REVIEWED_COMMIT,
            "artifact_self_hash": "9" * 64,
        },
        authorization_note="different artifact",
    )
    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=other_pin)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_TRUST_PIN_ALLOCATION_ARTIFACT_MISMATCH"
    assert excinfo.value.authorization_consumed is False


def test_step8_output_path_inside_repository_blocks(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, ROOT / "acquisitions_should_not_be_written_here")
    assert excinfo.value.reason == "OUTPUT_PATH_INSIDE_SOURCE_REPOSITORY"
    assert excinfo.value.authorization_consumed is False


# ---------------------------------------------------------------------------
# HIGH-4: exact V8 anchor / T2 hash enforcement -- self-consistent forgery still blocks
# ---------------------------------------------------------------------------


def test_t2_wrong_study_name_on_otherwise_valid_manifest_blocks(tmp_path, monkeypatch):
    """A manifest that matches the anchor's SHA/commit and the frozen T2
    ticker hash, but claims a different study_name, must still BLOCK --
    read_partition_manifest itself does not check study_name, so this
    module must."""
    t2_tickers = _tickers("T2", 300)
    patch_t2_expected_constants(monkeypatch, t2_tickers)
    manifest_path = tmp_path / "partition.json"
    manifest, anchor, bridge = build_t2_fixture(t2_tickers, manifest_path)
    tampered = dict(manifest)
    tampered["study_name"] = "NOT_V8_HISTORICAL_RESEARCH"
    tampered["manifest_sha256"] = partition.canonical_sha256(
        {k: v for k, v in tampered.items() if k != "manifest_sha256"}
    )
    manifest_path.write_bytes(partition.canonical_json_bytes(tampered))
    anchor = {**anchor, "authorized_partition_manifest_sha256": tampered["manifest_sha256"]}
    deps = default_deps(
        block="T2", opener=forbidden_opener, partition_manifest_path=manifest_path, anchor=anchor, bridge=bridge
    )
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "PARTITION_MANIFEST_STUDY_NAME_MISMATCH"
    assert excinfo.value.authorization_consumed is False


def test_t2_self_consistent_synthetic_manifest_still_blocks_without_frozen_pin(tmp_path):
    """A T2 manifest/anchor/bridge that are all internally self-consistent
    with EACH OTHER must still BLOCK, because none of them match the real
    frozen literal T2 count/hash this module pins to (HIGH-4) -- this is
    the default behavior of every T2 test in this file that does NOT call
    patch_t2_expected_constants."""
    t2_tickers = _tickers("T2", 300)
    manifest_path = tmp_path / "partition.json"
    manifest, anchor, bridge = build_t2_fixture(t2_tickers, manifest_path)
    deps = default_deps(
        block="T2", opener=forbidden_opener, partition_manifest_path=manifest_path, anchor=anchor, bridge=bridge
    )
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "PARTITION_TICKER_LIST_SHA_MISMATCH:T2"
    assert excinfo.value.authorization_consumed is False


def test_t2_anchor_repin_with_matching_forged_manifest_still_blocks_via_public_path(tmp_path):
    """Mirrors HIGH-4's anchor-mutation scenario end-to-end: a forged
    anchor whose stated values match a forged private manifest exactly
    still cannot authorize T2 acquisition, because the acquisition path's
    anchor_reader is expected to have already verified the anchor against
    its exact frozen Git blob -- simulated here by having anchor_reader
    itself raise, proving the acquisition function depends on that check."""
    def forged_anchor_reader(head):
        raise acquisition.V8BProductionProvenanceBlocked("V8_TRUSTED_PARTITION_BLOB_MUTATED")

    manifest_path = tmp_path / "partition.json"
    manifest_path.write_bytes(json.dumps({"manifest_sha256": "f" * 64}).encode())
    deps = default_deps(
        block="T2", opener=forbidden_opener, partition_manifest_path=manifest_path, anchor_reader=forged_anchor_reader
    )
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8_TRUSTED_PARTITION_BLOB_MUTATED"
    assert excinfo.value.authorization_consumed is False


def test_t2_bridge_cannot_be_bypassed_wrong_manifest_binding(tmp_path, monkeypatch):
    t2_tickers = _tickers("T2", 300)
    patch_t2_expected_constants(monkeypatch, t2_tickers)
    manifest_path = tmp_path / "partition.json"
    manifest, anchor, bridge = build_t2_fixture(t2_tickers, manifest_path)
    bridge["authorized_parent_v8_partition_manifest_sha256"] = "e" * 64
    deps = default_deps(
        block="T2", opener=forbidden_opener, partition_manifest_path=manifest_path, anchor=anchor, bridge=bridge
    )
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_T2_AUTHORITY_BRIDGE_MANIFEST_SHA_MISMATCH"
    assert excinfo.value.authorization_consumed is False


def test_t2_anchor_not_authorized_blocks(tmp_path, monkeypatch):
    t2_tickers = _tickers("T2", 300)
    patch_t2_expected_constants(monkeypatch, t2_tickers)
    manifest_path = tmp_path / "partition.json"
    manifest, anchor, bridge = build_t2_fixture(t2_tickers, manifest_path)
    anchor["authorization_status"] = "NOT_AUTHORIZED"
    deps = default_deps(
        block="T2", opener=forbidden_opener, partition_manifest_path=manifest_path, anchor=anchor, bridge=bridge
    )
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "TRUSTED_PARTITION_NOT_AUTHORIZED"
    assert excinfo.value.authorization_consumed is False


# ---------------------------------------------------------------------------
# Fake success: exactly 300 opener calls, atomic publication
# ---------------------------------------------------------------------------


def test_t1b_fake_success_exactly_300_opener_calls(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    opener = default_opener()
    deps = default_deps(block="T1B", opener=opener, artifact_path=artifact_path, pin=pin)
    manifest = run(deps, tmp_path / "private")
    assert len(opener.calls) == 300
    assert manifest["request_count"] == 300
    assert manifest["success_transport_count"] == 300
    assert manifest["ticker_count"] == 300
    assert manifest["block"] == "T1B"
    assert manifest["role"] == "VALIDATION"
    assert manifest["sealed"] is False
    assert manifest["retry_count"] == 0
    assert manifest["authority_chain"] == "V8B_SUCCESSOR_ALLOCATION_AUTHORITY"
    assert manifest["v8b_frozen_design_commit"] == acquisition.V8B_FROZEN_DESIGN_COMMIT
    # HIGH-2: recorded commit is the *reviewed* implementation commit, not
    # merely the (possibly later, audit-drifted) resolved HEAD.
    assert manifest["implementation_git_commit"] == SYNTHETIC_REVIEWED_COMMIT
    assert manifest["reviewed_production_implementation_commit"] == SYNTHETIC_REVIEWED_COMMIT
    assert manifest["canonical_parser_classifier_blob_sha"] == acquisition.CANONICAL_PARSER_CLASSIFIER_BLOB_SHA
    reread = acquisition.read_acquisition_manifest(tmp_path / "private", "T1B")
    assert reread == manifest


def test_t2_fake_success_exactly_300_opener_calls_and_sealed(tmp_path, monkeypatch):
    t2_tickers = _tickers("T2", 300)
    patch_t2_expected_constants(monkeypatch, t2_tickers)
    manifest_path = tmp_path / "partition.json"
    manifest_fixture, anchor, bridge = build_t2_fixture(t2_tickers, manifest_path)
    opener = default_opener()
    deps = default_deps(
        block="T2", opener=opener, partition_manifest_path=manifest_path, anchor=anchor, bridge=bridge
    )
    manifest = run(deps, tmp_path / "private")
    assert len(opener.calls) == 300
    assert manifest["block"] == "T2"
    assert manifest["role"] == "SEALED_HOLDOUT"
    assert manifest["sealed"] is True
    assert manifest["authority_chain"] == "ORIGINAL_IMMUTABLE_V8_T2_AUTHORITY_OPTION_2_BRIDGE"
    sealed_path = tmp_path / "private" / acquisition.ACQUISITIONS_DIRNAME / "T2" / acquisition.SEALED_FILENAME
    assert sealed_path.exists()


def test_t1b_atomic_no_partial_publication_on_mid_loop_block(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)

    def failing_payload(ticker: str) -> bytes:
        if ticker == artifact["t1b_tickers"][5]:
            return synthetic_payload(ticker, DEFAULT_DATES, bad_row_indices=[0, 1, 2])
        return synthetic_payload(ticker, DEFAULT_DATES)

    opener = FakeOpener(failing_payload)
    deps = default_deps(block="T1B", opener=opener, artifact_path=artifact_path, pin=pin)
    output_root = tmp_path / "private"
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked):
        run(deps, output_root)
    final_dir = output_root / acquisition.ACQUISITIONS_DIRNAME / "T1B"
    assert not final_dir.exists()
    staging_entries = list((output_root / acquisition.ACQUISITIONS_DIRNAME).iterdir())
    assert staging_entries == []


# ---------------------------------------------------------------------------
# Round-3 HIGH-1: exact block-specific confirmation token + one-shot
# authorization_consumed semantics.
# ---------------------------------------------------------------------------


def test_confirmation_constants_are_block_specific_and_distinct():
    assert acquisition.T1B_ACQUISITION_CONFIRMATION == "V8B_PRODUCTION_ACQUIRE_T1B"
    assert acquisition.T2_ACQUISITION_CONFIRMATION == "V8B_PRODUCTION_ACQUIRE_T2"
    assert acquisition.T1B_ACQUISITION_CONFIRMATION != acquisition.T2_ACQUISITION_CONFIRMATION
    assert acquisition.ACQUISITION_CONFIRMATION_BY_BLOCK == {
        "T1B": acquisition.T1B_ACQUISITION_CONFIRMATION,
        "T2": acquisition.T2_ACQUISITION_CONFIRMATION,
    }


def test_missing_confirmation_blocks_before_network(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    deps = default_deps(block="T1B", opener=forbidden_opener, artifact_path=artifact_path, pin=pin, confirmation="")
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_ACQUISITION_CONFIRMATION_INVALID"
    assert excinfo.value.authorization_consumed is False


def test_t2_token_cannot_authorize_t1b_acquisition(tmp_path):
    """A caller who supplies the T2 confirmation literal cannot acquire T1B
    -- proves the two block-specific gates cannot cross-authorize."""
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    deps = default_deps(
        block="T1B",
        opener=forbidden_opener,
        artifact_path=artifact_path,
        pin=pin,
        confirmation=acquisition.T2_ACQUISITION_CONFIRMATION,
    )
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_ACQUISITION_CONFIRMATION_INVALID"
    assert excinfo.value.authorization_consumed is False


def test_t1b_token_cannot_authorize_t2_acquisition(tmp_path, monkeypatch):
    """The inverse of the above: the T1B confirmation literal cannot
    authorize T2 acquisition."""
    t2_tickers = _tickers("T2", 300)
    patch_t2_expected_constants(monkeypatch, t2_tickers)
    manifest_path = tmp_path / "partition.json"
    manifest, anchor, bridge = build_t2_fixture(t2_tickers, manifest_path)
    deps = default_deps(
        block="T2",
        opener=forbidden_opener,
        partition_manifest_path=manifest_path,
        anchor=anchor,
        bridge=bridge,
        confirmation=acquisition.T1B_ACQUISITION_CONFIRMATION,
    )
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_ACQUISITION_CONFIRMATION_INVALID"
    assert excinfo.value.authorization_consumed is False


def test_confirmation_checked_before_any_other_pre_network_step(tmp_path):
    """An invalid confirmation blocks even when every other dependency
    would raise first if reached -- proving confirmation is checked as
    step (0), strictly before git-provenance resolution."""
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)

    def unreachable_resolver():
        raise AssertionError("git_commit_resolver must not run before confirmation is checked")

    deps = default_deps(
        block="T1B",
        opener=forbidden_opener,
        artifact_path=artifact_path,
        pin=pin,
        confirmation="WRONG_TOKEN",
        git_commit_resolver=unreachable_resolver,
    )
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_ACQUISITION_CONFIRMATION_INVALID"
    assert excinfo.value.authorization_consumed is False


def test_t2_reuse_recheck_blocked_prevents_network(tmp_path, monkeypatch):
    """HIGH-2 wiring: a T2 reuse-recheck that fails closed (as it does on
    the real repository today) must block before the first opener call."""
    t2_tickers = _tickers("T2", 300)
    patch_t2_expected_constants(monkeypatch, t2_tickers)
    manifest_path = tmp_path / "partition.json"
    manifest, anchor, bridge = build_t2_fixture(t2_tickers, manifest_path)

    def missing_recheck(head):
        raise acquisition.V8BT2PreservationRecheckBlocked("V8B_T2_REUSE_CONDITIONS_RECHECK_MISSING")

    deps = default_deps(
        block="T2",
        opener=forbidden_opener,
        partition_manifest_path=manifest_path,
        anchor=anchor,
        bridge=bridge,
        t2_reuse_recheck_resolver=missing_recheck,
    )
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.reason == "V8B_T2_REUSE_CONDITIONS_RECHECK_MISSING"
    assert excinfo.value.authorization_consumed is False


def test_authorization_consumed_true_only_once_first_opener_attempt_begins(tmp_path):
    """The very first ticker's transport failure must already report
    authorization_consumed=True -- consumption begins at the first Yahoo
    request attempt, not at its success."""
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    first_ticker = artifact["t1b_tickers"][0]

    def failing_opener(request_obj):
        raise OSError("simulated failure on the very first request")

    deps = default_deps(block="T1B", opener=failing_opener, artifact_path=artifact_path, pin=pin)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.authorization_consumed is True
    assert first_ticker  # sanity: fixture produced a first ticker to fail on


def test_authorization_consumed_true_on_failure_at_a_later_ticker_too(tmp_path):
    """Consumption never resets: a failure on ticker index 5 (after four
    successful requests) still reports authorization_consumed=True, proving
    the flag is set once at the loop's start and never toggled back."""
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    fifth_ticker = artifact["t1b_tickers"][5]

    def failing_payload(ticker: str) -> bytes:
        if ticker == fifth_ticker:
            return synthetic_payload(ticker, DEFAULT_DATES, bad_row_indices=[0, 1, 2])
        return synthetic_payload(ticker, DEFAULT_DATES)

    opener = FakeOpener(failing_payload)
    deps = default_deps(block="T1B", opener=opener, artifact_path=artifact_path, pin=pin)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert excinfo.value.authorization_consumed is True
    assert len(opener.calls) == 6


def test_no_automatic_or_manual_retry_after_authorization_consumed(tmp_path):
    """Once a V8BHistoricalAcquisitionBlocked is raised mid-loop, calling
    the same dependency-injected entrypoint again with the SAME deps must
    not resume or retry the interrupted acquisition -- it re-attempts the
    full pre-network sequence and the per-ticker loop from ticker 0, and
    the module offers no state that would let a second call "continue"
    from where the first one stopped. This proves there is no hidden retry
    path, not merely that RETRY_COUNT == 0 (already covered elsewhere)."""
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    calls: list[str] = []

    def always_failing_payload(ticker: str) -> bytes:
        calls.append(ticker)
        return synthetic_payload(ticker, DEFAULT_DATES, bad_row_indices=[0])

    opener = FakeOpener(always_failing_payload)
    deps = default_deps(block="T1B", opener=opener, artifact_path=artifact_path, pin=pin)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked):
        run(deps, tmp_path / "private")
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked):
        run(deps, tmp_path / "private")
    # Each independent call restarts from ticker 0 (no cross-call resume
    # state) and stops at the very first ticker both times.
    assert calls == [artifact["t1b_tickers"][0], artifact["t1b_tickers"][0]]


def test_successful_manifest_does_not_expose_authorization_consumed_field():
    """The manifest schema itself carries no authorization_consumed field
    -- that attribute is exposed only on the safe failure status
    (V8BHistoricalAcquisitionBlocked.authorization_consumed), never as
    part of the published, schema-checked manifest."""
    assert "authorization_consumed" not in acquisition.ACQUISITION_MANIFEST_FIELDS


# ---------------------------------------------------------------------------
# F1_C1 malformed-OHLCV thresholds: exact 1/252 and consecutive=1
# ---------------------------------------------------------------------------


def _observations(count_total: int, invalid_indices: list[int]) -> tuple[list[dict], list[dict]]:
    dates = _date_tuples((2020, 1, 1), count_total)
    valid, invalid = [], []
    for index, d in enumerate(dates):
        entry_date = date(*d).isoformat()
        if index in invalid_indices:
            invalid.append({"ticker": PLACEHOLDER_TICKER, "trading_date": entry_date, "reason": "NONFINITE_CLOSE"})
        else:
            valid.append({"ticker": PLACEHOLDER_TICKER, "trading_date": entry_date})
    return valid, invalid


def test_fraction_exactly_at_threshold_passes():
    valid, invalid = _observations(252, [0])
    acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)


def test_fraction_one_beyond_threshold_blocks():
    valid, invalid = _observations(251, [0])
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


def test_consecutive_run_of_1_passes():
    valid, invalid = _observations(260, [10])
    acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)


def test_consecutive_run_of_2_blocks():
    valid, invalid = _observations(600, [10, 11])
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:CONSECUTIVE_EXCEEDED"


def test_mandatory_2018_test_year_checked_even_though_outside_calibration_window():
    other_years_valid = [
        {"ticker": "9999", "trading_date": f"{year}-{offset // 28 + 1:02d}-{offset % 28 + 1:02d}"}
        for year in range(2016, 2026) if year != 2018
        for offset in range(60)
    ]
    year_2018 = [{"ticker": "9999", "trading_date": f"2018-01-{day:02d}"} for day in range(1, 11)]
    invalid_2018 = [dict(year_2018[0], reason="NONFINITE_CLOSE"), dict(year_2018[5], reason="NONFINITE_CLOSE")]
    valid_2018 = [row for index, row in enumerate(year_2018) if index not in (0, 5)]

    full_valid = other_years_valid + valid_2018
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(full_valid, invalid_2018)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:TEST_YEAR_FRACTION_EXCEEDED"


def test_full_p_hist_check_is_independent_of_per_year_checks():
    valid = []
    for year in acquisition.MALFORMED_OHLCV_TEST_YEARS:
        for day in range(1, 31):
            valid.append({"ticker": "9999", "trading_date": f"{year}-01-{day:02d}"})
    invalid = [{"ticker": "9999", "trading_date": "2018-02-01", "reason": "NONFINITE_CLOSE"}]
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert excinfo.value.reason == "MALFORMED_OHLCV_QUALITY_GATE:FRACTION_EXCEEDED"


def test_policy_metadata_is_exact_and_never_float():
    metadata = acquisition._malformed_ohlcv_policy_metadata()
    assert metadata["invalid_fraction_numerator"] == 1
    assert metadata["invalid_fraction_denominator"] == 252
    assert isinstance(metadata["invalid_fraction_numerator"], int)
    assert isinstance(metadata["invalid_fraction_denominator"], int)
    assert metadata["max_consecutive_invalid_returned_rows"] == 1
    assert metadata["test_years"] == list(range(2018, 2026))
    assert metadata["policy_name"] == "POLICY_V8B_Q2_F1_C1_UNIFORM_RETURNED_ROW_QUALITY_GATE"


# ---------------------------------------------------------------------------
# No retry
# ---------------------------------------------------------------------------


def test_retry_count_is_repository_fixed_zero_not_caller_overridable():
    import inspect

    assert acquisition.RETRY_COUNT == 0
    assert "retry_count" not in inspect.signature(acquisition.acquire_v8b_historical_block_bundle).parameters


def test_single_ticker_failure_aborts_whole_acquisition_no_second_attempt(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    calls: list[str] = []

    def failing_payload(ticker: str) -> bytes:
        calls.append(ticker)
        if ticker == artifact["t1b_tickers"][0]:
            return synthetic_payload(ticker, DEFAULT_DATES, bad_row_indices=[0, 1])
        return synthetic_payload(ticker, DEFAULT_DATES)

    opener = FakeOpener(failing_payload)
    deps = default_deps(block="T1B", opener=opener, artifact_path=artifact_path, pin=pin)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked):
        run(deps, tmp_path / "private")
    assert calls == [artifact["t1b_tickers"][0]]


# ---------------------------------------------------------------------------
# HIGH-6: no lower-layer error string leakage
# ---------------------------------------------------------------------------


def test_malformed_reason_strings_never_contain_a_ticker():
    valid, invalid = _observations(251, [0])
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        acquisition._require_malformed_ohlcv_quality_gate(valid, invalid)
    assert PLACEHOLDER_TICKER not in excinfo.value.reason


def test_prohibited_block_reason_never_leaks_a_file_path(tmp_path):
    deps = default_deps(block="T3", opener=forbidden_opener)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert str(tmp_path) not in excinfo.value.reason


class _SecretBearingV7Error(Exception):
    """Simulates a hypothetical future V7YahooCollectorBlocked whose
    ``.reason`` accidentally carries a secret -- proves the whitelist
    catches what it doesn't recognise, not just what it does."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@pytest.mark.parametrize(
    "secret_reason",
    (
        "TICKER_7203_INVALID",
        "https://query1.finance.yahoo.com/v8/finance/chart/7203.T?period1=123",
        "2026-08-12",
        "/home/user/private/t1b_allocation_artifact.json",
        "INVALID_DATE:" + "x" * 200,  # oversized suffix on an otherwise-safe prefix
    ),
)
def test_unrecognised_v7_collector_reason_is_redacted(secret_reason):
    safe = acquisition._safe_transport_reason(secret_reason)
    assert safe == "UNCLASSIFIED_PARSER_ERROR"
    for secret_fragment in ("7203", "yahoo.com", "2026-08-12", "private", "t1b_allocation_artifact"):
        assert secret_fragment not in safe


def test_known_safe_v7_collector_reasons_pass_through_unchanged():
    for reason in ("EMPTY_TICKER", "RESPONSE_HOST_MISMATCH", "HTTP_STATUS_404", "INDICATOR_SECTION_INVALID:quote"):
        assert acquisition._safe_transport_reason(reason) == reason


@pytest.mark.parametrize(
    "secret_error",
    (
        OSError("Connection refused to query1.finance.yahoo.com:443 for ticker 7203 at /private/path"),
        ConnectionError("secret ticker 7203 date 2026-08-12 leaked in message"),
        TimeoutError("timed out fetching https://query1.finance.yahoo.com/... ticker=7203"),
        RuntimeError("totally unexpected error mentioning ticker 7203 and /etc/passwd"),
    ),
)
def test_unclassified_exception_never_leaks_str_error_or_args(secret_error):
    reason, is_429 = acquisition._classify_transport_exception(secret_error)
    assert "7203" not in reason
    assert "yahoo.com" not in reason
    assert "2026-08-12" not in reason
    assert "/private/path" not in reason
    assert "/etc/passwd" not in reason
    assert reason in {"TRANSPORT_OS_ERROR", "TRANSPORT_CONNECTION_ERROR", "TRANSPORT_TIMEOUT", "UNCLASSIFIED_TRANSPORT_ERROR"}


def test_http_429_still_detected_without_leaking_other_text():
    class FakeHTTPError(Exception):
        def __init__(self):
            super().__init__("too many requests for ticker 7203")
            self.code = 429

    reason, is_429 = acquisition._classify_transport_exception(FakeHTTPError())
    assert reason == "HTTP_STATUS_429"
    assert is_429 is True
    assert "7203" not in reason


def test_ticker_fetch_secret_bearing_v7_error_redacted_end_to_end(tmp_path):
    """Full per-ticker-loop integration: a V7YahooCollectorBlocked whose
    .reason contains a synthetic secret must never appear in the raised
    BLOCK reason."""
    from src.v7_yahoo_collector import V7YahooCollectorBlocked

    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    secret_ticker = artifact["t1b_tickers"][0]

    def poisoned_opener(request_obj):
        raise V7YahooCollectorBlocked("SECRET_TICKER_" + secret_ticker + "_DATE_2026-08-12_LEAKED")

    deps = default_deps(block="T1B", opener=poisoned_opener, artifact_path=artifact_path, pin=pin)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert secret_ticker not in excinfo.value.reason
    assert "2026-08-12" not in excinfo.value.reason
    assert excinfo.value.reason == "TICKER_FETCH_BLOCKED:UNCLASSIFIED_PARSER_ERROR"


def test_ticker_fetch_secret_bearing_generic_exception_redacted_end_to_end(tmp_path):
    artifact, pin, artifact_path = build_t1b_fixture(tmp_path)
    secret_ticker = artifact["t1b_tickers"][0]

    def poisoned_opener(request_obj):
        raise OSError(f"connection reset while fetching {secret_ticker} at /private/output/root/path")

    deps = default_deps(block="T1B", opener=poisoned_opener, artifact_path=artifact_path, pin=pin)
    with pytest.raises(acquisition.V8BHistoricalAcquisitionBlocked) as excinfo:
        run(deps, tmp_path / "private")
    assert secret_ticker not in excinfo.value.reason
    assert "/private/output/root/path" not in excinfo.value.reason
    assert excinfo.value.reason == "TICKER_FETCH_BLOCKED:TRANSPORT_OS_ERROR"


# ---------------------------------------------------------------------------
# MEDIUM-3: no public research-opening bypass exists
# ---------------------------------------------------------------------------


def test_no_open_for_functions_are_exported():
    exported_open_for = [name for name in acquisition.__all__ if name.startswith("open_for")]
    assert exported_open_for == []


def test_no_open_for_functions_defined_at_all():
    assert not any(name.startswith("open_for") for name in dir(acquisition))


def test_no_sealed_holdout_guard_class_defined():
    assert not hasattr(acquisition, "V8BSealedHoldoutBlocked")
