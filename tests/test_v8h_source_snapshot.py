from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src import v8_partition as v8_partition_module
from src import v8h_source_snapshot as snapshot

REVIEWED_IMPL_SHA = "1" * 40
OTHER_IMPL_SHA = "2" * 40

# Deliberately far smaller than the frozen production values (300 / 900) --
# these tests exercise the acquisition *logic*, not production block-size
# semantics.
BLOCK_SIZE = 5
MIN_FRESH = 15  # 3 x BLOCK_SIZE, mirroring the frozen 900 = 3 x 300 relation


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Category 9: real network path is not invoked by tests."""

    def forbidden(*args, **kwargs):
        raise AssertionError("real network call executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def _clock():
    return datetime(2026, 8, 20, 12, 0, 0, tzinfo=timezone.utc)


def _authorization(
    design_candidate=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
    impl_sha=REVIEWED_IMPL_SHA,
):
    return snapshot.build_authorization_identity(
        reviewed_v8h_design_candidate_commit=design_candidate,
        reviewed_source_snapshot_support_implementation_sha=impl_sha,
    )


AUTHORIZATION = _authorization()


def _runtime_state(**overrides):
    value = {
        "head": REVIEWED_IMPL_SHA,
        "authoritative_remote_head": REVIEWED_IMPL_SHA,
        "worktree_clean": True,
        "commits_after_reviewed_implementation_sha": 0,
    }
    value.update(overrides)
    return value


def _preflight(**overrides):
    value = {
        "repository_identity": snapshot.V8H_REPOSITORY_IDENTITY,
        "head": "a" * 40,
        "authoritative_remote_head": "a" * 40,
        "worktree_clean": True,
        "reviewed_v8h_design_candidate_commit": snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
        "reviewed_v8h_design_blob_sha": snapshot.V8H_DESIGN_CANDIDATE_BLOB_SHA,
        "freeze_approval_blob_sha": snapshot.V8H_FREEZE_APPROVAL_BLOB_SHA,
        "freeze_approved_frozen": True,
    }
    value.update(overrides)
    return value


def _ticker_list_sha(tickers: list[str]) -> str:
    return hashlib.sha256(("\n".join(tickers) + "\n").encode("utf-8")).hexdigest()


def _ordered_codes(total: int, *, start: int = 1000, pool: int = 3000) -> list[str]:
    candidates = [str(start + i) for i in range(pool)]
    return sorted(candidates, key=lambda code: hashlib.sha256(code.encode("utf-8")).hexdigest())[:total]


def _t0_rows_for_csv(codes: list[str], market: str = "プライム（内国株式）") -> list[dict[str, str]]:
    return [{"code": code, "market": market, "industry": "SYN_INDUSTRY"} for code in codes]


def _build_frame(t0_codes: list[str], fresh_codes: list[str]) -> pd.DataFrame:
    rows = [
        {"コード": code, "銘柄名": "SYN", "市場・区分": "プライム（内国株式）", "33業種区分": "SYN_INDUSTRY"}
        for code in t0_codes
    ]
    rows += [
        {"コード": code, "銘柄名": "SYN", "市場・区分": "スタンダード（内国株式）", "33業種区分": "SYN_INDUSTRY"}
        for code in fresh_codes
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def all_codes() -> list[str]:
    return _ordered_codes(BLOCK_SIZE + MIN_FRESH + 20)


@pytest.fixture(scope="module")
def t0_codes(all_codes) -> list[str]:
    return all_codes[:BLOCK_SIZE]


@pytest.fixture(scope="module")
def fresh_codes(all_codes) -> list[str]:
    return all_codes[BLOCK_SIZE:]


@pytest.fixture(scope="module")
def v4_fixture(tmp_path_factory, t0_codes):
    workspace = tmp_path_factory.mktemp("v8h-source-snapshot-v4fixture")
    csv_bytes = v8_partition_module.build_universe_csv_bytes(_t0_rows_for_csv(t0_codes))
    universe_csv_path = workspace / "V4_UNIVERSE.csv"
    universe_csv_path.write_bytes(csv_bytes)

    manifest_path = workspace / "V4_UNIVERSE_MANIFEST.json"
    manifest_path.write_bytes(
        json.dumps(
            {
                "source_host": "www.jpx.co.jp",
                "source_page": "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html",
                "raw_file_sha256": hashlib.sha256(b"SYNTHETIC_V4_RAW_BYTES").hexdigest(),
                "universe_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
                "ticker_list_sha256": _ticker_list_sha(t0_codes),
                "selection_rule": "synthetic fixture",
                "selected_count": BLOCK_SIZE,
                "eligible_current_only": BLOCK_SIZE + MIN_FRESH + 20,
            },
            ensure_ascii=False,
        ).encode("utf-8")
    )
    return {"manifest_path": manifest_path, "universe_csv_path": universe_csv_path}


SYNTHETIC_RAW_BYTES = b"SYNTHETIC_JPX_RAW_SOURCE_BYTES"


def _acquisition_kwargs(v4_fixture, frame, *, raw=SYNTHETIC_RAW_BYTES):
    return dict(
        raw_source_bytes=raw,
        parse_source_table=lambda _raw: frame,
        v4_manifest_path=v4_fixture["manifest_path"],
        v4_universe_csv_path=v4_fixture["universe_csv_path"],
        source_acquisition_utc=datetime(2026, 8, 20, tzinfo=timezone.utc),
        block_size=BLOCK_SIZE,
        minimum_fresh_eligible_count=MIN_FRESH,
    )


# ---------------------------------------------------------------------------
# Category 1/2 -- exact frozen design SHA/blob accepted / rejected
# ---------------------------------------------------------------------------


def test_public_preflight_accepts_exact_frozen_binding():
    validated = snapshot._validate_public_preflight(_preflight())
    assert validated["reviewed_v8h_design_candidate_commit"] == snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT
    assert validated["reviewed_v8h_design_blob_sha"] == snapshot.V8H_DESIGN_CANDIDATE_BLOB_SHA


def test_public_preflight_rejects_wrong_design_candidate():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(reviewed_v8h_design_candidate_commit="9" * 40))
    assert excinfo.value.reason == "V8H_DESIGN_CANDIDATE_MISMATCH"


def test_public_preflight_rejects_wrong_design_blob():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(reviewed_v8h_design_blob_sha="9" * 40))
    assert excinfo.value.reason == "V8H_DESIGN_CANDIDATE_BLOB_MISMATCH"


def test_public_preflight_rejects_wrong_freeze_blob():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(freeze_approval_blob_sha="9" * 40))
    assert excinfo.value.reason == "V8H_FREEZE_APPROVAL_BLOB_MISMATCH"


def test_public_preflight_rejects_freeze_not_approved():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(freeze_approved_frozen=False))
    assert excinfo.value.reason == "V8H_FREEZE_NOT_APPROVED"


def test_public_preflight_rejects_dirty_worktree():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(worktree_clean=False))
    assert excinfo.value.reason == "V8H_PUBLIC_GIT_BINDING_INVALID"


def test_public_preflight_rejects_head_not_authoritative_remote():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(authoritative_remote_head="b" * 40))
    assert excinfo.value.reason == "V8H_PUBLIC_HEAD_NOT_AUTHORITATIVE_REMOTE"


def test_freeze_approval_content_rejects_non_matching_status():
    payload = {
        "study": snapshot.V8H_STUDY_NAME,
        "frozen_design_git_commit": snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
        "frozen_design_git_blob_sha": snapshot.V8H_DESIGN_CANDIDATE_BLOB_SHA,
        "approval_status": "AWAITING_HUMAN_FREEZE_APPROVAL",
        "human_approval_received": False,
        "human_design_freeze_complete": False,
        "final_independent_design_review_result": "PASS",
        "critical": 0,
        "high": 0,
        "medium": 0,
    }
    assert snapshot._validate_freeze_approval_content(json.dumps(payload).encode("utf-8")) is False


def test_freeze_approval_content_accepts_exact_approved_record():
    payload = {
        "study": snapshot.V8H_STUDY_NAME,
        "frozen_design_git_commit": snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
        "frozen_design_git_blob_sha": snapshot.V8H_DESIGN_CANDIDATE_BLOB_SHA,
        "approval_status": "APPROVED_FROZEN",
        "human_approval_received": True,
        "human_design_freeze_complete": True,
        "final_independent_design_review_result": "PASS",
        "critical": 0,
        "high": 0,
        "medium": 0,
    }
    assert snapshot._validate_freeze_approval_content(json.dumps(payload).encode("utf-8")) is True


# ---------------------------------------------------------------------------
# Category 3 -- malformed/wrong authorization rejected
# ---------------------------------------------------------------------------


def test_authorization_grammar_matches_frozen_form():
    assert AUTHORIZATION == (
        "V8H_HUMAN_AUTHORIZE_SOURCE_SNAPSHOT_ACQUISITION_AT_"
        + snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT
        + "_WITH_"
        + REVIEWED_IMPL_SHA
    )


def test_validate_authorization_identity_accepts_exact_grammar():
    snapshot.validate_authorization_identity(
        AUTHORIZATION, reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA
    )


@pytest.mark.parametrize(
    "tamper",
    [
        lambda s: s.replace("AT_", "AT2_"),
        lambda s: s[:-1],
        lambda s: s.upper(),
        lambda s: s + "x",
        lambda s: "",
    ],
)
def test_validate_authorization_identity_rejects_tampered_string(tamper):
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.validate_authorization_identity(
            tamper(AUTHORIZATION), reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA
        )
    assert excinfo.value.reason in ("V8H_AUTHORIZATION_GRAMMAR_MISMATCH", "V8H_AUTHORIZATION_IDENTITY_REQUIRED")


def test_validate_authorization_identity_rejects_wrong_implementation_sha():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.validate_authorization_identity(
            AUTHORIZATION, reviewed_source_snapshot_support_implementation_sha=OTHER_IMPL_SHA
        )
    assert excinfo.value.reason == "V8H_AUTHORIZATION_GRAMMAR_MISMATCH"


def test_validate_authorization_identity_rejects_wrong_design_candidate():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.validate_authorization_identity(
            AUTHORIZATION,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
            reviewed_v8h_design_candidate_commit="9" * 40,
        )
    assert excinfo.value.reason == "V8H_DESIGN_CANDIDATE_MISMATCH"


def test_build_authorization_identity_rejects_wrong_length_hex():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.build_authorization_identity(
            reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha="1" * 39,
        )
    assert excinfo.value.reason == "V8H_SOURCE_SNAPSHOT_IMPLEMENTATION_SHA_INVALID"


def test_build_authorization_identity_rejects_uppercase_hex():
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked):
        snapshot.build_authorization_identity(
            reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha="A" * 40,
        )


def test_authorization_identity_never_persists_raw_string_only_its_hash():
    digest = snapshot.authorization_identity_sha256(AUTHORIZATION)
    assert digest != AUTHORIZATION
    assert len(digest) == 64
    assert digest == hashlib.sha256(AUTHORIZATION.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Category 4/12/11 -- reused receipt rejected / exclusive no-overwrite /
# post-receipt failure semantics
# ---------------------------------------------------------------------------


def test_consume_gate_once_publishes_and_second_attempt_blocks(tmp_path):
    receipt = snapshot.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    assert receipt["consumed"] is True
    assert receipt["consumption_count"] == 1
    assert receipt["consumption_boundary"] == "IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST"

    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.consume_gate_once(
            tmp_path,
            AUTHORIZATION,
            clock=_clock,
            reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8H_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"


def test_second_attempt_blocks_even_with_fresh_authorization_and_implementation(tmp_path):
    snapshot.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    fresh_authorization = _authorization(impl_sha=OTHER_IMPL_SHA)
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.consume_gate_once(
            tmp_path,
            fresh_authorization,
            clock=_clock,
            reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha=OTHER_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8H_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"


def test_receipt_key_deterministic_and_independent_of_authorization_and_shas():
    key_a = snapshot.compute_source_snapshot_gate_receipt_key()
    key_b = snapshot.compute_source_snapshot_gate_receipt_key()
    assert key_a == key_b
    # Receipt key takes no arguments -- nothing can vary it.
    assert key_a == hashlib.sha256(
        b"V8H_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_KEY_V1\0"
        + b"ta1k1-arakawa/stock-analyzer\0"
        + b"V8H_HISTORICAL_RESEARCH\0"
        + b"HUMAN_V8H_SOURCE_SNAPSHOT_ACQUISITION_GATE"
    ).hexdigest()


def test_receipt_key_never_reuses_v8g_locator_receipt_key():
    from src import v8g_private_partition_locator as locator

    assert snapshot.compute_source_snapshot_gate_receipt_key() != locator.compute_locator_gate_receipt_key()


def test_malformed_existing_receipt_blocks_and_is_never_repaired(tmp_path):
    path = tmp_path / (snapshot.compute_source_snapshot_gate_receipt_key() + ".json")
    path.write_text("not valid json", encoding="utf-8")
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.consume_gate_once(
            tmp_path,
            AUTHORIZATION,
            clock=_clock,
            reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8H_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"
    assert path.read_text(encoding="utf-8") == "not valid json"


def test_read_and_bind_gate_receipt_rejects_mismatched_binding(tmp_path):
    snapshot.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._read_and_bind_gate_receipt(
            tmp_path,
            reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha=OTHER_IMPL_SHA,
            authorization_identity=AUTHORIZATION,
        )
    assert excinfo.value.reason == "V8H_RECEIPT_IMPLEMENTATION_SHA_MISMATCH"


# ---------------------------------------------------------------------------
# Category 5/6/7 -- synthetic bytes parse successfully / T0 mismatch /
# fresh-eligible-count fail-closed
# ---------------------------------------------------------------------------


def test_source_snapshot_acquisition_passes_with_synthetic_bytes(v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    result = snapshot._perform_source_snapshot_acquisition(**_acquisition_kwargs(v4_fixture, frame))
    assert result["t0_reproduction_status"] == "PASS"
    assert result["eligible_ticker_count"] == len(t0_codes) + len(fresh_codes)
    assert result["fresh_eligible_count"] == len(fresh_codes) - 0  # legacy exclusions not present in fixture
    assert result["source_raw_sha256"] == hashlib.sha256(SYNTHETIC_RAW_BYTES).hexdigest()
    assert result["eligible_ticker_list_sha256"] == _ticker_list_sha(sorted(
        t0_codes + fresh_codes, key=lambda c: (hashlib.sha256(c.encode("utf-8")).hexdigest(), c)
    ))


def test_t0_mismatch_fails_closed(v4_fixture, t0_codes, fresh_codes):
    wrong_t0 = list(reversed(t0_codes))  # different order -> different first-block reconstruction membership
    swapped_codes = wrong_t0 + [c for c in fresh_codes if c not in wrong_t0]
    # Force a genuine mismatch: replace one T0 code with a code outside the
    # original T0 set entirely, so the reconstructed first block differs.
    mutated_fresh = list(fresh_codes)
    mutated_t0 = [mutated_fresh.pop()] + t0_codes[1:]
    frame = _build_frame(mutated_t0, mutated_fresh + [t0_codes[0]])
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._perform_source_snapshot_acquisition(**_acquisition_kwargs(v4_fixture, frame))
    assert excinfo.value.reason == "V8H_V8_T0_REPRODUCTION_MISMATCH"


def test_fresh_eligible_pool_below_minimum_fails_closed(v4_fixture, t0_codes):
    too_few_fresh = _ordered_codes(MIN_FRESH - 1, start=90000, pool=MIN_FRESH * 3)
    frame = _build_frame(t0_codes, too_few_fresh)
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._perform_source_snapshot_acquisition(**_acquisition_kwargs(v4_fixture, frame))
    assert excinfo.value.reason == "V8H_FRESH_ELIGIBLE_POOL_INSUFFICIENT"


def test_fresh_eligible_pool_excludes_legacy_exposed_tickers(v4_fixture, t0_codes, fresh_codes, monkeypatch):
    legacy_code = fresh_codes[0]
    monkeypatch.setattr(snapshot, "LEGACY_EXPOSED_TICKERS_OUTSIDE_T0", (legacy_code,))
    frame = _build_frame(t0_codes, fresh_codes)
    result = snapshot._perform_source_snapshot_acquisition(**_acquisition_kwargs(v4_fixture, frame))
    assert result["fresh_eligible_count"] == len(fresh_codes) - 1


# ---------------------------------------------------------------------------
# Category 8 -- public evidence contains hashes/counts only
# ---------------------------------------------------------------------------


def _sample_evidence(v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    acquisition_result = snapshot._perform_source_snapshot_acquisition(**_acquisition_kwargs(v4_fixture, frame))
    return snapshot._build_source_snapshot_evidence(
        reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        source_snapshot_gate_receipt_key_sha256_value=snapshot.compute_source_snapshot_gate_receipt_key(),
        source_snapshot_gate_receipt_bytes_sha256_value="0" * 64,
        acquisition_result=acquisition_result,
    )


def test_evidence_schema_is_exactly_the_frozen_field_set(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    assert set(evidence) == set(snapshot.V8H_SOURCE_SNAPSHOT_EVIDENCE_FIELDS)
    validated = snapshot._validate_source_snapshot_evidence(evidence)
    assert validated["source_snapshot_result"] == "PASS"


def test_evidence_contains_no_ticker_identity_no_raw_bytes_no_private_path(v4_fixture, t0_codes, fresh_codes, tmp_path):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    serialized = json.dumps(evidence)
    for code in t0_codes + fresh_codes:
        assert code not in serialized
    assert SYNTHETIC_RAW_BYTES.decode("ascii") not in serialized
    assert str(tmp_path) not in serialized
    assert str(v4_fixture["manifest_path"]) not in serialized
    for value in evidence.values():
        assert not isinstance(value, (bytes, bytearray))
        if isinstance(value, str) and value not in ("PASS",):
            # every remaining string field is either an enum/schema literal
            # or a hex hash/timestamp -- never a raw ticker/path/byte value
            assert (
                snapshot._HEX.fullmatch(value)
                or snapshot._TIMESTAMP_SECONDS.fullmatch(value)
                or snapshot._TIMESTAMP_MICROS.fullmatch(value)
                or value in (
                    snapshot.V8H_SOURCE_SNAPSHOT_EVIDENCE_SCHEMA_VERSION,
                    snapshot.V8H_SOURCE_SNAPSHOT_EVIDENCE_ARTIFACT_ROLE,
                    snapshot.V8H_STUDY_NAME,
                    "IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT",
                    "266999a8e48c77905dd7c7312fd41c7f38241d78",
                )
            )


def test_evidence_self_hash_recomputed_and_verified(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    tampered = dict(evidence)
    tampered["eligible_ticker_count"] = evidence["eligible_ticker_count"] + 1
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._validate_source_snapshot_evidence(tampered)
    assert excinfo.value.reason == "V8H_SOURCE_SNAPSHOT_EVIDENCE_SELF_HASH_MISMATCH"


def test_verify_evidence_binding_rejects_wrong_design_candidate(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.verify_source_snapshot_evidence_binding(
            evidence,
            authorized_reviewed_v8h_design_candidate_commit="9" * 40,
            authorized_reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8H_SOURCE_SNAPSHOT_EVIDENCE_DESIGN_CANDIDATE_MISMATCH"


def test_verify_evidence_binding_accepts_matching_authority(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    validated = snapshot.verify_source_snapshot_evidence_binding(
        evidence,
        authorized_reviewed_v8h_design_candidate_commit=snapshot.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
        authorized_reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    assert validated["source_snapshot_result"] == "PASS"


# ---------------------------------------------------------------------------
# Category 10 -- no seed/allocation functionality reachable from this stage
# ---------------------------------------------------------------------------


def test_no_seed_or_allocation_functionality_is_exposed():
    forbidden_substrings = ("seed", "allocat", "hmac", "t1_ticker", "t2_ticker", "t3_ticker", "t_spare")
    exported = set(snapshot.__all__) | {name for name in dir(snapshot) if not name.startswith("__")}
    for name in exported:
        lowered = name.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lowered, f"unexpected generation-stage capability leaked: {name}"


def test_module_grants_no_partition_generation_or_membership_disclosure_authority(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    assert evidence["partition_generation_authorized"] is False
    assert evidence["membership_disclosure_authorized"] is False
    assert evidence["research_opened"] is False
    assert evidence["historical_price_raw_acquisition_performed"] is False


# ---------------------------------------------------------------------------
# Private raw-source preservation (exclusive/no-overwrite)
# ---------------------------------------------------------------------------


def test_preserve_raw_source_bytes_once_is_exclusive(tmp_path):
    digest = hashlib.sha256(SYNTHETIC_RAW_BYTES).hexdigest()
    destination = snapshot.preserve_raw_source_bytes_once(tmp_path, digest, SYNTHETIC_RAW_BYTES)
    assert destination.read_bytes() == SYNTHETIC_RAW_BYTES
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.preserve_raw_source_bytes_once(tmp_path, digest, SYNTHETIC_RAW_BYTES)
    assert excinfo.value.reason == "V8H_PRIVATE_RAW_SOURCE_ALREADY_PRESERVED"


def test_preserve_raw_source_bytes_once_rejects_wrong_digest(tmp_path):
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot.preserve_raw_source_bytes_once(tmp_path, "not-a-hash", SYNTHETIC_RAW_BYTES)
    assert excinfo.value.reason == "V8H_SOURCE_RAW_SHA_INVALID"


# ---------------------------------------------------------------------------
# Full DI execution boundary -- end to end PASS, gate-then-fetch ordering,
# and post-gate permanence (categories 4/9/11/12 in combination)
# ---------------------------------------------------------------------------


def _execute(
    tmp_path,
    v4_fixture,
    frame,
    *,
    authorization=None,
    fetch_calls=None,
    reviewed_impl_sha=REVIEWED_IMPL_SHA,
    runtime_overrides=None,
):
    fetch_calls = fetch_calls if fetch_calls is not None else []

    def fake_fetcher():
        fetch_calls.append(1)
        return SYNTHETIC_RAW_BYTES

    return snapshot._execute_source_snapshot_acquisition_with_dependencies(
        authorization_identity=authorization or _authorization(impl_sha=reviewed_impl_sha),
        gate_state_root=tmp_path / "gate-state",
        private_state_root=tmp_path / "private-state",
        evidence_output_path=tmp_path / "evidence.json",
        jpx_fetcher=fake_fetcher,
        parse_source_table=lambda _raw: frame,
        v4_manifest_path=v4_fixture["manifest_path"],
        v4_universe_csv_path=v4_fixture["universe_csv_path"],
        repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
        public_preflight=lambda: _preflight(),
        gate_consumer=snapshot.consume_gate_once,
        clock=_clock,
        reviewed_source_snapshot_support_implementation_sha=reviewed_impl_sha,
        runtime_state_reader=lambda *_args: _runtime_state(**(runtime_overrides or {})),
        block_size=BLOCK_SIZE,
        minimum_fresh_eligible_count=MIN_FRESH,
    ), fetch_calls


def test_full_execution_passes_and_calls_fetcher_exactly_once(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    result, fetch_calls = _execute(tmp_path, v4_fixture, frame)
    assert result["result"] == "PASS"
    assert result["evidence_written"] is True
    assert len(fetch_calls) == 1
    evidence_path = tmp_path / "evidence.json"
    assert evidence_path.exists()
    published = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert set(published) == set(snapshot.V8H_SOURCE_SNAPSHOT_EVIDENCE_FIELDS)


def test_gate_is_consumed_before_fetcher_is_called(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    order: list[str] = []

    def fake_fetcher():
        order.append("fetch")
        return SYNTHETIC_RAW_BYTES

    def recording_gate_consumer(*args, **kwargs):
        order.append("gate")
        return snapshot.consume_gate_once(*args, **kwargs)

    snapshot._execute_source_snapshot_acquisition_with_dependencies(
        authorization_identity=AUTHORIZATION,
        gate_state_root=tmp_path / "gate-state",
        private_state_root=tmp_path / "private-state",
        evidence_output_path=tmp_path / "evidence.json",
        jpx_fetcher=fake_fetcher,
        parse_source_table=lambda _raw: frame,
        v4_manifest_path=v4_fixture["manifest_path"],
        v4_universe_csv_path=v4_fixture["universe_csv_path"],
        repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
        public_preflight=lambda: _preflight(),
        gate_consumer=recording_gate_consumer,
        clock=_clock,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        runtime_state_reader=lambda *_args: _runtime_state(),
        block_size=BLOCK_SIZE,
        minimum_fresh_eligible_count=MIN_FRESH,
    )
    assert order == ["gate", "fetch"]


def test_wrong_authorization_blocks_before_any_fetch_or_gate_consumption(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    fetch_calls: list[int] = []
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked):
        _execute(tmp_path, v4_fixture, frame, authorization="garbage", fetch_calls=fetch_calls)
    assert fetch_calls == []
    assert not (tmp_path / "gate-state").exists() or not any((tmp_path / "gate-state").iterdir())


def test_wrong_design_candidate_blocks_before_gate_consumption(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    fetch_calls: list[int] = []

    def fake_fetcher():
        fetch_calls.append(1)
        return SYNTHETIC_RAW_BYTES

    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._execute_source_snapshot_acquisition_with_dependencies(
            authorization_identity=AUTHORIZATION,
            gate_state_root=tmp_path / "gate-state",
            private_state_root=tmp_path / "private-state",
            evidence_output_path=tmp_path / "evidence.json",
            jpx_fetcher=fake_fetcher,
            parse_source_table=lambda _raw: frame,
            v4_manifest_path=v4_fixture["manifest_path"],
            v4_universe_csv_path=v4_fixture["universe_csv_path"],
            repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
            public_preflight=lambda: _preflight(reviewed_v8h_design_candidate_commit="9" * 40),
            gate_consumer=snapshot.consume_gate_once,
            clock=_clock,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(),
            block_size=BLOCK_SIZE,
            minimum_fresh_eligible_count=MIN_FRESH,
        )
    assert excinfo.value.reason == "V8H_DESIGN_CANDIDATE_MISMATCH"
    assert fetch_calls == []
    assert not (tmp_path / "gate-state").exists() or not any((tmp_path / "gate-state").iterdir())


def test_second_execution_attempt_blocks_without_second_fetch(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    result, fetch_calls = _execute(tmp_path, v4_fixture, frame)
    assert result["result"] == "PASS"
    assert len(fetch_calls) == 1

    second_fetch_calls: list[int] = []

    def second_fetcher():
        second_fetch_calls.append(1)
        return SYNTHETIC_RAW_BYTES

    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._execute_source_snapshot_acquisition_with_dependencies(
            authorization_identity=_authorization(impl_sha=OTHER_IMPL_SHA),
            gate_state_root=tmp_path / "gate-state",
            private_state_root=tmp_path / "private-state",
            evidence_output_path=tmp_path / "evidence-second.json",
            jpx_fetcher=second_fetcher,
            parse_source_table=lambda _raw: frame,
            v4_manifest_path=v4_fixture["manifest_path"],
            v4_universe_csv_path=v4_fixture["universe_csv_path"],
            repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
            public_preflight=lambda: _preflight(),
            gate_consumer=snapshot.consume_gate_once,
            clock=_clock,
            reviewed_source_snapshot_support_implementation_sha=OTHER_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(head=OTHER_IMPL_SHA, authoritative_remote_head=OTHER_IMPL_SHA),
            block_size=BLOCK_SIZE,
            minimum_fresh_eligible_count=MIN_FRESH,
        )
    assert excinfo.value.reason == "V8H_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"
    assert second_fetch_calls == []  # no second JPX request, ever


def test_t0_mismatch_after_gate_consumption_is_permanent_block_closed(tmp_path, v4_fixture, t0_codes, fresh_codes):
    mutated_fresh = list(fresh_codes)
    mutated_t0 = [mutated_fresh.pop()] + t0_codes[1:]
    frame = _build_frame(mutated_t0, mutated_fresh + [t0_codes[0]])

    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        _execute(tmp_path, v4_fixture, frame)
    assert excinfo.value.reason == "V8H_V8_T0_REPRODUCTION_MISMATCH"

    # Gate was already durably consumed before the fetch/T0 check ran.
    receipt = snapshot.read_gate_receipt(tmp_path / "gate-state")
    assert receipt["consumed"] is True

    # No retry: a fresh, otherwise-valid attempt still blocks on the
    # already-consumed receipt, never on T0 again.
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo2:
        _execute(tmp_path, v4_fixture, _build_frame(t0_codes, fresh_codes))
    assert excinfo2.value.reason == "V8H_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"


def test_evidence_publication_is_exclusive_no_overwrite(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    output = tmp_path / "evidence.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        _execute(tmp_path, v4_fixture, frame)
    assert excinfo.value.reason == "V8H_SOURCE_SNAPSHOT_EVIDENCE_ALREADY_EXISTS"


def test_private_raw_source_is_actually_preserved(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    _execute(tmp_path, v4_fixture, frame)
    digest = hashlib.sha256(SYNTHETIC_RAW_BYTES).hexdigest()
    preserved = (tmp_path / "private-state") / (digest + ".raw")
    assert preserved.read_bytes() == SYNTHETIC_RAW_BYTES


def test_output_path_inside_repository_rejected(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    fetch_calls: list[int] = []

    def fake_fetcher():
        fetch_calls.append(1)
        return SYNTHETIC_RAW_BYTES

    with pytest.raises(snapshot.V8HSourceSnapshotBlocked) as excinfo:
        snapshot._execute_source_snapshot_acquisition_with_dependencies(
            authorization_identity=AUTHORIZATION,
            gate_state_root=tmp_path / "gate-state",
            private_state_root=tmp_path / "private-state",
            evidence_output_path=snapshot.CANONICAL_REPOSITORY_ROOT / "V8H_LEAKED_EVIDENCE.json",
            jpx_fetcher=fake_fetcher,
            parse_source_table=lambda _raw: frame,
            v4_manifest_path=v4_fixture["manifest_path"],
            v4_universe_csv_path=v4_fixture["universe_csv_path"],
            repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
            public_preflight=lambda: _preflight(),
            gate_consumer=snapshot.consume_gate_once,
            clock=_clock,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(),
            block_size=BLOCK_SIZE,
            minimum_fresh_eligible_count=MIN_FRESH,
        )
    assert excinfo.value.reason == "V8H_OUTPUT_PATH_INVALID"
    assert fetch_calls == []


def test_resolve_and_acquire_source_snapshot_requires_explicit_fetcher_and_parser():
    import inspect

    signature = inspect.signature(snapshot.resolve_and_acquire_source_snapshot)
    assert signature.parameters["jpx_fetcher"].default is inspect.Parameter.empty
    assert signature.parameters["parse_source_table"].default is inspect.Parameter.empty
