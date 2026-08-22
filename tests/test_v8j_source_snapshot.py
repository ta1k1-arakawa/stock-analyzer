from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src import v8_partition as v8_partition_module
from src import v8j_source_snapshot as snapshot

REVIEWED_IMPL_SHA = "1" * 40
OTHER_IMPL_SHA = "2" * 40

# Deliberately far smaller than the frozen production values (300 / 900) --
# these tests exercise the acquisition *logic*, not production block-size
# semantics.
BLOCK_SIZE = 5
MIN_FRESH = 15  # 3 x BLOCK_SIZE, mirroring the frozen 900 = 3 x 300 relation


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Category 20: real network path is not invoked by tests."""

    def forbidden(*args, **kwargs):
        raise AssertionError("real network call executed")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    yield


def _clock():
    return datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)


def _authorization(
    design_candidate=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
    impl_sha=REVIEWED_IMPL_SHA,
):
    return snapshot.build_authorization_identity(
        reviewed_v8j_design_candidate_commit=design_candidate,
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


def _environment_preflight(**overrides):
    value = {
        "canonical_interpreter": snapshot.V8J_CANONICAL_INTERPRETER_RELATIVE_PATH,
        "checker_exit_code": 0,
        "REAL_EXECUTION_ENVIRONMENT_READY": True,
        "ENVIRONMENT_LOCK_CHECK": "PASS",
        "ENVIRONMENT_FREEZE_CHECK": "PASS",
        "ENVIRONMENT_FREEZE_EVIDENCE_GIT_SHA256_MATCH": True,
        "ENVIRONMENT_LOCK_FINGERPRINT_STATUS": "FROZEN",
        "REAL_EXECUTION_ENVIRONMENT_FROZEN": True,
        "REAL_NETWORK_REQUESTS": 0,
        "PRIVATE_READS": 0,
        "GATES_CONSUMED": 0,
    }
    value.update(overrides)
    return value


def _preflight(**overrides):
    value = {
        "repository_identity": snapshot.V8J_REPOSITORY_IDENTITY,
        "head": "a" * 40,
        "authoritative_remote_head": "a" * 40,
        "worktree_clean": True,
        "reviewed_v8j_design_candidate_commit": snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        "reviewed_v8j_design_blob_sha": snapshot.V8J_DESIGN_CANDIDATE_BLOB_SHA,
        "freeze_record_commit": snapshot.REVIEWED_V8J_FREEZE_RECORD_COMMIT,
        "freeze_approval_blob_sha": snapshot.V8J_FREEZE_APPROVAL_BLOB_SHA,
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
    workspace = tmp_path_factory.mktemp("v8j-source-snapshot-v4fixture")
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


SYNTHETIC_RAW_BYTES = b"SYNTHETIC_JPX_RAW_SOURCE_BYTES_V8J"


def _acquisition_kwargs(v4_fixture, frame, *, raw=SYNTHETIC_RAW_BYTES):
    return dict(
        raw_source_bytes=raw,
        parse_source_table=lambda _raw: frame,
        v4_manifest_path=v4_fixture["manifest_path"],
        v4_universe_csv_path=v4_fixture["universe_csv_path"],
        source_acquisition_utc=datetime(2026, 8, 21, tzinfo=timezone.utc),
        block_size=BLOCK_SIZE,
        minimum_fresh_eligible_count=MIN_FRESH,
    )


# ---------------------------------------------------------------------------
# Category 1/2 -- exact frozen V8J design SHA/blob accepted / rejected PRE_GATE
# ---------------------------------------------------------------------------


def test_public_preflight_accepts_exact_frozen_binding():
    validated = snapshot._validate_public_preflight(_preflight())
    assert validated["reviewed_v8j_design_candidate_commit"] == snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT
    assert validated["reviewed_v8j_design_blob_sha"] == snapshot.V8J_DESIGN_CANDIDATE_BLOB_SHA


def test_public_preflight_rejects_wrong_design_candidate():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(reviewed_v8j_design_candidate_commit="9" * 40))
    assert excinfo.value.reason == "V8J_DESIGN_CANDIDATE_MISMATCH"


def test_public_preflight_rejects_wrong_design_blob():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(reviewed_v8j_design_blob_sha="9" * 40))
    assert excinfo.value.reason == "V8J_DESIGN_CANDIDATE_BLOB_MISMATCH"


# ---------------------------------------------------------------------------
# Category 3 -- freeze artifact mismatch rejected PRE_GATE
# ---------------------------------------------------------------------------


def test_public_preflight_rejects_wrong_freeze_blob():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(freeze_approval_blob_sha="9" * 40))
    assert excinfo.value.reason == "V8J_FREEZE_APPROVAL_BLOB_MISMATCH"


def test_public_preflight_rejects_freeze_not_approved():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(freeze_approved_frozen=False))
    assert excinfo.value.reason == "V8J_FREEZE_NOT_APPROVED"


def test_public_preflight_rejects_dirty_worktree():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(worktree_clean=False))
    assert excinfo.value.reason == "V8J_PUBLIC_GIT_BINDING_INVALID"


def test_public_preflight_rejects_head_not_authoritative_remote():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(authoritative_remote_head="b" * 40))
    assert excinfo.value.reason == "V8J_PUBLIC_HEAD_NOT_AUTHORITATIVE_REMOTE"


# ---------------------------------------------------------------------------
# Regression: FREEZE_APPROVAL_PROVENANCE_RESOLVED_FROM_PRE_FREEZE_DESIGN_COMMIT
#
# V8J_DESIGN_FREEZE_APPROVAL.json did not exist at all at
# REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT (the freeze artifact is necessarily
# created and approved only *after* that design candidate is independently
# reviewed). The production default preflight previously tried to resolve
# both the design blob and the freeze blob from that same pre-freeze
# commit, which is impossible against real Git history and would have
# failed against commit 5964a3896518e3fb2c6fe57dd6de5b94df32b31a. These
# tests exercise the real repository object database (never a synthetic
# fixture) via src.v8c_git_provenance's own primitives, exactly as
# `_default_public_preflight` does internally.
# ---------------------------------------------------------------------------


def test_freeze_artifact_does_not_exist_at_design_candidate_commit():
    """Directly demonstrates the root cause: resolving the freeze artifact
    from the pre-freeze design-candidate commit fails, because that file
    was not yet committed at that point in history."""
    from src.v8c_git_provenance import V8CGitProvenanceBlocked, resolve_git_blob

    with pytest.raises(V8CGitProvenanceBlocked):
        resolve_git_blob(
            snapshot.CANONICAL_REPOSITORY_ROOT,
            snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            snapshot.V8J_FREEZE_APPROVAL_GIT_PATH,
        )


def test_design_artifact_resolves_from_its_own_design_candidate_commit():
    from src.v8c_git_provenance import resolve_git_blob

    resolved = resolve_git_blob(
        snapshot.CANONICAL_REPOSITORY_ROOT,
        snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        snapshot.V8J_DESIGN_DRAFT_GIT_PATH,
    )
    assert resolved == snapshot.V8J_DESIGN_CANDIDATE_BLOB_SHA


def test_freeze_artifact_resolves_from_its_own_later_freeze_record_commit():
    from src.v8c_git_provenance import read_git_object_bytes, resolve_git_blob

    resolved = resolve_git_blob(
        snapshot.CANONICAL_REPOSITORY_ROOT,
        snapshot.REVIEWED_V8J_FREEZE_RECORD_COMMIT,
        snapshot.V8J_FREEZE_APPROVAL_GIT_PATH,
    )
    assert resolved == snapshot.V8J_FREEZE_APPROVAL_BLOB_SHA
    raw = read_git_object_bytes(
        snapshot.CANONICAL_REPOSITORY_ROOT,
        snapshot.REVIEWED_V8J_FREEZE_RECORD_COMMIT,
        snapshot.V8J_FREEZE_APPROVAL_GIT_PATH,
    )
    assert snapshot._validate_freeze_approval_content(raw) is True


def test_freeze_record_commit_is_reachable_from_current_branch_history():
    """The freeze record is either current during support preparation or a
    strict ancestor after the reviewed implementation is committed."""

    head_text = snapshot._git_text(
        snapshot.CANONICAL_REPOSITORY_ROOT, ["rev-parse", "HEAD"], "unused"
    )
    assert (
        head_text == snapshot.REVIEWED_V8J_FREEZE_RECORD_COMMIT
        or int(
            snapshot._git_text(
                snapshot.CANONICAL_REPOSITORY_ROOT,
                ["rev-list", "--count", f"{snapshot.REVIEWED_V8J_FREEZE_RECORD_COMMIT}..{head_text}"],
                "TEST_ANCESTRY_CHECK",
            )
        )
        > 0
    )


def test_successful_default_style_preflight_with_both_exact_real_bindings():
    """Reconstructs exactly what `_default_public_preflight` computes from
    real Git history (design blob from the design-candidate commit,
    freeze blob/content from the separate later freeze-record commit) and
    confirms `_validate_public_preflight` now accepts both bindings
    together -- this is the exact successful path that was previously
    impossible to reach against real repository history."""
    from src.v8c_git_provenance import read_git_object_bytes, resolve_git_blob

    design_blob = resolve_git_blob(
        snapshot.CANONICAL_REPOSITORY_ROOT,
        snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        snapshot.V8J_DESIGN_DRAFT_GIT_PATH,
    )
    freeze_blob = resolve_git_blob(
        snapshot.CANONICAL_REPOSITORY_ROOT,
        snapshot.REVIEWED_V8J_FREEZE_RECORD_COMMIT,
        snapshot.V8J_FREEZE_APPROVAL_GIT_PATH,
    )
    freeze_raw = read_git_object_bytes(
        snapshot.CANONICAL_REPOSITORY_ROOT,
        snapshot.REVIEWED_V8J_FREEZE_RECORD_COMMIT,
        snapshot.V8J_FREEZE_APPROVAL_GIT_PATH,
    )
    freeze_approved_frozen = snapshot._validate_freeze_approval_content(freeze_raw)

    validated = snapshot._validate_public_preflight(
        {
            "repository_identity": snapshot.V8J_REPOSITORY_IDENTITY,
            "head": "a" * 40,
            "authoritative_remote_head": "a" * 40,
            "worktree_clean": True,
            "reviewed_v8j_design_candidate_commit": snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            "reviewed_v8j_design_blob_sha": design_blob,
            "freeze_record_commit": snapshot.REVIEWED_V8J_FREEZE_RECORD_COMMIT,
            "freeze_approval_blob_sha": freeze_blob,
            "freeze_approved_frozen": freeze_approved_frozen,
        }
    )
    assert validated["reviewed_v8j_design_blob_sha"] == snapshot.V8J_DESIGN_CANDIDATE_BLOB_SHA
    assert validated["freeze_approval_blob_sha"] == snapshot.V8J_FREEZE_APPROVAL_BLOB_SHA
    assert validated["freeze_approved_frozen"] is True


def test_wrong_freeze_record_commit_rejected_pre_gate():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_public_preflight(_preflight(freeze_record_commit="9" * 40))
    assert excinfo.value.reason == "V8J_FREEZE_RECORD_COMMIT_MISMATCH"


def test_default_public_preflight_never_raises_provenance_invalid_for_freeze_binding():
    """End-to-end call into `_default_public_preflight` itself. Regardless
    of whether the working tree happens to be clean at test-run time (an
    artifact of the surrounding development session, not of this fix),
    the freeze-artifact provenance resolution must never be the failure
    reason -- only V8J_PUBLIC_GIT_BINDING_INVALID (dirty worktree) is an
    acceptable outcome here; any provenance-mismatch reason means the
    regression has returned."""
    try:
        snapshot._default_public_preflight(snapshot.CANONICAL_REPOSITORY_ROOT)
    except snapshot.V8JSourceSnapshotBlocked as error:
        assert error.reason == "V8J_PUBLIC_GIT_BINDING_INVALID", (
            f"unexpected failure reason, regression may have returned: {error.reason}"
        )


def test_freeze_approval_content_rejects_non_matching_status():
    payload = {
        "study": snapshot.V8J_STUDY_NAME,
        "frozen_design_git_commit": snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        "frozen_design_git_blob_sha": snapshot.V8J_DESIGN_CANDIDATE_BLOB_SHA,
        "approval_status": "AWAITING_HUMAN_FREEZE_APPROVAL",
        "human_approval_received": False,
        "human_design_freeze_complete": False,
        "final_independent_design_review_result": "PASS",
        "critical": 0,
        "high": 0,
        "medium": 0,
        "jpx_acquisition_authorized": False,
        "future_profitability_established": False,
    }
    assert snapshot._validate_freeze_approval_content(json.dumps(payload).encode("utf-8")) is False


def test_freeze_approval_content_accepts_exact_approved_record():
    payload = {
        "study": snapshot.V8J_STUDY_NAME,
        "frozen_design_git_commit": snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        "frozen_design_git_blob_sha": snapshot.V8J_DESIGN_CANDIDATE_BLOB_SHA,
        "approval_status": "APPROVED_FROZEN",
        "human_approval_received": True,
        "human_design_freeze_complete": True,
        "final_independent_design_review_result": "PASS",
        "critical": 0,
        "high": 0,
        "medium": 0,
        "jpx_acquisition_authorized": False,
        "future_profitability_established": False,
    }
    assert snapshot._validate_freeze_approval_content(json.dumps(payload).encode("utf-8")) is True


def test_freeze_approval_content_rejects_v8h_study_field():
    """A V8H freeze record can never satisfy the V8J freeze check."""
    payload = {
        "study": "V8H_HISTORICAL_RESEARCH",
        "frozen_design_git_commit": snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        "frozen_design_git_blob_sha": snapshot.V8J_DESIGN_CANDIDATE_BLOB_SHA,
        "approval_status": "APPROVED_FROZEN",
        "human_approval_received": True,
        "human_design_freeze_complete": True,
        "final_independent_design_review_result": "PASS",
        "critical": 0,
        "high": 0,
        "medium": 0,
        "jpx_acquisition_authorized": False,
        "future_profitability_established": False,
    }
    assert snapshot._validate_freeze_approval_content(json.dumps(payload).encode("utf-8")) is False


# ---------------------------------------------------------------------------
# Category 4/5 -- exact V8J authorization grammar accepted / malformed
# rejected PRE_GATE
# ---------------------------------------------------------------------------


def test_authorization_grammar_matches_frozen_form():
    assert AUTHORIZATION == (
        "V8J_HUMAN_AUTHORIZE_SOURCE_SNAPSHOT_ACQUISITION_AT_"
        + snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT
        + "_WITH_"
        + REVIEWED_IMPL_SHA
    )


def test_authorization_grammar_never_equals_v8h_grammar():
    from src import v8h_source_snapshot as v8h

    v8h_auth = v8h.build_authorization_identity(
        reviewed_v8h_design_candidate_commit=v8h.REVIEWED_V8H_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    assert AUTHORIZATION != v8h_auth
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.validate_authorization_identity(
            v8h_auth, reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA
        )
    assert excinfo.value.reason == "V8J_AUTHORIZATION_GRAMMAR_MISMATCH"


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
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.validate_authorization_identity(
            tamper(AUTHORIZATION), reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA
        )
    assert excinfo.value.reason in ("V8J_AUTHORIZATION_GRAMMAR_MISMATCH", "V8J_AUTHORIZATION_IDENTITY_REQUIRED")


def test_validate_authorization_identity_rejects_wrong_implementation_sha():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.validate_authorization_identity(
            AUTHORIZATION, reviewed_source_snapshot_support_implementation_sha=OTHER_IMPL_SHA
        )
    assert excinfo.value.reason == "V8J_AUTHORIZATION_GRAMMAR_MISMATCH"


def test_validate_authorization_identity_rejects_wrong_design_candidate():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.validate_authorization_identity(
            AUTHORIZATION,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
            reviewed_v8j_design_candidate_commit="9" * 40,
        )
    assert excinfo.value.reason == "V8J_DESIGN_CANDIDATE_MISMATCH"


def test_build_authorization_identity_rejects_wrong_length_hex():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.build_authorization_identity(
            reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha="1" * 39,
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_SHA_INVALID"


def test_build_authorization_identity_rejects_uppercase_hex():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked):
        snapshot.build_authorization_identity(
            reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha="A" * 40,
        )


def test_authorization_identity_never_persists_raw_string_only_its_hash():
    digest = snapshot.authorization_identity_sha256(AUTHORIZATION)
    assert digest != AUTHORIZATION
    assert len(digest) == 64
    assert digest == hashlib.sha256(AUTHORIZATION.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Category 6 -- fixed one-shot receipt key
# ---------------------------------------------------------------------------


def test_receipt_key_deterministic_and_independent_of_authorization_and_shas():
    key_a = snapshot.compute_source_snapshot_gate_receipt_key()
    key_b = snapshot.compute_source_snapshot_gate_receipt_key()
    assert key_a == key_b
    # Receipt key takes no arguments -- nothing can vary it.
    assert key_a == hashlib.sha256(
        b"V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE_RECEIPT_KEY_V1\0"
        + b"ta1k1-arakawa/stock-analyzer\0"
        + b"V8J_HISTORICAL_RESEARCH\0"
        + b"HUMAN_V8J_SOURCE_SNAPSHOT_ACQUISITION_GATE"
    ).hexdigest()


def test_receipt_key_never_reuses_v8h_or_v8g_receipt_key():
    from src import v8g_private_partition_locator as locator
    from src import v8h_source_snapshot as v8h

    key = snapshot.compute_source_snapshot_gate_receipt_key()
    assert key != v8h.compute_source_snapshot_gate_receipt_key()
    assert key != locator.compute_locator_gate_receipt_key()


# ---------------------------------------------------------------------------
# Category 7/10 -- existing/malformed receipt rejects second attempt /
# receipt publication exclusive/no-overwrite
# ---------------------------------------------------------------------------


def test_consume_gate_once_publishes_and_second_attempt_blocks(tmp_path):
    receipt = snapshot.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    assert receipt["consumed"] is True
    assert receipt["consumption_count"] == 1
    assert receipt["consumption_boundary"] == "IMMEDIATELY_BEFORE_FIRST_JPX_REQUEST"

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.consume_gate_once(
            tmp_path,
            AUTHORIZATION,
            clock=_clock,
            reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"


def test_second_attempt_blocks_even_with_fresh_authorization_and_implementation(tmp_path):
    snapshot.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    fresh_authorization = _authorization(impl_sha=OTHER_IMPL_SHA)
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.consume_gate_once(
            tmp_path,
            fresh_authorization,
            clock=_clock,
            reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha=OTHER_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"


def test_malformed_existing_receipt_blocks_and_is_never_repaired(tmp_path):
    path = tmp_path / (snapshot.compute_source_snapshot_gate_receipt_key() + ".json")
    path.write_text("not valid json", encoding="utf-8")
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.consume_gate_once(
            tmp_path,
            AUTHORIZATION,
            clock=_clock,
            reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"
    assert path.read_text(encoding="utf-8") == "not valid json"


def test_read_and_bind_gate_receipt_rejects_mismatched_binding(tmp_path):
    snapshot.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._read_and_bind_gate_receipt(
            tmp_path,
            reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha=OTHER_IMPL_SHA,
            authorization_identity=AUTHORIZATION,
        )
    assert excinfo.value.reason == "V8J_RECEIPT_IMPLEMENTATION_SHA_MISMATCH"


# ---------------------------------------------------------------------------
# Category 8/9 -- receipt contains only pre-request fields / cannot contain
# post-request provenance
# ---------------------------------------------------------------------------


def test_receipt_schema_contains_only_pre_request_fields(tmp_path):
    receipt = snapshot.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    assert set(receipt) == set(snapshot.V8J_SOURCE_SNAPSHOT_RECEIPT_FIELDS)
    forbidden_post_request_fields = {
        "source_raw_sha256",
        "source_acquisition_utc",
        "eligible_ticker_count",
        "eligible_ticker_list_sha256",
        "t0_reproduction_status",
    }
    assert forbidden_post_request_fields.isdisjoint(set(receipt))


def test_receipt_with_injected_post_request_field_is_schema_invalid(tmp_path):
    contaminated = {
        "schema_version": snapshot.V8J_SOURCE_SNAPSHOT_RECEIPT_SCHEMA_VERSION,
        "artifact_role": snapshot.V8J_SOURCE_SNAPSHOT_RECEIPT_ARTIFACT_ROLE,
        "study": snapshot.V8J_STUDY_NAME,
        "gate": snapshot.V8J_SOURCE_SNAPSHOT_GATE,
        "reviewed_v8j_design_candidate_commit": snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        "reviewed_source_snapshot_support_implementation_sha": REVIEWED_IMPL_SHA,
        "authorization_identity_sha256": snapshot.authorization_identity_sha256(AUTHORIZATION),
        "consumed": True,
        "consumption_count": 1,
        "consumption_boundary": snapshot.V8J_SOURCE_SNAPSHOT_GATE_CONSUMPTION_BOUNDARY,
        "consumption_timestamp_utc": "2026-08-21T12:00:00Z",
        "source_raw_sha256": "0" * 64,  # forbidden post-request field
    }
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_receipt(contaminated)
    assert excinfo.value.reason == "V8J_RECEIPT_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# Category 11/12/13 -- synthetic bytes parse successfully / T0 mismatch /
# fresh-eligible-count fail-closed
# ---------------------------------------------------------------------------


def test_source_snapshot_acquisition_passes_with_synthetic_bytes(v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    result = snapshot._perform_source_snapshot_acquisition(**_acquisition_kwargs(v4_fixture, frame))
    assert result["t0_reproduction_status"] == "PASS"
    assert result["eligible_ticker_count"] == len(t0_codes) + len(fresh_codes)
    assert result["fresh_eligible_count"] == len(fresh_codes)
    assert result["source_raw_sha256"] == hashlib.sha256(SYNTHETIC_RAW_BYTES).hexdigest()


def test_t0_mismatch_fails_closed(v4_fixture, t0_codes, fresh_codes):
    mutated_fresh = list(fresh_codes)
    mutated_t0 = [mutated_fresh.pop()] + t0_codes[1:]
    frame = _build_frame(mutated_t0, mutated_fresh + [t0_codes[0]])
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._perform_source_snapshot_acquisition(**_acquisition_kwargs(v4_fixture, frame))
    assert excinfo.value.reason == "V8J_V8_T0_REPRODUCTION_MISMATCH"


def test_fresh_eligible_pool_below_minimum_fails_closed(v4_fixture, t0_codes):
    too_few_fresh = _ordered_codes(MIN_FRESH - 1, start=90000, pool=MIN_FRESH * 3)
    frame = _build_frame(t0_codes, too_few_fresh)
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._perform_source_snapshot_acquisition(**_acquisition_kwargs(v4_fixture, frame))
    assert excinfo.value.reason == "V8J_FRESH_ELIGIBLE_POOL_INSUFFICIENT"


def _execute(
    tmp_path,
    v4_fixture,
    frame,
    *,
    authorization=None,
    fetch_calls=None,
    reviewed_impl_sha=REVIEWED_IMPL_SHA,
    runtime_overrides=None,
    clock=_clock,
    parse_source_table=None,
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
        parse_source_table=parse_source_table or (lambda _raw: frame),
        v4_manifest_path=v4_fixture["manifest_path"],
        v4_universe_csv_path=v4_fixture["universe_csv_path"],
        repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
        public_preflight=lambda: _preflight(),
        gate_consumer=snapshot.consume_gate_once,
        clock=clock,
        reviewed_source_snapshot_support_implementation_sha=reviewed_impl_sha,
        runtime_state_reader=lambda *_args: _runtime_state(**(runtime_overrides or {})),
        environment_preflight=lambda _root: _environment_preflight(),
        block_size=BLOCK_SIZE,
        minimum_fresh_eligible_count=MIN_FRESH,
    ), fetch_calls


def _preserved_raw_path(tmp_path):
    return tmp_path / "private-state" / (hashlib.sha256(SYNTHETIC_RAW_BYTES).hexdigest() + ".raw")


def test_successful_execution_preserves_exact_raw_bytes_before_parser(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    parser_calls: list[str] = []

    def parser(raw):
        parser_calls.append("parser")
        preserved = _preserved_raw_path(tmp_path)
        assert preserved.read_bytes() == raw
        return frame

    result, fetch_calls = _execute(tmp_path, v4_fixture, frame, parse_source_table=parser)
    assert result["result"] == "PASS"
    assert fetch_calls == [1]
    assert parser_calls == ["parser"]


def test_t0_mismatch_after_gate_consumption_is_permanent_terminal(tmp_path, v4_fixture, t0_codes, fresh_codes):
    mutated_fresh = list(fresh_codes)
    mutated_t0 = [mutated_fresh.pop()] + t0_codes[1:]
    frame = _build_frame(mutated_t0, mutated_fresh + [t0_codes[0]])

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        _execute(tmp_path, v4_fixture, frame)
    assert excinfo.value.reason == "V8J_V8_T0_REPRODUCTION_MISMATCH"

    # Gate was already durably consumed before the fetch/T0 check ran.
    receipt = snapshot.read_gate_receipt(tmp_path / "gate-state")
    assert receipt["consumed"] is True
    assert _preserved_raw_path(tmp_path).read_bytes() == SYNTHETIC_RAW_BYTES

    # No retry: a fresh, otherwise-valid attempt still blocks on the
    # already-consumed receipt, never on T0 again.
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo2:
        _execute(tmp_path, v4_fixture, _build_frame(t0_codes, fresh_codes))
    assert excinfo2.value.reason == "V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"


def test_fresh_count_below_minimum_after_gate_consumption_is_permanent_terminal(tmp_path, v4_fixture, t0_codes):
    too_few_fresh = _ordered_codes(MIN_FRESH - 1, start=90000, pool=MIN_FRESH * 3)
    frame = _build_frame(t0_codes, too_few_fresh)

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        _execute(tmp_path, v4_fixture, frame)
    assert excinfo.value.reason == "V8J_FRESH_ELIGIBLE_POOL_INSUFFICIENT"

    receipt = snapshot.read_gate_receipt(tmp_path / "gate-state")
    assert receipt["consumed"] is True
    assert _preserved_raw_path(tmp_path).read_bytes() == SYNTHETIC_RAW_BYTES

    second_fetch_calls: list[int] = []
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo2:
        _execute(tmp_path, v4_fixture, frame, fetch_calls=second_fetch_calls)
    assert excinfo2.value.reason == "V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"
    assert second_fetch_calls == []


def test_parser_failure_after_fetch_preserves_raw_and_blocks_second_fetch(tmp_path, v4_fixture, t0_codes, fresh_codes):
    parser_calls: list[str] = []

    def raising_parser(_raw):
        parser_calls.append("parser")
        raise ValueError("synthetic parser failure")

    with pytest.raises(ValueError, match="synthetic parser failure"):
        _execute(
            tmp_path,
            v4_fixture,
            _build_frame(t0_codes, fresh_codes),
            parse_source_table=raising_parser,
        )
    assert parser_calls == ["parser"]
    assert snapshot.read_gate_receipt(tmp_path / "gate-state")["consumed"] is True
    assert _preserved_raw_path(tmp_path).read_bytes() == SYNTHETIC_RAW_BYTES
    assert not (tmp_path / "evidence.json").exists()

    second_fetch_calls: list[int] = []
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        _execute(
            tmp_path,
            v4_fixture,
            _build_frame(t0_codes, fresh_codes),
            fetch_calls=second_fetch_calls,
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"
    assert second_fetch_calls == []


def test_post_fetch_clock_failure_preserves_raw_before_processing(tmp_path, v4_fixture, t0_codes, fresh_codes):
    class ReceiptThenFailClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls == 1:
                return _clock()
            raise RuntimeError("synthetic post-fetch clock failure")

    clock = ReceiptThenFailClock()
    parser_calls: list[str] = []
    with pytest.raises(RuntimeError, match="synthetic post-fetch clock failure"):
        _execute(
            tmp_path,
            v4_fixture,
            _build_frame(t0_codes, fresh_codes),
            clock=clock,
            parse_source_table=lambda _raw: parser_calls.append("parser"),
        )
    assert clock.calls == 2
    assert parser_calls == []
    assert snapshot.read_gate_receipt(tmp_path / "gate-state")["consumed"] is True
    assert _preserved_raw_path(tmp_path).read_bytes() == SYNTHETIC_RAW_BYTES
    assert not (tmp_path / "evidence.json").exists()


# ---------------------------------------------------------------------------
# Category 14/15/16 -- evidence binds exact receipt-key/bytes hash; stale or
# substituted receipt blocks evidence
# ---------------------------------------------------------------------------


def _sample_evidence(v4_fixture, t0_codes, fresh_codes, *, receipt_key=None, receipt_bytes=None):
    frame = _build_frame(t0_codes, fresh_codes)
    acquisition_result = snapshot._perform_source_snapshot_acquisition(**_acquisition_kwargs(v4_fixture, frame))
    return snapshot._build_source_snapshot_evidence(
        reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        source_snapshot_gate_receipt_key_sha256_value=receipt_key or snapshot.compute_source_snapshot_gate_receipt_key(),
        source_snapshot_gate_receipt_bytes_sha256_value=receipt_bytes or "0" * 64,
        acquisition_result=acquisition_result,
    )


def test_evidence_binds_exact_receipt_key_hash(tmp_path, v4_fixture, t0_codes, fresh_codes):
    result, _fetch_calls = _execute(tmp_path, v4_fixture, _build_frame(t0_codes, fresh_codes))
    published = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert published["source_snapshot_gate_receipt_key_sha256"] == snapshot.compute_source_snapshot_gate_receipt_key()
    assert result["source_snapshot_gate_receipt_key_sha256"] == snapshot.compute_source_snapshot_gate_receipt_key()


def test_evidence_binds_exact_receipt_bytes_hash(tmp_path, v4_fixture, t0_codes, fresh_codes):
    _execute(tmp_path, v4_fixture, _build_frame(t0_codes, fresh_codes))
    published = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    actual_receipt_bytes_sha = snapshot.gate_receipt_bytes_sha256(tmp_path / "gate-state")
    assert published["source_snapshot_gate_receipt_bytes_sha256"] == actual_receipt_bytes_sha


def test_evidence_with_wrong_receipt_key_binding_is_rejected(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes, receipt_key="9" * 64)
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.verify_source_snapshot_evidence_binding(
            evidence,
            authorized_reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            authorized_reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_EVIDENCE_RECEIPT_KEY_MISMATCH"


def test_stale_receipt_read_before_publication_bound_via_read_and_bind(tmp_path):
    snapshot.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    # A "substituted" receipt claim -- caller now presents a *different*
    # authorized implementation SHA than the one the durable receipt was
    # actually consumed under. This must block before any evidence step.
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._read_and_bind_gate_receipt(
            tmp_path,
            reviewed_v8j_design_candidate_commit=snapshot.REVIEWED_V8J_DESIGN_CANDIDATE_COMMIT,
            reviewed_source_snapshot_support_implementation_sha=OTHER_IMPL_SHA,
            authorization_identity=AUTHORIZATION,
        )
    assert excinfo.value.reason == "V8J_RECEIPT_IMPLEMENTATION_SHA_MISMATCH"


# ---------------------------------------------------------------------------
# Category 17 -- public evidence contains no ticker identities/private
# path/raw payload/raw authorization
# ---------------------------------------------------------------------------


def test_evidence_schema_is_exactly_the_frozen_field_set(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    assert set(evidence) == set(snapshot.V8J_SOURCE_SNAPSHOT_EVIDENCE_FIELDS)
    validated = snapshot._validate_source_snapshot_evidence(evidence)
    assert validated["source_snapshot_result"] == "PASS"


def test_evidence_contains_no_ticker_identity_no_raw_bytes_no_private_path_no_raw_auth(
    v4_fixture, t0_codes, fresh_codes, tmp_path
):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    serialized = json.dumps(evidence)
    for code in t0_codes + fresh_codes:
        assert code not in serialized
    assert SYNTHETIC_RAW_BYTES.decode("ascii") not in serialized
    assert str(tmp_path) not in serialized
    assert str(v4_fixture["manifest_path"]) not in serialized
    assert AUTHORIZATION not in serialized
    for value in evidence.values():
        assert not isinstance(value, (bytes, bytearray))
        if isinstance(value, str) and value not in ("PASS",):
            assert (
                snapshot._HEX.fullmatch(value)
                or snapshot._TIMESTAMP_SECONDS.fullmatch(value)
                or snapshot._TIMESTAMP_MICROS.fullmatch(value)
                or value in (
                    snapshot.V8J_SOURCE_SNAPSHOT_EVIDENCE_SCHEMA_VERSION,
                    snapshot.V8J_SOURCE_SNAPSHOT_EVIDENCE_ARTIFACT_ROLE,
                    snapshot.V8J_STUDY_NAME,
                    "IMPLEMENTATION_TIME_OFFICIAL_JPX_SNAPSHOT",
                    "266999a8e48c77905dd7c7312fd41c7f38241d78",
                )
            )


def test_evidence_self_hash_recomputed_and_verified(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    tampered = dict(evidence)
    tampered["eligible_ticker_count"] = evidence["eligible_ticker_count"] + 1
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_source_snapshot_evidence(tampered)
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_EVIDENCE_SELF_HASH_MISMATCH"


def test_verify_evidence_binding_rejects_wrong_design_candidate(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.verify_source_snapshot_evidence_binding(
            evidence,
            authorized_reviewed_v8j_design_candidate_commit="9" * 40,
            authorized_reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_EVIDENCE_DESIGN_CANDIDATE_MISMATCH"


# ---------------------------------------------------------------------------
# Category 18/19 -- private preservation failure / evidence publication
# failure both terminal post-gate
# ---------------------------------------------------------------------------


def test_preserve_raw_source_bytes_once_is_exclusive(tmp_path):
    digest = hashlib.sha256(SYNTHETIC_RAW_BYTES).hexdigest()
    destination = snapshot.preserve_raw_source_bytes_once(tmp_path, digest, SYNTHETIC_RAW_BYTES)
    assert destination.read_bytes() == SYNTHETIC_RAW_BYTES
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.preserve_raw_source_bytes_once(tmp_path, digest, SYNTHETIC_RAW_BYTES)
    assert excinfo.value.reason == "V8J_PRIVATE_RAW_SOURCE_ALREADY_PRESERVED"


def test_preserve_raw_source_bytes_once_rejects_wrong_digest(tmp_path):
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.preserve_raw_source_bytes_once(tmp_path, "not-a-hash", SYNTHETIC_RAW_BYTES)
    assert excinfo.value.reason == "V8J_SOURCE_RAW_SHA_INVALID"


def test_private_preservation_failure_is_terminal_post_gate(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    # Pre-seed a colliding private-state file so preservation fails after
    # the gate has already been consumed.
    digest = hashlib.sha256(SYNTHETIC_RAW_BYTES).hexdigest()
    private_state = tmp_path / "private-state"
    private_state.mkdir(parents=True)
    (private_state / (digest + ".raw")).write_bytes(b"already-there")

    parser_calls: list[str] = []
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        _execute(
            tmp_path,
            v4_fixture,
            frame,
            parse_source_table=lambda _raw: parser_calls.append("parser"),
        )
    assert excinfo.value.reason == "V8J_PRIVATE_RAW_SOURCE_ALREADY_PRESERVED"
    assert parser_calls == []

    receipt = snapshot.read_gate_receipt(tmp_path / "gate-state")
    assert receipt["consumed"] is True
    second_fetch_calls: list[int] = []
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo2:
        _execute(tmp_path, v4_fixture, frame, fetch_calls=second_fetch_calls)
    assert excinfo2.value.reason == "V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"
    assert second_fetch_calls == []


def test_acquisition_result_raw_hash_mismatch_is_terminal_after_preservation(
    monkeypatch, tmp_path, v4_fixture, t0_codes, fresh_codes
):
    monkeypatch.setattr(
        snapshot,
        "_perform_source_snapshot_acquisition",
        lambda **_kwargs: {"source_raw_sha256": "0" * 64},
    )

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        _execute(tmp_path, v4_fixture, _build_frame(t0_codes, fresh_codes))
    assert excinfo.value.reason == "V8J_ACQUISITION_RESULT_RAW_SHA_MISMATCH"
    assert snapshot.read_gate_receipt(tmp_path / "gate-state")["consumed"] is True
    assert _preserved_raw_path(tmp_path).read_bytes() == SYNTHETIC_RAW_BYTES
    assert not (tmp_path / "evidence.json").exists()


def test_evidence_output_preexisting_before_gate_consumption_is_pre_gate(tmp_path, v4_fixture, t0_codes, fresh_codes):
    """Evidence-destination collision detected before any gate work is a
    PRE_GATE safety check (mirrors V8G's own pre-flight path validation):
    the gate must remain unconsumed in this case."""
    frame = _build_frame(t0_codes, fresh_codes)
    output = tmp_path / "evidence.json"
    output.write_text("{}", encoding="utf-8")
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        _execute(tmp_path, v4_fixture, frame)
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_EVIDENCE_ALREADY_EXISTS"
    assert not (tmp_path / "gate-state").exists() or not any((tmp_path / "gate-state").iterdir())


def test_evidence_publication_failure_after_gate_consumption_is_terminal_post_gate(
    tmp_path, v4_fixture, t0_codes, fresh_codes
):
    """A race that creates the evidence destination strictly *after* the
    gate's durable receipt is already published must still be terminal:
    no retry, no reset, gate stays consumed."""
    frame = _build_frame(t0_codes, fresh_codes)
    evidence_output = tmp_path / "evidence.json"

    def racing_gate_consumer(*args, **kwargs):
        receipt = snapshot.consume_gate_once(*args, **kwargs)
        # Simulate a concurrent writer winning the race for the evidence
        # destination strictly after the gate is genuinely consumed.
        evidence_output.write_text("{}", encoding="utf-8")
        return receipt

    def fake_fetcher():
        return SYNTHETIC_RAW_BYTES

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._execute_source_snapshot_acquisition_with_dependencies(
            authorization_identity=AUTHORIZATION,
            gate_state_root=tmp_path / "gate-state",
            private_state_root=tmp_path / "private-state",
            evidence_output_path=evidence_output,
            jpx_fetcher=fake_fetcher,
            parse_source_table=lambda _raw: frame,
            v4_manifest_path=v4_fixture["manifest_path"],
            v4_universe_csv_path=v4_fixture["universe_csv_path"],
            repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
            public_preflight=lambda: _preflight(),
            gate_consumer=racing_gate_consumer,
            clock=_clock,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(),
            environment_preflight=lambda _root: _environment_preflight(),
            block_size=BLOCK_SIZE,
            minimum_fresh_eligible_count=MIN_FRESH,
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_EVIDENCE_ALREADY_EXISTS"

    receipt = snapshot.read_gate_receipt(tmp_path / "gate-state")
    assert receipt["consumed"] is True


# ---------------------------------------------------------------------------
# Category 20 -- no real network request occurs in tests
# (also enforced globally by the autouse no_real_network fixture)
# ---------------------------------------------------------------------------


def test_full_execution_passes_and_calls_fetcher_exactly_once(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    result, fetch_calls = _execute(tmp_path, v4_fixture, frame)
    assert result["result"] == "PASS"
    assert result["evidence_written"] is True
    assert len(fetch_calls) == 1
    evidence_path = tmp_path / "evidence.json"
    assert evidence_path.exists()
    published = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert set(published) == set(snapshot.V8J_SOURCE_SNAPSHOT_EVIDENCE_FIELDS)
    assert published["source_raw_sha256"] == hashlib.sha256(SYNTHETIC_RAW_BYTES).hexdigest()
    assert _preserved_raw_path(tmp_path).read_bytes() == SYNTHETIC_RAW_BYTES


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
        environment_preflight=lambda _root: _environment_preflight(),
        block_size=BLOCK_SIZE,
        minimum_fresh_eligible_count=MIN_FRESH,
    )
    assert order == ["gate", "fetch"]


def test_wrong_authorization_blocks_before_any_fetch_or_gate_consumption(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    fetch_calls: list[int] = []
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked):
        _execute(tmp_path, v4_fixture, frame, authorization="garbage", fetch_calls=fetch_calls)
    assert fetch_calls == []
    assert not (tmp_path / "gate-state").exists() or not any((tmp_path / "gate-state").iterdir())


def test_wrong_design_candidate_blocks_before_gate_consumption(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    fetch_calls: list[int] = []

    def fake_fetcher():
        fetch_calls.append(1)
        return SYNTHETIC_RAW_BYTES

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
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
            public_preflight=lambda: _preflight(reviewed_v8j_design_candidate_commit="9" * 40),
            gate_consumer=snapshot.consume_gate_once,
            clock=_clock,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(),
            environment_preflight=lambda _root: _environment_preflight(),
            block_size=BLOCK_SIZE,
            minimum_fresh_eligible_count=MIN_FRESH,
        )
    assert excinfo.value.reason == "V8J_DESIGN_CANDIDATE_MISMATCH"
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

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
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
            environment_preflight=lambda _root: _environment_preflight(),
            block_size=BLOCK_SIZE,
            minimum_fresh_eligible_count=MIN_FRESH,
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_GATE_ALREADY_CONSUMED"
    assert second_fetch_calls == []  # no second JPX request, ever


def test_output_path_inside_repository_rejected(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    fetch_calls: list[int] = []

    def fake_fetcher():
        fetch_calls.append(1)
        return SYNTHETIC_RAW_BYTES

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._execute_source_snapshot_acquisition_with_dependencies(
            authorization_identity=AUTHORIZATION,
            gate_state_root=tmp_path / "gate-state",
            private_state_root=tmp_path / "private-state",
            evidence_output_path=snapshot.CANONICAL_REPOSITORY_ROOT / "V8J_LEAKED_EVIDENCE.json",
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
            environment_preflight=lambda _root: _environment_preflight(),
            block_size=BLOCK_SIZE,
            minimum_fresh_eligible_count=MIN_FRESH,
        )
    assert excinfo.value.reason == "V8J_OUTPUT_PATH_INVALID"
    assert fetch_calls == []


def test_resolve_and_acquire_source_snapshot_requires_explicit_fetcher_and_parser():
    import inspect

    signature = inspect.signature(snapshot.resolve_and_acquire_source_snapshot)
    assert signature.parameters["jpx_fetcher"].default is inspect.Parameter.empty
    assert signature.parameters["parse_source_table"].default is inspect.Parameter.empty
    assert "clock" not in signature.parameters


# ---------------------------------------------------------------------------
# Category 21 -- no seed/allocation path exists
# ---------------------------------------------------------------------------


def test_no_seed_or_allocation_functionality_is_exposed():
    forbidden_substrings = ("seed", "allocat", "hmac", "t1_ticker", "t2_ticker", "t3_ticker", "t_spare")
    exported = set(snapshot.__all__) | {name for name in dir(snapshot) if not name.startswith("__")}
    for name in exported:
        lowered = name.lower()
        for forbidden in forbidden_substrings:
            assert forbidden not in lowered, f"unexpected generation-stage capability leaked: {name}"


# ---------------------------------------------------------------------------
# V8J-specific frozen environment and predecessor-isolation coverage.
# ---------------------------------------------------------------------------


def test_environment_critical_git_blobs_match_frozen_promotion_baseline():
    snapshot._verify_environment_critical_git_blobs(snapshot.CANONICAL_REPOSITORY_ROOT)


def test_environment_critical_blob_mismatch_is_pre_gate(monkeypatch):
    original = snapshot.resolve_git_blob

    def mismatched_blob(repository_root, commit, git_path):
        value = original(repository_root, commit, git_path)
        if commit == snapshot.V8J_ENVIRONMENT_FREEZE_PROMOTION_COMMIT and git_path.endswith("lock.txt"):
            return "0" * 40
        return value

    monkeypatch.setattr(snapshot, "resolve_git_blob", mismatched_blob)
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._verify_environment_critical_git_blobs(snapshot.CANONICAL_REPOSITORY_ROOT)
    assert excinfo.value.reason == "V8J_ENVIRONMENT_CRITICAL_BLOB_MISMATCH"


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("canonical_interpreter", "python.exe"),
        ("checker_exit_code", 1),
        ("REAL_EXECUTION_ENVIRONMENT_READY", False),
        ("ENVIRONMENT_LOCK_CHECK", "FAIL"),
        ("ENVIRONMENT_FREEZE_CHECK", "FAIL"),
        ("ENVIRONMENT_FREEZE_EVIDENCE_GIT_SHA256_MATCH", False),
        ("ENVIRONMENT_LOCK_FINGERPRINT_STATUS", "CANDIDATE_VERIFIED_NOT_FROZEN"),
        ("REAL_EXECUTION_ENVIRONMENT_FROZEN", False),
        ("REAL_NETWORK_REQUESTS", 1),
        ("PRIVATE_READS", 1),
        ("GATES_CONSUMED", 1),
    ],
)
def test_each_environment_pre_gate_requirement_fails_closed(field, bad_value):
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_environment_preflight(_environment_preflight(**{field: bad_value}))
    assert excinfo.value.reason == "V8J_ENVIRONMENT_PRE_GATE_BLOCK"


def test_environment_preflight_rejects_missing_or_extra_field():
    missing = _environment_preflight()
    missing.pop("GATES_CONSUMED")
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked):
        snapshot._validate_environment_preflight(missing)
    extra = _environment_preflight(unexpected=True)
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked):
        snapshot._validate_environment_preflight(extra)


def _checker_payload(**overrides):
    payload = _environment_preflight()
    payload.pop("canonical_interpreter")
    payload.pop("checker_exit_code")
    payload.update(overrides)
    return payload


def test_default_environment_preflight_rejects_canonical_interpreter_mismatch(monkeypatch):
    monkeypatch.setattr(snapshot.sys, "executable", str(snapshot.CANONICAL_REPOSITORY_ROOT / "wrong-python.exe"))
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._default_environment_preflight(snapshot.CANONICAL_REPOSITORY_ROOT)
    assert excinfo.value.reason == "V8J_CANONICAL_INTERPRETER_MISMATCH"


def test_default_environment_preflight_rejects_checker_invalid_json(monkeypatch):
    canonical = snapshot._canonical_interpreter_path(snapshot.CANONICAL_REPOSITORY_ROOT)
    monkeypatch.setattr(snapshot.sys, "executable", str(canonical))
    monkeypatch.setattr(snapshot.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"not-json"))
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._default_environment_preflight(snapshot.CANONICAL_REPOSITORY_ROOT)
    assert excinfo.value.reason == "V8J_ENVIRONMENT_CHECKER_INVALID_JSON"


def test_default_environment_preflight_rejects_checker_nonzero(monkeypatch):
    canonical = snapshot._canonical_interpreter_path(snapshot.CANONICAL_REPOSITORY_ROOT)
    monkeypatch.setattr(snapshot.sys, "executable", str(canonical))
    monkeypatch.setattr(
        snapshot.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=json.dumps(_checker_payload()).encode("utf-8")),
    )
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._default_environment_preflight(snapshot.CANONICAL_REPOSITORY_ROOT)
    assert excinfo.value.reason == "V8J_ENVIRONMENT_PRE_GATE_BLOCK"


def test_default_environment_preflight_accepts_complete_checker_pass(monkeypatch):
    canonical = snapshot._canonical_interpreter_path(snapshot.CANONICAL_REPOSITORY_ROOT)
    monkeypatch.setattr(snapshot.sys, "executable", str(canonical))
    monkeypatch.setattr(
        snapshot.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=json.dumps(_checker_payload()).encode("utf-8")),
    )
    validated = snapshot._default_environment_preflight(snapshot.CANONICAL_REPOSITORY_ROOT)
    assert validated["REAL_EXECUTION_ENVIRONMENT_READY"] is True


def test_environment_pre_gate_failure_publishes_no_receipt(tmp_path, v4_fixture, t0_codes, fresh_codes):
    frame = _build_frame(t0_codes, fresh_codes)
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._execute_source_snapshot_acquisition_with_dependencies(
            authorization_identity=AUTHORIZATION,
            gate_state_root=tmp_path / "gate-state",
            private_state_root=tmp_path / "private-state",
            evidence_output_path=tmp_path / "evidence.json",
            jpx_fetcher=lambda: SYNTHETIC_RAW_BYTES,
            parse_source_table=lambda _raw: frame,
            v4_manifest_path=v4_fixture["manifest_path"],
            v4_universe_csv_path=v4_fixture["universe_csv_path"],
            repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
            public_preflight=lambda: _preflight(),
            gate_consumer=snapshot.consume_gate_once,
            clock=_clock,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(),
            environment_preflight=lambda _root: _environment_preflight(ENVIRONMENT_LOCK_CHECK="FAIL"),
            block_size=BLOCK_SIZE,
            minimum_fresh_eligible_count=MIN_FRESH,
        )
    assert excinfo.value.reason == "V8J_ENVIRONMENT_PRE_GATE_BLOCK"
    assert not (tmp_path / "gate-state").exists() or not any((tmp_path / "gate-state").iterdir())


def test_v8i_authority_and_key_cannot_satisfy_v8j():
    from src import v8i_source_snapshot as v8i

    v8i_authorization = v8i.build_authorization_identity(
        reviewed_v8i_design_candidate_commit=v8i.REVIEWED_V8I_DESIGN_CANDIDATE_COMMIT,
        reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
    )
    assert snapshot.compute_source_snapshot_gate_receipt_key() != v8i.compute_source_snapshot_gate_receipt_key()
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot.validate_authorization_identity(
            v8i_authorization,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
        )
    assert excinfo.value.reason == "V8J_AUTHORIZATION_GRAMMAR_MISMATCH"


def _canonical_callable_binding(*, clock=snapshot._utc_clock):
    from scripts.build_v8_partition_manifest import default_parse_source_table, fetch_real_jpx_source

    return snapshot._require_canonical_post_gate_callable_binding(
        jpx_fetcher=fetch_real_jpx_source,
        parse_source_table=default_parse_source_table,
        clock=clock,
        repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
        reviewed_source_snapshot_support_implementation_sha="db3bff1dea20ef9d5c31c4f9df04a69d3aefb947",
    )


def test_canonical_parser_fetcher_and_clock_provenance_are_accepted_without_network():
    binding = _canonical_callable_binding()
    assert binding == {
        "CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE": "YES"
    }


def test_arbitrary_or_stateful_clock_is_rejected_before_gate():
    def wrapper_clock():
        return snapshot._utc_clock()

    class StatefulClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return snapshot._utc_clock()

    for clock in (lambda: snapshot._utc_clock(), wrapper_clock, StatefulClock()):
        with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
            _canonical_callable_binding(clock=clock)
        assert excinfo.value.reason == "V8J_CANONICAL_CLOCK_BINDING_INVALID"


def test_arbitrary_parser_is_rejected_before_gate():
    from scripts.build_v8_partition_manifest import fetch_real_jpx_source

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._require_canonical_post_gate_callable_binding(
            jpx_fetcher=fetch_real_jpx_source,
            parse_source_table=lambda _raw: None,
            clock=snapshot._utc_clock,
            repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
            reviewed_source_snapshot_support_implementation_sha="db3bff1dea20ef9d5c31c4f9df04a69d3aefb947",
        )
    assert excinfo.value.reason == "V8J_CANONICAL_PARSER_BINDING_INVALID"


def test_arbitrary_or_wrapped_fetcher_is_rejected_before_gate():
    from scripts.build_v8_partition_manifest import default_parse_source_table, fetch_real_jpx_source

    def wrapper():
        return fetch_real_jpx_source()

    for fetcher in (lambda: (b"synthetic", "https://www.jpx.co.jp/"), wrapper):
        with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
            snapshot._require_canonical_post_gate_callable_binding(
                jpx_fetcher=fetcher,
                parse_source_table=default_parse_source_table,
                clock=snapshot._utc_clock,
                repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
                reviewed_source_snapshot_support_implementation_sha="db3bff1dea20ef9d5c31c4f9df04a69d3aefb947",
            )
        assert excinfo.value.reason == "V8J_CANONICAL_FETCHER_BINDING_INVALID"


def test_wrong_canonical_callable_source_provenance_is_rejected(monkeypatch):
    def wrong_source(_callable):
        return str(snapshot.CANONICAL_REPOSITORY_ROOT / "unreviewed.py")

    monkeypatch.setattr(snapshot.inspect, "getsourcefile", wrong_source)
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        _canonical_callable_binding()
    assert excinfo.value.reason == "V8J_CANONICAL_CALLABLE_PROVENANCE_INVALID"


def test_callable_git_provenance_mismatch_is_rejected(monkeypatch):
    original = snapshot.resolve_git_blob
    calls = []

    def mismatched_blob(repository_root, commit, git_path):
        if git_path == snapshot.CANONICAL_JPX_SOURCE_GIT_PATH:
            calls.append((commit, git_path))
            return "a" * 40 if len(calls) == 1 else "b" * 40
        return original(repository_root, commit, git_path)

    monkeypatch.setattr(snapshot, "resolve_git_blob", mismatched_blob)
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        _canonical_callable_binding()
    assert excinfo.value.reason == "V8J_CANONICAL_CALLABLE_GIT_PROVENANCE_MISMATCH"


def test_callable_binding_failure_creates_no_receipt_and_performs_no_fetch(tmp_path, v4_fixture, t0_codes, fresh_codes):
    from scripts.build_v8_partition_manifest import fetch_real_jpx_source

    frame = _build_frame(t0_codes, fresh_codes)
    calls: list[str] = []

    def forbidden_fetch():
        calls.append("fetch")
        return fetch_real_jpx_source()

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._execute_source_snapshot_acquisition_with_dependencies(
            authorization_identity=AUTHORIZATION,
            gate_state_root=tmp_path / "gate-state",
            private_state_root=tmp_path / "private-state",
            evidence_output_path=tmp_path / "evidence.json",
            jpx_fetcher=forbidden_fetch,
            parse_source_table=lambda _raw: frame,
            v4_manifest_path=v4_fixture["manifest_path"],
            v4_universe_csv_path=v4_fixture["universe_csv_path"],
            repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
            public_preflight=lambda: _preflight(),
            gate_consumer=snapshot.consume_gate_once,
            clock=_clock,
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(),
            environment_preflight=lambda _root: _environment_preflight(),
            require_canonical_post_gate_callables=True,
            block_size=BLOCK_SIZE,
            minimum_fresh_eligible_count=MIN_FRESH,
        )
    assert excinfo.value.reason == "V8J_CANONICAL_PARSER_BINDING_INVALID"
    assert calls == []
    assert not (tmp_path / "gate-state").exists() or not any((tmp_path / "gate-state").iterdir())


def test_clock_binding_failure_creates_no_receipt_and_performs_no_fetch(monkeypatch, tmp_path):
    from scripts.build_v8_partition_manifest import default_parse_source_table, fetch_real_jpx_source

    gate_calls: list[str] = []

    def forbidden_gate(*_args, **_kwargs):
        gate_calls.append("gate")
        raise AssertionError("gate must not be consumed")

    monkeypatch.setattr(snapshot, "_validate_public_preflight", lambda _value: None)
    monkeypatch.setattr(
        snapshot,
        "_validate_reviewed_source_snapshot_support_implementation_binding",
        lambda *_args, **_kwargs: "47ff573c7fba26f7a2410436e1bd6b0e049063d6",
    )
    monkeypatch.setattr(snapshot, "_require_frozen_environment_pre_gate", lambda *_args, **_kwargs: None)

    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._execute_source_snapshot_acquisition_with_dependencies(
            authorization_identity=AUTHORIZATION,
            gate_state_root=tmp_path / "gate-state",
            private_state_root=tmp_path / "private-state",
            evidence_output_path=tmp_path / "evidence.json",
            jpx_fetcher=fetch_real_jpx_source,
            parse_source_table=default_parse_source_table,
            v4_manifest_path=tmp_path / "manifest.json",
            v4_universe_csv_path=tmp_path / "universe.csv",
            repository_root=snapshot.CANONICAL_REPOSITORY_ROOT,
            public_preflight=lambda: _preflight(),
            gate_consumer=forbidden_gate,
            clock=lambda: snapshot._utc_clock(),
            reviewed_source_snapshot_support_implementation_sha=REVIEWED_IMPL_SHA,
            require_canonical_post_gate_callables=True,
        )
    assert excinfo.value.reason == "V8J_CANONICAL_CLOCK_BINDING_INVALID"
    assert gate_calls == []
    assert not (tmp_path / "gate-state").exists()
    assert not (tmp_path / "evidence.json").exists()


def test_public_entry_forces_callable_binding_without_a_test_bypass(monkeypatch, tmp_path):
    captured = {}

    def capture_boundary(**kwargs):
        captured.update(kwargs)
        return {"result": "not-executed"}

    monkeypatch.setattr(snapshot, "_execute_source_snapshot_acquisition_with_dependencies", capture_boundary)
    result = snapshot.resolve_and_acquire_source_snapshot(
        "opaque",
        reviewed_source_snapshot_support_implementation_sha="1" * 40,
        jpx_fetcher=lambda: b"synthetic",
        parse_source_table=lambda _raw: None,
        v4_manifest_path=tmp_path / "manifest.json",
        v4_universe_csv_path=tmp_path / "universe.csv",
        evidence_output_path=tmp_path / "evidence.json",
    )
    assert result == {"result": "not-executed"}
    assert captured["require_canonical_post_gate_callables"] is True
    assert captured["clock"] is snapshot._utc_clock


def test_public_entry_rejects_stateful_clock_argument_before_any_boundary(monkeypatch, tmp_path):
    boundary_calls: list[object] = []

    def forbidden_boundary(**_kwargs):
        boundary_calls.append("boundary")
        raise AssertionError("public boundary must not be reached")

    class OneThenRaiseClock:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("post-gate clock failure")
            return _clock()

    clock = OneThenRaiseClock()
    monkeypatch.setattr(snapshot, "_execute_source_snapshot_acquisition_with_dependencies", forbidden_boundary)
    with pytest.raises(TypeError):
        snapshot.resolve_and_acquire_source_snapshot(
            "opaque",
            reviewed_source_snapshot_support_implementation_sha="1" * 40,
            jpx_fetcher=lambda: b"synthetic",
            parse_source_table=lambda _raw: None,
            v4_manifest_path=tmp_path / "manifest.json",
            v4_universe_csv_path=tmp_path / "universe.csv",
            evidence_output_path=tmp_path / "evidence.json",
            clock=clock,
        )
    assert boundary_calls == []
    assert clock.calls == 0


def test_private_synthetic_di_retains_clock_injection_only_without_canonical_enforcement(
    tmp_path, v4_fixture, t0_codes, fresh_codes
):
    result, fetch_calls = _execute(tmp_path, v4_fixture, _build_frame(t0_codes, fresh_codes))
    assert result["result"] == "PASS"
    assert fetch_calls == [1]


def test_dependency_yes_is_not_an_environment_preflight_value():
    assert "CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE" not in (
        snapshot._ENVIRONMENT_PREFLIGHT_FIELDS
    )
    assert _canonical_callable_binding()[
        "CAN_EVERY_REACHABLE_POST_GATE_SOFTWARE_DEPENDENCY_BE_PROVEN_READY_PRE_GATE"
    ] == "YES"


def test_module_grants_no_partition_generation_or_membership_disclosure_authority(v4_fixture, t0_codes, fresh_codes):
    evidence = _sample_evidence(v4_fixture, t0_codes, fresh_codes)
    assert evidence["partition_generation_authorized"] is False
    assert evidence["membership_disclosure_authorized"] is False
    assert evidence["research_opened"] is False
    assert evidence["historical_price_raw_acquisition_performed"] is False


# ---------------------------------------------------------------------------
# Category 22 -- point-of-use Git/design/freeze/implementation binding checks
# ---------------------------------------------------------------------------


def test_reviewed_implementation_binding_rejects_wrong_head(tmp_path):
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_reviewed_source_snapshot_support_implementation_binding(
            snapshot.CANONICAL_REPOSITORY_ROOT,
            REVIEWED_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(head=OTHER_IMPL_SHA),
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_RUNTIME_BINDING_INVALID"


def test_reviewed_implementation_binding_rejects_dirty_worktree():
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_reviewed_source_snapshot_support_implementation_binding(
            snapshot.CANONICAL_REPOSITORY_ROOT,
            REVIEWED_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(worktree_clean=False),
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_RUNTIME_BINDING_INVALID"


def test_reviewed_implementation_binding_rejects_commits_after(tmp_path):
    with pytest.raises(snapshot.V8JSourceSnapshotBlocked) as excinfo:
        snapshot._validate_reviewed_source_snapshot_support_implementation_binding(
            snapshot.CANONICAL_REPOSITORY_ROOT,
            REVIEWED_IMPL_SHA,
            runtime_state_reader=lambda *_args: _runtime_state(commits_after_reviewed_implementation_sha=1),
        )
    assert excinfo.value.reason == "V8J_SOURCE_SNAPSHOT_IMPLEMENTATION_RUNTIME_BINDING_INVALID"


def test_reviewed_implementation_binding_accepts_matching_runtime():
    accepted = snapshot._validate_reviewed_source_snapshot_support_implementation_binding(
        snapshot.CANONICAL_REPOSITORY_ROOT,
        REVIEWED_IMPL_SHA,
        runtime_state_reader=lambda *_args: _runtime_state(),
    )
    assert accepted == REVIEWED_IMPL_SHA
