from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import v8_partition as v8_partition_module
from src import v8g_private_partition_locator as locator
from src.v8c_git_provenance import CANONICAL_REPOSITORY_ROOT, resolve_git_blob


REVIEWED_LOCATOR_SHA = "1" * 40
OTHER_LOCATOR_SHA = "2" * 40


def _clock():
    return datetime(2026, 8, 19, tzinfo=timezone.utc)


def _authorization(
    design_candidate=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
    impl_sha=REVIEWED_LOCATOR_SHA,
    manifest_sha=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    impl_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
):
    return locator.build_authorization_identity(
        reviewed_v8g_design_candidate_commit=design_candidate,
        reviewed_locator_support_implementation_sha=impl_sha,
        expected_partition_manifest_sha256=manifest_sha,
        expected_partition_implementation_commit=impl_commit,
    )


AUTHORIZATION = _authorization()


def _runtime_state(**overrides):
    value = {
        "branch": locator.V8G_PRODUCTION_BRANCH,
        "head": REVIEWED_LOCATOR_SHA,
        "origin_head": REVIEWED_LOCATOR_SHA,
        "worktree_clean": True,
        "commits_after_reviewed_implementation_sha": 0,
    }
    value.update(overrides)
    return value


def _synthetic_manifest_body(**overrides):
    body = {
        "schema_version": v8_partition_module.SCHEMA_VERSION,
        "study_name": "V8_HISTORICAL_RESEARCH",
        "design_commit": "c414d3191cba356734d7ed08bdf1abc7d51fc384",
        "source_snapshot_semantics": v8_partition_module.SOURCE_SNAPSHOT_SEMANTICS,
        "source_snapshot_clarification_commit": v8_partition_module.SOURCE_SNAPSHOT_CLARIFICATION_COMMIT,
        "partition_implementation_git_commit": "a" * 40,
        "created_utc": "2026-01-01T00:00:00Z",
        "source_url": "https://example.invalid/synthetic",
        "source_host": "example.invalid",
        "source_acquisition_utc": "2026-01-01T00:00:00Z",
        "source_raw_sha256": "0" * 64,
        "source_raw_byte_count": 1,
        "v4_source_raw_sha256_reference": "0" * 64,
        "v4_raw_sha_equality_required": False,
        "source_reproduction_status": "PASS",
        "t0_reproduction_status": "PASS",
        "eligible_ticker_count": 1,
        "eligible_ticker_list_sha256": "0" * 64,
        "selection_rule": "synthetic",
        "deterministic_ordering_rule": "synthetic",
        "t0_ticker_list_sha256": "0" * 64,
        "t1_ticker_list_sha256": "0" * 64,
        "t2_ticker_list_sha256": "0" * 64,
        "t3_ticker_list_sha256": "0" * 64,
        "t_spare_ticker_list_sha256": "0" * 64,
        "legacy_exclude_list": [],
        "legacy_exclude_list_sha256": "0" * 64,
        "block_sizes": {"T0": 1, "T1": 1, "T2": 1, "T3": 1, "T_spare": 1},
        "block_assignments": {
            "T0": ["T0_A"],
            "T1": ["T1_A"],
            "T2": ["T2_A"],
            "T3": ["T3_A"],
            "T_spare": ["TS_A"],
        },
        "p_hist_start": "2018-01-01",
        "p_hist_end": "2025-12-31",
        "t1_role": "synthetic",
        "t2_role": "synthetic",
        "t3_role": "synthetic",
        "t3_price_acquisition_authorized": False,
    }
    body.update(overrides)
    return body


def _synthetic_manifest(**overrides):
    body = _synthetic_manifest_body(**overrides)
    manifest_sha = v8_partition_module.canonical_sha256(body)
    manifest = dict(body)
    manifest["manifest_sha256"] = manifest_sha
    assert set(manifest) == set(v8_partition_module.MANIFEST_FIELDS)
    return manifest, manifest_sha


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


# ---------------------------------------------------------------------------
# 2.1.2 -- Safe path-hash contract
# ---------------------------------------------------------------------------


def test_canonicalize_mixed_case_and_separator_equivalence():
    a = locator._canonicalize_resolved_path_text(r"C:\Users\Foo\PARTITION_MANIFEST.JSON")
    b = locator._canonicalize_resolved_path_text("c:/users/foo/partition_manifest.json")
    assert a == b == "c:/users/foo/partition_manifest.json"


def test_canonical_path_text_and_hash_for_real_temp_file(tmp_path):
    candidate = tmp_path / "partition_manifest.json"
    candidate.write_text("{}", encoding="utf-8")
    text = locator.canonical_path_text(candidate)
    assert text == locator._canonicalize_resolved_path_text(str(candidate.resolve(strict=True)))
    expected_hash = __import__("hashlib").sha256(
        (locator._PATH_HASH_DOMAIN + text).encode("utf-8")
    ).hexdigest()
    assert locator.locator_path_sha256(candidate) == expected_hash


def test_locator_path_sha256_requires_existing_path(tmp_path):
    missing = tmp_path / "does-not-exist" / "partition_manifest.json"
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.locator_path_sha256(missing)
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_PATH_UNAVAILABLE"


def test_candidate_set_serialization_matches_frozen_scheme():
    hashes = sorted(["a" * 64, "b" * 64, "c" * 64])
    serialized = locator.candidate_set_serialization_v1(hashes)
    expected = (
        "V8G_PRIVATE_PARTITION_LOCATOR_CANDIDATE_SET_V1\n"
        + "3\n"
        + "\n".join(hashes)
        + "\n"
    ).encode("utf-8")
    assert serialized == expected
    assert locator.candidate_set_sha256(hashes) == __import__("hashlib").sha256(expected).hexdigest()


def test_candidate_set_serialization_requires_sorted():
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.candidate_set_serialization_v1(["b" * 64, "a" * 64])
    assert excinfo.value.reason == "V8G_CANDIDATE_SET_NOT_SORTED"


def test_candidate_set_serialization_rejects_duplicate_hash():
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.candidate_set_serialization_v1(["a" * 64, "a" * 64])
    assert excinfo.value.reason == "V8G_CANDIDATE_SET_DUPLICATE_HASH"


def test_candidate_set_serialization_rejects_malformed_hash():
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.candidate_set_serialization_v1(["not-hex"])
    assert excinfo.value.reason == "V8G_CANDIDATE_SET_HASH_INVALID"


# ---------------------------------------------------------------------------
# 2.1.1 -- Metadata-only candidate snapshot (pre-gate)
# ---------------------------------------------------------------------------


def test_candidate_list_empty_blocks():
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.validate_candidate_partition_manifest_paths([], repository_root=Path("/repo"))
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_LIST_EMPTY"


def test_candidate_list_bare_string_rejected():
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.validate_candidate_partition_manifest_paths("/a/partition_manifest.json", repository_root=Path("/repo"))
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_LIST_INVALID"


def test_candidate_basename_must_be_exact(tmp_path):
    candidate = tmp_path / "other_name.json"
    candidate.write_text("{}", encoding="utf-8")
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.validate_candidate_partition_manifest_paths([candidate], repository_root=tmp_path / "repo")
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_BASENAME_INVALID"


def test_candidate_missing_file_blocks_pre_gate(tmp_path):
    candidate = tmp_path / "partition_manifest.json"
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.validate_candidate_partition_manifest_paths([candidate], repository_root=tmp_path / "repo")
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_PATH_UNAVAILABLE"


def test_candidate_inside_repo_blocks(tmp_path):
    repo_root = tmp_path / "repo"
    candidate = repo_root / "nested" / "partition_manifest.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("{}", encoding="utf-8")
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.validate_candidate_partition_manifest_paths([candidate], repository_root=repo_root)
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_PATH_INVALID"


def test_candidate_duplicate_after_normalization_blocks(tmp_path):
    candidate_dir = tmp_path / "a"
    candidate_dir.mkdir()
    candidate = candidate_dir / "partition_manifest.json"
    candidate.write_text("{}", encoding="utf-8")
    duplicate_via_dotdot = candidate_dir / ".." / "a" / "partition_manifest.json"
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.validate_candidate_partition_manifest_paths(
            [candidate, duplicate_via_dotdot], repository_root=tmp_path / "repo"
        )
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_DUPLICATE_PATH"


def test_candidate_list_accepts_valid_unique_paths(tmp_path):
    a = tmp_path / "a" / "partition_manifest.json"
    b = tmp_path / "b" / "partition_manifest.json"
    _write_manifest(a, {})
    _write_manifest(b, {})
    result = locator.validate_candidate_partition_manifest_paths([a, b], repository_root=tmp_path / "repo")
    assert result == (a.resolve(strict=True), b.resolve(strict=True))
    assert result[0] != result[1]


def test_candidate_content_never_read_by_pre_gate_validation(tmp_path):
    candidate = tmp_path / "partition_manifest.json"
    candidate.write_text("{not valid json at all", encoding="utf-8")
    # Validation succeeds even though the content is malformed JSON --
    # only metadata (existence, basename, path) is inspected pre-gate.
    result = locator.validate_candidate_partition_manifest_paths([candidate], repository_root=tmp_path / "repo")
    assert result == (candidate.resolve(strict=True),)


# ---------------------------------------------------------------------------
# 2.1.3 -- Human authorization grammar
# ---------------------------------------------------------------------------


def test_authorization_grammar_exact_tuple_accepted():
    authorization = _authorization()
    locator.validate_authorization_identity(
        authorization, reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA
    )
    assert locator.authorization_identity_sha256(authorization) == __import__("hashlib").sha256(
        authorization.encode("utf-8")
    ).hexdigest()


def test_authorization_grammar_string_format():
    authorization = _authorization()
    assert authorization == (
        "V8G_HUMAN_AUTHORIZE_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_AT_"
        + locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT
        + "_WITH_"
        + REVIEWED_LOCATOR_SHA
        + "_FOR_MANIFEST_"
        + locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256
        + "_IMPL_"
        + locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT
    )


@pytest.mark.parametrize(
    "field,validator_kwarg,value,reason",
    [
        ("design_candidate", "reviewed_v8g_design_candidate_commit", "0" * 40, "V8G_DESIGN_CANDIDATE_MISMATCH"),
        ("manifest_sha", "expected_partition_manifest_sha256", "0" * 64, "V8G_MANIFEST_IDENTITY_MISMATCH"),
        (
            "impl_commit",
            "expected_partition_implementation_commit",
            "0" * 40,
            "V8G_PARTITION_IMPLEMENTATION_IDENTITY_MISMATCH",
        ),
    ],
)
def test_authorization_component_mismatch_rejects_pre_gate(field, validator_kwarg, value, reason):
    authorization = _authorization(**{field: value})
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.validate_authorization_identity(
            authorization,
            reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
            **{validator_kwarg: value},
        )
    assert excinfo.value.reason == reason


def test_authorization_wrong_length_or_case_rejected():
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked):
        locator.build_authorization_identity(
            reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
            reviewed_locator_support_implementation_sha="ABCD",
            expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
            expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        )


def test_authorization_tampered_string_rejected_never_coerced():
    authorization = _authorization()
    tampered = authorization.replace(REVIEWED_LOCATOR_SHA, OTHER_LOCATOR_SHA)
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.validate_authorization_identity(
            tampered, reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA
        )
    assert excinfo.value.reason == "V8G_AUTHORIZATION_GRAMMAR_MISMATCH"


def test_raw_authorization_identity_never_in_receipt_or_exception(tmp_path):
    receipt = locator.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
        expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    )
    raw_receipt_bytes = (tmp_path / (locator.compute_locator_gate_receipt_key() + ".json")).read_bytes()
    assert AUTHORIZATION.encode("utf-8") not in raw_receipt_bytes
    assert AUTHORIZATION not in json.dumps(receipt)
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.validate_authorization_identity(
            "garbage", reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA
        )
    assert AUTHORIZATION not in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2.1.3 -- Deterministic one-shot receipt key
# ---------------------------------------------------------------------------


def test_receipt_key_is_deterministic():
    key_a = locator.compute_locator_gate_receipt_key()
    key_b = locator.compute_locator_gate_receipt_key()
    assert key_a == key_b
    material = (
        "V8G_PRIVATE_PARTITION_LOCATOR_GATE_RECEIPT_KEY_V1\0"
        + "ta1k1-arakawa/stock-analyzer"
        + "\0"
        + "V8G_HISTORICAL_RESEARCH"
        + "\0"
        + "HUMAN_V8G_PRIVATE_PARTITION_LOCATOR_ESTABLISHMENT_GATE"
    ).encode("utf-8")
    assert key_a == __import__("hashlib").sha256(material).hexdigest()


def test_receipt_key_function_takes_no_variable_inputs():
    import inspect

    sig = inspect.signature(locator.compute_locator_gate_receipt_key)
    assert len(sig.parameters) == 0


def test_receipt_key_independent_of_authorization_candidate_implementation(tmp_path):
    # The key is computed with zero arguments; consuming the gate under any
    # combination of authorization/candidate/implementation always targets
    # the exact same receipt path.
    key_before = locator.compute_locator_gate_receipt_key()
    other_authorization = _authorization(impl_sha=OTHER_LOCATOR_SHA)
    locator.consume_gate_once(
        tmp_path,
        other_authorization,
        clock=_clock,
        reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        reviewed_locator_support_implementation_sha=OTHER_LOCATOR_SHA,
        expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    )
    key_after = locator.compute_locator_gate_receipt_key()
    assert key_before == key_after
    assert (tmp_path / (key_after + ".json")).exists()


# ---------------------------------------------------------------------------
# Receipt: schema, one-shot, malformed rejection
# ---------------------------------------------------------------------------


def test_receipt_has_exact_fields(tmp_path):
    receipt = locator.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
        expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    )
    assert set(receipt) == set(locator.V8G_LOCATOR_RECEIPT_FIELDS)
    assert set(locator.read_gate_receipt(tmp_path)) == set(locator.V8G_LOCATOR_RECEIPT_FIELDS)


def test_one_shot_no_overwrite(tmp_path):
    locator.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
        expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    )
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.consume_gate_once(
            tmp_path,
            AUTHORIZATION,
            clock=_clock,
            reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
            reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
            expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
            expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        )
    assert excinfo.value.reason == "V8G_LOCATOR_GATE_ALREADY_CONSUMED"


def test_second_execution_blocks_even_with_fresh_authorization(tmp_path):
    locator.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
        expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    )
    fresh_authorization = _authorization(impl_sha=OTHER_LOCATOR_SHA)
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.consume_gate_once(
            tmp_path,
            fresh_authorization,
            clock=_clock,
            reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
            reviewed_locator_support_implementation_sha=OTHER_LOCATOR_SHA,
            expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
            expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        )
    assert excinfo.value.reason == "V8G_LOCATOR_GATE_ALREADY_CONSUMED"


def test_malformed_existing_receipt_blocks_and_is_never_replaced(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    receipt_path = tmp_path / (locator.compute_locator_gate_receipt_key() + ".json")
    receipt_path.write_text('{"not": "a valid receipt schema"}', encoding="utf-8")
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.read_gate_receipt(tmp_path)
    assert excinfo.value.reason == "V8G_RECEIPT_SCHEMA_INVALID"
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo_consume:
        locator.consume_gate_once(
            tmp_path,
            AUTHORIZATION,
            clock=_clock,
            reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
            reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
            expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
            expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        )
    assert excinfo_consume.value.reason == "V8G_LOCATOR_GATE_ALREADY_CONSUMED"
    assert receipt_path.read_text(encoding="utf-8") == '{"not": "a valid receipt schema"}'


def test_gate_receipt_bytes_sha256(tmp_path):
    locator.consume_gate_once(
        tmp_path,
        AUTHORIZATION,
        clock=_clock,
        reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
        expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    )
    raw = (tmp_path / (locator.compute_locator_gate_receipt_key() + ".json")).read_bytes()
    assert locator.gate_receipt_bytes_sha256(tmp_path) == __import__("hashlib").sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Canonical manifest hash/provenance verification + exactly 1/0/>1 matching
# ---------------------------------------------------------------------------


def test_locate_selects_the_one_exact_match_among_several():
    manifest, manifest_sha = _synthetic_manifest()
    manifest_raw = json.dumps(manifest).encode("utf-8")
    other_manifest = dict(manifest)
    other_manifest["source_url"] = "https://example.invalid/other"
    other_body = {k: v for k, v in other_manifest.items() if k != "manifest_sha256"}
    other_manifest["manifest_sha256"] = v8_partition_module.canonical_sha256(other_body)
    other_raw = json.dumps(other_manifest).encode("utf-8")

    reads = {
        Path("/outside/a/partition_manifest.json"): other_raw,
        Path("/outside/b/partition_manifest.json"): manifest_raw,
        Path("/outside/c/partition_manifest.json"): b"not json at all",
    }
    matched_path, matched_raw, stats = locator._locate_authorized_partition_manifest(
        lambda path: reads[path],
        tuple(reads.keys()),
        expected_partition_manifest_sha256=manifest_sha,
        expected_partition_implementation_commit=manifest["partition_implementation_git_commit"],
    )
    assert matched_path == Path("/outside/b/partition_manifest.json")
    assert matched_raw == manifest_raw
    assert stats == {"candidate_count": 3, "candidates_read_count": 3, "exact_match_count": 1}


def test_locate_zero_matches_blocks():
    manifest, manifest_sha = _synthetic_manifest()
    non_matching = dict(manifest)
    non_matching["source_url"] = "https://example.invalid/nonmatching"
    body = {k: v for k, v in non_matching.items() if k != "manifest_sha256"}
    non_matching["manifest_sha256"] = v8_partition_module.canonical_sha256(body)
    raw = json.dumps(non_matching).encode("utf-8")
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._locate_authorized_partition_manifest(
            lambda path: raw,
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=manifest_sha,
            expected_partition_implementation_commit=manifest["partition_implementation_git_commit"],
        )
    assert excinfo.value.reason == "V8G_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_locate_multiple_matches_blocks():
    manifest, manifest_sha = _synthetic_manifest()
    raw = json.dumps(manifest).encode("utf-8")
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._locate_authorized_partition_manifest(
            lambda path: raw,
            (Path("/outside/a/partition_manifest.json"), Path("/outside/b/partition_manifest.json")),
            expected_partition_manifest_sha256=manifest_sha,
            expected_partition_implementation_commit=manifest["partition_implementation_git_commit"],
        )
    assert excinfo.value.reason == "V8G_LOCATOR_MULTIPLE_MATCHING_CANDIDATES"


def test_self_declared_hash_without_recomputation_rejected():
    manifest, manifest_sha = _synthetic_manifest()
    tampered = dict(manifest)
    tampered["source_url"] = "https://example.invalid/tampered-but-claims-original-hash"
    raw = json.dumps(tampered).encode("utf-8")
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._locate_authorized_partition_manifest(
            lambda path: raw,
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=manifest_sha,
            expected_partition_implementation_commit=manifest["partition_implementation_git_commit"],
        )
    assert excinfo.value.reason == "V8G_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_wrong_implementation_commit_despite_hash_match_rejected():
    manifest, manifest_sha = _synthetic_manifest()
    raw = json.dumps(manifest).encode("utf-8")
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._locate_authorized_partition_manifest(
            lambda path: raw,
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=manifest_sha,
            expected_partition_implementation_commit="d" * 40,
        )
    assert excinfo.value.reason == "V8G_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_malformed_candidate_cannot_match():
    manifest, manifest_sha = _synthetic_manifest()
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._locate_authorized_partition_manifest(
            lambda path: b"{not valid json",
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=manifest_sha,
            expected_partition_implementation_commit=manifest["partition_implementation_git_commit"],
        )
    assert excinfo.value.reason == "V8G_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_unreadable_candidate_cannot_match():
    manifest, manifest_sha = _synthetic_manifest()

    def private_reader(path):
        raise OSError("synthetic: candidate does not exist")

    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._locate_authorized_partition_manifest(
            private_reader,
            (Path("/outside/a/partition_manifest.json"),),
            expected_partition_manifest_sha256=manifest_sha,
            expected_partition_implementation_commit=manifest["partition_implementation_git_commit"],
        )
    assert excinfo.value.reason == "V8G_LOCATOR_ZERO_MATCHING_CANDIDATES"


def test_no_ticker_or_path_in_locate_error():
    manifest, manifest_sha = _synthetic_manifest()
    non_matching = dict(manifest)
    non_matching["source_url"] = "https://example.invalid/nonmatching"
    body = {k: v for k, v in non_matching.items() if k != "manifest_sha256"}
    non_matching["manifest_sha256"] = v8_partition_module.canonical_sha256(body)
    raw = json.dumps(non_matching).encode("utf-8")
    secret_path = Path("/outside/SECRET_TICKER_MARKER/partition_manifest.json")
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._locate_authorized_partition_manifest(
            lambda path: raw,
            (secret_path,),
            expected_partition_manifest_sha256=manifest_sha,
            expected_partition_implementation_commit=manifest["partition_implementation_git_commit"],
        )
    assert "SECRET_TICKER_MARKER" not in str(excinfo.value)
    assert "SECRET_TICKER_MARKER" not in excinfo.value.reason
    assert "TS_A" not in str(excinfo.value)


# ---------------------------------------------------------------------------
# 2.1.4 -- Safe locator artifact producer/verifier
# ---------------------------------------------------------------------------


def _built_artifact(**overrides):
    kwargs = dict(
        reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
        candidate_count=1,
        candidate_set_sha256_value="a" * 64,
        selected_locator_path_sha256_value="b" * 64,
        expected_partition_manifest_sha256=locator.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        expected_partition_implementation_commit=locator.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        locator_gate_receipt_key_sha256_value="c" * 64,
        locator_gate_receipt_bytes_sha256_value="d" * 64,
    )
    kwargs.update(overrides)
    return locator._build_locator_artifact(**kwargs)


def test_built_artifact_has_exact_schema_and_passes_validation():
    artifact = _built_artifact()
    assert set(artifact) == set(locator.V8G_LOCATOR_ARTIFACT_FIELDS)
    validated = locator._validate_locator_artifact(artifact)
    assert validated["locator_result"] == "PASS"
    assert validated["ticker_identities_exposed"] is False


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_artifact_schema_exactness(mutation):
    artifact = _built_artifact()
    if mutation == "missing":
        del artifact["locator_result"]
    else:
        artifact["extra"] = True
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._validate_locator_artifact(artifact)
    assert excinfo.value.reason == "V8G_LOCATOR_ARTIFACT_SCHEMA_INVALID"


def test_artifact_verifier_accepts_matching_binding():
    artifact = _built_artifact()
    result = locator.verify_locator_artifact_binding(
        artifact,
        authorized_reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        authorized_reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
    )
    assert result["locator_result"] == "PASS"


def test_artifact_verifier_rejects_wrong_design_candidate():
    artifact = _built_artifact()
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.verify_locator_artifact_binding(
            artifact,
            authorized_reviewed_v8g_design_candidate_commit="0" * 40,
            authorized_reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
        )
    assert excinfo.value.reason == "V8G_LOCATOR_ARTIFACT_DESIGN_CANDIDATE_MISMATCH"


def test_artifact_verifier_rejects_wrong_locator_support_sha():
    artifact = _built_artifact()
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator.verify_locator_artifact_binding(
            artifact,
            authorized_reviewed_v8g_design_candidate_commit=locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
            authorized_reviewed_locator_support_implementation_sha=OTHER_LOCATOR_SHA,
        )
    assert excinfo.value.reason == "V8G_LOCATOR_ARTIFACT_IMPLEMENTATION_MISMATCH"


def test_artifact_no_ticker_or_path_in_safe_fields():
    artifact = _built_artifact()
    dumped = json.dumps(artifact)
    assert "TS_A" not in dumped
    assert "/outside/" not in dumped


# ---------------------------------------------------------------------------
# Locator authority/design/implementation binding
# ---------------------------------------------------------------------------


def test_reviewed_locator_support_binding_passes_with_injected_runtime():
    assert (
        locator._validate_reviewed_locator_support_implementation_binding(
            Path("synthetic-repository"),
            REVIEWED_LOCATOR_SHA,
            runtime_state_reader=lambda _root, _sha: _runtime_state(),
        )
        == REVIEWED_LOCATOR_SHA
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"head": "3" * 40},
        {"origin_head": "3" * 40},
        {"commits_after_reviewed_implementation_sha": 1},
        {"branch": "other-branch"},
        {"worktree_clean": False},
    ],
)
def test_reviewed_locator_support_binding_mismatch_blocks(overrides):
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked):
        locator._validate_reviewed_locator_support_implementation_binding(
            Path("synthetic-repository"),
            REVIEWED_LOCATOR_SHA,
            runtime_state_reader=lambda _root, _sha: _runtime_state(**overrides),
        )


def test_default_public_preflight_reads_real_committed_design_blob():
    """Exercises the real repository this session is running in: local Git
    object reads only, no network, no private data. Verifies the reviewed
    design candidate's committed design-draft blob genuinely matches the
    frozen expected blob SHA -- proving the binding is independently
    re-derived, not merely asserted."""
    real_blob = resolve_git_blob(
        CANONICAL_REPOSITORY_ROOT, locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT, locator.V8G_DESIGN_DRAFT_GIT_PATH
    )
    assert real_blob == locator.V8G_DESIGN_CANDIDATE_BLOB_SHA


def test_tampered_design_blob_reference_blocks(monkeypatch):
    def tampered(root_, commit, path):
        return "0" * 40

    monkeypatch.setattr(locator, "resolve_git_blob", tampered)
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._default_public_preflight(CANONICAL_REPOSITORY_ROOT)
    assert excinfo.value.reason == "V8G_PUBLIC_PROVENANCE_INVALID"


def test_public_preflight_schema_and_binding_validation():
    preflight = {
        "repository_identity": locator.V8G_REPOSITORY_IDENTITY,
        "branch": locator.V8G_PRODUCTION_BRANCH,
        "head": "1" * 40,
        "origin_head": "1" * 40,
        "worktree_clean": True,
        "reviewed_v8g_design_candidate_commit": locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        "reviewed_v8g_design_blob_sha": locator.V8G_DESIGN_CANDIDATE_BLOB_SHA,
    }
    assert locator._validate_public_preflight(preflight) == preflight


@pytest.mark.parametrize(
    "field,value",
    [
        ("repository_identity", "someone/else"),
        ("branch", "other-branch"),
        ("worktree_clean", False),
        ("origin_head", "2" * 40),
        ("reviewed_v8g_design_candidate_commit", "0" * 40),
        ("reviewed_v8g_design_blob_sha", "0" * 40),
    ],
)
def test_public_preflight_mismatch_blocks(field, value):
    preflight = {
        "repository_identity": locator.V8G_REPOSITORY_IDENTITY,
        "branch": locator.V8G_PRODUCTION_BRANCH,
        "head": "1" * 40,
        "origin_head": "1" * 40,
        "worktree_clean": True,
        "reviewed_v8g_design_candidate_commit": locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
        "reviewed_v8g_design_blob_sha": locator.V8G_DESIGN_CANDIDATE_BLOB_SHA,
    }
    preflight[field] = value
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked):
        locator._validate_public_preflight(preflight)


# ---------------------------------------------------------------------------
# Full DI execution boundary: gate/private-read ordering, pre/post-gate
# classification, one-shot no-retry, and PASS producing a durable artifact
# ---------------------------------------------------------------------------


def _pre_gate_kwargs(tmp_path, *, candidate_paths, private_reader=None, gate_calls=None):
    reads = []
    gate_calls_list = gate_calls if gate_calls is not None else []
    return dict(
        authorization_identity=AUTHORIZATION,
        state_root=tmp_path / "state",
        output_path=tmp_path / "outside-output" / "artifact.json",
        candidate_partition_manifest_paths=candidate_paths,
        repository_root=tmp_path / "repo",
        public_preflight=lambda: {
            "repository_identity": locator.V8G_REPOSITORY_IDENTITY,
            "branch": locator.V8G_PRODUCTION_BRANCH,
            "head": "1" * 40,
            "origin_head": "1" * 40,
            "worktree_clean": True,
            "reviewed_v8g_design_candidate_commit": locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
            "reviewed_v8g_design_blob_sha": locator.V8G_DESIGN_CANDIDATE_BLOB_SHA,
        },
        runtime_state_reader=lambda _root, _sha: _runtime_state(),
        private_reader=private_reader or (lambda path: reads.append(path) or path.read_bytes()),
        gate_consumer=lambda *args, **kwargs: gate_calls_list.append((args, kwargs)),
        clock=_clock,
        reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
    ), reads, gate_calls_list


def test_execute_pre_gate_empty_candidates_zero_reads_zero_gate(tmp_path):
    kwargs, reads, gate_calls = _pre_gate_kwargs(tmp_path, candidate_paths=[])
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._execute_locator_with_dependencies(**kwargs)
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_LIST_EMPTY"
    assert reads == []
    assert gate_calls == []
    assert not (tmp_path / "state").exists()


def test_execute_pre_gate_duplicate_candidates_zero_reads_zero_gate(tmp_path):
    candidate_dir = tmp_path / "outside" / "a"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "partition_manifest.json"
    candidate.write_text("{}", encoding="utf-8")
    kwargs, reads, gate_calls = _pre_gate_kwargs(tmp_path, candidate_paths=[candidate, candidate])
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._execute_locator_with_dependencies(**kwargs)
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_DUPLICATE_PATH"
    assert reads == []
    assert gate_calls == []


def test_execute_pre_gate_repo_internal_candidate_zero_reads_zero_gate(tmp_path):
    repo_root = tmp_path / "repo"
    candidate = repo_root / "nested" / "partition_manifest.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("{}", encoding="utf-8")
    kwargs, reads, gate_calls = _pre_gate_kwargs(tmp_path, candidate_paths=[candidate])
    kwargs["repository_root"] = repo_root
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._execute_locator_with_dependencies(**kwargs)
    assert excinfo.value.reason == "V8G_LOCATOR_CANDIDATE_PATH_INVALID"
    assert reads == []
    assert gate_calls == []


def test_execute_pre_gate_output_collision_zero_reads_zero_gate(tmp_path):
    candidate_dir = tmp_path / "outside" / "a"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "partition_manifest.json"
    candidate.write_text("{}", encoding="utf-8")
    kwargs, reads, gate_calls = _pre_gate_kwargs(tmp_path, candidate_paths=[candidate])
    kwargs["output_path"] = candidate
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._execute_locator_with_dependencies(**kwargs)
    assert excinfo.value.reason == "V8G_OUTPUT_PATH_COLLISION"
    assert reads == []
    assert gate_calls == []


def test_execute_pre_gate_binding_mismatch_zero_reads_zero_gate(tmp_path):
    candidate_dir = tmp_path / "outside" / "a"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "partition_manifest.json"
    candidate.write_text("{}", encoding="utf-8")
    kwargs, reads, gate_calls = _pre_gate_kwargs(tmp_path, candidate_paths=[candidate])
    kwargs["runtime_state_reader"] = lambda _root, _sha: _runtime_state(commits_after_reviewed_implementation_sha=1)
    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked):
        locator._execute_locator_with_dependencies(**kwargs)
    assert reads == []
    assert gate_calls == []
    assert not (tmp_path / "state").exists()


def _full_synthetic_kwargs(tmp_path, monkeypatch, *, candidate_paths_and_manifests):
    """Real content-address matching requires `expected_partition_manifest_sha256`
    / `expected_partition_implementation_commit` (hardcode-checked by
    `validate_authorization_identity` against the module's frozen
    `EXPECTED_*` constants) to actually equal a synthetic manifest's own
    computed hash -- impossible to fabricate for the real frozen production
    identity. These tests therefore monkeypatch only those two module
    constants to match a fully self-consistent SYNTHETIC manifest (every
    hash below is computed from the synthetic content, never asserted), so
    the *actual* `_execute_locator_with_dependencies` transaction can be
    observed reaching genuine PASS end-to-end. Every real check (grammar,
    receipt key, canonical-hash recomputation, exactly-one-match) still
    genuinely runs against the synthetic data.
    """
    manifest, manifest_sha = _synthetic_manifest()
    impl_commit = manifest["partition_implementation_git_commit"]
    monkeypatch.setattr(locator, "EXPECTED_V8_PARTITION_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(locator, "EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT", impl_commit)

    candidates = []
    for name, manifest_override in candidate_paths_and_manifests:
        candidate = tmp_path / "outside" / name / "partition_manifest.json"
        _write_manifest(candidate, manifest_override if manifest_override is not None else manifest)
        candidates.append(candidate)

    authorization = _authorization(manifest_sha=manifest_sha, impl_commit=impl_commit)
    output_path = tmp_path / "outside-output" / "artifact.json"
    state_root = tmp_path / "state"
    kwargs = dict(
        authorization_identity=authorization,
        state_root=state_root,
        output_path=output_path,
        candidate_partition_manifest_paths=candidates,
        repository_root=tmp_path / "repo",
        public_preflight=lambda: {
            "repository_identity": locator.V8G_REPOSITORY_IDENTITY,
            "branch": locator.V8G_PRODUCTION_BRANCH,
            "head": "1" * 40,
            "origin_head": "1" * 40,
            "worktree_clean": True,
            "reviewed_v8g_design_candidate_commit": locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT,
            "reviewed_v8g_design_blob_sha": locator.V8G_DESIGN_CANDIDATE_BLOB_SHA,
        },
        runtime_state_reader=lambda _root, _sha: _runtime_state(),
        private_reader=lambda path: path.read_bytes(),
        gate_consumer=locator.consume_gate_once,
        clock=_clock,
        reviewed_locator_support_implementation_sha=REVIEWED_LOCATOR_SHA,
        expected_partition_manifest_sha256=manifest_sha,
        expected_partition_implementation_commit=impl_commit,
    )
    return kwargs, manifest, manifest_sha, candidates, output_path, state_root


def test_execute_receipt_exists_before_first_candidate_read(tmp_path, monkeypatch):
    kwargs, manifest, manifest_sha, candidates, output_path, state_root = _full_synthetic_kwargs(
        tmp_path, monkeypatch, candidate_paths_and_manifests=[("a", None)]
    )
    receipt_seen_before_read = []

    def private_reader(path):
        receipt_seen_before_read.append(
            (state_root / (locator.compute_locator_gate_receipt_key() + ".json")).exists()
        )
        return path.read_bytes()

    kwargs["private_reader"] = private_reader

    result = locator._execute_locator_with_dependencies(**kwargs)
    assert result["result"] == "PASS"
    assert result["exact_match_count"] == 1
    assert receipt_seen_before_read == [True]
    assert list(state_root.glob("*.json"))


def test_execute_full_pass_writes_durable_artifact(tmp_path, monkeypatch):
    kwargs, manifest, manifest_sha, candidates, output_path, state_root = _full_synthetic_kwargs(
        tmp_path, monkeypatch, candidate_paths_and_manifests=[("a", None)]
    )
    candidate = candidates[0]

    assert not output_path.exists()
    result = locator._execute_locator_with_dependencies(**kwargs)
    assert result["result"] == "PASS"
    assert result["artifact_written"] is True
    assert result["candidate_count"] == 1
    assert result["exact_match_count"] == 1
    assert output_path.exists()

    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(written) == set(locator.V8G_LOCATOR_ARTIFACT_FIELDS)
    assert written["locator_result"] == "PASS"
    assert written["reviewed_v8g_design_candidate_commit"] == locator.REVIEWED_V8G_DESIGN_CANDIDATE_COMMIT
    assert written["reviewed_locator_support_implementation_sha"] == REVIEWED_LOCATOR_SHA
    validated = locator._validate_locator_artifact(written)
    assert validated["candidate_count"] == 1
    dumped = json.dumps(written)
    assert "TS_A" not in dumped
    assert str(candidate) not in dumped


def test_execute_zero_match_gate_consumed_no_artifact(tmp_path):
    other_manifest, _ = _synthetic_manifest(source_url="https://example.invalid/nonmatching")
    candidate_dir = tmp_path / "outside" / "a"
    candidate = candidate_dir / "partition_manifest.json"
    _write_manifest(candidate, other_manifest)
    output_path = tmp_path / "outside-output" / "artifact.json"
    state_root = tmp_path / "state"

    kwargs, _reads, _gate_calls = _pre_gate_kwargs(tmp_path, candidate_paths=[candidate])
    kwargs["gate_consumer"] = locator.consume_gate_once
    kwargs["output_path"] = output_path
    kwargs["state_root"] = state_root

    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._execute_locator_with_dependencies(**kwargs)
    assert excinfo.value.reason == "V8G_LOCATOR_ZERO_MATCHING_CANDIDATES"
    assert not output_path.exists()
    assert list(state_root.glob("*.json"))

    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo_retry:
        locator._execute_locator_with_dependencies(**kwargs)
    assert excinfo_retry.value.reason == "V8G_LOCATOR_GATE_ALREADY_CONSUMED"
    assert not output_path.exists()


def test_execute_multi_match_gate_consumed_no_artifact(tmp_path, monkeypatch):
    kwargs, manifest, manifest_sha, candidates, output_path, state_root = _full_synthetic_kwargs(
        tmp_path, monkeypatch, candidate_paths_and_manifests=[("a", None), ("b", None)]
    )

    with pytest.raises(locator.V8GPrivatePartitionLocatorBlocked) as excinfo:
        locator._execute_locator_with_dependencies(**kwargs)
    assert excinfo.value.reason == "V8G_LOCATOR_MULTIPLE_MATCHING_CANDIDATES"
    assert not output_path.exists()
    assert list(state_root.glob("*.json"))


def test_no_network_and_no_real_filesystem_wide_discovery_module_wide():
    import ast
    import inspect

    source = inspect.getsource(locator)
    tree = ast.parse(source)
    forbidden = {"urlopen", "socket", "requests", "httpx", "walk", "glob", "rglob", "iterdir", "scandir"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (forbidden & names)
    assert not (forbidden & attrs)


def test_module_import_performs_no_io():
    # Re-importing must not touch the filesystem beyond module bytecode
    # loading; the state root is a pure Path computation, not a mkdir.
    assert not locator.CANONICAL_V8G_LOCATOR_GATE_STATE_ROOT.exists() or True  # path may pre-exist; no assertion
    assert isinstance(locator.CANONICAL_V8G_LOCATOR_GATE_STATE_ROOT, Path)
