from __future__ import annotations

import json

import pytest

from src import v8f_t2_prefreeze_preservation as recheck


def _safe(**overrides):
    value = {
        "T2_real_data_acquired": False,
        "T2_opened": False,
        "T2_research_access_count": 0,
        "T2_features_observed": False,
        "T2_outcomes_observed": False,
        "T2_membership_reassigned": False,
        "universe_definition_compatible": True,
        "partition_algorithm_compatible": True,
        "data_quality_policy_unchanged": True,
        "v8_trusted_partition_git_blob": recheck.EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "original_v8_partition_manifest_sha256": recheck.EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "parent_v8_partition_implementation_commit": recheck.EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "t2_count": recheck.EXPECTED_T2_COUNT,
        "t2_ticker_list_sha256": recheck.EXPECTED_T2_TICKER_LIST_SHA256,
    }
    value.update(overrides)
    return value


def _record(**overrides):
    safe = _safe()
    value = {
        "study": recheck.V8F_STUDY_NAME,
        "document_type": recheck.V8F_T2_PREFREEZE_DOCUMENT_TYPE,
        "reviewed_v8f_design_candidate_commit": recheck.V8F_REVIEWED_DESIGN_CANDIDATE_COMMIT,
        "checkpoint": recheck.V8F_T2_PREFREEZE_CHECKPOINT,
        "recheck_1": "before_V8F_design_freeze",
        **safe,
        "T2_PREFREEZE_PRESERVATION_RECHECK": "PASS",
        "OVERALL_RESULT": "PASS",
    }
    value.update(overrides)
    return value


# ---------------------------------------------------------------------------
# Namespace exactness / exact V8F candidate binding
# ---------------------------------------------------------------------------


def test_namespace_literals_are_v8f_exact():
    assert recheck.V8F_STUDY_NAME == "V8F_HISTORICAL_RESEARCH"
    assert recheck.V8F_T2_PREFREEZE_CHECKPOINT == "V8F_T2_PREFREEZE_PRESERVATION_RECHECK"
    assert recheck.V8F_T2_PREFREEZE_DOCUMENT_TYPE == "T2_PREFREEZE_PRESERVATION_RECHECK_AUDIT_RECORD"


def test_wrong_v8f_candidate_blocks():
    record = _record(reviewed_v8f_design_candidate_commit="0" * 40)
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck.verify_t2_prefreeze_record(record, safe_evidence=_safe())
    assert excinfo.value.reason == "V8F_T2_RECORD_VALUE_MISMATCH:reviewed_v8f_design_candidate_commit"


# ---------------------------------------------------------------------------
# T2 exact nine-condition verification
# ---------------------------------------------------------------------------


def test_nine_conditions_pass_with_safe_defaults():
    assert recheck._validate_nine_conditions(
        {key: _safe()[key] for key in recheck.T2_SAFE_CONDITION_FIELDS}
    ) == {key: _safe()[key] for key in recheck.T2_SAFE_CONDITION_FIELDS}


@pytest.mark.parametrize(
    "field,value",
    [
        ("T2_real_data_acquired", True),
        ("T2_opened", True),
        ("T2_research_access_count", 1),
        ("T2_features_observed", True),
        ("T2_outcomes_observed", True),
        ("T2_membership_reassigned", True),
        ("universe_definition_compatible", False),
        ("partition_algorithm_compatible", False),
        ("data_quality_policy_unchanged", False),
    ],
)
def test_forbidden_t2_condition_blocks(field, value):
    conditions = {key: _safe()[key] for key in recheck.T2_SAFE_CONDITION_FIELDS}
    conditions[field] = value
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck._validate_nine_conditions(conditions)
    assert excinfo.value.reason == "V8F_T2_CONDITION_BLOCKED:" + field


@pytest.mark.parametrize(
    "field,value",
    [
        ("T2_real_data_acquired", "false"),
        ("T2_research_access_count", "0"),
        ("universe_definition_compatible", 1),
    ],
)
def test_wrong_condition_type_blocks(field, value):
    conditions = {key: _safe()[key] for key in recheck.T2_SAFE_CONDITION_FIELDS}
    conditions[field] = value
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck._validate_nine_conditions(conditions)
    assert excinfo.value.reason == "V8F_T2_CONDITION_TYPE_INVALID:" + field


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_nine_condition_schema_exactness(mutation):
    conditions = {key: _safe()[key] for key in recheck.T2_SAFE_CONDITION_FIELDS}
    if mutation == "missing":
        del conditions["T2_opened"]
    else:
        conditions["unexpected"] = False
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck._validate_nine_conditions(conditions)
    assert excinfo.value.reason == "V8F_T2_SAFE_CONDITIONS_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# T2 count/hash/provenance mismatch blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,value",
    [
        ("v8_trusted_partition_git_blob", "0" * 40),
        ("original_v8_partition_manifest_sha256", "0" * 64),
        ("parent_v8_partition_implementation_commit", "0" * 40),
        ("t2_count", 299),
        ("t2_ticker_list_sha256", "0" * 64),
    ],
)
def test_t2_provenance_mismatch_blocks(field, value):
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked):
        recheck._validate_safe_evidence(_safe(**{field: value}))


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_safe_evidence_schema_exactness(mutation):
    safe = _safe()
    if mutation == "missing":
        del safe["t2_count"]
    else:
        safe["unexpected"] = True
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck._validate_safe_evidence(safe)
    assert excinfo.value.reason == "V8F_T2_SAFE_EVIDENCE_SCHEMA_INVALID"


# ---------------------------------------------------------------------------
# Record schema exactness and evidence-record cross-check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_record_schema_exactness(mutation):
    record = _record()
    if mutation == "missing":
        del record["OVERALL_RESULT"]
    else:
        record["unexpected"] = True
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck.verify_t2_prefreeze_record(record, safe_evidence=_safe())
    assert excinfo.value.reason == "V8F_T2_RECORD_SCHEMA_INVALID"


def test_record_evidence_mismatch_blocks():
    record = _record(t2_count=299)
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck.verify_t2_prefreeze_record(record, safe_evidence=_safe())
    assert excinfo.value.reason == "V8F_T2_RECORD_EVIDENCE_MISMATCH:t2_count"


def test_verify_t2_prefreeze_record_bytes_matches_mapping_result():
    record = _record()
    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    result = recheck.verify_t2_prefreeze_record_bytes(raw, safe_evidence=_safe())
    assert result["result"] == "PASS"
    assert result["nine_conditions_independently_verified"] is True
    assert result["provenance_independently_verified"] is True


def test_duplicate_json_key_in_record_bytes_blocks():
    raw = b'{"study":"V8F_HISTORICAL_RESEARCH","study":"V8F_HISTORICAL_RESEARCH"}'
    with pytest.raises(recheck.V8FT2PrefreezePreservationBlocked) as excinfo:
        recheck.verify_t2_prefreeze_record_bytes(raw, safe_evidence=_safe())
    assert excinfo.value.reason == "V8F_T2_RECORD_DUPLICATE_KEY"


# ---------------------------------------------------------------------------
# Builder writes nothing; no real network/private read
# ---------------------------------------------------------------------------


def test_builder_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = sorted(tmp_path.rglob("*"))
    record = recheck.build_t2_prefreeze_record(_safe())
    after = sorted(tmp_path.rglob("*"))
    assert before == after
    assert record["OVERALL_RESULT"] == "PASS"
    assert set(record) == set(recheck.V8F_T2_PREFREEZE_RECORD_FIELDS)


def test_builder_result_is_independently_reverifiable():
    record = recheck.build_t2_prefreeze_record(_safe())
    verification = recheck.verify_t2_prefreeze_record(record, safe_evidence=_safe())
    assert verification["result"] == "PASS"


def test_no_network_module_wide():
    import ast
    import inspect

    source = inspect.getsource(recheck)
    tree = ast.parse(source)
    forbidden = {"urlopen", "socket", "requests", "httpx", "subprocess"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attrs = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (forbidden & names)
    assert not (forbidden & attrs)


def test_no_ticker_identity_or_private_path_in_record():
    public = json.dumps(_record())
    assert "allocation_artifact_path" not in public
    assert "partition_manifest_path" not in public
