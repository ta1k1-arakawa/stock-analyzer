from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

from src import v9_006_f1_candidate_token_neighborhood_probe as probe
from src import v9_006_f1_locked_root_locator_successor_diagnostic as diag
from src.v9_005_stage_a_jpx_probe import LISTED_ISSUES_PAGE_URL, SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, TERMINAL_DISCOVERY_ROOT

FILLER51 = "".join(f"<a>x{i}</a>" for i in range(1, 52))


def make(attrs52: str = "", body52: str = "", attrs55: str = "", body55: str = "", between: str = "<a>b1</a><a>b2</a>", tail: str = "") -> bytes:
    return (FILLER51 + f"<a{attrs52}>{body52}</a>" + between + f"<a{attrs55}>{body55}</a>" + tail).encode("utf-8")


def dummy_summary(ordinal: int, **overrides: object) -> dict[str, object]:
    base = {"anchor_ordinal": ordinal, "normalized_visible_text": f"anchor{ordinal}", "nearest_preceding_heading_ordinal": None, "same_jpx_domain_after_resolution": True, "target_extension_class": "XLS", "raw_href_sha256": None, "resolved_url_sha256": None}
    base.update(overrides)
    return base


DUMMY_PRIOR_ANCHORS = [dummy_summary(o) for o in range(1, 90)]


@pytest.fixture(autouse=True)
def _dummy_bindings(monkeypatch):
    monkeypatch.setattr(probe, "_CANDIDATE_BINDINGS", {52: dummy_summary(52), 55: dummy_summary(55)})


def parse(raw: bytes, prior_anchors: list[dict[str, object]] = DUMMY_PRIOR_ANCHORS) -> list[dict[str, object]]:
    return probe.parse_candidate_neighborhoods(raw, prior_anchors, (52, 55))


# ---------------------------------------------------------------------------
# Prior-diagnostic binding (full run_probe pipeline)
# ---------------------------------------------------------------------------

def output_root(tmp_path, raw: bytes, *, subdir: str = "output") -> Path:
    root = tmp_path / subdir
    raw_dir = root / "raw"
    raw_dir.mkdir(parents=True)
    stem = "a" * 64
    metadata = {"schema_version": "V9_005_STAGE_A_RAW_LOCK_V1", "source_family": SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, "applicable_period": TERMINAL_DISCOVERY_ROOT, "requested_url": LISTED_ISSUES_PAGE_URL, "resolved_url": LISTED_ISSUES_PAGE_URL, "http_status": 200, "retrieval_timestamp_utc": "2026-01-01T00:00:00Z", "byte_length": len(raw), "sha256": sha256(raw).hexdigest()}
    (raw_dir / f"{stem}.json").write_text(json.dumps(metadata), encoding="utf-8")
    (raw_dir / f"{stem}.bin").write_bytes(raw)
    return root


def bound_html(mid52: str = "", mid55: str = "") -> bytes:
    filler_before = "".join(f"<a>f{i}</a>" for i in range(1, 52))
    between = "<a>b53</a><a>b54</a>"
    tail = "".join(f"<a>t{i}</a>" for i in range(56, 84))
    return ("<h1>Head</h1>" + filler_before + f"<a href='a.xls'>{mid52}</a>" + between + f"<a href='b.xlsx'>{mid55}</a>" + tail).encode("utf-8")


def build_bound(monkeypatch, tmp_path, *, mid52: str = "", mid55: str = "", subdir: str = "output"):
    raw = bound_html(mid52, mid55)
    monkeypatch.setattr(diag, "EXPECTED_LENGTH", len(raw))
    monkeypatch.setattr(diag, "EXPECTED_SHA256", sha256(raw).hexdigest())
    monkeypatch.setattr(diag, "verify_raw_provenance", lambda _root: True)
    root = output_root(tmp_path, raw, subdir=subdir)
    prior_result = diag.run_diagnostic(root)
    assert prior_result["diagnostic_result"] == "EVIDENCE_CAPTURED"
    assert prior_result["candidate_anchor_ordinals"] == [52, 55]
    assert prior_result["total_anchor_count"] == 83 and prior_result["total_heading_count"] == 1
    by_ordinal = {item["anchor_ordinal"]: item for item in prior_result["anchors"]}
    bindings = {ordinal: {key: by_ordinal[ordinal][key] for key in probe._SUMMARY_KEYS} for ordinal in (52, 55)}
    monkeypatch.setattr(probe, "PAYLOAD_HASH", sha256(raw).hexdigest())
    monkeypatch.setattr(probe, "EXPECTED_LENGTH", len(raw))
    monkeypatch.setattr(probe, "PRIOR_STRUCTURAL_HASH", prior_result["structural_evidence_sha256"])
    monkeypatch.setattr(probe, "_CANDIDATE_BINDINGS", bindings)
    return root, raw


def test_prior_binding_success_reaches_evidence_captured_with_exact_ordinal_boundaries(monkeypatch, tmp_path):
    root, _raw = build_bound(monkeypatch, tmp_path, mid52="INTERNAL52", mid55="INTERNAL55")
    result = probe.run_probe(root)
    assert result["diagnostic_result"] == "EVIDENCE_CAPTURED"
    assert result["prior_diagnostic_binding_verified"] is True
    contexts = result["candidate_contexts"]
    assert [context["candidate_anchor_ordinal"] for context in contexts] == [52, 55]
    ctx52, ctx55 = contexts
    assert ctx52["preceding_data_tokens"] == [{"data_token_ordinal": 52 - i, "normalized_text": f"f{51 - i}"} for i in range(8)]
    assert ctx52["following_data_tokens"] == [{"data_token_ordinal": 54, "normalized_text": "b53"}, {"data_token_ordinal": 55, "normalized_text": "b54"}, {"data_token_ordinal": 57, "normalized_text": "t56"}, {"data_token_ordinal": 58, "normalized_text": "t57"}, {"data_token_ordinal": 59, "normalized_text": "t58"}, {"data_token_ordinal": 60, "normalized_text": "t59"}, {"data_token_ordinal": 61, "normalized_text": "t60"}, {"data_token_ordinal": 62, "normalized_text": "t61"}]
    assert ctx55["preceding_data_tokens"] == [{"data_token_ordinal": 55, "normalized_text": "b54"}, {"data_token_ordinal": 54, "normalized_text": "b53"}, {"data_token_ordinal": 52, "normalized_text": "f51"}, {"data_token_ordinal": 51, "normalized_text": "f50"}, {"data_token_ordinal": 50, "normalized_text": "f49"}, {"data_token_ordinal": 49, "normalized_text": "f48"}, {"data_token_ordinal": 48, "normalized_text": "f47"}, {"data_token_ordinal": 47, "normalized_text": "f46"}]
    assert ctx55["following_data_tokens"] == [{"data_token_ordinal": 57 + i, "normalized_text": f"t{56 + i}"} for i in range(8)]
    all_texts = [t["normalized_text"] for c in contexts for t in c["preceding_data_tokens"] + c["following_data_tokens"]]
    assert "INTERNAL52" not in all_texts and "INTERNAL55" not in all_texts
    assert [item["anchor_ordinal"] for item in ctx52["previous_anchor_summaries"]] == [51, 50, 49]
    assert [item["anchor_ordinal"] for item in ctx52["next_anchor_summaries"]] == [53, 54, 55]
    assert [item["anchor_ordinal"] for item in ctx55["previous_anchor_summaries"]] == [54, 53, 52]
    probe.validate_safe_result(result)


def test_binding_failure_when_prior_diagnostic_not_evidence_captured(tmp_path):
    result = probe.run_probe(tmp_path / "missing")
    assert result["diagnostic_result"] == "PRIOR_DIAGNOSTIC_BINDING_FAILURE"
    assert result["prior_diagnostic_binding_verified"] is False and result["candidate_contexts"] == []
    assert result["input_payload_sha256"] == probe.PAYLOAD_HASH and result["prior_diagnostic_structural_evidence_sha256"] == probe.PRIOR_STRUCTURAL_HASH


def test_binding_failure_when_structural_hash_mismatches(monkeypatch, tmp_path):
    root, _raw = build_bound(monkeypatch, tmp_path)
    monkeypatch.setattr(probe, "PRIOR_STRUCTURAL_HASH", "0" * 64)
    result = probe.run_probe(root)
    assert result["diagnostic_result"] == "PRIOR_DIAGNOSTIC_BINDING_FAILURE" and result["prior_diagnostic_binding_verified"] is False


def test_binding_failure_when_total_anchor_count_wrong(monkeypatch, tmp_path):
    raw = bound_html() + b"<a>extra</a>"
    monkeypatch.setattr(diag, "EXPECTED_LENGTH", len(raw))
    monkeypatch.setattr(diag, "EXPECTED_SHA256", sha256(raw).hexdigest())
    monkeypatch.setattr(diag, "verify_raw_provenance", lambda _root: True)
    root = output_root(tmp_path, raw)
    prior_result = diag.run_diagnostic(root)
    assert prior_result["total_anchor_count"] == 84 and prior_result["candidate_anchor_ordinals"] == [52, 55]
    by_ordinal = {item["anchor_ordinal"]: item for item in prior_result["anchors"]}
    bindings = {ordinal: {key: by_ordinal[ordinal][key] for key in probe._SUMMARY_KEYS} for ordinal in (52, 55)}
    monkeypatch.setattr(probe, "PAYLOAD_HASH", sha256(raw).hexdigest())
    monkeypatch.setattr(probe, "EXPECTED_LENGTH", len(raw))
    monkeypatch.setattr(probe, "PRIOR_STRUCTURAL_HASH", prior_result["structural_evidence_sha256"])
    monkeypatch.setattr(probe, "_CANDIDATE_BINDINGS", bindings)
    result = probe.run_probe(root)
    assert result["diagnostic_result"] == "PRIOR_DIAGNOSTIC_BINDING_FAILURE"


def test_binding_failure_when_candidate_binding_row_mismatches(monkeypatch, tmp_path):
    root, _raw = build_bound(monkeypatch, tmp_path)
    mutated = dict(probe._CANDIDATE_BINDINGS[52])
    mutated["target_extension_class"] = "XLSX"
    monkeypatch.setattr(probe, "_CANDIDATE_BINDINGS", {52: mutated, 55: probe._CANDIDATE_BINDINGS[55]})
    result = probe.run_probe(root)
    assert result["diagnostic_result"] == "PRIOR_DIAGNOSTIC_BINDING_FAILURE"


def test_binding_failure_when_reread_payload_hash_mismatches_after_binding(monkeypatch, tmp_path):
    root, raw = build_bound(monkeypatch, tmp_path)
    bin_path = next((root / "raw").glob("*.bin"))
    bin_path.write_bytes(raw + b"tamper")
    result = probe.run_probe(root)
    assert result["diagnostic_result"] == "PRIOR_DIAGNOSTIC_BINDING_FAILURE" and result["prior_diagnostic_binding_verified"] is False


# ---------------------------------------------------------------------------
# Parser/token mechanics (direct parse_candidate_neighborhoods calls)
# ---------------------------------------------------------------------------

def test_candidate_context_order_is_fixed_52_then_55():
    contexts = parse(make())
    assert [c["candidate_anchor_ordinal"] for c in contexts] == [52, 55]
    assert contexts[0]["candidate_binding"] == probe._CANDIDATE_BINDINGS[52]
    assert contexts[1]["candidate_binding"] == probe._CANDIDATE_BINDINGS[55]


def test_candidate_internal_data_excluded_from_neighborhoods():
    raw = make(body52="INSIDE52", between="<a>mid1</a><a>mid2</a>", body55="INSIDE55")
    contexts = parse(raw)
    texts = [t["normalized_text"] for c in contexts for t in c["preceding_data_tokens"] + c["following_data_tokens"]]
    assert "INSIDE52" not in texts and "INSIDE55" not in texts
    following52 = [t["normalized_text"] for t in contexts[0]["following_data_tokens"]]
    preceding55 = [t["normalized_text"] for t in contexts[1]["preceding_data_tokens"]]
    assert following52 == ["mid1", "mid2"]
    assert preceding55[:2] == ["mid2", "mid1"]


def test_preceding_and_following_truncate_to_nearest_eight_in_order():
    tail = "".join(f"<a>t{i}</a>" for i in range(1, 12))
    contexts = parse(make(tail=tail))
    following55 = contexts[1]["following_data_tokens"]
    assert [t["normalized_text"] for t in following55] == [f"t{i}" for i in range(1, 9)]
    assert [t["data_token_ordinal"] for t in following55] == sorted(t["data_token_ordinal"] for t in following55)
    preceding52 = contexts[0]["preceding_data_tokens"]
    assert [t["normalized_text"] for t in preceding52] == [f"x{i}" for i in range(51, 43, -1)]


@pytest.mark.parametrize("html", [
    b"<a><a>x</a></a>",
    b"<a/>",
    b"<a>x",
    b"</a>",
])
def test_general_anchor_structure_failures_are_unsupported(html):
    with pytest.raises(probe._Unsupported):
        parse(FILLER51.encode() + html + b"<a>b</a><a>y</a>")


@pytest.mark.parametrize("script", ["<script>ignored</script>", "<style>ignored</style>", "<noscript>ignored</noscript>", "<template>ignored</template>"])
def test_suppression_hides_data_same_and_cross_tag_nesting(script):
    raw = make(tail=f"{script}<script><style>nested</style></script>KEPT")
    contexts = parse(raw)
    following = contexts[1]["following_data_tokens"]
    assert [t["normalized_text"] for t in following] == ["KEPT"]


def test_suppression_unmatched_close_is_unsupported():
    with pytest.raises(probe._Unsupported):
        parse(make(tail="</script>"))


def test_suppression_startend_is_atomic_no_state_change():
    contexts = parse(make(tail="<script/>KEPT"))
    assert [t["normalized_text"] for t in contexts[1]["following_data_tokens"]] == ["KEPT"]


def test_suppression_nonzero_at_eof_is_unsupported():
    with pytest.raises(probe._Unsupported):
        parse(make(tail="<script>never closed"))


def test_img_start_and_startend_each_count_one_image():
    raw = make(body52="<img src='a.png'><img src='b.png'/>")
    contexts = parse(raw)
    assert contexts[0]["total_image_count"] == 2
    assert [img["image_ordinal_within_candidate"] for img in contexts[0]["images"]] == [1, 2]


def test_explicit_close_img_tag_is_inert_while_candidate_active_and_inactive():
    raw = make(body52="<img src='a.png'></img>", tail="<img src='outside.png'></img>")
    contexts = parse(raw)
    assert contexts[0]["total_image_count"] == 1 and len(contexts[0]["images"]) == 1


def test_noncandidate_image_has_no_effect():
    raw = make(tail="<img alt='x' alt='y'>")
    contexts = parse(raw)
    assert contexts[0]["total_image_count"] == 0 and contexts[1]["total_image_count"] == 0


def test_more_than_eight_images_counts_all_but_emits_first_eight():
    body = "".join(f"<img src='{i}.png'>" for i in range(1, 11))
    contexts = parse(make(body52=body))
    ctx = contexts[0]
    assert ctx["total_image_count"] == 10 and len(ctx["images"]) == 8
    assert [img["image_ordinal_within_candidate"] for img in ctx["images"]] == list(range(1, 9))


def test_duplicate_candidate_attrs_are_unsupported():
    for attrs in (" title='a' title='b'", " aria-label='a' aria-label='b'", " download='a' download='b'"):
        with pytest.raises(probe._Unsupported):
            parse(make(attrs52=attrs))


def test_duplicate_image_attrs_are_unsupported():
    for html in ("<img alt='a' alt='b'>", "<img title='a' title='b'>", "<img src='a' src='b'>"):
        with pytest.raises(probe._Unsupported):
            parse(make(body52=html))


@pytest.mark.parametrize(("attrs", "present_key", "normalized_key", "expected_present", "expected_normalized"), [
    ("", "title_present", "normalized_title", False, None),
    (" title", "title_present", "normalized_title", True, None),
    (" title=''", "title_present", "normalized_title", True, ""),
    (" title='  a  b '", "title_present", "normalized_title", True, "a b"),
    ("", "aria_label_present", "normalized_aria_label", False, None),
    (" aria-label", "aria_label_present", "normalized_aria_label", True, None),
    (" aria-label='x'", "aria_label_present", "normalized_aria_label", True, "x"),
])
def test_candidate_title_and_aria_label_absent_none_string_matrix(attrs, present_key, normalized_key, expected_present, expected_normalized):
    context = parse(make(attrs52=attrs))[0]
    assert context[present_key] is expected_present and context[normalized_key] == expected_normalized


@pytest.mark.parametrize(("attrs", "expected_present", "expected_hash"), [
    ("", False, None),
    (" download", True, None),
    (" download=''", True, sha256(b"").hexdigest()),
    (" download='file.xls'", True, sha256(b"file.xls").hexdigest()),
])
def test_candidate_download_absent_none_string_matrix(attrs, expected_present, expected_hash):
    context = parse(make(attrs52=attrs))[0]
    assert context["download_present"] is expected_present and context["download_value_sha256"] == expected_hash


@pytest.mark.parametrize(("attrs", "expected_present", "expected_hash"), [
    ("", False, None),
    (" src", True, None),
    (" src=''", True, sha256(b"").hexdigest()),
    (" src='pic.png'", True, sha256(b"pic.png").hexdigest()),
])
def test_image_src_absent_none_string_matrix(attrs, expected_present, expected_hash):
    context = parse(make(body52=f"<img{attrs}>"))[0]
    image = context["images"][0]
    assert image["src_present"] is expected_present and image["src_sha256"] == expected_hash


@pytest.mark.parametrize("text", ["http://bad", "https://bad", "file:x", "C:\\private\\x"])
def test_unsafe_text_in_preceding_token_emits_empty_validation_failure(text):
    with pytest.raises(probe._UnsafeText):
        parse(make(tail=text))


@pytest.mark.parametrize("attrs", [" title='http://bad'", " aria-label='https://bad'"])
def test_unsafe_text_in_candidate_attribute_emits_validation_failure(attrs):
    with pytest.raises(probe._UnsafeText):
        parse(make(attrs52=attrs))


def test_unsafe_text_in_image_alt_or_title_emits_validation_failure():
    with pytest.raises(probe._UnsafeText):
        parse(make(body52="<img alt='file:x'>"))
    with pytest.raises(probe._UnsafeText):
        parse(make(body52="<img title='C:\\\\x'>"))


def test_lowercased_tag_and_attribute_matching():
    contexts = parse(make(attrs52=" TITLE='X'", body52="<IMG SRC='a.png'>"))
    assert contexts[0]["title_present"] is True and contexts[0]["normalized_title"] == "X"
    assert contexts[0]["total_image_count"] == 1


def test_no_candidate_selection_fixed_fields(monkeypatch, tmp_path):
    root, _raw = build_bound(monkeypatch, tmp_path)
    result = probe.run_probe(root)
    assert result["locator_decision"] == "NOT_MADE" and result["replacement_locator_authorized"] is False and result["network_requests"] == 0


# ---------------------------------------------------------------------------
# Safe validator negative coverage
# ---------------------------------------------------------------------------

def evidence_result(monkeypatch, tmp_path):
    root, _raw = build_bound(monkeypatch, tmp_path, mid52="", mid55="")
    return probe.run_probe(root)


def evidence_result_with_neighbors(monkeypatch, tmp_path):
    root, _raw = build_bound(monkeypatch, tmp_path, mid52="", mid55="")
    result = probe.run_probe(root)
    prior_result = diag.run_diagnostic(root)
    prior_by_ordinal = {item["anchor_ordinal"]: item for item in prior_result["anchors"]}
    expected_neighbors = probe._build_expected_neighbors(prior_by_ordinal)
    return json.loads(probe.canonical_json(result)), expected_neighbors


def rehash(value: dict[str, object]) -> dict[str, object]:
    value["structural_evidence_sha256"] = sha256(probe.canonical_json({k: v for k, v in value.items() if k != "structural_evidence_sha256"}).encode()).hexdigest()
    return value


# ---------------------------------------------------------------------------
# MEDIUM_1: adjacent-anchor safe summaries must be exactly bound to the same
# recomputed prior diagnostic, not merely schema/ordering plausible.
# ---------------------------------------------------------------------------

def test_validator_accepts_the_real_exact_neighbor_binding(monkeypatch, tmp_path):
    result, expected = evidence_result_with_neighbors(monkeypatch, tmp_path)
    probe.validate_safe_result(result, expected_neighbors=expected)
    assert [item["anchor_ordinal"] for item in expected[52]["previous"]] == [51, 50, 49]
    assert [item["anchor_ordinal"] for item in expected[52]["next"]] == [53, 54, 55]
    assert [item["anchor_ordinal"] for item in expected[55]["previous"]] == [54, 53, 52]
    assert [item["anchor_ordinal"] for item in expected[55]["next"]] == [56, 57, 58]


def test_validator_rejects_candidate_52_previous_summaries_emptied(monkeypatch, tmp_path):
    result, expected = evidence_result_with_neighbors(monkeypatch, tmp_path)
    result["candidate_contexts"][0]["previous_anchor_summaries"] = []
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result, expected_neighbors=expected)


def test_validator_rejects_candidate_52_previous_summaries_shifted_by_one(monkeypatch, tmp_path):
    result, expected = evidence_result_with_neighbors(monkeypatch, tmp_path)
    for item in result["candidate_contexts"][0]["previous_anchor_summaries"]:
        item["anchor_ordinal"] -= 1
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result, expected_neighbors=expected)


def test_validator_rejects_candidate_52_next_summary_with_one_field_altered(monkeypatch, tmp_path):
    result, expected = evidence_result_with_neighbors(monkeypatch, tmp_path)
    next_summaries = result["candidate_contexts"][0]["next_anchor_summaries"]
    assert next_summaries[0]["anchor_ordinal"] == 53
    next_summaries[0]["normalized_visible_text"] = "tampered"
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result, expected_neighbors=expected)


def test_validator_rejects_candidate_55_previous_list_missing_ordinal_52(monkeypatch, tmp_path):
    result, expected = evidence_result_with_neighbors(monkeypatch, tmp_path)
    result["candidate_contexts"][1]["previous_anchor_summaries"] = result["candidate_contexts"][1]["previous_anchor_summaries"][:2]
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result, expected_neighbors=expected)


def test_validator_rejects_candidate_55_next_list_replaced_wholesale(monkeypatch, tmp_path):
    result, expected = evidence_result_with_neighbors(monkeypatch, tmp_path)
    substitute = [dummy_summary(ordinal) for ordinal in (56, 57, 58)]
    result["candidate_contexts"][1]["next_anchor_summaries"] = substitute
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result, expected_neighbors=expected)


def test_validator_accepts_valid_result_and_rejects_extra_key(monkeypatch, tmp_path):
    result = evidence_result(monkeypatch, tmp_path)
    probe.validate_safe_result(result)
    mutated = dict(result); mutated["extra"] = 1
    with pytest.raises(ValueError):
        probe.validate_safe_result(mutated)


def test_validator_rejects_network_requests_bool_vs_int(monkeypatch, tmp_path):
    result = json.loads(probe.canonical_json(evidence_result(monkeypatch, tmp_path)))
    result["network_requests"] = False
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result)


def test_validator_rejects_wrong_candidate_order(monkeypatch, tmp_path):
    result = json.loads(probe.canonical_json(evidence_result(monkeypatch, tmp_path)))
    result["candidate_contexts"] = list(reversed(result["candidate_contexts"]))
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result)


def test_validator_rejects_tampered_candidate_binding(monkeypatch, tmp_path):
    result = json.loads(probe.canonical_json(evidence_result(monkeypatch, tmp_path)))
    result["candidate_contexts"][0]["candidate_binding"]["target_extension_class"] = "OTHER"
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result)


def test_validator_rejects_out_of_order_tokens(monkeypatch, tmp_path):
    result = json.loads(probe.canonical_json(evidence_result(monkeypatch, tmp_path)))
    tokens = result["candidate_contexts"][0]["preceding_data_tokens"]
    if len(tokens) < 2:
        pytest.skip("need at least two preceding tokens")
    tokens[0], tokens[1] = tokens[1], tokens[0]
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result)


def test_validator_rejects_out_of_order_summaries(monkeypatch, tmp_path):
    result = json.loads(probe.canonical_json(evidence_result(monkeypatch, tmp_path)))
    summaries = result["candidate_contexts"][0]["next_anchor_summaries"]
    if len(summaries) < 2:
        pytest.skip("need at least two summaries")
    summaries[0], summaries[1] = summaries[1], summaries[0]
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result)


def test_validator_rejects_image_count_length_mismatch(monkeypatch, tmp_path):
    result = json.loads(probe.canonical_json(evidence_result(monkeypatch, tmp_path)))
    result["candidate_contexts"][0]["total_image_count"] = 5
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result)


def test_validator_rejects_wrong_image_ordinal(monkeypatch, tmp_path):
    root, _raw = build_bound(monkeypatch, tmp_path, mid52="<img src='a.png'><img src='b.png'>")
    result = json.loads(probe.canonical_json(probe.run_probe(root)))
    result["candidate_contexts"][0]["images"][0]["image_ordinal_within_candidate"] = 2
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result)


@pytest.mark.parametrize("mutate", [
    lambda value: value["candidate_contexts"][0].__setitem__("title_present", False),
    lambda value: value["candidate_contexts"][0].__setitem__("aria_label_present", True),
    lambda value: value["candidate_contexts"][0].__setitem__("download_present", "not-a-bool"),
])
def test_validator_rejects_present_normalized_state_matrix_violations(monkeypatch, tmp_path, mutate):
    root, _raw = build_bound(monkeypatch, tmp_path, mid52="", mid55="")
    result = json.loads(probe.canonical_json(probe.run_probe(root)))
    result["candidate_contexts"][0]["normalized_title"] = "still-set"
    mutate(result)
    rehash(result)
    with pytest.raises(ValueError):
        probe.validate_safe_result(result)


def test_validator_rejects_tampered_hash(monkeypatch, tmp_path):
    result = dict(evidence_result(monkeypatch, tmp_path))
    result["structural_evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        probe.validate_safe_result(result)


def test_validator_rejects_failure_result_with_nonempty_contexts():
    value = probe._empty("HTML_STRUCTURE_UNSUPPORTED", True)
    value["candidate_contexts"] = [{}]
    rehash(value)
    with pytest.raises(ValueError):
        probe.validate_safe_result(value)


def test_validator_rejects_binding_failure_with_verified_true():
    value = probe._empty("PRIOR_DIAGNOSTIC_BINDING_FAILURE", True)
    rehash(value)
    with pytest.raises(ValueError):
        probe.validate_safe_result(value)


def test_structural_hash_recomputation_matches(monkeypatch, tmp_path):
    result = evidence_result(monkeypatch, tmp_path)
    digestless = dict(result)
    digest = digestless.pop("structural_evidence_sha256")
    assert digest == sha256(probe.canonical_json(digestless).encode()).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_cli():
    path = Path("scripts/run_v9_006_f1_candidate_token_neighborhood_probe.py")
    spec = importlib.util.spec_from_file_location("f1_neighborhood_cli", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_one_safe_line_without_urls_or_paths(monkeypatch, tmp_path):
    root, _raw = build_bound(monkeypatch, tmp_path)
    cli = load_cli()
    monkeypatch.setattr(cli, "run_probe", lambda _root: probe.run_probe(root))
    import io
    import contextlib
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exit_code = cli.main(["--output-root", str(root)])
    out = buffer.getvalue().strip()
    assert exit_code == 0
    assert out.count("\n") == 0 and "https://" not in out and str(root) not in out
    assert json.loads(out)["diagnostic_result"] == "EVIDENCE_CAPTURED"


def test_cli_unexpected_failure_uses_only_fixed_stderr_marker(monkeypatch):
    cli = load_cli()
    monkeypatch.setattr(cli, "run_probe", lambda _root: (_ for _ in ()).throw(RuntimeError("secret https://x")))
    import io
    import contextlib
    out_buffer, err_buffer = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out_buffer), contextlib.redirect_stderr(err_buffer):
        exit_code = cli.main(["--output-root", "C:\\private"])
    assert exit_code == 3
    assert out_buffer.getvalue() == "" and err_buffer.getvalue() == "V9_006_F1_CANDIDATE_NEIGHBORHOOD_IMPLEMENTATION_FAILURE\n"


def test_cli_returns_two_for_designed_failures(monkeypatch):
    cli = load_cli()
    monkeypatch.setattr(cli, "run_probe", lambda _root: probe._finalize(probe._empty("HTML_STRUCTURE_UNSUPPORTED", True)))
    assert cli.main(["--output-root", "irrelevant"]) == 2


def test_zero_real_network_in_every_result(monkeypatch, tmp_path):
    root, _raw = build_bound(monkeypatch, tmp_path)
    assert probe.run_probe(root)["network_requests"] == 0
    assert probe._empty("PRIOR_DIAGNOSTIC_BINDING_FAILURE", False)["network_requests"] == 0
