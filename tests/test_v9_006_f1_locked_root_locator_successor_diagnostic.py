from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import pytest

from src import v9_006_f1_locked_root_locator_successor_diagnostic as diag
from src.v9_005_stage_a_jpx_probe import LISTED_ISSUES_PAGE_URL, SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, TERMINAL_DISCOVERY_ROOT


def configure_synthetic(monkeypatch, raw=b"<title>Title</title><h1>Head</h1><a href='data.xls'><b>Visible</b></a>"):
    monkeypatch.setattr(diag, "EXPECTED_LENGTH", len(raw))
    monkeypatch.setattr(diag, "EXPECTED_SHA256", sha256(raw).hexdigest())
    monkeypatch.setattr(diag, "verify_raw_provenance", lambda _root: True)
    return raw


def output_root(tmp_path, raw, *, metadata=None):
    root = tmp_path / "output"; raw_dir = root / "raw"; raw_dir.mkdir(parents=True)
    stem = "a" * 64
    if metadata is None:
        metadata = {"schema_version": "V9_005_STAGE_A_RAW_LOCK_V1", "source_family": SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, "applicable_period": TERMINAL_DISCOVERY_ROOT, "requested_url": LISTED_ISSUES_PAGE_URL, "resolved_url": LISTED_ISSUES_PAGE_URL, "http_status": 200, "retrieval_timestamp_utc": "2026-01-01T00:00:00Z", "byte_length": len(raw), "sha256": sha256(raw).hexdigest()}
    (raw_dir / f"{stem}.json").write_text(json.dumps(metadata), encoding="utf-8")
    (raw_dir / f"{stem}.bin").write_bytes(raw)
    return root


def test_evidence_captured_nested_text_heading_and_candidate(monkeypatch, tmp_path):
    raw = configure_synthetic(monkeypatch, b"<title>Title</title><h1>Heading <i>one</i></h1><a href='dir/a.XLS?x=.pdf'><b>Visible</b> text</a><a>none</a><a href>nil</a><a href=''>empty</a>")
    result = diag.run_diagnostic(output_root(tmp_path, raw))
    assert result["diagnostic_result"] == "EVIDENCE_CAPTURED"
    assert result["title"] == "Title" and result["headings"] == [{"ordinal": 1, "level": 1, "normalized_text": "Heading one"}]
    first, absent, none, empty = result["anchors"]
    assert first["normalized_visible_text"] == "Visible text" and first["nearest_preceding_heading_ordinal"] == 1
    assert first["target_extension_class"] == "XLS" and result["candidate_anchor_ordinals"] == [1]
    assert absent["href_present"] is False and absent["target_extension_class"] == "NONE"
    assert none["href_present"] is True and none["raw_href_sha256"] is None and none["target_extension_class"] == "OTHER"
    assert empty["href_present"] is True and empty["raw_href_sha256"] == sha256(b"").hexdigest()


@pytest.mark.parametrize("html", [b"<a href='x' href='y'>x</a>", b"<a><a>x</a></a>", b"<h1><h2>x</h2></h1>", b"<a/>", b"<h1>x", b"</a>", b"<title>x</title><title>y</title>"])
def test_tracked_structure_failures_emit_empty(monkeypatch, tmp_path, html):
    raw = configure_synthetic(monkeypatch, html)
    result = diag.run_diagnostic(output_root(tmp_path, raw))
    assert result["diagnostic_result"] == "HTML_STRUCTURE_UNSUPPORTED"
    assert result["document_parse_status"] == "UNSUPPORTED" and result["anchors"] == result["headings"] == [] and result["title"] is None


@pytest.mark.parametrize(("href", "expected"), [("a.xls", "XLS"), ("a.xlsx", "XLSX"), ("a.csv", "CSV"), ("a.zip", "ZIP"), ("a.pdf", "PDF"), ("a.html", "HTML"), ("a.htm", "HTML"), ("a.bin", "OTHER"), ("A.XLS", "XLS"), ("a%2Exls", "XLS"), ("a.pdf?x=.xls", "PDF")])
def test_extension_mapping(monkeypatch, href, expected):
    configure_synthetic(monkeypatch)
    parsed = diag.parse_locked_html(f"<a href='{href}'>x</a>".encode(), LISTED_ISSUES_PAGE_URL)
    assert parsed["anchors"][0]["target_extension_class"] == expected


def test_only_resolved_url_is_base_and_offdomain_or_exception_is_safe(monkeypatch):
    configure_synthetic(monkeypatch)
    parsed = diag.parse_locked_html(b"<a href='child.xls'>x</a><a href='https://evil.example/x.xls'>y</a>", "https://www.jpx.co.jp/base/page.html")
    assert parsed["anchors"][0]["same_jpx_domain_after_resolution"] is True
    assert parsed["anchors"][1]["same_jpx_domain_after_resolution"] is False
    monkeypatch.setattr(diag.urllib.parse, "urljoin", lambda *_args: (_ for _ in ()).throw(ValueError()))
    failed = diag.parse_locked_html(b"<a href='x.xls'>x</a>", LISTED_ISSUES_PAGE_URL)["anchors"][0]
    assert failed["same_jpx_domain_after_resolution"] == "unknown" and failed["resolved_url_sha256"] is None


@pytest.mark.parametrize("text", ["http://bad", "https://bad", "file:x", "C:\\private\\x"])
def test_unsafe_visible_text_emits_empty_validation_failure(monkeypatch, tmp_path, text):
    raw = configure_synthetic(monkeypatch, f"<a href='x.xls'>{text}</a>".encode())
    result = diag.run_diagnostic(output_root(tmp_path, raw))
    assert result["diagnostic_result"] == "SAFE_OUTPUT_VALIDATION_FAILURE"
    assert result["document_parse_status"] == "PARSED" and result["anchors"] == [] and result["candidate_count"] == 0


def test_binding_is_sequential_and_metadata_failure_never_reads_payload(monkeypatch, tmp_path):
    raw = configure_synthetic(monkeypatch)
    root = output_root(tmp_path, raw)
    meta_path = next((root / "raw").glob("*.json")); meta = json.loads(meta_path.read_text(encoding="utf-8")); meta["source_family"] = "bad"; meta_path.write_text(json.dumps(meta), encoding="utf-8")
    bin_path = next((root / "raw").glob("*.bin"))
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(AssertionError("payload read")) if self == bin_path else b"")
    result = diag.run_diagnostic(root)
    assert result["diagnostic_result"] == "INPUT_BINDING_FAILURE" and result["input"]["metadata_identity_verified"] is False and result["input"]["payload_binding_verified"] is False
    monkeypatch.undo(); raw = configure_synthetic(monkeypatch); root = output_root(tmp_path / "second", raw); monkeypatch.setattr(diag, "verify_raw_provenance", lambda _root: False)
    result = diag.run_diagnostic(root)
    assert result["input"]["metadata_identity_verified"] is True and result["input"]["payload_binding_verified"] is True and result["input"]["raw_provenance_verified"] is False


def test_closed_validator_hash_and_negative_schema_cases(monkeypatch, tmp_path):
    raw = configure_synthetic(monkeypatch); result = diag.run_diagnostic(output_root(tmp_path, raw))
    diag.validate_safe_result(result)
    changed = dict(result); changed["network_requests"] = False
    with pytest.raises(ValueError): diag.validate_safe_result(changed)
    changed = dict(result); changed["extra"] = 1
    with pytest.raises(ValueError): diag.validate_safe_result(changed)
    changed = json.loads(diag.canonical_json(result)); changed["candidate_anchor_ordinals"] = []
    with pytest.raises(ValueError): diag.validate_safe_result(changed)
    changed = dict(result); changed["structural_evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError): diag.validate_safe_result(changed)
    changed = json.loads(diag.canonical_json(result)); changed["total_anchor_count"] = True
    with pytest.raises(ValueError): diag.validate_safe_result(changed)
    changed = json.loads(diag.canonical_json(result)); changed["anchors"][0]["anchor_ordinal"] = 2
    changed["structural_evidence_sha256"] = sha256(diag.canonical_json({key: value for key, value in changed.items() if key != "structural_evidence_sha256"}).encode()).hexdigest()
    with pytest.raises(ValueError): diag.validate_safe_result(changed)
    changed = json.loads(diag.canonical_json(result)); changed["anchors"][0]["raw_href_sha256"] = "A" * 64
    with pytest.raises(ValueError): diag.validate_safe_result(changed)
    digestless = dict(result); digest = digestless.pop("structural_evidence_sha256")
    assert digest == sha256(diag.canonical_json(digestless).encode()).hexdigest()


def rehash(value):
    value["structural_evidence_sha256"] = sha256(diag.canonical_json({key: item for key, item in value.items() if key != "structural_evidence_sha256"}).encode()).hexdigest()
    return value


@pytest.mark.parametrize("mutate", [
    lambda value: value["headings"][0].__setitem__("ordinal", True),
    lambda value: value["anchors"][0].__setitem__("anchor_ordinal", True),
    lambda value: value.__setitem__("candidate_anchor_ordinals", [True]),
    lambda value: value["anchors"][0].update(raw_href_sha256=None, resolved_url_sha256=None),
    lambda value: value["anchors"][0].update(raw_href_sha256=None, same_jpx_domain_after_resolution=False),
    lambda value: value["anchors"][0].update(same_jpx_domain_after_resolution="unknown"),
    lambda value: value["anchors"][0].update(href_present=False, target_extension_class="XLS"),
    lambda value: value["anchors"][0].update(target_extension_class="NONE"),
])
def test_validator_rejects_closed_ordinal_and_href_matrix_violations(monkeypatch, tmp_path, mutate):
    raw = configure_synthetic(monkeypatch, b"<h1>head</h1><a href='x.xls'>x</a>")
    value = json.loads(diag.canonical_json(diag.run_diagnostic(output_root(tmp_path, raw))))
    mutate(value); rehash(value)
    with pytest.raises(ValueError): diag.validate_safe_result(value)


def test_validator_rejects_input_binding_failure_with_all_true_flags(monkeypatch):
    configure_synthetic(monkeypatch)
    value = diag._empty("INPUT_BINDING_FAILURE", (True, True, False), "NOT_PARSED")
    value["input"]["raw_provenance_verified"] = True; rehash(value)
    with pytest.raises(ValueError): diag.validate_safe_result(value)


def load_cli():
    path = Path("scripts/run_v9_006_f1_locked_root_locator_successor_diagnostic.py")
    spec = importlib.util.spec_from_file_location("f1_successor_cli", path); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_cli_one_safe_line_without_urls_or_paths(monkeypatch, tmp_path, capsys):
    raw = configure_synthetic(monkeypatch); root = output_root(tmp_path, raw); cli = load_cli()
    monkeypatch.setattr(cli, "run_diagnostic", lambda _root: diag.run_diagnostic(root))
    assert cli.main(["--output-root", str(root)]) == 0
    out = capsys.readouterr().out.strip()
    assert out.count("\n") == 0 and "https://" not in out and str(root) not in out
    assert json.loads(out)["diagnostic_result"] == "EVIDENCE_CAPTURED"


def test_cli_unexpected_failure_uses_only_fixed_stderr_marker(monkeypatch, capsys):
    cli = load_cli(); monkeypatch.setattr(cli, "run_diagnostic", lambda _root: (_ for _ in ()).throw(RuntimeError("secret https://x")))
    assert cli.main(["--output-root", "C:\\private"]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "V9_006_F1_DIAGNOSTIC_IMPLEMENTATION_FAILURE\n"
