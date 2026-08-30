from __future__ import annotations

from hashlib import sha256
import contextlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from src import v9_006_f1_locked_root_locator_successor_diagnostic as prior
from src import v9_006_f1_semantic_successor_locator as locator

BASE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
P1 = "List of TSE-listed Issues as of previous month-end is available."


def html(href: str = "a.xls", title: str = "List of TSE-listed Issues (Jan. 2026)", p1: str = P1, extra: str = "") -> bytes:
    return f"{p1}<p>{title}</p>{extra}<a href='{href}'>download</a>".encode()


def select(raw: bytes):
    return locator.locate_html(raw, BASE)


@pytest.mark.parametrize("month", ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
@pytest.mark.parametrize("extension", ["xls", "xlsx", "csv", "zip"])
def test_all_frozen_months_and_candidate_extensions_qualify(month, extension):
    mechanical, qualified = select(html(f"a.{extension}", f"List of TSE-listed Issues ({month}. 2026)"))
    assert len(mechanical) == len(qualified) == 1


@pytest.mark.parametrize("title", [
    "List of TSE-listed Issues (Jan. ２０２６)",
    "List of TSE-listed Issues (Jan 2026)",
    "List of TSE-listed Issues (Foo. 2026)",
    "List of TSE-listed Issues (Jan. 202)",
    "List of TSE-listed Issues (Jan. 20266)",
    "List of TSE-listed Issues Jan. 2026)",
])
def test_year_grammar_and_literal_punctuation_reject(title):
    mechanical, qualified = select(html(title=title))
    assert len(mechanical) == 1 and qualified == []


def test_wrong_p1_and_non_jpx_and_query_extension_rules():
    assert select(html(p1="wrong"))[1] == []
    assert select(html("https://example.invalid/a.xls"))[0] == []
    assert locator._extension(urllib_join(BASE, "a.xls?f=.zip#x")) == "XLS"


def test_no_preference_and_multiple_qualifiers_are_exposed_for_failure():
    raw = html("a.csv") + html("b.zip", "List of TSE-listed Issues (Feb. 2026)")
    mechanical, qualified = select(raw)
    assert len(mechanical) == len(qualified) == 2


@pytest.mark.parametrize("bad", [b"<a href='a.xls' href='b.xls'>x</a>", b"<a><a>x</a></a>", b"<a/>", b"<a>x", b"</a>"])
def test_inherited_anchor_failures(bad):
    with pytest.raises(locator._Unsupported):
        select(bad)


def test_suppression_excludes_tokens_and_hashes_are_exact():
    raw = b"<script>" + P1.encode() + b"</script><p>List of TSE-listed Issues (Jan. 2026)</p><a href='a.xls'>x</a>"
    assert select(raw)[1] == []
    mechanical, qualified = select(html("path/a.xls"))
    assert qualified[0]["raw_href_sha256"] == sha256(b"path/a.xls").hexdigest()
    assert qualified[0]["resolved_url_sha256"] == sha256(urllib_join(BASE, "path/a.xls").encode()).hexdigest()


def urllib_join(base, href):
    import urllib.parse
    return urllib.parse.urljoin(base, href)


def output_root(tmp_path: Path, raw: bytes) -> Path:
    root = (tmp_path / "root").resolve(); raw_dir = root / "raw"; raw_dir.mkdir(parents=True)
    stem = "a" * 64
    (raw_dir / f"{stem}.bin").write_bytes(raw)
    metadata = {"schema_version": "V9_005_STAGE_A_RAW_LOCK_V1", "source_family": prior.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, "applicable_period": prior.TERMINAL_DISCOVERY_ROOT, "requested_url": prior.LISTED_ISSUES_PAGE_URL, "resolved_url": BASE, "http_status": 200, "retrieval_timestamp_utc": "2026-01-01T00:00:00Z", "byte_length": len(raw), "sha256": sha256(raw).hexdigest()}
    (raw_dir / f"{stem}.json").write_text(json.dumps(metadata), encoding="utf-8")
    return root


def bound_root(monkeypatch, tmp_path, raw=None):
    raw = raw or html()
    digest = sha256(raw).hexdigest()
    monkeypatch.setattr(prior, "EXPECTED_LENGTH", len(raw)); monkeypatch.setattr(prior, "EXPECTED_SHA256", digest)
    monkeypatch.setattr(prior, "verify_raw_provenance", lambda _root: True)
    monkeypatch.setattr(locator, "EXPECTED_LENGTH", len(raw)); monkeypatch.setattr(locator, "EXPECTED_PAYLOAD_SHA256", digest)
    root = output_root(tmp_path, raw)
    prior_result = prior.run_diagnostic(root)
    monkeypatch.setattr(locator, "EXPECTED_PRIOR_STRUCTURAL", prior_result["structural_evidence_sha256"])
    return root, digest


def test_offline_runner_success_and_ambiguity_are_safe(monkeypatch, tmp_path):
    root, digest = bound_root(monkeypatch, tmp_path)
    result = locator.run_locator(root)
    assert result["result"] == "SUCCESSOR_LOCATOR_MATCHED" and result["input_payload_sha256"] == digest
    ambiguous = html() + html("b.xlsx", "List of TSE-listed Issues (Feb. 2026)")
    root, digest = bound_root(monkeypatch, tmp_path / "ambiguous", ambiguous)
    result = locator.run_locator(root)
    assert result["result"] == "SOURCE_OR_DATA_FEASIBILITY_FAILURE" and result["mechanical_candidate_count"] == result["qualifying_candidate_count"] == 0 and result["input_payload_sha256"] == digest


def rehash(value):
    value["structural_evidence_sha256"] = sha256(locator.canonical_json({k: v for k, v in value.items() if k != "structural_evidence_sha256"}).encode()).hexdigest(); return value


def test_safe_validator_closure(monkeypatch, tmp_path):
    root, _ = bound_root(monkeypatch, tmp_path)
    value = json.loads(locator.canonical_json(locator.run_locator(root)))
    locator.validate_safe_result(value)
    value["mechanical_candidate_count"] = True; rehash(value)
    with pytest.raises(ValueError): locator.validate_safe_result(value)
    failure = locator._finalize(locator._empty("SOURCE_OR_DATA_FEASIBILITY_FAILURE", locator.EXPECTED_PAYLOAD_SHA256))
    failure["selected_raw_href_sha256"] = "0" * 64; rehash(failure)
    with pytest.raises(ValueError): locator.validate_safe_result(failure)


def load_cli():
    spec = importlib.util.spec_from_file_location("semantic_cli", Path("scripts/run_v9_006_f1_semantic_successor_locator.py")); assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_cli_exit_codes_and_safe_failure_marker(monkeypatch, tmp_path):
    root, _ = bound_root(monkeypatch, tmp_path); cli = load_cli()
    monkeypatch.setattr(cli, "run_locator", lambda _root: locator.run_locator(root))
    out = io.StringIO()
    with contextlib.redirect_stdout(out): assert cli.main(["--output-root", str(root)]) == 0
    assert out.getvalue().count("\n") == 1 and "https://" not in out.getvalue()
    monkeypatch.setattr(cli, "run_locator", lambda _root: locator._finalize(locator._empty("HTML_STRUCTURE_UNSUPPORTED", locator.EXPECTED_PAYLOAD_SHA256)))
    assert cli.main(["--output-root", "x"]) == 2
    monkeypatch.setattr(cli, "run_locator", lambda _root: (_ for _ in ()).throw(RuntimeError("https://secret")))
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr): assert cli.main(["--output-root", "x"]) == 3
    assert stdout.getvalue() == "" and stderr.getvalue() == "V9_006_F1_SEMANTIC_SUCCESSOR_LOCATOR_IMPLEMENTATION_FAILURE\n"
