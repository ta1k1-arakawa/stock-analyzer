from __future__ import annotations

from http.client import HTTPException, IncompleteRead
from pathlib import Path
import ssl
from urllib.error import HTTPError, URLError

import pytest

from src import v9_006_f1_semantic_successor_public_acquisition as acq
from src import v9_006_f1_semantic_successor_public_acquisition_production as production
from src.v9_005_stage_a_jpx_probe import LISTED_ISSUES_PAGE_URL

SHA = "a" * 40
ROOT_URL = LISTED_ISSUES_PAGE_URL


class Response:
    def __init__(self, status, payload=b"payload", url=ROOT_URL, error=None):
        self.status, self.payload, self.url, self.error, self.reads = status, payload, url, error, 0
    def getcode(self): return self.status
    def geturl(self): return self.url
    def read(self):
        self.reads += 1
        if self.error: raise self.error
        return self.payload
    def close(self): pass


class Opener:
    def __init__(self, response=None, error=None): self.response, self.error, self.calls = response, error, []
    def open(self, request):
        self.calls.append(request)
        if self.error: raise self.error
        return self.response


class SequenceOpener:
    def __init__(self, outcomes): self.outcomes, self.calls = list(outcomes), []
    def open(self, request):
        self.calls.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException): raise outcome
        return outcome


def conflict_result():
    return acq.finalize_safe_result(acq._base(SHA, "GOVERNANCE_FAILURE", "EXECUTION_BINDING_CONFLICT"))


def test_http_200_returns_exact_bytes_and_requested_get():
    response, opener = Response(200, b"exact"), Opener(Response(200, b"exact"))
    transport = production.HttpTransport(opener)
    result = transport.fetch(ROOT_URL, 1)
    assert (result.http_status, result.payload, result.complete, result.resolved_url) == (200, b"exact", True, ROOT_URL)
    assert len(opener.calls) == 1 and opener.calls[0].get_method() == "GET" and opener.calls[0].full_url == ROOT_URL


def test_redirect_handler_never_creates_followup_and_3xx_body_is_not_read():
    assert production.NoRedirectHandler().redirect_request(None, None, 302, "Found", {}, ROOT_URL) is None
    response, opener = Response(302), Opener(Response(302))
    result = production.HttpTransport(opener).fetch(ROOT_URL, 1)
    assert (result.http_status, result.payload, result.complete) == (302, None, False)
    assert response.reads == 0 and len(opener.calls) == 1


@pytest.mark.parametrize("status", [400, 404, 500, 503])
def test_non_200_response_body_is_never_read(status):
    response, opener = Response(status, b"must-not-read"), Opener(Response(status, b"must-not-read"))
    result = production.HttpTransport(opener).fetch(ROOT_URL, 1)
    assert result.http_status == status and result.payload is None and not result.complete and response.reads == 0


def test_http_error_is_handled_before_url_error_and_transport_errors_are_empty():
    http_error = HTTPError(ROOT_URL, 503, "busy", {}, None)
    result = production.HttpTransport(Opener(error=http_error)).fetch(ROOT_URL, 1)
    assert (result.http_status, result.payload, result.complete, result.resolved_url) == (503, None, False, ROOT_URL)
    result = production.HttpTransport(Opener(error=URLError("offline"))).fetch(ROOT_URL, 1)
    assert (result.http_status, result.payload, result.complete) == (None, None, False)


@pytest.mark.parametrize("error", [ssl.SSLError("tls"), HTTPException("protocol")])
def test_expected_tls_or_http_open_failure_becomes_retriable_incomplete_outcome(error):
    result = production.HttpTransport(Opener(error=error)).fetch(ROOT_URL, 1)
    assert (result.http_status, result.payload, result.complete, result.resolved_url) == (None, None, False, ROOT_URL)


@pytest.mark.parametrize("error", [IncompleteRead(b"part"), TimeoutError(), ConnectionError()])
def test_interrupted_200_has_status_without_payload(error):
    response, opener = Response(200, error=error), Opener(Response(200, error=error))
    result = production.HttpTransport(opener).fetch(ROOT_URL, 1)
    assert (result.http_status, result.payload, result.complete) == (200, None, False)


@pytest.mark.parametrize("error", [ssl.SSLError("tls"), HTTPException("protocol")])
def test_expected_tls_or_http_body_failure_keeps_200_without_partial_payload(error):
    response = Response(200, error=error)
    result = production.HttpTransport(Opener(response)).fetch(ROOT_URL, 1)
    assert (result.http_status, result.payload, result.complete, result.resolved_url) == (200, None, False, ROOT_URL)


def test_unexpected_opener_exception_propagates():
    with pytest.raises(RuntimeError):
        production.HttpTransport(Opener(error=RuntimeError("bug"))).fetch(ROOT_URL, 1)


@pytest.mark.parametrize("error", [ssl.SSLError("tls"), HTTPException("protocol")])
def test_three_expected_open_failures_use_stage1_retry_budget(error):
    transport = production.HttpTransport(SequenceOpener([error, error, error]))
    result = acq.run_pure_acquisition(SHA, ROOT_URL, transport.fetch, lambda *_: pytest.fail("terminal"), lambda *_: None)
    assert (result["result"], result["failure_stage"], result["discovery_root_attempt_count"]) == ("PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED", "ROOT_TRANSPORT", 3)


@pytest.mark.parametrize("error", [ssl.SSLError("tls"), HTTPException("protocol")])
def test_expected_open_failure_then_complete_200_retries_and_succeeds(error):
    transport = production.HttpTransport(SequenceOpener([error, Response(200, b"root")]))
    # The bytes are transport-valid here; Stage-1 endpoint/persistence binding is the next boundary.
    calls = []
    result = acq.run_pure_acquisition(SHA, ROOT_URL, transport.fetch, lambda *_: pytest.fail("terminal"), lambda family, period, payload, url, attempt: calls.append(attempt) or acq.VerifiedLock(family, period, 200, acq.sha256(payload).hexdigest(), len(payload), url))
    assert result["discovery_root_attempt_count"] == 2 and calls == [1] and result["failure_stage"] == "ROOT_LOCATOR"


def test_mismatched_resolved_url_is_rejected_by_stage1_endpoint_binding():
    response = Response(200, b"root", ROOT_URL + "?redirected=1")
    opener = Opener(response)
    calls = []
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: production.HttpTransport(opener).fetch(ROOT_URL, 1), lambda *_: calls.append(True), lambda *_: None)
    assert result["failure_stage"] == "IMPLEMENTATION_ROOT_TRANSPORT" and calls == []


def test_production_binds_frozen_root_and_publishes_exact_safe_result(tmp_path):
    seen = []
    result = production.run_production(SHA, tmp_path / "state", root_fetch=lambda url, _attempt: seen.append(url) or acq.FetchOutcome(503, None, False, url), terminal_fetch=lambda *_: pytest.fail("terminal"), binding_check=lambda *_: None)
    assert seen == [ROOT_URL] * 3 and result[0]["failure_stage"] == "ROOT_TRANSPORT"
    path = tmp_path / "state" / production.SAFE_RESULT
    assert path.exists() and production._publish_safe_result(tmp_path / "state", result[0]) is None


def test_preflight_binding_failures_write_nothing_or_fetch_nothing(tmp_path, monkeypatch):
    for bad in ("wrong-branch", "wrong-head", "dirty", "wrong-design"):
        state_root, calls = tmp_path / bad, []
        expected = {"branch": production.AUTHORITATIVE_BRANCH, "head": SHA, "status": "", "design": production.DESIGN_BLOB}
        def fake_git(_root, *args):
            key = "branch" if args == ("branch", "--show-current") else "status" if args == ("status", "--porcelain") else "design" if args[0] == "rev-parse" and len(args) == 2 and args[1].startswith("HEAD:") else "head"
            target = {"wrong-branch": "branch", "wrong-head": "head", "dirty": "status", "wrong-design": "design"}[bad]
            return bad if key == target else expected[key]
        monkeypatch.setattr(production, "_git_output", fake_git)
        with pytest.raises(ValueError):
            production.run_production(SHA, state_root, root_fetch=lambda *_: calls.append(True), binding_check=production.check_bindings)
        assert not state_root.exists() and calls == []


def test_cli_success_and_closed_non_success_have_exact_stdout_and_exit_codes(monkeypatch, capsys):
    value = conflict_result()
    monkeypatch.setattr(production, "run_production", lambda *args, **kwargs: (value, acq.canonical_json(value)))
    assert production.main(["--state-root", "private-state", "--implementation-git-sha", SHA]) == 2
    captured = capsys.readouterr()
    assert captured.out == acq.canonical_json(value) + "\n" and captured.err == ""
    value = dict(value); value["result"], value["failure_stage"] = "SUCCESS", "NONE"
    monkeypatch.setattr(production, "run_production", lambda *args, **kwargs: (value, acq.canonical_json(value)))
    assert production.main(["--state-root", "private-state", "--implementation-git-sha", SHA]) == 0
    assert capsys.readouterr().out == acq.canonical_json(value) + "\n"


def test_cli_failure_publication_and_unexpected_exception_emit_only_fixed_marker(monkeypatch, capsys, tmp_path):
    state_root = tmp_path / "publication"
    state_root.mkdir()
    monkeypatch.setattr(production.runtime, "_write_exclusive", lambda *_: False)
    assert production._publish_safe_result(state_root, conflict_result()) is None
    assert not (state_root / production.SAFE_RESULT).exists()
    monkeypatch.setattr(production, "run_production", lambda *args, **kwargs: (conflict_result(), acq.canonical_json(conflict_result())))
    assert production.main(["--state-root", str(tmp_path), "--implementation-git-sha", SHA]) == 2
    assert capsys.readouterr().err == ""
    monkeypatch.setattr(production, "run_production", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("private path")))
    assert production.main(["--state-root", str(tmp_path), "--implementation-git-sha", SHA]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == production.IMPLEMENTATION_FAILURE_MARKER + "\n" and "private path" not in captured.err


def test_second_invocation_does_not_mutate_existing_state(tmp_path):
    state_root = tmp_path / "state"
    first = production.run_production(SHA, state_root, root_fetch=lambda url, _: acq.FetchOutcome(503, None, False, url), binding_check=lambda *_: None)
    receipt = (state_root / "execution-start-receipt.json").read_bytes()
    second = production.run_production(SHA, state_root, root_fetch=lambda *_: pytest.fail("fetch"), binding_check=lambda *_: None)
    assert first[0]["failure_stage"] == "ROOT_TRANSPORT" and second[0]["failure_stage"] == "EXECUTION_BINDING_CONFLICT" and (state_root / "execution-start-receipt.json").read_bytes() == receipt
