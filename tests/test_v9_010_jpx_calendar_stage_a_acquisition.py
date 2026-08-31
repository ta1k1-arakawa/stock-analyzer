from __future__ import annotations

import importlib
import inspect
import json
import urllib.error
from pathlib import Path

import pytest

import src.v9_010_jpx_calendar_stage_a_acquisition as acq
from src.v8c_transport import V8CTransportNamedFailure


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / acq.MANIFEST_FILENAME


def response(url: str, payload: bytes = b"opaque locked bytes", status: int = 200) -> acq.FetchResult:
    return acq.FetchResult(payload=payload, http_status=status, resolved_url=url)


def test_manifest_is_exact_109_slots_and_frozen_digest():
    manifest, digest = acq.load_manifest(MANIFEST_PATH)
    assert len(manifest) == 109
    assert digest == acq.MANIFEST_SHA256
    assert [item["source_slot"] for item in manifest] == list(acq._month_slots())
    assert manifest[0]["source_url"] == "https://www.jpx.co.jp/calendar/201701.html"
    assert manifest[-1]["source_url"] == "https://www.jpx.co.jp/calendar/202601.html"
    assert all(item["source_url"].startswith("https://www.jpx.co.jp/") for item in manifest)
    assert all(item["source_url_sha256"] == acq.sha256_bytes(item["source_url"].encode()) for item in manifest)
    assert all(set(item) == {"source_slot", "source_url", "source_url_sha256"} for item in manifest)
    assert acq.FALLBACK_SOURCE_OBJECTS == 0


@pytest.mark.parametrize("mutation", [
    lambda m: m[:-1],
    lambda m: m + [dict(m[-1])],
    lambda m: [m[1], m[0], *m[2:]],
    lambda m: [{**m[0], "source_url": m[0]["source_url"].replace("www.jpx.co.jp", "evil.example")}, *m[1:]],
    lambda m: [{**m[0], "source_url": m[0]["source_url"].replace("https://", "http://")}, *m[1:]],
    lambda m: [{**m[0], "source_url_sha256": "0" * 64}, *m[1:]],
])
def test_manifest_missing_duplicate_extra_reordered_identity_or_digest_rejected(tmp_path, mutation):
    original = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    bad = mutation(original)
    path = tmp_path / acq.MANIFEST_FILENAME
    path.write_bytes(acq.canonical_json_bytes(bad))
    with pytest.raises(acq.StageAError):
        acq.load_manifest(path)


def test_import_and_offline_calls_have_zero_network(monkeypatch):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("network attempted")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    monkeypatch.setattr("urllib.request.build_opener", forbidden)
    importlib.reload(acq)
    assert calls == []
    assert acq.expected_manifest()[0]["source_slot"] == "2017-01"


def _valid_readiness():
    return {
        "REAL_EXECUTION_ENVIRONMENT_READY": True,
        "REAL_EXECUTION_ENVIRONMENT_FROZEN": True,
        "INTERPRETER_MATCH": True,
        "PYTHON_PATCH_MATCH": True,
        "DEPENDENCY_READINESS": "PASS",
        "JPX_XLS_PARSER_SYNTHETIC_PROBE": "PASS",
        "TLS_STDLIB_PROBE": "PASS",
        "TRUSTED_HOST_REQUEST_CONSTRUCTION_PROBE": "PASS",
        "FILESYSTEM_PROBE": "PASS",
        "ENVIRONMENT_LOCK_CHECK": "PASS",
        "ENVIRONMENT_FREEZE_CHECK": "PASS",
        "REAL_NETWORK_REQUESTS": 0,
        "PRIVATE_READS": 0,
        "GATES_CONSUMED": 0,
    }


def _patch_synthetic_windows_environment(monkeypatch, readiness):
    expected = ROOT / ".venv-real-execution" / "Scripts" / "python.exe"
    monkeypatch.setattr(acq.os, "name", "nt")
    monkeypatch.setattr(acq.sys, "platform", "win32")
    monkeypatch.setattr(acq.sys, "executable", str(expected))
    monkeypatch.setattr(acq, "_run_real_readiness_checks", lambda: readiness)


@pytest.mark.parametrize("field, value, reason", [
    ("REAL_EXECUTION_ENVIRONMENT_READY", False, "PROTECTED_ENVIRONMENT_READINESS_NOT_PASS"),
    ("REAL_EXECUTION_ENVIRONMENT_FROZEN", False, "PROTECTED_ENVIRONMENT_READINESS_NOT_PASS"),
    ("ENVIRONMENT_LOCK_CHECK", "FAIL", "PROTECTED_ENVIRONMENT_READINESS_NOT_PASS"),
    ("ENVIRONMENT_FREEZE_CHECK", "FAIL", "PROTECTED_ENVIRONMENT_READINESS_NOT_PASS"),
    ("REAL_NETWORK_REQUESTS", 1, "PROTECTED_ENVIRONMENT_SAFETY_COUNTER_NONZERO"),
    ("PRIVATE_READS", 1, "PROTECTED_ENVIRONMENT_SAFETY_COUNTER_NONZERO"),
    ("GATES_CONSUMED", 1, "PROTECTED_ENVIRONMENT_SAFETY_COUNTER_NONZERO"),
])
def test_protected_environment_readiness_failures_block_before_fetcher(monkeypatch, field, value, reason):
    readiness = _valid_readiness()
    readiness[field] = value
    _patch_synthetic_windows_environment(monkeypatch, readiness)
    opener_calls = []
    monkeypatch.setattr(acq.urllib.request, "build_opener", lambda: opener_calls.append(True))
    with pytest.raises(acq.StageAError, match=reason):
        acq.fetch_http_once("https://www.jpx.co.jp/calendar/201701.html")
    assert opener_calls == []


def test_wrong_interpreter_blocks_before_checker_and_network(monkeypatch):
    readiness_calls = []
    monkeypatch.setattr(acq.os, "name", "nt")
    monkeypatch.setattr(acq.sys, "platform", "win32")
    monkeypatch.setattr(acq.sys, "executable", str(ROOT / ".venv" / "Scripts" / "python.exe"))
    monkeypatch.setattr(acq, "_run_real_readiness_checks", lambda: readiness_calls.append(True))
    network_calls = []
    monkeypatch.setattr(acq.urllib.request, "build_opener", lambda: network_calls.append(True))
    with pytest.raises(acq.StageAError, match="PROTECTED_ENVIRONMENT_WRONG_INTERPRETER"):
        acq.fetch_http_once("https://www.jpx.co.jp/calendar/201701.html")
    assert readiness_calls == [] and network_calls == []


def test_checker_failure_and_malformed_output_block(monkeypatch):
    _patch_synthetic_windows_environment(monkeypatch, {})
    with pytest.raises(acq.StageAError, match="PROTECTED_ENVIRONMENT_CHECKER_OUTPUT_MALFORMED"):
        acq.verify_protected_environment(ROOT)
    monkeypatch.setattr(acq, "_run_real_readiness_checks", lambda: (_ for _ in ()).throw(RuntimeError("checker")))
    with pytest.raises(acq.StageAError, match="PROTECTED_ENVIRONMENT_CHECKER_FAILURE"):
        acq.verify_protected_environment(ROOT)


def test_canonical_valid_synthetic_readiness_allows_preflight_to_continue(monkeypatch):
    readiness = _valid_readiness()
    _patch_synthetic_windows_environment(monkeypatch, readiness)
    result = acq.verify_protected_environment(ROOT)
    assert result["protected_environment_verified"] is True


def test_production_api_has_no_environment_check_bypass_parameter():
    names = set(inspect.signature(acq.run_production_acquisition).parameters)
    assert names.isdisjoint({"readiness_checker", "readiness_result", "bypass_environment_check", "environment_check"})


def test_retry_policy_is_exact_three_attempts_with_frozen_backoff(tmp_path):
    manifest, _ = acq.load_manifest(MANIFEST_PATH)
    sleeps = []
    calls = []

    def fetch(url):
        calls.append(url)
        raise urllib.error.HTTPError(url, 503, "", {}, None)

    with pytest.raises(acq.StageAError) as caught:
        acq.acquire_one(manifest[0], store=acq.RawLockStore(tmp_path / "state"), fetcher=fetch, sleep=sleeps.append)
    assert caught.value.reason == "PLUMBING_FAILURE_RETRIABLE"
    assert caught.value.attempts == acq.MAX_PRE_COMPLETE_ATTEMPTS == 3
    assert len(calls) == 3
    assert sleeps == [5.0, 30.0]
    assert acq.FROZEN_RETRYABLE_CLASSES == {
        "NETWORK_TIMEOUT", "CONNECTION_RESET", "TEMPORARY_DNS_FAILURE",
        "HTTP_408", "HTTP_425", "HTTP_429", "HTTP_500", "HTTP_502",
        "HTTP_503", "HTTP_504",
    }


def test_first_complete_http_200_locks_exact_bytes_and_stops_retrying(tmp_path, monkeypatch):
    manifest, _ = acq.load_manifest(MANIFEST_PATH)
    store = acq.RawLockStore(tmp_path / "state")
    payload = b"not HTML and not parseable; lock this exact byte sequence\x00\xff"
    calls = []

    def fetch(url):
        calls.append(url)
        return response(url, payload)

    monkeypatch.setattr(acq.json, "loads", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("parsed before lock")))
    record, used = acq.acquire_one(manifest[0], store=store, fetcher=fetch)
    assert used == 1 and len(calls) == 1
    assert (tmp_path / "state" / "raw_payloads" / "2017-01.bin").read_bytes() == payload
    assert record["payload_sha256"] == acq.sha256_bytes(payload)
    monkeypatch.undo()
    assert set(json.loads((tmp_path / "state" / "raw_locks" / "2017-01.json").read_text())) == acq.LOCK_SCHEMA_KEYS


def test_retry_exhaustion_and_nonretryable_response_or_error_stop_without_refetch(tmp_path):
    manifest, _ = acq.load_manifest(MANIFEST_PATH)
    for status in (404, 301):
        calls = []

        def fetch(url, status=status):
            calls.append(url)
            return response(url, b"redirect or error", status)

        with pytest.raises(acq.StageAError):
            acq.acquire_one(manifest[0], store=acq.RawLockStore(tmp_path / f"state-{status}"), fetcher=fetch)
        assert len(calls) == 1

    calls = []

    def error_fetch(url):
        calls.append(url)
        raise V8CTransportNamedFailure("UNTRUSTED_REDIRECT")

    with pytest.raises(acq.StageAError) as caught:
        acq.acquire_one(manifest[0], store=acq.RawLockStore(tmp_path / "redirect"), fetcher=error_fetch)
    assert caught.value.reason == "UNTRUSTED_REDIRECT" and len(calls) == 1


def test_redirect_handler_and_manifest_fetch_cannot_follow_off_manifest_url(tmp_path):
    handler = acq._NoRedirectHandler()
    with pytest.raises(V8CTransportNamedFailure) as caught:
        handler.http_error_302(None, None, 302, "redirect", {"Location": "https://evil.example/"})
    assert caught.value.condition == "UNTRUSTED_REDIRECT"
    manifest, _ = acq.load_manifest(MANIFEST_PATH)
    seen = []

    def fetch(url):
        seen.append(url)
        return response(url, b"", 302)

    with pytest.raises(acq.StageAError):
        acq.acquire_one(manifest[0], store=acq.RawLockStore(tmp_path / "state"), fetcher=fetch)
    assert seen == [manifest[0]["source_url"]]


def test_lock_overwrite_and_conflicting_state_rejected(tmp_path):
    manifest, _ = acq.load_manifest(MANIFEST_PATH)
    store = acq.RawLockStore(tmp_path / "state")
    first = response(manifest[0]["source_url"], b"first")
    store.lock(manifest[0], first)
    with pytest.raises(acq.LockConflictError):
        store.lock(manifest[0], response(manifest[0]["source_url"], b"different"))
    (tmp_path / "state" / "raw_payloads" / "2017-01.bin").write_bytes(b"tampered")
    with pytest.raises(acq.StageAError):
        store.read_existing(manifest)


def _synthetic_records(manifest):
    records = []
    for item in manifest:
        payload = ("payload:" + item["source_slot"]).encode("ascii")
        records.append({
            "source_slot": item["source_slot"],
            "source_url_sha256": item["source_url_sha256"],
            "http_status": 200,
            "byte_count": len(payload),
            "payload_sha256": acq.sha256_bytes(payload),
        })
    return records


def test_raw_lock_schema_count_hash_and_determinism():
    manifest, _ = acq.load_manifest(MANIFEST_PATH)
    records = _synthetic_records(manifest)
    digest1 = acq.raw_lock_set_sha256(records, manifest)
    digest2 = acq.raw_lock_set_sha256(list(records), manifest)
    assert len(records) == acq.SOURCE_SLOT_COUNT and digest1 == digest2
    assert digest1 == acq.sha256_bytes(acq.canonical_json_bytes(records))
    for bad in (records[:-1], records + [dict(records[-1])], [records[1], records[0], *records[2:]]):
        with pytest.raises(acq.StageAError):
            acq.raw_lock_set_sha256(bad, manifest)
    wrong = [dict(item) for item in records]
    wrong[0]["byte_count"] = 0
    with pytest.raises(acq.StageAError):
        acq.raw_lock_set_sha256(wrong, manifest)
    with pytest.raises(acq.StageAError):
        acq.validate_raw_lock_record(
            {**records[0], "payload_sha256": "0" * 64}, manifest[0], b"payload:2017-01"
        )


def test_safe_receipt_excludes_raw_urls_and_payloads():
    manifest, _ = acq.load_manifest(MANIFEST_PATH)
    records = _synthetic_records(manifest)
    receipt = acq.build_safe_receipt(records, 109)
    rendered = acq.canonical_json_bytes(receipt).decode("utf-8")
    assert "source_url" not in rendered and "payload:2017-01" not in rendered
    assert set(receipt) == acq.SAFE_RECEIPT_KEYS
    assert receipt["raw_lock_count"] == 109


def test_full_synthetic_acquisition_locks_exactly_109_and_is_restart_safe(tmp_path):
    manifest, _ = acq.load_manifest(MANIFEST_PATH)
    calls = []

    def fetch(url):
        calls.append(url)
        return response(url, ("payload:" + url).encode("ascii"))

    state = tmp_path / "state"
    receipt = acq.acquire_stage_a(state, fetcher=fetch)
    assert receipt["raw_lock_count"] == 109 and receipt["network_request_count"] == 109
    assert len(calls) == 109
    assert acq.raw_lock_set_sha256(
        [json.loads(path.read_text(encoding="utf-8")) for path in sorted((state / "raw_locks").glob("*.json"))],
        manifest,
    ) == receipt["raw_lock_set_sha256"]
    second_calls = []
    second = acq.acquire_stage_a(state, fetcher=lambda url: second_calls.append(url))
    assert second == receipt and second_calls == []


def test_parser_semantic_or_dq_failure_has_no_refetch_path(tmp_path):
    manifest, _ = acq.load_manifest(MANIFEST_PATH)
    calls = []

    def fetch(url):
        calls.append(url)
        return response(url, b"complete bytes")

    store = acq.RawLockStore(tmp_path / "state")
    record, _ = acq.acquire_one(manifest[0], store=store, fetcher=fetch)
    assert record["http_status"] == 200
    reused, used = acq.acquire_one(manifest[0], store=store, fetcher=lambda url: (_ for _ in ()).throw(AssertionError("refetch")))
    assert reused == record and used == 0
    assert len(calls) == 1


def test_provenance_sha_design_manifest_and_git_state_fail_closed(monkeypatch):
    _patch_synthetic_windows_environment(monkeypatch, _valid_readiness())
    with pytest.raises(acq.StageAError, match="IMPLEMENTATION_SHA_INVALID"):
        acq.verify_production_preflight(ROOT, expected_implementation_sha="not-a-sha", confirmation=acq.HUMAN_CONFIRMATION)

    expected_impl = "a" * 40
    values = {
        ("branch", "--show-current"): acq.AUTHORITATIVE_BRANCH,
        ("rev-parse", "HEAD"): expected_impl,
        ("rev-parse", f"refs/remotes/origin/{acq.AUTHORITATIVE_BRANCH}"): expected_impl,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("rev-parse", f"{acq.REVIEWED_DESIGN_GIT_SHA}^{{commit}}"): acq.REVIEWED_DESIGN_GIT_SHA,
        ("rev-parse", f"{acq.REVIEWED_DESIGN_GIT_SHA}:{acq.DESIGN_PATH}"): acq.REVIEWED_DESIGN_BLOB_GIT_SHA,
        ("rev-parse", f"HEAD:{acq.DESIGN_PATH}"): acq.REVIEWED_DESIGN_BLOB_GIT_SHA,
    }

    def git(args):
        return values[tuple(args)]

    result = acq.verify_production_preflight(
        ROOT,
        expected_implementation_sha=expected_impl,
        confirmation=acq.HUMAN_CONFIRMATION,
        git_runner=git,
    )
    assert result["source_manifest_sha256"] == acq.MANIFEST_SHA256
    values[("status", "--porcelain=v1", "--untracked-files=all")] = " M file"
    with pytest.raises(acq.StageAError, match="GIT_WORKTREE_DIRTY"):
        acq.verify_production_preflight(ROOT, expected_implementation_sha=expected_impl, confirmation=acq.HUMAN_CONFIRMATION, git_runner=git)
    values[("status", "--porcelain=v1", "--untracked-files=all")] = ""
    values[("rev-parse", "HEAD")] = "b" * 40
    with pytest.raises(acq.StageAError, match="GIT_PROVENANCE_MISMATCH"):
        acq.verify_production_preflight(ROOT, expected_implementation_sha=expected_impl, confirmation=acq.HUMAN_CONFIRMATION, git_runner=git)


def test_wrong_confirmation_and_wrong_git_branch_are_rejected_before_network():
    with pytest.raises(acq.StageAError, match="FRESH_HUMAN_CONFIRMATION_INVALID"):
        acq.verify_production_preflight(ROOT, expected_implementation_sha="a" * 40, confirmation="stale")


def test_production_raw_state_must_be_external_to_repository():
    with pytest.raises(acq.StageAError, match="RAW_STATE_MUST_BE_EXTERNAL"):
        acq.run_production_acquisition(
            ROOT / "machine-local-state",
            repo_root=ROOT,
            expected_implementation_sha="a" * 40,
            confirmation=acq.HUMAN_CONFIRMATION,
        )
