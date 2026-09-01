import datetime as dt
import hashlib
import importlib
import inspect
import json
import urllib.error

import pytest

import src.v9_011_jquants_trading_calendar as cal


def result(payload=b'{"data":[]}', status=200, url=None):
    if url is None:
        raise AssertionError("successful synthetic responses need an exact request URL")
    return cal.PageFetchResult(payload, status, url)


def result_for(request, payload=b'{"data":[]}', status=200, url=None):
    return result(payload, status, cal.expected_request_url(request) if url is None else url)


def page(data=None, pagination_key=None, *, include_key=False):
    value = {"data": [] if data is None else data}
    if include_key:
        value["pagination_key"] = pagination_key
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def full_rows(overrides=None):
    overrides = overrides or {}
    start = dt.date.fromisoformat(cal.COVERED_START)
    end = dt.date.fromisoformat(cal.COVERED_END)
    rows = []
    for offset in range((end - start).days + 1):
        date_value = (start + dt.timedelta(days=offset)).isoformat()
        rows.append({"Date": date_value, "HolDiv": overrides.get(date_value, "0")})
    return rows


def acquire_from_pages(tmp_path, payloads, *, sleep=None):
    calls = []

    def fetch(request):
        calls.append(request)
        value = payloads[request.page_index - 1]
        if isinstance(value, BaseException):
            raise value
        return result_for(request, value)

    provenance, requests = cal.acquire_page_chain(
        tmp_path, fetcher=fetch, sleep=(sleep or (lambda _seconds: None))
    )
    return provenance, requests, calls


def test_import_and_materialization_have_no_network_path(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("network must not be used")

    monkeypatch.setattr(cal.urllib.request, "build_opener", forbidden)
    monkeypatch.setattr(cal.urllib.request, "urlopen", forbidden)
    importlib.reload(cal)
    payload = page(full_rows({"2017-01-02": "1"}))
    cal.acquire_page_chain(tmp_path, fetcher=lambda request: result_for(request, payload))
    materialized = cal.materialize_calendar(
        tmp_path,
        acquisition_design_git_sha="a" * 40,
        acquisition_implementation_git_sha="b" * 40,
    )
    assert materialized.canonical_hash_receipt["status"] == "COMPLETE"


def test_one_page_query_and_raw_lock_precede_envelope_inspection(tmp_path, monkeypatch):
    payload = page([{"Date": "not-a-date", "HolDiv": "not-a-holiday-value"}])
    calls = []
    original = cal._inspect_pagination_envelope

    def inspect_after_lock(raw):
        calls.append((tmp_path / "raw_pages" / "000001.bin").read_bytes())
        assert (tmp_path / "page_locks" / "000001.json").exists()
        return original(raw)

    monkeypatch.setattr(cal, "_inspect_pagination_envelope", inspect_after_lock)
    provenance, requests, fetch_calls = acquire_from_pages(tmp_path, [payload])
    assert requests == 1
    assert fetch_calls[0].params == {"from": cal.COVERED_START, "to": cal.COVERED_END}
    assert calls == [payload]
    assert provenance["page_count"] == 1
    assert (tmp_path / "raw_pages" / "000001.bin").read_bytes() == payload


def test_multi_page_pagination_preserves_frozen_query_and_server_key(tmp_path):
    first = page([{"Date": "bad", "HolDiv": "bad"}], "server-issued-key", include_key=True)
    second = page([{"Date": "also-bad", "HolDiv": "bad"}])
    provenance, requests, calls = acquire_from_pages(tmp_path, [first, second])
    assert requests == 2
    assert calls[0].params == {"from": cal.COVERED_START, "to": cal.COVERED_END}
    assert calls[1].params == {
        "from": cal.COVERED_START,
        "to": cal.COVERED_END,
        "pagination_key": "server-issued-key",
    }
    assert provenance["pages"][0]["continuation_issued"] is True
    assert provenance["pages"][1]["continuation_issued"] is False
    assert provenance["pages"][0]["continuation_key_sha256"] == cal.sha256_utf8("server-issued-key")


def test_expected_request_urls_are_exact_and_api_key_is_header_only():
    first = cal.PageRequest(1)
    continuation = cal.PageRequest(2, "server-issued-key")
    assert cal.expected_request_url(first) == (
        "https://api.jquants.com/v2/markets/calendar?from=2017-01-01&to=2026-01-31"
    )
    assert cal.expected_request_url(continuation) == (
        "https://api.jquants.com/v2/markets/calendar?from=2017-01-01&to=2026-01-31&pagination_key=server-issued-key"
    )
    request = cal._build_request(first, "synthetic-api-key")
    assert request.full_url == cal.expected_request_url(first)
    assert request.get_header("X-api-key") == "synthetic-api-key"
    assert "synthetic-api-key" not in request.full_url
    accepted = cal._validate_transport_result(
        first,
        result(b"{}", url=cal.expected_request_url(first)),
    )
    assert accepted.resolved_url == cal.expected_request_url(first)


@pytest.mark.parametrize(
    "resolved_url",
    [
        cal.ENDPOINT,
        "http://api.jquants.com/v2/markets/calendar?from=2017-01-01&to=2026-01-31",
        "https://evil.example/v2/markets/calendar?from=2017-01-01&to=2026-01-31",
        "https://api.jquants.com/v2/other?from=2017-01-01&to=2026-01-31",
        "https://api.jquants.com/v2/markets/calendar?from=2017-01-02&to=2026-01-31",
        "https://api.jquants.com/v2/markets/calendar?from=2017-01-01&to=2026-02-01",
        "https://api.jquants.com/v2/markets/calendar?from=2017-01-01&to=2026-01-31&hol_div=0",
        "https://api.jquants.com/v2/markets/calendar?from=2017-01-01&to=2026-01-31&pagination_key=substituted",
        "https://api.jquants.com/v2/markets/calendar?from=2017-01-01&from=2017-01-01&to=2026-01-31",
        "https://api.jquants.com/v2/markets/calendar?from=2017-01-01&to=2026-01-31#fragment",
    ],
)
def test_http_200_requires_exact_request_url(resolved_url):
    request = cal.PageRequest(2, "server-issued-key")
    with pytest.raises(cal.V8CTransportNamedFailure, match="RESPONSE_HOST_MISMATCH") as caught:
        cal._validate_transport_result(request, result(b"{}", url=resolved_url))
    assert "server-issued-key" not in str(caught.value)


def test_exact_continuation_url_is_accepted_by_transport_validation():
    request = cal.PageRequest(2, "server-issued-key")
    accepted = cal._validate_transport_result(
        request,
        result(b"{}", url=cal.expected_request_url(request)),
    )
    assert accepted.resolved_url == cal.expected_request_url(request)


def test_retry_is_three_attempts_with_frozen_backoff(tmp_path):
    failures = [urllib.error.HTTPError(cal.ENDPOINT, 503, "", {}, None)] * 2
    calls = []
    sleeps = []

    def fetch(_request):
        calls.append(1)
        if failures:
            raise failures.pop(0)
        return result_for(_request, page())

    provenance, requests = cal.acquire_page_chain(
        tmp_path, fetcher=fetch, sleep=sleeps.append
    )
    assert provenance["page_count"] == 1
    assert requests == 3
    assert len(calls) == cal.MAX_PRE_COMPLETE_ATTEMPTS
    assert sleeps == [5.0, 30.0]


def test_retry_exhaustion_is_terminal(tmp_path):
    calls = []

    def fetch(_request):
        calls.append(1)
        raise urllib.error.URLError(TimeoutError("synthetic timeout"))

    with pytest.raises(cal.V9011Error, match="PLUMBING_FAILURE_RETRIABLE"):
        cal.acquire_page_chain(tmp_path, fetcher=fetch)
    assert len(calls) == 3


@pytest.mark.parametrize("status", [400, 404])
def test_nonretryable_http_does_not_retry(tmp_path, status):
    calls = []

    def fetch(_request):
        calls.append(1)
        return result_for(_request, b"nonretryable", status)

    with pytest.raises(cal.V9011Error, match=f"HTTP_{status}"):
        cal.acquire_page_chain(tmp_path, fetcher=fetch)
    assert calls == [1]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_are_separate_and_nonretryable(tmp_path, status):
    calls = []

    def fetch(_request):
        calls.append(1)
        return result_for(_request, b"auth", status)

    with pytest.raises(cal.V9011Error, match="AUTH_OR_PLAN_FAILURE"):
        cal.acquire_page_chain(tmp_path, fetcher=fetch)
    assert calls == [1]


@pytest.mark.parametrize("bad_value", [None, "", [], 1])
def test_invalid_pagination_metadata_fails_closed_after_lock(tmp_path, bad_value):
    payload = page([], bad_value, include_key=True)
    with pytest.raises(cal.V9011Error, match="PAGINATION_KEY_INVALID"):
        acquire_from_pages(tmp_path, [payload])
    assert (tmp_path / "raw_pages" / "000001.bin").read_bytes() == payload
    assert (tmp_path / "page_locks" / "000001.json").exists()


def test_repeated_pagination_key_fails_closed_without_page_three(tmp_path):
    first = page([], "repeat", include_key=True)
    second = page([], "repeat", include_key=True)
    calls = []

    def fetch(request):
        calls.append(request.page_index)
        return result_for(request, [first, second][request.page_index - 1])

    with pytest.raises(cal.V9011Error, match="PAGINATION_KEY_REPEATED"):
        cal.acquire_page_chain(tmp_path, fetcher=fetch)
    assert calls == [1, 2]
    assert not (tmp_path / "raw_pages" / "000003.bin").exists()


def test_restart_uses_locked_prefix_and_never_refetches_it(tmp_path):
    first = page([], "next-key", include_key=True)
    failure = urllib.error.HTTPError(cal.ENDPOINT, 404, "", {}, None)
    calls_one = []

    def first_run(request):
        calls_one.append(request.page_index)
        return result_for(request, first) if request.page_index == 1 else (_ for _ in ()).throw(failure)

    with pytest.raises(cal.V9011Error, match="HTTP_404"):
        cal.acquire_page_chain(tmp_path, fetcher=first_run)
    assert calls_one == [1, 2]
    locked_bytes = (tmp_path / "raw_pages" / "000001.bin").read_bytes()
    calls_two = []

    def second_run(request):
        calls_two.append(request)
        return result_for(request, page())

    provenance, requests = cal.acquire_page_chain(tmp_path, fetcher=second_run)
    assert provenance["page_count"] == 2
    assert requests == 1
    assert calls_two[0].page_index == 2
    assert calls_two[0].continuation_key == "next-key"
    assert (tmp_path / "raw_pages" / "000001.bin").read_bytes() == locked_bytes


def test_source_chain_known_hash_vector():
    payload = b"page-vector"
    request = cal.PageRequest(1)
    locked = cal.LockedPage(
        {
            "page_index": 1,
            "page_request_identity_sha256": cal.page_request_identity_sha256(request),
            "byte_count": len(payload),
            "payload_sha256": cal.sha256_bytes(payload),
        },
        payload,
        False,
        None,
    )
    provenance = cal.build_page_chain_provenance([locked])
    source = cal.build_source_chain_manifest(provenance)
    expected_source_bytes = cal.identity_json_bytes(source)
    assert cal.source_chain_sha256(source) == hashlib.sha256(expected_source_bytes).hexdigest()
    assert cal.ENDPOINT_IDENTITY_SHA256 == hashlib.sha256(cal.ENDPOINT.encode()).hexdigest()
    assert cal.BASE_QUERY_IDENTITY_SHA256 == hashlib.sha256(
        b'{"from":"2017-01-01","hol_div":null,"to":"2026-01-31"}'
    ).hexdigest()


def test_projected_hash_known_vector_and_sentinel():
    projected = {
        "covered_end": cal.COVERED_END,
        "covered_start": cal.COVERED_START,
        "rows": full_rows(),
    }
    expected = hashlib.sha256(cal.identity_json_bytes(projected)).hexdigest()
    assert cal.projected_calendar_sha256(projected) == expected
    bad = dict(projected)
    bad["rows"] = [*full_rows({"2020-10-01": "1"})]
    with pytest.raises(cal.V9011Error, match="CALENDAR_SEMANTIC_SENTINEL_FAILURE"):
        cal.validate_projected_calendar(bad)


def test_materialization_hashes_final_lf_without_self_reference(tmp_path):
    rows = full_rows({"2017-01-02": "1", "2017-01-03": "2", "2017-01-04": "3"})
    cal.acquire_page_chain(tmp_path, fetcher=lambda request: result_for(request, page(rows)))
    materialized = cal.materialize_calendar(
        tmp_path,
        acquisition_design_git_sha="a" * 40,
        acquisition_implementation_git_sha="b" * 40,
    )
    assert materialized.canonical_bytes.endswith(b"\n")
    assert materialized.canonical_bytes.endswith(b"\n") and not materialized.canonical_bytes.endswith(b"\n\n")
    assert b"canonical_calendar_sha256" not in materialized.canonical_bytes
    assert materialized.canonical_calendar_sha256 == hashlib.sha256(materialized.canonical_bytes).hexdigest()
    assert set(materialized.canonical_content) == cal.CANONICAL_CONTENT_KEYS
    assert materialized.canonical_content["trading_dates"] == ["2017-01-02", "2017-01-03"]
    assert materialized.canonical_content["source_row_count"] == len(rows)


def test_public_artifacts_exclude_raw_url_and_payload(tmp_path):
    secret = "synthetic-api-key-never-public"
    payload = json.dumps({"data": full_rows(), "secret": secret}, separators=(",", ":")).encode()
    cal.acquire_page_chain(tmp_path / "state", fetcher=lambda request: result_for(request, payload))
    materialized = cal.materialize_calendar(
        tmp_path / "state",
        acquisition_design_git_sha="a" * 40,
        acquisition_implementation_git_sha="b" * 40,
    )
    output = tmp_path / "public"
    cal.write_materialized_artifacts(materialized, output)
    public_bytes = b"".join(path.read_bytes() for path in output.iterdir())
    assert secret.encode() not in public_bytes
    assert cal.ENDPOINT.encode() not in public_bytes
    assert b"pagination_key" not in public_bytes


def test_lock_conflict_is_rejected_without_overwrite(tmp_path):
    store = cal.PageLockStore(tmp_path)
    request = cal.PageRequest(1)
    original = result_for(request, b"original")
    store.lock_page(request, original)
    with pytest.raises(cal.LockConflictError, match="DURABLE_STATE_CONFLICT"):
        store.lock_page(request, result_for(request, b"different"))
    assert (tmp_path / "raw_pages" / "000001.bin").read_bytes() == b"original"


def test_exact_schemas_reject_missing_and_extra_fields(tmp_path):
    cal.acquire_page_chain(tmp_path, fetcher=lambda request: result_for(request, page(full_rows())))
    materialized = cal.materialize_calendar(
        tmp_path,
        acquisition_design_git_sha="a" * 40,
        acquisition_implementation_git_sha="b" * 40,
    )
    for validator, value in [
        (cal.validate_page_chain_provenance, materialized.page_chain_provenance),
        (cal.validate_source_chain_manifest, materialized.source_chain_manifest),
        (cal.validate_projected_calendar, materialized.projected_calendar),
        (cal.validate_canonical_content, materialized.canonical_content),
        (cal.validate_canonical_hash_receipt, materialized.canonical_hash_receipt),
    ]:
        missing = dict(value)
        missing.pop(next(iter(missing)))
        with pytest.raises(cal.V9011Error):
            validator(missing)
        extra = dict(value)
        extra["unexpected"] = True
        with pytest.raises(cal.V9011Error):
            validator(extra)
    invalid_content = dict(materialized.canonical_content)
    invalid_content["trading_dates"] = ["2016-12-31"]
    invalid_content["trading_date_count"] = 1
    with pytest.raises(cal.V9011Error, match="CANONICAL_CONTENT_TRADING_DATES_INVALID"):
        cal.validate_canonical_content(invalid_content)


@pytest.mark.parametrize(
    "mutator, reason",
    [
        (lambda rows: rows[:1] + rows[2:], "PROJECTED_CALENDAR_COVERAGE_INVALID"),
        (lambda rows: rows + [rows[0]], "PROJECTED_CALENDAR_COVERAGE_INVALID"),
        (lambda rows: [{"Date": "2016-12-31", "HolDiv": "0"}] + rows[1:], "PROJECTED_CALENDAR_COVERAGE_INVALID"),
        (lambda rows: [{"Date": "bad", "HolDiv": "0"}] + rows[1:], "PROJECTED_CALENDAR_ROW_VALUE_INVALID"),
        (lambda rows: [{"Date": rows[0]["Date"], "HolDiv": "9"}] + rows[1:], "PROJECTED_CALENDAR_ROW_VALUE_INVALID"),
    ],
)
def test_projected_validation_rejects_bad_coverage_and_values(mutator, reason):
    projected = {"covered_end": cal.COVERED_END, "covered_start": cal.COVERED_START, "rows": mutator(full_rows())}
    with pytest.raises(cal.V9011Error, match=reason):
        cal.validate_projected_calendar(projected)


def test_trading_date_filter_and_sentinel_are_frozen(tmp_path):
    rows = full_rows({"2017-01-02": "1", "2017-01-03": "2", "2017-01-04": "3", "2020-10-01": "0"})
    cal.acquire_page_chain(tmp_path, fetcher=lambda request: result_for(request, page(rows)))
    materialized = cal.materialize_calendar(
        tmp_path,
        acquisition_design_git_sha="a" * 40,
        acquisition_implementation_git_sha="b" * 40,
    )
    assert materialized.canonical_content["trading_dates"] == ["2017-01-02", "2017-01-03"]
    assert materialized.canonical_content["trading_date_count"] == 2


def test_zero_page_chain_materialization_fails_closed(tmp_path):
    with pytest.raises(cal.V9011Error, match="PAGE_CHAIN_EMPTY"):
        cal.materialize_calendar(tmp_path, acquisition_design_git_sha="a" * 40, acquisition_implementation_git_sha="b" * 40)


def test_production_preflight_cannot_bypass_environment_or_provenance(tmp_path, monkeypatch):
    assert "fetcher" not in inspect.signature(cal.run_production_acquisition).parameters
    assert "api_key" not in inspect.signature(cal.run_production_acquisition).parameters
    assert "bypass" not in inspect.signature(cal.run_production_acquisition).parameters
    monkeypatch.setattr(cal, "_credential_exists", lambda: True)
    monkeypatch.setattr(cal, "verify_protected_environment", lambda _root: (_ for _ in ()).throw(cal.V9011Error("ENVIRONMENT_BLOCKED")))
    expected = "b" * 40

    def git_runner(args):
        if args == ["branch", "--show-current"]:
            return cal.AUTHORITATIVE_BRANCH
        if args == ["rev-parse", "HEAD"] or args == ["rev-parse", f"refs/remotes/origin/{cal.AUTHORITATIVE_BRANCH}"]:
            return expected
        if args == ["status", "--porcelain=v1", "--untracked-files=all"]:
            return ""
        if args == ["rev-parse", f"{cal.REVIEWED_DESIGN_GIT_SHA}^{{commit}}"]:
            return cal.REVIEWED_DESIGN_GIT_SHA
        if args == ["rev-parse", f"{cal.REVIEWED_DESIGN_GIT_SHA}:{cal.DESIGN_PATH}"]:
            return cal.REVIEWED_DESIGN_BLOB_SHA
        raise AssertionError(args)

    with pytest.raises(cal.V9011Error, match="ENVIRONMENT_BLOCKED"):
        cal.verify_production_preflight(
            tmp_path,
            tmp_path.parent / "external-root",
            expected_implementation_sha=expected,
            confirmation=cal.HUMAN_CONFIRMATION,
            git_runner=git_runner,
        )


def test_preflight_rejects_bad_sha_confirmation_and_git(tmp_path):
    with pytest.raises(cal.V9011Error, match="IMPLEMENTATION_SHA_INVALID"):
        cal.verify_production_preflight(tmp_path, tmp_path / "external", expected_implementation_sha="bad", confirmation=cal.HUMAN_CONFIRMATION)
    with pytest.raises(cal.V9011Error, match="FRESH_HUMAN_CONFIRMATION_INVALID"):
        cal.verify_production_preflight(tmp_path, tmp_path / "external", expected_implementation_sha="a" * 40, confirmation="wrong")
