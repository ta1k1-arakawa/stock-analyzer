import datetime as dt
import hashlib
import json
import os
import shutil
import urllib.error

import pytest

import src.v9_012_actual_tse_trading_day_authority as cal


MISSING = object()
SMALL_DATES = ["2020-09-30", "2020-10-01", "2020-10-02"]


def page(data=None, *, pagination_key=MISSING):
    value = {"data": [] if data is None else data}
    if pagination_key is not MISSING:
        value["pagination_key"] = pagination_key
    return json.dumps(value, separators=(",", ":"), allow_nan=True).encode("utf-8")


def result_for(request, payload=b'{"data":[]} ', status=200, url=None):
    return cal.PageFetchResult(
        payload,
        status,
        cal.expected_request_url(request) if url is None else url,
    )


def fetcher_for(payloads, calls=None):
    calls = [] if calls is None else calls

    def fetch(request):
        calls.append(request)
        value = payloads[(request.source_key, request.page_index)]
        if isinstance(value, BaseException):
            raise value
        return result_for(request, value)

    return fetch, calls


def use_small_coverage(monkeypatch):
    monkeypatch.setattr(cal, "_coverage_dates", lambda: list(SMALL_DATES))


def valid_source_a_rows():
    return [
        {"Date": "2020-09-30", "HolDiv": "1"},
        {"Date": "2020-10-01", "HolDiv": "1"},
        {"Date": "2020-10-02", "HolDiv": "1"},
    ]


def valid_source_b_rows():
    return [
        {"Date": "2020-09-30", "O": 1.0, "H": 2.0, "L": 0.5, "C": 1.5},
        {"Date": "2020-10-01", "O": None, "H": None, "L": None, "C": None},
        {"Date": "2020-10-02", "O": 2.0, "H": 3.0, "L": 1.5, "C": 2.5},
    ]


def valid_payloads():
    return {
        (cal.SOURCE_A, 1): page(valid_source_a_rows()),
        (cal.SOURCE_B, 1): page(valid_source_b_rows()),
    }


def acquire_valid(tmp_path, monkeypatch):
    use_small_coverage(monkeypatch)
    fetch, calls = fetcher_for(valid_payloads())
    chains, requests = cal.acquire_sources(tmp_path, fetcher=fetch)
    return chains, requests, calls


def locked(source, payload):
    return cal.LockedPage({}, payload, False, None)


def lock_synthetic_page(root, source_key, page_index=1, *, continuation_key=MISSING, data=None):
    request = cal.PageRequest(
        source_key,
        page_index,
        None if page_index == 1 else continuation_key,
    )
    payload = page(data, pagination_key=continuation_key)
    cal.PageLockStore(root, source_key).lock_page(request, result_for(request, payload))


def test_constants_bind_frozen_sources_and_queries():
    assert cal.STUDY == "V9_012_ACTUAL_TSE_TRADING_DAY_AUTHORITY_SUCCESSOR"
    assert cal.SOURCE_A_ENDPOINT == "https://api.jquants.com/v2/markets/calendar"
    assert cal.SOURCE_B_ENDPOINT == "https://api.jquants.com/v2/indices/bars/daily/topix"
    assert cal.base_query_object(cal.SOURCE_A) == {"from": "2017-01-01", "to": "2026-01-31"}
    assert cal.base_query_object(cal.SOURCE_B) == {"from": "2017-01-01", "to": "2026-01-31"}
    assert "hol_div" not in cal.base_query_object(cal.SOURCE_A)


def test_exact_urls_and_query_parameter_binding():
    a = cal.PageRequest(cal.SOURCE_A, 1)
    b = cal.PageRequest(cal.SOURCE_B, 1)
    continuation = cal.PageRequest(cal.SOURCE_A, 2, "server-key")
    assert a.params == {"from": "2017-01-01", "to": "2026-01-31"}
    assert b.params == {"from": "2017-01-01", "to": "2026-01-31"}
    assert cal.expected_request_url(a) == (
        "https://api.jquants.com/v2/markets/calendar?from=2017-01-01&to=2026-01-31"
    )
    assert cal.expected_request_url(b) == (
        "https://api.jquants.com/v2/indices/bars/daily/topix?from=2017-01-01&to=2026-01-31"
    )
    assert cal.expected_request_url(continuation).endswith(
        "&pagination_key=server-key"
    )
    assert "hol_div" not in cal.expected_request_url(a)


def test_page_request_identity_is_source_role_bound_and_paginated():
    a1 = cal.PageRequest(cal.SOURCE_A, 1)
    a2 = cal.PageRequest(cal.SOURCE_A, 2, "server-key")
    b1 = cal.PageRequest(cal.SOURCE_B, 1)
    assert cal.page_request_identity(a1)["continuation_key_sha256"] is None
    assert cal.page_request_identity(a2)["continuation_key_sha256"] == cal.sha256_utf8("server-key")
    assert cal.page_request_identity(a2)["source_role"] == cal.SOURCE_A_ROLE
    assert cal.page_request_identity(b1)["source_role"] == cal.SOURCE_B_ROLE
    assert cal.page_request_identity_sha256(a1) != cal.page_request_identity_sha256(b1)
    assert cal.base_query_sha256(cal.SOURCE_A) == cal.base_query_sha256(cal.SOURCE_B)


def test_raw_bytes_are_locked_before_pagination_inspection(tmp_path, monkeypatch):
    payload = page([], pagination_key="next")
    seen = []
    original = cal._inspect_pagination_envelope

    def inspect(raw):
        seen.append((tmp_path / "source_a" / "raw_pages" / "000001.bin").read_bytes())
        assert (tmp_path / "source_a" / "page_locks" / "000001.json").exists()
        return original(raw)

    monkeypatch.setattr(cal, "_inspect_pagination_envelope", inspect)
    fetch, _calls = fetcher_for({(cal.SOURCE_A, 1): payload, (cal.SOURCE_A, 2): page()})
    cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)
    assert seen[0] == payload


def test_new_lock_has_exact_http_status_schema_and_integer_200(tmp_path):
    fetch, _calls = fetcher_for({(cal.SOURCE_A, 1): page()})
    cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)
    lock_path = tmp_path / "source_a" / "page_locks" / "000001.json"
    record = json.loads(lock_path.read_bytes().decode("utf-8"))
    assert set(record) == cal.LOCK_KEYS
    assert type(record["http_status"]) is int
    assert record["http_status"] == 200
    assert cal.PageLockStore(tmp_path, cal.SOURCE_A).read_locked_chain()[-1].record == record


def test_legacy_lock_without_http_status_fails_closed_without_fetch(tmp_path):
    fetch, _calls = fetcher_for({(cal.SOURCE_A, 1): page()})
    cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)
    lock_path = tmp_path / "source_a" / "page_locks" / "000001.json"
    record = json.loads(lock_path.read_bytes().decode("utf-8"))
    record.pop("http_status")
    lock_path.write_bytes(cal.canonical_json_bytes(record))
    restart_fetch, calls = fetcher_for({(cal.SOURCE_A, 1): page()})
    with pytest.raises(cal.V9012Error, match="PAGE_LOCK_SCHEMA_INVALID"):
        cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=restart_fetch)
    assert calls == []


@pytest.mark.parametrize("bad_status", [201, 404, True])
def test_invalid_persisted_http_status_fails_closed_without_fetch(tmp_path, bad_status):
    fetch, _calls = fetcher_for({(cal.SOURCE_A, 1): page()})
    cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)
    lock_path = tmp_path / "source_a" / "page_locks" / "000001.json"
    record = json.loads(lock_path.read_bytes().decode("utf-8"))
    record["http_status"] = bad_status
    lock_path.write_bytes(cal.canonical_json_bytes(record))
    restart_fetch, calls = fetcher_for({(cal.SOURCE_A, 1): page()})
    with pytest.raises(cal.V9012Error, match="PAGE_LOCK_HTTP_STATUS_INVALID"):
        cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=restart_fetch)
    assert calls == []


def test_bad_source_a_http_status_with_source_b_fails_source_order_before_fetch(tmp_path):
    fetch, _calls = fetcher_for({(cal.SOURCE_A, 1): page()})
    cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)
    lock_path = tmp_path / "source_a" / "page_locks" / "000001.json"
    record = json.loads(lock_path.read_bytes().decode("utf-8"))
    record["http_status"] = 404
    lock_path.write_bytes(cal.canonical_json_bytes(record))
    (tmp_path / "source_b").mkdir()
    restart_fetch, calls = fetcher_for({(cal.SOURCE_A, 1): page(), (cal.SOURCE_B, 1): page()})
    with pytest.raises(cal.V9012Error, match="DURABLE_SOURCE_ORDER_VIOLATION"):
        cal.acquire_sources(tmp_path, fetcher=restart_fetch)
    assert calls == []


def test_http_status_is_not_in_source_chain_manifest_or_hash_domain(tmp_path):
    fetch, _calls = fetcher_for({(cal.SOURCE_A, 1): page()})
    manifest, _requests = cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)
    assert set(manifest["pages"][0]) == cal.SOURCE_CHAIN_PAGE_KEYS
    assert "http_status" not in manifest["pages"][0]
    assert "http_status" not in cal.canonical_json_no_lf(manifest).decode("utf-8")


def test_http_status_is_not_in_canonical_artifact_or_receipt(tmp_path, monkeypatch):
    acquire_valid(tmp_path, monkeypatch)
    result = cal.materialize_sources(
        tmp_path,
        acquisition_design_git_sha="a" * 40,
        acquisition_implementation_git_sha="b" * 40,
    )
    public = result.canonical_bytes + cal.canonical_json_bytes(result.receipt)
    assert b"http_status" not in public


def test_acquisition_does_not_inspect_source_semantics(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "validate_source_a_rows", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(cal, "validate_source_b_rows", lambda *_args: (_ for _ in ()).throw(AssertionError()))
    payloads = {
        (cal.SOURCE_A, 1): page([{"Date": "not-a-date", "HolDiv": "bad"}]),
        (cal.SOURCE_B, 1): page([{"Date": "bad", "O": "not-numeric"}]),
    }
    fetch, _calls = fetcher_for(payloads)
    chains, requests = cal.acquire_sources(tmp_path, fetcher=fetch)
    assert requests == 2
    assert chains[cal.SOURCE_A]["page_count"] == 1


def test_locked_pages_are_never_refetched_and_complete_restart_is_zero_network(tmp_path):
    payloads = {
        (cal.SOURCE_A, 1): page(),
        (cal.SOURCE_B, 1): page(),
    }
    fetch, first_calls = fetcher_for(payloads)
    cal.acquire_sources(tmp_path, fetcher=fetch)
    second_fetch, second_calls = fetcher_for(payloads)
    chains, requests = cal.acquire_sources(tmp_path, fetcher=second_fetch)
    assert requests == 0
    assert second_calls == []
    assert chains[cal.SOURCE_A]["page_count"] == 1


def test_restart_resumes_first_missing_source_a_page_before_source_b(tmp_path):
    first_payload = page([], pagination_key="a-next")
    first_calls = []

    def interrupted(request):
        first_calls.append(request)
        if request.source_key == cal.SOURCE_A and request.page_index == 2:
            raise TimeoutError()
        raise AssertionError("SOURCE_B must not start while SOURCE_A is partial")

    with pytest.raises(cal.V9012Error):
        cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=lambda request: result_for(request, first_payload) if request.page_index == 1 else interrupted(request))
    assert len(first_calls) == cal.MAX_PRE_COMPLETE_ATTEMPTS
    assert all((x.source_key, x.page_index) == (cal.SOURCE_A, 2) for x in first_calls)
    second_payloads = {
        (cal.SOURCE_A, 2): page(),
        (cal.SOURCE_B, 1): page(),
    }
    fetch, calls = fetcher_for(second_payloads)
    cal.acquire_sources(tmp_path, fetcher=fetch)
    assert [(x.source_key, x.page_index) for x in calls] == [(cal.SOURCE_A, 2), (cal.SOURCE_B, 1)]


def test_restart_source_a_complete_continues_only_source_b(tmp_path):
    payloads = {
        (cal.SOURCE_A, 1): page(),
        (cal.SOURCE_B, 1): page([], pagination_key="b-next"),
    }
    calls = []

    def interrupted(request):
        calls.append(request)
        if request.source_key == cal.SOURCE_B and request.page_index == 2:
            raise TimeoutError()
        return result_for(request, payloads[(request.source_key, request.page_index)])

    with pytest.raises(cal.V9012Error):
        cal.acquire_sources(tmp_path, fetcher=interrupted)
    calls.clear()
    resume, resume_calls = fetcher_for({(cal.SOURCE_B, 2): page()})
    cal.acquire_sources(tmp_path, fetcher=resume)
    assert [(x.source_key, x.page_index) for x in resume_calls] == [(cal.SOURCE_B, 2)]


def test_source_order_rejects_source_b_when_source_a_is_absent(tmp_path):
    (tmp_path / "source_b").mkdir()
    fetch, calls = fetcher_for({(cal.SOURCE_A, 1): page(), (cal.SOURCE_B, 1): page()})
    with pytest.raises(cal.V9012Error, match="DURABLE_SOURCE_ORDER_VIOLATION"):
        cal.acquire_sources(tmp_path, fetcher=fetch)
    assert calls == []


def test_source_order_rejects_source_a_partial_with_empty_source_b(tmp_path):
    lock_synthetic_page(tmp_path, cal.SOURCE_A, continuation_key="a-next")
    (tmp_path / "source_b").mkdir()
    fetch, calls = fetcher_for({(cal.SOURCE_A, 2): page(), (cal.SOURCE_B, 1): page()})
    with pytest.raises(cal.V9012Error, match="DURABLE_SOURCE_ORDER_VIOLATION"):
        cal.acquire_sources(tmp_path, fetcher=fetch)
    assert calls == []


def test_source_order_rejects_source_a_partial_with_locked_source_b(tmp_path):
    lock_synthetic_page(tmp_path, cal.SOURCE_A, continuation_key="a-next")
    lock_synthetic_page(tmp_path, cal.SOURCE_B)
    fetch, calls = fetcher_for({(cal.SOURCE_A, 2): page(), (cal.SOURCE_B, 2): page()})
    with pytest.raises(cal.V9012Error, match="DURABLE_SOURCE_ORDER_VIOLATION"):
        cal.acquire_sources(tmp_path, fetcher=fetch)
    assert calls == []


def test_source_order_accepts_source_a_terminal_and_fetches_only_source_b(tmp_path):
    lock_synthetic_page(tmp_path, cal.SOURCE_A)
    fetch, calls = fetcher_for({(cal.SOURCE_B, 1): page()})
    cal.acquire_sources(tmp_path, fetcher=fetch)
    assert [(request.source_key, request.page_index) for request in calls] == [(cal.SOURCE_B, 1)]


def test_source_order_resumes_only_first_missing_source_b_page(tmp_path):
    lock_synthetic_page(tmp_path, cal.SOURCE_A)
    lock_synthetic_page(tmp_path, cal.SOURCE_B, continuation_key="b-next")
    fetch, calls = fetcher_for({(cal.SOURCE_B, 2): page()})
    cal.acquire_sources(tmp_path, fetcher=fetch)
    assert [(request.source_key, request.page_index) for request in calls] == [(cal.SOURCE_B, 2)]


def test_source_order_accepts_both_terminal_chains_with_zero_fetches(tmp_path):
    lock_synthetic_page(tmp_path, cal.SOURCE_A)
    lock_synthetic_page(tmp_path, cal.SOURCE_B)
    fetch, calls = fetcher_for({})
    result, requests = cal.acquire_sources(tmp_path, fetcher=fetch)
    assert result[cal.SOURCE_A]["page_count"] == 1
    assert result[cal.SOURCE_B]["page_count"] == 1
    assert requests == 0
    assert calls == []


def test_source_order_proof_does_not_inspect_source_semantics(tmp_path, monkeypatch):
    lock_synthetic_page(
        tmp_path,
        cal.SOURCE_A,
        data=[{"Date": "not-a-date", "HolDiv": "not-a-holiday-value"}],
    )
    (tmp_path / "source_b").mkdir()
    monkeypatch.setattr(cal, "validate_source_a_rows", lambda *_args: pytest.fail("semantic A inspection"))
    monkeypatch.setattr(cal, "validate_source_b_rows", lambda *_args: pytest.fail("semantic B inspection"))
    state = cal.validate_durable_source_order(tmp_path)
    assert state == {
        "source_a_present": True,
        "source_a_terminal": True,
        "source_b_present": True,
    }


def test_source_identity_cannot_cross_satisfy_lock_state(tmp_path):
    payloads = {(cal.SOURCE_A, 1): page()}
    fetch, _calls = fetcher_for(payloads)
    cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)
    b_root = tmp_path / "source_b"
    (b_root / "raw_pages").mkdir(parents=True)
    (b_root / "page_locks").mkdir()
    shutil.copyfile(tmp_path / "source_a" / "raw_pages" / "000001.bin", b_root / "raw_pages" / "000001.bin")
    shutil.copyfile(tmp_path / "source_a" / "page_locks" / "000001.json", b_root / "page_locks" / "000001.json")
    with pytest.raises(cal.V9012Error, match="PAGE_LOCK_REQUEST_BINDING_MISMATCH"):
        cal.PageLockStore(tmp_path, cal.SOURCE_B).read_locked_chain()


@pytest.mark.parametrize("bad_key", [None, "", [], 1])
def test_malformed_null_empty_pagination_key_fails_closed(tmp_path, bad_key):
    fetch, _calls = fetcher_for({(cal.SOURCE_A, 1): page([], pagination_key=bad_key)})
    with pytest.raises(cal.V9012Error):
        cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)
    assert (tmp_path / "source_a" / "raw_pages" / "000001.bin").exists()


def test_repeated_pagination_key_fails_closed(tmp_path):
    payloads = {
        (cal.SOURCE_A, 1): page([], pagination_key="repeat"),
        (cal.SOURCE_A, 2): page([], pagination_key="repeat"),
    }
    fetch, _calls = fetcher_for(payloads)
    with pytest.raises(cal.V9012Error, match="PAGINATION_KEY_REPEATED"):
        cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)


def test_orphan_corrupt_and_mismatched_state_fail_closed(tmp_path):
    raw_dir = tmp_path / "source_a" / "raw_pages"
    raw_dir.mkdir(parents=True)
    (raw_dir / "000001.bin").write_bytes(b"orphan")
    with pytest.raises(cal.V9012Error, match="INCOMPLETE_PAIR"):
        cal.PageLockStore(tmp_path, cal.SOURCE_A).read_locked_chain()

    clean = tmp_path / "clean"
    fetch, _calls = fetcher_for({(cal.SOURCE_A, 1): page()})
    cal.acquire_source(clean, cal.SOURCE_A, fetcher=fetch)
    (clean / "source_a" / "raw_pages" / "000001.bin").write_bytes(b"changed")
    with pytest.raises(cal.V9012Error, match="PAGE_LOCK_PAYLOAD_MISMATCH"):
        cal.PageLockStore(clean, cal.SOURCE_A).read_locked_chain()


def test_redirect_and_response_url_mismatch_are_nonretryable(tmp_path):
    calls = []

    def redirect(request):
        calls.append(request)
        return result_for(request, b"{}", status=302)

    with pytest.raises(cal.V9012Error):
        cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=redirect)
    assert len(calls) == 1

    calls.clear()
    def mismatch(request):
        calls.append(request)
        return result_for(request, b"{}", url=cal.SOURCE_B_ENDPOINT)

    with pytest.raises(cal.V9012Error):
        cal.acquire_source(tmp_path / "other", cal.SOURCE_A, fetcher=mismatch)
    assert len(calls) == 1


def test_retryable_policy_is_three_attempts_with_frozen_backoff(tmp_path):
    calls = []
    sleeps = []

    def fetch(request):
        calls.append(request)
        raise urllib.error.HTTPError(cal.SOURCE_A_ENDPOINT, 503, "", {}, None)

    with pytest.raises(cal.V9012Error, match="PLUMBING_FAILURE_RETRIABLE"):
        cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch, sleep=sleeps.append)
    assert len(calls) == 3
    assert sleeps == [5.0, 30.0]
    assert cal.MAX_PRE_COMPLETE_ATTEMPTS == 3
    assert cal.FROZEN_BACKOFF_SECONDS == (5, 30)


def test_nonretryable_policy_is_immediate(tmp_path):
    calls = []

    def fetch(request):
        calls.append(request)
        raise urllib.error.HTTPError(cal.SOURCE_A_ENDPOINT, 404, "", {}, None)

    with pytest.raises(cal.V9012Error, match="HTTP_404"):
        cal.acquire_source(tmp_path, cal.SOURCE_A, fetcher=fetch)
    assert len(calls) == 1


def test_source_a_exact_coverage_and_holdiv_validation(monkeypatch):
    use_small_coverage(monkeypatch)
    rows, scheduled = cal.validate_source_a_rows([locked(cal.SOURCE_A, page(valid_source_a_rows()))])
    assert [row["Date"] for row in rows] == SMALL_DATES
    assert scheduled == set(SMALL_DATES)
    with pytest.raises(cal.V9012Error):
        cal.validate_source_a_rows([locked(cal.SOURCE_A, page(valid_source_a_rows()[:-1]))])
    bad = valid_source_a_rows()
    bad[0]["HolDiv"] = "4"
    with pytest.raises(cal.V9012Error):
        cal.validate_source_a_rows([locked(cal.SOURCE_A, page(bad))])


def test_topix_all_finite_values_are_active(monkeypatch):
    use_small_coverage(monkeypatch)
    assert cal.validate_source_b_rows([locked(cal.SOURCE_B, page(valid_source_b_rows()))]) == {
        "2020-09-30", "2020-10-02"
    }


def test_topix_all_null_row_is_inactive(monkeypatch):
    use_small_coverage(monkeypatch)
    rows = [{"Date": date, "O": None, "H": None, "L": None, "C": None} for date in SMALL_DATES]
    assert cal.validate_source_b_rows([locked(cal.SOURCE_B, page(rows))]) == set()


def test_topix_mixed_null_is_data_quality_failure(monkeypatch):
    use_small_coverage(monkeypatch)
    rows = [{"Date": "2020-09-30", "O": 1.0, "H": None, "L": 0.5, "C": 1.0}]
    with pytest.raises(cal.V9012Error, match="DATA_QUALITY_FAILURE"):
        cal.validate_source_b_rows([locked(cal.SOURCE_B, page(rows))])


@pytest.mark.parametrize("bad_value", [True, float("nan"), float("inf"), "1.0"])
def test_topix_bool_nan_inf_and_non_numeric_are_invalid(monkeypatch, bad_value):
    use_small_coverage(monkeypatch)
    row = {"Date": "2020-09-30", "O": bad_value, "H": 2.0, "L": 0.5, "C": 1.5}
    with pytest.raises(cal.V9012Error, match="DATA_QUALITY_FAILURE"):
        cal.validate_source_b_rows([locked(cal.SOURCE_B, page([row]))])


@pytest.mark.parametrize("bad_date", ["2020-09-30", "2016-12-31"])
def test_topix_duplicate_or_out_of_range_date_is_invalid(monkeypatch, bad_date):
    use_small_coverage(monkeypatch)
    rows = [
        {"Date": bad_date, "O": 1.0, "H": 2.0, "L": 0.5, "C": 1.5},
        {"Date": "2020-10-02", "O": 2.0, "H": 3.0, "L": 1.5, "C": 2.5},
    ]
    if bad_date == "2020-09-30":
        rows.append(rows[0].copy())
    with pytest.raises(cal.V9012Error, match="DATA_QUALITY_FAILURE"):
        cal.validate_source_b_rows([locked(cal.SOURCE_B, page(rows))])


def test_exact_exception_set_and_neighbor_sentinel_boundaries():
    assert cal._adjudicate_dates(
        {"2020-09-30", "2020-10-01", "2020-10-02"},
        {"2020-09-30", "2020-10-02"},
    ) == (["2020-10-01"], ["2020-10-01"])
    with pytest.raises(cal.V9012Error, match="ACTUAL_TRADING_DAY_AUTHORITY_FAILURE"):
        cal._adjudicate_dates(
            {"2020-09-30", "2020-10-01", "2020-10-02", "2020-10-05"},
            {"2020-09-30", "2020-10-02"},
        )
    with pytest.raises(cal.V9012Error, match="ACTUAL_TRADING_DAY_AUTHORITY_FAILURE"):
        cal._adjudicate_dates(
            {"2020-09-30", "2020-10-01", "2020-10-02"},
            {"2020-10-01", "2020-10-02"},
        )


def test_valid_materialization_uses_only_topix_active_dates_and_hides_values(tmp_path, monkeypatch):
    acquire_valid(tmp_path, monkeypatch)
    result = cal.materialize_sources(
        tmp_path,
        acquisition_design_git_sha="a" * 40,
        acquisition_implementation_git_sha="b" * 40,
    )
    assert result.canonical_content["trading_dates"] == ["2020-09-30", "2020-10-02"]
    public = result.canonical_bytes + cal.canonical_json_bytes(result.receipt)
    assert b"123.4" not in public
    assert b'"O"' not in public
    assert b"api_key" not in public.lower()


def test_exact_source_chain_hash_domain_and_equal_query_hash():
    payload = page()
    request = cal.PageRequest(cal.SOURCE_A, 1)
    record = {
        "byte_count": len(payload),
        "continuation_issued": False,
        "continuation_key_sha256": None,
        "page_index": 1,
        "page_request_identity_sha256": cal.page_request_identity_sha256(request),
        "payload_sha256": cal.sha256_bytes(payload),
    }
    manifest = {
        "base_query_sha256": cal.base_query_sha256(cal.SOURCE_A),
        "page_count": 1,
        "pages": [record],
        "source_api_identity": cal.SOURCE_A_ENDPOINT,
        "source_role": cal.SOURCE_A_ROLE,
        "terminal_page_index": 1,
    }
    expected = hashlib.sha256(cal.canonical_json_no_lf(manifest)).hexdigest()
    assert cal.source_chain_sha256(manifest, cal.SOURCE_A) == expected
    assert cal.base_query_sha256(cal.SOURCE_A) == cal.base_query_sha256(cal.SOURCE_B)
    b_request = cal.PageRequest(cal.SOURCE_B, 1)
    b_manifest = {
        **manifest,
        "pages": [{**record, "page_request_identity_sha256": cal.page_request_identity_sha256(b_request)}],
        "source_api_identity": cal.SOURCE_B_ENDPOINT,
        "source_role": cal.SOURCE_B_ROLE,
    }
    assert cal.source_chain_sha256(manifest, cal.SOURCE_A) != cal.source_chain_sha256(b_manifest, cal.SOURCE_B)


def test_canonical_artifact_final_lf_external_hash_and_receipt(tmp_path, monkeypatch):
    acquire_valid(tmp_path, monkeypatch)
    result = cal.materialize_sources(
        tmp_path,
        acquisition_design_git_sha="a" * 40,
        acquisition_implementation_git_sha="b" * 40,
    )
    assert result.canonical_bytes.endswith(b"\n")
    assert not result.canonical_bytes.endswith(b"\n\n")
    assert result.canonical_artifact_sha256 == hashlib.sha256(result.canonical_bytes).hexdigest()
    assert b"canonical_artifact_sha256" not in result.canonical_bytes
    assert result.receipt == {
        "schema_version": "V9_012_CANONICAL_HASH_RECEIPT_V1",
        "status": "COMPLETE",
        "canonical_artifact_sha256": result.canonical_artifact_sha256,
    }


def test_canonical_schema_rejects_missing_extra_and_bad_source_query():
    fields = {
        "schema_version": "V9_012_CANONICAL_ACTUAL_TSE_TRADING_DAYS_V1",
        "covered_start": cal.COVERED_START,
        "covered_end": cal.COVERED_END,
        "trading_dates": ["2020-09-30"],
        "scheduled_calendar_source_chain_sha256": "a" * 64,
        "topix_source_chain_sha256": "b" * 64,
        "scheduled_open_count": 1,
        "actual_trading_date_count": 1,
        "expected_exception_dates": ["2020-10-01"],
        "observed_exception_dates": ["2020-10-01"],
        "scheduled_calendar_source_api_identity": cal.SOURCE_A_ENDPOINT,
        "topix_source_api_identity": cal.SOURCE_B_ENDPOINT,
        "scheduled_calendar_base_query_sha256": cal.base_query_sha256(cal.SOURCE_A),
        "topix_base_query_sha256": cal.base_query_sha256(cal.SOURCE_B),
        "acquisition_design_git_sha": "c" * 40,
        "acquisition_implementation_git_sha": "d" * 40,
    }
    assert cal.validate_canonical_content(fields) == fields
    with pytest.raises(cal.V9012Error):
        cal.validate_canonical_content({**fields, "extra": True})
    with pytest.raises(cal.V9012Error):
        cal.validate_canonical_content({key: value for key, value in fields.items() if key != "topix_source_chain_sha256"})
    bad_query = {"from": "2017-01-01", "hol_div": None, "to": "2026-01-31"}
    bad_manifest = {
        "base_query_sha256": cal.sha256_bytes(cal.canonical_json_no_lf(bad_query)),
        "page_count": 1,
        "pages": [],
        "source_api_identity": cal.SOURCE_A_ENDPOINT,
        "source_role": cal.SOURCE_A_ROLE,
        "terminal_page_index": 1,
    }
    with pytest.raises(cal.V9012Error):
        cal.validate_source_chain_manifest(bad_manifest, cal.SOURCE_A)


def test_semantic_failure_cannot_trigger_refetch_or_reset(tmp_path, monkeypatch):
    use_small_coverage(monkeypatch)
    bad_b = page([
        {"Date": "2020-09-30", "O": 1.0, "H": 2.0, "L": 0.5, "C": 1.5},
        {"Date": "2020-10-01", "O": 1.0, "H": 2.0, "L": 0.5, "C": 1.5},
        {"Date": "2020-10-02", "O": 2.0, "H": 3.0, "L": 1.5, "C": 2.5},
    ])
    fetch, calls = fetcher_for({(cal.SOURCE_A, 1): page(valid_source_a_rows()), (cal.SOURCE_B, 1): bad_b})
    cal.acquire_sources(tmp_path, fetcher=fetch)
    monkeypatch.setattr(cal, "fetch_http_page", lambda *_args: pytest.fail("no refetch"))
    with pytest.raises(cal.V9012Error, match="ACTUAL_TRADING_DAY_AUTHORITY_FAILURE"):
        cal.materialize_sources(tmp_path, acquisition_design_git_sha="a" * 40, acquisition_implementation_git_sha="b" * 40)
    assert len(calls) == 2


def test_complete_offline_materialization_requires_no_api_key(tmp_path, monkeypatch):
    acquire_valid(tmp_path, monkeypatch)
    monkeypatch.delenv(cal.API_KEY_ENVIRONMENT_VARIABLE, raising=False)
    result = cal.materialize_sources(
        tmp_path,
        acquisition_design_git_sha="a" * 40,
        acquisition_implementation_git_sha="b" * 40,
    )
    assert result.receipt["status"] == "COMPLETE"


def test_production_credential_lookup_is_only_on_required_network_path(monkeypatch, tmp_path):
    monkeypatch.setattr(cal, "verify_production_preflight", lambda *args, **kwargs: {})
    env_gets = []
    original_get = cal.os.environ.get

    def tracked_get(key, default=None):
        env_gets.append(key)
        return original_get(key, default)

    monkeypatch.setattr(cal.os.environ, "get", tracked_get)
    source_a = tmp_path / "source_a" / "raw_pages"
    source_a.mkdir(parents=True)
    (tmp_path / "source_a" / "page_locks").mkdir()
    source_b = tmp_path / "source_b" / "raw_pages"
    source_b.mkdir(parents=True)
    (tmp_path / "source_b" / "page_locks").mkdir()
    # The state is structurally restartable but empty; the production seam
    # must reach its fetcher before credential lookup is meaningful.
    monkeypatch.setattr(cal, "fetch_http_page", lambda request: (_ for _ in ()).throw(cal.V9012Error("synthetic-stop")))
    with pytest.raises(cal.V9012Error):
        cal.run_production_acquisition(
            tmp_path,
            repo_root=tmp_path,
            expected_implementation_sha="a" * 40,
            confirmation=cal.HUMAN_CONFIRMATION,
        )
    assert cal.API_KEY_ENVIRONMENT_VARIABLE not in env_gets


def test_fetch_http_page_reads_credential_only_for_real_request_and_keeps_it_out_of_url(monkeypatch):
    monkeypatch.setenv(cal.API_KEY_ENVIRONMENT_VARIABLE, "synthetic-secret")
    captured = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return page()

        def getcode(self):
            return 200

        def geturl(self):
            return cal.expected_request_url(cal.PageRequest(cal.SOURCE_A, 1))

    class FakeOpener:
        def open(self, request, timeout):
            captured.append((request, timeout))
            return FakeResponse()

    monkeypatch.setattr(cal.urllib.request, "build_opener", lambda *_args: FakeOpener())
    response = cal.fetch_http_page(cal.PageRequest(cal.SOURCE_A, 1))
    assert response.payload == page()
    request, timeout = captured[0]
    assert timeout == 30.0
    assert request.full_url == cal.expected_request_url(cal.PageRequest(cal.SOURCE_A, 1))
    assert request.get_header("X-api-key") == "synthetic-secret"
    assert "synthetic-secret" not in request.full_url


def test_import_and_materialize_have_no_network_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cal.urllib.request, "build_opener", lambda *_args: pytest.fail("network"))
    monkeypatch.setattr(cal.urllib.request, "urlopen", lambda *_args: pytest.fail("network"))
    use_small_coverage(monkeypatch)
    acquire_valid(tmp_path, monkeypatch)
    cal.materialize_sources(tmp_path, acquisition_design_git_sha="a" * 40, acquisition_implementation_git_sha="b" * 40)
