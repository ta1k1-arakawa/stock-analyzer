from __future__ import annotations

from hashlib import sha256

import pytest

from src import v9_006_f1_semantic_successor_public_acquisition as acq
from src import v9_006_f1_semantic_successor_locator as locator
from src.v9_005_stage_a_jpx_probe import LISTED_ISSUES_PAGE_URL

SHA = "a" * 40
ROOT_URL = LISTED_ISSUES_PAGE_URL
TERM_URL = ROOT_URL.rsplit("/", 1)[0] + "/a.xls"
ROOT_BYTES = b"List of TSE-listed Issues as of previous month-end is available.<p>List of TSE-listed Issues (Jan. 2026)</p><a href='a.xls'>x</a>"


def lock(payload: bytes, period: str, url: str) -> acq.VerifiedLock:
    return acq.VerifiedLock(acq.ROOT_FAMILY, period, 200, sha256(payload).hexdigest(), len(payload), url)


def matrix_value(result: str, stage: str):
    root = lock(b"root", acq.ROOT_PERIOD, ROOT_URL) if acq._ROWS[(result, stage)][0] else None
    terminal = lock(b"terminal", acq.TERMINAL_PERIOD, TERM_URL) if acq._ROWS[(result, stage)][2] else None
    rule = acq._ROWS[(result, stage)][1]
    locator_result = next(iter(rule)) if type(rule) is set else rule
    value = acq._base(SHA, result, stage, root, terminal, root_status=200 if stage == "ROOT_PERSISTENCE_EXHAUSTED" else None, terminal_status=200 if stage == "TERMINAL_PERSISTENCE_EXHAUSTED" else None, root_attempts=1 if root or stage in {"ROOT_TRANSPORT", "ROOT_PERSISTENCE_EXHAUSTED", "IMPLEMENTATION_ROOT_TRANSPORT"} else 0, terminal_attempts=1 if terminal or stage in {"TERMINAL_TRANSPORT", "TERMINAL_PERSISTENCE_EXHAUSTED", "IMPLEMENTATION_TERMINAL_TRANSPORT"} else 0, locator_result=locator_result, locator_hash="b" * 64 if locator_result else None)
    if stage == "ROOT_TRANSPORT": value["discovery_root_attempt_count"] = value["network_request_count"] = 3
    if stage == "TERMINAL_TRANSPORT": value["terminal_attempt_count"] = 3; value["network_request_count"] = value["discovery_root_attempt_count"] + 3
    return acq.finalize_safe_result(value)


def rehash(value):
    value["structural_evidence_sha256"] = sha256(acq.canonical_json({key: item for key, item in value.items() if key != "structural_evidence_sha256"}).encode("utf-8")).hexdigest()
    return value


@pytest.mark.parametrize("pair", sorted(acq._ROWS))
def test_every_matrix_row_has_a_canonical_valid_fixture(pair):
    acq.validate_safe_acquisition_result(matrix_value(*pair))


def test_validator_rejects_bool_wrong_digest_and_wrong_provenance():
    value = matrix_value("SUCCESS", "NONE")
    for key, bad in (("discovery_root_attempt_count", True), ("raw_lock_set_sha256", "0" * 64), ("safe_provenance_verified", False)):
        altered = dict(value); altered[key] = bad
        with pytest.raises(ValueError): acq.validate_safe_acquisition_result(altered)


@pytest.mark.parametrize("pair", sorted(acq._ROWS))
def test_validator_accepts_exactly_each_row_attempt_domain(pair):
    domains = acq._ATTEMPT_DOMAINS[pair]
    for field, domain in zip(("discovery_root_attempt_count", "terminal_attempt_count"), domains):
        for attempts in range(5):
            value = matrix_value(*pair)
            value[field] = attempts
            value["network_request_count"] = value["discovery_root_attempt_count"] + value["terminal_attempt_count"]
            rehash(value)
            if domain[0] <= attempts <= domain[1]:
                acq.validate_safe_acquisition_result(value)
            else:
                with pytest.raises(ValueError): acq.validate_safe_acquisition_result(value)


@pytest.mark.parametrize("pair", sorted(acq._ROWS))
def test_validator_rejects_no_attempt_status_and_unlocked_payload_evidence(pair):
    value = matrix_value(*pair)
    for field, status in (("discovery_root_attempt_count", "discovery_root_http_status"), ("terminal_attempt_count", "terminal_http_status")):
        if acq._ATTEMPT_DOMAINS[pair][0 if field.startswith("discovery") else 1] == (0, 0):
            altered = dict(value); altered[status] = 503; rehash(altered)
            with pytest.raises(ValueError): acq.validate_safe_acquisition_result(altered)
    for prefix in ("discovery_root", "terminal"):
        if not value[f"{prefix}_locked"]:
            for field, bad in ((f"{prefix}_payload_sha256", "0" * 64), (f"{prefix}_byte_length", 1)):
                altered = dict(value); altered[field] = bad; rehash(altered)
                with pytest.raises(ValueError): acq.validate_safe_acquisition_result(altered)


def test_persistence_exhaustion_and_success_zero_attempt_mutations_are_rejected():
    for pair, prefix in ((("GOVERNANCE_FAILURE", "ROOT_PERSISTENCE_EXHAUSTED"), "discovery_root"), (("GOVERNANCE_FAILURE", "TERMINAL_PERSISTENCE_EXHAUSTED"), "terminal")):
        value = matrix_value(*pair); value[f"{prefix}_payload_sha256"] = "0" * 64; value[f"{prefix}_byte_length"] = 1; rehash(value)
        with pytest.raises(ValueError): acq.validate_safe_acquisition_result(value)
    for field in ("discovery_root_attempt_count", "terminal_attempt_count"):
        value = matrix_value("SUCCESS", "NONE"); value[field] = 0; value["network_request_count"] = value["discovery_root_attempt_count"] + value["terminal_attempt_count"]; rehash(value)
        with pytest.raises(ValueError): acq.validate_safe_acquisition_result(value)


def outcome(status, payload=None, complete=False, url=ROOT_URL): return acq.FetchOutcome(status, payload, complete, url)


def persist_ok(calls):
    def persist(family, period, payload, url, attempt):
        calls.append((period, id(payload), attempt)); return lock(payload, period, url)
    return persist


def locator_success(payload, url, digest, length): return locator.run_fresh_root_locator(payload, url, digest, length)


def test_root_success_first_attempt_and_terminal_success_have_exact_lock_set():
    root_calls, terminal_calls, persisted = [], [], []
    def rf(url, attempt): root_calls.append(attempt); return outcome(200, ROOT_BYTES, True, ROOT_URL)
    def tf(url, attempt): terminal_calls.append(attempt); return outcome(200, b"terminal", True, TERM_URL)
    result = acq.run_pure_acquisition(SHA, ROOT_URL, rf, tf, persist_ok(persisted), locator_runner=locator_success)
    assert result["result"] == "SUCCESS" and root_calls == [1] and terminal_calls == [1]
    assert result["raw_lock_set_sha256"] == acq.raw_lock_set_sha256(lock(ROOT_BYTES, acq.ROOT_PERIOD, ROOT_URL), lock(b"terminal", acq.TERMINAL_PERIOD, TERM_URL))


def test_only_exact_frozen_root_endpoint_can_start_transport():
    calls = []
    result = acq.run_pure_acquisition(SHA, ROOT_URL.rsplit("/", 1)[0] + "/02.html", lambda *_: calls.append(True), lambda *_: outcome(503), persist_ok([]))
    assert (result["result"], result["failure_stage"], result["discovery_root_attempt_count"], result["terminal_attempt_count"]) == ("INPUT_BINDING_FAILURE", "PRE_NETWORK_INPUT_BINDING", 0, 0)
    assert calls == []


@pytest.mark.parametrize("resolved_url", [ROOT_URL.rsplit("/", 1)[0] + "/redirect.html", "https://example.invalid/root.html"])
def test_root_complete_payload_with_mismatched_endpoint_is_not_persisted_or_located(resolved_url):
    persisted, located = [], []
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(200, ROOT_BYTES, True, resolved_url), lambda *_: outcome(503), persist_ok(persisted), locator_runner=lambda *_: located.append(True))
    assert (result["result"], result["failure_stage"], result["discovery_root_attempt_count"]) == ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_TRANSPORT", 1)
    assert persisted == [] and located == []


def test_terminal_complete_payload_with_mismatched_endpoint_is_not_persisted():
    persisted = []
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(200, ROOT_BYTES, True, ROOT_URL), lambda *_: outcome(200, b"terminal", True, TERM_URL + "?wrong=1"), persist_ok(persisted), locator_runner=locator_success)
    assert (result["result"], result["failure_stage"], result["terminal_attempt_count"]) == ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_TERMINAL_TRANSPORT", 1)
    assert [period for period, _ident, _attempt in persisted] == [acq.ROOT_PERIOD]


def test_locator_private_url_must_match_its_safe_selected_url_hash_before_terminal_fetch():
    terminal_calls = []
    def mismatched_private_url(*args):
        safe, _ = locator_success(*args)
        return safe, ROOT_URL
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(200, ROOT_BYTES, True, ROOT_URL), lambda *_: terminal_calls.append(True), persist_ok([]), locator_runner=mismatched_private_url)
    assert (result["result"], result["failure_stage"], result["terminal_attempt_count"]) == ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_POST_LOCATOR_PRE_TERMINAL", 0)
    assert terminal_calls == []


@pytest.mark.parametrize("resolved_url", [ROOT_URL.rsplit("/", 1)[0] + "/wrong.html", "https://example.invalid/non200"])
def test_root_noncomplete_off_contract_endpoint_stops_before_status_or_retry(resolved_url):
    fetches, persisted = [], []
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda _url, attempt: fetches.append(attempt) or outcome(503, url=resolved_url), lambda *_: outcome(503), persist_ok(persisted))
    assert (result["result"], result["failure_stage"], result["discovery_root_attempt_count"], result["discovery_root_http_status"]) == ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_TRANSPORT", 1, None)
    assert fetches == [1] and persisted == []


def test_terminal_noncomplete_off_contract_endpoint_stops_without_retry():
    calls = []
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(200, ROOT_BYTES, True, ROOT_URL), lambda _url, attempt: calls.append(attempt) or outcome(503, url=TERM_URL + "?wrong=1"), persist_ok([]), locator_runner=locator_success)
    assert (result["result"], result["failure_stage"], result["terminal_attempt_count"], result["terminal_http_status"]) == ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_TERMINAL_TRANSPORT", 1, None)
    assert calls == [1]


def test_earlier_exact_status_is_retained_when_later_endpoint_mismatches():
    calls = []
    def root_fetch(_url, attempt):
        calls.append(attempt)
        return outcome(503) if attempt == 1 else outcome(200, ROOT_BYTES, True, ROOT_URL + "?wrong=1")
    result = acq.run_pure_acquisition(SHA, ROOT_URL, root_fetch, lambda *_: outcome(503), persist_ok([]))
    assert (result["failure_stage"], result["discovery_root_attempt_count"], result["discovery_root_http_status"]) == ("IMPLEMENTATION_ROOT_TRANSPORT", 2, 503)
    assert calls == [1, 2]


def test_lock_ok_rejects_subclass_or_independently_invalid_lock_url():
    class UrlSubclass(str): pass
    assert acq._lock_ok(lock(ROOT_BYTES, acq.ROOT_PERIOD, ROOT_URL), ROOT_BYTES, acq.ROOT_FAMILY, acq.ROOT_PERIOD, ROOT_URL)
    subclass_lock = lock(ROOT_BYTES, acq.ROOT_PERIOD, UrlSubclass(ROOT_URL))
    invalid_lock = lock(ROOT_BYTES, acq.ROOT_PERIOD, "https://example.invalid/lock")
    assert not acq._lock_ok(subclass_lock, ROOT_BYTES, acq.ROOT_FAMILY, acq.ROOT_PERIOD, ROOT_URL)
    assert not acq._lock_ok(invalid_lock, ROOT_BYTES, acq.ROOT_FAMILY, acq.ROOT_PERIOD, ROOT_URL)


@pytest.mark.parametrize("lock_url", [ROOT_URL.rsplit("/", 1)[0] + "/wrong.html", "https://example.invalid/lock"])
def test_persistence_cannot_legitimize_mismatched_or_off_domain_endpoint(lock_url):
    calls = []
    def mismatched_lock(family, period, payload, url, attempt):
        calls.append(attempt); return lock(payload, period, lock_url)
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(200, ROOT_BYTES, True, ROOT_URL), lambda *_: outcome(503), mismatched_lock)
    assert result["failure_stage"] == "ROOT_PERSISTENCE_EXHAUSTED" and calls == [1, 2, 3]


@pytest.mark.parametrize("success_attempt, delays", [(2, [0, 2]), (3, [0, 2, 5])])
def test_root_transport_retries_with_frozen_delays(success_attempt, delays):
    seen, clock = [], []
    def rf(url, attempt): seen.append(attempt); return outcome(200, ROOT_BYTES, True, ROOT_URL) if attempt == success_attempt else outcome(503)
    result = acq.run_pure_acquisition(SHA, ROOT_URL, rf, lambda *_: outcome(503), persist_ok([]), delay=clock.append, locator_runner=lambda *args: ({**locator_success(*args)[0], "result": "SOURCE_OR_DATA_FEASIBILITY_FAILURE", "mechanical_candidate_count": 0, "qualifying_candidate_count": 0, "selected_raw_href_sha256": None, "selected_resolved_url_sha256": None, "structural_evidence_sha256": "0" * 64}, None))
    assert seen == list(range(1, success_attempt + 1)) and clock[:success_attempt] == delays
    assert result["failure_stage"] == "IMPLEMENTATION_ROOT_LOCATOR"


def test_transport_failures_and_callback_exceptions_keep_attempt_count():
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(503), lambda *_: outcome(503), persist_ok([]))
    assert (result["result"], result["failure_stage"], result["discovery_root_attempt_count"]) == ("PLUMBING_FAILURE_RETRY_BUDGET_EXHAUSTED", "ROOT_TRANSPORT", 3)
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda _url, attempt: (_ for _ in ()).throw(RuntimeError()) if attempt == 1 else outcome(503), lambda *_: outcome(503), persist_ok([]))
    assert (result["result"], result["failure_stage"], result["discovery_root_attempt_count"]) == ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_TRANSPORT", 1)


def test_latest_safely_completed_http_status_is_retained_after_transport_exhaustion():
    statuses = iter((503, 404, 502))
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(next(statuses)), lambda *_: outcome(503), persist_ok([]))
    assert result["discovery_root_http_status"] == 502


def test_persistence_retries_same_bytes_and_never_refetches():
    fetches, ids = [], []
    def rf(url, attempt): fetches.append(attempt); return outcome(200, ROOT_BYTES, True, ROOT_URL)
    def persist(family, period, payload, url, attempt):
        ids.append(id(payload)); return lock(payload, period, url) if attempt == 3 else None
    result = acq.run_pure_acquisition(SHA, ROOT_URL, rf, lambda *_: outcome(503, url=TERM_URL), persist, locator_runner=locator_success)
    assert fetches == [1] and ids == [id(ROOT_BYTES)] * 3 and result["failure_stage"] == "TERMINAL_TRANSPORT"


def test_terminal_persistence_exhaustion_never_refetches_terminal():
    terminal_fetches, terminal_ids = [], []
    def persist(family, period, payload, url, attempt):
        if period == acq.TERMINAL_PERIOD:
            terminal_ids.append(id(payload)); return None
        return lock(payload, period, url)
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(200, ROOT_BYTES, True, ROOT_URL), lambda _url, attempt: terminal_fetches.append(attempt) or outcome(200, b"terminal", True, TERM_URL), persist, locator_runner=locator_success)
    assert result["failure_stage"] == "TERMINAL_PERSISTENCE_EXHAUSTED"
    assert terminal_fetches == [1] and terminal_ids == [id(b"terminal")] * 3


@pytest.mark.parametrize("locator_result, expected", [("SOURCE_OR_DATA_FEASIBILITY_FAILURE", ("DATA_QUALITY_FAILURE", "ROOT_LOCATOR")), ("HTML_STRUCTURE_UNSUPPORTED", ("DATA_QUALITY_FAILURE", "ROOT_LOCATOR")), ("INPUT_BINDING_FAILURE", ("INPUT_BINDING_FAILURE", "ROOT_LOCATOR_INPUT_BINDING")), ("SAFE_OUTPUT_VALIDATION_FAILURE", ("IMPLEMENTATION_FAILURE", "IMPLEMENTATION_ROOT_LOCATOR"))])
def test_locator_result_mapping_stops_before_terminal(locator_result, expected):
    calls = []
    safe = {"schema_version": locator.FRESH_SCHEMA_VERSION, "task": locator.TASK, "input_payload_sha256": sha256(ROOT_BYTES).hexdigest(), "input_payload_byte_length": len(ROOT_BYTES), "result": locator_result, "mechanical_candidate_count": 0, "qualifying_candidate_count": 0, "selected_raw_href_sha256": None, "selected_resolved_url_sha256": None, "network_requests": 0, "replacement_locator_authorized": False}
    safe = locator._fresh_finalize(safe)
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(200, ROOT_BYTES, True, ROOT_URL), lambda *_: calls.append(True), persist_ok([]), locator_runner=lambda *_: (safe, None))
    assert (result["result"], result["failure_stage"]) == expected and calls == []


def test_invalid_terminal_url_and_unexpected_locator_are_implementation_failures():
    def bad_locator(*args):
        safe, _ = locator_success(*args); return safe, "https://example.invalid/x.xls"
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(200, ROOT_BYTES, True, ROOT_URL), lambda *_: outcome(503), persist_ok([]), locator_runner=bad_locator)
    assert result["failure_stage"] == "IMPLEMENTATION_POST_LOCATOR_PRE_TERMINAL"
    result = acq.run_pure_acquisition(SHA, ROOT_URL, lambda *_: outcome(200, ROOT_BYTES, True, ROOT_URL), lambda *_: outcome(503), persist_ok([]), locator_runner=lambda *_: (_ for _ in ()).throw(RuntimeError()))
    assert result["failure_stage"] == "IMPLEMENTATION_ROOT_LOCATOR"
