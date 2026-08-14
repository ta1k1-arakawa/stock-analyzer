from __future__ import annotations

import errno
import socket
import urllib.error

import pytest

from src import v8c_transport as transport


def test_frozen_retry_policy_constants():
    assert transport.MAXIMUM_ATTEMPTS_PER_TICKER == 3
    assert transport.MAXIMUM_RETRIES == 2
    assert transport.BACKOFF_SECONDS == (5, 30)
    assert transport.JITTER is False


def test_retryable_http_codes_are_exactly_the_frozen_set():
    assert transport.RETRYABLE_HTTP_CODES == {408, 425, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------------------
# Classification -- every concrete condition, no substring heuristic
# ---------------------------------------------------------------------------


def _http_error(code):
    return urllib.error.HTTPError("https://example.com", code, "msg", {}, None)


@pytest.mark.parametrize("code", [408, 425, 429, 500, 502, 503, 504])
def test_retryable_http_codes_classified_retryable(code):
    label, retryable = transport.classify_transport_exception(_http_error(code))
    assert label == f"HTTP_{code}"
    assert retryable is True


@pytest.mark.parametrize("code", [400, 401, 403, 404, 410, 422])
def test_nonretryable_http_codes_classified_nonretryable(code):
    label, retryable = transport.classify_transport_exception(_http_error(code))
    assert label == f"HTTP_{code}"
    assert retryable is False


def test_unlisted_http_code_is_nonretryable_fail_closed():
    label, retryable = transport.classify_transport_exception(_http_error(451))
    assert label == "HTTP_451"
    assert retryable is False


def test_timeout_error_is_network_timeout_retryable():
    label, retryable = transport.classify_transport_exception(TimeoutError("timed out"))
    assert label == "NETWORK_TIMEOUT"
    assert retryable is True


def test_socket_timeout_is_network_timeout_retryable():
    label, retryable = transport.classify_transport_exception(socket.timeout())
    assert label == "NETWORK_TIMEOUT"
    assert retryable is True


def test_urlerror_wrapping_timeout_is_network_timeout_retryable():
    error = urllib.error.URLError(TimeoutError("timed out"))
    label, retryable = transport.classify_transport_exception(error)
    assert label == "NETWORK_TIMEOUT"
    assert retryable is True


def test_connection_reset_error_is_retryable():
    label, retryable = transport.classify_transport_exception(ConnectionResetError())
    assert label == "CONNECTION_RESET"
    assert retryable is True


def test_oserror_with_econnreset_errno_is_retryable():
    error = OSError()
    error.errno = errno.ECONNRESET
    label, retryable = transport.classify_transport_exception(error)
    assert label == "CONNECTION_RESET"
    assert retryable is True


def test_oserror_with_other_errno_is_unknown_nonretryable():
    error = OSError()
    error.errno = errno.EACCES
    label, retryable = transport.classify_transport_exception(error)
    assert label == "UNKNOWN_FAIL_CLOSED_NONRETRYABLE"
    assert retryable is False


def test_urlerror_wrapping_connection_reset_is_retryable():
    inner = OSError()
    inner.errno = errno.ECONNRESET
    error = urllib.error.URLError(inner)
    label, retryable = transport.classify_transport_exception(error)
    assert label == "CONNECTION_RESET"
    assert retryable is True


def test_temporary_dns_failure_eai_again_is_retryable():
    error = socket.gaierror()
    error.errno = socket.EAI_AGAIN
    label, retryable = transport.classify_transport_exception(error)
    assert label == "TEMPORARY_DNS_FAILURE"
    assert retryable is True


def test_urlerror_wrapping_temporary_dns_failure_is_retryable():
    inner = socket.gaierror()
    inner.errno = socket.EAI_AGAIN
    error = urllib.error.URLError(inner)
    label, retryable = transport.classify_transport_exception(error)
    assert label == "TEMPORARY_DNS_FAILURE"
    assert retryable is True


def test_permanent_dns_failure_is_nonretryable():
    error = socket.gaierror()
    error.errno = socket.EAI_NONAME
    label, retryable = transport.classify_transport_exception(error)
    assert label == "UNKNOWN_FAIL_CLOSED_NONRETRYABLE"
    assert retryable is False


def test_unknown_exception_type_fails_closed_nonretryable():
    label, retryable = transport.classify_transport_exception(ValueError("weird error"))
    assert label == "UNKNOWN_FAIL_CLOSED_NONRETRYABLE"
    assert retryable is False


def test_classification_never_inspects_message_substring():
    # An exception whose message contains "timeout"/"reset"/"429" as text
    # must NOT be reclassified based on that text -- only concrete type/attrs.
    error = ValueError("connection reset: timeout after 429 retries")
    label, retryable = transport.classify_transport_exception(error)
    assert label == "UNKNOWN_FAIL_CLOSED_NONRETRYABLE"
    assert retryable is False


def test_named_condition_classification_is_always_nonretryable():
    for name in transport.NONRETRYABLE_CLASSES:
        label, retryable = transport.classify_named_condition(name)
        assert label == name
        assert retryable is False


def test_named_condition_unknown_name_rejected():
    with pytest.raises(transport.V8CTransportBlocked) as excinfo:
        transport.classify_named_condition("NOT_A_REAL_CONDITION")
    assert excinfo.value.reason == "V8C_TRANSPORT_UNKNOWN_NAMED_CONDITION"


# ---------------------------------------------------------------------------
# attempt_with_frozen_retry
# ---------------------------------------------------------------------------


def _sleeps_recorder():
    calls = []

    def sleep_fn(seconds):
        calls.append(seconds)

    return calls, sleep_fn


def test_success_on_first_attempt_no_sleep_no_retry():
    calls, sleep_fn = _sleeps_recorder()
    result, audit = transport.attempt_with_frozen_retry(lambda: "ok", sleep_fn=sleep_fn)
    assert result == "ok"
    assert audit["attempts"] == 1
    assert audit["retry_count"] == 0
    assert calls == []


def test_nonretryable_failure_never_retries():
    calls, sleep_fn = _sleeps_recorder()
    attempts = []

    def attempt():
        attempts.append(1)
        raise _http_error(403)

    with pytest.raises(urllib.error.HTTPError):
        transport.attempt_with_frozen_retry(attempt, sleep_fn=sleep_fn)
    assert len(attempts) == 1
    assert calls == []


def test_retryable_failure_retries_up_to_frozen_maximum_then_terminal():
    calls, sleep_fn = _sleeps_recorder()
    attempts = []

    def attempt():
        attempts.append(1)
        raise _http_error(503)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        transport.attempt_with_frozen_retry(attempt, sleep_fn=sleep_fn)
    assert len(attempts) == transport.MAXIMUM_ATTEMPTS_PER_TICKER
    assert calls == [5.0, 30.0]
    audit = excinfo.value.transport_audit
    assert len(audit) == 3
    assert [entry["classification"] for entry in audit] == ["HTTP_503"] * 3
    assert all(entry["retryable"] is True for entry in audit)


def test_retry_succeeds_on_second_attempt():
    calls, sleep_fn = _sleeps_recorder()
    attempts = []

    def attempt():
        attempts.append(1)
        if len(attempts) == 1:
            raise _http_error(500)
        return "recovered"

    result, audit = transport.attempt_with_frozen_retry(attempt, sleep_fn=sleep_fn)
    assert result == "recovered"
    assert audit["attempts"] == 2
    assert audit["retry_count"] == 1
    assert calls == [5.0]


def test_no_jitter_backoff_is_exactly_the_frozen_values():
    calls, sleep_fn = _sleeps_recorder()

    def attempt():
        raise ConnectionResetError()

    with pytest.raises(ConnectionResetError):
        transport.attempt_with_frozen_retry(attempt, sleep_fn=sleep_fn)
    assert calls == [5, 30]


def test_named_failure_used_for_transport_named_conditions():
    calls, sleep_fn = _sleeps_recorder()
    attempts = []

    def attempt():
        attempts.append(1)
        raise transport.V8CTransportNamedFailure("SYMBOL_MISMATCH")

    with pytest.raises(transport.V8CTransportNamedFailure) as excinfo:
        transport.attempt_with_frozen_retry(attempt, sleep_fn=sleep_fn)
    assert len(attempts) == 1  # nonretryable, never retried
    assert calls == []
    assert excinfo.value.transport_audit[0]["classification"] == "SYMBOL_MISMATCH"
    assert excinfo.value.transport_audit[0]["retryable"] is False


def test_unmapped_exception_fails_closed_after_one_attempt():
    calls, sleep_fn = _sleeps_recorder()
    attempts = []

    def attempt():
        attempts.append(1)
        raise ValueError("totally unexpected")

    with pytest.raises(ValueError) as excinfo:
        transport.attempt_with_frozen_retry(attempt, sleep_fn=sleep_fn)
    assert len(attempts) == 1
    assert calls == []
    assert excinfo.value.transport_audit[0]["classification"] == "UNKNOWN_FAIL_CLOSED_NONRETRYABLE"


def test_request_fingerprint_identical_across_retries_by_construction():
    """The wrapper never varies the closed-over ticker/window/parameters --
    every attempt is the exact same zero-arg callable, called repeatedly."""
    seen_identity = []

    def attempt():
        seen_identity.append(id(attempt))
        raise _http_error(500)

    calls, sleep_fn = _sleeps_recorder()
    with pytest.raises(urllib.error.HTTPError):
        transport.attempt_with_frozen_retry(attempt, sleep_fn=sleep_fn)
    assert len(set(seen_identity)) == 1
    assert len(seen_identity) == 3


def test_sleep_backoff_index_bounds():
    with pytest.raises(transport.V8CTransportBlocked):
        transport.sleep_backoff(-1)
    with pytest.raises(transport.V8CTransportBlocked):
        transport.sleep_backoff(2)
