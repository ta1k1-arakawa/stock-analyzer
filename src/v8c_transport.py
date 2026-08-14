"""Frozen V8C transport classification/retry layer (§4, §4.1, §4.3).

Shared by all four Yahoo-request-bearing production stages
(`T1C_transport_readiness`, `T1C_raw_acquisition`, `T2_transport_readiness`,
`T2_raw_acquisition`). Implements exactly the frozen §4.3 concrete Python
exception classification and the frozen §4 retry policy:

    maximum_attempts_per_ticker=3
    maximum_retries=2
    backoff_seconds=[5, 30]
    jitter=false

Classification never uses a substring or message heuristic (§4.3 explicit
prohibition) -- only exact numeric HTTP codes and concrete exception types/
attributes. This module performs no network I/O itself; it wraps a
caller-supplied zero-argument attempt callable, so the ticker, request
window, provider, host, and every request parameter are fixed by whatever
that callable closes over and can never change between retries by
construction.
"""

from __future__ import annotations

import errno
import socket
import time
import urllib.error
from typing import Any, Callable, TypeVar

MAXIMUM_ATTEMPTS_PER_TICKER = 3
MAXIMUM_RETRIES = 2
BACKOFF_SECONDS: tuple[int, ...] = (5, 30)
JITTER = False

RETRYABLE_HTTP_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
NONRETRYABLE_HTTP_CODES = frozenset({400, 401, 403, 404, 410, 422})

RETRYABLE_CLASSES = frozenset({
    "NETWORK_TIMEOUT",
    "CONNECTION_RESET",
    "TEMPORARY_DNS_FAILURE",
    "HTTP_408",
    "HTTP_425",
    "HTTP_429",
    "HTTP_500",
    "HTTP_502",
    "HTTP_503",
    "HTTP_504",
})

NONRETRYABLE_CLASSES = frozenset({
    "HTTP_400",
    "HTTP_401",
    "HTTP_403",
    "HTTP_404",
    "HTTP_410",
    "HTTP_422",
    "UNTRUSTED_REDIRECT",
    "RESPONSE_HOST_MISMATCH",
    "PARSER_SCHEMA_FAILURE",
    "SYMBOL_MISMATCH",
    "DATA_QUALITY_GATE_FAILURE",
    "UNKNOWN_FAIL_CLOSED_NONRETRYABLE",
})

assert len(BACKOFF_SECONDS) == MAXIMUM_RETRIES

T = TypeVar("T")


class V8CTransportBlocked(RuntimeError):
    """Fail-closed V8C transport wrapper error."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_timeout(value: object) -> bool:
    return isinstance(value, (TimeoutError, socket.timeout))


def _connection_reset_errno(value: object) -> bool:
    if isinstance(value, ConnectionResetError):
        return True
    if isinstance(value, OSError) and getattr(value, "errno", None) == errno.ECONNRESET:
        return True
    return False


def _temporary_dns_failure(value: object) -> bool:
    return isinstance(value, socket.gaierror) and getattr(value, "errno", None) == socket.EAI_AGAIN


def _permanent_dns_failure(value: object) -> bool:
    return isinstance(value, socket.gaierror) and not _temporary_dns_failure(value)


def classify_transport_exception(error: BaseException) -> tuple[str, bool]:
    """Classify a concrete Python transport exception into one of the
    frozen §4/§4.3 abstract retry classes. Returns
    ``(classification_label, retryable)``. Never inspects ``str(error)``,
    ``error.args``, or any substring/message content -- only concrete
    exception types and the exact numeric ``.code``/``.errno`` attributes
    §4.3 names.
    """
    if isinstance(error, V8CTransportNamedFailure):
        return classify_named_condition(error.condition)

    if isinstance(error, urllib.error.HTTPError):
        code = error.code
        if isinstance(code, int) and not isinstance(code, bool):
            label = f"HTTP_{code}"
            if code in RETRYABLE_HTTP_CODES:
                return label, True
            return label, False
        return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False

    if _is_timeout(error):
        return "NETWORK_TIMEOUT", True

    if isinstance(error, urllib.error.URLError):
        reason = error.reason
        if _is_timeout(reason):
            return "NETWORK_TIMEOUT", True
        if _temporary_dns_failure(reason):
            return "TEMPORARY_DNS_FAILURE", True
        if _permanent_dns_failure(reason):
            return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False
        if _connection_reset_errno(reason):
            return "CONNECTION_RESET", True
        return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False

    if _temporary_dns_failure(error):
        return "TEMPORARY_DNS_FAILURE", True
    if _permanent_dns_failure(error):
        return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False

    if _connection_reset_errno(error):
        return "CONNECTION_RESET", True

    return "UNKNOWN_FAIL_CLOSED_NONRETRYABLE", False


def classify_named_condition(name: str) -> tuple[str, bool]:
    """Classify a caller-determined, already-named condition (e.g. an
    ``UNTRUSTED_REDIRECT``, ``RESPONSE_HOST_MISMATCH``,
    ``PARSER_SCHEMA_FAILURE``, ``SYMBOL_MISMATCH``, or
    ``DATA_QUALITY_GATE_FAILURE`` detected by the caller's own parser/
    transport-security logic, not by this module). These are always
    nonretryable per §4."""
    if name not in NONRETRYABLE_CLASSES:
        raise V8CTransportBlocked("V8C_TRANSPORT_UNKNOWN_NAMED_CONDITION")
    return name, False


def _classification_metadata(error: BaseException, classification: str) -> dict[str, Any]:
    """Return concrete, non-message metadata for the private retry audit."""
    if isinstance(error, urllib.error.HTTPError):
        return {"exception_type": type(error).__name__, "http_code": error.code}
    if isinstance(error, V8CTransportNamedFailure):
        return {"exception_type": type(error).__name__, "named_condition": error.condition}
    reason = getattr(error, "reason", None) if isinstance(error, urllib.error.URLError) else error
    return {
        "exception_type": type(error).__name__,
        "reason_type": type(reason).__name__,
        "errno": getattr(reason, "errno", None),
        "classification": classification,
    }


def sleep_backoff(attempt_index: int, sleep_fn: Callable[[float], None] = time.sleep) -> None:
    """Sleep the frozen backoff for ``attempt_index`` (0-based index into
    the retry, i.e. the wait *before* retry attempt ``attempt_index + 2``).
    No jitter is ever applied (``JITTER = False``)."""
    if attempt_index < 0 or attempt_index >= len(BACKOFF_SECONDS):
        raise V8CTransportBlocked("V8C_TRANSPORT_BACKOFF_INDEX_INVALID")
    sleep_fn(float(BACKOFF_SECONDS[attempt_index]))


def attempt_with_frozen_retry(
    attempt_fn: Callable[[], T],
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
    request_fingerprint: str | None = None,
) -> tuple[T, dict[str, Any]]:
    """Execute ``attempt_fn`` under the frozen §4 retry policy.

    ``attempt_fn`` takes no arguments and must itself construct and issue
    exactly one request per call, using fixed, closed-over ticker/window/
    provider/host/parameters -- this wrapper cannot vary them, by
    construction. On success, returns ``(result, audit)`` where ``audit``
    records every attempt's classification for the frozen §10.1 retry
    audit. On terminal failure (a nonretryable classification, or
    exhaustion of all `MAXIMUM_ATTEMPTS_PER_TICKER` attempts), re-raises
    the final attempt's exception; ``audit`` is attached to it as
    ``transport_audit``.
    """
    audit_attempts: list[dict[str, Any]] = []
    last_error: BaseException | None = None
    for attempt_number in range(1, MAXIMUM_ATTEMPTS_PER_TICKER + 1):
        try:
            result = attempt_fn()
        except V8CTransportNamedFailure as error:
            classification, retryable = classify_named_condition(error.condition)
            audit_attempts.append({"attempt": attempt_number, "classification": classification, "retryable": retryable,
                                   "classification_metadata": _classification_metadata(error, classification),
                                   "request_fingerprint": request_fingerprint})
            last_error = error
            if not retryable or attempt_number == MAXIMUM_ATTEMPTS_PER_TICKER:
                setattr(error, "transport_audit", audit_attempts)
                raise
        except BaseException as error:  # noqa: BLE001 - must classify every exception, including unmapped ones
            classification, retryable = classify_transport_exception(error)
            audit_attempts.append({"attempt": attempt_number, "classification": classification, "retryable": retryable,
                                   "classification_metadata": _classification_metadata(error, classification),
                                   "request_fingerprint": request_fingerprint})
            last_error = error
            if not retryable or attempt_number == MAXIMUM_ATTEMPTS_PER_TICKER:
                setattr(error, "transport_audit", audit_attempts)
                raise
        else:
            audit_attempts.append({"attempt": attempt_number, "classification": "SUCCESS", "retryable": None,
                                   "classification_metadata": {"exception_type": None},
                                   "request_fingerprint": request_fingerprint})
            return result, {
                "attempts": len(audit_attempts),
                "retry_count": len(audit_attempts) - 1,
                "history": audit_attempts,
                "terminal_classification": None,
            }
        sleep_backoff(attempt_number - 1, sleep_fn)
    # Unreachable: the loop above always either returns or raises by the
    # final attempt.
    raise V8CTransportBlocked("V8C_TRANSPORT_RETRY_LOOP_INVARIANT_VIOLATED")


class V8CTransportNamedFailure(RuntimeError):
    """Raised by a caller's own parser/transport-security logic to report
    one of the frozen named nonretryable conditions (``UNTRUSTED_REDIRECT``,
    ``RESPONSE_HOST_MISMATCH``, ``PARSER_SCHEMA_FAILURE``,
    ``SYMBOL_MISMATCH``, ``DATA_QUALITY_GATE_FAILURE``) to
    ``attempt_with_frozen_retry`` for classification, rather than letting
    an arbitrary exception type reach the generic classifier."""

    def __init__(self, condition: str) -> None:
        super().__init__(condition)
        self.condition = condition


__all__ = [
    "BACKOFF_SECONDS",
    "JITTER",
    "MAXIMUM_ATTEMPTS_PER_TICKER",
    "MAXIMUM_RETRIES",
    "NONRETRYABLE_CLASSES",
    "NONRETRYABLE_HTTP_CODES",
    "RETRYABLE_CLASSES",
    "RETRYABLE_HTTP_CODES",
    "V8CTransportBlocked",
    "V8CTransportNamedFailure",
    "attempt_with_frozen_retry",
    "classify_named_condition",
    "classify_transport_exception",
    "sleep_backoff",
]
