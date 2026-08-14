"""V8C Yahoo transport readiness probes (§3, §3.1, §3.2, §4.1).

Implements the single canonical `T1C_TRANSPORT_READINESS_HUMAN_GATE` and
the separate `T2_TRANSPORT_READINESS_HUMAN_GATE`, both against the exact
fixed sentinel sourced from the original trusted V8 `T0`:

    indices=[0, 149, 299]
    probe_start=2025-12-01
    probe_end_exclusive=2025-12-08

Neither probe consumes the corresponding raw-acquisition gate; each is its
own separate one-time-per-authorization human gate
(`src.v8c_human_gate_consumption`'s ``PER_AUTHORIZATION_GATES``), consumed
exactly once per real probe execution regardless of outcome, strictly
before the first real Yahoo request for that execution. A prior readiness
authorization never authorizes a later probe -- a fresh, distinct
``human_authorization_identity`` is required for each execution.

This module resolves only the minimum read-only private membership access
`V8C_HISTORICAL_RESEARCH_DESIGN_DRAFT.md` §3.2 allows: the exact original
trusted V8 `T0` members at indices ``[0, 149, 299]`` from the already-
verified, trust-anchor-bound private V8 partition manifest -- never any
other `T0` member, never `T1C`/`T2`/`T3`/`T_spare` identity. The public
result is aggregate-only (status/counts), never a ticker name, price, raw
payload, or private path.

This module is **not executed** by this implementation phase -- no real
readiness probe has been run; every test exercising it is fake/synthetic
transport-only.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.v7_yahoo_collector import HOST, V7YahooCollectorBlocked, fetch_chart_once
from src.v8_partition import V8PartitionBlocked, read_partition_manifest
from src.v8c_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8CGitProvenanceBlocked,
    resolve_git_blob,
    resolve_verified_v8c_production_git_commit,
)
from src.v8c_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    GATE_T1C_TRANSPORT_READINESS,
    GATE_T2_TRANSPORT_READINESS,
    V8CHumanGateConsumptionBlocked,
    consume_gate_once,
    require_gate_not_yet_consumed,
)
from src.v8c_production_provenance import (
    CANONICAL_PARSER_CLASSIFIER_FILE,
    EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
    EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    V8CProductionProvenanceBlocked,
    read_and_verify_design_freeze_approval,
    read_and_verify_v8_trusted_partition_anchor,
    verify_classifier_blob,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8c_transport import V8CTransportNamedFailure, attempt_with_frozen_retry

SENTINEL_INDICES: tuple[int, ...] = (0, 149, 299)
SENTINEL_PROBE_START = "2025-12-01"
SENTINEL_PROBE_END_EXCLUSIVE = "2025-12-08"
T0_EXPECTED_COUNT = 300

STAGE_GATE = {
    "T1C": GATE_T1C_TRANSPORT_READINESS,
    "T2": GATE_T2_TRANSPORT_READINESS,
}


class V8CReadinessBlocked(RuntimeError):
    """Fail-closed V8C transport-readiness probe error."""

    def __init__(self, reason: str, *, authorization_consumed: bool = False) -> None:
        super().__init__(reason)
        self.reason = reason
        self.authorization_consumed = authorization_consumed


_GIT_OBJECT_MISSING_REASONS = frozenset({"GIT_OBJECT_READ_FAILED", "GIT_BLOB_RESOLUTION_FAILED"})


def _wrap(error: BaseException, missing_reason: str | None = None) -> V8CReadinessBlocked:
    reason = getattr(error, "reason", None)
    if missing_reason is not None and reason in _GIT_OBJECT_MISSING_REASONS:
        return V8CReadinessBlocked(missing_reason)
    if isinstance(reason, str):
        return V8CReadinessBlocked(reason)
    return V8CReadinessBlocked("PROVENANCE_CHECK_FAILED")


def _require_exact_origin(value: object, *, hostname: str) -> str:
    if not isinstance(value, str):
        raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH")
    try:
        parsed = urllib.parse.urlparse(value)
        port = parsed.port
    except ValueError as error:
        raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH")
    return value


class _TrustedYahooRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        try:
            _require_exact_origin(newurl, hostname=HOST)
        except V8CTransportNamedFailure as error:
            raise urllib.error.URLError(error.condition) from error
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_trusted_yahoo_opener(request_obj: Any) -> Any:
    _require_exact_origin(getattr(request_obj, "full_url", None), hostname=HOST)
    opener = urllib.request.build_opener(_TrustedYahooRedirectHandler())
    return opener.open(request_obj)


def _probe_one_sentinel(ticker: str, opener: Callable[[Any], Any], sleep_fn: Callable[[float], None]) -> dict[str, Any]:
    def attempt() -> dict[str, Any]:
        try:
            parsed = fetch_chart_once(ticker, SENTINEL_PROBE_START, SENTINEL_PROBE_END_EXCLUSIVE, opener=opener)
        except V7YahooCollectorBlocked as error:
            if error.reason == "SYMBOL_MISMATCH":
                raise V8CTransportNamedFailure("SYMBOL_MISMATCH") from error
            if error.reason == "RESPONSE_HOST_MISMATCH":
                raise V8CTransportNamedFailure("RESPONSE_HOST_MISMATCH") from error
            raise V8CTransportNamedFailure("PARSER_SCHEMA_FAILURE") from error
        valid_rows = parsed.get("valid_price_rows", [])
        if not isinstance(valid_rows, list) or len(valid_rows) < 1:
            raise V8CTransportNamedFailure("DATA_QUALITY_GATE_FAILURE")
        if not all(isinstance(row.get("trading_date"), str) and row["trading_date"] for row in valid_rows):
            raise V8CTransportNamedFailure("DATA_QUALITY_GATE_FAILURE")
        return {"http_status": 200, "trusted_yahoo_host": True, "response_bytes_received": True, "parser_success": True, "expected_symbol_binding": True, "nonempty_timestamp": True, "valid_price_row_count": len(valid_rows)}

    result, audit = attempt_with_frozen_retry(attempt, sleep_fn=sleep_fn)
    return {"result": result, "audit": audit}


def _execute_transport_readiness_probe(
    *,
    stage: str,
    human_authorization_identity: str,
    partition_manifest_path: str | Path,
    git_commit_resolver: Callable[[], str],
    design_freeze_approval_reader: Callable[[str], Mapping[str, Any]],
    frozen_design_object_verifier: Callable[[], None],
    reviewed_implementation_binder: Callable[[str], Mapping[str, Any]],
    classifier_blob_resolver: Callable[[str], str],
    anchor_reader: Callable[[str], Mapping[str, Any]],
    opener: Callable[[Any], Any],
    sleep_fn: Callable[[float], None],
    clock: Callable[[], datetime],
    consumption_state_root: str | Path,
) -> dict[str, Any]:
    if stage not in STAGE_GATE:
        raise V8CReadinessBlocked("V8C_READINESS_STAGE_INVALID")
    if not isinstance(human_authorization_identity, str) or not human_authorization_identity:
        raise V8CReadinessBlocked("V8C_READINESS_HUMAN_AUTHORIZATION_IDENTITY_INVALID")

    gate = STAGE_GATE[stage]

    # (0.5) fail-fast, read-only: this exact authorization identity must not
    # already have been consumed for this gate/design-commit.
    try:
        require_gate_not_yet_consumed(
            consumption_state_root, gate, EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
            authorization_identity=human_authorization_identity,
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CReadinessBlocked(error.reason) from error

    # (1) repo/provenance
    try:
        verified_head = git_commit_resolver()
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error) from error

    # (2) frozen design object + freeze approval
    try:
        frozen_design_object_verifier()
        design_freeze_approval_reader(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error

    # (3) reviewed implementation binding (future artifact; fails closed today)
    try:
        reviewed_implementation_binder(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error

    # (4) classifier blob -- before opener/gate consumption
    try:
        classifier_blob_sha = classifier_blob_resolver(verified_head)
    except V8CGitProvenanceBlocked as error:
        raise _wrap(error) from error
    try:
        verify_classifier_blob(classifier_blob_sha)
    except V8CProductionProvenanceBlocked as error:
        raise _wrap(error) from error

    # (5) original immutable V8 authority (exact anchor blob)
    try:
        anchor = anchor_reader(verified_head)
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error) from error
    if anchor["authorization_status"] != "AUTHORIZED":
        raise V8CReadinessBlocked("TRUSTED_PARTITION_NOT_AUTHORIZED")

    # (6) minimum read-only private T0 sentinel membership resolution (§3.2)
    try:
        partition_manifest = read_partition_manifest(partition_manifest_path)
    except V8PartitionBlocked as error:
        raise V8CReadinessBlocked(error.reason) from error
    if partition_manifest["manifest_sha256"] != anchor["authorized_partition_manifest_sha256"]:
        raise V8CReadinessBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    if partition_manifest["partition_implementation_git_commit"] != anchor["authorized_partition_implementation_git_commit"]:
        raise V8CReadinessBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")
    if partition_manifest["manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8CReadinessBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    if partition_manifest["partition_implementation_git_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8CReadinessBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")

    assignments = partition_manifest["block_assignments"]
    if not isinstance(assignments, Mapping) or "T0" not in assignments:
        raise V8CReadinessBlocked("PARTITION_BLOCK_ASSIGNMENT_MISSING:T0")
    t0_assignment = assignments["T0"]
    if not isinstance(t0_assignment, list) or len(t0_assignment) != T0_EXPECTED_COUNT:
        raise V8CReadinessBlocked("PARTITION_TICKER_COUNT_INVALID:T0")

    sentinel_tickers = tuple(t0_assignment[index] for index in SENTINEL_INDICES)
    if len(sentinel_tickers) != len(SENTINEL_INDICES) or any(not isinstance(t, str) or not t for t in sentinel_tickers):
        raise V8CReadinessBlocked("SENTINEL_MEMBERSHIP_INVALID")

    # (7) durably, fail-closed, per-authorization consume the readiness
    # gate -- strictly before the first real Yahoo request for this
    # execution, regardless of subsequent probe outcome.
    try:
        consume_gate_once(
            consumption_state_root, gate, EXPECTED_V8C_FROZEN_DESIGN_COMMIT, clock=clock,
            authorization_identity=human_authorization_identity,
        )
    except V8CHumanGateConsumptionBlocked as error:
        raise V8CReadinessBlocked(error.reason) from error

    sentinel_results: list[dict[str, Any]] = []
    all_pass = True
    for ticker in sentinel_tickers:
        try:
            probe = _probe_one_sentinel(ticker, opener, sleep_fn)
            sentinel_results.append({"pass": True, "attempts": probe["audit"]["attempts"]})
        except Exception:  # noqa: BLE001 - a failing sentinel is a readiness BLOCK, not a raised error
            all_pass = False
            sentinel_results.append({"pass": False})

    return {
        "stage": stage,
        "result": "PASS" if all_pass else "BLOCK",
        "all_three_sentinels_required": True,
        "sentinel_count": len(SENTINEL_INDICES),
        "sentinel_pass_count": sum(1 for entry in sentinel_results if entry["pass"]),
        "probe_start": SENTINEL_PROBE_START,
        "probe_end_exclusive": SENTINEL_PROBE_END_EXCLUSIVE,
        "readiness_failure_consumes_acquisition_gate": False,
        "readiness_is_research_opening": False,
    }


def execute_t1c_transport_readiness_probe(
    *, human_authorization_identity: str, partition_manifest_path: str | Path
) -> dict[str, Any]:
    """Sole production entrypoint for `T1C_TRANSPORT_READINESS_HUMAN_GATE`.
    **Not executed** by this implementation phase."""
    return _execute_transport_readiness_probe(
        stage="T1C",
        human_authorization_identity=human_authorization_identity,
        partition_manifest_path=partition_manifest_path,
        git_commit_resolver=lambda: resolve_verified_v8c_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        design_freeze_approval_reader=lambda head: read_and_verify_design_freeze_approval(CANONICAL_REPOSITORY_ROOT, head),
        frozen_design_object_verifier=lambda: verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(CANONICAL_REPOSITORY_ROOT, head),
        classifier_blob_resolver=lambda head: resolve_git_blob(CANONICAL_REPOSITORY_ROOT, head, CANONICAL_PARSER_CLASSIFIER_FILE),
        anchor_reader=lambda head: read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, head),
        opener=_default_trusted_yahoo_opener,
        sleep_fn=time.sleep,
        clock=lambda: datetime.now(timezone.utc),
        consumption_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
    )


def execute_t2_transport_readiness_probe(
    *, human_authorization_identity: str, partition_manifest_path: str | Path
) -> dict[str, Any]:
    """Sole production entrypoint for `T2_TRANSPORT_READINESS_HUMAN_GATE`.
    **Not executed** by this implementation phase."""
    return _execute_transport_readiness_probe(
        stage="T2",
        human_authorization_identity=human_authorization_identity,
        partition_manifest_path=partition_manifest_path,
        git_commit_resolver=lambda: resolve_verified_v8c_production_git_commit(CANONICAL_REPOSITORY_ROOT),
        design_freeze_approval_reader=lambda head: read_and_verify_design_freeze_approval(CANONICAL_REPOSITORY_ROOT, head),
        frozen_design_object_verifier=lambda: verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT),
        reviewed_implementation_binder=lambda head: verify_reviewed_implementation_binding(CANONICAL_REPOSITORY_ROOT, head),
        classifier_blob_resolver=lambda head: resolve_git_blob(CANONICAL_REPOSITORY_ROOT, head, CANONICAL_PARSER_CLASSIFIER_FILE),
        anchor_reader=lambda head: read_and_verify_v8_trusted_partition_anchor(CANONICAL_REPOSITORY_ROOT, head),
        opener=_default_trusted_yahoo_opener,
        sleep_fn=time.sleep,
        clock=lambda: datetime.now(timezone.utc),
        consumption_state_root=CANONICAL_CONSUMPTION_STATE_ROOT,
    )


__all__ = [
    "SENTINEL_INDICES",
    "SENTINEL_PROBE_END_EXCLUSIVE",
    "SENTINEL_PROBE_START",
    "STAGE_GATE",
    "T0_EXPECTED_COUNT",
    "V8CReadinessBlocked",
    "execute_t1c_transport_readiness_probe",
    "execute_t2_transport_readiness_probe",
]
