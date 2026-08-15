"""V8D synthetic-injectable transport readiness catch paths, plus the
fail-closed production readiness entrypoints
(`V8D_PROD_HIGH_1B_GATE_CONSUMPTION_RECEIPT_BINDING`).

``execute_transport_readiness_probe`` remains transport-only: it requires
an already-derived ``reviewed_implementation_commit`` and an already-
consumed ``gate_binding`` supplied by the caller, and performs no
provenance resolution or gate consumption of its own -- it is the shared
synthetic-injectable catch path used by both tests and the production
entrypoints below.

``execute_t1c_transport_readiness_production`` / ``execute_t2_transport_
readiness_production`` are the sole production entrypoints. Unlike the
transport-only function above, they never accept an authority-bearing
``reviewed_implementation_commit`` from the caller: they resolve verified
V8D Git HEAD (`src.v8d_git_provenance`), verify the frozen design and
freeze approval, derive the reviewed implementation commit through the
HIGH-1A `src.v8d_production_provenance.verify_reviewed_implementation_
binding`, and durably consume the exact applicable one-shot human gate
(`src.v8d_human_gate_consumption`) -- strictly before the first Yahoo
request -- before ever invoking the transport stage.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Any, Mapping

from src.v8d_git_provenance import (
    CANONICAL_REPOSITORY_ROOT,
    V8DGitProvenanceBlocked,
    resolve_verified_v8d_production_git_commit,
)
from src.v8d_human_gate_consumption import (
    CANONICAL_CONSUMPTION_STATE_ROOT,
    V8DHumanGateConsumptionBlocked,
    consume_gate_and_bind,
)
from src.v8d_production_provenance import (
    EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
    EXPECTED_V8_PARTITION_MANIFEST_SHA256,
    EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
    EXPECTED_V8D_FROZEN_DESIGN_COMMIT,
    V8DProductionProvenanceBlocked,
    verify_design_freeze_approval_blob,
    read_and_verify_v8_trusted_partition_anchor,
    verify_frozen_design_object,
    verify_reviewed_implementation_binding,
)
from src.v8d_authority_bridge import V8DAuthorityBridgeBlocked, verify_stage_authority_bridge
from src.v8d_transport import (
    DurableV8DAuditStore,
    READINESS_STAGES,
    SENTINEL_END_EXCLUSIVE,
    SENTINEL_INDICES,
    SENTINEL_START,
    V8DRequestPlan,
    V8DTransportBlocked,
    build_yahoo_request_plan,
    default_trusted_yahoo_opener,
    execute_v8d_stage,
    make_request_fingerprint,
    require_nonempty_quality,
    sha256_url,
)
from src.v8_partition import MANIFEST_FIELDS as V8_MANIFEST_FIELDS


class V8DReadinessBlocked(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


T0_EXPECTED_COUNT = 300
CANONICAL_PRODUCTION_AUDIT_ROOT = CANONICAL_CONSUMPTION_STATE_ROOT.parent / "v8d-transport-audit-state"


def _wrap(error: BaseException) -> V8DReadinessBlocked:
    reason = getattr(error, "reason", None)
    return V8DReadinessBlocked(reason if isinstance(reason, str) else "V8D_READINESS_PROVENANCE_CHECK_FAILED")


def _read_selective_t0_sentinels(partition_manifest_path: str | Path) -> tuple[str, str, tuple[str, ...]]:
    """Read only the three fixed T0 identities from a private manifest.

    The parser walks every JSON value for syntax/schema integrity but only
    decodes the T0 elements at the frozen coordinates. No other private
    assignment member is materialized as a Python string.
    """
    decoder = json.JSONDecoder()

    class _SelectiveManifestParseError(ValueError):
        pass

    def whitespace(raw: str, position: int) -> int:
        while position < len(raw) and raw[position] in " \t\r\n":
            position += 1
        return position

    def string(raw: str, position: int) -> tuple[str, int]:
        position = whitespace(raw, position)
        if position >= len(raw) or raw[position] != '"':
            raise _SelectiveManifestParseError
        value, end = decoder.raw_decode(raw, position)
        if not isinstance(value, str):
            raise _SelectiveManifestParseError
        return value, end

    def skip_value(raw: str, position: int) -> int:
        position = whitespace(raw, position)
        if position >= len(raw):
            raise _SelectiveManifestParseError
        marker = raw[position]
        if marker == '"':
            end = position + 1
            escaped = False
            while end < len(raw):
                char = raw[end]
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    return end + 1
                end += 1
            raise _SelectiveManifestParseError
        if marker == '[':
            position = whitespace(raw, position + 1)
            if position < len(raw) and raw[position] == ']':
                return position + 1
            while True:
                position = skip_value(raw, position)
                position = whitespace(raw, position)
                if position >= len(raw):
                    raise _SelectiveManifestParseError
                if raw[position] == ']':
                    return position + 1
                if raw[position] != ',':
                    raise _SelectiveManifestParseError
                position += 1
        if marker == '{':
            position = whitespace(raw, position + 1)
            keys: set[str] = set()
            if position < len(raw) and raw[position] == '}':
                return position + 1
            while True:
                key, position = string(raw, position)
                if key in keys:
                    raise _SelectiveManifestParseError
                keys.add(key)
                position = whitespace(raw, position)
                if position >= len(raw) or raw[position] != ':':
                    raise _SelectiveManifestParseError
                position = skip_value(raw, position + 1)
                position = whitespace(raw, position)
                if position >= len(raw):
                    raise _SelectiveManifestParseError
                if raw[position] == '}':
                    return position + 1
                if raw[position] != ',':
                    raise _SelectiveManifestParseError
                position += 1
        value, end = decoder.raw_decode(raw, position)
        if isinstance(value, (dict, list, str)):
            raise _SelectiveManifestParseError
        return end

    def selective_t0_array(raw: str, position: int) -> tuple[tuple[str, ...], int]:
        position = whitespace(raw, position)
        if position >= len(raw) or raw[position] != '[':
            raise _SelectiveManifestParseError
        position = whitespace(raw, position + 1)
        selected: list[str] = []
        for index in range(T0_EXPECTED_COUNT):
            if index in SENTINEL_INDICES:
                value, position = string(raw, position)
                if not value:
                    raise _SelectiveManifestParseError
                selected.append(value)
            else:
                position = skip_value(raw, position)
            position = whitespace(raw, position)
            if index < T0_EXPECTED_COUNT - 1:
                if position >= len(raw) or raw[position] != ',':
                    raise _SelectiveManifestParseError
                position = whitespace(raw, position + 1)
            elif position >= len(raw) or raw[position] != ']':
                raise _SelectiveManifestParseError
            else:
                position = whitespace(raw, position + 1)
        return tuple(selected), position

    def selective_assignments(raw: str, position: int) -> tuple[tuple[str, ...], int]:
        position = whitespace(raw, position)
        if position >= len(raw) or raw[position] != '{':
            raise _SelectiveManifestParseError
        position = whitespace(raw, position + 1)
        names: set[str] = set()
        selected: tuple[str, ...] | None = None
        if position < len(raw) and raw[position] == '}':
            raise _SelectiveManifestParseError
        while True:
            name, position = string(raw, position)
            if name in names:
                raise _SelectiveManifestParseError
            names.add(name)
            position = whitespace(raw, position)
            if position >= len(raw) or raw[position] != ':':
                raise _SelectiveManifestParseError
            if name == "T0":
                selected, position = selective_t0_array(raw, position + 1)
            else:
                position = skip_value(raw, position + 1)
            position = whitespace(raw, position)
            if position >= len(raw):
                raise _SelectiveManifestParseError
            if raw[position] == '}':
                position += 1
                break
            if raw[position] != ',':
                raise _SelectiveManifestParseError
            position = whitespace(raw, position + 1)
        if names != {"T0", "T1", "T2", "T3", "T_spare"} or selected is None:
            raise _SelectiveManifestParseError
        return selected, position

    try:
        raw = Path(partition_manifest_path).read_bytes()
        text = raw.decode("utf-8")
        position = whitespace(text, 0)
        if position >= len(text) or text[position] != '{':
            raise _SelectiveManifestParseError
        position = whitespace(text, position + 1)
        seen: set[str] = set()
        manifest_sha: str | None = None
        implementation: str | None = None
        selected: tuple[str, ...] | None = None
        pair_spans: list[tuple[str, int, int]] = []
        while position < len(text) and text[position] != '}':
            key_start = position
            key, position = string(text, position)
            if key in seen:
                raise _SelectiveManifestParseError
            seen.add(key)
            position = whitespace(text, position)
            if position >= len(text) or text[position] != ':':
                raise _SelectiveManifestParseError
            value_start = position + 1
            if key == "block_assignments":
                selected, position = selective_assignments(text, value_start)
            elif key == "partition_implementation_git_commit":
                implementation, position = string(text, value_start)
            elif key == "manifest_sha256":
                manifest_sha, position = string(text, value_start)
            else:
                position = skip_value(text, value_start)
            pair_spans.append((key, key_start, position))
            position = whitespace(text, position)
            if position < len(text) and text[position] == ',':
                position = whitespace(text, position + 1)
            elif position >= len(text) or text[position] != '}':
                raise _SelectiveManifestParseError
        if position >= len(text) or text[position] != '}':
            raise _SelectiveManifestParseError
        position = whitespace(text, position + 1)
        if position != len(text) or manifest_sha is None or implementation is None or selected is None:
            raise _SelectiveManifestParseError
        if seen != set(V8_MANIFEST_FIELDS):
            raise _SelectiveManifestParseError
        reconstructed = "{" + ",".join(
            text[start:end] for key, start, end in pair_spans if key != "manifest_sha256"
        ) + "}\n"
        recomputed = hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise V8DReadinessBlocked("V8D_PARTITION_MANIFEST_SELECTIVE_READ_FAILED") from error
    if recomputed != manifest_sha:
        raise V8DReadinessBlocked("V8D_PARTITION_MANIFEST_SELF_HASH_MISMATCH")
    return manifest_sha, implementation, selected


def execute_transport_readiness_probe(*, stage: str, request_factory: Callable[[int], V8DRequestPlan],
                                      audit_root: str | Path, reviewed_implementation_commit: str,
                                      gate_binding: Any, sleep_fn: Callable[[float], None]) -> dict[str, Any]:
    """Run transport against an already-created binding for synthetic tests.

    This helper intentionally has no gate-consumption capability.  Callers
    must provide a prebuilt binding; only the fixed production core may call
    ``consume_gate_and_bind``.
    """
    if stage not in READINESS_STAGES:
        raise V8DReadinessBlocked("V8D_READINESS_STAGE_INVALID")
    try:
        return execute_v8d_stage(
            stage=stage,
            request_factory=request_factory,
            store=DurableV8DAuditStore(audit_root),
            reviewed_implementation_commit=reviewed_implementation_commit,
            gate_binding=gate_binding,
            window_start=SENTINEL_START,
            window_end_exclusive=SENTINEL_END_EXCLUSIVE,
            request_count=len(SENTINEL_INDICES),
            sleep_fn=sleep_fn,
        )
    except V8DTransportBlocked as error:
        raise V8DReadinessBlocked(error.reason) from error


def _verify_readiness_authority(
    stage: str,
    verified_head: str,
    reviewed_commit: str,
    repository_root: str | Path,
    *,
    anchor_reader: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify the original V8 authority needed by either fixed T0 probe."""
    del reviewed_commit
    anchor = (anchor_reader or (lambda head: read_and_verify_v8_trusted_partition_anchor(
        repository_root, head
    )))(verified_head)
    if not isinstance(anchor, Mapping):
        raise V8DReadinessBlocked("V8D_READINESS_AUTHORITY_PREREQUISITES_BLOCKED")
    try:
        if anchor["authorization_status"] != "AUTHORIZED":
            raise V8DReadinessBlocked("V8D_READINESS_AUTHORITY_NOT_AUTHORIZED")
        if anchor["authorized_partition_manifest_sha256"] != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
            raise V8DReadinessBlocked("V8D_READINESS_PARTITION_PROVENANCE_MISMATCH")
        if anchor["authorized_partition_implementation_git_commit"] != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
            raise V8DReadinessBlocked("V8D_READINESS_PARTITION_PROVENANCE_MISMATCH")
    except KeyError as error:
        raise V8DReadinessBlocked("V8D_READINESS_AUTHORITY_PREREQUISITES_BLOCKED") from error
    return {
        "trusted_partition_anchor_blob_sha": EXPECTED_V8_TRUSTED_PARTITION_BLOB_SHA,
        "authorized_partition_manifest_sha256": EXPECTED_V8_PARTITION_MANIFEST_SHA256,
        "authorized_partition_implementation_git_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "authorization_status": "AUTHORIZED",
        "logical_stage": stage,
    }


def _frozen_request_factory(
    stage: str,
    sentinel_tickers: tuple[str, ...],
    *,
    opener: Callable[[Any], Any] = default_trusted_yahoo_opener,
) -> Callable[[int], V8DRequestPlan]:
    if len(sentinel_tickers) != len(SENTINEL_INDICES):
        raise V8DReadinessBlocked("V8D_READINESS_SENTINEL_RESOLUTION_INVALID")
    ticker_by_coordinate = dict(zip(SENTINEL_INDICES, sentinel_tickers))

    def request_factory(coordinate: int) -> V8DRequestPlan:
        try:
            ticker = ticker_by_coordinate[coordinate]
        except KeyError as error:
            raise V8DReadinessBlocked("V8D_READINESS_COORDINATE_INVALID") from error
        return build_yahoo_request_plan(
            logical_stage=stage,
            logical_block="T1C" if stage.startswith("T1C") else "T2",
            logical_coordinate=coordinate,
            ticker=ticker,
            request_start=SENTINEL_START,
            request_end_exclusive=SENTINEL_END_EXCLUSIVE,
            opener=opener,
            validate_result=require_nonempty_quality,
        )

    return request_factory


def _execute_production_transport_readiness(
    *,
    stage: str,
    human_authorization_identity: str,
    partition_manifest_path: str | Path,
) -> dict[str, Any]:
    """The fixed V8D production readiness flow.

    This function is intentionally the only gate-consuming readiness core,
    and it has no request, provenance, authority, opener, clock, or state
    injection seams. Synthetic tests use ``execute_transport_readiness_probe``
    with a prebuilt synthetic gate binding instead.
    """
    if stage not in READINESS_STAGES:
        raise V8DReadinessBlocked("V8D_READINESS_STAGE_INVALID")
    if not isinstance(human_authorization_identity, str) or not human_authorization_identity:
        raise V8DReadinessBlocked("V8D_READINESS_HUMAN_AUTHORIZATION_IDENTITY_REQUIRED")

    try:
        verified_head = resolve_verified_v8d_production_git_commit(CANONICAL_REPOSITORY_ROOT)
    except V8DGitProvenanceBlocked as error:
        raise _wrap(error) from error

    try:
        verify_frozen_design_object(CANONICAL_REPOSITORY_ROOT)
        verify_design_freeze_approval_blob(CANONICAL_REPOSITORY_ROOT, verified_head)
    except V8DProductionProvenanceBlocked as error:
        raise _wrap(error) from error

    try:
        review_binding = verify_reviewed_implementation_binding(CANONICAL_REPOSITORY_ROOT, verified_head)
    except V8DProductionProvenanceBlocked as error:
        raise _wrap(error) from error
    reviewed_commit = review_binding["reviewed_implementation_git_commit"]

    try:
        authority_prerequisites = _verify_readiness_authority(
            stage, verified_head, reviewed_commit, CANONICAL_REPOSITORY_ROOT
        )
        verify_stage_authority_bridge(CANONICAL_REPOSITORY_ROOT, verified_head, stage)
        manifest_sha, manifest_implementation, sentinels = _read_selective_t0_sentinels(
            partition_manifest_path
        )
    except V8DReadinessBlocked:
        raise
    except (V8DAuthorityBridgeBlocked, V8DProductionProvenanceBlocked, V8DGitProvenanceBlocked) as error:
        raise _wrap(error) from error
    except Exception as error:  # noqa: BLE001 - private input failures BLOCK
        raise V8DReadinessBlocked(
            getattr(error, "reason", "V8D_READINESS_AUTHORITY_PREREQUISITES_BLOCKED")
        ) from error
    if (
        manifest_sha != authority_prerequisites["authorized_partition_manifest_sha256"]
        or manifest_sha != EXPECTED_V8_PARTITION_MANIFEST_SHA256
        or manifest_implementation != authority_prerequisites["authorized_partition_implementation_git_commit"]
        or manifest_implementation != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT
        or len(sentinels) != len(SENTINEL_INDICES)
    ):
        raise V8DReadinessBlocked("V8D_READINESS_PARTITION_PROVENANCE_MISMATCH")
    request_factory = _frozen_request_factory(stage, sentinels)

    try:
        binding = consume_gate_and_bind(
            CANONICAL_CONSUMPTION_STATE_ROOT,
            logical_stage=stage,
            v8d_frozen_design_commit=EXPECTED_V8D_FROZEN_DESIGN_COMMIT,
            reviewed_production_implementation_commit=reviewed_commit,
            raw_authorization_identity=human_authorization_identity,
            clock=lambda: datetime.now(timezone.utc),
        )
    except V8DHumanGateConsumptionBlocked as error:
        raise V8DReadinessBlocked(error.reason) from error

    gate_binding = {
        "human_gate": binding.human_gate,
        "gate_receipt_key_sha256": binding.gate_receipt_key_sha256,
        "gate_receipt_bytes_sha256": binding.gate_receipt_bytes_sha256,
        "authorization_identity_sha256": binding.authorization_identity_sha256,
    }

    return execute_transport_readiness_probe(
        stage=stage,
        request_factory=request_factory,
        audit_root=CANONICAL_PRODUCTION_AUDIT_ROOT,
        reviewed_implementation_commit=reviewed_commit,
        gate_binding=gate_binding,
        sleep_fn=time.sleep,
    )


def execute_t1c_transport_readiness_production(
    *, human_authorization_identity: str, partition_manifest_path: str | Path,
) -> dict[str, Any]:
    """Sole production entrypoint for `T1C_TRANSPORT_READINESS_HUMAN_GATE`."""
    return _execute_production_transport_readiness(
        stage="T1C_TRANSPORT_READINESS", human_authorization_identity=human_authorization_identity,
        partition_manifest_path=partition_manifest_path,
    )


def execute_t2_transport_readiness_production(
    *, human_authorization_identity: str, partition_manifest_path: str | Path,
) -> dict[str, Any]:
    """Sole production entrypoint for `T2_TRANSPORT_READINESS_HUMAN_GATE`."""
    return _execute_production_transport_readiness(
        stage="T2_TRANSPORT_READINESS", human_authorization_identity=human_authorization_identity,
        partition_manifest_path=partition_manifest_path,
    )


def synthetic_request_plan(*, stage: str, coordinate: int, url: str,
                           request_fn: Callable[[], Any],
                           request_parameters: dict[str, Any] | None = None) -> V8DRequestPlan:
    """Build a plan for tests or a caller that already owns request creation.

    Only the URL digest is retained; the URL itself is never written to an
    audit or aggregate artifact.
    """
    block = "T1C" if stage.startswith("T1C") else "T2"
    fingerprint = make_request_fingerprint(
        logical_stage=stage,
        logical_block=block,
        logical_coordinate=coordinate,
        window_start=SENTINEL_START,
        window_end_exclusive=SENTINEL_END_EXCLUSIVE,
        request_parameters=request_parameters,
    )
    return V8DRequestPlan(request_fn=request_fn, request_fingerprint=fingerprint, request_url_sha256=sha256_url(url))


__all__ = [
    "CANONICAL_PRODUCTION_AUDIT_ROOT",
    "T0_EXPECTED_COUNT",
    "V8DReadinessBlocked",
    "_frozen_request_factory",
    "_read_selective_t0_sentinels",
    "execute_t1c_transport_readiness_production",
    "execute_t2_transport_readiness_production",
    "execute_transport_readiness_probe",
    "synthetic_request_plan",
]
