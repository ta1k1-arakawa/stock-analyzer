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
import hashlib
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.v7_yahoo_collector import HOST, V7YahooCollectorBlocked, fetch_chart_once
from src.v8_partition import MANIFEST_FIELDS as V8_MANIFEST_FIELDS, V8PartitionBlocked
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
from src.v8c_stage_state import V8CStageEvidenceBlocked, write_readiness_pass
from src.v8c_git_provenance import read_git_object_bytes
from src.v8c_trust_pin import read_trust_pin_bytes
from src.v8c_production_provenance import read_and_verify_trust_pin_independent_review
from src.v8c_t2_bridge import read_and_verify_v8c_t2_authority_bridge, read_and_verify_v8c_t2_authority_bridge_independent_review
from src.v8c_stage_state import read_valid_t2_recheck_pass

SENTINEL_INDICES: tuple[int, ...] = (0, 149, 299)
SENTINEL_PROBE_START = "2025-12-01"
SENTINEL_PROBE_END_EXCLUSIVE = "2025-12-08"
T0_EXPECTED_COUNT = 300

STAGE_GATE = {
    "T1C": GATE_T1C_TRANSPORT_READINESS,
    "T2": GATE_T2_TRANSPORT_READINESS,
}


def _production_stage_prerequisites(stage: str, verified_head: str, reviewed_commit: str) -> Mapping[str, Any]:
    """Verify the non-readiness stages immediately upstream of readiness.

    ``reviewed_commit`` is the exact *current* reviewed-implementation
    commit (resolved immediately before this call, from a fresh
    ``verify_reviewed_implementation_binding`` at ``verified_head``) -- both
    ``V8C_TRUSTED_ALLOCATION.json``'s own
    ``v8c_reviewed_production_implementation_commit`` and
    ``V8C_TRUST_PIN_INDEPENDENT_REVIEW.json``'s own
    ``reviewed_production_implementation_commit`` must equal it exactly, not
    merely equal each other -- a stale pair that only agrees with itself
    (but no longer with the live implementation binding) still BLOCKs.
    """
    if stage == "T1C":
        try:
            pin = read_trust_pin_bytes(read_git_object_bytes(CANONICAL_REPOSITORY_ROOT, verified_head, "V8C_TRUSTED_ALLOCATION.json"))
            review = read_and_verify_trust_pin_independent_review(
                CANONICAL_REPOSITORY_ROOT,
                verified_head,
                expected_allocation_artifact_self_hash=pin["authorized_allocation_artifact_self_hash"],
                expected_trust_pin_human_gate=pin["human_gate"],
            )
            if pin["v8c_reviewed_production_implementation_commit"] != reviewed_commit:
                raise V8CReadinessBlocked("V8C_T1C_TRUST_PIN_PRODUCTION_IMPLEMENTATION_STALE")
            if review["reviewed_production_implementation_commit"] != reviewed_commit:
                raise V8CReadinessBlocked("V8C_T1C_TRUST_PIN_REVIEW_PRODUCTION_IMPLEMENTATION_STALE")
        except Exception as error:  # noqa: BLE001
            raise V8CReadinessBlocked("V8C_T1C_AUTHORITY_PREREQUISITES_BLOCKED") from error
        return {
            "authorized_allocation_artifact_self_hash": pin["authorized_allocation_artifact_self_hash"],
            "parent_v8_partition_manifest_sha256": pin["parent_v8_partition_manifest_sha256"],
            "parent_v8_partition_implementation_commit": pin["parent_v8_partition_implementation_commit"],
            "trust_pin_human_gate": pin["human_gate"],
        }
    try:
        implementation = verify_reviewed_implementation_binding(CANONICAL_REPOSITORY_ROOT, verified_head)
        recheck = read_valid_t2_recheck_pass(
            CANONICAL_CONSUMPTION_STATE_ROOT,
            frozen_design_commit=EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
            reviewed_implementation_commit=implementation["reviewed_implementation_git_commit"],
        )
        bridge = read_and_verify_v8c_t2_authority_bridge(CANONICAL_REPOSITORY_ROOT, verified_head)
        bridge_blob = resolve_git_blob(CANONICAL_REPOSITORY_ROOT, verified_head, "V8C_T2_AUTHORITY_BRIDGE.json")
        review = read_and_verify_v8c_t2_authority_bridge_independent_review(
            CANONICAL_REPOSITORY_ROOT, verified_head, expected_bridge_git_blob_sha=bridge_blob
        )
    except Exception as error:  # noqa: BLE001
        raise V8CReadinessBlocked("V8C_T2_AUTHORITY_PREREQUISITES_BLOCKED") from error
    return {
        "v8_partition_manifest_sha256": bridge["authorized_parent_v8_partition_manifest_sha256"],
        "v8_partition_implementation_commit": EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT,
        "v8_trust_anchor_git_identity": bridge["v8_trust_anchor_git_identity"],
        "v8c_t2_authority_bridge_human_gate": bridge["exact_human_bridge_authorization_identity"],
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


def _require_trusted_redirect_target(value: object, *, hostname: str) -> str:
    """A rejected *redirect target* is ``UNTRUSTED_REDIRECT`` -- distinct
    from ``RESPONSE_HOST_MISMATCH``, which is reserved for the initial
    constructed request or a final/returned response URL whose origin is
    wrong (``_require_exact_origin`` above)."""
    if not isinstance(value, str):
        raise V8CTransportNamedFailure("UNTRUSTED_REDIRECT")
    try:
        parsed = urllib.parse.urlparse(value)
        port = parsed.port
    except ValueError as error:
        raise V8CTransportNamedFailure("UNTRUSTED_REDIRECT") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname != hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise V8CTransportNamedFailure("UNTRUSTED_REDIRECT")
    return value


class _TrustedYahooRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        try:
            _require_trusted_redirect_target(newurl, hostname=HOST)
        except V8CTransportNamedFailure as error:
            raise error
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


def _read_selective_t0_sentinels(partition_manifest_path: str | Path) -> tuple[str, str, tuple[str, ...]]:
    """Resolve only the fixed T0 sentinel coordinates.

    Readiness deliberately does not call the full partition-manifest reader,
    whose return value contains every private block assignment.  The parser
    is strict and the returned value contains only the three permitted
    sentinel identities.
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
            raise _SelectiveManifestParseError("string expected")
        value, end = decoder.raw_decode(raw, position)
        if not isinstance(value, str):
            raise _SelectiveManifestParseError("string expected")
        return value, end

    def skip_value(raw: str, position: int) -> int:
        """Skip one JSON value without deserializing its contents."""
        position = whitespace(raw, position)
        if position >= len(raw):
            raise _SelectiveManifestParseError("value expected")
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
            raise _SelectiveManifestParseError("unterminated string")
        if marker == '[':
            position = whitespace(raw, position + 1)
            if position < len(raw) and raw[position] == ']':
                return position + 1
            while True:
                position = skip_value(raw, position)
                position = whitespace(raw, position)
                if position >= len(raw):
                    raise _SelectiveManifestParseError("unterminated array")
                if raw[position] == ']':
                    return position + 1
                if raw[position] != ',':
                    raise _SelectiveManifestParseError("array separator expected")
                position += 1
        if marker == '{':
            position = whitespace(raw, position + 1)
            keys: set[str] = set()
            if position < len(raw) and raw[position] == '}':
                return position + 1
            while True:
                key, position = string(raw, position)
                if key in keys:
                    raise _SelectiveManifestParseError("duplicate object key")
                keys.add(key)
                position = whitespace(raw, position)
                if position >= len(raw) or raw[position] != ':':
                    raise _SelectiveManifestParseError("object colon expected")
                position = skip_value(raw, position + 1)
                position = whitespace(raw, position)
                if position >= len(raw):
                    raise _SelectiveManifestParseError("unterminated object")
                if raw[position] == '}':
                    return position + 1
                if raw[position] != ',':
                    raise _SelectiveManifestParseError("object separator expected")
                position += 1
        value, end = decoder.raw_decode(raw, position)
        if isinstance(value, (dict, list, str)):
            raise _SelectiveManifestParseError("unexpected value")
        return end

    def selective_t0_array(raw: str, position: int) -> tuple[tuple[str, ...], int]:
        position = whitespace(raw, position)
        if position >= len(raw) or raw[position] != '[':
            raise _SelectiveManifestParseError("T0 array expected")
        position = whitespace(raw, position + 1)
        selected: list[str] = []
        for index in range(T0_EXPECTED_COUNT):
            if index in SENTINEL_INDICES:
                value, position = string(raw, position)
                if not value:
                    raise _SelectiveManifestParseError("empty sentinel")
                selected.append(value)
            else:
                position = skip_value(raw, position)
            position = whitespace(raw, position)
            if index < T0_EXPECTED_COUNT - 1:
                if position >= len(raw) or raw[position] != ',':
                    raise _SelectiveManifestParseError("T0 separator expected")
                position = whitespace(raw, position + 1)
            elif position >= len(raw) or raw[position] != ']':
                raise _SelectiveManifestParseError("T0 terminator expected")
            else:
                position = whitespace(raw, position + 1)
        return tuple(selected), position

    def selective_assignments(raw: str, position: int) -> tuple[tuple[str, ...], int]:
        position = whitespace(raw, position)
        if position >= len(raw) or raw[position] != '{':
            raise _SelectiveManifestParseError("assignment object expected")
        position = whitespace(raw, position + 1)
        names: set[str] = set()
        selected: tuple[str, ...] | None = None
        if position < len(raw) and raw[position] == '}':
            raise _SelectiveManifestParseError("empty assignment object")
        while True:
            name, position = string(raw, position)
            if name in names:
                raise _SelectiveManifestParseError("duplicate assignment key")
            names.add(name)
            position = whitespace(raw, position)
            if position >= len(raw) or raw[position] != ':':
                raise _SelectiveManifestParseError("assignment colon expected")
            if name == "T0":
                selected, position = selective_t0_array(raw, position + 1)
            else:
                position = skip_value(raw, position + 1)
            position = whitespace(raw, position)
            if position >= len(raw):
                raise _SelectiveManifestParseError("unterminated assignment object")
            if raw[position] == '}':
                position += 1
                break
            if raw[position] != ',':
                raise _SelectiveManifestParseError("assignment separator expected")
            position += 1
        if names != {"T0", "T1", "T2", "T3", "T_spare"} or selected is None:
            raise _SelectiveManifestParseError("assignment schema invalid")
        # Deliberately no trailing whitespace-skip here: ``position`` is
        # returned exactly at the first character after the value's closing
        # ``}``, so the caller can capture a byte-exact top-level pair span
        # for canonical-hash reconstruction. The top-level loop performs its
        # own whitespace handling afterward.
        return selected, position

    try:
        raw = Path(partition_manifest_path).read_bytes()
        text = raw.decode("utf-8")
        position = whitespace(text, 0)
        if position >= len(text) or text[position] != '{':
            raise _SelectiveManifestParseError("top-level object expected")
        position = whitespace(text, position + 1)
        seen: set[str] = set()
        manifest_sha: str | None = None
        implementation: str | None = None
        selected: tuple[str, ...] | None = None
        # Byte-exact spans of every top-level "key":value pair, in file
        # order, so the manifest's own canonical self-hash can be
        # independently recomputed (never trusted from the embedded
        # ``manifest_sha256`` string) without ever materializing any
        # forbidden private array element as a Python string. Since the
        # only production writer of this manifest (``src.v8_partition.
        # write_partition_manifest_once``) always persists it via
        # ``canonical_json_bytes`` (sorted keys, compact separators,
        # trailing "\n"), splicing the ``manifest_sha256`` pair (plus its
        # one adjoining comma) out of these exact original byte spans and
        # re-joining the rest is byte-identical to re-serializing the
        # manifest without that one field via the same canonical scheme
        # ``src.v8_partition.canonical_sha256`` uses.
        pair_spans: list[tuple[str, int, int]] = []
        while position < len(text) and text[position] != '}':
            key_start = position
            key, position = string(text, position)
            if key in seen:
                raise _SelectiveManifestParseError("duplicate top-level key")
            seen.add(key)
            position = whitespace(text, position)
            if position >= len(text) or text[position] != ':':
                raise _SelectiveManifestParseError("top-level colon expected")
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
                raise _SelectiveManifestParseError("top-level separator expected")
        if position >= len(text) or text[position] != '}' or selected is None:
            raise _SelectiveManifestParseError("top-level object incomplete")
        position = whitespace(text, position + 1)
        if position != len(text) or manifest_sha is None or implementation is None:
            raise _SelectiveManifestParseError("required manifest field missing")
        if seen != set(V8_MANIFEST_FIELDS):
            raise _SelectiveManifestParseError("required manifest schema invalid")

        reconstructed = "{" + ",".join(
            text[start:end] for key, start, end in pair_spans if key != "manifest_sha256"
        ) + "}\n"
        recomputed_manifest_sha256 = hashlib.sha256(reconstructed.encode("utf-8")).hexdigest()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, _SelectiveManifestParseError) as error:
        raise V8CReadinessBlocked("PARTITION_MANIFEST_SELECTIVE_READ_FAILED") from error
    if recomputed_manifest_sha256 != manifest_sha:
        raise V8CReadinessBlocked("TRUSTED_PARTITION_MANIFEST_SELF_HASH_MISMATCH")
    return manifest_sha, implementation, selected


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
    stage_prerequisite_checker: Callable[[str, str, str], Mapping[str, Any]] | None = None,
    readiness_pass_state_writer: Callable[..., Mapping[str, Any]] | None = None,
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

    # (3) reviewed implementation binding
    try:
        review_binding = reviewed_implementation_binder(verified_head)
        reviewed_commit = review_binding["reviewed_implementation_git_commit"]
    except (V8CProductionProvenanceBlocked, V8CGitProvenanceBlocked) as error:
        raise _wrap(error, "V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error
    except KeyError as error:
        raise V8CReadinessBlocked("V8C_PRODUCTION_IMPLEMENTATION_REVIEW_MISSING") from error

    try:
        authority_prerequisites = (
            dict(stage_prerequisite_checker(stage, verified_head, reviewed_commit))
            if stage_prerequisite_checker
            else {"dependency_injection": "synthetic"}
        )
    except Exception as error:  # noqa: BLE001 - every unmet prerequisite BLOCKs
        raise V8CReadinessBlocked(getattr(error, "reason", "V8C_READINESS_STAGE_PREREQUISITES_BLOCKED")) from error

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
    manifest_sha, manifest_implementation, sentinel_tickers = _read_selective_t0_sentinels(partition_manifest_path)
    if manifest_sha != anchor["authorized_partition_manifest_sha256"]:
        raise V8CReadinessBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    if manifest_implementation != anchor["authorized_partition_implementation_git_commit"]:
        raise V8CReadinessBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")
    if manifest_sha != EXPECTED_V8_PARTITION_MANIFEST_SHA256:
        raise V8CReadinessBlocked("TRUSTED_PARTITION_MANIFEST_SHA_MISMATCH")
    if manifest_implementation != EXPECTED_V8_PARTITION_IMPLEMENTATION_COMMIT:
        raise V8CReadinessBlocked("TRUSTED_PARTITION_IMPLEMENTATION_GIT_COMMIT_MISMATCH")

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

    result = {
        "stage": stage,
        "result": "PASS" if all_pass else "BLOCK",
        "all_three_sentinels_required": True,
        "sentinel_count": len(SENTINEL_INDICES),
        "sentinel_pass_count": sum(1 for entry in sentinel_results if entry["pass"]),
        "probe_start": SENTINEL_PROBE_START,
        "probe_end_exclusive": SENTINEL_PROBE_END_EXCLUSIVE,
        "readiness_failure_consumes_acquisition_gate": False,
        "readiness_is_research_opening": False,
        "frozen_design_commit": EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
        "reviewed_implementation_commit": reviewed_commit,
        "classifier_blob_sha": classifier_blob_sha,
        "authority_prerequisites": authority_prerequisites,
    }
    # Record the authoritative durable result of *every* completed
    # readiness execution -- PASS or BLOCK -- not only PASS. A later
    # authorized BLOCK must overwrite (and thereby invalidate) an earlier
    # PASS at the same destination, so raw acquisition can never keep
    # consuming a stale PASS after a fresh authorized execution BLOCKed.
    try:
        (readiness_pass_state_writer or write_readiness_pass)(
            consumption_state_root,
            stage=stage,
            result=result["result"],
            frozen_design_commit=EXPECTED_V8C_FROZEN_DESIGN_COMMIT,
            reviewed_implementation_commit=reviewed_commit,
            sentinel_indices=list(SENTINEL_INDICES),
            probe_start=SENTINEL_PROBE_START,
            probe_end_exclusive=SENTINEL_PROBE_END_EXCLUSIVE,
            classifier_blob_sha=classifier_blob_sha,
            authority_prerequisites=authority_prerequisites,
            sentinel_count=result["sentinel_count"],
            sentinel_pass_count=result["sentinel_pass_count"],
            human_authorization_identity=human_authorization_identity,
            consumption_state_root=consumption_state_root,
            clock_text=clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
    except V8CStageEvidenceBlocked as error:
        raise V8CReadinessBlocked(error.reason) from error
    return result


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
        stage_prerequisite_checker=_production_stage_prerequisites,
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
        stage_prerequisite_checker=_production_stage_prerequisites,
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
