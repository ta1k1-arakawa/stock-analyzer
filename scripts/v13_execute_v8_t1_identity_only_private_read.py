"""One-shot V8 T1 private-read harness; never run during PRE_GATE review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from scripts.v13_resolve_v8_t1_identity_state import (
    IdentityResolutionBlocked,
    _PRODUCTION_BINDINGS,
    _validate_output_destination,
    _ensure_output_does_not_exist,
    _write_once,
    resolve_identity_state,
)

AUTHORIZATION_REVIEWED_SHA = "18a99fbb3740ecb827abc14513fad1402f7632ec"
RESOLVER_REVIEWED_SHA = "63fa6b7541694565eb171a98d11df0092ab20034"
DESIGN_BLOB_SHA1 = "3bfcd695c69f6dac480f8fc99ca4f3916f668e4a"
AUTHORIZATION_BLOB_SHA1 = "280baea899a576cbc3b705db838e5988dbed5027"
RECEIPT_SCHEMA = "V13_V8_T1_PRIVATE_READ_CONSUMED_RECEIPT_V1"
STUDY = "V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON"
BOUNDARY = "FIRST_CONTENT_BYTE_READ_FROM_VERIFIED_PRIVATE_V8_PARTITION_MANIFEST"

_SAFE_RECEIPT = {
    "schema": RECEIPT_SCHEMA,
    "study": STUDY,
    "operation_class": "STATISTICALLY_IRREVERSIBLE_GATE",
    "boundary": BOUNDARY,
    "authorization_reviewed_sha": AUTHORIZATION_REVIEWED_SHA,
    "resolver_reviewed_sha": RESOLVER_REVIEWED_SHA,
    "expected_partition_manifest_stated_sha256": _PRODUCTION_BINDINGS.manifest_stated_sha256,
    "expected_t1_ticker_list_sha256": _PRODUCTION_BINDINGS.t1_ticker_list_sha256,
    "expected_t1_count": _PRODUCTION_BINDINGS.t1_count,
    "authorization_consumed": True,
}


class HarnessBlocked(RuntimeError):
    def __init__(self, reason: str, *, crossed: bool = False, receipt_written: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.crossed = crossed
        self.receipt_written = receipt_written


def _fail(reason: str) -> None:
    raise HarnessBlocked(reason)


def _validate_authorization(path: Path, repository: Path) -> None:
    expected_path = (repository / "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION.json").resolve()
    try:
        if path.resolve() != expected_path:
            _fail("AUTHORIZATION_PATH_INVALID")
        authorization_bytes = path.read_bytes()
        document = json.loads(authorization_bytes.decode("utf-8"))
    except HarnessBlocked:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        _fail("AUTHORIZATION_INVALID")
    required = {
        "schema": "V13_V8_T1_IDENTITY_ONLY_PRIVATE_READ_AUTHORIZATION_V1",
        "study": STUDY,
        "authoritative_branch": "v13-conditional-cross-sectional-short-horizon",
        "human_approved": True,
        "operation_class": "STATISTICALLY_IRREVERSIBLE_GATE",
        "authorization_scope": "V8_T1_PARTITION_MEMBERSHIP_IDENTITY_ONLY_FOR_V13_EXCLUSION_PROVENANCE",
        "one_shot_authority": True,
        "authorization_consumed": False,
        "expected_partition_manifest_sha256": _PRODUCTION_BINDINGS.manifest_stated_sha256,
        "expected_t1_ticker_list_sha256": _PRODUCTION_BINDINGS.t1_ticker_list_sha256,
        "expected_t1_count": _PRODUCTION_BINDINGS.t1_count,
        "private_partition_manifest_read_authorized": True,
        "t1_membership_identity_read_authorized": True,
        "t1_price_payload_read_authorized": False,
        "t1_outcome_or_metric_read_authorized": False,
        "t2_membership_identity_read_authorized": False,
        "t3_membership_identity_read_authorized": False,
        "t_spare_membership_identity_read_authorized": False,
        "historical_price_payload_read_authorized": False,
        "historical_viability_authorized": False,
        "model_fit_authorized": False,
        "backtest_authorized": False,
        "forward_paper_authorized": False,
        "real_trading_authorized": False,
        "future_profitability_established": False,
    }
    auth_blob = hashlib.sha1(
        b"blob " + str(len(authorization_bytes)).encode("ascii") + b"\0" + authorization_bytes
    ).hexdigest()
    if auth_blob != AUTHORIZATION_BLOB_SHA1:
        _fail("AUTHORIZATION_BLOB_MISMATCH")
    if not isinstance(document, dict) or any(document.get(key) != value for key, value in required.items()):
        _fail("AUTHORIZATION_BINDING_MISMATCH")
    try:
        design = (repository / "V13_CONDITIONAL_CROSS_SECTIONAL_SHORT_HORIZON_DESIGN_DRAFT.md").read_bytes()
    except OSError:
        _fail("FROZEN_DESIGN_UNAVAILABLE")
    blob = hashlib.sha1(b"blob " + str(len(design)).encode("ascii") + b"\0" + design).hexdigest()
    if blob != DESIGN_BLOB_SHA1:
        _fail("FROZEN_DESIGN_BINDING_MISMATCH")


def _probe_parent(destination: Path) -> None:
    if not destination.parent.is_dir():
        _fail("OUTPUT_PARENT_UNAVAILABLE")
    descriptor: int | None = None
    probe: str | None = None
    try:
        descriptor, probe = tempfile.mkstemp(prefix=".v13-preflight-", dir=str(destination.parent))
        os.close(descriptor)
        descriptor = None
    except OSError:
        _fail("OUTPUT_PARENT_NOT_WRITABLE")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if probe is not None:
            try:
                os.unlink(probe)
            except OSError:
                _fail("OUTPUT_PREFLIGHT_CLEANUP_FAILED")


def _preflight(
    source_path: str | os.PathLike[str],
    state_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    authorization_path: str | os.PathLike[str],
) -> tuple[Path, Path, Path, Path]:
    try:
        repository = Path(repository_root).resolve(strict=True)
        if not repository.is_dir():
            _fail("REPOSITORY_PATH_INVALID")
        source = Path(source_path)
        if not source.is_absolute():
            _fail("SOURCE_PATH_INVALID")
        source = source.resolve(strict=True)
        if not source.is_file():
            _fail("SOURCE_NOT_REGULAR_FILE")
        state = _validate_output_destination(state_path, repository)
        receipt = _validate_output_destination(receipt_path, repository)
        if state == receipt:
            _fail("OUTPUT_PATH_COLLISION")
        _ensure_output_does_not_exist(receipt)
        _ensure_output_does_not_exist(state)
        try:
            _validate_output_destination(source, repository)
        except IdentityResolutionBlocked as exc:
            if exc.reason == "OUTPUT_PATH_INSIDE_REPOSITORY":
                _fail("SOURCE_PATH_INSIDE_REPOSITORY")
            raise
        _probe_parent(receipt)
        _probe_parent(state)
        _validate_authorization(Path(authorization_path), repository)
    except HarnessBlocked:
        raise
    except IdentityResolutionBlocked as exc:
        _fail(exc.reason)
    except (OSError, TypeError, ValueError):
        _fail("PREGATE_PATH_VALIDATION_FAILED")
    return source, state, receipt, repository


def _receipt_bytes() -> bytes:
    return (json.dumps(_SAFE_RECEIPT, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _report(
    *,
    pre_gate: str,
    crossed: str,
    consumed: str,
    reads: str,
    count: str,
    t1_match: str,
    manifest_match: str,
    state_written: bool,
    receipt_written: bool,
    result: str,
    failure: str,
    reusable: str,
    second_allowed: str,
) -> str:
    fields = {
        "PRE_GATE_STATUS": pre_gate,
        "PRIVATE_BOUNDARY_CROSSED": crossed,
        "GATE_CONSUMED": consumed,
        "PRIVATE_READS": reads,
        "NETWORK_REQUESTS": "0",
        "PRICE_PAYLOAD_READS": "0",
        "OUTCOME_READS": "0",
        "T1_COUNT": count,
        "T1_HASH_MATCH": t1_match,
        "MANIFEST_STATED_SHA_MATCH": manifest_match,
        "PRIVATE_STATE_WRITTEN": str(state_written).lower(),
        "CONSUMED_RECEIPT_WRITTEN": str(receipt_written).lower(),
        "T2_T3_TSPARE_IDENTITIES_RETAINED": "false",
        "EXECUTION_RESULT": result,
        "FAILURE_CLASS": failure,
        "AUTHORIZATION_REUSABLE": reusable,
        "SECOND_EXECUTION_ALLOWED": second_allowed,
    }
    return "\n".join(f"{key}={value}" for key, value in fields.items())


def execute(
    source_path: str | os.PathLike[str],
    state_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
    authorization_path: str | os.PathLike[str],
    *,
    resolver: Callable[..., dict[str, Any]] = resolve_identity_state,
) -> str:
    try:
        source, state, receipt, repository = _preflight(
            source_path, state_path, receipt_path, repository_root, authorization_path
        )
    except HarnessBlocked as exc:
        already_used = exc.reason == "OUTPUT_ALREADY_EXISTS"
        return _report(
            pre_gate="FAIL", crossed="false", consumed="false", reads="0", count="unknown",
            t1_match="unknown", manifest_match="unknown", state_written=False, receipt_written=False,
            result="PRE_GATE_STOP", failure=exc.reason,
            reusable="false" if already_used else "true", second_allowed="false",
        )

    crossed = False
    receipt_written = False
    state_written = False

    def consume() -> None:
        nonlocal crossed, receipt_written
        crossed = True
        _write_once(receipt, _receipt_bytes())
        receipt_written = True

    def mark_state_written() -> None:
        nonlocal state_written
        state_written = True

    try:
        resolver(
            source, state, repository, on_first_byte=consume, on_state_written=mark_state_written
        )
    except (IdentityResolutionBlocked, HarnessBlocked) as exc:
        boundary = crossed or getattr(exc, "reason", "").startswith("POST_BOUNDARY")
        reason = getattr(exc, "reason", "POST_BOUNDARY_FAILURE" if boundary else "SOURCE_READ_FAILED")
        result = "POST_BOUNDARY_FAILURE" if boundary else "PRE_BOUNDARY_FAILURE"
        pre_gate = "PASS" if boundary else "FAIL"
        t1_match = "false" if reason in {"T1_STATED_SHA_MISMATCH", "T1_HASH_MISMATCH"} else "unknown"
        manifest_match = "false" if reason == "MANIFEST_STATED_SHA_MISMATCH" else "unknown"
        return _report(
            pre_gate=pre_gate, crossed=str(boundary).lower(), consumed=str(boundary).lower(),
            reads="1" if boundary else "0", count="unknown", t1_match=t1_match,
            manifest_match=manifest_match, state_written=state_written, receipt_written=receipt_written,
            result=result, failure=reason, reusable="false" if boundary else "true",
            second_allowed="false",
        )
    except Exception:
        boundary = crossed
        return _report(
            pre_gate="PASS" if boundary else "UNKNOWN", crossed=str(boundary).lower(),
            consumed=str(boundary).lower(), reads="1" if boundary else "unknown", count="unknown",
            t1_match="unknown", manifest_match="unknown", state_written=state_written,
            receipt_written=receipt_written,
            result="POST_BOUNDARY_FAILURE" if boundary else "UNKNOWN_FAILURE",
            failure="UNEXPECTED_POST_BOUNDARY_FAILURE",
            reusable="false" if boundary else "unknown", second_allowed="false",
        )
    if not (crossed and receipt_written and state_written):
        return _report(
            pre_gate="UNKNOWN", crossed="unknown", consumed="unknown", reads="unknown",
            count="unknown", t1_match="unknown", manifest_match="unknown",
            state_written=state_written, receipt_written=receipt_written,
            result="UNKNOWN_FAILURE", failure="EXECUTION_STATE_INCOMPLETE",
            reusable="unknown", second_allowed="false",
        )
    return _report(
        pre_gate="PASS", crossed="true", consumed="true", reads="1", count="300",
        t1_match="true", manifest_match="true", state_written=True, receipt_written=True,
        result="PASS", failure="NONE", reusable="false", second_allowed="false",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--state-output", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args(argv)
    report = execute(args.source, args.state_output, args.receipt_output, args.repository_root, args.authorization)
    print(report)
    return 0 if "EXECUTION_RESULT=PASS\n" in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
