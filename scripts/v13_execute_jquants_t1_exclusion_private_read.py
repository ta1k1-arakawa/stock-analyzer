"""Future one-shot protected read. This module has no import-time private I/O."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path
from typing import BinaryIO, Callable

from scripts.v13_resolve_jquants_t1_exclusion_state import (
    Block, CONTRACT, ELIGIBLE_SHA, SCHEMA, SOURCE_BLOB, SOURCE_COMMIT, T1_SHA, resolve,
)
from scripts.v13_resolve_v8_t1_identity_state import _write_once

ROOT_NAME = "v13-t1-exclusion-provenance"
SOURCE_ROOT_NAME = "v8-jquants-identity-recovery"
RECEIPT_SCHEMA = "V13_JQUANTS_T1_EXCLUSION_CONSUMED_RECEIPT_V1"
BOUNDARY = "FIRST_CONTENT_BYTE_READ_FROM_VERIFIED_JQUANTS_RECOVERY_ARTIFACT"
APPROVAL_BLOB = "276feb56f417f9d5b931d594598a1d8591330bfd"


def _safe_path(path: Path) -> None:
    for node in (path, *path.parents):
        if node.is_symlink() or (node.exists() and os.name == "nt" and
            node.stat(follow_symlinks=False).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise Block("PRE_GATE_PATH_TOPOLOGY_BLOCK")


def _preflight(repo: Path, local_app_data: Path) -> tuple[Path, Path, Path]:
    repo = repo.resolve(strict=True)
    if not local_app_data.is_absolute() or not local_app_data.is_dir():
        raise Block("PRE_GATE_ROOT_BLOCK")
    base = local_app_data / "stock-analyzer" / "private"
    source_root = base / SOURCE_ROOT_NAME
    output_root = base / ROOT_NAME
    source = source_root / "recovery.json"
    receipt = output_root / "consumed-receipt.json"
    state = output_root / "t1-exclusion-state.json"
    for path in (source, receipt, state):
        _safe_path(path)
        if path == repo or repo in path.parents:
            raise Block("PRE_GATE_PATH_TOPOLOGY_BLOCK")
    if source_root == output_root or not source.is_file():
        raise Block("PRE_GATE_SOURCE_BLOCK")
    if receipt.exists() or state.exists():
        raise Block("PRE_GATE_EXISTING_OUTPUT_BLOCK")
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise Block("PRE_GATE_OUTPUT_ROOT_BLOCK")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        _safe_path(output_root)
    return source, receipt, state


def _receipt() -> bytes:
    value = {"schema": RECEIPT_SCHEMA, "boundary": BOUNDARY,
        "operation_class": "STATISTICALLY_IRREVERSIBLE_GATE",
        "source_schema": SCHEMA, "source_contract": CONTRACT,
        "source_commit": SOURCE_COMMIT, "source_blob": SOURCE_BLOB,
        "eligible_count": 3110, "eligible_sha256": ELIGIBLE_SHA,
        "t1_count": 300, "t1_ticker_list_sha256": T1_SHA,
        "issue73_approval_blob": APPROVAL_BLOB, "authorization_consumed": True}
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _report(*, stage: str, reason: str, crossed: bool, receipt: bool,
            state: bool, opens: int, binding: str = "unknown", t1_hash: str = "unknown") -> str:
    fields = {
        "PRE_GATE_STATUS": "PASS" if stage != "PRE_GATE_STOP" else "FAIL",
        "PRIVATE_BOUNDARY_CROSSED": str(crossed).lower(),
        "AUTHORIZATION_CONSUMED": str(crossed).lower(),
        "AUTHORIZATION_REUSABLE": str(not crossed and stage != "PASS" and reason != "PRE_GATE_EXISTING_OUTPUT_BLOCK").lower(),
        "SOURCE_OPENS": str(opens), "PRIVATE_CONTENT_READS": str(int(crossed)),
        "NETWORK_REQUESTS": "0", "PRICE_PAYLOAD_READS": "0", "OUTCOME_READS": "0",
        "NON_T1_IDENTITIES_RETAINED": "false", "V13_UNIVERSE_SELECTED": "false",
        "SOURCE_BINDING_MATCH": binding, "T1_HASH_MATCH": t1_hash,
        "CONSUMED_RECEIPT_WRITTEN": str(receipt).lower(),
        "PRIVATE_STATE_WRITTEN": str(state).lower(), "EXECUTION_RESULT": stage,
        "FAILURE_CLASS": reason, "AUTOMATIC_RETRY": "false", "SECOND_PRIVATE_SOURCE_READ": "false",
    }
    return "\n".join(f"{key}={value}" for key, value in fields.items())


def execute(repo: Path, local_app_data: Path, *,
            opener: Callable[[Path], BinaryIO] | None = None,
            receipt_writer: Callable[[Path, bytes], None] = _write_once) -> str:
    """Synthetic seam. The reviewed wrapper must complete external gates first."""
    try:
        source, receipt, state = _preflight(repo, local_app_data)
    except (Block, OSError, ValueError) as exc:
        reason = exc.reason if isinstance(exc, Block) else "PRE_GATE_BLOCK"
        return _report(stage="PRE_GATE_STOP", reason=reason, crossed=False,
                       receipt=False, state=False, opens=0)
    crossed = False
    receipt_written = False
    state_written = False
    opens = 0
    binding = "unknown"
    t1_hash = "unknown"
    try:
        # Unbuffered open prevents read(1) from prefetching the remainder
        # before the durable consumed receipt is published.
        with (opener(source) if opener else source.open("rb", buffering=0)) as stream:
            opens = 1
            first = stream.read(1)
            if not first:
                raise Block("EMPTY_SOURCE")
            crossed = True
            try:
                receipt_writer(receipt, _receipt())
                receipt_written = True
            except Exception:
                raise Block("POST_BOUNDARY_RECEIPT_PUBLISH_FAILED") from None
            result = resolve(stream, first, state)
            state_written = result["state_written"]
            binding = "true" if result["binding_match"] else "false"
            t1_hash = "true" if result["t1_hash_match"] else "false"
    except Exception as exc:
        reason = exc.reason if isinstance(exc, Block) else "PRIVATE_READ_OR_PUBLICATION_FAILED"
        return _report(stage="POST_BOUNDARY_FAILURE" if crossed else "PRE_BOUNDARY_FAILURE",
                       reason=reason, crossed=crossed, receipt=receipt_written,
                       state=state_written, opens=opens, binding=binding, t1_hash=t1_hash)
    return _report(stage="PASS", reason="NONE", crossed=True, receipt=True,
                   state=True, opens=1, binding="true", t1_hash="true")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args()
    if os.environ.get("V13_JQUANTS_T1_WRAPPER_GATE") != "REVIEWED_PRE_GATE_PASS":
        print(_report(stage="PRE_GATE_STOP", reason="PRE_GATE_WRAPPER_REQUIRED",
                      crossed=False, receipt=False, state=False, opens=0))
        return 1
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    report = execute(Path(args.repository_root), Path(local_app_data))
    print(report)
    return 0 if "EXECUTION_RESULT=PASS" in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
