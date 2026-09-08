"""V9_017 synthetic-tested public-schema discovery boundary.

The pure seam reports only the frozen structural observation universe.  The
execution wrapper is deliberately one-shot and has no network capability; it
is present for the later, separately authorized bound-input observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__ in {None, ""}:
    _script_directory = os.path.normcase(str(Path(__file__).resolve().parent))
    sys.path[:] = [
        entry
        for entry in sys.path
        if os.path.normcase(os.path.abspath(entry or os.curdir)) != _script_directory
    ]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.v9_005_stage_a_jpx_probe import _parse_monthly_statistics_html


SCHEMA_VERSION = "V9_017_DISCOVERY_EXECUTION_V1"
PURE_SCHEMA_VERSION = "V9_017_PUBLIC_SCHEMA_DISCOVERY_V1"
CONFIRMATION_CONTRACT = "V9_017_2017_PUBLIC_SCHEMA_DISCOVERY_ONE_SHOT"
TARGET_YEAR = 2017
EXPECTED_YEAR_BYTE_COUNT = 98936
EXPECTED_YEAR_SHA256 = (
    "1dc982e97b1d4ce7d52bc25631ddc46a219d22f82b393881f40e5d2478177821"
)

_EXPECTED_GIT_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")
_ROLE_ORDER = {"FIRST_CELL": 0, "TH": 1}


def _normalize(raw_text: str) -> str:
    return " ".join(raw_text.split())


class PublicSchemaDiscoveryResult:
    """Safe pure-discovery result with an exact public serialization."""

    def __init__(
        self,
        *,
        status: str,
        failure_class: str | None,
        parser_success: bool,
        table_count: int,
        tables: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> None:
        self.schema_version = PURE_SCHEMA_VERSION
        self.status = status
        self.failure_class = failure_class
        self.parser_success = parser_success
        self.table_count = table_count
        self.tables = tables
        self.observation_count = len(observations)
        self.observations = observations

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "failure_class": self.failure_class,
            "parser_success": self.parser_success,
            "table_count": self.table_count,
            "tables": self.tables,
            "observation_count": self.observation_count,
            "observations": self.observations,
        }

    def __repr__(self) -> str:
        return repr(self.to_dict())


def _failure_discovery_result() -> PublicSchemaDiscoveryResult:
    return PublicSchemaDiscoveryResult(
        status="FAIL",
        failure_class="IMPLEMENTATION_FAILURE",
        parser_success=False,
        table_count=0,
        tables=[],
        observations=[],
    )


def discover_public_schema(page_bytes: bytes) -> PublicSchemaDiscoveryResult:
    """Return only the deterministic, opaque structural evidence universe."""

    if not isinstance(page_bytes, bytes):
        return _failure_discovery_result()

    try:
        parser = _parse_monthly_statistics_html(page_bytes)
        tables: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []

        for table_index, table in enumerate(parser.tables):
            rows = table.rows
            tables.append(
                {
                    "table_index": table_index,
                    "row_count": len(rows),
                    "max_column_count": max((len(row) for row in rows), default=0),
                    "rows": [
                        {"row_index": row_index, "cell_count": len(row)}
                        for row_index, row in enumerate(rows)
                    ],
                }
            )

            for row_index, row in enumerate(rows):
                if row:
                    first_cell = row[0]
                    observations.append(
                        {
                            "table_index": table_index,
                            "row_index": row_index,
                            "column_index": 0,
                            "role": "FIRST_CELL",
                            "normalized_text": _normalize(first_cell.text),
                        }
                    )
                for column_index, cell in enumerate(row):
                    if cell.tag == "th":
                        observations.append(
                            {
                                "table_index": table_index,
                                "row_index": row_index,
                                "column_index": column_index,
                                "role": "TH",
                                "normalized_text": _normalize(cell.text),
                            }
                        )

        observations.sort(
            key=lambda item: (
                item["table_index"],
                item["row_index"],
                item["column_index"],
                _ROLE_ORDER[item["role"]],
            )
        )
        return PublicSchemaDiscoveryResult(
            status="PASS",
            failure_class=None,
            parser_success=True,
            table_count=len(parser.tables),
            tables=tables,
            observations=observations,
        )
    except Exception:
        return _failure_discovery_result()


def _safe_failure(
    *, failure_class: str, reason: str, semantic_discovery_invocations: int
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL",
        "failure_class": failure_class,
        "reason": reason,
        "semantic_discovery_invocations": semantic_discovery_invocations,
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _write_failure_if_possible(
    output_root: Path, *, failure_class: str, reason: str, invocations: int
) -> None:
    try:
        _write_exclusive_json(
            output_root / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "FAIL",
                "failure_class": failure_class,
                "reason": reason,
                "semantic_discovery_invocations": invocations,
            },
        )
    except Exception:
        return


def _run_execution(
    *,
    year_page: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    expected_git_sha: str,
    confirmation: str,
    execute_discovery: bool,
    _expected_year_byte_count: int = EXPECTED_YEAR_BYTE_COUNT,
    _expected_year_sha256: str = EXPECTED_YEAR_SHA256,
    _discovery_fn: Callable[[bytes], PublicSchemaDiscoveryResult] = discover_public_schema,
) -> dict[str, Any]:
    """Run the gated one-shot wrapper; private overrides are test-only seams."""

    if not execute_discovery:
        return _safe_failure(
            failure_class="GOVERNANCE_FAILURE",
            reason="EXECUTE_DISCOVERY_REQUIRED",
            semantic_discovery_invocations=0,
        )
    if not isinstance(expected_git_sha, str) or not _EXPECTED_GIT_SHA_RE.fullmatch(
        expected_git_sha
    ):
        return _safe_failure(
            failure_class="GOVERNANCE_FAILURE",
            reason="EXPECTED_GIT_SHA_INVALID",
            semantic_discovery_invocations=0,
        )
    if confirmation != CONFIRMATION_CONTRACT:
        return _safe_failure(
            failure_class="GOVERNANCE_FAILURE",
            reason="CONFIRMATION_CONTRACT_MISMATCH",
            semantic_discovery_invocations=0,
        )

    try:
        output_path = Path(output_root)
        if os.path.lexists(os.fspath(output_path)):
            return _safe_failure(
                failure_class="GOVERNANCE_FAILURE",
                reason="OUTPUT_ROOT_COLLISION",
                semantic_discovery_invocations=0,
            )
        output_path.mkdir(parents=True, exist_ok=False)
    except Exception:
        return _safe_failure(
            failure_class="GOVERNANCE_FAILURE",
            reason="OUTPUT_ROOT_CREATE_FAILURE",
            semantic_discovery_invocations=0,
        )

    attempt = {
        "schema_version": SCHEMA_VERSION,
        "status": "IN_PROGRESS",
        "expected_git_sha": expected_git_sha,
        "confirmation_contract": CONFIRMATION_CONTRACT,
        "target_year": TARGET_YEAR,
        "expected_year_byte_count": _expected_year_byte_count,
        "expected_year_sha256": _expected_year_sha256,
        "target_semantic_discovery_invocations": 1,
    }
    try:
        _write_exclusive_json(output_path / "attempt.json", attempt)
    except Exception:
        _write_failure_if_possible(
            output_path,
            failure_class="IMPLEMENTATION_FAILURE",
            reason="ATTEMPT_WRITE_FAILURE",
            invocations=0,
        )
        return _safe_failure(
            failure_class="IMPLEMENTATION_FAILURE",
            reason="ATTEMPT_WRITE_FAILURE",
            semantic_discovery_invocations=0,
        )

    try:
        with Path(year_page).open("rb") as stream:
            page_bytes = stream.read()
    except Exception:
        _write_failure_if_possible(
            output_path,
            failure_class="GOVERNANCE_FAILURE",
            reason="YEAR_PAGE_READ_FAILURE",
            invocations=0,
        )
        return _safe_failure(
            failure_class="GOVERNANCE_FAILURE",
            reason="YEAR_PAGE_READ_FAILURE",
            semantic_discovery_invocations=0,
        )

    if len(page_bytes) != _expected_year_byte_count:
        _write_failure_if_possible(
            output_path,
            failure_class="GOVERNANCE_FAILURE",
            reason="YEAR_PAGE_BYTE_COUNT_MISMATCH",
            invocations=0,
        )
        return _safe_failure(
            failure_class="GOVERNANCE_FAILURE",
            reason="YEAR_PAGE_BYTE_COUNT_MISMATCH",
            semantic_discovery_invocations=0,
        )
    if hashlib.sha256(page_bytes).hexdigest() != _expected_year_sha256:
        _write_failure_if_possible(
            output_path,
            failure_class="GOVERNANCE_FAILURE",
            reason="YEAR_PAGE_SHA256_MISMATCH",
            invocations=0,
        )
        return _safe_failure(
            failure_class="GOVERNANCE_FAILURE",
            reason="YEAR_PAGE_SHA256_MISMATCH",
            semantic_discovery_invocations=0,
        )

    invocations = 1
    try:
        discovery_result = _discovery_fn(page_bytes)
        safe_discovery_result = discovery_result.to_dict()
        if (
            safe_discovery_result.get("status") != "PASS"
            or safe_discovery_result.get("parser_success") is not True
        ):
            raise ValueError
    except Exception:
        _write_failure_if_possible(
            output_path,
            failure_class="IMPLEMENTATION_FAILURE",
            reason="DISCOVERY_FAILURE",
            invocations=invocations,
        )
        return _safe_failure(
            failure_class="IMPLEMENTATION_FAILURE",
            reason="DISCOVERY_FAILURE",
            semantic_discovery_invocations=invocations,
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "DISCOVERY_COMPLETE",
        "failure_class": None,
        "reason": None,
        "expected_git_sha": expected_git_sha,
        "year": TARGET_YEAR,
        "year_byte_count": _expected_year_byte_count,
        "year_sha256": _expected_year_sha256,
        "semantic_discovery_invocations": invocations,
        "discovery_result": safe_discovery_result,
    }
    try:
        _write_exclusive_json(output_path / "result.json", result)
    except Exception:
        _write_failure_if_possible(
            output_path,
            failure_class="IMPLEMENTATION_FAILURE",
            reason="RESULT_WRITE_FAILURE",
            invocations=invocations,
        )
        return _safe_failure(
            failure_class="IMPLEMENTATION_FAILURE",
            reason="RESULT_WRITE_FAILURE",
            semantic_discovery_invocations=invocations,
        )

    complete = {
        "schema_version": SCHEMA_VERSION,
        "status": "DISCOVERY_COMPLETE",
        "semantic_discovery_invocations": invocations,
    }
    try:
        _write_exclusive_json(output_path / "complete.json", complete)
    except Exception:
        _write_failure_if_possible(
            output_path,
            failure_class="IMPLEMENTATION_FAILURE",
            reason="COMPLETE_WRITE_FAILURE",
            invocations=invocations,
        )
        return _safe_failure(
            failure_class="IMPLEMENTATION_FAILURE",
            reason="COMPLETE_WRITE_FAILURE",
            semantic_discovery_invocations=invocations,
        )
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V9_017 public-schema discovery")
    parser.add_argument("--year-page", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--execute-discovery", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_argument_parser().parse_args(argv)
    safe_result = _run_execution(
        year_page=arguments.year_page,
        output_root=arguments.output_root,
        expected_git_sha=arguments.expected_git_sha,
        confirmation=arguments.confirmation,
        execute_discovery=arguments.execute_discovery,
    )
    print(json.dumps(safe_result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0 if safe_result.get("status") == "DISCOVERY_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = [
    "CONFIRMATION_CONTRACT",
    "EXPECTED_YEAR_BYTE_COUNT",
    "EXPECTED_YEAR_SHA256",
    "PURE_SCHEMA_VERSION",
    "PublicSchemaDiscoveryResult",
    "SCHEMA_VERSION",
    "TARGET_YEAR",
    "_run_execution",
    "_write_exclusive_json",
    "discover_public_schema",
    "main",
]
