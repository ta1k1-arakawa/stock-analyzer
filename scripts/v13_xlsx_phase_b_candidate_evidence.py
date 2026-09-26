"""Read-only, offline export of the consumed Phase-B wheel candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

from scripts import v13_xlsx_phase_b_resolution as phase_b
from scripts import v13_xlsx_successor_contract as successor
from scripts.v10c_t0_ml_environment_contract import ContractValidationError

EXECUTION_HEAD = "3430085b1cbb14fd2bceaf14b05db02da51c8ab0"
CANDIDATE_SHA256 = "2df61d002a780c38a361dbd51db501bc9253766ff26007f75eab214540fb2251"


def _require(value: bool, code: str) -> None:
    if not value:
        raise ValueError(code)


def _safe_node(path: Path, *, directory: bool) -> None:
    _require(os.path.lexists(path), "DURABLE_STATE_MISSING")
    info = os.lstat(path)
    _require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & 0x400),
             "REPARSE_PATH_BLOCKED")
    _require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
             "DURABLE_STATE_TYPE_INVALID")


def _safe_root() -> Path:
    root = phase_b._canonical_root(phase_b.AUTH_BLOB)
    _require(root.is_absolute() and ".." not in root.parts and "." not in root.parts,
             "OPERATION_ROOT_INVALID")
    node = root
    while True:
        _safe_node(node, directory=True)
        if node == node.parent:
            break
        node = node.parent
    repo = Path(os.path.realpath(phase_b.ROOT))
    real = Path(os.path.realpath(root))
    _require(not real.is_relative_to(repo) and not repo.is_relative_to(real), "GOVERNED_PATH_OVERLAP")
    return root


def _json_file(path: Path, raw: bytes | None = None) -> dict:
    _safe_node(path, directory=False)

    def unique(pairs: list[tuple[str, object]]) -> dict:
        value: dict = {}
        for key, item in pairs:
            _require(key not in value, "DUPLICATE_JSON_KEY")
            value[key] = item
        return value

    value = json.loads(raw if raw is not None else path.read_bytes(), object_pairs_hook=unique)
    _require(type(value) is dict, "DURABLE_JSON_INVALID")
    return value


def _exact(actual: dict, expected: dict, code: str) -> None:
    _require(type(actual) is dict and set(actual) == set(expected) and
             all(type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items()), code)


def _openpyxl_closure(inspected: dict, base: dict[str, str]) -> list[dict]:
    metadata = {item["name"]: item for item in inspected["dependency_metadata"]}
    packages = {item["name"]: item for item in inspected["resolved_packages"]}
    pending = ["openpyxl"]
    reached: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        _require(name in metadata and name in packages, "SUCCESSOR_CLOSURE_INVALID")
        reached.add(name)
        for requirement in metadata[name]["requires_dist"]:
            dependency, _, marker, extras = phase_b._parse_requirement(requirement)
            if phase_b._marker_applies(marker):
                _require(extras is None and dependency in packages, "SUCCESSOR_CLOSURE_INVALID")
                pending.append(dependency)
    closure = [packages[name] for name in sorted(reached - set(base))]
    _require(len(closure) == 2 and {item["name"] for item in closure} == set(packages) - set(base),
             "SUCCESSOR_CLOSURE_INVALID")
    return closure


def export_evidence() -> tuple[dict, str]:
    _require(sys.platform == "win32", "DIRECT_WINDOWS_REQUIRED")
    root = _safe_root()
    attempt = _json_file(root / "attempt.json")
    _exact(attempt, {
        "schema_version": "V13_XLSX_PHASE_B_ATTEMPT_V1", "status": "CONSUMED_PENDING",
        "execution_head": EXECUTION_HEAD, "authorization_blob": phase_b.AUTH_BLOB,
        "one_shot": True, "reusable": False, "package_installations": 0,
    }, "ATTEMPT_INVALID")
    candidate_path = root / "candidate.json"
    _safe_node(candidate_path, directory=False)
    candidate_raw = candidate_path.read_bytes()
    candidate = _json_file(candidate_path, candidate_raw)
    candidate_hash = hashlib.sha256(candidate_raw).hexdigest()
    _require(candidate_hash == CANDIDATE_SHA256, "CANDIDATE_SHA256_MISMATCH")
    receipt = _json_file(root / "receipt.json")
    _exact(receipt, {
        "schema_version": "V13_XLSX_PHASE_B_RECEIPT_V1", "status": "PASS_CONSUMED",
        "execution_head": EXECUTION_HEAD, "authorization_blob": phase_b.AUTH_BLOB,
        "candidate_sha256": CANDIDATE_SHA256, "package_count": 29, "artifact_count": 29,
        "package_index_request_count": 29, "wheel_download_count": 29,
        "package_installations": 0, "active_environment_mutated": False,
        "successor_promoted": False, "reusable": False,
    }, "RECEIPT_INVALID")
    base = successor.validate_direct_spec()
    _require(len(base) == 27, "CURRENT_AUTHORITY_INVALID")
    wheelhouse = root / "wheelhouse"
    _safe_node(wheelhouse, directory=True)
    for wheel in wheelhouse.iterdir():
        _safe_node(wheel, directory=False)
    inspected = phase_b.inspect_wheels(wheelhouse, base)
    expected = dict(inspected, execution_head=EXECUTION_HEAD, approved_phase_a_sha=phase_b.PHASE_A,
                    authorization_blob=phase_b.AUTH_BLOB, official_index_url=phase_b.INDEX,
                    resolver="pip download --only-binary=:all:", package_index_request_count=29,
                    wheel_download_count=29, package_installations=0,
                    active_environment_mutated=False, successor_promoted=False,
                    jpx_yahoo_requests=0, private_reads=0, selected_500_constructed=False,
                    model_fits=0, backtests=0, aq_executions=0, trades=0)
    _require(candidate == expected, "CANDIDATE_CONTENT_MISMATCH")
    packages = {p["name"]: p["version"] for p in inspected["resolved_packages"]}
    _require(len(packages) == 29 and all(packages.get(n) == v for n, v in base.items()),
             "CURRENT_PIN_DRIFT")
    closure = _openpyxl_closure(inspected, base)
    evidence = {
        "schema_version": "V13_XLSX_PHASE_B_SAFE_CANDIDATE_EVIDENCE_V1",
        "execution_head": EXECUTION_HEAD, "authorization_blob": phase_b.AUTH_BLOB,
        "candidate_sha256": CANDIDATE_SHA256, "status": "PASS_CONSUMED",
        "authorization_reusable": False, "candidate_status": "RESOLVED_NOT_INSTALLED",
        "current_authority_package_count": 27, "resolved_package_count": 29,
        "artifact_count": 29, "package_index_request_count": 29,
        "wheel_download_count": 29, "package_installations": 0,
        "active_environment_mutated": False, "successor_promoted": False,
        "openpyxl_version": packages["openpyxl"],
        "successor_closure": closure,
        "resolved_packages": inspected["resolved_packages"],
        "resolved_wheels": inspected["resolved_wheels"],
    }
    raw = (json.dumps(evidence, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    return evidence, hashlib.sha256(raw).hexdigest()


def main() -> int:
    class SafeArgumentParser(argparse.ArgumentParser):
        def error(self, message: str) -> None:
            raise ValueError("ARGUMENTS_INVALID")

    parser = SafeArgumentParser()
    parser.add_argument("--export", action="store_true")
    try:
        args = parser.parse_args()
        if not args.export:
            print(json.dumps({"rehearsal": True, "durable_state_reads": 0, "network_requests": 0,
                              "environment_mutations": 0}, sort_keys=True))
            return 0
        evidence, digest = export_evidence()
        print(json.dumps({"evidence": evidence, "evidence_sha256": digest}, sort_keys=True,
                         ensure_ascii=False, separators=(",", ":")))
        return 0
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError,
            ContractValidationError) as error:
        reason = str(error) if type(error) is ValueError else "EVIDENCE_EXPORT_FAILED"
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", reason):
            reason = "EVIDENCE_EXPORT_FAILED"
        print(json.dumps({"status": "FAIL", "reason": reason}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
