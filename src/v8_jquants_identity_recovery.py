"""Frozen V8 J-Quants identity recovery. No network or private I/O on import."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from src import v8_partition as historical

RAW_CONTENT_LOCK_SCHEMA = "V8_JQUANTS_RAW_CONTENT_LOCK_V1"
RECOVERY_ARTIFACT_SCHEMA = "V8_JQUANTS_IDENTITY_RECOVERY_MANIFEST_V1"
ENDPOINT = "https://api.jquants.com/v2/equities/master"
QUERY_DATE = "20260731"
EFFECTIVE_DATE = "2026-07-31"
MARKETS = frozenset(("0111", "0112"))
PRODUCTS = frozenset(("011",))
BLOCK_SIZE = 300
MAX_PAGES = 100
PINS = {
    "eligible": "37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405",
    "T0": "12777a83f259cd885ebb828e0ce895a5bf53be37c27928c1a487f629002ce4f7",
    "T1": "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d",
    "T2": "e7578db7202dcb6407d7bcd98d6365fc65f22e30aa05467313a347f9cc3d6500",
    "T3": "43a585f4c3341307e7c67561c54780322b0f253fefa628a7c6129773901a7b7a",
    "T_spare": "360d5c874e6c08471f118af8ac450dadb38ca138fecd1ecdb834cc08156a9e70",
}
REASONS = frozenset((
    "NONE", "PRE_GATE_REPOSITORY_BLOCK", "PRE_GATE_PROVENANCE_BLOCK",
    "PRE_GATE_ENVIRONMENT_BLOCK", "PRE_GATE_CREDENTIAL_BLOCK",
    "PRE_GATE_PRIVATE_ROOT_BLOCK", "PRE_GATE_EXISTING_ARTIFACT_BLOCK",
    "SOURCE_TIMEOUT", "SOURCE_TRANSPORT_FAILED", "SOURCE_HTTP_429_EXHAUSTED",
    "SOURCE_HTTP_5XX_EXHAUSTED", "SOURCE_HTTP_4XX", "SOURCE_RESPONSE_SCHEMA_INVALID",
    "PAGINATION_LOOP", "PAGINATION_LIMIT", "RAW_CONTENT_LOCK_PUBLICATION_FAILED",
    "EFFECTIVE_DATE_MISMATCH", "ELIGIBLE_COUNT_MISMATCH", "ELIGIBLE_HASH_MISMATCH",
    "T0_HASH_MISMATCH", "T1_HASH_MISMATCH", "T2_HASH_MISMATCH",
    "T3_HASH_MISMATCH", "T_SPARE_HASH_MISMATCH",
    "RECOVERY_ARTIFACT_PUBLICATION_FAILED", "UNEXPECTED_FAILURE",
))
STAGES = frozenset(("PRE_GATE", "SOURCE_ACQUISITION", "RAW_CONTENT_LOCK",
                    "OFFLINE_SEMANTICS", "RECOVERY_PUBLICATION", "COMPLETE"))
REPORT_FIELDS = ("JQUANTS_RECOVERY_RESULT", "NETWORK_BOUNDARY_CROSSED",
                 "JQUANTS_LOGICAL_ACQUISITIONS", "JQUANTS_HTTP_REQUESTS", "STAGE", "REASON",
                 "RAW_CONTENT_LOCK_PUBLISHED", "RECOVERY_ARTIFACT_PUBLISHED",
                 "ELIGIBLE_COUNT", "ELIGIBLE_HASH_MATCH", "T0_HASH_MATCH", "T1_HASH_MATCH",
                 "T2_HASH_MATCH", "T3_HASH_MATCH", "T_SPARE_HASH_MATCH")
RAW_KEYS = frozenset(("schema", "endpoint", "query_date", "pages", "page_count",
                      "complete", "source_commit", "source_blob", "sha256"))
PAGE_KEYS = frozenset(("index", "byte_count", "sha256"))
RECOVERY_KEYS = frozenset(("schema", "contract", "endpoint", "query_date", "effective_date",
    "markets", "products", "canonical_order", "block_size", "raw_schema", "raw_manifest_sha256",
    "raw_pages", "eligible_count", "eligible_sha256", "block_sizes", "block_hashes", "assignments",
    "source_commit", "source_blob", "recovery_timestamp_utc", "sha256"))
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
CODE4 = re.compile(r"[0-9A-Z]{4}\Z")
CODE5 = re.compile(r"[0-9A-Z]{5}\Z")


class Block(Exception):
    def __init__(self, reason: str):
        super().__init__(reason if reason in REASONS else "UNEXPECTED_FAILURE")
        self.reason = reason if reason in REASONS else "UNEXPECTED_FAILURE"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode("ascii")


def seal(value: dict) -> dict:
    result = dict(value)
    result["sha256"] = digest(canonical(result))
    return result


def _schema(condition: bool, reason: str) -> None:
    if not condition:
        raise Block(reason)


def _exact_json(raw: bytes, reason: str) -> dict:
    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (UnicodeError, ValueError, TypeError):
        raise Block(reason) from None
    _schema(type(value) is dict, reason)
    return value


def envelope(raw: bytes) -> tuple[list[dict], str | None]:
    """Inspect only transport envelope; rows remain opaque until lock publication."""
    value = _exact_json(raw, "SOURCE_RESPONSE_SCHEMA_INVALID")
    _schema(set(value) in ({"data"}, {"data", "pagination_key"}), "SOURCE_RESPONSE_SCHEMA_INVALID")
    rows, token = value["data"], value.get("pagination_key")
    _schema(type(rows) is list and all(type(row) is dict for row in rows), "SOURCE_RESPONSE_SCHEMA_INVALID")
    _schema("pagination_key" not in value or (type(token) is str and bool(token)), "SOURCE_RESPONSE_SCHEMA_INVALID")
    return rows, token


def verify_seal(value: dict, keys: frozenset[str], reason: str) -> None:
    _schema(type(value) is dict and set(value) == keys, reason)
    actual = value["sha256"]
    _schema(type(actual) is str and HEX64.fullmatch(actual) is not None, reason)
    _schema(digest(canonical({k: v for k, v in value.items() if k != "sha256"})) == actual, reason)


def validate_raw_manifest(value: dict) -> None:
    reason = "RAW_CONTENT_LOCK_PUBLICATION_FAILED"
    verify_seal(value, RAW_KEYS, reason)
    _schema(value["schema"] == RAW_CONTENT_LOCK_SCHEMA and value["endpoint"] == ENDPOINT
            and value["query_date"] == QUERY_DATE and value["complete"] is True, reason)
    _schema(type(value["source_commit"]) is str and HEX40.fullmatch(value["source_commit"]) is not None
            and type(value["source_blob"]) is str and HEX40.fullmatch(value["source_blob"]) is not None, reason)
    pages = value["pages"]
    _schema(type(pages) is list and 1 <= len(pages) <= MAX_PAGES
            and type(value["page_count"]) is int and value["page_count"] == len(pages), reason)
    for i, page in enumerate(pages, 1):
        _schema(type(page) is dict and set(page) == PAGE_KEYS and type(page["index"]) is int
                and page["index"] == i and type(page["byte_count"]) is int and page["byte_count"] > 0
                and type(page["sha256"]) is str and HEX64.fullmatch(page["sha256"]) is not None, reason)


def _page_name(index: int) -> str:
    return f"page-{index:03d}.json"


def _page_meta_name(index: int) -> str:
    return f"page-{index:03d}.meta.json"


def _write_new(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _reject_reparse(path: Path, reason: str) -> None:
    if hasattr(path, "is_junction") and path.is_junction():
        raise Block(reason)
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        raise Block(reason)
    if os.name == "nt":
        import stat
        if path.stat(follow_symlinks=False).st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise Block(reason)


def _safe_root(repo: Path, local_app_data: Path) -> Path:
    reason = "PRE_GATE_PRIVATE_ROOT_BLOCK"
    anchor = local_app_data.absolute()
    _schema(anchor.is_absolute(), reason)
    root = anchor / "stock-analyzer" / "private" / "v8-jquants-identity-recovery"
    repo_resolved = repo.resolve(strict=True)
    for path in (anchor, *anchor.parents, root, *root.parents):
        _reject_reparse(path, reason)
    resolved = root.resolve(strict=False)
    _schema(resolved != repo_resolved and repo_resolved not in resolved.parents, reason)
    return root


def private_root(repo: Path) -> Path:
    value = os.environ.get("LOCALAPPDATA")
    _schema(bool(value) and Path(value).is_absolute(), "PRE_GATE_PRIVATE_ROOT_BLOCK")
    return _safe_root(repo, Path(value))


def inspect_state_metadata(root: Path) -> str:
    """Inspect durable-state topology only; never open a private artifact body."""
    reason = "PRE_GATE_EXISTING_ARTIFACT_BLOCK"
    _reject_reparse(root, reason)
    if not root.exists():
        return "absent"
    _schema(root.is_dir(), reason)
    children = list(root.iterdir())
    _schema(len(children) == len({path.name for path in children}), reason)
    _schema({path.name for path in children} <= {"eq-master-20260731", "recovery.json"}, reason)
    raw, recovered = root / "eq-master-20260731", root / "recovery.json"
    for path in children:
        _reject_reparse(path, reason)
    _schema(not recovered.exists() or recovered.is_file(), reason)
    _schema(not raw.exists() or raw.is_dir(), reason)
    _schema(not recovered.exists() or raw.exists(), reason)
    if raw.exists():
        entries = list(raw.iterdir())
        names = {path.name for path in entries}
        _schema(len(entries) == len(names) and "manifest.json" in names, reason)
        page_names = names - {"manifest.json"}
        _schema(1 <= len(page_names) <= MAX_PAGES and
                page_names == {_page_name(i) for i in range(1, len(page_names) + 1)}, reason)
        for path in entries:
            _reject_reparse(path, reason)
            _schema(path.is_file(), reason)
    return "complete" if recovered.exists() else "raw" if raw.exists() else "absent"


def inspect_state(root: Path) -> str:
    """Validate private artifact content after reviewed execution binding."""
    state = inspect_state_metadata(root)
    raw, recovered = root / "eq-master-20260731", root / "recovery.json"
    if state in ("raw", "complete"):
        try:
            manifest, _ = load_raw(raw)
        except Block:
            raise Block("PRE_GATE_EXISTING_ARTIFACT_BLOCK") from None
        if state == "complete":
            try:
                value = _exact_json(recovered.read_bytes(), "PRE_GATE_EXISTING_ARTIFACT_BLOCK")
                validate_recovery(value, manifest)
                _schema(recovered.read_bytes() == canonical(value), "PRE_GATE_EXISTING_ARTIFACT_BLOCK")
            except (Block, OSError, KeyError, TypeError, ValueError, historical.V8PartitionBlocked):
                raise Block("PRE_GATE_EXISTING_ARTIFACT_BLOCK") from None
        return state
    return "absent"


def load_raw(directory: Path) -> tuple[dict, list[bytes]]:
    reason = "RAW_CONTENT_LOCK_PUBLICATION_FAILED"
    try:
        _reject_reparse(directory, reason)
        _reject_reparse(directory / "manifest.json", reason)
        manifest_raw = (directory / "manifest.json").read_bytes()
        manifest = _exact_json(manifest_raw, reason)
        validate_raw_manifest(manifest)
        _schema(manifest_raw == canonical(manifest), reason)
        expected = {"manifest.json"} | {_page_name(p["index"]) for p in manifest["pages"]}
        _schema({p.name for p in directory.iterdir()} == expected, reason)
        pages = []
        seen = set()
        for page in manifest["pages"]:
            _reject_reparse(directory / _page_name(page["index"]), reason)
            raw = (directory / _page_name(page["index"])).read_bytes()
            _schema(len(raw) == page["byte_count"] and digest(raw) == page["sha256"], reason)
            _, token = envelope(raw)
            if token is not None:
                _schema(token not in seen, reason)
                seen.add(token)
            _schema((token is None) == (page["index"] == manifest["page_count"]), reason)
            pages.append(raw)
        return manifest, pages
    except Block as exc:
        if exc.reason == reason:
            raise
        raise Block(reason) from None
    except (OSError, TypeError, KeyError):
        raise Block(reason) from None


def _default_transport(token: str | None, key: str) -> tuple[int, bytes]:
    params = {"date": QUERY_DATE}
    if token is not None:
        params["pagination_key"] = token
    url = ENDPOINT + "?" + urllib.parse.urlencode(params)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, request, fp, code, msg, headers, newurl):
            return None
    opener = urllib.request.build_opener(NoRedirect, urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"x-api-key": key}, method="GET")
    try:
        with opener.open(request, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, b""
        finally:
            exc.close()


def acquire(root: Path, key: str, commit: str, source_blob: str,
            transport: Callable[[str | None, str], tuple[int, bytes]] = _default_transport,
            sleeper: Callable[[int], None] = time.sleep,
            count_attempt: Callable[[], None] = lambda: None,
            repository_root: Path | None = None) -> Path:
    """Acquire one logical query. A failed staging tree remains for adjudication."""
    _schema(HEX40.fullmatch(commit) is not None and HEX40.fullmatch(source_blob) is not None,
            "PRE_GATE_PROVENANCE_BLOCK")
    _schema(inspect_state(root) == "absent", "PRE_GATE_EXISTING_ARTIFACT_BLOCK")
    try:
        root.mkdir(parents=True, exist_ok=True)
        stage = root / ("staging-" + uuid.uuid4().hex)
        stage.mkdir(mode=0o700)
    except OSError:
        raise Block("PRE_GATE_PRIVATE_ROOT_BLOCK") from None
    records, seen = [], set()
    token = None
    for index in range(1, MAX_PAGES + 1):
        for attempt in range(3):
            try:
                count_attempt()
                status, raw = transport(token, key)
            except (TimeoutError, socket.timeout):
                reason = "SOURCE_TIMEOUT"
            except urllib.error.URLError as exc:
                reason = "SOURCE_TIMEOUT" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "SOURCE_TRANSPORT_FAILED"
            except (ConnectionError, ssl.SSLError, OSError):
                reason = "SOURCE_TRANSPORT_FAILED"
            else:
                _schema(type(status) is int and type(raw) is bytes, "SOURCE_RESPONSE_SCHEMA_INVALID")
                if status == 200:
                    break
                if status == 429:
                    reason = "SOURCE_HTTP_429_EXHAUSTED"
                elif 500 <= status <= 599:
                    reason = "SOURCE_HTTP_5XX_EXHAUSTED"
                elif 400 <= status <= 499:
                    raise Block("SOURCE_HTTP_4XX")
                else:
                    raise Block("SOURCE_RESPONSE_SCHEMA_INVALID")
            if attempt == 2:
                raise Block(reason)
            sleeper((2, 5)[attempt])
        try:
            _write_new(stage / _page_name(index), raw)
        except OSError:
            raise Block("RAW_CONTENT_LOCK_PUBLICATION_FAILED") from None
        _, next_token = envelope(raw)
        record = {"index": index, "byte_count": len(raw), "sha256": digest(raw)}
        try:
            _write_new(stage / _page_meta_name(index), canonical(record))
        except OSError:
            raise Block("RAW_CONTENT_LOCK_PUBLICATION_FAILED") from None
        records.append(record)
        if next_token is None:
            break
        if next_token in seen:
            raise Block("PAGINATION_LOOP")
        seen.add(next_token)
        token = next_token
    else:
        raise Block("PAGINATION_LIMIT")
    manifest = seal({"schema": RAW_CONTENT_LOCK_SCHEMA, "endpoint": ENDPOINT,
                     "query_date": QUERY_DATE, "pages": records, "page_count": len(records),
                     "complete": True, "source_commit": commit, "source_blob": source_blob})
    try:
        for record in records:
            raw = (stage / _page_name(record["index"])).read_bytes()
            _schema(len(raw) == record["byte_count"] and digest(raw) == record["sha256"],
                    "RAW_CONTENT_LOCK_PUBLICATION_FAILED")
            receipt_raw = (stage / _page_meta_name(record["index"])).read_bytes()
            receipt = _exact_json(receipt_raw, "RAW_CONTENT_LOCK_PUBLICATION_FAILED")
            _schema(set(receipt) == PAGE_KEYS and receipt_raw == canonical(receipt)
                    and all(type(receipt[key]) is type(record[key]) and receipt[key] == record[key]
                            for key in PAGE_KEYS), "RAW_CONTENT_LOCK_PUBLICATION_FAILED")
        _schema(manifest["pages"] == records, "RAW_CONTENT_LOCK_PUBLICATION_FAILED")
        _write_new(stage / "manifest.json", canonical(manifest))
        for record in records:
            (stage / _page_meta_name(record["index"])).unlink()
        load_raw(stage)
        if repository_root is not None:
            _schema(private_root(repository_root) == root, "RAW_CONTENT_LOCK_PUBLICATION_FAILED")
        final = root / "eq-master-20260731"
        _schema(not final.exists(), "RAW_CONTENT_LOCK_PUBLICATION_FAILED")
        stage.rename(final)  # Windows refuses an existing destination directory.
        load_raw(final)
        return final
    except Block:
        raise
    except OSError:
        raise Block("RAW_CONTENT_LOCK_PUBLICATION_FAILED") from None


def normalize_code(value: object) -> str | None:
    code = str(value).strip().upper()
    if CODE5.fullmatch(code) and code.endswith("0"):
        code = code[:-1]
    return code if CODE4.fullmatch(code) else None


def semantic(pages: list[bytes], pins: Mapping[str, str] = PINS,
             required_count: int = 3110, block_size: int = BLOCK_SIZE) -> tuple[list[str], dict[str, list[str]], dict[str, str]]:
    codes = set()
    for raw in pages:
        rows, _ = envelope(raw)
        for row in rows:
            if row.get("Mkt") not in MARKETS or row.get("ProdCat") not in PRODUCTS:
                continue
            code = normalize_code(row.get("Code"))
            if code is None:
                continue
            if row.get("Date") != EFFECTIVE_DATE:
                raise Block("EFFECTIVE_DATE_MISMATCH")
            codes.add(code)
    ordered = historical.canonical_order(list(codes))
    if len(ordered) != required_count:
        raise Block("ELIGIBLE_COUNT_MISMATCH")
    hashes = {"eligible": historical.ticker_list_sha256(ordered)}
    if hashes["eligible"] != pins["eligible"]:
        raise Block("ELIGIBLE_HASH_MISMATCH")
    t0 = ordered[:block_size]
    hashes["T0"] = historical.ticker_list_sha256(t0)
    if hashes["T0"] != pins["T0"]:
        raise Block("T0_HASH_MISMATCH")
    try:
        blocks = historical.allocate_fresh_blocks(ordered, t0, block_size=block_size)
    except historical.V8PartitionBlocked:
        raise Block("UNEXPECTED_FAILURE") from None
    for name in ("T1", "T2", "T3", "T_spare"):
        hashes[name] = historical.ticker_list_sha256(blocks[name])
        if hashes[name] != pins[name]:
            raise Block(name.upper() + "_HASH_MISMATCH")
    return ordered, blocks, hashes


def build_recovery(raw_manifest: dict, ordered: list[str], blocks: dict[str, list[str]],
                   hashes: dict[str, str], timestamp: str, block_size: int = BLOCK_SIZE) -> dict:
    validate_raw_manifest(raw_manifest)
    value = {"schema": RECOVERY_ARTIFACT_SCHEMA, "contract": "V13_V8_JQUANTS_IDENTITY_RECOVERY_DESIGN",
             "endpoint": ENDPOINT, "query_date": QUERY_DATE, "effective_date": EFFECTIVE_DATE,
             "markets": sorted(MARKETS), "products": sorted(PRODUCTS),
             "canonical_order": "SHA256_UTF8_CODE_THEN_CODE_ASC", "block_size": block_size,
             "raw_schema": RAW_CONTENT_LOCK_SCHEMA, "raw_manifest_sha256": raw_manifest["sha256"],
             "raw_pages": raw_manifest["pages"], "eligible_count": len(ordered),
             "eligible_sha256": hashes["eligible"],
             "block_sizes": {name: len(blocks[name]) for name in ("T0", "T1", "T2", "T3", "T_spare")},
             "block_hashes": {name: hashes[name] for name in ("T0", "T1", "T2", "T3", "T_spare")},
             "assignments": {"eligible": ordered, **blocks},
             "source_commit": raw_manifest["source_commit"], "source_blob": raw_manifest["source_blob"],
             "recovery_timestamp_utc": timestamp}
    return seal(value)


def validate_recovery(value: dict, raw_manifest: dict, pins: Mapping[str, str] = PINS,
                      required_count: int = 3110, block_size: int = BLOCK_SIZE) -> None:
    reason = "RECOVERY_ARTIFACT_PUBLICATION_FAILED"
    verify_seal(value, RECOVERY_KEYS, reason)
    _schema(value["schema"] == RECOVERY_ARTIFACT_SCHEMA and value["contract"] == "V13_V8_JQUANTS_IDENTITY_RECOVERY_DESIGN"
            and value["endpoint"] == ENDPOINT and value["query_date"] == QUERY_DATE
            and value["effective_date"] == EFFECTIVE_DATE and value["markets"] == sorted(MARKETS)
            and value["products"] == sorted(PRODUCTS) and value["canonical_order"] == "SHA256_UTF8_CODE_THEN_CODE_ASC"
            and value["block_size"] == block_size and value["raw_schema"] == RAW_CONTENT_LOCK_SCHEMA
            and value["raw_manifest_sha256"] == raw_manifest["sha256"] and value["raw_pages"] == raw_manifest["pages"]
            and value["source_commit"] == raw_manifest["source_commit"]
            and value["source_blob"] == raw_manifest["source_blob"], reason)
    assignments = value["assignments"]
    _schema(type(assignments) is dict and set(assignments) == {"eligible", "T0", "T1", "T2", "T3", "T_spare"}, reason)
    ordered = assignments["eligible"]
    _schema(type(ordered) is list and len(ordered) == required_count
            and all(type(code) is str and CODE4.fullmatch(code) for code in ordered)
            and historical.canonical_order(ordered) == ordered, reason)
    _schema(value["eligible_count"] == required_count and value["eligible_sha256"] == pins["eligible"]
            and historical.ticker_list_sha256(ordered) == pins["eligible"], reason)
    expected = {"T0": ordered[:block_size]}
    expected.update(historical.allocate_fresh_blocks(ordered, expected["T0"], block_size=block_size))
    for name in ("T0", "T1", "T2", "T3", "T_spare"):
        _schema(assignments[name] == expected[name] and value["block_sizes"][name] == len(expected[name])
                and value["block_hashes"][name] == pins[name]
                and historical.ticker_list_sha256(expected[name]) == pins[name], reason)
    _schema(set(value["block_sizes"]) == set(expected) and set(value["block_hashes"]) == set(expected)
            and type(value["recovery_timestamp_utc"]) is str
            and value["recovery_timestamp_utc"].endswith("Z"), reason)


def publish_recovery(root: Path, value: dict, raw_manifest: dict,
                     pins: Mapping[str, str] = PINS, required_count: int = 3110,
                     block_size: int = BLOCK_SIZE) -> None:
    reason = "RECOVERY_ARTIFACT_PUBLICATION_FAILED"
    try:
        validate_recovery(value, raw_manifest, pins, required_count, block_size)
        target = root / "recovery.json"
        _reject_reparse(root, reason)
        _reject_reparse(target, reason)
        _schema(not target.exists(), reason)
        staged = root / ("staging-recovery-" + uuid.uuid4().hex + ".json")
        _write_new(staged, canonical(value))
        _schema(staged.read_bytes() == canonical(value), reason)
        os.link(staged, target)  # Atomic create-new; an existing target is never replaced.
        staged.unlink()
        _schema(target.read_bytes() == canonical(value), reason)
    except Block:
        raise
    except (OSError, KeyError, TypeError, ValueError):
        raise Block(reason) from None


def report(result: str = "BLOCK", crossed: bool = False, acquisitions: int = 0,
           requests: int = 0, stage: str = "PRE_GATE", reason: str = "UNEXPECTED_FAILURE",
           raw_published: bool = False, recovery_published: bool = False,
           eligible_count: int | None = None, matches: Mapping[str, str] | None = None) -> str:
    _schema(result in ("PASS", "BLOCK") and stage in STAGES and reason in REASONS
            and (reason == "NONE") == (result == "PASS") and type(crossed) is bool
            and type(raw_published) is bool and type(recovery_published) is bool
            and type(acquisitions) is int and acquisitions in (0, 1)
            and type(requests) is int and requests >= 0
            and (eligible_count is None or (type(eligible_count) is int and eligible_count >= 0)),
            "UNEXPECTED_FAILURE")
    values = {"JQUANTS_RECOVERY_RESULT": result, "NETWORK_BOUNDARY_CROSSED": str(crossed).lower(),
              "JQUANTS_LOGICAL_ACQUISITIONS": str(acquisitions), "JQUANTS_HTTP_REQUESTS": str(requests),
              "STAGE": stage, "REASON": reason,
              "RAW_CONTENT_LOCK_PUBLISHED": str(raw_published).lower(),
              "RECOVERY_ARTIFACT_PUBLISHED": str(recovery_published).lower()}
    if eligible_count is not None:
        values["ELIGIBLE_COUNT"] = str(eligible_count)
    for name, status in (matches or {}).items():
        field = name.upper() + "_HASH_MATCH"
        _schema(field in REPORT_FIELDS and status in ("true", "false", "unknown"), "UNEXPECTED_FAILURE")
        values[field] = status
    return " ".join(f"{name}={values[name]}" for name in REPORT_FIELDS if name in values)


def execute(repo: Path, commit: str, source_blob: str, key: str,
            transport: Callable[[str | None, str], tuple[int, bytes]] = _default_transport,
            sleeper: Callable[[int], None] = time.sleep,
            root_override: Path | None = None,
            pins: Mapping[str, str] = PINS, required_count: int = 3110,
            block_size: int = BLOCK_SIZE) -> str:
    """Protected runner entry. All exceptions become one closed safe line."""
    crossed = raw_published = recovered = False
    acquisitions = requests = 0
    stage = "PRE_GATE"
    try:
        root = root_override if root_override is not None else private_root(repo)
        _schema(bool(key), "PRE_GATE_CREDENTIAL_BLOCK")
        state = inspect_state(root)
        if state == "complete":
            raise Block("PRE_GATE_EXISTING_ARTIFACT_BLOCK")
        if state == "absent":
            def count() -> None:
                nonlocal requests, crossed, acquisitions, stage
                acquisitions = 1
                stage = "SOURCE_ACQUISITION"
                requests += 1
                crossed = True
            final = acquire(root, key, commit, source_blob, transport, sleeper, count,
                            repo if root_override is None else None)
        else:
            final = root / "eq-master-20260731"
        stage = "RAW_CONTENT_LOCK"
        raw_manifest, pages = load_raw(final)
        raw_published = True
        stage = "OFFLINE_SEMANTICS"
        ordered, blocks, hashes = semantic(pages, pins, required_count, block_size)
        stage = "RECOVERY_PUBLICATION"
        value = build_recovery(raw_manifest, ordered, blocks, hashes,
                               datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), block_size)
        if root_override is None:
            _schema(private_root(repo) == root, "RECOVERY_ARTIFACT_PUBLICATION_FAILED")
        publish_recovery(root, value, raw_manifest, pins, required_count, block_size)
        recovered = True
        return report("PASS", crossed, acquisitions, requests, "COMPLETE", "NONE", True, True,
                      len(ordered), {name: "true" for name in hashes})
    except Block as exc:
        reason = exc.reason
        if reason == "RAW_CONTENT_LOCK_PUBLICATION_FAILED":
            stage = "RAW_CONTENT_LOCK"
        elif reason == "RECOVERY_ARTIFACT_PUBLICATION_FAILED":
            stage = "RECOVERY_PUBLICATION"
    except Exception:
        reason = "UNEXPECTED_FAILURE"
    return report("BLOCK", crossed, acquisitions, requests, stage, reason, raw_published, recovered)


def readiness_probe() -> bool:
    """No-network operation probe used before the protected credential boundary."""
    codes = [f"{i:04d}" for i in range(1000, 1012)]
    ordered = historical.canonical_order(codes)
    blocks = historical.allocate_fresh_blocks(ordered, ordered[:1], block_size=1)
    pins = {"eligible": historical.ticker_list_sha256(ordered)}
    pins.update({name: historical.ticker_list_sha256(blocks[name]) for name in blocks})
    page1 = canonical({"data": [{"Date": EFFECTIVE_DATE, "Mkt": "0111", "ProdCat": "011",
                                 "Code": code + "0"} for code in codes[:6]], "pagination_key": "probe-only"}).rstrip(b"\n")
    page2 = canonical({"data": [{"Date": EFFECTIVE_DATE, "Mkt": "0112", "ProdCat": "011",
                                 "Code": code + "0"} for code in codes[6:]]}).rstrip(b"\n")
    calls = []
    def fake(token: str | None, _key: str) -> tuple[int, bytes]:
        calls.append(token)
        return 200, page1 if token is None else page2
    with tempfile.TemporaryDirectory(prefix="v8-jq-probe-") as location:
        root = Path(location)
        first = execute(root, "a" * 40, "b" * 40, "probe-key", fake, lambda _: None,
                        root_override=root, pins=pins, required_count=len(codes), block_size=1)
        manifest, pages = load_raw(root / "eq-master-20260731")
        second = semantic(pages, pins, len(codes), 1)
        recovered = _exact_json((root / "recovery.json").read_bytes(), "UNEXPECTED_FAILURE")
        validate_recovery(recovered, manifest, pins, len(codes), 1)
        return first.startswith("JQUANTS_RECOVERY_RESULT=PASS ") and calls == [None, "probe-only"] and bool(second)


def _protected_main_preflight(repo: Path, commit: str, source_blob: str, script_blob: str) -> None:
    """Repeat the essential runner bindings at the Python network boundary."""
    _schema(all(HEX40.fullmatch(item or "") for item in (commit, source_blob, script_blob)),
            "PRE_GATE_PROVENANCE_BLOCK")
    _schema("orca\\workspaces" not in str(repo).lower() and "orca/workspaces" not in str(repo).lower()
            and (repo / ".git").is_dir(), "PRE_GATE_REPOSITORY_BLOCK")
    _schema(Path(sys.executable).resolve() ==
            (repo / ".venv-real-execution" / "Scripts" / "python.exe").resolve(),
            "PRE_GATE_ENVIRONMENT_BLOCK")
    def git(*args: str) -> str:
        result = subprocess.run(("git", "-C", str(repo), *args), capture_output=True,
                                text=True, timeout=20, check=False)
        _schema(result.returncode == 0, "PRE_GATE_REPOSITORY_BLOCK")
        return result.stdout.strip()
    _schema(Path(git("rev-parse", "--show-toplevel")).resolve() == repo.resolve()
            and git("branch", "--show-current") == "v13-conditional-cross-sectional-short-horizon"
            and git("rev-parse", "HEAD") == commit and git("status", "--porcelain") == "",
            "PRE_GATE_REPOSITORY_BLOCK")
    origin = git("remote", "get-url", "origin")
    _schema(origin in ("https://github.com/ta1k1-arakawa/stock-analyzer.git",
                       "git@github.com:ta1k1-arakawa/stock-analyzer.git"), "PRE_GATE_REPOSITORY_BLOCK")
    remote = git("ls-remote", "--exit-code", "origin",
                 "refs/heads/v13-conditional-cross-sectional-short-horizon")
    _schema(len(remote.split()) == 2 and remote.split()[0] == commit, "PRE_GATE_REPOSITORY_BLOCK")
    for path, expected in (("src/v8_jquants_identity_recovery.py", source_blob),
                           ("scripts/run_v8_jquants_identity_recovery_direct_windows.ps1", script_blob)):
        _schema(git("rev-parse", f"{commit}:{path}") == expected
                and git("hash-object", "--path", path, path) == expected,
                "PRE_GATE_PROVENANCE_BLOCK")
    checker = subprocess.run((str(sys.executable), str(repo / "scripts" /
        "check_current_protected_environment.py")), cwd=repo, capture_output=True, timeout=60, check=False)
    _schema(checker.returncode == 0, "PRE_GATE_ENVIRONMENT_BLOCK")
    _schema(readiness_probe(), "PRE_GATE_ENVIRONMENT_BLOCK")
    root = private_root(repo)
    _schema(inspect_state_metadata(root) in ("absent", "raw"), "PRE_GATE_EXISTING_ARTIFACT_BLOCK")


def main() -> int:
    """Called only by the protected PowerShell runner after its preflight."""
    try:
        repo = Path(__file__).resolve().parents[1]
        commit = os.environ.get("V8_JQUANTS_REVIEWED_HEAD", "")
        blob = os.environ.get("V8_JQUANTS_REVIEWED_BLOB", "")
        script_blob = os.environ.get("V8_JQUANTS_REVIEWED_SCRIPT_BLOB", "")
        _protected_main_preflight(repo, commit, blob, script_blob)
        key = os.environ.get("JQUANTS_API_KEY", "")
        line = execute(repo, commit, blob, key)
    except Block as exc:
        line = report(reason=exc.reason)
    except Exception:
        line = report()
    print(line)
    return 0 if line.startswith("JQUANTS_RECOVERY_RESULT=PASS ") else 1


if __name__ == "__main__":
    raise SystemExit(main())
