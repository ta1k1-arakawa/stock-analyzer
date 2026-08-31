"""Offline-only semantic successor locator for the already-locked F1 root."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import urllib.parse
from typing import Any

from src import v9_006_f1_locked_root_locator_successor_diagnostic as _prior

SCHEMA_VERSION = "V9_006_F1_SEMANTIC_SUCCESSOR_LOCATOR_V1"
TASK = "V9_006_F1_SEMANTIC_SUCCESSOR_LOCATOR"
EXPECTED_PAYLOAD_SHA256 = "ab19c37ca50b23798b8c12c5dc7c4abc6ba865e9e9ec73f04a7daf1247c9720f"
EXPECTED_LENGTH = 30059
EXPECTED_PRIOR_STRUCTURAL = "986029641d10d36d33219d729f2c7bdb7c5495447e91be59e11650dd807efad5"
RESULTS = frozenset({"SUCCESSOR_LOCATOR_MATCHED", "INPUT_BINDING_FAILURE", "HTML_STRUCTURE_UNSUPPORTED", "SOURCE_OR_DATA_FEASIBILITY_FAILURE", "SAFE_OUTPUT_VALIDATION_FAILURE"})
_HEX = re.compile(r"[0-9a-f]{64}")
_YEAR = re.compile(r"^List of TSE-listed Issues \((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\. [0-9]{4}\)$")
_P1 = "List of TSE-listed Issues as of previous month-end is available."
_SUPPRESS = frozenset({"script", "style", "noscript", "template"})
_EXTENSIONS = frozenset({"XLS", "XLSX", "CSV", "ZIP"})
FRESH_SCHEMA_VERSION = "V9_006_F1_SEMANTIC_SUCCESSOR_LOCATOR_FRESH_ROOT_V1"


class _Unsupported(Exception):
    pass


class _Unsafe(Exception):
    pass


class _FreshInputContractViolation(TypeError):
    """Raised before safe projection when fresh input has no verified byte identity."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: str) -> str:
    return _prior._text(value)


def _extension(url: str) -> str:
    path = urllib.parse.unquote(urllib.parse.urlsplit(url).path).lower()
    for suffix, kind in ((".xlsx", "XLSX"), (".xls", "XLS"), (".csv", "CSV"), (".zip", "ZIP")):
        if path.endswith(suffix):
            return kind
    return "OTHER"


def _is_mechanical_candidate(base_url: str, has_href: bool, href: object) -> bool:
    if not has_href or type(href) is not str:
        return False
    try:
        resolved = urllib.parse.urljoin(base_url, href)
        _prior.validate_jpx_url(resolved)
        return _extension(resolved) in _EXTENSIONS
    except Exception:
        return False


class _Parser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.depth = {tag: 0 for tag in _SUPPRESS}
        self.tokens: list[str] = []
        self.anchors: list[dict[str, Any]] = []
        self.active: dict[str, Any] | None = None
        self.title: list[str] | None = None
        self.title_seen = False

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SUPPRESS:
            return
        if tag == "a" or tag == "title" or re.fullmatch(r"h[1-6]", tag):
            raise _Unsupported()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SUPPRESS:
            self.depth[tag] += 1; return
        if tag == "title":
            if self.title is not None or self.title_seen: raise _Unsupported()
            self.title, self.title_seen = [], True; return
        if re.fullmatch(r"h[1-6]", tag):
            return
        if tag != "a": return
        if self.active is not None: raise _Unsupported()
        hrefs = [value for name, value in attrs if name.lower() == "href"]
        if len(hrefs) > 1: raise _Unsupported()
        self.active = {"ordinal": len(self.anchors) + 1, "href": hrefs[0] if hrefs else None, "has_href": bool(hrefs), "before": list(self.tokens), "internal_tokens": []}

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SUPPRESS:
            if self.depth[tag] == 0: raise _Unsupported()
            self.depth[tag] -= 1; return
        if tag == "title":
            if self.title is None: raise _Unsupported()
            self.title = None; return
        if re.fullmatch(r"h[1-6]", tag): return
        if tag == "a":
            if self.active is None: raise _Unsupported()
            anchor = self.active; self.active = None
            if not _is_mechanical_candidate(self.base_url, anchor["has_href"], anchor["href"]):
                self.tokens.extend(anchor["internal_tokens"])
            self.anchors.append(anchor)

    def handle_data(self, data: str) -> None:
        if self.title is not None: self.title.append(data)
        if all(depth == 0 for depth in self.depth.values()):
            token = _text(data)
            if token:
                if _prior._UNSAFE_TEXT.search(token): raise _Unsafe()
                if self.active is None:
                    self.tokens.append(token)
                else:
                    self.active["internal_tokens"].append(token)

    def checked_close(self) -> None:
        self.close()
        if self.active is not None or self.title is not None or any(self.depth.values()): raise _Unsupported()


def _safe_candidate(candidate: dict[str, str]) -> dict[str, str]:
    return {key: candidate[key] for key in ("raw_href_sha256", "resolved_url_sha256")}


def _locate_private(raw: bytes, resolved_root_url: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return private candidates; callers must never serialize raw hrefs or URLs."""
    parser = _Parser(resolved_root_url)
    try:
        parser.feed(raw.decode("utf-8", errors="replace")); parser.checked_close()
    except (_Unsupported, _Unsafe): raise
    except Exception as exc: raise _Unsupported() from exc
    mechanical: list[dict[str, str]] = []
    qualified: list[dict[str, str]] = []
    for anchor in parser.anchors:
        href = anchor["href"]
        if not anchor["has_href"] or not isinstance(href, str): continue
        try:
            resolved = urllib.parse.urljoin(resolved_root_url, href)
            _prior.validate_jpx_url(resolved)
        except Exception:
            continue
        if _extension(resolved) not in _EXTENSIONS: continue
        item = {"raw_href": href, "resolved_url": resolved, "raw_href_sha256": sha256(href.encode("utf-8")).hexdigest(), "resolved_url_sha256": sha256(resolved.encode("utf-8")).hexdigest()}
        mechanical.append(item)
        before = anchor["before"]
        if len(before) >= 2 and _YEAR.fullmatch(before[-1]) and before[-2] == _P1:
            qualified.append(item)
    return mechanical, qualified


def locate_html(raw: bytes, resolved_root_url: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return hash-only mechanical and semantically qualifying candidates."""
    mechanical, qualified = _locate_private(raw, resolved_root_url)
    return [_safe_candidate(item) for item in mechanical], [_safe_candidate(item) for item in qualified]


def _post_uniqueness_revalidate(bound_resolved_root_url: str, selected: dict[str, str]) -> bool:
    """Re-resolve/re-validate the private selected href and require exact identity."""
    try:
        recomputed = urllib.parse.urljoin(bound_resolved_root_url, selected["raw_href"])
        _prior.validate_jpx_url(recomputed)
        return recomputed == selected["resolved_url"] and sha256(recomputed.encode("utf-8")).hexdigest() == selected["resolved_url_sha256"]
    except Exception:
        return False


@dataclass(frozen=True)
class _Selection:
    """Private shared semantic-selection result; never serialize `selected`."""
    result: str
    mechanical_candidate_count: int
    qualifying_candidate_count: int
    selected: dict[str, str] | None


def _run_selection_core(raw: bytes, resolved_root_url: str) -> _Selection:
    """Run the frozen selector sequence after a wrapper's input binding passes."""
    try:
        mechanical, qualifying = _locate_private(raw, resolved_root_url)
    except _Unsupported:
        return _Selection("HTML_STRUCTURE_UNSUPPORTED", 0, 0, None)
    except _Unsafe:
        return _Selection("SAFE_OUTPUT_VALIDATION_FAILURE", 0, 0, None)
    except Exception:
        return _Selection("INPUT_BINDING_FAILURE", 0, 0, None)
    if len(qualifying) != 1:
        return _Selection("SOURCE_OR_DATA_FEASIBILITY_FAILURE", 0, 0, None)
    selected = qualifying[0]
    if not _post_uniqueness_revalidate(resolved_root_url, selected):
        return _Selection("INPUT_BINDING_FAILURE", 0, 0, None)
    return _Selection("SUCCESSOR_LOCATOR_MATCHED", len(mechanical), 1, selected)


def _empty(result: str, payload: str | None = None) -> dict[str, object]:
    return {"schema_version": SCHEMA_VERSION, "task": TASK, "input_payload_sha256": payload, "result": result, "mechanical_candidate_count": 0, "qualifying_candidate_count": 0, "selected_raw_href_sha256": None, "selected_resolved_url_sha256": None, "network_requests": 0, "replacement_locator_authorized": False}


def _finalize(value: dict[str, object]) -> dict[str, object]:
    result = dict(value); result["structural_evidence_sha256"] = sha256(canonical_json(result).encode("utf-8")).hexdigest(); validate_safe_result(result); return result


def validate_safe_result(value: object) -> None:
    keys = {"schema_version", "task", "input_payload_sha256", "result", "mechanical_candidate_count", "qualifying_candidate_count", "selected_raw_href_sha256", "selected_resolved_url_sha256", "network_requests", "replacement_locator_authorized", "structural_evidence_sha256"}
    if type(value) is not dict or set(value) != keys or value["schema_version"] != SCHEMA_VERSION or value["task"] != TASK or value["result"] not in RESULTS or type(value["network_requests"]) is not int or value["network_requests"] != 0 or value["replacement_locator_authorized"] is not False: raise ValueError("schema")
    digestless = dict(value); digest = digestless.pop("structural_evidence_sha256")
    if type(digest) is not str or _HEX.fullmatch(digest) is None or sha256(canonical_json(digestless).encode("utf-8")).hexdigest() != digest: raise ValueError("hash")
    payload = value["input_payload_sha256"]
    if payload is not None and (type(payload) is not str or _HEX.fullmatch(payload) is None): raise ValueError("payload")
    for key in ("mechanical_candidate_count", "qualifying_candidate_count"):
        if type(value[key]) is not int or value[key] < 0: raise ValueError("count")
    hashes = (value["selected_raw_href_sha256"], value["selected_resolved_url_sha256"])
    if any(item is not None and (type(item) is not str or _HEX.fullmatch(item) is None) for item in hashes): raise ValueError("selected")
    if value["result"] == "SUCCESSOR_LOCATOR_MATCHED":
        if payload != EXPECTED_PAYLOAD_SHA256 or value["mechanical_candidate_count"] < 1 or value["qualifying_candidate_count"] != 1 or any(item is None for item in hashes): raise ValueError("success")
    elif payload is not None and payload != EXPECTED_PAYLOAD_SHA256: raise ValueError("failure payload")
    elif value["mechanical_candidate_count"] != 0 or value["qualifying_candidate_count"] != 0 or any(item is not None for item in hashes): raise ValueError("failure")


def _assert_prior(value: object) -> None:
    _prior.validate_safe_result(value)
    inp = value["input"]
    if value["diagnostic_result"] != "EVIDENCE_CAPTURED" or value["structural_evidence_sha256"] != EXPECTED_PRIOR_STRUCTURAL or inp["payload_sha256"] != EXPECTED_PAYLOAD_SHA256 or inp["byte_length"] != EXPECTED_LENGTH or inp["source_family"] != _prior.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END or inp["applicable_period"] != _prior.TERMINAL_DISCOVERY_ROOT or (inp["metadata_identity_verified"], inp["payload_binding_verified"], inp["raw_provenance_verified"]) != (True, True, True): raise ValueError("prior")


def _read_bound(root: object) -> tuple[bytes, str]:
    path = Path(root); raw_dir = path / "raw"
    entries = list(raw_dir.iterdir()); bins = [item for item in entries if re.fullmatch(r"[0-9a-f]{64}\.bin", item.name)]; metas = [item for item in entries if re.fullmatch(r"[0-9a-f]{64}\.json", item.name)]
    if not path.is_absolute() or len(entries) != 2 or len(bins) != 1 or len(metas) != 1 or bins[0].stem != metas[0].stem: raise ValueError("pair")
    metadata = json.loads(metas[0].read_text(encoding="utf-8"))
    if not _prior._metadata_matches(metadata): raise ValueError("metadata")
    _prior.validate_jpx_url(metadata["resolved_url"])
    raw = bins[0].read_bytes()
    if len(raw) != EXPECTED_LENGTH or sha256(raw).hexdigest() != EXPECTED_PAYLOAD_SHA256: raise ValueError("payload")
    return raw, metadata["resolved_url"]


def run_locator(output_root: object) -> dict[str, object]:
    try:
        _assert_prior(_prior.run_diagnostic(output_root))
    except Exception:
        return _finalize(_empty("INPUT_BINDING_FAILURE"))
    try:
        raw, base = _read_bound(output_root)
    except Exception:
        return _finalize(_empty("INPUT_BINDING_FAILURE"))
    selection = _run_selection_core(raw, base)
    if selection.result != "SUCCESSOR_LOCATOR_MATCHED":
        return _finalize(_empty(selection.result, EXPECTED_PAYLOAD_SHA256))
    selected = selection.selected
    assert selected is not None
    return _finalize({"schema_version": SCHEMA_VERSION, "task": TASK, "input_payload_sha256": EXPECTED_PAYLOAD_SHA256, "result": selection.result, "mechanical_candidate_count": selection.mechanical_candidate_count, "qualifying_candidate_count": selection.qualifying_candidate_count, "selected_raw_href_sha256": selected["raw_href_sha256"], "selected_resolved_url_sha256": selected["resolved_url_sha256"], "network_requests": 0, "replacement_locator_authorized": False})


def _fresh_finalize(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["structural_evidence_sha256"] = sha256(canonical_json(result).encode("utf-8")).hexdigest()
    validate_fresh_safe_result(result)
    return result


def _fresh_empty(result: str, digest: str, length: int) -> dict[str, object]:
    return {"schema_version": FRESH_SCHEMA_VERSION, "task": TASK, "input_payload_sha256": digest, "input_payload_byte_length": length, "result": result, "mechanical_candidate_count": 0, "qualifying_candidate_count": 0, "selected_raw_href_sha256": None, "selected_resolved_url_sha256": None, "network_requests": 0, "replacement_locator_authorized": False}


def validate_fresh_safe_result(value: object) -> None:
    keys = {"schema_version", "task", "input_payload_sha256", "input_payload_byte_length", "result", "mechanical_candidate_count", "qualifying_candidate_count", "selected_raw_href_sha256", "selected_resolved_url_sha256", "network_requests", "replacement_locator_authorized", "structural_evidence_sha256"}
    if type(value) is not dict or set(value) != keys or value["schema_version"] != FRESH_SCHEMA_VERSION or value["task"] != TASK or value["result"] not in RESULTS or type(value["network_requests"]) is not int or value["network_requests"] != 0 or value["replacement_locator_authorized"] is not False: raise ValueError("schema")
    for key in ("input_payload_byte_length", "mechanical_candidate_count", "qualifying_candidate_count"):
        if type(value[key]) is not int or value[key] < 0: raise ValueError("count")
    if type(value["input_payload_sha256"]) is not str or _HEX.fullmatch(value["input_payload_sha256"]) is None: raise ValueError("payload")
    digestless = dict(value); digest = digestless.pop("structural_evidence_sha256")
    if type(digest) is not str or _HEX.fullmatch(digest) is None or sha256(canonical_json(digestless).encode()).hexdigest() != digest: raise ValueError("hash")
    selected = (value["selected_raw_href_sha256"], value["selected_resolved_url_sha256"])
    if any(item is not None and (type(item) is not str or _HEX.fullmatch(item) is None) for item in selected): raise ValueError("selected")
    if value["result"] == "SUCCESSOR_LOCATOR_MATCHED":
        if value["mechanical_candidate_count"] < 1 or value["qualifying_candidate_count"] != 1 or any(item is None for item in selected): raise ValueError("success")
    elif value["mechanical_candidate_count"] != 0 or value["qualifying_candidate_count"] != 0 or any(item is not None for item in selected): raise ValueError("failure")


def run_fresh_root_locator(raw: bytes, resolved_root_url: str, payload_sha256: str, payload_byte_length: int) -> tuple[dict[str, object], str | None]:
    """Run the shared selector on already locked, provenance-verified fresh bytes.

    The second tuple member is private runtime state for a future executor and
    must never be serialized.
    """
    # This valid-domain selector accepts only the exact immutable bytes object
    # whose verified identity is recorded below.
    if type(raw) is not bytes:
        raise _FreshInputContractViolation("fresh selector raw must be exact built-in bytes")
    digest = sha256(raw).hexdigest()
    length = len(raw)
    if type(payload_sha256) is not str or _HEX.fullmatch(payload_sha256) is None or type(payload_byte_length) is not int or payload_byte_length < 0 or payload_sha256 != digest or payload_byte_length != length:
        return _fresh_finalize(_fresh_empty("INPUT_BINDING_FAILURE", digest, length)), None
    try:
        _prior.validate_jpx_url(resolved_root_url)
    except Exception:
        return _fresh_finalize(_fresh_empty("INPUT_BINDING_FAILURE", digest, length)), None
    selection = _run_selection_core(raw, resolved_root_url)
    if selection.result != "SUCCESSOR_LOCATOR_MATCHED":
        return _fresh_finalize(_fresh_empty(selection.result, digest, length)), None
    selected = selection.selected
    assert selected is not None
    return _fresh_finalize({"schema_version": FRESH_SCHEMA_VERSION, "task": TASK, "input_payload_sha256": digest, "input_payload_byte_length": length, "result": selection.result, "mechanical_candidate_count": selection.mechanical_candidate_count, "qualifying_candidate_count": selection.qualifying_candidate_count, "selected_raw_href_sha256": selected["raw_href_sha256"], "selected_resolved_url_sha256": selected["resolved_url_sha256"], "network_requests": 0, "replacement_locator_authorized": False}), selected["resolved_url"]
