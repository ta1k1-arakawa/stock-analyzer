"""Offline-only F1 locked-root successor diagnostic; it never performs I/O beyond its supplied lock."""
from __future__ import annotations

from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import stat
import urllib.parse
from typing import Any

from src.v9_005_stage_a_jpx_probe import (
    LISTED_ISSUES_PAGE_URL,
    SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
    TERMINAL_DISCOVERY_ROOT,
    V9005StageABlocked,
    validate_jpx_url,
    verify_raw_provenance,
)

SCHEMA_VERSION = "V9_006_F1_LOCKED_ROOT_LOCATOR_SUCCESSOR_DIAGNOSTIC_V1"
TASK = "V9_006_F1_LOCKED_ROOT_LOCATOR_SUCCESSOR_DIAGNOSTIC"
EXPECTED_LENGTH = 30059
EXPECTED_SHA256 = "ab19c37ca50b23798b8c12c5dc7c4abc6ba865e9e9ec73f04a7daf1247c9720f"
RAW_LOCK_ID_SET_SHA256 = "8e0f0798c6da09292c964e56efb6954dd2e57ac2191b1e80c7fc03eb1d9ba621"
RESULTS = frozenset({"EVIDENCE_CAPTURED", "INPUT_BINDING_FAILURE", "HTML_STRUCTURE_UNSUPPORTED", "SAFE_OUTPUT_VALIDATION_FAILURE"})
EXTENSIONS = frozenset({"XLS", "XLSX", "CSV", "ZIP", "PDF", "HTML", "OTHER", "NONE"})
_HEX = re.compile(r"[0-9a-f]{64}")
_UNSAFE_TEXT = re.compile(r"https?://|file:|[A-Za-z]:[\\/]", re.I)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _input(flags: tuple[bool, bool, bool]) -> dict[str, object]:
    return {
        "source_family": SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
        "applicable_period": TERMINAL_DISCOVERY_ROOT,
        "http_status": 200,
        "byte_length": EXPECTED_LENGTH,
        "payload_sha256": EXPECTED_SHA256,
        "terminal_adjudication_raw_lock_id_set_sha256": RAW_LOCK_ID_SET_SHA256,
        "metadata_identity_verified": flags[0],
        "payload_binding_verified": flags[1],
        "raw_provenance_verified": flags[2],
    }


def _empty(result: str, flags: tuple[bool, bool, bool], status: str) -> dict[str, object]:
    return _finalize({
        "schema_version": SCHEMA_VERSION, "task": TASK, "input": _input(flags),
        "diagnostic_result": result, "document_parse_status": status, "title": None,
        "total_anchor_count": 0, "total_heading_count": 0, "headings": [], "anchors": [],
        "candidate_anchor_ordinals": [], "candidate_count": 0,
        "locator_decision": "NOT_MADE", "replacement_locator_authorized": False,
        "network_requests": 0,
    })


def _finalize(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["structural_evidence_sha256"] = sha256(canonical_json(result).encode("utf-8")).hexdigest()
    validate_safe_result(result)
    return result


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()[:160]


def _extension(resolved: str | None) -> str:
    if resolved is None:
        return "OTHER"
    path = urllib.parse.unquote(urllib.parse.urlsplit(resolved).path).lower()
    for suffix, kind in ((".xlsx", "XLSX"), (".xls", "XLS"), (".csv", "CSV"), (".zip", "ZIP"), (".pdf", "PDF"), (".html", "HTML"), (".htm", "HTML")):
        if path.endswith(suffix):
            return kind
    return "OTHER"


class _Unsupported(Exception):
    pass


class _Parser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.anchors: list[dict[str, Any]] = []
        self.headings: list[dict[str, Any]] = []
        self.active_anchor: dict[str, Any] | None = None
        self.active_heading: dict[str, Any] | None = None
        self.active_title: list[str] | None = None
        self.title_seen = False
        self.last_heading_ordinal: int | None = None

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a" or tag.lower() == "title" or re.fullmatch(r"h[1-6]", tag.lower()):
            raise _Unsupported()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "a":
            if self.active_anchor is not None:
                raise _Unsupported()
            hrefs = [value for name, value in attrs if name.lower() == "href"]
            if len(hrefs) > 1:
                raise _Unsupported()
            self.active_anchor = {"anchor_ordinal": len(self.anchors) + 1, "normalized_visible_text": "", "nearest_preceding_heading_ordinal": self.last_heading_ordinal, "href": hrefs[0] if hrefs else _MISSING, "parts": []}
        elif re.fullmatch(r"h[1-6]", tag):
            if self.active_heading is not None:
                raise _Unsupported()
            ordinal = len(self.headings) + 1
            self.active_heading = {"ordinal": ordinal, "level": int(tag[1]), "normalized_text": "", "parts": []}
            self.last_heading_ordinal = ordinal
        elif tag == "title":
            if self.active_title is not None or self.title_seen:
                raise _Unsupported()
            self.active_title = []
            self.title_seen = True

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "a":
            if self.active_anchor is None:
                raise _Unsupported()
            anchor = self.active_anchor; self.active_anchor = None
            anchor["normalized_visible_text"] = _text("".join(anchor.pop("parts")))
            self.anchors.append(anchor)
        elif re.fullmatch(r"h[1-6]", tag):
            if self.active_heading is None or self.active_heading["level"] != int(tag[1]):
                raise _Unsupported()
            heading = self.active_heading; self.active_heading = None
            heading["normalized_text"] = _text("".join(heading.pop("parts")))
            self.headings.append(heading)
        elif tag == "title":
            if self.active_title is None:
                raise _Unsupported()
            self.title = _text("".join(self.active_title)); self.active_title = None

    def handle_data(self, data: str) -> None:
        if self.active_anchor is not None: self.active_anchor["parts"].append(data)
        if self.active_heading is not None: self.active_heading["parts"].append(data)
        if self.active_title is not None: self.active_title.append(data)

    def close_checked(self) -> None:
        self.close()
        if self.active_anchor is not None or self.active_heading is not None or self.active_title is not None:
            raise _Unsupported()


_MISSING = object()


def _anchor_safe(anchor: dict[str, Any], base_url: str) -> dict[str, object]:
    href = anchor.pop("href")
    visible = anchor["normalized_visible_text"]
    if href is _MISSING:
        return {**anchor, "href_present": False, "same_jpx_domain_after_resolution": "unknown", "target_extension_class": "NONE", "raw_href_sha256": None, "resolved_url_sha256": None}
    if href is None:
        return {**anchor, "href_present": True, "same_jpx_domain_after_resolution": "unknown", "target_extension_class": "OTHER", "raw_href_sha256": None, "resolved_url_sha256": None}
    raw_hash = sha256(href.encode("utf-8")).hexdigest()
    try:
        resolved = urllib.parse.urljoin(base_url, href)
        resolved_hash = sha256(resolved.encode("utf-8")).hexdigest()
        try:
            validate_jpx_url(resolved)
            same: bool | str = True
        except V9005StageABlocked:
            same = False
        return {**anchor, "href_present": True, "same_jpx_domain_after_resolution": same, "target_extension_class": _extension(resolved), "raw_href_sha256": raw_hash, "resolved_url_sha256": resolved_hash}
    except Exception:
        return {**anchor, "href_present": True, "same_jpx_domain_after_resolution": "unknown", "target_extension_class": "OTHER", "raw_href_sha256": raw_hash, "resolved_url_sha256": None}


def parse_locked_html(raw: bytes, resolved_url: str) -> dict[str, object]:
    """Pure parser for already-bound bytes; callers classify unsupported output."""
    parser = _Parser(resolved_url)
    parser.title = None
    try:
        parser.feed(raw.decode("utf-8", errors="replace")); parser.close_checked()
    except _Unsupported:
        raise
    except Exception as exc:
        raise _Unsupported() from exc
    anchors = [_anchor_safe(dict(item), resolved_url) for item in parser.anchors]
    headings = [{key: item[key] for key in ("ordinal", "level", "normalized_text")} for item in parser.headings]
    texts = [parser.title, *(item["normalized_text"] for item in headings), *(item["normalized_visible_text"] for item in anchors)]
    if any(isinstance(item, str) and _UNSAFE_TEXT.search(item) for item in texts):
        raise ValueError("unsafe safe-text")
    candidates = [item["anchor_ordinal"] for item in anchors if item["href_present"] is True and item["same_jpx_domain_after_resolution"] is True and item["target_extension_class"] in {"XLS", "XLSX", "CSV", "ZIP"}]
    return {"title": parser.title, "total_anchor_count": len(anchors), "total_heading_count": len(headings), "headings": headings, "anchors": anchors, "candidate_anchor_ordinals": candidates, "candidate_count": len(candidates)}


def _metadata_matches(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {"schema_version", "source_family", "applicable_period", "requested_url", "resolved_url", "http_status", "retrieval_timestamp_utc", "byte_length", "sha256"}:
        return False
    try: validate_jpx_url(value["resolved_url"])
    except Exception: return False
    return value["schema_version"] == "V9_005_STAGE_A_RAW_LOCK_V1" and value["source_family"] == SOURCE_FAMILY_LISTED_ISSUES_MONTH_END and value["applicable_period"] == TERMINAL_DISCOVERY_ROOT and value["requested_url"] == LISTED_ISSUES_PAGE_URL and value["http_status"] == 200 and value["byte_length"] == EXPECTED_LENGTH and value["sha256"] == EXPECTED_SHA256


def run_diagnostic(output_root: object) -> dict[str, object]:
    """Read exactly one existing raw-lock pair; never writes or performs network I/O."""
    try:
        root = Path(output_root)
        if not root.is_absolute() or not root.is_dir(): return _empty("INPUT_BINDING_FAILURE", (False, False, False), "NOT_PARSED")
        raw_dir = root / "raw"
        if not raw_dir.is_dir(): return _empty("INPUT_BINDING_FAILURE", (False, False, False), "NOT_PARSED")
        entries = list(raw_dir.iterdir())
        names = {entry.name for entry in entries}
        bins = [entry for entry in entries if re.fullmatch(r"[0-9a-f]{64}\.bin", entry.name)]
        metas = [entry for entry in entries if re.fullmatch(r"[0-9a-f]{64}\.json", entry.name)]
        if len(entries) != 2 or len(bins) != 1 or len(metas) != 1 or bins[0].stem != metas[0].stem or names != {bins[0].name, metas[0].name}:
            return _empty("INPUT_BINDING_FAILURE", (False, False, False), "NOT_PARSED")
        try: metadata = json.loads(metas[0].read_text(encoding="utf-8"))
        except Exception: return _empty("INPUT_BINDING_FAILURE", (False, False, False), "NOT_PARSED")
        if not _metadata_matches(metadata): return _empty("INPUT_BINDING_FAILURE", (False, False, False), "NOT_PARSED")
        try: raw = bins[0].read_bytes()
        except Exception: return _empty("INPUT_BINDING_FAILURE", (True, False, False), "NOT_PARSED")
        if len(raw) != EXPECTED_LENGTH or sha256(raw).hexdigest() != EXPECTED_SHA256:
            return _empty("INPUT_BINDING_FAILURE", (True, False, False), "NOT_PARSED")
        try: provenance = verify_raw_provenance(root) is True
        except Exception: provenance = False
        if not provenance: return _empty("INPUT_BINDING_FAILURE", (True, True, False), "NOT_PARSED")
        try: parsed = parse_locked_html(raw, metadata["resolved_url"])
        except _Unsupported: return _empty("HTML_STRUCTURE_UNSUPPORTED", (True, True, True), "UNSUPPORTED")
        except ValueError: return _empty("SAFE_OUTPUT_VALIDATION_FAILURE", (True, True, True), "PARSED")
        return _finalize({"schema_version": SCHEMA_VERSION, "task": TASK, "input": _input((True, True, True)), "diagnostic_result": "EVIDENCE_CAPTURED", "document_parse_status": "PARSED", **parsed, "locator_decision": "NOT_MADE", "replacement_locator_authorized": False, "network_requests": 0})
    except Exception:
        raise


def validate_safe_result(value: object) -> None:
    required = {"schema_version", "task", "input", "diagnostic_result", "document_parse_status", "title", "total_anchor_count", "total_heading_count", "headings", "anchors", "candidate_anchor_ordinals", "candidate_count", "locator_decision", "replacement_locator_authorized", "network_requests", "structural_evidence_sha256"}
    if not isinstance(value, dict) or set(value) != required or value["schema_version"] != SCHEMA_VERSION or value["task"] != TASK or value["diagnostic_result"] not in RESULTS or value["locator_decision"] != "NOT_MADE" or value["replacement_locator_authorized"] is not False or value["network_requests"] != 0 or type(value["network_requests"]) is not int:
        raise ValueError("invalid safe result")
    data = dict(value); digest = data.pop("structural_evidence_sha256")
    if not isinstance(digest, str) or _HEX.fullmatch(digest) is None or sha256(canonical_json(data).encode("utf-8")).hexdigest() != digest: raise ValueError("invalid hash")
    inp = value["input"]
    if not isinstance(inp, dict) or set(inp) != {"source_family", "applicable_period", "http_status", "byte_length", "payload_sha256", "terminal_adjudication_raw_lock_id_set_sha256", "metadata_identity_verified", "payload_binding_verified", "raw_provenance_verified"}: raise ValueError("invalid input")
    if inp["source_family"] != SOURCE_FAMILY_LISTED_ISSUES_MONTH_END or inp["applicable_period"] != TERMINAL_DISCOVERY_ROOT or inp["http_status"] != 200 or type(inp["http_status"]) is not int or inp["byte_length"] != EXPECTED_LENGTH or type(inp["byte_length"]) is not int or inp["payload_sha256"] != EXPECTED_SHA256 or inp["terminal_adjudication_raw_lock_id_set_sha256"] != RAW_LOCK_ID_SET_SHA256 or any(type(inp[key]) is not bool for key in ("metadata_identity_verified", "payload_binding_verified", "raw_provenance_verified")): raise ValueError("invalid input")
    result, status = value["diagnostic_result"], value["document_parse_status"]
    flags = (inp["metadata_identity_verified"], inp["payload_binding_verified"], inp["raw_provenance_verified"])
    if result == "INPUT_BINDING_FAILURE": expected = "NOT_PARSED"
    elif result == "HTML_STRUCTURE_UNSUPPORTED": expected = "UNSUPPORTED"
    else: expected = "PARSED"
    if status != expected: raise ValueError("invalid status")
    empty = value["title"] is None and value["total_anchor_count"] == 0 and value["total_heading_count"] == 0 and value["headings"] == [] and value["anchors"] == [] and value["candidate_anchor_ordinals"] == [] and value["candidate_count"] == 0
    if result != "EVIDENCE_CAPTURED":
        if not empty or (result != "INPUT_BINDING_FAILURE" and flags != (True, True, True)): raise ValueError("invalid failure")
        if result == "INPUT_BINDING_FAILURE" and flags not in {(False, False, False), (True, False, False), (True, True, False)}: raise ValueError("invalid sequential flags")
        return
    if flags != (True, True, True) or not isinstance(value["title"], (str, type(None))) or (isinstance(value["title"], str) and (len(value["title"]) > 160 or _UNSAFE_TEXT.search(value["title"]))) or any(type(value[key]) is not int or value[key] < 0 for key in ("total_anchor_count", "total_heading_count", "candidate_count")): raise ValueError("invalid captured")
    headings, anchors, candidates = value["headings"], value["anchors"], value["candidate_anchor_ordinals"]
    if not isinstance(headings, list) or not isinstance(anchors, list) or not isinstance(candidates, list) or len(headings) != value["total_heading_count"] or len(anchors) != value["total_anchor_count"] or len(candidates) != value["candidate_count"]: raise ValueError("invalid counts")
    if any(not isinstance(item, dict) or type(item.get("ordinal")) is not int for item in headings) or [item["ordinal"] for item in headings] != list(range(1, len(headings) + 1)): raise ValueError("heading order")
    for item in headings:
        if set(item) != {"ordinal", "level", "normalized_text"} or type(item["level"]) is not int or not 1 <= item["level"] <= 6 or not isinstance(item["normalized_text"], str) or len(item["normalized_text"]) > 160 or _UNSAFE_TEXT.search(item["normalized_text"]): raise ValueError("heading")
    if any(not isinstance(item, dict) or type(item.get("anchor_ordinal")) is not int for item in anchors) or [item["anchor_ordinal"] for item in anchors] != list(range(1, len(anchors) + 1)): raise ValueError("anchor order")
    expected_candidates: list[int] = []
    for item in anchors:
        if not isinstance(item, dict) or set(item) != {"anchor_ordinal", "normalized_visible_text", "nearest_preceding_heading_ordinal", "href_present", "same_jpx_domain_after_resolution", "target_extension_class", "raw_href_sha256", "resolved_url_sha256"}: raise ValueError("anchor")
        if not isinstance(item["normalized_visible_text"], str) or len(item["normalized_visible_text"]) > 160 or _UNSAFE_TEXT.search(item["normalized_visible_text"]) or type(item["href_present"]) is not bool or item["same_jpx_domain_after_resolution"] not in (True, False, "unknown") or item["target_extension_class"] not in EXTENSIONS: raise ValueError("anchor")
        nearest = item["nearest_preceding_heading_ordinal"]
        if nearest is not None and (type(nearest) is not int or not 1 <= nearest <= len(headings)): raise ValueError("nearest")
        for key in ("raw_href_sha256", "resolved_url_sha256"):
            if item[key] is not None and (not isinstance(item[key], str) or _HEX.fullmatch(item[key]) is None): raise ValueError("url hash")
        if not item["href_present"] and (item["same_jpx_domain_after_resolution"] != "unknown" or item["target_extension_class"] != "NONE" or item["raw_href_sha256"] is not None or item["resolved_url_sha256"] is not None): raise ValueError("no href")
        if item["href_present"]:
            if item["raw_href_sha256"] is None:
                if item["same_jpx_domain_after_resolution"] != "unknown" or item["target_extension_class"] != "OTHER" or item["resolved_url_sha256"] is not None: raise ValueError("none href")
            elif item["target_extension_class"] == "NONE": raise ValueError("string href")
            elif item["same_jpx_domain_after_resolution"] == "unknown":
                if item["resolved_url_sha256"] is not None or item["target_extension_class"] != "OTHER": raise ValueError("unknown href")
            elif item["resolved_url_sha256"] is None: raise ValueError("resolved href")
        if item["href_present"] and item["same_jpx_domain_after_resolution"] is True and item["target_extension_class"] in {"XLS", "XLSX", "CSV", "ZIP"}: expected_candidates.append(item["anchor_ordinal"])
    if any(type(item) is not int for item in candidates) or candidates != expected_candidates: raise ValueError("candidates")
