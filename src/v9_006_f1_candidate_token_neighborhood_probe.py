"""Offline-only F1 candidate token-neighborhood probe; it never performs I/O beyond the same already-locked payload the reviewed prior diagnostic binds to."""
from __future__ import annotations

from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from typing import Any

from src import v9_006_f1_locked_root_locator_successor_diagnostic as _prior

SCHEMA_VERSION = "V9_006_F1_CANDIDATE_TOKEN_NEIGHBORHOOD_PROBE_V1"
TASK = "V9_006_F1_CANDIDATE_TOKEN_NEIGHBORHOOD_PROBE"

PAYLOAD_HASH = "ab19c37ca50b23798b8c12c5dc7c4abc6ba865e9e9ec73f04a7daf1247c9720f"
EXPECTED_LENGTH = 30059
PRIOR_STRUCTURAL_HASH = "986029641d10d36d33219d729f2c7bdb7c5495447e91be59e11650dd807efad5"
CANDIDATE_ORDINALS = (52, 55)

_SUMMARY_KEYS = ("anchor_ordinal", "normalized_visible_text", "nearest_preceding_heading_ordinal", "same_jpx_domain_after_resolution", "target_extension_class", "raw_href_sha256", "resolved_url_sha256")

_CANDIDATE_BINDINGS: dict[int, dict[str, object]] = {
    52: {"anchor_ordinal": 52, "normalized_visible_text": "", "nearest_preceding_heading_ordinal": 1, "same_jpx_domain_after_resolution": True, "target_extension_class": "XLS", "raw_href_sha256": "ee97b7976663aa4dd55f9f02d33e96ceb66ad76bb43fd2e4523a31fe4d4a6ec9", "resolved_url_sha256": "a7088b6c7e5ea028ffad54bd95e835e32068dfafa324d737e2cef0424f90e613"},
    55: {"anchor_ordinal": 55, "normalized_visible_text": "", "nearest_preceding_heading_ordinal": 1, "same_jpx_domain_after_resolution": True, "target_extension_class": "XLSX", "raw_href_sha256": "759c2f9e683c85ebcd865ea962e7e24a28ce2db4d0c7ff0592e5cdd03dba632b", "resolved_url_sha256": "b8953e84885003e03ee9feafd5408c313c91404582d58f2c489030dfbb4b98b0"},
}

RESULTS = frozenset({"EVIDENCE_CAPTURED", "PRIOR_DIAGNOSTIC_BINDING_FAILURE", "HTML_STRUCTURE_UNSUPPORTED", "SAFE_OUTPUT_VALIDATION_FAILURE"})
_EXTENSIONS = frozenset({"XLS", "XLSX", "CSV", "ZIP", "PDF", "HTML", "OTHER", "NONE"})
_SUPPRESS = ("script", "style", "noscript", "template")
_HEX = re.compile(r"[0-9a-f]{64}")
_UNSAFE_TEXT = re.compile(r"https?://|file:|[A-Za-z]:[\\/]", re.I)

_CONTEXT_KEYS = frozenset({"candidate_anchor_ordinal", "candidate_binding", "preceding_data_tokens", "following_data_tokens", "previous_anchor_summaries", "next_anchor_summaries", "title_present", "normalized_title", "aria_label_present", "normalized_aria_label", "download_present", "download_value_sha256", "total_image_count", "images"})
_IMAGE_KEYS = frozenset({"image_ordinal_within_candidate", "alt_present", "normalized_alt", "title_present", "normalized_title", "src_present", "src_sha256"})
_TOKEN_KEYS = frozenset({"data_token_ordinal", "normalized_text"})
_TOP_KEYS = frozenset({"schema_version", "task", "input_payload_sha256", "prior_diagnostic_structural_evidence_sha256", "prior_diagnostic_binding_verified", "diagnostic_result", "candidate_contexts", "locator_decision", "replacement_locator_authorized", "network_requests", "structural_evidence_sha256"})


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()[:160]


def _summary(anchor: dict[str, Any]) -> dict[str, object]:
    return {key: anchor[key] for key in _SUMMARY_KEYS}


def _type_strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(_type_strict_equal(actual[key], expected[key]) for key in expected)
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_type_strict_equal(item, other) for item, other in zip(actual, expected))
    return actual == expected


class _Unsupported(Exception):
    pass


class _UnsafeText(Exception):
    pass


class _BindingMismatch(Exception):
    pass


def _attr_values(attrs: list[tuple[str, str | None]], name: str) -> list[str | None]:
    return [value for key, value in attrs if key.lower() == name]


def _attr_text_state(attrs: list[tuple[str, str | None]], name: str) -> tuple[bool, str | None]:
    values = _attr_values(attrs, name)
    if len(values) > 1:
        raise _Unsupported()
    if not values:
        return False, None
    value = values[0]
    return (True, None) if value is None else (True, _text(value))


def _attr_hash_state(attrs: list[tuple[str, str | None]], name: str) -> tuple[bool, str | None]:
    values = _attr_values(attrs, name)
    if len(values) > 1:
        raise _Unsupported()
    if not values:
        return False, None
    value = values[0]
    return (True, None) if value is None else (True, sha256(value.encode("utf-8")).hexdigest())


class _Candidate:
    def __init__(self, ordinal: int, attrs: list[tuple[str, str | None]], boundary_index: int) -> None:
        self.ordinal = ordinal
        self.boundary_index = boundary_index
        self.title_present, self.normalized_title = _attr_text_state(attrs, "title")
        self.aria_present, self.normalized_aria = _attr_text_state(attrs, "aria-label")
        self.download_present, self.download_hash = _attr_hash_state(attrs, "download")
        self.total_image_count = 0
        self.images: list[dict[str, object]] = []


class _Parser(HTMLParser):
    def __init__(self, candidate_ordinals: tuple[int, int]) -> None:
        super().__init__(convert_charrefs=True)
        self.candidate_ordinals = candidate_ordinals
        self.depths = {tag: 0 for tag in _SUPPRESS}
        self.anchor_ordinal = 0
        self.anchor_active = False
        self.token_ordinal = 0
        self.external_tokens: list[dict[str, object]] = []
        self.active_candidate: _Candidate | None = None
        self.contexts: dict[int, _Candidate] = {}

    def _suppressed(self) -> bool:
        return any(self.depths[tag] != 0 for tag in _SUPPRESS)

    def _count_image(self, attrs: list[tuple[str, str | None]]) -> None:
        if self.active_candidate is None:
            return
        alt_present, normalized_alt = _attr_text_state(attrs, "alt")
        title_present, normalized_title = _attr_text_state(attrs, "title")
        src_present, src_hash = _attr_hash_state(attrs, "src")
        candidate = self.active_candidate
        candidate.total_image_count += 1
        if len(candidate.images) < 8:
            candidate.images.append({
                "image_ordinal_within_candidate": candidate.total_image_count,
                "alt_present": alt_present, "normalized_alt": normalized_alt,
                "title_present": title_present, "normalized_title": normalized_title,
                "src_present": src_present, "src_sha256": src_hash,
            })

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SUPPRESS:
            self.depths[tag] += 1
            return
        if tag == "a":
            self.anchor_ordinal += 1
            if self.anchor_active:
                raise _Unsupported()
            self.anchor_active = True
            if self.anchor_ordinal in self.candidate_ordinals:
                self.active_candidate = _Candidate(self.anchor_ordinal, attrs, len(self.external_tokens))
            return
        if tag == "img":
            self._count_image(attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "a":
            raise _Unsupported()
        if tag == "img":
            self._count_image(attrs)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SUPPRESS:
            if self.depths[tag] == 0:
                raise _Unsupported()
            self.depths[tag] -= 1
            return
        if tag == "a":
            if not self.anchor_active:
                raise _Unsupported()
            self.anchor_active = False
            if self.active_candidate is not None:
                candidate = self.active_candidate
                self.active_candidate = None
                self.contexts[candidate.ordinal] = candidate

    def handle_data(self, data: str) -> None:
        if self._suppressed():
            return
        normalized = _text(data)
        if not normalized:
            return
        self.token_ordinal += 1
        if self.active_candidate is None:
            self.external_tokens.append({"data_token_ordinal": self.token_ordinal, "normalized_text": normalized})

    def close_checked(self) -> None:
        self.close()
        if self.anchor_active or self._suppressed():
            raise _Unsupported()


def _neighbor_summaries(ordinal: int, prior_by_ordinal: dict[int, dict[str, object]], step: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    current = ordinal + step
    while len(result) < 3 and current in prior_by_ordinal:
        result.append(_summary(prior_by_ordinal[current]))
        current += step
    return result


def _collect_texts(context: dict[str, object]) -> list[str]:
    texts = [item["normalized_text"] for item in context["preceding_data_tokens"]]
    texts += [item["normalized_text"] for item in context["following_data_tokens"]]
    if context["normalized_title"] is not None:
        texts.append(context["normalized_title"])
    if context["normalized_aria_label"] is not None:
        texts.append(context["normalized_aria_label"])
    for image in context["images"]:
        if image["normalized_alt"] is not None:
            texts.append(image["normalized_alt"])
        if image["normalized_title"] is not None:
            texts.append(image["normalized_title"])
    return texts


def _finalize_candidate(candidate: _Candidate, prior_by_ordinal: dict[int, dict[str, object]], external_tokens: list[dict[str, object]]) -> dict[str, object]:
    index = candidate.boundary_index
    preceding = [{"data_token_ordinal": item["data_token_ordinal"], "normalized_text": item["normalized_text"]} for item in reversed(external_tokens[max(0, index - 8):index])]
    following = [{"data_token_ordinal": item["data_token_ordinal"], "normalized_text": item["normalized_text"]} for item in external_tokens[index:index + 8]]
    return {
        "candidate_anchor_ordinal": candidate.ordinal,
        "candidate_binding": _CANDIDATE_BINDINGS[candidate.ordinal],
        "preceding_data_tokens": preceding,
        "following_data_tokens": following,
        "previous_anchor_summaries": _neighbor_summaries(candidate.ordinal, prior_by_ordinal, -1),
        "next_anchor_summaries": _neighbor_summaries(candidate.ordinal, prior_by_ordinal, 1),
        "title_present": candidate.title_present, "normalized_title": candidate.normalized_title,
        "aria_label_present": candidate.aria_present, "normalized_aria_label": candidate.normalized_aria,
        "download_present": candidate.download_present, "download_value_sha256": candidate.download_hash,
        "total_image_count": candidate.total_image_count,
        "images": candidate.images,
    }


def parse_candidate_neighborhoods(raw: bytes, prior_anchors: list[dict[str, object]], candidate_ordinals: tuple[int, int] = CANDIDATE_ORDINALS) -> list[dict[str, object]]:
    """Pure parser for already-bound bytes; callers classify unsupported/unsafe output."""
    parser = _Parser(candidate_ordinals)
    try:
        parser.feed(raw.decode("utf-8", errors="replace"))
        parser.close_checked()
    except _Unsupported:
        raise
    except Exception as exc:
        raise _Unsupported() from exc
    prior_by_ordinal = {item["anchor_ordinal"]: item for item in prior_anchors}
    contexts = []
    for ordinal in candidate_ordinals:
        candidate = parser.contexts.get(ordinal)
        if candidate is None:
            raise _Unsupported()
        contexts.append(_finalize_candidate(candidate, prior_by_ordinal, parser.external_tokens))
    texts: list[str] = []
    for context in contexts:
        texts.extend(_collect_texts(context))
    if any(_UNSAFE_TEXT.search(text) for text in texts):
        raise _UnsafeText()
    return contexts


def _assert_binding(prior_result: dict[str, object]) -> None:
    try:
        if prior_result["diagnostic_result"] != "EVIDENCE_CAPTURED":
            raise _BindingMismatch()
        if prior_result["structural_evidence_sha256"] != PRIOR_STRUCTURAL_HASH:
            raise _BindingMismatch()
        if prior_result["input"]["payload_sha256"] != PAYLOAD_HASH:
            raise _BindingMismatch()
        if prior_result["total_anchor_count"] != 83 or prior_result["total_heading_count"] != 1:
            raise _BindingMismatch()
        if prior_result["candidate_anchor_ordinals"] != list(CANDIDATE_ORDINALS):
            raise _BindingMismatch()
        by_ordinal = {item["anchor_ordinal"]: item for item in prior_result["anchors"]}
        for ordinal in CANDIDATE_ORDINALS:
            anchor = by_ordinal.get(ordinal)
            if anchor is None or _summary(anchor) != _CANDIDATE_BINDINGS[ordinal]:
                raise _BindingMismatch()
    except _BindingMismatch:
        raise
    except Exception as exc:
        raise _BindingMismatch() from exc


def _reread_locked_payload(output_root: object) -> bytes:
    try:
        root = Path(output_root)
        if not root.is_absolute() or not root.is_dir():
            raise _BindingMismatch()
        raw_dir = root / "raw"
        if not raw_dir.is_dir():
            raise _BindingMismatch()
        entries = list(raw_dir.iterdir())
        names = {entry.name for entry in entries}
        bins = [entry for entry in entries if re.fullmatch(r"[0-9a-f]{64}\.bin", entry.name)]
        metas = [entry for entry in entries if re.fullmatch(r"[0-9a-f]{64}\.json", entry.name)]
        if len(entries) != 2 or len(bins) != 1 or len(metas) != 1 or bins[0].stem != metas[0].stem or names != {bins[0].name, metas[0].name}:
            raise _BindingMismatch()
        raw = bins[0].read_bytes()
    except _BindingMismatch:
        raise
    except Exception as exc:
        raise _BindingMismatch() from exc
    if len(raw) != EXPECTED_LENGTH or sha256(raw).hexdigest() != PAYLOAD_HASH:
        raise _BindingMismatch()
    return raw


def _empty(result: str, binding_verified: bool) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION, "task": TASK,
        "input_payload_sha256": PAYLOAD_HASH,
        "prior_diagnostic_structural_evidence_sha256": PRIOR_STRUCTURAL_HASH,
        "prior_diagnostic_binding_verified": binding_verified,
        "diagnostic_result": result,
        "candidate_contexts": [],
        "locator_decision": "NOT_MADE", "replacement_locator_authorized": False, "network_requests": 0,
    }


def _build_expected_neighbors(prior_by_ordinal: dict[int, dict[str, object]]) -> dict[int, dict[str, list[dict[str, object]]]]:
    """Derived exclusively from the already-validated, exactly-bound prior diagnostic anchors; never a new URL-resolution algorithm."""
    return {
        ordinal: {
            "previous": _neighbor_summaries(ordinal, prior_by_ordinal, -1),
            "next": _neighbor_summaries(ordinal, prior_by_ordinal, 1),
        }
        for ordinal in CANDIDATE_ORDINALS
    }


def _finalize(value: dict[str, object], *, expected_neighbors: dict[int, dict[str, list[dict[str, object]]]] | None = None) -> dict[str, object]:
    result = dict(value)
    result["structural_evidence_sha256"] = sha256(canonical_json(result).encode("utf-8")).hexdigest()
    validate_safe_result(result, expected_neighbors=expected_neighbors)
    return result


def run_probe(output_root: object) -> dict[str, object]:
    """Bind to the reviewed prior diagnostic, then read only the same locked payload; never writes, never touches the network."""
    prior_result = _prior.run_diagnostic(output_root)
    try:
        _prior.validate_safe_result(prior_result)
        _assert_binding(prior_result)
        raw = _reread_locked_payload(output_root)
    except (ValueError, _BindingMismatch):
        return _finalize(_empty("PRIOR_DIAGNOSTIC_BINDING_FAILURE", False))
    try:
        contexts = parse_candidate_neighborhoods(raw, prior_result["anchors"], CANDIDATE_ORDINALS)
    except _Unsupported:
        return _finalize(_empty("HTML_STRUCTURE_UNSUPPORTED", True))
    except _UnsafeText:
        return _finalize(_empty("SAFE_OUTPUT_VALIDATION_FAILURE", True))
    prior_by_ordinal = {item["anchor_ordinal"]: item for item in prior_result["anchors"]}
    expected_neighbors = _build_expected_neighbors(prior_by_ordinal)
    return _finalize({
        "schema_version": SCHEMA_VERSION, "task": TASK,
        "input_payload_sha256": PAYLOAD_HASH,
        "prior_diagnostic_structural_evidence_sha256": PRIOR_STRUCTURAL_HASH,
        "prior_diagnostic_binding_verified": True,
        "diagnostic_result": "EVIDENCE_CAPTURED",
        "candidate_contexts": contexts,
        "locator_decision": "NOT_MADE", "replacement_locator_authorized": False, "network_requests": 0,
    }, expected_neighbors=expected_neighbors)


def _valid_text(value: object) -> bool:
    return isinstance(value, str) and len(value) <= 160 and _UNSAFE_TEXT.search(value) is None and _text(value) == value


def _valid_present_text(present: object, normalized: object) -> bool:
    if type(present) is not bool:
        return False
    if not present:
        return normalized is None
    return normalized is None or _valid_text(normalized)


def _valid_present_hash(present: object, digest: object) -> bool:
    if type(present) is not bool:
        return False
    if not present:
        return digest is None
    if digest is None:
        return True
    return isinstance(digest, str) and _HEX.fullmatch(digest) is not None


def _validate_tokens(tokens: object, ascending: bool) -> None:
    if type(tokens) is not list or len(tokens) > 8:
        raise ValueError("tokens")
    previous_ordinal: int | None = None
    for item in tokens:
        if not isinstance(item, dict) or set(item) != _TOKEN_KEYS:
            raise ValueError("token")
        ordinal = item["data_token_ordinal"]
        if type(ordinal) is not int or ordinal <= 0:
            raise ValueError("token ordinal")
        text = item["normalized_text"]
        if not _valid_text(text) or text == "":
            raise ValueError("token text")
        if previous_ordinal is not None:
            if ascending and ordinal <= previous_ordinal:
                raise ValueError("token order")
            if not ascending and ordinal >= previous_ordinal:
                raise ValueError("token order")
        previous_ordinal = ordinal


def _validate_summaries(summaries: object, ordinal: int, step: int, expected: list[dict[str, object]] | None) -> None:
    if type(summaries) is not list or len(summaries) > 3:
        raise ValueError("summaries")
    if expected is not None:
        if len(summaries) != len(expected) or any(not _type_strict_equal(item, other) for item, other in zip(summaries, expected)):
            raise ValueError("summary not exactly bound to prior diagnostic")
    for index, item in enumerate(summaries, start=1):
        if not isinstance(item, dict) or set(item) != set(_SUMMARY_KEYS):
            raise ValueError("summary")
        other = item["anchor_ordinal"]
        if type(other) is not int or other != ordinal + step * index:
            raise ValueError("summary ordinal")
        if not _valid_text(item["normalized_visible_text"]):
            raise ValueError("summary text")
        nearest = item["nearest_preceding_heading_ordinal"]
        if nearest is not None and (type(nearest) is not int or nearest <= 0):
            raise ValueError("summary heading")
        same = item["same_jpx_domain_after_resolution"]
        if not (type(same) is bool or same == "unknown"):
            raise ValueError("summary domain")
        if item["target_extension_class"] not in _EXTENSIONS:
            raise ValueError("summary extension")
        for key in ("raw_href_sha256", "resolved_url_sha256"):
            value = item[key]
            if value is not None and (not isinstance(value, str) or _HEX.fullmatch(value) is None):
                raise ValueError("summary hash")


def _validate_image(image: object, expected_ordinal: int) -> None:
    if not isinstance(image, dict) or set(image) != _IMAGE_KEYS:
        raise ValueError("image")
    if type(image["image_ordinal_within_candidate"]) is not int or image["image_ordinal_within_candidate"] != expected_ordinal:
        raise ValueError("image ordinal")
    if not _valid_present_text(image["alt_present"], image["normalized_alt"]):
        raise ValueError("image alt")
    if not _valid_present_text(image["title_present"], image["normalized_title"]):
        raise ValueError("image title")
    if not _valid_present_hash(image["src_present"], image["src_sha256"]):
        raise ValueError("image src")


def _validate_context(context: object, expected_ordinal: int, neighbor_binding: dict[str, list[dict[str, object]]] | None) -> None:
    if not isinstance(context, dict) or set(context) != _CONTEXT_KEYS:
        raise ValueError("context")
    if type(context["candidate_anchor_ordinal"]) is not int or context["candidate_anchor_ordinal"] != expected_ordinal:
        raise ValueError("context ordinal")
    binding = context["candidate_binding"]
    if not isinstance(binding, dict) or not _type_strict_equal(binding, _CANDIDATE_BINDINGS[expected_ordinal]):
        raise ValueError("context binding")
    _validate_tokens(context["preceding_data_tokens"], ascending=False)
    _validate_tokens(context["following_data_tokens"], ascending=True)
    _validate_summaries(context["previous_anchor_summaries"], expected_ordinal, -1, neighbor_binding["previous"] if neighbor_binding is not None else None)
    _validate_summaries(context["next_anchor_summaries"], expected_ordinal, 1, neighbor_binding["next"] if neighbor_binding is not None else None)
    if not _valid_present_text(context["title_present"], context["normalized_title"]):
        raise ValueError("title")
    if not _valid_present_text(context["aria_label_present"], context["normalized_aria_label"]):
        raise ValueError("aria")
    if not _valid_present_hash(context["download_present"], context["download_value_sha256"]):
        raise ValueError("download")
    count = context["total_image_count"]
    if type(count) is not int or count < 0:
        raise ValueError("image count")
    images = context["images"]
    if type(images) is not list or len(images) != min(count, 8):
        raise ValueError("images length")
    for index, image in enumerate(images, start=1):
        _validate_image(image, index)


def validate_safe_result(value: object, *, expected_neighbors: dict[int, dict[str, list[dict[str, object]]]] | None = None) -> None:
    if not isinstance(value, dict) or set(value) != _TOP_KEYS:
        raise ValueError("invalid safe result")
    if value["schema_version"] != SCHEMA_VERSION or value["task"] != TASK:
        raise ValueError("identity")
    if value["input_payload_sha256"] != PAYLOAD_HASH or value["prior_diagnostic_structural_evidence_sha256"] != PRIOR_STRUCTURAL_HASH:
        raise ValueError("fixed hash")
    if value["diagnostic_result"] not in RESULTS:
        raise ValueError("result enum")
    if value["locator_decision"] != "NOT_MADE" or value["replacement_locator_authorized"] is not False:
        raise ValueError("fixed decision")
    if type(value["network_requests"]) is not int or value["network_requests"] != 0:
        raise ValueError("network requests")
    if type(value["prior_diagnostic_binding_verified"]) is not bool:
        raise ValueError("binding verified type")
    data = dict(value)
    digest = data.pop("structural_evidence_sha256")
    if not isinstance(digest, str) or _HEX.fullmatch(digest) is None or sha256(canonical_json(data).encode("utf-8")).hexdigest() != digest:
        raise ValueError("hash")
    result = value["diagnostic_result"]
    contexts = value["candidate_contexts"]
    if type(contexts) is not list:
        raise ValueError("contexts type")
    if result == "PRIOR_DIAGNOSTIC_BINDING_FAILURE":
        if value["prior_diagnostic_binding_verified"] is not False or contexts != []:
            raise ValueError("binding failure shape")
        return
    if value["prior_diagnostic_binding_verified"] is not True:
        raise ValueError("binding verified")
    if result != "EVIDENCE_CAPTURED":
        if contexts != []:
            raise ValueError("failure contexts")
        return
    if len(contexts) != len(CANDIDATE_ORDINALS):
        raise ValueError("contexts count")
    for ordinal, context in zip(CANDIDATE_ORDINALS, contexts):
        _validate_context(context, ordinal, expected_neighbors.get(ordinal) if expected_neighbors is not None else None)
