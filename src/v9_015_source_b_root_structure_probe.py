"""Pure offline structural calibration probe for the V9_015 SOURCE_B root.

The probe accepts only caller-supplied bytes.  It deliberately reports
structural counts and closed decisions, never document text or candidate
attribute values.  It does not read files, access the environment, use the
network, resolve URLs, or invoke any downstream locator or research logic.
"""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from typing import Any

__all__ = ["probe_root_structure"]

_REQUIRED_YEARS = ("2017", "2019", "2020", "2022", "2026")
_TAG_KEYS = (
    "html",
    "head",
    "body",
    "table",
    "thead",
    "tbody",
    "tr",
    "th",
    "td",
    "a",
    "option",
)
_CANDIDATE_TAGS = frozenset(("a", "option"))
_MULTIPLICITIES = frozenset(("ZERO", "ONE", "MANY"))
_FAILURE_CLASSES = frozenset(("IMPLEMENTATION_FAILURE", "DATA_QUALITY_FAILURE"))
_RESULT_KEYS = (
    "schema_version",
    "root_sha256",
    "root_byte_count",
    "html_parser_success",
    "structure_failure_class",
    "tag_counts",
    "anchor_count",
    "option_count",
    "visible_text_node_count",
    "required_year_anchor_token_counts",
    "required_year_anchor_nonempty_href_counts",
    "required_year_anchor_multiplicity",
    "required_year_option_token_counts",
    "required_year_option_nonempty_value_counts",
    "required_year_option_multiplicity",
    "required_year_visible_token_counts",
    "anchor_category_complete_unique",
    "option_category_complete_unique",
    "other_candidate_category_all_zero",
    "deterministic_candidate_category",
    "all_required_years_deterministically_bindable",
    "safe_calibration_status",
)


class _MalformedCandidateStructure(Exception):
    """Internal marker; its text is never exposed."""


class _SchemaInvariantViolation(Exception):
    """Internal marker; its text is never exposed."""


class _Candidate:
    __slots__ = ("tag", "attribute_value", "text_parts")

    def __init__(self, tag: str, attribute_value: Any) -> None:
        self.tag = tag
        self.attribute_value = attribute_value
        self.text_parts: list[str] = []


class _StructuralParser(HTMLParser):
    """The sole parser implementation permitted by the frozen contract."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tag_counts = {tag: 0 for tag in _TAG_KEYS}
        self.anchor_records: list[tuple[str, bool]] = []
        self.option_records: list[tuple[str, bool]] = []
        self.visible_text_node_count = 0
        self.visible_year_counts = {year: 0 for year in _REQUIRED_YEARS}
        self._active_candidate: _Candidate | None = None
        self._script_style_depth = 0

    @staticmethod
    def _attribute_value(attrs: list[tuple[str, str | None]], name: str) -> Any:
        values = [value for attr_name, value in attrs if attr_name == name]
        if len(values) > 1:
            raise _MalformedCandidateStructure()
        return values[0] if values else None

    def _count_tag(self, tag: str) -> None:
        if tag in self.tag_counts:
            self.tag_counts[tag] += 1

    def _start_candidate(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._active_candidate is not None:
            raise _MalformedCandidateStructure()
        attribute_name = "href" if tag == "a" else "value"
        attribute_value = self._attribute_value(attrs, attribute_name)
        self._active_candidate = _Candidate(tag, attribute_value)

    def _finish_candidate(self, tag: str) -> None:
        candidate = self._active_candidate
        if candidate is None or candidate.tag != tag:
            raise _MalformedCandidateStructure()
        self._active_candidate = None
        label = " ".join("".join(candidate.text_parts).split())
        eligible = isinstance(candidate.attribute_value, str) and len(candidate.attribute_value) > 0
        record = (label, eligible)
        if tag == "a":
            self.anchor_records.append(record)
        else:
            self.option_records.append(record)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._count_tag(tag)
        if tag in _CANDIDATE_TAGS:
            self._start_candidate(tag, attrs)
        if tag in ("script", "style"):
            self._script_style_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._count_tag(tag)
        if tag in _CANDIDATE_TAGS:
            self._start_candidate(tag, attrs)
            self._finish_candidate(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in _CANDIDATE_TAGS:
            self._finish_candidate(tag)
        if tag in ("script", "style") and self._script_style_depth > 0:
            self._script_style_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._active_candidate is not None and self._script_style_depth == 0:
            self._active_candidate.text_parts.append(data)
        if self._script_style_depth == 0:
            normalized = " ".join(data.split())
            if normalized:
                self.visible_text_node_count += 1
                if normalized in self.visible_year_counts:
                    self.visible_year_counts[normalized] += 1

    def close(self) -> None:
        super().close()
        if self._active_candidate is not None:
            raise _MalformedCandidateStructure()


def _zero_map() -> dict[str, int]:
    return {year: 0 for year in _REQUIRED_YEARS}


def _multiplicity(count: int) -> str:
    if count == 0:
        return "ZERO"
    if count == 1:
        return "ONE"
    return "MANY"


def _category_maps(records: list[tuple[str, bool]]) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    token_counts = _zero_map()
    nonempty_counts = _zero_map()
    for label, eligible in records:
        if label in token_counts:
            token_counts[label] += 1
            if eligible:
                nonempty_counts[label] += 1
    multiplicity = {
        year: _multiplicity(nonempty_counts[year]) for year in _REQUIRED_YEARS
    }
    return token_counts, nonempty_counts, multiplicity


def _safe_failure(
    root_sha256: str | None,
    root_byte_count: int,
    failure_class: str,
) -> dict[str, Any]:
    zero_counts = _zero_map()
    zero_multiplicity = {year: "ZERO" for year in _REQUIRED_YEARS}
    result: dict[str, Any] = {
        "schema_version": "V9_015_ROOT_STRUCTURE_CALIBRATION_V1",
        "root_sha256": root_sha256,
        "root_byte_count": root_byte_count,
        "html_parser_success": False,
        "structure_failure_class": failure_class,
        "tag_counts": {tag: 0 for tag in _TAG_KEYS},
        "anchor_count": 0,
        "option_count": 0,
        "visible_text_node_count": 0,
        "required_year_anchor_token_counts": dict(zero_counts),
        "required_year_anchor_nonempty_href_counts": dict(zero_counts),
        "required_year_anchor_multiplicity": dict(zero_multiplicity),
        "required_year_option_token_counts": dict(zero_counts),
        "required_year_option_nonempty_value_counts": dict(zero_counts),
        "required_year_option_multiplicity": dict(zero_multiplicity),
        "required_year_visible_token_counts": dict(zero_counts),
        "anchor_category_complete_unique": False,
        "option_category_complete_unique": False,
        "other_candidate_category_all_zero": False,
        "deterministic_candidate_category": None,
        "all_required_years_deterministically_bindable": False,
        "safe_calibration_status": "FAIL_TERMINAL",
    }
    _validate_safe_result(result)
    return result


def _safe_success(root_sha256: str, root_byte_count: int, parser: _StructuralParser) -> dict[str, Any]:
    anchor_tokens, anchor_nonempty, anchor_multiplicity = _category_maps(parser.anchor_records)
    option_tokens, option_nonempty, option_multiplicity = _category_maps(parser.option_records)
    anchor_complete_unique = all(anchor_multiplicity[year] == "ONE" for year in _REQUIRED_YEARS)
    option_complete_unique = all(option_multiplicity[year] == "ONE" for year in _REQUIRED_YEARS)
    anchor_all_zero = all(anchor_multiplicity[year] == "ZERO" for year in _REQUIRED_YEARS)
    option_all_zero = all(option_multiplicity[year] == "ZERO" for year in _REQUIRED_YEARS)
    if anchor_complete_unique and option_all_zero:
        category = "ANCHOR_HREF"
    elif option_complete_unique and anchor_all_zero:
        category = "OPTION_VALUE"
    else:
        category = None
    result: dict[str, Any] = {
        "schema_version": "V9_015_ROOT_STRUCTURE_CALIBRATION_V1",
        "root_sha256": root_sha256,
        "root_byte_count": root_byte_count,
        "html_parser_success": True,
        "structure_failure_class": None,
        "tag_counts": dict(parser.tag_counts),
        "anchor_count": len(parser.anchor_records),
        "option_count": len(parser.option_records),
        "visible_text_node_count": parser.visible_text_node_count,
        "required_year_anchor_token_counts": anchor_tokens,
        "required_year_anchor_nonempty_href_counts": anchor_nonempty,
        "required_year_anchor_multiplicity": anchor_multiplicity,
        "required_year_option_token_counts": option_tokens,
        "required_year_option_nonempty_value_counts": option_nonempty,
        "required_year_option_multiplicity": option_multiplicity,
        "required_year_visible_token_counts": dict(parser.visible_year_counts),
        "anchor_category_complete_unique": anchor_complete_unique,
        "option_category_complete_unique": option_complete_unique,
        "other_candidate_category_all_zero": (
            category == "ANCHOR_HREF" and option_all_zero
        ) or (category == "OPTION_VALUE" and anchor_all_zero),
        "deterministic_candidate_category": category,
        "all_required_years_deterministically_bindable": category is not None,
        "safe_calibration_status": "PASS",
    }
    _validate_safe_result(result)
    return result


def _validate_safe_result(result: dict[str, Any]) -> None:
    if tuple(result.keys()) != _RESULT_KEYS:
        raise _SchemaInvariantViolation()
    root_sha256 = result["root_sha256"]
    if root_sha256 is not None and (
        not isinstance(root_sha256, str)
        or len(root_sha256) != 64
        or any(character not in "0123456789abcdef" for character in root_sha256)
    ):
        raise _SchemaInvariantViolation()
    root_byte_count = result["root_byte_count"]
    if isinstance(root_byte_count, bool) or not isinstance(root_byte_count, int) or root_byte_count < 0:
        raise _SchemaInvariantViolation()
    if result["structure_failure_class"] not in (None, *_FAILURE_CLASSES):
        raise _SchemaInvariantViolation()
    for key in (
        "html_parser_success",
        "anchor_category_complete_unique",
        "option_category_complete_unique",
        "other_candidate_category_all_zero",
        "all_required_years_deterministically_bindable",
    ):
        if not isinstance(result[key], bool):
            raise _SchemaInvariantViolation()
    if result["deterministic_candidate_category"] not in (None, "ANCHOR_HREF", "OPTION_VALUE"):
        raise _SchemaInvariantViolation()
    if result["safe_calibration_status"] not in ("PASS", "FAIL_TERMINAL"):
        raise _SchemaInvariantViolation()
    tag_counts = result["tag_counts"]
    if tuple(tag_counts.keys()) != _TAG_KEYS:
        raise _SchemaInvariantViolation()
    for count in tag_counts.values():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise _SchemaInvariantViolation()
    for key in (
        "anchor_count",
        "option_count",
        "visible_text_node_count",
    ):
        count = result[key]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise _SchemaInvariantViolation()
    map_keys = {
        "required_year_anchor_token_counts",
        "required_year_anchor_nonempty_href_counts",
        "required_year_option_token_counts",
        "required_year_option_nonempty_value_counts",
        "required_year_visible_token_counts",
    }
    for key in map_keys:
        values = result[key]
        if tuple(values.keys()) != _REQUIRED_YEARS:
            raise _SchemaInvariantViolation()
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
            raise _SchemaInvariantViolation()
    for key in ("required_year_anchor_multiplicity", "required_year_option_multiplicity"):
        values = result[key]
        if tuple(values.keys()) != _REQUIRED_YEARS or any(value not in _MULTIPLICITIES for value in values.values()):
            raise _SchemaInvariantViolation()


def probe_root_structure(root_bytes: bytes) -> dict[str, Any]:
    """Return only safe structural evidence for caller-supplied document bytes."""

    if not isinstance(root_bytes, bytes):
        return _safe_failure(None, 0, "IMPLEMENTATION_FAILURE")
    root_sha256 = hashlib.sha256(root_bytes).hexdigest()
    root_byte_count = len(root_bytes)
    try:
        document = root_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return _safe_failure(root_sha256, root_byte_count, "DATA_QUALITY_FAILURE")
    try:
        parser = _StructuralParser()
        parser.feed(document)
        parser.close()
        return _safe_success(root_sha256, root_byte_count, parser)
    except _MalformedCandidateStructure:
        return _safe_failure(root_sha256, root_byte_count, "DATA_QUALITY_FAILURE")
    except _SchemaInvariantViolation:
        return _safe_failure(root_sha256, root_byte_count, "IMPLEMENTATION_FAILURE")
    except Exception:
        return _safe_failure(root_sha256, root_byte_count, "IMPLEMENTATION_FAILURE")
