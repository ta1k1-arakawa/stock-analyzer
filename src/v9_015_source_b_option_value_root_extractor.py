"""Synthetic-only V9_015 OPTION_VALUE SOURCE_B root extractor.

This module implements only the Stage-D-frozen archive-root year binding.
It accepts caller-supplied HTML bytes, uses the reviewed C1 parser semantics
for candidate text, preserves candidate multiplicity, and returns the
existing ``RootYearCandidate`` representation.  It performs no filesystem
access, network access, root read, child request, or downstream research.
"""

from __future__ import annotations

from html.parser import HTMLParser
import urllib.parse
from typing import Any, Tuple

from src.v9_005_stage_a_jpx_probe import IMPLEMENTATION_FAILURE, V9005StageABlocked
from src.v9_014_jpx_monthly_auction_activity_source_b_locator import (
    SOURCE_B_ARCHIVE_ROOT,
    RootYearCandidate,
    validate_jpx_url as _validate_jpx_url,
)

__all__ = [
    "SOURCE_B_ARCHIVE_ROOT",
    "REQUIRED_YEAR_LABELS",
    "extract_option_value_root_year_candidates",
    "extract_root_year_candidates",
]

REQUIRED_YEAR_LABELS = ("2017", "2019", "2020", "2022", "2026")
_CANDIDATE_TAGS = frozenset(("a", "option"))


class _MalformedCandidateStructure(Exception):
    """Internal marker whose message is never exposed."""


class _Candidate:
    __slots__ = ("tag", "attribute_value", "text_parts")

    def __init__(self, tag: str, attribute_value: Any) -> None:
        self.tag = tag
        self.attribute_value = attribute_value
        self.text_parts: list[str] = []


class _OptionValueParser(HTMLParser):
    """The reviewed C1 structural parser semantics with private values kept."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.option_records: list[tuple[str, Any]] = []
        self.anchor_records: list[tuple[str, Any]] = []
        self._active_candidate: _Candidate | None = None
        self._script_style_depth = 0

    @staticmethod
    def _attribute_value(attrs: list[tuple[str, str | None]], name: str) -> Any:
        values = [value for attr_name, value in attrs if attr_name == name]
        if len(values) > 1:
            raise _MalformedCandidateStructure()
        return values[0] if values else None

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
        normalized_label = " ".join("".join(candidate.text_parts).split())
        record = (normalized_label, candidate.attribute_value)
        if tag == "a":
            self.anchor_records.append(record)
        else:
            self.option_records.append(record)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _CANDIDATE_TAGS:
            self._start_candidate(tag, attrs)
        if tag in ("script", "style"):
            self._script_style_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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

    def close(self) -> None:
        super().close()
        if self._active_candidate is not None:
            raise _MalformedCandidateStructure()


def _implementation_failure() -> V9005StageABlocked:
    return V9005StageABlocked(IMPLEMENTATION_FAILURE)


def _parse_root(root_bytes: bytes) -> _OptionValueParser:
    if not isinstance(root_bytes, bytes):
        raise _implementation_failure()
    try:
        document = root_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _implementation_failure() from exc
    parser = _OptionValueParser()
    try:
        parser.feed(document)
        parser.close()
    except _MalformedCandidateStructure as exc:
        raise _implementation_failure() from exc
    except Exception as exc:
        raise _implementation_failure() from exc
    return parser


def extract_option_value_root_year_candidates(
    root_bytes: bytes, root_url: str = SOURCE_B_ARCHIVE_ROOT
) -> Tuple[RootYearCandidate, ...]:
    """Extract exactly one eligible OPTION_VALUE candidate for every year.

    The root URL must equal the frozen SOURCE_B archive root byte-for-byte.
    Candidate labels use the exact C1 normalization.  Eligible anchors for a
    required year are prohibited, and zero or multiple eligible options fail
    closed.  Raw option values are never inspected for selection; after all
    multiplicity checks pass, each selected raw value is resolved only with
    ``urljoin(root_url, raw_value)`` and the inherited reviewed validator.
    """

    if root_url != SOURCE_B_ARCHIVE_ROOT:
        raise _implementation_failure()
    parser = _parse_root(root_bytes)

    eligible_options: dict[str, list[str]] = {year: [] for year in REQUIRED_YEAR_LABELS}
    for label, raw_value in parser.option_records:
        if label in eligible_options and isinstance(raw_value, str) and raw_value:
            eligible_options[label].append(raw_value)

    eligible_anchors: dict[str, int] = {year: 0 for year in REQUIRED_YEAR_LABELS}
    for label, raw_href in parser.anchor_records:
        if label in eligible_anchors and isinstance(raw_href, str) and raw_href:
            eligible_anchors[label] += 1

    if any(eligible_anchors[year] != 0 for year in REQUIRED_YEAR_LABELS):
        raise _implementation_failure()
    if any(len(eligible_options[year]) != 1 for year in REQUIRED_YEAR_LABELS):
        raise _implementation_failure()

    candidates = []
    for year in REQUIRED_YEAR_LABELS:
        raw_value = eligible_options[year][0]
        resolved_url = urllib.parse.urljoin(root_url, raw_value)
        _validate_jpx_url(resolved_url)
        candidates.append(RootYearCandidate(label=year, href=resolved_url))
    return tuple(candidates)


def extract_root_year_candidates(
    root_bytes: bytes, root_url: str = SOURCE_B_ARCHIVE_ROOT
) -> Tuple[RootYearCandidate, ...]:
    """V9_015 root-extraction entrypoint; this is OPTION_VALUE-only."""

    return extract_option_value_root_year_candidates(root_bytes, root_url)
