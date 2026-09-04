"""Offline structural calibration probe for the V9_014 SOURCE_B PDF path.

This module accepts only caller-supplied PDF bytes and preregistered logical
object identities.  It uses the one frozen parser, ``pdfplumber==0.11.10``,
and records low-level page, glyph, line, and rectangle structure only.  The
returned glyph text is outcome-safe: every Unicode decimal digit is replaced
with ``#`` and every other character is preserved byte-for-character at the
Python string level.  No semantic cell or date operation is performed here.

There is deliberately no URL, filesystem, network, alternate parser, OCR,
or package-management surface in this module.  Callers own acquisition and
must provide already-preserved bytes plus their expected SHA-256.
"""

from __future__ import annotations

import hashlib
import importlib
import io
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence


PDFPLUMBER_VERSION = "0.11.10"

NORMAL_MONTHLY_REPORT2_OBJECT = "NORMAL_MONTHLY_REPORT2_OBJECT"
PRE_APRIL_1_REFERENCE_OBJECT = "PRE_APRIL_1_REFERENCE_OBJECT"

CALIBRATION_PROBE_PASS = "CALIBRATION_PROBE_PASS"
CALIBRATION_BUNDLE_PASS = "CALIBRATION_BUNDLE_PASS"
CALIBRATION_INPUT_FAILURE = "CALIBRATION_INPUT_FAILURE"
CALIBRATION_IDENTITY_FAILURE = "CALIBRATION_IDENTITY_FAILURE"
CALIBRATION_SHA256_MISMATCH = "CALIBRATION_SHA256_MISMATCH"
CALIBRATION_EXPECTED_SHA256_INVALID = "CALIBRATION_EXPECTED_SHA256_INVALID"
CALIBRATION_PDFPLUMBER_IMPORT_FAILURE = "CALIBRATION_PDFPLUMBER_IMPORT_FAILURE"
CALIBRATION_PDFPLUMBER_VERSION_MISMATCH = "CALIBRATION_PDFPLUMBER_VERSION_MISMATCH"
CALIBRATION_PDF_PARSE_FAILURE = "CALIBRATION_PDF_PARSE_FAILURE"
CALIBRATION_BUNDLE_MEMBER_FAILURE = "CALIBRATION_BUNDLE_MEMBER_FAILURE"


@dataclass(frozen=True)
class CalibrationIdentity:
    logical_month: str
    object_part: str


REQUIRED_CALIBRATION_IDENTITIES: tuple[CalibrationIdentity, ...] = (
    CalibrationIdentity("2017-01", NORMAL_MONTHLY_REPORT2_OBJECT),
    CalibrationIdentity("2019-12", NORMAL_MONTHLY_REPORT2_OBJECT),
    CalibrationIdentity("2020-01", NORMAL_MONTHLY_REPORT2_OBJECT),
    CalibrationIdentity("2022-03", NORMAL_MONTHLY_REPORT2_OBJECT),
    CalibrationIdentity("2022-04", PRE_APRIL_1_REFERENCE_OBJECT),
    CalibrationIdentity("2022-04", NORMAL_MONTHLY_REPORT2_OBJECT),
    CalibrationIdentity("2022-05", NORMAL_MONTHLY_REPORT2_OBJECT),
    CalibrationIdentity("2026-01", NORMAL_MONTHLY_REPORT2_OBJECT),
)
CALIBRATION_OBJECT_COUNT = len(REQUIRED_CALIBRATION_IDENTITIES)
_REQUIRED_IDENTITY_SET = frozenset(REQUIRED_CALIBRATION_IDENTITIES)


@dataclass(frozen=True)
class CalibrationObjectInput:
    pdf_bytes: bytes
    logical_month: str
    object_part: str
    expected_sha256: str


@dataclass(frozen=True)
class CalibrationProbeResult:
    status: str
    logical_month: Optional[str] = None
    object_part: Optional[str] = None
    observed_sha256: Optional[str] = None
    evidence: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class CalibrationBundleResult:
    status: str
    results: tuple[CalibrationProbeResult, ...] = ()
    failing_identity: Optional[CalibrationIdentity] = None


def mask_decimal_digits(text: str) -> str:
    """Mask Unicode decimal digits without any other text transformation."""

    if not isinstance(text, str):
        raise TypeError("calibration glyph text must be str")
    return "".join("#" if character.isdecimal() else character for character in text)


def _is_sha256_hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _failure(
    status: str,
    *,
    logical_month: Optional[str] = None,
    object_part: Optional[str] = None,
    observed_sha256: Optional[str] = None,
) -> CalibrationProbeResult:
    return CalibrationProbeResult(
        status=status,
        logical_month=logical_month,
        object_part=object_part,
        observed_sha256=observed_sha256,
    )


def _structural_number(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    return None


def _geometry(item: Mapping[str, Any]) -> dict[str, object]:
    captured: dict[str, object] = {}
    for field in (
        "x0",
        "x1",
        "y0",
        "y1",
        "top",
        "bottom",
        "doctop",
        "width",
        "height",
        "linewidth",
        "adv",
    ):
        if field in item:
            value = _structural_number(item[field])
            if value is not None:
                captured[field] = value
    return captured


def _capture_character(character: Mapping[str, Any]) -> dict[str, object]:
    text = character.get("text")
    if not isinstance(text, str):
        raise ValueError("invalid glyph text")
    captured = _geometry(character)
    captured["text"] = mask_decimal_digits(text)
    for field in ("fontname", "size", "upright"):
        if field not in character:
            continue
        if field == "fontname":
            value = character[field]
            if isinstance(value, str):
                captured[field] = value
        elif field == "upright":
            value = character[field]
            if isinstance(value, bool):
                captured[field] = value
        else:
            value = _structural_number(character[field])
            if value is not None:
                captured[field] = value
    return captured


def _capture_shape(shape: Mapping[str, Any]) -> dict[str, object]:
    captured = _geometry(shape)
    for field in ("stroke", "fill"):
        if field in shape and isinstance(shape[field], bool):
            captured[field] = shape[field]
    return captured


def _load_pdfplumber() -> object | None:
    try:
        return importlib.import_module("pdfplumber")
    except Exception:
        return None


def _capture_pdf_structure(pdf: object) -> dict[str, object]:
    pages: list[dict[str, object]] = []
    for page_index, page in enumerate(pdf.pages):
        characters = [_capture_character(item) for item in page.chars]
        lines = [_capture_shape(item) for item in page.lines]
        rectangles = [_capture_shape(item) for item in page.rects]
        pages.append(
            {
                "index": page_index,
                "width": _structural_number(page.width),
                "height": _structural_number(page.height),
                "rotation": _structural_number(getattr(page, "rotation", None)),
                "masked_character_text": "".join(item["text"] for item in characters),
                "characters": characters,
                "lines": lines,
                "rectangles": rectangles,
            }
        )
    return {"page_count": len(pages), "pages": pages}


def probe_calibration_object(
    pdf_bytes: bytes,
    logical_month: str,
    object_part: str,
    expected_sha256: str,
) -> CalibrationProbeResult:
    """Probe one preregistered object from caller-supplied bytes only."""

    identity = CalibrationIdentity(logical_month, object_part)
    if not isinstance(pdf_bytes, bytes) or identity not in _REQUIRED_IDENTITY_SET:
        return _failure(
            CALIBRATION_INPUT_FAILURE,
            logical_month=logical_month if isinstance(logical_month, str) else None,
            object_part=object_part if isinstance(object_part, str) else None,
        )

    observed_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    if not _is_sha256_hex(expected_sha256):
        return _failure(
            CALIBRATION_EXPECTED_SHA256_INVALID,
            logical_month=logical_month,
            object_part=object_part,
            observed_sha256=observed_sha256,
        )
    if observed_sha256 != expected_sha256:
        return _failure(
            CALIBRATION_SHA256_MISMATCH,
            logical_month=logical_month,
            object_part=object_part,
            observed_sha256=observed_sha256,
        )

    pdfplumber = _load_pdfplumber()
    if pdfplumber is None:
        return _failure(
            CALIBRATION_PDFPLUMBER_IMPORT_FAILURE,
            logical_month=logical_month,
            object_part=object_part,
            observed_sha256=observed_sha256,
        )
    observed_version = getattr(pdfplumber, "__version__", None)
    if observed_version != PDFPLUMBER_VERSION:
        return _failure(
            CALIBRATION_PDFPLUMBER_VERSION_MISMATCH,
            logical_month=logical_month,
            object_part=object_part,
            observed_sha256=observed_sha256,
        )

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            evidence = _capture_pdf_structure(pdf)
    except Exception:
        return _failure(
            CALIBRATION_PDF_PARSE_FAILURE,
            logical_month=logical_month,
            object_part=object_part,
            observed_sha256=observed_sha256,
        )

    return CalibrationProbeResult(
        status=CALIBRATION_PROBE_PASS,
        logical_month=logical_month,
        object_part=object_part,
        observed_sha256=observed_sha256,
        evidence=evidence,
    )


def probe_calibration_bundle(
    objects: Sequence[CalibrationObjectInput],
) -> CalibrationBundleResult:
    """Require and probe exactly the eight frozen identities once each."""

    if not isinstance(objects, (list, tuple)) or len(objects) != CALIBRATION_OBJECT_COUNT:
        return CalibrationBundleResult(CALIBRATION_IDENTITY_FAILURE)

    indexed: dict[CalibrationIdentity, CalibrationObjectInput] = {}
    for item in objects:
        if not isinstance(item, CalibrationObjectInput):
            return CalibrationBundleResult(CALIBRATION_IDENTITY_FAILURE)
        identity = CalibrationIdentity(item.logical_month, item.object_part)
        if identity not in _REQUIRED_IDENTITY_SET or identity in indexed:
            return CalibrationBundleResult(CALIBRATION_IDENTITY_FAILURE)
        indexed[identity] = item
    if frozenset(indexed) != _REQUIRED_IDENTITY_SET:
        return CalibrationBundleResult(CALIBRATION_IDENTITY_FAILURE)

    results: list[CalibrationProbeResult] = []
    for identity in REQUIRED_CALIBRATION_IDENTITIES:
        item = indexed[identity]
        result = probe_calibration_object(
            item.pdf_bytes,
            item.logical_month,
            item.object_part,
            item.expected_sha256,
        )
        results.append(result)
        if result.status != CALIBRATION_PROBE_PASS:
            return CalibrationBundleResult(
                CALIBRATION_BUNDLE_MEMBER_FAILURE,
                tuple(results),
                identity,
            )
    return CalibrationBundleResult(CALIBRATION_BUNDLE_PASS, tuple(results))
