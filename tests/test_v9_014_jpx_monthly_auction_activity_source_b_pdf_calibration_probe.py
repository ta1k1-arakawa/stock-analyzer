from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import generate_v9_014_source_b_calibration_synthetic_pdf as generator
from src import v9_014_jpx_monthly_auction_activity_source_b_pdf_calibration_probe as probe


FIXTURE_BYTES = generator.build_pdf_bytes()
FIXTURE_SHA256 = hashlib.sha256(FIXTURE_BYTES).hexdigest()


def _item(identity: probe.CalibrationIdentity) -> probe.CalibrationObjectInput:
    return probe.CalibrationObjectInput(
        pdf_bytes=FIXTURE_BYTES,
        logical_month=identity.logical_month,
        object_part=identity.object_part,
        expected_sha256=FIXTURE_SHA256,
    )


def test_exact_eight_preregistered_identities_pass() -> None:
    assert probe.CALIBRATION_OBJECT_COUNT == 8
    assert len(set(probe.REQUIRED_CALIBRATION_IDENTITIES)) == 8
    result = probe.probe_calibration_bundle(
        [_item(identity) for identity in probe.REQUIRED_CALIBRATION_IDENTITIES]
    )
    assert result.status == probe.CALIBRATION_BUNDLE_PASS
    assert len(result.results) == 8
    assert all(item.status == probe.CALIBRATION_PROBE_PASS for item in result.results)


@pytest.mark.parametrize(
    "objects",
    [
        [_item(identity) for identity in probe.REQUIRED_CALIBRATION_IDENTITIES[:-1]],
        [_item(identity) for identity in probe.REQUIRED_CALIBRATION_IDENTITIES]
        + [_item(probe.CalibrationIdentity("2018-01", probe.NORMAL_MONTHLY_REPORT2_OBJECT))],
        [
            _item(identity)
            for identity in (
                probe.REQUIRED_CALIBRATION_IDENTITIES[:-1]
                + (probe.REQUIRED_CALIBRATION_IDENTITIES[0],)
            )
        ],
        [
            _item(identity)
            for identity in (
                probe.REQUIRED_CALIBRATION_IDENTITIES[:-1]
                + (probe.CalibrationIdentity("2018-01", probe.NORMAL_MONTHLY_REPORT2_OBJECT),)
            )
        ],
    ],
    ids=["missing", "extra", "duplicate", "substitute"],
)
def test_identity_bundle_variants_fail_closed(
    objects: list[probe.CalibrationObjectInput],
) -> None:
    result = probe.probe_calibration_bundle(objects)
    assert result.status == probe.CALIBRATION_IDENTITY_FAILURE
    assert result.results == ()


def test_wrong_expected_sha_fails_before_parser_load(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_loaded(_name: str) -> object:
        raise AssertionError("parser loaded before SHA verification")

    monkeypatch.setattr(probe.importlib, "import_module", fail_if_loaded)
    identity = probe.REQUIRED_CALIBRATION_IDENTITIES[0]
    result = probe.probe_calibration_object(
        FIXTURE_BYTES,
        identity.logical_month,
        identity.object_part,
        "0" * 64,
    )
    assert result.status == probe.CALIBRATION_SHA256_MISMATCH


def test_wrong_pdfplumber_version_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_parser = SimpleNamespace(__version__="0.11.9")
    monkeypatch.setattr(probe, "_load_pdfplumber", lambda: fake_parser)
    identity = probe.REQUIRED_CALIBRATION_IDENTITIES[0]
    result = probe.probe_calibration_object(
        FIXTURE_BYTES, identity.logical_month, identity.object_part, FIXTURE_SHA256
    )
    assert result.status == probe.CALIBRATION_PDFPLUMBER_VERSION_MISMATCH


def test_malformed_pdf_fails_closed() -> None:
    malformed = b"not a PDF"
    identity = probe.REQUIRED_CALIBRATION_IDENTITIES[0]
    result = probe.probe_calibration_object(
        malformed,
        identity.logical_month,
        identity.object_part,
        hashlib.sha256(malformed).hexdigest(),
    )
    assert result.status == probe.CALIBRATION_PDF_PARSE_FAILURE
    assert result.evidence is None


def test_synthetic_pdf_structural_probe_passes() -> None:
    identity = probe.REQUIRED_CALIBRATION_IDENTITIES[0]
    result = probe.probe_calibration_object(
        FIXTURE_BYTES, identity.logical_month, identity.object_part, FIXTURE_SHA256
    )
    assert result.status == probe.CALIBRATION_PROBE_PASS
    assert result.evidence is not None
    assert result.evidence["page_count"] == 1
    page = result.evidence["pages"][0]
    assert page["width"] == generator.PAGE_WIDTH
    assert page["height"] == generator.PAGE_HEIGHT
    assert page["lines"]
    assert page["rectangles"]
    assert page["characters"]


def test_numeric_glyph_content_is_masked_and_unit_token_is_preserved() -> None:
    identity = probe.REQUIRED_CALIBRATION_IDENTITIES[0]
    result = probe.probe_calibration_object(
        FIXTURE_BYTES, identity.logical_month, identity.object_part, FIXTURE_SHA256
    )
    page = result.evidence["pages"][0]
    masked_text = page["masked_character_text"]
    assert "DIGITS ####" in masked_text
    assert "GLYPH_TEST_#" in masked_text
    assert "4821" not in masked_text
    assert "thous.shs." in masked_text


def test_japanese_strings_are_unchanged_except_decimal_digits() -> None:
    assert probe.mask_decimal_digits("日本語の単位 千株") == "日本語の単位 千株"
    assert probe.mask_decimal_digits("日本語１２3テスト") == "日本語###テスト"
    assert probe.mask_decimal_digits("thous.shs.") == "thous.shs."


def test_forbidden_semantic_or_acquisition_paths_are_absent() -> None:
    source = inspect.getsource(probe)
    for forbidden in (
        "extract_text",
        "extract_table",
        "extract_tables",
        "find_tables",
        "classify_date",
        "evaluate_cross_source_relation",
        "trading_dates",
        "requests",
        "urlopen",
        "urllib",
        "subprocess",
    ):
        assert forbidden not in source


def test_generator_is_stdlib_only_and_deterministic() -> None:
    source = inspect.getsource(generator)
    assert "reportlab" not in source
    assert generator.build_pdf_bytes() == generator.build_pdf_bytes()
    assert "thous.shs." in source
    assert "4821" in source
