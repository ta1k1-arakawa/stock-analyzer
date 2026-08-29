from __future__ import annotations
from hashlib import sha256
import pytest
from src import v9_006_stage_a_schema_discovery as schema
from src.v9_005_stage_a_jpx_probe import SOURCE_FAMILY_JPX_CALENDAR, SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, V9005StageABlocked

def lock(raw=b"<html><title>T</title><h1>Head</h1><table><tr><th>Code</th><th>Date</th></tr><tr><td>ABC</td><td>2020</td></tr></table></html>", family=SOURCE_FAMILY_JPX_CALENDAR, period="2020-01"):
    return schema.VerifiedLockedObject(family,period,"a"*64,sha256(raw).hexdigest(),raw)

def test_html_profile_safe_deterministic_and_fingerprint_excludes_text():
    first=schema.profile_verified_lock(lock()); second=schema.profile_verified_lock(lock(b"<html><title>X</title><h1>Other</h1><table><tr><th>Other</th><th>Date</th></tr><tr><td>secret</td><td>2021</td></tr></table></html>"))
    assert first["container_format"]==schema.FORMAT_HTML and first["structural_profile_sha256"]==second["structural_profile_sha256"]
    assert first["structural_evidence"]["headings"][0]["text"]=="Head" and "http" not in repr(first)

@pytest.mark.parametrize("raw,expected", [(b"%PDF-x",schema.FORMAT_PDF),(b"nope",schema.FORMAT_UNKNOWN),(b"PK\x03\x04",schema.FORMAT_OOXML_ZIP)])
def test_unsupported_formats_explicit(raw,expected):
    result=schema.profile_verified_lock(lock(raw)); assert result["container_format"]==expected and result["status"]==schema.FORMAT_REQUIRES_FOLLOWUP

def test_lock_validation_and_exclusions_fail_closed():
    with pytest.raises(V9005StageABlocked): schema.profile_verified_lock(lock(b"x",family="F5"))
    with pytest.raises(V9005StageABlocked): schema.profile_verified_lock(lock(b"x",family="F6"))
    bad=lock(); object.__setattr__(bad,"sha256","b"*64)
    with pytest.raises(V9005StageABlocked): schema.profile_verified_lock(bad)

def test_representatives_are_structure_period_only_and_runner_is_gated():
    profiles=[schema.profile_verified_lock(lock(period="2020-02")),schema.profile_verified_lock(lock(period="2020-01")),schema.profile_verified_lock(lock(b"<html><table><tr><td>x</td></tr></table></html>",period="2020-03"))]
    assert [p["applicable_period"] for p in schema.select_representatives(profiles)]==["2020-01","2020-03"]
    with pytest.raises(V9005StageABlocked): schema.prepare_future_acquisition("wrong")
    with pytest.raises(V9005StageABlocked): schema.prepare_future_acquisition(schema.SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_CONFIRMATION)
