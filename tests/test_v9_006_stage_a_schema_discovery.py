from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
import copy
from types import SimpleNamespace

import pytest

from src import v9_006_stage_a_schema_discovery as schema
from src.v9_005_stage_a_jpx_probe import (
    CHATGPT_DECISION_REQUIRED, SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE as F3,
    SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE as F4, SOURCE_FAMILY_JPX_CALENDAR as F7,
    SOURCE_FAMILY_LISTED_ISSUES_MONTH_END as F1, SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT as F2,
    TERMINAL_PERIOD, V9005StageABlocked, source_object_slot_id,
)


def lock(raw=b"<html><table><tr><td>x</td></tr></table></html>", family=F7, period="2020-01", domain=schema.ObjectDomain.BASE):
    url = "https://www.jpx.co.jp/calendar/202001.html"
    return schema.VerifiedLockedObject("V9_005_STAGE_A_RAW_LOCK_V1", family, period, url, url, 200, "2020-01-01T00:00:00Z", len(raw), source_object_slot_id(family, period, url), sha256(raw).hexdigest(), raw, domain)


def assert_blocked(value):
    with pytest.raises(V9005StageABlocked) as exc: schema.profile_verified_lock(value)
    assert exc.value.reason == "IMPLEMENTATION_FAILURE"


def test_canonical_raw_lock_is_accepted():
    assert schema.profile_verified_lock(lock())["status"] == "PROFILED"


@pytest.mark.parametrize("mutator", [
    lambda value: replace(value, source_object_slot_id="a" * 64), lambda value: replace(value, requested_url="https://www.jpx.co.jp/other.html"),
    lambda value: replace(value, schema_version="bad"), lambda value: replace(value, source_family="BAD"),
    lambda value: replace(value, source_family="F5"), lambda value: replace(value, source_family="F6"),
    lambda value: replace(value, applicable_period=""), lambda value: replace(value, requested_url="https://example.com/a"),
    lambda value: replace(value, resolved_url="https://example.com/a"), lambda value: replace(value, retrieval_timestamp_utc="2020-01-01"),
    lambda value: replace(value, http_status=True), lambda value: replace(value, http_status=600),
    lambda value: replace(value, byte_length=True), lambda value: replace(value, byte_length=0),
    lambda value: replace(value, sha256="A" * 64), lambda value: replace(value, sha256="b" * 64),
    lambda value: replace(value, raw_bytes="not bytes"),
])
def test_raw_lock_provenance_rejects_every_noncanonical_variant(mutator): assert_blocked(mutator(lock()))


def test_raw_lock_validation_is_total_for_arbitrary_unhashable_fields():
    value = lock(); object.__setattr__(value, "source_family", []); assert_blocked(value)


def profile(family, period, domain, structure="a", text="ignored"):
    slot = sha256((family + period + domain.value + structure).encode()).hexdigest()
    return {"source_family": family, "applicable_period": period, "object_domain": domain.value, "source_object_slot_id": slot, "structural_profile_sha256": sha256(structure.encode()).hexdigest(), "header_text": text, "sample_text": text}


def periods(selected, family): return [item["applicable_period"] for item in selected if item["source_family"] == family]


def test_f1_terminal_and_all_f3_years_are_representatives():
    records = [profile(F1, TERMINAL_PERIOD, schema.ObjectDomain.TERMINAL)] + [profile(F3, str(year), schema.ObjectDomain.YEAR) for year in range(2017, 2026)]
    selected = schema.select_representatives(records)
    assert periods(selected, F1) == [TERMINAL_PERIOD]
    assert periods(selected, F3) == [str(year) for year in range(2017, 2026)]


@pytest.mark.parametrize("family", [F2, F4])
def test_base_earliest_and_latest_are_retained_even_when_profile_same(family):
    records = [profile(family, "2020-01", schema.ObjectDomain.BASE), profile(family, "2020-02", schema.ObjectDomain.BASE), profile(family, "2020-03", schema.ObjectDomain.BASE)]
    assert periods(schema.select_representatives(records), family) == ["2020-01", "2020-03"]


def test_profile_different_middle_selected_but_identical_middle_is_not():
    records = [profile(F2, "2020-01", schema.ObjectDomain.BASE, "a"), profile(F2, "2020-02", schema.ObjectDomain.BASE, "b"), profile(F2, "2020-03", schema.ObjectDomain.BASE, "a")]
    assert periods(schema.select_representatives(records), F2) == ["2020-01", "2020-02", "2020-03"]


def test_all_f2_bridges_and_f7_yearly_extremes_are_representatives():
    records = [profile(F2, "2026-01", schema.ObjectDomain.BRIDGE), profile(F2, "2026-02", schema.ObjectDomain.BRIDGE)]
    records += [profile(F7, "2020-01", schema.ObjectDomain.BASE), profile(F7, "2020-02", schema.ObjectDomain.BASE), profile(F7, "2021-01", schema.ObjectDomain.BASE), profile(F7, "2021-02", schema.ObjectDomain.BASE)]
    selected = schema.select_representatives(records)
    assert periods(selected, F2) == ["2026-01", "2026-02"]
    assert periods(selected, F7) == ["2020-01", "2020-02", "2021-01", "2021-02"]


@pytest.mark.parametrize("record", [profile(F1, "2020-01", schema.ObjectDomain.TERMINAL), profile(F2, "2020", schema.ObjectDomain.BASE)])
def test_invalid_family_domain_or_period_is_rejected(record):
    if record["source_family"] == F1: record["object_domain"] = schema.ObjectDomain.BASE.value
    with pytest.raises(V9005StageABlocked): schema.select_representatives([record])


def test_duplicate_identity_rejected_and_order_is_canonical_and_unique():
    a, b = profile(F4, "2020-02", schema.ObjectDomain.BASE), profile(F2, "2020-01", schema.ObjectDomain.BASE)
    selected = schema.select_representatives([a, b])
    assert selected == sorted(selected, key=lambda item: (item["source_family"], item["applicable_period"], item["source_object_slot_id"]))
    with pytest.raises(V9005StageABlocked): schema.select_representatives([a, dict(a)])


def test_selection_is_invariant_to_sampled_or_header_text():
    records = [profile(F2, "2020-01", schema.ObjectDomain.BASE, text="one"), profile(F2, "2020-02", schema.ObjectDomain.BASE, text="two")]
    changed = [{**item, "header_text": "changed", "sample_text": "changed"} for item in records]
    assert [item["source_object_slot_id"] for item in schema.select_representatives(records)] == [item["source_object_slot_id"] for item in schema.select_representatives(changed)]


def test_runner_has_no_argv_confirmation_and_entrypoint_is_decision_required():
    runner = Path("scripts/run_v9_006_stage_a_schema_discovery.py").read_text(encoding="utf-8")
    assert "--confirmation" not in runner and "argparse" not in runner and "os.environ" not in runner and "input(" not in runner
    with pytest.raises(V9005StageABlocked) as exc: schema.prepare_future_acquisition()
    assert exc.value.reason == CHATGPT_DECISION_REQUIRED


@pytest.mark.parametrize("family,period,domain", [
    (F1, TERMINAL_PERIOD, schema.ObjectDomain.TERMINAL),
    (F2, "2017-01", schema.ObjectDomain.BASE), (F2, "2025-12", schema.ObjectDomain.BASE),
    (F4, "2017-01", schema.ObjectDomain.BASE),
    (F7, "2017-01", schema.ObjectDomain.BASE), (F7, "2025-12", schema.ObjectDomain.BASE),
    (F7, "2016-09", schema.ObjectDomain.ENVELOPE_EXTRA), (F7, "2016-12", schema.ObjectDomain.ENVELOPE_EXTRA),
    (F7, "2026-01", schema.ObjectDomain.ENVELOPE_EXTRA), (F7, "2026-03", schema.ObjectDomain.ENVELOPE_EXTRA),
    (F3, "2017", schema.ObjectDomain.YEAR), (F3, "2025", schema.ObjectDomain.YEAR),
    (F2, "2026-01", schema.ObjectDomain.BRIDGE),
])
def test_domain_period_contract_accepts_exact_reviewed_domains(family, period, domain):
    assert schema._validate_domain_period(family, domain, period) is domain
    assert schema.profile_verified_lock(lock(family=family, period=period, domain=domain))["applicable_period"] == period


@pytest.mark.parametrize("family,period,domain", [
    (F1, "terminal", schema.ObjectDomain.TERMINAL), (F1, "2020-01", schema.ObjectDomain.TERMINAL),
    (F2, "2016-12", schema.ObjectDomain.BASE), (F2, "2026-01", schema.ObjectDomain.BASE),
    (F4, "2016-12", schema.ObjectDomain.BASE),
    (F7, "2016-12", schema.ObjectDomain.BASE), (F7, "2026-01", schema.ObjectDomain.BASE),
    (F7, "2020-02", schema.ObjectDomain.ENVELOPE_EXTRA), (F7, "2016-08", schema.ObjectDomain.ENVELOPE_EXTRA),
    (F7, "2026-04", schema.ObjectDomain.ENVELOPE_EXTRA),
    (F3, "2016", schema.ObjectDomain.YEAR), (F3, "2026", schema.ObjectDomain.YEAR),
    (F2, "2025-12", schema.ObjectDomain.BRIDGE),
])
def test_domain_period_contract_rejects_all_out_of_domain_values(family, period, domain):
    with pytest.raises(V9005StageABlocked) as exc: schema._validate_domain_period(family, domain, period)
    assert exc.value.reason == "IMPLEMENTATION_FAILURE"
    assert_blocked(lock(family=family, period=period, domain=domain))
    with pytest.raises(V9005StageABlocked): schema.select_representatives([profile(family, period, domain)])


class FakeSheet:
    def __init__(self, name, visibility, rows): self.name, self.visibility, self.rows = name, visibility, rows; self.nrows=len(rows); self.ncols=max(map(len, rows), default=0)
    def cell_type(self, row, col): return self.rows[row][col][0] if col < len(self.rows[row]) else 0
    def cell_value(self, row, col): return self.rows[row][col][1] if col < len(self.rows[row]) else ""


class FakeBook:
    def __init__(self, sheets): self.sheets=sheets; self.nsheets=len(sheets)
    def sheet_by_index(self, index): return self.sheets[index]


def ole_profile(monkeypatch, sheets):
    import xlrd
    monkeypatch.setattr(xlrd, "open_workbook", lambda **_: FakeBook(sheets))
    return schema.profile_verified_lock(lock(raw=b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"))


def test_ole_safe_schema_visibility_taxonomy_sampling_and_fingerprint(monkeypatch):
    import xlrd
    rows=[[(xlrd.XL_CELL_TEXT, "x" * 200), (xlrd.XL_CELL_NUMBER, 123), (xlrd.XL_CELL_DATE, 4), (xlrd.XL_CELL_BOOLEAN, 1), (xlrd.XL_CELL_ERROR, 7), (xlrd.XL_CELL_BLANK, "")] for _ in range(17)]
    result=ole_profile(monkeypatch, [FakeSheet("name", 0, rows), FakeSheet("hidden", 1, rows), FakeSheet("very", 2, rows)])
    evidence=result["structural_evidence"]
    assert [sheet["visibility"] for sheet in evidence["sheets"]] == ["VISIBLE", "HIDDEN", "VERY_HIDDEN"]
    assert set(evidence["sheets"][0]["column_cell_type_counts"][0]) == set(schema._CELL_TYPES)
    assert len([row for row in evidence["schema_neighborhood"] if row["sheet_ordinal"] == 1]) == 16
    assert not [row for row in evidence["schema_neighborhood"] if row["sheet_ordinal"] != 1]
    assert evidence["SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE"] is True
    assert all(set(cell) <= {"row_ordinal", "column_ordinal", "cell_type", "text"} and (cell["cell_type"] == "TEXT" or "text" not in cell) for row in evidence["schema_neighborhood"] for cell in row["cells"])
    assert len(evidence["schema_neighborhood"][0]["cells"][0]["text"]) == 160
    changed=ole_profile(monkeypatch, [FakeSheet("renamed", 0, rows), FakeSheet("hidden", 1, rows), FakeSheet("very", 2, rows)])
    assert changed["structural_profile_sha256"] == result["structural_profile_sha256"]
    dimensions=ole_profile(monkeypatch, [FakeSheet("name", 0, rows[:-1])])
    visibility=ole_profile(monkeypatch, [FakeSheet("name", 1, rows)])
    assert dimensions["structural_profile_sha256"] != result["structural_profile_sha256"] != visibility["structural_profile_sha256"]


def test_html_bounds_attributes_and_text_independent_fingerprint():
    html=("<html><title>" + "t" * 200 + "</title>" + "".join("<h1>h</h1>" for _ in range(34)) + "<table class='grid' id='x' role='table' href='https://bad'><tr><th>header</th><td>business</td></tr>" + "".join("<tr><td>x</td></tr>" for _ in range(17)) + "</table></html>").encode()
    result=schema.profile_verified_lock(lock(html)); evidence=result["structural_evidence"]
    assert len(evidence["title"]) == 160 and len(evidence["headings"]) == 32 and evidence["SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE"]
    assert evidence["tables"][0]["structural_attributes"] == [{"name":"class","value":"grid"},{"name":"id","value":"x"},{"name":"role","value":"table"}]
    assert "href" not in repr(evidence) and "https://" not in repr(evidence)
    changed=schema.profile_verified_lock(lock(html.replace(b"business", b"changed!")))
    assert changed["structural_profile_sha256"] == result["structural_profile_sha256"]
    topology=schema.profile_verified_lock(lock(html.replace(b"<td>business</td>", b"<td>business</td><td>new</td>")))
    attrs=schema.profile_verified_lock(lock(html.replace(b"class='grid'", b"class='other'")))
    assert topology["structural_profile_sha256"] != result["structural_profile_sha256"] != attrs["structural_profile_sha256"]


def test_safe_output_validator_is_closed_and_total(monkeypatch):
    valid=ole_profile(monkeypatch, [FakeSheet("ok", 0, [])])
    assert schema._validate_safe_profile(valid) == valid
    for mutate in (
        lambda item: item.update(extra=1),
        lambda item: item["structural_evidence"]["sheets"][0].update(extra=1),
        lambda item: item.update(raw=b"x"),
        lambda item: item["structural_evidence"].update(path="C:/x"),
        lambda item: item.update(structural_profile_sha256="bad"),
    ):
        bad=copy.deepcopy(valid); mutate(bad)
        with pytest.raises(V9005StageABlocked) as exc: schema._validate_safe_profile(bad)
        assert exc.value.reason == "IMPLEMENTATION_FAILURE"
    for malformed in ({"x": []}, {"x": {"y": []}}, ["bad"], {"x": b"bytes"}):
        with pytest.raises(V9005StageABlocked): schema._validate_safe_profile(malformed)


@pytest.mark.parametrize("html", [b"<title>" + b"x" * 161 + b"</title>", b"<h1>" + b"x" * 161 + b"</h1>", b"<table><tr><th>" + b"x" * 161 + b"</th></tr></table>", b"<table class='" + b"x" * 161 + b"'><tr><td>x</td></tr></table>"])
def test_all_html_text_truncation_has_narrower_provenance(html):
    assert schema.profile_verified_lock(lock(b"<html>" + html + b"</html>"))["structural_evidence"]["SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE"]


def test_heading_is_element_atomic_and_malformed_html_fails_closed():
    result=schema.profile_verified_lock(lock(b"<html><h1>Foo <span>Bar</span></h1></html>"))
    assert result["structural_evidence"]["headings"] == [{"tag":"h1","text":"Foo Bar"}]
    for raw in (b"<h1><h2>x</h2></h1>", b"<table><tr><td>x", b"<h1>x"):
        with pytest.raises(V9005StageABlocked): schema.profile_verified_lock(lock(b"<html>" + raw))


def test_final_validator_bounds_and_single_profiler_definitions(monkeypatch):
    import xlrd
    valid=ole_profile(monkeypatch, [FakeSheet("ok", 0, [[(xlrd.XL_CELL_TEXT, "x")]])])
    bad=copy.deepcopy(valid); row=bad["structural_evidence"]["schema_neighborhood"]
    row.extend(copy.deepcopy(row) for _ in range(16))
    with pytest.raises(V9005StageABlocked): schema._validate_safe_profile(bad)
    source=Path("src/v9_006_stage_a_schema_discovery.py").read_text(encoding="utf-8")
    assert source.count("def _html_structure(") == source.count("def _ole_structure(") == source.count("def profile_verified_lock(") == 1


def phase1_harness(monkeypatch):
    """Synthetic-only helper/lock harness; neither fetcher nor sockets are used."""
    events, locks = [], {}

    def add(family, period, domain):
        url = f"https://www.jpx.co.jp/synthetic/{family}-{period}-{domain.value}.html"
        slot = source_object_slot_id(family, period, url)
        locks[slot] = {
            "schema_version": "V9_005_STAGE_A_RAW_LOCK_V1", "source_family": family,
            "applicable_period": period, "requested_url": url, "resolved_url": url,
            "http_status": 200, "retrieval_timestamp_utc": "2020-01-01T00:00:00Z",
            "byte_length": 34, "sha256": sha256(b"<html><table></table></html>").hexdigest(),
            "raw": b"<html><table></table></html>",
        }
        locks[slot]["byte_length"] = len(locks[slot]["raw"])
        return slot

    f1 = add(F1, TERMINAL_PERIOD, schema.ObjectDomain.TERMINAL)
    f2 = {month: add(F2, month, schema.ObjectDomain.BASE) for month in schema.inventory_months()}
    f4 = {month: add(F4, month, schema.ObjectDomain.BASE) for month in schema.inventory_months()}
    f3 = {year: add(F3, str(year), schema.ObjectDomain.YEAR) for year in range(2017, 2026)}
    f7_base = {month: add(F7, month, schema.ObjectDomain.BASE) for month in schema.inventory_months()}
    f7_extra = {month: add(F7, month, schema.ObjectDomain.ENVELOPE_EXTRA) for month in schema.calendar_envelope_extra_months()}

    def f1_helper(*args, **kwargs): events.append(("F1",)); return f1, 1
    def f24_helper(*args, **kwargs):
        events.append((kwargs["source_family"], kwargs["requested_month"])); return (f2 if kwargs["source_family"] == F2 else f4)[kwargs["requested_month"]], 2
    def f3_helper(*args, **kwargs):
        events.append(("F3",))
        refs = {(F3, f"{year}-{month:02d}"): (f3[year],) for year in range(2017, 2026) for month in range(1, 13)}
        return SimpleNamespace(base_coverage_references=refs, network_attempt_count=3)
    def f7_helper(*args, **kwargs):
        events.append(("F7",))
        return SimpleNamespace(
            base_coverage_references={(F7, month): (slot,) for month, slot in f7_base.items()},
            envelope_extra_references={month: (slot,) for month, slot in f7_extra.items()}, network_attempt_count=4,
        )
    monkeypatch.setattr(schema, "acquire_f1_terminal_evidence", f1_helper)
    monkeypatch.setattr(schema, "acquire_f2_f4_monthly_evidence", f24_helper)
    monkeypatch.setattr(schema, "acquire_f3_required_slots", f3_helper)
    monkeypatch.setattr(schema, "acquire_f7_required_slots", f7_helper)
    monkeypatch.setattr(schema, "read_locked_payload_by_slot_id", lambda _root, slot: locks[slot])
    return events, locks, f3_helper, f7_helper


def test_phase1_core_exact_binding_counts_safe_profiles_and_representatives(monkeypatch):
    events, _locks, _f3, _f7 = phase1_harness(monkeypatch)
    reader_calls = []
    original_reader = schema.read_locked_payload_by_slot_id
    monkeypatch.setattr(schema, "read_locked_payload_by_slot_id", lambda root, slot: (reader_calls.append(slot), original_reader(root, slot))[1])
    result = schema.run_phase1_schema_discovery_core("unused", fetcher=object(), sleep=object(), clock=object())
    months = schema.inventory_months()
    assert events == [("F1",)] + [(family, month) for month in months for family in (F2, F4)] + [("F3",), ("F7",)]
    assert len(result.evidence_slot_ids) == len(set(result.evidence_slot_ids)) == len(result.safe_profiles) == 341
    assert sum(item["source_family"] == F1 for item in result.safe_profiles) == 1
    assert sum(item["source_family"] == F2 and item["object_domain"] == "BASE" for item in result.safe_profiles) == 108
    assert sum(item["source_family"] == F3 and item["object_domain"] == "YEAR" for item in result.safe_profiles) == 9
    assert sum(item["source_family"] == F4 and item["object_domain"] == "BASE" for item in result.safe_profiles) == 108
    assert sum(item["source_family"] == F7 for item in result.safe_profiles) == 115
    assert sum(item["source_family"] == F7 and item["object_domain"] == "BASE" for item in result.safe_profiles) == 108
    assert sum(item["source_family"] == F7 and item["object_domain"] == "ENVELOPE_EXTRA" for item in result.safe_profiles) == 7
    assert not [item for item in result.safe_profiles if item["source_family"] in {"F5", "F6"} or item["object_domain"] == "BRIDGE"]
    assert result.representative_safe_profiles == tuple(schema.select_representatives(result.safe_profiles))
    assert result.network_attempt_count == 1 + 216 * 2 + 3 + 4
    assert all("raw" not in item for item in result.safe_profiles)
    assert set(reader_calls) == set(result.evidence_slot_ids)  # support locks have no profiler path


@pytest.mark.parametrize("kind", ["bad_f3", "bad_f7", "duplicate", "missing", "mismatch"])
def test_phase1_core_malformed_evidence_fails_closed(monkeypatch, kind):
    _events, locks, f3_helper, f7_helper = phase1_harness(monkeypatch)
    if kind == "bad_f3":
        def bad_f3(*args, **kwargs):
            result = f3_helper(*args, **kwargs); result.base_coverage_references.pop((F3, "2017-01")); return result
        monkeypatch.setattr(schema, "acquire_f3_required_slots", bad_f3)
    elif kind == "bad_f7":
        def bad_f7(*args, **kwargs):
            result = f7_helper(*args, **kwargs); result.envelope_extra_references.pop(next(iter(result.envelope_extra_references))); return result
        monkeypatch.setattr(schema, "acquire_f7_required_slots", bad_f7)
    elif kind == "duplicate":
        first = next(iter(locks)); second = next(iter(list(locks)[1:]))
        locks[second] = locks[first]
        # Return F1's ID for the first F2 object, creating an exact duplicate evidence identity.
        original = schema.acquire_f2_f4_monthly_evidence
        monkeypatch.setattr(schema, "acquire_f2_f4_monthly_evidence", lambda *args, **kwargs: (first, original(*args, **kwargs)[1]) if kwargs["source_family"] == F2 and kwargs["requested_month"] == schema.inventory_months()[0] else original(*args, **kwargs))
    elif kind == "missing":
        locks.clear()
    else:
        first = next(iter(locks.values())); first["source_family"] = F4
    with pytest.raises(V9005StageABlocked) as exc:
        schema.run_phase1_schema_discovery_core("unused", fetcher=object(), sleep=object(), clock=object())
    assert exc.value.reason == "IMPLEMENTATION_FAILURE"


def test_phase1_core_source_excludes_bridge_seam_and_runner_stays_closed():
    source = Path("src/v9_006_stage_a_schema_discovery.py").read_text(encoding="utf-8")
    assert "f2_bridge_months" not in source
    assert "acquire_f2_f4_required_slots" not in source
    with pytest.raises(V9005StageABlocked) as exc: schema.prepare_future_acquisition()
    assert exc.value.reason == CHATGPT_DECISION_REQUIRED


def test_validator_final_status_and_html_state_regressions(monkeypatch):
    valid=ole_profile(monkeypatch, [FakeSheet("x", 0, [])])
    for status, fmt in (("BAD", schema.FORMAT_HTML), ("PROFILED", schema.FORMAT_PDF), (schema.FORMAT_REQUIRES_FOLLOWUP, schema.FORMAT_HTML)):
        bad=copy.deepcopy(valid); bad["status"], bad["container_format"] = status, fmt
        with pytest.raises(V9005StageABlocked): schema._validate_safe_profile(bad)
    for raw in (b"<table><tr><td>A</td><tr><td>B</td></tr></table>", b"<table><tr><td>A<td>B</td></tr></table>", b"<td>x</td>", b"<tr><td>x</td></tr>", b"</td>", b"</tr>", b"</table>"):
        with pytest.raises(V9005StageABlocked): schema.profile_verified_lock(lock(b"<html>" + raw + b"</html>"))
