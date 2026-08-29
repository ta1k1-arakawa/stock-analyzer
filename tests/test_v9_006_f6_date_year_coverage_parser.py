from __future__ import annotations
from hashlib import sha256
import json
import pytest
from src import v9_006_f6_date_year_coverage_parser as coverage

class Sheet:
    nrows = 2
    def __init__(self, types, values): self.types, self.values, self.calls = types, values, []
    def cell_type(self, row, col): return self.types[row][col]
    def cell_value(self, row, col): self.calls.append((row, col)); return self.values[row][col]
class Book:
    datemode = 0
    def __init__(self, sheet): self.sheet = sheet
    def sheet_by_index(self, index): assert index == 0; return self.sheet
def structural(c4=1, c6=1):
    profiles=[]
    for ordinal, count in ((4,c4),(6,c6)):
        profiles.append({"sheet_ordinal":1,"column_ordinal":ordinal,"cell_type_counts":{"EMPTY":0,"BLANK":0,"TEXT":0,"NUMBER":0,"DATE":count,"BOOLEAN":0,"ERROR":0}})
    return {"status":"STRUCTURAL_FORMAT_CAPTURED","container_format":"OLE_COMPOUND_FILE","open_parse_status":"OPEN_PARSE_OK","sheet_table_count":1,"structural_dimensions":[{"ordinal":1,"row_count":2,"column_count":6,"visibility":"VISIBLE","object_type":"WORKSHEET"}],"cell_type_profiles":profiles}
def digest(evidence): return sha256(json.dumps(evidence,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def captured(evidence):
    return {"status":"F6_YEAR_COVERAGE_CAPTURED","structural_profile_sha256":digest(evidence),"structural_profile_hash_verified":True,"date_column_ordinals":[4,6],"raw_bytes_read_for_integrity":True,"child_content_inspected":True,"date_year_value_read":True,"coverage_evaluated":True,"coverage_result_accepted":True,"network_request_count":0,"year_histograms":{"4":[{"year":2016,"count":1}],"6":[{"year":2016,"count":1}]},"covered_years":[2016],"covered_required_years":[],"missing_required_years":[2017,2018,2019,2020,2021,2022,2023,2024,2025],"all_required_years_covered":False}
def test_frozen_constants_exact():
    assert coverage.DATE_COLUMN_ORDINALS == (4,6) and coverage.REQUIRED_YEARS == tuple(range(2017,2026))
def test_validator_captured_and_forged_derivations_rejected(monkeypatch):
    evidence=structural(); monkeypatch.setattr(coverage,"EXPECTED_STRUCTURAL_PROFILE_SHA256",digest(evidence)); value=captured(evidence); assert coverage.safe_coverage_evidence(value,evidence)==value
    for key,bad in (("covered_years",[]),("covered_required_years",[2016]),("missing_required_years",[]),("all_required_years_covered",True)):
        forged=dict(value); forged[key]=bad
        with pytest.raises(coverage.CoverageBlocked): coverage.safe_coverage_evidence(forged,evidence)
def test_validator_rejects_histogram_sum_and_status_identity(monkeypatch):
    evidence=structural(); monkeypatch.setattr(coverage,"EXPECTED_STRUCTURAL_PROFILE_SHA256",digest(evidence)); value=captured(evidence)
    bad=json.loads(json.dumps(value)); bad["year_histograms"]["4"][0]["count"]=2
    with pytest.raises(coverage.CoverageBlocked): coverage.safe_coverage_evidence(bad,evidence)
    bad=dict(value); bad["status"]="F6_YEAR_COVERAGE_AMBIGUOUS"; bad["coverage_result_accepted"]=False
    for key in ("covered_years","covered_required_years","missing_required_years","all_required_years_covered"): bad.pop(key)
    with pytest.raises(coverage.CoverageBlocked): coverage.safe_coverage_evidence(bad,evidence)
def test_validator_unhashable_is_non_crashing():
    with pytest.raises(coverage.CoverageBlocked): coverage.safe_coverage_evidence({"status":[]}, {})
def test_run_exact_open_columns_and_no_nondates(monkeypatch):
    types=[[0,0,0,coverage.xlrd.XL_CELL_DATE,0,coverage.xlrd.XL_CELL_DATE],[0]*6]; values=[[None,None,None,1,None,1],[None]*6]
    sheet=Sheet(types,values); evidence=structural(); monkeypatch.setattr(coverage,"EXPECTED_STRUCTURAL_PROFILE_SHA256",digest(evidence)); monkeypatch.setattr(coverage,"_structural_gate",lambda *_:(evidence,digest(evidence))); opened={}
    monkeypatch.setattr(coverage.xlrd,"open_workbook",lambda **kwargs: (opened.update(kwargs) or Book(sheet)))
    monkeypatch.setattr(coverage.xlrd,"xldate_as_tuple",lambda serial,datemode:(2018,1,1,0,0,0))
    result=coverage._run_verified_bytes(b"synthetic",inspector=lambda _:evidence)
    assert result["status"]=="F6_YEAR_COVERAGE_CAPTURED" and sheet.calls==[(0,3),(0,5)]
    assert opened=={"file_contents":b"synthetic","formatting_info":True,"on_demand":False,"ragged_rows":False}
def test_unequal_is_ambiguous_without_derived(monkeypatch):
    types=[[0,0,0,coverage.xlrd.XL_CELL_DATE,0,coverage.xlrd.XL_CELL_DATE],[0]*6]; sheet=Sheet(types,[[None]*6,[None]*6]); evidence=structural(); monkeypatch.setattr(coverage,"EXPECTED_STRUCTURAL_PROFILE_SHA256",digest(evidence)); monkeypatch.setattr(coverage,"_structural_gate",lambda *_:(evidence,digest(evidence))); monkeypatch.setattr(coverage.xlrd,"open_workbook",lambda **_:Book(sheet)); years=iter((2018,2019)); monkeypatch.setattr(coverage.xlrd,"xldate_as_tuple",lambda *_: (next(years),1,1,0,0,0))
    result=coverage._run_verified_bytes(b"x",inspector=lambda _:evidence)
    assert result["status"]=="F6_YEAR_COVERAGE_AMBIGUOUS" and not ({"covered_years","missing_required_years"}&set(result))
def test_hash_failure_before_date_read(monkeypatch):
    monkeypatch.setattr(coverage,"_structural_gate",lambda *_: (_ for _ in ()).throw(RuntimeError()))
    with pytest.raises(coverage.CoverageBlocked) as exc: coverage._run_verified_bytes(b"x")
    assert exc.value.evidence["structural_profile_sha256"] is None and not exc.value.evidence["date_year_value_read"]
def test_hash_mismatch_never_opens(monkeypatch):
    evidence=structural(); monkeypatch.setattr(coverage,"_structural_gate",lambda *_:(evidence,digest(evidence))); monkeypatch.setattr(coverage.xlrd,"open_workbook",lambda **_: pytest.fail("opened"))
    with pytest.raises(coverage.CoverageBlocked) as exc: coverage._run_verified_bytes(b"x",inspector=lambda _:evidence)
    assert exc.value.evidence["structural_profile_sha256"]==digest(evidence) and not exc.value.evidence["structural_profile_hash_verified"]
def test_first_column_zero_date_failure_keeps_false(monkeypatch):
    evidence=structural(c4=1,c6=1); monkeypatch.setattr(coverage,"EXPECTED_STRUCTURAL_PROFILE_SHA256",digest(evidence)); monkeypatch.setattr(coverage,"_structural_gate",lambda *_:(evidence,digest(evidence))); monkeypatch.setattr(coverage.xlrd,"open_workbook",lambda **_:Book(Sheet([[0]*6,[0]*6],[[None]*6,[None]*6])))
    with pytest.raises(coverage.CoverageBlocked) as exc: coverage._run_verified_bytes(b"x",inspector=lambda _:evidence)
    assert not exc.value.evidence["date_year_value_read"]
