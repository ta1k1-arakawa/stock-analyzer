"""Offline, safe structural schema discovery for already verified Stage-A locks."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
import json
from typing import Any, Sequence

from src.v9_005_stage_a_jpx_probe import (
    CHATGPT_DECISION_REQUIRED, GOVERNANCE_FAILURE, IMPLEMENTATION_FAILURE,
    SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
    SOURCE_FAMILY_JPX_CALENDAR, SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
    SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, V9005StageABlocked,
)

SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_CONFIRMATION = "V9_006_STAGE_A_SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_ONE_SHOT"
SCHEMA_EVIDENCE_CLASS = "DEVELOPMENT_PUBLIC_SOURCE_STRUCTURE"
FORMAT_OLE_BIFF, FORMAT_OOXML_ZIP, FORMAT_HTML, FORMAT_PDF, FORMAT_UNKNOWN = "OLE_BIFF", "OOXML_ZIP", "HTML", "PDF", "UNKNOWN"
FORMAT_REQUIRES_FOLLOWUP = "FORMAT_REQUIRES_FOLLOWUP"
_ALLOWED_FAMILIES = frozenset({SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, SOURCE_FAMILY_JPX_CALENDAR})

@dataclass(frozen=True)
class VerifiedLockedObject:
    source_family: str
    applicable_period: str
    source_object_slot_id: str
    sha256: str
    raw_bytes: bytes

def _fail() -> None:
    raise V9005StageABlocked(IMPLEMENTATION_FAILURE)

def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def _text(value: Any) -> str:
    if not isinstance(value, str): _fail()
    return value.strip()[:160]

def detect_container_format(raw: Any) -> str:
    if not isinstance(raw, bytes): _fail()
    if raw.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"): return FORMAT_OLE_BIFF
    if raw.startswith(b"PK\x03\x04"): return FORMAT_OOXML_ZIP
    if raw.startswith(b"%PDF-"): return FORMAT_PDF
    prefix = raw[:1024].lstrip().lower()
    if prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body", b"<table")): return FORMAT_HTML
    return FORMAT_UNKNOWN

def _validate(lock: Any) -> VerifiedLockedObject:
    if type(lock) is not VerifiedLockedObject: _fail()
    if (lock.source_family not in _ALLOWED_FAMILIES or not isinstance(lock.applicable_period, str) or not lock.applicable_period
            or not isinstance(lock.source_object_slot_id, str) or len(lock.source_object_slot_id) != 64
            or not isinstance(lock.sha256, str) or len(lock.sha256) != 64 or not isinstance(lock.raw_bytes, bytes)):
        _fail()
    if any(c not in "0123456789abcdef" for c in lock.source_object_slot_id + lock.sha256): _fail()
    if sha256(lock.raw_bytes).hexdigest() != lock.sha256: _fail()
    return lock

class _HtmlProfile(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True); self.title=""; self.headings=[]; self.tables=[]; self._table=None; self._row=None; self._cell=None; self._depth=0
    def handle_starttag(self, tag, attrs):
        tag=tag.lower()
        if tag == "table": self._table={"rows":[], "headers":[], "tags":[]}; self.tables.append(self._table)
        if self._table is not None: self._table["tags"].append((tag, tuple(sorted(k for k,v in attrs if k in {"class","id","role"}))))
        if tag == "tr" and self._table is not None: self._row=[]
        if tag in {"td","th"} and self._row is not None: self._cell={"tag":tag,"text":"","column":len(self._row)+1}
        if tag in {"title","h1","h2","h3","h4","h5","h6"}: self._depth=tag
    def handle_data(self, data):
        if self._cell is not None: self._cell["text"] += data
        if self._depth:
            if self._depth == "title": self.title += data
            else: self.headings.append((self._depth, _text(data)))
    def handle_endtag(self, tag):
        tag=tag.lower()
        if tag in {"td","th"} and self._cell is not None:
            self._cell["text"]=_text(self._cell["text"]); self._row.append(self._cell)
            if tag == "th" and self._table is not None: self._table["headers"].append((len(self._table["rows"])+1,self._cell["column"],self._cell["text"]))
            self._cell=None
        if tag == "tr" and self._row is not None and self._table is not None: self._table["rows"].append(self._row); self._row=None
        if tag == "table": self._table=None
        if tag == self._depth: self._depth=0

def _html_structure(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try: parser=_HtmlProfile(); parser.feed(raw.decode("utf-8", "replace")); parser.close()
    except Exception: _fail()
    tables=[]; samples=[]
    for ordinal, table in enumerate(parser.tables, 1):
        rows=table["rows"]; tables.append({"ordinal":ordinal,"row_count":len(rows),"column_count":max((len(r) for r in rows),default=0),"headers":table["headers"],"tag_profile":sorted(table["tags"])})
        for rowno,row in enumerate(rows,1):
            if len(samples)<16 and any(cell["text"] for cell in row): samples.append({"table_ordinal":ordinal,"row_ordinal":rowno,"cells":[cell["text"] for cell in row[:64]]})
    structure={"format":FORMAT_HTML,"title_present":bool(_text(parser.title)),"heading_tags":[tag for tag,_ in parser.headings],"tables":[{"ordinal":t["ordinal"],"row_count":t["row_count"],"column_count":t["column_count"],"tag_profile":t["tag_profile"]} for t in tables]}
    evidence={"title":_text(parser.title),"headings":[{"tag":tag,"text":text} for tag,text in parser.headings],"tables":tables,"schema_neighborhood":samples}
    return structure,evidence

def _ole_structure(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import xlrd
        book=xlrd.open_workbook(file_contents=raw, formatting_info=True, on_demand=False, ragged_rows=False)
        sheets=[]; samples=[]
        for index in range(book.nsheets):
            sheet=book.sheet_by_index(index); types=[]
            for col in range(sheet.ncols):
                counts={}
                for row in range(sheet.nrows): counts[str(sheet.cell_type(row,col))]=counts.get(str(sheet.cell_type(row,col)),0)+1
                types.append(counts)
            sheets.append({"ordinal":index+1,"name":_text(sheet.name),"row_count":sheet.nrows,"column_count":sheet.ncols,"visibility":"VISIBLE","column_types":types})
            if sheet.ncols<=64:
                for row in range(sheet.nrows):
                    cells=[]
                    for col in range(sheet.ncols):
                        typ=sheet.cell_type(row,col)
                        cells.append({"column_ordinal":col+1,"cell_type":typ, **({"text":_text(str(sheet.cell_value(row,col)))} if typ==1 else {})})
                    if any("text" in c and c["text"] for c in cells) and len(samples)<16: samples.append({"sheet_ordinal":index+1,"row_ordinal":row+1,"cells":cells})
        structure={"format":FORMAT_OLE_BIFF,"sheets":[{k:v for k,v in s.items() if k!="name"} for s in sheets]}
        return structure,{"sheets":sheets,"schema_neighborhood":samples,"SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE":any(s["column_count"]>64 for s in sheets)}
    except V9005StageABlocked: raise
    except Exception: _fail()

def profile_verified_lock(lock: Any) -> dict[str, Any]:
    item=_validate(lock); fmt=detect_container_format(item.raw_bytes)
    if fmt == FORMAT_HTML: structure,evidence=_html_structure(item.raw_bytes); status="PROFILED"
    elif fmt == FORMAT_OLE_BIFF: structure,evidence=_ole_structure(item.raw_bytes); status="PROFILED"
    else: structure={"format":fmt}; evidence={}; status=FORMAT_REQUIRES_FOLLOWUP
    fingerprint=sha256(_canonical_json(structure).encode()).hexdigest()
    return {"status":status,"source_family":item.source_family,"applicable_period":item.applicable_period,"source_object_slot_id":item.source_object_slot_id,"sha256":item.sha256,"byte_length":len(item.raw_bytes),"container_format":fmt,"structural_profile_sha256":fingerprint,"structural_evidence":evidence}

def select_representatives(profiles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(profiles, Sequence) or isinstance(profiles,(str,bytes)): _fail()
    chosen=[]; seen=set()
    for p in sorted(profiles,key=lambda x:(x.get("source_family",""),x.get("applicable_period",""))):
        if not isinstance(p,dict) or p.get("source_family") not in _ALLOWED_FAMILIES or not isinstance(p.get("applicable_period"),str) or not isinstance(p.get("structural_profile_sha256"),str): _fail()
        key=(p["source_family"],p["applicable_period"][:4],p["structural_profile_sha256"])
        if key not in seen: chosen.append(p); seen.add(key)
    return chosen

def prepare_future_acquisition(confirmation: Any) -> None:
    """Gate only: aggregate acquisition remains deliberately unimplemented."""
    if confirmation != SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_CONFIRMATION: raise V9005StageABlocked(GOVERNANCE_FAILURE)
    raise V9005StageABlocked(CHATGPT_DECISION_REQUIRED)
