"""Offline, safe structural schema discovery for already verified Stage-A locks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from html.parser import HTMLParser
import json
from typing import Any, Sequence

from src.v9_005_stage_a_jpx_probe import (
    CHATGPT_DECISION_REQUIRED, GOVERNANCE_FAILURE, IMPLEMENTATION_FAILURE,
    SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
    SOURCE_FAMILY_JPX_CALENDAR, SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
    SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, V9005StageABlocked,
    INVENTORY_LAST_YEAR_MONTH, TERMINAL_PERIOD, _is_canonical_raw_lock_timestamp,
    _parse_year_month, calendar_envelope_extra_months, inventory_months,
    source_object_slot_id, validate_jpx_url,
)

SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_CONFIRMATION = "V9_006_STAGE_A_SCHEMA_DISCOVERY_PUBLIC_ACQUISITION_ONE_SHOT"
SCHEMA_EVIDENCE_CLASS = "DEVELOPMENT_PUBLIC_SOURCE_STRUCTURE"
FORMAT_OLE_BIFF, FORMAT_OOXML_ZIP, FORMAT_HTML, FORMAT_PDF, FORMAT_UNKNOWN = "OLE_BIFF", "OOXML_ZIP", "HTML", "PDF", "UNKNOWN"
FORMAT_REQUIRES_FOLLOWUP = "FORMAT_REQUIRES_FOLLOWUP"
_ALLOWED_FAMILIES = frozenset({SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, SOURCE_FAMILY_JPX_CALENDAR})

class ObjectDomain(str, Enum):
    TERMINAL = "TERMINAL"
    BASE = "BASE"
    BRIDGE = "BRIDGE"
    ENVELOPE_EXTRA = "ENVELOPE_EXTRA"
    YEAR = "YEAR"

_DOMAINS_BY_FAMILY = {
    SOURCE_FAMILY_LISTED_ISSUES_MONTH_END: frozenset({ObjectDomain.TERMINAL}),
    SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT: frozenset({ObjectDomain.BASE, ObjectDomain.BRIDGE}),
    SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE: frozenset({ObjectDomain.YEAR}),
    SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE: frozenset({ObjectDomain.BASE}),
    SOURCE_FAMILY_JPX_CALENDAR: frozenset({ObjectDomain.BASE, ObjectDomain.ENVELOPE_EXTRA}),
}

@dataclass(frozen=True)
class VerifiedLockedObject:
    schema_version: str
    source_family: str
    applicable_period: str
    requested_url: str
    resolved_url: str
    http_status: int
    retrieval_timestamp_utc: str
    byte_length: int
    source_object_slot_id: str
    sha256: str
    raw_bytes: bytes
    object_domain: ObjectDomain

def _fail() -> None:
    raise V9005StageABlocked(IMPLEMENTATION_FAILURE)

def _validate_domain_period(family: Any, domain: Any, period: Any) -> ObjectDomain:
    """Validate the one closed family/domain/period contract in both seams."""
    try:
        if family not in _ALLOWED_FAMILIES or type(domain) is not ObjectDomain or type(period) is not str:
            _fail()
        if domain not in _DOMAINS_BY_FAMILY[family]:
            _fail()
        if family == SOURCE_FAMILY_LISTED_ISSUES_MONTH_END:
            if period != TERMINAL_PERIOD: _fail()
        elif family in {SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE} and domain is ObjectDomain.BASE:
            if period not in inventory_months(): _fail()
        elif family == SOURCE_FAMILY_JPX_CALENDAR and domain is ObjectDomain.BASE:
            if period not in inventory_months(): _fail()
        elif family == SOURCE_FAMILY_JPX_CALENDAR and domain is ObjectDomain.ENVELOPE_EXTRA:
            if period not in calendar_envelope_extra_months(): _fail()
        elif family == SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE:
            if period not in {month[:4] for month in inventory_months()}: _fail()
        elif family == SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT and domain is ObjectDomain.BRIDGE:
            if _parse_year_month(period) <= INVENTORY_LAST_YEAR_MONTH: _fail()
        else:
            _fail()
        return domain
    except Exception as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc

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
    try:
        if type(lock) is not VerifiedLockedObject:
            _fail()
        if (
            lock.schema_version != "V9_005_STAGE_A_RAW_LOCK_V1"
            or lock.source_family not in _ALLOWED_FAMILIES
            or type(lock.applicable_period) is not str or not lock.applicable_period
            or type(lock.object_domain) is not ObjectDomain
            or type(lock.requested_url) is not str or type(lock.resolved_url) is not str
            or not _is_canonical_raw_lock_timestamp(lock.retrieval_timestamp_utc)
            or type(lock.http_status) is not int or not 100 <= lock.http_status <= 599
            or type(lock.byte_length) is not int or lock.byte_length < 0
            or type(lock.raw_bytes) is not bytes
            or type(lock.sha256) is not str or len(lock.sha256) != 64
            or type(lock.source_object_slot_id) is not str or len(lock.source_object_slot_id) != 64
        ):
            _fail()
        _validate_domain_period(lock.source_family, lock.object_domain, lock.applicable_period)
        validate_jpx_url(lock.requested_url)
        validate_jpx_url(lock.resolved_url)
        if any(c not in "0123456789abcdef" for c in lock.sha256 + lock.source_object_slot_id):
            _fail()
        if lock.byte_length != len(lock.raw_bytes) or sha256(lock.raw_bytes).hexdigest() != lock.sha256:
            _fail()
        if source_object_slot_id(lock.source_family, lock.applicable_period, lock.requested_url) != lock.source_object_slot_id:
            _fail()
        return lock
    except Exception as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc

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
    return {"status":status,"source_family":item.source_family,"object_domain":item.object_domain.value,"applicable_period":item.applicable_period,"source_object_slot_id":item.source_object_slot_id,"sha256":item.sha256,"byte_length":len(item.raw_bytes),"container_format":fmt,"structural_profile_sha256":fingerprint,"structural_evidence":evidence}

def _validated_profile(profile: Any) -> dict[str, Any]:
    try:
        if type(profile) is not dict: _fail()
        family, domain, period = profile.get("source_family"), profile.get("object_domain"), profile.get("applicable_period")
        slot, structural = profile.get("source_object_slot_id"), profile.get("structural_profile_sha256")
        if family not in _ALLOWED_FAMILIES or type(domain) is not str or type(period) is not str or type(slot) is not str or type(structural) is not str:
            _fail()
        domain_value = _validate_domain_period(family, ObjectDomain(domain), period)
        if len(slot) != 64 or len(structural) != 64:
            _fail()
        if any(c not in "0123456789abcdef" for c in slot + structural): _fail()
        return profile
    except V9005StageABlocked:
        raise
    except Exception as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc

def select_representatives(profiles: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the frozen domain/period-only representative contract."""
    try:
        if not isinstance(profiles, Sequence) or isinstance(profiles, (str, bytes)): _fail()
        records = [_validated_profile(profile) for profile in profiles]
        slots = [record["source_object_slot_id"] for record in records]
        if len(slots) != len(set(slots)): _fail()
        selected: list[dict[str, Any]] = []
        for record in records:
            family, domain = record["source_family"], ObjectDomain(record["object_domain"])
            if family == SOURCE_FAMILY_LISTED_ISSUES_MONTH_END or family == SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE or (family == SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT and domain is ObjectDomain.BRIDGE):
                selected.append(record)
        for family, domains in ((SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, {ObjectDomain.BASE}), (SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, {ObjectDomain.BASE}), (SOURCE_FAMILY_JPX_CALENDAR, {ObjectDomain.BASE, ObjectDomain.ENVELOPE_EXTRA})):
            groups: dict[int, list[dict[str, Any]]] = {}
            for record in records:
                if record["source_family"] == family and ObjectDomain(record["object_domain"]) in domains:
                    groups.setdefault(_parse_year_month(record["applicable_period"])[0], []).append(record)
            for group in groups.values():
                ordered = sorted(group, key=lambda item: item["applicable_period"])
                earliest, latest = ordered[0], ordered[-1]
                selected.extend((earliest, latest))
                selected.extend(item for item in ordered if item["structural_profile_sha256"] != earliest["structural_profile_sha256"])
        unique = {item["source_object_slot_id"]: item for item in selected}
        return sorted(unique.values(), key=lambda item: (item["source_family"], item["applicable_period"], item["source_object_slot_id"]))
    except V9005StageABlocked:
        raise
    except Exception as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc

def prepare_future_acquisition() -> None:
    """Gate only: aggregate acquisition remains deliberately unimplemented."""
    raise V9005StageABlocked(CHATGPT_DECISION_REQUIRED)
