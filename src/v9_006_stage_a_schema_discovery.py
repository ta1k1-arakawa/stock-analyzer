"""Offline, safe structural schema discovery for already verified Stage-A locks."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from html.parser import HTMLParser
import json
import re
from typing import Any, Mapping, Sequence

from src.v9_005_stage_a_jpx_probe import (
    CHATGPT_DECISION_REQUIRED, GOVERNANCE_FAILURE, IMPLEMENTATION_FAILURE,
    SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
    SOURCE_FAMILY_JPX_CALENDAR, SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
    SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, V9005StageABlocked,
    INVENTORY_LAST_YEAR_MONTH, TERMINAL_PERIOD, _is_canonical_raw_lock_timestamp,
    _parse_year_month, acquire_f1_terminal_evidence,
    acquire_f2_f4_monthly_evidence, acquire_f3_required_slots,
    acquire_f7_required_slots, calendar_envelope_extra_months, inventory_months,
    read_locked_payload_by_slot_id,
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

MAX_HEADINGS = 32
MAX_DETAILED_TABLES = 32
MAX_HEADERS_PER_TABLE = 256
MAX_SAMPLE_ROWS_PER_TABLE = 16
MAX_SAMPLE_CELLS_PER_ROW = 64
MAX_STRUCTURAL_ATTR_VALUES_PER_TABLE = 256
MAX_TEXT_CODEPOINTS = 160
_CELL_TYPES = ("EMPTY", "BLANK", "TEXT", "NUMBER", "DATE", "BOOLEAN", "ERROR")
_VISIBILITIES = ("VISIBLE", "HIDDEN", "VERY_HIDDEN")

def _bounded_text(value: Any) -> str:
    if not isinstance(value, str): _fail()
    return value.strip()[:MAX_TEXT_CODEPOINTS]

def _contains_unsafe(value: Any) -> bool:
    if isinstance(value, bytes): return True
    if isinstance(value, str): return bool(re.search(r"(?:https?://|file:|[A-Za-z]:[\\/])", value))
    if isinstance(value, list): return any(_contains_unsafe(item) for item in value)
    if isinstance(value, dict):
        return any((not isinstance(key, str) or key in {"requested_url", "resolved_url", "href", "src", "action", "path"} or "exception" in key.lower() or _contains_unsafe(item)) for key, item in value.items())
    return not (value is None or type(value) in {bool, int})

def _validate_safe_profile(value: Any) -> dict[str, Any]:
    """Closed, total safe-output boundary for schema-discovery evidence."""
    try:
        top = {"status", "source_family", "object_domain", "applicable_period", "source_object_slot_id", "sha256", "byte_length", "container_format", "structural_profile_sha256", "structural_evidence"}
        if type(value) is not dict or set(value) != top or _contains_unsafe(value): _fail()
        if value["status"] not in {"PROFILED", FORMAT_REQUIRES_FOLLOWUP} or (value["status"] == "PROFILED" and value["container_format"] not in {FORMAT_OLE_BIFF, FORMAT_HTML}) or (value["status"] == FORMAT_REQUIRES_FOLLOWUP and value["container_format"] not in {FORMAT_OOXML_ZIP, FORMAT_PDF, FORMAT_UNKNOWN}): _fail()
        if type(value["byte_length"]) is not int or value["byte_length"] < 0: _fail()
        for key in ("source_object_slot_id", "sha256", "structural_profile_sha256"):
            if type(value[key]) is not str or not re.fullmatch(r"[0-9a-f]{64}", value[key]): _fail()
        if value["source_family"] not in _ALLOWED_FAMILIES or type(value["object_domain"]) is not str:
            _fail()
        _validate_domain_period(value["source_family"], ObjectDomain(value["object_domain"]), value["applicable_period"])
        evidence = value["structural_evidence"]
        if value["status"] == FORMAT_REQUIRES_FOLLOWUP:
            if evidence != {}: _fail()
            return value
        if type(evidence) is not dict or evidence.get("format") != value["container_format"] or type(evidence.get("SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE")) is not bool: _fail()
        if value["container_format"] == FORMAT_OLE_BIFF:
            if set(evidence) != {"format", "sheet_count", "sheets", "schema_neighborhood", "SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE"}: _fail()
            if type(evidence["sheet_count"]) is not int or evidence["sheet_count"] < 0 or type(evidence["sheets"]) is not list or len(evidence["sheets"]) != evidence["sheet_count"]: _fail()
            for expected, sheet in enumerate(evidence["sheets"], 1):
                if type(sheet) is not dict or set(sheet) != {"sheet_ordinal", "sheet_name", "row_count", "column_count", "visibility", "object_type", "column_cell_type_counts"}: _fail()
                if sheet["sheet_ordinal"] != expected or sheet["visibility"] not in _VISIBILITIES or sheet["object_type"] != "WORKSHEET" or type(sheet["sheet_name"]) is not str or len(sheet["sheet_name"]) > MAX_TEXT_CODEPOINTS: _fail()
                if type(sheet["row_count"]) is not int or type(sheet["column_count"]) is not int or sheet["row_count"] < 0 or sheet["column_count"] < 0 or type(sheet["column_cell_type_counts"]) is not list or len(sheet["column_cell_type_counts"]) != sheet["column_count"]: _fail()
                for counts in sheet["column_cell_type_counts"]:
                    if type(counts) is not dict or set(counts) != set(_CELL_TYPES) or any(type(n) is not int or n < 0 for n in counts.values()) or sum(counts.values()) != sheet["row_count"]: _fail()
            visible = {sheet["sheet_ordinal"] for sheet in evidence["sheets"] if sheet["visibility"] == "VISIBLE"}
            if type(evidence["schema_neighborhood"]) is not list: _fail()
            rows_by_sheet: dict[int, list[int]] = {}
            for row in evidence["schema_neighborhood"]:
                if type(row) is not dict or set(row) != {"sheet_ordinal", "row_ordinal", "cells"} or row["sheet_ordinal"] not in visible or type(row["row_ordinal"]) is not int or row["row_ordinal"] < 1 or type(row["cells"]) is not list or len(row["cells"]) > MAX_SAMPLE_CELLS_PER_ROW: _fail()
                rows_by_sheet.setdefault(row["sheet_ordinal"], []).append(row["row_ordinal"])
                sheet=evidence["sheets"][row["sheet_ordinal"]-1]
                columns=[]
                if row["row_ordinal"] > sheet["row_count"]: _fail()
                for cell in row["cells"]:
                    if type(cell) is not dict or not set(cell) in ({"row_ordinal", "column_ordinal", "cell_type"}, {"row_ordinal", "column_ordinal", "cell_type", "text"}) or cell.get("cell_type") not in _CELL_TYPES or cell.get("row_ordinal") != row["row_ordinal"] or type(cell.get("column_ordinal")) is not int or not 1 <= cell["column_ordinal"] <= min(sheet["column_count"], MAX_SAMPLE_CELLS_PER_ROW) or (cell["cell_type"] == "TEXT" and ("text" not in cell or type(cell["text"]) is not str or len(cell["text"]) > MAX_TEXT_CODEPOINTS)) or (cell["cell_type"] != "TEXT" and "text" in cell): _fail()
                    columns.append(cell["column_ordinal"])
                if columns != sorted(set(columns)): _fail()
            if any(len(rows)>MAX_SAMPLE_ROWS_PER_TABLE or rows != sorted(set(rows)) for rows in rows_by_sheet.values()): _fail()
            if any(sheet["column_count"]>MAX_SAMPLE_CELLS_PER_ROW for sheet in evidence["sheets"]) and not evidence["SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE"]: _fail()
        elif value["container_format"] == FORMAT_HTML:
            if set(evidence) != {"format", "title", "headings", "table_count", "tables", "schema_neighborhood", "SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE"}: _fail()
            if type(evidence["title"]) is not str or len(evidence["title"]) > MAX_TEXT_CODEPOINTS or type(evidence["headings"]) is not list or len(evidence["headings"]) > MAX_HEADINGS or type(evidence["table_count"]) is not int or evidence["table_count"] < 0 or type(evidence["tables"]) is not list or len(evidence["tables"]) > MAX_DETAILED_TABLES: _fail()
            for heading in evidence["headings"]:
                if type(heading) is not dict or set(heading) != {"tag", "text"} or heading["tag"] not in {"h1", "h2", "h3", "h4", "h5", "h6"} or type(heading["text"]) is not str or len(heading["text"]) > MAX_TEXT_CODEPOINTS: _fail()
            for table in evidence["tables"]:
                if type(table) is not dict or set(table) != {"table_ordinal", "row_count", "column_count", "headers", "structural_attributes"} or type(table["table_ordinal"]) is not int or type(table["row_count"]) is not int or type(table["column_count"]) is not int or table["row_count"] < 0 or table["column_count"] < 0 or type(table["headers"]) is not list or len(table["headers"]) > MAX_HEADERS_PER_TABLE or type(table["structural_attributes"]) is not list or len(table["structural_attributes"]) > MAX_STRUCTURAL_ATTR_VALUES_PER_TABLE: _fail()
                headers=[]
                for header in table["headers"]:
                    if type(header) is not dict or set(header) != {"row_ordinal", "column_ordinal", "text"} or type(header["row_ordinal"]) is not int or type(header["column_ordinal"]) is not int or not 1 <= header["row_ordinal"] <= table["row_count"] or not 1 <= header["column_ordinal"] <= table["column_count"] or type(header["text"]) is not str or len(header["text"]) > MAX_TEXT_CODEPOINTS: _fail()
                    headers.append((header["row_ordinal"],header["column_ordinal"]))
                if headers != sorted(set(headers)): _fail()
                attributes=[]
                for attribute in table["structural_attributes"]:
                    if type(attribute) is not dict or set(attribute) != {"name", "value"} or attribute["name"] not in {"class", "id", "role"} or type(attribute["value"]) is not str or len(attribute["value"]) > MAX_TEXT_CODEPOINTS: _fail()
                    attributes.append((attribute["name"],attribute["value"]))
                if attributes != sorted(set(attributes)): _fail()
            if evidence["table_count"] < len(evidence["tables"]) or [table["table_ordinal"] for table in evidence["tables"]] != list(range(1, len(evidence["tables"])+1)) or (evidence["table_count"] > len(evidence["tables"]) and not evidence["SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE"]): _fail()
            if type(evidence["schema_neighborhood"]) is not list: _fail()
            tables_by_ordinal={table["table_ordinal"]:table for table in evidence["tables"]}; rows_by_table={}
            for row in evidence["schema_neighborhood"]:
                if type(row) is not dict or set(row) != {"table_ordinal", "row_ordinal", "cells"} or row.get("table_ordinal") not in tables_by_ordinal or type(row.get("row_ordinal")) is not int or row["row_ordinal"] < 1 or type(row["cells"]) is not list or len(row["cells"]) > MAX_SAMPLE_CELLS_PER_ROW: _fail()
                if row["row_ordinal"] > tables_by_ordinal[row["table_ordinal"]]["row_count"]: _fail()
                rows_by_table.setdefault(row["table_ordinal"], []).append(row["row_ordinal"]); columns=[]
                for cell in row["cells"]:
                    if type(cell) is not dict or set(cell) != {"column_ordinal", "cell_type", "text"} or cell["cell_type"] != "TEXT" or type(cell["column_ordinal"]) is not int or cell["column_ordinal"] < 1 or cell["column_ordinal"] > min(tables_by_ordinal[row["table_ordinal"]]["column_count"], MAX_SAMPLE_CELLS_PER_ROW) or type(cell["text"]) is not str or len(cell["text"]) > MAX_TEXT_CODEPOINTS: _fail()
                    columns.append(cell["column_ordinal"])
                if columns != sorted(set(columns)): _fail()
            if any(len(rows)>MAX_SAMPLE_ROWS_PER_TABLE or rows != sorted(set(rows)) for rows in rows_by_table.values()): _fail()
        else: _fail()
        return value
    except Exception as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc

class _SafeHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True); self.title=""; self.headings=[]; self.tables=[]; self.current=None; self.row=None; self.cell=None; self.heading=None; self.truncated=False
    def _capture(self, value: str) -> str:
        text=value.strip()
        if len(text)>MAX_TEXT_CODEPOINTS: self.truncated=True
        return text[:MAX_TEXT_CODEPOINTS]
    def handle_starttag(self, tag, attrs):
        tag=tag.lower()
        if tag == "table":
            if self.current is not None: _fail()
            self.current={"rows":[], "headers":[], "attrs":[], "tags":[]}; self.tables.append(self.current)
        if self.current is not None:
            self.current["tags"].append(tag)
            for key, value in attrs:
                if key.lower() in {"class", "id", "role"} and value is not None:
                    if len(self.current["attrs"]) < MAX_STRUCTURAL_ATTR_VALUES_PER_TABLE: self.current["attrs"].append((key.lower(), self._capture(value)))
                    else: self.truncated=True
        if tag == "tr":
            if self.current is None or self.row is not None: _fail()
            self.row=[]
        if tag in {"td", "th"}:
            if self.row is None or self.cell is not None: _fail()
            self.cell={"tag":tag,"text":"","column_ordinal":len(self.row)+1}
        if tag == "title" or tag in {"h1","h2","h3","h4","h5","h6"}:
            if self.heading is not None: _fail()
            self.heading={"tag":tag,"text":""}
    def handle_data(self, data):
        if self.cell is not None: self.cell["text"] += data
        if self.heading is not None: self.heading["text"] += data
    def handle_endtag(self, tag):
        tag=tag.lower()
        if tag in {"td","th"}:
            if self.cell is None or self.cell["tag"] != tag: _fail()
            self.cell["text"]=self._capture(self.cell["text"]); self.row.append(self.cell)
            if tag == "th" and self.current is not None:
                if len(self.current["headers"]) < MAX_HEADERS_PER_TABLE: self.current["headers"].append({"row_ordinal":len(self.current["rows"])+1,"column_ordinal":self.cell["column_ordinal"],"text":self.cell["text"]})
                else: self.truncated=True
            self.cell=None
        if tag == "tr":
            if self.current is None or self.row is None or self.cell is not None: _fail()
            self.current["rows"].append(self.row); self.row=None
        if tag == "table":
            if self.current is None or self.row is not None or self.cell is not None: _fail()
            self.current=None
        if self.heading is not None and tag == self.heading["tag"]:
            text=self._capture(self.heading["text"])
            if tag == "title": self.title=text
            elif len(self.headings) < MAX_HEADINGS: self.headings.append({"tag":tag,"text":text})
            else: self.truncated=True
            self.heading=None
    def close(self):
        super().close()
        if self.current is not None or self.row is not None or self.cell is not None or self.heading is not None: _fail()

def _html_structure(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try: parser=_SafeHtmlParser(); parser.feed(raw.decode("utf-8", "replace")); parser.close()
    except Exception: _fail()
    tables=[]; samples=[]; truncated=parser.truncated or len(parser.tables)>MAX_DETAILED_TABLES
    for ordinal, table in enumerate(parser.tables[:MAX_DETAILED_TABLES], 1):
        rows=table["rows"]; cols=max((len(row) for row in rows), default=0); attrs=sorted(set(table["attrs"]));
        tables.append({"table_ordinal":ordinal,"row_count":len(rows),"column_count":cols,"headers":table["headers"],"structural_attributes":[{"name":k,"value":v} for k,v in attrs]})
        nonempty=[row for row in rows if any(cell["text"] for cell in row)]
        if len(nonempty)>MAX_SAMPLE_ROWS_PER_TABLE: truncated=True
        for row_ordinal, row in [(index+1,row) for index,row in enumerate(rows) if any(cell["text"] for cell in row)][:MAX_SAMPLE_ROWS_PER_TABLE]:
            if len(row)>MAX_SAMPLE_CELLS_PER_ROW: truncated=True
            samples.append({"table_ordinal":ordinal,"row_ordinal":row_ordinal,"cells":[{"column_ordinal":cell["column_ordinal"],"cell_type":"TEXT","text":cell["text"]} for cell in row[:MAX_SAMPLE_CELLS_PER_ROW]]})
    structure={"format":FORMAT_HTML,"table_count":len(parser.tables),"tables":[{"ordinal":index+1,"rows":len(table["rows"]),"columns":max((len(row) for row in table["rows"]),default=0),"tags":table["tags"],"attrs":sorted(set(table["attrs"]))} for index,table in enumerate(parser.tables)]}
    return structure,{"format":FORMAT_HTML,"title":_bounded_text(parser.title),"headings":parser.headings,"table_count":len(parser.tables),"tables":tables,"schema_neighborhood":samples,"SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE":truncated}

def _ole_structure(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import xlrd; book=xlrd.open_workbook(file_contents=raw, formatting_info=True, on_demand=False, ragged_rows=False)
        types={xlrd.XL_CELL_EMPTY:"EMPTY",xlrd.XL_CELL_BLANK:"BLANK",xlrd.XL_CELL_TEXT:"TEXT",xlrd.XL_CELL_NUMBER:"NUMBER",xlrd.XL_CELL_DATE:"DATE",xlrd.XL_CELL_BOOLEAN:"BOOLEAN",xlrd.XL_CELL_ERROR:"ERROR"}; vis={0:"VISIBLE",1:"HIDDEN",2:"VERY_HIDDEN"}; sheets=[]; samples=[]; narrow=False; fingerprint=[]
        for index in range(book.nsheets):
            sheet=book.sheet_by_index(index); visibility=vis.get(sheet.visibility)
            if visibility is None: _fail()
            columns=[]
            for col in range(sheet.ncols):
                counts={key:0 for key in _CELL_TYPES}
                for row in range(sheet.nrows):
                    category=types.get(sheet.cell_type(row,col))
                    if category is None: _fail()
                    counts[category]+=1
                columns.append(counts)
            if len(sheet.name.strip()) > MAX_TEXT_CODEPOINTS: narrow=True
            sheets.append({"sheet_ordinal":index+1,"sheet_name":_bounded_text(sheet.name),"row_count":sheet.nrows,"column_count":sheet.ncols,"visibility":visibility,"object_type":"WORKSHEET","column_cell_type_counts":columns})
            fingerprint.append({"ordinal":index+1,"rows":sheet.nrows,"columns":sheet.ncols,"visibility":visibility,"profiles":columns})
            if sheet.ncols>MAX_SAMPLE_CELLS_PER_ROW: narrow=True
            if visibility == "VISIBLE":
                count=0
                for row in range(sheet.nrows):
                    cells=[]
                    for col in range(min(sheet.ncols,MAX_SAMPLE_CELLS_PER_ROW)):
                        category=types.get(sheet.cell_type(row,col)); cell={"row_ordinal":row+1,"column_ordinal":col+1,"cell_type":category}
                        if category == "TEXT":
                            source_text=str(sheet.cell_value(row,col))
                            if len(source_text.strip()) > MAX_TEXT_CODEPOINTS: narrow=True
                            cell["text"]=_bounded_text(source_text)
                        cells.append(cell)
                    if any(cell["cell_type"] != "EMPTY" for cell in cells):
                        if count < MAX_SAMPLE_ROWS_PER_TABLE: samples.append({"sheet_ordinal":index+1,"row_ordinal":row+1,"cells":cells})
                        else: narrow=True
                        count+=1
        return {"format":FORMAT_OLE_BIFF,"sheets":fingerprint},{"format":FORMAT_OLE_BIFF,"sheet_count":book.nsheets,"sheets":sheets,"schema_neighborhood":samples,"SCHEMA_NEIGHBORHOOD_REQUIRES_NARROWER_PROBE":narrow}
    except V9005StageABlocked: raise
    except Exception: _fail()

def profile_verified_lock(lock: Any) -> dict[str, Any]:
    item=_validate(lock); fmt=detect_container_format(item.raw_bytes)
    if fmt == FORMAT_HTML: structure,evidence=_html_structure(item.raw_bytes); status="PROFILED"
    elif fmt == FORMAT_OLE_BIFF: structure,evidence=_ole_structure(item.raw_bytes); status="PROFILED"
    else: structure={"format":fmt}; evidence={}; status=FORMAT_REQUIRES_FOLLOWUP
    result={"status":status,"source_family":item.source_family,"object_domain":item.object_domain.value,"applicable_period":item.applicable_period,"source_object_slot_id":item.source_object_slot_id,"sha256":item.sha256,"byte_length":len(item.raw_bytes),"container_format":fmt,"structural_profile_sha256":sha256(_canonical_json(structure).encode()).hexdigest(),"structural_evidence":evidence}
    return _validate_safe_profile(result)

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


@dataclass(frozen=True)
class Phase1SchemaDiscoveryResult:
    """Safe Phase-1 aggregate result; raw locked bytes never escape."""

    evidence_slot_ids: tuple[str, ...]
    safe_profiles: tuple[dict[str, Any], ...]
    representative_safe_profiles: tuple[dict[str, Any], ...]
    network_attempt_count: int


@dataclass(frozen=True)
class _ExpectedEvidence:
    slot_id: str
    family: str
    period: str
    domain: ObjectDomain


def _phase1_slot(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail()
    return value


def _phase1_attempts(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail()
    return value


def _phase1_single_reference(value: Any) -> str:
    if not isinstance(value, tuple) or len(value) != 1:
        _fail()
    return _phase1_slot(value[0])


def _phase1_f3_expected(value: Any) -> list[_ExpectedEvidence]:
    references = getattr(value, "base_coverage_references", None)
    _phase1_attempts(getattr(value, "network_attempt_count", None))
    months = inventory_months()
    family = SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE
    expected_keys = {(family, month) for month in months}
    if not isinstance(references, Mapping) or set(references) != expected_keys:
        _fail()
    evidence: list[_ExpectedEvidence] = []
    for year in range(2017, 2026):
        identifiers = [_phase1_single_reference(references[(family, f"{year}-{month:02d}")]) for month in range(1, 13)]
        if len(set(identifiers)) != 1:
            _fail()
        evidence.append(_ExpectedEvidence(identifiers[0], family, str(year), ObjectDomain.YEAR))
    if len({item.slot_id for item in evidence}) != 9:
        _fail()
    return evidence


def _phase1_f7_expected(value: Any) -> list[_ExpectedEvidence]:
    base_references = getattr(value, "base_coverage_references", None)
    extra_references = getattr(value, "envelope_extra_references", None)
    _phase1_attempts(getattr(value, "network_attempt_count", None))
    family = SOURCE_FAMILY_JPX_CALENDAR
    months, extras = inventory_months(), calendar_envelope_extra_months()
    if (
        not isinstance(base_references, Mapping)
        or not isinstance(extra_references, Mapping)
        or set(base_references) != {(family, month) for month in months}
        or set(extra_references) != set(extras)
    ):
        _fail()
    return (
        [_ExpectedEvidence(_phase1_single_reference(base_references[(family, month)]), family, month, ObjectDomain.BASE) for month in months]
        + [_ExpectedEvidence(_phase1_single_reference(extra_references[month]), family, month, ObjectDomain.ENVELOPE_EXTRA) for month in extras]
    )


def _phase1_profile_expected(output_root: Any, expected: _ExpectedEvidence) -> dict[str, Any]:
    try:
        locked = read_locked_payload_by_slot_id(output_root, expected.slot_id)
        required = {
            "schema_version", "source_family", "applicable_period", "requested_url",
            "resolved_url", "http_status", "retrieval_timestamp_utc", "byte_length",
            "sha256", "raw",
        }
        if not isinstance(locked, dict) or set(locked) != required:
            _fail()
        if locked["source_family"] != expected.family or locked["applicable_period"] != expected.period:
            _fail()
        item = VerifiedLockedObject(
            locked["schema_version"], locked["source_family"], locked["applicable_period"],
            locked["requested_url"], locked["resolved_url"], locked["http_status"],
            locked["retrieval_timestamp_utc"], locked["byte_length"], expected.slot_id,
            locked["sha256"], locked["raw"], expected.domain,
        )
        return profile_verified_lock(item)
    except V9005StageABlocked:
        raise
    except Exception as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc


def run_phase1_schema_discovery_core(output_root: Any, *, fetcher: Any, sleep: Any, clock: Any) -> Phase1SchemaDiscoveryResult:
    """Acquire/profile only the frozen Phase-1 evidence inventory via injected I/O."""
    try:
        expected: list[_ExpectedEvidence] = []
        f1 = acquire_f1_terminal_evidence(output_root, fetcher=fetcher, sleep=sleep, clock=clock)
        if not isinstance(f1, tuple) or len(f1) != 2:
            _fail()
        f1_slot, attempts = _phase1_slot(f1[0]), _phase1_attempts(f1[1])
        expected.append(_ExpectedEvidence(f1_slot, SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, TERMINAL_PERIOD, ObjectDomain.TERMINAL))
        for month in inventory_months():
            for family in (SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE):
                acquired = acquire_f2_f4_monthly_evidence(output_root, source_family=family, requested_month=month, fetcher=fetcher, sleep=sleep, clock=clock)
                if not isinstance(acquired, tuple) or len(acquired) != 2:
                    _fail()
                slot_id, count = _phase1_slot(acquired[0]), _phase1_attempts(acquired[1])
                expected.append(_ExpectedEvidence(slot_id, family, month, ObjectDomain.BASE))
                attempts += count
        f3 = acquire_f3_required_slots(output_root, fetcher=fetcher, sleep=sleep, clock=clock)
        f3_expected = _phase1_f3_expected(f3)
        expected.extend(f3_expected)
        attempts += _phase1_attempts(getattr(f3, "network_attempt_count", None))
        f7 = acquire_f7_required_slots(output_root, fetcher=fetcher, sleep=sleep, clock=clock)
        f7_expected = _phase1_f7_expected(f7)
        expected.extend(f7_expected)
        attempts += _phase1_attempts(getattr(f7, "network_attempt_count", None))
        if len(expected) != 341 or len({item.slot_id for item in expected}) != 341:
            _fail()
        profiles = tuple(_phase1_profile_expected(output_root, item) for item in expected)
        if len(profiles) != 341 or len({item["source_object_slot_id"] for item in profiles}) != 341:
            _fail()
        return Phase1SchemaDiscoveryResult(
            tuple(item.slot_id for item in expected), profiles,
            tuple(select_representatives(profiles)), attempts,
        )
    except V9005StageABlocked:
        raise
    except Exception as exc:
        raise V9005StageABlocked(IMPLEMENTATION_FAILURE) from exc

def prepare_future_acquisition() -> None:
    """Gate only: aggregate acquisition remains deliberately unimplemented."""
    raise V9005StageABlocked(CHATGPT_DECISION_REQUIRED)
