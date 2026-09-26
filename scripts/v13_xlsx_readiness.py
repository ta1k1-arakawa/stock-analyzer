"""Offline operational probe for the production JPX XLSX parser route.

The synthetic workbook is generated in memory with standard-library ZIP/XML.
This module never opens protected data or performs transport.
"""

from __future__ import annotations

import io
import zipfile


def synthetic_jpx_xlsx() -> bytes:
    parts = {
        "[Content_Types].xml": b'''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>''',
        "_rels/.rels": b'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>''',
        "xl/workbook.xml": b'''<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Listed" sheetId="1" r:id="rId1"/></sheets></workbook>''',
        "xl/_rels/workbook.xml.rels": b'''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>''',
        "xl/worksheets/sheet1.xml": '''<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>
<row r="1"><c r="A1" t="inlineStr"><is><t>コード</t></is></c><c r="B1" t="inlineStr"><is><t>市場・区分</t></is></c><c r="C1" t="inlineStr"><is><t>33業種区分</t></is></c></row>
<row r="2"><c r="A2" t="inlineStr"><is><t>1000</t></is></c><c r="B2" t="inlineStr"><is><t>Prime Domestic Stocks</t></is></c><c r="C2" t="inlineStr"><is><t>Machinery</t></is></c></row>
</sheetData></worksheet>'''.encode("utf-8"),
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return stream.getvalue()


def probe_production_xlsx_route() -> bool:
    from src.v13_public_data_lock import RawLock, parse_jpx

    return parse_jpx(RawLock.from_bytes(synthetic_jpx_xlsx())) == {"1000": "Machinery"}


if __name__ == "__main__":
    try:
        passed = probe_production_xlsx_route()
    except Exception:
        passed = False
    print(f"JPX_XLSX_READINESS_PASS={str(passed).lower()}")
    raise SystemExit(0 if passed else 1)
