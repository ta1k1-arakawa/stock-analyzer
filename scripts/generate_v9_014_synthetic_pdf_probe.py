"""Generate the committed synthetic PDF operational-readiness fixture for
V9_014's PDF real-execution environment successor (Stage E2).

Writes `tests/fixtures/v9_014_synthetic_pdf_env_probe.pdf`: a completely
artificial, hand-constructed single-page PDF used only to prove, in a
future reviewed checkpoint, that the pinned `pdfplumber==0.11.10` build
actually opens a PDF and extracts an exact, predetermined page/text/table
structure from it. It exists solely for environment-readiness, per
`V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_DESIGN.md` Section 5a and
the frozen `V9_014_SOURCE_B_PDF_STRUCTURAL_CALIBRATION_METHOD_CONTRACT.md`.

The fixture contains NO JPX names, NO ticker identities, NO dates, NO
prices, and NO trading outcomes of any kind. It is a single visibly
artificial 3x2 ruled table:

    SYNTHETIC_KEY | SYNTHETIC_VALUE
    ALPHA         | 11
    BETA          | 22

This generator uses the Python standard library ONLY -- no `reportlab`, no
new dependency of any kind. It builds the PDF's classic object structure
(catalog / page tree / page / content stream / standard Type1 Helvetica
font, no font embedding, no `/CreationDate`, no randomness of any kind)
directly in Python, so the output is byte-for-byte deterministic across
runs: unlike the legacy `.xls` fixture generator (which relies on the
third-party `xlwt` library and explicitly does NOT claim byte determinism,
see `scripts/generate_synthetic_jpx_xls_fixture.py`), this generator owns
every byte it writes and DOES claim exact reproducibility. `--check`
verifies that a freshly rebuilt PDF is byte-identical to the committed
fixture, not merely that it "happens to match".

This generator performs NO network access and reads no private/protected
data. It does not read, resolve, or use the real `pdfplumber` package at
all -- it emits raw PDF bytes only. Whether the pinned `pdfplumber==0.11.10`
actually extracts the exact predetermined text/table this fixture is
designed to produce is NOT verified by this generator, and is NOT verified
anywhere in V9_014 Stage E2: that mechanical proof occurs only at the
later, separately reviewed Stage E6 (staging environment) and Stage
E10/E14 (live canonical environment) checkpoints defined in
`V9_014_PDF_REAL_EXECUTION_ENVIRONMENT_SUCCESSOR_DESIGN.md`. This fixture
is deliberately built as simply as possible -- an explicit vector-line
table grid, standard non-overlapping text placement, generous inter-word
gaps -- to make correct extraction likely, but that likelihood is not
itself a claim of proof.

Run it only to regenerate or inspect the fixture:

    python3 scripts/generate_v9_014_synthetic_pdf_probe.py
    python3 scripts/generate_v9_014_synthetic_pdf_probe.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "v9_014_synthetic_pdf_env_probe.pdf"

# The single-page table this fixture renders. Every cell value is drawn
# from an unmistakably artificial namespace (SYNTHETIC_* headers, a NATO-
# alphabet row label, small placeholder integers) -- never a real JPX
# name, ticker code, date, price, or trading outcome.
TABLE_HEADER: tuple[str, str] = ("SYNTHETIC_KEY", "SYNTHETIC_VALUE")
TABLE_ROWS: tuple[tuple[str, str], ...] = (
    ("ALPHA", "11"),
    ("BETA", "22"),
)
EXPECTED_TABLE: list[list[str]] = [list(TABLE_HEADER)] + [list(row) for row in TABLE_ROWS]

# Page geometry (PDF points; origin bottom-left). A compact 200x120 page
# with a 2-column x 3-row grid, 80pt-wide columns and 30pt-tall rows.
PAGE_WIDTH = 200
PAGE_HEIGHT = 120
GRID_X = (20, 100, 180)  # 3 vertical line positions -> 2 columns
GRID_Y = (20, 50, 80, 110)  # 4 horizontal line positions -> 3 rows
FONT_SIZE = 10
# Text baselines sit 7pt above each row's bottom grid line, comfortably
# inside the row band and far from the ruled lines on every side.
TEXT_CELLS: tuple[tuple[int, int, str], ...] = (
    (25, 87, TABLE_HEADER[0]),
    (105, 87, TABLE_HEADER[1]),
    (25, 57, TABLE_ROWS[0][0]),
    (105, 57, TABLE_ROWS[0][1]),
    (25, 27, TABLE_ROWS[1][0]),
    (105, 27, TABLE_ROWS[1][1]),
)

# The predetermined text this fixture is designed to yield from
# `page.extract_text(x_tolerance=3, y_tolerance=3)`: each row's two cells
# joined by a single space, rows separated by a newline, top row first.
EXPECTED_TEXT = "SYNTHETIC_KEY SYNTHETIC_VALUE\nALPHA 11\nBETA 22"

# Recorded canonical identity of the committed fixture bytes. Because this
# generator is fully self-contained (no third-party library, no
# timestamps, no randomness), `--check` treats a mismatch between a
# freshly rebuilt PDF and these committed bytes as a hard failure, not
# merely a report.
EXPECTED_FIXTURE_SHA256 = "b02ac3773514eb749f031890c1d1fe449d1cf522d1e62af185b50e16516f5a23"


def _content_stream_bytes() -> bytes:
    """Build the page content stream: a stroked line grid plus six
    left-aligned text-showing operations, one per cell. Deterministic,
    ASCII-only, no comments.
    """
    lines: list[bytes] = [b"1 w"]
    for y in GRID_Y:
        lines.append(f"{GRID_X[0]} {y} m {GRID_X[-1]} {y} l S".encode("ascii"))
    for x in GRID_X:
        lines.append(f"{x} {GRID_Y[0]} m {x} {GRID_Y[-1]} l S".encode("ascii"))
    for x, y, text in TEXT_CELLS:
        if not text.isascii() or "(" in text or ")" in text or "\\" in text:
            raise AssertionError(f"SYNTHETIC_FIXTURE_CELL_TEXT_UNSAFE_FOR_LITERAL_ENCODING: {text!r}")
        lines.append(b"BT")
        lines.append(f"/F1 {FONT_SIZE} Tf".encode("ascii"))
        lines.append(f"{x} {y} Td".encode("ascii"))
        lines.append(b"(" + text.encode("ascii") + b") Tj")
        lines.append(b"ET")
    return b"\n".join(lines) + b"\n"


def build_pdf_bytes() -> bytes:
    """Build the complete single-page PDF in memory, byte-for-byte
    deterministic: a classic (non-linearized) object structure with a
    plain xref table, no `/CreationDate`, `/ID`, `/Producer`, or any other
    field whose value could vary run-to-run.
    """
    content = _content_stream_bytes()

    buf = bytearray()
    buf += b"%PDF-1.4\n"
    offsets: dict[int, int] = {}

    def add_object(number: int, body: bytes) -> None:
        offsets[number] = len(buf)
        buf.extend(f"{number} 0 obj\n".encode("ascii"))
        buf.extend(body)
        if not body.endswith(b"\n"):
            buf.extend(b"\n")
        buf.extend(b"endobj\n")

    add_object(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    add_object(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    add_object(
        3,
        (
            b"<< /Type /Page /Parent 2 0 R "
            + f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] ".encode("ascii")
            + b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
    )
    stream_body = f"<< /Length {len(content)} >>\nstream\n".encode("ascii") + content + b"endstream"
    add_object(4, stream_body)
    add_object(5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")

    # Each 20-byte xref entry ends with a 2-byte EOL. Per ISO 32000-1
    # SS7.5.4 that EOL may be "SP CR", "SP LF", or "CR LF" -- this uses
    # the common "SP LF" convention. Every one of the three spec-legal
    # choices ends the fixed-width entry in a byte (a space, or a bare
    # CR before the final LF) that line-oriented text tooling flags as
    # "trailing whitespace"; this is an unavoidable property of the
    # mandatory PDF cross-reference table's fixed-width 20-byte-per-entry
    # format (not accidental repository whitespace), documented at the
    # call site in this generator's `--check` output and in this task's
    # own final report rather than "fixed" by deviating from spec.
    object_count = len(offsets)
    xref_offset = len(buf)
    buf.extend(f"xref\n0 {object_count + 1}\n".encode("ascii"))
    buf.extend(b"0000000000 65535 f \n")
    for number in range(1, object_count + 1):
        buf.extend(f"{offsets[number]:010d} 00000 n \n".encode("ascii"))
    buf.extend(b"trailer\n")
    buf.extend(f"<< /Size {object_count + 1} /Root 1 0 R >>\n".encode("ascii"))
    buf.extend(b"startxref\n")
    buf.extend(f"{xref_offset}\n".encode("ascii"))
    buf.extend(b"%%EOF")
    return bytes(buf)


def write_fixture() -> tuple[Path, str]:
    payload = build_pdf_bytes()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_bytes(payload)
    return FIXTURE_PATH, hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write. Verify the committed fixture's SHA-256 against the recorded "
            "canonical value, and that a freshly rebuilt PDF is byte-identical to the "
            "committed bytes. Exits nonzero on any mismatch."
        ),
    )
    args = parser.parse_args()

    if args.check:
        if not FIXTURE_PATH.exists():
            print(f"committed_fixture_present=false path={FIXTURE_PATH}")
            return 1
        committed = FIXTURE_PATH.read_bytes()
        committed_digest = hashlib.sha256(committed).hexdigest()
        rebuilt = build_pdf_bytes()
        rebuilt_digest = hashlib.sha256(rebuilt).hexdigest()
        committed_matches_expected = committed_digest == EXPECTED_FIXTURE_SHA256
        rebuild_matches_committed = rebuilt == committed
        print(f"committed_fixture_present=true path={FIXTURE_PATH}")
        print(f"committed_fixture_sha256={committed_digest}")
        print(f"expected_fixture_sha256={EXPECTED_FIXTURE_SHA256}")
        print(f"committed_matches_expected={str(committed_matches_expected).lower()}")
        print(f"rebuilt_fixture_sha256={rebuilt_digest}")
        print(f"rebuild_matched_committed_bytes={str(rebuild_matches_committed).lower()}")
        print("byte_determinism_claimed=true")
        print("canonical_identity=COMMITTED_FIXTURE_SHA256")
        print("contains_jpx_names_or_ticker_identities=false")
        print("contains_dates_prices_or_trading_outcomes=false")
        print("network_requests=0")
        print("private_reads=0")
        print("real_pdfplumber_imported=false")
        if not (committed_matches_expected and rebuild_matches_committed):
            return 1
        return 0

    path, digest = write_fixture()
    print(f"fixture_written={path}")
    print(f"fixture_sha256={digest}")
    print(f"expected_fixture_sha256={EXPECTED_FIXTURE_SHA256}")
    print(f"matches_recorded_canonical_sha256={str(digest == EXPECTED_FIXTURE_SHA256).lower()}")
    print(f"page_count=1")
    print(f"table_rows={len(EXPECTED_TABLE)}")
    print(f"table_columns={len(TABLE_HEADER)}")
    print("contains_jpx_names_or_ticker_identities=false")
    print("contains_dates_prices_or_trading_outcomes=false")
    print("network_requests=0")
    print("private_reads=0")
    print("real_pdfplumber_imported=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
