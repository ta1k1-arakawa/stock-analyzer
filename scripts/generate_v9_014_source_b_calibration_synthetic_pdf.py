"""Generate the deterministic synthetic PDF for the V9_014 Stage C1 probe.

The fixture is a wholly artificial one-page ruled layout.  It contains no
JPX name, ticker, date, price, or trading outcome.  Its text deliberately
contains artificial decimal digits and the exact literal ``thous.shs.`` so
the offline probe's outcome-safe glyph representation can be tested.  This
generator uses only the Python standard library and owns every emitted byte.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "v9_014_source_b_calibration_synthetic.pdf"

PAGE_WIDTH = 420
PAGE_HEIGHT = 180
GRID_X = (20, 210, 400)
GRID_Y = (20, 70, 120, 160)
FONT_SIZE = 10
TEXT_CELLS: tuple[tuple[int, int, str], ...] = (
    (35, 137, "SYNTHETIC_CALIBRATION"),
    (225, 137, "DIGITS 4821"),
    (35, 87, "thous.shs."),
    (225, 87, "ARTIFICIAL_UNIT"),
    (35, 37, "NO_REAL_MARKET"),
    (225, 37, "GLYPH_TEST_7"),
)

# Filled after the first deterministic generation; --check treats this as a
# hard frozen identity, not as an informational digest.
EXPECTED_FIXTURE_SHA256 = "9e1b685b3415df73404afb001fcc77124a9ac665423f32dbcb566cb5e3b8e00d"


def _content_stream_bytes() -> bytes:
    lines: list[bytes] = [b"1 w"]
    lines.append(f"{GRID_X[0]} {GRID_Y[0]} {GRID_X[-1] - GRID_X[0]} {GRID_Y[-1] - GRID_Y[0]} re S".encode("ascii"))
    for y in GRID_Y[1:-1]:
        lines.append(f"{GRID_X[0]} {y} m {GRID_X[-1]} {y} l S".encode("ascii"))
    lines.append(f"{GRID_X[1]} {GRID_Y[0]} m {GRID_X[1]} {GRID_Y[-1]} l S".encode("ascii"))
    for x, y, text in TEXT_CELLS:
        if not text.isascii() or any(character in text for character in "()\\"):
            raise AssertionError("synthetic fixture text is unsafe for literal encoding")
        lines.extend(
            (
                b"BT",
                f"/F1 {FONT_SIZE} Tf".encode("ascii"),
                f"{x} {y} Td".encode("ascii"),
                b"(" + text.encode("ascii") + b") Tj",
                b"ET",
            )
        )
    return b"\n".join(lines) + b"\n"


def build_pdf_bytes() -> bytes:
    content = _content_stream_bytes()
    payload = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}

    def add_object(number: int, body: bytes) -> None:
        offsets[number] = len(payload)
        payload.extend(f"{number} 0 obj\n".encode("ascii"))
        payload.extend(body)
        if not body.endswith(b"\n"):
            payload.extend(b"\n")
        payload.extend(b"endobj\n")

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

    xref_offset = len(payload)
    object_count = len(offsets) + 1

    def xref_entry(entry_type: int, field_two: int, field_three: int) -> bytes:
        return (
            bytes((entry_type,))
            + field_two.to_bytes(4, "big")
            + field_three.to_bytes(2, "big")
        )

    xref_data = b"".join(
        [xref_entry(0, 0, 65535)]
        + [xref_entry(1, offsets[number], 0) for number in range(1, len(offsets) + 1)]
        + [xref_entry(1, xref_offset, 0)]
    )
    xref_body = (
        f"<< /Type /XRef /Size {object_count + 1} /Index [0 {object_count + 1}] "
        f"/W [1 4 2] /Root 1 0 R /Length {len(xref_data)} >>\nstream\n".encode("ascii")
        + xref_data
        + b"\nendstream"
    )
    add_object(len(offsets) + 1, xref_body)
    payload.extend(b"startxref\n")
    payload.extend(f"{xref_offset}\n".encode("ascii"))
    payload.extend(b"%%EOF")
    return bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify frozen bytes without writing")
    arguments = parser.parse_args()
    rebuilt = build_pdf_bytes()
    rebuilt_digest = hashlib.sha256(rebuilt).hexdigest()

    if arguments.check:
        if not FIXTURE_PATH.exists():
            print("fixture_present=false")
            return 1
        committed = FIXTURE_PATH.read_bytes()
        committed_digest = hashlib.sha256(committed).hexdigest()
        matches = committed == rebuilt and committed_digest == EXPECTED_FIXTURE_SHA256
        print("fixture_present=true")
        print(f"fixture_sha256={committed_digest}")
        print(f"rebuilt_sha256={rebuilt_digest}")
        print(f"frozen_sha256={EXPECTED_FIXTURE_SHA256}")
        print(f"byte_determinism_pass={str(matches).lower()}")
        print("network_requests=0")
        print("private_reads=0")
        return 0 if matches else 1

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_bytes(rebuilt)
    print(f"fixture_written={FIXTURE_PATH}")
    print(f"fixture_sha256={rebuilt_digest}")
    print("network_requests=0")
    print("private_reads=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
