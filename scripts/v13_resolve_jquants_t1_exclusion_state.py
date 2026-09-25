"""Strict, streaming T1-only reader for the reviewed recovery artifact."""

from __future__ import annotations

import codecs
import hashlib
import json
import re
from pathlib import Path
from typing import BinaryIO, Callable

from scripts.v13_resolve_v8_t1_identity_state import _write_once

SCHEMA = "V8_JQUANTS_IDENTITY_RECOVERY_MANIFEST_V1"
CONTRACT = "V13_V8_JQUANTS_IDENTITY_RECOVERY_DESIGN"
T1_SHA = "262201792183776e3bead4638646ee949c05d35c894c7a4053556befa6230e1d"
ELIGIBLE_SHA = "37630f8f754c1a1f0f3e07f0ffc26711c83e635b5eaf24533659f37970263405"
SOURCE_COMMIT = "7565ca723c76801d74d8d319d65d280a689b3cfa"
SOURCE_BLOB = "f46ea0c304b0bbd2d230b9850acba9eada9f6908"
EXPECTED = {
    "schema": SCHEMA, "contract": CONTRACT, "query_date": "20260731",
    "effective_date": "2026-07-31", "canonical_order": "SHA256_UTF8_CODE_THEN_CODE_ASC",
    "block_size": 300, "eligible_count": 3110, "eligible_sha256": ELIGIBLE_SHA,
    "source_commit": SOURCE_COMMIT, "source_blob": SOURCE_BLOB,
}
ROOT_KEYS = frozenset({"schema", "contract", "endpoint", "query_date", "effective_date",
    "markets", "products", "canonical_order", "block_size", "raw_schema",
    "raw_manifest_sha256", "raw_pages", "eligible_count", "eligible_sha256",
    "block_sizes", "block_hashes", "assignments", "source_commit", "source_blob",
    "recovery_timestamp_utc", "sha256"})
BLOCK_KEYS = frozenset({"T0", "T1", "T2", "T3", "T_spare"})
ASSIGNMENT_KEYS = BLOCK_KEYS | {"eligible"}
CODE = re.compile(r"[0-9A-Z]{4}\Z")


class Block(RuntimeError):
    def __init__(self, reason: str = "RECOVERY_INVALID_OR_AMBIGUOUS") -> None:
        super().__init__(reason)
        self.reason = reason


class Scanner:
    """One-pass JSON syntax scanner. Skipped values never become Python values."""

    def __init__(self, stream: BinaryIO, first: bytes):
        self.stream = stream
        self.look = first
        self.bytes_read = 1

    def peek(self) -> bytes:
        if self.look is None:
            self.look = self.stream.read(1)
            self.bytes_read += len(self.look)
        return self.look

    def take(self) -> bytes:
        value = self.peek()
        self.look = None
        return value

    def ws(self) -> None:
        while self.peek() in (b" ", b"\t", b"\r", b"\n"):
            self.take()

    def expect(self, byte: bytes) -> None:
        self.ws()
        if self.take() != byte:
            raise Block()

    def string(self, collect: bool = False) -> str | None:
        self.expect(b'"')
        chars: list[str] | None = [] if collect else None
        collected_bytes = 0
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        while True:
            b = self.take()
            if not b:
                raise Block()
            if chars is not None:
                collected_bytes += 1
                if collected_bytes > 256:
                    raise Block()
            if b == b'"':
                try:
                    tail = decoder.decode(b"", final=True)
                except UnicodeError:
                    raise Block() from None
                if chars is not None:
                    chars.append(tail)
                    return "".join(chars)
                return None
            if b == b"\\":
                try:
                    decoder.decode(b"", final=True)
                except UnicodeError:
                    raise Block() from None
                decoder = codecs.getincrementaldecoder("utf-8")("strict")
                esc = self.take()
                mapping = {b'"': '"', b"\\": "\\", b"/": "/", b"b": "\b", b"f": "\f", b"n": "\n", b"r": "\r", b"t": "\t"}
                if esc == b"u":
                    digits = b"".join(self.take() for _ in range(4))
                    if len(digits) != 4 or any(c not in b"0123456789abcdefABCDEF" for c in digits):
                        raise Block()
                    point = int(digits, 16)
                    if 0xD800 <= point <= 0xDBFF:
                        if self.take() != b"\\" or self.take() != b"u":
                            raise Block()
                        low = b"".join(self.take() for _ in range(4))
                        if len(low) != 4 or any(c not in b"0123456789abcdefABCDEF" for c in low):
                            raise Block()
                        low_point = int(low, 16)
                        if not 0xDC00 <= low_point <= 0xDFFF:
                            raise Block()
                        point = 0x10000 + (point - 0xD800) * 1024 + low_point - 0xDC00
                    elif 0xDC00 <= point <= 0xDFFF:
                        raise Block()
                    value = chr(point)
                elif esc in mapping:
                    value = mapping[esc]
                else:
                    raise Block()
                if chars is not None:
                    chars.append(value)
            else:
                if b[0] < 32:
                    raise Block()
                try:
                    decoded = decoder.decode(b)
                except UnicodeError:
                    raise Block() from None
                if chars is not None and decoded:
                    chars.append(decoded)

    def number(self) -> int:
        self.ws()
        digits = bytearray()
        if self.peek() == b"-":
            digits.extend(self.take())
        first = self.take()
        if first == b"0":
            digits.extend(first)
            if self.peek() and self.peek()[0] in b"0123456789":
                raise Block()
        elif first and first[0] in b"123456789":
            digits.extend(first)
            while self.peek() and self.peek()[0] in b"0123456789":
                digits.extend(self.take())
        else:
            raise Block()
        if self.peek() in (b".", b"e", b"E"):
            raise Block()
        if len(digits) > 20:
            raise Block()
        return int(digits)

    def skip(self, depth: int = 0) -> None:
        if depth > 32:
            raise Block()
        self.ws()
        c = self.peek()
        if c == b'"':
            self.string()
        elif c == b"[":
            self.take(); self.ws()
            if self.peek() == b"]":
                self.take(); return
            while True:
                self.skip(depth + 1); self.ws()
                if self.peek() == b"]":
                    self.take(); return
                self.expect(b",")
        elif c == b"{":
            self.take(); self.ws()
            if self.peek() == b"}":
                self.take(); return
            while True:
                self.string(); self.expect(b":"); self.skip(depth + 1); self.ws()
                if self.peek() == b"}":
                    self.take(); return
                self.expect(b",")
        elif c in (b"t", b"f", b"n"):
            token = {b"t": b"true", b"f": b"false", b"n": b"null"}[c]
            for byte in token:
                if self.take() != bytes((byte,)):
                    raise Block()
        else:
            self.number()

    def obj(self, allowed: frozenset[str], selected: dict[str, str], *, depth: int = 0) -> dict:
        self.expect(b"{")
        values: dict = {}
        seen: set[str] = set()
        self.ws()
        if self.peek() == b"}":
            self.take()
            raise Block()
        while True:
            key = self.string(True)
            if key not in allowed or key in seen:
                raise Block()
            seen.add(key)
            self.expect(b":")
            mode = selected.get(key)
            if mode == "string": values[key] = self.string(True)
            elif mode == "integer": values[key] = self.number()
            elif mode == "t1": values[key] = self.t1()
            elif mode == "sizes": values[key] = self.obj(BLOCK_KEYS, {"T1": "integer"}, depth=depth+1)["T1"]
            elif mode == "hashes": values[key] = self.obj(BLOCK_KEYS, {"T1": "string"}, depth=depth+1)["T1"]
            elif mode == "assignments": values[key] = self.obj(ASSIGNMENT_KEYS, {"T1": "t1"}, depth=depth+1)["T1"]
            else: self.skip(depth + 1)
            self.ws()
            if self.peek() == b"}":
                self.take(); break
            self.expect(b",")
        if seen != allowed:
            raise Block()
        return values

    def t1(self) -> list[str]:
        self.expect(b"[")
        members: list[str] = []
        self.ws()
        if self.peek() == b"]":
            self.take(); return members
        while True:
            value = self.string(True)
            if value is None or CODE.fullmatch(value) is None:
                raise Block("T1_CODE_INVALID")
            members.append(value)
            if len(members) > 300:
                raise Block("T1_COUNT_MISMATCH")
            self.ws()
            if self.peek() == b"]":
                self.take(); return members
            self.expect(b",")


def resolve(stream: BinaryIO, first: bytes, state_path: Path) -> dict:
    scanner = Scanner(stream, first)
    selected = {key: "integer" if isinstance(value, int) else "string" for key, value in EXPECTED.items()}
    selected.update({"block_sizes": "sizes", "block_hashes": "hashes", "assignments": "assignments"})
    values = scanner.obj(ROOT_KEYS, selected)
    scanner.ws()
    if scanner.peek():
        raise Block()
    for key, expected in EXPECTED.items():
        if type(values.get(key)) is not type(expected) or values[key] != expected:
            raise Block("SOURCE_BINDING_MISMATCH")
    members = values["assignments"]
    if values["block_sizes"] != 300 or len(members) != 300 or len(set(members)) != 300:
        raise Block("T1_COUNT_MISMATCH")
    actual = hashlib.sha256(("\n".join(members) + "\n").encode("ascii")).hexdigest()
    if values["block_hashes"] != T1_SHA or actual != T1_SHA:
        raise Block("T1_HASH_MISMATCH")
    state = {"schema": "V13_JQUANTS_T1_EXCLUSION_STATE_V1", "source_schema": SCHEMA,
        "source_contract": CONTRACT, "source_commit": SOURCE_COMMIT, "source_blob": SOURCE_BLOB,
        "eligible_count": 3110, "eligible_sha256": ELIGIBLE_SHA,
        "t1_ticker_list_sha256": actual, "t1_count": 300,
        "known_definitely_acquired_prefix_count": 297,
        "exclusion_disposition": "EXCLUDE_FULL_RECOVERED_T1_BLOCK", "t1_membership": members}
    payload = (json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    _write_once(state_path, payload)
    return {"t1_count": 300, "t1_hash_match": True, "binding_match": True,
            "state_written": True, "source_bytes_read": scanner.bytes_read}
