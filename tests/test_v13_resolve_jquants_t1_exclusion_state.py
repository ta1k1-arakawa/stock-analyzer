"""Synthetic-only narrow scanner checks."""

import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts import v13_resolve_jquants_t1_exclusion_state as resolver


MEMBERS = [f"{i:04d}" for i in range(300)]
SYNTHETIC_SHA = hashlib.sha256(("\n".join(MEMBERS) + "\n").encode()).hexdigest()
SENTINEL = "NON_T1_SENTINEL_PRIVATE_IDENTITY"


def artifact():
    value = {key: expected for key, expected in resolver.EXPECTED.items()}
    value.update({key: "public-synthetic" for key in resolver.ROOT_KEYS - set(value) -
                  {"block_sizes", "block_hashes", "assignments", "raw_pages", "markets", "products"}})
    value["block_sizes"] = {key: 300 for key in resolver.BLOCK_KEYS}
    value["block_hashes"] = {key: SYNTHETIC_SHA for key in resolver.BLOCK_KEYS}
    value["assignments"] = {key: [SENTINEL] for key in resolver.ASSIGNMENT_KEYS}
    value["assignments"]["T1"] = MEMBERS
    value["raw_pages"] = [{"opaque": SENTINEL}]
    value["markets"] = ["0111", "0112"]
    value["products"] = ["011"]
    return value


def raw(value):
    return json.dumps(value, separators=(",", ":")).encode()


def check(data, tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "T1_SHA", SYNTHETIC_SHA)
    with io.BytesIO(data) as stream:
        first = stream.read(1)
        return resolver.resolve(stream, first, tmp_path / "state.json")


def test_valid_retains_only_t1(tmp_path, monkeypatch):
    result = check(raw(artifact()), tmp_path, monkeypatch)
    state_raw = (tmp_path / "state.json").read_text()
    state = json.loads(state_raw)
    assert result["source_bytes_read"] == len(raw(artifact()))
    assert state["t1_membership"] == MEMBERS
    assert SENTINEL not in state_raw
    assert "eligible" not in state or "eligible_count" in state


@pytest.mark.parametrize("key,bad", [
    ("schema", "wrong"), ("contract", "wrong"), ("query_date", "wrong"),
    ("effective_date", "wrong"), ("source_commit", "wrong"),
    ("source_blob", "wrong"), ("eligible_count", 3000),
    ("eligible_sha256", "wrong"), ("canonical_order", "wrong"),
    ("block_size", 301),
])
def test_source_bindings_block(key, bad, tmp_path, monkeypatch):
    value = artifact(); value[key] = bad
    with pytest.raises(resolver.Block):
        check(raw(value), tmp_path, monkeypatch)
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("kind", ["count", "hash", "actual_count", "actual_hash", "duplicate", "invalid"])
def test_t1_fail_closed(kind, tmp_path, monkeypatch):
    value = artifact()
    if kind == "count": value["block_sizes"]["T1"] = 299
    if kind == "hash": value["block_hashes"]["T1"] = "bad"
    if kind == "actual_count": value["assignments"]["T1"] = MEMBERS[:-1]
    if kind == "actual_hash": value["assignments"]["T1"][0] = "9999"
    if kind == "duplicate": value["assignments"]["T1"][1] = MEMBERS[0]
    if kind == "invalid": value["assignments"]["T1"][0] = "BAD!"
    with pytest.raises(resolver.Block):
        check(raw(value), tmp_path, monkeypatch)
    assert not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("field", ["schema", "assignments", "T1"])
def test_duplicate_relevant_keys_block(field, tmp_path, monkeypatch):
    text = raw(artifact()).decode()
    if field == "schema":
        text = text.replace('"schema":', '"schema":"wrong","schema":', 1)
    elif field == "assignments":
        text = text.replace('"assignments":', '"assignments":{},"assignments":', 1)
    else:
        text = text.replace('"T1":', '"T1":[],"T1":', 1)
    with pytest.raises(resolver.Block):
        check(text.encode(), tmp_path, monkeypatch)


def test_non_t1_sentinel_not_in_error(tmp_path, monkeypatch):
    value = artifact(); value["schema"] = "wrong"
    with pytest.raises(resolver.Block) as exc:
        check(raw(value), tmp_path, monkeypatch)
    assert SENTINEL not in str(exc.value)


def test_malformed_skipped_json_blocks(tmp_path, monkeypatch):
    text = raw(artifact()).replace(SENTINEL.encode(), b"\\uD800", 1)
    with pytest.raises(resolver.Block):
        check(text, tmp_path, monkeypatch)
