import json
import socket
import ssl
import urllib.error
from pathlib import Path

import pytest

from src import v8_partition as historical
from src import v8_jquants_identity_recovery as jq


COMMIT = "a" * 40
BLOB = "b" * 40


def page(codes=(), token=None, *, extra=None):
    value = {"data": [{"Date": jq.EFFECTIVE_DATE, "Mkt": "0111", "ProdCat": "011",
                       "Code": code + "0"} for code in codes]}
    if token is not None:
        value["pagination_key"] = token
    if extra:
        value.update(extra)
    return json.dumps(value, separators=(",", ":")).encode()


def fixture():
    codes = [f"{i:04d}" for i in range(1000, 1012)]
    ordered = historical.canonical_order(codes)
    blocks = historical.allocate_fresh_blocks(ordered, ordered[:1], block_size=1)
    pins = {"eligible": historical.ticker_list_sha256(ordered)}
    pins.update({name: historical.ticker_list_sha256(blocks[name]) for name in blocks})
    return codes, ordered, blocks, pins


def acquire(root, responses, sleeper=lambda _: None):
    calls, waits, attempts = [], [], []
    responses = iter(responses)
    def transport(token, key):
        calls.append(token)
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response
    def count():
        attempts.append(True)
    try:
        final = jq.acquire(root, "fake-key", COMMIT, BLOB, transport,
                           lambda seconds: (waits.append(seconds), sleeper(seconds)), count)
        return final, calls, waits, attempts
    except jq.Block as exc:
        return exc.reason, calls, waits, attempts


def test_synthetic_full_pass_and_offline_replay(tmp_path):
    codes, ordered, blocks, pins = fixture()
    calls = []
    def fake(token, key):
        calls.append(token)
        return (200, page(codes[:6], "private-token")) if token is None else (200, page(codes[6:]))
    first = jq.execute(tmp_path, COMMIT, BLOB, "fake-key", fake, lambda _: None,
                       root_override=tmp_path, pins=pins, required_count=12, block_size=1)
    assert first.startswith("JQUANTS_RECOVERY_RESULT=PASS NETWORK_BOUNDARY_CROSSED=true JQUANTS_LOGICAL_ACQUISITIONS=1 JQUANTS_HTTP_REQUESTS=2")
    assert calls == [None, "private-token"]
    manifest, pages = jq.load_raw(tmp_path / "eq-master-20260731")
    assert len(pages) == 2 and manifest["page_count"] == 2
    recovery = json.loads((tmp_path / "recovery.json").read_bytes())
    jq.validate_recovery(recovery, manifest, pins, 12, 1)
    assert recovery["assignments"] == {"eligible": ordered, **blocks}
    assert jq.semantic(pages, pins, 12, 1)[0] == ordered
    assert calls == [None, "private-token"]
    second = jq.execute(tmp_path, COMMIT, BLOB, "fake-key", fake, root_override=tmp_path,
                        pins=pins, required_count=12, block_size=1)
    assert "REASON=PRE_GATE_EXISTING_ARTIFACT_BLOCK" in second and calls == [None, "private-token"]


def test_valid_raw_lock_replays_without_network(tmp_path):
    codes, _, _, pins = fixture()
    final, calls, _, attempts = acquire(tmp_path, [(200, page(codes))])
    assert final.is_dir() and len(calls) == len(attempts) == 1
    def forbidden(_token, _key):
        pytest.fail("offline replay attempted transport")
    line = jq.execute(tmp_path, COMMIT, BLOB, "fake-key", forbidden,
                      root_override=tmp_path, pins=pins, required_count=12, block_size=1)
    assert line.startswith("JQUANTS_RECOVERY_RESULT=PASS NETWORK_BOUNDARY_CROSSED=false "
                           "JQUANTS_LOGICAL_ACQUISITIONS=0 JQUANTS_HTTP_REQUESTS=0 ")
    assert (tmp_path / "recovery.json").is_file()


@pytest.mark.parametrize("status,reason,retries", [
    (301, "SOURCE_RESPONSE_SCHEMA_INVALID", 1), (302, "SOURCE_RESPONSE_SCHEMA_INVALID", 1),
    (307, "SOURCE_RESPONSE_SCHEMA_INVALID", 1),
    (400, "SOURCE_HTTP_4XX", 1), (401, "SOURCE_HTTP_4XX", 1),
    (403, "SOURCE_HTTP_4XX", 1), (404, "SOURCE_HTTP_4XX", 1),
    (429, "SOURCE_HTTP_429_EXHAUSTED", 3),
    (500, "SOURCE_HTTP_5XX_EXHAUSTED", 3), (502, "SOURCE_HTTP_5XX_EXHAUSTED", 3),
    (503, "SOURCE_HTTP_5XX_EXHAUSTED", 3), (504, "SOURCE_HTTP_5XX_EXHAUSTED", 3),
])
def test_http_attempt_ceiling(tmp_path, status, reason, retries):
    got, calls, waits, attempts = acquire(tmp_path, [(status, b"")] * retries)
    assert got == reason and len(calls) == len(attempts) == retries
    assert waits == ([2, 5] if retries == 3 else [])


@pytest.mark.parametrize("error,reason", [(TimeoutError(), "SOURCE_TIMEOUT"),
    (urllib.error.URLError(TimeoutError()), "SOURCE_TIMEOUT"),
    (ConnectionError(), "SOURCE_TRANSPORT_FAILED"), (ssl.SSLError(), "SOURCE_TRANSPORT_FAILED")])
def test_transport_failures_are_explicitly_bounded(tmp_path, error, reason):
    got, calls, waits, attempts = acquire(tmp_path, [error] * 3)
    assert got == reason and len(calls) == len(attempts) == 3 and waits == [2, 5]


def test_retry_then_success_and_page_progression(tmp_path):
    final, calls, waits, attempts = acquire(tmp_path, [(429, b""), (200, page([], "next")),
                                                      (200, page([]))])
    assert final.is_dir() and calls == [None, None, "next"] and waits == [2]
    assert len(attempts) == 3


@pytest.mark.parametrize("responses,reason", [
    ([(200, page([], "repeat")), (200, page([], "repeat"))], "PAGINATION_LOOP"),
    ([(200, b"not-json")], "SOURCE_RESPONSE_SCHEMA_INVALID"),
    ([(200, b'{}')], "SOURCE_RESPONSE_SCHEMA_INVALID"),
    ([(200, b'{"data":{}}')], "SOURCE_RESPONSE_SCHEMA_INVALID"),
    ([(200, b'{"data":[],"pagination_key":null}')], "SOURCE_RESPONSE_SCHEMA_INVALID"),
    ([(200, b'{"data":[],"pagination_key":4}')], "SOURCE_RESPONSE_SCHEMA_INVALID"),
    ([(200, b'{"data":[],"pagination_key":""}')], "SOURCE_RESPONSE_SCHEMA_INVALID"),
])
def test_envelope_and_loop_failures(tmp_path, responses, reason):
    got, calls, _, _ = acquire(tmp_path, responses)
    assert got == reason and len(calls) == len(responses)
    assert not (tmp_path / "eq-master-20260731").exists()
    assert any(p.name.startswith("staging-") for p in tmp_path.iterdir())


def test_page_limit(tmp_path):
    got, calls, waits, attempts = acquire(tmp_path, [(200, page([], f"token-{i}")) for i in range(100)])
    assert got == "PAGINATION_LIMIT" and len(calls) == len(attempts) == 100 and waits == []


def test_raw_lock_tampering_and_create_new(tmp_path):
    final, _, _, _ = acquire(tmp_path, [(200, page(["1000"]))])
    manifest, pages = jq.load_raw(final)
    assert manifest["pages"][0]["byte_count"] == len(pages[0])
    with pytest.raises(FileExistsError):
        jq._write_new(final / "page-001.json", b"replacement")
    tampered = final / "page-001.json"
    tampered.write_bytes(b"different")
    with pytest.raises(jq.Block, match="RAW_CONTENT_LOCK_PUBLICATION_FAILED"):
        jq.load_raw(final)
    tampered.write_bytes(pages[0])
    for change in (lambda m: m.update(page_count=2),
                   lambda m: m["pages"][0].update(index=2),
                   lambda m: m.update(complete=False),
                   lambda m: m.update(sha256="0" * 64)):
        trial = json.loads(jq.canonical(manifest))
        change(trial)
        if trial["sha256"] == manifest["sha256"]:
            trial["sha256"] = jq.digest(jq.canonical({k: v for k, v in trial.items() if k != "sha256"}))
        (final / "manifest.json").write_bytes(jq.canonical(trial))
        with pytest.raises(jq.Block, match="RAW_CONTENT_LOCK_PUBLICATION_FAILED"):
            jq.load_raw(final)
    (final / "manifest.json").write_bytes(jq.canonical(manifest))
    assert jq.load_raw(final)[0] == manifest
    (final / "page-002.json").write_bytes(b"extra")
    with pytest.raises(jq.Block, match="RAW_CONTENT_LOCK_PUBLICATION_FAILED"):
        jq.load_raw(final)


def test_staging_and_final_collision_block(tmp_path):
    (tmp_path / "staging-old").mkdir()
    got, calls, _, _ = acquire(tmp_path, [(200, page([]))])
    assert got == "PRE_GATE_EXISTING_ARTIFACT_BLOCK" and calls == []


@pytest.mark.parametrize("mutation,reason", [
    ("date", "EFFECTIVE_DATE_MISMATCH"), ("count", "ELIGIBLE_COUNT_MISMATCH"),
    ("eligible", "ELIGIBLE_HASH_MISMATCH"), ("T0", "T0_HASH_MISMATCH"),
    ("T1", "T1_HASH_MISMATCH"), ("T2", "T2_HASH_MISMATCH"),
    ("T3", "T3_HASH_MISMATCH"), ("T_spare", "T_SPARE_HASH_MISMATCH"),
])
def test_semantic_first_failure(mutation, reason):
    codes, ordered, blocks, pins = fixture()
    rows = json.loads(page(codes))
    if mutation == "date":
        rows["data"][0]["Date"] = "2026-07-30"
    if mutation in pins:
        pins[mutation] = "0" * 64
    count = 13 if mutation == "count" else 12
    with pytest.raises(jq.Block) as exc:
        jq.semantic([json.dumps(rows).encode()], pins, count, 1)
    assert exc.value.reason == reason


def test_historical_primitive_equivalence():
    codes, ordered, blocks, pins = fixture()
    actual = jq.semantic([page(codes)], pins, 12, 1)
    assert actual == (historical.canonical_order(codes), blocks, pins)


def test_public_report_is_closed_and_private_values_absent(tmp_path, capsys):
    secret = "fake-api-key-fake-token-private-path-fake-ticker-error"
    def fake(_token, _key):
        raise RuntimeError(secret)
    line = jq.execute(tmp_path, COMMIT, BLOB, secret, fake, root_override=tmp_path)
    print(line)
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err + line
    assert line == ("JQUANTS_RECOVERY_RESULT=BLOCK NETWORK_BOUNDARY_CROSSED=true "
                    "JQUANTS_LOGICAL_ACQUISITIONS=1 JQUANTS_HTTP_REQUESTS=1 STAGE=SOURCE_ACQUISITION "
                    "REASON=UNEXPECTED_FAILURE RAW_CONTENT_LOCK_PUBLISHED=false RECOVERY_ARTIFACT_PUBLISHED=false")
    assert jq.readiness_probe()


def test_missing_credential_predicate_blocks_before_transport(tmp_path):
    line = jq.execute(tmp_path, COMMIT, BLOB, "", lambda _t, _k: pytest.fail("network"),
                      root_override=tmp_path)
    assert line == ("JQUANTS_RECOVERY_RESULT=BLOCK NETWORK_BOUNDARY_CROSSED=false "
                    "JQUANTS_LOGICAL_ACQUISITIONS=0 JQUANTS_HTTP_REQUESTS=0 STAGE=PRE_GATE "
                    "REASON=PRE_GATE_CREDENTIAL_BLOCK RAW_CONTENT_LOCK_PUBLISHED=false "
                    "RECOVERY_ARTIFACT_PUBLISHED=false")


def test_recovery_write_once(tmp_path):
    codes, ordered, blocks, pins = fixture()
    final, _, _, _ = acquire(tmp_path, [(200, page(codes))])
    raw, pages = jq.load_raw(final)
    _, _, hashes = jq.semantic(pages, pins, 12, 1)
    value = jq.build_recovery(raw, ordered, blocks, hashes, "2026-09-24T00:00:00Z", 1)
    jq.publish_recovery(tmp_path, value, raw, pins, 12, 1)
    with pytest.raises(jq.Block, match="RECOVERY_ARTIFACT_PUBLICATION_FAILED"):
        jq.publish_recovery(tmp_path, value, raw, pins, 12, 1)
    assert set(value) == jq.RECOVERY_KEYS
    assert value["sha256"] == jq.digest(jq.canonical({k: v for k, v in value.items() if k != "sha256"}))


def test_semantic_block_never_publishes_recovery(tmp_path):
    codes, _, _, pins = fixture()
    line = jq.execute(tmp_path, COMMIT, BLOB, "fake-key", lambda _t, _k: (200, page(codes)),
                      root_override=tmp_path, pins={**pins, "T3": "0" * 64},
                      required_count=12, block_size=1)
    assert "REASON=T3_HASH_MISMATCH" in line
    assert (tmp_path / "eq-master-20260731").is_dir()
    assert not (tmp_path / "recovery.json").exists()


def test_recovery_publication_failure_is_safe_and_preserves_raw(tmp_path, monkeypatch):
    codes, _, _, pins = fixture()
    final, _, _, _ = acquire(tmp_path, [(200, page(codes))])
    assert final.is_dir()
    def blocked_link(_source, _target):
        raise OSError("fake-private-path fake-api-key fake-ticker")
    monkeypatch.setattr(jq.os, "link", blocked_link)
    line = jq.execute(tmp_path, COMMIT, BLOB, "fake-api-key", lambda _t, _k: pytest.fail("network"),
                      root_override=tmp_path, pins=pins, required_count=12, block_size=1)
    assert "STAGE=RECOVERY_PUBLICATION REASON=RECOVERY_ARTIFACT_PUBLICATION_FAILED" in line
    assert "fake-" not in line and not (tmp_path / "recovery.json").exists()
    assert final.is_dir()


def test_raw_publication_failure_is_safe(tmp_path, monkeypatch):
    original = Path.rename
    def blocked_rename(self, target):
        if self.name.startswith("staging-"):
            raise OSError("fake-private-path fake-api-key fake-ticker")
        return original(self, target)
    monkeypatch.setattr(Path, "rename", blocked_rename)
    line = jq.execute(tmp_path, COMMIT, BLOB, "fake-api-key", lambda _t, _k: (200, page([])),
                      root_override=tmp_path)
    assert "STAGE=RAW_CONTENT_LOCK REASON=RAW_CONTENT_LOCK_PUBLICATION_FAILED" in line
    assert "JQUANTS_HTTP_REQUESTS=1" in line and "fake-" not in line
    assert not (tmp_path / "eq-master-20260731").exists()


def test_duplicate_json_key_blocks_before_semantics(tmp_path):
    got, _, _, _ = acquire(tmp_path, [(200, b'{"data":[],"data":[]}')])
    assert got == "SOURCE_RESPONSE_SCHEMA_INVALID"
