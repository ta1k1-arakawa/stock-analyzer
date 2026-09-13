from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import src.v10b_training_cache_reacquisition as v10b


TICKERS = [f"{number:04d}" for number in range(300)]
BINDING = v10b.AttemptBinding("a" * 40, "b" * 40, "c" * 40, "d" * 40)


def _successful_transport(url: str, attempt: int) -> tuple[int, bytes, bool]:
    ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
    return 200, (f"payload-{ticker}".encode("ascii")), False


def _acquire(tmp_path: Path, transport=_successful_transport, semantic=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return v10b.acquire_cache(
        tmp_path / "repo",
        tmp_path / "attempt",
        TICKERS,
        transport=transport,
        semantic_validator=semantic or (lambda body: True),
        sleep=lambda _: None,
        binding=BINDING,
    )


def test_canonical_300_ticker_run_uses_real_manifest_validator(tmp_path):
    manifest = _acquire(tmp_path)

    assert manifest["complete"] is True
    assert manifest["successful_ticker_count"] == 300
    assert len(manifest["payloads"]) == 300
    assert manifest["failed_tickers"] == []
    assert manifest["ticker_order"] == TICKERS
    assert manifest["payload_hash_list_sha256"] == v10b._payload_hash_list_sha256(manifest["payloads"])
    v10b.validate_manifest(manifest, tmp_path / "attempt", TICKERS)


def test_complete_allows_observed_failed_tickers_and_requires_exhaustive_partition(tmp_path):
    failed_tickers = {TICKERS[3], TICKERS[177], TICKERS[299]}
    calls: dict[str, int] = {}

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls[ticker] = calls.get(ticker, 0) + 1
        if ticker in failed_tickers:
            return 404, b"", False
        return 200, f"payload-{ticker}".encode(), False

    manifest = _acquire(tmp_path, transport)
    assert manifest["complete"] is True
    assert manifest["successful_ticker_count"] == 297
    assert manifest["successful_ticker_count"] + len(manifest["failed_tickers"]) == 300
    assert manifest["failed_tickers"] == [TICKERS[index] for index in (3, 177, 299)]
    assert all(count == 1 for count in calls.values())
    v10b.validate_manifest(manifest, tmp_path / "attempt", TICKERS)


def test_historical_283_success_count_is_not_required(tmp_path):
    failed = set(TICKERS[:18])

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        return (404, b"", False) if ticker in failed else (200, b"ok", False)

    manifest = _acquire(tmp_path, transport)
    assert manifest["successful_ticker_count"] == 282
    assert manifest["complete"] is True
    v10b.validate_manifest(manifest, tmp_path / "attempt", TICKERS)


def test_manifest_and_payload_fieldsets_are_exact(tmp_path):
    manifest = _acquire(tmp_path)
    extra = dict(manifest)
    extra["unexpected"] = True
    with pytest.raises(v10b.ManifestValidationError):
        v10b.validate_manifest(extra, tmp_path / "attempt", TICKERS)

    missing = dict(manifest)
    del missing["payload_hash_list_sha256"]
    with pytest.raises(v10b.ManifestValidationError):
        v10b.validate_manifest(missing, tmp_path / "attempt", TICKERS)

    payloads = list(manifest["payloads"])
    payloads[0] = dict(payloads[0], extra=True)
    mutated = dict(manifest, payloads=payloads)
    with pytest.raises(v10b.ManifestValidationError):
        v10b.validate_manifest(mutated, tmp_path / "attempt", TICKERS)


def test_payload_hash_list_is_hash_of_canonical_ordered_json(tmp_path):
    manifest = _acquire(tmp_path)
    expected_bytes = v10b.canonical_json_bytes(manifest["payloads"])
    assert manifest["payload_hash_list_sha256"] == hashlib.sha256(expected_bytes).hexdigest()
    assert manifest["payload_hash_list_sha256"] != hashlib.sha256(
        ("\n".join(item["sha256"] for item in manifest["payloads"]) + "\n").encode()
    ).hexdigest()


def test_ticker_order_and_partition_are_enforced(tmp_path):
    manifest = _acquire(tmp_path)
    reordered = dict(manifest, ticker_order=list(reversed(TICKERS)))
    with pytest.raises(v10b.ManifestValidationError):
        v10b.validate_manifest(reordered, tmp_path / "attempt", TICKERS)

    failed = list(manifest["failed_tickers"])
    invalid = dict(manifest, failed_tickers=[TICKERS[0]])
    with pytest.raises(v10b.ManifestValidationError):
        v10b.validate_manifest(invalid, tmp_path / "attempt", TICKERS)


@pytest.mark.parametrize("status", [500, 599])
def test_each_5xx_boundary_retries(tmp_path, status):
    calls: dict[str, int] = {}
    ticker = TICKERS[0]

    def transport(url: str, attempt: int):
        current = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls[current] = calls.get(current, 0) + 1
        return (status, b"", False) if current == ticker and attempt == 1 else (200, b"ok", False)

    manifest = _acquire(tmp_path, transport)
    assert calls[ticker] == 2
    assert ticker in [item["ticker"] for item in manifest["payloads"]]


@pytest.mark.parametrize("status", [400, 404, 600])
def test_non_retryable_status_does_not_retry(tmp_path, status):
    calls: dict[str, int] = {}
    ticker = TICKERS[0]

    def transport(url: str, attempt: int):
        current = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls[current] = calls.get(current, 0) + 1
        return (status, b"", False) if current == ticker else (200, b"ok", False)

    manifest = _acquire(tmp_path, transport)
    assert calls[ticker] == 1
    assert manifest["failed_tickers"][0] == ticker


def test_429_and_transport_exception_retry_at_most_three_times(tmp_path):
    calls: dict[str, int] = {}
    ticker = TICKERS[0]

    def transport(url: str, attempt: int):
        current = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls[current] = calls.get(current, 0) + 1
        if current == ticker:
            if attempt == 1:
                return 429, b"", False
            if attempt == 2:
                raise OSError("synthetic transport only")
        return 200, b"ok", False

    manifest = _acquire(tmp_path, transport)
    assert calls[ticker] == 3
    assert ticker not in manifest["failed_tickers"]


def test_redirect_and_empty_200_do_not_retry(tmp_path):
    calls: dict[str, int] = {}

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls[ticker] = calls.get(ticker, 0) + 1
        if ticker == TICKERS[0]:
            return 302, b"", True
        if ticker == TICKERS[1]:
            return 200, b"", False
        return 200, b"ok", False

    manifest = _acquire(tmp_path, transport)
    assert calls[TICKERS[0]] == 1
    assert calls[TICKERS[1]] == 1
    assert manifest["failed_tickers"][:2] == TICKERS[:2]


@pytest.mark.parametrize("status", [301, 302, 307, 308, 429, 500, 599])
def test_redirect_always_terminal_regardless_of_http_status(tmp_path, status):
    calls: dict[str, int] = {}

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls[ticker] = calls.get(ticker, 0) + 1
        if ticker == TICKERS[0]:
            return status, b"redirect-body", True
        return 200, b"ok", False

    manifest = _acquire(tmp_path, transport)
    assert calls[TICKERS[0]] == 1
    first_audit = manifest["network_audit"][0]
    assert first_audit["ticker"] == TICKERS[0]
    assert first_audit["retry"] is False
    assert first_audit["final"] is True
    assert first_audit["success"] is False
    assert first_audit["error_type"] == "REDIRECT"


def test_audit_validator_rejects_redirect_retry_even_for_retryable_status(tmp_path):
    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        if ticker == TICKERS[0]:
            return 500, b"redirect-body", True
        return 200, b"ok", False

    manifest = _acquire(tmp_path, transport)
    audit = list(manifest["network_audit"])
    audit[0] = dict(audit[0], retry=True, final=False)
    with pytest.raises(v10b.ManifestValidationError):
        v10b.validate_manifest(dict(manifest, network_audit=audit), tmp_path / "attempt", TICKERS)


def test_first_complete_body_is_locked_before_semantic_use(tmp_path):
    observed: list[bytes] = []
    attempt_root = tmp_path / "attempt"

    def semantic(body: bytes):
        if body == b"payload-0000":
            ticker_path = attempt_root / v10b.LOCKED_RAW_DIRECTORY / f"{TICKERS[0]}.json"
            assert ticker_path.read_bytes() == body
            observed.append(body)
        return True

    _acquire(tmp_path, semantic=semantic)
    assert observed == [f"payload-{TICKERS[0]}".encode()]


def test_parser_failure_after_lock_is_terminal_without_refetch(tmp_path):
    calls: dict[str, int] = {}

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls[ticker] = calls.get(ticker, 0) + 1
        return 200, f"payload-{ticker}".encode(), False

    def semantic(body: bytes):
        if body == b"payload-0000":
            raise v10b.PayloadSemanticFailure("synthetic parser failure")
        return True

    manifest = _acquire(tmp_path, transport, semantic)
    assert calls[TICKERS[0]] == 1
    assert manifest["failed_tickers"][0] == TICKERS[0]
    assert (tmp_path / "attempt" / "locked_raw" / "0000.json").read_bytes() == b"payload-0000"


def test_exact_fixed_source_has_no_fallback_or_override_surface():
    assert v10b.YAHOO_SCHEME == "https"
    assert v10b.YAHOO_HOST == "query1.finance.yahoo.com"
    assert v10b.QUERY_SPECIFICATION == (
        ("period1", "1420070400"),
        ("period2", "1577836800"),
        ("interval", "1d"),
        ("events", "div,splits"),
        ("includeAdjustedClose", "true"),
    )
    assert "fallback" not in v10b.__doc__.lower()
    assert v10b.yahoo_url("0000").endswith(
        "period1=1420070400&period2=1577836800&interval=1d&events=div,splits&includeAdjustedClose=true"
    )


def test_attempt_root_preexistence_blocks_before_transport(tmp_path):
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    called = False

    def transport(url: str, attempt: int):
        nonlocal called
        called = True
        return 200, b"unexpected", False

    with pytest.raises(v10b.GovernanceFailure):
        _acquire(tmp_path, transport)
    assert called is False


def test_attempt_root_inside_repo_and_symlink_are_rejected(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    with pytest.raises(v10b.GovernanceFailure):
        v10b.create_exclusive_attempt_root(repo_root / "attempt", repo_root)

    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlink unavailable on this host")
    with pytest.raises(v10b.GovernanceFailure):
        v10b.create_exclusive_attempt_root(link / "attempt", repo_root)


def test_repeated_synthetic_runs_are_byte_identical(tmp_path):
    first = _acquire(tmp_path / "one")
    second = _acquire(tmp_path / "two")
    first_bytes = (tmp_path / "one" / "attempt" / v10b.MANIFEST_FILE).read_bytes()
    second_bytes = (tmp_path / "two" / "attempt" / v10b.MANIFEST_FILE).read_bytes()
    assert first == second
    assert first_bytes == second_bytes
    assert first_bytes.endswith(b"\n") and b"\r" not in first_bytes


def test_repository_preflight_failure_never_calls_acquisition(monkeypatch, tmp_path):
    called = False

    def fail_acquisition(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("transport boundary reached")

    monkeypatch.setattr(v10b, "acquire_cache", fail_acquisition)
    monkeypatch.setattr(
        v10b,
        "validate_repository_preflight",
        lambda repo_root, implementation_sha: (_ for _ in ()).throw(
            v10b.GovernanceFailure("synthetic repository mismatch")
        ),
    )
    with pytest.raises(v10b.GovernanceFailure):
        v10b.run_production(tmp_path / "repo", tmp_path / "attempt", "d" * 40)
    assert called is False


def test_inherited_parser_readiness_failure_is_preflight_before_attempt_creation(monkeypatch, tmp_path):
    transport_calls = 0

    def fail_acquisition(*args, **kwargs):
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("acquisition must not begin")

    monkeypatch.setattr(v10b, "validate_repository_preflight", lambda repo_root, implementation_sha: TICKERS)
    monkeypatch.setattr(
        v10b,
        "_resolve_inherited_parser",
        lambda: (_ for _ in ()).throw(v10b.GovernanceFailure("synthetic parser readiness failure")),
    )
    monkeypatch.setattr(v10b, "acquire_cache", fail_acquisition)

    with pytest.raises(v10b.GovernanceFailure):
        v10b.run_production(tmp_path / "repo", tmp_path / "attempt", "d" * 40)

    assert transport_calls == 0
    assert not (tmp_path / "attempt").exists()


def test_default_validator_maps_malformed_utf8_after_lock_without_refetch(tmp_path):
    calls: list[tuple[str, int]] = []

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls.append((ticker, attempt))
        if ticker == TICKERS[0]:
            return 200, b"\xff", False
        return 404, b"", False

    manifest = _acquire(
        tmp_path,
        transport,
        v10b._semantic_validator_for_parser(lambda payload: None),
    )

    assert calls[0] == (TICKERS[0], 1)
    assert calls.count((TICKERS[0], 1)) == 1
    assert manifest["failed_tickers"][0] == TICKERS[0]
    assert (tmp_path / "attempt" / "locked_raw" / "0000.json").read_bytes() == b"\xff"


def test_default_validator_maps_malformed_json_after_lock_without_refetch(tmp_path):
    calls: list[tuple[str, int]] = []

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls.append((ticker, attempt))
        if ticker == TICKERS[0]:
            return 200, b"{", False
        return 404, b"", False

    manifest = _acquire(
        tmp_path,
        transport,
        v10b._semantic_validator_for_parser(lambda payload: None),
    )

    assert calls.count((TICKERS[0], 1)) == 1
    assert manifest["failed_tickers"][0] == TICKERS[0]
    assert (tmp_path / "attempt" / "locked_raw" / "0000.json").read_bytes() == b"{"


def test_inherited_parser_value_error_is_payload_failure_without_refetch(tmp_path):
    calls: list[tuple[str, int]] = []

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls.append((ticker, attempt))
        return 200, b"{}", False

    def parser(payload):
        if len(calls) == 1:
            raise ValueError("malformed parser input")

    manifest = _acquire(tmp_path, transport, v10b._semantic_validator_for_parser(parser))

    assert calls.count((TICKERS[0], 1)) == 1
    assert manifest["failed_tickers"][0] == TICKERS[0]


@pytest.mark.parametrize("error", [RuntimeError("unexpected parser error"), AssertionError("unexpected parser assertion")])
def test_unexpected_parser_errors_abort_without_failed_ticker_or_terminal_manifest(tmp_path, error):
    calls: list[tuple[str, int]] = []

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls.append((ticker, attempt))
        return 200, b"{}", False

    def parser(payload):
        raise error

    with pytest.raises(type(error), match="unexpected parser"):
        _acquire(tmp_path, transport, v10b._semantic_validator_for_parser(parser))

    assert calls == [(TICKERS[0], 1)]
    assert not (tmp_path / "attempt" / v10b.MANIFEST_FILE).exists()


def test_attempt_receipt_failure_is_preflight_with_zero_transport_calls(monkeypatch, tmp_path):
    calls = 0
    original = v10b._write_exclusive

    def fail_receipt(path, body, *, failure_cls=v10b.GovernanceFailure):
        if path.name == v10b.ATTEMPT_RECEIPT_FILE:
            raise v10b.GovernanceFailure("synthetic receipt failure")
        return original(path, body, failure_cls=failure_cls)

    def transport(url: str, attempt: int):
        nonlocal calls
        calls += 1
        return 200, b"unexpected", False

    monkeypatch.setattr(v10b, "_write_exclusive", fail_receipt)
    with pytest.raises(v10b.GovernanceFailure):
        _acquire(tmp_path, transport)
    assert calls == 0


def test_raw_lock_failure_after_response_is_post_boundary_and_not_retried(monkeypatch, tmp_path):
    calls: list[tuple[str, int]] = []
    original = v10b._write_exclusive

    def fail_raw(path, body, *, failure_cls=v10b.GovernanceFailure):
        if path.name == "0000.json":
            raise v10b.PostBoundaryFailure("synthetic raw lock failure")
        return original(path, body, failure_cls=failure_cls)

    def transport(url: str, attempt: int):
        ticker = url.split("/chart/", 1)[1].split(".T?", 1)[0]
        calls.append((ticker, attempt))
        return 200, b"first-complete-body", False

    monkeypatch.setattr(v10b, "_write_exclusive", fail_raw)
    with pytest.raises(v10b.PostBoundaryFailure):
        _acquire(tmp_path, transport)
    assert calls == [("0000", 1)]


def test_legacy_governance_write_error_after_response_is_reclassified(monkeypatch, tmp_path):
    original = v10b._write_exclusive

    def legacy_failure(path, body, *, failure_cls=v10b.GovernanceFailure):
        if path.name == "0000.json":
            raise v10b.GovernanceFailure("legacy helper failure")
        return original(path, body, failure_cls=failure_cls)

    monkeypatch.setattr(v10b, "_write_exclusive", legacy_failure)
    with pytest.raises(v10b.PostBoundaryFailure):
        _acquire(tmp_path)


def test_manifest_write_failure_after_loop_is_post_boundary_without_second_acquisition(monkeypatch, tmp_path):
    calls = 0
    original = v10b._write_exclusive

    def fail_manifest(path, body, *, failure_cls=v10b.GovernanceFailure):
        if path.name == v10b.MANIFEST_FILE:
            raise v10b.PostBoundaryFailure("synthetic manifest write failure")
        return original(path, body, failure_cls=failure_cls)

    def transport(url: str, attempt: int):
        nonlocal calls
        calls += 1
        return 200, b"ok", False

    monkeypatch.setattr(v10b, "_write_exclusive", fail_manifest)
    with pytest.raises(v10b.PostBoundaryFailure):
        _acquire(tmp_path, transport)
    assert calls == 300


def test_post_network_payload_closure_failure_is_not_preflight(monkeypatch, tmp_path):
    original = v10b._safe_regular_file

    def fail_payload_stat(path: Path):
        if v10b.LOCKED_RAW_DIRECTORY in path.parts:
            raise v10b.GovernanceFailure("synthetic payload stat failure")
        return original(path)

    monkeypatch.setattr(v10b, "_safe_regular_file", fail_payload_stat)
    with pytest.raises(v10b.PostBoundaryFailure):
        _acquire(tmp_path)


def test_attempt_receipt_has_only_safe_governance_facts(tmp_path):
    _acquire(tmp_path)
    receipt = json.loads((tmp_path / "attempt" / v10b.ATTEMPT_RECEIPT_FILE).read_text())
    assert receipt == {
        "schema_version": v10b.ATTEMPT_RECEIPT_SCHEMA,
        "study_identity": v10b.STUDY_IDENTITY,
        "frozen_design_git_commit": "a" * 40,
        "frozen_design_git_blob_sha": "b" * 40,
        "freeze_approval_git_blob_sha": "c" * 40,
        "implementation_sha": "d" * 40,
        "attempt_started": True,
        "network_acquisition_scope": "V10B_PUBLIC_YAHOO_ONLY",
    }


def test_manifest_and_result_surface_never_contains_raw_payload_bytes(tmp_path, capsys):
    manifest = _acquire(tmp_path)
    captured = capsys.readouterr()
    encoded = json.dumps(manifest, sort_keys=True).encode("utf-8")
    assert captured.out == ""
    assert captured.err == ""
    assert b"payload-0000" not in encoded
    assert b"payload-0299" not in encoded


def test_preflight_branch_head_remote_and_dirty_failures_are_governance_failures(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    implementation_sha = "d" * 40
    valid = {
        "config --get remote.origin.url": v10b.EXPECTED_REPOSITORY_URL,
        "rev-parse --abbrev-ref HEAD": v10b.AUTHORITATIVE_BRANCH,
        "rev-parse HEAD": implementation_sha,
        f"rev-parse origin/{v10b.AUTHORITATIVE_BRANCH}": implementation_sha,
        "status --porcelain --untracked-files=all": "",
        f"rev-parse HEAD:{v10b.DESIGN_FILE}": v10b.FROZEN_DESIGN_BLOB,
        f"rev-parse HEAD:{v10b.FREEZE_APPROVAL_FILE}": v10b.FREEZE_APPROVAL_BLOB,
    }
    monkeypatch.setattr(v10b, "_validate_freeze_approval", lambda root: None)
    monkeypatch.setattr(v10b, "load_fixed_universe", lambda root: TICKERS)

    for key in (
        "rev-parse --abbrev-ref HEAD",
        "rev-parse HEAD",
        f"rev-parse origin/{v10b.AUTHORITATIVE_BRANCH}",
        "status --porcelain --untracked-files=all",
    ):
        values = dict(valid)
        values[key] = "wrong" if key != "status --porcelain --untracked-files=all" else " M file"
        monkeypatch.setattr(v10b, "_git", lambda root, *args, values=values: values[" ".join(args)])
        with pytest.raises(v10b.GovernanceFailure):
            v10b.validate_repository_preflight(repo_root, implementation_sha)


@pytest.mark.parametrize(
    "failure_key",
    [
        "repository",
        "design_blob",
        "approval",
    ],
)
def test_preflight_repository_design_and_approval_failures_stop_before_transport(monkeypatch, tmp_path, failure_key):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    implementation_sha = "d" * 40
    values = {
        "config --get remote.origin.url": v10b.EXPECTED_REPOSITORY_URL,
        "rev-parse --abbrev-ref HEAD": v10b.AUTHORITATIVE_BRANCH,
        "rev-parse HEAD": implementation_sha,
        f"rev-parse origin/{v10b.AUTHORITATIVE_BRANCH}": implementation_sha,
        "status --porcelain --untracked-files=all": "",
        f"rev-parse HEAD:{v10b.DESIGN_FILE}": v10b.FROZEN_DESIGN_BLOB,
        f"rev-parse HEAD:{v10b.FREEZE_APPROVAL_FILE}": v10b.FREEZE_APPROVAL_BLOB,
    }
    if failure_key == "repository":
        values["config --get remote.origin.url"] = "https://github.com/other-owner/stock-analyzer.git"
    elif failure_key == "design_blob":
        values[f"rev-parse HEAD:{v10b.DESIGN_FILE}"] = "e" * 40
    else:
        approval = repo_root / v10b.FREEZE_APPROVAL_FILE
        approval.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(v10b, "_git", lambda root, *args: values[" ".join(args)])
    monkeypatch.setattr(v10b, "load_fixed_universe", lambda root: TICKERS)
    called = False

    def no_transport(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("transport must not be reached")

    monkeypatch.setattr(v10b, "acquire_cache", no_transport)
    with pytest.raises(v10b.GovernanceFailure):
        v10b.run_production(repo_root, tmp_path / "outside-attempt", "d" * 40)
    assert called is False
