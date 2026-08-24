from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.v9_005_stage_a_jpx_probe as m

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 24, 3, 0, tzinfo=timezone.utc)


def _clock() -> datetime:
    return NOW


def _no_sleep(_seconds: int) -> None:
    return None


# --- 1. No real network in tests --------------------------------------------

def test_no_real_network_socket_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard: if any code path under test tried to open a real socket, this
    fails loudly instead of silently reaching the network."""

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real network socket attempted during tests")

    monkeypatch.setattr(socket, "socket", _blocked)
    inventory = m.build_source_inventory()
    assert len(inventory) == len(m.inventory_months()) * len(m.SOURCE_FAMILIES)


# --- 2/3. Off-domain request/redirect rejection -----------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/x",
        "http://www.jpx.co.jp/x",
        "https://jpx.co.jp.evil.example/x",
        "https://user@www.jpx.co.jp/x",
        "https://www.jpx.co.jp:444/x",
        "https://www.jpx.co.jp/x#frag",
        None,
        123,
    ],
)
def test_off_domain_url_rejected(url: object) -> None:
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.validate_jpx_url(url)
    assert excinfo.value.reason == "OFF_DOMAIN_REQUEST_REJECTED"
    assert excinfo.value.failure_class == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE


@pytest.mark.parametrize(
    "url",
    ["https://www.jpx.co.jp/x", "https://sub.jpx.co.jp/x", "https://jpx.co.jp/x"],
)
def test_allowed_jpx_url_accepted(url: str) -> None:
    assert m.validate_jpx_url(url) == url


def test_off_domain_redirect_rejected() -> None:
    def fetcher(_url: str) -> tuple[bytes, str]:
        return b"payload", "https://evil.example/redirected"

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.fetch_once_with_retry("https://www.jpx.co.jp/x", fetcher, _no_sleep)
    assert excinfo.value.reason == "OFF_DOMAIN_REDIRECT_REJECTED"


def test_off_domain_request_rejected_before_any_fetch_call() -> None:
    calls: list[str] = []

    def fetcher(url: str) -> tuple[bytes, str]:
        calls.append(url)
        return b"payload", url

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.fetch_once_with_retry("https://evil.example/x", fetcher, _no_sleep)
    assert excinfo.value.reason == "OFF_DOMAIN_REQUEST_REJECTED"
    assert calls == []


# --- 4/5/6. Raw locking: first-complete-payload, no overwrite, reprocessing --

def test_first_complete_payload_lock(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = m.lock_first_complete_payload(
        root,
        source_family=m.SOURCE_FAMILY_JPX_CALENDAR,
        applicable_period="CURRENT",
        requested_url=m.CALENDAR_PAGE_URL,
        resolved_url=m.CALENDAR_PAGE_URL,
        http_status=200,
        payload=b"raw-bytes",
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )
    assert locked["raw"] == b"raw-bytes"
    assert locked["sha256"] == m.sha256_bytes(b"raw-bytes")
    assert locked["byte_length"] == len(b"raw-bytes")
    assert set(locked) - {"raw"} == m._REQUIRED_LOCK_META_FIELDS
    reread = m.read_locked_payload(root, m.SOURCE_FAMILY_JPX_CALENDAR, "CURRENT", m.CALENDAR_PAGE_URL)
    assert reread is not None
    assert reread["raw"] == b"raw-bytes"


def test_no_overwrite_on_second_lock(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    kwargs = dict(
        source_family=m.SOURCE_FAMILY_JPX_CALENDAR,
        applicable_period="CURRENT",
        requested_url=m.CALENDAR_PAGE_URL,
        resolved_url=m.CALENDAR_PAGE_URL,
        http_status=200,
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )
    m.lock_first_complete_payload(root, payload=b"first", **kwargs)
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.lock_first_complete_payload(root, payload=b"second-attempt", **kwargs)
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    # never silently overwritten
    reread = m.read_locked_payload(root, m.SOURCE_FAMILY_JPX_CALENDAR, "CURRENT", m.CALENDAR_PAGE_URL)
    assert reread["raw"] == b"first"


def test_same_locked_payload_reprocessed_not_refetched(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    calls: list[str] = []

    def fetcher(url: str) -> tuple[bytes, str]:
        calls.append(url)
        return b"payload-one", url

    locked_first, requests_first = m.ensure_locked_payload(
        root,
        source_family=m.SOURCE_FAMILY_JPX_CALENDAR,
        applicable_period="CURRENT",
        requested_url=m.CALENDAR_PAGE_URL,
        fetcher=fetcher,
        sleep=_no_sleep,
        clock=_clock,
    )
    assert requests_first == 1
    assert len(calls) == 1

    locked_second, requests_second = m.ensure_locked_payload(
        root,
        source_family=m.SOURCE_FAMILY_JPX_CALENDAR,
        applicable_period="CURRENT",
        requested_url=m.CALENDAR_PAGE_URL,
        fetcher=fetcher,
        sleep=_no_sleep,
        clock=_clock,
    )
    assert requests_second == 0
    assert len(calls) == 1  # no second network call
    assert locked_second["raw"] == locked_first["raw"] == b"payload-one"


def test_output_root_collision_rejected(tmp_path: Path) -> None:
    target = tmp_path / "out"
    m.initialize_output_root(target)
    with pytest.raises(m.V9005StageABlocked):
        m.initialize_output_root(target)


# --- 7/8. Inventory: required missing => FAIL; ambiguous/unknown => MISSING -

def test_inventory_defaults_to_missing_for_every_cell() -> None:
    inventory = m.build_source_inventory()
    assert len(inventory) == len(m.inventory_months()) * len(m.SOURCE_FAMILIES)
    assert all(record["status"] == m.INVENTORY_MISSING for record in inventory)


def test_inventory_available_only_when_locked() -> None:
    month = m.inventory_months()[0]
    family = m.SOURCE_FAMILY_JPX_CALENDAR
    inventory = m.build_source_inventory(locked_index={(family, month): object()})
    record = next(r for r in inventory if r["source_family"] == family and r["month"] == month)
    assert record["status"] == m.INVENTORY_AVAILABLE
    other = next(r for r in inventory if r["source_family"] == family and r["month"] != month)
    assert other["status"] == m.INVENTORY_MISSING


def test_unknown_family_or_month_is_ambiguous_fail_closed() -> None:
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_month_locator("NOT_A_REAL_FAMILY", m.inventory_months()[0])
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_month_locator(m.SOURCE_FAMILY_JPX_CALENDAR, "2099-01")


def test_required_inventory_missing_causes_fail() -> None:
    inventory = m.build_source_inventory()  # all MISSING
    evidence = m.compute_stage_a_evidence(
        inventory=inventory,
        terminal_snapshot_locked=True,
        trading_calendar_derived=True,
        reconstruction_deterministic=True,
        comparable_month_end_mismatch_count=0,
        raw_provenance_pass=True,
    )
    assert evidence["required_inventory_missing_count"] > 0
    assert evidence["FREE_JPX_METADATA_PROBE_PASS"] is False
    assert evidence["failure_class"] == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE


# --- 9. Deterministic repeated reconstruction --------------------------------

def test_reconstruction_is_deterministic() -> None:
    evidence_input = {"terminal_snapshot": {"sha256": "a" * 64}}
    assert m.reconstruction_is_deterministic(evidence_input) is True
    first = m.reconstruct_security_state(evidence_input)
    second = m.reconstruct_security_state(evidence_input)
    assert first == second


def test_reconstruction_empty_input_is_still_deterministic() -> None:
    assert m.reconstruction_is_deterministic({}) is True


# --- 10. Comparable month-end count mismatch => FAIL ------------------------

def test_month_end_mismatch_detected() -> None:
    official = {"2018-01": 3700}
    reconstructed_ok = {"2018-01": 3700}
    reconstructed_bad = {"2018-01": 3699}
    assert m.compute_month_end_mismatch_count(official, reconstructed_ok) == 0
    assert m.compute_month_end_mismatch_count(official, reconstructed_bad) == 1


def test_month_end_mismatch_fails_overall_pass_even_if_everything_else_passes() -> None:
    full_inventory = m.build_source_inventory(
        locked_index={(family, month): object() for family in m.SOURCE_FAMILIES for month in m.inventory_months()}
    )
    evidence = m.compute_stage_a_evidence(
        inventory=full_inventory,
        terminal_snapshot_locked=True,
        trading_calendar_derived=True,
        reconstruction_deterministic=True,
        comparable_month_end_mismatch_count=1,
        raw_provenance_pass=True,
    )
    assert evidence["FREE_JPX_METADATA_PROBE_PASS"] is False


# --- 11. Design blob mismatch => STOP ---------------------------------------

def test_signal_grid_binding_mismatch_stops() -> None:
    def fake_git(args: list[str]) -> str:
        if args == ["rev-parse", "HEAD"]:
            return "a" * 40
        if args == ["rev-parse", f"HEAD:{m.BOUND_SIGNAL_GRID_PATH}"]:
            return "b" * 40  # wrong blob
        raise AssertionError(f"unexpected git args {args}")

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.verify_signal_grid_binding("/unused", git=fake_git)
    assert excinfo.value.reason == m.PROBE_SIGNAL_GRID_CONTRACT_MISMATCH
    assert excinfo.value.failure_class == m.PROBE_SIGNAL_GRID_CONTRACT_MISMATCH


def test_signal_grid_binding_match_passes() -> None:
    def fake_git(args: list[str]) -> str:
        if args == ["rev-parse", "HEAD"]:
            return "c" * 40
        if args == ["rev-parse", f"HEAD:{m.BOUND_SIGNAL_GRID_PATH}"]:
            return m.BOUND_SIGNAL_GRID_BLOB_SHA
        raise AssertionError(f"unexpected git args {args}")

    assert m.verify_signal_grid_binding("/unused", git=fake_git) == "c" * 40


def test_signal_grid_binding_verified_against_real_repository_head() -> None:
    """Real (local, no-network) check that the current repository actually
    carries the bound blob -- this is what the atomic .ps1 also verifies
    before any Stage-A network request."""
    result = subprocess.run(
        ["git", "rev-parse", f"HEAD:{m.BOUND_SIGNAL_GRID_PATH}"],
        cwd=str(ROOT), check=True, text=True, capture_output=True,
    )
    assert result.stdout.strip() == m.BOUND_SIGNAL_GRID_BLOB_SHA


# --- 12. Exact Stage-A PASS conjunction --------------------------------------

def _full_evidence(**overrides: object) -> dict[str, object]:
    full_inventory = m.build_source_inventory(
        locked_index={(family, month): object() for family in m.SOURCE_FAMILIES for month in m.inventory_months()}
    )
    kwargs = dict(
        inventory=full_inventory,
        terminal_snapshot_locked=True,
        trading_calendar_derived=True,
        reconstruction_deterministic=True,
        comparable_month_end_mismatch_count=0,
        raw_provenance_pass=True,
    )
    kwargs.update(overrides)
    return m.compute_stage_a_evidence(**kwargs)


def test_exact_pass_conjunction_true_when_everything_passes() -> None:
    evidence = _full_evidence()
    assert evidence["FREE_JPX_METADATA_PROBE_PASS"] is True
    assert evidence["failure_class"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"terminal_snapshot_locked": False},
        {"trading_calendar_derived": False},
        {"reconstruction_deterministic": False},
        {"raw_provenance_pass": False},
        {"comparable_month_end_mismatch_count": 1},
    ],
)
def test_exact_pass_conjunction_false_if_any_single_condition_fails(overrides: dict[str, object]) -> None:
    evidence = _full_evidence(**overrides)
    assert evidence["FREE_JPX_METADATA_PROBE_PASS"] is False
    assert evidence["failure_class"] == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE


# --- Endpoint derivation ------------------------------------------------------

def _weekday_only_trading_days(start: str, end: str) -> tuple[str, ...]:
    return m.build_trading_day_set(market_holiday_dates=(), coverage_start=start, coverage_end=end)


def test_derive_stage_b_global_end_exclusive_mechanical() -> None:
    trading_days = _weekday_only_trading_days("2018-01-01", "2026-06-30")
    endpoint = m.derive_stage_b_global_end_exclusive(trading_days, coverage_start="2018-01-01")
    final_d0 = endpoint["final_signal_d0"]
    assert final_d0 <= "2025-12-31"
    # j0/D0 cadence property directly re-derived from trading_days.
    j0 = next(i for i, d in enumerate(trading_days) if d >= "2018-01-01")
    d0_index = trading_days.index(final_d0)
    assert (d0_index - j0) % 3 == 0
    assert endpoint["final_planned_d3"] > final_d0
    assert endpoint["stage_b_global_end_exclusive"] > endpoint["final_possible_exit_day"]


def test_derive_endpoint_fails_closed_on_insufficient_calendar_coverage() -> None:
    trading_days = _weekday_only_trading_days("2026-01-01", "2026-03-01")
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.derive_stage_b_global_end_exclusive(trading_days, coverage_start="2026-01-01")
    assert excinfo.value.failure_class == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE


def test_derive_endpoint_fails_closed_on_insufficient_forward_tail() -> None:
    trading_days = _weekday_only_trading_days("2018-01-01", "2025-12-31")
    with pytest.raises(m.V9005StageABlocked):
        m.derive_stage_b_global_end_exclusive(trading_days, coverage_start="2018-01-01")


# --- Terminal-snapshot locator reuse (extract_data_j_xls_url) ---------------

def test_extract_data_j_xls_url_reused_pattern() -> None:
    page = b'<html><a href="/x/data_j.xls">Excel</a></html>'
    assert m.extract_data_j_xls_url(page) == "https://www.jpx.co.jp/x/data_j.xls"


def test_extract_data_j_xls_url_rejects_off_domain_link() -> None:
    page = b'<html><a href="https://evil.example/data_j.xls">Excel</a></html>'
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.extract_data_j_xls_url(page)
    assert excinfo.value.reason == "OFF_DOMAIN_REDIRECT_REJECTED"


def test_extract_data_j_xls_url_missing_link_fails_closed() -> None:
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.extract_data_j_xls_url(b"<html>no link here</html>")
    assert excinfo.value.failure_class == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE


# --- Transport retry classification (per AI_REAL_EXECUTION_RUNBOOK.md) -----

def test_transport_retryable_then_success() -> None:
    attempts: list[int] = []

    def fetcher(url: str) -> tuple[bytes, str]:
        attempts.append(1)
        if len(attempts) < 2:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        return b"payload", url

    payload, final_url, requests_used = m.fetch_once_with_retry("https://www.jpx.co.jp/x", fetcher, _no_sleep)
    assert payload == b"payload"
    assert requests_used == 2


def test_transport_nonretryable_fails_immediately() -> None:
    def fetcher(url: str) -> tuple[bytes, str]:
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.fetch_once_with_retry("https://www.jpx.co.jp/x", fetcher, _no_sleep)
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE


def test_transport_exhausted_retries_is_plumbing_failure_retriable() -> None:
    def fetcher(url: str) -> tuple[bytes, str]:
        raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.fetch_once_with_retry("https://www.jpx.co.jp/x", fetcher, _no_sleep)
    assert excinfo.value.reason == m.PLUMBING_FAILURE_RETRIABLE


# --- End-to-end run_stage_a with fully synthetic, offline fixtures ---------

def _synthetic_calendar_html() -> bytes:
    rows = [
        "<tr><th>2026</th></tr>",
        "<tr><td>Jan. 1</td><td>New Year's Day</td></tr>",
        "<tr><th>2027</th></tr>",
        "<tr><td>Jan. 1</td><td>New Year's Day</td></tr>",
    ]
    html = (
        "<html><body><h2>Market Holidays</h2><table>"
        + "\n".join(rows)
        + "</table></body></html>"
    )
    return html.encode("utf-8")


def _synthetic_listing_page() -> bytes:
    return b'<html><a href="/markets/statistics-equities/misc/data_j.xls">Excel</a></html>'


def _fake_git_bound() -> object:
    def fake_git(args: list[str]) -> str:
        if args == ["rev-parse", "HEAD"]:
            return "d" * 40
        if args == ["rev-parse", f"HEAD:{m.BOUND_SIGNAL_GRID_PATH}"]:
            return m.BOUND_SIGNAL_GRID_BLOB_SHA
        raise AssertionError(f"unexpected git args {args}")
    return fake_git


# --- V9_006_HIGH_1: incomplete locator contract stops before ANY network ---

def test_locator_contract_is_currently_incomplete() -> None:
    """Ground truth for this whole remediation: under current reviewed
    repository evidence, no source family has a resolvable locator for
    every required monthly slot, so the contract is incomplete."""
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.verify_locator_contract_complete()
    assert excinfo.value.reason == m.STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED


def test_locator_contract_complete_passes_when_no_missing_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "resolve_month_locator", lambda family, month: object())
    m.verify_locator_contract_complete()  # must not raise


def test_run_stage_a_incomplete_locator_contract_stops_before_any_network(tmp_path: Path) -> None:
    calls: list[str] = []
    git_calls: list[list[str]] = []

    def fetcher(url: str) -> tuple[bytes, str]:
        calls.append(url)
        raise AssertionError("must not fetch while the locator contract is incomplete")

    def fake_git(args: list[str]) -> str:
        git_calls.append(args)
        raise AssertionError("must not call git while the locator contract is incomplete")

    out = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_stage_a(
            output_root=out,
            repo_root=str(ROOT),
            confirmation=m.CONFIRMATION,
            fetcher=fetcher,
            sleep=_no_sleep,
            clock=_clock,
            git=fake_git,
        )
    assert excinfo.value.reason == m.STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED
    # Never SOURCE_OR_DATA_FEASIBILITY_FAILURE: that class is reserved for a
    # genuine result after the locator contract is complete and the probe
    # actually ran.
    assert excinfo.value.failure_class != m.SOURCE_OR_DATA_FEASIBILITY_FAILURE
    assert calls == []
    assert git_calls == []
    assert not out.exists()  # no durable state created for a stop this early


def test_run_stage_a_wrong_confirmation_never_fetches(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> tuple[bytes, str]:
        calls.append(url)
        raise AssertionError("must not fetch on bad confirmation")

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_stage_a(
            output_root=tmp_path / "out",
            repo_root=str(ROOT),
            confirmation="WRONG_TOKEN",
            fetcher=fetcher,
            sleep=_no_sleep,
            clock=_clock,
            git=_fake_git_bound(),
        )
    assert excinfo.value.failure_class == m.GOVERNANCE_FAILURE
    assert calls == []
    assert not (tmp_path / "out").exists()


# --- Regression coverage: once the locator contract IS complete (forced via
# monkeypatch, simulating a future, separately reviewed extension), the
# existing fetch/lock/evidence pipeline below the gate still behaves
# correctly. This is not itself a claim that the contract is complete today.

def _force_locator_contract_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "verify_locator_contract_complete", lambda: None)


def test_run_stage_a_offline_reports_fail_with_safe_evidence_once_contract_forced_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_locator_contract_complete(monkeypatch)
    responses = {
        m.LISTED_ISSUES_PAGE_URL: (_synthetic_listing_page(), m.LISTED_ISSUES_PAGE_URL),
        "https://www.jpx.co.jp/markets/statistics-equities/misc/data_j.xls": (b"xls-bytes", "https://www.jpx.co.jp/markets/statistics-equities/misc/data_j.xls"),
        m.CALENDAR_PAGE_URL: (_synthetic_calendar_html(), m.CALENDAR_PAGE_URL),
    }

    def fetcher(url: str) -> tuple[bytes, str]:
        return responses[url]

    summary = m.run_stage_a(
        output_root=tmp_path / "stage-a-out",
        repo_root=str(ROOT),
        confirmation=m.CONFIRMATION,
        fetcher=fetcher,
        sleep=_no_sleep,
        clock=_clock,
        git=_fake_git_bound(),
    )
    # Even with a forced-complete locator contract, the underlying monthly
    # SOURCE_INVENTORY built from real locked evidence is still empty here
    # (the synthetic fixtures only lock the two non-monthly artifacts), so
    # this remains the honest FAIL outcome -- this test exists to prove the
    # fetch/lock/evidence pipeline below the new gate still works, not to
    # claim the real contract is complete.
    assert summary["status"] == "FAIL"
    assert summary["failure_class"] == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE
    assert summary["required_inventory_missing_count"] > 0
    assert summary["terminal_snapshot_pass"] is True
    assert summary["signal_grid_binding_verified_head"] == "d" * 40
    forbidden_keys = {
        "raw", "payload", "price", "close", "open", "high", "low", "volume",
        "ticker", "security_identity", "canonical_security_identity",
    }
    assert set(summary) & forbidden_keys == set()
    serialized = json.dumps(summary, ensure_ascii=False)
    assert "xls-bytes" not in serialized
    durable_root = tmp_path / "stage-a-out"
    for name in ("inventory.json", "reconstruction.json", "result.json", "receipt.json"):
        assert (durable_root / name).exists()


def test_run_stage_a_wrong_signal_grid_blob_stops_before_any_fetch_once_contract_forced_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_locator_contract_complete(monkeypatch)
    calls: list[str] = []

    def fetcher(url: str) -> tuple[bytes, str]:
        calls.append(url)
        raise AssertionError("must not fetch on signal-grid contract mismatch")

    def fake_git(args: list[str]) -> str:
        if args == ["rev-parse", "HEAD"]:
            return "e" * 40
        if args == ["rev-parse", f"HEAD:{m.BOUND_SIGNAL_GRID_PATH}"]:
            return "f" * 40
        raise AssertionError(f"unexpected git args {args}")

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_stage_a(
            output_root=tmp_path / "out",
            repo_root=str(ROOT),
            confirmation=m.CONFIRMATION,
            fetcher=fetcher,
            sleep=_no_sleep,
            clock=_clock,
            git=fake_git,
        )
    assert excinfo.value.reason == m.PROBE_SIGNAL_GRID_CONTRACT_MISMATCH
    assert calls == []


# --- Safe stdout for the CHATGPT_DECISION_REQUIRED stop (CLI) --------------

def test_cli_script_incomplete_locator_contract_prints_safe_chatgpt_decision_required(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    monkeypatch.setenv("V9_005_STAGE_A_CONFIRMATION", m.CONFIRMATION)
    import importlib

    cli = importlib.import_module("run_v9_005_stage_a_jpx_probe")
    importlib.reload(cli)
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root), "--repo-root", str(ROOT)])
    assert exit_code == 2
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["execution_result"] == "BLOCKED"
    assert payload["failure_class"] == "CHATGPT_DECISION_REQUIRED"
    assert payload["status"] == "CHATGPT_DECISION_REQUIRED"
    assert payload["reason"] == "STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE"
    assert payload["network_request_count"] == 0
    assert not output_root.exists()


# --- Safe stdout / no identity leakage (CLI script) -------------------------

def test_cli_script_missing_confirmation_prints_only_safe_failure(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    monkeypatch.delenv("V9_005_STAGE_A_CONFIRMATION", raising=False)
    import importlib

    cli = importlib.import_module("run_v9_005_stage_a_jpx_probe")
    importlib.reload(cli)
    exit_code = cli.main(["--output-root", "/does/not/matter", "--repo-root", str(ROOT)])
    assert exit_code == 2
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["execution_result"] == "BLOCKED"
    assert payload["failure_class"] == "GOVERNANCE_FAILURE"
    assert set(payload) == {"schema_version", "study", "stage", "execution_result", "failure_class", "network_request_count"}


# --- PowerShell atomic-scope fail-closed semantics (static checks) ---------

PS1_PATH = ROOT / "scripts" / "run_v9_005_stage_a_jpx_probe.ps1"


def test_ps1_is_single_atomic_scope() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    non_comment_lines = [
        line for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    joined = "\n".join(non_comment_lines)
    # Exactly one top-level "& {" atomic scope opener (after the param block).
    assert joined.count("& {") == 1
    assert "$ErrorActionPreference = \"Stop\"" in text


def test_ps1_sets_error_action_stop_inside_the_atomic_scope() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    scope_start = text.index("& {")
    after_scope = text[scope_start:]
    stop_index = after_scope.index("$ErrorActionPreference = \"Stop\"")
    # Must appear early, before any network/git/preflight statements.
    assert stop_index < after_scope.index("git ")


def test_ps1_avoids_automatic_variable_name_collisions() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    forbidden = re.compile(r"\$(Matches|Error|Args|Input|Host|PID|HOME)\b", re.IGNORECASE)
    matches = forbidden.findall(text)
    assert matches == []


def test_ps1_verifies_all_required_preflight_boundaries_before_network() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    scope_start = text.index("& {")
    network_marker = text.index("Running Stage-A probe")
    preflight_region = text[scope_start:network_marker]
    for required_marker in (
        "EXPECTED_HEAD_MISMATCH",
        "authoritativeBranch",
        "git status --porcelain",
        "PROBE_SIGNAL_GRID_CONTRACT_MISMATCH",
        "OutputRoot already exists",
        "confirmation token",
        "canonical interpreter",
    ):
        assert required_marker.lower() in preflight_region.lower(), required_marker


def test_ps1_clears_confirmation_token_in_finally() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    finally_index = text.index("finally {")
    finally_block = text[finally_index:finally_index + 400]
    assert "confirmationEnvironmentVariableName" in finally_block
    assert "typedConfirmationToken" in finally_block


def test_ps1_never_hardcodes_a_confirmation_from_chat() -> None:
    text = PS1_PATH.read_text(encoding="utf-8")
    assert "Read-Host" in text
    # The only token compared against is the fixed contract string that
    # matches src/v9_005_stage_a_jpx_probe.CONFIRMATION -- never a
    # session-specific or chat-supplied authorization string.
    assert m.CONFIRMATION in text
