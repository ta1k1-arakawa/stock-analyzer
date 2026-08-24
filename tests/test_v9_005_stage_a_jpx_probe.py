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
    assert len(inventory) == len(m.inventory_months()) * len(m.MONTHLY_COVERAGE_FAMILIES)


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
    assert len(inventory) == 648
    assert len(inventory) == len(m.inventory_months()) * len(m.MONTHLY_COVERAGE_FAMILIES)
    assert all(record["status"] == m.INVENTORY_MISSING for record in inventory)


def test_monthly_coverage_matrix_is_exactly_f2_through_f7() -> None:
    assert len(m.MONTHLY_COVERAGE_FAMILIES) == 6
    assert m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END not in m.MONTHLY_COVERAGE_FAMILIES
    assert set(m.MONTHLY_COVERAGE_FAMILIES) == set(m.SOURCE_FAMILIES) - {m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END}


def test_f1_has_zero_monthly_cells_and_a_separate_terminal_slot() -> None:
    inventory = m.build_source_inventory()
    assert all(record["source_family"] != m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END for record in inventory)
    f1_strategy = m.LOCATOR_STRATEGIES[m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END]
    assert f1_strategy.slot_kind == m.SLOT_KIND_TERMINAL
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_month_locator(m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, m.inventory_months()[0])


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


# --- Reviewed deterministic locator strategy contract (F1-F7) ---------------

def test_no_monthly_auxiliary_slot_kind_exists() -> None:
    assert m.VALID_SLOT_KINDS == {m.SLOT_KIND_MONTHLY, m.SLOT_KIND_YEAR, m.SLOT_KIND_TERMINAL, m.SLOT_KIND_GLOBAL}
    assert "MONTHLY_AUXILIARY" not in m.VALID_SLOT_KINDS
    assert not hasattr(m, "SLOT_KIND_MONTHLY_AUXILIARY")


def test_no_hardcoded_archive_n_locator_anywhere_in_strategies() -> None:
    for strategy in m.LOCATOR_STRATEGIES.values():
        for value in (strategy.root_url, strategy.locator_template):
            if value is not None:
                assert not re.search(r"archive-?\d+", value, re.IGNORECASE), value


def test_f2_monthly_statistics_changes_report_strategy_is_deterministic() -> None:
    strategy = m.LOCATOR_STRATEGIES[m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT]
    assert strategy.slot_kind == m.SLOT_KIND_MONTHLY
    assert strategy.root_url == m.MONTHLY_STATISTICS_ROOT_URL
    assert m.F2_SEMANTIC_ROW_LABEL in strategy.traversal
    assert strategy.auxiliary is False
    for month in m.inventory_months():
        assert m.resolve_month_locator(m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, month) is strategy


def test_f3_delisted_company_archive_years_and_year_object_reuse() -> None:
    strategy = m.LOCATOR_STRATEGIES[m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE]
    assert strategy.slot_kind == m.SLOT_KIND_YEAR
    assert strategy.root_url == m.DELISTED_COMPANY_ROOT_URL
    years = sorted({int(month.split("-")[0]) for month in m.inventory_months()})
    assert years == list(range(2017, 2026))
    # One YEAR object's strategy identically supports every month of its
    # year -- never a per-month refetch.
    for month in m.inventory_months():
        assert m.resolve_month_locator(m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, month) is strategy


def test_f4_ex_rights_split_ratio_strategy_deterministic_and_ratio_unchanged() -> None:
    strategy = m.LOCATOR_STRATEGIES[m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE]
    assert strategy.slot_kind == m.SLOT_KIND_MONTHLY
    assert strategy.root_url == m.MONTHLY_STATISTICS_ROOT_URL
    assert m.F4_SEMANTIC_ROW_LABEL in strategy.traversal
    # F2 and F4 share the same root (Monthly Statistics) but are distinct
    # strategies via distinct semantic rows.
    f2_strategy = m.LOCATOR_STRATEGIES[m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT]
    assert strategy.root_url == f2_strategy.root_url
    assert strategy.traversal != f2_strategy.traversal


def test_f5_slot_kind_monthly_with_auxiliary_flag() -> None:
    strategy = m.LOCATOR_STRATEGIES[m.SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS]
    assert strategy.slot_kind == m.SLOT_KIND_MONTHLY
    assert strategy.auxiliary is True
    assert strategy.root_url == m.LISTING_CO_ROOT_URL


def test_f6_exactly_one_global_strategy() -> None:
    global_strategies = [s for s in m.LOCATOR_STRATEGIES.values() if s.slot_kind == m.SLOT_KIND_GLOBAL]
    assert len(global_strategies) == 1
    strategy = global_strategies[0]
    assert strategy.source_family == m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
    assert strategy.root_url == m.TOPIX_ROOT_URL
    assert m.F6_SEMANTIC_SECTION_LABEL in strategy.traversal


def test_f7_calendar_strategy_uses_exact_bound_template() -> None:
    strategy = m.LOCATOR_STRATEGIES[m.SOURCE_FAMILY_JPX_CALENDAR]
    assert strategy.slot_kind == m.SLOT_KIND_MONTHLY
    assert strategy.root_url is None
    assert strategy.locator_template == m.CALENDAR_MONTHLY_LOCATOR_TEMPLATE
    assert m.resolve_f7_calendar_url(2019, 6) == "https://www.jpx.co.jp/calendar/201906.html"


def test_f7_envelope_is_exactly_2016_09_through_2026_03() -> None:
    months = m.calendar_envelope_months()
    assert months[0] == "2016-09"
    assert months[-1] == "2026-03"
    assert len(months) == 115  # 4 (2016) + 9*12 (2017..2025) + 3 (2026)
    extra = m.calendar_envelope_extra_months()
    assert set(extra) == set(months) - set(m.inventory_months())
    assert "2016-09" in extra and "2026-03" in extra
    assert "2020-06" not in extra  # base-matrix month, not an extra envelope slot


def test_f2_bridge_slots_derive_mechanically_from_terminal_month() -> None:
    assert m.f2_bridge_months("2025-12") == ()
    assert m.f2_bridge_months("2025-01") == ()
    assert m.f2_bridge_months("2026-01") == ("2026-01",)
    assert m.f2_bridge_months("2026-07") == (
        "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07",
    )


def test_locator_contract_completeness_no_longer_requires_a_known_child_url() -> None:
    """The methodology-completeness gate must pass once every family has a
    reviewed root/traversal (or exact template) strategy bound -- it must
    never require that the concrete per-month child URL is already known,
    since that URL is only discoverable by traversing a locked official
    root response at real execution time."""
    for family in m.MONTHLY_COVERAGE_FAMILIES:
        strategy = m.LOCATOR_STRATEGIES[family]
        # No family's strategy embeds a concrete resolved child URL for any
        # specific month/year -- only a root/template plus a traversal rule.
        if strategy.root_url is not None:
            assert re.fullmatch(r"\d{4}(-\d{2})?", strategy.root_url.rsplit("/", 1)[-1]) is None
    m.verify_locator_contract_complete()  # must not raise


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
        locked_index={
            (family, month): object() for family in m.MONTHLY_COVERAGE_FAMILIES for month in m.inventory_months()
        }
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
        locked_index={
            (family, month): object() for family in m.MONTHLY_COVERAGE_FAMILIES for month in m.inventory_months()
        }
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


# --- V9_006_SOURCE_SLOT_LOCATOR: the reviewed contract is now complete -----

def test_locator_contract_is_now_complete() -> None:
    """Ground truth for this remediation: every one of the seven source
    families now has a reviewed deterministic locator strategy bound
    (F1's TERMINAL slot plus F2-F7's monthly/year/global strategies), so
    the pre-network methodology-completeness gate no longer fires. This is
    NOT a claim that any concrete per-month child URL is already known --
    only that a reviewed root/traversal (or F7's exact template) exists
    for every required slot."""
    m.verify_locator_contract_complete()  # must not raise


def _incomplete_locator_strategies() -> dict[str, m.LocatorStrategy]:
    """A deliberately incomplete registry (missing F7's strategy), used
    only to exercise the fail-closed path -- never a claim that the real,
    currently-bound contract is incomplete."""
    strategies = dict(m.LOCATOR_STRATEGIES)
    del strategies[m.SOURCE_FAMILY_JPX_CALENDAR]
    return strategies


def test_locator_contract_incomplete_if_a_family_strategy_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(m, "LOCATOR_STRATEGIES", _incomplete_locator_strategies())
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.verify_locator_contract_complete()
    assert excinfo.value.reason == m.STAGE_A_SOURCE_LOCATOR_CONTRACT_INCOMPLETE
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED


def test_run_stage_a_incomplete_locator_contract_stops_before_any_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercises the CHATGPT_DECISION_REQUIRED stop path with a
    deliberately incomplete registry -- the real, currently-bound registry
    is complete (see test_locator_contract_is_now_complete), so this test
    forces incompleteness rather than relying on today's contract state."""
    monkeypatch.setattr(m, "LOCATOR_STRATEGIES", _incomplete_locator_strategies())
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


# --- V9_006_LOCATOR_IMPL_HIGH_1: acquisition-implementation readiness is a
# distinct, separate, pre-network gate from locator-*methodology*
# completeness above. The reviewed LOCATOR_STRATEGIES registry is complete,
# but no F2-F7 traversal/fetch implementation exists yet, so real Stage-A
# execution must still stop, unconditionally, before touching the
# filesystem, git, or the network.

def test_acquisition_implementation_is_not_yet_complete() -> None:
    """Ground truth for this remediation: unlike the locator-strategy
    registry, the actual F2-F7 acquisition pipeline is NOT yet implemented,
    so this flag must remain False and the readiness check must raise."""
    assert m.ACQUISITION_IMPLEMENTATION_COMPLETE is False
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.verify_acquisition_implementation_ready()
    assert excinfo.value.reason == m.STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED


def test_run_stage_a_valid_confirmation_still_stops_before_any_network_or_git(
    tmp_path: Path,
) -> None:
    """The real, currently-bound locator registry is complete (see
    test_locator_contract_is_now_complete), so a valid confirmation now
    clears the FIRST pre-network gate. It must still be stopped by the
    SEPARATE acquisition-implementation-readiness gate, before output-root
    creation, before any git call, and before any fetcher call -- a
    knowingly incomplete acquisition pipeline must never be allowed to
    reach the network boundary. This exercises the real, unmocked registry
    and the real, unmocked (False) ACQUISITION_IMPLEMENTATION_COMPLETE
    flag -- no monkeypatching of either."""
    calls: list[str] = []
    git_calls: list[list[str]] = []

    def fetcher(url: str) -> tuple[bytes, str]:
        calls.append(url)
        raise AssertionError("must not fetch while acquisition implementation is incomplete")

    def fake_git(args: list[str]) -> str:
        git_calls.append(args)
        raise AssertionError("must not call git while acquisition implementation is incomplete")

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
    assert excinfo.value.reason == m.STAGE_A_ACQUISITION_IMPLEMENTATION_INCOMPLETE
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED
    # Never SOURCE_OR_DATA_FEASIBILITY_FAILURE: that class is reserved for a
    # genuine result after the complete acquisition pipeline actually ran.
    assert excinfo.value.failure_class != m.SOURCE_OR_DATA_FEASIBILITY_FAILURE
    assert calls == []
    assert git_calls == []
    assert not out.exists()  # no durable state created for a stop this early


# --- Regression coverage: with the locator contract now genuinely
# complete, the existing fetch/lock/evidence pipeline below the gate still
# behaves correctly. All fetchers below are synthetic/offline fakes -- no
# real network request is made.

def test_run_stage_a_offline_reports_fail_with_safe_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SAFETY/SCOPE: this test exercises the fetch/lock/evidence pipeline
    # BELOW the V9_006_LOCATOR_IMPL_HIGH_1 acquisition-implementation gate.
    # It forces ACQUISITION_IMPLEMENTATION_COMPLETE True for this test only
    # -- it is not a claim that the real F2-F7 acquisition pipeline exists;
    # the real, unmocked flag is proven False by
    # test_acquisition_implementation_is_not_yet_complete above. Without
    # this forcing, run_stage_a would now stop at the acquisition gate
    # before ever reaching this test's synthetic fetcher.
    monkeypatch.setattr(m, "ACQUISITION_IMPLEMENTATION_COMPLETE", True)
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
    # The pre-network locator-methodology gate now passes (every family has
    # a reviewed strategy), but the underlying 648-record F2-F7 monthly
    # SOURCE_INVENTORY is still empty here: this orchestration only fetches
    # F1's terminal snapshot and the calendar page, never any real F2-F7
    # object (that traversal-fetch implementation is a separate, future,
    # authorized task). So this remains the honest FAIL outcome.
    assert summary["status"] == "FAIL"
    assert summary["failure_class"] == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE
    assert summary["required_inventory_missing_count"] == 648
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


def test_run_stage_a_wrong_signal_grid_blob_stops_before_any_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SAFETY/SCOPE: forces ACQUISITION_IMPLEMENTATION_COMPLETE True so this
    # test can still reach and exercise the (later) signal-grid-binding
    # check below the V9_006_LOCATOR_IMPL_HIGH_1 gate -- see the identical
    # note on test_run_stage_a_offline_reports_fail_with_safe_evidence.
    monkeypatch.setattr(m, "ACQUISITION_IMPLEMENTATION_COMPLETE", True)
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
    # SAFETY: the real, currently-bound locator registry is complete (see
    # test_locator_contract_is_now_complete), so calling the CLI's
    # production entrypoint with a valid confirmation and an unmocked
    # registry would proceed past the gate and attempt a REAL network
    # request via urllib. Force the registry incomplete here so this test
    # exercises the CHATGPT_DECISION_REQUIRED path while guaranteeing the
    # CLI never reaches the production fetcher.
    monkeypatch.setattr(m, "LOCATOR_STRATEGIES", _incomplete_locator_strategies())
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
