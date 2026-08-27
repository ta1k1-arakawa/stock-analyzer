from __future__ import annotations

import importlib
import inspect
import json
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

import src.v9_005_stage_a_jpx_probe as m
import src.v9_005_stage_a_semantics as sem

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 8, 24, 3, 0, tzinfo=timezone.utc)


def _clock() -> datetime:
    return NOW


def _no_sleep(_seconds: int) -> None:
    return None


def _slot_id(source_family: str, month: str, suffix: str = "one") -> str:
    return m.source_object_slot_id(source_family, month, f"https://www.jpx.co.jp/{suffix}")


def _full_available_inventory() -> list[dict[str, object]]:
    return [
        {
            "source_family": family,
            "month": month,
            "status": m.INVENTORY_AVAILABLE,
            "source_object_slot_ids": ["a" * 64],
        }
        for family in m.MONTHLY_COVERAGE_FAMILIES
        for month in m.inventory_months()
    ]


def _lock_coverage_object(root: Path, source_family: str, month: str, suffix: str = "one") -> str:
    requested_url = f"https://www.jpx.co.jp/coverage/{suffix}"
    m.lock_first_complete_payload(
        root,
        source_family=source_family,
        applicable_period=month,
        requested_url=requested_url,
        fetch_result=m.FetchResult(b"coverage-bytes", requested_url, 200),
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )
    return m.source_object_slot_id(source_family, month, requested_url)


def _production_script_module() -> object:
    scripts_directory = str(ROOT / "scripts")
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    return importlib.reload(importlib.import_module("run_v9_005_stage_a_jpx_probe"))


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
    def fetcher(_url: str) -> m.FetchResult:
        return m.FetchResult(b"payload", "https://evil.example/redirected", 200)

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.fetch_once_with_retry("https://www.jpx.co.jp/x", fetcher, _no_sleep)
    assert excinfo.value.reason == "OFF_DOMAIN_REDIRECT_REJECTED"


def test_off_domain_request_rejected_before_any_fetch_call() -> None:
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return m.FetchResult(b"payload", url, 200)

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
        fetch_result=m.FetchResult(b"raw-bytes", m.CALENDAR_PAGE_URL, 206),
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )
    assert locked["raw"] == b"raw-bytes"
    assert locked["sha256"] == m.sha256_bytes(b"raw-bytes")
    assert locked["byte_length"] == len(b"raw-bytes")
    assert locked["http_status"] == 206
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
        fetch_result=m.FetchResult(b"first", m.CALENDAR_PAGE_URL, 200),
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )
    m.lock_first_complete_payload(root, **kwargs)
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.lock_first_complete_payload(
            root, **{**kwargs, "fetch_result": m.FetchResult(b"second-attempt", m.CALENDAR_PAGE_URL, 200)},
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    # never silently overwritten
    reread = m.read_locked_payload(root, m.SOURCE_FAMILY_JPX_CALENDAR, "CURRENT", m.CALENDAR_PAGE_URL)
    assert reread["raw"] == b"first"


def test_same_locked_payload_reprocessed_not_refetched(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return m.FetchResult(b"payload-one", url, 201)

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
    assert locked_first["http_status"] == 201


def test_raw_provenance_rejects_orphan_bin_and_orphan_metadata(tmp_path: Path) -> None:
    bin_root = m.initialize_output_root(tmp_path / "bin-out")
    (bin_root / "raw" / "orphan.bin").write_bytes(b"orphan")
    assert m.verify_raw_provenance(bin_root) is False

    meta_root = m.initialize_output_root(tmp_path / "meta-out")
    (meta_root / "raw" / "orphan.json").write_text("{}", encoding="utf-8")
    assert m.verify_raw_provenance(meta_root) is False


def test_raw_provenance_rejects_mismatched_hash_and_length(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = m.lock_first_complete_payload(
        root,
        source_family=m.SOURCE_FAMILY_JPX_CALENDAR,
        applicable_period="CURRENT",
        requested_url=m.CALENDAR_PAGE_URL,
        fetch_result=m.FetchResult(b"raw-bytes", m.CALENDAR_PAGE_URL, 200),
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )
    key = m._record_key(locked["source_family"], locked["applicable_period"], locked["requested_url"])
    _raw_path, meta_path = m._raw_paths(root, key)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["byte_length"] += 1
    meta_path.write_bytes(m.canonical_bytes(metadata))
    assert m.verify_raw_provenance(root) is False


def test_raw_lock_api_requires_coupled_fetch_result() -> None:
    parameters = inspect.signature(m.lock_first_complete_payload).parameters
    assert "fetch_result" in parameters
    assert parameters["fetch_result"].default is inspect.Parameter.empty
    assert "payload" not in parameters
    assert "resolved_url" not in parameters
    assert "http_status" not in parameters


@pytest.mark.parametrize(
    "timestamp",
    [
        "",
        "garbage",
        "2026-08-24 03:00:00Z",
        "2026-08-24T03:00:00+00:00",
        "2026-08-24T03:00:00.123Z",
        "2026-02-30T03:00:00Z",
    ],
)
def test_raw_lock_rejects_malformed_timestamp_at_write(tmp_path: Path, timestamp: str) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    with pytest.raises(m.V9005StageABlocked):
        m.lock_first_complete_payload(
            root,
            source_family=m.SOURCE_FAMILY_JPX_CALENDAR,
            applicable_period="CURRENT",
            requested_url=m.CALENDAR_PAGE_URL,
            fetch_result=m.FetchResult(b"raw-bytes", m.CALENDAR_PAGE_URL, 200),
            retrieval_timestamp_utc=timestamp,
        )


def test_raw_provenance_and_read_reject_malformed_persisted_timestamp(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = m.lock_first_complete_payload(
        root,
        source_family=m.SOURCE_FAMILY_JPX_CALENDAR,
        applicable_period="CURRENT",
        requested_url=m.CALENDAR_PAGE_URL,
        fetch_result=m.FetchResult(b"raw-bytes", m.CALENDAR_PAGE_URL, 200),
        retrieval_timestamp_utc="2026-08-24T03:00:00Z",
    )
    key = m._record_key(locked["source_family"], locked["applicable_period"], locked["requested_url"])
    _raw_path, meta_path = m._raw_paths(root, key)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["retrieval_timestamp_utc"] = "garbage"
    meta_path.write_bytes(m.canonical_bytes(metadata))
    assert m.verify_raw_provenance(root) is False
    with pytest.raises(m.V9005StageABlocked):
        m.read_locked_payload(root, m.SOURCE_FAMILY_JPX_CALENDAR, "CURRENT", m.CALENDAR_PAGE_URL)


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
    assert all(record["source_object_slot_ids"] == [] for record in inventory)


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


def test_source_object_slot_id_reuses_the_existing_raw_lock_key() -> None:
    slot_id = m.source_object_slot_id("FAMILY", "2020-01", "https://www.jpx.co.jp/object")
    assert slot_id == m._record_key("FAMILY", "2020-01", "https://www.jpx.co.jp/object")
    assert re.fullmatch(r"[0-9a-f]{64}", slot_id)


def test_inventory_rejects_nonempty_slot_ids_without_an_output_root() -> None:
    month = m.inventory_months()[0]
    family = m.SOURCE_FAMILY_JPX_CALENDAR
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.build_source_inventory(coverage_references={(family, month): ["a" * 64]})
    assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


def test_inventory_rejects_nonexistent_or_arbitrary_url_slot_id(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    month = m.inventory_months()[0]
    family = m.SOURCE_FAMILY_JPX_CALENDAR
    for slot_id in ("a" * 64, _slot_id(family, month)):
        with pytest.raises(m.V9005StageABlocked) as excinfo:
            m.build_source_inventory(coverage_references={(family, month): [slot_id]}, output_root=root)
        assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


def test_inventory_available_only_with_a_genuine_matching_raw_lock(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    month = m.inventory_months()[0]
    family = m.SOURCE_FAMILY_JPX_CALENDAR
    slot_id = _lock_coverage_object(root, family, month)
    inventory = m.build_source_inventory(coverage_references={(family, month): [slot_id]}, output_root=root)
    record = next(r for r in inventory if r["source_family"] == family and r["month"] == month)
    assert record["status"] == m.INVENTORY_AVAILABLE
    assert record["source_object_slot_ids"] == [slot_id]
    other = next(r for r in inventory if r["source_family"] == family and r["month"] != month)
    assert other["status"] == m.INVENTORY_MISSING


def test_inventory_slot_id_references_are_sorted_deduplicated_and_empty_is_missing(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    month = m.inventory_months()[0]
    family = m.SOURCE_FAMILY_JPX_CALENDAR
    first = _lock_coverage_object(root, family, month, "a")
    second = _lock_coverage_object(root, family, month, "b")
    inventory = m.build_source_inventory(
        coverage_references={(family, month): [second, first, second]}, output_root=root,
    )
    record = next(r for r in inventory if r["source_family"] == family and r["month"] == month)
    assert record["status"] == m.INVENTORY_AVAILABLE
    assert record["source_object_slot_ids"] == sorted({first, second})
    missing = m.build_source_inventory(coverage_references={(family, month): []})
    assert next(r for r in missing if r["source_family"] == family and r["month"] == month)["status"] == m.INVENTORY_MISSING


def test_inventory_rejects_verified_lock_from_the_wrong_family(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    month = m.inventory_months()[0]
    slot_id = _lock_coverage_object(root, m.SOURCE_FAMILY_JPX_CALENDAR, month)
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.build_source_inventory(
            coverage_references={(m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, month): [slot_id]},
            output_root=root,
        )
    assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


@pytest.mark.parametrize("corruption", ["sha256", "byte_length", "retrieval_timestamp_utc"])
def test_inventory_rejects_corrupt_referenced_raw_lock(tmp_path: Path, corruption: str) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    family, month = m.SOURCE_FAMILY_JPX_CALENDAR, m.inventory_months()[0]
    slot_id = _lock_coverage_object(root, family, month)
    _raw_path, meta_path = m._raw_paths(root, slot_id)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata[corruption] = {"sha256": "0" * 64, "byte_length": 999, "retrieval_timestamp_utc": "garbage"}[corruption]
    meta_path.write_bytes(m.canonical_bytes(metadata))
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.build_source_inventory(coverage_references={(family, month): [slot_id]}, output_root=root)
    assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


@pytest.mark.parametrize("suffix", [".bin", ".json"])
def test_inventory_rejects_orphan_referenced_lock(tmp_path: Path, suffix: str) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    family, month = m.SOURCE_FAMILY_JPX_CALENDAR, m.inventory_months()[0]
    slot_id = "a" * 64
    (root / "raw" / f"{slot_id}{suffix}").write_bytes(b"orphan")
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.build_source_inventory(coverage_references={(family, month): [slot_id]}, output_root=root)
    assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


@pytest.mark.parametrize("slot_ids", [object(), [object()], ["A" * 64], ["a" * 63], ["g" * 64]])
def test_inventory_rejects_arbitrary_or_invalid_slot_ids(slot_ids: object) -> None:
    family, month = m.SOURCE_FAMILY_JPX_CALENDAR, m.inventory_months()[0]
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.build_source_inventory(coverage_references={(family, month): slot_ids})  # type: ignore[dict-item]
    assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


@pytest.mark.parametrize(
    "key",
    [
        (m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, "2017-01"),
        ("UNKNOWN_FAMILY", "2017-01"),
        (m.SOURCE_FAMILY_JPX_CALENDAR, "2016-12"),
    ],
)
def test_inventory_rejects_non_base_coverage_keys(key: tuple[str, str]) -> None:
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.build_source_inventory(coverage_references={key: ["a" * 64]})
    assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


def test_unknown_family_or_month_is_ambiguous_fail_closed() -> None:
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_month_locator("NOT_A_REAL_FAMILY", m.inventory_months()[0])
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_month_locator(m.SOURCE_FAMILY_JPX_CALENDAR, "2099-01")


def _year_selector_html(*links: tuple[str, str]) -> bytes:
    return ("<html><body>" + "".join(f'<a href="{href}">{text}</a>' for href, text in links) + "</body></html>").encode()


def _monthly_statistics_year_html(*, f2_href: str = "f2.xlsx", f4_href: str = "f4.xlsx") -> bytes:
    return (
        "<table><tr><th>Report</th><th>2020-03</th></tr>"
        f"<tr><th>{m.F2_SEMANTIC_ROW_LABEL}</th><td><a href=\"{f2_href}\">F2</a></td></tr>"
        f"<tr><th>{m.F4_SEMANTIC_ROW_LABEL}</th><td><a href=\"{f4_href}\">F4</a></td></tr>"
        "</table>"
    ).encode()


def test_monthly_statistics_year_selector_resolves_one_exact_year() -> None:
    resolved = m.resolve_monthly_statistics_year_page_url(
        _year_selector_html(("archive/2020.html", "2020")), m.MONTHLY_STATISTICS_ROOT_URL, 2020,
    )
    assert resolved == "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archive/2020.html"


@pytest.mark.parametrize(
    "links",
    [
        (("2020-a.html", "2020"), ("2020-b.html", "2020")),
        (("2021.html", "2021"),),
        (("https://evil.example/2020.html", "2020"),),
    ],
)
def test_monthly_statistics_year_selector_fails_closed_on_ambiguous_missing_or_unsafe_links(
    links: tuple[tuple[str, str], ...],
) -> None:
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_monthly_statistics_year_page_url(_year_selector_html(*links), m.MONTHLY_STATISTICS_ROOT_URL, 2020)


def test_monthly_statistics_year_selector_rejects_malformed_html() -> None:
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_monthly_statistics_year_page_url(b'<a href="2020.html">2020', m.MONTHLY_STATISTICS_ROOT_URL, 2020)


@pytest.mark.parametrize(
    "page",
    [
        b"</a>",
        b'<a href="2020.html">outer<a href="other.html">2020</a></a>',
        b"<tr><a href=\"2020.html\">2020</a></tr>",
    ],
)
def test_monthly_statistics_year_selector_rejects_invalid_relevant_tag_structure(page: bytes) -> None:
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.resolve_monthly_statistics_year_page_url(page, m.MONTHLY_STATISTICS_ROOT_URL, 2020)
    assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


def test_monthly_statistics_f2_and_f4_rows_resolve_only_their_exact_children() -> None:
    page = _monthly_statistics_year_html()
    page_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archive/2020.html"
    assert m.resolve_monthly_statistics_evidence_url(
        page, page_url, m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2020-03", selected_year=2020,
    ) == "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archive/f2.xlsx"
    assert m.resolve_monthly_statistics_evidence_url(
        page, page_url, m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, "2020-03", selected_year=2020,
    ) == "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/archive/f4.xlsx"


@pytest.mark.parametrize(
    "page",
    [
        b"<table><tr><th>Report</th><th>2020-03</th></tr></table>",
        (
            f"<table><tr><th>Report</th><th>2020-03</th></tr>"
            f"<tr><th>{m.F2_SEMANTIC_ROW_LABEL}</th><td><a href=\"a.xlsx\">a</a></td></tr>"
            f"<tr><th>{m.F2_SEMANTIC_ROW_LABEL}</th><td><a href=\"b.xlsx\">b</a></td></tr></table>"
        ).encode(),
        (
            f"<table><tr><th>Report</th><th>2020-04</th></tr>"
            f"<tr><th>{m.F2_SEMANTIC_ROW_LABEL}</th><td><a href=\"a.xlsx\">a</a></td></tr></table>"
        ).encode(),
        (
            f"<table><tr><th>Report</th><th>2020-03</th></tr>"
            f"<tr><th>{m.F2_SEMANTIC_ROW_LABEL}</th><td><a href=\"a.xlsx\">a</a><a href=\"b.xlsx\">b</a></td></tr></table>"
        ).encode(),
    ],
)
def test_monthly_statistics_evidence_traversal_rejects_missing_or_ambiguous_structure(page: bytes) -> None:
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_monthly_statistics_evidence_url(
            page, "https://www.jpx.co.jp/monthly/2020.html", m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
            "2020-03", selected_year=2020,
        )


@pytest.mark.parametrize(
    "family, month, selected_year",
    [
        ("UNSUPPORTED", "2020-03", 2020),
        (m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "bad", 2020),
        (m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2020-03", 2021),
    ],
)
def test_monthly_statistics_evidence_traversal_rejects_invalid_inputs(
    family: str, month: str, selected_year: int,
) -> None:
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_monthly_statistics_evidence_url(
            _monthly_statistics_year_html(), "https://www.jpx.co.jp/monthly/2020.html", family, month,
            selected_year=selected_year,
        )


def test_monthly_statistics_evidence_traversal_rejects_unsafe_child_url() -> None:
    with pytest.raises(m.V9005StageABlocked):
        m.resolve_monthly_statistics_evidence_url(
            _monthly_statistics_year_html(f2_href="https://evil.example/f2.xlsx"),
            "https://www.jpx.co.jp/monthly/2020.html", m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
            "2020-03", selected_year=2020,
        )


@pytest.mark.parametrize(
    "page",
    [
        b"<table><tr><th>x</td></tr></table>",
        b"<table><tr><td>x</th></tr></table>",
        b"</td>",
        b"<td>outside</td>",
        b"<tr><td>outside</td></tr>",
        b"<table><tr><td>outer<td>nested</td></td></tr></table>",
        b"<table><tr><th>Report</th><th>2020-03</th></tr></table></tr>",
        b"<table><tr><td>premature</table></td></tr>",
    ],
)
def test_monthly_statistics_evidence_traversal_rejects_malformed_relevant_table_structure(page: bytes) -> None:
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.resolve_monthly_statistics_evidence_url(
            page, "https://www.jpx.co.jp/monthly/2020.html", m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
            "2020-03", selected_year=2020,
        )
    assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


def test_monthly_statistics_traversal_does_not_hardcode_archive_numbering() -> None:
    source = (ROOT / "src" / "v9_005_stage_a_jpx_probe.py").read_text(encoding="utf-8")
    assert re.search(r"archive-?\\d+", source, re.IGNORECASE) is None


def _monthly_statistics_acquisition_year_html(year: int) -> bytes:
    return (
        f"<table><tr><th>Report</th><th>{year}-03</th><th>{year}-04</th></tr>"
        f"<tr><th>{m.F2_SEMANTIC_ROW_LABEL}</th><td><a href=\"f2-{year}-03.xlsx\">F2</a></td>"
        f"<td><a href=\"f2-{year}-04.xlsx\">F2</a></td></tr>"
        f"<tr><th>{m.F4_SEMANTIC_ROW_LABEL}</th><td><a href=\"f4-{year}-03.xlsx\">F4</a></td>"
        f"<td><a href=\"f4-{year}-04.xlsx\">F4</a></td></tr></table>"
    ).encode()


def _monthly_statistics_acquisition_responses() -> dict[str, m.FetchResult]:
    root = m.MONTHLY_STATISTICS_ROOT_URL
    year_urls = {year: f"https://www.jpx.co.jp/english/markets/statistics-equities/monthly/{year}.html" for year in (2020, 2021)}
    responses: dict[str, m.FetchResult] = {
        root: m.FetchResult(b'<a href="2020.html">2020</a><a href="2021.html">2021</a>', root, 200),
    }
    for year, year_url in year_urls.items():
        responses[year_url] = m.FetchResult(_monthly_statistics_acquisition_year_html(year), year_url, 200)
        for family in ("f2", "f4"):
            for month in ("03", "04"):
                child_url = f"https://www.jpx.co.jp/english/markets/statistics-equities/monthly/{family}-{year}-{month}.xlsx"
                responses[child_url] = m.FetchResult(f"{family}-{year}-{month}".encode(), child_url, 206)
    return responses


def _monthly_statistics_enumeration_responses(years: range) -> dict[str, m.FetchResult]:
    responses: dict[str, m.FetchResult] = {}
    root_links = "".join(f'<a href="{year}.html">{year}</a>' for year in years)
    responses[m.MONTHLY_STATISTICS_ROOT_URL] = m.FetchResult(root_links.encode(), m.MONTHLY_STATISTICS_ROOT_URL, 200)
    for year in years:
        year_url = f"https://www.jpx.co.jp/english/markets/statistics-equities/monthly/{year}.html"
        headers = "".join(f"<th>{year}-{month:02d}</th>" for month in range(1, 13))
        f2_cells = "".join(f'<td><a href="f2-{year}-{month:02d}.xlsx">F2</a></td>' for month in range(1, 13))
        f4_cells = "".join(f'<td><a href="f4-{year}-{month:02d}.xlsx">F4</a></td>' for month in range(1, 13))
        html = (
            f"<table><tr><th>Report</th>{headers}</tr>"
            f"<tr><th>{m.F2_SEMANTIC_ROW_LABEL}</th>{f2_cells}</tr>"
            f"<tr><th>{m.F4_SEMANTIC_ROW_LABEL}</th>{f4_cells}</tr></table>"
        ).encode()
        responses[year_url] = m.FetchResult(html, year_url, 200)
        for family in ("f2", "f4"):
            for month in range(1, 13):
                child_url = f"https://www.jpx.co.jp/english/markets/statistics-equities/monthly/{family}-{year}-{month:02d}.xlsx"
                responses[child_url] = m.FetchResult(child_url.encode(), child_url, 200)
    return responses


def _f3_year_responses(root_final_url: str) -> dict[str, m.FetchResult]:
    root_html = "".join(f'<a href="{year}.html"> {year} </a>' for year in range(2017, 2026)).encode()
    responses = {
        m.DELISTED_COMPANY_ROOT_URL: m.FetchResult(root_html, root_final_url, 200),
    }
    base = root_final_url.rsplit("/", 1)[0]
    for year in range(2017, 2026):
        year_url = f"{base}/{year}.html"
        responses[year_url] = m.FetchResult(f"year-{year}".encode(), year_url, 200)
    return responses


def _f7_calendar_responses() -> dict[str, m.FetchResult]:
    responses: dict[str, m.FetchResult] = {}
    for month in m.calendar_envelope_months():
        year, month_number = map(int, month.split("-"))
        requested_url = m.resolve_f7_calendar_url(year, month_number)
        resolved_url = requested_url
        if month == "2017-01":
            resolved_url = "https://www.jpx.co.jp/calendar/final-201701.html"
        responses[requested_url] = m.FetchResult(month.encode(), resolved_url, 200)
    return responses


def test_f7_calendar_acquisition_uses_only_template_urls_and_separates_extras(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    responses = _f7_calendar_responses()
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return responses[url]

    result = m.acquire_f7_required_slots(root, fetcher=fetcher, sleep=_no_sleep, clock=_clock)
    assert len(m.calendar_envelope_months()) == 115
    assert len(result.base_coverage_references) == 108
    assert tuple(result.envelope_extra_references) == (
        "2016-09", "2016-10", "2016-11", "2016-12", "2026-01", "2026-02", "2026-03",
    )
    for month in ("2016-09", "2017-01", "2025-12", "2026-03"):
        year, month_number = map(int, month.split("-"))
        assert m.resolve_f7_calendar_url(year, month_number) in calls
    assert all(url.startswith("https://www.jpx.co.jp/calendar/") for url in calls)
    verified = m._verified_raw_lock_index(root)
    for month in m.calendar_envelope_months():
        slot_ids = result.base_coverage_references.get((m.SOURCE_FAMILY_JPX_CALENDAR, month), result.envelope_extra_references.get(month))
        assert slot_ids is not None and len(slot_ids) == 1
        slot_id = slot_ids[0]
        assert verified[slot_id]["source_family"] == m.SOURCE_FAMILY_JPX_CALENDAR
        assert verified[slot_id]["applicable_period"] == month
    requested_2017 = m.resolve_f7_calendar_url(2017, 1)
    redirected_slot = result.base_coverage_references[(m.SOURCE_FAMILY_JPX_CALENDAR, "2017-01")][0]
    assert redirected_slot == m.source_object_slot_id(m.SOURCE_FAMILY_JPX_CALENDAR, "2017-01", requested_2017)
    assert redirected_slot != m.source_object_slot_id(m.SOURCE_FAMILY_JPX_CALENDAR, "2017-01", responses[requested_2017].resolved_url)
    inventory = m.build_source_inventory(result.base_coverage_references, output_root=root)
    assert len(inventory) == 648
    assert sum(record["status"] == m.INVENTORY_AVAILABLE and record["source_family"] == m.SOURCE_FAMILY_JPX_CALENDAR for record in inventory) == 108
    assert all(record["status"] == m.INVENTORY_MISSING for record in inventory if record["source_family"] != m.SOURCE_FAMILY_JPX_CALENDAR)
    mismatched = dict(result.base_coverage_references)
    mismatched[(m.SOURCE_FAMILY_JPX_CALENDAR, "2017-01")] = result.base_coverage_references[(m.SOURCE_FAMILY_JPX_CALENDAR, "2017-02")]
    with pytest.raises(m.V9005StageABlocked):
        m._validate_f7_required_slot_references(root, mismatched, result.envelope_extra_references)
    repeated = m.acquire_f7_required_slots(root, fetcher=fetcher, sleep=_no_sleep, clock=_clock)
    assert repeated.network_attempt_count == 0


def test_f7_calendar_acquisition_fails_closed_without_refetch_or_partial_result(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    first_url = m.resolve_f7_calendar_url(2016, 9)
    first_slot = m.source_object_slot_id(m.SOURCE_FAMILY_JPX_CALENDAR, "2016-09", first_url)
    (root / "raw" / f"{first_slot}.bin").write_bytes(b"orphan")
    calls: list[str] = []

    def no_refetch(url: str) -> m.FetchResult:
        calls.append(url)
        raise AssertionError("orphan F7 lock must not be refetched")

    with pytest.raises(m.V9005StageABlocked):
        m.acquire_f7_required_slots(root, fetcher=no_refetch, sleep=_no_sleep, clock=_clock)
    assert calls == []

    healthy_root = m.initialize_output_root(tmp_path / "healthy")
    responses = _f7_calendar_responses()

    def failing_fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        if url == m.resolve_f7_calendar_url(2020, 6):
            raise m.V9005StageABlocked(m.IMPLEMENTATION_FAILURE)
        return responses[url]

    with pytest.raises(m.V9005StageABlocked):
        m.acquire_f7_required_slots(healthy_root, fetcher=failing_fetcher, sleep=_no_sleep, clock=_clock)
    assert m.resolve_f7_calendar_url(2020, 7) not in calls


def test_delisted_company_year_traversal_is_unique_safe_and_total() -> None:
    root_url = "https://www.jpx.co.jp/english/listing/stocks/delisted/archive/index.html"
    assert m.resolve_delisted_company_year_url(b'<a href="2017.html"> 2017 </a>', root_url, 2017) == (
        "https://www.jpx.co.jp/english/listing/stocks/delisted/archive/2017.html"
    )
    for raw in (
        b'<a href="2017.html">2017</a><a href="duplicate.html">2017</a>',
        b'<a href="2018.html">2018</a>',
        b'<a href="https://evil.example/2017.html">2017</a>',
        b'<a href="2017.html">2017',
        b'</a>',
        b'<a href="outer.html">2017<a href="inner.html">2017</a></a>',
    ):
        with pytest.raises(m.V9005StageABlocked) as excinfo:
            m.resolve_delisted_company_year_url(raw, root_url, 2017)
        assert excinfo.value.reason == m.IMPLEMENTATION_FAILURE


def test_f3_year_acquisition_fans_out_verified_year_locks_and_reuses_them(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    root_final_url = "https://www.jpx.co.jp/english/listing/stocks/delisted/archive/redirected/index.html"
    responses = _f3_year_responses(root_final_url)
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return responses[url]

    result = m.acquire_f3_required_slots(root, fetcher=fetcher, sleep=_no_sleep, clock=_clock)
    assert result.network_attempt_count == 10
    assert len(result.base_coverage_references) == 108
    root_lock = m.read_locked_payload(
        root, m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE,
        m.DELISTED_COMPANY_DISCOVERY_ROOT, m.DELISTED_COMPANY_ROOT_URL,
    )
    assert root_lock is not None
    assert root_lock["source_family"] == m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE
    assert root_lock["applicable_period"] == m.DELISTED_COMPANY_DISCOVERY_ROOT
    assert root_lock["requested_url"] == m.DELISTED_COMPANY_ROOT_URL
    redirected_2017 = "https://www.jpx.co.jp/english/listing/stocks/delisted/archive/redirected/2017.html"
    assert redirected_2017 in calls
    assert "https://www.jpx.co.jp/english/listing/stocks/delisted/2017.html" not in calls
    root_slot = m.source_object_slot_id(
        m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, m.DELISTED_COMPANY_DISCOVERY_ROOT, m.DELISTED_COMPANY_ROOT_URL,
    )
    returned_ids = {slot_id for slot_ids in result.base_coverage_references.values() for slot_id in slot_ids}
    assert root_slot not in returned_ids
    assert len(returned_ids) == 9
    verified = m._verified_raw_lock_index(root)
    for year in range(2017, 2026):
        month_ids = {
            result.base_coverage_references[(m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, f"{year}-{month:02d}")]
            for month in range(1, 13)
        }
        assert len(month_ids) == 1
        slot_id = next(iter(month_ids))[0]
        assert verified[slot_id]["source_family"] == m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE
        assert verified[slot_id]["applicable_period"] == str(year)
    inventory = m.build_source_inventory(result.base_coverage_references, output_root=root)
    assert len(inventory) == 648
    assert sum(record["status"] == m.INVENTORY_AVAILABLE and record["source_family"] == m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE for record in inventory) == 108
    assert all(record["status"] == m.INVENTORY_MISSING for record in inventory if record["source_family"] != m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE)
    repeated = m.acquire_f3_required_slots(root, fetcher=fetcher, sleep=_no_sleep, clock=_clock)
    assert repeated.network_attempt_count == 0


def test_f3_year_acquisition_fails_closed_for_corrupt_locks_or_a_year_failure(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    calls: list[str] = []
    root_slot = m.source_object_slot_id(
        m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, m.DELISTED_COMPANY_DISCOVERY_ROOT, m.DELISTED_COMPANY_ROOT_URL,
    )
    (root / "raw" / f"{root_slot}.bin").write_bytes(b"orphan")

    def no_fetch(url: str) -> m.FetchResult:
        calls.append(url)
        raise AssertionError("corrupt root must not be refetched")

    with pytest.raises(m.V9005StageABlocked):
        m.acquire_f3_required_slots(root, fetcher=no_fetch, sleep=_no_sleep, clock=_clock)
    assert calls == []

    healthy_root = m.initialize_output_root(tmp_path / "healthy")
    root_final_url = "https://www.jpx.co.jp/english/listing/stocks/delisted/archive/index.html"
    responses = _f3_year_responses(root_final_url)

    def failing_fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        if url.endswith("/2020.html"):
            raise m.V9005StageABlocked(m.IMPLEMENTATION_FAILURE)
        return responses[url]

    with pytest.raises(m.V9005StageABlocked):
        m.acquire_f3_required_slots(healthy_root, fetcher=failing_fetcher, sleep=_no_sleep, clock=_clock)
    assert not any(url.endswith("/2021.html") for url in calls)


def test_f3_year_acquisition_never_repairs_a_corrupt_year_lock(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    root_final_url = "https://www.jpx.co.jp/english/listing/stocks/delisted/archive/index.html"
    responses = _f3_year_responses(root_final_url)

    def fetcher(url: str) -> m.FetchResult:
        return responses[url]

    m.acquire_f3_required_slots(root, fetcher=fetcher, sleep=_no_sleep, clock=_clock)
    year_url = "https://www.jpx.co.jp/english/listing/stocks/delisted/archive/2017.html"
    year_slot = m.source_object_slot_id(m.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE, "2017", year_url)
    (root / "raw" / f"{year_slot}.json").write_text("{}", encoding="utf-8")
    calls: list[str] = []

    def no_refetch(url: str) -> m.FetchResult:
        calls.append(url)
        raise AssertionError("corrupt year lock must not be refetched")

    with pytest.raises(m.V9005StageABlocked):
        m.acquire_f3_required_slots(root, fetcher=no_refetch, sleep=_no_sleep, clock=_clock)
    assert calls == []


def test_f2_f4_required_slot_enumeration_is_exact_reusable_and_keeps_bridge_separate(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    responses = _monthly_statistics_enumeration_responses(range(2017, 2027))
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return responses[url]

    result = m.acquire_f2_f4_required_slots(
        root, terminal_month="2025-12", fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    expected_base_keys = {
        (family, month)
        for month in m.inventory_months()
        for family in (m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE)
    }
    assert set(result.base_coverage_references) == expected_base_keys
    assert len(result.base_coverage_references) == 216
    assert result.f2_bridge_references == {}
    assert all(len(slot_ids) == 1 for slot_ids in result.base_coverage_references.values())
    inventory = m.build_source_inventory(result.base_coverage_references, output_root=root)
    assert len(inventory) == 648
    assert sum(record["status"] == m.INVENTORY_AVAILABLE and record["source_family"] == m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT for record in inventory) == 108
    assert sum(record["status"] == m.INVENTORY_AVAILABLE and record["source_family"] == m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE for record in inventory) == 108
    assert all(record["status"] == m.INVENTORY_MISSING for record in inventory if record["source_family"] not in {
        m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
    })
    support_ids = {
        m.source_object_slot_id(m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, m.MONTHLY_STATISTICS_DISCOVERY_ROOT, m.MONTHLY_STATISTICS_ROOT_URL),
        *(m.source_object_slot_id(m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, m.monthly_statistics_discovery_year_period(year), f"https://www.jpx.co.jp/english/markets/statistics-equities/monthly/{year}.html") for year in range(2017, 2027)),
    }
    returned_ids = {slot_id for slot_ids in result.base_coverage_references.values() for slot_id in slot_ids}
    assert not returned_ids & support_ids
    assert calls.count(m.MONTHLY_STATISTICS_ROOT_URL) == 1
    for year in range(2017, 2026):
        assert calls.count(f"https://www.jpx.co.jp/english/markets/statistics-equities/monthly/{year}.html") == 1
    repeat = m.acquire_f2_f4_required_slots(
        root, terminal_month="2025-12", fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert repeat.network_attempt_count == 0
    bridge = m.acquire_f2_f4_required_slots(
        root, terminal_month="2026-03", fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert tuple(bridge.f2_bridge_references) == ("2026-01", "2026-02", "2026-03")
    assert all(len(slot_ids) == 1 for slot_ids in bridge.f2_bridge_references.values())
    verified = m._verified_raw_lock_index(root)
    for month, (slot_id,) in bridge.f2_bridge_references.items():
        assert verified[slot_id]["source_family"] == m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT
        assert verified[slot_id]["applicable_period"] == month
    assert not ({slot_id for ids in bridge.f2_bridge_references.values() for slot_id in ids} & support_ids)
    extension = m.acquire_f2_f4_required_slots(
        root, terminal_month="2026-04", fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert extension.network_attempt_count == 1
    assert calls.count(m.MONTHLY_STATISTICS_ROOT_URL) == 1
    assert calls.count("https://www.jpx.co.jp/english/markets/statistics-equities/monthly/2026.html") == 1


def test_f2_f4_required_slot_enumeration_fails_closed_before_or_during_acquisition(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    calls: list[str] = []

    def no_fetch(url: str) -> m.FetchResult:
        calls.append(url)
        raise AssertionError("malformed terminal month must fail before acquisition")

    with pytest.raises(m.V9005StageABlocked):
        m.acquire_f2_f4_required_slots(root, terminal_month="bad", fetcher=no_fetch, sleep=_no_sleep, clock=_clock)
    assert calls == []
    responses = _monthly_statistics_enumeration_responses(range(2017, 2026))

    def failing_fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        if url.endswith("f2-2017-02.xlsx"):
            raise m.V9005StageABlocked(m.IMPLEMENTATION_FAILURE)
        return responses[url]

    with pytest.raises(m.V9005StageABlocked):
        m.acquire_f2_f4_required_slots(root, terminal_month="2025-12", fetcher=failing_fetcher, sleep=_no_sleep, clock=_clock)
    assert not any(url.endswith("f4-2017-02.xlsx") for url in calls)


def test_f2_f4_required_slot_validation_rejects_a_family_or_period_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    monkeypatch.setattr(m, "inventory_months", lambda: ("2020-03",))
    f2_url = "https://www.jpx.co.jp/f2.xlsx"
    f4_url = "https://www.jpx.co.jp/f4.xlsx"
    m.lock_first_complete_payload(
        root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, applicable_period="2020-04",
        requested_url=f2_url, fetch_result=m.FetchResult(b"f2", f2_url, 200), retrieval_timestamp_utc="2026-08-24T03:00:00Z",
    )
    f2_slot = m.source_object_slot_id(m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2020-04", f2_url)
    m.lock_first_complete_payload(
        root, source_family=m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, applicable_period="2020-03",
        requested_url=f4_url, fetch_result=m.FetchResult(b"f4", f4_url, 200), retrieval_timestamp_utc="2026-08-24T03:00:00Z",
    )
    f4_slot = m.source_object_slot_id(m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, "2020-03", f4_url)
    with pytest.raises(m.V9005StageABlocked):
        m._validate_f2_f4_required_slot_references(
            root,
            {
                (m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2020-03"): (f2_slot,),
                (m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, "2020-03"): (f4_slot,),
            },
            {},
            (),
        )


def test_f2_f4_single_slot_acquisition_reuses_shared_support_and_returns_child_ids(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    responses = _monthly_statistics_acquisition_responses()
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return responses[url]

    f2_slot, first_attempts = m.acquire_f2_f4_monthly_evidence(
        root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, requested_month="2020-03",
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert first_attempts == 3
    root_lock = m.read_locked_payload(
        root, m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
        m.MONTHLY_STATISTICS_DISCOVERY_ROOT, m.MONTHLY_STATISTICS_ROOT_URL,
    )
    year_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/2020.html"
    year_lock = m.read_locked_payload(
        root, m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
        m.monthly_statistics_discovery_year_period(2020), year_url,
    )
    assert root_lock is not None and year_lock is not None
    assert root_lock["source_family"] == m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT
    assert root_lock["applicable_period"] == m.MONTHLY_STATISTICS_DISCOVERY_ROOT
    assert year_lock["applicable_period"] == "MONTHLY_STATISTICS_DISCOVERY_YEAR_2020"
    assert f2_slot == m.source_object_slot_id(
        m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2020-03",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/f2-2020-03.xlsx",
    )
    f2_lock = m.read_locked_payload(
        root, m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2020-03",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/f2-2020-03.xlsx",
    )
    assert f2_lock is not None
    assert f2_lock["source_family"] == m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT
    assert f2_lock["applicable_period"] == "2020-03"
    assert f2_lock["requested_url"] == "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/f2-2020-03.xlsx"
    assert f2_lock["resolved_url"] == "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/f2-2020-03.xlsx"
    f4_slot, f4_attempts = m.acquire_f2_f4_monthly_evidence(
        root, source_family=m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, requested_month="2020-03",
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert f4_attempts == 1
    assert calls.count(m.MONTHLY_STATISTICS_ROOT_URL) == 1
    assert calls.count(year_url) == 1
    assert f4_slot == m.source_object_slot_id(
        m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, "2020-03",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/f4-2020-03.xlsx",
    )
    f4_lock = m.read_locked_payload(
        root, m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, "2020-03",
        "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/f4-2020-03.xlsx",
    )
    assert f4_lock is not None
    assert f4_lock["source_family"] == m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE
    assert f4_lock["applicable_period"] == "2020-03"
    support_ids = {
        m.source_object_slot_id(m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, m.MONTHLY_STATISTICS_DISCOVERY_ROOT, m.MONTHLY_STATISTICS_ROOT_URL),
        m.source_object_slot_id(m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "MONTHLY_STATISTICS_DISCOVERY_YEAR_2020", year_url),
    }
    assert f2_slot not in support_ids and f4_slot not in support_ids
    f2_inventory = m.build_source_inventory(
        coverage_references={(m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2020-03"): [f2_slot]}, output_root=root,
    )
    f4_inventory = m.build_source_inventory(
        coverage_references={(m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, "2020-03"): [f4_slot]}, output_root=root,
    )
    assert next(r for r in f2_inventory if r["source_family"] == m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT and r["month"] == "2020-03")["status"] == m.INVENTORY_AVAILABLE
    assert next(r for r in f4_inventory if r["source_family"] == m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE and r["month"] == "2020-03")["status"] == m.INVENTORY_AVAILABLE
    _same_slot, same_attempts = m.acquire_f2_f4_monthly_evidence(
        root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, requested_month="2020-03",
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert same_attempts == 0
    _other_month, other_month_attempts = m.acquire_f2_f4_monthly_evidence(
        root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, requested_month="2020-04",
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert other_month_attempts == 1
    _other_year, other_year_attempts = m.acquire_f2_f4_monthly_evidence(
        root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, requested_month="2021-03",
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert other_year_attempts == 2


def test_f2_f4_single_slot_acquisition_uses_locked_resolved_urls_as_link_bases(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    root_final_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/redirected/index.html"
    year_requested_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/redirected/2020.html"
    year_final_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/redirected/2020/index.html"
    f2_requested_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/redirected/2020/f2.xlsx"
    f2_final_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/redirected/2020/f2-final.xlsx"
    f4_requested_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/redirected/2020/f4.xlsx"
    responses = {
        m.MONTHLY_STATISTICS_ROOT_URL: m.FetchResult(b'<a href="2020.html">2020</a>', root_final_url, 200),
        year_requested_url: m.FetchResult(
            b"<table><tr><th>Report</th><th>2020-03</th></tr>"
            + f'<tr><th>{m.F2_SEMANTIC_ROW_LABEL}</th><td><a href="f2.xlsx">F2</a></td></tr>'.encode()
            + f'<tr><th>{m.F4_SEMANTIC_ROW_LABEL}</th><td><a href="f4.xlsx">F4</a></td></tr></table>'.encode(),
            year_final_url,
            200,
        ),
        f2_requested_url: m.FetchResult(b"f2", f2_final_url, 206),
        f4_requested_url: m.FetchResult(b"f4", f4_requested_url, 200),
    }
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return responses[url]

    f2_slot, first_attempts = m.acquire_f2_f4_monthly_evidence(
        root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, requested_month="2020-03",
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert first_attempts == 3
    assert year_requested_url in calls
    assert "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/2020.html" not in calls
    assert f2_requested_url in calls
    assert "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/redirected/f2.xlsx" not in calls
    root_lock = m.read_locked_payload(
        root, m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
        m.MONTHLY_STATISTICS_DISCOVERY_ROOT, m.MONTHLY_STATISTICS_ROOT_URL,
    )
    year_lock = m.read_locked_payload(
        root, m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
        m.monthly_statistics_discovery_year_period(2020), year_requested_url,
    )
    assert root_lock is not None and year_lock is not None
    assert root_lock["requested_url"] == m.MONTHLY_STATISTICS_ROOT_URL
    assert root_lock["resolved_url"] == root_final_url
    assert year_lock["requested_url"] == year_requested_url
    assert year_lock["resolved_url"] == year_final_url
    assert f2_slot == m.source_object_slot_id(
        m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2020-03", f2_requested_url,
    )
    assert f2_slot != m.source_object_slot_id(
        m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2020-03", f2_final_url,
    )
    f4_slot, f4_attempts = m.acquire_f2_f4_monthly_evidence(
        root, source_family=m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, requested_month="2020-03",
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert f4_attempts == 1
    assert f4_slot == m.source_object_slot_id(m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, "2020-03", f4_requested_url)
    _same_slot, same_attempts = m.acquire_f2_f4_monthly_evidence(
        root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, requested_month="2020-03",
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert same_attempts == 0
    assert calls.count(m.MONTHLY_STATISTICS_ROOT_URL) == 1
    assert calls.count(year_requested_url) == 1
    assert calls.count(f2_requested_url) == 1


def test_f2_f4_single_slot_acquisition_fails_closed_before_or_after_support_lock(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        raise AssertionError("invalid input must not fetch")

    for family, month in (("UNSUPPORTED", "2020-03"), (m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "bad")):
        with pytest.raises(m.V9005StageABlocked):
            m.acquire_f2_f4_monthly_evidence(root, source_family=family, requested_month=month, fetcher=fetcher, sleep=_no_sleep, clock=_clock)
    assert calls == []
    root_key = m.source_object_slot_id(
        m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, m.MONTHLY_STATISTICS_DISCOVERY_ROOT, m.MONTHLY_STATISTICS_ROOT_URL,
    )
    (root / "raw" / f"{root_key}.bin").write_bytes(b"orphan")
    with pytest.raises(m.V9005StageABlocked):
        m.acquire_f2_f4_monthly_evidence(root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, requested_month="2020-03", fetcher=fetcher, sleep=_no_sleep, clock=_clock)
    assert calls == []


def test_f2_f4_single_slot_traversal_failure_never_fetches_a_child(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    m.lock_first_complete_payload(
        root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
        applicable_period=m.MONTHLY_STATISTICS_DISCOVERY_ROOT, requested_url=m.MONTHLY_STATISTICS_ROOT_URL,
        fetch_result=m.FetchResult(b'<a href="2020.html">2020</a>', m.MONTHLY_STATISTICS_ROOT_URL, 200),
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )
    year_url = "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/2020.html"
    m.lock_first_complete_payload(
        root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
        applicable_period=m.monthly_statistics_discovery_year_period(2020), requested_url=year_url,
        fetch_result=m.FetchResult(b"<table><tr><th>Report</th><th>2020-03</th></tr></table>", year_url, 200),
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )
    with pytest.raises(m.V9005StageABlocked):
        m.acquire_f2_f4_monthly_evidence(
            root, source_family=m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, requested_month="2020-03",
            fetcher=lambda _url: (_ for _ in ()).throw(AssertionError("must not fetch guessed child")), sleep=_no_sleep, clock=_clock,
        )


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
        # Synthetic full-pass semantic result: isolates
        # required_inventory_missing_count as the only cause of FAIL under
        # test -- not a claim that production semantic validation exists
        # (see test_production_semantic_evidence_computed_from_empty_input).
        semantic_result=_synthetic_semantic_result(),
        terminal_identities={},
        events=(),
        comparable_month_end_mismatch_count=0,
        raw_provenance_pass=True,
    )
    assert evidence["required_inventory_missing_count"] > 0
    assert evidence["FREE_JPX_METADATA_PROBE_PASS"] is False
    assert evidence["failure_class"] == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE


# --- 9. Deterministic repeated reconstruction --------------------------------

def test_reconstruction_is_deterministic() -> None:
    identities = {"1301": sem.TerminalIdentityState(
        listed_state=True, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE,
    )}
    events = (sem.SemanticEvent("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2"),)
    assert m.reconstruction_is_deterministic(terminal_identities=identities, events=events) is True
    first = m.reconstruct_security_state(terminal_identities=identities, events=events)
    second = m.reconstruct_security_state(terminal_identities=identities, events=events)
    assert first == second
    assert first["reconstructed_identity_count"] == 1


def test_reconstruction_empty_input_is_still_deterministic() -> None:
    assert m.reconstruction_is_deterministic(terminal_identities={}, events=()) is True
    reconstruction = m.reconstruct_security_state(terminal_identities={}, events=())
    assert reconstruction["reconstructed_identity_count"] == 0


# --- 10. Comparable month-end count mismatch => FAIL ------------------------

def test_month_end_mismatch_detected() -> None:
    official = {"2018-01": 3700}
    reconstructed_ok = {"2018-01": 3700}
    reconstructed_bad = {"2018-01": 3699}
    assert m.compute_month_end_mismatch_count(official, reconstructed_ok) == 0
    assert m.compute_month_end_mismatch_count(official, reconstructed_bad) == 1


def test_month_end_mismatch_fails_overall_pass_even_if_everything_else_passes() -> None:
    full_inventory = _full_available_inventory()
    evidence = m.compute_stage_a_evidence(
        inventory=full_inventory,
        terminal_snapshot_locked=True,
        trading_calendar_derived=True,
        semantic_result=_synthetic_semantic_result(),
        terminal_identities={},
        events=(),
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

_SEMANTIC_RESULT_KEYS = frozenset({
    "listing_transition_pass", "delisting_transition_pass", "market_transition_pass",
    "security_type_pass", "canonical_identity_pass", "effective_date_pass",
    "deterministic_reconstruction_pass",
})


def _synthetic_semantic_result(**overrides: object) -> dict[str, object]:
    """A fully-passing synthetic semantic-validation result, for exercising
    compute_stage_a_evidence()'s aggregation/conjunction mechanics only --
    never a claim that production semantic validation exists. Production
    always computes this for real via
    src.v9_005_stage_a_semantics.compute_semantic_validation_result (see
    test_production_semantic_evidence_computed_from_empty_input)."""
    result: dict[str, object] = {key: True for key in _SEMANTIC_RESULT_KEYS}
    result.update(overrides)
    return result


def _full_evidence(**overrides: object) -> dict[str, object]:
    full_inventory = _full_available_inventory()
    semantic_overrides = {key: overrides.pop(key) for key in list(overrides) if key in _SEMANTIC_RESULT_KEYS}
    kwargs = dict(
        inventory=full_inventory,
        terminal_snapshot_locked=True,
        trading_calendar_derived=True,
        semantic_result=_synthetic_semantic_result(**semantic_overrides),
        terminal_identities={
            "1301": sem.TerminalIdentityState(
                listed_state=True, market_state="PRIME", security_type_state=sem.SECURITY_TYPE_ELIGIBLE,
            ),
        },
        events=(sem.SemanticEvent("1301", "2017-01-10", sem.DIMENSION_LISTED_STATE, False, True, "F2"),),
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
        {"raw_provenance_pass": False},
        {"comparable_month_end_mismatch_count": 1},
        {"listing_transition_pass": False},
        {"delisting_transition_pass": False},
        {"market_transition_pass": False},
        {"security_type_pass": False},
        {"canonical_identity_pass": False},
        {"effective_date_pass": False},
        {"deterministic_reconstruction_pass": False},
    ],
)
def test_exact_pass_conjunction_false_if_any_single_condition_fails(overrides: dict[str, object]) -> None:
    evidence = _full_evidence(**overrides)
    assert evidence["FREE_JPX_METADATA_PROBE_PASS"] is False
    assert evidence["failure_class"] == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE


# --- V9_006_HIGH_2_SEMANTIC_VALIDATION_IMPLEMENTATION: the five semantic
# evidence booleans must come only from the real semantic-validation
# result, never from monthly SOURCE_INVENTORY family coverage, and never
# from an arbitrary caller-supplied boolean disguised as coverage.

def test_compute_stage_a_evidence_no_longer_accepts_coverage_proxy_kwargs() -> None:
    """The old proxy-based kwargs (reconstruction_deterministic,
    security_type_validation_pass) are gone; compute_stage_a_evidence only
    accepts the real semantic_result mapping."""
    params = set(inspect.signature(m.compute_stage_a_evidence).parameters)
    assert "reconstruction_deterministic" not in params
    assert "security_type_validation_pass" not in params
    assert "semantic_result" in params
    assert "two_run_determinism_pass" not in params
    assert "terminal_identities" in params
    assert "events" in params
    with pytest.raises(TypeError):
        _full_evidence(two_run_determinism_pass=True)


def test_full_inventory_coverage_alone_cannot_make_semantic_gates_pass() -> None:
    """Dummy/coverage-only evidence can no longer make the semantic gates
    PASS: full 648-record inventory coverage plus a semantic_result that
    reports every semantic gate False must still fail every one of them,
    proving listing/delisting/market/canonical_identity/effective_date/
    deterministic_reconstruction are driven by semantic_result alone."""
    failing_semantic_result = {key: False for key in _SEMANTIC_RESULT_KEYS}
    evidence = _full_evidence(**failing_semantic_result)
    for key in _SEMANTIC_RESULT_KEYS:
        assert evidence[key] is False, key
    assert evidence["FREE_JPX_METADATA_PROBE_PASS"] is False


def test_market_transition_pass_is_independent_of_listing_transition_pass_in_evidence() -> None:
    """market_transition_pass must not equal listing_transition_pass
    merely as a proxy at the compute_stage_a_evidence layer either --
    they can differ."""
    evidence = _full_evidence(listing_transition_pass=True, market_transition_pass=False)
    assert evidence["listing_transition_pass"] is True
    assert evidence["market_transition_pass"] is False


def test_canonical_identity_pass_is_not_terminal_snapshot_locked_and_security_type_pass() -> None:
    """canonical_identity_pass must not be recomputed in
    compute_stage_a_evidence as terminal_snapshot_locked AND
    security_type_pass -- it is fed straight from semantic_result and can
    be True even while terminal_snapshot_locked is False, or False while
    both terminal_snapshot_locked and security_type_pass are True."""
    evidence = _full_evidence(terminal_snapshot_locked=False, canonical_identity_pass=True)
    assert evidence["terminal_snapshot_pass"] is False
    assert evidence["canonical_identity_pass"] is True

    evidence2 = _full_evidence(terminal_snapshot_locked=True, security_type_pass=True, canonical_identity_pass=False)
    assert evidence2["terminal_snapshot_pass"] is True
    assert evidence2["security_type_pass"] is True
    assert evidence2["canonical_identity_pass"] is False


def test_effective_date_pass_is_not_a_coverage_conjunction() -> None:
    """effective_date_pass must not be recomputed as
    listing_transition_pass AND delisting_transition_pass AND
    market_transition_pass -- it is fed straight from semantic_result."""
    evidence = _full_evidence(
        listing_transition_pass=True, delisting_transition_pass=True, market_transition_pass=True,
        effective_date_pass=False,
    )
    assert evidence["effective_date_pass"] is False


def test_deterministic_reconstruction_gate_requires_reverse_forward_and_two_run() -> None:
    evidence = _full_evidence()
    assert evidence["deterministic_reconstruction_pass"] is True

    reverse_forward_failed = _full_evidence(deterministic_reconstruction_pass=False)
    assert reverse_forward_failed["deterministic_reconstruction_pass"] is False


def test_two_run_mismatch_fails_deterministic_and_free_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deterministic injection proves the actual two-run check is part
    of the Stage-A pass conjunction, without introducing randomness."""
    monkeypatch.setattr(m, "reconstruction_is_deterministic", lambda **_kwargs: False)
    evidence = _full_evidence()
    assert evidence["deterministic_reconstruction_pass"] is False
    assert evidence["FREE_JPX_METADATA_PROBE_PASS"] is False


def test_production_semantic_evidence_computed_from_empty_input() -> None:
    """Static-source proof that run_stage_a() -- the real-execution
    orchestration entrypoint -- computes its semantic_result via the real
    src.v9_005_stage_a_semantics.compute_semantic_validation_result with
    empty terminal_identities/events (no F2-F7 acquisition exists yet),
    never via a hardcoded or caller-supplied arbitrary PASS boolean. Since
    the semantics engine itself fails closed on empty input (see
    tests/test_v9_005_stage_a_semantics.py::
    test_no_terminal_identities_fails_closed_not_vacuous_pass), this
    guarantees production never fakes a semantic PASS."""
    source = inspect.getsource(m.run_stage_a)
    assert "terminal_identities: Mapping[str, TerminalIdentityState] = {}" in source
    assert "events: Sequence[SemanticEvent] = ()" in source
    assert "terminal_identities=terminal_identities," in source
    assert "events=events," in source
    # Confirm the real (unmocked) engine actually fails closed on that
    # exact call, so this static check is backed by real behavior.
    result = m.compute_semantic_validation_result(terminal_identities={}, events=())
    for key in _SEMANTIC_RESULT_KEYS:
        assert result[key] is False, key


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
    assert m.extract_data_j_xls_url(page, m.LISTED_ISSUES_PAGE_URL) == "https://www.jpx.co.jp/x/data_j.xls"


def test_extract_data_j_xls_url_rejects_off_domain_link() -> None:
    page = b'<html><a href="https://evil.example/data_j.xls">Excel</a></html>'
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.extract_data_j_xls_url(page, m.LISTED_ISSUES_PAGE_URL)
    assert excinfo.value.reason == "OFF_DOMAIN_REDIRECT_REJECTED"


def test_extract_data_j_xls_url_missing_link_fails_closed() -> None:
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.extract_data_j_xls_url(b"<html>no link here</html>", m.LISTED_ISSUES_PAGE_URL)
    assert excinfo.value.failure_class == m.SOURCE_OR_DATA_FEASIBILITY_FAILURE


# --- V9_006_LOCATOR_IMPL_HIGH_2: F1's authoritative root is the exact
# English listed-issues page bound in
# V9_006_STAGE_A_SOURCE_SLOT_LOCATOR_METHODOLOGY.md -- no alias, fallback,
# redirect-based substitution, non-English alternative, or guessed
# historical root.

def test_f1_root_is_exact_bound_english_root() -> None:
    assert m.LISTED_ISSUES_PAGE_URL == "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01.html"


def test_f1_locator_strategy_root_matches_bound_constant() -> None:
    strategy = m.LOCATOR_STRATEGIES[m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END]
    assert strategy.root_url == m.LISTED_ISSUES_PAGE_URL


def test_extract_data_j_xls_url_relative_link_resolves_against_english_root() -> None:
    """A genuinely relative (non-absolute-path) data_j.xls href must resolve
    relative to the exact bound English root, not any other JPX page --
    proving the traversal rule stays same-domain and does not freeze or
    guess a concrete child URL beyond what the supplied page actually
    contains."""
    page = b'<html><a href="data_j.xls">Excel</a></html>'
    resolved = m.extract_data_j_xls_url(page, m.LISTED_ISSUES_PAGE_URL)
    assert resolved == "https://www.jpx.co.jp/english/markets/statistics-equities/misc/data_j.xls"
    assert resolved.startswith("https://" + m.LISTED_ISSUES_PAGE_HOST + "/")


def test_extract_data_j_xls_url_uses_locked_discovery_final_url_as_relative_base() -> None:
    final_url = "https://www.jpx.co.jp/english/markets/statistics-equities/misc/redirected/index.html"
    derived = m.extract_data_j_xls_url(b'<a href="data_j.xls">Excel</a>', final_url)
    assert derived == "https://www.jpx.co.jp/english/markets/statistics-equities/misc/redirected/data_j.xls"
    assert derived != "https://www.jpx.co.jp/english/markets/statistics-equities/misc/data_j.xls"
    assert m.source_object_slot_id(m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, m.TERMINAL_PERIOD, derived) != m.source_object_slot_id(
        m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, m.TERMINAL_PERIOD, final_url,
    )
    with pytest.raises(m.V9005StageABlocked):
        m.extract_data_j_xls_url(b'<a href="data_j.xls">Excel</a>', "https://evil.example/index.html")


# --- Transport retry classification (per AI_REAL_EXECUTION_RUNBOOK.md) -----

def test_transport_retryable_then_success() -> None:
    attempts: list[int] = []

    def fetcher(url: str) -> m.FetchResult:
        attempts.append(1)
        if len(attempts) < 2:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        return m.FetchResult(b"payload", url, 200)

    result, requests_used = m.fetch_once_with_retry("https://www.jpx.co.jp/x", fetcher, _no_sleep)
    assert result.payload == b"payload"
    assert requests_used == 2


def test_transport_nonretryable_fails_immediately() -> None:
    def fetcher(url: str) -> m.FetchResult:
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.fetch_once_with_retry("https://www.jpx.co.jp/x", fetcher, _no_sleep)
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE


def test_transport_exhausted_retries_is_plumbing_failure_retriable() -> None:
    def fetcher(url: str) -> m.FetchResult:
        raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.fetch_once_with_retry("https://www.jpx.co.jp/x", fetcher, _no_sleep)
    assert excinfo.value.reason == m.PLUMBING_FAILURE_RETRIABLE


@pytest.mark.parametrize(
    "redirect_url",
    [
        "https://evil.example/redirected",
        "https://jpx.co.jp.evil.example/redirected",
        "http://www.jpx.co.jp/redirected",
        "https://user@www.jpx.co.jp/redirected",
        "https://www.jpx.co.jp:444/redirected",
        "https://www.jpx.co.jp/redirected#fragment",
    ],
)
def test_production_redirect_handler_rejects_unsafe_target_before_following(
    monkeypatch: pytest.MonkeyPatch, redirect_url: str
) -> None:
    production = _production_script_module()
    delegated: list[str] = []

    def _delegate(*_args: object, **_kwargs: object) -> object:
        delegated.append("followed")
        return object()

    monkeypatch.setattr(urllib.request.HTTPRedirectHandler, "redirect_request", _delegate)
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        production._JpxRedirectHandler().redirect_request(None, None, 302, "Found", None, redirect_url)
    assert excinfo.value.reason == "OFF_DOMAIN_REDIRECT_REJECTED"
    assert delegated == []


def test_production_redirect_handler_allows_same_domain_target(monkeypatch: pytest.MonkeyPatch) -> None:
    production = _production_script_module()
    delegated: list[str] = []
    sentinel = object()

    def _delegate(*_args: object, **_kwargs: object) -> object:
        delegated.append("followed")
        return sentinel

    monkeypatch.setattr(urllib.request.HTTPRedirectHandler, "redirect_request", _delegate)
    result = production._JpxRedirectHandler().redirect_request(
        None, None, 302, "Found", None, "https://sub.jpx.co.jp/redirected"
    )
    assert result is sentinel
    assert delegated == ["followed"]


class _FakeProductionResponse:
    def __init__(self, final_url: str, payload: bytes = b"payload", status: int = 200) -> None:
        self._final_url = final_url
        self._payload = payload
        self.status = status
        self.read_count = 0
        self.closed = False

    def geturl(self) -> str:
        return self._final_url

    def read(self) -> bytes:
        self.read_count += 1
        return self._payload

    def close(self) -> None:
        self.closed = True


class _FakeProductionOpener:
    def __init__(self, response: _FakeProductionResponse) -> None:
        self.response = response
        self.requests: list[object] = []

    def open(self, request: object, *, timeout: int) -> _FakeProductionResponse:
        assert timeout == 30
        self.requests.append(request)
        return self.response


def test_production_final_url_is_validated_before_body_read(monkeypatch: pytest.MonkeyPatch) -> None:
    production = _production_script_module()
    response = _FakeProductionResponse("https://evil.example/final")
    opener = _FakeProductionOpener(response)
    monkeypatch.setattr(production.urllib.request, "build_opener", lambda *_handlers: opener)

    with pytest.raises(m.V9005StageABlocked) as excinfo:
        production._production_fetcher("https://www.jpx.co.jp/start")
    assert excinfo.value.reason == "OFF_DOMAIN_REDIRECT_REJECTED"
    assert response.read_count == 0
    assert response.closed is True


def test_production_valid_final_response_is_read_once_with_observed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = _production_script_module()
    response = _FakeProductionResponse("https://sub.jpx.co.jp/final", b"exact bytes", 206)
    opener = _FakeProductionOpener(response)
    monkeypatch.setattr(production.urllib.request, "build_opener", lambda *_handlers: opener)

    result = production._production_fetcher("https://www.jpx.co.jp/start")
    assert result == m.FetchResult(b"exact bytes", "https://sub.jpx.co.jp/final", 206)
    assert response.read_count == 1
    assert response.closed is True
    assert len(opener.requests) == 1


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

    def fetcher(url: str) -> m.FetchResult:
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

    def fetcher(url: str) -> m.FetchResult:
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
    xls_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/data_j.xls"
    discovery_html = _synthetic_listing_page()
    responses = {
        m.LISTED_ISSUES_PAGE_URL: m.FetchResult(discovery_html, "https://www.jpx.co.jp/english/markets/statistics-equities/misc/01-final.html", 200),
        xls_url: m.FetchResult(b"xls-bytes", "https://www.jpx.co.jp/markets/statistics-equities/misc/data_j-final.xls", 206),
        m.CALENDAR_PAGE_URL: m.FetchResult(_synthetic_calendar_html(), m.CALENDAR_PAGE_URL, 200),
    }

    def fetcher(url: str) -> m.FetchResult:
        return responses[url]

    actual_extract = m.extract_data_j_xls_url

    def extract_only_after_discovery_lock(raw: bytes, page_url: str) -> str:
        locked = m.read_locked_payload(
            tmp_path / "stage-a-out", m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
            m.TERMINAL_DISCOVERY_ROOT, m.LISTED_ISSUES_PAGE_URL,
        )
        assert locked is not None and locked["raw"] == raw == discovery_html
        return actual_extract(raw, page_url)

    monkeypatch.setattr(m, "extract_data_j_xls_url", extract_only_after_discovery_lock)

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
    discovery = m.read_locked_payload(
        durable_root, m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END,
        m.TERMINAL_DISCOVERY_ROOT, m.LISTED_ISSUES_PAGE_URL,
    )
    terminal = m.read_locked_payload(
        durable_root, m.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END, m.TERMINAL_PERIOD, xls_url,
    )
    assert discovery is not None and terminal is not None
    assert discovery["requested_url"] == m.LISTED_ISSUES_PAGE_URL
    assert discovery["resolved_url"] == responses[m.LISTED_ISSUES_PAGE_URL].resolved_url
    assert terminal["requested_url"] == xls_url != m.LISTED_ISSUES_PAGE_URL
    assert terminal["resolved_url"] == responses[xls_url].resolved_url
    assert terminal["http_status"] == 206  # actual non-200 status is never rewritten
    for locked, response in ((discovery, responses[m.LISTED_ISSUES_PAGE_URL]), (terminal, responses[xls_url])):
        assert locked["raw"] == response.payload
        assert locked["byte_length"] == len(response.payload)
        assert locked["sha256"] == m.sha256_bytes(response.payload)
    assert m.verify_raw_provenance(durable_root) is True


def test_run_stage_a_wrong_signal_grid_blob_stops_before_any_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SAFETY/SCOPE: forces ACQUISITION_IMPLEMENTATION_COMPLETE True so this
    # test can still reach and exercise the (later) signal-grid-binding
    # check below the V9_006_LOCATOR_IMPL_HIGH_1 gate -- see the identical
    # note on test_run_stage_a_offline_reports_fail_with_safe_evidence.
    monkeypatch.setattr(m, "ACQUISITION_IMPLEMENTATION_COMPLETE", True)
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
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


# --- V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_OFFLINE_IMPLEMENTATION ----------
# Fully offline: every fixture below is synthetic, already-locked bytes.
# None of these tests perform, or are permitted to require, any network
# call, per V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_IMPLEMENTATION_CONTRACT.md.

def _lock_f6_diagnostic(
    root: Path, payload: bytes, *, resolved_url: str | None = None, http_status: int = 200,
) -> dict[str, object]:
    return m.lock_first_complete_payload(
        root,
        source_family=m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
        applicable_period=m.F6_ROOT_STRUCTURE_DIAGNOSTIC,
        requested_url=m.TOPIX_ROOT_URL,
        fetch_result=m.FetchResult(payload, resolved_url or m.TOPIX_ROOT_URL, http_status),
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )


def test_f6_root_structure_single_occurrence_is_captured(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"<html><body><h2>Historical Index Value</h2></body></html>")
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    assert artifact["label_occurrence_count"] == 1
    assert artifact["failure_reason"] is None
    assert len(artifact["occurrences"]) == 1
    assert artifact["occurrences"][0]["dom_path"][-1]["tag"] == "h2"
    assert artifact["target_label"] == m.F6_SEMANTIC_SECTION_LABEL
    assert artifact["requested_url"] == m.TOPIX_ROOT_URL
    assert artifact["schema_version"] == m.F6_ROOT_STRUCTURE_PROBE_RESULT_SCHEMA_VERSION
    assert artifact["diagnostic"] == m.F6_ROOT_STRUCTURE_PROBE_DIAGNOSTIC_NAME


def test_f6_root_structure_inline_markup_still_matches(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"<h2>Historical <em>Index</em> Value</h2>")
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    assert artifact["label_occurrence_count"] == 1
    assert artifact["occurrences"][0]["dom_path"][-1]["tag"] == "h2"


def test_f6_root_structure_leaf_most_excludes_matching_ancestor(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"<div><span>Historical Index Value</span></div>")
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    assert artifact["label_occurrence_count"] == 1
    only = artifact["occurrences"][0]
    assert only["dom_path"][-1]["tag"] == "span"
    assert [c["tag"] for c in only["dom_path"]] == ["div", "span"]


def test_f6_root_structure_zero_occurrences_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"<h2>Nothing Here</h2>")
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_AMBIGUOUS
    assert artifact["label_occurrence_count"] == 0
    assert artifact["occurrences"] == []
    assert artifact["failure_reason"] is None


def test_f6_root_structure_multiple_occurrences_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"<h2>Historical Index Value</h2><h3>Historical Index Value</h3>")
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_AMBIGUOUS
    assert artifact["label_occurrence_count"] == 2
    assert [o["dom_path"][-1]["tag"] for o in artifact["occurrences"]] == ["h2", "h3"]


@pytest.mark.parametrize(
    "payload, expected_status, expected_count",
    [
        (b"<h2>Historical   Index\n Value</h2>", m.STRUCTURE_CAPTURED, 1),
        # A real &nbsp; entity is resolved exactly once by the parser
        # (convert_charrefs=True) into an actual U+00A0, which is Unicode
        # whitespace and so collapses to the target's plain space.
        ("<h2>Historical&nbsp;Index Value</h2>".encode(), m.STRUCTURE_CAPTURED, 1),
        (b"<h2>historical index value</h2>", m.STRUCTURE_AMBIGUOUS, 0),
        # V9_006_F6_ROOT_OFFLINE_MEDIUM_1: only &amp; is a real entity here;
        # decoding it once leaves the literal text "Historical&#32;Index
        # Value" in the parsed DOM. A second (recursive) unescape pass
        # would wrongly decode &#32; into a space and falsely match the
        # target -- normalization must not do that second pass.
        (b"<h2>Historical&amp;#32;Index Value</h2>", m.STRUCTURE_AMBIGUOUS, 0),
        # Same double-decode trap with a named entity: only &amp; is real;
        # decoding it once leaves the literal text "Historical&nbsp;Index
        # Value" (no actual NBSP character) in the parsed DOM.
        (b"<h2>Historical&amp;nbsp;Index Value</h2>", m.STRUCTURE_AMBIGUOUS, 0),
    ],
)
def test_f6_root_structure_whitespace_entity_normalization_is_exact_and_case_sensitive(
    tmp_path: Path, payload: bytes, expected_status: str, expected_count: int,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, payload)
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == expected_status
    assert artifact["label_occurrence_count"] == expected_count


def test_f6_root_structure_anchor_visible_text_is_decoded_exactly_once(tmp_path: Path) -> None:
    """V9_006_F6_ROOT_OFFLINE_MEDIUM_1: anchor visible text must resolve
    only the one real entity (&amp;) in the source, never recursively
    decode the resulting literal "&nbsp;" text into an actual space. The
    anchor is a sibling of the occurrence element (parent_children), not a
    descendant, so its own visible text never pollutes the h2's matched
    label text."""
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<div><a href="x.html">A&amp;nbsp;B</a><h2>Historical Index Value</h2></div>',
    )
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    anchor_text = artifact["occurrences"][0]["anchors"]["parent_children"][0]["text"]
    assert anchor_text == "A&nbsp;B"


def test_f6_root_structure_sibling_index_ignores_text_nodes_and_classes_are_sorted_unique(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<div>loose text<span>skip</span><span class="b a a">Historical Index Value</span></div>',
    )
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    last = artifact["occurrences"][0]["dom_path"][-1]
    assert last == {"tag": "span", "sibling_index": 1, "id": None, "classes": ["a", "b"]}


def test_f6_root_structure_all_four_anchor_relation_categories(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b"<section>"
            b'<a href="parent1.html">P1</a>'
            b'<h2>Historical Index Value<a href="child1.html"><img src="icon.png"></a></h2>'
            b'<p><a href="follow1.html">F1</a><a href="follow2.html">F2</a></p>'
            b'<a href="parent2.html">P2</a>'
            b"</section>"
        ),
    )
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    anchors = artifact["occurrences"][0]["anchors"]
    assert anchors["self"] is None
    assert [a["href"] for a in anchors["children"]] == ["child1.html"]
    assert [a["href"] for a in anchors["parent_children"]] == ["parent1.html", "parent2.html"]
    assert [a["href"] for a in anchors["following_sibling_children"]] == ["follow1.html", "follow2.html"]


def test_f6_root_structure_self_anchor_when_occurrence_element_is_an_anchor(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b'<div><a href="self.html">Historical Index Value</a></div>')
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    anchors = artifact["occurrences"][0]["anchors"]
    assert anchors["self"] == {
        "text": "Historical Index Value", "href": "self.html",
        "dom_path": artifact["occurrences"][0]["dom_path"],
    }
    assert anchors["children"] == []


def test_f6_root_structure_raw_href_preserved_source_exact_never_resolved(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<h2>Historical Index Value<a href="page.html?a=1&amp;b=2"><img src="i.png"></a></h2>',
    )
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    href = artifact["occurrences"][0]["anchors"]["children"][0]["href"]
    assert href == "page.html?a=1&amp;b=2"  # entity spelling untouched, never decoded/resolved
    assert m.TOPIX_ROOT_URL not in href


def test_f6_root_structure_unrelated_numerical_page_text_absent_from_artifact(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b"<html><body>"
            b"<table><tr><td>2024-01-04</td><td>1783.51</td></tr></table>"
            b"<h2>Historical Index Value</h2>"
            b"</body></html>"
        ),
    )
    artifact = m.run_f6_root_structure_probe_offline(root)
    serialized = json.dumps(artifact)
    assert "1783.51" not in serialized
    assert "2024-01-04" not in serialized


def test_f6_root_structure_same_locked_bytes_reprocessed_is_byte_identical_no_overwrite(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"<h2>Historical Index Value</h2>")
    first = m.run_f6_root_structure_probe_offline(root)
    second = m.run_f6_root_structure_probe_offline(root)
    assert first == second
    result_path = root / m.F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME
    on_disk = result_path.read_bytes()
    assert on_disk == m.canonical_bytes(first)

    # A differing artifact for the same path must fail closed, never overwrite.
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.write_f6_root_structure_probe_artifact(root, {**first, "status": "TAMPERED"})
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert result_path.read_bytes() == on_disk


def test_f6_root_structure_missing_diagnostic_lock_fails_closed(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    with pytest.raises(m.V9005StageABlocked):
        m.read_f6_root_structure_diagnostic_lock(root)
    with pytest.raises(m.V9005StageABlocked):
        m.run_f6_root_structure_probe_offline(root)
    assert not (root / m.F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME).exists()


def test_f6_root_structure_corrupt_diagnostic_lock_fails_closed(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = _lock_f6_diagnostic(root, b"<h2>Historical Index Value</h2>")
    key = m.source_object_slot_id(
        m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC, m.TOPIX_ROOT_URL,
    )
    (root / "raw" / f"{key}.bin").write_bytes(b"tampered-bytes")
    with pytest.raises(m.V9005StageABlocked):
        m.run_f6_root_structure_probe_offline(root)


@pytest.mark.parametrize(
    "payload",
    [
        b"<h2>Historical Index Value</h3>",
        b"</div><h2>Historical Index Value</h2>",
        b"<div><h2>Historical Index Value</h2>",
        b'<h2>Historical Index Value<a href="x.html">outer<a href="y.html">nested</a></a></h2>',
    ],
)
def test_f6_root_structure_malformed_dom_fails_closed_deterministically(
    tmp_path: Path, payload: bytes,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, payload)
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert artifact["label_occurrence_count"] is None
    assert artifact["occurrences"] == []
    assert artifact["failure_reason"] == m._F6_MALFORMED_DOM_STRUCTURE
    # Deterministic: reprocessing the same locked bytes reproduces the same artifact.
    root2 = m.initialize_output_root(tmp_path / "out2")
    _lock_f6_diagnostic(root2, payload)
    assert m.run_f6_root_structure_probe_offline(root2) == artifact


def test_f6_root_structure_invalid_utf8_fails_closed_with_no_fallback(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"<h2>Historical Index Value \xff\xfe</h2>")
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert artifact["failure_reason"] == m._F6_PAYLOAD_DECODE_FAILED
    assert artifact["label_occurrence_count"] is None
    assert artifact["occurrences"] == []


def test_f6_root_structure_utf8_bom_is_allowed_and_stripped(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"\xef\xbb\xbf<h2>Historical Index Value</h2>")
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    assert artifact["label_occurrence_count"] == 1


def test_f6_root_structure_ambiguous_raw_href_fails_extraction(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    # Two href attributes on one anchor: the raw href cannot be determined
    # unambiguously, so extraction must fail closed rather than guess.
    _lock_f6_diagnostic(
        root,
        b'<h2>Historical Index Value<a href="a.html" href="b.html"><img src="i.png"></a></h2>',
    )
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert artifact["failure_reason"] == m._F6_AMBIGUOUS_RAW_HREF_ATTRIBUTE


def test_f6_root_structure_offline_seam_calls_no_network_fetch_retry_or_ensure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"<h2>Historical Index Value</h2>")

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network/fetch/retry function invoked by offline seam")

    monkeypatch.setattr(m, "fetch_once_with_retry", _blocked)
    monkeypatch.setattr(m, "ensure_locked_payload", _blocked)
    monkeypatch.setattr(m, "lock_first_complete_payload", _blocked)
    artifact = m.run_f6_root_structure_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_CAPTURED

    # None of the offline seam's entry points accept a fetcher/sleep/clock.
    for func in (
        m.read_f6_root_structure_diagnostic_lock,
        m.parse_f6_root_structure_probe,
        m.write_f6_root_structure_probe_artifact,
        m.run_f6_root_structure_probe_offline,
    ):
        params = set(inspect.signature(func).parameters)
        assert params.isdisjoint({"fetcher", "sleep", "clock"})


def test_f6_root_structure_diagnostic_slot_cannot_populate_f6_inventory(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = _lock_f6_diagnostic(root, b"<h2>Historical Index Value</h2>")
    diagnostic_slot_id = m.source_object_slot_id(
        m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC, m.TOPIX_ROOT_URL,
    )
    assert diagnostic_slot_id == m.source_object_slot_id(
        locked["source_family"], locked["applicable_period"], locked["requested_url"],
    )
    # The diagnostic applicable_period is not a valid inventory month, so it
    # can never be wired into build_source_inventory's coverage references.
    with pytest.raises(m.V9005StageABlocked):
        m.build_source_inventory({
            (m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC): (diagnostic_slot_id,),
        })
    # With no coverage references at all, F6 remains MISSING for every month
    # exactly as before this diagnostic seam existed.
    inventory = m.build_source_inventory()
    f6_records = [r for r in inventory if r["source_family"] == m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE]
    assert len(f6_records) == len(m.inventory_months())
    assert all(r["status"] == m.INVENTORY_MISSING for r in f6_records)


def test_f6_root_structure_acquisition_implementation_complete_still_false() -> None:
    assert m.ACQUISITION_IMPLEMENTATION_COMPLETE is False


# --- V9_006_STAGE_A_F6_SECTION_NEIGHBORHOOD_PROBE_OFFLINE_IMPLEMENTATION ----
# Fully offline: every fixture below is synthetic, already-locked bytes
# reusing the existing F6_ROOT_STRUCTURE_DIAGNOSTIC raw lock via
# _lock_f6_diagnostic (defined above). None of these tests perform, or are
# permitted to require, any network call, per
# V9_006_STAGE_A_F6_ROOT_STRUCTURE_ADJUDICATION_AND_NEIGHBORHOOD_PROBE_
# DESIGN.md section 4. No test ever hardcodes the literal `heading_14`
# value as a semantic-heading requirement -- it is used only once, in
# test_f6_neighborhood_literal_heading_14_not_hardcoded, as a deliberate
# decoy under the wrong tag that must NOT be selected.

def test_f6_neighborhood_observed_shape_identifies_semantic_heading_without_literal_id(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<html><body>'
            b'<nav><a href="#some_id">Historical Index Value</a></nav>'
            b'<div class="wrap">'
            b'<h2 id="some_id" class="heading-title x"><span>Historical Index Value</span></h2>'
            b'</div>'
            b'</body></html>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.NEIGHBORHOOD_CAPTURED
    assert artifact["schema_version"] == m.F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_SCHEMA_VERSION
    assert artifact["diagnostic"] == m.F6_SECTION_NEIGHBORHOOD_PROBE_DIAGNOSTIC_NAME
    assert artifact["semantic_heading"]["tag"] == "h2"
    assert artifact["semantic_heading"]["id"] == "some_id"
    assert "heading-title" in artifact["semantic_heading"]["classes"]
    assert artifact["parent_container"]["tag"] == "div"
    assert "heading_14" not in json.dumps(artifact)


def test_f6_neighborhood_literal_heading_14_not_hardcoded(tmp_path: Path) -> None:
    """A decoy element literally id="heading_14" (wrong tag, not the
    fragment target) must never be picked; the real target uses a
    different id entirely, proving the rule is derived mechanically, not
    matched against the literal string "heading_14"."""
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<html><body>'
            b'<div id="heading_14" class="heading-title"><p>decoy, not the fragment target</p></div>'
            b'<nav><a href="#other_frag">Historical Index Value</a></nav>'
            b'<section>'
            b'<h2 id="other_frag" class="heading-title"><span>Historical Index Value</span></h2>'
            b'</section>'
            b'</body></html>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.NEIGHBORHOOD_CAPTURED
    assert artifact["semantic_heading"]["id"] == "other_frag"
    assert artifact["parent_container"]["tag"] == "section"


def test_f6_neighborhood_zero_fragment_candidates_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b'<h2 id="x" class="heading-title">Historical Index Value</h2>')
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.SEMANTIC_HEADING_AMBIGUOUS
    assert artifact["failure_reason"] is None
    assert artifact["semantic_heading"] is None
    assert artifact["parent_container"] is None
    assert artifact["children"] == []
    assert artifact["anchors"] == []
    assert artifact["headings"] == []


def test_f6_neighborhood_multiple_fragment_candidates_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#a">Historical Index Value</a>'
            b'<a href="#b">Historical Index Value</a>'
            b'<h2 id="a" class="heading-title">Historical Index Value</h2>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.SEMANTIC_HEADING_AMBIGUOUS


def test_f6_neighborhood_duplicate_id_target_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#dup">Historical Index Value</a>'
            b'<h2 id="dup" class="heading-title">A</h2>'
            b'<h3 id="dup" class="heading-title">B</h3>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.SEMANTIC_HEADING_AMBIGUOUS


def test_f6_neighborhood_target_not_h2_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<a href="#x">Historical Index Value</a><h3 id="x" class="heading-title">Historical Index Value</h3>',
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.SEMANTIC_HEADING_AMBIGUOUS


def test_f6_neighborhood_missing_heading_title_class_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<a href="#x">Historical Index Value</a><h2 id="x" class="other">Historical Index Value</h2>',
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.SEMANTIC_HEADING_AMBIGUOUS


def test_f6_neighborhood_target_contains_zero_label_descendants_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<a href="#x">Historical Index Value</a><h2 id="x" class="heading-title">Something Else</h2>',
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.SEMANTIC_HEADING_AMBIGUOUS


def test_f6_neighborhood_target_contains_multiple_label_descendants_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#x">Historical Index Value</a>'
            b'<h2 id="x" class="heading-title">'
            b'<span>Historical Index Value</span><em>Historical Index Value</em>'
            b'</h2>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.SEMANTIC_HEADING_AMBIGUOUS


def test_f6_neighborhood_semantic_heading_self_text_not_accepted_as_descendant_is_ambiguous(
    tmp_path: Path,
) -> None:
    """V9_006_F6_NEIGHBORHOOD_MEDIUM_1: design section 2.2 step 6 requires
    the qualifying leaf-most exact-label occurrence to be found among the
    target h2's DESCENDANTS. When the h2's own entire text is the label
    (no descendant element carries it separately), the h2 itself is the
    leaf-most occurrence -- but it is not a descendant of itself, so step 6
    must fail and this must be SEMANTIC_HEADING_AMBIGUOUS, never a silently
    accepted match. Contrast with
    test_f6_neighborhood_observed_shape_identifies_semantic_heading_without_
    literal_id and test_f6_neighborhood_literal_heading_14_not_hardcoded
    above, both of which wrap the label in a descendant <span> and
    correctly resolve to a unique semantic heading."""
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#x">Historical Index Value</a>'
            b'<div>'
            b'<h2 id="x" class="heading-title">Historical Index Value</h2>'
            b'</div>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.SEMANTIC_HEADING_AMBIGUOUS
    assert artifact["failure_reason"] is None
    assert artifact["semantic_heading"] is None
    assert artifact["parent_container"] is None
    assert artifact["children"] == []
    assert artifact["anchors"] == []
    assert artifact["headings"] == []


def test_f6_neighborhood_parent_children_preserve_dom_order_and_relation(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#x">Historical Index Value</a>'
            b'<div>'
            b'<p>before1</p><span>before2</span>'
            b'<h2 id="x" class="heading-title"><span>Historical Index Value</span></h2>'
            b'<p>after1</p><span>after2</span>'
            b'</div>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.NEIGHBORHOOD_CAPTURED
    children = artifact["children"]
    assert [c["tag"] for c in children] == ["p", "span", "h2", "p", "span"]
    assert [c["relation"] for c in children] == [
        m.NEIGHBORHOOD_RELATION_BEFORE_HEADING, m.NEIGHBORHOOD_RELATION_BEFORE_HEADING,
        m.NEIGHBORHOOD_RELATION_HEADING,
        m.NEIGHBORHOOD_RELATION_AFTER_HEADING, m.NEIGHBORHOOD_RELATION_AFTER_HEADING,
    ]


def test_f6_neighborhood_descendant_anchors_preserve_dom_order_and_relation(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#x">Historical Index Value</a>'
            b'<div>'
            b'<a href="before.html">Before</a>'
            b'<h2 id="x" class="heading-title">'
            b'<span>Historical Index Value</span><a href="inside.html">Inside</a></h2>'
            b'<a href="after.html">After</a>'
            b'</div>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.NEIGHBORHOOD_CAPTURED
    anchors = artifact["anchors"]
    assert [a["href"] for a in anchors] == ["before.html", "inside.html", "after.html"]
    assert [a["relation"] for a in anchors] == [
        m.NEIGHBORHOOD_RELATION_BEFORE_HEADING,
        m.NEIGHBORHOOD_RELATION_INSIDE_HEADING,
        m.NEIGHBORHOOD_RELATION_AFTER_HEADING,
    ]


def test_f6_neighborhood_raw_href_source_exact_never_resolved(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#x">Historical Index Value</a>'
            b'<div><h2 id="x" class="heading-title"><span>Historical Index Value</span></h2>'
            b'<a href="page.html?a=1&amp;b=2">L</a></div>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.NEIGHBORHOOD_CAPTURED
    href = artifact["anchors"][0]["href"]
    assert href == "page.html?a=1&amp;b=2"  # entity spelling untouched, never decoded/resolved
    assert m.TOPIX_ROOT_URL not in href


def test_f6_neighborhood_ambiguous_duplicate_raw_href_fails_extraction(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#x">Historical Index Value</a>'
            b'<div><h2 id="x" class="heading-title"><span>Historical Index Value</span></h2>'
            b'<a href="a.html" href="b.html"><img src="i.png"></a></div>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert artifact["failure_reason"] == m._F6_AMBIGUOUS_RAW_HREF_ATTRIBUTE
    assert artifact["semantic_heading"] is None
    assert artifact["anchors"] == []


def test_f6_neighborhood_descendant_headings_only_h1_to_h6_normalized_text(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#x">Historical Index Value</a>'
            b'<div>'
            b'<h2 id="x" class="heading-title"><span>Historical   Index\n Value</span></h2>'
            b'<h4>Sub Section</h4>'
            b'<p>not a heading</p>'
            b'<strong>Also not a heading</strong>'
            b'</div>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.NEIGHBORHOOD_CAPTURED
    headings = artifact["headings"]
    assert [(h["tag"], h["text"]) for h in headings] == [
        ("h2", "Historical Index Value"), ("h4", "Sub Section"),
    ]


def test_f6_neighborhood_unrelated_text_and_topix_values_absent_from_artifact(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<a href="#x">Historical Index Value</a>'
            b'<div>'
            b'<h2 id="x" class="heading-title"><span>Historical Index Value</span></h2>'
            b'<table><tr><td>2024-01-04</td><td>1783.51</td></tr></table>'
            b'<p>some unrelated paragraph text</p>'
            b'</div>'
        ),
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.NEIGHBORHOOD_CAPTURED
    serialized = json.dumps(artifact)
    assert "1783.51" not in serialized
    assert "2024-01-04" not in serialized
    assert "some unrelated paragraph text" not in serialized


def test_f6_neighborhood_same_lock_reprocessed_is_byte_identical_no_overwrite(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<a href="#x">Historical Index Value</a>'
        b'<div><h2 id="x" class="heading-title"><span>Historical Index Value</span></h2></div>',
    )
    first = m.run_f6_section_neighborhood_probe_offline(root)
    second = m.run_f6_section_neighborhood_probe_offline(root)
    assert first == second
    result_path = root / m.F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_FILENAME
    assert result_path.read_bytes() == m.canonical_bytes(first)


def test_f6_neighborhood_divergent_existing_artifact_fails_closed_no_overwrite(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<a href="#x">Historical Index Value</a>'
        b'<div><h2 id="x" class="heading-title"><span>Historical Index Value</span></h2></div>',
    )
    first = m.run_f6_section_neighborhood_probe_offline(root)
    result_path = root / m.F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_FILENAME
    on_disk = result_path.read_bytes()
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.write_f6_section_neighborhood_probe_artifact(root, {**first, "status": "TAMPERED"})
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert result_path.read_bytes() == on_disk


def test_f6_neighborhood_missing_diagnostic_lock_fails_closed(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    with pytest.raises(m.V9005StageABlocked):
        m.run_f6_section_neighborhood_probe_offline(root)
    assert not (root / m.F6_SECTION_NEIGHBORHOOD_PROBE_RESULT_FILENAME).exists()


def test_f6_neighborhood_corrupt_diagnostic_lock_fails_closed(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b'<h2 id="x" class="heading-title">Historical Index Value</h2>')
    key = m.source_object_slot_id(
        m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC, m.TOPIX_ROOT_URL,
    )
    (root / "raw" / f"{key}.bin").write_bytes(b"tampered-bytes")
    with pytest.raises(m.V9005StageABlocked):
        m.run_f6_section_neighborhood_probe_offline(root)


def test_f6_neighborhood_wrong_identity_lock_fails_closed(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    # A raw lock under a *different* applicable_period exists, but the
    # dedicated F6_ROOT_STRUCTURE_DIAGNOSTIC lock does not; the reader must
    # not accidentally pick up an unrelated F6 raw lock.
    m.lock_first_complete_payload(
        root,
        source_family=m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
        applicable_period="TOPIX_DISCOVERY_ROOT",
        requested_url=m.TOPIX_ROOT_URL,
        fetch_result=m.FetchResult(
            b'<h2 id="x" class="heading-title">Historical Index Value</h2>', m.TOPIX_ROOT_URL, 200,
        ),
        retrieval_timestamp_utc="2026-08-25T00:00:00Z",
    )
    with pytest.raises(m.V9005StageABlocked):
        m.run_f6_section_neighborhood_probe_offline(root)


def test_f6_neighborhood_invalid_utf8_fails_closed_with_no_fallback(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, b"<h2>Historical Index Value \xff\xfe</h2>")
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert artifact["failure_reason"] == m._F6_PAYLOAD_DECODE_FAILED
    assert artifact["semantic_heading"] is None


@pytest.mark.parametrize(
    "payload",
    [
        b'<a href="#x">Historical Index Value</a><h2 id="x" class="heading-title">Historical Index Value</h3>',
        b'<a href="#x">Historical Index Value</a></div>'
        b'<h2 id="x" class="heading-title">Historical Index Value</h2>',
        b'<a href="#x">Historical Index Value</a><div>'
        b'<h2 id="x" class="heading-title">Historical Index Value</h2>',
    ],
)
def test_f6_neighborhood_malformed_dom_fails_closed(tmp_path: Path, payload: bytes) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, payload)
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert artifact["failure_reason"] == m._F6_MALFORMED_DOM_STRUCTURE


def test_f6_neighborhood_offline_seam_calls_no_network_fetch_retry_ensure_or_run_stage_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<a href="#x">Historical Index Value</a>'
        b'<div><h2 id="x" class="heading-title"><span>Historical Index Value</span></h2></div>',
    )

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network/fetch/retry/orchestration function invoked by offline neighborhood seam")

    monkeypatch.setattr(m, "fetch_once_with_retry", _blocked)
    monkeypatch.setattr(m, "ensure_locked_payload", _blocked)
    monkeypatch.setattr(m, "lock_first_complete_payload", _blocked)
    monkeypatch.setattr(m, "run_stage_a", _blocked)
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.NEIGHBORHOOD_CAPTURED

    for func in (
        m.parse_f6_section_neighborhood_probe,
        m.write_f6_section_neighborhood_probe_artifact,
        m.run_f6_section_neighborhood_probe_offline,
    ):
        params = set(inspect.signature(func).parameters)
        assert params.isdisjoint({"fetcher", "sleep", "clock"})


def test_f6_neighborhood_diagnostic_cannot_populate_f6_inventory(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = _lock_f6_diagnostic(
        root,
        b'<a href="#x">Historical Index Value</a>'
        b'<div><h2 id="x" class="heading-title"><span>Historical Index Value</span></h2></div>',
    )
    diagnostic_slot_id = m.source_object_slot_id(
        m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC, m.TOPIX_ROOT_URL,
    )
    assert diagnostic_slot_id == m.source_object_slot_id(
        locked["source_family"], locked["applicable_period"], locked["requested_url"],
    )
    with pytest.raises(m.V9005StageABlocked):
        m.build_source_inventory({
            (m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC): (diagnostic_slot_id,),
        })
    inventory = m.build_source_inventory()
    f6_records = [r for r in inventory if r["source_family"] == m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE]
    assert len(f6_records) == len(m.inventory_months())
    assert all(r["status"] == m.INVENTORY_MISSING for r in f6_records)


def test_f6_neighborhood_artifact_never_selects_or_binds_a_global_child(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<a href="#x">Historical Index Value</a>'
        b'<div><h2 id="x" class="heading-title"><span>Historical Index Value</span></h2>'
        b'<a href="child.html">Child</a></div>',
    )
    artifact = m.run_f6_section_neighborhood_probe_offline(root)
    assert artifact["status"] == m.NEIGHBORHOOD_CAPTURED
    assert set(artifact.keys()) == {
        "schema_version", "diagnostic", "requested_url", "resolved_url", "byte_length", "sha256",
        "retrieval_timestamp_utc", "status", "failure_reason", "semantic_heading", "parent_container",
        "children", "anchors", "headings",
    }


def test_f6_neighborhood_acquisition_implementation_complete_still_false() -> None:
    assert m.ACQUISITION_IMPLEMENTATION_COMPLETE is False


# --- V9_006_STAGE_A_F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_OFFLINE_IMPLEMENTATION
# Fully offline: every fixture below is synthetic and uses only an existing
# diagnostic raw-lock shape. No test invokes a real network or source-data
# acquisition path.

def test_f6_one_level_h_p_g_mechanics_use_nonliteral_identity(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b"<html><body>"
            b'<nav id="nav-before"><a href="#fragment-token">Historical Index Value</a></nav>'
            b'<article id="grand-token" class="grand z grand">'
            b'<div id="parent-token" class="parent-token">'
            b'<h2 id="fragment-token" class="heading-title"><span>Historical Index Value</span></h2>'
            b"</div></article>"
            b"</body></html>"
        ),
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.EXPANDED_NEIGHBORHOOD_CAPTURED
    assert artifact["semantic_heading"]["id"] == "fragment-token"
    assert artifact["parent_container"]["id"] == "parent-token"
    assert artifact["expanded_container"]["id"] == "grand-token"
    assert artifact["expanded_container"]["tag"] == "article"
    assert artifact["expanded_container"]["classes"] == ["grand", "z"]


def test_f6_one_level_does_not_hardcode_observed_heading_section_literals(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b"<html><body>"
            b'<div id="heading_14" class="heading-title"><p>decoy only</p></div>'
            b'<section class="JPX-section"><p>another decoy</p></section>'
            b'<nav><a href="#mechanically-derived">Historical Index Value</a></nav>'
            b'<article id="actual-grand" class="actual-grand">'
            b'<div id="actual-parent" class="actual-parent">'
            b'<h2 id="mechanically-derived" class="heading-title">'
            b"<span>Historical Index Value</span></h2>"
            b"</div></article>"
            b"</body></html>"
        ),
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.EXPANDED_NEIGHBORHOOD_CAPTURED
    assert artifact["semantic_heading"]["id"] == "mechanically-derived"
    assert artifact["expanded_container"]["id"] == "actual-grand"
    assert artifact["expanded_container"]["tag"] == "article"
    assert artifact["expanded_container"]["classes"] == ["actual-grand"]
    assert "heading_14" not in json.dumps(artifact)
    assert "JPX-section" not in json.dumps(artifact)


def test_f6_one_level_semantic_heading_failure_is_ambiguous(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>',
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.SEMANTIC_HEADING_AMBIGUOUS
    assert artifact["failure_reason"] is None
    assert artifact["semantic_heading"] is None
    assert artifact["parent_container"] is None
    assert artifact["expanded_container"] is None
    assert artifact["children"] == []
    assert artifact["anchors"] == []
    assert artifact["headings"] == []


@pytest.mark.parametrize(
    "payload",
    [
        (
            b'<a href="#target">Historical Index Value</a>'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
        ),
        (
            b'<a href="#target">Historical Index Value</a>'
            b'<div id="parent-only">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b"</div>"
        ),
    ],
)
def test_f6_one_level_missing_ancestor_fails_structure(tmp_path: Path, payload: bytes) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, payload)
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert artifact["semantic_heading"] is None
    assert artifact["parent_container"] is None
    assert artifact["expanded_container"] is None
    assert artifact["children"] == []
    assert artifact["anchors"] == []
    assert artifact["headings"] == []


def test_f6_one_level_direct_children_preserve_order_and_relation_to_p(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<main id="grand">'
            b'<nav id="before"><a href="#target">Historical Index Value</a></nav>'
            b'<div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b"</div>"
            b'<aside id="after"></aside>'
            b"</main>"
        ),
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.EXPANDED_NEIGHBORHOOD_CAPTURED
    children = artifact["children"]
    assert [child["id"] for child in children] == ["before", "parent", "after"]
    assert [child["relation_to_P"] for child in children] == [
        m.ONE_LEVEL_RELATION_BEFORE_P,
        m.ONE_LEVEL_RELATION_P,
        m.ONE_LEVEL_RELATION_AFTER_P,
    ]
    assert sum(child["relation_to_P"] == m.ONE_LEVEL_RELATION_P for child in children) == 1


def test_f6_one_level_requires_p_to_be_exactly_one_direct_child_of_g() -> None:
    grand = m._F6DomElement("main", {}, None)
    parent = m._F6DomElement("div", {}, grand)
    grand.children.extend([parent, parent])
    with pytest.raises(m._F6RootStructureExtractionFailed) as excinfo:
        m._f6_one_level_children(grand, parent)
    assert excinfo.value.reason == m._F6_ONE_LEVEL_PARENT_NOT_DIRECT_CHILD


def test_f6_one_level_all_descendant_anchors_preserve_order_and_ownership(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<main id="grand">'
            b'<nav id="nav-owner"><a href="#target">Historical Index Value</a></nav>'
            b'<a id="direct-anchor" href="direct.html">Direct</a>'
            b'<div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b'<a href="inside.html">Inside parent</a>'
            b"</div>"
            b'<section id="after-owner"><div><a href="nested.html">Nested</a></div></section>'
            b"</main>"
        ),
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.EXPANDED_NEIGHBORHOOD_CAPTURED
    anchors = artifact["anchors"]
    assert [anchor["raw_href"] for anchor in anchors] == [
        "#target", "direct.html", "inside.html", "nested.html",
    ]
    assert [anchor["owning_child_relation_to_P"] for anchor in anchors] == [
        m.ONE_LEVEL_RELATION_BEFORE_P,
        m.ONE_LEVEL_RELATION_BEFORE_P,
        m.ONE_LEVEL_RELATION_P,
        m.ONE_LEVEL_RELATION_AFTER_P,
    ]
    assert [anchor["owning_immediate_element_child_of_G"]["id"] for anchor in anchors] == [
        "nav-owner", "direct-anchor", "parent", "after-owner",
    ]
    direct_anchor = anchors[1]
    assert direct_anchor["owning_immediate_element_child_of_G"]["dom_path"] == direct_anchor["dom_path"]
    assert direct_anchor["owning_immediate_element_child_of_G"]["tag"] == "a"


def test_f6_one_level_raw_href_is_exact_and_never_resolved(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<main id="grand">'
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b'<a href="page.html?a=1&amp;b=2">Exact</a>'
            b"</div></main>"
        ),
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.EXPANDED_NEIGHBORHOOD_CAPTURED
    hrefs = [anchor["raw_href"] for anchor in artifact["anchors"]]
    assert hrefs == ["#target", "page.html?a=1&amp;b=2"]
    assert "page.html?a=1&b=2" not in hrefs
    assert m.TOPIX_ROOT_URL not in hrefs[1]


def test_f6_one_level_ambiguous_raw_href_fails_structure(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<main id="grand">'
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b'<a href="a.html" href="b.html">Ambiguous</a>'
            b"</div></main>"
        ),
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert artifact["failure_reason"] == m._F6_AMBIGUOUS_RAW_HREF_ATTRIBUTE
    assert artifact["semantic_heading"] is None
    assert artifact["parent_container"] is None
    assert artifact["expanded_container"] is None
    assert artifact["anchors"] == []
    assert artifact["headings"] == []


def test_f6_one_level_descendant_headings_preserve_order_normalization_ownership(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<main id="grand">'
            b'<h3 id="before-heading">  First   Heading </h3>'
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical   Index\n Value</span></h2>'
            b'<h1 id="parent-heading"> Parent Heading </h1>'
            b"</div>"
            b'<section id="after"><div><h6>  Last Heading  </h6></div></section>'
            b"</main>"
        ),
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.EXPANDED_NEIGHBORHOOD_CAPTURED
    headings = artifact["headings"]
    assert [(heading["tag"], heading["normalized_heading_text"]) for heading in headings] == [
        ("h3", "First Heading"),
        ("h2", "Historical Index Value"),
        ("h1", "Parent Heading"),
        ("h6", "Last Heading"),
    ]
    assert [heading["owning_immediate_element_child_of_G"]["id"] for heading in headings] == [
        "before-heading", "parent", "parent", "after",
    ]
    assert [heading["owning_child_relation_to_P"] for heading in headings] == [
        m.ONE_LEVEL_RELATION_BEFORE_P,
        m.ONE_LEVEL_RELATION_P,
        m.ONE_LEVEL_RELATION_P,
        m.ONE_LEVEL_RELATION_AFTER_P,
    ]


def test_f6_one_level_artifact_excludes_unrelated_text_and_numerical_topix_values(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<main id="grand">'
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b'<table><tbody><tr><td>2024-01-04</td><td>1783.51</td></tr></tbody></table>'
            b"<p>arbitrary surrounding page text</p>"
            b"</div></main>"
        ),
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.EXPANDED_NEIGHBORHOOD_CAPTURED
    serialized = json.dumps(artifact)
    assert "2024-01-04" not in serialized
    assert "1783.51" not in serialized
    assert "arbitrary surrounding page text" not in serialized


def test_f6_one_level_artifact_has_exact_top_level_key_set(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<main id="grand"><div id="parent">'
            b'<h2 id="target" class="heading-title">'
            b'<span>Historical Index Value</span></h2>'
            b"</div></main>"
            b'<nav><a href="#target">Historical Index Value</a></nav>'
        ),
    )
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.EXPANDED_NEIGHBORHOOD_CAPTURED
    assert set(artifact.keys()) == {
        "schema_version", "diagnostic", "requested_url", "resolved_url",
        "byte_length", "sha256", "retrieval_timestamp_utc", "status",
        "failure_reason", "semantic_heading", "parent_container",
        "expanded_container", "children", "anchors", "headings",
    }
    assert not {"selected_global_child", "ranked_candidates", "score", "resolved_href"} & set(artifact)


def test_f6_one_level_same_locked_bytes_are_byte_identical(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<main id="grand"><div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b"</div></main>"
        ),
    )
    first = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    first_bytes = (root / m.F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_FILENAME).read_bytes()
    second = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    second_bytes = (root / m.F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_FILENAME).read_bytes()
    assert first == second
    assert first_bytes == second_bytes == m.canonical_bytes(first)


def test_f6_one_level_divergent_artifact_fails_closed_without_overwrite(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<main id="grand"><div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b"</div></main>"
        ),
    )
    first = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    result_path = root / m.F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_FILENAME
    original_bytes = result_path.read_bytes()
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.write_f6_one_level_expanded_neighborhood_probe_artifact(
            root, {**first, "status": "TAMPERED"},
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert result_path.read_bytes() == original_bytes


def test_f6_one_level_missing_lock_fails_closed(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    with pytest.raises(m.V9005StageABlocked):
        m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert not (root / m.F6_ONE_LEVEL_EXPANDED_NEIGHBORHOOD_PROBE_RESULT_FILENAME).exists()


def test_f6_one_level_corrupt_lock_fails_closed(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = _lock_f6_diagnostic(
        root,
        b'<main id="grand"><div id="parent">'
        b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
        b"</div></main>",
    )
    key = m.source_object_slot_id(
        locked["source_family"], locked["applicable_period"], locked["requested_url"],
    )
    (root / "raw" / f"{key}.bin").write_bytes(b"tampered-raw-lock")
    with pytest.raises(m.V9005StageABlocked):
        m.run_f6_one_level_expanded_neighborhood_probe_offline(root)


def test_f6_one_level_wrong_identity_lock_fails_closed(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    m.lock_first_complete_payload(
        root,
        source_family=m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
        applicable_period="TOPIX_DISCOVERY_ROOT",
        requested_url=m.TOPIX_ROOT_URL,
        fetch_result=m.FetchResult(
            b'<main id="grand"><div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b"</div></main>",
            m.TOPIX_ROOT_URL,
            200,
        ),
        retrieval_timestamp_utc="2026-08-24T00:00:00Z",
    )
    with pytest.raises(m.V9005StageABlocked):
        m.run_f6_one_level_expanded_neighborhood_probe_offline(root)


@pytest.mark.parametrize(
    "payload",
    [
        (
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<main id="grand"><div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value \xff</span></h2>'
            b"</div></main>"
        ),
        (
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<main id="grand"><div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</h2>'
            b"</div></main>"
        ),
    ],
)
def test_f6_one_level_invalid_utf8_or_malformed_dom_fails_closed(
    tmp_path: Path, payload: bytes,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, payload)
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert artifact["semantic_heading"] is None
    assert artifact["parent_container"] is None
    assert artifact["expanded_container"] is None


def test_f6_one_level_offline_seam_calls_no_network_or_acquisition_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        (
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<main id="grand"><div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b"</div></main>"
        ),
    )

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network or acquisition function invoked by one-level offline seam")

    monkeypatch.setattr(m, "fetch_once_with_retry", _blocked)
    monkeypatch.setattr(m, "ensure_locked_payload", _blocked)
    monkeypatch.setattr(m, "lock_first_complete_payload", _blocked)
    monkeypatch.setattr(m, "run_stage_a", _blocked)
    artifact = m.run_f6_one_level_expanded_neighborhood_probe_offline(root)
    assert artifact["status"] == m.EXPANDED_NEIGHBORHOOD_CAPTURED

    for func in (
        m.parse_f6_one_level_expanded_neighborhood_probe,
        m.write_f6_one_level_expanded_neighborhood_probe_artifact,
        m.run_f6_one_level_expanded_neighborhood_probe_offline,
    ):
        parameters = set(inspect.signature(func).parameters)
        assert parameters.isdisjoint({"fetcher", "sleep", "clock"})


def test_f6_one_level_diagnostic_cannot_populate_f6_inventory(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = _lock_f6_diagnostic(
        root,
        (
            b'<nav><a href="#target">Historical Index Value</a></nav>'
            b'<main id="grand"><div id="parent">'
            b'<h2 id="target" class="heading-title"><span>Historical Index Value</span></h2>'
            b"</div></main>"
        ),
    )
    diagnostic_slot_id = m.source_object_slot_id(
        locked["source_family"], locked["applicable_period"], locked["requested_url"],
    )
    with pytest.raises(m.V9005StageABlocked):
        m.build_source_inventory({
            (m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC): (
                diagnostic_slot_id,
            ),
        })
    inventory = m.build_source_inventory()
    f6_records = [
        record for record in inventory
        if record["source_family"] == m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
    ]
    assert all(record["status"] == m.INVENTORY_MISSING for record in f6_records)


def test_f6_one_level_acquisition_implementation_complete_remains_false() -> None:
    assert m.ACQUISITION_IMPLEMENTATION_COMPLETE is False


# --- V9_006_STAGE_A_F6_GLOBAL_CHILD_LOCATOR_IMPLEMENTATION ------------------
# Fully offline: every fixture below is synthetic and uses only an existing
# F6_ROOT_STRUCTURE_DIAGNOSTIC raw-lock shape. No test fetches the located
# child or inspects child content.

def _global_locator_fixture(
    *,
    body: bytes,
    before_g: bytes = b"",
    after_g: bytes = b"",
    boundary_id: bytes = b"boundary-owner",
    boundary_heading_id: bytes = b"boundary-heading",
) -> bytes:
    return (
        b"<html><body>"
        b'<nav id="semantic-link"><a href="#semantic-token">Historical Index Value</a></nav>'
        + before_g
        + b'<article id="expanded-token" class="scope-token">'
        + b'<div id="parent-token" class="parent-token">'
        + b'<h2 id="semantic-token" class="heading-title">'
        + b"<span>Historical Index Value</span></h2>"
        + b"</div>"
        + body
        + b'<section id="' + boundary_id + b'"><h2 id="' + boundary_heading_id
        + b'"><span>Later structure marker</span></h2></section>'
        + after_g
        + b"</article></body></html>"
    )


def _assert_global_locator_chatgpt_stop(
    root: Path, payload: bytes, *, resolved_url: str | None = None,
) -> None:
    _lock_f6_diagnostic(root, payload, resolved_url=resolved_url)
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_global_child_locator_offline(root)
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED


def test_f6_global_locator_h_p_g_n_and_unique_anchor_are_mechanical(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(
            body=(
                b'<div id="body-token"><a href="objects/blob">not a filename signal</a></div>'
            ),
        ),
        resolved_url="https://www.jpx.co.jp/english/markets/indices/topix/root-final.html",
    )
    result = m.run_f6_global_child_locator_offline(root)
    assert result["status"] == m.GLOBAL_CHILD_LOCATOR_RESOLVED
    assert result["semantic_heading"]["id"] == "semantic-token"
    assert result["parent_container"]["id"] == "parent-token"
    assert result["expanded_container"]["id"] == "expanded-token"
    assert result["boundary_heading"]["id"] == "boundary-heading"
    assert result["boundary_owner"]["id"] == "boundary-owner"
    assert [child["id"] for child in result["section_body_children"]] == ["body-token"]
    assert result["candidate_anchor_count"] == 1
    assert result["candidate_anchor"]["raw_href"] == "objects/blob"
    assert result["resolved_global_child_url"] == (
        "https://www.jpx.co.jp/english/markets/indices/topix/objects/blob"
    )


def test_f6_global_locator_does_not_hardcode_observed_literals(tmp_path: Path) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    payload = (
        b"<html><body>"
        b'<div id="heading_14" class="heading-title"><p>decoy</p></div>'
        b'<section class="JPX-section"><p>decoy</p></section>'
        + _global_locator_fixture(
            before_g=b'<aside id="heading_18"><p>outside scope</p></aside>',
            body=b'<div id="body-actual"><a href="object-without-xls">arbitrary label</a></div>',
        )
        + b"</body></html>"
    )
    _lock_f6_diagnostic(root, payload)
    result = m.run_f6_global_child_locator_offline(root)
    serialized = json.dumps(result)
    assert result["expanded_container"]["id"] == "expanded-token"
    assert result["semantic_heading"]["id"] == "semantic-token"
    assert "heading_14" not in serialized
    assert "heading_18" not in serialized
    assert "JPX-section" not in serialized
    assert "topixyear_e.xls" not in serialized


@pytest.mark.parametrize(
    "payload",
    [
        b'<div><h2 id="not-target" class="heading-title"><span>not the label</span></h2></div>',
        (
            b'<a href="#semantic-token">Historical Index Value</a>'
            b'<h2 id="semantic-token" class="heading-title"><span>Historical Index Value</span></h2>'
        ),
        (
            b'<a href="#semantic-token">Historical Index Value</a>'
            b'<div id="parent-only"><h2 id="semantic-token" class="heading-title">'
            b'<span>Historical Index Value</span></h2></div>'
        ),
    ],
)
def test_f6_global_locator_missing_or_ambiguous_h_p_g_fails_closed(
    tmp_path: Path, payload: bytes,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _assert_global_locator_chatgpt_stop(root, payload)


def test_f6_global_locator_p_must_be_exactly_one_direct_child_of_g(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(
            body=b'<div id="body"><a href="object">one</a></div>',
        ),
    )

    def _duplicate_parent(*_args: object, **_kwargs: object) -> object:
        raise m._F6RootStructureExtractionFailed(m._F6_ONE_LEVEL_PARENT_NOT_DIRECT_CHILD)

    monkeypatch.setattr(m, "_f6_one_level_children", _duplicate_parent)
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_global_child_locator_offline(root)
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED


def test_f6_global_locator_no_later_boundary_fails_closed_without_end_fallback(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    payload = (
        b'<nav><a href="#semantic-token">Historical Index Value</a></nav>'
        b'<main id="expanded-token"><div id="parent-token">'
        b'<h2 id="semantic-token" class="heading-title"><span>Historical Index Value</span></h2>'
        b'</div><div id="body"><a href="object">one</a></div></main>'
    )
    _assert_global_locator_chatgpt_stop(root, payload)


def test_f6_global_locator_earliest_n_excludes_h2_inside_p_and_before_p(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    payload = (
        b'<nav><a href="#semantic-token">Historical Index Value</a></nav>'
        b'<main id="expanded-token">'
        b'<div id="before-owner"><h2 id="before-heading">Before P</h2></div>'
        b'<div id="parent-token">'
        b'<h2 id="semantic-token" class="heading-title"><span>Historical Index Value</span></h2>'
        b'<h2 id="inside-parent">Inside P</h2>'
        b'</div>'
        b'<div id="body-one"><a href="object-without-extension">one</a></div>'
        b'<div id="first-boundary"><h2 id="first-n">First N</h2></div>'
        b'<div id="second-boundary"><h2 id="second-n">Second N</h2></div>'
        b'</main>'
    )
    _lock_f6_diagnostic(root, payload)
    result = m.run_f6_global_child_locator_offline(root)
    assert result["boundary_heading"]["id"] == "first-n"
    assert result["boundary_owner"]["id"] == "first-boundary"
    assert [child["id"] for child in result["section_body_children"]] == ["body-one"]


@pytest.mark.parametrize(
    "body",
    [
        b'<div id="body-empty"></div>',
        (
            b'<div id="body-many"><a href="first">first</a>'
            b'<span><a href="second">second</a></span></div>'
        ),
    ],
)
def test_f6_global_locator_candidate_anchor_count_must_be_exactly_one(
    tmp_path: Path, body: bytes,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _assert_global_locator_chatgpt_stop(root, _global_locator_fixture(body=body))


def test_f6_global_locator_direct_child_anchor_is_included_and_owns_itself(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(
            body=b'<a id="direct-anchor" href="direct-object">visible text is not a heuristic</a>',
        ),
    )
    result = m.run_f6_global_child_locator_offline(root)
    assert result["candidate_anchor"]["raw_href"] == "direct-object"
    assert result["candidate_anchor"]["dom_path"][-1]["id"] == "direct-anchor"
    assert result["resolved_global_child_url"].endswith("/topix/direct-object")


def test_f6_global_locator_nested_anchor_is_included_under_body_child(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(
            body=b'<div id="body-nested"><span><a href="nested-object">nested</a></span></div>',
        ),
    )
    result = m.run_f6_global_child_locator_offline(root)
    assert result["candidate_anchor"]["raw_href"] == "nested-object"
    assert result["resolved_global_child_url"].endswith("/topix/nested-object")


def test_f6_global_locator_preserves_raw_href_and_does_not_use_requested_url(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    requested_url = m.TOPIX_ROOT_URL
    final_url = "https://www.jpx.co.jp/english/alternate/root-final.html"
    payload = _global_locator_fixture(
        body=b'<div id="body"><a href="object?x=1&amp;y=2">entity spelling</a></div>',
    )
    _lock_f6_diagnostic(root, payload, resolved_url=final_url)
    result = m.run_f6_global_child_locator_offline(root)
    assert result["requested_url"] == requested_url
    assert result["resolved_url"] == final_url
    assert result["candidate_anchor"]["raw_href"] == "object?x=1&amp;y=2"
    assert result["resolved_global_child_url"] == (
        "https://www.jpx.co.jp/english/alternate/object?x=1&amp;y=2"
    )
    assert requested_url not in result["resolved_global_child_url"]


@pytest.mark.parametrize(
    "href",
    [
        b"https://evil.example/object",
        b"http://www.jpx.co.jp/object",
    ],
)
def test_f6_global_locator_enforces_resolved_https_allowed_domain(
    tmp_path: Path, href: bytes,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(body=b'<div id="body"><a href="' + href + b'">unsafe</a></div>'),
    )
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_global_child_locator_offline(root)
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED
    assert excinfo.value.network_request_count == 0


def test_f6_global_locator_absent_or_ambiguous_raw_href_fails_closed(
    tmp_path: Path,
) -> None:
    payloads = [
        _global_locator_fixture(body=b'<div id="body"><a>missing</a></div>'),
        _global_locator_fixture(
            body=b'<div id="body"><a href="first" href="second">ambiguous</a></div>',
        ),
    ]
    for index, payload in enumerate(payloads):
        root = m.initialize_output_root(tmp_path / f"out-{index}")
        _lock_f6_diagnostic(root, payload)
        with pytest.raises(m.V9005StageABlocked) as excinfo:
            m.run_f6_global_child_locator_offline(root)
        assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED


def test_f6_global_locator_url_failure_translation_does_not_change_f2_f3_f4_f7() -> None:
    """The shared resolver keeps its existing IMPLEMENTATION_FAILURE class.

    F7 has a fixed template rather than a shared-page-link resolver, so its
    invalid template input is the corresponding unchanged local failure.
    """
    failure_calls = (
        lambda: m.resolve_monthly_statistics_year_page_url(
            _year_selector_html(("https://evil.example/2020.html", "2020")),
            m.MONTHLY_STATISTICS_ROOT_URL,
            2020,
        ),
        lambda: m.resolve_delisted_company_year_url(
            b'<a href="https://evil.example/2020.html">2020</a>',
            "https://www.jpx.co.jp/english/listing/stocks/delisted/archive/index.html",
            2020,
        ),
        lambda: m.resolve_monthly_statistics_evidence_url(
            _monthly_statistics_year_html(f2_href="https://evil.example/f2.xlsx"),
            "https://www.jpx.co.jp/monthly/2020.html",
            m.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT,
            "2020-03",
            selected_year=2020,
        ),
        lambda: m.resolve_monthly_statistics_evidence_url(
            _monthly_statistics_year_html(f4_href="https://evil.example/f4.xlsx"),
            "https://www.jpx.co.jp/monthly/2020.html",
            m.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE,
            "2020-03",
            selected_year=2020,
        ),
        lambda: m.resolve_f7_calendar_url(2020, 13),
    )
    for failure_call in failure_calls:
        with pytest.raises(m.V9005StageABlocked) as excinfo:
            failure_call()
        assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE


def test_f6_global_locator_corrupt_lock_remains_offline_implementation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = _lock_f6_diagnostic(
        root,
        _global_locator_fixture(body=b'<div id="body"><a href="opaque-object">x</a></div>'),
    )
    key = m.source_object_slot_id(
        locked["source_family"], locked["applicable_period"], locked["requested_url"],
    )
    (root / "raw" / f"{key}.bin").write_bytes(b"corrupt")

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("corrupt-lock path invoked network or acquisition code")

    monkeypatch.setattr(m, "fetch_once_with_retry", _blocked)
    monkeypatch.setattr(m, "ensure_locked_payload", _blocked)
    monkeypatch.setattr(m, "lock_first_complete_payload", _blocked)
    monkeypatch.setattr(m, "run_stage_a", _blocked)
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_global_child_locator_offline(root)
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE


def test_f6_global_locator_uses_no_filename_extension_or_text_heuristic(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(
            body=b'<div id="body"><a href="opaque?id=7">not preferred</a></div>',
        ),
    )
    result = m.run_f6_global_child_locator_offline(root)
    assert result["candidate_anchor_count"] == 1
    assert result["resolved_global_child_url"].endswith("/topix/opaque?id=7")


def test_f6_global_locator_excludes_unrelated_text_table_cells_and_year_values(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(
            body=(
                b'<div id="body"><table><tbody><tr><td>2017-01-04</td>'
                b'<td>1783.51</td></tr></tbody></table>'
                b"<p>arbitrary surrounding text</p>"
                b'<a href="opaque-object">anchor</a></div>'
            ),
        ),
    )
    result = m.run_f6_global_child_locator_offline(root)
    serialized = json.dumps(result)
    assert "2017-01-04" not in serialized
    assert "1783.51" not in serialized
    assert "arbitrary surrounding text" not in serialized


def test_f6_global_locator_result_has_no_selection_or_inventory_status_fields(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(body=b'<div id="body"><a href="opaque-object">x</a></div>'),
    )
    result = m.run_f6_global_child_locator_offline(root)
    assert result["status"] not in {m.INVENTORY_AVAILABLE, m.INVENTORY_MISSING}
    assert not {
        "selected_global_child", "bound_global_child", "ranked_candidates",
        "score", "candidate_scores", "resolved_href",
    } & set(result)
    assert result["resolved_global_child_url"]


def test_f6_global_locator_same_locked_input_is_deterministic_and_does_not_write_raw_lock(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    locked = _lock_f6_diagnostic(
        root,
        _global_locator_fixture(body=b'<div id="body"><a href="opaque-object">x</a></div>'),
    )
    raw_files_before = sorted(path.name for path in (root / "raw").iterdir())
    first = m.run_f6_global_child_locator_offline(root)
    second = m.run_f6_global_child_locator_offline(root)
    raw_files_after = sorted(path.name for path in (root / "raw").iterdir())
    assert first == second
    assert first["sha256"] == locked["sha256"]
    assert raw_files_after == raw_files_before


def test_f6_global_locator_offline_seam_calls_no_fetch_retry_lock_or_stage_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(body=b'<div id="body"><a href="opaque-object">x</a></div>'),
    )

    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("global locator invoked network or acquisition code")

    monkeypatch.setattr(m, "fetch_once_with_retry", _blocked)
    monkeypatch.setattr(m, "ensure_locked_payload", _blocked)
    monkeypatch.setattr(m, "lock_first_complete_payload", _blocked)
    monkeypatch.setattr(m, "run_stage_a", _blocked)
    result = m.run_f6_global_child_locator_offline(root)
    assert result["status"] == m.GLOBAL_CHILD_LOCATOR_RESOLVED
    for func in (
        m.parse_f6_global_child_locator,
        m.run_f6_global_child_locator_offline,
    ):
        parameters = set(inspect.signature(func).parameters)
        assert parameters.isdisjoint({"fetcher", "sleep", "clock"})


@pytest.mark.parametrize(
    "payload_kind",
    ["missing", "corrupt", "wrong_identity"],
)
def test_f6_global_locator_missing_corrupt_or_wrong_identity_lock_fails_closed(
    tmp_path: Path, payload_kind: str,
) -> None:
    root = m.initialize_output_root(tmp_path / payload_kind)
    if payload_kind == "missing":
        with pytest.raises(m.V9005StageABlocked):
            m.run_f6_global_child_locator_offline(root)
        return
    if payload_kind == "corrupt":
        locked = _lock_f6_diagnostic(
            root,
            _global_locator_fixture(body=b'<div id="body"><a href="opaque-object">x</a></div>'),
        )
        key = m.source_object_slot_id(
            locked["source_family"], locked["applicable_period"], locked["requested_url"],
        )
        (root / "raw" / f"{key}.bin").write_bytes(b"corrupt")
    else:
        m.lock_first_complete_payload(
            root,
            source_family=m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE,
            applicable_period="OTHER_ROOT",
            requested_url=m.TOPIX_ROOT_URL,
            fetch_result=m.FetchResult(
                _global_locator_fixture(body=b'<div id="body"><a href="opaque-object">x</a></div>'),
                m.TOPIX_ROOT_URL,
                200,
            ),
            retrieval_timestamp_utc="2026-08-24T00:00:00Z",
        )
    with pytest.raises(m.V9005StageABlocked):
        m.run_f6_global_child_locator_offline(root)


@pytest.mark.parametrize(
    "payload",
    [
        (
            b'<nav><a href="#semantic-token">Historical Index Value</a></nav>'
            b'<main><div><h2 id="semantic-token" class="heading-title">'
            b'<span>Historical Index Value \xff</span></h2></div></main>'
        ),
        (
            b'<nav><a href="#semantic-token">Historical Index Value</a></nav>'
            b'<main><div><h2 id="semantic-token" class="heading-title">'
            b'<span>Historical Index Value</h2></div></main>'
        ),
    ],
)
def test_f6_global_locator_invalid_utf8_or_malformed_dom_fails_closed(
    tmp_path: Path, payload: bytes,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(root, payload)
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_global_child_locator_offline(root)
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE


def test_f6_global_locator_does_not_populate_f6_inventory_or_acquisition_state(
    tmp_path: Path,
) -> None:
    root = m.initialize_output_root(tmp_path / "out")
    _lock_f6_diagnostic(
        root,
        _global_locator_fixture(body=b'<div id="body"><a href="opaque-object">x</a></div>'),
    )
    result = m.run_f6_global_child_locator_offline(root)
    assert result["status"] == m.GLOBAL_CHILD_LOCATOR_RESOLVED
    inventory = m.build_source_inventory()
    f6_records = [
        record for record in inventory
        if record["source_family"] == m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
    ]
    assert f6_records
    assert all(record["status"] == m.INVENTORY_MISSING for record in f6_records)
    assert m.ACQUISITION_IMPLEMENTATION_COMPLETE is False



# --- V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_NETWORK_EXECUTOR ----------------
# Every fetcher here is synthetic (no real socket, no real network). This
# section proves the network-executor plumbing without ever performing a
# real network request, per this task's authority boundary.

def _f6_captured_fetcher(url: str) -> m.FetchResult:
    return m.FetchResult(b"<h2>Historical Index Value</h2>", url, 200)


def test_f6_root_structure_network_wrong_confirmation_fails_closed_before_filesystem_or_fetch(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return _f6_captured_fetcher(url)

    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_root_structure_probe_network(
            output_root=output_root, confirmation="wrong-token",
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.GOVERNANCE_FAILURE
    assert calls == []
    assert not output_root.exists()


def test_f6_root_structure_network_production_stage_a_confirmation_does_not_satisfy_this_gate(
    tmp_path: Path,
) -> None:
    assert m.CONFIRMATION != m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return _f6_captured_fetcher(url)

    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_root_structure_probe_network(
            output_root=output_root, confirmation=m.CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.GOVERNANCE_FAILURE
    assert calls == []


def test_f6_root_structure_network_requests_only_topix_root_url_once(tmp_path: Path) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return _f6_captured_fetcher(url)

    output_root = tmp_path / "out"
    artifact = m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert calls == [m.TOPIX_ROOT_URL]
    assert artifact["requested_url"] == m.TOPIX_ROOT_URL
    assert artifact["network_request_count"] == 1
    assert artifact["status"] == m.STRUCTURE_CAPTURED


def test_f6_root_structure_network_same_domain_redirect_preserves_requested_and_resolved_url(
    tmp_path: Path,
) -> None:
    redirected_url = "https://www.jpx.co.jp/english/markets/indices/topix/redirected.html"

    def fetcher(_url: str) -> m.FetchResult:
        return m.FetchResult(b"<h2>Historical Index Value</h2>", redirected_url, 200)

    output_root = tmp_path / "out"
    artifact = m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["requested_url"] == m.TOPIX_ROOT_URL
    assert artifact["resolved_url"] == redirected_url
    locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC, m.TOPIX_ROOT_URL,
    )
    assert locked is not None
    assert locked["requested_url"] == m.TOPIX_ROOT_URL
    assert locked["resolved_url"] == redirected_url


def test_f6_root_structure_network_raw_lock_exists_before_parser_is_invoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, bool] = {}
    original_offline = m.run_f6_root_structure_probe_offline

    def spy_offline(root: object) -> dict[str, object]:
        locked = m.read_locked_payload(
            root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC, m.TOPIX_ROOT_URL,
        )
        seen["raw_lock_exists_when_offline_seam_called"] = locked is not None
        return original_offline(root)

    monkeypatch.setattr(m, "run_f6_root_structure_probe_offline", spy_offline)
    output_root = tmp_path / "out"
    m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=_f6_captured_fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert seen["raw_lock_exists_when_offline_seam_called"] is True


def test_f6_root_structure_network_successful_payload_produces_offline_seam_artifact(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "out"
    artifact = m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=_f6_captured_fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["status"] == m.STRUCTURE_CAPTURED
    assert artifact["label_occurrence_count"] == 1
    result_path = output_root / m.F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME
    assert result_path.exists()
    on_disk = json.loads(result_path.read_bytes())
    assert on_disk["status"] == m.STRUCTURE_CAPTURED
    assert on_disk["schema_version"] == m.F6_ROOT_STRUCTURE_PROBE_RESULT_SCHEMA_VERSION


def test_f6_root_structure_network_ambiguous_result_still_produces_durable_artifact_no_refetch(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return m.FetchResult(b"<h2>Nothing Here</h2>", url, 200)

    output_root = tmp_path / "out"
    artifact = m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["status"] == m.STRUCTURE_AMBIGUOUS
    assert artifact["label_occurrence_count"] == 0
    assert (output_root / m.F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME).exists()
    assert calls == [m.TOPIX_ROOT_URL]


def test_f6_root_structure_network_extraction_failed_preserves_raw_lock_no_refetch(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    malformed = b"<h2>Historical Index Value</h3>"

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return m.FetchResult(malformed, url, 200)

    output_root = tmp_path / "out"
    artifact = m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert calls == [m.TOPIX_ROOT_URL]
    locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC, m.TOPIX_ROOT_URL,
    )
    assert locked is not None
    assert locked["raw"] == malformed


def test_f6_root_structure_network_retryable_transport_before_payload_uses_existing_retry(
    tmp_path: Path,
) -> None:
    attempts: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        attempts.append(url)
        if len(attempts) < 2:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        return _f6_captured_fetcher(url)

    output_root = tmp_path / "out"
    artifact = m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert len(attempts) == 2
    assert artifact["network_request_count"] == 2
    assert artifact["status"] == m.STRUCTURE_CAPTURED


def test_f6_root_structure_network_exhausted_transport_failure_produces_no_artifact(
    tmp_path: Path,
) -> None:
    def fetcher(url: str) -> m.FetchResult:
        raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)

    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_root_structure_probe_network(
            output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.reason == m.PLUMBING_FAILURE_RETRIABLE
    assert not (output_root / m.F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME).exists()


def test_f6_root_structure_network_off_domain_redirect_rejected(tmp_path: Path) -> None:
    def fetcher(_url: str) -> m.FetchResult:
        return m.FetchResult(b"payload", "https://evil.example/redirected", 200)

    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_root_structure_probe_network(
            output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.reason == "OFF_DOMAIN_REDIRECT_REJECTED"
    assert not (output_root / m.F6_ROOT_STRUCTURE_PROBE_RESULT_FILENAME).exists()


def test_f6_root_structure_network_rerun_against_existing_output_root_fails_closed(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def fetcher(url: str) -> m.FetchResult:
        calls.append(url)
        return _f6_captured_fetcher(url)

    output_root = tmp_path / "out"
    m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert len(calls) == 1
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_root_structure_probe_network(
            output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert len(calls) == 1  # never acquired/refetched on rerun


def test_f6_root_structure_network_never_invokes_run_stage_a(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("run_stage_a invoked by the F6 diagnostic network executor")

    monkeypatch.setattr(m, "run_stage_a", _blocked)
    output_root = tmp_path / "out"
    artifact = m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=_f6_captured_fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["status"] == m.STRUCTURE_CAPTURED


def test_f6_root_structure_network_no_real_socket_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real network socket attempted during F6 network executor test")

    monkeypatch.setattr(socket, "socket", _blocked)
    output_root = tmp_path / "out"
    artifact = m.run_f6_root_structure_probe_network(
        output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
        fetcher=_f6_captured_fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["status"] == m.STRUCTURE_CAPTURED


def test_f6_root_structure_network_acquisition_implementation_complete_still_false() -> None:
    assert m.ACQUISITION_IMPLEMENTATION_COMPLETE is False


# --- V9_006_STAGE_A_F6_ROOT_STRUCTURE_PROBE_NETWORK_EXECUTOR: CLI script ----

F6_NETWORK_CLI_MODULE_NAME = "run_v9_006_f6_root_structure_probe"
F6_NETWORK_CLI_CONFIRMATION_ENV = "V9_006_F6_ROOT_STRUCTURE_PROBE_CONFIRMATION"


def _f6_network_cli() -> object:
    scripts_directory = str(ROOT / "scripts")
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    return importlib.reload(importlib.import_module(F6_NETWORK_CLI_MODULE_NAME))


def test_cli_f6_root_structure_never_imports_production_run_stage_a() -> None:
    text = (ROOT / "scripts" / "run_v9_006_f6_root_structure_probe.py").read_text(encoding="utf-8")
    assert "run_stage_a" not in text


def test_cli_f6_root_structure_missing_confirmation_makes_zero_fetch_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.delenv(F6_NETWORK_CLI_CONFIRMATION_ENV, raising=False)
    cli = _f6_network_cli()
    calls: list[str] = []
    monkeypatch.setattr(cli, "_production_fetcher", lambda url: calls.append(url) or m.FetchResult(b"x", url, 200))
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root)])
    assert exit_code == 2
    assert calls == []
    assert not output_root.exists()
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["execution_result"] == "BLOCKED"
    assert payload["failure_class"] == "GOVERNANCE_FAILURE"
    assert payload["network_request_count"] == 0


def test_cli_f6_root_structure_wrong_confirmation_makes_zero_fetch_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.setenv(F6_NETWORK_CLI_CONFIRMATION_ENV, "wrong-token")
    cli = _f6_network_cli()
    calls: list[str] = []
    monkeypatch.setattr(cli, "_production_fetcher", lambda url: calls.append(url) or m.FetchResult(b"x", url, 200))
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root)])
    assert exit_code == 2
    assert calls == []
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["failure_class"] == "GOVERNANCE_FAILURE"
    assert payload["network_request_count"] == 0


def test_cli_f6_root_structure_production_confirmation_does_not_satisfy_this_gate(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.setenv(F6_NETWORK_CLI_CONFIRMATION_ENV, m.CONFIRMATION)
    cli = _f6_network_cli()
    calls: list[str] = []
    monkeypatch.setattr(cli, "_production_fetcher", lambda url: calls.append(url) or m.FetchResult(b"x", url, 200))
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root)])
    assert exit_code == 2
    assert calls == []
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["failure_class"] == "GOVERNANCE_FAILURE"


def test_cli_f6_root_structure_safe_stdout_excludes_sensitive_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.setenv(F6_NETWORK_CLI_CONFIRMATION_ENV, m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION)
    cli = _f6_network_cli()

    def fake_fetcher(url: str) -> m.FetchResult:
        # The anchor and the unrelated numeric TOPIX-like table are
        # siblings of the matching <h2>, not descendants -- a descendant's
        # own text would pollute the h2's aggregated label text and break
        # the exact match.
        payload = (
            b"<div><a href=\"page.html?a=1&amp;b=2\">CSV</a>"
            b"<table><tr><td>2024-01-04</td><td>1783.51</td></tr></table>"
            b"<h2>Historical Index Value</h2></div>"
        )
        return m.FetchResult(payload, url, 200)

    monkeypatch.setattr(cli, "_production_fetcher", fake_fetcher)
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root)])
    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert set(payload) == {
        "status", "label_occurrence_count", "requested_url", "resolved_url",
        "http_status", "byte_length", "sha256", "retrieval_timestamp_utc",
        "network_request_count", "artifact_path",
    }
    assert payload["status"] == m.STRUCTURE_CAPTURED
    assert payload["label_occurrence_count"] == 1
    assert "occurrences" not in out
    assert "anchors" not in out
    assert "page.html?a=1&amp;b=2" not in out
    assert "Historical Index Value" not in out
    assert "CSV" not in out
    assert "1783.51" not in out
    assert "2024-01-04" not in out
    assert Path(payload["artifact_path"]).exists()


def test_cli_f6_root_structure_extraction_failure_still_prints_only_safe_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.setenv(F6_NETWORK_CLI_CONFIRMATION_ENV, m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION)
    cli = _f6_network_cli()

    def fake_fetcher(url: str) -> m.FetchResult:
        return m.FetchResult(b"<h2>Historical Index Value</h3>", url, 200)

    monkeypatch.setattr(cli, "_production_fetcher", fake_fetcher)
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root)])
    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["status"] == m.STRUCTURE_EXTRACTION_FAILED
    assert set(payload) == {
        "status", "label_occurrence_count", "requested_url", "resolved_url",
        "http_status", "byte_length", "sha256", "retrieval_timestamp_utc",
        "network_request_count", "artifact_path",
    }


# --- V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_IMPLEMENTATION
# Every fetcher here is synthetic (no real socket, no real network). This
# section proves the production ROOT/GLOBAL raw-acquisition network-executor
# plumbing without ever performing a real network request, per this task's
# authority boundary. Reuses _global_locator_fixture (defined above) for
# synthetic ROOT bytes containing a mechanically resolvable GLOBAL child
# anchor -- the exact same reviewed locator, never a diagnostic reader.

def _production_root_payload(href: bytes) -> bytes:
    return _global_locator_fixture(body=b'<div id="body"><a href="' + href + b'">x</a></div>')


def _production_no_candidate_payload() -> bytes:
    # SECTION_BODY has zero descendant anchors -> candidate_anchor_count=0.
    return _global_locator_fixture(body=b'<div id="body">no anchor here</div>')


def _production_routing_fetcher(
    root_payload: bytes,
    *,
    root_url: str | None = None,
    child_payload: bytes = b"synthetic-child-bytes-not-a-real-spreadsheet",
    calls: list[str] | None = None,
    root_attempts_before_success: int = 0,
    child_attempts_before_success: int = 0,
):
    state = {"root_seen": 0, "child_seen": 0}

    def fetcher(url: str) -> m.FetchResult:
        if calls is not None:
            calls.append(url)
        if url == m.TOPIX_ROOT_URL:
            state["root_seen"] += 1
            if state["root_seen"] <= root_attempts_before_success:
                raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
            return m.FetchResult(root_payload, root_url or url, 200)
        state["child_seen"] += 1
        if state["child_seen"] <= child_attempts_before_success:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        return m.FetchResult(child_payload, url, 200)

    return fetcher


def test_f6_production_acquisition_wrong_confirmation_zero_filesystem_and_fetch(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"), calls=calls)
    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation="wrong-token",
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.GOVERNANCE_FAILURE
    assert calls == []
    assert not output_root.exists()


def test_f6_production_acquisition_diagnostic_confirmation_does_not_satisfy_this_gate(
    tmp_path: Path,
) -> None:
    assert m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION != m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION
    calls: list[str] = []
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"), calls=calls)
    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_ROOT_STRUCTURE_PROBE_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.GOVERNANCE_FAILURE
    assert calls == []


def test_f6_production_acquisition_stage_a_confirmation_does_not_satisfy_this_gate(
    tmp_path: Path,
) -> None:
    assert m.CONFIRMATION != m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION
    calls: list[str] = []
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"), calls=calls)
    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.GOVERNANCE_FAILURE
    assert calls == []


def test_f6_production_acquisition_existing_output_root_fails_closed_before_fetch(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "out"
    m.initialize_output_root(output_root)  # simulates a prior attempt/state
    calls: list[str] = []
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"), calls=calls)
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert calls == []
    assert not (output_root / m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_FILENAME).exists()


def test_f6_production_acquisition_receipt_durable_before_first_fetch(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    seen: dict[str, bool] = {}

    def fetcher(url: str) -> m.FetchResult:
        seen["receipt_exists_at_first_fetch"] = (
            output_root / m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_FILENAME
        ).exists()
        if url == m.TOPIX_ROOT_URL:
            return m.FetchResult(_production_root_payload(b"child-object-alpha"), url, 200)
        return m.FetchResult(b"child-bytes", url, 200)

    m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert seen["receipt_exists_at_first_fetch"] is True


def test_f6_production_acquisition_full_success_locks_root_then_child(tmp_path: Path) -> None:
    calls: list[str] = []
    root_payload = _production_root_payload(b"child-object-alpha")
    fetcher = _production_routing_fetcher(root_payload, calls=calls)
    output_root = tmp_path / "out"
    artifact = m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["status"] == "F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_COMPLETE"
    assert artifact["gate_consumed"] is True
    assert artifact["locator_status"] == m.GLOBAL_CHILD_LOCATOR_RESOLVED
    assert artifact["candidate_anchor_count"] == 1
    assert calls[0] == m.TOPIX_ROOT_URL

    expected_child_url = urllib.parse.urljoin(m.TOPIX_ROOT_URL, "child-object-alpha")
    assert artifact["child"]["requested_url"] == expected_child_url
    assert calls == [m.TOPIX_ROOT_URL, expected_child_url]

    root_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_DISCOVERY_ROOT, m.TOPIX_ROOT_URL,
    )
    assert root_locked is not None
    assert root_locked["raw"] == root_payload

    child_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_GLOBAL_2017_2025, expected_child_url,
    )
    assert child_locked is not None
    assert child_locked["raw"] == b"synthetic-child-bytes-not-a-real-spreadsheet"

    receipt_path = output_root / m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_FILENAME
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["gate_consumed"] is True
    assert receipt["confirmation_contract"] == m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION
    assert receipt["schema_version"] == m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_SCHEMA_VERSION
    assert m.ACQUISITION_IMPLEMENTATION_COMPLETE is False


def test_f6_production_acquisition_root_locked_before_locator_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    original_locator = m.parse_f6_global_child_locator
    seen: dict[str, bool] = {}

    def spy_locator(locked: object) -> dict[str, object]:
        root_locked = m.read_locked_payload(
            output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_DISCOVERY_ROOT, m.TOPIX_ROOT_URL,
        )
        seen["root_locked_when_locator_called"] = root_locked is not None
        return original_locator(locked)

    monkeypatch.setattr(m, "parse_f6_global_child_locator", spy_locator)
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"))
    m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert seen["root_locked_when_locator_called"] is True


def test_f6_production_acquisition_locator_uses_root_final_resolved_url(tmp_path: Path) -> None:
    redirected_root_url = "https://www.jpx.co.jp/english/markets/indices/topix/nested/page.html"
    root_payload = _production_root_payload(b"child-object-alpha")
    fetcher = _production_routing_fetcher(root_payload, root_url=redirected_root_url)
    output_root = tmp_path / "out"
    artifact = m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    expected_from_resolved = urllib.parse.urljoin(redirected_root_url, "child-object-alpha")
    expected_from_requested = urllib.parse.urljoin(m.TOPIX_ROOT_URL, "child-object-alpha")
    assert expected_from_resolved != expected_from_requested
    assert artifact["root"]["resolved_url"] == redirected_root_url
    assert artifact["child"]["requested_url"] == expected_from_resolved


def test_f6_production_acquisition_never_uses_diagnostic_reader_or_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("diagnostic reader invoked by production acquisition")

    monkeypatch.setattr(m, "read_f6_root_structure_diagnostic_lock", _blocked)
    output_root = tmp_path / "out"
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"))
    m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    diagnostic_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.F6_ROOT_STRUCTURE_DIAGNOSTIC, m.TOPIX_ROOT_URL,
    )
    assert diagnostic_locked is None


@pytest.mark.parametrize("href", [b"child-object-alpha", b"different-name-beta.xls"])
def test_f6_production_acquisition_child_url_is_dynamic_not_hardcoded(tmp_path: Path, href: bytes) -> None:
    output_root = tmp_path / ("out-" + href.decode())
    fetcher = _production_routing_fetcher(_production_root_payload(href))
    artifact = m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    expected = urllib.parse.urljoin(m.TOPIX_ROOT_URL, href.decode())
    assert artifact["child"]["requested_url"] == expected


def test_f6_production_acquisition_chatgpt_decision_required_locator_failure_keeps_root_zero_child(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    fetcher = _production_routing_fetcher(_production_no_candidate_payload(), calls=calls)
    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED
    assert excinfo.value.network_request_count == 1  # the ROOT fetch, preserved
    assert calls == [m.TOPIX_ROOT_URL]
    root_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_DISCOVERY_ROOT, m.TOPIX_ROOT_URL,
    )
    assert root_locked is not None
    child_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_GLOBAL_2017_2025,
        urllib.parse.urljoin(m.TOPIX_ROOT_URL, "child-object-alpha"),
    )
    assert child_locked is None
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True


def test_f6_production_acquisition_off_domain_child_href_is_chatgpt_decision_required(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    fetcher = _production_routing_fetcher(
        _production_root_payload(b"https://evil.example/child.xls"), calls=calls,
    )
    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.CHATGPT_DECISION_REQUIRED
    assert calls == [m.TOPIX_ROOT_URL]


def test_f6_production_acquisition_implementation_failure_locator_failure_keeps_root_zero_child(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    fetcher = _production_routing_fetcher(b"<h2>Historical Index Value \xff\xfe</h2>", calls=calls)
    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert excinfo.value.network_request_count == 1  # the ROOT fetch, preserved
    assert calls == [m.TOPIX_ROOT_URL]
    root_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_DISCOVERY_ROOT, m.TOPIX_ROOT_URL,
    )
    assert root_locked is not None
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True


def test_f6_production_acquisition_rerun_after_consumed_receipt_zero_new_fetches(tmp_path: Path) -> None:
    calls: list[str] = []
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"), calls=calls)
    output_root = tmp_path / "out"
    m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert len(calls) == 2
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert len(calls) == 2  # never refetched/reacquired on rerun
    # A rerun against an already-consumed receipt must still mechanically
    # report gate_consumed=true (durable state), never look PRE_GATE.
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True


def test_f6_production_acquisition_child_bytes_never_parsed_or_decoded(tmp_path: Path) -> None:
    garbage_child = b"\x00\x01\xff\xfe\xfd not a spreadsheet, not valid utf-8 \x80\x81"
    fetcher = _production_routing_fetcher(
        _production_root_payload(b"child-object-alpha"), child_payload=garbage_child,
    )
    output_root = tmp_path / "out"
    artifact = m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["status"] == "F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_COMPLETE"
    assert artifact["child"]["sha256"] == m.sha256_bytes(garbage_child)
    assert artifact["child"]["byte_length"] == len(garbage_child)


def test_f6_production_acquisition_uses_unchanged_retry_policy_constants() -> None:
    assert m.MAX_ATTEMPTS == 3
    assert m.MAX_RETRIES == 2
    assert m.BACKOFF_SECONDS == (5, 30)


def test_f6_production_acquisition_retries_root_and_child_transport_failures(tmp_path: Path) -> None:
    fetcher = _production_routing_fetcher(
        _production_root_payload(b"child-object-alpha"),
        root_attempts_before_success=1, child_attempts_before_success=1,
    )
    output_root = tmp_path / "out"
    artifact = m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["root_network_request_count"] == 2
    assert artifact["child_network_request_count"] == 2
    assert artifact["network_request_count"] == 4


# --- V9_006_F6_PRODUCTION_RAW_IMPL_HIGH_1_POST_GATE_SAFE_REPORT_PROVENANCE --
# Cumulative network-request-count and gate-consumed provenance must survive
# every reachable post-gate failure -- transport exhaustion on either
# object, and an entirely unexpected (non-V9005StageABlocked) exception at
# any post-receipt stage -- never a fabricated 0/PRE_GATE.

def test_f6_production_acquisition_root_exhausted_reports_cumulative_total(tmp_path: Path) -> None:
    def fetcher(url: str) -> m.FetchResult:
        raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)

    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.PLUMBING_FAILURE_RETRIABLE
    assert excinfo.value.network_request_count == m.MAX_ATTEMPTS
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True
    root_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_DISCOVERY_ROOT, m.TOPIX_ROOT_URL,
    )
    assert root_locked is None  # never a complete payload


def test_f6_production_acquisition_root_success_child_exhausted_reports_cumulative_total(
    tmp_path: Path,
) -> None:
    root_payload = _production_root_payload(b"child-object-alpha")

    def fetcher(url: str) -> m.FetchResult:
        if url == m.TOPIX_ROOT_URL:
            return m.FetchResult(root_payload, url, 200)
        raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)

    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.PLUMBING_FAILURE_RETRIABLE
    assert excinfo.value.network_request_count == 1 + m.MAX_ATTEMPTS  # ROOT(1) + CHILD exhausted(3)
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True
    root_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_DISCOVERY_ROOT, m.TOPIX_ROOT_URL,
    )
    assert root_locked is not None


def test_f6_production_acquisition_root_retry_success_child_failure_cumulative_total(
    tmp_path: Path,
) -> None:
    root_payload = _production_root_payload(b"child-object-alpha")
    state = {"root_seen": 0, "child_seen": 0}

    def fetcher(url: str) -> m.FetchResult:
        if url == m.TOPIX_ROOT_URL:
            state["root_seen"] += 1
            if state["root_seen"] < 2:
                raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
            return m.FetchResult(root_payload, url, 200)
        state["child_seen"] += 1
        if state["child_seen"] < 2:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        # Second CHILD attempt fails non-retryably: exactly 2 total CHILD
        # attempts are consumed, not the full 3-attempt exhaustion.
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)

    output_root = tmp_path / "out"
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert state["root_seen"] == 2
    assert state["child_seen"] == 2
    assert excinfo.value.network_request_count == 4
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True


def test_f6_production_acquisition_unexpected_exception_after_root_fetch_before_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_lock = m.lock_first_complete_payload

    def spy_lock(root: object, *, applicable_period: str, **kwargs: object) -> dict[str, object]:
        if applicable_period == m.TOPIX_DISCOVERY_ROOT:
            raise RuntimeError("unexpected ROOT-lock failure, not a V9005StageABlocked")
        return original_lock(root, applicable_period=applicable_period, **kwargs)

    monkeypatch.setattr(m, "lock_first_complete_payload", spy_lock)
    output_root = tmp_path / "out"
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"))
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert excinfo.value.network_request_count == 1  # the ROOT fetch that already happened
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True


def test_f6_production_acquisition_unexpected_exception_in_locator_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("unexpected locator failure, not a V9005StageABlocked")

    monkeypatch.setattr(m, "parse_f6_global_child_locator", _boom)
    output_root = tmp_path / "out"
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"))
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert excinfo.value.network_request_count == 1
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True
    root_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_DISCOVERY_ROOT, m.TOPIX_ROOT_URL,
    )
    assert root_locked is not None


def test_f6_production_acquisition_unexpected_exception_after_child_fetch_before_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_lock = m.lock_first_complete_payload

    def spy_lock(root: object, *, applicable_period: str, **kwargs: object) -> dict[str, object]:
        if applicable_period == m.TOPIX_GLOBAL_2017_2025:
            raise RuntimeError("unexpected CHILD-lock failure, not a V9005StageABlocked")
        return original_lock(root, applicable_period=applicable_period, **kwargs)

    monkeypatch.setattr(m, "lock_first_complete_payload", spy_lock)
    output_root = tmp_path / "out"
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"))
    with pytest.raises(m.V9005StageABlocked) as excinfo:
        m.run_f6_production_root_global_raw_acquisition_network(
            output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
            fetcher=fetcher, sleep=_no_sleep, clock=_clock,
        )
    assert excinfo.value.failure_class == m.IMPLEMENTATION_FAILURE
    assert excinfo.value.network_request_count == 2  # ROOT(1) + CHILD(1)
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True
    child_locked = m.read_locked_payload(
        output_root, m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, m.TOPIX_GLOBAL_2017_2025,
        urllib.parse.urljoin(m.TOPIX_ROOT_URL, "child-object-alpha"),
    )
    assert child_locked is None  # never durably locked


def _valid_f6_production_gate_receipt() -> dict[str, object]:
    return {
        "schema_version": m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_SCHEMA_VERSION,
        "task": m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_TASK_ID,
        "confirmation_contract": m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        "gate_consumed": True,
        "consumption_timestamp_utc": "2026-08-24T00:00:00Z",
    }


def test_read_f6_production_acquisition_gate_consumed_state_tri_state(tmp_path: Path) -> None:
    output_root = tmp_path / "out"
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is False  # nothing there yet

    output_root.mkdir(parents=True)
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is False  # dir exists, no receipt

    receipt_path = output_root / m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_FILENAME
    receipt_path.write_bytes(m.canonical_bytes(_valid_f6_production_gate_receipt()))
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is True

    receipt_path.write_text("not valid json{{{", encoding="utf-8")
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("gate_consumed", False),
        ("gate_consumed", "true"),
        ("gate_consumed", 1),
        ("task", "wrong-task"),
        ("confirmation_contract", "wrong-contract"),
        ("schema_version", "wrong-schema"),
        ("consumption_timestamp_utc", "not-a-canonical-timestamp"),
    ],
)
def test_read_f6_production_acquisition_gate_consumed_state_invalid_receipt_is_unknown(
    tmp_path: Path, field: str, value: object,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    receipt = _valid_f6_production_gate_receipt()
    receipt[field] = value
    receipt_path = output_root / m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_FILENAME
    receipt_path.write_bytes(m.canonical_bytes(receipt))
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is None


def test_read_f6_production_acquisition_gate_consumed_state_filesystem_uncertainty_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"

    def denied_lstat(_path: Path) -> object:
        raise PermissionError("synthetic durable-state probe denial")

    monkeypatch.setattr(Path, "lstat", denied_lstat)
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is None


def test_read_f6_production_acquisition_gate_consumed_state_read_uncertainty_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "out"
    output_root.mkdir()
    receipt_path = output_root / m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_FILENAME
    receipt_path.write_bytes(m.canonical_bytes(_valid_f6_production_gate_receipt()))

    def denied_read_text(_path: Path, *args: object, **kwargs: object) -> str:
        raise PermissionError("synthetic durable-state read denial")

    monkeypatch.setattr(Path, "read_text", denied_read_text)
    assert m.read_f6_production_acquisition_gate_consumed_state(output_root) is None


def test_f6_production_acquisition_never_invokes_run_stage_a_or_populates_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("run_stage_a invoked by F6 production acquisition")

    monkeypatch.setattr(m, "run_stage_a", _blocked)
    output_root = tmp_path / "out"
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"))
    artifact = m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["status"] == "F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_COMPLETE"
    inventory = m.build_source_inventory()
    f6_records = [
        record for record in inventory
        if record["source_family"] == m.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE
    ]
    assert f6_records
    assert all(record["status"] == m.INVENTORY_MISSING for record in f6_records)


def test_f6_production_acquisition_no_real_socket_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blocked(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("real network socket attempted during F6 production acquisition test")

    monkeypatch.setattr(socket, "socket", _blocked)
    output_root = tmp_path / "out"
    fetcher = _production_routing_fetcher(_production_root_payload(b"child-object-alpha"))
    artifact = m.run_f6_production_root_global_raw_acquisition_network(
        output_root=output_root, confirmation=m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION,
        fetcher=fetcher, sleep=_no_sleep, clock=_clock,
    )
    assert artifact["status"] == "F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_COMPLETE"


def test_f6_production_acquisition_implementation_complete_still_false() -> None:
    assert m.ACQUISITION_IMPLEMENTATION_COMPLETE is False


# --- V9_006_STAGE_A_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_IMPLEMENTATION:
# CLI script -----------------------------------------------------------------

F6_PRODUCTION_CLI_MODULE_NAME = "run_v9_006_f6_production_root_global_raw_acquisition"
F6_PRODUCTION_CLI_CONFIRMATION_ENV = "V9_006_F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION"


def _f6_production_cli() -> object:
    scripts_directory = str(ROOT / "scripts")
    if scripts_directory not in sys.path:
        sys.path.insert(0, scripts_directory)
    return importlib.reload(importlib.import_module(F6_PRODUCTION_CLI_MODULE_NAME))


def test_cli_f6_production_never_imports_run_stage_a() -> None:
    text = (ROOT / "scripts" / "run_v9_006_f6_production_root_global_raw_acquisition.py").read_text(
        encoding="utf-8",
    )
    assert "run_stage_a" not in text


def test_cli_f6_production_missing_confirmation_zero_fetch_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.delenv(F6_PRODUCTION_CLI_CONFIRMATION_ENV, raising=False)
    cli = _f6_production_cli()
    calls: list[str] = []
    monkeypatch.setattr(cli, "_production_fetcher", lambda url: calls.append(url) or m.FetchResult(b"x", url, 200))
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root)])
    assert exit_code == 2
    assert calls == []
    assert not output_root.exists()
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["execution_result"] == "BLOCKED"
    assert payload["failure_class"] == "GOVERNANCE_FAILURE"
    assert payload["network_request_count"] == 0
    assert payload["gate_consumed"] is False
    assert payload["authorization_reusable"] is False
    assert payload["second_execution_allowed"] is False


def test_cli_f6_production_wrong_confirmation_zero_fetch_calls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.setenv(F6_PRODUCTION_CLI_CONFIRMATION_ENV, "wrong-token")
    cli = _f6_production_cli()
    calls: list[str] = []
    monkeypatch.setattr(cli, "_production_fetcher", lambda url: calls.append(url) or m.FetchResult(b"x", url, 200))
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root)])
    assert exit_code == 2
    assert calls == []
    assert not output_root.exists()
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["failure_class"] == "GOVERNANCE_FAILURE"
    assert payload["network_request_count"] == 0
    assert payload["gate_consumed"] is False


def test_cli_f6_production_safe_stdout_excludes_raw_urls_and_content(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.setenv(F6_PRODUCTION_CLI_CONFIRMATION_ENV, m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION)
    cli = _f6_production_cli()
    root_payload = _production_root_payload(b"child-object-alpha")

    def fake_fetcher(url: str) -> m.FetchResult:
        if url == m.TOPIX_ROOT_URL:
            return m.FetchResult(root_payload, url, 200)
        return m.FetchResult(b"synthetic-child-bytes", url, 200)

    monkeypatch.setattr(cli, "_production_fetcher", fake_fetcher)
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root)])
    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert set(payload) == {
        "schema_version", "execution_result", "status", "gate_consumed",
        "authorization_reusable", "second_execution_allowed",
        "root_http_status", "root_byte_length", "root_sha256", "root_retrieval_timestamp_utc",
        "root_requested_url_sha256", "root_resolved_url_sha256", "root_requested_resolved_url_equal",
        "locator_status", "candidate_anchor_count",
        "child_http_status", "child_byte_length", "child_sha256", "child_retrieval_timestamp_utc",
        "child_requested_url_sha256", "child_resolved_url_sha256", "child_requested_resolved_url_equal",
        "root_network_request_count", "child_network_request_count", "network_request_count",
    }
    assert "receipt_path" not in payload
    assert "output_root" not in payload
    assert payload["execution_result"] == "COMPLETE"
    assert payload["status"] == "F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_COMPLETE"
    assert payload["gate_consumed"] is True
    assert payload["authorization_reusable"] is False
    assert payload["second_execution_allowed"] is False
    assert payload["network_request_count"] == 2
    assert m.TOPIX_ROOT_URL not in out
    assert "child-object-alpha" not in out
    assert "synthetic-child-bytes" not in out
    assert "Historical Index Value" not in out
    assert m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_FILENAME not in out
    assert str(output_root) not in out


def test_cli_f6_production_failure_stdout_excludes_raw_urls_output_root_and_receipt_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.setenv(F6_PRODUCTION_CLI_CONFIRMATION_ENV, m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION)
    cli = _f6_production_cli()
    root_payload = _production_no_candidate_payload()

    def fake_fetcher(url: str) -> m.FetchResult:
        return m.FetchResult(root_payload, url, 200)

    monkeypatch.setattr(cli, "_production_fetcher", fake_fetcher)
    output_root = tmp_path / "cli-out"
    exit_code = cli.main(["--output-root", str(output_root)])
    assert exit_code == 2
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)
    assert payload["execution_result"] == "BLOCKED"
    assert payload["failure_class"] == m.CHATGPT_DECISION_REQUIRED
    assert payload["network_request_count"] == 1
    assert payload["gate_consumed"] is True
    assert payload["authorization_reusable"] is False
    assert payload["second_execution_allowed"] is False
    assert "receipt_path" not in payload
    assert "output_root" not in payload
    assert m.TOPIX_ROOT_URL not in out
    assert m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_GATE_RECEIPT_FILENAME not in out
    assert str(output_root) not in out


def test_cli_f6_production_durable_state_reader_failure_is_safe_unknown_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path,
) -> None:
    monkeypatch.setenv(F6_PRODUCTION_CLI_CONFIRMATION_ENV, m.F6_PRODUCTION_ROOT_GLOBAL_RAW_ACQUISITION_CONFIRMATION)
    cli = _f6_production_cli()
    output_root = tmp_path / "sensitive-output-root"

    def blocked_executor(**_kwargs: object) -> None:
        raise m.V9005StageABlocked(m.IMPLEMENTATION_FAILURE, network_request_count=0)

    def denied_reader(_root: object) -> None:
        raise PermissionError(f"durable-state inspection denied for {output_root}")

    def no_fetch(_url: str) -> m.FetchResult:
        raise AssertionError("network fetch attempted in durable-state reporting test")

    monkeypatch.setattr(cli, "run_f6_production_root_global_raw_acquisition_network", blocked_executor)
    monkeypatch.setattr(cli, "read_f6_production_acquisition_gate_consumed_state", denied_reader)
    monkeypatch.setattr(cli, "_production_fetcher", no_fetch)
    exit_code = cli.main(["--output-root", str(output_root)])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.err == ""
    payload = json.loads(captured.out.strip())
    assert payload["execution_result"] == "BLOCKED"
    assert payload["failure_class"] == m.IMPLEMENTATION_FAILURE
    assert payload["network_request_count"] == 0
    assert payload["gate_consumed"] == "unknown"
    assert str(output_root) not in captured.out
    assert "durable-state inspection denied" not in captured.out
    assert "Traceback" not in captured.out
