"""Offline synthetic coverage for the Stage-A Checkpoint A integration seam."""
from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from src import v9_005_stage_a_jpx_probe as probe


def _slot(family: str, period: str) -> str:
    return hashlib.sha256(f"{family}:{period}".encode()).hexdigest()


def _inputs(*, missing_years: tuple[int, ...] = ()):
    locks = {}
    f2f4 = {}
    f3 = {}
    f7 = {}
    for month in probe.inventory_months():
        year = month[:4]
        for family in (probe.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, probe.SOURCE_FAMILY_EX_RIGHTS_SPLIT_RATIO_ARCHIVE, probe.SOURCE_FAMILY_JPX_CALENDAR):
            slot = _slot(family, month)
            locks[slot] = {"source_family": family, "applicable_period": month}
            (f7 if family == probe.SOURCE_FAMILY_JPX_CALENDAR else f2f4)[(family, month)] = (slot,)
        family = probe.SOURCE_FAMILY_DELISTED_COMPANY_ARCHIVE
        slot = _slot(family, year)
        locks[slot] = {"source_family": family, "applicable_period": year}
        f3[(family, month)] = (slot,)
    f6_slot = _slot(probe.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, probe.TOPIX_GLOBAL_2017_2025)
    locks[f6_slot] = {"source_family": probe.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE, "applicable_period": probe.TOPIX_GLOBAL_2017_2025}
    missing = tuple(missing_years)
    coverage = probe.F6RequiredPeriodCoverage(f6_slot, tuple(year for year in range(2017, 2026) if year not in missing), missing, True)
    return (probe.F2F4RequiredSlotAcquisition(f2f4, {"2026-01": (_slot("bridge", "x"),)}, 0), probe.F3RequiredSlotAcquisition(f3, 0), coverage, probe.F7RequiredSlotAcquisition(f7, {"2016-12": (_slot("extra", "x"),)}, 0), locks)


def _matrix(monkeypatch, **kwargs):
    f2f4, f3, f6, f7, locks = _inputs(**kwargs)
    monkeypatch.setattr(probe, "_verified_raw_lock_index", lambda _root: locks)
    return probe.build_checkpoint_a_monthly_coverage_matrix(f2_f4=f2f4, f3=f3, f6=f6, f7=f7, output_root="synthetic")


def test_exact_deterministic_matrix_and_reviewed_base_consumption(monkeypatch):
    first = _matrix(monkeypatch)
    second = _matrix(monkeypatch)
    assert first == second and len(first) == 648
    assert [(r["source_family"], r["month"]) for r in first] == [(family, month) for month in probe.inventory_months() for family in probe.MONTHLY_COVERAGE_FAMILIES]
    assert {r["source_family"] for r in first} == set(probe.MONTHLY_COVERAGE_FAMILIES)
    assert probe.SOURCE_FAMILY_LISTED_ISSUES_MONTH_END not in {r["source_family"] for r in first}
    assert sum(r["status"] == probe.INVENTORY_AVAILABLE and r["source_family"] == probe.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE for r in first) == 108
    assert sum(r["status"] == probe.INVENTORY_MISSING and r["source_family"] == probe.SOURCE_FAMILY_MONTHLY_AGGREGATE_LISTED_ISSUE_COUNTS for r in first) == 108
    assert all(r["status"] != probe.INVENTORY_NOT_APPLICABLE for r in first)
    assert probe.required_inventory_missing_count(first) == 0


def test_f6_missing_year_and_invalid_partitions_fail_closed(monkeypatch):
    matrix = _matrix(monkeypatch, missing_years=(2020,))
    assert sum(r["source_family"] == probe.SOURCE_FAMILY_TOPIX_HISTORICAL_INDEX_VALUE and r["status"] == probe.INVENTORY_MISSING for r in matrix) == 12
    f2f4, f3, f6, f7, locks = _inputs()
    monkeypatch.setattr(probe, "_verified_raw_lock_index", lambda _root: locks)
    bad = probe.F6RequiredPeriodCoverage(f6.global_source_object_slot_id, (2017,), (), True)
    with pytest.raises(probe.V9005StageABlocked) as exc:
        probe.build_checkpoint_a_monthly_coverage_matrix(f2_f4=f2f4, f3=f3, f6=bad, f7=f7, output_root="synthetic")
    assert exc.value.reason == probe.IMPLEMENTATION_FAILURE


@pytest.mark.parametrize("covered,missing,accepted", [
    ((2017, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025), (), True),
    ((2018, 2017, 2019, 2020, 2021, 2022, 2023, 2024, 2025), (), True),
    ([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], (), True),
    ((True, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025), (), True),
    (("2017", 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025), (), True),
    (([2017], 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025), (), True),
    ((2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025), (2020,), True),
    ((2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025), (), True),
    ((2016, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025), (), True),
    ((2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025), (), False),
    ((2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025), (), 1),
])
def test_f6_malformed_partitions_are_total_and_fail_closed(monkeypatch, covered, missing, accepted):
    f2f4, f3, f6, f7, locks = _inputs()
    monkeypatch.setattr(probe, "_verified_raw_lock_index", lambda _root: locks)
    bad = probe.F6RequiredPeriodCoverage(f6.global_source_object_slot_id, covered, missing, accepted)
    with pytest.raises(probe.V9005StageABlocked) as exc:
        probe.build_checkpoint_a_monthly_coverage_matrix(f2_f4=f2f4, f3=f3, f6=bad, f7=f7, output_root="synthetic")
    assert exc.value.reason == probe.IMPLEMENTATION_FAILURE


@pytest.mark.parametrize("mutator", [
    lambda locks, f2f4, f3, f6, f7: locks.__setitem__(f2f4.base_coverage_references[(probe.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2017-01")][0], {"source_family": "wrong", "applicable_period": "2017-01"}),
    lambda locks, f2f4, f3, f6, f7: locks.__setitem__(f7.base_coverage_references[(probe.SOURCE_FAMILY_JPX_CALENDAR, "2017-01")][0], {"source_family": probe.SOURCE_FAMILY_JPX_CALENDAR, "applicable_period": "ROOT"}),
    lambda locks, f2f4, f3, f6, f7: f2f4.base_coverage_references.pop((probe.SOURCE_FAMILY_MONTHLY_STATISTICS_CHANGES_REPORT, "2017-01")),
])
def test_invalid_coverage_identity_or_matrix_domain_fails_closed(monkeypatch, mutator):
    f2f4, f3, f6, f7, locks = _inputs()
    mutator(locks, f2f4, f3, f6, f7)
    monkeypatch.setattr(probe, "_verified_raw_lock_index", lambda _root: locks)
    with pytest.raises(probe.V9005StageABlocked):
        probe.build_checkpoint_a_monthly_coverage_matrix(f2_f4=f2f4, f3=f3, f6=f6, f7=f7, output_root="synthetic")


def test_required_missing_count_is_generic_and_fail_closed(monkeypatch):
    matrix = _matrix(monkeypatch)
    changed = [dict(row) for row in matrix]
    changed[0]["status"], changed[0]["source_object_slot_ids"] = probe.INVENTORY_MISSING, []
    changed[1]["status"], changed[1]["source_object_slot_ids"] = probe.INVENTORY_MISSING, []
    assert probe.required_inventory_missing_count(changed) == 2
    for bad in ({"source_family": "unknown", "status": probe.INVENTORY_MISSING}, {"source_family": probe.SOURCE_FAMILY_JPX_CALENDAR, "status": "bad"}):
        with pytest.raises(probe.V9005StageABlocked):
            probe.required_inventory_missing_count([bad])
    original = probe.LOCATOR_STRATEGIES.pop(probe.SOURCE_FAMILY_JPX_CALENDAR)
    try:
        with pytest.raises(probe.V9005StageABlocked):
            probe.required_inventory_missing_count(matrix)
    finally:
        probe.LOCATOR_STRATEGIES[probe.SOURCE_FAMILY_JPX_CALENDAR] = original
    original = probe.LOCATOR_STRATEGIES[probe.SOURCE_FAMILY_JPX_CALENDAR]
    probe.LOCATOR_STRATEGIES[probe.SOURCE_FAMILY_JPX_CALENDAR] = replace(original, auxiliary="not-a-bool")
    try:
        with pytest.raises(probe.V9005StageABlocked):
            probe.required_inventory_missing_count(matrix)
    finally:
        probe.LOCATOR_STRATEGIES[probe.SOURCE_FAMILY_JPX_CALENDAR] = original


def test_matrix_validator_rejects_duplicate_extra_and_missing_keys(monkeypatch):
    matrix = _matrix(monkeypatch)
    for bad in (matrix[:-1], matrix + [dict(matrix[0])], matrix[1:] + [dict(matrix[0])]):
        with pytest.raises(probe.V9005StageABlocked):
            probe._validate_monthly_coverage_matrix(bad)


def test_evidence_boundary_requires_complete_closed_matrix(monkeypatch):
    matrix = _matrix(monkeypatch)
    assert probe.required_inventory_missing_count(matrix) == 0
    one_non_auxiliary_missing = [dict(record) for record in matrix]
    one_non_auxiliary_missing[0]["status"] = probe.INVENTORY_MISSING
    one_non_auxiliary_missing[0]["source_object_slot_ids"] = []
    assert probe.required_inventory_missing_count(one_non_auxiliary_missing) == 1
    malformed_cases = [
        matrix[:-1],
        matrix + [dict(matrix[0])],
        matrix[1:] + [dict(matrix[0])],
        [{"source_family": probe.SOURCE_FAMILY_JPX_CALENDAR, "month": "2017-01", "status": probe.INVENTORY_MISSING}],
    ]
    bad_slot = [dict(record) for record in matrix]
    bad_slot[0]["source_object_slot_ids"] = ["not-a-slot"]
    malformed_cases.append(bad_slot)
    duplicate_slots = [dict(record) for record in matrix]
    duplicate_slots[0]["source_object_slot_ids"] = ["0" * 64, "0" * 64]
    malformed_cases.append(duplicate_slots)
    unsorted_slots = [dict(record) for record in matrix]
    unsorted_slots[0]["source_object_slot_ids"] = ["f" * 64, "0" * 64]
    malformed_cases.append(unsorted_slots)
    for bad in malformed_cases:
        with pytest.raises(probe.V9005StageABlocked) as exc:
            probe.required_inventory_missing_count(bad)
        assert exc.value.reason == probe.IMPLEMENTATION_FAILURE
    with pytest.raises(probe.V9005StageABlocked) as exc:
        probe.compute_stage_a_evidence(inventory=matrix[:-1], terminal_snapshot_locked=True, trading_calendar_derived=True, semantic_result={}, terminal_identities={}, events=(), comparable_month_end_mismatch_count=0, raw_provenance_pass=True)
    assert exc.value.reason == probe.IMPLEMENTATION_FAILURE


def test_readiness_order_is_pre_side_effect_and_confirmation_cannot_bypass(tmp_path):
    assert probe.ACQUISITION_IMPLEMENTATION_COMPLETE is True
    assert probe.OVERALL_STAGE_A_IMPLEMENTATION_READY is False
    probe.verify_acquisition_implementation_ready()
    with pytest.raises(probe.V9005StageABlocked) as exc:
        probe.verify_overall_stage_a_implementation_ready()
    assert exc.value.reason == probe.STAGE_A_OVERALL_IMPLEMENTATION_INCOMPLETE
    calls = []
    with pytest.raises(probe.V9005StageABlocked) as exc:
        probe.run_stage_a(output_root=tmp_path / "must-not-exist", repo_root=tmp_path, confirmation=probe.CONFIRMATION, fetcher=lambda url: calls.append(url), sleep=lambda _: calls.append("sleep"), clock=lambda: calls.append("clock"), git=lambda args: calls.append("git"))
    assert exc.value.failure_class == probe.CHATGPT_DECISION_REQUIRED
    assert not (tmp_path / "must-not-exist").exists() and calls == []
