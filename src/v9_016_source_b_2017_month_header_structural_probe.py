"""Synthetic-only V9_016 C1 month-header structural probe.

This module deliberately reports bounded structural counts only.  It does not
select a category, resolve links, access files, or perform network I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.v9_005_stage_a_jpx_probe import _parse_monthly_statistics_html


SCHEMA_VERSION = "V9_016_C1_2017_MONTH_HEADER_STRUCTURE_V1"
SOURCE_B_REPORT = "Stock Trading Volume & Value"

CATEGORY_TOKENS = (
    ("LOGICAL_YYYY_MM", "2017-01"),
    ("EN_MONTH_ABBR_DOT", "Jan."),
    ("EN_MONTH_ABBR", "Jan"),
    ("EN_MONTH_FULL", "January"),
    ("NUMERIC_MONTH", "1"),
    ("NUMERIC_MONTH_ZERO_PADDED", "01"),
)
CATEGORY_NAMES = tuple(name for name, _ in CATEGORY_TOKENS)


@dataclass(frozen=True)
class CategoryStructure:
    th_count: int
    same_table_th_count: int
    intersection_pair_count: int
    in_bounds_intersection_count: int
    out_of_bounds_intersection_count: int
    intersection_href_count: int
    intersection_href_multiplicity: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "th_count": self.th_count,
            "same_table_th_count": self.same_table_th_count,
            "intersection_pair_count": self.intersection_pair_count,
            "in_bounds_intersection_count": self.in_bounds_intersection_count,
            "out_of_bounds_intersection_count": self.out_of_bounds_intersection_count,
            "intersection_href_count": self.intersection_href_count,
            "intersection_href_multiplicity": self.intersection_href_multiplicity,
        }


@dataclass(frozen=True)
class C1StructureResult:
    schema_version: str
    status: str
    failure_class: Optional[str]
    parser_success: bool
    table_count: int
    report_cell_count: int
    report_row_count: int
    categories: dict[str, CategoryStructure]
    legacy_candidate_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "failure_class": self.failure_class,
            "parser_success": self.parser_success,
            "table_count": self.table_count,
            "report_cell_count": self.report_cell_count,
            "report_row_count": self.report_row_count,
            "categories": {
                name: self.categories[name].to_dict() for name in CATEGORY_NAMES
            },
            "legacy_candidate_count": self.legacy_candidate_count,
        }

    def __repr__(self) -> str:
        return repr(self.to_dict())


def _empty_categories() -> dict[str, CategoryStructure]:
    return {
        name: CategoryStructure(
            th_count=0,
            same_table_th_count=0,
            intersection_pair_count=0,
            in_bounds_intersection_count=0,
            out_of_bounds_intersection_count=0,
            intersection_href_count=0,
            intersection_href_multiplicity="ZERO",
        )
        for name in CATEGORY_NAMES
    }


def _failure_result() -> C1StructureResult:
    return C1StructureResult(
        schema_version=SCHEMA_VERSION,
        status="FAIL",
        failure_class="IMPLEMENTATION_FAILURE",
        parser_success=False,
        table_count=0,
        report_cell_count=0,
        report_row_count=0,
        categories=_empty_categories(),
        legacy_candidate_count=0,
    )


def _multiplicity(count: int) -> str:
    if count == 0:
        return "ZERO"
    if count == 1:
        return "ONE"
    return "MANY"


def probe_2017_month_header_structure(page_bytes: bytes) -> C1StructureResult:
    """Return the frozen, bounded C1 structural counts for synthetic bytes."""

    if not isinstance(page_bytes, bytes):
        return _failure_result()

    try:
        parser = _parse_monthly_statistics_html(page_bytes)

        report_rows_by_table: list[tuple[object, list[list[object]]]] = []
        report_cell_count = 0
        report_row_count = 0
        for table in parser.tables:
            report_rows: list[list[object]] = []
            for row in table.rows:
                matching_report_cells = [
                    cell for cell in row if cell.text == SOURCE_B_REPORT
                ]
                report_cell_count += len(matching_report_cells)
                if matching_report_cells:
                    report_row_count += 1
                    report_rows.append(row)
            report_rows_by_table.append((table, report_rows))

        categories: dict[str, CategoryStructure] = {}
        for category_name, token in CATEGORY_TOKENS:
            th_count = 0
            same_table_th_count = 0
            intersection_pair_count = 0
            in_bounds_intersection_count = 0
            out_of_bounds_intersection_count = 0
            intersection_href_count = 0

            for table, report_rows in report_rows_by_table:
                matching_th_columns = [
                    column_index
                    for row in table.rows
                    for column_index, cell in enumerate(row)
                    if cell.tag == "th" and cell.text == token
                ]
                th_count += len(matching_th_columns)
                if report_rows:
                    same_table_th_count += len(matching_th_columns)

                for report_row in report_rows:
                    for column_index in matching_th_columns:
                        intersection_pair_count += 1
                        if column_index >= len(report_row):
                            out_of_bounds_intersection_count += 1
                            continue
                        in_bounds_intersection_count += 1
                        intersection_href_count += len(report_row[column_index].hrefs)

            categories[category_name] = CategoryStructure(
                th_count=th_count,
                same_table_th_count=same_table_th_count,
                intersection_pair_count=intersection_pair_count,
                in_bounds_intersection_count=in_bounds_intersection_count,
                out_of_bounds_intersection_count=out_of_bounds_intersection_count,
                intersection_href_count=intersection_href_count,
                intersection_href_multiplicity=_multiplicity(intersection_href_count),
            )

        return C1StructureResult(
            schema_version=SCHEMA_VERSION,
            status="PASS",
            failure_class=None,
            parser_success=True,
            table_count=len(parser.tables),
            report_cell_count=report_cell_count,
            report_row_count=report_row_count,
            categories=categories,
            legacy_candidate_count=categories["LOGICAL_YYYY_MM"].intersection_href_count,
        )
    except Exception:
        return _failure_result()


__all__ = [
    "CATEGORY_NAMES",
    "CATEGORY_TOKENS",
    "C1StructureResult",
    "SCHEMA_VERSION",
    "SOURCE_B_REPORT",
    "CategoryStructure",
    "probe_2017_month_header_structure",
]
