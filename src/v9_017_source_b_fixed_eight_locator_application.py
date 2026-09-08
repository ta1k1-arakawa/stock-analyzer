"""Synthetic-only composition for the frozen V9_017 fixed-eight locator set.

This module binds caller-supplied year-page bytes in memory and composes the
reviewed NORMAL locator with the unchanged V9_014 PRE resolver.  It has no
acquisition, filesystem, PDF, or outcome-calculation surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.v9_014_jpx_monthly_auction_activity_source_b_archive_parser import (
    extract_april_pre_candidates,
)
from src.v9_014_jpx_monthly_auction_activity_source_b_locator import (
    LOCATOR_OK,
    resolve_source_b_april_pre_object,
)
from src.v9_014_jpx_monthly_auction_activity_source_b_pdf_calibration_probe import (
    PRE_APRIL_1_REFERENCE_OBJECT,
    REQUIRED_CALIBRATION_IDENTITIES,
    NORMAL_MONTHLY_REPORT2_OBJECT,
)
from src.v9_017_source_b_frozen_normal_locator import (
    _sole_candidate_href,
    locate_frozen_normal_month,
)


SCHEMA_VERSION = "V9_017_FIXED_EIGHT_LOCATOR_APPLICATION_V1"
INPUT_FAILURE = "INPUT_FAILURE"
NORMAL_LOCATOR_FAILURE = "NORMAL_LOCATOR_FAILURE"
PRE_RESOLVER_FAILURE = "PRE_RESOLVER_FAILURE"


@dataclass(frozen=True)
class FixedEightLocatorApplicationResult:
    """Typed result with private selected candidates and safe public output."""

    status: str
    resolved_identity_count: int
    identity_results: tuple[dict[str, str], ...]
    failure_identity: dict[str, str] | None
    failure_reason: str | None
    _selected_urls: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "resolved_identity_count": self.resolved_identity_count,
            "identity_results": [dict(item) for item in self.identity_results],
            "failure_identity": (
                dict(self.failure_identity) if self.failure_identity is not None else None
            ),
            "failure_reason": self.failure_reason,
        }

    def __repr__(self) -> str:
        return repr(self.to_dict())


def _identity_dict(identity: Any, status: str, reason: str | None = None) -> dict[str, str]:
    item = {
        "logical_month": identity.logical_month,
        "object_part": identity.object_part,
        "status": status,
    }
    if reason is not None:
        item["reason"] = reason
    return item


def _failure_result(
    identities: tuple[Any, ...],
    completed: list[dict[str, str]],
    selected_urls: list[str],
    index: int,
    reason: str,
) -> FixedEightLocatorApplicationResult:
    failed = _identity_dict(identities[index], "FAIL", reason)
    remaining = [
        _identity_dict(identity, "NOT_REACHED")
        for identity in identities[index + 1 :]
    ]
    return FixedEightLocatorApplicationResult(
        status="FAIL",
        resolved_identity_count=len(completed),
        identity_results=tuple(completed + [failed] + remaining),
        failure_identity={
            "logical_month": identities[index].logical_month,
            "object_part": identities[index].object_part,
        },
        failure_reason=reason,
        _selected_urls=tuple(selected_urls),
    )


def apply_fixed_eight_locators(
    year_page_bytes_by_year: Mapping[int, bytes],
    selected_year_page_urls: Mapping[int, str] | None = None,
) -> FixedEightLocatorApplicationResult:
    """Resolve the inherited eight identities from caller-supplied bytes.

    The identities are processed in the inherited order.  The first failure
    is terminal and no alternate resolver or grammar is attempted.
    """

    identities = REQUIRED_CALIBRATION_IDENTITIES
    if not isinstance(year_page_bytes_by_year, Mapping):
        return _failure_result(identities, [], [], 0, INPUT_FAILURE)
    if selected_year_page_urls is not None and not isinstance(
        selected_year_page_urls, Mapping
    ):
        return _failure_result(identities, [], [], 0, INPUT_FAILURE)

    completed: list[dict[str, str]] = []
    selected_urls: list[str] = []

    for index, identity in enumerate(identities):
        try:
            year = int(identity.logical_month[:4])
            page_bytes = year_page_bytes_by_year[year]
        except (KeyError, TypeError, ValueError, IndexError):
            return _failure_result(identities, completed, selected_urls, index, INPUT_FAILURE)
        if not isinstance(page_bytes, bytes):
            return _failure_result(identities, completed, selected_urls, index, INPUT_FAILURE)

        if identity.object_part == NORMAL_MONTHLY_REPORT2_OBJECT:
            month = int(identity.logical_month[5:7])
            try:
                outcome = locate_frozen_normal_month(page_bytes, month)
            except Exception:
                return _failure_result(
                    identities, completed, selected_urls, index, NORMAL_LOCATOR_FAILURE
                )
            if outcome.status != "PASS":
                return _failure_result(
                    identities,
                    completed,
                    selected_urls,
                    index,
                    outcome.reason,
                )
            try:
                selected_urls.append(_sole_candidate_href(outcome))
            except Exception:
                return _failure_result(
                    identities, completed, selected_urls, index, NORMAL_LOCATOR_FAILURE
                )
        elif identity.object_part == PRE_APRIL_1_REFERENCE_OBJECT:
            if selected_year_page_urls is None:
                return _failure_result(
                    identities, completed, selected_urls, index, INPUT_FAILURE
                )
            selected_year_page_url = selected_year_page_urls.get(2022)
            if not isinstance(selected_year_page_url, str):
                return _failure_result(
                    identities, completed, selected_urls, index, INPUT_FAILURE
                )
            try:
                candidates = extract_april_pre_candidates(
                    page_bytes,
                    selected_year_page_url,
                    selected_year=2022,
                )
                outcome = resolve_source_b_april_pre_object(
                    candidates,
                    selected_year_page_url=selected_year_page_url,
                )
            except Exception:
                return _failure_result(
                    identities, completed, selected_urls, index, PRE_RESOLVER_FAILURE
                )
            if outcome.status != LOCATOR_OK or not isinstance(outcome.url, str):
                return _failure_result(
                    identities, completed, selected_urls, index, PRE_RESOLVER_FAILURE
                )
            selected_urls.append(outcome.url)
        else:
            return _failure_result(
                identities, completed, selected_urls, index, INPUT_FAILURE
            )

        completed.append(_identity_dict(identity, "PASS"))

    return FixedEightLocatorApplicationResult(
        status="PASS",
        resolved_identity_count=len(completed),
        identity_results=tuple(completed),
        failure_identity=None,
        failure_reason=None,
        _selected_urls=tuple(selected_urls),
    )


def _selected_urls(result: FixedEightLocatorApplicationResult) -> tuple[str, ...]:
    """Return internal candidates only for later in-process composition."""

    if result.status != "PASS":
        raise ValueError("selected candidates unavailable")
    return result._selected_urls


__all__ = [
    "FixedEightLocatorApplicationResult",
    "SCHEMA_VERSION",
    "apply_fixed_eight_locators",
]
