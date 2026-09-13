"""Small, explicit success measures for fixture-only contract validation."""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import FullDurationCoverage, MethodState, MethodStatus


@dataclass(frozen=True)
class SuccessMeasure:
    """A transparent readiness score, not a model-quality metric."""

    coverage_ratio: float
    successful_method_ratio: float
    combined_ratio: float


def calculate_success_measure(
    coverage: FullDurationCoverage, method_statuses: tuple[MethodStatus, ...]
) -> SuccessMeasure:
    """Combine source-time coverage and completed methods with equal weight."""
    if not method_statuses:
        raise ValueError("at least one method status is required")

    successful = sum(status.state == MethodState.SUCCEEDED for status in method_statuses)
    successful_method_ratio = successful / len(method_statuses)
    coverage_ratio = coverage.ratio
    return SuccessMeasure(
        coverage_ratio=coverage_ratio,
        successful_method_ratio=successful_method_ratio,
        combined_ratio=(coverage_ratio + successful_method_ratio) / 2,
    )
