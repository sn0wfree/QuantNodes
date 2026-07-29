"""CA-GCP validators."""
from .coverage import compute_coverage_metrics, width_bps
from .early_warning import detect_warnings, evaluate_against_events, width_timeseries
from .neighbor_quality import (
    NeighborQuality,
    compute_neighbor_quality,
    quality_dataframe,
    recommend_borrow_strategy,
)
from .theoretical_bound import (
    TheoreticalBound,
    compare_bound_to_empirical,
    theoretical_coverage_bound,
    total_variation_distance_ecdf,
)
from .width import width_stability, width_timeseries, width_volatility_correlation

__all__ = [
    "compute_coverage_metrics",
    "width_bps",
    "width_timeseries",
    "width_stability",
    "width_volatility_correlation",
    "detect_warnings",
    "evaluate_against_events",
    "NeighborQuality",
    "compute_neighbor_quality",
    "recommend_borrow_strategy",
    "quality_dataframe",
    "TheoreticalBound",
    "theoretical_coverage_bound",
    "total_variation_distance_ecdf",
    "compare_bound_to_empirical",
]