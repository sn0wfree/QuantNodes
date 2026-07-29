"""DEPRECATED shim — CA-GCP has moved to common/ca_gcp.py.

This file remains as a backward-compatibility shim.
New code should import from:
    QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp
"""
from __future__ import annotations

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (  # noqa: F401
    _LOSS_REGISTRY,
    CAGCPConfig,
    CAGCPipeline,
    NeighborQuality,
    PrecomputedWeightedQuantile,
    RiskFilterRules,
    SectorCAGCPResult,
    TheoreticalBound,
    apply_modulator,
    apply_scale_to_weights,
    build_knn_graph,
    build_sector_groups,
    build_v10_2_pipeline,
    ca_gcp_risk_filter,
    compare_bound_to_empirical,
    compute_coverage_metrics,
    compute_neighbor_quality,
    compute_systemic_stress,
    detect_warnings,
    estimate_volatility,
    evaluate_alert,
    experimental_rules,
    extract_risk_signals,
    fit_sector_ca_gcp,
    fit_sector_hybrid_ca_gcp,
    load_sector_map,
    predict_sector_ca_gcp,
    quality_dataframe,
    resolve_loss_fn,
    theoretical_coverage_bound,
    total_variation_distance_ecdf,
    width_bps,
    width_stability,
    width_timeseries,
    width_volatility_correlation,
    weighted_quantile,
)
