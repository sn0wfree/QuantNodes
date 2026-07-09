# coding=utf-8
"""v2 = 当前 enhanced 策略 (Stage 12A).

特性:
  - momentum_type: "price" | "slope_r2" | "hybrid" (默认 "hybrid")
  - VolTargeting: 波动率目标 (TV tv=0.15)
  - CostModel: 交易成本 (5bp+10bp)
  - CovEstimator: 协方差估计方法选择
  - Calmar 1.60, DD -3.93%

v1 (原始CICC复现) 在 ../v1/.
"""
from __future__ import annotations

from ..common.covariance import (
    CovMethod,
    condition_number,
    diagonal_covariance,
    estimate_covariance,
    ewma_covariance,
    is_positive_definite,
    ledoit_wolf_shrinkage,
    sample_covariance,
)
from .fi_plus_v2 import (
    FixedIncomePlus,
    FixedIncomePlusConfig,
    FixedIncomePlusResult,
)
from ..common.risk_parity import (
    risk_contribution,
    risk_parity_objective,
    solve_max_diversification,
    solve_risk_parity,
)
from .momentum_v2 import (
    below_ma_v2,
    compute_momentum_score_v2,
    distance_to_52w_high_v2,
    fused_signal_v2,
    hybrid_momentum_score_v2,
    pairwise_corr_v2,
    rank_by_momentum_v2,
    rank_pctl_v2,
    realized_vol_v2,
    slope_r2_score_v2,
)
from .portfolio_v2 import (
    ConcentrationCaps,
    CostModel,
    DiversificationCaps,
    PortfolioState,
    RotationConfig,
    TrendFilter,
    VolTargeting,
    apply_concentration_caps_v2 as apply_concentration_caps,
    apply_stops_v2,
    apply_trend_filter_v2,
    apply_vol_targeting_v2,
    calculate_turnover_cost_v2,
    check_trend_filter_v2,
    equal_weights_v2,
    inverse_vol_weights_v2,
    select_and_weight_v2,
    vol_targeting_scale_v2,
)
from .strategy_versions_v2 import (
    LATEST_v2,
    VERSIONS_v2,
    get_version_v2,
    v0_0_baseline_v2,
    v0_1_vt_only_v2,
    v0_2_tf_only_v2,
    v0_3_vt_cost_v2,
    v0_4_hybrid_v2,
    v1_0_v2,
)


__all__ = [
    # Momentum
    "rank_by_momentum_v2",
    "rank_pctl_v2",
    "compute_momentum_score_v2",
    "distance_to_52w_high_v2",
    "fused_signal_v2",
    "hybrid_momentum_score_v2",
    "realized_vol_v2",
    "below_ma_v2",
    "pairwise_corr_v2",
    "slope_r2_score_v2",
    # Portfolio
    "DiversificationCaps",
    "TrendFilter",
    "VolTargeting",
    "ConcentrationCaps",
    "CostModel",
    "RotationConfig",
    "PortfolioState",
    "select_and_weight_v2",
    "apply_stops_v2",
    "apply_trend_filter_v2",
    "apply_vol_targeting_v2",
    "apply_concentration_caps",
    "check_trend_filter_v2",
    "vol_targeting_scale_v2",
    "calculate_turnover_cost_v2",
    "inverse_vol_weights_v2",
    "equal_weights_v2",
    # Covariance
    "CovMethod",
    "estimate_covariance",
    "ledoit_wolf_shrinkage",
    "sample_covariance",
    "ewma_covariance",
    "diagonal_covariance",
    "is_positive_definite",
    "condition_number",
    # Risk Parity
    "solve_risk_parity",
    "solve_max_diversification",
    "risk_contribution",
    "risk_parity_objective",
    # Fixed Income
    "FixedIncomePlus",
    "FixedIncomePlusConfig",
    "FixedIncomePlusResult",
    # Versions
    "v0_0_baseline_v2",
    "v0_1_vt_only_v2",
    "v0_2_tf_only_v2",
    "v0_3_vt_cost_v2",
    "v0_4_hybrid_v2",
    "v1_0_v2",
    "VERSIONS_v2",
    "LATEST_v2",
    "get_version_v2",
]