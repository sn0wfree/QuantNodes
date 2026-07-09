# coding=utf-8
"""Momentum ETF Rotation 策略模块.

CICC 2026-07-03 《固收+:"可靠"的动量 ETF 轮动及 Agent 检验实践》复现.

四步组合管理:
    1. 去重 + 剔高相关 (同指数只留流动性最好, 相关 > corr_threshold 跳过)
    2. 强制分散 (A 股宽基/行业 ≤ a_share, HK ≤ hk, 必含商品/海外)
    3. 逆波动加权 (权重 ∝ 1/σ)
    4. 止损 + 补位 (跌破 ma_window 且 排名跌出后 rank_cutoff 分位)
"""
from __future__ import annotations

from .universe import ETFPool, ETFCategorizer, DEFAULT_POOL
from .momentum import (
    rank_by_momentum, rank_pctl, distance_to_52w_high,
    slope_r2_score, hybrid_momentum_score, compute_momentum_score,
)
from .portfolio import (
    DiversificationCaps,
    RotationConfig,
    PortfolioState,
    TrendFilter,
    VolTargeting,
    ConcentrationCaps,
    CostModel,
    select_and_weight,
    apply_stops,
    apply_trend_filter,
    apply_vol_targeting,
    apply_concentration_caps,
    vol_targeting_scale,
    check_trend_filter,
    _apply_concentration_caps,
    calculate_turnover_cost,
    inverse_vol_weights,
    equal_weights,
)
from .data import load_etf_nav_panel, load_bond_etf_nav
from .fi_plus import (
    FixedIncomePlus,
    FixedIncomePlusConfig,
    FixedIncomePlusResult,
    performance_metrics,
)
from .backtest import (
    BacktestConfig,
    RotationBacktestResult,
    CICC_BASELINES,
    compare_to_cicc,
    run_equal_weight_baseline,
    run_rotation_backtest,
)
from .validation import (
    ValidationConfig,
    ValidationResult,
    ValidationReport,
    ablation,
    run_full_validation,
    validate_parameter_perturbation,
    validate_rebalance_offsets,
    validate_starting_points,
)
from .extended_metrics import extended_metrics, format_metrics_table
from .contribution import (
    reconstruct_daily_weights,
    etf_contribution,
    category_contribution,
    risk_contribution,
    marginal_contribution,
    period_contribution,
    DEFAULT_PERIODS,
)
from .brinson import brinson_attribution, CATEGORIES

__all__ = [
    "ETFPool",
    "ETFCategorizer",
    "DEFAULT_POOL",
    "rank_by_momentum",
    "rank_pctl",
    "distance_to_52w_high",
    "DiversificationCaps",
    "RotationConfig",
    "PortfolioState",
    "select_and_weight",
    "apply_stops",
    "inverse_vol_weights",
    "equal_weights",
    "load_etf_nav_panel",
    "load_bond_etf_nav",
    "FixedIncomePlus",
    "FixedIncomePlusConfig",
    "FixedIncomePlusResult",
    "performance_metrics",
    "BacktestConfig",
    "RotationBacktestResult",
    "CICC_BASELINES",
    "compare_to_cicc",
    "run_equal_weight_baseline",
    "run_rotation_backtest",
    "ValidationConfig",
    "ValidationResult",
    "ValidationReport",
    "ablation",
    "run_full_validation",
    "validate_parameter_perturbation",
    "validate_rebalance_offsets",
    "validate_starting_points",
    "extended_metrics",
    "format_metrics_table",
    "reconstruct_daily_weights",
    "etf_contribution",
    "category_contribution",
    "risk_contribution",
    "marginal_contribution",
    "period_contribution",
    "DEFAULT_PERIODS",
    "brinson_attribution",
    "CATEGORIES",
    "estimate_covariance",
    "ledoit_wolf_shrinkage",
    "solve_risk_parity",
    "solve_max_diversification",
]

# 版本管理 (Stage 12A 引入)
from .strategy_versions import (
    v0_0_baseline, v0_1_vt_only, v0_2_tf_only, v0_3_vt_cost, v0_4_hybrid,
    v1_0, VERSIONS, LATEST, get_version,
)
