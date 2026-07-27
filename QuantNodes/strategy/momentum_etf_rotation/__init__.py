# coding=utf-8
"""Momentum ETF Rotation 策略模块.

CICC 2026-07-03 《固收+:"可靠"的动量 ETF 轮动及 Agent 检验实践》复现.

四步组合管理:
    1. 去重 + 剔高相关 (同指数只留流动性最好, 相关 > corr_threshold 跳过)
    2. 强制分散 (A 股宽基/行业 ≤ a_share, HK ≤ hk, 必含商品/海外)
    3. 逆波动加权 (权重 ∝ 1/σ)
    4. 止损 + 补位 (跌破 ma_window 且 排名跌出后 rank_cutoff 分位)

版本:
    - v1: CICC 原始复现 (Stage 8, 仅 price momentum, 纯 4 步组合管理)
    - v2: 增强版 (Stage 12A, hybrid momentum + VT + Cost + TF — 已冻结, 默认)
    - v3: 多策略组合 (Stage 16A, 动量 + 均值反转 + 行业轮动 — 新建)
"""
from __future__ import annotations

# ─── 共享基础 (common/) ─────────────────────────────────────
from .common.universe import ETFPool, ETFCategorizer, DEFAULT_POOL
from .common.data import load_etf_nav_panel, load_bond_etf_nav
from .common.extended_metrics import extended_metrics, format_metrics_table
from .common.contribution import (
    reconstruct_daily_weights,
    etf_contribution,
    category_contribution,
    risk_contribution,
    marginal_contribution,
    period_contribution,
)
from .common.brinson import brinson_attribution, CATEGORIES
from .common.validation import (
    ValidationConfig,
    ValidationResult,
    ValidationReport,
    ablation,
    run_full_validation,
    validate_parameter_perturbation,
    validate_rebalance_offsets,
    validate_starting_points,
)
from .common.covariance import (
    CovMethod,
    estimate_covariance,
    ledoit_wolf_shrinkage,
    sample_covariance,
    ewma_covariance,
    diagonal_covariance,
)
from .common.risk_parity import (
    risk_contribution as rp_risk_contribution,
    solve_risk_parity,
    solve_max_diversification,
)

# ─── v2 (默认, 当前增强版) ─────────────────────────────────
from .core.momentum import (
    rank_by_momentum, rank_pctl, distance_to_52w_high,
    slope_r2_score, hybrid_momentum_score, compute_momentum_score,
)
from .core.portfolio import (
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
    calculate_turnover_cost,
    inverse_vol_weights,
    equal_weights,
)
from .common.fixed_income_plus import (
    FixedIncomePlus,
    FixedIncomePlusConfig,
    FixedIncomePlusResult,
)
from .common.metrics import performance_metrics_legacy as performance_metrics
from .core.backtest import (
    BacktestConfig,
    RotationBacktestResult,
    CICC_BASELINES,
    compare_to_cicc,
    run_equal_weight_baseline,
    run_rotation_backtest,
)
from .core.strategy_versions import (
    v0_0_baseline, v0_1_vt_only, v0_2_tf_only, v0_3_vt_cost, v0_4_hybrid,
    v1_0, VERSIONS, LATEST, get_version,
)


# ─── v3 (Stage 16A, 多策略组合 — 新建) ──────────────────────
from .v3 import (
    SubStrategy as SubStrategy_v3,
    SubStrategyConfig as SubStrategyConfig_v3,
    SubStrategyResult as SubStrategyResult_v3,
)

# ─── v7 (Stage 30.4 宏观子策略 — baseline 锁定 2026-07-13) ────
from .v7 import (
    V7_3Config,
    V7_3SubStrategy,
    run_v7_3_backtest,
    v7_macro_baseline,
)

__all__ = [
    "ETFPool", "ETFCategorizer", "DEFAULT_POOL",
    "rank_by_momentum", "rank_pctl", "distance_to_52w_high",
    "DiversificationCaps", "RotationConfig", "PortfolioState",
    "select_and_weight", "apply_stops",
    "inverse_vol_weights", "equal_weights",
    "load_etf_nav_panel", "load_bond_etf_nav",
    "FixedIncomePlus", "FixedIncomePlusConfig", "FixedIncomePlusResult",
    "performance_metrics",
    "BacktestConfig", "RotationBacktestResult",
    "CICC_BASELINES", "compare_to_cicc",
    "run_equal_weight_baseline", "run_rotation_backtest",
    "ValidationConfig", "ValidationResult", "ValidationReport",
    "ablation", "run_full_validation",
    "validate_parameter_perturbation", "validate_rebalance_offsets",
    "validate_starting_points",
    "extended_metrics", "format_metrics_table",
    "reconstruct_daily_weights", "etf_contribution",
    "category_contribution", "risk_contribution",
    "marginal_contribution", "period_contribution",
    "brinson_attribution", "CATEGORIES",
    "TrendFilter", "VolTargeting", "ConcentrationCaps", "CostModel",
    "apply_trend_filter", "apply_vol_targeting", "apply_concentration_caps",
    "vol_targeting_scale", "check_trend_filter", "calculate_turnover_cost",
    "slope_r2_score", "hybrid_momentum_score", "compute_momentum_score",
    "CovMethod", "estimate_covariance",
    "ledoit_wolf_shrinkage", "sample_covariance",
    "ewma_covariance", "diagonal_covariance",
    "solve_risk_parity", "solve_max_diversification", "rp_risk_contribution",
    "v0_0_baseline", "v0_1_vt_only", "v0_2_tf_only",
    "v0_3_vt_cost", "v0_4_hybrid", "v1_0",
    "VERSIONS", "LATEST", "get_version",
    # v3 (Stage 16A)
    "SubStrategy_v3", "SubStrategyConfig_v3", "SubStrategyResult_v3",
    # v7 (Stage 30.4 宏观子策略, baseline 锁定 2026-07-13)
    "V7_3Config", "V7_3SubStrategy", "run_v7_3_backtest", "v7_macro_baseline",
]
