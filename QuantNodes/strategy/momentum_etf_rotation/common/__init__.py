# coding=utf-8
"""共享代码: v1-v10 共用的基础设施.

包含: ETF 池, 数据加载, 评估指标, 回测引擎.
不包含: 版本特定的策略逻辑 (在 v1/-v10/ 中).
"""
from __future__ import annotations

from .backtest_config import (
    BacktestConfig,
    CostConfig,
    StopLossConfig,
    TrendFilter,
    VolTargeting,
)
from .backtest_engine import BacktestCallbacks, BacktestResult, run_backtest
from .strategy_engine import BaseStrategy, StrategyEngine
from .backtest_utils import (
    apply_max_weight,
    calculate_turnover,
    calculate_turnover_cost,
    generate_rebalance_dates,
    normalize_weights,
)
from .brinson import CATEGORIES, brinson_attribution
from .contribution import (
    category_contribution,
    etf_contribution,
    marginal_contribution,
    period_contribution,
    risk_contribution as brinson_risk_contribution,
    reconstruct_daily_weights,
    risk_contribution,
)
from .covariance import (
    CovMethod,
    condition_number,
    diagonal_covariance,
    estimate_covariance,
    ewma_covariance,
    is_positive_definite,
    ledoit_wolf_shrinkage,
    sample_covariance,
)
from .data import load_bond_etf_nav, load_etf_nav_panel
from .extended_metrics import extended_metrics, format_metrics_table, kelly_audit
from .metrics import compute_metrics, detect_freq, performance_metrics_legacy, format_metrics_table as fmt_table
from .drawdown_controller import DrawdownConfig, DrawdownState, drawdown_multiplier
from .regime_detector import RegimeDetector
from .risk_parity import (
    risk_contribution as rp_risk_contribution,
    risk_parity_objective,
    solve_max_diversification,
    solve_risk_parity,
)
from .universe import (
    DEFAULT_POOL,
    ETFCategorizer,
    ETFPool,
    ETFMeta,
    Category,
)
from .validation import (
    ValidationConfig,
    ValidationReport,
    ValidationResult,
    ablation,
    run_full_validation,
    validate_parameter_perturbation,
    validate_rebalance_offsets,
    validate_starting_points,
)
from .walk_forward import (
    GridSearchSpace,
    WalkForwardConfig,
    WalkForwardResult,
    WalkResult,
    concat_oos_nav,
    grid_search,
    walk_forward,
)
from .rd_utils import (
    compute_weekly_metrics,
    compute_daily_metrics,
    compute_beta_stability,
    compute_tv_norm,
    compute_cross_sectional_ic,
    compute_ic_summary,
    align_daily_to_weekly,
)

__all__ = [
    # Metrics (统一指标计算)
    "compute_metrics", "detect_freq", "fmt_table", "performance_metrics_legacy",
    # Strategy engine
    "BaseStrategy", "StrategyEngine", "StrategyResult",
    # Backtest engine
    "BacktestCallbacks", "BacktestConfig", "BacktestResult", "run_backtest",
    # Config
    "CostConfig", "DrawdownConfig", "StopLossConfig", "TrendFilter", "VolTargeting",
    # Utils
    "apply_max_weight", "calculate_turnover", "calculate_turnover_cost",
    "generate_rebalance_dates", "normalize_weights",
    # Universe
    "Category", "DEFAULT_POOL", "ETFCategorizer", "ETFMeta", "ETFPool",
    # Others
    "DrawdownState", "RegimeDetector",
    "ValidationConfig", "ValidationReport", "ValidationResult",
    "ablation", "brinson_attribution", "CATEGORIES",
    "category_contribution", "condition_number", "CovMethod",
    "diagonal_covariance", "drawdown_multiplier", "estimate_covariance",
    "etf_contribution", "extended_metrics", "ewma_covariance",
    "format_metrics_table", "is_positive_definite", "kelly_audit",
    "ledoit_wolf_shrinkage", "load_bond_etf_nav", "load_etf_nav_panel",
    "marginal_contribution", "period_contribution",
    "reconstruct_daily_weights", "risk_contribution", "brinson_risk_contribution",
    "risk_parity_objective", "rp_risk_contribution",
    "run_full_validation", "sample_covariance",
    "solve_max_diversification", "solve_risk_parity",
    "validate_parameter_perturbation", "validate_rebalance_offsets",
    "validate_starting_points",
    # Walk-forward
    "GridSearchSpace", "WalkForwardConfig", "WalkForwardResult", "WalkResult",
    "concat_oos_nav", "grid_search", "walk_forward",
    # R&D utils
    "compute_weekly_metrics", "compute_daily_metrics", "compute_beta_stability",
    "compute_tv_norm", "compute_cross_sectional_ic", "compute_ic_summary",
    "align_daily_to_weekly",
]
