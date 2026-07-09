# coding=utf-8
"""共享代码: v1 和 v2 共用的基础设施.

包含: ETF 池, 数据加载, 评估指标, 贡献分析, Brinson, 验证.
不包含: 版本特定的策略逻辑 (在 v1/ 和 v2/ 中).
"""
from __future__ import annotations

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

from .extended_metrics import extended_metrics, format_metrics_table
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


__all__ = [
    "Category",
    "DEFAULT_POOL",
    "ETFCategorizer",
    "ETFMeta",
    "ETFPool",
    "RegimeDetector",
    "ValidationConfig",
    "ValidationReport",
    "ValidationResult",
    "ablation",
    "brinson_attribution",
    "CATEGORIES",
    "category_contribution",
    "condition_number",
    "CovMethod",
    "diagonal_covariance",
    "estimate_covariance",
    "etf_contribution",
    "extended_metrics",
    "ewma_covariance",
    "format_metrics_table",
    "is_positive_definite",
    "ledoit_wolf_shrinkage",
    "load_bond_etf_nav",
    "load_etf_nav_panel",
    "marginal_contribution",
    "period_contribution",
    "reconstruct_daily_weights",
    "risk_contribution",
    "risk_parity_objective",
    "rp_risk_contribution",
    "run_full_validation",
    "sample_covariance",
    "solve_max_diversification",
    "solve_risk_parity",

    "validate_parameter_perturbation",
    "validate_rebalance_offsets",
    "validate_starting_points",
]