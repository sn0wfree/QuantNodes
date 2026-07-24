# coding=utf-8
"""v4 = Stage 17 风格轮动 + Smart β + 因子择时 + Stage 27 重构 (适配 43 ETF).

与 v3 完全独立:
- v3 (Stage 16A) 多策略组合在 ../v3/
- v4 (Stage 17) 风格/Smart β/因子择时 在 ./ (本目录)

Stage 27 重构:
- 适配 43 ETF (宽基 6 + 行业 23 + 海外 11 + 黄金 3)
- 大类轮动替代风格轮动
- Smart β 用行业 ETF 代理筛选
- 行业轮动适配 43 ETF
- 宏观因子融合 regime 检测

子策略:
- 风格轮动 (style_rotation_v4.py) - 兼容旧版
- 大类轮动 (AssetClassRotation) - Stage 27 新增
- Smart β (smart_beta_v4.py) - 行业 ETF 代理
- 因子择时 (factor_timing_v4.py) - 8 因子 IC + HMM 融合
- 行业轮动 (industry_rotation_v4.py) - 23 行业 ETF + regime

参考: reports/momentum_etf_rotation/v4/STAGE17_PLAN.md
"""
from __future__ import annotations

from .universe_v4 import (
    # 43 ETF 分类
    BROAD_CODES,
    SECTOR_CODES,
    OVERSEAS_CODES,
    GOLD_CODES,
    DEFENSIVE_SECTOR_CODES,
    GROWTH_SECTOR_CODES,
    AssetClass,
    ASSET_CLASS_CODES,
    get_all_43_codes,
    classify_43_etf,
    select_smart_beta_proxy,
    select_defensive_smart_beta,
    # 兼容旧版
    ALL_V4_CODES,
    SMART_BETA_CODES,
    SMART_BETA_FACTOR_TYPE,
    SMART_BETA_METAS,
    STYLE_GROUP_CODES,
    STYLE_GROUP_METAS,
    SmartBetaFactor,
    StyleGroup,
    all_smart_beta_codes,
    all_style_codes,
    export_style_groups,
    load_smartbeta_panel,
    smart_beta_of,
    style_group_of,
)
from .sub_strategy_v4 import (
    SubStrategy,
    SubStrategyConfig,
    SubStrategyResult,
)
from .style_rotation_v4 import (
    StyleRotationConfig,
    StyleRotationSubStrategy,
    select_top_styles,
    style_etf_picks,
    style_rotation_score,
    # Stage 27: 大类轮动
    AssetClassRotationConfig,
    AssetClassRotation,
    asset_class_rotation_score,
    select_top_asset_classes,
    asset_class_etf_picks,
)
from .smart_beta_v4 import (
    SmartBetaConfig,
    SmartBetaSubStrategy,
    select_diversified_smart_beta,
    select_top_smart_beta,
    smart_beta_score,
)
from .factor_ic import (
    FACTOR_NAMES,
    compute_factor_scores,
    compute_forward_return,
    factor_ic_at,
    factor_ic_rolling_mean,
    rolling_factor_ic,
)
from .regime_detector_v4 import (
    REGIME_LABELS,
    RegimeConfig,
    RegimeDetector,
    detect_regime_simple,
    detect_regime_with_macro,
    get_regime_factor_weight,
)
from .factor_timing_v4 import (
    FactorTimingConfig,
    backtest_factor_timing,
    backtest_factor_weights_history,
    compute_factor_weights,
    compute_factor_weights_fusion,
    compute_factor_weights_hmm,
    compute_strategy_weights,
)
from .regime_transitions import (
    POTENTIAL,
    DistanceTransitionConfig,
    build_distance_transmat,
    distance_between,
    distance_rate,
    effective_distance,
    enforce_minimum_duration,
    soft_constrain,
    validate_transmat,
)
from .multi_strategy_v4 import (
    V4Config,
    V4Mode,
    V4Result,
    run_v4_backtest,
    run_v4_mode,
)
from .industry_rotation_v4 import (
    IndustryRotationConfig,
    IndustryRotationV4,
    DEFENSIVE_INDUSTRIES,
    GROWTH_INDUSTRIES,
)

__all__ = [
    # 43 ETF 分类 (Stage 27)
    "BROAD_CODES",
    "SECTOR_CODES",
    "OVERSEAS_CODES",
    "GOLD_CODES",
    "DEFENSIVE_SECTOR_CODES",
    "GROWTH_SECTOR_CODES",
    "AssetClass",
    "ASSET_CLASS_CODES",
    "get_all_43_codes",
    "classify_43_etf",
    "select_smart_beta_proxy",
    "select_defensive_smart_beta",
    # Universe (兼容旧版)
    "ALL_V4_CODES",
    "STYLE_GROUP_CODES",
    "STYLE_GROUP_METAS",
    "StyleGroup",
    "smart_beta_of",
    "style_group_of",
    "all_style_codes",
    "all_smart_beta_codes",
    "load_smartbeta_panel",
    "export_style_groups",
    # Smart β
    "SMART_BETA_CODES",
    "SMART_BETA_FACTOR_TYPE",
    "SMART_BETA_METAS",
    "SmartBetaFactor",
    # SubStrategy
    "SubStrategy",
    "SubStrategyConfig",
    "SubStrategyResult",
    # Style Rotation (兼容旧版)
    "StyleRotationConfig",
    "StyleRotationSubStrategy",
    "style_rotation_score",
    "select_top_styles",
    "style_etf_picks",
    # 大类轮动 (Stage 27)
    "AssetClassRotationConfig",
    "AssetClassRotation",
    "asset_class_rotation_score",
    "select_top_asset_classes",
    "asset_class_etf_picks",
    # Smart Beta
    "SmartBetaConfig",
    "SmartBetaSubStrategy",
    "smart_beta_score",
    "select_top_smart_beta",
    "select_diversified_smart_beta",
    # Factor IC
    "FACTOR_NAMES",
    "compute_factor_scores",
    "compute_forward_return",
    "factor_ic_at",
    "factor_ic_rolling_mean",
    "rolling_factor_ic",
    # Regime Detector
    "REGIME_LABELS",
    "RegimeConfig",
    "RegimeDetector",
    "detect_regime_simple",
    "detect_regime_with_macro",
    "get_regime_factor_weight",
    # Factor Timing
    "FactorTimingConfig",
    "compute_factor_weights",
    "compute_factor_weights_fusion",
    "compute_factor_weights_hmm",
    "compute_strategy_weights",
    "backtest_factor_timing",
    "backtest_factor_weights_history",
    # Regime Transitions (距离先验)
    "POTENTIAL",
    "DistanceTransitionConfig",
    "build_distance_transmat",
    "distance_between",
    "distance_rate",
    "effective_distance",
    "enforce_minimum_duration",
    "soft_constrain",
    "validate_transmat",
    # Multi Strategy
    "V4Config",
    "V4Mode",
    "V4Result",
    "run_v4_backtest",
    "run_v4_mode",
    # Industry Rotation (Stage 27)
    "IndustryRotationConfig",
    "IndustryRotationV4",
    "DEFENSIVE_INDUSTRIES",
    "GROWTH_INDUSTRIES",
]
