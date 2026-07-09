# coding=utf-8
"""v4 = Stage 17 风格轮动 + Smart β + 因子择时.

与 v3 完全独立:
- v3 (Stage 16A) 多策略组合在 ../v3/
- v4 (Stage 17) 风格/Smart β/因子择时 在 ./ (本目录)

子策略:
- 风格轮动 (style_rotation_v4.py) - 大盘/中盘/成长/科创/红利
- Smart β (smart_beta_v4.py) - 红利低波/低波/质量/价值/现金流
- 因子择时 (factor_timing_v4.py) - IC + HMM 融合, 控制子策略权重

参考: reports/momentum_etf_rotation/v4/STAGE17_PLAN.md
"""
from __future__ import annotations

from .universe_v4 import (
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
    get_regime_factor_weight,
)
from .factor_timing_v4 import (
    FactorTimingConfig,
    backtest_factor_timing,
    backtest_factor_weights_history,
    compute_factor_weights,
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

__all__ = [
    # Universe
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
    # Style Rotation
    "StyleRotationConfig",
    "StyleRotationSubStrategy",
    "style_rotation_score",
    "select_top_styles",
    "style_etf_picks",
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
    "get_regime_factor_weight",
    # Factor Timing
    "FactorTimingConfig",
    "compute_factor_weights",
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
]
