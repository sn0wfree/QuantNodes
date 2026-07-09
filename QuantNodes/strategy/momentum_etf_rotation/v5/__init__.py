# coding=utf-8
"""v5 策略模块 — Stage 18 升级版.

基于 Stage 17 v4 诊断研究, 实施 2 个独立子策略优化版:
- StyleRotationV5: 4 改进 (多窗口 + dividend 底仓 + Top-2 + sideways filter)
- FactorTimingV5: 5 改进 (因子特异性 FW + lag 平滑 + regime-conditioned + 删 low_vol + IC 质量过滤)

诊断: reports/momentum_etf_rotation/v4/SUB_STRATEGY_DIAGNOSTIC.md
"""
from .style_rotation_v5 import (
    StyleRotationV5Config,
    StyleRotationV5SubStrategy,
    multi_window_score,
    classify_regime,
)
from .factor_timing_v5 import (
    FactorTimingV5Config,
    FactorTimingV5SubStrategy,
    compute_v5_factor_weights,
    aggregate_factor_to_etf,
    compute_ic_history_v5,
    classify_regime_v5,
)

__all__ = [
    "StyleRotationV5Config",
    "StyleRotationV5SubStrategy",
    "FactorTimingV5Config",
    "FactorTimingV5SubStrategy",
    "multi_window_score",
    "classify_regime",
    "compute_v5_factor_weights",
    "aggregate_factor_to_etf",
    "compute_ic_history_v5",
    "classify_regime_v5",
]
