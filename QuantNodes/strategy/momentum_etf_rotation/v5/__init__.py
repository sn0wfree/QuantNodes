# coding=utf-8
"""v5 策略模块 — Stage 22 行业量价因子行业轮动 (等权, 论文做法).

基于华西证券《行业有效量价因子与行业轮动策略》 (2022-08-22):
- 6 大类 11 月频因子 (动量/交易波动/换手率/多空对比/量价背离/量幅同向)
- 复合因子 = z-score 等权 (论文用 IC 加权)
- 月末选 Top-N ETF 等权 (论文 Top-5)

v5 vs Stage 19 industry_factors.py:
- 升级为完整 SubStrategy (继承 v4.sub_strategy_v4)
- 完整接口: select/weight/run_step
- 可与 v3/v4 在 multi_strategy 框架下组合

升级版: v5_1/industry_rotation_v5_1.py (逆波动率加权, OOS Calmar 0.488 → 0.589)

诊断基础:
- reports/momentum_etf_rotation/v4/INDUSTRY_ROTATION_REPORT.md (Stage 19 实施)
- reports/momentum_etf_rotation/v4/SUB_STRATEGY_DIAGNOSTIC.md (Stage 18 诊断)

回测 (2018-2026 8y, 等权):
- 量价因子 Top-5 单独:   Calmar 0.745  OOS 0.488
- v3 80% + 量价 20%:    Calmar 0.706  OOS 0.955 ⭐
- v3 70% + 量价 30%:    Calmar 0.723  OOS 0.903
"""
from .industry_factors import (
    FactorEngineConfig,
    compute_single_factor,
    compute_all_factors,
    compute_all_factors_panel,
    _second_order_momentum,
    _momentum_term_diff,
    _amount_volatility,
    _volume_volatility,
    _turnover_change,
    _long_short_total,
    _long_short_change,
    _rank_covariance,
    _price_volume_correlation,
    _first_order_divergence,
    _volume_range_codirection,
)
from .industry_rotation_v5 import (
    IndustryRotationV5Config,
    IndustryRotationV5SubStrategy,
    cross_section_zscore,
    compute_composite_factor,
)
from .industry_rotation_v5_1 import (
    IndustryRotationV5_1Config,
    IndustryRotationV5_1SubStrategy,
    inverse_vol_weights_v5_1,
)

__all__ = [
    # v5 等权版
    "IndustryRotationV5Config",
    "IndustryRotationV5SubStrategy",
    "FactorEngineConfig",
    "compute_single_factor",
    "compute_all_factors",
    "compute_all_factors_panel",
    "cross_section_zscore",
    "compute_composite_factor",
    "_second_order_momentum",
    "_momentum_term_diff",
    "_amount_volatility",
    "_volume_volatility",
    "_turnover_change",
    "_long_short_total",
    "_long_short_change",
    "_rank_covariance",
    "_price_volume_correlation",
    "_first_order_divergence",
    "_volume_range_codirection",
    # v5.1 逆波动率版
    "IndustryRotationV5_1Config",
    "IndustryRotationV5_1SubStrategy",
    "inverse_vol_weights_v5_1",
]
