# coding=utf-8
"""v1 = 原始CICC复现 (Stage 8 baseline).

特性:
  - 4 步组合管理 (去重 + 剔高相关, 强制分散, 逆波动加权, 止损 + 补位)
  - momentum_type = "price" (固定, 无 hybrid 选项)
  - 无 VolTargeting, CostModel, ConcentrationCaps
  - Calmar ~0.78 (vs CICC 0.76)

如需增强功能 (hybrid 动量, VT, Cost, 等), 请使用 v2.
"""
from __future__ import annotations

from .momentum_v1 import (
    below_ma_v1,
    distance_to_52w_high_v1,
    pairwise_corr_v1,
    rank_by_momentum_v1,
    rank_pctl_v1,
    realized_vol_v1,
)
from .portfolio_v1 import (
    DiversificationCaps_v1,
    PortfolioState_v1,
    RotationConfig_v1,
    apply_stops_v1,
    equal_weights_v1,
    inverse_vol_weights_v1,
    select_and_weight_v1,
)
from .backtest_v1 import (
    BacktestConfig_v1,
    RotationBacktestResult_v1,
    run_equal_weight_baseline_v1,
    run_rotation_backtest_v1,
)
from .strategy_versions_v1 import (
    LATEST_v1,
    VERSIONS_v1,
    get_version_v1,
    v1_0_0,
)


__all__ = [
    # Momentum
    "rank_by_momentum_v1",
    "rank_pctl_v1",
    "distance_to_52w_high_v1",
    "realized_vol_v1",
    "below_ma_v1",
    "pairwise_corr_v1",
    # Portfolio
    "DiversificationCaps_v1",
    "RotationConfig_v1",
    "PortfolioState_v1",
    "select_and_weight_v1",
    "apply_stops_v1",
    "inverse_vol_weights_v1",
    "equal_weights_v1",
    # Backtest
    "BacktestConfig_v1",
    "RotationBacktestResult_v1",
    "run_rotation_backtest_v1",
    "run_equal_weight_baseline_v1",
    # Versions
    "v1_0_0",
    "VERSIONS_v1",
    "LATEST_v1",
    "get_version_v1",
]
