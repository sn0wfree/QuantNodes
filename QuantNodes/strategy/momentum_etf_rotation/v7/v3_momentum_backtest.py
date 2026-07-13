# coding=utf-8
"""v7_macro_baseline_v3_momentum 回测入口.

[设计原则]
- 不修改原始 v7+v2 代码
- 复用 run_v7_3_backtest (返回 weights=True) 获取 FRP 权重
- 在 FRP 权重上叠加动量倾斜 (Option A) 或修改 factor_panel (Option B)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .macro_substrategy_v7_3 import V7_3Config, run_v7_3_backtest, BOND_INDICES
from .data_loader import INDEX_COLS
from .momentum_overlay import (
    EQUITY_INDICES, COMMODITY_INDICES, MOMENTUM_UNIVERSE,
    apply_momentum_tilt_a,
    add_momentum_factor_to_panel,
)
from .factor_risk_parity import FactorRiskParityOptimizer
from .bootstrap_lasso import BootstrapLassoMapping
from .macro_substrategy_v7_3 import symmetry_full_window, V7_3SubStrategy


def v3_momentum_config(
    momentum_type: str = "hybrid",
    lookback: int = 90,
    alpha: float = 0.3,
    option: str = "A",
    tf_enabled: bool = False,
    fused_weight: float = 0.5,
) -> V7_3Config:
    """v7 + 动量增强版配置.

    Args:
        momentum_type: "price" / "slope_r2" / "hybrid"
        lookback: 动量回看天数
        alpha: 动量混合系数 (Option A)
        option: "A" = overlay, "B" = 10th factor
        tf_enabled: 是否同时启用 TF
        fused_weight: hybrid 中 slope_r2 的权重
    """
    cfg = V7_3Config()
    # Momentum
    cfg.momentum_enabled = True
    cfg.momentum_type = momentum_type
    cfg.momentum_lookback = lookback
    cfg.momentum_alpha = alpha
    cfg.momentum_option = option
    cfg.momentum_fused_weight = fused_weight
    # TF (可选)
    if tf_enabled:
        cfg.trend_filter_enabled = True
        cfg.trend_filter_benchmark = "沪深300指数"
        cfg.trend_filter_ma = 200
        cfg.trend_filter_bear = 0.5
    return cfg


# 给 V7_3Config 加动态字段 (不改原 dataclass)
# 用 monkey-patch 方式, 因为 momentum_overlay 需要这些字段
def _ensure_momentum_fields(cfg: V7_3Config) -> None:
    """确保 cfg 有 momentum 字段 (向后兼容)."""
    if not hasattr(cfg, 'momentum_enabled'):
        cfg.momentum_enabled = False
    if not hasattr(cfg, 'momentum_type'):
        cfg.momentum_type = "hybrid"
    if not hasattr(cfg, 'momentum_lookback'):
        cfg.momentum_lookback = 90
    if not hasattr(cfg, 'momentum_alpha'):
        cfg.momentum_alpha = 0.3
    if not hasattr(cfg, 'momentum_option'):
        cfg.momentum_option = "A"
    if not hasattr(cfg, 'momentum_fused_weight'):
        cfg.momentum_fused_weight = 0.5


def run_v3_momentum_backtest(
    index_panel: pd.DataFrame,
    factor_panel: pd.DataFrame,
    cfg: V7_3Config,
    benchmark_price: pd.Series | None = None,
    return_weights: bool = False,
) -> pd.Series | tuple[pd.Series, pd.DataFrame]:
    """v7 + 动量回测 (Option A: overlay, 复用 v7 FRP).

    流程:
    1. 用 v7 的 run_v7_3_backtest 获取每个调仓日的 FRP 权重
    2. 对每个调仓日, 用 apply_momentum_tilt_a 叠加动量倾斜
    3. 用倾斜后的权重计算收益

    Args:
        index_panel: 13 指数日收益
        factor_panel: 9 宏观因子周收益
        cfg: V7_3Config (含 momentum_* 字段)
        benchmark_price: 沪深300 价格 (TF 需要)
        return_weights: 是否返回权重 DataFrame

    Returns:
        nav Series or (nav, weights_df)
    """
    _ensure_momentum_fields(cfg)

    if not cfg.momentum_enabled:
        # 不启动动量, 直接调 v7
        return run_v7_3_backtest(index_panel, factor_panel, cfg, benchmark_price, return_weights)

    if cfg.momentum_option == "B":
        # Option B: 修改 factor_panel, 加第10因子
        factor_panel = add_momentum_factor_to_panel(
            factor_panel, index_panel, cfg.momentum_lookback,
        )

    # 获取 v7 的 FRP 权重
    nav_base, w_base = run_v7_3_backtest(
        index_panel, factor_panel, cfg, benchmark_price, return_weights=True,
    )

    if cfg.momentum_option == "A":
        # Option A: 在 FRP 权重上叠加动量倾斜
        w_momentum = {}
        for dt in w_base.index:
            w_series = w_base.loc[dt]
            w_tilted = apply_momentum_tilt_a(
                w_series, index_panel, dt,
                lookback=cfg.momentum_lookback,
                momentum_type=cfg.momentum_type,
                alpha=cfg.momentum_alpha,
                fused_weight=cfg.momentum_fused_weight,
                max_weight=cfg.max_weight,
            )
            w_momentum[dt] = w_tilted

        w_momentum_df = pd.DataFrame(w_momentum).T
        w_momentum_df.index.name = "rebalance_date"

        # 用倾斜后的权重重新计算收益
        from .macro_substrategy_v7_3 import BOND_INDICES as _BOND
        cost_rate = (cfg.commission_bp + cfg.slippage_bp) / 10000.0
        weight_dates = sorted(w_momentum_df.index)

        all_ret = []
        for s, e in zip(weight_dates[:-1], weight_dates[1:]):
            mask = (index_panel.index >= s) & (index_panel.index < e)
            if not mask.any():
                continue
            idx_ret_window = index_panel.loc[mask, list(cfg.index_pool)]
            ret_data = idx_ret_window.values @ w_momentum_df.loc[s].reindex(cfg.index_pool).fillna(0).values
            ret_series = pd.Series(ret_data, index=idx_ret_window.index)

            # 调仓日成本
            turnover = np.abs(
                w_momentum_df.loc[e].reindex(cfg.index_pool).fillna(0).values
                - w_momentum_df.loc[s].reindex(cfg.index_pool).fillna(0).values
            ).sum() / 2.0
            cost = turnover * cost_rate
            ret_series.iloc[0] -= cost
            all_ret.append(ret_series)

        if not all_ret:
            raise ValueError("No returns computed")

        all_ret_series = pd.concat(all_ret)
        nav = (1 + all_ret_series).cumprod()
        nav = nav / nav.iloc[0]

        if return_weights:
            return nav, w_momentum_df
        return nav

    # Option B: 直接用修改后的 factor_panel 跑 v7
    return run_v7_3_backtest(index_panel, factor_panel, cfg, benchmark_price, return_weights)
