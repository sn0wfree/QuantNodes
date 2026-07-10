# coding=utf-8
"""v6.2 量价族 + IC 加权 + 因子正交化 (Stage 27 v6.2).

v6.2 = v5 选股 + IC 加权 + 因子正交化 (残差化) + v5.1.1 加权.

与 v6.1 区别:
- 在因子计算之后, 选股之前, 加一层因子正交化
- 用 Gram-Schmidt 残差法, 顺序按 OOS IR 降序

无风控层 (与 v6.1 一致).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..v4.sub_strategy_v4 import SubStrategyConfig
from ..v5.industry_factors import FactorEngineConfig
from ..v5.industry_rotation_v5 import (
    IndustryRotationV5SubStrategy,
    compute_composite_factor,
)
from ..v5_1.industry_rotation_v5_1 import inverse_vol_weights_v5_1
from .factor_orthogonal import (
    get_factor_ir_order,
    orthogonalize_factor_panel,
)
from ..v6_1.factor_weighting import (
    compute_ic_timeseries,
    compute_factor_weights,
    align_weights_with_rebal_dates,
    DEFAULT_HORIZON_DAYS,
    MIN_MONTHS_FOR_IC,
    DEFAULT_SMOOTH_WINDOW,
)
from ..v6_1.industry_rotation_v6_1 import V6_1Config, V6_1SubStrategy


@dataclass
class V6_2Config(SubStrategyConfig):
    """v6.2 配置: v6.1 + 因子正交化.

    字段继承 V6_1Config 全部, 加:
    - use_orthogonal: 是否启用正交化
    - ir_order_min_periods: IR 排序最少样本
    """
    name: str = "industry_rotation_v6_2"

    # 继承 v6.1 全部 (复用)
    factor_cfg: FactorEngineConfig = field(default_factory=FactorEngineConfig)
    top_n: int = 5
    factor_weights: dict[str, float] | None = None

    # IC 加权
    use_ic_weighting: bool = True
    ic_horizon_days: int = DEFAULT_HORIZON_DAYS
    ic_min_months: int = MIN_MONTHS_FOR_IC
    ic_smooth_window: int = DEFAULT_SMOOTH_WINDOW

    # v6.2 新增
    use_orthogonal: bool = True
    ir_order_min_periods: int = 12  # IR 排序最少样本 (月)

    # 调仓
    rebalance_freq: str = "M"
    min_history: int = 252

    # ETF 池
    universe: tuple[str, ...] | None = None

    # 加权层 (来自 v5.1.1)
    max_weight: float = 0.25
    vol_window: int = 60
    vol_floor: float = 0.01
    rebal_lag: int = 1


class V6_2SubStrategy(V6_1SubStrategy):
    """v6.2 子策略, 继承 v6.1 (含 IC 加权), 加正交化."""
    config: V6_2Config

    def __init__(self, config: V6_2Config):
        super().__init__(config)
        self.config: V6_2Config = config


def _orthogonalize_panel(factor_panel, factor_cfg, panel_close, rebal_dates):
    """对原始 panel 做正交化, 返回 panel."""
    factors = list(factor_cfg.name_map.keys())
    # 1. 按 OOS IR 排序
    order = get_factor_ir_order(
        factor_panel, panel_close, rebal_dates, factors,
        horizon=21, min_periods=12,
    )
    if len(order) < 2:
        return factor_panel, factors  # 不正交化
    # 2. 正交化
    panel_orth = orthogonalize_factor_panel(factor_panel, order, rebal_dates)
    # 取新因子的列表 (按 IR 排序, 只保留 OOS IR > 0 的)
    return panel_orth, order


def run_v6_2_backtest(
    panel_close: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    cfg: V6_2Config | None = None,
    rebalance_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.Series:
    """v6.2 回测: 与 v6.1 流程一致 + 因子正交化.

    Args:
        panel_close: 收盘价面板
        panel_ohlcv: OHLCV 面板
        cfg: v6.2 config (None = 默认)
        rebalance_dates: 调仓日期 (None = 月末)

    Returns:
        NAV Series
    """
    if cfg is None:
        cfg = V6_2Config()

    dates = panel_close.index

    # 1. 调仓日期
    if rebalance_dates is None:
        rebal_dates_idx = dates.to_series().resample("ME").last().index
    else:
        rebal_dates_idx = pd.DatetimeIndex(rebalance_dates)
    rebal_set = set(d for d in rebal_dates_idx if d in dates)

    # 2. 预算 11 因子
    from ..v5.industry_factors import compute_all_factors_panel
    factor_panel_raw = compute_all_factors_panel(panel_ohlcv, cfg.factor_cfg)
    factors = list(cfg.factor_cfg.name_map.keys())

    # 2.5 正交化 (若启用)
    if cfg.use_orthogonal:
        factor_panel_used, factors_used = _orthogonalize_panel(
            factor_panel_raw, cfg.factor_cfg, panel_close, rebal_dates_idx,
        )
    else:
        factor_panel_used = factor_panel_raw
        factors_used = factors

    sub = V6_2SubStrategy(cfg)
    sub._factor_panel = factor_panel_used
    if len(dates) > 0:
        sub._last_init_date = dates[0]

    # 3. IC 时序 + 权重 (仅 in factors_used)
    factor_weights_ts = None
    if cfg.use_ic_weighting:
        ic_ts = compute_ic_timeseries(
            factor_panel_used, panel_close, rebal_dates_idx, factors_used,
            horizon=cfg.ic_horizon_days,
        )
        factor_weights_df = compute_factor_weights(
            ic_ts,
            min_months=cfg.ic_min_months,
            smooth_window=cfg.ic_smooth_window,
        )
        factor_weights_ts = align_weights_with_rebal_dates(
            factor_weights_df, rebal_dates_idx, dates,
        )

    # 4. 模拟回测 (与 v6.1 相同, 但用正交 panel)
    nav = pd.Series(1.0, index=dates, dtype=float)
    prev_weights: dict[str, float] = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue

        is_rebal = date in rebal_set and i > 252

        if is_rebal:
            if factor_weights_ts is not None and date in factor_weights_ts.index:
                curr_fw = factor_weights_ts.loc[date]
            else:
                curr_fw = None

            try:
                composite = compute_composite_factor(
                    factor_panel_used, cfg.factor_cfg, date, curr_fw,
                )
                if composite.empty or len(composite) < cfg.top_n:
                    chosen = []
                else:
                    chosen = composite.sort_values(ascending=False).head(cfg.top_n).index.tolist()
            except Exception:
                chosen = []

            weights = {}
            if chosen:
                try:
                    weights = sub.weight(panel_close, chosen, date)
                except Exception:
                    weights = {}
            prev_weights = weights

        if prev_weights:
            daily_ret = 0.0
            for code, w in prev_weights.items():
                if code in panel_close.columns:
                    p_t = panel_close[code].iloc[i]
                    p_prev = panel_close[code].iloc[i - 1]
                    if pd.notna(p_t) and pd.notna(p_prev) and p_prev > 0:
                        daily_ret += w * (p_t / p_prev - 1.0)
            nav.iloc[i] = nav.iloc[i - 1] * (1 + daily_ret)

    return nav


__all__ = [
    "V6_2Config",
    "V6_2SubStrategy",
    "run_v6_2_backtest",
]
