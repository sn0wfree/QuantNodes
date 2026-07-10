# coding=utf-8
"""v6.1 量价族 + IC 加权 (Stage 27 v6.1).

v6.1 = v5.1.1 量价族 + IC-IR 加权 (替代 11 等权).

设计动机:
- v5.1.1 用 11 因子等权, 但 IC 诊断显示仅 5-6 个因子 OOS 有效
- 用 expanding window IC 计算 → 自动剔除失效因子 + 大权重给稳定 alpha 源
- 防 look-ahead: t 期权重基于截至 t-1 的历史 IC (shift(1))

与 v5.1.1 的区别:
- 选股: 复用 v5.compute_composite_factor, 但 weights 参数化
- 加权: 复用 v5_1.inverse_vol_weights_v5_1
- 调仓: 月度, T+1 lag (与 v5.1.1 一致)
- top_n: 默认 5 (与 v5.1.1 一致)

无风控层 (VT/TF/Cost): v6.1 专注 IC 加权; 与 v6 风控融合待 v6.5 阶段.
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
    cross_section_zscore,
)
from ..v5_1.industry_rotation_v5_1 import inverse_vol_weights_v5_1

from .factor_weighting import (
    compute_ic_timeseries,
    compute_factor_weights,
    align_weights_with_rebal_dates,
    DEFAULT_HORIZON_DAYS,
    MIN_MONTHS_FOR_IC,
    DEFAULT_SMOOTH_WINDOW,
)


@dataclass
class V6_1Config(SubStrategyConfig):
    """v6.1 配置: v5.1.1 选股 + IC 加权 + v5.1.1 加权.

    关键字段:
    - factor_weights: None = 等权 (baseline), 或 dict = 静态权重
    - use_ic_weighting: True = 自动从 IC 计算权重 (推荐)
    - ic_min_months: IC 计算最少历史月数 (默认 24 防 look-ahead)
    - ic_smooth_window: IR 平滑窗口 (默认 6 月减抖)
    """
    name: str = "industry_rotation_v6_1"

    # 因子引擎 (与 v5.1.1 共享)
    factor_cfg: FactorEngineConfig = field(default_factory=FactorEngineConfig)

    # 选股层 (来自 v5.1.1)
    top_n: int = 5

    # 复合因子权重 (None = 等权, dict = 静态权重覆盖)
    factor_weights: dict[str, float] | None = None

    # IC 加权开关
    use_ic_weighting: bool = True

    # IC 评估窗口
    ic_horizon_days: int = DEFAULT_HORIZON_DAYS
    ic_min_months: int = MIN_MONTHS_FOR_IC
    ic_smooth_window: int = DEFAULT_SMOOTH_WINDOW

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


class V6_1SubStrategy(IndustryRotationV5SubStrategy):
    """v6.1 子策略: v5 选股 (含 IC 加权) + v5.1.1 逆波动加权.

    与 v5.1.1 区别:
    - select: 接受外部因子权重 (在 run_v6_1_backtest 中按调仓日注入)
    - weight: 复用 v5.1.1 逆波动
    """
    config: V6_1Config

    def __init__(self, config: V6_1Config):
        super().__init__(config)
        self.config: V6_1Config = config

    def weight(
        self,
        nav_df: pd.DataFrame,
        codes: Sequence[str],
        as_of: pd.Timestamp,
    ) -> dict[str, float]:
        """逆波动率加权 (v5.1.1 复用)."""
        if not codes:
            return {}
        cfg = self.config
        weights = inverse_vol_weights_v5_1(
            nav_df, codes, as_of,
            cfg.vol_window, cfg.vol_floor, cfg.rebal_lag,
        )
        weights = self._apply_max_weight(weights, cfg.max_weight)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights


def select_v6_1(
    panel_close: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    as_of: pd.Timestamp,
    cfg: V6_1Config,
    factor_weights: dict[str, float] | None = None,
    sub: V6_1SubStrategy | None = None,
) -> list[str]:
    """v6.1 选股: 计算复合因子 → 排序 → Top-N.

    Args:
        panel_close: 收盘价 (用于 warmup 检查)
        panel_ohlcv: OHLCV 面板 (用于因子计算)
        as_of: 调仓日
        cfg: v6.1 config
        factor_weights: 当期因子权重 (None = 等权)
        sub: 预计算的 SubStrategy (避免重复因子预算)

    Returns:
        list of ETF code (Top-N)
    """
    if sub is None:
        sub = V6_1SubStrategy(cfg)
        sub._factor_panel = None

    # 复用 v5._init_factor_panel 懒加载
    if sub._factor_panel is None:
        from ..v5.industry_factors import compute_all_factors_panel
        sub._factor_panel = compute_all_factors_panel(panel_ohlcv, cfg.factor_cfg)
        sub._last_init_date = panel_ohlcv.index[0]

    composite = compute_composite_factor(
        sub._factor_panel, cfg.factor_cfg, as_of, factor_weights,
    )
    if composite.empty or len(composite) < cfg.top_n:
        return []

    selected = composite.sort_values(ascending=False).head(cfg.top_n).index.tolist()
    return [str(c) for c in selected]


def run_v6_1_backtest(
    panel_close: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    cfg: V6_1Config | None = None,
    rebalance_dates: Sequence[pd.Timestamp] | None = None,
) -> pd.Series:
    """v6.1 回测: 复用 v5 选股 + IC 加权 + 逆波动加权.

    Args:
        panel_close: 收盘价面板
        panel_ohlcv: OHLCV 面板
        cfg: v6.1 config (None = 默认)
        rebalance_dates: 调仓日期 (None = 月末)

    Returns:
        NAV Series
    """
    if cfg is None:
        cfg = V6_1Config()

    dates = panel_close.index

    # 1. 调仓日期
    if rebalance_dates is None:
        rebal_dates_idx = dates.to_series().resample("ME").last().index
    else:
        rebal_dates_idx = pd.DatetimeIndex(rebalance_dates)
    rebal_set = set(d for d in rebal_dates_idx if d in dates)

    # 2. 预算 11 因子 (一次性)
    from ..v5.industry_factors import compute_all_factors_panel
    factor_panel = compute_all_factors_panel(panel_ohlcv, cfg.factor_cfg)
    factors = list(cfg.factor_cfg.name_map.keys())

    sub = V6_1SubStrategy(cfg)
    sub._factor_panel = factor_panel
    if len(dates) > 0:
        sub._last_init_date = dates[0]

    # 3. 预算 IC 时序 + 因子权重 (若启用)
    factor_weights_ts = None
    if cfg.use_ic_weighting:
        # 计算 IC 时序 (仅 rebal_dates)
        ic_ts = compute_ic_timeseries(
            factor_panel, panel_close, rebal_dates_idx, factors,
            horizon=cfg.ic_horizon_days,
        )
        # 转权重 (expanding window + 平滑)
        factor_weights_df = compute_factor_weights(
            ic_ts,
            min_months=cfg.ic_min_months,
            smooth_window=cfg.ic_smooth_window,
        )
        # 对齐到 rebal_dates
        factor_weights_ts = align_weights_with_rebal_dates(
            factor_weights_df, rebal_dates_idx, dates,
        )

    # 4. 模拟回测
    nav = pd.Series(1.0, index=dates, dtype=float)
    prev_weights: dict[str, float] = {}

    for i, date in enumerate(dates):
        if i == 0:
            continue

        is_rebal = date in rebal_set and i > 252

        if is_rebal:
            # 当期因子权重
            if factor_weights_ts is not None and date in factor_weights_ts.index:
                curr_fw = factor_weights_ts.loc[date]
            else:
                curr_fw = None  # 等权

            try:
                chosen = select_v6_1(
                    panel_close, panel_ohlcv, date, cfg,
                    factor_weights=curr_fw, sub=sub,
                )
            except Exception:
                chosen = []

            weights = {}
            if chosen:
                try:
                    weights = sub.weight(panel_close, chosen, date)
                except Exception:
                    weights = {}
            prev_weights = weights

        # 累积 NAV (无风控: 直接按权重算日收益, 无 cost)
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
    "V6_1Config",
    "V6_1SubStrategy",
    "select_v6_1",
    "run_v6_1_backtest",
]
