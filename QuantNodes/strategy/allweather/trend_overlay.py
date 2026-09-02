# coding=utf-8
"""AllWeather — 自适应负向趋势 overlay (PLAN Step 3 / §1.3).

文章"自适应负向趋势因子"的两种解释 (其余变体见 PLAN §1.3):
    - V1 减仓型 (主线): 资产 lookback 日动量 < 0 → 权重 × coef (默认 0, 即剔除)
    - V4 连续型 (国君系列之九): 风险预算 ∝ max(滚动夏普, 0)²,
      归一化到均值 1 → 只在资产间重分配, 不改变组合总敞口

无前视约定: 所有 rolling 统计仅用 t 及以前数据; NaN (未上市) 资产
overlay 输出 0, 避免 NaN × 0 = NaN 污染下游权重.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ETF_ANN = 252.0


def v1_overlay(prices: pd.DataFrame, lookback: int = 60, coef: float = 0.0) -> pd.DataFrame:
    """V1 减仓型: lookback 日动量 < 0 → coef, 否则 1; NaN 动量 → 0 (不持仓)."""
    mom = prices.pct_change(lookback, fill_method=None)
    overlay = pd.DataFrame(
        np.where(mom.isna(), 0.0, np.where(mom < 0, coef, 1.0)),
        index=prices.index, columns=prices.columns,
    )
    return overlay


def v4_overlay(
    prices: pd.DataFrame,
    lookback: int = 252,
    vol_window: int | None = None,
    floor: float = 0.0,
) -> pd.DataFrame:
    """V4 连续型: 预算 ∝ max(滚动夏普, 0)², 归一化到均值 1.

    Args:
        lookback: 夏普计算的收益窗口 (日)
        vol_window: 波动窗口, 默认 = lookback (国君做法: 同窗口收益/波动)
        floor: 夏普 ≤ 0 资产的保底系数

    Returns:
        overlay (T x N), 均值 ≈ 1 (仅对有数据资产), 未上市资产 = 0
    """
    if vol_window is None:
        vol_window = lookback
    ret = prices.pct_change(fill_method=None)
    mean_ret = ret.rolling(lookback, min_periods=lookback).mean()
    vol = ret.rolling(vol_window, min_periods=vol_window).std()
    sharpe = mean_ret / vol.replace(0, np.nan) * np.sqrt(ETF_ANN)

    listed = sharpe.notna()
    budget = sharpe.clip(lower=0.0).fillna(0.0) ** 2

    # 归一化: listed 资产的 budget 均值 = 1 (只重分配, 不改组合总敞口)
    row_mean = budget.where(listed, 0.0).sum(axis=1) / listed.sum(axis=1).clip(lower=1)
    overlay = budget.div(row_mean.where(row_mean > 0), axis=0)

    # 未上市 → 0; 全负夏普行 (row_mean=0) → floor
    overlay = overlay.where(listed, 0.0)
    overlay = overlay.fillna(floor)
    return overlay.clip(lower=0.0)


def v3_overlay(
    prices: pd.DataFrame,
    lookback: int = 60,
    coef: float = 0.0,
) -> pd.DataFrame:
    """V3 组合级过滤 (CTA v7 momentum filter 同构, 全组合同一系数).

    组合等权动量 (各资产 lookback 日动量的均值) < 0 → 全部权重 × coef, 否则 × 1.
    与 V1 的区别: 不做资产级剔除 (避免集中化), 只在组合层面降总敞口.
    """
    mom = prices.pct_change(lookback, fill_method=None)
    port_mom = mom.mean(axis=1)  # 组合动量 (NaN 资产自动跳过)
    scale = pd.Series(np.where(port_mom.isna(), 1.0,
                               np.where(port_mom < 0, coef, 1.0)),
                      index=prices.index)
    return pd.DataFrame(np.tile(scale.values[:, None], (1, prices.shape[1])),
                        index=prices.index, columns=prices.columns)


def momentum_tilt(
    prices: pd.DataFrame,
    lookback: int = 60,
    floor: float = 0.25,
    power: float = 1.0,
) -> pd.DataFrame:
    """动量正部倾斜: w_i ∝ max(mom_i, 0)^power, 归一化到均值 1.

    与 V1 (硬剔除) 的区别: 负动量资产保底 floor (不集中化), 正动量资产
    按动量强度加权; 全负行 → 全 floor (等权退化为现金减仓).
    """
    mom = prices.pct_change(lookback, fill_method=None)
    pos = mom.clip(lower=0.0) ** power
    listed = mom.notna()
    pos = pos.fillna(0.0)

    row_mean = pos.where(listed, 0.0).sum(axis=1) / listed.sum(axis=1).clip(lower=1)
    overlay = pos.div(row_mean.where(row_mean > 0), axis=0)
    # 无正动量 → floor; 未上市 → 0
    overlay = overlay.where(listed, 0.0)
    overlay = overlay.fillna(floor).clip(lower=floor * 0.0)
    # 负动量资产给 floor (软保底而非剔除)
    overlay = overlay.where(~(listed & (mom < 0)), floor)
    return overlay.clip(lower=0.0)


def combined(prices: pd.DataFrame, mode: str = "v1", **kwargs) -> pd.DataFrame:
    """趋势 overlay 分发器: v1 / v3 / v4 / tilt / none (全 1)."""
    if mode == "v1":
        return v1_overlay(prices, **kwargs)
    if mode == "v3":
        return v3_overlay(prices, **kwargs)
    if mode == "v4":
        return v4_overlay(prices, **kwargs)
    if mode == "tilt":
        return momentum_tilt(prices, **kwargs)
    if mode == "none":
        return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    raise ValueError(f"未知趋势模式: {mode}, 可选 v1/v3/v4/tilt/none")
