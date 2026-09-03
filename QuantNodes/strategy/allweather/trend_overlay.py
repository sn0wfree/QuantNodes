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
    risk_adj: bool = False,
) -> pd.DataFrame:
    """动量正部倾斜: w_i ∝ max(score_i, 0)^power, 归一化到均值 1.

    score = 动量 (risk_adj=False) 或 动量/波动 (risk_adj=True, 风险调整动量,
    偏好低波趋势资产 — 债券/黄金趋势期自动加权重).

    与 V1 (硬剔除) 的区别: 负动量资产保底 floor (不集中化), 正动量资产
    按强度加权; 全负行 → 全 floor.
    """
    mom = prices.pct_change(lookback, fill_method=None)
    if risk_adj:
        vol = prices.pct_change(fill_method=None).rolling(lookback,
                                                          min_periods=lookback).std()
        score = mom / vol.replace(0, np.nan) * np.sqrt(ETF_ANN)
    else:
        score = mom
    listed = score.notna()
    pos = score.clip(lower=0.0).fillna(0.0) ** power

    row_mean = pos.where(listed, 0.0).sum(axis=1) / listed.sum(axis=1).clip(lower=1)
    overlay = pos.div(row_mean.where(row_mean > 0), axis=0)
    # 无正动量 → floor; 未上市 → 0
    overlay = overlay.where(listed, 0.0)
    overlay = overlay.fillna(floor).clip(lower=floor * 0.0)
    # 负动量资产给 floor (软保底而非剔除)
    overlay = overlay.where(~(listed & (score < 0)), floor)
    return overlay.clip(lower=0.0)


def _apply_state(trigger: pd.DataFrame, release: pd.DataFrame) -> pd.DataFrame:
    """逐列状态机 (向量化): trigger=1 → 空仓(0); release=1 → 恢复(1).

    状态 = ffill(最近一次 trigger/release 信号), 初始为 1 (持仓).
    同日 trigger 与 release 并发 → trigger 优先 (保守).
    """
    t = trigger.fillna(False).to_numpy()
    r = release.fillna(False).to_numpy()
    sig = np.where(t, 0.0, np.where(r, 1.0, np.nan))
    state = pd.DataFrame(sig, index=trigger.index, columns=trigger.columns)
    state = state.ffill().fillna(1.0)
    return state


def trig_a(prices: pd.DataFrame, ma_window: int = 20, mom_window: int = 20,
           slope_lag: int = 5) -> pd.DataFrame:
    """触发器 A「趋势转弱」(锚定黄金 3-16 episode, 波动未扩张时触发).

    触发: close < MA{ma} 且 MA 斜率 < 0 (MA 较 slope_lag 日前下降) 且 {mom}d 动量 < 0
    解除: close > MA 且 {mom}d 动量 ≥ 0
    输出: 1=持仓, 0=空仓 (资产级)
    """
    ma = prices.rolling(ma_window, min_periods=ma_window).mean()
    mom = prices.pct_change(mom_window, fill_method=None)
    trigger = (prices < ma) & (ma.diff(slope_lag) < 0) & (mom < 0)
    release = (prices > ma) & (mom >= 0)
    return _apply_state(trigger, release)


def trig_b(prices: pd.DataFrame, vol_short: int = 20, vol_long: int = 120,
           dd_window: int = 5, dd_thr: float = -0.03,
           trig_thr: float = 1.4, rel_thr: float = 1.2) -> pd.DataFrame:
    """触发器 B「波动扩张」(锚定权益 7/1 episode, 动量未负时触发).

    触发: (vol{short}/vol{long}) > {trig} 且 {dd}d 收益 < {dd_thr}
    解除: 波动比 < {rel} (滞后带 1.2-1.4 之间维持原状态)
    输出: 1=持仓, 0=空仓 (资产级)
    """
    ret = prices.pct_change(fill_method=None)
    v_short = ret.rolling(vol_short, min_periods=vol_short).std()
    v_long = ret.rolling(vol_long, min_periods=vol_long).std()
    ratio = v_short / v_long.replace(0, np.nan)
    ret_dd = prices.pct_change(dd_window, fill_method=None)
    trigger = (ratio > trig_thr) & (ret_dd < dd_thr)
    release = ratio < rel_thr
    return _apply_state(trigger, release)


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
    if mode == "trigA":
        return trig_a(prices, **kwargs)
    if mode == "trigB":
        return trig_b(prices, **kwargs)
    if mode == "none":
        return pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    raise ValueError(f"未知趋势模式: {mode}, 可选 v1/v3/v4/tilt/trigA/trigB/none")
