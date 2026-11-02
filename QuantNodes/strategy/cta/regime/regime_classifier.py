# coding=utf-8
"""市场状态识别 (趋势 / 震荡).

文章 §"市场状态识别: 从隐含概率到可执行置信区间" 给出的判定规则:
    趋势主导 ← all(连续 10 日):
        TSI_CI 完全位于零轴上方或下方
        OR |TSI| 点估计的历史分位数 > 50%
        AND ρ > 0
        AND (ρ t-stat > 1.5 OR σ 历史分位数 < 95%)
    否则 → 震荡状态.

输出:
    - state ∈ {'trend_up', 'trend_down', 'range', 'unknown'}
    - 'trend_up' / 'trend_down' 由 TSI 符号决定
    - 'unknown' 当样本不足

研报: 华泰金工《时序CTA方法论综述》(2026-06-12).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


STATE_UNKNOWN = "unknown"
STATE_RANGE = "range"
STATE_TREND_UP = "trend_up"
STATE_TREND_DOWN = "trend_down"


def _t_stat_of_rho(rho: pd.Series, window: int | None = None) -> pd.Series:
    """ρ 的 t 统计量: t = ρ · √(n - 2) / √(1 - ρ²), 滚动计算.

    NaN (ρ² ≥ 1 或样本 < 3) 保留为 NaN.
    """
    if window is None:
        # 简单方案: 对每行用同一 rho 的"近邻"序列计算, 但通常 rho 已 rolling 得到, 此处不可行.
        # 实际使用方应传入 rolling t-stat 序列. 这里退化为 ρ 的简单绝对值标度.
        return rho.abs()
    # 标准 t-stat 公式 (n-2 自由度, 近似)
    n = window
    denom = np.sqrt(np.clip(1 - rho ** 2, 0, None))
    return rho * np.sqrt(max(n - 2, 1)) / denom.replace(0, np.nan)


def classify_regime(
    mu: pd.DataFrame,
    rho: pd.DataFrame,
    sigma: pd.DataFrame,
    tsi_lo: pd.DataFrame,
    tsi_hi: pd.DataFrame,
    confirm_window: int = 10,
    rho_window: int = 60,
    rho_required: bool = True,
) -> pd.DataFrame:
    """市场状态分类 (逐列独立).

    Args:
        mu: (T, N) 长期趋势估计
        rho: (T, N) AR(1) 系数
        sigma: (T, N) 背景噪声估计
        tsi_lo: (T, N) TSI 置信区间下界
        tsi_hi: (T, N) TSI 置信区间上界
        confirm_window: 连续满足的天数 (默认 10, 文章推荐)
        rho_window: 计算 ρ t-stat 的窗口长度 (默认 60)
        rho_required: ρ>0 是否为趋势必要条件 (默认 True, 文章原文).
            False 时仅要求 μ 显著 (CI 一侧) + σ 非极端:
            适用于"阶梯式牛市" (μ 强但日度自相关为负, 如 AU 2022-2026
            +164% 行情中 ρ>0 仅 2.8% 时间 → 原规则全程误判 range,
            趋势信号完全缺失).

    Returns:
        pd.DataFrame (T, N), dtype=str, 值为 STATE_* 常量.
    """
    assert mu.shape == rho.shape == sigma.shape == tsi_lo.shape == tsi_hi.shape
    states = pd.DataFrame(STATE_UNKNOWN, index=mu.index, columns=mu.columns)

    # 1. CI 完全位于零轴一侧: lo > 0 或 hi < 0
    ci_above = tsi_lo > 0
    ci_below = tsi_hi < 0
    ci_one_side = ci_above | ci_below
    # 2. |TSI| 历史分位数 > 50% (使用 |mu|/sigma 标度, 等价于 TSI 排序)
    tsi_score = (mu.abs() / sigma.replace(0, np.nan))
    pct_rank = tsi_score.rank(pct=True)
    above_median = pct_rank > 0.5
    # 3. ρ > 0
    rho_pos = rho > 0
    # 4. ρ t-stat > 1.5 (近似: |ρ| · √(n-2) / √(1-ρ²) > 1.5)
    rho_tstat = _t_stat_of_rho(rho, rho_window)
    rho_significant = rho_tstat > 1.5
    # 5. σ 历史分位数 < 95% (σ 不能在极端波动尾部)
    sigma_pct = sigma.rank(pct=True)
    sigma_not_extreme = sigma_pct < 0.95

    if rho_required:
        raw_trend = (
            (ci_one_side | above_median) & rho_pos & (rho_significant | sigma_not_extreme)
        )
    else:
        # μ 显著 (CI 一侧) 即判趋势, 方向由 CI 符号决定; ρ 仅作方向增强
        raw_trend = (ci_one_side | above_median) & sigma_not_extreme

    # 连续 confirm_window 日满足 → 趋势成立
    rolling_trend = (
        raw_trend.rolling(confirm_window, min_periods=confirm_window).sum()
        == confirm_window
    )
    # TSI 符号决定多空方向: ci_above → 多, ci_below → 空
    direction = np.where(ci_above, 1, np.where(ci_below, -1, 0))
    direction = pd.DataFrame(direction, index=mu.index, columns=mu.columns)

    states = states.mask(rolling_trend & (direction > 0), STATE_TREND_UP)
    states = states.mask(rolling_trend & (direction < 0), STATE_TREND_DOWN)
    # 剩余有估计但未触发趋势 → 标记为 RANGE
    has_estimate = mu.notna() & sigma.notna()
    not_trend = ~states.isin([STATE_TREND_UP, STATE_TREND_DOWN])
    states = states.mask(has_estimate & not_trend, STATE_RANGE)

    return states


def classify_regime_from_estimator(
    params: dict[str, pd.DataFrame],
    tsi_panel: pd.DataFrame,
    confirm_window: int = 10,
    rho_window: int = 60,
    rho_required: bool = True,
) -> pd.DataFrame:
    """从 ARGarchEstimator 与 compute_tsi_panel 的输出调用 classify_regime.

    Args:
        params: ARGarchEstimator.estimate() 输出, 含 mu/rho/sigma.
        tsi_panel: compute_tsi_panel() 输出, MultiIndex 列 (product, tsi|lo|hi).
        rho_required: ρ>0 是否为趋势必要条件 (见 classify_regime).
    """
    mu, rho, sigma = params["mu"], params["rho"], params["sigma"]
    tsi_lo = tsi_panel.xs("tsi_lo", axis=1, level=1)
    tsi_hi = tsi_panel.xs("tsi_hi", axis=1, level=1)
    return classify_regime(
        mu, rho, sigma, tsi_lo, tsi_hi, confirm_window, rho_window, rho_required,
    )
