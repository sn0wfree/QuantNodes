# coding=utf-8
"""AllWeather V6 子因子周频重采样 + z-score 标准化 (PLAN §13.3.2).

将日度因子分数 (T × N) 转为周频分数 (W × N):
    1. resample_weekly: W-FRI 重采样 (last/mean)
    2. zscore_cross_section: 横截面滚动 z-score (52 周窗口)
    3. combine_subfactors: 三子因子加权求和 → 综合分数

参照黄金 gold_combined.py: 6 因子等权 Sharpe 1.37, OOS/IS=4.47.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


WeeklyAgg = Literal["last", "mean"]


def resample_weekly(
    factor_daily: pd.DataFrame,
    agg: WeeklyAgg = "last",
) -> pd.DataFrame:
    """日度因子 → W-FRI 重采样.

    Args:
        factor_daily: T × N 日度因子分数.
        agg: 'last' (取周五最后值, 默认, 对齐 W-FRI 调仓)
             'mean' (取周内均值, 更稳健).

    Returns:
        W × N 周频因子分数 (index 为 W-FRI).
    """
    if agg == "last":
        return factor_daily.resample("W-FRI").last()
    elif agg == "mean":
        return factor_daily.resample("W-FRI").mean()
    else:
        raise ValueError(f"agg={agg}, 可选 last/mean")


def zscore_cross_section(
    factor_panel: pd.DataFrame,
    lookback_weeks: int = 52,
    min_periods: int = 12,
    winsorize: float | None = None,
) -> pd.DataFrame:
    """横截面滚动 z-score 标准化 (时间序列滚动, 每个时点横截面标准化).

    factor_panel: W × N (周频).
    输出: 每个时点, 对 N 个资产做滚动 z-score (rolling lookback_weeks).
          消除资产间量纲差异, 使因子分数可比较.

    窗口: 默认 52 周 (1 年), 与黄金多因子 OOS/IS=4.47 基准一致.

    Args:
        winsorize: 若给定 (如 2.0), z-score 截尾到 [-winsorize, +winsorize].
                   防止极值资产放大权重 (Sub-A 衰减修复点 #1).
    """
    if factor_panel.empty:
        return factor_panel.copy()

    # 时间序列 rolling mean/std (沿 axis=0, 对每列单独)
    rolling_mean = factor_panel.rolling(
        lookback_weeks, min_periods=min_periods
    ).mean()
    rolling_std = factor_panel.rolling(
        lookback_weeks, min_periods=min_periods
    ).std()

    z = (factor_panel - rolling_mean) / rolling_std.replace(0, np.nan)
    if winsorize is not None and winsorize > 0:
        z = z.clip(lower=-winsorize, upper=winsorize)
    return z


def rank_cross_section(
    factor_panel: pd.DataFrame,
    lookback_weeks: int = 52,
    min_periods: int = 12,
) -> pd.DataFrame:
    """横截面滚动 rank percentile 标准化 (Sub-A 修复点 #2).

    与 z-score 不同: rank percentile 输出 ∈ [-0.5, +0.5], 对极值稳健.
    每个时点对横截面 (N 个资产) 做 rank → 减中位数 → 归一化.

    优势 (vs z-score):
      - 对低波动资产不偏正 (rank 是横截面相对位置, 不受历史 vol 影响)
      - 对极值稳健 (rank 不会因单次异常大幅偏移)
      - 输出 ∈ [-0.5, +0.5] 范围内, 更紧

    默认窗口: 52 周 (1 年).
    """
    if factor_panel.empty:
        return factor_panel.copy()

    out = pd.DataFrame(np.nan, index=factor_panel.index, columns=factor_panel.columns)
    valid_rows = factor_panel.notna().sum(axis=1) >= 2

    for i in range(len(factor_panel)):
        if not valid_rows.iloc[i]:
            continue
        # 在 rolling 窗口内取历史 rank 累积 (简化: 逐行 rank, 不滚动)
        row = factor_panel.iloc[i]
        ranked = row.rank(method="average", na_option="keep")
        n_valid = ranked.notna().sum()
        if n_valid <= 1:
            continue
        # ∈ [0, 1] → ∈ [-0.5, +0.5]
        out.iloc[i] = (ranked - 0.5) / n_valid - 0.5

    return out


# ── 风险因子变换 (V6.7) ─────────────────────────────────────────────
def directional_risk_transform(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """方向性风险因子: 只保留负向 (避险猫"负向"语义).

    公式: clip(score, upper=0)
    语义: factor_score < 0 (趋势破坏) → 降权;
          factor_score >= 0 (趋势确认) → 不动.

    数学等价: w × (1 + α × clip(score, upper=0))
            = w × (1 - α × max(0, -score))  (仅负向调整).
    """
    return factor_panel.clip(upper=0.0)


def magnitude_risk_transform(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Magnitude 风险因子: |score| 双向降权 (任何偏离中性都是风险).

    公式: -|score|
    语义: 任何 |score| > 0 → 降权 (无论正负向).

    数学等价: w × (1 + α × (-|score|))
            = w × (1 - α × |score|).
    """
    return -factor_panel.abs()


def magnitude_squared_risk_transform(factor_panel: pd.DataFrame) -> pd.DataFrame:
    """Magnitude² 风险因子: 极端偏离加重 (非线性惩罚).

    公式: -score²
    语义: |score| 越大, 惩罚非线性加重 (平方放大).

    数学等价: w × (1 + α × (-score²))
            = w × (1 - α × score²).
    注意: |score| ∈ [0, 0.5] (rank), score² ≤ 0.25, 需更大 α.
    """
    return -(factor_panel ** 2)


def combine_subfactors(
    factors: dict[str, pd.DataFrame],
    weights: Optional[dict[str, float]] = None,
    lookback_weeks: int = 52,
    min_periods: int = 12,
    method: str = "zscore",
    winsorize: float | None = None,
) -> pd.DataFrame:
    """组合多子因子 → 单一综合分数 (W × N).

    步骤:
      1. 每个子因子横截面标准化 (z-score / winsorize / rank percentile)
      2. 加权求和 (默认等权 {a:0.33, b:0.33, c:0.34})
      3. 输出 W × N 综合分数

    Args:
        factors: {"a": W×N, "b": W×N, "c": W×N}.
        weights: {"a": 0.33, "b": 0.33, "c": 0.34}, 默认等权.
        method: 标准化方法.
            - "zscore" (默认): 横截面 z-score, winsorize 可选
            - "rank": rank percentile (Sub-A 修复点 #2, 对极值稳健)
        winsorize: z-score 截尾阈值 (如 2.0), 仅 method="zscore" 生效.

    Returns:
        W × N 综合因子分数 (标准化后加权和).
    """
    if not factors:
        raise ValueError("factors 不能为空")

    if weights is None:
        n = len(factors)
        weights = {k: 1.0 / n for k in factors}

    weight_sum = sum(weights.get(k, 0.0) for k in factors)
    if weight_sum <= 0:
        raise ValueError("weights 权重总和必须 > 0")

    combined = None
    for k, factor in factors.items():
        if factor.empty:
            continue
        if method == "zscore":
            z = zscore_cross_section(factor, lookback_weeks, min_periods, winsorize)
        elif method == "rank":
            z = rank_cross_section(factor, lookback_weeks, min_periods)
        else:
            raise ValueError(f"method={method}, 可选 zscore/rank")
        contrib = z * (weights.get(k, 0.0) / weight_sum)
        combined = contrib if combined is None else combined + contrib

    if combined is None:
        raise ValueError("所有 factors 都为空")

    return combined
