# coding=utf-8
"""动量信号: 144 日涨幅排名 + 52 周高点距离备选.

输入约定:
    nav_df: pd.DataFrame
        index = DatetimeIndex (升序)
        columns = ETF codes
        values = 单位净值 (复权后)
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def rank_by_momentum(
    nav_df: pd.DataFrame,
    lookback: int = 144,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """在 as_of 时点按过去 lookback 日涨幅降序排序.

    返回: pd.Series, index=code, values=momentum (含负值)
    """
    if as_of is None:
        as_of = nav_df.index.max()
    sub = nav_df.loc[:as_of].iloc[-lookback - 1:]
    if len(sub) < lookback + 1:
        raise ValueError(
            f"数据不足: as_of={as_of}, 需要 {lookback+1} 行, 实际 {len(sub)} 行"
        )
    mom = sub.iloc[-1] / sub.iloc[0] - 1.0
    return mom.sort_values(ascending=False)


def rank_pctl(
    nav_df: pd.DataFrame,
    lookback: int = 144,
    as_of: pd.Timestamp | None = None,
    momentum_type: str = "price",
) -> pd.Series:
    """返回动量在截面上的分位数排名 (0=最弱, 1=最强).

    用于规则 4 的"排名跌出后 30% 分位"判断.

    Args:
        momentum_type: "price" | "slope_r2" | "hybrid"
            - "price": 纯涨幅 (默认)
            - "slope_r2": 斜率 × R²
            - "hybrid": 50/50 混合
    """
    if momentum_type == "slope_r2":
        score = slope_r2_score(nav_df, lookback, as_of)
    elif momentum_type == "hybrid":
        score = hybrid_momentum_score(nav_df, lookback, as_of)
    else:  # "price"
        score = rank_by_momentum(nav_df, lookback, as_of)
    return score.rank(method="average", pct=True)


def compute_momentum_score(
    nav_df: pd.DataFrame,
    lookback: int = 144,
    as_of: pd.Timestamp | None = None,
    momentum_type: str = "price",
    fused_weight: float = 0.5,
) -> pd.Series:
    """统一动量计算接口 (Stage 12A).

    Args:
        momentum_type: "price" | "slope_r2" | "hybrid"
    """
    if momentum_type == "slope_r2":
        return slope_r2_score(nav_df, lookback, as_of)
    elif momentum_type == "hybrid":
        return hybrid_momentum_score(nav_df, lookback, as_of, fused_weight)
    else:  # "price"
        return rank_by_momentum(nav_df, lookback, as_of)


def distance_to_52w_high(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    window: int = 252,
) -> pd.Series:
    """(价格 / 过去 window 日最高价) − 1, 越接近 0 越强.

    CICC 报告的另一个动量备选信号.
    """
    if as_of is None:
        as_of = nav_df.index.max()
    sub = nav_df.loc[:as_of].iloc[-window:]
    return sub.iloc[-1] / sub.max() - 1.0


def slope_r2_score(
    nav_df: pd.DataFrame,
    lookback: int = 144,
    as_of: pd.Timestamp | None = None,
    scale: float = 10000.0,
) -> pd.Series:
    """斜率 × R² 动量打分 (Stage 12A, 来自猫哥 5年10倍策略).

    对过去 lookback 日的归一化价格做线性回归:
        y = price / price[0]  (归一化)
        x = np.arange(N)
    然后 score = scale × slope × R²

    优点:
        - 斜率: 量化趋势方向和强度
        - R²: 量化趋势稳定性 (vs 噪声)
        - 同时识别"涨得快 + 涨得稳"的标的

    Args:
        nav_df: 价格面板
        lookback: 回归窗口
        as_of: 当前日期
        scale: 缩放系数 (与价格量级匹配, 默认 10000)

    Returns:
        pd.Series, index=code, values=score (越大越强)
    """
    from sklearn.linear_model import LinearRegression

    if as_of is None:
        as_of = nav_df.index.max()
    sub = nav_df.loc[:as_of].iloc[-lookback:]
    if len(sub) < 20:  # 至少 20 天数据
        return pd.Series(dtype=float, index=nav_df.columns)

    x = np.arange(len(sub)).reshape(-1, 1)
    scores = {}
    for code in sub.columns:
        col = sub[code].dropna()
        if len(col) < 20:
            scores[code] = 0.0
            continue
        y = (col / col.iloc[0]).values
        try:
            lr = LinearRegression().fit(x, y)
            slope = float(lr.coef_[0])
            r2 = float(lr.score(x, y))
            scores[code] = scale * slope * r2
        except Exception:
            scores[code] = 0.0
    return pd.Series(scores).sort_values(ascending=False)


def hybrid_momentum_score(
    nav_df: pd.DataFrame,
    lookback: int = 144,
    as_of: pd.Timestamp | None = None,
    fused_weight: float = 0.5,
) -> pd.Series:
    """混合动量信号 (Stage 12A): 价格动量 + 斜率×R².

    score = (1-w) × normalized_momentum + w × normalized_slope_r2
    两个信号分别归一化到 [-1, 1] 后线性融合.

    Args:
        nav_df: 价格面板
        lookback: 回归窗口
        as_of: 当前日期
        fused_weight: 斜率×R² 的权重 (默认 0.5)

    Returns:
        pd.Series, index=code, values=score (越大越强)
    """
    mom = rank_by_momentum(nav_df, lookback, as_of)
    mom_max = mom.abs().max()
    mom_norm = mom / mom_max if mom_max > 0 else mom

    slope = slope_r2_score(nav_df, lookback, as_of, scale=10000.0)
    slope_max = slope.abs().max()
    slope_norm = slope / slope_max if slope_max > 0 else slope

    return (1.0 - fused_weight) * mom_norm + fused_weight * slope_norm


def fused_signal(
    nav_df: pd.DataFrame,
    lookback: int = 144,
    as_of: pd.Timestamp | None = None,
    fused_weight: float = 0.4,
    window_52w: int = 252,
) -> pd.Series:
    """融合信号: (1-w) × 动量 + w × 距离52周新高.

    归一化: 两个信号分别除以各自绝对值最大值, 再线性融合.
    优点:
        - 动量捕获长期趋势
        - 52周新高捕获趋势强度 (避免假突破)
    默认 w=0.4 与 CICC 报告图表 4 一致.
    """
    mom = rank_by_momentum(nav_df, lookback, as_of)
    # 归一化到 [-1, 1] 范围
    mom_max = mom.abs().max()
    mom_norm = mom / mom_max if mom_max > 0 else mom

    dist = distance_to_52w_high(nav_df, as_of, window=window_52w)
    dist_max = dist.abs().max()
    dist_norm = dist / dist_max if dist_max > 0 else dist

    return (1.0 - fused_weight) * mom_norm + fused_weight * dist_norm


def realized_vol(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    window: int = 60,
) -> pd.Series:
    """年化已实现波动率 (对数收益, × √252).

    用于逆波动加权. 列为 code.
    """
    if as_of is None:
        as_of = nav_df.index.max()
    sub = nav_df.loc[:as_of].iloc[-window - 1:]
    log_ret = np.log(sub / sub.shift(1))
    # 逐列计算 std 而非 DataFrame.dropna (避免跨列 NaN 污染)
    def _col_std(col) -> float:
        # 防御: 若列名重复, col 可能是 DataFrame
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        valid = col.dropna()
        if len(valid) < 2:
            return float("nan")
        return float(valid.std() * np.sqrt(252))

    return pd.Series({c: _col_std(log_ret[c]) for c in log_ret.columns})


def below_ma(
    nav_df: pd.DataFrame,
    code: str,
    ma_window: int,
    as_of: pd.Timestamp | None = None,
) -> bool:
    """as_of 收盘是否跌破 ma_window 日均线 (CICC 止损触发条件 1)."""
    if as_of is None:
        as_of = nav_df.index.max()
    s = nav_df[code].loc[:as_of]
    # 防御: 列重复时 s 可能是 DataFrame
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if len(s) < ma_window:
        return False
    ma = s.iloc[-ma_window:].mean()
    last = s.iloc[-1]
    if isinstance(last, pd.Series):
        last = last.iloc[0]
    return bool(last < ma)


def pairwise_corr(
    nav_df: pd.DataFrame,
    codes: Iterable[str],
    as_of: pd.Timestamp | None = None,
    window: int = 60,
) -> pd.DataFrame:
    """过去 window 日 log 收益的相关系数矩阵."""
    if as_of is None:
        as_of = nav_df.index.max()
    sub = nav_df.loc[:as_of, list(codes)].iloc[-window - 1:]
    log_ret = np.log(sub / sub.shift(1)).dropna()
    return log_ret.corr()
