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
) -> pd.Series:
    """返回动量在截面上的分位数排名 (0=最弱, 1=最强).

    用于规则 4 的"排名跌出后 30% 分位"判断.
    """
    mom = rank_by_momentum(nav_df, lookback, as_of)
    return mom.rank(method="average", pct=True)


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
    def _col_std(col: pd.Series) -> float:
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
    if len(s) < ma_window:
        return False
    ma = s.iloc[-ma_window:].mean()
    return bool(s.iloc[-1] < ma)


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
