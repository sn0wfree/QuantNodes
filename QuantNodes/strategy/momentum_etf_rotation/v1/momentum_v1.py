# coding=utf-8
"""v1 动量信号: 纯涨幅排名 (原始CICC复现).

v1 是 Stage 8 baseline, 不含 v2 增强:
  - 无 hybrid (momentum_type 总是 "price")
  - 无 slope_r2_score, hybrid_momentum_score
  - 无 VolTargeting, CostModel
  - 无 ConcentrationCaps

如需这些功能, 请使用 v2.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def rank_by_momentum_v1(
    nav_df: pd.DataFrame,
    lookback: int = 144,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """v1: 纯涨幅动量排名 (Stage 8 baseline).

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


def rank_pctl_v1(
    nav_df: pd.DataFrame,
    lookback: int = 144,
    as_of: pd.Timestamp | None = None,
) -> pd.Series:
    """v1: 动量分位数排名."""
    mom = rank_by_momentum_v1(nav_df, lookback, as_of)
    return mom.rank(method="average", pct=True)


def distance_to_52w_high_v1(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    window: int = 252,
) -> pd.Series:
    """v1: 距离52周新高 (CICC 备选信号)."""
    if as_of is None:
        as_of = nav_df.index.max()
    sub = nav_df.loc[:as_of].iloc[-window:]
    return sub.iloc[-1] / sub.max() - 1.0


def realized_vol_v1(
    nav_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    window: int = 60,
) -> pd.Series:
    """v1: 年化已实现波动率 (对数收益, × √252)."""
    if as_of is None:
        as_of = nav_df.index.max()
    sub = nav_df.loc[:as_of].iloc[-window - 1:]
    log_ret = np.log(sub / sub.shift(1))

    def _col_std(col: pd.Series) -> float:
        valid = col.dropna()
        if len(valid) < 2:
            return float("nan")
        return float(valid.std() * np.sqrt(252))

    return pd.Series({c: _col_std(log_ret[c]) for c in log_ret.columns})


def below_ma_v1(
    nav_df: pd.DataFrame,
    code: str,
    ma_window: int,
    as_of: pd.Timestamp | None = None,
) -> bool:
    """v1: 价格是否跌破 ma_window 日均线."""
    if as_of is None:
        as_of = nav_df.index.max()
    s = nav_df[code].loc[:as_of]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    if len(s) < ma_window:
        return False
    ma = s.iloc[-ma_window:].mean()
    last = s.iloc[-1]
    if isinstance(last, pd.Series):
        last = last.iloc[0]
    return bool(last < ma)


def pairwise_corr_v1(
    nav_df: pd.DataFrame,
    codes: Iterable,
    as_of: pd.Timestamp | None = None,
    window: int = 60,
) -> pd.DataFrame:
    """v1: 过去 window 日 log 收益的相关系数矩阵."""
    if as_of is None:
        as_of = nav_df.index.max()
    sub = nav_df.loc[:as_of, list(codes)].iloc[-window - 1:]
    log_ret = np.log(sub / sub.shift(1)).dropna()
    return log_ret.corr()


__all__ = [
    "rank_by_momentum_v1",
    "rank_pctl_v1",
    "distance_to_52w_high_v1",
    "realized_vol_v1",
    "below_ma_v1",
    "pairwise_corr_v1",
]