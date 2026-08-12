# coding=utf-8
"""CTA 因子基类.

Factor 是 CTA 因子的统一接口:
    - 输入: 主力连续面板 (prices), 日期
    - 输出: 因子值 (float 或 pandas Series)

实现风格对齐 momentum_etf_rotation 的因子面板约定.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Factor(ABC):
    """CTA 因子基类.

    子类实现 compute(), 返回因子值序列 (与 prices 对齐).
    """

    name: str = "base"

    @abstractmethod
    def compute(
        self,
        prices: pd.DataFrame,
        **kwargs,
    ) -> pd.Series:
        """计算因子值.

        Args:
            prices: (T, N) 价格面板, index=DatetimeIndex, columns=产品代码

        Returns:
            pd.Series, index 与 prices.index 对齐, values=因子值
        """
        raise NotImplementedError


class PanelFactor(Factor):
    """基于价格面板的因子.

    用于跨品种计算 (如横截面动量/排名), compute() 返回全面板因子值.
    """

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.Series:
        raise NotImplementedError


def time_series_factor(
    prices: pd.DataFrame,
    product: str,
    func,
) -> pd.Series:
    """单品种时序因子便捷函数."""
    col = prices[product] if product in prices.columns else prices.iloc[:, 0]
    return func(col)
