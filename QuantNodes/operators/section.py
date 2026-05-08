# coding=utf-8
"""
截面算子（代理层）

基于 factor_functions/section_ops.py 的实现，提供统一的类接口。

Available Operators:
    - rank: 截面排名
    - zscore: Z-score 标准化
    - winsorize: 去极值
    - neutralize: 行业中性
    - neutralize_market: 市场中性
    - scale: 归一化
    - percentile: 百分位排名
    - ic: 信息系数
    - rank_ic: 秩信息系数
    - group_norm: 分组标准化
    - group_winsorize: 分组去极值

Usage:
    >>> sec.rank(pl.col("factor"))
    >>> sec.zscore(pl.col("factor"))
    >>> sec.winsorize(pl.col("factor"), 0.01, 0.01)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import polars as pl
from polars import Expr

if TYPE_CHECKING:
    pass

from QuantNodes.factor_node.factor_functions.section_ops import (
    rank as _rank,
    zscore as _zscore,
    winsorize as _winsorize,
    neutralize as _neutralize,
    neutralize_market as _neutralize_market,
    scale as _scale,
    ic as _ic,
    rank_ic as _rank_ic,
    group_norm as _group_norm,
    group_winsorize as _group_winsorize,
)


class SectionOperators:
    """截面算子（代理层）"""

    @staticmethod
    def rank(expr: Union[Expr, str], method: str = "dense") -> Expr:
        return _rank(expr, method=method)

    @staticmethod
    def zscore(expr: Union[Expr, str], eps: float = 1e-8) -> Expr:
        return _zscore(expr, eps=eps)

    @staticmethod
    def winsorize(expr: Union[Expr, str], lower: float = 0.01, upper: float = 0.01,
                  method: str = "quantile") -> Expr:
        return _winsorize(expr, lower=lower, upper=upper, method=method)

    @staticmethod
    def neutralize(expr: Union[Expr, str], group: Union[Expr, str]) -> Expr:
        return _neutralize(expr, group=group)

    @staticmethod
    def neutralize_market(expr: Union[Expr, str]) -> Expr:
        return _neutralize_market(expr)

    @staticmethod
    def scale(expr: Union[Expr, str], method: str = "zscore") -> Expr:
        return _scale(expr, method=method)

    @staticmethod
    def percentile(expr: Union[Expr, str]) -> Expr:
        e = expr if isinstance(expr, Expr) else pl.col(expr)
        col_name = expr if isinstance(expr, str) else "percentile"
        return (e.rank() / e.count()).alias(col_name)

    @staticmethod
    def rank_ic(expr: Union[Expr, str], target: Union[Expr, str]) -> Expr:
        return _rank_ic(expr, target=target)

    @staticmethod
    def ic(expr: Union[Expr, str], target: Union[Expr, str]) -> Expr:
        return _ic(expr, target=target)

    @staticmethod
    def group_norm(expr: Union[Expr, str], group: Union[Expr, str], method: str = "zscore") -> Expr:
        return _group_norm(expr, group=group, method=method)

    @staticmethod
    def group_winsorize(expr: Union[Expr, str], group: Union[Expr, str],
                        lower: float = 0.01, upper: float = 0.01) -> Expr:
        return _group_winsorize(expr, group=group, lower=lower, upper=upper)
