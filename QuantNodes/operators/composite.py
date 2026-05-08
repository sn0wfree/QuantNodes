# coding=utf-8
"""
组合算子（代理层）

基于 factor_functions/composite_ops.py 的实现，提供统一的类接口。

Available Operators:
    - weighted_sum: 加权求和
    - weighted_avg: 加权平均
    - max: 最大值
    - min: 最小值
    - abs_max: 绝对值最大
    - combine: 组合两个因子
    - blend: 混合两个因子
    - select_top: 选择顶部
    - filter_positive: 过滤正信号
    - filter_negative: 过滤负信号
    - abs_filter: 绝对值过滤
    - rank_sort: 排名排序

Usage:
    >>> composite.weighted_sum([pl.col("f1"), pl.col("f2")], [0.6, 0.4])
    >>> composite.blend("f1", "f2", 0.5)
"""

from __future__ import annotations

from typing import Union, List, Optional

import polars as pl
from polars import Expr

from QuantNodes.factor_node.factor_functions.composite_ops import (
    merge as _merge,
    aggregate as _aggregate,
    blend as _blend,
)


class CompositeOperators:
    """组合算子（代理层）"""

    @staticmethod
    def weighted_sum(factors: List[Union[Expr, str]], weights: List[float]) -> Expr:
        exprs = [_ensure_expr(f) for f in factors]
        weights_arr = pl.Series(weights)
        weights_arr = weights_arr / weights_arr.sum()
        
        first_factor = factors[0]
        if isinstance(first_factor, str):
            col_name = first_factor
        elif hasattr(first_factor, 'meta') and hasattr(first_factor.meta, 'output_name'):
            try:
                col_name = first_factor.meta.output_name()
            except Exception:
                col_name = "result"
        else:
            col_name = "result"
        
        return sum(e * w for e, w in zip(exprs, weights_arr)).alias(col_name)

    @staticmethod
    def weighted_avg(factors: List[Union[Expr, str]], weights: Optional[List[float]] = None) -> Expr:
        if weights is None:
            weights = [1.0] * len(factors)
        return CompositeOperators.weighted_sum(factors, weights)

    @staticmethod
    def max(factors: List[Union[Expr, str]]) -> Expr:
        exprs = [_ensure_expr(f) for f in factors]
        first_factor = factors[0] if isinstance(factors[0], str) else "result"
        return pl.max_horizontal(*exprs).alias(first_factor)

    @staticmethod
    def min(factors: List[Union[Expr, str]]) -> Expr:
        exprs = [_ensure_expr(f) for f in factors]
        first_factor = factors[0] if isinstance(factors[0], str) else "result"
        return pl.min_horizontal(*exprs).alias(first_factor)

    @staticmethod
    def abs_max(factors: List[Union[Expr, str]]) -> Expr:
        exprs = [_ensure_expr(f).abs() for f in factors]
        first_factor = factors[0] if isinstance(factors[0], str) else "result"
        return pl.max_horizontal(*exprs).alias(first_factor)

    @staticmethod
    def combine(factors: List[Union[Expr, str]], method: str = "add") -> Expr:
        exprs = [_ensure_expr(f) for f in factors]
        first_factor = factors[0] if isinstance(factors[0], str) else "result"
        
        if method in ("add", "sum"):
            return sum(exprs).alias(first_factor)
        elif method == "avg":
            return (sum(exprs) / len(exprs)).alias(first_factor)
        elif method == "mul":
            result = exprs[0]
            for e in exprs[1:]:
                result = result * e
            return result.alias(first_factor)
        elif method == "max":
            return pl.max_horizontal(*exprs).alias(first_factor)
        elif method == "min":
            return pl.min_horizontal(*exprs).alias(first_factor)
        return exprs[0].alias(first_factor)

    @staticmethod
    def blend(f1: Union[Expr, str], f2: Union[Expr, str], alpha: float = 0.5) -> Expr:
        return _blend(f1, f2, alpha=alpha)

    @staticmethod
    def select_top(f: Union[Expr, str], n: int = 1, ascending: bool = False) -> Expr:
        e = _ensure_expr(f)
        col_name = f if isinstance(f, str) else "result"
        if ascending:
            return e.rank(method="dense").alias(col_name)
        else:
            return (e.count() - e.rank(method="dense") + 1).alias(col_name)

    @staticmethod
    def filter_positive(f: Union[Expr, str]) -> Expr:
        e = _ensure_expr(f)
        col_name = f if isinstance(f, str) else "result"
        return pl.when(e > 0).then(pl.lit(0.0)).otherwise(e).alias(col_name)

    @staticmethod
    def filter_negative(f: Union[Expr, str]) -> Expr:
        e = _ensure_expr(f)
        col_name = f if isinstance(f, str) else "result"
        return pl.when(e < 0).then(pl.lit(0.0)).otherwise(e).alias(col_name)

    @staticmethod
    def abs_filter(f: Union[Expr, str], threshold: float = 0.0) -> Expr:
        e = _ensure_expr(f)
        col_name = f if isinstance(f, str) else "result"
        return pl.when(e.abs() > threshold).then(e).otherwise(pl.lit(0.0)).alias(col_name)

    @staticmethod
    def rank_sort(factors: List[Union[Expr, str]], weights: Optional[List[float]] = None) -> List[Expr]:
        exprs = [_ensure_expr(f) for f in factors]
        
        if weights is not None:
            weights_arr = pl.Series(weights)
            weights_arr = weights_arr / weights_arr.sum()
            weighted_expr = sum(e * w for e, w in zip(exprs, weights_arr))
            combined = weighted_expr
        else:
            combined = pl.max_horizontal(*exprs)
        
        return [combined.rank().eq(i + 1).alias(f) if isinstance(f, str) else combined.rank().eq(i + 1) 
                for i, f in enumerate(factors)]


def _ensure_expr(f: Union[Expr, str]) -> Expr:
    if isinstance(f, str):
        return pl.col(f)
    return f
