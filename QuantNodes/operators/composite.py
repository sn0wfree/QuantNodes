# coding=utf-8
"""
组合算子

基于 Polars 的因子组合运算。

Available Operators:
    - weighted_sum: 加权求和
    - weighted_avg: 加权平均
    - max: 最大值
    - min: 最小值
    - combine: 组合多个因子
    - blend: 混合因子
    - select_top: 选择顶部因子
    - filter_positive: 过滤正信号

Usage:
    >>> composite.weighted_sum([pl.col("f1"), pl.col("f2')], [0.6, 0.4])
    >>> composite.blend("f1", "f2", 0.5)
"""

from __future__ import annotations

from typing import Union, List, Optional

import polars as pl
from polars import Expr


class CompositeOperators:
    """组合算子"""
    
    @staticmethod
    def weighted_sum(
        factors: List[Union[Expr, str]],
        weights: List[float]
    ) -> Expr:
        """
        加权求和
        
        Args:
            factors: 因子列表
            weights: 权重列表
        
        Example:
            >>> composite.weighted_sum(["f1", "f2"], [0.6, 0.4])
        """
        # 转换因子
        exprs = []
        for f in factors:
            if isinstance(f, str):
                exprs.append(pl.col(f))
            else:
                exprs.append(f)
        
        # 对应相乘后求和
        result = None
        for e, w in zip(exprs, weights):
            if result is None:
                result = e * w
            else:
                result = result + e * w
        
        return result
    
    @staticmethod
    def weighted_avg(
        factors: List[Union[Expr, str]],
        weights: Optional[List[float]] = None
    ) -> Expr:
        """
        加权平均
        
        Args:
            factors: 因子列表
            weights: 权重列表 (默认等权重)
        """
        n = len(factors)
        
        # 转换因子
        exprs = []
        for f in factors:
            if isinstance(f, str):
                exprs.append(pl.col(f))
            else:
                exprs.append(f)
        
        # 自动归一化权重
        if weights is None:
            weights = [1.0 / n] * n
        else:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        return CompositeOperators.weighted_sum(exprs, weights)
    
    @staticmethod
    def max(
        factors: List[Union[Expr, str]]
    ) -> Expr:
        """
        取最大值
        
        Args:
            factors: 因子列表
        """
        exprs = []
        for f in factors:
            if isinstance(f, str):
                exprs.append(pl.col(f))
            else:
                exprs.append(f)
        
        return pl.max_horizontal(exprs)
    
    @staticmethod
    def min(
        factors: List[Union[Expr, str]]
    ) -> Expr:
        """
        取最小值
        
        Args:
            factors: 因子列表
        """
        exprs = []
        for f in factors:
            if isinstance(f, str):
                exprs.append(pl.col(f))
            else:
                exprs.append(f)
        
        return pl.min_horizontal(exprs)
    
    @staticmethod
    def abs_max(
        factors: List[Union[Expr, str]]
    ) -> Expr:
        """
        取绝对值最大的
        
        Args:
            factors: 因子列表
        """
        exprs = []
        for f in factors:
            if isinstance(f, str):
                exprs.append(pl.col(f).abs())
            else:
                exprs.append(f.abs())
        
        return pl.max_horizontal(exprs)
    
    @staticmethod
    def blend(
        factor_a: Union[Expr, str],
        factor_b: Union[Expr, str],
        alpha: float = 0.5
    ) -> Expr:
        """
        线性混合两个因子
        
        Args:
            factor_a: 第一个因子
            factor_b: 第二个因子
            alpha: 混合系数 (factor_a * alpha + factor_b * (1-alpha))
        
        Example:
            >>> composite.blend("momentum", "value", 0.6)
        """
        if isinstance(factor_a, str):
            factor_a = pl.col(factor_a)
        if isinstance(factor_b, str):
            factor_b = pl.col(factor_b)
        
        return factor_a * alpha + factor_b * (1 - alpha)
    
    @staticmethod
    def combine(
        factors: List[Union[Expr, str]],
        method: str = "sum"
    ) -> Expr:
        """
        组合多个因子
        
        Args:
            factors: 因子列表
            method: 方法 (sum/avg/mul/max)
        """
        if method == "sum":
            return sum([f if isinstance(f, Expr) else pl.col(f) for f in factors])
        elif method == "avg":
            return CompositeOperators.weighted_avg(factors)
        elif method == "mul":
            result = None
            for f in factors:
                e = f if isinstance(f, Expr) else pl.col(f)
                if result is None:
                    result = e
                else:
                    result = result * e
            return result
        elif method == "max":
            return CompositeOperators.max(factors)
        
        return factors[0]
    
    @staticmethod
    def select_top(
        expr: Union[Expr, str],
        n: int = 10,
        ascending: bool = False
    ) -> Expr:
        """
        选择顶部 N 个
        
        Args:
            expr: 表达式
            n: 选择数量
            ascending: 是否升序 (False = 取最大的)
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        # 在截面中取 top N，需要配合 group 使用
        # 这里返回排名信息
        return expr.rank(descending=not ascending)
    
    @staticmethod
    def filter_positive(
        expr: Union[Expr, str]
    ) -> Expr:
        """
        过滤正信号 (置零)
        
        Args:
            expr: 表达式
        
        Returns:
            负值保留，正值置零
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        # 保留负值，正值置零
        return expr * (expr < 0).cast(pl.Float64)
    
    @staticmethod
    def filter_negative(
        expr: Union[Expr, str]
    ) -> Expr:
        """
        过滤负信号 (置零)
        
        Args:
            expr: 表达式
        
        Returns:
            正值保留，负值置零
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        # 保留正值，负值置零
        return expr * (expr > 0).cast(pl.Float64)
    
    @staticmethod
    def abs_filter(
        expr: Union[Expr, str],
        threshold: float = 0.0
    ) -> Expr:
        """
        绝对值过滤 (小于阈值置零)
        
        Args:
            expr: 表达式
            threshold: 阈值
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        # 绝对值小于阈值置零
        return expr * (expr.abs() > threshold).cast(pl.Float64)
    
    @staticmethod
    def rank_sort(
        factors: List[Union[Expr, str]],
        weights: Optional[List[float]] = None
    ) -> Expr:
        """
        排名加权组合
        
        Args:
            factors: 因子列表
            weights: 权重
        """
        # 先排名再加权
        ranks = []
        for f in factors:
            if isinstance(f, str):
                ranks.append(pl.col(f).rank())
            else:
                ranks.append(f.rank())
        
        return CompositeOperators.weighted_avg(ranks, weights)