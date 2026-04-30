# coding=utf-8
"""
截面算子

基于 Polars 的截面运算，即在同一时间点对所有标的进行计算。

Available Operators:
    - rank: 截面排名
    - zscore: Z-score 标准化
    - winsorize: 去极值
    - neutralize: 行业中性的
    - scale: 归一化
    - percentile: 百分位排名

Usage:
    >>> sec.rank(pl.col("factor"))
    >>> sec.zscore(pl.col("factor"))
    >>> sec.winsorize(pl.col("factor"), 0.01, 0.01)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional

import polars as pl
from polars import Expr

if TYPE_CHECKING:
    from polars import LazyFrame, DataFrame


class SectionOperators:
    """截面算子"""
    
    @staticmethod
    def rank(
        expr: Union[Expr, str],
        method: str = "dense"
    ) -> Expr:
        """
        截面排名 (归一化到 0-1)
        
        Args:
            expr: 表达式或列名
            method: 排名方法 (dense/ordinal/min/max/average)
        
        Example:
            >>> sec.rank("factor")
            >>> sec.rank(pl.col("factor"), "dense")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        # 使用 rank 并归一化到 0-1
        if method == "dense":
            return (expr.rank() - 1) / (expr.count() - 1)
        elif method == "ordinal":
            return expr.rank()
        elif method == "min":
            return (expr.rank() - expr.rank().min()) / (expr.rank().max() - expr.rank().min())
        elif method == "average":
            return (expr.rank() - 1) / (expr.count() - 1)
        else:
            return expr.rank() / expr.count()
    
    @staticmethod
    def zscore(
        expr: Union[Expr, str],
        eps: float = 1e-8
    ) -> Expr:
        """
        Z-score 标准化
        
        Args:
            expr: 表达式或列名
            eps: 防止除零的常数
        
        Example:
            >>> sec.zscore("factor")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        mean = expr.mean()
        std = expr.std()
        return (expr - mean) / (std + eps)
    
    @staticmethod
    def winsorize(
        expr: Union[Expr, str],
        lower: float = 0.01,
        upper: float = 0.01,
        method: str = "quantile"
    ) -> Expr:
        """
        去极值
        
        Args:
            expr: 表达式或列名
            lower: 下界比例
            upper: 上界比例
            method: 方法 (quantile/iqr)
        
        Example:
            >>> sec.winsorize("factor", 0.01, 0.01)
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        q_low = expr.quantile(lower)
        q_high = expr.quantile(1 - upper)
        return expr.clip(lower_bound=q_low, upper_bound=q_high)
    
    @staticmethod
    def neutralize(
        expr: Union[Expr, str],
        group: Union[Expr, str],
        eps: float = 1e-8
    ) -> Expr:
        """
        行业中性的 (减去行业均值)
        
        Args:
            expr: 表达式或列名
            group: 分组表达式或列名 (如行业)
        
        Example:
            >>> sec.neutralize("factor", "industry")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(group, str):
            group = pl.col(group)
        
        group_mean = expr.mean().over(group)
        return expr - group_mean
    
    @staticmethod
    def neutralize_market(
        expr: Union[Expr, str],
        market: Union[Expr, str] = None,
        eps: float = 1e-8
    ) -> Expr:
        """
        市场中性 (减去市场均值)
        
        Args:
            expr: 表达式或列名
            market: 市场指数列名
        
        Example:
            >>> sec.neutralize_market("factor")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        market_mean = expr.mean()
        return expr - market_mean
    
    @staticmethod
    def scale(
        expr: Union[Expr, str],
        method: str = "zscore"
    ) -> Expr:
        """
        归一化
        
        Args:
            expr: 表达式或列名
            method: 方法 (zscore/minmax)
        
        Example:
            >>> sec.scale("factor", "minmax")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        if method == "zscore":
            return SectionOperators.zscore(expr)
        elif method == "minmax":
            return (expr - expr.min()) / (expr.max() - expr.min() + 1e-8)
        elif method == "abs":
            return expr / expr.abs().max()
        else:
            return SectionOperators.zscore(expr)
    
    @staticmethod
    def percentile(
        expr: Union[Expr, str]
    ) -> Expr:
        """
        百分位排名
        
        Args:
            expr: 表达式或列名
        
        Example:
            >>> sec.percentile("factor")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        
        return expr.rank() / expr.count()
    
    @staticmethod
    def rank_ic(
        expr: Union[Expr, str],
        target: Union[Expr, str]
    ) -> Expr:
        """
        Rank IC (Spearman 相关系数)
        
        Args:
            expr: 因子表达式
            target: 目标表达式
        
        Example:
            >>> sec.rank_ic("factor", "return")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(target, str):
            target = pl.col(target)
        
        return pl.corr(expr.rank(), target.rank())
    
    @staticmethod
    def ic(
        expr: Union[Expr, str],
        target: Union[Expr, str]
    ) -> Expr:
        """
        IC (Pearson 相关系数)
        
        Args:
            expr: 因子表达式
            target: 目标表达式
        
        Example:
            >>> sec.ic("factor", "return")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(target, str):
            target = pl.col(target)
        
        return pl.corr(expr, target)
    
    @staticmethod
    def group_norm(
        expr: Union[Expr, str],
        group: Union[Expr, str],
        method: str = "zscore"
    ) -> Expr:
        """
        分组标准化
        
        Args:
            expr: 表达式或列名
            group: 分组列名
            method: 标准化方法
        
        Example:
            >>> sec.group_norm("factor", "industry", "zscore")
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(group, str):
            group = pl.col(group)
        
        if method == "zscore":
            group_mean = expr.mean().over(group)
            group_std = expr.std().over(group)
            return (expr - group_mean) / (group_std + 1e-8)
        elif method == "rank":
            return (expr.rank() - 1).over(group)
        else:
            return expr
    
    @staticmethod
    def group_winsorize(
        expr: Union[Expr, str],
        group: Union[Expr, str],
        lower: float = 0.01,
        upper: float = 0.01
    ) -> Expr:
        """
        分组去极值
        
        Args:
            expr: 表达式或列名
            group: 分组列名
            lower: 下界比例
            upper: 上界比例
        
        Example:
            >>> sec.group_winsorize("factor", "industry", 0.01, 0.01)
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        if isinstance(group, str):
            group = pl.col(group)
        
        q_low = expr.quantile(lower).over(group)
        q_high = expr.quantile(1 - upper).over(group)
        return expr.clip(lower_bound=q_low, upper_bound=q_high)