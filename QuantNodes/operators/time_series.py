# coding=utf-8
"""
时间序列算子

基于 Polars 的时间序列运算，满足滚动窗口计算需求。

Available Operators:
    - ts_mean: 滚动均值
    - ts_std: 滚动标准差
    - ts_max: 滚动最大值
    - ts_min: 滚动最小值
    - ts_sum: 滚动求和
    - ts_median: 滚动中位数
    - ts_corr: 滚动相关系数
    - ts_cov: 滚动协方差
    - ts_rank: 滚动排名
    - ts_delta: 差分
    - ts_pct_change: 百分比变化
    - ts_lag: 滞后

Usage:
    >>> ts.ts_mean(pl.col("close"), 20)
    >>> ts.ts_std(pl.col("close"), 20)
    >>> ts.ts_corr(pl.col("close"), pl.col("volume"), 20)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional

import polars as pl
from polars import Expr

if TYPE_CHECKING:
    from polars import LazyFrame


class TimeSeriesOperators:
    """时间序列算子"""
    
    @staticmethod
    def ts_mean(
        expr: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> Expr:
        """
        滚动均值
        
        Args:
            expr: 表达式或列名
            window: 窗口大小
            min_periods: 最小观测数，默认 window//2
        
        Example:
            >>> ts.ts_mean("close", 20)
            >>> ts.ts_mean(pl.col("close"), 20)
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_mean(window, min_samples=min_periods)
    
    @staticmethod
    def ts_std(
        expr: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None,
        ddof: int = 1
    ) -> Expr:
        """
        滚动标准差
        
        Args:
            expr: 表达式或列名
            window: 窗口大小
            min_periods: 最小观测数
            ddof: 自由度调整
        
        Example:
            >>> ts.ts_std("close", 20)
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_std(window, min_samples=min_periods)
    
    @staticmethod
    def ts_max(
        expr: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> Expr:
        """
        滚动最大值
        
        Args:
            expr: 表达式或列名
            window: 窗口大小
            min_periods: 最小观测数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_max(window, min_samples=min_periods)
    
    @staticmethod
    def ts_min(
        expr: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> Expr:
        """
        滚动最小值
        
        Args:
            expr: 表达式或列名
            window: 窗口大小
            min_periods: 最小观测数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_min(window, min_samples=min_periods)
    
    @staticmethod
    def ts_sum(
        expr: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> Expr:
        """
        滚动求和
        
        Args:
            expr: 表达式或列名
            window: 窗口大小
            min_periods: 最小观测数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_sum(window, min_samples=min_periods)
    
    @staticmethod
    def ts_prod(
        expr: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> Expr:
        """滚动求积（log-sum-exp 方法）
        
        Args:
            expr: 表达式或列名
            window: 窗口大小
            min_periods: 最小观测数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        min_periods = min_periods or max(1, window // 2)
        return expr.log().rolling_sum(window, min_samples=min_periods).exp()
    
    @staticmethod
    def ts_median(
        expr: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> Expr:
        """
        滚动中位数
        
        Args:
            expr: 表达式或列名
            window: 窗口大小
            min_periods: 最小观测数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        min_periods = min_periods or max(1, window // 2)
        return expr.rolling_median(window, min_samples=min_periods)
    
    @staticmethod
    def ts_corr(
        expr_a: Union[Expr, str],
        expr_b: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> Expr:
        """
        滚动相关系数
        
        Args:
            expr_a: 第一个表达式或列名
            expr_b: 第二个表达式或列名
            window: 窗口大小
            min_periods: 最小观测数
        """
        if isinstance(expr_a, str):
            expr_a = pl.col(expr_a)
        if isinstance(expr_b, str):
            expr_b = pl.col(expr_b)
        min_periods = min_periods or max(1, window // 2)
        
        mean_a = expr_a.rolling_mean(window, min_samples=min_periods)
        mean_b = expr_b.rolling_mean(window, min_samples=min_periods)
        mean_ab = (expr_a * expr_b).rolling_mean(window, min_samples=min_periods)
        cov = mean_ab - mean_a * mean_b
        std_a = expr_a.rolling_std(window, min_samples=min_periods, ddof=0)
        std_b = expr_b.rolling_std(window, min_samples=min_periods, ddof=0)
        
        return cov / (std_a * std_b + 1e-8)
    
    @staticmethod
    def ts_cov(
        expr_a: Union[Expr, str],
        expr_b: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> Expr:
        """
        滚动协方差
        
        Args:
            expr_a: 第一个表达式或列名
            expr_b: 第二个表达式或列名
            window: 窗口大小
            min_periods: 最小观测数
        """
        if isinstance(expr_a, str):
            expr_a = pl.col(expr_a)
        if isinstance(expr_b, str):
            expr_b = pl.col(expr_b)
        
        mean_a = expr_a.rolling_mean(window)
        mean_b = expr_b.rolling_mean(window)
        
        return ((expr_a - mean_a) * (expr_b - mean_b)).rolling_mean(window)
    
    @staticmethod
    def ts_rank(
        expr: Union[Expr, str],
        window: int = 20,
        min_periods: Optional[int] = None
    ) -> Expr:
        """
        滚动排名 (0-1 归一化)
        
        Args:
            expr: 表达式或列名
            window: 窗口大小
            min_periods: 最小观测数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.rolling_rank(window)
    
    @staticmethod
    def ts_delta(
        expr: Union[Expr, str],
        periods: int = 1
    ) -> Expr:
        """
        差分
        
        Args:
            expr: 表达式或列名
            periods: 差分阶数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.diff(periods)
    
    @staticmethod
    def ts_pct_change(
        expr: Union[Expr, str],
        periods: int = 1
    ) -> Expr:
        """
        百分比变化
        
        Args:
            expr: 表达式或列名
            periods: 变化阶数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.pct_change(periods)
    
    @staticmethod
    def ts_lag(
        expr: Union[Expr, str],
        periods: int = 1
    ) -> Expr:
        """
        滞后
        
        Args:
            expr: 表达式或列名
            periods: 滞后阶数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.shift(periods)
    
    @staticmethod
    def ts_shift(
        expr: Union[Expr, str],
        periods: int = 1
    ) -> Expr:
        """
        前向移动 (同 ts_lag)
        """
        return TimeSeriesOperators.ts_lag(expr, periods)
    
    @staticmethod
    def ts_lead(
        expr: Union[Expr, str],
        periods: int = 1
    ) -> Expr:
        """
        前向移动
        
        Args:
            expr: 表达式或列名
            periods: 移动阶数
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.shift(-periods)
    
    # ========== 指数加权 ==========
    
    @staticmethod
    def ewm_mean(
        expr: Union[Expr, str],
        alpha: float = 0.5,
        adjust: bool = True
    ) -> Expr:
        """
        指数加权移动平均
        
        Args:
            expr: 表达式或列名
            alpha: 平滑因子
            adjust: 是否调整
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.ewm_mean(alpha=alpha, adjust=adjust)
    
    @staticmethod
    def ewm_std(
        expr: Union[Expr, str],
        alpha: float = 0.5,
        adjust: bool = True
    ) -> Expr:
        """
        指数加权移动标准差
        
        Args:
            expr: 表达式或列名
            alpha: 平滑因子
            adjust: 是否调整
        """
        if isinstance(expr, str):
            expr = pl.col(expr)
        return expr.ewm_std(alpha=alpha, adjust=adjust)
    
    @staticmethod
    def ewm_corr(
        expr_a: Union[Expr, str],
        expr_b: Union[Expr, str],
        alpha: float = 0.5
    ) -> Expr:
        """
        指数加权相关系数
        
        Args:
            expr_a: 第一个表达式或列名
            expr_b: 第二个表达式或列名
            alpha: 平滑因子
        """
        if isinstance(expr_a, str):
            expr_a = pl.col(expr_a)
        if isinstance(expr_b, str):
            expr_b = pl.col(expr_b)
        mean_a = expr_a.ewm_mean(alpha=alpha)
        mean_b = expr_b.ewm_mean(alpha=alpha)
        var_a = expr_a.ewm_var(alpha=alpha)
        var_b = expr_b.ewm_var(alpha=alpha)
        cov = (expr_a * expr_b).ewm_mean(alpha=alpha) - mean_a * mean_b
        return cov / ((var_a * var_b).sqrt() + 1e-10)