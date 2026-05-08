# coding=utf-8
"""
时间序列算子（代理层）

基于 factor_functions/time_ops.py 的实现，提供统一的类接口。

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
    - ts_shift: 前向移动
    - ts_lead: 后向移动
    - ts_prod: 滚动求积
    - ewm_mean: 指数加权移动平均
    - ewm_std: 指数加权移动标准差
    - ewm_corr: 指数加权相关系数

Usage:
    >>> ts.ts_mean(pl.col("close"), 20)
    >>> ts.ts_std(pl.col("close"), 20)
    >>> ts.ts_corr(pl.col("close"), pl.col("volume"), 20)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, Optional

from polars import Expr

if TYPE_CHECKING:
    pass

from QuantNodes.factor_node.factor_functions.time_ops import (
    ts_mean as _ts_mean,
    ts_std as _ts_std,
    ts_max as _ts_max,
    ts_min as _ts_min,
    ts_sum as _ts_sum,
    rolling_prod as _ts_prod,
    ts_median as _ts_median,
    ts_corr as _ts_corr,
    ts_cov as _ts_cov,
    ts_rank as _ts_rank,
    ts_delta as _ts_delta,
    ts_pct_change as _ts_pct_change,
    ts_lag as _ts_lag,
    ts_lead as _ts_lead,
    ewm_mean as _ewm_mean,
    ewm_std as _ewm_std,
    ewm_corr as _ewm_corr,
)


class TimeSeriesOperators:
    """时间序列算子（代理层）"""

    @staticmethod
    def ts_mean(expr: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None) -> Expr:
        return _ts_mean(expr, window=window, min_periods=min_periods)

    @staticmethod
    def ts_std(expr: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None, ddof: int = 1) -> Expr:
        return _ts_std(expr, window=window, min_periods=min_periods, ddof=ddof)

    @staticmethod
    def ts_max(expr: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None) -> Expr:
        return _ts_max(expr, window=window, min_periods=min_periods)

    @staticmethod
    def ts_min(expr: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None) -> Expr:
        return _ts_min(expr, window=window, min_periods=min_periods)

    @staticmethod
    def ts_sum(expr: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None) -> Expr:
        return _ts_sum(expr, window=window, min_periods=min_periods)

    @staticmethod
    def ts_prod(expr: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None) -> Expr:
        return _ts_prod(expr, window=window, min_periods=min_periods)

    @staticmethod
    def ts_median(expr: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None) -> Expr:
        return _ts_median(expr, window=window, min_periods=min_periods)

    @staticmethod
    def ts_corr(expr_a: Union[Expr, str], expr_b: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None) -> Expr:
        return _ts_corr(expr_a, expr_b, window=window, min_periods=min_periods)

    @staticmethod
    def ts_cov(expr_a: Union[Expr, str], expr_b: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None) -> Expr:
        return _ts_cov(expr_a, expr_b, window=window, min_periods=min_periods)

    @staticmethod
    def ts_rank(expr: Union[Expr, str], window: int = 20, min_periods: Optional[int] = None) -> Expr:
        return _ts_rank(expr, window=window, min_periods=min_periods)

    @staticmethod
    def ts_delta(expr: Union[Expr, str], periods: int = 1) -> Expr:
        return _ts_delta(expr, periods=periods)

    @staticmethod
    def ts_pct_change(expr: Union[Expr, str], periods: int = 1) -> Expr:
        return _ts_pct_change(expr, periods=periods)

    @staticmethod
    def ts_lag(expr: Union[Expr, str], periods: int = 1) -> Expr:
        return _ts_lag(expr, periods=periods)

    @staticmethod
    def ts_shift(expr: Union[Expr, str], periods: int = 1) -> Expr:
        return _ts_lag(expr, periods=periods)

    @staticmethod
    def ts_lead(expr: Union[Expr, str], periods: int = 1) -> Expr:
        return _ts_lead(expr, periods=periods)

    @staticmethod
    def ewm_mean(expr: Union[Expr, str], alpha: float = 0.5, adjust: bool = True) -> Expr:
        return _ewm_mean(expr, alpha=alpha, adjust=adjust)

    @staticmethod
    def ewm_std(expr: Union[Expr, str], alpha: float = 0.5, adjust: bool = True) -> Expr:
        return _ewm_std(expr, alpha=alpha, adjust=adjust)

    @staticmethod
    def ewm_corr(expr_a: Union[Expr, str], expr_b: Union[Expr, str], alpha: float = 0.5) -> Expr:
        return _ewm_corr(expr_a, expr_b, alpha=alpha)
