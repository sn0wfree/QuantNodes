# coding=utf-8
"""
QuantNodes Operators - Polars 算子模块

基于 Polars 的因子运算算子，提供简洁的表达式接口。

Modules:
    time_series: 时间序列算子 (ts_mean, ts_std, ts_corr...)
    section: 截面算子 (rank, zscore, winsorize...)
    math: 数学算子 (add, mul, log, pow...)
    composite: 组合算子 (weighted_sum, combine...)

Usage:
    from QuantNodes.operators import ts, sec, math
    
    # 时间序列
    result = ts.ts_mean(pl.col("close"), 20)
    
    # 截面
    result = sec.rank(pl.col("factor"))
    
    # 数学
    result = math.add(pl.col("factor"), 1.0)
"""

from .time_series import TimeSeriesOperators as _ts
from .section import SectionOperators as _sec
from .math import MathOperators as _math
from .composite import CompositeOperators as _composite

# 统一导出
ts = _ts()
sec = _sec()
math = _math()
composite = _composite()

__all__ = ["ts", "sec", "math", "composite", "TimeSeriesOperators", "SectionOperators", "MathOperators", "CompositeOperators"]