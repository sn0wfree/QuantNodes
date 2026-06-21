# coding=utf-8
"""
QuantNodes Operators - Polars 算子模块

基于 Polars 的因子运算算子，提供简洁的表达式接口。

Modules:
    time_series: 时间序列算子 (ts_mean, ts_std, ts_corr...)
    section: 截面算子 (rank, zscore, winsorize...)
    math: 数学算子 (add, mul, log, pow...)
    composite: 组合算子 (weighted_sum, combine...)
    talib: TA-Lib 技术分析指标 (rsi, sma, macd, bbands, ...)

Usage:
    from QuantNodes.operators import ts, sec, math, talib_ops

    # 时间序列
    result = ts.ts_mean(pl.col("close"), 20)

    # 截面
    result = sec.rank(pl.col("factor"))

    # 数学
    result = math.add(pl.col("factor"), 1.0)

    # TA-Lib
    result = talib_ops.rsi(pl.col("close"), timeperiod=14)
"""

from .time_series import TimeSeriesOperators as _ts
from .section import SectionOperators as _sec
from .math import MathOperators as _math
from .composite import CompositeOperators as _composite
from .proxy import list_operators, get_operator, register_operator
from .custom import CustomOperator, OperatorTemplate, point, time, section
# PR-QN-3a (2026-06-21): Composite DAG re-exports
from .composite_dag import (
    composite_operator,
    CompositeSpec,
    ParamSpec,
    is_composite_op,
    get_composite_spec,
    list_composite_ops,
    get_composite_doc_for_llm,
    load_composites_from_yaml,
)

# 统一导出
ts = _ts()
sec = _sec()
math = _math()
composite = _composite()

# TA-Lib (可选)
try:
    from .talib import TaLibOperators as _talib
    talib_ops = _talib()
except ImportError:
    talib_ops = None

__all__ = [
    "ts", "sec", "math", "composite", "talib_ops",
    "TimeSeriesOperators", "SectionOperators", "MathOperators", "CompositeOperators",
    "list_operators", "get_operator", "register_operator",
    "CustomOperator", "OperatorTemplate",
    "point", "time", "section",
    # PR-QN-3a (2026-06-21): Composite DAG
    "composite_operator", "CompositeSpec", "ParamSpec",
    "is_composite_op", "get_composite_spec", "list_composite_ops",
    "get_composite_doc_for_llm", "load_composites_from_yaml",
]
