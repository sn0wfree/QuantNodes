# coding=utf-8
"""
符号计算引擎 - 技术指标函数库

提供常用的技术分析指标函数。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Union

if TYPE_CHECKING:
    from QuantNodes.symbolic.expression import SQLExpression
    from QuantNodes.symbolic.dialect import SQLDialect


class TechnicalFunctions:
    """
    技术分析函数库

    提供常用的技术指标函数，如移动平均、RSI、MACD 等。
    这些函数生成 SQL 表达式，可在不同数据库方言中执行。
    """

    @staticmethod
    def sma(expr: "SQLExpression", window: int) -> "SQLExpression":
        """
        简单移动平均

        Args:
            expr: 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("avg", expr)

    @staticmethod
    def ema(expr: "SQLExpression", window: int) -> "SQLExpression":
        """
        指数移动平均

        Args:
            expr: 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("expAverage", expr)

    @staticmethod
    def sum(expr: "SQLExpression", window: int) -> "SQLExpression":
        """
        滑动窗口求和

        Args:
            expr: 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("sum", expr)

    @staticmethod
    def stddev(expr: "SQLExpression", window: int) -> "SQLExpression":
        """
        滑动窗口标准差

        Args:
            expr: 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("stddevPop", expr)

    @staticmethod
    def variance(expr: "SQLExpression", window: int) -> "SQLExpression":
        """
        滑动窗口方差

        Args:
            expr: 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("varPop", expr)

    @staticmethod
    def min(expr: "SQLExpression", window: int) -> "SQLExpression":
        """
        滑动窗口最小值

        Args:
            expr: 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("min", expr)

    @staticmethod
    def max(expr: "SQLExpression", window: int) -> "SQLExpression":
        """
        滑动窗口最大值

        Args:
            expr: 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("max", expr)

    @staticmethod
    def median(expr: "SQLExpression", window: int) -> "SQLExpression":
        """
        滑动窗口中位数

        Args:
            expr: 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("median", expr)

    @staticmethod
    def quantile(expr: "SQLExpression", window: int, q: float) -> "SQLExpression":
        """
        滑动窗口分位数

        Args:
            expr: 表达式
            window: 窗口大小
            q: 分位数 (0-1)

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("quantile", expr)

    @staticmethod
    def delay(expr: "SQLExpression", n: int = 1) -> "SQLExpression":
        """
        滞后操作 (N日前值)

        Args:
            expr: 表达式
            n: 滞后周期数

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("lagInFrame", expr)

    @staticmethod
    def delta(expr: "SQLExpression", n: int = 1) -> "SQLExpression":
        """
        差分 (与N日前的差值)

        Args:
            expr: 表达式
            n: 滞后周期数

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLBinaryOp, SQLFunction
        current = expr
        lagged = SQLFunction("lagInFrame", expr)
        return current - lagged

    @staticmethod
    def pct_change(expr: "SQLExpression", n: int = 1) -> "SQLExpression":
        """
        百分比变化

        Args:
            expr: 表达式
            n: 滞后周期数

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLBinaryOp, SQLFunction
        current = expr
        lagged = SQLFunction("lagInFrame", expr)
        return (current - lagged) / lagged

    @staticmethod
    def correlation(x: "SQLExpression", y: "SQLExpression", window: int) -> "SQLExpression":
        """
        滑动窗口相关系数

        Args:
            x: X 表达式
            y: Y 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("corr", x, y)

    @staticmethod
    def covariance(x: "SQLExpression", y: "SQLExpression", window: int) -> "SQLExpression":
        """
        滑动窗口协方差

        Args:
            x: X 表达式
            y: Y 表达式
            window: 窗口大小

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("covarPop", x, y)

    @staticmethod
    def rank(expr: "SQLExpression") -> "SQLExpression":
        """
        排名

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("rank")

    @staticmethod
    def dense_rank(expr: "SQLExpression") -> "SQLExpression":
        """
        密集排名

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("dense_rank")

    @staticmethod
    def row_number(expr: "SQLExpression") -> "SQLExpression":
        """
        行号

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("row_number")

    @staticmethod
    def cumsum(expr: "SQLExpression") -> "SQLExpression":
        """
        累积求和

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("sum", expr)

    @staticmethod
    def cumprod(expr: "SQLExpression") -> "SQLExpression":
        """
        累积求积

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("groupArray", expr)

    @staticmethod
    def running_diff(expr: "SQLExpression") -> "SQLExpression":
        """
        行间差分

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("runningDifference", expr)

    @staticmethod
    def abs(expr: "SQLExpression") -> "SQLExpression":
        """
        绝对值

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("abs", expr)

    @staticmethod
    def sqrt(expr: "SQLExpression") -> "SQLExpression":
        """
        平方根

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("sqrt", expr)

    @staticmethod
    def log(expr: "SQLExpression") -> "SQLExpression":
        """
        自然对数

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("log", expr)

    @staticmethod
    def log10(expr: "SQLExpression") -> "SQLExpression":
        """
        底数为10的对数

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("log10", expr)

    @staticmethod
    def pow(expr: "SQLExpression", power: float) -> "SQLExpression":
        """
        幂函数

        Args:
            expr: 表达式
            power: 幂次

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("pow", expr)

    @staticmethod
    def round(expr: "SQLExpression", decimals: int = 0) -> "SQLExpression":
        """
        四舍五入

        Args:
            expr: 表达式
            decimals: 小数位数

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("round", expr)

    @staticmethod
    def ceil(expr: "SQLExpression") -> "SQLExpression":
        """
        向上取整

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("ceil", expr)

    @staticmethod
    def floor(expr: "SQLExpression") -> "SQLExpression":
        """
        向下取整

        Returns:
            SQLExpression
        """
        from QuantNodes.symbolic.expression import SQLFunction
        return SQLFunction("floor", expr)


TA_FUNCTIONS = {
    "sma": TechnicalFunctions.sma,
    "ema": TechnicalFunctions.ema,
    "sum": TechnicalFunctions.sum,
    "stddev": TechnicalFunctions.stddev,
    "variance": TechnicalFunctions.variance,
    "min": TechnicalFunctions.min,
    "max": TechnicalFunctions.max,
    "median": TechnicalFunctions.median,
    "quantile": TechnicalFunctions.quantile,
    "delay": TechnicalFunctions.delay,
    "delta": TechnicalFunctions.delta,
    "pct_change": TechnicalFunctions.pct_change,
    "correlation": TechnicalFunctions.correlation,
    "covariance": TechnicalFunctions.covariance,
    "rank": TechnicalFunctions.rank,
    "dense_rank": TechnicalFunctions.dense_rank,
    "row_number": TechnicalFunctions.row_number,
    "cumsum": TechnicalFunctions.cumsum,
    "cumprod": TechnicalFunctions.cumprod,
    "running_diff": TechnicalFunctions.running_diff,
    "abs": TechnicalFunctions.abs,
    "sqrt": TechnicalFunctions.sqrt,
    "log": TechnicalFunctions.log,
    "log10": TechnicalFunctions.log10,
    "pow": TechnicalFunctions.pow,
    "round": TechnicalFunctions.round,
    "ceil": TechnicalFunctions.ceil,
    "floor": TechnicalFunctions.floor,
}
