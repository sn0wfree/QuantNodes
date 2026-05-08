# coding=utf-8
"""TechnicalFunctions 单元测试"""

from QuantNodes.symbolic.functions import TechnicalFunctions, TA_FUNCTIONS
from QuantNodes.symbolic.expression import (
    ColumnRef, SQLBinaryOp, SQLFunction,
)


class TestTechnicalFunctions:
    """TechnicalFunctions 测试"""

    def test_sma(self):
        """简单移动平均"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.sma(expr, 20)
        assert isinstance(result, SQLFunction)
        assert result.name == "avg"

    def test_ema(self):
        """指数移动平均"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.ema(expr, 12)
        assert isinstance(result, SQLFunction)
        assert result.name == "expAverage"

    def test_sum(self):
        """滑动窗口求和"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.sum(expr, 20)
        assert isinstance(result, SQLFunction)
        assert result.name == "sum"

    def test_stddev(self):
        """滑动窗口标准差"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.stddev(expr, 20)
        assert isinstance(result, SQLFunction)
        assert result.name == "stddevPop"

    def test_variance(self):
        """滑动窗口方差"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.variance(expr, 20)
        assert isinstance(result, SQLFunction)
        assert result.name == "varPop"

    def test_min(self):
        """滑动窗口最小值"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.min(expr, 20)
        assert isinstance(result, SQLFunction)
        assert result.name == "min"

    def test_max(self):
        """滑动窗口最大值"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.max(expr, 20)
        assert isinstance(result, SQLFunction)
        assert result.name == "max"

    def test_median(self):
        """滑动窗口中位数"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.median(expr, 20)
        assert isinstance(result, SQLFunction)
        assert result.name == "median"

    def test_quantile(self):
        """滑动窗口分位数"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.quantile(expr, 20, 0.5)
        assert isinstance(result, SQLFunction)

    def test_delay(self):
        """滞后操作"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.delay(expr, 1)
        assert isinstance(result, SQLFunction)
        assert result.name == "lagInFrame"

    def test_delta(self):
        """差分"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.delta(expr, 1)
        assert isinstance(result, SQLBinaryOp)

    def test_pct_change(self):
        """百分比变化"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.pct_change(expr, 1)
        assert isinstance(result, SQLBinaryOp)

    def test_correlation(self):
        """滑动窗口相关系数"""
        x = ColumnRef("close")
        y = ColumnRef("volume")
        result = TechnicalFunctions.correlation(x, y, 20)
        assert isinstance(result, SQLFunction)
        assert result.name == "corr"

    def test_covariance(self):
        """滑动窗口协方差"""
        x = ColumnRef("close")
        y = ColumnRef("volume")
        result = TechnicalFunctions.covariance(x, y, 20)
        assert isinstance(result, SQLFunction)
        assert result.name == "covarPop"

    def test_rank(self):
        """排名"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.rank(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "rank"

    def test_dense_rank(self):
        """密集排名"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.dense_rank(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "dense_rank"

    def test_row_number(self):
        """行号"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.row_number(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "row_number"

    def test_cumsum(self):
        """累积求和"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.cumsum(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "sum"

    def test_cumprod(self):
        """累积求积"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.cumprod(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "groupArray"

    def test_running_diff(self):
        """行间差分"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.running_diff(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "runningDifference"

    def test_abs(self):
        """绝对值"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.abs(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "abs"

    def test_sqrt(self):
        """平方根"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.sqrt(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "sqrt"

    def test_log(self):
        """自然对数"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.log(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "log"

    def test_log10(self):
        """底数为10的对数"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.log10(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "log10"

    def test_pow(self):
        """幂函数"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.pow(expr, 2.0)
        assert isinstance(result, SQLFunction)
        assert result.name == "pow"

    def test_round(self):
        """四舍五入"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.round(expr, 2)
        assert isinstance(result, SQLFunction)
        assert result.name == "round"

    def test_ceil(self):
        """向上取整"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.ceil(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "ceil"

    def test_floor(self):
        """向下取整"""
        expr = ColumnRef("close")
        result = TechnicalFunctions.floor(expr)
        assert isinstance(result, SQLFunction)
        assert result.name == "floor"


class TestTAFunctionsDict:
    """TA_FUNCTIONS 字典测试"""

    def test_ta_functions_complete(self):
        """验证所有函数都在 TA_FUNCTIONS 中"""
        expected = [
            "sma", "ema", "sum", "stddev", "variance", "min", "max",
            "median", "quantile", "delay", "delta", "pct_change",
            "correlation", "covariance", "rank", "dense_rank",
            "row_number", "cumsum", "cumprod", "running_diff",
            "abs", "sqrt", "log", "log10", "pow", "round", "ceil", "floor",
        ]
        for func in expected:
            assert func in TA_FUNCTIONS, f"Missing function: {func}"

    def test_ta_functions_callable(self):
        """验证所有函数都是可调用的"""
        for name, func in TA_FUNCTIONS.items():
            assert callable(func), f"{name} is not callable"
