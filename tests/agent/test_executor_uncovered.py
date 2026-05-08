# coding=utf-8
"""
ConfigExecutor 未覆盖方法单元测试

覆盖:
- compile() 方法
- get_expressions() 方法
- _resolve_universe() 方法（全部路径）
- run_backtest() 日期筛选 (String 类型)
- _apply_operator() 空输入
- _parse_func_args() keyword 参数
- _parse_value() 引号字符串
"""

import polars as pl
import tempfile
import os

from QuantNodes.agent.config.types import (
    StrategyConfig, FactorConfig, OperationConfig, BacktestConfig, DataConfig,
)
from QuantNodes.agent.config.executor import ConfigExecutor


def _make_config(**overrides):
    defaults = dict(
        name="test",
        factors=[],
        operations=[],
        composite=[],
    )
    defaults.update(overrides)
    return StrategyConfig(**defaults)


def _make_data(n_rows=100):
    """生成测试数据"""
    codes = ["000001.SZ", "600000.SH", "000858.SZ"]
    dates = [f"2023-07-{i:02d}" for i in range(1, min(n_rows // len(codes) + 2, 32))]
    rows = []
    for code in codes:
        for d in dates:
            rows.append({"code": code, "date": d, "close": 10.0, "volume": 1000.0})
    df = pl.DataFrame(rows)
    return df.lazy()


class TestCompile:
    """compile() 方法测试"""

    def test_compile_returns_lazyframe(self):
        config = _make_config(
            factors=[FactorConfig(name="ma5", expr="ts_mean(close, 5)")],
        )
        executor = ConfigExecutor()
        lf = _make_data()
        result = executor.compile(config, lf)
        assert isinstance(result, pl.LazyFrame)

    def test_compile_with_operations(self):
        config = _make_config(
            factors=[FactorConfig(name="ret", expr="close / ts_lag(close, 1) - 1")],
            operations=[OperationConfig(
                type="section", name="ret_rank", category="rank", inputs=["ret"]
            )],
        )
        executor = ConfigExecutor()
        lf = _make_data()
        result = executor.compile(config, lf)
        collected = result.collect()
        assert "ret_rank" in collected.columns


class TestGetExpressions:
    """get_expressions() 测试"""

    def test_get_expressions_after_run(self):
        config = _make_config(
            factors=[FactorConfig(name="ma5", expr="ts_mean(close, 5)")],
        )
        executor = ConfigExecutor()
        lf = _make_data()
        executor.run(config, lf)
        exprs = executor.get_expressions()
        assert "ma5" in exprs


class TestResolveUniverse:
    """_resolve_universe() 全路径测试"""

    def test_empty_returns_none(self):
        executor = ConfigExecutor()
        assert executor._resolve_universe("") is None

    def test_all_returns_none(self):
        executor = ConfigExecutor()
        assert executor._resolve_universe("all") is None
        assert executor._resolve_universe("ALL") is None
        assert executor._resolve_universe("*") is None

    def test_a_stock_returns_none(self):
        executor = ConfigExecutor()
        assert executor._resolve_universe("A_stock") is None
        assert executor._resolve_universe("a-share") is None
        assert executor._resolve_universe("cn_stock") is None

    def test_comma_separated_codes(self):
        executor = ConfigExecutor()
        result = executor._resolve_universe("000001.SZ,600000.SH")
        assert result == ["000001.SZ", "600000.SH"]

    def test_single_code(self):
        executor = ConfigExecutor()
        result = executor._resolve_universe("000001.SZ")
        assert result == ["000001.SZ"]

    def test_file_path(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("000001.SZ\n600000.SH\n# comment\n")
            tmp_path = f.name
        try:
            executor = ConfigExecutor()
            result = executor._resolve_universe(tmp_path)
            assert result == ["000001.SZ", "600000.SH"]
        finally:
            os.unlink(tmp_path)

    def test_empty_file_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# only comments\n")
            tmp_path = f.name
        try:
            executor = ConfigExecutor()
            result = executor._resolve_universe(tmp_path)
            assert result is None
        finally:
            os.unlink(tmp_path)


class TestRunBacktestDateFiltering:
    """run_backtest() 日期筛选 - String 类型"""

    def test_string_date_filtering(self):
        """日期列是 String 类型时也能正确筛选"""
        data = pl.DataFrame({
            "code": ["A"] * 5,
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
            "close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "volume": [100.0] * 5,
        }).lazy()

        config = _make_config(
            data=DataConfig(date_column="date", code_column="code"),
            factors=[FactorConfig(name="ma2", expr="ts_mean(close, 2)")],
            backtest=BacktestConfig(
                start_date="2023-01-03", end_date="2023-01-05",
                initial_cash=100000, commission=0.001, slippage=0.001,
                signals={"buy_threshold": 0.6, "sell_threshold": 0.4},
            ),
        )

        executor = ConfigExecutor()
        result = executor.run_backtest(config, data)
        # Should not raise, date filtering works with String type
        assert result.status == "success"

    def test_date_column_with_universe_filter(self):
        """universe 过滤 + 日期筛选"""
        data = pl.DataFrame({
            "code": ["A", "A", "B", "B"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
            "close": [10.0, 11.0, 20.0, 21.0],
            "volume": [100.0] * 4,
        }).lazy()

        config = _make_config(
            data=DataConfig(date_column="date", code_column="code"),
            factors=[FactorConfig(name="ret", expr="close / ts_lag(close, 1) - 1")],
            backtest=BacktestConfig(
                start_date="2023-01-01", end_date="2023-01-02",
                initial_cash=100000, commission=0.001, slippage=0.001,
                universe="A",
                signals={"buy_threshold": 0.6, "sell_threshold": 0.4},
            ),
        )

        executor = ConfigExecutor()
        result = executor.run_backtest(config, data)
        assert result.status == "success"


class TestParseFuncArgs:
    """_parse_func_args() keyword 参数"""

    def test_keyword_args(self):
        executor = ConfigExecutor()
        args, kwargs = executor._parse_func_args("close, timeperiod=14")
        assert len(args) == 1
        assert kwargs["timeperiod"] == 14

    def test_multiple_keyword_args(self):
        executor = ConfigExecutor()
        args, kwargs = executor._parse_func_args("close, lower=0.01, upper=0.99")
        assert len(args) == 1
        assert kwargs["lower"] == 0.01
        assert kwargs["upper"] == 0.99

    def test_empty_args(self):
        executor = ConfigExecutor()
        args, kwargs = executor._parse_func_args("")
        assert args == []
        assert kwargs == {}


class TestParseValue:
    """_parse_value() 引号字符串"""

    def test_quoted_string(self):
        executor = ConfigExecutor()
        result = executor._parse_value('"hello"')
        assert result == "hello"

    def test_single_quoted_string(self):
        executor = ConfigExecutor()
        result = executor._parse_value("'world'")
        assert result == "world"

    def test_integer(self):
        executor = ConfigExecutor()
        result = executor._parse_value("42")
        assert result == 42

    def test_float(self):
        executor = ConfigExecutor()
        result = executor._parse_value("3.14")
        assert result == 3.14

    def test_column_reference(self):
        executor = ConfigExecutor()
        result = executor._parse_value("close")
        assert isinstance(result, pl.Expr)


class TestApplyOperatorEmptyInputs:
    """_apply_operator() 空输入"""

    def test_empty_inputs_returns_lit(self):
        executor = ConfigExecutor()
        from QuantNodes.agent.config.types import OperationConfig
        op = OperationConfig(type="section", name="test", category="rank", inputs=[])
        result = executor._apply_operator(op)
        assert isinstance(result, pl.Expr)
