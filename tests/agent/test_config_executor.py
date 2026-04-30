# coding=utf-8
"""
ConfigExecutor 单元测试
"""

import asyncio
import pytest
import polars as pl

from QuantNodes.agent.config.executor import ConfigExecutor, ExprParser
from QuantNodes.agent.config.types import (
    StrategyConfig,
    FactorConfig,
    OperationConfig,
    CompositeConfig,
    BacktestConfig,
)


class TestExprParser:
    """表达式解析器测试"""
    
    def test_simple_column(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("close")
        assert expr is not None
    
    def test_number_literal(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("20")
        assert expr is not None
    
    def test_float_literal(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("3.14")
        assert expr is not None
    
    def test_addition(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("close + volume")
        assert expr is not None
    
    def test_subtraction(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("close - open")
        assert expr is not None
    
    def test_multiplication(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("close * 2")
        assert expr is not None
    
    def test_division(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("close / open")
        assert expr is not None
    
    def test_complex_arithmetic(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("close / close.shift(20) - 1")
        assert expr is not None
    
    def test_unary_negation(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("-rank(close)")
        assert expr is not None
    
    def test_parentheses(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("(close + volume) / 2")
        assert expr is not None
    
    def test_nested_parentheses(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("((close + volume) / 2) - 1")
        assert expr is not None
    
    def test_ts_lag_call(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("ts_lag(close, 20)")
        assert expr is not None
    
    def test_ts_mean_call(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("ts_mean(close, 20)")
        assert expr is not None
    
    def test_method_chain(self):
        executor = ConfigExecutor()
        expr = executor._parse_expr("close.shift(20)")
        assert expr is not None


class TestConfigExecutor:
    """ConfigExecutor 测试"""
    
    def _make_config(self, factors=None, operations=None, composite=None, backtest=None):
        return StrategyConfig(
            name="test",
            factors=factors or [],
            operations=operations or [],
            composite=composite or [],
            backtest=backtest,
        )
    
    def _make_data(self):
        return pl.LazyFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "code": ["A", "A", "A"],
            "close": [100.0, 102.0, 101.0],
            "open": [99.0, 100.0, 102.0],
            "volume": [1000, 1200, 1100],
        })
    
    def test_run_simple_factor(self):
        executor = ConfigExecutor()
        config = self._make_config(
            factors=[FactorConfig(name="ret", expr="close / open - 1")]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "ret" in result.factors
    
    def test_run_preserves_original_columns(self):
        executor = ConfigExecutor()
        config = self._make_config(
            factors=[FactorConfig(name="ret", expr="close / open - 1")]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert hasattr(result, "data")
        assert "date" in result.data.columns
        assert "code" in result.data.columns
        assert "close" in result.data.columns
        assert "ret" in result.data.columns
    
    def test_run_with_operations(self):
        executor = ConfigExecutor()
        config = self._make_config(
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            operations=[
                OperationConfig(
                    type="time_series",
                    name="ret_ma",
                    category="ts_mean",
                    inputs=["ret"],
                    params={"window": 5}
                )
            ]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "ret_ma" in result.factors
    
    def test_run_with_composite(self):
        executor = ConfigExecutor()
        config = self._make_config(
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            composite=[
                CompositeConfig(name="alpha", formula="ret * 2")
            ]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "alpha" in result.factors
    
    def test_run_backtest_generates_signals(self):
        executor = ConfigExecutor()
        config = self._make_config(
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            backtest=BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_cash=1000000,
                commission=0.001,
                signals={"buy_threshold": 0.01, "sell_threshold": -0.01}
            )
        )
        result = executor.run_backtest(config, self._make_data())
        assert result.status == "success"
        assert result.backtest is not None
        assert "signals" in result.backtest
        assert "config" in result.backtest
    
    def test_run_backtest_compatible_threshold_names(self):
        executor = ConfigExecutor()
        config = self._make_config(
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            backtest=BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                initial_cash=1000000,
                commission=0.001,
                signals={"long_threshold": 0.02, "short_threshold": -0.02}
            )
        )
        result = executor.run_backtest(config, self._make_data())
        assert result.status == "success"
        assert result.backtest is not None
        assert result.backtest["buy_threshold"] == 0.02
        assert result.backtest["sell_threshold"] == -0.02


class TestConfigLoader:
    """ConfigLoader 测试"""
    
    def test_load_momentum_yaml(self):
        from QuantNodes.agent.config.loader import ConfigLoader
        from pathlib import Path
        
        templates_dir = Path(__file__).parent.parent.parent / "QuantNodes" / "agent" / "config" / "templates"
        yaml_path = templates_dir / "momentum.yaml"
        
        if yaml_path.exists():
            loader = ConfigLoader()
            config = loader.load(str(yaml_path))
            assert config.name == "momentum_20d"
            assert len(config.factors) > 0
            assert config.backtest is not None
    
    def test_load_mean_reversion_yaml(self):
        from QuantNodes.agent.config.loader import ConfigLoader
        from pathlib import Path
        
        templates_dir = Path(__file__).parent.parent.parent / "QuantNodes" / "agent" / "config" / "templates"
        yaml_path = templates_dir / "mean_reversion.yaml"
        
        if yaml_path.exists():
            loader = ConfigLoader()
            config = loader.load(str(yaml_path))
            assert config.name == "mean_reversion"
            assert len(config.factors) > 0
    
    def test_check_coverage(self):
        from QuantNodes.agent.config.loader import ConfigLoader
        
        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="f1", expr="close")],
            operations=[
                OperationConfig(
                    type="time_series",
                    name="f1_ma",
                    category="ts_mean",
                    inputs=["f1"],
                    params={"window": 20}
                )
            ]
        )
        
        loader = ConfigLoader()
        report = loader.check_coverage(config)
        assert report.is_complete


class TestConfigCodeGenerator:
    """ConfigCodeGenerator 测试"""
    
    def test_generate_imports(self):
        from QuantNodes.agent.config.generator import ConfigCodeGenerator
        
        generator = ConfigCodeGenerator()
        config = StrategyConfig(name="test")
        code = generator.generate(config)
        
        assert "import pandas as pd" in code
        assert "from QuantNodes.backtest.strategy_node import StrategyNode" in code
    
    def test_generate_with_factors(self):
        from QuantNodes.agent.config.generator import ConfigCodeGenerator
        
        generator = ConfigCodeGenerator()
        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="ret", expr="close / ts_lag(close, 20) - 1")]
        )
        code = generator.generate(config)
        
        assert 'quote_data["ret"]' in code
    
    def test_generate_with_backtest(self):
        from QuantNodes.agent.config.generator import ConfigCodeGenerator
        
        generator = ConfigCodeGenerator()
        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            backtest=BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-12-31",
                initial_cash=1000000,
                commission=0.001,
            )
        )
        code = generator.generate(config)
        
        assert "ConfigStrategy" in code
        assert "SimulatedBrokerNode" in code


class TestConfigBacktestTool:
    """ConfigBacktestTool 测试"""
    
    def test_tool_properties(self):
        from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
        
        tool = ConfigBacktestTool()
        assert tool.name == "config_backtest"
        assert tool.read_only is False
    
    def test_execute_with_yaml(self):
        from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool
        
        async def _test():
            tool = ConfigBacktestTool()
            yaml_str = """
name: "test_strategy"
factors:
  - name: ret
    expr: "close / open - 1"
"""
            result = await tool.execute(config_yaml=yaml_str)
            assert "status" in result
            assert "config_info" in result
        
        asyncio.run(_test())
