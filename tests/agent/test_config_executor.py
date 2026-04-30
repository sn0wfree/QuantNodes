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
    ValidationConfig,
    OutputConfig,
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


class TestTsOperatorsExtended:
    """扩展时间序列算子测试"""

    def _make_data(self):
        return pl.LazyFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03",
                      "2024-01-04", "2024-01-05"],
            "code": ["A"] * 5,
            "close": [100.0, 102.0, 101.0, 103.0, 105.0],
            "open": [99.0, 100.0, 102.0, 101.0, 103.0],
            "volume": [1000, 1200, 1100, 1300, 1400],
            "high": [101.0, 103.0, 102.0, 104.0, 106.0],
        })

    def test_ts_prod(self):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="time_series", name="prod_val",
                    category="ts_prod", inputs=["close"],
                    params={"window": 3}
                )
            ]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "prod_val" in result.factors

    def test_ts_median(self):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="time_series", name="med_val",
                    category="ts_median", inputs=["close"],
                    params={"window": 3}
                )
            ]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "med_val" in result.factors

    def test_ts_corr_two_inputs(self):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="time_series", name="corr_val",
                    category="ts_corr", inputs=["close", "volume"],
                    params={"window": 3}
                )
            ]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "corr_val" in result.factors

    def test_ts_cov_two_inputs(self):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="time_series", name="cov_val",
                    category="ts_cov", inputs=["close", "volume"],
                    params={"window": 3}
                )
            ]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "cov_val" in result.factors

    def test_ewm_mean(self):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="time_series", name="ewm_val",
                    category="ewm_mean", inputs=["close"],
                    params={"alpha": 0.5}
                )
            ]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "ewm_val" in result.factors

    def test_ewm_std(self):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="time_series", name="ewm_sd",
                    category="ewm_std", inputs=["close"],
                    params={"alpha": 0.5}
                )
            ]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "ewm_sd" in result.factors

    def test_ewm_corr_two_inputs(self):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="time_series", name="ewm_c",
                    category="ewm_corr", inputs=["close", "volume"],
                    params={"alpha": 0.5}
                )
            ]
        )
        result = executor.run(config, self._make_data())
        assert result.status == "success"
        assert "ewm_c" in result.factors


class TestToYamlExtended:
    """to_yaml 扩展测试"""

    def test_to_yaml_includes_validation(self, tmp_path):
        from QuantNodes.agent.config.loader import ConfigLoader

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="f1", expr="close")],
            validation=ValidationConfig(
                run_tests=True,
                test_files=["tests/test_*.py"],
                metrics={"ic_threshold": 0.02},
                custom_operators=["my_op"],
            ),
        )
        yaml_path = str(tmp_path / "test.yaml")
        loader = ConfigLoader()
        loader.to_yaml(config, yaml_path)

        with open(yaml_path) as f:
            import yaml
            data = yaml.safe_load(f)

        assert "validation" in data
        assert data["validation"]["run_tests"] is True
        assert data["validation"]["test_files"] == ["tests/test_*.py"]
        assert data["validation"]["metrics"] == {"ic_threshold": 0.02}
        assert data["validation"]["custom_operators"] == ["my_op"]

    def test_to_yaml_includes_output(self, tmp_path):
        from QuantNodes.agent.config.loader import ConfigLoader

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="f1", expr="close")],
            output=OutputConfig(
                format="parquet",
                path="outputs/result.parquet",
                save_signals=True,
                save_positions=False,
                save_equity_curve=True,
            ),
        )
        yaml_path = str(tmp_path / "test.yaml")
        loader = ConfigLoader()
        loader.to_yaml(config, yaml_path)

        with open(yaml_path) as f:
            import yaml
            data = yaml.safe_load(f)

        assert "output" in data
        assert data["output"]["format"] == "parquet"
        assert data["output"]["save_signals"] is True
        assert data["output"]["save_positions"] is False

    def test_to_yaml_includes_backtest_signals(self, tmp_path):
        from QuantNodes.agent.config.loader import ConfigLoader

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="f1", expr="close")],
            backtest=BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-12-31",
                signals={"buy_threshold": 0.05},
                positions={"max_positions": 10},
            ),
        )
        yaml_path = str(tmp_path / "test.yaml")
        loader = ConfigLoader()
        loader.to_yaml(config, yaml_path)

        with open(yaml_path) as f:
            import yaml
            data = yaml.safe_load(f)

        assert data["backtest"]["signals"] == {"buy_threshold": 0.05}
        assert data["backtest"]["positions"] == {"max_positions": 10}


class TestCheckCoverageExtended:
    """check_coverage 扩展测试"""

    def test_check_coverage_composite_formula_valid(self):
        from QuantNodes.agent.config.loader import ConfigLoader

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            composite=[
                CompositeConfig(name="alpha", formula="rank(ret)")
            ]
        )
        loader = ConfigLoader()
        report = loader.check_coverage(config)
        assert report.is_complete

    def test_check_coverage_composite_unknown_func(self):
        from QuantNodes.agent.config.loader import ConfigLoader

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            composite=[
                CompositeConfig(name="alpha", formula="unknown_op(ret)")
            ]
        )
        loader = ConfigLoader()
        report = loader.check_coverage(config)
        assert not report.is_complete
        assert any("unknown_func:unknown_op" in u for u in report.unresolved)

    def test_check_coverage_composite_unknown_ref(self):
        from QuantNodes.agent.config.loader import ConfigLoader

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            composite=[
                CompositeConfig(name="alpha", formula="nonexistent_col + 1")
            ]
        )
        loader = ConfigLoader()
        report = loader.check_coverage(config)
        assert not report.is_complete
        assert any("unknown_ref:nonexistent_col" in u for u in report.unresolved)

    def test_check_coverage_composite_empty_formula(self):
        from QuantNodes.agent.config.loader import ConfigLoader

        config = StrategyConfig(
            name="test",
            composite=[
                CompositeConfig(name="alpha", formula="")
            ]
        )
        loader = ConfigLoader()
        report = loader.check_coverage(config)
        assert not report.is_complete


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


class TestConfigStrategyNode:
    """ConfigStrategyNode 测试"""

    def test_buy_signal(self):
        import pandas as pd
        from QuantNodes.backtest.config_strategy import ConfigStrategyNode

        df = pd.DataFrame({
            "Code": ["A", "B"],
            "date": ["2024-01-01", "2024-01-01"],
            "Close": [100.0, 200.0],
            "signal": [1, 0],
        })
        node = ConfigStrategyNode(signal_col="signal")
        signals = node._generate_signals(df)

        assert len(signals) == 1
        assert signals[0].code == "A"
        assert signals[0].signal_type == "buy"

    def test_sell_signal(self):
        import pandas as pd
        from QuantNodes.backtest.config_strategy import ConfigStrategyNode

        df = pd.DataFrame({
            "Code": ["A", "B"],
            "date": ["2024-01-01", "2024-01-01"],
            "Close": [100.0, 200.0],
            "signal": [0, -1],
        })
        node = ConfigStrategyNode(signal_col="signal")
        signals = node._generate_signals(df)

        assert len(signals) == 1
        assert signals[0].code == "B"
        assert signals[0].signal_type == "sell"

    def test_hold_signal(self):
        import pandas as pd
        from QuantNodes.backtest.config_strategy import ConfigStrategyNode

        df = pd.DataFrame({
            "Code": ["A", "B"],
            "date": ["2024-01-01", "2024-01-01"],
            "Close": [100.0, 200.0],
            "signal": [0, 0],
        })
        node = ConfigStrategyNode(signal_col="signal")
        signals = node._generate_signals(df)

        assert len(signals) == 0

    def test_column_case_flexibility(self):
        import pandas as pd
        from QuantNodes.backtest.config_strategy import ConfigStrategyNode

        df = pd.DataFrame({
            "code": ["A"],
            "date": ["2024-01-01"],
            "close": [100.0],
            "signal": [1],
        })
        node = ConfigStrategyNode(signal_col="signal")
        signals = node._generate_signals(df)

        assert len(signals) == 1
        assert signals[0].code == "A"


class TestConfigBacktestRunner:
    """ConfigBacktestRunner 测试"""

    def _make_data(self):
        return pl.LazyFrame({
            "date": ["2024-01-01"] * 4 + ["2024-01-02"] * 4,
            "code": ["A", "B", "C", "D"] * 2,
            "close": [100.0, 102.0, 101.0, 103.0, 101.0, 103.0, 102.0, 104.0],
            "open": [99.0, 100.0, 102.0, 101.0, 100.0, 102.0, 101.0, 103.0],
            "volume": [1000, 1200, 1100, 1300, 1100, 1300, 1200, 1400],
        })

    def test_basic_backtest(self):
        from QuantNodes.backtest.config_runner import ConfigBacktestRunner

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            backtest=BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-02",
                initial_cash=1000000,
                commission=0.001,
                slippage=0.001,
                signals={"buy_threshold": 0.005, "sell_threshold": -0.005},
            ),
        )
        runner = ConfigBacktestRunner()
        result = runner.run(config, self._make_data())

        assert result is not None
        assert result.final_cash > 0

    def test_empty_signals(self):
        from QuantNodes.backtest.config_runner import ConfigBacktestRunner

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            backtest=BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-02",
                initial_cash=1000000,
                commission=0.001,
                signals={"buy_threshold": 0.99, "sell_threshold": -0.99},
            ),
        )
        runner = ConfigBacktestRunner()
        result = runner.run(config, self._make_data())

        assert result is not None
        assert result.statistics.get("total_trades", 0) == 0

    def test_no_backtest_config(self):
        from QuantNodes.backtest.config_runner import ConfigBacktestRunner

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
        )
        runner = ConfigBacktestRunner()
        result = runner.run(config, self._make_data())

        assert result is not None
        assert result.final_cash == 0

    def test_with_positions_config(self):
        from QuantNodes.backtest.config_runner import ConfigBacktestRunner

        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="ret", expr="close / open - 1")],
            backtest=BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-02",
                initial_cash=1000000,
                commission=0.001,
                signals={"buy_threshold": 0.005, "sell_threshold": -0.005},
                positions={"max_positions": 2},
            ),
        )
        runner = ConfigBacktestRunner()
        result = runner.run(config, self._make_data())

        assert result is not None


class TestConfigBacktestTool:
    """ConfigBacktestTool 测试"""

    def test_tool_properties(self):
        from QuantNodes.agent.tools.config_backtest import ConfigBacktestTool

        tool = ConfigBacktestTool()
        assert tool.name == "config_backtest"
        assert tool.read_only is False

    def test_execute_without_data(self):
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
            assert result["status"] == "error"
            assert "errors" in result

        asyncio.run(_test())


class TestSecOperatorsExtended:
    """截面算子扩展测试"""

    def _make_data(self):
        return pl.LazyFrame({
            "date": ["2024-01-01"] * 4,
            "code": ["A", "B", "C", "D"],
            "close": [100.0, 102.0, 101.0, 103.0],
            "open": [99.0, 100.0, 102.0, 101.0],
            "volume": [1000, 1200, 1100, 1300],
            "target": [0.01, -0.02, 0.03, -0.01],
            "industry": ["A", "A", "B", "B"],
        })

    def _run_ops(self, data, operations):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="close", expr="close")],
            operations=operations,
        )
        return executor.run(config, data)

    def test_rank_ic(self):
        ops = [OperationConfig(
            type="section", name="my_rank_ic", category="rank_ic",
            inputs=["close", "target"], params={}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "my_rank_ic" in result.factors

    def test_ic(self):
        ops = [OperationConfig(
            type="section", name="my_ic", category="ic",
            inputs=["close", "target"], params={}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "my_ic" in result.factors

    def test_group_norm(self):
        ops = [OperationConfig(
            type="section", name="my_gn", category="group_norm",
            inputs=["close"], params={"group": "industry", "method": "zscore"}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "my_gn" in result.factors

    def test_group_winsorize(self):
        ops = [OperationConfig(
            type="section", name="my_gw", category="group_winsorize",
            inputs=["close"], params={"group": "industry", "lower": 0.05, "upper": 0.05}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "my_gw" in result.factors

    def test_existing_section_ops(self):
        for cat in ["rank", "zscore", "winsorize", "neutralize", "scale", "percentile"]:
            ops = [OperationConfig(
                type="section", name=f"my_{cat}", category=cat,
                inputs=["close"], params={}
            )]
            result = self._run_ops(self._make_data(), ops)
            assert result.status == "success", f"Failed for {cat}"
            assert f"my_{cat}" in result.factors


class TestMathOperators:
    """数学算子测试"""

    def _make_data(self):
        return pl.LazyFrame({
            "date": ["2024-01-01"] * 3,
            "code": ["A", "B", "C"],
            "close": [100.0, 102.0, 101.0],
            "volume": [1000, 1200, 1100],
        })

    def _run_ops(self, data, operations):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test",
            factors=[FactorConfig(name="close", expr="close")],
            operations=operations,
        )
        return executor.run(config, data)

    def test_log1p(self):
        ops = [OperationConfig(
            type="math", name="m_log1p", category="log1p",
            inputs=["close"], params={}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_log1p" in result.factors

    def test_sqrt(self):
        ops = [OperationConfig(
            type="math", name="m_sqrt", category="sqrt",
            inputs=["close"], params={}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_sqrt" in result.factors

    def test_sign(self):
        ops = [OperationConfig(
            type="math", name="m_sign", category="sign",
            inputs=["close"], params={}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_sign" in result.factors

    def test_clip(self):
        ops = [OperationConfig(
            type="math", name="m_clip", category="clip",
            inputs=["close"], params={"lower": 100.5, "upper": 101.5}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_clip" in result.factors

    def test_floor(self):
        ops = [OperationConfig(
            type="math", name="m_floor", category="floor",
            inputs=["close"], params={}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_floor" in result.factors

    def test_ceil(self):
        ops = [OperationConfig(
            type="math", name="m_ceil", category="ceil",
            inputs=["close"], params={}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_ceil" in result.factors

    def test_round(self):
        ops = [OperationConfig(
            type="math", name="m_round", category="round",
            inputs=["close"], params={"decimals": 0}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_round" in result.factors

    def test_nan_to_null(self):
        ops = [OperationConfig(
            type="math", name="m_ntn", category="nan_to_null",
            inputs=["close"], params={}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_ntn" in result.factors

    def test_fill_null(self):
        ops = [OperationConfig(
            type="math", name="m_fn", category="fill_null",
            inputs=["close"], params={"value": 0.0}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_fn" in result.factors

    def test_fill_zero(self):
        ops = [OperationConfig(
            type="math", name="m_fz", category="fill_zero",
            inputs=["close"], params={}
        )]
        result = self._run_ops(self._make_data(), ops)
        assert result.status == "success"
        assert "m_fz" in result.factors

    def test_trig_functions(self):
        for cat in ["sin", "cos", "tan", "arcsin", "arccos", "arctan"]:
            ops = [OperationConfig(
                type="math", name=f"m_{cat}", category=cat,
                inputs=["close"], params={}
            )]
            result = self._run_ops(self._make_data(), ops)
            assert result.status == "success", f"Failed for {cat}"
            assert f"m_{cat}" in result.factors

    def test_existing_math_ops(self):
        for cat in ["add", "sub", "mul", "div", "log", "abs", "pow"]:
            ops = [OperationConfig(
                type="math", name=f"m_{cat}", category=cat,
                inputs=["close"], params={"value": 2.0, "exponent": 2}
            )]
            result = self._run_ops(self._make_data(), ops)
            assert result.status == "success", f"Failed for {cat}"
            assert f"m_{cat}" in result.factors


class TestCompositeOperatorsExtended:
    """组合算子扩展测试"""

    def _make_data(self):
        return pl.LazyFrame({
            "date": ["2024-01-01"] * 3,
            "code": ["A", "B", "C"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [3.0, 2.0, 1.0],
            "close": [100.0, 102.0, 101.0],
        })

    def _run_ops(self, data, factors, operations):
        executor = ConfigExecutor()
        config = StrategyConfig(
            name="test", factors=factors, operations=operations,
        )
        return executor.run(config, data)

    def test_abs_max(self):
        ops = [OperationConfig(
            type="composite", name="c_absmax", category="abs_max",
            inputs=["f1", "f2"], params={}
        )]
        result = self._run_ops(self._make_data(), [
            FactorConfig(name="f1", expr="f1"), FactorConfig(name="f2", expr="f2"),
        ], ops)
        assert result.status == "success"
        assert "c_absmax" in result.factors

    def test_combine_sum(self):
        ops = [OperationConfig(
            type="composite", name="c_combine", category="combine",
            inputs=["f1", "f2"], params={"method": "sum"}
        )]
        result = self._run_ops(self._make_data(), [
            FactorConfig(name="f1", expr="f1"), FactorConfig(name="f2", expr="f2"),
        ], ops)
        assert result.status == "success"
        assert "c_combine" in result.factors

    def test_select_top(self):
        ops = [OperationConfig(
            type="composite", name="c_stop", category="select_top",
            inputs=["f1"], params={"n": 2, "ascending": False}
        )]
        result = self._run_ops(self._make_data(), [
            FactorConfig(name="f1", expr="f1"),
        ], ops)
        assert result.status == "success"
        assert "c_stop" in result.factors

    def test_filter_positive(self):
        ops = [OperationConfig(
            type="composite", name="c_fp", category="filter_positive",
            inputs=["f1"], params={}
        )]
        result = self._run_ops(self._make_data(), [
            FactorConfig(name="f1", expr="f1"),
        ], ops)
        assert result.status == "success"
        assert "c_fp" in result.factors

    def test_filter_negative(self):
        ops = [OperationConfig(
            type="composite", name="c_fn", category="filter_negative",
            inputs=["f1"], params={}
        )]
        result = self._run_ops(self._make_data(), [
            FactorConfig(name="f1", expr="f1"),
        ], ops)
        assert result.status == "success"
        assert "c_fn" in result.factors

    def test_abs_filter(self):
        ops = [OperationConfig(
            type="composite", name="c_af", category="abs_filter",
            inputs=["f1"], params={"threshold": 1.5}
        )]
        result = self._run_ops(self._make_data(), [
            FactorConfig(name="f1", expr="f1"),
        ], ops)
        assert result.status == "success"
        assert "c_af" in result.factors

    def test_rank_sort(self):
        ops = [OperationConfig(
            type="composite", name="c_rs", category="rank_sort",
            inputs=["f1", "f2"], params={"weights": [0.6, 0.4]}
        )]
        result = self._run_ops(self._make_data(), [
            FactorConfig(name="f1", expr="f1"), FactorConfig(name="f2", expr="f2"),
        ], ops)
        assert result.status == "success"
        assert "c_rs" in result.factors

    def test_existing_composite_ops(self):
        ops = [OperationConfig(
            type="composite", name="c_ws", category="weighted_sum",
            inputs=["f1", "f2"], params={"weights": [0.5, 0.5]}
        )]
        result = self._run_ops(self._make_data(), [
            FactorConfig(name="f1", expr="f1"), FactorConfig(name="f2", expr="f2"),
        ], ops)
        assert result.status == "success"
        assert "c_ws" in result.factors
