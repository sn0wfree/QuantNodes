# coding=utf-8
"""
TA-Lib 算子测试

覆盖:
- operators/talib.py: 包装层直接调用
- factor_functions.py: 注册表注册
- executor.py: YAML 配置驱动
"""

import pytest
import polars as pl

from QuantNodes.operators.talib import TaLibOperators


@pytest.fixture
def sample_df():
    """生成足够的测试数据（30 天）"""
    n = 30
    return pl.DataFrame({
        "date": [f"2024-01-{i+1:02d}" for i in range(n)],
        "code": ["A"] * n,
        "open": [100.0 + i * 0.5 for i in range(n)],
        "high": [102.0 + i * 0.5 for i in range(n)],
        "low": [98.0 + i * 0.5 for i in range(n)],
        "close": [100.0 + i for i in range(n)],
        "volume": [1000 + i * 100 for i in range(n)],
    })


class TestTaLibOperators:
    """TaLibOperators 包装层测试"""

    def test_sma(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.sma("close", timeperiod=5).alias("sma")
        )
        assert "sma" in result.columns
        vals = result["sma"].to_list()
        import math
        assert math.isnan(vals[0])  # NaN for first values
        assert vals[4] is not None  # 第 5 个值应有结果

    def test_ema(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.ema("close", timeperiod=5).alias("ema")
        )
        assert "ema" in result.columns
        assert result["ema"].to_list()[4] is not None

    def test_rsi(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.rsi("close", timeperiod=14).alias("rsi")
        )
        assert "rsi" in result.columns
        assert result["rsi"].to_list()[13] is not None

    def test_macd_line(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.macd_line("close").alias("macd")
        )
        assert "macd" in result.columns

    def test_macd_signal(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.macd_signal("close").alias("signal")
        )
        assert "signal" in result.columns

    def test_macd_hist(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.macd_hist("close").alias("hist")
        )
        assert "hist" in result.columns

    def test_bbands_upper(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.bbands_upper("close", timeperiod=5).alias("bb_upper")
        )
        assert "bb_upper" in result.columns
        assert result["bb_upper"].to_list()[4] is not None

    def test_bbands_middle(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.bbands_middle("close", timeperiod=5).alias("bb_mid")
        )
        assert "bb_mid" in result.columns

    def test_bbands_lower(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.bbands_lower("close", timeperiod=5).alias("bb_lower"),
            TaLibOperators.bbands_middle("close", timeperiod=5).alias("bb_mid"),
            TaLibOperators.bbands_upper("close", timeperiod=5).alias("bb_upper"),
        )
        assert "bb_lower" in result.columns
        # lower < middle < upper
        mid = result["bb_mid"].to_list()
        lower = result["bb_lower"].to_list()
        upper = result["bb_upper"].to_list()
        for m, lo, hi in zip(mid[-3:], lower[-3:], upper[-3:]):
            assert lo < m < hi

    def test_cci(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.cci("close", timeperiod=14).alias("cci")
        )
        assert "cci" in result.columns

    def test_willr(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.willr("close", timeperiod=14).alias("willr")
        )
        assert "willr" in result.columns

    def test_roc(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.roc("close", timeperiod=5).alias("roc")
        )
        assert "roc" in result.columns
        assert result["roc"].to_list()[4] is not None

    def test_mom(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.mom("close", timeperiod=5).alias("mom")
        )
        assert "mom" in result.columns

    def test_adx(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.adx("close", timeperiod=14).alias("adx")
        )
        assert "adx" in result.columns

    def test_atr(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.atr("close", timeperiod=14).alias("atr")
        )
        assert "atr" in result.columns

    def test_obv(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.obv("close").alias("obv")
        )
        assert "obv" in result.columns

    def test_stddev(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.stddev("close", timeperiod=5).alias("std")
        )
        assert "std" in result.columns
        assert result["std"].to_list()[4] is not None

    def test_linearreg(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.linearreg("close", timeperiod=10).alias("lr")
        )
        assert "lr" in result.columns

    def test_cdl_doji(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.cdl_doji("close").alias("doji")
        )
        assert "doji" in result.columns

    def test_cdl_hammer(self, sample_df):
        result = sample_df.with_columns(
            TaLibOperators.cdl_hammer("close").alias("hammer")
        )
        assert "hammer" in result.columns


class TestTaLibRegistry:
    """TA-Lib 算子注册表测试"""

    def test_talib_category_exists(self):
        from QuantNodes.factor_node.factor_functions import OperatorCategory, _OPERATOR_REGISTRY
        assert OperatorCategory.TALIB in _OPERATOR_REGISTRY

    def test_talib_operators_registered(self):
        from QuantNodes.factor_node.factor_functions import list_operators, OperatorCategory
        ops = list_operators(OperatorCategory.TALIB)
        assert len(ops) > 50

    def test_talib_rsi_in_registry(self):
        from QuantNodes.factor_node.factor_functions import operator_info
        info = operator_info("talib_rsi")
        assert info is not None
        assert info["category"] == "talib"

    def test_talib_sma_in_registry(self):
        from QuantNodes.factor_node.factor_functions import get_operator
        op = get_operator("talib_sma")
        assert op is not None

    def test_talib_operators_callable(self):
        from QuantNodes.factor_node.factor_functions import get_operator
        import polars as pl

        df = pl.DataFrame({"close": [100.0 + i for i in range(20)]})
        rsi_func = get_operator("talib_rsi")
        result = df.with_columns(rsi_func("close", timeperiod=5).alias("rsi"))
        assert "rsi" in result.columns


class TestTaLibExecutor:
    """TA-Lib 算子通过 executor YAML 配置测试"""

    def test_executor_talib_rsi(self):
        from QuantNodes.agent.config.executor import ConfigExecutor
        from QuantNodes.agent.config.types import StrategyConfig, OperationConfig

        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="talib", name="rsi_14",
                    category="talib_rsi",
                    inputs=["close"],
                    params={"timeperiod": 14},
                ),
            ],
        )
        data = pl.LazyFrame({
            "date": [f"2024-01-{i+1:02d}" for i in range(30)],
            "code": ["A"] * 30,
            "close": [100.0 + i for i in range(30)],
            "open": [99.0 + i for i in range(30)],
            "volume": [1000 + i * 100 for i in range(30)],
        })
        executor = ConfigExecutor()
        result = executor.run(config, data)
        assert result.status == "success"
        df = result.data.collect()
        assert "rsi_14" in df.columns
        assert df["rsi_14"].to_list()[13] is not None

    def test_executor_talib_sma(self):
        from QuantNodes.agent.config.executor import ConfigExecutor
        from QuantNodes.agent.config.types import StrategyConfig, OperationConfig

        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="talib", name="sma_20",
                    category="talib_sma",
                    inputs=["close"],
                    params={"timeperiod": 5},
                ),
            ],
        )
        data = pl.LazyFrame({
            "date": [f"2024-01-{i+1:02d}" for i in range(20)],
            "code": ["A"] * 20,
            "close": [100.0 + i for i in range(20)],
            "open": [99.0 + i for i in range(20)],
            "volume": [1000 + i * 100 for i in range(20)],
        })
        executor = ConfigExecutor()
        result = executor.run(config, data)
        assert result.status == "success"
        df = result.data.collect()
        assert "sma_20" in df.columns
        assert df["sma_20"].to_list()[4] is not None

    def test_executor_talib_bbands(self):
        from QuantNodes.agent.config.executor import ConfigExecutor
        from QuantNodes.agent.config.types import StrategyConfig, OperationConfig

        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="talib", name="bb_upper",
                    category="talib_bbands_upper",
                    inputs=["close"],
                    params={"timeperiod": 5},
                ),
                OperationConfig(
                    type="talib", name="bb_lower",
                    category="talib_bbands_lower",
                    inputs=["close"],
                    params={"timeperiod": 5},
                ),
            ],
        )
        data = pl.LazyFrame({
            "date": [f"2024-01-{i+1:02d}" for i in range(20)],
            "code": ["A"] * 20,
            "close": [100.0 + i for i in range(20)],
            "open": [99.0 + i for i in range(20)],
            "volume": [1000 + i * 100 for i in range(20)],
        })
        executor = ConfigExecutor()
        result = executor.run(config, data)
        assert result.status == "success"
        df = result.data.collect()
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        # upper > lower
        upper_vals = df["bb_upper"].to_list()
        lower_vals = df["bb_lower"].to_list()
        for u, l in zip(upper_vals[-3:], lower_vals[-3:]):
            assert u > l

    def test_executor_talib_multiple_ops(self):
        from QuantNodes.agent.config.executor import ConfigExecutor
        from QuantNodes.agent.config.types import StrategyConfig, OperationConfig

        config = StrategyConfig(
            name="test",
            operations=[
                OperationConfig(
                    type="talib", name="sma_10",
                    category="talib_sma",
                    inputs=["close"],
                    params={"timeperiod": 10},
                ),
                OperationConfig(
                    type="talib", name="rsi_14",
                    category="talib_rsi",
                    inputs=["close"],
                    params={"timeperiod": 14},
                ),
                OperationConfig(
                    type="talib", name="bb_upper",
                    category="talib_bbands_upper",
                    inputs=["close"],
                    params={"timeperiod": 5},
                ),
            ],
        )
        data = pl.LazyFrame({
            "date": [f"2024-01-{i+1:02d}" for i in range(30)],
            "code": ["A"] * 30,
            "close": [100.0 + i for i in range(30)],
            "open": [99.0 + i for i in range(30)],
            "volume": [1000 + i * 100 for i in range(30)],
        })
        executor = ConfigExecutor()
        result = executor.run(config, data)
        assert result.status == "success"
        df = result.data.collect()
        assert "sma_10" in df.columns
        assert "rsi_14" in df.columns
        assert "bb_upper" in df.columns
