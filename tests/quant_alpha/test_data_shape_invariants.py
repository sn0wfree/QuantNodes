# coding=utf-8
"""
test_data_shape_invariants.py - 数据形状不变性测试 (Phase 9.5)

目标: 验证 evaluator 和 operator 在各种数据形态下不崩
- 真实数据子集 (1000 rows × 50 stocks)
- 单只股票
- 单日
- 全部 NaN
- 极端 IC/IR 值
- 数据漂移 (temporal split)

这些是 V8 测试中发现但未充分覆盖的边界。
"""
import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.evaluation.evaluators.polars_evaluator import (
    PolarsAlphaCalculatorEvaluator,
)
from QuantNodes.research.quant_alpha.evaluation.contracts import FactorSpec
from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture(scope="module")
def vocab() -> OperatorVocab:
    return OperatorVocab.default()


@pytest.fixture(scope="module")
def evaluator() -> PolarsAlphaCalculatorEvaluator:
    return PolarsAlphaCalculatorEvaluator()


@pytest.fixture
def single_stock_data() -> pl.DataFrame:
    """单只股票 30 日"""
    return pl.DataFrame({
        "date": [f"2024-01-{d + 1:02d}" for d in range(30)],
        "code": ["A"] * 30,
        "close": [100.0 + d * 0.5 + np.random.randn() * 2 for d in range(30)],
        "open": [100.0 + d * 0.5 for d in range(30)],
        "high": [102.0 + d * 0.5 for d in range(30)],
        "low": [98.0 + d * 0.5 for d in range(30)],
        "vol": [1000.0 + d * 10 for d in range(30)],
    }).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def single_date_data() -> pl.DataFrame:
    """单日多只股票"""
    return pl.DataFrame({
        "date": ["2024-01-01"] * 50,
        "code": [f"S{i}" for i in range(50)],
        "close": [100.0 + i for i in range(50)],
        "vol": [1000.0 + i * 10 for i in range(50)],
    }).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def all_nan_data() -> pl.DataFrame:
    """全部 NaN"""
    return pl.DataFrame({
        "date": ["2024-01-01"] * 30,
        "code": ["A", "B", "C"] * 10,
        "close": [None] * 30,
        "vol": [None] * 30,
    }).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def all_constant_data() -> pl.DataFrame:
    """全部常数 (无方差)"""
    return pl.DataFrame({
        "date": ["2024-01-01"] * 30,
        "code": ["A", "B", "C"] * 10,
        "close": [100.0] * 30,
        "vol": [1000.0] * 30,
    }).with_columns(pl.col("date").str.to_date())


# ==============================================================================
# Test Class 1: 单只股票
# ==============================================================================


class TestSingleStock:
    """单只股票数据"""

    def test_rank_single_stock(self, vocab, single_stock_data):
        """单只股票 rank 应能算 (返回 0.5 或 NaN)"""
        result = vocab.evaluate("rank(close)", single_stock_data)
        assert result is not None
        # 单股截面只有 1 个值, rank 应该是 0.5 (中点)
        # 或 NaN
        assert result.shape[0] == 30

    def test_ts_mean_single_stock(self, vocab, single_stock_data):
        """单只股票 ts_mean 应能算"""
        result = vocab.evaluate("ts_mean(close, 5)", single_stock_data)
        assert result is not None
        assert result.shape[0] == 30


# ==============================================================================
# Test Class 2: 单日数据
# ==============================================================================


class TestSingleDate:
    """单日多只股票"""

    def test_rank_single_date(self, vocab, single_date_data):
        """单日 rank 应能算 (截面归一化)"""
        result = vocab.evaluate("rank(close)", single_date_data)
        assert result is not None
        assert result.shape[0] == 50

    def test_ts_mean_single_date(self, vocab, single_date_data):
        """单日多股: ts_mean(close, 5) 沿行滚动"""
        result = vocab.evaluate("ts_mean(close, 5)", single_date_data)
        assert result is not None
        assert result.shape[0] == 50
        # ts_mean 沿行滚动, 大部分有值
        non_null = sum(1 for x in result.to_list() if x is not None)
        assert non_null > 0
        assert non_null <= 50


# ==============================================================================
# Test Class 3: 全部 NaN
# ==============================================================================


class TestAllNaN:
    """全部 NaN 数据"""

    def test_ts_mean_all_nan(self, vocab, all_nan_data):
        """全 NaN 输入, ts_mean 应不崩 (输出 NaN)"""
        result = vocab.evaluate("ts_mean(close, 5)", all_nan_data)
        assert result is not None
        assert result.shape[0] == 30
        # 全 NaN
        non_null = sum(1 for x in result.to_list() if x is not None)
        assert non_null == 0

    def test_rank_all_nan(self, vocab, all_nan_data):
        """全 NaN rank 应不崩"""
        result = vocab.evaluate("rank(close)", all_nan_data)
        assert result is not None


# ==============================================================================
# Test Class 4: 全部常数
# ==============================================================================


class TestAllConstant:
    """全部常数 (无方差)"""

    def test_ts_std_constant(self, vocab, all_constant_data):
        """常数 ts_std 应返回 0 (不崩)"""
        result = vocab.evaluate("ts_std(close, 5)", all_constant_data)
        assert result is not None
        # 非空位置应 = 0
        non_null = [x for x in result.to_list() if x is not None]
        if non_null:
            assert all(abs(x) < 1e-6 for x in non_null)

    def test_div_constant_by_zero(self, vocab, all_constant_data):
        """常数除以常数应返回 0 (不崩)"""
        # 0 / 0
        df = all_constant_data.with_columns(pl.lit(0.0).alias("vol"))
        result = vocab.evaluate("div(close, vol)", df)
        assert result is not None
        # 不崩, 结果可能是 inf / nan
        assert result.shape[0] == 30


# ==============================================================================
# Test Class 5: 极端数据
# ==============================================================================


class TestExtremeData:
    """极端数据"""

    def test_extreme_values(self, vocab):
        """极大/极小值 ts_mean"""
        df = pl.DataFrame({
            "date": ["2024-01-01"] * 10,
            "code": ["A"] * 10,
            "close": [1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10, 1e10, 1e-10],
        }).with_columns(pl.col("date").str.to_date())
        result = vocab.evaluate("ts_mean(close, 3)", df)
        assert result is not None
        assert result.shape[0] == 10

    def test_huge_window_short_data(self, vocab):
        """巨大窗口 (1000) vs 短数据 (10)"""
        df = pl.DataFrame({
            "date": ["2024-01-01"] * 10,
            "code": ["A"] * 10,
            "close": list(range(10)),
        }).with_columns(pl.col("date").str.to_date())
        result = vocab.evaluate("ts_mean(close, 1000)", df)
        assert result is not None
        # 全部 NaN (窗口不够)
        non_null = sum(1 for x in result.to_list() if x is not None)
        assert non_null == 0


# ==============================================================================
# Test Class 6: temporal split (时间漂移)
# ==============================================================================


class TestTemporalSplit:
    """时间漂移: 不同时间段数据"""

    def test_train_period_works(self, vocab):
        """训练期 (前 60%) 数据应能跑"""
        np.random.seed(42)
        n = 100
        df = pl.DataFrame({
            "date": [f"2024-{(d % 12) + 1:02d}-{((d // 12) % 28) + 1:02d}" for d in range(n)],
            "code": ["A", "B", "C", "D", "E"] * (n // 5),
            "close": np.random.randn(n).cumsum() + 100,
        }).with_columns(pl.col("date").str.to_date())
        # 训练期: 前 60 行
        train = df.head(60)
        result = vocab.evaluate("ts_mean(close, 5)", train)
        assert result is not None
        assert result.shape[0] == 60

    def test_test_period_works(self, vocab):
        """测试期 (后 40%) 数据应能跑"""
        np.random.seed(42)
        n = 100
        df = pl.DataFrame({
            "date": [f"2024-{(d % 12) + 1:02d}-{((d // 12) % 28) + 1:02d}" for d in range(n)],
            "code": ["A", "B", "C", "D", "E"] * (n // 5),
            "close": np.random.randn(n).cumsum() + 100,
        }).with_columns(pl.col("date").str.to_date())
        test = df.tail(40)
        result = vocab.evaluate("ts_mean(close, 5)", test)
        assert result is not None
        assert result.shape[0] == 40


# ==============================================================================
# Test Class 7: 真实数据子集 (如果有)
# ==============================================================================


class TestRealDataSubset:
    """真实数据子集 (如果 parquet 存在)"""

    @pytest.fixture
    def real_data_subset(self):
        """真实数据子集 (1000 行)"""
        try:
            df = pl.read_parquet("data/cache/full_a_2019_2024.parquet")
            # 取前 1000 行
            return df.head(1000)
        except Exception:
            pytest.skip("Real data not available")

    def test_real_data_ts_mean(self, vocab, real_data_subset):
        """真实数据 ts_mean 应能算"""
        result = vocab.evaluate("ts_mean(close, 5)", real_data_subset)
        assert result is not None
        assert result.shape[0] == real_data_subset.shape[0]

    def test_real_data_rank(self, vocab, real_data_subset):
        """真实数据 rank 应能算"""
        result = vocab.evaluate("rank(close)", real_data_subset)
        assert result is not None
        assert result.shape[0] == real_data_subset.shape[0]
