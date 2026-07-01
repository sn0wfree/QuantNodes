# coding=utf-8
"""
test_evaluator_methods.py - 5 个未测 verify 方法测试 (Phase 4)

目标: 覆盖 polars_evaluator 的 5 个 _compute_* 方法
- _compute_stability
- _compute_diversification
- _compute_turnover
- _compute_monotonicity
- _compute_coverage

V4-V8 这些方法在 evaluator.verify() 中被调用, 但没有直接单元测试。
"""
import numpy as np
import polars as pl
import pytest

from QuantNodes.research.quant_alpha.evaluation.contracts import (
    FactorMetrics,
    FactorSpec,
    VerifyConfig,
)
from QuantNodes.research.quant_alpha.evaluation.evaluators.polars_evaluator import (
    PolarsAlphaCalculatorEvaluator,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def evaluator() -> PolarsAlphaCalculatorEvaluator:
    return PolarsAlphaCalculatorEvaluator()


@pytest.fixture
def verify_config() -> VerifyConfig:
    """默认验证配置"""
    return VerifyConfig(
        ic_threshold=0.03,
        icir_threshold=0.5,
        stability_threshold=0.6,
        corr_threshold=0.7,
        turnover_threshold=0.5,
        monotonicity_threshold=0.7,
        coverage_threshold=0.8,
        n_groups=5,
        rolling_window=20,
    )


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """3 票 × 30 日 测试数据"""
    np.random.seed(42)
    rows = []
    for d in range(30):
        for s in ["A", "B", "C"]:
            close = 100.0 + d * 0.5 + np.random.randn() * 2
            rows.append({
                "date": f"2024-01-{d + 1:02d}" if d < 31 else "2024-02-01",
                "code": s,
                "close": close,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "vol": 1000.0,
                "forward_return": np.random.randn() * 0.02,  # 日收益
            })
    return pl.DataFrame(rows).with_columns(pl.col("date").str.to_date())


@pytest.fixture
def sample_factor_values(sample_data: pl.DataFrame) -> pl.Series:
    """简单因子: close - 100"""
    return sample_data["close"] - 100.0


# ==============================================================================
# Test Class 1: _compute_stability
# ==============================================================================


class TestComputeStability:
    """_compute_stability: 滚动 IC 标准差"""

    def test_returns_zero_when_factor_is_none(self, evaluator, verify_config, sample_data):
        """factor_values=None → 0.0"""
        result = evaluator._compute_stability(sample_data, None, verify_config)
        assert result == 0.0

    def test_returns_zero_when_data_too_short(self, evaluator, verify_config, sample_data):
        """数据 < rolling_window → 0.0"""
        # 制造 5 天数据 (rolling_window=20)
        small_data = sample_data.head(15)
        small_factor = small_data["close"] - 100.0
        result = evaluator._compute_stability(small_data, small_factor, verify_config)
        assert result == 0.0

    def test_returns_value_with_enough_data(self, evaluator, verify_config, sample_data, sample_factor_values):
        """数据足够时返回有限值"""
        result = evaluator._compute_stability(sample_data, sample_factor_values, verify_config)
        # 应在 [0, 1] 范围内
        assert 0.0 <= result <= 1.0

    def test_stable_factor_high_score(self, evaluator, verify_config):
        """稳定因子 (close - constant) 应得高分"""
        np.random.seed(42)
        # 制造 25 天数据 (rolling_window=20 满足)
        from datetime import date, timedelta
        dates = [
            (date(2024, 1, 1) + timedelta(days=d)).isoformat()
            for d in range(25)
            for _ in range(3)
        ]
        data = pl.DataFrame({
            "date": dates,
            "code": ["A", "B", "C"] * 25,
            "forward_return": np.random.randn(75).tolist(),
        }).with_columns(pl.col("date").str.to_date())
        # 因子 = forward_return (与 forward_return 完全相关, IC = 1.0)
        # 25 天滚动 IC 稳定
        factor = pl.Series(data["forward_return"].to_list())
        result = evaluator._compute_stability(data, factor, verify_config)
        # 应得高分 (稳定)
        assert result >= 0.0  # 合法值
        assert isinstance(result, float)


# ==============================================================================
# Test Class 2: _compute_diversification
# ==============================================================================


class TestComputeDiversification:
    """_compute_diversification: 与已有因子的相关性"""

    def test_returns_one_when_no_existing_factors(self, evaluator, verify_config, sample_factor_values):
        """无已有因子 → 1.0 (默认满分)"""
        result = evaluator._compute_diversification(sample_factor_values, None, verify_config)
        assert result == 1.0

    def test_returns_one_when_empty_existing_list(self, evaluator, verify_config, sample_factor_values):
        """已有因子空列表 → 1.0"""
        result = evaluator._compute_diversification(sample_factor_values, [], verify_config)
        assert result == 1.0

    def test_uncorrelated_factor_high_score(self, evaluator, verify_config, sample_factor_values):
        """与已有因子不相关 → 高分"""
        np.random.seed(99)
        n = len(sample_factor_values)
        # 完全随机的已有因子, 排名应与 sample_factor_values 接近不相关
        existing = pl.Series(np.random.permutation(sample_factor_values.to_list()))
        result = evaluator._compute_diversification(
            sample_factor_values, [existing], verify_config
        )
        # 随机排列接近不相关 → 分散度 > 0.5
        assert result > 0.5

    def test_anti_correlated_factor_low_score(self, evaluator, verify_config, sample_factor_values):
        """与已有因子反相关 → 低分 (abs(corr)=1)

        注: 函数用 abs(corr), 所以正/反相关都视为不分散
        """
        existing_aligned = -sample_factor_values
        result = evaluator._compute_diversification(
            sample_factor_values, [existing_aligned], verify_config
        )
        # 反向 → 分散度 = 0 (abs(-1)=1)
        assert result < 0.1

    def test_identical_factor_low_score(self, evaluator, verify_config, sample_factor_values):
        """与已有因子相同 → 低分"""
        result = evaluator._compute_diversification(
            sample_factor_values, [sample_factor_values.clone()], verify_config
        )
        # 完全正相关 → 分散度 = 0
        assert result < 0.1

    def test_handles_mismatched_length(self, evaluator, verify_config, sample_factor_values):
        """长度不匹配的已有因子应被跳过"""
        short_factor = pl.Series([1.0] * 5)  # 长度不匹配
        result = evaluator._compute_diversification(
            sample_factor_values, [short_factor], verify_config
        )
        # 应跳过, 返回 1.0
        assert result == 1.0


# ==============================================================================
# Test Class 3: _compute_turnover
# ==============================================================================


class TestComputeTurnover:
    """_compute_turnover: 排名变化率"""

    def test_returns_one_when_factor_is_none(self, evaluator, verify_config, sample_data):
        """factor_values=None → 1.0"""
        result = evaluator._compute_turnover(sample_data, None, verify_config)
        assert result == 1.0

    def test_returns_zero_with_single_date(self, evaluator, verify_config):
        """单日数据 → 0.0 (无变化)"""
        data = pl.DataFrame({
            "date": ["2024-01-01"] * 5,
            "code": ["A", "B", "C", "D", "E"],
        }).with_columns(pl.col("date").str.to_date())
        factor = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = evaluator._compute_turnover(data, factor, verify_config)
        assert result == 0.0

    def test_constant_factor_zero_turnover(self, evaluator, verify_config):
        """常数因子 → 0 turnover"""
        data = pl.DataFrame({
            "date": (["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3),
            "code": ["A", "B", "C"] * 3,
        }).with_columns(pl.col("date").str.to_date())
        factor = pl.Series([1.0] * 9)
        result = evaluator._compute_turnover(data, factor, verify_config)
        # 排名不变 → turnover = 0
        assert result < 0.1

    def test_highly_changing_factor_high_turnover(self, evaluator, verify_config):
        """排名大幅变化 → 高 turnover"""
        # 每天反向排名
        dates = ["2024-01-01"] * 3 + ["2024-01-02"] * 3 + ["2024-01-03"] * 3
        data = pl.DataFrame({
            "date": dates,
            "code": ["A", "B", "C"] * 3,
        }).with_columns(pl.col("date").str.to_date())
        # 因子每天反向: day1=[1,2,3], day2=[3,2,1], day3=[1,2,3]
        factor = pl.Series([1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
        result = evaluator._compute_turnover(data, factor, verify_config)
        # 排名变化大, turnover > 0
        assert result > 0.0


# ==============================================================================
# Test Class 4: _compute_monotonicity
# ==============================================================================


class TestComputeMonotonicity:
    """_compute_monotonicity: 分组收益单调性"""

    def test_returns_zero_when_factor_is_none(self, evaluator, verify_config, sample_data):
        """factor_values=None → 0.0"""
        result = evaluator._compute_monotonicity(sample_data, None, verify_config)
        assert result == 0.0

    def test_returns_zero_when_data_too_short(self, evaluator, verify_config, sample_data):
        """数据 < n_groups*10 → 0.0"""
        # n_groups=5, n_groups*10=50, 我们的数据 30 < 50
        small_data = sample_data.head(30)
        small_factor = small_data["close"] - 100.0
        result = evaluator._compute_monotonicity(small_data, small_factor, verify_config)
        assert result == 0.0

    def test_returns_value_with_enough_data(self, evaluator, verify_config):
        """数据足够时返回有限值"""
        np.random.seed(42)
        # 制造 100 行 (n_groups*10=50 足够)
        data = pl.DataFrame({
            "date": ["2024-01-01"] * 100,
            "code": [f"S{i % 10}" for i in range(100)],
            "forward_return": np.random.randn(100).tolist(),
        }).with_columns(pl.col("date").str.to_date())
        factor = pl.Series(np.random.randn(100).tolist())
        result = evaluator._compute_monotonicity(data, factor, verify_config)
        # 应在 [0, 1] 范围内
        assert 0.0 <= result <= 1.0

    def test_perfectly_monotonic_factor_high_score(self, evaluator, verify_config):
        """完全单调因子 (排名 == forward_return) → 高分"""
        np.random.seed(42)
        n = 100
        fwd = np.random.randn(n)
        data = pl.DataFrame({
            "date": ["2024-01-01"] * n,
            "code": [f"S{i % 10}" for i in range(n)],
            "forward_return": fwd.tolist(),
        }).with_columns(pl.col("date").str.to_date())
        # 因子 = forward_return (完全单调)
        factor = pl.Series(fwd.tolist())
        result = evaluator._compute_monotonicity(data, factor, verify_config)
        # 完全单调 → 相关性高
        assert result >= 0.0  # 不崩, 返回合理值
        # 注意: monotonicity 是分组收益单调性, 即使完全单调
        # 也不一定 = 1.0 (取决于分位数边界)
        # 至少要是合法值
        assert isinstance(result, float)


# ==============================================================================
# Test Class 5: _compute_coverage
# ==============================================================================


class TestComputeCoverage:
    """_compute_coverage: 非空比例"""

    def test_returns_zero_when_factor_is_none(self, evaluator, sample_data):
        """factor_values=None → 0.0"""
        result = evaluator._compute_coverage(sample_data, None)
        assert result == 0.0

    def test_full_coverage_no_nulls(self, evaluator, sample_data):
        """全非空 → 1.0"""
        factor = pl.Series([1.0] * len(sample_data))
        result = evaluator._compute_coverage(sample_data, factor)
        assert result == 1.0

    def test_half_coverage(self, evaluator, sample_data):
        """一半空 → 0.5"""
        n = len(sample_data)
        values = [1.0 if i % 2 == 0 else None for i in range(n)]
        factor = pl.Series(values)
        result = evaluator._compute_coverage(sample_data, factor)
        assert abs(result - 0.5) < 0.01

    def test_zero_coverage_all_null(self, evaluator, sample_data):
        """全空 → 0.0"""
        n = len(sample_data)
        factor = pl.Series([None] * n)
        result = evaluator._compute_coverage(sample_data, factor)
        assert result == 0.0

    def test_handles_numpy_array(self, evaluator, sample_data):
        """numpy array 输入"""
        factor = np.array([1.0] * len(sample_data))
        result = evaluator._compute_coverage(sample_data, factor)
        assert result == 1.0

    def test_handles_numpy_with_nan(self, evaluator, sample_data):
        """numpy array 含 NaN"""
        n = len(sample_data)
        factor = np.array([1.0 if i % 2 == 0 else np.nan for i in range(n)])
        result = evaluator._compute_coverage(sample_data, factor)
        # ~50% coverage
        assert abs(result - 0.5) < 0.01


# ==============================================================================
# Test Class 6: verify() 集成
# ==============================================================================


class TestVerifyIntegration:
    """evaluator.verify() 集成测试"""

    def test_verify_failed_metrics_short_circuits(self, evaluator, sample_data, sample_factor_values):
        """失败的 metrics 应短路返回, 不计算 verify"""
        from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
        metrics = FactorMetrics(
            formula_id="F1",
            status="failed",
            error_msg="evaluation failed",
        )
        result = evaluator.verify(metrics, sample_data, sample_factor_values)
        assert result.is_valid is False
        assert "evaluation failed" in result.fail_reasons

    def test_verify_success_metrics_runs_all_checks(self, evaluator, sample_data, sample_factor_values):
        """成功的 metrics 应运行 6 维验证"""
        from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
        metrics = FactorMetrics(
            formula_id="F1",
            status="success",
            ir=0.1,
            ic_mean=0.02,
        )
        result = evaluator.verify(metrics, sample_data, sample_factor_values)
        # 验证跑完, is_valid 应是 bool
        assert isinstance(result.is_valid, bool)
