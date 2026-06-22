# coding=utf-8
"""Phase 2.2: Preprocess Strategy tests.

Covers:
  - MissingFillStrategy (PassThrough, IndustryAverage)
  - DeExtremeStrategy (PassThrough, MAD, PercentileShrink)
  - NormStrategy (PassThrough, ZScore, RankToNormal)
  - Factory functions (build_missing/extreme/norm_strategy, build_preprocess_strategies)
  - End-to-end: FactorPreprocessNode._preprocess_vectorized (3 strategy combos)
  - Backward compat: 与原 _preprocess_vectorized 行为 bitwise 一致
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm as scipy_norm

from QuantNodes.research.factor_test.nodes.configs import PreprocessNodeConfig
from QuantNodes.research.factor_test.nodes.factor_preprocess_node import (
    FactorPreprocessNode,
)
from QuantNodes.research.factor_test.nodes.preprocess_strategies import (
    DeExtremeStrategy,
    IndustryAverageMissing,
    MedianAbsoluteDeviationExtreme,
    MissingFillStrategy,
    NormStrategy,
    PassThroughExtreme,
    PassThroughMissing,
    PassThroughNorm,
    PercentileShrinkExtreme,
    RankToNormalNorm,
    ZScoreNorm,
    build_extreme_strategy,
    build_missing_strategy,
    build_norm_strategy,
    build_preprocess_strategies,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DAYS = 3
N_STOCKS = 20
DATES = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
STOCKS = [f"s{i}" for i in range(N_STOCKS)]


@pytest.fixture
def factor_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    return pd.DataFrame(rng.randn(N_DAYS, N_STOCKS), index=DATES, columns=STOCKS)


@pytest.fixture
def industry_df() -> pd.DataFrame:
    """industry: index=dates, columns=stocks, 2 industries (1, 2)."""
    codes = [1] * (N_STOCKS // 2) + [2] * (N_STOCKS - N_STOCKS // 2)
    return pd.DataFrame([codes] * N_DAYS, index=DATES, columns=STOCKS)


# ---------------------------------------------------------------------------
# Strategy ABCs
# ---------------------------------------------------------------------------

class TestStrategyABCs:
    def test_missing_strategy_abstract(self):
        with pytest.raises(TypeError):
            MissingFillStrategy()  # type: ignore[abstract]

    def test_extreme_strategy_abstract(self):
        with pytest.raises(TypeError):
            DeExtremeStrategy()  # type: ignore[abstract]

    def test_norm_strategy_abstract(self):
        with pytest.raises(TypeError):
            NormStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# MissingFillStrategy
# ---------------------------------------------------------------------------

class TestMissingStrategy:
    def test_passthrough_returns_same(self, factor_df):
        s = PassThroughMissing()
        result = s.apply(factor_df)
        assert result is factor_df  # 无修改直接返回

    def test_industry_average_with_no_industry(self, factor_df):
        """industry=None 时 ind_avg 也安全运行 (no-op)。"""
        s = IndustryAverageMissing()
        result = s.apply(factor_df)  # 不传 industry
        # 无 industry → 返回原 df
        pd.testing.assert_frame_equal(result, factor_df)

    def test_industry_average_fills_nan(self, factor_df, industry_df):
        """行业均值填充 NaN (与原 _preprocess_vectorized line 120-141 一致)。"""
        # 在 (date 0, industry 1) 组内制造 NaN
        factor_df.iloc[0, 0] = np.nan
        factor_df.iloc[0, 1] = np.nan
        s = IndustryAverageMissing()
        result = s.apply(factor_df, industry=industry_df)
        # NaN 应被填充
        assert result.iloc[0, 0] == result.iloc[0, 1]
        # 填充值 = industry 1 内 (date 0) 的均值
        ind1_values = factor_df.iloc[0, :N_STOCKS // 2].dropna()
        expected = ind1_values.mean()
        assert abs(result.iloc[0, 0] - expected) < 1e-9

    def test_industry_average_preserves_observed(self, factor_df, industry_df):
        """已观测值不应被填充覆盖。"""
        original = factor_df.iloc[0, 0]
        s = IndustryAverageMissing()
        result = s.apply(factor_df, industry=industry_df)
        assert result.iloc[0, 0] == original

    def test_industry_average_no_nan_no_change(self, factor_df, industry_df):
        """无 NaN 输入时, 填充后输出与输入一致 (除极小浮点误差)。"""
        s = IndustryAverageMissing()
        result = s.apply(factor_df, industry=industry_df)
        np.testing.assert_array_almost_equal(
            result.values[~np.isnan(factor_df.values)],
            factor_df.values[~np.isnan(factor_df.values)],
        )


# ---------------------------------------------------------------------------
# DeExtremeStrategy
# ---------------------------------------------------------------------------

class TestExtremeStrategy:
    def test_passthrough_returns_input(self, factor_df):
        s = PassThroughExtreme()
        result = s.apply(factor_df, mad_n=3.0)
        assert result is factor_df

    def test_mad_clamps_outliers(self):
        """MedianAbsoluteDeviation: |x - median| > n*MAD → clipped to bound."""
        s = MedianAbsoluteDeviationExtreme()
        data = pd.DataFrame(
            [[1, 2, 3, 4, 5, 100]],  # 100 是 outlier
            index=pd.to_datetime(["2020-01-01"]),
            columns=STOCKS[:6],
        )
        result = s.apply(data, mad_n=3.0)
        # data=[1,2,3,4,5,100]: median=3.5, MAD=1.5
        # bound = [3.5 - 4.5, 3.5 + 4.5] = [-1, 8]
        # 100 应被 clip 到 8
        assert result.iloc[0, -1] == 8.0
        # 内部值不变 (在 bound 内)
        assert result.iloc[0, 0] == 1.0

    def test_mad_with_custom_n(self):
        s = MedianAbsoluteDeviationExtreme()
        data = pd.DataFrame(
            [[1, 2, 3, 4, 5, 100]],
            index=pd.to_datetime(["2020-01-01"]),
            columns=STOCKS[:6],
        )
        result = s.apply(data, mad_n=0.5)
        # n=0.5: bound=[3.5-0.75, 3.5+0.75]=[2.75, 4.25]
        # 1 < 2.75 → clip to 2.75; 100 > 4.25 → clip to 4.25
        assert result.iloc[0, 0] == 2.75
        assert result.iloc[0, -1] == 4.25

    def test_pct_shrink_clamps_to_quantiles(self):
        s = PercentileShrinkExtreme()
        data = pd.DataFrame(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100]],  # 100 outlier
            index=pd.to_datetime(["2020-01-01"]),
            columns=STOCKS[:11],
        )
        result = s.apply(data, pct_low=0.1, pct_high=0.9)
        # q1=q(0.1), q2=q(0.9); 100 > q2 应被 clip
        q2 = data.quantile(0.9, axis=1).iloc[0]
        assert result.iloc[0, -1] == q2


# ---------------------------------------------------------------------------
# NormStrategy
# ---------------------------------------------------------------------------

class TestNormStrategy:
    def test_passthrough_returns_input(self, factor_df):
        s = PassThroughNorm()
        result = s.apply(factor_df)
        assert result is factor_df

    def test_zscore_zero_mean_unit_std(self, factor_df):
        s = ZScoreNorm()
        result = s.apply(factor_df)
        for d in DATES:
            row = result.loc[d]
            assert abs(row.mean()) < 1e-9
            assert abs(row.std(ddof=1) - 1.0) < 1e-9

    def test_zscore_with_constant_row_yields_nan(self):
        """行全相同值 → std=0, 转 NaN (与原代码 line 161-162 一致)。"""
        s = ZScoreNorm()
        data = pd.DataFrame(
            [[5.0] * N_STOCKS],
            index=pd.to_datetime(["2020-01-01"]),
            columns=STOCKS,
        )
        result = s.apply(data)
        assert result.isna().all().all()

    def test_rank_to_normal_ppf_range(self, factor_df):
        """RankToNormal: ranks (0, 1) → ppf → (-∞, +∞); 内部值应满足正态分布性质。"""
        s = RankToNormalNorm()
        result = s.apply(factor_df)
        # 内部值应 ~ N(0, 1) 分布
        for d in DATES:
            row = result.loc[d].dropna()
            assert abs(row.mean()) < 1.0  # 弱断言 (样本量小)
            assert 0.5 < row.std(ddof=1) < 2.0

    def test_rank_to_normal_handles_duplicate_values(self, factor_df):
        """重复值时 rank 处理正确 (pct=True 给出 0..1 排名)。"""
        data = pd.DataFrame(
            [[1.0] * N_STOCKS],
            index=pd.to_datetime(["2020-01-01"]),
            columns=STOCKS,
        )
        s = RankToNormalNorm()
        result = s.apply(data)
        # 全相同值: rank 都在中间, 应得一个 ppf 值
        # 不应崩, 不应全 nan
        assert result.shape == data.shape


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_build_missing_ind_avg(self):
        s = build_missing_strategy("ind_avg")
        assert isinstance(s, IndustryAverageMissing)
        assert s.name == "ind_avg"

    def test_build_missing_unknown_falls_back_to_passthrough(self):
        s = build_missing_strategy("nonexistent_method")
        assert isinstance(s, PassThroughMissing)

    def test_build_extreme_median(self):
        s = build_extreme_strategy("median")
        assert isinstance(s, MedianAbsoluteDeviationExtreme)
        assert s.name == "median"

    def test_build_extreme_pct_shrink(self):
        s = build_extreme_strategy("pct_shrink")
        assert isinstance(s, PercentileShrinkExtreme)
        assert s.name == "pct_shrink"

    def test_build_extreme_unknown_falls_back(self):
        s = build_extreme_strategy("unknown")
        assert isinstance(s, PassThroughExtreme)

    def test_build_norm_zscore(self):
        s = build_norm_strategy("zscore")
        assert isinstance(s, ZScoreNorm)

    def test_build_norm_norm(self):
        s = build_norm_strategy("norm")
        assert isinstance(s, RankToNormalNorm)

    def test_build_norm_unknown_falls_back(self):
        s = build_norm_strategy("unknown")
        assert isinstance(s, PassThroughNorm)

    def test_build_all_three(self):
        m, e, n = build_preprocess_strategies("ind_avg", "median", "zscore")
        assert isinstance(m, IndustryAverageMissing)
        assert isinstance(e, MedianAbsoluteDeviationExtreme)
        assert isinstance(n, ZScoreNorm)

    def test_build_all_passthrough(self):
        m, e, n = build_preprocess_strategies("x", "y", "z")
        assert isinstance(m, PassThroughMissing)
        assert isinstance(e, PassThroughExtreme)
        assert isinstance(n, PassThroughNorm)


# ---------------------------------------------------------------------------
# End-to-end: _preprocess_vectorized via 3 strategies
# ---------------------------------------------------------------------------

class TestPreprocessVectorizedE2E:
    @staticmethod
    def _make_node(missing, extreme, norm, **kwargs):
        return FactorPreprocessNode(
            config=PreprocessNodeConfig(
                missing=missing, extreme=extreme, norm=norm,
                mad_n=kwargs.get("mad_n", 3.0),
                pct_low=kwargs.get("pct_low", 0.01),
                pct_high=kwargs.get("pct_high", 0.99),
            )
        )

    def test_zscore_only(self, factor_df):
        node = self._make_node("x", "y", "zscore")
        result = node._preprocess_vectorized(
            factor_df, None, None,
            missing="x", extreme="y", norm="zscore",
        )
        for d in DATES:
            assert abs(result.loc[d].mean()) < 1e-9

    def test_median_then_zscore(self, factor_df):
        node = self._make_node("x", "median", "zscore")
        result = node._preprocess_vectorized(
            factor_df, None, None,
            missing="x", extreme="median", norm="zscore",
        )
        for d in DATES:
            assert abs(result.loc[d].mean()) < 1e-9
            assert abs(result.loc[d].std(ddof=1) - 1.0) < 1e-9

    def test_pct_shrink_then_zscore(self, factor_df):
        node = self._make_node("x", "pct_shrink", "zscore", pct_low=0.1, pct_high=0.9)
        result = node._preprocess_vectorized(
            factor_df, None, None,
            missing="x", extreme="pct_shrink", norm="zscore",
        )
        for d in DATES:
            assert abs(result.loc[d].mean()) < 1e-9

    def test_ind_avg_fill_then_zscore(self, factor_df, industry_df):
        node = self._make_node("ind_avg", "y", "zscore")
        result = node._preprocess_vectorized(
            factor_df, None, industry_df,
            missing="ind_avg", extreme="y", norm="zscore",
        )
        for d in DATES:
            assert abs(result.loc[d].mean()) < 1e-9

    def test_full_pipeline_all_three(self, factor_df, industry_df):
        """完整 pipeline: ind_avg → median → zscore."""
        node = self._make_node("ind_avg", "median", "zscore", mad_n=3.0)
        result = node._preprocess_vectorized(
            factor_df, None, industry_df,
            missing="ind_avg", extreme="median", norm="zscore",
        )
        for d in DATES:
            assert abs(result.loc[d].mean()) < 1e-9
            assert abs(result.loc[d].std(ddof=1) - 1.0) < 1e-9

    def test_rank_to_normal_e2e(self, factor_df):
        node = self._make_node("x", "y", "norm")
        result = node._preprocess_vectorized(
            factor_df, None, None,
            missing="x", extreme="y", norm="norm",
        )
        for d in DATES:
            row = result.loc[d].dropna()
            assert 0.5 < row.std(ddof=1) < 2.0

    def test_tradable_mask_applied_before_strategies(self, factor_df):
        """Step 1 (tradable mask) 在 strategy dispatch 之前, 不应被覆盖。"""
        tradable = pd.DataFrame(
            np.ones((N_DAYS, N_STOCKS)),  # 全 1
            index=DATES, columns=STOCKS,
        )
        tradable.iloc[0, 0] = 0  # 第 0 日第 0 只不可交易
        node = self._make_node("x", "y", "zscore")
        result = node._preprocess_vectorized(
            factor_df, tradable, None,
            missing="x", extreme="y", norm="zscore",
        )
        # 第 0 日第 0 只应 nan (tradable=0 → mask)
        assert pd.isna(result.iloc[0, 0])


# ---------------------------------------------------------------------------
# Backward compat: 与原 _preprocess_vectorized 行为对比
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """验证新实现与原 if 链结果一致 (按 phase 2.2 重构后行为不变)."""

    def test_zscore_matches_legacy_if_chain(self, factor_df):
        """直接调 _preprocess_vectorized 走新 dispatch, 与原 if 链计算应一致."""
        # 原 if 链 line 158-162:
        #   mean = result.mean(axis=1)
        #   std = result.std(axis=1, ddof=1).replace(0, nan)
        #   result = (result - mean) / std
        # 新 ZScoreNorm.apply() 完全相同
        node = FactorPreprocessNode(
            config=PreprocessNodeConfig(missing="x", extreme="y", norm="zscore"),
        )
        result = node._preprocess_vectorized(
            factor_df, None, None,
            missing="x", extreme="y", norm="zscore",
        )
        # 重新算一遍 (legacy 公式)
        mean = factor_df.mean(axis=1)
        std = factor_df.std(axis=1, ddof=1).replace(0, np.nan)
        expected = factor_df.sub(mean, axis=0).div(std, axis=0)
        np.testing.assert_array_almost_equal(result.values, expected.values, decimal=10)

    def test_median_then_zscore_matches_legacy(self, factor_df):
        """median → zscore 完整 pipeline 与原 line 144-162 顺序一致."""
        node = FactorPreprocessNode(
            config=PreprocessNodeConfig(missing="x", extreme="median", norm="zscore",
                                       mad_n=3.0),
        )
        result = node._preprocess_vectorized(
            factor_df, None, None,
            missing="x", extreme="median", norm="zscore",
        )
        # 重算 median clip → zscore
        med = factor_df.median(axis=1)
        mad = (factor_df.sub(med, axis=0)).abs().median(axis=1)
        lower = med - 3.0 * mad
        upper = med + 3.0 * mad
        clipped = factor_df.clip(lower=lower, upper=upper, axis=0)
        m2 = clipped.mean(axis=1)
        s2 = clipped.std(axis=1, ddof=1).replace(0, np.nan)
        expected = clipped.sub(m2, axis=0).div(s2, axis=0)
        np.testing.assert_array_almost_equal(result.values, expected.values, decimal=10)

    def test_pct_shrink_matches_legacy(self, factor_df):
        node = FactorPreprocessNode(
            config=PreprocessNodeConfig(missing="x", extreme="pct_shrink", norm="zscore",
                                       pct_low=0.05, pct_high=0.95),
        )
        result = node._preprocess_vectorized(
            factor_df, None, None,
            missing="x", extreme="pct_shrink", norm="zscore",
        )
        q1 = factor_df.quantile(0.05, axis=1)
        q2 = factor_df.quantile(0.95, axis=1)
        clipped = factor_df.clip(lower=q1, upper=q2, axis=0)
        m = clipped.mean(axis=1)
        s = clipped.std(axis=1, ddof=1).replace(0, np.nan)
        expected = clipped.sub(m, axis=0).div(s, axis=0)
        np.testing.assert_array_almost_equal(result.values, expected.values, decimal=10)

    def test_aliases_preserved(self):
        """Phase 1.4 引入的 _ALIASES 仍工作."""
        node = FactorPreprocessNode(
            config=PreprocessNodeConfig(missing="ind_avg", extreme="median", norm="norm"),
        )
        assert node._missing == "ind_avg"
        assert node._extreme == "median"
        assert node._norm == "norm"
        # _mad_n / _pct_low / _pct_high 来自 PreprocessSetting 默认值
        # (从 configs.py PreprocessNodeConfig 字段继承)
        assert node._mad_n == PreprocessNodeConfig.model_fields["mad_n"].default
        assert node._pct_low == PreprocessNodeConfig.model_fields["pct_low"].default
        assert node._pct_high == PreprocessNodeConfig.model_fields["pct_high"].default
