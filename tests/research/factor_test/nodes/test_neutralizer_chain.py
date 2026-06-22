# coding=utf-8
"""Phase 2.1: Neutralizer Chain tests.

Covers:
  - Neutralizer ABC
  - IndustryNeutralizer (build_design_matrix + is_active)
  - RiskNeutralizer (build_design_matrix + is_active)
  - build_neutralizer_chain (4 flag 组合)
  - apply_neutralizer_chain (单/双/无 chain/无数据日)
  - Backward compat: 与原 _neutralize 行为等价
  - End-to-end: FactorNeutralizeNode._execute
  - Pre-existing bool bug fix (Industry dummies → float)
"""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.nodes.factor_neutralize_node import (
    FactorNeutralizeNode,
)
from QuantNodes.research.factor_test.nodes.configs import NeutralizeNodeConfig
from QuantNodes.research.factor_test.nodes.neutralizers import (
    IndustryNeutralizer,
    Neutralizer,
    RiskNeutralizer,
    apply_neutralizer_chain,
    build_neutralizer_chain,
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
    """(3 dates × 20 stocks) 连续因子, 无 nan。"""
    rng = np.random.RandomState(42)
    return pd.DataFrame(rng.randn(N_DAYS, N_STOCKS), index=DATES, columns=STOCKS)


@pytest.fixture
def industry_df() -> pd.DataFrame:
    """industry: index=dates, columns=stocks, values=1-4 (4 个行业)。"""
    codes = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    return pd.DataFrame([codes] * N_DAYS, index=DATES, columns=STOCKS)


@pytest.fixture
def risk_factors() -> list:
    """2 个风险因子, index=dates, columns=stocks。"""
    rng = np.random.RandomState(7)
    rf1 = pd.DataFrame(rng.randn(N_DAYS, N_STOCKS), index=DATES, columns=STOCKS)
    rf2 = pd.DataFrame(rng.randn(N_DAYS, N_STOCKS), index=DATES, columns=STOCKS)
    return [rf1, rf2]


# ---------------------------------------------------------------------------
# Neutralizer ABC
# ---------------------------------------------------------------------------

class TestNeutralizerABC:
    def test_is_subclassable(self):
        """Neutralizer 是 ABC, 不能直接实例化。"""
        with pytest.raises(TypeError):
            Neutralizer()  # type: ignore[abstract]

    def test_subclass_must_implement_build_design_matrix(self):
        class Incomplete(Neutralizer):
            name = "x"
        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_name_default_empty(self):
        class Minimal(Neutralizer):
            def build_design_matrix(self, date, factor_i):
                return None
        m = Minimal()
        assert m.name == ""
        assert m.is_active() is True


# ---------------------------------------------------------------------------
# IndustryNeutralizer
# ---------------------------------------------------------------------------

class TestIndustryNeutralizer:
    def test_is_active_with_industry(self, industry_df):
        n = IndustryNeutralizer(industry_df)
        assert n.is_active() is True

    def test_is_active_with_none_industry(self):
        n = IndustryNeutralizer(None)
        assert n.is_active() is False

    def test_build_design_matrix_shape(self, industry_df, factor_df):
        n = IndustryNeutralizer(industry_df)
        X = n.build_design_matrix(DATES[0], factor_df)
        assert X is not None
        assert X.shape == (N_STOCKS, 4)  # 4 industries
        assert list(X.columns) == [1, 2, 3, 4]
        # 每行有且仅有 1 个 1 (one-hot)
        assert (X.sum(axis=1) == 1).all()

    def test_build_design_matrix_drops_all_zero_columns(self, factor_df):
        """如果某日期某些行业无股票, 该列应被过滤。"""
        # day 0: 全是行业 1
        industry = pd.DataFrame(
            [[1] * N_STOCKS, [1] * 10 + [2] * 10, [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5],
            index=DATES, columns=STOCKS,
        )
        n = IndustryNeutralizer(industry)
        X0 = n.build_design_matrix(DATES[0], factor_df)
        assert X0 is not None
        assert list(X0.columns) == [1]  # 只有行业 1

    def test_build_design_matrix_date_not_in_industry(self, industry_df, factor_df):
        """date 不在 industry.index → return None。"""
        n = IndustryNeutralizer(industry_df)
        fake_date = pd.Timestamp("2099-01-01")
        X = n.build_design_matrix(fake_date, factor_df)
        assert X is None

    def test_nan_replaced_with_0(self, factor_df):
        """industry 里的 nan 在 __init__ 时被替换为 0。"""
        industry = pd.DataFrame(
            [[1, 1, np.nan, np.nan] + [2] * (N_STOCKS - 4)] * N_DAYS,
            index=DATES, columns=STOCKS,
        )
        n = IndustryNeutralizer(industry)
        # nan 位置被替换, 应能生成 dummies
        X = n.build_design_matrix(DATES[0], factor_df)
        assert X is not None
        # 应至少有 2 列 (1, 2)
        assert set(X.columns) >= {1, 2}


# ---------------------------------------------------------------------------
# RiskNeutralizer
# ---------------------------------------------------------------------------

class TestRiskNeutralizer:
    def test_is_active_empty_list(self):
        n = RiskNeutralizer([])
        assert n.is_active() is False

    def test_is_active_with_factors(self, risk_factors):
        n = RiskNeutralizer(risk_factors)
        assert n.is_active() is True

    def test_build_design_matrix_single_factor(self, risk_factors, factor_df):
        n = RiskNeutralizer([risk_factors[0]])
        X = n.build_design_matrix(DATES[0], factor_df)
        assert X is not None
        # Phase 2.1: X 形状 = (n_stocks, n_factors) — 与 IndustryNeutralizer 对齐
        # (index=股票代码, columns=risk factors)
        assert X.shape == (N_STOCKS, 1)
        assert list(X.columns) == ["rf_0"]

    def test_build_design_matrix_multi_factors(self, risk_factors, factor_df):
        n = RiskNeutralizer(risk_factors)
        X = n.build_design_matrix(DATES[0], factor_df)
        assert X is not None
        assert X.shape == (N_STOCKS, 2)
        assert list(X.columns) == ["rf_0", "rf_1"]

    def test_build_design_matrix_date_not_in_factor(self, risk_factors, factor_df):
        """date 不在某个 risk factor.index → 跳过该 factor。"""
        rf_partial = risk_factors[0].drop(DATES[0])  # 去掉 day 0
        n = RiskNeutralizer([rf_partial, risk_factors[1]])
        X = n.build_design_matrix(DATES[0], factor_df)
        # 只有 rf1 命中, X 仍非空, 1 个 factor
        assert X is not None
        assert X.shape == (N_STOCKS, 1)
        assert list(X.columns) == ["rf_1"]


# ---------------------------------------------------------------------------
# build_neutralizer_chain
# ---------------------------------------------------------------------------

class TestBuildChain:
    def test_empty_when_both_false(self):
        chain = build_neutralizer_chain(False, False, None, [])
        assert chain == []

    def test_industry_only(self, industry_df):
        chain = build_neutralizer_chain(True, False, industry_df, [])
        assert len(chain) == 1
        assert chain[0].name == "industry"

    def test_risk_only(self, risk_factors):
        chain = build_neutralizer_chain(False, True, None, risk_factors)
        assert len(chain) == 1
        assert chain[0].name == "risk"

    def test_both_order_industry_first(self, industry_df, risk_factors):
        chain = build_neutralizer_chain(True, True, industry_df, risk_factors)
        assert len(chain) == 2
        assert chain[0].name == "industry"
        assert chain[1].name == "risk"

    def test_industry_filtered_when_none(self, risk_factors):
        """if_industry=True 但 industry=None → IndustryNeutralizer inactive → 过滤掉。"""
        chain = build_neutralizer_chain(True, True, None, risk_factors)
        assert len(chain) == 1
        assert chain[0].name == "risk"

    def test_risk_filtered_when_empty(self, industry_df):
        chain = build_neutralizer_chain(True, True, industry_df, [])
        assert len(chain) == 1
        assert chain[0].name == "industry"


# ---------------------------------------------------------------------------
# apply_neutralizer_chain
# ---------------------------------------------------------------------------

class TestApplyChain:
    def test_empty_chain_returns_nan(self, factor_df):
        """空 chain → 返回全 nan, 保留 shape。"""
        result = apply_neutralizer_chain(factor_df, [])
        assert result.shape == factor_df.shape
        assert result.isna().all().all()

    def test_industry_only_ols(self, factor_df, industry_df):
        """单 Industry: 残差每日均值 ~ 0 (OLS 残差性质)。"""
        chain = [IndustryNeutralizer(industry_df)]
        result = apply_neutralizer_chain(factor_df, chain)
        assert result.shape == factor_df.shape
        # 每行非 nan (OLS 成功)
        for d in DATES:
            assert result.loc[d].notna().all()
            assert abs(result.loc[d].mean()) < 1e-9  # 残差均值 ~ 0

    def test_risk_only_ols(self, factor_df, risk_factors):
        chain = [RiskNeutralizer(risk_factors)]
        result = apply_neutralizer_chain(factor_df, chain)
        assert result.shape == factor_df.shape
        for d in DATES:
            assert result.loc[d].notna().all()

    def test_industry_plus_risk(self, factor_df, industry_df, risk_factors):
        chain = build_neutralizer_chain(True, True, industry_df, risk_factors)
        result = apply_neutralizer_chain(factor_df, chain)
        assert result.shape == factor_df.shape
        for d in DATES:
            assert result.loc[d].notna().all()

    def test_skips_dates_with_all_nan(self, factor_df, industry_df):
        """factor 某日全 nan → 跳过该日 (与原代码 line 77/101/119 一致)。"""
        factor_df.loc[DATES[1]] = np.nan
        chain = [IndustryNeutralizer(industry_df)]
        result = apply_neutralizer_chain(factor_df, chain)
        assert result.loc[DATES[1]].isna().all()
        # 其他日应仍计算
        for d in [DATES[0], DATES[2]]:
            assert result.loc[d].notna().all()

    def test_bool_dtypes_converted_to_float(self, factor_df, industry_df):
        """Regression: IndustryNeutralizer 输出 bool dummies, 必须转 float 才能 OLS。

        原 _neutralize branch 2 在此场景下崩 (numpy boolean subtract error)。
        chain 修复后应正常工作。
        """
        chain = [IndustryNeutralizer(industry_df)]
        # 直接调用, 不应抛
        result = apply_neutralizer_chain(factor_df, chain)
        assert result.notna().any().any()  # 至少部分成功


# ---------------------------------------------------------------------------
# Backward compat: 与原 _neutralize 行为对比
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    """与 factor_neutralize_node.py::_neutralize 的原行为对比。

    由于原代码有 latent bool bug (branch 2/3 在 industry/risk 为 bool 时崩),
    这些测试用 'numeric' risk_factors (避免 bool 转换) 来比较。
    """

    def test_no_neutralizer_returns_factor(self, factor_df):
        """无 chain: 输出全 nan 但 shape 保留 (与原 _neutralize 入口 line 71-72 一致)。"""
        result = apply_neutralizer_chain(factor_df, [])
        assert result.shape == factor_df.shape

    def test_industry_only_preserves_index(self, factor_df, industry_df):
        chain = [IndustryNeutralizer(industry_df)]
        result = apply_neutralizer_chain(factor_df, chain)
        assert list(result.index) == list(factor_df.index)
        assert list(result.columns) == list(factor_df.columns)

    def test_risk_only_with_numeric_factor(self, factor_df, risk_factors):
        """Risk factor 是 float, 无 bool 转换问题。"""
        chain = [RiskNeutralizer(risk_factors)]
        result = apply_neutralizer_chain(factor_df, chain)
        # OLS 残差: 每行均值 ~ 0
        for d in DATES:
            assert abs(result.loc[d].mean()) < 1e-9


# ---------------------------------------------------------------------------
# End-to-end: FactorNeutralizeNode
# ---------------------------------------------------------------------------

class TestFactorNeutralizeNodeE2E:
    @staticmethod
    def _make_node(if_industry, if_risk, risk_factors=None):
        return FactorNeutralizeNode(
            config=NeutralizeNodeConfig(
                industry_neutral=if_industry,
                risk_neutral=if_risk,
                risk_factors=risk_factors or [],
            )
        )

    def test_no_neutralizer_returns_input(self, factor_df):
        """_neutralize 在 chain 为空时返回输入 (与 _execute line 39-40 short-circuit 一致).

        注: 原 _neutralize (line 71-72) 在无匹配分支时返回 factor_i * np.nan (全 NaN).
        但 _execute 已经在 line 39-40 短路: if_industry=False AND if_risk=False 时
        直接 return factor_std. 我们的实现让 _neutralize 与 _execute 行为一致 (返回 input),
        避免调用方误解 "全 NaN 是预期输出".
        """
        node = self._make_node(False, False)
        result = node._neutralize(factor_df, False, None, False, [])
        pd.testing.assert_frame_equal(result, factor_df)

    def test_industry_neutral_end_to_end(self, factor_df, industry_df):
        node = self._make_node(True, False)
        result = node._neutralize(factor_df, True, industry_df, False, [])
        assert result.notna().any().any()

    def test_risk_neutral_end_to_end(self, factor_df, risk_factors):
        node = self._make_node(False, True)
        result = node._neutralize(factor_df, False, None, True, risk_factors)
        assert result.notna().any().any()

    def test_both_neutral_end_to_end(self, factor_df, industry_df, risk_factors):
        node = self._make_node(True, True)
        result = node._neutralize(factor_df, True, industry_df, True, risk_factors)
        assert result.notna().any().any()

    def test_execute_missing_factor_preprocess(self):
        node = self._make_node(True, False)
        with pytest.raises(ValueError, match="因子预处理数据缺失"):
            node._execute(context={})

    def test_execute_missing_industry_with_industry_flag(self):
        node = self._make_node(True, False)
        factor = pd.DataFrame([[1, 2, 3]], index=[20200101], columns=["a", "b", "c"])
        with pytest.raises(ValueError, match="行业数据缺失"):
            node._execute(context={"FactorPreprocess": factor})

    def test_execute_no_neutralization_returns_input(self, factor_df):
        node = self._make_node(False, False)
        result = node._execute(context={"FactorPreprocess": factor_df})
        # no-op: 旧 line 39-40 直接 return factor_std
        pd.testing.assert_frame_equal(result, factor_df)

    def test_aliases_preserved(self):
        """Phase 1.4 引入的 _ALIASES 仍工作 (向后兼容)。"""
        node = self._make_node(True, True, risk_factors=[("a.h5", "b")])
        assert node._if_industry is True
        assert node._if_risk is True
        assert node._risk_factor_specs == [("a.h5", "b")]
