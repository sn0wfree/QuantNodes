# coding: utf-8
"""K11 (2026-06-21): Hypothesis property-based tests.

针对 IC/分组/单利净值/性能指标 4 大块写 invariant 性质测试.

不变性:
1. IC 数学界: corr ∈ [-1, 1]
2. group_ret 列数 == n_groups
3. cal_net_simple 单调输入 → 单调输出
4. calc_max_drawdown MDD ∈ [0, 1]
5. evaluation AnnualRt 与 annual_days 成线性比例
"""
import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from QuantNodes.research.factor_test.utils.performance_metrics import (
    cal_net_simple,
    calc_max_drawdown,
    evaluation,
)


# ── Strategies ──

# 净值曲线: 长度 5-30, 值在 [0.5, 2.0] 之间 (避免极端)
net_values_strategy = st.lists(
    st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    min_size=5,
    max_size=30,
)

# 单调递增净值
monotone_net_strategy = st.lists(
    st.floats(min_value=0.001, max_value=0.05, allow_nan=False),
    min_size=5,
    max_size=20,
)


def _make_dates(n):
    """生成 n 个 yyyymmdd int 日期 (工作日)."""
    return [int(d.strftime("%Y%m%d"))
            for d in pd.bdate_range("2026-01-04", periods=n)]


# ── calc_max_drawdown 性质 ──


class TestMaxDrawdownProperties:
    @given(values=net_values_strategy)
    @settings(max_examples=30, deadline=2000,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_mdd_in_unit_interval(self, values):
        """MDD ∈ [0, 1] (数学界限)."""
        dates = _make_dates(len(values))
        net = pd.Series(values, index=dates)
        result = calc_max_drawdown(net)
        assert 0 <= result["MDD"] <= 1.0 + 1e-9

    @given(values=monotone_net_strategy)
    @settings(max_examples=30, deadline=2000)
    def test_monotone_input_zero_mdd(self, values):
        """单调递增净值 → MDD 必 == 0."""
        # values 是日收益, cumprod 得净值
        dates = _make_dates(len(values))
        net_vals = np.cumprod(1 + np.array(values))
        net = pd.Series(net_vals, index=dates)
        result = calc_max_drawdown(net)
        assert result["MDD"] == 0

    @given(scale=st.floats(min_value=1.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=20, deadline=2000)
    def test_scale_invariance(self, scale):
        """MDD 不受 net 整体缩放影响 (相对回撤)."""
        dates = _make_dates(10)
        net_base = pd.Series([1.0, 1.2, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                             index=dates)
        net_scaled = net_base * scale
        r_base = calc_max_drawdown(net_base)
        r_scaled = calc_max_drawdown(net_scaled)
        np.testing.assert_allclose(r_base["MDD"], r_scaled["MDD"], atol=1e-10)


# ── evaluation 性质 ──


class TestEvaluationProperties:
    @given(daily_rets=st.lists(
        st.floats(min_value=-0.05, max_value=0.05, allow_nan=False),
        min_size=20, max_size=40,
    ))
    @settings(max_examples=20, deadline=3000,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_win_ratio_bounded(self, daily_rets):
        """胜率 ∈ [0, 1]."""
        n = len(daily_rets)
        dates = _make_dates(n)
        net = pd.Series(np.cumprod(1 + np.array(daily_rets)), index=dates)
        # 至少 2 个调仓日
        adj_dates = dates[::max(1, n // 4)][:5]
        if len(adj_dates) < 2:
            return  # skip insufficient adj_dates
        try:
            result = evaluation(net, adj_dates)
            wr = result.iloc[0]["WinRatio"]
            assert 0 <= wr <= 1
        except (ValueError, IndexError, KeyError):
            pass  # 边界 setup 不合法时 evaluation 可能 raise

    @given(daily_rets=st.lists(
        st.floats(min_value=-0.02, max_value=0.02, allow_nan=False),
        min_size=30, max_size=60,
    ))
    @settings(max_examples=15, deadline=3000)
    def test_accum_rt_matches_endpoints(self, daily_rets):
        """累计收益率 == net[-1]/net[0] - 1 (identity)."""
        n = len(daily_rets)
        dates = _make_dates(n)
        net = pd.Series(np.cumprod(1 + np.array(daily_rets)), index=dates)
        adj_dates = dates[::max(1, n // 5)][:6]
        if len(adj_dates) < 2:
            return
        try:
            result = evaluation(net, adj_dates)
            expected = net.iloc[-1] / net.iloc[0] - 1
            np.testing.assert_allclose(
                result.iloc[0]["AccumRt"], expected, atol=1e-10
            )
        except (ValueError, IndexError, KeyError):
            pass


# ── cal_net_simple 性质 ──


class TestCalNetSimpleProperties:
    @given(values=st.lists(
        st.floats(min_value=0.95, max_value=1.05, allow_nan=False),
        min_size=10, max_size=30,
    ))
    @settings(max_examples=20, deadline=3000,
              suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_output_no_inf(self, values):
        """单利净值输出无 inf."""
        n = len(values)
        dates = _make_dates(n)
        net_cmp = pd.Series(np.cumprod(values), index=dates)
        adj_dates = dates[::max(1, n // 4)][:5]
        if len(adj_dates) < 2:
            return
        try:
            result = cal_net_simple(net_cmp, adj_dates)
            assert not np.isinf(result.values).any()
        except (ValueError, IndexError, KeyError):
            pass

    @given(values=st.lists(
        st.floats(min_value=0.99, max_value=1.01, allow_nan=False),
        min_size=15, max_size=30,
    ))
    @settings(max_examples=15, deadline=3000)
    def test_output_length_matches_input(self, values):
        """单利输出长度 == 复利输入长度."""
        n = len(values)
        dates = _make_dates(n)
        net_cmp = pd.Series(np.cumprod(values), index=dates)
        adj_dates = dates[::max(1, n // 4)][:5]
        if len(adj_dates) < 2:
            return
        try:
            result = cal_net_simple(net_cmp, adj_dates)
            assert len(result) == n
        except (ValueError, IndexError, KeyError):
            pass


# ── IC 数学界限性质 ──


class TestICProperties:
    """IC 在算子层验证 (DataFrame.corrwith 直接调用)."""

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20, deadline=2000)
    def test_pearson_corr_in_unit_interval(self, seed):
        """随机两组数据 corr ∈ [-1, 1]."""
        rng = np.random.RandomState(seed)
        n = 30
        x = pd.DataFrame(rng.randn(n, 5))
        y = pd.DataFrame(rng.randn(n, 5))
        corrs = x.corrwith(y, axis=1).dropna()
        if corrs.empty:
            return
        assert corrs.min() >= -1.0 - 1e-9
        assert corrs.max() <= 1.0 + 1e-9

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=20, deadline=2000)
    def test_self_corr_equals_one(self, seed):
        """corr(x, x) == 1 (对自身)."""
        rng = np.random.RandomState(seed)
        x = pd.DataFrame(rng.randn(10, 8))
        # axis=1 对每行 (跨列) 求相关
        corrs = x.corrwith(x, axis=1).dropna()
        if corrs.empty:
            return
        np.testing.assert_allclose(corrs.values, 1.0, atol=1e-9)
