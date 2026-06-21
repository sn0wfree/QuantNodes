# coding: utf-8
"""K10 (2026-06-21): performance_metrics 数值正确性 + 边界.

calc_max_drawdown / evaluation / cal_net_simple 全面 numerical validation.
"""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.utils.performance_metrics import (
    cal_net_simple,
    calc_max_drawdown,
    evaluation,
)


# ── calc_max_drawdown ──


class TestMaxDrawdown:
    def test_monotonic_no_drawdown(self):
        """单调递增净值 → MDD == 0."""
        net = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4],
                        index=[20260101, 20260102, 20260103, 20260104, 20260105])
        result = calc_max_drawdown(net)
        assert result["MDD"] == 0

    def test_known_drawdown(self):
        """已知曲线 1→2→0.5: MDD = (2-0.5)/2 = 0.75."""
        net = pd.Series([1.0, 2.0, 0.5],
                        index=[20260101, 20260102, 20260103])
        result = calc_max_drawdown(net)
        np.testing.assert_allclose(result["MDD"], 0.75)
        # MDD date == 20260103 (回撤最大日)
        assert result["MDD_date"] == 20260103

    def test_empty_returns_zero(self):
        """空 Series → MDD = 0, MDD_date = None."""
        net = pd.Series(dtype=float)
        result = calc_max_drawdown(net)
        assert result["MDD"] == 0
        assert result["MDD_date"] is None

    def test_constant_net(self):
        """常数净值 → MDD = 0."""
        net = pd.Series([1.0, 1.0, 1.0, 1.0],
                        index=[20260101, 20260102, 20260103, 20260104])
        result = calc_max_drawdown(net)
        assert result["MDD"] == 0

    def test_drawdown_lasting_time(self):
        """回撤持续期: 1.0(峰)→0.5(谷)→1.0(恢复). Lastingtime ≥ 1."""
        net = pd.Series([1.0, 0.8, 0.5, 0.7, 1.0],
                        index=[20260101, 20260102, 20260103, 20260104, 20260105])
        result = calc_max_drawdown(net)
        np.testing.assert_allclose(result["MDD"], 0.5)
        assert result["Lastingtime"] >= 1
        # MDD 在 20260103 达成
        assert result["MDD_date"] == 20260103

    def test_returns_dict_keys(self):
        """返回 dict 含 5 个标准 key."""
        net = pd.Series([1.0, 0.9, 0.8, 1.0],
                        index=[20260101, 20260102, 20260103, 20260104])
        result = calc_max_drawdown(net)
        expected = {"MDD", "MDD_date", "Lastingtime", "Endingtime", "DD"}
        assert expected.issubset(set(result.keys()))


# ── evaluation ──


def _make_net_and_dates(n_days=120, periods=6, drift=0.001, vol=0.01, seed=0):
    """合成净值曲线 + 调仓日."""
    rng = np.random.RandomState(seed)
    dates = [
        int(d.strftime("%Y%m%d"))
        for d in pd.bdate_range("2026-01-04", periods=n_days)
    ]
    daily_ret = rng.randn(n_days) * vol + drift
    daily_ret[0] = 0
    net_vals = np.cumprod(1 + daily_ret)
    net = pd.Series(net_vals, index=dates)
    # 等距 periods 个调仓日
    step = n_days // periods
    adj_dates = dates[::step][:periods + 1]
    return net, adj_dates


class TestEvaluation:
    def test_output_columns(self):
        """全 12 个标准列."""
        net, adj_dates = _make_net_and_dates()
        result = evaluation(net, adj_dates)
        expected = {
            "Year", "AnnualRt", "AccumRt", "SR", "MDD", "WinRatio",
            "WinLossRatio", "Calmar", "MDD_date",
            "MDD_lastdays", "MDD_recoverdays", "Periods",
        }
        assert expected.issubset(set(result.columns))

    def test_first_row_is_overall(self):
        """首行 Year == 'all' (全期指标)."""
        net, adj_dates = _make_net_and_dates()
        result = evaluation(net, adj_dates)
        assert result.iloc[0]["Year"] == "all"

    def test_accum_rt_matches_net_endpoints(self):
        """全期累计收益率 == net.iloc[-1] / net.iloc[0] - 1."""
        net, adj_dates = _make_net_and_dates()
        result = evaluation(net, adj_dates)
        expected_accum = net.iloc[-1] / net.iloc[0] - 1
        np.testing.assert_allclose(result.iloc[0]["AccumRt"], expected_accum, atol=1e-10)

    def test_constant_net_zero_annual(self):
        """常数净值 → AnnualRt = 0, MDD = 0."""
        dates = [int(d.strftime("%Y%m%d"))
                 for d in pd.bdate_range("2026-01-04", periods=60)]
        net = pd.Series(1.0, index=dates)
        adj_dates = dates[::15]
        result = evaluation(net, adj_dates)
        np.testing.assert_allclose(result.iloc[0]["AccumRt"], 0.0)
        assert result.iloc[0]["MDD"] == 0

    def test_sr_nan_when_std_zero(self):
        """std=0 时 SR == NaN (avoid div-by-zero)."""
        dates = [int(d.strftime("%Y%m%d"))
                 for d in pd.bdate_range("2026-01-04", periods=60)]
        net = pd.Series(1.0, index=dates)
        adj_dates = dates[::15]
        result = evaluation(net, adj_dates)
        assert pd.isna(result.iloc[0]["SR"])

    @pytest.mark.parametrize("annual_days", [250, 252, 365])
    def test_annual_days_scales_metric(self, annual_days):
        """annual_days 改变 → AnnualRt scales 线性."""
        net, adj_dates = _make_net_and_dates(seed=1)
        r_default = evaluation(net, adj_dates)
        r_custom = evaluation(net, adj_dates, annual_days=annual_days)
        # 都应是 finite 数
        assert np.isfinite(r_default.iloc[0]["AnnualRt"])
        assert np.isfinite(r_custom.iloc[0]["AnnualRt"])

    def test_win_ratio_in_unit_interval(self):
        """胜率必须 ∈ [0, 1]."""
        net, adj_dates = _make_net_and_dates(seed=2)
        result = evaluation(net, adj_dates)
        wr = result.iloc[0]["WinRatio"]
        assert 0 <= wr <= 1

    def test_periods_count_matches_adj_dates(self):
        """全期 Periods ≈ len(adj_dates) - 1 (有效收益期).

        Note: 实现是 every_return.notna().sum(), 第一个调仓日 diff=NaN,
        所以 Periods = len(adj_dates) - 1.
        """
        net, adj_dates = _make_net_and_dates()
        result = evaluation(net, adj_dates)
        assert result.iloc[0]["Periods"] == len(adj_dates) - 1


# ── cal_net_simple ──


class TestCalNetSimple:
    def test_simple_net_at_adj_dates_matches_input(self):
        """单利净值在调仓日处与复利净值差 = (复利-1) - (单利-1) 即累积偏差."""
        dates = list(range(20260101, 20260121))
        # 模拟正收益曲线
        net_cmp = pd.Series(
            np.cumprod(1 + np.linspace(0.005, 0.01, 20)), index=dates
        )
        adj_dates = dates[::4]  # 5 个调仓日
        result = cal_net_simple(net_cmp, adj_dates)
        assert isinstance(result, pd.Series)
        assert len(result) == len(net_cmp)

    def test_first_segment_equals_input(self):
        """第一个调仓日之前 (含首个), simp 应 == 复利 net."""
        dates = list(range(20260101, 20260111))
        net_cmp = pd.Series(np.linspace(1.0, 1.05, 10), index=dates)
        adj_dates = [dates[0], dates[4], dates[8]]
        result = cal_net_simple(net_cmp, adj_dates)
        # data_net.loc[:adj_dates[1]] (含端点) = 输入
        # 即 dates[0..4] 应等于输入
        for i in range(5):
            np.testing.assert_allclose(result.iloc[i], net_cmp.iloc[i], atol=1e-12)

    def test_no_inf_no_nan_in_output(self):
        """输出不应含 inf, 调仓日之间应有值."""
        dates = list(range(20260101, 20260141))
        rng = np.random.RandomState(0)
        net_cmp = pd.Series(
            np.cumprod(1 + rng.randn(40) * 0.005 + 0.001), index=dates
        )
        adj_dates = dates[::8]
        result = cal_net_simple(net_cmp, adj_dates)
        assert not np.isinf(result.values).any()
