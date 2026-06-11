"""performance_metrics.py 边界条件测试 (15 tests)。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.research.factor_test.utils.performance_metrics import (
    calc_max_drawdown,
    cal_net_simple,
    evaluation,
)


class TestCalcMaxDrawdown:
    def test_flat_line(self):
        n = pd.Series([1.0, 1.0, 1.0, 1.0])
        r = calc_max_drawdown(n)
        assert r["MDD"] == 0.0
        # 当前实现即使 MDD=0 也返回持续期 1
        assert r["Lastingtime"] >= 1

    def test_monotonic_up(self):
        n = pd.Series([1.0, 1.1, 1.2, 1.3])
        r = calc_max_drawdown(n)
        assert r["MDD"] == 0.0

    def test_then_recover(self):
        n = pd.Series([1.0, 0.8, 0.7, 0.9, 1.0, 1.1])
        r = calc_max_drawdown(n)
        assert r["MDD"] > 0
        assert r["Lastingtime"] >= 1
        assert r["Endingtime"] >= 1

    def test_single_value_no_drawdown(self):
        n = pd.Series([100.0])
        r = calc_max_drawdown(n)
        assert r["MDD"] == 0.0
        assert r["Endingtime"] >= 1


class TestCalNetSimple:
    def _net_and_dates(self):
        dates = pd.to_datetime(["2025-01-01", "2025-01-15", "2025-02-01", "2025-02-15"])
        dates_int = [int(d.strftime("%Y%m%d")) for d in dates]
        np.random.seed(42)
        net = pd.Series(np.cumprod(1 + np.random.randn(46) * 0.01) + 1,
                        index=pd.date_range("2025-01-01", periods=46, freq="D"))
        adj_dates = sorted([d for d in net.index
                            if d.day in (1, 15) and d >= net.index[0] and d <= net.index[-1]])[:4]
        return net, adj_dates

    def test_returns_series_with_net_col(self):
        net = pd.Series(np.exp(np.linspace(0, 0.05, 30)),
                        index=pd.date_range("2025-01-01", periods=30, freq="D"))
        adj = [net.index[0], net.index[15], net.index[29]]
        out = cal_net_simple(net, adj)
        assert "net" in out.name or out.name == "net"

    def test_simple_returns_finite(self):
        net = pd.Series(np.exp(np.linspace(0, 0.03, 20)),
                        index=pd.date_range("2025-01-01", periods=20, freq="D"))
        adj = [net.index[0], net.index[10], net.index[19]]
        out = cal_net_simple(net, adj)
        assert out.isna().sum() == 0
        assert (out > 0).all()


class TestEvaluation:
    def _make_inputs(self):
        dates = pd.date_range("2025-01-01", periods=300, freq="B")
        idx = [int(d.strftime("%Y%m%d")) for d in dates]
        cum = 1 + np.cumsum(np.random.randn(300) * 0.005)
        net = pd.Series(np.maximum(cum, 0.5), index=idx)
        adj = [idx[0], idx[50], idx[100], idx[200], idx[-1]]
        return net, adj

    def test_return_df_structure(self):
        net, adj = self._make_inputs()
        df = evaluation(net, adj)
        assert isinstance(df, pd.DataFrame)
        assert "Year" in df.columns
        assert "AnnualRt" in df.columns
        assert "Calmar" in df.columns

    def test_total_period_row(self):
        net, adj = self._make_inputs()
        df = evaluation(net, adj)
        total = df[df["Year"] == "all"]
        assert len(total) == 1

    def test_sharpe_nan_when_std_zero(self):
        """净值无波动时 SR = nan。"""
        n = 50
        dates = pd.date_range("2025-01-01", periods=n, freq="D")
        idx = [int(d.strftime("%Y%m%d")) for d in dates]
        net = pd.Series(np.ones(n), index=idx)
        net.iloc[0] = 1.0
        adj = [net.index[0], net.index[25], net.index[-1]]
        df = evaluation(net, adj)
        assert pd.isna(df.loc[0, "SR"])

    def test_calmar_nan_when_mdd_zero(self):
        n = 50
        dates = pd.date_range("2025-01-01", periods=n, freq="D")
        idx = [int(d.strftime("%Y%m%d")) for d in dates]
        net = pd.Series(np.linspace(1.0, 1.1, n), index=idx)
        adj = [net.index[0], net.index[-1]]
        df = evaluation(net, adj)
        assert pd.isna(df.loc[0, "Calmar"])