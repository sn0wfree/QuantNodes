# coding=utf-8
"""产业链上下游因子测试."""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.cta.factors.industrial_chain import (
    CHAIN_FORMULAS,
    ChainMarginFactor,
    LeadLagFactor,
    MarginMomentumFactor,
    PriceRatioFactor,
    chain_margin_series,
)


def _mk_panel(n=150):
    """构造包含产业链品种的面板: A 涨, I/J 跌, RB 滞后跟随."""
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    a = np.linspace(100, 200, n)          # 大豆涨
    m = np.linspace(50, 100, n)           # 豆粕涨 (跟随)
    y = np.linspace(40, 80, n)            # 豆油涨
    rb = np.linspace(200, 150, n)         # 螺纹跌
    i = np.linspace(100, 50, n)           # 铁矿跌
    j = np.linspace(150, 100, n)          # 焦炭跌
    au = np.linspace(300, 300, n)         # 金银比恒定
    ag = np.linspace(5, 5, n)
    return pd.DataFrame({
        "A": a, "M": m, "Y": y, "RB": rb, "I": i, "J": j, "AU": au, "AG": ag,
    }, index=idx)


class TestChainMarginSeries:
    def test_crush_profit(self):
        prices = _mk_panel()
        margin = chain_margin_series(prices, "crush_profit")
        # 0.78M + 0.18Y - A
        expected = 0.78 * prices["M"] + 0.18 * prices["Y"] - prices["A"]
        assert np.allclose(margin.values, expected.values)

    def test_steel_profit(self):
        prices = _mk_panel()
        margin = chain_margin_series(prices, "steel_profit_rb")
        expected = prices["RB"] - 1.6 * prices["I"] - 0.5 * prices["J"]
        assert np.allclose(margin.values, expected.values)

    def test_missing_product(self):
        prices = _mk_panel().drop(columns=["J"])
        margin = chain_margin_series(prices, "steel_profit_rb")
        assert margin.isna().all()

    def test_unknown_formula(self):
        with pytest.raises(KeyError):
            chain_margin_series(_mk_panel(), "nope")


class TestChainMarginFactor:
    def test_formulas_defined(self):
        assert len(CHAIN_FORMULAS) >= 8

    def test_output(self):
        prices = _mk_panel()
        sig = ChainMarginFactor("steel_profit_rb", 60, 1.0).compute(prices)
        assert sig.columns[0] == "steel_profit_rb"
        assert set(sig.iloc[:, 0].unique()).issubset({-1.0, 0.0, 1.0})

    def test_high_margin_short(self):
        # 构造高利润: RB 高, I/J 低 → 利润 z 高 → 空头
        n = 150
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        rb = np.linspace(300, 400, n)
        i = np.linspace(50, 50, n)
        j = np.linspace(100, 100, n)
        prices = pd.DataFrame({"RB": rb, "I": i, "J": j}, index=idx)
        sig = ChainMarginFactor("steel_profit_rb", 30, 1.0).compute(prices)
        assert sig.iloc[-1, 0] == -1.0  # 高利润 → 空


class TestPriceRatio:
    def test_ratio_output(self):
        prices = _mk_panel()
        sig = PriceRatioFactor("AU", "AG", 60, 1.0).compute(prices)
        assert sig.columns[0] == "AU_AG_ratio"
        # 恒定比值 → z=0 → 无信号
        assert (sig.iloc[:, 0] == 0).all()

    def test_missing_product(self):
        prices = _mk_panel()
        with pytest.raises(KeyError):
            PriceRatioFactor("AU", "ZZ", 60).compute(prices)


class TestMarginMomentum:
    def test_momentum_direction(self):
        prices = _mk_panel()
        sig = MarginMomentumFactor("crush_profit", 5).compute(prices)
        # 压榨利润 = 0.78M+0.18Y-A, M/Y 涨幅接近 A, 利润方向多变
        assert set(sig.iloc[:, 0].unique()).issubset({-1.0, 0.0, 1.0})
        assert (sig.iloc[:, 0] != 0).any()


class TestLeadLag:
    def test_positive_transmission(self):
        # 构造: 下游收益 = 上游 1 日滞后收益 (纯传导)
        n = 150
        idx = pd.date_range("2025-01-01", periods=n, freq="B")
        rng = np.random.RandomState(42)
        up_ret = rng.normal(0.001, 0.01, n)
        dn_ret = np.empty(n)
        dn_ret[0] = 0.0
        for t in range(1, n):
            dn_ret[t] = up_ret[t - 1]  # 滞后传导
        up = 100 * np.exp(np.cumsum(up_ret))
        dn = 100 * np.exp(np.cumsum(dn_ret))
        prices = pd.DataFrame({"UP": up, "DN": dn}, index=idx)
        sig = LeadLagFactor("UP", "DN", 60, 1, 0.3).compute(prices)
        assert (sig.iloc[:, 0] > 0).any()  # 正传导信号

    def test_output_values(self):
        prices = _mk_panel()
        sig = LeadLagFactor("I", "RB", 60, 1, 0.3).compute(prices)
        assert set(sig.iloc[:, 0].unique()).issubset({-1.0, 0.0, 1.0})
