# coding=utf-8
"""期权因子库测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.option_factors import (
    build_option_factor_panel,
    charm_decay_rate,
    compute_gex,
    gex_zscore,
    iv_term_structure_slope,
    iv_zscore,
    portfolio_charm,
    portfolio_vanna,
    portfolio_volga,
    skew_rolling_panel,
    vanna_volga_ratio,
)


class TestIVZScore:
    def test_basic(self):
        iv = pd.DataFrame({
            "A": [0.20] * 60 + [0.25] * 10,
            "B": [0.15] * 60 + [0.10] * 10,
        })
        result = iv_zscore(iv, window=60)
        assert result.shape == iv.shape
        # 最后一个 A 的 z-score 应为正 (IV 上升)
        assert result["A"].iloc[-1] > 0
        # 最后一个 B 的 z-score 应为负 (IV 下降)
        assert result["B"].iloc[-1] < 0

    def test_nan_handling(self):
        iv = pd.DataFrame({"A": [np.nan] * 10 + [0.20] * 60})
        result = iv_zscore(iv, window=60)
        assert result["A"].iloc[:10].isna().all()


class TestVannaVolga:
    def test_portfolio_vanna(self):
        df = pd.DataFrame({
            "dt": ["2025-01-10"] * 3,
            "vanna": [0.1, 0.2, 0.3],
            "weight": [1, 2, 3],
            "multiplier": [10, 10, 10],
        })
        result = portfolio_vanna(df)
        assert len(result) == 1
        # 0.1*1*10 + 0.2*2*10 + 0.3*3*10 = 1 + 4 + 9 = 14
        assert result.iloc[0] == 14.0

    def test_portfolio_volga(self):
        df = pd.DataFrame({
            "dt": ["2025-01-10"] * 2,
            "volga": [0.5, 0.3],
            "weight": [1, 2],
            "multiplier": [10, 10],
        })
        result = portfolio_volga(df)
        assert result.iloc[0] == 11.0  # 0.5*1*10 + 0.3*2*10

    def test_vanna_volga_ratio(self):
        vanna = pd.Series([1.0, 2.0, 0.5])
        volga = pd.Series([2.0, 1.0, 0.5])
        ratio = vanna_volga_ratio(vanna, volga)
        assert ratio.iloc[0] == 0.5
        assert ratio.iloc[1] == 2.0
        assert ratio.iloc[2] == 1.0


class TestCharm:
    def test_portfolio_charm(self):
        df = pd.DataFrame({
            "dt": ["2025-01-10"] * 2,
            "charm": [-0.01, -0.02],
            "weight": [1, 2],
            "multiplier": [10, 10],
        })
        result = portfolio_charm(df)
        assert result.iloc[0] == -0.5  # -0.01*1*10 + -0.02*2*10

    def test_charm_decay_rate(self):
        charm = pd.Series([-0.05, -0.10])
        delta = pd.Series([0.5, 0.5])
        rate = charm_decay_rate(charm, delta)
        assert rate.iloc[0] == 0.1
        assert rate.iloc[1] == 0.2


class TestIVTermStructure:
    def test_slope_contango(self):
        iv_near = pd.Series([0.20, 0.21, 0.22])
        iv_far = pd.Series([0.25, 0.26, 0.27])
        slope = iv_term_structure_slope(iv_near, iv_far)
        assert (slope > 0).all()  # contango

    def test_slope_backwardation(self):
        iv_near = pd.Series([0.25, 0.26, 0.27])
        iv_far = pd.Series([0.20, 0.21, 0.22])
        slope = iv_term_structure_slope(iv_near, iv_far)
        assert (slope < 0).all()  # backwardation


class TestSkewRollingPanel:
    def test_basic(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10"] * 5 + ["2025-01-11"] * 5,
            "product": ["A"] * 5 + ["A"] * 5,
            "k": [5400, 5500, 5600, 5700, 5800] * 2,
            "bs_iv": [0.20, 0.18, 0.16, 0.15, 0.14, 0.21, 0.19, 0.17, 0.16, 0.15],
            "f": [5500] * 10,
        })
        result = skew_rolling_panel(data, window=2)
        assert "A" in result.columns
        assert len(result) >= 1


class TestGEX:
    def test_basic(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10"] * 3,
            "Gamma": [0.001, 0.002, 0.001],
            "open_interest": [100, 200, 150],
            "multiplier": [10, 10, 10],
            "f": [5500, 5500, 5500],
            "cp": ["C", "C", "P"],
        })
        result = compute_gex(data)
        assert len(result) == 1
        # C: +, P: -
        # 0.001*100*10*5500*1 + 0.002*200*10*5500*1 + 0.001*150*10*5500*(-1)
        expected = 5500 + 22000 - 8250
        assert abs(result.iloc[0] - expected) < 1

    def test_gex_zscore(self):
        gex = pd.Series(np.random.randn(100).cumsum() + 100)
        result = gex_zscore(gex, window=30)
        assert len(result) == 100
        # 最后30个应该有值
        assert result.iloc[-1] != 0 or np.isnan(result.iloc[-1])


class TestOptionFactorPanel:
    def test_basic(self):
        iv = pd.DataFrame({
            "A": np.random.randn(100).cumsum() * 0.01 + 0.20,
            "B": np.random.randn(100).cumsum() * 0.01 + 0.15,
        }, index=pd.date_range("2025-01-01", periods=100))
        result = build_option_factor_panel(iv)
        assert "iv_zscore" in result.columns
        assert "iv_rank" in result.columns
        assert "iv_momentum_5d" in result.columns
        assert len(result) == 100

    def test_with_hv(self):
        iv = pd.DataFrame({
            "A": np.random.randn(100).cumsum() * 0.01 + 0.20,
        }, index=pd.date_range("2025-01-01", periods=100))
        hv = pd.DataFrame({
            "A": np.random.randn(100).cumsum() * 0.01 + 0.15,
        }, index=pd.date_range("2025-01-01", periods=100))
        result = build_option_factor_panel(iv, hv)
        assert "vrp_mean" in result.columns
        assert "vrp_rank" in result.columns
