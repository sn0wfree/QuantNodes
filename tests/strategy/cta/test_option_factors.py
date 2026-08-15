# coding=utf-8
"""期权因子库测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.option_factors import (
    build_option_factor_panel,
    charm_decay_rate,
    compute_gex,
    cross_product_iv_spread,
    cross_product_iv_spread_zscore,
    gex_flip_level,
    gex_zscore,
    iv_term_structure_curvature,
    iv_term_structure_slope,
    iv_zscore,
    net_dealer_gamma,
    oi_change_rate,
    oi_price_divergence,
    portfolio_charm,
    portfolio_color,
    portfolio_speed,
    portfolio_theta_gamma_ratio,
    portfolio_vanna,
    portfolio_veta,
    portfolio_volga,
    put_call_volume_ratio,
    put_call_volume_ratio_by_delta,
    realized_vol_asymmetry,
    skew_momentum,
    skew_percentile,
    skew_rolling_panel,
    skew_zscore,
    theta_gamma_ratio,
    vanna_volga_ratio,
    vwap_iv,
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


class TestPutCallVolumeRatio:
    def test_basic(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10"] * 4,
            "cp": ["P", "P", "C", "C"],
            "volume": [100, 200, 100, 100],
        })
        result = put_call_volume_ratio(data)
        assert result.iloc[0] == 1.5  # (100+200)/(100+100)

    def test_by_delta(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10"] * 4,
            "cp": ["P", "P", "C", "C"],
            "Delta": [-0.3, -0.2, 0.3, 0.2],
            "volume": [100, 200, 150, 50],
        })
        result = put_call_volume_ratio_by_delta(data)
        assert len(result) == 1


class TestOIRate:
    def test_basic(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10", "2025-01-10", "2025-01-11", "2025-01-11"],
            "cp": ["C", "P", "C", "P"],
            "open_interest": [100, 200, 120, 180],
        })
        result = oi_change_rate(data)
        assert "total_oi_delta" in result.columns
        assert "net_positioning" in result.columns

    def test_oi_price_divergence(self):
        price = pd.Series([0.01, -0.02, 0.03])
        oi = pd.Series([10, -5, 15])
        result = oi_price_divergence(price, oi)
        # same direction = +1, opposite = -1
        assert result.iloc[0] == 1.0   # price up + oi up = same
        assert result.iloc[1] == 1.0   # price down + oi down = same
        assert result.iloc[2] == 1.0   # price up + oi up = same


class TestNetDealerGamma:
    def test_basic(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10"] * 3,
            "Gamma": [0.001, 0.002, 0.001],
            "open_interest": [100, 200, 150],
            "multiplier": [10, 10, 10],
            "f": [5500, 5500, 5500],
            "cp": ["C", "C", "P"],
        })
        result = net_dealer_gamma(data)
        assert "net_gex" in result.columns

    def test_gex_flip_level(self):
        data = pd.DataFrame({
            "Gamma": [0.001, 0.002, 0.001, 0.0005],
            "open_interest": [100, 200, 300, 400],
            "multiplier": [10, 10, 10, 10],
            "k": [5400, 5500, 5600, 5700],
            "cp": ["C", "C", "P", "P"],
        })
        result = gex_flip_level(data)
        # May or may not find a flip level depending on data
        assert result is None or isinstance(result, float)


class TestThetaGammaRatio:
    def test_basic(self):
        # theta=-0.05, gamma=0.001, spot=5500, iv=0.20
        result = theta_gamma_ratio(-0.05, 0.001, 5500, 0.20)
        assert result > 0  # TGR positive

    def test_zero_gamma(self):
        result = theta_gamma_ratio(-0.05, 0.0, 5500, 0.20)
        assert np.isnan(result)


class TestSkewZScore:
    def test_basic(self):
        skew = pd.DataFrame({
            "A": np.random.randn(100).cumsum() * 0.001 + 0.01,
        }, index=pd.date_range("2025-01-01", periods=100))
        result = skew_zscore(skew, window=60)
        assert len(result) == 100

    def test_skew_momentum(self):
        skew = pd.DataFrame({
            "A": np.random.randn(100).cumsum() * 0.001 + 0.01,
        }, index=pd.date_range("2025-01-01", periods=100))
        result = skew_momentum(skew, short_window=5, long_window=20)
        assert len(result) == 100


class TestIVTermCurvature:
    def test_basic(self):
        near = pd.Series([0.20, 0.21, 0.22])
        mid = pd.Series([0.18, 0.19, 0.20])
        far = pd.Series([0.22, 0.23, 0.24])
        result = iv_term_structure_curvature(near, mid, far)
        assert (result < 0).all()  # mid cheap


class TestRealizedVolAsymmetry:
    def test_basic(self):
        returns = pd.Series(np.random.randn(100) * 0.02)
        result = realized_vol_asymmetry(returns, window=30)
        assert "vol_up" in result.columns
        assert "vol_down" in result.columns
        assert "vol_ratio" in result.columns
        assert "realized_skew" in result.columns


class TestVWAPlIV:
    def test_basic(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10"] * 3,
            "bs_iv": [0.20, 0.18, 0.16],
            "volume": [100, 200, 300],
        })
        result = vwap_iv(data)
        # VWAP = (0.20*100 + 0.18*200 + 0.16*300) / 600
        expected = (20 + 36 + 48) / 600
        assert abs(result.iloc[0] - expected) < 0.001


class TestCrossProductIVSpread:
    def test_basic(self):
        iv = pd.DataFrame({
            "A": [0.20, 0.21, 0.22],
            "B": [0.18, 0.19, 0.20],
        }, index=pd.date_range("2025-01-01", periods=3))
        result = cross_product_iv_spread(iv, "A", "B")
        assert all(abs(v - 0.02) < 1e-10 for v in result.values)

    def test_zscore(self):
        iv = pd.DataFrame({
            "A": np.random.randn(100).cumsum() * 0.01 + 0.20,
            "B": np.random.randn(100).cumsum() * 0.01 + 0.18,
        }, index=pd.date_range("2025-01-01", periods=100))
        result = cross_product_iv_spread_zscore(iv, "A", "B", window=30)
        assert len(result) == 100


class TestAdvancedGreekFactors:
    def test_portfolio_color(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10"],
            "color": [0.001],
            "weight": [2],
            "multiplier": [10],
        })
        result = portfolio_color(data)
        assert result.iloc[0] == 0.02

    def test_portfolio_veta(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10"],
            "veta": [-0.01],
            "weight": [3],
            "multiplier": [10],
        })
        result = portfolio_veta(data)
        assert result.iloc[0] == -0.3

    def test_portfolio_speed(self):
        data = pd.DataFrame({
            "dt": ["2025-01-10"],
            "speed": [0.0001],
            "weight": [5],
            "multiplier": [10],
        })
        result = portfolio_speed(data)
        assert result.iloc[0] == 0.005
