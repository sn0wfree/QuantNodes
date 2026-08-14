# coding=utf-8
"""波动率信号测试."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.vol_signals import (
    IVSSlopeResult,
    PCIVRatioResult,
    Regime,
    RegimeResult,
    VolSignalGenerator,
    compute_ivs_from_chain,
    compute_ivs_slope_ols,
    compute_pc_iv_ratio,
    compute_pc_iv_ratio_from_chain,
)
# OMA is a simple function, import directly
from QuantNodes.strategy.cta.options.vol_signals import compute_ivs_slope_oma


class TestIVSSlope:
    def test_ols_basic(self):
        strikes = np.array([5400, 5500, 5600, 5700, 5800])
        ivs = np.array([0.20, 0.18, 0.16, 0.15, 0.14])
        slope, r_sq, n = compute_ivs_slope_ols(strikes, ivs, spot=5500)
        assert slope < 0  # negative skew (OTM puts more expensive)
        assert r_sq > 0.8
        assert n == 5

    def test_ols_too_few(self):
        slope, r_sq, n = compute_ivs_slope_ols(np.array([5400]), np.array([0.2]), 5500)
        assert n == 0

    def test_oma_basic(self):
        result = compute_ivs_slope_oma(put_iv=0.25, call_iv=0.18)
        assert result == 0.07

    def test_compute_ivs_from_chain(self):
        data = pd.DataFrame({
            "k": [5300, 5400, 5500, 5600, 5700],
            "cp": ["P", "P", "C", "C", "C"],
            "bs_iv": [0.25, 0.22, 0.18, 0.16, 0.15],
        })
        result = compute_ivs_from_chain(data, spot=5500, date="2025-01-10")
        assert result is not None
        assert result.n_points == 5
        assert isinstance(result, IVSSlopeResult)


class TestPCIVRatio:
    def test_basic(self):
        ratio = compute_pc_iv_ratio(
            np.array([0.25, 0.22]),
            np.array([0.18, 0.16]),
        )
        assert ratio > 1  # put IV higher than call IV

    def test_empty(self):
        ratio = compute_pc_iv_ratio(np.array([]), np.array([0.18]))
        assert np.isnan(ratio)

    def test_from_chain(self):
        data = pd.DataFrame({
            "k": [5400, 5500, 5600],
            "cp": ["P", "C", "C"],
            "bs_iv": [0.22, 0.18, 0.16],
        })
        result = compute_pc_iv_ratio_from_chain(data)
        assert result is not None
        assert result.pc_ratio > 1


class TestRegime:
    def test_low_vol_no_drop(self):
        # Flat IV + HV below IV → low vol, no drop
        iv_panel = pd.DataFrame({"A": [0.12] * 20},
                                index=pd.date_range("2025-01-01", periods=20))
        hv_panel = pd.DataFrame({"A": [0.08] * 20},
                                index=pd.date_range("2025-01-01", periods=20))
        gen = VolSignalGenerator(iv_panel, hv_panel, iv_pct_window=10)
        regime = gen.compute_regime("A", pd.Timestamp("2025-01-20"))
        assert regime == Regime.LOW_VOL_NO_DROP

    def test_high_vol_drop(self):
        # IV rising sharply + HV > IV → high vol + drop
        iv_vals = [0.10] * 10 + list(np.linspace(0.10, 0.25, 10))
        hv_vals = [0.08] * 10 + list(np.linspace(0.08, 0.30, 10))
        iv_panel = pd.DataFrame({"A": iv_vals},
                                index=pd.date_range("2025-01-01", periods=20))
        hv_panel = pd.DataFrame({"A": hv_vals},
                                index=pd.date_range("2025-01-01", periods=20))
        gen = VolSignalGenerator(iv_panel, hv_panel, iv_pct_window=10)
        regime = gen.compute_regime("A", pd.Timestamp("2025-01-20"))
        assert regime == Regime.HIGH_VOL_DROP

    def test_signal_panel(self):
        iv_panel = pd.DataFrame({"A": np.random.randn(20).cumsum() * 0.01 + 0.15},
                                index=pd.date_range("2025-01-01", periods=20))
        hv_panel = pd.DataFrame({"A": np.random.randn(20).cumsum() * 0.01 + 0.12},
                                index=pd.date_range("2025-01-01", periods=20))
        gen = VolSignalGenerator(iv_panel, hv_panel, iv_pct_window=10)
        df = gen.compute_signal_panel("A", iv_panel.index)
        assert len(df) == 20
        assert "regime" in df.columns
        assert df["regime"].between(0, 3).all()
