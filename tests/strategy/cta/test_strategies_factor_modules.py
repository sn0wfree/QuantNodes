# coding=utf-8
"""多因子模块子策略测试 (策略 v2)."""
import numpy as np
import pytest

from QuantNodes.strategy.cta.strategies.factor_modules import (
    DEFAULT_PRODUCTS,
    ComboConfig,
    FactorModuleComboStrategy,
    MeanReversionModuleStrategy,
    ModuleConfig,
    OscillatorModuleStrategy,
    TrendAdvancedModuleStrategy,
)


class TestModuleStrategies:
    """三个模块子策略共用行为 (参数化)."""

    @pytest.mark.parametrize("cls", [
        OscillatorModuleStrategy,
        TrendAdvancedModuleStrategy,
        MeanReversionModuleStrategy,
    ])
    def test_construct(self, cls):
        s = cls(products=["AU", "RB"], start="2020-01-01", end="2020-06-30")
        assert s.products == ["AU", "RB"]

    @pytest.mark.parametrize("cls", [
        OscillatorModuleStrategy,
        TrendAdvancedModuleStrategy,
        MeanReversionModuleStrategy,
    ])
    def test_panel_loads(self, cls):
        s = cls(products=["AU"], start="2020-01-01")
        s.load_panel()
        assert "AU" in s._price_panel.columns
        assert len(s._price_panel) > 100

    @pytest.mark.parametrize("cls", [
        OscillatorModuleStrategy,
        TrendAdvancedModuleStrategy,
        MeanReversionModuleStrategy,
    ])
    def test_signals_valid(self, cls):
        s = cls(products=["AU", "RB"], start="2020-01-01")
        s.load_panel()
        sig = s._signals()
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})
        # 有实际交易信号
        assert (sig != 0).sum().sum() > 100

    @pytest.mark.parametrize("cls", [
        OscillatorModuleStrategy,
        TrendAdvancedModuleStrategy,
        MeanReversionModuleStrategy,
    ])
    def test_factor_signals_attribution(self, cls):
        """因子级归因面板: 每个因子都有独立信号面板."""
        s = cls(products=["AU"], start="2020-01-01")
        s.load_panel()
        s._signals_raw()
        assert set(s.factor_signals) == {name for name, *_ in cls.FACTORS}
        for name, panel in s.factor_signals.items():
            assert panel.shape == s._price_panel.shape

    @pytest.mark.parametrize("cls", [
        OscillatorModuleStrategy,
        TrendAdvancedModuleStrategy,
        MeanReversionModuleStrategy,
    ])
    def test_positions_integer(self, cls):
        s = cls(products=["AU", "RB"], start="2020-01-01")
        s.load_panel()
        pos = s._positions()
        non_zero = pos[pos != 0].stack()
        assert len(non_zero) > 0
        assert all(float(x).is_integer() for x in non_zero)

    @pytest.mark.parametrize("cls", [
        OscillatorModuleStrategy,
        TrendAdvancedModuleStrategy,
        MeanReversionModuleStrategy,
    ])
    def test_run_completes(self, cls):
        s = cls(products=["AU", "RB"], start="2020-01-01")
        result = s.run()
        assert result.metrics["Sharpe"] != 0 or result.metrics["Years"] > 0
        assert len(result.nav) > 100

    def test_run_factor_attribution(self):
        s = OscillatorModuleStrategy(products=["AU"], start="2020-01-01")
        result = s.run_factor("macd")
        assert result.metrics["Years"] > 0

    def test_signal_correlation_shape(self):
        s = OscillatorModuleStrategy(products=["AU"], start="2020-01-01")
        s.load_panel()
        corr = s.signal_correlation()
        assert corr.shape == (4, 4)  # MACD/Stoch/CCI/WilliamsR


class TestDefaults:
    def test_default_products(self):
        assert len(DEFAULT_PRODUCTS) == 10
        assert "AU" in DEFAULT_PRODUCTS and "IF" in DEFAULT_PRODUCTS


class TestCombo:
    def test_combo_config_defaults(self):
        cfg = ComboConfig()
        assert cfg.module_weights == (0.3, 0.5, 0.2)

    def test_combo_signals_valid(self):
        s = FactorModuleComboStrategy(products=["AU", "RB"], start="2020-01-01")
        s.load_panel()
        sig = s._signals()
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})
        assert (sig != 0).sum().sum() > 100

    def test_combo_module_signals(self):
        s = FactorModuleComboStrategy(products=["AU"], start="2020-01-01")
        s.load_panel()
        s._signals_raw()
        assert set(s.module_signals) == {
            "OscillatorModuleStrategy",
            "TrendAdvancedModuleStrategy",
            "MeanReversionModuleStrategy",
        }

    def test_combo_run(self):
        s = FactorModuleComboStrategy(products=["AU", "RB"], start="2020-01-01")
        result = s.run()
        assert result.metrics["Years"] > 0

    def test_module_correlation_shape(self):
        s = FactorModuleComboStrategy(products=["AU", "RB"], start="2020-01-01")
        s.load_panel()
        corr = s.module_correlation()
        assert corr.shape == (3, 3)
        # 对角 = 1
        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_module_correlation_low_cross(self):
        """跨模块相关应低于组内 (业界闸门: r > 0.7 无分散价值)."""
        s = FactorModuleComboStrategy(products=["AU", "RB"], start="2020-01-01")
        s.load_panel()
        corr = s.module_correlation()
        cross = corr.values[~np.eye(3, dtype=bool)]
        assert cross.max() < 0.7


class TestModuleConfig:
    def test_defaults(self):
        cfg = ModuleConfig()
        assert cfg.target_vol == 0.15
        assert cfg.initial_cash == 2_000_000
        assert cfg.z_window == 252
