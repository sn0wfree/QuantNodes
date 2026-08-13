# coding=utf-8
"""多因子模块子策略测试 (策略 v3: 4 组方向子策略)."""
import numpy as np
import pytest

from QuantNodes.strategy.cta.strategies.factor_modules import (
    DEFAULT_PRODUCTS,
    ComboConfig,
    EventCalendarModuleStrategy,
    FactorModuleComboStrategy,
    IndustrialChainModuleStrategy,
    MacroModuleStrategy,
    MeanReversionModuleStrategy,
    ModuleConfig,
    TermBasisModuleStrategy,
    TrendMomentumModuleStrategy,
    VolumeFlowModuleStrategy,
)

# 测试小池含产业链配方品种 (steel_profit_rb: RB/I/J, crush_profit: M/A, mto: PP/MA)
_TEST_POOL = ["AU", "RB", "M", "CU", "I", "TA", "MA", "C", "J", "A", "PP"]

ALL_MODULES = [
    TrendMomentumModuleStrategy,
    MeanReversionModuleStrategy,
    VolumeFlowModuleStrategy,
    EventCalendarModuleStrategy,
    TermBasisModuleStrategy,
    MacroModuleStrategy,
    IndustrialChainModuleStrategy,
]


class TestModuleStrategies:
    """四个方向子策略共用行为 (参数化)."""

    @pytest.mark.parametrize("cls", ALL_MODULES)
    def test_construct(self, cls):
        s = cls(products=_TEST_POOL, start="2020-01-01", end="2020-06-30")
        assert s.products == _TEST_POOL

    @pytest.mark.parametrize("cls", ALL_MODULES)
    def test_panel_loads(self, cls):
        s = cls(products=_TEST_POOL, start="2020-01-01")
        s.load_panel()
        assert "AU" in s._price_panel.columns
        assert len(s._price_panel) > 100
        assert s._vols is not None and s._ois is not None

    @pytest.mark.parametrize("cls", ALL_MODULES)
    def test_signals_valid(self, cls):
        s = cls(products=_TEST_POOL, start="2020-01-01")
        s.load_panel()
        sig = s._signals()
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})
        # 有实际交易信号
        assert (sig != 0).sum().sum() > 100

    @pytest.mark.parametrize("cls", ALL_MODULES)
    def test_factor_signals_attribution(self, cls):
        """因子级归因面板: 每个因子都有独立信号面板."""
        s = cls(products=_TEST_POOL, start="2020-01-01")
        s.load_panel()
        s._signals_raw()
        assert set(s.factor_signals) == {name for name, *_ in cls.FACTORS}
        for name, panel in s.factor_signals.items():
            assert panel.shape == s._price_panel.shape

    @pytest.mark.parametrize("cls", ALL_MODULES)
    def test_positions_integer(self, cls):
        s = cls(products=_TEST_POOL, start="2020-01-01")
        s.load_panel()
        pos = s._positions()
        non_zero = pos[pos != 0].stack()
        assert len(non_zero) > 0
        assert all(float(x).is_integer() for x in non_zero)

    @pytest.mark.parametrize("cls", ALL_MODULES)
    def test_run_completes(self, cls):
        s = cls(products=_TEST_POOL, start="2020-01-01")
        result = s.run()
        assert result.metrics["Years"] > 0
        assert len(result.nav) > 100

    def test_run_factor_attribution(self):
        s = TrendMomentumModuleStrategy(products=_TEST_POOL, start="2020-01-01")
        result = s.run_factor("macd")
        assert result.metrics["Years"] > 0

    def test_signal_correlation_shape(self):
        s = TrendMomentumModuleStrategy(products=_TEST_POOL, start="2020-01-01")
        s.load_panel()
        corr = s.signal_correlation()
        assert corr.shape == (len(TrendMomentumModuleStrategy.FACTORS),
                              len(TrendMomentumModuleStrategy.FACTORS))


class TestDefaults:
    def test_default_products(self):
        # 全部有交易量品种 (日均量 ≥ 1 万手), 含产业链配方品种
        assert len(DEFAULT_PRODUCTS) >= 70
        assert "AU" in DEFAULT_PRODUCTS and "IF" in DEFAULT_PRODUCTS
        for p in ["J", "A", "Y", "HC", "JM", "PX", "PF", "EG", "LH"]:
            assert p in DEFAULT_PRODUCTS  # 产业链配方品种全覆盖

    def test_chain_formulas_covered(self):
        """8 个产业链配方品种应全部在默认池中."""
        from QuantNodes.strategy.cta.factors.industrial_chain import CHAIN_FORMULAS
        pool = set(DEFAULT_PRODUCTS)
        for name, f in CHAIN_FORMULAS.items():
            need = set(f["long"]) | set(f["short"])
            assert need <= pool, f"{name} 缺 {need - pool}"

    def test_module_sizes(self):
        """7 组子策略覆盖主要因子类别."""
        total = sum(len(cls.FACTORS) for cls in ALL_MODULES)
        assert total >= 70
        assert len(TrendMomentumModuleStrategy.FACTORS) >= 20
        assert len(VolumeFlowModuleStrategy.FACTORS) >= 20


class TestCombo:
    def test_combo_config_defaults(self):
        cfg = ComboConfig()
        assert len(cfg.module_weights) == 7
        # 默认等权
        assert abs(sum(cfg.module_weights) - 1.0) < 1e-9
        assert len(set(round(w, 6) for w in cfg.module_weights)) == 1
        assert cfg.filter_enabled is True
        assert cfg.module_weighting == "fixed"

    def test_tvpr_inv_vol_combo(self):
        """组内 TV-PR × 组间 inv_vol (docs/94 §10.6 最优配置)."""
        s = FactorModuleComboStrategy(
            products=_TEST_POOL, start="2019-01-01", end="2021-12-31",
            config=ComboConfig(
                aggregation="tvpr", module_weighting="inv_vol",
                tvpr_min_history=120,
            ),
        )
        sig = s._signals()
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})
        # N=11 品种 ≥ 10 样本下限, 应有信号
        assert (sig != 0).sum().sum() > 0

    def test_tvpr_inv_vol_run(self):
        s = FactorModuleComboStrategy(
            products=_TEST_POOL, start="2019-01-01", end="2021-12-31",
            config=ComboConfig(
                aggregation="tvpr", module_weighting="inv_vol",
                tvpr_min_history=120,
            ),
        )
        result = s.run()
        assert result.metrics["Years"] > 0

    def test_inv_vol_module_weighting(self):
        """组间 inv_vol: 模块信号波动率倒数加权."""
        s = FactorModuleComboStrategy(
            products=_TEST_POOL, start="2020-01-01",
            config=ComboConfig(module_weighting="inv_vol"),
        )
        s.load_panel()
        combo = s._signals_raw()
        assert combo.shape == s._price_panel.shape
        # 前 agg_window 天为等权 fallback, 之后 inv_vol
        assert not combo.isna().all().all()

    def test_momentum_filter(self):
        """信号动量过滤 (docs/94 §10.7): 组合信号 N 日均 < 0 → 降仓."""
        # window=0: 关闭, 与 baseline 一致
        s_off = FactorModuleComboStrategy(
            products=_TEST_POOL, start="2020-01-01",
            config=ComboConfig(momentum_filter_window=0),
        )
        s_off.load_panel()
        sig_off = s_off._signals()
        # window=60: 启用
        s_on = FactorModuleComboStrategy(
            products=_TEST_POOL, start="2020-01-01",
            config=ComboConfig(momentum_filter_window=60, momentum_filter_coef=0.0),
        )
        s_on.load_panel()
        sig_on = s_on._signals()
        # 过滤生效: 启用后非零信号应 ≤ 关闭（空仓率升高）
        assert (sig_on != 0).sum().sum() <= (sig_off != 0).sum().sum()

    def test_combo_signals_valid(self):
        s = FactorModuleComboStrategy(products=_TEST_POOL, start="2020-01-01")
        s.load_panel()
        sig = s._signals()
        assert set(sig["AU"].unique()).issubset({-1.0, 0.0, 1.0})
        assert (sig != 0).sum().sum() > 100

    def test_combo_module_signals(self):
        s = FactorModuleComboStrategy(products=_TEST_POOL, start="2020-01-01")
        s.load_panel()
        s._signals_raw()
        assert set(s.module_signals) == {cls.__name__ for cls in ALL_MODULES}

    def test_combo_run(self):
        s = FactorModuleComboStrategy(products=_TEST_POOL, start="2020-01-01")
        result = s.run()
        assert result.metrics["Years"] > 0

    def test_module_correlation_shape(self):
        s = FactorModuleComboStrategy(products=_TEST_POOL, start="2020-01-01")
        s.load_panel()
        corr = s.module_correlation()
        assert corr.shape == (7, 7)
        np.testing.assert_allclose(np.diag(corr), 1.0)

    def test_module_correlation_low_cross(self):
        """跨模块相关应低于 0.7 (业界闸门)."""
        s = FactorModuleComboStrategy(products=_TEST_POOL, start="2020-01-01")
        s.load_panel()
        corr = s.module_correlation()
        cross = corr.values[~np.eye(7, dtype=bool)]
        assert cross.max() < 0.7

    def test_filter_off(self):
        """过滤层可关闭."""
        s = FactorModuleComboStrategy(
            products=_TEST_POOL, start="2020-01-01",
            config=ComboConfig(filter_enabled=False),
        )
        assert s._filter is None
        result = s.run()
        assert result.metrics["Years"] > 0


class TestModuleConfig:
    def test_defaults(self):
        cfg = ModuleConfig()
        assert cfg.target_vol == 0.15
        assert cfg.initial_cash == 2_000_000
