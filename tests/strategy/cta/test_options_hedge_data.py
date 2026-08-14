"""对冲数据测试: 月差/Backwardation 面板."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.options.hedge_data import (
    basis_percentile_panel, load_basis_panel,
)


class TestBasisPanel:
    def test_load_basis_panel_columns(self):
        panel = load_basis_panel(products=["RB"])
        assert not panel.empty
        assert "RB" in panel.columns
        assert panel.index.is_monotonic_increasing

    def test_load_basis_single_date(self):
        panel = load_basis_panel(products=["RB"], start="2025-01-06",
                                 end="2025-01-10")
        assert not panel.empty
        assert (panel.index >= "2025-01-06").all()
        assert (panel.index <= "2025-01-10").all()

    def test_load_basis_unknown_product_returns_empty(self):
        panel = load_basis_panel(products=["XX"])
        assert panel.empty

    def test_percentile_range(self):
        """分位面板值域 [0,1]."""
        rng = pd.date_range("2023-01-01", periods=400, freq="B")
        panel = pd.DataFrame(0.0, index=rng, columns=["RB"])
        # 构造先升水后贴水序列
        panel["RB"] = np.linspace(-50, 50, len(rng))
        pct = basis_percentile_panel(panel, window=252, min_periods=60)
        valid = pct["RB"].dropna()
        assert not valid.empty
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_percentile_high_for_backwardation(self):
        """后期深贴水 → 分位趋近 1."""
        rng = pd.date_range("2023-01-01", periods=400, freq="B")
        panel = pd.DataFrame(0.0, index=rng, columns=["RB"])
        panel["RB"] = np.linspace(-50, 50, len(rng))
        pct = basis_percentile_panel(panel, window=252, min_periods=60)
        tail = pct["RB"].dropna()[-10:]
        assert tail.mean() > 0.8

    def test_empty_panel(self):
        assert basis_percentile_panel(pd.DataFrame()).empty
        assert load_basis_panel.__doc__ is not None

