# coding=utf-8
"""宏观过滤器单元测试.

覆盖:
    - credit_spread 顺风判断
    - copper_oil_ratio 顺风判断
    - 6 维组合的 OR/AND 逻辑
    - 数据加载容错 (parquet 不存在时优雅降级)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.cta.regime.macro_gate import external_commodity_favorable
from QuantNodes.strategy.cta.regime.macro_long_gate import macro_long_gate


# ── 共享测试数据 ──────────────────────────────────────────────

@pytest.fixture
def px_panel() -> pd.DataFrame:
    """200 天合成价格面板 (单品种 AU)."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    px = pd.DataFrame(
        {"AU": 100 * np.cumprod(1 + rng.normal(0.001, 0.01, n))},
        index=idx,
    )
    return px


# ── cn_us_spread 维度测试 ──────────────────────────────────────

def test_credit_spread_favorable_widening(px_panel, monkeypatch):
    """中美利差扩大 → 顺风=True (禁空)."""
    dates = px_panel.index
    rising = pd.Series(np.linspace(1.0, 3.0, len(dates)), index=dates)

    def fake_load(self, name):
        if name == "dxy":
            return None
        if name == "cn_us_spread":
            return rising
        if name == "gold_oil_corr":
            return pd.Series(0.5, index=dates)
        if name == "vix":
            return None
        raise ValueError(name)

    from QuantNodes.strategy.cta import data_loader as dl_mod
    monkeypatch.setattr(dl_mod.CtaDataLoader, "load_macro_series", fake_load)
    monkeypatch.setattr(
        "QuantNodes.strategy.cta.regime.macro_gate._us10y_series",
        lambda: pd.Series(2.0, index=dates),
    )

    fav = external_commodity_favorable(
        use_vix=False, use_crb_proxy=False,
        px_panel=px_panel,
    )
    late = fav.iloc[150:]
    assert late.sum() > 0, "Late-stage widening should be favorable"


def test_credit_spread_narrowing_not_favorable(px_panel, monkeypatch):
    """中美利差收窄 → 不顺风."""
    dates = px_panel.index
    falling = pd.Series(np.linspace(3.0, 0.5, len(dates)), index=dates)

    def fake_load(self, name):
        if name == "dxy":
            return None
        if name == "cn_us_spread":
            return falling
        if name == "gold_oil_corr":
            return pd.Series(0.5, index=dates)
        if name == "vix":
            return None
        raise ValueError(name)

    from QuantNodes.strategy.cta import data_loader as dl_mod
    monkeypatch.setattr(dl_mod.CtaDataLoader, "load_macro_series", fake_load)
    monkeypatch.setattr(
        "QuantNodes.strategy.cta.regime.macro_gate._us10y_series",
        lambda: pd.Series(2.0, index=dates),
    )

    fav = external_commodity_favorable(
        use_vix=False, use_crb_proxy=False,
        px_panel=px_panel,
    )
    assert fav.sum() < len(fav) * 0.5


# ── gold_oil_corr 维度测试 ─────────────────────────────────────

def test_copper_oil_ratio_haven_risk(px_panel, monkeypatch):
    """金油相关性下降超阈值 → 避险 → 顺风 (禁空)."""
    dates = px_panel.index
    falling_corr = pd.Series(np.linspace(0.5, -0.5, len(dates)), index=dates)

    def fake_load(self, name):
        if name == "dxy":
            return None
        if name == "cn_us_spread":
            return pd.Series(1.0, index=dates)
        if name == "gold_oil_corr":
            return falling_corr
        if name == "vix":
            return None
        raise ValueError(name)

    from QuantNodes.strategy.cta import data_loader as dl_mod
    monkeypatch.setattr(dl_mod.CtaDataLoader, "load_macro_series", fake_load)
    monkeypatch.setattr(
        "QuantNodes.strategy.cta.regime.macro_gate._us10y_series",
        lambda: pd.Series(2.0, index=dates),
    )

    fav = external_commodity_favorable(
        use_vix=False, use_crb_proxy=False,
        px_panel=px_panel,
    )
    late = fav.iloc[150:]
    assert late.sum() > 0


def test_copper_oil_ratio_stable(px_panel, monkeypatch):
    """金油相关性稳定 → 不触发避险."""
    dates = px_panel.index
    stable_corr = pd.Series(0.5, index=dates)

    def fake_load(self, name):
        if name == "dxy":
            return None
        if name == "cn_us_spread":
            return pd.Series(1.0, index=dates)
        if name == "gold_oil_corr":
            return stable_corr
        if name == "vix":
            return None
        raise ValueError(name)

    from QuantNodes.strategy.cta import data_loader as dl_mod
    monkeypatch.setattr(dl_mod.CtaDataLoader, "load_macro_series", fake_load)
    monkeypatch.setattr(
        "QuantNodes.strategy.cta.regime.macro_gate._us10y_series",
        lambda: pd.Series(2.0, index=dates),
    )

    fav = external_commodity_favorable(
        use_vix=False, use_crb_proxy=False,
        px_panel=px_panel,
    )
    assert fav.sum() == 0


# ── 6 维组合测试 ──────────────────────────────────────────────

def test_six_dim_combined_or(px_panel, monkeypatch):
    """6 维任一顺风 → 整体顺风."""
    dates = px_panel.index

    def fake_load(self, name):
        if name == "dxy":
            return pd.Series(np.linspace(100, 80, len(dates)), index=dates)
        if name == "cn_us_spread":
            return pd.Series(1.0, index=dates)
        if name == "gold_oil_corr":
            return pd.Series(0.5, index=dates)
        if name == "vix":
            return None
        raise ValueError(name)

    from QuantNodes.strategy.cta import data_loader as dl_mod
    monkeypatch.setattr(dl_mod.CtaDataLoader, "load_macro_series", fake_load)
    monkeypatch.setattr(
        "QuantNodes.strategy.cta.regime.macro_gate._us10y_series",
        lambda: pd.Series(2.0, index=dates),
    )

    fav = external_commodity_favorable(
        use_vix=False, use_crb_proxy=False,
        px_panel=px_panel,
    )
    assert fav.sum() > 0


def test_load_credit_spread_data_loader():
    """真实 parquet 加载能正常工作 (cn_us_spread)."""
    from QuantNodes.strategy.cta.data_loader import CtaDataLoader
    dl = CtaDataLoader()
    try:
        s = dl.load_macro_series(name="cn_us_spread")
        assert len(s) > 100
        assert s.name == "cn_us_spread"
    except FileNotFoundError:
        pytest.skip("cn_us_spread parquet not found")


def test_load_gold_oil_corr_data_loader():
    """真实 parquet 加载能正常工作 (gold_oil_corr)."""
    from QuantNodes.strategy.cta.data_loader import CtaDataLoader
    dl = CtaDataLoader()
    try:
        s = dl.load_macro_series(name="gold_oil_corr")
        assert len(s) > 100
        assert s.name == "gold_oil_corr"
    except FileNotFoundError:
        pytest.skip("gold_oil_corr parquet not found")


# ── macro_long_gate 测试 ──────────────────────────────────────

def test_long_gate_off_returns_all_true(px_panel):
    """mode=off → 全 True."""
    out = macro_long_gate(px_panel, mode="off")
    assert out.shape == px_panel.shape
    assert bool(out.all().all())


def test_long_gate_external_shape(px_panel):
    """mode=external → 返回正确 shape (即使有缺失数据也不崩)."""
    out = macro_long_gate(px_panel, mode="external")
    assert out.shape == (len(px_panel), px_panel.shape[1])
    assert isinstance(out.dtypes.iloc[0], bool) or str(out.dtypes.iloc[0]) == "bool"


def test_long_gate_invalid_mode_raises(px_panel):
    """非法 mode → ValueError."""
    with pytest.raises(ValueError, match="mode"):
        macro_long_gate(px_panel, mode="invalid")
