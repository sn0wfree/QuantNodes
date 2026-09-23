"""Carry sleeve 单测 (§5.48)."""

import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.carry_sleeve import (
    build_carry_sleeve, combine_with_d, month_end_mask)


def _make_data(n_days=400, n_assets=8, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"fut_{c}" for c in "abcdefgh"][:n_assets]
    rets = rng.normal(0, 0.01, (n_days, n_assets))
    panel = pd.DataFrame(1 + rets, index=idx, columns=cols).cumprod()
    # carry: 部分产品正, 部分负, 缓慢变化
    base = np.linspace(-0.05, 0.10, n_assets)
    carry = pd.DataFrame(
        np.tile(base, (n_days, 1)) + rng.normal(0, 0.002, (n_days, n_assets)),
        index=idx, columns=cols)
    return panel, carry


def test_month_end_mask():
    idx = pd.bdate_range("2022-01-03", periods=60)
    m = month_end_mask(idx)
    # 每个自然月恰好一个 True
    assert m.groupby([idx.year, idx.month]).sum().eq(1).all()


def test_carry_sleeve_dollar_neutral_and_gross():
    panel, carry = _make_data()
    port, w = build_carry_sleeve(panel, carry, freq="ME")
    # 调仓日: 美元中性 + gross ≤ cap
    rb = month_end_mask(panel.index)
    w_rb = w[rb]
    assert (w_rb.sum(axis=1).abs() < 1e-9).all()
    assert (w_rb.abs().sum(axis=1) <= 1.0 + 1e-9).all()
    # 非调仓日持仓不变
    assert port.index.isin(panel.index).all()


def test_carry_sleeve_positive_on_persistent_carry():
    """carry 恒定且与收益正相关 → sleeve 应为正收益."""
    rng = np.random.default_rng(9)
    idx = pd.bdate_range("2022-01-03", periods=500)
    cols = [f"fut_{c}" for c in "abcdefgh"]
    # 高 carry 资产给正漂移
    # 漂移与 carry 同向 (高 carry → 高漂移), 与 carry 向量 [-0.08..0.12] 对齐
    drift = np.array([-0.003, -0.002, -0.001, -0.0005,
                      0.0005, 0.001, 0.002, 0.003])
    rets = drift + rng.normal(0, 0.003, (500, 8))
    panel = pd.DataFrame(1 + rets, index=idx, columns=cols).cumprod()
    carry = pd.DataFrame(
        np.tile(np.linspace(-0.08, 0.12, 8), (500, 1)),
        index=idx, columns=cols)
    port, _ = build_carry_sleeve(panel, carry, freq="ME")
    assert port.sum() > 0


def test_combine_with_d_weights_and_leverage():
    idx = pd.bdate_range("2022-01-03", periods=300)
    nav_d = pd.Series(np.linspace(1.0, 1.3, 300), index=idx)
    cs = pd.Series(np.full(300, 0.0004), index=idx)
    a = combine_with_d(nav_d, cs, carry_weight=0.2, leverage=1.0)
    b = combine_with_d(nav_d, cs, carry_weight=0.2, leverage=2.0)
    # 杠杆放大收益 (近似)
    assert b.iloc[-1] > a.iloc[-1]
    # wc=0 → 等于 D (无杠杆)
    z = combine_with_d(nav_d, cs, carry_weight=0.0, leverage=1.0)
    pd.testing.assert_series_equal(
        z.reset_index(drop=True), nav_d.reset_index(drop=True),
        check_names=False, rtol=1e-9)


def test_carry_sleeve_smoothing_reduces_turnover():
    """信号平滑 (§5.51) 降低换手, 且保持美元中性."""
    panel, carry = _make_data(n_days=600)
    _, w1 = build_carry_sleeve(panel, carry, freq="ME", smooth=1)
    _, w20 = build_carry_sleeve(panel, carry, freq="ME", smooth=20)
    to1 = w1.diff().abs().sum(axis=1).sum()
    to20 = w20.diff().abs().sum(axis=1).sum()
    assert to20 < to1
    rb = month_end_mask(panel.index)
    assert (w20[rb].sum(axis=1).abs() < 1e-9).all()
    assert (w20[rb].abs().sum(axis=1) <= 1.0 + 1e-9).all()
