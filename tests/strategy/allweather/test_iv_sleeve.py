"""IVTS sleeve + 通用 sleeve 单测 (§5.53)."""

import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.carry_sleeve import (
    build_carry_sleeve, build_factor_sleeve, combine_sleeves, month_end_mask)
from QuantNodes.strategy.allweather.iv_sleeve import (
    build_ivts_sleeve, load_ivts)


def _panel(n_days=400, n_assets=8, seed=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    cols = [f"fut_{c}" for c in "abcdefgh"][:n_assets]
    rets = rng.normal(0, 0.01, (n_days, n_assets))
    return pd.DataFrame(1 + rets, index=idx, columns=cols).cumprod()


def test_load_ivts(tmp_path):
    idx = pd.bdate_range("2022-01-03", periods=50)
    for name, n, f in [("fut_aa", 0.15, 0.18), ("fut_bb", 0.20, 0.22)]:
        pd.DataFrame({"iv_near": n, "iv_far": f,
                      "iv_put": n + 0.01, "iv_call": n - 0.01},
                     index=idx).to_parquet(tmp_path / f"{name}.parquet")
    ivts = load_ivts(tmp_path)
    assert list(ivts.columns) == ["fut_aa", "fut_bb"]
    assert np.allclose(ivts["fut_aa"], 0.03)
    assert np.allclose(ivts["fut_bb"], 0.02)


def test_build_ivts_sleeve_dollar_neutral():
    panel = _panel()
    rng = np.random.default_rng(1)
    ivts = pd.DataFrame(rng.normal(0.04, 0.01, panel.shape),
                        index=panel.index, columns=panel.columns)
    port, w = build_ivts_sleeve(panel, ivts)
    rb = month_end_mask(panel.index)
    assert (w[rb].sum(axis=1).abs() < 1e-9).all()
    assert (w[rb].abs().sum(axis=1) <= 1.0 + 1e-9).all()
    assert port.name == "ivts_sleeve"


def test_build_carry_sleeve_wraps_factor_sleeve():
    """build_carry_sleeve 是 build_factor_sleeve 的薄封装."""
    panel = _panel()
    rng = np.random.default_rng(2)
    sig = pd.DataFrame(rng.normal(0, 0.05, panel.shape),
                       index=panel.index, columns=panel.columns)
    a, _ = build_factor_sleeve(panel, sig, smooth=1)
    b, _ = build_carry_sleeve(panel, sig, smooth=1)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_combine_sleeves_multi():
    idx = pd.bdate_range("2022-01-03", periods=300)
    nav_d = pd.Series(np.linspace(1.0, 1.3, 300), index=idx)
    s1 = pd.Series(np.full(300, 0.003), index=idx)
    s2 = pd.Series(np.full(300, 0.004), index=idx)
    z = combine_sleeves(nav_d, {}, {}, leverage=1.0)
    pd.testing.assert_series_equal(
        z.reset_index(drop=True), nav_d.reset_index(drop=True),
        check_names=False, rtol=1e-6)
    nav = combine_sleeves(nav_d, {"a": s1, "b": s2}, {"a": 0.2, "b": 0.1},
                          leverage=1.0)
    assert nav.iloc[-1] > nav_d.iloc[-1]
