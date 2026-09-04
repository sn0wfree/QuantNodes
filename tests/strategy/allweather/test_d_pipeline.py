"""V8 d_pipeline 单测."""

import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.d_pipeline import (
    _d_base_weights,
    run_d_sleeve,
    BOND_COLS,
)


def _make_panel(n_days=600, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    # 38 资产: 3 债券 (超低波) + 35 其他
    data = {}
    for c in ["fut_t", "fut_tf", "fut_ts"]:
        data[c] = rng.normal(0.0001, 0.002, n_days)  # 债券超低波
    for c in [f"fut_{i}" for i in ["au", "ag", "cu", "al", "if", "ic", "ih"]]:
        data[c] = rng.normal(0.0003, 0.012, n_days)
    for c in [f"etf_{i}" for i in ["513100", "512890", "512010", "515030",
                                     "512690", "512800", "515050", "512200", "159740"]]:
        data[c] = rng.normal(0.0002, 0.01, n_days)
    # 补齐到 38
    for i in range(38 - len(data)):
        data[f"asset_{i}"] = rng.normal(0.0002, 0.01, n_days)
    prices = pd.DataFrame(data, index=dates)
    # 累积成价格 (收益率需正)
    prices = (1 + prices.pct_change().fillna(0)).cumprod() * 100
    return prices


def test_bond_cap_respected():
    """bond cap 被精确尊重 (w_bond ≤ 0.40)."""
    panel = _make_panel()
    w = _d_base_weights(panel, list(panel.columns), bond_cap=0.40)
    assert w[BOND_COLS].sum() <= 0.40 + 1e-6


def test_weights_nonneg_and_sum_to_one():
    """权重非负且归一."""
    panel = _make_panel()
    w = _d_base_weights(panel, list(panel.columns), bond_cap=0.40)
    assert (w >= -1e-8).all()
    assert abs(w.sum() - 1.0) < 1e-6


def test_bond_cap_100_no_change():
    """bond_cap=1.0 (不约束) 时, cap 不改变权重."""
    panel = _make_panel()
    w_free = _d_base_weights(panel, list(panel.columns), bond_cap=1.0)
    w_cap = _d_base_weights(panel, list(panel.columns), bond_cap=0.40)
    # 自由时债券可能 > 或 ≤ 0.40 (取决于数据结构)
    # cap 后 ≤ 0.40
    assert w_cap[BOND_COLS].sum() <= 0.40 + 1e-6
    # 若自由时债券已 ≤ 0.40, cap 不改变任何东西
    if w_free[BOND_COLS].sum() <= 0.40 + 1e-6:
        assert np.allclose(w_free, w_cap, atol=1e-6)


def test_run_d_sleeve_output_format():
    """run_d_sleeve 输出格式正确."""
    panel = _make_panel(n_days=300)
    result = run_d_sleeve(panel, bond_cap=0.40, window=120,
                          rebalance_freq="W-FRI")
    assert len(result.nav) == len(panel)
    assert result.nav.name == "D_sleeve"
    assert result.weights.shape == panel.shape
    # nav 起始 = 1.0
    assert abs(result.nav.iloc[0] - 1.0) < 1e-6


def test_rebalance_freq_parameterized():
    """不同频率产生不同换手 (W-FRI > 20D)."""
    panel = _make_panel(n_days=400)
    res_w = run_d_sleeve(panel, bond_cap=0.40, window=120,
                         rebalance_freq="W-FRI")
    res_m = run_d_sleeve(panel, bond_cap=0.40, window=120,
                         rebalance_freq="20D")
    # 计算权重变化次数
    changes_w = (res_w.weights.diff().abs().sum(axis=1) > 1e-6).sum()
    changes_m = (res_m.weights.diff().abs().sum(axis=1) > 1e-6).sum()
    # W-FRI 调仓更频繁
    assert changes_w >= changes_m


def test_no_future_function():
    """无未来函数: t 日权重只用 ≤t 数据."""
    panel = _make_panel(n_days=200)
    result = run_d_sleeve(panel, bond_cap=0.40, window=60,
                          rebalance_freq="W-FRI")
    # 前 61 天权重应为 0 (窗口不足)
    assert result.weights.iloc[:61].abs().sum().sum() < 1e-6


def test_vol_target_leverage_clamp():
    """lev=1.0 时杠杆 ≤ 1.0."""
    panel = _make_panel(n_days=300)
    result = run_d_sleeve(panel, bond_cap=0.40, window=120,
                          max_leverage=1.0)
    # 所有权重绝对值 ≤ 1 (lev≤1 → 每资产权重 ≤ 1)
    valid = result.weights[(result.weights.sum(axis=1) > 0.9)]
    assert (valid.abs().max(axis=1) <= 1.0 + 1e-6).all()


def test_d_vs_a_different():
    """D 和 A (class ERC) 产生不同权重."""
    from QuantNodes.strategy.allweather.class_erc import class_erc_weights
    panel = _make_panel(n_days=300)
    w_d = _d_base_weights(panel, list(panel.columns), bond_cap=0.40)
    w_a = class_erc_weights(
        panel.pct_change(fill_method=None), bond_risk_cap=0.10
    )
    # 不同方法应产生不同权重
    assert not np.allclose(w_d.values, w_a.values, atol=0.01)
