"""V8 D.2 动量倾斜 alpha 层单测 (PLAN §17.14)."""

import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.momentum_tilt import (
    apply_momentum_tilt,
    momentum_score,
    momentum_tilt_factors,
)


def _make_panel(n_days=400, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"a{i:02d}" for i in range(12)]
    data = {c: 100 * (1 + rng.normal(0.0003, 0.01, n_days)).cumprod()
            for c in cols}
    return pd.DataFrame(data, index=dates)


def _make_trending_panel(n_days=400, seed=7):
    """一半资产强趋势向上, 一半向下 (tilt 方向测试); 含债券列 (D 管线必需)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    data = {}
    for i in range(6):
        drift = 0.0015
        data[f"up_{i}"] = 100 * (1 + rng.normal(drift, 0.008, n_days)).cumprod()
    for i in range(6):
        drift = -0.0015
        data[f"dn_{i}"] = 100 * (1 + rng.normal(drift, 0.008, n_days)).cumprod()
    # 债券列 (超低波; D 管线合成债券路径需要 fut_t/tf/ts)
    for c in ["fut_t", "fut_tf", "fut_ts"]:
        data[c] = 100 * (1 + rng.normal(0.0001, 0.002, n_days)).cumprod()
    return pd.DataFrame(data, index=dates)


def test_momentum_score_bounds():
    """score ∈ [−1, 1] 且无 NaN."""
    panel = _make_panel()
    score = momentum_score(panel)
    assert score.notna().all().all()
    assert (score.abs() <= 1.0001).all().all()


def test_momentum_score_direction():
    """趋势面板: up 资产 score > dn 资产 score."""
    panel = _make_trending_panel()
    score = momentum_score(panel)
    up_mean = score[[c for c in panel.columns if c.startswith("up_")]].iloc[-60:].mean().mean()
    dn_mean = score[[c for c in panel.columns if c.startswith("dn_")]].iloc[-60:].mean().mean()
    assert up_mean > dn_mean


def test_tilt_bounds():
    """tilt ∈ [1−β, 1+β]."""
    panel = _make_panel()
    tilt = momentum_tilt_factors(panel, beta=0.20)
    assert (tilt >= 0.80 - 1e-9).all().all()
    assert (tilt <= 1.20 + 1e-9).all().all()


def test_tilt_beta_zero_neutral():
    """β=0 → tilt 恒 1 (开关关闭)."""
    panel = _make_trending_panel()
    tilt = momentum_tilt_factors(panel, beta=0.0)
    assert (tilt == 1.0).all().all()


def test_tilt_direction():
    """趋势面板: up 资产 tilt > 1, dn 资产 tilt < 1 (最近窗口)."""
    panel = _make_trending_panel()
    tilt = momentum_tilt_factors(panel, beta=0.20)
    recent = tilt.iloc[-60:]
    up_tilt = recent[[c for c in panel.columns if c.startswith("up_")]].mean().mean()
    dn_tilt = recent[[c for c in panel.columns if c.startswith("dn_")]].mean().mean()
    assert up_tilt > 1.0
    assert dn_tilt < 1.0


def test_apply_tilt_normalizes():
    """应用后权重归一且非负."""
    rng = np.random.default_rng(3)
    weights = pd.Series(rng.dirichlet(np.ones(6)), index=[f"a{i}" for i in range(6)])
    tilt_row = pd.Series(1.0 + 0.1 * rng.normal(size=6), index=weights.index)
    w = apply_momentum_tilt(weights, tilt_row)
    assert abs(w.sum() - 1.0) < 1e-9
    assert (w >= 0).all()


def test_apply_tilt_nan_neutral():
    """全 NaN → 中性 1.0 → 权重不变 (晚上市处理)."""
    weights = pd.Series([0.5, 0.5], index=["x", "y"])
    tilt_row = pd.Series([np.nan, np.nan], index=["x", "y"])
    w = apply_momentum_tilt(weights, tilt_row)
    assert np.allclose(w, weights)  # 全中性 → 原权重保持


def test_apply_tilt_single_tilted():
    """单资产倾斜: 该资产超配, 另一中性资产被稀释 (倾斜方向正确)."""
    weights = pd.Series([0.5, 0.5], index=["x", "y"])
    tilt_row = pd.Series([1.1, np.nan], index=["x", "y"])
    w = apply_momentum_tilt(weights, tilt_row)
    assert w["x"] > 0.5   # 倾斜资产超配
    assert w["y"] < 0.5   # 中性资产稀释
    assert abs(w.sum() - 1.0) < 1e-9


def test_run_d_sleeve_momentum_beta_wiring():
    """run_d_sleeve momentum_beta 透传: β>0 与 β=0 产生不同 nav."""
    from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
    panel = _make_trending_panel(n_days=300)
    res0 = run_d_sleeve(panel, bond_cap=0.40, window=120,
                        rebalance_freq="W-FRI", cov_estimator="lw",
                        momentum_beta=0.0)
    res1 = run_d_sleeve(panel, bond_cap=0.40, window=120,
                        rebalance_freq="W-FRI", cov_estimator="lw",
                        momentum_beta=0.20)
    assert not np.allclose(res0.nav.values, res1.nav.values)


def test_apply_tilt_preserves_gross_exposure():
    """回归测试 (§5.32 杠杆bug): tilt 保持总敞口 (杠杆语义不被归一化抹掉)."""
    weights = pd.Series([1.15, 1.15, 0], index=["x", "y", "z"])  # sum=2.3 (杠杆)
    tilt_row = pd.Series([1.1, 1.1, 1.0], index=["x", "y", "z"])
    w = apply_momentum_tilt(weights, tilt_row)
    assert abs(w.sum() - 2.3) < 1e-9  # 杠杆保持, 不被归一化回 1.0
    # 相对权重仍被倾斜 (x/y 同倾斜 → 比例不变)
    assert abs(w["x"] / w["y"] - 1.0) < 1e-9
    # 未上市资产保持 0
    assert w["z"] == 0


def test_apply_tilt_delever_preserved():
    """去杠杆 (sum<1) 同样保持: 0.81x 的减杠杆不被 tilt 归一化抹掉."""
    weights = pd.Series([0.405, 0.405], index=["x", "y"])  # sum=0.81
    tilt_row = pd.Series([0.9, 1.1], index=["x", "y"])
    w = apply_momentum_tilt(weights, tilt_row)
    assert abs(w.sum() - 0.81) < 1e-9
    assert w["x"] < w["y"]  # 负动量资产相对低配
