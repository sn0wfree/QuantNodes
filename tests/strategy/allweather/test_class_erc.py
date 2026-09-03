"""V7 class_erc 单测 (PLAN §16)."""

import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.class_erc import (
    ASSET_CLASSES,
    BOND_CLASS,
    _erc_weights,
    class_erc_weights,
    class_composite_returns,
    risk_contributions,
)


def _make_returns(n_days: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    cols = sorted({c for ms in ASSET_CLASSES.values() for c in ms})
    data = {c: rng.normal(0.0003, 0.01, n_days) for c in cols}
    return pd.DataFrame(data, index=dates)


# ── _erc_weights ──

def test_erc_equal_vols_uncorrelated():
    """等波动不相关资产 → ERC = 等权."""
    cov = np.eye(4) * 0.02
    w = _erc_weights(cov)
    assert np.allclose(w, np.full(4, 0.25), atol=1e-6)


def test_erc_risk_contributions_equal():
    """随机正定矩阵上 ERC 风险贡献相等."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(500, 6))
    cov = np.cov(a, rowvar=False) + np.eye(6) * 1e-8
    w = _erc_weights(cov)
    rc = risk_contributions(w, cov)
    assert np.allclose(rc, 1 / 6, atol=1e-4)


def test_erc_with_budgets():
    """非等预算: RC 占比 ≈ 预算占比."""
    rng = np.random.default_rng(11)
    a = rng.normal(size=(400, 3))
    cov = np.cov(a, rowvar=False) + np.eye(3) * 1e-8
    budgets = np.array([0.5, 0.3, 0.2])
    w = _erc_weights(cov, budgets)
    rc = risk_contributions(w, cov)
    assert np.allclose(rc, budgets, atol=1e-4)


# ── risk_contributions ──

def test_risk_contributions_sum_to_one():
    rng = np.random.default_rng(3)
    cov = np.cov(rng.normal(size=(300, 5)), rowvar=False)
    w = np.array([0.4, 0.3, 0.2, 0.08, 0.02])
    rc = risk_contributions(w, cov)
    assert abs(rc.sum() - 1.0) < 1e-9


# ── class_composite_returns ──

def test_composite_returns_shape_and_causality():
    rets = _make_returns()
    comp = class_composite_returns(rets, ASSET_CLASSES)
    assert list(comp.columns) == list(ASSET_CLASSES.keys())
    assert len(comp) == len(rets)
    # 早期窗口不足 → NaN
    assert comp.iloc[10].isna().all()


# ── class_erc_weights ──

def test_class_erc_sums_to_one_nonneg():
    rets = _make_returns()
    w = class_erc_weights(rets, ASSET_CLASSES, bond_risk_cap=0.3)
    assert abs(w.sum() - 1.0) < 1e-8
    assert (w >= 0).all()
    assert (w[w.index.str.startswith("etf_")] >= 0).all()


def test_bond_cap_binds_and_reduces_bond_weight():
    """cap < 自然 RC (1/K=0.1) → 触发, 债券权重与 RC 均下降 (合成低波债券)."""
    rets = _make_returns(seed=99)
    for b in ASSET_CLASSES[BOND_CLASS]:
        rets[b] = rets[b] * 0.15
    w_free = class_erc_weights(rets, ASSET_CLASSES, bond_risk_cap=1.0)
    w_cap = class_erc_weights(rets, ASSET_CLASSES, bond_risk_cap=0.05)
    bond_free = w_free[ASSET_CLASSES[BOND_CLASS]].sum()
    bond_cap = w_cap[ASSET_CLASSES[BOND_CLASS]].sum()
    assert bond_cap < bond_free
    # 验证 cap 后债券类 RC ≈ 0.05 (类级口径)
    comp = class_composite_returns(rets, ASSET_CLASSES)
    sub = comp.dropna(axis=0, how="any").iloc[-252:]
    cov = np.cov(sub.to_numpy(), rowvar=False) + np.eye(sub.shape[1]) * 1e-12
    cls_w = []
    names = list(sub.columns)
    for cls in names:
        members = ASSET_CLASSES[cls]
        cls_w.append(w_cap[members].sum())
    rc = risk_contributions(np.array(cls_w), cov)
    rc_bond = rc[names.index(BOND_CLASS)]
    assert rc_bond <= 0.05 + 1e-3


def test_bond_cap_loose_equals_plain_erc():
    """cap 宽松 (0.5 > 自然 RC 0.1) → 与不约束一致."""
    rets = _make_returns(seed=5)
    w1 = class_erc_weights(rets, ASSET_CLASSES, bond_risk_cap=1.0)
    w2 = class_erc_weights(rets, ASSET_CLASSES, bond_risk_cap=0.5)
    assert np.allclose(w1, w2)


def test_late_listed_asset_gets_zero_then_weight():
    """晚上市资产先 0 后有权重 (渐入池)."""
    rets = _make_returns()
    late = "etf_159740"
    rets.loc[rets.index[:400], late] = np.nan
    w_early = class_erc_weights(rets.iloc[:300], ASSET_CLASSES)
    assert w_early[late] == 0.0
    w_full = class_erc_weights(rets, ASSET_CLASSES)
    assert w_full[late] > 0.0


def test_custom_classes_override():
    """外部类别映射覆盖默认 (可配置性, 用户要求)."""
    rets = _make_returns()
    classes = {
        "bond_rate": ["fut_t", "fut_tf", "fut_ts"],
        "risk_all": [c for ms in ASSET_CLASSES.values() for c in ms
                     if c not in ASSET_CLASSES[BOND_CLASS]],
    }
    w = class_erc_weights(rets, classes, bond_risk_cap=0.25)
    assert abs(w.sum() - 1.0) < 1e-8
    assert w[ASSET_CLASSES[BOND_CLASS]].sum() > 0
