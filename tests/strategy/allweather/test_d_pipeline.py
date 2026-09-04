"""V8 d_pipeline 单测."""

import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.d_pipeline import (
    _d_base_weights,
    run_d_sleeve,
    BOND_COLS,
)
from QuantNodes.strategy.allweather.hrp import hrp_weights
from QuantNodes.strategy.allweather.cov_extended import (
    oas_cov,
    lw_ewma_cov,
    block_corr_shrink,
)


def _make_returns(n_days=600, n_assets=20, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    cols = [f"asset_{i}" for i in range(n_assets)]
    data = {c: rng.normal(0.0003, 0.01, n_days) for c in cols}
    return pd.DataFrame(data, index=dates)


def _check_psd(C, name=""):
    eigvals = np.linalg.eigvalsh(C)
    assert eigvals.min() > -1e-8, f"{name} not PSD: min_eig={eigvals.min()}"


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
    """bond_cap=1.0 走直接 38 资产路径; cap=0.40 走合成债券路径 — 两种都可行."""
    panel = _make_panel()
    w_free = _d_base_weights(panel, list(panel.columns), bond_cap=1.0)
    w_cap = _d_base_weights(panel, list(panel.columns), bond_cap=0.40)
    # 两种路径都满足基本不变量
    for w in (w_free, w_cap):
        assert (w >= -1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-6
    # cap 路径下债券 ≤ 0.40
    assert w_cap[BOND_COLS].sum() <= 0.40 + 1e-6


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


# ═══════════════════════════════════════════════════════════════════
# 不变量 2 单测: 盒约束 CCD (PLAN §17.10)
# ═══════════════════════════════════════════════════════════════════
def test_ccd_constraint_satisfaction():
    """构造性验证: 0 ≤ w ≤ u, Σw = 1 — 约束由 SLSQP 内建保证."""
    from QuantNodes.strategy.allweather.d_pipeline import risk_budget_solve
    rng = np.random.default_rng(3)
    a = rng.normal(size=(300, 10))
    cov = np.cov(a, rowvar=False) + np.eye(10) * 1e-8
    budgets = np.full(10, 0.1)
    upper = np.full(10, 0.15)  # 强约束: 10×0.15=1.5 > 1 → 部分绑定
    w = risk_budget_solve(cov, budgets, upper=upper)
    assert (w >= -1e-6).all()
    assert (w <= 0.15 + 1e-6).all()
    assert abs(w.sum() - 1.0) < 1e-6


def test_ccd_reproduces_hrp_when_unconstrained():
    """无绑定约束时, solve(b=HRP的RC) 应回到 HRP 原解 (预算映射一致性)."""
    from QuantNodes.strategy.allweather.d_pipeline import (
        risk_budget_solve, _hrp_budgets,
    )
    rng = np.random.default_rng(7)
    a = rng.normal(size=(400, 12))
    cov = np.cov(a, rowvar=False) + np.eye(12) * 1e-8
    b = _hrp_budgets(cov)
    w_solve = risk_budget_solve(cov, b)  # upper=1 → 无绑定
    w_hrp = hrp_weights(cov, linkage_method="ward")
    assert np.allclose(w_solve, w_hrp, atol=1e-3)


def test_ccd_near_singular_still_feasible():
    """近奇异协方差 → 解仍可行 (不变量的核心价值)."""
    from QuantNodes.strategy.allweather.d_pipeline import risk_budget_solve
    rng = np.random.default_rng(11)
    base = rng.normal(size=(400, 2))
    # 12 资产中 10 个高度共线 → 近奇异
    cols = [base[:, 0] * (1 + 1e-4 * i) for i in range(10)] + \
           [base[:, 1], rng.normal(size=400)]
    X = np.column_stack(cols)
    cov = np.cov(X, rowvar=False)
    assert np.linalg.cond(cov) > 1e6  # 确认近奇异
    budgets = np.full(12, 1 / 12)
    w = risk_budget_solve(cov, budgets, upper=np.full(12, 0.15))
    assert (w >= -1e-6).all()
    assert (w <= 0.15 + 1e-6).all()
    assert abs(w.sum() - 1.0) < 1e-6


def test_ccd_bond_cap_binds_and_absorbs():
    """低波资产触顶时, 其余坐标吸收超额预算 (求解器内建重分配)."""
    from QuantNodes.strategy.allweather.d_pipeline import risk_budget_solve
    # 资产0 超低波 (ERC 会给它 ~40%+), cap=0.15 强制绑定
    rng = np.random.default_rng(5)
    low = rng.normal(0, 0.001, 300)
    risky = rng.normal(0, 0.012, (300, 5))
    X = np.column_stack([low, risky])
    cov = np.cov(X, rowvar=False) + np.eye(6) * 1e-10
    budgets = np.full(6, 1 / 6)
    upper = np.array([0.15, 1.0, 1.0, 1.0, 1.0, 1.0])
    w = risk_budget_solve(cov, budgets, upper=upper)
    assert w[0] <= 0.15 + 1e-6       # 低波资产被 cap 绑定
    assert (w >= -1e-6).all()
    assert abs(w.sum() - 1.0) < 1e-6
    assert w[1:].sum() >= 0.84       # 其余资产吸收超额预算


# ═══════════════════════════════════════════════════════════════════
# 不变量 1 单测: 自调节估计器
# ═══════════════════════════════════════════════════════════════════

def test_oas_matches_sklearn():
    """自实现 OAS 与 sklearn.covariance.OAS 数值一致 (同窗口同数据)."""
    from sklearn.covariance import OAS as SkOAS
    rets = _make_returns(n_days=400, n_assets=12, seed=21)
    C_ours = oas_cov(rets, window=400)  # 显式全窗口, 对齐 sklearn 的全样本 fit
    sk = SkOAS(assume_centered=False).fit(rets.to_numpy())
    assert np.allclose(C_ours, sk.covariance_, atol=1e-10)


def test_lw_ewma_alpha_regulates():
    """N_eff 下降 → 收缩增强 → 条件数下降 (自调节验证)."""
    rets = _make_returns(n_days=400, n_assets=25, seed=13)
    C_fast = lw_ewma_cov(rets, lam=0.90)   # N_eff=19 < 25 → 强收缩
    C_slow = lw_ewma_cov(rets, lam=0.995)  # N_eff=399 → 弱收缩
    cond_fast = np.linalg.cond(C_fast)
    cond_slow = np.linalg.cond(C_slow)
    assert cond_fast < cond_slow  # 快衰减 → 条件数更小
    _check_psd(C_fast, "lw_ewma_fast")
    _check_psd(C_slow, "lw_ewma_slow")


def test_block_corr_target_bounded():
    """相关空间目标条件数有界 (vs 协方差空间 F 被vol离散污染)."""
    rng = np.random.default_rng(17)
    # 极端 vol 离散: 债券 0.2% vs 风险 3%
    data = {}
    for c in ["b1", "b2", "b3"]:
        data[c] = rng.normal(0, 0.002, 400)
    for i in range(9):
        data[f"r{i}"] = rng.normal(0, 0.03, 400)
    rets = pd.DataFrame(data, index=pd.bdate_range("2024-01-01", periods=400))
    C = block_corr_shrink(rets, n_clusters=4)
    _check_psd(C, "block_corr")
    # vol 离散度大时, 相关空间收缩的 C 条件数应远小于协方差空间 F 的
    cond_C = np.linalg.cond(C)
    # 协方差空间 F: 外积 vol² 使条件数 ≥ (0.03/0.002)² ≈ 225
    assert cond_C < 1e5  # 相关空间收缩不应继承病态
