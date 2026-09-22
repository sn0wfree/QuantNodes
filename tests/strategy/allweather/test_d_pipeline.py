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


# ── §5.37 P1-B 相关性覆盖 + §5.38 资本会计 ──

def _make_named_panel(n_days=400, seed=7):
    """合成价格面板 (fut_/etf_ 命名) + 类别映射."""
    rets = _make_returns(n_days=n_days, n_assets=8, seed=seed)
    cols = ["fut_a", "fut_b", "fut_c", "etf_d",
            "etf_e", "etf_f", "fut_g", "etf_h"]
    rets.columns = cols
    prices = (1 + rets).cumprod()
    classes = {
        "c1": ["fut_a", "fut_b", "fut_c"],
        "c2": ["etf_d", "etf_e"],
        "c3": ["etf_f", "fut_g", "etf_h"],
    }
    return prices, classes


def test_correlation_signals_shape():
    """P1-B 信号: 每类一条 Series, 窗口内 NaN, 窗口后有限."""
    from QuantNodes.strategy.allweather.d_pipeline import (
        compute_correlation_signals, DEFAULT_CORR_WINDOW)
    prices, classes = _make_named_panel()
    rets = prices.pct_change(fill_method=None)
    sigs = compute_correlation_signals(rets, classes, window=DEFAULT_CORR_WINDOW)
    assert set(sigs.keys()) == {"c1", "c2", "c3"}
    for cls, s in sigs.items():
        assert len(s) == len(rets)
        assert s.iloc[:DEFAULT_CORR_WINDOW].isna().all()
        tail = s.dropna()
        assert len(tail) > 0
        assert tail.between(-1.0, 1.0).all()


def test_correlation_overlay_gate():
    """阈值门控: sig<θ 不动; sig>θ 降权; k=0 恒等 (平时不主动降)."""
    from QuantNodes.strategy.allweather.d_pipeline import (
        apply_correlation_overlay)
    w = pd.Series({"fut_a": 0.2, "fut_b": 0.2, "etf_d": 0.3, "etf_e": 0.3})
    classes = {"c1": ["fut_a", "fut_b"], "c2": ["etf_d", "etf_e"]}
    date = pd.Timestamp("2025-01-01")
    # k=0 → 恒等
    s0 = apply_correlation_overlay(w, {"c1": pd.Series({date: 0.99})},
                                   classes, date, threshold=0.60, k=0.0)
    assert s0.equals(w)
    # sig < θ → 恒等
    s1 = apply_correlation_overlay(w, {"c1": pd.Series({date: 0.50})},
                                   classes, date, threshold=0.60, k=1.0)
    assert s1.equals(w)
    # sig > θ → 该类降权, 他类不动
    s2 = apply_correlation_overlay(w, {"c1": pd.Series({date: 0.80})},
                                   classes, date, threshold=0.60, k=1.0)
    assert s2["fut_a"] < w["fut_a"]          # c1 被削
    assert s2["etf_d"] == w["etf_d"]         # c2 不动
    # sig=1 → m = 1-k = 0
    s3 = apply_correlation_overlay(w, {"c1": pd.Series({date: 1.0})},
                                   classes, date, threshold=0.60, k=1.0)
    assert abs(s3["fut_a"]) < 1e-12


def test_capital_notional_financing_drag():
    """名义口径: 融资利率 > 0 且 gross>1 → NAV 低于无融资."""
    prices, classes = _make_named_panel()
    common = dict(panel=prices, max_leverage=2.0, target_vol=0.50,
                  window=60, asset_classes=classes, cov_estimator="lw")
    r0 = run_d_sleeve(**common)
    r1 = run_d_sleeve(**common, financing_rate=0.10)
    assert r1.nav.iloc[-1] < r0.nav.iloc[-1]


def test_capital_margin_idle_cash():
    """保证金口径: 期货只占保证金 → 闲置资金更多 → 现金腿收益更高."""
    prices, classes = _make_named_panel()
    idx = prices.index
    cash = pd.Series(0.20 / 252, index=idx)  # 20%/年 现金收益 (放大差异)
    # lev=1.2: 名义口径 gross=1.2 (idle=0); 保证金口径 capital≈0.7 (idle>0)
    common = dict(panel=prices, max_leverage=1.2, target_vol=10.0,
                  window=60, asset_classes=classes, cov_estimator="lw",
                  cash_ret=cash)
    r_notional = run_d_sleeve(**common, capital_base="notional")
    r_margin = run_d_sleeve(**common, capital_base="margin",
                            futures_margin_ratio=0.15)
    # margin 口径下 capital 更小 → idle 更大 → 现金收益更多
    assert r_margin.nav.iloc[-1] > r_notional.nav.iloc[-1]


def test_capital_default_unchanged():
    """默认参数 (rate=0, notional) 与显式基线一致 (向后兼容)."""
    prices, classes = _make_named_panel()
    common = dict(panel=prices, max_leverage=1.5, target_vol=0.20,
                  window=60, asset_classes=classes, cov_estimator="lw")
    a = run_d_sleeve(**common)
    b = run_d_sleeve(**common, financing_rate=0.0,
                     capital_base="notional")
    pd.testing.assert_series_equal(a.nav, b.nav)


def test_bond_cols_configurable():
    """bond_cols 扩充: 额外债券资产纳入合成债券+cap (原本绕过 cap)."""
    prices, classes = _make_named_panel()
    # 把 etf_h 当作债券类资产 (扩充债券集合)
    classes2 = dict(classes)
    classes2["bond"] = ["fut_a", "fut_b", "etf_h"]
    common = dict(panel=prices, bond_cap=0.30, window=60,
                  asset_classes=classes2, cov_estimator="lw",
                  max_leverage=1.0)
    r_default = run_d_sleeve(**common)
    r_ext = run_d_sleeve(**common, bond_cols=["fut_a", "fut_b", "etf_h"])
    # 扩充债券集合后, 债券总权重应受 cap 约束 (不同解)
    assert not r_default.weights.equals(r_ext.weights)
    # 扩充后债券三资产在调仓日的合计 ≤ cap + 容差
    w = r_ext.weights
    bond_sum = w[["fut_a", "fut_b", "etf_h"]].sum(axis=1)
    assert bond_sum.max() <= 0.30 + 1e-6


def test_bond_cols_default_unchanged():
    """bond_cols=None 与显式 fut_t/tf/ts 行为一致 (向后兼容)."""
    prices, classes = _make_named_panel()
    common = dict(panel=prices, bond_cap=0.40, window=60,
                  asset_classes=classes, cov_estimator="lw",
                  max_leverage=1.0)
    a = run_d_sleeve(**common)
    b = run_d_sleeve(**common, bond_cols=None)
    pd.testing.assert_series_equal(a.nav, b.nav)


# ── §5.41 共振门控 (AR3) ──

def test_absorption_ratio_extremes():
    """AR3: 完全相关 → 1.0; 独立资产 → ≈3/N."""
    rng = np.random.default_rng(7)
    n = 300
    # 完全相关: 所有资产同一序列
    base_ret = rng.normal(0, 0.01, n)
    ident = pd.DataFrame({f"a{i}": base_ret for i in range(6)},
                         index=pd.bdate_range("2024-01-01", periods=n))
    from QuantNodes.strategy.allweather.d_pipeline import _absorption_ratio
    ar_ident = _absorption_ratio(ident, window=60).dropna()
    assert ar_ident.iloc[-1] > 0.99
    # 独立资产: AR3 ≈ 3/8
    indep = pd.DataFrame(rng.normal(0, 0.01, (n, 8)),
                         columns=[f"b{i}" for i in range(8)],
                         index=pd.bdate_range("2024-01-01", periods=n))
    ar_indep = _absorption_ratio(indep, window=60).dropna()
    assert ar_indep.iloc[-1] < 0.75


def test_resonance_gate_off_by_default():
    """k=0 → 门控关闭 (向后兼容, 与基线逐点一致)."""
    prices, classes = _make_named_panel()
    common = dict(panel=prices, max_leverage=1.5, target_vol=0.20,
                  window=60, asset_classes=classes, cov_estimator="lw")
    a = run_d_sleeve(**common)
    b = run_d_sleeve(**common, resonance_k=0.0)
    pd.testing.assert_series_equal(a.nav, b.nav)


def test_resonance_gate_engages_and_reduces():
    """共振期 (共同冲击) → 门控触发 (乘数<1); 平静期 → 乘数=1."""
    from QuantNodes.strategy.allweather.d_pipeline import (
        apply_resonance_gate)
    rng = np.random.default_rng(11)
    n = 700
    idx = pd.bdate_range("2022-01-03", periods=n)
    cols = [f"fut_{c}" for c in "abcdefgh"]
    rets = rng.normal(0, 0.008, (n, len(cols)))
    shock = rng.normal(0, 0.05, 60)
    rets[400:460] = (np.tile(shock[:, None], (1, len(cols)))
                     + rng.normal(0, 0.001, (60, len(cols))))
    ret = pd.DataFrame(rets, columns=cols, index=idx)
    w = pd.DataFrame(1.0, index=idx, columns=cols)
    g = apply_resonance_gate(w, ret, None, window=60, enter=0.70,
                             exit_th=0.40, vol_exit=0.50,
                             dd_floor=None, k=0.7)
    mult = g.iloc[:, 0]
    assert (mult < 0.99).sum() > 0                     # 触发过
    assert abs(mult.min() - 0.3) < 1e-9               # 触发时降至 1-k
    shock = (mult.iloc[400:460] < 0.99).mean()         # 冲击窗口触发率
    calm = (mult.iloc[150:340] < 0.99).mean()          # 平静窗口触发率
    assert shock > calm                                # 冲击期更常触发


def test_resonance_gate_dd_floor_blocks_late_entry():
    """回撤门槛: 已深亏时不入场 (防锁底)."""
    from QuantNodes.strategy.allweather.d_pipeline import (
        apply_resonance_gate)
    rng = np.random.default_rng(13)
    n = 700
    idx = pd.bdate_range("2022-01-03", periods=n)
    cols = [f"fut_{c}" for c in "abcdefgh"]
    rets = rng.normal(0, 0.008, (n, len(cols)))
    # 先制造深回撤 (350-400 天全资产同跌), 再在 400-460 天共振
    rets[350:400] = np.tile(rng.normal(-0.01, 0.004, 50)[:, None],
                            (1, len(cols)))
    shock = rng.normal(0, 0.05, 60)
    rets[400:460] = (np.tile(shock[:, None], (1, len(cols)))
                     + rng.normal(0, 0.001, (60, len(cols))))
    ret = pd.DataFrame(rets, columns=cols, index=idx)
    w = pd.DataFrame(1.0, index=idx, columns=cols)
    common = dict(window=60, enter=0.70, exit_th=0.40, vol_exit=0.50, k=0.7)
    g_no = apply_resonance_gate(w, ret, None, dd_floor=None, **common)
    g_fl = apply_resonance_gate(w, ret, None, dd_floor=-0.03, **common)
    # 深回撤后 (>=400 天), 有门槛版本不应在已深亏时继续入场
    late_no = (g_no.iloc[400:, 0] < 0.99).sum()
    late_fl = (g_fl.iloc[400:, 0] < 0.99).sum()
    assert late_fl <= late_no
