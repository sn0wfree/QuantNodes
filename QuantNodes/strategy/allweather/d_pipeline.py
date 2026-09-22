"""D sleeve: 资产级配置管线 (设计文稿核心版 v2, PLAN §17.10).

管线 v2 (不变量 2 — 约束由求解器内建):
  协方差 (AVS/LW/OAS/lw_ewma/block_corr 可选)
    → HRP(Ward) 权重 → 风险预算 b = RC(w_HRP)/ΣRC
    → 合成债券资产 (3 债券 inv-vol 合成, 盒约束 u=0.40)
    → risk_budget_solve (SLSQP, 约束由求解器内建)
    → 合成权重按 inv-vol 摊回 3 债券
  约束 (w≥0, Σw=1, w_bond≤cap) 由求解器构造性满足 — 无重分配/无裁剪.

v1 的 RC share 重分配已删除 (近奇异协方差下无界放大, 见 §5.24 勘误).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from .cov_avs import hierarchical_avs
from .hrp import hrp_weights
from .backtest import _rebalance_mask, ANN

# ── 常量 ──
BOND_COLS = ["fut_t", "fut_tf", "fut_ts"]
MIN_ASSETS = 5
DEFAULT_BOND_CAP = 0.40
SYNTH_BOND = "__synth_bond__"
DEFAULT_CORR_WINDOW = 60
DEFAULT_CORR_THRESHOLD = 0.60
# 共振门控默认 (§5.41; 仅结构破碎池启用)
DEFAULT_RESONANCE_WINDOW = 60
DEFAULT_RESONANCE_ENTER = 0.70
DEFAULT_RESONANCE_EXIT = 0.40
DEFAULT_RESONANCE_VOL_EXIT = 0.50
DEFAULT_RESONANCE_DD_FLOOR = -0.03


# ── 输出结构 ──
@dataclass
class DSleeveResult:
    nav: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series | None = None


# ── 不变量 2: 盒约束风险预算求解器 ──
def risk_budget_solve(cov: np.ndarray, budgets: np.ndarray,
                      upper: np.ndarray | None = None,
                      max_iter: int = 500) -> np.ndarray:
    """盒约束风险预算权重 (SLSQP, 约束由求解器内建保证).

    目标: min Σ_i (RC_i(w) - b_i)²,  s.t. Σw=1, 0 ≤ w ≤ u.
    SLSQP 同时满足等式与边界约束 (tol 1e-8) — 构造性可行, 无需事后裁剪.
    目标非凸, 但以无约束 ERC 解为暖启动, 实践中收敛可靠.

    Args:
        cov: (K x K) 正定协方差
        budgets: (K,) 风险预算占比, sum=1
        upper: (K,) 每资产权重上限; None = 1.0

    Returns:
        (K,) 权重, Σ=1 (±1e-9), 0 ≤ w ≤ u
    """
    from scipy.optimize import minimize

    K = len(budgets)
    b = np.asarray(budgets, dtype=float)
    b = b / b.sum()
    u = np.ones(K) if upper is None else np.clip(np.asarray(upper, float), 0.0, 1.0)
    if u.sum() < 1.0 - 1e-9:
        # 不可行 (Σu<1): 退化为上限比例分配
        return u / u.sum()

    def obj(w):
        mrc = cov @ w
        var = float(w @ mrc)
        if var <= 0:
            return 1e6
        rc_share = w * mrc / var
        return float(np.sum((rc_share - b) ** 2))

    # 暖启动: 无约束 ERC 方向 (既有 CCD 核)
    diag = np.diag(cov).copy()
    diag = np.where(diag > 0, diag, 1e-12)
    x = 1.0 / np.sqrt(diag)
    x = x / x.sum()
    for _ in range(200):
        x_prev = x.copy()
        for i in range(K):
            s = float(cov[i] @ x) - diag[i] * x[i]
            disc = max(s * s + 4.0 * diag[i] * b[i], 0.0)
            x[i] = (-s + np.sqrt(disc)) / (2.0 * diag[i])
        if np.max(np.abs(x - x_prev)) < 1e-10:
            break
    w0 = np.clip(x / x.sum(), 1e-6, u)
    w0 = w0 / w0.sum()

    res = minimize(
        obj, w0, method="SLSQP",
        bounds=[(0.0, float(ui)) for ui in u],
        constraints={"type": "eq",
                     "fun": lambda w: w.sum() - 1.0,
                     "jac": lambda w: np.ones(K)},
        options={"maxiter": max_iter, "ftol": 1e-12},
    )
    w = np.asarray(res.x, dtype=float)
    # 数值清洗 (SLSQP tol 1e-8 内已可行; 只消 -1e-17 浮点尾, 不再归一化)
    w = np.clip(w, 0.0, u)
    drift = w.sum() - 1.0
    if abs(drift) > 1e-6:
        free = w < (u - 1e-9)
        if free.any():
            w[free] -= drift / free.sum()
    return w


# ── 合成债券资产 ──
def _synth_bond_returns(returns: pd.DataFrame,
                        window: int, min_obs: int = 60,
                        bond_cols: list[str] | None = None) -> pd.Series | None:
    """债券类 inv-vol 合成序列 (合成资产用于盒约束); 缺列防御."""
    bond_cols = bond_cols or BOND_COLS
    avail = [c for c in bond_cols
             if c in returns.columns and returns[c].notna().sum() >= min_obs]
    if not avail:
        return None
    sub = returns[avail].iloc[-window:]
    vols = sub.std().replace(0, np.nan).dropna()
    if vols.empty:
        return None
    inv = 1.0 / vols
    w = inv / inv.sum()
    return (sub[w.index].fillna(0.0) * w).sum(axis=1)


def _hrp_budgets(cov: np.ndarray, linkage_method: str = "ward") -> np.ndarray:
    """HRP 权重 → 风险预算 b = RC/ΣRC (层次洞察 → 求解器预算)."""
    w = hrp_weights(cov, linkage_method=linkage_method)
    mrc = cov @ w
    rc = w * mrc
    rc_sum = rc.sum()
    if rc_sum > 0:
        b = np.clip(rc / rc_sum, 0.0, None)
        if b.sum() > 0:
            return b / b.sum()
    return np.full(len(w), 1.0 / len(w))


# ── 核心权重计算 ──
def _d_base_weights(
    returns_hist: pd.DataFrame,
    panel_cols: list[str],
    window: int = 252,
    n_clusters: int | None = None,
    bond_cap: float = DEFAULT_BOND_CAP,
    min_obs_per_asset: int = 60,
    cov_estimator: str = "lw",
    bond_cols: list[str] | None = None,
) -> pd.Series:
    """单次调仓: 协方差 → HRP 预算 → 合成债券 → 盒约束 CCD → 摊回.

    cov_estimator: "lw" / "oas" / "lw_ewma" / "block_corr" / "avs"
    bond_cols: 债券资产集合 (默认 fut_t/tf/ts; 可扩充如 etf_511260)
    """
    bond_cols = bond_cols or BOND_COLS
    from .cov_extended import lw_ewma_cov, oas_cov, block_corr_shrink, \
        nqn_lw2012_cov, poet_cov

    avail = [c for c in panel_cols
             if returns_hist[c].notna().sum() >= min_obs_per_asset]
    if len(avail) < MIN_ASSETS:
        out = pd.Series(0.0, index=panel_cols)
        if avail:
            out[avail] = 1.0 / len(avail)
        return out

    sub = returns_hist[avail].iloc[-window:].dropna(axis=0, how="any")

    # ① 协方差 (可选估计器)
    if cov_estimator == "oas":
        cov_full = oas_cov(sub, window=window)
    elif cov_estimator == "lw_ewma":
        cov_full = lw_ewma_cov(sub, window=window)
    elif cov_estimator == "block_corr":
        cov_full = block_corr_shrink(sub, window=window)
    elif cov_estimator == "nqn":
        cov_full = nqn_lw2012_cov(sub, window=window)
    elif cov_estimator == "poet":
        cov_full = poet_cov(sub, window=window)
    elif cov_estimator == "avs":
        cov_full = hierarchical_avs(sub, window=window)[0]
    elif cov_estimator == "none":
        cov_full = np.cov(sub.to_numpy(), rowvar=False)
    else:
        from .cov_rmt import ledoit_wolf_shrink
        cov_full = ledoit_wolf_shrink(sub, window=window)

    # ② HRP(Ward) → 风险预算
    cov_cols = list(sub.columns)  # sub 的列序 = cov 行列序 (同一 dropna 交集)
    w_hrp = hrp_weights(cov_full, linkage_method="ward")
    mrc = cov_full @ w_hrp
    rc = w_hrp * mrc
    rc_sum = rc.sum()
    budgets_38 = np.clip(rc / rc_sum, 0.0, None) if rc_sum > 0 \
        else np.full(len(w_hrp), 1.0 / len(w_hrp))
    if budgets_38.sum() <= 0:
        budgets_38 = np.full(len(w_hrp), 1.0 / len(w_hrp))
    budgets_38 = budgets_38 / budgets_38.sum()

    # ③ 合成债券资产: 3 债券 → 1 合成 (盒约束 u=bond_cap 直接可执行)
    synth_ret = _synth_bond_returns(returns_hist, window, min_obs_per_asset,
                                    bond_cols=bond_cols)
    bond_in_avail = [c for c in bond_cols if c in avail]

    if synth_ret is None or not bond_in_avail or bond_cap >= 1.0:
        # 无债券或无 cap → 直接 38 资产 CCD (无盒约束, 退化为 HRP 预算 ERC)
        x = risk_budget_solve(cov_full, budgets_38)
        out = pd.Series(0.0, index=panel_cols)
        out[cov_cols] = x
        total = out.sum()
        if total > 0:
            out = out / total
        return out

    # 合成系统: 非债资产 + 1 合成债券
    non_bond = [c for c in cov_cols if c not in bond_cols]
    synth_series = synth_ret.reindex(sub.index).fillna(0.0)
    joint = pd.concat([sub[non_bond], synth_series.rename(SYNTH_BOND)], axis=1)
    # 合成列与已有资产重名防御
    if SYNTH_BOND in joint.columns[:len(non_bond)]:
        joint = pd.concat([sub[non_bond],
                           synth_series.rename(SYNTH_BOND + "_x")], axis=1)
    joint = joint.dropna(axis=0, how="any")
    if len(joint) < 30:
        x = risk_budget_solve(cov_full, budgets_38)
        out = pd.Series(0.0, index=panel_cols)
        out[cov_cols] = x
        total = out.sum()
        if total > 0:
            out = out / total
        return out

    cov_joint = np.cov(joint.to_numpy(), rowvar=False)
    K = cov_joint.shape[0]
    cov_joint += np.eye(K) * 1e-12

    # 预算映射: 非债资产各自预算; 合成债券 = 3 债券预算之和
    bud = []
    bond_budget_sum = 0.0
    bi = 0
    for c in cov_cols:
        if c in bond_cols:
            bud.append(None)  # 占位, 稍后合成
            bond_budget_sum += budgets_38[bi]
        else:
            bud.append(budgets_38[bi])
        bi += 1
    budgets_joint = [v for v in bud if v is not None] + [bond_budget_sum]
    budgets_joint = np.asarray(budgets_joint, dtype=float)
    if budgets_joint.sum() <= 0:
        budgets_joint = np.full(K, 1.0 / K)
    budgets_joint = budgets_joint / budgets_joint.sum()

    # 盒约束: 合成债券 ≤ bond_cap, 其余 ≤ 1.0
    upper = np.ones(K)
    upper[-1] = bond_cap

    x = risk_budget_solve(cov_joint, budgets_joint, upper=upper)

    # ④ 摊回: 合成债券权重按 inv-vol 分给 3 剔债券
    out = pd.Series(0.0, index=panel_cols)
    for j, c in enumerate(non_bond):
        out[c] = x[j]
    w_synth = x[-1]
    bond_sub = returns_hist[bond_in_avail].iloc[-window:]
    vols = bond_sub.std().replace(0, np.nan).dropna()
    if vols.empty or len(vols) != len(bond_in_avail):
        for c in bond_in_avail:
            out[c] = w_synth / len(bond_in_avail)
    else:
        inv = 1.0 / vols
        share = inv / inv.sum()
        for c in bond_in_avail:
            out[c] = w_synth * share[c]

    total = out.sum()
    if total > 0:
        out = out / total
    return out


# ── 相关性危机检测器 (P1-B, §5.37) ──
def compute_correlation_signals(
    returns: pd.DataFrame,
    asset_classes: dict[str, list[str]],
    window: int = DEFAULT_CORR_WINDOW,
) -> dict[str, pd.Series]:
    """计算各类内滚动两两相关系数均值 (结构性风控信号).

    信号含义: 类内相关飙升 → 该类分散化坍缩 → 风险平价前提失效.
    仅对 ≥2 成员的类计算; 阈值过滤在 apply_correlation_overlay 中完成.

    Args:
        returns: (T x N) 收益率面板
        asset_classes: 类别映射
        window: 滚动窗口 (默认 60)

    Returns:
        dict[类名 → Series of avg pairwise corr], 均为 reindexed to returns.index
    """
    dates = returns.index
    signals = {}
    for cls_name, members in asset_classes.items():
        cols = [c for c in members if c in returns.columns]
        if len(cols) < 2:
            continue
        sig = {}
        for i, d in enumerate(dates):
            if i < window:
                sig[d] = np.nan
                continue
            sub = returns.iloc[i - window:i][cols].dropna(axis=0, how="any")
            if sub.shape[0] < 30:
                sig[d] = np.nan
                continue
            c = sub.corr()
            upper = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1)).stack()
            sig[d] = float(upper.mean())
        signals[cls_name] = pd.Series(sig).reindex(dates).ffill()
    return signals


def apply_correlation_overlay(
    weights: pd.Series,
    corr_signals: dict[str, pd.Series],
    asset_classes: dict[str, list[str]],
    date,
    threshold: float = DEFAULT_CORR_THRESHOLD,
    k: float = 0.0,
) -> pd.Series:
    """单日相关性覆盖层: 类内相关 > 阈值 → 线性降权.

    平时 (sig < θ): m=1, 零影响.
    危机 (sig ≥ θ): m = 1 - k × (sig-θ)/(1-θ), 最多降 k.

    与 D.1 正交 (全样本相关 0.103): D.1 是价/量/波微观信号,
    P1-B 是类内相关结构信号 — 不同信息, 安全叠加.
    """
    if k <= 0:
        return weights
    w_out = weights.copy()
    for cls_name, sig_series in corr_signals.items():
        sig = sig_series.get(date, np.nan)
        if not np.isfinite(sig) or sig <= threshold:
            continue
        eff = (sig - threshold) / max(1.0 - threshold, 1e-6)
        m = 1.0 - k * min(eff, 1.0)
        members = [c for c in asset_classes.get(cls_name, []) if c in w_out.index]
        if members and m < 1.0:
            w_out[members] = w_out[members] * m
    return w_out


# ── 共振门控 (AR3 吸收比率, §5.41) ──
def _pit_pct(s: pd.Series, min_periods: int = 120) -> pd.Series:
    """扩张分位 (PIT, 无前视)."""
    return s.expanding(min_periods=min_periods).rank(pct=True)


def _absorption_ratio(returns: pd.DataFrame, window: int = 60) -> pd.Series:
    """AR3: window 日相关阵前 3 主成分解释方差占比 (Kritzman)."""
    out = np.full(len(returns), np.nan)
    vals = returns.to_numpy()
    for i in range(window, len(returns)):
        sub = vals[i - window:i]
        ok = ~np.isnan(sub).any(axis=0)
        k = int(ok.sum())
        if k < 3:
            continue
        sub = sub[:, ok]
        if sub.shape[0] < max(30, k):      # 有效率 <50% 或无足够样本
            continue
        c = np.corrcoef(sub, rowvar=False)
        if not np.isfinite(c).all():
            continue
        ev = np.clip(np.linalg.eigvalsh(c)[::-1], 0, None)
        if ev.sum() > 0:
            out[i] = float(ev[:3].sum() / ev.sum())
    return pd.Series(out, index=returns.index)


def apply_resonance_gate(
    w_exec: pd.DataFrame,
    returns: pd.DataFrame,
    cash_ret: pd.Series | None,
    window: int = DEFAULT_RESONANCE_WINDOW,
    enter: float = DEFAULT_RESONANCE_ENTER,
    exit_th: float = DEFAULT_RESONANCE_EXIT,
    vol_exit: float = DEFAULT_RESONANCE_VOL_EXIT,
    dd_floor: float = DEFAULT_RESONANCE_DD_FLOOR,
    k: float = 0.7,
    financing_rate: float = 0.0,
) -> pd.DataFrame:
    """共振门控 (事件式降杠杆): 入场领先 + 出场双确认.

    入场: AR3 分位 ≥ enter 且 组合回撤 ≥ dd_floor (未深亏才行动 —
          已深亏时 AR3 是滞后信号, 减仓即锁底).
    出场: AR3 分位 ≤ exit_th 且 组合波动分位 ≤ vol_exit (双确认).
    持仓: 事件期间敞口 ×(1−k); 平时 ×1 (零影响).

    信号全部 PIT (扩张分位, shift(1)); 影子 NAV 与引擎同口径
    (成本+融资+现金腿, 名义资本). 仅适用于结构性破碎池.
    """
    dates = w_exec.index
    ar3 = _absorption_ratio(returns.reindex(dates), window=window)
    p_ar3 = _pit_pct(ar3).shift(1).reindex(dates)
    # 影子 NAV (未门控): 与引擎同口径 (成本+融资+现金腿)
    r = returns.reindex(dates)
    rate = pd.Series({c: (0.0001 if c.startswith("fut_") else 0.0005)
                      for c in w_exec.columns})
    gross_prev = w_exec.shift(1)
    shadow = (gross_prev * r).sum(axis=1) \
        - (w_exec.diff().abs() * rate).sum(axis=1).fillna(0.0)
    capital_prev = gross_prev.sum(axis=1)
    if cash_ret is not None:
        cash_daily = cash_ret.reindex(dates).ffill().fillna(0.0)
        shadow = shadow + (1.0 - capital_prev).clip(lower=0.0) * cash_daily
    if financing_rate > 0:
        shadow = shadow - (capital_prev - 1.0).clip(lower=0.0) \
            * financing_rate / ANN
    shadow = (1.0 + shadow.fillna(0.0)).cumprod()
    vol20 = shadow.pct_change().rolling(20).std()
    # 波动分位用滚动窗 (252d): 避免渐进入池期低波值污染扩张分位参考
    p_vol = vol20.rolling(252, min_periods=120).rank(pct=True) \
        .shift(1).reindex(dates)
    dd = (shadow / shadow.cummax() - 1.0).reindex(dates)

    mult = np.ones(len(dates))
    on = False
    for i in range(len(dates)):
        a, v, d = p_ar3.iloc[i], p_vol.iloc[i], dd.iloc[i]
        if not on:
            ok = np.isfinite(a) and a >= enter
            if dd_floor is not None and np.isfinite(d):
                ok = ok and d >= dd_floor
            if ok:
                on = True
        else:
            if np.isfinite(a) and np.isfinite(v) and a <= exit_th and v <= vol_exit:
                on = False
        mult[i] = (1.0 - k) if on else 1.0
    return w_exec.mul(pd.Series(mult, index=dates), axis=0)


def run_d_sleeve(
    panel: pd.DataFrame,
    bond_cap: float = DEFAULT_BOND_CAP,
    window: int = 252,
    n_clusters: int | None = None,
    rebalance_freq: str = "W-FRI",
    target_vol: float = 0.10,
    max_leverage: float = 1.0,
    min_obs_per_asset: int = 60,
    cov_estimator: str = "lw",
    neg_trend_mode: str = "none",
    volume: pd.DataFrame | None = None,
    oi: pd.DataFrame | None = None,
    asset_classes: dict[str, list[str]] | None = None,
    base_threshold: float = 0.5,
    threshold_low: float = 0.3,
    neg_trend_preset: str | None = None,
    momentum_beta: float = 0.0,
    momentum_lookback: int = 60,
    include_rapid: bool = False,
    cash_ret: pd.Series | None = None,
    compute_metrics_fn=None,
    correlation_k: float = 0.0,
    correlation_threshold: float = DEFAULT_CORR_THRESHOLD,
    correlation_window: int = DEFAULT_CORR_WINDOW,
    resonance_k: float = 0.0,
    resonance_ar3_window: int = DEFAULT_RESONANCE_WINDOW,
    resonance_enter: float = DEFAULT_RESONANCE_ENTER,
    resonance_exit: float = DEFAULT_RESONANCE_EXIT,
    resonance_vol_exit: float = DEFAULT_RESONANCE_VOL_EXIT,
    resonance_dd_floor: float = DEFAULT_RESONANCE_DD_FLOOR,
    bond_cols: list[str] | None = None,
    financing_rate: float = 0.0,
    capital_base: str = "notional",
    etf_financing_rate: float | None = None,
    futures_margin_ratio: float = 0.15,
) -> DSleeveResult:
    """运行 D sleeve 回测.

    Args:
        panel: (T x N) 价格面板
        bond_cap: 债券硬上限 (默认0.40, 合成债券资产盒约束)
        window: 协方差估计窗口 (HRP+AVS共用)
        n_clusters: Ward簇数 (None=N//6)
        rebalance_freq: 调仓频率 ("W-FRI" / "20D" / "ME")
        target_vol: 目标波动率
        max_leverage: 杠杆上限
        min_obs_per_asset: 资产渐入池门槛
        cov_estimator: 协方差引擎 ("lw"/"oas"/"lw_ewma"/"block_corr"/"nqn"/"poet"/"avs"/"none")
        neg_trend_mode: 负趋势因子 ("3d"/"trend"/"none"); 非 none 需传 volume/oi
        volume/oi: 量价面板 (neg_trend_mode!='none' 时必需)
        asset_classes: 类别映射 (异质权重)
        compute_metrics_fn: 指标函数 (None = 不计算)
        correlation_k: 相关性覆盖层强度 (0=关闭, 推荐0.60-1.00; §5.37 P1-B)
        correlation_threshold: 类内相关触发阈值 (默认0.60, 信号>阈值才降权)
        correlation_window: 相关性滚动窗口 (默认60日)
        resonance_k: 共振门控强度 (0=关闭; 仅结构破碎池启用, §5.41)
        resonance_ar3_window: AR3 吸收比率窗口 (默认60日)
        resonance_enter: 入场 AR3 分位阈值 (默认0.70)
        resonance_exit: 出场 AR3 分位阈值 (默认0.40)
        resonance_vol_exit: 出场组合波动分位阈值 (默认0.50)
        resonance_dd_floor: 入场回撤门槛 (默认-0.03; 已深亏不入场, 防锁底)
        bond_cols: 债券资产集合 (默认 fut_t/tf/ts; 可扩充, 用于合成债券+cap)
        financing_rate: 融资年利率 (0=关闭; 回购口径 ~2%)
        capital_base: "notional" (gross 口径) / "margin" (期货按保证金口径)
        etf_financing_rate: ETF 腿专用融资利率 (None=用 financing_rate; 两融 ~5%)
        futures_margin_ratio: 期货保证金比例 (margin 模式, 默认0.15)

    Returns:
        DSleeveResult (nav, weights)
    """
    panel = panel.sort_index()
    dates = panel.index
    returns = panel.pct_change(fill_method=None)

    # 调仓掩码
    rb_mask = _rebalance_mask(dates, rebalance_freq)

    # D.2: 动量倾斜因子 (周频 alpha 层; momentum_beta=0 → 关闭)
    if momentum_beta != 0:
        from .momentum_tilt import momentum_tilt_factors
        tilt = momentum_tilt_factors(panel, lookback=momentum_lookback,
                                     beta=momentum_beta)
        tilt = tilt.reindex(dates).ffill()
    else:
        tilt = None

    # D.1: 负趋势因子日频 penalty (快层); mode="none"+include_rapid → 仅 rapid
    if neg_trend_mode != "none" or include_rapid:
        from .neg_trend_daily import compute_neg_trend_penalty, get_preset_params
        if neg_trend_preset is not None:
            preset = get_preset_params(neg_trend_preset)
            base_threshold = preset["base_threshold"]
            threshold_low = preset["threshold_low"]
        penalty = compute_neg_trend_penalty(
            panel, volume=volume, oi=oi, asset_classes=asset_classes,
            mode=neg_trend_mode, include_rapid=include_rapid,
            base_threshold=base_threshold, threshold_low=threshold_low,
        )
        penalty = penalty.reindex(dates).ffill().fillna(0.0)
    else:
        penalty = None

    # P1-B: 相关性覆盖层信号 (类内滚动两两相关; §5.37)
    if correlation_k > 0 and asset_classes:
        corr_signals = compute_correlation_signals(
            returns, asset_classes, window=correlation_window)
    else:
        corr_signals = None

    w_exec = pd.DataFrame(0.0, index=dates, columns=panel.columns)
    leverage_rb = 1.0

    for i, date in enumerate(dates):
        is_rb = bool(rb_mask[i]) and i >= window + 1
        if is_rb:
            rets_hist = returns.loc[:date]
            base = _d_base_weights(
                rets_hist, list(panel.columns), window=window,
                n_clusters=n_clusters, bond_cap=bond_cap,
                min_obs_per_asset=min_obs_per_asset,
                cov_estimator=cov_estimator,
                bond_cols=bond_cols,
            )

            # vol_target
            avail = [c for c in panel.columns
                     if rets_hist[c].notna().sum() >= min_obs_per_asset]
            if avail and max_leverage > 0:
                cov = np.cov(rets_hist[avail].iloc[-window:].dropna(
                    axis=0, how="any").to_numpy(), rowvar=False)
                base_avail = base[avail].values
                var_ex = float(base_avail @ cov @ base_avail)
                sigma_ex = np.sqrt(max(var_ex, 0.0)) * np.sqrt(ANN)
                if sigma_ex > 0:
                    leverage_rb = min(target_vol / sigma_ex, max_leverage)
                else:
                    leverage_rb = 1.0
            else:
                leverage_rb = 1.0

            w_exec.loc[date] = base * leverage_rb

        elif i > 0:
            # 非调仓日: 延用上次权重 (无杠杆微调)
            w_exec.iloc[i] = w_exec.iloc[i - 1]

        # D.2: 动量倾斜在调仓日应用 (周频 alpha; 非调仓日权重已携带)
        if tilt is not None and is_rb:
            from .momentum_tilt import apply_momentum_tilt
            w_exec.iloc[i] = apply_momentum_tilt(w_exec.iloc[i], tilt.loc[date])

        # D.1: 日频 penalty 叠加 (乘性减仓, w×(1−penalty))
        if penalty is not None:
            w_exec.iloc[i] = w_exec.iloc[i] * (1.0 - penalty.iloc[i])

        # P1-B: 日频相关性覆盖层 (类内相关 > θ → 线性降权; §5.37)
        if corr_signals is not None:
            w_exec.iloc[i] = apply_correlation_overlay(
                w_exec.iloc[i], corr_signals, asset_classes, date,
                threshold=correlation_threshold, k=correlation_k)

    # 共振门控 (§5.41): AR3 结构预警 → 事件式降杠杆 (仅结构破碎池适用)
    if resonance_k > 0:
        w_exec = apply_resonance_gate(
            w_exec, returns, cash_ret,
            window=resonance_ar3_window, enter=resonance_enter,
            exit_th=resonance_exit, vol_exit=resonance_vol_exit,
            dd_floor=resonance_dd_floor, k=resonance_k,
            financing_rate=financing_rate)

    # 组合收益 (含交易成本, 与 run_backtest 同口径: futures 1bp / ETF 5bp)
    rate = {c: (0.0001 if c.startswith("fut_") else 0.0005)
            for c in panel.columns}
    costs = (w_exec.diff().abs() * pd.Series(rate)).sum(axis=1).fillna(0.0)
    port_ret = (w_exec.shift(1) * returns).sum(axis=1) - costs
    # 资本会计层 (§5.38): 口径统一控制 现金腿 + 融资成本
    #  notional: 名义口径 — gross 视为占用资金 (idle=(1−gross)⁺, 超配付融资)
    #  margin:   保证金口径 — 期货仅占 nominal×margin_ratio (idle=(1−capital)⁺)
    cash_daily = None
    if cash_ret is not None:
        cash_daily = cash_ret.reindex(dates).ffill().fillna(0.0)
    gross_prev = w_exec.shift(1)
    if capital_base == "margin" and futures_margin_ratio > 0:
        fut_cols = [c for c in panel.columns if c.startswith("fut_")]
        etf_cols = [c for c in panel.columns if not c.startswith("fut_")]
        etf_cap = gross_prev[etf_cols].sum(axis=1)
        fut_cap = gross_prev[fut_cols].sum(axis=1) * futures_margin_ratio
        capital = etf_cap + fut_cap
        share_etf = (etf_cap / capital.replace(0.0, np.nan)).fillna(0.0)
    else:
        capital = gross_prev.sum(axis=1)
        share_etf = pd.Series(0.0, index=dates)
    # G5-lite 现金腿: 闲置资金赚现金收益 (§5.32)
    if cash_daily is not None:
        idle = (1.0 - capital).clip(lower=0.0)
        port_ret = port_ret + idle * cash_daily
    # 融资成本: 资金缺口付融资利率 (financing_rate=0 → 关闭, 向后兼容)
    if financing_rate > 0:
        excess = (capital - 1.0).clip(lower=0.0)
        r_etf = etf_financing_rate if etf_financing_rate is not None \
            else financing_rate
        fin_cost = (excess * share_etf * r_etf
                    + excess * (1.0 - share_etf) * financing_rate) / ANN
        port_ret = port_ret - fin_cost
    nav = (1 + port_ret).cumprod()
    nav.name = "D_sleeve"

    return DSleeveResult(nav=nav, weights=w_exec)
