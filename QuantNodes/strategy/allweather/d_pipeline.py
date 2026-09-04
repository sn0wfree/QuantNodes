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
                        window: int, min_obs: int = 60) -> pd.Series | None:
    """债券类 inv-vol 合成序列 (合成资产用于盒约束); 缺列防御."""
    avail = [c for c in BOND_COLS
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
) -> pd.Series:
    """单次调仓: 协方差 → HRP 预算 → 合成债券 → 盒约束 CCD → 摊回.

    cov_estimator: "lw" / "oas" / "lw_ewma" / "block_corr" / "avs"
    """
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
    synth_ret = _synth_bond_returns(returns_hist, window, min_obs_per_asset)
    bond_in_avail = [c for c in BOND_COLS if c in avail]

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
    non_bond = [c for c in cov_cols if c not in BOND_COLS]
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
        if c in BOND_COLS:
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
    compute_metrics_fn=None,
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

    # 组合收益
    port_ret = (w_exec.shift(1) * returns).sum(axis=1)
    nav = (1 + port_ret).cumprod()
    nav.name = "D_sleeve"

    return DSleeveResult(nav=nav, weights=w_exec)
