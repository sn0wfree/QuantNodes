"""类别级风险预算 (V7, PLAN §16): 类间 ERC + 债券风险贡献封顶.

背景 (报告 §5.17/§5.18): A sleeve HRP+MP 在 38 资产池退化为 3.5x 杠杆国债组合.
风险平价在权重空间 w ∝ 1/σ 天然向低波资产倾斜; 资产级 cap 封不住类别级集中
(3 个国债期货同属一个利率因子). 类别级风险预算是文献验证的 L3 层级
(2022 后行业共识, AQR 路径).

算法:
1. 类内 inverse-vol 合成类收益序列 → 类级协方差
2. 类间 ERC 等风险贡献 (不动点迭代)
3. 债券类 RC > bond_risk_cap → bond 预算固定为 cap, 其余类重新 ERC
   (cap 是约束而非定额, 未触发时完全 ERC)
4. 类权重按成员 inverse-vol 摊到资产
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cov_rmt import MIN_OBS
from .hrp import MIN_ASSETS_FOR_HRP

ASSET_CLASSES: dict[str, list[str]] = {
    "bond_rate": ["fut_t", "fut_tf", "fut_ts"],
    "equity_fut": ["fut_if", "fut_ic", "fut_ih"],
    "equity_etf_dom": ["etf_159915", "etf_510300",        # 创业板/沪深300 (panel_v3 新增)
                       "etf_512890", "etf_512010", "etf_515030", "etf_512690",
                       "etf_512800", "etf_515050", "etf_512200"],
    "equity_etf_off": ["etf_513100", "etf_513500", "etf_159740"],  # 纳指/标普/恒科
    "precious": ["fut_au", "fut_ag"],
    "nonferrous": ["fut_cu", "fut_al", "fut_zn", "fut_ni"],
    "ferrous": ["fut_rb", "fut_hc", "fut_i", "fut_j", "fut_jm"],
    "energy_chem": ["fut_ta", "fut_ma", "fut_pp", "fut_l", "fut_bu", "fut_sc"],
    "agri_oilseed": ["fut_m", "fut_y", "fut_p"],
    "agri_soft": ["fut_cf", "fut_sr", "fut_c"],
}

BOND_CLASS = "bond_rate"


def _invvol_weights(sub: pd.DataFrame) -> pd.Series:
    """窗口内 inverse-vol 权重 (归一); 全 NaN 列被剔除."""
    vols = sub.std()
    vols = vols.replace(0, np.nan).dropna()
    if vols.empty:
        return pd.Series(dtype=float)
    inv = 1.0 / vols
    return inv / inv.sum()


def class_composite_returns(returns: pd.DataFrame,
                            classes: dict[str, list[str]],
                            window: int = 252,
                            min_obs_per_asset: int = MIN_OBS) -> pd.DataFrame:
    """类内 inverse-vol 合成类收益序列 (T x K).

    每个调仓时点重算成本高, 此处按全历史一次性合成: 类内权重用截至每行的
    滚动窗口 inverse-vol (因果: 只用 t 及以前数据).
    """
    cols = list(returns.columns)
    out = pd.DataFrame(index=returns.index)
    for cls, members in classes.items():
        avail = [c for c in members if c in cols]
        if not avail:
            continue
        sub = returns[avail]
        # 逐行滚动 inverse-vol (窗口内非空观测 ≥ min_obs_per_asset 才计入)
        counts = sub.notna().rolling(window, min_periods=1).sum()
        w_all = {}
        for c in avail:
            v = sub[c].rolling(window, min_periods=min_obs_per_asset).std()
            w_all[c] = 1.0 / v
        w_df = pd.DataFrame(w_all)
        w_df = w_df.replace([np.inf, -np.inf], np.nan)
        w_sum = w_df.sum(axis=1, skipna=True)
        w_norm = w_df.div(w_sum, axis=0).fillna(0.0)
        # 观测不足的资产行剔除 (counts 以任一成员最低非空数计)
        ok = (counts.max(axis=1) >= min_obs_per_asset) & (w_sum > 0)
        comp = (sub.fillna(0.0) * w_norm).sum(axis=1)
        out[cls] = comp.where(ok)
    return out


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """各资产风险贡献占比 (sum=1)."""
    w = np.asarray(w, dtype=float)
    sigma2 = float(w @ cov @ w)
    if sigma2 <= 0:
        return np.full(len(w), np.nan)
    mrc = cov @ w
    rc = w * mrc / sigma2
    return rc


def _erc_weights(cov: np.ndarray, budgets: np.ndarray | None = None,
                 max_iter: int = 1000, tol: float = 1e-10) -> np.ndarray:
    """ERC / 风险预算权重 (CCD 坐标下降, Griveau-Billion et al. 2013).

    budgets: 各类目标风险贡献占比 (sum=1); None = 等预算.
    解条件: x_i (Σx)_i = b_i, 逐坐标二次方程精确解, 可证收敛.
    """
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, float)
    b = b / b.sum()
    diag = np.diag(cov).copy()
    diag = np.where(diag > 0, diag, 1e-12)
    vol = np.sqrt(diag)
    x = 1.0 / vol
    x = x / x.sum()
    for _ in range(max_iter):
        x_prev = x.copy()
        for i in range(n):
            s = float(cov[i] @ x) - diag[i] * x[i]
            disc = max(s * s + 4.0 * diag[i] * b[i], 0.0)
            x[i] = (-s + np.sqrt(disc)) / (2.0 * diag[i])
        if np.max(np.abs(x - x_prev)) < tol * max(1.0, float(np.max(np.abs(x)))):
            break
    return x / x.sum()


def class_erc_weights(returns: pd.DataFrame,
                      classes: dict[str, list[str]] | None = None,
                      window: int = 252,
                      bond_risk_cap: float = 1.0,
                      min_obs_per_asset: int = MIN_OBS,
                      bond_class: str = BOND_CLASS) -> pd.Series:
    """类别级 ERC + 债券风险贡献封顶 → 资产权重 (与 returns.columns 对齐).

    Args:
        returns: (T x N) 收益率面板, NaN = 未上市/无数据
        classes: 类别映射; None 用 ASSET_CLASSES
        window: 类级协方差估计窗口
        bond_risk_cap: 债券类风险贡献占比上限 (1.0 = 不约束)
        min_obs_per_asset: 资产纳入窗口所需最少非空观测
        bond_class: 债券类名

    Returns:
        pd.Series: 与 returns.columns 同索引; 数据不足资产权重 = 0
    """
    classes = classes or ASSET_CLASSES
    cols = list(returns.columns)
    out = pd.Series(0.0, index=cols)

    comp = class_composite_returns(returns, classes, window, min_obs_per_asset)
    comp_last = comp.iloc[-window:]
    # 类可用性: 窗口内类合成序列非空观测 ≥ MIN_OBS
    cls_avail = [c for c in comp.columns
                 if comp_last[c].notna().sum() >= MIN_OBS]

    # 窗口内可交易资产 (与 HRP 渐入池口径一致)
    w_last = returns.iloc[-window:]
    asset_avail = [c for c in cols if w_last[c].notna().sum() >= min_obs_per_asset]

    if len(cls_avail) < 2 or len(asset_avail) < MIN_ASSETS_FOR_HRP:
        if asset_avail:
            out[asset_avail] = 1.0 / len(asset_avail)
            return out
        return pd.Series(1.0 / len(cols), index=cols)

    sub = comp_last[cls_avail].dropna(axis=0, how="any")
    if len(sub) < MIN_OBS:
        out[asset_avail] = 1.0 / len(asset_avail)
        return out

    cov = np.cov(sub.to_numpy(), rowvar=False)
    k = cov.shape[0]
    cov = cov + np.eye(k) * 1e-12  # 数值正定

    w_cls = _erc_weights(cov)
    names = list(cls_avail)

    # 债券封顶: RC_bond > cap → bond 预算 = cap, 其余 (1-cap) 等预算重解
    if bond_class in names and bond_risk_cap < 1.0:
        rc = risk_contributions(w_cls, cov)
        rc_bond = rc[names.index(bond_class)]
        if np.isfinite(rc_bond) and rc_bond > bond_risk_cap:
            budgets = np.full(k, (1.0 - bond_risk_cap) / (k - 1))
            budgets[names.index(bond_class)] = bond_risk_cap
            w_cls = _erc_weights(cov, budgets)

    w_cls = pd.Series(w_cls, index=names)

    # 类内 inverse-vol 摊分 (只用窗口内可交易资产)
    sub_assets = returns[asset_avail].iloc[-window:]
    for cls in names:
        members = [c for c in classes.get(cls, [])
                   if c in asset_avail and sub_assets[c].notna().sum() >= min_obs_per_asset]
        if not members:
            continue
        member_w = _invvol_weights(sub_assets[members].dropna(axis=0, how="any"))
        if member_w.empty:
            member_w = pd.Series(1.0 / len(members), index=members)
        out[member_w.index] += w_cls[cls] * member_w

    # 未被任何可用类覆盖的可交易资产 (类别映射遗漏): 并入等权残余
    covered = out > 0
    residual = [c for c in asset_avail if not covered[c]]
    if residual:
        w_rest = w_cls.sum() - out[covered].sum()
        out[residual] = max(w_rest, 0.0) / len(residual)

    total = out.sum()
    if total > 0:
        out = out / total
    return out
