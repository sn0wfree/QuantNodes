#!/usr/bin/env python3
"""cta_allocation_optimization.py — 4方案+HRP+RMT资金分配优化.

目标: 在不降低年化收益情况下, 提高Sharpe & Calmar.

方案:
  A. 主CTA杠杆放大 (主CTA 80%×1.25=100% + 子策略30%)
  B. 风险平价 (1/vol加权)
  C. 浓缩子策略 (75%主CTA+5%ZN+20%CU)
  D. 主CTA保险增强 (75%主CTA+子策略25%分散到6个)
  E. HRP层级风险平价
  F. RMT最大分散

杠杆成本: 主CTA 1.25倍杠杆部分按3%年化扣除.

输出:
  - reports/cta/allocation_optimization.csv (8种方案对比)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.optimize as opt
from scipy.spatial.distance import squareform

import sys
sys.path.insert(0, "scripts")
from cta_shared.factor_research import REPORT_DIR  # noqa: E402

warnings.filterwarnings("ignore")

LEVERAGE_COST = 0.03
NAV_FILES = {
    "主CTA": "final_nav_lump.csv",
    "ZN库存": "zn_inventory_nav.csv",
    "CU溢价TC": "cu_premium_nav.csv",
    "IH50动40×OI10": "ih50_oi_nav.csv",
    "T10y OI变化5": "t10y_oi_nav.csv",
    "rbi_steel": "chain_rbi_steel_nav.csv",
    "jjm_coke": "chain_jjm_coke_nav.csv",
    "ym_oilmeal": "chain_ym_oilmeal_nav.csv",
}


def load_all_returns() -> pd.DataFrame:
    """加载所有策略日收益."""
    rets = {}
    common = None
    for name, fn in NAV_FILES.items():
        nav = pd.read_csv(REPORT_DIR / fn, index_col=0, parse_dates=True).iloc[:, 0]
        r = nav.pct_change(fill_method=None).fillna(0)
        rets[name] = r
        common = r.index if common is None else common.intersection(r.index)
    return pd.DataFrame({k: v.reindex(common) for k, v in rets.items()})


def stats(r: pd.Series, is_end: str = "2022-01-01") -> dict:
    """Sharpe/AnnRet/MaxDD/Calmar + IS/OOS."""
    r = r.dropna()
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    is_ = r[r.index < is_end]
    oos_ = r[r.index >= is_end]

    def _one(s):
        if len(s) < 50:
            return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0}
        nav_r = (1 + s).cumprod()
        a = nav_r.iloc[-1] ** (252 / len(s)) - 1
        v = s.std() * np.sqrt(252)
        d = (nav_r / nav_r.cummax() - 1).min()
        return {"sharpe": a / v if v > 0 else 0, "ann_ret": a, "max_dd": d}

    return {
        "sharpe": ann / vol if vol > 0 else 0,
        "ann_ret": ann, "vol": vol, "max_dd": dd,
        "calmar": ann / abs(dd) if dd != 0 else 0,
        "IS_Sharpe": _one(is_)["sharpe"],
        "OOS_Sharpe": _one(oos_)["sharpe"],
        "OOS_AnnRet": _one(oos_)["ann_ret"],
    }


def apply_leverage(rets: pd.DataFrame, col: str, leverage: float,
                    cost_annual: float = LEVERAGE_COST) -> pd.DataFrame:
    """对指定列施加杠杆, 扣除资金成本.

    net_ret = leverage * raw_ret - cost_annual/252 (杠杆成本每日扣)
    """
    rets = rets.copy()
    cost_daily = cost_annual / 252
    rets[col] = leverage * rets[col] - cost_daily
    return rets


def portfolio(rets: pd.DataFrame, weights: dict[str, float], label: str) -> dict:
    """计算加权组合的统计量."""
    w = pd.Series(weights).reindex(rets.columns).fillna(0)
    combo = (rets * w).sum(axis=1).fillna(0)
    st = stats(combo)
    st["组合"] = label
    st["主CTA权重"] = weights.get("主CTA", 0)
    st["子策略占比"] = sum(v for k, v in weights.items() if k != "主CTA")
    st["总敞口"] = sum(weights.values())
    return st


def exp_a_leverage(rets: pd.DataFrame) -> dict:
    """方案A: 主CTA 80% × 1.25 = 100%净敞口 + 子策略30%.

    总敞口130%, 资金110%, 杠杆成本扣除3%年化.
    """
    rets_lev = apply_leverage(rets, "主CTA", leverage=1.25)
    weights = {
        "主CTA": 0.80,
        "ZN库存": 0.05,
        "CU溢价TC": 0.10,
        "IH50动40×OI10": 0.05,
        "T10y OI变化5": 0.05,
        "rbi_steel": 0.02,
        "jjm_coke": 0.0,
        "ym_oilmeal": 0.03,
    }
    return portfolio(rets_lev, weights, "A: 主CTA杠杆1.25")


def exp_b_risk_parity(rets: pd.DataFrame) -> dict:
    """方案B: 风险平价 (1/vol加权).

    用最近252天的年化vol倒数加权.
    """
    vol = rets.rolling(252, min_periods=120).std().iloc[-1] * np.sqrt(252)
    inv_vol = 1 / vol
    weights = (inv_vol / inv_vol.sum()).to_dict()
    return portfolio(rets, weights, "B: 风险平价(1/vol)")


def exp_c_concentrated(rets: pd.DataFrame) -> dict:
    """方案C: 浓缩子策略 — 只用CU+ZN, 主CTA 75%."""
    weights = {
        "主CTA": 0.75,
        "ZN库存": 0.05,
        "CU溢价TC": 0.20,
    }
    return portfolio(rets, weights, "C: 浓缩(CU+ZN)")


def exp_d_insurance(rets: pd.DataFrame) -> dict:
    """方案D: 主CTA保险增强 — 主CTA 75%, 子策略分散到6个但各2-5%."""
    weights = {
        "主CTA": 0.75,
        "ZN库存": 0.05,
        "CU溢价TC": 0.05,
        "IH50动40×OI10": 0.05,
        "T10y OI变化5": 0.05,
        "rbi_steel": 0.02,
        "jjm_coke": 0.02,
        "ym_oilmeal": 0.01,
    }
    return portfolio(rets, weights, "D: 主CTA保险")


def _hrp_subcluster_weights(items: list, cov: pd.DataFrame, dist: pd.DataFrame) -> pd.Series:
    """递归二分HRP权重计算."""
    if len(items) == 1:
        return pd.Series(1.0, index=items)
    if len(items) == 2:
        inv_var = 1 / np.diag(cov.loc[items, items])
        w = inv_var / inv_var.sum()
        return pd.Series(w, index=items)
    sub_dist = dist.loc[items, items]
    sub_condensed = squareform(sub_dist.values, checks=False)
    if np.any(np.isnan(sub_condensed)):
        finite = sub_condensed[np.isfinite(sub_condensed)]
        if len(finite) == 0:
            return pd.Series(1.0 / len(items), index=items)
        sub_condensed = np.nan_to_num(sub_condensed, nan=finite.max())
    sub_link = sch.linkage(sub_condensed, method="single")
    cut = sch.fcluster(sub_link, t=2, criterion="maxclust")
    cluster_a = [items[i] for i in range(len(items)) if cut[i] == 1]
    cluster_b = [items[i] for i in range(len(items)) if cut[i] == 2]
    if not cluster_a or not cluster_b:
        return pd.Series(1.0 / len(items), index=items)
    w_a = _hrp_subcluster_weights(cluster_a, cov, dist)
    w_b = _hrp_subcluster_weights(cluster_b, cov, dist)
    var_a = w_a.values @ cov.loc[w_a.index, w_a.index].values @ w_a.values
    var_b = w_b.values @ cov.loc[w_b.index, w_b.index].values @ w_b.values
    if var_a <= 0 or var_b <= 0:
        return pd.Series(1.0 / len(items), index=items)
    alpha = (1 / var_a) / (1 / var_a + 1 / var_b)
    return pd.concat([w_a * alpha, w_b * (1 - alpha)])


def exp_e_hrp(rets: pd.DataFrame) -> dict:
    """方案E: HRP层级风险平价.

    步骤:
      1. 计算相关性矩阵→距离矩阵 (sqrt(2*(1-corr)))
      2. 单链接层次聚类
      3. 递归二分法分配权重 (inverse-variance对两簇)
    """
    corr = rets.corr()
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr.values, 1.0)
    dist = pd.DataFrame(
        np.sqrt(np.maximum(2 * (1 - corr.values), 0)),
        index=corr.index, columns=corr.columns,
    )
    np.fill_diagonal(dist.values, 0)

    cov = rets.cov() * 252
    w = _hrp_subcluster_weights(list(rets.columns), cov, dist)
    w = w / w.sum()
    weights = w.to_dict()
    return portfolio(rets, weights, "E: HRP层级风险平价")


def exp_f_max_div(rets: pd.DataFrame) -> dict:
    """方案F: RMT最大分散 — 最大化分散比率.

    max (w'σ)² / (w'Σw)
    s.t. Σw = 1, w >= 0
    """
    cov = rets.cov().values * 252
    sigma = np.sqrt(np.diag(cov))

    def neg_div(w):
        w = np.abs(w)
        s = w.sum()
        if s <= 0:
            return 1e10
        w = w / s
        weighted_vol = w @ sigma
        port_var = w @ cov @ w
        if port_var <= 0:
            return 1e10
        return -(weighted_vol ** 2 / port_var)

    n = len(rets.columns)
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = opt.minimize(neg_div, w0, bounds=bounds, constraints=cons,
                        method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
    w = np.abs(res.x)
    w = w / w.sum()
    weights = dict(zip(rets.columns, w))
    return portfolio(rets, weights, "F: RMT最大分散")


def baseline_current(rets: pd.DataFrame) -> dict:
    """基准: 当前最优跨商品组合 (45/5/15/10/10/5/5/5)."""
    weights = {
        "主CTA": 0.45,
        "ZN库存": 0.05,
        "CU溢价TC": 0.15,
        "IH50动40×OI10": 0.10,
        "T10y OI变化5": 0.10,
        "rbi_steel": 0.05,
        "jjm_coke": 0.05,
        "ym_oilmeal": 0.05,
    }
    return portfolio(rets, weights, "当前最优 45/5/15/10/10/5/5/5")


def baseline_main_only(rets: pd.DataFrame) -> dict:
    """基准: 主CTA单独."""
    return portfolio(rets, {"主CTA": 1.0}, "基准: 主CTA单独")


def main():
    rets = load_all_returns()
    print(f"数据范围: {rets.index.min().date()} ~ {rets.index.max().date()} ({len(rets)}天)")
    print("\n各策略年化vol:")
    for col in rets.columns:
        print(f"  {col}: {rets[col].std() * np.sqrt(252):.2%}")

    results = [
        baseline_main_only(rets),
        baseline_current(rets),
        exp_a_leverage(rets),
        exp_b_risk_parity(rets),
        exp_c_concentrated(rets),
        exp_d_insurance(rets),
        exp_e_hrp(rets),
        exp_f_max_div(rets),
    ]

    df = pd.DataFrame(results)
    cols = ["组合", "sharpe", "ann_ret", "max_dd", "calmar",
            "IS_Sharpe", "OOS_Sharpe", "OOS_AnnRet",
            "主CTA权重", "子策略占比", "总敞口"]
    df = df[cols]
    df.to_csv(REPORT_DIR / "allocation_optimization.csv", index=False)

    print("\n=== 资金分配优化综合对比 ===")
    print(f"{'组合':<32} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>8} {'Calmar':>7} | "
          f"{'IS':>6} {'OOS':>6} {'OOS-AR':>8} {'主CTA':>7} {'子':>5} {'敞口':>5}")
    print("-" * 130)
    for _, r in df.iterrows():
        print(f"{r['组合']:<32} {r['sharpe']:>7.3f} {r['ann_ret']:>7.2%} "
              f"{r['max_dd']:>8.2%} {r['calmar']:>7.2f} | "
              f"{r['IS_Sharpe']:>6.3f} {r['OOS_Sharpe']:>6.3f} {r['OOS_AnnRet']:>7.2%} "
              f"{r['主CTA权重']:>7.2%} {r['子策略占比']:>5.2%} {r['总敞口']:>5.2f}")

    print("\n=== 决策1: Calmar>基准 + AnnRet>13% ===")
    qual = df[(df["calmar"] > 4.5) & (df["ann_ret"] >= 0.13)]
    if len(qual) > 0:
        cols_show = ["组合", "sharpe", "ann_ret", "max_dd", "calmar", "OOS_Sharpe"]
        print(qual[cols_show].to_string(index=False))
    else:
        print("无方案 — 改用 Calmar>3.0 & AnnRet>13%")
        qual2 = df[(df["calmar"] >= 3.0) & (df["ann_ret"] >= 0.13)]
        if len(qual2) > 0:
            cols_show = ["组合", "sharpe", "ann_ret", "max_dd", "calmar", "OOS_Sharpe"]
            print(qual2[cols_show].to_string(index=False))

    print("\n=== 决策2: Sharpe最高 ===")
    best_sharpe = df.loc[df["sharpe"].idxmax()]
    print(f"  {best_sharpe['组合']}: Sharpe={best_sharpe['sharpe']:.3f} "
          f"AnnRet={best_sharpe['ann_ret']:.2%}")

    print("\n=== 决策3: Calmar最高 ===")
    best_calmar = df.loc[df["calmar"].idxmax()]
    print(f"  {best_calmar['组合']}: Calmar={best_calmar['calmar']:.2f} "
          f"AnnRet={best_calmar['ann_ret']:.2%}")


if __name__ == "__main__":
    main()
