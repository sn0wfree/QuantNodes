# coding: utf-8
"""Spanning Test vs 全部子策略 — P5 实装.

论文 §6 关键结论:
  - Zookeeper 回归 → 单因子 (basis/mom/basis-mom/hp) 截距不显著
  - 反向: 单因子回归 Zookeeper → 年化 α 4.8%~6.0%

本项目对手 (用户决策 vs 全部):
  - §25 carry
  - §26 basis momentum / §26 vrp
  - ZN inventory / jjm_coke / CU premium TC / AU layered
  - OscAlpha (§19 已拒但保留对照)
  - 主 CTA baseline (E2-R + AU10%)

模型: target_alpha = α + β_zk · Zookeeper_factor + ε
  α 即 Zookeeper 无法解释的独立 alpha.
  Spanning Test 决策:
    |α|/| < 2%/年: 归档
    2-5%: 子策略候选 (组合 ≤10%)
    > 5%: 谨慎审查
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (OUT_DIR, SPANNING_ALPHA_STRONG)

LOG = logging.getLogger(__name__)

REPORTS = Path("reports/cta")

# 子策略 NAV 路径 (open = 当期仓位, 沿 §22-§24 纪律)
TARGETS = {
    "§25 carry": REPORTS / "carry_navs.parquet",  # 收益序列 → 累计为 NAV
    "§26 basis momentum": None,  # 已证伪, 仅在 cross_sectional.csv 摘要中
    "§26 vrp": None,  # 同上
    "ZN inventory": REPORTS / "rebuild_subnav_ZN库存_open.csv",
    "jjm_coke": REPORTS / "rebuild_subnav_jjm_coke_open.csv",
    "CU premium TC": REPORTS / "rebuild_subnav_CU溢价TC_open.csv",
    "AU layered": REPORTS / "gold_layered_layered.parquet",
    "OscAlpha": REPORTS / "osc6_nav.parquet",
    "Main CTA baseline (E2-R+AU10%)": REPORTS / "main_cta_curve.csv",
}


def _to_monthly_ret(nav_or_ret: pd.Series, idx: pd.DatetimeIndex | None = None) -> pd.Series:
    """统一转换为月度收益 Series.

    输入可能是:
      - 日频 NAV → 月末 NAV → pct_change
      - 日频收益序列 → 月度 sum
      - 已月频收益 → 原样
    """
    if idx is not None:
        nav_or_ret.index = idx
    if not isinstance(nav_or_ret.index, pd.DatetimeIndex):
        nav_or_ret.index = pd.DatetimeIndex(nav_or_ret.index)
    # 月度化
    if nav_or_ret.index.to_period("M").nunique() > 0:
        if "nav" in str(nav_or_ret.name).lower() or nav_or_ret.iloc[0] == 1.0:
            # NAV → 月度收益
            me = nav_or_ret.groupby(nav_or_ret.index.to_period("M").to_timestamp("M")).last()
            return me.pct_change().dropna()
        # 收益 → 月度 sum
        ret_m = nav_or_ret.resample("M").sum()
        return ret_m.dropna()
    return pd.Series(dtype=float)


def _load_target_nav(name: str, path: Path | None) -> pd.Series | None:
    """加载目标子策略 NAV (日频)."""
    if path is None:
        return None
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "nav" in df.columns:
        return df.set_index(df.columns[0])["nav"] if df.shape[1] >= 2 else df["nav"]
    if "ret" in df.columns:
        return (1 + df["ret"]).cumprod()  # 收益转 NAV
    return None


def _load_main_cta(path: Path, version: str = "nav_E2-R+AU10%") -> pd.Series | None:
    """加载主 CTA 特定版本 NAV."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if version not in df.columns:
        return None
    return df.set_index("Unnamed: 0")[version]


def load_zk_factor(method: str = "weighted") -> pd.Series | None:
    """加载 Zookeeper 因子月度收益.

    method='weighted' (P6-C): 用 factor_nav_weighted.csv (加权 LS, 推荐)
    method='quantile' (P4):   用 factor_nav.csv (Q4/Q1 等权排序, 旧)
    """
    fname = "factor_nav_weighted.csv" if method == "weighted" else "factor_nav.csv"
    f = OUT_DIR / fname
    if not f.exists():
        return None
    df = pd.read_csv(f, index_col=0, parse_dates=[0])
    nav = df["nav"] if "nav" in df.columns else df.iloc[:, 0]
    return _to_monthly_ret(nav)


def spanning_regression(target_monthly_ret: pd.Series,
                         zk_factor_monthly_ret: pd.Series) -> dict:
    """单标的 Spanning Test.

    target = α + β · Zookeeper + ε

    Returns:
        dict: alpha_annual / t_alpha / beta / r_squared / decision / n_months
    """
    common = target_monthly_ret.index.intersection(zk_factor_monthly_ret.index)
    if len(common) < 12:
        LOG.warning(f"Spanning 数据不足: {len(common)} 月")
        return {"alpha_annual": np.nan, "t_alpha": np.nan, "beta": np.nan,
                "r_squared": np.nan, "decision": "数据不足", "n_months": len(common)}
    y = target_monthly_ret.loc[common].values
    X = zk_factor_monthly_ret.loc[common].values.reshape(-1, 1)
    X_const = np.hstack([np.ones((len(X), 1)), X])
    # OLS
    from numpy.linalg import lstsq
    coef, _, _, _ = lstsq(X_const, y, rcond=None)
    alpha = coef[0]
    beta = coef[1]
    resid = y - X_const @ coef
    sigma2 = np.sum(resid**2) / (len(y) - 2)
    XtX_inv = np.linalg.inv(X_const.T @ X_const)
    se_alpha = np.sqrt(sigma2 * XtX_inv[0, 0])
    t_alpha = alpha / se_alpha if se_alpha > 0 else 0.0
    # R²
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # 年化
    alpha_annual = alpha * 12
    # 决策
    a = abs(alpha_annual)
    if a < 0.02:
        decision = "归档 (<2%)"
    elif a < SPANNING_ALPHA_STRONG:
        decision = f"候选集成 ({a:.1%})"
    else:
        decision = f"谨慎审查 ({a:.1%})"
    return {
        "alpha_monthly": float(alpha),
        "alpha_annual": float(alpha_annual),
        "t_alpha": float(t_alpha),
        "beta": float(beta),
        "r_squared": float(r2),
        "decision": decision,
        "n_months": int(len(common)),
    }


def run_all_spanning(method: str = "weighted") -> pd.DataFrame:
    """对全部对手跑 Spanning Test → spanning_alpha.csv."""
    zk = load_zk_factor(method=method)
    if zk is None:
        LOG.error(f"Zookeeper 因子 NAV 不存在 (method={method}), 请先跑 P4/P6-C")
        return pd.DataFrame()
    LOG.info(f"Spanning Test 方法={method}, ZK 月度收益: {len(zk)} 月, "
             f"均值={zk.mean():.4f}, std={zk.std():.4f}, "
             f"Sharpe={zk.mean() / zk.std() * np.sqrt(12):.2f}")

    rows = []
    for name, path in TARGETS.items():
        LOG.info(f"Spanning Test: {name}")
        if name == "Main CTA baseline (E2-R+AU10%)":
            target_nav = _load_main_cta(path)
        else:
            target_nav = _load_target_nav(name, path)
        if target_nav is None:
            LOG.warning(f"  {name}: 跳过 (无数据或已证伪)")
            rows.append({"target": name, "alpha_annual": np.nan,
                         "t_alpha": np.nan, "beta": np.nan, "r_squared": np.nan,
                         "decision": "无数据", "n_months": 0})
            continue
        target_ret = _to_monthly_ret(target_nav)
        if target_ret is None or target_ret.empty:
            LOG.warning(f"  {name}: 月度收益为空")
            continue
        result = spanning_regression(target_ret, zk)
        result["target"] = name
        rows.append(result)
        LOG.info(f"  {name}: α_ann={result['alpha_annual']:.2%}, "
                 f"t={result['t_alpha']:.2f}, β={result['beta']:.3f}, "
                 f"R²={result['r_squared']:.2%}, {result['decision']}")
    df = pd.DataFrame(rows)
    return df
