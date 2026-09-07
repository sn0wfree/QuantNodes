# coding: utf-8
"""Zookeeper 因子作为状态变量接入主 CTA vol_target — P7-B 模拟式集成.

⚠️ 不修改 v11 主 CTA 引擎 (factor_modules/base.py/combo.py), 仅在历史主 CTA NAV 上
模拟 vol_target 调节效果. 输出三组曲线:
  A. 原主 CTA NAV (nav_E2-R+AU10%)
  B. ZK-vol 上调版 (vol_target_adjusted = base × (1 + α × (zk_strength - 0.5)))
  C. ZK-vol 下调版 (反方向)

评估: Sharpe / DD / α vs 主 CTA / 换手 / 极值

ZK 状态变量定义 (三选一, 由配置切换):
  - signal_strength: 月度 |r̂| 横截面均值 (信号越强 → vol 上调)
  - signal_dispersion: 月度 r̂ 横截面 std (分散度越高 → vol 上调)
  - signal_momentum: ZK 因子过去 3 月累计收益 (ZK 收益趋势 → vol 上调)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import OUT_DIR

LOG = logging.getLogger(__name__)

REPORTS = Path("reports/cta")
ALPHA_CANDID = [0.1, 0.2, 0.3, 0.5, 0.7]  # 状态变量对 vol 的调节强度


def _to_daily_returns(nav: pd.Series) -> pd.Series:
    """日频 NAV → 日频简单收益."""
    return nav.pct_change().fillna(0)


def _to_monthly_returns(daily_ret: pd.Series) -> pd.Series:
    return daily_ret.resample("M").sum()


def _load_main_cta_nav() -> pd.Series | None:
    """加载主 CTA baseline (E2-R+AU10%) 日频 NAV."""
    f = REPORTS / "main_cta_curve.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, index_col="Unnamed: 0", parse_dates=True)
    col = "nav_E2-R+AU10%"
    if col not in df.columns:
        # fallback: 第一个 nav 列
        for c in df.columns:
            if c.startswith("nav_"):
                col = c
                break
    return df[col].dropna()


def _load_zk_state_variable(method: str = "weighted",
                            stat: str = "signal_strength") -> pd.Series | None:
    """加载 ZK 因子月度状态变量.

    method: weighted (P6-C) / quantile (P4)
    stat: signal_strength / signal_dispersion / signal_momentum
    """
    fname = "factor_nav_weighted.csv" if method == "weighted" else "factor_nav.csv"
    nav_f = OUT_DIR / fname
    if not nav_f.exists():
        return None
    nav = pd.read_csv(nav_f, index_col=0, parse_dates=[0]).iloc[:, 0]
    monthly_ret = nav.resample("M").last().pct_change().dropna()
    cs_pred_f = OUT_DIR / "cs_predictions.parquet"
    if stat == "signal_momentum":
        return monthly_ret
    if not cs_pred_f.exists():
        return monthly_ret
    cs = pd.read_parquet(cs_pred_f)
    if stat == "signal_strength":
        # 月度 |r̂| 横截面均值 (信号强度)
        v = cs.abs().mean(axis=1)
    elif stat == "signal_dispersion":
        # 月度 r̂ 横截面 std (分散度)
        v = cs.std(axis=1)
    else:
        v = cs.abs().mean(axis=1)
    v.index = pd.DatetimeIndex(v.index)
    # 月度化 (月末取最后一日)
    return v.groupby(v.index.to_period("M").to_timestamp("M")).last()


def _normalize_state(s: pd.Series) -> pd.Series:
    """状态变量标准化到 [0, 1] (滚动中位数 + ±3σ 截尾)."""
    median = s.expanding().median()
    std = s.expanding().std()
    z = (s - median) / std.replace(0, np.nan)
    z = z.clip(-3, 3)
    # [-3, 3] → [0, 1]
    return (z / 6 + 0.5).fillna(0.5)


def simulate_vol_target_adjustment(
        main_cta_nav: pd.Series,
        zk_state: pd.Series,
        alpha: float = 0.3,
        direction: int = 1,
        base_vol: float = 1.0) -> pd.Series:
    """基于 ZK 状态变量模拟 vol_target 调整的主 CTA NAV.

    Args:
        main_cta_nav: 主 CTA 日频 NAV.
        zk_state: ZK 月度状态变量 (标准化到 [0,1]).
        alpha: vol 调节强度 (0 = 不调, 1 = 全调).
        direction: 1 = 正向 (强信号 → vol 上调),
- / -1 = 反向.
        base_vol: 基准 vol.

    Returns:
        调整后日频 NAV.
    """
    daily_ret = _to_daily_returns(main_cta_nav)
    monthly_ret = _to_monthly_returns(daily_ret)

    # 状态变量月度 (已在 _load_zk_state_variable 月度化)
    common = monthly_ret.index.intersection(zk_state.index)
    if common.empty:
        return main_cta_nav

    # 调整比例: vol_target_adjusted / vol_target_base
    # 中位信号 0.5 → 比例 = 1 (不变)
    # 强信号 1.0 → 比例 = 1 + alpha × (1 - 0.5) × direction = 1 + 0.5×alpha
    adjust_ratio = 1 + direction * alpha * (zk_state.loc[common] - 0.5) * 2
    adjust_ratio = adjust_ratio.clip(lower=0.3, upper=2.0)  # 防止过度调节

    # 按月度比例缩放月收益 (模拟 vol 调整)
    adj_monthly_ret = monthly_ret.loc[common] * adjust_ratio

    # 月度缩放后的日频收益: 简单按日历日均匀分布
    adj_nav = main_cta_nav.copy(deep=True)
    # 简化: 按月分段重新构造 NAV (近似)
    monthly_index = monthly_ret.index
    for m in common:
        if m not in monthly_index:
            continue
        # 找当月日频范围
        next_m = m + pd.DateOffset(months=1)
        month_mask = (main_cta_nav.index >= m) & (main_cta_nav.index < next_m)
        if month_mask.sum() == 0:
            continue
        # 原始月收益
        month_orig_ret = monthly_ret.loc[m]
        if month_orig_ret == 0:
            continue
        # 调整后月收益
        month_adj = adj_monthly_ret.loc[m]
        ratio = month_adj / month_orig_ret if month_orig_ret != 0 else 1.0
        # 在该月内对日频收益乘 ratio (NAV 也乘)
        for d in main_cta_nav.index[month_mask]:
            # 重新构造: prev * (1 + ret * ratio)
            idx = main_cta_nav.index.get_loc(d)
            if idx == 0:
                continue
            prev = adj_nav.iloc[idx - 1]
            orig_ret = daily_ret.iloc[idx]
            adj_nav.iloc[idx] = prev * (1 + orig_ret * ratio)
    return adj_nav


def evaluate_pair(nav_a: pd.Series, nav_b: pd.Series) -> dict:
    """双 NAV 对比评估."""
    ret_a = nav_a.pct_change().dropna()
    ret_b = nav_b.pct_change().dropna()
    common = ret_a.index.intersection(ret_b.index)
    if len(common) < 12:
        return {"sharpe_a": np.nan, "sharpe_b": np.nan}
    ra, rb = ret_a.loc[common], ret_b.loc[common]
    sharpe_a = ra.mean() / ra.std() * np.sqrt(252)
    sharpe_b = rb.mean() / rb.std() * np.sqrt(252)
    # DD
    dd_a = (nav_a / nav_a.cummax() - 1).min()
    dd_b = (nav_b / nav_b.cummax() - 1).min()
    # 年化
    ann_a = ra.mean() * 252
    ann_b = rb.mean() * 252
    # β (回归 nb on na)
    cov = np.cov(ra, rb)
    beta = cov[0, 1] / cov[0, 0] if cov[0, 0] > 0 else np.nan
    # alpha (年化)
    resid = rb - beta * ra
    alpha_ann = resid.mean() * 252
    t_alpha = resid.mean() / resid.std() * np.sqrt(len(resid)) if resid.std() > 0 else np.nan
    return {
        "sharpe_a": float(sharpe_a),
        "sharpe_b": float(sharpe_b),
        "dd_a": float(dd_a),
        "dd_b": float(dd_b),
        "annual_a": float(ann_a),
        "annual_b": float(ann_b),
        "beta_to_a": float(beta),
        "alpha_annual": float(alpha_ann),
        "t_alpha": float(t_alpha),
        "n_days": int(len(common)),
    }


def run_integration(method: str = "weighted",
                    stat: str = "signal_strength",
                    alpha: float = 0.3,
                    direction: int = 1) -> dict:
    """跑 vol_target 调节模拟 → factor_vol_adj_nav.csv."""
    main_cta_nav = _load_main_cta_nav()
    if main_cta_nav is None:
        LOG.error("主 CTA NAV 不存在")
        return {}
    zk_state = _load_zk_state_variable(method=method, stat=stat)
    if zk_state is None:
        LOG.error(f"ZK 状态变量不存在 (method={method}, stat={stat})")
        return {}
    zk_state_norm = _normalize_state(zk_state)
    LOG.info(f"ZK 状态变量: method={method}, stat={stat}, "
             f"均值={zk_state_norm.mean():.3f}, std={zk_state_norm.std():.3f}, "
             f"范围=[{zk_state_norm.min():.3f}, {zk_state_norm.max():.3f}]")

    adj_nav = simulate_vol_target_adjustment(
        main_cta_nav, zk_state_norm, alpha=alpha, direction=direction)
    fname = f"factor_vol_adj_{method}_{stat}_a{alpha}_d{direction}.csv"
    adj_nav.to_csv(OUT_DIR / fname, header=["nav"])
    LOG.info(f"输出: {fname}")

    metrics = evaluate_pair(main_cta_nav, adj_nav)
    LOG.info(f"评估: {metrics}")
    return metrics


def run_alpha_grid(method: str = "weighted",
                   stat: str = "signal_strength") -> pd.DataFrame:
    """网格搜 α, 找出 vol_target 调节最优强度."""
    rows = []
    for alpha in ALPHA_CANDID:
        for direction in [1, -1]:
            LOG.info(f"α={alpha}, direction={direction}")
            m = run_integration(method=method, stat=stat, alpha=alpha,
                                  direction=direction)
            if not m:
                continue
            m["alpha"] = alpha
            m["direction"] = direction
            m["method"] = method
            m["stat"] = stat
            rows.append(m)
    df = pd.DataFrame(rows)
    return df


def main() -> int:
    """CLI: 跑 vol_target 整合模拟 → 网格搜 + 最优配置单点."""
    import argparse

    parser = argparse.ArgumentParser(description="ZK 状态变量接入主 CTA vol_target")
    parser.add_argument("--method", default="weighted", choices=["weighted", "quantile"])
    parser.add_argument("--stat", default="signal_dispersion",
                        choices=["signal_strength", "signal_dispersion", "signal_momentum"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--direction", type=int, default=-1)
    args = parser.parse_args()

    LOG.info(f"参数: method={args.method}, stat={args.stat}, "
             f"alpha={args.alpha}, direction={args.direction}")
    df = run_alpha_grid(method=args.method, stat=args.stat)
    out = OUT_DIR / f"vol_adj_grid_{args.method}_{args.stat}.csv"
    df.to_csv(out, index=False)
    LOG.info(f"网格结果: {out}")

    # 最优配置
    best = df.loc[df["sharpe_b"].idxmax()]
    LOG.info(f"最优 (max Sharpe): α={best['alpha']}, dir={best['direction']}, "
             f"Sharpe={best['sharpe_b']:.3f}, DD={best['dd_b']:.2%}")
    run_integration(method=args.method, stat=args.stat,
                    alpha=float(best["alpha"]), direction=int(best["direction"]))
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
