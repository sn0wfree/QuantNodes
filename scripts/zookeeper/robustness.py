# coding: utf-8
"""E3: ZK vol_target 调节的鲁棒性测试.

1. 三 stat × 多 α × 多 dir 完整网格搜 → 汇总 best table
2. 多 stat 集成: 等权/加权合成三 stat 的调节信号
3. 滚动逐年评估: 按年拆分评估每年 α_ann 与 Sharpe 稳健性
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")
from scripts.zookeeper import integrate
from scripts.zookeeper.config import OUT_DIR

LOG = logging.getLogger(__name__)
LOG = logging.getLogger("zookeeper.robustness")
LOG.setLevel(logging.INFO)

STATS = ["signal_strength", "signal_dispersion", "signal_momentum"]
ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.7]
DIRECTIONS = [-1, 1]


def grid_all_stats(method: str = "weighted") -> pd.DataFrame:
    """三 stat × 多 α × ±1 完整网格."""
    rows = []
    for stat in STATS:
        for alpha in ALPHAS:
            for direction in DIRECTIONS:
                LOG.info(f"stat={stat} α={alpha} dir={direction}")
                m = integrate.run_integration(method=method, stat=stat,
                                                alpha=alpha, direction=direction)
                if not m:
                    continue
                m["stat"] = stat
                m["alpha"] = alpha
                m["direction"] = direction
                rows.append(m)
    df = pd.DataFrame(rows)
    out = OUT_DIR / f"vol_adj_grid_all_stats_{method}.csv"
    df.to_csv(out, index=False)
    LOG.info(f"网格汇总 ({len(df)} 条): {out}")
    return df


def best_per_stat(df: pd.DataFrame) -> pd.DataFrame:
    """每个 stat 的最优 (max Sharpe_B) 配置."""
    rows = []
    for stat in STATS:
        sub = df[df["stat"] == stat]
        if sub.empty:
            continue
        best = sub.loc[sub["sharpe_b"].idxmax()]
        rows.append({"stat": stat, **best.to_dict()})
    return pd.DataFrame(rows)


def composite_stat(method: str = "weighted",
                    weights: dict | None = None) -> pd.Series:
    """多 stat 集成: 等权或加权合成调节信号.

    Returns:
        月度 Series,  index=MonthEnd,  值在 [0, 1] 范围.
    """
    weights = weights or {"signal_strength": 1/3, "signal_dispersion": 1/3,
                           "signal_momentum": 1/3}
    frames = {}
    for stat in STATS:
        s = integrate._load_zk_state_variable(method=method, stat=stat)
        frames[stat] = integrate._normalize_state(s)
    common_idx = frames[STATS[0]].index
    for s in STATS[1:]:
        common_idx = common_idx.intersection(frames[s].index)
    composite = pd.Series(0.0, index=common_idx)
    for stat, w in weights.items():
        composite = composite + frames[stat].reindex(common_idx) * w
    return composite


def simulate_composite(method: str = "weighted",
                          alpha: float = 0.3,
                          direction: int = -1) -> dict:
    """跑多 stat 集成版 simulate."""
    state = composite_stat(method=method)
    main_nav = integrate._load_main_cta_nav()
    if main_nav is None:
        return {}
    adj_nav = integrate.simulate_vol_target_adjustment(
        main_nav, state, alpha=alpha, direction=direction)
    fname = f"factor_vol_adj_composite_a{alpha}_d{direction}.csv"
    adj_nav.to_csv(OUT_DIR / fname, header=["nav"])
    return integrate.evaluate_pair(main_nav, adj_nav)


def rolling_yearly_eval(method: str = "weighted",
                         stat: str = "signal_dispersion",
                         alpha: float = 0.3,
                         direction: int = -1) -> pd.DataFrame:
    """滚动逐年评估: 检验 vol_target 调节的跨期稳定性."""
    main_nav = integrate._load_main_cta_nav()
    zk_state = integrate._load_zk_state_variable(method=method, stat=stat)
    zk_state = integrate._normalize_state(zk_state)
    adj_nav = integrate.simulate_vol_target_adjustment(
        main_nav, zk_state, alpha=alpha, direction=direction)

    rows = []
    for year in range(2019, 2026):
        mask = (main_nav.index.year == year)
        if mask.sum() < 50:
            continue
        m_a = main_nav.loc[mask]
        m_b = adj_nav.loc[mask]
        ret_a = m_a.pct_change().dropna()
        ret_b = m_b.pct_change().dropna()
        if ret_a.empty or ret_b.empty:
            continue
        sh_a = ret_a.mean() / ret_a.std() * np.sqrt(252) if ret_a.std() > 0 else np.nan
        sh_b = ret_b.mean() / ret_b.std() * np.sqrt(252) if ret_b.std() > 0 else np.nan
        # OLS regression
        from numpy.linalg import lstsq
        common = ret_a.index.intersection(ret_b.index)
        if len(common) < 50:
            continue
        y = ret_b.loc[common].values
        X = np.vstack([np.ones(len(common)), ret_a.loc[common].values]).T
        coef, _, _, _ = lstsq(X, y, rcond=None)
        alpha_m = coef[0] * 252
        # DD
        dd_a = (m_a / m_a.cummax() - 1).min()
        dd_b = (m_b / m_b.cummax() - 1).min()
        rows.append({"year": year,
                     "sharpe_a": float(sh_a),
                     "sharpe_b": float(sh_b),
                     "delta_sharpe": float(sh_b - sh_a),
                     "dd_a": float(dd_a),
                     "dd_b": float(dd_b),
                     "alpha_annual": float(alpha_m),
                     "n_days": int(mask.sum())})
    return pd.DataFrame(rows)


def run_robustness(method: str = "weighted") -> dict:
    """一站式鲁棒性测试."""
    # 1. 三 stat 完整网格
    df_grid = grid_all_stats(method=method)
    best_per = best_per_stat(df_grid)
    LOG.info("=== 各 stat 最优 ===")
    for _, row in best_per.iterrows():
        LOG.info(f"{row['stat']}: α={row['alpha']} dir={int(row['direction'])} "
                 f"Sharpe_B={row['sharpe_b']:.3f} α_ann={row['alpha_annual']:.2%}")

    # 2. 多 stat 集成 (等权)
    LOG.info("\n=== 多 stat 集成 (等权) ===")
    composite_metrics = simulate_composite(method=method, alpha=0.3, direction=-1)
    LOG.info(f"Composite: {composite_metrics}")

    # 3. 滚动逐年评估 (取最优 stat)
    LOG.info("\n=== 滚动逐年评估 (signal_dispersion α=0.3 dir=-1) ===")
    df_yearly = rolling_yearly_eval(method=method, stat="signal_dispersion",
                                       alpha=0.3, direction=-1)
    out_y = OUT_DIR / "vol_adj_rolling_yearly.csv"
    df_yearly.to_csv(out_y, index=False)
    LOG.info(f"逐年评估 ({len(df_yearly)} 年): {out_y}")
    for _, row in df_yearly.iterrows():
        LOG.info(f"  {int(row['year'])}: ΔSharpe={row['delta_sharpe']:+.3f} "
                 f"α_ann={row['alpha_annual']:.2%} DD_B={row['dd_b']:.2%}")

    return {
        "grid": df_grid,
        "best_per_stat": best_per,
        "composite": composite_metrics,
        "yearly": df_yearly,
    }


if __name__ == "__main__":
    import argparse
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="ZK vol_adj 鲁棒性测试")
    parser.add_argument("--method", default="weighted")
    args = parser.parse_args()
    run_robustness(method=args.method)
