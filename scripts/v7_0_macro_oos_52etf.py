"""v7.0 5 方案 × {7 ETF, 52 ETF} × 5-fold OOS 对比 (Stage 30.5 Phase B2).

[动机] 验证 41 ETF 池 (量化筛选) 是否优于 7 ETF 池 (手工选).

[测试]
    5 方案 × 5 fold × {7 ETF, 41 ETF} = 50 backtests

[决策标准]
    1. 41 ETF 池 calmar_mean > 7 ETF 池 calmar_mean
    2. 41 ETF 池 dd_min < 7 ETF 池 dd_min (回撤改善)
    3. 5 方案中至少 3 个通过鲁棒筛选

[输出] reports/.../v7_0_52etf_oos_5fold.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    build_regime_timeline,
    run_topk_v7_backtest,
    run_bl_v7_backtest,
    run_beta_v7_backtest,
    run_momentum_v7_backtest,
    run_iv_v7_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.dynamic_allocation import _compute_metrics

warnings.filterwarnings("ignore")

OOS_END = "2026-06-30"
ETFS_7 = ['510300', '510500', '159915', '518880', '512760', '513100', '510880']


def load_panel(codes: list[str], nav_main: pd.DataFrame, sb: pd.DataFrame) -> pd.DataFrame:
    panel = pd.DataFrame()
    for c in codes:
        if c in nav_main.columns:
            s = nav_main[c]
        elif c in sb.columns:
            s = sb[c]
        else:
            continue
        panel[c] = s
    panel = panel.ffill().dropna(how='all')
    return panel


def oos_metrics(nav: pd.Series, oos_start: str, oos_end: str) -> dict:
    s = nav.loc[oos_start:oos_end].dropna()
    if len(s) < 2:
        return {"ann": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}
    n_days = (s.index[-1] - s.index[0]).days
    total_ret = s.iloc[-1] / s.iloc[0] - 1
    ann = (1 + total_ret) ** (365.25 / n_days) - 1
    monthly_ret = s.pct_change().dropna()
    ann_vol = monthly_ret.std() * np.sqrt(12) if len(monthly_ret) > 1 else 0.0
    dd = (s / s.cummax() - 1).min()
    return {
        "ann": ann, "sharpe": ann / ann_vol if ann_vol > 0 else 0.0,
        "dd": dd, "calmar": ann / abs(dd) if dd != 0 else 0.0,
    }


STRATEGIES = {
    "A_topk": ("Top-K (K=5)", lambda p, t: run_topk_v7_backtest(p, t, k=5)),
    "B_bl": ("Black-Litterman", lambda p, t: run_bl_v7_backtest(p, t, tau=0.05, max_weight=0.30)),
    "C_beta": ("Macro Beta (K=5)", lambda p, t: run_beta_v7_backtest(p, t, lookback=252, k=5)),
    "D_momentum": ("Momentum (63d)", lambda p, t: run_momentum_v7_backtest(p, t, lookback=63, k=5)),
    "E_iv": ("Inverse Vol", lambda p, t: run_iv_v7_backtest(p, t, lookback=252, max_weight=0.30)),
}

FOLD_CONFIGS = [
    (1, "2020-01-01", "2020-12-31"),
    (2, "2021-01-01", "2021-12-31"),
    (3, "2022-01-01", "2023-06-30"),
    (4, "2023-07-01", "2024-12-31"),
    (5, "2025-01-01", OOS_END),
]


def main() -> None:
    print("[v7.0 5 方案 × {7 ETF, 41 ETF} × 5-fold OOS] 加载数据...")
    nav_main = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    sb = pd.read_parquet(REPO / "data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")

    panel_7 = load_panel(ETFS_7, nav_main, sb)
    print(f"  7 ETF 池: {panel_7.shape}, range: {panel_7.index[0].date()} - {panel_7.index[-1].date()}")

    universe_path = REPO / "reports/momentum_etf_rotation/v7/v7_0_52etf_universe.csv"
    universe = pd.read_csv(universe_path)
    ETFS_41 = [str(c) for c in universe["code"].tolist()]
    panel_41 = load_panel(ETFS_41, nav_main, sb)
    print(f"  41 ETF 池: {panel_41.shape}, range: {panel_41.index[0].date()} - {panel_41.index[-1].date()}")

    tl_df = build_regime_timeline()
    tl_df['date'] = pd.to_datetime(tl_df['date'])
    tl_df = tl_df.set_index('date')

    pools = {"7etf": ("7 ETF (手工)", panel_7), "41etf": ("41 ETF (量化)", panel_41)}
    rows = []

    for pool_key, (pool_name, panel) in pools.items():
        print(f"\n=== {pool_name} ===")
        for fi, oos_start, oos_end in FOLD_CONFIGS:
            for strat_key, (strat_name, strat_fn) in STRATEGIES.items():
                try:
                    nav_df, weights_df, m = strat_fn(panel, tl_df)
                    m_oos = oos_metrics(nav_df["nav_cum"], oos_start, oos_end)
                    rows.append({
                        "pool": pool_key,
                        "fold": fi,
                        "strategy": strat_key,
                        **m_oos,
                    })
                    print(f"  Fold {fi} {strat_name:20s} ann={m_oos['ann']*100:+.2f}% DD={m_oos['dd']*100:.2f}% Calmar={m_oos['calmar']:.3f}")
                except Exception as e:
                    print(f"  Fold {fi} {strat_name:20s} ERROR: {type(e).__name__}: {e}")
                    rows.append({
                        "pool": pool_key, "fold": fi, "strategy": strat_key,
                        "ann": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0,
                    })

    df = pd.DataFrame(rows)
    out_dir = REPO / "reports/momentum_etf_rotation/v7"
    csv_path = out_dir / "v7_0_52etf_oos_5fold.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    print("\n=== 7 ETF vs 41 ETF 5-fold 平均对比 ===")
    summary = df.groupby(["pool", "strategy"]).agg(
        ann_mean=("ann", "mean"),
        ann_min=("ann", "min"),
        dd_min=("dd", "max"),
        calmar_mean=("calmar", "mean"),
        calmar_min=("calmar", "min"),
    ).reset_index()
    print(summary.round(3).to_string(index=False))

    print("\n=== 7 vs 41 改善 (41 - 7) ===")
    for strat in summary["strategy"].unique():
        s7 = summary[(summary["strategy"] == strat) & (summary["pool"] == "7etf")].iloc[0]
        s41 = summary[(summary["strategy"] == strat) & (summary["pool"] == "41etf")].iloc[0]
        calmar_delta = s41["calmar_mean"] - s7["calmar_mean"]
        ann_delta = s41["ann_mean"] - s7["ann_mean"]
        print(f"  {strat:12s}  calmar_mean Δ={calmar_delta:+.3f}  ann_mean Δ={ann_delta*100:+.2f}pp")


if __name__ == "__main__":
    main()
