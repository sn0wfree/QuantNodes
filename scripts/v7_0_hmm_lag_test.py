"""v7.0 HMM 滞后回测 (Stage 30.5 Phase A4).

[动机] HMM 5 状态检测在转折点可能滞后 1-5 日, 错过最佳调仓时机.
       故意把 HMM state 滞后 1/3/5/10 日, 对比原始 vs 滞后 的 metrics.

[测试]
    对 5 方案 (Top-K / BL / Beta / Momentum / IV), 各跑 4 个滞后水平:
        lag_0  (原始, 假设实时)
        lag_1  (滞后 1 日)
        lag_3  (滞后 3 日)
        lag_5  (滞后 5 日)
        lag_10 (滞后 10 日)
    输出: 5 × 5 = 25 行 (strategy × lag)

[输出] reports/.../v7_0_hmm_lag.csv
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

ETFS = ['510300', '510500', '159915', '518880', '512760', '513100', '510880']

STRATEGIES = {
    "A_topk": ("Top-K (K=5)", lambda p, t: run_topk_v7_backtest(p, t, k=5)),
    "B_bl": ("Black-Litterman", lambda p, t: run_bl_v7_backtest(p, t, tau=0.05, max_weight=0.30)),
    "C_beta": ("Macro Beta (K=5)", lambda p, t: run_beta_v7_backtest(p, t, lookback=252, k=5)),
    "D_momentum": ("Momentum (63d)", lambda p, t: run_momentum_v7_backtest(p, t, lookback=63, k=5)),
    "E_iv": ("Inverse Vol", lambda p, t: run_iv_v7_backtest(p, t, lookback=252, max_weight=0.30)),
}

LAG_DAYS = [0, 1, 3, 5, 10]


def shift_timeline(tl: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """HMM timeline 滞后 N 日."""
    if lag_days == 0:
        return tl
    shifted = tl.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)
    return shifted


def main() -> None:
    print("[v7.0 HMM 滞后回测] 加载数据...")
    nav_main = pd.read_parquet(REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet")
    sb = pd.read_parquet(REPO / "data/real/etf_nav_smartbeta_2018-01-01_2026-06-30.parquet")
    panel_full = pd.DataFrame()
    for c in ETFS:
        if c in nav_main.columns:
            s = nav_main[c].dropna()
        elif c in sb.columns:
            s = sb[c].dropna()
        else:
            continue
        panel_full[c] = s
    panel_full = panel_full.dropna(how='all').ffill().dropna()
    print(f"  panel: {panel_full.shape}")

    tl_orig = build_regime_timeline()
    tl_orig['date'] = pd.to_datetime(tl_orig['date'])
    tl_orig = tl_orig.set_index('date')
    print(f"  HMM timeline: {tl_orig.index[0].date()} - {tl_orig.index[-1].date()}")

    rows = []
    for lag in LAG_DAYS:
        tl_shifted = shift_timeline(tl_orig, lag)
        print(f"\n=== HMM 滞后 {lag} 日 ===")
        for strat_key, (strat_name, strat_fn) in STRATEGIES.items():
            try:
                nav_df, weights_df, m = strat_fn(panel_full, tl_shifted)
                rows.append({
                    "lag_days": lag,
                    "strategy": strat_key,
                    "ann": m["ann"],
                    "sharpe": m["sharpe"],
                    "dd": m["dd"],
                    "calmar": m["calmar"],
                })
                print(f"  {strat_name:25s}  ann={m['ann']*100:+.2f}% "
                      f"DD={m['dd']*100:.2f}% Calmar={m['calmar']:.3f}")
            except Exception as e:
                print(f"  {strat_name:25s}  ERROR: {type(e).__name__}: {e}")

    df = pd.DataFrame(rows)
    out_dir = REPO / "reports/momentum_etf_rotation/v7"
    csv_path = out_dir / "v7_0_hmm_lag.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[save] {csv_path}")

    print("\n=== HMM 滞后影响汇总 (ann / calmar) ===")
    pivot_ann = df.pivot(index="strategy", columns="lag_days", values="ann")
    pivot_calmar = df.pivot(index="strategy", columns="lag_days", values="calmar")
    print("\n[年化收益] lag_0 vs lag_1 vs lag_3 vs lag_5 vs lag_10")
    print(pivot_ann.multiply(100).round(2).to_string())
    print("\n[Calmar]")
    print(pivot_calmar.round(3).to_string())

    print("\n=== 滞后衰减 (lag_N/lag_0 比率) ===")
    for col in [1, 3, 5, 10]:
        ratio = pivot_ann[col] / pivot_ann[0]
        print(f"\nlag_{col} / lag_0 (年化比率):")
        print(ratio.round(3).to_string())


if __name__ == "__main__":
    main()
