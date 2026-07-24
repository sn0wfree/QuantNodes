# coding=utf-8
"""scripts/v9/v9_threshold_search.py — 阈值网格搜索 + v8集成回测.

v8 集成用 60日滚动波动率排名作为 Bear% 代理 (快速, 无需 Jump Model).

用法:
    python3.11 scripts/v9/v9_threshold_search.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from QuantNodes.strategy.momentum_etf_rotation.v9.coupling import coupling_analysis
from QuantNodes.strategy.momentum_etf_rotation.v9.scoring import (
    compute_score_timeseries,
    score_to_signal_hysteresis,
    DEFAULT_IMF_WEIGHTS,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import (
    run_backtest,
    cost_sensitivity,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.position import (
    compute_v9_only_position,
    v8_bear_to_factor,
)


def load_and_preprocess():
    data_dir = REPO / "data" / "high_freq_macro"
    y_weekly = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    vix_daily = pd.read_parquet(data_dir / "macro_vix_daily.parquet")["vix"]

    hs300 = y_weekly.mean(axis=1).fillna(0)
    nav_hs300 = (1 + hs300).cumprod()
    log_hs300 = np.log(nav_hs300)

    from statsmodels.tsa.filters.hp_filter import hpfilter
    from vmdpy import VMD

    cycle, _ = hpfilter(log_hs300, lamb=100)
    u, _, omega = VMD(cycle.values, 1000, 0, 4, 0, 1, 1e-6)
    imfs = u

    coupling_result = coupling_analysis(imfs, date_index=cycle.index)
    vix_weekly = vix_daily.resample("W").last().reindex(cycle.index, method="ffill")

    total, cyc, cou, vix_sc, mp, _ = compute_score_timeseries(
        imfs, coupling_result, vix_weekly, cycle.index, cn_calibration=True,
    )

    return y_weekly, cycle.index, total, omega


def compute_v8_proxy(daily_returns: pd.DataFrame, bear_window: int = 60) -> pd.DataFrame:
    """用滚动波动率排名作为 v8 Bear% 代理."""
    vol = daily_returns.rolling(20).std()
    bear_pct = vol.rolling(bear_window).rank(pct=True).clip(0, 1).fillna(0.5)
    threshold = 0.3
    factor = pd.DataFrame(1.0, index=bear_pct.index, columns=bear_pct.columns)
    mask = bear_pct > threshold
    factor[mask] = np.maximum(1 - (bear_pct[mask] - threshold) / (1 - threshold), 0)
    return factor


def grid_search(y_weekly, score_series, date_index):
    thresholds = [
        (45, 25), (40, 20), (35, 15), (30, 10), (25, 5), (20, 0),
        (50, 20), (45, 15), (40, 15), (35, 10), (30, 5), (25, 0),
        (35, 20), (30, 15), (28, 12), (32, 18),
    ]
    y_aligned = y_weekly.reindex(date_index, method="ffill").fillna(0)
    print(f"  y_aligned shape: {y_aligned.shape}, index type: {type(y_aligned.index.dtype)}")

    results = []
    for upper, lower in thresholds:
        sig = score_to_signal_hysteresis(score_series, upper=upper, lower=lower, initial=1)
        pos = compute_v9_only_position(sig, y_aligned)
        nav, ret, met = run_backtest(pos, y_aligned, cost_bps=10.0)
        met["upper"] = upper
        met["lower"] = lower
        met["hold_pct"] = float(sig.mean())
        met["n_bull"] = int((sig == 1).sum())
        results.append(met)
    return pd.DataFrame(results)


def v8_integration_test(y_weekly, score_series, date_index, upper, lower):
    daily_returns = y_weekly.fillna(0)
    v8_factors = compute_v8_proxy(daily_returns)
    v8_factors_weekly = v8_factors.resample("W").last().reindex(date_index, method="ffill").fillna(0.5)

    v9_sig = score_to_signal_hysteresis(score_series, upper=upper, lower=lower, initial=1)

    equal_weights = pd.DataFrame(1.0/43, index=date_index, columns=y_weekly.columns)
    y_aligned = y_weekly.reindex(date_index, method="ffill").fillna(0)

    nav_e, _, met_e = run_backtest(equal_weights, y_aligned, cost_bps=10.0)
    met_e["strategy"] = "etf_equal_weight"

    nav_v8, _, met_v8 = run_backtest(equal_weights * v8_factors_weekly, y_aligned, cost_bps=10.0)
    met_v8["strategy"] = "v8_only"

    v9_signal_aligned = v9_sig.reindex(date_index, method="ffill").fillna(0)
    nav_v9, _, met_v9 = run_backtest(equal_weights * v9_signal_aligned, y_aligned, cost_bps=10.0)
    met_v9["strategy"] = "v9_only"

    v98_weights = equal_weights * v9_signal_aligned * v8_factors_weekly
    nav_98, _, met_98 = run_backtest(v98_weights, y_aligned, cost_bps=10.0)
    met_98["strategy"] = "v9_plus_v8"

    return pd.DataFrame([met_e, met_v8, met_v9, met_98])


def main():
    print("=" * 60)
    print("v9 阈值搜索 + v8 集成回测")
    print("=" * 60)

    y_weekly, date_index, score, omega = load_and_preprocess()
    print(f"  IMFs 中心频率: {omega[-1, :]}")
    print(f"  评分范围: {score.min():.1f} ~ {score.max():.1f}")

    print("\n[Phase A] 阈值网格搜索")
    search_df = grid_search(y_weekly, score, date_index)
    display_cols = ["upper", "lower", "Sharpe", "Calmar", "MaxDD", "AnnRet", "hold_pct", "n_bull"]
    print(search_df[display_cols].to_string(index=False))

    best = search_df.loc[search_df["Sharpe"].idxmax()]
    print(f"\n  最优: upper={int(best['upper'])}, lower={int(best['lower'])}")
    print(f"  Sharpe={best['Sharpe']:.3f}, Calmar={best['Calmar']:.3f}, MaxDD={best['MaxDD']:.2%}")

    print("\n[Phase B] v8 集成回测")
    integrated = v8_integration_test(y_weekly, score, date_index,
                                     upper=int(best["upper"]),
                                     lower=int(best["lower"]))
    print(integrated.to_string(index=False))

    print("\n[Phase C] 输出")
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    search_df.to_csv(output_dir / "threshold_grid_search.csv", index=False)
    integrated.to_csv(output_dir / "v8_integration_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax1 = axes[0]
    sc = ax1.scatter(search_df["upper"], search_df["lower"], c=search_df["Sharpe"],
                     cmap="RdYlGn", s=120, edgecolors="black", linewidth=0.5)
    plt.colorbar(sc, ax=ax1, label="Sharpe")
    ax1.scatter([best["upper"]], [best["lower"]], c="red", s=200, marker="*", zorder=10, label="Best")
    ax1.set_xlabel("Upper Threshold")
    ax1.set_ylabel("Lower Threshold")
    ax1.set_title("Sharpe by Threshold Grid")
    ax1.legend()

    ax2 = axes[1]
    sc2 = ax2.scatter(search_df["hold_pct"] * 100, search_df["Sharpe"],
                      c=search_df["upper"], cmap="viridis", s=100, edgecolors="black", linewidth=0.5)
    ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Target 0.5")
    ax2.set_xlabel("Hold Period %")
    ax2.set_ylabel("Sharpe")
    ax2.set_title("Sharpe vs Hold Period")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    strategies = integrated["strategy"].tolist()
    sharpes = integrated["Sharpe"].tolist()
    maxdds = [x * 100 for x in integrated["MaxDD"].tolist()]
    colors = ["#94a3b8", "#3b82f6", "#10b981", "#f59e0b"]
    bars = ax3.bar(strategies, sharpes, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, sharpes):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax3.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Target 0.5")
    ax3.set_ylabel("Sharpe")
    ax3.set_title("v8 Integration Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "threshold_search.png", dpi=150)
    plt.close()

    print(f"  图表: {output_dir / 'threshold_search.png'}")
    print(f"  阈值搜索: {output_dir / 'threshold_grid_search.csv'}")
    print(f"  集成对比: {output_dir / 'v8_integration_comparison.csv'}")

    print("\n" + "=" * 60)
    print("完成!")


if __name__ == "__main__":
    main()