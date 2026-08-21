#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/7_copper_v5.py — 铜模型v5回测 (完整报告逻辑 + 期限结构)."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v5 import (
    build_signal_v5, backtest_v5,
)
from QuantNodes.strategy.commodity_fundamental.pipeline_v3 import build_signal_v3, _close
from QuantNodes.strategy.commodity_fundamental.portfolio import compute_metrics

DB = REPO / "data" / "cmf" / "fundamental.duckdb"
OUT = REPO / "reports" / "cmf"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("铜模型v5回测 (完整报告逻辑 + 期限结构)")
    print("=" * 70)

    con = duckdb.connect(str(DB), read_only=True)

    # 1. v5模型
    print("\n[1] v5模型回测...")
    sig_v5 = build_signal_v5("CU0", con)
    close = _close("CU0", con)
    m_v5 = backtest_v5(sig_v5, close)

    print(f"\n  v5模型指标:")
    for k, v in m_v5.items():
        if k != "nav":
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 2. v3模型 (对比, 只用2018年后的数据)
    print("\n[2] v3模型回测 (2018年后)...")
    sig_v3 = build_signal_v3("CU0", con)
    sig_v3 = sig_v3[sig_v3.index >= pd.Timestamp("2018-01-02")]
    common_v3 = sig_v3.index.intersection(close.index)
    pos_v3 = sig_v3.loc[common_v3, "position"].fillna(0)
    ret_v3 = close.pct_change().reindex(common_v3)
    strat_v3 = pos_v3.shift(1) * ret_v3
    strat_v3 = strat_v3.dropna()
    m_v3 = compute_metrics((1 + strat_v3).cumprod(), strat_v3)

    # 3. 买入持有
    print("\n[3] 买入持有回测...")
    close_w = close.resample("W-FRI").last()
    bh_ret = close_w.pct_change().dropna()
    bh_ret = bh_ret[bh_ret.index >= pd.Timestamp("2018-01-02")]
    m_bh = compute_metrics((1 + bh_ret).cumprod(), bh_ret)

    # 4. 对比
    print("\n[4] 模型对比 (2018年后):")
    print(f"\n{'指标':>20s} {'v5':>12s} {'v3':>12s} {'买入持有':>12s}")
    print("-" * 58)
    for k in ['ann_ret', 'ann_vol', 'sharpe', 'max_dd', 'calmar', 'win_rate']:
        v5 = m_v5[k]
        v3 = m_v3[k]
        bh = m_bh[k]
        print(f"{k:20s} {v5:12.4f} {v3:12.4f} {bh:12.4f}")

    # 5. 分年对比
    print("\n[5] 分年对比:")
    common_v5 = sig_v5.index.intersection(close.index)
    pos_v5 = sig_v5.loc[common_v5, "position"].fillna(0)
    ret_v5 = close.pct_change().reindex(common_v5)
    strat_v5 = pos_v5.shift(1) * ret_v5

    print(f"\n{'年份':>6s} {'v5年化':>10s} {'买入持有':>10s} {'v5超额':>10s}")
    print("-" * 40)
    for year in range(2018, 2027):
        mask = common_v5.year == year
        if mask.sum() > 10:
            v5_y = strat_v5[mask].mean() * 52
            bh_y = ret_v5[mask].mean() * 52
            alpha = v5_y - bh_y
            print(f"{year:>6d} {v5_y:>9.1%} {bh_y:>9.1%} {alpha:>+9.1%}")

    # 6. 期限结构信号分析
    print("\n[6] 期限结构信号分析:")
    if "gap" in sig_v5.columns:
        print(f"  估值缺口均值: {sig_v5['gap'].mean():.4f}")
        print(f"  估值缺口std: {sig_v5['gap'].std():.4f}")

    # 7. 各因子有效性
    print("\n[7] 各因子有效性 (2018年后):")
    for name, col in [("库存象限0.7", "quadrant_pos"), ("拥挤度", "crowded")]:
        if col in sig_v5.columns:
            mask = sig_v5[col] == 1
            if mask.sum() > 0:
                avg_ret = ret_v5[mask].mean() * 52
                print(f"  {name}=1时: 年化={avg_ret:.1%}, 样本数={mask.sum()}")

    # 8. 仓位分布
    print("\n[8] 仓位分布:")
    print(sig_v5["position"].value_counts(dropna=False))

    # 9. 图表
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # 价格 + 仓位
    ax1 = axes[0]
    ax1.plot(close_w.index, close_w.values, label="CU0", color="C0")
    ax1.set_ylabel("CU0 Price")
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(sig_v5.index, sig_v5["position"], 0, alpha=0.3, color="C1")
    ax1_twin.set_ylabel("Position")
    ax1_twin.set_ylim(0, 1.1)
    ax1.set_title("铜模型v5: 报告逻辑 + 期限结构 (2018+)")
    ax1.grid(alpha=0.3)

    # 估值缺口
    ax2 = axes[1]
    ax2.plot(sig_v5.index, sig_v5["gap"], label="gap", color="C2")
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_ylabel("Valuation Gap")
    ax2.set_title("估值缺口 (正=低估, 负=高估)")
    ax2.grid(alpha=0.3)

    # NAV
    ax3 = axes[2]
    ax3.plot(m_v5["nav"].index, m_v5["nav"].values, label="v5", lw=1.5)
    ax3.set_ylabel("NAV")
    ax3.set_title(f"NAV: 终值={m_v5['nav'].iloc[-1]:.2f}x, Sharpe={m_v5['sharpe']:.2f}")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "copper_v5_backtest.png", dpi=110)
    plt.close()
    print(f"\n✓ 图表已保存: {OUT / 'copper_v5_backtest.png'}")


if __name__ == "__main__":
    main()