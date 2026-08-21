#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/5_copper_pricing.py — 铜定价模型回测."""
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.commodity_fundamental.copper.copper_pricing import (
    load_copper_data, compute_copper_signal, backtest_copper,
)
from QuantNodes.strategy.commodity_fundamental.portfolio import compute_metrics

DB = REPO / "data" / "cmf" / "fundamental.duckdb"
OUT = REPO / "reports" / "cmf"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("铜定价模型回测")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    con = duckdb.connect(str(DB), read_only=True)
    data = load_copper_data(con)
    print(f"  因子数: {len(data)}")
    for name, s in data.items():
        print(f"  {name:12s}: {s.index.min()} ~ {s.index.max()}, n={len(s)}")

    # 2. 生成信号
    print("\n[2] 生成信号...")
    signal = compute_copper_signal(data)
    print(f"  信号范围: {signal.index.min()} ~ {signal.index.max()}")
    print(f"  仓位覆盖: {signal['position'].notna().sum()}/{len(signal)}")

    # 3. 回测
    print("\n[3] 回测...")
    close = data["close"].reindex(signal.index)
    metrics = backtest_copper(signal, close, cost_bps=5.0)

    print(f"\n  回测结果:")
    for k, v in metrics.items():
        if k != "nav":
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 4. 分年分析
    print("\n[4] 分年分析:")
    for year in range(2019, 2027):
        mask = signal.index.year == year
        if mask.sum() < 10:
            continue
        sig_y = signal.loc[mask, "position"]
        close_y = close.loc[mask]
        strat_ret = sig_y.shift(1) * close_y.pct_change()
        nav_y = (1 + strat_ret).cumprod()
        m = compute_metrics(nav_y, strat_ret)
        print(f"  {year}: 年化={m['ann_ret']:.1%}, Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1%}")

    # 5. 信号分布
    print("\n[5] 信号分布:")
    print(signal["position"].value_counts(dropna=False))

    # 6. 最近10周信号
    print("\n[6] 最近10周信号:")
    print(signal[["vol_regime", "inventory_cycle", "macro_env", "supply_demand",
                   "total_score", "position"]].tail(10).to_string())

    # 7. 图表
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    # 铜价 + 仓位
    ax1 = axes[0]
    ax1.plot(close.index, close.values, label="CU0", color="C0")
    ax1.set_ylabel("CU0 Price")
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(signal.index, signal["position"], 0, alpha=0.3, color="C1")
    ax1_twin.set_ylabel("Position")
    ax1_twin.set_ylim(0, 1.1)
    ax1.set_title("铜定价模型: 价格 + 仓位")
    ax1.grid(alpha=0.3)

    # 四层信号
    ax2 = axes[1]
    ax2.plot(signal.index, signal["vol_regime"], label="Vol Regime", lw=1.5)
    ax2.plot(signal.index, signal["inventory_cycle"], label="Inventory", lw=1.5)
    ax2.plot(signal.index, signal["macro_env"], label="Macro", lw=1.5)
    ax2.plot(signal.index, signal["supply_demand"], label="Supply/Demand", lw=1.5)
    ax2.set_ylabel("Signal")
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_title("四层信号")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    # 总得分
    ax3 = axes[2]
    ax3.plot(signal.index, signal["total_score"], label="Total Score", color="C2")
    ax3.axhline(0.5, color="r", ls="--", alpha=0.5, label="Buy")
    ax3.axhline(-0.2, color="g", ls="--", alpha=0.5, label="Sell")
    ax3.set_ylabel("Score")
    ax3.set_title("综合得分")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(alpha=0.3)

    # NAV
    ax4 = axes[3]
    nav = metrics["nav"]
    bh_nav = (1 + close.pct_change()).cumprod().reindex(nav.index)
    ax4.plot(nav.index, nav.values, label="Strategy", lw=1.5)
    ax4.plot(bh_nav.index, bh_nav.values, label="Buy & Hold", ls="--", alpha=0.6)
    ax4.set_ylabel("NAV")
    ax4.set_title(f"NAV: 终值={nav.iloc[-1]:.2f}x, Sharpe={metrics['sharpe']:.2f}")
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "copper_pricing_backtest.png", dpi=110)
    plt.close()
    print(f"\n✓ 图表已保存: {OUT / 'copper_pricing_backtest.png'}")


if __name__ == "__main__":
    main()