#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/6_copper_v4.py — 铜模型v4回测."""
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

from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v4 import (
    build_signal_v4, backtest_v4,
)
from QuantNodes.strategy.commodity_fundamental.pipeline_v3 import build_signal_v3, _close
from QuantNodes.strategy.commodity_fundamental.portfolio import compute_metrics

DB = REPO / "data" / "cmf" / "fundamental.duckdb"
OUT = REPO / "reports" / "cmf"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("铜模型v4回测 (VIX + DXY + 库存)")
    print("=" * 70)

    con = duckdb.connect(str(DB), read_only=True)

    # 1. v4模型
    print("\n[1] v4模型回测...")
    sig_v4 = build_signal_v4("CU0", con)
    close = _close("CU0", con)
    m_v4 = backtest_v4(sig_v4, close)

    print(f"\n  v4模型指标:")
    for k, v in m_v4.items():
        if k != "nav":
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # 2. v3模型 (对比)
    print("\n[2] v3模型回测...")
    sig_v3 = build_signal_v3("CU0", con)
    close = _close("CU0", con)
    common = sig_v3.index.intersection(close.index)
    pos_v3 = sig_v3.loc[common, "position"].fillna(0)
    ret = close.pct_change().reindex(common)
    strat_v3 = pos_v3.shift(1) * ret
    strat_v3 = strat_v3.dropna()
    m_v3 = compute_metrics((1 + strat_v3).cumprod(), strat_v3)
    m_v3["nav"] = (1 + strat_v3).cumprod()  # 确保nav键存在

    # 3. 买入持有 (对比)
    print("\n[3] 买入持有回测...")
    close_w = close.resample("W-FRI").last()
    bh_ret = close_w.pct_change()
    bh_nav = (1 + bh_ret).cumprod()
    bh_ret = bh_ret.dropna()
    m_bh = compute_metrics((1 + bh_ret).cumprod(), bh_ret)

    # 4. 对比
    print("\n[4] 模型对比:")
    print(f"\n{'指标':>20s} {'v4':>12s} {'v3':>12s} {'买入持有':>12s}")
    print("-" * 60)
    for k in ['ann_ret', 'ann_vol', 'sharpe', 'max_dd', 'calmar', 'win_rate']:
        v4 = m_v4[k]
        v3 = m_v3[k]
        bh = m_bh[k]
        print(f"{k:20s} {v4:12.4f} {v3:12.4f} {bh:12.4f}")

    # 5. 分年对比 (简化版)
    print("\n[5] 分年对比:")
    common_v4 = sig_v4.index.intersection(close.index)
    pos_v4 = sig_v4.loc[common_v4, "position"].fillna(0)
    ret_v4 = close.pct_change().reindex(common_v4)
    strat_v4 = pos_v4.shift(1) * ret_v4

    print(f"\n{'年份':>6s} {'v4年化':>10s} {'v4超额':>10s}")
    print("-" * 30)
    for year in range(2019, 2027):
        mask_v4 = common_v4.year == year
        if mask_v4.sum() > 10:
            v4_y = strat_v4[mask_v4].mean() * 52
            # 简化: 用v4的买入持有作为基准
            bh_y = ret_v4[mask_v4].mean() * 52
            alpha = v4_y - bh_y
            print(f"{year:>6d} {v4_y:>9.1%} {alpha:>+9.1%}")

    # 6. VIX信号有效性
    print("\n[6] VIX信号有效性:")
    print(f"  VIX z-score统计:")
    print(f"    均值: {sig_v4['vol_z'].mean():.2f}")
    print(f"    标准差: {sig_v4['vol_z'].std():.2f}")

    print(f"\n  各VIX区间的市场收益:")
    common_v4_6 = sig_v4.index.intersection(close.index)
    ret_v4_6 = close.pct_change().reindex(common_v4_6)
    vol_z_6 = sig_v4.loc[common_v4_6, "vol_z"]
    for label, mask in [("VIX>1 (高波动)", vol_z_6 > 1),
                        ("0<VIX<1 (中波动)", (vol_z_6 > 0) & (vol_z_6 <= 1)),
                        ("VIX<=0 (低波动)", vol_z_6 <= 0)]:
        if mask.sum() > 0:
            avg_ret = ret_v4_6[mask].mean() * 52
            print(f"    {label}: 年化={avg_ret:.1%}, 样本数={mask.sum()}")

    # 7. 图表
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # 价格 + 仓位
    ax1 = axes[0]
    ax1.plot(close_w.index, close_w.values, label="CU0", color="C0")
    ax1.set_ylabel("CU0 Price")
    ax1_twin = ax1.twinx()
    ax1_twin.fill_between(sig_v4.index, sig_v4["position"], 0, alpha=0.3, color="C1")
    ax1_twin.set_ylabel("Position")
    ax1_twin.set_ylim(0, 1.1)
    ax1.set_title("铜模型v4: VIX + DXY + 库存")
    ax1.grid(alpha=0.3)

    # VIX + DXY信号
    ax2 = axes[1]
    ax2.plot(sig_v4.index, sig_v4["vol_z"], label="VIX z-score", lw=1.5)
    ax2.plot(sig_v4.index, sig_v4["dxy_z"], label="DXY z-score", lw=1.5)
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_ylabel("Z-score")
    ax2.set_title("VIX与DXY信号")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)

    # NAV
    ax3 = axes[2]
    ax3.plot(m_v4["nav"].index, m_v4["nav"].values, label="v4", lw=1.5)
    ax3.plot(m_v3["nav"].index, m_v3["nav"].values, label="v3", lw=1.5, ls="--")
    ax3.set_ylabel("NAV")
    ax3.set_title(f"NAV对比: v4={m_v4['sharpe']:.2f}, v3={m_v3['sharpe']:.2f}")
    ax3.legend()
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "copper_v4_backtest.png", dpi=110)
    plt.close()
    print(f"\n✓ 图表已保存: {OUT / 'copper_v4_backtest.png'}")


if __name__ == "__main__":
    main()