#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/4_pipeline_v3.py — v3 pipeline 入口 + 对比 v2."""
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

from QuantNodes.strategy.commodity_fundamental.pipeline_v3 import build_signal_v3
from QuantNodes.strategy.commodity_fundamental.portfolio import (
    PortfolioConfig, simulate_portfolio, compute_metrics,
)

DB = REPO / "data" / "cmf" / "fundamental.duckdb"
OUT = REPO / "reports" / "cmf"
OUT.mkdir(parents=True, exist_ok=True)

PRODUCT_LABEL = {"CU0": "铜", "SC0": "原油", "M0": "豆粕"}

REPORT_REF = {
    "CU0": {"ann_ret": 0.0843, "sharpe": 0.942, "max_dd": -0.1581},
    "SC0": {"ann_ret": 0.2228, "sharpe": 0.982, "max_dd": -0.3452},
    "M0":  {"ann_ret": 0.0919, "sharpe": 0.754, "max_dd": -0.1181},
}


def load_weekly_returns(con) -> pd.DataFrame:
    panels = {}
    for sym, tbl in [("CU0", "futures_main_close_cu"), ("SC0", "futures_main_close_sc"), ("M0", "futures_main_close_m")]:
        df = con.execute(f"SELECT trade_date, close FROM {tbl} ORDER BY trade_date").df()
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        s = df.set_index("trade_date")["close"].resample("W-FRI").last()
        panels[sym] = s.pct_change()
    return pd.concat(panels, axis=1).dropna(how="all")


def main():
    print("=" * 70)
    print("CMF Phase 7 — v3 Pipeline (天风因子 + 完整拥挤度)")
    print("=" * 70)

    con = duckdb.connect(str(DB), read_only=True)
    weekly_ret = load_weekly_returns(con)
    print(f"\n周度收益矩阵: {weekly_ret.shape}, {weekly_ret.index.min()}~{weekly_ret.index.max()}")

    signal_results = {}
    single_metrics = {}

    for sym in ["CU0", "SC0", "M0"]:
        print(f"\n──── {PRODUCT_LABEL[sym]} ({sym}) v3 ────")
        sig = build_signal_v3(sym, con)
        sig.to_csv(OUT / f"signal_{sym}_v3.csv")

        pos = sig["position"].dropna()
        ret = weekly_ret[sym].reindex(pos.index).fillna(0.0)
        strat_ret = pos.shift(1).fillna(0.0) * ret
        nav = (1 + strat_ret).cumprod()
        m = compute_metrics(nav, strat_ret)
        single_metrics[sym] = m

        print(f"  仓位覆盖: {pos.notna().sum()}/{len(sig)} 周, mean={pos.mean():.3f}")
        print(f"  当前 ({pos.index[-1].date()}) 仓位: {pos.iloc[-1]:.2f}")
        print(f"  单品种: 年化={m['ann_ret']:.3f}, Sharpe={m['sharpe']:.3f}, MaxDD={m['max_dd']:.3f}")
        ref = REPORT_REF[sym]
        print(f"  vs 报告: 年化={ref['ann_ret']:.3f}, Sharpe={ref['sharpe']:.3f} (实现比 {m['sharpe']/ref['sharpe']*100:.0f}%)")
        signal_results[sym] = pos

    print("\n──── v3 组合回测 ────")
    cfg = PortfolioConfig()
    common = weekly_ret.dropna(how="any").index
    signals = {sym: signal_results[sym].reindex(common).fillna(0.0) for sym in signal_results}
    weekly_ret_clean = weekly_ret.loc[common]

    nav, port_ret, actual_w = simulate_portfolio(weekly_ret_clean, signals, cfg)
    metrics = compute_metrics(nav, port_ret)

    print(f"\n  v3 组合指标:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # ── 输出图表 ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, sym in enumerate(["CU0", "SC0", "M0"]):
        ax = axes[i]
        sig = signal_results[sym].dropna()
        ax.plot(sig.index, sig.values, lw=1.2)
        ax.fill_between(sig.index, sig.values, 0, alpha=0.25)
        ax.set_ylabel(f"{PRODUCT_LABEL[sym]}\n仓位")
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3)
    axes[0].set_title("v3 三品种仓位序列 (天风因子 + CFTC拥挤度)")
    plt.tight_layout()
    plt.savefig(OUT / "position_paths_v3.png", dpi=110)
    plt.close()

    # ── v2 vs v3 对比 ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(nav.index, nav.values, label="v3 (天风+CFTC)", lw=1.5)
    ax.set_title(f"v3 组合 NAV: 终值={nav.iloc[-1]:.2f}x, Sharpe={metrics['sharpe']:.2f}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "portfolio_nav_v3.png", dpi=110)
    plt.close()

    # ── 汇总报告 ──────────────────────────────────────────────────────────────
    summary_path = OUT / "summary_v3.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# CMF-IC v3 — Phase 7 天风全量接入报告\n\n")
        f.write("## 数据基础\n")
        f.write("- 铜: 天风洋山铜溢价 + 铜精矿TC + LME库存 + DXY + TIPS 10Y\n")
        f.write("- 原油: EIA WCRSTUS1 周度库存 + DXY + TIPS 10Y\n")
        f.write("- 豆粕: 天风豆粕库存_中国 + 江苏压榨利润 + 自繁自养 + DXY + TIPS 10Y\n")
        f.write("- 拥挤度: CFTC Managed Money net long % (D) + 波动率百分位 (U)\n\n")
        f.write("## 单品种回测 vs 报告参考\n\n")
        f.write("| 品种 | v3年化 | 报告年化 | v3 Sharpe | 报告Sharpe | 实现比 | v3 MaxDD | 报告MaxDD |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for sym in ["CU0", "SC0", "M0"]:
            m = single_metrics[sym]
            r = REPORT_REF[sym]
            f.write(f"| {PRODUCT_LABEL[sym]} | {m['ann_ret']*100:.1f}% | {r['ann_ret']*100:.1f}% | "
                    f"{m['sharpe']:.2f} | {r['sharpe']:.2f} | {m['sharpe']/r['sharpe']*100:.0f}% | "
                    f"{m['max_dd']*100:.1f}% | {r['max_dd']*100:.1f}% |\n")
        f.write("\n## 组合回测 (v3 vs 报告 22.78%/1.60/-15.10%)\n\n")
        f.write(f"- 年化: {metrics['ann_ret']*100:.2f}%\n")
        f.write(f"- 波动: {metrics['ann_vol']*100:.2f}%\n")
        f.write(f"- Sharpe: {metrics['sharpe']:.2f}\n")
        f.write(f"- 最大回撤: {metrics['max_dd']*100:.2f}%\n")
        f.write(f"- Calmar: {metrics['calmar']:.2f}\n")
        f.write(f"- 胜率: {metrics['win_rate']*100:.1f}%\n\n")
        f.write("## 当前仓位 (2026-08-21)\n\n")
        for sym in ["CU0", "SC0", "M0"]:
            f.write(f"- {PRODUCT_LABEL[sym]} ({sym}): {signal_results[sym].iloc[-1]:.2f}\n")
    print(f"\n✓ 报告已保存: {summary_path}")
    con.close()


if __name__ == "__main__":
    main()