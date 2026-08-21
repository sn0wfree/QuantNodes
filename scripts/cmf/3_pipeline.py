#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/3_pipeline.py — 单品种信号 + 组合回测 (最终版)."""
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

from QuantNodes.strategy.commodity_fundamental.pipeline import build_signal
from QuantNodes.strategy.commodity_fundamental.portfolio import (
    PortfolioConfig, simulate_portfolio, compute_metrics, build_target_weights,
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
    print("CMF Phase 3 — 单品种信号 + 组合回测 (最终版)")
    print("=" * 70)

    con = duckdb.connect(str(DB), read_only=True)
    weekly_ret = load_weekly_returns(con)
    print(f"\n周度收益矩阵: {weekly_ret.shape}, {weekly_ret.index.min()}~{weekly_ret.index.max()}")

    signal_results = {}
    single_metrics = {}

    for sym in ["CU0", "SC0", "M0"]:
        print(f"\n──── {PRODUCT_LABEL[sym]} ({sym}) ────")
        sig = build_signal(sym, con)
        sig.to_csv(OUT / f"signal_{sym}.csv")

        pos = sig["position"].dropna()
        ret = weekly_ret[sym].reindex(pos.index).fillna(0.0)
        strat_ret = pos.shift(1).fillna(0.0) * ret
        nav = (1 + strat_ret).cumprod()
        m = compute_metrics(nav, strat_ret)
        single_metrics[sym] = m
        print(f"  仓位覆盖: {pos.notna().sum()}/{len(sig)} 周, mean={pos.mean():.3f}")
        print(f"  当前 ({pos.index[-1].date()}) 仓位: {pos.iloc[-1]:.2f}")
        print(f"  单品种回测: 年化={m['ann_ret']:.3f}, Sharpe={m['sharpe']:.3f}, MaxDD={m['max_dd']:.3f}")
        ref = REPORT_REF[sym]
        print(f"  vs 报告参考: 年化={ref['ann_ret']:.3f}, Sharpe={ref['sharpe']:.3f}, MaxDD={ref['max_dd']:.3f}")
        print(f"  → Sharpe实现比: {m['sharpe'] / ref['sharpe'] * 100:.1f}%")
        signal_results[sym] = pos

    print("\n──── 组合回测 ────")
    cfg = PortfolioConfig()
    common = weekly_ret.dropna(how="any").index
    signals = {sym: signal_results[sym].reindex(common).fillna(0.0) for sym in signal_results}
    weekly_ret_clean = weekly_ret.loc[common]

    nav, port_ret, actual_w = simulate_portfolio(weekly_ret_clean, signals, cfg)
    metrics = compute_metrics(nav, port_ret)

    print(f"\n  组合指标:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")

    port_df = pd.DataFrame({"nav": nav, "ret": port_ret})
    port_df.to_csv(OUT / "portfolio_nav.csv")
    actual_w.to_csv(OUT / "portfolio_weights.csv")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, sym in enumerate(["CU0", "SC0", "M0"]):
        ax = axes[i]
        sig = signal_results[sym].dropna()
        ax.plot(sig.index, sig.values, lw=1.2)
        ax.fill_between(sig.index, sig.values, 0, alpha=0.25)
        ax.set_ylabel(f"{PRODUCT_LABEL[sym]}\n仓位")
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3)
    axes[0].set_title("三品种仓位序列")
    plt.tight_layout()
    plt.savefig(OUT / "position_paths.png", dpi=110)
    plt.close()

    bench_nav = (1 + weekly_ret_clean.mean(axis=1)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(nav.index, nav.values, label="CMF-IC 组合", lw=1.5)
    ax.plot(bench_nav.index, bench_nav.values / bench_nav.iloc[0],
            label="三品种等权基准", lw=1.0, ls="--", alpha=0.6)
    ax.set_title(f"组合 NAV: 终值={nav.iloc[-1]:.2f}x")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "portfolio_nav.png", dpi=110)
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.stackplot(actual_w.index, actual_w.T.values,
                 labels=actual_w.columns, alpha=0.7)
    ax.set_title("实际权重堆叠")
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "portfolio_weights.png", dpi=110)
    plt.close()

    summary_path = OUT / "summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# CMF-IC 商品基本面策略 — 复现总结\n\n")
        f.write("## 数据基础 (来自 akshare 免费源)\n")
        f.write("- 铜: LME库存(2014-07~) + 美国钻机数 + 中国PMI\n")
        f.write("- 原油: EIA库存变动(1982~) + 美国钻机数 + 中国PMI\n")
        f.write("- 豆粕: M主力价(库存代理) + 猪价指数 + 中国PMI\n")
        f.write("- 期货主力: CU0/SC0/M0 (akshare)\n\n")
        f.write("## 单品种回测 vs 报告参考\n\n")
        f.write("| 品种 | 实现年化 | 报告年化 | 实现Sharpe | 报告Sharpe | Sharpe比 | 实现MaxDD | 报告MaxDD |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for sym in ["CU0", "SC0", "M0"]:
            m = single_metrics[sym]
            r = REPORT_REF[sym]
            f.write(f"| {PRODUCT_LABEL[sym]} | {m['ann_ret']*100:.1f}% | {r['ann_ret']*100:.1f}% | "
                    f"{m['sharpe']:.2f} | {r['sharpe']:.2f} | {m['sharpe']/r['sharpe']*100:.0f}% | "
                    f"{m['max_dd']*100:.1f}% | {r['max_dd']*100:.1f}% |\n")
        f.write("\n## 组合回测 (vs 报告 22.78%/1.60/-15.10%)\n\n")
        f.write(f"- 年化: {metrics['ann_ret']*100:.2f}%\n")
        f.write(f"- 波动: {metrics['ann_vol']*100:.2f}%\n")
        f.write(f"- Sharpe: {metrics['sharpe']:.2f}\n")
        f.write(f"- 最大回撤: {metrics['max_dd']*100:.2f}%\n")
        f.write(f"- Calmar: {metrics['calmar']:.2f}\n")
        f.write(f"- 胜率: {metrics['win_rate']*100:.1f}%\n\n")
        f.write("## 已知偏差来源\n\n")
        f.write("- LME/ICE COT数据不可达 → 拥挤度§3.3退化为波动率单维\n")
        f.write("- 洋山铜溢价/铜精矿TC不可达 → 铜Lasso只用3因子\n")
        f.write("- 大豆压榨利润/豆粕历史库存不可达 → 豆粕Lasso只用价格+猪价\n")
        f.write("- 钻机数akshare停在2024-02, 后续用最后观察值占位\n")
        f.write("- 报告使用2021-2026区间, 我们用数据自然边界(2014-2026)\n\n")
        f.write("## 当前仓位 (2026-08-21)\n\n")
        for sym in ["CU0", "SC0", "M0"]:
            f.write(f"- {PRODUCT_LABEL[sym]} ({sym}): {signal_results[sym].iloc[-1]:.2f}\n")
    print(f"\n✓ 报告已保存到: {summary_path}")

    con.close()


if __name__ == "__main__":
    main()