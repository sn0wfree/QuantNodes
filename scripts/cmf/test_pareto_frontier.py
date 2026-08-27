"""scripts/cmf/test_pareto_frontier.py — Pareto前沿搜索.

对3个信号候选 (macro_regime / real_rate_tips_ma / momentum_long_only)
叠加 compute_risk_overlay, 做粗+精两阶段扫描, 找出Pareto前沿点.

评估指标:
  G = sqrt(Cal_OOS × Cal_IS)  主指标
  avg_ret = mean(ann_ret_OOS, ann_ret_IS)  副指标
  worst_dd = max(max_dd_OOS, max_dd_IS)  副指标(约束)
  Min(Cal_OOS, Cal_IS)  稳健性

Pareto支配: 点A被点B支配 ⟺
  B.ann_ret_OOS >= A.ann_ret_OOS AND B.max_dd_OOS >= A.max_dd_OOS AND
  B.ann_ret_IS >= A.ann_ret_IS AND B.max_dd_IS >= A.max_dd_IS
  AND 至少一项严格
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_macro_regime_signal,
    compute_risk_overlay,
    _load_indicator_d,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import (
    compute_real_rate_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import (
    compute_momentum_signal,
)

OOS_S, OOS_E = "2010-01-01", "2019-12-31"
IS_S, IS_E = "2020-01-01", "2026-08-25"
COST_BPS = 5.0


def load_close_daily():
    import duckdb
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return (df.drop_duplicates("trade_date", keep="last")
              .set_index("trade_date")["close"])


def load_close_weekly():
    return load_close_daily().resample("W-FRI").last().dropna()


def get_signal(name: str, close):
    """获取3个候选信号 (已reindex到close.index)."""
    if name == "macro_regime":
        return compute_macro_regime_signal()["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    elif name == "real_rate_tips_ma":
        return compute_real_rate_signal(variant="tips_ma")["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    elif name == "momentum_long_only":
        # weekly → reindex to daily
        pos_w = compute_momentum_signal(variant="long_only")["position"]
        return pos_w.reindex(close.index).fillna(method="ffill").fillna(0)
    else:
        raise ValueError(f"Unknown signal: {name}")


def metrics_freq(pos, close, s, e, freq="D"):
    periods_per_y = 52 if freq == "W" else 252
    sub = pos[(pos.index >= s) & (pos.index <= e)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "max_dd": 0, "calmar": 0}
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * COST_BPS / 10000
    sr = (sr - cost).dropna()
    sp = c.iloc[0]
    nav = sp * (1 + sr).cumprod()
    n_y = len(c) / periods_per_y
    ar = (nav.iloc[-1] / sp) ** (1 / n_y) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"ann_ret": ar, "max_dd": dd, "calmar": ar / abs(dd) if dd != 0 else 0}


def evaluate(pos, close):
    """双窗口评估."""
    oos = metrics_freq(pos, close, OOS_S, OOS_E, "D")
    is_ = metrics_freq(pos, close, IS_S, IS_E, "D")
    g = (oos["calmar"] * is_["calmar"]) ** 0.5
    avg_ret = (oos["ann_ret"] + is_["ann_ret"]) / 2
    worst_dd = min(oos["max_dd"], is_["max_dd"])
    min_cal = min(oos["calmar"], is_["calmar"])
    return {
        "oos_ann": oos["ann_ret"], "oos_dd": oos["max_dd"], "oos_cal": oos["calmar"],
        "is_ann": is_["ann_ret"], "is_dd": is_["max_dd"], "is_cal": is_["calmar"],
        "g": g, "avg_ret": avg_ret, "worst_dd": worst_dd, "min_cal": min_cal,
    }


def pareto_dominates(a, b):
    """点A被点B支配. (ann_ret越大越好, max_dd越大(接近0)越好)."""
    return (
        b["oos_ann"] >= a["oos_ann"] and b["oos_dd"] >= a["oos_dd"]
        and b["is_ann"] >= a["is_ann"] and b["is_dd"] >= a["is_dd"]
        and (
            b["oos_ann"] > a["oos_ann"] or b["oos_dd"] > a["oos_dd"]
            or b["is_ann"] > a["is_ann"] or b["is_dd"] > a["is_dd"]
        )
    )


def pareto_frontier(points):
    """提取Pareto前沿点."""
    frontier = []
    for i, p in enumerate(points):
        dominated = False
        for j, q in enumerate(points):
            if i != j and pareto_dominates(p, q):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return frontier


def main():
    close = load_close_daily()
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    signals = ["macro_regime", "real_rate_tips_ma", "momentum_long_only"]
    raw_signals = {name: get_signal(name, close) for name in signals}

    # 粗扫矩阵 (27组合 × 3信号 = 81)
    print("=" * 130)
    print("Phase 2: 粗扫 (27 风控参数 × 3 信号)")
    print("=" * 130)

    coarse_grid = list(itertools.product(
        [-0.10, -0.15, -0.20],       # hard_cap
        [-0.02, -0.03, -0.04],       # daily_cap
        [0.15, 0.20, 0.25],          # vol_cap
    ))
    coarse_results = []
    for sig_name in signals:
        raw_pos = raw_signals[sig_name]
        for (hc, dc, vc) in coarse_grid:
            soft = hc + 0.03
            overlaid = compute_risk_overlay(
                close=close, position=raw_pos, vix=vix, gvz=gvz,
                hard_cap=hc, soft_cap=soft, daily_cap=dc, vol_cap=vc,
                cooldown_days=5,
            )
            m = evaluate(overlaid, close)
            m["sig"] = sig_name
            m["hc"] = hc
            m["dc"] = dc
            m["vc"] = vc
            coarse_results.append(m)

    # 精扫Top3
    coarse_sorted = sorted(coarse_results, key=lambda x: x["g"], reverse=True)
    top3 = coarse_sorted[:3]

    print("粗扫Top5 (按G排序):")
    print(f'{"信号":<22} {"hc":>6} {"dc":>6} {"vc":>6} | '
          f'{"OOS年化":>8} {"OOSDD":>8} {"OOSCal":>7} | '
          f'{"IS年化":>8} {"ISDD":>8} {"ISCal":>7} {"G":>6}')
    print("-" * 130)
    for r in coarse_sorted[:5]:
        print(f'{r["sig"]:<22} {r["hc"]:>6.2f} {r["dc"]:>6.2f} {r["vc"]:>6.2f} | '
              f'{r["oos_ann"]*100:>7.2f}% {r["oos_dd"]*100:>7.2f}% {r["oos_cal"]:>7.2f} | '
              f'{r["is_ann"]*100:>7.2f}% {r["is_dd"]*100:>7.2f}% {r["is_cal"]:>7.2f} {r["g"]:>6.2f}')

    # 精扫
    print("\n" + "=" * 130)
    print("Phase 3: 精扫Top3 ±20%扰动 (3 × 27 = 81 组合)")
    print("=" * 130)

    fine_results = []
    for base_combo in top3:
        base_hc, base_dc, base_vc = base_combo["hc"], base_combo["dc"], base_combo["vc"]
        sig_name = base_combo["sig"]
        raw_pos = raw_signals[sig_name]
        for mult_h in [0.8, 1.0, 1.2]:
            for mult_d in [0.8, 1.0, 1.2]:
                for mult_v in [0.8, 1.0, 1.2]:
                    hc = base_hc * mult_h
                    dc = base_dc * mult_d
                    vc = base_vc * mult_v
                    soft = hc + 0.03
                    overlaid = compute_risk_overlay(
                        close=close, position=raw_pos, vix=vix, gvz=gvz,
                        hard_cap=hc, soft_cap=soft, daily_cap=dc, vol_cap=vc,
                        cooldown_days=5,
                    )
                    m = evaluate(overlaid, close)
                    m["sig"] = sig_name
                    m["hc"] = hc
                    m["dc"] = dc
                    m["vc"] = vc
                    fine_results.append(m)

    fine_sorted = sorted(fine_results, key=lambda x: x["g"], reverse=True)
    all_results = coarse_results + fine_results
    all_sorted = sorted(all_results, key=lambda x: x["g"], reverse=True)

    print("精扫Top5:")
    for r in fine_sorted[:5]:
        print(f'{r["sig"]:<22} {r["hc"]:>6.2f} {r["dc"]:>6.2f} {r["vc"]:>6.2f} | '
              f'{r["oos_ann"]*100:>7.2f}% {r["oos_dd"]*100:>7.2f}% {r["oos_cal"]:>7.2f} | '
              f'{r["is_ann"]*100:>7.2f}% {r["is_dd"]*100:>7.2f}% {r["is_cal"]:>7.2f} {r["g"]:>6.2f}')

    # Pareto前沿
    print("\n" + "=" * 130)
    print("Phase 4: Pareto前沿筛选 (双窗口收益 vs MaxDD)")
    print("=" * 130)

    # 必须满足 -15% 硬约束 (max_dd >= -0.15)
    HARD_CAP = -0.15
    compliant = [r for r in all_results if r["oos_dd"] >= HARD_CAP and r["is_dd"] >= HARD_CAP]
    print(f"\n合规组合 (MaxDD_OOS>=-15% AND MaxDD_IS>=-15%): {len(compliant)}/{len(all_results)}")

    if len(compliant) < 5:
        print("  合规组合太少, 放宽到 -20% 硬约束:")
        HARD_CAP = -0.20
        compliant = [r for r in all_results if r["oos_dd"] >= HARD_CAP and r["is_dd"] >= HARD_CAP]
        print(f"  放宽后合规组合: {len(compliant)}/{len(all_results)}")

    frontier = pareto_frontier(compliant)
    frontier.sort(key=lambda x: x["g"], reverse=True)

    print(f"\nPareto前沿点 ({len(frontier)}个, 按G排序):")
    print(f'{"#":<3} {"信号":<22} {"hc":>6} {"dc":>6} {"vc":>6} | '
          f'{"OOS年化":>8} {"OOSDD":>8} {"OOSCal":>7} | '
          f'{"IS年化":>8} {"ISDD":>8} {"ISCal":>7} {"G":>6} {"avgRet":>7}')
    print("-" * 130)
    for i, r in enumerate(frontier[:15], 1):
        print(f'{i:<3} {r["sig"]:<22} {r["hc"]:>6.2f} {r["dc"]:>6.2f} {r["vc"]:>6.2f} | '
              f'{r["oos_ann"]*100:>7.2f}% {r["oos_dd"]*100:>7.2f}% {r["oos_cal"]:>7.2f} | '
              f'{r["is_ann"]*100:>7.2f}% {r["is_dd"]*100:>7.2f}% {r["is_cal"]:>7.2f} {r["g"]:>6.2f} '
              f'{r["avg_ret"]*100:>6.2f}%')

    # 保存结果到markdown
    md = ["# Pareto前沿搜索结果\n",
          f"> 运行日期: 2026-08-27\n",
          f"> 测试脚本: scripts/cmf/test_pareto_frontier.py\n",
          f"> 测试窗口: OOS 2010-2019 / IS 2020-2026\n\n",
          "## 信号候选 (3个)\n\n",
          "1. `macro_regime` (生产基线)\n",
          "2. `real_rate_tips_ma` (单因子最强候选)\n",
          "3. `momentum_long_only` (动量辅助候选)\n\n",
          "## 粗扫结果 (27 风控参数 × 3 信号 = 81 组合)\n\n",
          "| 信号 | hc | dc | vc | OOS年化 | OOSDD | OOSCal | IS年化 | ISDD | ISCal | G |\n",
          "|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for r in coarse_sorted[:10]:
        md.append(f"| {r['sig']} | {r['hc']:.2f} | {r['dc']:.2f} | {r['vc']:.2f} | "
                  f"{r['oos_ann']*100:.2f}% | {r['oos_dd']*100:.2f}% | {r['oos_cal']:.2f} | "
                  f"{r['is_ann']*100:.2f}% | {r['is_dd']*100:.2f}% | {r['is_cal']:.2f} | {r['g']:.2f} |\n")

    md.append(f"\n## 精扫结果 (Top3 ±20%扰动 = 81 组合)\n\n")
    md.append("| 信号 | hc | dc | vc | OOS年化 | OOSDD | OOSCal | IS年化 | ISDD | ISCal | G |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in fine_sorted[:10]:
        md.append(f"| {r['sig']} | {r['hc']:.2f} | {r['dc']:.2f} | {r['vc']:.2f} | "
                  f"{r['oos_ann']*100:.2f}% | {r['oos_dd']*100:.2f}% | {r['oos_cal']:.2f} | "
                  f"{r['is_ann']*100:.2f}% | {r['is_dd']*100:.2f}% | {r['is_cal']:.2f} | {r['g']:.2f} |\n")

    md.append(f"\n## Pareto前沿 ({len(frontier)}个合规点)\n\n")
    md.append(f"> 硬约束: MaxDD_OOS>=-15%, MaxDD_IS>=-15%\n\n")
    md.append("| # | 信号 | hc | dc | vc | OOS年化 | OOSDD | OOSCal | IS年化 | ISDD | ISCal | G | avgRet |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(frontier[:20], 1):
        md.append(f"| {i} | {r['sig']} | {r['hc']:.2f} | {r['dc']:.2f} | {r['vc']:.2f} | "
                  f"{r['oos_ann']*100:.2f}% | {r['oos_dd']*100:.2f}% | {r['oos_cal']:.2f} | "
                  f"{r['is_ann']*100:.2f}% | {r['is_dd']*100:.2f}% | {r['is_cal']:.2f} | {r['g']:.2f} | "
                  f"{r['avg_ret']*100:.2f}% |\n")

    md_path = Path("docs/gold/pareto_frontier_results.md")
    md_path.write_text("".join(md))
    print(f"\n结果已写入: {md_path}")


if __name__ == "__main__":
    main()