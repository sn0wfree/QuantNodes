"""scripts/cmf/test_adaptive_ensemble.py — 自适应集成策略评估.

对compute_adaptive_ensemble (macro + real_rate + momentum 多信号集成)
与单信号Pareto候选对比.

评估: 双窗口 + 子周期 + 滚动 + 综合分
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_macro_regime_signal,
    compute_risk_overlay,
    compute_trend_bnh_signal,
    compute_adaptive_ensemble,
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

SUBPERIODS = [
    ("10-14", "2010-01-01", "2014-12-31"),
    ("15-19", "2015-01-01", "2019-12-31"),
    ("20-23", "2020-01-01", "2023-12-31"),
    ("24-26", "2024-01-01", "2026-08-25"),
]

ROLLING_FOLDS = [
    ("F1", "2010-01-01", "2013-12-31", "2014-01-01", "2015-12-31"),
    ("F2", "2012-01-01", "2015-12-31", "2016-01-01", "2017-12-31"),
    ("F3", "2014-01-01", "2017-12-31", "2018-01-01", "2019-12-31"),
    ("F4", "2016-01-01", "2019-12-31", "2020-01-01", "2021-12-31"),
    ("F5", "2018-01-01", "2021-12-31", "2022-01-01", "2023-12-31"),
    ("F6", "2020-01-01", "2023-12-31", "2024-01-01", "2026-08-25"),
]


def load_close():
    import duckdb
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return (df.drop_duplicates("trade_date", keep="last")
              .set_index("trade_date")["close"])


def evaluate(pos, close, s, e, freq="D"):
    periods_per_y = 52 if freq == "W" else 252
    sub = pos[(pos.index >= s) & (pos.index <= e)]
    c = close.reindex(sub.index)
    if len(c) < 2:
        return {"ann_ret": 0, "sharpe": 0, "max_dd": 0, "calmar": 0}
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * COST_BPS / 10000
    sr = (sr - cost).dropna()
    sp = c.iloc[0]
    nav = sp * (1 + sr).cumprod()
    n_y = len(c) / periods_per_y
    ar = (nav.iloc[-1] / sp) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(periods_per_y)
    dd = (nav / nav.cummax() - 1).min()
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "calmar": ar / abs(dd) if dd != 0 else 0}


def full_eval(pos, close):
    out = {}
    oos = evaluate(pos, close, OOS_S, OOS_E)
    is_ = evaluate(pos, close, IS_S, IS_E)
    out["OOS"] = oos
    out["IS"] = is_
    out["G"] = (oos["calmar"] * is_["calmar"]) ** 0.5
    out["MinCalmar"] = min(oos["calmar"], is_["calmar"])
    out["WorstDD"] = min(oos["max_dd"], is_["max_dd"])
    out["AvgRet"] = (oos["ann_ret"] + is_["ann_ret"]) / 2
    out["Subperiods"] = {n: evaluate(pos, close, s, e) for n, s, e in SUBPERIODS}
    out["Rolling"] = {}
    for fn, ts_, te_, vs_, ve_ in ROLLING_FOLDS:
        out["Rolling"][fn] = {
            "test": evaluate(pos, close, vs_, ve_),
            "train": evaluate(pos, close, ts_, te_),
        }
    return out


def main():
    close = load_close()
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()
    tips = _load_indicator_d("tips10y", "fred_dfii10").reindex(close.index).ffill()

    # 获取原始信号
    macro_pos = compute_macro_regime_signal()["position"]
    rr_pos = compute_real_rate_signal(variant="tips_ma")["position"]
    mom_pos = compute_momentum_signal(variant="long_only")["position"]
    tb_pos = compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")["position"]

    # 各信号 + 风控层 (hc=-0.10 严苛)
    def apply_overlay(pos_in):
        return compute_risk_overlay(
            close=close, position=pos_in, vix=vix, gvz=gvz,
            hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
            cooldown_days=5,
        )

    candidates = {
        "Buy & Hold": pd.Series(1.0, index=close.index),
        "macro 无风控": macro_pos,
        "macro + 风控 hc=-0.10": apply_overlay(macro_pos),
        "real_rate + 风控 hc=-0.10": apply_overlay(rr_pos),
        "momentum + 风控 hc=-0.10": apply_overlay(mom_pos),

        # ===== 自适应集成候选 =====
        "集成 fixed (0.4/0.3/0.3) + 风控": compute_adaptive_ensemble(
            macro_weight=0.4, real_rate_weight=0.3, momentum_weight=0.3,
            weight_method="fixed", use_risk_overlay=True,
            hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        )["position"],
        "集成 fixed (0.5/0.25/0.25) + 风控": compute_adaptive_ensemble(
            macro_weight=0.5, real_rate_weight=0.25, momentum_weight=0.25,
            weight_method="fixed", use_risk_overlay=True,
            hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        )["position"],
        "集成 fixed (1/3 等权) + 风控": compute_adaptive_ensemble(
            macro_weight=1/3, real_rate_weight=1/3, momentum_weight=1/3,
            weight_method="fixed", use_risk_overlay=True,
            hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        )["position"],
        "集成 adaptive (lb=126) + 风控": compute_adaptive_ensemble(
            weight_method="adaptive", adaptive_lookback=126, adaptive_temp=1.0,
            use_risk_overlay=True,
            hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        )["position"],
        "集成 adaptive (lb=252) + 风控": compute_adaptive_ensemble(
            weight_method="adaptive", adaptive_lookback=252, adaptive_temp=1.0,
            use_risk_overlay=True,
            hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        )["position"],
        "集成 adaptive (lb=126, temp=2.0) + 风控": compute_adaptive_ensemble(
            weight_method="adaptive", adaptive_lookback=126, adaptive_temp=2.0,
            use_risk_overlay=True,
            hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15,
        )["position"],
        "集成 adaptive + 放宽 hc=-0.15": compute_adaptive_ensemble(
            weight_method="adaptive", adaptive_lookback=126,
            use_risk_overlay=True,
            hard_cap=-0.15, soft_cap=-0.12, daily_cap=-0.03, vol_cap=0.20,
        )["position"],
    }

    print("=" * 130)
    print("评估候选 (双窗口 + 子周期 + 滚动)")
    print("=" * 130)

    results = {n: full_eval(pos, close) for n, pos in candidates.items()}

    # ===表1: 双窗口核心指标===
    print("\n" + "=" * 130)
    print("表1: 双窗口核心指标")
    print("=" * 130)
    print(f'{"策略":<35} | {"OOS年化":>8} {"OOSDD":>8} {"OOSCal":>7} | '
          f'{"IS年化":>8} {"ISDD":>8} {"ISCal":>7} | {"G":>5} {"MinCal":>7}')
    print("-" * 130)
    for name, ev in results.items():
        oos, is_ = ev["OOS"], ev["IS"]
        print(f'{name:<35} | '
              f'{oos["ann_ret"]*100:>7.2f}% {oos["max_dd"]*100:>7.2f}% {oos["calmar"]:>7.2f} | '
              f'{is_["ann_ret"]*100:>7.2f}% {is_["max_dd"]*100:>7.2f}% {is_["calmar"]:>7.2f} | '
              f'{ev["G"]:>5.2f} {ev["MinCalmar"]:>7.2f}')

    # ===表2: 子周期===
    print("\n" + "=" * 130)
    print("表2: 子周期稳定性 (年化/MaxDD)")
    print("=" * 130)
    print(f'{"策略":<35} | {"10-14":>16} | {"15-19":>16} | {"20-23":>16} | {"24-26":>16}')
    print(f'{"":<35} | {"年化":>7}{"MaxDD":>8} | {"年化":>7}{"MaxDD":>8} | '
          f'{"年化":>7}{"MaxDD":>8} | {"年化":>7}{"MaxDD":>8}')
    print("-" * 130)
    for name, ev in results.items():
        row = f'{name:<35} | '
        for sp_name, _, _ in SUBPERIODS:
            m = ev["Subperiods"][sp_name]
            row += f'{m["ann_ret"]*100:>6.2f}%{m["max_dd"]*100:>7.2f}% | '
        print(row)

    # ===表3: 滚动===
    print("\n" + "=" * 130)
    print("表3: 滚动验证 (test期Calmar, 6折)")
    print("=" * 130)
    print(f'{"策略":<35} | {"F1":>6} {"F2":>6} {"F3":>6} {"F4":>6} {"F5":>6} {"F6":>6} | {"avg":>6}')
    print("-" * 130)
    for name, ev in results.items():
        row = f'{name:<35} | '
        cals = []
        for fn, _, _, _, _ in ROLLING_FOLDS:
            cal = ev["Rolling"][fn]["test"]["calmar"]
            row += f'{cal:>6.2f} '
            cals.append(cal)
        row += f'| {np.mean(cals):>6.2f}'
        print(row)

    # ===综合评分===
    print("\n" + "=" * 130)
    print("综合评分排序")
    print("=" * 130)
    scores = []
    for name, ev in results.items():
        score = ev["G"] * 2 + ev["MinCalmar"] * 1.5
        oos_dd = ev["OOS"]["max_dd"]
        is_dd = ev["IS"]["max_dd"]
        if oos_dd >= -0.20 and is_dd >= -0.15:
            score += 0.5
        if oos_dd >= -0.15 and is_dd >= -0.15:
            score += 1.0
        scores.append((name, score, ev["G"], ev["MinCalmar"],
                       oos_dd, is_dd, ev["AvgRet"]))
    scores.sort(key=lambda x: -x[1])
    print(f'{"#":<3} {"策略":<35} {"综合分":>7} {"G":>5} {"MinCal":>7} '
          f'{"OOSDD":>8} {"ISDD":>8} {"avgRet":>8}')
    print("-" * 130)
    for i, (n, s, g, mc, odd, idd, ar) in enumerate(scores, 1):
        print(f'{i:<3} {n:<35} {s:>7.2f} {g:>5.2f} {mc:>7.2f} '
              f'{odd*100:>7.2f}% {idd*100:>7.2f}% {ar*100:>7.2f}%')

    # ===最终推荐===
    best = scores[0]
    print("\n" + "=" * 130)
    print(f"最终推荐: {best[0]}")
    print("=" * 130)
    print(f'综合分={best[1]:.2f}, G={best[2]:.2f}, MinCal={best[3]:.2f}')
    print(f'OOS年化/MaxDD/Calmar={results[best[0]]["OOS"]["ann_ret"]*100:.2f}%/{results[best[0]]["OOS"]["max_dd"]*100:.2f}%/{results[best[0]]["OOS"]["calmar"]:.2f}')
    print(f'IS年化/MaxDD/Calmar={results[best[0]]["IS"]["ann_ret"]*100:.2f}%/{results[best[0]]["IS"]["max_dd"]*100:.2f}%/{results[best[0]]["IS"]["calmar"]:.2f}')

    # 保存到markdown
    md = ["# 自适应集成策略评估\n\n",
          "> 运行日期: 2026-08-27\n",
          "> 测试脚本: scripts/cmf/test_adaptive_ensemble.py\n",
          "> 测试窗口: OOS 2010-2019 / IS 2020-2026\n\n",
          "## 候选策略\n\n",
          "**单信号基线** (Pareto前沿已验证):\n",
          "- `macro_regime` (生产基线, MaxDD=-29%)\n",
          "- `real_rate_tips_ma` (OOS最优, MaxDD=-13%)\n",
          "- `momentum_long_only` (G最优, MaxDD=-25%)\n\n",
          "**集成候选**:\n",
          "- `fixed (0.4/0.3/0.3)`: 固定权重\n",
          "- `fixed (0.5/0.25/0.25)`: macro权重更高\n",
          "- `fixed (1/3 等权)`: 等权\n",
          "- `adaptive (lb=126)`: 滚动Calmar动态权重\n",
          "- `adaptive (lb=252)`: 较长lookback\n",
          "- `adaptive (lb=126, temp=2.0)`: 较强自适应\n",
          "- `adaptive + 放宽 hc=-0.15`: 放宽硬约束\n\n",
          "## 双窗口核心指标\n\n",
          "| 策略 | OOS年化 | OOSDD | OOSCal | IS年化 | ISDD | ISCal | G | MinCalmar |\n",
          "|---|---|---|---|---|---|---|---|---|\n"]
    for n, ev in results.items():
        oos, is_ = ev["OOS"], ev["IS"]
        md.append(f"| {n} | {oos['ann_ret']*100:.2f}% | {oos['max_dd']*100:.2f}% | "
                  f"{oos['calmar']:.2f} | {is_['ann_ret']*100:.2f}% | "
                  f"{is_['max_dd']*100:.2f}% | {is_['calmar']:.2f} | "
                  f"{ev['G']:.2f} | {ev['MinCalmar']:.2f} |\n")

    md.append("\n## 子周期稳定性\n\n")
    md.append("| 策略 | 10-14 | 15-19 | 20-23 | 24-26 |\n")
    md.append("|---|---|---|---|---|\n")
    for n, ev in results.items():
        row = f"| {n} | "
        for sp_name, _, _ in SUBPERIODS:
            m = ev["Subperiods"][sp_name]
            row += f"{m['ann_ret']*100:.2f}%/{m['max_dd']*100:.2f}% | "
        md.append(row + "\n")

    md.append("\n## 滚动验证 (test期Calmar)\n\n")
    md.append("| 策略 | F1 | F2 | F3 | F4 | F5 | F6 | avg |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for n, ev in results.items():
        row = f"| {n} | "
        cals = []
        for fn, _, _, _, _ in ROLLING_FOLDS:
            cal = ev["Rolling"][fn]["test"]["calmar"]
            row += f"{cal:.2f} | "
            cals.append(cal)
        row += f"{np.mean(cals):.2f} |\n"
        md.append(row)

    md.append("\n## 综合评分\n\n")
    md.append(" = G×2 + MinCalmar×1.5 + MaxDD硬约束奖励\n\n")
    md.append("| # | 策略 | 综合分 | G | MinCalmar |\n")
    md.append("|---|---|---|---|---|\n")
    for i, (n, s, g, mc, _, _, _) in enumerate(scores, 1):
        md.append(f"| {i} | {n} | {s:.2f} | {g:.2f} | {mc:.2f} |\n")

    md.append(f"\n## 最终推荐: **{best[0]}**\n\n")
    md.append(f"- 综合分: {best[1]:.2f}, G: {best[2]:.2f}, MinCal: {best[3]:.2f}\n")
    md.append(f"- OOS: {results[best[0]]['OOS']['ann_ret']*100:.2f}%/{results[best[0]]['OOS']['max_dd']*100:.2f}%/{results[best[0]]['OOS']['calmar']:.2f}\n")
    md.append(f"- IS: {results[best[0]]['IS']['ann_ret']*100:.2f}%/{results[best[0]]['IS']['max_dd']*100:.2f}%/{results[best[0]]['IS']['calmar']:.2f}\n")

    md_path = Path("docs/gold/adaptive_ensemble_results.md")
    md_path.write_text("".join(md))
    print(f"\n结果已写入: {md_path}")


if __name__ == "__main__":
    main()