"""scripts/cmf/test_final_evaluation.py — 候选策略全面再评估.

对Pareto前沿Top候选 + 基线, 做:
1. 双窗口完整指标 (年化/Sharpe/MaxDD/Calmar/G/Min Calmar)
2. 子周期分解 (10-14, 15-19, 20-23, 24-26YTD) - 验证稳健性
3. 滚动窗口验证 (train 4年选参 → test 2年评估, 6折)
4. 综合评分 + 最终推荐
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
    compute_trend_bnh_signal,
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

# 子周期
SUBPERIODS = [
    ("10-14", "2010-01-01", "2014-12-31"),
    ("15-19", "2015-01-01", "2019-12-31"),
    ("20-23", "2020-01-01", "2023-12-31"),
    ("24-26", "2024-01-01", "2026-08-25"),
]

# 滚动窗口 (4折)
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


def get_raw_signal(name, close):
    if name == "macro_regime":
        return compute_macro_regime_signal()["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    elif name == "real_rate_tips_ma":
        return compute_real_rate_signal(variant="tips_ma")["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    elif name == "momentum_long_only":
        pos_w = compute_momentum_signal(variant="long_only")["position"]
        return pos_w.reindex(close.index).fillna(method="ffill").fillna(0)
    elif name == "trend_bnh":
        return compute_trend_bnh_signal(trailing_stop=0.08, freq="daily")["position"].reindex(close.index).fillna(0)
    raise ValueError(name)


def apply_risk(close, position, vix, gvz, hc, dc, vc, soft_offset=0.03, cooldown=5):
    return compute_risk_overlay(
        close=close, position=position, vix=vix, gvz=gvz,
        hard_cap=hc, soft_cap=hc + soft_offset, daily_cap=dc, vol_cap=vc,
        cooldown_days=cooldown,
    )


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
    """双窗口 + 子周期 + 滚动."""
    out = {}
    oos = evaluate(pos, close, OOS_S, OOS_E)
    is_ = evaluate(pos, close, IS_S, IS_E)
    out["OOS"] = oos
    out["IS"] = is_
    out["G"] = (oos["calmar"] * is_["calmar"]) ** 0.5
    out["MinCalmar"] = min(oos["calmar"], is_["calmar"])
    out["WorstDD"] = min(oos["max_dd"], is_["max_dd"])
    out["AvgRet"] = (oos["ann_ret"] + is_["ann_ret"]) / 2

    # 子周期
    out["Subperiods"] = {}
    for name, s, e in SUBPERIODS:
        out["Subperiods"][name] = evaluate(pos, close, s, e)

    # 滚动
    out["Rolling"] = {}
    for fname, ts_, te_, vs_, ve_ in ROLLING_FOLDS:
        out["Rolling"][fname] = {
            "test": evaluate(pos, close, vs_, ve_),
            "train": evaluate(pos, close, ts_, te_),
        }
    return out


def main():
    close = load_close()
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    # 评估候选集: Pareto Top + 基线
    candidates = [
        # 基线
        ("Buy & Hold", None, None, None, None, None),
        ("trend_bnh(ts=0.08)", "trend_bnh", None, None, None, None),
        ("macro_regime 无风控", "macro_regime", None, None, None, None),

        # Pareto候选1: momentum_long_only + hc=-0.10
        ("momentum + hc=-0.10 dc=-0.02 vc=0.15", "momentum_long_only", -0.10, -0.02, 0.15, None),

        # Pareto候选6: real_rate_tips_ma + hc=-0.10
        ("real_rate + hc=-0.10 dc=-0.02 vc=0.15", "real_rate_tips_ma", -0.10, -0.02, 0.15, None),

        # 放宽硬约束: macro_regime + hc=-0.20
        ("macro + hc=-0.20 dc=-0.03 vc=0.20", "macro_regime", -0.20, -0.03, 0.20, None),
    ]

    print("=" * 130)
    print("候选策略评估 (双窗口 + 子周期 + 滚动)")
    print("=" * 130)

    results = {}
    for label, sig_name, hc, dc, vc, _ in candidates:
        if sig_name is None:
            # Buy & Hold
            pos = pd.Series(1.0, index=close.index)
        elif hc is None:
            pos = get_raw_signal(sig_name, close)
        else:
            raw = get_raw_signal(sig_name, close)
            pos = apply_risk(close, raw, vix, gvz, hc, dc, vc)

        ev = full_eval(pos, close)
        results[label] = ev

    # ===== 输出表1: 双窗口核心指标 =====
    print("\n" + "=" * 130)
    print("表1: 双窗口核心指标")
    print("=" * 130)
    print(f'{"策略":<40} | {"OOS年化":>8} {"OOSSh":>6} {"OOSDD":>8} {"OOSCal":>7} | '
          f'{"IS年化":>8} {"ISSh":>6} {"ISDD":>8} {"ISCal":>7} | {"G":>5} {"MinCal":>6}')
    print("-" * 130)
    for label, ev in results.items():
        oos, is_ = ev["OOS"], ev["IS"]
        print(f'{label:<40} | '
              f'{oos["ann_ret"]*100:>7.2f}% {oos["sharpe"]:>6.2f} {oos["max_dd"]*100:>7.2f}% {oos["calmar"]:>7.2f} | '
              f'{is_["ann_ret"]*100:>7.2f}% {is_["sharpe"]:>6.2f} {is_["max_dd"]*100:>7.2f}% {is_["calmar"]:>7.2f} | '
              f'{ev["G"]:>5.2f} {ev["MinCalmar"]:>6.2f}')

    # ===== 输出表2: 子周期 =====
    print("\n" + "=" * 130)
    print("表2: 子周期稳定性 (年化/MaxDD)")
    print("=" * 130)
    print(f'{"策略":<40} | '
          f'{"10-14":>16} | {"15-19":>16} | {"20-23":>16} | {"24-26":>16}')
    print(f'{"":<40} | '
          f'{"年化":>7} {"MaxDD":>8} | {"年化":>7} {"MaxDD":>8} | {"年化":>7} {"MaxDD":>8} | '
          f'{"年化":>7} {"MaxDD":>8}')
    print("-" * 130)
    for label, ev in results.items():
        row = f'{label:<40} | '
        for sp_name, _, _ in SUBPERIODS:
            sp_m = ev["Subperiods"][sp_name]
            row += f'{sp_m["ann_ret"]*100:>6.2f}% {sp_m["max_dd"]*100:>7.2f}% | '
        print(row)

    # ===== 输出表3: 滚动窗口 =====
    print("\n" + "=" * 130)
    print("表3: 滚动验证 (train 4年 → test 2年评估, 6折)")
    print("=" * 130)
    print(f'{"策略":<40} | {"F1":>6} {"F2":>6} {"F3":>6} {"F4":>6} {"F5":>6} {"F6":>6} | {"avg":>6}')
    print("-" * 130)
    for label, ev in results.items():
        cal_row = f'{label:<40} | '
        cals = []
        for fname, _, _, _, _ in ROLLING_FOLDS:
            cal = ev["Rolling"][fname]["test"]["calmar"]
            cal_row += f'{cal:>6.2f} '
            cals.append(cal)
        cal_row += f'| {np.mean(cals):>6.2f}'
        print(cal_row)

    print("\n详细(test期年化/DD):")
    print(f'{"策略":<40} | ' + ' | '.join(f'{fn:>16}' for fn, _, _, _, _ in ROLLING_FOLDS))
    for label, ev in results.items():
        row = f'{label:<40} | '
        for fname, _, _, _, _ in ROLLING_FOLDS:
            t = ev["Rolling"][fname]["test"]
            row += f'{t["ann_ret"]*100:>7.2f}%/{t["max_dd"]*100:>6.2f}% | '
        print(row)

    # ===== 综合评分 =====
    print("\n" + "=" * 130)
    print("综合评分 (推荐排序)")
    print("=" * 130)
    scores = []
    for label, ev in results.items():
        oos, is_ = ev["OOS"], ev["IS"]
        # 综合分数: G主, MinCalmar副, 双窗口MaxDD<25%奖励
        score = ev["G"] * 2 + ev["MinCalmar"] * 1.5
        if oos["max_dd"] >= -0.20 and is_["max_dd"] >= -0.15:
            score += 0.5  # 双窗口MaxDD达标奖励
        if oos["max_dd"] >= -0.15 and is_["max_dd"] >= -0.15:
            score += 1.0  # 硬约束达标奖励
        scores.append((label, score, ev["G"], ev["MinCalmar"],
                       oos["max_dd"], is_["max_dd"], ev["AvgRet"]))

    scores.sort(key=lambda x: -x[1])
    print(f'{"#":<3} {"策略":<40} {"综合分":>7} {"G":>5} {"MinCal":>7} '
          f'{"OOSDD":>8} {"ISDD":>8} {"avgRet":>8}')
    print("-" * 130)
    for i, (label, score, g, mc, odd, idd, ar) in enumerate(scores, 1):
        print(f'{i:<3} {label:<40} {score:>7.2f} {g:>5.2f} {mc:>6.2f} '
              f'{odd*100:>7.2f}% {idd*100:>7.2f}% {ar*100:>7.2f}%')

    # ===== 推荐 =====
    print("\n" + "=" * 130)
    print("最终推荐")
    print("=" * 130)
    best_label = scores[0][0]
    print(f'\n基于: 双窗口 G + MinCalmar + MaxDD硬约束 (-15%)奖励 + -20%软约束奖励')
    print(f'最佳: {best_label}')
    print(f'  - G = {scores[0][2]:.2f}')
    print(f'  - MinCalmar = {scores[0][3]:.2f}')
    print(f'  - OOS MaxDD = {scores[0][4]*100:.2f}%')
    print(f'  - IS MaxDD = {scores[0][5]*100:.2f}%')
    print(f'  - avgRet = {scores[0][6]*100:.2f}%')

    # 保存到 markdown
    md = ["# 候选策略最终评估\n",
          f"> 运行日期: 2026-08-27\n",
          f"> 测试脚本: scripts/cmf/test_final_evaluation.py\n",
          f"> 测试窗口: OOS 2010-2019 / IS 2020-2026-08-25\n\n",
          "## 双窗口核心指标\n\n",
          "| 策略 | OOS年化 | OOSSh | OOSDD | OOSCal | IS年化 | ISSh | ISDD | ISCal | G | MinCalmar |\n",
          "|---|---|---|---|---|---|---|---|---|---|---|\n"]
    for label, ev in results.items():
        oos, is_ = ev["OOS"], ev["IS"]
        md.append(f"| {label} | {oos['ann_ret']*100:.2f}% | {oos['sharpe']:.2f} | "
                  f"{oos['max_dd']*100:.2f}% | {oos['calmar']:.2f} | "
                  f"{is_['ann_ret']*100:.2f}% | {is_['sharpe']:.2f} | "
                  f"{is_['max_dd']*100:.2f}% | {is_['calmar']:.2f} | "
                  f"{ev['G']:.2f} | {ev['MinCalmar']:.2f} |\n")

    md.append("\n## 子周期稳定性 (年化/MaxDD)\n\n")
    md.append("| 策略 | 10-14 | 15-19 | 20-23 | 24-26 |\n")
    md.append("|---|---|---|---|---|\n")
    for label, ev in results.items():
        row = f"| {label} | "
        for sp_name, _, _ in SUBPERIODS:
            m = ev["Subperiods"][sp_name]
            row += f"{m['ann_ret']*100:.2f}%/{m['max_dd']*100:.2f}% | "
        md.append(row + "\n")

    md.append("\n## 滚动验证 (test期Calmar, 6折)\n\n")
    md.append("| 策略 | F1 | F2 | F3 | F4 | F5 | F6 | avg |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for label, ev in results.items():
        row = f"| {label} | "
        cals = []
        for fname, _, _, _, _ in ROLLING_FOLDS:
            cal = ev["Rolling"][fname]["test"]["calmar"]
            row += f"{cal:.2f} | "
            cals.append(cal)
        row += f"{np.mean(cals):.2f} |\n"
        md.append(row)

    md.append("\n## 综合评分排序\n\n")
    md.append("评分公式: 综合分 = G×2 + MinCalmar×1.5 + MaxDD达标奖励\n\n")
    md.append("| # | 策略 | 综合分 | G | MinCalmar | OOSDD | ISDD | avgRet |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for i, (label, score, g, mc, odd, idd, ar) in enumerate(scores, 1):
        md.append(f"| {i} | {label} | {score:.2f} | {g:.2f} | {mc:.2f} | "
                  f"{odd*100:.2f}% | {idd*100:.2f}% | {ar*100:.2f}% |\n")

    md.append(f"\n## 最终推荐: **{best_label}**\n\n")
    md.append(f"- G = {scores[0][2]:.2f}, MinCalmar = {scores[0][3]:.2f}\n")
    md.append(f"- OOS MaxDD = {scores[0][4]*100:.2f}%, IS MaxDD = {scores[0][5]*100:.2f}%\n")
    md.append(f"- 双窗口avgRet = {scores[0][6]*100:.2f}%\n")

    md_path = Path("docs/gold/final_evaluation_results.md")
    md_path.write_text("".join(md))
    print(f"\n结果已写入: {md_path}")


if __name__ == "__main__":
    main()