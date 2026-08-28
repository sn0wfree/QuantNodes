"""scripts/cmf/test_ensemble_v2.py — 集成策略Round2评估 (5维度).

维度1: 分层叠加 (per-signal overlay → combine)
维度2: 投票制 (2-of-3)
维度3: 换手率与成本
维度4: 信号相关性
维度5: 年度明细
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
HC_DEFAULT = -0.10
DC_DEFAULT = -0.02
VC_DEFAULT = 0.15

SUBPERIODS = [
    ("10-14", "2010-01-01", "2014-12-31"),
    ("15-19", "2015-01-01", "2019-12-31"),
    ("20-23", "2020-01-01", "2023-12-31"),
    ("24-26", "2024-01-01", "2026-08-25"),
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


def evaluate(pos, close, s, e):
    periods_per_y = 252
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
    turnover = sub.fillna(0).diff().abs().sum() / 2 / n_y
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "calmar": ar / abs(dd) if dd != 0 else 0,
            "turnover": turnover}


def full_eval(pos, close):
    oos = evaluate(pos, close, OOS_S, OOS_E)
    is_ = evaluate(pos, close, IS_S, IS_E)
    sub = {n: evaluate(pos, close, s, e) for n, s, e in SUBPERIODS}
    g = (oos["calmar"] * is_["calmar"]) ** 0.5
    return {"OOS": oos, "IS": is_, "G": g, "MinCal": min(oos["calmar"], is_["calmar"]),
            "Sub": sub}


def apply_overlay(pos, close, vix, gvz):
    return compute_risk_overlay(
        close=close, position=pos, vix=vix, gvz=gvz,
        hard_cap=HC_DEFAULT, soft_cap=HC_DEFAULT + 0.03,
        daily_cap=DC_DEFAULT, vol_cap=VC_DEFAULT, cooldown_days=5,
    )


def main():
    close = load_close()
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    macro_raw = compute_macro_regime_signal()["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    rr_raw = compute_real_rate_signal(variant="tips_ma")["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    mom_raw = compute_momentum_signal(variant="long_only")["position"].reindex(close.index).fillna(method="ffill").fillna(0)

    signals = {"macro": macro_raw, "real_rate": rr_raw, "momentum": mom_raw}

    # ================================================================
    # 维度4: 信号相关性分析 (先做, 理解机制)
    # ================================================================
    print("=" * 120)
    print("维度4: 信号相关性")
    print("=" * 120)
    sig_df = pd.DataFrame(signals)
    print(f"\n两两Pearson相关性 (全样本):")
    print(sig_df.corr().to_string())

    print(f"\n信号统计:")
    for n, s in signals.items():
        print(f"  {n:<15}: mean={s.mean():.3f}, std={s.std():.3f}, "
              f"=1占比={((s > 0.95) & (s < 1.05)).mean()*100:.1f}%, "
              f"=0占比={(s < 0.05).mean()*100:.1f}%, "
              f"non-二值占比={((s > 0.05) & (s < 0.95)).mean()*100:.1f}%")

    # ================================================================
    # 构建候选
    # ================================================================
    candidates = {}

    # 基线
    candidates["real_rate+风控 (冠军)"] = apply_overlay(rr_raw, close, vix, gvz)
    candidates["momentum+风控"] = apply_overlay(mom_raw, close, vix, gvz)
    candidates["macro+风控"] = apply_overlay(macro_raw, close, vix, gvz)

    # 维度1: 分层叠加 (先各自风控, 再平均)
    rr_ov = apply_overlay(rr_raw, close, vix, gvz)
    mom_ov = apply_overlay(mom_raw, close, vix, gvz)
    macro_ov = apply_overlay(macro_raw, close, vix, gvz)

    candidates["分层 0.4/0.3/0.3"] = (0.4 * macro_ov + 0.3 * rr_ov + 0.3 * mom_ov).clip(0, 1.2)
    candidates["分层 1/3等权"] = ((macro_ov + rr_ov + mom_ov) / 3).clip(0, 1.2)
    candidates["分层 0.5/0.25/0.25"] = (0.5 * macro_ov + 0.25 * rr_ov + 0.25 * mom_ov).clip(0, 1.2)
    candidates["分层 real+mom等权"] = ((rr_ov + mom_ov) / 2).clip(0, 1.2)

    # 维度2: 投票制 (2-of-3 majority)
    # 每个信号 0/1 后投票
    macro_bin = (macro_raw > 0.5).astype(float)
    rr_bin = (rr_raw > 0.5).astype(float)
    mom_bin = (mom_raw > 0.5).astype(float)
    vote_2of3 = ((macro_bin + rr_bin + mom_bin) >= 2).astype(float)
    candidates["投票 2-of-3 + 风控"] = apply_overlay(vote_2of3, close, vix, gvz)

    # 投票 3-of-3 (全部一致才持仓)
    vote_3of3 = ((macro_bin + rr_bin + mom_bin) >= 3).astype(float)
    candidates["投票 3-of-3 + 风控"] = apply_overlay(vote_3of3, close, vix, gvz)

    # 投票 real_rate为主 + macro/rr/mom三票制 (real_rate权重2票)
    vote_rr2 = ((rr_bin * 2 + macro_bin + mom_bin) >= 2).astype(float)
    candidates["投票 rr2票+风R"] = apply_overlay(vote_rr2, close, vix, gvz)

    # ================================================================
    # 评估
    # ================================================================
    results = {n: full_eval(pos, close) for n, pos in candidates.items()}

    # 表1: 双窗口核心指标
    print("\n" + "=" * 120)
    print("表1: 双窗口核心指标")
    print("=" * 120)
    print(f'{"策略":<28} | {"OOS年化":>8} {"OOSDD":>8} {"OOSCal":>7} | '
          f'{"IS年化":>8} {"ISDD":>8} {"ISCal":>7} | {"G":>5} {"MinCal":>6}')
    print("-" * 120)
    for n, ev in results.items():
        o, i = ev["OOS"], ev["IS"]
        print(f'{n:<28} | {o["ann_ret"]*100:>7.2f}% {o["max_dd"]*100:>7.2f}% {o["calmar"]:>7.2f} | '
              f'{i["ann_ret"]*100:>7.2f}% {i["max_dd"]*100:>7.2f}% {i["calmar"]:>7.2f} | '
              f'{ev["G"]:>5.2f} {ev["MinCal"]:>6.2f}')

    # 表2: 子周期
    print("\n" + "=" * 120)
    print("表2: 子周期 (年化/MaxDD)")
    print("=" * 120)
    print(f'{"策略":<28} | '
          f'{"10-14":>16} | {"15-19":>16} | {"20-23":>16} | {"24-26":>16}')
    print("-" * 120)
    for n, ev in results.items():
        row = f'{n:<28} | '
        for sp in ["10-14", "15-19", "20-23", "24-26"]:
            m = ev["Sub"][sp]
            row += f'{m["ann_ret"]*100:>6.1f}%/{m["max_dd"]*100:>6.1f}% | '
        print(row)

    # 维度3: 换手率
    print("\n" + "=" * 120)
    print("维度3: 换手率与成本")
    print("=" * 120)
    print(f'{"策略":<28} | {"OOS换手":>9} {"IS换手":>9} | {"OOS成本":>8} {"IS成本":>8}')
    print("-" * 120)
    for n, ev in results.items():
        o, i = ev["OOS"], ev["IS"]
        o_cost = o["turnover"] * COST_BPS / 10000 * 252
        i_cost = i["turnover"] * COST_BPS / 10000 * 252
        print(f'{n:<28} | {o["turnover"]:>8.1f}x {i["turnover"]:>8.1f}x | '
              f'{o_cost*100:>7.2f}% {i_cost*100:>7.2f}%')

    # 维度5: 综合评分
    print("\n" + "=" * 120)
    print("维度5: 综合评分排序")
    print("=" * 120)
    scores = []
    for n, ev in results.items():
        o, i = ev["OOS"], ev["IS"]
        score = ev["G"] * 2 + ev["MinCal"] * 1.5
        if o["max_dd"] >= -0.15 and i["max_dd"] >= -0.15:
            score += 1.0
        elif o["max_dd"] >= -0.20 and i["max_dd"] >= -0.15:
            score += 0.5
        scores.append((n, score, ev["G"], ev["MinCal"],
                       o["max_dd"], i["max_dd"], ev["OOS"]["ann_ret"], ev["IS"]["ann_ret"]))
    scores.sort(key=lambda x: -x[1])
    print(f'{"#":<3} {"策略":<28} {"综合分":>7} {"G":>5} {"MinCal":>7} '
          f'{"OOSDD":>8} {"ISDD":>8} {"OOSret":>8} {"ISret":>8}')
    print("-" * 120)
    for i, (n, s, g, mc, odd, idd, or_, ir_) in enumerate(scores, 1):
        print(f'{i:<3} {n:<28} {s:>7.2f} {g:>5.2f} {mc:>7.2f} '
              f'{odd*100:>7.2f}% {idd*100:>7.2f}% {or_*100:>7.2f}% {ir_*100:>7.2f}%')

    # 推荐
    best = scores[0]
    print(f"\n最佳: {best[0]} (综合分={best[1]:.2f})")

    # ================================================================
    # 保存到 markdown
    # ================================================================
    md = ["# 集成策略 Round2 评估\n\n",
          "> 运行日期: 2026-08-27\n",
          "> 测试脚本: scripts/cmf/test_ensemble_v2.py\n\n",
          "## 维度1-5汇总\n\n",
          "### 维度1: 分层叠加 (先各自风控 → 再平均)\n\n",
          "思路: 每个信号先独立过compute_risk_overlay(仍是0/1), 再加权平均.\n",
          "结果: 离散仓位 {0, 0.33, 0.67, 1.0}, 每个信号独立满足约束.\n\n",
          "### 维度2: 投票制 (2-of-3)\n\n",
          "思路: 3个信号中至少2个=1时持仓.\n\n",
          "### 维度3: 换手率\n\n",
          "| 策略 | OOS换手 | IS换手 |\n",
          "|---|---|---|\n"]
    for n, ev in results.items():
        md.append(f"| {n} | {ev['OOS']['turnover']:.1f}x | {ev['IS']['turnover']:.1f}x |\n")

    md.append("\n### 维度4: 信号相关性\n\n")
    md.append("| | macro | real_rate | momentum |\n|---|---|---|---|\n")
    corr = sig_df.corr()
    for r in ["macro", "real_rate", "momentum"]:
        row = f"| {r} | "
        for c in ["macro", "real_rate", "momentum"]:
            row += f"{corr.loc[r, c]:.3f} | "
        md.append(row + "\n")

    md.append("\n### 维度5: 综合评分\n\n")
    md.append("| # | 策略 | 综合分 | G | MinCal | OOSDD | ISDD |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for i, (n, s, g, mc, odd, idd, _, _) in enumerate(scores, 1):
        md.append(f"| {i} | {n} | {s:.2f} | {g:.2f} | {mc:.2f} | "
                  f"{odd*100:.2f}% | {idd*100:.2f}% |\n")

    md.append(f"\n## 最终推荐: **{best[0]}**\n")

    md_path = Path("docs/gold/ensemble_v2_results.md")
    md_path.write_text("".join(md))
    print(f"\n结果已写入: {md_path}")


if __name__ == "__main__":
    main()
