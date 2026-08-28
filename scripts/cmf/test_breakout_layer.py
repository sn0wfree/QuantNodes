"""scripts/cmf/test_breakout_layer.py — 第4路breakout层评估.

结构: 0.3×macro + 0.3×rr + 0.3×mom + 0.1×breakout (每路独立风控)

评估:
- 突破信号源: gold_breakout (20日高点×0.95+3%止损)
- 每路独立过 compute_risk_overlay(hc=-0.10)
- 主方案 bw=0.10, 敏感性 bw∈{0.05, 0.15}
- 双窗口 + 子周期 + 信号相关性 + MaxDD合规
- 对照: 现有分层1/3等权 (综合分2.89)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_layered_ensemble_signal,
    compute_risk_overlay,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_macro_regime_signal,
    _load_indicator_d,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import (
    compute_real_rate_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import (
    compute_momentum_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_breakout import (
    compute_breakout_signal,
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


def evaluate(pos, close, s, e):
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
    n_y = len(c) / 252
    ar = (nav.iloc[-1] / sp) ** (1 / n_y) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    return {"ann_ret": ar, "sharpe": ar / vol if vol > 0 else 0,
            "max_dd": dd, "calmar": ar / abs(dd) if dd != 0 else 0}


def full_eval(pos, close):
    oos = evaluate(pos, close, OOS_S, OOS_E)
    is_ = evaluate(pos, close, IS_S, IS_E)
    sub = {n: evaluate(pos, close, s, e) for n, s, e in SUBPERIODS}
    g = (oos["calmar"] * is_["calmar"]) ** 0.5
    return {"OOS": oos, "IS": is_, "G": g, "MinCal": min(oos["calmar"], is_["calmar"]),
            "WorstDD": min(oos["max_dd"], is_["max_dd"]), "Sub": sub}


def main():
    # 获取基线 (现有分层1/3等权)
    baseline = compute_layered_ensemble_signal()
    close = baseline["close"]
    pos_base = baseline["position"]

    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    # 获取四路信号 (每路独立风控)
    macro_raw = compute_macro_regime_signal()["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    rr_raw = compute_real_rate_signal(variant="tips_ma")["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    mom_raw = compute_momentum_signal(variant="long_only")["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    bo_raw = compute_breakout_signal()["position"].reindex(close.index).fillna(0)

    def overlay(pos):
        return compute_risk_overlay(
            close=close, position=pos, vix=vix, gvz=gvz,
            hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15, cooldown_days=5)

    macro_ov = overlay(macro_raw)
    rr_ov = overlay(rr_raw)
    mom_ov = overlay(mom_raw)
    bo_ov = overlay(bo_raw)

    # 信号相关性 (原始信号)
    print("=" * 100)
    print("信号相关性 (原始信号, 风控前)")
    print("=" * 100)
    sig_df = pd.DataFrame({"macro": macro_raw, "real_rate": rr_raw,
                           "momentum": mom_raw, "breakout": bo_raw})
    print(sig_df.corr().to_string())
    print(f"\nbreakout 统计: mean={bo_raw.mean():.3f}, =1占比={(bo_raw>0.95).mean()*100:.1f}%, =0占比={(bo_raw<0.05).mean()*100:.1f}%")

    # 候选列表
    candidates = {}

    # 基线: 现有分层1/3等权
    candidates["基线 (三路1/3等权)"] = pos_base

    # 主方案: 0.3/0.3/0.3/0.1
    candidates["四路 0.3/0.3/0.3/0.1"] = (0.3 * macro_ov + 0.3 * rr_ov + 0.3 * mom_ov + 0.1 * bo_ov).clip(0, 1.2)

    # 敏感性: bw=0.05
    candidates["四路 0.3/0.3/0.3/0.05"] = (0.3167 * macro_ov + 0.3167 * rr_ov + 0.3167 * mom_ov + 0.05 * bo_ov).clip(0, 1.2)

    # 敏感性: bw=0.15
    candidates["四路 0.3/0.3/0.3/0.15"] = (0.2833 * macro_ov + 0.2833 * rr_ov + 0.2833 * mom_ov + 0.15 * bo_ov).clip(0, 1.2)

    # 对照: 去掉breakout的原始三路等权 (等价于基线但显式构造)
    candidates["三路 0.33/0.33/0.33 无breakout"] = ((macro_ov + rr_ov + mom_ov) / 3).clip(0, 1.2)

    # 单独breakout 0.1 (纯增量贡献)
    candidates["仅breakout×0.1"] = (0.1 * bo_ov).clip(0, 1.2)

    # 评估
    results = {n: full_eval(pos, close) for n, pos in candidates.items()}

    # 表1: 双窗口核心指标
    print("\n" + "=" * 120)
    print("表1: 双窗口核心指标")
    print("=" * 120)
    print(f'{"方案":<36} | {"OOS年化":>8} {"OOSDD":>8} {"OOSCal":>7} | '
          f'{"IS年化":>8} {"ISDD":>8} {"ISCal":>7} | {"G":>5} {"MinCal":>6} {"合规":>4}')
    print("-" * 120)
    for n, ev in results.items():
        o, i = ev["OOS"], ev["IS"]
        ok = "✓" if (o["max_dd"] >= -0.15 and i["max_dd"] >= -0.15) else "✗"
        print(f'{n:<36} | {o["ann_ret"]*100:>7.2f}% {o["max_dd"]*100:>7.2f}% {o["calmar"]:>7.2f} | '
              f'{i["ann_ret"]*100:>7.2f}% {i["max_dd"]*100:>7.2f}% {i["calmar"]:>7.2f} | '
              f'{ev["G"]:>5.2f} {ev["MinCal"]:>6.2f} {ok:>4}')

    # 表2: 子周期
    print("\n" + "=" * 120)
    print("表2: 子周期 (年化/MaxDD)")
    print("=" * 120)
    print(f'{"方案":<36} | '
          f'{"10-14":>16} | {"15-19":>16} | {"20-23":>16} | {"24-26":>16}')
    print("-" * 120)
    for n, ev in results.items():
        row = f'{n:<36} | '
        for sp in ["10-14", "15-19", "20-23", "24-26"]:
            m = ev["Sub"][sp]
            row += f'{m["ann_ret"]*100:>6.1f}%/{m["max_dd"]*100:>6.1f}% | '
        print(row)

    # 表3: 综合评分
    print("\n" + "=" * 100)
    print("综合评分排序 (G×2 + MinCal×1.5 + 合规奖励)")
    print("=" * 100)
    scores = []
    for n, ev in results.items():
        score = ev["G"] * 2 + ev["MinCal"] * 1.5
        o, i = ev["OOS"], ev["IS"]
        if o["max_dd"] >= -0.15 and i["max_dd"] >= -0.15:
            score += 1.0
        elif o["max_dd"] >= -0.20 and i["max_dd"] >= -0.15:
            score += 0.5
        scores.append((n, score, ev["G"], ev["MinCal"], o["max_dd"], i["max_dd"]))
    scores.sort(key=lambda x: -x[1])
    print(f'{"#":<3} {"方案":<36} {"综合分":>7} {"G":>5} {"MinCal":>7} {"OOSDD":>8} {"ISDD":>8}')
    print("-" * 100)
    for idx, (n, s, g, mc, odd, idd) in enumerate(scores, 1):
        print(f'{idx:<3} {n:<36} {s:>7.2f} {g:>5.2f} {mc:>7.2f} {odd*100:>7.2f}% {idd*100:>7.2f}%')

    # 决策门
    best = scores[0]
    print(f"\n最佳: {best[0]} (综合分={best[1]:.2f})")
    if best[1] > 2.89 and best[4] >= -0.15 and best[5] >= -0.15:
        print("✅ 通过决策门 (综合分>2.89且双窗口DD≤-15%)")
    else:
        print("❌ 未通过决策门")


if __name__ == "__main__":
    main()
