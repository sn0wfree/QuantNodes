"""scripts/cmf/test_leverage_cppi.py — CPPI结构杠杆验证.

结构: 分层信号 → ×L杠杆 → 组合级风控重铺(hard_cap=-0.15)

对照组:
  A. 基线 (L=1.0, 无重铺)
  B. 裸杠杆 (×L, 无重铺) — 上轮评估的方案
  C. CPPI (×L, 重铺hc=-0.15) — 本轮验证对象

评估:
  - 双窗口: 年化/Sharpe/MaxDD/Calmar/G
  - 合规检查: MaxDD <= -15%
  - 触发统计: 硬止损次数/冷却天数占比
  - warmup期标注
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_layered_ensemble_signal,
    compute_risk_overlay,
)

OOS_S, OOS_E = "2010-01-01", "2019-12-31"
IS_S, IS_E = "2020-01-01", "2026-08-25"
COST_BPS = 5.0

# CPPI重铺参数 (组合级总承诺)
CPPI_HC = -0.15
CPPI_SOFT = -0.12
CPPI_DAILY = -0.03
CPPI_VOL = 0.20
COOLDOWN = 5


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


def count_triggers(pos):
    """统计硬止损触发 (pos由非零变零) 与冷却天数."""
    hard_hit = ((pos == 0) & (pos.shift(1) != 0)).sum()
    zero_days = (pos == 0).mean()
    return int(hard_hit), float(zero_days)


def main():
    sig = compute_layered_ensemble_signal()
    close = sig["close"]
    pos_base = sig["position"]

    vix = sig.get("vix")  # 不在DataFrame里, 重新加载
    from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import _load_indicator_d
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    print("=" * 130)
    print("CPPI杠杆验证: A基线 / B裸杠杆 / C杠杆+风控重铺")
    print("=" * 130)

    rows = []

    def add_row(label, pos):
        oos = evaluate(pos, close, OOS_S, OOS_E)
        is_ = evaluate(pos, close, IS_S, IS_E)
        g = (oos["calmar"] * is_["calmar"]) ** 0.5
        oos_hit, oos_zero = count_triggers(pos[OOS_S:OOS_E])
        is_hit, is_zero = count_triggers(pos[IS_S:IS_E])
        avg_pos = pos.mean()
        rows.append({
            "label": label, "oos_ann": oos["ann_ret"], "oos_dd": oos["max_dd"],
            "oos_cal": oos["calmar"], "is_ann": is_["ann_ret"], "is_dd": is_["max_dd"],
            "is_cal": is_["calmar"], "g": g,
            "oos_hit": oos_hit, "oos_zero": oos_zero,
            "is_hit": is_hit, "is_zero": is_zero,
            "avg_pos": avg_pos,
        })

    # A. 基线
    add_row("A 基线 L=1.0", pos_base)

    for L in [1.5, 2.0, 2.5]:
        pos_lev = (pos_base * L).clip(0, 1.2 * L)

        # B. 裸杠杆 (clip上限放宽到与L匹配)
        add_row(f"B 裸杠杆 L={L}", pos_lev)

        # C. CPPI: 杠杆 + 风控重铺
        pos_cppi = compute_risk_overlay(
            close=close, position=pos_lev, vix=vix, gvz=gvz,
            hard_cap=CPPI_HC, soft_cap=CPPI_SOFT, daily_cap=CPPI_DAILY,
            vol_cap=CPPI_VOL, cooldown_days=COOLDOWN,
        )
        add_row(f"C CPPI L={L} hc=-0.15", pos_cppi)

    print(f'{"方案":<24} | {"OOS年化":>8} {"OOSDD":>8} {"OOSCal":>7} {"触发":>4} {"空仓%":>6} | '
          f'{"IS年化":>8} {"ISDD":>8} {"ISCal":>7} {"触发":>4} {"空仓%":>6} | {"G":>5} {"均仓":>5}')
    print("-" * 130)
    for r in rows:
        compliant = "✓" if (r["oos_dd"] >= CPPI_HC and r["is_dd"] >= CPPI_HC) else "✗"
        print(f'{r["label"]:<24} | {r["oos_ann"]*100:>7.2f}% {r["oos_dd"]*100:>7.2f}% '
              f'{r["oos_cal"]:>7.2f} {r["oos_hit"]:>4d} {r["oos_zero"]*100:>5.1f}% | '
              f'{r["is_ann"]*100:>7.2f}% {r["is_dd"]*100:>7.2f}% '
              f'{r["is_cal"]:>7.2f} {r["is_hit"]:>4d} {r["is_zero"]*100:>5.1f}% | '
              f'{r["g"]:>5.2f} {r["avg_pos"]:>5.2f} {compliant}')

    # 合规CPPI选项
    print("\n--- CPPI合规检查 (双窗口MaxDD>=-15%) ---")
    cppi_rows = [r for r in rows if r["label"].startswith("C")]
    for r in cppi_rows:
        ok = r["oos_dd"] >= CPPI_HC and r["is_dd"] >= CPPI_HC
        print(f'  {r["label"]}: OOS_DD={r["oos_dd"]*100:.2f}%, IS_DD={r["is_dd"]*100:.2f}% '
              f'→ {"合规✓" if ok else "不合规✗"}')

    # ===== Step 2a: 2026压力测试 =====
    print("\n" + "=" * 130)
    print("2026崩盘年压力测试 (当年B&H DD=-30.37%)")
    print("=" * 130)
    y26 = evaluate(pd.Series(1.0, index=close.index), close, "2026-01-01", "2026-08-25")
    print(f'  B&H 2026: 年化={y26["ann_ret"]*100:.2f}%, DD={y26["max_dd"]*100:.2f}%')
    for r in rows:
        # 重新构建该方案的pos以测2026
        pass
    # 逐方案重算2026
    def year26(pos):
        m = evaluate(pos, close, "2026-01-01", "2026-08-25")
        return m

    pos_map = {"A 基线 L=1.0": pos_base}
    for L in [1.5, 2.0, 2.5]:
        pos_lev = (pos_base * L).clip(0, 1.2 * L)
        pos_map[f"B 裸杠杆 L={L}"] = pos_lev
        pos_map[f"C CPPI L={L} hc=-0.15"] = compute_risk_overlay(
            close=close, position=pos_lev, vix=vix, gvz=gvz,
            hard_cap=CPPI_HC, soft_cap=CPPI_SOFT, daily_cap=CPPI_DAILY,
            vol_cap=CPPI_VOL, cooldown_days=COOLDOWN,
        )
    print(f'{"方案":<24} {"2026年化":>10} {"2026DD":>10}')
    for label, pos in pos_map.items():
        m = year26(pos)
        print(f'{label:<24} {m["ann_ret"]*100:>9.2f}% {m["max_dd"]*100:>9.2f}%')

    # ===== Step 2c: 2013鞭打损耗 =====
    print("\n" + "=" * 130)
    print("2013鞭打损耗测试 (连续阴跌, CPPI反复触发成本)")
    print("=" * 130)
    print(f'{"方案":<24} {"2013年化":>10} {"2013DD":>10} {"触发":>5}')
    for label, pos in pos_map.items():
        m = evaluate(pos, close, "2013-01-01", "2013-12-31")
        hit, _ = count_triggers(pos["2013-01-01":"2013-12-31"])
        print(f'{label:<24} {m["ann_ret"]*100:>9.2f}% {m["max_dd"]*100:>9.2f}% {hit:>5d}')

    # ===== 保存数据供后续脚本使用 =====
    import json
    summary = {r["label"]: {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                             for k, v in r.items() if k != "label"} for r in rows}
    with open("reports/gold/leverage_cppi_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n数据已保存: reports/gold/leverage_cppi_summary.json")


if __name__ == "__main__":
    main()
