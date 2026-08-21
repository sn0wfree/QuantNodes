#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/13_copper_alpha_probe_v2.py — 新一轮铜因子探针.

检验未探索的铜专属/跨品种信号:
    A. 跨品种截面动量 (CU0 vs M0相对强弱)
    B. 库存变化率/动量 (LME日频库存加速度)
    C. 波动率目标化 (仓位按波动率缩放)
    D. 铜价技术因子 (52周新高proximity, 价格动量z-score)

窗口: IS=2018-2025, OOS=2026只报告; 真实价口径, 5bps.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6 import (
    build_signal_v6, apply_bull_enhance,
)

DB = REPO / "data" / "cmf" / "fundamental.duckdb"
IS_END = pd.Timestamp("2025-12-31")
START = pd.Timestamp("2018-01-01")


def load_close(con, sym):
    tbl = {"CU0": "futures_main_close_cu", "SC0": "futures_main_close_sc",
           "M0": "futures_main_close_m"}[sym]
    df = con.execute(f"SELECT trade_date, close FROM {tbl} ORDER BY trade_date").df()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].resample("W-FRI").last()


def probe_factor(name, factor, fwd4, bull, v6e_pos, close_w):
    """探针: IC + 牛市条件收益 + v6.1叠加."""
    f = factor.dropna()
    idx = f.index.intersection(fwd4.dropna().index)
    f, fw = f.loc[idx], fwd4.loc[idx]

    # IC分年
    ics = {}
    for y in sorted(idx.year.unique()):
        m = idx.year == y
        if m.sum() >= 20:
            ics[y] = f[m].corr(fw[m])
    ic_arr = np.array([v for v in ics.values() if pd.notna(v)])

    # 牛市条件
    bm = bull.reindex(idx).fillna(False)
    cond = {}
    for tag, m in [("全样本因子>0", f > 0), ("牛市因子>0", (f > 0) & bm),
                   ("牛市因子<=0", (f <= 0) & bm)]:
        if m.sum() >= 10:
            cond[tag] = (fw[m].mean() * 13 * 100, int(m.sum()))

    # v6.1叠加: 牛市+因子>0 → boost到0.9
    v6i = v6e_pos.reindex(idx).fillna(0)
    enh = v6i.copy()
    enh[bm & (f > 0)] = np.maximum(v6i[bm & (f > 0)], 0.9)
    base_r, enh_r = {}, {}
    for tag, p in [("v6.1", v6i), ("v6.1+enh", enh)]:
        r = (p.shift(1) * close_w.reindex(idx).pct_change()
             - p.diff().abs() * 5e-4).dropna()
        m_is = (r.index >= START) & (r.index <= IS_END)
        r_is = r[m_is]
        if len(r_is) < 26:
            base_r[tag] = {"sharpe": np.nan, "maxdd": np.nan, "yearly": {}}
            continue
        nav = (1 + r_is).cumprod()
        ann = nav.iloc[-1] ** (52 / len(r_is)) - 1
        yr = {y: (g.iloc[-1] / g.iloc[0] - 1) * 100
              for y, g in nav.groupby(nav.index.year)}
        base_r[tag] = {"sharpe": ann / (r_is.std() * np.sqrt(52)),
                       "maxdd": (nav / nav.cummax() - 1).min() * 100,
                       "yearly": yr}

    # 输出
    ic_str = " ".join(f"{y}:{v:+.2f}" for y, v in ics.items())
    s_base = base_r.get("v6.1", {})
    s_enh = base_r.get("v6.1+enh", {})
    yr_b = s_base.get("yearly", {})
    yr_e = s_enh.get("yearly", {})
    diff_21 = yr_e.get(2021, 0) - yr_b.get(2021, 0)
    diff_25 = yr_e.get(2025, 0) - yr_b.get(2025, 0)

    print(f"\n--- {name} ---")
    print(f"  IC均值={np.nanmean(ic_arr):+.3f}  正IC年数={int((ic_arr>0).sum())}/{len(ic_arr)}  "
          f"分年: {ic_str}")
    for tag, (ann, n) in cond.items():
        print(f"  {tag:<14} 前瞻年化={ann:+6.1f}%  (n={n})")
    print(f"  叠加: Sharpe {s_base.get('sharpe',0):.2f}→{s_enh.get('sharpe',0):.2f}  "
          f"MaxDD {s_base.get('maxdd',0):.1f}→{s_enh.get('maxdd',0):.1f}  "
          f"2021:{diff_21:+.0f}pp  2025:{diff_25:+.0f}pp")


def main():
    con = duckdb.connect(str(DB), read_only=True)

    close_cu = load_close(con, "CU0")
    close_m = load_close(con, "M0")
    fwd4 = close_cu.pct_change(4).shift(-4)
    bull = close_cu.pct_change(26) > 0.10

    sig = build_signal_v6("CU0", con)
    sig_e = apply_bull_enhance(sig, con, threshold=2)
    v6e_pos = sig_e["position_enhanced"]

    print("=" * 88)
    print("新一轮铜因子探针 v2 (IS=2018-2025, 真实价, 5bps)")
    print("=" * 88)

    # ── A. 跨品种截面动量 ──────────────────────────────────────────────
    # CU0 vs M0 相对强弱
    ret_cu = close_cu.pct_change(26)
    ret_m = close_m.pct_change(26).reindex(close_cu.index).ffill()
    rel_mom = (ret_cu - ret_m).rename("CU-M相对动量")
    probe_factor("A1 CU-M相对动量26w", rel_mom, fwd4, bull, v6e_pos, close_cu)

    # CU0 vs M0 13w
    ret_cu13 = close_cu.pct_change(13)
    ret_m13 = close_m.pct_change(13).reindex(close_cu.index).ffill()
    rel_mom13 = (ret_cu13 - ret_m13).rename("CU-M 13w")
    probe_factor("A2 CU-M相对动量13w", rel_mom13, fwd4, bull, v6e_pos, close_cu)

    # CU0相对自身长期动量 (26w vs 52w) → 趋势加速
    mom_52w = close_cu.pct_change(52)
    trend_accel = (ret_cu - mom_52w).rename("趋势加速度")
    probe_factor("A3 趋势加速(CU 26w-52w)", trend_accel, fwd4, bull, v6e_pos, close_cu)

    # ── B. 库存变化率 ──────────────────────────────────────────────
    inv = con.execute(
        "SELECT trade_date, value FROM inventory_daily "
        "WHERE value_col='铜-库存' ORDER BY trade_date"
    ).df()
    inv["trade_date"] = pd.to_datetime(inv["trade_date"])
    inv_w = inv.set_index("trade_date")["value"].resample("W-FRI").last()

    inv_chg20 = inv_w.pct_change(20).rename("LME库存20w变化率")
    probe_factor("B1 LME库存去库率20w", -inv_chg20, fwd4, bull, v6e_pos, close_cu)

    inv_chg4 = inv_w.pct_change(4).rename("LME库存4w变化率")
    probe_factor("B2 LME库存去库率4w", -inv_chg4, fwd4, bull, v6e_pos, close_cu)

    # 库存加速度 (20w变化的4w变化)
    inv_accel = inv_chg20.diff(4).rename("库存加速度")
    probe_factor("B3 库存加速度(加速去库=多)", -inv_accel, fwd4, bull, v6e_pos, close_cu)

    # ── C. 波动率目标化 ──────────────────────────────────────────────
    # 不是alpha因子, 是仓位缩放: 用现有v6.1测试不同目标波动率
    print("\n--- C. 波动率目标化 (v6.1仓位按目标波动率缩放) ---")
    ret_w = close_cu.pct_change()
    vol_4w = ret_w.rolling(4).std() * np.sqrt(52)
    for target_vol in [0.12, 0.15, 0.18]:
        scale = (target_vol / vol_4w).clip(upper=2.0).fillna(1.0)
        scaled_pos = (v6e_pos * scale).clip(upper=1.5)
        r = (scaled_pos.shift(1) * ret_w - scaled_pos.diff().abs() * 5e-4).dropna()
        m_is = (r.index >= START) & (r.index <= IS_END)
        r_is = r[m_is]
        if len(r_is) < 26:
            continue
        nav = (1 + r_is).cumprod()
        ann = nav.iloc[-1] ** (52 / len(r_is)) - 1
        sh = ann / (r_is.std() * np.sqrt(52))
        dd = (nav / nav.cummax() - 1).min() * 100
        print(f"  target={target_vol*100:.0f}%: ann={ann*100:.1f}%  sh={sh:.2f}  dd={dd:.1f}%")
    print(f"  [v6.1基准]  ann={7.6:.1f}%  sh={0.69:.2f}  dd={-19.4:.1f}%")

    # ── D. 铜价技术因子 ──────────────────────────────────────────────
    # 52周新高proximity: price / 52w_max
    high_52w = close_cu.rolling(52).max()
    high_prox = (close_cu / high_52w).rename("52w新高proximity")
    probe_factor("D1 52w新高proximity", high_prox, fwd4, bull, v6e_pos, close_cu)

    # 价格动量z-score (26w)
    mom_z = ((ret_cu - ret_cu.rolling(52).mean()) / ret_cu.rolling(52).std()).rename("动量z")
    probe_factor("D2 价格动量z(26w)", mom_z, fwd4, bull, v6e_pos, close_cu)

    # RSI-14 (周频)
    delta = close_cu.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = (100 - 100 / (1 + rs)).rename("RSI14")
    # RSI<30超卖→看多, RSI>70超买→看空
    rsi_signal = pd.Series(0.0, index=rsi.index)
    rsi_signal[rsi < 30] = 1.0
    rsi_signal[rsi > 70] = -1.0
    probe_factor("D3 RSI14(超卖=多)", rsi_signal, fwd4, bull, v6e_pos, close_cu)

    con.close()
    print("\n探针v2完成.")


if __name__ == "__main__":
    main()
