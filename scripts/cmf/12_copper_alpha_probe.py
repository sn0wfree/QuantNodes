#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/12_copper_alpha_probe.py — 铜专属Alpha因子探针 (方向B, 进攻型).

检验五组从未单独利用的铜专属数据:
    B1a 洋山铜溢价   (tfapi, 2012-11+)     中国现货需求
    B1b TC粗炼费     (tfapi, 2011-03+)     矿端供给紧张度
    B2  全球库存合成 (LME+COMEX+SHFE)      库存全景 vs 单LME
    B3  期限结构深挖 (spread_1_3/1_4, OI)  carry增强
    B4  CFTC铜持仓   (mm_net_pct, 2010+)   持仓动量/拥挤度

每个因子输出:
    §x.1 全样本IC (前瞻4周收益, 分年, 方向稳定年数)
    §x.2 牛市子样本条件收益 (26周动量>10%)
    §x.3 叠加模拟: 因子作为v6牛市增强层的分年改变

纪律: IS=2018-2025, OOS=2026只报告; 真实周收盘价口径.
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

from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6 import build_signal_v6

DB = REPO / "data" / "cmf" / "fundamental.duckdb"
IS_END = pd.Timestamp("2025-12-31")
START = pd.Timestamp("2018-01-01")


def load_weekly(con, table, col_date, col_val, where=""):
    df = con.execute(
        f"SELECT {col_date} d, {col_val} v FROM {table} {where} ORDER BY d"
    ).df()
    df["d"] = pd.to_datetime(df["d"])
    return df.set_index("d")["v"].resample("W-FRI").last()


def probe(name: str, factor: pd.Series, fwd4: pd.Series, bull: pd.Series,
          v6_pos: pd.Series, close_w: pd.Series, direction: int = 1):
    """单因子标准探针: IC + 牛市条件 + v6叠加. direction=1因子大看多.

    低频因子 (如TC月度) ffill到周频网格, 最长回填13周.
    """
    f = (factor * direction).resample("W-FRI").last().ffill(limit=13).dropna()
    idx = f.index.intersection(fwd4.dropna().index)
    f, fw = f.loc[idx], fwd4.loc[idx]

    # §x.1 分年IC
    ics = {}
    for y in sorted(idx.year.unique()):
        m = idx.year == y
        if m.sum() >= 20:
            ics[y] = f[m].corr(fw[m])
    ic_arr = np.array([v for v in ics.values() if pd.notna(v)])
    n_pos = (ic_arr > 0).sum() if len(ic_arr) else 0

    # §x.2 牛市条件收益: 因子>0 vs <=0 的前瞻4周收益 (年化)
    bm = bull.reindex(idx).fillna(False)
    cond = {}
    for tag, m in [("全样本因子>0", f > 0), ("全样本因子<=0", f <= 0),
                   ("牛市因子>0", (f > 0) & bm), ("牛市因子<=0", (f <= 0) & bm)]:
        if m.sum() >= 10:
            cond[tag] = (fw[m].mean() * 13 * 100, int(m.sum()))  # 4周→年化≈×13

    # §x.3 v6叠加: 牛市时因子>0 → 仓位提升至max(pos, 0.9)
    v6i = v6_pos.reindex(idx).fillna(0)
    enh = v6i.copy()
    enh[bm & (f > 0)] = np.maximum(v6i[bm & (f > 0)], 0.9)
    base_r, enh_r = {}, {}
    for tag, (p, store) in {"v6": (v6i, base_r), "v6+enh": (enh, enh_r)}.items():
        r = (p.shift(1) * close_w.reindex(idx).pct_change()
             - p.diff().abs() * 5e-4).dropna()
        m_is = (r.index >= START) & (r.index <= IS_END)
        r_is = r[m_is]
        if len(r_is) < 26:
            store["sharpe"] = store["maxdd"] = np.nan
            store["yearly"] = {}
            continue
        nav = (1 + r_is).cumprod()
        ann = nav.iloc[-1] ** (52 / len(r_is)) - 1
        store["sharpe"] = ann / (r_is.std() * np.sqrt(52))
        store["maxdd"] = (nav / nav.cummax() - 1).min() * 100
        store["yearly"] = {y: (g.iloc[-1] / g.iloc[0] - 1) * 100
                           for y, g in nav.groupby(nav.index.year)}

    # 报告
    ic_str = " ".join(f"{y}:{v:+.2f}" for y, v in ics.items())
    print(f"\n--- {name} ---")
    print(f"  IC分年: {ic_str}")
    print(f"  IC均值={np.nanmean(ic_arr):+.3f}  正IC年数={n_pos}/{len(ic_arr)}")
    for tag, (ann, n) in cond.items():
        print(f"  {tag:<12} 前瞻年化={ann:+6.1f}%  (n={n})")
    print(f"  叠加: v6 Sharpe {base_r['sharpe']:.2f}→{enh_r['sharpe']:.2f}  "
          f"MaxDD {base_r['maxdd']:.1f}→{enh_r['maxdd']:.1f}")
    yr_keys = sorted(set(base_r["yearly"]) | set(enh_r["yearly"]))
    diff = [enh_r["yearly"].get(y, 0) - base_r["yearly"].get(y, 0) for y in yr_keys]
    print(f"  分年差(enh-v6): " + " ".join(
        f"{y}:{d:+.1f}" for y, d in zip(yr_keys, diff)))


def main():
    con = duckdb.connect(str(DB), read_only=True)

    close = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    close["trade_date"] = pd.to_datetime(close["trade_date"])
    close_w = close.set_index("trade_date")["close"].resample("W-FRI").last()
    fwd4 = close_w.pct_change(4).shift(-4)
    bull = close_w.pct_change(26) > 0.10

    v6_pos = build_signal_v6("CU0", con)["position"]

    def tfapi(name):
        return load_weekly(con, "tfapi_indicators", "trade_date", "value",
                           f"WHERE index_name='{name}'")

    def rz(s, w=52):
        return ((s - s.rolling(w).mean()) / s.rolling(w).std())

    print("=" * 88)
    print("铜专属Alpha因子探针 (IS=2018-2025 / OOS=2026只报告, 真实价, 5bps)")
    print("=" * 88)

    # B1a 洋山铜溢价: 溢价高=中国现货需求强 → 看多
    probe("B1a 洋山铜溢价 z52", rz(tfapi("洋山铜溢价")), fwd4, bull, v6_pos, close_w, +1)
    probe("B1a 洋山铜溢价 Δ4w", tfapi("洋山铜溢价").diff(4), fwd4, bull, v6_pos, close_w, +1)

    # B1b TC粗炼费: TC低=矿端紧 → 看多 (方向=-1)
    probe("B1b TC负z52 (矿紧=多)", rz(tfapi("现货:中国铜冶炼厂:粗炼费(TC)")),
          fwd4, bull, v6_pos, close_w, -1)
    probe("B1b TC Δ13w (TC降=多)", tfapi("现货:中国铜冶炼厂:粗炼费(TC)").diff(13),
          fwd4, bull, v6_pos, close_w, -1)

    # B2 全球库存: LME+COMEX+SHFE 合计 z负(库存低)=看多
    lme = tfapi("LME库存总计_铜")
    cmx = tfapi("COMEX:铜:库存")
    shfe = tfapi("SHFE库存小计")
    globe = (lme + cmx.reindex(lme.index).fillna(0)
             + shfe.reindex(lme.index).fillna(0))
    probe("B2 全球库存负z52 (低库存=多)", rz(globe), fwd4, bull, v6_pos, close_w, -1)
    probe("B2 全球库存Δ4w (去库=多)", globe.diff(4), fwd4, bull, v6_pos, close_w, -1)
    probe("B2 COMEX库存负z52", rz(cmx), fwd4, bull, v6_pos, close_w, -1)

    # B3 期限结构深挖
    ts = con.execute(
        "SELECT trade_date, spread_1_3, spread_1_4, oi_near, close_near2 "
        "FROM term_structure_cu ORDER BY trade_date"
    ).df()
    ts["trade_date"] = pd.to_datetime(ts["trade_date"])
    ts = ts.set_index("trade_date").resample("W-FRI").last()
    s13 = ts["spread_1_3"] / ts["close_near2"]
    s14 = ts["spread_1_4"] / ts["close_near2"]
    probe("B3 spread_1_3 z52 (back=多)", rz(s13), fwd4, bull, v6_pos, close_w, -1)
    probe("B3 spread_1_4 z52 (back=多)", rz(s14), fwd4, bull, v6_pos, close_w, -1)
    probe("B3 近月OI变化13w (增仓=多)", ts["oi_near"].pct_change(13),
          fwd4, bull, v6_pos, close_w, +1)

    # B4 CFTC: mm_net_pct 双向
    cftc = load_weekly(con, "tfapi_cftc", "report_date", "mm_net_pct",
                       "WHERE market LIKE '%COPPER%'")
    probe("B4 CFTC mm_net_pct z52 (顺势=多)", rz(cftc), fwd4, bull, v6_pos, close_w, +1)
    probe("B4 CFTC mm_net_pct 负z (逆向拥挤=多)", rz(cftc), fwd4, bull, v6_pos, close_w, -1)

    con.close()
    print("\n探针完成. 采纳标准: |IC|>0.08且方向稳定>=5/8年, 或牛市增强显著"
          "(牛市年化+3pp且回撤代价<4pp).")


if __name__ == "__main__":
    main()
