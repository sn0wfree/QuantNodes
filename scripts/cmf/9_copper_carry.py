#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/9_copper_carry.py — 铜期限结构/Carry独立子策略验证.

验证流程 (防过拟合: 2018-2022选参, 2023-2026 OOS验证):
    1. IC分析: carry_zscore vs 前瞻4周收益 (分年)
    2. 参数扫描: 变体 × min_duration × mom_window (IS窗口)
    3. OOS验证: 最优参数在2023-2026的表现
    4. 组合分析: w*carry + (1-w)*v5, 相关性检验
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

from QuantNodes.strategy.commodity_fundamental.copper.copper_carry import (
    load_term_structure, compute_carry_signal, build_signal_carry, backtest_carry,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v5 import (
    build_signal_v5, backtest_v5,
)

DB = REPO / "data" / "cmf" / "fundamental.duckdb"

IS_END = "2022-12-31"   # 选参窗口终点
OOS_START = "2023-01-01"  # OOS窗口起点


def yearly_returns(nav: pd.Series) -> dict:
    out = {}
    for year, grp in nav.groupby(nav.index.year):
        if len(grp) > 1:
            out[year] = (grp.iloc[-1] / grp.iloc[0] - 1) * 100
    return out


def main():
    con = duckdb.connect(str(DB), read_only=True)

    ts = load_term_structure(con)
    sig = compute_carry_signal(ts)

    print("=" * 72)
    print("§1 IC分析: carry_zscore vs 前瞻4周收益 (分年)")
    print("=" * 72)
    fwd4 = ts["close_near"].pct_change(4).shift(-4)
    ic_by_year = {}
    for year in sorted(ts.index.year.unique()):
        mask = ts.index.year == year
        ic = sig.loc[mask, "carry_zscore"].corr(fwd4[mask])
        if pd.notna(ic):
            ic_by_year[year] = ic
            print(f"  {year}: IC = {ic:+.3f}")
    avg_ic = np.mean(list(ic_by_year.values()))
    print(f"  平均IC = {avg_ic:+.3f}")

    print()
    print("=" * 72)
    print("§2 参数扫描 (IS窗口: 2018-2022)")
    print("=" * 72)
    print(f"{'变体':<14}{'min_dur':<9}{'mom_w':<7}{'年化%':<9}{'Sharpe':<9}{'MaxDD%':<9}{'胜率%':<7}")
    print("-" * 64)

    results = []
    for variant in ["state", "state_mom", "continuous",
                    "state_ls", "state_mom_ls", "continuous_ls", "bear_only_ls"]:
        for min_dur in [2, 4, 8]:
            for mom_w in [8, 13, 26]:
                signal = build_signal_carry(
                    con, variant=variant, min_duration=min_dur, mom_window=mom_w
                )
                signal_is = signal[signal.index <= IS_END]
                r = backtest_carry(signal_is)
                results.append({
                    "variant": variant, "min_dur": min_dur, "mom_w": mom_w,
                    "ann_ret": r["ann_ret"] * 100, "sharpe": r["sharpe"],
                    "max_dd": r["max_dd"] * 100, "win_rate": r["win_rate"] * 100,
                })
                print(f"{variant:<14}{min_dur:<9}{mom_w:<7}{r['ann_ret']*100:<9.1f}"
                      f"{r['sharpe']:<9.2f}{r['max_dd']*100:<9.1f}{r['win_rate']*100:<7.1f}")

    # 选Sharpe最高且MaxDD > -30%的参数
    valid = [x for x in results if x["max_dd"] > -30]
    best = max(valid, key=lambda x: x["sharpe"]) if valid else max(results, key=lambda x: x["sharpe"])
    print(f"\n  IS最优: {best['variant']} min_dur={best['min_dur']} mom_w={best['mom_w']} "
          f"(Sharpe={best['sharpe']:.2f})")

    print()
    print("=" * 72)
    print("§3 OOS验证 (2023-2026, 最优参数 + 全窗口)")
    print("=" * 72)
    best_signal = build_signal_carry(
        con, variant=best["variant"], min_duration=best["min_dur"], mom_window=best["mom_w"]
    )
    oos_signal = best_signal[best_signal.index >= OOS_START]
    r_oos = backtest_carry(oos_signal)
    r_full = backtest_carry(best_signal)

    print(f"  OOS (2023+):  年化={r_oos['ann_ret']*100:.1f}%  Sharpe={r_oos['sharpe']:.2f}  "
          f"MaxDD={r_oos['max_dd']*100:.1f}%")
    print(f"  全窗口:       年化={r_full['ann_ret']*100:.1f}%  Sharpe={r_full['sharpe']:.2f}  "
          f"MaxDD={r_full['max_dd']*100:.1f}%")

    # OOS基准: 买入持有
    bh_ret = best_signal["M"].pct_change()
    oos_ret = bh_ret[bh_ret.index >= OOS_START].dropna()
    bh_nav = (1 + oos_ret).cumprod()
    bh_ann = (bh_nav.iloc[-1]) ** (52 / len(oos_ret)) - 1
    bh_dd = (bh_nav / bh_nav.cummax() - 1).min()
    print(f"  OOS买入持有:  年化={bh_ann*100:.1f}%  MaxDD={bh_dd*100:.1f}%")

    print()
    print("=" * 72)
    print("§4 组合分析: w*carry + (1-w)*v5 (全窗口2018+)")
    print("=" * 72)

    # v5信号对齐到carry索引 (2018+)
    v5_signal = build_signal_v5("CU0", con)
    v5_ret = v5_signal["M"].pct_change()

    carry_ret = best_signal["position"].shift(1) * best_signal["M"].pct_change()
    v5_ret = v5_ret.reindex(best_signal.index)
    carry_ret = carry_ret.reindex(best_signal.index)

    # v5策略周收益 (从v5回测取)
    r_v5 = backtest_v5(v5_signal, v5_signal["M"])
    v5_nav_full = r_v5["nav"]
    v5_ret_full = v5_nav_full.pct_change().reindex(best_signal.index).fillna(0)
    carry_ret_f = carry_ret.fillna(0)

    # 相关性
    common_mask = (v5_ret_full != 0) | (carry_ret_f != 0)
    corr = v5_ret_full[common_mask].corr(carry_ret_f[common_mask])
    print(f"  carry vs v5 周收益相关性: {corr:.3f}")

    print(f"\n{'组合':<18}{'年化%':<9}{'Sharpe':<9}{'MaxDD%':<9}")
    print("-" * 45)

    # v5基准
    nav_v5_align = (1 + v5_ret_full).cumprod()
    v5_ann = (nav_v5_align.iloc[-1]) ** (52 / len(nav_v5_align)) - 1
    v5_vol = v5_ret_full.std() * np.sqrt(52)
    v5_dd = (nav_v5_align / nav_v5_align.cummax() - 1).min()
    print(f"{'v5单独':<18}{v5_ann*100:<9.1f}{v5_ann/v5_vol:<9.2f}{v5_dd*100:<9.1f}")

    # carry基准
    nav_carry_align = (1 + carry_ret_f).cumprod()
    c_ann = (nav_carry_align.iloc[-1]) ** (52 / len(nav_carry_align)) - 1
    c_vol = carry_ret_f.std() * np.sqrt(52)
    c_dd = (nav_carry_align / nav_carry_align.cummax() - 1).min()
    if c_vol > 0:
        print(f"{'carry单独':<18}{c_ann*100:<9.1f}{c_ann/c_vol:<9.2f}{c_dd*100:<9.1f}")

    for w in [0.3, 0.5]:
        combo_ret = w * carry_ret_f + (1 - w) * v5_ret_full
        combo_nav = (1 + combo_ret).cumprod()
        ann = combo_nav.iloc[-1] ** (52 / len(combo_nav)) - 1
        vol = combo_ret.std() * np.sqrt(52)
        dd = (combo_nav / combo_nav.cummax() - 1).min()
        print(f"{f'w={w}组合':<18}{ann*100:<9.1f}{ann/vol if vol>0 else 0:<9.2f}{dd*100:<9.1f}")

    print()
    print("=" * 72)
    print("§5 分年对比 (carry最优参数 vs 买入持有)")
    print("=" * 72)
    carry_nav = backtest_carry(best_signal)["nav"]
    y_carry = yearly_returns(carry_nav)
    bh_nav_full = (1 + best_signal["M"].pct_change().fillna(0)).cumprod()
    y_bh = yearly_returns(bh_nav_full)
    print(f"{'年份':<8}{'carry%':<10}{'买入持有%':<12}")
    for year in sorted(y_bh):
        c_val = y_carry.get(year, 0)
        print(f"{year:<8}{c_val:<10.1f}{y_bh[year]:<12.1f}")

    print()
    print("=" * 72)
    print("§6 Long-only vs Long-short 对比 + 空头贡献归因")
    print("=" * 72)

    # 找IS窗口内每个变体族的最优 (long-only vs long-short)
    lo = [x for x in results if not x["variant"].endswith("_ls")]
    ls = [x for x in results if x["variant"].endswith("_ls")]
    best_lo = max(lo, key=lambda x: x["sharpe"])
    best_ls = max(ls, key=lambda x: x["sharpe"])

    print(f"IS最优 long-only : {best_lo['variant']} min_dur={best_lo['min_dur']} "
          f"mom_w={best_lo['mom_w']} (Sharpe={best_lo['sharpe']:.2f})")
    print(f"IS最优 long-short: {best_ls['variant']} min_dur={best_ls['min_dur']} "
          f"mom_w={best_ls['mom_w']} (Sharpe={best_ls['sharpe']:.2f})")

    for tag, cfg in [("long-only", best_lo), ("long-short", best_ls)]:
        sig = build_signal_carry(
            con, variant=cfg["variant"], min_duration=cfg["min_dur"], mom_window=cfg["mom_w"]
        )
        r_full = backtest_carry(sig)
        r_oos = backtest_carry(sig[sig.index >= OOS_START])
        print(f"\n  [{tag}] {cfg['variant']} (全窗口 / OOS):")
        print(f"    年化: {r_full['ann_ret']*100:.1f}% / {r_oos['ann_ret']*100:.1f}%")
        print(f"    Sharpe: {r_full['sharpe']:.2f} / {r_oos['sharpe']:.2f}")
        print(f"    MaxDD: {r_full['max_dd']*100:.1f}% / {r_oos['max_dd']*100:.1f}%")

    # 空头贡献归因 (long-short最优参数)
    sig_ls = build_signal_carry(
        con, variant=best_ls["variant"], min_duration=best_ls["min_dur"],
        mom_window=best_ls["mom_w"],
    )
    pos = sig_ls["position"]
    ret = sig_ls["M"].pct_change()
    strat_ret = (pos.shift(1) * ret).dropna()
    short_mask = pos.shift(1).reindex(strat_ret.index) < 0
    long_mask = pos.shift(1).reindex(strat_ret.index) > 0
    n_short = short_mask.sum()
    print(f"\n  空头归因 ({best_ls['variant']}, 全窗口):")
    print(f"    空头周数: {n_short} ({n_short/len(strat_ret)*100:.0f}%)")
    if n_short > 0:
        short_ret = strat_ret[short_mask]
        short_nav = (1 + short_ret).cumprod()
        short_cum = short_nav.iloc[-1] - 1 if len(short_nav) > 0 else 0
        print(f"    空头累计贡献: {short_cum*100:.1f}%")
        print(f"    空头周胜率: {(short_ret > 0).mean()*100:.1f}%")
    long_ret = strat_ret[long_mask]
    if len(long_ret) > 0:
        long_nav = (1 + long_ret).cumprod()
        print(f"    多头累计贡献: {(long_nav.iloc[-1]-1)*100:.1f}%")

    # LS vs LO 组合对比 (v5组合, 用各自IS最优)
    print(f"\n  组合对比 (w=0.5, 全窗口):")
    v5_ret_full = None
    for tag, cfg in [("long-only", best_lo), ("long-short", best_ls)]:
        sig_i = build_signal_carry(
            con, variant=cfg["variant"], min_duration=cfg["min_dur"], mom_window=cfg["mom_w"]
        )
        c_ret = (sig_i["position"].shift(1) * sig_i["M"].pct_change()).fillna(0)
        if v5_ret_full is None:
            v5_signal = build_signal_v5("CU0", con)
            v5_nav_full = backtest_v5(v5_signal, v5_signal["M"])["nav"]
            v5_ret_full = v5_nav_full.pct_change().reindex(sig_i.index).fillna(0)
        c_ret = c_ret.reindex(sig_i.index)
        combo = (0.5 * c_ret + 0.5 * v5_ret_full).dropna()
        combo_nav = (1 + combo).cumprod()
        ann = combo_nav.iloc[-1] ** (52 / len(combo_nav)) - 1
        vol = combo.std() * np.sqrt(52)
        dd = (combo_nav / combo_nav.cummax() - 1).min()
        corr = c_ret.corr(v5_ret_full)
        print(f"    {tag:<12} 年化={ann*100:.1f}%  Sharpe={ann/vol if vol>0 else 0:.2f}  "
              f"MaxDD={dd*100:.1f}%  corr(v5)={corr:.3f}")

    con.close()


if __name__ == "__main__":
    main()
