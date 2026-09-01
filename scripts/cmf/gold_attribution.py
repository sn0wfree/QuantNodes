#!/usr/bin/env python3
"""gold_attribution.py — 黄金生产策略 (B3全栈) 归因分析.

回答三个问题:
    1. 各层贡献多少?  — 七级消融阶梯 (信号→风控→杠杆→网格→平滑→应急)
    2. 子信号谁在赚钱? — 4路子信号独立业绩 + 权重 + 相关性
    3. 每次调仓为什么? — 调仓归因日志 (触发源分类)

用法:
    python3 scripts/cmf/gold_attribution.py              # 全期终端报告
    python3 scripts/cmf/gold_attribution.py --year 2026  # 加输 trades_2026.csv

输出: reports/gold/attribution_ladder.csv (每次运行刷新)
      reports/gold/trades_{year}.csv (--year 时)
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/ll/Public/QuantNodes")
sys.path.insert(0, "/home/ll/Public/QuantNodes/scripts/cmf")

import argparse

import numpy as np
import pandas as pd

from QuantNodes.strategy.commodity_fundamental.gold import (
    build_chain,
    build_production_position,
    load_au_adj_ohlcv,
    load_indicator_d,
    load_vix_gvz,
    o2o_returns,
    weekly_grid_inventory,
    smooth_execution,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_ensemble import (
    compute_layered_ensemble_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_breakout import (
    compute_breakout_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import (
    compute_momentum_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import (
    compute_real_rate_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_macro_regime_signal,
)

OUTPUT_DIR = "reports/gold"

PIPELINE = [
    ("数据层", "DuckDB: futures_main_close_au_adj + vix/gvz(shift2) + tips10y"),
    ("信号层", "4路加权: macro 0.2667 | real_rate 0.2667 | momentum 0.2667 | breakout 0.2"),
    ("一层风控", "内置: daily-5% / hard-10% / soft-7% / vol 15%"),
    ("杠杆层", "×1.5 → clip 1.8"),
    ("二层风控", "hard-20% / soft-12% / daily-5% / vol 20% / cooldown 5d"),
    ("死带", "0.05"),
    ("核心裁剪", "≤1.25"),
    ("网格库存", "+0~0.25 (sp=2%, 3档, 周频)"),
    ("总封顶", "1.5"),
    ("日平滑", "cap=0.20"),
    ("应急", "单日<-5% 压0"),
    ("执行", "O2O +5bps"),
]

BREAKOUT_WEIGHT = 0.2
BASE_WEIGHT = (1 - BREAKOUT_WEIGHT) / 3


def perf(returns: pd.Series) -> dict:
    """六指标 (与包内口径一致)."""
    if returns is None or len(returns) < 2:
        return {"cagr": np.nan, "sharpe": np.nan, "mdd": np.nan,
                "calmar": np.nan, "y26": np.nan}
    nav = (1 + returns).cumprod()
    n = len(returns)
    cagr = nav.iloc[-1] ** (252 / n) - 1
    vol = returns.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    r26 = returns[returns.index >= "2026-01-01"]
    y26 = (1 + r26).prod() - 1 if len(r26) > 1 else np.nan
    return {
        "cagr": cagr, "sharpe": cagr / vol if vol > 0 else np.nan,
        "mdd": dd, "calmar": cagr / abs(dd) if dd != 0 else np.nan, "y26": y26,
    }


# ===== Part 1: 七级消融阶梯 =====

def build_ladder(close: pd.Series) -> tuple[dict[str, pd.Series], pd.Series]:
    """构建七级仓位序列 (逐层叠加). 返回 (levels, grid)."""
    sig_raw = compute_layered_ensemble_signal(use_risk_overlay=False)
    L0 = sig_raw["position"]
    L1 = compute_layered_ensemble_signal(use_risk_overlay=True, daily_cap=-0.05)["position"]
    L2 = (L1 * 1.5).clip(0, 1.8)
    L3 = build_chain()  # B3链 (-5/-5/-20)
    core = L3.clip(0, 1.25)
    grid = weekly_grid_inventory(close, L3.index, spacing_pct=0.02, step=0.0833, max_grids=3)
    L4 = (core + grid).clip(0, 1.5)
    L5 = smooth_execution(L4, close, cap=0.20, emergency=None)
    L6 = smooth_execution(L4, close, cap=0.20, emergency=-0.05)

    # 互锁: L6 必须与生产入口逐值一致
    prod = build_production_position()
    aligned = L6.reindex(prod.index)
    assert np.allclose(aligned.values, prod.values, atol=1e-12), \
        "L6 与 build_production_position 不一致, 消融阶梯口径失效!"

    levels = {
        "L0 裸信号": L0,
        "L1 +一层风控": L1,
        "L2 +杠杆×1.5": L2,
        "L3 +二层风控(链)": L3,
        "L4 +核心裁剪+网格": L4,
        "L5 +日平滑": L5,
        "L6 +应急(生产B3)": L6,
    }
    return levels, grid


def ladder_table(levels: dict[str, pd.Series], open_price: pd.Series) -> pd.DataFrame:
    """七级阶梯指标表 + 边际贡献."""
    rows = []
    prev = None
    for name, pos in levels.items():
        m = perf(o2o_returns(pos, open_price))
        row = {"level": name, "cagr": m["cagr"], "sharpe": m["sharpe"],
               "maxdd": m["mdd"], "calmar": m["calmar"], "y26": m["y26"]}
        if prev is not None:
            row["d_cagr"] = m["cagr"] - prev["cagr"]
            row["d_sharpe"] = m["sharpe"] - prev["sharpe"]
            row["d_maxdd"] = m["mdd"] - prev["mdd"]
        else:
            row["d_cagr"] = np.nan
            row["d_sharpe"] = np.nan
            row["d_maxdd"] = np.nan
        rows.append(row)
        prev = m
    return pd.DataFrame(rows).set_index("level")


# ===== Part 2: 子信号归因 =====

def sub_signal_table(open_price: pd.Series) -> pd.DataFrame:
    """4路子信号独立业绩 (重算参数与 ensemble 内部严格一致)."""
    _, _ = load_au_adj_ohlcv()
    vix, gvz = load_vix_gvz(open_price.index)
    tips = load_indicator_d("tips10y", "fred_dfii10").reindex(open_price.index).ffill().shift(2)

    sub_pos = {}
    sub_pos["macro_regime"] = compute_macro_regime_signal(
        vix=vix, gvz=gvz, tips=tips)["position"]
    sub_pos["real_rate"] = compute_real_rate_signal(
        variant="tips_ma", tips_daily=tips)["position"]
    sub_pos["momentum"] = compute_momentum_signal(variant="long_only")["position"]
    sub_pos["breakout"] = compute_breakout_signal()["position"]

    # 合成参考: 原始加权 (L0)
    L0 = compute_layered_ensemble_signal(use_risk_overlay=False)["position"]

    rows = []
    for name, pos in sub_pos.items():
        p = pos.reindex(open_price.index).ffill().fillna(0)
        m = perf(o2o_returns(p, open_price))
        corr = p.corr(L0.reindex(p.index).ffill().fillna(0))
        weight = BREAKOUT_WEIGHT if name == "breakout" else BASE_WEIGHT
        rows.append({"signal": name, "weight": weight, "cagr": m["cagr"],
                     "sharpe": m["sharpe"], "maxdd": m["mdd"], "y26": m["y26"],
                     "corr_L0": corr})
    return pd.DataFrame(rows).set_index("signal")


# ===== Part 3: 调仓归因日志 =====

def trade_log(close: pd.Series, year: int | None = None) -> pd.DataFrame:
    """逐个调仓日: 分解 Δcore/Δgrid + 触发源分类."""
    pos = build_production_position()
    chain = build_chain()
    core = chain.clip(0, 1.25)
    grid = weekly_grid_inventory(close, chain.index, spacing_pct=0.02, step=0.0833, max_grids=3)
    ret = close.pct_change()

    rows = []
    idx = pos.index
    for i in range(1, len(idx)):
        d = idx[i]
        if abs(pos.iloc[i] - pos.iloc[i - 1]) < 1e-9:
            continue
        if year is not None and d.year != year:
            continue
        c_i = core.reindex(idx).iloc[i]
        c_p = core.reindex(idx).iloc[i - 1]
        g_i = grid.reindex(idx).iloc[i]
        g_p = grid.reindex(idx).iloc[i - 1]
        delta_core = c_i - c_p
        delta_grid = g_i - g_p
        is_emergency = bool(ret.loc[d] < -0.05) if d in ret.index else False

        if is_emergency:
            trigger = "应急"
        elif abs(delta_core) > 1e-9 and abs(delta_grid) > 1e-9:
            trigger = "链层+网格"
        elif abs(delta_core) > 1e-9:
            trigger = "链层"
        elif abs(delta_grid) > 1e-9:
            trigger = "网格"
        else:
            trigger = "平滑追赶"

        rows.append({
            "date": d, "pos": pos.iloc[i], "delta": pos.iloc[i] - pos.iloc[i - 1],
            "core_delta": delta_core, "grid_delta": delta_grid,
            "trigger": trigger,
            "gold_ret": float(ret.loc[d]) if d in ret.index else np.nan,
        })
    return pd.DataFrame(rows)


# ===== 主流程 =====

def main():
    parser = argparse.ArgumentParser(description="黄金生产策略归因分析 (B3)")
    parser.add_argument("--year", type=int, default=None, help="调仓日志年份过滤")
    args = parser.parse_args()

    close, open_price = load_au_adj_ohlcv()

    # 报告头: 完整链路
    print("=" * 70)
    print("黄金生产策略 (B3全栈) 完整链路与归因")
    print("=" * 70)
    print(f"数据窗口: {close.index[0].date()} ~ {close.index[-1].date()}")
    print("\n--- 完整链路 ---")
    for stage, desc in PIPELINE:
        print(f"  {stage:<8} {desc}")

    # Part 1: 七级消融
    print("\n--- 七级消融阶梯 (边际贡献 = 与上一级的差) ---")
    levels, _ = build_ladder(close)
    ladder = ladder_table(levels, open_price)
    ladder.to_csv(f"{OUTPUT_DIR}/attribution_ladder.csv", float_format="%.6f")
    print(ladder.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.3f}"))
    print(f"\n✓ 阶梯表已输出: {OUTPUT_DIR}/attribution_ladder.csv")
    print(f"  风控全程贡献 (L6-L0): CAGR {ladder.loc['L0 裸信号', 'cagr']:+.2%} → "
          f"{ladder.loc['L6 +应急(生产B3)', 'cagr']:+.2%}, "
          f"MaxDD {ladder.loc['L0 裸信号', 'maxdd']:+.2%} → "
          f"{ladder.loc['L6 +应急(生产B3)', 'maxdd']:+.2%}")

    # Part 2: 子信号
    print("\n--- 子信号归因 (生产价格口径重估) ---")
    subs = sub_signal_table(open_price)
    print(subs.to_string(float_format=lambda x: f"{x:+.3f}" if abs(x) < 10 else f"{x:.3f}"))

    # Part 3: 调仓归因
    tl = trade_log(close, year=args.year)
    print(f"\n--- 调仓归因日志 ({args.year if args.year else '全期'}) ---")
    if len(tl) == 0:
        print("  无调仓记录")
        return
    print(f"  调仓次数: {len(tl)}")
    print(f"  触发源分布: {tl['trigger'].value_counts().to_dict()}")
    print(f"\n  {'日期':<12} {'调仓后':>8} {'Δpos':>9} {'Δcore':>9} {'Δgrid':>9} "
          f"{'触发源':<10} {'当日金价':>8}")
    print("  " + "-" * 78)
    for _, r in tl.iterrows():
        print(f"  {r['date'].date()!s:<12} {r['pos']:>8.4f} {r['delta']:>+9.4f} "
              f"{r['core_delta']:>+9.4f} {r['grid_delta']:>+9.4f} "
              f"{r['trigger']:<10} {r['gold_ret']:>+8.2%}")

    if args.year:
        out = f"{OUTPUT_DIR}/trades_{args.year}.csv"
        tl.to_csv(out, index=False, float_format="%.6f")
        print(f"\n✓ 调仓日志已输出: {out}")


if __name__ == "__main__":
    main()
