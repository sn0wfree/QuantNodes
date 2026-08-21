#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/10_copper_evaluation.py — 铜策略全景评估与迭代规划依据.

统一窗口(2018+, term_structure数据起点)横向对比全部铜策略:
    1. 单策略指标 (年化/Sharpe/MaxDD/胜率/换手)
    2. 分年收益 + 分市场状态暴露 (牛市/熊市/震荡)
    3. 策略间相关性矩阵
    4. 等权组合表现
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
    build_signal_carry, backtest_carry,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v5 import (
    build_signal_v5, backtest_v5,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v6 import (
    build_signal_v6, backtest_v6, apply_bull_enhance,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_model_v4 import (
    build_signal_v4, backtest_v4,
)
from QuantNodes.strategy.commodity_fundamental.copper.copper_ml import (
    build_signal_ml, backtest_ml,
)
from QuantNodes.strategy.commodity_fundamental.pipeline_v3 import build_signal_v3

DB = REPO / "data" / "cmf" / "fundamental.duckdb"

START = "2018-01-01"  # term_structure 起点, 统一可比窗口


def yearly(nav: pd.Series) -> dict:
    out = {}
    for y, g in nav.groupby(nav.index.year):
        if len(g) > 4:
            out[y] = (g.iloc[-1] / g.iloc[0] - 1) * 100
    return out


def metrics(nav: pd.Series, ret: pd.Series, pos: pd.Series) -> dict:
    n_years = len(ret) / 52
    total = nav.iloc[-1] / nav.iloc[0] - 1
    ann = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
    vol = ret.std() * np.sqrt(52)
    dd = (nav / nav.cummax() - 1).min()
    turnover = pos.diff().abs().sum() / n_years  # 年均仓位变化
    # 有仓位时胜率
    has_pos = pos.shift(1).reindex(ret.index).fillna(0) > 0.01
    win_active = (ret[has_pos] > 0).mean() * 100 if has_pos.sum() > 0 else 0
    pct_active = has_pos.mean() * 100
    return {
        "ann": ann * 100, "sharpe": ann / vol if vol > 0 else 0,
        "maxdd": dd * 100, "win": win_active, "pct_active": pct_active,
        "avg_pos": pos.mean() * 100, "turnover": turnover,
    }


def main():
    con = duckdb.connect(str(DB), read_only=True)

    print("=" * 90)
    print("§1 构建全部策略信号 (统一窗口 %s+)" % START)
    print("=" * 90)

    strategies = {}

    # v5
    sig = build_signal_v5("CU0", con)
    strategies["v5"] = (sig["M"], sig["position"])

    # v6 (当前主模型: avg(v4, v5精简))
    sig = build_signal_v6("CU0", con)
    strategies["v6"] = (sig["M"], sig["position"])

    # v6.1 (v6 + 牛市增强层: thr=2多数票确认, 进攻型)
    sig_e = apply_bull_enhance(sig, con, threshold=2)
    strategies["v6.1"] = (sig_e["M"], sig_e["position_enhanced"])
    # v6.2 (v6.1 + 波动率目标化12%, 无杠杆)
    strategies["v6.2"] = (sig_e["M"], sig_e["position_sized"])
    # v6.2-lev (v6.1 + 波动率目标化12%, 50%杠杆上限)
    strategies["v6.2-lev"] = (sig_e["M"], sig_e["position_sized_lev"])

    # carry (state_mom long-only)
    sig = build_signal_carry(con)
    strategies["carry"] = (sig["M"], sig["position"])

    # v4
    try:
        sig = build_signal_v4("CU0", con)
        strategies["v4"] = (sig["M"], sig["position"])
    except Exception as e:
        print(f"  v4 构建失败: {e}")

    # ML
    try:
        sig = build_signal_ml("CU0", con)
        strategies["ml"] = (sig["M"], sig["position"])
    except Exception as e:
        print(f"  ml 构建失败: {e}")

    # pricing模块无build_signal入口, 已跳过 (只有backtest)

    # pipeline_v3
    try:
        sig = build_signal_v3("CU0", con)
        strategies["pipeline_v3"] = (sig["M"], sig["position"])
    except Exception as e:
        print(f"  pipeline_v3 构建失败: {e}")

    # 统一起始时间
    for k in list(strategies):
        m, p = strategies[k]
        mask = m.index >= START
        strategies[k] = (m[mask], p[mask])

    names = list(strategies)
    print(f"  可用策略: {names}")

    # 真实周频收盘价 (所有策略统一口径: pipeline_v3的M是卡尔曼平滑价, 不可用于收益计算)
    close_real = con.execute(
        "SELECT trade_date, close FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    close_real["trade_date"] = pd.to_datetime(close_real["trade_date"])
    close_w = close_real.set_index("trade_date")["close"].resample("W-FRI").last()

    # 基准: 买入持有 (对齐v5索引)
    ref_m = strategies["v5"][0]
    strategies["buyhold"] = (ref_m, pd.Series(1.0, index=ref_m.index))

    print()
    print("=" * 90)
    print("§2 单策略指标 (%s+, 周频, 成本5bps)" % START)
    print("=" * 90)
    print(f"{'策略':<13}{'年化%':<9}{'Sharpe':<9}{'MaxDD%':<9}{'胜率%':<8}{'均仓%':<8}{'年换手':<8}")
    print("-" * 64)

    rets = {}
    for name, (m, p) in strategies.items():
        ret_real = close_w.reindex(m.index).pct_change()
        cost = p.diff().abs() * 5.0 / 10000
        r = (p.shift(1) * ret_real - cost).dropna()
        nav = (1 + r).cumprod()
        met = metrics(nav, r, p)
        rets[name] = r
        print(f"{name:<13}{met['ann']:<9.1f}{met['sharpe']:<9.2f}{met['maxdd']:<9.1f}"
              f"{met['win']:<8.1f}{met['avg_pos']:<8.0f}{met['turnover']:<8.1f}")

    print()
    print("=" * 90)
    print("§3 分年收益%")
    print("=" * 90)
    yr_table = {}
    for name, (m, p) in strategies.items():
        ret_real = close_w.reindex(m.index).pct_change()
        cost = p.diff().abs() * 5.0 / 10000
        r = (p.shift(1) * ret_real - cost).dropna()
        yr_table[name] = yearly((1 + r).cumprod())
    years = sorted({y for v in yr_table.values() for y in v})
    hdr = f"{'策略':<13}" + "".join(f"{y:<8}" for y in years if y <= 2025)
    print(hdr)
    for name, yv in yr_table.items():
        cells = (f"{yv.get(y, float('nan')):<8.1f}" for y in years if y <= 2025)
        row = f"{name:<13}" + "".join(cells)
        print(row)

    # 市场状态暴露 (以B&H定义: 26周动量)
    print()
    print("=" * 90)
    print("§4 市场状态平均仓位% (牛市=26周动量>10%, 熊市<-10%, 其余震荡)")
    print("=" * 90)
    ref_m = strategies["v5"][0]
    mom26 = ref_m.pct_change(26)
    bull = mom26 > 0.10
    bear = mom26 < -0.10
    mid = ~bull & ~bear
    print(f"{'策略':<13}{'牛市%':<10}{'震荡%':<10}{'熊市%':<10}")
    for name, (m, p) in strategies.items():
        if name == "buyhold":
            continue
        p = p.reindex(ref_m.index)
        print(f"{name:<13}{p[bull].mean()*100:<10.0f}{p[mid].mean()*100:<10.0f}{p[bear].mean()*100:<10.0f}")
    print(f"{'状态周数':<13}{bull.sum():<10}{mid.sum():<10}{bear.sum():<10}")

    print()
    print("=" * 90)
    print("§5 策略相关性矩阵 (周收益)")
    print("=" * 90)
    df = pd.DataFrame(rets)
    common = df.dropna().index
    print(df.loc[common].corr().round(2).to_string())

    print()
    print("=" * 90)
    print("§6 等权组合 (v5+carry 及 扩展)")
    print("=" * 90)
    combos = {
        "v6+carry": ["v6", "carry"],
    }
    live = [n for n in ["v6", "carry", "v4", "v5", "pipeline_v3"] if n in rets]
    if len(live) > 2:
        combos["all_live"] = live

    dfc = df.loc[common]
    for cname, members in combos.items():
        cr = dfc[members].mean(axis=1)
        nav = (1 + cr).cumprod()
        met = metrics(nav, cr, pd.Series(1.0, index=cr.index))
        print(f"  {cname:<12} 年化={met['ann']:.1f}%  Sharpe={met['sharpe']:.2f}  "
              f"MaxDD={met['maxdd']:.1f}%")

    con.close()


if __name__ == "__main__":
    main()
