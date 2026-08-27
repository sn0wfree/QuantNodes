"""scripts/cmf/test_oos_diagnosis.py — 阶段6: OOS失败诊断.

三部分:
A. 逐年分解基础策略 2010-2019 (年化/Sharpe/MaxDD)
B. 趋势信号失效模式: 熊市仓位占比/止损次数/再入场行为
C. 更长trailing stop扫描 (0.08→0.30) 对OOS的改善
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_breakout_signal, compute_trend_bnh_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    compute_vix_momentum_signal,
)

DB_PATH = Path("data/cmf/fundamental.duckdb")

OOS_START, OOS_END = "2010-01-01", "2019-12-31"
IS_START, IS_END = "2020-01-01", "2025-12-31"


def load_base():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    vix_df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_fred' AND value_col='vix' ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    vix_df["trade_date"] = pd.to_datetime(vix_df["trade_date"])
    close = df.set_index("trade_date")["close"]
    vix = vix_df.set_index("trade_date")["value"].reindex(close.index).ffill()
    return close, vix


def get_raw_signal(close, vix, trailing_stop=0.08):
    """基础策略信号 (trailing_stop可调)."""
    vix_pct = vix.rolling(252).rank(pct=True)
    trend = compute_trend_bnh_signal(
        trailing_stop=trailing_stop, freq="daily"
    )["position"].reindex(close.index).fillna(0)
    vix_mom = compute_vix_momentum_signal(
        close, vix, vix_high_threshold=0.80, vix_accel_threshold=0.08
    ).reindex(close.index).fillna(1.0)
    bo = compute_breakout_signal()["position"].reindex(close.index).fillna(0)
    base = trend * vix_mom
    vix_gate = (vix_pct > 0.7).astype(float)
    final_base = base + 0.3 * bo * vix_gate
    return final_base.clip(0, 1.2), trend


def yearly_metrics(pos, close):
    """逐年计算策略指标."""
    rows = []
    sr = pos.shift(1) * close.pct_change()
    cost = pos.diff().abs() * 5 / 10000
    net = (sr - cost).dropna()
    for year in sorted(net.index.year.unique()):
        yr_ret = net[net.index.year == year]
        c_yr = close[close.index.year == year]
        nav = (1 + yr_ret).cumprod()
        dd = (nav / nav.cummax() - 1).min() if len(nav) > 1 else 0
        ann = (1 + yr_ret.mean()) ** 252 - 1 if len(yr_ret) > 0 else 0
        sharpe = yr_ret.mean() / yr_ret.std() * np.sqrt(252) if yr_ret.std() > 0 else 0
        rows.append({
            "year": year,
            "strat": ann,
            "sharpe": sharpe,
            "maxdd": dd,
            "gold": c_yr.iloc[-1] / c_yr.iloc[0] - 1 if len(c_yr) > 1 else 0,
            "expo": pos[pos.index.year == year].mean(),
        })
    return pd.DataFrame(rows)


def part_a_yearly(close, vix):
    print("=" * 100)
    print("Part A: 基础策略逐年分解 (2010-2025)")
    print("=" * 100)
    raw, _ = get_raw_signal(close, vix)
    df = yearly_metrics(raw, close)
    print(f'{"年份":<6} {"策略年化":>9} {"Sharpe":>7} {"MaxDD":>8} '
          f'{"金价涨幅":>9} {"平均仓位":>8}')
    print("-" * 100)
    for _, r in df.iterrows():
        flag = " ←" if r["sharpe"] < 0 else ""
        print(f'{int(r["year"]):<6} {r["strat"]*100:>8.2f}% {r["sharpe"]:>7.2f} '
              f'{r["maxdd"]*100:>7.2f}% {r["gold"]*100:>8.2f}% '
              f'{r["expo"]:>8.2f}{flag}')


def part_b_failure_mode(close, vix):
    print("\n" + "=" * 100)
    print("Part B: 趋势信号失效模式 (trailing_stop=0.08)")
    print("=" * 100)
    _, trend = get_raw_signal(close, vix)

    # 止损触发日: position由1→0
    stopouts = ((trend == 0) & (trend.shift(1) == 1)).astype(int)
    # 平仓持续天数统计
    zero_runs = []
    run_len = 0
    for v in trend["2008-01-01":"2019-12-31"]:
        if v == 0:
            run_len += 1
        elif run_len > 0:
            zero_runs.append(run_len)
            run_len = 0
    if run_len > 0:
        zero_runs.append(run_len)

    print(f"\n止损平仓持续天数分布 (2008-2019):")
    zr = pd.Series(zero_runs)
    if len(zr) > 0:
        print(f"  次数: {len(zr)}, 均值={zr.mean():.1f}天, "
              f"中位数={zr.median():.0f}天, 最大={zr.max()}天")
        print(f"  仅1天即恢复仓位占比: {(zr==1).sum()}/{len(zr)} = {(zr==1).mean()*100:.0f}%")

    print(f'\n{"年份":<6} {"金价涨幅":>9} {"趋势仓位均值":>10} {"空仓天数占比":>10} {"止损次数":>8}')
    print("-" * 100)
    for year in range(2010, 2020):
        t_yr = trend[trend.index.year == year]
        c_yr = close[close.index.year == year]
        s_yr = stopouts[stopouts.index.year == year]
        gold_ret = c_yr.iloc[-1] / c_yr.iloc[0] - 1
        zero_pct = (t_yr == 0).mean()
        print(f'{year:<6} {gold_ret*100:>8.2f}% {t_yr.mean():>10.2f} '
              f'{zero_pct*100:>9.1f}% {int(s_yr.sum()):>8d}')


def part_c_trailing_sweep(close, vix):
    print("\n" + "=" * 130)
    print("Part C: trailing_stop 扫描 (基础策略整体表现)")
    print("=" * 130)
    print(f'{"ts":<6} | {"OOS年化":>8} {"OOSSharpe":>9} {"OOSMaxDD":>9} {"OOSCalmar":>9} | '
          f'{"IS年化":>8} {"ISSharpe":>9} {"ISMaxDD":>9} {"ISCalmar":>9}')
    print("-" * 130)

    for ts in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        raw, _ = get_raw_signal(close, vix, trailing_stop=ts)
        oos = calc_metrics(raw, close, OOS_START, OOS_END)
        is_ = calc_metrics(raw, close, IS_START, IS_END)
        print(f'{ts:<6.2f} | {oos["ann_ret"]*100:>7.2f}% {oos["sharpe"]:>8.2f} '
              f'{oos["max_dd"]*100:>8.2f}% {oos["calmar"]:>8.2f} | '
              f'{is_["ann_ret"]*100:>7.2f}% {is_["sharpe"]:>8.2f} '
              f'{is_["max_dd"]*100:>8.2f}% {is_["calmar"]:>8.2f}')

    # 最佳ts再叠加趋势组件单独看
    print("\n--- 趋势组件单独表现 (不含breakout/vix_mom) ---")
    print(f'{"ts":<6} | {"OOS年化":>8} {"OOSSharpe":>9} {"OOSMaxDD":>9} | '
          f'{"IS年化":>8} {"ISSharpe":>9} {"ISMaxDD":>9}')
    print("-" * 110)
    for ts in [0.08, 0.15, 0.20, 0.30]:
        trend = compute_trend_bnh_signal(
            trailing_stop=ts, freq="daily"
        )["position"].reindex(close.index).fillna(0)
        oos = calc_metrics(trend, close, OOS_START, OOS_END)
        is_ = calc_metrics(trend, close, IS_START, IS_END)
        print(f'{ts:<6.2f} | {oos["ann_ret"]*100:>7.2f}% {oos["sharpe"]:>8.2f} '
              f'{oos["max_dd"]*100:>8.2f}% | {is_["ann_ret"]*100:>7.2f}% '
              f'{is_["sharpe"]:>8.2f} {is_["max_dd"]*100:>8.2f}%')


if __name__ == "__main__":
    close, vix = load_base()
    part_a_yearly(close, vix)
    part_b_failure_mode(close, vix)
    part_c_trailing_sweep(close, vix)
